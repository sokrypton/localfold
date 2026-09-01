/**
 * Fold a protein with AF3 on the GPU, end to end, and write a PDB.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold.js --sequence=GWSTELEK... \
 *       --steps=200 --model=/model-af3-int5/manifest.json
 *     node tools/gpu-chrome.mjs tools/gpu/fold.js --dump=/af3-6mrr.json
 *
 * --sequence folds what you type, through src/af3/featurise.js. --dump folds
 * AF3's own batch and reports the disagreement at every point where the two can
 * be compared, which is the only way the trunk can be checked against AF3's.
 *
 * WHAT IS HERE AND NOT IN src/af3/fold.js: argument parsing, the comparison
 * against the dump, and the geometry report. The pipeline itself is shared with
 * the page, because the page has to run what was measured.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { foldBatch, toPdb, backboneGeometry } from "../../src/af3/fold.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const floats = (source) => Float32Array.from(source, (v) => Number(v));
const ints = (source) => Int32Array.from(source, (v) => Number(v));

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

/**
 * AF3's own batch, in the shape src/af3/featurise.js produces, so the fold reads
 * one object either way and the two paths cannot silently diverge in what they
 * supply.
 */
function batchFromDump(dump) {
  const tokens = dump.tokens;
  const dense = 24;
  const subsets = Math.ceil((tokens * dense) / 32);
  const raw = (name) => dump.inputs[name].data;
  // 🔴 count IS NOT DECORATION. convert() in atom-encoder-reference.js sizes its
  // output from it, so a gather without one silently produces a zero-length
  // tensor - which reads downstream as a model that runs and folds a 17 A
  // spaghetti rather than as an error.
  const gather = (name) => {
    const indices = ints(raw(`${name}:gather_idxs`));
    return { indices, mask: floats(raw(`${name}:gather_mask`)), count: indices.length };
  };
  const refMask = floats(raw("ref_mask"));
  let atomCount = 0;
  for (const value of refMask) atomCount += value;
  return {
    sequence: dump.sequence, tokens, dense, subsets, atomCount,
    shape: { tokens, dense, subsets, queries: 32, keys: 128 },
    aatype: ints(raw("aatype")), profile: floats(raw("profile")),
    deletionMean: floats(raw("deletion_mean")),
    msa: ints(raw("msa")), msaMask: floats(raw("msa_mask")),
    deletionMatrix: floats(raw("deletion_matrix")),
    seqMask: floats(raw("seq_mask")),
    refPos: floats(raw("ref_pos")), refMask,
    refElement: ints(raw("ref_element")), refCharge: floats(raw("ref_charge")),
    refAtomNameChars: ints(raw("ref_atom_name_chars")),
    refSpaceUid: ints(raw("ref_space_uid")),
    predDenseAtomMask: floats(raw("pred_dense_atom_mask")),
    tokenAtomsToQueries: gather("token_atoms_to_queries"),
    queriesToKeys: gather("queries_to_keys"),
    queriesToTokenAtoms: gather("queries_to_token_atoms"),
    tokensToQueries: gather("tokens_to_queries"),
    tokensToKeys: gather("tokens_to_keys"),
    tokenAtomsToPseudoBeta: gather("token_atoms_to_pseudo_beta"),
    features: {
      residueIndex: ints(raw("residue_index")), tokenIndex: ints(raw("token_index")),
      asymId: ints(raw("asym_id")), entityId: ints(raw("entity_id")),
      symId: ints(raw("sym_id")),
    },
  };
}

export async function main(device, args) {
  const dumpPath = option(args, "dump", "/af3-6mrr.json");
  const steps = Number(option(args, "steps", "50"));
  const blocks = Number(option(args, "blocks", "48"));
  const sequenceArg = option(args, "sequence", "");

  const dump = sequenceArg === "" || args.some((a) => a.startsWith("--dump="))
    ? await (async () => {
        const response = await fetch(dumpPath);
        if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
        return response.json();
      })()
    : null;

  const batch = sequenceArg !== "" ? featuriseProtein(sequenceArg) : batchFromDump(dump);
  console.log(`${batch.sequence.length} residues, ${batch.tokens} tokens,`
    + ` ${batch.atomCount} atoms, ${batch.subsets} atom subsets,`
    + ` ${blocks} pairformer blocks, ${steps} diffusion steps`);
  console.log(sequenceArg !== ""
    ? "featurised in JavaScript from the sequence"
    : "featurised by AF3, read from the dump");

  // --quant=int5:g32:asym[:search] round-trips every learned weight through a
  // storage precision before the fold, so the cost is measured in ANGSTROMS
  // rather than in weight error. --model points at an exported directory
  // instead; the int5 one is packed on disk rather than round-tripped at load.
  const quantSpec = option(args, "quant", "");
  const quant = quantSpec === "" ? null : (() => {
    const [bitsField, groupField, mode, search] = quantSpec.split(":");
    return { bits: Number(bitsField.replace("int", "")),
             group: Number(groupField.replace("g", "")),
             mode: mode ?? "asym", search: search === "search" };
  })();
  if (quant) {
    console.log(`quantised: int${quant.bits} group ${quant.group} ${quant.mode}`
      + `${quant.search ? " with range search" : ""}`
      + `   ${(quant.bits + (quant.mode === "sym" ? 16 : 32) / quant.group).toFixed(2)}`
      + ` bits/weight`);
  }

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"),
                                   quant);
  const weights = {
    trunk: await trunkWeights(store, blocks, 4),
    diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  const started = performance.now();
  let trunkStarted = 0;
  let diffusionStarted = 0;
  const trajectory = [];
  let lastDenoised = null;

  const result = await foldBatch(device, batch, weights, {
    mode: option(args, "mode", "diffusion"),
    recycles: Number(option(args, "recycles", "0")),
    steps, stopAfter: Number(option(args, "truncate", String(steps))),
    seed: Number(option(args, "seed", "20260831")),
    onStage: (name, detail) => {
      if (name === "target-feat") {
        const theirs = dump?.outputs["diffuser/evoformer/__call__:target_feat"];
        if (theirs) {
          console.log(`target_feat vs AF3  relRMS`
            + ` ${relativeRms(detail.targetFeat, floats(theirs.data)).toExponential(2)}`);
        }
        trunkStarted = performance.now();
      }
      if (name === "trunk") console.log(`  ${detail.name.padEnd(12)} ${detail.ms.toFixed(0)} ms`);
      if (name === "trunk-done") {
        console.log(`trunk done in ${((performance.now() - trunkStarted) / 1000).toFixed(1)} s`);
        // Against AF3's own trunk. Only meaningful on AF3's own batch: from a
        // sequence the reference conformers differ, which is worth about
        // 2.7e-2 on pair and 0.01 A of structure.
        const pair = dump?.outputs["diffuser/evoformer/__call__:pair"];
        const single = dump?.outputs["diffuser/evoformer/__call__:single"];
        if (pair && blocks === 48) {
          console.log(`pair   vs AF3  relRMS`
            + ` ${relativeRms(detail.trunk.pair, floats(pair.data)).toExponential(2)}`);
          console.log(`single vs AF3  relRMS`
            + ` ${relativeRms(detail.trunk.single, floats(single.data)).toExponential(2)}`);
        }
        diffusionStarted = performance.now();
      }
    },
    onStep: ({ step, noiseLevel, denoised, positions }) => {
      lastDenoised = denoised;
      // Every frame for a short run, every fourth for a long one - the whole
      // trajectory at 200 steps is 200 * 68 * 24 * 3 floats.
      if (steps <= 60 || step % 4 === 0 || step === steps) {
        // 🔴 BOTH TRACKS. `denoised` is the model's running guess and it is
        // already within about 1 A of the final structure at step ONE - the
        // trunk decides the fold and diffusion refines it. `positions` is the
        // actual trajectory, which starts as a cloud thousands of angstroms
        // across. Only the second one looks like folding.
        trajectory.push({ step, noiseLevel,
                          denoised: Array.from(denoised),
                          positions: Array.from(positions) });
      }
      if (step === 1 || step % Math.ceil(steps / 5) === 0 || step === steps) {
        console.log(`  step ${String(step).padStart(3)}/${steps}  sigma`
          + ` ${noiseLevel.toFixed(2).padStart(9)}`);
      }
    },
  });

  console.log(`diffusion done in ${((performance.now() - diffusionStarted) / 1000).toFixed(1)} s`);
  console.log(`mean pLDDT ${result.meanPlddt.toFixed(1)} over ${result.atoms} atoms`);

  // 🔴 GEOMETRY IS THE CHECK THAT MATTERS HERE, not pLDDT - see the note on
  // backboneGeometry.
  const { nca, cac, caca, gyration, residues } = result.geometry;
  console.log(`backbone  N-CA ${nca.toFixed(2)} A (ideal 1.46)`
    + `   CA-C ${cac.toFixed(2)} A (ideal 1.52)`
    + `   CA-CA ${caca.toFixed(2)} A (ideal 3.80)`);
  console.log(`radius of gyration ${gyration.toFixed(1)} A over ${residues} CA`
    + `   (a compact 68-mer is about 11-12 A)`);
  const elapsed = (performance.now() - started) / 1000;
  console.log(`total ${elapsed.toFixed(1)} s`);

  // 🔴 THE DENOISED PREDICTION IS NOT THE SAMPLE, and at a coarse schedule they
  // are not close. `positions` is where the sampler's walk ended; `denoised` is
  // what the model predicted on the last call. Reported side by side because
  // the difference between them IS the schedule: with many steps the walk has
  // been pulled onto the prediction and the two agree, and with few it has not.
  const denoisedGeometry = backboneGeometry(batch, lastDenoised);
  console.log(`last denoised  N-CA ${denoisedGeometry.nca.toFixed(2)}`
    + `   CA-C ${denoisedGeometry.cac.toFixed(2)}`
    + `   CA-CA ${denoisedGeometry.caca.toFixed(2)} A`
    + `   gyration ${denoisedGeometry.gyration.toFixed(1)} A`);

  // 🔴 sequence STAYS THE FIRST KEY. tools/score_fold.py finds the result in
  // this log by searching for `{\n  "sequence"`, so reordering the object
  // silently makes every score say "did the run fail?".
  return {
    sequence: batch.sequence, tokens: batch.tokens, steps,
    denoisedPdb: toPdb(batch, lastDenoised, result.scores.plddt),
    meanPlddt: result.meanPlddt,
    geometry: { nca: { median: nca }, cac: { median: cac }, caca: { median: caca } },
    gyration, seconds: elapsed, pdb: result.pdb, trajectory,
  };
}
