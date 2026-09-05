/**
 * Fold a protein with AF3 on the GPU, end to end, and write a PDB.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold.js --sequence=GWSTELEK... \
 *       --steps=200 --model=/model-af3-int5/manifest.json
 *     node tools/gpu-chrome.mjs tools/gpu/fold.js --dump=/oracle-dumps/af3-6mrr.json
 *
 * --sequence folds what you type, through src/af3/featurise.js. --dump folds
 * AF3's own batch and reports the disagreement at every point where the two can
 * be compared, which is the only way the trunk can be checked against AF3's.
 *
 * WHAT IS HERE AND NOT IN src/af3/fold.js: argument parsing, the comparison
 * against the dump, and the geometry report. The pipeline itself is shared with
 * the page, because the page has to run what was measured.
 */
import { memorySnapshot, setMemoryBudget } from "../../src/runtime/device-memory.js";import { featuriseProtein } from "../../src/af3/featurise.js";
import { af3MsaFromA3m } from "../../src/af3/msa-features.js";
import { mergeRowAlignedChainA3ms } from "../../src/input/chains.js";
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
export function batchFromDump(dump) {
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
    return {
    indices, mask: floats(raw(`${name}:gather_mask`)), count: indices.length };
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
    // 🔴 THE ROW COUNT IS A SHAPE, AND THE DUMP IS THE ONLY PLACE IT IS WRITTEN
    // DOWN. AF3 pads its msa array to the crop size and records how many rows
    // are real in `numMsa`, so deriving it from the array's length reads the
    // padding as alignment. Absent, the trunk's shader was built with
    // `const SEQUENCES: u32 = undefinedu;` and the fold died in WGSL parsing -
    // which is a shape bug wearing a compiler error.
    sequences: dump.numMsa ?? 1,
    // ...and the chain identity, which the confidence head's ipTM reduction
    // indexes directly. Absent, it threw inside reduceTmScore AFTER the whole
    // fold had run, which is the most expensive place to discover a missing
    // field.
    asymId: ints(raw("asym_id")),
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
  const dumpPath = option(args, "dump", "/oracle-dumps/af3-6mrr.json");
  const steps = Number(option(args, "steps", "50"));
  // Named here rather than inline at the fold, because the guard below reads it.
  const samplerMode = option(args, "mode", "diffusion");
  const blocks = Number(option(args, "blocks", "48"));
  const sequenceArg = option(args, "sequence", "");

  // 🔴 AF3's DIFFUSION SAMPLER NEEDS ITS WHOLE SCHEDULE, AND STOPPING EARLY
  // LOOKS EXACTLY LIKE A BROKEN MODEL. `--steps` sets the discretisation, not a
  // budget: eight steps of the stochastic sampler leaves the walk at high noise
  // and prints an N-CA of 27 A next to an ideal of 1.46, which reads as
  // corrupted weights rather than as the wrong flag. The flow reaches a
  // structure in eight because it is a different walk - see
  // src/af3/diffusion-sampler-webgpu.js - so say so rather than let the
  // geometry report take the blame.
  if (samplerMode === "diffusion" && steps < 50) {
    console.log(`🔴 ${steps} steps of the DIFFUSION sampler will not converge -`
      + " it wants about 200. Add --mode=flow to get a structure in this many,"
      + " or raise --steps. What follows is expected to look like noise.");
  }

  const dump = sequenceArg === "" || args.some((a) => a.startsWith("--dump="))
    ? await (async () => {
        const response = await fetch(dumpPath);
        if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
        return response.json();
      })()
    : null;

  // 🔴 AN MSA, WHICH THE BROWSER PATH HAS AND THIS DID NOT. --a3m takes one
  // path per CHAIN, comma-separated, and --paired-a3m the same for the paired
  // block; both are merged exactly as web/app.js merges them for AF3, so a
  // fold run here is the fold the page runs. Without this the CLI could only
  // ever reproduce a single-sequence prediction, which is not the case worth
  // debugging on a complex.
  const fetchText = async (path) => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`failed to load ${path}: ${response.status}`);
    return response.text();
  };
  const chainTexts = async (spec) => (spec === ""
    ? null
    : Promise.all(spec.split(",").map((path) => fetchText(path.trim()))));
  const unpairedTexts = await chainTexts(option(args, "a3m", ""));
  const pairedTexts = await chainTexts(option(args, "paired-a3m", ""));
  const mergeFor = (texts) => (texts === null ? null
    : (texts.length === 1 ? texts[0] : mergeRowAlignedChainA3ms(texts)));
  const rows = unpairedTexts === null && pairedTexts === null
    ? { msa: [], deletionMatrix: [], depth: 1, unpairedFrom: 0 }
    : af3MsaFromA3m({ paired: mergeFor(pairedTexts), unpaired: mergeFor(unpairedTexts) },
                    { maxSequences: Number(option(args, "max-msa", "512")) });

  const batch = sequenceArg !== ""
    ? featuriseProtein(sequenceArg,
      { msa: rows.msa, deletionMatrix: rows.deletionMatrix, unpairedFrom: rows.unpairedFrom })
    : batchFromDump(dump);
  if (rows.depth > 1) {
    console.log(`MSA ${rows.depth} rows, unpaired block starts at ${rows.unpairedFrom}`);
  }
  if (batch.sequences !== rows.depth) {
    console.log(`🔴 the batch carries ${batch.sequences} MSA rows, not ${rows.depth}`);
  }
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

  // --budget=<MiB> puts a ceiling on the device, which is how a machine too
  // small to keep the weights resident behaves. It is the only way that
  // fallback gets exercised on a Mac.
  const budgetMiB = Number(option(args, "budget", "0"));
  if (budgetMiB > 0) setMemoryBudget(device, budgetMiB * 1024 * 1024);
  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"),
                                   quant);
  const weights = {
    trunk: await trunkWeights(store, blocks, 4),
    diffusion: await diffusionWeights(store),
    confidence: await confidenceWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };

  // 🔴 A REPEAT FOLD IS THE ONE THE PAGE ACTUALLY SHOWS AFTER THE FIRST. The
  // pipelines and the resident f16 weights are cached for the life of the
  // DEVICE, so a second fold in the same session pays neither - and this file
  // measures a cold process, which is the slowest fold there is. `--folds=2`
  // runs it twice and reports both, so the two can be told apart.
  const folds = Number(option(args, "folds", "1"));
  const foldSeconds = [];
  let result;
  let trunkStarted = 0;
  let diffusionStarted = 0;
  const trajectory = [];
  let lastDenoised = null;
  for (let attempt = 0; attempt < folds; attempt += 1) {
  const started = performance.now();
  trajectory.length = 0;
  lastDenoised = null;

  result = await foldBatch(device, batch, weights, {
    // Omitted, each defaults to what the device supports. See docs/AF3.md.
    stagedPrecision: option(args, "staged", undefined),
    weightPrecision: option(args, "weights", undefined),
    pairWeightPrecision: option(args, "pair-weights", undefined),
    accumulatePrecision: option(args, "accumulate", undefined),
    mode: samplerMode,
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
      if (name === "trunk") {
        const gpu = memorySnapshot(device);
        console.log(`  ${detail.name.padEnd(12)} ${detail.ms.toFixed(0)} ms`
          + `   gpu ${(gpu.residentBytes / (1024 * 1024)).toFixed(0)} MiB`
          + ` (peak ${(gpu.peakBytes / (1024 * 1024)).toFixed(0)})`);
      }
      if (name === "trunk-done") {
        const gpu = memorySnapshot(device);
        console.log(`trunk done in ${((performance.now() - trunkStarted) / 1000).toFixed(1)} s`
          + `   gpu ${(gpu.residentBytes / (1024 * 1024)).toFixed(0)} MiB`
          + ` (peak ${(gpu.peakBytes / (1024 * 1024)).toFixed(0)})`);
        // 🔴 WHAT IS STILL HELD WHEN THE TRUNK IS DONE. The peak composition
        // shows the pairformer's scratch and the diffusion transformer's
        // resident weights in the same snapshot, though one finishes before
        // the other starts - so either the scratch is not given back, or it is
        // given back to a POOL that goes on holding it. This says which.
        for (const row of gpu.currentByLabel.slice(0, 6)) {
          console.log(`    held ${(row.bytes / (1024 * 1024)).toFixed(1).padStart(8)} MiB`
            + ` x${String(row.count).padEnd(3)} ${row.label}`);
        }
        console.log(`  peak so far ${(gpu.peakBytes / (1024 * 1024)).toFixed(1)} MiB, made of:`);
        for (const row of gpu.peakByLabel.slice(0, 6)) {
          console.log(`    peak ${(row.bytes / (1024 * 1024)).toFixed(1).padStart(8)} MiB`
            + ` x${String(row.count).padEnd(3)} ${row.label}`);
        }
        // Against AF3's own trunk. Only meaningful on AF3's own batch: from a
        // sequence the reference conformers differ, which is worth about
        // 2.7e-2 on pair and 0.01 A of structure.
        // 🔴 A RECYCLED DUMP NAMES ITS CAPTURES PER PASS. With --recycles the
        // evoformer is called once per pass and each capture gets a `#n`, so
        // the bare name matches nothing and the comparison silently does not
        // run - the report simply omits the two lines it exists to print. The
        // LAST pass is the one whose output reaches the diffusion head.
        const lastCapture = (base) => dump?.outputs[base]
          ?? Object.keys(dump?.outputs ?? {})
            .filter((key) => key.startsWith(`${base}#`))
            .sort((a, b) => Number(a.split("#")[1]) - Number(b.split("#")[1]))
            .map((key) => dump.outputs[key]).pop();
        const pair = lastCapture("diffuser/evoformer/__call__:pair");
        const single = lastCapture("diffuser/evoformer/__call__:single");
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
  const elapsed = (performance.now() - started) / 1000;
  foldSeconds.push(Number(elapsed.toFixed(3)));
  if (folds > 1) console.log(`fold ${attempt + 1} of ${folds}: ${elapsed.toFixed(1)} s`);
  }

  console.log(`diffusion done in ${((performance.now() - diffusionStarted) / 1000).toFixed(1)} s`);
  console.log(`mean pLDDT ${result.meanPlddt.toFixed(1)} over ${result.atoms} atoms`
    + `   pTM ${result.ptm.toFixed(3)}`
    + `   ipTM ${Number.isNaN(result.iptm) ? "n/a (one chain)" : result.iptm.toFixed(3)}`);

  // 🔴 GEOMETRY IS THE CHECK THAT MATTERS HERE, not pLDDT - see the note on
  // backboneGeometry.
  const { nca, cac, caca, gyration, residues } = result.geometry;
  console.log(`backbone  N-CA ${nca.toFixed(2)} A (ideal 1.46)`
    + `   CA-C ${cac.toFixed(2)} A (ideal 1.52)`
    + `   CA-CA ${caca.toFixed(2)} A (ideal 3.80)`);
  console.log(`radius of gyration ${gyration.toFixed(1)} A over ${residues} CA`
    + `   (a compact 68-mer is about 11-12 A)`);
  console.log(`total ${foldSeconds[foldSeconds.length - 1].toFixed(1)} s`);

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
    // Both folds, so a cold process and a warm one can be told apart.
    foldSeconds,
    sequence: batch.sequence, tokens: batch.tokens, steps,
    denoisedPdb: toPdb(batch, lastDenoised, result.scores.plddt),
    meanPlddt: result.meanPlddt,
    ptm: result.ptm,
    iptm: Number.isNaN(result.iptm) ? null : result.iptm,
    geometry: { nca: { median: nca }, cac: { median: cac }, caca: { median: caca } },
    gyration, seconds: foldSeconds[foldSeconds.length - 1], pdb: result.pdb, trajectory,
    // What the device is holding at the end, which nothing else reports.
    memory: memorySnapshot(device),
  };
}
