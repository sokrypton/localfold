/**
 * Fold a protein with AF3 on the GPU, end to end, and write a PDB.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/fold.js --dump=/af3-6mrr.json --steps=50
 *
 *     embedder -> template -> 4 x MSA block -> 48 x pairformer -> distogram
 *     -> 200-step diffusion sampler around the GPU denoiser
 *     -> confidence head
 *
 * --sequence=<SEQ> folds from a sequence alone, through src/af3/featurise.js;
 * --dump=<path> folds AF3's own batch and reports the disagreement at every
 * point where the two can be compared. Both paths build target_feat from
 * chemistry rather than reading it, so every fold checks that path.
 *
 * 🔴 THE PER-ATOM CONDITIONING IS COMPUTED ON THE CPU HERE, and it is a learned
 * operation, so this is a real gap against AGENTS.md rather than a convenience.
 * It is five embeddings summed over 68*24 rows - small, and it has no GPU kernel
 * yet. Flagged rather than hidden.
 *
 * 🔴 ONE PASS AND ONE MSA ROW, matching the dump: num_recycles=0 and num_msa=1.
 * A comparison against the dump's own outputs is only meaningful with the same
 * settings, and AF3's defaults are neither.
 */
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { atomCrossAttentionEncoder, targetFeatures }
  from "../../src/af3/atom-encoder-reference.js";
import { featuriseProtein } from "../../src/af3/featurise.js";
import { Af3TrunkGpu } from "../../src/af3/trunk-webgpu.js";
import { Af3ConfidenceHeadGpu } from "../../src/af3/confidence-webgpu.js";
import { sampleOnGpu } from "../../src/af3/diffusion-sampler-webgpu.js";
import { confidenceWeights, openAf3Store, trunkWeights } from "./af3-weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "./af3-diffusion-weights.js";

const DIALECT = { swapTransposedBias: false };

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
 * A PDB from the dense atom layout. The atom names come from the batch's
 * ref_atom_name_chars, which is where AF3 keeps them - four ASCII codes offset
 * by 32.
 */
const THREE_LETTER = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY",
  H: "HIS", I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER",
  T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};

function toPdb(positions, atomMask, nameChars, elements, residueIndex, tokens, dense, plddt,
               sequence) {
  const lines = [];
  let serial = 1;
  const elementSymbol = (z) => ({ 6: "C", 7: "N", 8: "O", 16: "S" })[z] ?? "C";
  for (let token = 0; token < tokens; token += 1) {
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!atomMask[slot]) continue;
      let name = "";
      for (let character = 0; character < 4; character += 1) {
        const code = nameChars[slot * 4 + character];
        if (code > 0) name += String.fromCharCode(code + 32);
      }
      name = name.trim();
      const x = positions[slot * 3];
      const y = positions[slot * 3 + 1];
      const z = positions[slot * 3 + 2];
      const confidence = plddt ? plddt[slot] : 0;
      lines.push(
        "ATOM  "
        + String(serial).padStart(5) + " "
        + (name.length < 4 ? ` ${name}`.padEnd(4) : name.slice(0, 4)) + " "
        + (THREE_LETTER[sequence[token]] ?? "UNK").padEnd(3) + " A"
        // 🔴 residue_index IS ALREADY 1-BASED. Adding one here shifted the whole
        // chain by a residue, which against a helical protein reads as a 3.7 A
        // RMSD and a TM-score of 0.37 - a plausible "wrong fold" rather than an
        // obvious bug. The real number was 0.69 A.
        + String(residueIndex[token]).padStart(4) + "    "
        + x.toFixed(3).padStart(8) + y.toFixed(3).padStart(8) + z.toFixed(3).padStart(8)
        + "  1.00" + confidence.toFixed(2).padStart(6) + "          "
        + elementSymbol(elements[slot]).padStart(2));
      serial += 1;
    }
  }
  lines.push("END");
  return lines.join("\n");
}

/**
 * AF3's own batch, in the shape src/af3/featurise.js produces, so the fold below
 * reads one object either way and the two paths cannot silently diverge in what
 * they supply.
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

  // --sequence folds what you type; --dump folds AF3's own batch, which is what
  // makes a comparison against AF3's own trunk meaningful.
  const sequenceArg = option(args, "sequence", "");
  const dump = sequenceArg === "" || args.some((a) => a.startsWith("--dump="))
    ? await (async () => {
        const response = await fetch(dumpPath);
        if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
        return response.json();
      })()
    : null;

  const batch = sequenceArg !== ""
    ? featuriseProtein(sequenceArg)
    : batchFromDump(dump);
  const { tokens, dense, subsets } = batch;
  console.log(`${batch.sequence.length} residues, ${tokens} tokens,`
    + ` ${batch.atomCount} atoms, ${subsets} atom subsets,`
    + ` ${blocks} pairformer blocks, ${steps} diffusion steps`);
  console.log(sequenceArg !== ""
    ? "featurised in JavaScript from the sequence"
    : "featurised by AF3, read from the dump");

  // --quant=int5:g32:asym[:search] round-trips every learned weight through a
  // storage precision before the fold, so the cost is measured in ANGSTROMS
  // rather than in weight error.
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
  // --model points at an exported directory; the int5 one is packed on disk
  // rather than round-tripped at load, so it is the real thing.
  const model = option(args, "model", "/model-af3-full-f32/manifest.json");
  const store = await openAf3Store(model, quant);
  const weights = await trunkWeights(store, blocks, 4);
  const diffusion = await diffusionWeights(store);
  const confidence = await confidenceWeights(store);
  const reference = await atomReference(store);

  const targetFeatWeights = await targetFeatureWeights(store);

  // 🔴 target_feat IS BUILT FROM CHEMISTRY ON EVERY FOLD, never read from the
  // dump - so the path a browser depends on is the path this exercises. It is
  // the bottom of the model reached from the top: reference conformers in, the
  // 447 columns the whole trunk reads out.
  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, targetFeatWeights.reference);

  const atomFeatures = atomCrossAttentionEncoder({
    shape: batch.shape, conditioning, atomMask: batch.refMask,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
  }, targetFeatWeights.encoder);
  const targetFeat = targetFeatures({
    aatype: batch.aatype, profile: batch.profile, deletionMean: batch.deletionMean,
    atomFeatures: atomFeatures.tokenAct,
  }, tokens);

  const theirTargetFeat = dump?.outputs["diffuser/evoformer/__call__:target_feat"];
  if (theirTargetFeat) {
    console.log(`target_feat vs AF3  relRMS`
      + ` ${relativeRms(targetFeat, floats(theirTargetFeat.data)).toExponential(2)}`);
  }

  const seqMask = batch.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  // 🔴 ONE MSA ROW. num_msa=1, so the evoformer sees only the query - which for
  // a de novo design is all there is anyway.
  const sequences = 1;
  const trunkInput = {
    tokens, sequences, templates: 4, targetFeat, features: batch.features,
    msaRows: batch.msa.subarray(0, tokens),
    deletionMatrix: batch.deletionMatrix.subarray(0, tokens),
    msaMask: batch.msaMask.subarray(0, tokens),
    pairMask, seqMask,
    previousPair: new Float32Array(tokens * tokens * 128),
    previousSingle: new Float32Array(tokens * 384),
  };

  const started = performance.now();
  const trunk = await new Af3TrunkGpu(device).run(trunkInput, weights, DIALECT, {
    onStage: (name, ms) => console.log(`  ${name.padEnd(12)} ${ms.toFixed(0)} ms`),
  });
  console.log(`trunk done in ${((performance.now() - started) / 1000).toFixed(1)} s`);

  // Against AF3's own trunk. Only meaningful on AF3's own batch: from a
  // sequence the reference conformers differ, which is worth about 2.7e-2 on
  // pair and 0.01 A of structure.
  const reference2d = dump?.outputs["diffuser/evoformer/__call__:pair"];
  const reference1d = dump?.outputs["diffuser/evoformer/__call__:single"];
  if (reference2d && blocks === 48) {
    console.log(`pair   vs AF3  relRMS ${relativeRms(trunk.pair, floats(reference2d.data)).toExponential(2)}`);
    console.log(`single vs AF3  relRMS ${relativeRms(trunk.single, floats(reference1d.data)).toExponential(2)}`);
  }

  // 🔴 THE DIFFUSION HEAD HAS ITS OWN REFERENCE EMBEDDINGS, and they are not
  // the conditioning module's - same five shapes, different weights. The
  // conditioning above cannot be reused here.
  const diffusionConditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, reference);

  const atomMask = batch.predDenseAtomMask;
  const headInput = {
    shape: batch.shape,
    conditioning: diffusionConditioning, atomMask, seqMask,
    features: batch.features, targetFeat,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    trunkSingle: trunk.single, trunkPair: trunk.pair,
  };

  let state = Number(option(args, "seed", "20260831")) >>> 0;
  const uniform = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  const normal = () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());

  const trajectory = [];
  const diffusionStarted = performance.now();
  const positions = await sampleOnGpu(device, headInput, diffusion, {
    steps, normal,
    onStep: ({ step, noiseLevel, denoised, positions: walk }) => {
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
                          positions: Array.from(walk) });
      }
      if (step === 1 || step % Math.ceil(steps / 5) === 0 || step === steps) {
        console.log(`  step ${String(step).padStart(3)}/${steps}  sigma`
          + ` ${noiseLevel.toFixed(2).padStart(9)}`);
      }
    },
  });
  console.log(`diffusion done in ${((performance.now() - diffusionStarted) / 1000).toFixed(1)} s`);

  // The confidence head reads the sample back.
  const betaGather = batch.tokenAtomsToPseudoBeta;
  const pseudoBeta = new Float32Array(tokens * 3);
  for (let token = 0; token < tokens; token += 1) {
    if (!betaGather.mask[token]) continue;
    const from = Number(betaGather.indices[token]) * 3;
    for (let axis = 0; axis < 3; axis += 1) pseudoBeta[token * 3 + axis] = positions[from + axis];
  }
  const scores = await new Af3ConfidenceHeadGpu(device).run({
    tokens, dense, seqMask, pair: trunk.pair, single: trunk.single,
    targetFeat, pseudoBeta,
  }, confidence, DIALECT);

  let plddtTotal = 0;
  let plddtCount = 0;
  for (let index = 0; index < tokens * dense; index += 1) {
    if (!atomMask[index]) continue;
    plddtTotal += scores.plddt[index];
    plddtCount += 1;
  }
  const meanPlddt = plddtTotal / plddtCount;
  console.log(`mean pLDDT ${meanPlddt.toFixed(1)} over ${plddtCount} atoms`);

  const pdb = toPdb(positions, atomMask, batch.refAtomNameChars,
                    batch.refElement, batch.features.residueIndex, tokens, dense,
                    scores.plddt, batch.sequence);

  // 🔴 GEOMETRY IS THE CHECK THAT MATTERS HERE, not pLDDT. A confident-looking
  // pLDDT comes off the trunk and says nothing about whether the diffusion
  // produced a molecule: the numbers below - backbone N-CA, CA-C, and the
  // radius of gyration - are what a wrong sampler cannot fake.
  const named = [];
  for (let token = 0; token < tokens; token += 1) {
    const atoms = {};
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!atomMask[slot]) continue;
      let name = "";
      for (let character = 0; character < 4; character += 1) {
        const code = batch.refAtomNameChars[slot * 4 + character];
        if (code > 0) name += String.fromCharCode(code + 32);
      }
      atoms[name.trim()] = slot;
    }
    named.push(atoms);
  }
  const distance = (a, b) => Math.hypot(positions[a * 3] - positions[b * 3],
                                        positions[a * 3 + 1] - positions[b * 3 + 1],
                                        positions[a * 3 + 2] - positions[b * 3 + 2]);
  const stats = (values) => {
    const sorted = [...values].sort((x, y) => x - y);
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    return { mean, median: sorted[Math.floor(sorted.length / 2)], n: values.length };
  };
  const nca = [];
  const cac = [];
  const caca = [];
  for (let token = 0; token < tokens; token += 1) {
    if (named[token].N !== undefined && named[token].CA !== undefined) {
      nca.push(distance(named[token].N, named[token].CA));
    }
    if (named[token].CA !== undefined && named[token].C !== undefined) {
      cac.push(distance(named[token].CA, named[token].C));
    }
    if (token + 1 < tokens && named[token].CA !== undefined
        && named[token + 1].CA !== undefined) {
      caca.push(distance(named[token].CA, named[token + 1].CA));
    }
  }
  const geometry = { nca: stats(nca), cac: stats(cac), caca: stats(caca) };
  console.log(`backbone  N-CA ${geometry.nca.median.toFixed(2)} A`
    + ` (ideal 1.46)   CA-C ${geometry.cac.median.toFixed(2)} A (ideal 1.52)`
    + `   CA-CA ${geometry.caca.median.toFixed(2)} A (ideal 3.80)`);

  const centre = [0, 0, 0];
  let count = 0;
  for (let token = 0; token < tokens; token += 1) {
    const ca = named[token].CA;
    if (ca === undefined) continue;
    for (let axis = 0; axis < 3; axis += 1) centre[axis] += positions[ca * 3 + axis];
    count += 1;
  }
  for (let axis = 0; axis < 3; axis += 1) centre[axis] /= count;
  let gyration = 0;
  for (let token = 0; token < tokens; token += 1) {
    const ca = named[token].CA;
    if (ca === undefined) continue;
    for (let axis = 0; axis < 3; axis += 1) {
      gyration += (positions[ca * 3 + axis] - centre[axis]) ** 2;
    }
  }
  gyration = Math.sqrt(gyration / count);
  console.log(`radius of gyration ${gyration.toFixed(1)} A over ${count} CA`
    + `   (a compact 68-mer is about 11-12 A)`);
  const elapsed = (performance.now() - started) / 1000;
  console.log(`total ${elapsed.toFixed(1)} s`);
  return {
    sequence: batch.sequence, tokens, steps, meanPlddt, geometry, gyration,
    seconds: elapsed, pdb, trajectory,
  };
}
