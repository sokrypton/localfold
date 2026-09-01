/**
 * One AF3 fold, from a featurised batch to coordinates and a pLDDT.
 *
 *     const batch = featuriseProtein("GWSTELEK...");
 *     const result = await foldBatch(device, batch, weights, { steps: 200 });
 *
 *     embedder -> template -> 4 x MSA block -> 48 x pairformer -> distogram
 *     -> N-step diffusion sampler around the GPU denoiser
 *     -> confidence head
 *
 * WHY THIS IS NOT IN tools/gpu/fold.js ANY MORE. The command line and the page
 * have to run the SAME pipeline or the page is not what was measured, and the
 * measurements are all from the command line. What is left there is argument
 * parsing, the comparison against AF3's dump, and the geometry report.
 *
 * 🔴 THE PER-ATOM CONDITIONING IS COMPUTED ON THE CPU, TWICE, and it is a
 * learned operation - a real gap against AGENTS.md rather than a convenience.
 * Once for target_feat's atom columns and once for the diffusion head, from two
 * different weight bundles of the same five shapes. It is 574 x 24 rows and has
 * no GPU kernel yet. Flagged rather than hidden.
 *
 * 🔴 ONE PASS AND ONE MSA ROW: num_recycles=0, num_msa=1. A de novo design has
 * no homologues, so for the sequences this is aimed at the MSA row IS the
 * query; a deeper one would change three arrays in featurise.js and nothing
 * here.
 */
import { perAtomConditioning } from "./atom-conditioning-reference.js";
import { atomCrossAttentionEncoder, targetFeatures } from "./atom-encoder-reference.js";
import { Af3TrunkGpu } from "./trunk-webgpu.js";
import { Af3ConfidenceHeadGpu } from "./confidence-webgpu.js";
import { sampleOnGpu } from "./diffusion-sampler-webgpu.js";

/**
 * 🔴 THE DIALECT IS NOT A PREFERENCE. `model='openfold3'` turns on four
 * branches stock AF3 does not have, and a checkpoint has to be read through the
 * graph it was converted for. This is the stock one.
 */
export const DIALECT = { swapTransposedBias: false };

const THREE_LETTER = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY",
  H: "HIS", I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER",
  T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};

const ELEMENT_SYMBOL = { 6: "C", 7: "N", 8: "O", 16: "S" };

/** The four-character atom name AF3 stores as codes offset by 32. */
export function atomName(nameChars, slot) {
  let name = "";
  for (let character = 0; character < 4; character += 1) {
    const code = nameChars[slot * 4 + character];
    if (code > 0) name += String.fromCharCode(code + 32);
  }
  return name.trim();
}

/**
 * A PDB from the dense atom layout, with pLDDT in the B-factor column so the
 * viewer can colour by it.
 */
export function toPdb(batch, positions, plddt) {
  const { tokens, dense, sequence } = batch;
  const lines = [];
  let serial = 1;
  for (let token = 0; token < tokens; token += 1) {
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      const name = atomName(batch.refAtomNameChars, slot);
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
        + String(batch.features.residueIndex[token]).padStart(4) + "    "
        + positions[slot * 3].toFixed(3).padStart(8)
        + positions[slot * 3 + 1].toFixed(3).padStart(8)
        + positions[slot * 3 + 2].toFixed(3).padStart(8)
        + "  1.00" + confidence.toFixed(2).padStart(6) + "          "
        + (ELEMENT_SYMBOL[batch.refElement[slot]] ?? "C").padStart(2));
      serial += 1;
    }
  }
  lines.push("END");
  return lines.join("\n");
}

/**
 * Backbone bond lengths and the radius of gyration.
 *
 * 🔴 THIS IS THE CHECK THAT MATTERS, NOT pLDDT. A confident-looking pLDDT comes
 * off the TRUNK and says nothing about whether the diffusion produced a
 * molecule. A batch built with a broken gather once folded a 17 A spaghetti at
 * a mean pLDDT of 55 with 15 A between consecutive CA - the pLDDT looked merely
 * unimpressive and the CA-CA was impossible.
 */
export function backboneGeometry(batch, positions) {
  const { tokens, dense } = batch;
  const slotOf = [];
  for (let token = 0; token < tokens; token += 1) {
    const atoms = {};
    for (let atom = 0; atom < dense; atom += 1) {
      const slot = token * dense + atom;
      if (!batch.predDenseAtomMask[slot]) continue;
      atoms[atomName(batch.refAtomNameChars, slot)] = slot;
    }
    slotOf.push(atoms);
  }
  const distance = (a, b) => Math.hypot(
    positions[a * 3] - positions[b * 3],
    positions[a * 3 + 1] - positions[b * 3 + 1],
    positions[a * 3 + 2] - positions[b * 3 + 2]);
  const median = (values) => {
    if (values.length === 0) return NaN;
    const sorted = [...values].sort((x, y) => x - y);
    return sorted[Math.floor(sorted.length / 2)];
  };

  const nca = [];
  const cac = [];
  const caca = [];
  for (let token = 0; token < tokens; token += 1) {
    const here = slotOf[token];
    if (here.N !== undefined && here.CA !== undefined) nca.push(distance(here.N, here.CA));
    if (here.CA !== undefined && here.C !== undefined) cac.push(distance(here.CA, here.C));
    if (token + 1 < tokens && here.CA !== undefined && slotOf[token + 1].CA !== undefined) {
      caca.push(distance(here.CA, slotOf[token + 1].CA));
    }
  }

  const centre = [0, 0, 0];
  let count = 0;
  for (let token = 0; token < tokens; token += 1) {
    const ca = slotOf[token].CA;
    if (ca === undefined) continue;
    for (let axis = 0; axis < 3; axis += 1) centre[axis] += positions[ca * 3 + axis];
    count += 1;
  }
  for (let axis = 0; axis < 3; axis += 1) centre[axis] /= count;
  let gyration = 0;
  for (let token = 0; token < tokens; token += 1) {
    const ca = slotOf[token].CA;
    if (ca === undefined) continue;
    for (let axis = 0; axis < 3; axis += 1) gyration += (positions[ca * 3 + axis] - centre[axis]) ** 2;
  }

  return {
    nca: median(nca), cac: median(cac), caca: median(caca),
    gyration: Math.sqrt(gyration / count), residues: count,
  };
}

/** AF3's target_feat: 447 columns, of which 384 come from the atom encoder. */
export function buildTargetFeat(batch, weights) {
  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, batch.tokens, batch.dense, weights.reference);

  const atomFeatures = atomCrossAttentionEncoder({
    shape: batch.shape, conditioning, atomMask: batch.refMask,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
  }, weights.encoder);

  return targetFeatures({
    aatype: batch.aatype, profile: batch.profile, deletionMean: batch.deletionMean,
    atomFeatures: atomFeatures.tokenAct,
  }, batch.tokens);
}

/** A reproducible normal deviate, so a seed names a structure. */
export function normalFrom(seed) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  };
  return () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
}

/**
 * @param {object} batch from featuriseProtein or an AF3 dump
 * @param {{trunk, diffusion, confidence, atomReference, targetFeat}} weights
 * @param {{steps?: number, seed?: number, blocks?: number,
 *          onStage?: (name: string, detail: object) => void,
 *          onStep?: (step: object) => void}} [options]
 */
export async function foldBatch(device, batch, weights, options = {}) {
  const steps = options.steps ?? 200;
  const { tokens, dense } = batch;
  const stage = (name, detail = {}) => options.onStage?.(name, detail);

  // 🔴 THIS IS NOT INSTANT AND IT IS NOT ON THE GPU. buildTargetFeat is the
  // per-atom conditioning and the conditioning atom encoder, both on the CPU,
  // and on a 68-residue chain it is 4.9 s - LONGER THAN THE 48-BLOCK TRUNK it
  // feeds. It is announced before it runs, and awaited, so a page can say what
  // is happening and hand back a frame first; otherwise the main thread simply
  // stops for five seconds with the last thing anyone was told still on screen.
  await stage("target-feat-start");
  const targetFeat = buildTargetFeat(batch, weights.targetFeat);
  await stage("target-feat", { targetFeat });

  const seqMask = batch.seqMask;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  const trunk = await new Af3TrunkGpu(device).run({
    tokens, sequences: 1, templates: 4, targetFeat, features: batch.features,
    msaRows: batch.msa.subarray(0, tokens),
    deletionMatrix: batch.deletionMatrix.subarray(0, tokens),
    msaMask: batch.msaMask.subarray(0, tokens),
    pairMask, seqMask,
    previousPair: new Float32Array(tokens * tokens * 128),
    previousSingle: new Float32Array(tokens * 384),
  }, weights.trunk, DIALECT, {
    onStage: (name, ms) => stage("trunk", { name, ms }),
    onPairformerBlock: (index, total) => stage("pairformer-block", { index, total }),
  });
  stage("trunk-done", { trunk });

  // 🔴 THE DIFFUSION HEAD HAS ITS OWN FIVE REFERENCE EMBEDDINGS - same shapes as
  // the conditioning module's, different weights. Reusing one for both
  // type-checks and is a different model.
  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, weights.atomReference);

  const headInput = {
    shape: batch.shape, conditioning, atomMask: batch.predDenseAtomMask, seqMask,
    features: batch.features, targetFeat,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    trunkSingle: trunk.single, trunkPair: trunk.pair,
  };

  const positions = await sampleOnGpu(device, headInput, weights.diffusion, {
    steps, stopAfter: options.stopAfter,
    normal: normalFrom(options.seed ?? 20260831),
    onStep: options.onStep,
  });

  // The confidence head reads the sample back.
  const beta = batch.tokenAtomsToPseudoBeta;
  const pseudoBeta = new Float32Array(tokens * 3);
  for (let token = 0; token < tokens; token += 1) {
    if (!beta.mask[token]) continue;
    const from = Number(beta.indices[token]) * 3;
    for (let axis = 0; axis < 3; axis += 1) pseudoBeta[token * 3 + axis] = positions[from + axis];
  }
  const scores = await new Af3ConfidenceHeadGpu(device).run({
    tokens, dense, seqMask, pair: trunk.pair, single: trunk.single, targetFeat, pseudoBeta,
  }, weights.confidence, DIALECT);

  let total = 0;
  let count = 0;
  for (let index = 0; index < tokens * dense; index += 1) {
    if (!batch.predDenseAtomMask[index]) continue;
    total += scores.plddt[index];
    count += 1;
  }

  return {
    positions, trunk, targetFeat, scores,
    meanPlddt: total / count, atoms: count,
    geometry: backboneGeometry(batch, positions),
    pdb: toPdb(batch, positions, scores.plddt),
  };
}
