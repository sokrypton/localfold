/**
 * When `maskPaddedKeys` matters, which is narrower than it looks.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-ablate.js
 *
 * 🔴 A PADDED KEY EXISTS ONLY WHEN THE MOLECULE IS SMALLER THAN THE KEY WINDOW.
 * The window is 128 atoms wide and `featurise.js` CLAMPS its start against the
 * real atom count rather than the padded query length, so for anything with 128
 * atoms or more every one of its 128 keys lands on a real atom and the key mask
 * is identically one. Below 128 the window cannot fit and the tail is padding.
 * The query layout is padded either way - 574 atoms occupy 1632 slots on a
 * 68-residue chain - but a padded QUERY is discarded downstream, and it is the
 * KEYS that AF3's `ref_space_uid = 0` collision reaches.
 *
 * So the branch is inert on nearly every real fold, and that is worth knowing
 * rather than assuming: an ablation of it on 6MRR came back BIT-IDENTICAL, and
 * the honest reading of that is "no padded keys on this input", not "the flag
 * did not arrive". This sweeps across the boundary so the rule can be seen
 * rather than argued.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { Af3AtomEncoderGpu } from "../../src/af3/atom-encoder-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { targetFeatureWeights } from "../../src/af3/diffusion-weights.js";
import { ALPHAFOLD3, OPENBIND0 } from "../../src/af3/dialect.js";
import { asDiffusionShapedWeights } from "./check-af3-target-feat-gpu.js";

const SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

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

export async function main(device, args) {
  const store = await openAf3Store(
    option(args, "model", "/model-openbind-int5/manifest.json"));
  const weights = await targetFeatureWeights(store);
  const lengths = option(args, "lengths", "4,8,12,16,20,68")
    .split(",").map(Number);
  const rows = [];
  for (const length of lengths) rows.push(await measure(device, weights, length));
  console.table?.(rows);
  for (const row of rows) {
    console.log(`${String(row.residues).padStart(3)} residues`
      + `\t${String(row.atoms).padStart(4)} atoms`
      + `\t${String(row.paddedKeys).padStart(4)} padded keys of ${row.keySlots}`
      + `\ttokenAct ${row.separation.toExponential(2)}`
      + `\tpairCond ${row.pairSeparation.toExponential(2)}`
      + `\tcontrol ${row.controlSeparation.toExponential(2)}`);
  }
  return { rows };
}

async function measure(device, weights, residues) {
  const batch = featuriseProtein(SEQUENCE.slice(0, residues));
  const { tokens, dense } = batch;

  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, weights.reference);

  const atoms = tokens * dense;
  const base = {
    shape: batch.shape, conditioning, atomMask: batch.refMask,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    tokenAtomsAct: new Float32Array(atoms * 3),
    trunkSingleCond: new Float32Array(tokens * 384),
    trunkPairCond: new Float32Array(tokens * tokens * 128),
  };
  const shaped = asDiffusionShapedWeights(weights.encoder);

  const run = async (dialect) => {
    const out = await new Af3AtomEncoderGpu(device).run({ ...base, dialect }, shaped);
    return { tokenAct: out.tokenAct, pairCond: out.pairCond };
  };

  const stock = await run(ALPHAFOLD3);
  const openbindOut = await run(OPENBIND0);
  const openbind = openbindOut.tokenAct;
  const pairSeparation = relativeRms(openbindOut.pairCond, stock.pairCond);
  const stockAct = stock.tokenAct;

  // 🔴 A POSITIVE CONTROL, because "relRMS 0.00e+0" is what a broken comparison
  // says too. Perturb the conditioning by one part in 10^6 and re-run: if that
  // does not move tokenAct either, the measurement above is measuring nothing.
  const nudged = Float32Array.from(base.conditioning, (v) => v * (1 + 1e-6));
  const control = await new Af3AtomEncoderGpu(device)
    .run({ ...base, conditioning: nudged, dialect: ALPHAFOLD3 }, shaped);
  const controlSeparation = relativeRms(control.tokenAct, stockAct);

  // 🔴 HOW MUCH PADDING THERE ACTUALLY IS, and it is NOT the queries-to-keys
  // gather's own mask - that is 1 nearly everywhere, because the 128-key window
  // is shifted in bounds rather than truncated. A key is padding when the QUERY
  // it points at is not a real atom, which is the mask the encoder builds.
  let liveGather = 0;
  for (const value of batch.queriesToKeys.mask) liveGather += value ? 1 : 0;
  let live = 0;
  for (let index = 0; index < batch.queriesToKeys.mask.length; index += 1) {
    const query = Number(batch.queriesToKeys.indices[index]);
    if (batch.queriesToKeys.mask[index] && batch.tokenAtomsToQueries.mask[query]
        && batch.refMask[Number(batch.tokenAtomsToQueries.indices[query])]) live += 1;
  }
  void liveGather;
  return {
    residues,
    tokens,
    atoms: batch.atomCount,
    querySlots: batch.tokenAtomsToQueries.mask.length,
    keySlots: batch.queriesToKeys.mask.length,
    paddedKeys: batch.queriesToKeys.mask.length - live,
    separation: relativeRms(openbind, stockAct),
    pairSeparation,
    controlSeparation,
  };
}
