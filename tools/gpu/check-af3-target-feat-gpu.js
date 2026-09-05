/**
 * target_feat's atom encoder on the GPU, against the CPU reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-target-feat-gpu.js
 *
 * WHY THIS EXISTS. Building target_feat is 5.0 s on a 68-residue chain and 4.9
 * of that is atomCrossAttentionEncoder on the CPU - longer than the 48-block
 * trunk it feeds, and the single slowest thing in a browser fold. There is
 * already a GPU kernel of exactly this shape.
 *
 * 🔴 THE DIFFUSION ENCODER IS A SUPERSET, NOT A DIFFERENT MODULE. Af3AtomEncoder
 * Gpu adds three terms the conditioning encoder does not have: the trunk's
 * single, the trunk's pair, and an embedding of the noisy positions. Each is a
 * BIAS-FREE linear of a layer-normed input, so feeding zeros makes all three
 * contribute exactly zero - layerNorm(0) is 0, and 0 through a linear with no
 * bias is 0. That is what makes one kernel serve both, and it is a property of
 * the arithmetic rather than a convention, which is why this file checks it
 * rather than assuming it.
 *
 * 🔴 THE WEIGHTS FOR THOSE TERMS STILL HAVE TO EXIST. The shader indexes them
 * whatever the input is, so they are supplied at the right lengths and their
 * VALUES are irrelevant. Zeros are used to make that explicit; anything else
 * would work identically and would suggest it mattered.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { atomCrossAttentionEncoder } from "../../src/af3/atom-encoder-reference.js";
import { ALPHAFOLD3 } from "../../src/af3/dialect.js";
import { Af3AtomEncoderGpu } from "../../src/af3/atom-encoder-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { targetFeatureWeights } from "../../src/af3/diffusion-weights.js";

const SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";

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

/** The conditioning encoder's weights, padded with the three the GPU wants. */
export function asDiffusionShapedWeights(encoder) {
  const trunkSingleChannels = 384;
  const trunkPairChannels = 128;
  return {
    ...encoder,
    trunkSingleChannels, trunkPairChannels,
    lnormTrunkSingleCondScale: new Float32Array(trunkSingleChannels),
    embedTrunkSingleCond: new Float32Array(trunkSingleChannels * encoder.channels),
    lnormTrunkPairCondScale: new Float32Array(trunkPairChannels),
    embedTrunkPairCond: new Float32Array(trunkPairChannels * encoder.pairChannels),
    atomPositionsToFeatures: new Float32Array(3 * encoder.channels),
  };
}

export async function main(device, args) {
  const model = args.find((a) => a.startsWith("--model="))?.slice(8)
    ?? "/model-af3-int5/manifest.json";
  const store = await openAf3Store(model);
  const weights = await targetFeatureWeights(store);
  const batch = featuriseProtein(SEQUENCE);
  const { tokens, dense } = batch;

  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, weights.reference);

  const shared = {
    shape: batch.shape, dialect: ALPHAFOLD3,
    conditioning, atomMask: batch.refMask,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
  };

  const cpuStarted = performance.now();
  const cpu = atomCrossAttentionEncoder(shared, weights.encoder);
  const cpuMs = performance.now() - cpuStarted;

  const atoms = tokens * dense;
  const gpuStarted = performance.now();
  const gpu = await new Af3AtomEncoderGpu(device).run({
    ...shared,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    // The three the conditioning encoder does not have - see the note above.
    tokenAtomsAct: new Float32Array(atoms * 3),
    trunkSingleCond: new Float32Array(tokens * 384),
    trunkPairCond: new Float32Array(tokens * tokens * 128),
  }, asDiffusionShapedWeights(weights.encoder));
  const gpuMs = performance.now() - gpuStarted;

  const relRms = relativeRms(gpu.tokenAct, cpu.tokenAct);
  console.log(`${SEQUENCE.length} residues, ${batch.atomCount} atoms,`
    + ` ${batch.subsets} subsets x 32 queries x 128 keys`);
  console.log(`token_act   relRMS ${relRms.toExponential(2)}`
    + `   over ${cpu.tokenAct.length} values`);
  console.log(`cpu ${cpuMs.toFixed(0)} ms   gpu ${gpuMs.toFixed(0)} ms`
    + `   ${(cpuMs / gpuMs).toFixed(1)}x`);

  return { relRms, cpuMs, gpuMs, tokens, atoms: batch.atomCount };
}
