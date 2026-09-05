/**
 * One denoising step, GPU against the CPU reference, on a REAL molecule.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-head-cpu-vs-gpu.js \
 *       --dump=/af3-sample200.json --sigma=16
 *
 * 🔴 THE TOY DUMP VERIFIES THE WRONG TWO THINGS. check_af3_denoiser.js proves
 * the CPU reference exact against AF3 (6.8e-6) - but on twelve residues, at
 * noise level 2560, with no ring in the sequence. check-af3-diffusion-head.js
 * proves the GPU against that CPU reference - but on the same twelve residues
 * and with a THREE-block truncated model. So the link that the page actually
 * runs, the full GPU head on a real chain, is checked by neither.
 *
 * This closes it: AF3's own batch and AF3's own trunk, the full 24-block
 * transformer and the full three-block atom encoder and decoder, both
 * implementations, one call, at whatever noise level the schedule reaches.
 *
 * 🔴 THE NOISE LEVEL IS THE EXPERIMENT, NOT A DETAIL. skip and out are both 1/2
 * at sigma = SIGMA_DATA and swap places either side of it, so a disagreement
 * that only appears low on the schedule is invisible at the top - which is the
 * only place the oracle has ever been asked.
 *
 * 🔴 AND THE INPUT IS AF3'S OWN FINAL STRUCTURE, NOISED. Denoising a Gaussian
 * asks a question whose right answer is a blur: AF3's own one-step output at
 * sigma 2560 has a bond ratio of 0.468, so "compressed" is CORRECT there and
 * measuring geometry off it says nothing. Starting from the real structure and
 * adding sigma of noise asks the question the end of a sampler asks.
 */
import { batchFromDump } from "./fold.js";
import { sidechainGeometry } from "./sidechain-geometry.js";
import { normalFrom } from "../../src/af3/fold.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { atomCrossAttentionEncoder } from "../../src/af3/atom-encoder-reference.js";
import { diffusionHead } from "../../src/af3/diffusion-reference.js";
import { Af3DiffusionHeadGpu } from "../../src/af3/diffusion-head-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../src/af3/diffusion-weights.js";
import { ALPHAFOLD3 } from "../../src/af3/dialect.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const floats = (source) => Float32Array.from(source, (v) => Number(v));

function lastCapture(dump, base) {
  if (dump.outputs[base] !== undefined) return dump.outputs[base];
  return Object.keys(dump.outputs)
    .filter((key) => key.startsWith(`${base}#`))
    .sort((a, b) => Number(a.split("#")[1]) - Number(b.split("#")[1]))
    .map((key) => dump.outputs[key]).pop();
}

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
  const dumpPath = option(args, "dump", "/af3-sample200.json");
  const sigmas = option(args, "sigmas", "16").split(",").map(Number);
  const withCpu = args.includes("--cpu");

  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const dump = await response.json();
  const batch = batchFromDump(dump);
  const { tokens, dense } = batch;

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const diffusion = await diffusionWeights(store);
  const reference = await atomReference(store);
  await targetFeatureWeights(store);

  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, tokens, dense, reference);

  // 🔴 CENTRED ON THE REAL ATOMS FIRST. The head embeds the scaled positions
  // through a Linear, which is not translation invariant, and AF3 only ever
  // calls it from inside the sampler - immediately after an augmentation that
  // centres. Handing it an off-centre structure asks a question AF3 never asks.
  const truth = floats(dump.outputs["diffusion_samples/atom_positions"].data);
  {
    const centre = [0, 0, 0];
    let count = 0;
    for (let atom = 0; atom < tokens * dense; atom += 1) {
      if (!batch.predDenseAtomMask[atom]) continue;
      count += 1;
      for (let axis = 0; axis < 3; axis += 1) centre[axis] += truth[atom * 3 + axis];
    }
    for (let atom = 0; atom < tokens * dense; atom += 1) {
      if (!batch.predDenseAtomMask[atom]) continue;
      for (let axis = 0; axis < 3; axis += 1) truth[atom * 3 + axis] -= centre[axis] / count;
    }
  }
  const base = {
    shape: batch.shape, conditioning, atomMask: batch.predDenseAtomMask,
    dialect: ALPHAFOLD3,
    seqMask: batch.seqMask, features: batch.features,
    targetFeat: floats(lastCapture(dump, "diffuser/evoformer/__call__:target_feat").data),
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    trunkSingle: floats(lastCapture(dump, "diffuser/evoformer/__call__:single").data),
    trunkPair: floats(lastCapture(dump, "diffuser/evoformer/__call__:pair").data),
  };

  const head = new Af3DiffusionHeadGpu(device);
  const rows = [];
  for (const sigma of sigmas) {
    // The same draw at every level, so the rows compare noise levels rather
    // than draws.
    const normal = normalFrom(7);
    const positionsNoisy = new Float32Array(tokens * dense * 3);
    for (let index = 0; index < positionsNoisy.length; index += 1) {
      positionsNoisy[index] = truth[index] + sigma * normal();
    }
    const input = { ...base, positionsNoisy, noiseLevel: sigma };
    const gpu = (await head.run(input, diffusion)).positions;
    // ...and the same call with the TRUE structure handed in unnoised. At low
    // sigma `skip` is nearly one, so a denoiser that returned its input
    // untouched would score a perfect ratio on the noisy row and give itself
    // away here: this asks what the network believes, not what it copied.
    const clean = (await head.run({ ...input, positionsNoisy: truth }, diffusion)).positions;
    const row = {
      sigma,
      noisyIn: sidechainGeometry(batch, positionsNoisy).scale[0].median,
      gpuOut: sidechainGeometry(batch, gpu).scale[0].median,
      gpuFromTruth: sidechainGeometry(batch, clean).scale[0].median,
      af3Truth: sidechainGeometry(batch, truth).scale[0].median,
    };
    if (withCpu) {
      const cpu = diffusionHead(input, diffusion, atomCrossAttentionEncoder);
      row.gpuVsCpu = relativeRms(gpu, cpu).toExponential(2);
      row.cpuOut = sidechainGeometry(batch, cpu).scale[0].median;
    }
    rows.push(row);
  }
  return rows;
}
