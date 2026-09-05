/**
 * Our denoiser against AF3's, at every noise level AF3's own sampler visited.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 48 --recycles 0 \
 *       --diffusion 20 --float32 --sequence PIAQ...ASK \
 *       --capture 'diffusion_head/__call__$|evoformer/__call__$' \
 *       --capture-args 'diffusion_head/__call__$' --out af3-rings20.json
 *     node tools/gpu-chrome.mjs tools/gpu/probe-head-vs-af3-steps.js \
 *       --dump=/af3-rings20.json
 *
 * 🔴 THE ONE ORACLE POINT THE DENOISER HAD WAS sigma = 2560, AND THE EDM
 * PRECONDITIONING IS DEGENERATE THERE. D = skip*x + out*F with
 * skip = sigma_d^2/(sigma^2+sigma_d^2) and out = sigma*sigma_d/sqrt(...): at the
 * top of the schedule skip is 3.9e-5 and out is 15.9997, so skip is
 * indistinguishable from zero and out from sigma_d. A wrong formula for either
 * still scores 6.8e-6 there. Every rung below is untested, and that is exactly
 * where a structure's side chains are decided.
 *
 * This walks AF3's OWN trajectory: its noisy positions and its noise level at
 * each step are captured as the head's arguments, so both denoisers are asked
 * the identical question twenty times, from 4608 A down to 0.03.
 *
 * 🔴 AND THE TRUNK IS AF3'S TOO. Otherwise a disagreement at step 19 could be a
 * trunk that sent the two somewhere different at step 0.
 */
import { batchFromDump } from "./fold.js";
import { sidechainGeometry } from "./sidechain-geometry.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { Af3DiffusionHeadGpu } from "../../src/af3/diffusion-head-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference } from "../../src/af3/diffusion-weights.js";
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
  const dumpPath = option(args, "dump", "/af3-rings20.json");
  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const dump = await response.json();
  const batch = batchFromDump(dump);
  const { tokens, dense } = batch;

  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const diffusion = await diffusionWeights(store);
  const reference = await atomReference(store);

  const HEAD = "diffuser/~/diffusion_head/__call__";
  const steps = Object.keys(dump.outputs)
    .filter((key) => /^diffuser\/~\/diffusion_head\/__call__#\d+$/.test(key))
    .map((key) => Number(key.split("#")[1])).sort((a, b) => a - b);

  const base = {
    shape: batch.shape,
    dialect: ALPHAFOLD3,
    conditioning: perAtomConditioning({
      positions: batch.refPos, mask: batch.refMask,
      element: batch.refElement, charge: batch.refCharge,
      atomNameChars: batch.refAtomNameChars,
    }, tokens, dense, reference),
    atomMask: batch.predDenseAtomMask, seqMask: batch.seqMask, features: batch.features,
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
  for (const step of steps) {
    const noiseLevel = Number(dump.outputs[`${HEAD}<1#${step}`].data[0]);
    const positionsNoisy = floats(dump.outputs[`${HEAD}<0#${step}`].data);
    const theirs = floats(dump.outputs[`${HEAD}#${step}`].data);
    const ours = (await head.run({ ...base, positionsNoisy, noiseLevel }, diffusion)).positions;
    rows.push({
      step, sigma: Number(noiseLevel.toPrecision(4)),
      relRms: Number(relativeRms(ours, theirs).toPrecision(3)),
      bondOurs: sidechainGeometry(batch, ours).scale[0].median,
      bondAf3: sidechainGeometry(batch, theirs).scale[0].median,
    });
  }
  return rows;
}
