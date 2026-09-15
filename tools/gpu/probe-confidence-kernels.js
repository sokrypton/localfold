/**
 * Which kernel of the CONFIDENCE head's pairformer stops computing its
 * reference - the trunk's own stack being the control.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-confidence-kernels.js \
 *       --model=/model-af3-full-f32/manifest.json
 *
 * 🔴 WHY IT EXISTS. `check-af3-confidence.js` fails all four heads by 500-3700x
 * their envelope, and its own bisection says the divergence is not compounding:
 * ONE block already reads 1.24e-3 on the pair track where a 5e-7 nudge to both
 * tracks each block produces 1.09e-6. But `check-af3-block.js` passes, and it
 * hand-builds its weight dict with `heads: 4` and `pairChannels: 128` typed in
 * - the confidence stack's widths exactly - so the block code path is verified
 * against SYNTHETIC weights and the real ones still disagree. A whole-block
 * relRMS cannot say which of its six updates did it.
 *
 * 🔴 AND THE CONTROL IS THE POINT, NOT THE MEASUREMENT. `probe-opendde-kernels`
 * runs these same four comparisons against the TRUNK's blocks and is where this
 * is taken from; running them on the confidence blocks alone would give four
 * numbers with nothing to be large against. `--stack=trunk` runs the identical
 * code on `pairformerBlockWeights`, so a kernel that is fine there and wrong
 * here is a kernel that breaks on the confidence stack's own weights rather
 * than one that was always broken.
 *
 * 🔴 AND IT COVERS THE SINGLE TRACK, WHICH THE OpenDDE PROBE DOES NOT. The
 * confidence block updates single as well as pair - single attention biased by
 * the pair logits, then a single transition - and those two have no per-kernel
 * checker at all. `check-af3-single-attention.js` exists but is pinned to the
 * trunk's 384 with its own hand-built weights.
 */
import { af3TriangleMultiplication } from "../../src/af3/trunk/triangle-webgpu.js";
import { Af3GridSelfAttentionGpu } from "../../src/af3/trunk/grid-attention-webgpu.js";
import {
  gridSelfAttention, pairformerBlock, transition, triangleMultiplication,
} from "../../src/af3/trunk/pairformer-reference.js";
import { Af3TransitionGpu } from "../../src/af3/trunk/transition-webgpu.js";
import { Af3PairformerStackGpu } from "../../src/af3/trunk/pairformer-block-webgpu.js";
import {
  af3Dialect, confidenceWeights, msaBlockWeights, openAf3Store, pairformerBlockWeights,
} from "../../src/af3/weights/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function deterministic(length, seed) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = (((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1;
  }
  return output;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const d = actual[index] - expected[index];
    error += d * d;
    scale += expected[index] * expected[index];
  }
  return Number(Math.sqrt(error / Math.max(scale, 1e-30)).toExponential(3));
}

export async function main(device, args) {
  const n = Number(option(args, "n", "24"));
  const manifest = option(args, "model", "/model-af3-full-f32/manifest.json");
  const which = option(args, "stack", "confidence");
  const at = Number(option(args, "block", "0"));
  const store = await openAf3Store(manifest);
  // 🔴 THE SAME DIALECT THE CHECKER PINS. check-af3-confidence.js runs both
  // sides with `{ swapTransposedBias: false }` rather than the store's, so a
  // probe reading the store's would be comparing a different model and its
  // disagreement would mean nothing about the failure being chased.
  const dialect = option(args, "dialect", "checker") === "store"
    ? af3Dialect(store) : { swapTransposedBias: false };

  // 🔴 THREE STACKS RUN THIS PAIR TRACK AND THEY ARE NOT INTERCHANGEABLE. The
  // MSA arm exists because check-af3-msa-block's VECTOR arm - fully f32, no
  // knob and no `--no-prior` moving it - reads 1.18e-5 at 35.8x its envelope
  // while its MSA track reads 4.65e-6 and the outer product mean alone reads
  // 5.88e-7. Same question as the confidence head's: is any one kernel wrong,
  // or is it the composition?
  const block = which === "confidence"
    ? (await confidenceWeights(store)).blocks[at]
    : which === "msa" ? await msaBlockWeights(store, at)
    : await pairformerBlockWeights(store, at);
  const channels = block.pairChannels;
  const pairs = n * n;
  // 🔴 AND THE INPUT MAGNITUDE IS AN AXIS TOO. check-af3-confidence.js reports
  // its pair growing from 0.58 to 177 through the stack, and a probe feeding
  // every kernel a [-1,1] tensor measures none of that range. f32 is
  // scale-invariant, so a kernel whose relRMS MOVES with `--scale` is holding
  // an absolute epsilon, clamp or bias somewhere it should not.
  const scale = Number(option(args, "scale", "1"));
  const pair = deterministic(pairs * channels, 7);
  if (scale !== 1) for (let i = 0; i < pair.length; i += 1) pair[i] *= scale;
  // 🔴 THE MASK IS AN AXIS, AND AN ALL-ONES ONE HIDES THE BUG BEING CHASED.
  // check-af3-confidence.js marks the last 20% of tokens as padding, so its
  // pair mask is an outer product with a zero block; a probe running every
  // kernel at mask 1 exercises no masking at all and reports every kernel
  // clean, which is exactly what the first run of this file did.
  const keep = Number(option(args, "keep", "0.8"));
  const seqMask = new Float32Array(n);
  for (let t = 0; t < n; t += 1) seqMask[t] = t < Math.ceil(n * keep) ? 1 : 0;
  const mask = new Float32Array(pairs);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) mask[i * n + j] = seqMask[i] * seqMask[j];
  }

  const results = {};

  for (const direction of ["outgoing", "incoming"]) {
    const weights = direction === "outgoing"
      ? block.triangleMultiplicationOutgoing : block.triangleMultiplicationIncoming;
    const expected = triangleMultiplication(pair, mask, n, channels, direction, weights);
    const { output } = await af3TriangleMultiplication(
      device, pair, mask, n, channels, direction, weights, { precision: "f32" });
    results[`triangle.${direction}`] = relativeRms(output, expected);
  }

  const grid = new Af3GridSelfAttentionGpu(device);
  for (const [name, weights, transpose] of [
    ["grid.1", block.pairAttention1, false], ["grid.2", block.pairAttention2, true]]) {
    const expected = gridSelfAttention(pair, mask, n, channels, transpose, weights, dialect);
    const actual = await grid.run(pair, mask, { n, channels, transpose }, weights, dialect,
                                  { stagedPrecision: "f32" });
    results[name] = relativeRms(actual.output ?? actual, expected);
  }

  const expectedTransition = transition(pair, pairs, channels, block.pairTransition);
  const gpuTransition = await new Af3TransitionGpu(device).run(
    pair, { rows: pairs, channels, factor: 4 }, block.pairTransition, {});
  results["pair-transition"] = relativeRms(
    gpuTransition.output ?? gpuTransition, expectedTransition);

  // The single track, which the OpenDDE probe has no reason to cover.
  const singleChannels = block.singleChannels;
  if (singleChannels !== undefined && block.singleTransition !== undefined) {
    const single = deterministic(n * singleChannels, 11);
    const expectedSingle = transition(single, n, singleChannels, block.singleTransition);
    const gpuSingle = await new Af3TransitionGpu(device).run(
      single, { rows: n, channels: singleChannels, factor: 4 }, block.singleTransition, {});
    results["single-transition"] = relativeRms(gpuSingle.output ?? gpuSingle, expectedSingle);
  }

  // 🔴 THE BLOCK ARM, WHICH IS WHERE THE KERNELS STOP EXPLAINING IT. Every
  // kernel above reads ~5e-7 on both stacks, at every mask and every scale,
  // and check-af3-confidence still reports 1.24e-3 after ONE block. So the
  // composition is the remaining suspect and this is the arm that says so:
  // one GPU block against the reference's own block, on the SAME real weights
  // the stack runs, rather than the hand-built dict check-af3-block.js types in.
  if (option(args, "level", "both") !== "kernel" && singleChannels !== undefined
      && block.singleAttention !== undefined) {
    const single = deterministic(n * singleChannels, 11);
    const state = { pair, single, pairMask: mask, seqMask, tokens: n };
    const expected = pairformerBlock(state, block, dialect);
    const got = await new Af3PairformerStackGpu(device, {
      stagedPrecision: "f32", weightPrecision: "f32", accumulatePrecision: "f32",
    }).run(state, [block], dialect, {});
    results["BLOCK.pair"] = relativeRms(got.pair, expected.pair);
    results["BLOCK.single"] = relativeRms(got.single, expected.single);
    // 🔴 AND ITS OWN ENVELOPE, BECAUSE A BLOCK ON RANDOM WEIGHTS IS CHAOTIC AND
    // ON REAL ONES IS NOT. check-af3-block.js builds its weight dict by hand,
    // and on that input one kernel's worth of rounding (1e-7) grows to 5.74e-4
    // over four blocks - so its bound is that envelope times 300 and swallows
    // almost anything. These are the bundle's REAL weights, so the same
    // perturbation is the honest scale for this comparison.
    const nudged = Float32Array.from(pair);
    for (let i = 0; i < nudged.length; i += 1) nudged[i] += pair[i] * 1e-7;
    const control = pairformerBlock({ ...state, pair: nudged }, block, dialect);
    results["BLOCK.envelope"] = relativeRms(control.pair, expected.pair);
    results["BLOCK.xEnvelope"] = Number(
      (results["BLOCK.pair"] / Math.max(results["BLOCK.envelope"], 1e-30)).toFixed(1));
  }

  return {
    model: manifest, stack: which, block: at, n, keep, scale,
    widths: { pairChannels: channels, singleChannels,
              gridHeads: block.pairAttention1.heads,
              gridDimension: block.pairAttention1.dimension,
              singleHeads: block.singleAttention?.heads,
              singleDimension: block.singleAttention?.dimension },
    results,
  };
}
