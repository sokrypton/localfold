/**
 * Which pair-track kernel stops computing its reference at OpenDDE's widths.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-opendde-kernels.js
 *
 * 🔴 EVERY PER-KERNEL CHECKER HERE IS PINNED TO AlphaFold 3's CONSTANTS.
 * check-af3-triangle.js has `const CHANNELS = 128`, check-af3-grid-attention.js
 * has 128 with 4 heads of 32, check-af3-single-attention.js has 384 - so the
 * whole differential suite is blind to a second bundle's widths, which is
 * exactly where a second bundle breaks. This runs the same four comparisons
 * with the widths taken from a real bundle, one update at a time, so a failing
 * BLOCK can be attributed to a kernel rather than bisected by hand.
 */
import { af3TriangleMultiplication } from "../../src/af3/triangle-webgpu.js";
import { Af3GridSelfAttentionGpu } from "../../src/af3/grid-attention-webgpu.js";
import {
  gridSelfAttention, transition, triangleMultiplication,
} from "../../src/af3/pairformer-reference.js";
import { Af3TransitionGpu } from "../../src/af3/transition-webgpu.js";
import { af3Dialect, openAf3Store, pairformerBlockWeights } from "../../src/af3/weights.js";

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
  const manifest = option(args, "model", "/model-opendde-trunk-f32/manifest.json");
  const store = await openAf3Store(manifest);
  const dialect = af3Dialect(store);
  const block = await pairformerBlockWeights(store, 0);
  const channels = block.pairChannels;
  const pairs = n * n;
  const pair = deterministic(pairs * channels, 7);
  const mask = new Float32Array(pairs).fill(1);

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

  return {
    model: manifest, n,
    widths: { channels, gridHeads: block.pairAttention1.heads,
              gridDimension: block.pairAttention1.dimension },
    results,
  };
}
