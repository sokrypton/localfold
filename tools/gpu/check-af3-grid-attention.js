/**
 * AF3 triangle (grid) self-attention: GPU against src/af3/pairformer-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-grid-attention.js
 *
 * Both directions, real trunk weights. pair_attention1 is the row direction and
 * pair_attention2 the column one, which the reference gets by transposing the
 * normalised activation - so the transposed case is where the interesting bugs
 * are, and it is checked with a RAGGED MASK so that a kernel using the query's
 * mask instead of the key's fails here rather than in a fold.
 */
import { gridSelfAttention } from "../../src/af3/pairformer-reference.js";
import { Af3GridSelfAttentionGpu } from "../../src/af3/grid-attention-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer";
const CHANNELS = 128;
const HEADS = 4;
const DIMENSION = 32;
// Stock AF3 does not transpose the pair bias; the openfold3 lineage does.
const DIALECT = { swapTransposedBias: false };

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

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
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const n = Number(option(args, "n", "24"));
  const block = Number(option(args, "block", "0"));
  const store = await HttpTensorStore.open(MANIFEST);
  const runner = new Af3GridSelfAttentionGpu(device);

  const layer = async (leaf) => {
    const name = `${STACK}/${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(block * stride, (block + 1) * stride);
  };

  const pair = deterministic(n * n * CHANNELS, 991 + n);
  // 🔴 RAGGED, so the key-vs-query mask confusion is visible. With an all-ones
  // mask both choices agree and the check passes on a kernel that is wrong.
  const sequence = new Float32Array(n);
  for (let i = 0; i < n; i += 1) sequence[i] = i < Math.ceil(n * 0.75) ? 1 : 0;
  const mask = new Float32Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) mask[i * n + j] = sequence[i] * sequence[j];
  }

  const results = {};
  const precisions = option(args, "precision", "f32,f16").split(",");
  const bounds = { f32: 1e-5, f16: 2e-3 };
  let failed = 0;
  for (const [module, transpose] of [["pair_attention1", false], ["pair_attention2", true]]) {
    const at = (leaf) => layer(`${module}/${leaf}`);
    const weights = {
      heads: HEADS, dimension: DIMENSION,
      actNormScale: await at("act_norm/scale"),
      actNormOffset: await at("act_norm/offset"),
      pairBiasProjection: await at("pair_bias_projection/weights"),
      qProjection: await at("q_projection/weights"),
      kProjection: await at("k_projection/weights"),
      vProjection: await at("v_projection/weights"),
      gatingQuery: await at("gating_query/weights"),
      outputProjection: await at("output_projection/weights"),
    };
    const expected = gridSelfAttention(pair, mask, n, CHANNELS, transpose, weights, DIALECT);
    // 🔴 THE STAGED TILE'S PRECISION IS AN AXIS, NOT A RAISED BOUND. A block
    // stages the key and the value in f16 wherever the device has shader-f16,
    // which is eleven significant bits - so one tolerance cannot hold both, and
    // widening the single bound would stop the f32 kernel being checked at the
    // 1e-5 it actually reaches. Each runs against the same CPU reference.
    for (const stagedPrecision of precisions) {
      if (stagedPrecision === "f16" && !device.features.has("shader-f16")) continue;
      const { output, elapsedMilliseconds, memory } = await runner.run(
        pair, mask, { n, channels: CHANNELS, transpose }, weights, DIALECT, { stagedPrecision });
      const relRms = relativeRms(output, expected);
      const bound = bounds[stagedPrecision];
      if (relRms > bound) failed += 1;
      results[`${module}/${stagedPrecision}`] = {
        transpose, stagedPrecision, relRms, bound,
        ms: Number(elapsedMilliseconds.toFixed(2)),
        peakMiB: Number((memory.peakBytes / 2 ** 20).toFixed(2)),
      };
      console.log(`${module}\t${stagedPrecision}\ttranspose=${transpose}`
        + `\trelRMS ${relRms.toExponential(2)}\tbound ${bound.toExponential(0)}`
        + `\t${elapsedMilliseconds.toFixed(1)} ms`);
    }
  }

  if (failed > 0) throw new Error(`${failed} grid attention precision(s) outside tolerance`);
  return { n, block, results };
}
