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

  // 🔴 SEVERAL DRAWS, AND THE WORST OF THEM, BECAUSE ONE DRAW WAS PASSING BY
  // LUCK. The input was `deterministic(n * n * CHANNELS, 991 + n)` - a
  // different random pair for every n - and this file only ever ran at its
  // default of 24 tokens. Swept, the f16 arm's error is dominated by WHICH
  // draw it got rather than by how big it is:
  //
  //     n     24      32      33      36      48      128     256
  //     f16   5.5e-4  1.1e-3  6.0e-4  1.2e-2  1.4e-2  7.1e-3  7.4e-3
  //     f32   9.6e-7  ...     ...     ...     1.0e-6  1.2e-6  1.3e-6
  //
  // The f32 arm is flat at about 1e-6 throughout; the f16 arm ranges over a
  // factor of 26. So a bound set from one draw says nothing, and the one that
  // was here - 2e-3, which is what n=24 happens to give - was failed by five
  // of the seven sizes above the moment anyone ran them.
  //
  // 🔴 AND THE MECHANISM IS THE SOFTMAX, NOT THE STORAGE. f16 holds eleven
  // mantissa bits, so a staged key is good to about 5e-4 - but the error lands
  // in a LOGIT, and exp turns an absolute logit error into a relative weight
  // error. Uniform noise makes large, poorly conditioned logits; a real pair
  // representation does not, which is why the whole trunk still agrees with
  // AF3 to 3.94e-4 end to end with this same path on. This input is harsher
  // than a fold, deliberately, and the bound below says so.
  const seeds = Number(option(args, "seeds", "4"));
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
  // 🔴 THE f16 BOUND IS THE MEASURED SPREAD, NOT ONE DRAW. Over four draws at
  // sizes from 24 to 256 the worst was 1.4e-2, so 2e-3 was a bound five of
  // seven sizes failed and nobody had run. 3e-2 is what this INPUT costs; the
  // fold does not, and the note above says why. The f32 arm keeps its 1e-5,
  // which it reaches at every size measured - widening one bound to cover both
  // is how the f32 path stops being checked at all.
  const bounds = { f32: 1e-5, f16: 3e-2 };
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
    // ...the worst of the draws, per precision, so one lucky input cannot
    // carry the check. Each draw is a fresh pair and its own reference.
    const draws = Array.from({ length: seeds }, (_, draw) => {
      const pair = deterministic(n * n * CHANNELS, 991 + n + draw * 7919);
      return { pair, expected: gridSelfAttention(
        pair, mask, n, CHANNELS, transpose, weights, DIALECT) };
    });
    // 🔴 THE STAGED TILE'S PRECISION IS AN AXIS, NOT A RAISED BOUND. A block
    // stages the key and the value in f16 wherever the device has shader-f16,
    // which is eleven significant bits - so one tolerance cannot hold both, and
    // widening the single bound would stop the f32 kernel being checked at the
    // 1e-5 it actually reaches. Each runs against the same CPU reference.
    for (const stagedPrecision of precisions) {
      if (stagedPrecision === "f16" && !device.features.has("shader-f16")) continue;
      let relRms = 0;
      let elapsedMilliseconds = 0;
      let memory;
      for (const draw of draws) {
        const run = await runner.run(
          draw.pair, mask, { n, channels: CHANNELS, transpose }, weights, DIALECT,
          { stagedPrecision });
        relRms = Math.max(relRms, relativeRms(run.output, draw.expected));
        elapsedMilliseconds = run.elapsedMilliseconds;
        memory = run.memory;
      }
      const bound = bounds[stagedPrecision];
      if (relRms > bound) failed += 1;
      results[`${module}/${stagedPrecision}`] = {
        transpose, stagedPrecision, seeds, relRms, bound,
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
