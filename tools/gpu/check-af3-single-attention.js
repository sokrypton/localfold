/**
 * AF3 single-track attention: GPU against src/af3/pairformer-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-single-attention.js
 *
 * Real trunk weights, a ragged sequence mask, and pair logits drawn at random -
 * the logits are an input to this module, built by the block from the pair
 * representation, so feeding real ones would be testing the block instead.
 */
import { singleAttention } from "../../src/af3/pairformer-reference.js";
import { Af3SingleAttentionGpu } from "../../src/af3/single-attention-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer";
const CHANNELS = 384;
const HEADS = 16;
const DIMENSION = 24;

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
  const n = Number(option(args, "n", "48"));
  const block = Number(option(args, "block", "0"));
  const store = await HttpTensorStore.open(MANIFEST);

  const layer = async (leaf) => {
    const name = `${STACK}/${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(block * stride, (block + 1) * stride);
  };

  const weights = {
    heads: HEADS, dimension: DIMENSION,
    layerNormScale: await layer("single_attention_layer_norm/scale"),
    layerNormOffset: await layer("single_attention_layer_norm/offset"),
    qProjection: await layer("single_attention_q_projection/weights"),
    qBias: await layer("single_attention_q_projection/bias"),
    kProjection: await layer("single_attention_k_projection/weights"),
    vProjection: await layer("single_attention_v_projection/weights"),
    gatingQuery: await layer("single_attention_gating_query/weights"),
    outputProjection: await layer("single_attention_transition2/weights"),
  };

  const single = deterministic(n * CHANNELS, 5150 + n);
  const pairLogits = deterministic(HEADS * n * n, 77 + n);
  // 🔴 RAGGED. The mask term is additive and finite here; with an all-ones mask
  // it vanishes and a kernel that drops it entirely still passes.
  const seqMask = new Float32Array(n);
  for (let i = 0; i < n; i += 1) seqMask[i] = i < Math.ceil(n * 0.7) ? 1 : 0;

  const expected = singleAttention(single, pairLogits, seqMask, n, CHANNELS, weights);
  const { output, elapsedMilliseconds, memory } = await new Af3SingleAttentionGpu(device)
    .run(single, pairLogits, seqMask, { n, channels: CHANNELS }, weights);
  const relRms = relativeRms(output, expected);
  console.log(`single\tn=${n}\trelRMS ${relRms.toExponential(2)}`
    + `\t${elapsedMilliseconds.toFixed(1)} ms\t${(memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);

  const bound = 1e-5;
  if (relRms > bound) throw new Error(`relRMS ${relRms.toExponential(2)} exceeds ${bound}`);
  return { n, block, relRms };
}
