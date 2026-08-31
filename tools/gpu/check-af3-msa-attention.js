/**
 * AF3's MSA attention: GPU against src/af3/msa-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-msa-attention.js
 *
 * The attention weights come from the pair representation alone and are shared
 * by every sequence, so the check needs BOTH a ragged MSA mask (for the
 * depth-maximum key mask) and more than one sequence (or the sharing is
 * untested).
 */
import { msaAttention } from "../../src/af3/msa-reference.js";
import { Af3MsaAttentionGpu } from "../../src/af3/msa-attention-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack";
const MSA_CHANNELS = 64;
const PAIR_CHANNELS = 128;
const HEADS = 8;
const DIMENSION = 8;

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
  const tokens = Number(option(args, "tokens", "24"));
  const sequences = Number(option(args, "sequences", "16"));
  const block = Number(option(args, "block", "0"));
  const store = await HttpTensorStore.open(MANIFEST);

  const layer = async (leaf) => {
    const name = `${STACK}/msa_attention1/${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(block * stride, (block + 1) * stride);
  };

  const weights = {
    heads: HEADS, dimension: DIMENSION,
    actNormScale: await layer("act_norm/scale"),
    actNormOffset: await layer("act_norm/offset"),
    pairNormScale: await layer("pair_norm/scale"),
    pairNormOffset: await layer("pair_norm/offset"),
    pairLogits: await layer("pair_logits/weights"),
    vProjection: await layer("v_projection/weights"),
    gatingQuery: await layer("gating_query/weights"),
    outputProjection: await layer("output_projection/weights"),
  };

  const msa = deterministic(sequences * tokens * MSA_CHANNELS, 606 + tokens);
  const pair = deterministic(tokens * tokens * PAIR_CHANNELS, 909 + tokens);
  const msaMask = new Float32Array(sequences * tokens);
  for (let s = 0; s < sequences; s += 1) {
    for (let t = 0; t < tokens; t += 1) {
      msaMask[s * tokens + t] = ((s * 5 + t * 3) % 13) < 10 ? 1 : 0;
    }
  }

  const expected = msaAttention(msa, msaMask, pair, sequences, tokens, MSA_CHANNELS,
                                PAIR_CHANNELS, weights);
  const { output, elapsedMilliseconds, memory } = await new Af3MsaAttentionGpu(device)
    .run(msa, msaMask, pair, { sequences, tokens, msaChannels: MSA_CHANNELS,
                               pairChannels: PAIR_CHANNELS }, weights);
  const relRms = relativeRms(output, expected);
  console.log(`msa-attention\ttokens=${tokens} sequences=${sequences}`
    + `\trelRMS ${relRms.toExponential(2)}`
    + `\t${elapsedMilliseconds.toFixed(1)} ms\t${(memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);

  const bound = 1e-5;
  if (relRms > bound) throw new Error(`relRMS ${relRms.toExponential(2)} exceeds ${bound}`);
  return { tokens, sequences, block, relRms };
}
