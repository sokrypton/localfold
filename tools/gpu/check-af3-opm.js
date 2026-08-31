/**
 * AF3's outer product mean: GPU against src/af3/msa-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-opm.js
 *
 * 🔴 THE MSA MASK HERE IS RAGGED PER SEQUENCE, not a single depth cut. The
 * denominator is the number of sequences covering BOTH tokens, and with a
 * rectangular mask that equals the count for either token alone and equals the
 * depth - so all three spellings agree and a wrong one passes. Gaps in
 * different places in different sequences are what separates them.
 */
import { outerProductMean } from "../../src/af3/msa-reference.js";
import { Af3OuterProductMeanGpu } from "../../src/af3/outer-product-mean-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack";
const MSA_CHANNELS = 64;
const PAIR_CHANNELS = 128;
const OUTER_CHANNELS = 32;

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
    const name = `${STACK}/outer_product_mean/${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(block * stride, (block + 1) * stride);
  };

  const weights = {
    outerChannels: OUTER_CHANNELS,
    layerNormInputScale: await layer("layer_norm_input/scale"),
    layerNormInputOffset: await layer("layer_norm_input/offset"),
    leftProjection: await layer("left_projection/weights"),
    rightProjection: await layer("right_projection/weights"),
    outputW: await layer("output_w"),
    outputB: await layer("output_b"),
  };

  const msa = deterministic(sequences * tokens * MSA_CHANNELS, 1234 + tokens);
  // Ragged in a different place per sequence - see the note above.
  const msaMask = new Float32Array(sequences * tokens);
  for (let s = 0; s < sequences; s += 1) {
    for (let t = 0; t < tokens; t += 1) {
      msaMask[s * tokens + t] = ((s * 7 + t * 3) % 11) < 8 ? 1 : 0;
    }
  }

  const expected = outerProductMean(msa, msaMask, sequences, tokens, MSA_CHANNELS,
                                    PAIR_CHANNELS, weights);
  const { output, elapsedMilliseconds, memory } = await new Af3OuterProductMeanGpu(device)
    .run(msa, msaMask, { sequences, tokens, msaChannels: MSA_CHANNELS,
                         pairChannels: PAIR_CHANNELS }, weights);
  const relRms = relativeRms(output, expected);
  console.log(`opm\ttokens=${tokens} sequences=${sequences}`
    + `\trelRMS ${relRms.toExponential(2)}`
    + `\t${elapsedMilliseconds.toFixed(1)} ms\t${(memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);

  const bound = 1e-5;
  if (relRms > bound) throw new Error(`relRMS ${relRms.toExponential(2)} exceeds ${bound}`);
  return { tokens, sequences, block, relRms,
           ms: elapsedMilliseconds, peakMiB: memory.peakBytes / 2 ** 20 };
}
