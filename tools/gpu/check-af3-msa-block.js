/**
 * AF3's MSA blocks on the GPU, against src/af3/msa-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-msa-block.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-msa-block.js --blocks=4
 *
 * What this pins that the kernel checkers cannot: the two tracks are
 * INTERLEAVED. The outer product mean reads the MSA as it arrived, and the MSA
 * attention then reads the pair that the outer product just changed. Doing the
 * MSA track first and the pair track second, or the reverse, runs and returns
 * both representations.
 */
import { msaBlock } from "../../src/af3/msa-reference.js";
import { Af3MsaStackGpu } from "../../src/af3/msa-stack-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack";
const DIALECT = { swapTransposedBias: false };
const MSA_CHANNELS = 64;
const PAIR_CHANNELS = 128;

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
  const sequences = Number(option(args, "sequences", "16"));
  const count = Number(option(args, "blocks", "1"));
  const store = await HttpTensorStore.open(MANIFEST);

  const layer = async (leaf, index) => {
    const name = `${STACK}/${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(index * stride, (index + 1) * stride);
  };

  const blockWeights = async (index) => {
    const at = (leaf) => layer(leaf, index);
    const triangle = async (direction) => ({
      leftNormInputScale: await at(`triangle_multiplication_${direction}/left_norm_input/scale`),
      leftNormInputOffset: await at(`triangle_multiplication_${direction}/left_norm_input/offset`),
      projection: await at(`triangle_multiplication_${direction}/projection/weights`),
      gate: await at(`triangle_multiplication_${direction}/gate/weights`),
      centerNormScale: await at(`triangle_multiplication_${direction}/center_norm/scale`),
      centerNormOffset: await at(`triangle_multiplication_${direction}/center_norm/offset`),
      outputProjection: await at(`triangle_multiplication_${direction}/output_projection/weights`),
      gatingLinear: await at(`triangle_multiplication_${direction}/gating_linear/weights`),
    });
    const grid = async (which) => ({
      heads: 4, dimension: 32,
      actNormScale: await at(`pair_attention${which}/act_norm/scale`),
      actNormOffset: await at(`pair_attention${which}/act_norm/offset`),
      pairBiasProjection: await at(`pair_attention${which}/pair_bias_projection/weights`),
      qProjection: await at(`pair_attention${which}/q_projection/weights`),
      kProjection: await at(`pair_attention${which}/k_projection/weights`),
      vProjection: await at(`pair_attention${which}/v_projection/weights`),
      gatingQuery: await at(`pair_attention${which}/gating_query/weights`),
      outputProjection: await at(`pair_attention${which}/output_projection/weights`),
    });
    return {
      pairChannels: PAIR_CHANNELS, msaChannels: MSA_CHANNELS,
      outerProductMean: {
        outerChannels: 32,
        layerNormInputScale: await at("outer_product_mean/layer_norm_input/scale"),
        layerNormInputOffset: await at("outer_product_mean/layer_norm_input/offset"),
        leftProjection: await at("outer_product_mean/left_projection/weights"),
        rightProjection: await at("outer_product_mean/right_projection/weights"),
        outputW: await at("outer_product_mean/output_w"),
        outputB: await at("outer_product_mean/output_b"),
      },
      msaAttention1: {
        heads: 8, dimension: 8,
        actNormScale: await at("msa_attention1/act_norm/scale"),
        actNormOffset: await at("msa_attention1/act_norm/offset"),
        pairNormScale: await at("msa_attention1/pair_norm/scale"),
        pairNormOffset: await at("msa_attention1/pair_norm/offset"),
        pairLogits: await at("msa_attention1/pair_logits/weights"),
        vProjection: await at("msa_attention1/v_projection/weights"),
        gatingQuery: await at("msa_attention1/gating_query/weights"),
        outputProjection: await at("msa_attention1/output_projection/weights"),
      },
      msaTransition: {
        inputLayerNormScale: await at("msa_transition/input_layer_norm/scale"),
        inputLayerNormOffset: await at("msa_transition/input_layer_norm/offset"),
        transition1: await at("msa_transition/transition1/weights"),
        transition2: await at("msa_transition/transition2/weights"),
      },
      triangleMultiplicationOutgoing: await triangle("outgoing"),
      triangleMultiplicationIncoming: await triangle("incoming"),
      pairAttention1: await grid(1),
      pairAttention2: await grid(2),
      pairTransition: {
        inputLayerNormScale: await at("pair_transition/input_layer_norm/scale"),
        inputLayerNormOffset: await at("pair_transition/input_layer_norm/offset"),
        transition1: await at("pair_transition/transition1/weights"),
        transition2: await at("pair_transition/transition2/weights"),
      },
    };
  };

  const blocks = [];
  for (let index = 0; index < count; index += 1) blocks.push(await blockWeights(index));

  const sequence = new Float32Array(n);
  for (let i = 0; i < n; i += 1) sequence[i] = i < Math.ceil(n * 0.8) ? 1 : 0;
  const pairMask = new Float32Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) pairMask[i * n + j] = sequence[i] * sequence[j];
  }
  const msaMask = new Float32Array(sequences * n);
  for (let s = 0; s < sequences; s += 1) {
    for (let t = 0; t < n; t += 1) {
      msaMask[s * n + t] = sequence[t] > 0 && ((s * 7 + t * 3) % 11) < 8 ? 1 : 0;
    }
  }
  const state = {
    tokens: n, sequences,
    pair: deterministic(n * n * PAIR_CHANNELS, 313 + n),
    msa: deterministic(sequences * n * MSA_CHANNELS, 727 + n),
    pairMask, msaMask,
  };

  let cpu = { pair: state.pair, msa: state.msa };
  for (const weights of blocks) {
    cpu = msaBlock({ ...cpu, pairMask, msaMask, sequences, tokens: n }, weights, DIALECT);
  }

  const gpu = await new Af3MsaStackGpu(device).run(state, blocks, DIALECT);
  const pairRms = relativeRms(gpu.pair, cpu.pair);
  const msaRms = relativeRms(gpu.msa, cpu.msa);
  console.log(`${count} MSA block(s), n=${n}, ${sequences} sequences`);
  console.log(`pair\trelRMS ${pairRms.toExponential(2)}`);
  console.log(`msa\trelRMS ${msaRms.toExponential(2)}`);
  console.log(`${gpu.elapsedMilliseconds.toFixed(1)} ms`
    + `\t${(gpu.memory.peakBytes / 2 ** 20).toFixed(1)} MiB peak`);

  const bound = 1e-5;
  const worst = Math.max(pairRms, msaRms);
  if (worst > bound) throw new Error(`relRMS ${worst.toExponential(2)} exceeds ${bound}`);
  return { n, sequences, blocks: count, pairRms, msaRms };
}
