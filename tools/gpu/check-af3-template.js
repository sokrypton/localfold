/**
 * AF3's template embedder (empty-template path): GPU against
 * src/af3/template-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-template.js
 *
 * The check also reports the output's std against the input pair's, because the
 * number is the argument for the module existing at all: with four EMPTY slots
 * this is not a small residual correction.
 */
import { templateEmbedding } from "../../src/af3/template-reference.js";
import { Af3TemplateEmbedderGpu } from "../../src/af3/template-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const ROOT = "diffuser/evoformer/template_embedding";
const SINGLE = `${ROOT}/single_template_embedding`;
const STACK = `${SINGLE}/__layer_stack_no_per_layer/template_embedding_iteration`;
const DIALECT = { swapTransposedBias: false };
const QUERY_CHANNELS = 128;

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

function standardDeviation(values) {
  let total = 0;
  for (const value of values) total += value;
  const mean = total / values.length;
  let squares = 0;
  for (const value of values) squares += (value - mean) ** 2;
  return Math.sqrt(squares / values.length);
}

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "32"));
  const templates = Number(option(args, "templates", "4"));
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
      heads: 4, dimension: 16,
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

  const T = (name) => store.tensor(name);
  const weights = {
    queryChannels: QUERY_CHANNELS,
    queryEmbeddingNormScale: await T(`${SINGLE}/query_embedding_norm/scale`),
    queryEmbeddingNormOffset: await T(`${SINGLE}/query_embedding_norm/offset`),
    templatePairEmbedding8: await T(`${SINGLE}/template_pair_embedding_8/weights`),
    templatePairEmbedding2: await T(`${SINGLE}/template_pair_embedding_2/weights`),
    templatePairEmbedding3: await T(`${SINGLE}/template_pair_embedding_3/weights`),
    outputLayerNormScale: await T(`${SINGLE}/output_layer_norm/scale`),
    outputLayerNormOffset: await T(`${SINGLE}/output_layer_norm/offset`),
    outputLinear: await T(`${ROOT}/output_linear/weights`),
    blocks: [await blockWeights(0), await blockWeights(1)],
  };

  const sequence = new Float32Array(tokens);
  for (let i = 0; i < tokens; i += 1) sequence[i] = i < Math.ceil(tokens * 0.8) ? 1 : 0;
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = sequence[i] * sequence[j];
  }
  const pair = deterministic(tokens * tokens * QUERY_CHANNELS, 555 + tokens);
  const input = { pair, pairMask, tokens, templates, templateOccupied: false };

  const expected = templateEmbedding(input, weights, DIALECT);
  const gpu = await new Af3TemplateEmbedderGpu(device).run(input, weights, DIALECT);
  const relRms = relativeRms(gpu.output, expected);

  console.log(`template\ttokens=${tokens} slots=${templates}`
    + `\trelRMS ${relRms.toExponential(2)}`
    + `\t${gpu.elapsedMilliseconds.toFixed(1)} ms`);
  // The argument for the module existing: this is not a small correction.
  console.log(`output std ${standardDeviation(gpu.output).toFixed(2)}`
    + ` against an input pair std of ${standardDeviation(pair).toFixed(2)}`
    + ` - with ${templates} EMPTY slots`);

  const bound = 1e-5;
  if (relRms > bound) throw new Error(`relRMS ${relRms.toExponential(2)} exceeds ${bound}`);
  return { tokens, templates, relRms, outputStd: standardDeviation(gpu.output) };
}
