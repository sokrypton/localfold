/*
 * AF2-MULTIMER'S TEMPLATE EMBEDDER.
 *
 * 🔴 IT RUNS WHETHER OR NOT THERE ARE TEMPLATES. Multimer's config has
 * template.enabled true, and the embedding wrapper adds this to the pair
 * unconditionally. Masking every template off does NOT zero the term: the input
 * Linears still contribute their biases, the aatype one-hot of zero is still a
 * one-hot, and the query embedding is still read. Omitting it put the pair track
 * 30% out from the first evoformer block, which is what shattered backbones at
 * six copies and made recycling wander instead of converge.
 *
 * WHAT IT COMPUTES, with the templates masked:
 *
 *     act = sum(biases) + W_aatype_row[0] + W_aatype_col[0]
 *         + W_query . LayerNorm(pair)                     [L, L, 64]
 *     act = templatePairBlock(act) x 2                    multimer's order
 *     act = relu(LayerNorm(act) / templates)
 *     pair += act . W_out                                 [L, L, 128]
 *
 * The distogram, the unit vectors and the backbone mask are all zero for a
 * masked template, so six of the nine input Linears reduce to their biases.
 * That is why this is far smaller than its 35 modules suggest.
 */
import { encodeTemplatePairBlock } from "./block.js";

const TEMPLATE_CHANNELS = 64;
const AATYPE_CLASSES = 22;

/**
 * The constant part of `construct_input`, folded on the CPU.
 *
 * Everything except the query-embedding term is fixed once the template is
 * masked, so it is a single [64] vector added to every pair - computed here
 * rather than as eight more GPU passes over an [L, L, 64] tensor.
 *
 * @param {{pairEmbeddings: {weight: Float32Array, bias: Float32Array}[]}} weights
 * @param {number} aatypeIndex the masked template's residue type; zero
 */
export function templateConstantTerm(weights, aatypeIndex = 0) {
  const constant = new Float32Array(TEMPLATE_CHANNELS);
  weights.pairEmbeddings.forEach(({ bias }, index) => {
    if (index === 8) return;            // the query term is not constant
    for (let c = 0; c < TEMPLATE_CHANNELS; c += 1) constant[c] += bias[c];
  });
  // ...the aatype one-hot of a masked template is one-hot at `aatypeIndex`, not
  // zero, so its row and column embeddings contribute a real weight row each.
  for (const index of [2, 3]) {
    const { weight } = weights.pairEmbeddings[index];
    for (let c = 0; c < TEMPLATE_CHANNELS; c += 1) {
      constant[c] += weight[aatypeIndex * TEMPLATE_CHANNELS + c];
    }
  }
  return constant;
}

/** Does this checkpoint carry a template embedder at all? */
export function hasTemplateEmbedder(weights) {
  return weights !== undefined && Array.isArray(weights.stack) && weights.stack.length > 0;
}

/** The channel counts and block count the embedder runs at. */
export function templateShape(weights, length) {
  return {
    length,
    cZ: TEMPLATE_CHANNELS,
    templateOrder: "multimer",
    blocks: weights.stack.length,
    triangleHidden: weights.stack[0].triangleMultiplicationOutgoing.linearAPBias.length,
    aatypeClasses: AATYPE_CLASSES,
  };
}

const COMMON = `
struct Parameters {
  length: u32, pair_channels: u32, template_channels: u32, templates: u32,
  layer_norm_epsilon: f32, padding0: u32, padding1: u32, padding2: u32,
};
const GRID_WIDTH: u32 = 32768u;
`;

/**
 * act = constant + W_query . LayerNorm(pair), one invocation per output channel.
 *
 * The layer norm is over the pair's 128 channels and is recomputed per output
 * channel rather than staged: at 64 outputs against a 128-wide norm that is
 * cheaper than a second pass and a second buffer, and this runs once per
 * recycle rather than once per block.
 */
export const TEMPLATE_INPUT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> constant: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.template_channels) { return; }
  let channel = index % p.template_channels;
  let entry = index / p.template_channels;
  let base = entry * p.pair_channels;
  var mean = 0.0;
  for (var c = 0u; c < p.pair_channels; c += 1u) { mean += pair[base + c]; }
  mean = mean / f32(p.pair_channels);
  var variance = 0.0;
  for (var c = 0u; c < p.pair_channels; c += 1u) {
    let centred = pair[base + c] - mean;
    variance += centred * centred;
  }
  let inverse = inverseSqrt(variance / f32(p.pair_channels) + p.layer_norm_epsilon);
  // weights: [scale (128), offset (128), queryWeight (128 x 64), queryBias (64)]
  let scale = 0u;
  let offset = p.pair_channels;
  let query = 2u * p.pair_channels;
  let query_bias = query + p.pair_channels * p.template_channels;
  var value = constant[channel] + weights[query_bias + channel];
  for (var c = 0u; c < p.pair_channels; c += 1u) {
    let normalized = (pair[base + c] - mean) * inverse * weights[scale + c] + weights[offset + c];
    value += normalized * weights[query + c * p.template_channels + channel];
  }
  output[index] = value;
}`;

/**
 * pair += relu(LayerNorm(act) / templates) . W_out, the embedder's last step.
 *
 * The mean over templates, the relu and the output projection collapse into one
 * pass because nothing between them is shared with another consumer.
 */
export const TEMPLATE_OUTPUT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> pair: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.pair_channels) { return; }
  let channel = index % p.pair_channels;
  let entry = index / p.pair_channels;
  let base = entry * p.template_channels;
  var mean = 0.0;
  for (var c = 0u; c < p.template_channels; c += 1u) { mean += act[base + c]; }
  mean = mean / f32(p.template_channels);
  var variance = 0.0;
  for (var c = 0u; c < p.template_channels; c += 1u) {
    let centred = act[base + c] - mean;
    variance += centred * centred;
  }
  let inverse = inverseSqrt(variance / f32(p.template_channels) + p.layer_norm_epsilon);
  // weights: [scale (64), offset (64), outWeight (64 x 128), outBias (128)]
  let scale = 0u;
  let offset = p.template_channels;
  let out_weight = 2u * p.template_channels;
  let out_bias = out_weight + p.template_channels * p.pair_channels;
  var value = weights[out_bias + channel];
  for (var c = 0u; c < p.template_channels; c += 1u) {
    let normalized = (act[base + c] - mean) * inverse * weights[scale + c] + weights[offset + c];
    let activated = max(normalized / f32(p.templates), 0.0);
    value += activated * weights[out_weight + c * p.pair_channels + channel];
  }
  pair[index] = pair[index] + value;
}`;

/** Pack the input shader's weights: norm scale, norm offset, query W, query b. */
function packInput(weights, pairChannels) {
  const query = weights.pairEmbeddings[8];
  const data = new Float32Array(2 * pairChannels + query.weight.length + query.bias.length);
  data.set(weights.queryNormScale, 0);
  data.set(weights.queryNormOffset, pairChannels);
  data.set(query.weight, 2 * pairChannels);
  data.set(query.bias, 2 * pairChannels + query.weight.length);
  return data;
}

/** ...and the output shader's: norm scale, norm offset, output W, output b. */
function packOutput(weights) {
  const channels = weights.outputNormScale.length;
  const data = new Float32Array(2 * channels + weights.outputWeight.length + weights.outputBias.length);
  data.set(weights.outputNormScale, 0);
  data.set(weights.outputNormOffset, channels);
  data.set(weights.outputWeight, 2 * channels);
  data.set(weights.outputBias, 2 * channels + weights.outputWeight.length);
  return data;
}

function parameters(length, pairChannels, templates) {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  [length, pairChannels, TEMPLATE_CHANNELS, templates].forEach(
    (value, index) => view.setUint32(index * 4, value, true),
  );
  view.setFloat32(16, 1e-5, true);
  return new Uint8Array(buffer);
}

/**
 * Add AF2-multimer's template term to the pair, in place.
 *
 * @param {number} templates how many templates were averaged; 1 when masked
 */
export async function encodeTemplateEmbedding(execution, encoder, input, weights, pair, pairMask) {
  const { length, pairChannels } = input;
  const templates = input.templates ?? 1;
  const [inputPipeline, outputPipeline] = await Promise.all([
    execution.pipelines.get("multimer:template:input", TEMPLATE_INPUT_SHADER),
    execution.pipelines.get("multimer:template:output", TEMPLATE_OUTPUT_SHADER),
  ]);
  const uniformBuffer = execution.upload(
    "template.parameters", parameters(length, pairChannels, templates), GPUBufferUsage.UNIFORM,
  );
  const constant = execution.upload("template.constant", templateConstantTerm(weights));
  const inputWeights = execution.upload("template.input-weights", packInput(weights, pairChannels));
  const outputWeights = execution.upload("template.output-weights", packOutput(weights));
  const act = execution.allocate("template.act", length * length * TEMPLATE_CHANNELS);

  let grid = execution.linearGrid(length * length * TEMPLATE_CHANNELS);
  execution.dispatch(encoder, inputPipeline, [pair, constant, inputWeights, uniformBuffer, act],
    grid[0], grid[1], 1, "template.input");

  const shape = templateShape(weights, length);
  for (let block = 0; block < weights.stack.length; block += 1) {
    await encodeTemplatePairBlock(execution, encoder, shape, weights.stack[block], act, pairMask);
  }

  grid = execution.linearGrid(length * length * pairChannels);
  execution.dispatch(encoder, outputPipeline, [act, outputWeights, uniformBuffer, pair],
    grid[0], grid[1], 1, "template.output");
}

export { TEMPLATE_CHANNELS, encodeTemplatePairBlock };
