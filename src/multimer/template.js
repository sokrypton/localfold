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
// 🔴 THE GEOMETRY IS SHARED WITH AF3, NOT COPIED. The two models compute the
// same six features in two atom layouts - see the note at the top of
// template-features.js - and that module is checked bit-exact against AF3's
// own featuriser. A second implementation here would be the one thing in this
// file no oracle covers.
import {
  AF2_ATOM37, GEOMETRY_STRIDE, coverageOf, multichainMaskFor,
  packTemplateGeometry, templateGeometry,
} from "../af3/template-features.js";

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
export function templateConstantTerm(weights) {
  const constant = new Float32Array(TEMPLATE_CHANNELS);
  weights.pairEmbeddings.forEach(({ bias }, index) => {
    if (index === 8) return;            // the query term is not constant
    for (let c = 0; c < TEMPLATE_CHANNELS; c += 1) constant[c] += bias[c];
  });
  return constant;
}

/**
 * The aatype terms, per residue along each axis.
 *
 * 🔴 THESE USED TO BE PART OF THE CONSTANT AND CANNOT BE. A masked template is
 * one residue type repeated, so its row and column embeddings are the SAME
 * weight row at every position and fold into a [64] vector added everywhere -
 * which is what this module did, correctly, while masked was the only case. A
 * real template has a different residue at every position, so the two terms
 * are [L, 64] each and vary along j and along i respectively.
 *
 * 🔴 22 CLASSES, IN RESTYPE ORDER, WHICH IS NOT WHAT hhsearch WRITES. AF2's
 * `fix_templates_aatype` maps HHBLITS order to restype order in the DATA
 * TRANSFORMS, before the model sees it - so a template built from a structure
 * is already in the right order and one read from a search hit is not. The two
 * share every letter, so the difference is a plausible protein rather than an
 * error. See the note in src/af3/template-features.js.
 *
 * @param {object} weights
 * @param {ArrayLike<number>} aatype [length], restype order
 * @param {number} length
 * @returns {{row: Float32Array, column: Float32Array}} [length * 64] each
 */
export function templateAatypeTerms(weights, aatype, length) {
  const row = new Float32Array(length * TEMPLATE_CHANNELS);
  const column = new Float32Array(length * TEMPLATE_CHANNELS);
  for (const [index, into] of [[2, row], [3, column]]) {
    const { weight } = weights.pairEmbeddings[index];
    for (let residue = 0; residue < length; residue += 1) {
      const code = Math.min(Math.max(aatype[residue] ?? 0, 0), AATYPE_CLASSES - 1);
      for (let c = 0; c < TEMPLATE_CHANNELS; c += 1) {
        into[residue * TEMPLATE_CHANNELS + c] = weight[code * TEMPLATE_CHANNELS + c];
      }
    }
  }
  return { row, column };
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
const GEOMETRY_STRIDE: u32 = 6u;
const DGRAM_BINS: u32 = 39u;
`;

/**
 * act = the nine input features, projected and summed, one invocation per
 * output channel.
 *
 * The layer norm is over the pair's 128 channels and is recomputed per output
 * channel rather than staged: at 64 outputs against a 128-wide norm that is
 * cheaper than a second pass and a second buffer, and this runs once per
 * recycle rather than once per block.
 *
 * 🔴 THE aatype TERMS ARE PER RESIDUE AND USED TO BE PART OF `constant`. That
 * was right while a masked template was the only case - one residue type
 * repeated is the same weight row everywhere - and is wrong the moment a
 * template has real residues in it. They are two [L, 64] buffers now, read at
 * j and at i.
 *
 * 🔴 AND THE GEOMETRY IS ZEROES FOR A MASKED TEMPLATE, so there is one shader
 * rather than two. The distogram bin is stored PLUS ONE, so 0 means "no bin" -
 * which is what both a masked template and a pair closer than 3.25 A have.
 */
export const TEMPLATE_INPUT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> constant: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> aatype_row: array<f32>;
@group(0) @binding(6) var<storage, read> aatype_column: array<f32>;
@group(0) @binding(7) var<storage, read> geometry: array<f32>;
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

  // Features 2 and 3: the template's aatype, along j and along i.
  let i = entry / p.length;
  let j = entry % p.length;
  value += aatype_row[j * p.template_channels + channel];
  value += aatype_column[i * p.template_channels + channel];

  // Features 0, 1, 4, 5, 6 and 7: the geometry, packed six floats a pair.
  // The five scalar ones are a per-channel weight each; the distogram's is
  // [39, 64], indexed by the bin.
  let g = entry * GEOMETRY_STRIDE;
  let w0 = query_bias + p.template_channels;
  let w1 = w0 + DGRAM_BINS * p.template_channels;
  let bin = u32(geometry[g]);
  if (bin > 0u) {
    value += weights[w0 + (bin - 1u) * p.template_channels + channel];
  }
  for (var f = 0u; f < 5u; f += 1u) {
    value += geometry[g + 1u + f] * weights[w1 + f * p.template_channels + channel];
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

/**
 * Pack the input shader's weights: norm scale, norm offset, query W, query b,
 * then the six geometry projections.
 *
 * The geometry ones are appended rather than given their own buffer because
 * five of the six are a [64] vector - AF2 builds them with `num_input_dims=0`,
 * so the feature is a SCALAR per pair and the weight a per-channel scale - and
 * a buffer per 256 bytes is five more bindings for no reason. Feature 0's is
 * [39, 64], the distogram's bins.
 */
function packInput(weights, pairChannels) {
  const query = weights.pairEmbeddings[8];
  const geometry = GEOMETRY_FEATURES.map((index) => weights.pairEmbeddings[index].weight);
  const head = 2 * pairChannels + query.weight.length + query.bias.length;
  const data = new Float32Array(
    head + geometry.reduce((total, weight) => total + weight.length, 0));
  data.set(weights.queryNormScale, 0);
  data.set(weights.queryNormOffset, pairChannels);
  data.set(query.weight, 2 * pairChannels);
  data.set(query.bias, 2 * pairChannels + query.weight.length);
  let at = head;
  for (const weight of geometry) { data.set(weight, at); at += weight.length; }
  return data;
}

/**
 * The nine features' indices, and which six are geometry.
 *
 * 🔴 THE ORDER IS THE INDEX IN `template_pair_embedding_{i}` AND IT IS AF3'S
 * ORDER TOO. AF2-multimer's construct_input and AF3's are the same nine
 * features in the same sequence: distogram, its mask, aatype along each axis,
 * three unit-vector components, the backbone mask, the query pair.
 */
const GEOMETRY_FEATURES = [0, 1, 4, 5, 6, 7];

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
 * Which pairs a template is allowed to speak about.
 *
 * 🔴 NO PERMISSIVE DEFAULT, AND THE FIRST VERSION OF THIS HAD ONE.
 * `input.asymId ?? new Int32Array(length)` puts every token in chain 0, which
 * is right for a monomer and silently lets a template speak across a complex's
 * chains - about pairs whose two ends were never in one coordinate frame. AF3
 * has the same shape of bug measured at relRMS 1.09 against a two-chain
 * reference, and it went unnoticed there because every check had ONE chain.
 * A template with no chain ids raises instead.
 *
 * 🔴 SPANNING IS A PROPERTY OF THE SLOT. AF2-multimer masks across chains for
 * the same reason AF3 does - its templates are searched per chain - but when
 * both chains came from ONE file they are in one frame and the cross-chain
 * distances are the interface geometry a binder method wants. See
 * multichainMaskFor in src/af3/template-features.js.
 */
function chainMaskFor(input, length) {
  if (input.multichainMask2d !== undefined) return input.multichainMask2d;
  if (input.asymId === undefined) {
    throw new Error("a template needs `asymId` (or `multichainMask2d`):"
      + " AF2-multimer masks the template's geometry across chains, and"
      + " assuming one chain lets it speak about pairs it has never seen in"
      + " one coordinate frame");
  }
  return multichainMaskFor(input.asymId, length, {
    coverage: coverageOf(input.template, length, AF2_ATOM37),
    spanChains: input.template.spanChains === true,
  });
}

/**
 * Add AF2-multimer's template term to the pair, in place.
 *
 * @param {number} templates how many templates were averaged; 1 when masked
 */
export async function encodeTemplateEmbedding(execution, encoder, input, weights, pair, pairMask,
                                              options = {}) {
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
  const act = execution.allocate("template.act", length * length * TEMPLATE_CHANNELS,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  // 🔴 A MASKED TEMPLATE IS aatype 0 EVERYWHERE AND ZERO GEOMETRY, which is
  // what this module shipped for and is now expressed as data rather than as a
  // separate code path. The alternative - branch on whether a template exists -
  // is two paths where one of them is exercised by every fold and the other by
  // almost none.
  const aatype = input.template?.aatype
    ?? new Int32Array(length).fill(input.maskedAatype ?? 0);
  const terms = templateAatypeTerms(weights, aatype, length);
  const geometry = input.template === undefined
    ? new Float32Array(length * length * GEOMETRY_STRIDE)
    : packTemplateGeometry(
      templateGeometry(input.template, chainMaskFor(input, length), length, AF2_ATOM37),
      length);
  const aatypeRow = execution.upload("template.aatype-row", terms.row);
  const aatypeColumn = execution.upload("template.aatype-column", terms.column);
  const geometryBuffer = execution.upload("template.geometry", geometry);

  let grid = execution.linearGrid(length * length * TEMPLATE_CHANNELS);
  execution.dispatch(encoder, inputPipeline,
    [pair, constant, inputWeights, uniformBuffer, act,
     aatypeRow, aatypeColumn, geometryBuffer],
    grid[0], grid[1], 1, "template.input");

  // 🔴 A SEAM FOR STAGE-BY-STAGE COMPARISON, opt-in and off by every fold's
  // path. The GPU and the numpy reference for this module disagreed by relRMS
  // 1.1e-2 at the OUTPUT, which says only that they disagree - and this module
  // is an input term plus two pair blocks plus a projection, so "which of the
  // four" is the whole question. Reading `act` after each is how it gets an
  // answer instead of a bisection by deletion.
  const stages = options.captureStages === true ? [] : undefined;
  stages?.push(execution.createReadback("template.stage0", act, encoder));

  const shape = templateShape(weights, length);
  for (let block = 0; block < weights.stack.length; block += 1) {
    await encodeTemplatePairBlock(execution, encoder, shape, weights.stack[block], act, pairMask);
    stages?.push(execution.createReadback(`template.stage${block + 1}`, act, encoder));
  }

  grid = execution.linearGrid(length * length * pairChannels);
  execution.dispatch(encoder, outputPipeline, [act, outputWeights, uniformBuffer, pair],
    grid[0], grid[1], 1, "template.output");
  return { stages };
}

export { TEMPLATE_CHANNELS, encodeTemplatePairBlock };
