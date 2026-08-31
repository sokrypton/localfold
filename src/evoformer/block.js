import {
  ATTENTION_NORMALIZE_SHADER,
  ATTENTION_OUTPUT_SHADER,
  ATTENTION_OUTPUT_RESIDUAL_SHADER,
  ATTENTION_PAIR_BIAS_SHADER,
  ATTENTION_PAIR_BIAS_CHAIN_MASKED_SHADER,
  ATTENTION_PROJECT_SHADER,
  createAttentionNormParameters,
  createAttentionParameters,
  packAttentionWeights,
  selectAttentionFlashKernel,

} from "./attention.js";
import {
  createOuterProductMeanParameters,
  OUTER_PRODUCT_MEAN_TILE_INTERMEDIATE_SHADER,
  OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER,
  OUTER_PRODUCT_MEAN_FINALIZE_SHADER,
  OUTER_PRODUCT_MEAN_CONTRACT_SHADER,
  OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_SHADER,
  outerProductMeanTileCapacity,
  OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_RESIDUAL_SHADER,
  OUTER_PRODUCT_MEAN_NORMALIZE_SHADER,
  OUTER_PRODUCT_MEAN_PROJECT_SHADER,
  OUTER_PRODUCT_MEAN_MARGINALS_SHADER,
  OUTER_PRODUCT_MEAN_CONTRACT_COV_MASKED_SHADER,
  OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_COV_MASKED_SHADER,
  OUTER_PRODUCT_MEAN_TILE_MARGINAL_SHADER,
  packOuterProductMeanWeights,
  useOuterFirstContraction,

} from "./outer-product-mean.js";
import {
  createTransitionNormalizeParameters,
  createTransitionShaders,
  packTransitionWeights,
  transitionChunkRows,
  TRANSITION_TILE_COLUMNS,
  TRANSITION_TILE_ROWS,

} from "./transition.js";
import { WebGpuExecution } from "../runtime/execution.js";
import { createTriangleShaders } from "../triangle/shaders.js";

import { packWeights as packTriangleWeights } from "../triangle/weights.js";

const GLOBAL_ATTENTION_COMMON = `
struct Parameters {
  length: u32, sequences: u32, channels: u32, heads: u32, head_dim: u32,
  query_weight: u32, key_weight: u32, value_weight: u32, gating_weight: u32,
  gating_bias: u32, output_weight: u32, output_bias: u32,
};
`;

const GLOBAL_ATTENTION_KV_SHADER = `${GLOBAL_ATTENTION_COMMON}
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> keys: array<f32>;
@group(0) @binding(4) var<storage, read_write> values: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= p.length * p.sequences * p.head_dim) { return; }
  let d = index % p.head_dim; let row = index / p.head_dim;
  var key = 0.0; var value = 0.0;
  for (var c = 0u; c < p.channels; c += 1u) {
    let x = normalized[row * p.channels + c];
    key += x * weights[p.key_weight + c * p.head_dim + d];
    value += x * weights[p.value_weight + c * p.head_dim + d];
  }
  keys[index] = key; values[index] = value;
}`;

const GLOBAL_ATTENTION_QUERY_SHADER = `${GLOBAL_ATTENTION_COMMON}
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> query: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= p.length * p.heads * p.head_dim) { return; }
  let d = index % p.head_dim; let head = (index / p.head_dim) % p.heads; let column = index / (p.head_dim * p.heads);
  var denominator = 1e-10; var result = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) { denominator += mask[sequence * p.length + column]; }
  for (var c = 0u; c < p.channels; c += 1u) {
    var average = 0.0;
    for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
      average += normalized[(column * p.sequences + sequence) * p.channels + c]
        * mask[sequence * p.length + column];
    }
    result += average / denominator * weights[p.query_weight + (c * p.heads + head) * p.head_dim + d];
  }
  query[index] = result * inverseSqrt(f32(p.head_dim));
}`;

const GLOBAL_ATTENTION_FLASH_SHADER = `${GLOBAL_ATTENTION_COMMON}
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x; let head = id.y;
  if (column >= p.length || head >= p.heads) { return; }
  var maximum = -1e30; var denominator = 0.0;
  var accumulated: array<f32, 32>;
  for (var d = 0u; d < p.head_dim; d += 1u) { accumulated[d] = 0.0; }
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    var logit = 0.0;
    for (var d = 0u; d < p.head_dim; d += 1u) {
      logit += query[(column * p.heads + head) * p.head_dim + d]
        * keys[(column * p.sequences + sequence) * p.head_dim + d];
    }
    if (mask[sequence * p.length + column] == 0.0) { logit = -1e9; }
    let next_maximum = max(maximum, logit);
    let previous_scale = exp(maximum - next_maximum);
    let weight = exp(logit - next_maximum);
    denominator = denominator * previous_scale + weight;
    for (var d = 0u; d < p.head_dim; d += 1u) {
      accumulated[d] = accumulated[d] * previous_scale
        + weight * values[(column * p.sequences + sequence) * p.head_dim + d];
    }
    maximum = next_maximum;
  }
  for (var d = 0u; d < p.head_dim; d += 1u) {
    output[(column * p.heads + head) * p.head_dim + d] = accumulated[d] / denominator;
  }
}`;

const GLOBAL_ATTENTION_OUTPUT_SHADER = `${GLOBAL_ATTENTION_COMMON}
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> attended: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.sequences * p.length * p.channels) { return; }
  let c_out = index % p.channels; let row = index / p.channels;
  let column = row % p.length; let sequence = row / p.length;
  let normalized_row = column * p.sequences + sequence;
  var result = weights[p.output_bias + c_out];
  for (var head = 0u; head < p.heads; head += 1u) {
    for (var d = 0u; d < p.head_dim; d += 1u) {
      var gate = weights[p.gating_bias + head * p.head_dim + d];
      for (var c = 0u; c < p.channels; c += 1u) {
        gate += normalized[normalized_row * p.channels + c]
          * weights[p.gating_weight + (c * p.heads + head) * p.head_dim + d];
      }
      let gated = attended[(column * p.heads + head) * p.head_dim + d] / (1.0 + exp(-gate));
      result += gated * weights[p.output_weight + (head * p.head_dim + d) * p.channels + c_out];
    }
  }
  output[index] = result;
}`;
const GLOBAL_ATTENTION_OUTPUT_RESIDUAL_SHADER = GLOBAL_ATTENTION_OUTPUT_SHADER.replace(
  "output[index] = result;", "output[index] += result;",
);

function uniform(execution, label, data) {
  return execution.upload(label, data, GPUBufferUsage.UNIFORM);
}

async function encodeTransition(
  execution,
  encoder,
  source,
  rows,
  channels,
  weightsValue,
  label,
  residualTarget,
) {
  const hiddenChannels = weightsValue.firstBias.length;
  const descriptor = {
    activations: new Float32Array(0), rows, channels, hiddenChannels, weights: weightsValue,
  };
  const packed = packTransitionWeights(descriptor);
  const shaders = createTransitionShaders(descriptor, packed.offsets);
  const [normalize, linear, linearResidual] = await Promise.all([
    execution.pipelines.get("block:transition:normalize", shaders[0]),
    execution.pipelines.get("block:transition:linear", shaders[1]),
    execution.pipelines.get("block:transition:linear-residual", shaders[2]),
  ]);
  const weights = execution.upload(`${label}.weights`, packed.data);
  const output = residualTarget ?? execution.allocate(`${label}.output`, rows * channels);
  // ...HOW MANY ROWS FIT IN ONE BINDING. Returns `rows` whenever everything
  // already binds, which is every short input, so the single-dispatch path
  // below is byte for byte the one that was there before chunking existed.
  const chunkRows = transitionChunkRows(
    rows, channels, hiddenChannels, execution.transitionBufferLimit,
    execution.device.limits.minStorageBufferOffsetAlignment,
  );

  if (chunkRows === rows) {
    const normalizeParams = uniform(execution, `${label}.normalize-parameters`,
      createTransitionNormalizeParameters(descriptor, packed.offsets));
    const firstParams = uniform(execution, `${label}.first-parameters`, new Uint32Array([
      rows, channels, hiddenChannels, packed.offsets[2], packed.offsets[3], 1, 0, 0,
    ]));
    const secondParams = uniform(execution, `${label}.second-parameters`, new Uint32Array([
      rows, hiddenChannels, channels, packed.offsets[4], packed.offsets[5], 0, 0, 0,
    ]));
    const normalized = execution.allocate(`${label}.normalized`, rows * channels);
    const hidden = execution.allocate(`${label}.hidden`, rows * hiddenChannels);
    const transitionNormGrid = execution.rowGrid(rows);
    execution.dispatch(encoder, normalize, [source, weights, normalizeParams, normalized],
      transitionNormGrid[0], transitionNormGrid[1], 1, `${label}.normalize`);
    execution.dispatch(encoder, linear, [normalized, weights, firstParams, hidden],
      Math.ceil(hiddenChannels / TRANSITION_TILE_COLUMNS), Math.ceil(rows / TRANSITION_TILE_ROWS), 1,
      `${label}.first`);
    execution.dispatch(encoder, residualTarget === undefined ? linear : linearResidual,
      [hidden, weights, secondParams, output],
      Math.ceil(channels / TRANSITION_TILE_COLUMNS), Math.ceil(rows / TRANSITION_TILE_ROWS), 1,
      `${label}.second`);
    return output;
  }

  // 🔴 ONE SCRATCH PAIR FOR THE WHOLE LOOP, viewed per chunk rather than
  // reallocated. The scratch is sized for a chunk, and what changes each time
  // round is which rows of the SOURCE and the OUTPUT are bound - so the big
  // tensors are never bound whole and never need to be bindable whole.
  const normalized = execution.allocate(`${label}.normalized-chunk`, chunkRows * channels);
  const hidden = execution.allocate(`${label}.hidden-chunk`, chunkRows * hiddenChannels);
  for (let rowOffset = 0; rowOffset < rows; rowOffset += chunkRows) {
    const count = Math.min(chunkRows, rows - rowOffset);
    // ...the row count in the uniforms is the CHUNK's, not the tensor's: the
    // shaders bound-check against it, and the last chunk is usually short.
    const chunkDescriptor = { ...descriptor, rows: count };
    const normalizeParams = uniform(execution, `${label}.normalize-parameters-${rowOffset}`,
      createTransitionNormalizeParameters(chunkDescriptor, packed.offsets));
    const firstParams = uniform(execution, `${label}.first-parameters-${rowOffset}`, new Uint32Array([
      count, channels, hiddenChannels, packed.offsets[2], packed.offsets[3], 1, 0, 0,
    ]));
    const secondParams = uniform(execution, `${label}.second-parameters-${rowOffset}`, new Uint32Array([
      count, hiddenChannels, channels, packed.offsets[4], packed.offsets[5], 0, 0, 0,
    ]));
    const sourceChunk = execution.view(source, rowOffset * channels, count * channels);
    const outputChunk = execution.view(output, rowOffset * channels, count * channels);
    const normalizedChunk = execution.view(normalized, 0, count * channels);
    const hiddenChunk = execution.view(hidden, 0, count * hiddenChannels);
    const chunkNormGrid = execution.rowGrid(count);
    execution.dispatch(encoder, normalize, [sourceChunk, weights, normalizeParams, normalizedChunk],
      chunkNormGrid[0], chunkNormGrid[1], 1, `${label}.normalize-${rowOffset}`);
    execution.dispatch(encoder, linear, [normalizedChunk, weights, firstParams, hiddenChunk],
      Math.ceil(hiddenChannels / TRANSITION_TILE_COLUMNS), Math.ceil(count / TRANSITION_TILE_ROWS), 1,
      `${label}.first-${rowOffset}`);
    execution.dispatch(encoder, residualTarget === undefined ? linear : linearResidual,
      [hiddenChunk, weights, secondParams, outputChunk],
      Math.ceil(channels / TRANSITION_TILE_COLUMNS), Math.ceil(count / TRANSITION_TILE_ROWS), 1,
      `${label}.second-${rowOffset}`);
  }
  return output;
}

async function encodeAttention(
  execution,
  encoder,
  options,
) {
  const descriptor = {
    activations: new Float32Array(0), mask: new Float32Array(0), batch: options.batch,
    queryLength: options.queries, channels: options.channels, heads: options.heads,
    transpose: options.transpose, weights: options.weights,
    ...(options.pairBias === undefined ? {} : { pairBias: options.pairBias }),
  };
  const packed = packAttentionWeights(descriptor);
  const flashKernel = selectAttentionFlashKernel(execution.device, options.channels / options.heads);
  const [normalize, project, pairProject, flash, outputProject] = await Promise.all([
    execution.pipelines.get("block:attention:normalize", ATTENTION_NORMALIZE_SHADER),
    execution.pipelines.get("block:attention:project", ATTENTION_PROJECT_SHADER),
    execution.pipelines.get(
      options.chainMask === undefined ? "block:attention:pair-bias" : "block:attention:pair-bias-chain-masked",
      options.chainMask === undefined
        ? ATTENTION_PAIR_BIAS_SHADER : ATTENTION_PAIR_BIAS_CHAIN_MASKED_SHADER,
    ),
    execution.pipelines.get(`block:${flashKernel.cacheKey}`, flashKernel.shader),
    execution.pipelines.get(
      options.residualTarget === undefined ? "block:attention:output" : "block:attention:output-residual",
      options.residualTarget === undefined ? ATTENTION_OUTPUT_SHADER : ATTENTION_OUTPUT_RESIDUAL_SHADER,
    ),
  ]);
  const rows = options.batch * options.queries;
  const elements = rows * options.channels;
  const weights = execution.upload(`${options.label}.weights`, packed.data);
  const params = uniform(execution, `${options.label}.parameters`, createAttentionParameters(descriptor, packed.offsets));
  const normParams = uniform(execution, `${options.label}.norm-parameters`, createAttentionNormParameters(
    rows, options.channels, packed.offsets[0], packed.offsets[1], options.transpose,
    options.batch, options.queries, 1e-5,
  ));
  const normalized = execution.allocate(`${options.label}.normalized`, elements);
  const query = execution.allocate(`${options.label}.query`, elements);
  const key = execution.allocate(`${options.label}.key`, elements);
  const value = execution.allocate(`${options.label}.value`, elements);
  const gate = execution.allocate(`${options.label}.gate`, elements);
  const weighted = execution.allocate(`${options.label}.weighted`, elements);
  const output = options.residualTarget ?? execution.allocate(`${options.label}.output`, elements);
  const attentionNormGrid = execution.rowGrid(rows);
  execution.dispatch(encoder, normalize, [options.source, weights, normParams, normalized],
    attentionNormGrid[0], attentionNormGrid[1], 1,
    `${options.label}.normalize`);

  let normalizedPair = normalized;
  if (options.pairBias?.source === "separate") {
    if (options.pairSource === undefined) throw new Error("separate attention pair bias requires a GPU source");
    normalizedPair = execution.allocate(
      `${options.label}.pair-normalized`, options.queries * options.queries * options.pairBias.channels,
    );
    const pairNormParams = uniform(execution, `${options.label}.pair-norm-parameters`,
      createAttentionNormParameters(
        options.queries * options.queries, options.pairBias.channels, packed.offsets[9], packed.offsets[10],
        false, 1, options.queries * options.queries, 1e-5,
      ));
    const pairNormGrid = execution.rowGrid(options.queries * options.queries);
    execution.dispatch(encoder, normalize, [options.pairSource, weights, pairNormParams, normalizedPair],
      pairNormGrid[0], pairNormGrid[1], 1, `${options.label}.pair-normalize`);
  }
  const pairBiasElements = options.pairBias === undefined ? 1 : options.heads * options.queries * options.queries;
  const pairBias = execution.allocate(`${options.label}.pair-bias`, pairBiasElements);
  if (options.pairBias !== undefined) {
    const grid = execution.linearGrid(pairBiasElements);
    execution.dispatch(encoder, pairProject,
      options.chainMask === undefined
        ? [normalizedPair, weights, params, pairBias]
        : [normalizedPair, weights, params, pairBias, options.chainMask],
      grid[0], grid[1], 1, `${options.label}.pair-bias`);
  }
  execution.dispatch(encoder, project, [normalized, weights, params, query, key, value, gate],
    Math.ceil(options.channels / 16), Math.ceil(rows / 16), 1,
    `${options.label}.project`);
  execution.dispatch(encoder, flash, [query, key, value, gate, options.mask, pairBias, params, weighted],
    Math.ceil(options.queries / flashKernel.queryTile),
    options.batch, options.heads, `${options.label}.flash`);
  execution.dispatch(encoder, outputProject, [weighted, weights, params, output],
    Math.ceil(options.channels / 32), Math.ceil(rows / 16), 1,
    `${options.label}.output`);
  return output;
}

async function encodeGlobalAttention(
  execution,
  encoder,
  source,
  mask,
  shape,
  weightsValue,
  label,
  residualTarget,
) {
  const w = weightsValue;
  const tensors = [w.queryNormScale, w.queryNormOffset, w.queryWeight, w.keyWeight, w.valueWeight,
    w.gatingWeight, w.gatingBias, w.outputWeight, w.outputBias];
  const offsets = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const packed = new Float32Array(size);
  tensors.forEach((tensor, index) => packed.set(tensor, offsets[index]));
  const headDim = w.gatingBias.length / w.heads;
  const params = new Uint32Array([
    shape.length, shape.sequences, shape.cM, w.heads, headDim,
    offsets[2], offsets[3], offsets[4], offsets[5], offsets[6], offsets[7], offsets[8],
  ]);
  const [normalize, kvPipeline, queryPipeline, flashPipeline, outputPipeline] = await Promise.all([
    execution.pipelines.get("block:global-attention:normalize", ATTENTION_NORMALIZE_SHADER),
    execution.pipelines.get("block:global-attention:kv", GLOBAL_ATTENTION_KV_SHADER),
    execution.pipelines.get("block:global-attention:query", GLOBAL_ATTENTION_QUERY_SHADER),
    execution.pipelines.get("block:global-attention:flash", GLOBAL_ATTENTION_FLASH_SHADER),
    execution.pipelines.get(
      residualTarget === undefined ? "block:global-attention:output" : "block:global-attention:output-residual",
      residualTarget === undefined ? GLOBAL_ATTENTION_OUTPUT_SHADER : GLOBAL_ATTENTION_OUTPUT_RESIDUAL_SHADER,
    ),
  ]);
  const weights = execution.upload(`${label}.weights`, packed);
  const parameters = uniform(execution, `${label}.parameters`, params);
  const normParameters = uniform(execution, `${label}.norm-parameters`, createAttentionNormParameters(
    shape.length * shape.sequences, shape.cM, offsets[0], offsets[1], true,
    shape.length, shape.sequences, 1e-5,
  ));
  const normalized = execution.allocate(`${label}.normalized`, shape.length * shape.sequences * shape.cM);
  const keys = execution.allocate(`${label}.keys`, shape.length * shape.sequences * headDim);
  const values = execution.allocate(`${label}.values`, shape.length * shape.sequences * headDim);
  const query = execution.allocate(`${label}.query`, shape.length * w.heads * headDim);
  const attended = execution.allocate(`${label}.attended`, shape.length * w.heads * headDim);
  const output = residualTarget ?? execution.allocate(`${label}.output`, shape.sequences * shape.length * shape.cM);
  const globalNormGrid = execution.rowGrid(shape.length * shape.sequences);
  execution.dispatch(encoder, normalize, [source, weights, normParameters, normalized],
    globalNormGrid[0], globalNormGrid[1], 1, `${label}.normalize`);
  let grid = execution.linearGrid(shape.length * shape.sequences * headDim);
  execution.dispatch(encoder, kvPipeline, [normalized, weights, parameters, keys, values],
    grid[0], grid[1], 1, `${label}.kv`);
  grid = execution.linearGrid(shape.length * w.heads * headDim);
  execution.dispatch(encoder, queryPipeline, [normalized, mask, weights, parameters, query],
    grid[0], grid[1], 1, `${label}.query`);
  execution.dispatch(encoder, flashPipeline, [query, keys, values, mask, parameters, attended],
    shape.length, w.heads, 1, `${label}.flash`);
  grid = execution.linearGrid(shape.sequences * shape.length * shape.cM);
  execution.dispatch(encoder, outputPipeline, [normalized, attended, weights, parameters, output],
    grid[0], grid[1], 1, `${label}.output`);
  return output;
}

async function encodeOuterProductMean(
  execution,
  encoder,
  msa,
  msaMask,
  input,
  weightsValue,
  residualTarget,
  covMask,
) {
  const descriptor = {
    activations: new Float32Array(0), mask: new Float32Array(0), sequences: input.sequences,
    length: input.length, cM: input.cM, cOuter: input.cOuter, cZ: input.cZ,
    weights: weightsValue,
  };
  const packed = packOuterProductMeanWeights(descriptor);
  const outerFirst = useOuterFirstContraction(descriptor);
  const [normalize, project, intermediatePipeline, accumulatePipeline, finalizePipeline,
    contractPipeline, projectOutputPipeline] = await Promise.all([
    execution.pipelines.get("block:opm:normalize", OUTER_PRODUCT_MEAN_NORMALIZE_SHADER),
    execution.pipelines.get("block:opm:project", OUTER_PRODUCT_MEAN_PROJECT_SHADER),
    execution.pipelines.get("block:opm:tile-intermediate", OUTER_PRODUCT_MEAN_TILE_INTERMEDIATE_SHADER),
    execution.pipelines.get(
      covMask === undefined ? "block:opm:tile-accumulate" : "block:opm:tile-accumulate-cov-masked",
      covMask === undefined
        ? OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER
        : OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_COV_MASKED_SHADER,
    ),
    execution.pipelines.get("block:opm:finalize", OUTER_PRODUCT_MEAN_FINALIZE_SHADER),
    execution.pipelines.get(
      covMask === undefined ? "block:opm:contract" : "block:opm:contract-cov-masked",
      covMask === undefined
        ? OUTER_PRODUCT_MEAN_CONTRACT_SHADER : OUTER_PRODUCT_MEAN_CONTRACT_COV_MASKED_SHADER,
    ),
    execution.pipelines.get(
      outerFirst && residualTarget !== undefined
        ? "block:opm:project-output-residual" : "block:opm:project-output",
      outerFirst && residualTarget !== undefined
        ? OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_RESIDUAL_SHADER : OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_SHADER,
    ),
  ]);
  const [marginalsPipeline, tileMarginalPipeline] = covMask === undefined ? [] : await Promise.all([
    execution.pipelines.get("block:opm:marginals", OUTER_PRODUCT_MEAN_MARGINALS_SHADER),
    execution.pipelines.get("block:opm:tile-marginal", OUTER_PRODUCT_MEAN_TILE_MARGINAL_SHADER),
  ]);
  const rows = input.sequences * input.length;
  const pairElements = input.length * input.length * input.cZ;
  const weights = execution.upload("opm.weights", packed.data);
  const params = uniform(execution, "opm.parameters", createOuterProductMeanParameters(descriptor, packed.offsets));
  const normalized = execution.allocate("opm.normalized", rows * input.cM);
  const left = execution.allocate("opm.left", rows * input.cOuter);
  const right = execution.allocate("opm.right", rows * input.cOuter);
  const tileCapacity = outerProductMeanTileCapacity(
    input, execution.device.limits.maxStorageBufferBindingSize);
  const intermediateElements = outerFirst
    ? input.length * input.length * input.cOuter * input.cOuter
    : tileCapacity * input.length * input.cOuter * input.cZ;
  const intermediate = execution.allocate("opm.intermediate", intermediateElements);
  const output = outerFirst && residualTarget !== undefined ? residualTarget
    : execution.allocate("opm.output", pairElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const opmNormGrid = execution.rowGrid(rows);
  execution.dispatch(encoder, normalize, [msa, weights, params, normalized],
    opmNormGrid[0], opmNormGrid[1], 1, "opm.normalize");
  let grid = execution.linearGrid(rows * input.cOuter);
  execution.dispatch(encoder, project, [normalized, msaMask, weights, params, left, right], grid[0], grid[1], 1,
    "opm.project");
  let leftSum;
  let rightMean;
  if (covMask !== undefined) {
    leftSum = execution.allocate("opm.left-sum", input.length * input.cOuter);
    rightMean = execution.allocate("opm.right-mean", input.length * input.cOuter);
    const marginalGrid = execution.linearGrid(input.length * input.cOuter);
    execution.dispatch(encoder, marginalsPipeline, [left, right, msaMask, params, leftSum, rightMean],
      marginalGrid[0], marginalGrid[1], 1, "opm.marginals");
  }
  const outputGrid = execution.linearGrid(pairElements);
  if (outerFirst) {
    grid = execution.linearGrid(intermediateElements);
    execution.dispatch(encoder, contractPipeline,
      covMask === undefined
        ? [left, right, params, intermediate]
        : [left, right, params, intermediate, covMask, leftSum, rightMean],
      grid[0], grid[1], 1, "opm.contract");
    execution.dispatch(encoder, projectOutputPipeline, [intermediate, msaMask, weights, params, output],
      outputGrid[0], outputGrid[1], 1, "opm.project-output");
  } else {
    execution.endComputePass(encoder);
    encoder.clearBuffer(output.allocation.buffer);
    for (let offset = 0; offset < input.sequences; offset += tileCapacity) {
      const count = Math.min(tileCapacity, input.sequences - offset);
      const tileParams = uniform(execution, `opm.tile-${offset}`, new Uint32Array([offset, count, 0, 0]));
      grid = execution.linearGrid(count * input.length * input.cOuter * input.cZ);
      execution.dispatch(encoder, intermediatePipeline, [left, weights, params, tileParams, intermediate],
        grid[0], grid[1], 1, `opm.intermediate-${offset}`);
      execution.dispatch(encoder, accumulatePipeline,
        covMask === undefined
          ? [right, intermediate, params, tileParams, output]
          : [right, intermediate, params, tileParams, output, covMask],
        outputGrid[0], outputGrid[1], 1, `opm.accumulate-${offset}`);
    }
    if (covMask !== undefined) {
      execution.dispatch(encoder, tileMarginalPipeline,
        [leftSum, rightMean, covMask, weights, params, output],
        outputGrid[0], outputGrid[1], 1, "opm.tile-marginal");
    }
    execution.dispatch(encoder, finalizePipeline, [msaMask, weights, params, output],
      outputGrid[0], outputGrid[1], 1, "opm.finalize");
  }
  return output;
}

async function encodeTriangleMultiplication(
  execution,
  encoder,
  pair,
  pairMask,
  input,
  weightsValue,
  direction,
  residualTarget,
) {
  const shape = { length: input.length, cZ: input.cZ, cHidden: input.triangleHidden };
  const packed = packTriangleWeights(weightsValue, "f32");
  const shaders = createTriangleShaders(shape, "f32", packed.offsets, 1e-5, direction);
  const pipelineKey = `block:triangle:${direction}:${input.length}:${input.cZ}:${input.triangleHidden}`;
  const [normalizeInput, projectAB, contract, normalizeHidden, projectOutput] = await Promise.all([
    execution.pipelines.get(`${pipelineKey}:normalize-input`, shaders.normalizeInput),
    execution.pipelines.get(`${pipelineKey}:project-ab`, shaders.projectAB),
    execution.pipelines.get(`${pipelineKey}:contract`, shaders.contract),
    execution.pipelines.get(`${pipelineKey}:normalize-hidden`, shaders.normalizeHidden),
    execution.pipelines.get(
      `${pipelineKey}:project-output${residualTarget === undefined ? "" : "-residual"}`,
      residualTarget === undefined ? shaders.projectOutput : shaders.projectOutput.replace(
        "output[index] = projected * logistic(gate);", "output[index] += projected * logistic(gate);",
      ),
    ),
  ]);
  const pairs = input.length * input.length;
  const weights = execution.upload(`triangle.${direction}.weights`, packed.data);
  const normalized = execution.allocate(`triangle.${direction}.normalized`, pairs * input.cZ);
  const a = execution.allocate(`triangle.${direction}.a`, pairs * input.triangleHidden);
  const b = execution.allocate(`triangle.${direction}.b`, pairs * input.triangleHidden);
  const contracted = execution.allocate(`triangle.${direction}.contracted`, pairs * input.triangleHidden);
  const hiddenNormalized = execution.allocate(`triangle.${direction}.hidden-normalized`, pairs * input.triangleHidden);
  const output = residualTarget ?? execution.allocate(`triangle.${direction}.output`, pairs * input.cZ);
  execution.dispatch(encoder, normalizeInput, [pair, weights, normalized], Math.ceil(pairs / 64), 1, 1,
    `triangle.${direction}.normalize-input`);
  let grid = execution.linearGrid(pairs * input.triangleHidden);
  execution.dispatch(encoder, projectAB, [normalized, pairMask, weights, a, b],
    Math.ceil(input.triangleHidden / 16), Math.ceil(pairs / 16), 1,
    `triangle.${direction}.project`);
  execution.dispatch(encoder, contract, [a, b, contracted], Math.ceil(input.length / 8),
    Math.ceil(input.length / 8), input.triangleHidden, `triangle.${direction}.contract`);
  execution.dispatch(encoder, normalizeHidden, [contracted, weights, hiddenNormalized], Math.ceil(pairs / 64), 1, 1,
    `triangle.${direction}.normalize-hidden`);
  execution.dispatch(encoder, projectOutput, [normalized, hiddenNormalized, weights, output],
    Math.ceil(input.cZ / 16), Math.ceil(pairs / 16), 1,
    `triangle.${direction}.output`);
  return output;
}

export async function encodeEvoformerBlock(
  execution,
  encoder,
  input,
  msa,
  pair,
  msaMask,
  pairMask,
) {
  // 🔴 WHERE THE OUTER PRODUCT MEAN SITS IS A PER-MODEL FACT, not a style. AF2
  // multimer runs it at the TOP of the block, monomer after the MSA transition,
  // and the weights were trained for their own ordering - so this cannot be
  // padded around the way a width can. It is the only structural difference
  // between the two evoformer blocks; everything else here is shared.
  const outerProductMeanFirst = input.outerProductMeanFirst === true;
  const outerProductMean = async() => {
    const update = await encodeOuterProductMean(
      execution, encoder, msa, msaMask, input, input.weights.outerProductMean, pair, input.covMask,
    );
    if (update !== pair) await execution.addInPlace(encoder, pair, update, "outer-product-mean.residual");
  };
  if (outerProductMeanFirst) await outerProductMean();

  const row = input.weights.msaRowAttention;
  await encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, pairSource: pair, batch: input.sequences, queries: input.length,
    channels: input.cM, heads: row.heads, transpose: false, weights: row.attention,
    chainMask: input.rowAttentionChainMask,
    pairBias: {
      source: "separate", activations: new Float32Array(0), channels: input.cZ,
      layerNormScale: row.pairLayerNormScale, layerNormOffset: row.pairLayerNormOffset,
      projectionWeight: row.pairProjectionWeight,
    },
    label: "msa-row-attention", residualTarget: msa,
  });

  const column = input.weights.msaColumnAttention;
  await encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, batch: input.length, queries: input.sequences,
    channels: input.cM, heads: column.heads, transpose: true, weights: column.attention,
    label: "msa-column-attention", residualTarget: msa,
  });

  await encodeTransition(
    execution, encoder, msa, input.sequences * input.length, input.cM,
    input.weights.msaTransition, "msa-transition", msa,
  );

  if (!outerProductMeanFirst) await outerProductMean();

  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, input, input.weights.triangleMultiplicationOutgoing, "outgoing", pair,
  );
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, input, input.weights.triangleMultiplicationIncoming, "incoming", pair,
  );

  const starting = input.weights.triangleAttentionStarting;
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: input.length, queries: input.length,
    channels: input.cZ, heads: starting.heads, transpose: false, weights: starting.attention,
    pairBias: { source: "normalized-input", projectionWeight: starting.pairProjectionWeight },
    label: "triangle-attention-starting", residualTarget: pair,
  });

  const ending = input.weights.triangleAttentionEnding;
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: input.length, queries: input.length,
    channels: input.cZ, heads: ending.heads, transpose: true, weights: ending.attention,
    pairBias: { source: "normalized-input", projectionWeight: ending.pairProjectionWeight },
    label: "triangle-attention-ending", residualTarget: pair,
  });

  await encodeTransition(
    execution, encoder, pair, input.length * input.length, input.cZ,
    input.weights.pairTransition, "pair-transition", pair,
  );
}

export async function encodeEvoformerPairBlock(
  execution,
  encoder,
  shape,
  weights,
  msa,
  pair,
  msaMask,
  pairMask,
) {
  let update = await encodeOuterProductMean(
    execution, encoder, msa, msaMask, shape, weights.outerProductMean, pair, shape.covMask,
  );
  if (update !== pair) await execution.addInPlace(encoder, pair, update, "extra.outer-product-mean.residual");
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationOutgoing, "outgoing", pair,
  );
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationIncoming, "incoming", pair,
  );
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionStarting.heads, transpose: false,
    weights: weights.triangleAttentionStarting.attention,
    pairBias: {
      source: "normalized-input", projectionWeight: weights.triangleAttentionStarting.pairProjectionWeight,
    },
    label: "extra.triangle-attention-starting", residualTarget: pair,
  });
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionEnding.heads, transpose: true,
    weights: weights.triangleAttentionEnding.attention,
    pairBias: {
      source: "normalized-input", projectionWeight: weights.triangleAttentionEnding.pairProjectionWeight,
    },
    label: "extra.triangle-attention-ending", residualTarget: pair,
  });
  await encodeTransition(
    execution, encoder, pair, shape.length * shape.length, shape.cZ,
    weights.pairTransition, "extra.pair-transition", pair,
  );
}

export async function encodeExtraMsaBlock(
  execution,
  encoder,
  shape,
  weights,
  msa,
  pair,
  msaMask,
  pairMask,
) {
  const row = weights.msaRowAttention;
  await encodeAttention(execution, encoder, {
    chainMask: shape.rowAttentionChainMask,
    source: msa, mask: msaMask, pairSource: pair, batch: shape.sequences, queries: shape.length,
    channels: shape.cM, heads: row.heads, transpose: false, weights: row.attention,
    pairBias: {
      source: "separate", activations: new Float32Array(0), channels: shape.cZ,
      layerNormScale: row.pairLayerNormScale, layerNormOffset: row.pairLayerNormOffset,
      projectionWeight: row.pairProjectionWeight,
    },
    label: "extra.msa-row-attention", residualTarget: msa,
  });
  await encodeGlobalAttention(
    execution, encoder, msa, msaMask, shape, weights.msaColumnGlobalAttention,
    "extra.msa-column-global-attention", msa,
  );
  await encodeTransition(
    execution, encoder, msa, shape.sequences * shape.length, shape.cM, weights.msaTransition,
    "extra.msa-transition", msa,
  );
  await encodeEvoformerPairBlock(execution, encoder, shape, weights, msa, pair, msaMask, pairMask);
}

export async function encodeTemplatePairBlock(
  execution,
  encoder,
  shape,
  weights,
  pair,
  pairMask,
) {
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionStarting.heads, transpose: false,
    weights: weights.triangleAttentionStarting.attention,
    pairBias: {
      source: "normalized-input", projectionWeight: weights.triangleAttentionStarting.pairProjectionWeight,
    },
    label: "template.triangle-attention-starting", residualTarget: pair,
  });
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionEnding.heads, transpose: true,
    weights: weights.triangleAttentionEnding.attention,
    pairBias: {
      source: "normalized-input", projectionWeight: weights.triangleAttentionEnding.pairProjectionWeight,
    },
    label: "template.triangle-attention-ending", residualTarget: pair,
  });
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationOutgoing, "outgoing", pair,
  );
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationIncoming, "incoming", pair,
  );
  await encodeTransition(
    execution, encoder, pair, shape.length * shape.length, shape.cZ,
    weights.pairTransition, "template.pair-transition", pair,
  );
}

export class EvoformerBlockGpu {
  device;

  constructor(device) { this.device = device; }

  async run(input) {
    const execution = new WebGpuExecution(this.device);
    try {
      const msaElements = input.sequences * input.length * input.cM;
      const pairElements = input.length * input.length * input.cZ;
      if (input.msa.length !== msaElements || input.pair.length !== pairElements) {
        throw new RangeError("Evoformer block activation shape mismatch");
      }
      const msa = execution.upload("block.msa", input.msa, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const pair = execution.upload("block.pair", input.pair, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaMask = execution.upload("block.msa-mask", input.msaMask);
      const pairMask = execution.upload("block.pair-mask", input.pairMask);
      const encoder = this.device.createCommandEncoder({ label: "evoformer-block" });
      this.device.pushErrorScope("validation");

      await encodeEvoformerBlock(execution, encoder, input, msa, pair, msaMask, pairMask);

      const msaReadback = execution.createReadback("block.msa-readback", msa, encoder);
      const pairReadback = execution.createReadback("block.pair-readback", pair, encoder);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      const [msaOutput, pairOutput] = await Promise.all([
        execution.mapFloat32(msaReadback), execution.mapFloat32(pairReadback),
      ]);
      return {
        msa: msaOutput,
        pair: pairOutput,
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
      };
    } finally {
      execution.release();
    }
  }
}
