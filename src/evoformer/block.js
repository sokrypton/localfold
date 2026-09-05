import {
  ATTENTION_NORMALIZE_SHADER,
  createAttentionNormalizeShader,
  attentionOutputTileColumns,
  attentionProjectTileColumns,
  attentionProjectTileRows,
  attentionOutputTileRows,
  createAttentionPairBiasShader,
  selectAttentionProjectKernel,
  selectAttentionOutputKernel,
  createAttentionNormParameters,
  createAttentionParameters,
  packAttentionWeights,
  buildAttentionFlashKernel,
  selectAttentionFlashKernel,

} from "./attention.js";
import {
  createOuterProductMeanParameters,
  OUTER_PRODUCT_MEAN_TILE_INTERMEDIATE_SHADER,
  OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER,
  OUTER_PRODUCT_MEAN_FINALIZE_SHADER,
  createOuterProductMeanContractShader,
  outerProductMeanTileCapacity,
  createOuterProductMeanProjectOutputShader,
  OUTER_PRODUCT_MEAN_NORMALIZE_SHADER,
  OUTER_PRODUCT_MEAN_PROJECT_SHADER,
  OPM_PROJECT_OUTPUT_PAIRS,
  opmProjectTileRows,
  opmProjectTileColumns,
  packOuterProductMeanWeights,
  useOuterFirstContraction,

} from "./outer-product-mean.js";
import {
  createTransitionNormalizeParameters,
  createTransitionShaders,
  chooseLinearKernel,
  linearTileColumns,
  packTransitionWeights,
  transitionChunkRows,
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
  // 🔴 THE TILE IS PART OF THE CACHE KEY, because it is part of the shader. The
  // deep MSA transitions want the wide tile and a 59-residue structure module
  // wants the narrow one, and a key that named neither would hand the second
  // shape the first shape's pipeline - dispatched with the wrong column stride,
  // which leaves columns unprojected and reads as a speedup.
  // 🔴 THE TILE AND THE PRECISION ARE ONE CHOICE - see chooseLinearKernel.
  const { tile, precision, weightPrecision } = chooseLinearKernel({
    rows, columns: Math.max(channels, hiddenChannels), device: execution.device,
  });
  const packed = packTransitionWeights(descriptor, weightPrecision);
  const tileColumns = linearTileColumns(tile);
  // 🔴 THE HIDDEN ACTIVATION IS THE BIGGEST THING A TRANSITION HOLDS and the
  // shortest-lived: four times the channels, written by the first pass and
  // read by the second and by nothing else. At 512 MSA rows it was 32 MiB of a
  // 572 MiB fold, capped there only because TRANSITION_CHUNK_TARGET_BYTES cuts
  // it into chunks. The store owns whole quads of columns, so a packed word is
  // never shared between lanes; a column count that is not a multiple of four
  // has no quad to own, which is what the guard below says.
  const hiddenStorage = hiddenChannels % 4 === 0 ? "f16" : "f32";
  const shaders = createTransitionShaders(
    descriptor, packed.offsets, tile, precision, weightPrecision, hiddenStorage);
  const key = `${precision}:${weightPrecision}:${tileColumns}:${hiddenStorage}`;
  const [normalize, linearFirst, linear, linearResidual] = await Promise.all([
    execution.pipelines.get(`block:transition:normalize:${weightPrecision}`, shaders[0]),
    execution.pipelines.get(`block:transition:linear-first:${key}`, shaders[1]),
    execution.pipelines.get(`block:transition:linear:${key}`, shaders[2]),
    execution.pipelines.get(`block:transition:linear-residual:${key}`, shaders[3]),
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
    const hidden = execution.allocate(
      `${label}.hidden`, rows * hiddenChannels, GPUBufferUsage.STORAGE, hiddenStorage);
    const transitionNormGrid = execution.rowGrid(rows);
    execution.dispatch(encoder, normalize, [source, weights, normalizeParams, normalized],
      transitionNormGrid[0], transitionNormGrid[1], 1, `${label}.normalize`);
    execution.dispatch(encoder, linearFirst, [normalized, weights, firstParams, hidden],
      Math.ceil(hiddenChannels / tileColumns), Math.ceil(rows / TRANSITION_TILE_ROWS), 1,
      `${label}.first`);
    execution.dispatch(encoder, residualTarget === undefined ? linear : linearResidual,
      [hidden, weights, secondParams, output],
      Math.ceil(channels / tileColumns), Math.ceil(rows / TRANSITION_TILE_ROWS), 1,
      `${label}.second`);
    return output;
  }

  // 🔴 ONE SCRATCH PAIR FOR THE WHOLE LOOP, viewed per chunk rather than
  // reallocated. The scratch is sized for a chunk, and what changes each time
  // round is which rows of the SOURCE and the OUTPUT are bound - so the big
  // tensors are never bound whole and never need to be bindable whole.
  const normalized = execution.allocate(`${label}.normalized-chunk`, chunkRows * channels);
  const hidden = execution.allocate(
    `${label}.hidden-chunk`, chunkRows * hiddenChannels, GPUBufferUsage.STORAGE, hiddenStorage);
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
    execution.dispatch(encoder, linearFirst, [normalizedChunk, weights, firstParams, hiddenChunk],
      Math.ceil(hiddenChannels / tileColumns), Math.ceil(count / TRANSITION_TILE_ROWS), 1,
      `${label}.first-${rowOffset}`);
    execution.dispatch(encoder, residualTarget === undefined ? linear : linearResidual,
      [hiddenChunk, weights, secondParams, outputChunk],
      Math.ceil(channels / tileColumns), Math.ceil(count / TRANSITION_TILE_ROWS), 1,
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
  // 🔴 BUILT SEPARATELY SO A REFUSED SUBGROUP PIPELINE CAN FALL BACK. See
  // buildAttentionFlashKernel: a device can carry both subgroup features and
  // still reject `@subgroup_size(32)`, and inside a Promise.all that rejection
  // is just a failed fold.
  // 🔴 THE STORAGE IS THE FLASH KERNEL'S TO REFUSE, AND IT IS ASKED FIRST.
  // Only the register-resident kernel reads the projected tensors packed - it
  // already reads them four floats at a time, and four halves are two words at
  // the same index. Every other variant is f32, and a caller that allocated
  // packed anyway would hand it four bindings of the right byte length holding
  // twice the values they should. So this is awaited BEFORE anything is
  // allocated, and what comes back decides, not what was asked for.
  const built = await buildAttentionFlashKernel(
    execution, execution.device, options.channels / options.heads, undefined, undefined,
    { input: "f16", output: "f16" });
  const flashKernel = built.kernel;
  const flash = built.pipeline;
  // A packed pair is two adjacent channels of one row, so a row has to hold a
  // whole number of them. Every attention here projects to 256 or 128.
  const projectedStorage = flashKernel.packedStorageSupported === true
    && options.channels % 2 === 0 ? "f16" : "f32";
  // 🔴 THE PROJECTION'S TILE TRAVELS WITH ITS SHADER, because the dispatch
  // below divides by it and the two shapes differ by precision - see
  // selectAttentionProjectKernel.
  // 🔴 THE NORMALISED ACTIVATION IS PACKED, AND NOTHING ELSE HERE IS YET. It
  // is the one tensor in this operation with exactly two touchers - the layer
  // norm writes it and the projection reads it - and both were already
  // generated shaders, so it is where the packed storage can be measured
  // without a five-kernel change. At 512 MSA rows it is 29.5 MiB of a fold
  // whose peak is 603; see src/runtime/storage.js for what a packed word costs.
  const normalizedStorage = "f16";
  const pairBiasStorage = options.pairBias?.source === "separate" ? "f32" : normalizedStorage;
  const projectKernel = selectAttentionProjectKernel(
    execution.device, options.projectPrecision ?? "auto", normalizedStorage, projectedStorage);
  const outputKernel = selectAttentionOutputKernel(
    execution.device, options.residualTarget !== undefined,
    options.outputPrecision ?? "auto", projectedStorage);
  const [normalize, packedNormalize, project, pairProject, outputProject] = await Promise.all([
    execution.pipelines.get("block:attention:normalize", ATTENTION_NORMALIZE_SHADER),
    execution.pipelines.get(`block:attention:normalize:${normalizedStorage}`,
      createAttentionNormalizeShader(normalizedStorage)),
    execution.pipelines.get(projectKernel.cacheKey, projectKernel.shader),
    // The bias reads `normalized` itself unless the caller gave a separate
    // source, so its storage is the normalised one in exactly that case.
    execution.pipelines.get(`block:attention:pair-bias:${pairBiasStorage}`,
      createAttentionPairBiasShader(pairBiasStorage)),
    execution.pipelines.get(outputKernel.cacheKey, outputKernel.shader),
  ]);
  const rows = options.batch * options.queries;
  const elements = rows * options.channels;
  const weights = execution.upload(`${options.label}.weights`, packed.data);
  const params = uniform(execution, `${options.label}.parameters`, createAttentionParameters(descriptor, packed.offsets));
  const normParams = uniform(execution, `${options.label}.norm-parameters`, createAttentionNormParameters(
    rows, options.channels, packed.offsets[0], packed.offsets[1], options.transpose,
    options.batch, options.queries, 1e-5,
  ));
  const normalized = execution.allocate(
    `${options.label}.normalized`, elements, GPUBufferUsage.STORAGE, normalizedStorage);
  const projected = (name) => execution.allocate(
    `${options.label}.${name}`, elements, GPUBufferUsage.STORAGE, projectedStorage);
  const query = projected("query");
  const key = projected("key");
  const value = projected("value");
  const gate = projected("gate");
  const weighted = projected("weighted");
  const output = options.residualTarget ?? execution.allocate(`${options.label}.output`, elements);
  const attentionNormGrid = execution.rowGrid(rows);
  execution.dispatch(encoder, packedNormalize, [options.source, weights, normParams, normalized],
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
    execution.dispatch(encoder, pairProject, [normalizedPair, weights, params, pairBias],
      grid[0], grid[1], 1, `${options.label}.pair-bias`);
  }
  execution.dispatch(encoder, project, [normalized, weights, params, query, key, value, gate],
    Math.ceil(options.channels / attentionProjectTileColumns(projectKernel.tile)),
    Math.ceil(rows / attentionProjectTileRows(projectKernel.tile)), 1,
    `${options.label}.project`);
  execution.dispatch(encoder, flash, [query, key, value, gate, options.mask, pairBias, params, weighted],
    Math.ceil(options.queries / flashKernel.queryTile),
    options.batch, options.heads, `${options.label}.flash`);
  execution.dispatch(encoder, outputProject, [weighted, weights, params, output],
    Math.ceil(options.channels / attentionOutputTileColumns(outputKernel.tile)),
    Math.ceil(rows / attentionOutputTileRows(outputKernel.tile)), 1,
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
    execution.pipelines.get("block:opm:tile-accumulate", OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER),
    execution.pipelines.get("block:opm:finalize", OUTER_PRODUCT_MEAN_FINALIZE_SHADER),
    execution.pipelines.get(`block:opm:contract:${input.cOuter}`,
      createOuterProductMeanContractShader(input.cOuter)),
    execution.pipelines.get(
      outerFirst && residualTarget !== undefined
        ? `block:opm:project-output-residual:${input.cOuter}`
        : `block:opm:project-output:${input.cOuter}`,
      outerFirst && residualTarget !== undefined
        ? createOuterProductMeanProjectOutputShader(input.cOuter, true)
        : createOuterProductMeanProjectOutputShader(input.cOuter),
    ),
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
  // 🔴 A TILE GRID, NOT A THREAD GRID. The projection used to give one thread
  // each (row, channel); it now gives one WORKGROUP a tile of both, so the
  // dispatch counts tiles. Row tiles are folded through x and y for the same
  // reason every other row grid here is - a row per sequence and residue passes
  // 32,768 at any real MSA depth - and the channel tile is z.
  const projectRowTiles = Math.ceil(rows / opmProjectTileRows());
  let grid = execution.rowGrid(projectRowTiles);
  execution.dispatch(encoder, project, [normalized, msaMask, weights, params, left, right],
    grid[0], grid[1], Math.ceil(input.cOuter / opmProjectTileColumns()),
    "opm.project");
  const outputGrid = execution.linearGrid(pairElements);
  if (outerFirst) {
    // ...both are one workgroup per PAIR; see outer-product-mean.js.
    const pairGrid = execution.linearGrid(input.length * input.length * 64);
    execution.dispatch(encoder, contractPipeline, [left, right, params, intermediate],
      pairGrid[0], pairGrid[1], 1, "opm.contract");
    // ...its OWN grid, because it carries several pairs a workgroup where the
    // contraction carries one; they shared `pairGrid` when both were one.
    const projectOutputGrid = execution.linearGrid(
      Math.ceil(input.length * input.length / OPM_PROJECT_OUTPUT_PAIRS) * 64);
    execution.dispatch(encoder, projectOutputPipeline, [intermediate, msaMask, weights, params, output],
      projectOutputGrid[0], projectOutputGrid[1], 1, "opm.project-output");
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
        [right, intermediate, params, tileParams, output],
        outputGrid[0], outputGrid[1], 1, `opm.accumulate-${offset}`);
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
  // 🔴 THE RESIDUAL FORM IS GENERATED, NOT PATCHED. It used to be a string
  // replacement on the finished WGSL; when the kernel's writeback was rewritten
  // the pattern stopped matching, and a replacement that matches nothing throws
  // nothing - the block would have OVERWRITTEN the pair representation instead
  // of adding to it, on the shipped AF2 path only.
  const residualShaders = createTriangleShaders(
    shape, "f32", packed.offsets, 1e-5, direction, "two-pass", shaders.projectTile, true);
  const pipelineKey = `block:triangle:${direction}:${input.length}:${input.cZ}:${input.triangleHidden}`;
  const [normalizeInput, projectAB, contract, normalizeHidden, projectOutput] = await Promise.all([
    execution.pipelines.get(`${pipelineKey}:normalize-input`, shaders.normalizeInput),
    execution.pipelines.get(`${pipelineKey}:project-ab`, shaders.projectAB),
    execution.pipelines.get(`${pipelineKey}:contract`, shaders.contract),
    execution.pipelines.get(`${pipelineKey}:normalize-hidden`, shaders.normalizeHidden),
    execution.pipelines.get(
      `${pipelineKey}:project-output${residualTarget === undefined ? "" : "-residual"}`,
      residualTarget === undefined ? shaders.projectOutput : residualShaders.projectOutput,
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
  const normalizeGroups = Math.ceil(pairs / shaders.normalizeRows);
  execution.dispatch(encoder, normalizeInput, [pair, weights, normalized], normalizeGroups, 1, 1,
    `triangle.${direction}.normalize-input`);
  let grid = execution.linearGrid(pairs * input.triangleHidden);
  execution.dispatch(encoder, projectAB, [normalized, pairMask, weights, a, b],
    Math.ceil(input.triangleHidden / shaders.projectTile.columns),
    Math.ceil(pairs / shaders.projectTile.rows), 1,
    `triangle.${direction}.project`);
  execution.dispatch(encoder, contract, [a, b, contracted],
    Math.ceil(input.length / shaders.contractTile.columns),
    Math.ceil(input.length / shaders.contractTile.rows),
    input.triangleHidden, `triangle.${direction}.contract`);
  execution.dispatch(encoder, normalizeHidden, [contracted, weights, hiddenNormalized],
    normalizeGroups, 1, 1, `triangle.${direction}.normalize-hidden`);
  execution.dispatch(encoder, projectOutput, [normalized, hiddenNormalized, weights, output],
    Math.ceil(input.cZ / shaders.projectTile.columns),
    Math.ceil(pairs / shaders.projectTile.rows), 1,
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
  const row = input.weights.msaRowAttention;
  await encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, pairSource: pair, batch: input.sequences, queries: input.length,
    channels: input.cM, heads: row.heads, transpose: false, weights: row.attention,
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

  let update = await encodeOuterProductMean(
    execution, encoder, msa, msaMask, input, input.weights.outerProductMean, pair,
  );
  if (update !== pair) await execution.addInPlace(encoder, pair, update, "outer-product-mean.residual");

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
    execution, encoder, msa, msaMask, shape, weights.outerProductMean, pair,
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
