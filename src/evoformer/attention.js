import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;
const ceilDivide = (value, divisor) => Math.ceil(value / divisor);

function validate(input) {
  const { batch, queryLength, channels, heads, activations, mask, weights } = input;
  if (![batch, queryLength, channels, heads].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("attention dimensions must be positive safe integers");
  }
  if (channels % heads !== 0 || channels / heads > 32) {
    throw new RangeError("the fused WebGPU attention path requires an integral head dimension no larger than 32");
  }
  const projected = channels;
  const expected = [
    ["activations", activations, batch * queryLength * channels],
    ["mask", mask, batch * queryLength],
    ["queryNormScale", weights.queryNormScale, channels],
    ["queryNormOffset", weights.queryNormOffset, channels],
    ["queryWeight", weights.queryWeight, channels * projected],
    ["keyWeight", weights.keyWeight, channels * projected],
    ["valueWeight", weights.valueWeight, channels * projected],
    ["gatingWeight", weights.gatingWeight, channels * projected],
    ["gatingBias", weights.gatingBias, projected],
    ["outputWeight", weights.outputWeight, projected * channels],
    ["outputBias", weights.outputBias, channels],
  ];
  for (const [name, value, size] of expected) {
    if (value.length !== size) throw new RangeError(`${name} has ${value.length} values; expected ${size}`);
  }
  const pair = input.pairBias;
  if (pair?.source === "separate") {
    const pairExpected = [
      ["pair activations", pair.activations, queryLength * queryLength * pair.channels],
      ["pair norm scale", pair.layerNormScale, pair.channels],
      ["pair norm offset", pair.layerNormOffset, pair.channels],
      ["pair projection", pair.projectionWeight, pair.channels * heads],
    ];
    for (const [name, value, size] of pairExpected) {
      if (value.length !== size) throw new RangeError(`${name} has ${value.length} values; expected ${size}`);
    }
  } else if (pair !== undefined && pair.projectionWeight.length !== channels * heads) {
    throw new RangeError("normalized-input pair projection has an invalid size");
  }
}

export function packAttentionWeights(input) {
  const w = input.weights;
  const tensors = [
    w.queryNormScale, w.queryNormOffset, w.queryWeight, w.keyWeight, w.valueWeight,
    w.gatingWeight, w.gatingBias, w.outputWeight, w.outputBias,
  ];
  const pair = input.pairBias;
  if (pair?.source === "separate") {
    tensors.push(pair.layerNormScale, pair.layerNormOffset, pair.projectionWeight);
  } else if (pair !== undefined) {
    tensors.push(pair.projectionWeight);
  }
  const offsets = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const data = new Float32Array(size);
  tensors.forEach((tensor, index) => data.set(tensor, offsets[index]));
  return { data, offsets };
}

export function createAttentionParameters(input, offsets) {
  const pairProjectionIndex = input.pairBias?.source === "separate" ? 11 : 9;
  return new Uint32Array([
    input.batch, input.queryLength, input.channels, input.heads, input.channels / input.heads,
    input.transpose === true ? 1 : 0, input.pairBias === undefined ? 0 : 1,
    offsets[2], offsets[3], offsets[4], offsets[5], offsets[6], offsets[7], offsets[8],
    input.pairBias === undefined ? 0 : offsets[pairProjectionIndex],
    input.pairBias?.source === "separate" ? input.pairBias.channels : input.channels,
  ]);
}

export function createAttentionNormParameters(
  rows, channels, scale, offset,
  transpose, batch, queries, epsilon,
) {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  [rows, channels, scale, offset, transpose ? 1 : 0, batch, queries].forEach(
    (value, index) => view.setUint32(index * 4, value, true),
  );
  view.setFloat32(28, epsilon, true);
  return new Uint8Array(buffer);
}

export const ATTENTION_NORMALIZE_SHADER = `
const GRID_WIDTH: u32 = 32768u;
struct NormParameters {
  rows: u32, channels: u32, scale: u32, offset: u32,
  transpose: u32, batch: u32, queries: u32, epsilon: f32,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: NormParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;

fn source_row(row: u32) -> u32 {
  if (p.transpose == 0u) { return row; }
  let b = row / p.queries;
  let q = row % p.queries;
  return q * p.batch + b;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  // ONE WORKGROUP PER ROW, folded across two dimensions: a pair track has L*L
  // rows, which passes 65535 at L=256 and is a validation error rather than a
  // slow run. Callers that dispatch a single row of workgroups pass y=1, so
  // this adds nothing for them.
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= p.rows) { return; }
  let input_base = source_row(row) * p.channels;
  let output_base = row * p.channels;
  var sum = 0.0;
  for (var c = local.x; c < p.channels; c += 64u) { sum += source[input_base + c]; }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(p.channels); }
  workgroupBarrier();
  var squared = 0.0;
  for (var c = local.x; c < p.channels; c += 64u) {
    let centered = source[input_base + c] - row_mean[0];
    squared += centered * centered;
  }
  partial[local.x] = squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(p.channels) + p.epsilon);
  for (var c = local.x; c < p.channels; c += 64u) {
    output[output_base + c] = (source[input_base + c] - row_mean[0]) * inverse_std
      * weights[p.scale + c] + weights[p.offset + c];
  }
}`;

const COMMON = `
struct Parameters {
  batch: u32, queries: u32, channels: u32, heads: u32,
  head_dim: u32, transpose: u32, has_pair_bias: u32,
  query_weight: u32, key_weight: u32, value_weight: u32,
  gating_weight: u32, gating_bias: u32, output_weight: u32,
  output_bias: u32, pair_weight: u32, pair_channels: u32,
};
const GRID_WIDTH: u32 = 32768u;
`;

export const ATTENTION_PROJECT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> query: array<f32>;
@group(0) @binding(4) var<storage, read_write> key: array<f32>;
@group(0) @binding(5) var<storage, read_write> value: array<f32>;
@group(0) @binding(6) var<storage, read_write> gate: array<f32>;
var<workgroup> tile_source: array<f32, 128>;
// 🔴 ONE vec4 A CELL, NOT FOUR ARRAYS. q, k, v and the gate are four matrices
// contracted over the same normalised activation at the same (channel, output)
// cell, so the four weights a cell needs are always wanted together: packed as
// a vec4 the inner loop reads them in ONE instruction and accumulates them in
// one multiply-add instead of four. Ten workgroup reads bought sixteen
// multiply-adds; four buy the same sixteen. tools/gpu/probe-alu.js puts
// workgroup reads at 394 billion a second against 580 billion vec4
// multiply-adds, so the reads were the larger term. AF3's triangle projection
// is the same change; see src/triangle/shaders.js.
var<workgroup> tile_weight: array<vec4<f32>, 128>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let projected = p.heads * p.head_dim;
  let rows = p.batch * p.queries;
  let row = group.y * 16u + local.y;
  let second_row = row + 8u;
  let hd = group.x * 16u + local.x;
  let second_hd = hd + 8u;
  let tile_index = local.y * 8u + local.x;
  // Each cell accumulates (query, key, value, gate).
  var acc_00 = vec4<f32>(0.0);
  var acc_01 = vec4<f32>(0.0);
  var acc_10 = vec4<f32>(0.0);
  var acc_11 = vec4<f32>(0.0);
  if (hd < projected) {
    acc_00.w = weights[p.gating_bias + hd];
    acc_10.w = acc_00.w;
  }
  if (second_hd < projected) {
    acc_01.w = weights[p.gating_bias + second_hd];
    acc_11.w = acc_01.w;
  }
  for (var c0 = 0u; c0 < p.channels; c0 += 8u) {
    let source_c = c0 + local.x;
    let weight_c = c0 + local.y;
    tile_source[tile_index] = 0.0;
    tile_source[tile_index + 64u] = 0.0;
    if (row < rows && source_c < p.channels) {
      tile_source[tile_index] = source[row * p.channels + source_c];
    }
    if (second_row < rows && source_c < p.channels) {
      tile_source[tile_index + 64u] = source[second_row * p.channels + source_c];
    }
    for (var column_block = 0u; column_block < 2u; column_block += 1u) {
      let tile_offset = tile_index + column_block * 64u;
      let output_hd = hd + column_block * 8u;
      var packed = vec4<f32>(0.0);
      if (output_hd < projected && weight_c < p.channels) {
        let weight_index = weight_c * projected + output_hd;
        packed = vec4<f32>(weights[p.query_weight + weight_index],
                           weights[p.key_weight + weight_index],
                           weights[p.value_weight + weight_index],
                           weights[p.gating_weight + weight_index]);
      }
      tile_weight[tile_offset] = packed;
    }
    workgroupBarrier();
    for (var c = 0u; c < 8u; c += 1u) {
      let x_0 = tile_source[local.y * 8u + c];
      let x_1 = tile_source[local.y * 8u + c + 64u];
      let packed_0 = tile_weight[c * 8u + local.x];
      let packed_1 = tile_weight[c * 8u + local.x + 64u];
      acc_00 += x_0 * packed_0;
      acc_01 += x_0 * packed_1;
      acc_10 += x_1 * packed_0;
      acc_11 += x_1 * packed_1;
    }
    workgroupBarrier();
  }
  if (row < rows && hd < projected) {
    let index = row * projected + hd;
    query[index] = acc_00.x * inverseSqrt(f32(p.head_dim));
    key[index] = acc_00.y;
    value[index] = acc_00.z;
    gate[index] = 1.0 / (1.0 + exp(-acc_00.w));
  }
  if (row < rows && second_hd < projected) {
    let index = row * projected + second_hd;
    query[index] = acc_01.x * inverseSqrt(f32(p.head_dim));
    key[index] = acc_01.y;
    value[index] = acc_01.z;
    gate[index] = 1.0 / (1.0 + exp(-acc_01.w));
  }
  if (second_row < rows && hd < projected) {
    let index = second_row * projected + hd;
    query[index] = acc_10.x * inverseSqrt(f32(p.head_dim));
    key[index] = acc_10.y;
    value[index] = acc_10.z;
    gate[index] = 1.0 / (1.0 + exp(-acc_10.w));
  }
  if (second_row < rows && second_hd < projected) {
    let index = second_row * projected + second_hd;
    query[index] = acc_11.x * inverseSqrt(f32(p.head_dim));
    key[index] = acc_11.y;
    value[index] = acc_11.z;
    gate[index] = 1.0 / (1.0 + exp(-acc_11.w));
  }
}`;

export const ATTENTION_PAIR_BIAS_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.heads * p.queries * p.queries) { return; }
  let k = index % p.queries;
  let q = (index / p.queries) % p.queries;
  let head = index / (p.queries * p.queries);
  var result = 0.0;
  for (var c = 0u; c < p.pair_channels; c += 1u) {
    result += pair[(q * p.queries + k) * p.pair_channels + c]
      * weights[p.pair_weight + c * p.heads + head];
  }
  output[index] = result;
}`;

export const ATTENTION_FLASH_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 32>;
var<workgroup> state: array<f32, 3>;

fn mask_index(batch: u32, key_index: u32) -> u32 {
  if (p.transpose == 0u) { return batch * p.queries + key_index; }
  return key_index * p.batch + batch;
}

@compute @workgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let q_index = group.x;
  let batch_index = group.y;
  let head = group.z;
  let lane = local.x;
  if (q_index >= p.queries || batch_index >= p.batch || head >= p.heads) { return; }
  let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
  var accumulated = 0.0;
  var running_max = -1e30;
  var running_sum = 0.0;
  for (var k_index = 0u; k_index < p.queries; k_index += 1u) {
    let k_base = ((batch_index * p.queries + k_index) * p.heads + head) * p.head_dim;
    partial[lane] = select(0.0, query[q_base + lane] * key[k_base + lane], lane < p.head_dim);
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride /= 2u) {
      if (lane < stride) { partial[lane] += partial[lane + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u) {
      var logit = partial[0] + 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
      if (p.has_pair_bias != 0u) {
        logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
      }
      logit = clamp(logit, -1e8, 1e8);
      let new_max = max(running_max, logit);
      state[0] = exp(running_max - new_max);
      state[1] = exp(logit - new_max);
      running_sum = running_sum * state[0] + state[1];
      running_max = new_max;
      state[2] = running_sum;
    }
    workgroupBarrier();
    if (lane < p.head_dim) {
      accumulated = accumulated * state[0] + state[1] * value[k_base + lane];
    }
    workgroupBarrier();
  }
  if (lane < p.head_dim) {
    output[q_base + lane] = (accumulated / state[2]) * gate[q_base + lane];
  }
}`;

// Fast path for devices that guarantee one 32-lane subgroup. All lanes keep
// identical online-softmax state, eliminating workgroup barriers in the key loop.
export const ATTENTION_SUBGROUP_FLASH_SHADER = `enable subgroups;
enable subgroup_size_control;
${COMMON}
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> key_tile: array<vec4<f32>, 64>;
var<workgroup> value_tile: array<vec4<f32>, 64>;

fn mask_index(batch: u32, key_index: u32) -> u32 {
  if (p.transpose == 0u) { return batch * p.queries + key_index; }
  return key_index * p.batch + batch;
}

@compute @workgroup_size(32, 4, 1) @subgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let q_index = group.x * 4u + local.y;
  let batch_index = group.y;
  let head = group.z;
  let lane = local.x;
  let valid_query = q_index < p.queries;
  let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
  var accumulated = 0.0;
  var running_max = -1e30;
  var running_sum = 0.0;
  let linear_lane = local.y * 32u + local.x;
  for (var k0 = 0u; k0 < p.queries; k0 += 8u) {
    if (linear_lane < 64u) {
      let tile_key_index = linear_lane / 8u;
      let vector_index = linear_lane % 8u;
      let k_index = k0 + tile_key_index;
      key_tile[linear_lane] = vec4<f32>(0.0);
      if (k_index < p.queries) {
        let k_base = (((batch_index * p.queries + k_index) * p.heads + head) * 8u);
        key_tile[linear_lane] = key[k_base + vector_index];
      }
    } else {
      let item = linear_lane - 64u;
      let tile_key_index = item / 8u;
      let vector_index = item % 8u;
      let k_index = k0 + tile_key_index;
      value_tile[item] = vec4<f32>(0.0);
      if (k_index < p.queries) {
        let k_base = (((batch_index * p.queries + k_index) * p.heads + head) * 8u);
        value_tile[item] = value[k_base + vector_index];
      }
    }
    workgroupBarrier();
    for (var tile_key_index = 0u; tile_key_index < 8u; tile_key_index += 1u) {
      let k_index = k0 + tile_key_index;
      if (k_index < p.queries) {
        let vector_index = lane / 4u;
        let component = lane % 4u;
        var product = 0.0;
        if (valid_query) {
          product = query[q_base / 4u + vector_index][component]
            * key_tile[tile_key_index * 8u + vector_index][component];
        }
        var logit = subgroupAdd(product) + 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
        if (valid_query && p.has_pair_bias != 0u) {
          logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
        }
        logit = clamp(logit, -1e8, 1e8);
        let new_max = max(running_max, logit);
        let previous_scale = exp(running_max - new_max);
        let weight = exp(logit - new_max);
        running_sum = running_sum * previous_scale + weight;
        if (valid_query) {
          accumulated = accumulated * previous_scale
            + weight * value_tile[tile_key_index * 8u + vector_index][component];
        }
        running_max = new_max;
      }
    }
    workgroupBarrier();
  }
  if (valid_query && lane < p.head_dim) {
    output[q_base + lane] = (accumulated / running_sum) * gate[q_base + lane];
  }
}`;


/**
 * Flash attention with the whole head in registers.
 *
 * 🔴 THIS REPLACED A KERNEL THAT WAS 11x SLOWER, and the reason is barriers.
 * ATTENTION_FLASH_SHADER below gives one 32-lane workgroup to each query and
 * forms every q.k dot product with a tree reduction: five barriered steps to
 * reduce, two more to publish the softmax state, for every key of every query -
 * about two thousand barriers per query at 256 residues. None of it is needed.
 * A head is 32 channels, which is eight vec4s, which fits in registers. So one
 * INVOCATION takes a whole (batch, head, query): it loads its query once, walks
 * the keys computing each dot product itself, and never synchronises with
 * anybody. Measured on an M2, interleaved, batch 32 and 8 heads:
 *
 *   Q=64   4.30 -> 0.60 ms     Q=128  16.67 -> 1.67 ms
 *   Q=192  37.30 -> 3.23 ms    Q=256  66.20 -> 5.83 ms   (11.4x)
 *
 * relRMS against the old kernel is 2e-7, which is f32 summation order.
 *
 * WHY IT IS GENERATED PER head_dim. The eight accumulators have to be eight
 * named registers: GLSL and WGSL both allow indexing a local array, but a
 * driver that decides to spill one to memory puts the entire point of this
 * kernel back where it started. Unrolling at generation time makes that
 * impossible rather than unlikely. head_dim is 32 in all 48 Evoformer blocks
 * and 8 in the 4 extra-MSA ones, so this generates two shaders in a fold.
 */
export function createAttentionRegisterFlashShader(headDim) {
  const vectors = headDim / 4;
  const each = (body) => Array.from({ length: vectors }, (_, t) => `    ${body(t)}`).join("\n");
  const declare = (name, init) => Array.from({ length: vectors },
    (_, t) => `  var ${name}${t} = ${init(t)};`).join("\n");
  return `${COMMON}
const HD4: u32 = ${vectors}u;
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;

fn mask_index(batch: u32, key_index: u32) -> u32 {
  if (p.transpose == 0u) { return batch * p.queries + key_index; }
  return key_index * p.batch + batch;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let q_index = id.x;
  let batch_index = id.y;
  let head = id.z;
  if (q_index >= p.queries || batch_index >= p.batch || head >= p.heads) { return; }
  let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * HD4;

${declare("qv", (t) => `query[q_base + ${t}u]`)}
${declare("acc", () => "vec4<f32>(0.0)")}
  var running_max = -1e30;
  var running_sum = 0.0;

  for (var k_index = 0u; k_index < p.queries; k_index += 1u) {
    let k_base = ((batch_index * p.queries + k_index) * p.heads + head) * HD4;
    var score = 0.0;
${each((t) => `score += dot(qv${t}, key[k_base + ${t}u]);`)}
    var logit = score + 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
    if (p.has_pair_bias != 0u) {
      logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
    }
    logit = clamp(logit, -1e8, 1e8);
    let new_max = max(running_max, logit);
    let previous_scale = exp(running_max - new_max);
    let weight = exp(logit - new_max);
    running_sum = running_sum * previous_scale + weight;
    running_max = new_max;
${each((t) => `acc${t} = acc${t} * previous_scale + weight * value[k_base + ${t}u];`)}
  }

${each((t) => `output[q_base + ${t}u] = (acc${t} / running_sum) * gate[q_base + ${t}u];`)}
}`;
}

/**
 * The subgroup size every one of these kernels is written for. It is pinned in
 * the shader as `@subgroup_size(32)` and assumed by the lane arithmetic, so it
 * is a correctness requirement and not a preference.
 */
export const ATTENTION_SUBGROUP_SIZE = 32;

/**
 * Whether this device allows the subgroup size the kernels declare.
 *
 * 🔴 A DEVICE CAN HAVE THE FEATURES AND STILL REFUSE THE SIZE, AND THE FAILURE
 * IS FATAL RATHER THAN SLOW. Reported from a Pixel: "The subgroup_size
 * attribute (32) is not in the allowed range ([16, 16])" while building
 * block:attention:flash-subgroup-key32. Adreno and Mali parts report a
 * 16-lane range, so `@subgroup_size(32)` fails PIPELINE CREATION - the fold
 * does not fall back, it stops - and this checked only for the two features,
 * both of which those devices have.
 *
 * 🔴 UNKNOWN IS TREATED AS ALLOWED, DELIBERATELY. Browsers that expose the
 * subgroups feature without the size range would otherwise lose the fast path
 * they have been running correctly; the range is what the error message quotes,
 * so a browser that can raise that error can also report it.
 */
export function allowsAttentionSubgroupSize(device, size = ATTENTION_SUBGROUP_SIZE) {
  const info = device.adapterInfo ?? device.info ?? {};
  const min = info.subgroupMinSize;
  const max = info.subgroupMaxSize;
  if (typeof min === "number" && size < min) return false;
  if (typeof max === "number" && size > max) return false;
  return true;
}

export function supportsAttentionSubgroups(device, headDim = 32) {
  return headDim === 32 && device.features.has("subgroups")
    && device.features.has("subgroup-size-control")
    && allowsAttentionSubgroupSize(device);
}

/**
 * Software analogues of the Pallas tiled flash-attention algorithm.
 *
 * Eight 32-lane subgroups cooperate on up to 64 query rows and 64 key rows.
 * One lane owns one head channel, and K/V are loaded once into workgroup
 * memory. Scores remain ephemeral: online softmax immediately folds each score
 * into the value accumulator. The 64x64 specialization is the closest Pallas
 * analogue; smaller query tiles trade data reuse for more parallelism.
 */
function createAttentionSubgroupTiledShader(queryTile, keyTile) {
  const querySlots = queryTile / 8;
  return `enable subgroups;
enable subgroup_size_control;
${COMMON}
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> key_tile: array<vec4<f32>, ${keyTile * 8}>;
var<workgroup> value_tile: array<vec4<f32>, ${keyTile * 8}>;

fn mask_index(batch: u32, key_index: u32) -> u32 {
  if (p.transpose == 0u) { return batch * p.queries + key_index; }
  return key_index * p.batch + batch;
}

@compute @workgroup_size(32, 8, 1) @subgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
) {
  let batch_index = group.y;
  let head = group.z;
  let vector_index = subgroup_lane / 4u;
  let component = subgroup_lane % 4u;
  let linear_lane = local.y * 32u + local.x;
  var accumulated: array<f32, ${querySlots}>;
  var running_max: array<f32, ${querySlots}>;
  var running_sum: array<f32, ${querySlots}>;
  var query_component: array<f32, ${querySlots}>;

  for (var query_slot = 0u; query_slot < ${querySlots}u; query_slot += 1u) {
    let q_index = group.x * ${queryTile}u + local.y + query_slot * 8u;
    accumulated[query_slot] = 0.0;
    running_max[query_slot] = -1e30;
    running_sum[query_slot] = 0.0;
    query_component[query_slot] = 0.0;
    if (q_index < p.queries) {
      let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * 8u;
      query_component[query_slot] = query[q_base + vector_index][component];
    }
  }

  for (var k0 = 0u; k0 < p.queries; k0 += ${keyTile}u) {
    for (var tile_vector = linear_lane; tile_vector < ${keyTile * 8}u; tile_vector += 256u) {
      let tile_key_index = tile_vector / 8u;
      let tile_vector_index = tile_vector % 8u;
      let k_index = k0 + tile_key_index;
      key_tile[tile_vector] = vec4<f32>(0.0);
      value_tile[tile_vector] = vec4<f32>(0.0);
      if (k_index < p.queries) {
        let k_base = ((batch_index * p.queries + k_index) * p.heads + head) * 8u;
        key_tile[tile_vector] = key[k_base + tile_vector_index];
        value_tile[tile_vector] = value[k_base + tile_vector_index];
      }
    }
    workgroupBarrier();

    for (var query_slot = 0u; query_slot < ${querySlots}u; query_slot += 1u) {
      let q_index = group.x * ${queryTile}u + local.y + query_slot * 8u;
      let valid_query = q_index < p.queries;
      for (var tile_key_index = 0u; tile_key_index < ${keyTile}u; tile_key_index += 1u) {
        let k_index = k0 + tile_key_index;
        if (k_index < p.queries) {
          let product = query_component[query_slot]
            * key_tile[tile_key_index * 8u + vector_index][component];
          var logit = subgroupAdd(select(0.0, product, valid_query))
            + 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
          if (valid_query && p.has_pair_bias != 0u) {
            logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
          }
          logit = clamp(logit, -1e8, 1e8);
          let new_max = max(running_max[query_slot], logit);
          let previous_scale = exp(running_max[query_slot] - new_max);
          let weight = exp(logit - new_max);
          running_sum[query_slot] = running_sum[query_slot] * previous_scale + weight;
          running_max[query_slot] = new_max;
          if (valid_query) {
            accumulated[query_slot] = accumulated[query_slot] * previous_scale
              + weight * value_tile[tile_key_index * 8u + vector_index][component];
          }
        }
      }
    }
    workgroupBarrier();
  }

  for (var query_slot = 0u; query_slot < ${querySlots}u; query_slot += 1u) {
    let q_index = group.x * ${queryTile}u + local.y + query_slot * 8u;
    if (q_index < p.queries) {
      let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
      output[q_base + subgroup_lane] = (accumulated[query_slot] / running_sum[query_slot])
        * gate[q_base + subgroup_lane];
    }
  }
}`;
}

export const ATTENTION_SUBGROUP_8X16_SHADER = createAttentionSubgroupTiledShader(8, 16);
export const ATTENTION_SUBGROUP_8X32_SHADER = createAttentionSubgroupTiledShader(8, 32);
export const ATTENTION_SUBGROUP_8X64_SHADER = createAttentionSubgroupTiledShader(8, 64);
export const ATTENTION_SUBGROUP_16X64_SHADER = createAttentionSubgroupTiledShader(16, 64);
export const ATTENTION_SUBGROUP_32X64_SHADER = createAttentionSubgroupTiledShader(32, 64);
export const ATTENTION_SUBGROUP_64X64_SHADER = createAttentionSubgroupTiledShader(64, 64);

/**
 * Key-parallel flash attention for a 32-channel head.
 *
 * Every subgroup handles one query. Its lanes calculate 32 key scores in
 * parallel, reduce the tile softmax, and then shuffle those probabilities
 * across lanes while each lane accumulates one value/output channel.
 */
export const ATTENTION_SUBGROUP_KEY32_SHADER = `enable subgroups;
enable subgroup_size_control;
${COMMON}
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> key_tile: array<vec4<f32>, 256>;
var<workgroup> value_tile: array<vec4<f32>, 256>;

fn mask_index(batch: u32, key_index: u32) -> u32 {
  if (p.transpose == 0u) { return batch * p.queries + key_index; }
  return key_index * p.batch + batch;
}

@compute @workgroup_size(32, 8, 1) @subgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
) {
  let q_index = group.x * 8u + local.y;
  let batch_index = group.y;
  let head = group.z;
  let valid_query = q_index < p.queries;
  let query_base = ((batch_index * p.queries + q_index) * p.heads + head) * 8u;
  let output_vector = subgroup_lane / 4u;
  let output_component = subgroup_lane % 4u;
  let linear_lane = local.y * 32u + local.x;
  var accumulated = 0.0;
  var running_max = -1e30;
  var running_sum = 0.0;

  for (var k0 = 0u; k0 < p.queries; k0 += 32u) {
    let load_key = linear_lane / 8u;
    let load_vector = linear_lane % 8u;
    let source_key = k0 + load_key;
    let transposed_index = load_vector * 32u + load_key;
    key_tile[transposed_index] = vec4<f32>(0.0);
    value_tile[transposed_index] = vec4<f32>(0.0);
    if (source_key < p.queries) {
      let source_base = ((batch_index * p.queries + source_key) * p.heads + head) * 8u;
      key_tile[transposed_index] = key[source_base + load_vector];
      value_tile[transposed_index] = value[source_base + load_vector];
    }
    workgroupBarrier();

    let k_index = k0 + subgroup_lane;
    let valid_key = k_index < p.queries;
    var logit = -1e30;
    if (valid_query && valid_key) {
      logit = 0.0;
      for (var vector = 0u; vector < 8u; vector += 1u) {
        logit += dot(query[query_base + vector], key_tile[vector * 32u + subgroup_lane]);
      }
      logit += 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
      if (p.has_pair_bias != 0u) {
        logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
      }
      logit = clamp(logit, -1e8, 1e8);
    }

    let tile_max = subgroupMax(logit);
    let probability = select(0.0, exp(logit - tile_max), valid_query && valid_key);
    let tile_sum = subgroupAdd(probability);
    var tile_weighted = 0.0;
    for (var source_lane = 0u; source_lane < 32u; source_lane += 1u) {
      let source_probability = subgroupShuffle(probability, source_lane);
      tile_weighted += source_probability
        * value_tile[output_vector * 32u + source_lane][output_component];
    }

    let new_max = max(running_max, tile_max);
    let previous_scale = exp(running_max - new_max);
    let tile_scale = exp(tile_max - new_max);
    accumulated = accumulated * previous_scale + tile_weighted * tile_scale;
    running_sum = running_sum * previous_scale + tile_sum * tile_scale;
    running_max = new_max;
    workgroupBarrier();
  }

  if (valid_query) {
    let output_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
    output[output_base + subgroup_lane] = (accumulated / running_sum)
      * gate[output_base + subgroup_lane];
  }
}`;

export function supportsAttentionSubgroup64x64(device, headDim = 32) {
  return supportsAttentionSubgroups(device, headDim)
    && device.limits.maxComputeInvocationsPerWorkgroup >= 256
    && device.limits.maxComputeWorkgroupStorageSize >= 16_384;
}

/**
 * The flash kernel AND its pipeline, falling back if the device refuses it.
 *
 * 🔴 INTROSPECTION IS NOT ENOUGH ON ITS OWN. supportsAttentionSubgroups now
 * checks the device's subgroup size range, but a browser that exposes the
 * subgroups feature WITHOUT that range reads as "allowed" - deliberately, so
 * devices that have been running these kernels correctly keep them - and on a
 * part that only does 16 lanes the pipeline then fails to build and the fold
 * stops. createComputePipelineAsync rejects rather than throwing, so the
 * refusal is catchable, and the register-resident kernel needs no subgroup
 * support at all. Cheaper to try and fall back than to guess.
 *
 * @param {{pipelines: {get: (key: string, code: string) => Promise<GPUComputePipeline>}}} execution
 */
export async function buildAttentionFlashKernel(execution, device, headDim, requested = "auto") {
  const kernel = selectAttentionFlashKernel(device, headDim, requested);
  try {
    return { kernel, pipeline: await execution.pipelines.get(`block:${kernel.cacheKey}`, kernel.shader) };
  } catch (error) {
    // Only a subgroup kernel has a fallback; anything else failing is real.
    if (!kernel.variant.startsWith("subgroup")) throw error;
    const portable = selectAttentionFlashKernel(device, headDim, "portable");
    if (portable.cacheKey === kernel.cacheKey) throw error;
    return {
      kernel: portable,
      pipeline: await execution.pipelines.get(`block:${portable.cacheKey}`, portable.shader),
    };
  }
}

export function selectAttentionFlashKernel(
  device,
  headDim = 32,
  requested = "auto",
) {
  const subgroup = supportsAttentionSubgroups(device, headDim);
  const subgroup64 = supportsAttentionSubgroup64x64(device, headDim);
  // 🔴 THE REGISTER-RESIDENT KERNEL IS THE DEFAULT, AND THE SUBGROUP ONES ARE
  // NOT FASTER HERE. This preferred subgroup-key32 wherever the device had the
  // features, on the strength of timings recorded upstream - and README.md
  // believed Chrome-on-Metal never reached that path at all, so the comparison
  // had never been run on this hardware. It has now, on one column-attention
  // shaped problem (512 queries, 59 batch, 8 heads, head dim 32), every variant
  // against the same input:
  //
  //     portable  87 ms   subgroup-key32 184   subgroup-16x64 263
  //     subgroup-32x64 284   subgroup-8x64 275   subgroup-64x64 301
  //     subgroup-4x8 333
  //
  // They agree to 1.3e-6, which is reassociation and not a disagreement. In a
  // whole evoformer block at 512 MSA rows the default costs 302.8 ms against
  // 198.8: column attention 128.9 against 33.9, row attention 17.7 against 5.1.
  // tools/gpu/check-attention-variants.js is that measurement and re-runs it.
  //
  // 🔴 THE SUBGROUP KERNELS ARE KEPT AND STILL SELECTABLE. They are
  // differentially tested, this is one device, and `requested` still names any
  // of them - what changes is only which one is picked when nobody asks.
  const variant = requested === "auto"
    ? (headDim % 4 === 0 ? "portable"
      : subgroup64 ? "subgroup-key32" : subgroup ? "subgroup-4x8" : "portable")
    : requested;
  if (variant.startsWith("subgroup-") && variant !== "subgroup-4x8" && !subgroup64) {
    throw new Error(`the ${variant} attention kernel is unsupported by this device`);
  }
  if (variant === "subgroup-4x8" && !subgroup) {
    throw new Error("the subgroup-4x8 attention kernel is unsupported by this device");
  }
  if (variant === "subgroup-key32") {
    return {
      cacheKey: "attention:flash-subgroup-key32", shader: ATTENTION_SUBGROUP_KEY32_SHADER,
      queryTile: 8, variant,
    };
  }
  const tiled = variant === "subgroup-8x16"
    ? { queryTile: 8, shader: ATTENTION_SUBGROUP_8X16_SHADER }
    : variant === "subgroup-8x32"
      ? { queryTile: 8, shader: ATTENTION_SUBGROUP_8X32_SHADER }
      : variant === "subgroup-8x64"
        ? { queryTile: 8, shader: ATTENTION_SUBGROUP_8X64_SHADER }
        : variant === "subgroup-16x64"
          ? { queryTile: 16, shader: ATTENTION_SUBGROUP_16X64_SHADER }
          : variant === "subgroup-32x64"
            ? { queryTile: 32, shader: ATTENTION_SUBGROUP_32X64_SHADER }
            : variant === "subgroup-64x64"
              ? { queryTile: 64, shader: ATTENTION_SUBGROUP_64X64_SHADER }
              : undefined;
  if (tiled !== undefined) {
    return {
      cacheKey: `attention:flash-${variant}`, shader: tiled.shader,
      queryTile: tiled.queryTile, variant,
    };
  }
  if (variant === "subgroup-4x8") {
    return {
      cacheKey: "attention:flash-subgroup4x8", shader: ATTENTION_SUBGROUP_FLASH_SHADER,
      queryTile: 4, variant,
    };
  }
  // ...THE DEFAULT IS THE REGISTER-RESIDENT KERNEL, whenever the head divides
  // into vec4s - which it does for every attention in this model. queryTile is
  // 64 because that is the workgroup size, and the caller's dispatch already
  // divides the query count by it; batch and head stay the y and z dimensions.
  if (headDim % 4 === 0) {
    return {
      cacheKey: `attention:flash-registers-${headDim}`,
      shader: createAttentionRegisterFlashShader(headDim),
      queryTile: 64, variant,
    };
  }
  return { cacheKey: "attention:flash", shader: ATTENTION_FLASH_SHADER, queryTile: 1, variant };
}

export const ATTENTION_OUTPUT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> tile_source: array<f32, 128>;
var<workgroup> tile_weight: array<f32, 256>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let rows = p.batch * p.queries;
  let projected = p.heads * p.head_dim;
  let canonical_row = group.y * 16u + local.y;
  let second_row = canonical_row + 8u;
  let channel = group.x * 32u + local.x;
  let tile_index = local.y * 8u + local.x;
  var result_00 = 0.0; var result_01 = 0.0; var result_02 = 0.0; var result_03 = 0.0;
  var result_10 = 0.0; var result_11 = 0.0; var result_12 = 0.0; var result_13 = 0.0;
  for (var k0 = 0u; k0 < projected; k0 += 8u) {
    let source_k = k0 + local.x;
    let weight_k = k0 + local.y;
    tile_source[tile_index] = 0.0;
    tile_source[tile_index + 64u] = 0.0;
    if (canonical_row < rows && source_k < projected) {
      tile_source[tile_index] = source[canonical_row * projected + source_k];
    }
    if (second_row < rows && source_k < projected) {
      tile_source[tile_index + 64u] = source[second_row * projected + source_k];
    }
    for (var column_block = 0u; column_block < 4u; column_block += 1u) {
      let tile_column = local.x + column_block * 8u;
      let output_channel = channel + column_block * 8u;
      let weight_index = local.y * 32u + tile_column;
      tile_weight[weight_index] = 0.0;
      if (output_channel < p.channels && weight_k < projected) {
        tile_weight[weight_index] = weights[p.output_weight + weight_k * p.channels + output_channel];
      }
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      let x_0 = tile_source[local.y * 8u + k];
      let x_1 = tile_source[local.y * 8u + k + 64u];
      result_00 += x_0 * tile_weight[k * 32u + local.x];
      result_01 += x_0 * tile_weight[k * 32u + local.x + 8u];
      result_02 += x_0 * tile_weight[k * 32u + local.x + 16u];
      result_03 += x_0 * tile_weight[k * 32u + local.x + 24u];
      result_10 += x_1 * tile_weight[k * 32u + local.x];
      result_11 += x_1 * tile_weight[k * 32u + local.x + 8u];
      result_12 += x_1 * tile_weight[k * 32u + local.x + 16u];
      result_13 += x_1 * tile_weight[k * 32u + local.x + 24u];
    }
    workgroupBarrier();
  }
  for (var row_block = 0u; row_block < 2u; row_block += 1u) {
    let row = canonical_row + row_block * 8u;
    if (row < rows) {
      let b = row / p.queries; let q = row % p.queries;
      let output_row = select(row, q * p.batch + b, p.transpose != 0u);
      for (var column_block = 0u; column_block < 4u; column_block += 1u) {
        let output_channel = channel + column_block * 8u;
        if (output_channel < p.channels) {
          var result = select(result_00, result_01, column_block == 1u);
          result = select(result, result_02, column_block == 2u);
          result = select(result, result_03, column_block == 3u);
          if (row_block == 1u) {
            result = select(result_10, result_11, column_block == 1u);
            result = select(result, result_12, column_block == 2u);
            result = select(result, result_13, column_block == 3u);
          }
          output[output_row * p.channels + output_channel] = result + weights[p.output_bias + output_channel];
        }
      }
    }
  }
}`;

/** Same projection as ATTENTION_OUTPUT_SHADER, but commits directly into an existing residual tensor. */
export const ATTENTION_OUTPUT_RESIDUAL_SHADER = ATTENTION_OUTPUT_SHADER.replace(
  "output[output_row * p.channels + output_channel] = result + weights[p.output_bias + output_channel];",
  "output[output_row * p.channels + output_channel] += result + weights[p.output_bias + output_channel];",
);

export class AttentionGpu {
  device;
  allocator;
  pipelines;
  options;

  constructor(device, options = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
    this.options = options;
  }

  async run(input) {
    validate(input);
    const packed = packAttentionWeights(input);
    const flashKernel = selectAttentionFlashKernel(
      this.device, input.channels / input.heads, this.options.flashVariant ?? "auto",
    );
    const [normalize, project, pairProject, flash, outputProject] = await Promise.all([
      this.pipelines.get("attention:normalize", ATTENTION_NORMALIZE_SHADER),
      this.pipelines.get("attention:project", ATTENTION_PROJECT_SHADER),
      this.pipelines.get("attention:pair-bias", ATTENTION_PAIR_BIAS_SHADER),
      this.pipelines.get(flashKernel.cacheKey, flashKernel.shader),
      this.pipelines.get("attention:output", ATTENTION_OUTPUT_SHADER),
    ]);
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const storage = GPUBufferUsage.STORAGE;
    const rows = input.batch * input.queryLength;
    const tensorBytes = rows * input.channels * 4;
    const linearGrid = (elements) => {
      const groups = ceilDivide(elements, 64);
      return [Math.min(groups, GRID_WIDTH), ceilDivide(groups, GRID_WIDTH)];
    };
    try {
      const source = keep(this.allocator.upload("attention.source", input.activations, storage));
      const mask = keep(this.allocator.upload("attention.mask", input.mask, storage));
      const weights = keep(this.allocator.upload("attention.weights", packed.data, storage));
      const params = keep(this.allocator.upload(
        "attention.parameters", createAttentionParameters(input, packed.offsets), GPUBufferUsage.UNIFORM,
      ));
      const queryNormParams = keep(this.allocator.upload("attention.query-norm-parameters", createAttentionNormParameters(
        rows, input.channels, packed.offsets[0], packed.offsets[1], input.transpose === true,
        input.batch, input.queryLength, input.epsilon ?? 1e-5,
      ), GPUBufferUsage.UNIFORM));
      const normalized = keep(this.allocator.allocate("attention.normalized", tensorBytes, storage));
      const query = keep(this.allocator.allocate("attention.query", tensorBytes, storage));
      const key = keep(this.allocator.allocate("attention.key", tensorBytes, storage));
      const value = keep(this.allocator.allocate("attention.value", tensorBytes, storage));
      const gate = keep(this.allocator.allocate("attention.gate", tensorBytes, storage));
      const weighted = keep(this.allocator.allocate("attention.weighted", tensorBytes, storage));
      const output = keep(this.allocator.allocate("attention.output", tensorBytes, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "attention.readback", tensorBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      let pairNormalized = normalized;
      let pairSource;
      let pairNormParams;
      if (input.pairBias?.source === "separate") {
        pairSource = keep(this.allocator.upload("attention.pair-source", input.pairBias.activations, storage));
        pairNormalized = keep(this.allocator.allocate(
          "attention.pair-normalized", input.queryLength * input.queryLength * input.pairBias.channels * 4, storage,
        ));
        pairNormParams = keep(this.allocator.upload("attention.pair-norm-parameters", createAttentionNormParameters(
          input.queryLength * input.queryLength, input.pairBias.channels,
          packed.offsets[9], packed.offsets[10], false, 1, input.queryLength * input.queryLength,
          input.epsilon ?? 1e-5,
        ), GPUBufferUsage.UNIFORM));
      }
      const pairBiasElements = input.pairBias === undefined
        ? 1 : input.heads * input.queryLength * input.queryLength;
      const pairBias = keep(this.allocator.allocate("attention.pair-bias", pairBiasElements * 4, storage));

      const encoder = this.device.createCommandEncoder({ label: "attention" });
      this.device.pushErrorScope("validation");
      const pass = (
        pipeline, buffers, x, y = 1, z = 1,
      ) => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        compute.dispatchWorkgroups(x, y, z);
        compute.end();
      };
      pass(normalize, [source.buffer, weights.buffer, queryNormParams.buffer, normalized.buffer], rows);
      if (pairSource !== undefined && pairNormParams !== undefined) {
        pass(normalize, [pairSource.buffer, weights.buffer, pairNormParams.buffer, pairNormalized.buffer],
          input.queryLength * input.queryLength);
      }
      if (input.pairBias !== undefined) {
        const pairGrid = linearGrid(input.heads * input.queryLength * input.queryLength);
        pass(pairProject, [pairNormalized.buffer, weights.buffer, params.buffer, pairBias.buffer],
          pairGrid[0], pairGrid[1]);
      }
      pass(project, [normalized.buffer, weights.buffer, params.buffer, query.buffer, key.buffer, value.buffer, gate.buffer],
        ceilDivide(input.channels, 16), ceilDivide(rows, 16));
      pass(flash, [query.buffer, key.buffer, value.buffer, gate.buffer, mask.buffer, pairBias.buffer, params.buffer,
        weighted.buffer], ceilDivide(input.queryLength, flashKernel.queryTile),
        input.batch, input.heads);
      pass(outputProject, [weighted.buffer, weights.buffer, params.buffer, output.buffer],
        ceilDivide(input.channels, 32), ceilDivide(rows, 16));
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, tensorBytes);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return { output: result, elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot() };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
