import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;

/**
 * How many MSA rows one outer-product tile may cover.
 *
 * The bounded path holds tileCapacity * length * cOuter * cZ floats at once,
 * and at 1,024 extra rows of a long sequence 32 of them is more than a storage
 * binding may be - so the tile shrinks to whatever does bind. Thirty-two is
 * still the ceiling; this only ever lowers it.
 */
export const OUTER_PRODUCT_MEAN_TILE_SEQUENCES = 32;

export function outerProductMeanTileCapacity(input, maxStorageBufferBindingSize) {
  const values = [input.sequences, input.length, input.cOuter, input.cZ, maxStorageBufferBindingSize];
  if (!values.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("outer-product tile dimensions and binding limit must be positive safe integers");
  }
  const bytesPerSequence = input.length * input.cOuter * input.cZ * Float32Array.BYTES_PER_ELEMENT;
  const capacity = Math.floor(maxStorageBufferBindingSize / bytesPerSequence);
  if (capacity < 1) {
    throw new RangeError("WebGPU storage binding is too small for one outer-product sequence tile");
  }
  const ceiling = (typeof globalThis !== "undefined"
    && Number.isFinite(globalThis.__OPM_TILE_CAP__))
    ? globalThis.__OPM_TILE_CAP__ : OUTER_PRODUCT_MEAN_TILE_SEQUENCES;
  return Math.min(input.sequences, ceiling, capacity);
}
const ceilDivide = (value, divisor) => Math.ceil(value / divisor);

function validate(input) {
  const { sequences, length, cM, cOuter, cZ, activations, mask, weights } = input;
  if (![sequences, length, cM, cOuter, cZ].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("outer product mean dimensions must be positive safe integers");
  }
  const expected = [
    ["activations", activations, sequences * length * cM],
    ["mask", mask, sequences * length],
    ["layerNormScale", weights.layerNormScale, cM],
    ["layerNormOffset", weights.layerNormOffset, cM],
    ["leftWeight", weights.leftWeight, cM * cOuter],
    ["leftBias", weights.leftBias, cOuter],
    ["rightWeight", weights.rightWeight, cM * cOuter],
    ["rightBias", weights.rightBias, cOuter],
    ["outputWeight", weights.outputWeight, cOuter * cOuter * cZ],
    ["outputBias", weights.outputBias, cZ],
  ];
  for (const [name, value, size] of expected) {
    if (value.length !== size) throw new RangeError(`${name} has ${value.length} values; expected ${size}`);
  }
}

export function packOuterProductMeanWeights(input) {
  const tensors = [
    input.weights.layerNormScale, input.weights.layerNormOffset,
    input.weights.leftWeight, input.weights.leftBias,
    input.weights.rightWeight, input.weights.rightBias,
    input.weights.outputWeight, input.weights.outputBias,
  ];
  const offsets = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const data = new Float32Array(size);
  tensors.forEach((tensor, index) => data.set(tensor, offsets[index]));
  return { data, offsets };
}

export function createOuterProductMeanParameters(input, offsets) {
  const buffer = new ArrayBuffer(64);
  const view = new DataView(buffer);
  const integers = [input.sequences, input.length, input.cM, input.cOuter, input.cZ, ...offsets];
  integers.forEach((value, index) => view.setUint32(index * 4, value, true));
  view.setFloat32(52, input.layerNormEpsilon ?? 1e-5, true);
  view.setFloat32(56, input.normalizationEpsilon ?? 1e-3, true);
  return new Uint8Array(buffer);
}

const COMMON = `
struct Parameters {
  sequences: u32,
  length: u32,
  c_m: u32,
  c_outer: u32,
  c_z: u32,
  layer_norm_scale: u32,
  layer_norm_offset: u32,
  left_weight: u32,
  left_bias: u32,
  right_weight: u32,
  right_bias: u32,
  output_weight: u32,
  output_bias: u32,
  layer_norm_epsilon: f32,
  normalization_epsilon: f32,
  padding: u32,
};
const GRID_WIDTH: u32 = 32768u;
`;

export const OUTER_PRODUCT_MEAN_NORMALIZE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let rows = p.sequences * p.length;
  // ...FOLDED, for the same reason as the transition's: a row per sequence and
  // residue is far more than a dispatch may be wide at any real MSA depth.
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= rows) { return; }
  let base = row * p.c_m;
  var sum = 0.0;
  for (var c = local.x; c < p.c_m; c += 64u) { sum += source[base + c]; }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(p.c_m); }
  workgroupBarrier();
  var squared = 0.0;
  for (var c = local.x; c < p.c_m; c += 64u) {
    let centered = source[base + c] - row_mean[0];
    squared += centered * centered;
  }
  partial[local.x] = squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(p.c_m) + p.layer_norm_epsilon);
  for (var c = local.x; c < p.c_m; c += 64u) {
    output[base + c] = (source[base + c] - row_mean[0]) * inverse_std
      * weights[p.layer_norm_scale + c] + weights[p.layer_norm_offset + c];
  }
}`;

export const OUTER_PRODUCT_MEAN_PROJECT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> left: array<f32>;
@group(0) @binding(5) var<storage, read_write> right: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  let rows = p.sequences * p.length;
  if (index >= rows * p.c_outer) { return; }
  let row = index / p.c_outer;
  let outer = index % p.c_outer;
  var left_value = weights[p.left_bias + outer];
  var right_value = weights[p.right_bias + outer];
  for (var c = 0u; c < p.c_m; c += 1u) {
    let value = source[row * p.c_m + c];
    left_value += value * weights[p.left_weight + c * p.c_outer + outer];
    right_value += value * weights[p.right_weight + c * p.c_outer + outer];
  }
  left[index] = mask[row] * left_value;
  right[index] = mask[row] * right_value;
}`;

export const OUTER_PRODUCT_MEAN_INTERMEDIATE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  let elements = p.sequences * p.length * p.c_outer * p.c_z;
  if (index >= elements) { return; }
  let z = index % p.c_z;
  let outer_right = (index / p.c_z) % p.c_outer;
  let residue = (index / (p.c_z * p.c_outer)) % p.length;
  let sequence = index / (p.c_z * p.c_outer * p.length);
  var value = 0.0;
  for (var outer_left = 0u; outer_left < p.c_outer; outer_left += 1u) {
    let left_index = (sequence * p.length + residue) * p.c_outer + outer_left;
    let weight_index = ((outer_left * p.c_outer + outer_right) * p.c_z) + z;
    value += left[left_index] * weights[p.output_weight + weight_index];
  }
  output[index] = value;
}`;

export const OUTER_PRODUCT_MEAN_OUTPUT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> right: array<f32>;
@group(0) @binding(1) var<storage, read> intermediate: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.c_z) { return; }
  let z = index % p.c_z;
  let pair = index / p.c_z;
  let i = pair / p.length;
  let j = pair % p.length;
  var value = weights[p.output_bias + z];
  var count = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    count += mask[sequence * p.length + i] * mask[sequence * p.length + j];
    for (var outer = 0u; outer < p.c_outer; outer += 1u) {
      let temp_index = (((sequence * p.length + i) * p.c_outer + outer) * p.c_z) + z;
      let right_index = (sequence * p.length + j) * p.c_outer + outer;
      value += intermediate[temp_index] * right[right_index];
    }
  }
  output[index] = value / (p.normalization_epsilon + count);
}`;

const OPM_TILE_COMMON = `${COMMON}
struct TileParameters { offset: u32, count: u32, padding0: u32, padding1: u32 };
`;

export const OUTER_PRODUCT_MEAN_TILE_INTERMEDIATE_SHADER = `${OPM_TILE_COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<uniform> tile: TileParameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  let elements = tile.count * p.length * p.c_outer * p.c_z;
  if (index >= elements) { return; }
  let z = index % p.c_z; let outer_right = (index / p.c_z) % p.c_outer;
  let residue = (index / (p.c_z * p.c_outer)) % p.length;
  let sequence = index / (p.c_z * p.c_outer * p.length);
  var value = 0.0;
  for (var outer_left = 0u; outer_left < p.c_outer; outer_left += 1u) {
    let left_index = ((tile.offset + sequence) * p.length + residue) * p.c_outer + outer_left;
    let weight_index = (outer_left * p.c_outer + outer_right) * p.c_z + z;
    value += left[left_index] * weights[p.output_weight + weight_index];
  }
  output[index] = value;
}`;

export const OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER = `${OPM_TILE_COMMON}
@group(0) @binding(0) var<storage, read> right: array<f32>;
@group(0) @binding(1) var<storage, read> intermediate: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<uniform> tile: TileParameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.c_z) { return; }
  let z = index % p.c_z; let pair = index / p.c_z;
  let i = pair / p.length; let j = pair % p.length; var value = 0.0;
  for (var sequence = 0u; sequence < tile.count; sequence += 1u) {
    for (var outer = 0u; outer < p.c_outer; outer += 1u) {
      let temp_index = (((sequence * p.length + i) * p.c_outer + outer) * p.c_z) + z;
      let right_index = (((tile.offset + sequence) * p.length + j) * p.c_outer) + outer;
      value += intermediate[temp_index] * right[right_index];
    }
  }
  output[index] += value;
}`;

export const OUTER_PRODUCT_MEAN_FINALIZE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> mask: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.c_z) { return; }
  let z = index % p.c_z; let pair = index / p.c_z; let i = pair / p.length; let j = pair % p.length;
  var count = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    count += mask[sequence * p.length + i] * mask[sequence * p.length + j];
  }
  output[index] = (output[index] + weights[p.output_bias + z]) / (p.normalization_epsilon + count);
}`;

/**
 * Per-residue marginals over the MSA, for the cov-masked pair entries.
 *
 * left_sum[i, o]   = sum_s left[s, i, o]
 * right_mean[j, o] = sum_s right[s, j, o] / (eps + sum_s mask[s, j])
 *
 * Their outer product is what sum_s left[s,i] (x) right[s,j] would be if the
 * two sides were independent across sequences - the first-order term with the
 * covariance removed. One dispatch over [L, c_outer]; the sequence walk here is
 * O(D) per element against the O(D) per (i,j) pair it lets the contraction skip.
 */
export const OUTER_PRODUCT_MEAN_MARGINALS_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> left_sum: array<f32>;
@group(0) @binding(5) var<storage, read_write> right_mean: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.c_outer) { return; }
  let outer = index % p.c_outer;
  let residue = index / p.c_outer;
  var left_total = 0.0;
  var right_total = 0.0;
  var covered = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    let slot = (sequence * p.length + residue) * p.c_outer + outer;
    left_total += left[slot];
    right_total += right[slot];
    covered += mask[sequence * p.length + residue];
  }
  left_sum[index] = left_total;
  right_mean[index] = right_total / (p.normalization_epsilon + covered);
}`;

/**
 * The tiled accumulation, skipping the pairs cov_mask zeroes. Their marginal
 * term is added once by the shader below rather than per tile.
 */
export const OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_COV_MASKED_SHADER =
  OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER
    .replace(
      "@group(0) @binding(4) var<storage, read_write> output: array<f32>;",
      "@group(0) @binding(4) var<storage, read_write> output: array<f32>;\n"
      + "@group(0) @binding(5) var<storage, read> cov_mask: array<f32>;",
    )
    .replace(
      "let i = pair / p.length; let j = pair % p.length; var value = 0.0;",
      "let i = pair / p.length; let j = pair % p.length; var value = 0.0;\n"
      + "  if (cov_mask[pair] == 0.0) { return; }",
    );

/**
 * The marginal term for the cov-masked pairs, on the tiled path.
 *
 * The outer-first path folds this into its contraction, where the [i,j,o,o]
 * cube it writes is the natural place for it. The tiled path never forms that
 * cube - it accumulates straight into [i,j,z] - so the marginal product is
 * contracted with output_w here and added before finalize applies the bias and
 * the normalisation, which leaves both of those shared between the two paths.
 */
export const OUTER_PRODUCT_MEAN_TILE_MARGINAL_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> left_sum: array<f32>;
@group(0) @binding(1) var<storage, read> right_mean: array<f32>;
@group(0) @binding(2) var<storage, read> cov_mask: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.c_z) { return; }
  let z = index % p.c_z;
  let pair = index / p.c_z;
  if (cov_mask[pair] != 0.0) { return; }
  let i = pair / p.length;
  let j = pair % p.length;
  var value = 0.0;
  for (var outer_left = 0u; outer_left < p.c_outer; outer_left += 1u) {
    let left_value = left_sum[i * p.c_outer + outer_left];
    for (var outer_right = 0u; outer_right < p.c_outer; outer_right += 1u) {
      let weight_index = (outer_left * p.c_outer + outer_right) * p.c_z + z;
      value += left_value * right_mean[j * p.c_outer + outer_right]
        * weights[p.output_weight + weight_index];
    }
  }
  output[index] += value;
}`;

// Algebraically identical to AF2's original einsum order: first contract the
// MSA sequence axis into [L, L, c_outer, c_outer], then apply output_w once.
export const OUTER_PRODUCT_MEAN_CONTRACT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> outer: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  let elements = p.length * p.length * p.c_outer * p.c_outer;
  if (index >= elements) { return; }
  let outer_right = index % p.c_outer;
  let outer_left = (index / p.c_outer) % p.c_outer;
  let pair = index / (p.c_outer * p.c_outer);
  let i = pair / p.length;
  let j = pair % p.length;
  var value = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    value += left[(sequence * p.length + i) * p.c_outer + outer_left]
      * right[(sequence * p.length + j) * p.c_outer + outer_right];
  }
  outer[index] = value;
}`;

/**
 * The contraction with ColabDesign's `cov_mask` applied: pairs the mask zeroes
 * take the marginal product instead of the sequence sum, and cost no walk at
 * all. For an N-mer only the N diagonal blocks keep the sum, so this is where
 * the masking pays for itself twice - it removes the invented coupling AND the
 * work that produced it.
 */
export const OUTER_PRODUCT_MEAN_CONTRACT_COV_MASKED_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> outer: array<f32>;
@group(0) @binding(4) var<storage, read> cov_mask: array<f32>;
@group(0) @binding(5) var<storage, read> left_sum: array<f32>;
@group(0) @binding(6) var<storage, read> right_mean: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  let elements = p.length * p.length * p.c_outer * p.c_outer;
  if (index >= elements) { return; }
  let outer_right = index % p.c_outer;
  let outer_left = (index / p.c_outer) % p.c_outer;
  let pair = index / (p.c_outer * p.c_outer);
  let i = pair / p.length;
  let j = pair % p.length;
  if (cov_mask[pair] == 0.0) {
    outer[index] = left_sum[i * p.c_outer + outer_left] * right_mean[j * p.c_outer + outer_right];
    return;
  }
  var value = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    value += left[(sequence * p.length + i) * p.c_outer + outer_left]
      * right[(sequence * p.length + j) * p.c_outer + outer_right];
  }
  outer[index] = value;
}`;

export const OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> outer: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.c_z) { return; }
  let z = index % p.c_z;
  let pair = index / p.c_z;
  let i = pair / p.length;
  let j = pair % p.length;
  var value = weights[p.output_bias + z];
  for (var outer_left = 0u; outer_left < p.c_outer; outer_left += 1u) {
    for (var outer_right = 0u; outer_right < p.c_outer; outer_right += 1u) {
      let outer_index = (pair * p.c_outer + outer_left) * p.c_outer + outer_right;
      let weight_index = (outer_left * p.c_outer + outer_right) * p.c_z + z;
      value += outer[outer_index] * weights[p.output_weight + weight_index];
    }
  }
  var count = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    count += mask[sequence * p.length + i] * mask[sequence * p.length + j];
  }
  output[index] = value / (p.normalization_epsilon + count);
}`;

export const OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_RESIDUAL_SHADER = OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_SHADER.replace(
  "output[index] = value / (p.normalization_epsilon + count);",
  "output[index] += value / (p.normalization_epsilon + count);",
);

const OUTER_FIRST_LIMIT_BYTES = 64 * 1024 * 1024;

export function useOuterFirstContraction(input
  ) {
  const bytes = input.length * input.length * input.cOuter * input.cOuter * 4;
  return input.sequences >= input.cOuter && bytes <= OUTER_FIRST_LIMIT_BYTES;
}

export class OuterProductMeanGpu {
  device;
  allocator;
  pipelines;

  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input) {
    validate(input);
    const packed = packOuterProductMeanWeights(input);
    const outerFirst = useOuterFirstContraction(input);
    const [normalize, project, intermediatePipeline, accumulatePipeline, finalizePipeline,
      contractPipeline, projectOutputPipeline] = await Promise.all([
      this.pipelines.get("opm:normalize", OUTER_PRODUCT_MEAN_NORMALIZE_SHADER),
      this.pipelines.get("opm:project", OUTER_PRODUCT_MEAN_PROJECT_SHADER),
      this.pipelines.get("opm:tile-intermediate", OUTER_PRODUCT_MEAN_TILE_INTERMEDIATE_SHADER),
      this.pipelines.get("opm:tile-accumulate", OUTER_PRODUCT_MEAN_TILE_ACCUMULATE_SHADER),
      this.pipelines.get("opm:finalize", OUTER_PRODUCT_MEAN_FINALIZE_SHADER),
      this.pipelines.get("opm:contract", OUTER_PRODUCT_MEAN_CONTRACT_SHADER),
      this.pipelines.get("opm:project-output", OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_SHADER),
    ]);
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const rows = input.sequences * input.length;
    const pairElements = input.length * input.length * input.cZ;
    // ONE WORKGROUP PER ROW, folded the same way linearGrid folds elements.
    const rowGrid = (rows) => [Math.min(rows, GRID_WIDTH), Math.ceil(rows / GRID_WIDTH)];
    const linearGrid = (elements) => {
      const groups = ceilDivide(elements, 64);
      return [Math.min(groups, GRID_WIDTH), ceilDivide(groups, GRID_WIDTH)];
    };
    try {
      const source = keep(this.allocator.upload("opm.source", input.activations, storage));
      const mask = keep(this.allocator.upload("opm.mask", input.mask, storage));
      const weights = keep(this.allocator.upload("opm.weights", packed.data, storage));
      const params = keep(this.allocator.upload(
        "opm.parameters", createOuterProductMeanParameters(input, packed.offsets), GPUBufferUsage.UNIFORM,
      ));
      const normalized = keep(this.allocator.allocate("opm.normalized", rows * input.cM * 4, storage));
      const left = keep(this.allocator.allocate("opm.left", rows * input.cOuter * 4, storage));
      const right = keep(this.allocator.allocate("opm.right", rows * input.cOuter * 4, storage));
      const tileCapacity = outerProductMeanTileCapacity(
        input, this.device.limits.maxStorageBufferBindingSize);
      const intermediateElements = outerFirst
        ? input.length * input.length * input.cOuter * input.cOuter
        : tileCapacity * input.length * input.cOuter * input.cZ;
      const intermediate = keep(this.allocator.allocate("opm.intermediate", intermediateElements * 4, storage));
      const output = keep(this.allocator.allocate(
        "opm.output", pairElements * 4, storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      ));
      const readback = keep(this.allocator.allocate(
        "opm.readback", pairElements * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      const encoder = this.device.createCommandEncoder({ label: "outer-product-mean" });
      this.device.pushErrorScope("validation");
      const pass = (pipeline, buffers, x, y = 1) => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        compute.dispatchWorkgroups(x, y);
        compute.end();
      };
      const normalizeGrid = rowGrid(rows);
      pass(normalize, [source.buffer, weights.buffer, params.buffer, normalized.buffer],
        normalizeGrid[0], normalizeGrid[1]);
      const projectGrid = linearGrid(rows * input.cOuter);
      pass(project, [normalized.buffer, mask.buffer, weights.buffer, params.buffer, left.buffer, right.buffer],
        projectGrid[0], projectGrid[1]);
      const outputGrid = linearGrid(pairElements);
      if (outerFirst) {
        const outerGrid = linearGrid(intermediateElements);
        pass(contractPipeline, [left.buffer, right.buffer, params.buffer, intermediate.buffer],
          outerGrid[0], outerGrid[1]);
        pass(projectOutputPipeline, [intermediate.buffer, mask.buffer, weights.buffer, params.buffer, output.buffer],
          outputGrid[0], outputGrid[1]);
      } else {
        encoder.clearBuffer(output.buffer);
        for (let offset = 0; offset < input.sequences; offset += tileCapacity) {
          const count = Math.min(tileCapacity, input.sequences - offset);
          const tileParams = keep(this.allocator.upload(
            `opm.tile-${offset}`, new Uint32Array([offset, count, 0, 0]), GPUBufferUsage.UNIFORM,
          ));
          const intermediateGrid = linearGrid(count * input.length * input.cOuter * input.cZ);
          pass(intermediatePipeline, [left.buffer, weights.buffer, params.buffer, tileParams.buffer, intermediate.buffer],
            intermediateGrid[0], intermediateGrid[1]);
          pass(accumulatePipeline, [right.buffer, intermediate.buffer, params.buffer, tileParams.buffer, output.buffer],
            outputGrid[0], outputGrid[1]);
        }
        pass(finalizePipeline, [mask.buffer, weights.buffer, params.buffer, output.buffer],
          outputGrid[0], outputGrid[1]);
      }
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, pairElements * 4);
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
