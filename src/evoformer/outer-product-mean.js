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
  // ...see MAX_C_OUTER: the contraction kernel's arrays are sized for it.
  if (cOuter > MAX_C_OUTER) {
    throw new RangeError(`cOuter ${cOuter} exceeds the ${MAX_C_OUTER} the contraction stages for`);
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

/**
 * The largest c_outer the contraction kernel stages for, and the cells a lane
 * then owns.
 *
 * 🔴 THE SHADER IS ONE PIPELINE FOR EVERY SHAPE - its dimensions come from a
 * uniform, not from generation - so its workgroup array and its accumulator
 * array have to be sized for the largest c_outer that can arrive, and the loop
 * bounds have to be the real ones. AlphaFold's c_outer is 32; anything larger
 * would overrun `totals` silently, so it is checked rather than assumed.
 */
const MAX_C_OUTER = 32;
const MAX_CELLS_PER_LANE = (MAX_C_OUTER * MAX_C_OUTER) / 64;

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

// Algebraically identical to AF2's original einsum order: first contract the
// MSA sequence axis into [L, L, c_outer, c_outer], then apply output_w once.
/**
 * The sequence contraction, generated for one c_outer.
 *
 * 🔴 IT IS A FUNCTION AND NOT A CONSTANT BECAUSE THE ACCUMULATORS HAVE TO BE
 * REGISTERS. A thread owns c_outer^2 / 64 cells and carries a running sum for
 * each across the whole sweep; written as an ARRAY indexed by a loop variable,
 * WGSL puts that in spillable local memory and the kernel measured 38 ms
 * against the 29 it replaced. Named variables need the count at generation
 * time, and c_outer arrives in a uniform - so the shader is generated per
 * c_outer and the pipeline cache key carries it. AF3's outer product learned
 * the same thing the same way; see the note in
 * src/af3/outer-product-mean-webgpu.js.
 *
 * 🔴 AND A CHUNK OF SEQUENCES IS STAGED, which is what the cells buy. A thread
 * that owned ONE cell re-read the same two 32-float slices as its 1023
 * neighbours: two global reads for every multiply-add, 3.6 billion reads a
 * block at 512 rows, which at the 111 billion a second tools/gpu/probe-alu.js
 * measures is most of what this cost. Staged, a chunk's reads are shared by the
 * whole workgroup and a thread's sixteen cells come out of eight of them.
 */
export function createOuterProductMeanContractShader(cOuter) {
  const cells = cOuter * cOuter;
  // ...rounded up, and each slot guarded: AlphaFold's c_outer is 32 and gives
  // exactly sixteen a lane, but the checkers use ragged shapes on purpose.
  const perLane = Math.ceil(cells / 64);
  const chunk = 8;
  const overCells = (body) =>
    Array.from({ length: perLane }, (_, slot) => body(slot)).join("\n    ");
  return `${COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> outer: array<f32>;

const OPM_CHUNK: u32 = ${chunk}u;
const C_OUTER: u32 = ${cOuter}u;
const CELLS: u32 = ${cells}u;

var<workgroup> left_tile: array<f32, ${chunk * cOuter}>;
var<workgroup> right_tile: array<f32, ${chunk * cOuter}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let pair = group.x + group.y * GRID_WIDTH;
  if (pair >= p.length * p.length) { return; }
  let i = pair / p.length;
  let j = pair % p.length;
  let local = local_id.x;

  // The cells this lane owns, strided so 64 lanes cover the grid.
  ${overCells((slot) => `let cell${slot} = local + ${slot}u * 64u;
    let live${slot} = cell${slot} < CELLS;
    let cl${slot} = select(0u, cell${slot} / C_OUTER, live${slot});
    let cr${slot} = select(0u, cell${slot} % C_OUTER, live${slot});
    var total${slot} = 0.0;`)}

  for (var s0 = 0u; s0 < p.sequences; s0 += OPM_CHUNK) {
    workgroupBarrier();
    for (var index = local; index < OPM_CHUNK * C_OUTER; index += 64u) {
      let s = s0 + index / C_OUTER;
      let o = index % C_OUTER;
      var l = 0.0;
      var r = 0.0;
      if (s < p.sequences) {
        l = left[(s * p.length + i) * C_OUTER + o];
        r = right[(s * p.length + j) * C_OUTER + o];
      }
      left_tile[index] = l;
      right_tile[index] = r;
    }
    workgroupBarrier();

    let available = min(OPM_CHUNK, p.sequences - s0);
    for (var t = 0u; t < available; t += 1u) {
      let base = t * C_OUTER;
      ${overCells((slot) =>
        `total${slot} += left_tile[base + cl${slot}] * right_tile[base + cr${slot}];`)}
    }
  }

  ${overCells((slot) =>
    `if (live${slot}) { outer[pair * CELLS + cell${slot}] = total${slot}; }`)}
}`;
}

/**
 * The output projection, generated for one c_outer.
 *
 * 🔴 IT RECOMPUTED THE DENOMINATOR ONCE PER OUTPUT CHANNEL. The count of
 * sequences covering both tokens depends on the PAIR alone, and a thread owned
 * a (pair, channel) - so 128 threads each swept all 512 sequences for the same
 * number: 456 million redundant reads a block at depth, about a third of what
 * this kernel cost. One workgroup a pair computes it once, cooperatively.
 *
 * 🔴 AND EVERY THREAD RE-READ THE WHOLE OUTER TENSOR. A pair's c_outer^2 values
 * are the same for all 128 channels; staged in workgroup memory they are one
 * global read each rather than 128, which halves what is left.
 */
export function createOuterProductMeanProjectOutputShader(cOuter, residual = false) {
  const cells = cOuter * cOuter;
  return `${COMMON}
@group(0) @binding(0) var<storage, read> outer: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const CELLS: u32 = ${cells}u;
var<workgroup> staged: array<f32, ${cells}>;
var<workgroup> reduce: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let pair = group.x + group.y * GRID_WIDTH;
  if (pair >= p.length * p.length) { return; }
  let i = pair / p.length;
  let j = pair % p.length;
  let local = local_id.x;

  for (var cell = local; cell < CELLS; cell += 64u) {
    staged[cell] = outer[pair * CELLS + cell];
  }
  // ...the denominator, once for the pair rather than once per channel.
  var count = 0.0;
  for (var sequence = local; sequence < p.sequences; sequence += 64u) {
    count += mask[sequence * p.length + i] * mask[sequence * p.length + j];
  }
  reduce[local] = count;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let scale = 1.0 / (p.normalization_epsilon + reduce[0]);

  for (var z = local; z < p.c_z; z += 64u) {
    var value = weights[p.output_bias + z];
    for (var cell = 0u; cell < CELLS; cell += 1u) {
      // ...read once for the pair, used by every channel.
      value += staged[cell] * weights[p.output_weight + cell * p.c_z + z];
    }
    output[pair * p.c_z + z] ${residual ? "+=" : "="} value * scale;
  }
}`;
}

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
      this.pipelines.get(`opm:contract:${input.cOuter}`,
        createOuterProductMeanContractShader(input.cOuter)),
      this.pipelines.get(`opm:project-output:${input.cOuter}`,
        createOuterProductMeanProjectOutputShader(input.cOuter)),
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
        // ...one workgroup per PAIR now, not per element; see the note on the
        // contraction kernel.
        const pairs = input.length * input.length;
        const outerGrid = [Math.min(pairs, GRID_WIDTH), Math.ceil(pairs / GRID_WIDTH)];
        pass(contractPipeline, [left.buffer, right.buffer, params.buffer, intermediate.buffer],
          outerGrid[0], outerGrid[1]);
        // ...one workgroup per PAIR now; see the note on the kernel.
        pass(projectOutputPipeline,
          [intermediate.buffer, mask.buffer, weights.buffer, params.buffer, output.buffer],
          outerGrid[0], outerGrid[1]);
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
