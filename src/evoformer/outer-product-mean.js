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

/**
 * The tile one workgroup of the left/right projection computes, and the only
 * statement of it - both call sites dispatch with these.
 */
export const OPM_PROJECT_TILE = { lanesX: 8, lanesY: 8, rowsPerLane: 4, columnsPerLane: 4 };
export const opmProjectTileRows = (tile = OPM_PROJECT_TILE) => tile.lanesY * tile.rowsPerLane;
export const opmProjectTileColumns = (tile = OPM_PROJECT_TILE) => tile.lanesX * tile.columnsPerLane;

/**
 * The left and right projections of the normalised MSA, in one pass.
 *
 * 🔴 IT WAS ONE THREAD PER (row, channel) AND IT WALKED THE WHOLE CONTRACTION.
 * Every thread read `source[row * c_m + c]` for all 256 channels - and the 32
 * threads holding that row's 32 outer channels each read the SAME value, so a
 * row's activation was fetched from global memory 32 times over - plus one
 * weight for the left projection and one for the right. Three global reads and
 * two multiply-adds per step of c: 0.4 useful operations an instruction, on a
 * machine that issues about 640 billion a second whatever their width. It
 * measured 4.16 ms of a 113 ms block at 512 MSA rows, which is 238 GFLOP/s
 * where every tuned kernel around it runs at 1000.
 *
 * It is a plain matrix multiply and it is now tiled like one: a workgroup takes
 * 32 rows by 32 outer channels, stages the source TRANSPOSED so one vec4 read
 * serves four rows, and stages the weights per thread so a lane's four strided
 * channels are one vec4. Per step of c an invocation issues three reads and
 * eight vector multiply-adds for thirty-two products - 2.9 an instruction.
 *
 * 🔴 LEFT AND RIGHT ARE THE SAME CONTRACTION OVER THE SAME SOURCE, which is why
 * they stay one kernel rather than becoming two. The source tile and its whole
 * staging cost are shared; splitting them would read it twice.
 */
export function createOuterProductMeanProjectShader(tile = OPM_PROJECT_TILE) {
  const { lanesX, lanesY, rowsPerLane, columnsPerLane } = tile;
  if (rowsPerLane % 4 !== 0 || columnsPerLane % 4 !== 0) {
    throw new RangeError("OPM project tile must be a multiple of 4 each way");
  }
  const lanes = lanesX * lanesY;
  const step = lanesY;
  const tileRows = lanesY * rowsPerLane;
  const tileColumns = lanesX * columnsPerLane;
  const rowVectors = tileRows / 4;
  const columnVectors = columnsPerLane / 4;
  const sourceTasks = step * rowVectors;
  const sourcePerLane = Math.ceil(sourceTasks / lanes);

  const declare = [];
  for (const side of ["left", "right"]) {
    for (let r = 0; r < rowsPerLane; r += 1) {
      for (let v = 0; v < columnVectors; v += 1) {
        // ...the bias rides in as the accumulator's initial value, read once a
        // lane rather than once a row.
        declare.push(`  var acc_${side}_${r}_${v} = bias_${side}_${v};`);
      }
    }
  }
  const bias = [];
  for (const side of ["left", "right"]) {
    for (let v = 0; v < columnVectors; v += 1) {
      bias.push(`  var bias_${side}_${v} = vec4<f32>(0.0);
  for (var j = 0u; j < 4u; j += 1u) {
    let outer = column_origin + (${v}u * 4u + j) * ${lanesX}u;
    if (outer < p.c_outer) { bias_${side}_${v}[j] = weights[p.${side}_bias + outer]; }
  }`);
    }
  }

  const inner = [];
  for (let k = 0; k < step; k += 1) {
    for (let g = 0; g < rowsPerLane / 4; g += 1) {
      inner.push(`    let s_${k}_${g} = tile_source[${k * rowVectors}u + local_y * ${rowsPerLane / 4}u + ${g}u];`);
    }
    for (const side of ["left", "right"]) {
      for (let v = 0; v < columnVectors; v += 1) {
        inner.push(`    let w_${side}_${k}_${v} = tile_${side}[${k * lanesX * columnVectors}u + local_x * ${columnVectors}u + ${v}u];`);
      }
    }
    for (const side of ["left", "right"]) {
      for (let r = 0; r < rowsPerLane; r += 1) {
        for (let v = 0; v < columnVectors; v += 1) {
          inner.push(`    acc_${side}_${r}_${v} += s_${k}_${Math.floor(r / 4)}[${r % 4}u] * w_${side}_${k}_${v};`);
        }
      }
    }
  }

  const stageSource = [];
  for (let n = 0; n < sourcePerLane; n += 1) {
    const task = n === 0 ? "linear_lane" : `(linear_lane + ${n * lanes}u)`;
    const guard = (n + 1) * lanes > sourceTasks ? `if (${task} < ${sourceTasks}u) ` : "";
    stageSource.push(`    ${guard}{
      let task = ${task};
      let row_group = task / ${step}u;
      let k_local = task % ${step}u;
      let c = c0 + k_local;
      let row_base = first_row + row_group * 4u;
      var staged = vec4<f32>(0.0);
      if (c < p.c_m) {
        for (var j = 0u; j < 4u; j += 1u) {
          let row = row_base + j;
          if (row < rows) { staged[j] = source[row * p.c_m + c]; }
        }
      }
      tile_source[k_local * ${rowVectors}u + row_group] = staged;
    }`);
  }

  const stageWeight = [];
  for (const side of ["left", "right"]) {
    for (let v = 0; v < columnVectors; v += 1) {
      stageWeight.push(`    {
      var staged = vec4<f32>(0.0);
      if (weight_c < p.c_m) {
        for (var j = 0u; j < 4u; j += 1u) {
          let outer = column_origin + (${v}u * 4u + j) * ${lanesX}u;
          if (outer < p.c_outer) {
            staged[j] = weights[p.${side}_weight + weight_c * p.c_outer + outer];
          }
        }
      }
      tile_${side}[local_y * ${lanesX * columnVectors}u + local_x * ${columnVectors}u + ${v}u] = staged;
    }`);
    }
  }

  const store = [];
  for (let r = 0; r < rowsPerLane; r += 1) {
    const body = [];
    for (let v = 0; v < columnVectors; v += 1) {
      for (let c = 0; c < 4; c += 1) {
        body.push(`      {
        let outer = column_origin + ${(v * 4 + c) * lanesX}u;
        if (outer < p.c_outer) {
          let index = row_${r} * p.c_outer + outer;
          left[index] = keep_${r} * acc_left_${r}_${v}[${c}u];
          right[index] = keep_${r} * acc_right_${r}_${v}[${c}u];
        }
      }`);
      }
    }
    store.push(`  {
    let row_${r} = first_row + local_y * ${rowsPerLane}u + ${r}u;
    if (row_${r} < rows) {
      let keep_${r} = mask[row_${r}];
${body.join("\n")}
    }
  }`);
  }

  return `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> left: array<f32>;
@group(0) @binding(5) var<storage, read_write> right: array<f32>;

// Transposed: four ROWS to a vector, so one read serves four accumulators.
var<workgroup> tile_source: array<vec4<f32>, ${step * rowVectors}>;
// Laid out per thread: a lane's own strided channels, contiguous where it reads.
var<workgroup> tile_left: array<vec4<f32>, ${step * lanesX * columnVectors}>;
var<workgroup> tile_right: array<vec4<f32>, ${step * lanesX * columnVectors}>;

@compute @workgroup_size(${lanesX}, ${lanesY}, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let rows = p.sequences * p.length;
  // ...FOLDED in x and y, because a row per sequence and residue is far more
  // than a dispatch may be wide at any real MSA depth; the channel tile is z.
  let tile_index = group.x + group.y * GRID_WIDTH;
  let first_row = tile_index * ${tileRows}u;
  if (first_row >= rows) { return; }
  let local_x = local.x;
  let local_y = local.y;
  let linear_lane = local_y * ${lanesX}u + local_x;
  let column_origin = group.z * ${tileColumns}u + local_x;
${bias.join("\n")}
${declare.join("\n")}

  for (var c0 = 0u; c0 < p.c_m; c0 += ${step}u) {
    let weight_c = c0 + local_y;
${stageSource.join("\n")}
${stageWeight.join("\n")}
    workgroupBarrier();
${inner.join("\n")}
    workgroupBarrier();
  }

${store.join("\n")}
}`;
}

export const OUTER_PRODUCT_MEAN_PROJECT_SHADER = createOuterProductMeanProjectShader();

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
  // 🔴 EIGHT IS MEASURED, AND IT WAS A HARDCODED GUESS UNTIL IT WAS. Swept
  // through profile-af2-block.js at 512 MSA rows, in runs where every untouched
  // kernel matched to 0.1 ms - the block and this kernel:
  //
  //     chunk  8   87.49 / 5.34 ms      16   87.89 / 5.33
  //           32   88.39 / 5.67         64   90.21 / 7.82
  //
  // Sixteen ties and everything above it loses, so the constant was right. It
  // is not obvious that it would be: a bigger chunk halves the barriers, and
  // this kernel takes 128 of them per workgroup at 512 rows. It is not what
  // governs.
  const chunk = 8;
  // Both operands are read four channels at a time, so the staged rows are
  // padded up to a whole vector and the tail staged as zero - which contributes
  // exactly zero to a product and needs no guard in the inner loop.
  const vectors = Math.ceil(cOuter / 4);
  const blocks = vectors * vectors;
  const blocksPerLane = Math.ceil(blocks / 64);
  const overBlocks = (body) =>
    Array.from({ length: blocksPerLane }, (_, slot) => body(slot)).join("\n    ");
  return `${COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> outer: array<f32>;

const OPM_CHUNK: u32 = ${chunk}u;
const C_OUTER: u32 = ${cOuter}u;
const CELLS: u32 = ${cells}u;
const VECTORS: u32 = ${vectors}u;
const BLOCKS: u32 = ${blocks}u;

var<workgroup> left_tile: array<vec4<f32>, ${chunk * vectors}>;
var<workgroup> right_tile: array<vec4<f32>, ${chunk * vectors}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let pair = group.x + group.y * GRID_WIDTH;
  if (pair >= p.length * p.length) { return; }
  let i = pair / p.length;
  let j = pair % p.length;
  let local = local_id.x;

  // A lane owns a 4x4 BLOCK of cells, not sixteen scattered ones - see the note
  // above. Four accumulators, each a vector over the right-hand channel.
  ${overBlocks((slot) => `let block${slot} = local + ${slot}u * 64u;
    let live${slot} = block${slot} < BLOCKS;
    let left_block${slot} = select(0u, block${slot} / VECTORS, live${slot});
    let right_block${slot} = select(0u, block${slot} % VECTORS, live${slot});
    var acc${slot}_0 = vec4<f32>(0.0);
    var acc${slot}_1 = vec4<f32>(0.0);
    var acc${slot}_2 = vec4<f32>(0.0);
    var acc${slot}_3 = vec4<f32>(0.0);`)}

  for (var s0 = 0u; s0 < p.sequences; s0 += OPM_CHUNK) {
    workgroupBarrier();
    for (var index = local; index < OPM_CHUNK * VECTORS; index += 64u) {
      let s = s0 + index / VECTORS;
      let v = index % VECTORS;
      var l = vec4<f32>(0.0);
      var r = vec4<f32>(0.0);
      if (s < p.sequences) {
        for (var c = 0u; c < 4u; c += 1u) {
          let o = v * 4u + c;
          if (o < C_OUTER) {
            l[c] = left[(s * p.length + i) * C_OUTER + o];
            r[c] = right[(s * p.length + j) * C_OUTER + o];
          }
        }
      }
      left_tile[index] = l;
      right_tile[index] = r;
    }
    workgroupBarrier();

    let available = min(OPM_CHUNK, p.sequences - s0);
    for (var t = 0u; t < available; t += 1u) {
      let base = t * VECTORS;
      ${overBlocks((slot) => `let lv${slot} = left_tile[base + left_block${slot}];
      let rv${slot} = right_tile[base + right_block${slot}];
      acc${slot}_0 += lv${slot}[0] * rv${slot};
      acc${slot}_1 += lv${slot}[1] * rv${slot};
      acc${slot}_2 += lv${slot}[2] * rv${slot};
      acc${slot}_3 += lv${slot}[3] * rv${slot};`)}
    }
  }

  ${overBlocks((slot) => `if (live${slot}) {
      for (var a = 0u; a < 4u; a += 1u) {
        let cl = left_block${slot} * 4u + a;
        if (cl < C_OUTER) {
          var row = acc${slot}_0;
          if (a == 1u) { row = acc${slot}_1; }
          if (a == 2u) { row = acc${slot}_2; }
          if (a == 3u) { row = acc${slot}_3; }
          for (var b = 0u; b < 4u; b += 1u) {
            let cr = right_block${slot} * 4u + b;
            if (cr < C_OUTER) { outer[pair * CELLS + cl * C_OUTER + cr] = row[b]; }
          }
        }
      }
    }`)}
}`;
}

/**
 * How many pairs one output-projection workgroup carries.
 *
 * 🔴 IT EXISTS TO AMORTISE THE WEIGHT READ AND NOTHING ELSE. Every pair
 * contracts its c_outer^2 cells against the SAME output matrix, so a workgroup
 * holding P pairs reads that matrix once for all of them - one global read
 * buying P multiply-adds where it used to buy one. The cost is P * CELLS floats
 * of workgroup memory, which is occupancy.
 *
 * Swept at 512 MSA rows, as this kernel's share of the block, with the
 * untouched kernels around it matching to 0.1 ms:
 *
 *     as it was  3.12      P=1  3.14      P=2  2.22      P=4  3.12
 *
 * P=1 is the loop swap alone - cell outside channel, so a staged value serves
 * both of a lane's channels - and it is worth NOTHING on its own, which is the
 * measurement that says this kernel was waiting on the weight read and not on
 * the staged one. P=4 gives it all back: 16 KiB of workgroup memory against
 * this device's 32 KiB leaves two workgroups a core, and that costs more than
 * halving the reads buys. The same shape of answer as every other tile in this
 * repository - see chooseLinearTile, and the attention key chunk.
 */
export const OPM_PROJECT_OUTPUT_PAIRS = 2;

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
 *
 * 🔴 WHAT WAS LEFT AFTER BOTH WAS ONE GLOBAL WEIGHT READ PER MULTIPLY-ADD. The
 * loops ran channel-outside-cell, so a lane re-read all CELLS staged values for
 * each of its channels, and read `weights[output_weight + cell * c_z + z]` for
 * every product it formed: one workgroup read, one global read and one
 * multiply-add, 0.33 useful operations an instruction. At 512 MSA rows it
 * measured 3.12 ms of a 110 ms block - 296 GFLOP/s, and 69% of this device's
 * instruction issue, so it was the count and not the bytes.
 *
 * Cell is now the OUTER loop, so a staged value is read once for both of a
 * lane's channels, and a workgroup carries several pairs whose products share
 * that one weight read. Per cell an invocation issues one staged read, two
 * weight reads and two vector multiply-adds for 2*P products.
 */
export function createOuterProductMeanProjectOutputShader(
  cOuter, residual = false, pairsPerGroup = OPM_PROJECT_OUTPUT_PAIRS,
) {
  const cells = cOuter * cOuter;
  const pairs = pairsPerGroup;
  if (!Number.isSafeInteger(pairs) || pairs < 1 || pairs > 4) {
    throw new RangeError(`OPM output pairs must be 1..4; got ${pairsPerGroup}`);
  }
  // A vector over the pairs a workgroup holds, so one weight read multiplies
  // into all of them at once.
  const vector = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[pairs];
  if (vector === undefined) throw new RangeError(`OPM output pairs ${pairs} is not 1, 2 or 4`);
  const at = (name, p) => (pairs === 1 ? name : `${name}.${"xyzw"[p]}`);
  const overPairs = (body) => Array.from({ length: pairs }, (_, p) => body(p)).join("\n");

  return `${COMMON}
@group(0) @binding(0) var<storage, read> outer: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const CELLS: u32 = ${cells}u;
const PAIRS_PER_GROUP: u32 = ${pairs}u;

// One vector a cell, holding that cell's value for each pair in the group.
var<workgroup> staged: array<${vector}, ${cells}>;
var<workgroup> reduce: array<f32, 64>;
var<workgroup> scales: array<f32, ${pairs}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let total_pairs = p.length * p.length;
  let first_pair = (group.x + group.y * GRID_WIDTH) * PAIRS_PER_GROUP;
  if (first_pair >= total_pairs) { return; }
  let local = local_id.x;

  // ...a pair past the end is clamped rather than skipped, so every lane
  // reaches every barrier below; it is dropped at the write.
${overPairs((n) => `  let pair${n} = min(first_pair + ${n}u, total_pairs - 1u);
  let live${n} = first_pair + ${n}u < total_pairs;`)}

  for (var cell = local; cell < CELLS; cell += 64u) {
    var value = ${vector}(0.0);
${overPairs((n) => `    ${at("value", n)} = outer[pair${n} * CELLS + cell];`)}
    staged[cell] = value;
  }
  workgroupBarrier();

  // ...the denominator, once for each pair rather than once per channel.
${overPairs((n) => `  {
    let i = pair${n} / p.length;
    let j = pair${n} % p.length;
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
    if (local == 0u) { scales[${n}u] = 1.0 / (p.normalization_epsilon + reduce[0]); }
    workgroupBarrier();
  }`)}

  // 🔴 CELL OUTSIDE CHANNEL. A lane's channels share the staged value, and all
  // PAIRS_PER_GROUP pairs share the weight read - which is the whole point of
  // carrying more than one.
  for (var z = local; z < p.c_z; z += 64u) {
    var acc = ${vector}(weights[p.output_bias + z]);
    for (var cell = 0u; cell < CELLS; cell += 1u) {
      acc += staged[cell] * weights[p.output_weight + cell * p.c_z + z];
    }
${overPairs((n) => `    if (live${n}) {
      output[pair${n} * p.c_z + z] ${residual ? "+=" : "="} ${at("acc", n)} * scales[${n}u];
    }`)}
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
      const pass = (pipeline, buffers, x, y = 1, z = 1) => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        compute.dispatchWorkgroups(x, y, z);
        compute.end();
      };
      const normalizeGrid = rowGrid(rows);
      pass(normalize, [source.buffer, weights.buffer, params.buffer, normalized.buffer],
        normalizeGrid[0], normalizeGrid[1]);
      // ...tiles, not threads, and folded the same way the block encoders fold
      // it - this path is what check-evoformer-opm.js drives, so a dispatch
      // that disagreed with theirs would leave the real one ungated.
      const projectTiles = Math.ceil(rows / opmProjectTileRows());
      const projectGrid = rowGrid(projectTiles);
      pass(project, [normalized.buffer, mask.buffer, weights.buffer, params.buffer, left.buffer, right.buffer],
        projectGrid[0], projectGrid[1], Math.ceil(input.cOuter / opmProjectTileColumns()));
      const outputGrid = linearGrid(pairElements);
      if (outerFirst) {
        // ...one workgroup per PAIR now, not per element; see the note on the
        // contraction kernel.
        const pairs = input.length * input.length;
        const outerGrid = [Math.min(pairs, GRID_WIDTH), Math.ceil(pairs / GRID_WIDTH)];
        pass(contractPipeline, [left.buffer, right.buffer, params.buffer, intermediate.buffer],
          outerGrid[0], outerGrid[1]);
        // ...several pairs a workgroup now; see the note on the kernel. This is
        // the path check-evoformer-opm.js drives, so its grid has to agree with
        // the block encoders' or the gate would test a dispatch no fold uses.
        const outputGroups = Math.ceil(pairs / OPM_PROJECT_OUTPUT_PAIRS);
        const projectOutputGrid = [
          Math.min(outputGroups, GRID_WIDTH), Math.ceil(outputGroups / GRID_WIDTH)];
        pass(projectOutputPipeline,
          [intermediate.buffer, mask.buffer, weights.buffer, params.buffer, output.buffer],
          projectOutputGrid[0], projectOutputGrid[1]);
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
