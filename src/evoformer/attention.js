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

/**
 * The tile the q/k/v/gate projection computes. One statement of it; all three
 * dispatch sites divide by these.
 */
export const ATTENTION_PROJECT_TILE = { lanesX: 8, lanesY: 8, rowsPerLane: 4, columnsPerLane: 2 };
export const attentionProjectTileRows = (tile = ATTENTION_PROJECT_TILE) =>
  tile.lanesY * tile.rowsPerLane;
export const attentionProjectTileColumns = (tile = ATTENTION_PROJECT_TILE) =>
  tile.lanesX * tile.columnsPerLane;

/**
 * q, k, v and the gate, from one normalised activation.
 *
 * 🔴 ONE vec4 A CELL, NOT FOUR ARRAYS. The four matrices are contracted over
 * the same activation at the same (channel, output) cell, so the four weights a
 * cell needs are always wanted together: packed as a vec4 the inner loop reads
 * them in ONE instruction and accumulates them in one multiply-add instead of
 * four. AF3's triangle projection is the same change; see src/triangle/
 * shaders.js.
 *
 * 🔴 AND THE SOURCE WAS STILL SCALAR, WHICH IS THE FOURTH KERNEL IN THIS BLOCK
 * TO HAVE BEEN. Two rows to an invocation, read one float at a time: two source
 * reads and two vec4 weight reads bought four vec4 multiply-adds. Staged
 * TRANSPOSED - four rows to a vector - one read serves four, so three reads buy
 * eight. 2.0 useful operations an instruction to 2.9.
 *
 * 🔴 THE TILE IS 32 BY 16 AND NOT WIDER, BECAUSE A CELL HERE IS ALREADY A
 * VECTOR. Every other kernel that took this treatment holds `rows * columns / 4`
 * vec4 accumulators; this one holds `rows * columns` of them, one per matrix
 * set - so the register budget runs out four times sooner, and it is the budget
 * and not the traffic that decides. Measured as the two projections of a
 * 512-row block, in runs where every other kernel matched to within 0.05 ms:
 *
 *     2x2 (as it was, scalar source)   16.29  16.20
 *     4x2                              13.85  13.64
 *     8x2                              17.32  17.32
 *     4x4                              19.30  19.04
 *
 * Four by two is eight vec4, 32 floats, the same budget the transition's
 * winning tile has. Sixteen vec4 spills, and the kernel is then slower than
 * the scalar-source form it replaced - which is the shape of every tile sweep
 * in this repo and the reason none of them is left to a guess.
 */
export function createAttentionProjectShader(
  tile = ATTENTION_PROJECT_TILE, precision = "f32", weightPrecision = "f32",
) {
  // 🔴 THE WEIGHT BUFFER'S ELEMENT, MEASURED AND NOT SHIPPED. This kernel
  // rereads the whole weight set once per row tile - 944 of them at 512 MSA
  // rows - so halving its bytes is a bandwidth win on top of the register one,
  // and bench-attention-project.js puts it at 9.762 -> 9.238 ms, 5.4%, with
  // BITWISE IDENTICAL output: the f16 accumulators already truncate every
  // weight to a half, so pre-rounding the buffer changes nothing.
  //
  // It is not wired into a block, and the reason is scope rather than doubt.
  // This buffer is bound by four shaders - the normalisation, this, the pair
  // bias and the output projection - and by the multimer's global-attention
  // path as well, so narrowing it is a five-shader change with a silent failure
  // mode, for 1.6% of a block. The measurement is here for whoever wants it.
  const weight16 = weightPrecision === "f16";
  const wf = (e) => (weight16 ? `f32(${e})` : e);
  const { lanesX, lanesY, rowsPerLane, columnsPerLane } = tile;
  if (!["f32", "f16"].includes(precision)) {
    throw new RangeError(`unknown attention project precision ${precision}`);
  }
  // 🔴 IN f16 THE ACCUMULATORS ARE HALF THE REGISTERS, WHICH IS THE ONLY THING
  // THAT COULD MOVE THIS KERNEL'S CEILING. The sweep below says the budget and
  // not the traffic decides its tile, and that sixteen vec4 spills - so a wider
  // tile, which halves the weight traffic, is only reachable if the
  // accumulators get smaller. The bias and the store stay f32.
  const half = precision === "f16";
  const vector = half ? "vec4<f16>" : "vec4<f32>";
  const narrow = (e) => (half ? `f16(${e})` : e);
  const widen = (e) => (half ? `f32(${e})` : e);
  if (rowsPerLane % 4 !== 0) throw new RangeError("project tile rowsPerLane must be a multiple of 4");
  const lanes = lanesX * lanesY;
  const step = lanesY;
  const tileRows = lanesY * rowsPerLane;
  const rowVectors = tileRows / 4;
  const sourceTasks = step * rowVectors;
  const sourcePerLane = Math.ceil(sourceTasks / lanes);

  const declare = [];
  for (let r = 0; r < rowsPerLane; r += 1) {
    for (let v = 0; v < columnsPerLane; v += 1) {
      // ...the gate's bias rides in the w lane, as it did before.
      declare.push(`  var acc_${r}_${v} = ${vector}(0.0, 0.0, 0.0, ${narrow(`bias_${v}`)});`);
    }
  }
  const bias = [];
  for (let v = 0; v < columnsPerLane; v += 1) {
    bias.push(`  let hd_${v} = column_origin + ${v * lanesX}u;
  var bias_${v} = 0.0;
  if (hd_${v} < projected) { bias_${v} = ${wf(`weights[p.gating_bias + hd_${v}]`)}; }`);
  }

  const inner = [];
  for (let k = 0; k < step; k += 1) {
    for (let g = 0; g < rowsPerLane / 4; g += 1) {
      inner.push(`    let s_${k}_${g} = tile_source[${k * rowVectors}u + local.y * ${rowsPerLane / 4}u + ${g}u];`);
    }
    for (let v = 0; v < columnsPerLane; v += 1) {
      inner.push(`    let w_${k}_${v} = tile_weight[${k * lanesX * columnsPerLane}u + local.x * ${columnsPerLane}u + ${v}u];`);
    }
    for (let r = 0; r < rowsPerLane; r += 1) {
      for (let v = 0; v < columnsPerLane; v += 1) {
        inner.push(`    acc_${r}_${v} += s_${k}_${Math.floor(r / 4)}[${r % 4}u] * w_${k}_${v};`);
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
      let row_base = group.y * ${tileRows}u + row_group * 4u;
      var staged = ${vector}(0.0);
      if (c < p.channels) {
        for (var j = 0u; j < 4u; j += 1u) {
          let row = row_base + j;
          if (row < rows) { staged[j] = ${narrow("source[row * p.channels + c]")}; }
        }
      }
      tile_source[k_local * ${rowVectors}u + row_group] = staged;
    }`);
  }

  const stageWeight = [];
  for (let v = 0; v < columnsPerLane; v += 1) {
    stageWeight.push(`    {
      var packed = ${vector}(0.0);
      let output_hd = column_origin + ${v * lanesX}u;
      if (output_hd < projected && weight_c < p.channels) {
        let weight_index = weight_c * projected + output_hd;
        packed = ${vector}(${narrow(wf("weights[p.query_weight + weight_index]"))},
                           ${narrow(wf("weights[p.key_weight + weight_index]"))},
                           ${narrow(wf("weights[p.value_weight + weight_index]"))},
                           ${narrow(wf("weights[p.gating_weight + weight_index]"))});
      }
      tile_weight[local.y * ${lanesX * columnsPerLane}u + local.x * ${columnsPerLane}u + ${v}u] = packed;
    }`);
  }

  const store = [];
  for (let r = 0; r < rowsPerLane; r += 1) {
    const body = [];
    for (let v = 0; v < columnsPerLane; v += 1) {
      body.push(`    if (hd_${v} < projected) {
      let index = row_${r} * projected + hd_${v};
      query[index] = ${widen(`acc_${r}_${v}.x`)} * inverseSqrt(f32(p.head_dim));
      key[index] = ${widen(`acc_${r}_${v}.y`)};
      value[index] = ${widen(`acc_${r}_${v}.z`)};
      gate[index] = 1.0 / (1.0 + exp(-${widen(`acc_${r}_${v}.w`)}));
    }`);
    }
    store.push(`  {
    let row_${r} = group.y * ${tileRows}u + local.y * ${rowsPerLane}u + ${r}u;
    if (row_${r} < rows) {
${body.join("\n")}
    }
  }`);
  }

  return `${half || weight16 ? "enable f16;\n" : ""}${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> query: array<f32>;
@group(0) @binding(4) var<storage, read_write> key: array<f32>;
@group(0) @binding(5) var<storage, read_write> value: array<f32>;
@group(0) @binding(6) var<storage, read_write> gate: array<f32>;

// Transposed: four ROWS to a vector, so one read serves four accumulators.
var<workgroup> tile_source: array<${vector}, ${step * rowVectors}>;
// One vec4 a cell: (query, key, value, gate), laid out per thread.
var<workgroup> tile_weight: array<${vector}, ${step * lanesX * columnsPerLane}>;

@compute @workgroup_size(${lanesX}, ${lanesY}, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let projected = p.heads * p.head_dim;
  let rows = p.batch * p.queries;
  let linear_lane = local.y * ${lanesX}u + local.x;
  let column_origin = group.x * ${lanesX * columnsPerLane}u + local.x;
${bias.join("\n")}
${declare.join("\n")}

  for (var c0 = 0u; c0 < p.channels; c0 += ${step}u) {
    let weight_c = c0 + local.y;
${stageSource.join("\n")}
${stageWeight.join("\n")}
    workgroupBarrier();
${inner.join("\n")}
    workgroupBarrier();
  }

${store.join("\n")}
}`;
}

export const ATTENTION_PROJECT_SHADER = createAttentionProjectShader();

/**
 * The wider tile the f16 accumulators pay for, and its shader.
 *
 * 🔴 THE TILE AND THE PRECISION ARE ONE CHOICE, NOT TWO. The sweep above says
 * this kernel is bound by its register budget - it holds `rowsPerLane *
 * columnsPerLane` vec4 accumulators where every other kernel here holds a
 * quarter as many - and that a 32x32 tile SPILLS in f32 and is then slower than
 * the 32x16 one. In f16 the accumulators are half the registers, the same tile
 * fits, and it is the fastest arm there is. Measured on the block's own shape
 * (30,208 rows, 256 channels, 8 heads of 32) by
 * tools/gpu/bench-attention-project.js:
 *
 *     f32   32x16  13.79 ms    32x32  19.24 ms   (spilling)
 *     f16   32x32   9.80        32x16 10.23      64x16 10.95
 *
 * 1.41x on the best f32 arm, and only reachable by changing both together -
 * which is why the two travel in one object rather than as two options.
 *
 * The error is 2.4e-3 on q, k, v and the gate, and it is a storage format for
 * them rather than a change of model: AF2's own inference runs in bfloat16,
 * whose eight mantissa bits are looser again. End to end
 * (tools/gpu/fold-af2.js): at 128 rows pLDDT 57.284 -> 57.300 with pTM and the
 * CA-CA median unchanged to four and three decimals, and 1619 -> 1529 ms; at
 * 512 rows 5285 -> 4920 ms.
 */
export const ATTENTION_PROJECT_TILE_F16 = {
  lanesX: 8, lanesY: 8, rowsPerLane: 4, columnsPerLane: 4,
};

/**
 * @param {"auto"|"f32"|"f16"} requested `auto` takes f16 wherever the device
 *   has shader-f16, which is where the wider tile is affordable.
 */
export function selectAttentionProjectKernel(device, requested = "auto") {
  const precision = requested !== "auto" ? requested
    : device?.features?.has("shader-f16") ? "f16" : "f32";
  if (precision === "f16" && device?.features?.has("shader-f16") !== true) {
    throw new Error("the f16 attention projection requires the shader-f16 feature");
  }
  const tile = precision === "f16" ? ATTENTION_PROJECT_TILE_F16 : ATTENTION_PROJECT_TILE;
  return {
    precision, tile,
    // 🔴 THE TILE IS IN THE KEY AS WELL AS THE PRECISION, because the dispatch
    // divides by it: a cache that handed back the other one would leave whole
    // tiles of rows unprojected, which reads as a speedup.
    cacheKey: `block:attention:project:${precision}`
      + `:${attentionProjectTileRows(tile)}x${attentionProjectTileColumns(tile)}`,
    shader: precision === "f16"
      ? createAttentionProjectShader(tile, "f16") : ATTENTION_PROJECT_SHADER,
  };
}

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
export function createAttentionRegisterFlashShader(headDim, keyChunk, options = {}) {
  const vectors = headDim / 4;
  // How many keys share one rescale of the accumulator, and whether the q.k
  // reduction runs through a vec4. Both default to what this kernel always did.
  const group = options.group ?? 1;
  const vectorScore = options.vectorScore ?? false;
  const lazyRescale = options.lazyRescale ?? false;
  const queriesPerLane = options.queriesPerLane ?? 1;
  // 🔴 THE REGISTER FILE IS WHAT THIS KERNEL IS SHORT OF, WHICH IS THE OPPOSITE
  // OF WHAT ITS FLOP RATE SUGGESTS. Eight query vectors and eight accumulators
  // are 64 registers a lane before anything else; two queries a lane doubles
  // that and measures 4.7x SLOWER, and grouping the keys to save the rescale
  // measures 2.3x slower for two more scalars. Nothing here is arithmetic
  // bound - removing the whole per-key rescale moves it 2%.
  //
  // So the move is to make the state SMALLER. In f16 the query, the
  // accumulators and both staged chunks halve: 64 registers become 32 and the
  // 8 KiB of workgroup memory becomes 4, both of which buy occupancy, and
  // probe-alu.js puts an f16 multiply-add at 1.7x an f32 one for the same
  // instruction on top. The softmax state - the running max, the running sum
  // and the logit - stays f32, because that is where the RANGE is and f16 tops
  // out at 65504.
  const precision = options.precision ?? "f32";
  if (!["f32", "f16", "chunk16"].includes(precision)) {
    throw new RangeError(`unknown attention precision ${precision}`);
  }
  const chunk16 = precision !== "f32";
  const register16 = precision === "f16";
  const chunkType = chunk16 ? "vec4<f16>" : "vec4<f32>";
  const registerType = register16 ? "vec4<f16>" : "vec4<f32>";
  const enable = chunk16 ? "enable f16;\n" : "";
  if (!Number.isSafeInteger(queriesPerLane) || queriesPerLane < 1) {
    throw new RangeError(`attention queries per lane must be positive; got ${queriesPerLane}`);
  }
  if (!Number.isSafeInteger(group) || group < 1) {
    throw new RangeError(`attention key group must be a positive integer; got ${group}`);
  }
  const each = (body) => Array.from({ length: vectors }, (_, t) => `    ${body(t)}`).join("\n");
  const declare = (name, init) => Array.from({ length: vectors },
    (_, t) => `  var ${name}${t} = ${init(t)};`).join("\n");
  // 🔴 EVERY LANE OF A WORKGROUP READS THE SAME KEY AND THE SAME VALUE. The
  // dispatch gives a workgroup 64 consecutive QUERIES of one (batch, head), and
  // the key loop runs over the same axis for all of them - so each of the
  // 2 * head_dim/4 vectors a key needs was fetched by 64 lanes issuing 64
  // identical global loads. Staged, that is one load and 64 workgroup reads.
  // AF3's grid attention is the same kernel and the same fix, worth 1.9x there;
  // see src/af3/grid-attention-webgpu.js.
  //
  // 🔴 THE MASK LOOKS LIKE THE FOURTH OPERAND OF THAT ARGUMENT AND STAGING IT
  // LOSES. It is indexed by the batch and the KEY, so all 64 lanes read the
  // same float, one global load per key per lane - exactly the shape above.
  // Staged in workgroup memory alongside key and value, with the arithmetic
  // left alone, the kernel measured 22.1 ms against 20.9 in an otherwise
  // identical block (every other kernel within 0.05 ms across the two runs).
  // One float a key is already broadcast and cached; the extra staging loop and
  // its workgroup memory cost more than the load did. Do not stage the mask.
  // ...8 KiB of workgroup memory at head_dim 32, which is the size AF3's grid
  // attention settled on. tools/gpu/bench-msa-attention.js sweeps it.
  const chunk = keyChunk ?? Math.max(8, Math.floor(512 / (vectors * 2)));
  if (!Number.isSafeInteger(chunk) || chunk < 1) {
    throw new RangeError(`attention key chunk must be a positive integer; got ${chunk}`);
  }
  if (chunk % group !== 0) {
    throw new RangeError(`attention key chunk ${chunk} is not a multiple of the group ${group}`);
  }

  // 🔴 THE q.k REDUCTION TARGET DECIDES THE ISSUE WIDTH OF HALF THIS KERNEL.
  // `score += dot(qv_t, k_t)` names a SCALAR accumulator, so head_dim scalar
  // multiply-adds are issued where head_dim/4 vec4 ones would do the same work
  // - and probe-alu.js puts a vec4 multiply-add at four times a scalar one for
  // the same instruction. Accumulating into a vec4 and folding it once at the
  // end costs three adds a key and issues eight instructions where it issued
  // thirty-two.
  const scoreOf = (g, indexExpression) => (vectorScore
    ? [`      var part${g} = ${registerType}(0.0);`,
       ...Array.from({ length: vectors }, (_, t) =>
         `      part${g} += qv${t} * ${register16 !== chunk16 ? `${registerType}(key_chunk[${indexExpression} + ${t}u])` : `key_chunk[${indexExpression} + ${t}u]`};`),
       `      let score${g} = f32(part${g}.x) + f32(part${g}.y) + f32(part${g}.z) + f32(part${g}.w);`]
    : [`      var sum${g} = 0.0;`,
       ...Array.from({ length: vectors }, (_, t) =>
         `      sum${g} += f32(dot(qv${t}, ${register16 !== chunk16 ? `${registerType}(key_chunk[${indexExpression} + ${t}u])` : `key_chunk[${indexExpression} + ${t}u]`}));`),
       `      let score${g} = sum${g};`]).join("\n");

  // 🔴 AND THE RESCALE IS PAID ONCE A KEY FOR A MAX THAT RARELY MOVES. The
  // online softmax multiplies every one of head_dim/4 accumulators by
  // exp(old_max - new_max) on every key. Taking the max of `group` keys FIRST
  // and rescaling once for all of them leaves the arithmetic exact - it is the
  // same associativity the key chunk already reassociates - and turns
  // head_dim/4 multiplies a key into head_dim/(4*group).
  //
  // An out-of-range key is scored -1e30 so it cannot raise the group max, and
  // its weight is selected to zero, which is what the per-key `break` did.
  const indices = Array.from({ length: group }, (_, g) => g);

  // 🔴 AND THE CHEAPEST WAY TO SKIP THE RESCALE IS TO ASK WHETHER THE MAX MOVED.
  // Grouping the keys removes the same multiplies and pays for them in
  // registers - the accumulators and the query already fill the file, so two
  // more scores an iteration measured 2.3x SLOWER (bench-msa-attention.js).
  // A branch costs nothing to hold. The running max of 512 keys is a record
  // sequence, so it advances about H(512) ~ 7 times rather than 512, and the
  // lanes of a workgroup are different queries only for as long as one of them
  // is still setting records - which is early and short.
  const lazy = lazyRescale ? [
    `    for (var slot = 0u; slot < KEY_CHUNK; slot += 1u) {`,
    `      let k_index = k0 + slot;`,
    `      if (k_index >= p.queries) { break; }`,
    `      let staged = slot * HD4;`,
    scoreOf("", "staged"),
    `      var logit = score + 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);`,
    `      if (p.has_pair_bias != 0u) {`,
    `        logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];`,
    `      }`,
    `      logit = clamp(logit, -1e8, 1e8);`,
    `      if (logit > running_max) {`,
    `        let previous_scale = exp(running_max - logit);`,
    `        running_sum = running_sum * previous_scale;`,
    ...Array.from({ length: vectors }, (_, t) => `        acc${t} = acc${t} * ${narrow("previous_scale")};`),
    `        running_max = logit;`,
    `      }`,
    `      let weight = exp(logit - running_max);`,
    `      running_sum = running_sum + weight;`,
    ...Array.from({ length: vectors }, (_, t) =>
      `      acc${t} = acc${t} + ${narrow("weight")} * ${chunkRead(`staged + ${t}u`)};`),
    `    }`,
  ].join("\n") : null;

  // 🔴 A KEY IS STAGED ONCE AND READ BY EVERY LANE, WHICH IS WHERE THIS KERNEL'S
  // TIME GOES. Priced by removing the reads (bench-msa-attention.js
  // `auto:noval`, `auto:nokey`): the sixteen workgroup vec4 reads a lane issues
  // per key are 8.7 ms of 20.8 at 512 sequences, against 0.4 ms for both
  // exponentials and nothing at all for the staging loop's global loads. The
  // arithmetic is not the bound - removing the whole per-key rescale changes
  // the time by 2% and costs more in registers than it saves.
  //
  // So a lane takes `queriesPerLane` queries instead of one. The staged key and
  // value vectors are read ONCE and used for all of them, which divides the
  // dominant term by that number; what it multiplies is the register file, and
  // the queries and accumulators already fill most of it. That is the trade,
  // and it is why this is a measured option rather than the shape of the
  // kernel.
  // A scalar narrowed to the accumulator's element, and a staged read widened
  // to it - both no-ops when the two already agree.
  const narrow = (e) => (register16 ? `f16(${e})` : e);
  const chunkRead = (e) => (register16 === chunk16 ? `value_chunk[${e}]` : `${registerType}(value_chunk[${e}])`);
  const queryIndex = (q) => `q_index_${q}`;
  const perQuery = (body) => Array.from({ length: queriesPerLane }, (_, q) => body(q)).join("\n");
  const multiDeclare = perQuery((q) => [
    `  let ${queryIndex(q)} = group.x * ${64 * queriesPerLane}u + local + ${q * 64}u;`,
    `  let live${q} = ${queryIndex(q)} < p.queries;`,
    `  let base${q} = ((batch_index * p.queries + select(0u, ${queryIndex(q)}, live${q})) * p.heads + head) * HD4;`,
    Array.from({ length: vectors }, (_, t) => `  var qv${q}_${t} = ${registerType}(query[base${q} + ${t}u]);`).join("\n"),
    Array.from({ length: vectors }, (_, t) => `  var acc${q}_${t} = ${registerType}(0.0);`).join("\n"),
    `  var running_max${q} = -1e30;`,
    `  var running_sum${q} = 0.0;`,
  ].join("\n"));

  const multiInner = [
    `    for (var slot = 0u; slot < KEY_CHUNK; slot += 1u) {`,
    `      let k_index = k0 + slot;`,
    `      if (k_index >= p.queries) { break; }`,
    `      let staged = slot * HD4;`,
    `      let masked = 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);`,
    perQuery((q) => `      var part${q} = ${registerType}(0.0);`),
    // One staged read, every query's partial sum fed from it.
    Array.from({ length: vectors }, (_, t) => [
      `      let kv_${t} = ${register16 === chunk16 ? `key_chunk[staged + ${t}u]` : `${registerType}(key_chunk[staged + ${t}u])`};`,
      perQuery((q) => `      part${q} += qv${q}_${t} * kv_${t};`),
    ].join("\n")).join("\n"),
    perQuery((q) => [
      `      var logit${q} = f32(part${q}.x) + f32(part${q}.y) + f32(part${q}.z) + f32(part${q}.w) + masked;`,
      `      if (p.has_pair_bias != 0u) {`,
      `        logit${q} += pair_bias[(head * p.queries + ${queryIndex(q)}) * p.queries + k_index];`,
      `      }`,
      `      logit${q} = clamp(logit${q}, -1e8, 1e8);`,
      `      let new_max${q} = max(running_max${q}, logit${q});`,
      `      let scale${q} = exp(running_max${q} - new_max${q});`,
      `      let weight${q} = exp(logit${q} - new_max${q});`,
      `      running_sum${q} = running_sum${q} * scale${q} + weight${q};`,
      `      running_max${q} = new_max${q};`,
    ].join("\n")),
    Array.from({ length: vectors }, (_, t) => [
      `      let vv_${t} = ${register16 === chunk16 ? `value_chunk[staged + ${t}u]` : `${registerType}(value_chunk[staged + ${t}u])`};`,
      perQuery((q) => `      acc${q}_${t} = acc${q}_${t} * ${narrow(`scale${q}`)} + ${narrow(`weight${q}`)} * vv_${t};`),
    ].join("\n")).join("\n"),
    `    }`,
  ].join("\n");

  if (queriesPerLane > 1) {
    return `${enable}${COMMON}
const HD4: u32 = ${vectors}u;
const KEY_CHUNK: u32 = ${chunk}u;
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> key_chunk: array<${chunkType}, ${chunk * vectors}>;
var<workgroup> value_chunk: array<${chunkType}, ${chunk * vectors}>;

fn mask_index(batch: u32, key_index: u32) -> u32 {
  if (p.transpose == 0u) { return batch * p.queries + key_index; }
  return key_index * p.batch + batch;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let local = local_id.x;
  let batch_index = group.y;
  let head = group.z;
  if (batch_index >= p.batch || head >= p.heads) { return; }
${multiDeclare}

  for (var k0 = 0u; k0 < p.queries; k0 += KEY_CHUNK) {
    workgroupBarrier();
    for (var index = local; index < KEY_CHUNK * HD4; index += 64u) {
      let k_index = min(k0 + index / HD4, p.queries - 1u);
      let k_base = ((batch_index * p.queries + k_index) * p.heads + head) * HD4 + index % HD4;
      key_chunk[index] = ${chunkType}(key[k_base]);
      value_chunk[index] = ${chunkType}(value[k_base]);
    }
    workgroupBarrier();
${multiInner}
  }

${perQuery((q) => `  if (live${q}) {
${Array.from({ length: vectors }, (_, t) =>
  `    output[base${q} + ${t}u] = (vec4<f32>(acc${q}_${t}) / running_sum${q}) * gate[base${q} + ${t}u];`).join("\n")}
  }`)}
}`;
  }

  const inner = lazy ?? [
    `    for (var slot = 0u; slot < KEY_CHUNK; slot += ${group}u) {`,
    `      if (k0 + slot >= p.queries) { break; }`,
    ...indices.flatMap((g) => [
      `      let key_at${g} = k0 + slot + ${g}u;`,
      `      let live${g} = key_at${g} < p.queries;`,
      `      let staged${g} = (slot + ${g}u) * HD4;`,
      scoreOf(g, `staged${g}`),
      `      var logit${g} = score${g} + 1e9 * (mask[mask_index(batch_index, min(key_at${g}, p.queries - 1u))] - 1.0);`,
      `      if (p.has_pair_bias != 0u) {`,
      `        logit${g} += pair_bias[(head * p.queries + q_index) * p.queries + min(key_at${g}, p.queries - 1u)];`,
      `      }`,
      `      logit${g} = select(-1e30, clamp(logit${g}, -1e8, 1e8), live${g});`,
    ]),
    `      let group_max = ${indices.map((g) => `logit${g}`).reduce((a, b) => `max(${a}, ${b})`)};`,
    `      let new_max = max(running_max, group_max);`,
    `      let previous_scale = exp(running_max - new_max);`,
    `      running_max = new_max;`,
    ...indices.map((g) => `      let w${g} = select(0.0, exp(logit${g} - new_max), live${g});`),
    `      running_sum = running_sum * previous_scale + ${indices.map((g) => `w${g}`).join(" + ")};`,
    ...Array.from({ length: vectors }, (_, t) =>
      `      acc${t} = acc${t} * ${narrow("previous_scale")}${indices.map((g) => ` + ${narrow(`w${g}`)} * ${chunkRead(`staged${g} + ${t}u`)}`).join("")};`),
    `    }`,
  ].join("\n");
  return `${enable}${COMMON}
const HD4: u32 = ${vectors}u;
const KEY_CHUNK: u32 = ${chunk}u;
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> key_chunk: array<${chunkType}, ${chunk * vectors}>;
var<workgroup> value_chunk: array<${chunkType}, ${chunk * vectors}>;

fn mask_index(batch: u32, key_index: u32) -> u32 {
  if (p.transpose == 0u) { return batch * p.queries + key_index; }
  return key_index * p.batch + batch;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  // 🔴 THE WORKGROUP id, NOT THE GLOBAL ONE, because the uniformity analysis
  // has to SEE that the batch and the head are workgroup-uniform or it rejects
  // the barriers below - and a lane past the last query cannot return, because
  // it has to reach every one of them. It is stopped at the write instead.
  let local = local_id.x;
  let q_index = group.x * 64u + local;
  let batch_index = group.y;
  let head = group.z;
  if (batch_index >= p.batch || head >= p.heads) { return; }
  let live = q_index < p.queries;
  let q_base = ((batch_index * p.queries + select(0u, q_index, live)) * p.heads + head) * HD4;

${declare("qv", (t) => `${registerType}(query[q_base + ${t}u])`)}
${declare("acc", () => `${registerType}(0.0)`)}
  var running_max = -1e30;
  var running_sum = 0.0;

  for (var k0 = 0u; k0 < p.queries; k0 += KEY_CHUNK) {
    // ...before overwriting the chunk the previous iteration is still reading.
    workgroupBarrier();
    for (var index = local; index < KEY_CHUNK * HD4; index += 64u) {
      let k_index = min(k0 + index / HD4, p.queries - 1u);
      let k_base = ((batch_index * p.queries + k_index) * p.heads + head) * HD4 + index % HD4;
      key_chunk[index] = ${chunkType}(key[k_base]);
      value_chunk[index] = ${chunkType}(value[k_base]);
    }
    workgroupBarrier();

${inner}
  }

  if (live) {
${each((t) => `output[q_base + ${t}u] = (vec4<f32>(acc${t}) / running_sum) * gate[q_base + ${t}u];`)}
  }
}`;
}

/**
 * The subgroup size every one of these kernels is written for. It is pinned in
 * the shader as `@subgroup_size(32)` and assumed by the lane arithmetic, so it
 * is a correctness requirement and not a preference.
 */
export const attentionFlashQueriesPerGroup = (queriesPerLane = 1) => 64 * queriesPerLane;

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
// One query's eight output vectors, so a lane can index the one it writes.
var<workgroup> gathered_tile: array<vec4<f32>, 64>;

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
  // 🔴 A LANE OWNS A KEY, NOT AN OUTPUT COMPONENT, AND THAT IS THE WHOLE
  // KERNEL. Weighting the values used to be a 32-iteration loop per lane -
  // shuffle a neighbour's probability, read that key's component, multiply -
  // three instructions for every useful multiply-add, and this is the largest
  // kernel of an AF2 block at depth because column attention is quadratic in
  // SEQUENCES. Owning a key instead, a lane multiplies ITS probability into all
  // 32 components with no shuffle at all: eight vec4 reads and eight vector
  // multiply-adds for thirty-two, and the cross-lane sum happens once at the
  // end rather than once per key.
  //
  // The online softmax survives it: the running maximum and sum are
  // subgroup-uniform, so every lane rescales its own partial by the same
  // factor, and summing the partials at the end is the same sum in a different
  // order.
  var accumulated_0 = vec4<f32>(0.0);
  var accumulated_1 = vec4<f32>(0.0);
  var accumulated_2 = vec4<f32>(0.0);
  var accumulated_3 = vec4<f32>(0.0);
  var accumulated_4 = vec4<f32>(0.0);
  var accumulated_5 = vec4<f32>(0.0);
  var accumulated_6 = vec4<f32>(0.0);
  var accumulated_7 = vec4<f32>(0.0);
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

    let new_max = max(running_max, tile_max);
    let previous_scale = exp(running_max - new_max);
    let tile_scale = exp(tile_max - new_max);
    let weight = probability * tile_scale;
    // ...this lane's own key, so there is nothing to shuffle.
    accumulated_0 = accumulated_0 * previous_scale + weight * value_tile[subgroup_lane];
    accumulated_1 = accumulated_1 * previous_scale + weight * value_tile[32u + subgroup_lane];
    accumulated_2 = accumulated_2 * previous_scale + weight * value_tile[64u + subgroup_lane];
    accumulated_3 = accumulated_3 * previous_scale + weight * value_tile[96u + subgroup_lane];
    accumulated_4 = accumulated_4 * previous_scale + weight * value_tile[128u + subgroup_lane];
    accumulated_5 = accumulated_5 * previous_scale + weight * value_tile[160u + subgroup_lane];
    accumulated_6 = accumulated_6 * previous_scale + weight * value_tile[192u + subgroup_lane];
    accumulated_7 = accumulated_7 * previous_scale + weight * value_tile[224u + subgroup_lane];
    running_sum = running_sum * previous_scale + tile_sum * tile_scale;
    running_max = new_max;
    workgroupBarrier();
  }

  // ...the partials meet here, once, instead of once per key.
  let total_0 = subgroupAdd(accumulated_0);
  let total_1 = subgroupAdd(accumulated_1);
  let total_2 = subgroupAdd(accumulated_2);
  let total_3 = subgroupAdd(accumulated_3);
  let total_4 = subgroupAdd(accumulated_4);
  let total_5 = subgroupAdd(accumulated_5);
  let total_6 = subgroupAdd(accumulated_6);
  let total_7 = subgroupAdd(accumulated_7);
  // Every lane holds all eight; one of them lands them where a lane can index
  // by its own output component.
  if (subgroup_lane == 0u) {
    gathered_tile[local.y * 8u + 0u] = total_0;
    gathered_tile[local.y * 8u + 1u] = total_1;
    gathered_tile[local.y * 8u + 2u] = total_2;
    gathered_tile[local.y * 8u + 3u] = total_3;
    gathered_tile[local.y * 8u + 4u] = total_4;
    gathered_tile[local.y * 8u + 5u] = total_5;
    gathered_tile[local.y * 8u + 6u] = total_6;
    gathered_tile[local.y * 8u + 7u] = total_7;
  }
  workgroupBarrier();

  if (valid_query) {
    let output_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
    let accumulated = gathered_tile[local.y * 8u + output_vector][output_component];
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
export async function buildAttentionFlashKernel(
  execution, device, headDim, requested = "auto", precision = "auto",
) {
  const kernel = selectAttentionFlashKernel(device, headDim, requested, precision);
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
  requestedPrecision = "auto",
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
    // 🔴 AND THE STAGED KEY AND VALUE GO IN f16 WHEREVER THE DEVICE ALLOWS IT.
    // The chunk is the kernel's largest cost and the reason is not arithmetic:
    // priced by removing them, the sixteen workgroup reads a lane issues per
    // key are 8.7 ms of 20.8 at 512 sequences. In f16 the chunk is 4 KiB
    // instead of 8, which is what buys the occupancy, and column attention
    // measures 20.83 -> 17.05 ms - 1.22x on the largest kernel in the block.
    //
    // 🔴 ONLY THE STAGED COPY NARROWS. The running max, the running sum, the
    // logit and the accumulators stay f32: the softmax is where the RANGE is,
    // and f16 tops out at 65504. Narrowing the accumulators and the query as
    // well reaches 16.4 ms, but at relRMS 3.4e-3 against this kernel's 2.0e-4,
    // and 0.65 ms is not worth an order of magnitude of error.
    //
    // 2.0e-4 is a storage format for the key and the value, not a change of
    // model: AF2's own inference runs in bfloat16, whose eight mantissa bits
    // put it an order of magnitude LOOSER than this.
    const precision = requestedPrecision !== "auto" ? requestedPrecision
      : device.features?.has("shader-f16") ? "chunk16" : "f32";
    return {
      cacheKey: `attention:flash-registers-${headDim}-${precision}`,
      shader: createAttentionRegisterFlashShader(headDim, undefined, { precision }),
      queryTile: 64, variant,
    };
  }
  return { cacheKey: "attention:flash", shader: ATTENTION_FLASH_SHADER, queryTile: 1, variant };
}

/**
 * The tile the attention output projection computes, and the only statement of
 * it - `encodeAttention` dispatches with these, so the shader and the grid
 * cannot drift apart.
 */
export const ATTENTION_OUTPUT_TILE = { lanesX: 8, lanesY: 8, rowsPerLane: 4, columnsPerLane: 8 };
export const attentionOutputTileRows = (tile = ATTENTION_OUTPUT_TILE) =>
  tile.lanesY * tile.rowsPerLane;
export const attentionOutputTileColumns = (tile = ATTENTION_OUTPUT_TILE) =>
  tile.lanesX * tile.columnsPerLane;

/**
 * The projection that turns the gated attention result back into c_m channels.
 *
 * 🔴 IT WAS THE LAST ALL-SCALAR KERNEL IN AN AF2 BLOCK. Two rows and four
 * columns to an invocation, every operand read one float at a time: two source
 * reads, eight weight reads and eight multiply-adds per step of k - eighteen
 * instructions to buy eight products, 0.44 useful operations an instruction.
 * The transition and the outer product mean both had the same shape of fault
 * and both answered to the same fix.
 *
 * The source tile is staged TRANSPOSED, four rows to a vector, so one read
 * serves four accumulators; the weight tile is staged per thread, so a lane's
 * eight strided columns are two vec4 reads. Per step of k that is three reads
 * and eight vector multiply-adds for thirty-two products - 2.9 an instruction,
 * against 0.44.
 *
 * 🔴 THE COLUMNS STAY STRIDED BY EIGHT, for the same reason they do in
 * src/evoformer/transition.js: contiguous columns per lane would make the
 * staged weights one flat read, but the store then goes out at stride eight
 * across a row's lanes instead of as a consecutive run.
 *
 * @param {boolean} residual whether the store accumulates into an existing tensor
 */
export function createAttentionOutputShader(
  tile = ATTENTION_OUTPUT_TILE, residual = false, precision = "f32",
) {
  const { lanesX, lanesY, rowsPerLane, columnsPerLane } = tile;
  // Same trade as the transition's linear kernel, which this kernel is a
  // sibling of: in f16 the staged operands and the accumulators halve, so the
  // narrow tile keeps its occupancy and gets the cheaper reads. The bias, the
  // residual and the store stay f32.
  if (!["f32", "f16"].includes(precision)) {
    throw new RangeError(`unknown attention output precision ${precision}`);
  }
  const half = precision === "f16";
  const vector = half ? "vec4<f16>" : "vec4<f32>";
  const narrow = (e) => (half ? `f16(${e})` : e);
  const widen = (e) => (half ? `f32(${e})` : e);
  if (rowsPerLane % 4 !== 0 || columnsPerLane % 4 !== 0) {
    throw new RangeError("attention output tile must be a multiple of 4 each way");
  }
  const lanes = lanesX * lanesY;
  const step = lanesY;
  const tileRows = lanesY * rowsPerLane;
  const rowVectors = tileRows / 4;
  const columnVectors = columnsPerLane / 4;
  const sourceTasks = step * rowVectors;
  const sourcePerLane = Math.ceil(sourceTasks / lanes);

  const lines = [];
  const declare = [];
  for (let r = 0; r < rowsPerLane; r += 1) {
    for (let v = 0; v < columnVectors; v += 1) declare.push(`  var acc_${r}_${v} = ${vector}(0.0);`);
  }
  for (let k = 0; k < step; k += 1) {
    for (let g = 0; g < rowsPerLane / 4; g += 1) {
      lines.push(`    let s_${k}_${g} = tile_source[${k * rowVectors}u + local.y * ${rowsPerLane / 4}u + ${g}u];`);
    }
    for (let v = 0; v < columnVectors; v += 1) {
      lines.push(`    let w_${k}_${v} = tile_weight[${k * lanesX * columnVectors}u + local.x * ${columnVectors}u + ${v}u];`);
    }
    for (let r = 0; r < rowsPerLane; r += 1) {
      for (let v = 0; v < columnVectors; v += 1) {
        lines.push(`    acc_${r}_${v} += s_${k}_${Math.floor(r / 4)}[${r % 4}u] * w_${k}_${v};`);
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
      let k = k0 + k_local;
      let row_base = group.y * ${tileRows}u + row_group * 4u;
      var staged = ${vector}(0.0);
      if (k < projected) {
        for (var j = 0u; j < 4u; j += 1u) {
          let row = row_base + j;
          if (row < rows) { staged[j] = ${narrow("source[row * projected + k]")}; }
        }
      }
      tile_source[k_local * ${rowVectors}u + row_group] = staged;
    }`);
  }

  const stageWeight = [];
  for (let v = 0; v < columnVectors; v += 1) {
    stageWeight.push(`    {
      var staged = ${vector}(0.0);
      if (weight_k < projected) {
        for (var j = 0u; j < 4u; j += 1u) {
          let output_channel = channel_origin + (${v}u * 4u + j) * ${lanesX}u;
          if (output_channel < p.channels) {
            staged[j] = ${narrow("weights[p.output_weight + weight_k * p.channels + output_channel]")};
          }
        }
      }
      tile_weight[local.y * ${lanesX * columnVectors}u + local.x * ${columnVectors}u + ${v}u] = staged;
    }`);
  }

  const store = [];
  for (let r = 0; r < rowsPerLane; r += 1) {
    const body = [];
    for (let v = 0; v < columnVectors; v += 1) {
      for (let c = 0; c < 4; c += 1) {
        body.push(`      {
        let output_channel = channel_origin + ${(v * 4 + c) * lanesX}u;
        if (output_channel < p.channels) {
          output[output_row_${r} * p.channels + output_channel] ${residual ? "+=" : "="}
            ${widen(`acc_${r}_${v}[${c}u]`)} + weights[p.output_bias + output_channel];
        }
      }`);
      }
    }
    // ...column attention writes its result back TRANSPOSED, which is why this
    // cannot simply be the transition's linear kernel.
    store.push(`  {
    let row_${r} = group.y * ${tileRows}u + local.y * ${rowsPerLane}u + ${r}u;
    if (row_${r} < rows) {
      let b_${r} = row_${r} / p.queries;
      let q_${r} = row_${r} % p.queries;
      let output_row_${r} = select(row_${r}, q_${r} * p.batch + b_${r}, p.transpose != 0u);
${body.join("\n")}
    }
  }`);
  }

  return `${half ? "enable f16;\n" : ""}${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Transposed: four ROWS to a vector, so one read serves four accumulators.
var<workgroup> tile_source: array<${vector}, ${step * rowVectors}>;
// Laid out per thread: a lane's own strided columns, contiguous where it reads.
var<workgroup> tile_weight: array<${vector}, ${step * lanesX * columnVectors}>;

@compute @workgroup_size(${lanesX}, ${lanesY}, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let rows = p.batch * p.queries;
  let projected = p.heads * p.head_dim;
  let linear_lane = local.y * ${lanesX}u + local.x;
  let channel_origin = group.x * ${lanesX * columnsPerLane}u + local.x;
${declare.join("\n")}

  for (var k0 = 0u; k0 < projected; k0 += ${step}u) {
    let weight_k = k0 + local.y;
${stageSource.join("\n")}
${stageWeight.join("\n")}
    workgroupBarrier();
${lines.join("\n")}
    workgroupBarrier();
  }

${store.join("\n")}
}`;
}

export const ATTENTION_OUTPUT_SHADER = createAttentionOutputShader(ATTENTION_OUTPUT_TILE, false);

/** Same projection, but commits directly into an existing residual tensor. */
export const ATTENTION_OUTPUT_RESIDUAL_SHADER =
  createAttentionOutputShader(ATTENTION_OUTPUT_TILE, true);

/**
 * The output projection's tile and the element its k loop works in.
 *
 * 🔴 THE SAME INVERSION AS THE TRANSITION'S LINEAR KERNEL, which this is a
 * sibling of - it differs only in writing its result back transposed. The wide
 * tile exists to halve the weight traffic at the cost of workgroups; f16 halves
 * the traffic anyway and halves the accumulators too, so the NARROW tile keeps
 * its occupancy and gets the cheaper reads. Measured as the block's two
 * attention outputs, in runs where every untouched kernel matched to 0.05 ms.
 *
 * `requested` is "auto", "f32" or "f16"; f32 keeps the 32x64 tile this shipped
 * with, which is still the right answer on a device without shader-f16.
 */
export const ATTENTION_OUTPUT_TILE_F16 = {
  lanesX: 8, lanesY: 8, rowsPerLane: 4, columnsPerLane: 4,
};

export function selectAttentionOutputKernel(device, residual, requested = "auto") {
  const precision = requested !== "auto" ? requested
    : device?.features?.has("shader-f16") ? "f16" : "f32";
  if (precision === "f16" && device?.features?.has("shader-f16") !== true) {
    throw new Error("the f16 attention output projection requires the shader-f16 feature");
  }
  const tile = precision === "f16" ? ATTENTION_OUTPUT_TILE_F16 : ATTENTION_OUTPUT_TILE;
  return {
    precision, tile,
    // The tile is in the key with the precision: the dispatch divides by it.
    cacheKey: `block:attention:output${residual ? "-residual" : ""}:${precision}`
      + `:${attentionOutputTileRows(tile)}x${attentionOutputTileColumns(tile)}`,
    shader: precision === "f16"
      ? createAttentionOutputShader(tile, residual, "f16")
      : (residual ? ATTENTION_OUTPUT_RESIDUAL_SHADER : ATTENTION_OUTPUT_SHADER),
  };
}
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
      this.options.flashPrecision ?? "auto",
    );
    const projectKernel = selectAttentionProjectKernel(
      this.device, this.options.projectPrecision ?? "auto");
    const outputKernel = selectAttentionOutputKernel(
      this.device, false, this.options.outputPrecision ?? "auto");
    const [normalize, project, pairProject, flash, outputProject] = await Promise.all([
      this.pipelines.get("attention:normalize", ATTENTION_NORMALIZE_SHADER),
      this.pipelines.get(projectKernel.cacheKey, projectKernel.shader),
      this.pipelines.get("attention:pair-bias", ATTENTION_PAIR_BIAS_SHADER),
      this.pipelines.get(flashKernel.cacheKey, flashKernel.shader),
      this.pipelines.get(outputKernel.cacheKey, outputKernel.shader),
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
        ceilDivide(input.channels, attentionProjectTileColumns(projectKernel.tile)),
        ceilDivide(rows, attentionProjectTileRows(projectKernel.tile)));
      pass(flash, [query.buffer, key.buffer, value.buffer, gate.buffer, mask.buffer, pairBias.buffer, params.buffer,
        weighted.buffer], ceilDivide(input.queryLength, flashKernel.queryTile),
        input.batch, input.heads);
      pass(outputProject, [weighted.buffer, weights.buffer, params.buffer, output.buffer],
        ceilDivide(input.channels, attentionOutputTileColumns(outputKernel.tile)),
        ceilDivide(rows, attentionOutputTileRows(outputKernel.tile)));
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
