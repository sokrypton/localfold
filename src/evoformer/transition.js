import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const ceilDivide = (value, divisor) => Math.ceil(value / divisor);
const GRID_WIDTH = 32_768;
export const TRANSITION_TILE_COLUMNS = 64;

/**
 * How large a single transition binding is allowed to get.
 *
 * Well under any device's limit on purpose: the point is not to squeeze to the
 * ceiling but to keep one chunk's scratch small enough that the pool can reuse
 * it, and 96 MiB is a window that divides the working set without making the
 * loop long.
 */
export const TRANSITION_CHUNK_TARGET_BYTES = 96 * 1024 * 1024;

const gcd = (left, right) => {
  let a = left; let b = right;
  while (b !== 0) { const remainder = a % b; a = b; b = remainder; }
  return a;
};

/**
 * How many rows one transition chunk may cover.
 *
 * 🔴 A TENSOR CAN FIT IN A BUFFER AND STILL NOT BE BINDABLE. The transition's
 * hidden activation is rows * hiddenChannels floats - at 508 MSA rows of a
 * 291-residue alignment that is 147,828 * 1024 * 4 bytes, 578 MiB, which
 * allocates on any modern adapter and then exceeds maxStorageBufferBindingSize
 * when it is bound. Splitting the rows is what makes long sequences possible.
 *
 * 🔴 THE CHUNK IS ALIGNED TWICE OVER, and both alignments are load-bearing:
 *   - to TRANSITION_TILE_ROWS, because the linear kernels tile rows by 16 and a
 *     chunk that is not a whole number of tiles would leave a ragged edge;
 *   - to minStorageBufferOffsetAlignment (256 bytes), because each chunk BINDS
 *     at a row offset, and a binding offset that is not a multiple of 256 is a
 *     validation error rather than a slow path.
 * The least common multiple of the two satisfies both at once. `channels * 4`
 * is the row stride the offset is measured in, so how many rows it takes to
 * reach a 256-byte boundary depends on it - hence the gcd.
 *
 * THE FULL PATH IS PRESERVED EXACTLY: when everything already binds, this
 * returns `rows` and the caller runs its single-dispatch branch unchanged.
 */
export function transitionChunkRows(
  rows,
  channels,
  hiddenChannels,
  maxStorageBufferBindingSize,
  minStorageBufferOffsetAlignment = 256,
) {
  if (![rows, channels, hiddenChannels, maxStorageBufferBindingSize, minStorageBufferOffsetAlignment]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("transition chunk dimensions and limits must be positive safe integers");
  }
  const rowBytes = Math.max(channels, hiddenChannels) * Float32Array.BYTES_PER_ELEMENT;
  if (rows * rowBytes <= maxStorageBufferBindingSize) return rows;
  const capacity = Math.floor(
    Math.min(maxStorageBufferBindingSize, TRANSITION_CHUNK_TARGET_BYTES) / rowBytes,
  );
  if (capacity < 1) throw new RangeError("WebGPU storage binding is too small for one transition row");
  const sourceRowBytes = channels * Float32Array.BYTES_PER_ELEMENT;
  const offsetRowAlignment = minStorageBufferOffsetAlignment
    / gcd(sourceRowBytes, minStorageBufferOffsetAlignment);
  const rowAlignment = TRANSITION_TILE_ROWS * offsetRowAlignment
    / gcd(TRANSITION_TILE_ROWS, offsetRowAlignment);
  if (rows <= capacity) return rows;
  if (capacity < rowAlignment) {
    throw new RangeError("WebGPU storage binding cannot hold one aligned transition chunk");
  }
  return Math.min(rows, Math.floor(capacity / rowAlignment) * rowAlignment);
}
export const TRANSITION_TILE_ROWS = 16;

function validate(input) {
  const { rows, channels, hiddenChannels, activations, weights } = input;
  if (![rows, channels, hiddenChannels].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("transition dimensions must be positive safe integers");
  }
  const lengths = [
    ["activations", activations, rows * channels],
    ["layerNormScale", weights.layerNormScale, channels],
    ["layerNormOffset", weights.layerNormOffset, channels],
    ["firstWeight", weights.firstWeight, channels * hiddenChannels],
    ["firstBias", weights.firstBias, hiddenChannels],
    ["secondWeight", weights.secondWeight, hiddenChannels * channels],
    ["secondBias", weights.secondBias, channels],
  ];
  for (const [name, value, expected] of lengths) {
    if (value.length !== expected) throw new RangeError(`${name} has ${value.length} values; expected ${expected}`);
  }
}

export function packTransitionWeights(input) {
  const values = [
    input.weights.layerNormScale,
    input.weights.layerNormOffset,
    input.weights.firstWeight,
    input.weights.firstBias,
    input.weights.secondWeight,
    input.weights.secondBias,
  ];
  const offsets = [];
  let length = 0;
  for (const value of values) {
    offsets.push(length);
    length += value.length;
  }
  const data = new Float32Array(length);
  for (let index = 0; index < values.length; index += 1) data.set(values[index], offsets[index]);
  return { data, offsets };
}

export function createTransitionShaders(input, offsets) {
  void input;
  void offsets;
  const normalize = `
struct NormalizeParameters {
  rows: u32,
  channels: u32,
  scale_offset: u32,
  offset_offset: u32,
  epsilon: f32,
  padding_0: u32,
  padding_1: u32,
  padding_2: u32,
};
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: NormalizeParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  // ONE WORKGROUP PER ROW, folded across two dimensions: a transition runs over
  // msaSequences * length rows, which is 147,828 at 508 rows of a 291-residue
  // alignment - well past the 65535 a dispatch may be wide. The row guard below
  // is what makes the fold safe: the grid is rounded up, and the extra
  // workgroups return before touching anything.
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= parameters.rows) { return; }
  let base = row * parameters.channels;
  var sum = 0.0;
  for (var c = local.x; c < parameters.channels; c += 64u) {
    sum += source[base + c];
  }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(parameters.channels); }
  workgroupBarrier();

  var sum_squared = 0.0;
  for (var c = local.x; c < parameters.channels; c += 64u) {
    let centered = source[base + c] - row_mean[0];
    sum_squared += centered * centered;
  }
  partial[local.x] = sum_squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(parameters.channels) + parameters.epsilon);
  for (var c = local.x; c < parameters.channels; c += 64u) {
    output[base + c] = (source[base + c] - row_mean[0]) * inverse_std
      * weights[parameters.scale_offset + c] + weights[parameters.offset_offset + c];
  }
}`;
  // One register-blocked projection serves both dense layers. A workgroup
  // computes a 16x64 output tile: every invocation retains sixteen accumulators,
  // so the source tile is loaded once for 64 columns instead of eight times by
  // separate 8x8 workgroups. The k loop order is unchanged from the reference.
  const linear = `
struct MatmulParameters {
  rows: u32,
  inner: u32,
  columns: u32,
  weight_offset: u32,
  bias_offset: u32,
  activation: u32,
  padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: MatmulParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> tile_source: array<f32, 128>;
var<workgroup> tile_weight: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row = group.y * 16u + local.y;
  let second_row = row + 8u;
  let column = group.x * 64u + local.x;
  let tile_index = local.y * 8u + local.x;
  var value_low = vec4<f32>(0.0);
  var value_high = vec4<f32>(0.0);
  var second_value_low = vec4<f32>(0.0);
  var second_value_high = vec4<f32>(0.0);

  for (var k0 = 0u; k0 < parameters.inner; k0 += 8u) {
    let source_k = k0 + local.x;
    let weight_k = k0 + local.y;
    tile_source[tile_index] = 0.0;
    tile_source[tile_index + 64u] = 0.0;
    if (row < parameters.rows && source_k < parameters.inner) {
      tile_source[tile_index] = source[row * parameters.inner + source_k];
    }
    if (second_row < parameters.rows && source_k < parameters.inner) {
      tile_source[tile_index + 64u] = source[second_row * parameters.inner + source_k];
    }
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let tile_column = local.x + column_block * 8u;
      let output_column = column + column_block * 8u;
      let weight_index = local.y * 64u + tile_column;
      tile_weight[weight_index] = 0.0;
      if (output_column < parameters.columns && weight_k < parameters.inner) {
        tile_weight[weight_index] = weights[
          parameters.weight_offset + weight_k * parameters.columns + output_column
        ];
      }
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      let source_value = tile_source[local.y * 8u + k];
      let second_source_value = tile_source[local.y * 8u + k + 64u];
      let weight_base = k * 64u + local.x;
      let weight_low = vec4<f32>(tile_weight[weight_base], tile_weight[weight_base + 8u],
        tile_weight[weight_base + 16u], tile_weight[weight_base + 24u]);
      let weight_high = vec4<f32>(tile_weight[weight_base + 32u], tile_weight[weight_base + 40u],
        tile_weight[weight_base + 48u], tile_weight[weight_base + 56u]);
      value_low += source_value * weight_low;
      value_high += source_value * weight_high;
      second_value_low += second_source_value * weight_low;
      second_value_high += second_source_value * weight_high;
    }
    workgroupBarrier();
  }

  if (row < parameters.rows) {
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let output_column = column + column_block * 8u;
      if (output_column < parameters.columns) {
        let values = select(value_low, value_high, column_block >= 4u);
        var value = values[column_block % 4u];
        value += weights[parameters.bias_offset + output_column];
        if (parameters.activation == 1u) { value = max(value, 0.0); }
        output[row * parameters.columns + output_column] = value;
      }
    }
  }
  if (second_row < parameters.rows) {
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let output_column = column + column_block * 8u;
      if (output_column < parameters.columns) {
        let values = select(second_value_low, second_value_high, column_block >= 4u);
        var value = values[column_block % 4u];
        value += weights[parameters.bias_offset + output_column];
        if (parameters.activation == 1u) { value = max(value, 0.0); }
        output[second_row * parameters.columns + output_column] = value;
      }
    }
  }
}`;
  const linearResidual = linear.replace(
    "output[row * parameters.columns + output_column] = value;",
    "output[row * parameters.columns + output_column] += value;",
  ).replace(
    "output[second_row * parameters.columns + output_column] = value;",
    "output[second_row * parameters.columns + output_column] += value;",
  );
  return [normalize, linear, linearResidual];
}

export function createTransitionNormalizeParameters(input, offsets) {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setUint32(0, input.rows, true);
  view.setUint32(4, input.channels, true);
  view.setUint32(8, offsets[0], true);
  view.setUint32(12, offsets[1], true);
  view.setFloat32(16, input.epsilon ?? 1e-5, true);
  return new Uint8Array(buffer);
}

export class TransitionGpu {
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
    const packed = packTransitionWeights(input);
    const code = createTransitionShaders(input, packed.offsets);
    const key = `transition:${input.rows}:${input.channels}:${input.hiddenChannels}:${input.epsilon ?? 1e-5}`;
    const pipelines = [];
    for (let index = 0; index < code.length; index += 1) {
      pipelines.push(await this.pipelines.get(`${key}:${index}`, code[index]));
    }
    const allocations = [];
    const keep = (value) => { allocations.push(value); return value; };
    const storage = GPUBufferUsage.STORAGE;
    try {
      const source = keep(this.allocator.upload("transition.source", input.activations, storage));
      const weights = keep(this.allocator.upload("transition.weights", packed.data, storage));
      const layerNormParameters = keep(this.allocator.upload(
        "transition.normalize.parameters", createTransitionNormalizeParameters(input, packed.offsets), GPUBufferUsage.UNIFORM,
      ));
      const firstParameters = keep(this.allocator.upload("transition.first.parameters", new Uint32Array([
        input.rows, input.channels, input.hiddenChannels, packed.offsets[2], packed.offsets[3], 1, 0, 0,
      ]), GPUBufferUsage.UNIFORM));
      const secondParameters = keep(this.allocator.upload("transition.second.parameters", new Uint32Array([
        input.rows, input.hiddenChannels, input.channels, packed.offsets[4], packed.offsets[5], 0, 0, 0,
      ]), GPUBufferUsage.UNIFORM));
      const normalized = keep(this.allocator.allocate("transition.normalized", input.rows * input.channels * 4, storage));
      const hidden = keep(this.allocator.allocate("transition.hidden", input.rows * input.hiddenChannels * 4, storage));
      const output = keep(this.allocator.allocate(
        "transition.output", input.rows * input.channels * 4, storage | GPUBufferUsage.COPY_SRC,
      ));
      const readback = keep(this.allocator.allocate(
        "transition.readback", input.rows * input.channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      const encoder = this.device.createCommandEncoder({ label: "transition" });
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
      pass(pipelines[0], [source.buffer, weights.buffer, layerNormParameters.buffer, normalized.buffer],
        Math.min(input.rows, GRID_WIDTH), ceilDivide(input.rows, GRID_WIDTH));
      pass(pipelines[1], [normalized.buffer, weights.buffer, firstParameters.buffer, hidden.buffer],
        ceilDivide(input.hiddenChannels, TRANSITION_TILE_COLUMNS), ceilDivide(input.rows, TRANSITION_TILE_ROWS));
      pass(pipelines[1], [hidden.buffer, weights.buffer, secondParameters.buffer, output.buffer],
        ceilDivide(input.channels, TRANSITION_TILE_COLUMNS), ceilDivide(input.rows, TRANSITION_TILE_ROWS));
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, input.rows * input.channels * 4);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        output: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index] .release();
    }
  }
}
