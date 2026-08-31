/**
 * AF3's transition block on the GPU: LayerNorm, then SwiGLU, then a projection
 * back down.
 *
 *     x -> LayerNorm -> [gate | value] (2 * channels * factor wide)
 *       -> swish(gate) * value -> down-projection -> out
 *
 * WHY THIS IS NOT AF2's TRANSITION KERNEL. AF2's is LayerNorm, Linear, ReLU,
 * Linear. AF3's widens to DOUBLE the intermediate in one weight and spends half
 * of it on a swish gate. Same name, same position in the block, different
 * function - so unlike triangle multiplication, this one could not be adapted.
 *
 * 🔴 THE DOUBLE-WIDTH SPLIT IS BLOCKED HERE AND INTERLEAVED IN TRIANGLE
 * MULTIPLICATION. transition1 is [all the gates | all the values]; the triangle
 * `projection` is a,b,a,b. Both are double-width weights in the same block of
 * the same model, and using either convention for the other conforms in shape
 * and returns a plausible tensor. See src/af3/triangle-webgpu.js.
 *
 * 🔴 THE GATE HALF IS FIRST. `swish(wide[i]) * wide[intermediate + i]`. Swapped,
 * the block still runs and still trains-looking output comes out, because both
 * halves are the same shape and similar scale.
 *
 * WHY ONE WORKGROUP PER ROW. The intermediate is `channels * factor * 2` - 1024
 * floats for the trunk pair track - which is 4 KB of workgroup memory, so a row
 * can be normalised, widened, gated and contracted without the widened tensor
 * ever reaching global memory. Materialising it would cost rows * 1024 * 4
 * bytes: 1.47 GB at 600 tokens, for a value read once. AF2's kernel chunks rows
 * to survive that; this one never allocates it.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const WORKGROUP = 128;
const GRID_WIDTH = 32_768;

/** The packing order of the four tensors this kernel reads. */
const ORDER = ["inputLayerNormScale", "inputLayerNormOffset", "transition1", "transition2"];

export function packTransitionWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`transition weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

export function createTransitionShader(shape, offsets, epsilon, variance) {
  const { rows, channels, factor } = shape;
  const intermediate = channels * factor;
  // The fast variance is the trunk's; the atom and diffusion stacks want the
  // two-pass one. See the note in src/triangle/shaders.js.
  const varianceCode = variance === "fast"
    ? `let variance = sum_squares / f32(CHANNELS) - mean * mean;`
    : `var centered_total = 0.0;
  for (var c = local; c < CHANNELS; c += WORKGROUP) {
    let d = input[base + c] - mean;
    centered_total += d * d;
  }
  reduce_a[local] = centered_total;
  workgroupBarrier();
  for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  let variance = reduce_a[0] / f32(CHANNELS);`;

  return `
const ROWS: u32 = ${rows}u;
const CHANNELS: u32 = ${channels}u;
const INTERMEDIATE: u32 = ${intermediate}u;
const WORKGROUP: u32 = ${WORKGROUP}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const W_SCALE: u32 = ${offsets.inputLayerNormScale}u;
const W_OFFSET: u32 = ${offsets.inputLayerNormOffset}u;
const W_T1: u32 = ${offsets.transition1}u;
const W_T2: u32 = ${offsets.transition2}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> normalized: array<f32, ${channels}>;
var<workgroup> gated: array<f32, ${intermediate}>;
var<workgroup> reduce_a: array<f32, ${WORKGROUP}>;
var<workgroup> reduce_b: array<f32, ${WORKGROUP}>;

fn swish(value: f32) -> f32 { return value / (1.0 + exp(-value)); }

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  // 🔴 THE ROW IS UNIFORM ACROSS THE WORKGROUP, which is what makes the early
  // return legal next to the barriers below - every invocation in the group
  // takes the same branch.
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= ROWS) { return; }
  let local = local_id.x;
  let base = row * CHANNELS;

  var total = 0.0;
  var sq_total = 0.0;
  for (var c = local; c < CHANNELS; c += WORKGROUP) {
    let value = input[base + c];
    total += value;
    sq_total += value * value;
  }
  reduce_a[local] = total;
  reduce_b[local] = sq_total;
  workgroupBarrier();
  for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      reduce_a[local] += reduce_a[local + stride];
      reduce_b[local] += reduce_b[local + stride];
    }
    workgroupBarrier();
  }
  let mean = reduce_a[0] / f32(CHANNELS);
  let sum_squares = reduce_b[0];
  ${varianceCode}
  let inverse_std = inverseSqrt(variance + EPSILON);
  workgroupBarrier();

  for (var c = local; c < CHANNELS; c += WORKGROUP) {
    normalized[c] = (input[base + c] - mean) * inverse_std * weights[W_SCALE + c]
      + weights[W_OFFSET + c];
  }
  workgroupBarrier();

  // transition1 is (channels, intermediate * 2), so a column is strided by the
  // full doubled width. Gate half first, value half second.
  let wide = INTERMEDIATE * 2u;
  for (var i = local; i < INTERMEDIATE; i += WORKGROUP) {
    var gate = 0.0;
    var value = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      let x = normalized[c];
      let column = W_T1 + c * wide;
      gate += x * weights[column + i];
      value += x * weights[column + INTERMEDIATE + i];
    }
    gated[i] = swish(gate) * value;
  }
  workgroupBarrier();

  // transition2 is (intermediate, channels).
  for (var c = local; c < CHANNELS; c += WORKGROUP) {
    var sum = 0.0;
    for (var i = 0u; i < INTERMEDIATE; i += 1u) {
      sum += gated[i] * weights[W_T2 + i * CHANNELS + c];
    }
    output[base + c] = sum;
  }
}`;
}

export class Af3TransitionGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} input rows * channels
   * @param {{rows: number, channels: number, factor?: number}} shape
   * @param {object} weights inputLayerNormScale/Offset, transition1, transition2
   * @param {{epsilon?: number, variance?: "fast"|"two-pass"}} options
   */
  async run(input, shape, weights, options = {}) {
    const { rows, channels } = shape;
    const factor = shape.factor ?? 4;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (input.length !== rows * channels) {
      throw new Error(`input has ${input.length} elements; expected ${rows * channels}`);
    }
    const intermediate = channels * factor;
    if (weights.transition1.length !== channels * intermediate * 2) {
      throw new Error(`transition1 has ${weights.transition1.length} elements; expected `
        + `${channels * intermediate * 2} - is the widening factor ${factor}?`);
    }
    if (weights.transition2.length !== intermediate * channels) {
      throw new Error(`transition2 has ${weights.transition2.length} elements; expected `
        + `${intermediate * channels}`);
    }

    const packed = packTransitionWeights(weights);
    const source = createTransitionShader({ rows, channels, factor }, packed.offsets,
                                          epsilon, variance);
    const pipeline = await this.pipelines.get(
      `af3-transition:${rows}:${channels}:${factor}:${epsilon}:${variance}`, source);

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const inputBuffer = keep(this.allocator.upload("af3-transition.input", input, storage));
      const weightBuffer = keep(this.allocator.upload("af3-transition.weights", packed.data, storage));
      const outputBuffer = keep(this.allocator.allocate(
        "af3-transition.output", rows * channels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "af3-transition.readback", rows * channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-transition" });
      const pass = encoder.beginComputePass({ label: "af3-transition" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [inputBuffer, weightBuffer, outputBuffer].map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      }));
      pass.dispatchWorkgroups(Math.min(rows, GRID_WIDTH), Math.ceil(rows / GRID_WIDTH));
      pass.end();
      encoder.copyBufferToBuffer(outputBuffer.buffer, 0, readback.buffer, 0, rows * channels * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        output: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
