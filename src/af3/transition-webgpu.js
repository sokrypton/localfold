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

/**
 * How many rows one workgroup transitions at once, given how many there are.
 *
 * 🔴 A FUNCTION OF THE ROW COUNT, NOT A CONSTANT, BECAUSE THE TWO CALLERS ARE
 * NOTHING ALIKE. The pair track transitions 3481 rows and the weight set is
 * what costs - tiling by four cut it from 241 ms to 85. The pairformer's SINGLE
 * transition has one row a token, 59 of them, and tiling by four leaves fifteen
 * workgroups: it went 30.6 ms to 43.7, because there is no occupancy left to
 * trade. So the tile is taken only when the rows can spare it.
 *
 * 🔴 AND EVERY CALLER MUST AGREE WITH THE SHADER. The shader is generated with
 * this tile and the dispatch is divided by it, so a caller that computes one
 * and not the other silently transitions a fraction of its rows. Both sides
 * call this with the same `rows`, which is why it is a pure function.
 */
export function transitionRowTile(rows) {
  for (const tile of [4, 2]) {
    if (rows / tile >= 256) return tile;
  }
  return 1;
}
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
  const tile = shape.tile ?? transitionRowTile(rows);
  // The fast variance is the trunk's; the atom and diffusion stacks want the
  // two-pass one. See the note in src/triangle/shaders.js.
  const varianceCode = variance === "fast"
    ? `let variance = sum_squares / f32(CHANNELS) - mean * mean;`
    : `var centered_total = 0.0;
  for (var c = local; c < CHANNELS; c += WORKGROUP) {
    let d = input[base + c] - mean;
    centered_total += d * d;
  }
  // 🔴 A BARRIER BEFORE REUSING THE REDUCTION BUFFER. Every invocation has
  // just read reduce_a[0] for the mean; writing reduce_a[local] without a
  // barrier lets a fast lane clobber slot 0 while a slow one is still reading
  // it. The result is a WRONG MEAN in some rows, some of the time - which
  // reads as a numerical problem, not a race.
  workgroupBarrier();
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
const TILE: u32 = ${tile}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const W_SCALE: u32 = ${offsets.inputLayerNormScale}u;
const W_OFFSET: u32 = ${offsets.inputLayerNormOffset}u;
const W_T1: u32 = ${offsets.transition1}u;
const W_T2: u32 = ${offsets.transition2}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> normalized: array<f32, ${tile * channels}>;
var<workgroup> gated: array<f32, ${tile * intermediate}>;
var<workgroup> reduce_a: array<f32, ${WORKGROUP}>;
var<workgroup> reduce_b: array<f32, ${WORKGROUP}>;

fn swish(value: f32) -> f32 { return value / (1.0 + exp(-value)); }

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  // 🔴 A TILE OF ROWS, NOT ONE, BECAUSE THE WEIGHTS WERE THE COST. Timestamp
  // profiling put this kernel at 241 ms of a 632 ms pairformer pass - the
  // largest single kernel in the trunk - and one workgroup a row means every
  // workgroup reads the whole weight set: 196k floats for 3481 pair rows, which
  // is 2.7 GB a block. Tiling divides that by TILE and the arithmetic is
  // unchanged. The disable-and-remeasure bisect this replaced said 129 ms.
  let base_row = (group.x + group.y * GRID_WIDTH) * TILE;
  if (base_row >= ROWS) { return; }
  let local = local_id.x;

  // 🔴 THE LAYER NORM IS PER ROW AND STAYS THAT WAY. Its reduction is over
  // CHANNELS and the tile's rows do not share it, so this loop is sequential -
  // it is cheap next to the two matmuls, which are what the tile is for.
  for (var t = 0u; t < TILE; t += 1u) {
    let row = base_row + t;
    if (row < ROWS) {
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
        normalized[t * CHANNELS + c] = (input[base + c] - mean) * inverse_std
          * weights[W_SCALE + c] + weights[W_OFFSET + c];
      }
    }
    workgroupBarrier();
  }

  // transition1 is (channels, intermediate * 2), so a column is strided by the
  // full doubled width. Gate half first, value half second.
  let wide = INTERMEDIATE * 2u;
  for (var i = local; i < INTERMEDIATE; i += WORKGROUP) {
    var gate: array<f32, ${tile}>;
    var value: array<f32, ${tile}>;
    for (var t = 0u; t < TILE; t += 1u) { gate[t] = 0.0; value[t] = 0.0; }
    for (var c = 0u; c < CHANNELS; c += 1u) {
      let column = W_T1 + c * wide;
      // ...read once, used TILE times. That ratio is the point.
      let wg = weights[column + i];
      let wv = weights[column + INTERMEDIATE + i];
      for (var t = 0u; t < TILE; t += 1u) {
        let x = normalized[t * CHANNELS + c];
        gate[t] += x * wg;
        value[t] += x * wv;
      }
    }
    for (var t = 0u; t < TILE; t += 1u) {
      gated[t * INTERMEDIATE + i] = swish(gate[t]) * value[t];
    }
  }
  workgroupBarrier();

  // transition2 is (intermediate, channels).
  for (var c = local; c < CHANNELS; c += WORKGROUP) {
    var sum: array<f32, ${tile}>;
    for (var t = 0u; t < TILE; t += 1u) { sum[t] = 0.0; }
    for (var i = 0u; i < INTERMEDIATE; i += 1u) {
      let w = weights[W_T2 + i * CHANNELS + c];
      for (var t = 0u; t < TILE; t += 1u) { sum[t] += gated[t * INTERMEDIATE + i] * w; }
    }
    for (var t = 0u; t < TILE; t += 1u) {
      let row = base_row + t;
      if (row < ROWS) { output[row * CHANNELS + c] = sum[t]; }
    }
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
      // ...one workgroup a TILE of rows; see TRANSITION_ROW_TILE.
      const groups = Math.ceil(rows / transitionRowTile(rows));
      pass.dispatchWorkgroups(Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH));
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
