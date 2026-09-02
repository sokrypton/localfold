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

/**
 * 🔴 128 EVEN WHERE THE ROWS RUN OUT, WHICH IS NOT WHAT IT LOOKS LIKE. One
 * workgroup a row means the pairformer's SINGLE transition launches 59 of them
 * on a 59-residue chain - 7,552 invocations on a part that wants tens of
 * thousands - so widening the workgroup looks like the only lever left. It is
 * slower: 256 lanes measured 0.728 ms against 128 lanes' 0.591 on exactly that
 * shape. The LayerNorm's tree reduction grows a level, and the second matmul
 * hands 384 channels to 256 lanes, so half of them do two and half do one.
 * `shape.width` still overrides it, for tools/gpu/bench-transition.js.
 */
const DEFAULT_WORKGROUP = 128;

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

/**
 * How much of the widened intermediate is resident in workgroup memory at once.
 *
 * 🔴 HALF, WHICH IS NEITHER THE LARGEST NOR THE SMALLEST THAT FITS. The chunk
 * sets two things against each other: it is the whole of this kernel's
 * workgroup memory, so a smaller one leaves more workgroups resident per core,
 * and it is also how many slots one invocation accumulates at once, so a larger
 * one reads the normalised tile fewer times. Measured on the pair track's shape
 * - 3481 rows, 512 intermediate, tile 4 - the whole intermediate is 1.559 ms,
 * half is 1.525, a quarter is 1.569, against 1.653 before the kernel was
 * chunked at all. The curve is shallow and it has an interior minimum, which is
 * why this is a function rather than "as small as possible".
 *
 * 🔴 AND HALVING ONLY PAYS WHILE THE HALF IS STILL BIG. The MSA track widens 64
 * channels to 256, where half is 128 and each invocation then accumulates one
 * slot: 0.266 ms against the whole intermediate's 0.256. So the floor is 256,
 * which leaves the MSA track unchunked and the pair and single tracks halved.
 */
export function transitionChunk(intermediate, width = DEFAULT_WORKGROUP) {
  const half = intermediate / 2;
  return half >= 256 && half % width === 0 ? half : intermediate;
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
  const WORKGROUP = shape.width ?? DEFAULT_WORKGROUP;
  const chunk = shape.chunk ?? transitionChunk(intermediate, WORKGROUP);
  if (intermediate % chunk !== 0) {
    throw new Error(`chunk ${chunk} does not divide intermediate ${intermediate}`);
  }
  // The second matmul gives each invocation this many output channels, and one
  // accumulator per (channel, tile row) has to survive the chunk loop.
  const channelsPerThread = Math.ceil(channels / WORKGROUP);
  if (chunk % WORKGROUP !== 0) {
    throw new Error(`chunk ${chunk} is not a multiple of the workgroup ${WORKGROUP}`);
  }
  const slotsPerThread = chunk / WORKGROUP;
  const accumulator = channelsPerThread === 1 ? "t" : "out_slot * TILE + t";
  const writeAccumulator = channelsPerThread === 1 ? "t" : "write_slot * TILE + t";
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
const CHUNK: u32 = ${chunk}u;
const BLOCK: u32 = ${slotsPerThread}u;
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
var<workgroup> gated: array<f32, ${tile * chunk}>;
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
  //
  // 🔴 THE INTERMEDIATE IS WALKED IN CHUNKS SO THAT THE ROW TILE COSTS NOTHING.
  // gated is the whole of this kernel's workgroup memory - TILE * 512 floats
  // for the pair track - and holding all of it is what capped the tile at four:
  // tile 8 fits in the 32 KiB this device grants and measured 1.77x SLOWER,
  // because 20 KiB a workgroup leaves one resident per core. The contraction
  // below only ever reads gated in the order it is written, so a chunk can be
  // produced, consumed into the output accumulators, and overwritten. Shared
  // memory then depends on CHUNK rather than INTERMEDIATE, and the tile buys
  // its halved weight traffic without spending occupancy for it.
  let wide = INTERMEDIATE * 2u;
  var sum: array<f32, ${tile * channelsPerThread}>;
  for (var s = 0u; s < ${tile * channelsPerThread}u; s += 1u) { sum[s] = 0.0; }

  for (var chunk0 = 0u; chunk0 < INTERMEDIATE; chunk0 += CHUNK) {
    // ...before overwriting the chunk the previous iteration is still reading.
    workgroupBarrier();
    // 🔴 EVERY SLOT THIS INVOCATION OWNS IS ACCUMULATED AT ONCE, so the tile's
    // normalised values are read from workgroup memory once for all of them
    // rather than once each. The loop used to be slot-outer, channel-inner:
    // TILE shared reads and two weight reads bought 2 * TILE multiply-adds.
    // Blocked, BLOCK * 2 weight reads and the same TILE shared reads buy
    // BLOCK * 2 * TILE. CHUNK is what sets BLOCK, which is why the chunk that
    // wins is not simply the smallest one that fits.
    var gate: array<f32, ${tile * slotsPerThread}>;
    var value: array<f32, ${tile * slotsPerThread}>;
    for (var s = 0u; s < ${tile * slotsPerThread}u; s += 1u) { gate[s] = 0.0; value[s] = 0.0; }
    for (var c = 0u; c < CHANNELS; c += 1u) {
      let column = W_T1 + c * wide;
      var x: array<f32, ${tile}>;
      for (var t = 0u; t < TILE; t += 1u) { x[t] = normalized[t * CHANNELS + c]; }
      for (var b = 0u; b < BLOCK; b += 1u) {
        // ...read once, used TILE times. That ratio is the point.
        let i = chunk0 + local + b * WORKGROUP;
        let wg = weights[column + i];
        let wv = weights[column + INTERMEDIATE + i];
        for (var t = 0u; t < TILE; t += 1u) {
          gate[b * TILE + t] += x[t] * wg;
          value[b * TILE + t] += x[t] * wv;
        }
      }
    }
    for (var b = 0u; b < BLOCK; b += 1u) {
      for (var t = 0u; t < TILE; t += 1u) {
        gated[t * CHUNK + local + b * WORKGROUP] = swish(gate[b * TILE + t]) * value[b * TILE + t];
      }
    }
    workgroupBarrier();

    // transition2 is (intermediate, channels).
    var out_slot = 0u;
    for (var c = local; c < CHANNELS; c += WORKGROUP) {
      for (var slot = 0u; slot < CHUNK; slot += 1u) {
        let w = weights[W_T2 + (chunk0 + slot) * CHANNELS + c];
        for (var t = 0u; t < TILE; t += 1u) {
          sum[${accumulator}] += gated[t * CHUNK + slot] * w;
        }
      }
      out_slot += 1u;
    }
  }

  var write_slot = 0u;
  for (var c = local; c < CHANNELS; c += WORKGROUP) {
    for (var t = 0u; t < TILE; t += 1u) {
      let row = base_row + t;
      if (row < ROWS) { output[row * CHANNELS + c] = sum[${writeAccumulator}]; }
    }
    write_slot += 1u;
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
