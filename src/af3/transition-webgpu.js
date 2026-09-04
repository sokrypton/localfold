import { concatenateAs } from "../runtime/float16.js";
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
 * 🔴 AND EIGHT ONLY BECAME THE RIGHT ANSWER ONCE THE ROWS WERE VECTOR LANES.
 * As scalar code tile 8 measured 1.556 ms against tile 4's 1.525 on the pair
 * shape; with four rows to a vec4 it is 1.394 against 1.494. Vectorising the
 * arithmetic bought nothing on its own - this kernel waits on its weight reads,
 * not its multiply-adds - but it freed the tile to grow.
 *
 * 🔴 AND EVERY CALLER MUST AGREE WITH THE SHADER. The shader is generated with
 * this tile and the dispatch is divided by it, so a caller that computes one
 * and not the other silently transitions a fraction of its rows. Both sides
 * call this with the same `rows`, which is why it is a pure function.
 */
export function transitionRowTile(rows) {
  for (const tile of [8, 4, 2]) {
    if (rows / tile >= 220) return tile;
  }
  return 1;
}

/**
 * How much of the widened intermediate is resident in workgroup memory at once.
 *
 * 🔴 NEITHER THE LARGEST NOR THE SMALLEST THAT FITS. The chunk sets two things
 * against each other: it is the whole of this kernel's workgroup memory, so a
 * smaller one leaves more workgroups resident per core, and it is also how many
 * slots one invocation accumulates at once, so a larger one reads the
 * normalised tile fewer times. Measured on the pair track's shape - 3481 rows,
 * 512 intermediate - at tile 4 the whole intermediate is 1.559 ms, half 1.525,
 * a quarter 1.569; at tile 8 a quarter is 1.394 and a half 1.713. What holds
 * across both is the product: TILE * CHUNK stays at two intermediates' worth of
 * floats, 4 KB for the pair track, so the tile buys its halved weight traffic
 * without also spending workgroup memory.
 *
 * The MSA track lands on the floor of one workgroup width and is flat there
 * anyway (0.262 to 0.269 ms across every arm), and the single track's tile is 1
 * so it stays unchunked.
 */
export function transitionChunk(intermediate, tile = 1, width = DEFAULT_WORKGROUP) {
  // Keep the staged block a constant size - two intermediates' worth of floats,
  // 4 KB for the pair track - so the tile trades weight traffic for occupancy
  // and not for workgroup memory as well.
  const wanted = Math.max(width, Math.round(2 * intermediate / tile / width) * width);
  const chunk = Math.min(intermediate, wanted);
  return intermediate % chunk === 0 ? chunk : intermediate;
}

const GRID_WIDTH = 32_768;

/** The packing order of the four tensors this kernel reads. */
const ORDER = ["inputLayerNormScale", "inputLayerNormOffset", "transition1", "transition2"];

/**
 * @param {"f32"|"f16"} precision the element the packed buffer holds. The
 *   offsets are in ELEMENTS and do not depend on it, so a caller that packs one
 *   way and builds the shader the other gets a wrong answer rather than an
 *   error - which is why `createTransitionShader` takes the same word.
 */
export function packTransitionWeights(weights, precision = "f32") {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`transition weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = concatenateAs(precision, total, (target) => {
    for (const name of ORDER) target.set(weights[name], offsets[name]);
  });
  return { data, offsets };
}

export function createTransitionShader(shape, offsets, epsilon, variance) {
  const { rows, channels, factor } = shape;
  const intermediate = channels * factor;
  const tile = shape.tile ?? transitionRowTile(rows);
  const WORKGROUP = shape.width ?? DEFAULT_WORKGROUP;
  const chunk = shape.chunk ?? transitionChunk(intermediate, tile, WORKGROUP);
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
  // 🔴 THE TILE'S ROWS ARE THE VECTOR LANES, AND THAT IS WORTH ABOUT 4x. This
  // device runs a scalar multiply-add at 1287 GFLOP/s and a vec4 one at 5034 -
  // measured, by tools/gpu/probe-alu.js, because the paper figure says nothing
  // about what WGSL reaches here. Both matmuls do the SAME multiply against
  // every row of the tile, so the tile is exactly the axis to vectorise: four
  // rows become one vec4 and four multiply-adds become one. It also quarters
  // the workgroup reads, since the four rows now live in one slot.
  //
  // The single track's tile is 1 - there are only 59 rows to give it - so this
  // generates scalar code there, which is what LANES = 1 means below.
  // 🔴 WRITING THE RESIDUAL IN PLACE RATHER THAN A DELTA AND AN ADD PASS. The
  // pair track's five updates each wrote a full pair tensor to scratch and then
  // read it, read the pair, and wrote the pair back - four passes over 11.5 MB
  // at 150 tokens where two will do, and `add` measured 113 ms of a 2106 ms
  // pairformer there. This kernel already reads every row it writes, into
  // workgroup memory, before it writes any of them, and no other workgroup
  // touches those rows - so the read-modify-write is safe within one dispatch.
  const residual = shape.residual ?? false;
  // 🔴 AND THE ACCUMULATORS ARE A THIRD FORMAT, FOR THE THIRD REASON AGAIN.
  // `sum` is a WGSL ARRAY of `groups * channelsPerThread` vectors - the shape a
  // driver spills first - and this is the largest kernel in the trunk. Halving
  // it is the same change as src/triangle/shaders.js's two projections, where
  // it was worth 1.55x and 1.43x. The layer norm's reductions, the swish and
  // the store stay f32; only the running sum and the weight it multiplies
  // narrow.
  const accumulatePrecision = shape.accumulatePrecision ?? "f32";
  if (!["f32", "f16"].includes(accumulatePrecision)) {
    throw new RangeError(`unknown transition accumulate precision ${accumulatePrecision}`);
  }
  const acc16 = accumulatePrecision === "f16";
  const lanes = shape.lanes ?? (tile % 4 === 0 ? 4 : 1);
  if (tile % lanes !== 0) throw new Error(`tile ${tile} is not a multiple of ${lanes} lanes`);
  const groups = tile / lanes;
  const vector = lanes === 1 ? "f32" : `vec${lanes}<f32>`;
  // The running sum's element; see the note on accumulatePrecision above.
  const sumVector = acc16 ? (lanes === 1 ? "f16" : `vec${lanes}<f16>`) : vector;
  const sumZero = acc16 ? `${sumVector}(0.0)` : null;
  const toSum = (e) => (acc16 ? `${sumVector}(${e})` : e);
  const fromSum = (e) => (acc16 ? `${vector}(${e})` : e);
  // 🔴 THE TWO STAGED BLOCKS ARE WHAT THIS KERNEL IS SHORT OF, NOT ARITHMETIC.
  // `normalized` and `gated` are 4 KiB each for the pair track, and both are
  // read once per output channel by every lane - the same shape as the staged
  // key and value in src/evoformer/attention.js, where narrowing them to f16
  // bought 1.22x through occupancy. Only the STAGED COPY narrows: the layer
  // norm's reductions, the accumulators and the store all stay f32.
  const stagePrecision = shape.stagePrecision ?? "f32";
  if (!["f32", "f16"].includes(stagePrecision)) {
    throw new RangeError(`unknown transition stage precision ${stagePrecision}`);
  }
  const stage16 = stagePrecision === "f16";
  // 🔴 AND THE WEIGHT BUFFER IS A STORAGE FORMAT TOO, WHICH IS A MEMORY WIN AND
  // NOT A SPEED ONE. This kernel reads its weights one scalar at a time and
  // this machine is instruction-bound, so halving their bytes changes the time
  // by nothing measurable - but `w.single-transition` is 324 MiB of the 567 a
  // 59-token AF3 fold keeps resident, more than every other tensor together,
  // and that is the number that decides whether a phone folds at all. Every
  // read is wrapped in f32() at the point of use, so the arithmetic is
  // unchanged.
  const weightPrecision = shape.weightPrecision ?? "f32";
  if (!["f32", "f16"].includes(weightPrecision)) {
    throw new RangeError(`unknown transition weight precision ${weightPrecision}`);
  }
  const weight16 = weightPrecision === "f16";
  const w = (e) => (weight16 ? `f32(${e})` : e);
  const stageVector = stage16
    ? (lanes === 1 ? "f16" : `vec${lanes}<f16>`) : vector;
  const narrow = (e) => (stage16 ? `${stageVector}(${e})` : e);
  const widen = (e) => (stage16 ? `${vector}(${e})` : e);
  const zero = lanes === 1 ? "0.0" : `${vector}(0.0)`;
  const overLanes = (body) =>
    Array.from({ length: lanes }, (_, l) => body(l, lanes === 1 ? "" : `.${"xyzw"[l]}`));
  const accumulator = channelsPerThread === 1 ? "g" : "out_slot * " + groups + "u + g";
  const writeAccumulator = channelsPerThread === 1 ? "g" : "write_slot * " + groups + "u + g";
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

  return `${stage16 || weight16 || acc16 ? "enable f16;\n" : ""}
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

${residual
  ? `// 🔴 ONE BINDING FOR BOTH, because WebGPU refuses a bind group that lists
// the same buffer as a read binding and a writable one. In place is what the
// residual form means here anyway: this kernel reads every row it touches into
// workgroup memory before it writes any of them.
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;`
  : `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;`}

var<workgroup> normalized: array<${stageVector}, ${groups * channels}>;
var<workgroup> gated: array<${stageVector}, ${groups * chunk}>;
var<workgroup> reduce_a: array<f32, ${WORKGROUP}>;
var<workgroup> reduce_b: array<f32, ${WORKGROUP}>;
var<workgroup> row_mean: array<f32, ${tile}>;
var<workgroup> row_inverse_std: array<f32, ${tile}>;

fn swish(value: ${vector}) -> ${vector} { return value / (${vector}(1.0) + exp(-value)); }

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
  // it is cheap next to the two matmuls, which are what the tile is for. What
  // it leaves behind is a mean and an inverse deviation per row, so that the
  // normalised tile can then be written LANES rows at a time.
  for (var t = 0u; t < TILE; t += 1u) {
    let row = min(base_row + t, ROWS - 1u);
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
    if (local == 0u) {
      row_mean[t] = mean;
      row_inverse_std[t] = inverseSqrt(variance + EPSILON);
    }
    workgroupBarrier();
  }

  // ...LANES rows of one channel per slot, which is the layout both matmuls
  // read. A row past the end is clamped rather than skipped: it contributes to
  // no output, and leaving the lane uninitialised would put a NaN in a vector
  // whose other lanes are real.
  for (var g = 0u; g < ${groups}u; g += 1u) {
    for (var c = local; c < CHANNELS; c += WORKGROUP) {
      let scale = ${w("weights[W_SCALE + c]")};
      let offset = ${w("weights[W_OFFSET + c]")};
      var packed: ${vector};
      ${overLanes((l, at) => `{
        let row${l} = min(base_row + g * ${lanes}u + ${l}u, ROWS - 1u);
        packed${at} = (input[row${l} * CHANNELS + c] - row_mean[g * ${lanes}u + ${l}u])
          * row_inverse_std[g * ${lanes}u + ${l}u] * scale + offset;
      }`).join("\n      ")}
      normalized[g * CHANNELS + c] = ${narrow("packed")};
    }
  }
  workgroupBarrier();

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
  var sum: array<${sumVector}, ${groups * channelsPerThread}>;
  for (var s = 0u; s < ${groups * channelsPerThread}u; s += 1u) { sum[s] = ${sumZero ?? zero}; }

  for (var chunk0 = 0u; chunk0 < INTERMEDIATE; chunk0 += CHUNK) {
    // ...before overwriting the chunk the previous iteration is still reading.
    workgroupBarrier();
    // 🔴 EVERY SLOT THIS INVOCATION OWNS IS ACCUMULATED AT ONCE, so the tile's
    // normalised values are read from workgroup memory once for all of them
    // rather than once each. The loop used to be slot-outer, channel-inner:
    // GROUPS shared reads and two weight reads bought 2 * GROUPS multiply-adds.
    // Blocked, BLOCK * 2 weight reads and the same GROUPS shared reads buy
    // BLOCK * 2 * GROUPS. CHUNK is what sets BLOCK, which is why the chunk that
    // wins is not simply the smallest one that fits.
    var gate: array<${vector}, ${groups * slotsPerThread}>;
    var value: array<${vector}, ${groups * slotsPerThread}>;
    for (var s = 0u; s < ${groups * slotsPerThread}u; s += 1u) {
      gate[s] = ${zero};
      value[s] = ${zero};
    }
    for (var c = 0u; c < CHANNELS; c += 1u) {
      let column = W_T1 + c * wide;
      for (var b = 0u; b < BLOCK; b += 1u) {
        // ...read once, used TILE times. That ratio is the point.
        let i = chunk0 + local + b * WORKGROUP;
        let wg = ${w("weights[column + i]")};
        let wv = ${w("weights[column + INTERMEDIATE + i]")};
        for (var g = 0u; g < ${groups}u; g += 1u) {
          let x = ${widen("normalized[g * CHANNELS + c]")};
          gate[b * ${groups}u + g] += x * wg;
          value[b * ${groups}u + g] += x * wv;
        }
      }
    }
    for (var b = 0u; b < BLOCK; b += 1u) {
      for (var g = 0u; g < ${groups}u; g += 1u) {
        gated[g * CHUNK + local + b * WORKGROUP] = ${narrow(
          "swish(gate[b * " + groups + "u + g]) * value[b * " + groups + "u + g]")};
      }
    }
    workgroupBarrier();

    // transition2 is (intermediate, channels).
    var out_slot = 0u;
    for (var c = local; c < CHANNELS; c += WORKGROUP) {
      for (var slot = 0u; slot < CHUNK; slot += 1u) {
        let w = ${w("weights[W_T2 + (chunk0 + slot) * CHANNELS + c]")};
        for (var g = 0u; g < ${groups}u; g += 1u) {
          sum[${accumulator}] += ${acc16
            ? `${sumVector}(gated[g * CHUNK + slot]) * f16(w)`
            : `${widen("gated[g * CHUNK + slot]")} * w`};
        }
      }
      out_slot += 1u;
    }
  }

  var write_slot = 0u;
  for (var c = local; c < CHANNELS; c += WORKGROUP) {
    for (var g = 0u; g < ${groups}u; g += 1u) {
      let packed = ${fromSum(`sum[${writeAccumulator}]`)};
      ${overLanes((l, at) => `{
        let row${l} = base_row + g * ${lanes}u + ${l}u;
        if (row${l} < ROWS) {
          ${residual ? "input" : "output"}[row${l} * CHANNELS + c] ${residual ? "+=" : "="} packed${at};
        }
      }`).join("\n      ")}
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
