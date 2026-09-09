import { concatenateAs, writeInto } from "../runtime/float16.js";
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
import { createStagedMatrixShader, stagedMatrixStorage } from "../runtime/matrix-linear.js";
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
export function transitionRowTile(rows, channels = STAGED_TILE_FLOATS / 8) {
  for (const tile of [8, 4, 2]) {
    if (rows / tile >= 220) return Math.max(1, Math.min(tile, Math.floor(STAGED_TILE_FLOATS / channels)));
  }
  return 1;
}

/**
 * How many floats each half of the staged block holds, whatever the shape.
 *
 * 🔴 THE TILE WAS A FUNCTION OF THE ROW COUNT ALONE, AND THE ROW COUNT IS NOT
 * WHAT FILLS THE WORKGROUP. The staged block is `tile * channels` for the
 * normalised rows plus `tile * chunk` for the gated intermediate, so the
 * CHANNELS decide what a tile costs - and at 256 channels the tile the row rule
 * picks is twice what fits well. ESMFold2's trunk runs 256 where AF3's pair
 * track runs 128, and it is 62.5% of that trunk's GPU time, so it is where this
 * showed. Swept at 90,000 rows x 256 channels by bench-transition.js, every arm
 * bit-identical (relRMS 0):
 *
 * | tile:chunk | 4:256 | 4:512 | 4:1024 | 8:128 | 8:256 | 2:1024 | 16:128 | 8:512 |
 * |---|---|---|---|---|---|---|---|---|
 * | ms | **230.9** | 245.2 | 353.5 | 369.1 | 394.8 | 396.1 | 560.2 | 773.5 |
 *
 * 8:256 is what the row rule picked; 4:256 is **1.71x** it.
 *
 * 🔴 AND 1024 REPRODUCES EVERY TUNED VALUE ALREADY IN THE TREE, which is the
 * only reason it is a rule rather than a second special case. AF3's pair track
 * measured 8:128 optimal at 128 channels and its MSA stack, its template stack
 * and both diffusion transitions land on exactly what they run today:
 *
 * | stack | channels | intermediate | tile | chunk |
 * |---|---|---|---|---|
 * | AF3 pair track | 128 | 512 | 8 | 128 |
 * | AF3 MSA stack | 64 | 256 | 8 | 128 (the workgroup floor) |
 * | AF3 template stack | 64 | 128 | 8 | 128 (the whole intermediate) |
 * | diffusion, pair | 128 | 256 | 8 | 128 |
 * | diffusion, single | 384 | 768 | 1 | 768 (the whole intermediate) |
 * | **ESMFold2 trunk** | **256** | **1024** | **4** | **256** |
 *
 * Only the last row moves. `test/transition-tile.test.js` pins the whole table,
 * because a rule that quietly re-tunes four shipped stacks while fixing a fifth
 * is not the change this is.
 */
export const STAGED_TILE_FLOATS = 1024;

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
  // Keep the staged block a constant size - see STAGED_TILE_FLOATS. The tile
  // then trades weight traffic for occupancy and not for workgroup memory as
  // well, at any channel count rather than at 128 alone.
  const wanted = Math.max(width, Math.round(STAGED_TILE_FLOATS / tile / width) * width);
  const chunk = Math.min(intermediate, wanted);
  return intermediate % chunk === 0 ? chunk : intermediate;
}

const GRID_WIDTH = 32_768;

/** The packing order of the four tensors this kernel reads. */
export const TRANSITION_ORDER =
  ["inputLayerNormScale", "inputLayerNormOffset", "transition1", "transition2"];
const ORDER = TRANSITION_ORDER;

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
    for (const name of ORDER) writeInto(target, weights[name], offsets[name]);
  });
  return { data, offsets };
}

/**
 * How wide the workgroup should be, from how many of them there will be.
 *
 * 🔴 THE DISPATCH IS ROWS ONLY, SO A SHORT TRACK CANNOT FILL A LARGE DEVICE.
 * `main` takes `base_row = (group.x + group.y * GRID_WIDTH) * TILE`, and that
 * is the whole grid - there is no second axis. So the thread count is
 * `ceil(rows / tile) * WORKGROUP` and nothing else, and at 128 the AF3 trunk's
 * SINGLE transition gets 200 rows x 128 = 25,600 threads on a device that
 * holds 221,184. It measured 2.3% of this device's arithmetic ceiling while
 * being 13.6% of the trunk's GPU time.
 *
 * Widening the workgroup is the only axis left, and it is enough. Measured on
 * bench-transition.js at the two shapes the AF3 trunk actually runs:
 *
 *     single, 200 rows x 384 x 1536:   128 -> 0.594 ms   512 -> 0.208   2.86x
 *     pair, 40000 rows x 128 x 512:    128 -> 1.098      512 -> 1.567   0.70x
 *
 * The pair track already has 5000 workgroups and a wider one only costs it
 * occupancy, so the rule has to be a function of the workgroup COUNT rather
 * than a new constant - which is what `threadTarget` is. On a device that
 * wants ~25,000 threads (an M2) it returns the 128 it always did.
 *
 * 🔴 AND 768 IS NOT A CHOICE, IT IS A WRONG ANSWER. The bench reports width
 * 768 at relRMS **0.55** against the same kernel at 128 - not slower, wrong -
 * so the widths here are powers of two and `createTransitionShader` rejects
 * anything else rather than trusting a caller. Whatever the reduction assumes,
 * it assumes it of a power of two.
 */
export function transitionWidth(rows, tile, threadTarget, width = DEFAULT_WORKGROUP) {
  if (!threadTarget) return width;
  const groups = Math.max(1, Math.ceil(rows / tile));
  for (const candidate of [512, 256]) {
    if (groups * candidate <= threadTarget * 2) return candidate;
  }
  return width;
}

export function createTransitionShader(shape, offsets, epsilon, variance) {
  const { rows, channels, factor } = shape;
  const intermediate = channels * factor;
  const tile = shape.tile ?? transitionRowTile(rows, channels);
  const WORKGROUP = shape.width
    ?? transitionWidth(rows, tile, shape.threadTarget);
  // 🔴 A POWER OF TWO OR NOTHING - see transitionWidth. 768 compiles, runs, and
  // returns relRMS 0.55.
  if ((WORKGROUP & (WORKGROUP - 1)) !== 0) {
    throw new Error(`transition workgroup ${WORKGROUP} is not a power of two`);
  }
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
  // by nothing measurable - but `w.single-transition` is 324 MiB of the 567 an
  // AF3 TRUNK keeps resident, more than every other tensor in it together,
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

/**
 * The same transition, as three passes instead of one.
 *
 * 🔴 THE FUSION STOPS PAYING AT 256 CHANNELS, WHICH IS MEASURED AND NOT
 * ASSUMED. docs/A100.md item 9 declined splitting AF3's transitions because the
 * fused kernel keeps the widened activation out of global memory, and at AF3's
 * 128 channels that is right: split into two plain GEMMs it is a WASH (1.134 ms
 * against 1.138 at 40,000 rows) and the split pays two intermediates for
 * nothing. At ESMFold2's 256 the same comparison is 7.82 ms against 16.07 - the
 * fused kernel loses by 2.05x before any matrix units are involved.
 *
 * The mechanism is the thing the fusion is for. What it holds in workgroup
 * memory is the WIDENED row, `2 * channels * factor` floats: 4 KB at 128
 * channels with a row tile of 16, 8 KB at 256 with the tile collapsed to 4 -
 * and the tile is the whole of its weight-read amortisation. Per row it is
 * 28.5 ns for AF3 and 178.6 for ESMFold2 at exactly four times the work, which
 * is 1.57x worse per multiply-add. The fusion's cost scales with the channel
 * count and its benefit does not.
 *
 * 🔴 AND THE SWISH GATE IS NOT A PASS. Fusing it into the FIRST projection's
 * epilogue is impossible at any block width - it pairs column `i` with column
 * `i + hidden`, `hidden` columns apart, which is a different workgroup - and as
 * its own elementwise pass it costs a rows x hidden tensor and 553 MB of
 * traffic at 300 tokens. `createStagedMatrixShader`'s `sourceGate` applies it
 * where the SECOND projection stages its operand, which is one extra vec4 read
 * in a loop that already runs. So it is three passes: normalise, widen,
 * contract-with-the-gate.
 *
 * @param {{rows, channels, factor}} shape
 * @param {object} offsets from packTransitionWeights
 * @param {{normalizedStorage?, wideStorage?, geometry?, weightPrecision?, matrix?}} [options]
 */
export function createTransitionSplitShaders(shape, offsets, epsilon, variance, options = {}) {
  const { rows, channels, factor } = shape;
  const intermediate = channels * factor;
  const wide = intermediate * 2;
  const normalizedStorage = options.normalizedStorage ?? "f16";
  const wideStorage = options.wideStorage ?? "f16";
  const weightPrecision = options.weightPrecision ?? "f32";
  const matrix = options.matrix ?? {};
  // 🔴 THE VEC4 STAGING NEEDS EVERY EXTENT IT FORMS AN OFFSET FROM DIVISIBLE BY
  // FOUR, AND THAT INCLUDES THE GATE'S. A vec4 read at element i returns
  // i & ~3 upward, so an odd stride shifts the value half against the gate.
  const vectorStaging = [channels, intermediate, wide].every((v) => v % 4 === 0);

  // 🔴 THE NORMALISE IS ITS OWN PASS BECAUSE A LAYER NORM IS A ROW REDUCTION
  // AND THE GEMM STAGES A K-SLICE. There is no hook in a staged matmul that can
  // see a whole row, so this is the one part of the fused kernel that has to
  // come out whole. It is cheap: `tri.normalize` is 1.5% of an ESMFold2 trunk
  // at the same shape and this is the same kernel.
  const NORMALIZE_ROWS = 8;
  const LANES = 64;
  const LANES_PER_ROW = LANES / NORMALIZE_ROWS;
  const w = (expression) => (weightPrecision === "f16" ? `f32(${expression})` : expression);
  const normalizeVariance = variance === "fast"
    ? "let variance = row_squares[slot] / f32(CHANNELS) - mean * mean;"
    : `var centered = 0.0;
    for (var c = lane; c < CHANNELS; c += LANES_PER_ROW) {
      let d = tile[slot * CHANNELS + c] - mean;
      centered += d * d;
    }
    partial_sum[local] = centered;
    workgroupBarrier();
    for (var step = LANES_PER_ROW / 2u; step > 0u; step >>= 1u) {
      if (lane < step) { partial_sum[local] += partial_sum[local + step]; }
      workgroupBarrier();
    }
    let variance = partial_sum[slot * LANES_PER_ROW] / f32(CHANNELS);`;

  const normalize = `${normalizedStorage === "f16" || weightPrecision === "f16" ? "enable f16;\n" : ""}
// 🔴 THE ROW COUNT IS A UNIFORM, NOT A CONSTANT, BECAUSE THE ROWS ARE CHUNKED.
// The widened activation is 369 MiB at 300 ESMFold2 tokens, so a caller walks
// the rows in chunks and the last one is ragged - see
// transitionSplitChunkRows. Baking the count would need a second shader for
// the tail, and the two would then be a pair that can drift.
struct NormalizeParameters { rows: u32, channels: u32, padding: vec2<u32> };
const CHANNELS: u32 = ${channels}u;
const NORMALIZE_ROWS: u32 = ${NORMALIZE_ROWS}u;
const LANES_PER_ROW: u32 = ${LANES_PER_ROW}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const W_SCALE: u32 = ${offsets.inputLayerNormScale}u;
const W_OFFSET: u32 = ${offsets.inputLayerNormOffset}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read_write> normalized: array<${normalizedStorage}>;
@group(0) @binding(3) var<uniform> p: NormalizeParameters;

// 🔴 A ROW A THREAD IS THE WRONG SHAPE FOR A LAYER NORM - a thread walking its
// own row reads CHANNELS * 4 bytes from its neighbours and pulls a cache line
// to use four bytes of it. Staged, both the load and the writeback are
// consecutive-lane-consecutive-address. Same finding as grid.normalize and
// src/triangle/shaders.js's input normalisation.
var<workgroup> tile: array<f32, ${NORMALIZE_ROWS * channels}>;
var<workgroup> partial_sum: array<f32, ${LANES}>;
var<workgroup> row_squares: array<f32, ${NORMALIZE_ROWS}>;
var<workgroup> row_mean: array<f32, ${NORMALIZE_ROWS}>;
var<workgroup> row_inverse_std: array<f32, ${NORMALIZE_ROWS}>;

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  // 🔴 THE FOLDED GRID'S y TERM. Past 32768 workgroups the caller folds into y,
  // and a kernel that reads only .x recomputes the first row block once per y
  // row and never writes its own - silently. See CLAUDE.md.
  let base_row = (group.x + group.y * GRID_WIDTH) * NORMALIZE_ROWS;
  if (base_row >= p.rows) { return; }
  let local = local_id.x;

  // The tail is zeroed rather than skipped: the reduction runs over the whole
  // staged block.
  for (var index = local; index < NORMALIZE_ROWS * CHANNELS; index += ${LANES}u) {
    let row = base_row + index / CHANNELS;
    tile[index] = select(0.0, input[row * CHANNELS + index % CHANNELS], row < p.rows);
  }
  workgroupBarrier();

  let slot = local / LANES_PER_ROW;
  let lane = local % LANES_PER_ROW;
  var total = 0.0;
  var squares = 0.0;
  for (var c = lane; c < CHANNELS; c += LANES_PER_ROW) {
    let value = tile[slot * CHANNELS + c];
    total += value;
    squares += value * value;
  }
  partial_sum[local] = total;
  workgroupBarrier();
  for (var step = LANES_PER_ROW / 2u; step > 0u; step >>= 1u) {
    if (lane < step) { partial_sum[local] += partial_sum[local + step]; }
    workgroupBarrier();
  }
  let mean = partial_sum[slot * LANES_PER_ROW] / f32(CHANNELS);
  workgroupBarrier();
  partial_sum[local] = squares;
  workgroupBarrier();
  for (var step = LANES_PER_ROW / 2u; step > 0u; step >>= 1u) {
    if (lane < step) { partial_sum[local] += partial_sum[local + step]; }
    workgroupBarrier();
  }
  if (lane == 0u) { row_squares[slot] = partial_sum[local]; }
  workgroupBarrier();
  ${normalizeVariance}
  if (lane == 0u) {
    row_mean[slot] = mean;
    row_inverse_std[slot] = inverseSqrt(variance + EPSILON);
  }
  workgroupBarrier();

  // 🔴 A LANE OWNS A PAIR OF CHANNELS, WHICH IS THE WHOLE 32-BIT WORD THEY
  // SHARE. Writing one f16 at a time makes two lanes write the two halves of
  // one word, and the hardware turns that into a read-modify-write it does not
  // charge for anywhere legible. The triangle's own layer norm beside this has
  // always paired them and says so; this did not, and measured 408 GB/s against
  // its 815 on the same shape.
  const PAIR_COUNT: u32 = CHANNELS / 2u;
  for (var word = local; word < NORMALIZE_ROWS * PAIR_COUNT; word += ${LANES}u) {
    let slot_of = word / PAIR_COUNT;
    let row = base_row + slot_of;
    if (row >= p.rows) { continue; }
    let c = (word % PAIR_COUNT) * 2u;
    let index = slot_of * CHANNELS + c;
    let centre = row_mean[slot_of];
    let inverse = row_inverse_std[slot_of];
    let low = (tile[index] - centre) * inverse
      * ${w("weights[W_SCALE + c]")} + ${w("weights[W_OFFSET + c]")};
    let high = (tile[index + 1u] - centre) * inverse
      * ${w("weights[W_SCALE + c + 1u]")} + ${w("weights[W_OFFSET + c + 1u]")};
    normalized[row * CHANNELS + c] = ${normalizedStorage}(low);
    normalized[row * CHANNELS + c + 1u] = ${normalizedStorage}(high);
  }
}`;

  const geometry = {
    blockRows: 128, blockColumns: 128, blockInner: 32,
    subgroupRows: 2, subgroupColumns: 4, ...matrix,
  };
  const staged = (extra) => createStagedMatrixShader({
    ...geometry, vectorStaging, bias: false, weightPrecision, ...extra,
  });

  return {
    normalize,
    // normalized (rows x channels) x transition1 (channels x wide) -> wide.
    wide: staged({
      sourcePrecision: normalizedStorage, outputPrecision: wideStorage,
    }),
    // ...and the gate is applied HERE, where the operand is staged, so the
    // hidden activation never exists as a tensor at all.
    down: staged({
      sourcePrecision: wideStorage, outputPrecision: "f32", residual: true,
      sourceGate: { stride: wide, offset: intermediate },
    }),
    shape: { rows, channels, intermediate, wide },
    geometry,
    tiles: { normalizeRows: NORMALIZE_ROWS, blockRows: geometry.blockRows,
             blockColumns: geometry.blockColumns },
  };
}

/**
 * The channel width at which splitting the transition starts to pay.
 *
 * 🔴 IT IS A WIDTH RULE BECAUSE THE MEASUREMENT IS MONOTONE IN THE WIDTH, and
 * the reason is mechanical: the fused kernel holds the WIDENED row in workgroup
 * memory, so its row tile - all of its weight-read amortisation - halves each
 * time the channels double. Against the fused kernel's own best tile at 200
 * tokens (tools/gpu/bench-transition.js --channels=N --arms=4,8:128,16:128,split):
 *
 *     channels    fused    split   speedup   who runs it
 *          128    1.100    0.969      1.13   AlphaFold 3
 *          256    7.156    2.588      2.77   ESMFold2
 *          384   19.606    5.288      3.71   OpenDDE
 *
 * 🔴 AND THE SPLIT IS NOT FREE: it brings back the widened activation the
 * fusion exists to avoid, chunked, which is 72 MiB of device memory at the
 * default chunk. At 256 channels that buys 1.51x on an ESMFold2 trunk; at AF3's
 * 128 it buys 1.19x on the kernel and **1.8% of the trunk** (390.7 -> 383.9 ms
 * of GPU at 400 tokens), which is not worth 72 MiB on a device that has a
 * budget. So the default declines it at 128 and takes it from 192 up, and
 * `bench-transition.js`'s `split` arm reaches the kernel either way.
 */
export const TRANSITION_SPLIT_MIN_CHANNELS = 192;

/**
 * How many rows one pass of the split transition may cover at once.
 *
 * 🔴 THE WIDENED ACTIVATION IS THE WHOLE REASON THE FUSED KERNEL EXISTS, and
 * the split brings it back: `rows * 2 * channels * factor`, which at ESMFold2's
 * 300 tokens is 369 MiB in f16 and at 600 is 1.47 GiB. Chunking the rows is
 * what keeps the split from trading a 2.9x for an allocation nothing can hold -
 * the same trade `transitionChunkRows` makes for AF2, and for the same reason.
 *
 * 🔴 AND THE CHUNK IS A SPEED KNOB AS WELL AS A MEMORY ONE, which the first
 * version of this did not say. A chunk is dispatched on its own, so a chunk
 * that does not fill the device leaves it idle - 16,384 rows is 2,048
 * workgroups of the normalise, 59% of what this card holds, six times a block.
 * Measured on an ESMFold2 trunk at 300 tokens, GPU time of the three passes:
 *
 *     budget    chunk rows   normalise   wide    down    trunk    peak
 *      64 MiB       16,384        7.77   72.2    57.8    332.7   517.8 MiB
 *     512 MiB       90,000        5.12   69.1    40.8    310.2   see below
 *
 * The `down` pass loses most - 29% - because it is the one whose dispatch is
 * smallest. So the budget is a tuning knob, `pairTransitionChunkBytes`, and a
 * device with room should raise it.
 *
 * 🔴 AND THE CHUNK IS ALIGNED TO THE BINDING, NOT ONLY TO THE TILE. Each chunk
 * BINDS the pair at a row offset, and an offset that is not a multiple of
 * `minStorageBufferOffsetAlignment` is a validation error rather than a slow
 * path. `channels * 4` is the pair's row stride, so how many rows reach a
 * 256-byte boundary depends on it.
 */
const gcd = (first, second) => {
  let a = first; let b = second;
  while (b !== 0) { const remainder = a % b; a = b; b = remainder; }
  return a;
};

export function transitionSplitChunkRows(rows, channels, factor, limits = {}) {
  const targetBytes = limits.targetBytes ?? 64 * 1024 * 1024;
  const alignment = limits.minStorageBufferOffsetAlignment ?? 256;
  const blockRows = limits.blockRows ?? 128;
  // The widened row, in f16 - the biggest of the three tensors a chunk holds.
  const rowBytes = channels * factor * 2 * 2;
  const ceiling = Math.min(limits.maxStorageBufferBindingSize ?? Infinity, targetBytes);
  if (rows * rowBytes <= ceiling) return rows;
  const pairRowBytes = channels * 4;
  const offsetRows = alignment / gcd(pairRowBytes, alignment);
  const step = blockRows * offsetRows / gcd(blockRows, offsetRows);
  const capacity = Math.floor(ceiling / rowBytes);
  if (capacity < step) {
    throw new RangeError("a split transition chunk cannot be aligned inside the binding limit");
  }
  return Math.min(rows, Math.floor(capacity / step) * step);
}

/**
 * The buffers a split transition needs beyond the pair track's own scratch.
 *
 * 🔴 THE CALLER ALLOCATES, BECAUSE encodePairTrack HAS NO ALLOCATOR AND SHOULD
 * NOT GROW ONE. Three stacks compile this track and each already owns its
 * buffers' lifetimes; handing the encoder a bag it did not make is how the pair
 * scratch works and this follows it.
 *
 * `parameters` is indexed by ROW COUNT, because a ragged tail chunk is a
 * different `rows` in the matmul uniform and there are at most two of them.
 */
export function allocateTransitionSplit(allocator, shape, keep = (a) => a) {
  const { rows, channels, factor, chunkRows, precision = "f16", offsets, label = "transition" } = shape;
  const intermediate = channels * factor;
  const width = precision === "f16" ? 2 : 4;
  const normalized = keep(allocator.allocate(
    `${label}.normalized`, chunkRows * channels * width, GPUBufferUsage.STORAGE));
  const wide = keep(allocator.allocate(
    `${label}.wide`, chunkRows * intermediate * 2 * width, GPUBufferUsage.STORAGE));
  const parameters = new Map();
  for (const count of new Set([chunkRows, rows % chunkRows].filter((c) => c > 0))) {
    parameters.set(count, {
      normalize: keep(allocator.upload(`${label}.p-norm-${count}`,
        new Uint32Array([count, channels, 0, 0]), GPUBufferUsage.UNIFORM)),
      wide: keep(allocator.upload(`${label}.p-wide-${count}`,
        new Uint32Array([count, channels, intermediate * 2, offsets.transition1, 0, 0, 0, 0]),
        GPUBufferUsage.UNIFORM)),
      down: keep(allocator.upload(`${label}.p-down-${count}`,
        new Uint32Array([count, intermediate, channels, offsets.transition2, 0, 0, 0, 0]),
        GPUBufferUsage.UNIFORM)),
    });
  }
  return { normalized, wide, parameters, chunkRows, precision };
}
