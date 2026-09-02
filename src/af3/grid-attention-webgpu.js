/**
 * AF3's triangle ("grid") self-attention on the GPU.
 *
 * Attention runs WITHIN A ROW of the pair grid: for each row and each head,
 * every position i attends over every position j of that row. The column
 * direction is the same kernel over the transposed activation, which is how AF3
 * gets both directions from one module.
 *
 * 🔴 IT IS THE NORMALISED ACTIVATION THAT IS TRANSPOSED, NOT THE PAIR. AF3
 * normalises once at the top and q, k, v and the gate all read that. Reading
 * the raw pair instead is not a crash and not a small error: it restores the
 * ~450x that the LayerNorm had removed, and the op returns a finite tensor
 * about eighty times too large.
 *
 * 🔴 THE BIAS COMES FROM THE UNTRANSPOSED ACTIVATION IN EVERY DIALECT. In the
 * column direction q/k/v read index (j,i) while the bias still reads (i,j).
 * They are computed in separate passes here for exactly that reason - fusing
 * the bias into the projection pass, which is otherwise the obvious saving,
 * silently gives it the transposed source.
 *
 * 🔴 AND THE OPENFOLD3 LINEAGE TRANSPOSES THE BIAS TOO, WHERE STOCK AF3 DOES
 * NOT. Same weights, same shapes, different answer. `swapTransposedBias` picks,
 * and it has no default.
 *
 * 🔴 THE MASK IS OVER THE KEY, AND IT IS TRANSPOSED WITH THE ACTIVATION.
 * mask[transpose ? (j,row) : (row,j)]. Using the query's mask instead leaves
 * padding attending to real positions, which changes real answers rather than
 * padded ones.
 *
 * q and k carry AF3's `transpose_weights` and v does not; the exported shapes
 * say so, (heads, dimension, channels) against (channels, heads, dimension).
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const GRID_WIDTH = 32_768;

/**
 * How many pair rows one projection workgroup handles.
 *
 * 🔴 IT IS AN ARITHMETIC-INTENSITY DIAL, not a tuning knob to leave alone. At
 * one row a thread does a single multiply per weight it loads; at ROWS it does
 * ROWS of them, and the weight traffic falls by the same factor. Eight fits the
 * activation tile in 4 KB of workgroup memory and leaves 32 accumulators a
 * thread, which is comfortable.
 */
export const PROJECT_ROWS = 8;

/**
 * How many pair rows one output-projection workgroup handles.
 *
 * 🔴 IT WAS ONE, AND THAT KERNEL READ THE WHOLE OUTPUT MATRIX PER ROW. With a
 * workgroup a row, each of 3481 workgroups pulled all 16,384 weights to project
 * one row down: 228 MB a pass, two global reads for every multiply-add. Every
 * row wants the same matrix, so a tile of them divides that traffic by the tile
 * and the gated activations it needs cost WIDTH floats of workgroup memory
 * each. Measured on the pair track's shape - see tools/gpu/bench-grid-project.js.
 */
export const PROJECT_OUT_ROWS = 8;

const ORDER = [
  "actNormScale", "actNormOffset", "pairBiasProjection",
  "qProjection", "kProjection", "vProjection", "gatingQuery", "outputProjection",
];

/**
 * 🔴 q, k AND THE GATE ARE TRANSPOSED INTO v'S LAYOUT WHEN THEY ARE PACKED, and
 * it is a memory-coalescing change rather than a mathematical one. AF3 stores
 * them as (out, channels), so the projection read them at
 * `local * CHANNELS + c` - consecutive threads reading addresses 128 floats
 * apart, which a GPU cannot coalesce into one transaction. v was already
 * (channels, out) and read at `c * WIDTH + local`, where consecutive threads
 * read consecutive addresses.
 *
 * Storing all four the same way makes every read in that loop coalesced. This
 * is the AF2 habit of packing weights in the layout the kernel wants rather
 * than the layout the checkpoint happens to use.
 */
const TRANSPOSED = new Set(["qProjection", "kProjection", "gatingQuery"]);

function transposeOutChannels(values, channels, width) {
  const out = new Float32Array(values.length);
  for (let o = 0; o < width; o += 1) {
    for (let c = 0; c < channels; c += 1) out[c * width + o] = values[o * channels + c];
  }
  return out;
}

export function packGridAttentionWeights(weights, shape = undefined) {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`grid attention weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  const width = (shape?.width) ?? weights.heads * weights.dimension;
  for (const name of ORDER) {
    const values = weights[name];
    const channels = values.length / width;
    data.set(TRANSPOSED.has(name) && Number.isInteger(channels)
      ? transposeOutChannels(values, channels, width)
      : values, offsets[name]);
  }
  return { data, offsets };
}

export function createGridAttentionShaders(shape, offsets, epsilon, variance, dialect) {
  const { n, channels, heads, dimension, transpose } = shape;
  // 🔴 THE TILES TRAVEL BACK OUT WITH THE SHADERS, as `tiles`, because the
  // dispatch divides by exactly these. A caller that reads the constants
  // instead would still compile against a shader generated from something else
  // - which is how a kernel here once processed half its rows and reported it
  // as a speedup.
  const projectRows = shape.projectRows ?? PROJECT_ROWS;
  const projectOutRows = shape.projectOutRows ?? PROJECT_OUT_ROWS;
  const width = heads * dimension;
  const pairs = n * n;
  const swapBias = transpose && dialect.swapTransposedBias;

  const common = `
const N: u32 = ${n}u;
const PAIRS: u32 = ${pairs}u;
const CHANNELS: u32 = ${channels}u;
const HEADS: u32 = ${heads}u;
const DIMENSION: u32 = ${dimension}u;
const WIDTH: u32 = ${width}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
const SCALE: f32 = ${1 / Math.sqrt(dimension)};
const W_NORM_SCALE: u32 = ${offsets.actNormScale}u;
const W_NORM_OFFSET: u32 = ${offsets.actNormOffset}u;
const W_BIAS: u32 = ${offsets.pairBiasProjection}u;
const W_Q: u32 = ${offsets.qProjection}u;
const W_K: u32 = ${offsets.kProjection}u;
const W_V: u32 = ${offsets.vProjection}u;
const W_GATE: u32 = ${offsets.gatingQuery}u;
const W_OUT: u32 = ${offsets.outputProjection}u;

fn logistic(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
`;

  // The activation index a pair row reads: (i,j) becomes (j,i) in the column
  // direction.
  const sourceRow = transpose ? "(row % N) * N + row / N" : "row";

  // 🔴 A ROW A THREAD IS THE WRONG SHAPE FOR A LAYER NORM. A thread walking its
  // own row reads addresses CHANNELS * 4 = 512 bytes from its neighbours, so
  // every lane pulls a cache line to use four bytes of it - measured at about
  // half this part's memory bandwidth. Staging a tile of rows through workgroup
  // memory makes both the load and the writeback consecutive-lane-consecutive-
  // address, and the reduction then runs over the staged copy. The same change
  // took the triangle stack's input normalisation down by 36%; see the note in
  // src/triangle/shaders.js.
  const NORMALIZE_ROWS = 8;
  const LANES_PER_ROW = 64 / NORMALIZE_ROWS;
  const normalize = `${common}
const NORMALIZE_ROWS: u32 = ${NORMALIZE_ROWS}u;
const LANES_PER_ROW: u32 = ${LANES_PER_ROW}u;
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> normalized: array<f32>;

var<workgroup> tile: array<f32, ${NORMALIZE_ROWS} * ${channels}>;
var<workgroup> partial_sum: array<f32, 64>;
var<workgroup> partial_squares: array<f32, 64>;
var<workgroup> row_mean: array<f32, ${NORMALIZE_ROWS}>;
var<workgroup> row_inverse_std: array<f32, ${NORMALIZE_ROWS}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_row = (group.x + group.y * GRID_WIDTH) * NORMALIZE_ROWS;
  if (base_row >= PAIRS) { return; }
  let local = local_id.x;

  // 🔴 THE TAIL IS ZEROED, NOT SKIPPED: PAIRS is rarely a multiple of the tile
  // and the reduction below runs over the whole staged block.
  for (var index = local; index < NORMALIZE_ROWS * CHANNELS; index += 64u) {
    let row = base_row + index / CHANNELS;
    tile[index] = select(0.0, pair[row * CHANNELS + index % CHANNELS], row < PAIRS);
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
  partial_squares[local] = squares;
  workgroupBarrier();
  if (lane == 0u) {
    var row_total = 0.0;
    var row_squares = 0.0;
    for (var l = 0u; l < LANES_PER_ROW; l += 1u) {
      row_total += partial_sum[slot * LANES_PER_ROW + l];
      row_squares += partial_squares[slot * LANES_PER_ROW + l];
    }
    let mean = row_total / f32(CHANNELS);
    ${variance === "fast"
      ? "let variance = row_squares / f32(CHANNELS) - mean * mean;"
      : `var variance = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      let d = tile[slot * CHANNELS + c] - mean;
      variance += d * d;
    }
    variance /= f32(CHANNELS);`}
    row_mean[slot] = mean;
    row_inverse_std[slot] = inverseSqrt(variance + EPSILON);
  }
  workgroupBarrier();

  for (var index = local; index < NORMALIZE_ROWS * CHANNELS; index += 64u) {
    let row = base_row + index / CHANNELS;
    if (row >= PAIRS) { continue; }
    let c = index % CHANNELS;
    normalized[row * CHANNELS + c] =
      (tile[index] - row_mean[index / CHANNELS]) * row_inverse_std[index / CHANNELS]
      * weights[W_NORM_SCALE + c] + weights[W_NORM_OFFSET + c];
  }
}`;

  // bias[h][i][j]. Laid out head-major because the attention pass reads a whole
  // row of it per (row, i, head).
  const biasPass = `${common}
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  // 🔴 THE UNTRANSPOSED ROW unless the dialect says otherwise.
  let source = ${swapBias ? "(row % N) * N + row / N" : "row"};
  let base = source * CHANNELS;
  for (var h = 0u; h < HEADS; h += 1u) {
    var total = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      total += normalized[base + c] * weights[W_BIAS + c * HEADS + h];
    }
    bias[h * PAIRS + row] = total;
  }
}`;

  // One workgroup per pair row; thread w owns output channel w of q, k, v and
  // the gate at once, so the normalised row is read from shared memory four
  // times instead of global memory four times.
  // 🔴 SEVERAL PAIR ROWS PER WORKGROUP, BECAUSE THIS IS MEMORY-BOUND AND NOT
  // COMPUTE-BOUND. One row per workgroup means each thread does ONE multiply
  // per weight it loads - an arithmetic intensity of 1, which no GPU can run
  // near its peak - and every workgroup re-reads all four 128x128 matrices.
  // Holding ROWS rows at once lets a weight loaded into a register serve all of
  // them, so the traffic falls by a factor of ROWS and the intensity rises to
  // it. This is AF2's habit of blocking a projection over a tile rather than a
  // row; measured, it is the largest remaining cost in the pairformer.
  const ROWS = projectRows;
  const overRows = (body) => Array.from({ length: ROWS }, (_, r) => body(r)).join("\n");
  const project = `${common}
const ROWS: u32 = ${ROWS}u;
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> gate: array<f32>;

// ROWS rows of activations, shared by every output channel in the workgroup.
var<workgroup> act: array<f32, ${channels} * ${ROWS}>;

@compute @workgroup_size(${width})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let tile = group.x + group.y * GRID_WIDTH;
  let first = tile * ROWS;
  if (first >= PAIRS) { return; }
  let local = local_id.x;

  for (var r = 0u; r < ROWS; r += 1u) {
    let row = first + r;
    // 🔴 THE TAIL IS ZEROED, NOT SKIPPED. PAIRS is rarely a multiple of ROWS,
    // and a thread that reads uninitialised workgroup memory for the last tile
    // would write NaN into q, k, v and the gate for real rows in the same tile.
    let source = select(0u, ${sourceRow.replace("row", "row")}, row < PAIRS);
    for (var c = local; c < CHANNELS; c += WIDTH) {
      act[r * CHANNELS + c] = select(0.0, normalized[source * CHANNELS + c], row < PAIRS);
    }
  }
  workgroupBarrier();

${overRows((r) => `  var q${r} = 0.0; var k${r} = 0.0; var v${r} = 0.0; var g${r} = 0.0;`)}

  for (var c = 0u; c < CHANNELS; c += 1u) {
    // All four are (channels, out) - see packGridAttentionWeights - so
    // consecutive threads read consecutive addresses, and each of these four
    // loads is then used ROWS times.
    let base = c * WIDTH + local;
    let wq = weights[W_Q + base];
    let wk = weights[W_K + base];
    let wv = weights[W_V + base];
    let wg = weights[W_GATE + base];
${overRows((r) => `    let x${r} = act[${r}u * CHANNELS + c];`)}
${overRows((r) => `    q${r} += x${r} * wq; k${r} += x${r} * wk; v${r} += x${r} * wv; g${r} += x${r} * wg;`)}
  }

${overRows((r) => `  if (first + ${r}u < PAIRS) {
    let index${r} = (first + ${r}u) * WIDTH + local;
    q[index${r}] = q${r};
    k[index${r}] = k${r};
    v[index${r}] = v${r};
    gate[index${r}] = g${r};
  }`)}
}`;

  // 🔴 AF2'S ATTENTION KERNEL, REWRITTEN AGAINST AF3'S OWN BINDINGS. AlphaFold
  // 2's triangle attention and this are the same operation, and AF2's kernel is
  // a tuned flash one; the version here was naive, and measured at 88% of the
  // whole pairformer - removing it took 48 blocks from 3084 ms to 358 ms.
  //
  // What it used to do, per (row, query, head): one 64-thread WORKGROUP - so
  // 18,496 of them at 68 tokens - four separate passes over the keys through
  // workgroup memory, two barrier-heavy tree reductions for the softmax, scalar
  // loads, and half the lanes idle in the output loop because the head
  // dimension is 32 and the workgroup is 64.
  //
  // What it does now, taking AF2's principles rather than its code:
  //
  //   ONE THREAD PER QUERY, not one workgroup. The dispatch is
  //   ceil(N/64) x N x heads, so a 68-token block launches 544 workgroups.
  //   THE SOFTMAX IS ONLINE. A running max and sum are rescaled as each key
  //   arrives, so the keys are read ONCE instead of four times and no
  //   workgroup memory or barrier is needed at all.
  //   THE ACCUMULATOR LIVES IN REGISTERS, as vec4s, unrolled at generation
  //   time - which is also why the loads are vectorised.
  //
  // 🔴 REWRITTEN RATHER THAN REUSED, deliberately. AF2's kernel takes a uniform
  // for its shape, folds the 1/sqrt(d) scale into the query projection, and
  // multiplies by the gate itself; AF3 has those as compile-time constants, in
  // the attend pass, and in the output projection. Binding AF2's shader in here
  // meant moving all three, and that adapter was wrong in a way that cost more
  // to find than this took to write. Nothing outside this shader changed.
  const vectors = dimension / 4;
  const unroll = (body) => Array.from({ length: vectors }, (_, t) => body(t)).join("\n");
  const attend = `${common}
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> gathered: array<vec4<f32>>;

const HD4: u32 = ${vectors}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let row = id.y;
  let head = id.z;
  if (i >= N || row >= N || head >= HEADS) { return; }

  // vec4 units: WIDTH and DIMENSION are both multiples of four, so a head's
  // slice starts on a vector boundary.
  let q_base = ((row * N + i) * HEADS + head) * HD4;
${unroll((t) => `  let qv${t} = q[q_base + ${t}u];`)}
${unroll((t) => `  var acc${t} = vec4<f32>(0.0);`)}
  var running_max = -3.0e38;
  var running_sum = 0.0;

  for (var j = 0u; j < N; j += 1u) {
    let k_base = ((row * N + j) * HEADS + head) * HD4;
    var score = 0.0;
${unroll((t) => `    score += dot(qv${t}, k[k_base + ${t}u]);`)}
    // The KEY's mask, transposed with the activation.
    let masked = mask[${transpose ? "j * N + row" : "row * N + j"}];
    var logit = score * SCALE + bias[head * PAIRS + i * N + j];
    if (masked <= 0.0) { logit = logit - 1.0e9; }

    // 🔴 THE RUNNING MAXIMUM IS WHY ONE PASS IS ENOUGH. Everything already
    // accumulated is rescaled by exp(old - new) whenever a larger logit
    // arrives, which is algebraically the same as subtracting the final maximum
    // at the end - and is what removes three passes over the keys.
    let new_max = max(running_max, logit);
    let previous = exp(running_max - new_max);
    let weight = exp(logit - new_max);
    running_sum = running_sum * previous + weight;
    running_max = new_max;
${unroll((t) => `    acc${t} = acc${t} * previous + weight * v[k_base + ${t}u];`)}
  }

${unroll((t) => `  gathered[q_base + ${t}u] = acc${t} / running_sum;`)}
}`;

  // Gate, project down, and undo the transpose so the residual lands on the
  // orientation it came from.
  const OUT_ROWS = projectOutRows;
  const overOutRows = (body) =>
    Array.from({ length: OUT_ROWS }, (_, r) => body(r)).join("\n");
  const project_out = `${common}
const OUT_ROWS: u32 = ${OUT_ROWS}u;
@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// OUT_ROWS gated rows, so one read of the output matrix serves all of them.
var<workgroup> gated: array<f32, ${width} * ${OUT_ROWS}>;

@compute @workgroup_size(${width})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let first = (group.x + group.y * GRID_WIDTH) * OUT_ROWS;
  if (first >= PAIRS) { return; }
  let local = local_id.x;

  // 🔴 THE TAIL IS ZEROED, NOT SKIPPED, for the reason the projection kernel
  // gives: PAIRS is rarely a multiple of the tile, and uninitialised workgroup
  // memory would reach real rows sharing the tile.
  for (var r = 0u; r < OUT_ROWS; r += 1u) {
    let row = first + r;
    let index = select(0u, row * WIDTH + local, row < PAIRS);
    gated[r * ${width}u + local] =
      select(0.0, gathered[index] * logistic(gate[index]), row < PAIRS);
  }
  workgroupBarrier();

  for (var c = local; c < CHANNELS; c += WIDTH) {
${overOutRows((r) => `    var sum${r} = 0.0;`)}
    for (var w = 0u; w < WIDTH; w += 1u) {
      // Consecutive threads read consecutive channels, and this one load is
      // then used OUT_ROWS times.
      let weight = weights[W_OUT + w * CHANNELS + c];
${overOutRows((r) => `      sum${r} += gated[${r}u * ${width}u + w] * weight;`)}
    }
${overOutRows((r) => `    if (first + ${r}u < PAIRS) {
      let row${r} = first + ${r}u;
      let destination${r} = ${transpose ? `(row${r} % N) * N + row${r} / N` : `row${r}`};
      output[destination${r} * CHANNELS + c] = sum${r};
    }`)}
  }
}`;

  return { normalize, bias: biasPass, project, attend, project_out,
           tiles: { projectRows, projectOutRows, normalizeRows: NORMALIZE_ROWS } };
}

export class Af3GridSelfAttentionGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} pair n*n*channels
   * @param {Float32Array} mask n*n
   * @param {{n: number, channels: number, transpose: boolean}} shape
   * @param {object} weights heads, dimension, and the eight tensors in ORDER
   * @param {{swapTransposedBias: boolean}} dialect
   * @param {{epsilon?: number, variance?: "fast"|"two-pass"}} options
   */
  async run(pair, mask, shape, weights, dialect, options = {}) {
    const { n, channels, transpose } = shape;
    const { heads, dimension } = weights;
    if (!Number.isInteger(heads) || !Number.isInteger(dimension)) {
      throw new Error("weights.heads and weights.dimension must be integers");
    }
    if (dialect?.swapTransposedBias === undefined) {
      throw new Error("dialect.swapTransposedBias has no default: stock AF3 is false, "
        + "the openfold3 lineage true");
    }
    const width = heads * dimension;
    const pairs = n * n;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";
    if (pair.length !== pairs * channels) {
      throw new Error(`pair has ${pair.length} elements; expected ${pairs * channels}`);
    }

    const packed = packGridAttentionWeights(weights);
    const sources = createGridAttentionShaders(
      { n, channels, heads, dimension, transpose }, packed.offsets, epsilon, variance, dialect);
    const key = `af3-grid:${n}:${channels}:${heads}:${dimension}:${transpose}`
      + `:${epsilon}:${variance}:${dialect.swapTransposedBias}`;
    const [normalize, bias, project, attend, projectOut] = await Promise.all([
      this.pipelines.get(`${key}:normalize`, sources.normalize),
      this.pipelines.get(`${key}:bias`, sources.bias),
      this.pipelines.get(`${key}:project`, sources.project),
      this.pipelines.get(`${key}:attend`, sources.attend),
      this.pipelines.get(`${key}:project-out`, sources.project_out),
    ]);

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    const linear2d = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
    try {
      const pairBuffer = keep(this.allocator.upload("af3-grid.pair", pair, storage));
      const maskBuffer = keep(this.allocator.upload("af3-grid.mask", mask, storage));
      const weightBuffer = keep(this.allocator.upload("af3-grid.weights", packed.data, storage));
      const normalized = keep(this.allocator.allocate("af3-grid.normalized", pairs * channels * 4, storage));
      const biasBuffer = keep(this.allocator.allocate("af3-grid.bias", heads * pairs * 4, storage));
      const q = keep(this.allocator.allocate("af3-grid.q", pairs * width * 4, storage));
      const k = keep(this.allocator.allocate("af3-grid.k", pairs * width * 4, storage));
      const v = keep(this.allocator.allocate("af3-grid.v", pairs * width * 4, storage));
      const gate = keep(this.allocator.allocate("af3-grid.gate", pairs * width * 4, storage));
      const gathered = keep(this.allocator.allocate("af3-grid.gathered", pairs * width * 4, storage));
      const output = keep(this.allocator.allocate(
        "af3-grid.output", pairs * channels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "af3-grid.readback", pairs * channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-grid-attention" });
      const runPass = (label, pipeline, buffers, groups) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(groups[0], groups[1], groups[2] ?? 1);
        pass.end();
      };

      runPass("normalize", normalize, [pairBuffer, weightBuffer, normalized],
              linear2d(Math.ceil(pairs / sources.tiles.normalizeRows)));
      runPass("bias", bias, [normalized, weightBuffer, biasBuffer],
              linear2d(Math.ceil(pairs / 64)));
      // One workgroup per tile of PROJECT_ROWS pair rows - see the kernel.
      runPass("project", project, [normalized, weightBuffer, q, k, v, gate],
              linear2d(Math.ceil(pairs / sources.tiles.projectRows)));
      // One thread per (query, row, head) - see the note on the kernel.
      runPass("attend", attend, [q, k, v, biasBuffer, maskBuffer, gathered],
              [Math.ceil(n / 64), n, heads]);
      // One workgroup per tile of PROJECT_OUT_ROWS pair rows - see the kernel.
      runPass("project-out", projectOut, [gathered, gate, weightBuffer, output],
              linear2d(Math.ceil(pairs / sources.tiles.projectOutRows)));
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, pairs * channels * 4);

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
