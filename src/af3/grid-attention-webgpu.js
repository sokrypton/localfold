import { storageArray, storedElement, storedPair } from "../runtime/storage.js";
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

/** The row tile the pair-owning packed store wants; see createGridAttentionShaders. */
export const PACKED_PROJECT_ROWS = 4;

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

/**
 * 🔴 THE ATTENTION IS CUBIC IN N AND EVERYTHING AROUND IT IS QUADRATIC, so
 * which kernel is worth attacking depends on the protein. At 59 tokens
 * `grid.attend` is 39 ms of a 398 ms pairformer; at 150 it is 563 of 2429 - the
 * largest in the trunk by half again - and it keeps growing.
 *
 * 🔴 AND CARRYING MORE THAN ONE QUERY AN INVOCATION IS NOT THE ANSWER, which is
 * the obvious thing to try: it reads DIMENSION/4 vectors of k and as many of v
 * per key and does the same number of vector operations with them, one load per
 * multiply-add, and those loads do not depend on the query. Two queries an
 * invocation should have halved them. Measured at 150 tokens it was 1.85x
 * SLOWER, and four queries 3.9x - a query costs DIMENSION/4 vectors of q plus
 * as many accumulators, so two is already 128 floats of register and it spills.
 * The kernel keeps one query a thread.
 */

/**
 * 🔴 q, k, v AND THE GATE ARE PACKED AS ONE INTERLEAVED BLOCK OF vec4, not as
 * four matrices. The projection kernel contracts all four over the same
 * activation and reads them at the same (channel, out) cell, so interleaved
 * they are one 16-byte load and one vector multiply-add where they were four of
 * each - the same arithmetic in a quarter of the issue slots and a quarter of
 * the memory transactions. `qkvgProjection` is that block; the four names it
 * replaces are still what the checkpoint calls them, and packing is where they
 * meet.
 */
const ORDER = [
  "actNormScale", "actNormOffset", "pairBiasProjection",
  "qkvgProjection", "outputProjection",
];

/**
 * How many keys the attention stages in workgroup memory at once.
 *
 * 🔴 EVERY LANE OF A WORKGROUP READS THE SAME KEY AND THE SAME VALUE. The
 * dispatch gives a workgroup one (pair row, head) and sixty-four queries, and
 * the key loop is over the same axis for all of them - so before this, sixty-
 * four lanes issued sixty-four identical global loads for each of the
 * 2 * dimension/4 vectors a key needs. Staging a chunk of keys makes that one
 * load and sixty-four workgroup reads: at 150 tokens the kernel went 5.75 ms to
 * 3.05, chunk 16 and 32 tying and 64 losing at 4.25 because 16 KiB a workgroup
 * costs residency.
 *
 * The bound is 8 KiB, which is 32 keys at the trunk's head dimension of 32 and
 * scales down with it, so the shape stays inside the 16 KiB a conservative
 * adapter grants and leaves room for nothing else - this kernel has no other
 * workgroup memory.
 */
export function attendKeyChunk(dimension) {
  const vectorsPerKey = 2 * (dimension / 4);
  for (const chunk of [32, 16, 8]) {
    if (chunk * vectorsPerKey * 16 <= 8192) return chunk;
  }
  return 0;
}

/** The four, in the vec4 lane order the shader reads them in. */
const QKVG = ["qProjection", "kProjection", "vProjection", "gatingQuery"];

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
  const width = (shape?.width) ?? weights.heads * weights.dimension;
  const sizeOf = (name) => name === "qkvgProjection"
    ? QKVG.reduce((total, part) => total + weights[part].length, 0)
    : weights[name].length;
  for (const name of [...ORDER.filter((n) => n !== "qkvgProjection"), ...QKVG]) {
    if (weights[name] === undefined) throw new Error(`grid attention weights missing ${name}`);
  }
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    offsets[name] = total;
    total += sizeOf(name);
  }
  const data = new Float32Array(total);
  const laid = (name) => {
    const values = weights[name];
    const channels = values.length / width;
    return TRANSPOSED.has(name) && Number.isInteger(channels)
      ? transposeOutChannels(values, channels, width)
      : values;
  };
  for (const name of ORDER) {
    if (name !== "qkvgProjection") { data.set(laid(name), offsets[name]); continue; }
    // ...(channels, out) for each, interleaved into (channels, out, 4).
    const parts = QKVG.map(laid);
    const base = offsets[name];
    for (let lane = 0; lane < 4; lane += 1) {
      const values = parts[lane];
      for (let index = 0; index < values.length; index += 1) {
        data[base + index * 4 + lane] = values[index];
      }
    }
  }
  return { data, offsets };
}

/**
 * @param {"f32"|"f16"} [normalizedStorage] how the layer-normed pair between
 *   `normalize` and its three readers - `bias`, `project` and the triangle's
 *   own projections, which share the buffer - is stored.
 * @param {"f32"|"f16"} [gatheredStorage] how the attended result between
 *   `attend` and `project_out` is stored. It is one of the seven pair-sized
 *   scratch buffers the pairformer block shares, and the ONLY one no other
 *   operation touches - which is why it is the one converted first. See
 *   src/runtime/storage.js.
 */
export function createGridAttentionShaders(
  shape, offsets, epsilon, variance, dialect, gatheredStorage = "f32",
  normalizedStorage = "f32", qkvgStorage = "f32",
) {
  // 🔴 PER TENSOR, NOT PER KERNEL. `project` writes q, k, v and the gate from
  // one dispatch, so at first they had one storage between them - which makes
  // scratch[1..4] a single all-or-nothing step, and the triangle side of those
  // four is four kernels of varying difficulty. Each output carries its own
  // storage so they can be turned on one at a time and each checked against
  // AF3's own tensors. A string still means all four, which is what every
  // caller outside the pair track passes.
  const store4 = typeof qkvgStorage === "string"
    ? { q: qkvgStorage, k: qkvgStorage, v: qkvgStorage, gate: qkvgStorage }
    : { q: "f32", k: "f32", v: "f32", gate: "f32", ...qkvgStorage };
  const packGathered = gatheredStorage === "f16";
  const packNormalized = normalizedStorage === "f16";
  // 🔴 PACKING q/k/v/gate IS A CHANGE TO WHO OWNS AN OUTPUT CHANNEL, not just
  // to the buffer. `project` gives each lane ONE channel and consecutive lanes
  // consecutive channels, so the two halves of a packed word are produced by
  // two different lanes - and WGSL cannot write sixteen bits, so neither could
  // store its half without reading the other's back. Under packing a lane takes
  // a PAIR of adjacent channels instead: half the lanes, twice the
  // accumulators, one word written where two floats were.
  //
  // Twice the accumulators is the risk, and it is why this is measured rather
  // than assumed - src/evoformer/attention.js records a tile in AF2 that got
  // SLOWER when its accumulators doubled, because sixteen vec4 spill.
  // tools/gpu/bench-grid-project.js races the two.
  // The ownership change below is needed as soon as ANY of the four is packed:
  // it is a property of the store, and one kernel has one store loop.
  const packQkvg = Object.values(store4).some((value) => value === "f16");
  /** A vec4 of four consecutive elements of `name`, whatever its storage. */
  const vec4Of = (name, index) => (store4[name] === "f16"
    ? `load4(${name}[${index}])` : `${name}[${index}]`);
  const needsLoad4 = ["q", "k", "v"].some((name) => store4[name] === "f16");
  const { n, channels, heads, dimension, transpose } = shape;
  // 🔴 THE TILES TRAVEL BACK OUT WITH THE SHADERS, as `tiles`, because the
  // dispatch divides by exactly these. A caller that reads the constants
  // instead would still compile against a shader generated from something else
  // - which is how a kernel here once processed half its rows and reported it
  // as a speedup.
  // ...and whether project-out adds into its target instead of overwriting it;
  // see the note in src/af3/transition-webgpu.js for why that removes a pass.
  const residual = shape.residual ?? false;
  // 🔴 THE PACKED STORE WANTS A NARROWER ROW TILE, AND THAT IS MEASURED. The
  // pair-owning form holds twice the accumulators, so the tile that is fastest
  // in f32 is past the register budget once packed. `project` in ms at 272
  // tokens, medians of nine with the arms interleaved
  // (tools/gpu/bench-grid-project.js --arms=4,4p,8,8p,12,12p,16,16p):
  //
  //     rows      4      8     12      16
  //     f32    8.16   7.54   9.93   10.14
  //     packed 8.21  11.01  13.73   42.64
  //
  // Four is free and sixteen is a 4.2x cliff - the spill AF2's own projection
  // sweep records in src/evoformer/attention.js. `projectOutRows` is a separate
  // knob and stays at eight, where it is fastest either way.
  const projectRows = shape.projectRows ?? (packQkvg ? PACKED_PROJECT_ROWS : PROJECT_ROWS);
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
// ...as a vec4 index, which is why the packed offset must be a multiple of 4.
const W_QKVG: u32 = ${offsets.qkvgProjection / 4}u;
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
@group(0) @binding(2) var<storage, read_write> normalized: array<${storageArray(normalizedStorage)}>;

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

  // A lane owns a PAIR of channels, so it owns the whole word they share; see
  // src/runtime/storage.js. CHANNELS is 128 here and even in every dialect.
  const PAIR_COUNT: u32 = CHANNELS / 2u;
  for (var word = local; word < NORMALIZE_ROWS * PAIR_COUNT; word += 64u) {
    let slot_of = word / PAIR_COUNT;
    let row = base_row + slot_of;
    if (row >= PAIRS) { continue; }
    let c = (word % PAIR_COUNT) * 2u;
    let index = slot_of * CHANNELS + c;
    let centre = row_mean[slot_of];
    let scaled = row_inverse_std[slot_of];
    let low = (tile[index] - centre) * scaled
      * weights[W_NORM_SCALE + c] + weights[W_NORM_OFFSET + c];
    let high = (tile[index + 1u] - centre) * scaled
      * weights[W_NORM_SCALE + c + 1u] + weights[W_NORM_OFFSET + c + 1u];
    let pair_word = row * PAIR_COUNT + (word % PAIR_COUNT);
    ${storedPair(normalizedStorage, "normalized", "pair_word", "low", "high")}
  }
}`;

  // bias[h][i][j]. Laid out head-major because the attention pass reads a whole
  // row of it per (row, i, head).
  const biasPass = `${common}
@group(0) @binding(0) var<storage, read> normalized: array<${storageArray(normalizedStorage)}>;
// ...as vec4, which is why W_BIAS and HEADS must both be multiples of four.
@group(0) @binding(1) var<storage, read> projection: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  // 🔴 THE UNTRANSPOSED ROW unless the dialect says otherwise.
  let source = ${swapBias ? "(row % N) * N + row / N" : "row"};
  let base = source * CHANNELS;
  // 🔴 THE HEADS ARE CONTIGUOUS IN THE PROJECTION, SO THEY ARE THE VECTOR. This
  // looped heads OUTSIDE channels, re-reading the normalised row for each of
  // them and reading one weight per multiply-add.
  ${Array.from({ length: heads / 4 }, (_, h) =>
    `var total${h} = vec4<f32>(0.0);`).join("\n  ")}
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let value = ${storedElement(normalizedStorage, "normalized", "base + c")};
    let column = (W_BIAS + c * HEADS) / 4u;
    ${Array.from({ length: heads / 4 }, (_, h) =>
      `total${h} += value * projection[column + ${h}u];`).join("\n    ")}
  }
  ${Array.from({ length: heads / 4 }, (_, h) => Array.from({ length: 4 }, (_, l) =>
    `bias[(${h * 4 + l}u) * PAIRS + row] = total${h}.${"xyzw"[l]};`).join("\n  ")).join("\n  ")}
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
  const projectLanes = packQkvg ? width / 2 : width;
  const project = `${common}
const ROWS: u32 = ${ROWS}u;
const LANES: u32 = ${projectLanes}u;
@group(0) @binding(0) var<storage, read> normalized: array<${storageArray(normalizedStorage)}>;
// 🔴 THE ONLY KERNEL HERE THAT READS THE WEIGHTS AS vec4, and it can because it
// reads nothing but the interleaved q/k/v/gate block. The other three passes
// bind the same buffer as scalars.
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> q: array<${storageArray(store4.q)}>;
@group(0) @binding(3) var<storage, read_write> k: array<${storageArray(store4.k)}>;
@group(0) @binding(4) var<storage, read_write> v: array<${storageArray(store4.v)}>;
@group(0) @binding(5) var<storage, read_write> gate: array<${storageArray(store4.gate)}>;

// ROWS rows of activations, shared by every output channel in the workgroup.
//
// 🔴 STAGING THESE TRANSPOSED - FOUR ROWS TO A VECTOR - IS THE CHANGE THAT WON
// FOUR TIMES IN AF2 AND LOSES HERE. The weights are already one vec4 a cell and
// the activations are read one float at a time, once per row, so an invocation
// spends one global read and eight workgroup reads to buy eight vector
// multiply-adds: 1.88 useful operations an instruction, where four rows to a
// vector would make it 2.9 - a 35% cut in instructions, and the identical fix
// took AF2's transition, its q/k/v/gate projection and its attention output
// projection 1.18x, 1.18x and 1.52x.
//
// Measured on the trunk at 150 tokens, where bench-trunk.js --profile
// reproduces a kernel to about 0.3% across processes: grid.project 239.1 ->
// 243.6 ms, 2% WORSE, with every other kernel inside 1% and the trunk total
// unmoved at 2324 ms. So this kernel is not waiting on its workgroup reads,
// and the instruction count does not govern it - the one global weight read a
// channel does. Do not transpose the activation tile.
var<workgroup> act: array<f32, ${channels} * ${ROWS}>;

@compute @workgroup_size(${projectLanes})
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
    for (var c = local; c < CHANNELS; c += LANES) {
      act[r * CHANNELS + c] = select(
        0.0, ${storedElement(normalizedStorage, "normalized", "source * CHANNELS + c")}, row < PAIRS);
    }
  }
  workgroupBarrier();

${packQkvg ? `  // Two accumulators a row: this lane's pair of adjacent channels.
  let c0 = local * 2u;
${overRows((r) => `  var lo${r} = vec4<f32>(0.0);
  var hi${r} = vec4<f32>(0.0);`)}

  for (var c = 0u; c < CHANNELS; c += 1u) {
    let base = W_QKVG + c * WIDTH + c0;
    let wlo = weights[base];
    let whi = weights[base + 1u];
${overRows((r) => `    let a${r} = act[${r}u * CHANNELS + c];
    lo${r} += a${r} * wlo;
    hi${r} += a${r} * whi;`)}
  }

${overRows((r) => {
    const emit = ([name, lane]) => (store4[name] === "f16"
      ? `    ${name}[word${r}] = pack2x16float(vec2<f32>(lo${r}.${lane}, hi${r}.${lane}));`
      : `    ${name}[word${r} * 2u] = lo${r}.${lane};\n`
        + `    ${name}[word${r} * 2u + 1u] = hi${r}.${lane};`);
    return `  if (first + ${r}u < PAIRS) {
    // A packed pair shares one word and this lane owns both halves of it; an
    // unpacked one is the same two values written where they always were,
    // since word * 2 is the channel c0 this lane owns.
    let word${r} = (first + ${r}u) * LANES + local;
${[["q", "x"], ["k", "y"], ["v", "z"], ["gate", "w"]].map(emit).join("\n")}
  }`;
  })}` : `  // One accumulator a row, holding (q, k, v, gate).
${overRows((r) => `  var acc${r} = vec4<f32>(0.0);`)}

  for (var c = 0u; c < CHANNELS; c += 1u) {
    // (channels, out, 4) - see packGridAttentionWeights - so consecutive
    // threads read consecutive vec4s, and this one load is used ROWS times.
    let w = weights[W_QKVG + c * WIDTH + local];
${overRows((r) => `    acc${r} += act[${r}u * CHANNELS + c] * w;`)}
  }

${overRows((r) => `  if (first + ${r}u < PAIRS) {
    let index${r} = (first + ${r}u) * WIDTH + local;
    q[index${r}] = acc${r}.x;
    k[index${r}] = acc${r}.y;
    v[index${r}] = acc${r}.z;
    gate[index${r}] = acc${r}.w;
  }`)}`}
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
  // How many keys are staged in workgroup memory at once; 0 reads them from
  // global. Every lane of a workgroup shares (row, head) and therefore reads
  // the SAME k and v, so staging replaces 64 identical global loads with one.
  const keyChunk = shape.attendKeyChunk ?? attendKeyChunk(dimension);
  const staged = keyChunk > 0;
  // 🔴 AND THE STAGED COPY GOES IN f16 WHERE THE DEVICE ALLOWS IT. This is the
  // same kernel and the same finding as AF2's - see
  // src/evoformer/attention.js, where the staged reads priced out at 8.7 ms of
  // 20.8 and narrowing them to f16 was worth 1.22x. Only the TILE narrows: the
  // running max, the running sum, the logit and the accumulators stay f32,
  // because the softmax is where the range is.
  const stagedPrecision = shape.stagedPrecision ?? "f32";
  if (!["f32", "f16"].includes(stagedPrecision)) {
    throw new RangeError(`unknown grid attention staged precision ${stagedPrecision}`);
  }
  const tile16 = staged && stagedPrecision === "f16";
  // 🔴 OFF, MEASURED. See the note in the kernel: skipping the rescale when the
  // running maximum does not move is arithmetically the obvious win and is a
  // LOSS on this device. `attendLazyRescale: true` is the arm
  // bench-grid-attend.js compares against, kept so the result is reproducible
  // rather than remembered.
  const lazyRescale = shape.attendLazyRescale === true;
  const tileType = tile16 ? "vec4<f16>" : "vec4<f32>";
  const widen = (e) => (tile16 ? `vec4<f32>(${e})` : e);
  const readK = (t) => (staged ? widen(`k_tile[slot * HD4 + ${t}u]`) : vec4Of("k", `k_base + ${t}u`));
  const readV = (t) => (staged ? widen(`v_tile[slot * HD4 + ${t}u]`) : vec4Of("v", `k_base + ${t}u`));
  const body = `
${staged ? "" : "    let k_base = ((row * N + j) * HEADS + head) * HD4;"}
    var score = 0.0;
${unroll((t) => `    score += dot(qv${t}, ${readK(t)});`)}
    // The KEY's mask, transposed with the activation.
    let masked = mask[${transpose ? "j * N + row" : "row * N + j"}];
    // 🔴 THE MASK STAYS A CONDITIONAL SUBTRACTION FROM THIS LOGIT. Lifting it
    // to a penalty computed once for the key and ADDED - either as
    // select(-1.0e9, 0.0, masked > 0.0) or as a plain var set in an if - is
    // algebraically the same expression and measured relRMS 2.24e-1 against the
    // CPU reference where this measures 9.63e-7, deterministically, and to the
    // same wrong value both ways. It was not run down; it is not needed; do not
    // rewrite it.
    var logit = score * SCALE + bias[head * PAIRS + i * N + j];
    if (masked <= 0.0) { logit = logit - 1.0e9; }

    // 🔴 THE RUNNING MAXIMUM IS WHY ONE PASS IS ENOUGH. Everything already
    // accumulated is rescaled by exp(old - new) whenever a larger logit
    // arrives, which is algebraically the same as subtracting the final maximum
    // at the end - and is what removes three passes over the keys.
${lazyRescale ? `    // 🔴 THE ARM THAT DOES NOT PAY, KEPT SO IT STAYS MEASURED. Written straight
    // through, every key costs acc = acc * previous + weight * v on all
    // ${vectors} accumulators - a multiply and a fused multiply-add - and on the keys
    // that do not raise the running maximum, which is nearly all of them once a
    // few have been seen, previous is exp(0) and the multiply is by one. Making
    // that conditional takes the inner loop from ${vectors * 3} vector operations a key
    // to ${vectors * 2} and skips an exp with them, and it is SLOWER:
    //
    //     tokens    lazy    always   speedup
    //        128   11.3 ms   11.6      1.027
    //        256   44.6      43.3      0.971
    //        400  131.1     125.5      0.957
    //
    // 🔴 BECAUSE THE BRANCH IS NOT PER LANE IN PRACTICE. The 64 lanes of a
    // workgroup share (row, head) and hold 64 different queries, each with its
    // own maximum - so the block runs whenever ANY of them needs it, which over
    // 64 lanes is most keys. What is left is a compare, a lost fused
    // multiply-add, and the divergence itself.
    let new_max = max(running_max, logit);
    if (new_max > running_max) {
      let previous = exp(running_max - new_max);
      running_sum = running_sum * previous;
${unroll((t) => `      acc${t} = acc${t} * previous;`)}
      running_max = new_max;
    }
    let weight = exp(logit - running_max);
    running_sum = running_sum + weight;
${unroll((t) => `    acc${t} = acc${t} + weight * ${readV(t)};`)}` : `    let new_max = max(running_max, logit);
    let previous = exp(running_max - new_max);
    let weight = exp(logit - new_max);
    running_sum = running_sum * previous + weight;
    running_max = new_max;
${unroll((t) => `    acc${t} = acc${t} * previous + weight * ${readV(t)};`)}`}`;

  const loop = staged ? `
  for (var j0 = 0u; j0 < N; j0 += ${keyChunk}u) {
    // ...before overwriting the tile the previous chunk is still reading.
    workgroupBarrier();
    for (var index = local; index < ${keyChunk}u * HD4; index += 64u) {
      let j = min(j0 + index / HD4, N - 1u);
      let source = ((row * N + j) * HEADS + head) * HD4 + index % HD4;
      k_tile[index] = ${tileType}(${vec4Of("k", "source")});
      v_tile[index] = ${tileType}(${vec4Of("v", "source")});
    }
    workgroupBarrier();
    for (var slot = 0u; slot < ${keyChunk}u; slot += 1u) {
      let j = j0 + slot;
      if (j >= N) { break; }
${body}
    }
  }` : `
  for (var j = 0u; j < N; j += 1u) {
${body}
  }`;

  const attend = `${tile16 ? "enable f16;\n" : ""}${common}
@group(0) @binding(0) var<storage, read> q: array<${store4.q === "f16" ? "vec2<u32>" : "vec4<f32>"}>;
@group(0) @binding(1) var<storage, read> k: array<${store4.k === "f16" ? "vec2<u32>" : "vec4<f32>"}>;
@group(0) @binding(2) var<storage, read> v: array<${store4.v === "f16" ? "vec2<u32>" : "vec4<f32>"}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> gathered: array<${packGathered ? "vec2<u32>" : "vec4<f32>"}>;
${packGathered ? `
fn store4(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}` : ""}${needsLoad4 ? `
fn load4(w: vec2<u32>) -> vec4<f32> {
  let lo = unpack2x16float(w.x);
  let hi = unpack2x16float(w.y);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}` : ""}

const HD4: u32 = ${vectors}u;
${staged ? `var<workgroup> k_tile: array<${tileType}, ${keyChunk * vectors}>;
var<workgroup> v_tile: array<${tileType}, ${keyChunk * vectors}>;` : ""}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  // 🔴 THE WORKGROUP id, NOT THE GLOBAL ONE, because the uniformity analysis
  // has to SEE that row and head are workgroup-uniform: a barrier inside a
  // branch on a global id is rejected, and this kernel's staging loop has two.
  let i = group.x * 64u + local_id.x;
  let row = group.y;
  let head = group.z;
  // 🔴 row AND head ARE WORKGROUP-UNIFORM AND i IS NOT, which is why only the
  // first two are a return. A lane past the end still has to reach every
  // barrier the staging loop makes; it is stopped at the write instead.
  if (row >= N || head >= HEADS) { return; }
  let live = i < N;
  let local = local_id.x;

  // vec4 units: WIDTH and DIMENSION are both multiples of four, so a head's
  // slice starts on a vector boundary.
  let q_base = ((row * N + select(0u, i, live)) * HEADS + head) * HD4;
${unroll((t) => `  let qv${t} = ${vec4Of("q", `q_base + ${t}u`)};`)}
${unroll((t) => `  var acc${t} = vec4<f32>(0.0);`)}
  var running_max = -3.0e38;
  var running_sum = 0.0;
${loop}

  if (live) {
${unroll((t) => packGathered
    ? `    gathered[q_base + ${t}u] = store4(acc${t} / running_sum);`
    : `    gathered[q_base + ${t}u] = acc${t} / running_sum;`)}
  }
}`;

  // Gate, project down, and undo the transpose so the residual lands on the
  // orientation it came from.
  const OUT_ROWS = projectOutRows;
  const overOutRows = (body) =>
    Array.from({ length: OUT_ROWS }, (_, r) => body(r)).join("\n");
  const project_out = `${common}
const OUT_ROWS: u32 = ${OUT_ROWS}u;
@group(0) @binding(0) var<storage, read> gathered: array<${storageArray(gatheredStorage)}>;
@group(0) @binding(1) var<storage, read> gate: array<${storageArray(store4.gate)}>;
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
      select(0.0, ${storedElement(gatheredStorage, "gathered", "index")}
        * logistic(${storedElement(store4.gate, "gate", "index")}), row < PAIRS);
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
      output[destination${r} * CHANNELS + c] ${residual ? "+=" : "="} sum${r};
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
    const stagedPrecision = options.stagedPrecision ?? "f32";
    // ...forceable, so both arms can be timed in ONE process. See the note on
    // the lazy rescale in the attend kernel, and bench-grid-attend.js.
    const attendLazyRescale = options.attendLazyRescale !== false;
    const sources = createGridAttentionShaders(
      { n, channels, heads, dimension, transpose, stagedPrecision, attendLazyRescale },
      packed.offsets, epsilon, variance, dialect);
    const key = `af3-grid:${n}:${channels}:${heads}:${dimension}:${transpose}`
      + `:${epsilon}:${variance}:${dialect.swapTransposedBias}:${stagedPrecision}`
      + `:${attendLazyRescale}`;
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
