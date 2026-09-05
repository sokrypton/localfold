import { storageArray, storedElement, storedPair } from "../runtime/storage.js";
const declaration = (precision) => precision === "f16" ? "enable f16;\n" : "";
const scalar = (precision) => precision;
const read = (precision, expression) =>
  precision === "f16" ? `f32(${expression})` : expression;

function prelude(shape, precision, offsets, epsilon, weightPrecision = precision) {
  const offsetConstants = Object.entries(offsets)
    .map(([name, offset]) => `const W_${name.toUpperCase()}: u32 = ${offset}u;`)
    .join("\n");
  return `${declaration(precision === "f16" || weightPrecision === "f16"
    || shape.accumulatePrecision === "f16" ? "f16" : "f32")}
const L: u32 = ${shape.length}u;
const CZ: u32 = ${shape.cZ}u;
const CH: u32 = ${shape.cHidden}u;
const PAIRS: u32 = L * L;
const LINEAR_GRID_WIDTH: u32 = ${LINEAR_GRID_WIDTH}u;
// 🔴 THE PROJECTIONS' OWN, BECAUSE IT HAS TO BE FORCEABLE. Exceeding 65535 row
// tiles needs about 1450 residues, and the contraction is O(n^3) - no CPU
// reference can follow a differential at that size, so the z path could only
// ever be checked by folding a real 1500-mer and hoping. Lowering this instead
// puts group.z > 0 at 68 residues, where the reference is a few milliseconds.
const PROJECT_GRID_WIDTH: u32 = ${shape.projectGridWidth ?? LINEAR_GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon.toPrecision(9)};
${offsetConstants}

fn logistic(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value));
}
`;
}

/**
 * The two LayerNorm variance formulas, which are equal in algebra and not in
 * floating point.
 *
 * 🔴 AF2 AND AF3 DISAGREE HERE AND NEITHER IS WRONG. AF2 takes a second pass to
 * average `(x - mean)^2`. AF3's trunk sets `use_fast_variance=True`, which is
 * `E[x^2] - E[x]^2` in one pass; its atom and diffusion stacks set it False and
 * take AF2's route. So the formula is a property of the CALLER, not of this
 * file, and it cannot be a constant. Picking the wrong one still returns
 * plausible numbers - the gap is ~1e-7, far under AF3's own bfloat16 floor -
 * which is exactly why it has to be chosen explicitly rather than inherited.
 *
 * @param {"two-pass"|"fast"} variance
 * @param {string} count loop bound, "CZ" or "CH"
 * @param {(index: string) => string} at how to read element `index`
 */
function varianceCode(variance, count, at) {
  if (variance === "fast") {
    return `var sum_squares = 0.0;
  for (var c = 0u; c < ${count}; c += 1u) {
    let value = ${at("c")};
    sum_squares += value * value;
  }
  let variance = sum_squares / f32(${count}) - mean * mean;`;
  }
  return `var variance = 0.0;
  for (var c = 0u; c < ${count}; c += 1u) {
    let centered = ${at("c")} - mean;
    variance += centered * centered;
  }
  variance /= f32(${count});`;
}

/**
 * The pair rows and hidden channels one workgroup of the two projection kernels
 * covers. Both are eight times the per-invocation register block.
 *
 * 🔴 THE DISPATCH IS DIVIDED BY THESE AND THE SHADER IS GENERATED FROM THEM, so
 * they are returned from createTriangleShaders as `projectTile` rather than
 * exported for callers to look up. Resolving a kernel's shape in two places is
 * what once had the shaders tiling by four under a dispatch dividing by eight -
 * half the work silently skipped, reported as a 30% speedup.
 */
const PROJECT_TILE = { rows: 32, columns: 16 };

/**
 * The same idea for the contraction, and it does NOT peak where the projection
 * does. 32x32 against 8x8 measured 0.162 ms against 0.238 at L=59, 0.80 against
 * 1.53 at L=128 and 5.25 against 11.4 at L=256; 32x64 and 64x32 tie with it and
 * 64x64 falls off a cliff at 0.725. See tools/gpu/bench-triangle-project.js,
 * whose arms take "projection@contraction" for exactly this reason.
 */
const CONTRACT_TILE_DEFAULT = { rows: 32, columns: 32 };

/**
 * The width these kernels fold a linear workgroup count at.
 *
 * 🔴 EXPORTED SO THE CALLERS CANNOT DRIFT FROM THE KERNEL. Three files dispatch
 * these shaders - src/af3/pair-track-gpu.js, src/multimer/block.js and
 * src/triangle/webgpu.js - and each had its own 32768 written out. The kernels
 * read `group.y + group.z * LINEAR_GRID_WIDTH`, so a caller splitting at a
 * different width addresses rows that do not exist and skips rows that do, with
 * nothing to report it: the dispatch is valid and the answer is wrong.
 */
export const LINEAR_GRID_WIDTH = 32_768;

export function createTriangleShaders(
  shape,
  precision,
  offsets,
  epsilon = 1e-5,
  direction = "outgoing",
  variance = "two-pass",
  projectTile = PROJECT_TILE,
  residual = false,
  contractTile = CONTRACT_TILE_DEFAULT,
  storage = {},
) {
  // 🔴 THE ACTIVATION STORAGE IS A THIRD AXIS, beside the weight element `tw`
  // and the arithmetic `precision`/`accumulatePrecision`, and it composes with
  // both without either knowing. Every field defaults to f32, so AF2's
  // evoformer and multimer blocks - which share these shaders and pass none of
  // this - generate exactly the WGSL they generated before.
  //
  // `normalized` is the layer-normed pair (the block's scratch[0]); `hidden`
  // is the normalised contraction that projectOutput reads (scratch[4]).
  const {
    normalized: normalizedStorage = "f32",
    hidden: hiddenStorage = "f32",
    ab: abStorage = "f32",
  } = typeof storage === "string" ? { normalized: storage } : storage;
  const packAB = abStorage === "f16";
  if (variance !== "two-pass" && variance !== "fast") {
    throw new Error(`variance must be "two-pass" or "fast", not ${variance}`);
  }
  // 🔴 THE WEIGHTS AND THE ACTIVATIONS ARE TWO DIFFERENT FORMATS, AND CONFLATING
  // THEM READS f32 DATA AS f16. `precision` used to name both: at "f16" the
  // normalize shader declared `source` - the PAIR REPRESENTATION - as an f16
  // array too, which is right for the standalone runner (it converts z on the
  // way in) and wrong for anything sharing that buffer with the rest of a
  // track. AF3's pair track hands it an f32 pair, so every value came back as
  // two halves of one float and the trunk produced NaN.
  //
  // `shape.weightPrecision` narrows the WEIGHT buffer alone, which is what buys
  // the memory: w.tri.out and w.tri.in are 40 MiB of an AF3 fold's residency
  // between them. It defaults to `precision`, so the standalone runner and its
  // checkers are unchanged.
  const weightPrecision = shape.weightPrecision ?? precision;
  // 🔴 AND THE ACCUMULATORS ARE A THIRD FORMAT, FOR A THIRD REASON. The
  // projection holds `rowsPerThread * columnsPerThread` vec4 in a WGSL ARRAY -
  // eight of them at the default tile, which is 32 registers, and an array is
  // the thing a driver is most willing to spill. In f16 they are 16, which is
  // what lets a wider tile fit; this is the same finding as
  // src/evoformer/attention.js's projection, where the 32x32 tile spills in
  // f32 and is the fastest arm there is in f16. The staged operands narrow with
  // them, since they feed the same multiply-add.
  //
  // The bias, the layer norm, the gate and the store all stay f32.
  const accumulatePrecision = shape.accumulatePrecision ?? "f32";
  if (!["f32", "f16"].includes(accumulatePrecision)) {
    throw new RangeError(`unknown triangle accumulate precision ${accumulatePrecision}`);
  }
  const acc16 = accumulatePrecision === "f16";
  const accVector = acc16 ? "vec4<f16>" : "vec4<f32>";
  const accNarrow = (e) => (acc16 ? `f16(${e})` : e);
  const common = prelude(shape, precision, offsets, epsilon, weightPrecision);
  const t = scalar(precision);
  const tw = scalar(weightPrecision);
  const readWeight = (expression) => read(weightPrecision, expression);

  // 🔴 A ROW A THREAD IS THE WRONG SHAPE FOR A LAYER NORM, and it cost half the
  // memory bandwidth of both normalisation passes. A thread walking its own row
  // reads addresses CZ * 4 = 512 bytes apart from its neighbours, so every lane
  // pulls a separate cache line and uses four bytes of it: measured at 49-56
  // GB/s on a part that does 100. The rows are staged through workgroup memory
  // instead - loaded and written back by a flat index, which makes consecutive
  // lanes consecutive addresses - and the reduction runs over the staged copy,
  // which also drops the second and third passes over global memory that the
  // mean, the variance and the write used to take.
  //
  // NORMALIZE_ROWS rows a workgroup of 64, so eight lanes share a row's
  // reduction and the staging tile is NORMALIZE_ROWS * channels floats.
  const NORMALIZE_ROWS = 8;
  const LANES_PER_ROW = 64 / NORMALIZE_ROWS;

  /**
   * The staged LayerNorm, in the two layouts it is needed in.
   *
   * @param {string} count "CZ" or "CH", the channel axis
   * @param {(row: string, channel: string) => string} load where a row's
   *   channel lives in the source, which is channel-minor for the input and
   *   channel-MAJOR for the hidden pass - see the note on the contraction.
   * @param {string} scale weight offset name for the scale
   * @param {string} offset weight offset name for the bias
   */
  // 🔴 THE STORE MAY BE PACKED, AND THE WRITE LOOP IS WHY IT CAN BE. A
  // workgroup owns a tile of whole rows, so both channels of a packed word are
  // produced by the same dispatch; walking WORDS rather than channels puts
  // them in the same INVOCATION, which is what makes it a store and not a
  // read-modify-write race. See src/runtime/storage.js.
  const stagedLayerNorm = (count, load, scale, offset, sourceMajor = "row",
                           outputStorage = "f32") => `
var<workgroup> tile: array<f32, ${NORMALIZE_ROWS} * ${shape[count === "CZ" ? "cZ" : "cHidden"]}>;
var<workgroup> partial_sum: array<f32, 64>;
var<workgroup> partial_squares: array<f32, 64>;
var<workgroup> row_mean: array<f32, ${NORMALIZE_ROWS}>;
var<workgroup> row_inverse_std: array<f32, ${NORMALIZE_ROWS}>;

const NORMALIZE_ROWS: u32 = ${NORMALIZE_ROWS}u;
const LANES_PER_ROW: u32 = ${LANES_PER_ROW}u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_row = (group.x + group.y * LINEAR_GRID_WIDTH) * NORMALIZE_ROWS;
  if (base_row >= PAIRS) { return; }
  let local = local_id.x;

  // 🔴 THE TAIL IS ZEROED, NOT SKIPPED. PAIRS is rarely a multiple of the tile
  // and the reduction below runs over the whole staged block.
  // 🔴 THE LOAD WALKS THE SOURCE'S OWN MAJOR AXIS. The input pass reads a
  // channel-MINOR tensor, so a flat index makes consecutive lanes consecutive
  // channels of one row; the hidden pass reads the contraction's channel-MAJOR
  // output, where consecutive lanes have to be consecutive ROWS or every lane
  // pulls its own cache line again. Either way the tile lands as
  // [slot][channel] and everything below is the same.
  for (var index = local; index < NORMALIZE_ROWS * ${count}; index += 64u) {
    ${sourceMajor === "channel"
      ? `let slot_of = index % NORMALIZE_ROWS;
    let channel = index / NORMALIZE_ROWS;`
      : `let slot_of = index / ${count};
    let channel = index % ${count};`}
    let row = base_row + slot_of;
    tile[slot_of * ${count} + channel] = select(0.0, ${load("row", "channel")}, row < PAIRS);
  }
  workgroupBarrier();

  let slot = local / LANES_PER_ROW;
  let lane = local % LANES_PER_ROW;
  var total = 0.0;
  var squares = 0.0;
  for (var c = lane; c < ${count}; c += LANES_PER_ROW) {
    let value = tile[slot * ${count} + c];
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
    let mean = row_total / f32(${count});
    ${variance === "fast"
      ? `let variance = row_squares / f32(${count}) - mean * mean;`
      : `var variance = 0.0;
    for (var c = 0u; c < ${count}; c += 1u) {
      let centered = tile[slot * ${count} + c] - mean;
      variance += centered * centered;
    }
    variance /= f32(${count});`}
    row_mean[slot] = mean;
    row_inverse_std[slot] = inverseSqrt(variance + EPSILON);
  }
  workgroupBarrier();

  // A lane owns a PAIR of channels, so it owns the whole word they share. The
  // channel count is even in every caller; the pair loop is the assertion.
  const PAIR_COUNT: u32 = ${count} / 2u;
  for (var word = local; word < NORMALIZE_ROWS * PAIR_COUNT; word += 64u) {
    let slot_of = word / PAIR_COUNT;
    let row = base_row + slot_of;
    if (row >= PAIRS) { continue; }
    let channel = (word % PAIR_COUNT) * 2u;
    let index = slot_of * ${count} + channel;
    let scaled = row_inverse_std[slot_of];
    let centre = row_mean[slot_of];
    let low = (tile[index] - centre) * scaled
      * ${readWeight(`weights[W_${scale} + channel]`)}
      + ${readWeight(`weights[W_${offset} + channel]`)};
    let high = (tile[index + 1u] - centre) * scaled
      * ${readWeight(`weights[W_${scale} + channel + 1u]`)}
      + ${readWeight(`weights[W_${offset} + channel + 1u]`)};
    let pair_word = row * PAIR_COUNT + (word % PAIR_COUNT);
    ${storedPair(outputStorage, "normalized", "pair_word", "low", "high")}
  }
}`;

  const normalizeInput = `${common}
@group(0) @binding(0) var<storage, read> source: array<${t}>;
@group(0) @binding(1) var<storage, read> weights: array<${tw}>;
@group(0) @binding(2) var<storage, read_write> normalized: array<${storageArray(normalizedStorage)}>;
${stagedLayerNorm("CZ", (row, channel) => read(precision, `source[${row} * CZ + ${channel}]`),
                  "LAYERNORMINWEIGHT", "LAYERNORMINBIAS", "row", normalizedStorage)}`;

  // Where the register block is spent. Each invocation owns ROWS_PER_THREAD
  // pair rows by COLUMNS_PER_THREAD hidden channels, and the workgroup is
  // always 8x8, so the tile a workgroup covers is eight times each.
  //
  // 🔴 THIS IS THE ARITHMETIC INTENSITY OF THE WHOLE PROJECTION AND IT IS A
  // RATIO OF READS TO MULTIPLY-ADDS, not of bytes to flops. Per step of k an
  // invocation reads ROWS_PER_THREAD values of the source tile and
  // 4 * COLUMNS_PER_THREAD weights, and does 4 * ROWS * COLUMNS multiply-adds:
  // at 2x2 that is ten workgroup reads buying sixteen, and at 4x2 twelve buying
  // thirty-two. Growing it costs registers - four accumulators per (row,
  // column) because a, b and their two gates share one source tile - so this is
  // the knob that trades occupancy for issue slots and it is measured, not
  // assumed. See tools/gpu/bench-triangle-project.js.
  // ...resolved once and REPORTED, not restated by each caller. See
  // LINEAR_GRID_WIDTH: three files dispatch these kernels.
  const projectGridWidth = shape.projectGridWidth ?? LINEAR_GRID_WIDTH;
  const { rows: PROJECT_TILE_ROWS, columns: PROJECT_TILE_COLUMNS } = projectTile;
  if (PROJECT_TILE_ROWS % 8 !== 0 || PROJECT_TILE_COLUMNS % 8 !== 0) {
    throw new Error(`projectTile ${PROJECT_TILE_ROWS}x${PROJECT_TILE_COLUMNS} is not a `
      + "multiple of the 8x8 workgroup");
  }
  const rowsPerThread = PROJECT_TILE_ROWS / 8;
  const columnsPerThread = PROJECT_TILE_COLUMNS / 8;
  // 🔴 THE STAGED ROWS ARE ONE VECTOR, WHICH HALVES THIS KERNEL'S WORKGROUP
  // READS. Every step of k reads the source tile once per row an invocation
  // owns, and those rows are adjacent slots, so four of them are one vec4 read
  // and three swizzles - and a swizzle is free where a workgroup read is not.
  // tools/gpu/probe-alu.js puts workgroup reads at 394 billion a second against
  // 580 billion vec4 multiply-adds, so for these kernels the reads are the
  // larger of the two terms.
  const rowVector = rowsPerThread === 1 ? "f32" : `vec${rowsPerThread}<f32>`;
  // 🔴 ONLY THE PROJECTION'S STAGED SOURCE NARROWS, not the output kernel's.
  // They share this shape but not their accumulators - the projection holds
  // four matrices in a vec4 and the output kernel two in a vec2 - and WGSL will
  // not mix widths in one multiply-add, so a single type here would have made
  // the output kernel assign an f32 into an f16 and fail to compile. It did.
  const outVector2 = acc16 ? "vec2<f16>" : "vec2<f32>";
  const outRowVector = acc16
    ? (rowsPerThread === 1 ? "f16" : `vec${rowsPerThread}<f16>`) : rowVector;
  const projectRowVector = acc16
    ? (rowsPerThread === 1 ? "f16" : `vec${rowsPerThread}<f16>`) : rowVector;
  if (![1, 2, 4].includes(rowsPerThread)) {
    throw new Error(`projectTile rows ${PROJECT_TILE_ROWS} gives ${rowsPerThread} rows an `
      + "invocation, which is not 1, 2 or 4");
  }
  const rowAt = (name, r) => rowsPerThread === 1 ? name : `${name}.${"xyzw"[r]}`;
  const overRows = (body) =>
    Array.from({ length: rowsPerThread }, (_, r) => body(r)).join("\n      ");
  const MATRICES = [["ap", "LINEARAPWEIGHT", "LINEARAPBIAS"], ["ag", "LINEARAGWEIGHT", "LINEARAGBIAS"],
                    ["bp", "LINEARBPWEIGHT", "LINEARBPBIAS"], ["bg", "LINEARBGWEIGHT", "LINEARBGBIAS"]];

  // 🔴 A THREAD TAKES ADJACENT CHANNELS WHEN a AND b ARE PACKED, not channels
  // eight apart. They are stored channel-major - a[h][i][j] - so two adjacent
  // LINEAR indices are two adjacent ROWS, and pairing rows would be wrong for
  // an odd PAIRS: `h * PAIRS + row` is odd at odd h when n is odd, so the pair
  // would straddle a word. n = 59 and n = 68 are the sizes every check here
  // runs at, one of each, and only one of them would have shown it.
  //
  // Pairing CHANNELS instead has no parity to get wrong: the word holding
  // channels h and h+1 of a row is `(h / 2) * PAIRS + row`, whatever PAIRS is.
  // A thread already owns `columnsPerThread` channels, so owning them adjacent
  // costs no extra accumulator - unlike the grid projection, where a lane owned
  // ONE output and had to take two.
  if (packAB && columnsPerThread % 2 !== 0) {
    throw new RangeError("a packed a/b projection needs an even columnsPerThread");
  }
  // 🔴 THIS BELONGS TO projectAB ALONE. `projectOutput` has a column loop of
  // the same shape reading the same-shaped weight tile, and a substitution that
  // matched both left it indexing the tile by the packed mapping while its
  // store still used the plain one. `a` was right, the contraction was right,
  // and the fold came out at relRMS 1.42 - the two kernels this touches were
  // both innocent. tools/gpu/check-triangle-packed.js is what localised it, by
  // running the whole five-kernel update and not only the two.
  const columnOf = packAB
    ? `local.x * ${columnsPerThread}u + column` : `local.x + column * 8u`;
  const plainStore = `    for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
      let h = h0 + ${columnOf};
      if (h >= CH) { continue; }
      let cell = vec4<f32>(acc[r * ${columnsPerThread}u + column]);
      let index = h * PAIRS + row;
      a[index] = pair_mask * cell.x * logistic(cell.y);
      b[index] = pair_mask * cell.z * logistic(cell.w);
    }`;
  const gated = (cell, value, gate) => `pair_mask * ${cell}.${value} * logistic(${cell}.${gate})`;
  const abStore = Array.from({ length: columnsPerThread / 2 }, (_, m) => `    {
      let h = h0 + local.x * ${columnsPerThread}u + ${m * 2}u;
      if (h + 1u < CH) {
        let lo = vec4<f32>(acc[r * ${columnsPerThread}u + ${m * 2}u]);
        let hi = vec4<f32>(acc[r * ${columnsPerThread}u + ${m * 2 + 1}u]);
        // The word holding channels h and h+1 of this row; this thread owns both.
        let word = (h / 2u) * PAIRS + row;
        a[word] = pack2x16float(vec2<f32>(${gated("lo", "x", "y")}, ${gated("hi", "x", "y")}));
        b[word] = pack2x16float(vec2<f32>(${gated("lo", "z", "w")}, ${gated("hi", "z", "w")}));
      }
    }`).join("\n");

  const projectAB = `${common}
const TILE_ROWS: u32 = ${PROJECT_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${PROJECT_TILE_COLUMNS}u;

@group(0) @binding(0) var<storage, read> z: array<${storageArray(normalizedStorage)}>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${tw}>;
@group(0) @binding(3) var<storage, read_write> a: array<${storageArray(abStorage)}>;
@group(0) @binding(4) var<storage, read_write> b: array<${storageArray(abStorage)}>;

var<workgroup> tile_source: array<${projectRowVector}, 64>;
// 🔴 ONE vec4 A CELL, NOT FOUR ARRAYS. a, b and their two gates are four
// separate matrices contracted over the same source, so the four weights a
// (k, channel) cell needs are always wanted together. Packed as a vec4 the
// inner loop reads them in ONE instruction and accumulates them in one
// multiply-add instead of four - the same arithmetic, a quarter of the issue
// slots. Priced before it was written: replacing these reads with a constant
// took the kernel from 0.525 ms to 0.375, so they were 29% of it.
var<workgroup> tile_weight: array<${accVector}, ${columnsPerThread * 64}>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  // 🔴 THE ROW TILE IS SPREAD OVER y AND z, because a dispatch is at most
  // 65535 workgroups in ANY dimension and there are n^2 pair rows. At 1566
  // residues that is 2.45M rows, 76637 tiles of 32, and the browser refuses the
  // pass: "Dispatch workgroup count Y (76637) exceeds max compute workgroups
  // per dimension (65535)". Every other pass in the track already folded
  // through a linear grid; these two divided by the tile and dispatched the
  // quotient raw.
  //
  // With a single z slice this is exactly group.y, so a caller that has not
  // been taught to fold is unchanged.
  let row0 = (group.y + group.z * PROJECT_GRID_WIDTH) * TILE_ROWS + local.y;
  let h0 = group.x * TILE_COLUMNS;
  let tile_index = local.y * 8u + local.x;
  // Each cell accumulates (a, a's gate, b, b's gate).
  var acc: array<${accVector}, ${rowsPerThread * columnsPerThread}>;
  for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
    let h = h0 + ${columnOf};
    var bias = ${accVector}(0.0);
    if (h < CH) {
      bias = ${accVector}(
        ${MATRICES.map(([, , name]) => accNarrow(readWeight(`weights[W_${name} + h]`))).join(",\n        ")});
    }
    for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
      acc[r * ${columnsPerThread}u + column] = bias;
    }
  }
  for (var c0 = 0u; c0 < CZ; c0 += 8u) {
    let source_c = c0 + local.x;
    let weight_c = c0 + local.y;
    var staged: ${projectRowVector};
    ${overRows((r) => `{
        let row = row0 + ${r}u * 8u;
        var value = 0.0;
        if (row < PAIRS && source_c < CZ) { value = ${storedElement(normalizedStorage, "z", "row * CZ + source_c")}; }
        ${rowAt("staged", r)} = ${accNarrow("value")};
      }`)}
    tile_source[tile_index] = staged;
    for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
      let h = h0 + ${columnOf};
      let slot = local.y * TILE_COLUMNS + ${columnOf};
      var packed = ${accVector}(0.0);
      if (h < CH && weight_c < CZ) {
        let weight_index = h * CZ + weight_c;
        packed = ${accVector}(
          ${MATRICES.map(([, name]) =>
            accNarrow(readWeight(`weights[W_${name} + weight_index]`))).join(",\n          ")});
      }
      tile_weight[slot] = packed;
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      let x = tile_source[local.y * 8u + k];
      for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
        let packed = tile_weight[k * TILE_COLUMNS + ${columnOf}];
        ${overRows((r) =>
          `acc[${r}u * ${columnsPerThread}u + column] += ${rowAt("x", r)} * packed;`)}
      }
    }
    workgroupBarrier();
  }
  // ...AND a AND b COME OUT CHANNEL-MAJOR, a[h][i][j], not a[i][j][h]. The
  // contraction below walks one h at a time, so this is the layout that decides
  // whether it runs or crawls - see the note on its k loop. Writing it here
  // costs this pass a scattered store; it saved the contraction 14x at L=256.
  //
  // 🔴 STAGING THE TILE THROUGH WORKGROUP MEMORY TO COALESCE THIS WAS TRIED AND
  // WAS SLOWER. The store does put PAIRS floats between the lanes of a
  // subgroup, so parking the tile in workgroup memory and writing it back
  // indexed by row looks like the obvious fix. Measured interleaved against
  // this version, bitwise-identical output, it lost at every length:
  //   L=64 0.92x   L=128 0.88x   L=192 0.95x   L=256 0.93x   (and 0.74x
  // incoming at 64). Two more arrays of 256 floats cost occupancy, and the
  // extra barrier and indexed writeback cost more than the coalescing bought.
  for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
    let row = row0 + r * 8u;
    if (row >= PAIRS) { continue; }
    let pair_mask = mask[row];
${packAB ? abStore : plainStore}
  }
}`;

  // 🔴 h IS THE SLOWEST INDEX IN a AND b, AND THAT IS WORTH 14x. Channel-minor
  // - a[(i*L + k)*CH + h] - reads 64 addresses CH*4 = 512 bytes apart per
  // workgroup, so every lane pulls its own cache line and uses four bytes of
  // it. A 128-byte line holds 32 consecutive h for one (i,k), so the lines one
  // h-slice touches span L*L lines of the whole buffer and the next 31 slices
  // touch them again: 2*L*L*128 bytes of footprint, 4 MiB at L=128 but 16 MiB
  // at L=256. Past the last-level cache that 32x reuse becomes 32x DRAM
  // re-fetches, and the kernel fell off a cliff exactly there - 171 ms at
  // L=256 against 28 ms at L=192, 5.8x the time for 2.4x the work.
  // Channel-major makes a slice 256 KiB and the lanes contiguous. Measured on
  // an M2, interleaved against the old layout, bitwise-identical output:
  //   L=128 4.7x   L=192 5.5x   L=224 9.1x   L=256 14.5x   L=288 13.9x
  // The same channel-paired layout projectAB writes; `h` is `group.z` here, so
  // which half of the word to take is uniform across the whole workgroup.
  const channelMajor = (name, row) => (packAB
    ? `unpack2x16float(${name}[(h / 2u) * PAIRS + ${row}])[h & 1u]`
    : `${name}[h * PAIRS + ${row}]`);
  const loadATile = direction === "outgoing"
    ? channelMajor("a", "i * L + a_k") : channelMajor("b", "a_k * L + i");
  const loadBTile = direction === "outgoing"
    ? channelMajor("b", "j * L + b_k") : channelMajor("a", "b_k * L + j");
  const CONTRACT_TILE = contractTile;
  // 🔴 THE CONTRACTION HAD THE WORST READ-TO-ARITHMETIC RATIO IN THE FILE: one
  // output a thread meant two workgroup reads bought a single multiply-add, and
  // it ran at 244 GFLOP/s where the projections beside it reach a thousand.
  // A register block of CONTRACT_TILE / 8 each way buys ROWS * COLUMNS of them
  // from ROWS + COLUMNS reads. There is occupancy to spend: at 59 tokens and
  // 128 channels the old shape launched 8,192 workgroups.
  //
  // 🔴 AND BOTH STAGED TILES ARE VECTORS, WHICH TAKES IT FURTHER. The rows an
  // invocation owns are adjacent slots of a, and the columns are adjacent slots
  // of b, so a 4x4 block reads ONE vec4 from each and does four vector
  // multiply-adds - two reads and four instructions where the scalar form had
  // eight reads and sixteen.
  const contractRows = CONTRACT_TILE.rows / 8;
  const contractColumns = CONTRACT_TILE.columns / 8;
  const contractRowVector = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[contractRows];
  const contractColumnVector = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[contractColumns];
  if (contractRowVector === undefined || contractColumnVector === undefined) {
    throw new Error(`contractTile ${CONTRACT_TILE.rows}x${CONTRACT_TILE.columns} gives a block `
      + "that is not 1, 2 or 4 each way");
  }
  const contractRowAt = (name, r) =>
    contractRows === 1 ? name : `${name}.${"xyzw"[r]}`;
  const contractColumnAt = (name, c) =>
    contractColumns === 1 ? name : `${name}.${"xyzw"[c]}`;
  const overContractRows = (body) =>
    Array.from({ length: contractRows }, (_, r) => body(r)).join("\n      ");
  const overContractColumns = (body) =>
    Array.from({ length: contractColumns }, (_, c) => body(c)).join("\n      ");
  const contract = `${common}
const CONTRACT_ROWS: u32 = ${CONTRACT_TILE.rows}u;
const CONTRACT_COLUMNS: u32 = ${CONTRACT_TILE.columns}u;

@group(0) @binding(0) var<storage, read> a: array<${storageArray(abStorage)}>;
@group(0) @binding(1) var<storage, read> b: array<${storageArray(abStorage)}>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// a's rows an invocation owns, and b's columns, each packed into one vector.
var<workgroup> tile_a: array<${contractRowVector}, 64>;
var<workgroup> tile_b: array<${contractColumnVector}, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let i0 = group.y * CONTRACT_ROWS + local.y;
  let j0 = group.x * CONTRACT_COLUMNS + local.x;
  let h = group.z;
  let tile_index = local.y * 8u + local.x;
  // One accumulator a row, holding that row's columns.
  var sum: array<${contractColumnVector}, ${contractRows}>;
  for (var r = 0u; r < ${contractRows}u; r += 1u) { sum[r] = ${contractColumnVector}(0.0); }

  for (var k0 = 0u; k0 < L; k0 += 8u) {
    let a_k = k0 + local.x;
    let b_k = k0 + local.y;
    var staged_a: ${contractRowVector};
    ${overContractRows((r) => `{
        let i = i0 + ${r}u * 8u;
        var value = 0.0;
        if (i < L && a_k < L) { value = ${loadATile}; }
        ${contractRowAt("staged_a", r)} = value;
      }`)}
    var staged_b: ${contractColumnVector};
    ${overContractColumns((c) => `{
        let j = j0 + ${c}u * 8u;
        var value = 0.0;
        if (j < L && b_k < L) { value = ${loadBTile}; }
        ${contractColumnAt("staged_b", c)} = value;
      }`)}
    tile_a[tile_index] = staged_a;
    tile_b[tile_index] = staged_b;
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      let left = tile_a[local.y * 8u + k];
      let right = tile_b[k * 8u + local.x];
      ${overContractRows((r) => `sum[${r}u] += ${contractRowAt("left", r)} * right;`)}
    }
    workgroupBarrier();
  }

  for (var r = 0u; r < ${contractRows}u; r += 1u) {
    let i = i0 + r * 8u;
    if (i >= L) { continue; }
    let row = sum[r];
    ${overContractColumns((c) => `{
        let j = j0 + ${c}u * 8u;
        if (j < L) { output[h * PAIRS + i * L + j] = ${contractColumnAt("row", c)}; }
      }`)}
  }
}`;

  const normalizeHidden = `${common}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${tw}>;
@group(0) @binding(2) var<storage, read_write> normalized: array<${storageArray(hiddenStorage)}>;
// ...READS CHANNEL-MAJOR AND WRITES CHANNEL-MINOR, because this is where the
// two layouts meet: the contraction upstream wants h slowest, and
// projectOutput downstream reads a row's channels contiguously. Doing the
// transpose in a pass that already touches every element once is free next to
// a dedicated one, and staging it through workgroup memory is what lets BOTH
// sides be read and written along their own major axis.
${stagedLayerNorm("CH", (row, channel) => `source[${channel} * PAIRS + ${row}]`,
                  "LAYERNORMOUTWEIGHT", "LAYERNORMOUTBIAS", "channel", hiddenStorage)}`;

  const projectOutput = `${common}
const TILE_ROWS: u32 = ${PROJECT_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${PROJECT_TILE_COLUMNS}u;

@group(0) @binding(0) var<storage, read> z: array<${storageArray(normalizedStorage)}>;
@group(0) @binding(1) var<storage, read> x: array<${storageArray(hiddenStorage)}>;
@group(0) @binding(2) var<storage, read> weights: array<${tw}>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> tile_x: array<${outRowVector}, 64>;
var<workgroup> tile_z: array<${outRowVector}, 64>;
// 🔴 ONE vec2 A CELL, NOT TWO SCALAR ARRAYS - the same argument projectAB
// makes with four matrices and a vec4. The output projection and its gate are
// contracted at the SAME (k, output channel) cell, so their two weights are
// always wanted together: packed, the inner loop reads them in one instruction
// and accumulates them in one multiply-add instead of two. Scalar, this kernel
// spent two source reads, four weight reads and sixteen multiply-adds a step of
// k to buy sixteen products - 0.73 useful operations an instruction, against
// projectAB's 2.9 on the same shape, and it showed: 672 GFLOP/s where the
// projection that feeds it runs at 977.
var<workgroup> tile_weight: array<${outVector2}, ${columnsPerThread * 64}>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  // 🔴 THE ROW TILE IS SPREAD OVER y AND z, because a dispatch is at most
  // 65535 workgroups in ANY dimension and there are n^2 pair rows. At 1566
  // residues that is 2.45M rows, 76637 tiles of 32, and the browser refuses the
  // pass: "Dispatch workgroup count Y (76637) exceeds max compute workgroups
  // per dimension (65535)". Every other pass in the track already folded
  // through a linear grid; these two divided by the tile and dispatched the
  // quotient raw.
  //
  // With a single z slice this is exactly group.y, so a caller that has not
  // been taught to fold is unchanged.
  let row0 = (group.y + group.z * PROJECT_GRID_WIDTH) * TILE_ROWS + local.y;
  let channel0 = group.x * TILE_COLUMNS + local.x;
  let tile_index = local.y * 8u + local.x;
  // ...the projection contracts over CH and the gate over CZ, on the same
  // output channel, so one accumulator a cell carries both: x is the
  // projection, y the gate.
  var acc: array<${outVector2}, ${rowsPerThread * columnsPerThread}>;
  for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
    let out_channel = channel0 + column * 8u;
    var bias = ${outVector2}(0.0);
    if (out_channel < CZ) {
      bias = ${outVector2}(
        ${accNarrow(readWeight("weights[W_LINEARZBIAS + out_channel]"))},
        ${accNarrow(readWeight("weights[W_LINEARGBIAS + out_channel]"))});
    }
    for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
      acc[r * ${columnsPerThread}u + column] = bias;
    }
  }
  for (var k0 = 0u; k0 < max(CH, CZ); k0 += 8u) {
    let source_k = k0 + local.x;
    let weight_k = k0 + local.y;
    var staged_x: ${outRowVector};
    var staged_z: ${outRowVector};
    ${overRows((r) => `{
        let row = row0 + ${r}u * 8u;
        var x_value = 0.0;
        var z_value = 0.0;
        if (row < PAIRS && source_k < CH) { x_value = ${storedElement(hiddenStorage, "x", "row * CH + source_k")}; }
        if (row < PAIRS && source_k < CZ) { z_value = ${storedElement(normalizedStorage, "z", "row * CZ + source_k")}; }
        ${rowAt("staged_x", r)} = ${accNarrow("x_value")};
        ${rowAt("staged_z", r)} = ${accNarrow("z_value")};
      }`)}
    tile_x[tile_index] = staged_x;
    tile_z[tile_index] = staged_z;
    for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
      let out_channel = channel0 + column * 8u;
      let slot = local.y * TILE_COLUMNS + local.x + column * 8u;
      var projection_w = 0.0;
      var gate_w = 0.0;
      if (out_channel < CZ && weight_k < CH) {
        projection_w = ${readWeight("weights[W_LINEARZWEIGHT + out_channel * CH + weight_k]")};
      }
      if (out_channel < CZ && weight_k < CZ) {
        gate_w = ${readWeight("weights[W_LINEARGWEIGHT + out_channel * CZ + weight_k]")};
      }
      tile_weight[slot] = ${outVector2}(${accNarrow("projection_w")}, ${accNarrow("gate_w")});
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      let xs = tile_x[local.y * 8u + k];
      let zs = tile_z[local.y * 8u + k];
      // ...paired once a row, outside the column loop, because the pairing
      // depends on the row and the weight does not.
      ${overRows((r) => `let xz${r} = ${outVector2}(${rowAt("xs", r)}, ${rowAt("zs", r)});`)}
      for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
        let packed = tile_weight[k * TILE_COLUMNS + local.x + column * 8u];
        ${overRows((r) => `acc[${r}u * ${columnsPerThread}u + column] += xz${r} * packed;`)}
      }
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
    let row = row0 + r * 8u;
    if (row >= PAIRS) { continue; }
    for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
      let out_channel = channel0 + column * 8u;
      if (out_channel >= CZ) { continue; }
      let cell = vec2<f32>(acc[r * ${columnsPerThread}u + column]);
      output[row * CZ + out_channel] ${residual ? "+=" : "="} cell.x * logistic(cell.y);
    }
  }
}`;

  return { normalizeInput, projectAB, contract, normalizeHidden, projectOutput,
           projectTile: { ...projectTile }, contractTile: { ...contractTile },
           // ...the width the two projections fold their row tile at, so a
           // caller splits at the number the kernel was GENERATED with rather
           // than at a constant of its own that happens to match.
           projectGridWidth,
           normalizeRows: NORMALIZE_ROWS };
}
