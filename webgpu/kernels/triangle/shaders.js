import { storageArray, storedElement, storedPair } from "../../runtime/storage.js";
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
// 🔴 THE NORMALISED BUFFERS ARE STORED IN PAIRS, SO THEIR ROW STRIDE ROUNDS UP.
// A lane owns both channels of a word, which is what lets one shader serve the
// packed f16 store and the f32 one; an ODD channel count has a half-used last
// word, and the row stride is the rounded-up count and not the count. Reading
// them at row * CZ instead cost the OpenFold fixture - cZ 7 - its last
// channel AND aligned every row one channel early: 8.26e-2 against a 1e-5
// bound, on a kernel three AlphaFold fixtures at cZ 128 passed. Every shipped
// width is even, where these are equal and the shader is unchanged.
const CZ_STRIDE: u32 = ${2 * Math.ceil(shape.cZ / 2)}u;
const CH_STRIDE: u32 = ${2 * Math.ceil(shape.cHidden / 2)}u;
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
export const PROJECT_TILE = { rows: 32, columns: 16 };

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
 * these shaders - webgpu/af3/trunk/pair-track.js, webgpu/af2/multimer/block.js and
 * webgpu/kernels/triangle/webgpu.js - and each had its own 32768 written out. The kernels
 * read `group.y + group.z * LINEAR_GRID_WIDTH`, so a caller splitting at a
 * different width addresses rows that do not exist and skips rows that do, with
 * nothing to report it: the dispatch is valid and the answer is wrong.
 */
export const LINEAR_GRID_WIDTH = 32_768;

/**
 * The largest of a tile's candidates whose workgroup storage FITS this device.
 *
 * 🔴 EVERY ROW TILE HERE IS A CONSTANT THAT NEVER ASKED THE DEVICE, and at
 * AlphaFold 3's 128 channels none of them had to: `grid.normalize` stages
 * `rows * channels` floats, which is 4,640 bytes at 8 rows and fits anything.
 * IntelliFold-2's pair is **512 channels**, and the same tile is
 * `8 * 512 * 4 + 512 + 64` = **16,960 bytes against WebGPU's guaranteed 16,384**
 * - so if2 folds on this A100, folds at `PORTABLE_CEILINGS`, and cannot create
 * that pipeline on a device at the standard's minimum. The error names a shader
 * and a byte count and no limit, which is the hardest kind to read from another
 * machine.
 *
 * 🔴 AND IT IS THE FIFTH OF EXACTLY THIS BUG. `transitionWidth`,
 * `splitTransitionConfig`, `projectMatrixConfig` and two projection workgroups
 * were all performance choices that never priced the storage they stage; see
 * the spec-floor row in CLAUDE.md. The pattern is always the same - a constant
 * fitted at one model's width, correct until a wider model arrives - and it is
 * the same pattern docs/ARCHITECTURE.md names as the thing to design out.
 *
 * Halving the tile costs weight traffic and nothing else: the kernel is the
 * kernel, only fewer rows share a staging pass. On this device the limit is
 * 49152 and every model keeps the tile it had.
 */
export function tileThatFits(candidates, bytesFor, limitBytes) {
  for (const rows of candidates) if (bytesFor(rows) <= limitBytes) return rows;
  // 🔴 THE SMALLEST CANDIDATE, NOT A THROW. One row is the least this kernel
  // can stage, and a device that cannot hold one row of the pair has no
  // business here - the pipeline creation will say so, naming the shader,
  // which is more use than a message from this helper.
  return candidates[candidates.length - 1];
}

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
  // 🔴 RoseTTAFold3 DIVIDES THE CONTRACTION BY THE CHAIN LENGTH, AND IT IS
  // OBSERVABLE ONLY BECAUSE A LayerNorm's EPSILON DOES NOT COMMUTE WITH A
  // SCALE. Its triangle multiplication is
  // `out = einsum("bikd,bjkd->bijd", left, right / float(L))` followed by the
  // centre norm - so the mean and the variance are 1/L and 1/L^2 of ours while
  // EPSILON is not, and the normalised result differs by a factor that depends
  // on the row. Applied at THIS pass's input rather than at the contraction's
  // output because they are the same number and this pass already reads every
  // element once: the contraction is the most expensive kernel in the track and
  // does not need a multiply in its inner loop. Divided rather than multiplied
  // by a reciprocal, so it is the CPU reference's `total / n` to the last bit.
  const divideByLength = shape.triangleMulDivideByLength === true;
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
  // webgpu/kernels/attention.js's projection, where the 32x32 tile spills in
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
  // 🔴 PRICED AGAINST THE DEVICE, NOT TYPED IN - AND THIS IS THE SECOND COPY OF
  // THIS TILE. `grid-attention.js` has the same eight-row staged
  // LayerNorm and the same arithmetic; both stage `rows * channels` floats plus
  // two 64-lane reductions and two per-row scalars, and both come to EXACTLY
  // 16,960 bytes at IntelliFold-2's 512 channels against WebGPU's guaranteed
  // 16,384. Fixing one and not the other left the error byte-for-byte
  // identical, which reads as "the fix did not take". docs/ARCHITECTURE.md
  // lists this LayerNorm among the things written four times.
  const NORMALIZE_ROWS = tileThatFits([8, 4, 2, 1],
    (rows) => rows * Math.max(shape.cZ, shape.cHidden) * 4 + 64 * 4 * 2 + rows * 4 * 2,
    shape.maxComputeWorkgroupStorageSize ?? 49152);
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
  // read-modify-write race. See webgpu/runtime/storage.js.
  const countOf = (axis) => (axis === "CZ" ? shape.cZ : shape.cHidden);
  const fastVariance = variance === "fast";
  const stagedLayerNorm = (count, load, scale, offset, sourceMajor = "row",
                           outputStorage = "f32") => `
var<workgroup> tile: array<f32, ${NORMALIZE_ROWS} * ${shape[count === "CZ" ? "cZ" : "cHidden"]}>;
var<workgroup> partial_sum: array<f32, 64>;${fastVariance ? "\nvar<workgroup> partial_squares: array<f32, 64>;" : ""}
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
  var total = 0.0;${fastVariance ? "\n  var squares = 0.0;" : ""}
  for (var c = lane; c < ${count}; c += LANES_PER_ROW) {
    let value = tile[slot * ${count} + c];
    total += value;${fastVariance ? "\n    squares += value * value;" : ""}
  }
  partial_sum[local] = total;${fastVariance ? "\n  partial_squares[local] = squares;" : ""}
  workgroupBarrier();
  if (lane == 0u) {
    var row_total = 0.0;${fastVariance ? "\n    var row_squares = 0.0;" : ""}
    for (var l = 0u; l < LANES_PER_ROW; l += 1u) {
      row_total += partial_sum[slot * LANES_PER_ROW + l];${fastVariance ? "\n      row_squares += partial_squares[slot * LANES_PER_ROW + l];" : ""}
    }
    let mean = row_total / f32(${count});
    ${fastVariance
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

  // A lane owns a PAIR of channels, so it owns the whole word they share - and
  // an ODD count leaves the last word half used rather than dropping it. The
  // comment here used to say the count was even in every caller and call the
  // loop the assertion; a WGSL loop asserts nothing, and at cZ 7 this wrote
  // three words a row where the readers expected four.
  const PAIR_COUNT: u32 = ${Math.ceil(countOf(count) / 2)}u;
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
    // ...and where the count is ODD the last word's high half is past the end,
    // and contributes nothing - which is what an absent channel is worth. The
    // guard is emitted ONLY there, so every even width generates the shader it
    // generated before and pays no select in the normalize loop.
    let high = ${countOf(count) % 2 === 0 ? "" : "select(0.0, "}(tile[index + 1u] - centre) * scaled
      * ${readWeight(`weights[W_${scale} + channel + 1u]`)}
      + ${readWeight(`weights[W_${offset} + channel + 1u]`)}${
  countOf(count) % 2 === 0 ? "" : `, channel + 1u < ${count})`};
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
  const { rows: PROJECT_TILE_ROWS, columns: PROJECT_TILE_COLUMNS, workgroup: PROJECT_W = 8 } = projectTile;
  if (PROJECT_W !== 8 && PROJECT_W !== 16) {
    throw new Error(`projectTile workgroup ${PROJECT_W} is not 8 or 16`);
  }
  const PROJECT_W2 = PROJECT_W * PROJECT_W;
  if (PROJECT_TILE_ROWS % PROJECT_W !== 0 || PROJECT_TILE_COLUMNS % PROJECT_W !== 0) {
    throw new Error(`projectTile ${PROJECT_TILE_ROWS}x${PROJECT_TILE_COLUMNS} is not a `
      + `multiple of the ${PROJECT_W}x${PROJECT_W} workgroup`);
  }
  const rowsPerThread = PROJECT_TILE_ROWS / PROJECT_W;
  const columnsPerThread = PROJECT_TILE_COLUMNS / PROJECT_W;
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
  const columnOfCol = (c) => packAB
    ? `local.x * ${columnsPerThread}u + ${c}u` : `local.x + ${c * PROJECT_W}u`;
  const channelChunked = shape.channelChunked === true && !packAB;
  const H_END = channelChunked ? "min(CH, chunk.x + chunk.y)" : "CH";
  const gated = (cell, value, gate) => `pair_mask * ${cell}.${value} * logistic(${cell}.${gate})`;
  const exactCzSteps = shape.cZ % PROJECT_W === 0;
  const exactChSteps = shape.cHidden % PROJECT_W === 0;
  const exactHCols = !channelChunked && shape.cHidden % PROJECT_TILE_COLUMNS === 0;
  const abBiasDeclare = [];
  for (let c = 0; c < columnsPerThread; c += 1) {
    abBiasDeclare.push(`  let h_${c} = h0 + ${columnOfCol(c)};
  ${exactHCols ? "" : `let valid_h_${c} = h_${c} < ${H_END};\n  `}let w_slot_${c} = local.y * TILE_COLUMNS + ${columnOfCol(c)};
  let w_base_${c} = h_${c} * CZ + local.y;
  ${exactHCols
    ? `let bias_${c} = ${accVector}(
      ${MATRICES.map(([, , name]) => accNarrow(readWeight(`weights[W_${name} + h_${c}]`))).join(",\n      ")});`
    : `var bias_${c} = ${accVector}(0.0);
  if (valid_h_${c}) {
    bias_${c} = ${accVector}(
      ${MATRICES.map(([, , name]) => accNarrow(readWeight(`weights[W_${name} + h_${c}]`))).join(",\n      ")});
  }`}`);
    for (let r = 0; r < rowsPerThread; r += 1) {
      abBiasDeclare.push(`  var acc_${r}_${c} = bias_${c};`);
    }
  }
  const abStageWeight = Array.from({ length: columnsPerThread }, (_, c) => {
    const cond = [exactHCols ? null : `valid_h_${c}`, exactCzSteps ? null : "weight_c < CZ"].filter(Boolean).join(" && ");
    return cond === ""
      ? `    {
      let weight_index = w_base_${c} + c0;
      tile_weight[w_slot_${c}] = ${accVector}(
        ${MATRICES.map(([, name]) =>
          accNarrow(readWeight(`weights[W_${name} + weight_index]`))).join(",\n        ")});
    }`
      : `    {
      var packed = ${accVector}(0.0);
      if (${cond}) {
        let weight_index = w_base_${c} + c0;
        packed = ${accVector}(
          ${MATRICES.map(([, name]) =>
            accNarrow(readWeight(`weights[W_${name} + weight_index]`))).join(",\n          ")});
      }
      tile_weight[w_slot_${c}] = packed;
    }`;
  }).join("\n");
  const abWeightIndex = (k, c) => (packAB
    ? `${k * PROJECT_TILE_COLUMNS + c}u + local.x * ${columnsPerThread}u`
    : `${k * PROJECT_TILE_COLUMNS + c * PROJECT_W}u + local.x`);
  const abInner = Array.from({ length: PROJECT_W }, (_, k) => `    {
      let x = tile_source[local.y * ${PROJECT_W}u + ${k}u];
${Array.from({ length: columnsPerThread }, (_, c) => `      {
        let packed = tile_weight[${abWeightIndex(k, c)}];
        ${overRows((r) => `acc_${r}_${c} += ${rowAt("x", r)} * packed;`)}
      }`).join("\n")}
    }`).join("\n");
  const abRowStores = Array.from({ length: rowsPerThread }, (_, r) => {
    const body = packAB
      ? Array.from({ length: columnsPerThread / 2 }, (_, m) => `      {
        let h = h0 + local.x * ${columnsPerThread}u + ${m * 2}u;
        if (h + 1u < CH) {
          let lo = vec4<f32>(acc_${r}_${m * 2});
          let hi = vec4<f32>(acc_${r}_${m * 2 + 1});
          let word = (h / 2u) * PAIRS + row;
          a[word] = pack2x16float(vec2<f32>(${gated("lo", "x", "y")}, ${gated("hi", "x", "y")}));
          b[word] = pack2x16float(vec2<f32>(${gated("lo", "z", "w")}, ${gated("hi", "z", "w")}));
        }
      }`).join("\n")
      : Array.from({ length: columnsPerThread }, (_, c) => `      {
        let h = h0 + ${columnOfCol(c)};
        if (h < ${H_END}) {
          let cell = vec4<f32>(acc_${r}_${c});
          let index = ${channelChunked ? "(h - chunk.x)" : "h"} * PAIRS + row;
          a[index] = pair_mask * cell.x * logistic(cell.y);
          b[index] = pair_mask * cell.z * logistic(cell.w);
        }
      }`).join("\n");
    return `  {
    let row = row0 + ${r * PROJECT_W}u;
    if (row < PAIRS) {
      let pair_mask = mask[row];
${body}
    }
  }`;
  }).join("\n");

  const projectAB = `${common}
const TILE_ROWS: u32 = ${PROJECT_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${PROJECT_TILE_COLUMNS}u;

@group(0) @binding(0) var<storage, read> z: array<${storageArray(normalizedStorage)}>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${tw}>;
@group(0) @binding(3) var<storage, read_write> a: array<${storageArray(abStorage)}>;
@group(0) @binding(4) var<storage, read_write> b: array<${storageArray(abStorage)}>;
${channelChunked ? "@group(0) @binding(5) var<uniform> chunk: vec4<u32>;" : ""}

var<workgroup> tile_source: array<${projectRowVector}, ${PROJECT_W2}>;
// 🔴 ONE vec4 A CELL, NOT FOUR ARRAYS. a, b and their two gates are four
// separate matrices contracted over the same source, so the four weights a
// (k, channel) cell needs are always wanted together. Packed as a vec4 the
// inner loop reads them in ONE instruction and accumulates them in one
// multiply-add instead of four - the same arithmetic, a quarter of the issue
// slots. Priced before it was written: replacing these reads with a constant
// took the kernel from 0.525 ms to 0.375, so they were 29% of it.
var<workgroup> tile_weight: array<${accVector}, ${columnsPerThread * PROJECT_W2}>;

@compute @workgroup_size(${PROJECT_W}, ${PROJECT_W}, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row0 = (group.y + group.z * PROJECT_GRID_WIDTH) * TILE_ROWS + local.y;
  let h0 = ${channelChunked ? "chunk.x + " : ""}group.x * TILE_COLUMNS;
  let tile_index = local.y * ${PROJECT_W}u + local.x;
  let full_rows = row0 + ${(rowsPerThread - 1) * PROJECT_W}u < PAIRS;
  let s_base0 = row0 * CZ_STRIDE + local.x;
  // Each cell accumulates (a, a's gate, b, b's gate).
${abBiasDeclare.join("\n")}
  for (var c0 = 0u; c0 < CZ; c0 += ${PROJECT_W}u) {
    let source_c = c0 + local.x;
    ${exactCzSteps ? "" : "let weight_c = c0 + local.y;\n    "}var staged = ${projectRowVector}(0.0);
    ${exactCzSteps ? "" : "if (source_c < CZ) {\n      "}if (full_rows) {
        let s0 = s_base0 + c0;
        ${overRows((r) => `${rowAt("staged", r)} = ${accNarrow(storedElement(normalizedStorage, "z", `s0 + ${r * PROJECT_W}u * CZ_STRIDE`))};`)}
      } else {
        ${overRows((r) => `{
          let row = row0 + ${r * PROJECT_W}u;
          if (row < PAIRS) { ${rowAt("staged", r)} = ${accNarrow(storedElement(normalizedStorage, "z", "row * CZ_STRIDE + source_c"))}; }
        }`)}
      }${exactCzSteps ? "" : "\n    }"}
    tile_source[tile_index] = staged;
${abStageWeight}
    workgroupBarrier();
${abInner}
    workgroupBarrier();
  }
${abRowStores}
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
  if ((contractRowVector === undefined || contractColumnVector === undefined) && contractTile.arithmetic !== "f16") {
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
  const contractInner = Array.from({ length: 8 }, (_, k) => `    {
      let left = tile_a[local.y * 8u + ${k}u];
      let right = tile_b[${k * 8}u + local.x];
      ${overContractRows((r) => `sum_${r} += ${contractRowAt("left", r)} * right;`)}
    }`).join("\n");
  const contractStores = Array.from({ length: contractRows }, (_, r) => `  {
    let i = i0 + ${r * 8}u;
    if (i < L) {
      let row = sum_${r};
      ${overContractColumns((c) => `{
        let j = j0 + ${c}u * 8u;
        if (j < L) { output[${channelChunked ? "(chunk.x + h)" : "h"} * PAIRS + i * L + j] = ${contractColumnAt("row", c)}; }
      }`)}
    }
  }`).join("\n");
  const contract = `${common}
const CONTRACT_ROWS: u32 = ${CONTRACT_TILE.rows}u;
const CONTRACT_COLUMNS: u32 = ${CONTRACT_TILE.columns}u;

@group(0) @binding(0) var<storage, read> a: array<${storageArray(abStorage)}>;
@group(0) @binding(1) var<storage, read> b: array<${storageArray(abStorage)}>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
${channelChunked ? "@group(0) @binding(3) var<uniform> chunk: vec4<u32>;" : ""}

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
${Array.from({ length: contractRows }, (_, r) => `  var sum_${r} = ${contractColumnVector}(0.0);`).join("\n")}

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
${contractInner}
    workgroupBarrier();
  }

${contractStores}
}`;

  // 🔴 THE CONTRACTION IN f16 ARITHMETIC, ON A 64 x 64 TILE OF 16 x 16 LANES (contractTile.arithmetic "f16"), as the
  // pair transition's VECTOR_GEMM_F16 (webgpu/af3/trunk/transition.js): a and b staged in halves 16 k at a time, each
  // step's 16 products summed in f16 and then added into f32 accumulators - so no f16 partial sum runs past 16 terms.
  // A lane keeps a 4 x 4 block; each operand is staged along the axis it is contiguous on (k outgoing, the row
  // incoming), so every load is coalesced. On an M5 under stock flags at 522 tokens, 32 channels: 6.38 -> 4.77 ms
  // (1.34x), relRMS 6.6e-4 against the f32 kernel on uniform inputs (tools/gpu/probe-contract-f16.js); larger lane
  // blocks (8 x 4, 8 x 8) and a 32-deep step were slower or less exact. The same tile in f32 is level with the 8 x 8
  // kernel, so it is the arithmetic that pays, not the tile.
  // 🔴 AND EACH OPERAND IS SCALED BY 2^-4 ON ITS WAY TO HALF, the sum back by 2^8 - exact wherever a value stays in
  // half's normal range, and 256x the headroom: IntelliFold-2's products reach rms 1e5 and peaks of 2.5e6 a sum
  // (docs/M5-HANDOFF.md, the Metal port's half product), and unscaled its 16-term partial sums overflowed - its
  // template fold came back NaN (test:template). A value under 1e-3 goes subnormal, which a sum of ~0.1-sized terms
  // does not see.
  let contractF16;
  if (contractTile.arithmetic === "f16") {
    const kContiguous = direction === "outgoing";
    const lane = kContiguous ? "local.y" : "local.x";
    const kLane = kContiguous ? "local.x" : "local.y";
    const loadOf = (operand) => (row) => (operand === "a"
      ? (direction === "outgoing" ? channelMajor("a", `${row} * L + k`) : channelMajor("b", `k * L + ${row}`))
      : (direction === "outgoing" ? channelMajor("b", `${row} * L + k`) : channelMajor("a", `k * L + ${row}`)));
    const stage = (tile, load, row0) => `{
      let k = k0 + ${kLane};
      var staged = vec4<f16>(0.0);
      if (k < L) {
        let r0 = ${row0} + ${lane};
        let r1 = r0 + 16u;
        let r2 = r0 + 32u;
        let r3 = r0 + 48u;
        if (r3 < L) {
          staged = vec4<f16>(
            f16(${load("r0")} * 0.0625),
            f16(${load("r1")} * 0.0625),
            f16(${load("r2")} * 0.0625),
            f16(${load("r3")} * 0.0625),
          );
        } else {
          if (r0 < L) { staged.x = f16(${load("r0")} * 0.0625); }
          if (r1 < L) { staged.y = f16(${load("r1")} * 0.0625); }
          if (r2 < L) { staged.z = f16(${load("r2")} * 0.0625); }
        }
      }
      ${tile}[${kLane} * 16u + ${lane}] = staged;
    }`;
    const f16Inner = Array.from({ length: 16 }, (_, k) => `    {
      let left = tile_a[${k * 16}u + local.y];
      let right = tile_b[${k * 16}u + local.x];
      part_0 = fma(vec4<f16>(left.x), right, part_0);
      part_1 = fma(vec4<f16>(left.y), right, part_1);
      part_2 = fma(vec4<f16>(left.z), right, part_2);
      part_3 = fma(vec4<f16>(left.w), right, part_3);
    }`).join("\n");
    const f16Stores = [0, 1, 2, 3].map((r) => `  {
    let i = i0 + local.y + ${r * 16}u;
    if (i < L) {
      let row_base = ${channelChunked ? "(chunk.x + h)" : "h"} * PAIRS + i * L + j0 + local.x;
      if (j0 + local.x < L) { output[row_base] = sum_${r}.x * 256.0; }
      if (j0 + local.x + 16u < L) { output[row_base + 16u] = sum_${r}.y * 256.0; }
      if (j0 + local.x + 32u < L) { output[row_base + 32u] = sum_${r}.z * 256.0; }
      if (j0 + local.x + 48u < L) { output[row_base + 48u] = sum_${r}.w * 256.0; }
    }
  }`).join("\n");
    contractF16 = `${common.startsWith("enable f16;") ? "" : "enable f16;\n"}${common}
@group(0) @binding(0) var<storage, read> a: array<${storageArray(abStorage)}>;
@group(0) @binding(1) var<storage, read> b: array<${storageArray(abStorage)}>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
${channelChunked ? "@group(0) @binding(3) var<uniform> chunk: vec4<u32>;" : ""}

// [k][lane]: a's four rows a lane owns (lane + 16 r), and b's four columns
var<workgroup> tile_a: array<vec4<f16>, 256>;
var<workgroup> tile_b: array<vec4<f16>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let i0 = group.y * 64u;
  let j0 = group.x * 64u;
  let h = group.z;
  var sum_0 = vec4<f32>(0.0);
  var sum_1 = vec4<f32>(0.0);
  var sum_2 = vec4<f32>(0.0);
  var sum_3 = vec4<f32>(0.0);
  for (var k0 = 0u; k0 < L; k0 += 16u) {
    ${stage("tile_a", loadOf("a"), "i0")}
    ${stage("tile_b", loadOf("b"), "j0")}
    workgroupBarrier();
    var part_0 = vec4<f16>(0.0);
    var part_1 = vec4<f16>(0.0);
    var part_2 = vec4<f16>(0.0);
    var part_3 = vec4<f16>(0.0);
${f16Inner}
    sum_0 += vec4<f32>(part_0);
    sum_1 += vec4<f32>(part_1);
    sum_2 += vec4<f32>(part_2);
    sum_3 += vec4<f32>(part_3);
    workgroupBarrier();
  }
${f16Stores}
}`;
  }

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
${stagedLayerNorm("CH", (row, channel) => divideByLength
                    ? `(source[${channel} * PAIRS + ${row}] / f32(L))`
                    : `source[${channel} * PAIRS + ${row}]`,
                  "LAYERNORMOUTWEIGHT", "LAYERNORMOUTBIAS", "channel", hiddenStorage)}`;

  // 🔴 THE OUTPUT PROJECTION MAY TAKE A WIDER TILE THAN THE INPUT ONE, because
  // its accumulator is a vec2 where projectAB's is a vec4: at the same tile it
  // holds half the registers and does half the multiply-adds a staged read.
  // `outColumns` on the tile says so; absent, it is the input tile's. Measured
  // on an A100 under stock flags at 255 tokens: 32 x 64 against 32 x 32 is
  // 4.34 -> 3.26 ms at 384 channels and 0.56 -> 0.44 at 128, bit-identical,
  // where projectAB is slower at 32 x 64 (5.43 -> 6.10).
  const OUT_TILE_COLUMNS = projectTile.outColumns ?? PROJECT_TILE_COLUMNS;
  if (OUT_TILE_COLUMNS % PROJECT_W !== 0) {
    throw new Error(`projectTile outColumns ${OUT_TILE_COLUMNS} is not a multiple of ${PROJECT_W}`);
  }
  const outColumnsPerThread = OUT_TILE_COLUMNS / PROJECT_W;
  const exactOutCols = shape.cZ % OUT_TILE_COLUMNS === 0;
  const exactOutSteps = shape.cZ === shape.cHidden && exactCzSteps;
  const outBiasDeclare = [];
  for (let c = 0; c < outColumnsPerThread; c += 1) {
    outBiasDeclare.push(`  let out_col_${c} = channel0 + ${c * PROJECT_W}u;
  ${exactOutCols ? "" : `let valid_col_${c} = out_col_${c} < CZ;\n  `}let w_slot_${c} = local.y * TILE_COLUMNS + local.x + ${c * PROJECT_W}u;
  let wz_base_${c} = W_LINEARZWEIGHT + out_col_${c} * CH + local.y;
  let wg_base_${c} = W_LINEARGWEIGHT + out_col_${c} * CZ + local.y;
  ${exactOutCols
    ? `let bias_${c} = ${outVector2}(
      ${accNarrow(readWeight(`weights[W_LINEARZBIAS + out_col_${c}]`))},
      ${accNarrow(readWeight(`weights[W_LINEARGBIAS + out_col_${c}]`))});`
    : `var bias_${c} = ${outVector2}(0.0);
  if (valid_col_${c}) {
    bias_${c} = ${outVector2}(
      ${accNarrow(readWeight(`weights[W_LINEARZBIAS + out_col_${c}]`))},
      ${accNarrow(readWeight(`weights[W_LINEARGBIAS + out_col_${c}]`))});
  }`}`);
    for (let r = 0; r < rowsPerThread; r += 1) {
      outBiasDeclare.push(`  var acc_${r}_${c} = bias_${c};`);
    }
  }
  const outStageWeight = Array.from({ length: outColumnsPerThread }, (_, c) => (exactOutCols && exactOutSteps)
    ? `    {
      let projection_w = ${readWeight(`weights[wz_base_${c} + k0]`)};
      let gate_w = ${readWeight(`weights[wg_base_${c} + k0]`)};
      tile_weight[w_slot_${c}] = ${outVector2}(${accNarrow("projection_w")}, ${accNarrow("gate_w")});
    }`
    : `    {
      var projection_w = 0.0;
      var gate_w = 0.0;
      ${exactOutCols ? "" : `if (valid_col_${c}) {\n        `}if (weight_k < CH) {
          projection_w = ${readWeight(`weights[wz_base_${c} + k0]`)};
        }
        if (weight_k < CZ) {
          gate_w = ${readWeight(`weights[wg_base_${c} + k0]`)};
        }${exactOutCols ? "" : "\n      }"}
      tile_weight[w_slot_${c}] = ${outVector2}(${accNarrow("projection_w")}, ${accNarrow("gate_w")});
    }`).join("\n");
  const outInner = Array.from({ length: PROJECT_W }, (_, k) => `    {
      let xs = tile_x[local.y * ${PROJECT_W}u + ${k}u];
      let zs = tile_z[local.y * ${PROJECT_W}u + ${k}u];
      ${overRows((r) => `let xz${r} = ${outVector2}(${rowAt("xs", r)}, ${rowAt("zs", r)});`)}
${Array.from({ length: outColumnsPerThread }, (_, c) => `      {
        let packed = tile_weight[${k * OUT_TILE_COLUMNS + c * PROJECT_W}u + local.x];
        ${overRows((r) => `acc_${r}_${c} += xz${r} * packed;`)}
      }`).join("\n")}
    }`).join("\n");
  const outStores = Array.from({ length: rowsPerThread }, (_, r) => `  {
    let row = row0 + ${r * PROJECT_W}u;
    if (row < PAIRS) {
${Array.from({ length: outColumnsPerThread }, (_, c) => `      {
        let out_channel = channel0 + ${c * PROJECT_W}u;
        if (out_channel < CZ) {
          store_output(row, out_channel, vec2<f32>(acc_${r}_${c}));
        }
      }`).join("\n")}
    }
  }`).join("\n");
  const projectOutput = `${common}
const TILE_ROWS: u32 = ${PROJECT_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${OUT_TILE_COLUMNS}u;

@group(0) @binding(0) var<storage, read> z: array<${storageArray(normalizedStorage)}>;
@group(0) @binding(1) var<storage, read> x: array<${storageArray(hiddenStorage)}>;
@group(0) @binding(2) var<storage, read> weights: array<${tw}>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> tile_x: array<${outRowVector}, ${PROJECT_W2}>;
var<workgroup> tile_z: array<${outRowVector}, ${PROJECT_W2}>;
// 🔴 ONE vec2 A CELL, NOT TWO SCALAR ARRAYS - the same argument projectAB
// makes with four matrices and a vec4. The output projection and its gate are
// contracted at the SAME (k, output channel) cell, so their two weights are
// always wanted together: packed, the inner loop reads them in one instruction
// and accumulates them in one multiply-add instead of two. Scalar, this kernel
// spent two source reads, four weight reads and sixteen multiply-adds a step of
// k to buy sixteen products - 0.73 useful operations an instruction, against
// projectAB's 2.9 on the same shape, and it showed: 672 GFLOP/s where the
// projection that feeds it runs at 977.
var<workgroup> tile_weight: array<${outVector2}, ${outColumnsPerThread * PROJECT_W2}>;

fn store_output(row: u32, out_channel: u32, cell: vec2<f32>) {
  output[row * CZ + out_channel] ${residual ? "+=" : "="} cell.x * logistic(cell.y);
}

@compute @workgroup_size(${PROJECT_W}, ${PROJECT_W}, 1)
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
  let tile_index = local.y * ${PROJECT_W}u + local.x;
  let full_rows = row0 + ${(rowsPerThread - 1) * PROJECT_W}u < PAIRS;
  let sx_base0 = row0 * CH_STRIDE + local.x;
  let sz_base0 = row0 * CZ_STRIDE + local.x;
  // ...the projection contracts over CH and the gate over CZ, on the same
  // output channel, so one accumulator a cell carries both: x is the
  // projection, y the gate.
${outBiasDeclare.join("\n")}
  for (var k0 = 0u; k0 < max(CH, CZ); k0 += ${PROJECT_W}u) {
    let source_k = k0 + local.x;
    ${exactOutSteps ? "" : "let weight_k = k0 + local.y;\n    "}var staged_x = ${outRowVector}(0.0);
    var staged_z = ${outRowVector}(0.0);
    if (full_rows) {
      ${exactOutSteps
        ? `let sx0 = sx_base0 + k0;
      let sz0 = sz_base0 + k0;
      ${overRows((r) => `${rowAt("staged_x", r)} = ${accNarrow(storedElement(hiddenStorage, "x", `sx0 + ${r * PROJECT_W}u * CH_STRIDE`))};`)}
      ${overRows((r) => `${rowAt("staged_z", r)} = ${accNarrow(storedElement(normalizedStorage, "z", `sz0 + ${r * PROJECT_W}u * CZ_STRIDE`))};`)}`
        : `if (source_k < CH) {
        let sx0 = sx_base0 + k0;
        ${overRows((r) => `${rowAt("staged_x", r)} = ${accNarrow(storedElement(hiddenStorage, "x", `sx0 + ${r * PROJECT_W}u * CH_STRIDE`))};`)}
      }
      if (source_k < CZ) {
        let sz0 = sz_base0 + k0;
        ${overRows((r) => `${rowAt("staged_z", r)} = ${accNarrow(storedElement(normalizedStorage, "z", `sz0 + ${r * PROJECT_W}u * CZ_STRIDE`))};`)}
      }`}
    } else {
      ${overRows((r) => `{
        let row = row0 + ${r * PROJECT_W}u;
        if (row < PAIRS) {
          ${exactOutSteps ? "" : "if (source_k < CH) { "}${rowAt("staged_x", r)} = ${accNarrow(storedElement(hiddenStorage, "x", "row * CH_STRIDE + source_k"))};${exactOutSteps ? "" : " }"}
          ${exactOutSteps ? "" : "if (source_k < CZ) { "}${rowAt("staged_z", r)} = ${accNarrow(storedElement(normalizedStorage, "z", "row * CZ_STRIDE + source_k"))};${exactOutSteps ? "" : " }"}
        }
      }`)}
    }
    tile_x[tile_index] = staged_x;
    tile_z[tile_index] = staged_z;
${outStageWeight}
    workgroupBarrier();
${outInner}
    workgroupBarrier();
  }
${outStores}
}`;

  return { normalizeInput, projectAB, contract: contractF16 ?? contract, normalizeHidden, projectOutput,
           projectTile: { ...projectTile },
           // ...the f16 contraction's tile is its own, whatever the 8 x 8 kernel's was
           contractTile: contractF16 === undefined ? { ...contractTile } : { rows: 64, columns: 64, arithmetic: "f16" },
           // ...the width the two projections fold their row tile at, so a
           // caller splits at the number the kernel was GENERATED with rather
           // than at a constant of its own that happens to match.
           projectGridWidth,
           normalizeRows: NORMALIZE_ROWS };
}
