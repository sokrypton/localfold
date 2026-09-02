const declaration = (precision) => precision === "f16" ? "enable f16;\n" : "";
const scalar = (precision) => precision;
const read = (precision, expression) =>
  precision === "f16" ? `f32(${expression})` : expression;

function prelude(shape, precision, offsets, epsilon) {
  const offsetConstants = Object.entries(offsets)
    .map(([name, offset]) => `const W_${name.toUpperCase()}: u32 = ${offset}u;`)
    .join("\n");
  return `${declaration(precision)}
const L: u32 = ${shape.length}u;
const CZ: u32 = ${shape.cZ}u;
const CH: u32 = ${shape.cHidden}u;
const PAIRS: u32 = L * L;
const LINEAR_GRID_WIDTH: u32 = 32768u;
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

export function createTriangleShaders(
  shape,
  precision,
  offsets,
  epsilon = 1e-5,
  direction = "outgoing",
  variance = "two-pass",
  projectTile = PROJECT_TILE,
  residual = false,
) {
  if (variance !== "two-pass" && variance !== "fast") {
    throw new Error(`variance must be "two-pass" or "fast", not ${variance}`);
  }
  const common = prelude(shape, precision, offsets, epsilon);
  const t = scalar(precision);

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
  const stagedLayerNorm = (count, load, scale, offset, sourceMajor = "row") => `
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

  for (var index = local; index < NORMALIZE_ROWS * ${count}; index += 64u) {
    let row = base_row + index / ${count};
    if (row >= PAIRS) { continue; }
    let channel = index % ${count};
    let value = (tile[index] - row_mean[index / ${count}])
      * row_inverse_std[index / ${count}];
    normalized[row * ${count} + channel] = value
      * ${read(precision, `weights[W_${scale} + channel]`)}
      + ${read(precision, `weights[W_${offset} + channel]`)};
  }
}`;

  const normalizeInput = `${common}
@group(0) @binding(0) var<storage, read> source: array<${t}>;
@group(0) @binding(1) var<storage, read> weights: array<${t}>;
@group(0) @binding(2) var<storage, read_write> normalized: array<f32>;
${stagedLayerNorm("CZ", (row, channel) => read(precision, `source[${row} * CZ + ${channel}]`),
                  "LAYERNORMINWEIGHT", "LAYERNORMINBIAS")}`;

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
  const { rows: PROJECT_TILE_ROWS, columns: PROJECT_TILE_COLUMNS } = projectTile;
  if (PROJECT_TILE_ROWS % 8 !== 0 || PROJECT_TILE_COLUMNS % 8 !== 0) {
    throw new Error(`projectTile ${PROJECT_TILE_ROWS}x${PROJECT_TILE_COLUMNS} is not a `
      + "multiple of the 8x8 workgroup");
  }
  const rowsPerThread = PROJECT_TILE_ROWS / 8;
  const columnsPerThread = PROJECT_TILE_COLUMNS / 8;
  const MATRICES = [["ap", "LINEARAPWEIGHT", "LINEARAPBIAS"], ["ag", "LINEARAGWEIGHT", "LINEARAGBIAS"],
                    ["bp", "LINEARBPWEIGHT", "LINEARBPBIAS"], ["bg", "LINEARBGWEIGHT", "LINEARBGBIAS"]];

  const projectAB = `${common}
const TILE_ROWS: u32 = ${PROJECT_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${PROJECT_TILE_COLUMNS}u;

@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${t}>;
@group(0) @binding(3) var<storage, read_write> a: array<f32>;
@group(0) @binding(4) var<storage, read_write> b: array<f32>;

var<workgroup> tile_source: array<f32, ${rowsPerThread * 64}>;
${MATRICES.map(([name]) =>
  `var<workgroup> tile_${name}_weight: array<f32, ${columnsPerThread * 64}>;`).join("\n")}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row0 = group.y * TILE_ROWS + local.y;
  let h0 = group.x * TILE_COLUMNS + local.x;
  let tile_index = local.y * 8u + local.x;
${MATRICES.map(([name]) =>
  `  var ${name}: array<f32, ${rowsPerThread * columnsPerThread}>;`).join("\n")}
  for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
    let h = h0 + column * 8u;
    ${MATRICES.map(([name, , bias]) =>
      `var ${name}_bias = 0.0;`).join(" ")}
    if (h < CH) {
      ${MATRICES.map(([name, , bias]) =>
        `${name}_bias = ${read(precision, `weights[W_${bias} + h]`)};`).join("\n      ")}
    }
    for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
      ${MATRICES.map(([name]) =>
        `${name}[r * ${columnsPerThread}u + column] = ${name}_bias;`).join("\n      ")}
    }
  }
  for (var c0 = 0u; c0 < CZ; c0 += 8u) {
    let source_c = c0 + local.x;
    let weight_c = c0 + local.y;
    for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
      let row = row0 + r * 8u;
      var value = 0.0;
      if (row < PAIRS && source_c < CZ) { value = z[row * CZ + source_c]; }
      tile_source[r * 64u + tile_index] = value;
    }
    for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
      let h = h0 + column * 8u;
      let slot = local.y * TILE_COLUMNS + local.x + column * 8u;
      ${MATRICES.map(([name]) => `var ${name}_w = 0.0;`).join(" ")}
      if (h < CH && weight_c < CZ) {
        let weight_index = h * CZ + weight_c;
        ${MATRICES.map(([name, weight]) =>
          `${name}_w = ${read(precision, `weights[W_${weight} + weight_index]`)};`).join("\n        ")}
      }
      ${MATRICES.map(([name]) => `tile_${name}_weight[slot] = ${name}_w;`).join("\n      ")}
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      var x: array<f32, ${rowsPerThread}>;
      for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
        x[r] = tile_source[r * 64u + local.y * 8u + k];
      }
      for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
        let slot = k * TILE_COLUMNS + local.x + column * 8u;
        ${MATRICES.map(([name]) => `let ${name}_w = tile_${name}_weight[slot];`).join("\n        ")}
        for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
          let at = r * ${columnsPerThread}u + column;
          ${MATRICES.map(([name]) => `${name}[at] += x[r] * ${name}_w;`).join("\n          ")}
        }
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
  // subgroup, so parking the 16x16 tile in workgroup memory and writing it back
  // indexed by row looks like the obvious fix. Measured interleaved against
  // this version, bitwise-identical output, it lost at every length:
  //   L=64 0.92x   L=128 0.88x   L=192 0.95x   L=256 0.93x   (and 0.74x
  // incoming at 64). Two more arrays of 256 floats cost occupancy, and the
  // extra barrier and indexed writeback cost more than the coalescing bought.
  for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
    let row = row0 + r * 8u;
    if (row >= PAIRS) { continue; }
    let pair_mask = mask[row];
    for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
      let h = h0 + column * 8u;
      if (h >= CH) { continue; }
      let at = r * ${columnsPerThread}u + column;
      let index = h * PAIRS + row;
      a[index] = pair_mask * ap[at] * logistic(ag[at]);
      b[index] = pair_mask * bp[at] * logistic(bg[at]);
    }
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
  const loadATile = direction === "outgoing"
    ? "a[h * PAIRS + i * L + a_k]"
    : "b[h * PAIRS + a_k * L + i]";
  const loadBTile = direction === "outgoing"
    ? "b[h * PAIRS + j * L + b_k]"
    : "a[h * PAIRS + b_k * L + j]";
  const contract = `${common}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> tile_a: array<f32, 64>;
var<workgroup> tile_b: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let i = group.y * 8u + local.y;
  let j = group.x * 8u + local.x;
  let h = group.z;
  let tile_index = local.y * 8u + local.x;
  var sum = 0.0;

  for (var k0 = 0u; k0 < L; k0 += 8u) {
    let a_k = k0 + local.x;
    let b_k = k0 + local.y;
    tile_a[tile_index] = 0.0;
    tile_b[tile_index] = 0.0;
    if (i < L && a_k < L) {
      tile_a[tile_index] = ${loadATile};
    }
    if (j < L && b_k < L) {
      tile_b[tile_index] = ${loadBTile};
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      sum += tile_a[local.y * 8u + k] * tile_b[k * 8u + local.x];
    }
    workgroupBarrier();
  }

  if (i < L && j < L) {
    output[h * PAIRS + i * L + j] = sum;
  }
}`;

  const normalizeHidden = `${common}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${t}>;
@group(0) @binding(2) var<storage, read_write> normalized: array<f32>;
// ...READS CHANNEL-MAJOR AND WRITES CHANNEL-MINOR, because this is where the
// two layouts meet: the contraction upstream wants h slowest, and
// projectOutput downstream reads a row's channels contiguously. Doing the
// transpose in a pass that already touches every element once is free next to
// a dedicated one, and staging it through workgroup memory is what lets BOTH
// sides be read and written along their own major axis.
${stagedLayerNorm("CH", (row, channel) => `source[${channel} * PAIRS + ${row}]`,
                  "LAYERNORMOUTWEIGHT", "LAYERNORMOUTBIAS", "channel")}`;

  const projectOutput = `${common}
const TILE_ROWS: u32 = ${PROJECT_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${PROJECT_TILE_COLUMNS}u;

@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${t}>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> tile_x: array<f32, ${rowsPerThread * 64}>;
var<workgroup> tile_z: array<f32, ${rowsPerThread * 64}>;
var<workgroup> tile_projection_weight: array<f32, ${columnsPerThread * 64}>;
var<workgroup> tile_gate_weight: array<f32, ${columnsPerThread * 64}>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row0 = group.y * TILE_ROWS + local.y;
  let channel0 = group.x * TILE_COLUMNS + local.x;
  let tile_index = local.y * 8u + local.x;
  // ...the projection contracts over CH and the gate over CZ, on the same
  // output channel, so one register block carries two accumulators per cell.
  var projected: array<f32, ${rowsPerThread * columnsPerThread}>;
  var gated: array<f32, ${rowsPerThread * columnsPerThread}>;
  for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
    let out_channel = channel0 + column * 8u;
    var projection_bias = 0.0;
    var gate_bias = 0.0;
    if (out_channel < CZ) {
      projection_bias = ${read(precision, "weights[W_LINEARZBIAS + out_channel]")};
      gate_bias = ${read(precision, "weights[W_LINEARGBIAS + out_channel]")};
    }
    for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
      projected[r * ${columnsPerThread}u + column] = projection_bias;
      gated[r * ${columnsPerThread}u + column] = gate_bias;
    }
  }
  for (var k0 = 0u; k0 < max(CH, CZ); k0 += 8u) {
    let source_k = k0 + local.x;
    let weight_k = k0 + local.y;
    for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
      let row = row0 + r * 8u;
      var x_value = 0.0;
      var z_value = 0.0;
      if (row < PAIRS && source_k < CH) { x_value = x[row * CH + source_k]; }
      if (row < PAIRS && source_k < CZ) { z_value = z[row * CZ + source_k]; }
      tile_x[r * 64u + tile_index] = x_value;
      tile_z[r * 64u + tile_index] = z_value;
    }
    for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
      let out_channel = channel0 + column * 8u;
      let slot = local.y * TILE_COLUMNS + local.x + column * 8u;
      var projection_w = 0.0;
      var gate_w = 0.0;
      if (out_channel < CZ && weight_k < CH) {
        projection_w = ${read(precision, "weights[W_LINEARZWEIGHT + out_channel * CH + weight_k]")};
      }
      if (out_channel < CZ && weight_k < CZ) {
        gate_w = ${read(precision, "weights[W_LINEARGWEIGHT + out_channel * CZ + weight_k]")};
      }
      tile_projection_weight[slot] = projection_w;
      tile_gate_weight[slot] = gate_w;
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      var xs: array<f32, ${rowsPerThread}>;
      var zs: array<f32, ${rowsPerThread}>;
      for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
        xs[r] = tile_x[r * 64u + local.y * 8u + k];
        zs[r] = tile_z[r * 64u + local.y * 8u + k];
      }
      for (var column = 0u; column < ${columnsPerThread}u; column += 1u) {
        let slot = k * TILE_COLUMNS + local.x + column * 8u;
        let projection_w = tile_projection_weight[slot];
        let gate_w = tile_gate_weight[slot];
        for (var r = 0u; r < ${rowsPerThread}u; r += 1u) {
          let at = r * ${columnsPerThread}u + column;
          projected[at] += xs[r] * projection_w;
          gated[at] += zs[r] * gate_w;
        }
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
      let at = r * ${columnsPerThread}u + column;
      output[row * CZ + out_channel] ${residual ? "+=" : "="} projected[at] * logistic(gated[at]);
    }
  }
}`;

  return { normalizeInput, projectAB, contract, normalizeHidden, projectOutput,
           projectTile: { ...projectTile }, normalizeRows: NORMALIZE_ROWS };
}
