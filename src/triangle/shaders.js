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

export function createTriangleShaders(
  shape,
  precision,
  offsets,
  epsilon = 1e-5,
  direction = "outgoing",
) {
  const common = prelude(shape, precision, offsets, epsilon);
  const t = scalar(precision);

  const normalizeInput = `${common}
@group(0) @binding(0) var<storage, read> source: array<${t}>;
@group(0) @binding(1) var<storage, read> weights: array<${t}>;
@group(0) @binding(2) var<storage, read_write> normalized: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= PAIRS) { return; }
  let base = row * CZ;
  var mean = 0.0;
  for (var c = 0u; c < CZ; c += 1u) {
    mean += ${read(precision, "source[base + c]")};
  }
  mean /= f32(CZ);
  var variance = 0.0;
  for (var c = 0u; c < CZ; c += 1u) {
    let centered = ${read(precision, "source[base + c]")} - mean;
    variance += centered * centered;
  }
  let inverse_std = inverseSqrt(variance / f32(CZ) + EPSILON);
  for (var c = 0u; c < CZ; c += 1u) {
    var value = (${read(precision, "source[base + c]")} - mean) * inverse_std;
    value = value * ${read(precision, "weights[W_LAYERNORMINWEIGHT + c]")}
      + ${read(precision, "weights[W_LAYERNORMINBIAS + c]")};
    normalized[base + c] = value;
  }
}`;

  const projectAB = `${common}
@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${t}>;
@group(0) @binding(3) var<storage, read_write> a: array<f32>;
@group(0) @binding(4) var<storage, read_write> b: array<f32>;

var<workgroup> tile_source: array<f32, 128>;
var<workgroup> tile_ap_weight: array<f32, 128>;
var<workgroup> tile_ag_weight: array<f32, 128>;
var<workgroup> tile_bp_weight: array<f32, 128>;
var<workgroup> tile_bg_weight: array<f32, 128>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row = group.y * 16u + local.y;
  let second_row = row + 8u;
  let h = group.x * 16u + local.x;
  let second_h = h + 8u;
  let tile_index = local.y * 8u + local.x;
  var ap_00 = 0.0; var ag_00 = 0.0; var bp_00 = 0.0; var bg_00 = 0.0;
  var ap_01 = 0.0; var ag_01 = 0.0; var bp_01 = 0.0; var bg_01 = 0.0;
  var ap_10 = 0.0; var ag_10 = 0.0; var bp_10 = 0.0; var bg_10 = 0.0;
  var ap_11 = 0.0; var ag_11 = 0.0; var bp_11 = 0.0; var bg_11 = 0.0;
  if (h < CH) {
    ap_00 = ${read(precision, "weights[W_LINEARAPBIAS + h]")}; ap_10 = ap_00;
    ag_00 = ${read(precision, "weights[W_LINEARAGBIAS + h]")}; ag_10 = ag_00;
    bp_00 = ${read(precision, "weights[W_LINEARBPBIAS + h]")}; bp_10 = bp_00;
    bg_00 = ${read(precision, "weights[W_LINEARBGBIAS + h]")}; bg_10 = bg_00;
  }
  if (second_h < CH) {
    ap_01 = ${read(precision, "weights[W_LINEARAPBIAS + second_h]")}; ap_11 = ap_01;
    ag_01 = ${read(precision, "weights[W_LINEARAGBIAS + second_h]")}; ag_11 = ag_01;
    bp_01 = ${read(precision, "weights[W_LINEARBPBIAS + second_h]")}; bp_11 = bp_01;
    bg_01 = ${read(precision, "weights[W_LINEARBGBIAS + second_h]")}; bg_11 = bg_01;
  }
  for (var c0 = 0u; c0 < CZ; c0 += 8u) {
    let source_c = c0 + local.x;
    let weight_c = c0 + local.y;
    tile_source[tile_index] = 0.0;
    tile_source[tile_index + 64u] = 0.0;
    if (row < PAIRS && source_c < CZ) { tile_source[tile_index] = z[row * CZ + source_c]; }
    if (second_row < PAIRS && source_c < CZ) {
      tile_source[tile_index + 64u] = z[second_row * CZ + source_c];
    }
    for (var h_block = 0u; h_block < 2u; h_block += 1u) {
      let output_h = h + h_block * 8u;
      let weight_tile_index = local.y * 16u + local.x + h_block * 8u;
      tile_ap_weight[weight_tile_index] = 0.0; tile_ag_weight[weight_tile_index] = 0.0;
      tile_bp_weight[weight_tile_index] = 0.0; tile_bg_weight[weight_tile_index] = 0.0;
      if (output_h < CH && weight_c < CZ) {
        let weight_index = output_h * CZ + weight_c;
        tile_ap_weight[weight_tile_index] = ${read(precision, "weights[W_LINEARAPWEIGHT + weight_index]")};
        tile_ag_weight[weight_tile_index] = ${read(precision, "weights[W_LINEARAGWEIGHT + weight_index]")};
        tile_bp_weight[weight_tile_index] = ${read(precision, "weights[W_LINEARBPWEIGHT + weight_index]")};
        tile_bg_weight[weight_tile_index] = ${read(precision, "weights[W_LINEARBGWEIGHT + weight_index]")};
      }
    }
    workgroupBarrier();
    for (var c = 0u; c < 8u; c += 1u) {
      let x_0 = tile_source[local.y * 8u + c];
      let x_1 = tile_source[local.y * 8u + c + 64u];
      let weight_0 = c * 16u + local.x;
      let weight_1 = weight_0 + 8u;
      ap_00 += x_0 * tile_ap_weight[weight_0]; ag_00 += x_0 * tile_ag_weight[weight_0];
      bp_00 += x_0 * tile_bp_weight[weight_0]; bg_00 += x_0 * tile_bg_weight[weight_0];
      ap_01 += x_0 * tile_ap_weight[weight_1]; ag_01 += x_0 * tile_ag_weight[weight_1];
      bp_01 += x_0 * tile_bp_weight[weight_1]; bg_01 += x_0 * tile_bg_weight[weight_1];
      ap_10 += x_1 * tile_ap_weight[weight_0]; ag_10 += x_1 * tile_ag_weight[weight_0];
      bp_10 += x_1 * tile_bp_weight[weight_0]; bg_10 += x_1 * tile_bg_weight[weight_0];
      ap_11 += x_1 * tile_ap_weight[weight_1]; ag_11 += x_1 * tile_ag_weight[weight_1];
      bp_11 += x_1 * tile_bp_weight[weight_1]; bg_11 += x_1 * tile_bg_weight[weight_1];
    }
    workgroupBarrier();
  }
  // ...AND a AND b COME OUT CHANNEL-MAJOR, a[h][i][j], not a[i][j][h]. The
  // contraction below walks one h at a time, so this is the layout that decides
  // whether it runs or crawls - see the note on its k loop. Writing it here
  // costs this pass a scattered store; it saved the contraction 14x at L=256.
  if (row < PAIRS && h < CH) {
    let index = h * PAIRS + row; let pair_mask = mask[row];
    a[index] = pair_mask * ap_00 * logistic(ag_00); b[index] = pair_mask * bp_00 * logistic(bg_00);
  }
  if (row < PAIRS && second_h < CH) {
    let index = second_h * PAIRS + row; let pair_mask = mask[row];
    a[index] = pair_mask * ap_01 * logistic(ag_01); b[index] = pair_mask * bp_01 * logistic(bg_01);
  }
  if (second_row < PAIRS && h < CH) {
    let index = h * PAIRS + second_row; let pair_mask = mask[second_row];
    a[index] = pair_mask * ap_10 * logistic(ag_10); b[index] = pair_mask * bp_10 * logistic(bg_10);
  }
  if (second_row < PAIRS && second_h < CH) {
    let index = second_h * PAIRS + second_row; let pair_mask = mask[second_row];
    a[index] = pair_mask * ap_11 * logistic(ag_11); b[index] = pair_mask * bp_11 * logistic(bg_11);
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= PAIRS) { return; }
  // ...READS CHANNEL-MAJOR AND WRITES CHANNEL-MINOR, because this is where the
  // two layouts meet: the contraction upstream wants h slowest, and
  // projectOutput downstream reads a row's channels contiguously. Doing the
  // transpose in a pass that already touches every element once is free next
  // to a dedicated one. The read is the coalesced half - consecutive lanes are
  // consecutive rows - which is the better half to give the three passes over
  // source.
  let base = row * CH;
  var mean = 0.0;
  for (var h = 0u; h < CH; h += 1u) { mean += source[h * PAIRS + row]; }
  mean /= f32(CH);
  var variance = 0.0;
  for (var h = 0u; h < CH; h += 1u) {
    let centered = source[h * PAIRS + row] - mean;
    variance += centered * centered;
  }
  let inverse_std = inverseSqrt(variance / f32(CH) + EPSILON);
  for (var h = 0u; h < CH; h += 1u) {
    var value = (source[h * PAIRS + row] - mean) * inverse_std;
    value = value * ${read(precision, "weights[W_LAYERNORMOUTWEIGHT + h]")}
      + ${read(precision, "weights[W_LAYERNORMOUTBIAS + h]")};
    normalized[base + h] = value;
  }
}`;

  const projectOutput = `${common}
@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${t}>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> tile_x: array<f32, 128>;
var<workgroup> tile_z: array<f32, 128>;
var<workgroup> tile_projection_weight: array<f32, 128>;
var<workgroup> tile_gate_weight: array<f32, 128>;

fn write_output(row: u32, out_channel: u32, projected: f32, gate: f32) {
  let index = row * CZ + out_channel;
  output[index] = projected * logistic(gate);
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row = group.y * 16u + local.y;
  let second_row = row + 8u;
  let out_channel = group.x * 16u + local.x;
  let second_channel = out_channel + 8u;
  let tile_index = local.y * 8u + local.x;
  var projected_00 = 0.0; var gate_00 = 0.0;
  var projected_01 = 0.0; var gate_01 = 0.0;
  var projected_10 = 0.0; var gate_10 = 0.0;
  var projected_11 = 0.0; var gate_11 = 0.0;
  if (out_channel < CZ) {
    projected_00 = ${read(precision, "weights[W_LINEARZBIAS + out_channel]")}; projected_10 = projected_00;
    gate_00 = ${read(precision, "weights[W_LINEARGBIAS + out_channel]")}; gate_10 = gate_00;
  }
  if (second_channel < CZ) {
    projected_01 = ${read(precision, "weights[W_LINEARZBIAS + second_channel]")}; projected_11 = projected_01;
    gate_01 = ${read(precision, "weights[W_LINEARGBIAS + second_channel]")}; gate_11 = gate_01;
  }
  for (var k0 = 0u; k0 < max(CH, CZ); k0 += 8u) {
    let source_k = k0 + local.x;
    let weight_k = k0 + local.y;
    tile_x[tile_index] = 0.0; tile_x[tile_index + 64u] = 0.0;
    tile_z[tile_index] = 0.0; tile_z[tile_index + 64u] = 0.0;
    if (row < PAIRS && source_k < CH) { tile_x[tile_index] = x[row * CH + source_k]; }
    if (second_row < PAIRS && source_k < CH) {
      tile_x[tile_index + 64u] = x[second_row * CH + source_k];
    }
    if (row < PAIRS && source_k < CZ) { tile_z[tile_index] = z[row * CZ + source_k]; }
    if (second_row < PAIRS && source_k < CZ) {
      tile_z[tile_index + 64u] = z[second_row * CZ + source_k];
    }
    for (var channel_block = 0u; channel_block < 2u; channel_block += 1u) {
      let channel = out_channel + channel_block * 8u;
      let weight_tile_index = local.y * 16u + local.x + channel_block * 8u;
      tile_projection_weight[weight_tile_index] = 0.0; tile_gate_weight[weight_tile_index] = 0.0;
      if (channel < CZ && weight_k < CH) {
        tile_projection_weight[weight_tile_index] = ${read(precision,
          "weights[W_LINEARZWEIGHT + channel * CH + weight_k]")};
      }
      if (channel < CZ && weight_k < CZ) {
        tile_gate_weight[weight_tile_index] = ${read(precision,
          "weights[W_LINEARGWEIGHT + channel * CZ + weight_k]")};
      }
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      let x_0 = tile_x[local.y * 8u + k]; let x_1 = tile_x[local.y * 8u + k + 64u];
      let z_0 = tile_z[local.y * 8u + k]; let z_1 = tile_z[local.y * 8u + k + 64u];
      let weight_0 = k * 16u + local.x; let weight_1 = weight_0 + 8u;
      projected_00 += x_0 * tile_projection_weight[weight_0]; gate_00 += z_0 * tile_gate_weight[weight_0];
      projected_01 += x_0 * tile_projection_weight[weight_1]; gate_01 += z_0 * tile_gate_weight[weight_1];
      projected_10 += x_1 * tile_projection_weight[weight_0]; gate_10 += z_1 * tile_gate_weight[weight_0];
      projected_11 += x_1 * tile_projection_weight[weight_1]; gate_11 += z_1 * tile_gate_weight[weight_1];
    }
    workgroupBarrier();
  }
  if (row < PAIRS && out_channel < CZ) { write_output(row, out_channel, projected_00, gate_00); }
  if (row < PAIRS && second_channel < CZ) { write_output(row, second_channel, projected_01, gate_01); }
  if (second_row < PAIRS && out_channel < CZ) { write_output(second_row, out_channel, projected_10, gate_10); }
  if (second_row < PAIRS && second_channel < CZ) {
    write_output(second_row, second_channel, projected_11, gate_11);
  }
}`;

  return { normalizeInput, projectAB, contract, normalizeHidden, projectOutput };
}
