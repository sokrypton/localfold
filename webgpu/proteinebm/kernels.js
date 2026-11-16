/**
 * The row kernels ProteinEBM's forward and backward need beyond a GEMM and an elementwise map: a LayerNorm that SAVES
 * its statistics, the LayerNorm's input gradient from them, and a row softmax and its backward.
 *
 * 🔴 FIREFOX-SAFE REDUCTIONS: every lane writes its own partial, and after the barrier EVERY lane folds all 64 itself.
 * A single lane storing the total for the others to read is invisible to them after a barrier in Firefox on Apple
 * silicon (docs/FIREFOX.md), which is the first thing a reduction written the usual way does.
 *
 * Rows are one workgroup each, folded into y past 65535 and clamped: a lane past the last row computes on the last row
 * (so every barrier stays uniform) and writes nothing.
 */

const HEAD = `
const LANES = 64u;
var<workgroup> red: array<f32, 64>;
fn total(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  var t = 0.0;
  for (var k = 0u; k < LANES; k += 1u) { t += red[k]; }
  workgroupBarrier();
  return t;
}
fn biggest(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  var t = red[0];
  for (var k = 1u; k < LANES; k += 1u) { t = max(t, red[k]); }
  workgroupBarrier();
  return t;
}`;

/**
 * y = (x - mean) rstd [* w] [+ b] over rows of `width`, and stats[2 r] = mean, stats[2 r + 1] = rstd.
 * affine: "none" | "w" | "wb". y may be omitted (`noY`): the statistics alone.
 */
export function lnForwardSource(affine, noY = false) {
  return `
struct P { rows: u32, width: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
${noY ? "" : "@group(0) @binding(3) var<storage, read_write> y: array<f32>;"}
${affine !== "none" ? "@group(0) @binding(4) var<storage, read> w: array<f32>;" : ""}
${affine === "wb" ? "@group(0) @binding(5) var<storage, read> b: array<f32>;" : ""}
${HEAD}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.x + wid.y * 65535u;
  let live = row < p.rows;
  let r = min(row, p.rows - 1u);
  let base = r * p.width;
  var s = 0.0;
  for (var c = li; c < p.width; c += LANES) { s += x[base + c]; }
  let mean = total(s, li) / f32(p.width);
  var q = 0.0;
  for (var c = li; c < p.width; c += LANES) { let d = x[base + c] - mean; q += d * d; }
  let rstd = inverseSqrt(total(q, li) / f32(p.width) + p.eps);
  if (!live) { return; }
  if (li == 0u) { stats[2u * r] = mean; stats[2u * r + 1u] = rstd; }
  ${noY ? "" : `for (var c = li; c < p.width; c += LANES) {
    var v = (x[base + c] - mean) * rstd;
    ${affine !== "none" ? "v = v * w[c];" : ""}
    ${affine === "wb" ? "v = v + b[c];" : ""}
    y[base + c] = v;
  }`}
}`;
}

/**
 * dx (+)= rstd (g - mean(g) - xhat mean(g xhat)), g = dy [* w]: a LayerNorm's input gradient from its saved
 * statistics (a bias does not enter it).
 */
export function lnBackwardSource(hasW, accumulate) {
  return `
struct P { rows: u32, width: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
${hasW ? "@group(0) @binding(5) var<storage, read> w: array<f32>;" : ""}
${HEAD}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.x + wid.y * 65535u;
  let live = row < p.rows;
  let r = min(row, p.rows - 1u);
  let base = r * p.width;
  let mean = stats[2u * r];
  let rstd = stats[2u * r + 1u];
  var s1 = 0.0;
  var s2 = 0.0;
  for (var c = li; c < p.width; c += LANES) {
    let g = dy[base + c]${hasW ? " * w[c]" : ""};
    s1 += g;
    s2 += g * (x[base + c] - mean) * rstd;
  }
  let m1 = total(s1, li) / f32(p.width);
  let m2 = total(s2, li) / f32(p.width);
  if (!live) { return; }
  for (var c = li; c < p.width; c += LANES) {
    let g = dy[base + c]${hasW ? " * w[c]" : ""};
    let xh = (x[base + c] - mean) * rstd;
    let v = rstd * (g - m1 - xh * m2);
    ${accumulate ? "dx[base + c] = dx[base + c] + v;" : "dx[base + c] = v;"}
  }
}`;
}

/** In place over rows of `len`: l = softmax(l). */
export const SOFTMAX_SOURCE = `
struct P { rows: u32, len: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> l: array<f32>;
${HEAD}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.x + wid.y * 65535u;
  let live = row < p.rows;
  let r = min(row, p.rows - 1u);
  let base = r * p.len;
  var m = -3.0e38;
  for (var c = li; c < p.len; c += LANES) { m = max(m, l[base + c]); }
  let top = biggest(m, li);
  var s = 0.0;
  for (var c = li; c < p.len; c += LANES) { s += exp(l[base + c] - top); }
  let inv = 1.0 / total(s, li);
  if (!live) { return; }
  for (var c = li; c < p.len; c += LANES) { l[base + c] = exp(l[base + c] - top) * inv; }
}`;

/** In place over rows of `len`: d = P (d - sum(d P)), the softmax's input gradient from its output P. */
export const SOFTMAX_BACKWARD_SOURCE = `
struct P { rows: u32, len: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> prob: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
${HEAD}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.x + wid.y * 65535u;
  let live = row < p.rows;
  let r = min(row, p.rows - 1u);
  let base = r * p.len;
  var s = 0.0;
  for (var c = li; c < p.len; c += LANES) { s += d[base + c] * prob[base + c]; }
  let dot = total(s, li);
  if (!live) { return; }
  for (var c = li; c < p.len; c += LANES) { d[base + c] = prob[base + c] * (d[base + c] - dot); }
}`;
