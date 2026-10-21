/**
 * The kernels BioEmu2 runs, as RECORDED dispatches.
 *
 * 🔴 A DENOISER CALL IS ~700 DISPATCHES AND A FOLD MAKES 200 OF THEM PER
 * SAMPLE, SO NOTHING HERE IS CREATED PER CALL. Every op takes its buffers and
 * shape when it is RECORDED - a bind group and a small uniform block made once
 * - and returns a step that is replayed: `program.run(pass)` encodes the same
 * list of pipelines and bind groups every time. What changes between sampler
 * steps (the coordinates, sigma) lives in buffers the steps read, written with
 * one `writeBuffer` each.
 *
 * Activations are float32. Weights are float32, or float16 packed two to a u32
 * where `half` is asked and the reduction is even (`unpack2x16float` is core
 * WGSL and needs no `shader-f16`).
 *
 * 🔴 EVERY KERNEL GUARDS ITS FOLDED INDEX before it subscripts a buffer
 * (test/folded-grid-guard.test.js): a grid is rounded up to whole workgroups
 * and whole rows of 65535, and an unguarded tail clamps on Metal.
 */
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { planBlockUpload, runBlockUpload } from "../weights/quantised-upload.js";

const MAX_X = 65535;
const STORAGE = 0x80;      // GPUBufferUsage.STORAGE
const COPY_SRC = 0x4;
const COPY_DST = 0x8;
const UNIFORM = 0x40;

/** A grid of `groups` workgroups folded into (x, y) under the per-dimension limit. */
export function fold2(groups) {
  if (groups <= MAX_X) return [Math.max(groups, 1), 1];
  return [MAX_X, Math.ceil(groups / MAX_X)];
}

const FOLD = "let flat = wid.x + wid.y * 65535u;";

// --- the shaders -------------------------------------------------------------------------------------------

/**
 * C = alpha * A . B^T (+ bias) [act] (+ C), over (b0, b1) batches, every operand addressed by strides.
 *
 * A (m, k) at offA + b0 sA0 + b1 sA1 + m sAm + k sAk; B (n, k) likewise; C (m, n) likewise. A 64 x 64 tile, 256
 * lanes of 4 x 4, K in steps of 16. `wordB` means B is float16 packed two to a word along k (sBk must be 1, every
 * offset even): its element (n, k) is half k & 1 of word (offB + ... + n sBn + k) / 2.
 */
function gemmSource(wordB) {
  return `
struct P { m: u32, n: u32, k: u32, b1: u32,
  offA: u32, sA0: u32, sA1: u32, sAm: u32, sAk: u32,
  offB: u32, sB0: u32, sB1: u32, sBn: u32, sBk: u32,
  offC: u32, sC0: u32, sC1: u32, sCm: u32, sCn: u32,
  bias: u32, act: u32, accumulate: u32, alpha: f32, batches: u32, split: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<${wordB ? "u32" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
var<workgroup> ta: array<f32, 1024>;   // [16][64], k-major
var<workgroup> tb: array<f32, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 63u) / 64u;
  let tilesM = (p.m + 63u) / 64u;
  let flat = wid.x + wid.y * 65535u;
  let batch = flat / (tilesN * tilesM);
  let inBatch = batch < p.batches;
  let tile = flat % (tilesN * tilesM);
  let m0 = (tile / tilesN) * 64u;
  let n0 = (tile % tilesN) * 64u;
  let b0 = batch / p.b1;
  let b1 = batch % p.b1;
  let baseA = p.offA + b0 * p.sA0 + b1 * p.sA1;
  let baseB = p.offB + b0 * p.sB0 + b1 * p.sB1;
  let tm = (li / 16u) * 4u;
  let tn = (li % 16u) * 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i += 1u) { acc[i] = 0.0; }
  // split-K: batch index b1 is the K slice, [kStart, kEnd), and the epilogue is the reduction's
  let kStart = select(0u, b1 * p.split, p.split > 0u);
  let kEnd = select(p.k, min(p.k, kStart + p.split), p.split > 0u);
  for (var k0 = kStart; k0 < kEnd; k0 += 16u) {
    for (var e = 0u; e < 4u; e += 1u) {
      let slot = li * 4u + e;          // 0..1023: row = slot / 16, k = slot % 16 (k fastest, for row-major operands)
      let kk = slot % 16u;
      let rr = slot / 16u;
      let k = k0 + kk;
      var va = 0.0;
      if (inBatch && k < kEnd && m0 + rr < p.m) { va = a[baseA + (m0 + rr) * p.sAm + k * p.sAk]; }
      ta[kk * 64u + rr] = va;
      var vb = 0.0;
      if (inBatch && k < kEnd && n0 + rr < p.n) {
        ${wordB
          ? `let at = baseB + (n0 + rr) * p.sBn + k;
        let pair = unpack2x16float(b[at / 2u]);
        vb = select(pair.x, pair.y, (at & 1u) == 1u);`
          : "vb = b[baseB + (n0 + rr) * p.sBn + k * p.sBk];"}
      }
      tb[kk * 64u + rr] = vb;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk += 1u) {
      let av = vec4<f32>(ta[kk * 64u + tm], ta[kk * 64u + tm + 1u], ta[kk * 64u + tm + 2u], ta[kk * 64u + tm + 3u]);
      let bv = vec4<f32>(tb[kk * 64u + tn], tb[kk * 64u + tn + 1u], tb[kk * 64u + tn + 2u], tb[kk * 64u + tn + 3u]);
      for (var i = 0u; i < 4u; i += 1u) {
        acc[i * 4u] += av[i] * bv.x;
        acc[i * 4u + 1u] += av[i] * bv.y;
        acc[i * 4u + 2u] += av[i] * bv.z;
        acc[i * 4u + 3u] += av[i] * bv.w;
      }
    }
    workgroupBarrier();
  }
  if (!inBatch) { return; }
  let baseC = p.offC + b0 * p.sC0 + b1 * p.sC1;
  for (var i = 0u; i < 4u; i += 1u) {
    let m = m0 + tm + i;
    if (m >= p.m) { continue; }
    for (var j = 0u; j < 4u; j += 1u) {
      let n = n0 + tn + j;
      if (n >= p.n) { continue; }
      var v = acc[i * 4u + j] * p.alpha;
      if (p.split > 0u) { c[baseC + m * p.sCm + n * p.sCn] = v; continue; }
      if (p.bias != 0u) { v += bias[n]; }
      if (p.act == 1u) { v = max(v, 0.0); }
      if (p.act == 2u) { v = 1.0 / (1.0 + exp(-v)); }
      let at = baseC + m * p.sCm + n * p.sCn;
      if (p.accumulate != 0u) { v += c[at]; }
      c[at] = v;
    }
  }
}`;
}

/**
 * The same GEMM retiled for an Apple part (webgpu/kernels/vector-gemm-block.js's lesson): a 64 x 128 block, 256 lanes of
 * 4 x 8, the panels read as vec4s - three workgroup reads for 32 multiply-adds where gemmSource's 4 x 4 lane makes eight
 * for 16. The same interface (strides, batches, split-K, epilogue); 10-15% a GEMM on an M5, a denoiser call 287 -> 266 ms.
 *
 * 🔴 FLOAT32 ON PURPOSE. f16 staging (the panels in half, multiplied in half, promoted to a float accumulator) was built
 * and measured on the normalised-input GEMMs: promoted every 32 products, 266 -> 237 ms at five times the shipped arm's
 * error against a float32 reference (sigma 16: 2.8e-3 -> 1.4e-2) - a running sum in half rounds against itself; every 8
 * or 4, the error back to metal/bioemu2's fast arm's and the speed with it (267, 278 ms). This arm keeps float32
 * activations and only the weights in half, which is why it is the more accurate of the two ports.
 */
function gemm2Source(wordB, pair = null, b16 = false, gate = null, rpt = 4) {
  const BK = 16;
  const BM = rpt * 16;
  const E = "f32";
  // (b16: B's panel - a half weight, unpacked - staged as f16, exact, and widened as it is read: a third less
  // workgroup memory a step, which is what bounds this kernel)
  const EB = b16 ? "f16" : "f32";
  const rows = Array.from({ length: rpt }, (_, i) => i);
  return `${b16 ? "enable f16;\n" : ""}
struct P { m: u32, n: u32, k: u32, b1: u32,
  offA: u32, sA0: u32, sA1: u32, sAm: u32, sAk: u32,
  offB: u32, sB0: u32, sB1: u32, sBn: u32, sBk: u32,
  offC: u32, sC0: u32, sC1: u32, sCm: u32, sCn: u32,
  bias: u32, act: u32, accumulate: u32, alpha: f32, batches: u32, split: u32, biasOff: u32, biasS0: u32,
  gS: u32, gOff: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<${wordB ? "u32" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
${gate ? `// (gate: c += gcond[row' gS + gOff + n] v, row' the row through gmap - a negative entry none - an adaLN-gated residual)
@group(0) @binding(5) var<storage, read> gcond: array<f32>;
${gate === "map" ? "@group(0) @binding(6) var<storage, read> gmap: array<i32>;" : ""}` : ""}
${pair ? `${pair.decls}
// (pair mode: tile column 0..63 is the gate, and 64..127 is its value at +${pair.half} - so
// a lane's acc_i_0 (tx) and acc_i_1 (tx + 16) are four gates and their four values, handed to pairOut together)
fn bRow(v: u32) -> u32 { return ((v / 64u) % 2u) * ${pair.half}u + (v / 128u) * 64u + v % 64u; }
${pair.body}` : "fn bRow(v: u32) -> u32 { return v; }"}
var<workgroup> ta: array<vec4<${E}>, ${BK * 17}>;    // [BK][64] k-major, padded to 17 vec4s to avoid bank conflicts
${rpt === 5 ? `var<workgroup> ta4: array<${E}, ${BK * 17}>;` : ""}
var<workgroup> tb: array<vec4<${EB}>, ${BK * 33}>;    // [BK][128], padded to 33 vec4s to avoid bank conflicts
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 127u) / 128u;
  let tilesM = (p.m + ${BM - 1}u) / ${BM}u;
  let flat = wid.x + wid.y * 65535u;
  let batch = flat / (tilesN * tilesM);
  let inBatch = batch < p.batches;
  let tile = flat % (tilesN * tilesM);
  let m0 = (tile / tilesN) * ${BM}u;
  let n0 = (tile % tilesN) * 128u;
  let b0 = batch / p.b1;
  let b1 = batch % p.b1;
  let baseA = p.offA + b0 * p.sA0 + b1 * p.sA1;
  let baseB = p.offB + b0 * p.sB0 + b1 * p.sB1;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * ${rpt}u;
  let tn0 = tx * 4u;
  let tn1 = tn0 + 64u;
  ${rows.map((i) => `var acc_${i}_0 = vec4<f32>(0.0); var acc_${i}_1 = vec4<f32>(0.0);`).join("\n  ")}
  let kStart = select(0u, b1 * p.split, p.split > 0u);
  let kEnd = select(p.k, min(p.k, kStart + p.split), p.split > 0u);
  let kkA = li % ${BK}u;
  let r4A = li / ${BK}u;
  let rowA0 = m0 + r4A * ${rpt}u;
  let aBase0 = baseA + rowA0 * p.sAm + kkA * p.sAk;
  let aBase1 = aBase0 + p.sAm;
  let aBase2 = aBase0 + 2u * p.sAm;
  let aBase3 = aBase0 + 3u * p.sAm;
  ${rpt === 5 ? "let aBase4 = aBase0 + 4u * p.sAm;" : ""}
  let fullA = rowA0 + ${rpt - 1}u < p.m;
  let taWrite = kkA * 17u + r4A;
  ${wordB ? `let kkB = (li % ${BK / 2}u) * 2u;
  let r4B = li / ${BK / 2}u;
  let rowB0 = n0 + r4B * 4u;
  let bRowBase = baseB + bRow(rowB0) * p.sBn;
  let sBnHalf = p.sBn / 2u;
  let bBase0 = (bRowBase + kkB) / 2u;
  let bBase1 = bBase0 + sBnHalf;
  let bBase2 = bBase0 + 2u * sBnHalf;
  let bBase3 = bBase0 + 3u * sBnHalf;
  let fullB = rowB0 + 3u < p.n;
  let tbWrite0 = kkB * 33u + r4B;
  let tbWrite1 = (kkB + 1u) * 33u + r4B;` : ""}
  for (var k0 = kStart; k0 < kEnd; k0 += ${BK}u) {
    let fullK = inBatch && (k0 + ${BK}u <= kEnd);
    // 🔴 EVERY PANEL WRITE IS A WHOLE vec4, one lane's: lanes writing different components of one shared vec4 race -
    // a component store can be a read-modify-write of the vector, and it was (every stage wrong from s_inputs on, 0.61).
    // A lane takes four consecutive rows at one k, so consecutive lanes still read consecutive k (coalesced).
    // A's panel: ${16 * BK} vec4s
    {
      var v = vec4<f32>(0.0);
      ${rpt === 5 ? "var v4 = 0.0;" : ""}
      if (fullK || (inBatch && k0 + kkA < kEnd)) {
        let aOff = k0 * p.sAk;
        if (fullA) {
          v = vec4<f32>(a[aBase0 + aOff], a[aBase1 + aOff], a[aBase2 + aOff], a[aBase3 + aOff]);
          ${rpt === 5 ? "v4 = a[aBase4 + aOff];" : ""}
        } else {
          if (rowA0 < p.m) { v.x = a[aBase0 + aOff]; }
          if (rowA0 + 1u < p.m) { v.y = a[aBase1 + aOff]; }
          if (rowA0 + 2u < p.m) { v.z = a[aBase2 + aOff]; }
          if (rowA0 + 3u < p.m) { v.w = a[aBase3 + aOff]; }
          ${rpt === 5 ? "if (rowA0 + 4u < p.m) { v4 = a[aBase4 + aOff]; }" : ""}
        }
      }
      ta[taWrite] = vec4<${E}>(v);
      ${rpt === 5 ? `ta4[taWrite] = ${E}(v4);` : ""}
    }
    // B's panel: ${32 * BK} vec4s${wordB ? ", each lane a packed pair of k for its four rows" : ""}
    ${wordB ? `{
      var lo = vec4<${EB}>(0.0);
      var hi = vec4<${EB}>(0.0);
      if (fullK) {
        let bOff = k0 / 2u;
        ${b16 ? `if (fullB) {
          let p0 = bitcast<vec2<f16>>(b[bBase0 + bOff]);
          let p1 = bitcast<vec2<f16>>(b[bBase1 + bOff]);
          let p2 = bitcast<vec2<f16>>(b[bBase2 + bOff]);
          let p3 = bitcast<vec2<f16>>(b[bBase3 + bOff]);
          lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x);
          hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y);
        } else {
          if (rowB0 < p.n) { let p0 = bitcast<vec2<f16>>(b[bBase0 + bOff]); lo.x = p0.x; hi.x = p0.y; }
          if (rowB0 + 1u < p.n) { let p1 = bitcast<vec2<f16>>(b[bBase1 + bOff]); lo.y = p1.x; hi.y = p1.y; }
          if (rowB0 + 2u < p.n) { let p2 = bitcast<vec2<f16>>(b[bBase2 + bOff]); lo.z = p2.x; hi.z = p2.y; }
          if (rowB0 + 3u < p.n) { let p3 = bitcast<vec2<f16>>(b[bBase3 + bOff]); lo.w = p3.x; hi.w = p3.y; }
        }` : `if (fullB) {
          let p0 = unpack2x16float(b[bBase0 + bOff]);
          let p1 = unpack2x16float(b[bBase1 + bOff]);
          let p2 = unpack2x16float(b[bBase2 + bOff]);
          let p3 = unpack2x16float(b[bBase3 + bOff]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
          hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y);
        } else {
          if (rowB0 < p.n) { let p0 = unpack2x16float(b[bBase0 + bOff]); lo.x = p0.x; hi.x = p0.y; }
          if (rowB0 + 1u < p.n) { let p1 = unpack2x16float(b[bBase1 + bOff]); lo.y = p1.x; hi.y = p1.y; }
          if (rowB0 + 2u < p.n) { let p2 = unpack2x16float(b[bBase2 + bOff]); lo.z = p2.x; hi.z = p2.y; }
          if (rowB0 + 3u < p.n) { let p3 = unpack2x16float(b[bBase3 + bOff]); lo.w = p3.x; hi.w = p3.y; }
        }`}
      } else if (inBatch && k0 + kkB < kEnd) {
        let hasHi = k0 + kkB + 1u < kEnd;
        let bOff = k0 / 2u;
        ${b16 ? `if (fullB) {
          let p0 = bitcast<vec2<f16>>(b[bBase0 + bOff]);
          let p1 = bitcast<vec2<f16>>(b[bBase1 + bOff]);
          let p2 = bitcast<vec2<f16>>(b[bBase2 + bOff]);
          let p3 = bitcast<vec2<f16>>(b[bBase3 + bOff]);
          lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x);
          if (hasHi) { hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (rowB0 < p.n) { let p0 = bitcast<vec2<f16>>(b[bBase0 + bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (rowB0 + 1u < p.n) { let p1 = bitcast<vec2<f16>>(b[bBase1 + bOff]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (rowB0 + 2u < p.n) { let p2 = bitcast<vec2<f16>>(b[bBase2 + bOff]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (rowB0 + 3u < p.n) { let p3 = bitcast<vec2<f16>>(b[bBase3 + bOff]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
        }` : `if (fullB) {
          let p0 = unpack2x16float(b[bBase0 + bOff]);
          let p1 = unpack2x16float(b[bBase1 + bOff]);
          let p2 = unpack2x16float(b[bBase2 + bOff]);
          let p3 = unpack2x16float(b[bBase3 + bOff]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
          if (hasHi) { hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (rowB0 < p.n) { let p0 = unpack2x16float(b[bBase0 + bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (rowB0 + 1u < p.n) { let p1 = unpack2x16float(b[bBase1 + bOff]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (rowB0 + 2u < p.n) { let p2 = unpack2x16float(b[bBase2 + bOff]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (rowB0 + 3u < p.n) { let p3 = unpack2x16float(b[bBase3 + bOff]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
        }`}
      }
      tb[tbWrite0] = lo;
      tb[tbWrite1] = hi;
    }` : `for (var e = 0u; e < ${32 * BK / 256}u; e += 1u) {
      let slot = li + e * 256u;
      let kk = slot % ${BK}u;
      let r4 = slot / ${BK}u;
      let k = k0 + kk;
      var v = vec4<f32>(0.0);
      if (inBatch && k < kEnd) {
        let row0 = n0 + r4 * 4u;
        let bAddr0 = baseB + bRow(row0) * p.sBn + k * p.sBk;
        if (row0 + 3u < p.n) {
          v = vec4<f32>(b[bAddr0], b[bAddr0 + p.sBn], b[bAddr0 + 2u * p.sBn], b[bAddr0 + 3u * p.sBn]);
        } else {
          if (row0 < p.n) { v.x = b[bAddr0]; }
          if (row0 + 1u < p.n) { v.y = b[bAddr0 + p.sBn]; }
          if (row0 + 2u < p.n) { v.z = b[bAddr0 + 2u * p.sBn]; }
          if (row0 + 3u < p.n) { v.w = b[bAddr0 + 3u * p.sBn]; }
        }
      }
      tb[kk * 33u + r4] = vec4<${EB}>(v);
    }`}
    workgroupBarrier();
    ${Array.from({ length: BK }, (_, kk) => `{
      let av = ta[${kk * 17}u + tm4];
      ${rpt === 5 ? `let av4 = ta4[${kk * 17}u + tm4];` : ""}
      let bv0 = vec4<f32>(tb[${kk * 33}u + tx]);
      let bv1 = vec4<f32>(tb[${kk * 33 + 16}u + tx]);
      acc_0_0 += av.x * bv0; acc_0_1 += av.x * bv1;
      acc_1_0 += av.y * bv0; acc_1_1 += av.y * bv1;
      acc_2_0 += av.z * bv0; acc_2_1 += av.z * bv1;
      acc_3_0 += av.w * bv0; acc_3_1 += av.w * bv1;
      ${rpt === 5 ? "acc_4_0 += av4 * bv0; acc_4_1 += av4 * bv1;" : ""}
    }`).join("\n    ")}
    workgroupBarrier();
  }
  if (!inBatch) { return; }
  ${pair ? `${rows.map((i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m && n0 + tn1 < p.n) { pairOut(m, (n0 / 2u) + tn0, acc_${i}_0 * p.alpha, acc_${i}_1 * p.alpha); }
  }`).join("\n  ")}
  return;` : ""}
  let baseC = p.offC + b0 * p.sC0 + b1 * p.sC1;
  ${rows.map((i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) {
      let r0 = acc_${i}_0 * p.alpha;
      let r1 = acc_${i}_1 * p.alpha;
      let rowC = baseC + m * p.sCm;
      ${gate ? `let gr = ${gate === "map" ? "gmap[m]" : "i32(m)"};
      if (gr >= 0) {
        let gBase = u32(gr) * p.gS + p.gOff;
        ${[0, 1, 2, 3, 4, 5, 6, 7].map((j) => {
          const nExpr = j < 4 ? `n0 + tn0 + ${j}u` : `n0 + tn1 + ${j - 4}u`;
          const valExpr = j < 4 ? `r0.${"xyzw"[j]}` : `r1.${"xyzw"[j - 4]}`;
          return `{
          let n = ${nExpr};
          if (n < p.n) {
            var v = ${valExpr};
            if (p.bias != 0u) { v += bias[p.biasOff + b0 * p.biasS0 + n]; }
            if (p.act == 1u) { v = max(v, 0.0); }
            if (p.act == 2u) { v = 1.0 / (1.0 + exp(-v)); }
            let at = rowC + n * p.sCn;
            c[at] = c[at] + gcond[gBase + n] * v;
          }
        }`;
        }).join("\n        ")}
      }` : `${[0, 1, 2, 3, 4, 5, 6, 7].map((j) => {
        const nExpr = j < 4 ? `n0 + tn0 + ${j}u` : `n0 + tn1 + ${j - 4}u`;
        const valExpr = j < 4 ? `r0.${"xyzw"[j]}` : `r1.${"xyzw"[j - 4]}`;
        return `{
        let n = ${nExpr};
        if (n < p.n) {
          let at = rowC + n * p.sCn;
          var v = ${valExpr};
          if (p.split == 0u) {
            if (p.bias != 0u) { v += bias[p.biasOff + b0 * p.biasS0 + n]; }
            if (p.act == 1u) { v = max(v, 0.0); }
            if (p.act == 2u) { v = 1.0 / (1.0 + exp(-v)); }
            if (p.accumulate != 0u) { v += c[at]; }
          }
          c[at] = v;
        }
      }`;
      }).join("\n      ")}`}
    }
  }`).join("\n  ")}
}`;
}

/**
 * y = LN(x) [* w] [+ b], then optionally modulated, (1 + alpha) y + beta, alpha/beta read from a COND buffer row
 * (rowMap[r], or r; a negative map entry means alpha = beta = 0) at column offsets. Biased variance, two passes,
 * as torch. One 64-lane workgroup a row.
 */
const LN_SOURCE = `
struct P { rows: u32, width: u32, xStride: u32, yStride: u32, xOff: u32, yOff: u32,
  hasW: u32, hasB: u32, modulate: u32, condStride: u32, alphaOff: u32, betaOff: u32, useMap: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bv: array<f32>;
@group(0) @binding(5) var<storage, read> cond: array<f32>;
@group(0) @binding(6) var<storage, read> rowMap: array<i32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let xb = p.xOff + row * p.xStride;
  var s = 0.0;
  for (var c = li; c < p.width; c += 64u) { s += x[xb + c]; }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  let mean = red[0] / f32(p.width);
  workgroupBarrier();
  var q = 0.0;
  for (var c = li; c < p.width; c += 64u) { let d = x[xb + c] - mean; q += d * d; }
  red[li] = q;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  let inv = inverseSqrt(red[0] / f32(p.width) + p.eps);
  if (!live) { return; }
  var crow = -1;
  if (p.modulate != 0u) { crow = select(i32(row), rowMap[row], p.useMap != 0u); }
  let cb = u32(max(crow, 0)) * p.condStride;
  let caBase = cb + p.alphaOff;
  let cbBase = cb + p.betaOff;
  let yb = p.yOff + row * p.yStride;
  for (var c = li; c < p.width; c += 64u) {
    var v = (x[xb + c] - mean) * inv;
    if (p.hasW != 0u) { v *= w[c]; }
    if (p.hasB != 0u) { v += bv[c]; }
    if (p.modulate != 0u) {
      var al = 0.0;
      var be = 0.0;
      if (crow >= 0) {
        al = cond[caBase + c];
        be = cond[cbBase + c];
      }
      v = v * (1.0 + al) + be;
    }
    y[yb + c] = v;
  }
}`;

/**
 * Softmax along rows of `len`, in place, after `+ bias` (an optional tensor addressed by strides) and a key mask:
 * a masked logit is REPLACED by -1e6, as the vendor's `where(mask, logits, -INF)` does. The row is
 * (b0, b1, r): logits at offL + b0 sL0 + b1 sL1 + r sLr + j; bias at offBias + b0 sB0 + b1 sB1 + r sBr + j;
 * mask[offM + b0 sM0 + j] != 0 is a live key.
 */
const SOFTMAX_SOURCE = `
struct P { rows: u32, len: u32, b1: u32, rowsPer: u32,
  offL: u32, sL0: u32, sL1: u32, sLr: u32,
  hasBias: u32, offBias: u32, sB0: u32, sB1: u32, sBr: u32,
  hasMask: u32, offM: u32, sM0: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> l: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<i32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = wid.x + wid.y * 65535u;
  let live = g < p.rows;
  let row = min(g, p.rows - 1u);
  let r = row % p.rowsPer;
  let bb = row / p.rowsPer;
  let b0 = bb / p.b1;
  let b1 = bb % p.b1;
  let lb = p.offL + b0 * p.sL0 + b1 * p.sL1 + r * p.sLr;
  let bias0 = p.offBias + b0 * p.sB0 + b1 * p.sB1 + r * p.sBr;
  let m0 = p.offM + b0 * p.sM0;
  var mx = -3.0e38;
  for (var j = li; j < p.len; j += 64u) {
    var v = l[lb + j] * p.scale;
    if (p.hasBias != 0u) { v += bias[bias0 + j]; }
    if (p.hasMask != 0u && mask[m0 + j] == 0) { v = -1.0e6; }
    mx = max(mx, v);
  }
  red[li] = mx;
  workgroupBarrier();
  if (li < 32u) { red[li] = max(red[li], red[li + 32u]); } workgroupBarrier();
  if (li < 16u) { red[li] = max(red[li], red[li + 16u]); } workgroupBarrier();
  if (li < 8u) { red[li] = max(red[li], red[li + 8u]); } workgroupBarrier();
  if (li < 4u) { red[li] = max(red[li], red[li + 4u]); } workgroupBarrier();
  if (li < 2u) { red[li] = max(red[li], red[li + 2u]); } workgroupBarrier();
  if (li == 0u) { red[0] = max(red[0], red[1]); } workgroupBarrier();
  let top = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var j = li; j < p.len; j += 64u) {
    var v = l[lb + j] * p.scale;
    if (p.hasBias != 0u) { v += bias[bias0 + j]; }
    if (p.hasMask != 0u && mask[m0 + j] == 0) { v = -1.0e6; }
    s += exp(v - top);
  }
  red[li] = s;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  let total = red[0];
  if (!live) { return; }
  for (var j = li; j < p.len; j += 64u) {
    var v = l[lb + j] * p.scale;
    if (p.hasBias != 0u) { v += bias[bias0 + j]; }
    if (p.hasMask != 0u && mask[m0 + j] == 0) { v = -1.0e6; }
    l[lb + j] = exp(v - top) / total;
  }
}`;

/**
 * A generic elementwise kernel over `count` items: `body` is WGSL that reads the uniform `p.u[k]` / `p.f[k]` and
 * the bindings `b0..b{n-1}` (f32 unless typed), with `i` the item index.
 */
function elementwiseSource(bindings, body) {
  const decls = bindings.map((b, at) => (b === null ? "" : `@group(0) @binding(${at + 1}) var<storage, `
    + `${b.write ? "read_write" : "read"}> b${at}: array<${b.type ?? "f32"}>;`)).join("\n");
  return `
struct P { count: u32, u: array<vec4<u32>, 4>, f: array<vec4<f32>, 4> };
@group(0) @binding(0) var<uniform> p: P;
${decls}
fn U(k: u32) -> u32 { return p.u[k / 4u][k % 4u]; }
fn F(k: u32) -> f32 { return p.f[k / 4u][k % 4u]; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.x + gid.y * nw.x * 256u;
  if (i >= p.count) { return; }
${body}
}`;
}

// --- the recorder ---------------------------------------------------------------------------------------------

export class Bioemu2Ops {
  constructor(device) {
    this.device = device;
    // gemm2Source by default; 1 is gemmSource, the control
    this.gemmVersion = 2;
    this.pipelines = pipelineCacheForDevice(device);
    this.compiled = new Map();
    this.dummy = device.createBuffer({ size: 16, usage: STORAGE });
    this.dummyInt = this.dummy;
    this.allocated = [];
    this.bytes = 0;
    // weights outlive a fold's activations: `destroy()` frees the activations, `destroyWeights()` both
    this.weightBuffers = [];
    this.weightBytes = 0;
  }

  async pipeline(key, source) {
    if (!this.compiled.has(key)) this.compiled.set(key, this.pipelines.get(`bioemu2:${key}`, source));
    return this.compiled.get(key);
  }

  /** A storage buffer of `floats` elements (4 bytes each), optionally filled. */
  buffer(label, floats, data) {
    const size = Math.max(16, Math.ceil(floats * 4 / 16) * 16);
    const buffer = this.device.createBuffer({ label: `bioemu2.${label}`, size, usage: STORAGE | COPY_SRC | COPY_DST });
    if (data !== undefined) this.write(buffer, data);
    this.allocated.push(buffer);
    this.bytes += size;
    buffer.floats = floats;
    return buffer;
  }

  write(buffer, data, offsetFloats = 0) {
    const view = data instanceof ArrayBuffer ? new Uint8Array(data)
      : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const padded = view.byteLength % 4 === 0 ? view : (() => {
      const out = new Uint8Array(Math.ceil(view.byteLength / 4) * 4); out.set(view); return out;
    })();
    this.device.queue.writeBuffer(buffer, offsetFloats * 4, padded);
  }

  /**
   * A weight: float32, or float16 two to a word (`half`) - packed ON THE DEVICE from a float32 upload, since a
   * host conversion of 512 M parameters is seconds of a first fold.
   */
  async weight(label, values, half = false) {
    const floats = values instanceof Float32Array ? values : Float32Array.from(values);
    if (!half) {
      const buffer = this.buffer(label, floats.length, floats);
      this.allocated.pop();
      this.bytes -= buffer.size;
      this.weightBuffers.push(buffer);
      this.weightBytes += buffer.size;
      return buffer;
    }
    const words = Math.ceil(floats.length / 2);
    const staging = this.device.createBuffer({ size: Math.max(16, Math.ceil(floats.length * 4 / 16) * 16),
                                               usage: STORAGE | COPY_DST });
    this.write(staging, floats);
    const buffer = this.buffer(label, words);
    this.allocated.pop();
    this.bytes -= Math.max(16, Math.ceil(words * 4 / 16) * 16);
    this.weightBuffers.push(buffer);
    this.weightBytes += buffer.size;
    const step = await this.map("pack-half", words, [{ buffer, type: "u32", write: true }, { buffer: staging }],
      "let n = U(0u); let a = b1[2u * i]; var b = 0.0; if (2u * i + 1u < n) { b = b1[2u * i + 1u]; }\n  b0[i] = pack2x16float(vec2<f32>(a, b));",
      [floats.length]);
    const uniform = this.allocated.pop();
    this.run(step);
    staging.destroy();
    uniform.destroy();
    buffer.half = true;
    return buffer;
  }

  /**
   * One or more contiguous weight tensors uploaded directly from their bundle sources, decoding quantised blocks
   * (int5/int8) on the device into f16 or f32 without expanding to Float32Array on the host.
   */
  async weightFromSources(label, sources, half, decode) {
    let offset = 0;
    const entries = sources.map((s, idx) => {
      const length = (s.record?.shape ?? []).reduce((a, b) => a * b, 1);
      const name = `${label}:${idx}`;
      const thunk = { store: { tensorSource: () => s }, tensorName: name, first: 0, count: length };
      const entry = { name, thunk, offset, length };
      offset += length;
      return entry;
    });
    const dest = half && offset % 2 === 0 ? "f16" : "f32";
    const planned = offset > 0 ? planBlockUpload(entries, dest) : undefined;
    if (planned && planned.host.length === 0 && planned.gpu.params.length > 0) {
      const words = dest === "f32" ? offset : Math.ceil(offset / 2);
      const buffer = this.buffer(label, words);
      this.allocated.pop();
      this.bytes -= buffer.size;
      this.weightBuffers.push(buffer);
      this.weightBytes += buffer.size;
      if (dest === "f16") buffer.half = true;
      const release = await runBlockUpload(this.device, planned.gpu, buffer);
      void this.device.queue.onSubmittedWorkDone().then(release);
      return buffer;
    }
    const parts = sources.map((s) => decode(s));
    if (parts.length === 1) return this.weight(label, parts[0], half && parts[0].length % 2 === 0);
    const all = new Float32Array(offset);
    let at = 0;
    for (const v of parts) { all.set(v, at); at += v.length; }
    return this.weight(label, all, half && all.length % 2 === 0);
  }

  /** A slice of `buffer`, `count` floats from float `at`, bound where a buffer would be (the offset 256-byte aligned). */
  view(buffer, at, count) {
    if ((at * 4) % 256) throw new Error(`a view at float ${at} is not 256-byte aligned`);
    return { view: true, buffer, offset: at * 4, size: count * 4 };
  }

  uniform(words) {
    const size = Math.ceil(words.byteLength / 16) * 16;
    const buffer = this.device.createBuffer({ size: Math.max(size, 16), usage: UNIFORM | COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, words.buffer, words.byteOffset, words.byteLength);
    this.allocated.push(buffer);
    return buffer;
  }

  async step(key, source, groups, buffers, uniformWords) {
    const pipeline = await this.pipeline(key, source);
    // (a buffer, or a VIEW of one - ops.view: a slice bound at its offset, so a kernel indexes it from 0)
    const entries = [{ binding: 0, resource: { buffer: this.uniform(uniformWords) } },
      ...buffers.map((b, at) => ({ binding: at + 1, resource: b?.view ? { buffer: b.buffer, offset: b.offset, size: b.size } : { buffer: b } }))
        .filter((e) => e.resource.buffer !== null && e.resource.buffer !== undefined)];
    const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const [x, y] = fold2(groups);
    return { pipeline, bindGroup, x, y, key };
  }

  /**
   * C = alpha A B^T + bias, batched and strided; see gemmSource. `spec` = {m, n, k, a, b, c, bias?, act?,
   * accumulate?, alpha?, batches?: [B0, B1], A: {off, s0, s1, m, k}, B: {off, s0, s1, n, k}, C: {off, s0, s1, m, n}}.
   */
  async gemm(spec) {
    const { m, n, k } = spec;
    const [B0, B1] = spec.batches ?? [1, 1];
    const A = { off: 0, s0: 0, s1: 0, m: k, k: 1, ...spec.A };
    const B = { off: 0, s0: 0, s1: 0, n: k, k: 1, ...spec.B };
    const C = { off: 0, s0: 0, s1: 0, m: n, n: 1, ...spec.C };
    const word = spec.b.half === true;
    if (word && (B.k !== 1 || B.off % 2 || B.s0 % 2 || B.s1 % 2 || B.n % 2)) throw new Error("a half weight wants even offsets");
    // 🔴 A FEW ROWS AND A DEEP K LEAVE THE DEVICE IDLE: the token stream's 71-row projections launched 24
    // workgroups over K = 3072 and ran at 0.4 TFLOP/s. Such a GEMM is SPLIT over K into partial planes and
    // summed by a second kernel, which takes the epilogue (bias, activation, accumulation).
    const v2 = this.gemmVersion === 2;
    const tiles = Math.ceil(m / 64) * Math.ceil(n / (v2 ? 128 : 64)) * B0 * B1;
    // (the line is in WORK, not tiles: gemm2's 64 x 128 tile is twice gemmSource's, and counting it as one split twice as
    // many GEMMs - the reduce went 16 -> 32 ms of a denoiser call, eating the faster kernel's gain)
    const work = v2 ? 2 : 1;
    if (spec.pair && !v2) throw new Error("a pair epilogue is gemm2Source's");
    // spec.gate = {cond, stride, off, map?}: c[r][n] += cond[map(r) stride + off + n] (alpha A B^T)[r][n] - an adaLN-gated
    // residual (bioemu2's gatedAdd) in the epilogue; a negative map entry adds nothing. gemmSource's: the two passes
    if (spec.gate && (!v2 || B0 * B1 !== 1)) {
      const g = spec.gate;
      const tmp = this.gateScratch?.n >= m * n ? this.gateScratch : (this.gateScratch = { buf: this.buffer("gemm.gate", m * n), n: m * n });
      const prod = await this.gemm({ ...spec, gate: undefined, c: tmp.buf, accumulate: false, C: { off: 0, m: n, n: 1 } });
      const add = await this.map(g.map ? "gate-add-map" : "gate-add", m * n,
        [{ buffer: spec.c, write: true }, { buffer: tmp.buf }, { buffer: g.cond }, ...(g.map ? [{ buffer: g.map, type: "i32" }] : [])],
        `let r = i / U(0u); let col = i % U(0u); let at = U(3u) + r * U(4u) + col * U(5u);
  ${g.map ? "let gr = b3[r];" : "let gr = i32(r);"}
  if (gr >= 0) { b0[at] = b0[at] + b2[u32(gr) * U(1u) + U(2u) + col] * b1[i]; }`, [n, g.stride, g.off, C.off, C.m, C.n]);
      return this.program([prod, add]);
    }
    // 🔴 THE TARGET IS 128 WORK UNITS, NOT 1024: swept on an M5 (ABAB, the shipped gemm2 and the fused combine), a 1-sample
    // call is 107 ms at 1024, 101 at 256 and 128, 106 with no split at all, and a 5-sample call 258 / 248 / 249 / 249 - the
    // partial planes' traffic was most of what a deeper split bought. One part is no split (no reduce).
    if (spec.noSplit !== true && !spec.pair && !spec.biasOff && this.splits(m, n, k, B0 * B1) >= 2) {
      const splits = this.splits(m, n, k, 1);
      const chunk = Math.ceil(k / splits / 32) * 32;      // (a multiple of either kernel's K step)
      const parts = Math.ceil(k / chunk);
      // ...into ONE scratch per size class, shared by every split GEMM recorded: the dispatches run in order, so
      // no two of them hold their partials at once, and one each was 1.6 GiB of a five-sample 68-residue fold
      const floats = 2 ** Math.ceil(Math.log2(parts * m * n));
      this.partials ??= new Map();
      if (!this.partials.has(floats)) this.partials.set(floats, this.buffer("gemm.partial", floats));
      const partial = this.partials.get(floats);
      const first = await this.gemm({ ...spec, noSplit: true, c: partial, bias: undefined, act: undefined, accumulate: false, gate: undefined,
                                      batches: [1, parts], splitChunk: chunk,
                                      A: { ...A, s1: 0 }, B: { ...B, s1: 0 }, C: { off: 0, s0: 0, s1: m * n, m: n, n: 1 } });
      const g = spec.gate;
      const reduce = await this.map(g ? (g.map ? "gemm-reduce-gm" : "gemm-reduce-g") : "gemm-reduce", m * n, [{ buffer: spec.c, write: true }, { buffer: partial },
        { buffer: spec.bias ?? this.dummy }, ...(g ? [{ buffer: g.cond }, ...(g.map ? [{ buffer: g.map, type: "i32" }] : [])] : [])],
      `let n = U(1u); let row = i / n; let col = i % n; let stride = U(0u);
  var v = b1[i];
  if (U(2u) == 2u) {
    v += b1[stride + i];
  } else {
    for (var s = 1u; s < U(2u); s += 1u) { v += b1[s * stride + i]; }
  }
  if (U(3u) != 0u) { v += b2[col]; }
  if (U(4u) == 1u) { v = max(v, 0.0); }
  if (U(4u) == 2u) { v = 1.0 / (1.0 + exp(-v)); }
  let at = U(6u) + row * U(7u) + col * U(8u);
  ${g ? `let gr = ${g.map ? "b4[row]" : "i32(row)"};
  if (gr >= 0) { b0[at] = b0[at] + b3[u32(gr) * U(9u) + U(10u) + col] * v; }` : `if (U(5u) != 0u) { v += b0[at]; }
  b0[at] = v;`}`, [m * n, n, parts, spec.bias ? 1 : 0, { relu: 1, sigmoid: 2 }[spec.act] ?? 0, spec.accumulate ? 1 : 0,
        C.off, C.m, C.n, g?.stride ?? 0, g?.off ?? 0]);
      first.label = `gemm ${m}x${n}x${k} split ${parts}`;
      return this.program([first, reduce]);
    }
    const u = new Uint32Array(32);
    const f = new Float32Array(u.buffer);
    u.set([m, n, k, B1, A.off, A.s0, A.s1, A.m, A.k, B.off, B.s0, B.s1, B.n, B.k, C.off, C.s0, C.s1, C.m, C.n,
      spec.bias ? 1 : 0, { relu: 1, sigmoid: 2 }[spec.act] ?? 0, spec.accumulate ? 1 : 0]);
    f[22] = spec.alpha ?? 1;
    u[23] = B0 * B1;
    u[24] = spec.splitChunk ?? 0;
    // (gemm2Source only: the bias of batch b0 at biasOff + b0 biasS0 - one bias a batch, a stacked layer's)
    if ((spec.biasOff || spec.biasS0) && !v2) throw new Error("a per-batch bias is gemm2Source's");
    u[25] = spec.biasOff ?? 0;
    u[26] = spec.biasS0 ?? 0;
    u[27] = spec.gate?.stride ?? 0;
    u[28] = spec.gate?.off ?? 0;
    const rpt = v2 && m > 64 && m <= 80 ? 5 : 4;
    const bm = rpt * 16;
    const groups = Math.ceil(m / bm) * Math.ceil(n / (v2 ? 128 : 64)) * B0 * B1;
    // spec.pair = {key, half, decls, body, buffers}: the SwiGLU-pair epilogue (gemm2Source), its buffers at binding 5 on
    // (a half weight's panel staged in f16 where the device has shader-f16: b16Panels false is the control)
    const b16 = v2 && word && this.b16Panels !== false && this.device.features?.has?.("shader-f16") === true;
    const gate = spec.gate ? (spec.gate.map ? "map" : "plain") : null;
    const kind = `${v2 ? "gemm2" : "gemm"}${rpt === 5 ? "-m80" : ""}${word ? "-h" : ""}${b16 ? "-b16" : ""}${spec.pair ? `-pair-${spec.pair.key}` : ""}${gate ? `-g${gate}` : ""}`;
    const step = await this.step(kind, v2 ? gemm2Source(word, spec.pair, b16, gate, rpt) : gemmSource(word), groups,
                                 [spec.a, spec.b, spec.c, spec.bias ?? this.dummy, ...(spec.pair?.buffers ?? []),
                                  ...(spec.gate ? [spec.gate.cond, ...(spec.gate.map ? [spec.gate.map] : [])] : [])], u);
    step.label = spec.label ?? `gemm ${m}x${n}x${k}${B0 * B1 > 1 ? ` [${B0}x${B1}]` : ""}`;
    return step;
  }

  /** How many K parts gemm() splits an (m, n, k) GEMM into; under 2 it is not split. */
  splits(m, n, k, batches = 1) {
    const v2 = this.gemmVersion === 2;
    const tiles = Math.ceil(m / 64) * Math.ceil(n / (v2 ? 128 : 64)) * batches;
    const work = v2 ? 2 : 1;
    const target = this.splitTarget ?? 80;
    if (batches !== 1 || tiles * work >= target || k < 512) return 1;
    return Math.min(Math.floor(k / 128), Math.ceil(target / (tiles * work)));
  }

  /**
   * y = silu(x fc1^T) * (x fc2^T) over `rows` rows, w = [fc1; fc2] stacked ([2 hidden][inC]): one GEMM whose epilogue
   * is the SwiGLU (gemm2Source's pair mode), the [rows][2 hidden] pre-activations never stored. Null where that GEMM
   * would be split over K (a pair GEMM is not) or the kernel is not gemm2Source: the caller takes the two-pass path.
   */
  async swigluLinear({ x, w, y, rows, inC, hidden }) {
    if (this.fuseSwiglu === false || this.gemmVersion !== 2 || hidden % 4 || this.splits(rows, 2 * hidden, inC) >= 2) return null;
    return this.gemm({ m: rows, n: 2 * hidden, k: inC, a: x, b: w, c: y,
      A: { off: 0, m: inC, k: 1 }, B: { off: 0, n: inC, k: 1 }, C: { off: 0, m: hidden, n: 1 },
      pair: { key: `swiglu${hidden}`, half: hidden, buffers: [], decls: "", body: `fn pairOut(r: u32, j0: u32, g: vec4<f32>, v: vec4<f32>) {
  let y = g / (vec4<f32>(1.0) + exp(-g)) * v;
  let base = r * ${hidden}u + j0;
  c[base] = y.x; c[base + 1u] = y.y; c[base + 2u] = y.z; c[base + 3u] = y.w;
}` } });
  }

  /**
   * The atom stream's WINDOWED attention fused: out[r][h hd + d] = g[r][h hd + d] * sum_j softmax_j(q.k_j alpha +
   * bias[w][h][qi][j], masked keys -1e6) v_j[d] - what the scores GEMM, the softmax kernel, the P.V GEMM and the gating
   * pass computed, in one dispatch a (sample, window, head). 128 lanes, four a query, each holding its query's row; the
   * 128 keys in chunks of 32 with an online softmax. Workgroup memory 16 KiB exactly (q, a chunk's k and v, its
   * logits) - WebGPU's guaranteed floor. The softmax sums in a different order from the three-pass path: ~1e-7.
   * qkv [S N][3 D] or [S N][4 D] (when g is omitted and fused in qkv), kvPad [S pad][2 D] (k | v, CTX zero rows each end).
   */
  windowAttention({ qkv, qBias, kvPad, bias, mask, g, out, S, N, H, pad, alpha, ctx = 48 }) {
    const WQ = 32, WK = 128, HD = 32, D = H * HD, NW = N / WQ;
    if (N % WQ) throw new Error("windowAttention: N is not whole windows");
    const fusedG = !g;
    const directKv = !kvPad;
    const hasQBias = Boolean(qBias);
    const qkvStride = fusedG ? 4 * D : 3 * D;
    const u = new Uint32Array(8);
    const f = new Float32Array(u.buffer);
    u.set([S, N, NW, pad]);
    f[4] = alpha;
    let nextBind = 2;
    const kvBind = directKv ? "" : `@group(0) @binding(${nextBind++}) var<storage, read> kv: array<f32>;`;
    const biasBindIdx = nextBind++;
    const maskBindIdx = nextBind++;
    const gBind = fusedG ? "" : `@group(0) @binding(${nextBind++}) var<storage, read> gate: array<f32>;`;
    const qbBind = hasQBias ? `@group(0) @binding(${nextBind++}) var<storage, read> qBias: array<f32>;` : "";
    const outBindIdx = nextBind++;
    const buffers = [qkv, ...(directKv ? [] : [kvPad]), bias, mask, ...(fusedG ? [] : [g]), ...(hasQBias ? [qBias] : []), out];
    return this.step(`win-attn-${H}${directKv ? "-dkv" : ""}${fusedG ? "-fg" : ""}${hasQBias ? "-qb" : ""}`, `
struct P { S: u32, N: u32, NW: u32, pad: u32, alpha: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
${kvBind}
@group(0) @binding(${biasBindIdx}) var<storage, read> bias: array<f32>;
@group(0) @binding(${maskBindIdx}) var<storage, read> mask: array<i32>;
${gBind}
${qbBind}
@group(0) @binding(${outBindIdx}) var<storage, read_write> out: array<f32>;
var<workgroup> ks: array<vec4<f32>, 256>;     // [32 keys][32 dims]
var<workgroup> vs: array<vec4<f32>, 256>;
var<workgroup> ps: array<f32, 1024>;          // [32 queries][32 keys] this chunk's logits
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wid.x + wid.y * 65535u;
  let h = flat % ${H}u;
  let w = (flat / ${H}u) % p.NW;
  let sm = flat / (${H}u * p.NW);
  let live = sm < p.S;
  let smc = min(sm, p.S - 1u);
  let qi = li / 4u;
  let sub = li % 4u;
  let qrow = smc * p.N + w * ${WQ}u + qi;
  let qBase = qrow * ${qkvStride}u + h * ${HD}u;
  ${hasQBias ? `let qbBase = h * ${HD}u;` : ""}
  ${[0, 1, 2, 3, 4, 5, 6, 7].map((d) => hasQBias
    ? `let q_${d} = vec4<f32>(qkv[qBase + ${d * 4}u] + qBias[qbBase + ${d * 4}u], qkv[qBase + ${d * 4 + 1}u] + qBias[qbBase + ${d * 4 + 1}u], qkv[qBase + ${d * 4 + 2}u] + qBias[qbBase + ${d * 4 + 2}u], qkv[qBase + ${d * 4 + 3}u] + qBias[qbBase + ${d * 4 + 3}u]);`
    : `let q_${d} = vec4<f32>(qkv[qBase + ${d * 4}u], qkv[qBase + ${d * 4 + 1}u], qkv[qBase + ${d * 4 + 2}u], qkv[qBase + ${d * 4 + 3}u]);`).join("\n  ")}
  var m = -3.0e38;
  var l = 0.0;
  var o0 = vec4<f32>(0.0);
  var o1 = vec4<f32>(0.0);
  let biasRow = ((w * ${H}u + h) * ${WQ}u + qi) * ${WK}u;
  for (var c = 0u; c < ${WK / 32}u; c += 1u) {
    // the chunk's k and v: 32 keys x 32 dims each, a lane two vec4s of each
    for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 128u;
      let key = slot / 8u;
      let d4 = slot % 8u;
      ${directKv ? `let kPad = w * ${WQ}u + c * 32u + key;
      if (kPad >= ${ctx}u && kPad < p.N + ${ctx}u) {
        let row = (smc * p.N + kPad - ${ctx}u) * ${qkvStride}u + ${D}u + h * ${HD}u + d4 * 4u;
        ks[slot] = vec4<f32>(qkv[row], qkv[row + 1u], qkv[row + 2u], qkv[row + 3u]);
        vs[slot] = vec4<f32>(qkv[row + ${D}u], qkv[row + ${D + 1}u], qkv[row + ${D + 2}u], qkv[row + ${D + 3}u]);
      } else {
        ks[slot] = vec4<f32>(0.0);
        vs[slot] = vec4<f32>(0.0);
      }` : `let row = (smc * p.pad + w * ${WQ}u + c * 32u + key) * ${2 * D}u + h * ${HD}u + d4 * 4u;
      ks[slot] = vec4<f32>(kv[row], kv[row + 1u], kv[row + 2u], kv[row + 3u]);
      vs[slot] = vec4<f32>(kv[row + ${D}u], kv[row + ${D + 1}u], kv[row + ${D + 2}u], kv[row + ${D + 3}u]);`}
    }
    workgroupBarrier();
    // this lane's 8 keys of the chunk
    for (var jj = 0u; jj < 8u; jj += 1u) {
      let j = sub * 8u + jj;
      let jb = j * 8u;
      let qk = ${[0, 1, 2, 3, 4, 5, 6, 7].map((d) => `dot(q_${d}, ks[jb + ${d}u])`).join(" + ")};
      let key = c * 32u + j;
      var v = qk * p.alpha + bias[biasRow + key];
      if (mask[w * ${WQ}u + key] == 0) { v = -1.0e6; }
      ps[qi * 32u + j] = v;
    }
    workgroupBarrier();
    // the online softmax over the chunk's 32 (every lane of a query alike), and this lane's 8 output dims
    var cm = m;
    let pBase = qi * 32u;
    for (var j = 0u; j < 32u; j += 4u) {
      cm = max(max(cm, ps[pBase + j]), max(ps[pBase + j + 1u], max(ps[pBase + j + 2u], ps[pBase + j + 3u])));
    }
    let rescale = exp(m - cm);
    l = l * rescale;
    o0 = o0 * rescale;
    o1 = o1 * rescale;
    for (var j = 0u; j < 32u; j += 2u) {
      let e0 = exp(ps[pBase + j] - cm);
      let e1 = exp(ps[pBase + j + 1u] - cm);
      l += e0 + e1;
      let vb0 = j * 8u + sub * 2u;
      let vb1 = vb0 + 8u;
      o0 += e0 * vs[vb0] + e1 * vs[vb1];
      o1 += e0 * vs[vb0 + 1u] + e1 * vs[vb1 + 1u];
    }
    m = cm;
    workgroupBarrier();
  }
  if (!live) { return; }
  let ob = qrow * ${D}u + h * ${HD}u + sub * 8u;
  let invL = 1.0 / l;
  ${fusedG ? `let gb = qBase + ${3 * D}u + sub * 8u;
  out[ob] = o0.x * invL / (1.0 + exp(-qkv[gb]));
  out[ob + 1u] = o0.y * invL / (1.0 + exp(-qkv[gb + 1u]));
  out[ob + 2u] = o0.z * invL / (1.0 + exp(-qkv[gb + 2u]));
  out[ob + 3u] = o0.w * invL / (1.0 + exp(-qkv[gb + 3u]));
  out[ob + 4u] = o1.x * invL / (1.0 + exp(-qkv[gb + 4u]));
  out[ob + 5u] = o1.y * invL / (1.0 + exp(-qkv[gb + 5u]));
  out[ob + 6u] = o1.z * invL / (1.0 + exp(-qkv[gb + 6u]));
  out[ob + 7u] = o1.w * invL / (1.0 + exp(-qkv[gb + 7u]));` : `out[ob] = o0.x * invL * gate[ob];
  out[ob + 1u] = o0.y * invL * gate[ob + 1u];
  out[ob + 2u] = o0.z * invL * gate[ob + 2u];
  out[ob + 3u] = o0.w * invL * gate[ob + 3u];
  out[ob + 4u] = o1.x * invL * gate[ob + 4u];
  out[ob + 5u] = o1.y * invL * gate[ob + 5u];
  out[ob + 6u] = o1.z * invL * gate[ob + 6u];
  out[ob + 7u] = o1.w * invL * gate[ob + 7u];`}
}`, S * NW * H, buffers, u);
  }

  /**
   * The token stream's attention fused, windowAttention's shape over the whole stream: out[r][h hd + d] = g[r][h hd + d]
   * sum_j softmax_j(q.k_j alpha + bias[h][qi][j]) v_j[d] for one sample's T tokens - the scores GEMM, the softmax, P.V
   * and the gating in one dispatch a (sample, head, block of 32 queries). The keys in chunks of 32 (past T: none), an
   * online softmax, 16 KiB of workgroup memory at a head of 48. qkv [S T][3 D] or [S T][4 D], bias [H][T][T].
   */
  tokenAttention({ qkv, qBias, bias, g, out, S, T, H, hd, alpha }) {
    if (hd % 4) throw new Error("tokenAttention: a head width that is not whole vec4s");
    const D = H * hd, V = hd / 4, VL = V / 4, QB = Math.ceil(T / 32);
    if (V % 4) throw new Error("tokenAttention: a head width that is not four lanes of whole vec4s");
    const fusedG = !g;
    const hasQBias = Boolean(qBias);
    const qkvStride = fusedG ? 4 * D : 3 * D;
    const u = new Uint32Array(8);
    const f = new Float32Array(u.buffer);
    u.set([S, T, QB]);
    f[4] = alpha;
    const vIdx = Array.from({ length: V }, (_, i) => i);
    const vlIdx = Array.from({ length: VL }, (_, i) => i);
    const gBind = fusedG ? "" : "@group(0) @binding(3) var<storage, read> gate: array<f32>;";
    const outBindIdx = fusedG ? 3 : 4;
    const qbBindIdx = fusedG ? 4 : 5;
    const buffers = [qkv, bias, ...(fusedG ? [] : [g]), out, ...(hasQBias ? [qBias] : [])];
    return this.step(`tok-attn-${H}x${hd}${fusedG ? "-fg" : ""}${hasQBias ? "-qb" : ""}`, `
struct P { S: u32, T: u32, QB: u32, pad0: u32, alpha: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
${gBind}
@group(0) @binding(${outBindIdx}) var<storage, read_write> out: array<f32>;
${hasQBias ? `@group(0) @binding(${qbBindIdx}) var<storage, read> qBias: array<f32>;` : ""}
var<workgroup> ks: array<vec4<f32>, ${32 * V}>;     // [32 keys][hd]
var<workgroup> vs: array<vec4<f32>, ${32 * V}>;
var<workgroup> ps: array<f32, 1024>;              // [32 queries][32 keys]
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wid.x + wid.y * 65535u;
  let qb = flat % p.QB;
  let h = (flat / p.QB) % ${H}u;
  let sm = flat / (p.QB * ${H}u);
  let smc = min(sm, p.S - 1u);
  let ql = li / 4u;
  let sub = li % 4u;
  let qi = qb * 32u + ql;
  let live = sm < p.S && qi < p.T;
  let qic = min(qi, p.T - 1u);
  let qrow = smc * p.T + qic;
  let qBase = qrow * ${qkvStride}u + h * ${hd}u;
  ${hasQBias ? `let qbBase = h * ${hd}u;` : ""}
  ${vIdx.map((d) => hasQBias
    ? `let q_${d} = vec4<f32>(qkv[qBase + ${d * 4}u] + qBias[qbBase + ${d * 4}u], qkv[qBase + ${d * 4 + 1}u] + qBias[qbBase + ${d * 4 + 1}u], qkv[qBase + ${d * 4 + 2}u] + qBias[qbBase + ${d * 4 + 2}u], qkv[qBase + ${d * 4 + 3}u] + qBias[qbBase + ${d * 4 + 3}u]);`
    : `let q_${d} = vec4<f32>(qkv[qBase + ${d * 4}u], qkv[qBase + ${d * 4 + 1}u], qkv[qBase + ${d * 4 + 2}u], qkv[qBase + ${d * 4 + 3}u]);`).join("\n  ")}
  var m = -3.0e38;
  var l = 0.0;
  ${vlIdx.map((e) => `var o_${e} = vec4<f32>(0.0);`).join("\n  ")}
  let biasRow = (h * p.T + qic) * p.T;
  let chunks = (p.T + 31u) / 32u;
  for (var c = 0u; c < chunks; c += 1u) {
    for (var e = 0u; e < ${Math.ceil(32 * V / 128)}u; e += 1u) {
      let slot = li + e * 128u;
      if (slot < ${32 * V}u) {
        let key = slot / ${V}u;
        let d4 = slot % ${V}u;
        let kt = c * 32u + key;
        var kv = vec4<f32>(0.0);
        var vv = vec4<f32>(0.0);
        if (kt < p.T) {
          let row = (smc * p.T + kt) * ${qkvStride}u + h * ${hd}u + d4 * 4u;
          kv = vec4<f32>(qkv[row + ${D}u], qkv[row + ${D + 1}u], qkv[row + ${D + 2}u], qkv[row + ${D + 3}u]);
          vv = vec4<f32>(qkv[row + ${2 * D}u], qkv[row + ${2 * D + 1}u], qkv[row + ${2 * D + 2}u], qkv[row + ${2 * D + 3}u]);
        }
        ks[slot] = kv;
        vs[slot] = vv;
      }
    }
    workgroupBarrier();
    for (var jj = 0u; jj < 8u; jj += 1u) {
      let j = sub * 8u + jj;
      let kt = c * 32u + j;
      var v = -3.0e38;
      if (kt < p.T) {
        let jb = j * ${V}u;
        let qk = ${vIdx.map((d) => `dot(q_${d}, ks[jb + ${d}u])`).join(" + ")};
        v = qk * p.alpha + bias[biasRow + kt];
      }
      ps[ql * 32u + j] = v;
    }
    workgroupBarrier();
    var cm = m;
    let pBase = ql * 32u;
    for (var j = 0u; j < 32u; j += 4u) {
      cm = max(max(cm, ps[pBase + j]), max(ps[pBase + j + 1u], max(ps[pBase + j + 2u], ps[pBase + j + 3u])));
    }
    let rescale = exp(m - cm);
    l = l * rescale;
    ${vlIdx.map((e) => `o_${e} = o_${e} * rescale;`).join("\n    ")}
    for (var j = 0u; j < 32u; j += 1u) {
      let w = exp(ps[pBase + j] - cm);
      l += w;
      let vb = j * ${V}u + sub * ${VL}u;
      ${vlIdx.map((e) => `o_${e} += w * vs[vb + ${e}u];`).join("\n      ")}
    }
    m = cm;
    workgroupBarrier();
  }
  if (!live) { return; }
  let ob = qrow * ${D}u + h * ${hd}u + sub * ${4 * VL}u;
  let invL = 1.0 / l;
  ${fusedG ? `let gb = qBase + ${3 * D}u + sub * ${4 * VL}u;
  ${vlIdx.map((e) => `out[ob + ${e * 4}u] = o_${e}.x * invL / (1.0 + exp(-qkv[gb + ${e * 4}u]));
  out[ob + ${e * 4 + 1}u] = o_${e}.y * invL / (1.0 + exp(-qkv[gb + ${e * 4 + 1}u]));
  out[ob + ${e * 4 + 2}u] = o_${e}.z * invL / (1.0 + exp(-qkv[gb + ${e * 4 + 2}u]));
  out[ob + ${e * 4 + 3}u] = o_${e}.w * invL / (1.0 + exp(-qkv[gb + ${e * 4 + 3}u]));`).join("\n  ")}` : vlIdx.map((e) => `out[ob + ${e * 4}u] = o_${e}.x * invL * gate[ob + ${e * 4}u];
  out[ob + ${e * 4 + 1}u] = o_${e}.y * invL * gate[ob + ${e * 4 + 1}u];
  out[ob + ${e * 4 + 2}u] = o_${e}.z * invL * gate[ob + ${e * 4 + 2}u];
  out[ob + ${e * 4 + 3}u] = o_${e}.w * invL * gate[ob + ${e * 4 + 3}u];`).join("\n  ")}
}`, S * H * QB, buffers, u);
  }

  /** Fused couplingSplit token row statistics + Xa scaling (one 64-lane workgroup per token row). */
  csTokenPrep({ a, gA, tstats, xa, rows, T }) {
    const u = new Uint32Array([rows, T, 0, 0]);
    return this.step("cs-tprep", `
struct P { rows: u32, T: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> gA: array<f32>;
@group(0) @binding(3) var<storage, read_write> tstats: array<f32>;
@group(0) @binding(4) var<storage, read_write> xa: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let base = row * 768u;
  ${Array.from({ length: 12 }, (_, k) => `let v_${k} = a[base + li + ${k * 64}u];`).join("\n  ")}
  red[li] = ${Array.from({ length: 12 }, (_, k) => `v_${k}`).join(" + ")};
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  let m = red[0] / 768.0;
  workgroupBarrier();
  ${Array.from({ length: 12 }, (_, k) => `let d_${k} = v_${k} - m;`).join("\n  ")}
  red[li] = ${Array.from({ length: 12 }, (_, k) => `d_${k} * d_${k}`).join(" + ")};
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  if (!live) { return; }
  let v = red[0];
  let sa = sqrt(v / 768.0 + 1e-5);
  if (li == 0u) {
    tstats[row * 3u] = m;
    tstats[row * 3u + 1u] = v;
    tstats[row * 3u + 2u] = sa;
  }
  let gBase = (row % p.T) * 768u + li;
  let invSa = 1.0 / sa;
  ${Array.from({ length: 12 }, (_, k) => `xa[base + li + ${k * 64}u] = gA[gBase + ${k * 64}u] * d_${k} * invSa;`).join("\n  ")}
}`, rows, [a, gA, tstats, xa], u);
  }

  /** Fused cat-qa + modulated LayerNorm(896) for q_to_s_mixer (one 64-lane workgroup per token row). */
  catQaLn({ q, a, centre, condS, qsh, rows, T, N, eps = 1e-5 }) {
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u[0] = rows; u[1] = T; u[2] = N; f[3] = eps;
    return this.step("cat-qa-ln", `
struct P { rows: u32, T: u32, N: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<f32>;
@group(0) @binding(3) var<storage, read> centre: array<i32>;
@group(0) @binding(4) var<storage, read> cond: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let sm = row / p.T;
  let t = row % p.T;
  let qBase = (sm * p.N + u32(centre[t])) * 128u + li;
  let aBase = row * 768u + li;
  let v_0 = q[qBase];
  let v_1 = q[qBase + 64u];
  ${Array.from({ length: 12 }, (_, k) => `let v_${k + 2} = a[aBase + ${k * 64}u];`).join("\n  ")}
  red[li] = ${Array.from({ length: 14 }, (_, k) => `v_${k}`).join(" + ")};
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  let mean = red[0] / 896.0;
  workgroupBarrier();
  ${Array.from({ length: 14 }, (_, k) => `let d_${k} = v_${k} - mean;`).join("\n  ")}
  red[li] = ${Array.from({ length: 14 }, (_, k) => `d_${k} * d_${k}`).join(" + ")};
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  if (!live) { return; }
  let inv = inverseSqrt(red[0] / 896.0 + p.eps);
  let cb = t * 2560u + li;
  let yb = row * 896u + li;
  ${Array.from({ length: 14 }, (_, k) => `y[yb + ${k * 64}u] = d_${k} * inv * (1.0 + cond[cb + ${k * 64}u]) + cond[cb + ${896 + k * 64}u];`).join("\n  ")}
}`, rows, [q, a, centre, condS, qsh], u);
  }

  /** Fused couplingSplit atom row statistics + Xq scaling (one 64-lane workgroup per atom row). */
  csAtomPrep({ q, gQ, tstats, atomTokenS, astats, xq, rows, T }) {
    const u = new Uint32Array([rows, T, 0, 0]);
    return this.step("cs-aprep", `
struct P { rows: u32, T: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> gQ: array<f32>;
@group(0) @binding(3) var<storage, read> tstats: array<f32>;
@group(0) @binding(4) var<storage, read> atomTokenS: array<i32>;
@group(0) @binding(5) var<storage, read_write> astats: array<f32>;
@group(0) @binding(6) var<storage, read_write> xq: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let base = row * 128u;
  let q0 = q[base + li];
  let q1 = q[base + li + 64u];
  red[li] = q0 + q1;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  let mq = red[0] / 128.0;
  workgroupBarrier();
  let d0 = q0 - mq;
  let d1 = q1 - mq;
  red[li] = d0 * d0 + d1 * d1;
  workgroupBarrier();
  if (li < 32u) { red[li] += red[li + 32u]; } workgroupBarrier();
  if (li < 16u) { red[li] += red[li + 16u]; } workgroupBarrier();
  if (li < 8u) { red[li] += red[li + 8u]; } workgroupBarrier();
  if (li < 4u) { red[li] += red[li + 4u]; } workgroupBarrier();
  if (li < 2u) { red[li] += red[li + 2u]; } workgroupBarrier();
  if (li == 0u) { red[0] += red[1]; } workgroupBarrier();
  if (!live) { return; }
  let ssq = red[0];
  let sq = sqrt(ssq / 128.0 + 1e-5);
  let st = atomTokenS[row];
  if (li == 0u) {
    var ma = 0.0; var ssa = 0.0; var sa = 0.0;
    if (st >= 0) {
      let k = u32(st) * 3u;
      ma = tstats[k]; ssa = tstats[k + 1u]; sa = tstats[k + 2u];
    }
    let mu = (768.0 * ma + 128.0 * mq) / 896.0;
    let da = ma - mu;
    let dq = mq - mu;
    let v = (ssa + ssq + 768.0 * da * da + 128.0 * dq * dq) / 896.0;
    let o = row * 6u;
    astats[o] = inverseSqrt(v + 1e-5);
    astats[o + 1u] = da;
    astats[o + 2u] = dq;
    astats[o + 3u] = sq;
    astats[o + 4u] = mq;
    astats[o + 5u] = sa;
  }
  var t = p.T;
  if (st >= 0) { t = u32(st) % p.T; }
  let gBase = t * 128u + li;
  let invSq = 1.0 / sq;
  xq[base + li] = gQ[gBase] * d0 * invSq;
  xq[base + li + 64u] = gQ[gBase + 64u] * d1 * invSq;
}`, rows, [q, gQ, tstats, atomTokenS, astats, xq], u);
  }

  /** y = x . W^T (+ bias) for a torch (out, in) weight over `rows` rows. */
  linear({ x, w, y, rows, inC, outC, bias, act, accumulate, xOff = 0, xStride = inC, yOff = 0, yStride = outC, wOff = 0, alpha, gate }) {
    return this.gemm({ m: rows, n: outC, k: inC, a: x, b: w, c: y, bias, act, accumulate, alpha, gate,
                       A: { off: xOff, m: xStride, k: 1 }, B: { off: wOff, n: inC, k: 1 }, C: { off: yOff, m: yStride, n: 1 } });
  }

  layerNorm({ x, y, rows, width, w, b, xStride = width, yStride = width, xOff = 0, yOff = 0, mod, eps = 1e-5 }) {
    const u = new Uint32Array(16);
    const f = new Float32Array(u.buffer);
    u.set([rows, width, xStride, yStride, xOff, yOff, w ? 1 : 0, b ? 1 : 0, mod ? 1 : 0,
      mod?.stride ?? 0, mod?.alpha ?? 0, mod?.beta ?? 0, mod?.map ? 1 : 0]);
    f[13] = eps;
    // in place: x is y, read through the read_write binding (one buffer cannot be bound both ways in a dispatch)
    const inPlace = x === y;
    const source = inPlace ? LN_SOURCE.replaceAll("x[xb + c]", "y[xb + c]")
      .replace("@group(0) @binding(1) var<storage, read> x: array<f32>;\n", "") : LN_SOURCE;
    return this.step(inPlace ? "ln-inplace" : "ln", source, rows, [inPlace ? null : x, y, w ?? this.dummy,
                                                                   b ?? this.dummy, mod?.cond ?? this.dummy, mod?.map ?? this.dummyInt], u);
  }

  softmax({ l, rows, len, rowsPer = rows, b1 = 1, L, bias, Bias, mask, M, scale = 1 }) {
    const u = new Uint32Array(20);
    const f = new Float32Array(u.buffer);
    const Ls = { off: 0, s0: 0, s1: 0, r: len, ...L };
    const Bs = { off: 0, s0: 0, s1: 0, r: len, ...Bias };
    const Ms = { off: 0, s0: 0, ...M };
    u.set([rows, len, b1, rowsPer, Ls.off, Ls.s0, Ls.s1, Ls.r, bias ? 1 : 0, Bs.off, Bs.s0, Bs.s1, Bs.r,
      mask ? 1 : 0, Ms.off, Ms.s0]);
    f[16] = scale;
    return this.step("softmax", SOFTMAX_SOURCE, rows, [l, bias ?? this.dummy, mask ?? this.dummyInt], u);
  }

  /** An elementwise kernel; `name` keys its source. */
  map(name, count, bindings, body, uints = [], floats = []) {
    const u = new Uint32Array(36);
    const f = new Float32Array(u.buffer);
    u[0] = count;
    u.set(uints, 4);
    f.set(floats, 20);
    // a binding the body never names is left out of the source AND the bind group: `auto` layouts drop it
    const used = bindings.map((_, k) => new RegExp(`\\bb${k}\\b`).test(body));
    const source = elementwiseSource(bindings.map((b, k) => (used[k] ? { type: b.type, write: b.write } : null)), body);
    return this.step(`map:${name}:${hashText(source)}`, source, Math.ceil(count / 256),
                     bindings.map((b, k) => (used[k] ? b.buffer : null)), u);
  }

  /** A list of steps, encoded together. */
  program(steps) {
    // a split GEMM is itself a two-step program; flattened, so a program is always a list of dispatches
    const flat = steps.flatMap((s) => (s.steps ? s.steps : [s]));
    return {
      steps: flat,
      run: (pass) => {
        for (const s of flat) {
          pass.setPipeline(s.pipeline);
          pass.setBindGroup(0, s.bindGroup);
          pass.dispatchWorkgroups(s.x, s.y);
        }
      },
    };
  }

  /**
   * A per-dispatch GPU profile: every step its own pass with timestamps, summed by key. Diagnostic only - one pass a
   * dispatch costs more than it measures - and on only when `profile` is set.
   */
  async runProfiled(...items) {
    const steps = items.flat().flatMap((item) => (item.steps ? item.steps : [item]));
    const query = this.device.createQuerySet({ type: "timestamp", count: 2 * steps.length });
    const resolve = this.device.createBuffer({ size: 16 * steps.length, usage: 0x200 | COPY_SRC });
    const encoder = this.device.createCommandEncoder({ label: "bioemu2.profile" });
    steps.forEach((s, at) => {
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: 2 * at,
                                                                 endOfPassWriteIndex: 2 * at + 1 } });
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bindGroup);
      pass.dispatchWorkgroups(s.x, s.y);
      pass.end();
    });
    encoder.resolveQuerySet(query, 0, 2 * steps.length, resolve, 0);
    this.device.queue.submit([encoder.finish()]);
    const times = await this.read(resolve, 4 * steps.length, 0, BigUint64Array === undefined ? Uint32Array : Uint32Array);
    const words = new BigUint64Array(times.buffer);
    this.profile ??= new Map();
    steps.forEach((s, at) => {
      const ns = Number(words[2 * at + 1] - words[2 * at]);
      const key = s.label ?? s.key;
      const row = this.profile.get(key) ?? { ms: 0, calls: 0 };
      row.ms += ns / 1e6;
      row.calls += 1;
      this.profile.set(key, row);
    });
    query.destroy();
    resolve.destroy();
  }

  /** Run steps (or programs) now, in one compute pass. */
  run(...items) {
    const encoder = this.device.createCommandEncoder({ label: "bioemu2" });
    const pass = encoder.beginComputePass({ label: "bioemu2" });
    for (const item of items.flat()) {
      if (item.run) item.run(pass);
      else { pass.setPipeline(item.pipeline); pass.setBindGroup(0, item.bindGroup); pass.dispatchWorkgroups(item.x, item.y); }
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async read(buffer, floats = buffer.floats, offsetFloats = 0, Type = Float32Array) {
    const size = Math.ceil(floats * 4 / 4) * 4;
    const staging = this.device.createBuffer({ size: Math.max(size, 4), usage: 0x1 | COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offsetFloats * 4, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(0x1);
    const out = new Type(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  destroy() {
    for (const b of this.allocated) b.destroy();
    this.allocated = [];
    this.bytes = 0;
    this.partials = undefined;
  }

  destroyWeights() {
    this.destroy();
    for (const b of this.weightBuffers) b.destroy();
    this.weightBuffers = [];
    this.weightBytes = 0;
  }
}

/** A short content hash, so two kernels of one name with different bodies are two keys. */
function hashText(text) {
  let h = 5381;
  for (let i = 0; i < text.length; i += 1) h = ((h * 33) ^ text.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

/** float32 -> float16 bits, round to nearest even. */
export function toHalf(value) {
  const f = new Float32Array([value]);
  const x = new Uint32Array(f.buffer)[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (((x >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const mid = 1 << (shift - 1);
    if (rem > mid || (rem === mid && (half & 1))) half += 1;
    return sign | half;
  }
  let half = (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half += 1;
  return sign | half;
}
