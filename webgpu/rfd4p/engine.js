/**
 * A small WebGPU op library for RFD4-Proteina: every op is RECORDED once (pipeline, bind group, uniform block)
 * and replayed, so a sampling step allocates nothing. Patterned on LocalFold's BioEmu2 recorder
 * (`../localfold/webgpu/bioemu2/ops.js`, whose tiled GEMM this adapts).
 *
 * Activations are float32. Weights are float32, or float16 packed two to a u32 (`half`), unpacked with the core
 * `unpack2x16float` (or staged as `f16` in workgroup memory when `shader-f16` is available).
 *
 * Every kernel that folds its workgroup index (`wid.x + wid.y * 65535u`) guards it against the real count before
 * touching a buffer (AGENTS.md).
 */

const MAX_X = 65535;
const STORAGE = 0x80;
const COPY_SRC = 0x4;
const COPY_DST = 0x8;
const UNIFORM = 0x40;
const MAP_READ = 0x1;

/** Fold `groups` workgroups into (x, y) under the per-dimension limit. */
export function fold2(groups) {
  if (groups <= MAX_X) return [Math.max(groups, 1), 1];
  return [MAX_X, Math.ceil(groups / MAX_X)];
}

/**
 * C = alpha * A . B^T (+ bias) [act] (+ C), over (b0, b1) batches, every operand addressed by strides.
 * A (m, k) at offA + b0 sA0 + b1 sA1 + m sAm + k sAk; B (n, k) likewise; C (m, n) likewise.
 * 64 x 128 tile, 256 lanes of 4 x 8 (two vec4s per row), K in steps of 16, vec4 shared-memory panels
 * padded to 17 and 33 vec4s per k-row to eliminate shared-memory bank conflicts.
 * `half` B: float16 packed along k (sBk = 1, even offsets); `b16`: B panel staged as vec4<f16>.
 */
function gemmSource(half, b16 = false, pair = null, gate = false, aF16 = false, cF16 = false, gF16 = false, fullK = false, vecEpilogue = false, addOnly = false, gSigm = true, hasAlpha = false, hasBias = false, actType = 0, acc = false, inLn = false, variant = null, vecA = false, vecB = false) {
  const rows = [0, 1, 2, 3];
  const EA = aF16 ? 'f16' : 'f32';
  const EB = b16 ? 'f16' : 'f32';
  // Elementwise B shares A's type, as in `gemmSubmatSource`; packed (`half`) B is read as words.
  const EBelem = aF16 ? 'f16' : 'f32';
  const isBase = Boolean(variant?.baseline);
  const useShufTa2 = Boolean(!isBase && (variant?.shufTa2 ?? true) && vecA);
  const useShufTa = useShufTa2;
  const useVecB = Boolean(!isBase && (variant?.vecB ?? true) && vecB && half);
  const usePrefA = Boolean(!isBase && (variant?.prefAll ?? true) && useShufTa && useVecB && b16);
  const usePrefB = Boolean(!isBase && (variant?.prefAll ?? true) && useVecB && b16);
  const EC = (vecEpilogue || pair?.cVec4) ? ((pair?.cF16 ?? cF16) ? 'vec2<u32>' : 'vec4<f32>') : ((pair?.cF16 ?? cF16) ? 'f16' : 'f32');
  const EG = vecEpilogue ? (gF16 ? 'vec2<u32>' : 'vec4<f32>') : (gF16 ? 'f16' : 'f32');
  const useF16 = b16 || aF16 || cF16 || gF16 || Boolean(pair?.cF16);
  const useSubgroups = inLn || useShufTa || useVecB;
  return `${useF16 ? 'enable f16;\n' : ''}${useSubgroups ? 'enable subgroups;\n' : ''}
struct P { m: u32, n: u32, k: u32, b1: u32,
  offA: u32, sA0: u32, sA1: u32, sAm: u32, sAk: u32,
  offB: u32, sB0: u32, sB1: u32, sBn: u32, sBk: u32,
  offC: u32, sC0: u32, sC1: u32, sCm: u32, sCn: u32,
  bias: u32, act: u32, accumulate: u32, alpha: f32, batches: u32,
  gOff: u32, gStride: u32, gSigm: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<${useShufTa ? `vec4<${EA}>` : EA}>;
@group(0) @binding(2) var<storage, read> b: array<${useVecB ? 'vec4<u32>' : (half ? 'u32' : EBelem)}>;
@group(0) @binding(3) var<storage, read_write> c: array<${EC}>;
@group(0) @binding(4) var<storage, read> bias: array<${vecEpilogue ? 'vec4<f32>' : 'f32'}>;
${gate ? `@group(0) @binding(5) var<storage, read> gsc: array<${EG}>;` : ''}
${vecEpilogue && (cF16 || gF16) ? `fn unpack4f16(u: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(vec4<f16>(bitcast<vec2<f16>>(u.x), bitcast<vec2<f16>>(u.y)));
}
fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}` : ''}
${pair ? `${pair.decls ?? ''}
fn bRow(v: u32) -> u32 { ${pair.bRow ?? 'return v;'} }
${pair.body}` : 'fn bRow(v: u32) -> u32 { return v; }'}
var<workgroup> ta: array<vec4<${EB}>, ${16 * 17}>;
var<workgroup> tb: array<vec4<${EB}>, ${16 * 33}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 127u) / 128u;
  let tilesM = (p.m + 63u) / 64u;
  let flat = wid.x + wid.y * 65535u;
  ${pair || !hasBias ? '_ = bias[0];' : ''}
  let batch = flat / (tilesN * tilesM);
  let inBatch = batch < p.batches;
  let tile = flat % (tilesN * tilesM);
  let m0 = (tile / tilesN) * 64u;
  let n0 = (tile % tilesN) * 128u;
  let b0 = batch / p.b1;
  let b1 = batch % p.b1;
  let baseA = p.offA + b0 * p.sA0 + b1 * p.sA1;
  let baseB = p.offB + b0 * p.sB0 + b1 * p.sB1;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 4u;
  let tn0 = tx * 4u;
  let tn1 = tn0 + 64u;
  var acc_0_0 = vec4<f32>(0.0); var acc_0_1 = vec4<f32>(0.0);
  var acc_1_0 = vec4<f32>(0.0); var acc_1_1 = vec4<f32>(0.0);
  var acc_2_0 = vec4<f32>(0.0); var acc_2_1 = vec4<f32>(0.0);
  var acc_3_0 = vec4<f32>(0.0); var acc_3_1 = vec4<f32>(0.0);
  ${inLn ? 'var sumA = vec4<f32>(0.0);\n  var sqSumA = vec4<f32>(0.0);' : ''}
  ${useShufTa2 ? `let rSubA = tx & 3u;
  let kSubA4 = tx >> 2u;
  let rowA = m0 + tm + rSubA;
  let aValid = inBatch && (rowA < p.m);
  let aBase4 = (baseA + min(rowA, p.m - 1u) * p.sAm) / 4u + kSubA4;
  let taWrite = tx * 17u + tm4;` : `let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let aBase0 = baseA + rowA0 * p.sAm + kkA * p.sAk;
  let fullA = rowA0 + 3u < p.m;
  let taWrite = kkA * 17u + r4A;`}
  ${useVecB ? `let r4B = li / 8u;
  let kHalfB = (li / 4u) & 1u;
  let rSubB = li & 3u;
  let rowB = n0 + r4B * 4u + rSubB;
  let bValid = inBatch && (rowB < p.n);
  let bBase8 = (baseB + bRow(min(rowB, p.n - 1u)) * p.sBn) / 8u + kHalfB;
  let kkB = kHalfB * 8u + rSubB * 2u;
  let tbWrite0 = kkB * 33u + r4B;` : half ? `let kkB = (li % 8u) * 2u;
  let r4B = li / 8u;
  let rowB0 = n0 + r4B * 4u;
  let bRowBase = baseB + bRow(rowB0) * p.sBn;
  let sBnHalf = p.sBn / 2u;
  let bBase0 = (bRowBase + kkB) / 2u;
  let fullB = rowB0 + 3u < p.n;
  let tbWrite0 = kkB * 33u + r4B;
  let tbWrite1 = (kkB + 1u) * 33u + r4B;` : ''}
  ${usePrefA ? `var aPref = select(vec4<${EA}>(0.0), a[aBase4], aValid);` : ''}
  ${usePrefB ? `var bPref = select(vec4<u32>(0u), b[bBase8], bValid);` : ''}
  for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    ${fullK ? '' : 'let fullK = inBatch && (k0 + 16u <= p.k);'}
    ${useShufTa ? `{
      let aVec = ${usePrefA ? 'aPref' : `select(vec4<${EA}>(0.0), a[aBase4 + k0 / 4u], aValid)`};
      let av = vec4<${EB}>(aVec);
      let s0 = select(vec2<${EB}>(av.x, av.z), vec2<${EB}>(av.y, av.w), (rSubA & 1u) == 0u);
      let q0 = subgroupShuffleXor(s0, 1u);
      let t = select(vec4<${EB}>(av.x, q0.x, av.z, q0.y), vec4<${EB}>(q0.x, av.y, q0.y, av.w), (rSubA & 1u) != 0u);
      let s1 = select(t.xy, t.zw, (rSubA & 2u) == 0u);
      let q1 = subgroupShuffleXor(s1, 2u);
      let u = select(vec4<${EB}>(t.xy, q1), vec4<${EB}>(q1, t.zw), (rSubA & 2u) != 0u);
      ${inLn ? 'let vf = vec4<f32>(u); sumA += vf; sqSumA += vf * vf;' : ''}
      ta[taWrite] = u;
    }` : `{
      var v = vec4<${EB}>(0.0);
      if (${fullK ? 'inBatch' : 'fullK || (inBatch && k0 + kkA < p.k)'}) {
        let aOff = aBase0 + k0${fullK ? '' : ' * p.sAk'};
        if (fullA) {
          v = vec4<${EB}>(${EB}(a[aOff]), ${EB}(a[aOff + p.sAm]), ${EB}(a[aOff + 2u * p.sAm]), ${EB}(a[aOff + 3u * p.sAm]));
        } else {
          if (rowA0 < p.m) { v.x = ${EB}(a[aOff]); }
          if (rowA0 + 1u < p.m) { v.y = ${EB}(a[aOff + p.sAm]); }
          if (rowA0 + 2u < p.m) { v.z = ${EB}(a[aOff + 2u * p.sAm]); }
          if (rowA0 + 3u < p.m) { v.w = ${EB}(a[aOff + 3u * p.sAm]); }
        }
      }
      ${inLn ? 'let vf = vec4<f32>(v); sumA += vf; sqSumA += vf * vf;' : ''}
      ta[taWrite] = v;
    }`}
    ${useVecB ? `{
      let w = ${usePrefB ? 'bPref' : `select(vec4<u32>(0u), b[bBase8 + k0 / 8u], bValid)`};
      let s0 = select(vec2<u32>(w.x, w.z), vec2<u32>(w.y, w.w), (rSubB & 1u) == 0u);
      let q0 = subgroupShuffleXor(s0, 1u);
      let t = select(vec4<u32>(w.x, q0.x, w.z, q0.y), vec4<u32>(q0.x, w.y, q0.y, w.w), (rSubB & 1u) != 0u);
      let s1 = select(t.xy, t.zw, (rSubB & 2u) == 0u);
      let q1 = subgroupShuffleXor(s1, 2u);
      let u = select(vec4<u32>(t.xy, q1), vec4<u32>(q1, t.zw), (rSubB & 2u) != 0u);
      ${b16 ? `let p0 = bitcast<vec2<f16>>(u.x);
      let p1 = bitcast<vec2<f16>>(u.y);
      let p2 = bitcast<vec2<f16>>(u.z);
      let p3 = bitcast<vec2<f16>>(u.w);
      tb[tbWrite0] = vec4<f16>(p0.x, p1.x, p2.x, p3.x);
      tb[tbWrite0 + 33u] = vec4<f16>(p0.y, p1.y, p2.y, p3.y);` : `let p0 = unpack2x16float(u.x);
      let p1 = unpack2x16float(u.y);
      let p2 = unpack2x16float(u.z);
      let p3 = unpack2x16float(u.w);
      tb[tbWrite0] = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
      tb[tbWrite0 + 33u] = vec4<f32>(p0.y, p1.y, p2.y, p3.y);`}
    }` : half ? `{
      var lo = vec4<${EB}>(0.0);
      var hi = vec4<${EB}>(0.0);
      if (${fullK ? 'inBatch' : 'fullK'}) {
        let bOff = bBase0 + (k0 / 2u);
        ${b16 ? `if (fullB) {
          let p0 = bitcast<vec2<f16>>(b[bOff]);
          let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]);
          let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]);
          let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]);
          lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x);
          hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y);
        } else {
          if (rowB0 < p.n) { let p0 = bitcast<vec2<f16>>(b[bOff]); lo.x = p0.x; hi.x = p0.y; }
          if (rowB0 + 1u < p.n) { let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]); lo.y = p1.x; hi.y = p1.y; }
          if (rowB0 + 2u < p.n) { let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]); lo.z = p2.x; hi.z = p2.y; }
          if (rowB0 + 3u < p.n) { let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]); lo.w = p3.x; hi.w = p3.y; }
        }` : `if (fullB) {
          let p0 = unpack2x16float(b[bOff]);
          let p1 = unpack2x16float(b[bOff + sBnHalf]);
          let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
          hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y);
        } else {
          if (rowB0 < p.n) { let p0 = unpack2x16float(b[bOff]); lo.x = p0.x; hi.x = p0.y; }
          if (rowB0 + 1u < p.n) { let p1 = unpack2x16float(b[bOff + sBnHalf]); lo.y = p1.x; hi.y = p1.y; }
          if (rowB0 + 2u < p.n) { let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]); lo.z = p2.x; hi.z = p2.y; }
          if (rowB0 + 3u < p.n) { let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]); lo.w = p3.x; hi.w = p3.y; }
        }`}
      }${fullK ? '' : ` else if (inBatch && k0 + kkB < p.k) {
        let hasHi = k0 + kkB + 1u < p.k;
        let bOff = bBase0 + (k0 / 2u);
        ${b16 ? `if (fullB) {
          let p0 = bitcast<vec2<f16>>(b[bOff]);
          let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]);
          let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]);
          let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]);
          lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x);
          if (hasHi) { hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (rowB0 < p.n) { let p0 = bitcast<vec2<f16>>(b[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (rowB0 + 1u < p.n) { let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (rowB0 + 2u < p.n) { let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (rowB0 + 3u < p.n) { let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
        }` : `if (fullB) {
          let p0 = unpack2x16float(b[bOff]);
          let p1 = unpack2x16float(b[bOff + sBnHalf]);
          let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
          if (hasHi) { hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (rowB0 < p.n) { let p0 = unpack2x16float(b[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (rowB0 + 1u < p.n) { let p1 = unpack2x16float(b[bOff + sBnHalf]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (rowB0 + 2u < p.n) { let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (rowB0 + 3u < p.n) { let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
        }`}
      }`}
      tb[tbWrite0] = lo;
      tb[tbWrite1] = hi;
    }` : `for (var e = 0u; e < 2u; e += 1u) {
      let slot = li + e * 256u;
      let kk = slot % 16u;
      let r4 = slot / 16u;
      let k = k0 + kk;
      var v = vec4<f32>(0.0);
      if (${fullK ? 'inBatch' : 'inBatch && k < p.k'}) {
        let row0 = n0 + r4 * 4u;
        let bAddr0 = baseB + bRow(row0) * p.sBn + k * p.sBk;
        if (row0 + 3u < p.n) {
          v = vec4<f32>(f32(b[bAddr0]), f32(b[bAddr0 + p.sBn]), f32(b[bAddr0 + 2u * p.sBn]), f32(b[bAddr0 + 3u * p.sBn]));
        } else {
          if (row0 < p.n) { v.x = f32(b[bAddr0]); }
          if (row0 + 1u < p.n) { v.y = f32(b[bAddr0 + p.sBn]); }
          if (row0 + 2u < p.n) { v.z = f32(b[bAddr0 + 2u * p.sBn]); }
          if (row0 + 3u < p.n) { v.w = f32(b[bAddr0 + 3u * p.sBn]); }
        }
      }
      tb[kk * 33u + r4] = vec4<${EB}>(v);
    }`}
    workgroupBarrier();
    ${Array.from({ length: 16 }, (_, kk) => `{
      let av = vec4<f32>(ta[${kk * 17}u + tm4]);
      let bv0 = vec4<f32>(tb[${kk * 33}u + tx]);
      let bv1 = vec4<f32>(tb[${kk * 33 + 16}u + tx]);
      acc_0_0 = fma(vec4<f32>(av.x), bv0, acc_0_0); acc_0_1 = fma(vec4<f32>(av.x), bv1, acc_0_1);
      acc_1_0 = fma(vec4<f32>(av.y), bv0, acc_1_0); acc_1_1 = fma(vec4<f32>(av.y), bv1, acc_1_1);
      acc_2_0 = fma(vec4<f32>(av.z), bv0, acc_2_0); acc_2_1 = fma(vec4<f32>(av.z), bv1, acc_2_1);
      acc_3_0 = fma(vec4<f32>(av.w), bv0, acc_3_0); acc_3_1 = fma(vec4<f32>(av.w), bv1, acc_3_1);
    }`).join('\n    ')}
    ${usePrefA || usePrefB ? `{
      let nextK = k0 + 16u;
      let hasNext = nextK < p.k;
      ${usePrefA ? `aPref = select(vec4<${EA}>(0.0), a[aBase4 + nextK / 4u], aValid && hasNext);` : ''}
      ${usePrefB ? `bPref = select(vec4<u32>(0u), b[bBase8 + nextK / 8u], bValid && hasNext);` : ''}
    }` : ''}
    workgroupBarrier();
  }
  ${inLn ? `sumA += subgroupShuffleXor(sumA, 1u);
  sumA += subgroupShuffleXor(sumA, 2u);
  sumA += subgroupShuffleXor(sumA, 4u);
  sumA += subgroupShuffleXor(sumA, 8u);
  sqSumA += subgroupShuffleXor(sqSumA, 1u);
  sqSumA += subgroupShuffleXor(sqSumA, 2u);
  sqSumA += subgroupShuffleXor(sqSumA, 4u);
  sqSumA += subgroupShuffleXor(sqSumA, 8u);
  let invK = 1.0 / f32(p.k);
  let mean4 = sumA * invK;
  let invStd4 = inverseSqrt(max(sqSumA * invK - mean4 * mean4, vec4<f32>(0.0)) + vec4<f32>(1e-5));` : ''}
  ${pair?.epilogueBody ? `${pair.epilogueBody}\n  return;` : pair ? `if (!inBatch) { return; }
  ${rows.map((i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) { pairOut(m, n0, tn0, acc_${i}_0 * p.alpha, acc_${i}_1 * p.alpha); }
  }`).join('\n  ')}` : vecEpilogue ? `if (!inBatch) { return; }
  let baseC4 = (p.offC + b0 * p.sC0 + b1 * p.sC1) / 4u;
  let sCm4 = p.sCm / 4u;
  let col0 = n0 + tn0;
  let col1 = n0 + tn1;
  let valid0 = col0 < p.n;
  let valid1 = col1 < p.n;
  ${hasBias ? `let bVal0 = select(vec4<f32>(0.0), bias[(b0 * p.n + col0) / 4u], valid0);
  let bVal1 = select(vec4<f32>(0.0), bias[(b0 * p.n + col1) / 4u], valid1);` : ''}
  ${inLn ? `let wSum0 = select(vec4<f32>(0.0), bias[(p.n + col0) / 4u], valid0);
  let wSum1 = select(vec4<f32>(0.0), bias[(p.n + col1) / 4u], valid1);` : ''}
  ${gate ? `let baseG4 = (p.gOff + b0 * p.sC0 + b1 * p.sC1) / 4u;
  let gStride4 = p.gStride / 4u;` : ''}
  ${rows.map((i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) {
      let rowC4 = baseC4 + m * sCm4;
      ${[0, 1].map((h) => `if (valid${h}) {
        ${inLn ? `var v = (acc_${i}_${h} - mean4[${i}] * wSum${h}) * invStd4[${i}]${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ` + bVal${h}` : ''};` : `var v = acc_${i}_${h}${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ` + bVal${h}` : ''};`}
        ${actType === 1 ? 'v = max(v, vec4<f32>(0.0));' : actType === 2 ? 'v = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v));' : ''}
        let at4 = rowC4 + (col${h} / 4u);
        ${gate ? `let gv = ${gF16 ? `unpack4f16(gsc[baseG4 + m * gStride4 + (col${h} / 4u)])` : `gsc[baseG4 + m * gStride4 + (col${h} / 4u)]`};
        ${addOnly ? 'var res = v + gv;' : gSigm ? 'var res = v / (vec4<f32>(1.0) + exp(-gv));' : 'var res = v * gv;'}
        ${!addOnly && acc ? `res += ${cF16 ? 'unpack4f16(c[at4])' : 'c[at4]'};` : ''}
        c[at4] = ${cF16 ? 'pack4f16(res)' : 'res'};` : `${acc ? `v += ${cF16 ? 'unpack4f16(c[at4])' : 'c[at4]'};` : ''}
        c[at4] = ${cF16 ? 'pack4f16(v)' : 'v'};`}
      }`).join('\n      ')}
    }
  }`).join('\n  ')}` : `if (!inBatch) { return; }
  let baseC = p.offC + b0 * p.sC0 + b1 * p.sC1;
  ${rows.map((i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) {
      let r0 = acc_${i}_0${hasAlpha ? ' * p.alpha' : ''};
      let r1 = acc_${i}_1${hasAlpha ? ' * p.alpha' : ''};
      let rowC = baseC + m * p.sCm;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((j) => {
        const nExpr = j < 4 ? `n0 + tn0 + ${j}u` : `n0 + tn1 + ${j - 4}u`;
        const valExpr = j < 4 ? `r0.${'xyzw'[j]}` : `r1.${'xyzw'[j - 4]}`;
        return `{
        let n = ${nExpr};
        if (n < p.n) {
          var v = ${valExpr}${hasBias ? ' + bias[b0 * p.n + n]' : ''};
          ${actType === 1 ? 'v = max(v, 0.0);' : actType === 2 ? 'v = 1.0 / (1.0 + exp(-v));' : ''}
          let at = rowC + n * p.sCn;
          ${gate ? `let gv = f32(gsc[p.gOff + b0 * p.sC0 + b1 * p.sC1 + m * p.gStride + n]);
          ${addOnly ? `c[at] = ${EC}(v + gv);` : `let gf = ${gSigm ? '1.0 / (1.0 + exp(-gv))' : 'gv'};
          var vg = v * gf;
          ${acc ? 'vg += f32(c[at]);' : ''}
          c[at] = ${EC}(vg);`}` : `${acc ? 'v += f32(c[at]);' : ''}
          c[at] = ${EC}(v);`}
        }
      }`;
      }).join('\n      ')}
    }
  }`).join('\n  ')}`}
}`;
}

/**
 * Zero-barrier subgroup GEMV for `m in {1, 2}` and `k = K128 * 128` (`vec4<f32>` A, `vec2<u32>` packed `f16` B).
 * Each subgroup of 32 lanes computes 4 output columns (`16` columns per 128-thread workgroup).
 */
function gemvSource(mRows, K128, hasGate = false, addOnly = false, gSigm = true, hasAlpha = false, hasBias = false, actType = 0, hasAcc = false, inNorm = '') {
  const sIdx = Array.from({ length: K128 }, (_, s) => s);
  const hasNormW = inNorm === 'rmsPre';
  return `enable f16;
enable subgroups;
struct P { m: u32, n: u32, k: u32, b1: u32,
  offA: u32, sA0: u32, sA1: u32, sAm: u32, sAk: u32,
  offB: u32, sB0: u32, sB1: u32, sBn: u32, sBk: u32,
  offC: u32, sC0: u32, sC1: u32, sCm: u32, sCn: u32,
  bias: u32, act: u32, accumulate: u32, alpha: f32, batches: u32,
  gOff: u32, gStride: u32, gSigm: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> c: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bias: array<vec4<f32>>;
${hasGate ? '@group(0) @binding(5) var<storage, read> gsc: array<vec4<f32>>;' : hasNormW ? '@group(0) @binding(5) var<storage, read> wNorm: array<vec4<f32>>;' : ''}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let wg = wid.x + wid.y * 65535u;
  let gpb = (p.n + 15u) / 16u;
  let batchRaw = wg / gpb;
  let warp = li / 32u;
  let lane = li % 32u;
  let col4 = (wg % gpb) * 4u + warp;
  let col0 = col4 * 4u;
  let live = (batchRaw < p.batches) && (col0 < p.n);
  let batch = min(batchRaw, p.batches - 1u);
  let cCol0 = min(col0, p.n - 4u);
  let b0 = batch / p.b1;
  let b1 = batch % p.b1;
  let baseA0 = (p.offA + b0 * p.sA0 + b1 * p.sA1) / 4u + lane;
  ${mRows === 2 ? 'let baseA1 = baseA0 + (p.sAm / 4u);' : ''}
  ${sIdx.map((s) => `var a0_${s} = a[baseA0 + ${s * 32}u];`).join('\n  ')}
  ${mRows === 2 ? sIdx.map((s) => `var a1_${s} = a[baseA1 + ${s * 32}u];`).join('\n  ') : ''}
  ${inNorm === 'rmsCond' ? `let q0 = ${sIdx.map((s) => `a0_${s} * a0_${s}`).join(' + ')};
  let inv0 = inverseSqrt(r32Add(q0.x + q0.y + q0.z + q0.w) / ${K128 * 128}.0 + 1e-5);
  ${mRows === 2 ? `let q1 = ${sIdx.map((s) => `a1_${s} * a1_${s}`).join(' + ')};
  let inv1 = inverseSqrt(r32Add(q1.x + q1.y + q1.z + q1.w) / ${K128 * 128}.0 + 1e-5);` : ''}
  if (b0 % 3u < 2u) {
    ${sIdx.map((s) => `a0_${s} *= inv0;`).join(' ')}
    ${mRows === 2 ? sIdx.map((s) => `a1_${s} *= inv1;`).join(' ') : ''}
  }` : inNorm === 'lnCond' ? `let s0 = ${sIdx.map((s) => `a0_${s}`).join(' + ')};
  let mean0 = r32Add(s0.x + s0.y + s0.z + s0.w) / ${K128 * 128}.0;
  ${sIdx.map((s) => `let dm0_${s} = a0_${s} - vec4<f32>(mean0);`).join('\n  ')}
  let q0 = ${sIdx.map((s) => `dm0_${s} * dm0_${s}`).join(' + ')};
  let inv0 = inverseSqrt(r32Add(q0.x + q0.y + q0.z + q0.w) / ${K128 * 128}.0 + 1e-5);
  ${mRows === 2 ? `let s1 = ${sIdx.map((s) => `a1_${s}`).join(' + ')};
  let mean1 = r32Add(s1.x + s1.y + s1.z + s1.w) / ${K128 * 128}.0;
  ${sIdx.map((s) => `let dm1_${s} = a1_${s} - vec4<f32>(mean1);`).join('\n  ')}
  let q1 = ${sIdx.map((s) => `dm1_${s} * dm1_${s}`).join(' + ')};
  let inv1 = inverseSqrt(r32Add(q1.x + q1.y + q1.z + q1.w) / ${K128 * 128}.0 + 1e-5);` : ''}
  if (b0 % 3u < 2u) {
    ${sIdx.map((s) => `a0_${s} = dm0_${s} * inv0;`).join(' ')}
    ${mRows === 2 ? sIdx.map((s) => `a1_${s} = dm1_${s} * inv1;`).join(' ') : ''}
  }` : inNorm === 'rmsPre' ? `let q0 = ${sIdx.map((s) => `a0_${s} * a0_${s}`).join(' + ')};
  let inv0 = inverseSqrt(r32Add(q0.x + q0.y + q0.z + q0.w) / ${K128 * 128}.0 + 1e-5);
  ${sIdx.map((s) => `a0_${s} = a0_${s} * inv0 * wNorm[lane + ${s * 32}u];`).join('\n  ')}
  let q2 = ${sIdx.map((s) => `a0_${s} * a0_${s}`).join(' + ')};
  let inv2 = inverseSqrt(r32Add(q2.x + q2.y + q2.z + q2.w) / ${K128 * 128}.0 + 1e-5);
  ${sIdx.map((s) => `a0_${s} *= inv2;`).join(' ')}` : ''}
  let sBn4 = p.sBn / 4u;
  let baseB = (p.offB + b0 * p.sB0 + b1 * p.sB1) / 4u + cCol0 * sBn4 + lane;
  ${[0, 1, 2, 3].map((j) => {
    const bOff = j === 0 ? 'baseB' : `baseB + ${j}u * sBn4`;
    return `var sum0_${j} = 0.0;
  ${mRows === 2 ? `var sum1_${j} = 0.0;` : ''}
  ${sIdx.map((s) => `{
    let w = vec4<f32>(bitcast<vec4<f16>>(b[${bOff} + ${s * 32}u]));
    sum0_${j} += dot(a0_${s}, w);
    ${mRows === 2 ? `sum1_${j} += dot(a1_${s}, w);` : ''}
  }`).join('\n  ')}
  let d0_${j} = r32Add(sum0_${j});
  ${mRows === 2 ? `let d1_${j} = r32Add(sum1_${j});` : ''}`;
  }).join('\n  ')}
  ${!hasBias ? '_ = bias[0];' : ''}
  if (!live) { return; }
  if (lane != 0u) { return; }
  let baseC0 = (p.offC + b0 * p.sC0 + b1 * p.sC1) / 4u + col4;
  ${hasBias ? 'let bv = bias[(b0 * p.n) / 4u + col4];' : ''}
  var v0 = vec4<f32>(d0_0, d0_1, d0_2, d0_3)${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bv' : ''};
  ${actType === 1 ? 'v0 = max(v0, vec4<f32>(0.0));' : actType === 2 ? 'v0 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v0));' : ''}
  ${hasGate ? `{
    let gv0 = gsc[(p.gOff + b0 * p.sC0 + b1 * p.sC1) / 4u + col4];
    ${addOnly ? 'v0 = v0 + gv0;' : gSigm ? 'v0 = v0 / (vec4<f32>(1.0) + exp(-gv0));' : 'v0 = v0 * gv0;'}
  }` : ''}
  ${hasAcc ? 'v0 += c[baseC0];' : ''}
  c[baseC0] = v0;
  ${mRows === 2 ? `let baseC1 = baseC0 + (p.sCm / 4u);
  var v1 = vec4<f32>(d1_0, d1_1, d1_2, d1_3)${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bv' : ''};
  ${actType === 1 ? 'v1 = max(v1, vec4<f32>(0.0));' : actType === 2 ? 'v1 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v1));' : ''}
  ${hasGate ? `{
    let gv1 = gsc[(p.gOff + b0 * p.sC0 + b1 * p.sC1 + p.gStride) / 4u + col4];
    ${addOnly ? 'v1 = v1 + gv1;' : gSigm ? 'v1 = v1 / (vec4<f32>(1.0) + exp(-gv1));' : 'v1 = v1 * gv1;'}
  }` : ''}
  ${hasAcc ? 'v1 += c[baseC1];' : ''}
  c[baseC1] = v1;` : ''}
}`;
}

function swigluGemvSource(mRows, K128, inner) {
  const sIdx = Array.from({ length: K128 }, (_, s) => s);
  return `enable f16;
enable subgroups;
struct P { m: u32, inner: u32, k: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> c: array<vec4<f32>>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let wg = wid.x + wid.y * 65535u;
  let warp = li / 32u;
  let lane = li % 32u;
  let col4 = wg * 4u + warp;
  let col0 = col4 * 4u;
  let live = col0 < p.inner;
  let cCol0 = min(col0, ${inner - 4}u);
  ${sIdx.map((s) => `let a0_${s} = a[lane + ${s * 32}u];`).join('\n  ')}
  ${mRows === 2 ? sIdx.map((s) => `let a1_${s} = a[${K128 * 32}u + lane + ${s * 32}u];`).join('\n  ') : ''}
  let bVal0 = cCol0 * ${K128 * 32}u + lane;
  let bGate0 = (cCol0 + ${inner}u) * ${K128 * 32}u + lane;
  ${[0, 1, 2, 3].map((j) => `var sv0_${j} = 0.0; var sg0_${j} = 0.0;
  ${mRows === 2 ? `var sv1_${j} = 0.0; var sg1_${j} = 0.0;` : ''}
  ${sIdx.map((s) => `{
    let wv = vec4<f32>(bitcast<vec4<f16>>(b[bVal0 + ${(j * K128 + s) * 32}u]));
    let wg_v = vec4<f32>(bitcast<vec4<f16>>(b[bGate0 + ${(j * K128 + s) * 32}u]));
    sv0_${j} += dot(a0_${s}, wv);
    sg0_${j} += dot(a0_${s}, wg_v);
    ${mRows === 2 ? `sv1_${j} += dot(a1_${s}, wv);
    sg1_${j} += dot(a1_${s}, wg_v);` : ''}
  }`).join('\n  ')}
  let dv0_${j} = r32Add(sv0_${j});
  let dg0_${j} = r32Add(sg0_${j});
  ${mRows === 2 ? `let dv1_${j} = r32Add(sv1_${j});
  let dg1_${j} = r32Add(sg1_${j});` : ''}`).join('\n  ')}
  if (!live) { return; }
  if (lane != 0u) { return; }
  let v0 = vec4<f32>(dv0_0, dv0_1, dv0_2, dv0_3);
  let g0 = vec4<f32>(dg0_0, dg0_1, dg0_2, dg0_3);
  c[col4] = (g0 / (vec4<f32>(1.0) + exp(-g0))) * v0;
  ${mRows === 2 ? `let v1 = vec4<f32>(dv1_0, dv1_1, dv1_2, dv1_3);
  let g1 = vec4<f32>(dg1_0, dg1_1, dg1_2, dg1_3);
  c[${inner / 4}u + col4] = (g1 / (vec4<f32>(1.0) + exp(-g1))) * v1;` : ''}
}`;
}

/**
 * Small-M GEMM: 32 x 64 tile, 256 lanes of 2 x 4 (one vec4 per row), K in steps of 16.
 * Used when `m * batches <= 512` to give 4x more workgroups and higher SM occupancy on token-track projections.
 */
function gemmSmallSource(half, b16 = false, gate = false, aF16 = false, cF16 = false, gF16 = false) {
  const rows = [0, 1];
  const EA = aF16 ? 'f16' : 'f32';
  const EB = b16 ? 'f16' : 'f32';
  // Elementwise B shares A's type, as in `gemmSubmatSource`; packed (`half`) B is read as words.
  const EBelem = aF16 ? 'f16' : 'f32';
  const EC = cF16 ? 'f16' : 'f32';
  const EG = gF16 ? 'f16' : 'f32';
  const useF16 = b16 || aF16 || cF16 || gF16;
  return `${useF16 ? 'enable f16;\n' : ''}
struct P { m: u32, n: u32, k: u32, b1: u32,
  offA: u32, sA0: u32, sA1: u32, sAm: u32, sAk: u32,
  offB: u32, sB0: u32, sB1: u32, sBn: u32, sBk: u32,
  offC: u32, sC0: u32, sC1: u32, sCm: u32, sCn: u32,
  bias: u32, act: u32, accumulate: u32, alpha: f32, batches: u32,
  gOff: u32, gStride: u32, gSigm: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<${EA}>;
@group(0) @binding(2) var<storage, read> b: array<${half ? 'u32' : EBelem}>;
@group(0) @binding(3) var<storage, read_write> c: array<${EC}>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
${gate ? `@group(0) @binding(5) var<storage, read> gsc: array<${EG}>;` : ''}
var<workgroup> ta: array<vec4<${EB}>, 144>;
var<workgroup> tb: array<vec4<${EB}>, 272>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tilesN = (p.n + 63u) / 64u;
  let tilesM = (p.m + 31u) / 32u;
  let flat = wid.x + wid.y * 65535u;
  let batch = flat / (tilesN * tilesM);
  let inBatch = batch < p.batches;
  let tile = flat % (tilesN * tilesM);
  let m0 = (tile / tilesN) * 32u;
  let n0 = (tile % tilesN) * 64u;
  let b0 = batch / p.b1;
  let b1 = batch % p.b1;
  let baseA = p.offA + b0 * p.sA0 + b1 * p.sA1;
  let baseB = p.offB + b0 * p.sB0 + b1 * p.sB1;
  let tm4 = li / 16u;
  let tx = li % 16u;
  let tm = tm4 * 2u;
  let tn0 = tx * 4u;
  let r4Read = tm4 / 2u;
  let hiPair = (tm4 & 1u) == 1u;
  var acc_0 = vec4<f32>(0.0);
  var acc_1 = vec4<f32>(0.0);
  let kkA = li % 16u;
  let r4A = li / 16u;
  let rowA0 = m0 + r4A * 4u;
  let aBase0 = baseA + rowA0 * p.sAm + kkA * p.sAk;
  let fullA = rowA0 + 3u < p.m;
  let taWrite = kkA * 9u + r4A;
  ${half ? `let kkB = (li % 8u) * 2u;
  let r4B = li / 8u;
  let rowB0 = n0 + r4B * 4u;
  let bRowBase = baseB + rowB0 * p.sBn;
  let sBnHalf = p.sBn / 2u;
  let bBase0 = (bRowBase + kkB) / 2u;
  let fullB = rowB0 + 3u < p.n;
  let tbWrite0 = kkB * 17u + r4B;
  let tbWrite1 = (kkB + 1u) * 17u + r4B;` : `let kkB = li % 16u;
  let r4B = li / 16u;
  let rowB0 = n0 + r4B * 4u;
  let bBase0 = baseB + rowB0 * p.sBn + kkB * p.sBk;
  let fullB = rowB0 + 3u < p.n;
  let tbWrite = kkB * 17u + r4B;`}
  for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    let fullK = inBatch && (k0 + 16u <= p.k);
    if (li < 128u) {
      var v = vec4<${EB}>(0.0);
      if (fullK || (inBatch && k0 + kkA < p.k)) {
        let aOff = aBase0 + k0 * p.sAk;
        if (fullA) {
          v = vec4<${EB}>(${EB}(a[aOff]), ${EB}(a[aOff + p.sAm]), ${EB}(a[aOff + 2u * p.sAm]), ${EB}(a[aOff + 3u * p.sAm]));
        } else {
          if (rowA0 < p.m) { v.x = ${EB}(a[aOff]); }
          if (rowA0 + 1u < p.m) { v.y = ${EB}(a[aOff + p.sAm]); }
          if (rowA0 + 2u < p.m) { v.z = ${EB}(a[aOff + 2u * p.sAm]); }
          if (rowA0 + 3u < p.m) { v.w = ${EB}(a[aOff + 3u * p.sAm]); }
        }
      }
      ta[taWrite] = v;
    }
    ${half ? `if (li < 128u) {
      var lo = vec4<${EB}>(0.0);
      var hi = vec4<${EB}>(0.0);
      if (fullK) {
        let bOff = bBase0 + (k0 / 2u);
        ${b16 ? `if (fullB) {
          let p0 = bitcast<vec2<f16>>(b[bOff]);
          let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]);
          let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]);
          let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]);
          lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x);
          hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y);
        } else {
          if (rowB0 < p.n) { let p0 = bitcast<vec2<f16>>(b[bOff]); lo.x = p0.x; hi.x = p0.y; }
          if (rowB0 + 1u < p.n) { let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]); lo.y = p1.x; hi.y = p1.y; }
          if (rowB0 + 2u < p.n) { let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]); lo.z = p2.x; hi.z = p2.y; }
          if (rowB0 + 3u < p.n) { let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]); lo.w = p3.x; hi.w = p3.y; }
        }` : `if (fullB) {
          let p0 = unpack2x16float(b[bOff]);
          let p1 = unpack2x16float(b[bOff + sBnHalf]);
          let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
          hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y);
        } else {
          if (rowB0 < p.n) { let p0 = unpack2x16float(b[bOff]); lo.x = p0.x; hi.x = p0.y; }
          if (rowB0 + 1u < p.n) { let p1 = unpack2x16float(b[bOff + sBnHalf]); lo.y = p1.x; hi.y = p1.y; }
          if (rowB0 + 2u < p.n) { let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]); lo.z = p2.x; hi.z = p2.y; }
          if (rowB0 + 3u < p.n) { let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]); lo.w = p3.x; hi.w = p3.y; }
        }`}
      } else if (inBatch && k0 + kkB < p.k) {
        let hasHi = k0 + kkB + 1u < p.k;
        let bOff = bBase0 + (k0 / 2u);
        ${b16 ? `if (fullB) {
          let p0 = bitcast<vec2<f16>>(b[bOff]);
          let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]);
          let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]);
          let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]);
          lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x);
          if (hasHi) { hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (rowB0 < p.n) { let p0 = bitcast<vec2<f16>>(b[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (rowB0 + 1u < p.n) { let p1 = bitcast<vec2<f16>>(b[bOff + sBnHalf]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (rowB0 + 2u < p.n) { let p2 = bitcast<vec2<f16>>(b[bOff + 2u * sBnHalf]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (rowB0 + 3u < p.n) { let p3 = bitcast<vec2<f16>>(b[bOff + 3u * sBnHalf]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
        }` : `if (fullB) {
          let p0 = unpack2x16float(b[bOff]);
          let p1 = unpack2x16float(b[bOff + sBnHalf]);
          let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]);
          let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]);
          lo = vec4<f32>(p0.x, p1.x, p2.x, p3.x);
          if (hasHi) { hi = vec4<f32>(p0.y, p1.y, p2.y, p3.y); }
        } else {
          if (rowB0 < p.n) { let p0 = unpack2x16float(b[bOff]); lo.x = p0.x; if (hasHi) { hi.x = p0.y; } }
          if (rowB0 + 1u < p.n) { let p1 = unpack2x16float(b[bOff + sBnHalf]); lo.y = p1.x; if (hasHi) { hi.y = p1.y; } }
          if (rowB0 + 2u < p.n) { let p2 = unpack2x16float(b[bOff + 2u * sBnHalf]); lo.z = p2.x; if (hasHi) { hi.z = p2.y; } }
          if (rowB0 + 3u < p.n) { let p3 = unpack2x16float(b[bOff + 3u * sBnHalf]); lo.w = p3.x; if (hasHi) { hi.w = p3.y; } }
        }`}
      }
      tb[tbWrite0] = lo;
      tb[tbWrite1] = hi;
    }` : `{
      var v = vec4<f32>(0.0);
      if (fullK || (inBatch && k0 + kkB < p.k)) {
        let bOff = k0 * p.sBk;
        let bAddr0 = bBase0 + bOff;
        if (fullB) {
          v = vec4<f32>(f32(b[bAddr0]), f32(b[bAddr0 + p.sBn]), f32(b[bAddr0 + 2u * p.sBn]), f32(b[bAddr0 + 3u * p.sBn]));
        } else {
          if (rowB0 < p.n) { v.x = f32(b[bAddr0]); }
          if (rowB0 + 1u < p.n) { v.y = f32(b[bAddr0 + p.sBn]); }
          if (rowB0 + 2u < p.n) { v.z = f32(b[bAddr0 + 2u * p.sBn]); }
          if (rowB0 + 3u < p.n) { v.w = f32(b[bAddr0 + 3u * p.sBn]); }
        }
      }
      tb[tbWrite] = vec4<${EB}>(v);
    }`}
    workgroupBarrier();
    ${Array.from({ length: 16 }, (_, kk) => `{
      let av4 = vec4<f32>(ta[${kk * 9}u + r4Read]);
      let a0 = select(av4.x, av4.z, hiPair);
      let a1 = select(av4.y, av4.w, hiPair);
      let bv = vec4<f32>(tb[${kk * 17}u + tx]);
      acc_0 = fma(vec4<f32>(a0), bv, acc_0);
      acc_1 = fma(vec4<f32>(a1), bv, acc_1);
    }`).join('\n    ')}
    workgroupBarrier();
  }
  if (!inBatch) { return; }
  let baseC = p.offC + b0 * p.sC0 + b1 * p.sC1;
  ${rows.map((i) => `{
    let m = m0 + tm + ${i}u;
    if (m < p.m) {
      let r0 = acc_${i} * p.alpha;
      let rowC = baseC + m * p.sCm;
      ${[0, 1, 2, 3].map((j) => `{
        let n = n0 + tn0 + ${j}u;
        if (n < p.n) {
          var v = r0.${'xyzw'[j]};
          if (p.bias != 0u) { v += bias[b0 * p.n + n]; }
          if (p.act == 1u) { v = max(v, 0.0); }
          if (p.act == 2u) { v = 1.0 / (1.0 + exp(-v)); }
          let at = rowC + n * p.sCn;
          ${gate ? `let gv = f32(gsc[p.gOff + b0 * p.sC0 + b1 * p.sC1 + m * p.gStride + n]);
          if (p.gSigm == 2u) {
            c[at] = ${EC}(v + gv);
          } else {
            let gf = select(gv, 1.0 / (1.0 + exp(-gv)), p.gSigm != 0u);
            var vg = v * gf;
            if (p.accumulate != 0u) { vg += f32(c[at]); }
            c[at] = ${EC}(vg);
          }` : `if (p.accumulate != 0u) { v += f32(c[at]); }
          c[at] = ${EC}(v);`}
        }
      }`).join('\n      ')}
    }
  }`).join('\n  ')}
}`;
}

/**
 * Tensor Core (`chromium_experimental_subgroup_matrix`) `tileM x 128` GEMM (`SGY = 1, SGX = 8`, 256 threads).
 * Supports `tileM = 64` (4 row blocks per subgroup) and `tileM = 32` (2 row blocks per subgroup), as well as
 * packed `u32` (`vec2<f16>`) weights (`half = true`) and float32 activation `b` (`half = false`).
 */
function gemmSubmatSource(pair = null, gate = false, half = true, tileM = 64, vecIO = false, aF16 = false, gF16 = false, cF16 = false, addOnly = false, gSigm = true, hasAlpha = false, hasBias = false, actType = 0, acc = false, vecB = vecIO, vecEpilogue = false, inLn = false, sm8 = false) {
  const numBlocks = tileM / 16;
  const blocks = Array.from({ length: numBlocks }, (_, i) => i);
  const numPairs = tileM / 32;
  const pairs = Array.from({ length: numPairs }, (_, i) => i);
  const aPerThread = tileM / 16;
  const aThreadsPerRow = 16 / aPerThread;
  const sm8f32 = sm8 && pair?.colMajorStore !== 'cond1024' && !addOnly && actType !== 1 && (!gate || gSigm) && !(!pair && aF16 && cF16 && !acc);
  const stType = sm8f32 ? 'f32' : 'f16';
  const aVecType = aPerThread === 4 ? `vec4<${stType}>` : `vec2<${stType}>`;
  const cType = (pair?.cF16 || cF16) ? 'f16' : 'f32';
  const scType = sm8 && !sm8f32 ? 'f16' : 'f32';
  const stA = sm8 ? 'scratch' : 'staged_a';
  const stB = sm8 ? 'scratch' : 'staged_b';
  const aBaseOff = sm8 ? '2048u + ' : '';
  const lnOff = 2048 + tileM * 24;
  const scratchSize = sm8 && (!pair || pair.singleBlockBody) ? (inLn && sm8f32 ? lnOff + tileM * 2 : Math.max(lnOff, 2176)) : 4352;
  const mmaTile = sm8
    ? [0, 8].map((ks) => `{
      let right0 = subgroupMatrixLoad<subgroup_matrix_right<${stType}, 8, 8>>(&${stB}, sg * 256u + ${ks}u, true, 16u);
      let right1 = subgroupMatrixLoad<subgroup_matrix_right<${stType}, 8, 8>>(&${stB}, sg * 256u + ${128 + ks}u, true, 16u);
      ${blocks.map((i) => `{
        let left0 = subgroupMatrixLoad<subgroup_matrix_left<${stType}, 8, 8>>(&${stA}, ${2048 + i * 384 + ks}u, false, 24u);
        let left1 = subgroupMatrixLoad<subgroup_matrix_left<${stType}, 8, 8>>(&${stA}, ${2048 + i * 384 + 192 + ks}u, false, 24u);
        mAcc_${i}_00 = subgroupMatrixMultiplyAccumulate(left0, right0, mAcc_${i}_00);
        mAcc_${i}_01 = subgroupMatrixMultiplyAccumulate(left0, right1, mAcc_${i}_01);
        mAcc_${i}_10 = subgroupMatrixMultiplyAccumulate(left1, right0, mAcc_${i}_10);
        mAcc_${i}_11 = subgroupMatrixMultiplyAccumulate(left1, right1, mAcc_${i}_11);
      }`).join('\n      ')}
    }`).join('\n    ')
    : `let right = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&staged_b, sg * 256u, true, 16u);
    ${blocks.map((i) => `{
      let left = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&staged_a, ${i * 384}u, false, 24u);
      mAcc_${i} = subgroupMatrixMultiplyAccumulate(left, right, mAcc_${i});
    }`).join('\n    ')}`;
  const stageNext = `${tileM === 32 ? 'if (li < 128u) {' : '{'}
      ${aF16 ? `let av = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_a.x), bitcast<vec2<f16>>(next_a.y)));` : `let av = vec4<${stType}>(next_a);`}
      ${inLn ? 'let af = vec4<f32>(av); sumA += af.x + af.y + af.z + af.w; sqSumA += dot(af, af);' : ''}
      ${stA}[aWrite] = av.x;
      ${stA}[aWrite + 1u] = av.y;
      ${stA}[aWrite + 2u] = av.z;
      ${stA}[aWrite + 3u] = av.w;
    }
    {
      let p0 = ${sm8f32 ? 'unpack2x16float(next_b.x)' : 'bitcast<vec2<f16>>(next_b.x)'};
      let p1 = ${sm8f32 ? 'unpack2x16float(next_b.y)' : 'bitcast<vec2<f16>>(next_b.y)'};
      let p2 = ${sm8f32 ? 'unpack2x16float(next_b.z)' : 'bitcast<vec2<f16>>(next_b.z)'};
      let p3 = ${sm8f32 ? 'unpack2x16float(next_b.w)' : 'bitcast<vec2<f16>>(next_b.w)'};
      ${stB}[bDst] = p0.x;
      ${stB}[bDst + 1u] = p0.y;
      ${stB}[bDst + 2u] = p1.x;
      ${stB}[bDst + 3u] = p1.y;
      ${stB}[bDst + 4u] = p2.x;
      ${stB}[bDst + 5u] = p2.y;
      ${stB}[bDst + 6u] = p3.x;
      ${stB}[bDst + 7u] = p3.y;
    }`;
  const storeMat16 = (i, base, colMajor, stride) => {
    if (!sm8) {
      return `subgroupMatrixStore(&scratch, ${base}, mAcc_${i}, ${colMajor}, ${stride}u);`;
    }
    if (colMajor) {
      return `subgroupMatrixStore(&scratch, ${base}, mAcc_${i}_00, true, ${stride}u);
    subgroupMatrixStore(&scratch, ${base} + ${8 * stride}u, mAcc_${i}_01, true, ${stride}u);
    subgroupMatrixStore(&scratch, ${base} + 8u, mAcc_${i}_10, true, ${stride}u);
    subgroupMatrixStore(&scratch, ${base} + ${8 * stride + 8}u, mAcc_${i}_11, true, ${stride}u);`;
    }
    return `subgroupMatrixStore(&scratch, ${base}, mAcc_${i}_00, false, ${stride}u);
    subgroupMatrixStore(&scratch, ${base} + 8u, mAcc_${i}_01, false, ${stride}u);
    subgroupMatrixStore(&scratch, ${base} + ${8 * stride}u, mAcc_${i}_10, false, ${stride}u);
    subgroupMatrixStore(&scratch, ${base} + ${8 * stride + 8}u, mAcc_${i}_11, false, ${stride}u);`;
  };
  return `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
struct P { m: u32, n: u32, k: u32, b1: u32,
  offA: u32, sA0: u32, sA1: u32, sAm: u32, sAk: u32,
  offB: u32, sB0: u32, sB1: u32, sBn: u32, sBk: u32,
  offC: u32, sC0: u32, sC1: u32, sCm: u32, sCn: u32,
  bias: u32, act: u32, accumulate: u32, alpha: f32, batches: u32,
  gOff: u32, gStride: u32, gSigm: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<${vecIO ? (aF16 ? 'vec2<u32>' : 'vec4<f32>') : (aF16 ? 'f16' : 'f32')}>;
@group(0) @binding(2) var<storage, read> b: array<${vecB ? 'vec4<u32>' : (half ? 'u32' : (aF16 ? 'f16' : 'f32'))}>;
@group(0) @binding(3) var<storage, read_write> c: array<${vecEpilogue ? (cF16 ? 'vec2<u32>' : 'vec4<f32>') : (pair?.cVec4 ? (cF16 ? 'vec2<u32>' : 'vec4<f32>') : cType)}>;
@group(0) @binding(4) var<storage, read> bias: array<${vecEpilogue ? 'vec4<f32>' : 'f32'}>;
${gate ? `@group(0) @binding(5) var<storage, read> gsc: array<${vecEpilogue ? (gF16 ? 'vec2<u32>' : 'vec4<f32>') : (gF16 ? 'f16' : 'f32')}>;` : ''}
${vecEpilogue && (cF16 || gF16) ? `fn unpack4f16(u: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(vec4<f16>(bitcast<vec2<f16>>(u.x), bitcast<vec2<f16>>(u.y)));
}
fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}` : ''}
${pair ? `${pair.decls ?? ''}
fn bRow(v: u32) -> u32 { ${pair.bRow ?? 'return v;'} }
${pair.body}` : 'fn bRow(v: u32) -> u32 { return v; }'}
${sm8 ? '' : `var<workgroup> staged_a: array<f16, ${tileM * 24}>;
var<workgroup> staged_b: array<f16, 2048>;`}
var<workgroup> scratch: array<${scType}, ${scratchSize}>;
${inLn && sm8 && !sm8f32 ? `var<workgroup> lnStat: array<f32, ${tileM * 2}>;` : ''}
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
  @builtin(subgroup_id) sg: u32
) {
  let tilesN = (p.n + 127u) / 128u;
  let tilesM = (p.m + ${tileM - 1}u) / ${tileM}u;
  let flat = wid.x + wid.y * 65535u;
  ${pair ? '_ = bias[0];' : ''}
  let batch = flat / (tilesN * tilesM);
  let inBatch = batch < p.batches;
  let batchClamped = min(batch, p.batches - 1u);
  let tile = flat % (tilesN * tilesM);
  let m0 = (tile / tilesN) * ${tileM}u;
  let n0 = (tile % tilesN) * 128u;
  let b0 = batchClamped / p.b1;
  let b1 = batchClamped % p.b1;
  let baseA = p.offA + b0 * p.sA0 + b1 * p.sA1;
  let baseB = p.offB + b0 * p.sB0 + b1 * p.sB1;

  ${vecIO ? `let rA = li / 4u;
  let kA4 = li % 4u;
  let rowA = min(m0 + rA, p.m - 1u);
  let aRowBase4 = (baseA + rowA * p.sAm) / 4u + kA4;
  let aWrite = ${aBaseOff}rA * 24u + kA4 * 4u;` : `let rA = li / ${aThreadsPerRow}u;
  let kA = (li % ${aThreadsPerRow}u) * ${aPerThread}u;
  let rowA = m0 + rA;
  let aValid = inBatch && (rowA < p.m);
  let aRowBase = baseA + rowA * p.sAm;
  let aWrite = ${aBaseOff}rA * 24u + kA;`}

  let cB = li / 2u;
  let kkB = (li & 1u) * 8u;
  let rowB = ${vecB ? 'min(n0 + cB, p.n - 1u)' : 'n0 + cB'};
  ${vecB ? '' : 'let bValid = inBatch && (rowB < p.n);'}
  ${vecB
    ? 'let bRowBase8 = (baseB + bRow(rowB) * p.sBn) / 8u + (li & 1u);'
    : (half ? 'let bRowBase = (baseB + bRow(rowB) * p.sBn + kkB) / 2u;' : 'let bRowBase = baseB + bRow(rowB) * p.sBn;')}

  ${blocks.map((i) => sm8
    ? `var mAcc_${i}_00 = subgroup_matrix_result<${stType}, 8, 8>();
  var mAcc_${i}_01 = subgroup_matrix_result<${stType}, 8, 8>();
  var mAcc_${i}_10 = subgroup_matrix_result<${stType}, 8, 8>();
  var mAcc_${i}_11 = subgroup_matrix_result<${stType}, 8, 8>();`
    : `var mAcc_${i} = subgroup_matrix_result<f32, 16, 16>();`).join('\n  ')}

  let bDst = li * 8u;
  ${inLn ? 'var sumA = 0.0;\n  var sqSumA = 0.0;' : ''}
  ${vecIO ? `var next_a = ${aF16 ? 'vec2<u32>(0u)' : 'vec4<f32>(0.0)'};` : `var next_a = ${aVecType}(0.0);`}
  ${vecB ? 'var next_b = vec4<u32>(0u);' : `var next_p0 = vec2<${stType}>(0.0);
  var next_p1 = vec2<${stType}>(0.0);
  var next_p2 = vec2<${stType}>(0.0);
  var next_p3 = vec2<${stType}>(0.0);`}
  {
    ${vecIO ? `${tileM === 32 ? 'if (li < 128u) {' : '{'}
      let a0 = a[aRowBase4];
      ${aF16 ? `let av = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(a0.x), bitcast<vec2<f16>>(a0.y)));` : `let av = vec4<${stType}>(a0);`}
      ${inLn ? 'let af = vec4<f32>(av); sumA += af.x + af.y + af.z + af.w; sqSumA += dot(af, af);' : ''}
      ${stA}[aWrite] = av.x;
      ${stA}[aWrite + 1u] = av.y;
      ${stA}[aWrite + 2u] = av.z;
      ${stA}[aWrite + 3u] = av.w;
    }` : `let fullK0 = 16u <= p.k;
    if (aValid) {
      ${aPerThread === 4 ? `if (fullK0 || kA + 3u < p.k) {
        let aAddr = aRowBase + kA * p.sAk;
        next_a = vec4<${stType}>(${stType}(a[aAddr]), ${stType}(a[aAddr + p.sAk]), ${stType}(a[aAddr + 2u * p.sAk]), ${stType}(a[aAddr + 3u * p.sAk]));
      } else {
        if (kA < p.k) { next_a.x = ${stType}(a[aRowBase + kA * p.sAk]); }
        if (kA + 1u < p.k) { next_a.y = ${stType}(a[aRowBase + (kA + 1u) * p.sAk]); }
        if (kA + 2u < p.k) { next_a.z = ${stType}(a[aRowBase + (kA + 2u) * p.sAk]); }
      }` : `if (fullK0 || kA + 1u < p.k) {
        let aAddr = aRowBase + kA * p.sAk;
        next_a = vec2<${stType}>(${stType}(a[aAddr]), ${stType}(a[aAddr + p.sAk]));
      } else {
        if (kA < p.k) { next_a.x = ${stType}(a[aRowBase + kA * p.sAk]); }
      }`}
    }
    ${stA}[aWrite] = next_a.x;
    ${stA}[aWrite + 1u] = next_a.y;
    ${aPerThread === 4 ? `${stA}[aWrite + 2u] = next_a.z;
    ${stA}[aWrite + 3u] = next_a.w;` : ''}`}
    ${vecB ? `{
      let w4 = b[bRowBase8];
      let p0 = ${sm8f32 ? 'unpack2x16float(w4.x)' : 'bitcast<vec2<f16>>(w4.x)'};
      let p1 = ${sm8f32 ? 'unpack2x16float(w4.y)' : 'bitcast<vec2<f16>>(w4.y)'};
      let p2 = ${sm8f32 ? 'unpack2x16float(w4.z)' : 'bitcast<vec2<f16>>(w4.z)'};
      let p3 = ${sm8f32 ? 'unpack2x16float(w4.w)' : 'bitcast<vec2<f16>>(w4.w)'};
      ${stB}[bDst] = p0.x;
      ${stB}[bDst + 1u] = p0.y;
      ${stB}[bDst + 2u] = p1.x;
      ${stB}[bDst + 3u] = p1.y;
      ${stB}[bDst + 4u] = p2.x;
      ${stB}[bDst + 5u] = p2.y;
      ${stB}[bDst + 6u] = p3.x;
      ${stB}[bDst + 7u] = p3.y;
    }` : `if (bValid) {
      ${half ? `if (fullK0) {
        next_p0 = ${sm8f32 ? 'unpack2x16float(b[bRowBase])' : 'bitcast<vec2<f16>>(b[bRowBase])'};
        next_p1 = ${sm8f32 ? 'unpack2x16float(b[bRowBase + 1u])' : 'bitcast<vec2<f16>>(b[bRowBase + 1u])'};
        next_p2 = ${sm8f32 ? 'unpack2x16float(b[bRowBase + 2u])' : 'bitcast<vec2<f16>>(b[bRowBase + 2u])'};
        next_p3 = ${sm8f32 ? 'unpack2x16float(b[bRowBase + 3u])' : 'bitcast<vec2<f16>>(b[bRowBase + 3u])'};
      } else {
        if (kkB < p.k) { let t = ${sm8f32 ? 'unpack2x16float(b[bRowBase])' : 'bitcast<vec2<f16>>(b[bRowBase])'}; next_p0.x = t.x; if (kkB + 1u < p.k) { next_p0.y = t.y; } }
        if (kkB + 2u < p.k) { let t = ${sm8f32 ? 'unpack2x16float(b[bRowBase + 1u])' : 'bitcast<vec2<f16>>(b[bRowBase + 1u])'}; next_p1.x = t.x; if (kkB + 3u < p.k) { next_p1.y = t.y; } }
        if (kkB + 4u < p.k) { let t = ${sm8f32 ? 'unpack2x16float(b[bRowBase + 2u])' : 'bitcast<vec2<f16>>(b[bRowBase + 2u])'}; next_p2.x = t.x; if (kkB + 5u < p.k) { next_p2.y = t.y; } }
        if (kkB + 6u < p.k) { let t = ${sm8f32 ? 'unpack2x16float(b[bRowBase + 3u])' : 'bitcast<vec2<f16>>(b[bRowBase + 3u])'}; next_p3.x = t.x; if (kkB + 7u < p.k) { next_p3.y = t.y; } }
      }` : `let bOff = bRowBase + kkB * p.sBk;
      if (fullK0 || kkB + 7u < p.k) {
        next_p0 = vec2<${stType}>(${stType}(b[bOff]), ${stType}(b[bOff + p.sBk]));
        next_p1 = vec2<${stType}>(${stType}(b[bOff + 2u * p.sBk]), ${stType}(b[bOff + 3u * p.sBk]));
        next_p2 = vec2<${stType}>(${stType}(b[bOff + 4u * p.sBk]), ${stType}(b[bOff + 5u * p.sBk]));
        next_p3 = vec2<${stType}>(${stType}(b[bOff + 6u * p.sBk]), ${stType}(b[bOff + 7u * p.sBk]));
      } else {
        if (kkB < p.k) { next_p0.x = ${stType}(b[bOff]); }
        if (kkB + 1u < p.k) { next_p0.y = ${stType}(b[bOff + p.sBk]); }
        if (kkB + 2u < p.k) { next_p1.x = ${stType}(b[bOff + 2u * p.sBk]); }
        if (kkB + 3u < p.k) { next_p1.y = ${stType}(b[bOff + 3u * p.sBk]); }
        if (kkB + 4u < p.k) { next_p2.x = ${stType}(b[bOff + 4u * p.sBk]); }
        if (kkB + 5u < p.k) { next_p2.y = ${stType}(b[bOff + 5u * p.sBk]); }
        if (kkB + 6u < p.k) { next_p3.x = ${stType}(b[bOff + 6u * p.sBk]); }
        if (kkB + 7u < p.k) { next_p3.y = ${stType}(b[bOff + 7u * p.sBk]); }
      }`}
    }
    ${stB}[bDst] = next_p0.x;
    ${stB}[bDst + 1u] = next_p0.y;
    ${stB}[bDst + 2u] = next_p1.x;
    ${stB}[bDst + 3u] = next_p1.y;
    ${stB}[bDst + 4u] = next_p2.x;
    ${stB}[bDst + 5u] = next_p2.y;
    ${stB}[bDst + 6u] = next_p3.x;
    ${stB}[bDst + 7u] = next_p3.y;`}
  }
  workgroupBarrier();

  ${vecIO && vecB ? `for (var kNext = 16u; kNext < p.k; kNext += 16u) {
    next_a = a[aRowBase4 + kNext / 4u];
    next_b = b[bRowBase8 + kNext / 8u];
    ${mmaTile}
    workgroupBarrier();
    ${stageNext}
    workgroupBarrier();
  }
  ${mmaTile}
  workgroupBarrier();` : `for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    let kNext = k0 + 16u;
    let hasNext = kNext < p.k;
    if (hasNext) {
      ${vecIO ? `next_a = a[aRowBase4 + kNext / 4u];` : `let fullK = kNext + 16u <= p.k;
      next_a = ${aVecType}(0.0);
      if (aValid) {
        let k = kNext + kA;
        ${aPerThread === 4 ? `if (fullK || k + 3u < p.k) {
          let aAddr = aRowBase + k * p.sAk;
          next_a = vec4<${stType}>(${stType}(a[aAddr]), ${stType}(a[aAddr + p.sAk]), ${stType}(a[aAddr + 2u * p.sAk]), ${stType}(a[aAddr + 3u * p.sAk]));
        } else {
          if (k < p.k) { next_a.x = ${stType}(a[aRowBase + k * p.sAk]); }
          if (k + 1u < p.k) { next_a.y = ${stType}(a[aRowBase + (k + 1u) * p.sAk]); }
          if (k + 2u < p.k) { next_a.z = ${stType}(a[aRowBase + (k + 2u) * p.sAk]); }
        }` : `if (fullK || k + 1u < p.k) {
          let aAddr = aRowBase + k * p.sAk;
          next_a = vec2<${stType}>(${stType}(a[aAddr]), ${stType}(a[aAddr + p.sAk]));
        } else {
          if (k < p.k) { next_a.x = ${stType}(a[aRowBase + k * p.sAk]); }
        }`}
      }`}
      ${vecB ? `next_b = b[bRowBase8 + kNext / 8u];` : `next_p0 = vec2<${stType}>(0.0);
      next_p1 = vec2<${stType}>(0.0);
      next_p2 = vec2<${stType}>(0.0);
      next_p3 = vec2<${stType}>(0.0);
      if (bValid) {
        ${half ? `let bOff = bRowBase + kNext / 2u;
        if (fullK) {
          next_p0 = ${sm8f32 ? 'unpack2x16float(b[bOff])' : 'bitcast<vec2<f16>>(b[bOff])'};
          next_p1 = ${sm8f32 ? 'unpack2x16float(b[bOff + 1u])' : 'bitcast<vec2<f16>>(b[bOff + 1u])'};
          next_p2 = ${sm8f32 ? 'unpack2x16float(b[bOff + 2u])' : 'bitcast<vec2<f16>>(b[bOff + 2u])'};
          next_p3 = ${sm8f32 ? 'unpack2x16float(b[bOff + 3u])' : 'bitcast<vec2<f16>>(b[bOff + 3u])'};
        } else {
          let k = kNext + kkB;
          if (k < p.k) { let t = ${sm8f32 ? 'unpack2x16float(b[bOff])' : 'bitcast<vec2<f16>>(b[bOff])'}; next_p0.x = t.x; if (k + 1u < p.k) { next_p0.y = t.y; } }
          if (k + 2u < p.k) { let t = ${sm8f32 ? 'unpack2x16float(b[bOff + 1u])' : 'bitcast<vec2<f16>>(b[bOff + 1u])'}; next_p1.x = t.x; if (k + 3u < p.k) { next_p1.y = t.y; } }
          if (k + 4u < p.k) { let t = ${sm8f32 ? 'unpack2x16float(b[bOff + 2u])' : 'bitcast<vec2<f16>>(b[bOff + 2u])'}; next_p2.x = t.x; if (k + 5u < p.k) { next_p2.y = t.y; } }
          if (k + 6u < p.k) { let t = ${sm8f32 ? 'unpack2x16float(b[bOff + 3u])' : 'bitcast<vec2<f16>>(b[bOff + 3u])'}; next_p3.x = t.x; if (k + 7u < p.k) { next_p3.y = t.y; } }
        }` : `let k = kNext + kkB;
        let bOff = bRowBase + k * p.sBk;
        if (fullK || k + 7u < p.k) {
          next_p0 = vec2<${stType}>(${stType}(b[bOff]), ${stType}(b[bOff + p.sBk]));
          next_p1 = vec2<${stType}>(${stType}(b[bOff + 2u * p.sBk]), ${stType}(b[bOff + 3u * p.sBk]));
          next_p2 = vec2<${stType}>(${stType}(b[bOff + 4u * p.sBk]), ${stType}(b[bOff + 5u * p.sBk]));
          next_p3 = vec2<${stType}>(${stType}(b[bOff + 6u * p.sBk]), ${stType}(b[bOff + 7u * p.sBk]));
        } else {
          if (k < p.k) { next_p0.x = ${stType}(b[bOff]); }
          if (k + 1u < p.k) { next_p0.y = ${stType}(b[bOff + p.sBk]); }
          if (k + 2u < p.k) { next_p1.x = ${stType}(b[bOff + 2u * p.sBk]); }
          if (k + 3u < p.k) { next_p1.y = ${stType}(b[bOff + 3u * p.sBk]); }
          if (k + 4u < p.k) { next_p2.x = ${stType}(b[bOff + 4u * p.sBk]); }
          if (k + 5u < p.k) { next_p2.y = ${stType}(b[bOff + 5u * p.sBk]); }
          if (k + 6u < p.k) { next_p3.x = ${stType}(b[bOff + 6u * p.sBk]); }
          if (k + 7u < p.k) { next_p3.y = ${stType}(b[bOff + 7u * p.sBk]); }
        }`}
      }`}
    }
    ${mmaTile}
    workgroupBarrier();
    if (hasNext) {
      ${vecIO ? `${tileM === 32 ? 'if (li < 128u) {' : '{'}
        ${aF16 ? `let av = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_a.x), bitcast<vec2<f16>>(next_a.y)));` : `let av = vec4<${stType}>(next_a);`}
        ${inLn ? 'let af = vec4<f32>(av); sumA += af.x + af.y + af.z + af.w; sqSumA += dot(af, af);' : ''}
        ${stA}[aWrite] = av.x;
        ${stA}[aWrite + 1u] = av.y;
        ${stA}[aWrite + 2u] = av.z;
        ${stA}[aWrite + 3u] = av.w;
      }` : `${stA}[aWrite] = next_a.x;
      ${stA}[aWrite + 1u] = next_a.y;
      ${aPerThread === 4 ? `${stA}[aWrite + 2u] = next_a.z;
      ${stA}[aWrite + 3u] = next_a.w;` : ''}`}
      ${vecB ? `{
        let p0 = ${sm8f32 ? 'unpack2x16float(next_b.x)' : 'bitcast<vec2<f16>>(next_b.x)'};
        let p1 = ${sm8f32 ? 'unpack2x16float(next_b.y)' : 'bitcast<vec2<f16>>(next_b.y)'};
        let p2 = ${sm8f32 ? 'unpack2x16float(next_b.z)' : 'bitcast<vec2<f16>>(next_b.z)'};
        let p3 = ${sm8f32 ? 'unpack2x16float(next_b.w)' : 'bitcast<vec2<f16>>(next_b.w)'};
        ${stB}[bDst] = p0.x;
        ${stB}[bDst + 1u] = p0.y;
        ${stB}[bDst + 2u] = p1.x;
        ${stB}[bDst + 3u] = p1.y;
        ${stB}[bDst + 4u] = p2.x;
        ${stB}[bDst + 5u] = p2.y;
        ${stB}[bDst + 6u] = p3.x;
        ${stB}[bDst + 7u] = p3.y;
      }` : `${stB}[bDst] = next_p0.x;
      ${stB}[bDst + 1u] = next_p0.y;
      ${stB}[bDst + 2u] = next_p1.x;
      ${stB}[bDst + 3u] = next_p1.y;
      ${stB}[bDst + 4u] = next_p2.x;
      ${stB}[bDst + 5u] = next_p2.y;
      ${stB}[bDst + 6u] = next_p3.x;
      ${stB}[bDst + 7u] = next_p3.y;`}
    }
    workgroupBarrier();
  }`}
  ${inLn ? `{
    sumA += subgroupShuffleXor(sumA, 1u);
    sumA += subgroupShuffleXor(sumA, 2u);
    sqSumA += subgroupShuffleXor(sqSumA, 1u);
    sqSumA += subgroupShuffleXor(sqSumA, 2u);
    if (kA4 == 0u && rA < ${tileM}u) {
      let invK = 1.0 / f32(p.k);
      let mean = sumA * invK;
      let invStd = inverseSqrt(max(sqSumA * invK - mean * mean, 0.0) + 1e-5);
      ${sm8 && !sm8f32 ? `lnStat[rA * 2u] = mean;
      lnStat[rA * 2u + 1u] = invStd;` : `let scIdx = ${sm8f32
        ? `${lnOff}u + rA * 2u`
        : pair?.colMajorStore === 'cond1024'
          ? 'select((rA / 4u) * 136u + 128u + (rA % 4u) * 2u, rA * 34u + 32u, n0 < 1024u)'
          : '(rA / 4u) * 136u + 128u + (rA % 4u) * 2u'};
      scratch[scIdx] = mean;
      scratch[scIdx + 1u] = invStd;`}
    }
  }` : ''}
  ${vecEpilogue ? `let rE = li / 32u;
  let c4 = li % 32u;
  let nCol = n0 + c4 * 4u;
  let colValid = inBatch && (nCol < p.n);
  let baseC = p.offC + b0 * p.sC0 + b1 * p.sC1;
  ${!hasBias ? '_ = bias[0];' : `var bVal4 = vec4<f32>(0.0);
  ${inLn ? 'var wSum4 = vec4<f32>(0.0);' : ''}
  if (colValid) {
    bVal4 = bias[(b0 * p.n + nCol) / 4u];
    ${inLn ? 'wSum4 = bias[(p.n + nCol) / 4u];' : ''}
  }`}` : `let rBase = li / 64u;
  let col = li % 64u;
  let baseC = p.offC + b0 * p.sC0 + b1 * p.sC1;
  ${pair || !hasBias ? '_ = bias[0];' : `var bVal0 = 0.0;
  var bVal1 = 0.0;
  if (inBatch) {
    if (n0 + col < p.n) { bVal0 = bias[b0 * p.n + n0 + col]; }
    if (n0 + col + 64u < p.n) { bVal1 = bias[b0 * p.n + n0 + col + 64u]; }
  }`}`}
  ${sm8 && (!pair || pair.singleBlockBody) ? blocks.map((bIdx) => {
    const storeCode = pair?.colMajorStore === 'cond1024'
      ? `if (n0 < 1024u) {
    ${storeMat16(bIdx, 'sg * 288u', true, 18)}
  } else {
    ${storeMat16(bIdx, 'sg * 16u', false, 136)}
  }`
      : storeMat16(bIdx, 'sg * 16u', false, 136);
    return `${storeCode}
  workgroupBarrier();
  ${bIdx + 1 === numBlocks ? 'if (!inBatch) { return; }\n  {' : 'if (inBatch) {'}
    ${pair?.singleBlockBody ? pair.singleBlockBody(bIdx) : vecEpilogue ? `{
      if (colValid) {
        ${[0, 1].map((s) => `{
          let rLocal = ${bIdx * 16 + s * 8}u + rE;
          let m = m0 + rLocal;
          if (m < p.m) {
            let sb = (rE + ${s * 8}u) * 136u + c4 * 4u;
            let raw4 = vec4<f32>(f32(scratch[sb]), f32(scratch[sb + 1u]), f32(scratch[sb + 2u]), f32(scratch[sb + 3u]));
            ${inLn ? (sm8 && !sm8f32
              ? `let v4 = (raw4 - lnStat[rLocal * 2u] * wSum4) * lnStat[rLocal * 2u + 1u]${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bVal4' : ''};`
              : `let scIdx = ${lnOff}u + rLocal * 2u;
            let v4 = (raw4 - f32(scratch[scIdx]) * wSum4) * f32(scratch[scIdx + 1u])${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bVal4' : ''};`) : `let v4 = raw4${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bVal4' : ''};`}
            let v = ${actType === 1 ? 'max(v4, vec4<f32>(0.0))' : actType === 2 ? 'vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v4))' : 'v4'};
            let at4 = (baseC + m * p.sCm + nCol) / 4u;
            ${gate ? `let gv = ${gF16 ? 'unpack4f16(gsc[(p.gOff + m * p.gStride + nCol) / 4u])' : 'gsc[(p.gOff + m * p.gStride + nCol) / 4u]'};
            ${addOnly ? 'let res = v + gv;' : gSigm ? `let res = ${cF16 ? 'unpack4f16(c[at4])' : 'c[at4]'} + v / (vec4<f32>(1.0) + exp(-gv));` : `let res = ${cF16 ? 'unpack4f16(c[at4])' : 'c[at4]'} + v * gv;`}
            c[at4] = ${cF16 ? 'pack4f16(res)' : 'res'};` : acc ? `let res = ${cF16 ? 'unpack4f16(c[at4])' : 'c[at4]'} + v;
            c[at4] = ${cF16 ? 'pack4f16(res)' : 'res'};` : `c[at4] = ${cF16 ? 'pack4f16(v)' : 'v'};`}
          }
        }`).join('\n        ')}
      }
    }` : `{
      ${[0, 1, 2, 3].map((s) => `let sb_${s} = (rBase + ${s * 4}u) * 136u + col;
      let v0_${s} = f32(scratch[sb_${s}])${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bVal0' : ''};
      let v1_${s} = f32(scratch[sb_${s} + 64u])${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bVal1' : ''};`).join('\n      ')}
      ${[0, 1, 2, 3].map((s) => `{
        let m = m0 + ${bIdx * 16 + s * 4}u + rBase;
        if (m < p.m) {
          let v0 = v0_${s};
          let v1 = v1_${s};
          let rowC = baseC + m * p.sCm;
          let n_0 = n0 + col;
          if (n_0 < p.n) {
            let v = ${actType === 1 ? 'max(v0, 0.0)' : actType === 2 ? '1.0 / (1.0 + exp(-v0))' : 'v0'};
            let at = rowC + n_0 * p.sCn;
            ${gate ? (addOnly ? `let gv = f32(gsc[p.gOff + m * p.gStride + n_0]);
            c[at] = ${cType}(v + gv);` : gSigm ? `let gv = f32(gsc[p.gOff + m * p.gStride + n_0]);
            c[at] = ${cType}(f32(c[at]) + v / (1.0 + exp(-gv)));` : `let gv = f32(gsc[p.gOff + m * p.gStride + n_0]);
            c[at] = ${cType}(f32(c[at]) + v * gv);`) : acc ? `c[at] = ${cType}(f32(c[at]) + v);` : `c[at] = ${cType}(v);`}
          }
          let n_1 = n_0 + 64u;
          if (n_1 < p.n) {
            let v = ${actType === 1 ? 'max(v1, 0.0)' : actType === 2 ? '1.0 / (1.0 + exp(-v1))' : 'v1'};
            let at = rowC + n_1 * p.sCn;
            ${gate ? (addOnly ? `let gv = f32(gsc[p.gOff + m * p.gStride + n_1]);
            c[at] = ${cType}(v + gv);` : gSigm ? `let gv = f32(gsc[p.gOff + m * p.gStride + n_1]);
            c[at] = ${cType}(f32(c[at]) + v / (1.0 + exp(-gv)));` : `let gv = f32(gsc[p.gOff + m * p.gStride + n_1]);
            c[at] = ${cType}(f32(c[at]) + v * gv);`) : acc ? `c[at] = ${cType}(f32(c[at]) + v);` : `c[at] = ${cType}(v);`}
          }
        }
      }`).join('\n      ')}
    }`}
  }${bIdx + 1 < numBlocks ? '\n  workgroupBarrier();' : ''}`;
  }).join('\n  ') : pairs.map((pIdx) => {
    const i0 = pIdx * 2;
    const i1 = pIdx * 2 + 1;
    const storeCode = pair?.colMajorStore === 'cond1024'
      ? `if (n0 < 1024u) {
    ${storeMat16(i0, 'sg * 544u', true, 34)}
    ${storeMat16(i1, 'sg * 544u + 16u', true, 34)}
  } else {
    ${storeMat16(i0, 'sg * 16u', false, 136)}
    ${storeMat16(i1, '2176u + sg * 16u', false, 136)}
  }`
      : pair?.colMajorStore
        ? `${storeMat16(i0, 'sg * 544u', true, 34)}
  ${storeMat16(i1, 'sg * 544u + 16u', true, 34)}`
        : `${storeMat16(i0, 'sg * 16u', false, 136)}
  ${storeMat16(i1, '2176u + sg * 16u', false, 136)}`;
    return `${storeCode}
  workgroupBarrier();
  ${pIdx + 1 === numPairs ? 'if (!inBatch) { return; }\n  {' : 'if (inBatch) {'}
    ${pair?.blockBody ? pair.blockBody(pIdx) : vecEpilogue ? `{
      if (colValid) {
        ${[0, 1, 2, 3].map((s) => `{
          let rLocal = ${pIdx * 32 + s * 8}u + rE;
          let m = m0 + rLocal;
          if (m < p.m) {
            let sb = (rE + ${s * 8}u) * 136u + c4 * 4u;
            let raw4 = vec4<f32>(f32(scratch[sb]), f32(scratch[sb + 1u]), f32(scratch[sb + 2u]), f32(scratch[sb + 3u]));
            ${inLn ? (sm8 ? `let v4 = (raw4 - lnStat[rLocal * 2u] * wSum4) * lnStat[rLocal * 2u + 1u]${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bVal4' : ''};` : `let scIdx = (rLocal / 4u) * 136u + 128u + (rLocal % 4u) * 2u;
            let v4 = (raw4 - scratch[scIdx] * wSum4) * scratch[scIdx + 1u]${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bVal4' : ''};`) : `let v4 = raw4${hasAlpha ? ' * p.alpha' : ''}${hasBias ? ' + bVal4' : ''};`}
            let v = ${actType === 1 ? 'max(v4, vec4<f32>(0.0))' : actType === 2 ? 'vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v4))' : 'v4'};
            let at4 = (baseC + m * p.sCm + nCol) / 4u;
            ${gate ? `let gv = ${gF16 ? 'unpack4f16(gsc[(p.gOff + m * p.gStride + nCol) / 4u])' : 'gsc[(p.gOff + m * p.gStride + nCol) / 4u]'};
            ${addOnly ? 'let res = v + gv;' : gSigm ? `let res = ${cF16 ? 'unpack4f16(c[at4])' : 'c[at4]'} + v / (vec4<f32>(1.0) + exp(-gv));` : `let res = ${cF16 ? 'unpack4f16(c[at4])' : 'c[at4]'} + v * gv;`}
            c[at4] = ${cF16 ? 'pack4f16(res)' : 'res'};` : acc ? `let res = ${cF16 ? 'unpack4f16(c[at4])' : 'c[at4]'} + v;
            c[at4] = ${cF16 ? 'pack4f16(res)' : 'res'};` : `c[at4] = ${cF16 ? 'pack4f16(v)' : 'v'};`}
          }
        }`).join('\n        ')}
      }
    }` : `{
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((s) => `let sb_${s} = (rBase + ${s * 4}u) * 136u + col;
      let v0_${s} = f32(scratch[sb_${s}])${hasAlpha ? ' * p.alpha' : ''}${!pair && hasBias ? ' + bVal0' : ''};
      let v1_${s} = f32(scratch[sb_${s} + 64u])${hasAlpha ? ' * p.alpha' : ''}${!pair && hasBias ? ' + bVal1' : ''};`).join('\n      ')}
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((s) => `{
        let m = m0 + ${pIdx * 32 + s * 4}u + rBase;
        if (m < p.m) {
          let v0 = v0_${s};
          let v1 = v1_${s};
          ${pair ? 'pairOutCol(m, n0, col, v0, v1);' : `let rowC = baseC + m * p.sCm;
          let n_0 = n0 + col;
          if (n_0 < p.n) {
            let v = ${actType === 1 ? 'max(v0, 0.0)' : actType === 2 ? '1.0 / (1.0 + exp(-v0))' : 'v0'};
            let at = rowC + n_0 * p.sCn;
            ${gate ? (addOnly ? `let gv = f32(gsc[p.gOff + m * p.gStride + n_0]);
            c[at] = ${cType}(v + gv);` : gSigm ? `let gv = f32(gsc[p.gOff + m * p.gStride + n_0]);
            c[at] = ${cType}(f32(c[at]) + v / (1.0 + exp(-gv)));` : `let gv = f32(gsc[p.gOff + m * p.gStride + n_0]);
            c[at] = ${cType}(f32(c[at]) + v * gv);`) : acc ? `c[at] = ${cType}(f32(c[at]) + v);` : `c[at] = ${cType}(v);`}
          }
          let n_1 = n_0 + 64u;
          if (n_1 < p.n) {
            let v = ${actType === 1 ? 'max(v1, 0.0)' : actType === 2 ? '1.0 / (1.0 + exp(-v1))' : 'v1'};
            let at = rowC + n_1 * p.sCn;
            ${gate ? (addOnly ? `let gv = f32(gsc[p.gOff + m * p.gStride + n_1]);
            c[at] = ${cType}(v + gv);` : gSigm ? `let gv = f32(gsc[p.gOff + m * p.gStride + n_1]);
            c[at] = ${cType}(f32(c[at]) + v / (1.0 + exp(-gv)));` : `let gv = f32(gsc[p.gOff + m * p.gStride + n_1]);
            c[at] = ${cType}(f32(c[at]) + v * gv);`) : acc ? `c[at] = ${cType}(f32(c[at]) + v);` : `c[at] = ${cType}(v);`}
          }`}
        }
      }`).join('\n      ')}
    }`}
  }${pIdx + 1 < numPairs ? '\n  workgroupBarrier();' : ''}`;
  }).join('\n  ')}
}`;
}

/**
 * Tensor Core (`chromium_experimental_subgroup_matrix`) `32 x 64` attention `P * V * sigm(G)` (`SGY = 2, SGX = 4`, 256 threads).
 * All 8 subgroups in the workgroup are 100% utilized across the 64 columns of a single head `h`, loading `V` (`[16, 64]`)
 * with 1 coalesced `vec4<f32>` per thread and writing `att` with 128-bit `vec4<f32>` stores.
 */
function attPvSource(vgF16 = false, sm8 = false) {
  const lt = vgF16 ? 'f16' : 'f32';
  const vt = vgF16 ? 'vec2<u32>' : 'vec4<f32>';
  const stType = sm8 ? 'f32' : 'f16';
  return `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
struct P { n: u32, nn: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> logits: array<${lt}>;
@group(0) @binding(2) var<storage, read> qkv: array<${vt}>;
@group(0) @binding(3) var<storage, read_write> att: array<${vt}>;
var<workgroup> staged_a: array<${stType}, 768>;
var<workgroup> staged_b: array<${stType}, 1152>;
var<workgroup> scratch: array<f32, 2304>;
${vgF16 ? `fn unpack4f16(u: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(vec4<f16>(bitcast<vec2<f16>>(u.x), bitcast<vec2<f16>>(u.y)));
}
fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}` : ''}
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
  @builtin(subgroup_id) sg: u32
) {
  let tilesM = (p.n + 31u) / 32u;
  let flat = wid.x + wid.y * 65535u;
  let h = flat / tilesM;
  let inBatch = h < 12u;
  let hClamped = min(h, 11u);
  let m0 = (flat % tilesM) * 32u;
  let sgM = sg / 4u;
  let sgN = sg % 4u;

  let rA = li / 8u;
  let kA = (li % 8u) * 2u;
  let rowA = min(m0 + rA, p.n - 1u);
  let aRowBase = hClamped * p.nn + rowA * p.n;
  let aWrite = rA * 24u + kA;

  let rV = li / 16u;
  let cV4 = li % 16u;
  let vColBase4 = ${vgF16 ? '0u' : '384u'} + hClamped * 16u + cV4;
  let bWrite = rV * 72u + cV4 * 4u;

  ${sm8 ? `var mAcc_00 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc_01 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc_10 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc_11 = subgroup_matrix_result<f32, 8, 8>();` : 'var mAcc = subgroup_matrix_result<f32, 16, 16>();'}
  var next_a0 = logits[aRowBase + min(kA, p.n - 1u)];
  var next_a1 = logits[aRowBase + min(kA + 1u, p.n - 1u)];
  var next_v4 = qkv[min(rV, p.n - 1u) * ${vgF16 ? '384u' : '768u'} + vColBase4];
  staged_a[aWrite] = select(${stType}(0.0), ${stType}(next_a0), kA < p.n);
  staged_a[aWrite + 1u] = select(${stType}(0.0), ${stType}(next_a1), kA + 1u < p.n);
  ${vgF16 ? `{
    let pv0 = ${sm8 ? 'unpack2x16float(next_v4.x)' : 'bitcast<vec2<f16>>(next_v4.x)'};
    let pv1 = ${sm8 ? 'unpack2x16float(next_v4.y)' : 'bitcast<vec2<f16>>(next_v4.y)'};
    staged_b[bWrite] = pv0.x;
    staged_b[bWrite + 1u] = pv0.y;
    staged_b[bWrite + 2u] = pv1.x;
    staged_b[bWrite + 3u] = pv1.y;
  }` : `staged_b[bWrite] = ${stType}(next_v4.x);
  staged_b[bWrite + 1u] = ${stType}(next_v4.y);
  staged_b[bWrite + 2u] = ${stType}(next_v4.z);
  staged_b[bWrite + 3u] = ${stType}(next_v4.w);`}
  workgroupBarrier();

  for (var k0 = 0u; k0 < p.n; k0 += 16u) {
    let kNext = k0 + 16u;
    let hasNext = kNext < p.n;
    if (hasNext) {
      next_a0 = logits[aRowBase + min(kNext + kA, p.n - 1u)];
      next_a1 = logits[aRowBase + min(kNext + kA + 1u, p.n - 1u)];
      next_v4 = qkv[min(kNext + rV, p.n - 1u) * ${vgF16 ? '384u' : '768u'} + vColBase4];
    }
    ${sm8 ? [0, 8].map((ks) => `{
      let left0 = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&staged_a, sgM * 384u + ${ks}u, false, 24u);
      let left1 = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&staged_a, sgM * 384u + ${192 + ks}u, false, 24u);
      let right0 = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&staged_b, ${ks * 72}u + sgN * 16u, false, 72u);
      let right1 = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&staged_b, ${ks * 72 + 8}u + sgN * 16u, false, 72u);
      mAcc_00 = subgroupMatrixMultiplyAccumulate(left0, right0, mAcc_00);
      mAcc_01 = subgroupMatrixMultiplyAccumulate(left0, right1, mAcc_01);
      mAcc_10 = subgroupMatrixMultiplyAccumulate(left1, right0, mAcc_10);
      mAcc_11 = subgroupMatrixMultiplyAccumulate(left1, right1, mAcc_11);
    }`).join('\n    ') : `let left = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&staged_a, sgM * 384u, false, 24u);
    let right = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&staged_b, sgN * 16u, false, 72u);
    mAcc = subgroupMatrixMultiplyAccumulate(left, right, mAcc);`}
    workgroupBarrier();
    if (hasNext) {
      staged_a[aWrite] = select(${stType}(0.0), ${stType}(next_a0), kNext + kA < p.n);
      staged_a[aWrite + 1u] = select(${stType}(0.0), ${stType}(next_a1), kNext + kA + 1u < p.n);
      ${vgF16 ? `let pv0 = ${sm8 ? 'unpack2x16float(next_v4.x)' : 'bitcast<vec2<f16>>(next_v4.x)'};
      let pv1 = ${sm8 ? 'unpack2x16float(next_v4.y)' : 'bitcast<vec2<f16>>(next_v4.y)'};
      staged_b[bWrite] = pv0.x;
      staged_b[bWrite + 1u] = pv0.y;
      staged_b[bWrite + 2u] = pv1.x;
      staged_b[bWrite + 3u] = pv1.y;` : `staged_b[bWrite] = ${stType}(next_v4.x);
      staged_b[bWrite + 1u] = ${stType}(next_v4.y);
      staged_b[bWrite + 2u] = ${stType}(next_v4.z);
      staged_b[bWrite + 3u] = ${stType}(next_v4.w);`}
    }
    workgroupBarrier();
  }

  ${sm8 ? `let scBase = (sgM * 16u) * 72u + sgN * 16u;
  subgroupMatrixStore(&scratch, scBase, mAcc_00, false, 72u);
  subgroupMatrixStore(&scratch, scBase + 8u, mAcc_01, false, 72u);
  subgroupMatrixStore(&scratch, scBase + 8u * 72u, mAcc_10, false, 72u);
  subgroupMatrixStore(&scratch, scBase + 8u * 72u + 8u, mAcc_11, false, 72u);` : 'subgroupMatrixStore(&scratch, (sgM * 16u) * 72u + sgN * 16u, mAcc, false, 72u);'}
  workgroupBarrier();
  if (!inBatch) { return; }

  let gColBase4 = ${vgF16 ? '192u' : '576u'} + hClamped * 16u + cV4;
  let attColBase4 = hClamped * 16u + cV4;
  let r0 = m0 + rV;
  if (r0 < p.n) {
    let sb0 = rV * 72u + cV4 * 4u;
    let o0 = vec4<f32>(scratch[sb0], scratch[sb0 + 1u], scratch[sb0 + 2u], scratch[sb0 + 3u]);
    ${vgF16 ? `let g0 = unpack4f16(qkv[r0 * 384u + gColBase4]);
    att[r0 * 192u + attColBase4] = pack4f16(o0 * g0);` : `let g0 = qkv[r0 * 768u + gColBase4];
    att[r0 * 192u + attColBase4] = o0 / (vec4<f32>(1.0) + exp(-g0));`}
  }
  let r1 = r0 + 16u;
  if (r1 < p.n) {
    let sb1 = (rV + 16u) * 72u + cV4 * 4u;
    let o1 = vec4<f32>(scratch[sb1], scratch[sb1 + 1u], scratch[sb1 + 2u], scratch[sb1 + 3u]);
    ${vgF16 ? `let g1 = unpack4f16(qkv[r1 * 384u + gColBase4]);
    att[r1 * 192u + attColBase4] = pack4f16(o1 * g1);` : `let g1 = qkv[r1 * 768u + gColBase4];
    att[r1 * 192u + attColBase4] = o1 / (vec4<f32>(1.0) + exp(-g1));`}
  }
}`;
}

/**
 * Fused `[256, nn] -> [rows, 256]` coalesced tile transpose + LayerNorm(width=256).
 * Processes 16 rows per 256-thread workgroup so both the global reads from `x[c * nn + r]`
 * and global writes to `y[r * 256 + c]` are 100% coalesced across 16-thread half-warps.
 */
function transposeLnOutSource(outF16 = false, inF16 = false) {
  const k4Idx = [0, 1];
  const yt = outF16 ? 'vec2<u32>' : 'vec4<f32>';
  const xType = inF16 ? 'vec2<u32>' : 'vec4<f32>';
  return `${outF16 || inF16 ? 'enable f16;\n' : ''}enable subgroups;
struct P { rows: u32, nnStride4: u32, r0: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<${xType}>;
@group(0) @binding(2) var<storage, read_write> y: array<${yt}>;
@group(0) @binding(3) var<storage, read> w: array<vec4<f32>>;
var<workgroup> tile: array<f32, 4128>;
${inF16 ? `fn unpack4f16(u: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(vec4<f16>(bitcast<vec2<f16>>(u.x), bitcast<vec2<f16>>(u.y)));
}` : ''}
${outF16 ? `fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}` : ''}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let wg = wid.x + wid.y * 65535u;
  let baseRow = wg * 16u;
  let gLoad = li % 4u;
  let cLoad0 = li / 4u;
  let rLocal = baseRow + gLoad * 4u;
  let rGlobal4 = (p.r0 + rLocal) / 4u;
  let gValid = rLocal < p.rows;
  let tb0 = (gLoad * 4u) * 258u + cLoad0;
  for (var s = 0u; s < 4u; s += 1u) {
    let c = s * 64u + cLoad0;
    var v4 = vec4<f32>(0.0);
    if (gValid) {
      let raw = x[c * p.nnStride4 + rGlobal4];
      v4 = ${inF16 ? 'unpack4f16(raw)' : 'raw'};
    }
    let tb = tb0 + s * 64u;
    tile[tb] = v4.x;
    tile[tb + 258u] = v4.y;
    tile[tb + 516u] = v4.z;
    tile[tb + 774u] = v4.w;
  }
  workgroupBarrier();

  let rowBase = (li / 32u) * 516u;
  let lane = li % 32u;
  let w0 = w[lane];
  let w1 = w[32u + lane];
  ${[0, 1].map((r) => `let r_${r} = (li / 32u) * 2u + ${r}u;
  let rb4_${r} = (rowBase + ${r * 258}u) + lane * 4u;
  ${k4Idx.map((k4) => `let sb${k4}_${r} = rb4_${r} + ${k4 * 128}u;
  let v${k4}_${r} = vec4<f32>(tile[sb${k4}_${r}], tile[sb${k4}_${r} + 1u], tile[sb${k4}_${r} + 2u], tile[sb${k4}_${r} + 3u]);`).join('\n  ')}
  let s4_${r} = v0_${r} + v1_${r};
  let mean_${r} = r32Add(s4_${r}.x + s4_${r}.y + s4_${r}.z + s4_${r}.w) / 256.0;
  let d0_${r} = v0_${r} - vec4<f32>(mean_${r});
  let d1_${r} = v1_${r} - vec4<f32>(mean_${r});
  let q4_${r} = d0_${r} * d0_${r} + d1_${r} * d1_${r};
  let inv_${r} = inverseSqrt(r32Add(q4_${r}.x + q4_${r}.y + q4_${r}.z + q4_${r}.w) / 256.0 + p.eps);
  if (baseRow + r_${r} < p.rows) {
    let yb4 = (baseRow + r_${r}) * 64u + lane;
    y[yb4] = ${outF16 ? `pack4f16(d0_${r} * inv_${r} * w0)` : `d0_${r} * inv_${r} * w0`};
    y[yb4 + 32u] = ${outF16 ? `pack4f16(d1_${r} * inv_${r} * w1)` : `d1_${r} * inv_${r} * w1`};
  }`).join('\n  ')}
  let live = baseRow < p.rows;
  if (!live) { return; }
}`;
}

/**
 * Fused `z[a, j, d] += pp[j, d] * pp[a, 384 + d]` + `zc[a, j, :] = LayerNorm_w(z[a, j, :])` (width=384).
 * Uses 4 subgroups (128 threads = 4 rows/workgroup), 128-bit `vec4<f32>` loads/stores, and `subgroupAdd` (0 barriers).
 */
function pairWriteLnInSource(outF16 = false, separateZIn = false, splitPp = false, noLn = false) {
  const kIdx = [0, 1, 2];
  const vt = outF16 ? 'vec2<u32>' : 'vec4<f32>';
  const zSrc = separateZIn ? 'zIn' : 'z';
  return `${outF16 ? 'enable f16;\n' : ''}enable subgroups;
struct P { rows: u32, n: u32, r0: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> z: array<${vt}>;
@group(0) @binding(2) var<storage, read> pp: array<vec4<f32>>;
${noLn ? (separateZIn ? `@group(0) @binding(3) var<storage, read> zIn: array<${vt}>;` : '') : `@group(0) @binding(3) var<storage, read_write> zc: array<${vt}>;
@group(0) @binding(4) var<storage, read> w: array<vec4<f32>>;
${separateZIn ? `@group(0) @binding(5) var<storage, read> zIn: array<${vt}>;` : ''}`}
var<workgroup> shA: array<vec4<f32>, 192>;
${splitPp ? `fn PP(idx: u32) -> vec4<f32> {
  let s4 = p.n * 192u;
  return (pp[idx] + pp[idx + s4]) + (pp[idx + 2u * s4] + pp[idx + 3u * s4]);
}` : 'fn PP(idx: u32) -> vec4<f32> { return pp[idx]; }'}
${outF16 ? `fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}
fn unpack4f16(u: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(vec4<f16>(bitcast<vec2<f16>>(u.x), bitcast<vec2<f16>>(u.y)));
}` : ''}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let rBase = (wid.x + wid.y * 65535u) * 8u;
  let sg = li / 32u;
  let lane = li % 32u;
  let r = rBase + sg;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let rGlobal = p.r0 + row;
  let a = rGlobal / p.n;
  let j = rGlobal % p.n;
  let a0 = (p.r0 + min(rBase, p.rows - 1u)) / p.n;
  let a1 = (p.r0 + min(rBase + 7u, p.rows - 1u)) / p.n;
  if (sg == 0u) {
    let aBase0 = a0 * 192u + 96u + lane;
    ${kIdx.map((k) => `shA[lane + ${k * 32}u] = PP(aBase0 + ${k * 32}u);`).join('\n    ')}
  } else if (sg == 1u && a1 != a0) {
    let aBase1 = a1 * 192u + 96u + lane;
    ${kIdx.map((k) => `shA[96u + lane + ${k * 32}u] = PP(aBase1 + ${k * 32}u);`).join('\n    ')}
  }
  let xb = rGlobal * 96u + lane;
  let jBase = j * 192u + lane;
  ${kIdx.map((k) => `let z_${k} = ${outF16 ? `unpack4f16(${zSrc}[xb + ${k * 32}u])` : `${zSrc}[xb + ${k * 32}u]`};
  let pj_${k} = PP(jBase + ${k * 32}u);`).join('\n  ')}
  workgroupBarrier();
  let aOff = select(0u, 96u, a != a0) + lane;
  ${kIdx.map((k) => `let v_${k} = z_${k} + pj_${k} * shA[aOff + ${k * 32}u];`).join('\n  ')}
  if (live) {
    ${kIdx.map((k) => `z[xb + ${k * 32}u] = ${outF16 ? `pack4f16(v_${k})` : `v_${k}`};`).join('\n    ')}
  }
  ${noLn ? '' : `let s4 = v_0 + v_1 + v_2;
  let mean = r32Add(s4.x + s4.y + s4.z + s4.w) / 384.0;
  ${kIdx.map((k) => `let d_${k} = v_${k} - vec4<f32>(mean);`).join('\n  ')}
  let q4 = d_0 * d_0 + d_1 * d_1 + d_2 * d_2;
  let inv = inverseSqrt(r32Add(q4.x + q4.y + q4.z + q4.w) / 384.0 + p.eps);
  if (!live) { return; }
  let yb = row * 96u + lane;
  ${kIdx.map((k) => `zc[yb + ${k * 32}u] = ${outF16 ? `pack4f16(d_${k} * inv * w[lane + ${k * 32}u])` : `d_${k} * inv * w[lane + ${k * 32}u]`};`).join('\n  ')}`}
}`;
}

function splitReduceNormSource(splits = 4, width = 768, rms = true, hasAdaln = true, postNorm = true, hasNorm2 = false, hasW1 = true, hasLinB = false, addOnly = false, separateGb = false, norm2Rms = false, outF16 = false) {
  const K4 = width / 128;
  const kIdx = Array.from({ length: K4 }, (_, k) => k);
  const outC4 = width / 4;
  const gbVar = separateGb ? 'gbBuf' : 'gsc';
  return `${outF16 ? 'enable f16;\n' : ''}enable subgroups;
struct P { rows: u32, total4: u32, gOff4: u32, gStride4: u32, gbOff4: u32, gbStride4: u32, hasB1: u32, eps: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gsc: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> yn1: array<${outF16 ? 'vec2<u32>' : 'vec4<f32>'}>;
@group(0) @binding(5) var<storage, read> w1: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> b1: array<vec4<f32>>;
${hasNorm2 ? `@group(0) @binding(7) var<storage, read_write> yn2: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> wb2: array<vec4<f32>>;` : separateGb ? '@group(0) @binding(7) var<storage, read> gbBuf: array<vec4<f32>>;' : ''}
fn sigm4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v)); }
${outF16 ? `fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}` : ''}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = (wid.x + wid.y * 65535u) * 4u + (li / 32u);
  let lane = li % 32u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let rb = row * ${outC4}u + lane;
  let gb0 = p.gOff4 + row * p.gStride4 + lane;
  ${!hasW1 ? '_ = w1[0];' : ''}
  ${!hasLinB ? '' : '_ = p.hasB1;'}
  ${kIdx.map((k) => {
    const idx = `rb + ${k * 32}u`;
    const terms = Array.from({ length: splits }, (_, sp) => `s[${idx} + ${sp}u * p.total4]`);
    const sumExpr = splits === 4 ? `(${terms[0]} + ${terms[1]}) + (${terms[2]} + ${terms[3]})` : terms.join(' + ');
    const biased = hasLinB ? `((${sumExpr}) + b1[lane + ${k * 32}u])` : `(${sumExpr})`;
    return addOnly
      ? `let v_${k} = ${biased} + gsc[gb0 + ${k * 32}u];`
      : `let v_${k} = y[${idx}] + ${biased} * sigm4(gsc[gb0 + ${k * 32}u]);`;
  }).join('\n  ')}
  if (live) {
    ${kIdx.map((k) => `y[rb + ${k * 32}u] = v_${k};`).join('\n    ')}
  }
  ${!rms || (hasNorm2 && !norm2Rms) ? `let s4 = ${kIdx.map((k) => `v_${k}`).join(' + ')};
  let mean = r32Add(s4.x + s4.y + s4.z + s4.w) / ${width}.0;
  ${kIdx.map((k) => `let dl_${k} = v_${k} - vec4<f32>(mean);`).join('\n  ')}` : ''}
  ${hasNorm2 && !norm2Rms ? `let ql4 = ${kIdx.map((k) => `dl_${k} * dl_${k}`).join(' + ')};
  let invL = inverseSqrt(r32Add(ql4.x + ql4.y + ql4.z + ql4.w) / ${width}.0 + p.eps);
  if (live) {
    ${kIdx.map((k) => `yn2[rb + ${k * 32}u] = dl_${k} * invL * wb2[lane + ${k * 32}u] + wb2[${outC4}u + lane + ${k * 32}u];`).join('\n    ')}
  }` : ''}
  ${rms ? kIdx.map((k) => `let d_${k} = v_${k};`).join('\n  ') : kIdx.map((k) => `let d_${k} = dl_${k};`).join('\n  ')}
  let q4 = ${kIdx.map((k) => `d_${k} * d_${k}`).join(' + ')};
  let inv = inverseSqrt(r32Add(q4.x + q4.y + q4.z + q4.w) / ${width}.0 + p.eps);
  ${hasNorm2 && norm2Rms ? `if (live) {
    ${kIdx.map((k) => `yn2[rb + ${k * 32}u] = d_${k} * inv * wb2[lane + ${k * 32}u];`).join('\n    ')}
  }` : ''}
  ${hasAdaln && postNorm ? `let adaBase = p.gbOff4 + row * p.gbStride4 + lane;
  ${kIdx.map((k) => `let a_${k} = d_${k} * inv * sigm4(${gbVar}[adaBase + ${k * 32}u]) + ${gbVar}[adaBase + ${outC4 + k * 32}u];`).join('\n  ')}
  ${rms ? kIdx.map((k) => `let d2_${k} = a_${k};`).join('\n  ') : `let s2 = ${kIdx.map((k) => `a_${k}`).join(' + ')};
  let mean2 = r32Add(s2.x + s2.y + s2.z + s2.w) / ${width}.0;
  ${kIdx.map((k) => `let d2_${k} = a_${k} - vec4<f32>(mean2);`).join('\n  ')}`}
  let q2 = ${kIdx.map((k) => `d2_${k} * d2_${k}`).join(' + ')};
  let inv2 = inverseSqrt(r32Add(q2.x + q2.y + q2.z + q2.w) / ${width}.0 + p.eps);
  if (!live) { return; }
  ${kIdx.map((k) => `{
    let c = lane + ${k * 32}u;
    var u = d2_${k} * inv2 * w1[c];
    if (p.hasB1 != 0u) { u += b1[c]; }
    yn1[rb + ${k * 32}u] = ${outF16 ? 'pack4f16(u)' : 'u'};
  }`).join('\n  ')}` : hasAdaln ? `if (!live) { return; }
  let adaBase = p.gbOff4 + row * p.gbStride4 + lane;
  ${kIdx.map((k) => `{
    var u = d_${k} * inv;
    ${hasW1 ? `let c = lane + ${k * 32}u; u *= w1[c]; if (p.hasB1 != 0u) { u += b1[c]; }` : ''}
    let outVal = u * sigm4(${gbVar}[adaBase + ${k * 32}u]) + ${gbVar}[adaBase + ${outC4 + k * 32}u];
    yn1[rb + ${k * 32}u] = ${outF16 ? 'pack4f16(outVal)' : 'outVal'};
  }`).join('\n  ')}` : `if (!live) { return; }
  ${kIdx.map((k) => `{
    let c = lane + ${k * 32}u;
    var u = d_${k} * inv${hasW1 ? ' * w1[c]' : ''};
    if (p.hasB1 != 0u) { u += b1[c]; }
    yn1[rb + ${k * 32}u] = ${outF16 ? 'pack4f16(u)' : 'u'};
  }`).join('\n  ')}`}
}`;
}

function splitReduceQkNormSource(splits = 2, rms = true, hasB = false, outF16 = false) {
  const kIdx = [0, 1, 2, 3, 4, 5];
  const vt = outF16 ? 'vec2<u32>' : 'vec4<f32>';
  return `${outF16 ? 'enable f16;\n' : ''}enable subgroups;
struct P { rows: u32, total4: u32, eps: f32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> qkvgB: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> qkv: array<${vt}>;
@group(0) @binding(4) var<storage, read_write> qk: array<${vt}>;
@group(0) @binding(5) var<storage, read> w: array<vec4<f32>>;
${hasB ? '@group(0) @binding(6) var<storage, read> b: array<vec4<f32>>;' : ''}
${outF16 ? `fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}` : ''}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let seg = li / 32u;
  let lane = li % 32u;
  let base4 = row * 768u + seg * 192u + lane;
  let biasBase4 = seg * 192u + lane;
  ${kIdx.map((k) => {
    const idx = `base4 + ${k * 32}u`;
    const sumExpr = Array.from({ length: splits }, (_, sp) => `s[${idx} + ${sp}u * p.total4]`).join(' + ');
    return `let v_${k} = (${sumExpr}) + qkvgB[biasBase4 + ${k * 32}u];`;
  }).join('\n  ')}
  ${rms ? kIdx.map((k) => `let d_${k} = v_${k};`).join('\n  ') : `let s4 = ${kIdx.map((k) => `v_${k}`).join(' + ')};
  let mean = r32Add(s4.x + s4.y + s4.z + s4.w) / 768.0;
  ${kIdx.map((k) => `let d_${k} = v_${k} - vec4<f32>(mean);`).join('\n  ')}`}
  let q4 = ${kIdx.map((k) => `d_${k} * d_${k}`).join(' + ')};
  let inv = inverseSqrt(r32Add(q4.x + q4.y + q4.z + q4.w) / 768.0 + p.eps);
  if (!live) { return; }
  if (seg < 2u) {
    let qkBase4 = row * 384u + seg * 192u + lane;
    ${kIdx.map((k) => `{
      let c = biasBase4 + ${k * 32}u;
      let u = d_${k} * inv * w[c]${hasB ? ' + b[c]' : ''};
      qk[qkBase4 + ${k * 32}u] = ${outF16 ? 'pack4f16(u)' : 'u'};
    }`).join('\n    ')}
  } else {
    ${outF16 ? `let vgBase4 = row * 384u + (seg - 2u) * 192u + lane;
    if (seg == 2u) {
      ${kIdx.map((k) => `qkv[vgBase4 + ${k * 32}u] = pack4f16(v_${k});`).join('\n      ')}
    } else {
      ${kIdx.map((k) => `qkv[vgBase4 + ${k * 32}u] = pack4f16(vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v_${k})));`).join('\n      ')}
    }` : kIdx.map((k) => `qkv[base4 + ${k * 32}u] = v_${k};`).join('\n    ')}
  }
}`;
}

function splitReduceSource(splits = 4, gate = false, vec4 = false) {
  if (vec4) {
    const terms = Array.from({ length: splits }, (_, s) => `s[idx4 + ${s}u * total4]`);
    const sumExpr = splits === 4
      ? `(${terms[0]} + ${terms[1]}) + (${terms[2]} + ${terms[3]})`
      : terms.join(' + ');
    return `
struct P { total: u32, outC: u32, yOff: u32, yStride: u32, bias: u32, act: u32, accumulate: u32, gOff: u32, gStride: u32, gSigm: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
${gate ? '@group(0) @binding(4) var<storage, read> gsc: array<vec4<f32>>;' : ''}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx4 = (wid.x + wid.y * 65535u) * 256u + li;
  let total4 = p.total / 4u;
  let live = idx4 < total4;
  if (!live) { return; }
  let outC4 = p.outC / 4u;
  let m = idx4 / outC4;
  let n4 = idx4 % outC4;
  var v = ${sumExpr};
  if (p.bias != 0u) { v += bias[n4]; }
  if (p.act == 1u) { v = max(v, vec4<f32>(0.0)); }
  if (p.act == 2u) { v = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v)); }
  let at4 = (p.yOff / 4u) + m * (p.yStride / 4u) + n4;
  ${gate ? `let gv = gsc[(p.gOff / 4u) + m * (p.gStride / 4u) + n4];
  if (p.gSigm == 2u) {
    y[at4] = v + gv;
  } else if (p.gSigm != 0u) {
    y[at4] = y[at4] + v / (vec4<f32>(1.0) + exp(-gv));
  } else {
    y[at4] = y[at4] + v * gv;
  }` : `if (p.accumulate != 0u) { v += y[at4]; }
  y[at4] = v;`}
}`;
  }
  const terms = Array.from({ length: splits }, (_, s) => `s[idx + ${s}u * p.total]`);
  const sumExpr = splits === 4
    ? `(${terms[0]} + ${terms[1]}) + (${terms[2]} + ${terms[3]})`
    : terms.join(' + ');
  return `
struct P { total: u32, outC: u32, yOff: u32, yStride: u32, bias: u32, act: u32, accumulate: u32, gOff: u32, gStride: u32, gSigm: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
${gate ? '@group(0) @binding(4) var<storage, read> gsc: array<f32>;' : ''}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + li;
  let live = idx < p.total;
  if (!live) { return; }
  let m = idx / p.outC;
  let n = idx % p.outC;
  var v = ${sumExpr};
  if (p.bias != 0u) { v += bias[n]; }
  if (p.act == 1u) { v = max(v, 0.0); }
  if (p.act == 2u) { v = 1.0 / (1.0 + exp(-v)); }
  let at = p.yOff + m * p.yStride + n;
  ${gate ? `let gv = gsc[p.gOff + m * p.gStride + n];
  if (p.gSigm == 2u) {
    y[at] = v + gv;
  } else if (p.gSigm != 0u) {
    y[at] = y[at] + v / (1.0 + exp(-gv));
  } else {
    y[at] = y[at] + v * gv;
  }` : `if (p.accumulate != 0u) { v += y[at]; }
  y[at] = v;`}
}`;
}

/**
 * Row norm: y = norm(x) [* w] [+ b] [* sigm(gb_gamma) + gb_beta].
 * Specializes on `K = width / 64` when `width` is a multiple of 64 (<= 1024) so each row is read from global memory ONCE
 * and kept in registers across mean, variance, and writeback.
 */
function normSource(rms, inPlace, width = 0, hasAdaln = false, postNorm = false, preNorm = false, outF16 = false, copyX = false, dual = false, hasW = true, hasB = false, transpose = false, inF16 = false) {
  const K4 = width > 0 && width % 128 === 0 && width <= 1024 ? width / 128 : 0;
  if (K4 > 0) {
    const kIdx = Array.from({ length: K4 }, (_, k) => k);
    const xt = inF16 ? 'vec2<u32>' : 'vec4<f32>';
    const vt = outF16 ? 'vec2<u32>' : 'vec4<f32>';
    const packY = (expr) => (outF16 ? `pack4f16(${expr})` : expr);
    const needW = hasW || postNorm || preNorm;
    const needB = hasB;
    return `${outF16 || inF16 ? 'enable f16;\n' : ''}enable subgroups;
struct P { rows: u32, width: u32, xStride: u32, yStride: u32, xOff: u32, yOff: u32, hasW: u32, hasB: u32,
  eps: f32, gbOff: u32, gbStride: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
${inPlace ? '' : `@group(0) @binding(1) var<storage, read> x: array<${xt}>;`}
@group(0) @binding(2) var<storage, read_write> y: array<${vt}>;
${needW ? '@group(0) @binding(3) var<storage, read> w: array<vec4<f32>>;' : ''}
${needB ? '@group(0) @binding(4) var<storage, read> bv: array<vec4<f32>>;' : ''}
${hasAdaln ? '@group(0) @binding(5) var<storage, read> gb: array<vec4<f32>>;' : ''}
${copyX ? '@group(0) @binding(6) var<storage, read_write> xCopy: array<vec4<f32>>;' : ''}
${inF16 ? `fn unpack4f16(u: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(vec4<f16>(bitcast<vec2<f16>>(u.x), bitcast<vec2<f16>>(u.y)));
}` : ''}
fn X4(i: u32) -> vec4<f32> { return ${inF16 ? 'unpack4f16(x[i])' : `${inPlace ? 'y' : 'x'}[i]`}; }
fn sigm4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v)); }
${outF16 ? `fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}` : ''}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = (wid.x + wid.y * 65535u) * 4u + (li / 32u);
  let lane = li % 32u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  ${dual ? `let groups = select(1u, p.pad0, p.pad0 > 1u);
  let gRows = p.rows / groups;
  let gIdx = row / gRows;
  let rInG = row % gRows;
  let halfRows = gRows / 2u;
  let isSecond = rInG >= halfRows;
  let rEff = select(rInG, rInG - halfRows, isSecond);
  let wBase = (gIdx * 2u + select(0u, 1u, isSecond)) * ${K4 * 32}u;
  let xColOff = select(0u, select(0u, ${width}u, p.xStride > ${width}u), isSecond);
  let xb = (p.xOff + xColOff + rEff * p.xStride) / 4u + lane;` : transpose ? `let xRow = (row % p.pad0) * p.pad0 + (row / p.pad0);
  let xb = (p.xOff + xRow * p.xStride) / 4u + lane;` : `let xb = (p.xOff + row * p.xStride) / 4u + lane;`}
  ${kIdx.map((k) => `let v_${k} = X4(xb + ${k * 32}u);`).join('\n  ')}
  ${rms ? kIdx.map((k) => `let d_${k} = v_${k};`).join('\n  ') : `let s4 = ${kIdx.map((k) => `v_${k}`).join(' + ')};
  let mean = r32Add(s4.x + s4.y + s4.z + s4.w) / ${width}.0;
  ${kIdx.map((k) => `let d_${k} = v_${k} - vec4<f32>(mean);`).join('\n  ')}`}
  let q4 = ${kIdx.map((k) => `d_${k} * d_${k}`).join(' + ')};
  let inv = inverseSqrt(r32Add(q4.x + q4.y + q4.z + q4.w) / ${width}.0 + p.eps);
  ${hasAdaln && postNorm ? `let gBase = (p.gbOff + row * p.gbStride) / 4u + lane;
  ${kIdx.map((k) => `let a_${k} = d_${k} * inv * sigm4(gb[gBase + ${k * 32}u]) + gb[gBase + ${width / 4 + k * 32}u];`).join('\n  ')}
  ${rms ? kIdx.map((k) => `let d2_${k} = a_${k};`).join('\n  ') : `let s2 = ${kIdx.map((k) => `a_${k}`).join(' + ')};
  let mean2 = r32Add(s2.x + s2.y + s2.z + s2.w) / ${width}.0;
  ${kIdx.map((k) => `let d2_${k} = a_${k} - vec4<f32>(mean2);`).join('\n  ')}`}
  let q2 = ${kIdx.map((k) => `d2_${k} * d2_${k}`).join(' + ')};
  let inv2 = inverseSqrt(r32Add(q2.x + q2.y + q2.z + q2.w) / ${width}.0 + p.eps);
  if (!live) { return; }
  let yb = (p.yOff + row * p.yStride) / 4u + lane;
  ${kIdx.map((k) => `{
    let c = lane + ${k * 32}u;
    let v = d2_${k} * inv2 * w[c]${hasB ? ' + bv[c]' : ''};
    y[yb + ${k * 32}u] = ${packY('v')};
    ${copyX ? `xCopy[yb + ${k * 32}u] = v_${k};` : ''}
  }`).join('\n  ')}` : preNorm ? `${kIdx.map((k) => `let a_${k} = d_${k} * inv * w[lane + ${k * 32}u];`).join('\n  ')}
  let q2 = ${kIdx.map((k) => `a_${k} * a_${k}`).join(' + ')};
  let inv2 = inverseSqrt(r32Add(q2.x + q2.y + q2.z + q2.w) / ${width}.0 + p.eps);
  if (!live) { return; }
  let yb = (p.yOff + row * p.yStride) / 4u + lane;
  ${hasAdaln ? `let gBase = (p.gbOff + row * p.gbStride) / 4u + lane;
  ${kIdx.map((k) => `{
    y[yb + ${k * 32}u] = ${packY(`a_${k} * inv2 * sigm4(gb[gBase + ${k * 32}u]) + gb[gBase + ${width / 4 + k * 32}u]`)};
  }`).join('\n  ')}` : kIdx.map((k) => `{
    y[yb + ${k * 32}u] = ${packY(`a_${k} * inv2${hasB ? ` * bv[lane + ${k * 32}u]` : ''}`)};
  }`).join('\n  ')}` : dual ? `if (!live) { return; }
  let yRow = select(rEff, gIdx * 3u * halfRows + rInG, p.yStride <= ${width}u);
  let yb = (p.yOff + select(0u, ${width}u, isSecond && (p.yStride > ${width}u)) + yRow * p.yStride) / 4u + lane;
  ${hasAdaln ? `let gBase = (p.gbOff + rEff * p.gbStride) / 4u + lane;` : ''}
  ${kIdx.map((k) => `{
    let c = wBase + lane + ${k * 32}u;
    var v = d_${k} * inv${hasW ? ' * w[c]' : ''}${hasB ? ' + bv[c]' : ''};
    ${hasAdaln ? `v = v * sigm4(gb[gBase + ${k * 32}u]) + gb[gBase + ${width / 4 + k * 32}u];` : ''}
    y[yb + ${k * 32}u] = ${packY('v')};
    if (isSecond && (p.yStride <= ${width}u)) {
      y[yb + halfRows * (${width / 4}u) + ${k * 32}u] = ${packY(`v_${k}`)};
    }
  }`).join('\n  ')}` : `if (!live) { return; }
  let yb = (p.yOff + row * p.yStride) / 4u + lane;
  ${hasAdaln ? `let gBase = (p.gbOff + row * p.gbStride) / 4u + lane;` : ''}
  ${kIdx.map((k) => `{
    let c = lane + ${k * 32}u;
    var v = d_${k} * inv${hasW ? ' * w[c]' : ''}${hasB ? ' + bv[c]' : ''};
    ${hasAdaln ? `v = v * sigm4(gb[gBase + ${k * 32}u]) + gb[gBase + ${width / 4 + k * 32}u];` : ''}
    y[yb + ${k * 32}u] = ${packY('v')};
  }`).join('\n  ')}`}
}`;
  }
  return `enable subgroups;
struct P { rows: u32, width: u32, xStride: u32, yStride: u32, xOff: u32, yOff: u32, hasW: u32, hasB: u32,
  eps: f32, gbOff: u32, gbStride: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
${inPlace ? '' : '@group(0) @binding(1) var<storage, read> x: array<f32>;'}
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bv: array<f32>;
${hasAdaln ? '@group(0) @binding(5) var<storage, read> gb: array<f32>;' : ''}
var<workgroup> red: array<f32, 2>;
fn X(i: u32) -> f32 { return ${inPlace ? 'y' : 'x'}[i]; }
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
fn reduce(li: u32, val: f32) -> f32 {
  let s = r32Add(val);
  if (li % 32u == 0u) { red[li / 32u] = s; }
  workgroupBarrier();
  return red[0] + red[1];
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wid.x + wid.y * 65535u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let xRow = select(row, (row % p.pad0) * p.pad0 + (row / p.pad0), p.pad0 != 0u);
  let xb = p.xOff + xRow * p.xStride;
  var mean = 0.0;
  ${rms ? '' : `var s = 0.0;
  for (var c = li; c < p.width; c += 64u) { s += X(xb + c); }
  mean = reduce(li, s) / f32(p.width);`}
  var q = 0.0;
  for (var c = li; c < p.width; c += 64u) { let d = X(xb + c) - mean; q += d * d; }
  let inv = inverseSqrt(reduce(li, q) / f32(p.width) + p.eps);
  if (!live) { return; }
  let yb = p.yOff + row * p.yStride;
  ${hasAdaln ? `let gBase = p.gbOff + row * p.gbStride;` : ''}
  for (var c = li; c < p.width; c += 64u) {
    var v = (X(xb + c) - mean) * inv;
    if (p.hasW != 0u) { v *= w[c]; }
    if (p.hasB != 0u) { v += bv[c]; }
    ${hasAdaln ? `v = v * sigm(gb[gBase + c]) + gb[gBase + p.width + c];` : ''}
    y[yb + c] = v;
  }
}`;
}

/**
 * Tensor Core (`chromium_experimental_subgroup_matrix`) `64 x 32` fused `pairToBias` (`[rows, 384] x [24, 384]^T -> [12, nn] x 2`).
 * All 8 subgroups in the 256-thread workgroup compute one `16x16` tile (`sgM = sg / 2` in `0..3`, `sgN = sg % 2` in `0..1`),
 * accumulating `sqSum = sum(z^2)` on the fly during global loads and reducing each row across 4 lanes via 2 `subgroupShuffleXor`s.
 */
function pairToBiasSubmatSource(dual = false, zF16 = true, bF16 = true, sm8 = false) {
  const bt = bF16 ? 'f16' : 'f32';
  const stType = sm8 ? 'f32' : 'f16';
  const stA = sm8 ? 'scratch' : 'staged_a';
  const stB = sm8 ? 'scratch' : 'staged_b';
  const aBaseOff = sm8 ? '512u + ' : '';
  const nHeads = dual ? 24 : 12;
  return `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
struct P { rows: u32, nn: u32, r0: u32, eps384: f32, n: u32, seqStride: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> z: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> wBias: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> bias: array<${bt}>;
${dual ? `@group(0) @binding(4) var<storage, read_write> bias2: array<${bt}>;` : ''}
${sm8 ? '' : `var<workgroup> staged_a: array<f16, 1536>;
var<workgroup> staged_b: array<f16, 512>;`}
var<workgroup> scratch: array<f32, 2176>;
var<workgroup> invRow: array<f32, 64>;
@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) li: u32,
  @builtin(subgroup_id) sg: u32
) {
  let flat = wid.x + wid.y * 65535u;
  let m0 = flat * 64u;
  let live = m0 < p.rows;

  let rA = li / 2u;
  let kA8 = li & 1u;
  let rowA = min(m0 + rA, p.rows - 1u);
  let zRowBase8 = (p.r0 + rowA) * 48u + kA8;
  let aWrite = ${aBaseOff}rA * 24u + kA8 * 8u;

  let hB = li / 2u;
  let kB8 = li & 1u;
  let bValid = hB < ${nHeads}u;
  let wRowBase8 = min(hB, ${nHeads - 1}u) * 48u + kB8;
  let bWrite = hB * 16u + kB8 * 8u;

  ${sm8 ? `var mAcc0_00 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc0_01 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc0_10 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc0_11 = subgroup_matrix_result<f32, 8, 8>();
  ${dual ? `var mAcc1_00 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc1_01 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc1_10 = subgroup_matrix_result<f32, 8, 8>();
  var mAcc1_11 = subgroup_matrix_result<f32, 8, 8>();` : ''}` : `var mAcc0 = subgroup_matrix_result<f32, 16, 16>();
  ${dual ? 'var mAcc1 = subgroup_matrix_result<f32, 16, 16>();' : ''}`}
  var sqSum = 0.0;

  var next_z = z[zRowBase8];
  var next_w = vec4<u32>(0u);
  if (bValid) { next_w = wBias[wRowBase8]; }
  {
    let av0 = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_z.x), bitcast<vec2<f16>>(next_z.y)));
    let av1 = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_z.z), bitcast<vec2<f16>>(next_z.w)));
    let af0 = vec4<f32>(av0);
    let af1 = vec4<f32>(av1);
    sqSum += dot(af0, af0) + dot(af1, af1);
    ${stA}[aWrite] = av0.x; ${stA}[aWrite + 1u] = av0.y; ${stA}[aWrite + 2u] = av0.z; ${stA}[aWrite + 3u] = av0.w;
    ${stA}[aWrite + 4u] = av1.x; ${stA}[aWrite + 5u] = av1.y; ${stA}[aWrite + 6u] = av1.z; ${stA}[aWrite + 7u] = av1.w;
    if (bValid) {
      let bv0 = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_w.x), bitcast<vec2<f16>>(next_w.y)));
      let bv1 = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_w.z), bitcast<vec2<f16>>(next_w.w)));
      ${stB}[bWrite] = bv0.x; ${stB}[bWrite + 1u] = bv0.y; ${stB}[bWrite + 2u] = bv0.z; ${stB}[bWrite + 3u] = bv0.w;
      ${stB}[bWrite + 4u] = bv1.x; ${stB}[bWrite + 5u] = bv1.y; ${stB}[bWrite + 6u] = bv1.z; ${stB}[bWrite + 7u] = bv1.w;
    }
  }
  workgroupBarrier();

  for (var k8 = 0u; k8 < 48u; k8 += 2u) {
    let kNext8 = k8 + 2u;
    let hasNext = kNext8 < 48u;
    if (hasNext) {
      next_z = z[zRowBase8 + kNext8];
      if (bValid) { next_w = wBias[wRowBase8 + kNext8]; }
    }
    ${sm8 ? [0, 8].map((ks) => `{
      let left0 = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&scratch, ${512 + ks}u + sg * 384u, false, 24u);
      let left1 = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&scratch, ${704 + ks}u + sg * 384u, false, 24u);
      let right00 = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&scratch, ${ks}u, true, 16u);
      let right01 = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&scratch, ${128 + ks}u, true, 16u);
      mAcc0_00 = subgroupMatrixMultiplyAccumulate(left0, right00, mAcc0_00);
      mAcc0_01 = subgroupMatrixMultiplyAccumulate(left0, right01, mAcc0_01);
      mAcc0_10 = subgroupMatrixMultiplyAccumulate(left1, right00, mAcc0_10);
      mAcc0_11 = subgroupMatrixMultiplyAccumulate(left1, right01, mAcc0_11);
      ${dual ? `let right10 = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&scratch, ${192 + ks}u, true, 16u);
      let right11 = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&scratch, ${320 + ks}u, true, 16u);
      mAcc1_00 = subgroupMatrixMultiplyAccumulate(left0, right10, mAcc1_00);
      mAcc1_01 = subgroupMatrixMultiplyAccumulate(left0, right11, mAcc1_01);
      mAcc1_10 = subgroupMatrixMultiplyAccumulate(left1, right10, mAcc1_10);
      mAcc1_11 = subgroupMatrixMultiplyAccumulate(left1, right11, mAcc1_11);` : ''}
    }`).join('\n    ') : `let left = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&staged_a, sg * 384u, false, 24u);
    let right0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&staged_b, 0u, true, 16u);
    mAcc0 = subgroupMatrixMultiplyAccumulate(left, right0, mAcc0);
    ${dual ? `let right1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&staged_b, 192u, true, 16u);
    mAcc1 = subgroupMatrixMultiplyAccumulate(left, right1, mAcc1);` : ''}`}
    workgroupBarrier();
    if (hasNext) {
      let av0 = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_z.x), bitcast<vec2<f16>>(next_z.y)));
      let av1 = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_z.z), bitcast<vec2<f16>>(next_z.w)));
      let af0 = vec4<f32>(av0);
      let af1 = vec4<f32>(av1);
      sqSum += dot(af0, af0) + dot(af1, af1);
      ${stA}[aWrite] = av0.x; ${stA}[aWrite + 1u] = av0.y; ${stA}[aWrite + 2u] = av0.z; ${stA}[aWrite + 3u] = av0.w;
      ${stA}[aWrite + 4u] = av1.x; ${stA}[aWrite + 5u] = av1.y; ${stA}[aWrite + 6u] = av1.z; ${stA}[aWrite + 7u] = av1.w;
      if (bValid) {
        let bv0 = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_w.x), bitcast<vec2<f16>>(next_w.y)));
        let bv1 = vec4<${stType}>(vec4<f16>(bitcast<vec2<f16>>(next_w.z), bitcast<vec2<f16>>(next_w.w)));
        ${stB}[bWrite] = bv0.x; ${stB}[bWrite + 1u] = bv0.y; ${stB}[bWrite + 2u] = bv0.z; ${stB}[bWrite + 3u] = bv0.w;
        ${stB}[bWrite + 4u] = bv1.x; ${stB}[bWrite + 5u] = bv1.y; ${stB}[bWrite + 6u] = bv1.z; ${stB}[bWrite + 7u] = bv1.w;
      }
    }
    workgroupBarrier();
  }

  sqSum += subgroupShuffleXor(sqSum, 1u);
  if (kA8 == 0u) {
    invRow[rA] = inverseSqrt(sqSum + p.eps384);
  }
  ${sm8 ? `subgroupMatrixStore(&scratch, sg * 16u, mAcc0_00, true, 68u);
  subgroupMatrixStore(&scratch, sg * 16u + 544u, mAcc0_01, true, 68u);
  subgroupMatrixStore(&scratch, sg * 16u + 8u, mAcc0_10, true, 68u);
  subgroupMatrixStore(&scratch, sg * 16u + 552u, mAcc0_11, true, 68u);
  ${dual ? `subgroupMatrixStore(&scratch, 1088u + sg * 16u, mAcc1_00, true, 68u);
  subgroupMatrixStore(&scratch, 1632u + sg * 16u, mAcc1_01, true, 68u);
  subgroupMatrixStore(&scratch, 1096u + sg * 16u, mAcc1_10, true, 68u);
  subgroupMatrixStore(&scratch, 1640u + sg * 16u, mAcc1_11, true, 68u);` : ''}` : `subgroupMatrixStore(&scratch, sg * 16u, mAcc0, true, 68u);
  ${dual ? 'subgroupMatrixStore(&scratch, 1088u + sg * 16u, mAcc1, true, 68u);' : ''}`}
  workgroupBarrier();
  if (!live) { return; }

  let g = li / 64u;
  let rOut = li % 64u;
  if (m0 + rOut < p.rows) {
    let inv = invRow[rOut];
    let rIdx = p.r0 + m0 + rOut;
    ${[0, 1, 2, 3, 4, 5].map((k) => `bias[(g + ${k * 2}u) * p.nn + rIdx] = ${bt}(scratch[(g + ${k * 2}u) * 68u + rOut] * inv);`).join('\n    ')}
    ${dual ? [0, 1, 2, 3, 4, 5].map((k) => `bias2[(g + ${k * 2}u) * p.nn + rIdx] = ${bt}(scratch[(16u + g + ${k * 2}u) * 68u + rOut] * inv);`).join('\n    ') : ''}
  }
}`;
}

/**
 * Fused pair RMSNorm(384) + linear projection to 12 attention bias heads (`bias: [12, nn]`),
 * optionally computing the next layer's 12 attention bias heads (`bias2: [12, nn]`) from the same `z` read (`dual = true`),
 * or fusing `zInit` in-place `RMSNorm + AdaLN` (`zInitAda = true`) or `zCycNorm` output (`outZc = true`).
 * 8 subgroups (256 threads) process 4 rows/subgroup (32 rows/workgroup), reusing `wBias` (`w_bias * w_norm * sqrt(384)`) 4x,
 * reducing 4 rows simultaneously via `subgroupAdd(vec4<f32>)`, and writing 32-float coalesced cache lines per head.
 */
function pairToBiasSource(half, dual = false, zInitAda = false, outZc = false, zcF16 = false, zF16 = false, bF16 = false) {
  const nHeads = dual ? 24 : 12;
  const zt = zF16 ? 'vec2<u32>' : 'vec4<f32>';
  const bt = bF16 ? 'f16' : 'f32';
  return `${zcF16 || zF16 || bF16 ? 'enable f16;\n' : ''}enable subgroups;
struct P { rows: u32, nn: u32, r0: u32, eps384: f32, n: u32, seqStride: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, ${zInitAda ? 'read_write' : 'read'}> z: array<${zt}>;
@group(0) @binding(2) var<storage, read> wBias: array<${half ? 'vec2<u32>' : 'vec4<f32>'}>;
@group(0) @binding(3) var<storage, read_write> bias: array<${bt}>;
${dual ? `@group(0) @binding(4) var<storage, read_write> bias2: array<${bt}>;` : ''}
${zInitAda ? `@group(0) @binding(4) var<storage, read> wLnBins: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> gb: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> zStatic: array<${zt}>;
@group(0) @binding(7) var<storage, read> seqF: array<f32>;
fn sigm4(v: vec4<f32>) -> vec4<f32> { return vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v)); }` : ''}
${outZc ? `@group(0) @binding(4) var<storage, read> wZc: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> zc: array<${zcF16 ? 'vec2<u32>' : 'vec4<f32>'}>;` : ''}
${zcF16 || zF16 ? `fn pack4f16(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(vec2<f16>(v.xy)), bitcast<u32>(vec2<f16>(v.zw)));
}` : ''}
var<workgroup> outTile: array<f32, ${nHeads * 32}>;
${half || zF16 ? `fn unpack4(u: vec2<u32>) -> vec4<f32> {
  let lo = unpack2x16float(u.x);
  let hi = unpack2x16float(u.y);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}` : ''}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wid.x + wid.y * 65535u;
  let sg = li / 32u;
  let lane = li % 32u;
  let rBase = flat * 32u + sg * 4u;
  let live = rBase < p.rows;
  ${zInitAda ? `let lim = 0.1 + (2.9 * f32(lane)) / 28.0;
  ${[0, 1, 2, 3].map((k) => `let row_${k} = min(rBase + ${k}u, p.rows - 1u);
  let rIdx_${k} = p.r0 + row_${k};
  let ri_${k} = (rIdx_${k} / p.n) * p.seqStride;
  let rj_${k} = (rIdx_${k} % p.n) * p.seqStride;
  let dx_${k} = seqF[ri_${k}] - seqF[rj_${k}];
  let dy_${k} = seqF[ri_${k} + 1u] - seqF[rj_${k} + 1u];
  let dz_${k} = seqF[ri_${k} + 2u] - seqF[rj_${k} + 2u];
  let dist_${k} = sqrt(dx_${k} * dx_${k} + dy_${k} * dy_${k} + dz_${k} * dz_${k});
  let bin_${k} = r32MaxU(select(0u, lane + 1u, lane < 29u && lim < dist_${k}));
  let binOff_${k} = (1u + bin_${k}) * 96u + lane;
  let zb_${k} = rIdx_${k} * 96u + lane;
  var z0_${k} = ${zF16 ? `unpack4(zStatic[zb_${k}])` : `zStatic[zb_${k}]`} + wLnBins[binOff_${k}];
  var z1_${k} = ${zF16 ? `unpack4(zStatic[zb_${k} + 32u])` : `zStatic[zb_${k} + 32u]`} + wLnBins[binOff_${k} + 32u];
  var z2_${k} = ${zF16 ? `unpack4(zStatic[zb_${k} + 64u])` : `zStatic[zb_${k} + 64u]`} + wLnBins[binOff_${k} + 64u];`).join('\n  ')}
  {
    let eps = p.eps384 * (1.0 / 384.0);
    ${[0, 1, 2, 3].map((k) => `let sq1_${k} = dot(z0_${k}, z0_${k}) + dot(z1_${k}, z1_${k}) + dot(z2_${k}, z2_${k});`).join('\n    ')}
    let inv1 = inverseSqrt(r32Add4(vec4<f32>(sq1_0, sq1_1, sq1_2, sq1_3)) * (1.0 / 384.0) + vec4<f32>(eps));
    let ln0 = wLnBins[lane];
    let ln1 = wLnBins[lane + 32u];
    let ln2 = wLnBins[lane + 64u];
    ${[0, 1, 2, 3].map((k) => `z0_${k} = z0_${k} * inv1[${k}] * ln0;
    z1_${k} = z1_${k} * inv1[${k}] * ln1;
    z2_${k} = z2_${k} * inv1[${k}] * ln2;
    let sq2_${k} = dot(z0_${k}, z0_${k}) + dot(z1_${k}, z1_${k}) + dot(z2_${k}, z2_${k});`).join('\n    ')}
    let inv2 = inverseSqrt(r32Add4(vec4<f32>(sq2_0, sq2_1, sq2_2, sq2_3)) * (1.0 / 384.0) + vec4<f32>(eps));
    let g0 = sigm4(gb[lane]);
    let g1 = sigm4(gb[lane + 32u]);
    let g2 = sigm4(gb[lane + 64u]);
    let bt0 = gb[96u + lane];
    let bt1 = gb[128u + lane];
    let bt2 = gb[160u + lane];
    ${[0, 1, 2, 3].map((k) => `z0_${k} = z0_${k} * inv2[${k}] * g0 + bt0;
    z1_${k} = z1_${k} * inv2[${k}] * g1 + bt1;
    z2_${k} = z2_${k} * inv2[${k}] * g2 + bt2;
    if (rBase + ${k}u < p.rows) {
      z[zb_${k}] = ${zF16 ? `pack4f16(z0_${k})` : `z0_${k}`};
      z[zb_${k} + 32u] = ${zF16 ? `pack4f16(z1_${k})` : `z1_${k}`};
      z[zb_${k} + 64u] = ${zF16 ? `pack4f16(z2_${k})` : `z2_${k}`};
    }`).join('\n    ')}
  }` : [0, 1, 2, 3].map((k) => `let row_${k} = min(rBase + ${k}u, p.rows - 1u);
  let zb_${k} = (p.r0 + row_${k}) * 96u + lane;
  var z0_${k} = ${zF16 ? `unpack4(z[zb_${k}])` : `z[zb_${k}]`};
  var z1_${k} = ${zF16 ? `unpack4(z[zb_${k} + 32u])` : `z[zb_${k} + 32u]`};
  var z2_${k} = ${zF16 ? `unpack4(z[zb_${k} + 64u])` : `z[zb_${k} + 64u]`};`).join('\n  ')}
  ${[0, 1, 2, 3].map((k) => `let sq_${k} = dot(z0_${k}, z0_${k}) + dot(z1_${k}, z1_${k}) + dot(z2_${k}, z2_${k});`).join('\n  ')}
  let inv4 = inverseSqrt(r32Add4(vec4<f32>(sq_0, sq_1, sq_2, sq_3)) + vec4<f32>(p.eps384));
  ${[0, 1, 2, 3].map((k) => `z0_${k} *= inv4[${k}];
  z1_${k} *= inv4[${k}];
  z2_${k} *= inv4[${k}];`).join('\n  ')}
  ${outZc ? `{
    let cn0 = wZc[lane];
    let cn1 = wZc[lane + 32u];
    let cn2 = wZc[lane + 64u];
    ${[0, 1, 2, 3].map((k) => `if (rBase + ${k}u < p.rows) {
      let zcb_${k} = row_${k} * 96u + lane;
      zc[zcb_${k}] = ${zcF16 ? `pack4f16(z0_${k} * cn0)` : `z0_${k} * cn0`};
      zc[zcb_${k} + 32u] = ${zcF16 ? `pack4f16(z1_${k} * cn1)` : `z1_${k} * cn1`};
      zc[zcb_${k} + 64u] = ${zcF16 ? `pack4f16(z2_${k} * cn2)` : `z2_${k} * cn2`};
    }`).join('\n    ')}
  }` : ''}
  for (var h = 0u; h < ${nHeads}u; h += 1u) {
    let wb = h * 96u + lane;
    let b0 = ${half ? 'unpack4(wBias[wb])' : 'wBias[wb]'};
    let b1 = ${half ? 'unpack4(wBias[wb + 32u])' : 'wBias[wb + 32u]'};
    let b2 = ${half ? 'unpack4(wBias[wb + 64u])' : 'wBias[wb + 64u]'};
    let d4 = vec4<f32>(
      dot(z0_0, b0) + dot(z1_0, b1) + dot(z2_0, b2),
      dot(z0_1, b0) + dot(z1_1, b1) + dot(z2_1, b2),
      dot(z0_2, b0) + dot(z1_2, b1) + dot(z2_2, b2),
      dot(z0_3, b0) + dot(z1_3, b1) + dot(z2_3, b2),
    );
    let sum4 = r32Add4(d4);
    if (lane == 0u) {
      let tb = h * 32u + sg * 4u;
      outTile[tb] = sum4.x;
      outTile[tb + 1u] = sum4.y;
      outTile[tb + 2u] = sum4.z;
      outTile[tb + 3u] = sum4.w;
    }
  }
  workgroupBarrier();
  let rOut = flat * 32u + lane;
  if (rOut < p.rows) {
    let rIdx = p.r0 + rOut;
    bias[sg * p.nn + rIdx] = ${bt}(outTile[li]);
    ${dual ? `if (sg < 4u) {
      bias[(sg + 8u) * p.nn + rIdx] = ${bt}(outTile[li + 256u]);
    } else {
      bias2[(sg - 4u) * p.nn + rIdx] = ${bt}(outTile[li + 256u]);
    }
    bias2[(sg + 4u) * p.nn + rIdx] = ${bt}(outTile[li + 512u]);` : `if (sg < 4u) {
      bias[(sg + 8u) * p.nn + rIdx] = ${bt}(outTile[li + 256u]);
    }`}
  }
  if (!live) { return; }
}`;
}

/**
 * Fused Trunk readout heads (`ca_linear` `[n, 3]` + `local_latents_linear` `[n, 32]`):
 * 4 subgroups (128 threads) process 4 rows/subgroup (16 rows/workgroup), computing the shared RMSNorm inverse std once,
 * applying `w_ca_norm` and `w_ll_norm` in registers, dotting with the `3 + 32 = 35` rows of `wLin` via `subgroupAdd(vec4<f32>)`,
 * and writing `out[0 .. 3*n]` and `out[3*n .. 35*n]` in a single dispatch.
 */
function headFastSource(half) {
  const mIdx = [0, 1, 2, 3, 4, 5];
  return `enable subgroups;
struct P { n: u32, eps768: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> wLin: array<${half ? 'vec2<u32>' : 'vec4<f32>'}>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> outTile: array<f32, 512>;
${half ? `fn unpack4(u: vec2<u32>) -> vec4<f32> {
  let lo = unpack2x16float(u.x);
  let hi = unpack2x16float(u.y);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}` : ''}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wid.x + wid.y * 65535u;
  let sg = li / 32u;
  let lane = li % 32u;
  let rBase = flat * 16u + sg * 4u;
  let live = rBase < p.n;
  ${[0, 1, 2, 3].map((k) => `let row_${k} = min(rBase + ${k}u, p.n - 1u);
  let sb_${k} = row_${k} * 192u + lane;
  ${mIdx.map((m) => `var s${m}_${k} = s[sb_${k} + ${m * 32}u];`).join('\n  ')}
  let sq_${k} = ${mIdx.map((m) => `dot(s${m}_${k}, s${m}_${k})`).join(' + ')};`).join('\n  ')}
  let inv4 = inverseSqrt(r32Add4(vec4<f32>(sq_0, sq_1, sq_2, sq_3)) + vec4<f32>(p.eps768));
  ${[0, 1, 2, 3].map((k) => mIdx.map((m) => `s${m}_${k} *= inv4[${k}];`).join('\n  ')).join('\n  ')}
  for (var c = 0u; c < 3u; c += 1u) {
    let wb = c * 192u + lane;
    ${mIdx.map((m) => `let b_${m} = ${half ? `unpack4(wLin[wb + ${m * 32}u])` : `wLin[wb + ${m * 32}u]`};`).join('\n    ')}
    let d4 = vec4<f32>(
      ${[0, 1, 2, 3].map((k) => mIdx.map((m) => `dot(s${m}_${k}, b_${m})`).join(' + ')).join(',\n      ')}
    );
    let sum4 = r32Add4(d4);
    if (lane == 0u) {
      if (rBase < p.n) { out[rBase * 3u + c] = sum4.x; }
      if (rBase + 1u < p.n) { out[(rBase + 1u) * 3u + c] = sum4.y; }
      if (rBase + 2u < p.n) { out[(rBase + 2u) * 3u + c] = sum4.z; }
      if (rBase + 3u < p.n) { out[(rBase + 3u) * 3u + c] = sum4.w; }
    }
  }
  for (var c = 0u; c < 32u; c += 1u) {
    let wb = (3u + c) * 192u + lane;
    ${mIdx.map((m) => `let b_${m} = ${half ? `unpack4(wLin[wb + ${m * 32}u])` : `wLin[wb + ${m * 32}u]`};`).join('\n    ')}
    let d4 = vec4<f32>(
      ${[0, 1, 2, 3].map((k) => mIdx.map((m) => `dot(s${m}_${k}, b_${m})`).join(' + ')).join(',\n      ')}
    );
    let sum4 = r32Add4(d4);
    if (lane == 0u) {
      let tb = sg * 128u + c;
      outTile[tb] = sum4.x;
      outTile[tb + 32u] = sum4.y;
      outTile[tb + 64u] = sum4.z;
      outTile[tb + 96u] = sum4.w;
    }
  }
  workgroupBarrier();
  if (!live) { return; }
  let llBase = 3u * p.n;
  ${[0, 1, 2, 3].map((k) => `if (rBase + ${k}u < p.n) {
    out[llBase + (rBase + ${k}u) * 32u + lane] = outTile[(sg * 4u + ${k}u) * 32u + lane];
  }`).join('\n  ')}
}`;
}

/**
 * In-place softmax over rows of `len` after scale and an additive bias of the same layout:
 * row r at l[r * len + j], bias at bias[r * len + j].
 * 4 subgroups (128 threads) process 4 rows/workgroup with zero `workgroupBarrier()`s; when `K > 0` (`len <= K * 32`),
 * each row is read from global memory ONCE and kept in registers across `subgroupMax`, `exp`, `subgroupAdd`, and writeback.
 */
function softmaxSource(K = 0, hasBias = false, hasScale = false, lF16 = false) {
  const kIdx = Array.from({ length: K }, (_, k) => k);
  const lt = lF16 ? 'f16' : 'f32';
  const valExpr = (idx) => `${hasScale ? `f32(l[${idx}]) * p.scale` : `f32(l[${idx}])`}${hasBias ? ` + f32(bias[${idx}])` : ''}`;
  return `${lF16 ? 'enable f16;\n' : ''}enable subgroups;
struct P { rows: u32, len: u32, hasBias: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> l: array<${lt}>;
${hasBias ? `@group(0) @binding(2) var<storage, read> bias: array<${lt}>;` : ''}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = (wid.x + wid.y * 65535u) * 4u + (li / 32u);
  let lane = li % 32u;
  let live = r < p.rows;
  let row = min(r, p.rows - 1u);
  let base = row * p.len;
  ${K > 0 ? `var mx = -3.0e38;
  ${kIdx.map((k) => `let j_${k} = lane + ${k * 32}u;
  var v_${k} = -3.0e38;
  if (j_${k} < p.len) {
    v_${k} = ${valExpr(`base + j_${k}`)};
    mx = max(mx, v_${k});
  }`).join('\n  ')}
  let top = r32MaxF(mx);
  var s = 0.0;
  ${kIdx.map((k) => `var e_${k} = 0.0;
  if (j_${k} < p.len) {
    e_${k} = exp(v_${k} - top);
    s += e_${k};
  }`).join('\n  ')}
  let invTotal = 1.0 / r32Add(s);
  if (!live) { return; }
  ${kIdx.map((k) => `if (j_${k} < p.len) { l[base + j_${k}] = ${lt}(e_${k} * invTotal); }`).join('\n  ')}` : `var mx = -3.0e38;
  for (var j = lane; j < p.len; j += 32u) { mx = max(mx, ${valExpr('base + j')}); }
  let top = r32MaxF(mx);
  var s = 0.0;
  for (var j = lane; j < p.len; j += 32u) { s += exp((${valExpr('base + j')}) - top); }
  let invTotal = 1.0 / r32Add(s);
  if (!live) { return; }
  for (var j = lane; j < p.len; j += 32u) { l[base + j] = ${lt}(exp((${valExpr('base + j')}) - top) * invTotal); }`}
}`;
}

/** Coalesced 16x16 tiled transpose for triangle multiplication (`proj [n*n, 512]` (pre-gated) or `[n*n, 1024] -> aT, bT [256, n, n]`). */
function triGateSource(incoming, preGated = false) {
  return `
struct P { n: u32, tilesI: u32, totalGroups: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read_write> aT: array<f32>;
@group(0) @binding(3) var<storage, read_write> bT: array<f32>;
var<workgroup> sa: array<f32, 272>;
var<workgroup> sb: array<f32, 272>;
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let flat = wid.x + wid.y * 65535u;
  let live = flat < p.totalGroups;
  let tc = flat % 16u;
  let rem = flat / 16u;
  let ti = rem % p.tilesI;
  let j = rem / p.tilesI;
  let tx = li % 16u;
  let ty = li / 16u;
  let c0 = tc * 16u;
  let i0 = ti * 16u;
  let n = p.n;
  let nn = n * n;
  var av = 0.0;
  var bv = 0.0;
  if (live && i0 + ty < n) {
    let i = i0 + ty;
    let rIn = ${incoming ? 'i * n + j' : 'j * n + i'};
    ${preGated ? `let pr = rIn * 512u + c0 + tx;
    av = proj[pr];
    bv = proj[pr + 256u];` : `let pr = rIn * 1024u + c0 + tx;
    av = proj[pr] * sigm(proj[pr + 256u]);
    bv = proj[pr + 512u] * sigm(proj[pr + 768u]);`}
  }
  sa[ty * 17u + tx] = av;
  sb[ty * 17u + tx] = bv;
  workgroupBarrier();
  if (live && i0 + tx < n) {
    let c = c0 + ty;
    let rOut = j * n + i0 + tx;
    let at = c * nn + rOut;
    aT[at] = sa[tx * 17u + ty];
    bT[at] = sb[tx * 17u + ty];
  }
}`;
}

function mapSource(bindings, body) {
  const decls = bindings
    .map((b, at) => (b === null ? '' : `@group(0) @binding(${at + 1}) var<storage, ${b.write ? 'read_write' : 'read'}> b${at}: array<${b.type ?? 'f32'}>;`))
    .join('\n');
  // An f16-typed binding needs the extension declared; `enable` must precede everything.
  const f16 = bindings.some((b) => b !== null && /f16/.test(b.type ?? ''));
  return `${f16 ? 'enable f16;\n' : ''}
struct P { count: u32, u: array<vec4<u32>, 4>, f: array<vec4<f32>, 4> };
@group(0) @binding(0) var<uniform> p: P;
${decls}
fn U(k: u32) -> u32 { return p.u[k / 4u][k % 4u]; }
fn F(k: u32) -> f32 { return p.f[k / 4u][k % 4u]; }
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + li;
  if (i >= p.count) { return; }
${body}
}`;
}

function hashText(text) {
  let h = 5381;
  for (let i = 0; i < text.length; i += 1) h = ((h * 33) ^ text.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/** float32 -> float16 bits, round to nearest even. */
export function toHalf(value) {
  _f32[0] = value;
  const x = _u32[0];
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
    if (rem > mid || (rem === mid && half & 1)) half += 1;
    return sign | half;
  }
  let half = (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && half & 1)) half += 1;
  return sign | half;
}

/** Pack float32 values into float16 pairs (u32 words), host side. */
export function packHalf(values) {
  if (typeof Float16Array !== 'undefined') {
    const padded = new Float16Array(Math.ceil(values.length / 2) * 2);
    padded.set(values);
    return new Uint32Array(padded.buffer);
  }
  const words = new Uint32Array(Math.ceil(values.length / 2));
  for (let i = 0; i < values.length; i += 2) {
    words[i >> 1] = toHalf(values[i]) | ((i + 1 < values.length ? toHalf(values[i + 1]) : 0) << 16);
  }
  return words;
}

/**
 * How the kernels' 32-lane reductions run on this device. Every kernel lays its threads out as rows of 32 lanes
 * (`lane = li % 32`) and reduces a row with `r32Add` / `r32Add4` / `r32MaxU` / `r32MaxF`, which `subgroupLib` defines:
 *   native  subgroups are exactly 32 lanes (Apple, NVIDIA): the subgroup op itself.
 *   xor     subgroups are 32 lanes or wider (Adreno, AMD wave64): a shuffle butterfly confined to each 32-lane row.
 *   pin32   the size varies but 32 is allowed (Intel): `@subgroup_size(32)` on every subgroup kernel.
 *   fixed   subgroups are a fixed 16 lanes (Mali): subgroup op, then the row's halves are added in workgroup memory.
 *   shared  anything else: the whole row is reduced in workgroup memory.
 * A plain subgroupAdd on a 16-lane Mali GPU sums half a row, which silently wrecked every norm and GEMV there.
 */
export function subgroupMode(device) {
  // Checkers can force a mode (`--subgroup-mode=xor|shared`) to exercise it on any GPU.
  if (device.subgroupModeOverride) return { mode: device.subgroupModeOverride, size: 32 };
  const info = device.adapterInfo ?? {};
  const lo = info.subgroupMinSize;
  const hi = info.subgroupMaxSize;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { mode: 'native', size: 32 };
  if (lo === 32 && hi === 32) return { mode: 'native', size: 32 };
  if (lo >= 32) return { mode: 'xor', size: lo };
  if (hi >= 32 && device.features?.has?.('subgroup-size-control')) return { mode: 'pin32', size: 32 };
  if (lo === hi && lo >= 16) return { mode: 'fixed', size: lo };
  return { mode: 'shared', size: lo };
}

const R32_TYPES = [['r32Add', 'f32', 'Add'], ['r32Add4', 'vec4<f32>', 'Add'], ['r32MaxU', 'u32', 'Max'], ['r32MaxF', 'f32', 'Max']];

/** WGSL definitions of the 32-lane reductions a kernel uses (see `subgroupMode`); `wgSize` threads per workgroup. */
function subgroupLib(sg, used, wgSize) {
  const defs = [];
  const pre = [];
  for (const [name, type, op] of R32_TYPES.filter(([name]) => used.has(name))) {
    const comb = (a, b) => (op === 'Add' ? `${a} + ${b}` : `max(${a}, ${b})`);
    if (sg.mode === 'native' || sg.mode === 'pin32') {
      defs.push(`fn ${name}(x: ${type}) -> ${type} { return subgroup${op}(x); }`);
    } else if (sg.mode === 'xor') {
      defs.push(`fn ${name}(x: ${type}) -> ${type} {
  var v = x;
  ${[16, 8, 4, 2, 1].map((m) => `v = ${comb('v', `subgroupShuffleXor(v, ${m}u)`)};`).join('\n  ')}
  return v;
}`);
    } else {
      // One workgroup slot per subgroup (fixed) or per thread (shared); barrier, write, barrier, read the row.
      const per = sg.mode === 'fixed' ? 32 / sg.size : 32;
      const buf = `${name}_buf`;
      pre.push(`var<workgroup> ${buf}: array<${type}, ${(wgSize / 32) * per}>;`);
      const first = sg.mode === 'fixed' ? `subgroup${op}(x)` : 'x';
      const slot = sg.mode === 'fixed' ? `r32_li / ${sg.size}u` : 'r32_li';
      const write = sg.mode === 'fixed' ? `if (r32_li % ${sg.size}u == 0u) { ${buf}[${slot}] = s; }` : `${buf}[${slot}] = s;`;
      defs.push(`fn ${name}(x: ${type}) -> ${type} {
  let s = ${first};
  workgroupBarrier();
  ${write}
  workgroupBarrier();
  let base = (r32_li / 32u) * ${per}u;
  var v = ${buf}[base];
  for (var k = 1u; k < ${per}u; k += 1u) { v = ${comb('v', `${buf}[base + k]`)}; }
  return v;
}`);
    }
  }
  if (pre.length) pre.unshift('var<private> r32_li: u32;');
  return [...pre, ...defs].join('\n');
}

/** Insert the reduction helpers (and, where needed, `@subgroup_size(32)` and the lane index) into a kernel. */
function applySubgroupMode(source, sg) {
  const uses = source.includes('enable subgroups;');
  if (!uses) return source;
  const used = new Set(R32_TYPES.map(([name]) => name).filter((name) => new RegExp(`\\b${name}\\(`).test(source)));
  let src = source;
  if (sg.mode === 'pin32') {
    src = src.replace('enable subgroups;', 'enable subgroups;\nenable subgroup_size_control;')
      .replace(/(@workgroup_size\([^)]*\))/, '$1 @subgroup_size(32)');
  }
  if (!used.size) return src;
  const wg = /@workgroup_size\(([^)]*)\)/.exec(src)[1].split(',').map((v) => Number.parseInt(v, 10) || 1)
    .reduce((a, b) => a * b, 1);
  const lib = subgroupLib(sg, used, wg);
  // After the enable directives, which must come first.
  const enables = [...src.matchAll(/^enable [^;]+;\s*$/gm)];
  const at = enables.length ? enables[enables.length - 1].index + enables[enables.length - 1][0].length : 0;
  src = `${src.slice(0, at)}\n${lib}\n${src.slice(at)}`;
  if (lib.includes('r32_li')) {
    // The parameter list has parentheses of its own (`@builtin(...)`): find its closing one by depth.
    const start = src.indexOf('fn main(') + 'fn main('.length;
    let end = start;
    for (let depth = 1; depth > 0; end += 1) depth += src[end] === '(' ? 1 : src[end] === ')' ? -1 : 0;
    const paramList = src.slice(start, end - 1);
    const body = src.indexOf('{', end);
    const named = /@builtin\(local_invocation_index\)\s+(\w+)/.exec(paramList);
    const params = named ? paramList
      : `${paramList.trim().replace(/,\s*$/, '')}${paramList.trim() ? ', ' : ''}@builtin(local_invocation_index) r32_li_in: u32`;
    src = `${src.slice(0, start)}${params})${src.slice(end, body + 1)}\n  r32_li = ${named ? named[1] : 'r32_li_in'};${src.slice(body + 1)}`;
  }
  return src;
}

export class Rfd4pEngine {
  /** @param {GPUDevice} device */
  constructor(device) {
    this.device = device;
    this.hasF16 = device.features?.has?.('shader-f16') === true;
    this.subgroups = subgroupMode(device);
    // On a Mali-G715 (Pixel 9) the desktop-tuned GEMM refinements (prefetch, shuffles, paired epilogue) cost 8% per
    // trunk pass (976 -> 900 ms at n=100), so ARM GPUs take the plain kernels. Also measured there and slower:
    // f16 multiply-add in the GEMM inner loop (943 ms), 32x64 tiles (1319 ms), and no workgroup-memory staging
    // (triProjG 445 -> 1304 ms).
    if (device.adapterInfo?.vendor === 'arm') this.gemmVariant = { baseline: true };
    if (device.gemmVariantOverride) this.gemmVariant = device.gemmVariantOverride;
    const smConfigs = [...(device.adapterInfo?.subgroupMatrixConfigs ?? device.subgroupMatrixConfigs ?? [])];
    this.subgroupMatrixConfigs = smConfigs.map((c) => ({
      componentType: c.componentType,
      resultComponentType: c.resultComponentType,
      M: c.M,
      N: c.N,
      K: c.K,
    }));
    const has16x16x16 = this.subgroupMatrixConfigs.some(
      (c) => c.M === 16 && c.N === 16 && c.K === 16 && c.componentType === 'f16' && c.resultComponentType === 'f32',
    );
    const has8x8x8 = this.subgroupMatrixConfigs.some(
      (c) => c.M === 8 && c.N === 8 && c.K === 8 && c.componentType === 'f32' && c.resultComponentType === 'f32',
    );
    this.submat8 = !has16x16x16 && has8x8x8;
    this.hasSubgroupMatrix =
      !device.noSubmat &&
      this.hasF16 &&
      device.features?.has?.('chromium-experimental-subgroup-matrix') === true &&
      (this.subgroupMatrixConfigs.length === 0 || has16x16x16 || has8x8x8);
    this.pipelines = new Map();
    this.dummy = device.createBuffer({ label: 'rfd4p.dummy', size: 16, usage: STORAGE });
    this.activations = [];
    this.activationBytes = 0;
    this.weights = [];
    this.weightBytes = 0;
    this.uniforms = [];
  }

  async pipeline(key, source) {
    let p = this.pipelines.get(key);
    if (!p) {
      const code = applySubgroupMode(source, this.subgroups);
      p = this.device.createComputePipelineAsync({
        label: `rfd4p.${key}`,
        layout: 'auto',
        compute: { module: this.device.createShaderModule({ label: `rfd4p.${key}`, code }), entryPoint: 'main' },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  /** A float32 activation buffer of `floats` elements. */
  buffer(label, floats) {
    const size = Math.max(16, Math.ceil((floats * 4) / 16) * 16);
    const maxSize = this.device.limits.maxStorageBufferBindingSize;
    if (size > maxSize) throw new Error(`rfd4p buffer ${label} needs ${size} bytes > binding limit ${maxSize}`);
    const buffer = this.device.createBuffer({ label: `rfd4p.${label}`, size, usage: STORAGE | COPY_SRC | COPY_DST });
    buffer.floats = floats;
    this.activations.push(buffer);
    this.activationBytes += size;
    return buffer;
  }

  /** A weight buffer from float32 values; `half` packs to float16 pairs on the host. */
  weight(label, values, half = false, outC = 0, align = 16) {
    let src = values instanceof Float32Array ? values : Float32Array.from(values);
    let stride = 0;
    if (half && outC > 0) {
      const inC = src.length / outC;
      stride = Math.ceil(inC / align) * align;
      if (stride !== inC) {
        const padded = new Float32Array(outC * stride);
        for (let r = 0; r < outC; r += 1) {
          padded.set(src.subarray(r * inC, (r + 1) * inC), r * stride);
        }
        src = padded;
      }
    }
    const data = half ? packHalf(src) : src;
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    const buffer = this.device.createBuffer({ label: `rfd4p.w.${label}`, size, usage: STORAGE | COPY_DST });
    this.write(buffer, data);
    buffer.half = half;
    buffer.floats = values.length;
    if (stride > 0) buffer.stride = stride;
    this.weights.push(buffer);
    this.weightBytes += size;
    return buffer;
  }

  write(buffer, data, offsetFloats = 0) {
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const padded = bytes.byteLength % 4 === 0 ? bytes : (() => {
      const out = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
      out.set(bytes);
      return out;
    })();
    this.device.queue.writeBuffer(buffer, offsetFloats * 4, padded);
  }

  uniform(words) {
    const size = Math.max(16, Math.ceil(words.byteLength / 16) * 16);
    const buffer = this.device.createBuffer({ label: 'rfd4p.uniform', size, usage: UNIFORM | COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, words.buffer, words.byteOffset, words.byteLength);
    this.uniforms.push(buffer);
    return buffer;
  }

  /**
   * One dispatch. An entry of `buffers` is a `GPUBuffer`, or a VIEW of one -
   * `{ buffer, offsetBytes, sizeBytes }` - which is how a chunked kernel reaches a window
   * of a whole-tensor buffer while still indexing from zero.
   *
   * 🔴 THE VIEW IS WHAT LETS A CHUNK KEEP THE UNCHUNKED KERNEL. The alternative is a row
   * offset in the uniform and `+ rowOff` in the hot epilogue, specialised per variant; a
   * binding offset moves the window instead and the shader is untouched, so the single-
   * chunk and chunked arms run the SAME text. WebGPU wants the offset on
   * `minStorageBufferOffsetAlignment` (256 bytes), and every pair-row offset here is a
   * multiple of 384 f32s (1536 bytes) or 192 words (768 bytes), so it always is - checked
   * rather than assumed, because the failure is a validation error naming a binding.
   */
  async step(key, source, groups, buffers, words, label) {
    const pipeline = await this.pipeline(key, source);
    const entries = [{ binding: 0, resource: { buffer: this.uniform(words) } }];
    buffers.forEach((b, at) => {
      if (!b) return;
      if (b.buffer === undefined) {
        entries.push({ binding: at + 1, resource: { buffer: b } });
        return;
      }
      const align = this.device.limits.minStorageBufferOffsetAlignment ?? 256;
      if (b.offsetBytes % align !== 0) {
        throw new Error(`${key}: binding ${at + 1} offset ${b.offsetBytes} is not a multiple of ${align}`);
      }
      entries.push({ binding: at + 1, resource: { buffer: b.buffer, offset: b.offsetBytes, size: b.sizeBytes } });
    });
    const bindGroup = this.device.createBindGroup({ label: `rfd4p.${key}`, layout: pipeline.getBindGroupLayout(0), entries });
    const [x, y] = fold2(groups);
    return { pipeline, bindGroup, x, y, label: label ?? key };
  }

  /**
   * C = alpha A B^T (+bias)(act)(+C). spec: {m, n, k, a, b, c, bias?, act?: 'relu'|'sigmoid', accumulate?, alpha?,
   * gate?: {sc, off?, stride?, sigmoid?}, pair?: {key, bRow, decls?, body, buffers?}, noSmall?,
   * batches?: [B0, B1], A: {off, s0, s1, m, k}, B: {...n, k}, C: {...m, n}} (strides in elements).
   */
  gemm(spec) {
    const { m, n, k } = spec;
    const [B0, B1] = spec.batches ?? [1, 1];
    const A = { off: 0, s0: 0, s1: 0, m: k, k: 1, ...spec.A };
    const B = { off: 0, s0: 0, s1: 0, n: k, k: 1, ...spec.B };
    const C = { off: 0, s0: 0, s1: 0, m: n, n: 1, ...spec.C };
    const half = spec.b.half === true;
    const b16 = (half || spec.f16 === true) && this.hasF16;
    if (half && (B.k !== 1 || B.off % 2 || B.s0 % 2 || B.s1 % 2 || B.n % 2)) throw new Error('half weight wants even strides');
    if (spec.a === spec.c || spec.b === spec.c) throw new Error('gemm output aliases an input');
    const hasGate = Boolean(spec.gate);
    const addOnly = Boolean(spec.gate?.addOnly);
    const acc = (spec.accumulate ?? Boolean(spec.gate && !spec.gate.mulOnly && !addOnly)) ? 1 : 0;
    const u = new Uint32Array(28);
    const f = new Float32Array(u.buffer);
    u.set([m, n, k, B1, A.off, A.s0, A.s1, A.m, A.k, B.off, B.s0, B.s1, B.n, B.k, C.off, C.s0, C.s1, C.m, C.n,
      spec.bias ? 1 : 0, { relu: 1, sigmoid: 2 }[spec.act] ?? 0, acc]);
    f[22] = spec.alpha ?? 1;
    u[23] = B0 * B1;
    u[24] = spec.gate?.off ?? 0;
    u[25] = spec.gate?.stride ?? n;
    u[26] = addOnly ? 2 : ((spec.gate?.sigmoid ?? true) ? 1 : 0);
    u[27] = spec.pad0 ?? 0;
    const submat = b16 && this.hasSubgroupMatrix;
    const tiles64 = Math.ceil(m / 64) * Math.ceil(n / 128) * B0 * B1;
    // The stock small kernel has no fused input LayerNorm, so an `inNorm: 'ln'` GEMM stays on the tiled one.
    const small = (submat ? !spec.noSmall : !spec.pair && !spec.noSmall && spec.inNorm !== 'ln') &&
      (submat ? (m <= 32 || tiles64 < (k <= 64 ? 100 : 80)) : (m <= 32 || n <= 64 || tiles64 < 32));
    const aF16 = this.hasF16 && spec.aF16 === true;
    const cF16 = this.hasF16 && spec.cF16 === true;
    const gF16 = this.hasF16 && hasGate && spec.gate?.f16 === true;
    // 🔴 THESE THREE TYPE THE BUFFER, NOT THE STAGING TILE, so dropping one on a device without
    // f16 is not a graceful degrade: the caller has already packed halves and the kernel would
    // declare `array<f32>` and read them as floats - the elementwise-B fault that `EBelem`
    // fixes above, one operand along.
    // `b16` is different and needs no guard, because it chooses the TILE's precision and both
    // choices compute the same thing. Every caller gates on `e.hasF16` already (trunk.js asks
    // `wantF16: this.half && e.hasF16`); this is what says so if one stops.
    if (!this.hasF16 && (spec.aF16 === true || spec.cF16 === true || spec.gate?.f16 === true)) {
      throw new Error(`gemm ${m}x${n}x${k}: f16 operands asked of a device without shader-f16`);
    }
    const wordB = half || (aF16 && B.k === 1 && B.n % 2 === 0 && B.off % 2 === 0 && B.s0 % 2 === 0 && B.s1 % 2 === 0);
    const vecBStock = wordB &&
      B.k === 1 && B.n % 8 === 0 && B.off % 8 === 0 && B.s0 % 8 === 0 && B.s1 % 8 === 0 &&
      k % 16 === 0;
    const vecB = submat && half &&
      B.k === 1 && B.n % 8 === 0 && B.off % 8 === 0 && B.s0 % 8 === 0 && B.s1 % 8 === 0 &&
      (k % 16 === 0 || spec.b.stride === B.n);
    const vecIO = submat && (half || aF16) &&
      A.k === 1 && A.m % (aF16 ? 8 : 4) === 0 && A.off % (aF16 ? 8 : 4) === 0 && A.s0 % (aF16 ? 8 : 4) === 0 && A.s1 % (aF16 ? 8 : 4) === 0 &&
      B.k === 1 && B.n % (aF16 && !half ? 16 : 8) === 0 && B.off % (aF16 && !half ? 16 : 8) === 0 && B.s0 % (aF16 && !half ? 16 : 8) === 0 && B.s1 % (aF16 && !half ? 16 : 8) === 0 &&
      k % 16 === 0;
    const groups = submat
      ? Math.ceil(m / (small ? 32 : 64)) * Math.ceil(n / 128) * B0 * B1
      : small
        ? Math.ceil(m / 32) * Math.ceil(n / 64) * B0 * B1
        : Math.ceil(m / 64) * Math.ceil(n / 128) * B0 * B1;
    const gSigm = Boolean(spec.gate?.sigmoid ?? true);
    const hasAlpha = (spec.alpha ?? 1) !== 1;
    const hasBias = Boolean(spec.bias);
    const actType = { relu: 1, sigmoid: 2 }[spec.act] ?? 0;
    const vecC = !spec.pair &&
      C.n === 1 && n % 4 === 0 && C.m % 4 === 0 && C.off % 4 === 0 && C.s0 % 4 === 0 && C.s1 % 4 === 0 &&
      (!hasGate || ((spec.gate.off ?? 0) % 4 === 0 && (spec.gate.stride ?? n) % 4 === 0));
    const vecEpilogue = vecC;
    const fullK = k % 16 === 0 && A.k === 1;
    const vecAStock = fullK && A.m % 4 === 0 && A.off % 4 === 0 && A.s0 % 4 === 0 && A.s1 % 4 === 0;
    const useGemv = this.hasF16 && half && !aF16 && !cF16 && !gF16 && !spec.pair && vecC &&
      (m === 1 || m === 2) && k % 128 === 0 && k <= 1024 &&
      A.k === 1 && A.off % 4 === 0 && A.s0 % 4 === 0 && A.s1 % 4 === 0 && A.m % 4 === 0 &&
      B.k === 1 && B.off % 4 === 0 && B.s0 % 4 === 0 && B.s1 % 4 === 0 && B.n % 4 === 0;
    if (useGemv) {
      const inNorm = spec.inNorm ?? '';
      // 🔴 AND THIS RETURNS BEFORE THE `inLn` CHECK BELOW, so the gemv path needs its own: the
      // two kernel families implement DISJOINT `inNorm` sets, and a norm the selected one does
      // not know was dropped in silence. `gemvSource` does `rmsCond`, `lnCond` and `rmsPre`.
      if (inNorm && !['rmsCond', 'lnCond', 'rmsPre'].includes(inNorm)) {
        throw new Error(`gemm ${m}x${n}x${k}: inNorm '${inNorm}' is not implemented by the gemv kernel this shape selects`);
      }
      const gemvGroups = Math.ceil(n / 16) * B0 * B1;
      const gemvKey = `gemv-m${m}-k${k}${inNorm ? `-${inNorm}` : ''}${hasAlpha ? '-al' : ''}${hasBias ? '-b' : ''}${actType ? `-act${actType}` : ''}${acc ? '-ac' : ''}${hasGate ? (addOnly ? '-ga' : (gSigm ? '-gs' : '-gm')) : ''}`;
      const gemvSrc = gemvSource(m, k / 128, hasGate, addOnly, gSigm, hasAlpha, hasBias, actType, acc === 1, inNorm);
      const bufs = [spec.a, spec.b, spec.c, spec.bias ?? this.dummy, ...(hasGate ? [spec.gate.sc] : (inNorm === 'rmsPre' ? [spec.inNormW] : []))];
      return this.step(gemvKey, gemvSrc, gemvGroups, bufs, u,
        spec.label ?? `gemm ${m}x${n}x${k}${B0 * B1 > 1 ? `[${B0 * B1}]` : ''}`);
    }
    const inLn = (submat ? (vecIO && (vecEpilogue || Boolean(spec.pair))) : (fullK && vecEpilogue && !small)) && spec.inNorm === 'ln';
    // 🔴 AND THE OTHER DIRECTION OF THE SAME HOLE: `gemmSource` and `gemmSubmatSource` implement
    // `'ln'` AND NOTHING ELSE, so `rmsPre` - the only other value any caller asks for - would be
    // dropped in silence by whichever of them a shape selects. It reaches the gemv kernel today
    // only because `useGemv` wants `k % 128 == 0` and this checkpoint's conditioning width is
    // 256; a width of 192 would decline the gemv, take this path, and quietly stop normalising.
    if (spec.inNorm && spec.inNorm !== 'ln') {
      throw new Error(`gemm ${m}x${n}x${k}: inNorm '${spec.inNorm}' is implemented only by the gemv kernel, which this shape does not select`);
    }
    if (spec.inNorm === 'ln' && !inLn) throw new Error(`gemm ${m}x${n}x${k}: inNorm 'ln' is not supported by the kernel this shape selects`);
    const vKey = !submat && !small && this.gemmVariant ? `-v${JSON.stringify(this.gemmVariant)}` : '';
    const key = `${small ? 'gemm-s' : 'gemm'}${submat ? `-sm${this.submat8 ? '8' : ''}${half ? '-h' : '-f'}${aF16 ? '-a16' : ''}${cF16 ? '-c16' : ''}${vecIO ? '-v' : (vecB ? '-vb' : '')}${vecEpilogue ? '-ve' : ''}${inLn ? '-ln' : ''}${hasAlpha ? '-al' : ''}${hasBias ? '-b' : ''}${actType ? `-act${actType}` : ''}${acc ? '-ac' : ''}${hasGate && addOnly ? '-ga' : (hasGate && !gSigm ? '-gns' : '')}` : `${wordB ? '-h' : ''}${b16 ? '-b16' : ''}${aF16 ? '-a16' : ''}${cF16 ? '-c16' : ''}${fullK ? '-fk' : ''}${vecAStock ? '-va' : ''}${vecBStock ? '-vb' : ''}${vecEpilogue ? '-ve' : ''}${inLn ? '-ln' : ''}${hasAlpha ? '-al' : ''}${hasBias ? '-b' : ''}${actType ? `-act${actType}` : ''}${acc ? '-ac' : ''}${hasGate && addOnly ? '-ga' : (hasGate && !gSigm ? '-gns' : '')}${vKey}`}${spec.pair ? `-p-${spec.pair.key}` : ''}${hasGate ? `-g${gF16 ? '16' : ''}` : ''}`;
    const src = submat
      ? gemmSubmatSource(spec.pair ?? null, hasGate, half, small ? 32 : 64, vecIO, aF16, gF16, cF16, addOnly, gSigm, hasAlpha, hasBias, actType, acc === 1, vecIO || vecB, vecEpilogue, inLn, this.submat8)
      : (small ? gemmSmallSource(wordB, b16, hasGate, aF16, cF16, gF16) : gemmSource(wordB, b16, spec.pair ?? null, hasGate, aF16, cF16, gF16, fullK, vecEpilogue, addOnly, gSigm, hasAlpha, hasBias, actType, acc === 1, inLn, this.gemmVariant, vecAStock, vecBStock));
    const bufs = [spec.a, spec.b, spec.c, spec.bias ?? this.dummy, ...(hasGate ? [spec.gate.sc] : []), ...(spec.pair?.buffers ?? [])];
    return this.step(key, src, groups, bufs, u,
      spec.label ?? `gemm ${m}x${n}x${k}${B0 * B1 > 1 ? `[${B0 * B1}]` : ''}`);
  }

  /** y[rows, out] = x[rows, in] W^T (+ bias): a linear layer over contiguous (or strided) rows. */
  async linear({ x, w, y, rows, inC, outC, bias, act, accumulate, gate, inNorm = '', inNormW = null, nextNorm = null, nextNorm2 = null, skipSplitReduce = false, aF16 = false, cF16 = false, xOff = 0, xStride = inC, yOff = 0, yStride = outC, wOff = 0, label }) {
    if (rows <= 512 && inC >= 768 && outC >= 256 && outC <= 1024 && inC % 64 === 0 && !cF16) {
      const splits = inC >= 3072 ? 8 : 4;
      const kSlice = inC / splits;
      const total = rows * outC;
      if (!this.splitBuf || this.splitBuf.floats < splits * total) {
        this.splitBuf = this.buffer('splitBuf', Math.max(splits * total, splits * 512 * 1024));
      }
      const gemmStep = await this.gemm({
        m: rows, n: outC, k: kSlice, a: x, b: w, c: this.splitBuf, batches: [splits, 1], aF16,
        A: { off: xOff, s0: kSlice, m: xStride, k: 1 },
        B: { off: wOff, s0: kSlice, n: inC, k: 1 },
        C: { off: 0, s0: total, m: outC, n: 1 },
        label: label ?? `gemm ${rows}x${outC}x${inC}`,
      });
      if (skipSplitReduce) return gemmStep;
      const hasGate = Boolean(gate);
      const addOnly = Boolean(gate?.addOnly);
      const gOff = gate?.off ?? 0;
      const gStride = gate?.stride ?? outC;
      if (nextNorm && hasGate && (!bias || !nextNorm.b) && !act && yOff === 0 && yStride === outC && outC % 128 === 0) {
        const rms = nextNorm.type === 'rms';
        const hasAdaln = Boolean(nextNorm.adaln);
        const postNorm = Boolean(nextNorm.adaln?.postNorm);
        const hasNorm2 = Boolean(nextNorm2);
        const norm2Rms = Boolean(nextNorm2?.type === 'rms');
        const hasW1 = Boolean(nextNorm.w);
        const hasLinB = Boolean(bias);
        const separateGb = Boolean(hasAdaln && nextNorm.adaln.gb !== gate.sc);
        // Consumers (TokenBlocks.swiglu) only read an f16 `nextNorm.y` on the subgroup-matrix path.
        const outF16 = Boolean(hasAdaln && w.half && this.hasSubgroupMatrix && this.hasF16);
        const u = new Uint32Array(8);
        const f = new Float32Array(u.buffer);
        u[0] = rows;
        u[1] = total / 4;
        u[2] = gOff / 4;
        u[3] = gStride / 4;
        u[4] = (nextNorm.adaln?.off ?? 0) / 4;
        u[5] = (nextNorm.adaln?.stride ?? 2 * outC) / 4;
        u[6] = nextNorm.b ? 1 : 0;
        f[7] = nextNorm.eps ?? 1e-5;
        const bufs = [
          this.splitBuf, y, gate.sc, nextNorm.y, nextNorm.w ?? this.dummy, bias ?? nextNorm.b ?? this.dummy,
          ...(hasNorm2 ? [nextNorm2.y, nextNorm2.wb] : (separateGb ? [nextNorm.adaln.gb] : [])),
        ];
        const reduceStep = await this.step(
          `split-red-norm-${splits}-${outC}-${rms ? 'r' : 'l'}${hasAdaln ? (postNorm ? '-ap' : '-a') : ''}${hasW1 ? '-w' : ''}${hasLinB ? '-lb' : ''}${addOnly ? '-ao' : ''}${separateGb ? '-sg' : ''}${hasNorm2 ? (norm2Rms ? '-n2r' : '-n2') : ''}${outF16 ? '-h' : ''}`,
          splitReduceNormSource(splits, outC, rms, hasAdaln, postNorm, hasNorm2, hasW1, hasLinB, addOnly, separateGb, norm2Rms, outF16),
          Math.ceil(rows / 4),
          bufs,
          u,
          `splitReduce+norm ${rows}x${outC}`,
        );
        return {
          steps: [gemmStep, reduceStep],
          run(pass) {
            for (const s of this.steps) {
              pass.setPipeline(s.pipeline);
              pass.setBindGroup(0, s.bindGroup);
              pass.dispatchWorkgroups(s.x, s.y);
            }
          },
        };
      }
      const vec4 = total % 4 === 0 && outC % 4 === 0 && yOff % 4 === 0 && yStride % 4 === 0 &&
        (!hasGate || (gOff % 4 === 0 && gStride % 4 === 0));
      const acc = (accumulate ?? Boolean(gate && !gate.mulOnly && !addOnly)) ? 1 : 0;
      const u = new Uint32Array([
        total, outC, yOff, yStride, bias ? 1 : 0, { relu: 1, sigmoid: 2 }[act] ?? 0, acc,
        gOff, gStride, addOnly ? 2 : ((gate?.sigmoid ?? true) ? 1 : 0), 0, 0,
      ]);
      const bufs = [this.splitBuf, y, bias ?? this.dummy, ...(hasGate ? [gate.sc] : [])];
      const reduceStep = await this.step(
        `split-red-${splits}${hasGate ? '-g' : ''}${vec4 ? '-v4' : ''}`,
        splitReduceSource(splits, hasGate, vec4),
        Math.ceil((vec4 ? total / 4 : total) / 256),
        bufs,
        u,
        `splitReduce ${rows}x${outC}`,
      );
      return {
        steps: [gemmStep, reduceStep],
        run(pass) {
          for (const s of this.steps) {
            pass.setPipeline(s.pipeline);
            pass.setBindGroup(0, s.bindGroup);
            pass.dispatchWorkgroups(s.x, s.y);
          }
        },
      };
    }
    const bStride = w.stride ?? inC;
    const gemmStep = await this.gemm({ m: rows, n: outC, k: inC, a: x, b: w, c: y, bias, act, accumulate, gate, inNorm, inNormW, aF16, cF16, label,
      A: { off: xOff, m: xStride, k: 1 }, B: { off: wOff, n: bStride, k: 1 }, C: { off: yOff, m: yStride, n: 1 } });
    if (!nextNorm && !nextNorm2) return gemmStep;
    const steps = [gemmStep];
    if (nextNorm) steps.push(await this.norm(nextNorm));
    if (nextNorm2) steps.push(await this.norm(nextNorm2));
    return {
      steps,
      run(pass) {
        for (const s of this.steps) {
          pass.setPipeline(s.pipeline);
          pass.setBindGroup(0, s.bindGroup);
          pass.dispatchWorkgroups(s.x, s.y);
        }
      },
    };
  }

  async qkvgQkNorm({ x, w, bias, qkv, qk, normType, qkNormW, qkNormB = null, rows, inF16 = false, outF16 = false, eps = 1e-5 }) {
    const D = 768;
    const outC = 4 * D;
    if (rows <= 512 && w.half === true && this.hasF16) {
      const splits = 2;
      const kSlice = D / splits;
      const total = rows * outC;
      if (!this.splitBuf || this.splitBuf.floats < splits * total) {
        this.splitBuf = this.buffer('splitBuf', Math.max(splits * total, 8 * 512 * 1024));
      }
      const gemmStep = await this.gemm({
        m: rows, n: outC, k: kSlice, a: x, b: w, c: this.splitBuf, batches: [splits, 1], aF16: inF16,
        A: { off: 0, s0: kSlice, m: D, k: 1 },
        B: { off: 0, s0: kSlice, n: D, k: 1 },
        C: { off: 0, s0: total, m: outC, n: 1 },
        label: `gemm ${rows}x${outC}x${D}`,
      });
      const rms = normType === 'rms';
      const hasB = Boolean(qkNormB);
      const f16Out = Boolean(outF16 && this.hasF16);
      const u = new Uint32Array(4);
      const f = new Float32Array(u.buffer);
      u[0] = rows;
      u[1] = total / 4;
      f[2] = eps;
      const reduceStep = await this.step(
        `split-red-qknorm-${splits}-${rms ? 'r' : 'l'}${hasB ? '-b' : ''}${f16Out ? '-h' : ''}`,
        splitReduceQkNormSource(splits, rms, hasB, f16Out),
        rows,
        [this.splitBuf, bias, qkv, qk, qkNormW, ...(hasB ? [qkNormB] : [])],
        u,
        `${normType}2 ${2 * rows}x${D}`,
      );
      return [gemmStep, reduceStep];
    }
    return [
      await this.linear({ x, w, y: qkv, rows, inC: D, outC, bias }),
      await this.norm({ type: normType, x: qkv, y: qk, rows: 2 * rows, width: D, w: qkNormW, b: qkNormB, dual: true, outF16, xStride: outC, yStride: 2 * D }),
    ];
  }

  /**
   * Attention `att[i, h*64 + d] = (sum_j P[h, i, j] V[j, h*64 + d]) * sigm(G[i, h*64 + d])`.
   */
  attPv({ logits, qkv, att, n, half = false, vgF16 = false }) {
    const nn = n * n;
    if (half && this.hasSubgroupMatrix) {
      const f16Vg = Boolean(vgF16 && this.hasF16);
      const u = new Uint32Array([n, nn, 0, 0]);
      return this.step(`att-pv-sm${this.submat8 ? '8' : ''}${f16Vg ? '-h' : ''}`, attPvSource(f16Vg, this.submat8), Math.ceil(n / 32) * 12, [logits, qkv, att], u, 'pv');
    }
    return this.gemm({
      m: n, n: 64, k: n, a: logits, b: qkv, c: att, batches: [12, 1],
      gate: { sc: qkv, off: 3 * 768, stride: 4 * 768, sigmoid: true, mulOnly: true },
      A: { s0: nn, m: n, k: 1 }, B: { off: 2 * 768, s0: 64, n: 1, k: 4 * 768 }, C: { s0: 64, m: 768, n: 1 }, label: 'pv',
    });
  }

  /**
   * Fused SwiGLU projection: `y[rows, inner] = silu(x W_gate^T) * (x W_val^T)` where `w = [W_val; W_gate]` (`[2*inner, inC]`).
   * Computes `gate * sigm(gate) * val` in registers in the GEMM epilogue without materializing `[rows, 2*inner]`.
   */
  swigluLinear({ x, w, y, rows, inC, inner, aF16 = false, cF16 = false, label }) {
    const submat = this.hasSubgroupMatrix && w.half === true;
    const f16Out = this.hasF16 && w.half === true && cF16;
    if (this.hasF16 && w.half === true && !aF16 && !f16Out && (rows === 1 || rows === 2) && inner % 4 === 0 && inC % 128 === 0 && inC <= 1024) {
      const u = new Uint32Array([rows, inner, inC, 0]);
      const key = `swiglu-gemv-m${rows}-n${inner}-k${inC}`;
      return this.step(key, swigluGemvSource(rows, inC / 128, inner), Math.ceil(inner / 16), [x, w, y], u,
        label ?? `swigluLinear ${rows}x${inner}x${inC}`);
    }
    return this.gemm({
      m: rows, n: 2 * inner, k: inC, a: x, b: w, c: y, aF16, cF16: f16Out,
      A: { off: 0, m: inC, k: 1 }, B: { off: 0, n: inC, k: 1 }, C: { off: 0, m: inner, n: 1 },
      pair: {
        key: `swiglu${inner}${f16Out ? '-h' : ''}`,
        cVec4: true,
        cF16: f16Out,
        bRow: inner === 3072 ? undefined : `return ((v / 64u) % 2u) * ${inner}u + (v / 128u) * 64u + (v % 64u);`,
        body: submat ? '' : `fn pairOut(m: u32, n0: u32, tn0: u32, r0: vec4<f32>, r1: vec4<f32>) {
  let y = r1 / (vec4<f32>(1.0) + exp(-r1)) * r0;
  let base4 = m * ${inner / 4}u + (n0 / 8u) + (tn0 / 4u);
  c[base4] = ${f16Out ? 'vec2<u32>(bitcast<u32>(vec2<f16>(y.xy)), bitcast<u32>(vec2<f16>(y.zw)))' : 'y'};
}`,
        singleBlockBody: (bIdx) => `{
      let rSw = li / 16u;
      let cSw4 = li % 16u;
      let cSw = cSw4 * 4u;
      let m = m0 + ${bIdx * 16}u + rSw;
      if (m < p.m) {
        let sb = rSw * 136u + cSw;
        let v0 = vec4<f32>(f32(scratch[sb]), f32(scratch[sb + 1u]), f32(scratch[sb + 2u]), f32(scratch[sb + 3u]));
        let v1 = vec4<f32>(f32(scratch[sb + 64u]), f32(scratch[sb + 65u]), f32(scratch[sb + 66u]), f32(scratch[sb + 67u]));
        let y = (v1 / (vec4<f32>(1.0) + exp(-v1))) * v0;
        c[m * ${inner / 4}u + (n0 / 8u) + cSw4] = ${f16Out ? 'vec2<u32>(bitcast<u32>(vec2<f16>(y.xy)), bitcast<u32>(vec2<f16>(y.zw)))' : 'y'};
      }
    }`,
        blockBody: (pIdx) => `{
      let rSw = li / 16u;
      let cSw4 = li % 16u;
      let cSw = cSw4 * 4u;
      ${[0, 1].map((s) => `{
        let m = m0 + ${pIdx * 32 + s * 16}u + rSw;
        if (m < p.m) {
          let sb = (rSw + ${s * 16}u) * 136u + cSw;
          let v0 = vec4<f32>(f32(scratch[sb]), f32(scratch[sb + 1u]), f32(scratch[sb + 2u]), f32(scratch[sb + 3u]));
          let v1 = vec4<f32>(f32(scratch[sb + 64u]), f32(scratch[sb + 65u]), f32(scratch[sb + 66u]), f32(scratch[sb + 67u]));
          let y = (v1 / (vec4<f32>(1.0) + exp(-v1))) * v0;
          c[m * ${inner / 4}u + (n0 / 8u) + cSw4] = ${f16Out ? 'vec2<u32>(bitcast<u32>(vec2<f16>(y.xy)), bitcast<u32>(vec2<f16>(y.zw)))' : 'y'};
        }
      }`).join('\n      ')}
    }`,
      },
      label: label ?? `swigluLinear ${rows}x${inner}x${inC}`,
    });
  }

  /**
   * Fused triangle input projection + output gate (`w = [a_p; a_g; b_p; b_g; g]` of shape `[1408, 384]`):
   * tiles 0..7 write `a = a_p * sigm(a_g)` and `b = b_p * sigm(b_g)` (`[rows, 512]`) to `proj`;
   * tiles 8..10 write `sigm(g)` (`[rows, 384]`) to `gz`.
   */
  /**
   * The fused pair projection: a, b and the output gate in one m x 1408 x 384 GEMM.
   *
   * 🔴 `rowOff` IS WHAT LETS A CHUNK WRITE aT/bT DIRECTLY, so the chunked arm needs neither
   * a `proj` staging buffer nor a scatter pass over it. It rides on `C.off`, which the pair
   * epilogue does not otherwise read (that path indexes `c[cBase * p.sC0 + mPad]` with no
   * `offC` term), so one uniform carries it and the shader text is the same for both arms -
   * at `rowOff = 0` the added term is `m + 0` and the single-chunk arm is unchanged.
   *
   * It has to be added BEFORE the row index is divided into the padded layout, not after:
   * `mPad` is `(m / sCn) * sCm + m % sCn`, so a flat offset on the result would only be
   * right when r0 happened to be a multiple of n.
   *
   * `batchStride` is the per-channel slice, which the chunked arm pads in BOTH dimensions
   * (nPad x nPad) because its incoming contraction reads the operand transposed by stride
   * and so reaches row nPad - 1.
   */
  triProjG({ x, w, wSum = null, proj, aT, bT, gz, rows, incoming = false, n = 0, nPad = n, rowOff = 0, batchStride = n * nPad, aF16 = false, outF16 = false, label }) {
    if (aT && bT) {
      const f16Out = outF16 && this.hasF16 && w.half === true;
      const submat = this.hasSubgroupMatrix && w.half === true;
      const useTbOut = !submat && w.half === true && this.hasF16;
      const inLn = Boolean(wSum && submat);
      const t = f16Out ? 'f16' : 'f32';
      const shArr = useTbOut ? 'tb' : 'shOut';
      const shType = useTbOut ? 'f16' : t;
      const gzT = f16Out ? 'vec2<u32>' : 'vec4<f32>';
      const triEpi2 = !this.gemmVariant?.baseline && (this.gemmVariant?.triEpi2 ?? true);
      return this.gemm({
        m: rows, n: 1408, k: 384, a: x, b: w, c: aT, bias: wSum ?? undefined, inNorm: inLn ? 'ln' : '', noSmall: true, aF16, pad0: incoming ? n : 0,
        A: { off: 0, m: 384, k: 1 }, B: { off: 0, n: 384, k: 1 }, C: { off: rowOff, s0: batchStride, m: nPad, n },
        pair: {
          key: `triprojg-ab${f16Out ? '-h' : ''}${inLn ? '-ln' : ''}`,
          colMajorStore: 'cond1024',
          cF16: f16Out,
          buffers: [gz, bT],
          decls: `@group(0) @binding(5) var<storage, read_write> gz: array<${gzT}>;
@group(0) @binding(6) var<storage, read_write> bT: array<${t}>;${submat || useTbOut ? '' : `\nvar<workgroup> shOut: array<vec4<${t}>, 512>;`}`,
          body: '',
          epilogueBody: submat ? undefined : (triEpi2 ? `if (n0 >= 1024u) {
    if (inBatch) {
      let nColOff4 = (n0 - 1024u) / 4u + tx;
      ${[0, 1, 2, 3].map((i) => `{
        let mRow = m0 + tm + ${i}u;
        if (mRow < p.m) {
          let g0 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-acc_${i}_0));
          let g1 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-acc_${i}_1));
          let mOut = select(mRow, (mRow % p.pad0) * p.pad0 + (mRow / p.pad0), p.pad0 != 0u);
          let base4 = mOut * 96u + nColOff4;
          gz[base4] = ${f16Out ? 'vec2<u32>(bitcast<u32>(vec2<f16>(g0.xy)), bitcast<u32>(vec2<f16>(g0.zw)))' : 'g0'};
          gz[base4 + 16u] = ${f16Out ? 'vec2<u32>(bitcast<u32>(vec2<f16>(g1.xy)), bitcast<u32>(vec2<f16>(g1.zw)))' : 'g1'};
        }
      }`).join('\n      ')}
    }
    return;
  }
  let y_0 = acc_0_0 / (vec4<f32>(1.0) + exp(-acc_0_1));
  let y_1 = acc_1_0 / (vec4<f32>(1.0) + exp(-acc_1_1));
  let y_2 = acc_2_0 / (vec4<f32>(1.0) + exp(-acc_2_1));
  let y_3 = acc_3_0 / (vec4<f32>(1.0) + exp(-acc_3_1));
  if (tx < 8u) {
    let c0 = tx * 4u;
    ${shArr}[c0 * 16u + tm4] = vec4<${shType}>(${shType}(y_0.x), ${shType}(y_1.x), ${shType}(y_2.x), ${shType}(y_3.x));
    ${shArr}[(c0 + 1u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.y), ${shType}(y_1.y), ${shType}(y_2.y), ${shType}(y_3.y));
    ${shArr}[(c0 + 2u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.z), ${shType}(y_1.z), ${shType}(y_2.z), ${shType}(y_3.z));
    ${shArr}[(c0 + 3u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.w), ${shType}(y_1.w), ${shType}(y_2.w), ${shType}(y_3.w));
  }
  workgroupBarrier();
  let mLocal = li % 64u;
  let cSub = li / 64u;
  let m = m0 + mLocal;
  let mValid = inBatch && (m < p.m);
  let mg = m + p.offC; let mPad = select(mg, (mg / p.sCn) * p.sCm + (mg % p.sCn), p.sCm != p.sCn);
  let r4 = mLocal / 4u;
  let comp = mLocal % 4u;
  if (mValid) {
    if (n0 < 512u) {
      let cBase0 = (n0 / 2u) + cSub;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `c[(cBase0 + ${k * 4}u) * p.sC0 + mPad] = ${t}(${shArr}[(cSub + ${k * 4}u) * 16u + r4][comp]);`).join('\n      ')}
    } else {
      let cBase0 = ((n0 - 512u) / 2u) + cSub;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `bT[(cBase0 + ${k * 4}u) * p.sC0 + mPad] = ${t}(${shArr}[(cSub + ${k * 4}u) * 16u + r4][comp]);`).join('\n      ')}
    }
  }
  workgroupBarrier();
  if (tx >= 8u) {
    let c0 = (tx - 8u) * 4u;
    ${shArr}[c0 * 16u + tm4] = vec4<${shType}>(${shType}(y_0.x), ${shType}(y_1.x), ${shType}(y_2.x), ${shType}(y_3.x));
    ${shArr}[(c0 + 1u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.y), ${shType}(y_1.y), ${shType}(y_2.y), ${shType}(y_3.y));
    ${shArr}[(c0 + 2u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.z), ${shType}(y_1.z), ${shType}(y_2.z), ${shType}(y_3.z));
    ${shArr}[(c0 + 3u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.w), ${shType}(y_1.w), ${shType}(y_2.w), ${shType}(y_3.w));
  }
  workgroupBarrier();
  if (mValid) {
    if (n0 < 512u) {
      let cBase0 = (n0 / 2u) + 32u + cSub;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `c[(cBase0 + ${k * 4}u) * p.sC0 + mPad] = ${t}(${shArr}[(cSub + ${k * 4}u) * 16u + r4][comp]);`).join('\n      ')}
    } else {
      let cBase0 = ((n0 - 512u) / 2u) + 32u + cSub;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `bT[(cBase0 + ${k * 4}u) * p.sC0 + mPad] = ${t}(${shArr}[(cSub + ${k * 4}u) * 16u + r4][comp]);`).join('\n      ')}
    }
  }` : `let y_0 = (acc_0_0 * p.alpha) / (vec4<f32>(1.0) + exp(-(acc_0_1 * p.alpha)));
  let y_1 = (acc_1_0 * p.alpha) / (vec4<f32>(1.0) + exp(-(acc_1_1 * p.alpha)));
  let y_2 = (acc_2_0 * p.alpha) / (vec4<f32>(1.0) + exp(-(acc_2_1 * p.alpha)));
  let y_3 = (acc_3_0 * p.alpha) / (vec4<f32>(1.0) + exp(-(acc_3_1 * p.alpha)));
  if (tx < 8u) {
    let c0 = tx * 4u;
    ${shArr}[c0 * 16u + tm4] = vec4<${shType}>(${shType}(y_0.x), ${shType}(y_1.x), ${shType}(y_2.x), ${shType}(y_3.x));
    ${shArr}[(c0 + 1u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.y), ${shType}(y_1.y), ${shType}(y_2.y), ${shType}(y_3.y));
    ${shArr}[(c0 + 2u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.z), ${shType}(y_1.z), ${shType}(y_2.z), ${shType}(y_3.z));
    ${shArr}[(c0 + 3u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.w), ${shType}(y_1.w), ${shType}(y_2.w), ${shType}(y_3.w));
  }
  workgroupBarrier();
  let mLocal = li % 64u;
  let cSub = li / 64u;
  let m = m0 + mLocal;
  let mValid = inBatch && (m < p.m) && (n0 < 1024u);
  let mg = m + p.offC; let mPad = select(mg, (mg / p.sCn) * p.sCm + (mg % p.sCn), p.sCm != p.sCn);
  let r4 = mLocal / 4u;
  let comp = mLocal % 4u;
  if (mValid) {
    if (n0 < 512u) {
      let cBase0 = (n0 / 2u) + cSub;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `c[(cBase0 + ${k * 4}u) * p.sC0 + mPad] = ${t}(${shArr}[(cSub + ${k * 4}u) * 16u + r4][comp]);`).join('\n      ')}
    } else {
      let cBase0 = ((n0 - 512u) / 2u) + cSub;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `bT[(cBase0 + ${k * 4}u) * p.sC0 + mPad] = ${t}(${shArr}[(cSub + ${k * 4}u) * 16u + r4][comp]);`).join('\n      ')}
    }
  }
  workgroupBarrier();
  if (tx >= 8u) {
    let c0 = (tx - 8u) * 4u;
    ${shArr}[c0 * 16u + tm4] = vec4<${shType}>(${shType}(y_0.x), ${shType}(y_1.x), ${shType}(y_2.x), ${shType}(y_3.x));
    ${shArr}[(c0 + 1u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.y), ${shType}(y_1.y), ${shType}(y_2.y), ${shType}(y_3.y));
    ${shArr}[(c0 + 2u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.z), ${shType}(y_1.z), ${shType}(y_2.z), ${shType}(y_3.z));
    ${shArr}[(c0 + 3u) * 16u + tm4] = vec4<${shType}>(${shType}(y_0.w), ${shType}(y_1.w), ${shType}(y_2.w), ${shType}(y_3.w));
  }
  workgroupBarrier();
  if (mValid) {
    if (n0 < 512u) {
      let cBase0 = (n0 / 2u) + 32u + cSub;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `c[(cBase0 + ${k * 4}u) * p.sC0 + mPad] = ${t}(${shArr}[(cSub + ${k * 4}u) * 16u + r4][comp]);`).join('\n      ')}
    } else {
      let cBase0 = ((n0 - 512u) / 2u) + 32u + cSub;
      ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `bT[(cBase0 + ${k * 4}u) * p.sC0 + mPad] = ${t}(${shArr}[(cSub + ${k * 4}u) * 16u + r4][comp]);`).join('\n      ')}
    }
  }
  if (inBatch && n0 >= 1024u) {
    let nColOff4 = (n0 - 1024u) / 4u + tx;
    ${[0, 1, 2, 3].map((i) => `{
      let mRow = m0 + tm + ${i}u;
      if (mRow < p.m) {
        let g0 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-(acc_${i}_0 * p.alpha)));
        let g1 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-(acc_${i}_1 * p.alpha)));
        let mOut = select(mRow, (mRow % p.pad0) * p.pad0 + (mRow / p.pad0), p.pad0 != 0u);
        let base4 = mOut * 96u + nColOff4;
        gz[base4] = ${f16Out ? 'vec2<u32>(bitcast<u32>(vec2<f16>(g0.xy)), bitcast<u32>(vec2<f16>(g0.zw)))' : 'g0'};
        gz[base4 + 16u] = ${f16Out ? 'vec2<u32>(bitcast<u32>(vec2<f16>(g1.xy)), bitcast<u32>(vec2<f16>(g1.zw)))' : 'g1'};
      }
    }`).join('\n    ')}
  }`),
          blockBody: (pIdx) => `if (n0 < 1024u) {
      let rr = li % 32u;
      let cSub = li / 32u;
      let rLocal = ${pIdx * 32}u + rr;
      let m = m0 + rLocal;
      if (m < p.m) {
        ${inLn ? (this.submat8 ? `let mean = lnStat[rLocal * 2u];
        let inv = lnStat[rLocal * 2u + 1u];` : `let mean = f32(scratch[rLocal * 34u + 32u]);
        let inv = f32(scratch[rLocal * 34u + 33u]);`) : ''}
        let mg = m + p.offC; let mPad = select(mg, (mg / p.sCn) * p.sCm + (mg % p.sCn), p.sCm != p.sCn);
        let s8 = 8u * p.sC0;
        if (n0 < 512u) {
          let c0 = ((n0 / 2u) + cSub) * p.sC0 + mPad;
          ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => inLn
            ? `{
            let vp = (f32(scratch[(cSub + ${k * 8}u) * 34u + rr]) - mean * bias[n0 + cSub + ${k * 8}u]) * inv;
            let vg = (f32(scratch[(cSub + ${k * 8 + 64}u) * 34u + rr]) - mean * bias[n0 + cSub + ${k * 8 + 64}u]) * inv;
            c[c0 + ${k}u * s8] = ${t}(vp / (1.0 + exp(-vg)));
          }`
            : `c[c0 + ${k}u * s8] = ${t}(f32(scratch[(cSub + ${k * 8}u) * 34u + rr]) / (1.0 + exp(-f32(scratch[(cSub + ${k * 8 + 64}u) * 34u + rr]))));`).join('\n          ')}
        } else {
          let c0 = (((n0 - 512u) / 2u) + cSub) * p.sC0 + mPad;
          ${[0, 1, 2, 3, 4, 5, 6, 7].map((k) => inLn
            ? `{
            let vp = (f32(scratch[(cSub + ${k * 8}u) * 34u + rr]) - mean * bias[n0 + cSub + ${k * 8}u]) * inv;
            let vg = (f32(scratch[(cSub + ${k * 8 + 64}u) * 34u + rr]) - mean * bias[n0 + cSub + ${k * 8 + 64}u]) * inv;
            bT[c0 + ${k}u * s8] = ${t}(vp / (1.0 + exp(-vg)));
          }`
            : `bT[c0 + ${k}u * s8] = ${t}(f32(scratch[(cSub + ${k * 8}u) * 34u + rr]) / (1.0 + exp(-f32(scratch[(cSub + ${k * 8 + 64}u) * 34u + rr]))));`).join('\n          ')}
        }
      }
    } else {
      let rE = li / 32u;
      let c4 = li % 32u;
      ${inLn ? `let nCol = n0 + c4 * 4u;
      let wSum4 = vec4<f32>(bias[nCol], bias[nCol + 1u], bias[nCol + 2u], bias[nCol + 3u]);` : ''}
      let nColOff4 = (n0 - 1024u) / 4u + c4;
      ${[0, 1, 2, 3].map((s) => `{
        let rLocal = ${pIdx * 32 + s * 8}u + rE;
        let m = m0 + rLocal;
        if (m < p.m) {
          let sb = (rE + ${s * 8}u) * 136u + c4 * 4u;
          let raw4 = vec4<f32>(f32(scratch[sb]), f32(scratch[sb + 1u]), f32(scratch[sb + 2u]), f32(scratch[sb + 3u]));
          ${inLn ? (this.submat8 ? `let vg4 = (raw4 - lnStat[rLocal * 2u] * wSum4) * lnStat[rLocal * 2u + 1u];` : `let scIdx = (rLocal / 4u) * 136u + 128u + (rLocal % 4u) * 2u;
          let vg4 = (raw4 - f32(scratch[scIdx]) * wSum4) * f32(scratch[scIdx + 1u]);`) : 'let vg4 = raw4;'}
          let g = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-vg4));
          let mOut = select(m, (m % p.pad0) * p.pad0 + (m / p.pad0), p.pad0 != 0u);
          let at4 = mOut * 96u + nColOff4;
          gz[at4] = ${f16Out ? 'vec2<u32>(bitcast<u32>(vec2<f16>(g.xy)), bitcast<u32>(vec2<f16>(g.zw)))' : 'g'};
        }
      }`).join('\n      ')}
    }`,
        },
        label: label ?? `triProjG ${rows}x1408x384`,
      });
    }
    // 🔴 THE CHUNKED FORM NARROWS TOO, and both of its outputs are written ELEMENTWISE
    // rather than packed. `proj` is the gemm's own C, so `cF16` decides its type; `gz` is
    // this epilogue's, and the gate that later reads it indexes per element and casts
    // (`f32(gsc[...])`), so an `array<f16>` is exactly what it wants - the fused arm's
    // `vec2<u32>` store is the same bytes seen four halves at a time, not another layout.
    const projF16 = outF16 && this.hasF16;
    const pt = projF16 ? 'f16' : 'f32';
    const cast = projF16 ? 'f16' : '';
    return this.gemm({
      m: rows, n: 1408, k: 384, a: x, b: w, c: proj, noSmall: true, aF16, cF16: projF16,
      A: { off: 0, m: 384, k: 1 }, B: { off: 0, n: 384, k: 1 }, C: { off: 0, m: 512, n: 1 },
      pair: {
        key: `triprojg${projF16 ? '-h' : ''}`,
        buffers: [gz],
        decls: `@group(0) @binding(5) var<storage, read_write> gz: array<${pt}>;`,
        body: `fn pairOutCol(m: u32, n0: u32, col: u32, v0: f32, v1: f32) {
  if (n0 < 1024u) {
    c[m * 512u + (n0 / 2u) + col] = ${cast}(v0 / (1.0 + exp(-v1)));
  } else {
    let base0 = m * 384u + (n0 - 1024u) + col;
    gz[base0] = ${cast}(1.0 / (1.0 + exp(-v0)));
    gz[base0 + 64u] = ${cast}(1.0 / (1.0 + exp(-v1)));
  }
}
fn pairOut(m: u32, n0: u32, tn0: u32, r0: vec4<f32>, r1: vec4<f32>) {
  if (n0 < 1024u) {
    let y = r0 / (vec4<f32>(1.0) + exp(-r1));
    let base = m * 512u + (n0 / 2u) + tn0;
    c[base] = ${cast}(y.x); c[base + 1u] = ${cast}(y.y); c[base + 2u] = ${cast}(y.z); c[base + 3u] = ${cast}(y.w);
  } else {
    let g0 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-r0));
    let g1 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-r1));
    let base0 = m * 384u + (n0 - 1024u) + tn0;
    let base1 = base0 + 64u;
    gz[base0] = ${cast}(g0.x); gz[base0 + 1u] = ${cast}(g0.y); gz[base0 + 2u] = ${cast}(g0.z); gz[base0 + 3u] = ${cast}(g0.w);
    gz[base1] = ${cast}(g1.x); gz[base1 + 1u] = ${cast}(g1.y); gz[base1 + 2u] = ${cast}(g1.z); gz[base1 + 3u] = ${cast}(g1.w);
  }
}`,
      },
      label: label ?? `triProjG ${rows}x1408x384`,
    });
  }

  /** Row norm (`type` 'rms' | 'ln'); `x === y` runs in place; optional `adaln: { gb, off?, stride? }` fuses modulation. */
  norm({ type, x, y, rows, width, w, b, adaln, copyXTo = null, preNorm: preNormOpt = false, dual = false, inF16 = false, outF16 = false, transposeN = 0, xStride = width, yStride = width, xOff = 0, yOff = 0, eps = 1e-5 }) {
    const u = new Uint32Array(12);
    const f = new Float32Array(u.buffer);
    u.set([rows, width, xStride, yStride, xOff, yOff, w ? (dual ? 2 : 1) : 0, b ? 1 : 0]);
    f[8] = eps;
    u[9] = adaln?.off ?? 0;
    u[10] = adaln?.stride ?? 2 * width;
    u[11] = transposeN;
    const inPlace = x === y;
    const hasAdaln = Boolean(adaln);
    const postNorm = Boolean(adaln?.postNorm);
    const preNorm = Boolean(adaln?.preNorm || preNormOpt);
    const copyX = Boolean(copyXTo && postNorm);
    const specWidth = width % 128 === 0 && width <= 1024 ? width : 0;
    const f16In = inF16 && !inPlace && specWidth > 0 && this.hasF16;
    const f16Out = outF16 && !inPlace && specWidth > 0 && this.hasF16;
    const groups = specWidth > 0 ? Math.ceil(rows / 4) : rows;
    const hasW = Boolean(w);
    const hasB = Boolean(b);
    const transpose = transposeN > 0 && !dual;
    const needW = specWidth === 0 || hasW || postNorm || preNorm;
    const needB = specWidth === 0 || hasB;
    const key = `norm-${type}-${specWidth}${inPlace ? '-ip' : ''}${f16In ? '-ih' : ''}${f16Out ? '-h' : ''}${dual ? '-d' : ''}${hasW ? '-w' : ''}${hasB ? '-b' : ''}${transpose ? '-tr' : ''}${hasAdaln ? `-ada${postNorm ? '-post' : ''}` : ''}${preNorm ? '-pre' : ''}${copyX ? '-cx' : ''}`;
    return this.step(key, normSource(type === 'rms', inPlace, specWidth, hasAdaln, postNorm, preNorm, f16Out, copyX, dual, hasW, hasB, transpose, f16In), groups,
      [inPlace ? null : x, y, needW ? (w ?? this.dummy) : null, needB ? (b ?? this.dummy) : null, ...(hasAdaln ? [adaln.gb] : []), ...(copyX ? [copyXTo] : [])], u,
      `${type}${dual ? '2' : ''}${hasAdaln ? `+ada${postNorm ? '+n' : ''}` : ''}${preNorm ? '+p' : ''} ${rows}x${width}`);
  }

  /** Fused pair RMSNorm(384) + linear projection to 12 attention bias heads (`[12, nn]`), optionally for 2 consecutive layers (`bias2`), `zInitAda`, or `outZc`. */
  pairToBias({ z, wBias, bias, bias2 = null, zInitAda = null, outZc = null, zF16 = false, rows, nn, r0 = 0, eps = 1e-5 }) {
    const half = wBias.half === true;
    const dual = Boolean(bias2);
    const hasZInitAda = Boolean(zInitAda);
    const hasOutZc = Boolean(outZc);
    const zcF16 = Boolean(outZc?.f16 && this.hasF16);
    const f16Z = Boolean(zF16 && this.hasF16);
    const bF16 = Boolean(half && this.hasSubgroupMatrix && this.hasF16 && nn <= 512 * 512);
    const useSubmat = half && f16Z && this.hasSubgroupMatrix && !hasZInitAda && !hasOutZc;
    const groups = Math.ceil(rows / (useSubmat ? 64 : 32));
    const u = new Uint32Array(8);
    const f = new Float32Array(u.buffer);
    u[0] = rows;
    u[1] = nn;
    u[2] = r0;
    f[3] = 384 * eps;
    u[4] = zInitAda?.n ?? 0;
    u[5] = zInitAda?.seqStride ?? 0;
    return this.step(
      `pair-to-bias${useSubmat ? `-sm${this.submat8 ? '8' : ''}` : ''}${half ? '-h' : ''}${f16Z ? '-z16' : ''}${bF16 ? '-b16' : ''}${dual ? '-d' : ''}${hasZInitAda ? '-za' : ''}${hasOutZc ? `-zc${zcF16 ? '16' : ''}` : ''}`,
      useSubmat
        ? pairToBiasSubmatSource(dual, f16Z, bF16, this.submat8)
        : pairToBiasSource(half, dual, hasZInitAda, hasOutZc, zcF16, f16Z, bF16),
      groups,
      [
        z,
        wBias,
        bias,
        ...(dual ? [bias2] : []),
        ...(hasZInitAda ? [zInitAda.wLnBins, zInitAda.gb, zInitAda.zStatic, zInitAda.seqF] : []),
        ...(hasOutZc ? [outZc.wZc, outZc.zc] : []),
      ],
      u,
      'pairToBias',
    );
  }

  softmax({ l, rows, len, bias, scale = 1, lF16 = false }) {
    const hasBias = Boolean(bias);
    const hasScale = scale !== 1;
    const f16L = Boolean(lF16 && this.hasF16);
    const K = len <= 128 ? 4 : (len <= 256 ? 8 : (len <= 512 ? 16 : 0));
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u.set([rows, len, hasBias ? 1 : 0]);
    f[3] = scale;
    const key = `softmax-${K}${hasBias ? '-b' : ''}${hasScale ? '-s' : ''}${f16L ? '-h' : ''}`;
    return this.step(key, softmaxSource(K, hasBias, hasScale, f16L), Math.ceil(rows / 4), [l, ...(hasBias ? [bias] : [])], u, `softmax ${rows}x${len}`);
  }

  /** Coalesced 16x16 tiled gate + transpose for triangle multiplication (`proj -> aT, bT [256, n, n]`). */
  triGate({ proj, aT, bT, n, incoming, preGated = false }) {
    const tilesI = Math.ceil(n / 16);
    const groups = n * tilesI * 16;
    const u = new Uint32Array([n, tilesI, groups, 0]);
    const label = `${incoming ? 'tri-gate-in' : 'tri-gate-out'}${preGated ? '-512' : ''}`;
    return this.step(label, triGateSource(incoming, preGated), groups, [proj, aT, bT], u, label);
  }

  /** Fused `[256, nn] -> [rows, 256]` coalesced transpose + LayerNorm(width=256) for triangle output. */
  /**
   * `y = LayerNorm_w(x^T)` over 256 channels, reading `x` channel-major.
   *
   * 🔴 `r0` MUST BE A MULTIPLE OF FOUR. `x` is read as `array<vec4<f32>>` along the ROW
   * axis - `x[c * nnStride4 + (r0 + rLocal) / 4]` - so four consecutive pair rows share a
   * word and a chunk starting off that boundary reads every row of every channel from the
   * wrong place. It is not a tidiness alignment: misaligned it is relRMS ~1.0 on the pair
   * track, measured. `chunkPlan` in trunk.js is what guarantees it, and says so; this
   * throws rather than trusting the caller, because the symptom is a plausible structure.
   */
  transposeLnOut({ x, y, rows, nn, w, inF16 = false, outF16 = false, r0 = 0, eps = 1e-5 }) {
    if (r0 % 4 !== 0) throw new Error(`transposeLnOut: r0=${r0} is not a multiple of 4 - it reads x four pair rows a word`);
    const f16In = inF16 && this.hasF16;
    const f16Out = outF16 && this.hasF16;
    const groups = Math.ceil(rows / 16);
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u[0] = rows;
    u[1] = Math.ceil(nn / 4);
    u[2] = r0;
    f[3] = eps;
    return this.step(`tri-ln-out${f16In ? '-i16' : ''}${f16Out ? '-h' : ''}`, transposeLnOutSource(f16Out, f16In), groups, [x, y, w], u, `triLnOut ${rows}x256`);
  }

  /** Fused `z[a, j, :] = (zIn ?? z)[a, j, :] + pp[j, :384] * pp[a, 384:]` + `zc = LayerNorm_w(z)` (width=384). */
  pairWriteLnIn({ z, zIn = null, pp, zc = null, w = null, rows, n, outF16 = false, splitPp = false, r0 = 0, eps = 1e-5 }) {
    const f16Out = outF16 && this.hasF16;
    const separateZIn = Boolean(zIn && zIn !== z);
    const noLn = !zc;
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u[0] = rows;
    u[1] = n;
    u[2] = r0;
    f[3] = eps;
    return this.step(
      `pair-write-ln-in${f16Out ? '-h' : ''}${separateZIn ? '-zin' : ''}${splitPp ? '-sp' : ''}${noLn ? '-noln' : ''}`,
      pairWriteLnInSource(f16Out, separateZIn, splitPp, noLn),
      Math.ceil(rows / 8),
      [z, splitPp ? this.splitBuf : pp, ...(noLn ? [] : [zc, w]), ...(separateZIn ? [zIn] : [])],
      u,
      `pairWriteLnIn ${rows}x384`,
    );
  }

  /**
   * Fused Trunk readout heads (`ca_linear` `[n, 3]` + `local_latents_linear` `[n, 32]`).
   */
  headFast({ s, wLin, out, n, eps = 1e-5 }) {
    const half = wLin.half === true;
    const u = new Uint32Array(4);
    const f = new Float32Array(u.buffer);
    u[0] = n;
    f[1] = 768 * eps;
    return this.step(
      `head-fast${half ? '-h' : ''}`,
      headFastSource(half),
      Math.ceil(n / 16),
      [s, wLin, out],
      u,
      'headFast',
    );
  }

  /** Elementwise op; `bindings` = [{buffer, write?, type?}], `uints`/`floats` up to 16 each. */
  map(name, count, bindings, body, uints = [], floats = []) {
    const u = new Uint32Array(36);
    const f = new Float32Array(u.buffer);
    u[0] = count;
    u.set(uints, 4);
    f.set(floats, 20);
    const used = bindings.map((_, k) => new RegExp(`\\bb${k}\\b`).test(body));
    const source = mapSource(bindings.map((b, k) => (used[k] ? { type: b.type, write: b.write } : null)), body);
    return this.step(`map:${name}:${hashText(source)}`, source, Math.ceil(count / 256),
      bindings.map((b, k) => (used[k] ? b.buffer : null)), u, name);
  }

  /** Flatten steps/programs into one replayable program. */
  program(steps) {
    const flat = steps.flat(Infinity).flatMap((s) => (s.steps ? s.steps : [s]));
    return {
      steps: flat,
      run(pass) {
        for (const s of flat) {
          pass.setPipeline(s.pipeline);
          pass.setBindGroup(0, s.bindGroup);
          pass.dispatchWorkgroups(s.x, s.y);
        }
      },
    };
  }

  /**
   * One command buffer per program (`copy` appended to the last). macOS kills a command buffer that runs for
   * seconds ("Caused GPU Hang Error", device lost), and a whole trunk step in one buffer took 2.8 s (none) to 13.6 s
   * (high thinking, n=280); callers record programs small enough (trunk: a few layers each) and they still go to
   * the queue in one submit, in order.
   */
  encodePrograms(programs, copy = null) {
    const list = [programs].flat(Infinity);
    return list.map((p, k) => {
      const encoder = this.device.createCommandEncoder({ label: 'rfd4p' });
      const pass = encoder.beginComputePass({ label: 'rfd4p' });
      p.run(pass);
      pass.end();
      if (copy && k === list.length - 1) copy(encoder);
      return encoder.finish();
    });
  }

  /** Encode and submit programs. */
  run(...programs) {
    this.device.queue.submit(this.encodePrograms(programs));
  }

  /** Per-dispatch GPU timing (diagnostic): returns Map(label -> {ms, calls}). Needs 'timestamp-query'. */
  async profile(program, into = new Map()) {
    const steps = [program].flat(Infinity).flatMap((p) => p.steps);
    const query = this.device.createQuerySet({ type: 'timestamp', count: 2 * steps.length });
    const resolve = this.device.createBuffer({ size: 16 * steps.length, usage: 0x200 | COPY_SRC });
    const encoder = this.device.createCommandEncoder();
    steps.forEach((s, at) => {
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: 2 * at, endOfPassWriteIndex: 2 * at + 1 } });
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bindGroup);
      pass.dispatchWorkgroups(s.x, s.y);
      pass.end();
    });
    encoder.resolveQuerySet(query, 0, 2 * steps.length, resolve, 0);
    this.device.queue.submit([encoder.finish()]);
    const words = new BigUint64Array((await this.read(resolve, 4 * steps.length, 0, Uint32Array)).buffer);
    steps.forEach((s, at) => {
      const row = into.get(s.label) ?? { ms: 0, calls: 0 };
      row.ms += Number(words[2 * at + 1] - words[2 * at]) / 1e6;
      row.calls += 1;
      into.set(s.label, row);
    });
    query.destroy();
    resolve.destroy();
    return into;
  }

  async read(buffer, floats = buffer.floats, offsetFloats = 0, Type = Float32Array) {
    const size = Math.max(floats * 4, 4);
    let staging = this.staging;
    if (!staging || staging.size < size) {
      if (staging) staging.destroy();
      staging = this.device.createBuffer({ label: 'rfd4p.staging', size: Math.ceil(size / 4096) * 4096, usage: MAP_READ | COPY_DST });
      this.staging = staging;
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offsetFloats * 4, staging, 0, floats * 4);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_READ, 0, size);
    const out = new Type(staging.getMappedRange(0, size).slice(0, floats * 4));
    staging.unmap();
    return out;
  }

  /** Encode programs + staging copy in a single submit and return the readback array. */
  async runAndRead(programs, buffer, floats = buffer.floats, offsetFloats = 0, Type = Float32Array) {
    const size = Math.max(floats * 4, 4);
    let staging = this.staging;
    if (!staging || staging.size < size) {
      if (staging) staging.destroy();
      staging = this.device.createBuffer({ label: 'rfd4p.staging', size: Math.ceil(size / 4096) * 4096, usage: MAP_READ | COPY_DST });
      this.staging = staging;
    }
    const copy = (encoder) => encoder.copyBufferToBuffer(buffer, offsetFloats * 4, staging, 0, floats * 4);
    const buffers = this.encodePrograms(programs, copy);
    if (!buffers.length) {
      const encoder = this.device.createCommandEncoder({ label: 'rfd4p' });
      copy(encoder);
      buffers.push(encoder.finish());
    }
    this.device.queue.submit(buffers);
    await staging.mapAsync(MAP_READ, 0, size);
    const out = new Type(staging.getMappedRange(0, size).slice(0, floats * 4));
    staging.unmap();
    return out;
  }

  /** Free activations and uniforms (recorded programs become invalid); keep weights. */
  releaseActivations() {
    for (const b of this.activations) b.destroy();
    for (const b of this.uniforms) b.destroy();
    this.activations = [];
    this.uniforms = [];
    this.activationBytes = 0;
    this.splitBuf = null;
  }

  destroy() {
    this.releaseActivations();
    if (this.staging) {
      this.staging.destroy();
      this.staging = null;
    }
    for (const b of this.weights) b.destroy();
    this.weights = [];
    this.weightBytes = 0;
    this.dummy.destroy();
  }
}
