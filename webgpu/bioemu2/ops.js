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
  bias: u32, act: u32, accumulate: u32, alpha: f32, batches: u32 };
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
  for (var k0 = 0u; k0 < p.k; k0 += 16u) {
    for (var e = 0u; e < 4u; e += 1u) {
      let slot = li * 4u + e;          // 0..1023: row = slot / 16, k = slot % 16 (k fastest, for row-major operands)
      let kk = slot % 16u;
      let rr = slot / 16u;
      let k = k0 + kk;
      var va = 0.0;
      if (inBatch && k < p.k && m0 + rr < p.m) { va = a[baseA + (m0 + rr) * p.sAm + k * p.sAk]; }
      ta[kk * 64u + rr] = va;
      var vb = 0.0;
      if (inBatch && k < p.k && n0 + rr < p.n) {
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
  for (var h = 32u; h > 0u; h >>= 1u) { if (li < h) { red[li] += red[li + h]; } workgroupBarrier(); }
  let mean = red[0] / f32(p.width);
  workgroupBarrier();
  var q = 0.0;
  for (var c = li; c < p.width; c += 64u) { let d = x[xb + c] - mean; q += d * d; }
  red[li] = q;
  workgroupBarrier();
  for (var h = 32u; h > 0u; h >>= 1u) { if (li < h) { red[li] += red[li + h]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.width) + p.eps);
  if (!live) { return; }
  var crow = -1;
  if (p.modulate != 0u) { crow = select(i32(row), rowMap[row], p.useMap != 0u); }
  let yb = p.yOff + row * p.yStride;
  for (var c = li; c < p.width; c += 64u) {
    var v = (x[xb + c] - mean) * inv;
    if (p.hasW != 0u) { v *= w[c]; }
    if (p.hasB != 0u) { v += bv[c]; }
    if (p.modulate != 0u) {
      var al = 0.0;
      var be = 0.0;
      if (crow >= 0) {
        let cb = u32(crow) * p.condStride;
        al = cond[cb + p.alphaOff + c];
        be = cond[cb + p.betaOff + c];
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
  for (var h = 32u; h > 0u; h >>= 1u) { if (li < h) { red[li] = max(red[li], red[li + h]); } workgroupBarrier(); }
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
  for (var h = 32u; h > 0u; h >>= 1u) { if (li < h) { red[li] += red[li + h]; } workgroupBarrier(); }
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
    this.pipelines = pipelineCacheForDevice(device);
    this.compiled = new Map();
    this.dummy = device.createBuffer({ size: 16, usage: STORAGE });
    this.dummyInt = this.dummy;
    this.allocated = [];
    this.bytes = 0;
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

  /** A weight: float32, or float16 two to a word (`half`). */
  weight(label, values, half = false) {
    if (!half) return this.buffer(label, values.length, values instanceof Float32Array ? values : Float32Array.from(values));
    const words = Math.ceil(values.length / 2);
    const halves = new Uint16Array(words * 2);
    for (let i = 0; i < values.length; i += 1) halves[i] = toHalf(values[i]);
    const buffer = this.buffer(label, words);
    this.write(buffer, new Uint32Array(halves.buffer));
    buffer.half = true;
    return buffer;
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
    const entries = [{ binding: 0, resource: { buffer: this.uniform(uniformWords) } },
      ...buffers.map((buffer, at) => ({ binding: at + 1, resource: { buffer } })).filter((e) => e.resource.buffer !== null)];
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
    const u = new Uint32Array(24);
    const f = new Float32Array(u.buffer);
    u.set([m, n, k, B1, A.off, A.s0, A.s1, A.m, A.k, B.off, B.s0, B.s1, B.n, B.k, C.off, C.s0, C.s1, C.m, C.n,
      spec.bias ? 1 : 0, { relu: 1, sigmoid: 2 }[spec.act] ?? 0, spec.accumulate ? 1 : 0]);
    f[22] = spec.alpha ?? 1;
    u[23] = B0 * B1;
    const groups = Math.ceil(m / 64) * Math.ceil(n / 64) * B0 * B1;
    return this.step(word ? "gemm-h" : "gemm", gemmSource(word), groups,
                     [spec.a, spec.b, spec.c, spec.bias ?? this.dummy], u);
  }

  /** y = x . W^T (+ bias) for a torch (out, in) weight over `rows` rows. */
  linear({ x, w, y, rows, inC, outC, bias, act, accumulate, xOff = 0, xStride = inC, yOff = 0, yStride = outC, wOff = 0, alpha }) {
    return this.gemm({ m: rows, n: outC, k: inC, a: x, b: w, c: y, bias, act, accumulate, alpha,
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
    return {
      steps,
      run: (pass) => {
        for (const s of steps) {
          pass.setPipeline(s.pipeline);
          pass.setBindGroup(0, s.bindGroup);
          pass.dispatchWorkgroups(s.x, s.y);
        }
      },
    };
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
