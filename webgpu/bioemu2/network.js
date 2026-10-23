/**
 * BioEmu2 (microsoft/bioemu2, bioemu2-preview) on the GPU.
 *
 * A transcription of af3-any-model's `network/bioemu2.py`, stage for stage, held
 * to it by tools/gpu/check-bioemu2-network.js on tools/oracle/dump_bioemu2.py's
 * dumps:
 *
 *   * the INPUT EMBEDDER - three conditioned windowed atom-transformer layers,
 *     then a token mean - gives s_inputs (T, 455);
 *   * the TRUNK is a PairMixer: four MSA layers (pair-weighted averaging, the
 *     MSA transition, an outer product mean whose mean is over VALID ROWS, then
 *     a pair-only block) and 48 pair-only blocks, run four times with the
 *     recycle. Its pair-only block - tri_mul_out, tri_mul_in, a transition - is
 *     ESMFold2's, so it runs on webgpu/esmfold2/trunk.js and AF3's pair-track
 *     kernels (the bundle carries it in their layout);
 *   * the DENOISER is a dual stream: 16 layers each running a windowed atom
 *     transformer and a full token transformer side by side, then mixing them
 *     through every token's centre atom.
 *
 * Everything static in a fold - the pair conditioning, the per-layer attention
 * biases (16 x 16 heads x T^2 for the token stream, 16 x 4 x windows for the
 * atoms), the single conditioning's coordinate-free half - is computed once in
 * `prepare`; a denoiser call is one recorded program replayed per sampler step.
 */
import { Bioemu2Ops } from "./ops.js";
import {
  addTokenPairToWindows, atomFeatures, atomPairTail, dampedHarmonic, linearRows, lnRows, normalizeWindowPair, sInputs, windowBias,
} from "./host.js";
import { BIOEMU2 } from "../../shared/bioemu2/weights.js";
import { trunkBlockWeights } from "../../shared/esmfold2/weights.js";
import { Esmfold2TrunkGpu } from "../esmfold2/trunk.js";
import { GpuBufferAllocator } from "../runtime/allocator.js";

const W_Q = BIOEMU2.windowQueries;
const W_K = BIOEMU2.windowKeys;
const CTX = (W_K - W_Q) / 2;
const SIGMA_DATA = BIOEMU2.sigmaData;

export class Bioemu2Gpu {
  /**
   * @param read a shared/bioemu2/weights.js reader
   * @param options `half`: weights as float16 on the device (default true); `trunkPrecision`: "f32" pins the
   *   pair track's staging and accumulation to float32 (the oracle arm), otherwise the device's choice
   */
  constructor(device, read, options = {}) {
    this.device = device;
    this.read = read;
    this.options = options;
    this.half = options.half ?? true;
    this.ops = new Bioemu2Ops(device);
    this.weights = new Map();
    this.host = new Map();
  }

  /** A weight on the device, once. */
  async w(name, { half = this.half, transform } = {}) {
    if (!transform) return this.stacked(`${name}|${half}|`, [name], half);
    const key = `${name}|${half}|t`;
    if (!this.weights.has(key)) {
      this.weights.set(key, (async () => {
        const values = transform(await this.read(name));
        return this.ops.weight(name, values, half && values.length % 2 === 0);
      })());
    }
    return this.weights.get(key);
  }

  /** A weight on the host, once. */
  async h(name) {
    if (!this.host.has(name)) this.host.set(name, this.read(name));
    return this.host.get(name);
  }

  async hostGroup(prefix, names) {
    const out = {};
    await Promise.all(names.map(async (n) => { out[n] = await this.h(`${prefix}/${n}`); }));
    return out;
  }

  /**
   * `names`' tensors concatenated into ONE weight buffer, in order (each [fc1; fc2] pair when `swiglu`): a stack of the
   * denoiser's layers, so their conditioning is one batched GEMM (conditionGroup). The per-layer tensors are then
   * never uploaded on their own.
   */
  async stacked(key, names, half, swiglu = false) {
    if (!this.weights.has(key)) {
      this.weights.set(key, (async () => {
        const flat = names.flatMap((n) => (swiglu ? [`${n}/fc1/weight`, `${n}/fc2/weight`] : [n]));
        if (typeof this.read.source === "function" && typeof this.read.decode === "function") {
          const sources = await Promise.all(flat.map((n) => this.read.source(n)));
          return this.ops.weightFromSources(key, sources, half, this.read.decode);
        }
        const parts = await Promise.all(flat.map((n) => this.read(n)));
        const all = new Float32Array(parts.reduce((t, v) => t + v.length, 0));
        let at = 0;
        for (const v of parts) { all.set(v, at); at += v.length; }
        return this.ops.weight(key, all, half && all.length % 2 === 0);
      })());
    }
    return this.weights.get(key);
  }

  /** [fc1; fc2] stacked, so a SwiGLU's two projections are one GEMM. */
  async swiglu(prefix) {
    return this.stacked(`${prefix}/fc12`, [prefix], this.half, true);
  }

  /** [proj_qkv; proj_g] stacked, so an attention layer's QKV and gate projections are one GEMM. */
  async qkvg(att) {
    return this.stacked(`${att}/proj_qkvg`, [`${att}/proj_qkv/weight`, `${att}/proj_g/weight`], this.half);
  }

  // --- reusable recorded blocks ---------------------------------------------------------------------------

  /**
   * The coupling's ATOM-side mixer, split - the steps that write its SwiGLU product `out` [S N][I].
   *
   * s_to_q reads y = LN([a_token | q_atom]) (1 + alpha) + beta, one 896-wide row an atom of every sample: 2880 x 896 ->
   * 3584 at 68 tokens x 5 samples, 44% of a denoiser call here (tools/gpu/profile-bioemu2.js). But exactly
   *   W y = (sa P[token] + sq Z[atom] + da Qa[tok] + dq Qq[tok]) / sigma + B[tok]
   *   P = W_a ((1 + alpha_a) (a - mean a) / sa)   once a TOKEN row (W_a: W's first 768 columns)
   *   Z = W_q ((1 + alpha_q) (q - mean q) / sq)   the atom's own 128 columns
   *   Qa = W_a (1 + alpha_a), Qq = W_q (1 + alpha_q), B = W beta   once a token, sample-free
   * (and since 768 da + 128 dq = 0, da Qa + dq Qq = dq R, R = Qq - Qa / 6: what is stored is R | B)
   * the row's statistics combined from its two halves' (da = mean a - mu, dq = mean q - mu, var = (SS_a + SS_q +
   * 768 da^2 + 128 dq^2) / 896 - the parallel-variance form, never E[x^2] - mu^2), and sa, sq each half's own spread,
   * which any positive scale would do: they keep the GEMMs' inputs O(1) as a LayerNorm's are. 9.25 -> 2.7 G
   * multiply-adds. The inputs stay float32 (this port's GEMM unpacks the half weight and never rounds its input - on
   * metal/bioemu2, which rounds it, Xq in half cost the split its precision). A padding atom (map -1) conditions on
   * the identity, gamma 1 and beta 0, as this port's adaLN gives it.
   */
  async couplingSplit(steps, prefix, { a, q, condQ, atomTokenS, S, T, N, cs, out, pre = null, preTok = false }) {
    const o = this.ops;
    const DT = BIOEMU2.token, DA = BIOEMU2.atom, DC = DT + DA, I = 1792, W2 = 2 * I;
    // (pre: conditionGroup's - the layer's [fc1; fc2] a slice of the stacked one at wOff, and gamma and R | B
    // already computed for the group, this layer's views of them)
    const W = pre ? pre.W : await this.swiglu(prefix);     // [fc1; fc2], [W2][DC]
    const wo = pre ? pre.wOff : 0;
    const gA = pre ? pre.gA : cs.gA, gQ = pre ? pre.gQ : cs.gQ, QB = pre ? pre.QB : cs.QB;
    if (!pre) {
    // gamma_a, gamma_q and beta from the T conditioning rows, and row T the padding's identity
    steps.push(await o.map("cs-gamma", (T + 1) * DC, [{ buffer: cs.gA, write: true }, { buffer: cs.gQ, write: true },
      { buffer: cs.beta, write: true }, { buffer: condQ }],
    `let t = i / 896u; let c = i % 896u; let real = t < U(0u);
  var al = 0.0; var be = 0.0;
  if (real) { al = b3[t * 1920u + c]; be = b3[t * 1920u + 896u + c]; }
  if (c < 768u) { b0[t * 768u + c] = 1.0 + al; } else { b1[t * 128u + c - 768u] = 1.0 + al; }
  b2[i] = be;`, [T]));
    // R | B, one buffer [T + 1][2][W2] (the combine reads them through one binding): R = Qq - Qa / 6, since 768 da +
    // 128 dq = 0 makes da Qa + dq Qq = dq (Qq - Qa / 6) - Qa scaled by -1/6, then Qq added
    steps.push(await o.gemm({ m: T + 1, n: W2, k: DT, a: cs.gA, b: W, c: cs.QB, alpha: -DA / DT,
      A: { off: 0, m: DT, k: 1 }, B: { off: 0, n: DC, k: 1 }, C: { off: 0, m: 2 * W2, n: 1 } }));
    steps.push(await o.gemm({ m: T + 1, n: W2, k: DA, a: cs.gQ, b: W, c: cs.QB, accumulate: true,
      A: { off: 0, m: DA, k: 1 }, B: { off: DT, n: DC, k: 1 }, C: { off: 0, m: 2 * W2, n: 1 } }));
    steps.push(await o.gemm({ m: T + 1, n: W2, k: DC, a: cs.beta, b: W, c: cs.QB,
      A: { off: 0, m: DC, k: 1 }, B: { off: 0, n: DC, k: 1 }, C: { off: W2, m: 2 * W2, n: 1 } }));
    }
    // a token row (sample-stacked): its mean, centred sum of squares and spread; then Xa and P
    if (!preTok) steps.push(await o.csTokenPrep({ a, gA, tstats: cs.tstats, xa: cs.xa, rows: S * T, T }));
    steps.push(await o.gemm({ m: S * T, n: W2, k: DT, a: cs.xa, b: W, c: cs.P,
      A: { off: 0, m: DT, k: 1 }, B: { off: wo, n: DC, k: 1 }, C: { off: 0, m: W2, n: 1 } }));
    // an atom row: q's statistics, the whole row's combined (a padding atom's a is zero), its token's spread; Xq; Z
    steps.push(await o.csAtomPrep({ q, gQ, tstats: cs.tstats, atomTokenS, astats: cs.astats, xq: cs.xq, rows: S * N, T }));
    if (this.options.fuseCombine !== false) {
      // Z never stored: the GEMM's columns reordered so a lane holds four gates and their four values, and the combine
      // and the SwiGLU run in its epilogue (fuseCombine: false is the control - Z written, then cs-combine)
      steps.push(await o.gemm({ m: S * N, n: W2, k: DA, a: cs.xq, b: W, c: out,
        A: { off: 0, m: DA, k: 1 }, B: { off: wo + DT, n: DC, k: 1 }, C: { off: 0, m: I, n: 1 },
        pair: { key: `cs${T}`, half: I, rows4: true, buffers: [cs.astats, cs.P, QB, atomTokenS],
          decls: `@group(0) @binding(5) var<storage, read> astats: array<f32>;
@group(0) @binding(6) var<storage, read> pTok: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> atomTok: array<i32>;`,
          body: `fn pairOut4(r0: u32, j0: u32, z1_0: vec4<f32>, z2_0: vec4<f32>, z1_1: vec4<f32>, z2_1: vec4<f32>, z1_2: vec4<f32>, z2_2: vec4<f32>, z1_3: vec4<f32>, z2_3: vec4<f32>) {
  let j4 = j0 / 4u;
  var prev_st = -2;
  var p1 = vec4<f32>(0.0); var p2 = vec4<f32>(0.0);
  var q1 = vec4<f32>(0.0); var q2 = vec4<f32>(0.0);
  var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0);
  ${[0, 1, 2, 3].map((k) => `{
    let r = r0 + ${k}u;
    if (r < p.m) {
      let st = atomTok[r];
      let s = r * 6u;
      let inv = astats[s]; let dq = astats[s + 2u]; let sq = astats[s + 3u];
      if (st != prev_st) {
        prev_st = st;
        var t = ${T}u;
        if (st >= 0) {
          t = u32(st) % ${T}u;
          let sa = astats[s + 5u];
          let pr4 = u32(st) * 896u + j4;
          p1 = sa * pTok[pr4];
          p2 = sa * pTok[pr4 + 448u];
        } else {
          p1 = vec4<f32>(0.0);
          p2 = vec4<f32>(0.0);
        }
        let cr4 = t * 1792u + j4;
        q1 = qb[cr4];
        q2 = qb[cr4 + 448u];
        b1 = qb[cr4 + 896u];
        b2 = qb[cr4 + 1344u];
      }
      let u1 = inv * (p1 + sq * z1_${k} + dq * q1) + b1;
      let u2 = inv * (p2 + sq * z2_${k} + dq * q2) + b2;
      let y = u1 / (vec4<f32>(1.0) + exp(-u1)) * u2;
      let base = r * 1792u + j0;
      c[base] = y.x; c[base + 1u] = y.y; c[base + 2u] = y.z; c[base + 3u] = y.w;
    }
  }`).join("\n  ")}
}` } }));
      return;
    }
    steps.push(await o.gemm({ m: S * N, n: W2, k: DA, a: cs.xq, b: W, c: cs.Z,
      A: { off: 0, m: DA, k: 1 }, B: { off: wo + DT, n: DC, k: 1 }, C: { off: 0, m: W2, n: 1 } }));
    // each atom's two pre-activations, and the SwiGLU of them
    steps.push(await o.map("cs-combine", S * N * I, [{ buffer: out, write: true }, { buffer: cs.astats }, { buffer: cs.P },
      { buffer: cs.Z }, { buffer: QB }, { buffer: atomTokenS, type: "i32" }],
    `let r = i / 1792u; let j = i % 1792u; let st = b5[r]; var t = U(0u);
  if (st >= 0) { t = u32(st) % U(0u); }
  let s = r * 6u; let inv = b1[s]; let dq = b1[s + 2u]; let sq = b1[s + 3u]; let sa = b1[s + 5u];
  var p1 = 0.0; var p2 = 0.0;
  if (st >= 0) { let pr = u32(st) * 3584u; p1 = sa * b2[pr + j]; p2 = sa * b2[pr + 1792u + j]; }
  let zr = r * 3584u; let cr = t * 7168u;
  let u1 = inv * (p1 + sq * b3[zr + j] + dq * b4[cr + j]) + b4[cr + 3584u + j];
  let u2 = inv * (p2 + sq * b3[zr + 1792u + j] + dq * b4[cr + 1792u + j]) + b4[cr + 3584u + 1792u + j];
  b0[i] = u1 / (1.0 + exp(-u1)) * u2;`, [T]));
  }

  /**
   * SwiGLU transition over `rows` x `width` (hidden `hidden`): x [+]= fc3(silu(fc1 h) * fc2 h), where h is
   * `input` (already normalised/modulated when `norm` is false) - returned steps write `out`.
   */
  async transition(steps, prefix, { input, rows, width, hidden, out, scratch, norm, accumulate = false, gate }) {
    const o = this.ops;
    let src = input;
    if (norm) {
      steps.push(await o.layerNorm({ x: input, y: scratch.ln, rows, width,
        w: await this.w(`${prefix}/norm/weight`, { half: false }), b: await this.w(`${prefix}/norm/bias`, { half: false }) }));
      src = scratch.ln;
    }
    const fused = await o.swigluLinear({ x: src, w: await this.swiglu(prefix), y: scratch.t, rows, inC: width, hidden });
    if (fused) steps.push(fused);
    else {
      steps.push(await o.linear({ x: src, w: await this.swiglu(prefix), y: scratch.t12, rows, inC: width, outC: 2 * hidden }));
      steps.push(await o.map("swiglu", rows * hidden, [{ buffer: scratch.t12 }, { buffer: scratch.t, write: true }],
        `let r = i / U(0u); let c = i % U(0u);
  let g = b0[r * 2u * U(0u) + c]; let v = b0[r * 2u * U(0u) + U(0u) + c];
  b1[i] = g / (1.0 + exp(-g)) * v;`, [hidden]));
    }
    // (gate: out += cond[map(r)][off..] fc3(...), the adaLN-gated residual in fc3's epilogue)
    steps.push(await o.linear({ x: scratch.t, w: await this.w(`${prefix}/fc3/weight`), y: out, rows, inC: hidden, outC: width, accumulate, gate }));
  }

  /** silu(LN(c) w + b) . cond_2 + bias, per row of the conditioning source. */
  async condition(steps, prefix, { c, rows, width, outC, ln, out, l0 = "cond_0", l2 = "cond_2" }) {
    const o = this.ops;
    steps.push(await o.layerNorm({ x: c, y: ln, rows, width,
      w: await this.w(`${prefix}/${l0}/weight`, { half: false }), b: await this.w(`${prefix}/${l0}/bias`, { half: false }) }));
    steps.push(await o.map("silu", rows * width, [{ buffer: ln, write: true }], "let v = b0[i]; b0[i] = v / (1.0 + exp(-v));"));
    steps.push(await o.linear({ x: ln, w: await this.w(`${prefix}/${l2}/weight`), y: out, rows, inC: width, outC,
                                bias: await this.w(`${prefix}/${l2}/bias`, { half: false }) }));
  }

  /**
   * The conditioning of denoiser layers l0 .. l0 + n - 1, as BATCHED GEMMs over the layers. Every layer conditions on
   * the SAME rows (`cond`, the call's), so each of its four adaLN MLPs - silu(LN(c) w + b) . cond_2 + bias, LN's
   * statistics shared and only its affine a layer's - is one GEMM with the layers as the batch, over a stack of their
   * weights (`stacked`), where it was 16 GEMMs of 68 rows. The coupling split's gamma and R | B go the same way.
   * `xhat` is LN(cond) without its affine, once a call. Returns, per layer, views of the group's buffers.
   */
  async conditionGroup(steps, { xhat, T, l0, n, G, P, buffers }) {
    const o = this.ops;
    const DT = BIOEMU2.token, DA = BIOEMU2.atom, DC = DT + DA, W2 = 3584, L = BIOEMU2.denoiserLayers;
    const layer = (l) => `${P}/layers/${l}`;
    const all = [...Array(L).keys()];
    const kinds = [
      { key: "atom", outC: 6 * DA, prefix: (l) => `${layer(l)}/atom_block/transformer_layer`, l0: "cond_0", l2: "cond_2" },
      { key: "tok", outC: 6 * DT, prefix: (l) => `${layer(l)}/token_block`, l0: "cond_0", l2: "cond_2" },
      { key: "q", outC: 1920, prefix: (l) => `${layer(l)}/coupling`, l0: "q_coupling_cond_0", l2: "q_coupling_cond_2" },
      { key: "s", outC: 2560, prefix: (l) => `${layer(l)}/coupling`, l0: "s_coupling_cond_0", l2: "s_coupling_cond_2" },
    ];
    const lwb = await this.stacked("stack/cond0/wb", kinds.flatMap((k) => [
      ...all.map((l) => `${k.prefix(l)}/${k.l0}/weight`),
      ...all.map((l) => `${k.prefix(l)}/${k.l0}/bias`),
    ]), false);
    const plane = G * T * DT;
    const wbPlane = 2 * L * DT;
    const wPlane = L * DT;
    // (LN's affine as the LayerNorm kernel applies it, (x - mean) inv, times w, plus b: then silu for all 4 kinds in one pass)
    const plane4 = plane / 4, wbPlane4 = wbPlane / 4, wPlane4 = wPlane / 4;
    steps.push(await o.map("cond-affine", (n * T * DT) / 4, [
      { buffer: buffers.cin, type: "vec4<f32>", write: true },
      { buffer: xhat, type: "vec4<f32>" },
      { buffer: lwb, type: "vec4<f32>" },
    ],
    `let c4 = i % 192u; let r = i / 192u; let t = r % U(0u); let l = U(1u) + r / U(0u);
  let x = b1[t * 192u + c4];
  let wIdx = l * 192u + c4;
  let v0 = x * b2[wIdx] + b2[${wPlane4}u + wIdx];
  let v1 = x * b2[${wbPlane4}u + wIdx] + b2[${wbPlane4 + wPlane4}u + wIdx];
  let v2 = x * b2[${2 * wbPlane4}u + wIdx] + b2[${2 * wbPlane4 + wPlane4}u + wIdx];
  let v3 = x * b2[${3 * wbPlane4}u + wIdx] + b2[${3 * wbPlane4 + wPlane4}u + wIdx];
  b0[i] = v0 / (vec4<f32>(1.0) + exp(-v0));
  b0[${plane4}u + i] = v1 / (vec4<f32>(1.0) + exp(-v1));
  b0[${2 * plane4}u + i] = v2 / (vec4<f32>(1.0) + exp(-v2));
  b0[${3 * plane4}u + i] = v3 / (vec4<f32>(1.0) + exp(-v3));`, [T, l0]));
    for (let kIdx = 0; kIdx < kinds.length; kIdx += 1) {
      const k = kinds[kIdx];
      const w2 = await this.stacked(`stack/${k.key}/${k.l2}/weight`, all.map((l) => `${k.prefix(l)}/${k.l2}/weight`), this.half);
      const b2 = await this.stacked(`stack/${k.key}/${k.l2}/bias`, all.map((l) => `${k.prefix(l)}/${k.l2}/bias`), false);
      steps.push(await o.gemm({ m: T, n: k.outC, k: DT, a: buffers.cin, b: w2, c: buffers[k.key], bias: b2,
        batches: [n, 1], biasOff: l0 * k.outC, biasS0: k.outC,
        A: { off: kIdx * plane, s0: T * DT, m: DT, k: 1 }, B: { off: l0 * k.outC * DT, s0: k.outC * DT, n: DT, k: 1 },
        C: { off: 0, s0: T * k.outC, m: k.outC, n: 1 }, label: `cond ${k.key} ${T}x${k.outC}x${DT} [${n}]` }));
    }
    const out = [];
    for (let j = 0; j < n; j += 1) {
      out.push({ atom: o.view(buffers.atom, j * T * 6 * DA, T * 6 * DA), tok: o.view(buffers.tok, j * T * 6 * DT, T * 6 * DT),
                 q: o.view(buffers.q, j * T * 1920, T * 1920), s: o.view(buffers.s, j * T * 2560, T * 2560) });
    }
    if (!buffers.gA) return out;
    // the coupling split's: gamma_a, gamma_q and beta a layer (row T the padding's identity), then R | B
    const R = T + 1;
    steps.push(await o.map("cs-gamma-g", (n * R * DC) / 4, [
      { buffer: buffers.gA, type: "vec4<f32>", write: true },
      { buffer: buffers.gQ, type: "vec4<f32>", write: true },
      { buffer: buffers.q, type: "vec4<f32>" },
      { buffer: buffers.gAQ, type: "vec4<f32>", write: true },
    ],
    `let j = i / U(1u); let e4 = i % U(1u); let t = e4 / 224u; let c4 = e4 % 224u; let real = t < U(0u);
  var al = vec4<f32>(0.0); var be = vec4<f32>(0.0);
  if (real) { let at4 = (j * U(0u) + t) * 480u; al = b2[at4 + c4]; be = b2[at4 + 224u + c4]; }
  let g = vec4<f32>(1.0) + al;
  let rb4 = (j * U(2u) + t) * 448u;
  if (c4 < 192u) {
    b0[(j * U(2u) + t) * 192u + c4] = g;
    b3[rb4 + c4] = g * (-1.0 / 6.0);
  } else {
    b1[(j * U(2u) + t) * 32u + c4 - 192u] = g;
    b3[rb4 + c4] = g;
  }
  b3[rb4 + 224u + c4] = be;`, [T, R * (DC / 4), R]));
    const W = await this.stacked("stack/s_to_q_mixer/fc12", all.map((l) => `${layer(l)}/coupling/s_to_q_mixer`), this.half, true);
    const wl = W2 * DC;
    // (R | B: R = Qq - Qa / 6 - couplingSplit's, stacked with beta into a single [2R][DC] GEMM)
    steps.push(await o.gemm({ m: 2 * R, n: W2, k: DC, a: buffers.gAQ, b: W, c: buffers.QB, batches: [n, 1],
      A: { off: 0, s0: 2 * R * DC, m: DC, k: 1 }, B: { off: l0 * wl, s0: wl, n: DC, k: 1 },
      C: { off: 0, s0: 2 * R * W2, m: W2, n: 1 } }));
    for (let j = 0; j < n; j += 1) {
      out[j].pre = { W, wOff: (l0 + j) * wl, gA: o.view(buffers.gA, j * R * DT, R * DT), gQ: o.view(buffers.gQ, j * R * DA, R * DA),
                     QB: o.view(buffers.QB, j * R * 2 * W2, R * 2 * W2) };
    }
    return out;
  }

  /**
   * ConditionedTransformerLayer over a WINDOWED atom stream: x (S * N, 128) - S samples, sample-major - cond rows
   * (R, 768) with `map` (each of the S * N rows -> its cond row, -1 for none) or the identity, the layer's window
   * bias (NW, 4, 32, 128) and the padded key mask, both shared by the samples.
   */
  async windowedLayer(steps, prefix, { x, cond, map, bias, keyMask, N, s, S = 1 }) {
    const o = this.ops;
    const D = BIOEMU2.atom;
    const H = BIOEMU2.atomHeads;
    const hd = D / H;
    const NW = N / W_Q;
    const R = S * N;
    const pad = N + 2 * CTX;
    const att = `${prefix}/attention/attention_layer/pair_bias_attn`;
    const mod = (alpha, beta) => ({ cond, stride: 6 * D, alpha, beta, map });
    steps.push(await o.layerNorm({ x, y: s.h, rows: R, width: D, mod: mod(0, D) }));
    const qBias = await this.w(`${att}/proj_q_bias`, { half: false });
    if (this.options.fuseWindow !== false && hd === 32 && W_Q === 32 && W_K === 128) {
      steps.push(await o.linear({ x: s.h, w: await this.qkvg(att), y: s.qkv, rows: R, inC: D, outC: 4 * D }));
      // (the three passes, the kv window gather, and the gating in one kernel: ops.windowAttention; fuseWindow: false is the control)
      steps.push(await o.windowAttention({ qkv: s.qkv, qBias, kvPad: null, bias, mask: keyMask, g: null, out: s.o, S, N, H, pad,
                                           alpha: 1 / Math.sqrt(hd), ctx: CTX }));
    } else {
      steps.push(await o.linear({ x: s.h, w: await this.w(`${att}/proj_qkv/weight`), y: s.qkv, rows: R, inC: D, outC: 3 * D }));
      steps.push(await o.linear({ x: s.h, w: await this.w(`${att}/proj_g/weight`), y: s.g, rows: R, inC: D, outC: D, act: "sigmoid" }));
      // q (+ bias) in place, and k, v into each sample's padded key array (48 zero rows each end)
      steps.push(await o.map("win-qkv", R * 3 * D, [{ buffer: s.qkv, write: true }, { buffer: qBias },
        { buffer: s.kvPad, write: true }],
      `let r = i / 384u; let c = i % 384u; let sm = r / U(0u); let ra = r % U(0u);
  if (c < 128u) { b0[i] = b0[i] + b1[c]; } else { b2[(sm * U(1u) + ra + ${CTX}u) * 256u + c - 128u] = b0[i]; }`, [N, pad]));
      // per sample: logits (NW, H, 32, 128) = q . k^T / sqrt(hd), the softmax, o = P . v
      for (let sm = 0; sm < S; sm += 1) {
        steps.push(await o.gemm({ m: W_Q, n: W_K, k: hd, a: s.qkv, b: s.kvPad, c: s.logits, batches: [NW, H], alpha: 1 / Math.sqrt(hd),
          A: { off: sm * N * 3 * D, s0: W_Q * 3 * D, s1: hd, m: 3 * D, k: 1 },
          B: { off: sm * pad * 2 * D, s0: W_Q * 2 * D, s1: hd, n: 2 * D, k: 1 },
          C: { off: 0, s0: H * W_Q * W_K, s1: W_Q * W_K, m: W_K, n: 1 } }));
        steps.push(await o.softmax({ l: s.logits, rows: NW * H * W_Q, len: W_K, rowsPer: W_Q, b1: H,
          L: { s0: H * W_Q * W_K, s1: W_Q * W_K, r: W_K }, bias, Bias: { s0: H * W_Q * W_K, s1: W_Q * W_K, r: W_K },
          mask: keyMask, M: { off: 0, s0: W_Q } }));
        steps.push(await o.gemm({ m: W_Q, n: hd, k: W_K, a: s.logits, b: s.kvPad, c: s.o, batches: [NW, H],
          A: { off: 0, s0: H * W_Q * W_K, s1: W_Q * W_K, m: W_K, k: 1 },
          B: { off: sm * pad * 2 * D + D, s0: W_Q * 2 * D, s1: hd, n: 1, k: 2 * D },
          C: { off: sm * N * D, s0: W_Q * D, s1: hd, m: D, n: 1 } }));
      }
      steps.push(await o.map("mul", R * D, [{ buffer: s.o, write: true }, { buffer: s.g }], "b0[i] = b0[i] * b1[i];"));
    }
    // (x += gamma_a proj_o(...) and x += gamma_m transition(...) in their GEMMs' epilogues - ops.gemm's gate)
    steps.push(await o.linear({ x: s.o, w: await this.w(`${att}/proj_o/weight`), y: x, rows: R, inC: D, outC: D,
                                gate: { cond, stride: 6 * D, off: 2 * D, map } }));
    steps.push(await o.layerNorm({ x, y: s.h, rows: R, width: D, mod: mod(3 * D, 4 * D) }));
    await this.transition(steps, `${prefix}/transition`, { input: s.h, rows: R, width: D, hidden: 4 * D, out: x,
                                                         scratch: s, norm: false, gate: { cond, stride: 6 * D, off: 5 * D, map } });
  }

  /**
   * ConditionedTransformerLayer over the FULL token stream: x (S * T, 768), 16 heads, the pair bias (16, T, T)
   * precomputed and shared by the samples; `map` takes each of the S * T rows to its (shared) cond row.
   */
  async tokenLayer(steps, prefix, { x, cond, map = null, bias, T, s, S = 1 }) {
    const o = this.ops;
    const D = BIOEMU2.token;
    const H = BIOEMU2.tokenHeads;
    const hd = D / H;
    const R = S * T;
    const att = `${prefix}/attention/pair_bias_attn`;
    const mod = (alpha, beta) => ({ cond, stride: 6 * D, alpha, beta, map });
    steps.push(await o.layerNorm({ x, y: s.h, rows: R, width: D, mod: mod(0, D) }));
    const qBias = await this.w(`${att}/proj_q_bias`, { half: false });
    if (this.options.fuseTokenAttention !== false && hd % 16 === 0) {
      steps.push(await o.linear({ x: s.h, w: await this.qkvg(att), y: s.qkv, rows: R, inC: D, outC: 4 * D }));
      // (scores, softmax, P.V and the gating in one kernel: ops.tokenAttention; fuseTokenAttention: false the control)
      steps.push(await o.tokenAttention({ qkv: s.qkv, qBias, bias, g: null, out: s.o, S, T, H, hd, alpha: 1 / Math.sqrt(hd) }));
    } else {
      steps.push(await o.linear({ x: s.h, w: await this.w(`${att}/proj_qkv/weight`), y: s.qkv, rows: R, inC: D, outC: 3 * D }));
      steps.push(await o.linear({ x: s.h, w: await this.w(`${att}/proj_g/weight`), y: s.g, rows: R, inC: D, outC: D, act: "sigmoid" }));
      steps.push(await o.map("tok-qbias", R * D, [{ buffer: s.qkv, write: true }, { buffer: qBias }],
        "let r = i / 768u; let c = i % 768u; b0[r * 2304u + c] = b0[r * 2304u + c] + b1[c];"));
      steps.push(await o.gemm({ m: T, n: T, k: hd, a: s.qkv, b: s.qkv, c: s.logits, batches: [S, H], alpha: 1 / Math.sqrt(hd),
        A: { off: 0, s0: T * 3 * D, s1: hd, m: 3 * D, k: 1 }, B: { off: D, s0: T * 3 * D, s1: hd, n: 3 * D, k: 1 },
        C: { off: 0, s0: H * T * T, s1: T * T, m: T, n: 1 } }));
      steps.push(await o.softmax({ l: s.logits, rows: S * H * T, len: T, rowsPer: T, b1: H,
        L: { s0: H * T * T, s1: T * T, r: T }, bias, Bias: { s0: 0, s1: T * T, r: T } }));
      steps.push(await o.gemm({ m: T, n: hd, k: T, a: s.logits, b: s.qkv, c: s.o, batches: [S, H],
        A: { off: 0, s0: H * T * T, s1: T * T, m: T, k: 1 }, B: { off: 2 * D, s0: T * 3 * D, s1: hd, n: 1, k: 3 * D },
        C: { off: 0, s0: T * D, s1: hd, m: D, n: 1 } }));
      steps.push(await o.map("mul", R * D, [{ buffer: s.o, write: true }, { buffer: s.g }], "b0[i] = b0[i] * b1[i];"));
    }
    // (x += gamma_a proj_o(...) and x += gamma_m transition(...) in their GEMMs' epilogues - ops.gemm's gate)
    steps.push(await o.linear({ x: s.o, w: await this.w(`${att}/proj_o/weight`), y: x, rows: R, inC: D, outC: D,
                                gate: { cond, stride: 6 * D, off: 2 * D, map } }));
    steps.push(await o.layerNorm({ x, y: s.h, rows: R, width: D, mod: mod(3 * D, 4 * D) }));
    await this.transition(steps, `${prefix}/transition`, { input: s.h, rows: R, width: D, hidden: 4 * D, out: x,
                                                         scratch: s, norm: false, gate: { cond, stride: 6 * D, off: 5 * D, map } });
  }

  /** Scratch for a stream of `rows` x `width`. */
  scratch(name, rows, width, extra = {}) {
    const o = this.ops;
    return {
      h: o.buffer(`${name}.h`, rows * width), qkv: o.buffer(`${name}.qkv`, rows * 4 * width),
      g: o.buffer(`${name}.g`, rows * width), o: o.buffer(`${name}.o`, rows * width),
      t3: o.buffer(`${name}.t3`, rows * width), t12: o.buffer(`${name}.t12`, rows * 8 * width),
      t: o.buffer(`${name}.t`, rows * 4 * width), ln: o.buffer(`${name}.ln`, rows * width), ...extra,
    };
  }

  // --- the stages -------------------------------------------------------------------------------------------

  /** Device buffers of a feature dict, and its shapes. */
  upload(f) {
    const o = this.ops;
    const T = f.res_type.length;
    const N = f.atom_pad_mask.length;
    const keyMask = new Int32Array(N + 2 * CTX);
    keyMask.set(f.atom_pad_mask, CTX);
    const atomToken = Int32Array.from(f.token_idx_at_atom, (t, a) => (f.atom_pad_mask[a] ? t : -1));
    return {
      f, T, N, NW: N / W_Q,
      keyMask: o.buffer("keyMask", keyMask.length, keyMask),
      atomToken: o.buffer("atomToken", N, atomToken),
    };
  }

  /** InputEmbedder -> s_inputs (T, 455), on the host but for the three atom layers. */
  async inputEmbedder(g) {
    const { f, T, N, NW } = g;
    const o = this.ops;
    const P = "input_embedder/atom_attention_encoder";
    const host = await this.hostGroup(P, ["proj_atom_continuous/weight", "embed_element/weight",
      ...[0, 1, 2, 3].map((i) => `embed_atom_name_chars_${i}/weight`), "embed_atompair_ref_pos/weight",
      "embed_atompair_ref_dist/weight", "embed_atompair_mask/weight", "c_to_p_trans_q_1/weight",
      "c_to_p_trans_k_1/weight", "p_mlp_1/weight", "p_mlp_3/weight", "p_mlp_5/weight", "atom_to_token_trans_0/weight"]);
    const { c, pr } = atomFeatures(host, f);
    atomPairTail(host, c, pr, N);
    const D = BIOEMU2.atom;
    const cBuf = o.buffer("ie.c", N * D, c);
    const q = o.buffer("ie.q", N * D, c);
    const s = this.scratch("ie", N, D, {
      kvPad: o.buffer("ie.kvPad", (N + 2 * CTX) * 2 * D), logits: o.buffer("ie.logits", NW * BIOEMU2.atomHeads * W_Q * W_K),
    });
    const cond = o.buffer("ie.cond", N * 6 * D);
    const steps = [];
    const prNorm = normalizeWindowPair(pr);
    for (let l = 0; l < BIOEMU2.inputLayers; l += 1) {
      const prefix = `${P}/atom_encoder_layers_${l}/transformer_layer`;
      const att = `${prefix}/attention/attention_layer/pair_bias_attn`;
      const bias = o.buffer(`ie.bias${l}`, NW * BIOEMU2.atomHeads * W_Q * W_K,
        windowBias(pr, N, await this.h(`${att}/norm_z/weight`), await this.h(`${att}/proj_z/weight`), BIOEMU2.atomHeads, prNorm));
      await this.condition(steps, prefix, { c: cBuf, rows: N, width: D, outC: 6 * D, ln: s.ln, out: cond });
      await this.windowedLayer(steps, prefix, { x: q, cond, map: null, bias, keyMask: g.keyMask, N, s });
    }
    o.run(steps);
    const qHost = await o.read(q);
    return sInputs(qHost, f, host["atom_to_token_trans_0/weight"], T);
  }

  /** A relative-position encoding (FusedRelativePositionEncoder) into `out` (T^2, 256), from `prefix`'s tables. */
  async relpos(steps, prefix, g, out, accumulate) {
    const o = this.ops;
    const { f, T } = g;
    const ints = new Int32Array(4 * T);
    for (let t = 0; t < T; t += 1) {
      ints[t] = f.asym_id[t]; ints[T + t] = f.entity_id[t]; ints[2 * T + t] = f.residue_index[t]; ints[3 * T + t] = f.sym_id[t];
    }
    g.tokenInts ??= o.buffer("tokenInts", 4 * T, ints);
    const tables = await Promise.all(["embed_rel_pos/weight", "embed_rel_token/weight", "embed_same_entity/weight",
      "embed_rel_chain/weight"].map((n) => this.w(`${prefix}/${n}`, { half: false })));
    steps.push(await o.map(`relpos${accumulate ? "+" : ""}`, T * T * 256, [{ buffer: out, write: true }, { buffer: g.tokenInts, type: "i32" },
      ...tables.map((buffer) => ({ buffer }))],
    `let T = U(0u); let c = i % 256u; let p = i / 256u; let a = p / T; let b = p % T;
  let sameChain = b1[a] == b1[b]; let sameRes = b1[2u * T + a] == b1[2u * T + b]; let sameEnt = b1[T + a] == b1[T + b];
  var dres = clamp(b1[2u * T + a] - b1[2u * T + b] + 32, 0, 64); if (!sameChain) { dres = 65; }
  var dtok = clamp(i32(a) - i32(b) + 32, 0, 64); if (!(sameChain && sameRes)) { dtok = 65; }
  var dch = clamp(b1[3u * T + a] - b1[3u * T + b] + 2, 0, 4); if (!sameEnt) { dch = 5; }
  var v = b2[u32(dres) * 256u + c] + b3[u32(dtok) * 256u + c] + b5[u32(dch) * 256u + c];
  if (sameEnt) { v += b4[c]; }
  ${accumulate ? "b0[i] = b0[i] + v;" : "b0[i] = v;"}`, [T]));
  }

  /** s_init and z_init: everything before the recycle loop. */
  async trunkInit(g, sIn) {
    const { f, T } = g;
    const o = this.ops;
    const P = "sequence_embedder";
    const [sInitW, z1, z2, me, pe, tb] = await Promise.all(["s_init/weight", "z_init_1/weight", "z_init_2/weight",
      "method_conditioning/embedding/weight", "method_conditioning/pair_embedding/weight", "token_bonds/weight"]
      .map((n) => this.h(`${P}/${n}`)));
    const sInit = linearRows(sIn, T, BIOEMU2.sInputs, sInitW, BIOEMU2.single);
    for (let t = 0; t < T; t += 1) {
      for (let c = 0; c < BIOEMU2.single; c += 1) sInit[t * BIOEMU2.single + c] += me[f.method_feature[t] * BIOEMU2.single + c];
    }
    const a = linearRows(sIn, T, BIOEMU2.sInputs, z1, 256);
    const b = linearRows(sIn, T, BIOEMU2.sInputs, z2, 256);
    // the per-token half of z_init on the host: (a_i + pe_i), (b_j + pe_j); the pairwise half on the device
    for (let t = 0; t < T; t += 1) {
      for (let c = 0; c < 256; c += 1) {
        a[t * 256 + c] += pe[f.method_feature[t] * 256 + c];
        b[t * 256 + c] += pe[f.method_feature[t] * 256 + c];
      }
    }
    // contact conditioning: classes 0 and 1 are constants; a CONTACT pair (class >= 2) runs the encoder, here
    const cc = await this.hostGroup(`${P}/contact_conditioning`, ["encoder/weight", "encoder/bias", "encoding_unspecified",
      "encoding_unselected", "fourier_embedding/proj/weight", "fourier_embedding/proj/bias"]);
    const classes = new Int32Array(T * T);
    const special = [];
    const slot = new Int32Array(T * T).fill(-1);
    for (let p = 0; p < T * T; p += 1) {
      let cls = 0;
      for (let k = 0; k < 5; k += 1) if (f.contact_conditioning[p * 5 + k]) cls = k;
      classes[p] = cls;
      if (cls >= 2) {
        const thr = (f.contact_threshold[p] - 4.0) / 16.0;
        const x = new Float64Array(260);
        for (let k = 0; k < 3; k += 1) x[k] = f.contact_conditioning[p * 5 + 2 + k];
        x[3] = thr;
        for (let k = 0; k < 256; k += 1) {
          x[4 + k] = Math.cos(2 * Math.PI * (thr * cc["fourier_embedding/proj/weight"][k] + cc["fourier_embedding/proj/bias"][k]));
        }
        const v = new Float32Array(256);
        for (let c = 0; c < 256; c += 1) {
          let s = cc["encoder/bias"][c];
          for (let k = 0; k < 260; k += 1) s += x[k] * cc["encoder/weight"][c * 260 + k];
          v[c] = s;
        }
        slot[p] = special.length;
        special.push(v);
      }
    }
    const table = new Float32Array(Math.max(1, special.length) * 256 + 512);
    table.set(cc.encoding_unspecified, 0);
    table.set(cc.encoding_unselected, 256);
    special.forEach((v, k) => table.set(v, 512 + k * 256));
    const zInit = o.buffer("zInit", T * T * 256);
    const steps = [];
    await this.relpos(steps, `${P}/rel_pos`, g, zInit, false);
    const ab = new Float32Array(2 * T * 256);
    ab.set(a);
    ab.set(b, T * 256);
    const bonds = new Float32Array(T * T);
    for (let p = 0; p < T * T; p += 1) bonds[p] = f.token_bonds[p];
    steps.push(await o.map("zinit", T * T * 256, [{ buffer: zInit, write: true }, { buffer: o.buffer("zi.ab", ab.length, ab) },
      { buffer: o.buffer("zi.bonds", T * T, bonds) }, { buffer: o.buffer("zi.bondw", 256, tb) },
      { buffer: o.buffer("zi.cls", T * T, classes), type: "i32" }, { buffer: o.buffer("zi.slot", T * T, slot), type: "i32" },
      { buffer: o.buffer("zi.table", table.length, table) }],
    `let T = U(0u); let c = i % 256u; let p = i / 256u; let ii = p / T; let jj = p % T;
  var v = b0[i] + b1[ii * 256u + c] + b1[(T + jj) * 256u + c] + b2[p] * b3[c];
  let cls = b4[p];
  if (cls == 0) { v += b6[c]; } else if (cls == 1) { v += b6[256u + c]; } else { v += b6[512u + u32(b5[p]) * 256u + c]; }
  b0[i] = v;`, [T]));
    o.run(steps);
    return { sInit, zInit };
  }

  /** The MSA module's inputs, once a fold: m0 = msa_proj(feat) + s_proj(s_inputs). */
  async msaInputs(g, sIn) {
    const { f, T } = g;
    const o = this.ops;
    const S = f.msa.length / T;
    const P = "sequence_embedder/msa_module";
    const msaProj = await this.h(`${P}/msa_proj/weight`);     // (128, 36)
    const sProj = linearRows(sIn, T, BIOEMU2.sInputs, await this.h(`${P}/s_proj/weight`), 128);
    const m0 = o.buffer("msa.m0", S * T * 128);
    const feats = new Float32Array(S * T * 3);
    for (let i = 0; i < S * T; i += 1) {
      feats[i * 3] = f.has_deletion[i]; feats[i * 3 + 1] = f.deletion_value[i]; feats[i * 3 + 2] = f.msa_paired[i];
    }
    const msaT = new Float32Array(36 * 128);
    for (let c = 0; c < 128; c += 1) for (let k = 0; k < 36; k += 1) msaT[k * 128 + c] = msaProj[c * 36 + k];
    o.run([await o.map("msa-m0", S * T * 128, [{ buffer: m0, write: true }, { buffer: o.buffer("msa.idx", S * T, Int32Array.from(f.msa)), type: "i32" },
      { buffer: o.buffer("msa.feat", feats.length, feats) }, { buffer: o.buffer("msa.w", msaT.length, msaT) },
      { buffer: o.buffer("msa.sp", sProj.length, sProj) }],
    `let T = U(0u); let c = i % 128u; let st = i / 128u; let t = st % T;
  b0[i] = b3[u32(b1[st]) * 128u + c] + b2[st * 3u] * b3[33u * 128u + c] + b2[st * 3u + 1u] * b3[34u * 128u + c]
    + b2[st * 3u + 2u] * b3[35u * 128u + c] + b4[t * 128u + c];`, [T])]);
    // n_valid: rows with any live column (msa_mask)
    let valid = 0;
    for (let s = 0; s < S; s += 1) {
      let any = false;
      for (let t = 0; t < T; t += 1) any ||= f.msa_mask[s * T + t] !== 0;
      if (any) valid += 1;
    }
    return { m0, S, valid: Math.max(valid, 1) };
  }

  /** Compile everything a trunk pass runs, once per fold. */
  async trunkProgram(g, sIn) {
    const { T } = g;
    const o = this.ops;
    const P = "sequence_embedder";
    const { sInit, zInit } = await this.trunkInit(g, sIn);
    const msa = await this.msaInputs(g, sIn);
    const { S } = msa;
    const z = o.buffer("z", T * T * 256);
    const zPrev = o.buffer("zPrev", T * T * 256);
    const zln = o.buffer("zln", T * T * 256);
    const sPrev = o.buffer("sPrev", T * BIOEMU2.single);
    const s = o.buffer("s", T * BIOEMU2.single);
    const sInitBuf = o.buffer("sInit", sInit.length, sInit);
    const sln = o.buffer("sln", T * BIOEMU2.single);
    const m = o.buffer("m", S * T * 128);
    // recycle: s = s_init + s_recycle(LN s_prev); z = z_init + z_recycle(LN z_prev)
    const recycle = [];
    recycle.push(await o.layerNorm({ x: sPrev, y: sln, rows: T, width: BIOEMU2.single,
      w: await this.w(`${P}/s_norm/weight`, { half: false }), b: await this.w(`${P}/s_norm/bias`, { half: false }) }));
    recycle.push(await o.map("copy", T * BIOEMU2.single, [{ buffer: s, write: true }, { buffer: sInitBuf }], "b0[i] = b1[i];"));
    recycle.push(await o.linear({ x: sln, w: await this.w(`${P}/s_recycle/weight`), y: s, rows: T, inC: BIOEMU2.single, outC: BIOEMU2.single, accumulate: true }));
    recycle.push(await o.layerNorm({ x: zPrev, y: zln, rows: T * T, width: 256,
      w: await this.w(`${P}/z_norm/weight`, { half: false }), b: await this.w(`${P}/z_norm/bias`, { half: false }) }));
    recycle.push(await o.map("copy", T * T * 256, [{ buffer: z, write: true }, { buffer: zInit }], "b0[i] = b1[i];"));
    recycle.push(await o.linear({ x: zln, w: await this.w(`${P}/z_recycle/weight`), y: z, rows: T * T, inC: 256, outC: 256, accumulate: true }));
    recycle.push(await o.map("copy", S * T * 128, [{ buffer: m, write: true }, { buffer: msa.m0 }], "b0[i] = b1[i];"));
    // the MSA layers' MSA halves
    const msaSteps = [];
    const sm = this.scratch("msa", S * T, 128);
    const v = o.buffer("pwa.v", S * T * 256);
    const gate = o.buffer("pwa.g", S * T * 256);
    const ow = o.buffer("pwa.o", S * T * 256);
    const w8 = o.buffer("pwa.w", 8 * T * T);
    const ab = o.buffer("opm.ab", 2 * S * T * 32);
    const chunk = Math.max(1, Math.min(T, Math.floor((32 << 20) / (T * 1024))));
    const outer = o.buffer("opm.outer", chunk * T * 1024);
    for (let l = 0; l < BIOEMU2.msaBlocks; l += 1) {
      const L = `${P}/msa_module/layers/${l}`;
      const steps = [];
      const pwa = `${L}/pair_weighted_averaging`;
      steps.push(await o.layerNorm({ x: m, y: sm.ln, rows: S * T, width: 128,
        w: await this.w(`${pwa}/norm_m/weight`, { half: false }), b: await this.w(`${pwa}/norm_m/bias`, { half: false }) }));
      steps.push(await o.linear({ x: sm.ln, w: await this.w(`${pwa}/proj_m/weight`), y: v, rows: S * T, inC: 128, outC: 256 }));
      steps.push(await o.linear({ x: sm.ln, w: await this.w(`${pwa}/proj_g/weight`), y: gate, rows: S * T, inC: 128, outC: 256, act: "sigmoid" }));
      steps.push(await o.layerNorm({ x: z, y: zln, rows: T * T, width: 256,
        w: await this.w(`${pwa}/norm_z/weight`, { half: false }), b: await this.w(`${pwa}/norm_z/bias`, { half: false }) }));
      // logits (8, T, T): heads first
      steps.push(await o.gemm({ m: T * T, n: 8, k: 256, a: zln, b: await this.w(`${pwa}/proj_z/weight`), c: w8,
        A: { m: 256, k: 1 }, B: { n: 256, k: 1 }, C: { m: 1, n: T * T } }));
      steps.push(await o.softmax({ l: w8, rows: 8 * T, len: T, rowsPer: T, L: { s0: T * T, r: T } }));
      steps.push(await o.gemm({ m: T, n: 32, k: T, a: w8, b: v, c: ow, batches: [S, 8],
        A: { off: 0, s0: 0, s1: T * T, m: T, k: 1 }, B: { off: 0, s0: T * 256, s1: 32, n: 1, k: 256 },
        C: { off: 0, s0: T * 256, s1: 32, m: 256, n: 1 } }));
      steps.push(await o.map("mul", S * T * 256, [{ buffer: ow, write: true }, { buffer: gate }], "b0[i] = b0[i] * b1[i];"));
      steps.push(await o.linear({ x: ow, w: await this.w(`${pwa}/proj_o/weight`), y: m, rows: S * T, inC: 256, outC: 128, accumulate: true }));
      await this.transition(steps, `${L}/msa_transition`, { input: m, rows: S * T, width: 128, hidden: 512, out: m,
                                                          scratch: sm, norm: true, accumulate: true });
      // outer product mean: z += proj_o((a (x) b) / n_valid) + bias, by chunks of rows
      const opm = `${L}/outer_product_mean`;
      steps.push(await o.layerNorm({ x: m, y: sm.ln, rows: S * T, width: 128,
        w: await this.w(`${opm}/norm/weight`, { half: false }), b: await this.w(`${opm}/norm/bias`, { half: false }) }));
      const projAb = await this.w(`${opm}/proj_ab/weight`);
      steps.push(await o.linear({ x: sm.ln, w: projAb, y: ab, rows: S * T, inC: 128, outC: 64 }));
      for (let i0 = 0; i0 < T; i0 += chunk) {
        const rows = Math.min(chunk, T - i0);
        // outer[il, j, c, d] = sum_s a[s, i0 + il, c] b[s, j, d] / n_valid
        steps.push(await o.gemm({ m: 32, n: 32, k: S, a: ab, b: ab, c: outer, batches: [rows, T], alpha: 1 / msa.valid,
          A: { off: i0 * 64, s0: 64, s1: 0, m: 1, k: T * 64 },
          B: { off: 32, s0: 0, s1: 64, n: 1, k: T * 64 },
          C: { off: 0, s0: T * 1024, s1: 1024, m: 32, n: 1 } }));
        steps.push(await o.linear({ x: outer, w: await this.w(`${opm}/proj_o/weight`), y: z, yOff: i0 * T * 256, rows: rows * T,
          inC: 1024, outC: 256, accumulate: true, bias: await this.w(`${opm}/proj_o/bias`, { half: false }) }));
      }
      msaSteps.push(o.program(steps));
    }
    // the pair track's state: AllocatedGpuBuffer-shaped wrappers over our buffers (it binds `.buffer`)
    const pairState = { buffer: { buffer: z }, maskBuffer: { buffer: o.buffer("pairMask", T * T, new Float32Array(T * T).fill(1)) } };
    // 🔴 THE ACCUMULATOR IS FLOAT32 ALWAYS: BioEmu2's pair reaches ~3e5 (6MRR, float32 arm) - past float16's 65,504 -
    // where ESMFold2's, which this trunk was tuned for, stays small, and the trunk's device default on a part with
    // shader-f16 is f16 staging AND f16 accumulation. A stock Chrome on Apple silicon took that and returned a fold of
    // NaN (pass 1's pair: 3 infinities, every stage after NaN); the developer flags hid it, their matrix-unit kernels
    // summing wider. Staging stays the device's (a LayerNorm's output, bounded)
    const trunkAllocator = new GpuBufferAllocator(this.device, true);
    const trunk = new Esmfold2TrunkGpu(this.device, {
      allocator: trunkAllocator,
      ...(this.options.trunkPrecision === "f32"
        ? { stagedPrecision: "f32", accumulatePrecision: "f32" }
        : { accumulatePrecision: "f32" }),
    });
    const [msaBlocks, mixerBlocks] = await Promise.all([
      Promise.all(Array.from({ length: BIOEMU2.msaBlocks }, (_, l) => trunkBlockWeights(this.read, l, "msa"))),
      Promise.all(Array.from({ length: BIOEMU2.pairmixerBlocks }, (_, l) => trunkBlockWeights(this.read, l, "pairmixer"))),
    ]);
    const allBlocks = [...msaBlocks, ...mixerBlocks];
    const copyBack = [await o.map("copy", T * T * 256, [{ buffer: zPrev, write: true }, { buffer: z }], "b0[i] = b1[i];"),
                      await o.map("copy", T * BIOEMU2.single, [{ buffer: sPrev, write: true }, { buffer: s }], "b0[i] = b1[i];")];
    return {
      z, s, sPrev, zPrev, zInit, sInit, trunkAllocator,
      pass: async (onBlock) => {
        o.run(recycle);
        await trunk.run(pairState, allBlocks, {
          n: T, channels: 256, readback: false,
          beforeBlock: (l) => { if (l < msaSteps.length) o.run(msaSteps[l]); },
          onBlockDone: onBlock && ((done) => {
            if (done > msaBlocks.length) onBlock(done - msaBlocks.length, mixerBlocks.length);
          }),
        });
        o.run(copyBack);
      },
    };
  }

  /** The distogram's contact probability P(d < 8 A), and its logits, from the trunk pair. */
  async distogram(g, z) {
    const { T } = g;
    const o = this.ops;
    const sym = o.buffer("dg.sym", T * T * 256);
    const logits = o.buffer("dg.logits", T * T * 64);
    const steps = [await o.map("transpose-add", T * T * 256, [{ buffer: sym, write: true }, { buffer: z }],
      "let T = U(0u); let c = i % 256u; let p = i / 256u; let a = p / T; let b = p % T; b0[i] = b1[i] + b1[(b * T + a) * 256u + c];", [T]),
    await o.linear({ x: sym, w: await this.w("distogram_module/distogram/weight"), y: logits, rows: T * T, inC: 256, outC: 64,
                     bias: await this.w("distogram_module/distogram/bias", { half: false }) })];
    o.run(steps);
    const host = await o.read(logits);
    // Boltz's bins (64 over 2-22 A, edges linspace(2, 22, 63)); P(d < 8) is the first 19
    const contact = new Float32Array(T * T);
    for (let p = 0; p < T * T; p += 1) {
      let mx = -Infinity;
      for (let k = 0; k < 64; k += 1) mx = Math.max(mx, host[p * 64 + k]);
      let sum = 0;
      let near = 0;
      for (let k = 0; k < 64; k += 1) { const e = Math.exp(host[p * 64 + k] - mx); sum += e; if (k < 19) near += e; }
      contact[p] = near / sum;
    }
    sym.destroy();
    logits.destroy();
    return { logits: host, contact };
  }

  /**
   * DualStreamDiffusionModule.compute_static_context and everything else a denoiser call reads that does not
   * depend on the coordinates; returns a recorded program and the buffers a sampler step writes and reads.
   */
  async denoiserProgram(g, sIn, sTrunkHost, zTrunk, { dt = 0, dynamics = false, coordsCond = null, samples = 1 } = {}) {
    const { f, T, N, NW } = g;
    const o = this.ops;
    const P = "structure_module";
    const DT = BIOEMU2.token;
    const DA = BIOEMU2.atom;
    // pairwise conditioner: z = proj(LN(concat(z_trunk, relpos))) + 2 transitions
    const rel = o.buffer("ctx.rel", T * T * 256);
    const cat = o.buffer("ctx.cat", T * T * 512);
    const zc = o.buffer("ctx.z", T * T * 256);
    const pre = [];
    await this.relpos(pre, "rel_pos", g, rel, false);
    pre.push(await o.map("concat2", T * T * 512, [{ buffer: cat, write: true }, { buffer: zTrunk }, { buffer: rel }],
      "let p = i / 512u; let c = i % 512u; if (c < 256u) { b0[i] = b1[p * 256u + c]; } else { b0[i] = b2[p * 256u + c - 256u]; }"));
    const pc = `${P}/pairwise_conditioner`;
    pre.push(await o.layerNorm({ x: cat, y: cat, rows: T * T, width: 512,
      w: await this.w(`${pc}/dim_pairwise_init_proj_0/weight`, { half: false }), b: await this.w(`${pc}/dim_pairwise_init_proj_0/bias`, { half: false }) }));
    pre.push(await o.linear({ x: cat, w: await this.w(`${pc}/dim_pairwise_init_proj_1/weight`), y: zc, rows: T * T, inC: 512, outC: 256 }));
    const ps = this.scratch("pc", T * T, 256);
    for (let i = 0; i < 2; i += 1) {
      await this.transition(pre, `${pc}/transitions_${i}`, { input: zc, rows: T * T, width: 256, hidden: 512, out: zc,
                                                             scratch: ps, norm: true, accumulate: true });
    }
    // the token stream's per-layer pair biases (16, T, T): LN(z) (no offset) . proj_z
    const tokenBias = [];
    const zln = o.buffer("ctx.zln", T * T * 256);
    for (let l = 0; l < BIOEMU2.denoiserLayers; l += 1) {
      const att = `${P}/layers/${l}/token_block/attention/pair_bias_attn`;
      const bias = o.buffer(`ctx.tbias${l}`, BIOEMU2.tokenHeads * T * T);
      pre.push(await o.layerNorm({ x: zc, y: zln, rows: T * T, width: 256, w: await this.w(`${att}/norm_z/weight`, { half: false }) }),
        await o.gemm({ m: T * T, n: BIOEMU2.tokenHeads, k: 256, a: zln, b: await this.w(`${att}/proj_z/weight`), c: bias,
          A: { m: 256, k: 1 }, B: { n: 256, k: 1 }, C: { m: 1, n: T * T } }));
      tokenBias.push(bias);
    }
    const pa = `${P}/atom_feature_embedder`;
    // z_to_p on the device (T^2 x 256 -> 16), submitted before `pre` and overlapped with host atomFeatures
    const zp = o.buffer("ctx.zp", T * T * 16);
    o.run([await o.layerNorm({ x: zTrunk, y: zln, rows: T * T, width: 256,
      w: await this.w(`${pa}/z_to_p_trans_0/weight`, { half: false }), b: await this.w(`${pa}/z_to_p_trans_0/bias`, { half: false }) }),
    await o.linear({ x: zln, w: await this.w(`${pa}/z_to_p_trans_1/weight`, { half: false }), y: zp, rows: T * T, inC: 256, outC: 16 })]);
    const zpRead = o.read(zp);
    o.run(pre);
    rel.destroy();
    cat.destroy();
    zln.destroy();
    for (const b of Object.values(ps)) b.destroy();
    // the atom features and their window pair, on the host
    const host = await this.hostGroup(pa, ["proj_atom_continuous/weight", "embed_element/weight",
      ...[0, 1, 2, 3].map((i) => `embed_atom_name_chars_${i}/weight`), "embed_atompair_ref_pos/weight",
      "embed_atompair_ref_dist/weight", "embed_atompair_mask/weight", "c_to_p_trans_q_1/weight", "c_to_p_trans_k_1/weight",
      "p_mlp_1/weight", "p_mlp_3/weight", "p_mlp_5/weight", "s_to_c_trans_0/weight", "s_to_c_trans_0/bias",
      "s_to_c_trans_1/weight",
      "r_to_q_trans/weight", "cond_to_q_trans/weight", "dt_gate_embed/taus", "dt_gate_log_w"]);
    const { c, pr } = atomFeatures(host, f);
    const q0 = Float32Array.from(c);
    const sToC = linearRows(lnRows(sTrunkHost, T, BIOEMU2.single, host["s_to_c_trans_0/weight"], host["s_to_c_trans_0/bias"]),
                            T, BIOEMU2.single, host["s_to_c_trans_1/weight"], DA);
    for (let a = 0; a < N; a += 1) {
      if (!f.atom_pad_mask[a]) continue;
      const t = f.token_idx_at_atom[a];
      for (let k = 0; k < DA; k += 1) c[a * DA + k] += sToC[t * DA + k];
    }
    const zpHost = await zpRead;
    zp.destroy();
    addTokenPairToWindows(pr, zpHost, T, f);
    atomPairTail(host, c, pr, N);
    const prNorm = normalizeWindowPair(pr);
    const atomBiasWeights = await Promise.all(Array.from({ length: BIOEMU2.denoiserLayers }, async (_, l) => {
      const att = `${P}/layers/${l}/atom_block/transformer_layer/attention/attention_layer/pair_bias_attn`;
      const [normZ, projZ] = await Promise.all([this.h(`${att}/norm_z/weight`), this.h(`${att}/proj_z/weight`)]);
      return { normZ, projZ };
    }));
    const atomBias = [];
    for (let l = 0; l < BIOEMU2.denoiserLayers; l += 1) {
      atomBias.push(o.buffer(`ctx.abias${l}`, NW * BIOEMU2.atomHeads * W_Q * W_K,
        windowBias(pr, N, atomBiasWeights[l].normZ, atomBiasWeights[l].projZ, BIOEMU2.atomHeads, prNorm)));
    }

    // the single conditioning's coordinate-free half: single_embed(LN(concat(s_trunk, s_inputs))) + the lag term
    const sc = `${P}/single_conditioner`;
    const sch = await this.hostGroup(sc, ["norm_single/weight", "norm_single/bias", "single_embed/weight", "single_embed/bias",
      "fourier_embed/proj/weight", "fourier_embed/proj/bias", "norm_fourier/weight", "norm_fourier/bias",
      "fourier_to_single/weight", "dt_embed/taus", "dt_embed/quality_factors", "dt_embed_to_single/weight"]);
    const catS = new Float32Array(T * 839);
    for (let t = 0; t < T; t += 1) {
      catS.set(sTrunkHost.subarray(t * 384, t * 384 + 384), t * 839);
      catS.set(sIn.subarray(t * 455, t * 455 + 455), t * 839 + 384);
    }
    const s0 = linearRows(lnRows(catS, T, 839, sch["norm_single/weight"], sch["norm_single/bias"]), T, 839,
                          sch["single_embed/weight"], DT, sch["single_embed/bias"]);
    if (dynamics) {
      const dtv = linearRows(dampedHarmonic(dt, sch["dt_embed/taus"], sch["dt_embed/quality_factors"]), 1, 288,
                             sch["dt_embed_to_single/weight"], DT);
      for (let t = 0; t < T; t += 1) for (let k = 0; k < DT; k += 1) s0[t * DT + k] += dtv[k];
    }
    // a0 = s_trunk_to_a(s_trunk)
    const a0 = linearRows(lnRows(sTrunkHost, T, 384, await this.h(`${P}/s_trunk_to_a_0/weight`), await this.h(`${P}/s_trunk_to_a_0/bias`)),
                          T, 384, await this.h(`${P}/s_trunk_to_a_1/weight`), DT);
    // the companion's coordinates, gated by the lag (BioEmu2's propagation)
    let gate = 0;
    if (dynamics) {
      const taus = host["dt_gate_embed/taus"];
      for (let i = 0; i < taus.length; i += 1) {
        const lw = host["dt_gate_log_w"][i];
        gate += Math.exp(-dt / taus[i]) * (lw > 20 ? lw : Math.log1p(Math.exp(lw)));
      }
    }

    // device buffers the step reads and writes. S samples run through one call, SAMPLE-MAJOR: the conditioning
    // (single, cond and every per-layer projection of it) depends on sigma alone and is computed once for all of them
    const S = samples;
    const X = o.buffer("dn.x", S * N * 3);
    const out = o.buffer("dn.out", S * N * 3);
    const scalars = o.buffer("dn.scalars", 8);
    const fvec = o.buffer("dn.fourier", DT);
    const companion = o.buffer("dn.companion", S * N * 3);
    if (coordsCond) for (let sm = 0; sm < S; sm += 1) o.write(companion, coordsCond, sm * N * 3);
    const s0Buf = o.buffer("dn.s0", T * DT, s0);
    const cond = o.buffer("dn.cond", T * DT);
    const a = o.buffer("dn.a", S * T * DT);
    const a0Buf = o.buffer("dn.a0", T * DT, a0);
    const q = o.buffer("dn.q", S * N * DA);
    const q0Buf = o.buffer("dn.q0", N * DA, q0);
    const centre = o.buffer("dn.centre", T, Int32Array.from(f.token_center_atom_index));
    const atomMask = o.buffer("dn.mask", N, Int32Array.from(f.atom_mask));
    const rToQ = o.buffer("dn.rtoq", DA * 3, host["r_to_q_trans/weight"]);
    const condToQ = o.buffer("dn.ctoq", DA * 3, host["cond_to_q_trans/weight"]);
    const rTokToA = await this.w(`${P}/r_token_to_a/weight`, { half: false });
    // row maps over the S samples: an atom row to its token's cond row (-1 on padding), a token row to its cond
    // row, and an atom row to its own sample's token row
    const atomCond = new Int32Array(S * N);
    const tokenCond = new Int32Array(S * T);
    const atomSampleToken = new Int32Array(S * N);
    for (let sm = 0; sm < S; sm += 1) {
      for (let at = 0; at < N; at += 1) {
        const t = f.atom_pad_mask[at] ? f.token_idx_at_atom[at] : -1;
        atomCond[sm * N + at] = t;
        atomSampleToken[sm * N + at] = t < 0 ? -1 : sm * T + t;
      }
      for (let t = 0; t < T; t += 1) tokenCond[sm * T + t] = t;
    }
    const atomMap = o.buffer("dn.atomMap", S * N, atomCond);
    const tokenMap = o.buffer("dn.tokenMap", S * T, tokenCond);
    const atomTokenS = o.buffer("dn.atomTokenS", S * N, atomSampleToken);
    // (a layer at a time - the control - or conditionGroup's views below)
    const unbatched = this.options.batchConditioning === false;
    const condAtom1 = unbatched ? o.buffer("dn.condAtom", T * 6 * DA) : null;
    const condTok1 = unbatched ? o.buffer("dn.condTok", T * 6 * DT) : null;
    const condQ1 = unbatched ? o.buffer("dn.condQ", T * 1920) : null;
    const condS1 = unbatched ? o.buffer("dn.condS", T * 2560) : null;
    const sC = this.scratch("dc", T, DT);
    const sT = this.scratch("dt", S * T, DT, { logits: o.buffer("dt.logits", S * BIOEMU2.tokenHeads * T * T) });
    const sA = this.scratch("da", S * N, DA, { kvPad: o.buffer("da.kvPad", S * (N + 2 * CTX) * 2 * DA),
                                               logits: o.buffer("da.logits", NW * BIOEMU2.atomHeads * W_Q * W_K) });
    // (the coupling's atom-side mixer split - couplingSplit - unless the options turn it off: the control)
    const split = this.options.couplingSplit !== false;
    const cs = !split ? null : {
      gA: o.buffer("cs.gA", (T + 1) * DT), gQ: o.buffer("cs.gQ", (T + 1) * DA), beta: o.buffer("cs.beta", (T + 1) * 896),
      QB: o.buffer("cs.QB", (T + 1) * 2 * 3584), tstats: o.buffer("cs.tstats", S * T * 3), xa: o.buffer("cs.xa", S * T * DT),
      P: o.buffer("cs.P", S * T * 3584), astats: o.buffer("cs.astats", S * N * 6), xq: o.buffer("cs.xq", S * N * DA),
      Z: this.options.fuseCombine === false ? o.buffer("cs.Z", S * N * 3584) : null,   // (fused: never stored)
    };
    const sq = o.buffer("dn.sq", S * N * 896);
    const sqh = o.buffer("dn.sqh", S * N * 896);
    const sq12 = o.buffer("dn.sq12", S * N * 3584);
    const sqt = o.buffer("dn.sqt", S * N * 1792);
    const dq = o.buffer("dn.dq", S * N * DA);
    const qs = o.buffer("dn.qs", S * T * 896);
    const qsh = o.buffer("dn.qsh", S * T * 896);
    const qs12 = o.buffer("dn.qs12", S * T * 3584);
    const qst = o.buffer("dn.qst", S * T * 1792);
    const da = o.buffer("dn.da", S * T * DT);
    const lnT = o.buffer("dn.lnT", T * DT);
    // the layers' conditioning in GROUPS of G (conditionGroup; batchConditioning: false is the control, a layer at a
    // time): G as many as fit 64 MiB of group buffers - 10 at 68 tokens (measured: 128 MiB, all 16, 239.2 ms a call; 32 MiB 240.6; 16 MiB 241.2)
    const batchCond = this.options.batchConditioning !== false;
    const LAYERS = BIOEMU2.denoiserLayers;
    const perLayer = T * (DT + 6 * DA + 6 * DT + 1920 + 2560) + (split ? (T + 1) * (DT + DA + 896 + 2 * 3584) : 0);
    const G = batchCond ? Math.max(1, Math.min(LAYERS, Math.floor((this.options.condGroupMiB ?? 96) * 2 ** 20 / 4 / perLayer))) : 0;
    const groupBufs = !batchCond ? null : {
      cin: o.buffer("cg.cin", 4 * G * T * DT), atom: o.buffer("cg.atom", G * T * 6 * DA), tok: o.buffer("cg.tok", G * T * 6 * DT),
      q: o.buffer("cg.q", G * T * 1920), s: o.buffer("cg.s", G * T * 2560),
      ...(split ? { gA: o.buffer("cg.gA", G * (T + 1) * DT), gQ: o.buffer("cg.gQ", G * (T + 1) * DA),
                    gAQ: o.buffer("cg.gAQ", G * 2 * (T + 1) * 896),
                    QB: o.buffer("cg.QB", G * (T + 1) * 2 * 3584) } : {}),
    };
    const xhat = batchCond ? o.buffer("cg.xhat", T * DT) : null;
    const hid = o.buffer("dn.hid", S * N * 384);
    const hidT = o.buffer("dn.hidT", S * T * 384);

    const steps = [];
    // the single conditioning (shared); q and a per sample from r = x c_in on real atoms
    steps.push(await o.map("cond-init", T * DT, [{ buffer: cond, write: true }, { buffer: s0Buf }, { buffer: fvec }],
      "b0[i] = b1[i] + b2[i % 768u];"));
    for (let i = 0; i < 2; i += 1) {
      await this.transition(steps, `${sc}/transitions_${i}`, { input: cond, rows: T, width: DT, hidden: 2 * DT, out: cond,
                                                             scratch: sC, norm: true, accumulate: true });
    }
    steps.push(await o.map(dynamics ? "q-init-dyn" : "q-init", S * N * DA, [{ buffer: q, write: true }, { buffer: q0Buf }, { buffer: X },
      { buffer: scalars }, { buffer: rToQ }, { buffer: atomMask, type: "i32" }, { buffer: companion }, { buffer: condToQ }],
    `let r = i / 128u; let c = i % 128u; let a = r % U(0u); var v = b1[a * 128u + c];
  if (b5[a] != 0) { let s = b3[0]; v += s * (b2[r * 3u] * b4[c * 3u] + b2[r * 3u + 1u] * b4[c * 3u + 1u] + b2[r * 3u + 2u] * b4[c * 3u + 2u]); }
  ${dynamics ? "v += F(0u) * (b6[r * 3u] * b7[c * 3u] + b6[r * 3u + 1u] * b7[c * 3u + 1u] + b6[r * 3u + 2u] * b7[c * 3u + 2u]);" : ""}
  b0[i] = v;`, [N], [gate]));
    steps.push(await o.map("a-init", S * T * DT, [{ buffer: a, write: true }, { buffer: a0Buf }, { buffer: X }, { buffer: scalars },
      { buffer: rTokToA }, { buffer: centre, type: "i32" }],
    `let r = i / 768u; let c = i % 768u; let sm = r / U(0u); let t = r % U(0u); let at = sm * U(1u) + u32(b5[t]); let s = b3[0];
  b0[i] = b1[t * 768u + c] + s * (b2[at * 3u] * b4[c * 3u] + b2[at * 3u + 1u] * b4[c * 3u + 1u] + b2[at * 3u + 2u] * b4[c * 3u + 2u]);`,
    [T, N]));
    if (batchCond) steps.push(await o.layerNorm({ x: cond, y: xhat, rows: T, width: DT }));
    let group = [];
    for (let l = 0; l < LAYERS; l += 1) {
      const L = `${P}/layers/${l}`;
      const ab = `${L}/atom_block/transformer_layer`;
      const tb = `${L}/token_block`;
      const cp = `${L}/coupling`;
      if (batchCond && l % G === 0) {
        group = await this.conditionGroup(steps, { xhat, T, l0: l, n: Math.min(G, LAYERS - l), G, P, buffers: groupBufs });
      }
      const cv = batchCond ? group[l % G] : null;
      const condAtom = cv ? cv.atom : condAtom1, condTok = cv ? cv.tok : condTok1;
      const condQ = cv ? cv.q : condQ1, condS = cv ? cv.s : condS1;
      if (!cv) await this.condition(steps, ab, { c: cond, rows: T, width: DT, outC: 6 * DA, ln: lnT, out: condAtom });
      await this.windowedLayer(steps, ab, { x: q, cond: condAtom, map: atomMap, bias: atomBias[l], keyMask: g.keyMask, N, s: sA, S });
      if (!cv) await this.condition(steps, tb, { c: cond, rows: T, width: DT, outC: 6 * DT, ln: lnT, out: condTok });
      await this.tokenLayer(steps, tb, { x: a, cond: condTok, map: tokenMap, bias: tokenBias[l], T, s: sT, S });
      if (!cv) {
        await this.condition(steps, cp, { c: cond, rows: T, width: DT, outC: 1920, ln: lnT, out: condQ,
                                          l0: "q_coupling_cond_0", l2: "q_coupling_cond_2" });
        await this.condition(steps, cp, { c: cond, rows: T, width: DT, outC: 2560, ln: lnT, out: condS,
                                          l0: "s_coupling_cond_0", l2: "s_coupling_cond_2" });
      }
      // q_to_s's input FIRST - concat(q[own sample, centre(t)], a) per token, modulated by the token's s coupling: s_to_q
      // writes q in place (its fc3's gated residual), and both mixers read the layer's q and a from before either update
      const preTok = Boolean(split && cv?.pre);
      steps.push(await o.catQaLn({ q, a, centre, condS, qsh, rows: S * T, T, N,
        ...(preTok ? { gA: cv.pre.gA, tstats: cs.tstats, xa: cs.xa } : {}) }));
      // s_to_q: concat(a[own sample, token(atom)], q) per atom, modulated by the token's q coupling, a SwiGLU to 128
      if (split) {
        await this.couplingSplit(steps, `${cp}/s_to_q_mixer`, { a, q, condQ, atomTokenS, S, T, N, cs, out: sqt, pre: cv?.pre, preTok });
      } else {
        steps.push(await o.map("cat-aq", S * N * 896, [{ buffer: sq, write: true }, { buffer: a }, { buffer: q }, { buffer: atomTokenS, type: "i32" }],
          `let at = i / 896u; let c = i % 896u; let t = b3[at];
  if (c < 768u) { if (t >= 0) { b0[i] = b1[u32(t) * 768u + c]; } else { b0[i] = 0.0; } } else { b0[i] = b2[at * 128u + c - 768u]; }`));
        steps.push(await o.layerNorm({ x: sq, y: sqh, rows: S * N, width: 896, mod: { cond: condQ, stride: 1920, alpha: 0, beta: 896, map: atomMap } }));
        steps.push(await o.linear({ x: sqh, w: await this.swiglu(`${cp}/s_to_q_mixer`), y: sq12, rows: S * N, inC: 896, outC: 3584 }));
        steps.push(await o.map("swiglu", S * N * 1792, [{ buffer: sq12 }, { buffer: sqt, write: true }],
          `let r = i / U(0u); let c = i % U(0u); let g = b0[r * 2u * U(0u) + c]; let v = b0[r * 2u * U(0u) + U(0u) + c];
  b1[i] = g / (1.0 + exp(-g)) * v;`, [1792]));
      }
      // (q += gamma_q s_to_q in fc3's epilogue - ops.gemm's gate)
      steps.push(await o.linear({ x: sqt, w: await this.w(`${cp}/s_to_q_mixer/fc3/weight`), y: q, rows: S * N, inC: 1792, outC: DA,
                                  gate: { cond: condQ, stride: 1920, off: 1792, map: atomMap } }));
      const qsFused = await o.swigluLinear({ x: qsh, w: await this.swiglu(`${cp}/q_to_s_mixer`), y: qst, rows: S * T, inC: 896, hidden: 1792 });
      if (qsFused) steps.push(qsFused);
      else {
        steps.push(await o.linear({ x: qsh, w: await this.swiglu(`${cp}/q_to_s_mixer`), y: qs12, rows: S * T, inC: 896, outC: 3584 }));
        steps.push(await o.map("swiglu", S * T * 1792, [{ buffer: qs12 }, { buffer: qst, write: true }],
          `let r = i / U(0u); let c = i % U(0u); let g = b0[r * 2u * U(0u) + c]; let v = b0[r * 2u * U(0u) + U(0u) + c];
  b1[i] = g / (1.0 + exp(-g)) * v;`, [1792]));
      }
      // (a += gamma_s q_to_s, likewise)
      steps.push(await o.linear({ x: qst, w: await this.w(`${cp}/q_to_s_mixer/fc3/weight`), y: a, rows: S * T, inC: 1792, outC: DT,
                                  gate: { cond: condS, stride: 2560, off: 1792, map: tokenMap } }));
    }
    // readout: LN(q W_q + a[token] W_s) (weight only) . to_out_1, then the EDM output
    const ro = `${P}/readout`;
    steps.push(await o.linear({ x: q, w: await this.w(`${ro}/q_to_hidden/weight`), y: hid, rows: S * N, inC: DA, outC: 384 }));
    steps.push(await o.linear({ x: a, w: await this.w(`${ro}/s_to_hidden/weight`), y: hidT, rows: S * T, inC: DT, outC: 384 }));
    steps.push(await o.readoutTail({ hid, hidT, atomTokenS,
      wLn: await this.w(`${ro}/to_out_0/weight`, { half: false }),
      wOut: await this.w(`${ro}/to_out_1/weight`, { half: false }),
      X, scalars, out, rows: S * N }));
    const program = o.program(steps);
    const fourier = sch;
    return {
      X, out, companion, program, N, T, samples: S, context: { z: zc, q0, p: pr },
      /** One denoiser call: S samples' noisy coordinates (S x N x 3, padded, sample-major) at sigma -> denoised. */
      denoise: async (x, sigma) => {
        const cIn = 1 / Math.sqrt(sigma * sigma + SIGMA_DATA * SIGMA_DATA);
        const cSkip = SIGMA_DATA * SIGMA_DATA / (sigma * sigma + SIGMA_DATA * SIGMA_DATA);
        const cOut = sigma * SIGMA_DATA / Math.sqrt(SIGMA_DATA * SIGMA_DATA + sigma * sigma);
        // r = x c_in on real atoms: the mask is applied where X is read (q-init's b5, a-init reads centres)
        o.write(scalars, new Float32Array([cIn, sigma, cSkip, cOut, 0, 0, 0, 0]));
        const t = 0.25 * Math.log(sigma / SIGMA_DATA);
        const four = new Float32Array(256);
        for (let k = 0; k < 256; k += 1) {
          four[k] = Math.cos(2 * Math.PI * (t * fourier["fourier_embed/proj/weight"][k] + fourier["fourier_embed/proj/bias"][k]));
        }
        const fv = linearRows(lnRows(four, 1, 256, fourier["norm_fourier/weight"], fourier["norm_fourier/bias"]), 1, 256,
                              fourier["fourier_to_single/weight"], DT);
        o.write(fvec, fv);
        o.write(X, x);
        o.run(program);
        return o.read(out);
      },
      setCompanion: (values) => o.write(companion, values),
    };
  }
}
