/**
 * ProteinEBM (jproney/ProteinEBM, MIT) on WebGPU: the energy E(r, t) of a CA trace given its sequence, and its
 * conservative score -dE/dr - a forward pass and the backward pass to the coordinates, both recorded once per length
 * and replayed at every Langevin step.
 *
 * The network (protein_ebm/model/ebm.py, layers.py; Boltz-1's diffusion transformer):
 *
 *   X0 = [seq_emb(aa) | W_n (0.1 r) | contact_emb(0) | W_sc (0.1 sc)]                 (N, 1024)
 *   s  = single_embed(LN(X0)) + fourier_to_single(LN(cos 2pi (w t + b)));  s += T(s) twice          (N, 512)
 *   z  = W_p LN([s_i | s_j | relpos_ij]);  z += T(z) twice                                          (N^2, 128)
 *   a  = s;  16 x { a += sigmoid(s W_op) * Attn(AdaLN(a, s), bias = LN(z) W_z);
 *                   a += sigmoid(s W_top) * W_b2a (SwiGLU(AdaLN(a, s) W_sg) * AdaLN(a, s) W_a2b) }
 *   E  = sum_i |W_r LN(a)_i|^2
 *
 * The backward is written by hand, module by module, after colabfold-cuda's grad/af3diff.cuh (AdaLN, AdaLN-zero, the
 * pair-biased attention) - but for the INPUT only: no weight gradient is formed, which is half of a training backward.
 * The gradient reaches r through the coordinate embedding alone; s, and through it z, depend on r, so it flows back
 * through every gate and AdaLN conditioning in all sixteen layers and through the pair track.
 *
 * Three things are folded so the backward stays cheap:
 *
 *   * AdaLN's s_norm has a weight and no bias, so LN(s; w) = shat * w with ONE shat shared by all 32 AdaLNs: w is folded
 *     into s_scale / s_bias on the host, the gradient to shat is accumulated over every layer, and ONE LayerNorm
 *     backward takes it to s. The same for z: every layer's proj_z LayerNorm is the same zhat with its own affine, so
 *     W_z * w is folded and the gradient to zhat summed over layers; proj_z's bias term is constant along a softmax
 *     row and drops out.
 *   * The pair LayerNorm over [s_i | s_j | relpos_ij] (1152 wide) is never materialised: its statistics come from
 *     per-residue sums and a relative-position table, its projection from per-residue products (A_i + B_j + T_ij), and
 *     its backward reduces to per-residue sums of rstd-weighted pair gradients and three scalars per pair
 *     (pairInitBackward). O(N^2 128), not O(N^2 1152).
 *   * The Fourier time embedding is one 512-vector a time level, computed on the host and folded into single_embed's
 *     bias.
 *
 * Masks are all ones (one chain, no padding), as every protocol in the vendor's repository runs.
 */
import { Bioemu2Ops } from "../bioemu2/ops.js";
import { lnForwardSource, lnBackwardSource, SOFTMAX_SOURCE, SOFTMAX_BACKWARD_SOURCE } from "./kernels.js";
import { PROTEINEBM, relposClass, fourierRow } from "../../shared/proteinebm/model.js";

const D = 512;          // the single (and transformer) width: 2 token_s
const Z = 128;          // the pair width
const H = 8;
const HD = 64;
const X0W = 1024;       // the residue embedding: four 256-wide parts

export class ProteinEbmGpu {
  constructor(device) {
    this.device = device;
    this.ops = new Bioemu2Ops(device);
    this.ops.compiled = new Map();
    this.layers = PROTEINEBM.layers;
  }

  /** Upload the weights (`read` is shared/proteinebm/model.js's bundle reader), with the host-side folds. */
  static async load(device, read) {
    const net = new ProteinEbmGpu(device);
    await net.#loadWeights(read);
    return net;
  }

  async #loadWeights(read) {
    const ops = this.ops;
    const t = async (name) => read(name);
    const up = async (label, values) => ops.weight(label, values);
    const host = {};
    this.host = host;
    // --- the residue embedding and the single conditioner
    host.seq = await t("sequence_embedding/weight");                  // (21, 256)
    host.contact = await t("contact_embedding/weight");               // (2, 256)
    this.wNoisy = await up("noisy_coord", await t("noisy_coord_embedding/weight"));      // (256, 3)
    this.wSelf = await up("selfcond", await t("self_conditioning_embedding/weight"));    // (256, 3)
    const sc = "single_conditioner/";
    this.normSingle = { w: await up("norm_single.w", await t(`${sc}norm_single/weight`)),
                        b: await up("norm_single.b", await t(`${sc}norm_single/bias`)) };
    this.singleEmbed = await up("single_embed", await t(`${sc}single_embed/weight`));     // (512, 1024)
    host.singleEmbedBias = await t(`${sc}single_embed/bias`);
    host.fourier = { w: await t(`${sc}fourier_embed/proj/weight`), b: await t(`${sc}fourier_embed/proj/bias`),
                     nw: await t(`${sc}norm_fourier/weight`), nb: await t(`${sc}norm_fourier/bias`),
                     out: await t(`${sc}fourier_to_single/weight`) };
    this.singleBias = ops.buffer("single_embed.bias", D);
    this.singleTransitions = [];
    for (let k = 0; k < 2; k += 1) this.singleTransitions.push(await this.#transition(read, `${sc}transitions/${k}/`, D, 2 * D));
    // --- the pair conditioner: the init LayerNorm's affine folded into per-part projections
    const pc = "pairwise_conditioner/";
    const wp = await t(`${pc}dim_pairwise_init_proj/1/weight`);       // (128, 1152)
    const lw = await t(`${pc}dim_pairwise_init_proj/0/weight`);
    const lb = await t(`${pc}dim_pairwise_init_proj/0/bias`);
    const F = 2 * D + Z;
    const part = (from, width) => {
      const out = new Float32Array(Z * width);
      for (let o = 0; o < Z; o += 1) for (let f = 0; f < width; f += 1) out[o * width + f] = wp[o * F + from + f] * lw[from + f];
      return out;
    };
    const wa = part(0, D), wb = part(D, D), wc = part(2 * D, Z);
    const u = new Float32Array(Z), c = new Float32Array(Z);
    for (let o = 0; o < Z; o += 1) {
      let su = 0, sc2 = 0;
      for (let f = 0; f < F; f += 1) { su += wp[o * F + f] * lw[f]; sc2 += wp[o * F + f] * lb[f]; }
      u[o] = su; c[o] = sc2;
    }
    this.pairA = await up("pair.wa", wa);
    this.pairB = await up("pair.wb", wb);
    this.pairU = await up("pair.u", u);
    this.pairC = await up("pair.c", c);
    // the relative-position table: class (d_residue 0..65, d_chain 0..1) -> its 128 features, their projection
    // through W_c', and their sum and sum of squares (for the 1152-wide statistics)
    const wrel = await t("rel_pos/linear_layer/weight");               // (128, 68)
    const classes = 66 * 2;
    const tc = new Float32Array(classes * Z), r12 = new Float32Array(classes * 2);
    for (let cls = 0; cls < classes; cls += 1) {
      const dres = cls >> 1, dchain = cls & 1;
      const feat = new Float64Array(Z);
      for (let o = 0; o < Z; o += 1) feat[o] = wrel[o * 68 + dres] + wrel[o * 68 + 66 + dchain];
      let s1 = 0, s2 = 0;
      for (let o = 0; o < Z; o += 1) { s1 += feat[o]; s2 += feat[o] * feat[o]; }
      r12[2 * cls] = s1; r12[2 * cls + 1] = s2;
      for (let o = 0; o < Z; o += 1) {
        let v = 0;
        for (let f = 0; f < Z; f += 1) v += wc[o * Z + f] * feat[f];
        tc[cls * Z + o] = v;
      }
    }
    this.relTable = await up("pair.reltable", tc);
    this.relStats = await up("pair.relstats", r12);
    this.pairTransitions = [];
    for (let k = 0; k < 2; k += 1) this.pairTransitions.push(await this.#transition(read, `${pc}transitions/${k}/`, Z, 2 * Z));
    // --- the token transformer
    this.blocks = [];
    for (let l = 0; l < this.layers; l += 1) {
      const p = `token_transformer/layers/${l}/`;
      const adaln = async (q) => {
        const w = await t(`${q}s_norm/weight`);
        const fold = async (name) => {
          const m = await t(name);
          const out = new Float32Array(m.length);
          for (let o = 0; o < D; o += 1) for (let i = 0; i < D; i += 1) out[o * D + i] = m[o * D + i] * w[i];
          return up(name, out);
        };
        return { scale: await fold(`${q}s_scale/weight`), scaleBias: await up(`${q}s_scale/bias`, await t(`${q}s_scale/bias`)),
                 shift: await fold(`${q}s_bias/weight`) };
      };
      const a = `${p}pair_bias_attn/`;
      const wz = await t(`${a}proj_z/1/weight`);                       // (8, 128)
      const zw = await t(`${a}proj_z/0/weight`);
      const wzw = new Float32Array(H * Z);
      for (let h = 0; h < H; h += 1) for (let k = 0; k < Z; k += 1) wzw[h * Z + k] = wz[h * Z + k] * zw[k];
      const tr = `${p}transition/`;
      this.blocks.push({
        ada: await adaln(`${p}adaln/`),
        q: await up(`${a}proj_q/weight`, await t(`${a}proj_q/weight`)),
        qBias: await up(`${a}proj_q/bias`, await t(`${a}proj_q/bias`)),
        k: await up(`${a}proj_k/weight`, await t(`${a}proj_k/weight`)),
        v: await up(`${a}proj_v/weight`, await t(`${a}proj_v/weight`)),
        g: await up(`${a}proj_g/weight`, await t(`${a}proj_g/weight`)),
        o: await up(`${a}proj_o/weight`, await t(`${a}proj_o/weight`)),
        zw: await up(`${a}proj_z.folded`, wzw),
        gateO: await up(`${p}output_projection_linear/weight`, await t(`${p}output_projection_linear/weight`)),
        gateOBias: await up(`${p}output_projection_linear/bias`, await t(`${p}output_projection_linear/bias`)),
        tAda: await adaln(`${tr}adaln/`),
        swish: await up(`${tr}swish_gate/0/weight`, await t(`${tr}swish_gate/0/weight`)),     // (2048, 512)
        a2b: await up(`${tr}a_to_b/weight`, await t(`${tr}a_to_b/weight`)),                   // (1024, 512)
        b2a: await up(`${tr}b_to_a/weight`, await t(`${tr}b_to_a/weight`)),                   // (512, 1024)
        gateT: await up(`${tr}output_projection/0/weight`, await t(`${tr}output_projection/0/weight`)),
        gateTBias: await up(`${tr}output_projection/0/bias`, await t(`${tr}output_projection/0/bias`)),
      });
    }
    this.aNorm = { w: await up("a_norm.w", await t("a_norm/weight")), b: await up("a_norm.b", await t("a_norm/bias")) };
    this.rProj = await up("r_update_proj", await t("r_update_proj/weight"));                  // (3, 512)
  }

  async #transition(read, p, width, hidden) {
    const ops = this.ops;
    return { width, hidden,
             nw: await ops.weight(`${p}norm.w`, await read(`${p}norm/weight`)),
             nb: await ops.weight(`${p}norm.b`, await read(`${p}norm/bias`)),
             fc1: await ops.weight(`${p}fc1`, await read(`${p}fc1/weight`)),
             fc2: await ops.weight(`${p}fc2`, await read(`${p}fc2/weight`)),
             fc3: await ops.weight(`${p}fc3`, await read(`${p}fc3/weight`)) };
  }

  // --- recording helpers -------------------------------------------------------------------------------------

  /** y (+)= x W^T, W torch (out, in). */
  #lin(x, w, y, rows, inC, outC, extra = {}) {
    return this.ops.linear({ x, w, y, rows, inC, outC, ...extra });
  }

  /** dx (+)= dy W, W torch (out, in): the input gradient of #lin. */
  #linT(dy, w, dx, rows, outC, inC, accumulate = false, alpha) {
    return this.ops.gemm({ m: rows, n: inC, k: outC, a: dy, b: w, c: dx, accumulate, alpha,
                           A: { m: outC, k: 1 }, B: { n: 1, k: inC }, C: { m: inC, n: 1 } });
  }

  #ln(x, stats, y, rows, width, affine = null) {
    const kind = !affine ? "none" : affine.b ? "wb" : "w";
    const u = new Uint32Array(4);
    u[0] = rows; u[1] = width; new Float32Array(u.buffer)[2] = 1e-5;
    return this.ops.step(`pebm-ln-${kind}${y ? "" : "-stats"}`, lnForwardSource(kind, !y), rows,
                         [x, stats, y ?? null, affine?.w ?? null, affine?.b ?? null], u);
  }

  #lnBack(x, stats, dy, dx, rows, width, w = null, accumulate = false) {
    const u = new Uint32Array([rows, width, 0, 0]);
    return this.ops.step(`pebm-lnb-${w ? 1 : 0}-${accumulate ? 1 : 0}`, lnBackwardSource(Boolean(w), accumulate), rows,
                         [x, stats, dy, dx, w], u);
  }

  #map(name, count, bindings, body, uints = [], floats = []) {
    return this.ops.map(`pebm-${name}`, count, bindings, body, uints, floats);
  }

  // --- recording ---------------------------------------------------------------------------------------------

  /**
   * Record the forward and backward for one chain of N residues. `aatype` in AlphaFold's order (20 = unknown),
   * `residueIdx` the residue numbering (only differences matter), `chain` per residue (default one chain).
   */
  async prepare({ aatype, residueIdx, chain = null }) {
    const ops = this.ops;
    if (this.prepared) ops.destroy();
    this.trScratch = new Map();
    const N = aatype.length;
    const NN = N * N;
    this.N = N;
    const buf = (label, floats) => ops.buffer(label, floats);
    const B = {};
    this.B = B;
    // inputs
    B.r = buf("r", N * 3);
    B.sc = buf("sc", N * 3);
    // the constant half of X0: the sequence and contact embeddings, written once
    const e0 = new Float32Array(N * X0W);
    for (let i = 0; i < N; i += 1) {
      const aa = Math.min(Math.max(aatype[i] | 0, 0), 20);
      e0.set(this.host.seq.subarray(aa * 256, aa * 256 + 256), i * X0W);
      e0.set(this.host.contact.subarray(0, 256), i * X0W + 512);
    }
    B.e0 = buf("e0", N * X0W);
    ops.write(B.e0, e0);
    const cls = new Uint32Array(NN);
    for (let i = 0; i < N; i += 1) for (let j = 0; j < N; j += 1) {
      cls[i * N + j] = relposClass(residueIdx[i], residueIdx[j], chain ? chain[i] : 0, chain ? chain[j] : 0);
    }
    B.cls = buf("cls", NN);
    ops.write(B.cls, cls);

    const fwd = [];
    const bwd = [];
    const F = (s) => fwd.push(s);
    const R = (s) => bwd.push(s);

    // ===== forward: the single conditioner
    B.x0 = buf("x0", N * X0W);
    F(await this.#map("x0", N * X0W, [{ buffer: B.x0, write: true }, { buffer: B.e0 }, { buffer: B.r }, { buffer: B.sc },
                                       { buffer: this.wNoisy }, { buffer: this.wSelf }],
      `let row = i / 1024u; let f = i % 1024u; var v = b1[i];
  if (f >= 256u && f < 512u) { let o = f - 256u; v = 0.1 * (b4[o * 3u] * b2[row * 3u] + b4[o * 3u + 1u] * b2[row * 3u + 1u] + b4[o * 3u + 2u] * b2[row * 3u + 2u]); }
  if (f >= 768u) { let o = f - 768u; v = 0.1 * (b5[o * 3u] * b3[row * 3u] + b5[o * 3u + 1u] * b3[row * 3u + 1u] + b5[o * 3u + 2u] * b3[row * 3u + 2u]); }
  b0[i] = v;`));
    B.x0Stats = buf("x0.stats", 2 * N);
    B.x1 = buf("x1", N * X0W);
    F(await this.#ln(B.x0, B.x0Stats, B.x1, N, X0W, this.normSingle));
    B.s0 = buf("s0", N * D);
    F(await this.#lin(B.x1, this.singleEmbed, B.s0, N, X0W, D, { bias: this.singleBias }));
    const singleSaves = [];
    let s = B.s0;
    for (let k = 0; k < 2; k += 1) {
      const out = buf(`s${k + 1}`, N * D);
      const save = await this.#transitionForward(F, this.singleTransitions[k], s, out, N, `single${k}`);
      singleSaves.push(save);
      s = out;
    }
    B.s = s;

    // ===== the pair conditioner
    B.pa = buf("pair.A", N * Z);
    B.pb = buf("pair.B", N * Z);
    F(await this.#lin(B.s, this.pairA, B.pa, N, D, Z));
    F(await this.#lin(B.s, this.pairB, B.pb, N, D, Z));
    B.sSums = buf("s.sums", 2 * N);
    F(await this.#map("rowsums", N, [{ buffer: B.sSums, write: true }, { buffer: B.s }],
      `var a = 0.0; var q = 0.0; for (var c = 0u; c < ${D}u; c += 1u) { let v = b1[i * ${D}u + c]; a += v; q += v * v; }
  b0[2u * i] = a; b0[2u * i + 1u] = q;`));
    B.pairStats = buf("pair.stats", 2 * NN);
    F(await this.#map("pairstats", NN, [{ buffer: B.pairStats, write: true }, { buffer: B.sSums }, { buffer: B.cls, type: "u32" },
                                        { buffer: this.relStats }],
      `let n = U(0u); let a = i / n; let b = i % n; let k = b2[i];
  let mean = (b1[2u * a] + b1[2u * b] + b3[2u * k]) / ${2 * D + Z}.0;
  let sq = (b1[2u * a + 1u] + b1[2u * b + 1u] + b3[2u * k + 1u]) / ${2 * D + Z}.0;
  b0[2u * i] = mean; b0[2u * i + 1u] = inverseSqrt(max(sq - mean * mean, 0.0) + 1e-5);`, [N]));
    B.z0 = buf("z0", NN * Z);
    F(await this.#map("pairinit", NN * Z, [{ buffer: B.z0, write: true }, { buffer: B.pa }, { buffer: B.pb },
                                           { buffer: this.relTable }, { buffer: B.cls, type: "u32" }, { buffer: B.pairStats },
                                           { buffer: this.pairU }, { buffer: this.pairC }],
      `let n = U(0u); let pair = i / ${Z}u; let o = i % ${Z}u; let a = pair / n; let b = pair % n; let k = b4[pair];
  let mean = b5[2u * pair]; let rstd = b5[2u * pair + 1u];
  b0[i] = rstd * (b1[a * ${Z}u + o] + b2[b * ${Z}u + o] + b3[k * ${Z}u + o] - mean * b6[o]) + b7[o];`, [N]));
    const pairSaves = [];
    let z = B.z0;
    for (let k = 0; k < 2; k += 1) {
      const out = buf(`z${k + 1}`, NN * Z);
      pairSaves.push(await this.#transitionForward(F, this.pairTransitions[k], z, out, NN, `pair${k}`));
      z = out;
    }
    B.z = z;
    B.zStats = buf("z.stats", 2 * NN);
    B.zhat = buf("zhat", NN * Z);
    F(await this.#ln(B.z, B.zStats, B.zhat, NN, Z));

    // ===== the token transformer
    B.sStats = buf("s.stats", 2 * N);
    B.shat = buf("shat", N * D);
    F(await this.#ln(B.s, B.sStats, B.shat, N, D));
    const saves = [];
    let a = B.s;
    for (let l = 0; l < this.layers; l += 1) {
      const out = buf(`a${l + 1}`, N * D);
      saves.push(await this.#blockForward(F, this.blocks[l], a, out, N, l));
      a = out;
    }
    B.aLast = a;
    B.afStats = buf("af.stats", 2 * N);
    B.af = buf("af", N * D);
    F(await this.#ln(a, B.afStats, B.af, N, D, this.aNorm));
    B.rUpdate = buf("r_update", N * 3);
    F(await this.#lin(B.af, this.rProj, B.rUpdate, N, D, 3));

    // ===== backward
    B.drUpdate = buf("d.r_update", N * 3);
    R(await this.#map("seed", N * 3, [{ buffer: B.drUpdate, write: true }, { buffer: B.rUpdate }], "b0[i] = 2.0 * b1[i];"));
    B.daf = buf("d.af", N * D);
    R(await this.#linT(B.drUpdate, this.rProj, B.daf, N, 3, D));
    let da = buf("d.a", N * D);
    R(await this.#lnBack(B.aLast, B.afStats, B.daf, da, N, D, this.aNorm.w));
    B.dshat = buf("d.shat", N * D);
    B.dsDirect = buf("d.s.direct", N * D);
    B.dzhat = buf("d.zhat", NN * Z);
    R(await this.#map("zero3", NN * Z, [{ buffer: B.dshat, write: true }, { buffer: B.dsDirect, write: true }, { buffer: B.dzhat, write: true }],
      `if (i < U(0u)) { b0[i] = 0.0; b1[i] = 0.0; } b2[i] = 0.0;`, [N * D]));
    // scratch shared by every layer's backward (the layers run in order)
    const scratch = {
      dmid: buf("d.mid", N * D), da2: buf("d.a2", N * D), dpre: buf("d.pre", N * D), dy: buf("d.y", N * D),
      du: buf("d.u", N * 2 * D), dG: buf("d.G", N * 4 * D), dA2B: buf("d.a2b", N * 2 * D), dc: buf("d.c", N * D),
      dxn: buf("d.xn", N * D), dgo: buf("d.go", N * D), dgpre: buf("d.gpre", N * D), dO: buf("d.o", N * D),
      dP: buf("d.P", H * NN), dq: buf("d.q", N * D), dk: buf("d.k", N * D), dv: buf("d.v", N * D), db: buf("d.b", N * D),
    };
    for (let l = this.layers - 1; l >= 0; l -= 1) {
      const next = buf(`d.a${l}`, N * D);
      await this.#blockBackward(R, this.blocks[l], saves[l], da, next, scratch, N, l);
      da = next;
    }
    // ds = da (a0 = s) + the gates' direct gradient + LN(s)'s backward of everything that reached shat
    B.ds = buf("d.s", N * D);
    R(await this.#map("add2", N * D, [{ buffer: B.ds, write: true }, { buffer: da }, { buffer: B.dsDirect }], "b0[i] = b1[i] + b2[i];"));
    R(await this.#lnBack(B.s, B.sStats, B.dshat, B.ds, N, D, null, true));
    // the pair track: zhat's gradient -> z -> through the transitions -> the init LayerNorm -> s
    B.dz = buf("d.z", NN * Z);
    R(await this.#lnBack(B.z, B.zStats, B.dzhat, B.dz, NN, Z));
    // (the transitions' backward runs in place, so it runs on a copy and dE/dz stays readable)
    const dz = buf("d.z.init", NN * Z);
    R(await this.#map("copy", NN * Z, [{ buffer: dz, write: true }, { buffer: B.dz }], "b0[i] = b1[i];"));
    for (let k = 1; k >= 0; k -= 1) {
      await this.#transitionBackward(R, this.pairTransitions[k], pairSaves[k], dz, NN, `pair${k}`);
    }
    B.dzInit = dz;
    await this.#pairInitBackward(R, N);
    // the single conditioner
    const ds = buf("d.s.cond", N * D);
    R(await this.#map("copy", N * D, [{ buffer: ds, write: true }, { buffer: B.ds }], "b0[i] = b1[i];"));
    for (let k = 1; k >= 0; k -= 1) await this.#transitionBackward(R, this.singleTransitions[k], singleSaves[k], ds, N, `single${k}`);
    B.dx1 = buf("d.x1", N * X0W);
    R(await this.#linT(ds, this.singleEmbed, B.dx1, N, D, X0W));
    B.dx0 = buf("d.x0", N * X0W);
    R(await this.#lnBack(B.x0, B.x0Stats, B.dx1, B.dx0, N, X0W, this.normSingle.w));
    // dE/dr = 0.1 (dX0[:, 256:512] W_n); the score is its negative
    B.score = buf("score", N * 3);
    R(await this.#map("score", N * 3, [{ buffer: B.score, write: true }, { buffer: B.dx0 }, { buffer: this.wNoisy }],
      `let row = i / 3u; let c = i % 3u; var v = 0.0;
  for (var o = 0u; o < 256u; o += 1u) { v += b1[row * 1024u + 256u + o] * b2[o * 3u + c]; }
  b0[i] = -0.1 * v;`));

    this.forward = ops.program(fwd);
    this.backward = ops.program(bwd);
    this.prepared = true;
  }

  /** Transition (layers.py): x + fc3(silu(fc1 LN(x)) * fc2 LN(x)), LN affine. */
  async #transitionForward(F, tr, x, out, rows, tag) {
    const ops = this.ops;
    const save = { x, stats: ops.buffer(`${tag}.stats`, 2 * rows), xn: ops.buffer(`${tag}.xn`, rows * tr.width),
                   h1: ops.buffer(`${tag}.h1`, rows * tr.hidden), h2: ops.buffer(`${tag}.h2`, rows * tr.hidden),
                   m: ops.buffer(`${tag}.m`, rows * tr.hidden) };
    F(await this.#ln(x, save.stats, save.xn, rows, tr.width, { w: tr.nw, b: tr.nb }));
    F(await this.#lin(save.xn, tr.fc1, save.h1, rows, tr.width, tr.hidden));
    F(await this.#lin(save.xn, tr.fc2, save.h2, rows, tr.width, tr.hidden));
    F(await this.#map("silumul", rows * tr.hidden, [{ buffer: save.m, write: true }, { buffer: save.h1 }, { buffer: save.h2 }],
      "let a = b1[i]; b0[i] = a / (1.0 + exp(-a)) * b2[i];"));
    F(await this.#map("copy", rows * tr.width, [{ buffer: out, write: true }, { buffer: x }], "b0[i] = b1[i];"));
    F(await this.#lin(save.m, tr.fc3, out, rows, tr.hidden, tr.width, { accumulate: true }));
    return save;
  }

  /** In place on dout (the gradient of the transition's output, which is also its residual's): dout += dT/dx. */
  async #transitionBackward(R, tr, save, dout, rows, tag) {
    const ops = this.ops;
    const key = `${rows}x${tr.width}x${tr.hidden}`;
    const want = rows * tr.hidden;
    if (!this.trScratch.has(key)) {
      this.trScratch.set(key, { dm: ops.buffer(`${tag}.dm`, want), dh1: ops.buffer(`${tag}.dh1`, want),
                                dh2: ops.buffer(`${tag}.dh2`, want), dxn: ops.buffer(`${tag}.dxn`, rows * tr.width) });
    }
    const s = this.trScratch.get(key);
    R(await this.#linT(dout, tr.fc3, s.dm, rows, tr.width, tr.hidden));
    R(await this.#map("silumul-back", want, [{ buffer: s.dh1, write: true }, { buffer: s.dh2, write: true }, { buffer: s.dm },
                                             { buffer: save.h1 }, { buffer: save.h2 }],
      `let a = b3[i]; let sg = 1.0 / (1.0 + exp(-a)); let silu = a * sg;
  b0[i] = b2[i] * b4[i] * sg * (1.0 + a * (1.0 - sg)); b1[i] = b2[i] * silu;`));
    R(await this.#linT(s.dh1, tr.fc1, s.dxn, rows, tr.hidden, tr.width));
    R(await this.#linT(s.dh2, tr.fc2, s.dxn, rows, tr.hidden, tr.width, true));
    R(await this.#lnBack(save.x, save.stats, s.dxn, dout, rows, tr.width, tr.nw, true));
  }

  /** One DiffusionTransformerLayer; `out` = a + attention + transition. */
  async #blockForward(F, blk, a, out, N, l) {
    const ops = this.ops;
    const B = this.B;
    const NN = N * N;
    const tag = `L${l}`;
    const buf = (name, n) => ops.buffer(`${tag}.${name}`, n);
    const sv = {
      a, sig: buf("sig", N * D), shift: buf("shift", N * D), xnStats: buf("xn.stats", 2 * N), xn: buf("xn", N * D),
      b: buf("b", N * D), q: buf("q", N * D), k: buf("k", N * D), v: buf("v", N * D), g: buf("g", N * D),
      P: buf("P", H * NN), o: buf("o", N * D), go: buf("go", N * D), o2: buf("o2", N * D), gateO: buf("gateO", N * D),
      mid: buf("mid", N * D),
      tsig: buf("tsig", N * D), tshift: buf("tshift", N * D), x2Stats: buf("x2.stats", 2 * N), x2n: buf("x2n", N * D),
      c: buf("c", N * D), G: buf("G", N * 4 * D), A2B: buf("A2B", N * 2 * D), u: buf("u", N * 2 * D),
      y: buf("y", N * D), gateT: buf("gateT", N * D),
    };
    // the conditioning (s alone): AdaLN's scale and shift for both AdaLNs, and both output gates
    F(await this.#lin(B.shat, blk.ada.scale, sv.sig, N, D, D, { bias: blk.ada.scaleBias, act: "sigmoid" }));
    F(await this.#lin(B.shat, blk.ada.shift, sv.shift, N, D, D));
    F(await this.#lin(B.shat, blk.tAda.scale, sv.tsig, N, D, D, { bias: blk.tAda.scaleBias, act: "sigmoid" }));
    F(await this.#lin(B.shat, blk.tAda.shift, sv.tshift, N, D, D));
    F(await this.#lin(B.s, blk.gateO, sv.gateO, N, D, D, { bias: blk.gateOBias, act: "sigmoid" }));
    F(await this.#lin(B.s, blk.gateT, sv.gateT, N, D, D, { bias: blk.gateTBias, act: "sigmoid" }));
    // attention
    F(await this.#ln(a, sv.xnStats, sv.xn, N, D));
    F(await this.#map("ada", N * D, [{ buffer: sv.b, write: true }, { buffer: sv.xn }, { buffer: sv.sig }, { buffer: sv.shift }],
      "b0[i] = b1[i] * b2[i] + b3[i];"));
    F(await this.#lin(sv.b, blk.q, sv.q, N, D, D, { bias: blk.qBias }));
    F(await this.#lin(sv.b, blk.k, sv.k, N, D, D));
    F(await this.#lin(sv.b, blk.v, sv.v, N, D, D));
    F(await this.#lin(sv.b, blk.g, sv.g, N, D, D, { act: "sigmoid" }));
    F(await ops.gemm({ m: N, n: N, k: HD, a: sv.q, b: sv.k, c: sv.P, alpha: 1 / 8, batches: [H, 1],
                       A: { s0: HD, m: D, k: 1 }, B: { s0: HD, n: D, k: 1 }, C: { s0: NN, m: N, n: 1 } }));
    F(await ops.gemm({ m: NN, n: H, k: Z, a: B.zhat, b: blk.zw, c: sv.P, accumulate: true,
                       A: { m: Z, k: 1 }, B: { n: Z, k: 1 }, C: { m: 1, n: NN } }));
    F(await ops.step("pebm-softmax", SOFTMAX_SOURCE, H * N, [sv.P], new Uint32Array([H * N, N, 0, 0])));
    F(await ops.gemm({ m: N, n: HD, k: N, a: sv.P, b: sv.v, c: sv.o, batches: [H, 1],
                       A: { s0: NN, m: N, k: 1 }, B: { s0: HD, n: 1, k: D }, C: { s0: HD, m: D, n: 1 } }));
    F(await this.#map("mul", N * D, [{ buffer: sv.go, write: true }, { buffer: sv.g }, { buffer: sv.o }], "b0[i] = b1[i] * b2[i];"));
    F(await this.#lin(sv.go, blk.o, sv.o2, N, D, D));
    F(await this.#map("gated-add", N * D, [{ buffer: sv.mid, write: true }, { buffer: a }, { buffer: sv.gateO }, { buffer: sv.o2 }],
      "b0[i] = b1[i] + b2[i] * b3[i];"));
    // the conditioned transition
    F(await this.#ln(sv.mid, sv.x2Stats, sv.x2n, N, D));
    F(await this.#map("ada", N * D, [{ buffer: sv.c, write: true }, { buffer: sv.x2n }, { buffer: sv.tsig }, { buffer: sv.tshift }],
      "b0[i] = b1[i] * b2[i] + b3[i];"));
    F(await this.#lin(sv.c, blk.swish, sv.G, N, D, 4 * D));
    F(await this.#lin(sv.c, blk.a2b, sv.A2B, N, D, 2 * D));
    F(await this.#map("swiglu", N * 2 * D, [{ buffer: sv.u, write: true }, { buffer: sv.G }, { buffer: sv.A2B }],
      `let r = i / ${2 * D}u; let c = i % ${2 * D}u; let x = b1[r * ${4 * D}u + c]; let gt = b1[r * ${4 * D}u + ${2 * D}u + c];
  b0[i] = x * gt / (1.0 + exp(-gt)) * b2[i];`));
    F(await this.#lin(sv.u, blk.b2a, sv.y, N, 2 * D, D));
    F(await this.#map("gated-add", N * D, [{ buffer: out, write: true }, { buffer: sv.mid }, { buffer: sv.gateT }, { buffer: sv.y }],
      "b0[i] = b1[i] + b2[i] * b3[i];"));
    return sv;
  }

  /** The block's input gradient: `dnext` = dE/d(a_in) from dE/d(a_out) = `da`; s's and zhat's into the accumulators. */
  async #blockBackward(R, blk, sv, da, dnext, S, N, l) {
    const ops = this.ops;
    const B = this.B;
    const NN = N * N;
    // --- transition: out = mid + gateT * y
    R(await this.#map("gate-back", N * D, [{ buffer: S.dpre, write: true }, { buffer: S.dy, write: true }, { buffer: da },
                                           { buffer: sv.gateT }, { buffer: sv.y }],
      "let g = b3[i]; b0[i] = b2[i] * b4[i] * g * (1.0 - g); b1[i] = b2[i] * g;"));
    R(await this.#linT(S.dpre, blk.gateT, B.dsDirect, N, D, D, true));
    R(await this.#linT(S.dy, blk.b2a, S.du, N, D, 2 * D));
    R(await this.#map("swiglu-back", N * 2 * D, [{ buffer: S.dG, write: true }, { buffer: S.dA2B, write: true }, { buffer: S.du },
                                                 { buffer: sv.G }, { buffer: sv.A2B }],
      `let r = i / ${2 * D}u; let c = i % ${2 * D}u; let x = b3[r * ${4 * D}u + c]; let gt = b3[r * ${4 * D}u + ${2 * D}u + c];
  let sg = 1.0 / (1.0 + exp(-gt)); let silu = gt * sg; let ab = b4[i]; let du = b2[i];
  b0[r * ${4 * D}u + c] = du * silu * ab;
  b0[r * ${4 * D}u + ${2 * D}u + c] = du * x * ab * sg * (1.0 + gt * (1.0 - sg));
  b1[i] = du * x * silu;`));
    R(await this.#linT(S.dG, blk.swish, S.dc, N, 4 * D, D));
    R(await this.#linT(S.dA2B, blk.a2b, S.dc, N, 2 * D, D, true));
    await this.#adaBackward(R, blk.tAda, S.dc, sv.x2n, sv.tsig, S, N);
    // d(mid) = da + LN backward of dxn
    R(await this.#map("copy", N * D, [{ buffer: S.dmid, write: true }, { buffer: da }], "b0[i] = b1[i];"));
    R(await this.#lnBack(sv.mid, sv.x2Stats, S.dxn, S.dmid, N, D, null, true));
    // --- attention: mid = a + gateO * o2
    R(await this.#map("gate-back", N * D, [{ buffer: S.dpre, write: true }, { buffer: S.dy, write: true }, { buffer: S.dmid },
                                           { buffer: sv.gateO }, { buffer: sv.o2 }],
      "let g = b3[i]; b0[i] = b2[i] * b4[i] * g * (1.0 - g); b1[i] = b2[i] * g;"));
    R(await this.#linT(S.dpre, blk.gateO, B.dsDirect, N, D, D, true));
    R(await this.#linT(S.dy, blk.o, S.dgo, N, D, D));
    R(await this.#map("gmul-back", N * D, [{ buffer: S.dgpre, write: true }, { buffer: S.dO, write: true }, { buffer: S.dgo },
                                           { buffer: sv.g }, { buffer: sv.o }],
      "let g = b3[i]; b0[i] = b2[i] * b4[i] * g * (1.0 - g); b1[i] = b2[i] * g;"));
    R(await ops.gemm({ m: N, n: N, k: HD, a: S.dO, b: sv.v, c: S.dP, batches: [H, 1],
                       A: { s0: HD, m: D, k: 1 }, B: { s0: HD, n: D, k: 1 }, C: { s0: NN, m: N, n: 1 } }));
    R(await ops.step("pebm-softmax-back", SOFTMAX_BACKWARD_SOURCE, H * N, [sv.P, S.dP], new Uint32Array([H * N, N, 0, 0])));
    R(await ops.gemm({ m: N, n: HD, k: N, a: S.dP, b: sv.k, c: S.dq, alpha: 1 / 8, batches: [H, 1],
                       A: { s0: NN, m: N, k: 1 }, B: { s0: HD, n: 1, k: D }, C: { s0: HD, m: D, n: 1 } }));
    R(await ops.gemm({ m: N, n: HD, k: N, a: S.dP, b: sv.q, c: S.dk, alpha: 1 / 8, batches: [H, 1],
                       A: { s0: NN, m: 1, k: N }, B: { s0: HD, n: 1, k: D }, C: { s0: HD, m: D, n: 1 } }));
    R(await ops.gemm({ m: N, n: HD, k: N, a: sv.P, b: S.dO, c: S.dv, batches: [H, 1],
                       A: { s0: NN, m: 1, k: N }, B: { s0: HD, n: 1, k: D }, C: { s0: HD, m: D, n: 1 } }));
    R(await ops.gemm({ m: NN, n: Z, k: H, a: S.dP, b: blk.zw, c: B.dzhat, accumulate: true,
                       A: { m: 1, k: NN }, B: { n: 1, k: Z }, C: { m: Z, n: 1 } }));
    R(await this.#linT(S.dq, blk.q, S.db, N, D, D));
    R(await this.#linT(S.dk, blk.k, S.db, N, D, D, true));
    R(await this.#linT(S.dv, blk.v, S.db, N, D, D, true));
    R(await this.#linT(S.dgpre, blk.g, S.db, N, D, D, true));
    await this.#adaBackward(R, blk.ada, S.db, sv.xn, sv.sig, S, N);
    R(await this.#map("copy", N * D, [{ buffer: dnext, write: true }, { buffer: S.dmid }], "b0[i] = b1[i];"));
    R(await this.#lnBack(sv.a, sv.xnStats, S.dxn, dnext, N, D, null, true));
  }

  /** AdaLN's backward from d(out): S.dxn = d(xn), and shat's gradient accumulated through the scale and the shift. */
  async #adaBackward(R, ada, dout, xn, sig, S, N) {
    R(await this.#map("ada-back", N * D, [{ buffer: S.dxn, write: true }, { buffer: S.da2, write: true }, { buffer: dout },
                                          { buffer: xn }, { buffer: sig }],
      "let g = b4[i]; b0[i] = b2[i] * g; b1[i] = b2[i] * b3[i] * g * (1.0 - g);"));
    R(await this.#linT(S.da2, ada.scale, this.B.dshat, N, D, D, true));
    R(await this.#linT(dout, ada.shift, this.B.dshat, N, D, D, true));
  }

  /**
   * The pair init LayerNorm's backward to s, without the 1152-wide tensor. With g = dz0_ij, m1 = g.u / 1152,
   * m2 = g.(z0_ij - c) / 1152 (Wp' xhat_ij = z0_ij - c), xhat_ij[s_i part] = (s_i - mean_ij) rstd_ij:
   *   ds_i += sum_j rstd g Wa' - rstd m1 - rstd^2 m2 (s_i - mean)       (i first)
   *         + sum_j rstd_ji g_ji Wb' - ...                              (i second)
   */
  async #pairInitBackward(R, N) {
    const ops = this.ops;
    const B = this.B;
    const NN = N * N;
    const F1152 = 2 * D + Z;
    B.pairScal = ops.buffer("pair.scalars", 2 * NN);
    R(await this.#map("pair-scalars", NN, [{ buffer: B.pairScal, write: true }, { buffer: B.dzInit }, { buffer: B.z0 },
                                           { buffer: this.pairU }, { buffer: this.pairC }],
      `var m1 = 0.0; var m2 = 0.0;
  for (var o = 0u; o < ${Z}u; o += 1u) { let g = b1[i * ${Z}u + o]; m1 += g * b3[o]; m2 += g * (b2[i * ${Z}u + o] - b4[o]); }
  b0[2u * i] = m1 / ${F1152}.0; b0[2u * i + 1u] = m2 / ${F1152}.0;`));
    // G (first index) and Gb (second index): rstd-weighted sums of the pair gradient
    B.pairG = ops.buffer("pair.G", N * Z);
    B.pairGb = ops.buffer("pair.Gb", N * Z);
    R(await this.#map("pair-G", N * Z, [{ buffer: B.pairG, write: true }, { buffer: B.pairGb, write: true }, { buffer: B.dzInit },
                                        { buffer: B.pairStats }],
      `let n = U(0u); let a = i / ${Z}u; let o = i % ${Z}u; var g = 0.0; var gb = 0.0;
  for (var j = 0u; j < n; j += 1u) {
    let p1 = a * n + j; let p2 = j * n + a;
    g += b3[2u * p1 + 1u] * b2[p1 * ${Z}u + o];
    gb += b3[2u * p2 + 1u] * b2[p2 * ${Z}u + o];
  }
  b0[i] = g; b1[i] = gb;`, [N]));
    // the per-residue scalars: (alpha - gamma) and beta, both sides summed
    B.pairAB = ops.buffer("pair.ab", 2 * N);
    R(await this.#map("pair-ab", N, [{ buffer: B.pairAB, write: true }, { buffer: B.pairScal }, { buffer: B.pairStats }],
      `let n = U(0u); var ag = 0.0; var be = 0.0;
  for (var j = 0u; j < n; j += 1u) {
    for (var side = 0u; side < 2u; side += 1u) {
      let p = select(j * n + i, i * n + j, side == 0u);
      let mean = b2[2u * p]; let rstd = b2[2u * p + 1u];
      let m1 = b1[2u * p]; let m2 = b1[2u * p + 1u];
      ag += rstd * m1 - rstd * rstd * m2 * mean;
      be += rstd * rstd * m2;
    }
  }
  b0[2u * i] = ag; b0[2u * i + 1u] = be;`, [N]));
    R(await this.#map("pair-ds", N * D, [{ buffer: B.ds, write: true }, { buffer: B.pairAB }, { buffer: B.s }],
      `let r = i / ${D}u; b0[i] = b0[i] - b1[2u * r] - b1[2u * r + 1u] * b2[i];`));
    R(await this.#linT(B.pairG, this.pairA, B.ds, N, Z, D, true));
    R(await this.#linT(B.pairGb, this.pairB, B.ds, N, Z, D, true));
  }

  // --- running -----------------------------------------------------------------------------------------------

  /** Set the time level: the Fourier row folded into single_embed's bias. */
  setTime(t) {
    if (this.t === t) return;
    const row = fourierRow(this.host, t);
    const bias = new Float32Array(D);
    for (let o = 0; o < D; o += 1) bias[o] = this.host.singleEmbedBias[o] + row[o];
    this.ops.write(this.singleBias, bias);
    this.t = t;
  }

  /**
   * The energy and score at coordinates r (Float32Array N*3, Angstroms) with self-conditioning coordinates sc (or
   * zeros), at time t. `withScore: false` runs the forward alone.
   */
  async evaluate(r, { sc = null, t = 0.1, withScore = true, readSeams = false } = {}) {
    const ops = this.ops;
    const B = this.B;
    this.setTime(t);
    ops.write(B.r, r);
    ops.write(B.sc, sc ?? new Float32Array(this.N * 3));
    ops.run(withScore ? [this.forward, this.backward] : [this.forward], { dispatchesEach: 400 });
    const rUpdate = await ops.read(B.rUpdate);
    let energy = 0;
    for (const v of rUpdate) energy += v * v;
    const out = { energy, rUpdate };
    if (withScore) out.score = await ops.read(B.score);
    if (readSeams) {
      out.s = await ops.read(B.s);
      out.z = await ops.read(B.z);
      out.a = await ops.read(B.af);
      if (withScore) { out.dEds = await ops.read(B.ds); out.dEdz = await ops.read(B.dz); }
    }
    return out;
  }

  destroy() { this.ops.destroyWeights(); }
}
