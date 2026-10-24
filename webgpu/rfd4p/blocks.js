/**
 * The token-track block shared by the trunk and the autoencoder (`MultiheadAttnAndTransition`): AdaLN, gated
 * multi-head attention with QK norm and an optional pair bias, adaptive output scale, then an AdaLN SwiGLU
 * transition with its own output scale. `norm` is 'rms' (trunk: RMSNorm, weights only) or 'ln' (autoencoder:
 * LayerNorm with biases).
 *
 * Weights are uploaded per layer by `uploadTokenLayer`; `TokenBlocks` records a layer's dispatches against a set of
 * activation buffers sized for `n` tokens (no padding).
 */

const D = 768;
const H = 12;
const HD = 64;
export const LAYER_COND_FLOATS = 6 * D; // mhaGb (2D) + trGb (2D) + [mhaSc | trSc] (2D) = 4608 floats/token/layer

function cat(...arrs) {
  const out = new Float32Array(arrs.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of arrs) {
    out.set(a, o);
    o += a.length;
  }
  return out;
}

/**
 * @param {import('./engine.js').Rfd4pEngine} e
 * @param {(k: string) => Float32Array} get
 * @param {(k: string) => Float32Array|null} opt
 */
export function uploadTokenLayer(e, get, opt, p, half, dimCond) {
  const mat = (k) => e.weight(k, get(k), half);
  const vec = (k) => e.weight(k, get(k), false);
  const ovec = (k) => (opt(k) ? e.weight(k, opt(k), false) : null);
  const adaln = (q) => ({
    normCond: vec(`${q}.norm_cond.weight`),
    normCondB: ovec(`${q}.norm_cond.bias`),
    gb: e.weight(`${q}.gb`, cat(get(`${q}.to_gamma.0.weight`), get(`${q}.to_beta.weight`)), half),
    gbBias: e.weight(`${q}.gb.bias`, cat(get(`${q}.to_gamma.0.bias`), new Float32Array(D))),
  });
  const mhaAdalnP = `${p}.mhba.adaln`;
  const trAdalnP = `${p}.transition.adaln`;
  const m = `${p}.mhba.mha`;
  const mhaScP = `${p}.mhba.scale_output.to_adaln_zero_gamma.0`;
  const trScP = `${p}.transition.scale_output.to_adaln_zero_gamma.0`;
  const foldCond = (q) => {
    const wn = get(`${q}.norm_cond.weight`);
    const bn = opt(`${q}.norm_cond.bias`);
    const wg = cat(get(`${q}.to_gamma.0.weight`), get(`${q}.to_beta.weight`));
    const bg = cat(get(`${q}.to_gamma.0.bias`), new Float32Array(D));
    const wOut = new Float32Array(wg.length);
    const bOut = new Float32Array(bg);
    const outRows = 2 * D;
    for (let r = 0; r < outRows; r += 1) {
      const off = r * dimCond;
      let shift = 0;
      for (let k = 0; k < dimCond; k += 1) {
        const w = wg[off + k];
        if (bn) shift += w * bn[k];
        wOut[off + k] = w * wn[k];
      }
      bOut[r] += shift;
    }
    return { w: wOut, b: bOut };
  };
  const mhaFold = foldCond(mhaAdalnP);
  const trFold = foldCond(trAdalnP);
  const condGbWVal = cat(
    mhaFold.w,
    trFold.w,
    get(`${mhaScP}.weight`), get(`${trScP}.weight`),
  );
  const condGbBVal = cat(
    mhaFold.b,
    trFold.b,
    get(`${mhaScP}.bias`), get(`${trScP}.bias`),
  );
  return {
    dimCond,
    half,
    mhaAdaln: adaln(mhaAdalnP),
    nodeNorm: vec(`${m}.node_norm.weight`),
    nodeNormB: ovec(`${m}.node_norm.bias`),
    qkvg: e.weight(`${m}.to_qkvg.weight`, cat(get(`${m}.to_qkv.weight`), get(`${m}.to_g.weight`)), half),
    qkvgB: e.weight(`${m}.to_qkvg.bias`, cat(get(`${m}.to_qkv.bias`), get(`${m}.to_g.bias`)), false),
    qkNorm: e.weight(`${m}.qk_norm.weight`, cat(get(`${m}.q_layer_norm.weight`), get(`${m}.k_layer_norm.weight`)), false),
    qkNormB: opt(`${m}.q_layer_norm.bias`)
      ? e.weight(`${m}.qk_norm.bias`, cat(opt(`${m}.q_layer_norm.bias`), opt(`${m}.k_layer_norm.bias`)), false)
      : null,
    pairNormVal: opt(`${m}.pair_norm.weight`),
    toBias: opt(`${m}.to_bias.weight`)
      ? e.weight(
          `${m}.to_bias.weight`,
          (() => {
            const wn = get(`${m}.pair_norm.weight`);
            const wb = get(`${m}.to_bias.weight`);
            const out = new Float32Array(wb.length);
            const s = Math.sqrt(384);
            for (let i = 0; i < wb.length; i += 1) out[i] = wb[i] * wn[i % 384] * s;
            return out;
          })(),
          half,
        )
      : null,
    out: mat(`${m}.to_out_node.weight`),
    outB: vec(`${m}.to_out_node.bias`),
    trAdaln: adaln(trAdalnP),
    condGbWVal,
    condGbBVal,
    sw: e.weight(
      `${p}.transition.transition.swish_linear.0.weight`,
      (() => {
        const raw = get(`${p}.transition.transition.swish_linear.0.weight`);
        const out = new Float32Array(raw.length);
        const chunk = 64 * D;
        const halfLen = 4 * D * D;
        for (let t = 0; t < 48; t += 1) {
          out.set(raw.subarray(t * chunk, (t + 1) * chunk), (2 * t) * chunk);
          out.set(raw.subarray(halfLen + t * chunk, halfLen + (t + 1) * chunk), (2 * t + 1) * chunk);
        }
        return out;
      })(),
      half,
    ),
    swOut: mat(`${p}.transition.transition.linear_out.weight`),
    scalesW: e.weight(`${p}.scales.w`, cat(get(`${mhaScP}.weight`), get(`${trScP}.weight`)), half),
    scalesB: e.weight(`${p}.scales.b`, cat(get(`${mhaScP}.bias`), get(`${trScP}.bias`)), false),
  };
}

/** Activation buffers a token stack needs for `n` tokens. */
export function tokenBuffers(e, n, dimCond, tag, nlayers = 14, condRows = n) {
  const b = (label, floats) => e.buffer(`${tag}.${label}`, floats);
  return {
    xn: b('xn', n * D), cn: b('cn', 3 * nlayers * condRows * dimCond), qkv: b('qkv', n * 4 * D),
    qk: b('qk', n * 2 * D), att: b('att', n * D),
    hsw: b('hsw', n * 4 * D), logits: b('logits', H * n * n),
  };
}

export class TokenBlocks {
  /**
   * @param {import('./engine.js').Rfd4pEngine} e
   * @param {number} n
   * @param {'rms'|'ln'} norm
   * @param {object} buf from `tokenBuffers`
   * @param {number} [condRows]
   */
  constructor(e, n, norm, buf, condRows = n) {
    this.e = e;
    this.n = n;
    this.norm = norm;
    this.buf = buf;
    this.condRows = condRows;
  }

  /** y = linear_out(silu(gate) * x), [x | gate] = swish_linear(input) (bias-free SwiGLU). */
  async swiglu(sw, swOut, input, output, rows, dim, inner, _hidBuf, swBuf, gate = undefined, nextNorm = null, nextNorm2 = null) {
    const e = this.e;
    const swF16 = sw.half === true && e.hasSubgroupMatrix && rows > 2;
    const inF16 = swF16 && input === this.buf.xn && rows <= 512;
    return [
      await e.swigluLinear({ x: input, w: sw, y: swBuf, rows, inC: dim, inner, aF16: inF16, cF16: swF16 }),
      await e.linear({ x: swBuf, w: swOut, y: output, rows, inC: inner, outC: dim, gate, nextNorm, nextNorm2, aF16: swF16 }),
    ];
  }

  /**
   * Precompute `[mhaGb (2D), trGb (2D), [mhaSc | trSc] (2D)]` across all `layers` for a conditioning buffer `cBuf`
   * into `condBuf` (`layers.length * condRows * 6 * D` floats).
   */
  async precomputeCond(layers, cBuf, condBuf) {
    const { e, condRows: cr, buf: B } = this;
    const L0 = layers[0];
    const dc = L0.dimCond;
    const nl = layers.length;
    if (!L0.allCondGbW) {
      L0.allCondGbW = e.weight('allCondGbW', cat(...layers.map((L) => L.condGbWVal)), L0.half);
      L0.allCondGbB = e.weight('allCondGbB', cat(...layers.map((L) => L.condGbBVal)), false);
    }
    if (L0.half && e.hasF16 && cr <= 2 && dc % 128 === 0) {
      return [
        await e.gemm({ m: cr, n: 2 * D, k: dc, a: cBuf, b: L0.allCondGbW, c: condBuf, bias: L0.allCondGbB,
          batches: [3 * nl, 1], inNorm: this.norm === 'rms' ? 'rmsCond' : 'lnCond',
          A: { off: 0, s0: 0, m: dc, k: 1 },
          B: { off: 0, s0: 2 * D * dc, n: dc, k: 1 },
          C: { off: 0, s0: cr * 2 * D, m: 2 * D, n: 1 },
          label: `condGb ${cr}x${2 * D}x${dc}[${3 * nl}]` }),
      ];
    }
    return [
      await e.norm({ type: this.norm, x: cBuf, y: B.cn, rows: 2 * nl * cr, width: dc,
        dual: true, transposeN: nl, xStride: dc, yStride: dc }),
      await e.gemm({ m: cr, n: 2 * D, k: dc, a: B.cn, b: L0.allCondGbW, c: condBuf, bias: L0.allCondGbB, batches: [3 * nl, 1],
        A: { off: 0, s0: cr * dc, m: dc, k: 1 },
        B: { off: 0, s0: 2 * D * dc, n: dc, k: 1 },
        C: { off: 0, s0: cr * 2 * D, m: 2 * D, n: 1 },
        label: `condGb ${cr}x${2 * D}x${dc}[${3 * nl}]` }),
    ];
  }

  /**
   * One layer, in place on `s`, reading precomputed conditioning from `condBuf` at `layerIdx * condRows * 6 * D`.
   * @param {object} L from `uploadTokenLayer`
   * @param {GPUBuffer} s `[n, 768]`
   * @param {GPUBuffer} condBuf precomputed conditioning buffer from `precomputeCond`
   * @param {number} layerIdx layer index in `condBuf`
   * @param {GPUBuffer|null} biasBuf `[12, n, n]` pair bias, or null (no pair bias)
   * @param {object[]} [biasSteps] dispatches that fill `biasBuf`, run after the QKV projections
   */
  async layer(L, s, condBuf, layerIdx, biasBuf, biasSteps = [], opts = {}) {
    const { e, n, condRows: cr, buf: B } = this;
    const nn = n * n;
    const cStride = opts.condStride ?? (cr === 1 ? 0 : 2 * D);
    const cOff = opts.condOff ?? 0;
    const base = layerIdx * cr * LAYER_COND_FLOATS + cOff;
    const mhaGbOff = base;
    const trGbOff = base + cr * 2 * D;
    const mhaScOff = base + cr * 4 * D;
    const trScOff = base + cr * 4 * D + D;
    const qkF16 = L.half && e.hasSubgroupMatrix;
    const xnF16 = qkF16 && n <= 512;
    return [
      ...(opts.skipFirstNorm
        ? []
        : [await e.norm({ type: this.norm, x: opts.xIn ?? s, y: B.xn, rows: n, width: D, w: L.nodeNorm, b: L.nodeNormB,
            outF16: xnF16,
            copyXTo: opts.xIn && opts.xIn !== s ? s : null,
            adaln: { gb: condBuf, off: mhaGbOff, stride: cStride, postNorm: true } })]),
      ...(await e.qkvgQkNorm({ x: B.xn, w: L.qkvg, bias: L.qkvgB, qkv: B.qkv, qk: B.qk,
        normType: this.norm, qkNormW: L.qkNorm, qkNormB: L.qkNormB, rows: n, inF16: xnF16, outF16: xnF16 })),
      ...biasSteps,
      // logits[h, i, j] = q_i . k_j / 8
      await e.gemm({ m: n, n, k: HD, a: B.qk, b: B.qk, c: B.logits, alpha: 1 / Math.sqrt(HD), batches: [H, 1], f16: L.half, aF16: xnF16, cF16: xnF16,
        A: { off: 0, s0: HD, m: 2 * D, k: 1 }, B: { off: D, s0: HD, n: 2 * D, k: 1 }, C: { s0: nn, m: n, n: 1 }, label: 'qk' }),
      await e.softmax({ l: B.logits, rows: H * n, len: n, bias: biasBuf, lF16: xnF16 }),
      // att[i, h*64 + d] = (sum_j P[h, i, j] v[j, h*64 + d]) * sigm(g[i, h*64 + d])
      await e.attPv({ logits: B.logits, qkv: B.qkv, att: B.att, n, half: L.half, vgF16: xnF16 }),
      await e.linear({ x: B.att, w: L.out, y: s, rows: n, inC: D, outC: D, bias: L.outB, aF16: xnF16,
        gate: { sc: condBuf, off: mhaScOff, stride: cStride, sigmoid: true },
        nextNorm: { type: this.norm, x: s, y: B.xn, rows: n, width: D, adaln: { gb: condBuf, off: trGbOff, stride: cStride } } }),
      ...(await this.swiglu(L.sw, L.swOut, B.xn, s, n, D, 4 * D, null, B.hsw,
        { sc: condBuf, off: trScOff, stride: cStride, sigmoid: true }, opts.nextNorm ?? null, opts.nextNorm2 ?? null)),
    ];
  }
}
