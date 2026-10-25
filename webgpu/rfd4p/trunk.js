/**
 * The RFD4-Proteina denoiser trunk (`LocalLatentsTransformer`) on WebGPU.
 *
 * Mirrors `cpu/rfd4p/reference.js` (itself checked against PyTorch) op for op:
 *   c      = transition_c_2(transition_c_1(cond_factory(features)))            two variants: first pass / later passes
 *   s_init = init_repr_factory(features)
 *   z_init = AdaLN(RMSNorm(pair_repr_builder(pair features)), pair cond)
 *   per pass: s = s_init + Lin(RMS(s)), z = z_init + Lin(RMS(z)); 14 x [attention + transition], pair update after
 *   even layers but the last (7 updates: outer hadamard write, triangle out, triangle in, pair transition)
 *   heads: ca (3), local latents (32), sequence logits (147).
 *
 * Inference only ever runs one design without padding, so every token is real: the padding mask is all ones and
 * the kernels apply no masks (the emit masks are applied by the caller to the outputs).
 *
 * The pair track never materialises O(n^3): triangle multiplication is a GEMM batched over the 256 hidden channels,
 * with a/b written channel-major (transposed for the incoming update). Per-pair linear layers run in row chunks so
 * scratch is bounded.
 */

import { Rfd4pEngine } from './engine.js';
import { LAYER_COND_FLOATS, TokenBlocks, tokenBuffers, uploadTokenLayer } from './blocks.js';

const D = 768; // token dim
const P = 384; // pair dim
const C = 256; // dim_cond
const H = 12;
const T = 256; // tri_mult_c
/** Trunk layers per command buffer (see `Rfd4pEngine.encodePrograms`). */
const LAYERS_PER_SUBMIT = 2;
const OUT = 3 + 32 + 147;

/** The default bound on the R-indexed pair scratch; see `chunkPlan`. */
export const DEFAULT_CHUNK_BYTES = 896 * 1024 * 1024;

/**
 * How the pair track is sliced, and in which arithmetic.
 *
 * 🔴 THE ARITHMETIC IS CHOSEN BEFORE THE SLICE, NOT AFTER, and it used to be the other way
 * round: `triF16` tested `R === nn` while R came from a bound computed in f32 terms. So
 * narrowing the pair track - the thing that HALVES this scratch - could never buy any of the
 * length it paid for, and at n = 363 the whole f16 path switched off at once. Measured on an
 * Apple M5 (8-core GPU), one trunk pass over 14 layers:
 *
 *        n=100    n=200    n=362    n=400
 *   f16   115.8    444.5   1437.7   1815.8     one chunk
 *   f32       -        -        -   3523.7     two chunks, and 495 ms of it a COPY
 *
 * n=362 -> n=400 is 1.22x the pairs and was 2.45x the time, because the fallback is also the
 * path that never got the f16 kernels or the fused gate. The cliff sat just below the lengths
 * people design at, which is the trap `cuda/rfd4p/README.md` names: a kernel measured at one
 * length is measured at one length.
 *
 * 🔴 AND THE BOUND IS IN BYTES BECAUSE THAT IS WHAT IT WAS ALWAYS MEASURING. The old divisor
 * was `2 * T`, the per-row cost of `proj` ALONE - a buffer only the CHUNKED path allocates -
 * so the real f32 scratch at the old R was 939 MB rather than the 256 MB the float bound
 * appeared to state. 896 MB is that same figure said plainly, which is why f32 lands exactly
 * where it did.
 *
 * 🔴 AND THE TWO SHAPES COST DIFFERENT AMOUNTS A ROW, so the question is asked in that order:
 * does the whole pair tensor fit as ONE chunk, which allocates no `proj` and may be f16 - and
 * only if it does not, how many rows fit the chunked f32 shape, which pays for `proj` as
 * well. Sizing R once from one of the two costs and then picking the other shape is how a
 * bound gets quietly doubled; an earlier draft of this did exactly that, taking f16's row
 * cost and then allocating f32 buffers against it.
 *
 * @param {number} nn pairs, i.e. n * n
 * @param {{wantF16?: boolean, chunkBytes?: number}} [opts]
 * @returns {{triF16: boolean, R: number, chunks: number, scratchBytes: number}}
 */
export function chunkPlan(nn, opts = {}) {
  const wantF16 = opts.wantF16 === true;
  const chunkBytes = opts.chunkBytes ?? DEFAULT_CHUNK_BYTES;
  // zc + xc + gz + phid, as `Rfd4pEngine.buffer` floats a pair row.
  const oneChunkRow = wantF16 ? (P / 2 + T / 2 + P / 2 + P) : (P + T + P + 2 * P);
  const chunkedRow = (P + T + P + 2 * P) + 2 * T;
  const oneChunk = nn * oneChunkRow * 4 <= chunkBytes;
  const R = oneChunk ? nn : Math.max(1, Math.min(nn, Math.floor(chunkBytes / (chunkedRow * 4))));
  return {
    triF16: wantF16 && oneChunk,
    R,
    chunks: Math.ceil(nn / R),
    scratchBytes: R * (oneChunk ? oneChunkRow : chunkedRow) * 4,
  };
}

/**
 * @param {object} weights name -> {data: Float32Array, shape}
 * @param {object} cfg bundle `config.nn`
 */
function checkConfig(cfg) {
  const want = { token_dim: D, pair_repr_dim: P, dim_cond: C, nheads: H, tri_mult_c: T, update_pair_repr_every_n: 2 };
  for (const [k, v] of Object.entries(want)) {
    if (cfg[k] !== v) throw new Error(`rfd4p trunk: config ${k}=${cfg[k]}, kernels are built for ${v}`);
  }
  if (cfg.pair_write !== 'hadamard' || !cfg.use_tri_mult || !cfg.use_qkln) throw new Error('rfd4p trunk: unsupported trunk variant');
}

export class Rfd4pTrunkGpu {
  /**
   * @param {GPUDevice} device
   * @param {object} weights
   * @param {object} cfg bundle `config.nn`
   * @param {{half?: boolean, chunkBytes?: number}} [opts]
   *   `chunkBytes` bounds the R-indexed pair scratch; see `record` for why it is bytes.
   */
  constructor(device, weights, cfg, opts = {}) {
    checkConfig(cfg);
    this.engine = new Rfd4pEngine(device);
    this.cfg = cfg;
    this.half = opts.half ?? false;
    // 896 MB, which is what the old `64 * 1024 * 1024` float bound came to in f32 once the
    // buffers it did not count are counted: one chunk to n = 362 in f32, n = 512 in f16.
    this.chunkBytes = opts.chunkBytes ?? DEFAULT_CHUNK_BYTES;
    this.nlayers = cfg.nlayers;
    this.n = 0;
    this.w = this.uploadWeights(weights);
  }

  uploadWeights(src) {
    const e = this.engine;
    const get = (k) => {
      const t = src[k];
      if (!t) throw new Error(`rfd4p trunk: missing weight ${k}`);
      return t.data ?? t;
    };
    const cat = (...arrs) => {
      const out = new Float32Array(arrs.reduce((s, a) => s + a.length, 0));
      let o = 0;
      for (const a of arrs) {
        out.set(a, o);
        o += a.length;
      }
      return out;
    };
    const mat = (k, label = k) => e.weight(label, get(k), this.half);
    const vec = (k) => e.weight(k, get(k), false);
    const adaln = (p, dim) => {
      const wn = get(`${p}.norm_cond.weight`);
      const wg = cat(get(`${p}.to_gamma.0.weight`), get(`${p}.to_beta.weight`));
      for (let i = 0; i < wg.length; i += 1) wg[i] *= wn[i % C];
      return {
        gb: e.weight(`${p}.gb`, wg, this.half),
        gbBias: e.weight(`${p}.gb.bias`, cat(get(`${p}.to_gamma.0.bias`), new Float32Array(dim))),
      };
    };
    const condW = get('nn.cond_factory.linear_out.weight');
    const seqW = get('nn.init_repr_factory.linear_out.weight');
    const pairW = get('nn.pair_repr_builder.init_repr_factory.linear_out.weight');
    const pairLnW = get('nn.pair_repr_builder.init_repr_factory.ln_out.weight');
    this.condDim = condW.length / C;
    this.seqDim = seqW.length / D;
    const pairDim = pairW.length / P;
    this.pairDim = pairDim;
    const pairWStatic = new Float32Array(pairW.length);
    const pairLnBins = new Float32Array(31 * P);
    pairLnBins.set(pairLnW, 0);
    for (let c = 0; c < P; c += 1) {
      const rowOff = c * pairDim;
      for (let b = 0; b < 30; b += 1) {
        pairLnBins[(1 + b) * P + c] = pairW[rowOff + b];
      }
      for (let k = 30; k < pairDim; k += 1) {
        pairWStatic[rowOff + k] = pairW[rowOff + k];
      }
    }
    const w = {
      condLin: e.weight('nn.cond_factory.linear_out.weight', condW, this.half, C, 128),
      tc: [1, 2].map((k) => ({ sw: mat(`nn.transition_c_${k}.swish_linear.0.weight`), out: mat(`nn.transition_c_${k}.linear_out.weight`) })),
      seqLin: e.weight('nn.init_repr_factory.linear_out.weight', seqW, this.half, D, 64),
      pairLin: e.weight('nn.pair_repr_builder.init_repr_factory.linear_out.weight', pairWStatic, this.half, P),
      pairLnBins: e.weight('nn.pair_repr_builder.init_repr_factory.ln_bins', pairLnBins, false),
      pairCondLin: mat('nn.pair_repr_builder.cond_factory.linear_out.weight'),
      pairCondLn: vec('nn.pair_repr_builder.cond_factory.ln_out.weight'),
      pairAdaln: adaln('nn.pair_repr_builder.adaln', P),
      sCycNorm: vec('nn.s_cycle_norm.weight'),
      sCycProj: mat('nn.s_cycle_proj.weight'),
      zCycNorm: vec('nn.z_cycle_norm.weight'),
      zCycNormScaled: e.weight(
        'nn.z_cycle_norm.scaled',
        Float32Array.from(get('nn.z_cycle_norm.weight'), (x) => x * Math.sqrt(P)),
        false,
      ),
      zCycProj: mat('nn.z_cycle_proj.weight'),
      heads: ['ca_linear', 'local_latents_linear', 'sequence_readout'].map((h) => ({ norm: vec(`nn.${h}.0.weight`), lin: mat(`nn.${h}.1.weight`) })),
      headsFastLin: e.weight(
        'nn.heads_fast.lin',
        (() => {
          const scaleHead = (wn, wl) => {
            const out = new Float32Array(wl.length);
            const s = Math.sqrt(D);
            for (let i = 0; i < wl.length; i += 1) out[i] = wl[i] * wn[i % D] * s;
            return out;
          };
          return cat(
            scaleHead(get('nn.ca_linear.0.weight'), get('nn.ca_linear.1.weight')),
            scaleHead(get('nn.local_latents_linear.0.weight'), get('nn.local_latents_linear.1.weight')),
          );
        })(),
        this.half,
      ),
      insQ: mat('nn.insertion_q_proj.weight'),
      insK: mat('nn.insertion_k_proj.weight'),
      layers: [],
      pairs: [],
    };
    const opt = (k) => (src[k] ? src[k].data ?? src[k] : null);
    for (let i = 0; i < this.nlayers; i += 1) {
      w.layers.push(uploadTokenLayer(e, get, opt, `nn.transformer_layers.${i}`, this.half, C));
      if (i < this.nlayers - 1 && i % 2 === 0) {
        const q = `nn.pair_update_layers.${i}`;
        const tri = (d) => ({
          lnIn: vec(`${q}.tri_mult_${d}.layer_norm_in.weight`),
          projG: e.weight(
            `${q}.tri_${d}.projG`,
            (() => {
              const ap = get(`${q}.tri_mult_${d}.linear_a_p.weight`);
              const ag = get(`${q}.tri_mult_${d}.linear_a_g.weight`);
              const bp = get(`${q}.tri_mult_${d}.linear_b_p.weight`);
              const bg = get(`${q}.tri_mult_${d}.linear_b_g.weight`);
              const g = get(`${q}.tri_mult_${d}.linear_g.weight`);
              const out = new Float32Array(1408 * P);
              const chunk = 64 * P;
              for (let t = 0; t < 4; t += 1) {
                out.set(ap.subarray(t * chunk, (t + 1) * chunk), (2 * t) * chunk);
                out.set(ag.subarray(t * chunk, (t + 1) * chunk), (2 * t + 1) * chunk);
                out.set(bp.subarray(t * chunk, (t + 1) * chunk), (8 + 2 * t) * chunk);
                out.set(bg.subarray(t * chunk, (t + 1) * chunk), (8 + 2 * t + 1) * chunk);
              }
              out.set(g, 1024 * P);
              return out;
            })(),
            this.half,
          ),
          lnOut: vec(`${q}.tri_mult_${d}.layer_norm_out.weight`),
          z: mat(`${q}.tri_mult_${d}.linear_z.weight`),
        });
        w.pairs[i] = {
          lnW: vec(`${q}.layer_norm_in.weight`),
          lnB: vec(`${q}.layer_norm_in.bias`),
          lnWB: e.weight(`${q}.layer_norm_in.wb`, cat(get(`${q}.layer_norm_in.weight`), get(`${q}.layer_norm_in.bias`)), false),
          linX: mat(`${q}.linear_x.weight`),
          out: tri('out'),
          in: tri('in'),
          trLnW: vec(`${q}.transition_out.layer_norm.weight`),
          trLnB: vec(`${q}.transition_out.layer_norm.bias`),
          l1: mat(`${q}.transition_out.linear_1.weight`),
          l1B: vec(`${q}.transition_out.linear_1.bias`),
          l1Fused: this.half
            ? (() => {
                const lnW = get(`${q}.transition_out.layer_norm.weight`);
                const lnB = get(`${q}.transition_out.layer_norm.bias`);
                const w1 = get(`${q}.transition_out.linear_1.weight`);
                const b1 = get(`${q}.transition_out.linear_1.bias`);
                const wScaled = new Float32Array(w1.length);
                for (let j = 0; j < w1.length; j += 1) wScaled[j] = w1[j] * lnW[j % P];
                const wQuant = typeof Float16Array !== 'undefined' ? new Float16Array(wScaled) : wScaled;
                const bCombined = new Float32Array(4 * P);
                for (let r = 0; r < 2 * P; r += 1) {
                  const rOff = r * P;
                  let bSum = b1[r];
                  let wSum = 0;
                  for (let c = 0; c < P; c += 1) {
                    bSum += lnB[c] * w1[rOff + c];
                    wSum += wQuant[rOff + c];
                  }
                  bCombined[r] = bSum;
                  bCombined[2 * P + r] = wSum;
                }
                return {
                  w: e.weight(`${q}.transition_out.l1.fused.w`, wScaled, true),
                  b: e.weight(`${q}.transition_out.l1.fused.b`, bCombined, false),
                };
              })()
            : null,
          l2: mat(`${q}.transition_out.linear_2.weight`),
          l2B: vec(`${q}.transition_out.linear_2.bias`),
        };
      }
    }
    for (let i = 1; i + 1 < this.nlayers; i += 2) {
      const m0 = `nn.transformer_layers.${i}.mhba.mha`;
      const m1 = `nn.transformer_layers.${i + 1}.mhba.mha`;
      const scaleBias = (wn, wb) => {
        const out = new Float32Array(wb.length);
        const s = Math.sqrt(P);
        for (let j = 0; j < wb.length; j += 1) out[j] = wb[j] * wn[j % P] * s;
        return out;
      };
      w.layers[i].toBiasDual = e.weight(
        `${m0}.to_bias_dual.weight`,
        cat(
          scaleBias(get(`${m0}.pair_norm.weight`), get(`${m0}.to_bias.weight`)),
          scaleBias(get(`${m1}.pair_norm.weight`), get(`${m1}.to_bias.weight`)),
        ),
        this.half,
      );
    }
    return w;
  }

  /** Allocate activations for `n` tokens and record every program. */
  async record(n, dims, condUniform = true) {
    const e = this.engine;
    if (this.n === n && this.condUniform === condUniform) return;
    e.releaseActivations();
    this.n = n;
    this.condUniform = condUniform;
    const condRows = condUniform ? 1 : n;
    const batchCondRows = 2 * condRows;
    this.condRows = condRows;
    this.batchCondRows = batchCondRows;
    const w = this.w;
    const seqStride = w.seqLin.stride || dims.seq;
    const condStride = w.condLin.stride || dims.cond;
    const pairStride = w.pairLin.stride || dims.pair;
    this.seqStride = seqStride;
    this.condStride = condStride;
    this.pairStride = pairStride;
    const nn = n * n;
    this.seqPad = seqStride === dims.seq ? null : new Float32Array(n * seqStride);
    this.condPad = new Float32Array(batchCondRows * condStride);
    this.pairPad = pairStride === dims.pair ? null : new Float32Array(nn * pairStride);
    const nPad = Math.ceil(n / 16) * 16;
    const nnPad = n * nPad;
    this.nPad = nPad;
    // The slice and the arithmetic together; `chunkPlan` carries why.
    const { triF16, R } = chunkPlan(nn, { wantF16: this.half && e.hasF16, chunkBytes: this.chunkBytes });
    this.triF16 = triF16;
    this.R = R;
    const b = (label, floats) => e.buffer(label, floats);
    const B = (this.buf = {
      seqF: b('seqF', n * seqStride), condF: b('condF', batchCondRows * condStride),
      pairF: b('pairF', nn * pairStride), pairCondF: b('pairCondF', dims.pairCond),
      c: b('c', batchCondRows * C), cTmp: b('cTmp', batchCondRows * C), cSw: b('cSw', batchCondRows * 2 * C),
      condBuf: b('condBuf', this.nlayers * batchCondRows * LAYER_COND_FLOATS),
      sInit: b('sInit', n * D), s: b('s', n * D), sT: b('sT', 2 * n * D), bias: b('bias', H * nn), bias2: b('bias2', H * nn), pp: b('pp', n * D),
      ...tokenBuffers(e, n, C, 'trunk', this.nlayers, batchCondRows),
      zInitStatic: b('zInitStatic', triF16 ? Math.ceil((nn * P) / 2) : nn * P),
      zInit: b('zInit', triF16 ? Math.ceil((nn * P) / 2) : nn * P),
      z: b('z', triF16 ? Math.ceil((nn * P) / 2) : nn * P),
      aT: b('aT', triF16 ? Math.ceil((T * nnPad) / 2) : T * nn),
      bT: b('bT', triF16 ? Math.ceil((T * nnPad) / 2) : T * nn),
      x: b('x', triF16 ? Math.ceil((T * Math.ceil(nn / 4) * 4) / 2) : T * Math.ceil(nn / 4) * 4),
      zc: b('zc', triF16 ? Math.ceil((R * P) / 2) : R * P), proj: R < nn ? b('proj', R * 2 * T) : null,
      xc: b('xc', triF16 ? Math.ceil((R * T) / 2) : R * T),
      gz: b('gz', triF16 ? Math.ceil((R * P) / 2) : R * P),
      phid: b('phid', triF16 ? R * P : R * 2 * P),
      zcond: b('zcond', C), zcondN: b('zcondN', C), zgb: b('zgb', 2 * P), out: b('out', n * (OUT + 35)),
      simBuf: b('simBuf', 16 + n * 35), emitBuf: b('emitBuf', 2 * n),
    });
    this.blocks = new TokenBlocks(e, n, 'rms', B, batchCondRows);
    const chunks = [];
    for (let r0 = 0; r0 < nn; r0 += R) chunks.push([r0, Math.min(R, nn - r0)]);
    this.chunks = chunks;
    this.lastPairSrc = null;

    const pairStatic = [];
    for (const [r0, rows] of chunks) {
      pairStatic.push(
        await e.linear({ x: B.pairF, w: w.pairLin, y: B.zInitStatic, rows, inC: pairStride, outC: P, cF16: triF16, xOff: r0 * pairStride, yOff: r0 * P }),
      );
    }
    this.pairStaticProgram = e.program(pairStatic);

    // --- c (two variants batched into batchCondRows), s_init, z_init -----------------------------------
    const init = [
      await e.linear({ x: B.condF, w: w.condLin, y: B.c, rows: batchCondRows, inC: condStride, outC: C }),
      ...(await this.blocks.swiglu(w.tc[0].sw, w.tc[0].out, B.c, B.cTmp, batchCondRows, C, 2 * C, null, B.cSw)),
      ...(await this.blocks.swiglu(w.tc[1].sw, w.tc[1].out, B.cTmp, B.c, batchCondRows, C, 2 * C, null, B.cSw)),
      ...(await this.blocks.precomputeCond(w.layers, B.c, B.condBuf)),
      await e.linear({ x: B.seqF, w: w.seqLin, y: B.sInit, rows: n, inC: seqStride, outC: D }),
      // compact pair conditioning -> AdaLN gamma/beta, one row
      await e.linear({ x: B.pairCondF, w: w.pairCondLin, y: B.zcond, rows: 1, inC: dims.pairCond, outC: C }),
      ...(this.half && e.hasF16
        ? [await e.linear({ x: B.zcond, w: w.pairAdaln.gb, y: B.zgb, rows: 1, inC: C, outC: 2 * P, bias: w.pairAdaln.gbBias, inNorm: 'rmsPre', inNormW: w.pairCondLn })]
        : [
            await e.norm({ type: 'rms', x: B.zcond, y: B.zcondN, rows: 1, width: C, w: w.pairCondLn, preNorm: true }),
            await e.linear({ x: B.zcondN, w: w.pairAdaln.gb, y: B.zgb, rows: 1, inC: C, outC: 2 * P, bias: w.pairAdaln.gbBias }),
          ]),
    ];
    for (const [r0, rows] of chunks) {
      init.push(
        await e.pairToBias({
          z: B.zInit,
          wBias: w.layers[0].toBias,
          bias: B.bias,
          zInitAda: { wLnBins: w.pairLnBins, gb: B.zgb, zStatic: B.zInitStatic, seqF: B.seqF, n, seqStride },
          zF16: triF16,
          rows,
          nn,
          r0,
        }),
      );
    }
    this.initProgram = e.program(init);

    // --- one trunk pass for each c ---------------------------------------------------------------------
    const condRowStride = condRows === 1 ? 0 : 2 * D;
    this.passPrograms = [];
    for (const [firstCycle, hasNextCycle] of [
      [true, false],
      [false, false],
      [false, true],
      [true, true],
    ]) {
      const condOff = firstCycle ? 0 : condRows * 2 * D;
      const layer0Norm = {
        type: 'rms', x: B.s, y: B.xn, rows: n, width: D, w: w.layers[0].nodeNorm, b: w.layers[0].nodeNormB,
        adaln: { gb: B.condBuf, off: condOff, stride: condRowStride, postNorm: true },
      };
      const steps = [...(await this.recycleIn(firstCycle, layer0Norm))];
      const chunks = [];
      for (let i = 0; i < this.nlayers; i += 1) {
        const nextL = i + 1 < this.nlayers ? w.layers[i + 1] : null;
        const nextNorm = nextL
          ? { type: 'rms', x: B.s, y: B.xn, rows: n, width: D, w: nextL.nodeNorm, b: nextL.nodeNormB,
              adaln: { gb: B.condBuf, off: (i + 1) * batchCondRows * LAYER_COND_FLOATS + condOff, stride: condRowStride, postNorm: true } }
          : { type: 'rms', x: B.s, y: B.xn, rows: n, width: D, w: w.heads[2].norm };
        const Q = w.pairs[i];
        const nextNorm2 = Q
          ? { type: 'ln', x: B.s, y: B.sT, rows: n, width: D, w: Q.lnW, b: Q.lnB, wb: Q.lnWB }
          : (!nextL && hasNextCycle ? { type: 'rms', x: B.s, y: B.sT, rows: n, width: D, w: w.sCycNorm, wb: w.sCycNorm } : null);
        steps.push(...(await this.attentionLayer(i, B.condBuf, firstCycle && i === 0, {
          skipFirstNorm: i > 0 || !firstCycle,
          xIn: firstCycle && i === 0 ? B.sInit : null,
          hasNextCycle,
          condOff,
          condStride: condRowStride,
          nextNorm,
          nextNorm2,
        })));
        if (Q) steps.push(...(await this.pairUpdate(i, firstCycle && i === 0, true)));
        // Bound each command buffer (macOS GPU watchdog): a cycle is several programs of LAYERS_PER_SUBMIT layers.
        if ((i + 1) % LAYERS_PER_SUBMIT === 0 && i + 1 < this.nlayers) {
          chunks.push(e.program(steps));
          steps.length = 0;
        }
      }
      chunks.push(e.program(steps));
      this.passPrograms.push(chunks);
    }

    // --- heads -------------------------------------------------------------------------------------------
    const headFastStep = await e.headFast({ s: B.s, wLin: w.headsFastLin, out: B.out, n });
    const heads = [
      headFastStep,
      await e.linear({ x: B.xn, w: w.heads[2].lin, y: B.out, rows: n, inC: D, outC: 147, yOff: 35 * n }),
    ];
    this.headProgram = e.program(heads);
    this.headFastProgram = e.program([headFastStep]);
    this.insertionProgram = e.program([
      await e.linear({ x: B.s, w: w.insQ, y: B.sT, rows: n, inC: D, outC: D, label: 'ins_q' }),
      await e.linear({ x: B.s, w: w.insK, y: B.xn, rows: n, inC: D, outC: D, label: 'ins_k' }),
      await e.gemm({
        m: n, n, k: D, a: B.sT, b: B.xn, c: B.logits, alpha: 1 / Math.sqrt(D),
        A: { m: D, k: 1 }, B: { n: D, k: 1 }, C: { m: n, n: 1 }, label: 'ins_logits',
      }),
    ]);
    this.simStepProgram = e.program([
      await e.map(
        'sim-step',
        n * 35,
        [
          { buffer: B.seqF, write: true },
          { buffer: B.out, write: true },
          { buffer: B.simBuf },
          { buffer: B.emitBuf, type: 'u32' },
        ],
        `let n = U(0u);
  let stride = U(1u);
  let isBb = i < n * 3u;
  let j = select(i - n * 3u, i, isBb);
  let tok = select(j / 32u, j / 3u, isBb);
  let c = select(j % 32u, j % 3u, isBb);
  let em = b3[select(n + tok, tok, isBb)];
  let seqIdx = tok * stride + select(3u, 0u, isBb) + c;
  var x_next = 0.0;
  var x1 = 0.0;
  if (em != 0u) {
    let pBase = select(6u, 0u, isBb);
    let t = b2[pBase];
    let dt = b2[pBase + 1u];
    let gt = b2[pBase + 2u];
    let r = b2[pBase + 3u];
    let sd = b2[pBase + 4u];
    let stoch = b2[pBase + 5u];
    let x_i = b0[seqIdx];
    let v_i = b1[i];
    x1 = x_i + (1.0 - t) * v_i;
    let eps_i = b2[16u + i];
    let score = (t * v_i - x_i) / (1.0 - t);
    let ss = score * r;
    let vs = (x_i + (1.0 - t) * ss) / t;
    let delta = select(vs * dt, (vs + gt * ss) * dt + sd * eps_i, stoch != 0.0);
    x_next = select(x_i + delta, x_i, t == 0.0);
  }
  b0[seqIdx] = x_next;
  b1[i] = x_next;
  b1[n * 182u + i] = x1;`,
        [n, seqStride],
      ),
    ]);
  }

  /** One token layer with the pair bias [H, n, n] = to_bias(RMS_w(z)), fused without materializing B.zc. */
  async attentionLayer(i, condBuf, useZInit = false, opts = {}) {
    const e = this.engine;
    const nn = this.n * this.n;
    const B = this.buf;
    const L = this.w.layers[i];
    const zBuf = useZInit ? B.zInit : B.z;
    if (useZInit) {
      return this.blocks.layer(L, B.s, condBuf, i, B.bias, [], opts);
    }
    if (i > 0 && i % 2 === 0) {
      return this.blocks.layer(L, B.s, condBuf, i, B.bias2, [], opts);
    }
    const dual = Boolean(L.toBiasDual);
    const outZc = i === this.nlayers - 1 && opts.hasNextCycle && this.chunks.length === 1
      ? { wZc: this.w.zCycNormScaled, zc: B.zc, f16: this.triF16 }
      : null;
    const biasSteps = [];
    for (const [r0, rows] of this.chunks) {
      biasSteps.push(
        await e.pairToBias({
          z: zBuf,
          wBias: dual ? L.toBiasDual : L.toBias,
          bias: B.bias,
          bias2: dual ? B.bias2 : null,
          outZc,
          zF16: this.triF16,
          rows,
          nn,
          r0,
        }),
      );
    }
    return this.blocks.layer(L, B.s, condBuf, i, B.bias, biasSteps, opts);
  }

  async triangle(t, incoming, fusePairWrite = false, zIn = null, splitPp = false) {
    const e = this.engine;
    const { n, nPad } = this;
    const nn = n * n;
    const nnPad = n * nPad;
    const B = this.buf;
    if (this.chunks.length === 1) {
      const triF16 = this.triF16;
      const stride = triF16 ? nPad : n;
      const batchStride = triF16 ? nnPad : nn;
      const xBatchStride = Math.ceil(nn / 4) * 4;
      return [
        fusePairWrite
          ? await e.pairWriteLnIn({ z: B.z, zIn, pp: B.pp, zc: B.zc, w: t.lnIn, rows: nn, n, outF16: triF16, splitPp })
          : await e.norm({ type: 'ln', x: B.z, y: B.zc, rows: nn, width: P, w: t.lnIn, inF16: triF16, outF16: triF16, transposeN: incoming ? n : 0 }),
        await e.triProjG({ x: B.zc, w: t.projG, aT: B.aT, bT: B.bT, gz: B.gz, rows: nn, incoming, n, nPad: stride, aF16: triF16, outF16: triF16 }),
        await e.gemm({ m: n, n, k: stride, a: B.aT, b: B.bT, c: B.x, batches: [T, 1], f16: this.half, aF16: triF16, cF16: triF16,
          A: { s0: batchStride, m: stride, k: 1 }, B: { s0: batchStride, n: stride, k: 1 }, C: { s0: xBatchStride, m: n, n: 1 }, label: incoming ? 'tri-in' : 'tri-out' }),
        await e.transposeLnOut({ x: B.x, y: B.xc, rows: nn, nn: xBatchStride, w: t.lnOut, inF16: triF16, outF16: triF16 }),
        await e.linear({ x: B.xc, w: t.z, y: B.z, rows: nn, inC: T, outC: P, aF16: triF16, cF16: triF16, gate: { sc: B.gz, off: 0, stride: P, sigmoid: false, f16: triF16 } }),
      ];
    }
    const xBatchStride = Math.ceil(nn / 4) * 4;
    const steps = [];
    for (const [r0, rows] of this.chunks) {
      steps.push(
        fusePairWrite
          ? await e.pairWriteLnIn({ z: B.z, zIn, pp: B.pp, zc: B.zc, w: t.lnIn, rows, n, r0, splitPp })
          : await e.norm({ type: 'ln', x: B.z, y: B.zc, rows, width: P, w: t.lnIn, xOff: r0 * P }),
        await e.triProjG({ x: B.zc, w: t.projG, proj: B.proj, gz: B.gz, rows }),
        await e.map(incoming ? 'tri-gate-in' : 'tri-gate-out', rows * T,
          [{ buffer: B.aT, write: true }, { buffer: B.bT, write: true }, { buffer: B.proj }],
          `let rr = i / 256u; let c = i % 256u; let r = U(0u) + rr; let n = U(1u);
  let pr = rr * 512u;
  let av = b2[pr + c];
  let bv = b2[pr + 256u + c];
  ${incoming ? 'let at = c * n * n + (r % n) * n + r / n;' : 'let at = c * n * n + r;'}
  b0[at] = av; b1[at] = bv;`, [r0, n]),
      );
    }
    steps.push(await e.gemm({ m: n, n, k: n, a: B.aT, b: B.bT, c: B.x, batches: [T, 1], f16: this.half,
      A: { s0: nn, m: n, k: 1 }, B: { s0: nn, n: n, k: 1 }, C: { s0: xBatchStride, m: n, n: 1 }, label: incoming ? 'tri-in' : 'tri-out' }));
    for (const [r0, rows] of this.chunks) {
      steps.push(
        await e.transposeLnOut({ x: B.x, y: B.xc, rows, nn: xBatchStride, w: t.lnOut, r0 }),
        await e.norm({ type: 'ln', x: B.z, y: B.zc, rows, width: P, w: t.lnIn, xOff: r0 * P }),
        await e.triProjG({ x: B.zc, w: t.projG, proj: B.proj, gz: B.gz, rows }),
        await e.linear({ x: B.xc, w: t.z, y: B.z, rows, inC: T, outC: P, yOff: r0 * P, gate: { sc: B.gz, off: 0, stride: P, sigmoid: false } }),
      );
    }
    return steps;
  }

  async pairUpdate(i, useZInit = false, skipLn = false) {
    const e = this.engine;
    const { n } = this;
    const B = this.buf;
    const Q = this.w.pairs[i];
    const splitPp = n <= 512;
    const steps = [
      ...(skipLn ? [] : [await e.norm({ type: 'ln', x: B.s, y: B.sT, rows: n, width: D, w: Q.lnW, b: Q.lnB })]),
      await e.linear({ x: B.sT, w: Q.linX, y: B.pp, rows: n, inC: D, outC: D, skipSplitReduce: splitPp }),
      ...(await this.triangle(Q.out, false, true, useZInit ? B.zInit : null, splitPp)),
      ...(await this.triangle(Q.in, true, false)),
    ];
    const triF16 = this.triF16;
    for (const [r0, rows] of this.chunks) {
      if (triF16 && Q.l1Fused) {
        steps.push(
          await e.linear({ x: B.z, w: Q.l1Fused.w, y: B.phid, rows, inC: P, outC: 2 * P, bias: Q.l1Fused.b, act: 'relu', inNorm: 'ln', aF16: true, cF16: true, xOff: r0 * P }),
          await e.linear({ x: B.phid, w: Q.l2, y: B.z, rows, inC: 2 * P, outC: P, bias: Q.l2B, accumulate: true, aF16: true, cF16: true, yOff: r0 * P }),
        );
      } else {
        steps.push(
          await e.norm({ type: 'ln', x: B.z, y: B.zc, rows, width: P, w: Q.trLnW, b: Q.trLnB, inF16: triF16, outF16: triF16, xOff: r0 * P }),
          await e.linear({ x: B.zc, w: Q.l1, y: B.phid, rows, inC: P, outC: 2 * P, bias: Q.l1B, act: 'relu', aF16: triF16, cF16: triF16 }),
          await e.linear({ x: B.phid, w: Q.l2, y: B.z, rows, inC: 2 * P, outC: P, bias: Q.l2B, accumulate: true, aF16: triF16, cF16: triF16, yOff: r0 * P }),
        );
      }
    }
    return steps;
  }

  /** s = s_init + Lin(RMS(s)), z = z_init + Lin(RMS(z)) (zero-initialised on cycle 0 -> fused into layer 0). */
  async recycleIn(firstCycle = false, nextNorm = null) {
    const e = this.engine;
    const { n } = this;
    const B = this.buf;
    const w = this.w;
    if (firstCycle) {
      return [];
    }
    const steps = [
      await e.linear({ x: B.sT, w: w.sCycProj, y: B.s, rows: n, inC: D, outC: D,
        gate: { sc: B.sInit, off: 0, stride: D, addOnly: true }, nextNorm }),
    ];
    const triF16 = this.triF16;
    for (const [r0, rows] of this.chunks) {
      if (this.chunks.length > 1) {
        steps.push(
          await e.norm({ type: 'rms', x: B.z, y: B.zc, rows, width: P, w: w.zCycNorm, outF16: triF16, xOff: r0 * P }),
        );
      }
      steps.push(
        await e.linear({ x: B.zc, w: w.zCycProj, y: B.z, rows, inC: P, outC: P, aF16: triF16, cF16: triF16, yOff: r0 * P,
          gate: { sc: B.zInit, off: r0 * P, stride: P, addOnly: true, f16: triF16 } }),
      );
    }
    return steps;
  }

  /**
   * One network evaluation.
   * @param {{seq: Float32Array, cond0: Float32Array, cond1: Float32Array, pair: Float32Array, pairCond: Float32Array}} f
   *   pre-projection features (`shared/rfd4p/features.js`): cond0 without, cond1 with recycle-remaining conditioning
   * @param {number} nRecycle total trunk passes (>= 1)
   * @param {{insertion?: boolean}} [opts]
   * @returns {Promise<{bb_ca: Float32Array, local_latents: Float32Array, logits: Float32Array, insertion?: Float32Array}>}
   */
  async forward(n, f, nRecycle, opts = {}) {
    const dims = {
      seq: this.seqDim,
      cond: this.condDim,
      pair: this.pairDim,
      pairCond: f.pairCond.length,
    };
    const isCondUniform = (c) => {
      if (c.uniform !== undefined) return c.uniform;
      if (c.length === dims.cond) return true;
      for (let i = 1; i < n; i += 1) {
        const r = i * dims.cond;
        for (let k = 0; k < dims.cond; k += 1) {
          if (c[r + k] !== c[k]) return false;
        }
      }
      return true;
    };
    const condUniform = isCondUniform(f.cond0) && (!f.cond1 || nRecycle <= 1 || isCondUniform(f.cond1));
    await this.record(n, dims, condUniform);
    const e = this.engine;
    const B = this.buf;
    const { seqStride, condStride, pairStride, condRows } = this;
    if (f.seq) {
      if (f.seq.length === n * seqStride) {
        e.write(B.seqF, f.seq);
      } else {
        for (let i = 0; i < n; i += 1) {
          this.seqPad.set(f.seq.subarray(i * dims.seq, (i + 1) * dims.seq), i * seqStride);
        }
        e.write(B.seqF, this.seqPad);
      }
    }
    if (opts.emit) {
      const emitU32 = new Uint32Array(2 * n);
      emitU32.set(opts.emit.bb_ca, 0);
      emitU32.set(opts.emit.local_latents, n);
      e.write(B.emitBuf, emitU32);
    }
    if (opts.simBuf) {
      e.write(B.simBuf, opts.simBuf);
    }
    if (condRows === 1) {
      this.condPad.set(f.cond0.subarray(0, dims.cond), 0);
      if (f.cond1 && nRecycle > 1) {
        this.condPad.set(f.cond1.subarray(0, dims.cond), condStride);
      }
    } else {
      for (let i = 0; i < n; i += 1) {
        this.condPad.set(f.cond0.subarray(i * dims.cond, (i + 1) * dims.cond), i * condStride);
      }
      if (f.cond1 && nRecycle > 1) {
        for (let i = 0; i < n; i += 1) {
          this.condPad.set(f.cond1.subarray(i * dims.cond, (i + 1) * dims.cond), (n + i) * condStride);
        }
      }
    }
    e.write(B.condF, this.condPad);
    let needPairStatic = false;
    if (f.pair && this.lastPairSrc !== f.pair) {
      this.lastPairSrc = f.pair;
      needPairStatic = true;
      const nn = n * n;
      if (f.pair.length === nn * pairStride) {
        e.write(B.pairF, f.pair);
      } else {
        const dPair = dims.pair;
        const src = f.pair;
        const dst = this.pairPad;
        for (let i = 0; i < nn; i += 1) {
          const s0 = i * dPair;
          const d0 = i * pairStride;
          for (let k = 0; k < dPair; k += 1) dst[d0 + k] = src[s0 + k];
        }
        e.write(B.pairF, this.pairPad);
      }
    }
    e.write(B.pairCondF, f.pairCond);
    const wantLogits = opts.logits !== false;
    const passes = needPairStatic ? [this.pairStaticProgram, this.initProgram] : [this.initProgram];
    const totalCycles = Math.max(1, nRecycle);
    for (let cycle = 0; cycle < totalCycles; cycle += 1) {
      const hasNext = cycle + 1 < totalCycles;
      passes.push(this.passPrograms[cycle === 0 ? (hasNext ? 3 : 0) : (hasNext ? 2 : 1)]);
    }
    passes.push(wantLogits ? this.headProgram : this.headFastProgram);
    if (opts.insertion) passes.push(this.insertionProgram);
    if (opts.simBuf) {
      passes.push(this.simStepProgram);
      if (opts.readback === false) {
        e.run(passes);
        return null;
      }
    }
    const readFloats = opts.simBuf ? n * (OUT + 35) : n * (wantLogits ? OUT : 35);
    const out = await e.runAndRead(passes, B.out, readFloats);
    const insertion = opts.insertion ? await e.read(B.logits, n * n) : undefined;
    const bb = out.subarray(0, n * 3);
    const ll = out.subarray(n * 3, n * 35);
    const logits = wantLogits ? out.subarray(n * 35, n * OUT) : undefined;
    const x1 = opts.simBuf
      ? { bb_ca: out.subarray(n * OUT, n * (OUT + 3)), local_latents: out.subarray(n * (OUT + 3), n * (OUT + 35)) }
      : undefined;
    return { bb_ca: bb, local_latents: ll, logits, insertion, x1 };
  }

  /** Debug: read a named activation buffer. */
  async read(name, floats) {
    if (name === 'c0') {
      if (this.condRows === 1) {
        const row0 = await this.engine.read(this.buf.c, C);
        const out = new Float32Array(floats);
        for (let o = 0; o < floats; o += C) out.set(row0, o);
        return out;
      }
      return this.engine.read(this.buf.c, floats);
    }
    if ((name === 'z' || name === 'zInit' || name === 'zInitStatic') && this.triF16) {
      const raw = await this.engine.read(this.buf[name], Math.ceil(floats / 2));
      const u16 = new Uint16Array(raw.buffer, raw.byteOffset, floats);
      const out = new Float32Array(floats);
      const ab = new ArrayBuffer(4);
      const u32 = new Uint32Array(ab);
      const f32 = new Float32Array(ab);
      for (let i = 0; i < floats; i += 1) {
        const h = u16[i];
        const s = (h & 0x8000) << 16;
        const e = (h >> 10) & 0x1f;
        const m = h & 0x03ff;
        if (e === 0) {
          out[i] = m === 0 ? 0 : (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
        } else if (e === 31) {
          out[i] = m ? NaN : (s ? -Infinity : Infinity);
        } else {
          u32[0] = s | ((e + 112) << 23) | (m << 13);
          out[i] = f32[0];
        }
      }
      return out;
    }
    return this.engine.read(this.buf[name], floats);
  }

  destroy() {
    this.engine.destroy();
  }
}
