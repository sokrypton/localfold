/**
 * The RFD4-Proteina autoencoder decoder (`Autoencoder.decode` + `DecoderTransformer.forward`) on WebGPU.
 *
 * Mirrors `cpu/rfd4p/decoder.js`:
 *   - `c` is identically zero (`dim_cond = 128`, bias-free transitions on zero features).
 *   - Only attended tokens (`attn = padding_mask & ~is_in_context_conditioning`) participate in self-attention
 *     or are read by `decoderPostprocess` (`emit = attn & supports_local_latent`). Gathering the `na` attended
 *     rows and running unmasked attention over `na` tokens is exact.
 *   - Input features `[n, 214]` are assembled on CPU by `decoderSeqFeatures`, gathered to `[na, 214]`, projected
 *     by `init_repr_factory.linear_out`, transformed by 12 LayerNorm `TokenBlocks` layers, and projected by the
 *     three heads (`logit_linear` 147, `struct_linear` 111, `atom_presence_linear` 37).
 */

import { aeMasks, decoderPostprocess, decoderSeqFeatures } from '../../cpu/rfd4p/decoder.js';
import { LAYER_COND_FLOATS, TokenBlocks, tokenBuffers, uploadTokenLayer } from './blocks.js';
import { Rfd4pEngine } from './engine.js';

const PREFIX = 'autoencoder.decoder';
const D = 768;
const C = 128;
const H = 12;

function checkConfig(cfg) {
  const want = { token_dim: D, dim_cond: C, nheads: H };
  for (const [k, v] of Object.entries(want)) {
    if (cfg[k] !== v) throw new Error(`rfd4p decoder: config ${k}=${cfg[k]}, kernels are built for ${v}`);
  }
}

export class Rfd4pDecoderGpu {
  /**
   * @param {GPUDevice} device
   * @param {object} weights name -> {data: Float32Array, shape}
   * @param {object} cfg bundle `config.autoencoder_config.nn_ae.decoder`
   * @param {{half?: boolean}} [opts]
   */
  constructor(device, weights, cfg, opts = {}) {
    checkConfig(cfg);
    this.engine = new Rfd4pEngine(device);
    this.rawWeights = weights;
    this.cfg = cfg;
    this.half = opts.half ?? false;
    this.nlayers = cfg.nlayers;
    this.nTypes = cfg.n_residue_types ?? 147;
    this.outDim = this.nTypes + 111 + 37;
    this.na = 0;
    this.fdim = 0;
    this.w = this.uploadWeights(weights);
  }

  uploadWeights(src) {
    const e = this.engine;
    const get = (k) => {
      const t = src[k];
      if (!t) throw new Error(`rfd4p decoder: missing weight ${k}`);
      return t.data ?? t;
    };
    const opt = (k) => (src[k] ? src[k].data ?? src[k] : null);
    const cat = (...arrs) => {
      const out = new Float32Array(arrs.reduce((s, a) => s + a.length, 0));
      let o = 0;
      for (const a of arrs) {
        out.set(a, o);
        o += a.length;
      }
      return out;
    };
    const scaleHead = (h) => {
      const nw = get(`${PREFIX}.${h}.0.weight`);
      const nb = get(`${PREFIX}.${h}.0.bias`);
      const lw = get(`${PREFIX}.${h}.1.weight`);
      const outC = lw.length / D;
      const wScaled = new Float32Array(lw.length);
      const bEff = new Float32Array(outC);
      for (let o = 0; o < outC; o += 1) {
        const base = o * D;
        let acc = 0;
        for (let c = 0; c < D; c += 1) {
          const v = lw[base + c];
          wScaled[base + c] = v * nw[c];
          acc += v * nb[c];
        }
        bEff[o] = acc;
      }
      return { wScaled, bEff };
    };
    const hLogit = scaleHead('logit_linear');
    const hStruct = scaleHead('struct_linear');
    const hPres = scaleHead('atom_presence_linear');
    const padRowW = new Float32Array(D);
    const padRowB = new Float32Array(1);
    const w = {
      seqLin: e.weight(`${PREFIX}.init_repr_factory.linear_out.weight`, get(`${PREFIX}.init_repr_factory.linear_out.weight`), this.half, D),
      layers: [],
      headsFullLin: e.weight(
        `${PREFIX}.heads_full.lin`,
        cat(hLogit.wScaled, hStruct.wScaled, hPres.wScaled, padRowW),
        this.half,
      ),
      headsFullBias: e.weight(
        `${PREFIX}.heads_full.bias`,
        cat(hLogit.bEff, hStruct.bEff, hPres.bEff, padRowB),
        false,
      ),
      headsFastLin: e.weight(
        `${PREFIX}.heads_fast.lin`,
        cat(hStruct.wScaled, hPres.wScaled),
        this.half,
      ),
      headsFastBias: e.weight(
        `${PREFIX}.heads_fast.bias`,
        cat(hStruct.bEff, hPres.bEff),
        false,
      ),
    };
    for (let i = 0; i < this.nlayers; i += 1) {
      w.layers.push(uploadTokenLayer(e, get, opt, `${PREFIX}.transformer_layers.${i}`, this.half, C));
    }
    return w;
  }

  /** Allocate activations for `na` attended tokens and record the decoder program. */
  async record(na, fdim) {
    if (this.na === na && this.fdim === fdim) return;
    const e = this.engine;
    e.releaseActivations();
    this.na = na;
    this.fdim = fdim;
    const fdimPad = this.w.seqLin.stride ?? fdim;
    this.fdimPad = fdimPad;
    const OUT_PAD = Math.ceil(this.outDim / 4) * 4;
    this.outPad = OUT_PAD;
    const b = (label, floats) => e.buffer(label, floats);
    const B = (this.buf = {
      seqF: b('dec.seqF', na * fdimPad),
      cZero: b('dec.cZero', C),
      layerCond: b('dec.layerCond', this.nlayers * LAYER_COND_FLOATS),
      s: b('dec.s', na * D),
      ...tokenBuffers(e, na, C, 'dec', this.nlayers, 1),
      out: b('dec.out', na * OUT_PAD),
    });
    e.write(B.cZero, new Float32Array(C));
    const w = this.w;
    this.blocks = new TokenBlocks(e, na, 'ln', B, 1);
    e.run(e.program(await this.blocks.precomputeCond(w.layers, B.cZero, B.layerCond)));

    const steps = [
      await e.linear({ x: B.seqF, w: w.seqLin, y: B.s, rows: na, inC: fdimPad, outC: D }),
    ];
    for (let i = 0; i < this.nlayers; i += 1) {
      const nextL = i + 1 < this.nlayers ? w.layers[i + 1] : null;
      const nextNorm = nextL
        ? { type: 'ln', x: B.s, y: B.xn, rows: na, width: D, w: nextL.nodeNorm, b: nextL.nodeNormB,
            adaln: { gb: B.layerCond, off: (i + 1) * LAYER_COND_FLOATS, stride: 0, postNorm: true } }
        : { type: 'ln', x: B.s, y: B.xn, rows: na, width: D };
      steps.push(...(await this.blocks.layer(w.layers[i], B.s, B.layerCond, i, null, [], {
        skipFirstNorm: i > 0,
        nextNorm,
      })));
    }
    const fullHeads = [
      await e.linear({ x: B.xn, w: w.headsFullLin, y: B.out, rows: na, inC: D, outC: OUT_PAD, bias: w.headsFullBias }),
    ];
    const fastHeads = [
      await e.linear({ x: B.xn, w: w.headsFastLin, y: B.out, rows: na, inC: D, outC: 148, bias: w.headsFastBias }),
    ];
    this.decodeProgram = e.program([...steps, ...fullHeads]);
    this.decodeFastProgram = e.program([...steps, ...fastHeads]);
  }

  /**
   * Decode latents and CA coordinates (nm) to all-atom coordinates (nm), atom mask, and residue types.
   * @param {object} batch upstream-layout batch
   * @param {{n: number, caNm: Float32Array, z: Float32Array, seq: Int32Array}} o
   * @returns {Promise<{coorsNm: Float32Array, residueType: Int32Array, emitMask: Uint8Array, atomMask: Uint8Array}>}
   */
  async decode(batch, { n, caNm, z, seq }) {
    const { attn, emit } = aeMasks(batch, n);
    const idx = [];
    let allSeqKnown = true;
    for (let i = 0; i < n; i += 1) {
      if (attn[i]) idx.push(i);
      if (emit[i] && seq[i] === 0) allSeqKnown = false;
    }
    const na = idx.length;
    const nTypes = this.nTypes;
    const OUT = this.outDim;

    const logits = new Float32Array(n * nTypes);
    const coorsFlat = new Float32Array(n * 111);
    const presence = new Float32Array(n * 37);

    if (na > 0) {
      const { data: feats, dim: fdim } = decoderSeqFeatures(batch, {
        n, caNm, z, seq, weights: this.rawWeights, cfg: this.cfg,
      });
      await this.record(na, fdim);
      const fdimPad = this.fdimPad;
      const OUT_PAD = this.outPad;
      const featsA = new Float32Array(na * fdimPad);
      for (let r = 0; r < na; r += 1) {
        const i = idx[r];
        featsA.set(feats.subarray(i * fdim, (i + 1) * fdim), r * fdimPad);
      }
      this.engine.write(this.buf.seqF, featsA);
      if (allSeqKnown) {
        const outA = await this.engine.runAndRead([this.decodeFastProgram], this.buf.out, na * 148);
        for (let r = 0; r < na; r += 1) {
          const i = idx[r];
          const base = r * 148;
          coorsFlat.set(outA.subarray(base, base + 111), i * 111);
          presence.set(outA.subarray(base + 111, base + 148), i * 37);
        }
      } else {
        const outA = await this.engine.runAndRead([this.decodeProgram], this.buf.out, na * OUT_PAD);
        for (let r = 0; r < na; r += 1) {
          const i = idx[r];
          const base = r * OUT_PAD;
          logits.set(outA.subarray(base, base + nTypes), i * nTypes);
          coorsFlat.set(outA.subarray(base + nTypes, base + nTypes + 111), i * 111);
          presence.set(outA.subarray(base + nTypes + 111, base + OUT), i * 37);
        }
      }
    }

    return decoderPostprocess(batch, { n, caNm, seq, logits, coorsFlat, presence, nTypes });
  }

  destroy() {
    this.engine.destroy();
  }
}
