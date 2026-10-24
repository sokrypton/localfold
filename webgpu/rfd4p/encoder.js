/**
 * The RFD4-Proteina autoencoder encoder (`AutoEncoder.encode` + `EncoderTransformer.forward`) on WebGPU.
 *
 * Mirrors `cpu/rfd4p/encoder.js` and `webgpu/rfd4p/decoder.js`:
 *   - `c` is identically zero (`dim_cond = 128`, bias-free transitions on zero features).
 *   - Only attended tokens (`attn = padding_mask & ~is_in_context_conditioning`) participate in self-attention
 *     or are read by `encoderPostprocess` (`emit = attn & supports_local_latent`). Gathering the `na` attended
 *     rows and running unmasked attention over `na` tokens is exact.
 *   - Input features `[n, 1587]` are assembled on CPU by `encoderSeqFeatures`, gathered to `[na, 1587]`,
 *     projected by `init_repr_factory.linear_out` (kept fp32 since 1587 is odd), transformed by 12 LayerNorm
 *     `TokenBlocks` layers, and projected by `latent_decoder_mean_n_log_scale` (`[na, 64]`).
 */

import { aeMasks } from '../../cpu/rfd4p/decoder.js';
import { encoderPostprocess, encoderSeqFeatures } from '../../cpu/rfd4p/encoder.js';
import { LAYER_COND_FLOATS, TokenBlocks, tokenBuffers, uploadTokenLayer } from './blocks.js';
import { Rfd4pEngine } from './engine.js';

const PREFIX = 'autoencoder.encoder';
const D = 768;
const C = 128;
const H = 12;

function checkConfig(cfg) {
  const want = { token_dim: D, dim_cond: C, nheads: H };
  for (const [k, v] of Object.entries(want)) {
    if (cfg[k] !== v) throw new Error(`rfd4p encoder: config ${k}=${cfg[k]}, kernels are built for ${v}`);
  }
}

export class Rfd4pEncoderGpu {
  /**
   * @param {GPUDevice} device
   * @param {object} weights name -> {data: Float32Array, shape}
   * @param {object} cfg bundle `config.autoencoder_config.nn_ae.encoder`
   * @param {{half?: boolean}} [opts]
   */
  constructor(device, weights, cfg, opts = {}) {
    checkConfig(cfg);
    this.engine = new Rfd4pEngine(device);
    this.rawWeights = weights;
    this.cfg = cfg;
    this.half = opts.half ?? false;
    this.nlayers = cfg.nlayers;
    this.latentDim = cfg.latent_z_dim ?? 32;
    this.outDim = 2 * this.latentDim;
    this.na = 0;
    this.fdim = 0;
    this.w = this.uploadWeights(weights);
  }

  uploadWeights(src) {
    const e = this.engine;
    const get = (k) => {
      const t = src[k];
      if (!t) throw new Error(`rfd4p encoder: missing weight ${k}`);
      return t.data ?? t;
    };
    const opt = (k) => (src[k] ? src[k].data ?? src[k] : null);
    const mat = (k) => e.weight(k, get(k), this.half);
    const vec = (k) => e.weight(k, get(k), false);
    const w = {
      seqLin: e.weight(`${PREFIX}.init_repr_factory.linear_out.weight`, get(`${PREFIX}.init_repr_factory.linear_out.weight`), this.half, D),
      layers: [],
      head: {
        normW: vec(`${PREFIX}.latent_decoder_mean_n_log_scale.0.weight`),
        normB: vec(`${PREFIX}.latent_decoder_mean_n_log_scale.0.bias`),
        lin: mat(`${PREFIX}.latent_decoder_mean_n_log_scale.1.weight`),
      },
    };
    for (let i = 0; i < this.nlayers; i += 1) {
      w.layers.push(uploadTokenLayer(e, get, opt, `${PREFIX}.transformer_layers.${i}`, this.half, C));
    }
    return w;
  }

  /** Allocate activations for `na` attended tokens and record the encoder program. */
  async record(na, fdim) {
    if (this.na === na && this.fdim === fdim) return;
    const e = this.engine;
    e.releaseActivations();
    this.na = na;
    this.fdim = fdim;
    const fdimPad = this.w.seqLin.stride ?? fdim;
    this.fdimPad = fdimPad;
    const OUT = this.outDim;
    const b = (label, floats) => e.buffer(label, floats);
    const B = (this.buf = {
      seqF: b('enc.seqF', na * fdimPad),
      cZero: b('enc.cZero', C),
      layerCond: b('enc.layerCond', this.nlayers * LAYER_COND_FLOATS),
      s: b('enc.s', na * D),
      ...tokenBuffers(e, na, C, 'enc', this.nlayers, 1),
      out: b('enc.out', na * OUT),
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
        : { type: 'ln', x: B.s, y: B.xn, rows: na, width: D, w: w.head.normW, b: w.head.normB };
      steps.push(...(await this.blocks.layer(w.layers[i], B.s, B.layerCond, i, null, [], {
        skipFirstNorm: i > 0,
        nextNorm,
      })));
    }
    steps.push(
      await e.linear({ x: B.xn, w: w.head.lin, y: B.out, rows: na, inC: D, outC: OUT }),
    );
    this.encodeProgram = e.program(steps);
  }

  /**
   * Encode all-atom coordinates and token features to local latents `[n, 32]`.
   * @param {object} batch upstream-layout batch
   * @param {{n: number, rng?: {randn: (k: number) => ArrayLike<number>}, deterministic?: boolean}} o
   * @returns {Promise<{mean: Float32Array, logScale: Float32Array, zLatent: Float32Array, emitMask: Uint8Array}>}
   */
  async encode(batch, { n, rng, deterministic = false }) {
    const { attn } = aeMasks(batch, n);
    const idx = [];
    for (let i = 0; i < n; i += 1) if (attn[i]) idx.push(i);
    const na = idx.length;
    const OUT = this.outDim;
    const flatOut = new Float32Array(n * OUT);

    if (na > 0) {
      const { data: feats, dim: fdim } = encoderSeqFeatures(batch, {
        n, weights: this.rawWeights, cfg: this.cfg,
      });
      await this.record(na, fdim);
      const fdimPad = this.fdimPad;
      const featsA = new Float32Array(na * fdimPad);
      for (let r = 0; r < na; r += 1) {
        const i = idx[r];
        featsA.set(feats.subarray(i * fdim, (i + 1) * fdim), r * fdimPad);
      }
      this.engine.write(this.buf.seqF, featsA);
      const outA = await this.engine.runAndRead([this.encodeProgram], this.buf.out, na * OUT);
      for (let r = 0; r < na; r += 1) {
        const i = idx[r];
        flatOut.set(outA.subarray(r * OUT, (r + 1) * OUT), i * OUT);
      }
    }

    return encoderPostprocess(batch, {
      n, flatOut, rng, latentDim: this.latentDim, deterministic,
    });
  }

  destroy() {
    this.engine.destroy();
  }
}
