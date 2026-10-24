/**
 * CPU reference for the RFD4-Proteina autoencoder encoder (`AutoEncoder.encode` +
 * `EncoderTransformer.forward`, upstream `partial_autoencoder/`), on an upstream-layout batch.
 */

import { aeMasks } from './decoder.js';
import { layerNorm, linearLayer, multiheadAttnAndTransition, swiGluTransition } from './reference.js';

const PREFIX = 'autoencoder.encoder';

function data(batch, key) {
  const v = batch[key];
  if (!v) throw new Error(`batch is missing '${key}'`);
  return v.data ?? v;
}

/**
 * Encoder input features (`init_repr_factory`, `feats_seq` order: `token_type`, `x1_a37coors_nm`,
 * `x1_a37coors_nm_rel`, `chain_idx_seq`, `incident_bond_type_counts`,
 * `atom37_element_charge_embedding`), `[n, 1587]`, before `linear_out`.
 */
export function encoderSeqFeatures(batch, { n, weights, cfg }) {
  const W = (k) => {
    const w = weights[`${PREFIX}.${k}`];
    if (!w) throw new Error(`missing weight ${PREFIX}.${k}`);
    return w.data ?? w;
  };
  const pad = data(batch, 'padding_mask');
  const tokenType = data(batch, 'token_type');
  const atomMask = batch.atom37_mask ? data(batch, 'atom37_mask') : data(batch, 'coord_mask');
  const coorsNm = batch.coords_nm
    ? data(batch, 'coords_nm')
    : Float32Array.from(data(batch, 'coords'), (v) => Math.fround(v / 10));
  const chainIid = data(batch, 'chain_iid');
  const bondCounts = data(batch, 'incident_bond_type_counts');
  const gtElement = data(batch, 'ground_truth_element');
  const charge = data(batch, 'charge');

  const ttDim = cfg.token_type_emb_dim ?? 32;
  const chargeDim = cfg.charge_emb_dim ?? 16;
  const elChargeSlotDim = 8 + chargeDim;
  const dims = {
    token_type: ttDim,
    x1_a37coors_nm: 37 * 4,
    x1_a37coors_nm_rel: 37 * 4,
    chain_idx_seq: 1,
    incident_bond_type_counts: 37 * 10,
    atom37_element_charge_embedding: 37 * elChargeSlotDim,
  };
  const fdim = cfg.feats_seq.reduce((s, f) => s + dims[f], 0);
  const feats = new Float32Array(n * fdim);

  for (let i = 0; i < n; i += 1) {
    if (!pad[i]) continue;
    let o = i * fdim;
    cfg.feats_seq.forEach((f, idx) => {
      if (f === 'token_type') {
        const tab = W(`init_repr_factory.feat_creators.${idx}.embedding.weight`);
        const tt = tokenType[i];
        for (let k = 0; k < ttDim; k += 1) feats[o + k] = tab[tt * ttDim + k];
      } else if (f === 'x1_a37coors_nm' || f === 'x1_a37coors_nm_rel') {
        const rel = f === 'x1_a37coors_nm_rel';
        const caMask = atomMask[i * 37 + 1] ? 1 : 0;
        const caBase = (i * 37 + 1) * 3;
        const ca0 = rel && caMask ? coorsNm[caBase] : 0;
        const ca1 = rel && caMask ? coorsNm[caBase + 1] : 0;
        const ca2 = rel && caMask ? coorsNm[caBase + 2] : 0;
        for (let a = 0; a < 37; a += 1) {
          const m = atomMask[i * 37 + a] ? 1 : 0;
          if (m) {
            const b = (i * 37 + a) * 3;
            feats[o + a * 3] = Math.fround(coorsNm[b] - ca0);
            feats[o + a * 3 + 1] = Math.fround(coorsNm[b + 1] - ca1);
            feats[o + a * 3 + 2] = Math.fround(coorsNm[b + 2] - ca2);
          }
          feats[o + 111 + a] = m;
        }
      } else if (f === 'chain_idx_seq') {
        feats[o] = chainIid[i];
      } else if (f === 'incident_bond_type_counts') {
        for (let a = 0; a < 37; a += 1) {
          if (!atomMask[i * 37 + a]) continue;
          const b = (i * 37 + a) * 10;
          for (let k = 0; k < 10; k += 1) feats[o + a * 10 + k] = bondCounts[b + k];
        }
      } else if (f === 'atom37_element_charge_embedding') {
        const elTab = W(`init_repr_factory.feat_creators.${idx}.element_embedding.weight`);
        const chTab = W(`init_repr_factory.feat_creators.${idx}.charge_embedding.weight`);
        for (let a = 0; a < 37; a += 1) {
          if (!atomMask[i * 37 + a]) continue;
          const slot = i * 37 + a;
          const el = gtElement[slot];
          const ch = Math.min(10, Math.max(0, charge[slot] + 4));
          const dst = o + a * elChargeSlotDim;
          for (let k = 0; k < 8; k += 1) feats[dst + k] = elTab[el * 8 + k];
          for (let k = 0; k < chargeDim; k += 1) feats[dst + 8 + k] = chTab[ch * chargeDim + k];
        }
      } else {
        throw new Error(`encoder feature ${f} not implemented`);
      }
      o += dims[f];
    });
  }
  return { data: feats, dim: fdim };
}

/**
 * Encoder head postprocessing (`EncoderTransformer.forward` tail): split `flatOut [n, 2 * latentDim]`
 * into `mean` and `logScale` masked by `emit`, then sample `zLatent = (mean + eps * exp(logScale)) * emit`.
 */
export function encoderPostprocess(batch, { n, flatOut, rng, latentDim = 32, deterministic = false }) {
  const { emit } = aeMasks(batch, n);
  const mean = new Float32Array(n * latentDim);
  const logScale = new Float32Array(n * latentDim);
  const zLatent = new Float32Array(n * latentDim);
  const eps = !deterministic && rng ? rng.randn(n * latentDim) : null;

  for (let i = 0; i < n; i += 1) {
    if (!emit[i]) continue;
    const inBase = i * 2 * latentDim;
    const outBase = i * latentDim;
    for (let k = 0; k < latentDim; k += 1) {
      const m = flatOut[inBase + k];
      const ls = flatOut[inBase + latentDim + k];
      mean[outBase + k] = m;
      logScale[outBase + k] = ls;
      zLatent[outBase + k] = eps
        ? Math.fround(m + Math.fround(eps[outBase + k] * Math.fround(Math.exp(ls))))
        : m;
    }
  }
  return { mean, logScale, zLatent, emitMask: emit };
}

/**
 * Encode all-atom coordinates and metadata into token-level local latents `[n, 32]`.
 * @param {object} batch upstream-layout batch
 * @param {object} o
 * @param {number} o.n
 * @param {object} o.weights
 * @param {object} o.cfg bundle `config.autoencoder_config.nn_ae.encoder`
 * @param {{randn: (k: number) => ArrayLike<number>}} [o.rng]
 * @param {boolean} [o.deterministic]
 * @returns {{mean: Float32Array, logScale: Float32Array, zLatent: Float32Array, emitMask: Uint8Array}}
 */
export function encodeLatentsCpu(batch, { n, weights, cfg, rng, deterministic = false }) {
  const W = (k) => {
    const w = weights[`${PREFIX}.${k}`];
    if (!w) throw new Error(`missing weight ${PREFIX}.${k}`);
    return w.data ?? w;
  };
  const D = cfg.token_dim;
  const C = cfg.dim_cond;
  const latentDim = cfg.latent_z_dim ?? 32;
  const { attn } = aeMasks(batch, n);

  // c: empty cond factory -> ZeroFeat(128) -> two bias-free SwiGLU transitions (stays zero).
  let c = new Float32Array(n * C);
  c = swiGluTransition(c, attn, n, C, 2, weights, `${PREFIX}.transition_c_1`);
  c = swiGluTransition(c, attn, n, C, 2, weights, `${PREFIX}.transition_c_2`);

  const { data: feats, dim: fdim } = encoderSeqFeatures(batch, { n, weights, cfg });
  let s = linearLayer(feats, n, fdim, W('init_repr_factory.linear_out.weight'), D);
  for (let i = 0; i < n; i += 1) if (!attn[i]) s.fill(0, i * D, (i + 1) * D);

  for (let l = 0; l < cfg.nlayers; l += 1) {
    s = multiheadAttnAndTransition(
      s, null, c, attn, n, D, cfg.pair_repr_dim, cfg.nheads, C, false, cfg.use_qkln ?? true,
      'layer_norm', weights, `${PREFIX}.transformer_layers.${l}`,
    );
  }

  const h = layerNorm(
    s, n, D,
    W('latent_decoder_mean_n_log_scale.0.weight'),
    W('latent_decoder_mean_n_log_scale.0.bias'),
  );
  const flatOut = linearLayer(h, n, D, W('latent_decoder_mean_n_log_scale.1.weight'), 2 * latentDim);

  return encoderPostprocess(batch, { n, flatOut, rng, latentDim, deterministic });
}
