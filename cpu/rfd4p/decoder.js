/**
 * CPU reference for the RFD4-Proteina autoencoder decoder (`Autoencoder.decode` +
 * `DecoderTransformer.forward`, upstream `partial_autoencoder/`), on an upstream-layout batch.
 *
 * Also `sequenceForDecoder` (`Proteina._format_sample_local_latents`): with
 * `condition_decoder_on_sequence_prediction`, emitted tokens without a conditioned sequence
 * take the argmax of the trunk's allowed sequence logits as their `C_SEQ`.
 */

import {
  ALLOWED_RESIDUE_TYPES,
  NUM_TOKEN_TYPES,
  UNIFIED_RESTYPE_ATOM37_MASK,
} from '../../shared/rfd4p/encoding.js';
import { COND } from '../../shared/rfd4p/features.js';
import { layerNorm, linearLayer, multiheadAttnAndTransition, swiGluTransition } from './reference.js';

const MASK_TOKEN = 0; // `<M>`
const PREFIX = 'autoencoder.decoder';

function data(batch, key) {
  const v = batch[key];
  if (!v) throw new Error(`batch is missing '${key}'`);
  return v.data ?? v;
}

/** `compute_ae_masks`: attention over non-context tokens; emit tokens that carry latents. */
export function aeMasks(batch, n) {
  const pad = data(batch, 'padding_mask');
  const ctx = data(batch, 'is_in_context_conditioning');
  const lat = data(batch, 'supports_local_latent');
  const attn = new Uint8Array(n);
  const emit = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) {
    attn[i] = pad[i] && !ctx[i] ? 1 : 0;
    emit[i] = attn[i] && lat[i] ? 1 : 0;
  }
  return { attn, emit };
}

/**
 * The `C_SEQ` the decoder is conditioned on: user-conditioned residues kept, emitted
 * unconditioned tokens filled with the argmax of the token-type-allowed trunk logits.
 * @param {object} batch
 * @param {Float32Array} seqLogits `[n, 147]` trunk readout of the last step
 */
export function sequenceForDecoder(batch, n, seqLogits, nTypes = 147) {
  const seq = Int32Array.from(data(batch, COND.SEQ));
  const tokenType = data(batch, 'token_type');
  const { emit } = aeMasks(batch, n);
  for (let i = 0; i < n; i += 1) {
    if (!emit[i] || seq[i] !== MASK_TOKEN) continue;
    const allowed = ALLOWED_RESIDUE_TYPES[tokenType[i]];
    let best = 0;
    let bestV = -Infinity;
    for (let r = 0; r < nTypes; r += 1) {
      const v = allowed[r] ? seqLogits[i * nTypes + r] : -3.4028234663852886e38;
      if (v > bestV) {
        bestV = v;
        best = r;
      }
    }
    seq[i] = best;
  }
  return seq;
}

/**
 * Decode latents to all-atom coordinates.
 * @param {object} batch upstream-layout batch (needs padding/context/latent masks, token_type,
 *   `condition_CoordinateGroup_1_atom` for the required-atom mask)
 * @param {object} o
 * @param {number} o.n
 * @param {Float32Array} o.caNm `[n, 3]`
 * @param {Float32Array} o.z `[n, 32]`
 * @param {Int32Array} o.seq `[n]` decoder `C_SEQ` (see `sequenceForDecoder`)
 * @param {object} o.weights
 * @param {object} o.cfg bundle `config.autoencoder_config.nn_ae.decoder`
 * @returns {{coorsNm: Float32Array, residueType: Int32Array, emitMask: Uint8Array, atomMask: Uint8Array}}
 */
export function decodeLatentsCpu(batch, { n, caNm, z, seq, weights, cfg }) {
  const W = (k) => {
    const w = weights[`${PREFIX}.${k}`];
    if (!w) throw new Error(`missing weight ${PREFIX}.${k}`);
    return w.data ?? w;
  };
  const D = cfg.token_dim;
  const C = cfg.dim_cond;
  const nTypes = cfg.n_residue_types ?? 147;
  const { attn } = aeMasks(batch, n);

  // c: empty cond factory -> ZeroFeat(128) -> two bias-free SwiGLU transitions (stays zero).
  let c = new Float32Array(n * C);
  c = swiGluTransition(c, attn, n, C, 2, weights, `${PREFIX}.transition_c_1`);
  c = swiGluTransition(c, attn, n, C, 2, weights, `${PREFIX}.transition_c_2`);

  const { data: feats, dim: fdim } = decoderSeqFeatures(batch, { n, caNm, z, seq, weights, cfg });
  let s = linearLayer(feats, n, fdim, W('init_repr_factory.linear_out.weight'), D);
  for (let i = 0; i < n; i += 1) if (!attn[i]) s.fill(0, i * D, (i + 1) * D);

  for (let l = 0; l < cfg.nlayers; l += 1) {
    s = multiheadAttnAndTransition(s, null, c, attn, n, D, cfg.pair_repr_dim, cfg.nheads, C, false, cfg.use_qkln ?? true,
      'layer_norm', weights, `${PREFIX}.transformer_layers.${l}`);
  }

  const head = (name, out) => {
    const h = layerNorm(s, n, D, W(`${name}.0.weight`), W(`${name}.0.bias`));
    return linearLayer(h, n, D, W(`${name}.1.weight`), out);
  };
  const logits = head('logit_linear', nTypes);
  const coorsFlat = head('struct_linear', 111);
  const presence = head('atom_presence_linear', 37);

  return decoderPostprocess(batch, { n, caNm, seq, logits, coorsFlat, presence, nTypes });
}


/**
 * Decoder input features (`init_repr_factory`, `feats_seq` order: ca_coors_nm, z_latent_seq, token_type,
 * cond_residue_type), `[n, 214]`, before `linear_out`.
 */
export function decoderSeqFeatures(batch, { n, caNm, z, seq, weights, cfg }) {
  const W = (k) => {
    const w = weights[`${PREFIX}.${k}`];
    if (!w) throw new Error(`missing weight ${PREFIX}.${k}`);
    return w.data ?? w;
  };
  const nTypes = cfg.n_residue_types ?? 147;
  const latentDim = cfg.latent_z_dim ?? 32;
  const tokenType = data(batch, 'token_type');
  const dims = { ca_coors_nm: 3, z_latent_seq: latentDim, token_type: 32, cond_residue_type: nTypes };
  const fdim = cfg.feats_seq.reduce((s, f) => s + dims[f], 0);
  const feats = new Float32Array(n * fdim);
  for (let i = 0; i < n; i += 1) {
    if (!data(batch, 'padding_mask')[i]) continue;
    let o = i * fdim;
    cfg.feats_seq.forEach((f, idx) => {
      if (f === 'ca_coors_nm') for (let k = 0; k < 3; k += 1) feats[o + k] = caNm[i * 3 + k];
      else if (f === 'z_latent_seq') for (let k = 0; k < latentDim; k += 1) feats[o + k] = z[i * latentDim + k];
      else if (f === 'token_type') {
        const tab = W(`init_repr_factory.feat_creators.${idx}.embedding.weight`);
        for (let k = 0; k < 32; k += 1) feats[o + k] = tab[tokenType[i] * 32 + k];
      } else if (f === 'cond_residue_type') {
        if (seq[i] !== MASK_TOKEN) {
          const tab = W(`init_repr_factory.feat_creators.${idx}.embedding.weight`);
          for (let k = 0; k < nTypes; k += 1) feats[o + k] = tab[seq[i] * nTypes + k];
        }
      } else throw new Error(`decoder feature ${f} not implemented`);
      o += dims[f];
    });
  }
  return { data: feats, dim: fdim };
}

/**
 * Decoder heads -> structure (`DecoderTransformer.forward` tail + `Autoencoder.decode`): allowed-type argmax unless
 * `C_SEQ` is set, legal & present atom slots (slot 1 always; coordinate-group atoms required), CA-relative coords.
 * @param {object} batch
 * @param {object} o `logits [n,147]`, `coorsFlat [n,111]`, `presence [n,37]` from the heads
 */
export function decoderPostprocess(batch, { n, caNm, seq, logits, coorsFlat, presence, nTypes = 147 }) {
  const { emit } = aeMasks(batch, n);
  const tokenType = data(batch, 'token_type');
  const groups = batch[COND.CRDGRP] ? data(batch, COND.CRDGRP) : null;
  const residueType = new Int32Array(n);
  const atomMask = new Uint8Array(n * 37);
  const coorsNm = new Float32Array(n * 111);
  for (let i = 0; i < n; i += 1) {
    if (!emit[i]) continue;
    const tt = Math.max(0, Math.min(NUM_TOKEN_TYPES - 1, tokenType[i]));
    const allowed = ALLOWED_RESIDUE_TYPES[tt];
    let best = 0;
    let bestV = -Infinity;
    for (let r = 0; r < nTypes; r += 1) {
      const v = allowed[r] ? logits[i * nTypes + r] : -3.4028234663852886e38;
      if (v > bestV) {
        bestV = v;
        best = r;
      }
    }
    const rt = seq[i] !== MASK_TOKEN ? seq[i] : best;
    residueType[i] = rt;
    const legal = UNIFIED_RESTYPE_ATOM37_MASK[rt];
    for (let a = 0; a < 37; a += 1) {
      const required = groups ? groups[i * 37 + a] >= 0 : false;
      const present = presence[i * 37 + a] > 0 || a === 1 || required;
      if (!(legal[a] && present)) continue;
      atomMask[i * 37 + a] = 1;
      for (let k = 0; k < 3; k += 1) {
        const rel = a === 1 ? 0 : coorsFlat[i * 111 + a * 3 + k];
        coorsNm[i * 111 + a * 3 + k] = Math.fround(rel + caNm[i * 3 + k]);
      }
    }
  }
  return { coorsNm, residueType, emitMask: emit, atomMask };
}

/**
 * `Proteina._format_sample_local_latents` tail: convert decoded nm output into Angstrom `coors [n, 37, 3]`,
 * `atom_mask [n, 37]`, and `residue_type [n]`, placing sampled CA in slot 1 and setting `residue_type = -1`
 * for non-emitted tokens.
 */
export function formatSampleLocalLatents(batch, { n, caNm, decoded }) {
  const pad = data(batch, 'padding_mask');
  const { emitMask } = decoded;
  const coors = new Float32Array(n * 111);
  const atomMask = Uint8Array.from(decoded.atomMask);
  const residueType = Int32Array.from(decoded.residueType);

  for (let i = 0; i < n; i += 1) {
    if (emitMask[i]) {
      const base = i * 111;
      for (let k = 0; k < 111; k += 1) {
        coors[base + k] = Math.fround(decoded.coorsNm[base + k] * 10);
      }
    } else {
      residueType[i] = -1;
      if (pad[i]) {
        atomMask[i * 37 + 1] = 1;
        const base = i * 111 + 3; // slot 1
        for (let k = 0; k < 3; k += 1) {
          coors[base + k] = Math.fround(caNm[i * 3 + k] * 10);
        }
      }
    }
  }
  return { coors, atomMask, residueType };
}
