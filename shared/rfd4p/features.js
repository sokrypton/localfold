/**
 * Feature construction for RFD4-Proteina, ported from upstream
 * `src/rfproteina/nn/feature_factory.py`.
 *
 * Input is an upstream-layout batch: a map from upstream batch key (e.g. `token_type`,
 * `condition_hbond_1_atom`, `token_bond_path_length`) to `{ data, shape }`, with the leading
 * batch axis of size 1 (or no batch axis). Feature order and widths come from the bundle
 * `config.json` feature lists, so the concatenation layout matches each factory's `linear_out`.
 *
 * Every factory output is the pre-`linear_out` feature matrix (upstream concatenates, applies the
 * padding mask, then projects): seq `[n, d]`, pair `[n*n, d]`, compact per-example `[d]`.
 */

import { STANDARD_POLYMER_RESIDUES, UNIFIED_ATOM37_TOKENS } from './encoding.js';

/** Upstream condition annotation keys as they appear in the model batch. */
export const COND = {
  CRD: 'condition_coordinate_1_atom',
  INTCRD: 'condition_InternalCoordinate_1_atom',
  CRDGRP: 'condition_CoordinateGroup_1_atom',
  IDX: 'condition_index_1_residue',
  IDXGRP: 'condition_index-group_1_token',
  SEQ: 'condition_sequence_1_residue',
  CATH: 'condition_cath-label_1_residue',
  CENTROID: 'condition_ChainCentroid_1_atom',
  RAS: 'condition_rasa-binned_1_atom',
  SEC: 'condition_secondary-structure_1_residue',
  HB: 'condition_hbond_1_atom',
  PISTACK: 'condition_pistack_1_atom',
  PICATION: 'condition_pication_1_atom',
  SALT: 'condition_saltbridge_1_atom',
  HPHOB: 'condition_hydrophobic_1_atom',
  MCOORD: 'condition_metal-coord_1_atom',
  HOT: 'condition_hotspot_1_atom',
};

/** Unified-encoding tokens with a sequence-probability channel, in channel order. */
export const SEQUENCE_PROBABILITY_TOKENS = UNIFIED_ATOM37_TOKENS.filter((tok) =>
  STANDARD_POLYMER_RESIDUES.includes(tok)
);

export const seqProbKey = (tok) => `condition_sequence-probability-${tok}_1_residue`;

const TOKEN_TYPE_ATOMIZED = 3;
const MAX_CHIRAL_SLOTS = 6;
const RASA_UNSPECIFIED = 0;
const CRDGRP_UNKNOWN = -1;
const CRDGRP_ABSOLUTE = 0;

function need(batch, key) {
  const v = batch[key];
  if (!v) throw new Error(`batch is missing required key '${key}'`);
  return v.data ?? v;
}

function optional(batch, key) {
  const v = batch[key];
  return v ? v.data ?? v : null;
}

function weight(weights, name) {
  const w = weights[name];
  if (!w) throw new Error(`missing weight ${name}`);
  return w.data ?? w;
}

/**
 * Sinusoidal time embedding (`get_time_embedding`, max_positions 2000): `[sin, cos]`.
 * @param {number} t
 * @param {number} dim
 */
export function timeEmbedding(t, dim) {
  const half = dim >> 1;
  const out = new Float32Array(dim);
  const scale = Math.log(2000) / (half - 1);
  const tt = Math.fround(t * 2000);
  for (let i = 0; i < half; i += 1) {
    const freq = Math.fround(Math.exp(Math.fround(-scale * i)));
    const a = Math.fround(tt * freq);
    out[i] = Math.sin(a);
    out[half + i] = Math.cos(a);
  }
  return out;
}

/** Copy embedding row `idx` of a `[rows, dim]` table into `dst` at `off`. */
function embedRow(table, dim, idx, dst, off) {
  for (let k = 0; k < dim; k += 1) dst[off + k] = table[idx * dim + k];
}

/**
 * 3x3 proper rotation `R` maximising `tr(R^T C)` (row-vector convention `src @ R ~ tgt`,
 * `C = src^T tgt`), via Horn's quaternion method. Matches upstream `proper_rotation` (SVD) for
 * non-degenerate fits.
 * @param {Float64Array|number[]} C row-major 3x3
 */
export function properRotation(C) {
  const [Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz] = C;
  const N = [
    [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
    [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
    [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
    [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
  ];
  const { values, vectors } = jacobiEigen4(N);
  let best = 0;
  for (let i = 1; i < 4; i += 1) if (values[i] > values[best]) best = i;
  const [w, x, y, z] = [vectors[0][best], vectors[1][best], vectors[2][best], vectors[3][best]];
  // Column-vector rotation Q (tgt ~ Q src); the row-vector rotation is R = Q^T.
  const Q = [
    [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
  ];
  return [Q[0][0], Q[1][0], Q[2][0], Q[0][1], Q[1][1], Q[2][1], Q[0][2], Q[1][2], Q[2][2]];
}

function jacobiEigen4(A0) {
  const A = A0.map((r) => r.slice());
  const V = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep += 1) {
    let off = 0;
    for (let p = 0; p < 4; p += 1) for (let q = p + 1; q < 4; q += 1) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < 4; p += 1) {
      for (let q = p + 1; q < 4; q += 1) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k += 1) {
          const akp = A[k][p];
          const akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k += 1) {
          const apk = A[p][k];
          const aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k += 1) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: [A[0][0], A[1][1], A[2][2], A[3][3]], vectors: V };
}

/**
 * `align_chirality` (rotate only, the released config): rotate each chiral frame's packed
 * vectors onto the directions of its neighbours in the noisy CA state.
 * @returns {Float32Array} `[n, 18]`
 */
export function alignChirality(batch, n, xtBbCa, cache = null) {
  const chir = need(batch, 'chirality');
  const mask = need(batch, 'padding_mask');
  const frames = need(batch, 'chiral_frame_atoms'); // [n, S, 2+K]
  const owners = need(batch, 'chiral_slot_owner'); // [n, S]
  const S = MAX_CHIRAL_SLOTS;
  const F = batch.chiral_frame_atoms.shape.at(-1);
  const K = F - 2;
  if (cache && !cache.chiralInit) {
    const activeFrames = [];
    for (let ci = 0; ci < n; ci += 1) {
      if (!mask[ci]) continue;
      for (let b = 0; b < S; b += 1) {
        const fr = (ci * S + b) * F;
        if (frames[fr] < 0) continue;
        const nbs = [];
        for (let k = 0; k < K; k += 1) {
          const nb = frames[fr + 2 + k];
          if (nb < 0) continue;
          nbs.push({ nb, sx: chir[nb * 18 + b * 3], sy: chir[nb * 18 + b * 3 + 1], sz: chir[nb * 18 + b * 3 + 2] });
        }
        activeFrames.push({ id: ci * S + b, o0: frames[fr], o1: frames[fr + 1], nbs });
      }
    }
    const activeOwners = [];
    for (let i = 0; i < n; i += 1) {
      if (!mask[i]) continue;
      for (let s = 0; s < S; s += 1) {
        const owner = owners[i * S + s];
        if (owner < 0) continue;
        const o = i * 18 + s * 3;
        activeOwners.push({ o, owner, vx: chir[o], vy: chir[o + 1], vz: chir[o + 2] });
      }
    }
    const chirOut = Float32Array.from(chir);
    for (let i = 0; i < n; i += 1) if (!mask[i]) chirOut.fill(0, i * 18, i * 18 + 18);
    cache.chiralInit = true;
    cache.activeFrames = activeFrames;
    cache.activeOwners = activeOwners;
    cache.chirOut = chirOut;
    cache.rotArr = new Array(n * S);
    cache.chirC = new Float64Array(9);
  }
  if (cache && cache.chiralInit) {
    const { activeFrames, activeOwners, chirOut: out, rotArr, chirC: C } = cache;
    for (let fIdx = 0; fIdx < activeFrames.length; fIdx += 1) {
      const f = activeFrames[fIdx];
      const ox = (xtBbCa[f.o0 * 3] + xtBbCa[f.o1 * 3]) / 2;
      const oy = (xtBbCa[f.o0 * 3 + 1] + xtBbCa[f.o1 * 3 + 1]) / 2;
      const oz = (xtBbCa[f.o0 * 3 + 2] + xtBbCa[f.o1 * 3 + 2]) / 2;
      C.fill(0);
      const { nbs } = f;
      for (let k = 0; k < nbs.length; k += 1) {
        const { nb, sx, sy, sz } = nbs[k];
        let tx = xtBbCa[nb * 3] - ox;
        let ty = xtBbCa[nb * 3 + 1] - oy;
        let tz = xtBbCa[nb * 3 + 2] - oz;
        const norm = Math.max(Math.hypot(tx, ty, tz), 1e-8);
        tx /= norm;
        ty /= norm;
        tz /= norm;
        C[0] += sx * tx; C[1] += sx * ty; C[2] += sx * tz;
        C[3] += sy * tx; C[4] += sy * ty; C[5] += sy * tz;
        C[6] += sz * tx; C[7] += sz * ty; C[8] += sz * tz;
      }
      rotArr[f.id] = properRotation(C);
    }
    for (let aIdx = 0; aIdx < activeOwners.length; aIdx += 1) {
      const { o, owner, vx, vy, vz } = activeOwners[aIdx];
      const R = rotArr[owner];
      if (!R) continue;
      out[o] = vx * R[0] + vy * R[3] + vz * R[6];
      out[o + 1] = vx * R[1] + vy * R[4] + vz * R[7];
      out[o + 2] = vx * R[2] + vy * R[5] + vz * R[8];
    }
    return out;
  }
  const out = Float32Array.from(chir);
  const rotations = new Map(); // owner id (token*S + slot) -> R
  for (let ci = 0; ci < n; ci += 1) {
    if (!mask[ci]) continue;
    for (let b = 0; b < S; b += 1) {
      const fr = (ci * S + b) * F;
      if (frames[fr] < 0) continue;
      const o0 = frames[fr];
      const o1 = frames[fr + 1];
      const ox = (xtBbCa[o0 * 3] + xtBbCa[o1 * 3]) / 2;
      const oy = (xtBbCa[o0 * 3 + 1] + xtBbCa[o1 * 3 + 1]) / 2;
      const oz = (xtBbCa[o0 * 3 + 2] + xtBbCa[o1 * 3 + 2]) / 2;
      const C = new Float64Array(9);
      for (let k = 0; k < K; k += 1) {
        const nb = frames[fr + 2 + k];
        if (nb < 0) continue;
        const sv = [chir[nb * 18 + b * 3], chir[nb * 18 + b * 3 + 1], chir[nb * 18 + b * 3 + 2]];
        let tx = xtBbCa[nb * 3] - ox;
        let ty = xtBbCa[nb * 3 + 1] - oy;
        let tz = xtBbCa[nb * 3 + 2] - oz;
        const norm = Math.max(Math.hypot(tx, ty, tz), 1e-8);
        tx /= norm;
        ty /= norm;
        tz /= norm;
        const tv = [tx, ty, tz];
        for (let i = 0; i < 3; i += 1) for (let j = 0; j < 3; j += 1) C[i * 3 + j] += sv[i] * tv[j];
      }
      rotations.set(ci * S + b, properRotation(C));
    }
  }
  for (let i = 0; i < n; i += 1) {
    for (let s = 0; s < S; s += 1) {
      const owner = owners[i * S + s];
      if (owner < 0) continue;
      const R = rotations.get(owner);
      if (!R) continue;
      const o = i * 18 + s * 3;
      const v = [chir[o], chir[o + 1], chir[o + 2]];
      for (let j = 0; j < 3; j += 1) out[o + j] = v[0] * R[j] + v[1] * R[3 + j] + v[2] * R[6 + j];
    }
  }
  for (let i = 0; i < n; i += 1) if (!mask[i]) out.fill(0, i * 18, i * 18 + 18);
  return out;
}

/**
 * Relabel the non-`UNKNOWN` coordinate groups through a permutation of their sorted unique ids
 * (`CoordGroupSeqFeat`); `perm` is the recorded `torch.randperm` draw (or a fresh one).
 * @param {ArrayLike<number>} groups `[n, 37]`
 * @param {(k: number) => ArrayLike<number>} randperm
 */
export function permuteCoordGroups(groups, randperm) {
  const uniq = [...new Set(Array.from(groups).filter((g) => g !== CRDGRP_UNKNOWN))].sort((a, b) => a - b);
  const perm = randperm(uniq.length);
  const map = new Map(uniq.map((g, i) => [g, perm[i]]));
  return Float32Array.from(groups, (g) => (g === CRDGRP_UNKNOWN ? CRDGRP_UNKNOWN : map.get(g)));
}

/** Default `randperm` (Fisher–Yates over `Math.random`) for interactive use. */
export function defaultRandperm(k) {
  const p = Array.from({ length: k }, (_, i) => i);
  for (let i = k - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  return p;
}

/**
 * Trunk sequence features (`nn.init_repr_factory`, `feats_seq`).
 * @param {object} batch upstream-layout batch
 * @param {object} opts
 * @param {number} opts.n
 * @param {{bb_ca: Float32Array, local_latents: Float32Array}} opts.xt
 * @param {string[]} opts.feats `config.nn.feats_seq`
 * @param {object} opts.weights
 * @param {string} [opts.prefix]
 * @param {(k: number) => ArrayLike<number>} [opts.randperm]
 * @param {boolean} [opts.noiseAlignChirality]
 * @returns {{ data: Float32Array, dim: number }}
 */
export function trunkSeqFeatures(batch, opts) {
  const { n, xt, feats, weights, prefix = 'nn.init_repr_factory', randperm = defaultRandperm, cache = null, rowStride = 0 } = opts;
  const noiseAlignChirality = opts.noiseAlignChirality ?? true;
  const mask = need(batch, 'padding_mask');

  if (cache && cache.seqData && cache.seqN === n && cache.seqStride === (rowStride || cache.seqDim)) {
    const { seqData: data, seqDim: dim, seqStride: stride, caOff, latOff, grpOff, chirOff, crdUniq, crdGroups } = cache;
    const { bb_ca: ca, local_latents: lat } = xt;
    let permMap = null;
    if (grpOff >= 0) {
      const perm = randperm(crdUniq.length);
      if (crdUniq.length > 1) {
        permMap = new Map();
        for (let k = 0; k < crdUniq.length; k += 1) permMap.set(crdUniq[k], perm[k]);
      }
    }
    const chirV = chirOff >= 0 && noiseAlignChirality ? alignChirality(batch, n, ca, cache) : null;
    for (let i = 0; i < n; i += 1) {
      if (!mask[i]) continue;
      const row = i * stride;
      if (caOff >= 0) {
        const s = i * 3;
        data[row + caOff] = ca[s];
        data[row + caOff + 1] = ca[s + 1];
        data[row + caOff + 2] = ca[s + 2];
      }
      if (latOff >= 0) {
        const s = i * 32;
        const d0 = row + latOff;
        for (let k = 0; k < 32; k += 1) data[d0 + k] = lat[s + k];
      }
      if (permMap) {
        const s = i * 37;
        const d0 = row + grpOff;
        for (let k = 0; k < 37; k += 1) {
          const g = crdGroups[s + k];
          data[d0 + k] = g === CRDGRP_UNKNOWN ? CRDGRP_UNKNOWN : permMap.get(g);
        }
      }
      if (chirV) {
        const s = i * 18;
        const d0 = row + chirOff;
        for (let k = 0; k < 18; k += 1) data[d0 + k] = chirV[s + k];
      }
    }
    return { data, dim };
  }

  const atomMask = need(batch, 'atom37_mask');
  const tokenType = need(batch, 'token_type');
  let curOff = 0;
  let caOff = -1;
  let latOff = -1;
  let grpOff = -1;
  let chirOff = -1;
  let crdUniq = null;
  let crdGroups = null;
  const blocks = feats.map((f, idx) => {
    const startOff = curOff;
    if (f === 'xt_bb_ca') caOff = startOff;
    else if (f === 'xt_local_latents') latOff = startOff;
    else if (f === 'coord_group_seq') {
      grpOff = startOff;
      crdGroups = need(batch, COND.CRDGRP);
      crdUniq = [...new Set(Array.from(crdGroups).filter((g) => g !== CRDGRP_UNKNOWN))].sort((a, b) => a - b);
    } else if (f === 'chirality') chirOff = startOff;
    const blk = seqFeature(f, idx);
    curOff += blk.dim;
    return blk;
  });

  function seqFeature(f, idx) {
    const emb = (suffix = 'embedding') => weight(weights, `${prefix}.feat_creators.${idx}.${suffix}.weight`);
    switch (f) {
      case 'xt_bb_ca':
        return { dim: 3, fill: (i, d, o) => copy(xt.bb_ca, i * 3, 3, d, o) };
      case 'xt_local_latents':
        return { dim: 32, fill: (i, d, o) => copy(xt.local_latents, i * 32, 32, d, o) };
      case 'token_type': {
        const table = emb();
        return { dim: 32, fill: (i, d, o) => embedRow(table, 32, tokenType[i], d, o) };
      }
      case 'sequence_probabilities': {
        const chans = SEQUENCE_PROBABILITY_TOKENS.map((tok) => need(batch, seqProbKey(tok)));
        const C = chans.length;
        return {
          dim: 2 * C,
          fill: (i, d, o) => {
            for (let c = 0; c < C; c += 1) {
              const p = chans[c][i];
              const known = !Number.isNaN(p);
              d[o + c] = known ? p : 0;
              d[o + C + c] = known ? 1 : 0;
            }
          },
        };
      }
      case 'cond_a37coors_nm': {
        const crd = optional(batch, COND.CRD);
        return {
          dim: 148,
          fill: (i, d, o) => {
            if (!crd) return;
            for (let a = 0; a < 37; a += 1) {
              const b = (i * 37 + a) * 3;
              const known = Number.isFinite(crd[b]) || Number.isFinite(crd[b + 1]) || Number.isFinite(crd[b + 2]);
              if (!known) continue;
              for (let k = 0; k < 3; k += 1) d[o + a * 3 + k] = Number.isNaN(crd[b + k]) ? 0 : crd[b + k] / 10;
              d[o + 111 + a] = 1;
            }
          },
        };
      }
      case 'cond_a37coors_internal_nm': {
        const ic = optional(batch, COND.INTCRD);
        const grp = optional(batch, COND.CRDGRP);
        return {
          dim: 148,
          fill: (i, d, o) => {
            if (!ic || !grp) return;
            for (let a = 0; a < 37; a += 1) {
              const g = grp[i * 37 + a];
              if (g === CRDGRP_UNKNOWN || g === CRDGRP_ABSOLUTE) continue;
              const b = (i * 37 + a) * 3;
              for (let k = 0; k < 3; k += 1) d[o + a * 3 + k] = Number.isNaN(ic[b + k]) ? 0 : ic[b + k] / 10;
              d[o + 111 + a] = 1;
            }
          },
        };
      }
      case 'rasa_binned': {
        const ras = optional(batch, COND.RAS);
        return {
          dim: 37,
          fill: (i, d, o) => {
            if (!ras) return;
            for (let a = 0; a < 37; a += 1) d[o + a] = atomMask[i * 37 + a] ? ras[i * 37 + a] : RASA_UNSPECIFIED;
          },
        };
      }
      case 'secondary_structure': {
        const sec = optional(batch, COND.SEC);
        const table = emb('ss_embedding');
        return { dim: 16, fill: (i, d, o) => sec && embedRow(table, 16, sec[i], d, o) };
      }
      case 'hbond_modernized':
        return twoChannel(need(batch, COND.HB), 1, 2, 3);
      case 'pication':
        return twoChannel(need(batch, COND.PICATION), 1, 2, 3);
      case 'pistack':
        return atomFlag(need(batch, COND.PISTACK), (v) => v !== 0);
      case 'hydrophobic':
        return atomFlag(need(batch, COND.HPHOB), (v) => v !== 0);
      case 'saltbridge':
        return atomFlag(need(batch, COND.SALT), (v) => v !== 0);
      case 'metal_coord':
        return atomFlag(need(batch, COND.MCOORD), (v) => v !== 0);
      case 'hotspot': {
        const hot = optional(batch, COND.HOT);
        return hot ? atomFlag(hot, (v) => v !== 0) : { dim: 37, fill: () => {} };
      }
      case 'na_ss_state': {
        const st = optional(batch, 'na_ss_state');
        const sm = optional(batch, 'na_ss_state_mask');
        return {
          dim: 2,
          fill: (i, d, o) => {
            if (!st || !sm) return;
            d[o] = st[i];
            d[o + 1] = sm[i];
          },
        };
      }
      case 'atom_charge': {
        const charge = need(batch, 'charge');
        const table = emb();
        return {
          dim: 16,
          fill: (i, d, o) => {
            if (tokenType[i] !== TOKEN_TYPE_ATOMIZED) return;
            const cls = Math.min(10, Math.max(0, charge[i * 37 + 1] + 4));
            embedRow(table, 16, cls, d, o);
          },
        };
      }
      case 'atomic_element': {
        const element = need(batch, 'element');
        const table = emb();
        return {
          dim: 16,
          fill: (i, d, o) => {
            if (tokenType[i] !== TOKEN_TYPE_ATOMIZED) return;
            embedRow(table, 16, Math.min(127, Math.max(0, element[i * 37 + 1])), d, o);
          },
        };
      }
      case 'coord_group_seq': {
        const groups = permuteCoordGroups(need(batch, COND.CRDGRP), randperm);
        return { dim: 37, fill: (i, d, o) => copy(groups, i * 37, 37, d, o) };
      }
      case 'chirality': {
        const v = noiseAlignChirality ? alignChirality(batch, n, xt.bb_ca, cache) : Float32Array.from(need(batch, 'chirality'));
        return { dim: 18, fill: (i, d, o) => copy(v, i * 18, 18, d, o) };
      }
      case 'chain_centroid_xyz': {
        const cen = optional(batch, COND.CENTROID);
        return {
          dim: 3,
          fill: (i, d, o) => {
            if (!cen) return;
            const ok = [0, 1, 2].every((k) => Number.isFinite(cen[i * 3 + k]));
            if (ok) for (let k = 0; k < 3; k += 1) d[o + k] = cen[i * 3 + k] / 10;
          },
        };
      }
      default:
        throw new Error(`trunk sequence feature '${f}' not implemented`);
    }
  }

  function twoChannel(ann, a, b, both) {
    return {
      dim: 74,
      fill: (i, d, o) => {
        for (let k = 0; k < 37; k += 1) {
          if (!atomMask[i * 37 + k]) continue;
          const v = ann[i * 37 + k];
          d[o + k] = v === a || v === both ? 1 : 0;
          d[o + 37 + k] = v === b || v === both ? 1 : 0;
        }
      },
    };
  }

  function atomFlag(ann, pred) {
    return {
      dim: 37,
      fill: (i, d, o) => {
        for (let k = 0; k < 37; k += 1) d[o + k] = atomMask[i * 37 + k] && pred(ann[i * 37 + k]) ? 1 : 0;
      },
    };
  }

  const res = assembleRows(blocks, n, mask, rowStride);
  if (cache) {
    cache.seqData = res.data;
    cache.seqDim = res.dim;
    cache.seqStride = rowStride || res.dim;
    cache.seqN = n;
    cache.caOff = caOff;
    cache.latOff = latOff;
    cache.grpOff = grpOff;
    cache.chirOff = chirOff;
    cache.crdUniq = crdUniq;
    cache.crdGroups = crdGroups;
  }
  return res;
}

function copy(src, from, len, dst, off) {
  for (let k = 0; k < len; k += 1) dst[off + k] = src[from + k];
}

function assembleRows(blocks, n, mask, rowStride = 0) {
  const dim = blocks.reduce((s, b) => s + b.dim, 0);
  const stride = rowStride || dim;
  const data = new Float32Array(n * stride);
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) continue;
    let off = i * stride;
    for (const b of blocks) {
      b.fill(i, data, off);
      off += b.dim;
    }
  }
  return { data, dim };
}

/**
 * Trunk per-token conditioning features (`nn.cond_factory`, `feats_cond_seq`), broadcast over
 * tokens: time embeddings, CATH embedding, recycle-remaining one-hot.
 * @param {object} batch
 * @param {object} opts
 * @param {number} opts.n
 * @param {{bb_ca: number, local_latents: number}} opts.t
 * @param {number|null} opts.recycleRemaining null on the first pass
 * @param {string[]} opts.feats `config.nn.feats_cond_seq`
 * @param {object} opts.weights
 * @param {number} [opts.tEmbDim]
 * @param {number} [opts.recycleDim]
 */
export function trunkCondFeatures(batch, opts) {
  const { n, t, recycleRemaining, feats, weights, tEmbDim = 256, recycleDim = 1, cache = null } = opts;
  const mask = need(batch, 'padding_mask');
  const key = recycleRemaining === null ? 'null' : String(recycleRemaining);
  if (cache && cache.condBuffers && cache.condBuffers[key] && cache.condN === n) {
    const { data, dim, tCaOff, tLatOff } = cache.condBuffers[key];
    const eCa = tCaOff >= 0 ? timeEmbedding(t.bb_ca, tEmbDim) : null;
    const eLat = tLatOff >= 0 ? timeEmbedding(t.local_latents, tEmbDim) : null;
    const rows = opts.singleRow && data.uniform ? 1 : n;
    for (let i = 0; i < rows; i += 1) {
      if (!mask[i]) continue;
      const row = i * dim;
      if (eCa) data.set(eCa, row + tCaOff);
      if (eLat) data.set(eLat, row + tLatOff);
    }
    return { data, dim };
  }
  const prefix = 'nn.cond_factory';
  const cath = optional(batch, COND.CATH);
  let condUniform = true;
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) {
      condUniform = false;
      break;
    }
    if (cath && i > 0) {
      if (cath[i * 3] !== cath[0] || cath[i * 3 + 1] !== cath[1] || cath[i * 3 + 2] !== cath[2]) {
        condUniform = false;
        break;
      }
    }
  }
  let curOff = 0;
  let tCaOff = -1;
  let tLatOff = -1;
  const blocks = feats.map((f, idx) => {
    const startOff = curOff;
    let blk;
    switch (f) {
      case 'time_emb_bb_ca': {
        tCaOff = startOff;
        const e = timeEmbedding(t.bb_ca, tEmbDim);
        blk = { dim: tEmbDim, fill: (i, d, o) => d.set(e, o) };
        break;
      }
      case 'time_emb_local_latents': {
        tLatOff = startOff;
        const e = timeEmbedding(t.local_latents, tEmbDim);
        blk = { dim: tEmbDim, fill: (i, d, o) => d.set(e, o) };
        break;
      }
      case 'cath_code': {
        const p = `${prefix}.feat_creators.${idx}`;
        const tabs = [weight(weights, `${p}.embedding_C.weight`), weight(weights, `${p}.embedding_A.weight`),
          weight(weights, `${p}.embedding_T.weight`)];
        blk = {
          dim: 96,
          fill: (i, d, o) => {
            for (let k = 0; k < 3; k += 1) embedRow(tabs[k], 32, (cath ? cath[i * 3 + k] : -1) + 1, d, o + 32 * k);
          },
        };
        break;
      }
      case 'recycle_remaining':
        blk = {
          dim: recycleDim,
          fill: (i, d, o) => {
            if (recycleRemaining == null) return;
            d[o + Math.min(recycleRemaining, recycleDim - 1)] = 1;
          },
        };
        break;
      default:
        throw new Error(`trunk conditioning feature '${f}' not implemented`);
    }
    curOff += blk.dim;
    return blk;
  });
  const res = assembleRows(blocks, n, mask);
  res.data.uniform = condUniform;
  if (cache) {
    if (!cache.condBuffers || cache.condN !== n) {
      cache.condBuffers = {};
      cache.condN = n;
    }
    cache.condBuffers[key] = { data: res.data, dim: res.dim, tCaOff, tLatOff };
  }
  return res;
}

/**
 * Trunk pair features (`nn.pair_repr_builder.init_repr_factory`, `feats_pair_repr`).
 * @param {object} batch
 * @param {object} opts
 * @param {number} opts.n
 * @param {Float32Array} opts.caNm `[n, 3]` CA coordinates (x_t for the trunk, nm)
 * @param {string[]} opts.feats
 * @param {{dim: number, min: number, max: number}} [opts.dist]
 * @param {number} [opts.seqSepDim]
 * @returns {{ data: Float32Array, dim: number }} `[n*n, dim]`
 */
export function pairFeatures(batch, opts) {
  const { n, caNm, feats, dist = { dim: 30, min: 0.1, max: 3 }, seqSepDim = 64, cache = null, rowStride = 0, skipPairDists = false } = opts;
  const mask = need(batch, 'padding_mask');
  const limits = new Float32Array(dist.dim - 1);
  for (let k = 0; k < dist.dim - 1; k += 1) {
    limits[k] = dist.min + ((dist.max - dist.min) * k) / (dist.dim - 2);
  }
  if (cache && cache.pairData && cache.pairN === n && cache.distOffset !== undefined && cache.pairStride === (rowStride || cache.pairDim)) {
    const { pairData: data, pairDim: dim, pairStride: stride, distOffset, prevBins } = cache;
    if (!skipPairDists) {
      for (let i = 0; i < n; i += 1) {
        if (!mask[i]) continue;
        const ix = caNm[i * 3];
        const iy = caNm[i * 3 + 1];
        const iz = caNm[i * 3 + 2];
        for (let j = 0; j < n; j += 1) {
          if (!mask[j]) continue;
          const ij = i * n + j;
          const o = ij * stride + distOffset;
          const prev = prevBins[ij];
          if (prev !== 255) data[o + prev] = 0;
          const dx = ix - caNm[j * 3];
          const dy = iy - caNm[j * 3 + 1];
          const dz = iz - caNm[j * 3 + 2];
          const r = Math.fround(Math.sqrt(Math.fround(dx * dx + dy * dy + dz * dz)));
          let bin = 0;
          while (bin < limits.length && limits[bin] < r) bin += 1;
          data[o + bin] = 1;
          prevBins[ij] = bin;
        }
      }
    }
    return { data, dim };
  }
  let distOffset = -1;
  let curOffset = 0;
  const prevBins = cache && !skipPairDists ? new Uint8Array(n * n).fill(255) : null;
  const blocks = feats.map((f) => {
    const startOff = curOffset;
    let blk;
    switch (f) {
      case 'xt_bb_ca_pair_dists':
      case 'ca_coors_nm_pair_dists':
        distOffset = startOff;
        blk = {
          dim: dist.dim,
          fill: skipPairDists
            ? () => {}
            : (i, j, d, o) => {
                const dx = caNm[i * 3] - caNm[j * 3];
                const dy = caNm[i * 3 + 1] - caNm[j * 3 + 1];
                const dz = caNm[i * 3 + 2] - caNm[j * 3 + 2];
                const r = Math.fround(Math.sqrt(Math.fround(dx * dx + dy * dy + dz * dz)));
                let bin = 0;
                while (bin < limits.length && limits[bin] < r) bin += 1; // torch.bucketize(right=False)
                d[o + bin] = 1;
                if (prevBins) prevBins[i * n + j] = bin;
              },
        };
        break;
      case 'bond_mask':
        blk = pairScalar(optional(batch, 'bond_mask'));
        break;
      case 'is_aromatic':
        blk = pairScalar(optional(batch, 'is_aromatic'));
        break;
      case 'canonical_polymer_ordering':
        blk = pairScalar(optional(batch, 'canonical_polymer_ordering'));
        break;
      case 'bond_order': {
        const bo = optional(batch, 'bond_order');
        blk = { dim: 5, fill: (i, j, d, o) => bo && (d[o + bo[i * n + j]] = 1) };
        break;
      }
      case 'index_group_pair': {
        const g = need(batch, COND.IDXGRP);
        blk = { dim: 1, fill: (i, j, d, o) => (d[o] = g[i] === g[j] ? 1 : 0) };
        break;
      }
      case 'molecule_idx_pair': {
        const g = need(batch, 'molecule_iid');
        blk = { dim: 1, fill: (i, j, d, o) => (d[o] = g[i] === g[j] ? 1 : 0) };
        break;
      }
      case 'molecule_entity_pair': {
        const g = need(batch, 'molecule_entity_id');
        blk = { dim: 1, fill: (i, j, d, o) => (d[o] = g[i] === g[j] ? 1 : 0) };
        break;
      }
      case 'token_bond_path_length': {
        const pl = need(batch, 'token_bond_path_length');
        blk = { dim: seqSepDim, fill: (i, j, d, o) => (d[o + Math.min(pl[i * n + j], seqSepDim - 1)] = 1) };
        break;
      }
      case 'na_ss_pair': {
        const p = optional(batch, 'na_ss_pair');
        const m = optional(batch, 'na_ss_pair_mask');
        blk = {
          dim: 2,
          fill: (i, j, d, o) => {
            if (!p || !m) return;
            d[o] = p[i * n + j];
            d[o + 1] = m[i * n + j];
          },
        };
        break;
      }
      default:
        throw new Error(`pair feature '${f}' not implemented`);
    }
    curOffset += blk.dim;
    return blk;
  });

  function pairScalar(src) {
    return { dim: 1, fill: (i, j, d, o) => src && (d[o] = src[i * n + j]) };
  }

  const dim = blocks.reduce((s, b) => s + b.dim, 0);
  const stride = rowStride || dim;
  const data = new Float32Array(n * n * stride);
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) continue;
    for (let j = 0; j < n; j += 1) {
      if (!mask[j]) continue;
      let off = (i * n + j) * stride;
      for (const b of blocks) {
        b.fill(i, j, data, off);
        off += b.dim;
      }
    }
  }
  if (cache && distOffset >= 0 && !skipPairDists) {
    cache.pairData = data;
    cache.pairDim = dim;
    cache.pairStride = stride;
    cache.pairN = n;
    cache.distOffset = distOffset;
    cache.prevBins = prevBins;
  }
  return { data, dim };
}

/**
 * Compact per-example pair conditioning (`nn.pair_repr_builder.cond_factory`): `[512]`.
 * @param {{bb_ca: number, local_latents: number}} t
 * @param {string[]} feats `config.nn.feats_pair_cond`
 */
export function pairCondFeatures(t, feats, tEmbDim = 256) {
  const out = new Float32Array(feats.length * tEmbDim);
  feats.forEach((f, k) => {
    if (f === 'time_emb_bb_ca') out.set(timeEmbedding(t.bb_ca, tEmbDim), k * tEmbDim);
    else if (f === 'time_emb_local_latents') out.set(timeEmbedding(t.local_latents, tEmbDim), k * tEmbDim);
    else throw new Error(`pair conditioning feature '${f}' not implemented`);
  });
  return out;
}
