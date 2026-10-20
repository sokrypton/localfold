/**
 * BioEmu2's small once-per-fold arithmetic, on the host.
 *
 * What runs here is per ATOM or per atom-WINDOW PAIR and touches tens of
 * thousands of numbers, once a fold: the atom features, the windowed pair
 * geometry and its MLP, the per-layer window biases, the token means of the
 * input embedder and the trunk's single initialisation. Each is a line-for-line
 * transcription of af3-any-model's `network/bioemu2.py`, in float64 with a
 * float32 store, as numpy rounds a float32 tensor.
 */
import { BIOEMU2 } from "../../shared/bioemu2/weights.js";

const W_Q = BIOEMU2.windowQueries;
const W_K = BIOEMU2.windowKeys;
const CTX = (W_K - W_Q) / 2;

/** y[r] = LN(x[r]) * w + b over `width`, biased variance, eps 1e-5. */
export function lnRows(x, rows, width, w, b, eps = 1e-5) {
  const y = new Float32Array(rows * width);
  for (let r = 0; r < rows; r += 1) {
    let mean = 0;
    for (let c = 0; c < width; c += 1) mean += x[r * width + c];
    mean /= width;
    let v = 0;
    for (let c = 0; c < width; c += 1) { const d = x[r * width + c] - mean; v += d * d; }
    const inv = 1 / Math.sqrt(v / width + eps);
    for (let c = 0; c < width; c += 1) {
      let value = (x[r * width + c] - mean) * inv;
      if (w) value *= w[c];
      if (b) value += b[c];
      y[r * width + c] = value;
    }
  }
  return y;
}

/** y = x . W^T (+ b): W torch (out, in). */
export function linearRows(x, rows, inC, w, outC, b) {
  const y = new Float32Array(rows * outC);
  for (let r = 0; r < rows; r += 1) {
    for (let o = 0; o < outC; o += 1) {
      let s = b ? b[o] : 0;
      const wo = o * inC;
      const xr = r * inC;
      for (let i = 0; i < inC; i += 1) s += x[xr + i] * w[wo + i];
      y[r * outC + o] = s;
    }
  }
  return y;
}

/** The front of compute_static_context: c (N, 128) and the reference-geometry pair (NW, 32, 128, 16). */
export function atomFeatures(p, f) {
  const N = f.atom_pad_mask.length;
  const NW = N / W_Q;
  const C = BIOEMU2.atom;
  const Z = BIOEMU2.atomPair;
  const c = new Float32Array(N * C);
  const cont = p["proj_atom_continuous/weight"];       // (128, 4)
  const element = p["embed_element/weight"];          // (128, 128)
  const names = [0, 1, 2, 3].map((i) => p[`embed_atom_name_chars_${i}/weight`]);   // (64, 128)
  for (let a = 0; a < N; a += 1) {
    if (!f.atom_pad_mask[a]) continue;
    const ref = [f.template_coords[a * 3], f.template_coords[a * 3 + 1], f.template_coords[a * 3 + 2], f.charge[a]];
    for (let o = 0; o < C; o += 1) {
      let s = 0;
      for (let i = 0; i < 4; i += 1) s += ref[i] * cont[o * 4 + i];
      s += element[f.element[a] * C + o];
      for (let i = 0; i < 4; i += 1) s += names[i][f.name_encoding[a * 4 + i] * C + o];
      c[a * C + o] = s;
    }
  }
  // windowed_atom_pair_geometry
  const pos = p["embed_atompair_ref_pos/weight"];      // (16, 3)
  const dist = p["embed_atompair_ref_dist/weight"];    // (16, 1)
  const maskW = p["embed_atompair_mask/weight"];       // (16, 1)
  const pr = new Float32Array(NW * W_Q * W_K * Z);
  for (let w = 0; w < NW; w += 1) {
    for (let i = 0; i < W_Q; i += 1) {
      const qa = w * W_Q + i;
      if (!f.atom_pad_mask[qa]) continue;
      for (let j = 0; j < W_K; j += 1) {
        const ka = w * W_Q - CTX + j;
        const inside = ka >= 0 && ka < N;
        const kuid = inside && f.atom_pad_mask[ka] ? f.ref_space_uid[ka] : -1;
        if (f.ref_space_uid[qa] !== kuid) continue;
        const d = [0, 1, 2].map((x) => (inside ? f.template_coords[ka * 3 + x] : 0) - f.template_coords[qa * 3 + x]);
        const inv = 1 / (1 + d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const at = ((w * W_Q + i) * W_K + j) * Z;
        for (let o = 0; o < Z; o += 1) {
          pr[at + o] = pos[o * 3] * d[0] + pos[o * 3 + 1] * d[1] + pos[o * 3 + 2] * d[2] + dist[o] * inv + maskW[o];
        }
      }
    }
  }
  return { c, pr, N, NW };
}

/** c -> p (queries and keys), then the three-layer p MLP. In place on `pr`. */
export function atomPairTail(p, c, pr, N) {
  const NW = N / W_Q;
  const C = BIOEMU2.atom;
  const Z = BIOEMU2.atomPair;
  const relu = (v) => (v > 0 ? v : 0);
  const reluC = Float32Array.from(c, relu);
  const cq = linearRows(reluC, N, C, p["c_to_p_trans_q_1/weight"], Z);
  const ck = linearRows(reluC, N, C, p["c_to_p_trans_k_1/weight"], Z);
  const m1 = p["p_mlp_1/weight"];
  const m3 = p["p_mlp_3/weight"];
  const m5 = p["p_mlp_5/weight"];
  const h = new Float64Array(Z);
  const g = new Float64Array(Z);
  for (let w = 0; w < NW; w += 1) {
    for (let i = 0; i < W_Q; i += 1) {
      const qa = w * W_Q + i;
      for (let j = 0; j < W_K; j += 1) {
        const ka = w * W_Q - CTX + j;
        const at = ((w * W_Q + i) * W_K + j) * Z;
        for (let o = 0; o < Z; o += 1) {
          pr[at + o] += cq[qa * Z + o] + (ka >= 0 && ka < N ? ck[ka * Z + o] : 0);
        }
        for (let o = 0; o < Z; o += 1) { let s = 0; for (let k = 0; k < Z; k += 1) s += relu(pr[at + k]) * m1[o * Z + k]; h[o] = s; }
        for (let o = 0; o < Z; o += 1) { let s = 0; for (let k = 0; k < Z; k += 1) s += relu(h[k]) * m3[o * Z + k]; g[o] = s; }
        for (let o = 0; o < Z; o += 1) { let s = 0; for (let k = 0; k < Z; k += 1) s += relu(g[k]) * m5[o * Z + k]; pr[at + o] += s; }
      }
    }
  }
  return pr;
}

/** LN(pair) * norm_z -> proj_z, heads first: (NW, H, 32, 128). */
export function windowBias(pr, N, normZ, projZ, heads) {
  const NW = N / W_Q;
  const Z = BIOEMU2.atomPair;
  const out = new Float32Array(NW * heads * W_Q * W_K);
  const x = new Float64Array(Z);
  for (let w = 0; w < NW; w += 1) {
    for (let i = 0; i < W_Q; i += 1) {
      for (let j = 0; j < W_K; j += 1) {
        const at = ((w * W_Q + i) * W_K + j) * Z;
        let mean = 0;
        for (let k = 0; k < Z; k += 1) mean += pr[at + k];
        mean /= Z;
        let v = 0;
        for (let k = 0; k < Z; k += 1) { const d = pr[at + k] - mean; v += d * d; }
        const inv = 1 / Math.sqrt(v / Z + 1e-5);
        for (let k = 0; k < Z; k += 1) x[k] = (pr[at + k] - mean) * inv * normZ[k];
        for (let h = 0; h < heads; h += 1) {
          let s = 0;
          for (let k = 0; k < Z; k += 1) s += x[k] * projZ[h * Z + k];
          out[((w * heads + h) * W_Q + i) * W_K + j] = s;
        }
      }
    }
  }
  return out;
}

/** broadcast_token_pair_to_atom_window, added into `pr`: z[tok_q, tok_k] where both atoms are real. */
export function addTokenPairToWindows(pr, zp, T, f) {
  const N = f.atom_pad_mask.length;
  const NW = N / W_Q;
  const Z = BIOEMU2.atomPair;
  for (let w = 0; w < NW; w += 1) {
    for (let i = 0; i < W_Q; i += 1) {
      const qa = w * W_Q + i;
      if (!f.atom_pad_mask[qa]) continue;
      const tq = f.token_idx_at_atom[qa];
      for (let j = 0; j < W_K; j += 1) {
        const ka = w * W_Q - CTX + j;
        if (ka < 0 || ka >= N || !f.atom_pad_mask[ka]) continue;
        const tk = f.token_idx_at_atom[ka];
        const at = ((w * W_Q + i) * W_K + j) * Z;
        const src = (tq * T + tk) * Z;
        for (let o = 0; o < Z; o += 1) pr[at + o] += zp[src + o];
      }
    }
  }
}

/** s_inputs = [token mean of relu(q W) | one-hot res_type | profile | deletion_mean | one-hot pocket]. */
export function sInputs(q, f, w, T) {
  const N = f.atom_pad_mask.length;
  const proj = linearRows(q, N, BIOEMU2.atom, w, BIOEMU2.single);
  const sums = new Float64Array(T * BIOEMU2.single);
  const counts = new Float64Array(T);
  for (let a = 0; a < N; a += 1) {
    if (!f.atom_pad_mask[a]) continue;
    const t = f.token_idx_at_atom[a];
    counts[t] += 1;
    for (let o = 0; o < BIOEMU2.single; o += 1) sums[t * BIOEMU2.single + o] += Math.max(proj[a * BIOEMU2.single + o], 0);
  }
  const width = BIOEMU2.sInputs;
  const out = new Float32Array(T * width);
  for (let t = 0; t < T; t += 1) {
    const row = t * width;
    for (let o = 0; o < BIOEMU2.single; o += 1) out[row + o] = sums[t * BIOEMU2.single + o] / (counts[t] + 1e-6);
    out[row + 384 + f.res_type[t]] = 1;
    for (let k = 0; k < 33; k += 1) out[row + 417 + k] = f.profile[t * 33 + k];
    out[row + 450] = f.deletion_mean[t];
    out[row + 451 + f.pocket_feature[t]] = 1;
  }
  return out;
}

/** DampedHarmonicTimeEmbedding for a scalar lag: [envelope, sin, cos], timescale-major. */
export function dampedHarmonic(dt, taus, quality) {
  const n = taus.length;
  const Q = quality.length;
  const env = Array.from(taus, (tau) => Math.exp(-dt / tau));
  const out = new Float32Array(n * (1 + 2 * Q));
  for (let i = 0; i < n; i += 1) out[i] = env[i];
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < Q; k += 1) {
      const phase = 2 * Math.PI * quality[k] * (dt / taus[i]);
      out[n + i * Q + k] = env[i] * Math.sin(phase);
      out[n + n * Q + i * Q + k] = env[i] * Math.cos(phase);
    }
  }
  return out;
}
