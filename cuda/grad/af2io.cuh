// AlphaFold 2's input embedder and output heads for training, around the Evoformer (af2.cuh) and the structure module
// (af2sm.cuh). cuda/af2/src/evoformer.cuh's embed() and af2.cu's heads are what the forward is held to - and, through
// cuda/af2/oracle.py's dumps, af3-any-model's JAX itself.
//
//   embedder  msa   = msa_feat W + b + target_feat W1d + b1d   (row 0: + LN(prev msa first row))
//             pair  = left[aa_i] + right[aa_j] + prev_pos_linear(dgram of prev pseudo-beta) + LN(prev pair) + relpos
//             extra = extra_msa_activations([one-hot(23) | has_deletion | deletion_value])
//   heads     single = single_activations(msa row 0); pLDDT (LN structure act, two ReLU layers, 50 bins); PAE (pair, 64);
//             distogram (pair half_logits, symmetrised); masked MSA (msa, 22); experimentally resolved (single, 37)
//
// The recycled inputs (prev msa row, pair, positions) are stop_gradient in AlphaFold 2: their LayerNorms' parameters
// train, the inputs take no gradient. The residue type clipped to 0..19 and the relative encoding's cross-chain rows as
// the port has them (monomer: the 65 offset rows alone; multimer: + same-entity and symmetry rows).
#pragma once
#include "ops.cuh"

namespace grad {
namespace af2 {

// ---------------------------------------------------------------- the embedder
struct Embedder {
  int C = 256, Cz = 128, Ce = 64, F = 49, rel = 73;
  bool multimer = false;
  Linear msa, oneD, extra;                // msa_feat [F] -> C; target one-hot [21] -> C (as a gather); extra [25] -> Ce
  Param left, leftB, right, rightB, dg, dgB, relW, relB;   // [21, Cz], [15, Cz], [rel, Cz]
  LayerNorm prevRow, prevPair;
};
inline void forEachParam(Embedder& e, const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string E = "evoformer/";
  auto lin = [&](const std::string& n, Linear& l, int K, int N) {
    l.K = K; l.N = N; l.hasBias = true; f(E + n + "/weights", l.w, (size_t)K * N); f(E + n + "/bias", l.b, N);
  };
  lin("preprocess_msa", e.msa, e.F, e.C);
  lin("preprocess_1d", e.oneD, 21, e.C);
  lin("extra_msa_activations", e.extra, 25, e.Ce);
  f(E + "left_single/weights", e.left, (size_t)21 * e.Cz); f(E + "left_single/bias", e.leftB, e.Cz);
  f(E + "right_single/weights", e.right, (size_t)21 * e.Cz); f(E + "right_single/bias", e.rightB, e.Cz);
  f(E + "prev_pos_linear/weights", e.dg, (size_t)15 * e.Cz); f(E + "prev_pos_linear/bias", e.dgB, e.Cz);
  f(E + "~_relative_encoding/position_activations/weights", e.relW, (size_t)e.rel * e.Cz);
  f(E + "~_relative_encoding/position_activations/bias", e.relB, e.Cz);
  e.prevRow.C = e.C; f(E + "prev_msa_first_row_norm/scale", e.prevRow.g, e.C); f(E + "prev_msa_first_row_norm/offset", e.prevRow.b, e.C);
  e.prevPair.C = e.Cz; f(E + "prev_pair_norm/scale", e.prevPair.g, e.Cz); f(E + "prev_pair_norm/offset", e.prevPair.b, e.Cz);
}
struct EmbedInputs {
  const real* msaFeat;  const int* aatype;     // [N, L, F], [L]
  const int *ri, *asym, *entity, *sym;         // [L]
  const int* extraMsa; const real *hasDel, *delVal;   // [E, L]
  const real *prevRow, *prevPair, *prevPos;    // [L, C], [L, L, Cz], [L, 37, 3]
  int L, N, E;
};
__device__ inline int clipAa(int a) { return a < 0 ? 0 : a > 19 ? 19 : a; }
// the pair's per-(i, j) row indices: the dgram bin (or -1) and the relative encoding's rows
__device__ inline void pairRows(const EmbedInputs& in, int i, int j, bool multimer, int& bin, int& r0, int& r1, int& r2) {
  auto pb = [&](int r, int k) { int a = in.aatype[r] == 7 ? 1 : 3; return in.prevPos[((size_t)r * 37 + a) * 3 + k]; };
  real d2 = 0;
  for (int k = 0; k < 3; ++k) { real d = pb(i, k) - pb(j, k); d2 += d * d; }
  bin = -1;
  for (int b = 0; b < 15; ++b) {
    real lo = (real)(3.25f + (20.75f - 3.25f) * b / 14.f), hi = b + 1 < 15 ? (real)(3.25f + (20.75f - 3.25f) * (b + 1) / 14.f) : 0;
    real lower = lo * lo, upper = b + 1 < 15 ? hi * hi : (real)1e8;
    if (d2 > lower && d2 < upper) bin = b;
  }
  int off = in.ri[i] - in.ri[j], clipped = min(max(off + 32, 0), 64);
  r0 = in.asym[i] == in.asym[j] ? clipped : multimer ? 65 : in.asym[i] > in.asym[j] ? 64 : 0;
  r1 = r2 = -1;
  if (multimer) {
    bool same = in.entity[i] == in.entity[j];
    if (same) r1 = 66;
    r2 = 67 + (same ? min(max(in.sym[i] - in.sym[j] + 2, 0), 4) : 5);
  }
}
// pair = left + right + dgram + LN(prev pair) (pre-normalised by the caller) + relpos
__global__ void embedPairK(EmbedInputs in, const real* left, const real* leftB, const real* right, const real* rightB,
                           const real* dg, const real* dgB, const real* relW, const real* relB, const real* prevLn,
                           real* pair, int Cz, bool multimer) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int L = in.L;
  if (t >= (size_t)L * L * Cz) return;
  int c = t % Cz; size_t ij = t / Cz; int i = ij / L, j = ij % L;
  int bin, r0, r1, r2; pairRows(in, i, j, multimer, bin, r0, r1, r2);
  int ai = clipAa(in.aatype[i]), aj = clipAa(in.aatype[j]);
  real v = left[ai * Cz + c] + leftB[c] + right[aj * Cz + c] + rightB[c];
  v += (bin >= 0 ? dg[bin * Cz + c] : 0) + dgB[c];
  v += prevLn[t];
  v += relW[r0 * Cz + c] + (r1 >= 0 ? relW[r1 * Cz + c] : 0) + (r2 >= 0 ? relW[r2 * Cz + c] : 0) + relB[c];
  pair[t] = v;
}
// d pair -> the table gradients, by channel: each thread a channel sums every pair it reads (no atomics on a table row
// shared by thousands of pairs)
__global__ void embedPairBwdK(EmbedInputs in, const real* d, real* gLeft, real* gLeftB, real* gRight, real* gRightB, real* gDg,
                              real* gDgB, real* gRel, real* gRelB, int Cz, bool multimer, int rel) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= Cz) return;
  const int L = in.L;
  real sAll = 0;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) {
      real g = d[((size_t)i * L + j) * Cz + c];
      if (g == 0) continue;
      int bin, r0, r1, r2; pairRows(in, i, j, multimer, bin, r0, r1, r2);
      gLeft[clipAa(in.aatype[i]) * Cz + c] += g;
      gRight[clipAa(in.aatype[j]) * Cz + c] += g;
      if (bin >= 0) gDg[bin * Cz + c] += g;
      gRel[r0 * Cz + c] += g;
      if (r1 >= 0) gRel[r1 * Cz + c] += g;
      if (r2 >= 0) gRel[r2 * Cz + c] += g;
      sAll += g;
    }
  gLeftB[c] += sAll; gRightB[c] += sAll; gDgB[c] += sAll; gRelB[c] += sAll;
}
// msa += oneD[aa_i] + b1d (+ LN(prev row) on row 0)
__global__ void msaAddK(real* msa, const int* aatype, const real* w1d, const real* b1d, const real* prevLn, size_t rows, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  int c = t % C, i = t / C % L;
  real v = w1d[clipAa(aatype[i]) * C + c] + b1d[c];
  if (t < (size_t)L * C) v += prevLn[t];
  msa[t] += v;
}
// by channel: gW1d[aa_i] += sum_rows d[row, i], gb1d += sum; the prev row's d is d's row 0 (returned in d itself)
__global__ void msaAddBwdK(const int* aatype, const real* d, real* gW, real* gB, size_t rows, int L, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  real s = 0;
  for (int i = 0; i < L; ++i) {
    real si = 0;
    for (size_t r = 0; r < rows / L; ++r) si += d[(r * L + i) * C + c];
    gW[clipAa(aatype[i]) * C + c] += si; s += si;
  }
  gB[c] += s;
}
// [one-hot(23) | has_deletion | deletion_value] per extra row
__global__ void extraFeatK(const int* codes, const real* hasDel, const real* delVal, real* f, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * 25) return;
  size_t r = t / 25; int k = t % 25;
  f[t] = k < 23 ? (codes[r] == k ? (real)1 : 0) : k == 23 ? hasDel[r] : delVal[r];
}
struct EmbedSave { real *prevRowLn, *prevPairLn, *extraFeat; LnSave lr, lp; };
inline EmbedSave embedForward(Embedder& e, const EmbedInputs& in, real* msa, real* pair, real* extra) {
  EmbedSave s{};
  const int L = in.L; const size_t rows = (size_t)in.N * L, pairs = (size_t)L * L;
  e.msa.forward(in.msaFeat, msa, rows);
  s.prevRowLn = ARENA.get((size_t)L * e.C); s.lr = lnForward(e.prevRow, in.prevRow, s.prevRowLn, L);
  msaAddK<<<blocks(rows * e.C), 256, 0, STREAM>>>(msa, in.aatype, e.oneD.w.w, e.oneD.b.w, s.prevRowLn, rows, L, e.C);
  s.prevPairLn = ARENA.get(pairs * e.Cz); s.lp = lnForward(e.prevPair, in.prevPair, s.prevPairLn, pairs);
  embedPairK<<<blocks(pairs * e.Cz), 256, 0, STREAM>>>(in, e.left.w, e.leftB.w, e.right.w, e.rightB.w, e.dg.w, e.dgB.w, e.relW.w,
                                                       e.relB.w, s.prevPairLn, pair, e.Cz, e.multimer);
  if (extra && in.E > 0) {
    const size_t er = (size_t)in.E * L;
    s.extraFeat = ARENA.get(er * 25);
    extraFeatK<<<blocks(er * 25), 256, 0, STREAM>>>(in.extraMsa, in.hasDel, in.delVal, s.extraFeat, er);
    e.extra.forward(s.extraFeat, extra, er);
  }
  return s;
}
// dmsa, dpair, dextra (any null) -> every parameter (the inputs are data or stop_gradient)
inline void embedBackward(Embedder& e, const EmbedInputs& in, const EmbedSave& s, const real* dmsa, const real* dpair, const real* dextra) {
  Scratch scratch;
  const int L = in.L; const size_t rows = (size_t)in.N * L, pairs = (size_t)L * L;
  if (dmsa) {
    e.msa.backward(in.msaFeat, dmsa, nullptr, rows);
    msaAddBwdK<<<blocks(e.C, 64), 64, 0, STREAM>>>(in.aatype, dmsa, e.oneD.w.g, e.oneD.b.g, rows, L, e.C);
    real* t = ARENA.get((size_t)L * e.C); lnBackward(e.prevRow, s.lr, dmsa, t, L, false);   // row 0's
  }
  if (dpair) {
    embedPairBwdK<<<blocks(e.Cz, 32), 32, 0, STREAM>>>(in, dpair, e.left.g, e.leftB.g, e.right.g, e.rightB.g, e.dg.g, e.dgB.g,
                                                        e.relW.g, e.relB.g, e.Cz, e.multimer, e.rel);
    real* t = ARENA.get(pairs * e.Cz); lnBackward(e.prevPair, s.lp, dpair, t, pairs, false);
  }
  if (dextra && in.E > 0) e.extra.backward(s.extraFeat, dextra, nullptr, (size_t)in.E * L);
}

// ---------------------------------------------------------------- the heads
struct Heads {
  int Cm = 256, Cz = 128, Cs = 384, msaClasses = 22, dgBins = 64, paeBins = 64, plddtBins = 50, plddtHidden = 128;
  Linear single, plddt0, plddt1, plddtOut, pae, dgHalf, maskedMsa, resolved;
  LayerNorm plddtLn;
  bool hasPae = true;
};
inline void forEachParam(Heads& h, const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto lin = [&](const std::string& n, Linear& l, int K, int N) {
    l.K = K; l.N = N; l.hasBias = true; f(n + "/weights", l.w, (size_t)K * N); f(n + "/bias", l.b, N);
  };
  lin("evoformer/single_activations", h.single, h.Cm, h.Cs);
  h.plddtLn.C = h.Cs; f("predicted_lddt_head/input_layer_norm/scale", h.plddtLn.g, h.Cs);
  f("predicted_lddt_head/input_layer_norm/offset", h.plddtLn.b, h.Cs);
  lin("predicted_lddt_head/act_0", h.plddt0, h.Cs, h.plddtHidden);
  lin("predicted_lddt_head/act_1", h.plddt1, h.plddtHidden, h.plddtHidden);
  lin("predicted_lddt_head/logits", h.plddtOut, h.plddtHidden, h.plddtBins);
  if (h.hasPae) lin("predicted_aligned_error_head/logits", h.pae, h.Cz, h.paeBins);
  lin("distogram_head/half_logits", h.dgHalf, h.Cz, h.dgBins);
  lin("masked_msa_head/logits", h.maskedMsa, h.Cm, h.msaClasses);
  lin("experimentally_resolved_head/logits", h.resolved, h.Cs, 37);
}
// out[i, j] = h[i, j] + h[j, i] (and its own transpose in the backward)
__global__ void symK(const real* h, real* out, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / L, j = ij % L;
  out[t] = h[t] + h[((size_t)j * L + i) * C + c];
}
struct HeadSave { real *singleOut, *lnA, *h0, *r0, *h1, *r1, *dgh; LnSave ln; };
struct HeadOut { real *single, *plddt, *pae, *distogram, *maskedMsa, *resolved; };
// msa [N, L, Cm] (the evoformer's), pair [L, L, Cz], act [L, Cs] (the structure module's last) -> every head; the
// single is the structure module's input (and the resolved head's)
inline HeadSave headsForward(Heads& h, const real* msa, const real* pair, const real* act, int L, int N, HeadOut& o) {
  HeadSave s{};
  const size_t pairs = (size_t)L * L;
  o.single = ARENA.get((size_t)L * h.Cs); h.single.forward(msa, o.single, L);
  s.lnA = ARENA.get((size_t)L * h.Cs); s.ln = lnForward(h.plddtLn, act, s.lnA, L);
  s.h0 = ARENA.get((size_t)L * h.plddtHidden); h.plddt0.forward(s.lnA, s.h0, L);
  s.r0 = ARENA.get((size_t)L * h.plddtHidden); reluK<<<blocks((size_t)L * h.plddtHidden), 256, 0, STREAM>>>(s.r0, s.h0, (size_t)L * h.plddtHidden);
  s.h1 = ARENA.get((size_t)L * h.plddtHidden); h.plddt1.forward(s.r0, s.h1, L);
  s.r1 = ARENA.get((size_t)L * h.plddtHidden); reluK<<<blocks((size_t)L * h.plddtHidden), 256, 0, STREAM>>>(s.r1, s.h1, (size_t)L * h.plddtHidden);
  o.plddt = ARENA.get((size_t)L * h.plddtBins); h.plddtOut.forward(s.r1, o.plddt, L);
  o.pae = nullptr;
  if (h.hasPae) { o.pae = ARENA.get(pairs * h.paeBins); h.pae.forward(pair, o.pae, pairs); }
  s.dgh = ARENA.get(pairs * h.dgBins); h.dgHalf.forward(pair, s.dgh, pairs);
  o.distogram = ARENA.get(pairs * h.dgBins); symK<<<blocks(pairs * h.dgBins), 256, 0, STREAM>>>(s.dgh, o.distogram, L, h.dgBins);
  o.maskedMsa = ARENA.get((size_t)N * L * h.msaClasses); h.maskedMsa.forward(msa, o.maskedMsa, (size_t)N * L);
  o.resolved = ARENA.get((size_t)L * 37); h.resolved.forward(o.single, o.resolved, L);
  s.singleOut = o.single;
  return s;
}
// the heads' cotangents (any null; dsingle the structure module's d single, added to the resolved head's) ->
// every parameter; dmsa, dpair, dact (added)
inline void headsBackward(Heads& h, const HeadSave& s, const real* msa, const real* pair, const real* act, int L, int N,
                          const HeadOut& d, const real* dsingleIn, real* dmsa, real* dpair, real* dact) {
  Scratch scratch;
  const size_t pairs = (size_t)L * L;
  if (d.plddt) {
    real* dr1 = ARENA.get((size_t)L * h.plddtHidden); h.plddtOut.backward(s.r1, d.plddt, dr1, L);
    reluBwdK<<<blocks((size_t)L * h.plddtHidden), 256, 0, STREAM>>>(dr1, s.h1, (size_t)L * h.plddtHidden);
    real* dr0 = ARENA.get((size_t)L * h.plddtHidden); h.plddt1.backward(s.r0, dr1, dr0, L);
    reluBwdK<<<blocks((size_t)L * h.plddtHidden), 256, 0, STREAM>>>(dr0, s.h0, (size_t)L * h.plddtHidden);
    real* dln = ARENA.get((size_t)L * h.Cs); h.plddt0.backward(s.lnA, dr0, dln, L);
    lnBackward(h.plddtLn, s.ln, dln, dact, L, true);
  }
  if (d.pae && h.hasPae) h.pae.backward(pair, d.pae, dpair, pairs, true);
  if (d.distogram) {
    real* dh = ARENA.get(pairs * h.dgBins); symK<<<blocks(pairs * h.dgBins), 256, 0, STREAM>>>(d.distogram, dh, L, h.dgBins);
    h.dgHalf.backward(pair, dh, dpair, pairs, true);
  }
  if (d.maskedMsa) h.maskedMsa.backward(msa, d.maskedMsa, dmsa, (size_t)N * L, true);
  real* ds = ARENA.zeros((size_t)L * h.Cs);
  if (dsingleIn) add(ds, dsingleIn, (size_t)L * h.Cs);
  if (d.resolved) h.resolved.backward(s.singleOut, d.resolved, ds, L, true);
  h.single.backward(msa, ds, dmsa, L, true);              // msa row 0
}

}  // namespace af2
}  // namespace grad
