// The AF3 lineage's confidence head for training: the pair from the trunk's plus target_feat's left/right projections
// and the pseudo-beta distogram (the coordinates detached, as AF3 trains it), its pairformer blocks (af3.cuh), and the
// PDE, PAE and per-atom pLDDT logits - the logits being what a confidence loss reads. cuda/af3/src/confidence.cuh's
// confidenceHead() is what the forward is held to (through the expectations it returns).
//
// The generic path, with protenix2's clamped-and-normalised single and pre-symmetrised PDE (LN(z + z^T) W where AF3
// takes LN(z) W + its transpose). Not here: boltz2's re-embedding and split intra/inter-chain heads, rf3's global norms
// and CA distogram, chai-1's 16-bin distance embedding and ATOM37 pLDDT, OpenDDE's own head.
#pragma once
#include "af3.cuh"

namespace grad {
namespace af3 {

// the pseudo-beta distance's bin (AF3's 39 from 3.25 to 50.75 A, squared-distance bounds) and squared distance
__global__ void confBinK(const real* beta, int n, int bins, real dmin, real dmax, int* binOut, real* sqOut) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = ij / n, j = ij % n;
  double sq = 0;
  for (int k = 0; k < 3; ++k) { double d = (double)beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  // (the port squares in float: the bin of a distance within rounding of a bound follows its float32 square)
  float sqf = 0;
  for (int k = 0; k < 3; ++k) { float d = (float)beta[i * 3 + k] - (float)beta[j * 3 + k]; sqf += d * d; }
  int bin = -1;
  double dist = sqrt((double)sqf);
  int b0 = (int)((dist - dmin) * ((bins - 1) / (double)(dmax - dmin)));
  if (b0 < 0) b0 = 0; else if (b0 >= bins) b0 = bins - 1;
  for (int d = -1; d <= 1; ++d) {
    int b = b0 + d;
    if (b < 0 || b >= bins) continue;
    double lo = dmin + (double)(dmax - dmin) * b / (bins - 1), hi = dmin + (double)(dmax - dmin) * (b + 1) / (bins - 1);
    double lower = lo * lo, upper = b + 1 < bins ? hi * hi : 1e8;
    if (sqf > lower && sqf < upper) { bin = b; break; }
  }
  binOut[ij] = bin; sqOut[ij] = (real)sq;
}
// pair += left[j] + right[i] + Wd[bin] mask + |d| Wdist
__global__ void confPairInitK(real* pair, const real* left, const real* right, const int* bin, const real* sq, const real* pm,
                              const real* Wd, const real* Wdist, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / n, j = ij % n;
  real v = left[(size_t)j * C + c] + right[(size_t)i * C + c];
  if (bin[ij] >= 0) v += Wd[(size_t)bin[ij] * C + c] * pm[ij];
  if (Wdist) v += sqrt(sq[ij] + (real)1e-10) * Wdist[c];
  pair[t] += v;
}
// by (token, channel): dleft (from column j = token), dright (row i = token)
__global__ void confPairInitBwdLRK(const real* d, real* dl, real* dr, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int c = t % C, k = t / C;
  real sl = 0, sr = 0;
  for (int o = 0; o < n; ++o) { sl += d[((size_t)o * n + k) * C + c]; sr += d[((size_t)k * n + o) * C + c]; }
  dl[t] = sl; dr[t] = sr;
}
// by channel: the distogram's rows and the distance weight
__global__ void confPairInitBwdWK(const real* d, const int* bin, const real* sq, const real* pm, real* dWd, real* dWdist, int n, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  real sd = 0;
  for (size_t ij = 0; ij < (size_t)n * n; ++ij) {
    real g = d[ij * C + c];
    if (bin[ij] >= 0) dWd[(size_t)bin[ij] * C + c] += g * pm[ij];
    if (dWdist) sd += g * sqrt(sq[ij] + (real)1e-10);
  }
  if (dWdist) dWdist[c] += sd;
}
__global__ void clampFwdK(const real* x, real* y, real lim, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmin(fmax(x[t], -lim), lim);
}
__global__ void clampBwdK(const real* x, real* d, real lim, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n && (x[t] > lim || x[t] < -lim)) d[t] = 0;
}
// out = x + x^T (by pair), in and back
__global__ void pairSymK(const real* x, real* out, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / n, j = ij % n;
  out[t] = x[t] + x[((size_t)j * n + i) * C + c];
}

struct ConfidenceHead {
  int C = 0, Cs = 0, F = 0, bins = 39, dense = 24, NB = 64, PB = 50;
  bool preSym = false, singleNorm = false, hasDist = false, hasLogitsLn = false, hasPaeLn = false, hasPlddtLn = false;
  Linear leftTf, rightTf, pde, pae, plddt;
  Param distW, distF;
  LayerNorm logitsLn, paeLn, plddtLn, inSingleLn;
  std::vector<Pairformer> blocks;
  Dialect d;
};
// nb blocks with the pairformer dims (as test_af3.cu reads them from the bundle)
inline void forEachParam(ConfidenceHead& h, int nb, int Hp, int Dp, int Hs, int Ds, int pf, int sf, bool gridBiases, bool hasGrid,
                         bool parallel, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string P = "confidence.";
  auto ln = [&](const std::string& n, LayerNorm& l, int C) {
    l.C = C; f(n + "Scale", l.g, C); if (has(n + "Offset")) f(n + "Offset", l.b, C);
  };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; f(n, l.w, (size_t)K * N); };
  lin(P + "leftTargetFeatProject", h.leftTf, h.F, h.C); lin(P + "rightTargetFeatProject", h.rightTf, h.F, h.C);
  f(P + "distogramFeatProject", h.distW, (size_t)h.bins * h.C);
  h.hasDist = has(P + "distanceFeatProject"); if (h.hasDist) f(P + "distanceFeatProject", h.distF, h.C);
  h.singleNorm = has(P + "inputSingleNormScale"); if (h.singleNorm) ln(P + "inputSingleNorm", h.inSingleLn, h.Cs);
  h.blocks.assign(nb, Pairformer{});
  for (int k = 0; k < nb; ++k)
    forEachParam(h.blocks[k], P + "blocks." + std::to_string(k), h.C, h.Cs, Hp, Dp, Hs, Ds, pf, sf, gridBiases, true, f, hasGrid, parallel);
  h.hasLogitsLn = has(P + "logitsLnScale"); if (h.hasLogitsLn) ln(P + "logitsLn", h.logitsLn, h.C);
  lin(P + "leftHalfDistanceLogits", h.pde, h.C, h.NB);
  h.hasPaeLn = has(P + "paeLogitsLnScale"); if (h.hasPaeLn) ln(P + "paeLogitsLn", h.paeLn, h.C);
  lin(P + "paeLogits", h.pae, h.C, h.NB);
  h.hasPlddtLn = has(P + "plddtLnScale"); if (h.hasPlddtLn) ln(P + "plddtLn", h.plddtLn, h.Cs);
  lin(P + "plddtLogits", h.plddt, h.Cs, h.dense * h.PB);
}
struct ConfidenceSave {
  real *l, *r, *sq, *pair0, *single0, *clamped, *sym, *xn, *pn, *sn;
  int* bin;
  std::vector<real*> pairs, singles;     // each block's input (and the last one's output)
  LnSave lin_, lx, lp, ls;
  int n;
};
struct ConfidenceLogits { real *pde, *pae, *plddt; };    // [pairs, 64] (symmetrised), [pairs, 64], [n, dense * 50]
inline ConfidenceSave confidenceForward(ConfidenceHead& h, const real* trunkPair, const real* trunkSingle, const real* tf,
                                        const real* beta, const real* seqMask, const real* pairMask, const real* maskT, int n,
                                        ConfidenceLogits& o) {
  ConfidenceSave s{}; s.n = n;
  const size_t nn = (size_t)n * n;
  s.l = ARENA.get((size_t)n * h.C); h.leftTf.forward(tf, s.l, n);
  s.r = ARENA.get((size_t)n * h.C); h.rightTf.forward(tf, s.r, n);
  s.bin = (int*)ARENA.get(nn); s.sq = ARENA.get(nn);
  confBinK<<<blocks(nn), 256, 0, STREAM>>>(beta, n, h.bins, (real)3.25, (real)50.75, s.bin, s.sq);
  s.pairs.assign(h.blocks.size() + 1, nullptr); s.singles.assign(h.blocks.size() + 1, nullptr);
  s.pairs[0] = cloneR(trunkPair, nn * h.C);
  confPairInitK<<<blocks(nn * h.C), 256, 0, STREAM>>>(s.pairs[0], s.l, s.r, s.bin, s.sq, pairMask, h.distW.w,
                                                      h.hasDist ? h.distF.w : nullptr, n, h.C);
  if (h.singleNorm) {
    s.clamped = ARENA.get((size_t)n * h.Cs);
    clampFwdK<<<blocks((size_t)n * h.Cs), 256, 0, STREAM>>>(trunkSingle, s.clamped, (real)512, (size_t)n * h.Cs);
    s.singles[0] = ARENA.get((size_t)n * h.Cs); s.lin_ = lnForward(h.inSingleLn, s.clamped, s.singles[0], n);
  } else s.singles[0] = cloneR(trunkSingle, (size_t)n * h.Cs);
  for (size_t k = 0; k < h.blocks.size(); ++k) {
    s.pairs[k + 1] = ARENA.get(nn * h.C); s.singles[k + 1] = ARENA.get((size_t)n * h.Cs);
    Scratch sc;
    pairformerForward(h.blocks[k], h.d, s.pairs[k], s.singles[k], pairMask, maskT, seqMask, nullptr, s.pairs[k + 1],
                      s.singles[k + 1], n);
  }
  const real* z = s.pairs.back(); const real* sg = s.singles.back();
  // PDE: AF3 LN(z) W, symmetrised after; protenix2 LN(z + z^T) W
  const real* src = z;
  if (h.preSym) { s.sym = ARENA.get(nn * h.C); pairSymK<<<blocks(nn * h.C), 256, 0, STREAM>>>(z, s.sym, n, h.C); src = s.sym; }
  s.xn = const_cast<real*>(src);
  if (h.hasLogitsLn) { s.xn = ARENA.get(nn * h.C); s.lx = lnForward(h.logitsLn, src, s.xn, nn); }
  o.pde = ARENA.get(nn * h.NB);
  if (h.preSym) h.pde.forward(s.xn, o.pde, nn);
  else { real* half = ARENA.get(nn * h.NB); h.pde.forward(s.xn, half, nn); pairSymK<<<blocks(nn * h.NB), 256, 0, STREAM>>>(half, o.pde, n, h.NB); }
  s.pn = const_cast<real*>(z);
  if (h.hasPaeLn) { s.pn = ARENA.get(nn * h.C); s.lp = lnForward(h.paeLn, z, s.pn, nn); }
  o.pae = ARENA.get(nn * h.NB); h.pae.forward(s.pn, o.pae, nn);
  s.sn = const_cast<real*>(sg);
  if (h.hasPlddtLn) { s.sn = ARENA.get((size_t)n * h.Cs); s.ls = lnForward(h.plddtLn, sg, s.sn, n); }
  o.plddt = ARENA.get((size_t)n * h.dense * h.PB); h.plddt.forward(s.sn, o.plddt, n);
  return s;
}
// the logits' cotangents (any null) -> every parameter; dtrunkPair, dtrunkSingle, dtf (added; any may be null)
inline void confidenceBackward(ConfidenceHead& h, const ConfidenceSave& s, const real* trunkSingle, const real* tf, const real* seqMask,
                               const real* pairMask, const real* maskT, const ConfidenceLogits& d, real* dtrunkPair,
                               real* dtrunkSingle, real* dtf) {
  Scratch scratch;
  const int n = s.n; const size_t nn = (size_t)n * n;
  real* dz = ARENA.zeros(nn * h.C);
  real* dsg = ARENA.zeros((size_t)n * h.Cs);
  const real* z = s.pairs.back(); const real* sg = s.singles.back();
  if (d.plddt) {
    real* dsn = ARENA.get((size_t)n * h.Cs); h.plddt.backward(s.sn, d.plddt, dsn, n);
    if (h.hasPlddtLn) lnBackward(h.plddtLn, s.ls, dsn, dsg, n, true); else add(dsg, dsn, (size_t)n * h.Cs);
  }
  if (d.pae) {
    real* dpn = ARENA.get(nn * h.C); h.pae.backward(s.pn, d.pae, dpn, nn);
    if (h.hasPaeLn) lnBackward(h.paeLn, s.lp, dpn, dz, nn, true); else add(dz, dpn, nn * h.C);
  }
  if (d.pde) {
    real* dxn = ARENA.get(nn * h.C);
    if (h.preSym) h.pde.backward(s.xn, d.pde, dxn, nn);
    else { real* dh = ARENA.get(nn * h.NB); pairSymK<<<blocks(nn * h.NB), 256, 0, STREAM>>>(d.pde, dh, n, h.NB); h.pde.backward(s.xn, dh, dxn, nn); }
    real* dsrc = dxn;
    if (h.hasLogitsLn) { dsrc = ARENA.get(nn * h.C); lnBackward(h.logitsLn, s.lx, dxn, dsrc, nn, false); }
    if (h.preSym) { real* t = ARENA.get(nn * h.C); pairSymK<<<blocks(nn * h.C), 256, 0, STREAM>>>(dsrc, t, n, h.C); dsrc = t; }
    add(dz, dsrc, nn * h.C);
  }
  for (size_t k = h.blocks.size(); k-- > 0;) {
    Scratch sc;
    real *po = ARENA.get(nn * h.C), *so = ARENA.get((size_t)n * h.Cs);
    auto bs = pairformerForward(h.blocks[k], h.d, s.pairs[k], s.singles[k], pairMask, maskT, seqMask, nullptr, po, so, n);
    pairformerBackward(h.blocks[k], h.d, bs, pairMask, maskT, seqMask, dz, dsg, dz, dsg, nullptr);
  }
  (void)z; (void)sg;
  // the single's input
  if (h.singleNorm) {
    real* dc = ARENA.get((size_t)n * h.Cs); lnBackward(h.inSingleLn, s.lin_, dsg, dc, n, false);
    clampBwdK<<<blocks((size_t)n * h.Cs), 256, 0, STREAM>>>(trunkSingle, dc, (real)512, (size_t)n * h.Cs);
    if (dtrunkSingle) add(dtrunkSingle, dc, (size_t)n * h.Cs);
  } else if (dtrunkSingle) add(dtrunkSingle, dsg, (size_t)n * h.Cs);
  // the pair's: the trunk's, the projections, the distogram
  if (dtrunkPair) add(dtrunkPair, dz, nn * h.C);
  real *dl = ARENA.get((size_t)n * h.C), *dr = ARENA.get((size_t)n * h.C);
  confPairInitBwdLRK<<<blocks((size_t)n * h.C), 256, 0, STREAM>>>(dz, dl, dr, n, h.C);
  confPairInitBwdWK<<<blocks(h.C, 64), 64, 0, STREAM>>>(dz, s.bin, s.sq, pairMask, h.distW.g, h.hasDist ? h.distF.g : nullptr, n, h.C);
  h.leftTf.backward(tf, dl, dtf, n, dtf != nullptr);
  h.rightTf.backward(tf, dr, dtf, n, dtf != nullptr);
}

}  // namespace af3
}  // namespace grad
