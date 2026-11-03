// OpenDDE's structural-token stage for training (cuda/af3/src/structural.cuh is the forward it follows): after the
// trunk each standard residue becomes a backbone and a sidechain token, and
//   the expander:  s_inputs' = tf[parent] + roleIn[role];  single = ps + split2(silu(split1(LN(ps)))) + role[role],
//                  ps = trunkSingle[parent];  pair_ij = g + g W_{role_i, role_j} + the five pair-feature embeddings,
//                  g = trunkPair[parent_i, parent_j];  bias_ij = the same features' scalar weights
//   the refiner:   four pairformer blocks, the bias added to every single-attention head
//   its confidence head:  single = LN(clamp(single, 512)); pair += s1(s_inputs')[j] + s2(s_inputs')[i] + Wd[bin] + d Wraw
//                  (the sampler's per-token coordinates, detached); its pairformer blocks with the bias; the PDE
//                  LN(z + z^T) W, the PAE LN(z) W, the pLDDT LN(single) against each dense slot's matrix
// Every structural pair's projection matrix is one of 49; the forward sorts the pairs by it and runs one GEMM each, and
// the backward scatters each pair's gradient onto its residue pair (atomics: several structural pairs share one).
#pragma once
#include "af3.cuh"

namespace grad {
namespace dde {

constexpr int ROLES = 7;
struct Layout {
  int n = 0, nRes = 0;
  const int *parent, *role;                                       // [n]
  const int *sameParent, *twin, *prev, *next, *type;              // [n, n]
  std::vector<int> order, start;                                   // the pairs sorted by their matrix; [50]
  int* dOrder = nullptr;
};
inline void buildOrder(Layout& L, const int* roleHost) {
  const size_t pairs = (size_t)L.n * L.n;
  L.order.assign(pairs, 0); L.start.assign(ROLES * ROLES + 1, 0);
  for (size_t ij = 0; ij < pairs; ++ij) ++L.start[roleHost[ij / L.n] * ROLES + roleHost[ij % L.n] + 1];
  for (int k = 0; k < ROLES * ROLES; ++k) L.start[k + 1] += L.start[k];
  std::vector<int> fill(L.start.begin(), L.start.end() - 1);
  for (size_t ij = 0; ij < pairs; ++ij) L.order[fill[roleHost[ij / L.n] * ROLES + roleHost[ij % L.n]]++] = (int)ij;
  GK(cudaMalloc(&L.dOrder, pairs * sizeof(int)));
  GK(cudaMemcpy(L.dOrder, L.order.data(), pairs * sizeof(int), cudaMemcpyHostToDevice));
}

struct Expander {
  Param roleIn, role, pairProj, eSame, eTwin, ePrev, eNext, eType, bSame, bTwin, bPrev, bNext, bType;
  LayerNorm splitLn;
  Linear split1, split2;
  int F = 0, Cs = 0, C = 0, types = 0, flagRows = 2;
};
inline void forEachParam(Expander& x, int F, int Cs, int C, int types, const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string E = "expander.";
  x.F = F; x.Cs = Cs; x.C = C; x.types = types;
  f(E + "singleInputRoleEmbedding", x.roleIn, (size_t)ROLES * F);
  x.splitLn.C = Cs; f(E + "singleSplitNormScale", x.splitLn.g, Cs); f(E + "singleSplitNormOffset", x.splitLn.b, Cs);
  x.split1.K = Cs; x.split1.N = 2 * Cs; x.split1.hasBias = false; f(E + "singleSplit1", x.split1.w, (size_t)Cs * 2 * Cs);
  x.split2.K = 2 * Cs; x.split2.N = Cs; x.split2.hasBias = false; f(E + "singleSplit2", x.split2.w, (size_t)2 * Cs * Cs);
  f(E + "singleRoleEmbedding", x.role, (size_t)ROLES * Cs);
  f(E + "pairBlockProj", x.pairProj, (size_t)ROLES * ROLES * C * C);
  f(E + "sameParentEmbedding", x.eSame, (size_t)x.flagRows * C); f(E + "sameResidueTwinEmbedding", x.eTwin, (size_t)x.flagRows * C);
  f(E + "prevBbChainEmbedding", x.ePrev, (size_t)x.flagRows * C); f(E + "nextBbChainEmbedding", x.eNext, (size_t)x.flagRows * C);
  f(E + "rolePairTypeEmbedding", x.eType, (size_t)types * C);
  f(E + "attnBiasSameParent", x.bSame, 1); f(E + "attnBiasSameResidueTwin", x.bTwin, 1);
  f(E + "attnBiasPrevBbChain", x.bPrev, 1); f(E + "attnBiasNextBbChain", x.bNext, 1); f(E + "attnBiasRolePairType", x.bType, types);
}
// out[i] = src[parent[i]] (+ emb[role[i]]), and back (atomics: a parent has several tokens)
__global__ void gatherParentK(const real* src, const int* parent, const int* role, const real* emb, real* out, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int i = t / C, c = t % C;
  out[t] = src[(size_t)parent[i] * C + c] + (emb ? emb[(size_t)role[i] * C + c] : (real)0);
}
__global__ void gatherParentBwdK(const real* d, const int* parent, const int* role, real* dsrc, real* demb, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int i = t / C, c = t % C;
  if (dsrc) atomicAdd(&dsrc[(size_t)parent[i] * C + c], d[t]);
  if (demb) atomicAdd(&demb[(size_t)role[i] * C + c], d[t]);
}
__global__ void siluFK(const real* x, real* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = x[t] / (1 + exp(-x[t]));
}
__global__ void siluBK(const real* x, real* d, size_t n) {      // d *= silu'(x)
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real s = 1 / (1 + exp(-x[t]));
  d[t] *= s * (1 + x[t] * (1 - s));
}
// the sorted pairs' residue pairs, and back
__global__ void gatherPairSortedK(const real* pair, const int* order, const int* parent, real* out, int n, int nRes, int C, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = t % C;
  int ij = order[r], i = ij / n, j = ij % n;
  out[t] = pair[((size_t)parent[i] * nRes + parent[j]) * C + c];
}
__global__ void scatterPairSortedK(const real* d, const int* order, const int* parent, real* dpair, int n, int nRes, int C, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = t % C;
  int ij = order[r], i = ij / n, j = ij % n;
  atomicAdd(&dpair[((size_t)parent[i] * nRes + parent[j]) * C + c], d[t]);
}
// pair[order[r]] = gathered[r] + projected[r] + the five embeddings; the sorted rows' gradient back out of the pair's
__global__ void scatterFeatK(real* pair, const real* gathered, const real* projected, const int* order, const int* sp, const int* tw,
                             const int* pv, const int* nx, const int* ty, const real* eS, const real* eT, const real* eP,
                             const real* eN, const real* eY, int C, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = t % C;
  int ij = order[r];
  pair[(size_t)ij * C + c] = gathered[t] + projected[t] + eS[sp[ij] * C + c] + eT[tw[ij] * C + c] + eP[pv[ij] * C + c]
                           + eN[nx[ij] * C + c] + eY[ty[ij] * C + c];
}
__global__ void sortRowsK(const real* d, const int* order, real* out, int C, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  out[t] = d[(size_t)order[t / C] * C + t % C];
}
// the five tables' gradients, one thread a (table row, channel) over every pair
__global__ void featBwdK(const real* d, const int* sp, const int* tw, const int* pv, const int* nx, const int* ty, real* gS,
                         real* gT, real* gP, real* gN, real* gY, size_t pairs, int C, int flagRows, int types) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int rows = 4 * flagRows + types;
  if (t >= (size_t)rows * C) return;
  int k = t / C, c = t % C;
  int table = k < 4 * flagRows ? k / flagRows : 4, row = k < 4 * flagRows ? k % flagRows : k - 4 * flagRows;
  const int* idx = table == 0 ? sp : table == 1 ? tw : table == 2 ? pv : table == 3 ? nx : ty;
  real acc = 0;
  for (size_t ij = 0; ij < pairs; ++ij) if (idx[ij] == row) acc += d[ij * C + c];
  real* g = table == 0 ? gS : table == 1 ? gT : table == 2 ? gP : table == 3 ? gN : gY;
  g[(size_t)row * C + c] += acc;
}
__global__ void attnBiasK(real* bias, const int* sp, const int* tw, const int* pv, const int* nx, const int* ty, const real* bS,
                          const real* bT, const real* bP, const real* bN, const real* bY, size_t pairs) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij < pairs) bias[ij] = bS[0] * sp[ij] + bT[0] * tw[ij] + bP[0] * pv[ij] + bN[0] * nx[ij] + bY[ty[ij]];
}
__global__ void attnBiasBwdK(const real* d, const int* sp, const int* tw, const int* pv, const int* nx, const int* ty, real* gS,
                             real* gT, real* gP, real* gN, real* gY, size_t pairs, int types) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= 4 + types) return;
  real acc = 0;
  for (size_t ij = 0; ij < pairs; ++ij) {
    real g = d[ij];
    if (k == 0) acc += g * sp[ij]; else if (k == 1) acc += g * tw[ij]; else if (k == 2) acc += g * pv[ij];
    else if (k == 3) acc += g * nx[ij]; else if (ty[ij] == k - 4) acc += g;
  }
  (k == 0 ? gS : k == 1 ? gT : k == 2 ? gP : k == 3 ? gN : gY + (k - 4))[0] += acc;
}
struct ExpandSave { real *ps, *ln, *h, *hs; LnSave lsp; };
// trunkSingle [nRes, Cs], trunkPair [nRes nRes, C], tf [nRes, F] -> sIn [n, F], single [n, Cs], pair [n n, C], bias [n n]
inline ExpandSave expandForward(Expander& x, const Layout& L, const real* trunkSingle, const real* trunkPair, const real* tf,
                                real* sIn, real* single, real* pair, real* bias) {
  ExpandSave s{};
  const int n = L.n, Cs = x.Cs, C = x.C; const size_t pairs = (size_t)n * n;
  gatherParentK<<<blocks((size_t)n * x.F), 256, 0, STREAM>>>(tf, L.parent, L.role, x.roleIn.w, sIn, n, x.F);
  s.ps = ARENA.get((size_t)n * Cs);
  gatherParentK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(trunkSingle, L.parent, L.role, nullptr, s.ps, n, Cs);
  s.ln = ARENA.get((size_t)n * Cs); s.lsp = lnForward(x.splitLn, s.ps, s.ln, n);
  s.h = ARENA.get((size_t)n * 2 * Cs); x.split1.forward(s.ln, s.h, n);
  s.hs = ARENA.get((size_t)n * 2 * Cs); siluFK<<<blocks((size_t)n * 2 * Cs), 256, 0, STREAM>>>(s.h, s.hs, (size_t)n * 2 * Cs);
  x.split2.forward(s.hs, single, n);
  add(single, s.ps, (size_t)n * Cs);
  {
    // single += role[role]: a gather of the table by the role
    Scratch sc;
    real* re = ARENA.get((size_t)n * Cs);
    gatherParentK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(x.role.w, L.role, L.role, nullptr, re, n, Cs);
    add(single, re, (size_t)n * Cs);
  }
  Scratch sc;
  real *g = ARENA.get(pairs * C), *p = ARENA.get(pairs * C);
  gatherPairSortedK<<<blocks(pairs * C), 256, 0, STREAM>>>(trunkPair, L.dOrder, L.parent, g, n, L.nRes, C, pairs);
  for (int k = 0; k < ROLES * ROLES; ++k) {
    const size_t lo = L.start[k], hi = L.start[k + 1];
    if (lo < hi) mm(g + lo * C, x.pairProj.w + (size_t)k * C * C, p + lo * C, (int)(hi - lo), C, C);
  }
  scatterFeatK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, g, p, L.dOrder, L.sameParent, L.twin, L.prev, L.next, L.type, x.eSame.w,
                                                     x.eTwin.w, x.ePrev.w, x.eNext.w, x.eType.w, C, pairs);
  attnBiasK<<<blocks(pairs), 256, 0, STREAM>>>(bias, L.sameParent, L.twin, L.prev, L.next, L.type, x.bSame.w, x.bTwin.w, x.bPrev.w,
                                               x.bNext.w, x.bType.w, pairs);
  return s;
}
// dsIn, dsingle, dpair, dbias -> every parameter; dtrunkSingle, dtrunkPair, dtf added (any may be null)
inline void expandBackward(Expander& x, const Layout& L, const ExpandSave& s, const real* trunkPair, const real* dsIn,
                           const real* dsingle, const real* dpair, const real* dbias, real* dtrunkSingle, real* dtrunkPair, real* dtf) {
  Scratch scratch;
  const int n = L.n, Cs = x.Cs, C = x.C; const size_t pairs = (size_t)n * n;
  if (dsIn) gatherParentBwdK<<<blocks((size_t)n * x.F), 256, 0, STREAM>>>(dsIn, L.parent, L.role, dtf, x.roleIn.g, n, x.F);
  if (dsingle) {
    gatherParentBwdK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(dsingle, L.role, L.role, x.role.g, nullptr, n, Cs);
    real* dhs = ARENA.get((size_t)n * 2 * Cs); x.split2.backward(s.hs, dsingle, dhs, n);
    siluBK<<<blocks((size_t)n * 2 * Cs), 256, 0, STREAM>>>(s.h, dhs, (size_t)n * 2 * Cs);
    real* dln = ARENA.get((size_t)n * Cs); x.split1.backward(s.ln, dhs, dln, n);
    real* dps = cloneR(dsingle, (size_t)n * Cs);
    lnBackward(x.splitLn, s.lsp, dln, dps, n, true);
    if (dtrunkSingle) gatherParentBwdK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(dps, L.parent, L.role, dtrunkSingle, nullptr, n, Cs);
  }
  if (dbias) attnBiasBwdK<<<1, 64, 0, STREAM>>>(dbias, L.sameParent, L.twin, L.prev, L.next, L.type, x.bSame.g, x.bTwin.g,
                                               x.bPrev.g, x.bNext.g, x.bType.g, pairs, x.types);
  if (!dpair) return;
  featBwdK<<<blocks((size_t)(4 * x.flagRows + x.types) * C, 64), 64, 0, STREAM>>>(dpair, L.sameParent, L.twin, L.prev, L.next, L.type,
      x.eSame.g, x.eTwin.g, x.ePrev.g, x.eNext.g, x.eType.g, pairs, C, x.flagRows, x.types);
  real *g = ARENA.get(pairs * C), *dp = ARENA.get(pairs * C), *dg = ARENA.get(pairs * C);
  gatherPairSortedK<<<blocks(pairs * C), 256, 0, STREAM>>>(trunkPair, L.dOrder, L.parent, g, n, L.nRes, C, pairs);
  sortRowsK<<<blocks(pairs * C), 256, 0, STREAM>>>(dpair, L.dOrder, dp, C, pairs);
  GK(cudaMemcpyAsync(dg, dp, pairs * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  for (int k = 0; k < ROLES * ROLES; ++k) {
    const size_t lo = L.start[k], hi = L.start[k + 1];
    if (lo >= hi) continue;
    mmAT(g + lo * C, dp + lo * C, x.pairProj.g + (size_t)k * C * C, (int)(hi - lo), C, C);
    mmBT(dp + lo * C, x.pairProj.w + (size_t)k * C * C, dg + lo * C, (int)(hi - lo), C, C, 1);
  }
  if (dtrunkPair) scatterPairSortedK<<<blocks(pairs * C), 256, 0, STREAM>>>(dg, L.dOrder, L.parent, dtrunkPair, n, L.nRes, C, pairs);
}

// ---------------------------------------------------------------- the refiner: pairformer blocks with the bias
struct RefineRun { std::vector<real*> pairs, singles; int n = 0; };
inline void refineForward(std::vector<af3::Pairformer>& bs, const af3::Dialect& d, RefineRun& r, real* pair, real* single,
                          const real* pairMask, const real* seqMask, const real* bias, int n) {
  const int C = bs[0].triOut.C, Cs = bs[0].single.C; const size_t nn = (size_t)n * n;
  r.n = n; r.pairs.assign(bs.size(), nullptr); r.singles.assign(bs.size(), nullptr);
  for (size_t k = 0; k < bs.size(); ++k) {
    r.pairs[k] = cloneR(pair, nn * C); r.singles[k] = cloneR(single, (size_t)n * Cs);
    Scratch sc;
    af3::pairformerForward(bs[k], d, r.pairs[k], r.singles[k], pairMask, pairMask, seqMask, bias, pair, single, n);
  }
}
// dpair, dsingle: the outputs' gradients, replaced by the inputs'; dbias added
inline void refineBackward(std::vector<af3::Pairformer>& bs, const af3::Dialect& d, const RefineRun& r, real* dpair, real* dsingle,
                           const real* pairMask, const real* seqMask, const real* bias, real* dbias) {
  const int n = r.n, C = bs[0].triOut.C, Cs = bs[0].single.C; const size_t nn = (size_t)n * n;
  for (size_t k = bs.size(); k-- > 0;) {
    Scratch sc;
    real *po = ARENA.get(nn * C), *so = ARENA.get((size_t)n * Cs);
    auto s = af3::pairformerForward(bs[k], d, r.pairs[k], r.singles[k], pairMask, pairMask, seqMask, bias, po, so, n);
    af3::pairformerBackward(bs[k], d, s, pairMask, pairMask, seqMask, dpair, dsingle, dpair, dsingle, dbias);
  }
}

// ---------------------------------------------------------------- OpenDDE's confidence head
struct Confidence {
  LayerNorm inLn, pdeLn, paeLn, plddtLn;
  Linear s1, s2, pde, pae;
  Param dist, distRaw, plddtW;       // [bins, C], [C], [slots, Cs, PB]
  std::vector<af3::Pairformer> blocks;
  int C = 0, Cs = 0, F = 0, bins = 0, slots = 0, PB = 50, paeBins = 64, pdeBins = 64;
};
inline void forEachParam(Confidence& h, int nb, int Hp, int Dp, int Hs, int Ds, int pf, int sf, bool gridBiases,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string P = "ddeConfidence.";
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(P + n + "Scale", l.g, C); f(P + n + "Offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; f(P + n, l.w, (size_t)K * N); };
  ln("inputStrunkLn", h.inLn, h.Cs);
  lin("s1", h.s1, h.F, h.C); lin("s2", h.s2, h.F, h.C);
  f(P + "distance", h.dist, (size_t)h.bins * h.C); f(P + "distanceRaw", h.distRaw, h.C);
  h.blocks.assign(nb, af3::Pairformer{});
  for (int k = 0; k < nb; ++k)
    af3::forEachParam(h.blocks[k], P + "blocks." + std::to_string(k), h.C, h.Cs, Hp, Dp, Hs, Ds, pf, sf, gridBiases, true, f);
  ln("pdeLn", h.pdeLn, h.C); lin("pde", h.pde, h.C, h.pdeBins);
  ln("paeLn", h.paeLn, h.C); lin("pae", h.pae, h.C, h.paeBins);
  ln("plddtLn", h.plddtLn, h.Cs);
  f(P + "plddtWeight", h.plddtW, (size_t)h.slots * h.Cs * h.PB);
}
__global__ void ddeDistK(const real* x, real* dist, int* bin, int n, int bins) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = ij / n, j = ij % n;
  float sq = 0;
  for (int k = 0; k < 3; ++k) { float d = (float)x[i * 3 + k] - (float)x[j * 3 + k]; sq += d * d; }
  float distance = sqrtf(fmaxf(1e-10f, sq));
  int b = (int)floorf((distance - 3.25f) * 0.8f);
  if (distance < 3.25f) b = -1;
  if (b >= bins) b = bins - 1;
  dist[ij] = distance; bin[ij] = b;
}
__global__ void ddeInitK(real* pair, const real* s1, const real* s2, const real* dist, const int* bin, const real* Wd, const real* Wr,
                         int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / n, j = ij % n;
  pair[t] += s1[(size_t)j * C + c] + s2[(size_t)i * C + c] + (bin[ij] >= 0 ? Wd[(size_t)bin[ij] * C + c] : (real)0) + dist[ij] * Wr[c];
}
__global__ void ddeInitBwdK(const real* d, const real* dist, const int* bin, real* ds1, real* ds2, real* dWd, real* dWr, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int c = t % C, k = t / C;
  real a = 0, b = 0;
  for (int o = 0; o < n; ++o) { a += d[((size_t)o * n + k) * C + c]; b += d[((size_t)k * n + o) * C + c]; }
  ds1[t] = a; ds2[t] = b;
  if (k == 0) {
    real r = 0;
    for (size_t ij = 0; ij < (size_t)n * n; ++ij) {
      real g = d[ij * C + c]; r += dist[ij] * g;
      if (bin[ij] >= 0) dWd[(size_t)bin[ij] * C + c] += g;
    }
    dWr[c] += r;
  }
}
__global__ void ddeSymK(const real* x, real* out, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / n, j = ij % n;
  out[t] = x[t] + x[((size_t)j * n + i) * C + c];
}
__global__ void clampK(const real* x, real* y, real lim, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmin(fmax(x[t], -lim), lim);
}
__global__ void clampBK(const real* x, real* d, real lim, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n && (x[t] > lim || x[t] < -lim)) d[t] = 0;
}
// plddt_weight [slot][Cs][bin] as one [Cs, slots bins] matrix, and back
__global__ void slotMajorK(const real* w, real* out, int slots, int C, int bins, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)slots * C * bins) return;
  int b = t % bins; size_t r = t / bins; int c = r % C, slot = r / C;
  size_t o = (size_t)c * slots * bins + slot * bins + b;
  if (back) out[t] += w[o]; else out[o] = w[t];
}
struct ConfLogits { real *pde, *pae, *plddt; };    // [n n, pdeBins], [n n, paeBins], [n, slots PB]
struct ConfSave { real *single0, *sn, *pair0, *s1, *s2, *dist, *sym, *symn, *pn, *pln, *wslot; int* bin; LnSave lin_, lpde, lpae, lpl;
                  std::vector<real*> pairs, singles; int n; };
// refined pair [n n, C], single [n, Cs], sIn [n, F], the sampler's coordinates [n, 3] (detached) -> the logits
inline ConfSave confForward(Confidence& h, const af3::Dialect& d, const real* refinedPair, const real* refinedSingle, const real* sIn,
                            const real* coords, const real* seqMask, const real* pairMask, const real* bias, int n, ConfLogits& o) {
  ConfSave s{}; s.n = n;
  const size_t nn = (size_t)n * n;
  s.single0 = ARENA.get((size_t)n * h.Cs); clampK<<<blocks((size_t)n * h.Cs), 256, 0, STREAM>>>(refinedSingle, s.single0, 512, (size_t)n * h.Cs);
  s.pairs.assign(h.blocks.size() + 1, nullptr); s.singles.assign(h.blocks.size() + 1, nullptr);
  s.singles[0] = ARENA.get((size_t)n * h.Cs); s.lin_ = lnForward(h.inLn, s.single0, s.singles[0], n);
  s.s1 = ARENA.get((size_t)n * h.C); h.s1.forward(sIn, s.s1, n);
  s.s2 = ARENA.get((size_t)n * h.C); h.s2.forward(sIn, s.s2, n);
  s.dist = ARENA.get(nn); s.bin = (int*)ARENA.get(nn);
  ddeDistK<<<blocks(nn), 256, 0, STREAM>>>(coords, s.dist, s.bin, n, h.bins);
  s.pairs[0] = cloneR(refinedPair, nn * h.C);
  ddeInitK<<<blocks(nn * h.C), 256, 0, STREAM>>>(s.pairs[0], s.s1, s.s2, s.dist, s.bin, h.dist.w, h.distRaw.w, n, h.C);
  for (size_t k = 0; k < h.blocks.size(); ++k) {
    s.pairs[k + 1] = ARENA.get(nn * h.C); s.singles[k + 1] = ARENA.get((size_t)n * h.Cs);
    Scratch sc;
    af3::pairformerForward(h.blocks[k], d, s.pairs[k], s.singles[k], pairMask, pairMask, seqMask, bias, s.pairs[k + 1], s.singles[k + 1], n);
  }
  const real* z = s.pairs.back(); const real* sg = s.singles.back();
  s.sym = ARENA.get(nn * h.C); ddeSymK<<<blocks(nn * h.C), 256, 0, STREAM>>>(z, s.sym, n, h.C);
  s.symn = ARENA.get(nn * h.C); s.lpde = lnForward(h.pdeLn, s.sym, s.symn, nn);
  o.pde = ARENA.get(nn * h.pdeBins); h.pde.forward(s.symn, o.pde, nn);
  s.pn = ARENA.get(nn * h.C); s.lpae = lnForward(h.paeLn, z, s.pn, nn);
  o.pae = ARENA.get(nn * h.paeBins); h.pae.forward(s.pn, o.pae, nn);
  s.pln = ARENA.get((size_t)n * h.Cs); s.lpl = lnForward(h.plddtLn, sg, s.pln, n);
  s.wslot = ARENA.get((size_t)h.slots * h.Cs * h.PB);
  slotMajorK<<<blocks((size_t)h.slots * h.Cs * h.PB), 256, 0, STREAM>>>(h.plddtW.w, s.wslot, h.slots, h.Cs, h.PB, false);
  o.plddt = ARENA.get((size_t)n * h.slots * h.PB); mm(s.pln, s.wslot, o.plddt, n, h.Cs, h.slots * h.PB);
  return s;
}
// the logits' cotangents (any null) -> every parameter; drefinedPair, drefinedSingle, dsIn added (any null), dbias added
inline void confBackward(Confidence& h, const af3::Dialect& d, const ConfSave& s, const real* refinedSingle, const real* sIn,
                         const real* seqMask, const real* pairMask, const real* bias, const ConfLogits& g, real* dpairIn,
                         real* dsingleIn, real* dsIn, real* dbias) {
  Scratch scratch;
  const int n = s.n; const size_t nn = (size_t)n * n;
  real* dz = ARENA.zeros(nn * h.C); real* dsg = ARENA.zeros((size_t)n * h.Cs);
  if (g.plddt) {
    real* dw = ARENA.zeros((size_t)h.slots * h.Cs * h.PB);
    mmAT(s.pln, g.plddt, dw, n, h.Cs, h.slots * h.PB);
    slotMajorK<<<blocks((size_t)h.slots * h.Cs * h.PB), 256, 0, STREAM>>>(dw, h.plddtW.g, h.slots, h.Cs, h.PB, true);
    real* dpl = ARENA.get((size_t)n * h.Cs); mmBT(g.plddt, s.wslot, dpl, n, h.Cs, h.slots * h.PB);
    lnBackward(h.plddtLn, s.lpl, dpl, dsg, n, true);
  }
  if (g.pae) { real* dpn = ARENA.get(nn * h.C); h.pae.backward(s.pn, g.pae, dpn, nn); lnBackward(h.paeLn, s.lpae, dpn, dz, nn, true); }
  if (g.pde) {
    real* dsn = ARENA.get(nn * h.C); h.pde.backward(s.symn, g.pde, dsn, nn);
    real* dsym = ARENA.get(nn * h.C); lnBackward(h.pdeLn, s.lpde, dsn, dsym, nn, false);
    real* t = ARENA.get(nn * h.C); ddeSymK<<<blocks(nn * h.C), 256, 0, STREAM>>>(dsym, t, n, h.C); add(dz, t, nn * h.C);
  }
  for (size_t k = h.blocks.size(); k-- > 0;) {
    Scratch sc;
    real *po = ARENA.get(nn * h.C), *so = ARENA.get((size_t)n * h.Cs);
    auto bs = af3::pairformerForward(h.blocks[k], d, s.pairs[k], s.singles[k], pairMask, pairMask, seqMask, bias, po, so, n);
    af3::pairformerBackward(h.blocks[k], d, bs, pairMask, pairMask, seqMask, dz, dsg, dz, dsg, dbias);
  }
  real *ds1 = ARENA.get((size_t)n * h.C), *ds2 = ARENA.get((size_t)n * h.C);
  ddeInitBwdK<<<blocks((size_t)n * h.C), 256, 0, STREAM>>>(dz, s.dist, s.bin, ds1, ds2, h.dist.g, h.distRaw.g, n, h.C);
  if (dsIn) { h.s1.backward(sIn, ds1, dsIn, n, true); h.s2.backward(sIn, ds2, dsIn, n, true); }
  else { h.s1.backward(sIn, ds1, nullptr, n); h.s2.backward(sIn, ds2, nullptr, n); }
  if (dpairIn) add(dpairIn, dz, nn * h.C);
  real* dc = ARENA.get((size_t)n * h.Cs); lnBackward(h.inLn, s.lin_, dsg, dc, n, false);
  clampBK<<<blocks((size_t)n * h.Cs), 256, 0, STREAM>>>(refinedSingle, dc, 512, (size_t)n * h.Cs);
  if (dsingleIn) add(dsingleIn, dc, (size_t)n * h.Cs);
}

}  // namespace dde
}  // namespace grad
