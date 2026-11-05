// The AF3 lineage's confidence head for training: the pair from the trunk's plus target_feat's left/right projections
// and the pseudo-beta distogram (the coordinates detached, as AF3 trains it), its pairformer blocks (af3.cuh), and the
// PDE, PAE and per-atom pLDDT logits - the logits being what a confidence loss reads. cuda/af3/src/confidence.cuh's
// confidenceHead() is what the forward is held to (through the expectations it returns).
//
// The generic path, with protenix2's clamped-and-normalised single and pre-symmetrised PDE (LN(z + z^T) W where AF3
// takes LN(z) W + its transpose); boltz2's re-embedded pair and single (its LayerNorms, the relative encoding, the
// s_inputs projections and their outer product, its 64-bin distance embedding, the bond and bond-order terms) and split
// intra/inter-chain heads; rf3's parameter-free global norms of every input (target_feat's over the vendor's 449
// columns, its two absent columns at their normalised value) and its CA distogram; chai-1's 16-bin unmasked distance
// embedding and its pLDDT over the 37 ATOM37 slots (the logits as they are; the gather to the dense slots is a
// loss's). Not here: OpenDDE's own head.
#pragma once
#include "af3.cuh"
#include "af3denoise.cuh"

namespace grad {
namespace af3 {

// the pseudo-beta distance's bin (AF3's 39 from 3.25 to 50.75 A, squared-distance bounds) and squared distance;
// mode 1 rf3's (how many of bins - 1 even bounds the distance is past), 2 chai-1's (16, from 3.375 by 18 / 14), 3
// boltz2's re-embedding (64 over 2..22 A) - each as cuda/af3's confidenceBinK / reembedBinK takes it
enum { BIN_AF3 = 0, BIN_CA = 1, BIN_CHAI = 2, BIN_REEMBED = 3 };
__global__ void confBinK(const real* beta, int n, int bins, real dmin, real dmax, int* binOut, real* sqOut, int mode = BIN_AF3) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = ij / n, j = ij % n;
  double sq = 0;
  for (int k = 0; k < 3; ++k) { double d = (double)beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  // (the port squares in float: the bin of a distance within rounding of a bound follows its float32 square)
  float sqf = 0;
  for (int k = 0; k < 3; ++k) { float d = (float)beta[i * 3 + k] - (float)beta[j * 3 + k]; sqf += d * d; }
  int bin = -1;
  if (mode == BIN_CHAI) {
    float distance = sqrtf(sqf + 1e-10f);
    bin = 0;
    for (int at = 0; at < bins - 1; ++at) bin += distance > 3.375f + at * (18.f / (bins - 2));
    binOut[ij] = bin; sqOut[ij] = (real)sq; return;
  }
  if (mode == BIN_CA || mode == BIN_REEMBED) {
    double distance, lo, step;
    if (mode == BIN_CA) { distance = sqrt((double)sqf + 1e-10); lo = dmin; step = (double)(dmax - dmin) / (bins - 1); }
    else {
      double s2 = 1e-10;
      for (int k = 0; k < 3; ++k) { double d = (double)(float)beta[i * 3 + k] - (double)(float)beta[j * 3 + k]; s2 += d * d; }
      distance = sqrt(s2); lo = 2.0; step = 20.0 / 62; bins = 64;
    }
    bin = (int)((distance - lo) / step);
    if (bin < 0) bin = 0; else if (bin > bins - 1) bin = bins - 1;
    while (bin > 0 && !(distance > lo + (bin - 1) * step)) --bin;
    while (bin < bins - 1 && distance > lo + bin * step) ++bin;
    binOut[ij] = bin; sqOut[ij] = (real)sq; return;
  }
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
  if (bin[ij] >= 0) v += Wd[(size_t)bin[ij] * C + c] * (pm ? pm[ij] : (real)1);
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
    if (bin[ij] >= 0) dWd[(size_t)bin[ij] * C + c] += g * (pm ? pm[ij] : (real)1);
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

// boltz2's re-embedded pair terms: bonds * wb + wbt[order] + unspecified, and back (one thread a channel)
__global__ void reembedTermsK(real* pair, const real* bonds, const real* orders, const real* wb, const real* wbt,
                              const real* un, size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  int c = t % C; size_t ij = t / C;
  int o = orders ? (int)orders[ij] : 0;
  if (o < 0 || o >= 7) o = 0;
  pair[t] += (bonds ? bonds[ij] * wb[c] : 0) + wbt[(size_t)o * C + c] + un[c];
}
__global__ void reembedTermsBwdK(const real* d, const real* bonds, const real* orders, real* dwb, real* dwbt, real* dun,
                                 size_t pairs, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  real sb = 0, su = 0;
  for (size_t ij = 0; ij < pairs; ++ij) {
    real g = d[ij * C + c];
    int o = orders ? (int)orders[ij] : 0;
    if (o < 0 || o >= 7) o = 0;
    dwbt[(size_t)o * C + c] += g;
    if (bonds) sb += bonds[ij] * g;
    su += g;
  }
  if (bonds) dwb[c] += sb;
  dun[c] += su;
}
// pair[i, j] += a[i] * b[j] (by channel), and back
__global__ void outerRowsK(const real* a, const real* b, real* out, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = t % C; size_t ij = t / C; int i = ij / n, j = ij % n;
  out[t] = a[(size_t)i * C + c] * b[(size_t)j * C + c];
}
__global__ void outerRowsBwdK(const real* d, const real* a, const real* b, real* da, real* db, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * C) return;
  int c = t % C, k = t / C;
  real sa = 0, sb = 0;
  for (int o = 0; o < n; ++o) { sa += d[((size_t)k * n + o) * C + c] * b[(size_t)o * C + c]; sb += d[((size_t)o * n + k) * C + c] * a[(size_t)o * C + c]; }
  da[t] = sa; db[t] = sb;
}
// the split heads: where the two tokens' chains differ the inter-chain logits, else the intra; and back
__global__ void interMergeK(real* logits, const real* inter, const int* asym, int n, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * bins) return;
  size_t ij = t / bins;
  if (asym[ij / n] != asym[ij % n]) logits[t] = inter[t];
}
__global__ void interSplitK(const real* d, real* dIntra, real* dInter, const int* asym, int n, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * bins) return;
  size_t ij = t / bins;
  bool inter = asym[ij / n] != asym[ij % n];
  dIntra[t] = inter ? 0 : d[t]; dInter[t] = inter ? d[t] : 0;
}

// ---------------------------------------------------------------- rf3's parameter-free global LayerNorm
// y = (x - mean) / sqrt(var + 1e-5) for every element; the statistics over the live rows' `width` columns, the ones past
// C zero (each adding mean^2 to the variance). Its backward: var's own derivative in the mean vanishes, so a live
// element takes dmean / N + dvar 2 (x - mean) / N, dmean = -istd sum dy, dvar = -istd^3 / 2 sum dy (x - mean).
struct GlobalNorm { double mean = 0, istd = 1, count = 1; int C = 0, width = 0; };
__global__ void gnSumsK(const real* x, const real* mask, size_t rows, int C, double mean, int which, double* partial) {
  __shared__ double red[256];
  double acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * C; i += (size_t)gridDim.x * blockDim.x) {
    real m = mask[i / C];
    if (which == 2) { if (i % C == 0 && m > 0) acc += 1; continue; }
    if (!(m > 0)) continue;
    double v = x[i];
    acc += which == 0 ? v : (v - mean) * (v - mean);
  }
  red[threadIdx.x] = acc; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) { if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w]; __syncthreads(); }
  if (threadIdx.x == 0) partial[blockIdx.x] = red[0];
}
inline double gnSum(const real* x, const real* mask, size_t rows, int C, double mean, int which) {
  Scratch sc;
  const int parts = 256;
  double* p = (double*)ARENA.get(parts * sizeof(double) / sizeof(real) + 1);
  gnSumsK<<<parts, 256, 0, STREAM>>>(x, mask, rows, C, mean, which, p);
  std::vector<double> h(parts); GK(cudaMemcpy(h.data(), p, parts * sizeof(double), cudaMemcpyDeviceToHost));
  double s = 0; for (double v : h) s += v;
  return s;
}
__global__ void gnApplyK(const real* x, real* y, double mean, double istd, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = (real)((x[t] - mean) * istd);
}
inline GlobalNorm gnForward(const real* x, const real* mask, size_t rows, int C, int width, real* y) {
  GlobalNorm g; g.C = C; g.width = width;
  double live = gnSum(x, mask, rows, C, 0, 2);
  g.count = std::max(live * width, 1.0);
  g.mean = gnSum(x, mask, rows, C, 0, 0) / g.count;
  double var = gnSum(x, mask, rows, C, g.mean, 1) + (double)(width - C) * live * g.mean * g.mean;
  g.istd = 1 / std::sqrt(var / g.count + 1e-5);
  gnApplyK<<<blocks(rows * C), 256, 0, STREAM>>>(x, y, g.mean, g.istd, rows * C);
  return g;
}
__global__ void gnDotsK(const real* dy, const real* x, double mean, size_t n, double* partial) {
  __shared__ double r0[256], r1[256];
  double a = 0, b = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    a += dy[i]; b += dy[i] * (x[i] - mean);
  }
  r0[threadIdx.x] = a; r1[threadIdx.x] = b; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) {
    if (threadIdx.x < w) { r0[threadIdx.x] += r0[threadIdx.x + w]; r1[threadIdx.x] += r1[threadIdx.x + w]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { partial[2 * blockIdx.x] = r0[0]; partial[2 * blockIdx.x + 1] = r1[0]; }
}
__global__ void gnBackK(const real* dy, const real* x, const real* mask, real* dx, double istd, double a, double b,
                        double mean, size_t rows, int C, bool accumulate) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  double v = dy[t] * istd;
  if (mask[t / C] > 0) v += a + b * (x[t] - mean);
  if (accumulate) dx[t] += (real)v; else dx[t] = (real)v;
}
// dy -> dx; extraMean, extraIstd: further cotangents on the statistics themselves (rf3's absent columns)
inline void gnBackward(const GlobalNorm& g, const real* x, const real* mask, size_t rows, const real* dy, real* dx,
                       bool accumulate, double extraMean = 0, double extraIstd = 0) {
  Scratch sc;
  const int parts = 256;
  double* p = (double*)ARENA.get(2 * parts * sizeof(double) / sizeof(real) + 1);
  gnDotsK<<<parts, 256, 0, STREAM>>>(dy, x, g.mean, rows * g.C, p);
  std::vector<double> h(2 * parts); GK(cudaMemcpy(h.data(), p, 2 * parts * sizeof(double), cudaMemcpyDeviceToHost));
  double sdy = 0, sdyx = 0; for (int k = 0; k < parts; ++k) { sdy += h[2 * k]; sdyx += h[2 * k + 1]; }
  const double dmean = -g.istd * sdy + extraMean;
  const double distd = sdyx + extraIstd;
  const double dvar = -0.5 * g.istd * g.istd * g.istd * distd;
  gnBackK<<<blocks(rows * g.C), 256, 0, STREAM>>>(dy, x, mask, dx, g.istd, dmean / g.count, 2 * dvar / g.count, g.mean,
                                                  rows, g.C, accumulate);
}
// rf3's absent columns: left[t, c] += v (w[c] + w[C + c]), v = -mean istd of target_feat's norm
__global__ void absentK(real* out, const real* w, real v, size_t n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n * C) out[t] += v * (w[t % C] + w[C + t % C]);
}

struct ConfidenceHead {
  int C = 0, Cs = 0, F = 0, bins = 39, dense = 24, NB = 64, PB = 50;
  bool preSym = false, singleNorm = false, hasDist = false, hasLogitsLn = false, hasPaeLn = false, hasPlddtLn = false;
  bool hasResolved = false, hasResolvedLn = false;   // the experimentally-resolved head (2 logits a dense slot), where a bundle has it
  int binMode = BIN_AF3;
  bool unmasked = false;            // chai-1: the distance embedding over every pair
  bool reembed = false, globalNorm = false, split = false, absent = false;
  Linear leftTf, rightTf, pde, pae, plddt, pdeInter, paeInter, resolved;
  Param distW, distF;
  LayerNorm logitsLn, paeLn, plddtLn, inSingleLn, resolvedLn;
  // boltz2's re-embedding
  LayerNorm sInLn, sLn, zLn;
  Linear sInToS, relPos, prodIn1, prodIn2, prodOut;
  Param bondW, bondType, unspecified;
  // rf3's absent target_feat rows [2, C]
  Param absL, absR;
  // chai-1's grid attentions' transposed output twins: the projection applied is outputProjection + its twin
  struct Dual { int block, attn; Param twin; };
  std::vector<Dual> duals;
  std::vector<Pairformer> blocks;
  Dialect d;
};
// nb blocks with the pairformer dims (as test_af3.cu reads them from the bundle); h.reembed, h.globalNorm, h.binMode and
// h.unmasked set by the caller from the dialect
inline void forEachParam(ConfidenceHead& h, int nb, int Hp, int Dp, int Hs, int Ds, int pf, int sf, bool gridBiases, bool hasGrid,
                         bool parallel, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string P = "confidence.";
  auto ln = [&](const std::string& n, LayerNorm& l, int C) {
    l.C = C; f(n + "Scale", l.g, C); if (has(n + "Offset")) f(n + "Offset", l.b, C);
  };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; l.w.fanIn = K; f(n, l.w, (size_t)K * N); };
  if (h.reembed) {
    const std::string R = P + "reembed.";
    ln(R + "sInputsNorm", h.sInLn, h.F); ln(R + "sNorm", h.sLn, h.Cs); ln(R + "zNorm", h.zLn, h.C);
    lin(R + "sInputToS", h.sInToS, h.F, h.Cs);
    lin(R + "relPosProject", h.relPos, 139, h.C);
    lin(R + "leftTargetFeatProject", h.leftTf, h.F, h.C); lin(R + "rightTargetFeatProject", h.rightTf, h.F, h.C);
    lin(R + "sToZProdIn1", h.prodIn1, h.F, h.C); lin(R + "sToZProdIn2", h.prodIn2, h.F, h.C);
    lin(R + "sToZProdOut", h.prodOut, h.C, h.C);
    h.bins = 64; f(R + "distogramFeatProject", h.distW, (size_t)64 * h.C);
    f(R + "tokenBondsProject", h.bondW, h.C); f(R + "tokenBondsTypeEmbed", h.bondType, (size_t)7 * h.C);
    f(R + "contactEncodingUnspecified", h.unspecified, h.C);
  } else {
    lin(P + "leftTargetFeatProject", h.leftTf, h.F, h.C); lin(P + "rightTargetFeatProject", h.rightTf, h.F, h.C);
    f(P + "distogramFeatProject", h.distW, (size_t)h.bins * h.C);
    h.hasDist = has(P + "distanceFeatProject") && h.binMode != BIN_CA;
    if (h.hasDist) f(P + "distanceFeatProject", h.distF, h.C);
    h.singleNorm = has(P + "inputSingleNormScale"); if (h.singleNorm) ln(P + "inputSingleNorm", h.inSingleLn, h.Cs);
    h.absent = h.globalNorm && has(P + "leftTargetFeatProjectAbsent");
    if (h.absent) { f(P + "leftTargetFeatProjectAbsent", h.absL, (size_t)2 * h.C); f(P + "rightTargetFeatProjectAbsent", h.absR, (size_t)2 * h.C); }
  }
  h.blocks.assign(nb, Pairformer{});
  for (int k = 0; k < nb; ++k)
    forEachParam(h.blocks[k], P + "blocks." + std::to_string(k), h.C, h.Cs, Hp, Dp, Hs, Ds, pf, sf, gridBiases, true, f, hasGrid, parallel);
  h.duals.clear();
  for (int k = 0; k < nb; ++k) for (int a = 1; a <= 2; ++a) {
    const std::string G = P + "blocks." + std::to_string(k) + ".pairAttention" + std::to_string(a) + ".outputProjectionTransposed";
    if (has(G)) h.duals.push_back({k, a, Param{}});
  }
  for (auto& d : h.duals) {
    Linear& o = d.attn == 1 ? h.blocks[d.block].grid1.out : h.blocks[d.block].grid2.out;
    f(P + "blocks." + std::to_string(d.block) + ".pairAttention" + std::to_string(d.attn) + ".outputProjectionTransposed", d.twin,
      (size_t)o.K * o.N);
  }
  h.hasLogitsLn = has(P + "logitsLnScale"); if (h.hasLogitsLn) ln(P + "logitsLn", h.logitsLn, h.C);
  lin(P + "leftHalfDistanceLogits", h.pde, h.C, h.NB);
  h.hasPaeLn = has(P + "paeLogitsLnScale"); if (h.hasPaeLn) ln(P + "paeLogitsLn", h.paeLn, h.C);
  lin(P + "paeLogits", h.pae, h.C, h.NB);
  h.split = has(P + "interHalfDistanceLogits");
  if (h.split) { lin(P + "interHalfDistanceLogits", h.pdeInter, h.C, h.NB); lin(P + "paeInterLogits", h.paeInter, h.C, h.NB); }
  h.hasPlddtLn = has(P + "plddtLnScale"); if (h.hasPlddtLn) ln(P + "plddtLn", h.plddtLn, h.Cs);
  lin(P + "plddtLogits", h.plddt, h.Cs, h.dense * h.PB);
  h.hasResolved = has(P + "experimentallyResolvedLogits");
  if (h.hasResolved) {
    h.hasResolvedLn = has(P + "resolvedLnScale"); if (h.hasResolvedLn) ln(P + "resolvedLn", h.resolvedLn, h.Cs);
    lin(P + "experimentallyResolvedLogits", h.resolved, h.Cs, h.dense * 2);
  }
}
__global__ void sumIntoK(real* y, const real* a, const real* b, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = a[t] + b[t];
}
// for the blocks' duration, each dual projection's weight W + W^T-twin (and a gradient of its own, added into both
// on the way out)
struct DualScope {
  ConfidenceHead& h;
  std::vector<std::pair<real*, real*>> saved;
  std::vector<real*> grads;
  explicit DualScope(ConfidenceHead& h_) : h(h_) {
    for (auto& d : h.duals) {
      Param& w = (d.attn == 1 ? h.blocks[d.block].grid1.out : h.blocks[d.block].grid2.out).w;
      saved.push_back({w.w, w.g});
      real *sw, *sg;
      GK(cudaMallocAsync(&sw, w.n * sizeof(real), STREAM)); GK(cudaMallocAsync(&sg, w.n * sizeof(real), STREAM));
      sumIntoK<<<blocks(w.n), 256, 0, STREAM>>>(sw, w.w, d.twin.w, w.n);
      GK(cudaMemsetAsync(sg, 0, w.n * sizeof(real), STREAM));
      w.w = sw; w.g = sg; grads.push_back(sg);
    }
  }
  ~DualScope() {
    for (size_t k = 0; k < h.duals.size(); ++k) {
      auto& d = h.duals[k];
      Param& w = (d.attn == 1 ? h.blocks[d.block].grid1.out : h.blocks[d.block].grid2.out).w;
      add(saved[k].second, grads[k], w.n); add(d.twin.g, grads[k], w.n);
      GK(cudaFreeAsync(w.w, STREAM)); GK(cudaFreeAsync(w.g, STREAM));
      w.w = saved[k].first; w.g = saved[k].second;
    }
  }
};
// what the re-embedding and the split heads read (null where a head has neither)
struct ConfidenceBatch { const int *ri, *ti, *asym, *ent, *sym; const real *bonds, *orders; };
struct ConfidenceSave {
  real *l, *r, *sq, *pair0, *single0, *clamped, *sym, *xn, *pn, *sn;
  int* bin;
  std::vector<real*> pairs, singles;     // each block's input (and the last one's output)
  LnSave lin_, lx, lp, ls, lr;
  real* rn = nullptr;            // the resolved head's normalised single
  // the re-embedding's
  real *sIn, *p1, *p2, *zn;
  LnSave lsIn, lsS, lz;
  // rf3's normalised inputs
  real *tfn;
  GlobalNorm gz, gs, gt;
  int n;
};
struct ConfidenceLogits { real *pde, *pae, *plddt, *resolved = nullptr; };    // [pairs, 64] (symmetrised), [pairs, 64], [n, dense * 50], [n, dense * 2]
inline ConfidenceSave confidenceForward(ConfidenceHead& h, const real* trunkPair, const real* trunkSingle, const real* tf,
                                        const real* beta, const real* seqMask, const real* pairMask, const real* maskT, int n,
                                        ConfidenceLogits& o, const ConfidenceBatch& cb = {}) {
  ConfidenceSave s{}; s.n = n;
  const size_t nn = (size_t)n * n;
  s.bin = (int*)ARENA.get(nn); s.sq = ARENA.get(nn);
  confBinK<<<blocks(nn), 256, 0, STREAM>>>(beta, n, h.bins, (real)3.25, (real)50.75, s.bin, s.sq, h.reembed ? BIN_REEMBED : h.binMode);
  s.pairs.assign(h.blocks.size() + 1, nullptr); s.singles.assign(h.blocks.size() + 1, nullptr);
  if (h.reembed) {
    s.sIn = ARENA.get((size_t)n * h.F); s.lsIn = lnForward(h.sInLn, tf, s.sIn, n);
    s.singles[0] = ARENA.get((size_t)n * h.Cs); s.lsS = lnForward(h.sLn, trunkSingle, s.singles[0], n);
    h.sInToS.forward(s.sIn, s.singles[0], n, 1);
    s.pairs[0] = ARENA.get(nn * h.C); s.lz = lnForward(h.zLn, trunkPair, s.pairs[0], nn);
    {
      Scratch sc;
      real* oh = ARENA.get(nn * 139);
      relposK<<<blocks(nn * 139), 256, 0, STREAM>>>(cb.ri, cb.ti, cb.asym, cb.ent, cb.sym, oh, n);
      h.relPos.forward(oh, s.pairs[0], nn, 1);
    }
    s.l = ARENA.get((size_t)n * h.C); h.leftTf.forward(s.sIn, s.l, n);
    s.r = ARENA.get((size_t)n * h.C); h.rightTf.forward(s.sIn, s.r, n);
    s.p1 = ARENA.get((size_t)n * h.C); h.prodIn1.forward(s.sIn, s.p1, n);
    s.p2 = ARENA.get((size_t)n * h.C); h.prodIn2.forward(s.sIn, s.p2, n);
    confPairInitK<<<blocks(nn * h.C), 256, 0, STREAM>>>(s.pairs[0], s.l, s.r, s.bin, s.sq, pairMask, h.distW.w, nullptr, n, h.C);
    reembedTermsK<<<blocks(nn * h.C), 256, 0, STREAM>>>(s.pairs[0], cb.bonds, cb.orders, h.bondW.w, h.bondType.w, h.unspecified.w, nn, h.C);
    Scratch sc;
    real* prod = ARENA.get(nn * h.C);
    outerRowsK<<<blocks(nn * h.C), 256, 0, STREAM>>>(s.p1, s.p2, prod, n, h.C);
    h.prodOut.forward(prod, s.pairs[0], nn, 1);
  } else {
    const real* tfIn = tf;
    s.pairs[0] = ARENA.get(nn * h.C);
    if (h.globalNorm) {
      s.gz = gnForward(trunkPair, pairMask, nn, h.C, h.C, s.pairs[0]);
      s.tfn = ARENA.get((size_t)n * h.F); s.gt = gnForward(tf, seqMask, n, h.F, 449, s.tfn); tfIn = s.tfn;
    } else GK(cudaMemcpyAsync(s.pairs[0], trunkPair, nn * h.C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    s.l = ARENA.get((size_t)n * h.C); h.leftTf.forward(tfIn, s.l, n);
    s.r = ARENA.get((size_t)n * h.C); h.rightTf.forward(tfIn, s.r, n);
    if (h.absent) {
      const real v = (real)(-s.gt.mean * s.gt.istd);
      absentK<<<blocks((size_t)n * h.C), 256, 0, STREAM>>>(s.l, h.absL.w, v, n, h.C);
      absentK<<<blocks((size_t)n * h.C), 256, 0, STREAM>>>(s.r, h.absR.w, v, n, h.C);
    }
    confPairInitK<<<blocks(nn * h.C), 256, 0, STREAM>>>(s.pairs[0], s.l, s.r, s.bin, s.sq, h.unmasked ? nullptr : pairMask, h.distW.w,
                                                        h.hasDist ? h.distF.w : nullptr, n, h.C);
    if (h.globalNorm) { s.singles[0] = ARENA.get((size_t)n * h.Cs); s.gs = gnForward(trunkSingle, seqMask, n, h.Cs, h.Cs, s.singles[0]); }
    else if (h.singleNorm) {
      s.clamped = ARENA.get((size_t)n * h.Cs);
      clampFwdK<<<blocks((size_t)n * h.Cs), 256, 0, STREAM>>>(trunkSingle, s.clamped, (real)512, (size_t)n * h.Cs);
      s.singles[0] = ARENA.get((size_t)n * h.Cs); s.lin_ = lnForward(h.inSingleLn, s.clamped, s.singles[0], n);
    } else s.singles[0] = cloneR(trunkSingle, (size_t)n * h.Cs);
  }
  DualScope duals(h);
  for (size_t k = 0; k < h.blocks.size(); ++k) {
    s.pairs[k + 1] = ARENA.get(nn * h.C); s.singles[k + 1] = ARENA.get((size_t)n * h.Cs);
    Scratch sc;
    pairformerForward(h.blocks[k], h.d, s.pairs[k], s.singles[k], pairMask, maskT, seqMask, nullptr, s.pairs[k + 1],
                      s.singles[k + 1], n);
  }
  const real* z = s.pairs.back(); const real* sg = s.singles.back();
  // PDE: AF3 LN(z) W, symmetrised after; protenix2 and boltz2 LN(z + z^T) W (boltz2's split by chain)
  const real* src = z;
  if (h.preSym) { s.sym = ARENA.get(nn * h.C); pairSymK<<<blocks(nn * h.C), 256, 0, STREAM>>>(z, s.sym, n, h.C); src = s.sym; }
  s.xn = const_cast<real*>(src);
  if (h.hasLogitsLn) { s.xn = ARENA.get(nn * h.C); s.lx = lnForward(h.logitsLn, src, s.xn, nn); }
  o.pde = ARENA.get(nn * h.NB);
  real* inter = h.split ? ARENA.get(nn * h.NB) : nullptr;
  if (h.preSym) {
    h.pde.forward(s.xn, o.pde, nn);
    if (h.split) { h.pdeInter.forward(s.xn, inter, nn); interMergeK<<<blocks(nn * h.NB), 256, 0, STREAM>>>(o.pde, inter, cb.asym, n, h.NB); }
  } else { real* half = ARENA.get(nn * h.NB); h.pde.forward(s.xn, half, nn); pairSymK<<<blocks(nn * h.NB), 256, 0, STREAM>>>(half, o.pde, n, h.NB); }
  s.pn = const_cast<real*>(z);
  if (h.hasPaeLn) { s.pn = ARENA.get(nn * h.C); s.lp = lnForward(h.paeLn, z, s.pn, nn); }
  o.pae = ARENA.get(nn * h.NB); h.pae.forward(s.pn, o.pae, nn);
  if (h.split) { h.paeInter.forward(s.pn, inter, nn); interMergeK<<<blocks(nn * h.NB), 256, 0, STREAM>>>(o.pae, inter, cb.asym, n, h.NB); }
  s.sn = const_cast<real*>(sg);
  if (h.hasPlddtLn) { s.sn = ARENA.get((size_t)n * h.Cs); s.ls = lnForward(h.plddtLn, sg, s.sn, n); }
  o.plddt = ARENA.get((size_t)n * h.dense * h.PB); h.plddt.forward(s.sn, o.plddt, n);
  if (h.hasResolved) {
    s.rn = const_cast<real*>(sg);
    if (h.hasResolvedLn) { s.rn = ARENA.get((size_t)n * h.Cs); s.lr = lnForward(h.resolvedLn, sg, s.rn, n); }
    o.resolved = ARENA.get((size_t)n * h.dense * 2); h.resolved.forward(s.rn, o.resolved, n);
  }
  return s;
}
// the logits' cotangents (any null) -> every parameter; dtrunkPair, dtrunkSingle, dtf (added; any may be null)
inline void confidenceBackward(ConfidenceHead& h, const ConfidenceSave& s, const real* trunkPair, const real* trunkSingle,
                               const real* tf, const real* seqMask, const real* pairMask, const real* maskT,
                               const ConfidenceLogits& d, real* dtrunkPair, real* dtrunkSingle, real* dtf,
                               const ConfidenceBatch& cb = {}) {
  Scratch scratch;
  const int n = s.n; const size_t nn = (size_t)n * n;
  real* dz = ARENA.zeros(nn * h.C);
  real* dsg = ARENA.zeros((size_t)n * h.Cs);
  // a head's logits' cotangent into its projection (and its inter-chain half)
  auto head = [&](Linear& intra, Linear& inter, const real* x, const real* dl, real* dx) {
    if (!h.split) { intra.backward(x, dl, dx, nn); return; }
    Scratch sc;
    real *da = ARENA.get(nn * h.NB), *de = ARENA.get(nn * h.NB);
    interSplitK<<<blocks(nn * h.NB), 256, 0, STREAM>>>(dl, da, de, cb.asym, n, h.NB);
    intra.backward(x, da, dx, nn);
    inter.backward(x, de, dx, nn, true);
  };
  if (d.plddt) {
    real* dsn = ARENA.get((size_t)n * h.Cs); h.plddt.backward(s.sn, d.plddt, dsn, n);
    if (h.hasPlddtLn) lnBackward(h.plddtLn, s.ls, dsn, dsg, n, true); else add(dsg, dsn, (size_t)n * h.Cs);
  }
  if (d.resolved && h.hasResolved) {
    real* drn = ARENA.get((size_t)n * h.Cs); h.resolved.backward(s.rn, d.resolved, drn, n);
    if (h.hasResolvedLn) lnBackward(h.resolvedLn, s.lr, drn, dsg, n, true); else add(dsg, drn, (size_t)n * h.Cs);
  }
  if (d.pae) {
    real* dpn = ARENA.get(nn * h.C); head(h.pae, h.paeInter, s.pn, d.pae, dpn);
    if (h.hasPaeLn) lnBackward(h.paeLn, s.lp, dpn, dz, nn, true); else add(dz, dpn, nn * h.C);
  }
  if (d.pde) {
    real* dxn = ARENA.get(nn * h.C);
    if (h.preSym) head(h.pde, h.pdeInter, s.xn, d.pde, dxn);
    else { real* dh = ARENA.get(nn * h.NB); pairSymK<<<blocks(nn * h.NB), 256, 0, STREAM>>>(d.pde, dh, n, h.NB); h.pde.backward(s.xn, dh, dxn, nn); }
    real* dsrc = dxn;
    if (h.hasLogitsLn) { dsrc = ARENA.get(nn * h.C); lnBackward(h.logitsLn, s.lx, dxn, dsrc, nn, false); }
    if (h.preSym) { real* t = ARENA.get(nn * h.C); pairSymK<<<blocks(nn * h.C), 256, 0, STREAM>>>(dsrc, t, n, h.C); dsrc = t; }
    add(dz, dsrc, nn * h.C);
  }
  for (size_t k = h.blocks.size(); k-- > 0;) {
    DualScope duals(h);
    Scratch sc;
    real *po = ARENA.get(nn * h.C), *so = ARENA.get((size_t)n * h.Cs);
    auto bs = pairformerForward(h.blocks[k], h.d, s.pairs[k], s.singles[k], pairMask, maskT, seqMask, nullptr, po, so, n);
    pairformerBackward(h.blocks[k], h.d, bs, pairMask, maskT, seqMask, dz, dsg, dz, dsg, nullptr);
  }
  real *dl = ARENA.get((size_t)n * h.C), *dr = ARENA.get((size_t)n * h.C);
  confPairInitBwdLRK<<<blocks((size_t)n * h.C), 256, 0, STREAM>>>(dz, dl, dr, n, h.C);
  if (h.reembed) {
    confPairInitBwdWK<<<blocks(h.C, 64), 64, 0, STREAM>>>(dz, s.bin, s.sq, pairMask, h.distW.g, nullptr, n, h.C);
    reembedTermsBwdK<<<blocks(h.C, 64), 64, 0, STREAM>>>(dz, cb.bonds, cb.orders, h.bondW.g, h.bondType.g, h.unspecified.g, nn, h.C);
    {
      Scratch sc;
      real* prod = ARENA.get(nn * h.C);
      outerRowsK<<<blocks(nn * h.C), 256, 0, STREAM>>>(s.p1, s.p2, prod, n, h.C);
      real* dprod = ARENA.get(nn * h.C); h.prodOut.backward(prod, dz, dprod, nn);
      real *dp1 = ARENA.get((size_t)n * h.C), *dp2 = ARENA.get((size_t)n * h.C);
      outerRowsBwdK<<<blocks((size_t)n * h.C), 256, 0, STREAM>>>(dprod, s.p1, s.p2, dp1, dp2, n, h.C);
      real* oh = ARENA.get(nn * 139);
      relposK<<<blocks(nn * 139), 256, 0, STREAM>>>(cb.ri, cb.ti, cb.asym, cb.ent, cb.sym, oh, n);
      h.relPos.backward(oh, dz, nullptr, nn);
      real* dsIn = ARENA.get((size_t)n * h.F);
      h.leftTf.backward(s.sIn, dl, dsIn, n);
      h.rightTf.backward(s.sIn, dr, dsIn, n, true);
      h.prodIn1.backward(s.sIn, dp1, dsIn, n, true);
      h.prodIn2.backward(s.sIn, dp2, dsIn, n, true);
      h.sInToS.backward(s.sIn, dsg, dsIn, n, true);
      if (dtf) lnBackward(h.sInLn, s.lsIn, dsIn, dtf, n, true);
      else { real* t = ARENA.get((size_t)n * h.F); lnBackward(h.sInLn, s.lsIn, dsIn, t, n, false); }
    }
    if (dtrunkSingle) lnBackward(h.sLn, s.lsS, dsg, dtrunkSingle, n, true);
    else { Scratch sc; real* t = ARENA.get((size_t)n * h.Cs); lnBackward(h.sLn, s.lsS, dsg, t, n, false); }
    if (dtrunkPair) lnBackward(h.zLn, s.lz, dz, dtrunkPair, nn, true);
    else { Scratch sc; real* t = ARENA.get(nn * h.C); lnBackward(h.zLn, s.lz, dz, t, nn, false); }
    return;
  }
  confPairInitBwdWK<<<blocks(h.C, 64), 64, 0, STREAM>>>(dz, s.bin, s.sq, h.unmasked ? nullptr : pairMask, h.distW.g,
                                                        h.hasDist ? h.distF.g : nullptr, n, h.C);
  // the single's input
  if (h.globalNorm) { if (dtrunkSingle) gnBackward(s.gs, trunkSingle, seqMask, n, dsg, dtrunkSingle, true); }
  else if (h.singleNorm) {
    real* dc = ARENA.get((size_t)n * h.Cs); lnBackward(h.inSingleLn, s.lin_, dsg, dc, n, false);
    clampBwdK<<<blocks((size_t)n * h.Cs), 256, 0, STREAM>>>(trunkSingle, dc, (real)512, (size_t)n * h.Cs);
    if (dtrunkSingle) add(dtrunkSingle, dc, (size_t)n * h.Cs);
  } else if (dtrunkSingle) add(dtrunkSingle, dsg, (size_t)n * h.Cs);
  // the pair's: the trunk's, the projections, the distogram
  if (dtrunkPair) { if (h.globalNorm) gnBackward(s.gz, trunkPair, pairMask, nn, dz, dtrunkPair, true); else add(dtrunkPair, dz, nn * h.C); }
  const real* tfIn = h.globalNorm ? s.tfn : tf;
  real* dtfn = h.globalNorm ? ARENA.zeros((size_t)n * h.F) : dtf;
  if (dtfn) {
    h.leftTf.backward(tfIn, dl, dtfn, n, true);
    h.rightTf.backward(tfIn, dr, dtfn, n, true);
  } else { h.leftTf.backward(tfIn, dl, nullptr, n); h.rightTf.backward(tfIn, dr, nullptr, n); }
  if (!h.globalNorm) return;
  double extraMean = 0, extraIstd = 0;
  if (h.absent) {
    // v = -mean istd: its rows' gradients v sum_t d, and dv = sum d . (w0 + w1), into the statistics
    const real v = (real)(-s.gt.mean * s.gt.istd);
    std::vector<real> one(n, 1);
    real* ones = ARENA.get(n); GK(cudaMemcpy(ones, one.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    real* cl = ARENA.zeros(h.C); real* cr = ARENA.zeros(h.C);
    mmAT(ones, dl, cl, n, 1, h.C, 0); mmAT(ones, dr, cr, n, 1, h.C, 0);    // (the column sums)
    axpyR<<<blocks(h.C), 256, 0, STREAM>>>(h.absL.g, cl, v, h.C); axpyR<<<blocks(h.C), 256, 0, STREAM>>>(h.absL.g + h.C, cl, v, h.C);
    axpyR<<<blocks(h.C), 256, 0, STREAM>>>(h.absR.g, cr, v, h.C); axpyR<<<blocks(h.C), 256, 0, STREAM>>>(h.absR.g + h.C, cr, v, h.C);
    double dv = dot(cl, h.absL.w, h.C) + dot(cl, h.absL.w + h.C, h.C) + dot(cr, h.absR.w, h.C) + dot(cr, h.absR.w + h.C, h.C);
    extraMean = -dv * s.gt.istd; extraIstd = -dv * s.gt.mean;
  }
  if (dtf) gnBackward(s.gt, tf, seqMask, n, dtfn, dtf, true, extraMean, extraIstd);
}

}  // namespace af3
}  // namespace grad
