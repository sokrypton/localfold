// The confidence head: pLDDT, PAE, PDE. Transcribed from src/af3/confidence/confidence-reference.js.
// f32 throughout - the WebGPU head pins f32 too, for accuracy, and it is four blocks.
#pragma once
#include "trunk.cuh"
#include "atom.cuh"

// pair[i][j] += left[j] + right[i] + W_dgram[bin(|b_i - b_j|^2)] * mask
// (Wdist: protenix2's second, unbinned distance term - a bias-free projection of the raw distance)
__global__ void confidencePairInitK(float* pair, const float* left, const float* right, const float* beta,
                                    const float* pairMask, const float* Wd, int n, int C, int bins,
                                    float dmin, float dmax, const float* Wdist, bool caBins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  float sq = 0;
  for (int k = 0; k < 3; ++k) { float d = beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  float v = left[(size_t)j * C + c] + right[(size_t)i * C + c];
  if (caBins) {
    // rf3's: the bin is how many of `bins - 1` evenly spaced bounds the (real, +1e-10) distance is past
    double distance = sqrt((double)sq + 1e-10);
    int bin = 0;
    for (int at = 0; at < bins - 1; ++at) if (distance > dmin + at * ((double)(dmax - dmin) / (bins - 1))) ++bin;
    v += Wd[(size_t)bin * C + c] * pairMask[ij];
    pair[t] += v;
    return;
  }
  for (int b = 0; b < bins; ++b) {
    double lo = dmin + (double)(dmax - dmin) * b / (bins - 1), hi = dmin + (double)(dmax - dmin) * (b + 1) / (bins - 1);
    double lower = lo * lo, upper = b + 1 < bins ? hi * hi : 1e8;
    if (sq > lower && sq < upper) { v += Wd[(size_t)b * C + c] * pairMask[ij]; break; }
  }
  if (Wdist) v += sqrtf(sq + 1e-10f) * Wdist[c];
  pair[t] += v;
}
// expectation over bins of softmax(logits) . centres, optionally symmetrised (logits[ij] + logits[ji])
__global__ void expectationK(const float* logits, float* out, const float* mask, size_t rows, int bins,
                             const float* centres, int symmetricN, float scale) {
  size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  auto at = [&](int b) {
    if (symmetricN == 0) return logits[r * bins + b];
    size_t i = r / symmetricN, j = r % symmetricN;
    return logits[r * bins + b] + logits[(j * symmetricN + i) * bins + b];
  };
  float mx = -INFINITY;
  for (int b = 0; b < bins; ++b) mx = fmaxf(mx, at(b));
  float total = 0, weighted = 0;
  for (int b = 0; b < bins; ++b) { float p = expf(at(b) - mx); total += p; weighted += p * centres[b]; }
  out[r] = weighted / total * scale * (mask ? mask[r] : 1.f);
}

inline bool CONF_HALF = false;   // the head's four pairformer blocks in f16
// rf3's parameter-free LayerNorm over a WHOLE tensor, real rows only (mask per row), the mean
// and variance over `vendorWidth` columns (wider than C: the extra ones zero, each adding mean^2).
// Deterministic: per-block partials, then one block sums them in order.
__global__ void maskedSumK(const float* x, const float* mask, size_t rows, int C, const double* mean, double* partial) {
  __shared__ double red[256];
  double acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * C; i += (size_t)gridDim.x * blockDim.x) {
    if (!(mask[i / C] > 0)) continue;
    double v = x[i];
    if (mean) { v -= *mean; v *= v; }
    acc += v;
  }
  red[threadIdx.x] = acc; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) { if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w]; __syncthreads(); }
  if (threadIdx.x == 0) partial[blockIdx.x] = red[0];
}
__global__ void globalStatK(const double* partial, int parts, const float* mask, size_t rows, int C, int vendorWidth,
                            double* stat, int which) {
  if (threadIdx.x || blockIdx.x) return;
  double total = 0; for (int k = 0; k < parts; ++k) total += partial[k];
  double live = 0; for (size_t r = 0; r < rows; ++r) live += mask[r] > 0;
  double count = fmax(live * vendorWidth, 1.0);
  if (which == 0) stat[0] = total / count;                      // the mean
  else {                                                        // 1 / std
    double variance = total + (double)(vendorWidth - C) * live * stat[0] * stat[0];
    stat[1] = 1.0 / sqrt(variance / count + 1e-5);
  }
}
__global__ void applyGlobalNormK(float* x, size_t n, const double* stat) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = (float)((x[i] - stat[0]) * stat[1]);
}
inline void maskedGlobalNorm(float* x, const float* rowMask, size_t rows, int C, int vendorWidth) {
  const int parts = 512;
  double* partial = scratch<double>("gn.partial", parts); double* stat = scratch<double>("gn.stat", 2);
  maskedSumK<<<parts, 256, 0, STREAM>>>(x, rowMask, rows, C, nullptr, partial);
  globalStatK<<<1, 1, 0, STREAM>>>(partial, parts, rowMask, rows, C, vendorWidth, stat, 0);
  maskedSumK<<<parts, 256, 0, STREAM>>>(x, rowMask, rows, C, stat, partial);
  globalStatK<<<1, 1, 0, STREAM>>>(partial, parts, rowMask, rows, C, vendorWidth, stat, 1);
  applyGlobalNormK<<<blocks(rows * C), 256, 0, STREAM>>>(x, rows * C, stat);
}
__global__ void clampK(float* x, float limit, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = fminf(limit, fmaxf(-limit, x[i]));
}
struct ConfidenceOut { std::vector<float> plddt, pae, pde; double meanPlddt, ptm, iptm; };

inline ConfidenceOut confidenceHead(const float* trunkPair, const float* trunkSingle, const float* targetFeat,
                                    const float* pseudoBeta, const float* seqMask, const float* pairMask, int n) {
  const std::string P = "confidence";
  int C = (int)M.meta(P + ".pairChannels"), Cs = (int)M.meta(P + ".singleChannels"), F = (int)M.meta(P + ".targetFeatWidth");
  int dense = (int)M.meta("batch.dense");
  if (M.flag("trunk.dialect.reembedConfidencePair")) { fprintf(stderr, "reembedConfidencePair: not ported\n"); exit(1); }
  bool caDgram = M.flag("trunk.dialect.confidenceCaDgram");
  if (hasW(P + ".interHalfDistanceLogits") || !hasW(P + ".logitsLnScale")) {
    fprintf(stderr, "confidence head variant (split heads / no head LayerNorm): not ported\n"); exit(1);
  }
  size_t pairs = (size_t)n * n;
  float* pair = scratch<float>("conf.pair", pairs * C);
  CK(cudaMemcpyAsync(pair, trunkPair, pairs * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  float* single = scratch<float>("conf.single", (size_t)n * Cs);
  CK(cudaMemcpyAsync(single, trunkSingle, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  if (M.flag("trunk.dialect.confidenceGlobalNorm")) {
    // rf3 normalises every detached trunk input over the whole tensor first (target_feat over the
    // vendor's 449 columns, two wider than ours)
    float* tf = scratch<float>("conf.targetFeat", (size_t)n * F);
    CK(cudaMemcpyAsync(tf, targetFeat, (size_t)n * F * 4, cudaMemcpyDeviceToDevice, STREAM));
    maskedGlobalNorm(pair, pairMask, pairs, C, C);
    maskedGlobalNorm(single, seqMask, n, Cs, Cs);
    maskedGlobalNorm(tf, seqMask, n, F, 449);
    targetFeat = tf;
  }
  float* left = scratch<float>("conf.left", (size_t)n * C); float* right = scratch<float>("conf.right", (size_t)n * C);
  linear<float, float>(targetFeat, left, n, F, C, P + ".leftTargetFeatProject");
  linear<float, float>(targetFeat, right, n, F, C, P + ".rightTargetFeatProject");
  int bins = (int)(lenW(P + ".distogramFeatProject") / C);
  confidencePairInitK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, left, right, pseudoBeta, pairMask,
    W(P + ".distogramFeatProject"), n, C, bins, 3.25f, 50.75f, Wopt(P + ".distanceFeatProject"), caDgram);
  if (bins != (caDgram ? 40 : 39)) { fprintf(stderr, "confidence distogram has %d bins\n", bins); exit(1); }
  if (hasW(P + ".inputSingleNormScale")) {
    // the trunk single clamped to +-512 and LayerNormed before any use (protenix2)
    clampK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(single, 512.f, (size_t)n * Cs);
    float* sn = scratch<float>("conf.singleNorm", (size_t)n * Cs);
    layerNorm2<float, float>(single, sn, n, Cs, P + ".inputSingleNormScale", P + ".inputSingleNormOffset");
    CK(cudaMemcpyAsync(single, sn, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  int nb = 0; while (M.has(P + ".blocks." + std::to_string(nb) + ".singleChannels")) ++nb;
  bool swap = M.flag("trunk.dialect.swapTransposedBias"), divide = M.flag("trunk.dialect.triangleMulDivideByLength");
  for (int k = 0; k < nb; ++k)
    if (CONF_HALF) pairformerBlockAt<half>(pair, single, pairMask, seqMask, n, C, Cs, P + ".blocks." + std::to_string(k), swap, divide);
    else pairformerBlockAt<float>(pair, single, pairMask, seqMask, n, C, Cs, P + ".blocks." + std::to_string(k), swap, divide);
  // the error bins: 64 of them up to 31 A, the last one step past the second-to-last
  const int NB = 64; double step = 31.0 / (NB - 2);
  std::vector<float> centres(NB);
  for (int b = 0; b < NB - 1; ++b) centres[b] = (float)(b * step + step / 2);
  centres[NB - 1] = (float)(centres[NB - 2] + step);
  float* dCentres = upload(centres.data(), NB);
  float* ln = scratch<float>("conf.ln", pairs * C);
  float* logits = scratch<float>("conf.logits", pairs * NB);
  ConfidenceOut out;
  float* pde = scratch<float>("conf.pde", pairs); float* pae = scratch<float>("conf.pae", pairs);
  bool preSym = M.flag("trunk.dialect.preSymmetrisedPde");
  if (preSym) {
    // symmetrised BEFORE the projection: LN(z + z^T) W (protenix2); AF3 adds the transpose after
    float* sym = scratch<float>("conf.sym", pairs * C);
    symmetriseK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, sym, n, C);
    layerNorm2<float, float>(sym, ln, pairs, C, P + ".logitsLnScale", P + ".logitsLnOffset");
  } else {
    layerNorm2<float, float>(pair, ln, pairs, C, P + ".logitsLnScale", P + ".logitsLnOffset");
  }
  linear<float, float>(ln, logits, pairs, C, NB, P + ".leftHalfDistanceLogits");
  expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, pde, pairMask, pairs, NB, dCentres, preSym ? 0 : n, 1.f);
  layerNorm2<float, float>(pair, ln, pairs, C, P + ".paeLogitsLnScale", P + ".paeLogitsLnOffset");
  linear<float, float>(ln, logits, pairs, C, NB, P + ".paeLogits");
  expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, pae, pairMask, pairs, NB, dCentres, 0, 1.f);
  // pTM and ipTM off the PAE logits: per pair the expected TM term, then the best anchor's
  // mean over the pairs it selects (ipTM: other chains only). src/heads/tm-score.js.
  {
    std::vector<float> seq = download(seqMask, n);
    const int* asym = M.i("batch.asymId");
    int real = 0; for (float v : seq) real += v > 0;
    double d0 = 1.24 * std::cbrt(std::max(real, 19) - 15.0) - 1.8;
    std::vector<float> perBin(NB);
    for (int b = 0; b < NB; ++b) perBin[b] = (float)(1 / (1 + (double)centres[b] * centres[b] / (d0 * d0)));
    float* dPerBin = upload(perBin.data(), NB);
    float* dTerm = scratch<float>("conf.tmTerm", pairs);
    expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, dTerm, nullptr, pairs, NB, dPerBin, 0, 1.f);
    std::vector<float> term = download(dTerm, pairs);
    CK(cudaFree(dPerBin));
    auto reduce = [&](bool interOnly) {
      double best = -1e30; bool any = false;
      for (int i = 0; i < n; ++i) {
        double tot = 0; int cnt = 0;
        for (int j = 0; j < n; ++j) {
          if (!(seq[i] > 0 && seq[j] > 0) || (interOnly && asym[i] == asym[j])) continue;
          tot += term[(size_t)i * n + j]; ++cnt;
        }
        if (cnt) { any = true; best = std::max(best, tot / cnt); }
      }
      return any ? best : NAN;
    };
    out.ptm = reduce(false); out.iptm = reduce(true);
  }
  const int PB = 50;
  std::vector<float> pc(PB);
  for (int b = 0; b < PB; ++b) pc[b] = 0.5f / PB + (float)b / PB;
  float* dpc = upload(pc.data(), PB);
  float* sln = scratch<float>("conf.sln", (size_t)n * Cs);
  layerNorm2<float, float>(single, sln, n, Cs, P + ".plddtLnScale", P + ".plddtLnOffset");
  float* pl = scratch<float>("conf.plddtLogits", (size_t)n * dense * PB);
  linear<float, float>(sln, pl, n, Cs, dense * PB, P + ".plddtLogits");
  float* plddt = scratch<float>("conf.plddt", (size_t)n * dense);
  expectationK<<<blocks((size_t)n * dense), 256, 0, STREAM>>>(pl, plddt, nullptr, (size_t)n * dense, PB, dpc, 0, 100.f);
  out.plddt = download(plddt, (size_t)n * dense);
  out.pae = download(pae, pairs);
  out.pde = download(pde, pairs);
  CK(cudaFree(dCentres)); CK(cudaFree(dpc));
  // the mean over real atoms
  const float* mask = M.f("batch.refMask");
  double sum = 0, count = 0;
  for (size_t i = 0; i < out.plddt.size(); ++i) if (mask[i]) { sum += out.plddt[i]; count += 1; }
  out.meanPlddt = sum / std::max(count, 1.0);
  return out;
}
