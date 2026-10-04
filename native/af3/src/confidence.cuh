// The confidence head: pLDDT, PAE, PDE. Transcribed from src/af3/confidence/confidence-reference.js.
// f32 throughout - the WebGPU head pins f32 too, for accuracy, and it is four blocks.
#pragma once
#include "trunk.cuh"
#include "atom.cuh"

// pair[i][j] += left[j] + right[i] + W_dgram[bin(|b_i - b_j|^2)] * mask
// (Wdist: protenix2's second, unbinned distance term - a bias-free projection of the raw distance)
// the distogram bin of each pair (-1: none) and its squared distance, once a pair rather than once a
// channel (a 39-step double-precision search per element was 24 ms of a 1044-token fold)
__global__ void confidenceBinK(const float* beta, int n, int bins, float dmin, float dmax, bool caBins, int* binOut,
                               float* sqOut) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = (int)(ij / n), j = (int)(ij % n);
  float sq = 0;
  for (int k = 0; k < 3; ++k) { float d = beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  int bin = -1;
  if (caBins) {
    // rf3's: the bin is how many of `bins - 1` evenly spaced bounds the (real, +1e-10) distance is past
    double distance = sqrt((double)sq + 1e-10);
    bin = 0;
    for (int at = 0; at < bins - 1; ++at) if (distance > dmin + at * ((double)(dmax - dmin) / (bins - 1))) ++bin;
  } else {
    for (int b = 0; b < bins; ++b) {
      double lo = dmin + (double)(dmax - dmin) * b / (bins - 1), hi = dmin + (double)(dmax - dmin) * (b + 1) / (bins - 1);
      double lower = lo * lo, upper = b + 1 < bins ? hi * hi : 1e8;
      if (sq > lower && sq < upper) { bin = b; break; }
    }
  }
  binOut[ij] = bin; sqOut[ij] = sq;
}
__global__ void confidencePairInitK(float* pair, const float* left, const float* right, const int* binOf,
                                    const float* sqOf, const float* pairMask, const float* Wd, int n, int C,
                                    const float* Wdist, bool caBins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  float v = left[(size_t)j * C + c] + right[(size_t)i * C + c];
  int bin = binOf[ij];
  if (bin >= 0) v += Wd[(size_t)bin * C + c] * pairMask[ij];
  if (!caBins && Wdist) v += sqrtf(sqOf[ij] + 1e-10f) * Wdist[c];
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
// boltz2's re-embedded pair (after LN_z and the relative encoding): right[i] + left[j] off the
// normalised s_inputs, its own 64-bin distance embedding (bounds evenly over 2..22 A), the bond
// contact and bond-order terms and the contact conditioning's unspecified constant
__global__ void reembedPairK(float* pair, const float* left, const float* right, const float* beta, const float* pairMask,
                             const float* Wd, const float* bonds, const float* orders, const float* wBond,
                             const float* wBondType, const float* unspecified, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  double sq = 1e-10;
  for (int k = 0; k < 3; ++k) { double d = (double)beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  double distance = sqrt(sq);
  int bin = 0;
  for (int e = 0; e < 63; ++e) if (distance > 2.0 + 20.0 * e / 62) ++bin;
  int o = orders ? (int)orders[ij] : 0;
  if (o < 0 || o >= 7) o = 0;
  float v = right[(size_t)i * C + c] + left[(size_t)j * C + c] + Wd[(size_t)bin * C + c] * pairMask[ij]
          + (bonds ? bonds[ij] * wBond[c] : 0.f) + wBondType[(size_t)o * C + c] + unspecified[c];
  pair[t] += v;
}
// prod[i][j][e] = a[i][e] b[j][e], rows i0.. of a chunk
__global__ void outerProductRowsK(const float* a, const float* b, float* out, int i0, int rowsI, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rowsI * n * C) return;
  int e = (int)(t % C); size_t ij = t / C; int i = i0 + (int)(ij / n), j = (int)(ij % n);
  out[t] = a[(size_t)i * C + e] * b[(size_t)j * C + e];
}
// boltz2's split pair heads: the inter-chain logits where the two tokens' chains differ
// rows [r0, r0 + cnt) of z + z^T, row-major
__global__ void symmetriseRowsK(const float* z, float* out, size_t r0, size_t cnt, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = r0 + t / C; size_t i = ij / T, j = ij % T;
  out[t] = z[ij * C + c] + z[(j * T + i) * C + c];
}
// the same over pairs [r0, r0 + cnt) of logits, from a chunk of inter-chain logits
__global__ void interChainLogitsRowsK(float* logits, const float* inter, const int* asym, int n, int bins, size_t r0,
                                      size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * bins) return;
  size_t ij = r0 + t / bins; int i = (int)(ij / n), j = (int)(ij % n);
  if (asym[i] != asym[j]) logits[r0 * bins + t] = inter[t];
}
__global__ void interChainLogitsK(float* logits, const float* inter, const int* asym, int n, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * bins) return;
  size_t ij = t / bins; int i = (int)(ij / n), j = (int)(ij % n);
  if (asym[i] != asym[j]) logits[t] = inter[t];
}
inline void boltz2Reembed(float* pair, float* single, const float* trunkPair, const float* trunkSingle,
                          const float* targetFeat, const float* pseudoBeta, const float* pairMask, int n, int C, int Cs, int F) {
  const std::string R = "confidence.reembed.";
  size_t pairs = (size_t)n * n;
  float* sIn = scratch<float>("conf.sInputs", (size_t)n * F);
  layerNorm2<float, float>(targetFeat, sIn, n, F, R + "sInputsNormScale", R + "sInputsNormOffset");
  layerNorm2<float, float>(trunkSingle, single, n, Cs, R + "sNormScale", R + "sNormOffset");
  linear<float, float>(sIn, single, n, F, Cs, R + "sInputToS", false, 1.f);
  layerNorm2<float, float>(trunkPair, pair, pairs, C, R + "zNormScale", R + "zNormOffset");
  if (lenW(R + "relPosProject") != (size_t)139 * C) { fprintf(stderr, "relPosProject is not 139 x %d\n", C); exit(1); }
  relativeEncodingK<<<blocks(pairs * C), 256, 0, STREAM>>>(
    Idev("batch.features.residueIndex"), Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"),
    Idev("batch.features.entityId"), Idev("batch.features.symId"), W(R + "relPosProject"), pair, n, C, 32, 2);
  float* left = scratch<float>("conf.left", (size_t)n * C); float* right = scratch<float>("conf.right", (size_t)n * C);
  float* p1 = scratch<float>("conf.p1", (size_t)n * C); float* p2 = scratch<float>("conf.p2", (size_t)n * C);
  linear<float, float>(sIn, left, n, F, C, R + "leftTargetFeatProject");
  linear<float, float>(sIn, right, n, F, C, R + "rightTargetFeatProject");
  linear<float, float>(sIn, p1, n, F, C, R + "sToZProdIn1");
  linear<float, float>(sIn, p2, n, F, C, R + "sToZProdIn2");
  if (lenW(R + "distogramFeatProject") != (size_t)64 * C) { fprintf(stderr, "the reembed distogram is not 64 bins\n"); exit(1); }
  reembedPairK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, left, right, pseudoBeta, pairMask, W(R + "distogramFeatProject"),
    M.has("batch.bondMatrix") ? Fdev("batch.bondMatrix") : nullptr,
    M.has("batch.bondOrderMatrix") ? Fdev("batch.bondOrderMatrix") : nullptr,
    W(R + "tokenBondsProject"), W(R + "tokenBondsTypeEmbed"), W(R + "contactEncodingUnspecified"), n, C);
  // the outer product of the two s_inputs projections through sToZProdOut, in row chunks
  int rowsPer = std::max<int>(1, (int)(CHUNK / ((size_t)n * C)));
  float* prod = scratch<float>("conf.prod", (size_t)std::min(rowsPer, n) * n * C);
  for (int i0 = 0; i0 < n; i0 += rowsPer) {
    int r = std::min(rowsPer, n - i0);
    outerProductRowsK<<<blocks((size_t)r * n * C), 256, 0, STREAM>>>(p1, p2, prod, i0, r, n, C);
    linear<float, float>(prod, pair + (size_t)i0 * n * C, (size_t)r * n, C, C, R + "sToZProdOut", false, 1.f);
  }
}
struct ConfidenceOut { std::vector<float> plddt, pae, pde, tmTerm; double meanPlddt, ptm, iptm; };

// consumeTrunkPair: the trunk's pair is read by nothing after this call, so the head works in it rather
// than in a copy (the caller says so for a card short of room, on its last confidence call)
inline ConfidenceOut confidenceHead(const float* trunkPair, const float* trunkSingle, const float* targetFeat,
                                    const float* pseudoBeta, const float* seqMask, const float* pairMask, int n,
                                    bool consumeTrunkPair = false) {
  const std::string P = "confidence";
  int C = (int)M.meta(P + ".pairChannels"), Cs = (int)M.meta(P + ".singleChannels"), F = (int)M.meta(P + ".targetFeatWidth");
  int dense = (int)M.meta("batch.dense");
  bool caDgram = M.flag("trunk.dialect.confidenceCaDgram");
  size_t pairs = (size_t)n * n;
  bool reembed = M.flag("trunk.dialect.reembedConfidencePair");
  bool inPlace = consumeTrunkPair && !reembed;                 // (boltz2 builds its pair from the trunk's: no aliasing)
  float* pair = inPlace ? const_cast<float*>(trunkPair) : scratch<float>("conf.pair", pairs * C);
  float* single = scratch<float>("conf.single", (size_t)n * Cs);
  if (reembed) {
    boltz2Reembed(pair, single, trunkPair, trunkSingle, targetFeat, pseudoBeta, pairMask, n, C, Cs, F);
  } else {
  if (!inPlace) CK(cudaMemcpyAsync(pair, trunkPair, pairs * C * 4, cudaMemcpyDeviceToDevice, STREAM));
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
  int* binOf = scratch<int>("conf.bin", pairs); float* sqOf = scratch<float>("conf.sq", pairs);
  confidenceBinK<<<blocks(pairs), 256, 0, STREAM>>>(pseudoBeta, n, bins, 3.25f, 50.75f, caDgram, binOf, sqOf);
  confidencePairInitK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, left, right, binOf, sqOf, pairMask,
    W(P + ".distogramFeatProject"), n, C, Wopt(P + ".distanceFeatProject"), caDgram);
  if (bins != (caDgram ? 40 : 39)) { fprintf(stderr, "confidence distogram has %d bins\n", bins); exit(1); }
  if (hasW(P + ".inputSingleNormScale")) {
    // the trunk single clamped to +-512 and LayerNormed before any use (protenix2)
    clampK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(single, 512.f, (size_t)n * Cs);
    float* sn = scratch<float>("conf.singleNorm", (size_t)n * Cs);
    layerNorm2<float, float>(single, sn, n, Cs, P + ".inputSingleNormScale", P + ".inputSingleNormOffset");
    CK(cudaMemcpyAsync(single, sn, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  }
  int nb = 0; while (M.has(P + ".blocks." + std::to_string(nb) + ".singleChannels")) ++nb;
  bool swap = M.flag("trunk.dialect.swapTransposedBias"), divide = M.flag("trunk.dialect.triangleMulDivideByLength");
  for (int k = 0; k < nb; ++k)
    if (CONF_HALF) pairformerBlockAt<half>(pair, single, pairMask, seqMask, n, C, Cs, P + ".blocks." + std::to_string(k), swap, divide);
    else pairformerBlockAt<float>(pair, single, pairMask, seqMask, n, C, Cs, P + ".blocks." + std::to_string(k), swap, divide);
  // on a card short of room the stack's scratch goes before the heads allocate theirs, so the two peak
  // apart rather than together
  if (shortPair(pairs, C)) releaseScratch({ "tri.", "trib.", "grid.", "tr.", "st." });
  // the error bins: 64 of them up to 31 A, the last one step past the second-to-last
  const int NB = 64; double step = 31.0 / (NB - 2);
  std::vector<float> centres(NB);
  for (int b = 0; b < NB - 1; ++b) centres[b] = (float)(b * step + step / 2);
  centres[NB - 1] = (float)(centres[NB - 2] + step);
  float* dCentres = upload(centres.data(), NB);
  // on a card short of room the heads' LayerNorm (and the symmetrised pair it reads) run in row chunks
  bool chunked = shortPair(pairs, C);
  size_t rowsPer = chunked ? std::max<size_t>(1, std::min(pairs, CHUNK / C)) : pairs;
  float* ln = scratch<float>("conf.ln", rowsPer * C);
  float* logits = scratch<float>("conf.logits", pairs * NB);
  ConfidenceOut out;
  float* pde = scratch<float>("conf.pde", pairs); float* pae = scratch<float>("conf.pae", pairs);
  bool preSym = M.flag("trunk.dialect.preSymmetrisedPde");
  // a head LayerNorm the bundle does not carry is no LayerNorm (boltz2 reads z and s directly)
  auto headNorm = [&](const float* x, float* buf, size_t rows, int Cx, const std::string& name) -> const float* {
    if (!hasW(P + "." + name + "Scale")) return x;
    layerNorm2<float, float>(x, buf, rows, Cx, P + "." + name + "Scale", P + "." + name + "Offset");
    return buf;
  };
  // boltz2 splits each pair head into an intra-chain and an inter-chain projection
  const int* asymDev = Idev("batch.asymId");
  float* interLogits = hasW(P + ".interHalfDistanceLogits") || hasW(P + ".paeInterLogits")
    ? scratch<float>("conf.interLogits", (chunked ? rowsPer : pairs) * NB) : nullptr;
  auto project = [&](const float* x, const std::string& w, const std::string& inter) {
    linear<float, float>(x, logits, pairs, C, NB, P + "." + w);
    if (!hasW(P + "." + inter)) return;
    linear<float, float>(x, interLogits, pairs, C, NB, P + "." + inter);
    interChainLogitsK<<<blocks(pairs * NB), 256, 0, STREAM>>>(logits, interLogits, asymDev, n, NB);
  };
  if (hasW(P + ".interHalfDistanceLogits") && !preSym) { fprintf(stderr, "split PDE heads need the pre-symmetrised PDE\n"); exit(1); }
  // a head's logits, LN(x) W (with boltz2's inter-chain half), a chunk of rows at a time when chunked
  float* sym = preSym && chunked ? scratch<float>("conf.sym", rowsPer * C) : nullptr;
  auto headLogits = [&](bool symmetrised, const std::string& norm, const std::string& w, const std::string& inter) {
    for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
      size_t cnt = std::min(rowsPer, pairs - r0);
      const float* x = pair + r0 * C;
      if (symmetrised) { symmetriseRowsK<<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym, r0, cnt, n, C); x = sym; }
      const float* xn = headNorm(x, ln, cnt, C, norm);
      linear<float, float>(xn, logits + r0 * NB, cnt, C, NB, P + "." + w);
      if (!hasW(P + "." + inter)) continue;
      linear<float, float>(xn, interLogits, cnt, C, NB, P + "." + inter);
      interChainLogitsRowsK<<<blocks(cnt * NB), 256, 0, STREAM>>>(logits, interLogits, asymDev, n, NB, r0, cnt);
    }
  };
  if (chunked) {
    headLogits(preSym, "logitsLn", "leftHalfDistanceLogits", "interHalfDistanceLogits");
  } else {
    const float* src = pair;
    if (preSym) {
      // symmetrised BEFORE the projection: LN(z + z^T) W (protenix2, boltz2); AF3 adds the transpose after
      float* whole = scratch<float>("conf.sym", pairs * C);
      symmetriseK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, whole, n, C);
      src = whole;
    }
    project(headNorm(src, ln, pairs, C, "logitsLn"), "leftHalfDistanceLogits", "interHalfDistanceLogits");
  }
  expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, pde, pairMask, pairs, NB, dCentres, preSym ? 0 : n, 1.f);
  if (chunked) headLogits(false, "paeLogitsLn", "paeLogits", "paeInterLogits");
  else project(headNorm(pair, ln, pairs, C, "paeLogitsLn"), "paeLogits", "paeInterLogits");
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
    out.tmTerm = std::move(term);
  }
  const int PB = 50;
  std::vector<float> pc(PB);
  for (int b = 0; b < PB; ++b) pc[b] = 0.5f / PB + (float)b / PB;
  float* dpc = upload(pc.data(), PB);
  float* slnBuf = scratch<float>("conf.sln", (size_t)n * Cs);
  const float* sln = headNorm(single, slnBuf, n, Cs, "plddtLn");
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
