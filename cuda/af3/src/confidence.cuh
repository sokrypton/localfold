// The confidence head: pLDDT, PAE, PDE. Transcribed from cpu/af3/confidence/confidence.js.
// f32 throughout - the WebGPU head pins f32 too, for accuracy, and it is four blocks.
#pragma once
#include "trunk.cuh"
#include "atom.cuh"

// pair[i][j] += left[j] + right[i] + W_dgram[bin(|b_i - b_j|^2)] * mask
// (Wdist: protenix2's second, unbinned distance term - a bias-free projection of the raw distance)
// the distogram bin of each pair (-1: none) and its squared distance, once a pair rather than once a
// channel (a 39-step double-precision search per element was 24 ms of a 1044-token fold)
__global__ void confidenceBinK(const float* beta, int n, int bins, float dmin, float dmax, bool caBins, int* binOut,
                               float* sqOut, bool chaiBins = false) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = (int)(ij / n), j = (int)(ij % n);
  float sq = 0;
  for (int k = 0; k < 3; ++k) { float d = beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  int bin = -1;
  if (chaiBins) {
    // chai-1's: 16 bins, how many of 15 evenly spaced bounds from 3.375 to 21.375 the distance (+1e-10) is past
    float distance = sqrtf(sq + 1e-10f);
    bin = 0;
    for (int at = 0; at < bins - 1; ++at) bin += distance > 3.375f + at * (18.f / (bins - 2));
  } else if (caBins) {
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
template <class PT = float>
__global__ void confidencePairInitK(float* pair, const float* left, const float* right, const int* binOf,
                                    const float* sqOf, const float* pairMask, const float* Wd, int n, int C,
                                    const float* Wdist, bool caBins, bool unmasked = false) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  float v = left[(size_t)j * C + c] + right[(size_t)i * C + c];
  int bin = binOf[ij];
  if (bin >= 0) v += Wd[(size_t)bin * C + c] * (unmasked ? 1.f : pairMask[ij]);
  if (!caBins && Wdist) v += sqrtf(sqOf[ij] + 1e-10f) * Wdist[c];
  pairSt<PT>(pair, t, pairLd<PT>(pair, t) + v);
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
// (the pair's element type PT: a bf16 confidence pair, TRUNK_PAIR16; the live rows counted in the same pass - one
// thread counting them over every pair was a serial loop of tokens^2)
template <class PT = float>
__global__ void maskedSumK(const float* x, const float* mask, size_t rows, int C, const double* mean, double* partial,
                           double* livePartial) {
  __shared__ double red[256], liveRed[256];
  double acc = 0, live = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * C; i += (size_t)gridDim.x * blockDim.x) {
    if (!(mask[i / C] > 0)) continue;
    if (i % C == 0) live += 1;
    double v = pairLd<PT>(x, i);
    if (mean) { v -= *mean; v *= v; }
    acc += v;
  }
  red[threadIdx.x] = acc; liveRed[threadIdx.x] = live; __syncthreads();
  for (int w = blockDim.x / 2; w; w >>= 1) {
    if (threadIdx.x < w) { red[threadIdx.x] += red[threadIdx.x + w]; liveRed[threadIdx.x] += liveRed[threadIdx.x + w]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { partial[blockIdx.x] = red[0]; livePartial[blockIdx.x] = liveRed[0]; }
}
__global__ void globalStatK(const double* partial, const double* livePartial, int parts, int C, int vendorWidth,
                            double* stat, int which) {
  if (threadIdx.x || blockIdx.x) return;
  double total = 0, live = 0; for (int k = 0; k < parts; ++k) { total += partial[k]; live += livePartial[k]; }
  double count = fmax(live * vendorWidth, 1.0);
  if (which == 0) stat[0] = total / count;                      // the mean
  else {                                                        // 1 / std
    double variance = total + (double)(vendorWidth - C) * live * stat[0] * stat[0];
    stat[1] = 1.0 / sqrt(variance / count + 1e-5);
  }
}
template <class PT = float>
__global__ void applyGlobalNormK(float* x, size_t n, const double* stat) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pairSt<PT>(x, i, (float)((pairLd<PT>(x, i) - stat[0]) * stat[1]));
}
template <class PT = float>
inline void maskedGlobalNorm(float* x, const float* rowMask, size_t rows, int C, int vendorWidth) {
  const int parts = 512;
  double* partial = scratch<double>("gn.partial", parts); double* stat = scratch<double>("gn.stat", 2);
  double* live = scratch<double>("gn.live", parts);
  maskedSumK<PT><<<parts, 256, 0, STREAM>>>(x, rowMask, rows, C, nullptr, partial, live);
  globalStatK<<<1, 1, 0, STREAM>>>(partial, live, parts, C, vendorWidth, stat, 0);
  maskedSumK<PT><<<parts, 256, 0, STREAM>>>(x, rowMask, rows, C, stat, partial, live);
  globalStatK<<<1, 1, 0, STREAM>>>(partial, live, parts, C, vendorWidth, stat, 1);
  applyGlobalNormK<PT><<<blocks(rows * C), 256, 0, STREAM>>>(x, rows * C, stat);
}
__global__ void clampK(float* x, float limit, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = fminf(limit, fmaxf(-limit, x[i]));
}
// boltz2's re-embedded pair (after LN_z and the relative encoding): right[i] + left[j] off the
// normalised s_inputs, its own 64-bin distance embedding (bounds evenly over 2..22 A), the bond
// contact and bond-order terms and the contact conditioning's unspecified constant
// each pair's distance bin, once (in double, as it always was): taken per ELEMENT - C times a pair, a double sqrt
// and a 63-step double comparison loop each - it was 252 ms of one launch on a T4 at 510 tokens, whose FP64 runs at
// 1/32 of its f32
__global__ void reembedBinK(const float* beta, int* binOut, int n) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)n * n) return;
  int i = (int)(ij / n), j = (int)(ij % n);
  double sq = 1e-10;
  for (int k = 0; k < 3; ++k) { double d = (double)beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
  double distance = sqrt(sq);
  int bin = 0;
  for (int e = 0; e < 63; ++e) if (distance > 2.0 + 20.0 * e / 62) ++bin;
  binOut[ij] = bin;
}
__global__ void reembedPairK(float* pair, const float* left, const float* right, const int* bins, const float* pairMask,
                             const float* Wd, const float* bonds, const float* orders, const float* wBond,
                             const float* wBondType, const float* unspecified, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  int bin = bins[ij];
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
template <class PT = float>
__global__ void symmetriseRowsK(const float* z, float* out, size_t r0, size_t cnt, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = r0 + t / C; size_t i = ij / T, j = ij % T;
  out[t] = pairLd<PT>(z, ij * C + c) + pairLd<PT>(z, (j * T + i) * C + c);
}
// the transposed pairs of [r0, r0 + cnt): out[(ij - r0) C + c] = z[ji C + c]
template <class PT = float>
__global__ void transposedRowsK(const float* z, float* out, size_t r0, size_t cnt, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = r0 + t / C; size_t i = ij / T, j = ij % T;
  out[t] = pairLd<PT>(z, (j * T + i) * C + c);
}
// a chunk of the pair's own rows [r0, r0 + cnt), widened (a bf16 pair's, TRUNK_PAIR16)
template <class PT = float>
__global__ void ownRowsK(const float* z, float* out, size_t r0, size_t cnt, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < cnt * C) out[t] = pairLd<PT>(z, r0 * C + t);
}
// a chunk's own logits from its inter-chain ones (interChainLogitsRowsK, into a chunk-sized buffer)
__global__ void interChainLogitsChunkK(float* chunk, const float* inter, const int* asym, int n, int bins, size_t r0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * bins) return;
  size_t ij = r0 + t / bins; int i = (int)(ij / n), j = (int)(ij % n);
  if (asym[i] != asym[j]) chunk[t] = inter[t];
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
  int* bins = scratch<int>("conf.bin", pairs);
  reembedBinK<<<blocks(pairs), 256, 0, STREAM>>>(pseudoBeta, bins, n);
  reembedPairK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, left, right, bins, pairMask, W(R + "distogramFeatProject"),
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
// chai-1's pLDDT: each (token, dense slot)'s ATOM37 index, by its name (AF3's 4 characters, ASCII - 32)
inline const int* atom37Index(int n, int dense) {
  static const char* ATOM37[37] = { "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
    "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH",
    "CZ", "CZ2", "CZ3", "NZ", "OXT" };
  const int* chars = M.i("batch.refAtomNameChars");
  std::vector<int> idx((size_t)n * dense, 0);
  for (size_t a = 0; a < idx.size(); ++a)
    for (int k = 0; k < 37; ++k) {
      bool same = true;
      for (int c = 0; c < 4; ++c) {
        int want = c < (int)strlen(ATOM37[k]) ? ATOM37[k][c] - 32 : 0;
        if (chars[a * 4 + c] != want) { same = false; break; }
      }
      if (same) { idx[a] = k; break; }
    }
  int* d = scratch<int>("conf.atom37", idx.size());
  CK(cudaMemcpy(d, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
  return d;
}
__global__ void gatherPlddt37K(const float* p37, const int* idx, float* out, int n, int dense, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * dense * bins) return;
  int b = (int)(t % bins); size_t slot = t / bins; size_t token = slot / dense;
  out[t] = p37[(token * 37 + idx[slot]) * bins + b];
}
// chai-1's confidence triangle attention: each direction's output projection plus its transposed twin, summed in
// place once (their two applications cancel to one at inference - af3-any-model modules.py, dual_output)
__global__ void addInPlaceK(float* a, const float* b, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] += b[i];
}
inline void foldDualOutputs() {
  static bool done = false;
  if (done) return;
  done = true;
  for (int k = 0; M.has("confidence.blocks." + std::to_string(k) + ".singleChannels"); ++k)
    for (int a = 1; a <= 2; ++a) {
      std::string G = "confidence.blocks." + std::to_string(k) + ".pairAttention" + std::to_string(a);
      if (!hasW(G + ".outputProjectionTransposed")) continue;
      addInPlaceK<<<blocks(lenW(G + ".outputProjection")), 256, 0, STREAM>>>(const_cast<float*>(W(G + ".outputProjection")),
        W(G + ".outputProjectionTransposed"), lenW(G + ".outputProjection"));
    }
  CK(cudaStreamSynchronize(STREAM));
}
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
  // a bf16 trunk pair (TRUNK_PAIR16, a card short of room): the head's pair is bf16 too, its blocks run under PAIR16
  // and its heads widen a chunk of rows at a time (af3.cu keeps it only for a head that takes it: no re-embedding)
  const bool p16 = TRUNK_PAIR16;
  if (p16 && (reembed || !shortPair(pairs, C))) {
    fprintf(stderr, "the confidence head was handed a bf16 pair it does not take\n"); exit(1);
  }
  const size_t pairBytes = pairs * C * (p16 ? 2 : 4);
  float* pair = inPlace ? const_cast<float*>(trunkPair) : scratch<float>("conf.pair", pairBytes / 4);
  float* single = scratch<float>("conf.single", (size_t)n * Cs);
  if (reembed) {
    boltz2Reembed(pair, single, trunkPair, trunkSingle, targetFeat, pseudoBeta, pairMask, n, C, Cs, F);
  } else {
  if (!inPlace) CK(cudaMemcpyAsync(pair, trunkPair, pairBytes, cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemcpyAsync(single, trunkSingle, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  if (M.flag("trunk.dialect.confidenceGlobalNorm")) {
    // rf3 normalises every detached trunk input over the whole tensor first (target_feat over the
    // vendor's 449 columns, two wider than ours)
    float* tf = scratch<float>("conf.targetFeat", (size_t)n * F);
    CK(cudaMemcpyAsync(tf, targetFeat, (size_t)n * F * 4, cudaMemcpyDeviceToDevice, STREAM));
    if (p16) maskedGlobalNorm<__nv_bfloat16>(pair, pairMask, pairs, C, C);
    else maskedGlobalNorm(pair, pairMask, pairs, C, C);
    maskedGlobalNorm(single, seqMask, n, Cs, Cs);
    maskedGlobalNorm(tf, seqMask, n, F, 449);
    targetFeat = tf;
  }
  float* left = scratch<float>("conf.left", (size_t)n * C); float* right = scratch<float>("conf.right", (size_t)n * C);
  linear<float, float>(targetFeat, left, n, F, C, P + ".leftTargetFeatProject");
  linear<float, float>(targetFeat, right, n, F, C, P + ".rightTargetFeatProject");
  int bins = (int)(lenW(P + ".distogramFeatProject") / C);
  int* binOf = scratch<int>("conf.bin", pairs); float* sqOf = scratch<float>("conf.sq", pairs);
  const bool chai = M.flag("trunk.dialect.chaiConfidence");     // chai-1's 16-bin distance embedding, unmasked
  confidenceBinK<<<blocks(pairs), 256, 0, STREAM>>>(pseudoBeta, n, bins, 3.25f, 50.75f, caDgram, binOf, sqOf, chai);
  if (p16) confidencePairInitK<__nv_bfloat16><<<blocks(pairs * C), 256, 0, STREAM>>>(pair, left, right, binOf, sqOf, pairMask,
    W(P + ".distogramFeatProject"), n, C, Wopt(P + ".distanceFeatProject"), caDgram, chai);
  else confidencePairInitK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, left, right, binOf, sqOf, pairMask,
    W(P + ".distogramFeatProject"), n, C, Wopt(P + ".distanceFeatProject"), caDgram, chai);
  if (bins != (chai ? 16 : caDgram ? 40 : 39)) { fprintf(stderr, "confidence distogram has %d bins\n", bins); exit(1); }
  if (hasW(P + ".inputSingleNormScale")) {
    // the trunk single clamped to +-512 and LayerNormed before any use (protenix2)
    clampK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(single, 512.f, (size_t)n * Cs);
    float* sn = scratch<float>("conf.singleNorm", (size_t)n * Cs);
    layerNorm2<float, float>(single, sn, n, Cs, P + ".inputSingleNormScale", P + ".inputSingleNormOffset");
    CK(cudaMemcpyAsync(single, sn, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  }
  foldDualOutputs();
  memReport("confidence: pair initialised");
  int nb = 0; while (M.has(P + ".blocks." + std::to_string(nb) + ".singleChannels")) ++nb;
  bool swap = M.flag("trunk.dialect.swapTransposedBias"), divide = M.flag("trunk.dialect.triangleMulDivideByLength");
  struct Pair16Scope { bool was; explicit Pair16Scope(bool on) : was(PAIR16) { PAIR16 = on; } ~Pair16Scope() { PAIR16 = was; } };
  {
  Pair16Scope scope(p16);
  if (p16 && !CONF_HALF) { fprintf(stderr, "a bf16 confidence pair wants the f16 head (--fast)\n"); exit(1); }
  for (int k = 0; k < nb; ++k)
    if (CONF_HALF) pairformerBlockAt<half>(pair, single, pairMask, seqMask, n, C, Cs, P + ".blocks." + std::to_string(k), swap, divide);
    else pairformerBlockAt<float>(pair, single, pairMask, seqMask, n, C, Cs, P + ".blocks." + std::to_string(k), swap, divide);
  }
  // on a card short of room the stack's scratch goes before the heads allocate theirs, so the two peak
  // apart rather than together
  if (shortPair(pairs, C)) releaseScratch({ "tri.", "trib.", "grid.", "tr.", "st." });
  memReport("confidence: blocks done");
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
  // chunked, each chunk's logits are taken down to their expectations at once, so they are a chunk's and not the
  // whole matrix's - [pairs, 64] f32 is TWICE a 128-channel f32 pair (25.6 GB at 10,000 tokens)
  float* logits = scratch<float>("conf.logits", (chunked ? rowsPer : pairs) * NB);
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
  // chunked: a chunk's logits (with boltz2's inter-chain half) and their expectation, pair by pair
  auto chunkLogits = [&](const float* xn, size_t r0, size_t cnt, const std::string& w, const std::string& inter) {
    linear<float, float>(xn, logits, cnt, C, NB, P + "." + w);
    if (!hasW(P + "." + inter)) return;
    linear<float, float>(xn, interLogits, cnt, C, NB, P + "." + inter);
    interChainLogitsChunkK<<<blocks(cnt * NB), 256, 0, STREAM>>>(logits, interLogits, asymDev, n, NB, r0, cnt);
  };
  float* tmTerm = scratch<float>("conf.tmTerm", pairs);
  std::vector<float> perBin(NB);
  double d0 = 0;
  {
    std::vector<float> seq = download(seqMask, n);
    int real = 0; for (float v : seq) real += v > 0;
    d0 = 1.24 * std::cbrt(std::max(real, 19) - 15.0) - 1.8;
    for (int b = 0; b < NB; ++b) perBin[b] = (float)(1 / (1 + (double)centres[b] * centres[b] / (d0 * d0)));
  }
  float* dPerBin = upload(perBin.data(), NB);
  if (chunked) {
    // the PDE: AF3 symmetrises AFTER the projection (expectation of l_ij + l_ji), protenix2 and boltz2 before it
    // (LN(z + z^T) W). Chunked, AF3's form is taken through linearity, W LN(z_ij) + W LN(z_ji) = W (LN(z_ij) +
    // LN(z_ji)): one projection of the summed norms, its own chunk - the same logits up to rounding
    float* sym2 = !preSym ? scratch<float>("conf.symT", rowsPer * C) : nullptr;
    float* ln2 = !preSym ? scratch<float>("conf.ln2", rowsPer * C) : nullptr;
    float* own = p16 ? scratch<float>("conf.own", rowsPer * C) : nullptr;
    auto ownRows = [&](size_t r0, size_t cnt) -> const float* {     // a chunk of the pair's rows, as f32
      if (!p16) return pair + r0 * C;
      ownRowsK<__nv_bfloat16><<<blocks(cnt * C), 256, 0, STREAM>>>(pair, own, r0, cnt, C);
      return own;
    };
    for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
      size_t cnt = std::min(rowsPer, pairs - r0);
      const float* x = preSym ? nullptr : ownRows(r0, cnt);
      if (preSym) {
        if (p16) symmetriseRowsK<__nv_bfloat16><<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym, r0, cnt, n, C);
        else symmetriseRowsK<<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym, r0, cnt, n, C);
        x = sym;
      }
      const float* xn = headNorm(x, ln, cnt, C, "logitsLn");
      if (!preSym) {
        // (no head norm, the rows are the pair's own: copied before the transposed rows are added to them)
        if (xn != ln) { CK(cudaMemcpyAsync(ln, xn, cnt * C * 4, cudaMemcpyDeviceToDevice, STREAM)); xn = ln; }
        if (p16) transposedRowsK<__nv_bfloat16><<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym2, r0, cnt, n, C);
        else transposedRowsK<<<blocks(cnt * C), 256, 0, STREAM>>>(pair, sym2, r0, cnt, n, C);
        addK<<<blocks(cnt * C), 256, 0, STREAM>>>(ln, headNorm(sym2, ln2, cnt, C, "logitsLn"), cnt * C);
      }
      chunkLogits(xn, r0, cnt, "leftHalfDistanceLogits", "interHalfDistanceLogits");
      expectationK<<<blocks(cnt), 256, 0, STREAM>>>(logits, pde + r0, pairMask + r0, cnt, NB, dCentres, 0, 1.f);
    }
    for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
      size_t cnt = std::min(rowsPer, pairs - r0);
      chunkLogits(headNorm(ownRows(r0, cnt), ln, cnt, C, "paeLogitsLn"), r0, cnt, "paeLogits", "paeInterLogits");
      expectationK<<<blocks(cnt), 256, 0, STREAM>>>(logits, pae + r0, pairMask + r0, cnt, NB, dCentres, 0, 1.f);
      expectationK<<<blocks(cnt), 256, 0, STREAM>>>(logits, tmTerm + r0, nullptr, cnt, NB, dPerBin, 0, 1.f);
    }
  } else {
    const float* src = pair;
    if (preSym) {
      // symmetrised BEFORE the projection: LN(z + z^T) W (protenix2, boltz2); AF3 adds the transpose after
      float* whole = scratch<float>("conf.sym", pairs * C);
      symmetriseK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, whole, n, C);
      src = whole;
    }
    project(headNorm(src, ln, pairs, C, "logitsLn"), "leftHalfDistanceLogits", "interHalfDistanceLogits");
    expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, pde, pairMask, pairs, NB, dCentres, preSym ? 0 : n, 1.f);
    project(headNorm(pair, ln, pairs, C, "paeLogitsLn"), "paeLogits", "paeInterLogits");
    expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, pae, pairMask, pairs, NB, dCentres, 0, 1.f);
    expectationK<<<blocks(pairs), 256, 0, STREAM>>>(logits, tmTerm, nullptr, pairs, NB, dPerBin, 0, 1.f);
  }
  // pTM and ipTM off the PAE logits: per pair the expected TM term, then the best anchor's
  // mean over the pairs it selects (ipTM: other chains only). shared/heads/tm-score.js.
  {
    std::vector<float> seq = download(seqMask, n);
    const int* asym = M.i("batch.asymId");
    std::vector<float> term = download(tmTerm, pairs);
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
  if (M.flag("trunk.dialect.chaiConfidence")) {
    // chai-1 predicts pLDDT over the 37 ATOM37 slots and gathers each dense slot's by its atom NAME (no match: slot 0)
    float* p37 = scratch<float>("conf.plddt37", (size_t)n * 37 * PB);
    linear<float, float>(sln, p37, n, Cs, 37 * PB, P + ".plddtLogits");
    gatherPlddt37K<<<blocks((size_t)n * dense * PB), 256, 0, STREAM>>>(p37, atom37Index(n, dense), pl, n, dense, PB);
  } else
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
