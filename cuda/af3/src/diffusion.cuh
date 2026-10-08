// The diffusion head - conditioning, the atom encoder, the 24-block conditioned transformer,
// the atom decoder - and the sampler. Transcribed from cpu/af3/diffusion/diffusion.js
// and diffusion-sampler.js.
#pragma once
#include "atom.cuh"

constexpr float SIGMA_DATA = 16.f;

// features2d = [trunk pair (Cz) | relative one-hot (139)] per pair; one-hot built in place
// (pair rows p0 .. p0 + rows; out holds just those rows)
__global__ void pairFeaturesK(const float* trunkPair, const int* residueIndex, const int* tokenIndex,
                              const int* asymId, const int* entityId, const int* symId, float* out, int n,
                              int Cz, int maxIdx, int maxChain, size_t p0, size_t rows) {
  int rel = (2 * maxIdx + 2) * 2 + 1 + (2 * maxChain + 2), width = Cz + rel;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * width) return;
  int c = (int)(t % width); size_t ij = p0 + t / width; int i = (int)(ij / n), j = (int)(ij % n);
  if (c < Cz) { out[t] = trunkPair[ij * Cz + c]; return; }
  int k = c - Cz, positionBins = 2 * maxIdx + 2;
  auto clamp = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
  bool sameChain = asymId[i] == asymId[j], sameEntity = entityId[i] == entityId[j];
  int c0 = sameChain ? clamp(residueIndex[i] - residueIndex[j] + maxIdx, 2 * maxIdx) : 2 * maxIdx + 1;
  bool sameResidue = sameChain && residueIndex[i] == residueIndex[j];
  int c1 = positionBins + (sameResidue ? clamp(tokenIndex[i] - tokenIndex[j] + maxIdx, 2 * maxIdx) : 2 * maxIdx + 1);
  int c2 = positionBins * 2;
  int c3 = positionBins * 2 + 1 + (sameEntity ? clamp(symId[i] - symId[j] + maxChain, 2 * maxChain) : 2 * maxChain + 1);
  out[t] = (k == c0 || k == c1 || k == c3 || (k == c2 && sameEntity)) ? 1.f : 0.f;
}
__global__ void concatK(const float* a, int wa, const float* b, int wb, float* out, int rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int w = wa + wb;
  if (t >= (size_t)rows * w) return;
  int r = (int)(t / w), c = (int)(t % w);
  out[t] = c < wa ? a[(size_t)r * wa + c] : b[(size_t)r * wb + (c - wa)];
}
// out[r] = [a[r] | b[r]] with zero columns at p0 and p1 (-1: none) of the padded row
__global__ void concatPadK(const float* a, int wa, const float* b, int wb, float* out, int rows, int p0, int p1) {
  int w = wa + wb + (p0 >= 0) + (p1 >= 0);
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rows * w) return;
  int r = (int)(t / w), c = (int)(t % w);
  if (c == p0 || c == p1) { out[t] = 0.f; return; }
  int src = c - (p0 >= 0 && c > p0) - (p1 >= 0 && c > p1);
  out[t] = src < wa ? a[(size_t)r * wa + src] : b[(size_t)r * wb + (src - wa)];
}
__global__ void addVectorK(float* x, const float* v, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < rows * C) x[t] += v[t % C];
}

// The plain gated transition used by the conditioning: LN (scale, offset), SwiGLU, project.
// (in row chunks: at 2088 tokens the pair's wide intermediate alone was 8.9 GB)
inline void plainTransition(float* x, size_t rows, int C, int factor, const std::string& P) {
  int I = C * factor;
  size_t per = std::max<size_t>(1, std::min(rows, CHUNK / (2 * I)));
  float* xn = scratch<float>("pt.xn", per * C);
  float* wide = scratch<float>("pt.wide", per * 2 * I);
  float* gated = scratch<float>("pt.gated", per * I);
  for (size_t r0 = 0; r0 < rows; r0 += per) {
    size_t r = std::min(per, rows - r0);
    layerNormSlow(x + r0 * C, xn, r, C, W(P + ".ffwLayerNormScale"), Wopt(P + ".ffwLayerNormOffset"));
    linear<float, float>(xn, wide, r, C, 2 * I, P + ".ffwTransition1");
    swigluK<float><<<blocks(r * I), 256, 0, STREAM>>>(wide, gated, r, I, false);
    linear<float, float>(gated, x + r0 * C, r, I, C, P + ".ffwTransition2", false, 1.f);
  }
}

inline bool DIFF_HALF = false;     // the denoiser's transformer in f16 (set by --fast)
struct Conditioning { float *single, *pair; };
// Everything in the conditioning but the noise term is the same at every step; the pair is
// computed once and the single's noise-free part kept, so a step adds one projection.
struct DiffusionCache { bool ready = false; float *pair, *singleBase; };
inline DiffusionCache DCACHE;

// The step's noise level lives in device memory, so a captured step reads the current one.
inline float* noiseParams = nullptr;
__global__ void fourierK(const float* params, const float* w, const float* b, float* out, int nc) {
  int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= nc) return;
  float tr = 0.25f * logf(params[0] / 16.f);
  out[k] = cosf(6.283185307179586f * (tr * w[k] + b[k]));
}
__global__ void fourierBatchK(const float* levels, const float* w, const float* b, float* out, int nc, int S) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (size_t)S * nc) return;
  int k = (int)(t % nc), s = (int)(t / nc);
  float tr = 0.25f * logf(levels[s] / 16.f);
  out[t] = cosf(6.283185307179586f * (tr * w[k] + b[k]));
}
// out[s][i][c] = base[i][c] + v[s][c]
__global__ void baseplusK(const float* base, const float* v, float* out, int S, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (size_t)S * n * C) return;
  int c = (int)(t % C); size_t si = t / C; int s = (int)(si / n), i = (int)(si % n);
  out[t] = base[(size_t)i * C + c] + v[(size_t)s * C + c];
}
inline void setNoise(float noiseLevel) {
  static float* pinned = nullptr;
  if (!noiseParams) { noiseParams = dalloc(4); CK(cudaMallocHost(&pinned, 16)); }
  pinned[0] = noiseLevel;
  CK(cudaMemcpyAsync(noiseParams, pinned, 4, cudaMemcpyHostToDevice, STREAM));
}
// Set by a streamed preparation (prepareDiffusion, a card short of room): the conditioning pair is not
// kept - each chunk of its rows, once transitioned, is handed here and dropped.
inline std::function<void(const float*, size_t, size_t)> PAIR_CHUNK_SINK;
// chai-1's diffusion pair input beside the trunk pair, rows [p0, p0 + r): its STRUCTURE token-pair features
// (chai-lab chai1.py, token_pair_structure_input_feats), the 163 generator columns through the structure half of
// its projection - docking (class 5: none), relative chain (the dense sym-id rank, 2 +- 2, or 5 across entities),
// relative entity (the dense entity rank, 1 +- 1), the residue and token separations (as the trunk's), the two
// restraints at their masked column - plus a bond term from the token bond matrix
__global__ void chaiStructurePairK(const float* trunkPair, int Czt, const int* residueIndex, const int* tokenIndex,
                                   const int* asymId, const int* entityRank, const int* symRank, const float* Wp,
                                   const float* bias, const float* bonds, const float* Wb, float* out, int n, int Cz,
                                   size_t p0, size_t r) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int width = Czt + Cz;
  if (t >= r * width) return;
  int c = (int)(t % width); size_t row = t / width; size_t ij = p0 + row;
  if (c < Czt) { out[t] = trunkPair[ij * Czt + c]; return; }
  c -= Czt;
  int i = (int)(ij / n), j = (int)(ij % n);
  auto clip = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
  bool sameChain = asymId[i] == asymId[j];
  int rss = sameChain ? clip(residueIndex[i] - residueIndex[j] + 33, 0, 65) : 66;
  int rts = sameChain && residueIndex[i] == residueIndex[j] ? clip(tokenIndex[i] - tokenIndex[j] + 32, 0, 65) : 66;
  int relEntity = entityRank[i] - entityRank[j];
  int rchain = relEntity != 0 ? 5 : clip(symRank[i] - symRank[j] + 2, 0, 4);
  int rent = clip(relEntity + 1, 0, 2);
  float v = bias[c] + Wp[(size_t)5 * Cz + c] + Wp[(size_t)155 * Cz + c] + Wp[(size_t)162 * Cz + c]
          + Wp[(size_t)(6 + rchain) * Cz + c] + Wp[(size_t)(12 + rent) * Cz + c] + Wp[(size_t)(15 + rss) * Cz + c]
          + Wp[(size_t)(82 + rts) * Cz + c];
  if (bonds) v += bonds[ij] * Wb[c];
  out[t] = v;
}
// a dense rank of a per-token id (torch.unique(..., return_inverse=True)), uploaded once per input
inline const int* denseRank(const std::string& key) {
  static std::map<std::string, std::pair<const int*, int*>> cache;
  const int* host = M.i(key); size_t n = M.len(key);
  auto it = cache.find(key);
  if (it != cache.end() && it->second.first == host) return it->second.second;
  std::vector<int> sorted(host, host + n); std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  std::vector<int> rank(n);
  for (size_t k = 0; k < n; ++k) rank[k] = (int)(std::lower_bound(sorted.begin(), sorted.end(), host[k]) - sorted.begin());
  int* d = (int*)upload((const float*)rank.data(), n);
  if (it != cache.end()) CK(cudaFree(it->second.second));
  cache[key] = { host, d };
  return d;
}
// chai-1 closes both conditioning tracks with an affine LayerNorm (af3-any-model diffusion_head.py), and has no
// LayerNorm before the single conditioning's projection into the token transformer (it would undo that one)
inline void pairFinalNorm(float* x, size_t rows) {
  const std::string P = "diffusion.conditioning";
  int Cz = (int)M.meta(P + ".pairChannels");
  float* tmp = scratch<float>("dc.fnorm", rows * Cz);
  layerNormSlow(x, tmp, rows, Cz, W(P + ".pairCondFinalNormScale"), W(P + ".pairCondFinalNormOffset"));
  CK(cudaMemcpyAsync(x, tmp, rows * Cz * 4, cudaMemcpyDeviceToDevice, STREAM));
}
inline void singleFinalNorm(float* x, size_t rows) {
  const std::string P = "diffusion.conditioning";
  if (!hasW(P + ".singleCondFinalNormScale")) return;
  int Cs = (int)M.meta(P + ".seqChannels");
  float* tmp = scratch<float>("dc.sfnorm", rows * Cs);
  layerNormSlow(x, tmp, rows, Cs, W(P + ".singleCondFinalNormScale"), W(P + ".singleCondFinalNormOffset"));
  CK(cudaMemcpyAsync(x, tmp, rows * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
}
inline void singleCondEmbeddingNorm(const float* in, float* out, size_t rows, int Cs) {
  if (!hasW("diffusion.singleCondEmbeddingNormScale")) {     // chai
    CK(cudaMemcpyAsync(out, in, rows * Cs * 4, cudaMemcpyDeviceToDevice, STREAM)); return;
  }
  layerNormSlow(in, out, rows, Cs, W("diffusion.singleCondEmbeddingNormScale"), Wopt("diffusion.singleCondEmbeddingNormOffset"));
}
inline Conditioning diffusionConditioning(const float* trunkSingle, const float* trunkPair,
                                          const float* targetFeat, float noiseLevel, int n) {
  const std::string P = "diffusion.conditioning";
  int Cz = (int)M.meta(P + ".pairChannels"), Cs = (int)M.meta(P + ".seqChannels");
  int Czt = (int)M.meta(P + ".trunkPairChannels"), Cst = (int)M.meta(P + ".trunkSingleChannels");
  int F = (int)M.meta(P + ".targetFeatWidth"), rel = (int)M.meta(P + ".relativeWidth");
  size_t pairs = (size_t)n * n;
  if (!DCACHE.ready) {
    // [trunk pair | relative one-hot] under AF3; the relative encoding projected to the pair width
    // first under some (relpeProjection); and the trunk pair LayerNormed and projected too, both
    // halves Cz wide (zTrunkProjection - OpenDDE, protenix2): three ways into one LayerNorm
    bool split = hasW(P + ".zTrunkProjection"), relpe = !split && hasW(P + ".relpeProjection");
    const bool chai = M.flag("trunk.dialect.chaiDiffusionConditioning");
    int width = chai ? Czt + Cz : split ? 2 * Cz : relpe ? Czt + Cz : Czt + rel;
    // the pair features, normalised and projected in row chunks (whole, they were 9.3 GB at 2088)
    size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / std::max(width, rel + Czt)));
    float* f2 = scratch<float>("dc.f2", per * width);
    float* f2n = scratch<float>("dc.f2n", per * width);
    const bool streamed = (bool)PAIR_CHUNK_SINK;
    DCACHE.pair = streamed ? nullptr : scratch<float>("dc.pair", pairs * Cz);
    float* chunk = streamed ? scratch<float>("dc.pairChunk", per * Cz) : nullptr;
    // a trunk pair parked in host memory (af3.cu, a card short of room) is read a chunk of rows at a time:
    // each chunk copied up, and `trunkPair` a base the chunk's own row offsets land inside
    const bool fromHost = streamed && !trunkPair;
    if (fromHost && !PARK_HOST) { fprintf(stderr, "the conditioning has no trunk pair\n"); exit(1); }
    float* trunkRows = fromHost ? scratch<float>("dc.trunkRows", per * Czt) : nullptr;
    for (size_t p0 = 0; p0 < pairs; p0 += per) {
      size_t r = std::min(per, pairs - p0);
      if (fromHost) {
        CK(cudaMemcpyAsync(trunkRows, PARK_HOST + p0 * Czt, r * Czt * 4, cudaMemcpyHostToDevice, STREAM));
        trunkPair = trunkRows - p0 * Czt;
      }
      auto features = [&](float* outRows, int trunkWidth) {
        pairFeaturesK<<<blocks(r * (trunkWidth + rel)), 256, 0, STREAM>>>(trunkPair, Idev("batch.features.residueIndex"),
          Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"), Idev("batch.features.entityId"),
          Idev("batch.features.symId"), outRows, n, trunkWidth, 32, 2, p0, r);
      };
      if (chai) {
        chaiStructurePairK<<<blocks(r * width), 256, 0, STREAM>>>(trunkPair, Czt, Idev("batch.features.residueIndex"),
          Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"), denseRank("batch.features.entityId"),
          denseRank("batch.features.symId"), W(P + ".structurePairWeights"), W(P + ".structurePairBias"),
          M.has("batch.bondMatrix") ? Fdev("batch.bondMatrix") : nullptr, W(P + ".structureBondWeights"), f2, n, Cz, p0, r);
      } else if (!split && !relpe) features(f2, Czt);
      else {
        float* relRows = scratch<float>("dc.rel", per * rel);
        features(relRows, 0);                              // the one-hot alone
        float* relProj = scratch<float>("dc.relProj", per * Cz);
        linear<float, float>(relRows, relProj, r, rel, Cz, P + ".relpeProjection");
        const float* first = trunkPair + p0 * Czt; int firstWidth = Czt;
        if (split) {
          float* tln = scratch<float>("dc.tln", per * Czt);
          layerNormSlow(trunkPair + p0 * Czt, tln, r, Czt, W(P + ".zTrunkNormScale"), Wopt(P + ".zTrunkNormOffset"));
          float* tproj = scratch<float>("dc.tproj", per * Cz);
          linear<float, float>(tln, tproj, r, Czt, Cz, P + ".zTrunkProjection");
          first = tproj; firstWidth = Cz;
        }
        concatK<<<blocks(r * width), 256, 0, STREAM>>>(first, firstWidth, relProj, Cz, f2, (int)r);
      }
      layerNormSlow(f2, f2n, r, width, W(P + ".pairCondInitialNormScale"), Wopt(P + ".pairCondInitialNormOffset"));
      linear<float, float>(f2n, streamed ? chunk : DCACHE.pair + p0 * Cz, r, width, Cz, P + ".pairCondInitialProjection");
      if (!streamed) continue;
      // (the transitions are row-wise, so a chunk's rows are final once they have run)
      for (int k = 0; k < 2; ++k) plainTransition(chunk, r, Cz, 2, P + ".pairTransitions." + std::to_string(k));
      if (chai) pairFinalNorm(chunk, r);
      PAIR_CHUNK_SINK(chunk, p0, r);
    }
    if (!streamed) {
      for (int k = 0; k < 2; ++k) plainTransition(DCACHE.pair, pairs, Cz, 2, P + ".pairTransitions." + std::to_string(k));
      if (chai) pairFinalNorm(DCACHE.pair, pairs);
    }
    // [trunk single | target_feat]; the openfold3 lineage pads two always-zero columns (unknown DNA,
    // after the restype and the profile blocks) - free before a linear, not before this LayerNorm
    bool pad = M.flag("trunk.dialect.padSingleCondUnknownDna");
    int sw = Cst + F + (pad ? 2 : 0);
    if (lenW(P + ".singleCondInitialNormScale") != (size_t)sw) {
      fprintf(stderr, "single conditioning: the LayerNorm is %zu wide, the features %d\n", lenW(P + ".singleCondInitialNormScale"), sw); exit(1);
    }
    float* f1 = scratch<float>("dc.f1", (size_t)n * sw);
    concatPadK<<<blocks((size_t)n * sw), 256, 0, STREAM>>>(trunkSingle, Cst, targetFeat, F, f1, n,
                                                          pad ? Cst + 31 : -1, pad ? Cst + 63 : -1);
    float* f1n = scratch<float>("dc.f1n", (size_t)n * sw);
    layerNormSlow(f1, f1n, n, sw, W(P + ".singleCondInitialNormScale"), Wopt(P + ".singleCondInitialNormOffset"));
    DCACHE.singleBase = scratch<float>("dc.singleBase", (size_t)n * Cs);
    linear<float, float>(f1n, DCACHE.singleBase, n, sw, Cs, P + ".singleCondInitialProjection");
    if (hasW(P + ".singleCondInitialProjectionBias"))
      addVectorK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(DCACHE.singleBase, W(P + ".singleCondInitialProjectionBias"), n, Cs);
    DCACHE.ready = true;
  }
  // the Fourier noise embedding, on the device off the step's noise level (a graph replays it)
  size_t nc = lenW(P + ".fourierWeight");
  float* e = scratch<float>("dc.emb", nc);
  fourierK<<<blocks(nc), 256, 0, STREAM>>>(noiseParams, W(P + ".fourierWeight"), W(P + ".fourierBias"), e, (int)nc);
  float* en = scratch<float>("dc.embn", nc);
  layerNormSlow(e, en, 1, (int)nc, W(P + ".noiseEmbeddingInitialNormScale"), Wopt(P + ".noiseEmbeddingInitialNormOffset"));
  float* proj = scratch<float>("dc.noiseproj", Cs);
  linear<float, float>(en, proj, 1, (int)nc, Cs, P + ".noiseEmbeddingInitialProjection");
  float* single = scratch<float>("dc.single", (size_t)n * Cs);
  CK(cudaMemcpyAsync(single, DCACHE.singleBase, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  addVectorK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(single, proj, n, Cs);
  for (int k = 0; k < 2; ++k) plainTransition(single, n, Cs, 2, P + ".singleTransitions." + std::to_string(k));
  singleFinalNorm(single, n);
  return { single, DCACHE.pair };
}

// ---------------------------------------------------------------- the transformer
// softmax over keys of logits[h][i][j] * scale + mask + pairLogits[h][i][j], in place
__global__ void tokenSoftmaxK(float* logits, const float* pairLogits, const float* mask, int n, float scale) {
  size_t rowId = blockIdx.x;
  float* L = logits + rowId * n; const float* B = pairLogits + rowId * n;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    float v = L[j] * scale + 1e9f * (mask[j] - 1.f) + B[j]; L[j] = v; mx = fmaxf(mx, v);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  float s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) { float e = expf(L[j] - mx); L[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x) L[j] *= inv;
}
__global__ void gateFromQkvgK(float* o, const float* qkvg, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  size_t i = t / Wd; int c = (int)(t % Wd);
  o[t] *= sigm(qkvg[i * 4 * Wd + 3 * Wd + c]);
}
__global__ void addQBiasF32K(float* qkvg, const float* b, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  qkvg[(t / Wd) * 4 * Wd + (t % Wd)] += b[t % Wd];
}
// LN without affine, two-pass variance, to T
template <class TO>
__global__ void layerNormPlainK(const float* in, TO* out, size_t rows, int C) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* x = in + row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += x[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = x[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) out[row * C + c] = fromF<TO>((x[c] - mean) * inv);
}
// [LN0(x) | 1] and [x | 1], rows of C+1: the inputs of the two conditioning GEMMs
// cuBLAS's kernel choice moves with the ROW count, not just the shape: 68 rows computed as 80 run the
// N = 768 projections 6.7 -> 5.1 us, 261 as 272 the N = 3072 ones 14.6 -> 12.1, and a multiple of 16
// was neutral at every other size measured (150, 522, 1044) - so the transformer's GEMMs run on the
// row count rounded up to 16 (a fixed rule, not a timed one: the kernel picked must not depend on a
// run's timing, or neither would the output). The padding rows compute values nobody reads.
inline size_t padRows16(size_t M) { return (M + 15) / 16 * 16; }
// a scratch buffer's padding rows zeroed when first seen (finite values in, finite out)
inline void zeroOnce(void* p, size_t bytes) {
  static std::map<void*, size_t> seen;
  auto it = seen.find(p);
  if (it != seen.end() && it->second >= bytes) return;
  CK(cudaMemsetAsync(p, 0, bytes, STREAM)); seen[p] = bytes;
}
template <class TO>
__global__ void layerNormPlainOnesK(const float* in, TO* outNorm, TO* outRaw, size_t rows, int C, int Ca) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* x = in + row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += x[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = x[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  size_t base = row * Ca;
  for (int c = lane; c < C; c += 32) { outNorm[base + c] = fromF<TO>((x[c] - mean) * inv); outRaw[base + c] = fromF<TO>(x[c]); }
  for (int c = C + lane; c < Ca; c += 32) {
    float one = c == C ? 1.f : 0.f;
    outNorm[base + c] = fromF<TO>(one); outRaw[base + c] = fromF<TO>(one);
  }
}
template <class T>
__global__ void addQBiasTK(T* qkvg, const float* b, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  size_t k = (t / Wd) * 4 * Wd + (t % Wd);
  qkvg[k] = fromF<T>(toF(qkvg[k]) + b[t % Wd]);
}
template <class T>
__global__ void tokenSoftmaxTK(const float* logits, const float* pairLogits, const float* mask, T* P, int n, float scale) {
  size_t rowId = blockIdx.x;
  const float* L = logits + rowId * n; const float* B = pairLogits + rowId * n;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, L[j] * scale + 1e9f * (mask[j] - 1.f) + B[j]);
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  float s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) s += expf(L[j] * scale + 1e9f * (mask[j] - 1.f) + B[j] - mx);
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    P[rowId * n + j] = fromF<T>(expf(L[j] * scale + 1e9f * (mask[j] - 1.f) + B[j] - mx) * inv);
}
template <class T>
__global__ void gateTK(T* o, const T* qkvg, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  size_t i = t / Wd; int c = (int)(t % Wd);
  o[t] = fromF<T>(toF(o[t]) * sigm(toF(qkvg[i * 4 * Wd + 3 * Wd + c])));
}
// sigmoid(scale) * LN(x) + shift with scale/shift read at a row stride (a column slice), to T
template <class TO, bool RAW = false>
__global__ void adaLnStridedTK(const float* x, const float* scale, const float* shift, int ld, TO* out,
                               size_t rows, int C) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + ADA_EPS);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = fromF<TO>(ADA(scale[row * ld + c]) * ((xr[c] - mean) * inv) + shift[row * ld + c]);
}
// sigmoid(scale) * LN(x) + shift with scale/shift read at a row stride (a column slice)
template <bool RAW = false>
__global__ void adaLnStridedK(const float* x, const float* scale, const float* shift, int ld, float* out,
                              size_t rows, int C) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + ADA_EPS);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = ADA(scale[row * ld + c]) * ((xr[c] - mean) * inv) + shift[row * ld + c];
}
// One row a block: act += y * sigmoid(gate) (if y), then out = sigmoid(scale) * LN(act) + shift,
// scale/shift/gate read at their row strides. Fuses a residual add with the next adaptive LN.
template <class TO, bool RAW = false>
__global__ void gatedAddAdaLnK(float* act, const float* y, const float* gate, int ldg, const float* scale,
                               const float* shift, int lds, TO* out, int C, int period) {
  extern __shared__ float row[];
  __shared__ float red[32];
  size_t r = blockIdx.x, pr = r % period;
  float* a = act + r * C;
  float s = 0;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float v = a[c];
    if (y) { v += y[r * C + c] * sigm(gate[pr * ldg + c]); a[c] = v; }
    row[c] = v; s += v;
  }
  auto blockSum = [&](float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0;
    for (int w = 0; w < (int)(blockDim.x / 32); ++w) t += red[w];
    return t;
  };
  float mean = blockSum(s) / C, v = 0;
  for (int c = threadIdx.x; c < C; c += blockDim.x) { float d = row[c] - mean; v += d * d; }
  float inv = 1.f / sqrtf(blockSum(v) / C + ADA_EPS);
  if (!out) return;
  for (int c = threadIdx.x; c < C; c += blockDim.x)
    out[r * C + c] = fromF<TO>(ADA(scale[pr * lds + c]) * ((row[c] - mean) * inv) + shift[pr * lds + c]);
}
// The same, a warp per row with float4 loads (C a multiple of 128): 68 rows of 768 are too
// few for a block each to pay for its two block-wide reductions.
template <class TO, bool RAW = false>
__global__ void gatedAddAdaLnWarpK(float* act, const float* y, const float* gate, int ldg, const float* scale,
                                   const float* shift, int lds, TO* out, int rows, int C) {
  int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (r >= rows) return;
  float4* a = (float4*)(act + (size_t)r * C);
  const int V = C / 128;                 // float4s per lane
  float4 v[8];
  float s = 0;
  for (int k = 0; k < V; ++k) {
    int c4 = lane + k * 32;
    float4 x = a[c4];
    if (y) {
      float4 yy = ((const float4*)(y + (size_t)r * C))[c4];
      float4 g = ((const float4*)(gate + (size_t)r * ldg))[c4];
      x.x += yy.x * sigm(g.x); x.y += yy.y * sigm(g.y); x.z += yy.z * sigm(g.z); x.w += yy.w * sigm(g.w);
      a[c4] = x;
    }
    v[k] = x; s += x.x + x.y + x.z + x.w;
  }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, q = 0;
  for (int k = 0; k < V; ++k) {
    float dx = v[k].x - mean, dy = v[k].y - mean, dz = v[k].z - mean, dw = v[k].w - mean;
    q += dx * dx + dy * dy + dz * dz + dw * dw;
  }
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
  float inv = 1.f / sqrtf(q / C + ADA_EPS);
  if (!out) return;
  for (int k = 0; k < V; ++k) {
    int c = (lane + k * 32) * 4;
    float4 sc = *(const float4*)(scale + (size_t)r * lds + c), sh = *(const float4*)(shift + (size_t)r * lds + c);
    TO* o = out + (size_t)r * C + c;
    o[0] = fromF<TO>(ADA(sc.x) * ((v[k].x - mean) * inv) + sh.x);
    o[1] = fromF<TO>(ADA(sc.y) * ((v[k].y - mean) * inv) + sh.y);
    o[2] = fromF<TO>(ADA(sc.z) * ((v[k].z - mean) * inv) + sh.z);
    o[3] = fromF<TO>(ADA(sc.w) * ((v[k].w - mean) * inv) + sh.w);
  }
}
// four consecutive values as f32, from f32 or f16
__device__ __forceinline__ float4 load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ float4 load4(const half* p) {
  uint2 u = *reinterpret_cast<const uint2*>(p);
  float2 a = __half22float2(*reinterpret_cast<half2*>(&u.x)), b = __half22float2(*reinterpret_cast<half2*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
// A block a row, one float4 a thread (C/4 threads), the row held in registers. TI: y, gate,
// scale and shift (f16 on the fast path - half the bytes of a kernel that is all bytes).
template <class TO, class TI, bool RAW = false>
__global__ void gatedAddAdaLnVecK(float* act, const TI* y, const TI* gate, int ldg, const TI* scale,
                                  const TI* shift, int lds, TO* out, int C, int period) {
  __shared__ float red[32];
  size_t r = blockIdx.x, pr = r % period; int c = threadIdx.x * 4;
  float4 x = *(float4*)(act + r * C + c);
  if (y) {
    float4 yy = load4(y + r * C + c), g = load4(gate + pr * ldg + c);
    x.x += yy.x * sigm(g.x); x.y += yy.y * sigm(g.y); x.z += yy.z * sigm(g.z); x.w += yy.w * sigm(g.w);
    *(float4*)(act + r * C + c) = x;
  }
  auto blockSum = [&](float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0;
    for (int w = 0; w < (int)(blockDim.x / 32); ++w) t += red[w];
    return t;
  };
  float mean = blockSum(x.x + x.y + x.z + x.w) / C;
  float dx = x.x - mean, dy = x.y - mean, dz = x.z - mean, dw = x.w - mean;
  float inv = rsqrtf(blockSum(dx * dx + dy * dy + dz * dz + dw * dw) / C + ADA_EPS);
  if (!out) return;
  float4 sc = load4(scale + pr * lds + c), sh = load4(shift + pr * lds + c);
  TO* o = out + r * C + c;
  o[0] = fromF<TO>(ADA(sc.x) * (dx * inv) + sh.x); o[1] = fromF<TO>(ADA(sc.y) * (dy * inv) + sh.y);
  o[2] = fromF<TO>(ADA(sc.z) * (dz * inv) + sh.z); o[3] = fromF<TO>(ADA(sc.w) * (dw * inv) + sh.w);
}
// (gate, scale and shift shared by the samples: row r reads row r % period)
template <class TO, class TI>
void gatedAddAdaLn(float* act, const TI* y, const TI* gate, int ldg, const TI* scale, const TI* shift,
                   int lds, TO* out, int rows, int C, int period) {
  if (C % 128 == 0 && C / 4 <= 1024 && ldg % 4 == 0 && lds % 4 == 0) {
    if (ADA_RAW) gatedAddAdaLnVecK<TO, TI, true><<<rows, C / 4, 0, STREAM>>>(act, y, gate, ldg, scale, shift, lds, out, C, period);
    else gatedAddAdaLnVecK<TO, TI><<<rows, C / 4, 0, STREAM>>>(act, y, gate, ldg, scale, shift, lds, out, C, period);
  } else if constexpr (std::is_same_v<TI, float>) {
    if (ADA_RAW) gatedAddAdaLnK<TO, true><<<rows, 256, C * 4, STREAM>>>(act, y, gate, ldg, scale, shift, lds, out, C, period);
    else gatedAddAdaLnK<TO><<<rows, 256, C * 4, STREAM>>>(act, y, gate, ldg, scale, shift, lds, out, C, period);
  } else { fprintf(stderr, "gatedAddAdaLn: no f16-input kernel for %d channels\n", C); exit(1); }
}
// x += y * sigmoid(gate) with the gate read at a row stride, row r % period
template <class TI>
__global__ void addGatedStridedK(float* x, const TI* y, const TI* gate, int ld, size_t rows, int C, int period) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  x[t] += toF(y[t]) * sigm(toF(gate[(r % period) * ld + c]));
}
// x[k * n + i] += v[i] for every sample k (n elements a sample)
__global__ void addBroadcastK(float* x, const float* v, size_t n, size_t total) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < total) x[t] += v[t % n];
}

// Every block's conditioning projections as two matrices, built once on the host:
//   cond' = LN0(cond) (no affine) -> [attn scale | attn shift | ffw scale | ffw shift] per block,
//   the per-block LayerNorm scale folded into the weights (LN_s(x) W = LN0(x) diag(s) W);
//   cond -> [attn zero gate | ffw zero gate] per block.
// out[k][c] = scale[k] * W[k][c] and out[k][C + c] = scale[k] * Wshift[k][c], rows ld apart
__global__ void foldCondK(float* out, size_t ld, const float* scale, const float* w, const float* wshift, int Cc, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)Cc * C) return;
  int k = (int)(t / C), c = (int)(t % C);
  out[k * ld + c] = scale[k] * w[t];
  out[k * ld + C + c] = scale[k] * wshift[t];
}
// flat [(i, j)][blocks * heads] -> block b's [h][i][stride] f16, log2(e)-scaled, zeros past n
__global__ void flatToBiasHalfK(const float* flat, half* out, int block, int nblocks, int heads, int n, int stride) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * n * stride) return;
  int j = (int)(t % stride); size_t rest = t / stride; int i = (int)(rest % n), h = (int)(rest / n);
  out[t] = __float2half(j < n ? flat[((size_t)i * n + j) * nblocks * heads + block * heads + h] * LOG2E : 0.f);
}
// the same over rows [i0, i0 + ri) of i, from a flat chunk holding only those rows
__global__ void flatToBiasHalfRowsK(const float* flat, half* out, int block, int nblocks, int heads, int n, int stride,
                                    int i0, int ri) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * ri * stride) return;
  int j = (int)(t % stride); size_t rest = t / stride; int ii = (int)(rest % ri), h = (int)(rest / ri);
  out[((size_t)h * n + i0 + ii) * stride + j] =
    __float2half(j < n ? flat[((size_t)ii * n + j) * nblocks * heads + block * heads + h] * LOG2E : 0.f);
}
// layerNormSlowK into f16, no offset: the two-pass variance, the module's convention
__global__ void layerNormSlowHalfK(const float* in, half* out, size_t rows, int C, const float* scale) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* x = in + row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += x[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = x[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) out[row * C + c] = __float2half((x[c] - mean) * inv * scale[c]);
}
inline half* PN16_GIVEN = nullptr;     // a streamed preparation's LayerNorm'd pair (prepareTransformer)
inline void refreshSuperBlockBias(int sb, int n);
struct TransformerCache {
  bool ready = false; int n = 0, nblocks = 0;
  // on a card short of room the biases are not kept for all blocks: the LayerNorm'd pair is, in f16, and
  // each super block's biases are made from it as the step reaches it (refreshSuperBlockBias)
  half* pn16 = nullptr; int perSuper = 0, heads = 0, Cz = 0;
  int kept = 0;                        // super blocks [0, kept) keep their own biases; the rest share one set, refreshed
  std::vector<float*> pairLogits;      // [h][i][j] per block
  std::vector<half*> biasHalf;         // the same, f16, log2(e)-scaled, rows padded: the flash kernel's
  int stride = 0;
  std::string wNorm, wRaw;             // synthetic weight names
  float *bNorm, *bRaw;                 // the matching biases (zero where none)
};
inline TransformerCache TCACHE;

inline void prepareTransformer(const float* pairCond, int n) {
  const std::string T = "diffusion.transformer";
  int C = (int)M.meta(T + ".channels"), Cc = (int)M.meta(T + ".condChannels"), Cz = (int)M.meta(T + ".pairChannels");
  int heads = (int)M.meta(T + ".heads"), perSuper = (int)M.meta(T + ".blocksPerSuperBlock");
  // a per-block pair LayerNorm (OpenDDE, protenix2) arrives folded into AF3's layout - its scales
  // inside the per-block projections, the shared scale all ones - so it needs nothing here

  TransformerCache& tc = TCACHE;
  size_t pairs = (size_t)n * n;
  std::vector<std::string> names;
  for (int sb = 0; hasW(T + ".superBlocks." + std::to_string(sb) + ".pairLogitsProjection"); ++sb)
    for (int k = 0; k < perSuper; ++k) names.push_back(T + ".superBlocks." + std::to_string(sb) + ".blocks." + std::to_string(k));
  tc.nblocks = (int)names.size();
  if (tc.wNorm.empty()) {
    // the folded weights, once per process, built on the device from its copy of the weights (on
    // the host this was 250 ms of every first fold): rows are the conditioning's Cc channels, then
    // the biases (the GEMM's input carries a column of ones - the add after it was 60 us a step at
    // 261 tokens), then zero rows up to a multiple of 8 so the f16 GEMM's leading dimension stays
    // aligned (K = 385 put cuBLAS on a slow kernel and the whole step got slower)
    int Ca = (Cc + 1 + 7) / 8 * 8;
    size_t ldn = (size_t)tc.nblocks * 4 * C, ldr = (size_t)tc.nblocks * 2 * C;
    float* wn = dalloc((size_t)Ca * ldn); float* wr = dalloc((size_t)Ca * ldr);
    CK(cudaMemsetAsync(wn, 0, (size_t)Ca * ldn * 4, STREAM)); CK(cudaMemsetAsync(wr, 0, (size_t)Ca * ldr * 4, STREAM));
    for (int b = 0; b < tc.nblocks; ++b) {
      const std::string& B = names[b];
      for (int slot = 0; slot < 2; ++slot) {
        std::string pre = B + (slot ? ".ffw" : ".");
        // [scale | shift] for this block's (slot 0) attention or (slot 1) transition LN, the LN
        // scale folded in; the scale's bias in the bias row
        // (chai's adaLN: the conditioning not normalised - a scale of ones folded - and the scale's +1 as its bias)
        foldCondK<<<blocks((size_t)Cc * C), 256, 0, STREAM>>>(wn + (size_t)b * 4 * C + slot * 2 * C, ldn,
          ADA_RAW ? ones(Cc) : W(pre + "SingleCondLayerNormScale"), W(pre + "SingleCondScaleWeights"), W(pre + "SingleCondBias"),
          Cc, C);
        CK(cudaMemcpyAsync(wn + (size_t)Cc * ldn + (size_t)b * 4 * C + slot * 2 * C, ADA_RAW ? ones(C) : W(pre + "SingleCondScaleBias"),
                           C * 4, cudaMemcpyDeviceToDevice, STREAM));
        // the zero-init gate: raw weights and its bias
        CK(cudaMemcpy2DAsync(wr + (size_t)b * 2 * C + slot * C, ldr * 4, W(pre + "AdaptiveZeroCondWeights"), C * 4,
                             C * 4, Cc, cudaMemcpyDeviceToDevice, STREAM));
        CK(cudaMemcpyAsync(wr + (size_t)Cc * ldr + (size_t)b * 2 * C + slot * C, W(pre + "AdaptiveZeroCondBias"),
                           C * 4, cudaMemcpyDeviceToDevice, STREAM));
      }
    }
    tc.wNorm = T + ".condNorm~"; tc.wRaw = T + ".condRaw~";
    deviceWeight(tc.wNorm, wn, (size_t)Ca * ldn); deviceWeight(tc.wRaw, wr, (size_t)Ca * ldr);
  }
  // the pair logits of every block, from the (fold-constant) pair conditioning
  tc.stride = (n + 7) / 8 * 8;
  tc.pn16 = nullptr; tc.kept = 0;      // (an earlier fold's, in a resident process)
  // (recomputing costs a step time - 2.3 s of a 2096-token fold - so only where every block's biases would
  // not fit with room to spare)
  // a streamed preparation hands the LayerNorm'd pair in f16 (pairCond null): the biases are made from it,
  // every block's now where they fit with room to spare, else a super block at a time in the step
  const bool given = pairCond == nullptr;
  if (given && !(DIFF_HALF && PN16_GIVEN)) { fprintf(stderr, "a streamed preparation needs the f16 path\n"); exit(1); }
  if (DIFF_HALF && (given || shortPair(pairs, Cz))) {
    // as many super blocks' biases kept as the card has room for (what an earlier fold's bias and f16-pair buffers
    // hold counted), the rest made in the step - all or nothing it was: at 4,000 tokens the 12.3 GB of every
    // block's biases did not fit, so all six super blocks were remade at every step (diffusion 4.2 s at 3,000
    // tokens, 75.6 s at 4,000)
    const int nSB = (tc.nblocks + perSuper - 1) / perSuper;
    const size_t sbBytes = (size_t)perSuper * heads * n * tc.stride * 2, pnBytes = given ? 0 : pairs * Cz * 2;
    size_t held = 0;
    for (auto& [k, v] : SCRATCH) if (!k.compare(0, 5, "dt.bh") || k == "dt.pn16") held += v.second;
    auto fits = [&](size_t need) { return roomFor(need > held ? need - held : 0); };
    int kept = fits((size_t)tc.nblocks * heads * n * tc.stride * 2) ? nSB : 0;
    if (kept < nSB) while (kept + 1 < nSB && fits((size_t)(kept + 2) * sbBytes + pnBytes)) ++kept;   // kept + the shared set
    bool lazy = kept < nSB;
    if (given || lazy) {
      // the LayerNorm'd pair kept in f16 (half the f32 pair it is made from); with `lazy`, biases for one
      // super block at a time, made as the step reaches it - 24 blocks' biases held were 768 bytes a pair,
      // 3.4 GB at 2096 tokens, the diffusion's largest tensor
      tc.pn16 = given ? PN16_GIVEN : scratch<half>("dt.pn16", pairs * Cz);
      if (!given) {
        size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / Cz));
        for (size_t r0 = 0; r0 < pairs; r0 += per) {
          size_t r = std::min(per, pairs - r0);
          layerNormSlowHalfK<<<(unsigned)((r + 7) / 8), 256, 0, STREAM>>>(pairCond + r0 * Cz, tc.pn16 + r0 * Cz, r, Cz,
                                                                          W(T + ".pairInputLayerNormScale"));
        }
      }
      tc.perSuper = perSuper; tc.heads = heads; tc.Cz = Cz;
      tc.pairLogits.assign(tc.nblocks, nullptr);
      tc.biasHalf.assign(tc.nblocks, nullptr);
      for (int b = 0; b < tc.nblocks; ++b)
        tc.biasHalf[b] = scratch<half>(b / perSuper < kept ? "dt.bh" + std::to_string(b) : "dt.bhL" + std::to_string(b % perSuper),
                                       (size_t)heads * n * tc.stride);
      tc.n = n; tc.ready = true; tc.kept = kept;
      for (int sb = 0; sb < kept; ++sb) refreshSuperBlockBias(sb, n);    // the kept super blocks' biases now
      if (!lazy) { releaseScratch({ "dt.pn16" }); tc.pn16 = nullptr; }  // every one kept: the f16 pair given back
      return;
    }
  }
  if (given) { fprintf(stderr, "a streamed preparation reached the whole-pair path\n"); exit(1); }
  float* pn = scratch<float>("dt.pn", pairs * Cz);
  layerNormSlow(pairCond, pn, pairs, Cz, W(T + ".pairInputLayerNormScale"), nullptr);
  float* flat = scratch<float>("dt.flat", pairs * perSuper * heads);
  // per block: the f32 [h][i][j] logits for the precise path, or (f16 path) only the flash
  // kernel's form - f16, scaled by log2(e), rows padded to 8 - which is half the bytes and all the
  // f16 path reads (the f32 set was 6.7 GB at 2088 tokens)
  tc.stride = (n + 7) / 8 * 8;
  tc.pairLogits.assign(tc.nblocks, nullptr);
  tc.biasHalf.assign(tc.nblocks, nullptr);
  for (int b = 0; b < tc.nblocks; ++b) {
    if (b % perSuper == 0)
      linear<float, float>(pn, flat, pairs, Cz, perSuper * heads,
                           T + ".superBlocks." + std::to_string(b / perSuper) + ".pairLogitsProjection");
    if (DIFF_HALF) {
      tc.biasHalf[b] = scratch<half>("dt.bh" + std::to_string(b), (size_t)heads * n * tc.stride);
      flatToBiasHalfK<<<blocks((size_t)heads * n * tc.stride), 256, 0, STREAM>>>(flat, tc.biasHalf[b], b % perSuper,
                                                                              perSuper, heads, n, tc.stride);
    } else {
      tc.pairLogits[b] = scratch<float>("dt.pl" + std::to_string(b), (size_t)heads * pairs);
      atomLogitsLayoutK<<<blocks((size_t)heads * pairs), 256, 0, STREAM>>>(flat, tc.pairLogits[b], b % perSuper, perSuper, 1, heads, n, n);
    }
  }
  tc.n = n; tc.ready = true;
}

// q and k LayerNormed over each token's whole heads x dimension row, two-pass, scale and offset
// (rf3's kq_norm: after the projection and its bias, before the key_dim scaling); a block a row
template <class T>
__global__ void kqNormK(T* qkvg, const float* qs, const float* qo, const float* ks, const float* ko, int Wd) {
  __shared__ float red[32];
  T* row = qkvg + (size_t)blockIdx.x * 4 * Wd;
  auto blockSum = [&](float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0; for (int w = 0; w < (int)(blockDim.x / 32); ++w) t += red[w];
    return t;
  };
  for (int side = 0; side < 2; ++side) {
    T* x = row + side * Wd; const float* sc = side ? ks : qs; const float* of = side ? ko : qo;
    float s = 0; for (int c = threadIdx.x; c < Wd; c += blockDim.x) s += toF(x[c]);
    float mean = blockSum(s) / Wd, q = 0;
    for (int c = threadIdx.x; c < Wd; c += blockDim.x) { float d = toF(x[c]) - mean; q += d * d; }
    float inv = rsqrtf(blockSum(q) / Wd + 1e-5f);
    for (int c = threadIdx.x; c < Wd; c += blockDim.x) x[c] = fromF<T>((toF(x[c]) - mean) * inv * sc[c] + of[c]);
    __syncthreads();
  }
}
// super block sb's biases from the f16 LayerNorm'd pair, into the buffers its blocks share with every other
// super block's (see TransformerCache::pn16): row chunks of the projection, laid out as the flash kernel reads
inline void refreshSuperBlockBias(int sb, int n) {
  TransformerCache& tc = TCACHE;
  int ps = tc.perSuper, heads = tc.heads, Cz = tc.Cz;
  int ri = (int)std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * std::max(Cz, ps * heads))));
  float* flatc = scratch<float>("dt.flat", (size_t)ri * n * ps * heads);
  std::string w = "diffusion.transformer.superBlocks." + std::to_string(sb) + ".pairLogitsProjection";
  for (int i0 = 0; i0 < n; i0 += ri) {
    int r = std::min(ri, n - i0); size_t rows = (size_t)r * n;
    linear<half, float>(tc.pn16 + (size_t)i0 * n * Cz, flatc, rows, Cz, ps * heads, w);
    for (int b = sb * ps; b < std::min(tc.nblocks, (sb + 1) * ps); ++b)
      flatToBiasHalfRowsK<<<blocks((size_t)heads * r * tc.stride), 256, 0, STREAM>>>(flatc, tc.biasHalf[b], b % ps,
                                                                                ps, heads, n, tc.stride, i0, r);
  }
}
template <class T>
void diffusionTransformer(float* act, const float* cond, const float* mask, int n) {
  const std::string Tn = "diffusion.transformer";
  TransformerCache& tc = TCACHE;
  int C = (int)M.meta(Tn + ".channels"), Cc = (int)M.meta(Tn + ".condChannels");
  int heads = (int)M.meta(Tn + ".heads"), D = (int)M.meta(Tn + ".dimension"), Wd = heads * D;
  int perSuper = (int)M.meta(Tn + ".blocksPerSuperBlock"), factor = (int)M.meta(Tn + ".transitionFactor");
  size_t pairs = (size_t)n * n;
  int ldn = tc.nblocks * 4 * C, ldr = tc.nblocks * 2 * C;
  size_t rows = (size_t)n * NS;                    // every sample's tokens; the conditioning is shared
  // every block's conditioning, two GEMMs over [x | 1] (the biases are the weights' last row)
  int Ca = (Cc + 1 + 7) / 8 * 8;
  T* gNorm = scratch<T>("dt.gNorm", (size_t)n * ldn);
  T* gRaw = scratch<T>("dt.gRaw", (size_t)n * ldr);
  {
    T* cn = scratch<T>("dt.cn", (size_t)n * Ca);
    T* condT = scratch<T>("dt.condT", (size_t)n * Ca);
    layerNormPlainOnesK<T><<<(unsigned)((n + 7) / 8), 256, 0, STREAM>>>(cond, cn, condT, n, Cc, Ca);
    linear<T, T>(ADA_RAW ? condT : cn, gNorm, n, Ca, ldn, tc.wNorm);      // (chai: the raw conditioning)
    linear<T, T>(condT, gRaw, n, Ca, ldr, tc.wRaw);
  }
  // the key mask once per sample (the flash kernel reads it per row of its batch)
  const float* maskRows = mask;
  if (NS > 1) {
    float* mr = scratch<float>("dt.maskRows", rows);
    for (int k = 0; k < NS; ++k) CK(cudaMemcpyAsync(mr + (size_t)k * n, mask, n * 4, cudaMemcpyDeviceToDevice, STREAM));
    maskRows = mr;
  }
  size_t prows = std::is_same_v<T, half> ? padRows16(rows) : rows;   // the fast path's GEMM rows
  T* x = scratch<T>("dt.x", prows * C);
  T* qkvg = scratch<T>("dt.qkvg", (prows + 128) * 4 * Wd);
  // (the precise path's alone: the f16 path's flash kernel holds no [heads, n, n] logits)
  float* logits = std::is_same_v<T, half> ? nullptr : scratch<float>("dt.logits", (size_t)heads * pairs);
  T* P = std::is_same_v<T, half> ? nullptr : scratch<T>("dt.P", (size_t)heads * pairs);
  T* o = scratch<T>("dt.o", prows * Wd);
  T* att = scratch<T>("dt.att", prows * C);
  T* tn = scratch<T>("dt.tn", prows * C);
  int I = C * factor;
  T* wide = scratch<T>("dt.wide", prows * 3 * I);
  T* gated = scratch<T>("dt.gated", prows * I);
  T* proj = scratch<T>("dt.proj", prows * C);
  zeroOnce(x, prows * C * sizeof(T)); zeroOnce(o, prows * Wd * sizeof(T));
  zeroOnce(tn, prows * C * sizeof(T)); zeroOnce(gated, prows * I * sizeof(T));

  // rf3's block wiring: the transition reads the block's INPUT (both still add to act)
  bool noResidual = M.flag(Tn + ".noResidual");
  float* pre = noResidual ? scratch<float>("dt.pre", rows * C) : nullptr;
  const float one = 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  for (int b = 0; b < tc.nblocks; ++b) {
    std::string B = Tn + ".superBlocks." + std::to_string(b / perSuper) + ".blocks." + std::to_string(b % perSuper);
    const T* g = gNorm + (size_t)b * 4 * C; const T* z = gRaw + (size_t)b * 2 * C;
    // the previous block's transition residual, fused with this block's first adaptive LN
    const T* zPrev = b > 0 ? gRaw + (size_t)(b - 1) * 2 * C + C : nullptr;
    gatedAddAdaLn<T>(act, b > 0 ? proj : nullptr, zPrev, ldr, g, g + C, ldn, x, (int)rows, C, n);
    if (noResidual) CK(cudaMemcpyAsync(pre, act, rows * C * 4, cudaMemcpyDeviceToDevice, STREAM));
    { std::string wq = qkvgWeight(B, C, Wd, false); linear<T, T>(x, qkvg, prows, C, 4 * Wd, wq); }
    bool kqNorm = hasW(B + ".queryLayerNormScale");
    if (kqNorm) {
      addQBiasTK<T><<<blocks(rows * Wd), 256, 0, STREAM>>>(qkvg, W(B + ".qBias"), (int)rows, Wd);
      kqNormK<T><<<(unsigned)rows, 256, 0, STREAM>>>(qkvg, W(B + ".queryLayerNormScale"), W(B + ".queryLayerNormOffset"),
                                                    W(B + ".keyLayerNormScale"), W(B + ".keyLayerNormOffset"), Wd);
    }
    if constexpr (std::is_same_v<T, half>) {
      // one fused kernel: the query bias, QK^T, pair bias, mask, online softmax, PV and the gate;
      // the samples are its batch rows, the pair bias shared
      if (tc.pn16 && b % perSuper == 0 && b / perSuper >= tc.kept) refreshSuperBlockBias(b / perSuper, n);
      flashGrid<half>(qkvg, tc.biasHalf[b], tc.stride, MASK_ALL_ONES ? nullptr : maskRows, o, n, heads, D, 0, NS, false,
                      1.f / sqrtf((float)D), kqNorm ? nullptr : W(B + ".qBias"));
    } else {
      if (!kqNorm) addQBiasTK<T><<<blocks(rows * Wd), 256, 0, STREAM>>>(qkvg, W(B + ".qBias"), (int)rows, Wd);
      for (int k = 0; k < NS; ++k) {               // the precise path one sample at a time
        T* qk = qkvg + (size_t)k * n * 4 * Wd; T* ok = o + (size_t)k * n * Wd;
        CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, D, &one, qk + Wd, cudaType<T>(), 4 * Wd, D,
           qk, cudaType<T>(), 4 * Wd, D, &zero, logits, CUDA_R_32F, n, pairs, heads, CUBLAS_COMPUTE_32F, algo));
        tokenSoftmaxTK<T><<<(unsigned)(heads * n), 128, 0, STREAM>>>(logits, tc.pairLogits[b], mask, P, n, 1.f / sqrtf((float)D));
        CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, D, n, n, &one, qk + 2 * Wd, cudaType<T>(), 4 * Wd, D,
           P, cudaType<T>(), n, pairs, &zero, ok, cudaType<T>(), Wd, D, heads, CUBLAS_COMPUTE_32F, algo));
      }
      gateTK<T><<<blocks(rows * Wd), 256, 0, STREAM>>>(o, qkvg, (int)rows, Wd);
    }
    // (chai has no gating query: its zero weights gate by exactly 0.5, undone here)
    linear<T, T>(o, att, prows, Wd, C, B + ".Transition2", false, 0.f, ADA_RAW ? 2.f : 1.f);
    if (noResidual) {
      gatedAddAdaLn<T>(act, att, z, ldr, g + 2 * C, g + 3 * C, ldn, (T*)nullptr, (int)rows, C, n);
      gatedAddAdaLn<T>(pre, (const T*)nullptr, (const T*)nullptr, ldr, g + 2 * C, g + 3 * C, ldn, tn, (int)rows, C, n);
    } else {
      gatedAddAdaLn<T>(act, att, z, ldr, g + 2 * C, g + 3 * C, ldn, tn, (int)rows, C, n);
    }
    bool up; std::string w1 = upGatedTransition1(B, C, I, up);
    linear<T, T>(tn, wide, prows, C, up ? 3 * I : 2 * I, w1);
    swiglu<T>(wide, gated, rows, I, up);
    linear<T, T>(gated, proj, prows, I, C, B + ".ffwTransition2");
  }
  // the last block's transition residual
  addGatedStridedK<T><<<blocks(rows * C), 256, 0, STREAM>>>(act, proj, gRaw + (size_t)(tc.nblocks - 1) * 2 * C + C, ldr,
                                                         rows, C, n);
}

// ---------------------------------------------------------------- the decoder
__global__ void broadcastTokensK(const float* proj, float* perAtom, int tokens, int dense, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * dense * C) return;
  int c = (int)(t % C); size_t ta = t / C; int token = (int)(ta / dense);
  perAtom[t] = proj[(size_t)token * C + c];
}
// act[q] = (t2q.mask[q] ? proj[token of t2q.idx[q]] : 0) + skip[q], times the query's mask - what
// broadcastTokensK, convert(t2q, ...) and addSkipMaskK computed in three passes
__global__ void broadcastSkipK(const float* proj, const int* idx, const float* gmask, const float* skip,
                               const float* qMask, float* act, size_t rows, int C, size_t q1, int tokens, int dense) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t q = t / C; int c = (int)(t % C);
  size_t gq = q % q1, k = q / q1;
  float v = gmask[gq] != 0 ? proj[((size_t)idx[gq] / dense + k * tokens) * C + c] : 0.f;
  act[t] = (v + skip[t]) * qMask[gq];
}
// upd[row] = LN(act[row] * qMask) W, W (C, 3): scaleByRowK, layerNormSlowK and linear in one pass
__global__ void maskLnProject3K(const float* act, const float* qMask, size_t q1, size_t rows, int C,
                                const float* scale, const float* offset, const float* Wp, float* upd) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* x = act + row * C;
  float m = qMask[row % q1];
  float s = 0;
  for (int c = lane; c < C; c += 32) s += x[c] * m;
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = x[c] * m - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  float y0 = 0, y1 = 0, y2 = 0;
  for (int c = lane; c < C; c += 32) {
    float l = (x[c] * m - mean) * inv * scale[c] + (offset ? offset[c] : 0.f);
    y0 += l * Wp[c * 3]; y1 += l * Wp[c * 3 + 1]; y2 += l * Wp[c * 3 + 2];
  }
  for (int o = 16; o; o >>= 1) {
    y0 += __shfl_xor_sync(~0u, y0, o); y1 += __shfl_xor_sync(~0u, y1, o); y2 += __shfl_xor_sync(~0u, y2, o);
  }
  if (lane == 0) { upd[row * 3] = y0; upd[row * 3 + 1] = y1; upd[row * 3 + 2] = y2; }
}
__global__ void addSkipMaskK(float* act, const float* skip, const float* mask, size_t rows, int C, size_t period) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) act[t] = (act[t] + skip[t]) * mask[(t / C) % period];
}
struct DecoderCache { std::vector<AtomBlockCache> blocks; int C, heads, D, perToken; };
inline DecoderCache prepareDecoder(const EncoderOut& enc) {
  const std::string Dd = "diffusion.decoder";
  AtomShape sh = atomShape();
  DecoderCache d;
  d.C = (int)M.meta(Dd + ".channels"); int Cp = (int)M.meta(Dd + ".pairChannels");
  d.heads = (int)M.meta(Dd + ".heads"); d.D = (int)M.meta(Dd + ".dimension");
  d.perToken = (int)M.meta(Dd + ".perTokenChannels");
  size_t qRows = (size_t)sh.subsets * sh.queries;
  int nblocks = 0; while (M.has(Dd + ".blocks." + std::to_string(nblocks) + ".qProjection")) ++nblocks;
  std::vector<float*> logits = atomPairLogits(Dd, enc.pair, qRows * sh.keys, Cp, nblocks, d.heads, sh);
  const float* cond = enc.qCond;
  if (M.flag("trunk.dialect.chaiAtomStack")) {
    // chai conditions its decoder on a second, affine LayerNorm of the encoder's conditioning, and restricts its
    // attention to one reference space as the encoder does
    float* c2 = scratch<float>("dec.cond", qRows * d.C);
    layerNormSlow(enc.qCond, c2, qRows, d.C, W(Dd + ".postAtomCondLayerNormScale"), W(Dd + ".postAtomCondLayerNormOffset"));
    cond = c2;
    Gather tq = gatherOf("batch.tokensToQueries"), tk = gatherOf("batch.tokensToKeys");
    size_t per = (size_t)sh.subsets * d.heads * sh.queries * sh.keys;
    for (float* pl : logits)
      sameRefSpaceMaskK<<<blocks(per), 256, 0, STREAM>>>(pl, enc.qUid, tq.mask, enc.kUid, tk.mask, sh.subsets, d.heads,
                                                         sh.queries, sh.keys);
  }
  for (int b = 0; b < nblocks; ++b)
    d.blocks.push_back(prepareAtomBlock(Dd + ".blocks." + std::to_string(b), cond, qRows, d.C, logits[b]));
  return d;
}
inline float* atomDecoder(const float* tokenAct, const EncoderOut& enc, const DecoderCache& d) {
  const std::string Dd = "diffusion.decoder";
  AtomShape sh = atomShape();
  int C = d.C;
  size_t atoms = (size_t)sh.tokens * sh.dense, q1 = (size_t)sh.subsets * sh.queries, qRows = q1 * NS;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2t = gatherOf("batch.queriesToTokenAtoms");
  float* proj = scratch<float>("dec.proj", (size_t)sh.tokens * NS * C);
  linear<float, float>(tokenAct, proj, (size_t)sh.tokens * NS, d.perToken, C, Dd + ".projectTokenFeaturesForBroadcast");
  // broadcast to token atoms, gather to queries, add the skip and mask: one pass, nothing between
  float* act = scratch<float>("dec.act", qRows * C);
  broadcastSkipK<<<blocks(qRows * C), 256, 0, STREAM>>>(proj, t2q.idx, t2q.mask, enc.skip, enc.qMask, act, qRows, C,
                                                       q1, sh.tokens, sh.dense);
  AtomStep st{ gatherOf("batch.queriesToKeys"), enc.qMask, enc.kMask, M.flag(Dd + ".blocks.0.keyMaskedAtomAttention"),
               M.flag(Dd + ".blocks.0.diffusionNoResidual") };
  bool maskPerBlock = M.flag(Dd + ".blocks.0.maskAtomActPerBlock");
  for (size_t b = 0; b < d.blocks.size(); ++b) {
    if (maskPerBlock) scaleByRowK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, enc.qMask, qRows, C, q1);
    crossAttentionBlock(act, st, d.blocks[b], sh, C, d.heads, d.D, Dd + ".blocks." + std::to_string(b));
  }
  // the query mask, the LayerNorm and the three-column projection, a warp a row: as a cuBLAS GEMM
  // with N = 3 the projection read the normalised rows at a fraction of the bandwidth
  float* upd = scratch<float>("dec.upd", qRows * 3);
  if (lenW(Dd + ".atomFeaturesToPositionUpdate") != (size_t)C * 3) { fprintf(stderr, "decoder: position update is not C x 3\n"); exit(1); }
  maskLnProject3K<<<(unsigned)((qRows + 7) / 8), 256, 0, STREAM>>>(act, enc.qMask, q1, qRows, C,
    W(Dd + ".atomFeaturesLayerNormScale"), Wopt(Dd + ".atomFeaturesLayerNormOffset"), W(Dd + ".atomFeaturesToPositionUpdate"), upd);
  float* out = scratch<float>("dec.out", atoms * NS * 3);
  convert(q2t, upd, out, 3, q1, NS);
  return out;
}

// ---------------------------------------------------------------- one denoiser call
// EDM's scalings off the device noise level: skip, out and input
__device__ inline void scalings(const float* params, float& skip, float& out, float& in) {
  float s = params[0], d = s * s + 256.f;
  skip = 256.f / d; out = s * 16.f * rsqrtf(d); in = rsqrtf(d);
}
// (over every sample's atoms: `total` atoms, the mask one sample's, `atoms` long)
__global__ void scalePositionsK(const float* x, const float* mask, float* y, size_t total, size_t atoms,
                                const float* params) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total * 3) return;
  float skip, out, in; scalings(params, skip, out, in);
  y[t] = x[t] * mask[(t / 3) % atoms] * in;
}
__global__ void denoiseOutK(const float* x, const float* upd, const float* mask, float* o, size_t total, size_t atoms,
                            const float* params) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total * 3) return;
  float skip, out, in; scalings(params, skip, out, in);
  o[t] = (skip * x[t] + out * upd[t]) * mask[(t / 3) % atoms];
}

// Everything the denoiser computes once per fold.
struct DiffusionFold {
  const float *trunkSingle, *trunkPair, *targetFeat, *seqMask;
  int n;
  EncoderOut enc; DecoderCache dec;
  cudaGraphExec_t graph = nullptr; const float* graphInput = nullptr; int calls = 0, graphNs = 0;
  // every step's single conditioning and its embedding projection, computed in one batch before
  // sampling (precomputeConditioning) - a step only copies its slices in
  std::vector<float> preLevels; float *preSingle = nullptr, *preSnProj = nullptr; bool usePre = false;
};
inline bool GRAPHS = true;
// the denoiser's seams against a stage oracle (oracle.stages.stages.<name>), when one was exported
inline void dtap(const char* name, const float* d, size_t n) {
  if (!d) return;                 // (a tensor given back: the conditioning pair of a large fold)
  std::string k = std::string("oracle.stages.stages.") + name;
  cudaStreamCaptureStatus capturing;
  CK(cudaStreamIsCapturing(STREAM, &capturing));
  if (capturing != cudaStreamCaptureStatusNone) return;
  if (M.has(k) && M.len(k) == n) check((std::string("  ") + name).c_str(), d, n, k);
}
inline DiffusionFold prepareDiffusion(const float* trunkSingle, const float* trunkPair, const float* targetFeat,
                                      const float* seqMask, int n) {
  DiffusionFold f{ trunkSingle, trunkPair, targetFeat, seqMask, n, {}, {} };
  DCACHE.ready = false;
  setNoise(SIGMA_DATA);
  const std::string Pc = "diffusion.conditioning", E = "diffusion.encoder", T = "diffusion.transformer";
  int Cz = (int)M.meta(Pc + ".pairChannels");
  size_t pairs = (size_t)n * n;
  if (DIFF_HALF && shortPair(pairs, Cz) && hasW(E + ".embedTrunkPairCond")) {
    // a card short of room: the conditioning pair streamed - each chunk of rows straight into the
    // encoder's pair projection and the transformer's f16 LayerNorm'd pair, never the f32 pair whole
    // (9 GB at 4192 tokens, beside the trunk's)
    int Cp = (int)M.meta(E + ".pairChannels");
    float* tp = scratch<float>("enc.tp", pairs * Cp);
    half* pn16 = scratch<half>("dt.pn16", pairs * Cz);
    PAIR_CHUNK_SINK = [&](const float* chunk, size_t p0, size_t r) {
      float* ln = scratch<float>("enc.tpln", r * Cz);
      layerNormSlow(chunk, ln, r, Cz, W(E + ".lnormTrunkPairCondScale"), Wopt(E + ".lnormTrunkPairCondOffset"));
      linear<float, float>(ln, tp + p0 * Cp, r, Cz, Cp, E + ".embedTrunkPairCond");
      layerNormSlowHalfK<<<(unsigned)((r + 7) / 8), 256, 0, STREAM>>>(chunk, pn16 + p0 * Cz, r, Cz,
                                                                      W(T + ".pairInputLayerNormScale"));
    };
    diffusionConditioning(trunkSingle, trunkPair, targetFeat, SIGMA_DATA, n);
    PAIR_CHUNK_SINK = nullptr;
    // the chunk loop's working rows given back before the encoder and decoder prepare beside tp and pn16
    // (0.9 GB at 1530 tokens, which is what they ran out of)
    releaseScratch({ "dc.f2", "dc.f2n", "dc.pairChunk", "dc.trunkRows", "dc.rel", "dc.relProj", "dc.tln", "dc.tproj",
                     "enc.tpln", "pt." });
    ENC_TP_GIVEN = tp;
    f.enc = prepareEncoder(E, "atomReference", trunkSingle, nullptr);
    ENC_TP_GIVEN = nullptr;
    f.dec = prepareDecoder(f.enc);
    PN16_GIVEN = pn16;
    prepareTransformer(nullptr, n);
    PN16_GIVEN = nullptr;
    return f;
  }
  Conditioning cond = diffusionConditioning(trunkSingle, trunkPair, targetFeat, SIGMA_DATA, n);  // builds the pair
  f.enc = prepareEncoder(E, "atomReference", trunkSingle, cond.pair);
  f.dec = prepareDecoder(f.enc);
  prepareTransformer(cond.pair, n);
  return f;
}
// The conditioning of every step at once: it depends on the step only through its noise level,
// and the sampler knows all of them before it starts - so the Fourier embedding, its projection,
// both transitions and the single-conditioning projection run as a few large GEMMs over
// steps x tokens rows instead of ~15 small kernels a step (68 tokens: ~0.1 ms of a 1.9 ms step)
inline void precomputeConditioning(DiffusionFold& f, const std::vector<float>& levels) {
  const std::string P = "diffusion.conditioning";
  int S = (int)levels.size(), n = f.n;
  int Cs = (int)M.meta(P + ".seqChannels"), Cse = (int)M.meta("diffusion.seqChannels"), perToken = (int)M.meta("diffusion.perTokenChannels");
  if (Cs != Cse) { fprintf(stderr, "conditioning width %d against %d\n", Cs, Cse); exit(1); }
  if (!DCACHE.ready) diffusionConditioning(f.trunkSingle, f.trunkPair, f.targetFeat, SIGMA_DATA, n);   // the base
  // every step's at once is steps x tokens rows - 1.8 GB at 100 steps and 2900 tokens - so where that does
  // not fit with room to spare each step makes its own (denoiseCore's !usePre path; a few small kernels)
  if (f.preSingle) { CK(cudaFree(f.preSingle)); CK(cudaFree(f.preSnProj)); f.preSingle = f.preSnProj = nullptr; }
  if (!roomFor((size_t)S * n * (2 * Cs + perToken) * 4)) { f.usePre = false; return; }
  size_t nc = lenW(P + ".fourierWeight");
  float* lv = upload(levels.data(), S);
  float* e = dalloc((size_t)S * nc); float* en = dalloc((size_t)S * nc); float* proj = dalloc((size_t)S * Cs);
  fourierBatchK<<<blocks((size_t)S * nc), 256, 0, STREAM>>>(lv, W(P + ".fourierWeight"), W(P + ".fourierBias"), e, (int)nc, S);
  layerNormSlow(e, en, S, (int)nc, W(P + ".noiseEmbeddingInitialNormScale"), Wopt(P + ".noiseEmbeddingInitialNormOffset"));
  linear<float, float>(en, proj, S, (int)nc, Cs, P + ".noiseEmbeddingInitialProjection");
  size_t rows = (size_t)S * n;
  f.preSingle = dalloc(rows * Cs); f.preSnProj = dalloc(rows * perToken);
  baseplusK<<<blocks(rows * Cs), 256, 0, STREAM>>>(DCACHE.singleBase, proj, f.preSingle, S, n, Cs);
  for (int k = 0; k < 2; ++k) plainTransition(f.preSingle, rows, Cs, 2, P + ".singleTransitions." + std::to_string(k));
  singleFinalNorm(f.preSingle, rows);
  float* sn = dalloc(rows * Cs);
  singleCondEmbeddingNorm(f.preSingle, sn, rows, Cs);
  linear<float, float>(sn, f.preSnProj, rows, Cs, perToken, "diffusion.singleCondEmbeddingProjection");
  CK(cudaStreamSynchronize(STREAM));
  for (float* p : {lv, e, en, proj, sn}) CK(cudaFree(p));
  scratch<float>("dc.single", (size_t)n * Cs); scratch<float>("dn.snProj", (size_t)n * perToken);   // a step's slices land here
  f.preLevels = levels; f.usePre = true;
  // 🔴 NOT the transformer's two conditioning GEMMs over every step too: held for the schedule they
  // were rows x 24 blocks x 6C halves - 3.0 GB at 68 tokens x 200 steps, 2.3 at 525 x 25 - for 2% of
  // the diffusion at 68 x 200 (308.8 against 315.6 ms), 1% at 68 x 25, and a LOSS at 262 and 525
  // (71.5 against 69.9 ms, 101.3 against 98.1), interleaved on an A100.
}
// D(x; sigma): positions [NS][tokens*dense][3] in, the denoised positions out.
inline float* denoiseCore(DiffusionFold& f, const float* positionsNoisy, float noiseLevel) {
  int n = f.n, dense = (int)M.meta("batch.dense");
  size_t atoms = (size_t)n * dense, total = atoms * NS;
  stage(nullptr);
  Conditioning cond;
  if (f.usePre) cond = { scratch<float>("dc.single", (size_t)n * (int)M.meta("diffusion.conditioning.seqChannels")), DCACHE.pair };
  else cond = diffusionConditioning(f.trunkSingle, f.trunkPair, f.targetFeat, noiseLevel, n);
  stage("d.conditioning");
  dtap("conditioning.single", cond.single, (size_t)n * (int)M.meta("diffusion.conditioning.seqChannels"));
  dtap("conditioning.pair", cond.pair, (size_t)n * n * (int)M.meta("diffusion.conditioning.pairChannels"));
  const float* atomMask = Fdev("batch.refMask");
  float* scaled = scratch<float>("dn.scaled", total * 3);
  scalePositionsK<<<blocks(total * 3), 256, 0, STREAM>>>(positionsNoisy, atomMask, scaled, total, atoms, noiseParams);
  encoderStep("diffusion.encoder", f.enc, scaled); stage("d.encoder");
  int Cs = (int)M.meta("diffusion.seqChannels"), perToken = (int)M.meta("diffusion.perTokenChannels");
  float* snProj = scratch<float>("dn.snProj", (size_t)n * perToken);
  if (!f.usePre) {
    float* sn = scratch<float>("dn.sn", (size_t)n * Cs);
    singleCondEmbeddingNorm(cond.single, sn, n, Cs);
    linear<float, float>(sn, snProj, n, Cs, perToken, "diffusion.singleCondEmbeddingProjection");
  }
  size_t rows = (size_t)n * NS;
  float* act = scratch<float>("dn.act", rows * perToken);
  CK(cudaMemcpyAsync(act, f.enc.tokenAct, rows * perToken * 4, cudaMemcpyDeviceToDevice, STREAM));
  dtap("encoder.tokenAct", f.enc.tokenAct, rows * perToken);
  addBroadcastK<<<blocks(rows * perToken), 256, 0, STREAM>>>(act, snProj, (size_t)n * perToken, rows * perToken);
  dtap("transformer.act", act, rows * perToken);
  if (getenv("TX_ORACLE_IN") && M.has("oracle.stages.stages.transformer.act"))    // the transformer alone, on the oracle's input
    CK(cudaMemcpyAsync(act, Fdev("oracle.stages.stages.transformer.act"), rows * perToken * 4, cudaMemcpyDeviceToDevice, STREAM));
  if (DIFF_HALF) diffusionTransformer<half>(act, cond.single, f.seqMask, n);
  else diffusionTransformer<float>(act, cond.single, f.seqMask, n);
  stage("d.transformer");
  dtap("transformer.out", act, rows * perToken);
  float* actn = scratch<float>("dn.actn", rows * perToken);
  layerNormSlow(act, actn, rows, perToken, W("diffusion.outputNormScale"), Wopt("diffusion.outputNormOffset"));
  float* upd = atomDecoder(actn, f.enc, f.dec); stage("d.decoder");
  dtap("decoder.update", upd, total * 3);
  float* out = scratch<float>("dn.out", total * 3);
  denoiseOutK<<<blocks(total * 3), 256, 0, STREAM>>>(positionsNoisy, upd, atomMask, out, total, atoms, noiseParams);
  return out;
}
// One denoiser call. After a first (allocating) call the step is captured as a CUDA graph and
// replayed: about 300 small launches become one.
// With `deviceLevel`, the noise level is read from device memory, so a caller can enqueue steps
// without waiting on the host.
inline float* denoiseStep(DiffusionFold& f, const float* positionsNoisy, float noiseLevel,
                          const float* deviceLevel = nullptr) {
  if (deviceLevel) {
    if (!noiseParams) setNoise(noiseLevel);
    CK(cudaMemcpyAsync(noiseParams, deviceLevel, 4, cudaMemcpyDeviceToDevice, STREAM));
  } else setNoise(noiseLevel);
  if (f.usePre) {           // this step's precomputed conditioning, into the buffers the step reads
    auto at = std::find(f.preLevels.begin(), f.preLevels.end(), noiseLevel);
    if (at == f.preLevels.end()) { fprintf(stderr, "noise level %g was not precomputed\n", noiseLevel); exit(1); }
    size_t s = at - f.preLevels.begin(), n = f.n;
    int Cs = (int)M.meta("diffusion.conditioning.seqChannels"), perToken = (int)M.meta("diffusion.perTokenChannels");
    CK(cudaMemcpyAsync(scratch<float>("dc.single", n * Cs), f.preSingle + s * n * Cs, n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
    CK(cudaMemcpyAsync(scratch<float>("dn.snProj", n * perToken), f.preSnProj + s * n * perToken, n * perToken * 4,
                       cudaMemcpyDeviceToDevice, STREAM));
  }
  if (!GRAPHS || STAGES || f.calls++ == 0) return denoiseCore(f, positionsNoisy, noiseLevel);
  if (!f.graph || positionsNoisy != f.graphInput || NS != f.graphNs) {     // (a batch of another size: its own graph)
    if (f.graph) CK(cudaGraphExecDestroy(f.graph));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(STREAM, cudaStreamCaptureModeThreadLocal));
    denoiseCore(f, positionsNoisy, noiseLevel);
    CK(cudaStreamEndCapture(STREAM, &g));
    CK(cudaGraphInstantiate(&f.graph, g, 0));
    CK(cudaGraphDestroy(g));
    f.graphInput = positionsNoisy; f.graphNs = NS;
  }
  CK(cudaGraphLaunch(f.graph, STREAM));
  return scratch<float>("dn.out", (size_t)f.n * (int)M.meta("batch.dense") * 3 * NS);
}
inline float* denoise(const float* trunkSingle, const float* trunkPair, const float* targetFeat,
                      const float* seqMask, const float* positionsNoisy, float noiseLevel) {
  DiffusionFold f = prepareDiffusion(trunkSingle, trunkPair, targetFeat, seqMask, (int)M.meta("batch.tokens"));
  return denoiseStep(f, positionsNoisy, noiseLevel);
}
