// The diffusion head - conditioning, the atom encoder, the 24-block conditioned transformer,
// the atom decoder - and the sampler. Transcribed from src/af3/diffusion/diffusion-reference.js
// and diffusion-sampler-reference.js.
#pragma once
#include "atom.cuh"

constexpr float SIGMA_DATA = 16.f;

// features2d = [trunk pair (Cz) | relative one-hot (139)] per pair; one-hot built in place
__global__ void pairFeaturesK(const float* trunkPair, const int* residueIndex, const int* tokenIndex,
                              const int* asymId, const int* entityId, const int* symId, float* out, int n,
                              int Cz, int maxIdx, int maxChain) {
  int rel = (2 * maxIdx + 2) * 2 + 1 + (2 * maxChain + 2), width = Cz + rel;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * width) return;
  int c = (int)(t % width); size_t ij = t / width; int i = (int)(ij / n), j = (int)(ij % n);
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
__global__ void addVectorK(float* x, const float* v, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < rows * C) x[t] += v[t % C];
}

// The plain gated transition used by the conditioning: LN (scale, offset), SwiGLU, project.
inline void plainTransition(float* x, size_t rows, int C, int factor, const std::string& P) {
  int I = C * factor;
  float* xn = scratch<float>("pt.xn", rows * C);
  float* wide = scratch<float>("pt.wide", rows * 2 * I);
  float* gated = scratch<float>("pt.gated", rows * I);
  layerNormSlow(x, xn, rows, C, W(P + ".ffwLayerNormScale"), Wopt(P + ".ffwLayerNormOffset"));
  linear<float, float>(xn, wide, rows, C, 2 * I, P + ".ffwTransition1");
  swigluK<float><<<blocks(rows * I), 256, 0, STREAM>>>(wide, gated, rows, I);
  linear<float, float>(gated, x, rows, I, C, P + ".ffwTransition2", false, 1.f);
}

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
inline void setNoise(float noiseLevel) {
  static float* pinned = nullptr;
  if (!noiseParams) { noiseParams = dalloc(4); CK(cudaMallocHost(&pinned, 16)); }
  pinned[0] = noiseLevel;
  CK(cudaMemcpyAsync(noiseParams, pinned, 4, cudaMemcpyHostToDevice, STREAM));
}
inline Conditioning diffusionConditioning(const float* trunkSingle, const float* trunkPair,
                                          const float* targetFeat, float noiseLevel, int n) {
  const std::string P = "diffusion.conditioning";
  int Cz = (int)M.meta(P + ".pairChannels"), Cs = (int)M.meta(P + ".seqChannels");
  int Czt = (int)M.meta(P + ".trunkPairChannels"), Cst = (int)M.meta(P + ".trunkSingleChannels");
  int F = (int)M.meta(P + ".targetFeatWidth"), rel = (int)M.meta(P + ".relativeWidth");
  size_t pairs = (size_t)n * n;
  if (hasW(P + ".zTrunkProjection") || hasW(P + ".relpeProjection")) {
    fprintf(stderr, "split/projected relpos conditioning: not ported\n"); exit(1);
  }
  if (M.flag("trunk.dialect.padSingleCondUnknownDna")) { fprintf(stderr, "padded single cond: not ported\n"); exit(1); }
  if (!DCACHE.ready) {
    int width = Czt + rel;
    float* f2 = scratch<float>("dc.f2", pairs * width);
    pairFeaturesK<<<blocks(pairs * width), 256, 0, STREAM>>>(trunkPair, Idev("batch.features.residueIndex"),
      Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"), Idev("batch.features.entityId"),
      Idev("batch.features.symId"), f2, n, Czt, 32, 2);
    float* f2n = scratch<float>("dc.f2n", pairs * width);
    layerNormSlow(f2, f2n, pairs, width, W(P + ".pairCondInitialNormScale"), Wopt(P + ".pairCondInitialNormOffset"));
    DCACHE.pair = dalloc(pairs * Cz);
    linear<float, float>(f2n, DCACHE.pair, pairs, width, Cz, P + ".pairCondInitialProjection");
    for (int k = 0; k < 2; ++k) plainTransition(DCACHE.pair, pairs, Cz, 2, P + ".pairTransitions." + std::to_string(k));
    int sw = Cst + F;
    float* f1 = scratch<float>("dc.f1", (size_t)n * sw);
    concatK<<<blocks((size_t)n * sw), 256, 0, STREAM>>>(trunkSingle, Cst, targetFeat, F, f1, n);
    float* f1n = scratch<float>("dc.f1n", (size_t)n * sw);
    layerNormSlow(f1, f1n, n, sw, W(P + ".singleCondInitialNormScale"), Wopt(P + ".singleCondInitialNormOffset"));
    DCACHE.singleBase = dalloc((size_t)n * Cs);
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
template <class TO>
__global__ void castK(const float* x, TO* y, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = fromF<TO>(x[i]);
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
template <class TO>
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
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = fromF<TO>(sigm(scale[row * ld + c]) * ((xr[c] - mean) * inv) + shift[row * ld + c]);
}
// sigmoid(scale) * LN(x) + shift with scale/shift read at a row stride (a column slice)
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
  float inv = 1.f / sqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = sigm(scale[row * ld + c]) * ((xr[c] - mean) * inv) + shift[row * ld + c];
}
// One row a block: act += y * sigmoid(gate) (if y), then out = sigmoid(scale) * LN(act) + shift,
// scale/shift/gate read at their row strides. Fuses a residual add with the next adaptive LN.
template <class TO>
__global__ void gatedAddAdaLnK(float* act, const float* y, const float* gate, int ldg, const float* scale,
                               const float* shift, int lds, TO* out, int C) {
  extern __shared__ float row[];
  __shared__ float red[32];
  size_t r = blockIdx.x;
  float* a = act + r * C;
  float s = 0;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float v = a[c];
    if (y) { v += y[r * C + c] * sigm(gate[r * ldg + c]); a[c] = v; }
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
  float inv = 1.f / sqrtf(blockSum(v) / C + 1e-5f);
  if (!out) return;
  for (int c = threadIdx.x; c < C; c += blockDim.x)
    out[r * C + c] = fromF<TO>(sigm(scale[r * lds + c]) * ((row[c] - mean) * inv) + shift[r * lds + c]);
}
// The same, a warp per row with float4 loads (C a multiple of 128): 68 rows of 768 are too
// few for a block each to pay for its two block-wide reductions.
template <class TO>
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
  float inv = 1.f / sqrtf(q / C + 1e-5f);
  if (!out) return;
  for (int k = 0; k < V; ++k) {
    int c = (lane + k * 32) * 4;
    float4 sc = *(const float4*)(scale + (size_t)r * lds + c), sh = *(const float4*)(shift + (size_t)r * lds + c);
    TO* o = out + (size_t)r * C + c;
    o[0] = fromF<TO>(sigm(sc.x) * ((v[k].x - mean) * inv) + sh.x);
    o[1] = fromF<TO>(sigm(sc.y) * ((v[k].y - mean) * inv) + sh.y);
    o[2] = fromF<TO>(sigm(sc.z) * ((v[k].z - mean) * inv) + sh.z);
    o[3] = fromF<TO>(sigm(sc.w) * ((v[k].w - mean) * inv) + sh.w);
  }
}
// A block a row, one float4 a thread (C/4 threads), the row held in registers.
template <class TO>
__global__ void gatedAddAdaLnVecK(float* act, const float* y, const float* gate, int ldg, const float* scale,
                                  const float* shift, int lds, TO* out, int C) {
  __shared__ float red[32];
  size_t r = blockIdx.x; int c = threadIdx.x * 4;
  float4 x = *(float4*)(act + r * C + c);
  if (y) {
    float4 yy = *(const float4*)(y + r * C + c), g = *(const float4*)(gate + r * ldg + c);
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
  float inv = rsqrtf(blockSum(dx * dx + dy * dy + dz * dz + dw * dw) / C + 1e-5f);
  if (!out) return;
  float4 sc = *(const float4*)(scale + r * lds + c), sh = *(const float4*)(shift + r * lds + c);
  TO* o = out + r * C + c;
  o[0] = fromF<TO>(sigm(sc.x) * (dx * inv) + sh.x); o[1] = fromF<TO>(sigm(sc.y) * (dy * inv) + sh.y);
  o[2] = fromF<TO>(sigm(sc.z) * (dz * inv) + sh.z); o[3] = fromF<TO>(sigm(sc.w) * (dw * inv) + sh.w);
}
template <class TO>
void gatedAddAdaLn(float* act, const float* y, const float* gate, int ldg, const float* scale, const float* shift,
                   int lds, TO* out, int rows, int C) {
  if (C % 128 == 0 && C / 4 <= 1024 && ldg % 4 == 0 && lds % 4 == 0)
    gatedAddAdaLnVecK<TO><<<rows, C / 4, 0, STREAM>>>(act, y, gate, ldg, scale, shift, lds, out, C);
  else
    gatedAddAdaLnK<TO><<<rows, 256, C * 4, STREAM>>>(act, y, gate, ldg, scale, shift, lds, out, C);
}
// x += y * sigmoid(gate) with the gate read at a row stride
__global__ void addGatedStridedK(float* x, const float* y, const float* gate, int ld, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  x[t] += y[t] * sigm(gate[r * ld + c]);
}

// Every block's conditioning projections as two matrices, built once on the host:
//   cond' = LN0(cond) (no affine) -> [attn scale | attn shift | ffw scale | ffw shift] per block,
//   the per-block LayerNorm scale folded into the weights (LN_s(x) W = LN0(x) diag(s) W);
//   cond -> [attn zero gate | ffw zero gate] per block.
// [h][i][j] f32 -> [h][i][stride] f16 * log2(e), zeros past n
__global__ void padBiasK(const float* in, half* out, int n, int stride, int heads) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * n * stride) return;
  int j = (int)(t % stride); size_t hi = t / stride;
  out[t] = __float2half(j < n ? in[hi * n + j] * LOG2E : 0.f);
}
struct TransformerCache {
  bool ready = false; int n = 0, nblocks = 0;
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
  if (M.flag(T + ".pairNormPerBlock") || M.flag(T + ".noResidual")) { fprintf(stderr, "transformer dialect: not ported\n"); exit(1); }
  TransformerCache& tc = TCACHE;
  size_t pairs = (size_t)n * n;
  std::vector<std::string> names;
  for (int sb = 0; hasW(T + ".superBlocks." + std::to_string(sb) + ".pairLogitsProjection"); ++sb)
    for (int k = 0; k < perSuper; ++k) names.push_back(T + ".superBlocks." + std::to_string(sb) + ".blocks." + std::to_string(k));
  tc.nblocks = (int)names.size();
  if (tc.wNorm.empty()) {
    // the folded weights, once per process
    std::vector<float> wn((size_t)Cc * tc.nblocks * 4 * C), wr((size_t)Cc * tc.nblocks * 2 * C);
    std::vector<float> bn((size_t)tc.nblocks * 4 * C, 0.f), br((size_t)tc.nblocks * 2 * C, 0.f);
    size_t ldn = (size_t)tc.nblocks * 4 * C, ldr = (size_t)tc.nblocks * 2 * C;
    for (int b = 0; b < tc.nblocks; ++b) {
      const std::string& B = names[b];
      auto fold = [&](const std::string& prefix, int slot) {
        const float* sc = M.f(B + prefix + "SingleCondLayerNormScale");
        const float* ws = M.f(B + prefix + "SingleCondScaleWeights"); const float* wh = M.f(B + prefix + "SingleCondBias");
        const float* bs = M.f(B + prefix + "SingleCondScaleBias");
        for (int k = 0; k < Cc; ++k) for (int c = 0; c < C; ++c) {
          wn[k * ldn + (size_t)b * 4 * C + slot * C + c] = sc[k] * ws[(size_t)k * C + c];
          wn[k * ldn + (size_t)b * 4 * C + (slot + 1) * C + c] = sc[k] * wh[(size_t)k * C + c];
        }
        for (int c = 0; c < C; ++c) bn[(size_t)b * 4 * C + slot * C + c] = bs[c];
      };
      fold(".", 0); fold(".ffw", 2);
      auto gate = [&](const std::string& prefix, int slot) {
        const float* w = M.f(B + prefix + "AdaptiveZeroCondWeights"); const float* bb = M.f(B + prefix + "AdaptiveZeroCondBias");
        for (int k = 0; k < Cc; ++k) for (int c = 0; c < C; ++c) wr[k * ldr + (size_t)b * 2 * C + slot * C + c] = w[(size_t)k * C + c];
        for (int c = 0; c < C; ++c) br[(size_t)b * 2 * C + slot * C + c] = bb[c];
      };
      gate(".", 0); gate(".ffw", 1);
      if (hasW(B + ".ffwAToB")) { fprintf(stderr, "ffwAToB: not ported\n"); exit(1); }
    }
    tc.wNorm = T + ".condNorm~"; tc.wRaw = T + ".condRaw~";
    SYNTH[tc.wNorm] = std::move(wn); SYNTH[tc.wRaw] = std::move(wr);
    tc.bNorm = upload(bn.data(), bn.size()); tc.bRaw = upload(br.data(), br.size());
  }
  // the pair logits of every block, from the (fold-constant) pair conditioning
  float* pn = scratch<float>("dt.pn", pairs * Cz);
  layerNormSlow(pairCond, pn, pairs, Cz, W(T + ".pairInputLayerNormScale"), nullptr);
  float* flat = scratch<float>("dt.flat", pairs * perSuper * heads);
  tc.pairLogits.resize(tc.nblocks);
  for (int b = 0; b < tc.nblocks; ++b) {
    if (b % perSuper == 0)
      linear<float, float>(pn, flat, pairs, Cz, perSuper * heads,
                           T + ".superBlocks." + std::to_string(b / perSuper) + ".pairLogitsProjection");
    tc.pairLogits[b] = scratch<float>("dt.pl" + std::to_string(b), (size_t)heads * pairs);
    atomLogitsLayoutK<<<blocks((size_t)heads * pairs), 256, 0, STREAM>>>(flat, tc.pairLogits[b], b % perSuper, perSuper, 1, heads, n, n);
  }
  // the flash kernel's form of the same logits: f16, scaled by log2(e), rows padded to 8
  tc.stride = (n + 7) / 8 * 8;
  tc.biasHalf.resize(tc.nblocks);
  for (int b = 0; b < tc.nblocks; ++b) {
    tc.biasHalf[b] = scratch<half>("dt.bh" + std::to_string(b), (size_t)heads * n * tc.stride);
    padBiasK<<<blocks((size_t)heads * n * tc.stride), 256, 0, STREAM>>>(tc.pairLogits[b], tc.biasHalf[b], n, tc.stride, heads);
  }
  tc.n = n; tc.ready = true;
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
  // every block's conditioning, two GEMMs
  T* cn = scratch<T>("dt.cn", (size_t)n * Cc);
  layerNormPlainK<T><<<(unsigned)((n + 7) / 8), 256, 0, STREAM>>>(cond, cn, n, Cc);
  T* condT = scratch<T>("dt.condT", (size_t)n * Cc);
  castK<T><<<blocks((size_t)n * Cc), 256, 0, STREAM>>>(cond, condT, (size_t)n * Cc);
  float* gNorm = scratch<float>("dt.gNorm", (size_t)n * ldn);
  float* gRaw = scratch<float>("dt.gRaw", (size_t)n * ldr);
  linear<T, float>(cn, gNorm, n, Cc, ldn, tc.wNorm);
  addVectorK<<<blocks((size_t)n * ldn), 256, 0, STREAM>>>(gNorm, tc.bNorm, n, ldn);
  linear<T, float>(condT, gRaw, n, Cc, ldr, tc.wRaw);
  addVectorK<<<blocks((size_t)n * ldr), 256, 0, STREAM>>>(gRaw, tc.bRaw, n, ldr);
  T* x = scratch<T>("dt.x", (size_t)n * C);
  T* qkvg = scratch<T>("dt.qkvg", (size_t)(n + 128) * 4 * Wd);
  float* logits = scratch<float>("dt.logits", (size_t)heads * pairs);
  T* P = scratch<T>("dt.P", (size_t)heads * pairs);
  T* o = scratch<T>("dt.o", (size_t)n * Wd);
  float* att = scratch<float>("dt.att", (size_t)n * C);
  T* tn = scratch<T>("dt.tn", (size_t)n * C);
  int I = C * factor;
  T* wide = scratch<T>("dt.wide", (size_t)n * 2 * I);
  T* gated = scratch<T>("dt.gated", (size_t)n * I);
  float* proj = scratch<float>("dt.proj", (size_t)n * C);
  const float one = 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  for (int b = 0; b < tc.nblocks; ++b) {
    std::string B = Tn + ".superBlocks." + std::to_string(b / perSuper) + ".blocks." + std::to_string(b % perSuper);
    const float* g = gNorm + (size_t)b * 4 * C; const float* z = gRaw + (size_t)b * 2 * C;
    // the previous block's transition residual, fused with this block's first adaptive LN
    const float* zPrev = b > 0 ? gRaw + (size_t)(b - 1) * 2 * C + C : nullptr;
    gatedAddAdaLn<T>(act, b > 0 ? proj : nullptr, zPrev, ldr, g, g + C, ldn, x, n, C);
    linear<T, T>(x, qkvg, n, C, 4 * Wd, qkvgWeight(B, C, Wd, false));
    if constexpr (std::is_same_v<T, half>) {
      // one fused kernel: the query bias, QK^T, pair bias, mask, online softmax, PV and the gate
      flashGrid<half>(qkvg, tc.biasHalf[b], tc.stride, mask, o, n, heads, D, 0, 1, false, 1.f / sqrtf((float)D),
                      W(B + ".qBias"));
    } else {
      addQBiasTK<T><<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(qkvg, W(B + ".qBias"), n, Wd);
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, D, &one, qkvg + Wd, cudaType<T>(), 4 * Wd, D,
         qkvg, cudaType<T>(), 4 * Wd, D, &zero, logits, CUDA_R_32F, n, pairs, heads, CUBLAS_COMPUTE_32F, algo));
      tokenSoftmaxTK<T><<<(unsigned)(heads * n), 128, 0, STREAM>>>(logits, tc.pairLogits[b], mask, P, n, 1.f / sqrtf((float)D));
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, D, n, n, &one, qkvg + 2 * Wd, cudaType<T>(), 4 * Wd, D,
         P, cudaType<T>(), n, pairs, &zero, o, cudaType<T>(), Wd, D, heads, CUBLAS_COMPUTE_32F, algo));
      gateTK<T><<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(o, qkvg, n, Wd);
    }
    linear<T, float>(o, att, n, Wd, C, B + ".Transition2");
    gatedAddAdaLn<T>(act, att, z, ldr, g + 2 * C, g + 3 * C, ldn, tn, n, C);
    linear<T, T>(tn, wide, n, C, 2 * I, B + ".ffwTransition1");
    swigluK<T><<<blocks((size_t)n * I), 256, 0, STREAM>>>(wide, gated, n, I);
    linear<T, float>(gated, proj, n, I, C, B + ".ffwTransition2");
  }
  // the last block's transition residual
  addGatedStridedK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(act, proj, gRaw + (size_t)(tc.nblocks - 1) * 2 * C + C, ldr, n, C);
}
inline bool DIFF_HALF = false;     // the denoiser's transformer in f16 (set by --fast)

// ---------------------------------------------------------------- the decoder
__global__ void broadcastTokensK(const float* proj, float* perAtom, int tokens, int dense, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * dense * C) return;
  int c = (int)(t % C); size_t ta = t / C; int token = (int)(ta / dense);
  perAtom[t] = proj[(size_t)token * C + c];
}
__global__ void addSkipMaskK(float* act, const float* skip, const float* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) act[t] = (act[t] + skip[t]) * mask[t / C];
}
struct DecoderCache { std::vector<AtomBlockCache> blocks; int C, heads, D, perToken; };
inline DecoderCache prepareDecoder(const EncoderOut& enc) {
  const std::string Dd = "diffusion.decoder";
  AtomShape sh = atomShape();
  DecoderCache d;
  d.C = (int)M.meta(Dd + ".channels"); int Cp = (int)M.meta(Dd + ".pairChannels");
  d.heads = (int)M.meta(Dd + ".heads"); d.D = (int)M.meta(Dd + ".dimension");
  d.perToken = (int)M.meta(Dd + ".perTokenChannels");
  size_t qRows = (size_t)sh.subsets * sh.queries, kRows = (size_t)sh.subsets * sh.keys;
  int nblocks = 0; while (M.has(Dd + ".blocks." + std::to_string(nblocks) + ".qProjection")) ++nblocks;
  std::vector<float*> logits = atomPairLogits(Dd, enc.pair, qRows * sh.keys, Cp, nblocks, d.heads, sh);
  for (int b = 0; b < nblocks; ++b)
    d.blocks.push_back(prepareAtomBlock(Dd + ".blocks." + std::to_string(b), enc.qCond, enc.kCond, qRows, kRows, d.C, logits[b]));
  return d;
}
inline float* atomDecoder(const float* tokenAct, const EncoderOut& enc, const DecoderCache& d) {
  const std::string Dd = "diffusion.decoder";
  AtomShape sh = atomShape();
  int C = d.C;
  size_t atoms = (size_t)sh.tokens * sh.dense, qRows = (size_t)sh.subsets * sh.queries;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2t = gatherOf("batch.queriesToTokenAtoms");
  float* proj = scratch<float>("dec.proj", (size_t)sh.tokens * C);
  linear<float, float>(tokenAct, proj, sh.tokens, d.perToken, C, Dd + ".projectTokenFeaturesForBroadcast");
  float* perAtom = scratch<float>("dec.perAtom", atoms * C);
  broadcastTokensK<<<blocks(atoms * C), 256, 0, STREAM>>>(proj, perAtom, sh.tokens, sh.dense, C);
  float* act = scratch<float>("dec.act", qRows * C);
  convert(t2q, perAtom, act, C);
  addSkipMaskK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, enc.skip, enc.qMask, qRows, C);
  AtomStep st{ gatherOf("batch.queriesToKeys"), enc.qMask, enc.kMask, M.flag(Dd + ".blocks.0.keyMaskedAtomAttention"),
               M.flag(Dd + ".blocks.0.diffusionNoResidual") };
  for (size_t b = 0; b < d.blocks.size(); ++b)
    crossAttentionBlock(act, st, d.blocks[b], sh, C, d.heads, d.D, Dd + ".blocks." + std::to_string(b));
  scaleByRowK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, enc.qMask, qRows, C);
  float* ln = scratch<float>("dec.ln", qRows * C);
  layerNormSlow(act, ln, qRows, C, W(Dd + ".atomFeaturesLayerNormScale"), Wopt(Dd + ".atomFeaturesLayerNormOffset"));
  float* upd = scratch<float>("dec.upd", qRows * 3);
  linear<float, float>(ln, upd, qRows, C, 3, Dd + ".atomFeaturesToPositionUpdate");
  float* out = scratch<float>("dec.out", atoms * 3);
  convert(q2t, upd, out, 3);
  return out;
}

// ---------------------------------------------------------------- one denoiser call
// EDM's scalings off the device noise level: skip, out and input
__device__ inline void scalings(const float* params, float& skip, float& out, float& in) {
  float s = params[0], d = s * s + 256.f;
  skip = 256.f / d; out = s * 16.f * rsqrtf(d); in = rsqrtf(d);
}
__global__ void scalePositionsK(const float* x, const float* mask, float* y, size_t atoms, const float* params) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= atoms * 3) return;
  float skip, out, in; scalings(params, skip, out, in);
  y[t] = x[t] * mask[t / 3] * in;
}
__global__ void denoiseOutK(const float* x, const float* upd, const float* mask, float* o, size_t atoms,
                            const float* params) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= atoms * 3) return;
  float skip, out, in; scalings(params, skip, out, in);
  o[t] = (skip * x[t] + out * upd[t]) * mask[t / 3];
}

// Everything the denoiser computes once per fold.
struct DiffusionFold {
  const float *trunkSingle, *trunkPair, *targetFeat, *seqMask;
  int n;
  EncoderOut enc; DecoderCache dec;
  cudaGraphExec_t graph = nullptr; const float* graphInput = nullptr; int calls = 0;
};
inline bool GRAPHS = true;
inline DiffusionFold prepareDiffusion(const float* trunkSingle, const float* trunkPair, const float* targetFeat,
                                      const float* seqMask, int n) {
  DiffusionFold f{ trunkSingle, trunkPair, targetFeat, seqMask, n, {}, {} };
  DCACHE.ready = false;
  setNoise(SIGMA_DATA);
  Conditioning cond = diffusionConditioning(trunkSingle, trunkPair, targetFeat, SIGMA_DATA, n);  // builds the pair
  f.enc = prepareEncoder("diffusion.encoder", "atomReference", trunkSingle, cond.pair);
  f.dec = prepareDecoder(f.enc);
  prepareTransformer(cond.pair, n);
  return f;
}
// D(x; sigma): positions [tokens*dense][3] in, the denoised positions out.
inline float* denoiseCore(DiffusionFold& f, const float* positionsNoisy, float noiseLevel) {
  int n = f.n, dense = (int)M.meta("batch.dense");
  size_t atoms = (size_t)n * dense;
  stage(nullptr);
  Conditioning cond = diffusionConditioning(f.trunkSingle, f.trunkPair, f.targetFeat, noiseLevel, n); stage("d.conditioning");
  const float* atomMask = Fdev("batch.refMask");
  float* scaled = scratch<float>("dn.scaled", atoms * 3);
  scalePositionsK<<<blocks(atoms * 3), 256, 0, STREAM>>>(positionsNoisy, atomMask, scaled, atoms, noiseParams);
  encoderStep("diffusion.encoder", f.enc, scaled); stage("d.encoder");
  int Cs = (int)M.meta("diffusion.seqChannels"), perToken = (int)M.meta("diffusion.perTokenChannels");
  float* sn = scratch<float>("dn.sn", (size_t)n * Cs);
  layerNormSlow(cond.single, sn, n, Cs, W("diffusion.singleCondEmbeddingNormScale"), Wopt("diffusion.singleCondEmbeddingNormOffset"));
  float* act = scratch<float>("dn.act", (size_t)n * perToken);
  CK(cudaMemcpyAsync(act, f.enc.tokenAct, (size_t)n * perToken * 4, cudaMemcpyDeviceToDevice, STREAM));
  linear<float, float>(sn, act, n, Cs, perToken, "diffusion.singleCondEmbeddingProjection", false, 1.f);
  if (DIFF_HALF) diffusionTransformer<half>(act, cond.single, f.seqMask, n);
  else diffusionTransformer<float>(act, cond.single, f.seqMask, n);
  stage("d.transformer");
  float* actn = scratch<float>("dn.actn", (size_t)n * perToken);
  layerNormSlow(act, actn, n, perToken, W("diffusion.outputNormScale"), Wopt("diffusion.outputNormOffset"));
  float* upd = atomDecoder(actn, f.enc, f.dec); stage("d.decoder");
  float* out = scratch<float>("dn.out", atoms * 3);
  denoiseOutK<<<blocks(atoms * 3), 256, 0, STREAM>>>(positionsNoisy, upd, atomMask, out, atoms, noiseParams);
  return out;
}
// One denoiser call. After a first (allocating) call the step is captured as a CUDA graph and
// replayed: about 300 small launches become one.
inline float* denoiseStep(DiffusionFold& f, const float* positionsNoisy, float noiseLevel) {
  setNoise(noiseLevel);
  if (!GRAPHS || STAGES || f.calls++ == 0) return denoiseCore(f, positionsNoisy, noiseLevel);
  if (!f.graph || positionsNoisy != f.graphInput) {
    if (f.graph) CK(cudaGraphExecDestroy(f.graph));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(STREAM, cudaStreamCaptureModeThreadLocal));
    denoiseCore(f, positionsNoisy, noiseLevel);
    CK(cudaStreamEndCapture(STREAM, &g));
    CK(cudaGraphInstantiate(&f.graph, g, 0));
    CK(cudaGraphDestroy(g));
    f.graphInput = positionsNoisy;
  }
  CK(cudaGraphLaunch(f.graph, STREAM));
  return scratch<float>("dn.out", (size_t)f.n * (int)M.meta("batch.dense") * 3);
}
inline float* denoise(const float* trunkSingle, const float* trunkPair, const float* targetFeat,
                      const float* seqMask, const float* positionsNoisy, float noiseLevel) {
  DiffusionFold f = prepareDiffusion(trunkSingle, trunkPair, targetFeat, seqMask, (int)M.meta("batch.tokens"));
  return denoiseStep(f, positionsNoisy, noiseLevel);
}
