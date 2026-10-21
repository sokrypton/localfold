// AlphaFold 2's building blocks, in float32: the reference path every stage is checked against
// af3-any-model's JAX AF2 with (cuda/af2/oracle.py). Shared infrastructure - the exported model
// file, device weights, scratch, cuBLAS - is cuda/af3's.
#pragma once
#include "../../af3/src/common.cuh"
#include "../../af3/src/flash.cuh"

// --fast: TF32 GEMMs and the f16 tensor-core flash attention (cuda/af3/src/flash.cuh) where a head
// is 16 or 32 wide; residual streams and everything else stay f32
inline bool FAST = false;

// ---------------------------------------------------------------- weights
// A haiku parameter, "w/<module>/<param>"; `block` picks one slice of a layer-stacked one.
inline size_t dimW(const std::string& name, int k) { return (size_t)M.meta("w/" + name + "#" + std::to_string(k)); }
inline const float* P(const std::string& name, int block = -1) {
  const float* base = W("w/" + name);
  if (block < 0) return base;
  size_t per = M.len("w/" + name) / dimW(name, 0);
  return base + per * block;
}

// ---------------------------------------------------------------- GEMM, row-major
// Y[rows, out] = X[rows, in] Wt[in, out] (+ beta Y); f32 accumulate, no TF32 on this path
inline void gemm(const float* X, const float* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f) {
  const float one = 1.f;
  CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one, Wt, out, X, in, &beta, Y, out));
}

// ---------------------------------------------------------------- elementwise
__global__ void reluBiasK(float* __restrict__ y, const float* __restrict__ b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    int c = ((c4 & (c4 - 1)) == 0) ? (int)(t & (size_t)(c4 - 1)) : (n4 <= UINT32_MAX ? (int)((unsigned)t % (unsigned)c4) : (int)(t % c4));
    float4 yv = reinterpret_cast<const float4*>(y)[t];
    float4 bv = __ldg(reinterpret_cast<const float4*>(b) + c);
    reinterpret_cast<float4*>(y)[t] = make_float4(fmaxf(yv.x + bv.x, 0.f), fmaxf(yv.y + bv.y, 0.f), fmaxf(yv.z + bv.z, 0.f), fmaxf(yv.w + bv.w, 0.f));
    return;
  }
  size_t n = rows * C;
  if (t >= n) return;
  int c = ((C & (C - 1)) == 0) ? (int)(t & (size_t)(C - 1)) : (n <= UINT32_MAX ? (int)((unsigned)t % (unsigned)C) : (int)(t % C));
  y[t] = fmaxf(y[t] + __ldg(b + c), 0.f);
}
__global__ void addK2(float* __restrict__ y, const float* __restrict__ x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)y | (uintptr_t)x) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 yv = reinterpret_cast<const float4*>(y)[t];
      float4 xv = reinterpret_cast<const float4*>(x)[t];
      reinterpret_cast<float4*>(y)[t] = make_float4(yv.x + xv.x, yv.y + xv.y, yv.z + xv.z, yv.w + xv.w);
    }
    return;
  }
  if (t < n) y[t] += x[t];
}
// y[(b * A + a), c] += x[swap ? (a * B + b) : (b * A + a), c] + bias[c]
__global__ void addBiasSwapIntoK(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ b,
                                 int B, int A, int C, bool swap) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t rows = (size_t)B * A;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)x | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    size_t r; int c;
    if ((c4 & (c4 - 1)) == 0 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, uc4 = (unsigned)c4;
      c = (int)(u & (uc4 - 1));
      r = u >> (__ffs(c4) - 1);
    } else if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, ru = u / (unsigned)c4;
      r = ru; c = (int)(u - ru * (unsigned)c4);
    } else {
      r = t / c4; c = (int)(t % c4);
    }
    size_t sr = r;
    if (swap) {
      if (rows <= UINT32_MAX) {
        unsigned ru = (unsigned)r, bu = ru / (unsigned)A, au = ru - bu * (unsigned)A;
        sr = (size_t)au * B + bu;
      } else {
        sr = (r % A) * B + (r / A);
      }
    }
    float4 yv = reinterpret_cast<const float4*>(y)[t];
    float4 xv = __ldg(reinterpret_cast<const float4*>(x) + sr * c4 + c);
    float4 bv = __ldg(reinterpret_cast<const float4*>(b) + c);
    reinterpret_cast<float4*>(y)[t] = make_float4(yv.x + (xv.x + bv.x), yv.y + (xv.y + bv.y),
                                                  yv.z + (xv.z + bv.z), yv.w + (xv.w + bv.w));
    return;
  }
  size_t n = rows * C;
  if (t >= n) return;
  size_t r = t / C; int c = (int)(t % C);
  size_t sr = swap ? ((r % A) * B + (r / A)) : r;
  y[t] += x[sr * C + c] + __ldg(b + c);
}
inline void addBiasSwapInto(float* y, const float* x, const float* b, int B, int A, int C, bool swap = false) {
  size_t rows = (size_t)B * A;
  size_t work = ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)x | (uintptr_t)b) & 15) == 0) ? rows * (size_t)(C >> 2) : rows * C;
  addBiasSwapIntoK<<<blocks(work), 256, 0, STREAM>>>(y, x, b, B, A, C, swap);
}
__global__ void addBiasScaleRowsK(float* __restrict__ y, const float* __restrict__ b,
                                  const float* __restrict__ rowScale, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    size_t r; int c;
    if ((c4 & (c4 - 1)) == 0) {
      int sh = __ffs(c4) - 1;
      r = t >> sh;
      c = (int)(t & (size_t)(c4 - 1));
    } else if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, uc4 = (unsigned)c4, ru = u / uc4;
      r = ru; c = (int)(u - ru * uc4);
    } else {
      r = t / (size_t)c4; c = (int)(t % (size_t)c4);
    }
    float4 yv = reinterpret_cast<const float4*>(y)[t];
    float4 bv = __ldg(reinterpret_cast<const float4*>(b) + c);
    float s = __ldg(rowScale + r);
    reinterpret_cast<float4*>(y)[t] = make_float4((yv.x + bv.x) * s, (yv.y + bv.y) * s, (yv.z + bv.z) * s, (yv.w + bv.w) * s);
    return;
  }
  size_t n = rows * (size_t)C;
  if (t >= n) return;
  size_t r = t / (size_t)C; int c = (int)(t % (size_t)C);
  y[t] = (y[t] + __ldg(b + c)) * __ldg(rowScale + r);
}
inline void linearB(const float* X, const std::string& w, int block, float* Y, size_t rows, int in, int out,
                    bool relu = false, const float* rowScale = nullptr) {
  gemm(X, P(w + "/weights", block), Y, rows, in, out);
  const float* b = P(w + "/bias", block);
  size_t work = ((out & 3) == 0 && (((uintptr_t)Y | (uintptr_t)b) & 15) == 0) ? rows * (size_t)(out >> 2) : rows * (size_t)out;
  if (relu) reluBiasK<<<blocks(work), 256, 0, STREAM>>>(Y, b, rows, out);
  else if (rowScale) addBiasScaleRowsK<<<blocks(work), 256, 0, STREAM>>>(Y, b, rowScale, rows, out);
  else addBiasK<<<blocks(work), 256, 0, STREAM>>>(Y, b, rows, out);
}

// ---------------------------------------------------------------- LayerNorm
// haiku's: eps 1e-5, the variance as the mean squared deviation (two passes), a warp a row
__global__ void layerNormK(const float* x, float* y, size_t rows, int C, const float* scale, const float* offset) {
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
  float inv = rsqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) y[row * C + c] = (xr[c] - mean) * inv * scale[c] + offset[c];
}
template <int C, class TI = float, bool RELU = false>
__global__ void __launch_bounds__(256, C >= 384 ? 4 : 8)
layerNormFastK(const TI* __restrict__ x, float* __restrict__ y, size_t rows,
               const float* __restrict__ scale, const float* __restrict__ offset) {
  constexpr int K = C / 32;
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const TI* xr = x + row * C;
  float v[K], s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { float xk = (float)__ldg(xr + lane + 32 * k); v[k] = xk; s += xk; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, q2 = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { float d = v[k] - mean; q2 += d * d; }
  for (int o = 16; o; o >>= 1) q2 += __shfl_xor_sync(~0u, q2, o);
  float inv = rsqrtf(q2 / C + 1e-5f);
  float* yr = y + row * C;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    int c = lane + 32 * k;
    float out = (v[k] - mean) * inv * __ldg(scale + c) + __ldg(offset + c);
    yr[c] = RELU ? fmaxf(out, 0.f) : out;
  }
}
inline void layerNorm(const float* x, float* y, size_t rows, int C, const std::string& w, int block = -1) {
  const float *sc = P(w + "/scale", block), *of = P(w + "/offset", block);
  unsigned g = (unsigned)((rows + 7) / 8);
  if (C == 64) layerNormFastK<64><<<g, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else if (C == 128) layerNormFastK<128><<<g, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else if (C == 256) layerNormFastK<256><<<g, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else if (C == 384) layerNormFastK<384><<<g, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else layerNormK<<<g, 256, 0, STREAM>>>(x, y, rows, C, sc, of);
}

// Channel-major [C, pairs] -> row-major [pairs, C] + LayerNorm in one pass
template <int C>
__global__ void __launch_bounds__(256, 6)
centerNormF32K(const float* __restrict__ prod, float* __restrict__ out, size_t pairs,
               const float* __restrict__ scale, const float* __restrict__ offset) {
  __shared__ float tile[C][33];
  size_t r0 = (size_t)blockIdx.x * 32;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  size_t r = r0 + lane;
  if (r < pairs) {
#pragma unroll
    for (int c = warp; c < C; c += 8) tile[c][lane] = __ldg(prod + (size_t)c * pairs + r);
  }
  __syncthreads();
  constexpr int K = C / 32;
  float sc[K], of[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    int c = lane + 32 * k;
    sc[k] = __ldg(scale + c);
    of[k] = __ldg(offset + c);
  }
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    int row = warp + m * 8;
    size_t rm = r0 + row;
    if (rm >= pairs) break;
    float v[K], s = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) { float xk = tile[lane + 32 * k][row]; v[k] = xk; s += xk; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, q2 = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) { float d = v[k] - mean; q2 += d * d; }
    for (int o = 16; o; o >>= 1) q2 += __shfl_xor_sync(~0u, q2, o);
    float inv = rsqrtf(q2 / C + 1e-5f);
    float* yr = out + rm * C;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      int c = lane + 32 * k;
      yr[c] = (v[k] - mean) * inv * sc[k] + of[k];
    }
  }
}
inline void centerNormF32(const float* prod, float* cn, size_t pairs, int C, const std::string& w, int blk) {
  const float *sc = P(w + "/scale", blk), *of = P(w + "/offset", blk);
  unsigned g = (unsigned)((pairs + 31) / 32);
  if (C == 64) centerNormF32K<64><<<g, 256, 0, STREAM>>>(prod, cn, pairs, sc, of);
  else if (C == 128) centerNormF32K<128><<<g, 256, 0, STREAM>>>(prod, cn, pairs, sc, of);
  else {
    float* rowsP = scratch<float>("tri.rows", pairs * C);
    channelMajorToRowsTileK<<<dim3(g, (unsigned)((C + 31) / 32)), dim3(32, 8), 0, STREAM>>>(prod, rowsP, pairs, C);
    layerNorm(rowsP, cn, pairs, C, w, blk);
  }
}

// [A, B, C] -> [B, A, C]
__global__ void swap01K(const float* __restrict__ x, float* __restrict__ y, int A, int B, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0) {
    int c4 = C >> 2;
    size_t total4 = (size_t)A * B * c4;
    if (t >= total4) return;
    int c, b, a;
    if ((c4 & (c4 - 1)) == 0 && total4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, uc4 = (unsigned)c4;
      c = (int)(u & (uc4 - 1));
      unsigned r = u >> (__ffs(c4) - 1);
      a = (int)(r / (unsigned)B);
      b = (int)(r - (unsigned)a * (unsigned)B);
    } else if (total4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, r = u / (unsigned)c4;
      c = (int)(u - r * (unsigned)c4);
      a = (int)(r / (unsigned)B);
      b = (int)(r - (unsigned)a * (unsigned)B);
    } else {
      c = (int)(t % c4); size_t r = t / c4; b = (int)(r % B); a = (int)(r / B);
    }
    reinterpret_cast<float4*>(y)[((size_t)b * A + a) * c4 + c] = reinterpret_cast<const float4*>(x)[t];
    return;
  }
  size_t total = (size_t)A * B * C;
  if (t >= total) return;
  if (total <= UINT32_MAX) {
    unsigned u = (unsigned)t, r = C == 1 ? u : u / (unsigned)C;
    int c = C == 1 ? 0 : (int)(u - r * (unsigned)C);
    unsigned a = r / (unsigned)B, b = r - a * (unsigned)B;
    y[((size_t)b * A + a) * C + c] = x[t];
    return;
  }
  int c = (int)(t % C); size_t r = t / C; int b = (int)(r % B), a = (int)(r / B);
  y[((size_t)b * A + a) * C + c] = x[t];
}
inline void swap01(const float* x, float* y, int A, int B, int C) {
  size_t work = (C & 3) == 0 ? (size_t)A * B * (C >> 2) : (size_t)A * B * C;
  swap01K<<<blocks(work), 256, 0, STREAM>>>(x, y, A, B, C);
}

// ---------------------------------------------------------------- attention
// Gated multi-head attention over a batch of independent sequences, a warp a query, the keys split
// over its lanes with an online softmax each, merged at the end:
//   q, k, v, g: [Bt, n, H*D] (q not yet scaled; g the gate's logits, its bias added)
//   keyMask:    [Bt, n] or null  - 1e9 * (mask - 1) on each key, AF2's `bias`
//   pairBias:   [H, n, n] or null - AF2's `nonbatched_bias`, shared by every batch row
//   out:        [Bt, n, H*D] = softmax(q k^T / sqrt(D) + biases) v * sigmoid(g)
template <int D>
__global__ void __launch_bounds__(256, 4)
attentionK(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
           const float* __restrict__ g, const float* __restrict__ gBias, const float* __restrict__ keyMask,
           const float* __restrict__ pairBias, float* __restrict__ out, int Bt, int n, int H, float scale) {
  constexpr int LDK = (D + 4) & ~3;
  __shared__ __align__(16) float Qs[8][LDK];
  __shared__ __align__(16) float Ks[32][LDK];
  __shared__ __align__(16) float Vs[32][LDK];
  __shared__ float Ms[32];
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  unsigned nq = (unsigned)((n + 7) / 8);
  size_t b = blockIdx.x / nq;
  int qi0 = (int)(blockIdx.x - b * nq) * 8;
  int qi = qi0 + warp;
  int h = (int)blockIdx.y;
  const int W = H * D;
  const bool validQ = (b < (size_t)Bt && qi < n);
  size_t bq = b * (size_t)n + (size_t)qi;
  if (validQ && lane < D) {
    Qs[warp][lane] = __ldg(q + bq * W + h * D + lane) * scale;
  }
  float m = -INFINITY, l = 0.f, o[D] = {};
  const float* pbRow = (validQ && pairBias) ? (pairBias + ((size_t)h * n + qi) * n) : nullptr;
  constexpr int D4 = D / 4;
  constexpr int THREADS_FOR_KV = 32 * D4;
  const float* kBase = k + b * (size_t)n * W + h * D;
  const float* vBase = v + b * (size_t)n * W + h * D;
  const float* mBase = keyMask ? (keyMask + b * (size_t)n) : nullptr;
  if constexpr ((D & 3) == 0 && D <= 16) {
    const int jFull = n & ~31;
    int kj = tid / D4, d4 = (tid % D4) * 4;
    float4 k4_reg = make_float4(0.f, 0.f, 0.f, 0.f), v4_reg = make_float4(0.f, 0.f, 0.f, 0.f);
    float ms_reg = 0.f, pb_reg = 0.f;
    if (jFull > 0) {
      if (tid < THREADS_FOR_KV) {
        size_t ko = (size_t)kj * W + d4;
        k4_reg = __ldg(reinterpret_cast<const float4*>(kBase + ko));
        v4_reg = __ldg(reinterpret_cast<const float4*>(vBase + ko));
      }
      if (mBase && tid < 32) ms_reg = 1e9f * (__ldg(mBase + tid) - 1.f);
      if (pbRow) pb_reg = __ldg(pbRow + lane);
    }
    for (int j0 = 0; j0 < jFull; j0 += 32) {
      if (tid < THREADS_FOR_KV) {
        *reinterpret_cast<float4*>(&Ks[kj][d4]) = k4_reg;
        *reinterpret_cast<float4*>(&Vs[kj][d4]) = v4_reg;
      }
      if (mBase && tid < 32) Ms[tid] = ms_reg;
      float pbj = pb_reg;
      __syncthreads();
      int jNext = j0 + 32;
      if (jNext < jFull) {
        if (tid < THREADS_FOR_KV) {
          size_t ko = (size_t)(jNext + kj) * W + d4;
          k4_reg = __ldg(reinterpret_cast<const float4*>(kBase + ko));
          v4_reg = __ldg(reinterpret_cast<const float4*>(vBase + ko));
        }
        if (mBase && tid < 32) ms_reg = 1e9f * (__ldg(mBase + jNext + tid) - 1.f);
        if (pbRow) pb_reg = __ldg(pbRow + jNext + lane);
      }
      if (validQ) {
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < D; d += 4) {
          float4 q4 = *reinterpret_cast<const float4*>(&Qs[warp][d]);
          float4 kv = *reinterpret_cast<const float4*>(&Ks[lane][d]);
          s += q4.x * kv.x; s += q4.y * kv.y; s += q4.z * kv.z; s += q4.w * kv.w;
        }
        if (mBase) s += Ms[lane];
        if (pbRow) s += pbj;
        s = fminf(fmaxf(s, -1e8f), 1e8f);
        float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
        l = l * c + p;
#pragma unroll
        for (int d = 0; d < D; d += 4) {
          float4 vv = *reinterpret_cast<const float4*>(&Vs[lane][d]);
          o[d]     = o[d]     * c + p * vv.x;
          o[d + 1] = o[d + 1] * c + p * vv.y;
          o[d + 2] = o[d + 2] * c + p * vv.z;
          o[d + 3] = o[d + 3] * c + p * vv.w;
        }
        m = mn;
      }
      __syncthreads();
    }
    if (jFull < n) {
      int j0 = jFull;
      if (tid < THREADS_FOR_KV && j0 + kj < n) {
        size_t ko = (size_t)(j0 + kj) * W + d4;
        *reinterpret_cast<float4*>(&Ks[kj][d4]) = __ldg(reinterpret_cast<const float4*>(kBase + ko));
        *reinterpret_cast<float4*>(&Vs[kj][d4]) = __ldg(reinterpret_cast<const float4*>(vBase + ko));
      }
      if (mBase && tid < 32 && j0 + tid < n) Ms[tid] = 1e9f * (__ldg(mBase + j0 + tid) - 1.f);
      int j = j0 + lane;
      float pbj = (pbRow && j < n) ? __ldg(pbRow + j) : 0.f;
      __syncthreads();
      if (validQ && j < n) {
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < D; d += 4) {
          float4 q4 = *reinterpret_cast<const float4*>(&Qs[warp][d]);
          float4 kv = *reinterpret_cast<const float4*>(&Ks[lane][d]);
          s += q4.x * kv.x; s += q4.y * kv.y; s += q4.z * kv.z; s += q4.w * kv.w;
        }
        if (mBase) s += Ms[lane];
        if (pbRow) s += pbj;
        s = fminf(fmaxf(s, -1e8f), 1e8f);
        float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
        l = l * c + p;
#pragma unroll
        for (int d = 0; d < D; d += 4) {
          float4 vv = *reinterpret_cast<const float4*>(&Vs[lane][d]);
          o[d]     = o[d]     * c + p * vv.x;
          o[d + 1] = o[d + 1] * c + p * vv.y;
          o[d + 2] = o[d + 2] * c + p * vv.z;
          o[d + 3] = o[d + 3] * c + p * vv.w;
        }
        m = mn;
      }
    }
  } else {
    for (int j0 = 0; j0 < n; j0 += 32) {
      if (j0 + 32 <= n) {
        if constexpr ((D & 3) == 0 && THREADS_FOR_KV <= 256) {
          if (tid < THREADS_FOR_KV) {
            int kj = tid / D4, d4 = (tid % D4) * 4;
            size_t ko = (size_t)(j0 + kj) * W + d4;
            *reinterpret_cast<float4*>(&Ks[kj][d4]) = __ldg(reinterpret_cast<const float4*>(kBase + ko));
            *reinterpret_cast<float4*>(&Vs[kj][d4]) = __ldg(reinterpret_cast<const float4*>(vBase + ko));
          }
        } else {
          for (int idx = tid; idx < 32 * D; idx += 256) {
            int kj = idx / D, d = idx % D;
            size_t ko = (size_t)(j0 + kj) * W + d;
            Ks[kj][d] = __ldg(kBase + ko);
            Vs[kj][d] = __ldg(vBase + ko);
          }
        }
        if (mBase && tid < 32) Ms[tid] = 1e9f * (__ldg(mBase + j0 + tid) - 1.f);
        __syncthreads();
        if (validQ) {
          int j = j0 + lane;
          float s = 0.f;
          if constexpr ((D & 3) == 0) {
#pragma unroll
            for (int d = 0; d < D; d += 4) {
              float4 q4 = *reinterpret_cast<const float4*>(&Qs[warp][d]);
              float4 kv = *reinterpret_cast<const float4*>(&Ks[lane][d]);
              s += q4.x * kv.x; s += q4.y * kv.y; s += q4.z * kv.z; s += q4.w * kv.w;
            }
          } else {
#pragma unroll
            for (int d = 0; d < D; ++d) s += Qs[warp][d] * Ks[lane][d];
          }
          if (mBase) s += Ms[lane];
          if (pbRow) s += __ldg(pbRow + j);
          s = fminf(fmaxf(s, -1e8f), 1e8f);
          float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
          l = l * c + p;
          if constexpr ((D & 3) == 0) {
#pragma unroll
            for (int d = 0; d < D; d += 4) {
              float4 vv = *reinterpret_cast<const float4*>(&Vs[lane][d]);
              o[d]     = o[d]     * c + p * vv.x;
              o[d + 1] = o[d + 1] * c + p * vv.y;
              o[d + 2] = o[d + 2] * c + p * vv.z;
              o[d + 3] = o[d + 3] * c + p * vv.w;
            }
          } else {
#pragma unroll
            for (int d = 0; d < D; ++d) o[d] = o[d] * c + p * Vs[lane][d];
          }
          m = mn;
        }
        __syncthreads();
      } else {
        if constexpr ((D & 3) == 0 && THREADS_FOR_KV <= 256) {
          if (tid < THREADS_FOR_KV) {
            int kj = tid / D4, d4 = (tid % D4) * 4;
            if (j0 + kj < n) {
              size_t ko = (size_t)(j0 + kj) * W + d4;
              *reinterpret_cast<float4*>(&Ks[kj][d4]) = __ldg(reinterpret_cast<const float4*>(kBase + ko));
              *reinterpret_cast<float4*>(&Vs[kj][d4]) = __ldg(reinterpret_cast<const float4*>(vBase + ko));
            }
          }
        } else {
          for (int idx = tid; idx < 32 * D; idx += 256) {
            int kj = idx / D, d = idx % D;
            if (j0 + kj < n) {
              size_t ko = (size_t)(j0 + kj) * W + d;
              Ks[kj][d] = __ldg(kBase + ko);
              Vs[kj][d] = __ldg(vBase + ko);
            }
          }
        }
        if (mBase && tid < 32 && j0 + tid < n) Ms[tid] = 1e9f * (__ldg(mBase + j0 + tid) - 1.f);
        __syncthreads();
        int j = j0 + lane;
        if (validQ && j < n) {
          float s = 0.f;
          if constexpr ((D & 3) == 0) {
#pragma unroll
            for (int d = 0; d < D; d += 4) {
              float4 q4 = *reinterpret_cast<const float4*>(&Qs[warp][d]);
              float4 kv = *reinterpret_cast<const float4*>(&Ks[lane][d]);
              s += q4.x * kv.x; s += q4.y * kv.y; s += q4.z * kv.z; s += q4.w * kv.w;
            }
          } else {
#pragma unroll
            for (int d = 0; d < D; ++d) s += Qs[warp][d] * Ks[lane][d];
          }
          if (mBase) s += Ms[lane];
          if (pbRow) s += __ldg(pbRow + j);
          s = fminf(fmaxf(s, -1e8f), 1e8f);
          float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
          l = l * c + p;
          if constexpr ((D & 3) == 0) {
#pragma unroll
            for (int d = 0; d < D; d += 4) {
              float4 vv = *reinterpret_cast<const float4*>(&Vs[lane][d]);
              o[d]     = o[d]     * c + p * vv.x;
              o[d + 1] = o[d + 1] * c + p * vv.y;
              o[d + 2] = o[d + 2] * c + p * vv.z;
              o[d + 3] = o[d + 3] * c + p * vv.w;
            }
          } else {
#pragma unroll
            for (int d = 0; d < D; ++d) o[d] = o[d] * c + p * Vs[lane][d];
          }
          m = mn;
        }
        __syncthreads();
      }
    }
  }
  if (!validQ) return;
  float M_ = m;
#pragma unroll
  for (int off = 16; off; off >>= 1) M_ = fmaxf(M_, __shfl_xor_sync(~0u, M_, off));
  float c = m == -INFINITY ? 0.f : __expf(m - M_);
  l *= c;
#pragma unroll
  for (int off = 16; off; off >>= 1) l += __shfl_xor_sync(~0u, l, off);
  l = __shfl_sync(~0u, l, 0);
  float x_lane = 0.f;
#pragma unroll
  for (int d = 0; d < D; ++d) {
    float x = o[d] * c;
#pragma unroll
    for (int off = 16; off; off >>= 1) x += __shfl_xor_sync(~0u, x, off);
    float x0 = __shfl_sync(~0u, x, 0);
    if (lane == d) x_lane = x0;
  }
  if (lane < D) {
    float gv = __ldg(g + bq * W + h * D + lane) + (gBias ? __ldg(gBias + h * D + lane) : 0.f);
    float gate = 1.f / (1.f + __expf(-gv));
    out[bq * W + h * D + lane] = x_lane / l * gate;
  }
}
// q, k, v, g [rows, W] f32 -> qkvg [rows][4W] f16 (the flash kernel's layout, roles in that order)
__global__ void packQkvgK(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                          const float* __restrict__ g, const float* __restrict__ gBias, half* __restrict__ out, size_t rows, int W) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * W) return;
  size_t r = t / W; int w = (int)(t % W);
  half* o = out + r * 4 * W;
  float gv = g[t] + (gBias ? __ldg(gBias + w) : 0.f);
  o[w] = __float2half(q[t]); o[W + w] = __float2half(k[t]); o[2 * W + w] = __float2half(v[t]); o[3 * W + w] = __float2half(gv);
}
// pair bias [H, n, n] f32 -> [H, n, stride] f16 times log2(e), as the flash kernel takes it (it scores in
// the log2 domain: Q carries scale * log2e, the bias log2e), -1e9 masks held to f16's range
__global__ void biasToHalfK(const float* b, half* out, int H, int n, int stride) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)H * n * n) return;
  size_t hq = t / n; int j = (int)(t % n);
  out[hq * stride + j] = __float2half(fmaxf(b[t] * LOG2E, -6e4f));     // the kernel scores in log2 units
}
__global__ void halfToFloatK(const half* x, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = __half2float(x[t]);
}
inline void attention(int D, const float* q, const float* k, const float* v, const float* g, const float* keyMask,
                      const float* pairBias, float* out, int Bt, int n, int H, const float* gBias = nullptr) {
  if (FAST && (D == 16 || D == 32)) {
    size_t rows = (size_t)Bt * n; int W = H * D;
    half* qkvg = scratch<half>("fa.qkvg", (rows + 128) * 4 * W);
    packQkvgK<<<blocks(rows * W), 256, 0, STREAM>>>(q, k, v, g, gBias, qkvg, rows, W);
    int stride = (n + 7) / 8 * 8;
    half* bias = scratch<half>("fa.bias", (size_t)H * n * stride);
    if (pairBias && stride != n) CK(cudaMemsetAsync(bias, 0, (size_t)H * n * stride * 2, STREAM));   // pad columns: see pairBiasFast
    if (pairBias) biasToHalfK<<<blocks((size_t)H * n * n), 256, 0, STREAM>>>(pairBias, bias, H, n, stride);
    else CK(cudaMemsetAsync(bias, 0, (size_t)H * n * stride * 2, STREAM));
    half* o = scratch<half>("fa.out", rows * W);
    flashGrid<half>(qkvg, bias, stride, keyMask, o, n, H, D, 0, Bt, false, 1.f / sqrtf((float)D));
    halfToFloatK<<<blocks(rows * W), 256, 0, STREAM>>>(o, out, rows * W);
    return;
  }
  unsigned nq = (unsigned)((n + 7) / 8);
  dim3 grid((unsigned)((size_t)Bt * nq), (unsigned)H);
  float scale = 1.f / sqrtf((float)D);
  if (D == 32) attentionK<32><<<grid, 256, 0, STREAM>>>(q, k, v, g, gBias, keyMask, pairBias, out, Bt, n, H, scale);
  else if (D == 8) attentionK<8><<<grid, 256, 0, STREAM>>>(q, k, v, g, gBias, keyMask, pairBias, out, Bt, n, H, scale);
  else if (D == 16) attentionK<16><<<grid, 256, 0, STREAM>>>(q, k, v, g, gBias, keyMask, pairBias, out, Bt, n, H, scale);
  else { fprintf(stderr, "attention: no kernel for head width %d\n", D); exit(1); }
}
// add the gate's bias [H*D] to g, a [rows, H*D] block with row stride `ld`
__global__ void addStridedBiasK(float* __restrict__ g, const float* __restrict__ b, size_t rows, int W, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((W & 3) == 0 && (ld & 3) == 0 && (((uintptr_t)g | (uintptr_t)b) & 15) == 0) {
    int w4 = W >> 2, ld4 = ld >> 2;
    size_t n4 = rows * (size_t)w4;
    if (t < n4) {
      size_t r = (n4 <= UINT32_MAX) ? (unsigned)t / (unsigned)w4 : t / (size_t)w4;
      int c4 = (int)(t - r * (size_t)w4);
      float4 gv = reinterpret_cast<const float4*>(g)[r * ld4 + c4];
      float4 bv = __ldg(reinterpret_cast<const float4*>(b) + c4);
      reinterpret_cast<float4*>(g)[r * ld4 + c4] = make_float4(gv.x + bv.x, gv.y + bv.y, gv.z + bv.z, gv.w + bv.w);
    }
    return;
  }
  if (t < rows * W) g[(t / W) * ld + t % W] += __ldg(b + t % W);
}

// ---------------------------------------------------------------- global (column) attention
// AF2's MSAColumnGlobalAttention core for one column b (a block a column): the query the masked
// mean over the column's sequences, one key and value a sequence shared by every head.
//   x [Bt, n, C] (normalised), mask [Bt, n]; q_w [C, H, D], k_w [C, D], v_w [C, D];
//   avg [Bt, H, D] = softmax_k(qavg_h . k_k / sqrt(D) + 1e9 (mask - 1)) v_k
__global__ void globalAttentionK(const float* x, const float* mask, const float* qw, const float* kw, const float* vw,
                                 float* avg, int n, int C, int H, int D) {
  int b = blockIdx.x;
  extern __shared__ float sh[];
  float* qavg = sh;                 // C
  float* qh = qavg + C;             // H*D
  float* logits = qh + H * D;       // H*n
  float* kk = logits + H * n;       // n*D
  float* vv = kk + n * D;           // n*D
  const float* xb = x + (size_t)b * n * C;
  const float* mb = mask + (size_t)b * n;
  float msum = 0;
  for (int s = 0; s < n; ++s) msum += mb[s];
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float a = 0;
    for (int s = 0; s < n; ++s) a += mb[s] * xb[(size_t)s * C + c];
    qavg[c] = a / (msum + 1e-10f);
  }
  __syncthreads();
  for (int t = threadIdx.x; t < H * D; t += blockDim.x) {
    float a = 0;
    for (int c = 0; c < C; ++c) a += qavg[c] * qw[(size_t)c * H * D + t];
    qh[t] = a / sqrtf((float)D);
  }
  for (int t = threadIdx.x; t < n * D; t += blockDim.x) {
    int s = t / D, d = t % D;
    float a = 0, e = 0;
    for (int c = 0; c < C; ++c) { float xv = xb[(size_t)s * C + c]; a += xv * kw[c * D + d]; e += xv * vw[c * D + d]; }
    kk[t] = a; vv[t] = e;
  }
  __syncthreads();
  for (int t = threadIdx.x; t < H * n; t += blockDim.x) {
    int h = t / n, s = t % n;
    float a = 0;
    for (int d = 0; d < D; ++d) a += qh[h * D + d] * kk[s * D + d];
    logits[t] = a + 1e9f * (mb[s] - 1.f);
  }
  __syncthreads();
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float mx = -INFINITY;
    for (int s = 0; s < n; ++s) mx = fmaxf(mx, logits[h * n + s]);
    float sum = 0;
    for (int s = 0; s < n; ++s) { float e = __expf(logits[h * n + s] - mx); logits[h * n + s] = e; sum += e; }
    for (int d = 0; d < D; ++d) {
      float a = 0;
      for (int s = 0; s < n; ++s) a += logits[h * n + s] * vv[s * D + d];
      avg[((size_t)b * H + h) * D + d] = a / sum;
    }
  }
}
// The same with almost no shared memory (the keys, values and logits of a column above held whole: 99 KB at
// 1024 extra sequences, past a T4's 64 KB): each thread streams its keys - k and v from the input as it
// goes - with an online softmax per head, and the block merges the threads' states (warp shuffles, then
// the warps through shared memory)
template <int H, int D>
__global__ void __launch_bounds__(256) globalAttentionStreamK(const float* x, const float* mask, const float* qw,
    const float* kw, const float* vw, float* avg, int n, int C) {
  int b = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  extern __shared__ float sh[];
  float* qavg = sh;                     // C
  float* qh = qavg + C;                 // H*D
  float* part = qh + H * D;             // [8 warps][H][D + 2]
  const float* xb = x + (size_t)b * n * C;
  const float* mb = mask + (size_t)b * n;
  float msum = 0;
  for (int s = 0; s < n; ++s) msum += mb[s];
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float a = 0;
    for (int s = 0; s < n; ++s) a += mb[s] * xb[(size_t)s * C + c];
    qavg[c] = a / (msum + 1e-10f);
  }
  __syncthreads();
  for (int t = threadIdx.x; t < H * D; t += blockDim.x) {
    float a = 0;
    for (int c = 0; c < C; ++c) a += qavg[c] * qw[(size_t)c * H * D + t];
    qh[t] = a / sqrtf((float)D);
  }
  __syncthreads();
  float m[H], l[H], acc[H][D];
#pragma unroll
  for (int h = 0; h < H; ++h) { m[h] = -INFINITY; l[h] = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) acc[h][d] = 0.f; }
  for (int s = threadIdx.x; s < n; s += blockDim.x) {
    float k[D], v[D];
#pragma unroll
    for (int d = 0; d < D; ++d) { k[d] = 0.f; v[d] = 0.f; }
    for (int c = 0; c < C; ++c) {
      float xv = xb[(size_t)s * C + c];
#pragma unroll
      for (int d = 0; d < D; ++d) { k[d] += xv * kw[c * D + d]; v[d] += xv * vw[c * D + d]; }
    }
    float bias = 1e9f * (mb[s] - 1.f);
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float lg = bias;
#pragma unroll
      for (int d = 0; d < D; ++d) lg += qh[h * D + d] * k[d];
      float mn = fmaxf(m[h], lg), cs = __expf(m[h] - mn), e = __expf(lg - mn);
      l[h] = l[h] * cs + e;
#pragma unroll
      for (int d = 0; d < D; ++d) acc[h][d] = acc[h][d] * cs + e * v[d];
      m[h] = mn;
    }
  }
  // merge two states (an empty one, m = -inf, contributes nothing)
  auto merge = [&](int h, float m2, float l2, const float* a2) {
    float M = fmaxf(m[h], m2);
    if (M == -INFINITY) return;
    float ca = __expf(m[h] - M), cb = __expf(m2 - M);
    l[h] = l[h] * ca + l2 * cb;
#pragma unroll
    for (int d = 0; d < D; ++d) acc[h][d] = acc[h][d] * ca + a2[d] * cb;
    m[h] = M;
  };
  for (int o = 16; o; o >>= 1) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float a2[D];
      float m2 = __shfl_xor_sync(~0u, m[h], o), l2 = __shfl_xor_sync(~0u, l[h], o);
#pragma unroll
      for (int d = 0; d < D; ++d) a2[d] = __shfl_xor_sync(~0u, acc[h][d], o);
      merge(h, m2, l2, a2);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      float* p = part + (warp * H + h) * (D + 2);
      p[0] = m[h]; p[1] = l[h];
#pragma unroll
      for (int d = 0; d < D; ++d) p[2 + d] = acc[h][d];
    }
  }
  __syncthreads();
  for (int t = threadIdx.x; t < H * D; t += blockDim.x) {
    int h = t / D, d = t % D;
    float M = -INFINITY;
    for (int w = 0; w < 8; ++w) M = fmaxf(M, part[(w * H + h) * (D + 2)]);
    float L = 0.f, A = 0.f;
    for (int w = 0; w < 8; ++w) {
      const float* p = part + (w * H + h) * (D + 2);
      if (p[0] == -INFINITY) continue;
      float c = __expf(p[0] - M);
      L += p[1] * c; A += p[2 + d] * c;
    }
    avg[((size_t)b * H + h) * D + d] = A / L;
  }
}
// out[b, s, h*D + d] = avg[b, h, d] * sigmoid(gate[b, s, h*D + d] + gBias[h*D + d])
__global__ void globalGateK(const float* __restrict__ avg, const float* __restrict__ gate, float* __restrict__ out,
                            int Bt, int n, int W, const float* __restrict__ gBias = nullptr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((W & 3) == 0 && (((uintptr_t)avg | (uintptr_t)gate | (uintptr_t)out | (uintptr_t)gBias) & 15) == 0) {
    int w4 = W >> 2;
    size_t n4 = (size_t)Bt * n * w4;
    if (t < n4) {
      size_t r = (n4 <= UINT32_MAX) ? (unsigned)t / (unsigned)w4 : t / (size_t)w4;
      int c4 = (int)(t - r * (size_t)w4);
      size_t b = (r <= UINT32_MAX) ? (unsigned)r / (unsigned)n : r / (size_t)n;
      float4 av = __ldg(reinterpret_cast<const float4*>(avg) + b * w4 + c4);
      float4 gv = __ldg(reinterpret_cast<const float4*>(gate) + t);
      if (gBias) {
        float4 bv = __ldg(reinterpret_cast<const float4*>(gBias) + c4);
        gv.x += bv.x; gv.y += bv.y; gv.z += bv.z; gv.w += bv.w;
      }
      reinterpret_cast<float4*>(out)[t] = make_float4(
        av.x / (1.f + __expf(-gv.x)), av.y / (1.f + __expf(-gv.y)),
        av.z / (1.f + __expf(-gv.z)), av.w / (1.f + __expf(-gv.w)));
    }
    return;
  }
  if (t >= (size_t)Bt * n * W) return;
  int w = (int)(t % W); size_t b = t / W / n;
  float gv = gate[t] + (gBias ? __ldg(gBias + w) : 0.f);
  out[t] = avg[b * W + w] / (1.f + __expf(-gv));
}

__global__ void scaleRowsK2(float* x, const float* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] *= mask[t / C];
}

// ---------------------------------------------------------------- checks
// relRMS against an oracle tensor in the loaded model, if it is there
inline double checkOracle(const char* label, const float* d, size_t n, const std::string& oracle) {
  if (!M.has(oracle)) { printf("  %-28s (no oracle %s)\n", label, oracle.c_str()); return -1; }
  if (M.len(oracle) != n) { printf("  %-28s LENGTH %zu against the oracle's %zu\n", label, n, M.len(oracle)); return -1; }
  auto h = download(d, n);
  double r = relRms(h.data(), M.f(oracle), n);
  printf("  %-28s relRMS %.3e\n", label, r);
  return r;
}
