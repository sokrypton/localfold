// AlphaFold 2's building blocks, in float32: the reference path every stage is checked against
// af3-any-model's JAX AF2 with (native/af2/oracle.py). Shared infrastructure - the exported model
// file, device weights, scratch, cuBLAS - is native/af3's.
#pragma once
#include "../../af3/src/common.cuh"
#include "../../af3/src/flash.cuh"

// --fast: TF32 GEMMs and the f16 tensor-core flash attention (native/af3/src/flash.cuh) where a head
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
__global__ void reluBiasK(float* y, const float* b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[t] = fmaxf(y[t] + b[t % C], 0.f);
}
__global__ void addK2(float* y, const float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += x[t];
}
inline void linearB(const float* X, const std::string& w, int block, float* Y, size_t rows, int in, int out,
                    bool relu = false) {
  gemm(X, P(w + "/weights", block), Y, rows, in, out);
  if (relu) reluBiasK<<<blocks(rows * out), 256, 0, STREAM>>>(Y, P(w + "/bias", block), rows, out);
  else addBiasK<<<blocks(rows * out), 256, 0, STREAM>>>(Y, P(w + "/bias", block), rows, out);
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
inline void layerNorm(const float* x, float* y, size_t rows, int C, const std::string& w, int block = -1) {
  layerNormK<<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(x, y, rows, C, P(w + "/scale", block), P(w + "/offset", block));
}

// [A, B, C] -> [B, A, C]
__global__ void swap01K(const float* x, float* y, int A, int B, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * B * C) return;
  int c = (int)(t % C); size_t r = t / C; int b = (int)(r % B), a = (int)(r / B);
  y[((size_t)b * A + a) * C + c] = x[t];
}
inline void swap01(const float* x, float* y, int A, int B, int C) {
  swap01K<<<blocks((size_t)A * B * C), 256, 0, STREAM>>>(x, y, A, B, C);
}

// ---------------------------------------------------------------- attention
// Gated multi-head attention over a batch of independent sequences, a warp a query, the keys split
// over its lanes with an online softmax each, merged at the end:
//   q, k, v, g: [Bt, n, H*D] (q not yet scaled; g the gate's logits, its bias added)
//   keyMask:    [Bt, n] or null  - 1e9 * (mask - 1) on each key, AF2's `bias`
//   pairBias:   [H, n, n] or null - AF2's `nonbatched_bias`, shared by every batch row
//   out:        [Bt, n, H*D] = softmax(q k^T / sqrt(D) + biases) v * sigmoid(g)
template <int D>
__global__ void attentionK(const float* q, const float* k, const float* v, const float* g, const float* keyMask,
                           const float* pairBias, float* out, int Bt, int n, int H, float scale) {
  size_t wid = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (wid >= (size_t)Bt * n * H) return;
  int h = (int)(wid % H); size_t bq = wid / H; int qi = (int)(bq % n); size_t b = bq / n;
  const int W = H * D;
  float qv[D];
  for (int d = 0; d < D; ++d) qv[d] = q[bq * W + h * D + d] * scale;
  float m = -INFINITY, l = 0, o[D] = {};
  for (int j = lane; j < n; j += 32) {
    const float* kr = k + (b * n + j) * W + h * D;
    float s = 0;
    for (int d = 0; d < D; ++d) s += qv[d] * kr[d];
    if (keyMask) s += 1e9f * (keyMask[b * n + j] - 1.f);
    if (pairBias) s += pairBias[((size_t)h * n + qi) * n + j];
    s = fminf(fmaxf(s, -1e8f), 1e8f);
    float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
    l = l * c + p;
    const float* vr = v + (b * n + j) * W + h * D;
    for (int d = 0; d < D; ++d) o[d] = o[d] * c + p * vr[d];
    m = mn;
  }
  float M_ = m;
  for (int off = 16; off; off >>= 1) M_ = fmaxf(M_, __shfl_xor_sync(~0u, M_, off));
  float c = m == -INFINITY ? 0.f : __expf(m - M_);
  l *= c;
  for (int off = 16; off; off >>= 1) l += __shfl_xor_sync(~0u, l, off);
  for (int d = 0; d < D; ++d) {
    float x = o[d] * c;
    for (int off = 16; off; off >>= 1) x += __shfl_xor_sync(~0u, x, off);
    if (lane == 0) {
      float gate = 1.f / (1.f + __expf(-g[bq * W + h * D + d]));
      out[bq * W + h * D + d] = x / l * gate;
    }
  }
}
// q, k, v, g [rows, W] f32 -> qkvg [rows][4W] f16 (the flash kernel's layout, roles in that order)
__global__ void packQkvgK(const float* q, const float* k, const float* v, const float* g, half* out, size_t rows, int W) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * W) return;
  size_t r = t / W; int w = (int)(t % W);
  half* o = out + r * 4 * W;
  o[w] = __float2half(q[t]); o[W + w] = __float2half(k[t]); o[2 * W + w] = __float2half(v[t]); o[3 * W + w] = __float2half(g[t]);
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
                      const float* pairBias, float* out, int Bt, int n, int H) {
  if (FAST && (D == 16 || D == 32)) {
    size_t rows = (size_t)Bt * n; int W = H * D;
    half* qkvg = scratch<half>("fa.qkvg", (rows + 128) * 4 * W);
    packQkvgK<<<blocks(rows * W), 256, 0, STREAM>>>(q, k, v, g, qkvg, rows, W);
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
  size_t warps = (size_t)Bt * n * H;
  unsigned grid = (unsigned)((warps + 7) / 8);
  float scale = 1.f / sqrtf((float)D);
  if (D == 32) attentionK<32><<<grid, 256, 0, STREAM>>>(q, k, v, g, keyMask, pairBias, out, Bt, n, H, scale);
  else if (D == 8) attentionK<8><<<grid, 256, 0, STREAM>>>(q, k, v, g, keyMask, pairBias, out, Bt, n, H, scale);
  else if (D == 16) attentionK<16><<<grid, 256, 0, STREAM>>>(q, k, v, g, keyMask, pairBias, out, Bt, n, H, scale);
  else { fprintf(stderr, "attention: no kernel for head width %d\n", D); exit(1); }
}
// add the gate's bias [H*D] to g, a [rows, H*D] block with row stride `ld`
__global__ void addStridedBiasK(float* g, const float* b, size_t rows, int W, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * W) g[(t / W) * ld + t % W] += b[t % W];
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
// out[b, s, h*D + d] = avg[b, h, d] * sigmoid(gate[b, s, h*D + d])
__global__ void globalGateK(const float* avg, const float* gate, float* out, int Bt, int n, int W) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)Bt * n * W) return;
  int w = (int)(t % W); size_t b = t / W / n;
  out[t] = avg[b * W + w] / (1.f + __expf(-gate[t]));
}

// ---------------------------------------------------------------- outer product mean
// P[(i, c), (j, e)] (rows i0.., a block of bi of them) -> X[(i, j), (c, e)]
__global__ void opmPermuteK2(const float* Pm, float* X, int bi, int L, int O) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * L * O * O) return;
  int e = (int)(t % O); size_t r = t / O; int c = (int)(r % O); r /= O; int j = (int)(r % L), i = (int)(r / L);
  X[t] = Pm[((size_t)i * O + c) * ((size_t)L * O) + (size_t)j * O + e];
}
// pair[(i0+i), j] += (X[i, j] + b) / (1e-3 + norm[i0+i, j])
__global__ void opmAddK2(float* pair, const float* X, const float* b, const float* norm, int i0, int bi, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * L * C) return;
  int f = (int)(t % C); size_t ij = t / C; size_t i = i0 + ij / L, j = ij % L;
  pair[(i * L + j) * C + f] += (X[t] + b[f]) / (1e-3f + norm[i * L + j]);
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
