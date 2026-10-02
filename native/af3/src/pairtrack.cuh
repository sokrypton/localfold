// The pair track - triangle multiplication, grid ("triangle") attention, transition -
// shared by the pairformer, the MSA stack and the template stack. Transcribed from
// src/af3/trunk/pairformer-reference.js; generic in channels, heads and head width.
#pragma once
#include "common.cuh"

// ---------------------------------------------------------------- elementwise kernels
// LayerNorm over the last axis with AF3's fast variance E[x^2] - E[x]^2. One warp a row.
template <class TI, class TO>
__global__ void layerNormK(const TI* in, TO* out, size_t rows, int C, const float* scale,
                           const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const TI* x = in + row * C;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) { float v = toF(x[c]); s += v; ss += v * v; }
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
  float mean = s / C, inv = rsqrtf(ss / C - mean * mean + 1e-5f);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = fromF<TO>((toF(x[c]) - mean) * inv * scale[c] + offset[c]);
}
template <class TI, class TO>
void layerNorm(const TI* in, TO* out, size_t rows, int C, const std::string& w) {
  layerNormK<TI, TO><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(
    in, out, rows, C, W(w + "Scale"), W(w + "Offset"));
}
template <class TI, class TO>
void layerNorm2(const TI* in, TO* out, size_t rows, int C, const std::string& scale,
                const std::string& offset) {
  layerNormK<TI, TO><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(in, out, rows, C, W(scale), W(offset));
}
__global__ void addK(float* y, const float* x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += x[i];
}
template <class T>
__global__ void gatedAddK(float* pair, const T* proj, const T* gate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pair[i] += toF(proj[i]) * sigm(toF(gate[i]));
}
template <class T>
__global__ void swigluK(const T* wide, T* gated, size_t rows, int I) {
  if constexpr (std::is_same_v<T, half>) {
    if (I % 8 == 0) {                 // eight halves (16 bytes) a thread
      size_t t = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
      if (t >= rows * I) return;
      size_t r = t / I; int i = (int)(t % I);
      uint4 a = *(const uint4*)(wide + r * 2 * I + i), b = *(const uint4*)(wide + r * 2 * I + I + i), o;
      const half2* a2 = (const half2*)&a; const half2* b2 = (const half2*)&b; half2* o2 = (half2*)&o;
      for (int k = 0; k < 4; ++k) {
        float2 g = __half22float2(a2[k]), v = __half22float2(b2[k]);
        o2[k] = __floats2half2_rn(g.x * sigm(g.x) * v.x, g.y * sigm(g.y) * v.y);
      }
      *(uint4*)(gated + t) = o;
      return;
    }
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = (int)(t % I);
  float g = toF(wide[r * 2 * I + i]);
  gated[t] = fromF<T>(g * sigm(g) * toF(wide[r * 2 * I + I + i]));
}
// the launch: an eighth of the threads for the vector path
template <class T>
void swiglu(const T* wide, T* gated, size_t rows, int I) {
  size_t work = std::is_same_v<T, half> && I % 8 == 0 ? rows * I / 8 : rows * I;
  swigluK<T><<<blocks(work), 256, 0, STREAM>>>(wide, gated, rows, I);
}
__global__ void addBiasK(float* y, const float* b, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * C) y[i] += b[i % C];
}

// ---------------------------------------------------------------- fused weights
// The triangle's projection and gate as one (C, 4C) matrix: [projection (2C) | gate (2C)].
inline std::string projectionGate(const std::string& pre, int C) {
  return concatColumns(pre + ".projectionGate~", C, {{pre + ".projection", 2 * C, false}, {pre + ".gate", 2 * C, false}});
}
// q, k, v and the gate as one (C, 4W) matrix. The grid attention stores q, k and the gate
// (out, in) and v (in, out); `transposedQkg` says whether that holds.
inline std::string qkvgWeight(const std::string& pre, int C, int Wd, bool transposedQkg) {
  return concatColumns(pre + ".qkvg~", C, {{pre + ".qProjection", Wd, transposedQkg}, {pre + ".kProjection", Wd, transposedQkg},
                                          {pre + ".vProjection", Wd, false}, {pre + ".gatingQuery", Wd, transposedQkg}});
}

// ---------------------------------------------------------------- triangle multiplication
// a, b from [rows][4C] = [projection | gate], the INTERLEAVED split (channel ch's halves
// are 2ch and 2ch+1), masked, written channel-major (c, pairs) through shared memory.
template <class T>
__global__ void triGateK(const T* pg, const float* mask, T* a, T* b, size_t r0, size_t rows, int C,
                         size_t pairs) {
  __shared__ float A[32][33], B[32][33];
  size_t row0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = row0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (local < rows && c < C) {
      float m = mask[r0 + local];
      const T* p = pg + local * 4 * C; const T* g = p + 2 * C;
      va = toF(p[c * 2]) * m * sigm(toF(g[c * 2]));
      vb = toF(p[c * 2 + 1]) * m * sigm(toF(g[c * 2 + 1]));
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t local = row0 + tx; int c = c0 + cy;
    if (local < rows && c < C) {
      a[(size_t)c * pairs + r0 + local] = fromF<T>(A[tx][cy]);
      b[(size_t)c * pairs + r0 + local] = fromF<T>(B[tx][cy]);
    }
  }
}
// center_norm over the channels of a (c, pairs) array -> row-major, 32 pairs a block.
template <class TO>
__global__ void centerNormK(const float* prod, TO* out, size_t r0, size_t rows, int C, size_t pairs,
                            const float* scale, const float* offset) {
  extern __shared__ float T_[];
  __shared__ float mean[32], inv[32];
  size_t i0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  for (int c = ty; c < C; c += 8) {
    size_t local = i0 + tx;
    T_[c * 33 + tx] = local < rows ? prod[(size_t)c * pairs + r0 + local] : 0.f;
  }
  __syncthreads();
  if (ty == 0) {
    float s = 0, ss = 0;
    for (int c = 0; c < C; ++c) { float v = T_[c * 33 + tx]; s += v; ss += v * v; }
    float m = s / C; mean[tx] = m; inv[tx] = rsqrtf(ss / C - m * m + 1e-5f);
  }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = i0 + ry; if (local >= rows) continue;
    for (int c = tx; c < C; c += 32)
      out[local * C + c] = fromF<TO>((T_[c * 33 + ry] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}

inline size_t CHUNK = (size_t)64 << 20;   // elements in a chunk tensor
inline bool FUSED_GRID = true;
#include "fusedtriangle.cuh"

template <class T>
void triangle(float* pair, const float* mask, int n, int C, const std::string& pre, bool outgoing,
              bool divideByLength) {
  size_t pairs = (size_t)n * n;
  // the channel-major arrays' channel stride, padded to 8 elements so a channel's start is
  // 16-byte aligned whatever n is (261^2 is odd)
  size_t cs = (pairs + 7) / 8 * 8;
  T* a = scratch<T>("tri.a", cs * C);
  T* b = scratch<T>("tri.b", cs * C);
  // f32: in f16 the contraction (a sum over n of products) overflows - 5CAJ's went to inf
  float* prod = scratch<float>("tri.prod", cs * C);
  std::string pg = projectionGate(pre, C);
  float alpha = divideByLength ? 1.f / n : 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  auto contract = [&]() {         // one n x n GEMM per channel: outgoing P = A B^T, incoming P = B^T A
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, b, cudaType<T>(), n,
        cs, a, cudaType<T>(), n, cs, &zero, prod, CUDA_R_32F, n, cs, C, CUBLAS_COMPUTE_32F, algo));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, a, cudaType<T>(), n,
        cs, b, cudaType<T>(), n, cs, &zero, prod, CUDA_R_32F, n, cs, C, CUBLAS_COMPUTE_32F, algo));
  };
  if constexpr (std::is_same_v<T, half>) {
    if (FUSED_TRIANGLE && C == 128) {           // three kernels: see fusedtriangle.cuh
      half* t2 = scratch<half>("tri.t2whole", pairs * C);
      triIn128(pair, mask, pre, pg, a, b, t2, pairs, cs);
      contract();
      triOut128(prod, pre, t2, pair, pairs, cs);
      return;
    }
  }
  T* norm = scratch<T>("tri.norm", pairs * C);
  size_t rowsPer = std::max<size_t>(1, CHUNK / (4 * C));
  T* pgOut = scratch<T>("tri.pg", rowsPer * 4 * C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    layerNorm2<float, T>(pair + r0 * C, norm + r0 * C, rows, C, pre + ".leftNormInputScale",
                         pre + ".leftNormInputOffset");
    linear<T, T>(norm + r0 * C, pgOut, rows, C, 4 * C, pg);
    triGateK<T><<<dim3((unsigned)((rows + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(
      pgOut, mask, a, b, r0, rows, C, cs);
  }
  contract();
  rowsPer = std::max<size_t>(1, CHUNK / C);
  T* centred = scratch<T>("tri.centred", std::min(rowsPer, pairs) * C);
  T* t1 = scratch<T>("tri.t1", std::min(rowsPer, pairs) * C);
  T* t2 = scratch<T>("tri.t2", std::min(rowsPer, pairs) * C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    centerNormK<T><<<(unsigned)((rows + 31) / 32), dim3(32, 8), C * 33 * 4, STREAM>>>(prod, centred, r0,
      rows, C, cs, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"));
    linear<T, T>(centred, t1, rows, C, C, pre + ".outputProjection");
    linear<T, T>(norm + r0 * C, t2, rows, C, C, pre + ".gatingLinear");
    gatedAddK<T><<<blocks(rows * C), 256, 0, STREAM>>>(pair + r0 * C, t1, t2, rows * C);
  }
}

// ---------------------------------------------------------------- transition
#include "fusedtransition.cuh"
template <class T>
void transition(float* x, size_t rows, int C, int factor, const std::string& pre) {
  int I = C * factor;
  if constexpr (std::is_same_v<T, half>) if (fusedTransition(x, rows, C, I, pre)) return;
  size_t rowsPer = std::max<size_t>(1, CHUNK / (2 * I));
  T* xn = scratch<T>("tr.x", std::min(rowsPer, rows) * C);
  T* wide = scratch<T>("tr.wide", std::min(rowsPer, rows) * 2 * I);
  T* gated = scratch<T>("tr.gated", std::min(rowsPer, rows) * I);
  for (size_t r0 = 0; r0 < rows; r0 += rowsPer) {
    size_t r = std::min(rowsPer, rows - r0);
    layerNorm2<float, T>(x + r0 * C, xn, r, C, pre + ".inputLayerNormScale", pre + ".inputLayerNormOffset");
    linear<T, T>(xn, wide, r, C, 2 * I, pre + ".transition1");
    swiglu<T>(wide, gated, r, I);
    linear<T, float>(gated, x + r0 * C, r, I, C, pre + ".transition2", false, 1.f);
  }
}

// ---------------------------------------------------------------- grid attention kernels
#include "flash.cuh"

// act[r][j] = norm[j][r] for the column direction, 16 bytes a thread (C * elem a multiple of 16)
__global__ void gatherTransposedK(const void* normV, void* actV, int n, int C, size_t r0, size_t R,
                                  int elem) {
  int chunks = C * elem / 16;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * chunks) return;
  int c = (int)(t % chunks); size_t rest = t / chunks; size_t j = rest % n; size_t r = r0 + rest / n;
  reinterpret_cast<uint4*>(actV)[t] = reinterpret_cast<const uint4*>(normV)[(j * n + r) * chunks + c];
}
// bias[h][i][stride] (j padded to a multiple of 8, zeros past n), scaled by `scale`
template <class TB>
__global__ void biasLayoutK(const float* raw, TB* bias, int n, int stride, int heads, bool swap,
                            float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)heads * n * stride;
  if (t >= total) return;
  int j = (int)(t % stride); size_t rest = t / stride; int i = (int)(rest % n); int h = (int)(rest / n);
  float v = j < n ? scale * raw[(swap ? ((size_t)j * n + i) : ((size_t)i * n + j)) * heads + h] : 0.f;
  bias[t] = fromF<TB>(v);
}
// the same from head-major raw logits [hp][pairs] (hp >= heads, the padded projection's width)
template <class TB>
__global__ void biasLayoutHeadMajorK(const float* raw, TB* bias, int n, int stride, int heads, bool swap,
                                     float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)heads * n * stride, pairs = (size_t)n * n;
  if (t >= total) return;
  int j = (int)(t % stride); size_t rest = t / stride; int i = (int)(rest % n); int h = (int)(rest / n);
  float v = j < n ? scale * raw[(size_t)h * pairs + (swap ? ((size_t)j * n + i) : ((size_t)i * n + j))] : 0.f;
  bias[t] = fromF<TB>(v);
}
// a (C, k) weight zero-padded to (C, kp) columns
inline std::string paddedColumns(const std::string& w, int C, int k, int kp) {
  return concatColumns(w + "~pad" + std::to_string(kp), C, {{w, k, false}, {"", kp - k, false}});
}
// pair[(r, j) or (j, r)] += out[r][j], four channels a thread (C a multiple of 4)
__global__ void addGridK(float* pair, const float* out, int n, int C, size_t r0, size_t R, bool tr) {
  int c4 = C / 4;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * c4) return;
  int c = (int)(t % c4); size_t rest = t / c4; size_t j = rest % n; size_t r = r0 + rest / n;
  size_t to = tr ? (j * n + r) : (r * n + j);
  float4* p = reinterpret_cast<float4*>(pair) + to * c4 + c;
  float4 v = *p, o = reinterpret_cast<const float4*>(out)[t];
  v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
  *p = v;
}
__global__ void addGateBiasK(float* qkvg, const float* bias, size_t rows, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * Wd) return;
  size_t r = t / Wd; int c = (int)(t % Wd);
  qkvg[r * 4 * Wd + 3 * Wd + c] += bias[c];
}

// Grid attention over the pair, rows (tr = false) or columns (tr = true), residual added.
template <class T>
void gridAttention(float* pair, const float* mask, int n, int C, int heads, int D,
                   const std::string& pre, bool tr, bool swapBias) {
  size_t pairs = (size_t)n * n; int Wd = heads * D;
  if constexpr (std::is_same_v<T, half>) {
    // three fused kernels (fusedtriangle.cuh): LN + the bias projection (head-major), then per
    // chunk LN + q/k/v/gate (reading the column direction's rows transposed in place), the flash
    // kernel, and the output projection added into the pair
    if (FUSED_GRID && C == 128 && Wd == 128 && heads <= 16 && !hasW(pre + ".gatingQueryBias") &&
        !hasW(pre + ".outputProjectionBias")) {
      int stride = (n + 7) / 8 * 8;
      half* bias = scratch<half>("grid.bias", (size_t)heads * n * stride);
      std::string qkvg = qkvgWeight(pre, C, Wd, true);
      std::string wb = paddedColumns(pre + ".pairBiasProjection", C, heads, 16);
      float scale = 1.f / sqrtf((float)D);
      if (pairs * 4 * Wd * 2 <= ((size_t)4 << 30)) {
        // every row in one pass (this card has the memory): the bias written by the same kernel
        CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * 2, STREAM));    // the padding columns
        half* qkvgOut = scratch<half>("grid.qkvg", (pairs + 128) * 4 * Wd);
        gridIn128(pair, pre, qkvg, qkvgOut, n, 0, pairs, tr, Wh(wb), bias, heads, stride, tr && swapBias);
        half* gathered = scratch<half>("grid.gathered", pairs * Wd);
        flashGrid<half>(qkvgOut, bias, stride, mask, gathered, n, heads, D, 0, n, tr, scale);
        if (!tr) linear<half, float>(gathered, pair, pairs, Wd, C, pre + ".outputProjection", false, 1.f);
        else gridOut128(gathered, pre + ".outputProjection", pair, n, 0, pairs, tr);
        return;
      }
      float* raw = scratch<float>("grid.raw16", pairs * 16);
      lnHeads128<16>(pair, pre + ".actNormScale", pre + ".actNormOffset", wb, raw, pairs);
      biasLayoutHeadMajorK<half><<<blocks((size_t)heads * n * stride), 256, 0, STREAM>>>(
        raw, bias, n, stride, heads, tr && swapBias, LOG2E);
      size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * 4 * Wd)));
      for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
        size_t rows = std::min(R, (size_t)n - r0), prs = rows * n;
        half* qkvgOut = scratch<half>("grid.qkvg", (prs + 128) * 4 * Wd);   // padding: the last query block
        gridIn128(pair, pre, qkvg, qkvgOut, n, r0 * n, prs, tr);
        half* gathered = scratch<half>("grid.gathered", prs * Wd);
        flashGrid<half>(qkvgOut, bias, stride, mask, gathered, n, heads, D, r0, rows, tr, scale);
        if (!tr) linear<half, float>(gathered, pair + r0 * n * C, prs, Wd, C, pre + ".outputProjection", false, 1.f);
        else gridOut128(gathered, pre + ".outputProjection", pair, n, r0 * n, prs, tr);
      }
      return;
    }
  }
  T* norm = scratch<T>("grid.norm", pairs * C);
  layerNorm2<float, T>(pair, norm, pairs, C, pre + ".actNormScale", pre + ".actNormOffset");
  float* raw = scratch<float>("grid.rawbias", pairs * heads);
  linear<T, float>(norm, raw, pairs, C, heads, pre + ".pairBiasProjection");
  int stride = (n + 7) / 8 * 8;
  constexpr bool fast = std::is_same_v<T, half>;
  T* bias = scratch<T>("grid.bias", (size_t)heads * n * stride);
  biasLayoutK<T><<<blocks((size_t)heads * n * stride), 256, 0, STREAM>>>(
    raw, bias, n, stride, heads, tr && swapBias, fast ? LOG2E : 1.f);
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  const float* gateBias = hasW(pre + ".gatingQueryBias") ? W(pre + ".gatingQueryBias") : nullptr;
  const float* outBias = hasW(pre + ".outputProjectionBias") ? W(pre + ".outputProjectionBias") : nullptr;
  if (fast && gateBias) { fprintf(stderr, "%s: a gate bias on the f16 path is not wired\n", pre.c_str()); exit(1); }
  size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * 4 * Wd)));
  float scale = 1.f / sqrtf((float)D);
  for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
    size_t rows = std::min(R, (size_t)n - r0), prs = rows * n;
    const T* act = norm + r0 * n * C;
    if (tr) {
      T* g = scratch<T>("grid.act", prs * C);
      if (C * sizeof(T) % 16) { fprintf(stderr, "grid attention: %d channels are not 16-byte rows\n", C); exit(1); }
      gatherTransposedK<<<blocks(prs * C * sizeof(T) / 16), 256, 0, STREAM>>>(norm, g, n, C, r0, rows, sizeof(T));
      act = g;
    }
    T* qkvgOut = scratch<T>("grid.qkvg", (prs + 128) * 4 * Wd);   // padding: the last query block
    linear<T, T>(act, qkvgOut, prs, C, 4 * Wd, qkvg);
    if constexpr (!fast) if (gateBias) addGateBiasK<<<blocks(prs * Wd), 256, 0, STREAM>>>(qkvgOut, gateBias, prs, Wd);
    T* gathered = scratch<T>("grid.gathered", prs * Wd);
    flashGrid<T>(qkvgOut, bias, stride, mask, gathered, n, heads, D, r0, rows, tr, scale);
    if (!tr && !outBias) {
      linear<T, float>(gathered, pair + r0 * n * C, prs, Wd, C, pre + ".outputProjection", false, 1.f);
      continue;
    }
    float* o = scratch<float>("grid.out", prs * C);
    linear<T, float>(gathered, o, prs, Wd, C, pre + ".outputProjection");
    if (outBias) addBiasK<<<blocks(prs * C), 256, 0, STREAM>>>(o, outBias, prs, C);
    addGridK<<<blocks(prs * C / 4), 256, 0, STREAM>>>(pair, o, n, C, r0, rows, tr);
  }
}

// The five pair updates of a pairformer/MSA/template block, in AF3's order.
template <class T>
void pairUpdates(float* pair, const float* mask, int n, int C, const std::string& pre, bool swap,
                 bool divide, int transitionFactor) {
  triangle<T>(pair, mask, n, C, pre + ".triangleMultiplicationOutgoing", true, divide); stage("tri.out");
  triangle<T>(pair, mask, n, C, pre + ".triangleMultiplicationIncoming", false, divide); stage("tri.in");
  int heads = (int)M.meta(pre + ".pairAttention1.heads"), D = (int)M.meta(pre + ".pairAttention1.dimension");
  gridAttention<T>(pair, mask, n, C, heads, D, pre + ".pairAttention1", false, swap); stage("grid.row");
  gridAttention<T>(pair, mask, n, C, heads, D, pre + ".pairAttention2", true, swap); stage("grid.col");
  transition<T>(pair, (size_t)n * n, C, transitionFactor, pre + ".pairTransition"); stage("transition");
}
