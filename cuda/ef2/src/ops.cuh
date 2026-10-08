// ESMFold2's building blocks, float32: the reference path every stage is held to biohub's own forward
// with (cuda/ef2/oracle.py). Shared infrastructure - the exported model file, device weights,
// scratch, cuBLAS - is cuda/af3's.
#pragma once
#include "../../af3/src/common.cuh"

inline bool FAST = false;
// the GEMMs' f16 tensor-core arm (inputs rounded to f16 inside cuBLAS, f32 accumulation): the sampler
// sets it under --fast, where every other GEMM takes TF32
inline bool GEMM16 = false;

// ---------------------------------------------------------------- weights
// "f/<bundle name>" (the folding bundle) or "c/<bundle name>" (ESM-C); dims from "#k"
inline size_t dimOf(const std::string& key, int k) { return (size_t)M.meta(key + "#" + std::to_string(k)); }
inline const float* F(const std::string& name) { return W("f/" + name); }
// the shim ("lm/...") is per folding model and the tower is shared: a folding bundle that carries its own
// shim is read first (the full ESMFold2's differs from ESMFold2-Fast's in all twelve tensors)
inline std::string shimKey(const std::string& name) {
  return !name.rfind("lm/", 0) && M.has("f/" + name) ? "f/" + name : "c/" + name;
}
inline const float* Cw(const std::string& name) { return W(shimKey(name)); }

// ---------------------------------------------------------------- GEMM, row-major
// Y[rows, out] = X[rows, in] Wt[in, out] (+ beta Y); f32 accumulate (TF32 only under --fast)
inline void gemm(const float* X, const float* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f,
                 int ldx = 0, int ldy = 0) {
  const float one = 1.f;
  if (GEMM16) {
    CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one, Wt, CUDA_R_32F, out, X, CUDA_R_32F,
                    ldx ? ldx : in, &beta, Y, CUDA_R_32F, ldy ? ldy : out, CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return;
  }
  CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one, Wt, out, X, ldx ? ldx : in, &beta, Y,
                 ldy ? ldy : out));
}

// the same with an f16 weight (W's mirror) and f32 X and Y: X narrowed to f16, tensor cores, f32 accumulation
inline void gemmH(const float* X, const half* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f) {
  half* xh = scratch<half>("gemmh.x", rows * in);
  toHalfK<<<blocks(rows * in), 256, 0, STREAM>>>(X, xh, rows * in);
  const float one = 1.f;
  CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one, Wt, CUDA_R_16F, out, xh, CUDA_R_16F, in, &beta,
                  Y, CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------- elementwise
inline void addBias(float* y, const float* b, size_t rows, int C) {
  addBiasK<<<blocks(rows * C), 256, 0, STREAM>>>(y, b, rows, C);
}
// LayerNorm over the last axis, a warp a row; offset may be null, scale may be null (affine-free)
__global__ void layerNormK(const float* x, float* y, size_t rows, int C, const float* scale, const float* offset,
                           float eps, int ldx, int ldy) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * ldx;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / C + eps);
  for (int c = lane; c < C; c += 32) {
    float n = (xr[c] - mean) * inv;
    if (scale) n *= scale[c];
    if (offset) n += offset[c];
    y[row * ldy + c] = n;
  }
}
inline void layerNorm(const float* x, float* y, size_t rows, int C, const float* scale, const float* offset,
                      float eps = 1e-5f, int ldx = 0, int ldy = 0) {
  layerNormK<<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, ldx ? ldx : C,
                                                              ldy ? ldy : C);
}
__device__ __forceinline__ float siluF(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float geluF(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

__global__ void swigluK(const float* h, float* g, size_t rows, int F) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * F) return;
  size_t r = t / F; int c = (int)(t % F);
  g[t] = siluF(h[r * 2 * F + c]) * h[r * 2 * F + F + c];
}

// ---------------------------------------------------------------- checks
inline double checkOracle(const char* label, const float* d, size_t n, const std::string& oracle) {
  if (!M.has(oracle)) { printf("  %-30s (no oracle %s)\n", label, oracle.c_str()); return -1; }
  if (M.len(oracle) != n) { printf("  %-30s length %zu against the oracle's %zu\n", label, n, M.len(oracle)); return -1; }
  std::vector<float> h = download(d, n);
  double r = relRms(h.data(), M.f(oracle), n);
  printf("  %-30s relRMS %.3e\n", label, r);
  return r;
}
