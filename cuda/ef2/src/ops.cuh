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
                 int ldx = 0, int ldy = 0, float alpha = 1.f) {
  const float one = alpha;      // (alpha: ESM-C's residual scale, esmc.cuh towerGemm; 1 everywhere else)
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
inline void gemmH(const float* X, const half* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f,
                  float alpha = 1.f) {
  half* xh = scratch<half>("gemmh.x", rows * in);
  toHalfK<<<blocks(rows * in), 256, 0, STREAM>>>(X, xh, rows * in);
  const float one = alpha;
  CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one, Wt, CUDA_R_16F, out, xh, CUDA_R_16F, in, &beta,
                  Y, CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------- elementwise
// Exact double-precision range-reduced sin/cos to float32 without Payne-Hanek stack frame (STACK:0).
// Valid for |angle| <= 1e7 rad with 0-ULP float32 error vs (float)sin(angle), (float)cos(angle).
__device__ __forceinline__ void fastSincosD(double angle, float* sOut, float* cOut) {
  constexpr double INV_PIO2 = 0x1.45f306dc9c883p-1;   // 2 / pi
  constexpr double PIO2_HI  = 0x1.921fb54442d18p+0;   // pi / 2 high 53 bits
  constexpr double PIO2_LO  = 0x1.1a62633145c07p-54;  // pi / 2 low 53 bits
  double kf = nearbyint(angle * INV_PIO2);
  long long q = (long long)kf;
  double r = fma(-kf, PIO2_HI, angle) - kf * PIO2_LO;
  double r2 = r * r;
  double s = (((((-0x1.ae64567f544e4p-26 * r2 + 0x1.71de3a556c734p-19) * r2
                 - 0x1.a01a01a01a01ap-13) * r2 + 0x1.1111111111111p-7) * r2
               - 0x1.5555555555555p-3) * r2) * r + r;
  double c = (((((-0x1.27e4fb7789f5cp-22 * r2 + 0x1.a01a01a01a01ap-16) * r2
                 - 0x1.6c16c16c16c17p-10) * r2 + 0x1.5555555555555p-5) * r2
               - 0x1.0000000000000p-1) * r2) + 1.0;
  double sinA = (q & 1) ? c : s;
  double cosA = (q & 1) ? -s : c;
  if (q & 2) { sinA = -sinA; cosA = -cosA; }
  *sOut = (float)sinA;
  *cOut = (float)cosA;
}

inline void addBias(float* y, const float* b, size_t rows, int C) {
  size_t n = ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) ? rows * (size_t)(C >> 2) : rows * C;
  addBiasK<<<blocks(n), 256, 0, STREAM>>>(y, b, rows, C);
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
template <int MAX_K, bool EXACT>
__global__ void __launch_bounds__(256, (MAX_K <= 16 && EXACT) ? 4 : 2)
layerNormRegK(const float* __restrict__ x, float* __restrict__ y, size_t rows, int C,
              const float* __restrict__ scale, const float* __restrict__ offset, float eps, int ldx, int ldy) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * ldx;
  float* yr = y + row * ldy;
  float xv[MAX_K];
  float s = 0;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    int c = k * 32 + lane;
    float val = (EXACT || c < C) ? __ldg(xr + c) : 0.f;
    xv[k] = val;
    s += val;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / (EXACT ? (float)(MAX_K * 32) : (float)C), v = 0;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    int c = k * 32 + lane;
    float d = (EXACT || c < C) ? (xv[k] - mean) : 0.f;
    v += d * d;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / (EXACT ? (float)(MAX_K * 32) : (float)C) + eps);
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    int c = k * 32 + lane;
    if (EXACT || c < C) {
      float n = (xv[k] - mean) * inv;
      if (scale) n *= __ldg(scale + c);
      if (offset) n += __ldg(offset + c);
      yr[c] = n;
    }
  }
}
inline void layerNorm(const float* x, float* y, size_t rows, int C, const float* scale, const float* offset,
                      float eps = 1e-5f, int ldx = 0, int ldy = 0) {
  unsigned g = (unsigned)((rows + 7) / 8);
  int lx = ldx ? ldx : C, ly = ldy ? ldy : C;
  switch (C) {
    case 128:  layerNormRegK<4,  true><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 256:  layerNormRegK<8,  true><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 384:  layerNormRegK<12, true><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 512:  layerNormRegK<16, true><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 768:  layerNormRegK<24, true><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 960:  layerNormRegK<30, true><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 1152: layerNormRegK<36, true><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    default:
      if (C <= 512) layerNormRegK<16, false><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly);
      else layerNormK<<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly);
  }
}
__device__ __forceinline__ float siluF(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float geluF(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

__global__ void swigluK(const float* __restrict__ h, float* __restrict__ g, size_t rows, int F) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((F & 3) == 0 && (((uintptr_t)h | (uintptr_t)g) & 15) == 0) {
    int f4 = F >> 2;
    size_t n4 = rows * (size_t)f4;
    if (t >= n4) return;
    size_t r; int c4;
    if ((f4 & (f4 - 1)) == 0) {
      c4 = (int)(t & (size_t)(f4 - 1));
      r = t >> (__ffs(f4) - 1);
    } else if (f4 == 768 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 768u;
      r = q; c4 = (int)(u - q * 768u);
    } else if (f4 == 640 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 640u;
      r = q; c4 = (int)(u - q * 640u);
    } else if (f4 == 480 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 480u;
      r = q; c4 = (int)(u - q * 480u);
    } else if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, uf = (unsigned)f4, q = u / uf;
      r = q; c4 = (int)(u - q * uf);
    } else {
      r = t / f4; c4 = (int)(t % f4);
    }
    const float4* hr = reinterpret_cast<const float4*>(h + r * 2 * F);
    float4 a = __ldg(hr + c4), b = __ldg(hr + f4 + c4);
    reinterpret_cast<float4*>(g)[t] = make_float4(siluF(a.x) * b.x, siluF(a.y) * b.y, siluF(a.z) * b.z, siluF(a.w) * b.w);
    return;
  }
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
