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
inline const half* Fh(const std::string& name) { return Wh("f/" + name); }
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
inline cublasLtHandle_t LT = nullptr;
template <class T> T* wpool(size_t n) {
  static char* chunk = nullptr; static size_t used = 0, have = 0;
  size_t bytes = (n * sizeof(T) + 255) / 256 * 256;
  if (used + bytes > have) {
    have = std::max<size_t>(bytes, (size_t)64 << 20);
    CK(devMalloc(&chunk, have)); used = 0;
  }
  T* p = (T*)(chunk + used); used += bytes; return p;
}
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; bool ok; };
inline std::map<const float*, half*> BIAS_HALF;     // f16 copies of biases (forgotten with the derived weights)
inline const bool BIAS_HALF_HOOK = (FORGET_HOOKS.push_back([] { BIAS_HALF.clear(); }), true);
inline const void* biasFor(const float* bias, int n, bool asHalf) {
  if (!bias || !asHalf) return bias;
  auto& copies = BIAS_HALF;
  auto it = copies.find(bias);
  if (it != copies.end()) return it->second;
  half* h = wpool<half>(n);
  toHalfK<<<blocks(n), 256, 0, STREAM>>>(bias, h, n);
  return copies[bias] = h;
}
inline void ltGemm(const half* X, const half* Wt, void* Y, bool yHalf, size_t rows, int in, int out, const float* biasF,
                   bool relu, float beta, int ldx = 0, int ldy = 0) {
  if (ldx == 0) ldx = in;
  if (ldy == 0) ldy = out;
  if (!LT) CB(cublasLtCreate(&LT));
  const void* bias = biasFor(biasF, out, yHalf);
  static std::map<std::tuple<size_t, int, int, bool, int, bool, int, int>, LtPlan> plans;
  int epi = bias ? (relu ? 2 : 1) : (relu ? 3 : 0);
  auto key = std::make_tuple(rows, in, out, yHalf, epi, beta != 0.f, ldx, ldy);
  auto it = plans.find(key);
  if (it == plans.end()) {
    LtPlan p{};
    CB(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtEpilogue_t e = epi == 1 ? CUBLASLT_EPILOGUE_BIAS : epi == 2 ? CUBLASLT_EPILOGUE_RELU_BIAS
                         : epi == 3 ? CUBLASLT_EPILOGUE_RELU : CUBLASLT_EPILOGUE_DEFAULT;
    CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
    CB(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, out, in, out));
    CB(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, in, (uint64_t)rows, ldx));
    CB(cublasLtMatrixLayoutCreate(&p.c, yHalf ? CUDA_R_16F : CUDA_R_32F, out, (uint64_t)rows, ldy));
    cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 0;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    uint32_t noReduction = CUBLASLT_REDUCTION_SCHEME_NONE;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &noReduction, sizeof(noReduction)));
    cublasLtMatmulHeuristicResult_t res[1]; int got = 0;
    if (bias) CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(LT, p.op, p.a, p.b, p.c, p.c, pref, 1, res, &got);
    if (hs != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLASLt heuristic %d for %zu x %d x %d, f16 out %d, epilogue %d\n", (int)hs, rows, in, out, yHalf, epi);
      exit(1);
    }
    CB(cublasLtMatmulPreferenceDestroy(pref));
    if (got == 0) { fprintf(stderr, "no cuBLASLt algorithm for %zu x %d x %d\n", rows, in, out); exit(1); }
    p.algo = res[0].algo; p.ok = true;
    it = plans.emplace(key, p).first;
  }
  LtPlan& p = it->second;
  if (bias) CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  const float one = 1.f;
  CB(cublasLtMatmul(LT, p.op, &one, Wt, p.a, X, p.b, &beta, Y, p.c, Y, p.c, &p.algo, nullptr, 0, STREAM));
}
inline void gemmHPre(const half* xh, const half* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f,
                     float alpha = 1.f, int ldx = 0, int ldy = 0) {
  if (alpha == 1.f && (out & 7) == 0 && (in & 7) == 0) {
    ltGemm(xh, Wt, Y, false, rows, in, out, nullptr, false, beta, ldx, ldy);
    return;
  }
  const float one = alpha;
  CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one, Wt, CUDA_R_16F, out, xh, CUDA_R_16F,
                  ldx ? ldx : in, &beta, Y, CUDA_R_32F, ldy ? ldy : out, CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
inline void gemmHPreOutH(const half* xh, const half* Wt, half* Y, size_t rows, int in, int out, float beta = 0.f,
                         float alpha = 1.f, int ldx = 0, int ldy = 0) {
  if (alpha == 1.f && (out & 7) == 0 && (in & 7) == 0) {
    ltGemm(xh, Wt, Y, true, rows, in, out, nullptr, false, beta, ldx, ldy);
    return;
  }
  const float one = alpha;
  CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one, Wt, CUDA_R_16F, out, xh, CUDA_R_16F,
                  ldx ? ldx : in, &beta, Y, CUDA_R_16F, ldy ? ldy : out, CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
inline void gemmH(const float* X, const half* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f,
                  float alpha = 1.f) {
  half* xh = scratch<half>("gemmh.x", rows * in);
  toHalfK<<<blocks(rows * in), 256, 0, STREAM>>>(X, xh, rows * in);
  gemmHPre(xh, Wt, Y, rows, in, out, beta, alpha);
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
__device__ __forceinline__ float fastCosF(float angle) {
  float s, c;
  fastSincosD((double)angle, &s, &c);
  return c;
}

inline void addBias(float* y, const float* b, size_t rows, int C) {
  size_t n = ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) ? rows * (size_t)(C >> 2) : rows * C;
  addBiasK<<<blocks(n), 256, 0, STREAM>>>(y, b, rows, C);
}
// LayerNorm over the last axis, a warp a row; offset may be null, scale may be null (affine-free)
template <class TY = float>
__global__ void layerNormK(const float* x, TY* y, size_t rows, int C, const float* scale, const float* offset,
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
    if constexpr (std::is_same_v<TY, half>) y[row * ldy + c] = __float2half(n);
    else y[row * ldy + c] = n;
  }
}
template <int MAX_K, bool EXACT, class TY = float>
__global__ void __launch_bounds__(256, (MAX_K <= 16 && EXACT) ? 4 : 2)
layerNormRegK(const float* __restrict__ x, TY* __restrict__ y, size_t rows, int C,
              const float* __restrict__ scale, const float* __restrict__ offset, float eps, int ldx, int ldy) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * ldx;
  TY* yr = y + row * ldy;
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
      if constexpr (std::is_same_v<TY, half>) yr[c] = __float2half(n);
      else yr[c] = n;
    }
  }
}
template <int MAX_K, class TY2 = float>
__global__ void __launch_bounds__(256, 2)
layerNormDualRegK(const float* __restrict__ x, float* __restrict__ y1, TY2* __restrict__ y2, size_t rows,
                  const float* __restrict__ scale1, const float* __restrict__ offset1,
                  const float* __restrict__ scale2, const float* __restrict__ offset2, float eps) {
  constexpr int C = MAX_K * 32;
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  float* yr1 = y1 + row * C;
  TY2* yr2 = y2 + row * C;
  float xv[MAX_K];
  float s = 0;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    int c = k * 32 + lane;
    float val = __ldg(xr + c);
    xv[k] = val;
    s += val;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / (float)C, v = 0;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    float d = xv[k] - mean;
    v += d * d;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / (float)C + eps);
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    int c = k * 32 + lane;
    float n = (xv[k] - mean) * inv;
    float n1 = n * __ldg(scale1 + c) + __ldg(offset1 + c);
    float n2 = n * __ldg(scale2 + c) + __ldg(offset2 + c);
    yr1[c] = n1;
    if constexpr (std::is_same_v<TY2, half>) yr2[c] = __float2half(n2);
    else yr2[c] = n2;
  }
}
template <class TY = float>
inline void layerNorm(const float* x, TY* y, size_t rows, int C, const float* scale, const float* offset,
                      float eps = 1e-5f, int ldx = 0, int ldy = 0) {
  unsigned g = (unsigned)((rows + 7) / 8);
  int lx = ldx ? ldx : C, ly = ldy ? ldy : C;
  switch (C) {
    case 128:  layerNormRegK<4,  true, TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 256:  layerNormRegK<8,  true, TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 384:  layerNormRegK<12, true, TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 512:  layerNormRegK<16, true, TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 768:  layerNormRegK<24, true, TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 960:  layerNormRegK<30, true, TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    case 1152: layerNormRegK<36, true, TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly); return;
    default:
      if (C <= 512) layerNormRegK<16, false, TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly);
      else layerNormK<TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, eps, lx, ly);
  }
}
template <class TY2 = float>
inline void layerNormDual(const float* x, float* y1, TY2* y2, size_t rows, int C,
                          const float* scale1, const float* offset1,
                          const float* scale2, const float* offset2, float eps = 1e-5f) {
  unsigned g = (unsigned)((rows + 7) / 8);
  if (C == 960) {
    layerNormDualRegK<30, TY2><<<g, 256, 0, STREAM>>>(x, y1, y2, rows, scale1, offset1, scale2, offset2, eps);
    return;
  }
  if (C == 1152) {
    layerNormDualRegK<36, TY2><<<g, 256, 0, STREAM>>>(x, y1, y2, rows, scale1, offset1, scale2, offset2, eps);
    return;
  }
  layerNorm<float>(x, y1, rows, C, scale1, offset1, eps);
  layerNorm<TY2>(x, y2, rows, C, scale2, offset2, eps);
}
__device__ __forceinline__ float siluF(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float geluF(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

template <class TY = float>
__global__ void swigluK(const float* __restrict__ h, TY* __restrict__ g, size_t rows, int F) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  constexpr uintptr_t gAlign = std::is_same_v<TY, half> ? 7 : 15;
  if ((F & 3) == 0 && ((uintptr_t)h & 15) == 0 && ((uintptr_t)g & gAlign) == 0) {
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
    float o0 = siluF(a.x) * b.x, o1 = siluF(a.y) * b.y, o2 = siluF(a.z) * b.z, o3 = siluF(a.w) * b.w;
    if constexpr (std::is_same_v<TY, half>) {
      __half2 h0 = __floats2half2_rn(o0, o1);
      __half2 h1 = __floats2half2_rn(o2, o3);
      reinterpret_cast<uint2*>(g)[t] = make_uint2(*reinterpret_cast<const unsigned*>(&h0),
                                                  *reinterpret_cast<const unsigned*>(&h1));
    } else {
      reinterpret_cast<float4*>(g)[t] = make_float4(o0, o1, o2, o3);
    }
    return;
  }
  if (t >= rows * F) return;
  size_t r = t / F; int c = (int)(t % F);
  float v = siluF(h[r * 2 * F + c]) * h[r * 2 * F + F + c];
  if constexpr (std::is_same_v<TY, half>) g[t] = __float2half(v);
  else g[t] = v;
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
