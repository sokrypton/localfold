// BioEmu2's building blocks: the weights read straight out of af3-any-model's blob (one scope,
// diffuser/~/bioemu2, the vendor's PyTorch names and (out, in) layouts - converters/bioemu2.py is a rename), and
// the kernels every stage shares. float32 throughout on the reference path (cuda/bioemu2/oracle.py); --fast
// takes TF32 in cuBLAS. Shared infrastructure - the model file, scratch, cuBLAS - is cuda/af3's, and the
// LayerNorm and SwiGLU kernels are cuda/ef2's.
#pragma once
#include "../../ef2/src/ops.cuh"

namespace be2 {

constexpr int WQ = 32, WK = 128, CTX = (WK - WQ) / 2;   // atom query / key windows; keys centred on queries
constexpr float SIGMA_DATA = 16.f;
constexpr float NEG_INF = -1e6f;                         // the vendor's -INF (1e6): a masked logit, not -inf

// ---------------------------------------------------------------- weights
inline const std::string SCOPE = "w/diffuser/~/bioemu2/";
inline bool hasP(const std::string& n) { return M.has(SCOPE + n); }
inline const float* P(const std::string& n) { return W(SCOPE + n); }
inline int dimP(const std::string& n, int k) { return (int)M.meta(SCOPE + n + "#" + std::to_string(k)); }
inline size_t lenP(const std::string& n) { return M.len(SCOPE + n); }
// layer l of a stacked tensor ([layers, ...])
inline const float* PL(const std::string& n, int l) {
  return P(n) + (size_t)l * (lenP(n) / (size_t)dimP(n, 0));
}
inline const float* PLopt(const std::string& n, int l) { return hasP(n) ? PL(n, l) : nullptr; }
inline const float* Popt(const std::string& n) { return hasP(n) ? P(n) : nullptr; }

// ---------------------------------------------------------------- GEMM, row-major, torch weights
// Y[rows, out] = alpha X[rows, in] W[out, in]^T + beta Y   (ldx/ldy: row strides, for a slice of a wider row)
// GEMM_HALF (the sampler under --fast): X rounded to f16 and the weight read from its f16 mirror, f32 accumulation
// and output - the tensor cores' f16 rate, twice TF32's. Every input it takes is a normalised activation (LN or adaLN)
// or an attention/SwiGLU product of them, far inside f16's range.
// 🔴 EXCEPT A SwiGLU PRODUCT, WHICH IS UNBOUNDED: silu(fc1 x) fc2 x reaches 87,750 in the atom stream's transition at
// 6MRR (f16 stops at 65,504, and the fold came back NaN), so fc3 takes bf16 - f32's range, 8 bits of mantissa
// (`wide`).
inline bool GEMM_HALF = false;
template <class TH> inline std::map<const float*, TH*>& mirrors() { static std::map<const float*, TH*> m; return m; }
__global__ void be2ToHalfK(const float* __restrict__ x, half* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)x & 15) == 0) && (((uintptr_t)y & 7) == 0)) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 v = __ldg(reinterpret_cast<const float4*>(x) + t);
      __half2 h0 = __floats2half2_rn(v.x, v.y);
      __half2 h1 = __floats2half2_rn(v.z, v.w);
      uint2 out;
      out.x = *reinterpret_cast<const unsigned*>(&h0);
      out.y = *reinterpret_cast<const unsigned*>(&h1);
      reinterpret_cast<uint2*>(y)[t] = out;
    }
    return;
  }
  if (t < n) y[t] = __float2half(x[t]);
}
__global__ void toBf16K(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)x & 15) == 0) && (((uintptr_t)y & 7) == 0)) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 v = __ldg(reinterpret_cast<const float4*>(x) + t);
      __nv_bfloat162 b0 = __floats2bfloat162_rn(v.x, v.y);
      __nv_bfloat162 b1 = __floats2bfloat162_rn(v.z, v.w);
      uint2 out;
      out.x = *reinterpret_cast<const unsigned*>(&b0);
      out.y = *reinterpret_cast<const unsigned*>(&b1);
      reinterpret_cast<uint2*>(y)[t] = out;
    }
    return;
  }
  if (t < n) y[t] = __float2bfloat16(x[t]);
}
template <class TH> inline const TH* halfMirror(const float* w, size_t n) {
  auto& m = mirrors<TH>();
  auto it = m.find(w);
  if (it != m.end()) return it->second;
  TH* h = dallocT<TH>(n);
  size_t nt = ((n & 3) == 0 && (((uintptr_t)w & 15) == 0) && (((uintptr_t)h & 7) == 0)) ? (n >> 2) : n;
  if constexpr (std::is_same_v<TH, half>) be2ToHalfK<<<blocks(nt), 256, 0, STREAM>>>(w, h, n);
  else toBf16K<<<blocks(nt), 256, 0, STREAM>>>(w, h, n);
  return m[w] = h;
}
template <class TH> __global__ void toHalfRowsK(const float* __restrict__ x, TH* __restrict__ y, size_t rows, int in, int ldx) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((in & 3) == 0 && (ldx & 3) == 0 && (((uintptr_t)x & 15) == 0) && (((uintptr_t)y & 7) == 0)) {
    int in4 = in >> 2, ldx4 = ldx >> 2;
    size_t n4 = rows * (size_t)in4;
    if (t >= n4) return;
    size_t r; int c4;
    if ((in4 & (in4 - 1)) == 0) {
      c4 = (int)(t & (size_t)(in4 - 1));
      r = t >> (__ffs(in4) - 1);
    } else if (in4 == 192 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 192u; r = q; c4 = (int)(u - q * 192u);
    } else if (in4 == 224 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 224u; r = q; c4 = (int)(u - q * 224u);
    } else if (in4 == 96 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 96u; r = q; c4 = (int)(u - q * 96u);
    } else if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, ui = (unsigned)in4, q = u / ui; r = q; c4 = (int)(u - q * ui);
    } else {
      r = t / in4; c4 = (int)(t % in4);
    }
    float4 v = __ldg(reinterpret_cast<const float4*>(x) + r * ldx4 + c4);
    uint2 out;
    if constexpr (std::is_same_v<TH, half>) {
      auto cl = [](float a) { return fminf(fmaxf(a, -65504.f), 65504.f); };
      __half2 h0 = __floats2half2_rn(cl(v.x), cl(v.y));
      __half2 h1 = __floats2half2_rn(cl(v.z), cl(v.w));
      out.x = *reinterpret_cast<const unsigned*>(&h0);
      out.y = *reinterpret_cast<const unsigned*>(&h1);
    } else {
      __nv_bfloat162 b0 = __floats2bfloat162_rn(v.x, v.y);
      __nv_bfloat162 b1 = __floats2bfloat162_rn(v.z, v.w);
      out.x = *reinterpret_cast<const unsigned*>(&b0);
      out.y = *reinterpret_cast<const unsigned*>(&b1);
    }
    reinterpret_cast<uint2*>(y)[t] = out;
    return;
  }
  if (t >= rows * in) return;
  const float v = x[(t / in) * ldx + t % in];
  // (saturated: a padding atom's state, which nothing reads, grows past f16 and would be inf)
  if constexpr (std::is_same_v<TH, half>) y[t] = __float2half(fminf(fmaxf(v, -65504.f), 65504.f)); else y[t] = __float2bfloat16(v);
}
template <class TH> inline void toHalfRows(const float* X, TH* xh, size_t rows, int in, int ldx) {
  int lx = ldx ? ldx : in;
  size_t nt = ((in & 3) == 0 && (lx & 3) == 0 && (((uintptr_t)X & 15) == 0) && (((uintptr_t)xh & 7) == 0))
              ? rows * (size_t)(in >> 2) : rows * in;
  toHalfRowsK<TH><<<blocks(nt), 256, 0, STREAM>>>(X, xh, rows, in, lx);
}
template <class TH> inline void linearHalf(const float* X, const float* Wt, float* Y, size_t rows, int in, int out,
                                           float beta, int ldx, int ldy, float alpha) {
  const TH* wh = halfMirror<TH>(Wt, (size_t)in * out);
  TH* xh = scratch<TH>(std::is_same_v<TH, half> ? "be2.gemm.xh" : "be2.gemm.xb", rows * in);
  toHalfRows<TH>(X, xh, rows, in, ldx);
  const cudaDataType t = std::is_same_v<TH, half> ? CUDA_R_16F : CUDA_R_16BF;
  CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, wh, t, in, xh, t, in,
                  &beta, Y, CUDA_R_32F, ldy ? ldy : out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
inline void linear(const float* X, const float* Wt, float* Y, size_t rows, int in, int out, float beta = 0.f,
                   int ldx = 0, int ldy = 0, float alpha = 1.f, bool wide = false) {
  if (GEMM_HALF && in >= 16 && out >= 16) {
    if (wide) linearHalf<__nv_bfloat16>(X, Wt, Y, rows, in, out, beta, ldx, ldy, alpha);
    else linearHalf<half>(X, Wt, Y, rows, in, out, beta, ldx, ldy, alpha);
    return;
  }
  CB(cublasSgemm(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, Wt, in, X, ldx ? ldx : in, &beta, Y,
                 ldy ? ldy : out));
}
// ...from an input already in f16 / bf16 (written so by the kernel that made it: halfLn, adaLnTo, sigmoidMulTo,
// swigluTo), so no conversion pass runs between
template <class TH> inline void linearPre(const TH* Xh, const TH* Wh, float* Y, size_t rows, int in, int out,
                                          float beta = 0.f, int ldy = 0) {
  const float alpha = 1.f;
  const cudaDataType t = std::is_same_v<TH, half> ? CUDA_R_16F : CUDA_R_16BF;
  CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha, Wh, t, in, Xh, t, in,
                  &beta, Y, CUDA_R_32F, ldy ? ldy : out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
template <class TH>
inline void linearBatchPre(const TH* Xh, long long strideX, const TH* Wh, long long strideW,
                           float* Y, long long strideY, size_t rows, int in, int out, int batch) {
  const float alpha = 1.f, beta = 0.f;
  const cudaDataType t = std::is_same_v<TH, half> ? CUDA_R_16F : CUDA_R_16BF;
  CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha,
                                Wh, t, in, strideW, Xh, t, in, strideX,
                                &beta, Y, CUDA_R_32F, out, strideY, batch,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
inline void linearBatch(const float* X, long long strideX, const float* Wt, long long strideW,
                        float* Y, long long strideY, size_t rows, int in, int out, int batch) {
  const float alpha = 1.f, beta = 0.f;
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &alpha,
                               Wt, in, strideW, X, in, strideX, &beta, Y, out, strideY, batch));
}
// two torch-layout weights stacked as one: [W1; W2] (rows), its mirror - fc1|fc2 or proj_qkv|proj_g as one GEMM
template <class TH> inline const TH* stackedMirror2(const float* w1, size_t n1, const float* w2, size_t n2) {
  static std::map<const float*, TH*> m;
  auto it = m.find(w1);
  if (it != m.end()) return it->second;
  TH* h = dallocT<TH>(n1 + n2);
  if constexpr (std::is_same_v<TH, float>) {
    CK(cudaMemcpyAsync(h, w1, n1 * 4, cudaMemcpyDeviceToDevice, STREAM));
    CK(cudaMemcpyAsync(h + n1, w2, n2 * 4, cudaMemcpyDeviceToDevice, STREAM));
  } else {
    size_t nt1 = ((n1 & 3) == 0 && (((uintptr_t)w1 & 15) == 0) && (((uintptr_t)h & 7) == 0)) ? (n1 >> 2) : n1;
    size_t nt2 = ((n2 & 3) == 0 && (((uintptr_t)w2 & 15) == 0) && (((uintptr_t)(h + n1) & 7) == 0)) ? (n2 >> 2) : n2;
    if constexpr (std::is_same_v<TH, half>) { be2ToHalfK<<<blocks(nt1), 256, 0, STREAM>>>(w1, h, n1); be2ToHalfK<<<blocks(nt2), 256, 0, STREAM>>>(w2, h + n1, n2); }
    else { toBf16K<<<blocks(nt1), 256, 0, STREAM>>>(w1, h, n1); toBf16K<<<blocks(nt2), 256, 0, STREAM>>>(w2, h + n1, n2); }
  }
  return m[w1] = h;
}
template <class TH> inline const TH* stackedMirror(const float* w1, const float* w2, size_t n) {
  return stackedMirror2<TH>(w1, n, w2, n);
}
template <class TH> __device__ __forceinline__ TH toH(float v) {
  if constexpr (std::is_same_v<TH, half>) return __float2half(fminf(fmaxf(v, -65504.f), 65504.f));
  else return __float2bfloat16(v);
}
template <class TY> __device__ __forceinline__ void storeVec4(TY* dst, int c4, float o0, float o1, float o2, float o3) {
  if constexpr (std::is_same_v<TY, float>) {
    reinterpret_cast<float4*>(dst)[c4] = make_float4(o0, o1, o2, o3);
  } else if constexpr (std::is_same_v<TY, half>) {
    auto cl = [](float a) { return fminf(fmaxf(a, -65504.f), 65504.f); };
    __half2 h0 = __floats2half2_rn(cl(o0), cl(o1));
    __half2 h1 = __floats2half2_rn(cl(o2), cl(o3));
    uint2 u;
    u.x = *reinterpret_cast<const unsigned*>(&h0);
    u.y = *reinterpret_cast<const unsigned*>(&h1);
    reinterpret_cast<uint2*>(dst)[c4] = u;
  } else {
    __nv_bfloat162 b0 = __floats2bfloat162_rn(o0, o1);
    __nv_bfloat162 b1 = __floats2bfloat162_rn(o2, o3);
    uint2 u;
    u.x = *reinterpret_cast<const unsigned*>(&b0);
    u.y = *reinterpret_cast<const unsigned*>(&b1);
    reinterpret_cast<uint2*>(dst)[c4] = u;
  }
}

// ---------------------------------------------------------------- elementwise
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
// LayerNorm into f16/f32 (scale/offset may be null), with optional fused SiLU, a warp a row
template <class TH, bool APPLY_SILU = false>
__global__ void lnToK(const float* __restrict__ x, TH* __restrict__ y, size_t rows, int C,
                      const float* __restrict__ scale, const float* __restrict__ offset, int ldx) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * ldx;
  float s = 0; for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) {
    float n = (xr[c] - mean) * inv;
    if (scale) n *= scale[c];
    if (offset) n += offset[c];
    if constexpr (APPLY_SILU) n = siluF(n);
    if constexpr (std::is_same_v<TH, float>) y[row * C + c] = n;
    else y[row * C + c] = toH<TH>(n);
  }
}
template <class TH, int K, bool APPLY_SILU = false>
__global__ void __launch_bounds__(256, (K <= 16 ? 4 : 2))
lnToRegK(const float* __restrict__ x, TH* __restrict__ y, size_t rows,
         const float* __restrict__ scale, const float* __restrict__ offset, int ldx) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  constexpr int C = K * 32;
  const float* xr = x + row * ldx;
  float xv[K];
  float s = 0.f;
  #pragma unroll
  for (int k = 0; k < K; ++k) {
    float v = __ldg(xr + k * 32 + lane);
    xv[k] = v;
    s += v;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, var = 0.f;
  #pragma unroll
  for (int k = 0; k < K; ++k) {
    float d = xv[k] - mean;
    var += d * d;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
  float inv = rsqrtf(var / C + 1e-5f);
  TH* yr = y + row * C;
  #pragma unroll
  for (int k = 0; k < K; ++k) {
    int c = k * 32 + lane;
    float n = (xv[k] - mean) * inv;
    if (scale) n *= __ldg(scale + c);
    if (offset) n += __ldg(offset + c);
    if constexpr (APPLY_SILU) n = siluF(n);
    if constexpr (std::is_same_v<TH, float>) yr[c] = n;
    else yr[c] = toH<TH>(n);
  }
}
struct Cond0Tables {
  const float* scale[4];
  const float* offset[4];
};
// Fused across all layers l = 0..layers-1 and all 4 conditioning streams (atom, tok, q_coup, s_coup) for C = 768:
// normalizes cnd[row, :] once per (row, l) and writes all 4 SiLU(LN(cnd)) outputs.
template <class TH>
__global__ void __launch_bounds__(256, 2)
lnToCond4K(const float* __restrict__ cnd, TH* __restrict__ yAll, int T, int rows, Cond0Tables tab) {
  int row = (int)blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  int l = (int)blockIdx.y;
  if (row >= rows) return;
  int rLocal = row % (T + 1);
  constexpr int K4 = 6, C = 768;
  const float4* xr4 = reinterpret_cast<const float4*>(cnd + (size_t)row * C);
  float4 xv[K4];
  float s = 0.f;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    float4 v = __ldg(xr4 + k * 32 + lane);
    xv[k] = v;
    s += (v.x + v.y) + (v.z + v.w);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, var = 0.f;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    float dx = xv[k].x - mean, dy = xv[k].y - mean, dz = xv[k].z - mean, dw = xv[k].w - mean;
    var += (dx * dx + dy * dy) + (dz * dz + dw * dw);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
  float inv = rsqrtf(var / C + 1e-5f);
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    xv[k] = make_float4((xv[k].x - mean) * inv, (xv[k].y - mean) * inv,
                        (xv[k].z - mean) * inv, (xv[k].w - mean) * inv);
  }
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    if ((m & 1) && rLocal >= T) continue;
    const float4* sc4 = reinterpret_cast<const float4*>(tab.scale[m] + (size_t)l * C);
    const float4* of4 = tab.offset[m] ? reinterpret_cast<const float4*>(tab.offset[m] + (size_t)l * C) : nullptr;
    TH* yr = yAll + ((size_t)(m * (int)gridDim.y + l) * (size_t)rows + (size_t)row) * C;
    #pragma unroll
    for (int k = 0; k < K4; ++k) {
      int c4 = k * 32 + lane;
      float4 sv = __ldg(sc4 + c4);
      float4 ov = of4 ? __ldg(of4 + c4) : make_float4(0.f, 0.f, 0.f, 0.f);
      storeVec4<TH>(yr, c4,
                    siluF(xv[k].x * sv.x + ov.x),
                    siluF(xv[k].y * sv.y + ov.y),
                    siluF(xv[k].z * sv.z + ov.z),
                    siluF(xv[k].w * sv.w + ov.w));
    }
  }
}
template <class TH, bool APPLY_SILU = false>
inline void lnTo(const float* x, TH* y, size_t rows, int C, const float* scale, const float* offset, int ldx = 0) {
  int lx = ldx ? ldx : C;
  unsigned g = (unsigned)((rows + 7) / 8);
  if (C == 128) { lnToRegK<TH, 4, APPLY_SILU><<<g, 256, 0, STREAM>>>(x, y, rows, scale, offset, lx); return; }
  if (C == 256) { lnToRegK<TH, 8, APPLY_SILU><<<g, 256, 0, STREAM>>>(x, y, rows, scale, offset, lx); return; }
  if (C == 384) { lnToRegK<TH, 12, APPLY_SILU><<<g, 256, 0, STREAM>>>(x, y, rows, scale, offset, lx); return; }
  if (C == 512) { lnToRegK<TH, 16, APPLY_SILU><<<g, 256, 0, STREAM>>>(x, y, rows, scale, offset, lx); return; }
  if (C == 768) { lnToRegK<TH, 24, APPLY_SILU><<<g, 256, 0, STREAM>>>(x, y, rows, scale, offset, lx); return; }
  if (C == 896) { lnToRegK<TH, 28, APPLY_SILU><<<g, 256, 0, STREAM>>>(x, y, rows, scale, offset, lx); return; }
  lnToK<TH, APPLY_SILU><<<g, 256, 0, STREAM>>>(x, y, rows, C, scale, offset, lx);
}
// the SwiGLU's product from fc1|fc2's [rows, 2I] f16 output, into fc3's bf16 input
__global__ void swigluToK(const half* __restrict__ h12, __nv_bfloat16* __restrict__ g, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((I & 7) == 0 && (((uintptr_t)h12 | (uintptr_t)g) & 15) == 0) {
    int i8 = I >> 3;
    size_t n8 = rows * (size_t)i8;
    if (t >= n8) return;
    size_t r; int c8;
    if ((i8 & (i8 - 1)) == 0) {
      c8 = (int)(t & (size_t)(i8 - 1));
      r = t >> (__ffs(i8) - 1);
    } else if (i8 == 384 && n8 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 384u; r = q; c8 = (int)(u - q * 384u);
    } else if (i8 == 336 && n8 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 336u; r = q; c8 = (int)(u - q * 336u);
    } else if (i8 == 192 && n8 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 192u; r = q; c8 = (int)(u - q * 192u);
    } else if (n8 <= UINT32_MAX) {
      unsigned u = (unsigned)t, ui = (unsigned)i8, q = u / ui; r = q; c8 = (int)(u - q * ui);
    } else {
      r = t / i8; c8 = (int)(t % i8);
    }
    const uint4* hr = reinterpret_cast<const uint4*>(h12 + r * 2 * I);
    uint4 a4 = __ldg(hr + c8), b4 = __ldg(hr + i8 + c8);
    float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&a4.x));
    float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&a4.y));
    float2 a2 = __half22float2(*reinterpret_cast<const __half2*>(&a4.z));
    float2 a3 = __half22float2(*reinterpret_cast<const __half2*>(&a4.w));
    float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(&b4.x));
    float2 b1 = __half22float2(*reinterpret_cast<const __half2*>(&b4.y));
    float2 b2 = __half22float2(*reinterpret_cast<const __half2*>(&b4.z));
    float2 b3 = __half22float2(*reinterpret_cast<const __half2*>(&b4.w));
    __nv_bfloat162 o0 = __floats2bfloat162_rn(siluF(a0.x) * b0.x, siluF(a0.y) * b0.y);
    __nv_bfloat162 o1 = __floats2bfloat162_rn(siluF(a1.x) * b1.x, siluF(a1.y) * b1.y);
    __nv_bfloat162 o2 = __floats2bfloat162_rn(siluF(a2.x) * b2.x, siluF(a2.y) * b2.y);
    __nv_bfloat162 o3 = __floats2bfloat162_rn(siluF(a3.x) * b3.x, siluF(a3.y) * b3.y);
    uint4 out;
    out.x = *reinterpret_cast<const unsigned*>(&o0);
    out.y = *reinterpret_cast<const unsigned*>(&o1);
    out.z = *reinterpret_cast<const unsigned*>(&o2);
    out.w = *reinterpret_cast<const unsigned*>(&o3);
    reinterpret_cast<uint4*>(g)[t] = out;
    return;
  }
  if ((I & 3) == 0 && (((uintptr_t)h12 | (uintptr_t)g) & 7) == 0) {
    int i4 = I >> 2;
    size_t n4 = rows * (size_t)i4;
    if (t >= n4) return;
    size_t r; int c4;
    if ((i4 & (i4 - 1)) == 0) {
      c4 = (int)(t & (size_t)(i4 - 1));
      r = t >> (__ffs(i4) - 1);
    } else if (i4 == 768 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 768u; r = q; c4 = (int)(u - q * 768u);
    } else if (i4 == 672 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 672u; r = q; c4 = (int)(u - q * 672u);
    } else if (i4 == 384 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 384u; r = q; c4 = (int)(u - q * 384u);
    } else if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, ui = (unsigned)i4, q = u / ui; r = q; c4 = (int)(u - q * ui);
    } else {
      r = t / i4; c4 = (int)(t % i4);
    }
    const uint2* hr = reinterpret_cast<const uint2*>(h12 + r * 2 * I);
    uint2 a2 = __ldg(hr + c4), b2 = __ldg(hr + i4 + c4);
    float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&a2.x));
    float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&a2.y));
    float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(&b2.x));
    float2 b1 = __half22float2(*reinterpret_cast<const __half2*>(&b2.y));
    __nv_bfloat162 o0 = __floats2bfloat162_rn(siluF(a0.x) * b0.x, siluF(a0.y) * b0.y);
    __nv_bfloat162 o1 = __floats2bfloat162_rn(siluF(a1.x) * b1.x, siluF(a1.y) * b1.y);
    uint2 out;
    out.x = *reinterpret_cast<const unsigned*>(&o0);
    out.y = *reinterpret_cast<const unsigned*>(&o1);
    reinterpret_cast<uint2*>(g)[t] = out;
    return;
  }
  if (t >= rows * I) return;
  size_t r = t / I; int c = (int)(t % I);
  g[t] = __float2bfloat16(siluF(__half2float(h12[r * 2 * I + c])) * __half2float(h12[r * 2 * I + I + c]));
}
// y = h sigmoid(g), into f16
__global__ void sigmoidMulToK(const float* __restrict__ h, const float* __restrict__ g, half* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)h | (uintptr_t)g) & 15) == 0 && (((uintptr_t)y & 7) == 0)) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 hv = __ldg(reinterpret_cast<const float4*>(h) + t);
      float4 gv = __ldg(reinterpret_cast<const float4*>(g) + t);
      storeVec4<half>(y, (int)t,
                      hv.x / (1.f + expf(-gv.x)), hv.y / (1.f + expf(-gv.y)),
                      hv.z / (1.f + expf(-gv.z)), hv.w / (1.f + expf(-gv.w)));
    }
    return;
  }
  if (t < n) y[t] = toH<half>(h[t] / (1.f + expf(-g[t])));
}
__global__ void biasRowsK(float* __restrict__ y, const float* __restrict__ b, size_t rows, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (ld & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2, ld4 = ld >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    size_t r; int col4;
    if ((c4 & (c4 - 1)) == 0) {
      col4 = (int)(t & (size_t)(c4 - 1));
      r = t >> (__ffs(c4) - 1);
    } else if (c4 == 192 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 192u; r = q; col4 = (int)(u - q * 192u);
    } else if (c4 == 1152 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 1152u; r = q; col4 = (int)(u - q * 1152u);
    } else if (c4 == 480 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 480u; r = q; col4 = (int)(u - q * 480u);
    } else if (c4 == 640 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 640u; r = q; col4 = (int)(u - q * 640u);
    } else if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, uc = (unsigned)c4, q = u / uc; r = q; col4 = (int)(u - q * uc);
    } else {
      r = t / c4; col4 = (int)(t % c4);
    }
    float4* yp = reinterpret_cast<float4*>(y) + r * ld4 + col4;
    float4 yv = *yp, bv = __ldg(reinterpret_cast<const float4*>(b) + col4);
    *yp = make_float4(yv.x + bv.x, yv.y + bv.y, yv.z + bv.z, yv.w + bv.w);
    return;
  }
  if (t >= rows * C) return;
  y[(t / C) * ld + t % C] += b[t % C];
}
inline void addBiasRows(float* y, const float* b, size_t rows, int C, int ld = 0) {
  int l = ld ? ld : C;
  size_t nt = ((C & 3) == 0 && (l & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0)
              ? rows * (size_t)(C >> 2) : rows * C;
  biasRowsK<<<blocks(nt), 256, 0, STREAM>>>(y, b, rows, C, l);
}
__global__ void biasBatchRowsK(float* __restrict__ y, const float* __restrict__ b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int l = (int)blockIdx.y;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    int col4;
    if (c4 == 192 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t; col4 = (int)(u - (u / 192u) * 192u);
    } else if (c4 == 1152 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t; col4 = (int)(u - (u / 1152u) * 1152u);
    } else if (c4 == 480 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t; col4 = (int)(u - (u / 480u) * 480u);
    } else if (c4 == 640 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t; col4 = (int)(u - (u / 640u) * 640u);
    } else {
      col4 = (int)(t % (size_t)c4);
    }
    float4* yp = reinterpret_cast<float4*>(y) + (size_t)l * n4 + t;
    float4 yv = *yp;
    float4 bv = __ldg(reinterpret_cast<const float4*>(b) + (size_t)l * c4 + col4);
    *yp = make_float4(yv.x + bv.x, yv.y + bv.y, yv.z + bv.z, yv.w + bv.w);
    return;
  }
  if (t >= rows * C) return;
  y[((size_t)l * rows + t / C) * C + (t % C)] += b[(size_t)l * C + (t % C)];
}
inline void addBiasBatchRows(float* y, const float* b, size_t rows, int C, int layers) {
  if (!b) return;
  size_t nt = ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) ? rows * (size_t)(C >> 2) : rows * C;
  biasBatchRowsK<<<dim3(blocks(nt), (unsigned)layers), 256, 0, STREAM>>>(y, b, rows, C);
}
__global__ void siluK(const float* __restrict__ x, float* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)x | (uintptr_t)y) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 v = reinterpret_cast<const float4*>(x)[t];
      reinterpret_cast<float4*>(y)[t] = make_float4(siluF(v.x), siluF(v.y), siluF(v.z), siluF(v.w));
    }
    return;
  }
  if (t < n) y[t] = siluF(x[t]);
}
__global__ void reluK(const float* __restrict__ x, float* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)x | (uintptr_t)y) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 v = reinterpret_cast<const float4*>(x)[t];
      reinterpret_cast<float4*>(y)[t] = make_float4(fmaxf(v.x, 0.f), fmaxf(v.y, 0.f), fmaxf(v.z, 0.f), fmaxf(v.w, 0.f));
    }
    return;
  }
  if (t < n) y[t] = fmaxf(x[t], 0.f);
}
// g = silu(h1) * h2 (two separate projections, fc1 and fc2)
__global__ void siluMulK(const float* __restrict__ h1, const float* __restrict__ h2, float* __restrict__ g, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)h1 | (uintptr_t)h2 | (uintptr_t)g) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 a = reinterpret_cast<const float4*>(h1)[t];
      float4 b = __ldg(reinterpret_cast<const float4*>(h2) + t);
      reinterpret_cast<float4*>(g)[t] = make_float4(siluF(a.x) * b.x, siluF(a.y) * b.y, siluF(a.z) * b.z, siluF(a.w) * b.w);
    }
    return;
  }
  if (t < n) g[t] = siluF(h1[t]) * h2[t];
}
// x[r, c] += gate[map(r) * ldg + c] * h[r, c]   (map null: the identity)
__global__ void gatedAddK(float* __restrict__ x, const float* __restrict__ h, const float* __restrict__ gate,
                          size_t rows, int C, int ldg, const int* __restrict__ map = nullptr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (ldg & 3) == 0 && (((uintptr_t)x | (uintptr_t)h | (uintptr_t)gate) & 15) == 0) {
    int c4 = C >> 2, ldg4 = ldg >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    size_t r; int col4;
    if ((c4 & (c4 - 1)) == 0) {
      col4 = (int)(t & (size_t)(c4 - 1));
      r = t >> (__ffs(c4) - 1);
    } else if (c4 == 192 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 192u; r = q; col4 = (int)(u - q * 192u);
    } else if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, uc = (unsigned)c4, q = u / uc; r = q; col4 = (int)(u - q * uc);
    } else {
      r = t / c4; col4 = (int)(t % c4);
    }
    size_t gr = map ? (size_t)__ldg(map + r) : r;
    float4 xv = reinterpret_cast<const float4*>(x)[t];
    float4 hv = __ldg(reinterpret_cast<const float4*>(h) + t);
    float4 gv = __ldg(reinterpret_cast<const float4*>(gate) + gr * ldg4 + col4);
    reinterpret_cast<float4*>(x)[t] = make_float4(xv.x + gv.x * hv.x, xv.y + gv.y * hv.y, xv.z + gv.z * hv.z, xv.w + gv.w * hv.w);
    return;
  }
  if (t >= rows * C) return;
  size_t r = t / C; int c = (int)(t % C);
  size_t g = map ? (size_t)map[r] : r;
  x[t] += gate[g * ldg + c] * h[t];
}
inline void gatedAdd(float* x, const float* h, const float* gate, size_t rows, int C, int ldg, const int* map) {
  size_t nt = ((C & 3) == 0 && (ldg & 3) == 0 && (((uintptr_t)x | (uintptr_t)h | (uintptr_t)gate) & 15) == 0)
              ? rows * (size_t)(C >> 2) : rows * C;
  gatedAddK<<<blocks(nt), 256, 0, STREAM>>>(x, h, gate, rows, C, ldg, map);
}
// y = h * sigmoid(g), elementwise
__global__ void sigmoidMulK(const float* __restrict__ h, const float* __restrict__ g, float* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)h | (uintptr_t)g | (uintptr_t)y) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 hv = reinterpret_cast<const float4*>(h)[t];
      float4 gv = __ldg(reinterpret_cast<const float4*>(g) + t);
      reinterpret_cast<float4*>(y)[t] = make_float4(hv.x * sigm(gv.x), hv.y * sigm(gv.y), hv.z * sigm(gv.z), hv.w * sigm(gv.w));
    }
    return;
  }
  if (t < n) y[t] = h[t] * sigm(g[t]);
}
// adaLN: y = LN(x) (1 + alpha) + beta, alpha and beta rows of a wider conditioning tensor (stride ldc), x's own
// stride ldx; a warp a row
template <class TY>
__global__ void adaLnK(const float* x, TY* y, size_t rows, int C, int ldx, const float* alpha, const float* beta,
                       int ldc, float scale, const int* map) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * ldx;
  float s = 0; for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / C + 1e-5f);
  const size_t cr = map ? (size_t)map[row] : row;
  for (int c = lane; c < C; c += 32) {
    const float v2 = scale * ((xr[c] - mean) * inv * (1.f + alpha[cr * ldc + c]) + beta[cr * ldc + c]);
    if constexpr (std::is_same_v<TY, float>) y[row * C + c] = v2; else y[row * C + c] = toH<TY>(v2);
  }
}
template <class TY, int K, bool GATED = false>
__global__ void __launch_bounds__(256, (K <= 16 ? 4 : 2))
adaLnRegK(float* __restrict__ x, TY* __restrict__ y, size_t rows, int ldx,
          const float* __restrict__ alpha, const float* __restrict__ beta, int ldc,
          float scale, const int* __restrict__ map,
          const float* __restrict__ h = nullptr, const float* __restrict__ gate = nullptr, int ldg = 0) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  constexpr int C = K * 32, K4 = K / 4;
  const size_t cr = map ? (size_t)__ldg(map + row) : row;
  float4* xr4 = reinterpret_cast<float4*>(x + row * ldx);
  const float4* hr4 = GATED ? reinterpret_cast<const float4*>(h + row * C) : nullptr;
  const float4* gr4 = GATED ? reinterpret_cast<const float4*>(gate + cr * (ldg ? ldg : ldc)) : nullptr;
  float4 xv[K4];
  float s = 0.f;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    int c4 = k * 32 + lane;
    float4 v = __ldg(xr4 + c4);
    if constexpr (GATED) {
      float4 gv = __ldg(gr4 + c4);
      float4 hv = __ldg(hr4 + c4);
      v.x += gv.x * hv.x;
      v.y += gv.y * hv.y;
      v.z += gv.z * hv.z;
      v.w += gv.w * hv.w;
      xr4[c4] = v;
    }
    xv[k] = v;
    s += (v.x + v.y) + (v.z + v.w);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, var = 0.f;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    float dx = xv[k].x - mean, dy = xv[k].y - mean, dz = xv[k].z - mean, dw = xv[k].w - mean;
    var += (dx * dx + dy * dy) + (dz * dz + dw * dw);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
  float inv = rsqrtf(var / C + 1e-5f);
  const float4* ar4 = reinterpret_cast<const float4*>(alpha + cr * ldc);
  const float4* br4 = reinterpret_cast<const float4*>(beta + cr * ldc);
  TY* yr = y + row * C;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    int c4 = k * 32 + lane;
    float4 av = __ldg(ar4 + c4);
    float4 bv = __ldg(br4 + c4);
    storeVec4<TY>(yr, c4,
                  scale * ((xv[k].x - mean) * inv * (1.f + av.x) + bv.x),
                  scale * ((xv[k].y - mean) * inv * (1.f + av.y) + bv.y),
                  scale * ((xv[k].z - mean) * inv * (1.f + av.z) + bv.z),
                  scale * ((xv[k].w - mean) * inv * (1.f + av.w) + bv.w));
  }
}
template <class TY = float>
inline void adaLn(const float* x, TY* y, size_t rows, int C, const float* alpha, const float* beta, int ldc,
                  int ldx = 0, float scale = 1.f, const int* map = nullptr) {
  int lx = ldx ? ldx : C;
  unsigned g = (unsigned)((rows + 7) / 8);
  float* xm = const_cast<float*>(x);
  if (C == 128) { adaLnRegK<TY, 4><<<g, 256, 0, STREAM>>>(xm, y, rows, lx, alpha, beta, ldc, scale, map); return; }
  if (C == 768) { adaLnRegK<TY, 24><<<g, 256, 0, STREAM>>>(xm, y, rows, lx, alpha, beta, ldc, scale, map); return; }
  if (C == 896) { adaLnRegK<TY, 28><<<g, 256, 0, STREAM>>>(xm, y, rows, lx, alpha, beta, ldc, scale, map); return; }
  adaLnK<TY><<<g, 256, 0, STREAM>>>(x, y, rows, C, lx, alpha, beta, ldc, scale, map);
}
template <class TY = float>
inline void adaLnGatedAdd(float* x, const float* h, const float* gate, TY* y, size_t rows, int C,
                          const float* alpha, const float* beta, int ldc, const int* map = nullptr, int ldg = 0) {
  int lg = ldg ? ldg : ldc;
  unsigned g = (unsigned)((rows + 7) / 8);
  if (C == 128) { adaLnRegK<TY, 4, true><<<g, 256, 0, STREAM>>>(x, y, rows, C, alpha, beta, ldc, 1.f, map, h, gate, lg); return; }
  if (C == 768) { adaLnRegK<TY, 24, true><<<g, 256, 0, STREAM>>>(x, y, rows, C, alpha, beta, ldc, 1.f, map, h, gate, lg); return; }
  gatedAdd(x, h, gate, rows, C, lg, map);
  adaLn<TY>(x, y, rows, C, alpha, beta, ldc, 0, 1.f, map);
}
// Fused q += gateA * tAtom (in place) + [tokensToAtoms(a) | q] (768 + 128 = 896) + adaLN directly into y (f32 or f16)
template <class TY>
__global__ void __launch_bounds__(256, 2)
adaLnS2QK(const float* __restrict__ a, float* __restrict__ q,
          const float* __restrict__ tAtom, const float* __restrict__ gateA, int ldgA,
          const int* __restrict__ atomTok, const int* __restrict__ atomPad,
          TY* __restrict__ y, size_t rows, int N, int T,
          const float* __restrict__ alpha, const float* __restrict__ beta, int ldc,
          const int* __restrict__ map) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  constexpr int K4 = 7, C = 896;
  int s = (int)(row / (size_t)N), al = (int)(row - (size_t)s * N);
  int pad = __ldg(atomPad + al);
  int tokIdx = pad ? __ldg(atomTok + al) : 0;
  const size_t cr = map ? (size_t)__ldg(map + row) : row;
  const float4* ar4 = reinterpret_cast<const float4*>(a + ((size_t)s * T + tokIdx) * 768);
  float4* qr4 = reinterpret_cast<float4*>(q + row * 128);
  const float4* tar4 = reinterpret_cast<const float4*>(tAtom + row * 128);
  const float4* gar4 = reinterpret_cast<const float4*>(gateA + cr * ldgA);
  float4 xv[K4];
  float sum = 0.f;
  #pragma unroll
  for (int k = 0; k < 6; ++k) {
    float4 v = pad ? __ldg(ar4 + k * 32 + lane) : make_float4(0.f, 0.f, 0.f, 0.f);
    xv[k] = v;
    sum += (v.x + v.y) + (v.z + v.w);
  }
  {
    float4 qv = __ldg(qr4 + lane), tav = __ldg(tar4 + lane), gav = __ldg(gar4 + lane);
    float4 v = make_float4(qv.x + gav.x * tav.x, qv.y + gav.y * tav.y,
                           qv.z + gav.z * tav.z, qv.w + gav.w * tav.w);
    qr4[lane] = v;
    xv[6] = v;
    sum += (v.x + v.y) + (v.z + v.w);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  float mean = sum / C, var = 0.f;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    float dx = xv[k].x - mean, dy = xv[k].y - mean, dz = xv[k].z - mean, dw = xv[k].w - mean;
    var += (dx * dx + dy * dy) + (dz * dz + dw * dw);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
  float inv = rsqrtf(var / C + 1e-5f);
  const float4* alr4 = reinterpret_cast<const float4*>(alpha + cr * ldc);
  const float4* ber4 = reinterpret_cast<const float4*>(beta + cr * ldc);
  TY* yr = y + row * C;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    int c4 = k * 32 + lane;
    float4 av = __ldg(alr4 + c4);
    float4 bv = __ldg(ber4 + c4);
    storeVec4<TY>(yr, c4,
                  (xv[k].x - mean) * inv * (1.f + av.x) + bv.x,
                  (xv[k].y - mean) * inv * (1.f + av.y) + bv.y,
                  (xv[k].z - mean) * inv * (1.f + av.z) + bv.z,
                  (xv[k].w - mean) * inv * (1.f + av.w) + bv.w);
  }
}
// Fused a += gateT * tTok (in place) + [centreAtoms(q + gateA * tAtom) | a] (128 + 768 = 896) + adaLN directly into y (f32 or f16)
template <class TY>
__global__ void __launch_bounds__(256, 2)
adaLnQ2SK(const float* __restrict__ q, const float* __restrict__ tAtom,
          const float* __restrict__ gateA, int ldgA, const int* __restrict__ atomTok,
          float* __restrict__ a, const float* __restrict__ tTok,
          const float* __restrict__ gateT, int ldgT,
          const int* __restrict__ centre,
          TY* __restrict__ y, size_t rows, int N, int T,
          const float* __restrict__ alpha, const float* __restrict__ beta, int ldc,
          const int* __restrict__ map) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  constexpr int K4 = 7, C = 896;
  int s = (int)(row / (size_t)T), il = (int)(row - (size_t)s * T);
  int caIdx = __ldg(centre + il);
  size_t caRow = (size_t)s * N + caIdx;
  int caTok = __ldg(atomTok + caIdx);
  size_t caCr = caTok >= 0 ? (size_t)caTok : (size_t)T;
  const size_t cr = map ? (size_t)__ldg(map + row) : row;
  const float4* qr4 = reinterpret_cast<const float4*>(q + caRow * 128);
  const float4* tar4 = reinterpret_cast<const float4*>(tAtom + caRow * 128);
  const float4* gar4 = reinterpret_cast<const float4*>(gateA + caCr * ldgA);
  float4* ar4 = reinterpret_cast<float4*>(a + row * 768);
  const float4* ttr4 = reinterpret_cast<const float4*>(tTok + row * 768);
  const float4* gtr4 = reinterpret_cast<const float4*>(gateT + cr * ldgT);
  float4 xv[K4];
  float sum = 0.f;
  {
    float4 qv = __ldg(qr4 + lane), tav = __ldg(tar4 + lane), gav = __ldg(gar4 + lane);
    float4 v = make_float4(qv.x + gav.x * tav.x, qv.y + gav.y * tav.y,
                           qv.z + gav.z * tav.z, qv.w + gav.w * tav.w);
    xv[0] = v;
    sum += (v.x + v.y) + (v.z + v.w);
  }
  #pragma unroll
  for (int k = 0; k < 6; ++k) {
    int c4 = k * 32 + lane;
    float4 av = __ldg(ar4 + c4), ttv = __ldg(ttr4 + c4), gtv = __ldg(gtr4 + c4);
    float4 v = make_float4(av.x + gtv.x * ttv.x, av.y + gtv.y * ttv.y,
                           av.z + gtv.z * ttv.z, av.w + gtv.w * ttv.w);
    ar4[c4] = v;
    xv[1 + k] = v;
    sum += (v.x + v.y) + (v.z + v.w);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  float mean = sum / C, var = 0.f;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    float dx = xv[k].x - mean, dy = xv[k].y - mean, dz = xv[k].z - mean, dw = xv[k].w - mean;
    var += (dx * dx + dy * dy) + (dz * dz + dw * dw);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_xor_sync(~0u, var, o);
  float inv = rsqrtf(var / C + 1e-5f);
  const float4* alr4 = reinterpret_cast<const float4*>(alpha + cr * ldc);
  const float4* ber4 = reinterpret_cast<const float4*>(beta + cr * ldc);
  TY* yr = y + row * C;
  #pragma unroll
  for (int k = 0; k < K4; ++k) {
    int c4 = k * 32 + lane;
    float4 av = __ldg(alr4 + c4);
    float4 bv = __ldg(ber4 + c4);
    storeVec4<TY>(yr, c4,
                  (xv[k].x - mean) * inv * (1.f + av.x) + bv.x,
                  (xv[k].y - mean) * inv * (1.f + av.y) + bv.y,
                  (xv[k].z - mean) * inv * (1.f + av.z) + bv.z,
                  (xv[k].w - mean) * inv * (1.f + av.w) + bv.w);
  }
}
// rows of a token tensor broadcast to atoms (zero on padding atoms); and each token's centre atom (zero on
// padding tokens)
// (several samples stacked: `samples` blocks of N atoms and of T tokens)
__global__ void tokensToAtomsK(const float* __restrict__ tok, const int* __restrict__ atomTok,
                               const int* __restrict__ atomPad, float* __restrict__ out,
                               size_t N, int T, int C, int ldo, float beta, int samples) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (ldo & 3) == 0 && (((uintptr_t)tok | (uintptr_t)out) & 15) == 0) {
    int c4 = C >> 2, ldo4 = ldo >> 2;
    size_t n4 = (size_t)samples * N * c4;
    if (t >= n4) return;
    size_t a; int col4;
    if ((c4 & (c4 - 1)) == 0) {
      col4 = (int)(t & (size_t)(c4 - 1));
      a = t >> (__ffs(c4) - 1);
    } else if (c4 == 192 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 192u; a = q; col4 = (int)(u - q * 192u);
    } else if (c4 == 96 && n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / 96u; a = q; col4 = (int)(u - q * 96u);
    } else {
      a = t / c4; col4 = (int)(t % c4);
    }
    size_t s = a / N, al = a - s * N;
    float4 v = __ldg(atomPad + al)
               ? __ldg(reinterpret_cast<const float4*>(tok) + (s * T + __ldg(atomTok + al)) * c4 + col4)
               : make_float4(0.f, 0.f, 0.f, 0.f);
    float4* op = reinterpret_cast<float4*>(out) + a * ldo4 + col4;
    if (beta == 0.f) *op = v;
    else {
      float4 ov = *op;
      *op = make_float4(beta * ov.x + v.x, beta * ov.y + v.y, beta * ov.z + v.z, beta * ov.w + v.w);
    }
    return;
  }
  if (t >= (size_t)samples * N * C) return;
  size_t a = t / C, s = a / N, al = a % N; int c = (int)(t % C);
  float v = atomPad[al] ? tok[(s * T + atomTok[al]) * C + c] : 0.f;
  out[a * ldo + c] = beta == 0.f ? v : beta * out[a * ldo + c] + v;
}
inline void tokensToAtoms(const float* tok, float* out, size_t N, int C, int ldo = 0, float beta = 0.f, int samples = 1,
                          int T = 0) {
  int l = ldo ? ldo : C;
  size_t nt = ((C & 3) == 0 && (l & 3) == 0 && (((uintptr_t)tok | (uintptr_t)out) & 15) == 0)
              ? (size_t)samples * N * (C >> 2) : (size_t)samples * N * C;
  tokensToAtomsK<<<blocks(nt), 256, 0, STREAM>>>(tok, Idev("token_idx_at_atom"),
      Idev("atom_pad_mask"), out, N, T, C, l, beta, samples);
}
__global__ void centreAtomsK(const float* atom, const int* centre, float* out, int T, int N, int C, int ldo, int samples) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)samples * T * C) return;
  size_t i = t / C, s = i / T, il = i % T; int c = (int)(t % C);
  out[i * ldo + c] = atom[(s * N + centre[il]) * C + c];   // (every token here is real: no token padding)
}
inline void centreAtoms(const float* atom, float* out, int T, int C, int ldo = 0, int samples = 1, int N = 0) {
  centreAtomsK<<<blocks((size_t)samples * T * C), 256, 0, STREAM>>>(atom, Idev("token_center_atom_index"), out, T, N, C,
                                                                    ldo ? ldo : C, samples);
}

// ---------------------------------------------------------------- the SwiGLU transition
// out = fc3(silu(fc1 x') * fc2 x'), x' = LN(x) when the module has a norm; with `residual` the output is added to
// `out` (beta 1). A chunk of rows at a time: the widened rows are 2I a row.
// GEMM_HALF: the input in f16 (from the caller's kernel, `xh`, or normed/converted here), fc1 | fc2 as ONE GEMM over
// their stacked mirror, the SwiGLU into fc3's bf16 input
inline void transitionHalf(const float* x, const half* xh, float* out, size_t rows, int C, const std::string& p, int layer,
                           bool residual, bool norm, int ldx) {
  auto w = [&](const std::string& n) { return layer < 0 ? P(p + n) : PL(p + n, layer); };
  const int I = layer < 0 ? dimP(p + "fc1/weight", 0) : dimP(p + "fc1/weight", 1);
  const int Co = layer < 0 ? dimP(p + "fc3/weight", 0) : dimP(p + "fc3/weight", 1);
  const half* w12 = stackedMirror<half>(w("fc1/weight"), w("fc2/weight"), (size_t)I * C);
  const __nv_bfloat16* w3 = halfMirror<__nv_bfloat16>(w("fc3/weight"), (size_t)Co * I);
  // (a chunk sized for the L2 - 16 MB of widened rows - measured 8% SLOWER at 494 tokens than this: the GEMMs lose more
  // to their smaller M than the SwiGLU gains from the cache)
  size_t chunk = std::max<size_t>(1, ((size_t)128 << 20) / (4 * (size_t)I * 2));
  if (!xh) {
    half* c = scratch<half>("be2.tr.xh", rows * C);
    if (norm) lnTo<half>(x, c, rows, C, w("norm/weight"), w("norm/bias"), ldx);
    else toHalfRows<half>(x, c, rows, C, ldx);
    xh = c;
  }
  // (fc1 | fc2's output in bf16: the SwiGLU reads it once and it is the widest tensor of the layer - written f32 it was
  // 9 ms of an 82 ms step at 494 tokens)
  half* h12 = scratch<half>("be2.tr.h12", std::min(rows, chunk) * 2 * I);
  __nv_bfloat16* g = scratch<__nv_bfloat16>("be2.tr.g", std::min(rows, chunk) * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    {
      const float one = 1.f, zero = 0.f;
      CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, 2 * I, (int)r, C, &one, w12, CUDA_R_16F, C, xh + r0 * C, CUDA_R_16F, C,
                      &zero, h12, CUDA_R_16F, 2 * I, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      if (getenv("BE2_DEBUG_HALF")) {
        CK(cudaStreamSynchronize(STREAM));
        std::vector<half> hh(r * 2 * I); CK(cudaMemcpy(hh.data(), h12, hh.size() * 2, cudaMemcpyDeviceToHost));
        float mx = 0; for (auto v : hh) mx = fmaxf(mx, fabsf(__half2float(v)));
        static float worst = 0;
        if (mx > worst) { worst = mx; printf("fc1|fc2 max %g (%s)\n", mx, p.c_str()); }
      }
    }
    size_t nSw = ((I & 7) == 0 && (((uintptr_t)h12 | (uintptr_t)g) & 15) == 0) ? r * (size_t)(I >> 3)
               : ((I & 3) == 0 && (((uintptr_t)h12 | (uintptr_t)g) & 7) == 0) ? r * (size_t)(I >> 2) : r * I;
    swigluToK<<<blocks(nSw), 256, 0, STREAM>>>(h12, g, r, I);
    linearPre<__nv_bfloat16>(g, w3, out + r0 * Co, r, I, Co, residual ? 1.f : 0.f);
  }
}
inline void transition(const float* x, float* out, size_t rows, int C, const std::string& p, int layer = -1,
                       bool residual = true, bool norm = true, int ldx = 0) {
  if (GEMM_HALF) { transitionHalf(x, nullptr, out, rows, C, p, layer, residual, norm, ldx); return; }
  auto w = [&](const std::string& n) { return layer < 0 ? P(p + n) : PL(p + n, layer); };
  const int I = layer < 0 ? dimP(p + "fc1/weight", 0) : dimP(p + "fc1/weight", 1);
  const int Co = layer < 0 ? dimP(p + "fc3/weight", 0) : dimP(p + "fc3/weight", 1);   // (a mixer narrows: 896 -> 128)
  const float* xin = x; int ld = ldx ? ldx : C;
  if (norm) {
    float* xn = scratch<float>("be2.tr.xn", rows * C);
    layerNorm(x, xn, rows, C, w("norm/weight"), w("norm/bias"), 1e-5f, ld, C);
    xin = xn; ld = C;
  }
  size_t chunk = std::max<size_t>(1, ((size_t)96 << 20) / (4 * (size_t)I * 3));
  float* h1 = scratch<float>("be2.tr.h1", std::min(rows, chunk) * I);
  float* h2 = scratch<float>("be2.tr.h2", std::min(rows, chunk) * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    linear(xin + r0 * ld, w("fc1/weight"), h1, r, C, I, 0.f, ld);
    linear(xin + r0 * ld, w("fc2/weight"), h2, r, C, I, 0.f, ld);
    size_t nSm = ((r * I & 3) == 0 && (((uintptr_t)h1 | (uintptr_t)h2) & 15) == 0) ? (r * I) >> 2 : r * I;
    siluMulK<<<blocks(nSm), 256, 0, STREAM>>>(h1, h2, h1, r * I);
    linear(h1, w("fc3/weight"), out + r0 * Co, r, I, Co, residual ? 1.f : 0.f, 0, 0, 1.f, true);
  }
}

// ---------------------------------------------------------------- the relative position encoding
// FusedRelativePositionEncoder over the token index (arange), into `out` (+= when add)
struct RelTables { const float *pos, *tok, *ent, *chain; };
inline RelTables relTables(const std::string& p) {
  return {P(p + "embed_rel_pos/weight"), P(p + "embed_rel_token/weight"), P(p + "embed_same_entity/weight"),
          P(p + "embed_rel_chain/weight")};
}
__global__ void relPosK(RelTables w, const int* __restrict__ asym, const int* __restrict__ ent,
                        const int* __restrict__ res, const int* __restrict__ sym, float* __restrict__ out,
                        int T, int C, bool add, int ldo) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int rmax = 32, smax = 2;
  auto clip = [](int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; };
  if ((C & 3) == 0 && (ldo & 3) == 0 && (((uintptr_t)out & 15) == 0)) {
    int c4 = C >> 2, ldo4 = ldo >> 2;
    size_t n4 = (size_t)T * T * c4;
    if (t >= n4) return;
    int col4 = (int)(t % c4); size_t ij = t / c4; int i = (int)(ij / T), j = (int)(ij % T);
    bool sameChain = asym[i] == asym[j], sameRes = res[i] == res[j], sameEnt = ent[i] == ent[j];
    int dRes = sameChain ? clip(res[i] - res[j] + rmax, 0, 2 * rmax) : 2 * rmax + 1;
    int dTok = sameChain && sameRes ? clip(i - j + rmax, 0, 2 * rmax) : 2 * rmax + 1;
    int dChain = sameEnt ? clip(sym[i] - sym[j] + smax, 0, 2 * smax) : 2 * smax + 1;
    float4 pv = __ldg(reinterpret_cast<const float4*>(w.pos) + (size_t)dRes * c4 + col4);
    float4 tv = __ldg(reinterpret_cast<const float4*>(w.tok) + (size_t)dTok * c4 + col4);
    float4 ev = sameEnt ? __ldg(reinterpret_cast<const float4*>(w.ent) + col4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float4 cv = __ldg(reinterpret_cast<const float4*>(w.chain) + (size_t)dChain * c4 + col4);
    float4 v = make_float4(pv.x + tv.x + ev.x + cv.x, pv.y + tv.y + ev.y + cv.y,
                           pv.z + tv.z + ev.z + cv.z, pv.w + tv.w + ev.w + cv.w);
    float4* op = reinterpret_cast<float4*>(out) + ij * ldo4 + col4;
    if (add) { float4 ov = *op; *op = make_float4(ov.x + v.x, ov.y + v.y, ov.z + v.z, ov.w + v.w); }
    else *op = v;
    return;
  }
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  bool sameChain = asym[i] == asym[j], sameRes = res[i] == res[j], sameEnt = ent[i] == ent[j];
  int dRes = sameChain ? clip(res[i] - res[j] + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  int dTok = sameChain && sameRes ? clip(i - j + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  int dChain = sameEnt ? clip(sym[i] - sym[j] + smax, 0, 2 * smax) : 2 * smax + 1;
  float v = w.pos[(size_t)dRes * C + c] + w.tok[(size_t)dTok * C + c] + (sameEnt ? w.ent[c] : 0.f)
            + w.chain[(size_t)dChain * C + c];
  float* o = out + ij * ldo + c;
  *o = add ? *o + v : v;
}
inline void relPos(const std::string& p, float* out, int T, int C, bool add, int ldo = 0) {
  int l = ldo ? ldo : C;
  size_t nt = ((C & 3) == 0 && (l & 3) == 0 && (((uintptr_t)out & 15) == 0))
              ? (size_t)T * T * (C >> 2) : (size_t)T * T * C;
  relPosK<<<blocks(nt), 256, 0, STREAM>>>(relTables(p), Idev("asym_id"), Idev("entity_id"),
      Idev("residue_index"), Idev("sym_id"), out, T, C, add, l);
}

}  // namespace be2
