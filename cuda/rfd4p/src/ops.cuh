// The four pieces of RFD4-Proteina that cuda/ does not already have: RMSNorm, its adaptive LayerNorm,
// its output scale, and a SwiGLU whose two halves are the other way round.
//
// Everything else is reused - `gemm`/`linear` (cuda/ef2/src/ops.cuh, cuda/af3/src/common.cuh),
// `layerNorm` for the pair track's LayerNorms, `flashGrid` for the attention.
#pragma once
#include <set>
#include "../../ef2/src/ops.cuh"

namespace rfd4p {

// The compute type for the two GEMMs this port issues directly (the triangle's contraction and the
// insertion head). `linear()` reads `F32_TF32` itself; these have to ask.
//
// 🔴 TF32 IS NOT f32 AND THE ORACLE BOUND MOVES WITH IT. Ten mantissa bits against twenty-three, so a
// trunk that reads 6e-7 exact reads ~1e-3 here. That is the reason it is a FLAG and the strict path
// stays the default: the parity gate runs the exact arm, and --fast is held to a looser bar.
inline cublasComputeType_t computeType() {
  return F32_TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

// ---------------------------------------------------------------- RMSNorm
//
// 🔴 NOT LayerNorm WITH THE MEAN LEFT IN. `rmsnorm_torch` divides by sqrt(mean(x^2) + eps) and never
// centres, so it is `ops.cuh`'s `layerNormRegK` with the mean reduction deleted - the register staging
// and the C/32 unroll are that kernel's. `scale` may be null (the adaptive LN normalises without one).
template <int MAX_K, bool EXACT, class TY = float>
__global__ void __launch_bounds__(256, (MAX_K <= 16 && EXACT) ? 4 : 2)
rmsNormRegK(const float* __restrict__ x, TY* __restrict__ y, size_t rows, int C,
            const float* __restrict__ scale, float eps, int ldx, int ldy) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * ldx;
  TY* yr = y + row * ldy;
  float xv[MAX_K];
  float v = 0;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    int c = k * 32 + lane;
    float val = (EXACT || c < C) ? __ldg(xr + c) : 0.f;
    xv[k] = val;
    v += val * val;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / (float)C + eps);
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    int c = k * 32 + lane;
    if (EXACT || c < C) {
      float r = xv[k] * inv;
      if (scale) r *= __ldg(scale + c);
      if constexpr (std::is_same_v<TY, half>) yr[c] = __float2half(r);
      else yr[c] = r;
    }
  }
}

// 🔴 ef2's `layerNorm` AND THIS FILE'S `rmsNorm` BOTH TAKE A float INPUT, which is the one thing
// standing between the pair tensor and half. These are the same register kernels with the input type
// templated: the reduction and the arithmetic stay f32, only the loads narrow.
template <int MAX_K, class TI, class TO>
__global__ void __launch_bounds__(256, 2)
layerNormInK(const TI* __restrict__ x, TO* __restrict__ y, size_t rows, int C,
             const float* __restrict__ scale, const float* __restrict__ offset, float eps) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const TI* xr = x + row * C;
  float xv[MAX_K], s = 0.f;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    xv[k] = (float)xr[k * 32 + lane];
    s += xv[k];
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  const float mean = s / (float)C;
  float v = 0.f;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) { const float d = xv[k] - mean; v += d * d; }
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  const float inv = rsqrtf(v / (float)C + eps);
  TO* yr = y + row * C;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    const int c = k * 32 + lane;
    float r = (xv[k] - mean) * inv;
    if (scale) r *= __ldg(scale + c);
    if (offset) r += __ldg(offset + c);
    yr[c] = (TO)r;
  }
}

template <int MAX_K, class TI, class TO>
__global__ void __launch_bounds__(256, 2)
rmsNormInK(const TI* __restrict__ x, TO* __restrict__ y, size_t rows, int C,
           const float* __restrict__ scale, float eps) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const TI* xr = x + row * C;
  float xv[MAX_K], v = 0.f;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    xv[k] = (float)xr[k * 32 + lane];
    v += xv[k] * xv[k];
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  const float inv = rsqrtf(v / (float)C + eps);
  TO* yr = y + row * C;
  #pragma unroll
  for (int k = 0; k < MAX_K; ++k) {
    const int c = k * 32 + lane;
    float r = xv[k] * inv;
    if (scale) r *= __ldg(scale + c);
    yr[c] = (TO)r;
  }
}

// Dispatch on the width; every width this model norms over divides by 32.
#define RFD4P_NORM_IN(NAME, KERNEL, ARGS)                                                        \
  template <class TI, class TO>                                                                  \
  inline void NAME(const TI* x, TO* y, size_t rows, int C, const float* scale,                   \
                   const float* offset, float eps = 1e-5f) {                                     \
    const unsigned b = blocks(rows * 32, 256);                                                   \
    (void)offset;                                                                                \
    switch (C) {                                                                                 \
      case 768: KERNEL<24, TI, TO><<<b, 256, 0, STREAM>>> ARGS; return;                          \
      case 384: KERNEL<12, TI, TO><<<b, 256, 0, STREAM>>> ARGS; return;                          \
      case 256: KERNEL<8, TI, TO><<<b, 256, 0, STREAM>>> ARGS; return;                           \
      default: fprintf(stderr, #NAME ": no kernel for %d channels\n", C); exit(1);               \
    }                                                                                            \
  }
RFD4P_NORM_IN(layerNormIn, layerNormInK, (x, y, rows, C, scale, offset, eps))
RFD4P_NORM_IN(rmsNormIn, rmsNormInK, (x, y, rows, C, scale, eps))
#undef RFD4P_NORM_IN

// The generic form, a warp a row, for a width the register kernel does not cover.
template <class TY = float>
__global__ void rmsNormK(const float* x, TY* y, size_t rows, int C, const float* scale, float eps,
                         int ldx, int ldy) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * ldx;
  float v = 0;
  for (int c = lane; c < C; c += 32) v += xr[c] * xr[c];
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / (float)C + eps);
  for (int c = lane; c < C; c += 32) {
    float r = xr[c] * inv;
    if (scale) r *= scale[c];
    if constexpr (std::is_same_v<TY, half>) y[row * ldy + c] = __float2half(r);
    else y[row * ldy + c] = r;
  }
}

template <class TY = float>
inline void rmsNorm(const float* x, TY* y, size_t rows, int C, const float* scale, float eps = 1e-5f,
                    int ldx = 0, int ldy = 0) {
  if (!ldx) ldx = C;
  if (!ldy) ldy = C;
  unsigned b = blocks(rows * 32, 256);
  // 768, 384, 256 and 32 are every width this model norms over, and all divide by 32.
  #define RFD4P_RMS(K) \
    if (C == (K) * 32) { rmsNormRegK<(K), true, TY><<<b, 256, 0, STREAM>>>(x, y, rows, C, scale, eps, ldx, ldy); return; }
  RFD4P_RMS(24)   // 768
  RFD4P_RMS(12)   // 384
  RFD4P_RMS(8)    // 256
  RFD4P_RMS(1)    // 32
  #undef RFD4P_RMS
  rmsNormK<TY><<<b, 256, 0, STREAM>>>(x, y, rows, C, scale, eps, ldx, ldy);
}

// ---------------------------------------------------------------- adaptive LayerNorm
//
// `out[r][d] = RMSNorm(x)[r][d] * sigmoid(gamma[cr][d]) + beta[cr][d]`, where gamma and beta are two
// linears of `RMSNorm(cond; norm_cond.weight)` and `cr` is `r` when the conditioning is per token and 0
// when one row is broadcast over every pair. A masked-off row is left at zero.
//
// 🔴 THIS IS NOT AF3's adaLN AND NOT DiT's SIX-WAY SPLIT. AF3 (cuda/af3/src/atom.cuh) packs the scale
// and shift under one `SingleCond*` family and chai-1 uses `(scale + 1) * LN(x)`; ESMFold2
// (cuda/ef2/src/atoms.cuh) takes one `(C, 6C)` matrix off `silu(c)`. RFD4-Proteina has two separate
// matrices, a sigmoid on the scale, no `+1`, and no silu - `rfproteina/nn/modules/adaptive_ln_scale.py`.
__global__ void adaLnApplyK(const float* __restrict__ normedX, const float* __restrict__ gamma,
                            const float* __restrict__ gammaBias, const float* __restrict__ beta,
                            const float* __restrict__ mask, float* __restrict__ out, size_t rows,
                            int C, bool broadcastCond) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  size_t r = i / C;
  int c = (int)(i - r * C);
  if (mask && mask[r] == 0.f) { out[i] = 0.f; return; }
  size_t ci = (broadcastCond ? 0 : r) * (size_t)C + c;
  float g = 1.f / (1.f + __expf(-(gamma[ci] + (gammaBias ? gammaBias[c] : 0.f))));
  out[i] = normedX[i] * g + beta[ci];
}

// `residual += x * sigmoid(gamma + bias)` - AdaptiveOutputScale AND the residual add it always feeds,
// in one pass. Keeping them apart cost two launches over [n, 768] per half-layer and a round trip
// through memory for a tensor nothing else reads.
__global__ void adaScaleAddK(const float* __restrict__ x, const float* __restrict__ gamma,
                             const float* __restrict__ bias, const float* __restrict__ mask,
                             float* __restrict__ residual, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  if (mask && mask[i / C] == 0.f) return;
  const float g = gamma[i] + (bias ? bias[i % C] : 0.f);
  residual[i] += x[i] * (1.f / (1.f + __expf(-g)));
}

// ---------------------------------------------------------------- SwiGLU
//
// The same, with one broadcast gamma row - the decoder's conditioning is constant across tokens.
__global__ void adaScaleAddBroadcastK(const float* __restrict__ x, const float* __restrict__ gamma,
                                      const float* __restrict__ mask, float* __restrict__ residual,
                                      size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  size_t r = i / C;
  if (mask && mask[r] == 0.f) return;
  residual[i] += x[i] * (1.f / (1.f + __expf(-gamma[i - r * C])));
}

// 🔴 THE HALVES ARE THE OTHER WAY ROUND FROM `ops.cuh`'s `swigluK`, WHICH IS `silu(a) * b`. RFD4's
// `swish_linear.0.weight` is `[2 * inner, dim]` chunked into `(xPart, gates)`, and the product is
// `silu(gates) * xPart` - the silu is on the SECOND half (`seq_transition_af3.py`). Reusing the other
// kernel returns a tensor of the right shape and magnitude and the wrong answer.
__global__ void swiGluSecondK(const float* __restrict__ wide, float* __restrict__ gated, size_t rows, int F) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)F) return;
  size_t r = i / F;
  int c = (int)(i - r * F);
  const float* w = wide + r * 2 * (size_t)F;
  float xPart = w[c];
  float gate = w[F + c];
  gated[i] = (gate / (1.f + __expf(-gate))) * xPart;
}

// ---------------------------------------------------------------- the attention's pair bias
//
// RMSNorm the pair row, project it to `heads` numbers, and write them in flash's bias layout - one
// kernel, one warp a pair, where it used to be a norm over [pairs, 384], a GEMM whose output is
// twelve columns wide, and a layout pass. The normed pair tensor was materialised (15 MB written and
// read back a layer, 84 layers a pass) for the sake of 12 numbers a row, and the GEMM that consumed
// it ran at 3.2 TFLOP/s because its N is 12.
//
// A lane holds C/32 channels in registers, so the row is read once: the sum of squares and all
// `heads` dot products come off the same registers.
template <class TZ, class T, int K, int HEADS>
__global__ void __launch_bounds__(256)
pairBiasK(const TZ* __restrict__ z, const float* __restrict__ normW,
          const float* __restrict__ biasW, const float* __restrict__ pairMask, T* __restrict__ bias,
          int n, int stride, float scale) {
  constexpr int C = K * 32;
  constexpr int heads = HEADS;
  // 🔴 THE PROJECTION IS NOT STAGED IN SHARED MEMORY. At [12, 384] it is 18 KB, which with eight
  // warps a block caps this kernel at two blocks an SM; every lane reads the same words every pair,
  // so L1 broadcasts them and the occupancy is worth more than the staging.
  const size_t pair = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (pair >= (size_t)n * n) return;
  const int lane = threadIdx.x & 31;
  const int i = (int)(pair / n), j = (int)(pair % n);

  // 🔴 FOUR CHANNELS A LANE, NOT ONE. A lane strided by 32 reads one word a transaction and this
  // kernel sat 2.3x off its own bandwidth; with `float4` a warp's load is one 128-byte line. C is a
  // multiple of 128 for every width this runs at, so the mapping is exact.
  constexpr int Q = K / 4;                                  // four channels a lane
  float4 v4[Q];
  float sq = 0.f;
  const TZ* zr = z + pair * C;
  const float4* nw = (const float4*)normW;
  #pragma unroll
  for (int q = 0; q < Q; ++q) {
    if constexpr (std::is_same_v<TZ, float>) {
      v4[q] = __ldg((const float4*)zr + q * 32 + lane);
    } else {
      // Four halves is one 8-byte load; `__half2` pairs widen without touching memory twice.
      const __half2* h = (const __half2*)(zr + (q * 32 + lane) * 4);
      const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
      v4[q] = make_float4(a.x, a.y, b.x, b.y);
    }
    sq += v4[q].x * v4[q].x + v4[q].y * v4[q].y + v4[q].z * v4[q].z + v4[q].w * v4[q].w;
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) sq += __shfl_xor_sync(~0u, sq, o);
  const float inv = rsqrtf(sq / (float)C + 1e-5f);
  #pragma unroll
  for (int q = 0; q < Q; ++q) {
    const float4 w = __ldg(nw + q * 32 + lane);
    v4[q].x *= inv * w.x; v4[q].y *= inv * w.y; v4[q].z *= inv * w.z; v4[q].w *= inv * w.w;
  }

  // 🔴 ALL `heads` BUTTERFLIES AT ONCE, NOT ONE HEAD AT A TIME. A warp reduction is five dependent
  // shuffles; done in a loop the twelve of them serialise into sixty latencies and the kernel runs
  // 4.5x off its own bandwidth. Unrolled over a compile-time head count they are twelve independent
  // chains and the scheduler interleaves them.
  const bool dead = pairMask && pairMask[pair] == 0.f;
  float dot[HEADS];
  #pragma unroll
  for (int h = 0; h < HEADS; ++h) {
    dot[h] = 0.f;
    const float4* bw = (const float4*)(biasW + h * C);
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
      const float4 b = __ldg(bw + q * 32 + lane);
      dot[h] += v4[q].x * b.x + v4[q].y * b.y + v4[q].z * b.z + v4[q].w * b.w;
    }
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) {
    #pragma unroll
    for (int h = 0; h < HEADS; ++h) dot[h] += __shfl_xor_sync(~0u, dot[h], o);
  }
  if (lane < HEADS) {
    const float out = dead ? -1e4f : dot[lane] * scale;
    bias[(size_t)lane * n * stride + (size_t)i * stride + j] =
        std::is_same_v<T, half> ? (T)fmaxf(out, -6e4f) : (T)out;
  }
}

// The padding columns flash reads past the chain's end, zeroed once a shape rather than every layer.
template <class T>
__global__ void biasPadK(T* __restrict__ bias, int n, int stride, int heads) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int pad = stride - n;
  if (pad <= 0 || i >= (size_t)heads * n * pad) return;
  const int h = (int)(i / ((size_t)n * pad));
  const size_t within = i - (size_t)h * n * pad;
  const int row = (int)(within / pad), col = n + (int)(within % pad);
  bias[(size_t)h * n * stride + (size_t)row * stride + col] = (T)0.f;
}

template <class TZ, class T>
inline void pairBias(const TZ* z, const std::string& normW, const std::string& biasW,
                     const float* pairMask, T* bias, int n, int stride, int heads, int C,
                     float scale) {
  const size_t pairs = (size_t)n * n;
  const unsigned grid = (unsigned)((pairs + 7) / 8);
  static_assert(true, "");
  if (C == 384 && heads == 12) {
    pairBiasK<TZ, T, 12, 12><<<grid, 256, 0, STREAM>>>(z, W(normW), W(biasW), pairMask, bias, n,
                                                          stride, scale);
  } else if (C == 256 && heads == 12) {
    pairBiasK<TZ, T, 8, 12><<<grid, 256, 0, STREAM>>>(z, W(normW), W(biasW), pairMask, bias, n,
                                                         stride, scale);
  } else {
    fprintf(stderr, "pairBias: no kernel for %d channels and %d heads\n", C, heads);
    exit(1);
  }
  if (stride > n) {
    biasPadK<T><<<blocks((size_t)heads * n * (stride - n)), 256, 0, STREAM>>>(bias, n, stride, heads);
  }
}

// ---------------------------------------------------------------- small helpers
__global__ void addIntoK(float* __restrict__ y, const float* __restrict__ x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += x[i];
}
template <class T>
__global__ void addIntoTK(T* __restrict__ y, const float* __restrict__ x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (T)((float)y[i] + x[i]);
}
template <class T>
__global__ void widenK(const T* __restrict__ x, float* __restrict__ y, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (float)x[i];
}
// y += x, but only where the row's mask is set (a masked row must stay exactly zero).
__global__ void addIntoMaskedK(float* __restrict__ y, const float* __restrict__ x,
                               const float* __restrict__ mask, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  if (mask && mask[i / C] == 0.f) return;
  y[i] += x[i];
}
// `z += a * sigmoid(b)` on the live pairs - the triangle's output gate and the residual it feeds.
__global__ void mulSigmoidAddK(const float* __restrict__ a, const float* __restrict__ b,
                               const float* __restrict__ pairMask, float* __restrict__ z,
                               size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  if (pairMask && pairMask[i / C] == 0.f) return;
  z[i] += a[i] * (1.f / (1.f + __expf(-b[i])));
}
// As above, with the gate read CHANNEL-MAJOR: it comes out of the triangle's fused projection in
// plane layout, and an index is cheaper than the GEMM that producing it separately would cost.
template <class T, class TZ>
__global__ void mulSigmoidAddPlaneGateK(const float* __restrict__ a, const T* __restrict__ gPlanes,
                                        const float* __restrict__ pairMask, TZ* __restrict__ z,
                                        size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  size_t p = i / C;
  if (pairMask && pairMask[p] == 0.f) return;
  const float g = (float)gPlanes[(size_t)(i - p * C) * rows + p];
  z[i] = (TZ)((float)z[i] + a[i] * (1.f / (1.f + __expf(-g))));
}

template <class T, class TZ>
__global__ void mulSigmoidAddGateK(const float* __restrict__ a, const T* __restrict__ gate,
                                   const float* __restrict__ pairMask, TZ* __restrict__ z,
                                   size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  if (pairMask && pairMask[i / C] == 0.f) return;
  z[i] = (TZ)((float)z[i] + a[i] * (1.f / (1.f + __expf(-(float)gate[i]))));
}
template <class T>
__global__ void mulSigmoidAddGate4K(const float4* __restrict__ a, const T* __restrict__ gate,
                                    const float* __restrict__ pairMask, float4* __restrict__ z,
                                    size_t rows, int C4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C4) return;
  if (pairMask && pairMask[i / C4] == 0.f) return;
  const float4 aa = a[i];
  float4 zz = z[i];
  float g[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) g[k] = (float)gate[4 * i + k];
  zz.x += aa.x / (1.f + __expf(-g[0]));
  zz.y += aa.y / (1.f + __expf(-g[1]));
  zz.z += aa.z / (1.f + __expf(-g[2]));
  zz.w += aa.w / (1.f + __expf(-g[3]));
  z[i] = zz;
}
// The same for the gated add. The gate stays a strided read - that is the layout the fused
// projection produces and the short form keeps - but x, z and the sigmoid go four at a time.
template <class T>
__global__ void mulSigmoidAddPlaneGate4HK(const float4* __restrict__ a, const T* __restrict__ gPlanes,
                                          const float* __restrict__ pairMask,
                                          __half2* __restrict__ z, size_t rows, int C4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C4) return;
  const size_t p = i / C4;
  if (pairMask && pairMask[p] == 0.f) return;
  const int c0 = (int)(i - p * C4) * 4;
  const float4 aa = a[i];
  float g[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) g[k] = (float)gPlanes[(size_t)(c0 + k) * rows + p];
  float2 lo = __half22float2(z[2 * i]), hi = __half22float2(z[2 * i + 1]);
  lo.x += aa.x / (1.f + __expf(-g[0]));
  lo.y += aa.y / (1.f + __expf(-g[1]));
  hi.x += aa.z / (1.f + __expf(-g[2]));
  hi.y += aa.w / (1.f + __expf(-g[3]));
  z[2 * i] = __floats2half2_rn(lo.x, lo.y);
  z[2 * i + 1] = __floats2half2_rn(hi.x, hi.y);
}

template <class T>
__global__ void mulSigmoidAddGate4HK(const float4* __restrict__ a, const T* __restrict__ gate,
                                     const float* __restrict__ pairMask, __half2* __restrict__ z,
                                     size_t rows, int C4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C4) return;
  if (pairMask && pairMask[i / C4] == 0.f) return;
  const float4 aa = a[i];
  float g[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) g[k] = (float)gate[4 * i + k];
  float2 lo = __half22float2(z[2 * i]), hi = __half22float2(z[2 * i + 1]);
  lo.x += aa.x / (1.f + __expf(-g[0]));
  lo.y += aa.y / (1.f + __expf(-g[1]));
  hi.x += aa.z / (1.f + __expf(-g[2]));
  hi.y += aa.w / (1.f + __expf(-g[3]));
  z[2 * i] = __floats2half2_rn(lo.x, lo.y);
  z[2 * i + 1] = __floats2half2_rn(hi.x, hi.y);
}

template <class T, class TZ>
inline void mulSigmoidAddPlaneGate(const float* a, const T* gPlanes, const float* pairMask, TZ* z,
                                   size_t rows, int C) {
  if constexpr (std::is_same_v<TZ, half>) {
    if ((C & 3) == 0 && ((uintptr_t)a & 15) == 0 && ((uintptr_t)z & 7) == 0) {
      mulSigmoidAddPlaneGate4HK<T><<<blocks(rows * (size_t)(C >> 2)), 256, 0, STREAM>>>(
          (const float4*)a, gPlanes, pairMask, (__half2*)z, rows, C >> 2);
      return;
    }
  }
  mulSigmoidAddPlaneGateK<T, TZ><<<blocks(rows * (size_t)C), 256, 0, STREAM>>>(a, gPlanes, pairMask,
                                                                               z, rows, C);
}

template <class T, class TZ>
inline void mulSigmoidAddGate(const float* a, const T* gate, const float* pairMask, TZ* z,
                              size_t rows, int C) {
  if constexpr (std::is_same_v<TZ, float>) {
    if ((C & 3) == 0 && ((((uintptr_t)a | (uintptr_t)z) & 15) == 0)) {
      mulSigmoidAddGate4K<T><<<blocks(rows * (size_t)(C >> 2)), 256, 0, STREAM>>>(
          (const float4*)a, gate, pairMask, (float4*)z, rows, C >> 2);
      return;
    }
  }
  if constexpr (std::is_same_v<TZ, half>) {
    // The dense-gate form of the kernel above: the gate is row-major here, so it reads four in a row.
    if ((C & 3) == 0 && ((uintptr_t)a & 15) == 0 && ((uintptr_t)z & 7) == 0) {
      mulSigmoidAddGate4HK<T><<<blocks(rows * (size_t)(C >> 2)), 256, 0, STREAM>>>(
          (const float4*)a, gate, pairMask, (__half2*)z, rows, C >> 2);
      return;
    }
  }
  mulSigmoidAddGateK<T, TZ><<<blocks(rows * (size_t)C), 256, 0, STREAM>>>(a, gate, pairMask, z, rows,
                                                                          C);
}

// `x = relu(x + bias)` - the pair transition's up projection, whose bias was its own launch.
template <class T>
__global__ void biasReluK(T* __restrict__ x, const float* __restrict__ bias, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  const float v = (float)x[i] + bias[i % C];
  x[i] = (T)(v < 0.f ? 0.f : v);
}

// 🔴 A SCALAR half KERNEL MOVES HALF THE BYTES IT COULD. One `half` a thread is a 2-byte access, and
// these tensors are tens of megabytes: `biasReluK` on [pairs, 768] measured 2.1x off its own
// bandwidth for no reason but the access width. A `half2` a thread fixes it, and the tail is handled
// by only taking this path when the row length is even.
__global__ void biasRelu2K(__half2* __restrict__ x, const float* __restrict__ bias, size_t pairs,
                           int C2) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs * (size_t)C2) return;
  const int c = (int)(i % C2);
  const float2 b = {bias[2 * c], bias[2 * c + 1]};
  float2 v = __half22float2(x[i]);
  v.x += b.x;
  v.y += b.y;
  x[i] = __floats2half2_rn(v.x < 0.f ? 0.f : v.x, v.y < 0.f ? 0.f : v.y);
}

template <class T>
inline void biasRelu(T* x, const float* bias, size_t rows, int C) {
  if constexpr (std::is_same_v<T, half>) {
    if ((C & 1) == 0 && ((uintptr_t)x & 3) == 0) {
      const size_t n2 = rows * (size_t)(C >> 1);
      biasRelu2K<<<blocks(n2), 256, 0, STREAM>>>((__half2*)x, bias, rows, C >> 1);
      return;
    }
  }
  biasReluK<T><<<blocks(rows * (size_t)C), 256, 0, STREAM>>>(x, bias, rows, C);
}
// `z += x + bias` on the live pairs - the pair transition's down projection and its residual.
template <class TZ>
__global__ void biasAddK(const float* __restrict__ x, const float* __restrict__ bias,
                         const float* __restrict__ pairMask, TZ* __restrict__ z, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  if (pairMask && pairMask[i / C] == 0.f) return;
  z[i] = (TZ)((float)z[i] + x[i] + bias[i % C]);
}
// Four at a time. The row length divides by four everywhere this runs, so a quad never straddles
// two pairs and the mask is still one lookup a thread.
__global__ void biasAdd4K(const float4* __restrict__ x, const float* __restrict__ bias,
                          const float* __restrict__ pairMask, float4* __restrict__ z, size_t rows,
                          int C4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C4) return;
  if (pairMask && pairMask[i / C4] == 0.f) return;
  const float4 bb = ((const float4*)bias)[i % C4], xx = x[i];
  float4 zz = z[i];
  zz.x += xx.x + bb.x; zz.y += xx.y + bb.y; zz.z += xx.z + bb.z; zz.w += xx.w + bb.w;
  z[i] = zz;
}
// 🔴 A half z COST THE VECTORISED PATHS THAT A float z HAD, which is most of why narrowing the
// pair tensor paid 2-3% instead of the 8% its byte count promised: `biasAdd`, the hadamard write
// and the gated add all fell back to one element a thread. Four halves is one 8-byte access, the
// same shape as the float4 path beside it.
__global__ void biasAdd4HK(const float4* __restrict__ x, const float* __restrict__ bias,
                           const float* __restrict__ pairMask, __half2* __restrict__ z, size_t rows,
                           int C4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C4) return;
  if (pairMask && pairMask[i / C4] == 0.f) return;
  const float4 bb = ((const float4*)bias)[i % C4], xx = x[i];
  float2 lo = __half22float2(z[2 * i]), hi = __half22float2(z[2 * i + 1]);
  lo.x += xx.x + bb.x; lo.y += xx.y + bb.y; hi.x += xx.z + bb.z; hi.y += xx.w + bb.w;
  z[2 * i] = __floats2half2_rn(lo.x, lo.y);
  z[2 * i + 1] = __floats2half2_rn(hi.x, hi.y);
}

template <class TZ>
inline void biasAdd(const float* x, const float* bias, const float* pairMask, TZ* z, size_t rows,
                    int C) {
  if constexpr (std::is_same_v<TZ, half>) {
    if ((C & 3) == 0 && ((((uintptr_t)x | (uintptr_t)bias) & 15) == 0) && ((uintptr_t)z & 7) == 0) {
      biasAdd4HK<<<blocks(rows * (size_t)(C >> 2)), 256, 0, STREAM>>>(
          (const float4*)x, bias, pairMask, (__half2*)z, rows, C >> 2);
      return;
    }
  }
  if constexpr (std::is_same_v<TZ, float>) {
    if ((C & 3) == 0 && ((((uintptr_t)x | (uintptr_t)z | (uintptr_t)bias) & 15) == 0)) {
      biasAdd4K<<<blocks(rows * (size_t)(C >> 2)), 256, 0, STREAM>>>(
          (const float4*)x, bias, pairMask, (float4*)z, rows, C >> 2);
      return;
    }
  }
  biasAddK<TZ><<<blocks(rows * (size_t)C), 256, 0, STREAM>>>(x, bias, pairMask, z, rows, C);
}
// Zero every row the mask clears; the reference zeroes a padded token's whole row at several seams.
__global__ void maskRowsK(float* __restrict__ x, const float* __restrict__ mask, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * (size_t)C && mask[i / C] == 0.f) x[i] = 0.f;
}
// The outer product of the token mask, `pairMask[i][j] = mask[i] * mask[j]`.
__global__ void pairMaskK(const float* __restrict__ mask, float* __restrict__ out, int n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n * n) out[i] = mask[i / n] * mask[i % n];
}

// `Y[out][rows] = X[rows][in] * W[out][in]^T` - the projection written CHANNEL-MAJOR, which is the
// layout the triangle's contraction wants. Same arithmetic as `linear`, transposed on the way out, so
// the transpose costs nothing instead of a kernel over the whole pair tensor.
template <class T>
inline void linearPlanes(const T* X, T* Y, size_t rows, int in, int out, const std::string& w) {
  if (lenW(w) != (size_t)in * out) {
    fprintf(stderr, "%s has %zu elements, not %d x %d\n", w.c_str(), lenW(w), in, out);
    exit(1);
  }
  const float one = 1.f, zero = 0.f;
  constexpr cudaDataType_t TY = std::is_same_v<T, half> ? CUDA_R_16F : CUDA_R_32F;
  const void* Wp = std::is_same_v<T, half> ? (const void*)Wh(w) : (const void*)W(w);
  CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, out, in, &one, X, TY, in, Wp, TY, in,
                  &zero, Y, TY, (int)rows,
                  std::is_same_v<T, half> ? CUBLAS_COMPUTE_32F : computeType(),
                  CUBLAS_GEMM_DEFAULT));
}

// ---------------------------------------------------------------- channel planes
//
// `[pair][channel]` <-> `[channel][pair]`.
//
// 🔴 THE TRIANGLE'S CONTRACTION CANNOT READ THE INTERLEAVED FORM, AND THE FAILURE IS SILENT. A
// column-major GEMM operand's column must be contiguous; in `[i][k][C]` the elements a channel's
// matrix needs are `C` apart, so handing cuBLAS `lda = C` makes it read a column out of the WRONG
// channels and return a full tensor of plausible numbers. One `n x n` plane a channel is why AF3's
// pairtrack transposes before it contracts, and this is that transpose.
__global__ void toPlanesK(const float* __restrict__ in, float* __restrict__ out, size_t pairs, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs * (size_t)C) return;
  size_t p = i / C;
  int c = (int)(i - p * C);
  out[(size_t)c * pairs + p] = in[i];
}
// The triangle's gating, in plane layout and in one launch: `a = aP * sigmoid(aG) * mask` and
// `b = bP * sigmoid(bG) * mask`, reading the four quarters of one fused projection and writing over
// the first and third. Replaces two sigmoid-multiplies, two mask passes and two transposes.
// The same, for the triangle's gating: two adjacent PAIRS of one channel a thread.
__global__ void gatePlanes2K(__half2* __restrict__ p, const float* __restrict__ pairMask,
                             size_t pairs2, int cH) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs2 * (size_t)cH) return;
  const size_t within = i % pairs2;
  float2 m = {1.f, 1.f};
  if (pairMask) {
    m.x = pairMask[2 * within];
    m.y = pairMask[2 * within + 1];
  }
  __half2* aP = p;
  const __half2* aG = p + pairs2 * (size_t)cH;
  __half2* bP = p + 2 * pairs2 * (size_t)cH;
  const __half2* bG = p + 3 * pairs2 * (size_t)cH;
  const float2 ap = __half22float2(aP[i]), ag = __half22float2(aG[i]);
  const float2 bp = __half22float2(bP[i]), bg = __half22float2(bG[i]);
  aP[i] = __floats2half2_rn(m.x * ap.x / (1.f + __expf(-ag.x)), m.y * ap.y / (1.f + __expf(-ag.y)));
  bP[i] = __floats2half2_rn(m.x * bp.x / (1.f + __expf(-bg.x)), m.y * bp.y / (1.f + __expf(-bg.y)));
}

template <class T>
__global__ void gatePlanesK(T* __restrict__ p, const float* __restrict__ pairMask, size_t pairs,
                            int cH) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs * (size_t)cH) return;
  const float m = pairMask ? pairMask[i % pairs] : 1.f;
  T* aP = p;
  const T* aG = p + pairs * (size_t)cH;
  T* bP = p + 2 * pairs * (size_t)cH;
  const T* bG = p + 3 * pairs * (size_t)cH;
  const float a = m * (float)aP[i] / (1.f + __expf(-(float)aG[i]));
  const float b = m * (float)bP[i] / (1.f + __expf(-(float)bG[i]));
  aP[i] = (T)a;
  bP[i] = (T)b;
}

template <class T>
inline void gatePlanes(T* p, const float* pairMask, size_t pairs, int cH) {
  if constexpr (std::is_same_v<T, half>) {
    // Two adjacent pairs share a thread, so the plane length must be even and the planes aligned.
    if ((pairs & 1) == 0 && ((uintptr_t)p & 3) == 0) {
      const size_t pairs2 = pairs >> 1;
      gatePlanes2K<<<blocks(pairs2 * (size_t)cH), 256, 0, STREAM>>>((__half2*)p, pairMask, pairs2, cH);
      return;
    }
  }
  gatePlanesK<T><<<blocks(pairs * (size_t)cH), 256, 0, STREAM>>>(p, pairMask, pairs, cH);
}

__global__ void fromPlanesK(const float* __restrict__ in, float* __restrict__ out, size_t pairs, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs * (size_t)C) return;
  size_t p = i / C;
  int c = (int)(i - p * C);
  out[i] = in[(size_t)c * pairs + p];
}

// 🔴 AND FUSING THAT TRANSPOSE INTO THE LayerNorm THAT FOLLOWS IT WAS MEASURED AND IS WORSE: 66.5 ->
// 73.4 ms. One thread a pair loops the channels, which is 10,000 threads and ~39 blocks at n = 100 -
// every load coalesced and the card two thirds idle, against `layerNormRegK`'s warp a row with the
// whole row in registers. The coalescing argument was right and the occupancy one decides it.
//
// A 32x32 tiled transpose instead, which is the part that was actually slow: the naive form reads
// `in[c * pairs + p]` with consecutive threads `pairs` apart, so its loads are one word a
// transaction. Staging a tile through shared memory makes both sides coalesced.
// The staging tile is the SOURCE's type, not float: a half tensor staged through a float tile moves
// twice the shared-memory bytes and converts every element twice for nothing.
template <class TI, class TO, int TILE = 32>
__global__ void __launch_bounds__(TILE * 8)
fromPlanesTiledK(const TI* __restrict__ in, TO* __restrict__ out, size_t pairs, int C) {
  __shared__ TI tile[TILE][TILE + 1];
  const size_t p0 = (size_t)blockIdx.x * TILE;
  const int c0 = blockIdx.y * TILE;
  for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
    const int c = c0 + r;
    const size_t p = p0 + threadIdx.x;
    tile[r][threadIdx.x] = (c < C && p < pairs) ? in[(size_t)c * pairs + p] : (TI)0.f;
  }
  __syncthreads();
  for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
    const size_t p = p0 + r;
    const int c = c0 + threadIdx.x;
    if (c < C && p < pairs) out[p * C + c] = (TO)(float)tile[threadIdx.x][r];
  }
}
template <class TI, class TO>
inline void fromPlanes(const TI* in, TO* out, size_t pairs, int C) {
  dim3 grid((unsigned)((pairs + 31) / 32), (unsigned)((C + 31) / 32));
  fromPlanesTiledK<TI, TO, 32><<<grid, dim3(32, 8), 0, STREAM>>>(in, out, pairs, C);
}

// ---------------------------------------------------------------- attention packing
//
// flash reads one `[position][q | k | v | gate]` row of `4W`. q and k are written there already, normed
// straight out of the `to_qkv` projection by two strided `rmsNorm`s; this puts v (still in that
// projection, at its third `W`) and the gate beside them.
// `src` is the fused [q | k | v | g] projection, one row of `4 * W` a token; q and k are already in
// `qkvg`, normed. 🔴 THE ROW STRIDE IS AN ARGUMENT because it was 3W when q/k/v came from one GEMM and
// the gate from another, and 4W once they were fused - baked in, it reads the wrong token's v and
// still produces a full, finite attention.
template <class T>
__global__ void packVGK(const float* __restrict__ src, T* __restrict__ qkvg, size_t n, int W) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * (size_t)W) return;
  size_t pos = i / W;
  int c = (int)(i - pos * W);
  const float vv = src[pos * 4 * W + 2 * W + c], gg = src[pos * 4 * W + 3 * W + c];
  if constexpr (std::is_same_v<T, half>) {
    qkvg[pos * 4 * W + 2 * W + c] = __float2half(vv);
    qkvg[pos * 4 * W + 3 * W + c] = __float2half(gg);
  } else {
    qkvg[pos * 4 * W + 2 * W + c] = vv;
    qkvg[pos * 4 * W + 3 * W + c] = gg;
  }
}

// `raw [i * n + j][heads]` -> `bias [head][i][j]`, `n` rows a head at a padded row stride.
//
// 🔴 THE PLANE IS `n * stride`, NOT `stride * stride`. flash reads
// `bias + (h * n + i) * biasStride` (flash.cuh:1046), so padding the ROW count as well as the column
// count sends every head after the first to the wrong place - and quietly, since it stays inside the
// allocation and returns plausible numbers.
//
// 🔴 A PAIR OUTSIDE THE MASK IS NOT ZERO-BIASED, IT IS LARGELY NEGATIVE. The reference REPLACES the
// score with -1e4 for an invalid pair rather than adding to it, which an additive bias cannot express
// exactly; `-1e4` added to a dot product leaves the same ~0 softmax weight, so the two agree to the
// width of a float and not to the bit. Nothing here is padded yet (every gated case is one full
// chain), and a padded case wants this checked before it is believed.
// 🔴 AND THE f16 KERNELS WORK IN THE LOG2 DOMAIN, SO THE BIAS MUST ARRIVE PRE-MULTIPLIED BY LOG2E.
// They fold `scale * LOG2E` into q and leave the bias alone (flash.cuh:434), where the f32 reference
// kernel is in the natural domain and wants `scale` 1. Handing the half path an unscaled bias is a
// softmax at the wrong temperature - every logit still finite, every weight wrong.
template <class T>
__global__ void biasLayoutK(const float* __restrict__ raw, T* __restrict__ bias,
                            const float* __restrict__ pairMask, int n, int stride, int heads,
                            float scale) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)heads * n * stride) return;
  int h = (int)(i / ((size_t)n * stride));
  size_t within = i - (size_t)h * n * stride;
  int row = (int)(within / stride), col = (int)(within % stride);
  float v = 0.f;
  if (col < n) {
    size_t pair = (size_t)row * n + col;
    v = (pairMask && pairMask[pair] == 0.f) ? -1e4f : raw[pair * heads + h] * scale;
  }
  if constexpr (std::is_same_v<T, half>) bias[i] = __float2half(fmaxf(v, -6e4f));
  else bias[i] = v;
}

}  // namespace rfd4p
