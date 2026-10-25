// The four pieces of RFD4-Proteina that cuda/ does not already have: RMSNorm, its adaptive LayerNorm,
// its output scale, and a SwiGLU whose two halves are the other way round.
//
// Everything else is reused - `gemm`/`linear` (cuda/ef2/src/ops.cuh, cuda/af3/src/common.cuh),
// `layerNorm` for the pair track's LayerNorms, `flashGrid` for the attention.
#pragma once
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

// ---------------------------------------------------------------- small helpers
__global__ void addIntoK(float* __restrict__ y, const float* __restrict__ x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += x[i];
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
// `x = relu(x + bias)` - the pair transition's up projection, whose bias was its own launch.
__global__ void biasReluK(float* __restrict__ x, const float* __restrict__ bias, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  const float v = x[i] + bias[i % C];
  x[i] = v < 0.f ? 0.f : v;
}
// `z += x + bias` on the live pairs - the pair transition's down projection and its residual.
__global__ void biasAddK(const float* __restrict__ x, const float* __restrict__ bias,
                         const float* __restrict__ pairMask, float* __restrict__ z, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (size_t)C) return;
  if (pairMask && pairMask[i / C] == 0.f) return;
  z[i] += x[i] + bias[i % C];
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
inline void linearPlanes(const float* X, float* Y, size_t rows, int in, int out, const std::string& w) {
  if (lenW(w) != (size_t)in * out) {
    fprintf(stderr, "%s has %zu elements, not %d x %d\n", w.c_str(), lenW(w), in, out);
    exit(1);
  }
  const float one = 1.f, zero = 0.f;
  CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, out, in, &one, X, CUDA_R_32F, in, W(w),
                  CUDA_R_32F, in, &zero, Y, CUDA_R_32F, (int)rows, computeType(),
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
__global__ void gatePlanesK(float* __restrict__ p, const float* __restrict__ pairMask, size_t pairs,
                            int cH) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs * (size_t)cH) return;
  const float m = pairMask ? pairMask[i % pairs] : 1.f;
  float* aP = p;
  const float* aG = p + pairs * (size_t)cH;
  float* bP = p + 2 * pairs * (size_t)cH;
  const float* bG = p + 3 * pairs * (size_t)cH;
  aP[i] = m * aP[i] / (1.f + __expf(-aG[i]));
  bP[i] = m * bP[i] / (1.f + __expf(-bG[i]));
}

__global__ void fromPlanesK(const float* __restrict__ in, float* __restrict__ out, size_t pairs, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs * (size_t)C) return;
  size_t p = i / C;
  int c = (int)(i - p * C);
  out[i] = in[(size_t)c * pairs + p];
}

// ---------------------------------------------------------------- attention packing
//
// flash reads one `[position][q | k | v | gate]` row of `4W`. q and k are written there already, normed
// straight out of the `to_qkv` projection by two strided `rmsNorm`s; this puts v (still in that
// projection, at its third `W`) and the gate beside them.
__global__ void packVGK(const float* __restrict__ v, const float* __restrict__ gate,
                        float* __restrict__ qkvg, size_t n, int W) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * (size_t)W) return;
  size_t pos = i / W;
  int c = (int)(i - pos * W);
  qkvg[pos * 4 * W + 2 * W + c] = v[pos * 3 * W + c];
  qkvg[pos * 4 * W + 3 * W + c] = gate[i];
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
__global__ void biasLayoutK(const float* __restrict__ raw, float* __restrict__ bias,
                            const float* __restrict__ pairMask, int n, int stride, int heads) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)heads * n * stride) return;
  int h = (int)(i / ((size_t)n * stride));
  size_t within = i - (size_t)h * n * stride;
  int row = (int)(within / stride), col = (int)(within % stride);
  if (col >= n) { bias[i] = 0.f; return; }
  size_t pair = (size_t)row * n + col;
  bias[i] = (pairMask && pairMask[pair] == 0.f) ? -1e4f : raw[pair * heads + h];
}

}  // namespace rfd4p
