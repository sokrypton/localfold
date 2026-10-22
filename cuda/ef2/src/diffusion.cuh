// ESMFold2's diffusion module (cpu/esmfold2/diffusion.js and biohub's DiffusionModule are the
// reading), one denoiser call:
//
//   conditioning: pair = [z_trunk | rel_pos] -> LN -> zProjection, + 2 transitions   (once a fold)
//                 single = LN(s_inputs) @ sProjection + noise(t), + 2 transitions     (every step)
//                 noise(t) = LN(cos(2 pi (t_noise w + b))) @ noiseProjection, t_noise = log(t / sigma) / 4
//   r = x_noisy / sqrt(t^2 + sigma^2);  q = c0 + [r | 0] @ coordsLinear;  q = 3 windowed SWA blocks (c0)
//   a = pool(relu(q @ toToken)) + LN(single; stepNorm) @ singleToToken
//   12 token blocks:  a += attention(adaLN(a, single), pair bias) * sigmoid(single @ outGate + b)
//                     a += swiglu(adaLN(a, single)) * sigmoid(single @ outGate + b)
//   a = LN(a; tokenNorm);  q = skip + gather(a @ tokenToAtom);  q = 3 windowed SWA blocks (c0)
//   x_denoised = sigma^2 / (sigma^2 + t^2) x_noisy + sigma t / sqrt(sigma^2 + t^2) * (LN(q) @ outputLinear)
#pragma once
#include "atoms.cuh"
#include "../../af3/src/multigpu.cuh"      // (several GPUs: the token attention's query rows, EF2_QLO)

constexpr int DIFFUSION_HALF_WINDOW = 64;     // the diffusion's atom stacks are windowed (128) in both references

// rows [p0, p0 + n) of [z | rel_pos]
__global__ void joinPairRelK(const float* z, RelIdx rel, float* out, size_t p0, size_t n, int T, int C, bool z16 = false) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * 2 * C) return;
  size_t p = p0 + t / (2 * C); int c = (int)(t % (2 * C));
  out[t] = c < C ? (z16 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(z)[p * C + c]) : z[p * C + c])   // (EF2_Z16)
                 : relPosAt(rel, (int)(p / T), (int)(p % T), c - C, C);
}
__global__ void siluMulK(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)a | (uintptr_t)b | (uintptr_t)out) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 av = __ldg(reinterpret_cast<const float4*>(a) + t);
      float4 bv = __ldg(reinterpret_cast<const float4*>(b) + t);
      reinterpret_cast<float4*>(out)[t] = make_float4(siluF(av.x) * bv.x, siluF(av.y) * bv.y, siluF(av.z) * bv.z, siluF(av.w) * bv.w);
    }
    return;
  }
  if (t < n) out[t] = siluF(a[t]) * b[t];
}
// TransitionLayer: x += out(silu(LN(x) @ a) * (LN(x) @ b)), over rows in chunks
inline void transitionLayer(float* x, size_t rows, int C, const std::string& B) {
  int Hd = (int)dimOf("f/" + B + "aProjection", 1);
  size_t chunk = std::max<size_t>(1, ((size_t)32 << 20) / (3 * (size_t)Hd));
  float* xn = scratch<float>("dtr.xn", std::min(rows, chunk) * C);
  float* a = scratch<float>("dtr.a", std::min(rows, chunk) * Hd); float* b = scratch<float>("dtr.b", std::min(rows, chunk) * Hd);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    layerNorm(x + r0 * C, xn, r, C, F(B + "norm/scale"), F(B + "norm/offset"));
    gemm(xn, F(B + "aProjection"), a, r, C, Hd);
    gemm(xn, F(B + "bProjection"), b, r, C, Hd);
    size_t n = r * Hd;
    siluMulK<<<blocks((n & 3) == 0 ? (n >> 2) : n), 256, 0, STREAM>>>(a, b, a, n);
    gemm(a, F(B + "outProjection"), x + r0 * C, r, Hd, C, 1.f);
  }
}
// the noise level's scalars, on the device so one captured step serves every level:
// [0] t_noise = log(t / sigma) / 4, [1] 1 / sqrt(t^2 + sigma^2), [2] sigma^2 / (sigma^2 + t^2) (keep),
// [3] sigma t / sqrt(sigma^2 + t^2) (take)
struct NoiseLevel { float v[4]; };
inline NoiseLevel noiseLevel(float t, float sigma) {
  float s2 = sigma * sigma, t2 = t * t;
  return {{0.25f * logf(fmaxf(t / sigma, 1e-20f)), 1.f / sqrtf(t2 + s2), s2 / (s2 + t2), sigma * t / sqrtf(s2 + t2)}};
}
__global__ void fourierK(const float* w, const float* b, const float* level, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cosf(2.f * 3.14159265358979323846f * (level[0] * w[i] + b[i]));
}
__global__ void addRowK(float* __restrict__ x, const float* __restrict__ row, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)x | (uintptr_t)row) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t < n4) {
      int c = (int)((unsigned)t % (unsigned)c4);
      float4 xv = reinterpret_cast<const float4*>(x)[t];
      float4 rv = __ldg(reinterpret_cast<const float4*>(row) + c);
      reinterpret_cast<float4*>(x)[t] = make_float4(xv.x + rv.x, xv.y + rv.y, xv.z + rv.z, xv.w + rv.w);
    }
    return;
  }
  if (t < rows * C) x[t] += row[t % C];
}
__global__ void coordsInputK(const float* x, const float* level, float* out, int A) {   // [A, 6] = [x / denom | 0]
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= A * 6) return;
  int a = t / 6, k = t % 6;
  out[t] = k < 3 ? x[a * 3 + k] * level[1] : 0.f;
}
// adaLN: sigmoid(LN(s; scale) @ gate + gateBias) * LN(a) + LN(s; scale) @ shift
// (g and sh rows ld apart)
__global__ void adaCombineK(const float* an, const float* g, const float* gb, const float* sh, float* out, size_t T, int C,
                            int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T * C) return;
  size_t gi = (t / C) * ld + t % C;
  out[t] = an[t] / (1.f + expf(-(g[gi] + gb[t % C]))) + sh[gi];
}
// ...the LayerNorm (affine-free) and that combine in one pass: a block a row, its first warp taking the statistics
// with layerNormK's arithmetic exactly, then all of the block writing the row (a warp a row put the whole combine on
// T warps - 9 blocks at 68 tokens - and was slower than the two kernels)
__global__ void adaLNK(float* __restrict__ a, const float* __restrict__ g, const float* __restrict__ gb,
                       const float* __restrict__ sh, float* __restrict__ out, size_t T, int C, int ld,
                       const float* __restrict__ o = nullptr, const float* __restrict__ og = nullptr,
                       const float* __restrict__ ogb = nullptr) {
  __shared__ float4 sx4[256];
  __shared__ float stats[2];
  float* sx = reinterpret_cast<float*>(sx4);
  size_t row = blockIdx.x;
  float* xr = a + row * C;
  int tid = threadIdx.x;
  if (C == 768 && (((uintptr_t)xr | (uintptr_t)(g + row * ld) | (uintptr_t)gb | (uintptr_t)(sh + row * ld) | (uintptr_t)(out + row * 768) | (uintptr_t)o | (uintptr_t)(og ? og + row * ld : nullptr) | (uintptr_t)ogb) & 15) == 0) {
    float4 x4, g4, gb4, sh4;
    if (tid < 192) {
      x4 = __ldg(reinterpret_cast<const float4*>(xr) + tid);
      if (o) {
        float4 o4 = __ldg(reinterpret_cast<const float4*>(o + row * 768) + tid);
        float4 og4 = __ldg(reinterpret_cast<const float4*>(og + row * ld) + tid);
        float4 ogb4 = __ldg(reinterpret_cast<const float4*>(ogb) + tid);
        x4.x = __fadd_rn(x4.x, __fmul_rn(o4.x, 1.f / (1.f + expf(-(og4.x + ogb4.x)))));
        x4.y = __fadd_rn(x4.y, __fmul_rn(o4.y, 1.f / (1.f + expf(-(og4.y + ogb4.y)))));
        x4.z = __fadd_rn(x4.z, __fmul_rn(o4.z, 1.f / (1.f + expf(-(og4.z + ogb4.z)))));
        x4.w = __fadd_rn(x4.w, __fmul_rn(o4.w, 1.f / (1.f + expf(-(og4.w + ogb4.w)))));
        reinterpret_cast<float4*>(xr)[tid] = x4;
      }
      g4 = __ldg(reinterpret_cast<const float4*>(g + row * ld) + tid);
      gb4 = __ldg(reinterpret_cast<const float4*>(gb) + tid);
      sh4 = __ldg(reinterpret_cast<const float4*>(sh + row * ld) + tid);
      sx4[tid] = x4;
    }
    __syncthreads();
    if (tid < 32) {
      float s = 0;
      #pragma unroll
      for (int k = 0; k < 24; ++k) s += sx[k * 32 + tid];
      #pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
      float mean = s * (1.f / 768.f), v = 0;
      #pragma unroll
      for (int k = 0; k < 24; ++k) { float d = sx[k * 32 + tid] - mean; v += d * d; }
      #pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (tid == 0) { stats[0] = mean; stats[1] = rsqrtf(v * (1.f / 768.f) + 1e-5f); }
    }
    __syncthreads();
    if (tid < 192) {
      float mean = stats[0], inv = stats[1];
      float n0 = __fmul_rn(x4.x - mean, inv);
      float n1 = __fmul_rn(x4.y - mean, inv);
      float n2 = __fmul_rn(x4.z - mean, inv);
      float n3 = __fmul_rn(x4.w - mean, inv);
      reinterpret_cast<float4*>(out + row * 768)[tid] = make_float4(
          __fadd_rn(n0 / (1.f + expf(-(g4.x + gb4.x))), sh4.x),
          __fadd_rn(n1 / (1.f + expf(-(g4.y + gb4.y))), sh4.y),
          __fadd_rn(n2 / (1.f + expf(-(g4.z + gb4.z))), sh4.z),
          __fadd_rn(n3 / (1.f + expf(-(g4.w + gb4.w))), sh4.w));
    }
    return;
  }
  if (o) {
    for (int c = tid; c < C; c += blockDim.x) {
      float ov = __fmul_rn(o[row * C + c], 1.f / (1.f + expf(-(og[row * ld + c] + ogb[c]))));
      xr[c] = __fadd_rn(xr[c], ov);
    }
    __syncthreads();
  }
  if (tid < 32) {
    int lane = tid;
    float s = 0;
    for (int c = lane; c < C; c += 32) s += xr[c];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, v = 0;
    for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (lane == 0) { stats[0] = mean; stats[1] = rsqrtf(v / C + 1e-5f); }
  }
  __syncthreads();
  float mean = stats[0], inv = stats[1];
  for (int c = tid; c < C; c += blockDim.x) {
    float n = __fmul_rn(xr[c] - mean, inv);
    size_t gi = row * ld + c;
    out[row * C + c] = __fadd_rn(n / (1.f + expf(-(g[gi] + gb[c]))), sh[gi]);
  }
}
// a[t] += x[t] * sigmoid(g + gb): sigmoidMulK then addK in one pass, with their two roundings (no contraction)
__global__ void gateAddK(float* __restrict__ a, const float* __restrict__ x, const float* __restrict__ g,
                         const float* __restrict__ gb, size_t T, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (ld & 3) == 0 && (((uintptr_t)a | (uintptr_t)x | (uintptr_t)g | (uintptr_t)gb) & 15) == 0) {
    int c4 = C >> 2, ld4 = ld >> 2;
    size_t n4 = T * (size_t)c4;
    if (t >= n4) return;
    unsigned u = (unsigned)t, uc = (unsigned)c4, r = (c4 == 192) ? (u / 192u) : (u / uc), c = u - r * uc;
    float4 av = reinterpret_cast<const float4*>(a)[t];
    float4 xv = __ldg(reinterpret_cast<const float4*>(x) + t);
    float4 gv = __ldg(reinterpret_cast<const float4*>(g) + (size_t)r * ld4 + c);
    float4 gbv = __ldg(reinterpret_cast<const float4*>(gb) + c);
    reinterpret_cast<float4*>(a)[t] = make_float4(
        __fadd_rn(av.x, __fmul_rn(xv.x, 1.f / (1.f + expf(-(gv.x + gbv.x))))),
        __fadd_rn(av.y, __fmul_rn(xv.y, 1.f / (1.f + expf(-(gv.y + gbv.y))))),
        __fadd_rn(av.z, __fmul_rn(xv.z, 1.f / (1.f + expf(-(gv.z + gbv.z))))),
        __fadd_rn(av.w, __fmul_rn(xv.w, 1.f / (1.f + expf(-(gv.w + gbv.w))))));
    return;
  }
  size_t n = T * (size_t)C;
  if (t >= n) return;
  size_t r; int c;
  if (C == 768 && n <= UINT32_MAX) {
    unsigned u = (unsigned)t, q = u / 768u;
    r = q; c = (int)(u - q * 768u);
  } else if (n <= UINT32_MAX) {
    unsigned u = (unsigned)t, uc = (unsigned)C, q = u / uc;
    r = q; c = (int)(u - q * uc);
  } else {
    r = t / C; c = (int)(t % C);
  }
  float o = __fmul_rn(__ldg(x + t), 1.f / (1.f + expf(-(__ldg(g + r * (size_t)ld + c) + __ldg(gb + c)))));
  a[t] = __fadd_rn(a[t], o);
}
__global__ void sigmoidMulK(float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ gb, size_t T, int C, int ld) {   // x *= sigmoid(g (+ gb))
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (ld & 3) == 0 && (((uintptr_t)x | (uintptr_t)g | (uintptr_t)gb) & 15) == 0) {
    int c4 = C >> 2, ld4 = ld >> 2;
    size_t n4 = T * (size_t)c4;
    if (t >= n4) return;
    unsigned u = (unsigned)t, uc = (unsigned)c4, r = u / uc, c = u - r * uc;
    float4 xv = reinterpret_cast<const float4*>(x)[t];
    float4 gv = __ldg(reinterpret_cast<const float4*>(g) + (size_t)r * ld4 + c);
    float4 gbv = gb ? __ldg(reinterpret_cast<const float4*>(gb) + c) : make_float4(0.f, 0.f, 0.f, 0.f);
    reinterpret_cast<float4*>(x)[t] = make_float4(
        xv.x / (1.f + expf(-(gv.x + gbv.x))),
        xv.y / (1.f + expf(-(gv.y + gbv.y))),
        xv.z / (1.f + expf(-(gv.z + gbv.z))),
        xv.w / (1.f + expf(-(gv.w + gbv.w))));
    return;
  }
  if (t >= T * C) return;
  x[t] *= 1.f / (1.f + expf(-(g[(t / C) * ld + t % C] + (gb ? gb[t % C] : 0.f))));
}
// out[j] = x * scale[j] for each of n scales, [n, rows, C]
__global__ void scaleCopiesK(const float* __restrict__ x, const float* __restrict__ scales, float* __restrict__ out,
                             size_t rows, int C, int n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)x | (uintptr_t)scales | (uintptr_t)out) & 15) == 0) {
    int c4 = C >> 2;
    size_t per4 = rows * (size_t)c4, total4 = (size_t)n * per4;
    if (t >= total4) return;
    unsigned u = (unsigned)t, uper = (unsigned)per4, uc = (unsigned)c4;
    unsigned j = u / uper, r = u - j * uper, q = r / uc, c = r - q * uc;
    float4 xv = __ldg(reinterpret_cast<const float4*>(x) + r);
    float4 sv = __ldg(reinterpret_cast<const float4*>(scales) + (size_t)j * c4 + c);
    reinterpret_cast<float4*>(out)[t] = make_float4(xv.x * sv.x, xv.y * sv.y, xv.z * sv.z, xv.w * sv.w);
    return;
  }
  if (t >= (size_t)n * rows * C) return;
  size_t per = rows * C; int j = (int)(t / per); size_t r = t % per;
  out[t] = x[r] * scales[(size_t)j * C + r % C];
}
// scores [H, T, T] + bias [H, T, T], softmax per row
template <class B>
__global__ void biasSoftmaxK(float* __restrict__ S, const B* __restrict__ bias, int T, float scale) {
  size_t row = blockIdx.x;
  float* s = S + row * T; const B* b = bias + row * T;
  __shared__ float red[32];
  if (T <= 1024) {
    float sv[4];
    float m = -INFINITY;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int j = threadIdx.x + k * 256;
      if (j < T) {
        float v = s[j] * scale + (float)b[j];
        sv[k] = v;
        m = fmaxf(m, v);
      }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
      #pragma unroll
      for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
      if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    m = red[0];
    float sum = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int j = threadIdx.x + k * 256;
      if (j < T) {
        float e = expf(sv[k] - m);
        sv[k] = e;
        sum += e;
      }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
      float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
      #pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    float inv = 1.f / red[0];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int j = threadIdx.x + k * 256;
      if (j < T) s[j] = sv[k] * inv;
    }
    return;
  }
  float m = -INFINITY;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float v = s[j] * scale + (float)b[j]; s[j] = v; m = fmaxf(m, v); }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  m = red[0];
  __syncthreads();
  float sum = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) { float e = expf(s[j] - m); s[j] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < T; j += blockDim.x) s[j] *= inv;
}
template <class B>
__global__ void pairToHeadsK(const float* pb, B* out, size_t P, int Hh) {   // [P, H] -> [H, P]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < P * Hh) out[(t % Hh) * P + t / Hh] = (B)pb[t];
}
// a chunk of pairs [p0, p0 + n) of [P, H] laid into a block's bias: the flash layout's rows, or [H, P]
template <class B>
__global__ void pairRowsToHeadsK(const float* pb, B* out, size_t p0, size_t n, size_t P, int Hh) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n * Hh) out[(t % Hh) * P + p0 + t / Hh] = (B)pb[t];
}
// (rlo, rows: a bias holding query rows [rlo, rlo + rows) alone - several GPUs, EF2_QLO)
__global__ void pairRowsToFlashBiasK(const float* pb, half* out, size_t p0, size_t n, int T, int Hh, int stride,
                                     int rlo = 0, int rows = -1) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * Hh) return;
  int h = (int)(t % Hh); size_t p = p0 + t / Hh; int i = (int)(p / T), j = (int)(p % T);
  out[((size_t)h * (rows < 0 ? T : rows) + (i - rlo)) * stride + j] = __float2half(pb[t] * 1.4426950408889634f);
}
// several GPUs (cuda/ef2/src/mgtrunk.cuh): this rank's query rows of the token attention [EF2_QLO, EF2_QLO + EF2_QROWS) -
// its biases those rows alone, made from its rows of the pair; the attention's output rows all-gathered after each
// block. 0: every query (one GPU)
inline int EF2_QLO = 0, EF2_QROWS = 0;
// [T * T, H] -> the flash kernel's [H, T, stride] f16, log2(e)-scaled (the padding columns stay as cleared)
__global__ void pairToFlashBiasK(const float* pb, half* out, int T, int Hh, int stride) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * Hh) return;
  int h = (int)(t % Hh); size_t p = t / Hh; int i = (int)(p / T), j = (int)(p % T);
  out[((size_t)h * T + i) * stride + j] = __float2half(pb[t] * 1.4426950408889634f);
}
// [T, 4C] f32 -> f16, the flash kernel's q | k | v | gate rows (the projection's columns in that order)
__global__ void toHalfRowsK(const float* __restrict__ x, half* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && ((uintptr_t)x & 15) == 0 && ((uintptr_t)y & 7) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 v = __ldg(reinterpret_cast<const float4*>(x) + t);
      half2 h0 = __floats2half2_rn(v.x, v.y);
      half2 h1 = __floats2half2_rn(v.z, v.w);
      reinterpret_cast<uint2*>(y)[t] = make_uint2(*reinterpret_cast<uint32_t*>(&h0), *reinterpret_cast<uint32_t*>(&h1));
    }
    return;
  }
  if (t < n) y[t] = __float2half(x[t]);
}
__global__ void halfToFloatK(const half* __restrict__ x, float* __restrict__ y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && ((uintptr_t)x & 7) == 0 && ((uintptr_t)y & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      uint2 u = __ldg(reinterpret_cast<const uint2*>(x) + t);
      float2 f0 = __half22float2(*reinterpret_cast<const half2*>(&u.x));
      float2 f1 = __half22float2(*reinterpret_cast<const half2*>(&u.y));
      reinterpret_cast<float4*>(y)[t] = make_float4(f0.x, f0.y, f1.x, f1.y);
    }
    return;
  }
  if (t < n) y[t] = __half2float(x[t]);
}
__global__ void gatherTokensK(const float* __restrict__ perToken, const int* __restrict__ atomToToken,
                              const float* __restrict__ mask, float* __restrict__ q, int A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)perToken | (uintptr_t)q) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = (size_t)A * c4;
    if (t >= n4) return;
    int a = (c4 == 32) ? (int)(t >> 5) : (int)(t / (size_t)c4);
    int c = (c4 == 32) ? (int)(t & 31) : (int)(t % (size_t)c4);
    int token = __ldg(mask + a) != 0.f ? __ldg(atomToToken + a) : 0;
    float4 qv = reinterpret_cast<const float4*>(q)[t];
    float4 pv = __ldg(reinterpret_cast<const float4*>(perToken + (size_t)token * C) + c);
    reinterpret_cast<float4*>(q)[t] = make_float4(qv.x + pv.x, qv.y + pv.y, qv.z + pv.z, qv.w + pv.w);
    return;
  }
  if (t >= (size_t)A * C) return;
  int a = (int)(t / C), c = (int)(t % C);
  int token = mask[a] != 0.f ? atomToToken[a] : 0;
  q[t] += perToken[(size_t)token * C + c];
}
__global__ void edmCombineK(const float* xNoisy, const float* r, float* out, const float* level, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = level[2] * xNoisy[t] + level[3] * r[t];
}

struct Denoiser {
  int T, A, Cz, Ct, heads, tokenBlocks, Si, atomBlocks;
  float sigma;
  Atoms atoms;                    // the diffusion encoder's c0, rope table, ranks
  // per token block, [H, T, T] (the pair itself is not kept): f16 under --fast, f32 otherwise
  // (f16 costs the float32 path 5e-5 on the denoiser)
  std::vector<void*> biases; bool biasHalf;
  // flashBias (--fast, a head width the flash kernel has): the biases instead in cuda/af3's flash layout -
  // [H, T, stride] f16, log2(e)-scaled, rows padded to `stride` - and the token blocks' attention on that
  // kernel (scores never written), as cuda/af3's own denoiser runs
  bool flashBias = false; int stride = 0;
  const float* sInputs;
  // every token block's projections of the single alone, one batched GEMM a step: entry e of G [T, 72 Ct]
  // (columns e Ct..): 4b+0..3 block b's attention gate, shift, transition gate, shift (from LN(single)
  // scaled by that adaLN's singleScale: snScaled [2 blocks, T, Ct]); 48+2b, 48+2b+1 its two out gates
  float* single; float* snScaled; float* G; float* scales; const void** ptrs; int entries;
  float* level;                   // NoiseLevel on the device
  float* encMod = nullptr;        // precomputed silu(c0) @ adaln for diffusionAtomEncoder blocks [atomBlocks, A, 6*Ca]
  float* decMod = nullptr;        // precomputed silu(c0) @ adaln for diffusionAtomDecoder blocks [atomBlocks, A, 6*Ca]
};

inline bool TOKEN_FLASH = true;   // --no-token-flash: the token blocks' attention through the [H, T, T] scores
inline bool SAMPLER16 = true;     // --fast: the denoiser's GEMMs on f16 tensor cores (--no-sampler16: TF32)
inline Denoiser makeDenoiser(int T, int A, const float* zTrunk, const float* sInputs, bool check) {
  Denoiser d{};
  d.T = T; d.A = A; d.Cz = (int)M.meta("meta/pairChannels"); d.Ct = (int)M.meta("meta/tokenChannels2");
  d.heads = (int)M.meta("meta/tokenHeads"); d.tokenBlocks = (int)M.meta("meta/tokenBlocks");
  d.Si = (int)M.meta("meta/singleInputs"); d.atomBlocks = (int)M.meta("meta/atomBlocks"); d.sigma = (float)M.meta("meta/sigmaData");
  d.sInputs = sInputs;
  size_t P = (size_t)T * T; int Cz = d.Cz;
  // row chunks of 64 MB: neither [z | rel_pos] nor a normalised copy of the pair is ever whole
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (8 * (size_t)Cz));
  float* joined = scratch<float>("dc.joined", chunk * 2 * Cz);
  d.biasHalf = FAST;
  int D = d.Ct / d.heads;
  d.flashBias = FAST && TOKEN_FLASH && (D == 16 || D == 32 || D == 48 || D == 64);
  d.stride = (T + 7) / 8 * 8;
  // 🔴 STREAMED where the conditioning pair does not fit: every step below is row-wise - [z | rel_pos], its norm, the
  // projection, the two transitions, each block's norm and bias projection - so each chunk of rows goes through all of
  // them and into every block's bias at once, and the f32 pair [P, Cz] (16 GB at 3,952 tokens, where ESMFold2 ran out)
  // never exists. Elsewhere the whole-pair path below, unchanged (its GEMMs chunked as before: the streamed one's
  // transitions see other chunk boundaries, so it rounds differently)
  // (the question includes every block's bias, which both paths allocate: asked without them, 3,952 tokens took the
  // whole-pair path and ran out allocating the biases beside the 16 GB pair)
  const size_t biasBytes = (size_t)d.tokenBlocks * d.heads * (d.flashBias ? (size_t)T * d.stride : P) * (d.biasHalf ? 2 : 4);
  const bool shardedQ = EF2_QROWS > 0;
  if (shardedQ && !d.flashBias) { fprintf(stderr, "several GPUs: the token attention wants the flash form (--fast)\n"); exit(1); }
  const int biasRows = shardedQ ? EF2_QROWS : T;
  if (shardedQ || BIG_FORCED || !roomFor(P * Cz * 4 + P * d.heads * 4 + biasBytes)) {
    if (check) { fprintf(stderr, "the oracle check wants the whole conditioning pair\n"); exit(1); }
    for (int b = 0; b < d.tokenBlocks; ++b) {
      if (d.flashBias) {
        half* h = dallocT<half>((size_t)d.heads * biasRows * d.stride);
        CK(cudaMemsetAsync(h, 0, (size_t)d.heads * biasRows * d.stride * 2, STREAM));
        d.biases.push_back(h);
      } else if (d.biasHalf) d.biases.push_back(dallocT<half>(P * d.heads));
      else d.biases.push_back(dalloc(P * d.heads));
    }
    float* rows = scratch<float>("dc.rows", chunk * Cz);
    float* pn = scratch<float>("dc.pn", chunk * Cz); float* pb = scratch<float>("dc.pbRows", chunk * d.heads);
    const size_t pBeg = shardedQ ? (size_t)EF2_QLO * T : 0, pEnd = shardedQ ? (size_t)(EF2_QLO + EF2_QROWS) * T : P;
    for (size_t p0 = pBeg; p0 < pEnd; p0 += chunk) {
      size_t n = std::min(chunk, pEnd - p0);
      joinPairRelK<<<blocks(n * 2 * Cz), 256, 0, STREAM>>>(zTrunk, relIdx(), joined, p0, n, T, Cz, EF2_Z16);
      layerNorm(joined, joined, n, 2 * Cz, F("diffusion/zInputNorm/scale"), F("diffusion/zInputNorm/offset"));
      gemm(joined, F("diffusion/zProjection"), rows, n, 2 * Cz, Cz);
      for (int l = 0; l < 2; ++l) transitionLayer(rows, n, Cz, "diffusion/zTransitions/" + std::to_string(l) + "/");
      for (int b = 0; b < d.tokenBlocks; ++b) {
        std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/attention/";
        layerNorm(rows, pn, n, Cz, F(B + "pairNormScale"), F(B + "pairNormOffset"));
        gemm(pn, F(B + "pairBiasWeights"), pb, n, Cz, d.heads);
        if (d.flashBias) pairRowsToFlashBiasK<<<blocks(n * d.heads), 256, 0, STREAM>>>(pb, (half*)d.biases[b], p0, n, T, d.heads, d.stride,
                                                                                    shardedQ ? EF2_QLO : 0, biasRows);
        else if (d.biasHalf) pairRowsToHeadsK<half><<<blocks(n * d.heads), 256, 0, STREAM>>>(pb, (half*)d.biases[b], p0, n, P, d.heads);
        else pairRowsToHeadsK<float><<<blocks(n * d.heads), 256, 0, STREAM>>>(pb, (float*)d.biases[b], p0, n, P, d.heads);
      }
    }
    releaseScratch({ "dc.joined", "dc.rows", "dc.pn", "dc.pbRows", "dtr." });
  } else {
  float* pair = scratch<float>("dc.pair", P * Cz);
  for (size_t p0 = 0; p0 < P; p0 += chunk) {
    size_t n = std::min(chunk, P - p0);
    joinPairRelK<<<blocks(n * 2 * Cz), 256, 0, STREAM>>>(zTrunk, relIdx(), joined, p0, n, T, Cz, EF2_Z16);
    layerNorm(joined, joined, n, 2 * Cz, F("diffusion/zInputNorm/scale"), F("diffusion/zInputNorm/offset"));
    gemm(joined, F("diffusion/zProjection"), pair + p0 * Cz, n, 2 * Cz, Cz);
  }
  for (int l = 0; l < 2; ++l) transitionLayer(pair, P, Cz, "diffusion/zTransitions/" + std::to_string(l) + "/");
  if (check) checkOracle("diffusion conditioning pair", pair, P * Cz, "o/cond/pair");
  float* pn = scratch<float>("dc.pn", chunk * Cz); float* pb = scratch<float>("dc.pb", P * d.heads);
  for (int b = 0; b < d.tokenBlocks; ++b) {
    std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/attention/";
    for (size_t p0 = 0; p0 < P; p0 += chunk) {
      size_t n = std::min(chunk, P - p0);
      layerNorm(pair + p0 * Cz, pn, n, Cz, F(B + "pairNormScale"), F(B + "pairNormOffset"));
      gemm(pn, F(B + "pairBiasWeights"), pb + p0 * d.heads, n, Cz, d.heads);
    }
    void* bias;
    if (d.flashBias) {
      half* h = dallocT<half>((size_t)d.heads * T * d.stride);
      CK(cudaMemsetAsync(h, 0, (size_t)d.heads * T * d.stride * 2, STREAM));
      pairToFlashBiasK<<<blocks(P * d.heads), 256, 0, STREAM>>>(pb, h, T, d.heads, d.stride);
      bias = h;
    } else if (d.biasHalf) { half* h = dallocT<half>(P * d.heads); pairToHeadsK<<<blocks(P * d.heads), 256, 0, STREAM>>>(pb, h, P, d.heads); bias = h; }
    else { float* f = dalloc(P * d.heads); pairToHeadsK<<<blocks(P * d.heads), 256, 0, STREAM>>>(pb, f, P, d.heads); bias = f; }
    d.biases.push_back(bias);
  }
  if ((size_t)P * Cz * 4 > ((size_t)128 << 20)) releaseScratch();     // the joined and projected pairs (a large input)
  }
  d.atoms = prepareAtoms(A, "diffusionAtomEncoder");
  {
    struct Restore { bool was = GEMM16; ~Restore() { GEMM16 = was; } } restore;
    GEMM16 = FAST && SAMPLER16;
    int Ca = d.atoms.ctx.C, ab = d.atomBlocks;
    size_t acVec = (Ca & 3) == 0 ? (size_t)A * (Ca >> 2) : (size_t)A * Ca;
    size_t modStride = (size_t)A * 6 * Ca;
    d.encMod = dalloc(ab * modStride);
    d.decMod = dalloc(ab * modStride);
    float* sc = scratch<float>("atom.silu", (size_t)A * Ca);
    siluK<<<blocks(acVec), 256, 0, STREAM>>>(d.atoms.c0, sc, (size_t)A * Ca);
    for (int b = 0; b < ab; ++b) {
      gemm(sc, F("diffusionAtomEncoder/blocks/" + std::to_string(b) + "/adaln"), d.encMod + b * modStride, A, Ca, 6 * Ca);
      gemm(sc, F("diffusionAtomDecoder/blocks/" + std::to_string(b) + "/adaln"), d.decMod + b * modStride, A, Ca, 6 * Ca);
    }
  }
  int Ct = d.Ct, nb = d.tokenBlocks;
  d.entries = 6 * nb;
  d.single = dalloc((size_t)T * Ct); d.snScaled = dalloc((size_t)2 * nb * T * Ct);
  d.G = dalloc((size_t)T * d.entries * Ct); d.scales = dalloc((size_t)2 * nb * Ct);
  std::vector<const void*> w(d.entries), x(d.entries), y(d.entries);
  for (int b = 0; b < nb; ++b) {
    std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/";
    const char* part[2] = {"attention/", "transition/"};
    for (int k = 0; k < 2; ++k) {
      std::string P = B + part[k];
      CK(cudaMemcpyAsync(d.scales + (size_t)(2 * b + k) * Ct, F(P + "adaln/singleScale"), Ct * 4, cudaMemcpyDeviceToDevice, STREAM));
      const float* in = d.snScaled + (size_t)(2 * b + k) * T * Ct;
      w[4 * b + 2 * k] = F(P + "adaln/gateWeights"); x[4 * b + 2 * k] = in;
      w[4 * b + 2 * k + 1] = F(P + "adaln/shiftWeights"); x[4 * b + 2 * k + 1] = in;
      w[4 * nb + 2 * b + k] = F(P + "outGateWeights"); x[4 * nb + 2 * b + k] = d.single;
    }
  }
  for (int e = 0; e < d.entries; ++e) y[e] = d.G + (size_t)e * Ct;
  std::vector<const void*> all(w); all.insert(all.end(), x.begin(), x.end()); all.insert(all.end(), y.begin(), y.end());
  d.ptrs = (const void**)upload((const uintptr_t*)all.data(), all.size());
  d.level = dalloc(4);
  return d;
}

// single [T, Ct] at noise level t
inline void conditioningSingle(const Denoiser& d, float* single) {
  int T = d.T, Ct = d.Ct;
  float* sn = scratch<float>("dc.sn", (size_t)T * d.Si);
  layerNorm(d.sInputs, sn, T, d.Si, F("diffusion/sInputNorm/scale"), F("diffusion/sInputNorm/offset"));
  gemm(sn, F("diffusion/sProjection"), single, T, d.Si, Ct);
  int nf = (int)M.len("f/diffusion/fourier/weights");
  float* four = scratch<float>("dc.fourier", nf); float* noise = scratch<float>("dc.noise", Ct);
  fourierK<<<blocks(nf), 256, 0, STREAM>>>(F("diffusion/fourier/weights"), F("diffusion/fourier/offsets"), d.level, four, nf);
  layerNorm(four, four, 1, nf, F("diffusion/noiseNorm/scale"), F("diffusion/noiseNorm/offset"));
  gemm(four, F("diffusion/noiseProjection"), noise, 1, nf, Ct);
  addRowK<<<blocks((Ct & 3) == 0 ? (size_t)T * (Ct >> 2) : (size_t)T * Ct), 256, 0, STREAM>>>(single, noise, T, Ct);
  for (int l = 0; l < 2; ++l) transitionLayer(single, T, Ct, "diffusion/sTransitions/" + std::to_string(l) + "/");
}

// the single's projections for every token block, one batched GEMM
inline void singleProjections(const Denoiser& d) {
  int T = d.T, C = d.Ct, n = 2 * d.tokenBlocks;
  float* sn = scratch<float>("dn.sn0", (size_t)T * C);
  layerNorm(d.single, sn, T, C, nullptr, nullptr);
  scaleCopiesK<<<blocks((C & 3) == 0 ? (size_t)n * T * (C >> 2) : (size_t)n * T * C), 256, 0, STREAM>>>(sn, d.scales, d.snScaled, T, C, n);
  const float one = 1.f, zero = 0.f;
  cublasComputeType_t ct = GEMM16 ? CUBLAS_COMPUTE_32F_FAST_16F : FAST ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  CB(cublasGemmBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, C, T, C, &one, d.ptrs, CUDA_R_32F, C, d.ptrs + d.entries,
                         CUDA_R_32F, C, &zero, (void* const*)(d.ptrs + 2 * d.entries), CUDA_R_32F, d.entries * C,
                         d.entries, ct, FAST ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
}
// adaLN: sigmoid(LN(s; scale) @ gate + gateBias) * LN(a) + LN(s; scale) @ shift, the projections from G
inline void adaLN(const Denoiser& d, float* a, float* out, int e, const std::string& B,
                  const float* o = nullptr, const float* og = nullptr, const float* ogb = nullptr) {
  int T = d.T, C = d.Ct, ld = d.entries * C;
  adaLNK<<<(unsigned)T, 256, 0, STREAM>>>(a, d.G + (size_t)e * C, F(B + "gateBias"), d.G + (size_t)(e + 1) * C,
                                          out, T, C, ld, o, og, ogb);
}

inline void tokenBlock(const Denoiser& d, float* a, int b) {
  int T = d.T, C = d.Ct, Hh = d.heads, D = C / Hh, ld = d.entries * C, nb = d.tokenBlocks;
  size_t tcVec = (C & 3) == 0 ? (size_t)T * (C >> 2) : (size_t)T * C;
  std::string B = "diffusion/tokenBlocks/" + std::to_string(b) + "/";
  float* x = scratch<float>("tb.x", (size_t)T * C);
  float* o = scratch<float>("tb.o", (size_t)T * C);
  if (b == 0) {
    adaLN(d, a, x, 4 * b, B + "attention/adaln/");
  } else {
    std::string Bp = "diffusion/tokenBlocks/" + std::to_string(b - 1) + "/";
    adaLN(d, a, x, 4 * b, B + "attention/adaln/",
          o, d.G + (size_t)(4 * nb + 2 * (b - 1) + 1) * C, F(Bp + "transition/outGateBias"));
  }
  if (d.flashBias) {
    // q | k | v | gate in one GEMM (the weights' columns concatenated once), then cuda/af3's flash kernel:
    // the query bias, QK^T, the pair bias, the softmax, PV and the gate, no [H, T, T] scores at all
    // (they were 52 of the sampler's 139 GPU ms at 1,044 tokens: the score matrix's softmax and two GEMMs)
    std::string wq = concatColumns("f/" + B + "attention/qkvg~", C, {{"f/" + B + "attention/queryWeights", C, false},
                                                                     {"f/" + B + "attention/kvWeights", 2 * C, false},
                                                                     {"f/" + B + "attention/gateWeights", C, false}});
    float* qkvg = scratch<float>("tb.qkvg", (size_t)T * 4 * C);
    gemm(x, W(wq), qkvg, T, C, 4 * C);
    half* qh = scratch<half>("tb.qkvgh", (size_t)(T + 128) * 4 * C);    // (padding: the last query block's rows)
    toHalfRowsK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(qkvg, qh, (size_t)T * 4 * C);
    mg::Shared* ohS = EF2_QROWS ? &mg::shared("ef2.tok.oh", (size_t)T * C * 2) : nullptr;
    half* oh = ohS ? (half*)ohS->local : scratch<half>("tb.oh", (size_t)T * C);
    if (EF2_QROWS) { FLASH_QLO = EF2_QLO; FLASH_QHI = EF2_QLO + EF2_QROWS; FLASH_BIAS_ROWS = EF2_QROWS; }
    flashGrid<half>(qh, (const half*)d.biases[b], d.stride, nullptr, oh, T, Hh, D, 0, 1, false, 1.f / sqrtf((float)D),
                    F(B + "attention/queryBias"));
    FLASH_QLO = FLASH_QHI = FLASH_BIAS_ROWS = 0;
    if (ohS) mg::allGather(*ohS, (size_t)EF2_QLO * C * 2, (size_t)EF2_QROWS * C * 2);   // (every rank's query rows)
    float* ctx = scratch<float>("tb.ctx", (size_t)T * C);
    halfToFloatK<<<blocks(tcVec), 256, 0, STREAM>>>(oh, ctx, (size_t)T * C);
    gemm(ctx, F(B + "attention/outWeights"), o, T, C, C);
  } else {
  float* q = scratch<float>("tb.q", (size_t)T * C); float* kv = scratch<float>("tb.kv", (size_t)T * 2 * C);
  float* gt = scratch<float>("tb.gate", (size_t)T * C);
  gemm(x, F(B + "attention/queryWeights"), q, T, C, C);
  addBias(q, F(B + "attention/queryBias"), T, C);
  gemm(x, F(B + "attention/kvWeights"), kv, T, C, 2 * C);
  gemm(x, F(B + "attention/gateWeights"), gt, T, C, C);
  float* S = scratch<float>("tb.scores", (size_t)Hh * T * T); float* ctx = scratch<float>("tb.ctx", (size_t)T * C);
  const float one = 1.f, zero = 0.f;
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, T, T, D, &one, kv, 2 * C, D, q, C, D, &zero, S, T,
                               (long long)T * T, Hh));
  if (d.biasHalf) biasSoftmaxK<<<(unsigned)(Hh * T), 256, 0, STREAM>>>(S, (const half*)d.biases[b], T, 1.f / sqrtf((float)D));
  else biasSoftmaxK<<<(unsigned)(Hh * T), 256, 0, STREAM>>>(S, (const float*)d.biases[b], T, 1.f / sqrtf((float)D));
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, D, T, T, &one, kv + C, 2 * C, D, S, T, (long long)T * T,
                               &zero, ctx, C, D, Hh));
  sigmoidMulK<<<blocks(tcVec), 256, 0, STREAM>>>(ctx, gt, nullptr, T, C, C);
  gemm(ctx, F(B + "attention/outWeights"), o, T, C, C);
  }
  // the conditioned transition (fusing the attention's gateAdd in place)
  adaLN(d, a, x, 4 * b + 2, B + "transition/adaln/",
        o, d.G + (size_t)(4 * nb + 2 * b) * C, F(B + "attention/outGateBias"));
  int Hd = (int)dimOf("f/" + B + "transition/outWeights", 0);
  float* w = scratch<float>("tb.wide", (size_t)T * 2 * Hd); float* g = scratch<float>("tb.gated", (size_t)T * Hd);
  gemm(x, F(B + "transition/swishWeights"), w, T, C, 2 * Hd);
  swigluK<<<blocks((Hd & 3) == 0 ? (size_t)T * (Hd >> 2) : (size_t)T * Hd), 256, 0, STREAM>>>(w, g, T, Hd);
  gemm(g, F(B + "transition/outWeights"), o, T, Hd, C);
  if (b + 1 == nb)
    gateAddK<<<blocks(tcVec), 256, 0, STREAM>>>(a, o, d.G + (size_t)(4 * nb + 2 * b + 1) * C, F(B + "transition/outGateBias"), T, C, ld);
}

// one denoiser call: x_noisy [A, 3] at the noise level in d.level -> x_denoised [A, 3]; nothing here
// reads the host, so the sampler captures it once and replays it at every level
inline void denoiseAtLevel(const Denoiser& d, const float* xNoisy, float* xDenoised, bool check = false) {
  struct Restore { bool was = GEMM16; ~Restore() { GEMM16 = was; } } restore;
  GEMM16 = FAST && SAMPLER16;
  int T = d.T, A = d.A, Ct = d.Ct; const AtomCtx& ac = d.atoms.ctx; int Ca = ac.C;
  float* single = d.single;
  conditioningSingle(d, single);
  singleProjections(d);
  if (check) checkOracle("diffusion conditioning single", single, (size_t)T * Ct, "o/cond/single");
  float* r6 = scratch<float>("dn.r6", (size_t)A * 6);
  coordsInputK<<<blocks((size_t)A * 6), 256, 0, STREAM>>>(xNoisy, d.level, r6, A);
  float* q = scratch<float>("dn.q", (size_t)A * Ca);
  gemm(r6, F("diffusionAtomEncoder/coordsLinear"), q, A, 6, Ca);
  addK<<<blocks((Ca & 3) == 0 ? (size_t)A * (Ca >> 2) : (size_t)A * Ca), 256, 0, STREAM>>>(q, d.atoms.c0, (size_t)A * Ca);
  swaStack(ac, q, d.atoms.c0, "diffusionAtomEncoder", d.atomBlocks, DIFFUSION_HALF_WINDOW, d.encMod);
  float* tok = scratch<float>("dn.tok", (size_t)A * Ct);
  gemm(q, F("diffusionAtomEncoder/toToken"), tok, A, Ca, Ct);
  reluK<<<blocks((Ct & 3) == 0 ? (size_t)A * (Ct >> 2) : (size_t)A * Ct), 256, 0, STREAM>>>(tok, (size_t)A * Ct);
  float* a = scratch<float>("dn.a", (size_t)T * Ct);
  scatterMeanK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(tok, ac.tokenStart, ac.tokenAtoms, ac.mask, a, T, Ct, Ct);
  float* sn = scratch<float>("dn.sn", (size_t)T * Ct); float* st = scratch<float>("dn.st", (size_t)T * Ct);
  layerNorm(single, sn, T, Ct, F("diffusion/stepNorm/scale"), F("diffusion/stepNorm/offset"));
  gemm(sn, F("diffusion/singleToToken"), st, T, Ct, Ct);
  addK<<<blocks((Ct & 3) == 0 ? (size_t)T * (Ct >> 2) : (size_t)T * Ct), 256, 0, STREAM>>>(a, st, (size_t)T * Ct);
  for (int b = 0; b < d.tokenBlocks; ++b) tokenBlock(d, a, b);
  layerNorm(a, a, T, Ct, F("diffusion/tokenNorm/scale"), F("diffusion/tokenNorm/offset"));
  // the decoder: q (the encoder's output, the skip) + the tokens gathered back
  float* pt = scratch<float>("dn.pt", (size_t)T * Ca);
  gemm(a, F("diffusionAtomDecoder/tokenToAtom"), pt, T, Ct, Ca);
  gatherTokensK<<<blocks((Ca & 3) == 0 ? (size_t)A * (Ca >> 2) : (size_t)A * Ca), 256, 0, STREAM>>>(pt, Idev("atom_to_token"), ac.mask, q, A, Ca);
  swaStack(ac, q, d.atoms.c0, "diffusionAtomDecoder", d.atomBlocks, DIFFUSION_HALF_WINDOW, d.decMod);
  layerNorm(q, q, A, Ca, F("diffusionAtomDecoder/norm/scale"), F("diffusionAtomDecoder/norm/offset"));
  float* r = scratch<float>("dn.r", (size_t)A * 3);
  gemm(q, F("diffusionAtomDecoder/outputLinear"), r, A, Ca, 3);
  edmCombineK<<<blocks((size_t)A * 3), 256, 0, STREAM>>>(xNoisy, r, xDenoised, d.level, A * 3);
}
inline void freeDenoiser(Denoiser& d) {
  CK(cudaStreamSynchronize(STREAM));
  for (void* b : d.biases) CK(cudaFree(b));
  for (void* p : {(void*)d.single, (void*)d.snScaled, (void*)d.G, (void*)d.scales, (void*)d.ptrs, (void*)d.level,
                  (void*)d.encMod, (void*)d.decMod})
    if (p) CK(cudaFree(p));
  freeAtoms(d.atoms);
  d.biases.clear();
}
inline void setLevel(const Denoiser& d, float t) {
  NoiseLevel lv = noiseLevel(t, d.sigma);
  CK(cudaMemcpyAsync(d.level, lv.v, sizeof lv.v, cudaMemcpyHostToDevice, STREAM));
}
inline void denoise(const Denoiser& d, const float* xNoisy, float t, float* xDenoised, bool check = false) {
  setLevel(d, t);
  denoiseAtLevel(d, xNoisy, xDenoised, check);
}
