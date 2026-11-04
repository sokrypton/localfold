// cuda/grad's layers, forward and backward: LayerNorm, gated multi-head attention with a shared (pair) bias,
// AF2's fused triangle multiplication, the outer product mean, a ReLU transition and global attention. Each forward
// writes its output and the few tensors its backward cannot cheaply recompute into a Save the caller keeps for the
// block; each backward takes the output's gradient, accumulates every parameter's gradient, and writes (or adds)
// the input's.
//
// Attention never keeps its logits: the backward recomputes them a batch chunk at a time (ATTN_CHUNK_BYTES), so a
// triangle attention at L residues holds O(L^2 C) and not O(L^3 H).
#pragma once
#include "core.cuh"

namespace grad {

inline size_t ATTN_CHUNK_BYTES = (size_t)256 << 20;
#ifndef FLASH_DEFAULT
#define FLASH_DEFAULT true
#endif
inline bool FLASH = FLASH_DEFAULT;   // the tensor-core flash attention (flash.cuh) where it has a kernel; false: the bmm path

// ---------------------------------------------------------------- LayerNorm (eps 1e-5, haiku's)
struct LayerNorm {
  Param g, b;
  int C = 0;
  real eps = 1e-5f;
};

// an element as real, from real or (float builds) bf16 storage
__device__ __forceinline__ real ldr(const real* p) { return *p; }
__device__ __forceinline__ void str(real* p, real v) { *p = v; }
#ifndef GRAD_F64
__device__ __forceinline__ float ldr(const bf* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void str(bf* p, float v) { *p = __float2bfloat16(v); }
#endif
// one warp a row; mean and rstd saved for the backward
template <class Y, class XC = Y>   // (Y: real, or bf where a module keeps its normalised input in bf16; xc, when not
                                   // null, a bf16 copy of x for the backward)
__global__ void lnFwdK(const real* __restrict__ x, const real* __restrict__ g, const real* __restrict__ b,
                       Y* __restrict__ y, real* __restrict__ mean, real* __restrict__ rstd, size_t rows, int C,
                       real eps, XC* __restrict__ xc = nullptr) {
  size_t r = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (r >= rows) return;
  const real* xr = x + r * C;
  real s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  real mu = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { real d = xr[c] - mu; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  real rs = rsqrt(v / C + eps);
  for (int c = lane; c < C; c += 32) {
    y[r * C + c] = (Y)((xr[c] - mu) * rs * (g ? g[c] : (real)1) + (b ? b[c] : (real)0));
    if (xc) xc[r * C + c] = (XC)xr[c];
  }
  if (lane == 0) { mean[r] = mu; rstd[r] = rs; }
}
// dx = rstd (dxhat - mean(dxhat) - xhat mean(dxhat xhat)), dxhat = dy g; dg += dy xhat, db += dy (atomics a block)
__global__ void lnBwdK(const real* __restrict__ x, const real* __restrict__ dy, const real* __restrict__ g,
                       const real* __restrict__ mean, const real* __restrict__ rstd, real* __restrict__ dx,
                       real* __restrict__ dg, real* __restrict__ db, size_t rows, int C, bool accumulate) {
  extern __shared__ real sh[];           // [2 C] block partials of dg, db
  for (int c = threadIdx.x; c < 2 * C; c += blockDim.x) sh[c] = 0;
  __syncthreads();
  int warps = blockDim.x / 32, lane = threadIdx.x & 31;
  for (size_t r = (size_t)blockIdx.x * warps + threadIdx.x / 32; r < rows; r += (size_t)gridDim.x * warps) {
    const real *xr = x + r * C, *dr = dy + r * C;
    real mu = mean[r], rs = rstd[r], s1 = 0, s2 = 0;
    for (int c = lane; c < C; c += 32) {
      real xh = (xr[c] - mu) * rs, dh = dr[c] * (g ? g[c] : (real)1);
      s1 += dh; s2 += dh * xh;
      atomicAdd(&sh[c], dr[c] * xh);
      atomicAdd(&sh[C + c], dr[c]);
    }
    for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(~0u, s1, o); s2 += __shfl_xor_sync(~0u, s2, o); }
    s1 /= C; s2 /= C;
    for (int c = lane; c < C; c += 32) {
      real xh = (xr[c] - mu) * rs, v = rs * (dr[c] * (g ? g[c] : (real)1) - s1 - xh * s2);
      dx[r * C + c] = accumulate ? dx[r * C + c] + v : v;
    }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) { if (dg) atomicAdd(&dg[c], sh[c]); if (db) atomicAdd(&db[c], sh[C + c]); }
}

struct LnSave { const real* x; real* mean; real* rstd; const void* xb = nullptr; };   // (xb: x in bf16, the backward's)
inline LnSave lnForward(const LayerNorm& ln, const real* x, real* y, size_t rows) {
  LnSave s{x, ARENA.get(rows), ARENA.get(rows)};
  lnFwdK<<<blocks(rows, 8), 256, 0, STREAM>>>(x, ln.g.w, ln.b.w, y, s.mean, s.rstd, rows, ln.C, ln.eps);
  return s;
}
#ifndef GRAD_F64
// BF16: the normalised output in bf16. (Keeping x in bf16 too, s.xb, for the backward to read half the bytes was
// measured: 0.1-0.7 ms off a block's backward, as much onto its forward, and a pair's worth more saved - not taken;
// lnBackward still reads s.xb where a caller sets it.)
inline LnSave lnForward(const LayerNorm& ln, const real* x, bf* y, size_t rows) {
  LnSave s{x, ARENA.get(rows), ARENA.get(rows)};
  lnFwdK<<<blocks(rows, 8), 256, 0, STREAM>>>(x, ln.g.w, ln.b.w, y, s.mean, s.rstd, rows, ln.C, ln.eps);
  return s;
}
#endif
// the same with C = 32 N: a lane holds its N channels of the row (x and dy read once) and its own partial dg, db
// across every row it takes; the block's warps reduce through shared memory once, at the end
template <int N, class X = real, class DY = real, class DX = real>   // (DX bf16: a bf16 running gradient)
__global__ void lnBwdRegK(const X* __restrict__ x, const DY* __restrict__ dy, const real* __restrict__ g,
                          const real* __restrict__ mean, const real* __restrict__ rstd, DX* __restrict__ dx,
                          real* __restrict__ dg, real* __restrict__ db, size_t rows, bool accumulate) {
  constexpr int C = 32 * N;
  extern __shared__ real sh[];           // [warps][2 C]
  const int warps = blockDim.x / 32, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  real gg[N], pg[N], pb[N];
#pragma unroll
  for (int k = 0; k < N; ++k) { gg[k] = g ? g[lane + 32 * k] : (real)1; pg[k] = pb[k] = 0; }
  for (size_t r = (size_t)blockIdx.x * warps + warp; r < rows; r += (size_t)gridDim.x * warps) {
    const X* xr = x + r * C;
    const DY* dr = dy + r * C;
    const real mu = mean[r], rs = rstd[r];
    real xh[N], dv[N], s1 = 0, s2 = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      xh[k] = (ldr(xr + lane + 32 * k) - mu) * rs; dv[k] = ldr(dr + lane + 32 * k);
      real dh = dv[k] * gg[k];
      s1 += dh; s2 += dh * xh[k];
      pg[k] += dv[k] * xh[k]; pb[k] += dv[k];
    }
    for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(~0u, s1, o); s2 += __shfl_xor_sync(~0u, s2, o); }
    s1 /= C; s2 /= C;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      real v = rs * (dv[k] * gg[k] - s1 - xh[k] * s2);
      DX* o = dx + r * C + lane + 32 * k;
      str(o, accumulate ? ldr(o) + v : v);
    }
  }
#pragma unroll
  for (int k = 0; k < N; ++k) { sh[warp * 2 * C + lane + 32 * k] = pg[k]; sh[warp * 2 * C + C + lane + 32 * k] = pb[k]; }
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    real a = 0, b = 0;
    for (int w = 0; w < warps; ++w) { a += sh[w * 2 * C + c]; b += sh[w * 2 * C + C + c]; }
    if (dg) atomicAdd(&dg[c], a);
    if (db) atomicAdd(&db[c], b);
  }
}
#ifndef GRAD_F64
__global__ void bfToFK(const bf* x, float* y, size_t n);
inline void lnBackward(LayerNorm& ln, const LnSave& s, const real* dy, real* dx, size_t rows, bool accumulate);
// x from its bf16 copy (s.xb) and/or dy in bf16, where C is a multiple of 32 the register kernel covers
template <class DY, class DX = real>
inline bool lnBackwardBf(LayerNorm& ln, const LnSave& s, const DY* dy, DX* dx, size_t rows, bool accumulate) {
  unsigned gr = (unsigned)std::min<size_t>(blocks(rows, 8), 1024);
  const size_t sh = 8 * 2 * ln.C * sizeof(real);
  const bf* xb = static_cast<const bf*>(s.xb);
#define LNREGB(N) case N: if (xb) lnBwdRegK<N, bf, DY, DX><<<gr, 256, sh, STREAM>>>(xb, dy, ln.g.w, s.mean, s.rstd, dx, ln.g.g, ln.b.g, rows, accumulate); \
                          else lnBwdRegK<N, real, DY, DX><<<gr, 256, sh, STREAM>>>(s.x, dy, ln.g.w, s.mean, s.rstd, dx, ln.g.g, ln.b.g, rows, accumulate); return true;
  if (ln.C % 32 == 0 && sh <= 48 * 1024) switch (ln.C / 32) { LNREGB(1) LNREGB(2) LNREGB(3) LNREGB(4) LNREGB(6) LNREGB(8) LNREGB(12) default: break; }
#undef LNREGB
  return false;
}
inline void lnBackward(LayerNorm& ln, const LnSave& s, const bf* dy, real* dx, size_t rows, bool accumulate) {
  if (lnBackwardBf(ln, s, dy, dx, rows, accumulate)) return;
  Scratch scratch;
  real* t = ARENA.get(rows * ln.C);
  bfToFK<<<blocks(rows * ln.C), 256, 0, STREAM>>>(dy, t, rows * ln.C);
  LnSave f = s; f.xb = nullptr;
  if (s.xb) { real* xf = ARENA.get(rows * ln.C); bfToFK<<<blocks(rows * ln.C), 256, 0, STREAM>>>(static_cast<const bf*>(s.xb), xf, rows * ln.C); f.x = xf; }
  lnBackward(ln, f, t, dx, rows, accumulate);
}
// into a bf16 running gradient (BF16 mode's pair track)
template <class DY>
inline void lnBackward(LayerNorm& ln, const LnSave& s, const DY* dy, bf* dx, size_t rows, bool accumulate) {
  if (lnBackwardBf(ln, s, dy, dx, rows, accumulate)) return;
  fprintf(stderr, "lnBackward into bf16: C %d has no register kernel\n", ln.C);
  exit(1);
}
#endif
inline void lnBackward(LayerNorm& ln, const LnSave& s, const real* dy, real* dx, size_t rows, bool accumulate) {
#ifndef GRAD_F64
  if (s.xb && lnBackwardBf(ln, s, dy, dx, rows, accumulate)) return;
#endif
  unsigned gr = (unsigned)std::min<size_t>(blocks(rows, 8), 1024);
  const size_t sh = 8 * 2 * ln.C * sizeof(real);
#define LNREG(N) case N: lnBwdRegK<N><<<gr, 256, sh, STREAM>>>(s.x, dy, ln.g.w, s.mean, s.rstd, dx, ln.g.g, ln.b.g, rows, accumulate); return;
  if (ln.C % 32 == 0 && sh <= 48 * 1024) switch (ln.C / 32) { LNREG(1) LNREG(2) LNREG(3) LNREG(4) LNREG(6) LNREG(8) LNREG(12) default: break; }
#undef LNREG
  lnBwdK<<<gr, 256, 2 * ln.C * sizeof(real), STREAM>>>(s.x, dy, ln.g.w, s.mean, s.rstd, dx, ln.g.g, ln.b.g, rows, ln.C, accumulate);
}

// an arena copy (af3.cuh keeps its own in its namespace)
inline real* cloneR(const real* x, size_t n) {
  real* y = ARENA.get(n);
  GK(cudaMemcpyAsync(y, x, n * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  return y;
}

// ---------------------------------------------------------------- elementwise pieces
__global__ void sigmoidK(real* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = 1.f / (1.f + exp(-x[t]));
}
__global__ void mulK(real* y, const real* a, const real* b, size_t n) {   // y = a b
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = a[t] * b[t];
}
// given y = a * sig (sig already the sigmoid): da = dy sig, dpre = dy a sig (1 - sig)
__global__ void gateBwdK(const real* dy, const real* a, const real* sig, real* da, real* dpre, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real s = sig[t];
  if (da) da[t] = dy[t] * s;
  dpre[t] = dy[t] * a[t] * s * (1.f - s);
}
__global__ void rowMaskK(real* x, const real* mask, size_t rows, int C) {   // x[r, :] *= mask[r]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] *= mask[t / C];
}
__global__ void reluK(real* y, const real* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmax(x[t], (real)0);
}
__global__ void reluBwdK(real* d, const real* pre, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n && pre[t] <= 0.f) d[t] = 0.f;
}
// [A, B, C] -> [B, A, C]  (accumulate: out += )
__global__ void swap01K(const real* __restrict__ in, real* __restrict__ out, int A, int B, int C, bool accumulate) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)A * B * C;
  if (t >= n) return;
  int c = t % C; size_t ab = t / C; int b = ab % B, a = (int)(ab / B);
  size_t o = ((size_t)b * A + a) * C + c;
  out[o] = accumulate ? out[o] + in[t] : in[t];
}
// (four channels a thread where C allows: one float a thread ran a pair transpose at ~450 GB/s)
__global__ void swap01V4K(const float4* __restrict__ in, float4* __restrict__ out, int A, int B, int C4, bool accumulate) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * B * C4) return;
  int c = t % C4; size_t ab = t / C4; int b = ab % B, a = (int)(ab / B);
  size_t o = ((size_t)b * A + a) * C4 + c;
  float4 v = in[t];
  if (accumulate) { const float4 w = out[o]; v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w; }
  out[o] = v;
}
inline void swap01(const real* in, real* out, int A, int B, int C, bool accumulate = false) {
  if (sizeof(real) == 4 && C % 4 == 0 && ((uintptr_t)in & 15) == 0 && ((uintptr_t)out & 15) == 0) {
    swap01V4K<<<blocks((size_t)A * B * C / 4), 256, 0, STREAM>>>(reinterpret_cast<const float4*>(in),
                                                                reinterpret_cast<float4*>(out), A, B, C / 4, accumulate);
    return;
  }
  swap01K<<<blocks((size_t)A * B * C), 256, 0, STREAM>>>(in, out, A, B, C, accumulate);
}
// [R, C] <-> [C, R]
inline void transpose(const real* in, real* out, int R, int C, bool accumulate = false) { swap01(in, out, R, C, 1, accumulate); }

// ---------------------------------------------------------------- scaled dot-product attention, any bias
// q [B, Q, ldq], k [B, K, ldk], v [B, K, ldv]: head h at columns h D .. h D + D (so a packed qkv is read in place);
// o [B, Q, ldo]. logits = scale q k^T + 1e9 (keyMask[b, k] - 1) + sum of up to two biases, each addressed with
// broadcast strides (a stride of 0 repeats it) and the key axis contiguous - AF2's pair bias [H, Q, K] is
// {sb 0, sh Q K, sq K}, a chain mask shared by every head {0, 0, K}. A bias with a `grad` gets dS accumulated there
// over everything it was broadcast across. The logits are never kept: the backward recomputes them a batch chunk
// at a time.
struct Bias {
  const real* p = nullptr;
  long long sb = 0, sh = 0, sq = 0;
  real* grad = nullptr;
};
struct Sdpa {
  int B = 1, H = 1, Q = 0, K = 0, D = 0;
  real scale = 1.f;
  const real* keyMask = nullptr;   // [B, K], 1 kept, 0 masked (-1e9)
  Bias bias[2];
  bool clip = false;                // AF2's clip of the logits to +-1e8
  const int* seqQ = nullptr;        // [Q] and [K]: a key whose id differs from the query's is excluded (a chain mask)
  const int* seqK = nullptr;
  // the bf16 backward's bias gradients summed in fixed point, bitwise the same on every run; off, float atomics
  // (5% of AF3's pairformer backward at n 384, where the bias is shared by 384 batch rows: 226 M atomics a call)
  bool deterministic = false;
};

__global__ void softmaxBiasK(real* __restrict__ S, Sdpa a, int b0, int nb) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= (size_t)nb * a.H * a.Q) return;
  int q = row % a.Q, h = (row / a.Q) % a.H, b = b0 + (int)(row / ((size_t)a.Q * a.H));
  const int K = a.K;
  real* s = S + row * K;
  const real* m = a.keyMask ? a.keyMask + (size_t)b * K : nullptr;
  const real* n0 = a.bias[0].p ? a.bias[0].p + b * a.bias[0].sb + h * a.bias[0].sh + q * a.bias[0].sq : nullptr;
  const real* n1 = a.bias[1].p ? a.bias[1].p + b * a.bias[1].sb + h * a.bias[1].sh + q * a.bias[1].sq : nullptr;
  real mx = -INFINITY;
  const int sid = a.seqQ ? a.seqQ[q] : 0;
  for (int k = lane; k < K; k += 32) {
    real v = s[k];
    if (a.seqQ && a.seqK[k] != sid) { s[k] = -INFINITY; continue; }
    if (m) v += 1e9f * (m[k] - 1.f);
    if (n0) v += n0[k];
    if (n1) v += n1[k];
    if (a.clip) v = fmin(fmax(v, (real)-1e8), (real)1e8);
    s[k] = v; mx = fmax(mx, v);
  }
  for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(~0u, mx, o));
  real sum = 0;
  for (int k = lane; k < K; k += 32) { real e = s[k] == -INFINITY ? (real)0 : exp(s[k] - mx); s[k] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  real inv = sum > 0 ? (real)1 / sum : (real)0;
  for (int k = lane; k < K; k += 32) s[k] *= inv;
}
// dS = P (dP - sum_k dP P), in place on dP; every bias with a grad gets dS accumulated
__global__ void softmaxBwdK(real* __restrict__ dP, const real* __restrict__ P, Sdpa a, int b0, int nb) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= (size_t)nb * a.H * a.Q) return;
  int q = row % a.Q, h = (row / a.Q) % a.H, b = b0 + (int)(row / ((size_t)a.Q * a.H));
  const int K = a.K;
  real* d = dP + row * K;
  const real* p = P + row * K;
  real s = 0;
  for (int k = lane; k < K; k += 32) s += d[k] * p[k];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  real* g0 = a.bias[0].grad ? a.bias[0].grad + b * a.bias[0].sb + h * a.bias[0].sh + q * a.bias[0].sq : nullptr;
  real* g1 = a.bias[1].grad ? a.bias[1].grad + b * a.bias[1].sb + h * a.bias[1].sh + q * a.bias[1].sq : nullptr;
  for (int k = lane; k < K; k += 32) {
    real v = p[k] * (d[k] - s);
    d[k] = v;
    if (g0) atomicAdd(&g0[k], v);
    if (g1) atomicAdd(&g1[k], v);
  }
}

// [B, Q, ld] (head h at h D) <-> [B, H, Q, D]
__global__ void toHeadsK(const real* __restrict__ in, int ldIn, real* __restrict__ out, int B, int Q, int H, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * Q * H * D) return;
  int d = t % D; size_t r = t / D; int h = r % H; r /= H; int q = r % Q; int b = (int)(r / Q);
  out[(((size_t)b * H + h) * Q + q) * D + d] = in[((size_t)b * Q + q) * ldIn + h * D + d];
}
__global__ void fromHeadsK(const real* __restrict__ in, real* __restrict__ out, int ldOut, int B, int Q, int H, int D,
                           real scale, bool accumulate) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * Q * H * D) return;
  int d = t % D; size_t r = t / D; int h = r % H; r /= H; int q = r % Q; int b = (int)(r / Q);
  real v = in[(((size_t)b * H + h) * Q + q) * D + d] * scale;
  real& o = out[((size_t)b * Q + q) * ldOut + h * D + d];
  o = accumulate ? o + v : v;
}
inline int sdpaChunk(const Sdpa& a, int matrices, int vectors) {
  size_t per = (size_t)matrices * a.H * a.Q * a.K * sizeof(real) + (size_t)vectors * a.H * std::max(a.Q, a.K) * a.D * sizeof(real);
  return (int)std::max<size_t>(1, std::min<size_t>(a.B, ATTN_CHUNK_BYTES / per));
}

}  // namespace grad
#ifndef GRAD_F64
#include "flash.cuh"
#endif
namespace grad {
// lse [2][B, H, Q] (optional): each query's logit max and the log of its exp-sum, which sdpaBackward takes with o to skip
// recomputing the forward. Float builds run the tensor-core flash kernels (flash.cuh) where D has one.
inline void sdpaForward(const Sdpa& a, const real* q, int ldq, const real* k, int ldk, const real* v, int ldv,
                        real* o, int ldo, real* lse = nullptr) {
  Scratch scratch;
#ifndef GRAD_F64
  if (FLASH && flash::supported(a.D)) {
    flash::runForward(a, q, ldq, k, ldk, v, ldv, o, ldo, lse ? lse : ARENA.get((size_t)2 * a.B * a.H * a.Q));
    return;
  }
#endif
  const int H = a.H, Q = a.Q, K = a.K, D = a.D;
  int chunk = sdpaChunk(a, 1, 3);
  real* S = ARENA.get((size_t)chunk * H * Q * K);
  size_t hq = (size_t)chunk * H * Q * D, hk = (size_t)chunk * H * K * D;
  real *qh = ARENA.get(hq), *kh = ARENA.get(hk), *vh = ARENA.get(hk);
  for (int b0 = 0; b0 < a.B; b0 += chunk) {
    int nb = std::min(chunk, a.B - b0);
    toHeadsK<<<blocks((size_t)nb * Q * H * D), 256, 0, STREAM>>>(q + (size_t)b0 * Q * ldq, ldq, qh, nb, Q, H, D);
    toHeadsK<<<blocks((size_t)nb * K * H * D), 256, 0, STREAM>>>(k + (size_t)b0 * K * ldk, ldk, kh, nb, K, H, D);
    toHeadsK<<<blocks((size_t)nb * K * H * D), 256, 0, STREAM>>>(v + (size_t)b0 * K * ldv, ldv, vh, nb, K, H, D);
    bmm(false, true, Q, K, D, qh, (long long)Q * D, kh, (long long)K * D, S, (long long)Q * K, nb * H, 0.f, a.scale);
    softmaxBiasK<<<blocks((size_t)nb * H * Q, 8), 256, 0, STREAM>>>(S, a, b0, nb);
    bmm(false, false, Q, D, K, S, (long long)Q * K, vh, (long long)K * D, qh, (long long)Q * D, nb * H);
    fromHeadsK<<<blocks((size_t)nb * Q * H * D), 256, 0, STREAM>>>(qh, o + (size_t)b0 * Q * ldo, ldo, nb, Q, H, D, 1.f,
                                                                    false);
  }
}
// dO [B, Q, lddo] -> dq, dk, dv (written, or added where acc*), at their own leading dimensions
inline void sdpaBackward(const Sdpa& a, const real* q, int ldq, const real* k, int ldk, const real* v, int ldv,
                         const real* dO, int lddo, real* dq, int lddq, real* dk, int lddk, real* dv, int lddv,
                         bool accQ = false, bool accK = false, bool accV = false, const real* o = nullptr,
                         int ldo = 0, const real* lse = nullptr) {
  Scratch scratch;
  const int H = a.H, Q = a.Q, K = a.K, D = a.D;
#ifndef GRAD_F64
  if (FLASH && flash::supported(D)) {
    if (!o || !lse) {                       // no saved forward: recompute o and the log-sum-exp
      real* oo = ARENA.get((size_t)a.B * Q * H * D);
      real* ll = ARENA.get((size_t)2 * a.B * H * Q);
      flash::runForward(a, q, ldq, k, ldk, v, ldv, oo, H * D, ll);
      o = oo; ldo = H * D; lse = ll;
    }
    real* Di = ARENA.get((size_t)a.B * H * Q);
    flash::rowDot(dO, lddo, o, ldo, Di, a.B, H, Q, D);
    if (!accQ) {
      if (lddq == H * D) GK(cudaMemsetAsync(dq, 0, (size_t)a.B * Q * H * D * sizeof(real), STREAM));
      else flash::zeroColsK<<<blocks((size_t)a.B * Q * H * D), 256, 0, STREAM>>>(dq, lddq, (size_t)a.B * Q, H * D);
    }
    flash::runBackward(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, accK, accV);
    return;
  }
#endif
  int chunk = sdpaChunk(a, 2, 5);
  real *P = ARENA.get((size_t)chunk * H * Q * K), *dP = ARENA.get((size_t)chunk * H * Q * K);
  size_t hq = (size_t)chunk * H * Q * D, hk = (size_t)chunk * H * K * D;
  real *qh = ARENA.get(hq), *kh = ARENA.get(hk), *vh = ARENA.get(hk), *doh = ARENA.get(hq),
        *tmp = ARENA.get(std::max(hq, hk));
  long long sq = (long long)Q * D, sk = (long long)K * D, ss = (long long)Q * K;
  for (int b0 = 0; b0 < a.B; b0 += chunk) {
    int nb = std::min(chunk, a.B - b0);
    size_t nq = (size_t)nb * Q * H * D, nk = (size_t)nb * K * H * D;
    toHeadsK<<<blocks(nq), 256, 0, STREAM>>>(q + (size_t)b0 * Q * ldq, ldq, qh, nb, Q, H, D);
    toHeadsK<<<blocks(nk), 256, 0, STREAM>>>(k + (size_t)b0 * K * ldk, ldk, kh, nb, K, H, D);
    toHeadsK<<<blocks(nk), 256, 0, STREAM>>>(v + (size_t)b0 * K * ldv, ldv, vh, nb, K, H, D);
    toHeadsK<<<blocks(nq), 256, 0, STREAM>>>(dO + (size_t)b0 * Q * lddo, lddo, doh, nb, Q, H, D);
    bmm(false, true, Q, K, D, qh, sq, kh, sk, P, ss, nb * H, 0.f, a.scale);
    softmaxBiasK<<<blocks((size_t)nb * H * Q, 8), 256, 0, STREAM>>>(P, a, b0, nb);
    bmm(true, false, K, D, Q, P, ss, doh, sq, tmp, sk, nb * H);                         // dV = P^T dO
    fromHeadsK<<<blocks(nk), 256, 0, STREAM>>>(tmp, dv + (size_t)b0 * K * lddv, lddv, nb, K, H, D, 1.f, accV);
    bmm(false, true, Q, K, D, doh, sq, vh, sk, dP, ss, nb * H);                         // dP = dO V^T
    softmaxBwdK<<<blocks((size_t)nb * H * Q, 8), 256, 0, STREAM>>>(dP, P, a, b0, nb);
    bmm(false, false, Q, D, K, dP, ss, kh, sk, tmp, sq, nb * H);                        // dq = scale dS K
    fromHeadsK<<<blocks(nq), 256, 0, STREAM>>>(tmp, dq + (size_t)b0 * Q * lddq, lddq, nb, Q, H, D, a.scale, accQ);
    bmm(true, false, K, D, Q, dP, ss, qh, sq, tmp, sk, nb * H);                         // dk = scale dS^T q
    fromHeadsK<<<blocks(nk), 256, 0, STREAM>>>(tmp, dk + (size_t)b0 * K * lddk, lddk, nb, K, H, D, a.scale, accK);
  }
}
#ifndef GRAD_F64
// the same with q, k, v in bf16 (a module on bf16 storage): the flash kernels read them as they are; without flash
// they are widened to float for the bmm path
__global__ void narrowColsK(const float* x, bf* y, int ld, size_t rows, int C) {   // y[r, 0:C] = x[r, 0:C]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) { size_t i = (t / C) * ld + t % C; y[i] = __float2bfloat16(x[i]); }
}
__global__ void bfToFK(const bf* x, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = __bfloat162float(x[t]);
}
inline float* widen(const bf* x, size_t n) {
  float* y = ARENA.get(n);
  bfToFK<<<blocks(n), 256, 0, STREAM>>>(x, y, n);
  return y;
}
inline void sdpaForwardBf(const Sdpa& a, const bf* q, int ldq, const bf* k, int ldk, const bf* v, int ldv, float* o,
                          int ldo, float* lse) {
  if (FLASH && flash::supported(a.D)) { flash::runForwardBfIn(a, q, ldq, k, ldk, v, ldv, o, ldo, lse); return; }
  Scratch scratch;
  sdpaForward(a, widen(q, (size_t)a.B * a.Q * ldq), ldq, widen(k, (size_t)a.B * a.K * ldk), ldk,
              widen(v, (size_t)a.B * a.K * ldv), ldv, o, ldo, lse);
}
template <class DKV, class DOT>   // (dk, dv in bf16 for a module whose projections consume them so; dq is float, its
                                  // atomics'; dO float or bf16 - bf16 needs the row term DiIn formed by the caller)
inline void sdpaBackwardBf(const Sdpa& a, const bf* q, int ldq, const bf* k, int ldk, const bf* v, int ldv,
                           const DOT* dO, int lddo, float* dq, int lddq, DKV* dk, int lddk, DKV* dv, int lddv,
                           const float* o, int ldo, const float* lse, const float* DiIn = nullptr) {
  Scratch scratch;
  const int H = a.H, Q = a.Q, D = a.D;
  if (FLASH && flash::supported(D)) {
    const real* Di = DiIn;   // (the row term sum_d dO o, when the gate's backward has already formed it)
    if (!Di) {
      if constexpr (sizeof(DOT) == 2) { fprintf(stderr, "sdpaBackwardBf: a bf16 dO needs its row term\n"); exit(1); }
      else { real* t = ARENA.get((size_t)a.B * H * Q); flash::rowDot(dO, lddo, o, ldo, t, a.B, H, Q, D); Di = t; }
    }
    if (lddq == H * D) GK(cudaMemsetAsync(dq, 0, (size_t)a.B * Q * H * D * sizeof(real), STREAM));
    else flash::zeroColsK<<<blocks((size_t)a.B * Q * H * D), 256, 0, STREAM>>>(dq, lddq, (size_t)a.B * Q, H * D);
    flash::runBackwardBfIn(a, q, ldq, k, ldk, v, ldv, dO, lddo, lse, Di, dq, lddq, dk, lddk, dv, lddv, false, false);
    return;
  }
  const float* dOf;
  if constexpr (sizeof(DOT) == 2) dOf = widen(dO, (size_t)a.B * a.Q * lddo); else dOf = dO;
  if constexpr (sizeof(DKV) == 4) {
    sdpaBackward(a, widen(q, (size_t)a.B * a.Q * ldq), ldq, widen(k, (size_t)a.B * a.K * ldk), ldk,
                 widen(v, (size_t)a.B * a.K * ldv), ldv, dOf, lddo, dq, lddq, dk, lddk, dv, lddv);
  } else {   // the bmm path writes float: dk, dv through a float copy at the same leading dimensions
    float *fk = ARENA.get((size_t)a.B * a.K * lddk), *fv = ARENA.get((size_t)a.B * a.K * lddv);
    sdpaBackward(a, widen(q, (size_t)a.B * a.Q * ldq), ldq, widen(k, (size_t)a.B * a.K * ldk), ldk,
                 widen(v, (size_t)a.B * a.K * ldv), ldv, dOf, lddo, dq, lddq, fk, lddk, fv, lddv);
    narrowColsK<<<blocks((size_t)a.B * a.K * a.H * a.D), 256, 0, STREAM>>>(fk, dk, lddk, (size_t)a.B * a.K, a.H * a.D);
    narrowColsK<<<blocks((size_t)a.B * a.K * a.H * a.D), 256, 0, STREAM>>>(fv, dv, lddv, (size_t)a.B * a.K, a.H * a.D);
  }
}
// [A, B, C] -> [B, A, C] on bf16
__global__ void swap01BfK(const __nv_bfloat162* __restrict__ in, __nv_bfloat162* __restrict__ out, int A, int B,
                          int C2, bool accumulate) {   // (a channel pair a thread)
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * B * C2) return;
  int c = t % C2; size_t ab = t / C2; int b = ab % B, a = (int)(ab / B);
  const size_t o = ((size_t)b * A + a) * C2 + c;
  out[o] = accumulate ? __hadd2(out[o], in[t]) : in[t];
}
inline void swap01(const bf* in, bf* out, int A, int B, int C, bool accumulate = false) {   // (C even)
  swap01BfK<<<blocks((size_t)A * B * C / 2), 256, 0, STREAM>>>(reinterpret_cast<const __nv_bfloat162*>(in),
                                                              reinterpret_cast<__nv_bfloat162*>(out), A, B, C / 2, accumulate);
}
// og = o * gate (bf16 out), and its backward: dO = dog gate, dgate_pre = dog o gate (1 - gate)
__global__ void mulOgBfK(bf* og, const float* o, const bf* g, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) og[t] = __float2bfloat16(o[t] * __bfloat162float(g[t]));
}
// the same, a warp a (row, head), also forming the softmax backward's row term Di[b, h, q] = sum_d dO o (rows are
// b Q + q of [B, Q, H D]: what rowDot would read again)
__global__ void ogBwdDiBfK(const float* __restrict__ dog, const float* __restrict__ o, const bf* __restrict__ g,
                           float* __restrict__ dO, bf* __restrict__ dg, float* __restrict__ Di, size_t rows, int H,
                           int D, int Q) {
  const size_t w = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (w >= rows * H) return;
  const size_t row = w / H; const int h = (int)(w % H);
  float acc = 0.f;
  for (int d = lane; d < D; d += 32) {
    const size_t t = row * H * D + (size_t)h * D + d;
    const float s = __bfloat162float(g[t]), x = dog[t] * s;
    dO[t] = x;
    dg[t] = __float2bfloat16(dog[t] * o[t] * s * (1.f - s));
    acc += x * o[t];
  }
  for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(~0u, acc, off);
  if (!lane) { const size_t b = row / Q, q = row % Q; Di[(b * H + h) * Q + q] = acc; }
}
__global__ void ogBwdBfK(const float* dog, const float* o, const bf* g, float* dO, bf* dg, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const float s = __bfloat162float(g[t]);
  dO[t] = dog[t] * s;
  dg[t] = __float2bfloat16(dog[t] * o[t] * s * (1.f - s));
}
#endif

// ---------------------------------------------------------------- AF2's gated attention on top of it
// per batch row b: q = x Wq * D^-1/2, k = x Wk, v = x Wv (one LN'd input for both, as every AF2 self-attention),
// logits = q k^T + maskBias[b, k] + nb[h, q, k], softmax over k, o = P v, o *= sigmoid(x Wg + bg), out = o Wo + bo.
// The four projections are separate parameters, each written to its own slab of qkvg ([4][rows][HD]).
struct Attention {
  Param wq, wk, wv, wg, bg, wo, bo;
  int C = 0, H = 0, D = 0, O = 0;
  bool gating = true;
};
// The projections are interleaved, qkvg [B Q, P] with q, k, v, g at columns 0, HD, 2 HD, 3 HD (P = 3 or 4 HD), from
// one GEMM against the four weights packed side by side; the backward is then one GEMM for the four weight gradients
// and one for dx, where four of each at a quarter the width ran far below the device in float32 (the sdpa reads q, k,
// v at a leading dimension of P in place).
struct AttnSave {
  const real* x;    // [B Q, C] the (normalised) input
  real* qkvg;       // [B Q, P] the projections, gate already a sigmoid (q unscaled)
  real* o;          // [B Q, H D] the attention's output before the gate
  real* og;         // [B Q, H D] after it (Wo's input)
  int B, Q;
  real* lse;        // [2][B, H, Q] the logits' max and log exp-sum (the flash backward's)
  real* W;          // [C, P] the packed projection weights
  bool bf = false;  // BF16: x, qkvg, og and W held in bf16 (o stays float: the softmax backward's row term reads it)
};
inline Sdpa af2Sdpa(const Attention& a, int B, int Q, const real* mask, const real* nb, real* dnb = nullptr) {
  Sdpa d;
  d.B = B; d.H = a.H; d.Q = Q; d.K = Q; d.D = a.D; d.scale = 1.f / sqrt((real)a.D);
  d.keyMask = mask; d.clip = true;
  if (nb) d.bias[0] = Bias{nb, 0, (long long)Q * Q, Q, dnb};
  return d;
}
// W [C, roles HD] <- the roles' [C, HD] side by side (back: each role's gradient += its columns of W)
__global__ void packRolesK(real* W, const real* w0, const real* w1, const real* w2, const real* w3, int C, int HD,
                           int roles, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * roles * HD) return;
  int j = t % HD; size_t r = t / HD; int role = r % roles; int c = (int)(r / roles);
  const real* src[4] = {w0, w1, w2, w3};
  real* x = (real*)src[role] + (size_t)c * HD + j;
  if (back) *x += W[t]; else W[t] = *x;
}
// g = sigmoid(g + b) in place, the gate's columns of qkvg (row stride ld)
__global__ void gateFwdK(real* g, const real* b, size_t rows, int HD, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * HD) return;
  real& x = g[(t / HD) * ld + t % HD];
  x = (real)1 / ((real)1 + exp(-(x + b[t % HD])));
}
// og = o g; with dog: dO = dog g, dgp = dog o g (1 - g) - the gate at row stride ld
__global__ void gateMulK(real* og, const real* o, const real* g, size_t rows, int HD, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * HD) og[t] = o[t] * g[(t / HD) * ld + t % HD];
}
__global__ void gateBwdLdK(const real* dog, const real* o, const real* g, real* dO, real* dgp, size_t rows, int HD,
                           int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * HD) return;
  size_t i = (t / HD) * ld + t % HD;
  real s = g[i];
  dO[t] = dog[t] * s;
  dgp[i] = dog[t] * o[t] * s * (1.f - s);
}

#ifndef GRAD_F64
__global__ void gateFwdBfK(bf* g, const real* b, size_t rows, int HD, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * HD) return;
  bf& x = g[(t / HD) * ld + t % HD];
  x = __float2bfloat16(1.f / (1.f + __expf(-(__bfloat162float(x) + b[t % HD]))));
}
__global__ void gateMulBfK(bf* og, const real* o, const bf* g, size_t rows, int HD, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * HD) og[t] = __float2bfloat16(o[t] * __bfloat162float(g[(t / HD) * ld + t % HD]));
}
__global__ void gateBwdLdBfK(const real* dog, const real* o, const bf* g, real* dO, bf* dgp, size_t rows, int HD,
                             int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * HD) return;
  size_t i = (t / HD) * ld + t % HD;
  const float s = __bfloat162float(g[i]);
  dO[t] = dog[t] * s;
  dgp[i] = __float2bfloat16(dog[t] * o[t] * s * (1.f - s));
}
__device__ __forceinline__ float stR(float* p, float v) { *p = v; return v; }
__device__ __forceinline__ float stR(bf* p, float v) { *p = __float2bfloat16(v); return __bfloat162float(*p); }
template <class DOT>   // (dO written float, or bf16 for the attention to read in place; the row term is of what is written)
__global__ void gateBwdLdDiBfK(const real* __restrict__ dog, const real* __restrict__ o, const bf* __restrict__ g,
                               DOT* __restrict__ dO, bf* __restrict__ dgp, float* __restrict__ Di, size_t rows,
                               int H, int D, int ld, int Q) {
  const size_t w = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (w >= rows * H) return;
  const size_t row = w / H; const int h = (int)(w % H);
  float acc = 0.f;
  for (int d = lane; d < D; d += 32) {
    const size_t t = row * H * D + (size_t)h * D + d, i = row * ld + (size_t)h * D + d;
    const float s = __bfloat162float(g[i]);
    const float x = stR(dO + t, dog[t] * s);
    dgp[i] = __float2bfloat16(dog[t] * o[t] * s * (1.f - s));
    acc += x * o[t];
  }
  for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(~0u, acc, off);
  if (!lane) { const size_t b = row / Q, q = row % Q; Di[(b * H + h) * Q + q] = acc; }
}
__global__ void biasAddLdK(bf* y, const real* b, size_t rows, int C, int ld) {   // y[r, 0:C] (row stride ld) += b
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) { bf& x = y[(t / C) * ld + t % C]; x = __float2bfloat16(__bfloat162float(x) + b[t % C]); }
}
__global__ void gateSigLdBfK(bf* g, size_t rows, int C, int ld) {                 // g[r, 0:C] = sigmoid(g)
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) { bf& x = g[(t / C) * ld + t % C]; x = __float2bfloat16(1.f / (1.f + __expf(-__bfloat162float(x)))); }
}
__global__ void narrowRowsK(const real* x, bf* y, int C, int ld, size_t rows) {   // y[r, 0:C] (row stride ld) = x[r]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[(t / C) * ld + t % C] = __float2bfloat16(x[t]);
}
inline AttnSave attnForwardBf(const Attention& a, const real* x, const real* mask, const real* nb, real* out, int B,
                              int Q) {
  const int HD = a.H * a.D, roles = a.gating ? 4 : 3, P = roles * HD;
  const size_t rows = (size_t)B * Q;
  AttnSave s{};
  s.B = B; s.Q = Q; s.bf = true;
  bf* xb = toBf(x, rows * a.C);
  bf* W = bfGet((size_t)a.C * P);
  bf* qkvg = bfGet(rows * P);
  s.o = ARENA.get(rows * HD);
  bf* og = bfGet(rows * HD);
  s.lse = ARENA.get(2 * rows * a.H);
  {
    Scratch scratch;
    real* Wf = ARENA.get((size_t)a.C * P);
    packRolesK<<<blocks((size_t)a.C * P), 256, 0, STREAM>>>(Wf, a.wq.w, a.wk.w, a.wv.w, a.gating ? a.wg.w : nullptr,
                                                           a.C, HD, roles, false);
    toBf(Wf, W, (size_t)a.C * P);
  }
  mmBf(xb, W, qkvg, (int)rows, a.C, P);
  if (a.gating) gateFwdBfK<<<blocks(rows * HD), 256, 0, STREAM>>>(qkvg + 3 * HD, a.bg.w, rows, HD, P);
  sdpaForwardBf(af2Sdpa(a, B, Q, mask, nb), qkvg, P, qkvg + HD, P, qkvg + 2 * HD, P, s.o, HD, s.lse);
  if (a.gating) gateMulBfK<<<blocks(rows * HD), 256, 0, STREAM>>>(og, s.o, qkvg + 3 * HD, rows, HD, P);
  else toBf(s.o, og, rows * HD);
  {
    Scratch scratch;
    mmBf(og, toBf(a.wo.w, (size_t)HD * a.O), out, (int)rows, HD, a.O);
  }
  biasK<<<blocks(rows * a.O), 256, 0, STREAM>>>(out, a.bo.w, rows, a.O);
  s.x = reinterpret_cast<const real*>(xb); s.W = reinterpret_cast<real*>(W);
  s.qkvg = reinterpret_cast<real*>(qkvg); s.og = reinterpret_cast<real*>(og);
  return s;
}
template <class DOUT>   // (DOUT bf16: BF16 mode's pair-track gradient)
inline void attnBackwardBf(Attention& a, const AttnSave& s, const real* mask, const real* nb, const DOUT* dout,
                           real* dx, real* dnb, bool accumulate) {
  Scratch scratch;
  const int B = s.B, Q = s.Q, HD = a.H * a.D, roles = a.gating ? 4 : 3, P = roles * HD;
  const size_t rows = (size_t)B * Q;
  const bf *xb = reinterpret_cast<const bf*>(s.x), *W = reinterpret_cast<const bf*>(s.W),
           *qkvg = reinterpret_cast<const bf*>(s.qkvg), *og = reinterpret_cast<const bf*>(s.og);
  const bf* db;
  if constexpr (sizeof(DOUT) == 2) { db = dout; colSumBf(a.bo.g, db, rows, a.O, a.O); }
  else { colSum(a.bo.g, dout, rows, a.O); db = toBf(dout, rows * a.O); }
  mmATBf(og, db, a.wo.g, (int)rows, HD, a.O);
  real* dog = ARENA.get(rows * HD);
  mmBTBf(db, toBf(a.wo.w, (size_t)HD * a.O), dog, (int)rows, HD, a.O);
  // dq in float (its atomics'), dk, dv and the gate's gradient straight into the bf16 [rows, P] the GEMMs read
  bf* dqb = bfGet(rows * P);
  real* dq = ARENA.get(rows * HD);
  if (a.gating) {   // dO in bf16, its row term formed with it
    bf* dO = bfGet(rows * HD);
    real* Di = ARENA.get(rows * a.H);
    gateBwdLdDiBfK<<<blocks(rows * a.H * 32), 256, 0, STREAM>>>(dog, s.o, qkvg + 3 * HD, dO, dqb + 3 * HD, Di, rows,
                                                               a.H, a.D, P, Q);
    sdpaBackwardBf(af2Sdpa(a, B, Q, mask, nb, dnb), qkvg, P, qkvg + HD, P, qkvg + 2 * HD, P, dO, HD,
                   dq, HD, dqb + HD, P, dqb + 2 * HD, P, s.o, HD, s.lse, Di);
  } else {
    sdpaBackwardBf(af2Sdpa(a, B, Q, mask, nb, dnb), qkvg, P, qkvg + HD, P, qkvg + 2 * HD, P, (const real*)dog, HD,
                   dq, HD, dqb + HD, P, dqb + 2 * HD, P, s.o, HD, s.lse);
  }
  narrowRowsK<<<blocks(rows * HD), 256, 0, STREAM>>>(dq, dqb, HD, P, rows);
  if (a.gating) colSumBf(a.bg.g, dqb + 3 * HD, rows, HD, P);
  real* dW = ARENA.get((size_t)a.C * P);
  mmATBf(xb, dqb, dW, (int)rows, a.C, P, 0.f);
  packRolesK<<<blocks((size_t)a.C * P), 256, 0, STREAM>>>(dW, a.wq.g, a.wk.g, a.wv.g, a.gating ? a.wg.g : nullptr,
                                                         a.C, HD, roles, true);
  mmBTBf(dqb, W, dx, (int)rows, a.C, P, accumulate ? 1.f : 0.f);
}
#endif
// x [B, Q, C] -> out [B, Q, O]; mask [B, Q] (key mask, may be null); nb [H, Q, Q] (may be null)
inline AttnSave attnForward(const Attention& a, const real* x, const real* mask, const real* nb, real* out, int B,
                            int Q) {
#ifndef GRAD_F64
  if (BF16) return attnForwardBf(a, x, mask, nb, out, B, Q);
#endif
  const int HD = a.H * a.D, roles = a.gating ? 4 : 3, P = roles * HD;
  size_t rows = (size_t)B * Q;
  AttnSave s{x, ARENA.get(rows * P), ARENA.get(rows * HD), ARENA.get(rows * HD), B, Q, ARENA.get(2 * rows * a.H),
             ARENA.get((size_t)a.C * P)};
  packRolesK<<<blocks((size_t)a.C * P), 256, 0, STREAM>>>(s.W, a.wq.w, a.wk.w, a.wv.w, a.gating ? a.wg.w : nullptr,
                                                         a.C, HD, roles, false);
  mm(x, s.W, s.qkvg, (int)rows, a.C, P);
  if (a.gating) gateFwdK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.qkvg + 3 * HD, a.bg.w, rows, HD, P);
  sdpaForward(af2Sdpa(a, B, Q, mask, nb), s.qkvg, P, s.qkvg + HD, P, s.qkvg + 2 * HD, P, s.o, HD, s.lse);
  if (a.gating) gateMulK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.og, s.o, s.qkvg + 3 * HD, rows, HD, P);
  else GK(cudaMemcpyAsync(s.og, s.o, rows * HD * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  mm(s.og, a.wo.w, out, (int)rows, HD, a.O);
  biasK<<<blocks(rows * a.O), 256, 0, STREAM>>>(out, a.bo.w, rows, a.O);
  return s;
}

#ifndef GRAD_F64
inline void attnBackward(Attention& a, const AttnSave& s, const real* mask, const real* nb, const bf* dout, real* dx,
                         real* dnb, bool accumulate) {
  attnBackwardBf(a, s, mask, nb, dout, dx, dnb, accumulate);
}
#endif
// dout [B Q, O] -> dx [B Q, C] (written, or added with accumulate); dnb [H, Q, Q] accumulated when not null
inline void attnBackward(Attention& a, const AttnSave& s, const real* mask, const real* nb, const real* dout,
                         real* dx, real* dnb, bool accumulate) {
#ifndef GRAD_F64
  if (s.bf) { attnBackwardBf(a, s, mask, nb, dout, dx, dnb, accumulate); return; }
#endif
  Scratch scratch;
  const int B = s.B, Q = s.Q, HD = a.H * a.D, roles = a.gating ? 4 : 3, P = roles * HD;
  size_t rows = (size_t)B * Q;
  colSum(a.bo.g, dout, rows, a.O);
  mmAT(s.og, dout, a.wo.g, (int)rows, HD, a.O);
  real* dog = ARENA.get(rows * HD);
  mmBT(dout, a.wo.w, dog, (int)rows, HD, a.O);
  real* dqkvg = ARENA.get(rows * P);
  real* dO = ARENA.get(rows * HD);
  if (a.gating) gateBwdLdK<<<blocks(rows * HD), 256, 0, STREAM>>>(dog, s.o, s.qkvg + 3 * HD, dO, dqkvg + 3 * HD, rows, HD, P);
  else GK(cudaMemcpyAsync(dO, dog, rows * HD * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  sdpaBackward(af2Sdpa(a, B, Q, mask, nb, dnb), s.qkvg, P, s.qkvg + HD, P, s.qkvg + 2 * HD, P, dO, HD,
               dqkvg, P, dqkvg + HD, P, dqkvg + 2 * HD, P, false, false, false, s.o, HD, s.lse);
  real* dW = ARENA.get((size_t)a.C * P);
  mmAT(s.x, dqkvg, dW, (int)rows, a.C, P, 0.f);
  packRolesK<<<blocks((size_t)a.C * P), 256, 0, STREAM>>>(dW, a.wq.g, a.wk.g, a.wv.g, a.gating ? a.wg.g : nullptr,
                                                         a.C, HD, roles, true);
  if (a.gating) colSum(a.bg.g, dqkvg + 3 * HD, rows, HD, P);
  mmBT(dqkvg, s.W, dx, (int)rows, a.C, P, accumulate ? 1.f : 0.f);
}

// ---------------------------------------------------------------- language-model pieces
// rotary embedding, split halves within each head of D: channel d with d + D/2, angle pos * base^(-2d/D).
// sign -1 is the inverse rotation - the backward.
__global__ void ropeK(real* x, int rows, int heads, int D, int ld, const int* pos, double base, real sign) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int half = D / 2;
  if (t >= (size_t)rows * heads * half) return;
  int d = t % half, h = (t / half) % heads, row = (int)(t / ((size_t)half * heads));
  double ang = (double)(pos ? pos[row] : row) * pow(base, -2.0 * d / D);
  real s = (real)sin(ang) * sign, c = (real)cos(ang);
  real* p = x + (size_t)row * ld + h * D + d;
  real a = p[0], b = p[half];
  p[0] = a * c - b * s;
  p[half] = a * s + b * c;
}
inline void rope(real* x, int rows, int heads, int D, int ld, const int* pos = nullptr, double base = 10000, bool inverse = false) {
  ropeK<<<blocks((size_t)rows * heads * (D / 2)), 256, 0, STREAM>>>(x, rows, heads, D, ld, pos, base, inverse ? -1 : 1);
}
__device__ inline real siluR(real a) { return a / (1 + exp(-a)); }
// g[r, f] = silu(h[r, f]) h[r, F + f]   (the first half gated)
__global__ void swigluK(const real* h, real* g, size_t rows, int F) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * F) return;
  size_t r = t / F; int f = t % F;
  g[t] = siluR(h[r * 2 * F + f]) * h[r * 2 * F + F + f];
}
__global__ void swigluBwdK(const real* h, const real* dg, real* dh, size_t rows, int F) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * F) return;
  size_t r = t / F; int f = t % F;
  real a = h[r * 2 * F + f], b = h[r * 2 * F + F + f], sg = 1 / (1 + exp(-a));
  dh[r * 2 * F + f] = dg[t] * b * (sg + a * sg * (1 - sg));
  dh[r * 2 * F + F + f] = dg[t] * a * sg;
}
#ifndef GRAD_F64
// the same on bf16 storage, computed in float
__global__ void swigluBfK(const bf* __restrict__ h, bf* __restrict__ g, size_t rows, int F) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * F) return;
  size_t r = t / F; int f = t % F;
  const float a = __bfloat162float(h[r * 2 * F + f]), b = __bfloat162float(h[r * 2 * F + F + f]);
  g[t] = __float2bfloat16(a / (1.f + __expf(-a)) * b);
}
__global__ void swigluBwdBfK(const bf* __restrict__ h, const bf* __restrict__ dg, bf* __restrict__ dh, size_t rows,
                             int F) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * F) return;
  size_t r = t / F; int f = t % F;
  const float a = __bfloat162float(h[r * 2 * F + f]), b = __bfloat162float(h[r * 2 * F + F + f]);
  const float sg = 1.f / (1.f + __expf(-a)), d = __bfloat162float(dg[t]);
  dh[r * 2 * F + f] = __float2bfloat16(d * b * (sg + a * sg * (1.f - sg)));
  dh[r * 2 * F + F + f] = __float2bfloat16(d * a * sg);
}
#endif
#ifndef GRAD_F64
// ---- bf16-storage elementwise kernels (computed in float)
__device__ __forceinline__ float bfv(bf x) { return __bfloat162float(x); }
__device__ __forceinline__ float bfv(float x) { return x; }
__device__ __forceinline__ void bfs(bf* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void bfs(float* p, float v) { *p = v; }
template <class Y> __global__ void biasAddK(Y* y, const float* b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) bfs(y + t, bfv(y[t]) + b[t % C]);
}
__global__ void sigmoidBfK(bf* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) bfs(x + t, 1.f / (1.f + __expf(-bfv(x[t]))));
}
__global__ void rowMaskBfK(bf* x, const float* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) bfs(x + t, bfv(x[t]) * mask[t / C]);
}
__global__ void mulBfK(float* y, const bf* g, size_t n) {   // y *= g
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] *= bfv(g[t]);
}
// gateBwdK with the sigmoid saved in bf16 and the two gradients written in bf16
__global__ void gateBwdBfK(const float* dy, const float* a, const bf* sig, bf* da, bf* dpre, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const float s = bfv(sig[t]);
  bfs(da + t, dy[t] * s);
  bfs(dpre + t, dy[t] * a[t] * s * (1.f - s));
}
// g[c] += sum over rows of d[r, c], d bf16: two columns a thread, 8 row lanes, one atomic a column a block
__global__ void colSumBfK(float* g, const __nv_bfloat162* d, size_t rows, int C2, int ld2) {
  __shared__ float2 part[8][32];
  const int c = blockIdx.x * 32 + threadIdx.x;
  float2 s = make_float2(0.f, 0.f);
  if (c < C2)
    for (size_t r = blockIdx.y * blockDim.y + threadIdx.y; r < rows; r += (size_t)gridDim.y * blockDim.y) {
      const float2 v = __bfloat1622float2(d[r * ld2 + c]);
      s.x += v.x; s.y += v.y;
    }
  part[threadIdx.y][threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.y || c >= C2) return;
  for (int y = 1; y < 8; ++y) { s.x += part[y][threadIdx.x].x; s.y += part[y][threadIdx.x].y; }
  atomicAdd(&g[2 * c], s.x); atomicAdd(&g[2 * c + 1], s.y);
}
inline void colSumBf(float* g, const bf* d, size_t rows, int C, int ld) {   // (C, ld even; d 4-byte aligned)
  const int C2 = C / 2;
  const unsigned xb = (C2 + 31) / 32;
  // (1728 blocks: 432 left a thread ~150 dependent rows at 65,536 x 512 - 0.15 ms where the bytes are 0.05)
  dim3 b(32, 8), gr(xb, (unsigned)std::min<size_t>((rows + 7) / 8, std::max(1u, 1728 / xb)));
  colSumBfK<<<gr, b, 0, STREAM>>>(g, reinterpret_cast<const __nv_bfloat162*>(d), rows, C2, ld / 2);
}
// batched C_b = op(A_b) op(B_b) on bf16 operands (bmm's row-major convention), C float or bf16
template <class Y>
inline void bmmBf(bool ta, bool tb, int M, int N, int K, const bf* A, long long sa, const bf* B, long long sb, Y* C,
                  long long sc, int count, float alpha = 1.f) {
  const int lda = ta ? M : K, ldb = tb ? K : N;
  const float zero = 0.f;
  GB(cublasGemmStridedBatchedEx(BLAS, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                                B, CUDA_R_16BF, ldb, sb, A, CUDA_R_16BF, lda, sa, &zero, C, dtype<Y>(), N, sc, count,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
#endif
// the exact (erf) GELU
__global__ void geluK(real* y, const real* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = (real)0.5 * x[t] * (1 + erf(x[t] * (real)0.70710678118654752));
}
__global__ void geluBwdK(real* d, const real* x, size_t n) {       // d *= gelu'(x)
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real v = x[t];
  d[t] *= (real)0.5 * (1 + erf(v * (real)0.70710678118654752)) + v * exp(-(real)0.5 * v * v) * (real)0.3989422804014327;
}
__global__ void scaleK(real* x, real a, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] *= a;
}
// x[r] = table[ids[r]] * scale ; the backward scatters into the table's gradient
__global__ void embedK(const int* ids, const real* table, real* x, size_t rows, int C, real scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] = table[(size_t)ids[t / C] * C + t % C] * scale;
}
__global__ void embedBwdK(const int* ids, const real* dx, real* dtable, size_t rows, int C, real scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) atomicAdd(&dtable[(size_t)ids[t / C] * C + t % C], dx[t] * scale);
}
// mean cross entropy over the rows with a target >= 0 (a row's target < 0 is not scored); dlogits written, the
// loss accumulated in double
__global__ void crossEntropyK(const real* logits, const int* target, real* dlogits, int rows, int V, double* loss,
                              const int* count) {
  int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (r >= rows) return;
  const real* z = logits + (size_t)r * V;
  real* dz = dlogits + (size_t)r * V;
  int tg = target[r];
  if (tg < 0) { for (int v = lane; v < V; v += 32) dz[v] = 0; return; }
  real mx = -INFINITY;
  for (int v = lane; v < V; v += 32) mx = fmax(mx, z[v]);
  for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(~0u, mx, o));
  real s = 0;
  for (int v = lane; v < V; v += 32) s += exp(z[v] - mx);
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  real inv = 1 / (real)*count;
  for (int v = lane; v < V; v += 32) dz[v] = (exp(z[v] - mx) / s - (v == tg ? 1 : 0)) * inv;
  if (lane == 0) atomicAdd(loss, (double)((log(s) + mx - z[tg]) * inv));
}

// ---------------------------------------------------------------- AF2's ReLU transition: LN, Linear, relu, Linear
struct Transition {
  LayerNorm ln;
  Linear l1, l2;
};
struct TransSave { LnSave ln; real* xn; real* h; size_t rows; bool bf = false; };   // h: the hidden layer AFTER the relu
#ifndef GRAD_F64
__global__ void reluBfK(bf* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n && !(__bfloat162float(x[t]) > 0.f)) x[t] = __float2bfloat16(0.f);
}
__global__ void reluBwdBfK(bf* d, const bf* h, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n && !(__bfloat162float(h[t]) > 0.f)) d[t] = __float2bfloat16(0.f);
}
// the relu's backward with the first layer's bias gradient in the same pass: d[r, c] = 0 where h <= 0, and g[c] +=
// sum over rows of the masked d (colSumBfK's layout: two columns a thread, 8 row lanes)
__global__ void reluBwdSumBfK(__nv_bfloat162* d, const __nv_bfloat162* h, float* g, size_t rows, int C2) {
  __shared__ float2 part[8][32];
  const int c = blockIdx.x * 32 + threadIdx.x;
  float2 s = make_float2(0.f, 0.f);
  if (c < C2)
    for (size_t r = blockIdx.y * blockDim.y + threadIdx.y; r < rows; r += (size_t)gridDim.y * blockDim.y) {
      const size_t i = r * C2 + c;
      const float2 hv = __bfloat1622float2(h[i]);
      float2 v = __bfloat1622float2(d[i]);
      v.x = hv.x > 0.f ? v.x : 0.f; v.y = hv.y > 0.f ? v.y : 0.f;
      d[i] = __floats2bfloat162_rn(v.x, v.y);
      s.x += __bfloat162float(__float2bfloat16(v.x)); s.y += __bfloat162float(__float2bfloat16(v.y));
    }
  part[threadIdx.y][threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.y || c >= C2 || !g) return;
  for (int y = 1; y < 8; ++y) { s.x += part[y][threadIdx.x].x; s.y += part[y][threadIdx.x].y; }
  atomicAdd(&g[2 * c], s.x); atomicAdd(&g[2 * c + 1], s.y);
}
// BF16: the normalised input and the hidden layer in bf16, both GEMMs on bf16 operands
inline TransSave transForwardBf(const Transition& t, const real* x, real* out, size_t rows) {
  TransSave s; s.rows = rows; s.bf = true;
  bf* xn = bfGet(rows * t.ln.C);
  s.ln = lnForward(t.ln, x, xn, rows);
  bf* h = bfGet(rows * t.l1.N);
  linearFwdBf(t.l1, xn, h, rows, 0.f, true);   // (the relu in the GEMM's epilogue)
  linearFwdBf(t.l2, h, out, rows);
  s.xn = reinterpret_cast<real*>(xn); s.h = reinterpret_cast<real*>(h);
  return s;
}
template <class PD>   // (PD: the running gradient's type)
inline void transBackwardBf(Transition& t, const TransSave& s, const PD* dout, PD* dx, bool accumulate) {
  Scratch scratch;
  const size_t rows = s.rows, n = rows * t.l1.N;
  const bf *xn = reinterpret_cast<const bf*>(s.xn), *h = reinterpret_cast<const bf*>(s.h);
  const bf* db;
  if constexpr (sizeof(PD) == 2) db = dout; else db = toBf(dout, rows * t.l2.N);
  bf* dr = bfGet(n);
  linearBwdBf(t.l2, h, db, dr, rows);
  bf* dxn = bfGet(rows * t.ln.C);   // (only the LayerNorm's backward reads it)
  if (t.l1.N % 2 == 0) {   // the mask and l1's bias gradient in one pass
    const int C2 = t.l1.N / 2;
    const unsigned xb = (C2 + 31) / 32;
    dim3 b(32, 8), gr(xb, (unsigned)std::min<size_t>((rows + 7) / 8, std::max(1u, 1728 / xb)));
    reluBwdSumBfK<<<gr, b, 0, STREAM>>>(reinterpret_cast<__nv_bfloat162*>(dr), reinterpret_cast<const __nv_bfloat162*>(h),
                                        t.l1.hasBias ? t.l1.b.g : nullptr, rows, C2);
    Linear l1 = t.l1; l1.hasBias = false;   // (its bias gradient taken above)
    linearBwdBf(l1, xn, dr, dxn, rows);
  } else {
    reluBwdBfK<<<blocks(n), 256, 0, STREAM>>>(dr, h, n);
    linearBwdBf(t.l1, xn, dr, dxn, rows);
  }
  lnBackward(t.ln, s.ln, dxn, dx, rows, accumulate);
}
#endif
// (relu(h) > 0 exactly where h > 0, so the backward's mask reads the saved output and nothing recomputes it)
inline TransSave transForward(const Transition& t, const real* x, real* out, size_t rows) {
#ifndef GRAD_F64
  if (BF16) return transForwardBf(t, x, out, rows);
#endif
  TransSave s;
  s.rows = rows;
  s.xn = ARENA.get(rows * t.ln.C);
  s.ln = lnForward(t.ln, x, s.xn, rows);
  s.h = ARENA.get(rows * t.l1.N);
  t.l1.forward(s.xn, s.h, rows);
  reluK<<<blocks(rows * t.l1.N), 256, 0, STREAM>>>(s.h, s.h, rows * t.l1.N);
  t.l2.forward(s.h, out, rows);
  return s;
}
#ifndef GRAD_F64
inline void transBackward(Transition& t, const TransSave& s, const bf* dout, bf* dx, bool accumulate) {
  transBackwardBf(t, s, dout, dx, accumulate);
}
#endif
inline void transBackward(Transition& t, const TransSave& s, const real* dout, real* dx, bool accumulate) {
#ifndef GRAD_F64
  if (s.bf) { transBackwardBf(t, s, dout, dx, accumulate); return; }
#endif
  Scratch scratch;
  size_t rows = s.rows, n = rows * t.l1.N;
  real* dr = ARENA.get(n);
  t.l2.backward(s.h, dout, dr, rows);
  reluBwdK<<<blocks(n), 256, 0, STREAM>>>(dr, s.h, n);
  real* dxn = ARENA.get(rows * t.ln.C);
  t.l1.backward(s.xn, dr, dxn, rows);
  lnBackward(t.ln, s.ln, dxn, dx, rows, accumulate);
}

}  // namespace grad
