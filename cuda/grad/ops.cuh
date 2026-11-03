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

// ---------------------------------------------------------------- LayerNorm (eps 1e-5, haiku's)
struct LayerNorm {
  Param g, b;
  int C = 0;
  real eps = 1e-5f;
};

// one warp a row; mean and rstd saved for the backward
__global__ void lnFwdK(const real* __restrict__ x, const real* __restrict__ g, const real* __restrict__ b,
                       real* __restrict__ y, real* __restrict__ mean, real* __restrict__ rstd, size_t rows, int C,
                       real eps) {
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
  for (int c = lane; c < C; c += 32) y[r * C + c] = (xr[c] - mu) * rs * g[c] + b[c];
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
      real xh = (xr[c] - mu) * rs, dh = dr[c] * g[c];
      s1 += dh; s2 += dh * xh;
      atomicAdd(&sh[c], dr[c] * xh);
      atomicAdd(&sh[C + c], dr[c]);
    }
    for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(~0u, s1, o); s2 += __shfl_xor_sync(~0u, s2, o); }
    s1 /= C; s2 /= C;
    for (int c = lane; c < C; c += 32) {
      real xh = (xr[c] - mu) * rs, v = rs * (dr[c] * g[c] - s1 - xh * s2);
      dx[r * C + c] = accumulate ? dx[r * C + c] + v : v;
    }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) { atomicAdd(&dg[c], sh[c]); atomicAdd(&db[c], sh[C + c]); }
}

struct LnSave { const real* x; real* mean; real* rstd; };
inline LnSave lnForward(const LayerNorm& ln, const real* x, real* y, size_t rows) {
  LnSave s{x, ARENA.get(rows), ARENA.get(rows)};
  lnFwdK<<<blocks(rows, 8), 256, 0, STREAM>>>(x, ln.g.w, ln.b.w, y, s.mean, s.rstd, rows, ln.C, ln.eps);
  return s;
}
inline void lnBackward(LayerNorm& ln, const LnSave& s, const real* dy, real* dx, size_t rows, bool accumulate) {
  unsigned gr = (unsigned)std::min<size_t>(blocks(rows, 8), 1024);
  lnBwdK<<<gr, 256, 2 * ln.C * sizeof(real), STREAM>>>(s.x, dy, ln.g.w, s.mean, s.rstd, dx, ln.g.g, ln.b.g, rows, ln.C, accumulate);
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
inline void swap01(const real* in, real* out, int A, int B, int C, bool accumulate = false) {
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
  for (int k = lane; k < K; k += 32) {
    real v = s[k];
    if (m) v += 1e9f * (m[k] - 1.f);
    if (n0) v += n0[k];
    if (n1) v += n1[k];
    if (a.clip) v = fmin(fmax(v, (real)-1e8), (real)1e8);
    s[k] = v; mx = fmax(mx, v);
  }
  for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(~0u, mx, o));
  real sum = 0;
  for (int k = lane; k < K; k += 32) { real e = s[k] == -INFINITY ? 0.f : exp(s[k] - mx); s[k] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  real inv = sum > 0 ? 1.f / sum : 0.f;
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

inline void sdpaForward(const Sdpa& a, const real* q, int ldq, const real* k, int ldk, const real* v, int ldv,
                        real* o, int ldo) {
  Scratch scratch;
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
                         bool accQ = false, bool accK = false, bool accV = false) {
  Scratch scratch;
  const int H = a.H, Q = a.Q, K = a.K, D = a.D;
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

// ---------------------------------------------------------------- AF2's gated attention on top of it
// per batch row b: q = x Wq * D^-1/2, k = x Wk, v = x Wv (one LN'd input for both, as every AF2 self-attention),
// logits = q k^T + maskBias[b, k] + nb[h, q, k], softmax over k, o = P v, o *= sigmoid(x Wg + bg), out = o Wo + bo.
// The four projections are separate parameters, each written to its own slab of qkvg ([4][rows][HD]).
struct Attention {
  Param wq, wk, wv, wg, bg, wo, bo;
  int C = 0, H = 0, D = 0, O = 0;
  bool gating = true;
};
struct AttnSave {
  const real* x;    // [B Q, C] the (normalised) input
  real* qkvg;       // [4][B Q, H D] the projections, gate already a sigmoid (q unscaled)
  real* o;          // [B Q, H D] the attention's output before the gate
  real* og;         // [B Q, H D] after it (Wo's input)
  int B, Q;
};
inline Sdpa af2Sdpa(const Attention& a, int B, int Q, const real* mask, const real* nb, real* dnb = nullptr) {
  Sdpa d;
  d.B = B; d.H = a.H; d.Q = Q; d.K = Q; d.D = a.D; d.scale = 1.f / sqrt((real)a.D);
  d.keyMask = mask; d.clip = true;
  if (nb) d.bias[0] = Bias{nb, 0, (long long)Q * Q, Q, dnb};
  return d;
}

// x [B, Q, C] -> out [B, Q, O]; mask [B, Q] (key mask, may be null); nb [H, Q, Q] (may be null)
inline AttnSave attnForward(const Attention& a, const real* x, const real* mask, const real* nb, real* out, int B,
                            int Q) {
  const int HD = a.H * a.D, P = (a.gating ? 4 : 3) * HD;
  size_t rows = (size_t)B * Q;
  AttnSave s{x, ARENA.get(rows * P), ARENA.get(rows * HD), ARENA.get(rows * HD), B, Q};
  mm(x, a.wq.w, s.qkvg, (int)rows, a.C, HD);
  mm(x, a.wk.w, s.qkvg + rows * HD, (int)rows, a.C, HD);
  mm(x, a.wv.w, s.qkvg + 2 * rows * HD, (int)rows, a.C, HD);
  if (a.gating) {
    real* g = s.qkvg + 3 * rows * HD;
    mm(x, a.wg.w, g, (int)rows, a.C, HD);
    biasK<<<blocks(rows * HD), 256, 0, STREAM>>>(g, a.bg.w, rows, HD);
    sigmoidK<<<blocks(rows * HD), 256, 0, STREAM>>>(g, rows * HD);
  }
  sdpaForward(af2Sdpa(a, B, Q, mask, nb), s.qkvg, HD, s.qkvg + rows * HD, HD, s.qkvg + 2 * rows * HD, HD, s.o, HD);
  if (a.gating) mulK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.og, s.o, s.qkvg + 3 * rows * HD, rows * HD);
  else GK(cudaMemcpyAsync(s.og, s.o, rows * HD * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  mm(s.og, a.wo.w, out, (int)rows, HD, a.O);
  biasK<<<blocks(rows * a.O), 256, 0, STREAM>>>(out, a.bo.w, rows, a.O);
  return s;
}

// dout [B Q, O] -> dx [B Q, C] (written, or added with accumulate); dnb [H, Q, Q] accumulated when not null
inline void attnBackward(Attention& a, const AttnSave& s, const real* mask, const real* nb, const real* dout,
                         real* dx, real* dnb, bool accumulate) {
  Scratch scratch;
  const int B = s.B, Q = s.Q, HD = a.H * a.D;
  size_t rows = (size_t)B * Q;
  colSum(a.bo.g, dout, rows, a.O);
  mmAT(s.og, dout, a.wo.g, (int)rows, HD, a.O);
  real* dog = ARENA.get(rows * HD);
  mmBT(dout, a.wo.w, dog, (int)rows, HD, a.O);
  real* dqkvg = ARENA.get(rows * 4 * HD);
  real* dO = ARENA.get(rows * HD);
  if (a.gating) {
    const real* g = s.qkvg + 3 * rows * HD;
    gateBwdK<<<blocks(rows * HD), 256, 0, STREAM>>>(dog, s.o, g, dO, dqkvg + 3 * rows * HD, rows * HD);
  } else {
    GK(cudaMemcpyAsync(dO, dog, rows * HD * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  sdpaBackward(af2Sdpa(a, B, Q, mask, nb, dnb), s.qkvg, HD, s.qkvg + rows * HD, HD, s.qkvg + 2 * rows * HD, HD, dO, HD,
               dqkvg, HD, dqkvg + rows * HD, HD, dqkvg + 2 * rows * HD, HD);
  mmAT(s.x, dqkvg, a.wq.g, (int)rows, a.C, HD);
  mmAT(s.x, dqkvg + rows * HD, a.wk.g, (int)rows, a.C, HD);
  mmAT(s.x, dqkvg + 2 * rows * HD, a.wv.g, (int)rows, a.C, HD);
  mmBT(dqkvg, a.wq.w, dx, (int)rows, a.C, HD, accumulate ? 1.f : 0.f);
  mmBT(dqkvg + rows * HD, a.wk.w, dx, (int)rows, a.C, HD, 1.f);
  mmBT(dqkvg + 2 * rows * HD, a.wv.w, dx, (int)rows, a.C, HD, 1.f);
  if (a.gating) {
    const real* dg = dqkvg + 3 * rows * HD;
    mmAT(s.x, dg, a.wg.g, (int)rows, a.C, HD);
    colSum(a.bg.g, dg, rows, HD);
    mmBT(dg, a.wg.w, dx, (int)rows, a.C, HD, 1.f);
  }
}

// ---------------------------------------------------------------- AF2's ReLU transition: LN, Linear, relu, Linear
struct Transition {
  LayerNorm ln;
  Linear l1, l2;
};
struct TransSave { LnSave ln; real* xn; real* h; size_t rows; };
inline TransSave transForward(const Transition& t, const real* x, real* out, size_t rows) {
  TransSave s;
  s.rows = rows;
  s.xn = ARENA.get(rows * t.ln.C);
  s.ln = lnForward(t.ln, x, s.xn, rows);
  s.h = ARENA.get(rows * t.l1.N);
  t.l1.forward(s.xn, s.h, rows);
  Scratch scratch;
  real* r = ARENA.get(rows * t.l1.N);
  reluK<<<blocks(rows * t.l1.N), 256, 0, STREAM>>>(r, s.h, rows * t.l1.N);
  t.l2.forward(r, out, rows);
  return s;
}
inline void transBackward(Transition& t, const TransSave& s, const real* dout, real* dx, bool accumulate) {
  Scratch scratch;
  size_t rows = s.rows, n = rows * t.l1.N;
  real* r = ARENA.get(n);
  reluK<<<blocks(n), 256, 0, STREAM>>>(r, s.h, n);
  real* dr = ARENA.get(n);
  t.l2.backward(r, dout, dr, rows);
  reluBwdK<<<blocks(n), 256, 0, STREAM>>>(dr, s.h, n);
  real* dxn = ARENA.get(rows * t.ln.C);
  t.l1.backward(s.xn, dr, dxn, rows);
  lnBackward(t.ln, s.ln, dxn, dx, rows, accumulate);
}

}  // namespace grad
