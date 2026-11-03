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
  float eps = 1e-5f;
};

// one warp a row; mean and rstd saved for the backward
__global__ void lnFwdK(const float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b,
                       float* __restrict__ y, float* __restrict__ mean, float* __restrict__ rstd, size_t rows, int C,
                       float eps) {
  size_t r = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (r >= rows) return;
  const float* xr = x + r * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mu = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mu; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float rs = rsqrtf(v / C + eps);
  for (int c = lane; c < C; c += 32) y[r * C + c] = (xr[c] - mu) * rs * g[c] + b[c];
  if (lane == 0) { mean[r] = mu; rstd[r] = rs; }
}
// dx = rstd (dxhat - mean(dxhat) - xhat mean(dxhat xhat)), dxhat = dy g; dg += dy xhat, db += dy (atomics a block)
__global__ void lnBwdK(const float* __restrict__ x, const float* __restrict__ dy, const float* __restrict__ g,
                       const float* __restrict__ mean, const float* __restrict__ rstd, float* __restrict__ dx,
                       float* __restrict__ dg, float* __restrict__ db, size_t rows, int C, bool accumulate) {
  extern __shared__ float sh[];           // [2 C] block partials of dg, db
  for (int c = threadIdx.x; c < 2 * C; c += blockDim.x) sh[c] = 0;
  __syncthreads();
  int warps = blockDim.x / 32, lane = threadIdx.x & 31;
  for (size_t r = (size_t)blockIdx.x * warps + threadIdx.x / 32; r < rows; r += (size_t)gridDim.x * warps) {
    const float *xr = x + r * C, *dr = dy + r * C;
    float mu = mean[r], rs = rstd[r], s1 = 0, s2 = 0;
    for (int c = lane; c < C; c += 32) {
      float xh = (xr[c] - mu) * rs, dh = dr[c] * g[c];
      s1 += dh; s2 += dh * xh;
      atomicAdd(&sh[c], dr[c] * xh);
      atomicAdd(&sh[C + c], dr[c]);
    }
    for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(~0u, s1, o); s2 += __shfl_xor_sync(~0u, s2, o); }
    s1 /= C; s2 /= C;
    for (int c = lane; c < C; c += 32) {
      float xh = (xr[c] - mu) * rs, v = rs * (dr[c] * g[c] - s1 - xh * s2);
      dx[r * C + c] = accumulate ? dx[r * C + c] + v : v;
    }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) { atomicAdd(&dg[c], sh[c]); atomicAdd(&db[c], sh[C + c]); }
}

struct LnSave { const float* x; float* mean; float* rstd; };
inline LnSave lnForward(const LayerNorm& ln, const float* x, float* y, size_t rows) {
  LnSave s{x, ARENA.get(rows), ARENA.get(rows)};
  lnFwdK<<<blocks(rows, 8), 256, 0, STREAM>>>(x, ln.g.w, ln.b.w, y, s.mean, s.rstd, rows, ln.C, ln.eps);
  return s;
}
inline void lnBackward(LayerNorm& ln, const LnSave& s, const float* dy, float* dx, size_t rows, bool accumulate) {
  unsigned gr = (unsigned)std::min<size_t>(blocks(rows, 8), 1024);
  lnBwdK<<<gr, 256, 2 * ln.C * 4, STREAM>>>(s.x, dy, ln.g.w, s.mean, s.rstd, dx, ln.g.g, ln.b.g, rows, ln.C, accumulate);
}

// ---------------------------------------------------------------- elementwise pieces
__global__ void sigmoidK(float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = 1.f / (1.f + expf(-x[t]));
}
__global__ void mulK(float* y, const float* a, const float* b, size_t n) {   // y = a b
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = a[t] * b[t];
}
// given y = a * sig (sig already the sigmoid): da = dy sig, dpre = dy a sig (1 - sig)
__global__ void gateBwdK(const float* dy, const float* a, const float* sig, float* da, float* dpre, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  float s = sig[t];
  if (da) da[t] = dy[t] * s;
  dpre[t] = dy[t] * a[t] * s * (1.f - s);
}
__global__ void rowMaskK(float* x, const float* mask, size_t rows, int C) {   // x[r, :] *= mask[r]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] *= mask[t / C];
}
__global__ void reluK(float* y, const float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmaxf(x[t], 0.f);
}
__global__ void reluBwdK(float* d, const float* pre, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n && pre[t] <= 0.f) d[t] = 0.f;
}
// [A, B, C] -> [B, A, C]  (accumulate: out += )
__global__ void swap01K(const float* __restrict__ in, float* __restrict__ out, int A, int B, int C, bool accumulate) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)A * B * C;
  if (t >= n) return;
  int c = t % C; size_t ab = t / C; int b = ab % B, a = (int)(ab / B);
  size_t o = ((size_t)b * A + a) * C + c;
  out[o] = accumulate ? out[o] + in[t] : in[t];
}
inline void swap01(const float* in, float* out, int A, int B, int C, bool accumulate = false) {
  swap01K<<<blocks((size_t)A * B * C), 256, 0, STREAM>>>(in, out, A, B, C, accumulate);
}
// [R, C] <-> [C, R]
inline void transpose(const float* in, float* out, int R, int C, bool accumulate = false) { swap01(in, out, R, C, 1, accumulate); }

// [B, Q, H, D] -> [B, H, Q, D] and back (accumulate on the way back)
__global__ void toHeadsK(const float* __restrict__ in, int ldIn, float* __restrict__ out, int B, int Q, int H, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * Q * H * D) return;
  int d = t % D; size_t r = t / D; int h = r % H; r /= H; int q = r % Q; int b = (int)(r / Q);
  out[(((size_t)b * H + h) * Q + q) * D + d] = in[((size_t)b * Q + q) * ldIn + h * D + d];
}
__global__ void fromHeadsK(const float* __restrict__ in, float* __restrict__ out, int ldOut, int B, int Q, int H, int D,
                           float scale, bool accumulate) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * Q * H * D) return;
  int d = t % D; size_t r = t / D; int h = r % H; r /= H; int q = r % Q; int b = (int)(r / Q);
  float v = in[(((size_t)b * H + h) * Q + q) * D + d] * scale;
  float& o = out[((size_t)b * Q + q) * ldOut + h * D + d];
  o = accumulate ? o + v : v;
}

// ---------------------------------------------------------------- gated attention with a shared bias
// AF2's Attention: per batch row b, q = x Wq * D^-1/2, k = x Wk, v = x Wv (one LN'd input for both, as every AF2
// self-attention), logits = q k^T + maskBias[b, k] + nb[h, q, k], softmax over k, o = P v, o *= sigmoid(x Wg + bg),
// out = o Wo + bo. Wq|Wk|Wv|Wg are packed into one [C, 4 H D] projection for one GEMM; gradients land in their own
// parameters.
struct Attention {
  Param wq, wk, wv, wg, bg, wo, bo;
  int C = 0, H = 0, D = 0, O = 0;
  bool gating = true;
};

// logits[b, h, q, k] += 1e9 (mask[b, k] - 1) + nb[h, q, k]; then softmax over k in place
__global__ void softmaxBiasK(float* __restrict__ S, const float* __restrict__ mask, const float* __restrict__ nb,
                             int B, int H, int Q, int K) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= (size_t)B * H * Q) return;
  int q = row % Q, h = (row / Q) % H, b = (int)(row / ((size_t)Q * H));
  float* s = S + row * K;
  const float* m = mask ? mask + (size_t)b * K : nullptr;
  const float* n = nb ? nb + ((size_t)h * Q + q) * K : nullptr;
  float mx = -INFINITY;
  for (int k = lane; k < K; k += 32) {
    float v = s[k];
    if (m) v += 1e9f * (m[k] - 1.f);
    if (n) v += n[k];
    v = fminf(fmaxf(v, -1e8f), 1e8f);
    s[k] = v; mx = fmaxf(mx, v);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  float sum = 0;
  for (int k = lane; k < K; k += 32) { float e = expf(s[k] - mx); s[k] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  float inv = 1.f / sum;
  for (int k = lane; k < K; k += 32) s[k] *= inv;
}
// dS = P (dP - sum_k dP P), in place on dP; also dnb[h, q, k] += dS (the shared bias's gradient, over b)
__global__ void softmaxBwdK(float* __restrict__ dP, const float* __restrict__ P, float* __restrict__ dnb, int B, int H,
                            int Q, int K) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= (size_t)B * H * Q) return;
  int q = row % Q, h = (row / Q) % H;
  float* d = dP + row * K;
  const float* p = P + row * K;
  float s = 0;
  for (int k = lane; k < K; k += 32) s += d[k] * p[k];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float* n = dnb ? dnb + ((size_t)h * Q + q) * K : nullptr;
  for (int k = lane; k < K; k += 32) {
    float v = p[k] * (d[k] - s);
    d[k] = v;
    if (n) atomicAdd(&n[k], v);
  }
}

struct AttnSave {
  const float* x;    // [B Q, C] the (normalised) input
  float* qkvg;       // [B Q, 4 H D] the projections, gate already a sigmoid (q unscaled)
  float* o;          // [B Q, H D] the attention's output before the gate
  float* og;         // [B Q, H D] after it (Wo's input)
  int B, Q;
};

// x [B, Q, C] -> out [B, Q, O]; mask [B, Q] (key mask, may be null); nb [H, Q, Q] (may be null)
inline AttnSave attnForward(const Attention& a, const float* x, const float* mask, const float* nb, float* out, int B,
                            int Q) {
  const int H = a.H, D = a.D, HD = H * D, P = (a.gating ? 4 : 3) * HD;
  size_t rows = (size_t)B * Q;
  AttnSave s{x, ARENA.get(rows * P), ARENA.get(rows * HD), ARENA.get(rows * HD), B, Q};
  mm(x, a.wq.w, s.qkvg, (int)rows, a.C, HD);
  // the four projections are separate parameters; four GEMMs writing strided columns would need ld != N, so write
  // each to its own slab of qkvg (slab-major [4][rows][HD])
  mm(x, a.wk.w, s.qkvg + rows * HD, (int)rows, a.C, HD);
  mm(x, a.wv.w, s.qkvg + 2 * rows * HD, (int)rows, a.C, HD);
  if (a.gating) {
    float* g = s.qkvg + 3 * rows * HD;
    mm(x, a.wg.w, g, (int)rows, a.C, HD);
    biasK<<<blocks(rows * HD), 256, 0, STREAM>>>(g, a.bg.w, rows, HD);
    sigmoidK<<<blocks(rows * HD), 256, 0, STREAM>>>(g, rows * HD);
  }
  const float scale = 1.f / sqrtf((float)D);
  size_t perB = (size_t)H * Q * Q * 4 + (size_t)3 * H * Q * D * 4;
  int chunk = (int)std::max<size_t>(1, std::min<size_t>(B, ATTN_CHUNK_BYTES / perB));
  {
  Scratch scratch;
  float* S = ARENA.get((size_t)chunk * H * Q * Q);
  float *qh = ARENA.get((size_t)chunk * H * Q * D), *kh = ARENA.get((size_t)chunk * H * Q * D),
        *vh = ARENA.get((size_t)chunk * H * Q * D);
  for (int b0 = 0; b0 < B; b0 += chunk) {
    int nb_ = std::min(chunk, B - b0);
    size_t r0 = (size_t)b0 * Q, n = (size_t)nb_ * Q * HD;
    toHeadsK<<<blocks(n), 256, 0, STREAM>>>(s.qkvg + r0 * HD, HD, qh, nb_, Q, H, D);
    toHeadsK<<<blocks(n), 256, 0, STREAM>>>(s.qkvg + rows * HD + r0 * HD, HD, kh, nb_, Q, H, D);
    toHeadsK<<<blocks(n), 256, 0, STREAM>>>(s.qkvg + 2 * rows * HD + r0 * HD, HD, vh, nb_, Q, H, D);
    bmm(false, true, Q, Q, D, qh, (long long)Q * D, kh, (long long)Q * D, S, (long long)Q * Q, nb_ * H, 0.f, scale);
    softmaxBiasK<<<blocks((size_t)nb_ * H * Q, 8), 256, 0, STREAM>>>(S, mask ? mask + (size_t)b0 * Q : nullptr, nb,
                                                                     nb_, H, Q, Q);
    bmm(false, false, Q, D, Q, S, (long long)Q * Q, vh, (long long)Q * D, qh, (long long)Q * D, nb_ * H);
    fromHeadsK<<<blocks(n), 256, 0, STREAM>>>(qh, s.o + r0 * HD, HD, nb_, Q, H, D, 1.f, false);
  }
  }
  if (a.gating) mulK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.og, s.o, s.qkvg + 3 * rows * HD, rows * HD);
  else GK(cudaMemcpyAsync(s.og, s.o, rows * HD * 4, cudaMemcpyDeviceToDevice, STREAM));
  mm(s.og, a.wo.w, out, (int)rows, HD, a.O);
  biasK<<<blocks(rows * a.O), 256, 0, STREAM>>>(out, a.bo.w, rows, a.O);
  return s;
}

// dout [B Q, O] -> dx [B Q, C] (written, or added with accumulate); dnb [H, Q, Q] accumulated when not null
inline void attnBackward(Attention& a, const AttnSave& s, const float* mask, const float* nb, const float* dout,
                         float* dx, float* dnb, bool accumulate) {
  Scratch scratch;
  const int B = s.B, Q = s.Q, H = a.H, D = a.D, HD = H * D;
  size_t rows = (size_t)B * Q;
  const float scale = 1.f / sqrtf((float)D);
  colSum(a.bo.g, dout, rows, a.O);
  mmAT(s.og, dout, a.wo.g, (int)rows, HD, a.O);
  float* dog = ARENA.get(rows * HD);
  mmBT(dout, a.wo.w, dog, (int)rows, HD, a.O);
  float* dqkvg = ARENA.get(rows * 4 * HD);
  float* dO = ARENA.get(rows * HD);
  if (a.gating) {
    const float* g = s.qkvg + 3 * rows * HD;
    gateBwdK<<<blocks(rows * HD), 256, 0, STREAM>>>(dog, s.o, g, dO, dqkvg + 3 * rows * HD, rows * HD);
  } else {
    GK(cudaMemcpyAsync(dO, dog, rows * HD * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  size_t perB = (size_t)2 * H * Q * Q * 4 + (size_t)5 * H * Q * D * 4;
  int chunk = (int)std::max<size_t>(1, std::min<size_t>(B, ATTN_CHUNK_BYTES / perB));
  float *P = ARENA.get((size_t)chunk * H * Q * Q), *dP = ARENA.get((size_t)chunk * H * Q * Q);
  size_t hs = (size_t)chunk * H * Q * D;
  float *qh = ARENA.get(hs), *kh = ARENA.get(hs), *vh = ARENA.get(hs), *doh = ARENA.get(hs), *tmp = ARENA.get(hs);
  for (int b0 = 0; b0 < B; b0 += chunk) {
    int nb_ = std::min(chunk, B - b0);
    size_t r0 = (size_t)b0 * Q, n = (size_t)nb_ * Q * HD;
    long long sq = (long long)Q * D, ss = (long long)Q * Q;
    toHeadsK<<<blocks(n), 256, 0, STREAM>>>(s.qkvg + r0 * HD, HD, qh, nb_, Q, H, D);
    toHeadsK<<<blocks(n), 256, 0, STREAM>>>(s.qkvg + rows * HD + r0 * HD, HD, kh, nb_, Q, H, D);
    toHeadsK<<<blocks(n), 256, 0, STREAM>>>(s.qkvg + 2 * rows * HD + r0 * HD, HD, vh, nb_, Q, H, D);
    toHeadsK<<<blocks(n), 256, 0, STREAM>>>(dO + r0 * HD, HD, doh, nb_, Q, H, D);
    // recompute P
    bmm(false, true, Q, Q, D, qh, sq, kh, sq, P, ss, nb_ * H, 0.f, scale);
    softmaxBiasK<<<blocks((size_t)nb_ * H * Q, 8), 256, 0, STREAM>>>(P, mask ? mask + (size_t)b0 * Q : nullptr, nb, nb_,
                                                                     H, Q, Q);
    // dV = P^T dO ; dP = dO V^T
    bmm(true, false, Q, D, Q, P, ss, doh, sq, tmp, sq, nb_ * H);
    fromHeadsK<<<blocks(n), 256, 0, STREAM>>>(tmp, dqkvg + 2 * rows * HD + r0 * HD, HD, nb_, Q, H, D, 1.f, false);
    bmm(false, true, Q, Q, D, doh, sq, vh, sq, dP, ss, nb_ * H);
    softmaxBwdK<<<blocks((size_t)nb_ * H * Q, 8), 256, 0, STREAM>>>(dP, P, dnb, nb_, H, Q, Q);
    // dq = dS K * scale ; dk = dS^T Q * scale  (q, k unscaled in the save)
    bmm(false, false, Q, D, Q, dP, ss, kh, sq, tmp, sq, nb_ * H);
    fromHeadsK<<<blocks(n), 256, 0, STREAM>>>(tmp, dqkvg + r0 * HD, HD, nb_, Q, H, D, scale, false);
    bmm(true, false, Q, D, Q, dP, ss, qh, sq, tmp, sq, nb_ * H);
    fromHeadsK<<<blocks(n), 256, 0, STREAM>>>(tmp, dqkvg + rows * HD + r0 * HD, HD, nb_, Q, H, D, scale, false);
  }
  // the four projections' parameter gradients and the input's
  mmAT(s.x, dqkvg, a.wq.g, (int)rows, a.C, HD);
  mmAT(s.x, dqkvg + rows * HD, a.wk.g, (int)rows, a.C, HD);
  mmAT(s.x, dqkvg + 2 * rows * HD, a.wv.g, (int)rows, a.C, HD);
  mmBT(dqkvg, a.wq.w, dx, (int)rows, a.C, HD, accumulate ? 1.f : 0.f);
  mmBT(dqkvg + rows * HD, a.wk.w, dx, (int)rows, a.C, HD, 1.f);
  mmBT(dqkvg + 2 * rows * HD, a.wv.w, dx, (int)rows, a.C, HD, 1.f);
  if (a.gating) {
    const float* dg = dqkvg + 3 * rows * HD;
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
struct TransSave { LnSave ln; float* xn; float* h; size_t rows; };
inline TransSave transForward(const Transition& t, const float* x, float* out, size_t rows) {
  TransSave s;
  s.rows = rows;
  s.xn = ARENA.get(rows * t.ln.C);
  s.ln = lnForward(t.ln, x, s.xn, rows);
  s.h = ARENA.get(rows * t.l1.N);
  t.l1.forward(s.xn, s.h, rows);
  Scratch scratch;
  float* r = ARENA.get(rows * t.l1.N);
  reluK<<<blocks(rows * t.l1.N), 256, 0, STREAM>>>(r, s.h, rows * t.l1.N);
  t.l2.forward(r, out, rows);
  return s;
}
inline void transBackward(Transition& t, const TransSave& s, const float* dout, float* dx, bool accumulate) {
  Scratch scratch;
  size_t rows = s.rows, n = rows * t.l1.N;
  float* r = ARENA.get(n);
  reluK<<<blocks(n), 256, 0, STREAM>>>(r, s.h, n);
  float* dr = ARENA.get(n);
  t.l2.backward(r, dout, dr, rows);
  reluBwdK<<<blocks(n), 256, 0, STREAM>>>(dr, s.h, n);
  float* dxn = ARENA.get(rows * t.ln.C);
  t.l1.backward(s.xn, dr, dxn, rows);
  lnBackward(t.ln, s.ln, dxn, dx, rows, accumulate);
}

}  // namespace grad
