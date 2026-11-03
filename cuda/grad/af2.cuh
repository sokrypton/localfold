// AlphaFold 2's Evoformer block for training: forward with saves, backward to the inputs and to every parameter.
// af3-any-model's EvoformerIteration (alphafold3/af2/model/modules.py) - the multimer graph every AF2 checkpoint
// runs on - with the main stack's column attention or the extra-MSA stack's global column attention, and the outer
// product mean before the MSA track (multimer) or after it (a converted monomer). Fused triangle multiplication
// projections, as both regimes use.
//
// Held to cuda/grad/oracle_af2_block.py (jax.vjp in float64) by cuda/grad/test_af2_block.cu.
#pragma once
#include <functional>
#include <map>
#include "ops.cuh"

namespace grad {
namespace af2 {

// ---------------------------------------------------------------- triangle multiplication (fused projections)
struct TriMul {
  LayerNorm lnIn, lnCenter;
  Linear proj, gate, out, gating;     // [C, 2I], [C, 2I], [I, C], [C, C]
  int C = 128, I = 128;
  bool outgoing = true;
};
struct TriMulSave {
  LnSave lnIn, lnC;
  float *xn, *proj, *gate, *ab, *center, *cn, *og;   // og: sigmoid of the output gate
  int L;
};
// [L, L, 2I] (a|b halves) -> channel-major [2I][L][L]
__global__ void splitChannelsK(const float* __restrict__ pg, float* __restrict__ cm, int L, int C2) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C2) return;
  int c = t % C2; size_t ij = t / C2;
  cm[(size_t)c * L * L + ij] = pg[t];
}
__global__ void mergeChannelsK(const float* __restrict__ cm, float* __restrict__ pg, int L, int C2) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C2) return;
  int c = t % C2; size_t ij = t / C2;
  pg[t] = cm[(size_t)c * L * L + ij];
}

inline TriMulSave triMulForward(const TriMul& m, const float* pair, const float* mask, float* out, int L) {
  size_t rows = (size_t)L * L;
  const int I2 = 2 * m.I;
  TriMulSave s;
  s.L = L;
  s.xn = ARENA.get(rows * m.C);
  s.lnIn = lnForward(m.lnIn, pair, s.xn, rows);
  s.proj = ARENA.get(rows * I2); s.gate = ARENA.get(rows * I2);
  m.proj.forward(s.xn, s.proj, rows);
  rowMaskK<<<blocks(rows * I2), 256, 0, STREAM>>>(s.proj, mask, rows, I2);
  m.gate.forward(s.xn, s.gate, rows);
  sigmoidK<<<blocks(rows * I2), 256, 0, STREAM>>>(s.gate, rows * I2);
  s.ab = ARENA.get(rows * I2);                                    // channel-major a then b
  s.center = ARENA.get(rows * m.I);
  s.cn = ARENA.get(rows * m.I);
  s.og = ARENA.get(rows * m.C);
  {
    Scratch scratch;
    float* pg = ARENA.get(rows * I2);
    mulK<<<blocks(rows * I2), 256, 0, STREAM>>>(pg, s.proj, s.gate, rows * I2);
    splitChannelsK<<<blocks(rows * I2), 256, 0, STREAM>>>(pg, s.ab, L, I2);
    const float *A = s.ab, *B = s.ab + (size_t)m.I * rows;
    float* ccm = pg;                                              // reuse: [I][L][L]
    long long st = (long long)L * L;
    if (m.outgoing) bmm(false, true, L, L, L, A, st, B, st, ccm, st, m.I);   // o_ij = sum_k a_ik b_jk
    else bmm(true, false, L, L, L, B, st, A, st, ccm, st, m.I);              // o_ij = sum_k a_kj b_ki
    mergeChannelsK<<<blocks(rows * m.I), 256, 0, STREAM>>>(ccm, s.center, L, m.I);
  }
  s.lnC = lnForward(m.lnCenter, s.center, s.cn, rows);
  m.out.forward(s.cn, out, rows);
  m.gating.forward(s.xn, s.og, rows);
  sigmoidK<<<blocks(rows * m.C), 256, 0, STREAM>>>(s.og, rows * m.C);
  mulK<<<blocks(rows * m.C), 256, 0, STREAM>>>(out, out, s.og, rows * m.C);
  return s;
}

inline void triMulBackward(TriMul& m, const TriMulSave& s, const float* mask, const float* dout, float* dpair) {
  Scratch scratch;
  const int L = s.L, I2 = 2 * m.I;
  size_t rows = (size_t)L * L;
  // out = (cn Wout + bout) * og ; og = sigmoid(xn Wgating + b)
  float* pre = ARENA.get(rows * m.C);
  m.out.forward(s.cn, pre, rows);
  float *dpre = ARENA.get(rows * m.C), *dgl = ARENA.get(rows * m.C);
  gateBwdK<<<blocks(rows * m.C), 256, 0, STREAM>>>(dout, pre, s.og, dpre, dgl, rows * m.C);
  float* dxn = ARENA.get(rows * m.C);
  m.gating.backward(s.xn, dgl, dxn, rows);
  float* dcn = ARENA.get(rows * m.I);
  m.out.backward(s.cn, dpre, dcn, rows);
  float* dcenter = ARENA.get(rows * m.I);
  lnBackward(m.lnCenter, s.lnC, dcn, dcenter, rows, false);
  float* dccm = ARENA.get(rows * m.I);
  splitChannelsK<<<blocks(rows * m.I), 256, 0, STREAM>>>(dcenter, dccm, L, m.I);
  float* dab = ARENA.get(rows * I2);
  const float *A = s.ab, *B = s.ab + (size_t)m.I * rows;
  float *dA = dab, *dB = dab + (size_t)m.I * rows;
  long long st = (long long)L * L;
  if (m.outgoing) {          // O = A B^T : dA = dO B, dB = dO^T A
    bmm(false, false, L, L, L, dccm, st, B, st, dA, st, m.I);
    bmm(true, false, L, L, L, dccm, st, A, st, dB, st, m.I);
  } else {                   // O = B^T A : dA = B dO, dB = A dO^T
    bmm(false, false, L, L, L, B, st, dccm, st, dA, st, m.I);
    bmm(false, true, L, L, L, A, st, dccm, st, dB, st, m.I);
  }
  float* dpg = ARENA.get(rows * I2);
  mergeChannelsK<<<blocks(rows * I2), 256, 0, STREAM>>>(dab, dpg, L, I2);
  // pg = proj * gate (proj already masked): dproj = dpg gate (then masked), dgate_pre = dpg proj gate (1 - gate)
  float *dproj = ARENA.get(rows * I2), *dgp = ARENA.get(rows * I2);
  gateBwdK<<<blocks(rows * I2), 256, 0, STREAM>>>(dpg, s.proj, s.gate, dproj, dgp, rows * I2);
  rowMaskK<<<blocks(rows * I2), 256, 0, STREAM>>>(dproj, mask, rows, I2);
  m.proj.backward(s.xn, dproj, dxn, rows, true);
  m.gate.backward(s.xn, dgp, dxn, rows, true);
  lnBackward(m.lnIn, s.lnIn, dxn, dpair, rows, true);
}

// ---------------------------------------------------------------- outer product mean
struct Opm {
  LayerNorm ln;
  Linear left, right;     // [C, 32]
  Param wo, bo;           // [32, 32, Cz], [Cz]
  int C = 256, O = 32, Cz = 128;
};
struct OpmSave { LnSave ln; float *xn, *lm, *rm, *norm; int S, L; };
// lm [S, L, O] -> [S][L O] is already that layout. outer [(i c), (j e)] -> [(i j), (c e)]
__global__ void outerToPairK(const float* __restrict__ o2, float* __restrict__ op, int L, int O, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)L * L * O * O;
  if (t >= n) return;
  int e = t % O; size_t r = t / O; int c = r % O; r /= O; int j = r % L; int i = (int)(r / L);   // op[i, j, c, e]
  size_t q = (((size_t)i * O + c) * L + j) * O + e;
  if (back) op[q] = o2[t]; else op[t] = o2[q];
}
__global__ void normK(const float* __restrict__ mask, float* __restrict__ norm, int S, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L) return;
  int i = t / L, j = t % L;
  float s = 0;
  for (int a = 0; a < S; ++a) s += mask[(size_t)a * L + i] * mask[(size_t)a * L + j];
  norm[t] = 1e-3f + s;
}
__global__ void divRowsK(float* x, const float* norm, size_t rows, int C, bool accumulateInto, float* into) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  float v = x[t] / norm[t / C];
  if (accumulateInto) into[t] += v; else x[t] = v;
}

// pair += opm(msa)
inline OpmSave opmForward(const Opm& m, const float* msa, const float* msaMask, float* pair, int S, int L) {
  size_t rows = (size_t)S * L, pr = (size_t)L * L;
  OpmSave s; s.S = S; s.L = L;
  s.xn = ARENA.get(rows * m.C);
  s.ln = lnForward(m.ln, msa, s.xn, rows);
  s.lm = ARENA.get(rows * m.O); s.rm = ARENA.get(rows * m.O);
  m.left.forward(s.xn, s.lm, rows);  rowMaskK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.lm, msaMask, rows, m.O);
  m.right.forward(s.xn, s.rm, rows); rowMaskK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.rm, msaMask, rows, m.O);
  s.norm = ARENA.get(pr);
  normK<<<blocks(pr), 256, 0, STREAM>>>(msaMask, s.norm, S, L);
  const int LO = L * m.O, OO = m.O * m.O;
  Scratch scratch;
  float* o2 = ARENA.get((size_t)LO * LO);
  // o2[(i c), (j e)] = sum_s lm[s, (i c)] rm[s, (j e)]  =  lm^T rm
  mmAT(s.lm, s.rm, o2, S, LO, LO, 0.f);
  float* op = ARENA.get(pr * OO);
  outerToPairK<<<blocks(pr * OO), 256, 0, STREAM>>>(o2, op, L, m.O, false);
  float* act = o2;                         // reuse: [(i j), Cz]
  mm(op, m.wo.w, act, (int)pr, OO, m.Cz);
  biasK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(act, m.bo.w, pr, m.Cz);
  divRowsK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(act, s.norm, pr, m.Cz, true, pair);
  return s;
}
inline void opmBackward(Opm& m, const OpmSave& s, const float* msaMask, const float* dpair, float* dmsa) {
  Scratch scratch;
  const int S = s.S, L = s.L, LO = L * m.O, OO = m.O * m.O;
  size_t rows = (size_t)S * L, pr = (size_t)L * L;
  float* dact = ARENA.get(pr * m.Cz);
  GK(cudaMemcpyAsync(dact, dpair, pr * m.Cz * 4, cudaMemcpyDeviceToDevice, STREAM));
  divRowsK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(dact, s.norm, pr, m.Cz, false, nullptr);
  colSum(m.bo.g, dact, pr, m.Cz);
  float* o2 = ARENA.get((size_t)LO * LO);
  mmAT(s.lm, s.rm, o2, S, LO, LO, 0.f);
  float* op = ARENA.get(pr * OO);
  outerToPairK<<<blocks(pr * OO), 256, 0, STREAM>>>(o2, op, L, m.O, false);
  mmAT(op, dact, m.wo.g, (int)pr, OO, m.Cz);
  mmBT(dact, m.wo.w, op, (int)pr, OO, m.Cz);            // op <- d outer [(i j), (c e)]
  outerToPairK<<<blocks(pr * OO), 256, 0, STREAM>>>(op, o2, L, m.O, true);   // o2 <- d o2 [(i c), (j e)]
  float *dlm = ARENA.get(rows * m.O), *drm = ARENA.get(rows * m.O);
  mmBT(s.rm, o2, dlm, S, LO, LO);          // dlm = rm dO2^T
  mm(s.lm, o2, drm, S, LO, LO);            // drm = lm dO2
  rowMaskK<<<blocks(rows * m.O), 256, 0, STREAM>>>(dlm, msaMask, rows, m.O);
  rowMaskK<<<blocks(rows * m.O), 256, 0, STREAM>>>(drm, msaMask, rows, m.O);
  float* dxn = ARENA.get(rows * m.C);
  m.left.backward(s.xn, dlm, dxn, rows);
  m.right.backward(s.xn, drm, dxn, rows, true);
  lnBackward(m.ln, s.ln, dxn, dmsa, rows, true);
}

// ---------------------------------------------------------------- global column attention (the extra stack)
// per batch b (a residue column) over S rows: qa = masked mean of x; q = qa Wq D^-1/2 [H, D]; k = x Wk, v = x Wv
// [S, D] (one head's worth, shared); w = softmax_s(q k^T + mask); wa = w v [H, D]; out = (wa * sigmoid(x Wg + bg))
// Wo + bo.
struct GlobalAttention {
  Param wq, wk, wv, wg, bg, wo, bo;
  int C = 64, H = 8, D = 8;
};
struct GaSave { const float* x; float *k, *v, *g, *w, *wa, *q; int B, S; };
// one block a column b; S rows, H heads of D (D <= 32, H D <= 256)
__global__ void gaFwdK(const float* __restrict__ x, const float* __restrict__ mask, const float* __restrict__ wq,
                       const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ q,
                       float* __restrict__ w, float* __restrict__ wa, int B, int S, int C, int H, int D) {
  int b = blockIdx.x;
  extern __shared__ float sh[];
  float* qa = sh;               // [C]
  float* qs = sh + C;           // [H D]
  const float* xb = x + (size_t)b * S * C;
  const float* mb = mask + (size_t)b * S;
  float msum = 0;
  for (int s = 0; s < S; ++s) msum += mb[s];
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float a = 0;
    for (int s = 0; s < S; ++s) a += mb[s] * xb[(size_t)s * C + c];
    qa[c] = a / (msum + 1e-10f);
  }
  __syncthreads();
  float scale = rsqrtf((float)D);
  for (int t = threadIdx.x; t < H * D; t += blockDim.x) {
    float a = 0;
    for (int c = 0; c < C; ++c) a += qa[c] * wq[(size_t)c * H * D + t];
    qs[t] = a * scale;
    q[(size_t)b * H * D + t] = a * scale;
  }
  __syncthreads();
  // logits and softmax: one warp a head
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31, warps = blockDim.x / 32;
  for (int h = warp; h < H; h += warps) {
    float* wr = w + ((size_t)b * H + h) * S;
    float mx = -INFINITY;
    for (int s = lane; s < S; s += 32) {
      float a = 0;
      for (int d = 0; d < D; ++d) a += qs[h * D + d] * k[((size_t)b * S + s) * D + d];
      a += 1e9f * (mb[s] - 1.f);
      wr[s] = a; mx = fmaxf(mx, a);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    float sum = 0;
    for (int s = lane; s < S; s += 32) { float e = expf(wr[s] - mx); wr[s] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    for (int s = lane; s < S; s += 32) wr[s] /= sum;
    __syncwarp();
    for (int d = 0; d < D; ++d) {
      float a = 0;
      for (int s = lane; s < S; s += 32) a += wr[s] * v[((size_t)b * S + s) * D + d];
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
      if (lane == 0) wa[((size_t)b * H + h) * D + d] = a;
    }
  }
}
// y[b, s, (h d)] = wa[b, (h d)] * g[b, s, (h d)]
__global__ void gaMixK(const float* wa, const float* g, float* y, int B, int S, int HD) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * S * HD) return;
  int hd = t % HD; size_t bs = t / HD; int b = (int)(bs / S);
  y[t] = wa[(size_t)b * HD + hd] * g[t];
}
// dwa[b, hd] = sum_s dy[b, s, hd] g[b, s, hd] ; dgp = dy wa g (1 - g)
__global__ void gaMixBwdK(const float* dy, const float* wa, const float* g, float* dwa, float* dgp, int B, int S,
                          int HD) {
  int b = blockIdx.x;
  for (int hd = threadIdx.x; hd < HD; hd += blockDim.x) {
    float a = 0, w = wa[(size_t)b * HD + hd];
    for (int s = 0; s < S; ++s) {
      size_t t = ((size_t)b * S + s) * HD + hd;
      float gv = g[t];
      a += dy[t] * gv;
      dgp[t] = dy[t] * w * gv * (1.f - gv);
    }
    dwa[(size_t)b * HD + hd] = a;
  }
}
// the softmax/attention backward for one column: dwa [H, D] -> dq [H D] (scaled q), dk, dv [S, D]
__global__ void gaBwdK(const float* __restrict__ mask, const float* __restrict__ q, const float* __restrict__ k,
                       const float* __restrict__ v, const float* __restrict__ w, const float* __restrict__ dwa,
                       float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv, int B, int S, int H,
                       int D) {
  int b = blockIdx.x;
  extern __shared__ float sh[];      // ds [H][S]
  float* ds = sh;
  for (int t = threadIdx.x; t < H * S; t += blockDim.x) {
    int h = t / S, s = t % S;
    float dp = 0;
    for (int d = 0; d < D; ++d) dp += dwa[((size_t)b * H + h) * D + d] * v[((size_t)b * S + s) * D + d];
    ds[t] = dp;                       // dP
  }
  __syncthreads();
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31, warps = blockDim.x / 32;
  for (int h = warp; h < H; h += warps) {
    const float* wr = w + ((size_t)b * H + h) * S;
    float a = 0;
    for (int s = lane; s < S; s += 32) a += ds[h * S + s] * wr[s];
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
    for (int s = lane; s < S; s += 32) ds[h * S + s] = wr[s] * (ds[h * S + s] - a);
  }
  __syncthreads();
  for (int t = threadIdx.x; t < H * D; t += blockDim.x) {
    int h = t / D, d = t % D;
    float a = 0;
    for (int s = 0; s < S; ++s) a += ds[h * S + s] * k[((size_t)b * S + s) * D + d];
    dq[(size_t)b * H * D + t] = a;
  }
  for (int t = threadIdx.x; t < S * D; t += blockDim.x) {
    int s = t / D, d = t % D;
    float ak = 0, av = 0;
    for (int h = 0; h < H; ++h) {
      ak += ds[h * S + s] * q[((size_t)b * H + h) * D + d];
      av += w[((size_t)b * H + h) * S + s] * dwa[((size_t)b * H + h) * D + d];
    }
    dk[((size_t)b * S + s) * D + d] = ak;
    dv[((size_t)b * S + s) * D + d] = av;
  }
}
// dx[b, s, c] += mask[b, s] / (msum + eps) * dqa[b, c]
__global__ void gaMeanBwdK(const float* mask, const float* dqa, float* dx, int B, int S, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * S * C) return;
  int c = t % C; size_t bs = t / C; int b = (int)(bs / S);
  const float* mb = mask + (size_t)b * S;
  float msum = 0;
  for (int s = 0; s < S; ++s) msum += mb[s];
  dx[t] += mb[bs % S] / (msum + 1e-10f) * dqa[(size_t)b * C + c];
}
__global__ void gaMeanK(const float* x, const float* mask, float* qa, int B, int S, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)B * C) return;
  int c = t % C, b = (int)(t / C);
  float a = 0, m = 0;
  for (int s = 0; s < S; ++s) { float mv = mask[(size_t)b * S + s]; a += mv * x[((size_t)b * S + s) * C + c]; m += mv; }
  qa[t] = a / (m + 1e-10f);
}

inline GaSave gaForward(const GlobalAttention& a, const float* x, const float* mask, float* out, int B, int S) {
  size_t rows = (size_t)B * S;
  const int HD = a.H * a.D;
  GaSave s{x, ARENA.get(rows * a.D), ARENA.get(rows * a.D), ARENA.get(rows * HD), ARENA.get((size_t)B * a.H * S),
           ARENA.get((size_t)B * HD), ARENA.get((size_t)B * HD), B, S};
  mm(x, a.wk.w, s.k, (int)rows, a.C, a.D);
  mm(x, a.wv.w, s.v, (int)rows, a.C, a.D);
  mm(x, a.wg.w, s.g, (int)rows, a.C, HD);
  biasK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.g, a.bg.w, rows, HD);
  sigmoidK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.g, rows * HD);
  gaFwdK<<<B, 256, (a.C + HD) * 4, STREAM>>>(x, mask, a.wq.w, s.k, s.v, s.q, s.w, s.wa, B, S, a.C, a.H, a.D);
  Scratch scratch;
  float* y = ARENA.get(rows * HD);
  gaMixK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.wa, s.g, y, B, S, HD);
  mm(y, a.wo.w, out, (int)rows, HD, a.C);
  biasK<<<blocks(rows * a.C), 256, 0, STREAM>>>(out, a.bo.w, rows, a.C);
  return s;
}
inline void gaBackward(GlobalAttention& a, const GaSave& s, const float* mask, const float* dout, float* dx,
                       bool accumulate) {
  Scratch scratch;
  const int B = s.B, S = s.S, HD = a.H * a.D;
  size_t rows = (size_t)B * S;
  colSum(a.bo.g, dout, rows, a.C);
  float* y = ARENA.get(rows * HD);
  gaMixK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.wa, s.g, y, B, S, HD);
  mmAT(y, dout, a.wo.g, (int)rows, HD, a.C);
  float* dy = y;
  mmBT(dout, a.wo.w, dy, (int)rows, HD, a.C);
  float *dwa = ARENA.get((size_t)B * HD), *dgp = ARENA.get(rows * HD);
  gaMixBwdK<<<B, 128, 0, STREAM>>>(dy, s.wa, s.g, dwa, dgp, B, S, HD);
  float *dq = ARENA.get((size_t)B * HD), *dk = ARENA.get(rows * a.D), *dv = ARENA.get(rows * a.D);
  gaBwdK<<<B, 256, (size_t)a.H * S * 4, STREAM>>>(mask, s.q, s.k, s.v, s.w, dwa, dq, dk, dv, B, S, a.H, a.D);
  // q = qa Wq * scale: dWq += qa^T dq scale, dqa = dq Wq^T scale
  float* qa = ARENA.get((size_t)B * a.C);
  gaMeanK<<<blocks((size_t)B * a.C), 256, 0, STREAM>>>(s.x, mask, qa, B, S, a.C);
  const float scale = 1.f / sqrtf((float)a.D);
  mmAT(qa, dq, a.wq.g, B, a.C, HD, 1.f, scale);
  float* dqa = ARENA.get((size_t)B * a.C);
  mmBT(dq, a.wq.w, dqa, B, a.C, HD, 0.f, scale);
  mmAT(s.x, dk, a.wk.g, (int)rows, a.C, a.D);
  mmAT(s.x, dv, a.wv.g, (int)rows, a.C, a.D);
  mmAT(s.x, dgp, a.wg.g, (int)rows, a.C, HD);
  colSum(a.bg.g, dgp, rows, HD);
  mmBT(dk, a.wk.w, dx, (int)rows, a.C, a.D, accumulate ? 1.f : 0.f);
  mmBT(dv, a.wv.w, dx, (int)rows, a.C, a.D, 1.f);
  mmBT(dgp, a.wg.w, dx, (int)rows, a.C, HD, 1.f);
  gaMeanBwdK<<<blocks(rows * a.C), 256, 0, STREAM>>>(mask, dqa, dx, B, S, a.C);
}

// ---------------------------------------------------------------- attention modules with their norms
struct RowAttn {           // MSA row attention with pair bias
  LayerNorm qn, pn;
  Param feat2d;            // [Cz, H]
  Attention at;
};
struct TriAttn {           // triangle attention (starting node: per_row; ending node: per_column)
  LayerNorm qn;
  Param feat2d;            // [Cz, H]
  Attention at;
  bool perColumn = false;
};
struct ColAttn {           // the main stack's MSA column attention
  LayerNorm qn;
  Attention at;
};
struct ColGlobal {
  LayerNorm qn;
  GlobalAttention at;
};

// nb[h, q, k] = sum_c pn[q, k, c] w[c, h] (+ 1e9 (pairMask - 1) for the row attention): pn [L L, Cz] -> [H, L L]
inline void pairBias(const float* pn, const float* w, float* nb, int L, int Cz, int H) {
  Scratch scratch;
  float* t = ARENA.get((size_t)L * L * H);
  mm(pn, w, t, L * L, Cz, H);
  transpose(t, nb, L * L, H);
}
__global__ void addMaskBiasK(float* nb, const float* pairMask, int H, size_t LL) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)H * LL) nb[t] += 1e9f * (pairMask[t % LL] - 1.f);
}

// ---------------------------------------------------------------- the block
struct Block {
  bool extra = false, opmFirst = true;
  int Cm = 256, Cz = 128;
  Opm opm;
  RowAttn row;
  ColAttn col;
  ColGlobal colg;
  Transition msaTrans, pairTrans;
  TriMul triOut, triIn;
  TriAttn triStart, triEnd;
  Params params;
};

// every parameter, by haiku's name under the block
inline void forEachParam(Block& b, const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(n + "/scale", l.g, C); f(n + "/offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; f(n + "/weights", l.w, (size_t)K * N); f(n + "/bias", l.b, N); };
  auto att = [&](const std::string& n, Attention& a, int C, int H, int D, int O) {
    a.C = C; a.H = H; a.D = D; a.O = O; a.gating = true;
    size_t HD = (size_t)H * D;
    f(n + "/query_w", a.wq, C * HD); f(n + "/key_w", a.wk, C * HD); f(n + "/value_w", a.wv, C * HD);
    f(n + "/gating_w", a.wg, C * HD); f(n + "/gating_b", a.bg, HD);
    f(n + "/output_w", a.wo, HD * O); f(n + "/output_b", a.bo, O);
  };
  const int Cm = b.Cm, Cz = b.Cz;
  ln("outer_product_mean/layer_norm_input", b.opm.ln, Cm);
  b.opm.C = Cm;
  lin("outer_product_mean/left_projection", b.opm.left, Cm, 32);
  lin("outer_product_mean/right_projection", b.opm.right, Cm, 32);
  f("outer_product_mean/output_w", b.opm.wo, (size_t)32 * 32 * Cz); f("outer_product_mean/output_b", b.opm.bo, Cz);
  ln("msa_row_attention_with_pair_bias/query_norm", b.row.qn, Cm);
  ln("msa_row_attention_with_pair_bias/feat_2d_norm", b.row.pn, Cz);
  f("msa_row_attention_with_pair_bias/feat_2d_weights", b.row.feat2d, (size_t)Cz * 8);
  att("msa_row_attention_with_pair_bias/attention", b.row.at, Cm, 8, Cm / 8, Cm);
  if (!b.extra) {
    ln("msa_column_attention/query_norm", b.col.qn, Cm);
    att("msa_column_attention/attention", b.col.at, Cm, 8, Cm / 8, Cm);
  } else {
    auto& g = b.colg.at;
    ln("msa_column_global_attention/query_norm", b.colg.qn, Cm);
    g.C = Cm; g.H = 8; g.D = Cm / 8;
    const std::string n = "msa_column_global_attention/attention";
    size_t HD = (size_t)g.H * g.D;
    f(n + "/query_w", g.wq, Cm * HD); f(n + "/key_w", g.wk, (size_t)Cm * g.D); f(n + "/value_w", g.wv, (size_t)Cm * g.D);
    f(n + "/gating_w", g.wg, Cm * HD); f(n + "/gating_b", g.bg, HD);
    f(n + "/output_w", g.wo, HD * Cm); f(n + "/output_b", g.bo, Cm);
  }
  ln("msa_transition/input_layer_norm", b.msaTrans.ln, Cm);
  lin("msa_transition/transition1", b.msaTrans.l1, Cm, 4 * Cm);
  lin("msa_transition/transition2", b.msaTrans.l2, 4 * Cm, Cm);
  for (auto* t : {&b.triOut, &b.triIn}) {
    std::string n = t == &b.triOut ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming";
    t->C = Cz; t->I = Cz; t->outgoing = t == &b.triOut;
    ln(n + "/left_norm_input", t->lnIn, Cz);
    ln(n + "/center_norm", t->lnCenter, Cz);
    lin(n + "/projection", t->proj, Cz, 2 * Cz);
    lin(n + "/gate", t->gate, Cz, 2 * Cz);
    lin(n + "/output_projection", t->out, Cz, Cz);
    lin(n + "/gating_linear", t->gating, Cz, Cz);
  }
  for (auto* t : {&b.triStart, &b.triEnd}) {
    std::string n = t == &b.triStart ? "triangle_attention_starting_node" : "triangle_attention_ending_node";
    t->perColumn = t == &b.triEnd;
    ln(n + "/query_norm", t->qn, Cz);
    f(n + "/feat_2d_weights", t->feat2d, (size_t)Cz * 4);
    att(n + "/attention", t->at, Cz, 4, Cz / 4, Cz);
  }
  ln("pair_transition/input_layer_norm", b.pairTrans.ln, Cz);
  lin("pair_transition/transition1", b.pairTrans.l1, Cz, 4 * Cz);
  lin("pair_transition/transition2", b.pairTrans.l2, 4 * Cz, Cz);
}

inline void allocBlock(Block& b) {
  forEachParam(b, [&](const std::string& n, Param& p, size_t len) { allocParam(p, n, len); b.params.add(p); });
}

// What the backward needs from the forward, per module.
struct BlockSave {
  int S, L;
  const float *msa0, *pair0;      // the block's inputs (owned by the caller)
  float *msa1, *msa2, *msa3;      // the MSA track after row attention, column attention (transposed back), transition
  float *pair1, *pair2, *pair3, *pair4, *pair5;   // after OPM (or the input), triOut, triIn, triStart, triEnd
  OpmSave opm;
  LnSave rowQ, rowP; float *rowXn, *rowPn, *rowNb; AttnSave rowA;
  LnSave colQ; float *colX, *colXn, *colMask; AttnSave colA; GaSave colG;
  TransSave msaT, pairT;
  TriMulSave tOut, tIn;
  LnSave stQ, enQ; float *stXn, *enX, *enXn, *stNb, *enNb, *pairMaskT; AttnSave stA, enA;
  float *pairOpm;                 // the pair after OPM when opm runs last
};

__global__ void copyK(float* d, const float* s, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) d[t] = s[t];
}
inline float* clone(const float* x, size_t n) {
  float* y = ARENA.get(n);
  GK(cudaMemcpyAsync(y, x, n * 4, cudaMemcpyDeviceToDevice, STREAM));
  return y;
}

// msa [S, L, Cm], pair [L, L, Cz] -> msaOut, pairOut (may alias nothing). msaMask [S, L], pairMask [L, L].
inline BlockSave blockForward(Block& b, const float* msa, const float* pair, const float* msaMask, const float* pairMask,
                              float* msaOut, float* pairOut, int S, int L) {
  BlockSave s{};
  s.S = S; s.L = L; s.msa0 = msa; s.pair0 = pair;
  const int Cm = b.Cm, Cz = b.Cz;
  size_t mr = (size_t)S * L, pr = (size_t)L * L;
  float* pcur = clone(pair, pr * Cz);
  if (b.opmFirst) s.opm = opmForward(b.opm, msa, msaMask, pcur, S, L);
  s.pair1 = pcur;
  // MSA row attention with pair bias
  s.rowXn = ARENA.get(mr * Cm);
  s.rowQ = lnForward(b.row.qn, msa, s.rowXn, mr);
  s.rowPn = ARENA.get(pr * Cz);
  s.rowP = lnForward(b.row.pn, s.pair1, s.rowPn, pr);
  s.rowNb = ARENA.get(pr * 8);
  pairBias(s.rowPn, b.row.feat2d.w, s.rowNb, L, Cz, 8);
  addMaskBiasK<<<blocks(pr * 8), 256, 0, STREAM>>>(s.rowNb, pairMask, 8, pr);
  float* upd = ARENA.get(mr * Cm);
  s.rowA = attnForward(b.row.at, s.rowXn, msaMask, s.rowNb, upd, S, L);
  s.msa1 = clone(msa, mr * Cm);
  add(s.msa1, upd, mr * Cm);
  // MSA column attention, on the transposed MSA [L, S, C]
  s.colX = ARENA.get(mr * Cm);
  swap01(s.msa1, s.colX, S, L, Cm);
  s.colMask = ARENA.get(mr);
  transpose(msaMask, s.colMask, S, L);
  s.colXn = ARENA.get(mr * Cm);
  float* updT = ARENA.get(mr * Cm);
  if (!b.extra) {
    s.colQ = lnForward(b.col.qn, s.colX, s.colXn, mr);
    s.colA = attnForward(b.col.at, s.colXn, s.colMask, nullptr, updT, L, S);
  } else {
    s.colQ = lnForward(b.colg.qn, s.colX, s.colXn, mr);
    s.colG = gaForward(b.colg.at, s.colXn, s.colMask, updT, L, S);
  }
  s.msa2 = clone(s.msa1, mr * Cm);
  swap01(updT, s.msa2, L, S, Cm, true);
  // MSA transition
  s.msaT = transForward(b.msaTrans, s.msa2, upd, mr);
  s.msa3 = clone(s.msa2, mr * Cm);
  add(s.msa3, upd, mr * Cm);
  float* pc = s.pair1;
  if (!b.opmFirst) {
    pc = clone(s.pair1, pr * Cz);
    s.opm = opmForward(b.opm, s.msa3, msaMask, pc, S, L);
    s.pairOpm = pc;
  }
  // triangle multiplications
  float* pu = ARENA.get(pr * Cz);
  s.tOut = triMulForward(b.triOut, pc, pairMask, pu, L);
  s.pair2 = clone(pc, pr * Cz); add(s.pair2, pu, pr * Cz);
  s.tIn = triMulForward(b.triIn, s.pair2, pairMask, pu, L);
  s.pair3 = clone(s.pair2, pr * Cz); add(s.pair3, pu, pr * Cz);
  // triangle attention, starting node (per row)
  s.stXn = ARENA.get(pr * Cz);
  s.stQ = lnForward(b.triStart.qn, s.pair3, s.stXn, pr);
  s.stNb = ARENA.get(pr * 4);
  pairBias(s.stXn, b.triStart.feat2d.w, s.stNb, L, Cz, 4);
  s.stA = attnForward(b.triStart.at, s.stXn, pairMask, s.stNb, pu, L, L);
  s.pair4 = clone(s.pair3, pr * Cz); add(s.pair4, pu, pr * Cz);
  // ending node: on the transposed pair, with the transposed mask
  s.enX = ARENA.get(pr * Cz);
  swap01(s.pair4, s.enX, L, L, Cz);
  s.pairMaskT = ARENA.get(pr);
  transpose(pairMask, s.pairMaskT, L, L);
  s.enXn = ARENA.get(pr * Cz);
  s.enQ = lnForward(b.triEnd.qn, s.enX, s.enXn, pr);
  s.enNb = ARENA.get(pr * 4);
  pairBias(s.enXn, b.triEnd.feat2d.w, s.enNb, L, Cz, 4);
  s.enA = attnForward(b.triEnd.at, s.enXn, s.pairMaskT, s.enNb, pu, L, L);
  s.pair5 = clone(s.pair4, pr * Cz);
  swap01(pu, s.pair5, L, L, Cz, true);
  // pair transition
  s.pairT = transForward(b.pairTrans, s.pair5, pu, pr);
  GK(cudaMemcpyAsync(pairOut, s.pair5, pr * Cz * 4, cudaMemcpyDeviceToDevice, STREAM));
  add(pairOut, pu, pr * Cz);
  GK(cudaMemcpyAsync(msaOut, s.msa3, mr * Cm * 4, cudaMemcpyDeviceToDevice, STREAM));
  return s;
}

// the gradient of a pair bias nb = transpose(pn W) to pn and W: dnb [H, L L]
inline void pairBiasBackward(const float* pn, Param& w, const float* dnb, float* dpn, int L, int Cz, int H,
                             bool accumulate) {
  Scratch scratch;
  float* t = ARENA.get((size_t)L * L * H);
  transpose(dnb, t, H, L * L);                // [L L, H]
  mmAT(pn, t, w.g, L * L, Cz, H);
  mmBT(t, w.w, dpn, L * L, Cz, H, accumulate ? 1.f : 0.f);
}

// dmsaOut, dpairOut -> dmsa, dpair (written). Every parameter gradient accumulates.
inline void blockBackward(Block& b, const BlockSave& s, const float* msaMask, const float* pairMask,
                          const float* dmsaOut, const float* dpairOut, float* dmsa, float* dpair) {
  const int S = s.S, L = s.L, Cm = b.Cm, Cz = b.Cz;
  size_t mr = (size_t)S * L, pr = (size_t)L * L;
  float* dp = clone(dpairOut, pr * Cz);        // the running gradient of the pair
  float* dm = clone(dmsaOut, mr * Cm);         // and of the MSA
  // pair transition
  transBackward(b.pairTrans, s.pairT, dpairOut, dp, true);
  // ending node: pair5 = pair4 + swap(attn(enXn))
  {
    Scratch scratch;
    float* dupdT = ARENA.get(pr * Cz);
    swap01(dp, dupdT, L, L, Cz);
    float* dnb = ARENA.zeros(pr * 4);
    float* dxn = ARENA.get(pr * Cz);
    attnBackward(b.triEnd.at, s.enA, s.pairMaskT, s.enNb, dupdT, dxn, dnb, false);
    pairBiasBackward(s.enXn, b.triEnd.feat2d, dnb, dxn, L, Cz, 4, true);
    float* dx = ARENA.get(pr * Cz);
    lnBackward(b.triEnd.qn, s.enQ, dxn, dx, pr, false);
    swap01(dx, dp, L, L, Cz, true);
  }
  // starting node
  {
    Scratch scratch;
    float* dnb = ARENA.zeros(pr * 4);
    float* dxn = ARENA.get(pr * Cz);
    attnBackward(b.triStart.at, s.stA, pairMask, s.stNb, dp, dxn, dnb, false);
    pairBiasBackward(s.stXn, b.triStart.feat2d, dnb, dxn, L, Cz, 4, true);
    lnBackward(b.triStart.qn, s.stQ, dxn, dp, pr, true);
  }
  // triangle multiplications: pair3 = pair2 + tIn(pair2); pair2 = pc + tOut(pc)
  {
    Scratch scratch;
    float* dpu = clone(dp, pr * Cz);
    triMulBackward(b.triIn, s.tIn, pairMask, dpu, dp);
    dpu = clone(dp, pr * Cz);
    triMulBackward(b.triOut, s.tOut, pairMask, dpu, dp);
  }
  if (!b.opmFirst) opmBackward(b.opm, s.opm, msaMask, dp, dm);    // pc = pair1 + opm(msa3)
  // MSA transition: msa3 = msa2 + T(msa2)
  transBackward(b.msaTrans, s.msaT, dm, dm, true);
  // column attention: msa2 = msa1 + swap(col(swap(msa1)))
  {
    Scratch scratch;
    float* dupdT = ARENA.get(mr * Cm);
    swap01(dm, dupdT, S, L, Cm);
    float* dxn = ARENA.get(mr * Cm);
    float* dx = ARENA.get(mr * Cm);
    if (!b.extra) {
      attnBackward(b.col.at, s.colA, s.colMask, nullptr, dupdT, dxn, nullptr, false);
      lnBackward(b.col.qn, s.colQ, dxn, dx, mr, false);
    } else {
      gaBackward(b.colg.at, s.colG, s.colMask, dupdT, dxn, false);
      lnBackward(b.colg.qn, s.colQ, dxn, dx, mr, false);
    }
    swap01(dx, dm, L, S, Cm, true);
  }
  // row attention: msa1 = msa + row(LN(msa), bias(LN(pair1)))
  {
    Scratch scratch;
    float* dnb = ARENA.zeros(pr * 8);
    float* dxn = ARENA.get(mr * Cm);
    attnBackward(b.row.at, s.rowA, msaMask, s.rowNb, dm, dxn, dnb, false);
    lnBackward(b.row.qn, s.rowQ, dxn, dm, mr, true);
    float* dpn = ARENA.get(pr * Cz);
    pairBiasBackward(s.rowPn, b.row.feat2d, dnb, dpn, L, Cz, 8, false);
    lnBackward(b.row.pn, s.rowP, dpn, dp, pr, true);
  }
  if (b.opmFirst) opmBackward(b.opm, s.opm, msaMask, dp, dm);     // pair1 = pair + opm(msa)
  GK(cudaMemcpyAsync(dmsa, dm, mr * Cm * 4, cudaMemcpyDeviceToDevice, STREAM));
  GK(cudaMemcpyAsync(dpair, dp, pr * Cz * 4, cudaMemcpyDeviceToDevice, STREAM));
}

}  // namespace af2
}  // namespace grad

namespace grad {
namespace af2 {

// ---------------------------------------------------------------- a stack of blocks, rematerialised
// The forward keeps every block's INPUT (msa and pair, fp32) and nothing else; the backward replays block i's
// forward with its saves straight before block i's backward, and resets the arena after. Held: n block inputs plus
// ONE block's activations, for the price of one extra forward - AF2's own use_remat.
struct Stack {
  std::vector<Block> blocks;
  std::vector<float*> msaIn, pairIn;     // [n + 1]: block i's input, and the stack's output last
  int S = 0, L = 0;
  size_t inputBytes() const { return (msaIn.size() * (size_t)S * L * blocks[0].Cm + pairIn.size() * (size_t)L * L * 128) * 4; }
  void ensure(int S_, int L_) {
    if (S_ == S && L_ == L && !msaIn.empty()) return;
    for (float* p : msaIn) GK(cudaFree(p));
    for (float* p : pairIn) GK(cudaFree(p));
    msaIn.assign(blocks.size() + 1, nullptr); pairIn.assign(blocks.size() + 1, nullptr);
    S = S_; L = L_;
    for (size_t i = 0; i <= blocks.size(); ++i) {
      GK(cudaMalloc(&msaIn[i], (size_t)S * L * blocks[0].Cm * 4));
      GK(cudaMalloc(&pairIn[i], (size_t)L * L * 128 * 4));
    }
  }
  // msa/pair -> msaIn.back()/pairIn.back()
  void forward(const float* msa, const float* pair, const float* msaMask, const float* pairMask, int S_, int L_) {
    ensure(S_, L_);
    GK(cudaMemcpyAsync(msaIn[0], msa, (size_t)S * L * blocks[0].Cm * 4, cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(pairIn[0], pair, (size_t)L * L * 128 * 4, cudaMemcpyDeviceToDevice, STREAM));
    for (size_t i = 0; i < blocks.size(); ++i) {
      ARENA.reset();
      blockForward(blocks[i], msaIn[i], pairIn[i], msaMask, pairMask, msaIn[i + 1], pairIn[i + 1], S, L);
    }
    ARENA.reset();
  }
  // gradients of the stack's output -> of its input (dmsa, dpair: in place)
  void backward(const float* msaMask, const float* pairMask, float* dmsa, float* dpair) {
    float *om, *op;
    GK(cudaMallocAsync(&om, (size_t)S * L * blocks[0].Cm * 4, STREAM));
    GK(cudaMallocAsync(&op, (size_t)L * L * 128 * 4, STREAM));
    for (size_t i = blocks.size(); i-- > 0;) {
      ARENA.reset();
      BlockSave s = blockForward(blocks[i], msaIn[i], pairIn[i], msaMask, pairMask, om, op, S, L);
      blockBackward(blocks[i], s, msaMask, pairMask, dmsa, dpair, dmsa, dpair);
    }
    ARENA.reset();
    GK(cudaFreeAsync(om, STREAM)); GK(cudaFreeAsync(op, STREAM));
  }
};

}  // namespace af2
}  // namespace grad
