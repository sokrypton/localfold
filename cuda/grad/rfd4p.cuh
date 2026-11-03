// RFD4-Proteina's trunk (`local_latents_transformer`) for training; cpu/rfd4p/reference.js is the specification and
// cuda/rfd4p/src/trunk.cuh's float32 path what the forward is held to. Torch (out, in) weights throughout.
//
//   token layer:  s += adaScale(PairBiasAttention(adaLN(s, c)), c);  s += adaScale(SwiGLU(adaLN(s, c)), c)
//     adaLN(x, c)   = RMS(x) * sigmoid(RMS(c; w) Wg + bg) + RMS(c; w) Wb          (zero on a masked token)
//     adaScale(x,c) = x * sigmoid(c Wz + bz)                                       (zero on a masked token)
//     attention     = RMS-normed nodes; q, k RMS-normed; logits + LN-free RMS(pair; w) Wbias, key-masked; times
//                     sigmoid(g); to_out_node with its bias
//     SwiGLU        = silu(second half) * first half, then linear_out
//   pair update:  z = masked(z + pij[j, :P] * pij[i, P:]),  pij = LN(masked s) Wx;
//                 z += triangle out; z += triangle in (four projections, LayerNorms in and out, the incoming form
//                 sum_k a_ki b_kj); z += relu transition - every add on the valid pairs only
#pragma once
#include "af3.cuh"

namespace grad {
namespace rfd4p {

// ---------------------------------------------------------------- RMSNorm (eps 1e-5), strided rows
__global__ void rmsFwdK(const real* x, int ldx, const real* w, real* y, int ldy, real* inv, size_t rows, int C, real eps) {
  size_t r = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (r >= rows) return;
  const real* xr = x + r * ldx;
  real v = 0;
  for (int c = lane; c < C; c += 32) v += xr[c] * xr[c];
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  real iv = 1 / sqrt(v / C + eps);
  for (int c = lane; c < C; c += 32) y[r * ldy + c] = xr[c] * iv * (w ? w[c] : (real)1);
  if (lane == 0) inv[r] = iv;
}
// y = x iv w: dx = iv (dy w - x iv^2 mean(dy w x)), dw += dy x iv
__global__ void rmsBwdK(const real* x, int ldx, const real* w, const real* inv, const real* dy, int lddy, real* dx, int lddx,
                        real* dw, size_t rows, int C, bool accumulate) {
  extern __shared__ real sh[];
  for (int c = threadIdx.x; c < C; c += blockDim.x) sh[c] = 0;
  __syncthreads();
  int warps = blockDim.x / 32, lane = threadIdx.x & 31;
  for (size_t r = (size_t)blockIdx.x * warps + threadIdx.x / 32; r < rows; r += (size_t)gridDim.x * warps) {
    const real *xr = x + r * ldx, *dr = dy + r * lddy;
    real iv = inv[r], s = 0;
    for (int c = lane; c < C; c += 32) {
      real dw_ = dr[c] * (w ? w[c] : (real)1);
      s += dw_ * xr[c];
      if (dw) atomicAdd(&sh[c], dr[c] * xr[c] * iv);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    s /= C;
    for (int c = lane; c < C; c += 32) {
      real v = iv * (dr[c] * (w ? w[c] : (real)1) - xr[c] * iv * iv * s);
      real& o = dx[r * lddx + c];
      o = accumulate ? o + v : v;
    }
  }
  __syncthreads();
  if (dw) for (int c = threadIdx.x; c < C; c += blockDim.x) atomicAdd(&dw[c], sh[c]);
}
struct Rms { Param w; int C = 0; bool hasW = true; };
struct RmsSave { const real* x; int ldx; real* inv; size_t rows; };
inline RmsSave rmsForward(const Rms& n, const real* x, int ldx, real* y, int ldy, size_t rows) {
  RmsSave s{x, ldx, ARENA.get(rows), rows};
  rmsFwdK<<<blocks(rows, 8), 256, 0, STREAM>>>(x, ldx, n.hasW ? n.w.w : nullptr, y, ldy, s.inv, rows, n.C, (real)1e-5);
  return s;
}
inline void rmsBackward(Rms& n, const RmsSave& s, const real* dy, int lddy, real* dx, int lddx, bool accumulate) {
  unsigned g = (unsigned)std::min<size_t>(blocks(s.rows, 8), 1024);
  rmsBwdK<<<g, 256, n.C * sizeof(real), STREAM>>>(s.x, s.ldx, n.hasW ? n.w.w : nullptr, s.inv, dy, lddy, dx, lddx,
                                                  n.hasW ? n.w.g : nullptr, s.rows, n.C, accumulate);
}

__global__ void maskRowsK(real* x, const real* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C && mask[t / C] == 0) x[t] = 0;
}

// ---------------------------------------------------------------- adaLN and the output scale
struct AdaLnR { Rms xn, cn; Linear gamma, beta; int D = 0; };    // gamma with a bias; beta without
struct AdaRSave { RmsSave sx, sc; real *xn, *cn, *g; size_t rows; };
__global__ void adaRK(const real* xn, const real* gpre, const real* beta, const real* mask, real* out, size_t rows, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * D) return;
  if (mask[t / D] == 0) { out[t] = 0; return; }
  out[t] = xn[t] / (1 + exp(-gpre[t])) + beta[t];
}
__global__ void adaRBwdK(const real* xn, const real* gpre, const real* mask, const real* dout, real* dxn, real* dg,
                         real* dbeta, size_t rows, int D) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * D) return;
  real m = mask[t / D] == 0 ? 0 : 1, s = 1 / (1 + exp(-gpre[t])), d = dout[t] * m;
  dxn[t] = d * s; dg[t] = d * xn[t] * s * (1 - s); dbeta[t] = d;
}
inline AdaRSave adaRForward(const AdaLnR& a, const real* x, const real* c, const real* mask, real* out, size_t rows) {
  AdaRSave s{}; s.rows = rows;
  s.xn = ARENA.get(rows * a.D); s.sx = rmsForward(a.xn, x, a.D, s.xn, a.D, rows);
  s.cn = ARENA.get(rows * a.cn.C); s.sc = rmsForward(a.cn, c, a.cn.C, s.cn, a.cn.C, rows);
  s.g = ARENA.get(rows * a.D); a.gamma.forward(s.cn, s.g, rows);
  Scratch scratch;
  real* b = ARENA.get(rows * a.D); a.beta.forward(s.cn, b, rows);
  adaRK<<<blocks(rows * a.D), 256, 0, STREAM>>>(s.xn, s.g, b, mask, out, rows, a.D);
  return s;
}
// dout -> dx (added), dc (added)
inline void adaRBackward(AdaLnR& a, const AdaRSave& s, const real* mask, const real* dout, real* dx, real* dc) {
  Scratch scratch;
  size_t rows = s.rows;
  real *dxn = ARENA.get(rows * a.D), *dg = ARENA.get(rows * a.D), *db = ARENA.get(rows * a.D), *dcn = ARENA.get(rows * a.cn.C);
  adaRBwdK<<<blocks(rows * a.D), 256, 0, STREAM>>>(s.xn, s.g, mask, dout, dxn, dg, db, rows, a.D);
  a.gamma.backward(s.cn, dg, dcn, rows);
  a.beta.backward(s.cn, db, dcn, rows, true);
  rmsBackward(a.cn, s.sc, dcn, a.cn.C, dc, a.cn.C, true);
  rmsBackward(a.xn, s.sx, dxn, a.D, dx, a.D, true);
}
struct AdaScale { Linear z; };                 // [C, D] with its bias
struct AsSave { const real* x; const real* c; real* g; size_t rows; };
// res += mask x sigmoid(c Wz + bz)
inline AsSave asForward(const AdaScale& a, const real* x, const real* c, const real* mask, real* res, size_t rows) {
  AsSave s{x, c, ARENA.get(rows * a.z.N), rows};
  a.z.forward(c, s.g, rows);
  sigmoidK<<<blocks(rows * a.z.N), 256, 0, STREAM>>>(s.g, rows * a.z.N);
  Scratch scratch;
  real* t = ARENA.get(rows * a.z.N);
  mulK<<<blocks(rows * a.z.N), 256, 0, STREAM>>>(t, x, s.g, rows * a.z.N);
  maskRowsK<<<blocks(rows * a.z.N), 256, 0, STREAM>>>(t, mask, rows, a.z.N);
  add(res, t, rows * a.z.N);
  return s;
}
inline void asBackward(AdaScale& a, const AsSave& s, const real* mask, const real* dres, real* dx, real* dc) {
  Scratch scratch;
  const size_t rows = s.rows; const int D = a.z.N;
  real *d = af3::cloneR(dres, rows * D), *dz = ARENA.get(rows * D);
  maskRowsK<<<blocks(rows * D), 256, 0, STREAM>>>(d, mask, rows, D);
  gateBwdK<<<blocks(rows * D), 256, 0, STREAM>>>(d, s.x, s.g, dx, dz, rows * D);
  a.z.backward(s.c, dz, dc, rows, true);
}

// ---------------------------------------------------------------- pair-biased attention
struct PbAttn {
  Rms node, qn, kn, pn;
  Linear qkv, g, out, bias;      // qkv [D, 3D] with a bias, g [D, D] with a bias, out [D, D] with a bias, bias [P, H]
  int D = 0, P = 0, H = 0;
};
struct PbSave { RmsSave sn, sq, sk, sp; real *nn, *qkv, *q, *k, *g, *pl, *pn, *o; int n; };
inline Sdpa pbSdpa(const PbAttn& a, int n, const real* mask, const real* pl, real* dpl = nullptr) {
  Sdpa s;
  s.B = 1; s.H = a.H; s.Q = n; s.K = n; s.D = a.D / a.H; s.scale = 1 / sqrt((real)(a.D / a.H));
  s.keyMask = mask;
  s.bias[0] = Bias{pl, 0, (long long)n * n, n, dpl};
  return s;
}
inline PbSave pbForward(const PbAttn& a, const real* x, const real* pair, const real* mask, real* out, int n) {
  const int D = a.D; const size_t nn = (size_t)n * n;
  PbSave s{}; s.n = n;
  s.nn = ARENA.get((size_t)n * D); s.sn = rmsForward(a.node, x, D, s.nn, D, n);
  s.qkv = ARENA.get((size_t)n * 3 * D); a.qkv.forward(s.nn, s.qkv, n);
  s.q = ARENA.get((size_t)n * D); s.k = ARENA.get((size_t)n * D);
  s.sq = rmsForward(a.qn, s.qkv, 3 * D, s.q, D, n);
  s.sk = rmsForward(a.kn, s.qkv + D, 3 * D, s.k, D, n);
  s.g = ARENA.get((size_t)n * D); a.g.forward(s.nn, s.g, n);
  sigmoidK<<<blocks((size_t)n * D), 256, 0, STREAM>>>(s.g, (size_t)n * D);
  s.pn = ARENA.get(nn * a.P); s.sp = rmsForward(a.pn, pair, a.P, s.pn, a.P, nn);
  s.pl = ARENA.get(nn * a.H);
  {
    Scratch scratch;
    real* raw = ARENA.get(nn * a.H);
    a.bias.forward(s.pn, raw, nn);
    af3::gridBiasK<<<blocks(nn * a.H), 256, 0, STREAM>>>(raw, s.pl, n, a.H, false, false);
  }
  s.o = ARENA.get((size_t)n * D);
  sdpaForward(pbSdpa(a, n, mask, s.pl), s.q, D, s.k, D, s.qkv + 2 * D, 3 * D, s.o, D);
  Scratch scratch;
  real* og = ARENA.get((size_t)n * D);
  mulK<<<blocks((size_t)n * D), 256, 0, STREAM>>>(og, s.o, s.g, (size_t)n * D);
  a.out.forward(og, out, n);
  return s;
}
// dout -> dx (added), dpair (added)
inline void pbBackward(PbAttn& a, const PbSave& s, const real* mask, const real* dout, real* dx, real* dpair) {
  Scratch scratch;
  const int n = s.n, D = a.D; const size_t nn = (size_t)n * n;
  real* og = ARENA.get((size_t)n * D);
  mulK<<<blocks((size_t)n * D), 256, 0, STREAM>>>(og, s.o, s.g, (size_t)n * D);
  real* dog = ARENA.get((size_t)n * D);
  a.out.backward(og, dout, dog, n);
  real *dO = ARENA.get((size_t)n * D), *dg = ARENA.get((size_t)n * D);
  gateBwdK<<<blocks((size_t)n * D), 256, 0, STREAM>>>(dog, s.o, s.g, dO, dg, (size_t)n * D);
  real *dq = ARENA.get((size_t)n * D), *dk = ARENA.get((size_t)n * D), *dqkv = ARENA.get((size_t)n * 3 * D);
  real* dpl = ARENA.zeros(nn * a.H);
  sdpaBackward(pbSdpa(a, n, mask, s.pl, dpl), s.q, D, s.k, D, s.qkv + 2 * D, 3 * D, dO, D, dq, D, dk, D, dqkv + 2 * D, 3 * D);
  rmsBackward(a.qn, s.sq, dq, D, dqkv, 3 * D, false);
  rmsBackward(a.kn, s.sk, dk, D, dqkv + D, 3 * D, false);
  real* dnn = ARENA.get((size_t)n * D);
  a.qkv.backward(s.nn, dqkv, dnn, n);
  a.g.backward(s.nn, dg, dnn, n, true);
  rmsBackward(a.node, s.sn, dnn, D, dx, D, true);
  real* draw = ARENA.zeros(nn * a.H);
  af3::gridBiasK<<<blocks(nn * a.H), 256, 0, STREAM>>>(draw, dpl, n, a.H, false, true);
  real* dpn = ARENA.get(nn * a.P);
  a.bias.backward(s.pn, draw, dpn, nn);
  rmsBackward(a.pn, s.sp, dpn, a.P, dpair, a.P, true);
}

// ---------------------------------------------------------------- SwiGLU (the second half gated)
struct SwiGluR { Linear w1, out; int D = 0, I = 0; };
struct SgSave { const real* x; real* h; size_t rows; };
__global__ void swiSecondK(const real* h, real* g, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = t % I;
  real a = h[r * 2 * I + i], b = h[r * 2 * I + I + i];
  g[t] = b / (1 + exp(-b)) * a;
}
__global__ void swiSecondBwdK(const real* h, const real* dg, real* dh, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = t % I;
  real a = h[r * 2 * I + i], b = h[r * 2 * I + I + i], s = 1 / (1 + exp(-b));
  dh[r * 2 * I + i] = dg[t] * b * s;
  dh[r * 2 * I + I + i] = dg[t] * a * (s + b * s * (1 - s));
}
inline SgSave sgForward(const SwiGluR& w, const real* x, const real* mask, real* out, size_t rows) {
  SgSave s{x, ARENA.get(rows * 2 * w.I), rows};
  w.w1.forward(x, s.h, rows);
  Scratch scratch;
  real* g = ARENA.get(rows * w.I);
  swiSecondK<<<blocks(rows * w.I), 256, 0, STREAM>>>(s.h, g, rows, w.I);
  w.out.forward(g, out, rows);
  maskRowsK<<<blocks(rows * w.D), 256, 0, STREAM>>>(out, mask, rows, w.D);
  return s;
}
inline void sgBackward(SwiGluR& w, const SgSave& s, const real* mask, const real* dout, real* dx) {
  Scratch scratch;
  const size_t rows = s.rows;
  real* d = af3::cloneR(dout, rows * w.D);
  maskRowsK<<<blocks(rows * w.D), 256, 0, STREAM>>>(d, mask, rows, w.D);
  real *g = ARENA.get(rows * w.I), *dg = ARENA.get(rows * w.I), *dh = ARENA.get(rows * 2 * w.I);
  swiSecondK<<<blocks(rows * w.I), 256, 0, STREAM>>>(s.h, g, rows, w.I);
  w.out.backward(g, d, dg, rows);
  swiSecondBwdK<<<blocks(rows * w.I), 256, 0, STREAM>>>(s.h, dg, dh, rows, w.I);
  w.w1.backward(s.x, dh, dx, rows, true);
}

// ---------------------------------------------------------------- the token layer
struct TokenLayer { AdaLnR a1, a2; PbAttn att; AdaScale s1, s2; SwiGluR sg; };
struct TokenSave { AdaRSave a1, a2; PbSave att; AsSave s1, s2; SgSave sg; real *x1, *ada1, *attn, *ada2, *tr; };
// s [n, D] -> sOut (s + attention + transition)
inline TokenSave tokenForward(TokenLayer& l, const real* s, const real* pair, const real* c, const real* mask, real* sOut, int n) {
  const int D = l.att.D;
  TokenSave t{};
  t.ada1 = ARENA.get((size_t)n * D); t.a1 = adaRForward(l.a1, s, c, mask, t.ada1, n);
  t.attn = ARENA.get((size_t)n * D); t.att = pbForward(l.att, t.ada1, pair, mask, t.attn, n);
  t.x1 = af3::cloneR(s, (size_t)n * D);
  t.s1 = asForward(l.s1, t.attn, c, mask, t.x1, n);
  t.ada2 = ARENA.get((size_t)n * D); t.a2 = adaRForward(l.a2, t.x1, c, mask, t.ada2, n);
  t.tr = ARENA.get((size_t)n * D); t.sg = sgForward(l.sg, t.ada2, mask, t.tr, n);
  GK(cudaMemcpyAsync(sOut, t.x1, (size_t)n * D * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  t.s2 = asForward(l.s2, t.tr, c, mask, sOut, n);
  return t;
}
// dsOut -> ds (written), dpair (added), dc (added)
inline void tokenBackward(TokenLayer& l, const TokenSave& t, const real* mask, const real* dsOut, real* ds, real* dpair,
                          real* dc, int n) {
  Scratch scratch;
  const int D = l.att.D;
  real* dx1 = af3::cloneR(dsOut, (size_t)n * D);
  real *dtr = ARENA.get((size_t)n * D), *dada2 = ARENA.zeros((size_t)n * D);
  asBackward(l.s2, t.s2, mask, dsOut, dtr, dc);
  sgBackward(l.sg, t.sg, mask, dtr, dada2);
  adaRBackward(l.a2, t.a2, mask, dada2, dx1, dc);
  real *dattn = ARENA.get((size_t)n * D), *dada1 = ARENA.zeros((size_t)n * D);
  asBackward(l.s1, t.s1, mask, dx1, dattn, dc);
  pbBackward(l.att, t.att, mask, dattn, dada1, dpair);
  adaRBackward(l.a1, t.a1, mask, dada1, dx1, dc);
  GK(cudaMemcpyAsync(ds, dx1, (size_t)n * D * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// ---------------------------------------------------------------- the pair update
struct PairUpdate {
  LayerNorm lnIn;              // over the token width, with its bias
  Linear linX;                 // [D, 2P]
  TriMul out, in;              // four projections as two stacked (out, in) matrices; LayerNorms in and out
  LayerNorm trLn;
  Linear tr1, tr2;             // relu transition with biases
  int D = 0, P = 0, H = 0;
};
struct PuSave { LnSave ln, tl; real *xm, *xn, *pij, *z0, *z1, *z2, *zn, *h; TriMulSave to, ti; int n; };
// z0[ij] = valid ? z[ij] + pij[j, :P] pij[i, P:] : 0
__global__ void hadamardK(const real* z, const real* pij, const real* pm, real* out, int n, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * P) return;
  size_t ij = t / P; int c = t % P; int i = ij / n, j = ij % n;
  out[t] = pm[ij] == 0 ? 0 : z[t] + pij[(size_t)j * 2 * P + c] * pij[(size_t)i * 2 * P + P + c];
}
// dz (masked copy) and dpij: dpij[j, c] += sum_i d[ij] pij[i, P + c];  dpij[i, P + c] += sum_j d[ij] pij[j, c]
__global__ void hadamardBwdK(const real* d, const real* pij, const real* pm, real* dpij, int n, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * 2 * P) return;
  int r = t / (2 * P), c = t % (2 * P);
  real a = 0;
  if (c < P) { for (int i = 0; i < n; ++i) { size_t ij = (size_t)i * n + r; if (pm[ij] != 0) a += d[ij * P + c] * pij[(size_t)i * 2 * P + P + c]; } }
  else { int cc = c - P; for (int j = 0; j < n; ++j) { size_t ij = (size_t)r * n + j; if (pm[ij] != 0) a += d[ij * P + cc] * pij[(size_t)j * 2 * P + cc]; } }
  dpij[t] = a;
}
__global__ void reluBiasK(real* h, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) h[t] = fmax(h[t], (real)0);
}
inline PuSave puForward(PairUpdate& u, const real* s, const real* z, const real* mask, const real* pm, real* zOut, int n) {
  const int D = u.D, P = u.P; const size_t nn = (size_t)n * n;
  PuSave t{}; t.n = n;
  t.xm = af3::cloneR(s, (size_t)n * D);
  maskRowsK<<<blocks((size_t)n * D), 256, 0, STREAM>>>(t.xm, mask, n, D);
  t.xn = ARENA.get((size_t)n * D); t.ln = lnForward(u.lnIn, t.xm, t.xn, n);
  t.pij = ARENA.get((size_t)n * 2 * P); u.linX.forward(t.xn, t.pij, n);
  t.z0 = ARENA.get(nn * P);
  hadamardK<<<blocks(nn * P), 256, 0, STREAM>>>(z, t.pij, pm, t.z0, n, P);
  real* upd = ARENA.get(nn * P);
  t.z1 = af3::cloneR(t.z0, nn * P);
  t.to = triMulForward(u.out, t.z0, pm, upd, n);
  maskRowsK<<<blocks(nn * P), 256, 0, STREAM>>>(upd, pm, nn, P); add(t.z1, upd, nn * P);
  t.z2 = af3::cloneR(t.z1, nn * P);
  t.ti = triMulForward(u.in, t.z1, pm, upd, n);
  maskRowsK<<<blocks(nn * P), 256, 0, STREAM>>>(upd, pm, nn, P); add(t.z2, upd, nn * P);
  t.zn = ARENA.get(nn * P); t.tl = lnForward(u.trLn, t.z2, t.zn, nn);
  t.h = ARENA.get(nn * u.H); u.tr1.forward(t.zn, t.h, nn);
  {
    Scratch scratch;
    real* r = af3::cloneR(t.h, nn * u.H);
    reluBiasK<<<blocks(nn * u.H), 256, 0, STREAM>>>(r, nn * u.H);
    u.tr2.forward(r, upd, nn);
  }
  maskRowsK<<<blocks(nn * P), 256, 0, STREAM>>>(upd, pm, nn, P);
  GK(cudaMemcpyAsync(zOut, t.z2, nn * P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  add(zOut, upd, nn * P);
  return t;
}
// dzOut -> dz (written), ds (added)
inline void puBackward(PairUpdate& u, const PuSave& t, const real* mask, const real* pm, const real* dzOut, real* dz, real* ds) {
  Scratch scratch;
  const int n = t.n, D = u.D, P = u.P; const size_t nn = (size_t)n * n;
  real* d = af3::cloneR(dzOut, nn * P);                       // d z2
  real* du = af3::cloneR(dzOut, nn * P);
  maskRowsK<<<blocks(nn * P), 256, 0, STREAM>>>(du, pm, nn, P);
  {
    Scratch sc;
    real* r = af3::cloneR(t.h, nn * u.H);
    reluBiasK<<<blocks(nn * u.H), 256, 0, STREAM>>>(r, nn * u.H);
    real* dr = ARENA.get(nn * u.H);
    u.tr2.backward(r, du, dr, nn);
    reluBwdK<<<blocks(nn * u.H), 256, 0, STREAM>>>(dr, t.h, nn * u.H);
    real* dzn = ARENA.get(nn * P);
    u.tr1.backward(t.zn, dr, dzn, nn);
    lnBackward(u.trLn, t.tl, dzn, d, nn, true);
  }
  { Scratch sc; real* dm = af3::cloneR(d, nn * P); maskRowsK<<<blocks(nn * P), 256, 0, STREAM>>>(dm, pm, nn, P); triMulBackward(u.in, t.ti, pm, dm, d); }
  { Scratch sc; real* dm = af3::cloneR(d, nn * P); maskRowsK<<<blocks(nn * P), 256, 0, STREAM>>>(dm, pm, nn, P); triMulBackward(u.out, t.to, pm, dm, d); }
  // z0 = masked(z + hadamard)
  maskRowsK<<<blocks(nn * P), 256, 0, STREAM>>>(d, pm, nn, P);
  real* dpij = ARENA.get((size_t)n * 2 * P);
  hadamardBwdK<<<blocks((size_t)n * 2 * P), 256, 0, STREAM>>>(d, t.pij, pm, dpij, n, P);
  real* dxn = ARENA.get((size_t)n * D);
  u.linX.backward(t.xn, dpij, dxn, n);
  real* dxm = ARENA.get((size_t)n * D);
  lnBackward(u.lnIn, t.ln, dxn, dxm, n, false);
  maskRowsK<<<blocks((size_t)n * D), 256, 0, STREAM>>>(dxm, mask, n, D);
  add(ds, dxm, (size_t)n * D);
  GK(cudaMemcpyAsync(dz, d, nn * P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// every parameter, the checkpoint's names under `L` ("w/nn.transformer_layers.0") and `U` ("w/nn.pair_update_layers.0");
// the triangle's four projections arrive as "<a_p>|<b_p>" and "<a_g>|<b_g>", stacked by the caller
using ParamFn = std::function<void(const std::string&, Param&, size_t)>;
inline void tokenParams(TokenLayer& l, const std::string& L, int D, int C, int P, int H, const ParamFn& f) {
  auto lin = [&](const std::string& n, Linear& x, int K, int N, bool bias) {
    x.K = K; x.N = N; x.trans = true; x.hasBias = bias; f(n + ".weight", x.w, (size_t)K * N); if (bias) f(n + ".bias", x.b, N);
  };
  auto rms = [&](const std::string& n, Rms& r, int C_) { r.C = C_; f(n + ".weight", r.w, C_); };
  for (int k = 0; k < 2; ++k) {
    AdaLnR& a = k ? l.a2 : l.a1;
    const std::string A = L + (k ? ".transition.adaln" : ".mhba.adaln");
    a.D = D; a.xn.C = D; a.xn.hasW = false;
    rms(A + ".norm_cond", a.cn, C);
    lin(A + ".to_gamma.0", a.gamma, C, D, true);
    lin(A + ".to_beta", a.beta, C, D, false);
    lin(L + (k ? ".transition.scale_output.to_adaln_zero_gamma.0" : ".mhba.scale_output.to_adaln_zero_gamma.0"),
        (k ? l.s2 : l.s1).z, C, D, true);
  }
  const std::string M_ = L + ".mhba.mha";
  l.att.D = D; l.att.P = P; l.att.H = H;
  rms(M_ + ".node_norm", l.att.node, D);
  lin(M_ + ".to_qkv", l.att.qkv, D, 3 * D, true);
  rms(M_ + ".q_layer_norm", l.att.qn, D);
  rms(M_ + ".k_layer_norm", l.att.kn, D);
  lin(M_ + ".to_g", l.att.g, D, D, true);
  rms(M_ + ".pair_norm", l.att.pn, P);
  lin(M_ + ".to_bias", l.att.bias, P, H, false);
  lin(M_ + ".to_out_node", l.att.out, D, D, true);
  l.sg.D = D; l.sg.I = 4 * D;
  lin(L + ".transition.transition.swish_linear.0", l.sg.w1, D, 8 * D, false);
  lin(L + ".transition.transition.linear_out", l.sg.out, 4 * D, D, false);
}
inline void pairParams(PairUpdate& u, const std::string& U, int D, int P, int TRI, int Hd, const ParamFn& f) {
  auto lin = [&](const std::string& n, Linear& x, int K, int N, bool bias) {
    x.K = K; x.N = N; x.trans = true; x.hasBias = bias; f(n + ".weight", x.w, (size_t)K * N); if (bias) f(n + ".bias", x.b, N);
  };
  u.D = D; u.P = P; u.H = Hd;
  u.lnIn.C = D; f(U + ".layer_norm_in.weight", u.lnIn.g, D); f(U + ".layer_norm_in.bias", u.lnIn.b, D);
  lin(U + ".linear_x", u.linX, D, 2 * P, false);
  for (int k = 0; k < 2; ++k) {
    TriMul& t = k ? u.in : u.out;
    const std::string T = U + (k ? ".tri_mult_in" : ".tri_mult_out");
    t.C = P; t.I = TRI; t.outgoing = !k; t.swapAB = k; t.interleaved = false; t.masked = true;
    t.lnIn.C = P; f(T + ".layer_norm_in.weight", t.lnIn.g, P);
    t.lnCenter.C = TRI; f(T + ".layer_norm_out.weight", t.lnCenter.g, TRI);
    lin(T + ".linear_a_p|" + T + ".linear_b_p", t.proj, P, 2 * TRI, false);
    lin(T + ".linear_a_g|" + T + ".linear_b_g", t.gate, P, 2 * TRI, false);
    lin(T + ".linear_z", t.out, TRI, P, false);
    lin(T + ".linear_g", t.gating, P, P, false);
  }
  u.trLn.C = P; f(U + ".transition_out.layer_norm.weight", u.trLn.g, P); f(U + ".transition_out.layer_norm.bias", u.trLn.b, P);
  lin(U + ".transition_out.linear_1", u.tr1, P, Hd, true);
  lin(U + ".transition_out.linear_2", u.tr2, Hd, P, true);
}

}  // namespace rfd4p
}  // namespace grad
