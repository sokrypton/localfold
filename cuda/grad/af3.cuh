// The AlphaFold 3 lineage's layers for training, forward with saves and backward into every parameter, under the
// CUDA port's own weight names (cuda/featurise/af3_weights.h's walk: the bundle read as published). The
// specification is cpu/af3/ (trunk/pairformer.js and its neighbours); the CUDA port, held to af3-any-model's oracles,
// is what the forward is compared with (test_af3.cu).
//
//   pairformer block:  pair += triMul(out); pair += triMul(in); pair += grid(row); pair += grid(column);
//                      pair += transition; single += singleAttention(single, pairLogits(pair)); single += transition
//
// Dialects carried here (shared/af3/dialect.js): swapTransposedBias (the column direction's pair bias transposed:
// openfold3, opendde, boltz2, protenix2), triangleMulDivideByLength (rosettafold3), the grid attention's gate and
// output biases (rosettafold3: present when the bundle has them), and an extra pair-logit bias broadcast over heads
// (OpenDDE's refiner).
#pragma once
#include "pair.cuh"

namespace grad {
namespace af3 {

struct Dialect {
  bool swapTransposedBias = false, divideByLength = false;
};

// AF3's gated transition: LN, transition1 [C, 2F] -> swish(first half) * second half, transition2 [F, C]; no biases
struct GatedTransition {
  LayerNorm ln;
  Linear t1, t2;
  int C = 0, F = 0;
};
struct GtSave { LnSave ln; real *xn, *h; size_t rows; void *xb = nullptr, *hb = nullptr, *gb = nullptr; };   // (gb: swish(h1) h2, kept)
// BF16: the normalised input and the hidden layer kept in bf16, every GEMM on bf16 operands
#ifndef GRAD_F64
inline GtSave gtForwardBf(const GatedTransition& t, const real* x, real* out, size_t rows, real beta) {
  GtSave s; s.rows = rows;
  bf* xb = bfGet(rows * t.C);
  s.ln = lnForward(t.ln, x, xb, rows);
  bf* hb = bfGet(rows * 2 * t.F);
  linearFwdBf(t.t1, xb, hb, rows);
  s.xb = xb; s.hb = hb;
  bf* g = bfGet(rows * t.F);   // (kept for t2's weight gradient: recomputing it was 0.25 ms of AF3's backward)
  swigluBfK<<<blocks(rows * t.F), 256, 0, STREAM>>>(hb, g, rows, t.F);
  s.gb = g;
  linearFwdBf(t.t2, g, out, rows, beta);
  return s;
}
// (PD: the running gradient's type - real, or bf16 where BF16 mode keeps the pair track's gradient so)
template <class PD>
inline void gtBackwardBf(GatedTransition& t, const GtSave& s, const PD* dout, PD* dx) {
  Scratch scratch;
  const size_t rows = s.rows;
  const bf *xb = static_cast<const bf*>(s.xb), *hb = static_cast<const bf*>(s.hb);
  const bf* g = static_cast<const bf*>(s.gb);
  const bf* db;
  if constexpr (sizeof(PD) == 2) db = dout; else db = toBf(dout, rows * t.C);
  bf* dg = bfGet(rows * t.F);
  linearBwdBf(t.t2, g, db, dg, rows);
  bf* dh = bfGet(rows * 2 * t.F);
  swigluBwdBfK<<<blocks(rows * t.F), 256, 0, STREAM>>>(hb, dg, dh, rows, t.F);
  bf* dxn = bfGet(rows * t.C);   // (only the LayerNorm's backward reads it)
  linearBwdBf(t.t1, xb, dh, dxn, rows);
  lnBackward(t.ln, s.ln, dxn, dx, rows, true);
}
#endif
inline GtSave gtForward(const GatedTransition& t, const real* x, real* out, size_t rows, real beta = 0) {
#ifndef GRAD_F64
  if (BF16) return gtForwardBf(t, x, out, rows, beta);
#endif
  GtSave s; s.rows = rows;
  s.xn = ARENA.get(rows * t.C);
  s.ln = lnForward(t.ln, x, s.xn, rows);
  s.h = ARENA.get(rows * 2 * t.F);
  t.t1.forward(s.xn, s.h, rows);
  Scratch scratch;
  real* g = ARENA.get(rows * t.F);
  swigluK<<<blocks(rows * t.F), 256, 0, STREAM>>>(s.h, g, rows, t.F);
  t.t2.forward(g, out, rows, beta);
  return s;
}
// dout -> dx (added)
#ifndef GRAD_F64
inline void gtBackward(GatedTransition& t, const GtSave& s, const bf* dout, bf* dx) { gtBackwardBf(t, s, dout, dx); }
#endif
inline void gtBackward(GatedTransition& t, const GtSave& s, const real* dout, real* dx) {
#ifndef GRAD_F64
  if (s.xb) { gtBackwardBf(t, s, dout, dx); return; }
#endif
  Scratch scratch;
  size_t rows = s.rows;
  real* g = ARENA.get(rows * t.F);
  swigluK<<<blocks(rows * t.F), 256, 0, STREAM>>>(s.h, g, rows, t.F);
  real* dg = ARENA.get(rows * t.F);
  t.t2.backward(g, dout, dg, rows);
  real* dh = ARENA.get(rows * 2 * t.F);
  swigluBwdK<<<blocks(rows * t.F), 256, 0, STREAM>>>(s.h, dg, dh, rows, t.F);
  real* dxn = ARENA.get(rows * t.C);
  t.t1.backward(s.xn, dh, dxn, rows);
  lnBackward(t.ln, s.ln, dxn, dx, rows, true);
}

__global__ void addConstK(real* x, real a, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] += a;
}

// ---------------------------------------------------------------- grid ("triangle") self-attention
struct GridAttn {
  LayerNorm ln;
  Linear bias, q, k, v, gate, out;     // bias [C, H]; q, k, gate transposed [HD, C]; v [C, HD]; out [HD, C]
  int C = 0, H = 0, D = 0;
  bool transpose = false;
  bool untransposedResidual = false;   // chai-1's parallel block: the column pass's update added as it comes out
};
struct GridSave { LnSave ln; real *xn, *act, *nb, *qkvg, *o; int n; real* lse = nullptr; bool bf = false; real* W = nullptr; void* og = nullptr; };   // (bf: xn, act, qkvg [nn, 4 HD] and W [C, 4 HD] in bf16)
// rawBias [nn, H] -> nb [H, nn], the column direction's source transposed when swap
__global__ void gridBiasK(const real* raw, real* nb, int n, int H, bool swap, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t nn = (size_t)n * n;
  if (t >= nn * H) return;
  int h = t / nn; size_t ij = t % nn;
  size_t i = ij / n, j = ij % n, src = swap ? j * n + i : ij;
  if (back) atomicAdd(const_cast<real*>(&raw[src * H + h]), nb[t]);   // (raw is the gradient here)
  else nb[t] = raw[src * H + h];
}
inline Sdpa gridSdpa(const GridAttn& g, int n, const real* keyMask, const real* nb, real* dnb = nullptr) {
  Sdpa a;
  a.B = n; a.H = g.H; a.Q = n; a.K = n; a.D = g.D; a.scale = 1 / sqrt((real)g.D);
  a.keyMask = keyMask;
  a.bias[0] = Bias{nb, 0, (long long)n * n, n, dnb};
  return a;
}
#ifndef GRAD_F64
// q, k, v, gate's weights packed side by side as one [C, 4 HD] bf16 matrix (each role [HD, C] when its Linear is
// transposed, [C, HD] otherwise); back: dW [C, 4 HD] (float) added into each role's gradient in its own layout
__global__ void packQkvgK(bf* W, float* dW, const float* w0, const float* w1, const float* w2, const float* w3,
                          float* g0, float* g1, float* g2, float* g3, bool t0, bool t1, bool t2, bool t3, int C,
                          int HD, bool back) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)C * 4 * HD) return;
  const int col = (int)(i % (4 * HD)), c = (int)(i / (4 * HD)), r = col / HD, j = col % HD;
  const bool tr = r == 0 ? t0 : r == 1 ? t1 : r == 2 ? t2 : t3;
  const size_t at = tr ? (size_t)j * C + c : (size_t)c * HD + j;
  if (back) atomicAdd((r == 0 ? g0 : r == 1 ? g1 : r == 2 ? g2 : g3) + at, dW[i]);
  else W[i] = __float2bfloat16((r == 0 ? w0 : r == 1 ? w1 : r == 2 ? w2 : w3)[at]);
}
inline void packQkvg(const GridAttn& g, bf* W, float* dW, bool back) {
  const int HD = g.H * g.D;
  const Linear* L[4] = {&g.q, &g.k, &g.v, &g.gate};
  packQkvgK<<<blocks((size_t)g.C * 4 * HD), 256, 0, STREAM>>>(W, dW, L[0]->w.w, L[1]->w.w, L[2]->w.w, L[3]->w.w,
      L[0]->w.g, L[1]->w.g, L[2]->w.g, L[3]->w.g, L[0]->trans, L[1]->trans, L[2]->trans, L[3]->trans, g.C, HD, back);
}
// BF16: the normalised pair and q, k, v, gate kept in bf16, the four projections one bf16 GEMM into [nn, 4 HD]
// (bindcraft2's packed Q|K|V|G), the attention reading q, k, v there in place
inline GridSave gridForwardBf(const GridAttn& g, const Dialect& d, const real* pair, const real* mask,
                              const real* maskT, real* out, int n) {
  const size_t nn = (size_t)n * n;
  const int HD = g.H * g.D;
  GridSave s; s.n = n; s.bf = true;
  bf* xn = bfGet(nn * g.C);
  s.ln = lnForward(g.ln, pair, xn, nn);
  s.nb = ARENA.get(nn * g.H);
  {
    Scratch scratch;
    real* raw = ARENA.get(nn * g.H);
    linearFwdBf(g.bias, xn, raw, nn);
    gridBiasK<<<blocks(nn * g.H), 256, 0, STREAM>>>(raw, s.nb, n, g.H, g.transpose && d.swapTransposedBias, false);
  }
  bf* act = xn;
  if (g.transpose) { act = bfGet(nn * g.C); swap01(xn, act, n, n, g.C); }
  const int P = 4 * HD;
  bf* W = bfGet((size_t)g.C * P);
  packQkvg(g, W, nullptr, false);
  bf* qkvg = bfGet(nn * P);
  mmBf(act, W, qkvg, (int)nn, g.C, P);
  const Linear* roles[4] = {&g.q, &g.k, &g.v, &g.gate};
  for (int r = 0; r < 4; ++r)
    if (roles[r]->hasBias) biasAddLdK<<<blocks(nn * HD), 256, 0, STREAM>>>(qkvg + r * HD, roles[r]->b.w, nn, HD, P);
  bf *q = qkvg, *k = q + HD, *v = k + HD, *gt = v + HD;
  gateSigLdBfK<<<blocks(nn * HD), 256, 0, STREAM>>>(gt, nn, HD, P);
  bf* ob = bfGet(nn * HD);   // the attention's output in bf16
  s.o = reinterpret_cast<real*>(ob);
  { const Sdpa sd_ = gridSdpa(g, n, g.transpose ? maskT : mask, s.nb); s.lse = ARENA.get((size_t)2 * sd_.B * sd_.H * sd_.Q); sdpaForwardBf(sd_, q, P, k, P, v, P, ob, HD, s.lse); }
  s.xn = reinterpret_cast<real*>(xn); s.act = reinterpret_cast<real*>(act); s.qkvg = reinterpret_cast<real*>(qkvg);
  s.W = reinterpret_cast<real*>(W);
  bf* og = bfGet(nn * HD);   // (kept for the output projection's weight gradient)
  gateMulBfK<<<blocks(nn * HD), 256, 0, STREAM>>>(og, ob, gt, nn, HD, P);
  s.og = og;
  Scratch scratch;
  if (!g.transpose || g.untransposedResidual) { linearFwdBf(g.out, og, out, nn, 1.f); return s; }
  real* t = ARENA.get(nn * g.C);
  linearFwdBf(g.out, og, t, nn);
  swap01(t, out, n, n, g.C, true);
  return s;
}
template <class PD>
inline void gridBackwardBf(GridAttn& g, const Dialect& d, const GridSave& s, const real* mask, const real* maskT,
                           const PD* dout, PD* dpair) {
  Scratch scratch;
  const int n = s.n, HD = g.H * g.D;
  const size_t nn = (size_t)n * n;
  const bf *xn = reinterpret_cast<const bf*>(s.xn), *act = reinterpret_cast<const bf*>(s.act),
           *qkvg = reinterpret_cast<const bf*>(s.qkvg);
  const int P = 4 * HD;
  const bf *q = qkvg, *k = q + HD, *v = k + HD, *gt = v + HD, *W = reinterpret_cast<const bf*>(s.W);
  const PD* dy = dout;
  if (g.transpose && !g.untransposedResidual) {
    PD* t = reinterpret_cast<PD*>(sizeof(PD) == 2 ? (real*)bfGet(nn * g.C) : ARENA.get(nn * g.C));
    swap01(dout, t, n, n, g.C); dy = t;
  }
  const bf* dyb;
  if constexpr (sizeof(PD) == 2) dyb = dy; else dyb = toBf(dy, nn * g.C);
  const bf* og = static_cast<const bf*>(s.og);
  bf* dog = bfGet(nn * HD);
  linearBwdBf(g.out, og, dyb, dog, nn);
  const bf* ob = reinterpret_cast<const bf*>(s.o);
  // d q, k, v, gate straight into one bf16 [nn, 4 HD]: dq through float (its atomics'), dk, dv from the attention,
  // the gate's from its backward, which also forms the softmax backward's row term
  bf* dqkvg = bfGet(nn * P);
  bf* dO = bfGet(nn * HD);   // (bf16: the attention reads it in place, and its row term is formed here)
  real* Di = ARENA.get(nn * g.H);
  gateBwdLdDiBfK<<<blocks(nn * g.H * 32), 256, 0, STREAM>>>(dog, ob, gt, dO, dqkvg + 3 * HD, Di, nn, g.H, g.D, P, n);
  real* dq = ARENA.get(nn * HD);
  real* dnb = ARENA.zeros(nn * g.H);
  sdpaBackwardBf(gridSdpa(g, n, g.transpose ? maskT : mask, s.nb, dnb), q, P, k, P, v, P, dO, HD, dq, HD,
                 dqkvg + HD, P, dqkvg + 2 * HD, P, nullptr, HD, s.lse, Di, dqkvg, P);
  const Linear* roles[4] = {&g.q, &g.k, &g.v, &g.gate};
  for (int r = 0; r < 4; ++r)
    if (roles[r]->hasBias) colSumBf(roles[r]->b.g, dqkvg + r * HD, nn, HD, P);
  {
    Scratch sc;
    real* dW = ARENA.get((size_t)g.C * P);
    mmATBf(act, dqkvg, dW, (int)nn, g.C, P, 0.f);
    packQkvg(g, nullptr, dW, true);
  }
  // dx in bf16 (its transpose too): the LayerNorm's backward is its only reader
  bf* dact = bfGet(nn * g.C);
  mmBTBf(dqkvg, W, dact, (int)nn, g.C, P);
  bf* dxn = dact;
  if (g.transpose) { dxn = bfGet(nn * g.C); swap01(dact, dxn, n, n, g.C); }
  real* draw = ARENA.zeros(nn * g.H);
  gridBiasK<<<blocks(nn * g.H), 256, 0, STREAM>>>(draw, dnb, n, g.H, g.transpose && d.swapTransposedBias, true);
  linearBwdBf(g.bias, xn, toBf(draw, nn * g.H), dxn, nn, true);
  lnBackward(g.ln, s.ln, dxn, dpair, nn, true);
}
#endif
// pair [n, n, C] -> out (+= when beta 1), mask / maskT [n, n]
inline GridSave gridForward(const GridAttn& g, const Dialect& d, const real* pair, const real* mask, const real* maskT,
                            real* out, int n) {
#ifndef GRAD_F64
  if (BF16) return gridForwardBf(g, d, pair, mask, maskT, out, n);
#endif
  size_t nn = (size_t)n * n;
  const int HD = g.H * g.D;
  GridSave s; s.n = n;
  s.xn = ARENA.get(nn * g.C);
  s.ln = lnForward(g.ln, pair, s.xn, nn);
  s.nb = ARENA.get(nn * g.H);
  {
    Scratch scratch;
    real* raw = ARENA.get(nn * g.H);
    g.bias.forward(s.xn, raw, nn);
    gridBiasK<<<blocks(nn * g.H), 256, 0, STREAM>>>(raw, s.nb, n, g.H, g.transpose && d.swapTransposedBias, false);
  }
  s.act = s.xn;
  if (g.transpose) { s.act = ARENA.get(nn * g.C); swap01(s.xn, s.act, n, n, g.C); }
  s.qkvg = ARENA.get(nn * 4 * HD);
  real *q = s.qkvg, *k = q + nn * HD, *v = k + nn * HD, *gt = v + nn * HD;
  g.q.forward(s.act, q, nn); g.k.forward(s.act, k, nn); g.v.forward(s.act, v, nn); g.gate.forward(s.act, gt, nn);
  sigmoidK<<<blocks(nn * HD), 256, 0, STREAM>>>(gt, nn * HD);
  s.o = ARENA.get(nn * HD);
  { const Sdpa sd_ = gridSdpa(g, n, g.transpose ? maskT : mask, s.nb); s.lse = ARENA.get((size_t)2 * sd_.B * sd_.H * sd_.Q); sdpaForward(sd_, q, HD, k, HD, v, HD, s.o, HD, s.lse); }
  Scratch scratch;
  real* og = ARENA.get(nn * HD);
  mulK<<<blocks(nn * HD), 256, 0, STREAM>>>(og, s.o, gt, nn * HD);
  if (!g.transpose || g.untransposedResidual) { g.out.forward(og, out, nn, 1); return s; }
  real* t = ARENA.get(nn * g.C);
  g.out.forward(og, t, nn);
  swap01(t, out, n, n, g.C, true);
  return s;
}
#ifndef GRAD_F64
inline void gridBackward(GridAttn& g, const Dialect& d, const GridSave& s, const real* mask, const real* maskT,
                         const bf* dout, bf* dpair) {
  gridBackwardBf(g, d, s, mask, maskT, dout, dpair);
}
#endif
inline void gridBackward(GridAttn& g, const Dialect& d, const GridSave& s, const real* mask, const real* maskT,
                         const real* dout, real* dpair) {
#ifndef GRAD_F64
  if (s.bf) { gridBackwardBf(g, d, s, mask, maskT, dout, dpair); return; }
#endif
  Scratch scratch;
  const int n = s.n, HD = g.H * g.D;
  size_t nn = (size_t)n * n;
  const real* dy = dout;
  if (g.transpose && !g.untransposedResidual) { real* t = ARENA.get(nn * g.C); swap01(dout, t, n, n, g.C); dy = t; }
  real *q = s.qkvg, *k = q + nn * HD, *v = k + nn * HD, *gt = v + nn * HD;
  real* og = ARENA.get(nn * HD);
  mulK<<<blocks(nn * HD), 256, 0, STREAM>>>(og, s.o, gt, nn * HD);
  real* dog = ARENA.get(nn * HD);
  g.out.backward(og, dy, dog, nn);
  real* dqkvg = ARENA.get(nn * 4 * HD);
  real *dq = dqkvg, *dk = dq + nn * HD, *dv = dk + nn * HD, *dg = dv + nn * HD;
  real* dO = ARENA.get(nn * HD);
  gateBwdK<<<blocks(nn * HD), 256, 0, STREAM>>>(dog, s.o, gt, dO, dg, nn * HD);
  real* dnb = ARENA.zeros(nn * g.H);
  sdpaBackward(gridSdpa(g, n, g.transpose ? maskT : mask, s.nb, dnb), q, HD, k, HD, v, HD, dO, HD, dq, HD, dk, HD, dv, HD, false, false, false, s.o, HD, s.lse);
  real* dact = ARENA.get(nn * g.C);
  g.q.backward(s.act, dq, dact, nn);
  g.k.backward(s.act, dk, dact, nn, true);
  g.v.backward(s.act, dv, dact, nn, true);
  g.gate.backward(s.act, dg, dact, nn, true);
  real* dxn = dact;
  if (g.transpose) { dxn = ARENA.get(nn * g.C); swap01(dact, dxn, n, n, g.C); }
  real* draw = ARENA.zeros(nn * g.H);
  gridBiasK<<<blocks(nn * g.H), 256, 0, STREAM>>>(draw, dnb, n, g.H, g.transpose && d.swapTransposedBias, true);
  g.bias.backward(s.xn, draw, dxn, nn, true);
  lnBackward(g.ln, s.ln, dxn, dpair, nn, true);
}

// ---------------------------------------------------------------- the single track's attention, pair-biased
struct SingleAttn {
  LayerNorm ln;
  Linear q, k, v, gate, out;           // all [C, HD] / [HD, C]; q has the block's one bias
  LayerNorm pairLn;                    // singlePairLogitsNorm
  Linear pairProj;                     // singlePairLogitsProjection [Cz, H]
  int C = 0, Cz = 0, H = 0, D = 0;
  real gateConst = 0;                  // chai-1: sigmoid(g + 1), its gating linear's bias a constant
};
struct SingleSave { LnSave ln, pln; real *xn, *pn, *pl, *qkvg, *o; int n; real* lse = nullptr; };
inline Sdpa singleSdpa(const SingleAttn& a, int n, const real* seqMask, const real* pl, real* dpl = nullptr) {
  Sdpa s;
  s.B = 1; s.H = a.H; s.Q = n; s.K = n; s.D = a.D; s.scale = 1 / sqrt((real)a.D);
  s.keyMask = seqMask;
  s.bias[0] = Bias{pl, 0, (long long)n * n, n, dpl};
  return s;
}
// single [n, Cs] += attention; pair [n, n, Cz] read for the logits; extra [n, n] (may be null) added to every head
inline SingleSave singleForward(const SingleAttn& a, const real* single, const real* pair, const real* seqMask,
                                const real* extra, real* out, int n) {
  size_t nn = (size_t)n * n;
  const int HD = a.H * a.D;
  SingleSave s; s.n = n;
  s.pn = ARENA.get(nn * a.Cz);
  s.pln = lnForward(a.pairLn, pair, s.pn, nn);
  s.pl = ARENA.get(nn * a.H);
  {
    Scratch scratch;
    real* raw = ARENA.get(nn * a.H);
    a.pairProj.forward(s.pn, raw, nn);
    gridBiasK<<<blocks(nn * a.H), 256, 0, STREAM>>>(raw, s.pl, n, a.H, false, false);
    if (extra) for (int h = 0; h < a.H; ++h) add(s.pl + h * nn, extra, nn);
  }
  s.xn = ARENA.get((size_t)n * a.C);
  s.ln = lnForward(a.ln, single, s.xn, n);
  s.qkvg = ARENA.get((size_t)n * 4 * HD);
  real *q = s.qkvg, *k = q + (size_t)n * HD, *v = k + (size_t)n * HD, *g = v + (size_t)n * HD;
  a.q.forward(s.xn, q, n); a.k.forward(s.xn, k, n); a.v.forward(s.xn, v, n); a.gate.forward(s.xn, g, n);
  if (a.gateConst != 0) addConstK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(g, a.gateConst, (size_t)n * HD);
  sigmoidK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(g, (size_t)n * HD);
  s.o = ARENA.get((size_t)n * HD);
  { const Sdpa sd_ = singleSdpa(a, n, seqMask, s.pl); s.lse = ARENA.get((size_t)2 * sd_.B * sd_.H * sd_.Q); sdpaForward(sd_, q, HD, k, HD, v, HD, s.o, HD, s.lse); }
  Scratch scratch;
  real* og = ARENA.get((size_t)n * HD);
  mulK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(og, s.o, g, (size_t)n * HD);
  a.out.forward(og, out, n, 1);
  return s;
}
// dout [n, Cs] -> dsingle (added), dpair (added); dextra [n, n] (added, may be null)
template <class PD>   // (PD: the pair gradient's type)
inline void singleBackward(SingleAttn& a, const SingleSave& s, const real* seqMask, const real* dout, real* dsingle,
                           PD* dpair, real* dextra) {
  Scratch scratch;
  const int n = s.n, HD = a.H * a.D;
  size_t nn = (size_t)n * n;
  real *q = s.qkvg, *k = q + (size_t)n * HD, *v = k + (size_t)n * HD, *g = v + (size_t)n * HD;
  real* og = ARENA.get((size_t)n * HD);
  mulK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(og, s.o, g, (size_t)n * HD);
  real* dog = ARENA.get((size_t)n * HD);
  a.out.backward(og, dout, dog, n);
  real* dqkvg = ARENA.get((size_t)n * 4 * HD);
  real *dq = dqkvg, *dk = dq + (size_t)n * HD, *dv = dk + (size_t)n * HD, *dg = dv + (size_t)n * HD;
  real* dO = ARENA.get((size_t)n * HD);
  gateBwdK<<<blocks((size_t)n * HD), 256, 0, STREAM>>>(dog, s.o, g, dO, dg, (size_t)n * HD);
  real* dpl = ARENA.zeros(nn * a.H);
  sdpaBackward(singleSdpa(a, n, seqMask, s.pl, dpl), q, HD, k, HD, v, HD, dO, HD, dq, HD, dk, HD, dv, HD, false, false, false, s.o, HD, s.lse);
  real* dxn = ARENA.get((size_t)n * a.C);
  a.q.backward(s.xn, dq, dxn, n);
  a.k.backward(s.xn, dk, dxn, n, true);
  a.v.backward(s.xn, dv, dxn, n, true);
  a.gate.backward(s.xn, dg, dxn, n, true);
  lnBackward(a.ln, s.ln, dxn, dsingle, n, true);
  if (dextra) for (int h = 0; h < a.H; ++h) add(dextra, dpl + h * nn, nn);
  real* draw = ARENA.zeros(nn * a.H);
  gridBiasK<<<blocks(nn * a.H), 256, 0, STREAM>>>(draw, dpl, n, a.H, false, true);
  real* dpn = ARENA.get(nn * a.Cz);
  a.pairProj.backward(s.pn, draw, dpn, nn);
  lnBackward(a.pairLn, s.pln, dpn, dpair, nn, true);
}

// ---------------------------------------------------------------- the pairformer block
struct Pairformer {
  TriMul triOut, triIn;
  GridAttn grid1, grid2;
  GatedTransition pairTrans, singleTrans;
  SingleAttn single;
  bool hasSingle = true;              // the template stack and the MSA module's pair half have no single track
  bool hasGrid = true;                // Mol-1's PairMixer blocks carry no triangle attention
  bool parallel = false;              // chai-1: every update reads the block's input, and they are summed
  // training's dropout (OpenFold3's pair_dropout, AF3 Algorithm 17): the two triangle multiplications' and the
  // starting attention's updates rowwise, the ending attention's columnwise, at dropRate (0: off, inference). The
  // masks come from a counter hash of (dropSeed, module, position), so a recomputed forward and the backward regenerate
  // them and none is stored.
  float dropRate = 0;
  unsigned long long dropSeed = 0;
};
// the dropout itself (af2.cuh's dropoutK, the same draws): an update [A, B, C] in place, keep / (1 - rate); perRow, the
// mask shared over A (DropoutRowwise, batch_dim -3), else over B (columnwise)
__host__ __device__ __forceinline__ unsigned long long dropMix(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
__device__ __forceinline__ float dropKeep(size_t t, int B, int C, bool perRow, float rate, unsigned long long seed) {
  const int c = (int)(t % C);
  const size_t rr = t / C;
  const int a = (int)(rr / B), b = (int)(rr % B);
  const unsigned long long key = perRow ? (unsigned long long)b * C + c : ((unsigned long long)a * C + c) | (1ull << 62);
  const float u = (float)(dropMix(seed ^ dropMix(key)) >> 40) * (1.f / 16777216.f);
  return u < rate ? 0.f : 1.f;
}
__global__ void dropK(real* x, int A, int B, int C, bool perRow, float rate, unsigned long long seed) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * B * C) return;
  x[t] = dropKeep(t, B, C, perRow, rate, seed) == 0.f ? (real)0 : x[t] / (1 - (real)rate);
}
#ifndef GRAD_F64
// (the bf16 backward's: y = mask(x) / (1 - rate) from a bf16 running gradient, into a fresh bf16 copy)
__global__ void dropCopyBfK(const bf* x, bf* y, int A, int B, int C, bool perRow, float rate, unsigned long long seed) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * B * C) return;
  y[t] = __float2bfloat16(dropKeep(t, B, C, perRow, rate, seed) * __bfloat162float(x[t]) / (1 - rate));
}
#endif
enum { DROP3_TRI_OUT = 1, DROP3_TRI_IN, DROP3_START, DROP3_END, DROP3_MSA };
inline void drop3(real* x, int A, int B, int C, bool perRow, float rate, unsigned long long seed, int module) {
  if (rate <= 0) return;
  dropK<<<blocks((size_t)A * B * C), 256, 0, STREAM>>>(x, A, B, C, perRow, rate, dropMix(seed * 8 + module));
}
#ifndef GRAD_F64
inline bf* drop3Bf(const bf* x, int A, int B, int C, bool perRow, float rate, unsigned long long seed, int module) {
  bf* y = bfGet((size_t)A * B * C);
  dropCopyBfK<<<blocks((size_t)A * B * C), 256, 0, STREAM>>>(x, y, A, B, C, perRow, rate, dropMix(seed * 8 + module));
  return y;
}
#endif
struct PairformerSave {
  const real *pair0, *single0;
  real *p1, *p2, *p3, *p4, *pair5, *s1;
  TriMulSave tOut, tIn;
  GridSave g1, g2;
  GtSave pt, st;
  SingleSave sa;
  int n;
};

// every parameter under `pre` (e.g. "trunk.pairformerBlocks.0"), the port's names, sized from the channel counts
inline void forEachParam(Pairformer& b, const std::string& pre, int Cz, int Cs, int Hp, int Dp, int Hs, int Ds,
                         int pairFactor, int singleFactor, bool gridBiases, bool hasSingle,
                         const std::function<void(const std::string&, Param&, size_t)>& f, bool hasGrid = true,
                         bool parallel = false, const std::string& sep = ".") {
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(n + "Scale", l.g, C); f(n + "Offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N, bool trans = false, bool bias = false) {
    l.K = K; l.N = N; l.trans = trans; l.hasBias = bias; l.w.fanIn = K; f(n, l.w, (size_t)K * N);
  };
  b.hasSingle = hasSingle; b.hasGrid = hasGrid; b.parallel = parallel;
  for (auto* t : {&b.triOut, &b.triIn}) {
    std::string n = pre + (t == &b.triOut ? sep + "triangleMultiplicationOutgoing" : sep + "triangleMultiplicationIncoming");
    t->C = Cz; t->I = Cz; t->outgoing = t == &b.triOut; t->interleaved = true;
    ln(n + sep + "leftNormInput", t->lnIn, Cz);
    ln(n + sep + "centerNorm", t->lnCenter, Cz);
    lin(n + sep + "projection", t->proj, Cz, 2 * Cz);
    lin(n + sep + "gate", t->gate, Cz, 2 * Cz);
    lin(n + sep + "outputProjection", t->out, Cz, Cz);
    lin(n + sep + "gatingLinear", t->gating, Cz, Cz);
  }
  for (auto* g : {&b.grid1, &b.grid2}) {
    if (!hasGrid) break;
    std::string n = pre + (g == &b.grid1 ? sep + "pairAttention1" : sep + "pairAttention2");
    g->C = Cz; g->H = Hp; g->D = Dp; g->transpose = g == &b.grid2; g->untransposedResidual = parallel && g == &b.grid2;
    const int HD = Hp * Dp;
    ln(n + sep + "actNorm", g->ln, Cz);
    lin(n + sep + "pairBiasProjection", g->bias, Cz, Hp);
    lin(n + sep + "qProjection", g->q, Cz, HD, true);
    lin(n + sep + "kProjection", g->k, Cz, HD, true);
    lin(n + sep + "vProjection", g->v, Cz, HD);
    lin(n + sep + "gatingQuery", g->gate, Cz, HD, true);
    lin(n + sep + "outputProjection", g->out, HD, Cz);
    if (gridBiases) {
      g->gate.hasBias = true; f(n + sep + "gatingQueryBias", g->gate.b, HD);
      g->out.hasBias = true; f(n + sep + "outputProjectionBias", g->out.b, Cz);
    }
  }
  b.pairTrans.C = Cz; b.pairTrans.F = Cz * pairFactor;
  ln(pre + sep + "pairTransition" + sep + "inputLayerNorm", b.pairTrans.ln, Cz);
  lin(pre + sep + "pairTransition" + sep + "transition1", b.pairTrans.t1, Cz, 2 * Cz * pairFactor);
  lin(pre + sep + "pairTransition" + sep + "transition2", b.pairTrans.t2, Cz * pairFactor, Cz);
  if (!hasSingle) return;
  SingleAttn& a = b.single;
  a.C = Cs; a.Cz = Cz; a.H = Hs; a.D = Ds; a.gateConst = parallel ? 1 : 0;
  const int HD = Hs * Ds;
  ln(pre + sep + "singleAttention" + sep + "layerNorm", a.ln, Cs);
  lin(pre + sep + "singleAttention" + sep + "qProjection", a.q, Cs, HD);
  a.q.hasBias = true; f(pre + sep + "singleAttention" + sep + "qBias", a.q.b, HD);
  lin(pre + sep + "singleAttention" + sep + "kProjection", a.k, Cs, HD);
  lin(pre + sep + "singleAttention" + sep + "vProjection", a.v, Cs, HD);
  lin(pre + sep + "singleAttention" + sep + "gatingQuery", a.gate, Cs, HD);
  lin(pre + sep + "singleAttention" + sep + "outputProjection", a.out, HD, Cs);
  ln(pre + sep + "singlePairLogitsNorm", a.pairLn, Cz);
  lin(pre + sep + "singlePairLogitsProjection", a.pairProj, Cz, Hs);
  b.singleTrans.C = Cs; b.singleTrans.F = Cs * singleFactor;
  ln(pre + sep + "singleTransition" + sep + "inputLayerNorm", b.singleTrans.ln, Cs);
  lin(pre + sep + "singleTransition" + sep + "transition1", b.singleTrans.t1, Cs, 2 * Cs * singleFactor);
  lin(pre + sep + "singleTransition" + sep + "transition2", b.singleTrans.t2, Cs * singleFactor, Cs);
}

inline real* cloneR(const real* x, size_t n) {
  real* y = ARENA.get(n);
  GK(cudaMemcpyAsync(y, x, n * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  return y;
}

// pair [n, n, Cz], single [n, Cs] -> pairOut, singleOut. maskT is pairMask transposed.
// Sequential (AF3): each update reads the running pair; the single track reads the pair after all five.
// Parallel (chai-1): every update reads the block's input pair and they are summed; the single track's attention
// reads the input pair, and its transition the input single.
inline PairformerSave pairformerForward(Pairformer& b, const Dialect& d, const real* pair, const real* single,
                                        const real* pairMask, const real* maskT, const real* seqMask, const real* extra,
                                        real* pairOut, real* singleOut, int n) {
  const int Cz = b.triOut.C;
  size_t nn = (size_t)n * n;
  PairformerSave s{};
  s.n = n; s.pair0 = pair; s.single0 = single;
  b.triOut.divideByLength = b.triIn.divideByLength = d.divideByLength;
  real* u = ARENA.get(nn * Cz);                 // a triangle update's output, before its residual
  if (b.parallel) {
    if (b.dropRate > 0) { fprintf(stderr, "the parallel pairformer's dropout is not written\n"); exit(1); }
    real* acc = cloneR(pair, nn * Cz);
    s.tOut = triMulForward(b.triOut, pair, pairMask, u, n); add(acc, u, nn * Cz);
    s.tIn = triMulForward(b.triIn, pair, pairMask, u, n); add(acc, u, nn * Cz);
    if (b.hasGrid) {
      s.g1 = gridForward(b.grid1, d, pair, pairMask, maskT, acc, n);
      s.g2 = gridForward(b.grid2, d, pair, pairMask, maskT, acc, n);
    }
    s.pt = gtForward(b.pairTrans, pair, acc, nn, 1);
    s.pair5 = acc;
  } else {
    const float rate = b.dropRate;
    s.p1 = cloneR(pair, nn * Cz);
    s.tOut = triMulForward(b.triOut, pair, pairMask, u, n); drop3(u, n, n, Cz, true, rate, b.dropSeed, DROP3_TRI_OUT); add(s.p1, u, nn * Cz);
    s.p2 = cloneR(s.p1, nn * Cz);
    s.tIn = triMulForward(b.triIn, s.p1, pairMask, u, n); drop3(u, n, n, Cz, true, rate, b.dropSeed, DROP3_TRI_IN); add(s.p2, u, nn * Cz);
    s.p4 = s.p2;
    if (b.hasGrid) {
      s.p3 = cloneR(s.p2, nn * Cz);
      if (rate > 0) {   // (the attentions add into their output: an update alone, masked, then added)
        real* gu = ARENA.zeros(nn * Cz);
        s.g1 = gridForward(b.grid1, d, s.p2, pairMask, maskT, gu, n);
        drop3(gu, n, n, Cz, true, rate, b.dropSeed, DROP3_START); add(s.p3, gu, nn * Cz);
        s.p4 = cloneR(s.p3, nn * Cz);
        GK(cudaMemsetAsync(gu, 0, nn * Cz * sizeof(real), STREAM));
        s.g2 = gridForward(b.grid2, d, s.p3, pairMask, maskT, gu, n);
        drop3(gu, n, n, Cz, false, rate, b.dropSeed, DROP3_END); add(s.p4, gu, nn * Cz);
      } else {
        s.g1 = gridForward(b.grid1, d, s.p2, pairMask, maskT, s.p3, n);
        s.p4 = cloneR(s.p3, nn * Cz);
        s.g2 = gridForward(b.grid2, d, s.p3, pairMask, maskT, s.p4, n);
      }
    }
    s.pair5 = cloneR(s.p4, nn * Cz);
    s.pt = gtForward(b.pairTrans, s.p4, s.pair5, nn, 1);
  }
  GK(cudaMemcpyAsync(pairOut, s.pair5, nn * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  if (!b.hasSingle) return s;
  const int Cs = b.single.C;
  s.s1 = cloneR(single, (size_t)n * Cs);
  s.sa = singleForward(b.single, single, b.parallel ? pair : s.pair5, seqMask, extra, s.s1, n);
  GK(cudaMemcpyAsync(singleOut, s.s1, (size_t)n * Cs * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.st = gtForward(b.singleTrans, b.parallel ? single : s.s1, singleOut, n, 1);
  return s;
}
// dpairOut, dsingleOut -> dpair, dsingle (written; may alias the outputs' gradients); dextra added when not null
inline void pairformerBackward(Pairformer& b, const Dialect& d, const PairformerSave& s, const real* pairMask,
                               const real* maskT, const real* seqMask, const real* dpairOut, const real* dsingleOut,
                               real* dpair, real* dsingle, real* dextra) {
  const int n = s.n, Cz = b.triOut.C;
  size_t nn = (size_t)n * n;
  real* dp = cloneR(dpairOut, nn * Cz);
  if (b.parallel) {
    real* dIn = cloneR(dpairOut, nn * Cz);       // the input's gradient: the residual, and every update's
    if (b.hasSingle) {
      const int Cs = b.single.C;
      real* ds = cloneR(dsingleOut, (size_t)n * Cs);
      gtBackward(b.singleTrans, s.st, dsingleOut, ds);
      singleBackward(b.single, s.sa, seqMask, dsingleOut, ds, dIn, dextra);
      GK(cudaMemcpyAsync(dsingle, ds, (size_t)n * Cs * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    }
    gtBackward(b.pairTrans, s.pt, dp, dIn);
    if (b.hasGrid) {
      gridBackward(b.grid2, d, s.g2, pairMask, maskT, dp, dIn);
      gridBackward(b.grid1, d, s.g1, pairMask, maskT, dp, dIn);
    }
    triMulBackward(b.triIn, s.tIn, pairMask, dp, dIn);
    triMulBackward(b.triOut, s.tOut, pairMask, dp, dIn);
    GK(cudaMemcpyAsync(dpair, dIn, nn * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    return;
  }
#ifndef GRAD_F64
  if (BF16) {   // the pair track's running gradient in bf16 (bindcraft2's dp): every module reads it and adds into it
    bf* dpb = toBf(dpairOut, nn * Cz);
    if (b.hasSingle) {
      Range r_("bw.single");
      const int Cs = b.single.C;
      real* ds = cloneR(dsingleOut, (size_t)n * Cs);
      gtBackward(b.singleTrans, s.st, dsingleOut, ds);
      { Scratch sc; real* dsa = cloneR(ds, (size_t)n * Cs); singleBackward(b.single, s.sa, seqMask, dsa, ds, dpb, dextra); }
      GK(cudaMemcpyAsync(dsingle, ds, (size_t)n * Cs * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    }
    { Range r_("bw.pairTrans"); gtBackward(b.pairTrans, s.pt, dpb, dpb); }
    // (dropout: an update's gradient is a masked copy of the running one, so the in-place pass reads the copy)
    const float rate = b.dropRate;
    auto du = [&](bool perRow, int module) -> const bf* { return rate > 0 ? drop3Bf(dpb, n, n, Cz, perRow, rate, b.dropSeed, module) : dpb; };
    if (b.hasGrid) {
      { Range r_("bw.grid2"); Scratch sc; gridBackward(b.grid2, d, s.g2, pairMask, maskT, du(false, DROP3_END), dpb); }
      { Range r_("bw.grid1"); Scratch sc; gridBackward(b.grid1, d, s.g1, pairMask, maskT, du(true, DROP3_START), dpb); }
    }
    { Range r_("bw.triIn"); Scratch sc; triMulBackward(b.triIn, s.tIn, pairMask, du(true, DROP3_TRI_IN), dpb); }
    { Range r_("bw.triOut"); Scratch sc; triMulBackward(b.triOut, s.tOut, pairMask, du(true, DROP3_TRI_OUT), dpb); }
    bfToFK<<<blocks(nn * Cz), 256, 0, STREAM>>>(dpb, dpair, nn * Cz);
    return;
  }
#endif
  if (b.hasSingle) {
    Range r_("bw.single");
    const int Cs = b.single.C;
    real* ds = cloneR(dsingleOut, (size_t)n * Cs);
    gtBackward(b.singleTrans, s.st, dsingleOut, ds);
    { Scratch sc; real* dsa = cloneR(ds, (size_t)n * Cs); singleBackward(b.single, s.sa, seqMask, dsa, ds, dp, dextra); }
    GK(cudaMemcpyAsync(dsingle, ds, (size_t)n * Cs * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  // (a module's bf16 backward reads its incoming gradient whole before it adds into dpair, so in BF16 the running
  // gradient is passed as both - bindcraft2's ACC - where float's backwards need it copied first)
  auto cur = [&](Scratch&) { return BF16 ? dp : cloneR(dp, nn * Cz); };
  // (with dropout an update's gradient is the running gradient under the forward's mask: always a masked copy)
  const float rate = b.dropRate;
  auto masked = [&](Scratch&, bool perRow, int module) { real* du = cloneR(dp, nn * Cz); drop3(du, n, n, Cz, perRow, rate, b.dropSeed, module); return du; };
  { Range r_("bw.pairTrans"); Scratch sc; real* du = cur(sc); gtBackward(b.pairTrans, s.pt, du, dp); }
  if (b.hasGrid) {
    { Range r_("bw.grid2"); Scratch sc; real* du = rate > 0 ? masked(sc, false, DROP3_END) : cur(sc); gridBackward(b.grid2, d, s.g2, pairMask, maskT, du, dp); }
    { Range r_("bw.grid1"); Scratch sc; real* du = rate > 0 ? masked(sc, true, DROP3_START) : cur(sc); gridBackward(b.grid1, d, s.g1, pairMask, maskT, du, dp); }
  }
  { Range r_("bw.triIn"); Scratch sc; real* du = rate > 0 ? masked(sc, true, DROP3_TRI_IN) : cur(sc); triMulBackward(b.triIn, s.tIn, pairMask, du, dp); }
  { Range r_("bw.triOut"); Scratch sc; real* du = rate > 0 ? masked(sc, true, DROP3_TRI_OUT) : cur(sc); triMulBackward(b.triOut, s.tOut, pairMask, du, dp); }
  GK(cudaMemcpyAsync(dpair, dp, nn * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}


// ---------------------------------------------------------------- the MSA module
// Pair-weighted averaging (AF3's MSA attention): no queries and no keys - the weights are softmax_j of LN(pair) Wl
// plus a key mask (the coverage: max over the rows; chai-1 the token mask, with masked rows' values zeroed), shared
// by every MSA row; values and a gate from LN(msa).
struct Pwa {
  LayerNorm ln, pn;
  Linear logits, v, gate, out;     // logits [Cz, H]; v, gate [Cm, HD]; out [HD, Cm]
  int Cm = 0, Cz = 0, H = 0, D = 0;
  bool chaiMask = false;
  bool noKeyMask = false;          // ESMFold2's MSA encoder and MSA Pairformer: no mask on the weights at all
};
struct PwaSave { LnSave ln, lp; real *xn, *pn, *w, *vh, *oh, *g, *km; int S, n; };
__global__ void fillK(real* x, real v, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = v;
}
// keyMask[j] = max_s msaMask[s, j]
__global__ void coverageK(const real* m, real* km, int S, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  real a = 0; for (int s = 0; s < S; ++s) a = fmax(a, m[(size_t)s * n + j]);
  km[j] = a;
}
// w[h, i, j] = softmax_j(flat[(i j), h] + 1e9 (km[j] - 1))
__global__ void pwaWeightsK(const real* flat, const real* km, real* w, int n, int H) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= (size_t)H * n) return;
  int h = row / n, i = row % n;
  real* wr = w + row * n;
  real mx = -INFINITY;
  for (int j = lane; j < n; j += 32) { real v = flat[((size_t)i * n + j) * H + h] + 1e9f * (km[j] - 1); wr[j] = v; mx = fmax(mx, v); }
  for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(~0u, mx, o));
  real sum = 0;
  for (int j = lane; j < n; j += 32) { real e = exp(wr[j] - mx); wr[j] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  for (int j = lane; j < n; j += 32) wr[j] /= sum;
}
// dW [H, n, n] in place -> dlogits, then scattered to dflat [(i j), h]
__global__ void pwaWeightsBwdK(real* dw, const real* w, real* dflat, int n, int H) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= (size_t)H * n) return;
  int h = row / n, i = row % n;
  real* d = dw + row * n;
  const real* p = w + row * n;
  real s = 0;
  for (int j = lane; j < n; j += 32) s += d[j] * p[j];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  for (int j = lane; j < n; j += 32) dflat[((size_t)i * n + j) * H + h] = p[j] * (d[j] - s);
}
// [S, n, H, D] <-> [H][n][S D]
__global__ void pwaHeadsK(const real* x, real* y, int S, int n, int H, int D, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * n * H * D) return;
  int d = t % D; size_t r = t / D; int h = r % H; r /= H; int j = r % n; int s = (int)(r / n);
  size_t q = (((size_t)h * n + j) * S + s) * D + d;
  if (back) y[t] = x[q]; else y[q] = x[t];
}
// msa [S, n, Cm] += pwa(msa, pair)
inline PwaSave pwaForward(const Pwa& a, const real* msa, const real* msaMask, const real* seqMask, const real* pair,
                          real* msaOut, int S, int n) {
  size_t rows = (size_t)S * n, nn = (size_t)n * n;
  const int HD = a.H * a.D;
  PwaSave s{}; s.S = S; s.n = n;
  s.xn = ARENA.get(rows * a.Cm);
  s.ln = lnForward(a.ln, msa, s.xn, rows);
  s.pn = ARENA.get(nn * a.Cz);
  s.lp = lnForward(a.pn, pair, s.pn, nn);
  s.km = ARENA.get(n);
  if (a.noKeyMask) fillK<<<blocks(n), 256, 0, STREAM>>>(s.km, 1, n);
  else if (a.chaiMask) GK(cudaMemcpyAsync(s.km, seqMask, n * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  else coverageK<<<blocks(n), 256, 0, STREAM>>>(msaMask, s.km, S, n);
  s.w = ARENA.get((size_t)a.H * nn);
  {
    Scratch scratch;
    real* flat = ARENA.get(nn * a.H);
    a.logits.forward(s.pn, flat, nn);
    pwaWeightsK<<<blocks((size_t)a.H * n, 8), 256, 0, STREAM>>>(flat, s.km, s.w, n, a.H);
  }
  s.vh = ARENA.get(rows * HD); s.oh = ARENA.get(rows * HD); s.g = ARENA.get(rows * HD);
  Scratch scratch;
  real* v = ARENA.get(rows * HD);
  a.v.forward(s.xn, v, rows);
  if (a.chaiMask) rowMaskK<<<blocks(rows * HD), 256, 0, STREAM>>>(v, msaMask, rows, HD);
  pwaHeadsK<<<blocks(rows * HD), 256, 0, STREAM>>>(v, s.vh, S, n, a.H, a.D, false);
  const int SD = S * a.D;
  bmm(false, false, n, SD, n, s.w, (long long)nn, s.vh, (long long)n * SD, s.oh, (long long)n * SD, a.H);
  a.gate.forward(s.xn, s.g, rows);
  sigmoidK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.g, rows * HD);
  real* o = ARENA.get(rows * HD);
  pwaHeadsK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.oh, o, S, n, a.H, a.D, true);
  mulK<<<blocks(rows * HD), 256, 0, STREAM>>>(o, o, s.g, rows * HD);
  a.out.forward(o, msaOut, rows, 1);
  return s;
}
// dout [S n, Cm] -> dmsa (added), dpair (added)
inline void pwaBackward(Pwa& a, const PwaSave& s, const real* msaMask, const real* dout, real* dmsa, real* dpair) {
  Scratch scratch;
  const int S = s.S, n = s.n, HD = a.H * a.D, SD = S * a.D;
  size_t rows = (size_t)S * n, nn = (size_t)n * n;
  real *o = ARENA.get(rows * HD), *og = ARENA.get(rows * HD);
  pwaHeadsK<<<blocks(rows * HD), 256, 0, STREAM>>>(s.oh, o, S, n, a.H, a.D, true);
  mulK<<<blocks(rows * HD), 256, 0, STREAM>>>(og, o, s.g, rows * HD);
  real* dog = ARENA.get(rows * HD);
  a.out.backward(og, dout, dog, rows);
  real *dO = ARENA.get(rows * HD), *dg = ARENA.get(rows * HD);
  gateBwdK<<<blocks(rows * HD), 256, 0, STREAM>>>(dog, o, s.g, dO, dg, rows * HD);
  real* dxn = ARENA.get(rows * a.Cm);
  a.gate.backward(s.xn, dg, dxn, rows);
  real* doh = ARENA.get(rows * HD);
  pwaHeadsK<<<blocks(rows * HD), 256, 0, STREAM>>>(dO, doh, S, n, a.H, a.D, false);
  real *dw = ARENA.get((size_t)a.H * nn), *dvh = ARENA.get(rows * HD);
  bmm(false, true, n, n, SD, doh, (long long)n * SD, s.vh, (long long)n * SD, dw, (long long)nn, a.H);   // dW = dO V^T
  bmm(true, false, n, SD, n, s.w, (long long)nn, doh, (long long)n * SD, dvh, (long long)n * SD, a.H);  // dV = W^T dO
  real* dv = ARENA.get(rows * HD);
  pwaHeadsK<<<blocks(rows * HD), 256, 0, STREAM>>>(dvh, dv, S, n, a.H, a.D, true);
  if (a.chaiMask) rowMaskK<<<blocks(rows * HD), 256, 0, STREAM>>>(dv, msaMask, rows, HD);
  a.v.backward(s.xn, dv, dxn, rows, true);
  lnBackward(a.ln, s.ln, dxn, dmsa, rows, true);
  real* dflat = ARENA.get(nn * a.H);
  pwaWeightsBwdK<<<blocks((size_t)a.H * n, 8), 256, 0, STREAM>>>(dw, s.w, dflat, n, a.H);
  real* dpn = ARENA.get(nn * a.Cz);
  a.logits.backward(s.pn, dflat, dpn, nn);
  lnBackward(a.pn, s.lp, dpn, dpair, nn, true);
}

// One MSA block:  AF3:  pair += OPM(msa); msa += PWA(msa, pair); msa += T(msa); then the pair half
//                 updateFirst (OpenDDE, Boltz-2): the MSA update first, the outer product of the updated MSA
//                 chai-1 (grouped): pair += gOPM(msa); msa += T(msa) + PWA(msa, pair) off the same msa; then the
//                 pair half in two parallel stages: {triOut, triIn, transition}, then {grid row, grid column}
struct MsaBlock {
  Opm opm;
  Pwa att;
  GatedTransition trans;
  Pairformer pairHalf;
  bool updateFirst = false, chai = false;
  // training's MSA dropout (OpenFold3's msa_dropout): the pair-weighted averaging's update rowwise (shared over the
  // MSA's rows), at dropRate; the pair half's own in pairHalf
  float dropRate = 0;
  unsigned long long dropSeed = 0;
};
struct MsaSave {
  OpmSave opm;
  PwaSave att;
  GtSave tr;
  PairformerSave ph;       // sequential pair half
  TriMulSave tOut, tIn; GtSave pt; GridSave g1, g2;  // chai's two parallel stages
  real *pairA, *pairB, *msaA;
  int S, n;
};
inline MsaSave msaForward(MsaBlock& b, const Dialect& d, const real* msa, const real* pair, const real* msaMask,
                          const real* pairMask, const real* maskT, const real* seqMask, real* msaOut, real* pairOut,
                          int S, int n) {
  const int Cz = b.pairHalf.triOut.C, Cm = b.att.Cm;
  size_t rows = (size_t)S * n, nn = (size_t)n * n;
  MsaSave s{}; s.S = S; s.n = n;
  real* p = cloneR(pair, nn * Cz);
  if (b.chai) {
    if (b.dropRate > 0 || b.pairHalf.dropRate > 0) { fprintf(stderr, "chai-1's MSA block's dropout is not written\n"); exit(1); }
    s.opm = opmForward(b.opm, msa, msaMask, p, S, n);
    GK(cudaMemcpyAsync(msaOut, msa, rows * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    s.tr = gtForward(b.trans, msa, msaOut, rows, 1);
    s.att = pwaForward(b.att, msa, msaMask, seqMask, p, msaOut, S, n);
    s.pairA = p;
    real* a2 = cloneR(p, nn * Cz);
    real* u = ARENA.get(nn * Cz);
    b.pairHalf.triOut.divideByLength = b.pairHalf.triIn.divideByLength = d.divideByLength;
    s.tOut = triMulForward(b.pairHalf.triOut, p, pairMask, u, n); add(a2, u, nn * Cz);
    s.tIn = triMulForward(b.pairHalf.triIn, p, pairMask, u, n); add(a2, u, nn * Cz);
    s.pt = gtForward(b.pairHalf.pairTrans, p, a2, nn, 1);
    s.pairB = a2;
    real* a3 = cloneR(a2, nn * Cz);
    s.g1 = gridForward(b.pairHalf.grid1, d, a2, pairMask, maskT, a3, n);
    s.g2 = gridForward(b.pairHalf.grid2, d, a2, pairMask, maskT, a3, n);
    GK(cudaMemcpyAsync(pairOut, a3, nn * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    return s;
  }
  const real* mIn = msa;
  if (!b.updateFirst) s.opm = opmForward(b.opm, msa, msaMask, p, S, n);
  s.pairA = p;
  s.msaA = cloneR(msa, rows * Cm);
  // (update first: the attention reads the incoming pair, which p is about to stop being - a save keeps a pointer)
  if (b.dropRate > 0) {
    real* mu = ARENA.zeros(rows * Cm);
    s.att = pwaForward(b.att, msa, msaMask, seqMask, b.updateFirst ? pair : p, mu, S, n);
    drop3(mu, S, n, Cm, true, b.dropRate, b.dropSeed, DROP3_MSA); add(s.msaA, mu, rows * Cm);
  } else
    s.att = pwaForward(b.att, msa, msaMask, seqMask, b.updateFirst ? pair : p, s.msaA, S, n);
  GK(cudaMemcpyAsync(msaOut, s.msaA, rows * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  s.tr = gtForward(b.trans, s.msaA, msaOut, rows, 1);
  if (b.updateFirst) { s.pairA = p; s.opm = opmForward(b.opm, msaOut, msaMask, p, S, n); }
  (void)mIn;
  s.ph = pairformerForward(b.pairHalf, d, p, nullptr, pairMask, maskT, seqMask, nullptr, pairOut, nullptr, n);
  return s;
}
// dmsaOut, dpairOut -> dmsa, dpair (written; may alias them)
inline void msaBackward(MsaBlock& b, const Dialect& d, const MsaSave& s, const real* msaMask, const real* pairMask,
                        const real* maskT, const real* seqMask, const real* dmsaOut, const real* dpairOut, real* dmsa,
                        real* dpair) {
  const int S = s.S, n = s.n, Cz = b.pairHalf.triOut.C, Cm = b.att.Cm;
  size_t rows = (size_t)S * n, nn = (size_t)n * n;
  real* dm = cloneR(dmsaOut, rows * Cm);
  real* dp = ARENA.get(nn * Cz);
  if (b.chai) {
    // pairOut = a2 + g1(a2) + g2(a2); a2 = p + triOut(p) + triIn(p) + T(p); p = pair + gOPM(msa)
    real* da2 = cloneR(dpairOut, nn * Cz);
    gridBackward(b.pairHalf.grid2, d, s.g2, pairMask, maskT, dpairOut, da2);
    gridBackward(b.pairHalf.grid1, d, s.g1, pairMask, maskT, dpairOut, da2);
    GK(cudaMemcpyAsync(dp, da2, nn * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    gtBackward(b.pairHalf.pairTrans, s.pt, da2, dp);
    triMulBackward(b.pairHalf.triIn, s.tIn, pairMask, da2, dp);
    triMulBackward(b.pairHalf.triOut, s.tOut, pairMask, da2, dp);
    // msaOut = msa + T(msa) + PWA(msa, p)
    gtBackward(b.trans, s.tr, dmsaOut, dm);
    pwaBackward(b.att, s.att, msaMask, dmsaOut, dm, dp);
    opmBackward(b.opm, s.opm, msaMask, dp, dm);
  } else {
    pairformerBackward(b.pairHalf, d, s.ph, pairMask, maskT, seqMask, dpairOut, nullptr, dp, nullptr, nullptr);
    if (b.updateFirst) opmBackward(b.opm, s.opm, msaMask, dp, dm);    // p = pairA + OPM(msaOut)
    { Scratch sc; real* du = cloneR(dm, rows * Cm); gtBackward(b.trans, s.tr, du, dm); }   // msaOut = msaA + T(msaA)
    { Scratch sc; real* du = cloneR(dm, rows * Cm); drop3(du, S, n, Cm, true, b.dropRate, b.dropSeed, DROP3_MSA); pwaBackward(b.att, s.att, msaMask, du, dm, dp); }
    if (!b.updateFirst) opmBackward(b.opm, s.opm, msaMask, dp, dm);   // pairA = pair + OPM(msa)
  }
  GK(cudaMemcpyAsync(dmsa, dm, rows * Cm * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  GK(cudaMemcpyAsync(dpair, dp, nn * Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
}

// every parameter of an MSA block under `pre` ("trunk.msaBlocks.0"), the port's names
inline void forEachParam(MsaBlock& b, const std::string& pre, int Cz, int Cm, int Hp, int Dp, int Ha, int Da, int O, int G,
                         bool opmBiases, bool gridBiases, const std::function<void(const std::string&, Param&, size_t)>& f) {
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(n + "Scale", l.g, C); f(n + "Offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N, bool bias = false, const std::string& bn = "") {
    l.K = K; l.N = N; l.hasBias = bias; l.w.fanIn = K; f(n, l.w, (size_t)K * N); if (bias) f(bn, l.b, N);
  };
  const std::string O_ = pre + ".outerProductMean";
  b.opm.C = Cm; b.opm.O = O; b.opm.Cz = Cz; b.opm.grouped = b.chai; b.opm.G = G;
  ln(O_ + ".layerNormInput", b.opm.ln, Cm);
  lin(O_ + ".leftProjection", b.opm.left, Cm, O, opmBiases, O_ + ".leftProjectionBias");
  lin(O_ + ".rightProjection", b.opm.right, Cm, O, opmBiases, O_ + ".rightProjectionBias");
  const int OO = b.chai ? O / G * O : O * O;
  if (b.chai) { ln(O_ + ".productNorm", b.opm.pn, OO); b.opm.pn.eps = 0.1f; }
  f(O_ + ".outputW", b.opm.wo, (size_t)OO * Cz);
  f(O_ + ".outputB", b.opm.bo, Cz);
  const std::string A = pre + ".msaAttention1";
  b.att.Cm = Cm; b.att.Cz = Cz; b.att.H = Ha; b.att.D = Da; b.att.chaiMask = b.chai;
  ln(A + ".actNorm", b.att.ln, Cm);
  ln(A + ".pairNorm", b.att.pn, Cz);
  lin(A + ".pairLogits", b.att.logits, Cz, Ha);
  lin(A + ".vProjection", b.att.v, Cm, Ha * Da);
  lin(A + ".gatingQuery", b.att.gate, Cm, Ha * Da);
  lin(A + ".outputProjection", b.att.out, Ha * Da, Cm);
  b.trans.C = Cm; b.trans.F = 4 * Cm;
  ln(pre + ".msaTransition.inputLayerNorm", b.trans.ln, Cm);
  lin(pre + ".msaTransition.transition1", b.trans.t1, Cm, 8 * Cm);
  lin(pre + ".msaTransition.transition2", b.trans.t2, 4 * Cm, Cm);
  forEachParam(b.pairHalf, pre, Cz, 0, Hp, Dp, 0, 0, 4, 4, gridBiases, false, f, true, b.chai);
}

}  // namespace af3
}  // namespace grad
