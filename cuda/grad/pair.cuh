// The pair track's shared layers: the triangle multiplication of AF2 (blocked split, biases) and of the AF3
// lineage (interleaved split, no biases, RoseTTAFold3's division by the length), forward with saves and backward.
#pragma once
#include "ops.cuh"

namespace grad {

// ---------------------------------------------------------------- triangle multiplication (fused projections)
struct TriMul {
  LayerNorm lnIn, lnCenter;
  Linear proj, gate, out, gating;     // [C, 2I], [C, 2I], [I, C], [C, C]
  int C = 128, I = 128;
  bool outgoing = true;
  bool interleaved = false;           // AF3: channel c's halves are outputs 2c and 2c + 1, not c and I + c
  bool divideByLength = false;        // RoseTTAFold3: the contraction / n before the centre norm
};
struct TriMulSave {
  LnSave lnIn, lnC;
  real *xn, *proj, *gate, *ab, *center, *cn, *og;   // og: sigmoid of the output gate
  int L;
};
// [L, L, C2] -> channel-major [C2][L][L]; interleaved, column 2p goes to plane p and 2p + 1 to plane C2/2 + p
__device__ inline int planeOf(int c, int C2, bool inter) { return inter ? (c & 1) * (C2 / 2) + (c >> 1) : c; }
__global__ void splitChannelsK(const real* __restrict__ pg, real* __restrict__ cm, int L, int C2, bool inter = false) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C2) return;
  int c = t % C2; size_t ij = t / C2;
  cm[(size_t)planeOf(c, C2, inter) * L * L + ij] = pg[t];
}
__global__ void mergeChannelsK(const real* __restrict__ cm, real* __restrict__ pg, int L, int C2, bool inter = false) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C2) return;
  int c = t % C2; size_t ij = t / C2;
  pg[t] = cm[(size_t)planeOf(c, C2, inter) * L * L + ij];
}

inline TriMulSave triMulForward(const TriMul& m, const real* pair, const real* mask, real* out, int L) {
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
    real* pg = ARENA.get(rows * I2);
    mulK<<<blocks(rows * I2), 256, 0, STREAM>>>(pg, s.proj, s.gate, rows * I2);
    splitChannelsK<<<blocks(rows * I2), 256, 0, STREAM>>>(pg, s.ab, L, I2, m.interleaved);
    const real *A = s.ab, *B = s.ab + (size_t)m.I * rows;
    real* ccm = pg;                                              // reuse: [I][L][L]
    long long st = (long long)L * L;
    const real alpha = m.divideByLength ? (real)1 / L : (real)1;
    if (m.outgoing) bmm(false, true, L, L, L, A, st, B, st, ccm, st, m.I, 0, alpha);   // o_ij = sum_k a_ik b_jk
    else bmm(true, false, L, L, L, B, st, A, st, ccm, st, m.I, 0, alpha);              // o_ij = sum_k a_kj b_ki
    mergeChannelsK<<<blocks(rows * m.I), 256, 0, STREAM>>>(ccm, s.center, L, m.I);
  }
  s.lnC = lnForward(m.lnCenter, s.center, s.cn, rows);
  m.out.forward(s.cn, out, rows);
  m.gating.forward(s.xn, s.og, rows);
  sigmoidK<<<blocks(rows * m.C), 256, 0, STREAM>>>(s.og, rows * m.C);
  mulK<<<blocks(rows * m.C), 256, 0, STREAM>>>(out, out, s.og, rows * m.C);
  return s;
}

inline void triMulBackward(TriMul& m, const TriMulSave& s, const real* mask, const real* dout, real* dpair) {
  Scratch scratch;
  const int L = s.L, I2 = 2 * m.I;
  size_t rows = (size_t)L * L;
  // out = (cn Wout + bout) * og ; og = sigmoid(xn Wgating + b)
  real* pre = ARENA.get(rows * m.C);
  m.out.forward(s.cn, pre, rows);
  real *dpre = ARENA.get(rows * m.C), *dgl = ARENA.get(rows * m.C);
  gateBwdK<<<blocks(rows * m.C), 256, 0, STREAM>>>(dout, pre, s.og, dpre, dgl, rows * m.C);
  real* dxn = ARENA.get(rows * m.C);
  m.gating.backward(s.xn, dgl, dxn, rows);
  real* dcn = ARENA.get(rows * m.I);
  m.out.backward(s.cn, dpre, dcn, rows);
  real* dcenter = ARENA.get(rows * m.I);
  lnBackward(m.lnCenter, s.lnC, dcn, dcenter, rows, false);
  real* dccm = ARENA.get(rows * m.I);
  splitChannelsK<<<blocks(rows * m.I), 256, 0, STREAM>>>(dcenter, dccm, L, m.I);
  real* dab = ARENA.get(rows * I2);
  const real *A = s.ab, *B = s.ab + (size_t)m.I * rows;
  real *dA = dab, *dB = dab + (size_t)m.I * rows;
  long long st = (long long)L * L;
  const real alpha = m.divideByLength ? (real)1 / L : (real)1;
  if (m.outgoing) {          // O = A B^T : dA = dO B, dB = dO^T A
    bmm(false, false, L, L, L, dccm, st, B, st, dA, st, m.I, 0, alpha);
    bmm(true, false, L, L, L, dccm, st, A, st, dB, st, m.I, 0, alpha);
  } else {                   // O = B^T A : dA = B dO, dB = A dO^T
    bmm(false, false, L, L, L, B, st, dccm, st, dA, st, m.I, 0, alpha);
    bmm(false, true, L, L, L, A, st, dccm, st, dB, st, m.I, 0, alpha);
  }
  real* dpg = ARENA.get(rows * I2);
  mergeChannelsK<<<blocks(rows * I2), 256, 0, STREAM>>>(dab, dpg, L, I2, m.interleaved);
  // pg = proj * gate (proj already masked): dproj = dpg gate (then masked), dgate_pre = dpg proj gate (1 - gate)
  real *dproj = ARENA.get(rows * I2), *dgp = ARENA.get(rows * I2);
  gateBwdK<<<blocks(rows * I2), 256, 0, STREAM>>>(dpg, s.proj, s.gate, dproj, dgp, rows * I2);
  rowMaskK<<<blocks(rows * I2), 256, 0, STREAM>>>(dproj, mask, rows, I2);
  m.proj.backward(s.xn, dproj, dxn, rows, true);
  m.gate.backward(s.xn, dgp, dxn, rows, true);
  lnBackward(m.lnIn, s.lnIn, dxn, dpair, rows, true);
}

}  // namespace grad
