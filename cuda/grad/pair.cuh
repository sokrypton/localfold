// The pair track's shared layers, forward with saves and backward: the triangle multiplication of AF2 (blocked split,
// biases) and of the AF3 lineage (interleaved split, no biases, RoseTTAFold3's division by the length); and the
// outer product mean of both (AF2's and AF3's, Boltz-2's bias after the count, RoseTTAFold3's projection biases,
// chai-1's grouped form).
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
  bool swapAB = false;                // Boltz's incoming form, sum_k a_ki b_kj: a and b in each other's place
  bool masked = true;                 // (BioEmu2's carry no mask)
};
struct TriMulSave {
  LnSave lnIn, lnC;
  real *xn, *proj, *gate, *ab, *center, *cn, *og;   // og: sigmoid of the output gate
  int L;
};
// [L, L, C2] -> channel-major [C2][L][L]; interleaved, column 2p goes to plane p and 2p + 1 to plane C2/2 + p
__device__ inline int planeOf(int c, int C2, bool inter) { return inter ? (c & 1) * (C2 / 2) + (c >> 1) : c; }
// [rows][C2] <-> [plane(c)][rows], a 32 x 32 tile through shared memory so both sides are coalesced
// mode 0: cm = transpose(pg); mode 1: pg = transpose(cm); mode 2 (split of proj * gate): cm = transpose(a b)
__global__ void channelsT(const real* __restrict__ x, const real* __restrict__ y, real* __restrict__ out, size_t rows,
                          int C2, bool inter, int mode) {
  __shared__ real tile[32][33];
  const size_t r0 = (size_t)blockIdx.x * 32;
  const int c0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
  if (mode != 1) {
    for (int k = ty; k < 32; k += 8) {
      size_t r = r0 + k; int c = c0 + tx;
      if (r < rows && c < C2) tile[k][tx] = mode == 2 ? x[r * C2 + c] * y[r * C2 + c] : x[r * C2 + c];
    }
    __syncthreads();
    for (int k = ty; k < 32; k += 8) {
      int c = c0 + k; size_t r = r0 + tx;
      if (r < rows && c < C2) out[(size_t)planeOf(c, C2, inter) * rows + r] = tile[tx][k];
    }
  } else {
    for (int k = ty; k < 32; k += 8) {
      int c = c0 + k; size_t r = r0 + tx;
      if (r < rows && c < C2) tile[tx][k] = x[(size_t)planeOf(c, C2, inter) * rows + r];
    }
    __syncthreads();
    for (int k = ty; k < 32; k += 8) {
      size_t r = r0 + k; int c = c0 + tx;
      if (r < rows && c < C2) out[r * C2 + c] = tile[k][tx];
    }
  }
}
inline void splitChannels(const real* pg, real* cm, size_t rows, int C2, bool inter, const real* gate = nullptr) {
  dim3 b(32, 8), g((unsigned)((rows + 31) / 32), (C2 + 31) / 32);
  channelsT<<<g, b, 0, STREAM>>>(pg, gate, cm, rows, C2, inter, gate ? 2 : 0);
}
inline void mergeChannels(const real* cm, real* pg, size_t rows, int C2, bool inter) {
  dim3 b(32, 8), g((unsigned)((rows + 31) / 32), (C2 + 31) / 32);
  channelsT<<<g, b, 0, STREAM>>>(cm, nullptr, pg, rows, C2, inter, 1);
}
// the backward of pg = proj * gate, with the merge from channel-major dab fused in: dproj = d gate (row-masked),
// dgate_pre = d proj gate (1 - gate)
__global__ void mergeGateBwdT(const real* __restrict__ dab, const real* __restrict__ proj, const real* __restrict__ gate,
                              const real* __restrict__ mask, real* __restrict__ dproj, real* __restrict__ dgp,
                              size_t rows, int C2, bool inter) {
  __shared__ real tile[32][33];
  const size_t r0 = (size_t)blockIdx.x * 32;
  const int c0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
  for (int k = ty; k < 32; k += 8) {
    int c = c0 + k; size_t r = r0 + tx;
    if (r < rows && c < C2) tile[tx][k] = dab[(size_t)planeOf(c, C2, inter) * rows + r];
  }
  __syncthreads();
  for (int k = ty; k < 32; k += 8) {
    size_t r = r0 + k; int c = c0 + tx;
    if (r >= rows || c >= C2) continue;
    size_t i = r * C2 + c;
    real d = tile[k][tx], sg = gate[i];
    dproj[i] = mask ? d * sg * mask[r] : d * sg;
    dgp[i] = d * proj[i] * sg * (1.f - sg);
  }
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
  if (m.masked) rowMaskK<<<blocks(rows * I2), 256, 0, STREAM>>>(s.proj, mask, rows, I2);
  m.gate.forward(s.xn, s.gate, rows);
  sigmoidK<<<blocks(rows * I2), 256, 0, STREAM>>>(s.gate, rows * I2);
  s.ab = ARENA.get(rows * I2);                                    // channel-major a then b
  s.center = ARENA.get(rows * m.I);
  s.cn = ARENA.get(rows * m.I);
  s.og = ARENA.get(rows * m.C);
  {
    Scratch scratch;
    real* pg = ARENA.get(rows * I2);
    splitChannels(s.proj, s.ab, rows, I2, m.interleaved, s.gate);   // (proj * gate, channel-major)
    const real *A = s.ab, *B = s.ab + (size_t)m.I * rows;
    if (m.swapAB) std::swap(A, B);
    real* ccm = pg;                                              // reuse: [I][L][L]
    long long st = (long long)L * L;
    const real alpha = m.divideByLength ? (real)1 / L : (real)1;
    if (m.outgoing) bmm(false, true, L, L, L, A, st, B, st, ccm, st, m.I, 0, alpha);   // o_ij = sum_k a_ik b_jk
    else bmm(true, false, L, L, L, B, st, A, st, ccm, st, m.I, 0, alpha);              // o_ij = sum_k a_kj b_ki
    mergeChannels(ccm, s.center, rows, m.I, false);
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
  splitChannels(dcenter, dccm, rows, m.I, false);
  real* dab = ARENA.get(rows * I2);
  const real *A = s.ab, *B = s.ab + (size_t)m.I * rows;
  real *dA = dab, *dB = dab + (size_t)m.I * rows;
  if (m.swapAB) { std::swap(A, B); std::swap(dA, dB); }
  long long st = (long long)L * L;
  const real alpha = m.divideByLength ? (real)1 / L : (real)1;
  if (m.outgoing) {          // O = A B^T : dA = dO B, dB = dO^T A
    bmm(false, false, L, L, L, dccm, st, B, st, dA, st, m.I, 0, alpha);
    bmm(true, false, L, L, L, dccm, st, A, st, dB, st, m.I, 0, alpha);
  } else {                   // O = B^T A : dA = B dO, dB = A dO^T
    bmm(false, false, L, L, L, B, st, dccm, st, dA, st, m.I, 0, alpha);
    bmm(false, true, L, L, L, A, st, dccm, st, dB, st, m.I, 0, alpha);
  }
  // pg = proj * gate (proj already masked): dproj = dpg gate (then masked), dgate_pre = dpg proj gate (1 - gate)
  real *dproj = ARENA.get(rows * I2), *dgp = ARENA.get(rows * I2);
  {
    dim3 b(32, 8), g((unsigned)((rows + 31) / 32), (I2 + 31) / 32);
    mergeGateBwdT<<<g, b, 0, STREAM>>>(dab, s.proj, s.gate, m.masked ? mask : nullptr, dproj, dgp, rows, I2, m.interleaved);
  }
  m.proj.backward(s.xn, dproj, dxn, rows, true);
  m.gate.backward(s.xn, dgp, dxn, rows, true);
  lnBackward(m.lnIn, s.lnIn, dxn, dpair, rows, true);
}

// ---------------------------------------------------------------- outer product mean
// pair[i, j] += (sum_s L[s, i] (x) R[s, j]) Wo + b, divided by 1e-3 + the rows covering both (AF2, AF3), or
// Wo-projected, divided by max(count, 1), plus b (Boltz-2: biasAfterNorm). L, R = mask * (LN(msa) W (+ b)).
// Grouped (chai-1): the O channels as G groups of K, only within-group products [(i, j), (g, k, l)], LayerNorm
// (eps 0.1) over them, then Wo + b, and no count at all.
struct Opm {
  LayerNorm ln;
  Linear left, right;     // [C, O]
  Param wo, bo;           // [O O (or G K K), Cz], [Cz]
  int C = 256, O = 32, Cz = 128;
  bool biasAfterNorm = false, grouped = false;
  int G = 1;
  LayerNorm pn;           // grouped: the product's norm (eps 0.1)
  bool fused = false;     // one projection [C, 2O] (or [2O, C] transposed), a then b (ESMFold2, BioEmu2)
  Linear proj;
  bool clampCount = false;    // the divisor max(count, 1), the bias inside it (ESMFold2)
  real fixedCount = 0;        // a constant divisor, the bias after it (BioEmu2: the valid rows)
};
struct OpmSave { LnSave ln, lp; real *xn, *lm, *rm, *norm, *pnorm; int S, L; };
// lm [S, L, O] -> [S][L O] is already that layout. outer [(i c), (j e)] -> [(i j), (c e)]
__global__ void outerToPairK(const real* __restrict__ o2, real* __restrict__ op, int L, int O, bool back) {
  if (O % 4 == 0 && sizeof(real) == 4) {          // a float4 of e at a time (O = 32: one 128-byte row in 8 threads)
    const int O4 = O / 4;
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)L * L * O * O4) return;
    int e4 = t % O4; size_t r = t / O4; int c = r % O; r /= O; int j = r % L; int i = (int)(r / L);
    size_t q = (((size_t)i * O + c) * L + j) * O4 + e4;
    const float4* src = reinterpret_cast<const float4*>(o2);
    float4* dst = reinterpret_cast<float4*>(op);
    if (back) dst[q] = src[t]; else dst[t] = src[q];
    return;
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)L * L * O * O;
  if (t >= n) return;
  int e = t % O; size_t r = t / O; int c = r % O; r /= O; int j = r % L; int i = (int)(r / L);   // op[i, j, c, e]
  size_t q = (((size_t)i * O + c) * L + j) * O + e;
  if (back) op[q] = o2[t]; else op[t] = o2[q];
}
inline void outerToPair(const real* o2, real* op, int L, int O, bool back) {
  size_t n = (size_t)L * L * O * O;
  if (O % 4 == 0 && sizeof(real) == 4) n /= 4;
  outerToPairK<<<blocks(n), 256, 0, STREAM>>>(o2, op, L, O, back);
}
// [S, L, G, K] <-> group-major [G][S][L K]
__global__ void groupMajorK(const real* __restrict__ x, real* __restrict__ y, int S, int L, int G, int K, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * L * G * K) return;
  int k = t % K; size_t r = t / K; int g = r % G; r /= G; int i = r % L; int s = (int)(r / L);
  size_t q = (((size_t)g * S + s) * L + i) * K + k;
  if (back) y[t] = x[q]; else y[q] = x[t];
}
// group-major products [G][(i k)][(j l)] <-> [(i j)][(g k l)]
__global__ void groupedToPairK(const real* __restrict__ p, real* __restrict__ op, int L, int G, int K, bool back) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * G * K * K) return;
  int l = t % K; size_t r = t / K; int k = r % K; r /= K; int g = r % G; r /= G; int j = r % L; int i = (int)(r / L);
  size_t q = (size_t)g * L * K * L * K + (((size_t)i * K + k) * L + j) * K + l;
  if (back) op[q] = p[t]; else op[t] = p[q];
}
__global__ void normK(const real* __restrict__ mask, real* __restrict__ norm, int S, int L, bool clampOne) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L) return;
  int i = t / L, j = t % L;
  real s = 0;
  for (int a = 0; a < S; ++a) s += mask[(size_t)a * L + i] * mask[(size_t)a * L + j];
  norm[t] = clampOne ? fmax(s, (real)1) : (real)1e-3 + s;
}
__global__ void fillCountK(real* x, real v, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = v;
}
// y[r, 0..cols) = x[r * ldx + 0..cols) (accumulate: +=)
__global__ void copyColsK(const real* x, int ldx, real* y, int ldy, size_t rows, int cols, bool accumulate) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * cols) return;
  size_t r = t / cols; int c = t % cols;
  real v = x[r * ldx + c];
  y[r * ldy + c] = accumulate ? y[r * ldy + c] + v : v;
}
__global__ void divRowsK(real* x, const real* norm, size_t rows, int C, bool accumulateInto, real* into) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  real v = x[t] / norm[t / C];
  if (accumulateInto) into[t] += v; else x[t] = v;
}

// the outer products, [(i j), OO]: plain, or grouped and normalised (its LayerNorm's save in s)
inline void opmProducts(const Opm& m, OpmSave& s, real* op, bool forward) {
  const int S = s.S, L = s.L;
  size_t pr = (size_t)L * L, rows = (size_t)S * L;
  Scratch scratch;
  if (!m.grouped) {
    const int LO = L * m.O;
    real* o2 = ARENA.get((size_t)LO * LO);
    mmAT(s.lm, s.rm, o2, S, LO, LO, 0.f);                 // o2[(i c), (j e)] = lm^T rm
    outerToPair(o2, op, L, m.O, false);
    return;
  }
  const int G = m.G, K = m.O / G, LK = L * K;
  real *lg = ARENA.get(rows * m.O), *rg = ARENA.get(rows * m.O), *p = ARENA.get((size_t)G * LK * LK);
  groupMajorK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.lm, lg, S, L, G, K, false);
  groupMajorK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.rm, rg, S, L, G, K, false);
  bmm(true, false, LK, LK, S, lg, (long long)S * LK, rg, (long long)S * LK, p, (long long)LK * LK, G);
  real* raw = forward ? ARENA.get(pr * G * K * K) : op;
  groupedToPairK<<<blocks(pr * G * K * K), 256, 0, STREAM>>>(p, raw, L, G, K, false);
  if (forward) s.lp = lnForward(m.pn, raw, op, pr);      // (s.lp's stats outlive the scratch: allocated before it)
}

// pair += opm(msa)
inline OpmSave opmForward(const Opm& m, const real* msa, const real* msaMask, real* pair, int S, int L) {
  size_t rows = (size_t)S * L, pr = (size_t)L * L;
  const int OO = m.grouped ? m.O / m.G * m.O : m.O * m.O;
  OpmSave s{}; s.S = S; s.L = L;
  s.xn = ARENA.get(rows * m.C);
  s.ln = lnForward(m.ln, msa, s.xn, rows);
  s.lm = ARENA.get(rows * m.O); s.rm = ARENA.get(rows * m.O);
  if (m.fused) {
    Scratch scratch;
    real* x = ARENA.get(rows * 2 * m.O);
    m.proj.forward(s.xn, x, rows);
    copyColsK<<<blocks(rows * m.O), 256, 0, STREAM>>>(x, 2 * m.O, s.lm, m.O, rows, m.O, false);
    copyColsK<<<blocks(rows * m.O), 256, 0, STREAM>>>(x + m.O, 2 * m.O, s.rm, m.O, rows, m.O, false);
  } else { m.left.forward(s.xn, s.lm, rows); m.right.forward(s.xn, s.rm, rows); }
  if (msaMask) {
    rowMaskK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.lm, msaMask, rows, m.O);
    rowMaskK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.rm, msaMask, rows, m.O);
  }
  if (!m.grouped) {
    s.norm = ARENA.get(pr);
    if (m.fixedCount > 0) fillCountK<<<blocks(pr), 256, 0, STREAM>>>(s.norm, m.fixedCount, pr);
    else normK<<<blocks(pr), 256, 0, STREAM>>>(msaMask, s.norm, S, L, m.biasAfterNorm || m.clampCount);
  } else {
    s.pnorm = ARENA.get(pr * OO);                        // the grouped product's normalised form, Wo's input
    s.lp = LnSave{nullptr, ARENA.get(pr), ARENA.get(pr)};
  }
  Scratch scratch;
  real* op = m.grouped ? s.pnorm : ARENA.get(pr * OO);
  if (m.grouped) {
    // the LayerNorm's stats into the save, its input in scratch
    const int G = m.G, K = m.O / G, LK = L * K;
    real *lg = ARENA.get(rows * m.O), *rg = ARENA.get(rows * m.O), *p = ARENA.get((size_t)G * LK * LK), *raw = ARENA.get(pr * OO);
    groupMajorK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.lm, lg, S, L, G, K, false);
    groupMajorK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.rm, rg, S, L, G, K, false);
    bmm(true, false, LK, LK, S, lg, (long long)S * LK, rg, (long long)S * LK, p, (long long)LK * LK, G);
    groupedToPairK<<<blocks(pr * OO), 256, 0, STREAM>>>(p, raw, L, G, K, false);
    lnFwdK<<<blocks(pr, 8), 256, 0, STREAM>>>(raw, m.pn.g.w, m.pn.b.w, op, s.lp.mean, s.lp.rstd, pr, OO, m.pn.eps);
  } else {
    opmProducts(m, s, op, true);
  }
  real* act = ARENA.get(pr * m.Cz);
  mm(op, m.wo.w, act, (int)pr, OO, m.Cz);
  if (m.grouped) {
    biasK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(act, m.bo.w, pr, m.Cz);
    add(pair, act, pr * m.Cz);
  } else if (m.biasAfterNorm || m.fixedCount > 0) {
    divRowsK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(act, s.norm, pr, m.Cz, false, nullptr);
    biasK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(act, m.bo.w, pr, m.Cz);
    add(pair, act, pr * m.Cz);
  } else {
    biasK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(act, m.bo.w, pr, m.Cz);
    divRowsK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(act, s.norm, pr, m.Cz, true, pair);
  }
  return s;
}
inline void opmBackward(Opm& m, const OpmSave& s, const real* msaMask, const real* dpair, real* dmsa) {
  Scratch scratch;
  const int S = s.S, L = s.L;
  const int OO = m.grouped ? m.O / m.G * m.O : m.O * m.O;
  size_t rows = (size_t)S * L, pr = (size_t)L * L;
  real* dact = ARENA.get(pr * m.Cz);
  GK(cudaMemcpyAsync(dact, dpair, pr * m.Cz * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  const bool after = m.biasAfterNorm || m.fixedCount > 0;
  if (m.grouped || after) colSum(m.bo.g, dact, pr, m.Cz);
  if (!m.grouped) divRowsK<<<blocks(pr * m.Cz), 256, 0, STREAM>>>(dact, s.norm, pr, m.Cz, false, nullptr);
  if (!m.grouped && !after) colSum(m.bo.g, dact, pr, m.Cz);
  real* op = m.grouped ? s.pnorm : ARENA.get(pr * OO);
  if (!m.grouped) { OpmSave t = s; opmProducts(m, t, op, false); }
  mmAT(op, dact, m.wo.g, (int)pr, OO, m.Cz);
  real* dop = ARENA.get(pr * OO);
  mmBT(dact, m.wo.w, dop, (int)pr, OO, m.Cz);           // d (the product, or its normalised form) [(i j), OO]
  real *dlm = ARENA.get(rows * m.O), *drm = ARENA.get(rows * m.O);
  if (!m.grouped) {
    const int LO = L * m.O;
    real* do2 = ARENA.get((size_t)LO * LO);
    outerToPair(dop, do2, L, m.O, true);   // [(i c), (j e)]
    mmBT(s.rm, do2, dlm, S, LO, LO);                      // dlm = rm dO2^T
    mm(s.lm, do2, drm, S, LO, LO);                        // drm = lm dO2
  } else {
    const int G = m.G, K = m.O / G, LK = L * K;
    // the norm's backward needs its input: rebuild the raw products
    real *lg = ARENA.get(rows * m.O), *rg = ARENA.get(rows * m.O), *p = ARENA.get((size_t)G * LK * LK), *raw = ARENA.get(pr * OO);
    groupMajorK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.lm, lg, S, L, G, K, false);
    groupMajorK<<<blocks(rows * m.O), 256, 0, STREAM>>>(s.rm, rg, S, L, G, K, false);
    bmm(true, false, LK, LK, S, lg, (long long)S * LK, rg, (long long)S * LK, p, (long long)LK * LK, G);
    groupedToPairK<<<blocks(pr * OO), 256, 0, STREAM>>>(p, raw, L, G, K, false);
    real* draw = ARENA.get(pr * OO);
    LnSave ls{raw, s.lp.mean, s.lp.rstd};
    lnBackward(m.pn, ls, dop, draw, pr, false);
    groupedToPairK<<<blocks(pr * OO), 256, 0, STREAM>>>(draw, p, L, G, K, true);   // dP [G][(i k)][(j l)]
    real *dlg = ARENA.get(rows * m.O), *drg = ARENA.get(rows * m.O);
    bmm(false, true, S, LK, LK, rg, (long long)S * LK, p, (long long)LK * LK, dlg, (long long)S * LK, G);  // rg dP^T
    bmm(false, false, S, LK, LK, lg, (long long)S * LK, p, (long long)LK * LK, drg, (long long)S * LK, G); // lg dP
    groupMajorK<<<blocks(rows * m.O), 256, 0, STREAM>>>(dlg, dlm, S, L, G, K, true);
    groupMajorK<<<blocks(rows * m.O), 256, 0, STREAM>>>(drg, drm, S, L, G, K, true);
  }
  if (msaMask) {
    rowMaskK<<<blocks(rows * m.O), 256, 0, STREAM>>>(dlm, msaMask, rows, m.O);
    rowMaskK<<<blocks(rows * m.O), 256, 0, STREAM>>>(drm, msaMask, rows, m.O);
  }
  real* dxn = ARENA.get(rows * m.C);
  if (m.fused) {
    real* dx = ARENA.get(rows * 2 * m.O);
    copyColsK<<<blocks(rows * m.O), 256, 0, STREAM>>>(dlm, m.O, dx, 2 * m.O, rows, m.O, false);
    copyColsK<<<blocks(rows * m.O), 256, 0, STREAM>>>(drm, m.O, dx + m.O, 2 * m.O, rows, m.O, false);
    m.proj.backward(s.xn, dx, dxn, rows);
  } else { m.left.backward(s.xn, dlm, dxn, rows); m.right.backward(s.xn, drm, dxn, rows, true); }
  lnBackward(m.ln, s.ln, dxn, dmsa, rows, true);
}

}  // namespace grad
