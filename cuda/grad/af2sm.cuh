// AlphaFold 2's structure module for training: the eight shared-weight fold iterations (invariant point attention,
// the transition, the backbone update) and the side chain of the last, forward with saves and backward into every
// parameter, the single and pair representations - what FAPE, the side-chain and the torsion losses need.
// cuda/af2/src/structure.cuh's structureModule() is what the forward is held to (af3-any-model's folding.py:
// FoldIteration, InvariantPointAttention, MultiRigidSidechain). A rigid is 12 numbers: the rotation row-major, then
// the translation (nanometres inside the module; the outputs scale it by positionScale).
//
// Per iteration only its input activation and rigid are kept; each iteration is replayed before its backward. The
// pair's LayerNorm and its attention_2d bias are made once and shared by all eight, as in the forward.
// stopRotGradient: AlphaFold 2's own training stops the rotation's gradient between iterations; af3-any-model's
// folding.py, which the port follows, does not (the stop_gradient there is commented out) - off by default.
#pragma once
#include "ops.cuh"

namespace grad {
namespace af2 {

// ---------------------------------------------------------------- rigid helpers (rotation row-major, then t)
__device__ inline void rApply(const real* r, const real* p, real* o) {
  for (int a = 0; a < 3; ++a) o[a] = r[a * 3] * p[0] + r[a * 3 + 1] * p[1] + r[a * 3 + 2] * p[2] + r[9 + a];
}
__device__ inline void rCompose(const real* A, const real* B, real* C) {        // A @ B
  real R[9], t[3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  rApply(A, B + 9, t);
  for (int k = 0; k < 9; ++k) C[k] = R[k];
  for (int k = 0; k < 3; ++k) C[9 + k] = t[k];
}
// C = A @ B: dC -> dA, dB (added)
__device__ inline void rComposeBwd(const real* A, const real* B, const real* dC, real* dA, real* dB) {
  // CR = AR BR, Ct = AR Bt + At
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      real s = 0, u = 0;
      for (int k = 0; k < 3; ++k) { s += dC[i * 3 + k] * B[j * 3 + k]; u += A[k * 3 + i] * dC[k * 3 + j]; }
      dA[i * 3 + j] += s + dC[9 + i] * B[9 + j];
      dB[i * 3 + j] += u;
    }
  for (int i = 0; i < 3; ++i) {
    dA[9 + i] += dC[9 + i];
    real u = 0; for (int k = 0; k < 3; ++k) u += A[k * 3 + i] * dC[9 + k];
    dB[9 + i] += u;
  }
}
__global__ void identityRigidK(real* r, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  for (int k = 0; k < 12; ++k) r[i * 12 + k] = (k == 0 || k == 4 || k == 8) ? 1 : 0;
}

// ---------------------------------------------------------------- IPA kernels
// local [L, H, 3P] (x[0:P] y[P:2P] z[2P:3P] a head) -> global [L, H, P, 3]
__global__ void toGlobalK(const real* loc, const real* rig, real* g, int L, int H, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * H * P) return;
  int p = t % P; size_t r = t / P; int h = r % H; int i = r / H;
  const real* s = loc + ((size_t)i * H + h) * 3 * P;
  real l[3] = {s[p], s[P + p], s[2 * P + p]}, o[3];
  rApply(rig + i * 12, l, o);
  for (int a = 0; a < 3; ++a) g[t * 3 + a] = o[a];
}
// dg -> dloc (written), drig (added, atomically: every point of a residue)
__global__ void toGlobalBwdK(const real* loc, const real* rig, const real* dg, real* dloc, real* drig, int L, int H, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * H * P) return;
  int p = t % P; size_t r = t / P; int h = r % H; int i = r / H;
  const real* s = loc + ((size_t)i * H + h) * 3 * P;
  real* ds = dloc + ((size_t)i * H + h) * 3 * P;
  const real* R = rig + i * 12;
  real l[3] = {s[p], s[P + p], s[2 * P + p]}, d[3] = {dg[t * 3], dg[t * 3 + 1], dg[t * 3 + 2]};
  for (int a = 0; a < 3; ++a) ds[a * P + p] = R[a] * d[0] + R[3 + a] * d[1] + R[6 + a] * d[2];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) atomicAdd(&drig[i * 12 + a * 3 + b], d[a] * l[b]);
    atomicAdd(&drig[i * 12 + 9 + a], d[a]);
  }
}
// logits [q, h, k] = sqrt(1/3) (qs.ks - w_h/2 |qp - kp|^2 + b2d[q, k, h] - 1e5 (1 - m_q m_k)), qs already scaled
__global__ void ipaLogitsK(const real* qs, const real* ks, const real* qp, const real* kp, const real* b2d, const real* w,
                           const real* mask, real* lg, int L, int H, int Cs, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * H * L) return;
  int k = t % L; size_t r = t / L; int h = r % H; int q = r / H;
  real sc = 0, d2 = 0;
  for (int c = 0; c < Cs; ++c) sc += qs[((size_t)q * H + h) * Cs + c] * ks[((size_t)k * H + h) * Cs + c];
  for (int j = 0; j < 3 * P; ++j) { real d = qp[((size_t)q * H + h) * 3 * P + j] - kp[((size_t)k * H + h) * 3 * P + j]; d2 += d * d; }
  real s = sc - (real)0.5 * w[h] * d2 + b2d[((size_t)q * L + k) * H + h] - (real)1e5 * (1 - mask[q] * mask[k]);
  lg[t] = s * sqrt((real)1 / 3);
}
// one block a row: softmax over the last axis, in place
__global__ void rowSoftmaxK(real* x, int n) {
  real* r = x + (size_t)blockIdx.x * n;
  __shared__ real red[32];
  real mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmax(mx, r[j]);
  for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x == 0) { real v = red[0]; for (int w = 1; w < (int)blockDim.x / 32; ++w) v = fmax(v, red[w]); red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  real s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) { real e = exp(r[j] - mx); r[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x == 0) { real v = 0; for (int w = 0; w < (int)blockDim.x / 32; ++w) v += red[w]; red[0] = v; }
  __syncthreads();
  real inv = 1 / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x) r[j] *= inv;
}
// dlogit = a (da - sum_k a da), one block a row, in place over da
__global__ void rowSoftmaxBwdK(const real* a, real* d, int n) {
  const real* ar = a + (size_t)blockIdx.x * n; real* dr = d + (size_t)blockIdx.x * n;
  __shared__ real red[32];
  real s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) s += ar[j] * dr[j];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x == 0) { real v = 0; for (int w = 0; w < (int)blockDim.x / 32; ++w) v += red[w]; red[0] = v; }
  __syncthreads();
  s = red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x) dr[j] = ar[j] * (dr[j] - s);
}
// the logits' backward, by query: dS = sqrt(1/3) dlogit -> dqs, dqp (written), db2d (added), dw (added)
__global__ void ipaLogitsBwdQK(const real* dl, const real* qs, const real* ks, const real* qp, const real* kp, const real* w,
                               real* dqs, real* dqp, real* db2d, real* dw, int L, int H, int Cs, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * H) return;
  int h = t % H, q = t / H;
  const real c3 = sqrt((real)1 / 3);
  real* gq = dqs + t * Cs; real* gp = dqp + t * 3 * P;
  for (int c = 0; c < Cs; ++c) gq[c] = 0;
  for (int j = 0; j < 3 * P; ++j) gp[j] = 0;
  real gw = 0;
  for (int k = 0; k < L; ++k) {
    real s = dl[t * L + k] * c3;
    db2d[((size_t)q * L + k) * H + h] += s;
    const real* kk = ks + ((size_t)k * H + h) * Cs;
    for (int c = 0; c < Cs; ++c) gq[c] += s * kk[c];
    const real* a = qp + t * 3 * P; const real* b = kp + ((size_t)k * H + h) * 3 * P;
    real d2 = 0;
    for (int j = 0; j < 3 * P; ++j) { real d = a[j] - b[j]; d2 += d * d; gp[j] -= s * w[h] * d; }
    gw -= (real)0.5 * s * d2;
  }
  atomicAdd(&dw[h], gw);
}
// ...and by key: dks, dkp (written)
__global__ void ipaLogitsBwdKK(const real* dl, const real* qs, const real* qp, const real* kp, const real* w,
                               real* dks, real* dkp, int L, int H, int Cs, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * H) return;
  int h = t % H, k = t / H;
  const real c3 = sqrt((real)1 / 3);
  real* gk = dks + t * Cs; real* gp = dkp + t * 3 * P;
  for (int c = 0; c < Cs; ++c) gk[c] = 0;
  for (int j = 0; j < 3 * P; ++j) gp[j] = 0;
  for (int q = 0; q < L; ++q) {
    real s = dl[((size_t)q * H + h) * L + k] * c3;
    const real* qq = qs + ((size_t)q * H + h) * Cs;
    for (int c = 0; c < Cs; ++c) gk[c] += s * qq[c];
    const real* a = qp + ((size_t)q * H + h) * 3 * P; const real* b = kp + t * 3 * P;
    for (int j = 0; j < 3 * P; ++j) gp[j] += s * w[h] * (a[j] - b[j]);
  }
}
// o[q, h, j] = sum_k a[q, h, k] V[k, h, j], V [L, H, J]
__global__ void attnApplyK(const real* a, const real* V, real* o, int L, int H, int J) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * H * J) return;
  int j = t % J; size_t r = t / J; int h = r % H;
  const real* ar = a + r * L;
  real s = 0;
  for (int k = 0; k < L; ++k) s += ar[k] * V[((size_t)k * H + h) * J + j];
  o[t] = s;
}
// dV[k, h, j] = sum_q a[q, h, k] dO[q, h, j] (written); da[q, h, k] += sum_j dO[q, h, j] V[k, h, j]
__global__ void attnApplyBwdVK(const real* a, const real* dO, real* dV, int L, int H, int J) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * H * J) return;
  int j = t % J; size_t r = t / J; int h = r % H, k = r / H;
  real s = 0;
  for (int q = 0; q < L; ++q) s += a[((size_t)q * H + h) * L + k] * dO[((size_t)q * H + h) * J + j];
  dV[t] = s;
}
__global__ void attnApplyBwdAK(const real* dO, const real* V, real* da, int L, int H, int J) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * H * L) return;
  int k = t % L; size_t r = t / L; int h = r % H;
  real s = 0;
  for (int j = 0; j < J; ++j) s += dO[r * J + j] * V[((size_t)k * H + h) * J + j];
  da[t] += s;
}
// V = [vs (Cs) | vp global (P, 3)] a head
__global__ void packVK(const real* vs, const real* vg, real* V, int L, int H, int Cs, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int J = Cs + 3 * P;
  if (t >= (size_t)L * H * J) return;
  int j = t % J; size_t r = t / J;
  V[t] = j < Cs ? vs[r * Cs + j] : vg[r * 3 * P + (j - Cs)];
}
__global__ void unpackVK(const real* dV, real* dvs, real* dvg, int L, int H, int Cs, int P) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int J = Cs + 3 * P;
  if (t >= (size_t)L * H * J) return;
  int j = t % J; size_t r = t / J;
  if (j < Cs) dvs[r * Cs + j] = dV[t]; else dvg[r * 3 * P + (j - Cs)] = dV[t];
}
// fin [q] = [os (H Cs) | lx (H P) | ly | lz | |l| | o2d (H C2)], the points back in q's frame
__global__ void ipaFinalK(const real* O, const real* o2d, const real* rig, real* fin, int L, int H, int Cs, int P, int C2) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int J = Cs + 3 * P, Fw = H * Cs + 4 * H * P + H * C2;
  if (t >= (size_t)L * H) return;
  int h = t % H, q = t / H;
  real* f = fin + (size_t)q * Fw;
  const real* o = O + t * J;
  for (int c = 0; c < Cs; ++c) f[h * Cs + c] = o[c];
  const real* R = rig + q * 12;
  for (int p = 0; p < P; ++p) {
    real d[3] = {o[Cs + p * 3] - R[9], o[Cs + p * 3 + 1] - R[10], o[Cs + p * 3 + 2] - R[11]}, l[3];
    for (int a = 0; a < 3; ++a) l[a] = R[a] * d[0] + R[3 + a] * d[1] + R[6 + a] * d[2];
    const int base = H * Cs, idx = h * P + p;
    f[base + idx] = l[0]; f[base + H * P + idx] = l[1]; f[base + 2 * H * P + idx] = l[2];
    f[base + 3 * H * P + idx] = sqrt(fmax(l[0] * l[0] + l[1] * l[1] + l[2] * l[2], (real)1e-16));
  }
  for (int c = 0; c < C2; ++c) f[H * Cs + 4 * H * P + h * C2 + c] = o2d[((size_t)q * H + h) * C2 + c];
}
// dfin -> dO, do2d (written), drig (added)
__global__ void ipaFinalBwdK(const real* O, const real* rig, const real* dfin, real* dO, real* do2d, real* drig,
                             int L, int H, int Cs, int P, int C2) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int J = Cs + 3 * P, Fw = H * Cs + 4 * H * P + H * C2;
  if (t >= (size_t)L * H) return;
  int h = t % H, q = t / H;
  const real* f = dfin + (size_t)q * Fw;
  const real* o = O + t * J;
  real* go = dO + t * J;
  for (int c = 0; c < Cs; ++c) go[c] = f[h * Cs + c];
  const real* R = rig + q * 12;
  for (int p = 0; p < P; ++p) {
    real d[3] = {o[Cs + p * 3] - R[9], o[Cs + p * 3 + 1] - R[10], o[Cs + p * 3 + 2] - R[11]}, l[3];
    for (int a = 0; a < 3; ++a) l[a] = R[a] * d[0] + R[3 + a] * d[1] + R[6 + a] * d[2];
    const int base = H * Cs, idx = h * P + p;
    real dl[3] = {f[base + idx], f[base + H * P + idx], f[base + 2 * H * P + idx]};
    real n2 = l[0] * l[0] + l[1] * l[1] + l[2] * l[2];
    if (n2 > (real)1e-16) { real dn = f[base + 3 * H * P + idx] / sqrt(n2); for (int a = 0; a < 3; ++a) dl[a] += dn * l[a]; }
    // l = R^T (g - t): dg = R dl, dt = -R dl, dR[b][a] += dl[a] d[b]
    for (int b = 0; b < 3; ++b) {
      real g = R[b * 3] * dl[0] + R[b * 3 + 1] * dl[1] + R[b * 3 + 2] * dl[2];
      go[Cs + p * 3 + b] = g;
      atomicAdd(&drig[q * 12 + 9 + b], -g);
      for (int a = 0; a < 3; ++a) atomicAdd(&drig[q * 12 + b * 3 + a], dl[a] * d[b]);
    }
  }
  for (int c = 0; c < C2; ++c) do2d[((size_t)q * H + h) * C2 + c] = f[H * Cs + 4 * H * P + h * C2 + c];
}
__global__ void softplusScaleK(const real* raw, real* out, int H, real base) {
  int h = threadIdx.x;
  if (h < H) out[h] = base * log1p(exp(raw[h]));
}
__global__ void softplusScaleBwdK(const real* raw, const real* dw, real* draw, int H, real base) {
  int h = threadIdx.x;
  if (h < H) draw[h] += dw[h] * base / (1 + exp(-raw[h]));
}
// the backbone update: (1, x, y, z) normalised -> rotation, (tx, ty, tz); rig' = rig @ update
__device__ inline void quatRigid(const real* u, real* U, real* qn) {
  real w = 1, x = u[0], y = u[1], z = u[2];
  real inv = 1 / sqrt(fmax((real)1e-6, w * w + x * x + y * y + z * z));
  w *= inv; x *= inv; y *= inv; z *= inv;
  qn[0] = w; qn[1] = x; qn[2] = y; qn[3] = z;
  real R[12] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), u[3], u[4], u[5]};
  for (int k = 0; k < 12; ++k) U[k] = R[k];
}
__global__ void rigidUpdateK(const real* rig, const real* u, real* out, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  real U[12], qn[4];
  quatRigid(u + i * 6, U, qn);
  rCompose(rig + i * 12, U, out + i * 12);
}
// dout -> drig (added), du (written)
__global__ void rigidUpdateBwdK(const real* rig, const real* u, const real* dout, real* drig, real* du, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  real U[12], qn[4], dU[12] = {0};
  quatRigid(u + i * 6, U, qn);
  real dR[12] = {0};
  rComposeBwd(rig + i * 12, U, dout + i * 12, dR, dU);
  for (int k = 0; k < 12; ++k) drig[i * 12 + k] += dR[k];
  const real w = qn[0], x = qn[1], y = qn[2], z = qn[3];
  const real* G = dU;
  real dq[4];
  dq[0] = 2 * (-z * G[1] + y * G[2] + z * G[3] - x * G[5] - y * G[6] + x * G[7]);
  dq[1] = 2 * (y * G[1] + z * G[2] + y * G[3] - 2 * x * G[4] - w * G[5] + z * G[6] + w * G[7] - 2 * x * G[8]);
  dq[2] = 2 * (-2 * y * G[0] + x * G[1] + w * G[2] + x * G[3] + z * G[5] - w * G[6] + z * G[7] - 2 * y * G[8]);
  dq[3] = 2 * (-2 * z * G[0] - w * G[1] + x * G[2] + w * G[3] - 2 * z * G[4] + y * G[5] + x * G[6] + y * G[7]);
  const real* r = u + i * 6;
  real n2 = 1 + r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
  real inv = 1 / sqrt(fmax((real)1e-6, n2));
  real dot = qn[0] * dq[0] + qn[1] * dq[1] + qn[2] * dq[2] + qn[3] * dq[3];
  for (int a = 0; a < 3; ++a) du[i * 6 + a] = n2 > (real)1e-6 ? inv * (dq[a + 1] - qn[a + 1] * dot) : inv * dq[a + 1];
  for (int a = 0; a < 3; ++a) du[i * 6 + 3 + a] = dU[9 + a];
}
// traj = rig with its translation scaled; and back
__global__ void scaleRigidK(const real* rig, real* out, int L, real s, bool accumulate) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= L * 12) return;
  real v = rig[t] * ((t % 12) >= 9 ? s : (real)1);
  out[t] = accumulate ? out[t] + v : v;
}
__global__ void zeroRotationK(real* r, int L) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < L * 12 && (t % 12) < 9) r[t] = 0;
}

// ---------------------------------------------------------------- the side chain: angles -> frames -> atom14
struct ResidueTables { const real *defaultFrames, *litPos, *atom14Mask, *atom37Mask; const int *atom14Group, *atom37To14, *aatype; };
__device__ inline void groupFrames(const ResidueTables& T, int aa, const real* sinA, const real* cosA, real cur[8][12]) {
  for (int g = 0; g < 8; ++g) {
    const real* m = T.defaultFrames + ((size_t)aa * 8 + g) * 16;
    real D[12] = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10], m[3], m[7], m[11]};
    real Rx[12] = {1, 0, 0, 0, cosA[g], -sinA[g], 0, sinA[g], cosA[g], 0, 0, 0};
    real R[12];
    rCompose(D, Rx, R);
    for (int k = 9; k < 12; ++k) R[k] = D[k];
    if (g >= 5) rCompose(cur[g - 1], R, cur[g]);
    else for (int k = 0; k < 12; ++k) cur[g][k] = R[k];
  }
}
__device__ inline void normAngles(const real* un, real* sinA, real* cosA, real* angles) {
  sinA[0] = 0; cosA[0] = 1;
  for (int a = 0; a < 7; ++a) {
    real s = un[a * 2], c = un[a * 2 + 1];
    real inv = 1 / sqrt(fmax(s * s + c * c, (real)1e-12));
    s *= inv; c *= inv;
    if (angles) { angles[a * 2] = s; angles[a * 2 + 1] = c; }
    sinA[a + 1] = s; cosA[a + 1] = c;
  }
}
__global__ void sidechainK(const real* un, const real* rig, real scale, ResidueTables T, const real* seqMask, real* angles,
                           real* pos14, real* pos37, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aa = min(max(T.aatype[i], 0), 19);
  real bb[12];
  for (int k = 0; k < 12; ++k) bb[k] = rig[i * 12 + k] * (k >= 9 ? scale : (real)1);
  real sinA[8], cosA[8], cur[8][12];
  normAngles(un + i * 14, sinA, cosA, angles + i * 14);
  groupFrames(T, aa, sinA, cosA, cur);
  for (int g = 0; g < 8; ++g) {
    real glob[12];
    rCompose(bb, cur[g], glob);
    for (int a = 0; a < 14; ++a) {
      if (T.atom14Group[aa * 14 + a] != g) continue;
      real p[3];
      rApply(glob, T.litPos + ((size_t)aa * 14 + a) * 3, p);
      for (int x = 0; x < 3; ++x) pos14[((size_t)i * 14 + a) * 3 + x] = p[x] * T.atom14Mask[aa * 14 + a];
    }
  }
  for (int a = 0; a < 37; ++a) {
    int s = T.atom37To14[aa * 37 + a];
    real m = T.atom37Mask[aa * 37 + a] * seqMask[i];
    for (int x = 0; x < 3; ++x) pos37[((size_t)i * 37 + a) * 3 + x] = pos14[((size_t)i * 14 + s) * 3 + x] * m;
  }
}
// d angles (normalised), d pos14, d pos37 (any null) -> dun (written), drig (added)
__global__ void sidechainBwdK(const real* un, const real* rig, real scale, ResidueTables T, const real* seqMask,
                              const real* dangles, const real* dpos14, const real* dpos37, real* dun, real* drig, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aa = min(max(T.aatype[i], 0), 19);
  real bb[12];
  for (int k = 0; k < 12; ++k) bb[k] = rig[i * 12 + k] * (k >= 9 ? scale : (real)1);
  real sinA[8], cosA[8], cur[8][12];
  normAngles(un + i * 14, sinA, cosA, nullptr);
  groupFrames(T, aa, sinA, cosA, cur);
  real d14[14][3];
  for (int a = 0; a < 14; ++a) for (int x = 0; x < 3; ++x) d14[a][x] = dpos14 ? dpos14[((size_t)i * 14 + a) * 3 + x] : 0;
  if (dpos37)
    for (int a = 0; a < 37; ++a) {
      int s = T.atom37To14[aa * 37 + a];
      real m = T.atom37Mask[aa * 37 + a] * seqMask[i];
      for (int x = 0; x < 3; ++x) d14[s][x] += dpos37[((size_t)i * 37 + a) * 3 + x] * m;
    }
  real dbb[12] = {0}, dcur[8][12] = {{0}};
  for (int g = 0; g < 8; ++g) {
    real dglob[12] = {0};
    for (int a = 0; a < 14; ++a) {
      if (T.atom14Group[aa * 14 + a] != g) continue;
      const real* lp = T.litPos + ((size_t)aa * 14 + a) * 3;
      real mk = T.atom14Mask[aa * 14 + a];
      for (int x = 0; x < 3; ++x) {
        real dp = d14[a][x] * mk;
        for (int y = 0; y < 3; ++y) dglob[x * 3 + y] += dp * lp[y];
        dglob[9 + x] += dp;
      }
    }
    rComposeBwd(bb, cur[g], dglob, dbb, dcur[g]);
  }
  real dsin[8] = {0}, dcos[8] = {0};
  for (int g = 7; g >= 0; --g) {
    const real* m = T.defaultFrames + ((size_t)aa * 8 + g) * 16;
    real D[12] = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10], m[3], m[7], m[11]};
    real Rx[12] = {1, 0, 0, 0, cosA[g], -sinA[g], 0, sinA[g], cosA[g], 0, 0, 0};
    real R[12];
    rCompose(D, Rx, R);
    for (int k = 9; k < 12; ++k) R[k] = D[k];
    real dR[12] = {0};
    if (g >= 5) rComposeBwd(cur[g - 1], R, dcur[g], dcur[g - 1], dR);
    else for (int k = 0; k < 12; ++k) dR[k] = dcur[g][k];
    // R's rotation = D_R Rx: dRx = D_R^T dR_R (its translation is D's, constant)
    real dRx[9];
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
      real s = 0; for (int k = 0; k < 3; ++k) s += D[k * 3 + a] * dR[k * 3 + b];
      dRx[a * 3 + b] = s;
    }
    dcos[g] += dRx[4] + dRx[8];
    dsin[g] += -dRx[5] + dRx[7];
  }
  for (int a = 0; a < 7; ++a) {
    real s = un[i * 14 + a * 2], c = un[i * 14 + a * 2 + 1];
    real n2 = s * s + c * c, r = sqrt(fmax(n2, (real)1e-12));
    real sn = s / r, cn = c / r;
    real ds = dsin[a + 1] + (dangles ? dangles[i * 14 + a * 2] : 0), dc = dcos[a + 1] + (dangles ? dangles[i * 14 + a * 2 + 1] : 0);
    if (n2 > (real)1e-12) {
      real dot = sn * ds + cn * dc;
      dun[i * 14 + a * 2] = (ds - sn * dot) / r; dun[i * 14 + a * 2 + 1] = (dc - cn * dot) / r;
    } else { dun[i * 14 + a * 2] = ds / r; dun[i * 14 + a * 2 + 1] = dc / r; }
  }
  for (int k = 0; k < 12; ++k) drig[i * 12 + k] += dbb[k] * (k >= 9 ? scale : (real)1);
}

// ---------------------------------------------------------------- the module
struct StructureModule {
  int C = 384, C2 = 128, H = 12, Cs = 16, Pq = 4, Pv = 8, iterations = 8, Csc = 128;
  real positionScale = 10;
  bool stopRotGradient = false;
  LayerNorm singleLn, pairLn, attnLn, transLn;
  Linear initial, qs, ks, vs, qp, kp, vp, b2d, out, tr0, tr1, tr2, quat;
  Param pointWeights;
  Linear scIn, scIn1, res1[2], res2[2], angles;
  int Fw() const { return H * Cs + 4 * H * Pv + H * C2; }
};
inline void forEachParam(StructureModule& m, const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string S = "structure_module/", F = S + "fold_iteration/", I = F + "invariant_point_attention/",
                    R = F + "rigid_sidechain/";
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(n + "/scale", l.g, C); f(n + "/offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) {
    l.K = K; l.N = N; l.hasBias = true; f(n + "/weights", l.w, (size_t)K * N); f(n + "/bias", l.b, N);
  };
  const int C = m.C, H = m.H;
  ln(S + "single_layer_norm", m.singleLn, C); lin(S + "initial_projection", m.initial, C, C);
  ln(S + "pair_layer_norm", m.pairLn, m.C2);
  lin(I + "q_scalar_projection", m.qs, C, H * m.Cs); lin(I + "k_scalar_projection", m.ks, C, H * m.Cs);
  lin(I + "v_scalar_projection", m.vs, C, H * m.Cs);
  lin(I + "q_point_projection/point_projection", m.qp, C, H * 3 * m.Pq);
  lin(I + "k_point_projection/point_projection", m.kp, C, H * 3 * m.Pq);
  lin(I + "v_point_projection/point_projection", m.vp, C, H * 3 * m.Pv);
  f(I + "trainable_point_weights", m.pointWeights, H);
  lin(I + "attention_2d", m.b2d, m.C2, H);
  lin(I + "output_projection", m.out, m.Fw(), C);
  ln(F + "attention_layer_norm", m.attnLn, C);
  lin(F + "transition", m.tr0, C, C); lin(F + "transition_1", m.tr1, C, C); lin(F + "transition_2", m.tr2, C, C);
  ln(F + "transition_layer_norm", m.transLn, C);
  lin(F + "quat_rigid/rigid", m.quat, C, 6);
  lin(R + "input_projection", m.scIn, C, m.Csc); lin(R + "input_projection_1", m.scIn1, C, m.Csc);
  for (int r = 0; r < 2; ++r) {
    const std::string sfx = r == 0 ? "" : "_1";
    lin(R + "resblock1" + sfx, m.res1[r], m.Csc, m.Csc); lin(R + "resblock2" + sfx, m.res2[r], m.Csc, m.Csc);
  }
  lin(R + "unnormalized_angles", m.angles, m.Csc, 14);
}

// one iteration's saves (replayed before its backward)
struct IterSave {
  real *qs, *ks, *vs, *qpl, *kpl, *vpl, *qg, *kg, *vg, *V, *attn, *O, *o2d, *fin, *upd, *x1, *t1, *h0, *h1, *x2, *act, *u, *rig;
  LnSave la, lt;
};
// everything a call keeps for its backward: the shared part, each iteration's input, the side chain's
struct SmRun {
  real *initial, *act0, *act2d, *b2d, *w;
  LnSave ls, lp;
  std::vector<real*> acts, rigs;      // [iterations + 1] each: an iteration's input (and the last one's output)
  real *traj;                          // [iterations, L, 12] the rigids with translations scaled
  real *ra, *ri, *sc0, *sb[3], *sc[3], *r1[2], *un, *angles, *pos14, *pos37;
  int L = 0;
};
inline IterSave iterForward(StructureModule& m, const SmRun& run, const real* act, const real* rig, const real* mask, int L) {
  IterSave s{};
  const int H = m.H, Cs = m.Cs, Pq = m.Pq, Pv = m.Pv, C = m.C, J = Cs + 3 * Pv;
  s.qs = ARENA.get((size_t)L * H * Cs); m.qs.forward(act, s.qs, L);
  scaleK<<<blocks((size_t)L * H * Cs), 256, 0, STREAM>>>(s.qs, sqrt((real)1 / Cs), (size_t)L * H * Cs);
  s.ks = ARENA.get((size_t)L * H * Cs); m.ks.forward(act, s.ks, L);
  s.vs = ARENA.get((size_t)L * H * Cs); m.vs.forward(act, s.vs, L);
  s.qpl = ARENA.get((size_t)L * H * 3 * Pq); m.qp.forward(act, s.qpl, L);
  s.kpl = ARENA.get((size_t)L * H * 3 * Pq); m.kp.forward(act, s.kpl, L);
  s.vpl = ARENA.get((size_t)L * H * 3 * Pv); m.vp.forward(act, s.vpl, L);
  s.qg = ARENA.get((size_t)L * H * 3 * Pq); toGlobalK<<<blocks((size_t)L * H * Pq), 256, 0, STREAM>>>(s.qpl, rig, s.qg, L, H, Pq);
  s.kg = ARENA.get((size_t)L * H * 3 * Pq); toGlobalK<<<blocks((size_t)L * H * Pq), 256, 0, STREAM>>>(s.kpl, rig, s.kg, L, H, Pq);
  s.vg = ARENA.get((size_t)L * H * 3 * Pv); toGlobalK<<<blocks((size_t)L * H * Pv), 256, 0, STREAM>>>(s.vpl, rig, s.vg, L, H, Pv);
  s.attn = ARENA.get((size_t)L * H * L);
  ipaLogitsK<<<blocks((size_t)L * H * L), 256, 0, STREAM>>>(s.qs, s.ks, s.qg, s.kg, run.b2d, run.w, mask, s.attn, L, H, Cs, Pq);
  rowSoftmaxK<<<(unsigned)((size_t)L * H), 128, 0, STREAM>>>(s.attn, L);
  s.V = ARENA.get((size_t)L * H * J); packVK<<<blocks((size_t)L * H * J), 256, 0, STREAM>>>(s.vs, s.vg, s.V, L, H, Cs, Pv);
  s.O = ARENA.get((size_t)L * H * J); attnApplyK<<<blocks((size_t)L * H * J), 256, 0, STREAM>>>(s.attn, s.V, s.O, L, H, J);
  s.o2d = ARENA.get((size_t)L * H * m.C2);
  bmm(false, false, H, m.C2, L, s.attn, (long long)H * L, run.act2d, (long long)L * m.C2, s.o2d, (long long)H * m.C2, L);
  s.fin = ARENA.get((size_t)L * m.Fw());
  ipaFinalK<<<blocks((size_t)L * H), 256, 0, STREAM>>>(s.O, s.o2d, rig, s.fin, L, H, Cs, Pv, m.C2);
  s.upd = ARENA.get((size_t)L * C); m.out.forward(s.fin, s.upd, L);
  s.x1 = cloneR(act, (size_t)L * C); add(s.x1, s.upd, (size_t)L * C);
  s.t1 = ARENA.get((size_t)L * C); s.la = lnForward(m.attnLn, s.x1, s.t1, L);
  s.h0 = ARENA.get((size_t)L * C); m.tr0.forward(s.t1, s.h0, L);
  real* r0 = ARENA.get((size_t)L * C); reluK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(r0, s.h0, (size_t)L * C);
  s.h1 = ARENA.get((size_t)L * C); m.tr1.forward(r0, s.h1, L);
  real* r1 = ARENA.get((size_t)L * C); reluK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(r1, s.h1, (size_t)L * C);
  s.x2 = cloneR(s.t1, (size_t)L * C); m.tr2.forward(r1, s.x2, L, 1);
  s.act = ARENA.get((size_t)L * C); s.lt = lnForward(m.transLn, s.x2, s.act, L);
  s.u = ARENA.get((size_t)L * 6); m.quat.forward(s.act, s.u, L);
  s.rig = ARENA.get((size_t)L * 12); rigidUpdateK<<<blocks(L), 128, 0, STREAM>>>(rig, s.u, s.rig, L);
  return s;
}
// d act', d rig' -> d act, d rig (written); d act2d, d b2d, d w (added)
inline void iterBackward(StructureModule& m, const SmRun& run, const IterSave& s, const real* act, const real* rig,
                         const real* dactOut, const real* drigOut, real* dact, real* drig, real* dact2d, real* db2d, real* dw, int L) {
  Scratch scratch;
  const int H = m.H, Cs = m.Cs, Pq = m.Pq, Pv = m.Pv, C = m.C, J = Cs + 3 * Pv;
  GK(cudaMemsetAsync(drig, 0, (size_t)L * 12 * sizeof(real), STREAM));
  real* du = ARENA.get((size_t)L * 6);
  rigidUpdateBwdK<<<blocks(L), 128, 0, STREAM>>>(rig, s.u, drigOut, drig, du, L);
  real* dA = cloneR(dactOut, (size_t)L * C);
  m.quat.backward(s.act, du, dA, L, true);
  real* dx2 = ARENA.get((size_t)L * C); lnBackward(m.transLn, s.lt, dA, dx2, L, false);
  real* dt1 = cloneR(dx2, (size_t)L * C);
  {
    real *r1 = ARENA.get((size_t)L * C), *dr1 = ARENA.get((size_t)L * C), *r0 = ARENA.get((size_t)L * C), *dr0 = ARENA.get((size_t)L * C);
    reluK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(r1, s.h1, (size_t)L * C);
    m.tr2.backward(r1, dx2, dr1, L);
    reluBwdK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(dr1, s.h1, (size_t)L * C);
    reluK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(r0, s.h0, (size_t)L * C);
    m.tr1.backward(r0, dr1, dr0, L);
    reluBwdK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(dr0, s.h0, (size_t)L * C);
    m.tr0.backward(s.t1, dr0, dt1, L, true);
  }
  real* dx1 = ARENA.get((size_t)L * C); lnBackward(m.attnLn, s.la, dt1, dx1, L, false);
  GK(cudaMemcpyAsync(dact, dx1, (size_t)L * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));   // the residual
  real* dfin = ARENA.get((size_t)L * m.Fw()); m.out.backward(s.fin, dx1, dfin, L);
  real *dO = ARENA.get((size_t)L * H * J), *do2d = ARENA.get((size_t)L * H * m.C2);
  ipaFinalBwdK<<<blocks((size_t)L * H), 256, 0, STREAM>>>(s.O, rig, dfin, dO, do2d, drig, L, H, Cs, Pv, m.C2);
  real* da = ARENA.zeros((size_t)L * H * L);
  // o2d = a act2d (by query): da += do2d act2d^T, dact2d += a^T do2d
  bmm(false, true, H, L, m.C2, do2d, (long long)H * m.C2, run.act2d, (long long)L * m.C2, da, (long long)H * L, L, 1);
  bmm(true, false, L, m.C2, H, s.attn, (long long)H * L, do2d, (long long)H * m.C2, dact2d, (long long)L * m.C2, L, 1);
  attnApplyBwdAK<<<blocks((size_t)L * H * L), 256, 0, STREAM>>>(dO, s.V, da, L, H, J);
  real* dV = ARENA.get((size_t)L * H * J); attnApplyBwdVK<<<blocks((size_t)L * H * J), 256, 0, STREAM>>>(s.attn, dO, dV, L, H, J);
  real *dvs = ARENA.get((size_t)L * H * Cs), *dvg = ARENA.get((size_t)L * H * 3 * Pv);
  unpackVK<<<blocks((size_t)L * H * J), 256, 0, STREAM>>>(dV, dvs, dvg, L, H, Cs, Pv);
  rowSoftmaxBwdK<<<(unsigned)((size_t)L * H), 128, 0, STREAM>>>(s.attn, da, L);
  real *dqs = ARENA.get((size_t)L * H * Cs), *dks = ARENA.get((size_t)L * H * Cs);
  real *dqg = ARENA.get((size_t)L * H * 3 * Pq), *dkg = ARENA.get((size_t)L * H * 3 * Pq);
  ipaLogitsBwdQK<<<blocks((size_t)L * H, 128), 128, 0, STREAM>>>(da, s.qs, s.ks, s.qg, s.kg, run.w, dqs, dqg, db2d, dw, L, H, Cs, Pq);
  ipaLogitsBwdKK<<<blocks((size_t)L * H, 128), 128, 0, STREAM>>>(da, s.qs, s.qg, s.kg, run.w, dks, dkg, L, H, Cs, Pq);
  real *dqpl = ARENA.get((size_t)L * H * 3 * Pq), *dkpl = ARENA.get((size_t)L * H * 3 * Pq), *dvpl = ARENA.get((size_t)L * H * 3 * Pv);
  toGlobalBwdK<<<blocks((size_t)L * H * Pq), 256, 0, STREAM>>>(s.qpl, rig, dqg, dqpl, drig, L, H, Pq);
  toGlobalBwdK<<<blocks((size_t)L * H * Pq), 256, 0, STREAM>>>(s.kpl, rig, dkg, dkpl, drig, L, H, Pq);
  toGlobalBwdK<<<blocks((size_t)L * H * Pv), 256, 0, STREAM>>>(s.vpl, rig, dvg, dvpl, drig, L, H, Pv);
  scaleK<<<blocks((size_t)L * H * Cs), 256, 0, STREAM>>>(dqs, sqrt((real)1 / Cs), (size_t)L * H * Cs);
  m.qs.backward(act, dqs, dact, L, true); m.ks.backward(act, dks, dact, L, true); m.vs.backward(act, dvs, dact, L, true);
  m.qp.backward(act, dqpl, dact, L, true); m.kp.backward(act, dkpl, dact, L, true); m.vp.backward(act, dvpl, dact, L, true);
}

// The whole module. single [L, C], pair [L*L, C2], seqMask [L], tables (aatype, ...). Writes run.traj, the last
// iteration's act, angles, pos14, pos37.
inline void smForward(StructureModule& m, SmRun& r, const real* single, const real* pair, const real* mask,
                      const ResidueTables& T, int L) {
  const int C = m.C, N = m.iterations; const size_t pairs = (size_t)L * L;
  r.L = L;
  r.initial = ARENA.get((size_t)L * C); r.ls = lnForward(m.singleLn, single, r.initial, L);
  r.act0 = ARENA.get((size_t)L * C); m.initial.forward(r.initial, r.act0, L);
  r.act2d = ARENA.get(pairs * m.C2); r.lp = lnForward(m.pairLn, pair, r.act2d, pairs);
  r.b2d = ARENA.get(pairs * m.H); m.b2d.forward(r.act2d, r.b2d, pairs);
  r.w = ARENA.get(m.H);
  softplusScaleK<<<1, 32, 0, STREAM>>>(m.pointWeights.w, r.w, m.H, sqrt((real)1 / (m.Pq * (real)4.5)));
  r.acts.assign(N + 1, nullptr); r.rigs.assign(N + 1, nullptr);
  for (int k = 0; k <= N; ++k) { r.acts[k] = ARENA.get((size_t)L * C); r.rigs[k] = ARENA.get((size_t)L * 12); }
  r.traj = ARENA.get((size_t)N * L * 12);
  GK(cudaMemcpyAsync(r.acts[0], r.act0, (size_t)L * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  identityRigidK<<<blocks(L), 128, 0, STREAM>>>(r.rigs[0], L);
  for (int k = 0; k < N; ++k) {
    Scratch sc;
    IterSave s = iterForward(m, r, r.acts[k], r.rigs[k], mask, L);
    GK(cudaMemcpyAsync(r.acts[k + 1], s.act, (size_t)L * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(r.rigs[k + 1], s.rig, (size_t)L * 12 * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    scaleRigidK<<<blocks((size_t)L * 12), 256, 0, STREAM>>>(s.rig, r.traj + (size_t)k * L * 12, L, m.positionScale, false);
  }
  // the side chain on the last iteration: sum of relu inputs' projections, two residual blocks, the angles
  const int S = m.Csc; const real* act = r.acts[N];
  r.ra = ARENA.get((size_t)L * C); reluK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(r.ra, act, (size_t)L * C);
  r.ri = ARENA.get((size_t)L * C); reluK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(r.ri, r.initial, (size_t)L * C);
  r.sc[0] = ARENA.get((size_t)L * S); m.scIn.forward(r.ra, r.sc[0], L); m.scIn1.forward(r.ri, r.sc[0], L, 1);
  for (int b = 0; b < 2; ++b) {
    r.sb[b] = ARENA.get((size_t)L * S); reluK<<<blocks((size_t)L * S), 256, 0, STREAM>>>(r.sb[b], r.sc[b], (size_t)L * S);
    real* h = ARENA.get((size_t)L * S); m.res1[b].forward(r.sb[b], h, L);
    r.r1[b] = ARENA.get((size_t)L * S); reluK<<<blocks((size_t)L * S), 256, 0, STREAM>>>(r.r1[b], h, (size_t)L * S);
    r.sc[b + 1] = cloneR(r.sc[b], (size_t)L * S); m.res2[b].forward(r.r1[b], r.sc[b + 1], L, 1);
  }
  r.sb[2] = ARENA.get((size_t)L * S); reluK<<<blocks((size_t)L * S), 256, 0, STREAM>>>(r.sb[2], r.sc[2], (size_t)L * S);
  r.un = ARENA.get((size_t)L * 14); m.angles.forward(r.sb[2], r.un, L);
  r.angles = ARENA.get((size_t)L * 14); r.pos14 = ARENA.get((size_t)L * 42); r.pos37 = ARENA.get((size_t)L * 111);
  sidechainK<<<blocks(L, 128), 128, 0, STREAM>>>(r.un, r.rigs[N], m.positionScale, T, mask, r.angles, r.pos14, r.pos37, L);
}
// cotangents (any null): dtraj [N, L, 12], dact [L, C] (the last act), dun [L, 14] (unnormalised angles),
// dangles [L, 14], dpos14 [L, 14, 3], dpos37 [L, 37, 3] -> every parameter; dsingle, dpair (added; may be null)
inline void smBackward(StructureModule& m, SmRun& r, const real* mask, const ResidueTables& T, const real* dtraj,
                       const real* dactLast, const real* dunIn, const real* dangles, const real* dpos14, const real* dpos37,
                       real* dsingle, real* dpair) {
  Scratch scratch;
  const int L = r.L, C = m.C, N = m.iterations, S = m.Csc; const size_t pairs = (size_t)L * L;
  real* dact = dactLast ? cloneR(dactLast, (size_t)L * C) : ARENA.zeros((size_t)L * C);
  real* drig = ARENA.zeros((size_t)L * 12);
  real* dinitial = ARENA.zeros((size_t)L * C);
  // the side chain
  {
    real* dun = ARENA.get((size_t)L * 14);
    sidechainBwdK<<<blocks(L, 128), 128, 0, STREAM>>>(r.un, r.rigs[N], m.positionScale, T, mask, dangles, dpos14, dpos37, dun, drig, L);
    if (dunIn) add(dun, dunIn, (size_t)L * 14);
    real* dsc = ARENA.get((size_t)L * S); m.angles.backward(r.sb[2], dun, dsc, L);
    reluBwdK<<<blocks((size_t)L * S), 256, 0, STREAM>>>(dsc, r.sc[2], (size_t)L * S);
    for (int b = 1; b >= 0; --b) {
      real* dr1 = ARENA.get((size_t)L * S); m.res2[b].backward(r.r1[b], dsc, dr1, L);
      real* h = ARENA.get((size_t)L * S); m.res1[b].forward(r.sb[b], h, L);
      reluBwdK<<<blocks((size_t)L * S), 256, 0, STREAM>>>(dr1, h, (size_t)L * S);
      real* dsb = ARENA.get((size_t)L * S); m.res1[b].backward(r.sb[b], dr1, dsb, L);
      reluBwdK<<<blocks((size_t)L * S), 256, 0, STREAM>>>(dsb, r.sc[b], (size_t)L * S);
      add(dsc, dsb, (size_t)L * S);            // sc[b + 1] = sc[b] + ...
    }
    real* dra = ARENA.get((size_t)L * C); m.scIn.backward(r.ra, dsc, dra, L);
    reluBwdK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(dra, r.acts[N], (size_t)L * C); add(dact, dra, (size_t)L * C);
    real* dri = ARENA.get((size_t)L * C); m.scIn1.backward(r.ri, dsc, dri, L);
    reluBwdK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(dri, r.initial, (size_t)L * C); add(dinitial, dri, (size_t)L * C);
  }
  real *dact2d = ARENA.zeros(pairs * m.C2), *db2d = ARENA.zeros(pairs * m.H), *dw = ARENA.zeros(m.H);
  real *dactPrev = ARENA.get((size_t)L * C), *drigPrev = ARENA.get((size_t)L * 12);
  for (int k = N; k-- > 0;) {
    Scratch sc;
    if (dtraj) scaleRigidK<<<blocks((size_t)L * 12), 256, 0, STREAM>>>(dtraj + (size_t)k * L * 12, drig, L, m.positionScale, true);
    IterSave s = iterForward(m, r, r.acts[k], r.rigs[k], mask, L);
    iterBackward(m, r, s, r.acts[k], r.rigs[k], dact, drig, dactPrev, drigPrev, dact2d, db2d, dw, L);
    GK(cudaMemcpyAsync(dact, dactPrev, (size_t)L * C * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(drig, drigPrev, (size_t)L * 12 * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    if (m.stopRotGradient) zeroRotationK<<<blocks((size_t)L * 12), 256, 0, STREAM>>>(drig, L);
  }
  // (the identity rigid at the start has no parameters: drig ends here)
  softplusScaleBwdK<<<1, 32, 0, STREAM>>>(m.pointWeights.w, dw, m.pointWeights.g, m.H, sqrt((real)1 / (m.Pq * (real)4.5)));
  m.b2d.backward(r.act2d, db2d, dact2d, pairs, true);
  if (dpair) lnBackward(m.pairLn, r.lp, dact2d, dpair, pairs, true);
  else { real* t = ARENA.get(pairs * m.C2); lnBackward(m.pairLn, r.lp, dact2d, t, pairs, false); }
  m.initial.backward(r.initial, dact, dinitial, L, true);
  if (dsingle) lnBackward(m.singleLn, r.ls, dinitial, dsingle, L, true);
  else { real* t = ARENA.get((size_t)L * C); lnBackward(m.singleLn, r.ls, dinitial, t, L, false); }
}

}  // namespace af2
}  // namespace grad
