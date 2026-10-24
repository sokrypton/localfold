// The weighted rigid alignment the realigning samplers take (ESMFold2's, boltz2's: cpu/af3/diffusion/rigid-align.js)
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

// x (rows, 3) aligned onto target by the weighted Kabsch rotation (float64, Jacobi on H^T H; a
// reflection is refused through the determinant's sign)
inline void rigidAlign(float* x, const float* target, const float* w, int n,
                       float* eulerOut = nullptr, float eulerCoef = 0.f) {
  double tot = 0, cx[3] = {0, 0, 0}, ct[3] = {0, 0, 0};
  for (int a = 0; a < n; ++a) { tot += w[a]; for (int k = 0; k < 3; ++k) { cx[k] += w[a] * x[a * 3 + k]; ct[k] += w[a] * target[a * 3 + k]; } }
  tot = std::max(tot, 1e-8);
  for (int k = 0; k < 3; ++k) { cx[k] /= tot; ct[k] /= tot; }
  double h[9] = {0};                                   // h[i][j] = sum w (t_i)(x_j)
  for (int a = 0; a < n; ++a) {
    if (w[a] == 0) continue;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
      h[i * 3 + j] += w[a] * (target[a * 3 + i] - ct[i]) * (x[a * 3 + j] - cx[j]);
  }
  // SVD of h through the eigendecomposition of h^T h
  double m[9], v[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += h[k * 3 + i] * h[k * 3 + j]; m[i * 3 + j] = s; }
  for (int sweep = 0; sweep < 32; ++sweep) {
    double off = m[1] * m[1] + m[2] * m[2] + m[5] * m[5];
    if (off <= 1e-30 * (m[0] * m[0] + m[4] * m[4] + m[8] * m[8])) break;
    const int P[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (auto& pq : P) {
      int p = pq[0], q = pq[1];
      double apq = m[p * 3 + q];
      if (fabs(apq) < 1e-300) continue;
      double theta = (m[q * 3 + q] - m[p * 3 + p]) / (2 * apq);
      double t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1));
      double c = 1 / sqrt(t * t + 1), s = t * c;
      for (int k = 0; k < 3; ++k) { double akp = m[k * 3 + p], akq = m[k * 3 + q]; m[k * 3 + p] = c * akp - s * akq; m[k * 3 + q] = s * akp + c * akq; }
      for (int k = 0; k < 3; ++k) { double apk = m[p * 3 + k], aqk = m[q * 3 + k]; m[p * 3 + k] = c * apk - s * aqk; m[q * 3 + k] = s * apk + c * aqk; }
      for (int k = 0; k < 3; ++k) { double vkp = v[k * 3 + p], vkq = v[k * 3 + q]; v[k * 3 + p] = c * vkp - s * vkq; v[k * 3 + q] = s * vkp + c * vkq; }
    }
  }
  double ev[3] = {m[0], m[4], m[8]};
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int a, int b) { return ev[a] > ev[b]; });
  double vv[9], uu[9] = {0};
  bool good[3] = {false, false, false};
  double floor = 1e-12 * fabs(ev[order[0]] ? ev[order[0]] : 1);
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) vv[r * 3 + c] = v[r * 3 + order[c]];
    double sig = sqrt(std::max(ev[order[c]], 0.0));
    if (ev[order[c]] <= floor) continue;
    for (int r = 0; r < 3; ++r) { double s = 0; for (int k = 0; k < 3; ++k) s += h[r * 3 + k] * vv[k * 3 + c]; uu[r * 3 + c] = s / sig; }
    double len = sqrt(uu[c] * uu[c] + uu[3 + c] * uu[3 + c] + uu[6 + c] * uu[6 + c]);
    if (fabs(len - 1) > 1e-6) { uu[c] = uu[3 + c] = uu[6 + c] = 0; } else good[c] = true;
  }
  for (int c = 0; c < 3; ++c) {                       // complete a rank-deficient U orthonormally
    if (good[c]) continue;
    double best[3] = {0, 0, 0}, bestLen = -1;
    for (int sd = 0; sd < 3; ++sd) {
      double wv[3] = {0, 0, 0}; wv[sd] = 1;
      for (int k = 0; k < 3; ++k) if (good[k]) { double dot = wv[0] * uu[k] + wv[1] * uu[3 + k] + wv[2] * uu[6 + k]; for (int r = 0; r < 3; ++r) wv[r] -= dot * uu[r * 3 + k]; }
      double len = sqrt(wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2]);
      if (len > bestLen) { bestLen = len; for (int r = 0; r < 3; ++r) best[r] = wv[r]; }
    }
    for (int r = 0; r < 3; ++r) uu[r * 3 + c] = best[r] / bestLen;
    good[c] = true;
  }
  double uvt[9];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += uu[i * 3 + k] * vv[j * 3 + k]; uvt[i * 3 + j] = s; }
  double det = uvt[0] * (uvt[4] * uvt[8] - uvt[5] * uvt[7]) - uvt[1] * (uvt[3] * uvt[8] - uvt[5] * uvt[6]) + uvt[2] * (uvt[3] * uvt[7] - uvt[4] * uvt[6]);
  double sign = det < 0 ? -1 : 1, r[9];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += uu[i * 3 + k] * (k == 2 ? sign : 1) * vv[j * 3 + k]; r[i * 3 + j] = s; }
  for (int a = 0; a < n; ++a) {
    double d[3] = {x[a * 3] - cx[0], x[a * 3 + 1] - cx[1], x[a * 3 + 2] - cx[2]};
    for (int i = 0; i < 3; ++i) {
      float al = (float)(d[0] * r[i * 3] + d[1] * r[i * 3 + 1] + d[2] * r[i * 3 + 2] + ct[i]);
      if (eulerOut) eulerOut[a * 3 + i] = al + eulerCoef * (al - target[a * 3 + i]);
      else x[a * 3 + i] = al;
    }
  }
}
inline void rigidAlign(std::vector<float>& x, const std::vector<float>& target, const std::vector<float>& w, int n) {
  rigidAlign(x.data(), target.data(), w.data(), n);
}

#ifdef __CUDACC__
__global__ void __launch_bounds__(256)
rigidAlignK(float* __restrict__ xAll, const float* __restrict__ targetAll, const float* __restrict__ w, int n) {
  __shared__ double sRed[8][9];
  __shared__ double sCx[3], sCt[3], sR[9];
  float* x = xAll + (size_t)blockIdx.x * n * 3;
  const float* target = targetAll + (size_t)blockIdx.x * n * 3;
  int tx = threadIdx.x, warp = tx >> 5, lane = tx & 31;
  double tot = 0, cx0 = 0, cx1 = 0, cx2 = 0, ct0 = 0, ct1 = 0, ct2 = 0;
  float rx[16][3], rt[16][3], rw[16];
  const bool reg = n <= 4096;
  if (reg) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      int a = tx + k * 256;
      if (a >= n) break;
      float wa = w[a];
      rw[k] = wa;
      float x0 = x[a * 3], x1 = x[a * 3 + 1], x2 = x[a * 3 + 2];
      float t0 = target[a * 3], t1 = target[a * 3 + 1], t2 = target[a * 3 + 2];
      rx[k][0] = x0; rx[k][1] = x1; rx[k][2] = x2;
      rt[k][0] = t0; rt[k][1] = t1; rt[k][2] = t2;
      double dw = wa;
      tot += dw;
      cx0 += dw * x0; cx1 += dw * x1; cx2 += dw * x2;
      ct0 += dw * t0; ct1 += dw * t1; ct2 += dw * t2;
    }
  } else {
    for (int a = tx; a < n; a += 256) {
      double wa = w[a];
      tot += wa;
      cx0 += wa * x[a * 3];     cx1 += wa * x[a * 3 + 1];     cx2 += wa * x[a * 3 + 2];
      ct0 += wa * target[a * 3]; ct1 += wa * target[a * 3 + 1]; ct2 += wa * target[a * 3 + 2];
    }
  }
  for (int o = 16; o; o >>= 1) {
    tot += __shfl_xor_sync(~0u, tot, o);
    cx0 += __shfl_xor_sync(~0u, cx0, o); cx1 += __shfl_xor_sync(~0u, cx1, o); cx2 += __shfl_xor_sync(~0u, cx2, o);
    ct0 += __shfl_xor_sync(~0u, ct0, o); ct1 += __shfl_xor_sync(~0u, ct1, o); ct2 += __shfl_xor_sync(~0u, ct2, o);
  }
  if (lane == 0) {
    sRed[warp][0] = tot;
    sRed[warp][1] = cx0; sRed[warp][2] = cx1; sRed[warp][3] = cx2;
    sRed[warp][4] = ct0; sRed[warp][5] = ct1; sRed[warp][6] = ct2;
  }
  __syncthreads();
  if (tx == 0) {
    double t = 0, x0 = 0, x1 = 0, x2 = 0, y0 = 0, y1 = 0, y2 = 0;
    for (int k = 0; k < 8; ++k) {
      t  += sRed[k][0];
      x0 += sRed[k][1]; x1 += sRed[k][2]; x2 += sRed[k][3];
      y0 += sRed[k][4]; y1 += sRed[k][5]; y2 += sRed[k][6];
    }
    if (t < 1e-8) t = 1e-8;
    double invT = 1.0 / t;
    sCx[0] = x0 * invT; sCx[1] = x1 * invT; sCx[2] = x2 * invT;
    sCt[0] = y0 * invT; sCt[1] = y1 * invT; sCt[2] = y2 * invT;
  }
  __syncthreads();
  const double cx[3] = {sCx[0], sCx[1], sCx[2]}, ct[3] = {sCt[0], sCt[1], sCt[2]};
  double h[9] = {0};
  if (reg) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      int a = tx + k * 256;
      if (a >= n) break;
      double wa = rw[k];
      if (wa == 0) continue;
      double dt0 = rt[k][0] - ct[0], dt1 = rt[k][1] - ct[1], dt2 = rt[k][2] - ct[2];
      double dx0 = rx[k][0] - cx[0], dx1 = rx[k][1] - cx[1], dx2 = rx[k][2] - cx[2];
      h[0] += wa * dt0 * dx0; h[1] += wa * dt0 * dx1; h[2] += wa * dt0 * dx2;
      h[3] += wa * dt1 * dx0; h[4] += wa * dt1 * dx1; h[5] += wa * dt1 * dx2;
      h[6] += wa * dt2 * dx0; h[7] += wa * dt2 * dx1; h[8] += wa * dt2 * dx2;
    }
  } else {
    for (int a = tx; a < n; a += 256) {
      double wa = w[a];
      if (wa == 0) continue;
      double dt0 = target[a * 3] - ct[0], dt1 = target[a * 3 + 1] - ct[1], dt2 = target[a * 3 + 2] - ct[2];
      double dx0 = x[a * 3] - cx[0],      dx1 = x[a * 3 + 1] - cx[1],      dx2 = x[a * 3 + 2] - cx[2];
      h[0] += wa * dt0 * dx0; h[1] += wa * dt0 * dx1; h[2] += wa * dt0 * dx2;
      h[3] += wa * dt1 * dx0; h[4] += wa * dt1 * dx1; h[5] += wa * dt1 * dx2;
      h[6] += wa * dt2 * dx0; h[7] += wa * dt2 * dx1; h[8] += wa * dt2 * dx2;
    }
  }
  for (int i = 0; i < 9; ++i) {
    for (int o = 16; o; o >>= 1) h[i] += __shfl_xor_sync(~0u, h[i], o);
    if (lane == 0) sRed[warp][i] = h[i];
  }
  __syncthreads();
  if (tx == 0) {
    for (int i = 0; i < 9; ++i) {
      double s = 0;
      for (int k = 0; k < 8; ++k) s += sRed[k][i];
      h[i] = s;
    }
    double* m = sRed[0];
    double* v = sRed[1];
    double* vv = sRed[2];
    double* uu = sRed[3];
    double* uvt = sRed[4];
    double* ev = sRed[5];
    for (int i = 0; i < 9; ++i) { v[i] = (i % 4 == 0) ? 1.0 : 0.0; uu[i] = 0.0; }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      double s = 0; for (int k = 0; k < 3; ++k) s += h[k * 3 + i] * h[k * 3 + j];
      m[i * 3 + j] = s;
    }
    for (int sweep = 0; sweep < 32; ++sweep) {
      double off = m[1] * m[1] + m[2] * m[2] + m[5] * m[5];
      if (off <= 1e-30 * (m[0] * m[0] + m[4] * m[4] + m[8] * m[8])) break;
      const int P[3][2] = {{0, 1}, {0, 2}, {1, 2}};
#pragma unroll
      for (int pi = 0; pi < 3; ++pi) {
        int p = P[pi][0], q = P[pi][1];
        double apq = m[p * 3 + q];
        if (fabs(apq) < 1e-300) continue;
        double theta = (m[q * 3 + q] - m[p * 3 + p]) / (2 * apq);
        double t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1));
        double c = rsqrt(t * t + 1), s = t * c;
#pragma unroll
        for (int k = 0; k < 3; ++k) { double akp = m[k * 3 + p], akq = m[k * 3 + q]; m[k * 3 + p] = c * akp - s * akq; m[k * 3 + q] = s * akp + c * akq; }
#pragma unroll
        for (int k = 0; k < 3; ++k) { double apk = m[p * 3 + k], aqk = m[q * 3 + k]; m[p * 3 + k] = c * apk - s * aqk; m[q * 3 + k] = s * apk + c * aqk; }
#pragma unroll
        for (int k = 0; k < 3; ++k) { double vkp = v[k * 3 + p], vkq = v[k * 3 + q]; v[k * 3 + p] = c * vkp - s * vkq; v[k * 3 + q] = s * vkp + c * vkq; }
      }
    }
    ev[0] = m[0]; ev[1] = m[4]; ev[2] = m[8];
    int o0 = 0, o1 = 1, o2 = 2;
    if (ev[o1] > ev[o0]) { int tmp = o0; o0 = o1; o1 = tmp; }
    if (ev[o2] > ev[o1]) { int tmp = o1; o1 = o2; o2 = tmp; }
    if (ev[o1] > ev[o0]) { int tmp = o0; o0 = o1; o1 = tmp; }
    int order[3] = {o0, o1, o2};
    bool good[3] = {false, false, false};
    double floor = 1e-12 * fabs(ev[o0] ? ev[o0] : 1);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      int oc = order[c];
      for (int r = 0; r < 3; ++r) vv[r * 3 + c] = v[r * 3 + oc];
      if (ev[oc] <= floor) continue;
      double invSig = rsqrt(ev[oc]);
      for (int r = 0; r < 3; ++r) { double s = 0; for (int k = 0; k < 3; ++k) s += h[r * 3 + k] * vv[k * 3 + c]; uu[r * 3 + c] = s * invSig; }
      double lenSq = uu[c] * uu[c] + uu[3 + c] * uu[3 + c] + uu[6 + c] * uu[6 + c];
      if (fabs(lenSq - 1.0) > 2e-6) { uu[c] = uu[3 + c] = uu[6 + c] = 0; } else good[c] = true;
    }
    for (int c = 0; c < 3; ++c) {
      if (good[c]) continue;
      double best0 = 0, best1 = 0, best2 = 0, bestLen = -1;
      for (int sd = 0; sd < 3; ++sd) {
        double w0 = sd == 0 ? 1 : 0, w1 = sd == 1 ? 1 : 0, w2 = sd == 2 ? 1 : 0;
        for (int k = 0; k < 3; ++k) if (good[k]) {
          double dot = w0 * uu[k] + w1 * uu[3 + k] + w2 * uu[6 + k];
          w0 -= dot * uu[k]; w1 -= dot * uu[3 + k]; w2 -= dot * uu[6 + k];
        }
        double len = sqrt(w0 * w0 + w1 * w1 + w2 * w2);
        if (len > bestLen) { bestLen = len; best0 = w0; best1 = w1; best2 = w2; }
      }
      double invBest = 1.0 / bestLen;
      uu[c] = best0 * invBest; uu[3 + c] = best1 * invBest; uu[6 + c] = best2 * invBest;
      good[c] = true;
    }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += uu[i * 3 + k] * vv[j * 3 + k]; uvt[i * 3 + j] = s; }
    double det = uvt[0] * (uvt[4] * uvt[8] - uvt[5] * uvt[7]) - uvt[1] * (uvt[3] * uvt[8] - uvt[5] * uvt[6]) + uvt[2] * (uvt[3] * uvt[7] - uvt[4] * uvt[6]);
    double sign = det < 0 ? -1 : 1;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { double s = 0; for (int k = 0; k < 3; ++k) s += uu[i * 3 + k] * (k == 2 ? sign : 1) * vv[j * 3 + k]; sR[i * 3 + j] = s; }
  }
  __syncthreads();
  const double r00 = sR[0], r01 = sR[1], r02 = sR[2];
  const double r10 = sR[3], r11 = sR[4], r12 = sR[5];
  const double r20 = sR[6], r21 = sR[7], r22 = sR[8];
  if (reg) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      int a = tx + k * 256;
      if (a >= n) break;
      double d0 = rx[k][0] - cx[0], d1 = rx[k][1] - cx[1], d2 = rx[k][2] - cx[2];
      x[a * 3]     = (float)(d0 * r00 + d1 * r01 + d2 * r02 + ct[0]);
      x[a * 3 + 1] = (float)(d0 * r10 + d1 * r11 + d2 * r12 + ct[1]);
      x[a * 3 + 2] = (float)(d0 * r20 + d1 * r21 + d2 * r22 + ct[2]);
    }
  } else {
    for (int a = tx; a < n; a += 256) {
      double d0 = x[a * 3] - cx[0], d1 = x[a * 3 + 1] - cx[1], d2 = x[a * 3 + 2] - cx[2];
      x[a * 3]     = (float)(d0 * r00 + d1 * r01 + d2 * r02 + ct[0]);
      x[a * 3 + 1] = (float)(d0 * r10 + d1 * r11 + d2 * r12 + ct[1]);
      x[a * 3 + 2] = (float)(d0 * r20 + d1 * r21 + d2 * r22 + ct[2]);
    }
  }
}
#endif
