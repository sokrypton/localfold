// The weighted rigid alignment the realigning samplers take (ESMFold2's, boltz2's: cpu/af3/diffusion/rigid-align.js)
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

// x (rows, 3) aligned onto target by the weighted Kabsch rotation (float64, Jacobi on H^T H; a
// reflection is refused through the determinant's sign)
inline void rigidAlign(std::vector<float>& x, const std::vector<float>& target, const std::vector<float>& w, int n) {
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
    if (off < 1e-30) break;
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
    for (int i = 0; i < 3; ++i) x[a * 3 + i] = (float)(d[0] * r[i * 3] + d[1] * r[i * 3 + 1] + d[2] * r[i * 3 + 2] + ct[i]);
  }
}
