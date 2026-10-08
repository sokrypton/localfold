// ESMFold2's EDM sampler (biohub's DiffusionStructureHead.sample, cpu/esmfold2/sampler-reference.js):
//   schedule: sigma_data (sMax^(1/p) + k/(n-1) (sMin^(1/p) - sMax^(1/p)))^p, k < n, then 0; the levels
//             above max_sigma dropped and max_sigma prepended (15 scheduled -> 11 steps at 256)
//   x = schedule[0] N(0, 1)
//   each step (sigma -> sigma'): centre, rotate at random, translate by N(0, 1);
//     t = sigma (1 + gamma), gamma = gamma0 where the NEXT level exceeds gammaMin;
//     x_noisy = x + noiseScale sqrt(t^2 - sigma^2) N(0, 1);  d = D(x_noisy, t);
//     x_noisy = Kabsch(x_noisy onto d);  x = x_noisy + stepScale (sigma' - t) (x_noisy - d) / t
// The draws are this port's own (a seeded mt19937), so a fold is a different sample from the
// reference's; the denoiser is held to the reference's own recorded inputs instead.
#pragma once
#include <random>
#include "diffusion.cuh"

struct SamplerSettings { int steps = 15; double sMax = 160, sMin = 4e-4, p = 8, maxSigma = 256, gamma0 = 0.605,
                         gammaMin = 1.107, noiseScale = 0.901, stepScale = 1.638; };

inline std::vector<double> noiseSchedule(const SamplerSettings& s, double sigmaData) {
  std::vector<double> v;
  if (s.steps == 1) v = {s.sMax * sigmaData, 0};
  else {
    double hi = pow(s.sMax, 1 / s.p), lo = pow(s.sMin, 1 / s.p);
    for (int k = 0; k < s.steps; ++k) v.push_back(sigmaData * pow(hi + (double)k / (s.steps - 1) * (lo - hi), s.p));
    v.push_back(0);
  }
  if (s.maxSigma <= 0) return v;
  std::vector<double> out{s.maxSigma};
  for (double x : v) if (x <= s.maxSigma) out.push_back(x);
  return out;
}

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

inline bool SAMPLER_GRAPH = true;    // --no-sampler-graph: every step launched as it is
// the whole sampler; returns the final coordinates [A, 3] on the host
// what a sampler step's prediction is handed to (ef2.cu's --frames): the denoised positions on the device
inline std::function<void(const float*, int, int)> FRAME_HOOK;
inline std::vector<float> sample(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int* stepsRun = nullptr) {
  int A = d.A;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> N(0, 1);
  std::vector<double> sched = noiseSchedule(s, d.sigma);
  std::vector<float> mask = download(d.atoms.ctx.mask, A);
  std::vector<float> x(A * 3), xd(A * 3);
  for (auto& v : x) v = (float)(sched[0] * N(rng));
  float* dx = dalloc((size_t)A * 3); float* dd = dalloc((size_t)A * 3);
  int steps = (int)sched.size() - 1;
  cudaGraphExec_t graph = nullptr;
  for (int i = 0; i < steps; ++i) {
    double sigma = sched[i], next = sched[i + 1], gamma = next > s.gammaMin ? s.gamma0 : 0;
    // centre, a uniform random rotation (a normalised Gaussian quaternion), a N(0, 1) translation
    double tot = 0, c[3] = {0, 0, 0};
    for (int a = 0; a < A; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
    for (int k = 0; k < 3; ++k) c[k] /= std::max(tot, 1.0);
    double q[4]; for (auto& v : q) v = N(rng);
    double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    double qr = q[0] / qn, qi = q[1] / qn, qj = q[2] / qn, qk = q[3] / qn;
    double R[9] = {1 - 2 * (qj * qj + qk * qk), 2 * (qi * qj - qk * qr), 2 * (qi * qk + qj * qr),
                   2 * (qi * qj + qk * qr), 1 - 2 * (qi * qi + qk * qk), 2 * (qj * qk - qi * qr),
                   2 * (qi * qk - qj * qr), 2 * (qj * qk + qi * qr), 1 - 2 * (qi * qi + qj * qj)};
    double sh[3] = {N(rng), N(rng), N(rng)};
    for (int a = 0; a < A; ++a) {
      double p[3] = {x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]};
      for (int k = 0; k < 3; ++k) x[a * 3 + k] = (float)(p[0] * R[0 * 3 + k] + p[1] * R[1 * 3 + k] + p[2] * R[2 * 3 + k] + sh[k]);
    }
    double t = sigma * (1 + gamma);
    double eps = s.noiseScale * sqrt(std::max(t * t - sigma * sigma, 0.0));
    for (auto& v : x) v += (float)(eps * N(rng));
    CK(cudaMemcpyAsync(dx, x.data(), (size_t)A * 12, cudaMemcpyHostToDevice, STREAM));
    setLevel(d, (float)t);
    // the first step runs as it is (every scratch buffer and cuBLAS plan made), the second is captured
    // and every later one replays it: ~350 launches a step become one
    if (i == 0 || !SAMPLER_GRAPH) denoiseAtLevel(d, dx, dd);
    else {
      if (!graph) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(STREAM, cudaStreamCaptureModeThreadLocal));
        denoiseAtLevel(d, dx, dd);
        CK(cudaStreamEndCapture(STREAM, &g));
        CK(cudaGraphInstantiate(&graph, g, 0));
        CK(cudaGraphDestroy(g));
      }
      CK(cudaGraphLaunch(graph, STREAM));
    }
    if (FRAME_HOOK) FRAME_HOOK(dd, i + 1, steps);
    CK(cudaMemcpyAsync(xd.data(), dd, (size_t)A * 12, cudaMemcpyDeviceToHost, STREAM));
    CK(cudaStreamSynchronize(STREAM));
    rigidAlign(x, xd, mask, A);
    double f = s.stepScale * (next - t) / t;
    for (int k = 0; k < A * 3; ++k) x[k] = (float)(x[k] + f * (x[k] - xd[k]));
  }
  if (graph) CK(cudaGraphExecDestroy(graph));
  CK(cudaFree(dx)); CK(cudaFree(dd));
  if (stepsRun) *stepsRun = steps;
  return x;
}

// the page's records (pdb.template, from export_input.mjs) with these coordinates
inline void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x,
                     const std::vector<float>* bfactor = nullptr) {
  std::ifstream in(templatePath);
  if (!in) { fprintf(stderr, "no %s (export_input.mjs writes it)\n", templatePath.c_str()); exit(1); }
  FILE* f = fopen(out.c_str(), "w");
  std::string line;
  while (std::getline(in, line)) {
    if ((line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0) && line.size() >= 66) {
      int atom = (int)lround(atof(line.substr(30, 8).c_str())) * 1000 + (int)lround(atof(line.substr(38, 8).c_str()));
      char coords[64];
      snprintf(coords, sizeof coords, "%8.3f%8.3f%8.3f  1.00%6.2f", x[atom * 3], x[atom * 3 + 1], x[atom * 3 + 2],
               bfactor ? (*bfactor)[atom] : 0.0);
      line = line.substr(0, 30) + coords + line.substr(66);
    }
    fprintf(f, "%s\n", line.c_str());
  }
  fclose(f);
}
