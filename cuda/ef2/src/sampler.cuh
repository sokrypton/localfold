// ESMFold2's EDM sampler (biohub's DiffusionStructureHead.sample, cpu/esmfold2/sampler.js):
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

// geometric, exponential, churn: the discretisation options cuda/af3/src/sampler.cuh's DiffusionOptions has (after
// Blitz) - the levels evenly spaced in log sigma (the zero and max_sigma's cap kept), the step
// x = r^S x_noisy + (1 - r^S) d with r = sigma' / t in place of Euler's, and the noise injection off. The defaults
// are biohub's own sampler
struct SamplerSettings { int steps = 15; double sMax = 160, sMin = 4e-4, p = 8, maxSigma = 256, gamma0 = 0.605,
                         gammaMin = 1.107, noiseScale = 0.901, stepScale = 1.638;
                         bool geometric = false, exponential = false, churn = true; };
// a discretisation flag read into s (--schedule=karras|geometric, --update=euler|exponential, --no-churn); false
// for any other flag, and an unknown value refused by name
inline bool samplerFlag(const char* f, SamplerSettings& s) {
  auto value = [&](const char* name, const char* a, const char* b, bool& set) {
    const size_t n = strlen(name);
    if (strncmp(f, name, n)) return false;
    if (!strcmp(f + n, a)) set = false; else if (!strcmp(f + n, b)) set = true;
    else { fprintf(stderr, "%s takes %s or %s, not %s\n", name, a, b, f + n); exit(1); }
    return true;
  };
  if (value("--schedule=", "karras", "geometric", s.geometric) || value("--update=", "euler", "exponential", s.exponential)) return true;
  if (!strcmp(f, "--no-churn")) { s.churn = false; return true; }
  return false;
}

inline std::vector<double> noiseSchedule(const SamplerSettings& s, double sigmaData) {
  std::vector<double> v;
  if (s.steps == 1) v = {s.sMax * sigmaData, 0};
  else {
    double hi = pow(s.sMax, 1 / s.p), lo = pow(s.sMin, 1 / s.p);
    for (int k = 0; k < s.steps; ++k)
      v.push_back(s.geometric ? sigmaData * exp(log(s.sMax) + (double)k / (s.steps - 1) * (log(s.sMin) - log(s.sMax)))
                              : sigmaData * pow(hi + (double)k / (s.steps - 1) * (lo - hi), s.p));
    v.push_back(0);
  }
  if (s.maxSigma <= 0) return v;
  std::vector<double> out{s.maxSigma};
  for (double x : v) if (x <= s.maxSigma) out.push_back(x);
  return out;
}

#include "../../af3/src/rigid.cuh"   // rigidAlign, shared with boltz2's sampler

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
  std::vector<float> x(A * 3), xd(A * 3), noise(A * 3), nextNoise(A * 3);
  for (auto& v : x) v = (float)(sched[0] * N(rng));
  float* dx = scratch<float>("samp.dx", (size_t)A * 3);
  float* dd = scratch<float>("samp.dd", (size_t)A * 3);
  int steps = (int)sched.size() - 1;
  cudaGraphExec_t graph = nullptr;
  double R[9], sh[3], nextR[9], nextSh[3];
  auto drawStep = [&](int idx, double* rot, double* shift, std::vector<float>& ns) {
    double sig = sched[idx], nxt = sched[idx + 1], gam = s.churn && nxt > s.gammaMin ? s.gamma0 : 0;
    double q[4]; for (auto& v : q) v = N(rng);
    double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    double qr = q[0] / qn, qi = q[1] / qn, qj = q[2] / qn, qk = q[3] / qn;
    rot[0] = 1 - 2 * (qj * qj + qk * qk); rot[1] = 2 * (qi * qj - qk * qr); rot[2] = 2 * (qi * qk + qj * qr);
    rot[3] = 2 * (qi * qj + qk * qr); rot[4] = 1 - 2 * (qi * qi + qk * qk); rot[5] = 2 * (qj * qk - qi * qr);
    rot[6] = 2 * (qi * qk - qj * qr); rot[7] = 2 * (qj * qk + qi * qr); rot[8] = 1 - 2 * (qi * qi + qj * qj);
    shift[0] = N(rng); shift[1] = N(rng); shift[2] = N(rng);
    double t = sig * (1 + gam);
    double eps = s.noiseScale * sqrt(std::max(t * t - sig * sig, 0.0));
    for (int k = 0; k < A * 3; ++k) ns[k] = (float)(eps * N(rng));
  };
  if (steps > 0) drawStep(0, R, sh, noise);
  for (int i = 0; i < steps; ++i) {
    double sigma = sched[i], next = sched[i + 1], gamma = s.churn && next > s.gammaMin ? s.gamma0 : 0;
    // centre, a uniform random rotation (a normalised Gaussian quaternion), a N(0, 1) translation
    double tot = 0, c[3] = {0, 0, 0};
    for (int a = 0; a < A; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
    for (int k = 0; k < 3; ++k) c[k] /= std::max(tot, 1.0);
    for (int a = 0; a < A; ++a) {
      double p[3] = {x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]};
      for (int k = 0; k < 3; ++k)
        x[a * 3 + k] = (float)(p[0] * R[0 * 3 + k] + p[1] * R[1 * 3 + k] + p[2] * R[2 * 3 + k] + sh[k]) + noise[a * 3 + k];
    }
    double t = sigma * (1 + gamma);
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
    if (i + 1 < steps) {
      drawStep(i + 1, nextR, nextSh, nextNoise);
      memcpy(R, nextR, sizeof(R));
      memcpy(sh, nextSh, sizeof(sh));
      noise.swap(nextNoise);
    }
    CK(cudaStreamSynchronize(STREAM));
    rigidAlign(x, xd, mask, A);
    // (Euler's x + f (x - d) is the exponential step's r^S x + (1 - r^S) d at f = r^S - 1)
    double f = s.exponential ? pow(next / t, s.stepScale) - 1 : s.stepScale * (next - t) / t;
    for (int k = 0; k < A * 3; ++k) x[k] = (float)(x[k] + f * (x[k] - xd[k]));
  }
  if (graph) CK(cudaGraphExecDestroy(graph));
  if (stepsRun) *stepsRun = steps;
  return x;
}

// the page's records (pdb.template, from export_input.mjs) with these coordinates
inline void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x,
                     const std::vector<float>* bfactor = nullptr) {
  std::ifstream in(templatePath);
  if (!in) { fprintf(stderr, "no %s (export_input.mjs writes it)\n", templatePath.c_str()); exit(1); }
  FILE* f = fopen(out.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot write %s: %s\n", out.c_str(), strerror(errno)); exit(1); }
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
