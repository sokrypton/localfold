// BioEmu2's EDM sampler: af3-any-model's diffusion_head.sample under the bioemu2 constants (model_registry's
// _SAMPLER_CONSTANTS: bioemu2's EDMSampler with the checkpoint's EDMSchedule) - boltz's loop:
//   levels: sigma_data (sMax^(1/rho) + k/(n-1) (sMin^(1/rho) - sMax^(1/rho)))^rho, k < n, then a 0 appended
//   x = levels[0] N(0, 1)
//   each step (sigma -> sigma'): centre on the real atoms, rotate at random, translate by N(0, 1), zero the padding;
//     gamma = gamma0 where the NEXT level exceeds gammaMin;  t = sigma (1 + gamma)
//     x_noisy = x + noiseScale sqrt(t^2 - sigma^2) N(0, 1);  d = D(x_noisy, t)
//     x_noisy = Kabsch(x_noisy onto d) (REALIGN_SAMPLER);  x = x_noisy + stepScale (sigma' - t) (x_noisy - d) / t
//   each sample centred on its real atoms at the end
// With a COMPANION (propagation's conditioning frame) the frame takes the state's rotation and translation every
// step, each centred on its own mean, and is the denoiser's coords_cond.
// The draws are this port's own (a seeded mt19937 per particle), so a sample is a different sample from the
// reference's; the denoiser is held to the reference's recorded inputs instead (oracle.py).
//
// 🔴 ALL PARTICLES STEP TOGETHER, WHICH IS THE SEAM STEERING NEEDS. BioEmu2's edm_gd/edm_fk/edm_fkc presets couple
// the samples (Feynman-Kac resampling reorders particles mid-trajectory), so the loop is particle-inner: every
// particle's denoised estimate exists before any takes its step. `StepHook` is where they would go - `guide` may
// move each denoised estimate (gradient guidance on the energy of D), and `resample` may permute the particles
// (and their companions) after the step. Neither is implemented here; a null hook is the plain sampler.
#pragma once
#include <random>
#include "denoise.cuh"
#include "../../af3/src/rigid.cuh"

namespace be2 {

struct SamplerSettings { int steps = 200; double sMax = 160, sMin = 4e-4, rho = 7, gamma0 = 0.8, gammaMin = 1.0,
                         noiseScale = 1.0, stepScale = 1.5; };
struct StepHook {
  // (step, sigma', t, the particles' denoised estimates [n][N * 3], host): may move the estimates
  std::function<void(int, double, double, std::vector<std::vector<float>>&)> guide;
  // (step, the particles' states and companions after the step): may permute / replace them
  std::function<void(int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&)> resample;
};

// the levels as af3-any-model computes them: float32 times and a float32 power, then the 0
inline std::vector<float> schedule(const SamplerSettings& s) {
  std::vector<float> v;
  const float hi = powf((float)s.sMax, 1.f / (float)s.rho), lo = powf((float)s.sMin, 1.f / (float)s.rho);
  for (int k = 0; k < s.steps; ++k) {
    float t = s.steps == 1 ? 0.f : (float)k / (float)(s.steps - 1);
    v.push_back(SIGMA_DATA * powf(hi + t * (lo - hi), (float)s.rho));
  }
  v.push_back(0.f);
  return v;
}

// a uniform random rotation (a normalised Gaussian quaternion) and a N(0, 1) translation, applied to x
// centred on its masked mean; padding zeroed
inline void augment(std::vector<float>& x, const std::vector<float>& mask, const double R[9], const double sh[3]) {
  const size_t N = mask.size();
  double tot = 0, c[3] = {0, 0, 0};
  for (size_t a = 0; a < N; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
  for (int k = 0; k < 3; ++k) c[k] /= tot + 1e-6;
  for (size_t a = 0; a < N; ++a) {
    double p[3] = {x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]};
    for (int k = 0; k < 3; ++k)
      x[a * 3 + k] = mask[a] ? (float)(p[0] * R[0 * 3 + k] + p[1] * R[1 * 3 + k] + p[2] * R[2 * 3 + k] + sh[k]) : 0.f;
  }
}
inline void randomRigid(std::mt19937_64& rng, double R[9], double sh[3]) {
  std::normal_distribution<double> Nd(0, 1);
  double q[4]; for (auto& v : q) v = Nd(rng);
  double qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  double w = q[0] / qn, i = q[1] / qn, j = q[2] / qn, k = q[3] / qn;
  double r[9] = {1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w),
                 2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w),
                 2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j)};
  for (int a = 0; a < 9; ++a) R[a] = r[a];
  for (int a = 0; a < 3; ++a) sh[a] = Nd(rng);
}
inline void centreMasked(std::vector<float>& x, const std::vector<float>& mask) {
  const size_t N = mask.size(); double tot = 0, c[3] = {0, 0, 0};
  for (size_t a = 0; a < N; ++a) { tot += mask[a]; for (int k = 0; k < 3; ++k) c[k] += mask[a] * x[a * 3 + k]; }
  for (int k = 0; k < 3; ++k) c[k] /= std::max(tot, 1e-6);
  for (size_t a = 0; a < N; ++a) for (int k = 0; k < 3; ++k) x[a * 3 + k] = mask[a] ? (float)(x[a * 3 + k] - c[k]) : 0.f;
}

// what each step's predictions are handed to (--frames): (particle, step, steps, the denoised estimate on the host)
inline std::function<void(int, int, int, const std::vector<float>&)> FRAME_HOOK;

// n particles from noise (companions: one centred conditioning frame each, or empty); returns their final states
inline std::vector<std::vector<float>> sample(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int n,
                                              std::vector<std::vector<float>> companions = {},
                                              const StepHook* hook = nullptr, bool centreEnd = true) {
  const int N = d.N;
  std::vector<float> mask(N);
  { const int* m = M.i("atom_mask"); for (int a = 0; a < N; ++a) mask[a] = m[a] ? 1.f : 0.f; }
  const std::vector<float> levels = schedule(s);
  std::vector<std::mt19937_64> rng;
  for (int p = 0; p < n; ++p) rng.emplace_back(seed * 1000003ull + (uint64_t)p);
  std::normal_distribution<double> Nd(0, 1);
  std::vector<std::vector<float>> x(n, std::vector<float>((size_t)N * 3)), den(n, std::vector<float>((size_t)N * 3));
  for (int p = 0; p < n; ++p) for (auto& v : x[p]) v = (float)(levels[0] * Nd(rng[p]));
  const bool comp = !companions.empty();
  float* dx = dalloc((size_t)N * 3); float* dd = dalloc((size_t)N * 3); float* dc = comp ? dalloc((size_t)N * 3) : nullptr;
  const int steps = (int)levels.size() - 1;
  std::vector<double> tHat(n);
  for (int i = 0; i < steps; ++i) {
    const double sigma = levels[i], next = levels[i + 1];
    const double gamma = next > s.gammaMin ? s.gamma0 : 0;
    const double t = sigma * (1 + gamma);
    const double eps = s.noiseScale * sqrt(std::max(t * t - sigma * sigma, 0.0));
    for (int p = 0; p < n; ++p) {
      double R[9], sh[3];
      randomRigid(rng[p], R, sh);
      augment(x[p], mask, R, sh);
      if (comp) augment(companions[p], mask, R, sh);
      for (int a = 0; a < N * 3; ++a) x[p][a] += (float)(eps * Nd(rng[p]));
      CK(cudaMemcpyAsync(dx, x[p].data(), (size_t)N * 12, cudaMemcpyHostToDevice, STREAM));
      if (comp) CK(cudaMemcpyAsync(dc, companions[p].data(), (size_t)N * 12, cudaMemcpyHostToDevice, STREAM));
      denoise(d, dx, (float)t, dd, dc);
      CK(cudaMemcpyAsync(den[p].data(), dd, (size_t)N * 12, cudaMemcpyDeviceToHost, STREAM));
    }
    CK(cudaStreamSynchronize(STREAM));
    if (hook && hook->guide) hook->guide(i, next, t, den);
    for (int p = 0; p < n; ++p) {
      if (FRAME_HOOK) FRAME_HOOK(p, i + 1, steps, den[p]);
      rigidAlign(x[p], den[p], mask, N);
      const double f = s.stepScale * (next - t) / t;
      for (int k = 0; k < N * 3; ++k) x[p][k] = (float)(x[p][k] + f * (x[p][k] - den[p][k]));
    }
    if (hook && hook->resample) hook->resample(i, x, companions);
  }
  CK(cudaFree(dx)); CK(cudaFree(dd)); if (dc) CK(cudaFree(dc));
  if (centreEnd) for (auto& v : x) centreMasked(v, mask);
  else for (auto& v : x) for (int a = 0; a < N; ++a) if (!mask[a]) v[a * 3] = v[a * 3 + 1] = v[a * 3 + 2] = 0.f;
  return x;
}

// propagation (bioemu2.propagation): each particle a trajectory of `chain` lag-time steps from x0. A step samples
// conditioned on the previous frame (centred, the companion), then superposes the endpoint on that frame. Returns
// [particle][step] frames
inline std::vector<std::vector<std::vector<float>>> propagate(const Denoiser& d, const SamplerSettings& s, uint64_t seed,
                                                             int n, int chain, const std::vector<float>& x0,
                                                             const StepHook* hook = nullptr) {
  const int N = d.N;
  std::vector<float> mask(N);
  { const int* m = M.i("atom_mask"); for (int a = 0; a < N; ++a) mask[a] = m[a] ? 1.f : 0.f; }
  std::vector<std::vector<float>> prev(n, x0);
  std::vector<std::vector<std::vector<float>>> frames(n);
  for (int k = 0; k < chain; ++k) {
    std::vector<std::vector<float>> comps = prev;
    for (auto& c : comps) centreMasked(c, mask);
    // (the reference folds the step into its key; here the step is part of the seed)
    std::vector<std::vector<float>> end = sample(d, s, seed * 7919ull + (uint64_t)k, n, comps, hook, false);
    for (int p = 0; p < n; ++p) {
      rigidAlign(end[p], prev[p], mask, N);
      for (int a = 0; a < N; ++a) if (!mask[a]) end[p][a * 3] = end[p][a * 3 + 1] = end[p][a * 3 + 2] = 0.f;
      frames[p].push_back(end[p]);
      prev[p] = end[p];
    }
  }
  return frames;
}

// the records of pdb.template (export_input.mjs) with these coordinates
inline void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x) {
  std::ifstream in(templatePath);
  if (!in) { fprintf(stderr, "no %s (export_input.mjs writes it)\n", templatePath.c_str()); exit(1); }
  FILE* f = fopen(out.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot write %s: %s\n", out.c_str(), strerror(errno)); exit(1); }
  fprintf(f, "REMARK   1 BioEmu2 (bioemu2-preview): no confidence head - the B-factor column is zero, not a pLDDT\n");
  std::string line;
  while (std::getline(in, line)) {
    if ((line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0) && line.size() >= 66) {
      int atom = (int)lround(atof(line.substr(30, 8).c_str())) * 1000 + (int)lround(atof(line.substr(38, 8).c_str()));
      char coords[64];
      snprintf(coords, sizeof coords, "%8.3f%8.3f%8.3f", x[atom * 3], x[atom * 3 + 1], x[atom * 3 + 2]);
      line = line.substr(0, 30) + coords + line.substr(54);
    }
    fprintf(f, "%s\n", line.c_str());
  }
  fclose(f);
}

}  // namespace be2
