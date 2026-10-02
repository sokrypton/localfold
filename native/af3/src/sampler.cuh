// AF3's EDM sampler around the denoiser, and the PDB the fold writes.
// Transcribed from src/af3/diffusion/diffusion-sampler-reference.js.
#pragma once
#include "diffusion.cuh"
#include <random>

struct Normal {                 // seeded Gaussian deviates
  std::mt19937_64 gen; std::normal_distribution<double> dist{0.0, 1.0};
  explicit Normal(uint64_t seed) : gen(seed) {}
  double operator()() { return dist(gen); }
};

inline double noiseSchedule(double t, double sigmaData = 16, double sigmaMin = 0.0004, double sigmaMax = 160,
                            double rho = 7) {
  double lo = std::pow(sigmaMin, 1 / rho), hi = std::pow(sigmaMax, 1 / rho);
  return sigmaData * std::pow(hi + t * (lo - hi), rho);
}

// Centre on the real atoms, rotate by a random rotation, translate by a unit normal.
inline void randomAugmentation(std::vector<float>& x, const std::vector<float>& mask, Normal& normal) {
  size_t atoms = mask.size();
  double c[3] = {0, 0, 0}, count = 0;
  for (size_t a = 0; a < atoms; ++a) if (mask[a]) { count += 1; for (int k = 0; k < 3; ++k) c[k] += x[a * 3 + k]; }
  for (int k = 0; k < 3; ++k) c[k] /= (count + 1e-6);
  double v0[3] = {normal(), normal(), normal()}, v1[3] = {normal(), normal(), normal()};
  auto norm = [](const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
  double e0[3], e1[3], e2[3], s0 = 1 / std::max(1e-10, norm(v0));
  for (int k = 0; k < 3; ++k) e0[k] = v0[k] * s0;
  double dot = v1[0] * e0[0] + v1[1] * e0[1] + v1[2] * e0[2], w[3];
  for (int k = 0; k < 3; ++k) w[k] = v1[k] - e0[k] * dot;
  double s1 = 1 / std::max(1e-10, norm(w));
  for (int k = 0; k < 3; ++k) e1[k] = w[k] * s1;
  e2[0] = e0[1] * e1[2] - e0[2] * e1[1]; e2[1] = e0[2] * e1[0] - e0[0] * e1[2]; e2[2] = e0[0] * e1[1] - e0[1] * e1[0];
  double tr[3] = {normal(), normal(), normal()};
  for (size_t a = 0; a < atoms; ++a) {
    if (!mask[a]) { for (int k = 0; k < 3; ++k) x[a * 3 + k] = 0; continue; }
    double p[3] = {x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]};
    for (int k = 0; k < 3; ++k) x[a * 3 + k] = (float)(p[0] * e0[k] + p[1] * e1[k] + p[2] * e2[k] + tr[k]);
  }
}

// `denoiseFn(noisyDevice, tHat)` returns the denoised positions on the device.
inline std::vector<float> sample(int steps, uint64_t seed, const std::vector<float>& mask,
                                 const std::function<const float*(const float*, float)>& denoiseFn,
                                 double gamma0 = 0.8, double gammaMin = 1.0, double noiseScale = 1.003,
                                 double stepScale = 1.5) {
  size_t atoms = mask.size();
  Normal normal(seed);
  std::vector<double> levels(steps + 1);
  for (int k = 0; k <= steps; ++k) levels[k] = noiseSchedule((double)k / steps);
  std::vector<float> x(atoms * 3), noisy(atoms * 3), denoised(atoms * 3);
  for (auto& v : x) v = (float)(normal() * levels[0]);
  float* dNoisy = scratch<float>("sample.noisy", atoms * 3);
  double previous = levels[0];
  for (int step = 1; step <= steps; ++step) {
    double level = levels[step];
    randomAugmentation(x, mask, normal);
    double gamma = level > gammaMin ? gamma0 : 0;
    double tHat = previous * (1 + gamma);
    double injected = noiseScale * std::sqrt(std::max(0.0, tHat * tHat - previous * previous));
    for (size_t i = 0; i < x.size(); ++i) noisy[i] = (float)(x[i] + injected * normal());
    CK(cudaMemcpyAsync(dNoisy, noisy.data(), noisy.size() * 4, cudaMemcpyHostToDevice, STREAM));
    const float* d = denoiseFn(dNoisy, (float)tHat);
    if (step == 1) STAGE_MS.clear();    // the first call uploads weights and allocates; not a profile
    CK(cudaMemcpyAsync(denoised.data(), d, denoised.size() * 4, cudaMemcpyDeviceToHost, STREAM));
    CK(cudaStreamSynchronize(STREAM));
    double delta = level - tHat;
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = (float)(noisy[i] + stepScale * delta * (noisy[i] - denoised[i]) / tHat);
    previous = level;
  }
  return x;
}

// The dense atom grid as a PDB: one residue per token, atom names from ref_atom_name_chars.
inline void writePdb(const std::string& path, const std::vector<float>& x, const float* bfactors = nullptr) {
  static const char* RES[20] = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"};
  static const char* ELEM[] = {"X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
                               "P", "S", "Cl"};
  int n = (int)M.meta("batch.tokens"), dense = (int)M.meta("batch.dense");
  const int* aatype = M.i("batch.aatype"); const int* names = M.i("batch.refAtomNameChars");
  const int* elem = M.i("batch.refElement"); const float* mask = M.f("batch.refMask");
  const int* chain = M.i("batch.asymId"); const int* resIdx = M.i("batch.residueIndex");
  FILE* f = fopen(path.c_str(), "w");
  int serial = 1;
  for (int t = 0; t < n; ++t) for (int a = 0; a < dense; ++a) {
    size_t i = (size_t)t * dense + a;
    if (!mask[i]) continue;
    char name[5] = {0};
    for (int k = 0; k < 4; ++k) { int c = names[i * 4 + k]; name[k] = c > 0 ? (char)(c + 32) : 0; }
    int z = elem[i]; const char* el = z >= 0 && z < 18 ? ELEM[z] : "X";
    const char* res = aatype[t] >= 0 && aatype[t] < 20 ? RES[aatype[t]] : "UNK";
    fprintf(f, "ATOM  %5d %-4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n", serial++,
            strlen(name) < 4 ? (std::string(" ") + name).c_str() : name, res, 'A' + (chain[t] - 1) % 26,
            resIdx[t], x[i * 3], x[i * 3 + 1], x[i * 3 + 2], 1.0, bfactors ? bfactors[t] : 0.0, el);
  }
  fprintf(f, "END\n");
  fclose(f);
}
