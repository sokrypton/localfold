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

// AF3's sampler, every step on the device: each step centres the real atoms, rotates by a random
// rotation, translates by a unit normal, injects noise and takes the Euler step. The host draws the identical random sequence up
// front (the start, then per step the augmentation's nine deviates and the injected noise), so
// the only host work per step is enqueueing it - no synchronisation until the end.
__global__ void centroidK(const float* x, const float* mask, size_t atoms, float* c) {
  __shared__ float s[4][256];
  float a[4] = {0, 0, 0, 0};
  for (size_t i = threadIdx.x; i < atoms; i += blockDim.x)
    if (mask[i]) { a[0] += x[i * 3]; a[1] += x[i * 3 + 1]; a[2] += x[i * 3 + 2]; a[3] += 1; }
  for (int k = 0; k < 4; ++k) s[k][threadIdx.x] = a[k];
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) for (int k = 0; k < 4; ++k) s[k][threadIdx.x] += s[k][threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x < 3) c[threadIdx.x] = s[threadIdx.x][0] / (s[3][0] + 1e-6f);
}
// rot = e0, e1, e2 (rows) and the translation; noisy = augmented x + injected * noise
__global__ void augmentNoiseK(float* x, float* noisy, const float* mask, const float* c, const float* rot,
                              const float* noise, float injected, size_t atoms) {
  size_t a = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= atoms) return;
  float p[3] = {x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]};
  for (int k = 0; k < 3; ++k) {
    float v = mask[a] ? p[0] * rot[k] + p[1] * rot[3 + k] + p[2] * rot[6 + k] + rot[9 + k] : 0.f;
    x[a * 3 + k] = v;
    noisy[a * 3 + k] = v + injected * noise[a * 3 + k];
  }
}
__global__ void eulerK(float* x, const float* noisy, const float* denoised, float scale, size_t n3) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n3) x[i] = noisy[i] + scale * (noisy[i] - denoised[i]);
}
inline std::vector<float> sample(int steps, uint64_t seed, const std::vector<float>& mask,
                                       const std::function<const float*(const float*, float, const float*)>& denoiseFn,
                                       double gamma0 = 0.8, double gammaMin = 1.0, double noiseScale = 1.003,
                                       double stepScale = 1.5) {
  size_t atoms = mask.size(), n3 = atoms * 3;
  Normal normal(seed);
  std::vector<double> levels(steps + 1);
  for (int k = 0; k <= steps; ++k) levels[k] = noiseSchedule((double)k / steps);
  std::vector<float> x(n3), rot((size_t)steps * 12), noise((size_t)steps * n3), tHats(steps);
  for (auto& v : x) v = (float)(normal() * levels[0]);
  for (int s = 0; s < steps; ++s) {
    double v0[3] = {normal(), normal(), normal()}, v1[3] = {normal(), normal(), normal()};
    auto norm = [](const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    double e0[3], e1[3], e2[3], s0 = 1 / std::max(1e-10, norm(v0));
    for (int k = 0; k < 3; ++k) e0[k] = v0[k] * s0;
    double dot = v1[0] * e0[0] + v1[1] * e0[1] + v1[2] * e0[2], w[3];
    for (int k = 0; k < 3; ++k) w[k] = v1[k] - e0[k] * dot;
    double s1 = 1 / std::max(1e-10, norm(w));
    for (int k = 0; k < 3; ++k) e1[k] = w[k] * s1;
    e2[0] = e0[1] * e1[2] - e0[2] * e1[1]; e2[1] = e0[2] * e1[0] - e0[0] * e1[2]; e2[2] = e0[0] * e1[1] - e0[1] * e1[0];
    float* r = &rot[(size_t)s * 12];
    for (int k = 0; k < 3; ++k) { r[k] = (float)e0[k]; r[3 + k] = (float)e1[k]; r[6 + k] = (float)e2[k]; }
    for (int k = 0; k < 3; ++k) r[9 + k] = (float)normal();
    for (size_t i = 0; i < n3; ++i) noise[(size_t)s * n3 + i] = (float)normal();
  }
  float* dX = upload(x.data(), n3); float* dRot = upload(rot.data(), rot.size());
  float* dNoise = upload(noise.data(), noise.size()); float* dMask = upload(mask.data(), atoms);
  float* dNoisy = scratch<float>("sample.noisy", n3); float* dC = scratch<float>("sample.centroid", 3);
  for (int s = 0; s < steps; ++s) {
    double previous = levels[s], level = levels[s + 1];
    double tHat = previous * (1 + (level > gammaMin ? gamma0 : 0));
    tHats[s] = (float)tHat;
  }
  float* dLevels = upload(tHats.data(), steps);
  for (int step = 1; step <= steps; ++step) {
    double previous = levels[step - 1], level = levels[step], tHat = tHats[step - 1];
    double injected = noiseScale * std::sqrt(std::max(0.0, tHat * tHat - previous * previous));
    centroidK<<<1, 256, 0, STREAM>>>(dX, dMask, atoms, dC);
    augmentNoiseK<<<blocks(atoms), 256, 0, STREAM>>>(dX, dNoisy, dMask, dC, dRot + (size_t)(step - 1) * 12,
                                                    dNoise + (size_t)(step - 1) * n3, (float)injected, atoms);
    const float* d = denoiseFn(dNoisy, (float)tHat, dLevels + step - 1);
    if (step == 1) { CK(cudaStreamSynchronize(STREAM)); STAGE_MS.clear(); }
    eulerK<<<blocks(n3), 256, 0, STREAM>>>(dX, dNoisy, d, (float)(stepScale * (level - tHat) / tHat), n3);
  }
  std::vector<float> out = download(dX, n3);
  for (float* p : {dX, dRot, dNoise, dMask, dLevels}) CK(cudaFree(p));
  return out;
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
