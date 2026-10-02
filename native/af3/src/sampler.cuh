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
// rotation, translates by a unit normal, injects noise and takes the Euler step. The per-atom
// Gaussians (the start and every step's injected noise) are a counter-based hash of (seed, step,
// element) evaluated where they are used - drawing them on the host was 7.5 million deviates and
// ~200 ms a fold at 522 tokens - and the augmentation's twelve per step come from the host's
// seeded stream. Nothing synchronises until the end. `ns` samples run as one batch (NS), sample k
// seeded `seed + k`: what a single-sample run with that seed draws.
__device__ __forceinline__ uint64_t mix64(uint64_t z) {        // splitmix64's finaliser
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
// a standard normal from (seed, step, index): Box-Muller on two 32-bit uniforms
__device__ __forceinline__ float gaussian(uint64_t seed, uint32_t step, uint64_t index) {
  uint64_t h = mix64(seed * 0x2545f4914f6cdd1dull ^ mix64(((uint64_t)step << 40) ^ index));
  float u1 = ((uint32_t)h + 1.f) * 2.3283064e-10f, u2 = (uint32_t)(h >> 32) * 2.3283064e-10f;
  return sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
}
__global__ void initialNoiseK(float* x, size_t n3, uint64_t seed0, float scale, size_t total) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) x[i] = scale * gaussian(seed0 + i / n3, 0, i % n3);
}
__global__ void centroidK(const float* x, const float* mask, size_t atoms, float* c) {
  __shared__ float s[4][256];
  const float* xs = x + (size_t)blockIdx.x * atoms * 3;         // one block a sample
  float a[4] = {0, 0, 0, 0};
  for (size_t i = threadIdx.x; i < atoms; i += blockDim.x)
    if (mask[i]) { a[0] += xs[i * 3]; a[1] += xs[i * 3 + 1]; a[2] += xs[i * 3 + 2]; a[3] += 1; }
  for (int k = 0; k < 4; ++k) s[k][threadIdx.x] = a[k];
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) for (int k = 0; k < 4; ++k) s[k][threadIdx.x] += s[k][threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x < 3) c[blockIdx.x * 3 + threadIdx.x] = s[threadIdx.x][0] / (s[3][0] + 1e-6f);
}
// rot = e0, e1, e2 (rows) and the translation, 12 a sample; noisy = augmented x + injected * noise
__global__ void augmentNoiseK(float* x, float* noisy, const float* mask, const float* c, const float* rot,
                              uint64_t seed0, uint32_t step, float injected, size_t atoms, size_t total) {
  size_t a = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= total) return;
  size_t k = a / atoms; const float* r = rot + k * 12; const float* ck = c + k * 3;
  float p[3] = {x[a * 3] - ck[0], x[a * 3 + 1] - ck[1], x[a * 3 + 2] - ck[2]};
  bool live = mask[a % atoms] != 0;
  for (int d = 0; d < 3; ++d) {
    float v = live ? p[0] * r[d] + p[1] * r[3 + d] + p[2] * r[6 + d] + r[9 + d] : 0.f;
    x[a * 3 + d] = v;
    noisy[a * 3 + d] = v + injected * gaussian(seed0 + k, step, (a % atoms) * 3 + d);
  }
}
__global__ void eulerK(float* x, const float* noisy, const float* denoised, float scale, size_t n3) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n3) x[i] = noisy[i] + scale * (noisy[i] - denoised[i]);
}
// Returns every sample's positions, sample-major [ns][atoms][3].
inline std::vector<float> sample(int steps, uint64_t seed, const std::vector<float>& mask,
                                 const std::function<const float*(const float*, float, const float*)>& denoiseFn,
                                 int ns = 1, double gamma0 = 0.8, double gammaMin = 1.0, double noiseScale = 1.003,
                                 double stepScale = 1.5,
                                 const std::function<void(const std::vector<float>&)>& onLevels = {}) {
  size_t atoms = mask.size(), n3 = atoms * 3, all3 = n3 * ns;
  // a model's own EDM constants where its dialect carries them (boltz2)
  double sigmaMin = 0.0004, sigmaMax = 160, rho = 7;
  const std::string S = "trunk.dialect.sampler.";
  if (M.has(S + "gamma0")) {
    gamma0 = M.meta(S + "gamma0"); gammaMin = M.meta(S + "gammaMin"); noiseScale = M.meta(S + "noiseScale");
    stepScale = M.meta(S + "stepScale"); rho = M.meta(S + "rho"); sigmaMin = M.meta(S + "sigmaMin"); sigmaMax = M.meta(S + "sigmaMax");
  }
  std::vector<double> levels(steps + 1);
  for (int k = 0; k <= steps; ++k) levels[k] = noiseSchedule((double)k / steps, 16, sigmaMin, sigmaMax, rho);
  std::vector<float> rot((size_t)steps * ns * 12), tHats(steps);
  for (int k = 0; k < ns; ++k) {
    Normal normal(seed + k);
    for (int s = 0; s < steps; ++s) {
      double v0[3] = {normal(), normal(), normal()}, v1[3] = {normal(), normal(), normal()};
      auto norm = [](const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
      double e0[3], e1[3], e2[3], s0 = 1 / std::max(1e-10, norm(v0));
      for (int d = 0; d < 3; ++d) e0[d] = v0[d] * s0;
      double dot = v1[0] * e0[0] + v1[1] * e0[1] + v1[2] * e0[2], w[3];
      for (int d = 0; d < 3; ++d) w[d] = v1[d] - e0[d] * dot;
      double s1 = 1 / std::max(1e-10, norm(w));
      for (int d = 0; d < 3; ++d) e1[d] = w[d] * s1;
      e2[0] = e0[1] * e1[2] - e0[2] * e1[1]; e2[1] = e0[2] * e1[0] - e0[0] * e1[2]; e2[2] = e0[0] * e1[1] - e0[1] * e1[0];
      float* r = &rot[((size_t)s * ns + k) * 12];
      for (int d = 0; d < 3; ++d) { r[d] = (float)e0[d]; r[3 + d] = (float)e1[d]; r[6 + d] = (float)e2[d]; }
      for (int d = 0; d < 3; ++d) r[9 + d] = (float)normal();
    }
  }
  float* dX; CK(cudaMalloc(&dX, all3 * 4));
  initialNoiseK<<<blocks(all3), 256, 0, STREAM>>>(dX, n3, seed, (float)levels[0], all3);
  float* dRot = upload(rot.data(), rot.size()); float* dMask = upload(mask.data(), atoms);
  float* dNoisy = scratch<float>("sample.noisy", all3); float* dC = scratch<float>("sample.centroid", 3 * ns);
  for (int s = 0; s < steps; ++s) {
    double previous = levels[s], level = levels[s + 1];
    tHats[s] = (float)(previous * (1 + (level > gammaMin ? gamma0 : 0)));
  }
  if (onLevels) onLevels(tHats);       // every step's noise level, before the first step
  float* dLevels = upload(tHats.data(), steps);
  for (int step = 1; step <= steps; ++step) {
    double previous = levels[step - 1], level = levels[step], tHat = tHats[step - 1];
    double injected = noiseScale * std::sqrt(std::max(0.0, tHat * tHat - previous * previous));
    centroidK<<<ns, 256, 0, STREAM>>>(dX, dMask, atoms, dC);
    augmentNoiseK<<<blocks(atoms * ns), 256, 0, STREAM>>>(dX, dNoisy, dMask, dC, dRot + (size_t)(step - 1) * ns * 12,
                                                         seed, (uint32_t)step, (float)injected, atoms, atoms * ns);
    const float* d = denoiseFn(dNoisy, (float)tHat, dLevels + step - 1);
    if (step == 1) { CK(cudaStreamSynchronize(STREAM)); STAGE_MS.clear(); }
    eulerK<<<blocks(all3), 256, 0, STREAM>>>(dX, dNoisy, d, (float)(stepScale * (level - tHat) / tHat), all3);
  }
  std::vector<float> out = download(dX, all3);
  for (float* p : {dX, dRot, dMask, dLevels}) CK(cudaFree(p));
  return out;
}

// The PDB: through the exporter's template.pdb when there is one (the page's own records, the
// slot in the x column), else one residue per token named here. `bfactors` per atom slot.
inline std::string DATA_DIR;
// Returns the dense slot of every ATOM/HETATM record, in file order.
inline std::vector<size_t> writePdb(const std::string& path, const std::vector<float>& x, const float* bfactors = nullptr) {
  std::vector<size_t> order;
  std::ifstream tf(DATA_DIR + "/template.pdb");
  if (tf) {
    FILE* f = fopen(path.c_str(), "w");
    std::string line;
    while (std::getline(tf, line)) {
      if (line.size() >= 66 && (!line.compare(0, 4, "ATOM") || !line.compare(0, 6, "HETATM"))) {
        size_t slot = (size_t)std::lround(std::stod(line.substr(30, 8)));
        order.push_back(slot);
        char coords[64];
        snprintf(coords, sizeof coords, "%8.3f%8.3f%8.3f%6.2f%6.2f", x[slot * 3], x[slot * 3 + 1], x[slot * 3 + 2],
                 1.0, bfactors ? bfactors[slot] : 0.0);
        line = line.substr(0, 30) + coords + line.substr(66);
      }
      fprintf(f, "%s\n", line.c_str());
    }
    fclose(f);
    return order;
  }
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
    order.push_back(i);
    char name[5] = {0};
    for (int k = 0; k < 4; ++k) { int c = names[i * 4 + k]; name[k] = c > 0 ? (char)(c + 32) : 0; }
    int z = elem[i]; const char* el = z >= 0 && z < 18 ? ELEM[z] : "X";
    const char* res = aatype[t] >= 0 && aatype[t] < 20 ? RES[aatype[t]] : "UNK";
    fprintf(f, "ATOM  %5d %-4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n", serial++,
            strlen(name) < 4 ? (std::string(" ") + name).c_str() : name, res, 'A' + (chain[t] - 1) % 26,
            resIdx[t], x[i * 3], x[i * 3 + 1], x[i * 3 + 2], 1.0, bfactors ? bfactors[i] : 0.0, el);
  }
  fprintf(f, "END\n");
  fclose(f);
  return order;
}

// AlphaFold 3's confidence files beside a structure: <stem>_confidences.json (atom_plddts in the
// PDB's atom order, pae, token_chain_ids, token_res_ids) and <stem>_summary_confidences.json
// (ptm, iptm, ranking_score with AF3's disorder and clash terms - src/scores.cuh).
// <stem>_confidences.json and <stem>_summary_confidences.json, as the page's archive writes them
// (web/fold-archive.js): the per-atom pLDDTs in the PDB's atom order, the distogram's contact
// probabilities, the PAE; and the scalar scores with their per-chain and per-chain-pair forms, the
// TM ones reduced from the head's per-pair TM term (src/heads/tm-score.js)
inline void writeConfidences(const std::string& pdbPath, const std::vector<size_t>& order, int n, int dense,
                             const std::vector<float>& plddt, const std::vector<float>& pae,
                             const std::vector<float>& tmTerm, const std::vector<float>& contact,
                             double ptm, double iptm, double ranking, double meanPlddt,
                             bool hasClash, double fractionDisordered) {
  std::string stem = pdbPath.size() > 4 && pdbPath.substr(pdbPath.size() - 4) == ".pdb"
    ? pdbPath.substr(0, pdbPath.size() - 4) : pdbPath;
  const int* asym = M.i("batch.asymId"); const int* res = M.i("batch.residueIndex");
  const float* seq = M.f("batch.seqMask");
  auto chainId = [](int a) {
    std::string id; for (a = a - 1; ; a = a / 26 - 1) { id.insert(id.begin(), (char)('A' + a % 26)); if (a < 26) break; }
    return id;
  };
  auto matrix = [&](FILE* f, const std::vector<float>& m) {
    fprintf(f, "[");
    for (int i = 0; i < n; ++i) {
      fprintf(f, "%s[", i ? ",\n  " : "");
      for (int j = 0; j < n; ++j) fprintf(f, "%s%.2f", j ? ", " : "", m[(size_t)i * n + j]);
      fprintf(f, "]");
    }
    fprintf(f, "]");
  };
  FILE* f = fopen((stem + "_confidences.json").c_str(), "w");
  fprintf(f, "{\"atom_chain_ids\": [");
  for (size_t k = 0; k < order.size(); ++k) fprintf(f, "%s\"%s\"", k ? ", " : "", chainId(asym[order[k] / dense]).c_str());
  fprintf(f, "],\n \"atom_plddts\": [");
  for (size_t k = 0; k < order.size(); ++k) fprintf(f, "%s%.2f", k ? ", " : "", plddt[order[k]]);
  fprintf(f, "],\n");
  if (!contact.empty()) { fprintf(f, " \"contact_probs\": "); matrix(f, contact); fprintf(f, ",\n"); }
  fprintf(f, " \"pae\": "); matrix(f, pae);
  fprintf(f, ",\n \"token_chain_ids\": [");
  for (int i = 0; i < n; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", chainId(asym[i]).c_str());
  fprintf(f, "],\n \"token_res_ids\": [");
  for (int i = 0; i < n; ++i) fprintf(f, "%s%d", i ? ", " : "", res[i]);
  fprintf(f, "]}\n");
  fclose(f);

  // the chains, by asym id in token order
  std::vector<int> chains;
  for (int i = 0; i < n; ++i) if (seq[i] > 0 && std::find(chains.begin(), chains.end(), asym[i]) == chains.end()) chains.push_back(asym[i]);
  std::sort(chains.begin(), chains.end());
  int nc = (int)chains.size();
  // max over anchors of the mean over the selected pairs (NaN when nothing is selected)
  auto reduce = [&](const std::function<bool(int, int)>& selects) {
    double best = -1e30; bool any = false;
    for (int i = 0; i < n; ++i) {
      double total = 0; int count = 0;
      for (int j = 0; j < n; ++j) {
        if (!(seq[i] > 0 && seq[j] > 0) || !selects(i, j)) continue;
        total += tmTerm[(size_t)i * n + j]; ++count;
      }
      if (count) { any = true; best = std::max(best, total / count); }
    }
    return any ? best : NAN;
  };
  auto num = [](FILE* f, double v) { if (std::isfinite(v)) fprintf(f, "%.2f", v); else fprintf(f, "null"); };
  std::vector<double> chainPtm(nc, NAN), chainIptm(nc, NAN);
  for (int a = 0; a < nc; ++a) {
    int c = chains[a];
    chainPtm[a] = reduce([&](int i, int j) { return asym[i] == c && asym[j] == c; });
    chainIptm[a] = reduce([&](int i, int j) { return (asym[i] == c) != (asym[j] == c); });
  }
  f = fopen((stem + "_summary_confidences.json").c_str(), "w");
  fprintf(f, "{\n  \"chain_ids\": [");
  for (int i = 0; i < n; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", chainId(asym[i]).c_str());
  fprintf(f, "],\n");
  if (!tmTerm.empty()) {
    fprintf(f, "  \"chain_pair_iptm\": [");
    for (int a = 0; a < nc; ++a) {
      fprintf(f, "%s[", a ? ", " : "");
      for (int b = 0; b < nc; ++b) {
        if (b) fprintf(f, ", ");
        if (a == b) { num(f, chainPtm[a]); continue; }        // the diagonal is the chain's own pTM
        int ca = chains[a], cb = chains[b];
        num(f, reduce([&](int i, int j) { return (asym[i] == ca && asym[j] == cb) || (asym[i] == cb && asym[j] == ca); }));
      }
      fprintf(f, "]");
    }
    fprintf(f, "],\n");
  }
  if (!contact.empty()) {      // the strongest predicted contact within and between chains, sequence neighbours out
    std::vector<double> mx((size_t)nc * nc, -1);
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
      int a = (int)(std::find(chains.begin(), chains.end(), asym[i]) - chains.begin());
      int b = (int)(std::find(chains.begin(), chains.end(), asym[j]) - chains.begin());
      if (a >= nc || b >= nc || (a == b && std::abs(res[i] - res[j]) <= 6)) continue;
      double v = contact[(size_t)i * n + j];
      if (v > mx[(size_t)a * nc + b]) mx[(size_t)a * nc + b] = mx[(size_t)b * nc + a] = v;
    }
    fprintf(f, "  \"chain_pair_max_contact\": [");
    for (int a = 0; a < nc; ++a) {
      fprintf(f, "%s[", a ? ", " : "");
      for (int b = 0; b < nc; ++b) { if (b) fprintf(f, ", "); num(f, mx[(size_t)a * nc + b] < 0 ? NAN : mx[(size_t)a * nc + b]); }
      fprintf(f, "]");
    }
    fprintf(f, "],\n");
  }
  {   // mean atom pLDDT per chain, over the structure's atoms
    std::vector<double> sum(nc, 0), cnt(nc, 0);
    for (size_t k = 0; k < order.size(); ++k) {
      int a = (int)(std::find(chains.begin(), chains.end(), asym[order[k] / dense]) - chains.begin());
      float v = plddt[order[k]];
      if (a < nc) { sum[a] += v; cnt[a] += 1; }
    }
    fprintf(f, "  \"chain_plddt\": [");
    for (int a = 0; a < nc; ++a) { if (a) fprintf(f, ", "); num(f, cnt[a] ? sum[a] / cnt[a] : NAN); }
    fprintf(f, "],\n");
    if (!tmTerm.empty()) {
      fprintf(f, "  \"chain_ptm\": [");
      for (int a = 0; a < nc; ++a) { if (a) fprintf(f, ", "); num(f, chainPtm[a]); }
      fprintf(f, "],\n");
      if (nc > 1) {
        fprintf(f, "  \"chain_iptm\": [");
        for (int a = 0; a < nc; ++a) { if (a) fprintf(f, ", "); num(f, chainIptm[a]); }
        fprintf(f, "],\n");
      }
    }
    // the minimum PAE over ordered pairs, row in one chain and column in the other
    std::vector<double> mn((size_t)nc * nc, INFINITY);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
      int a = (int)(std::find(chains.begin(), chains.end(), asym[i]) - chains.begin());
      int b = (int)(std::find(chains.begin(), chains.end(), asym[j]) - chains.begin());
      if (a < nc && b < nc) mn[(size_t)a * nc + b] = std::min(mn[(size_t)a * nc + b], (double)pae[(size_t)i * n + j]);
    }
    fprintf(f, "  \"chain_pair_pae_min\": [");
    for (int a = 0; a < nc; ++a) {
      fprintf(f, "%s[", a ? ", " : "");
      for (int b = 0; b < nc; ++b) { if (b) fprintf(f, ", "); num(f, mn[(size_t)a * nc + b]); }
      fprintf(f, "]");
    }
    fprintf(f, "],\n");
    if (std::isfinite(iptm)) fprintf(f, "  \"iptm\": %.2f,\n", iptm);
    fprintf(f, "  \"ptm\": %.2f,\n  \"ranking_score\": %.2f,\n", ptm, ranking);
    fprintf(f, "  \"fraction_disordered\": %.2f,\n  \"has_clash\": %.1f,\n  \"mean_plddt\": %.2f\n}\n",
            fractionDisordered, hasClash ? 1.0 : 0.0, meanPlddt);
  }
  fclose(f);
}
