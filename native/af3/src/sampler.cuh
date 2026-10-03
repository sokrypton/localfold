// AF3's EDM sampler around the denoiser, and the PDB the fold writes.
// Transcribed from src/af3/diffusion/diffusion-sampler-reference.js.
#pragma once
#include "diffusion.cuh"
#include <random>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

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
// seeded stream. Nothing synchronises until the end. The samples run as one batch (NS), each with
// its own seed: sample k of seed s draws from sampleSeed(s, k) = s + k 2^32 - sample 0 is what a
// single-sample run with that seed draws, and no two (seed, sample) pairs of distinct seeds below
// 2^32 share a stream (AF3's seeds are small integers, and five samples of seeds 1 and 2 would
// overlap under s + k).
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
inline uint64_t sampleSeed(uint64_t seed, int k) { return seed + ((uint64_t)k << 32); }
__global__ void initialNoiseK(float* x, size_t n3, const uint64_t* seeds, float scale, size_t total) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) x[i] = scale * gaussian(seeds[i / n3], 0, i % n3);
}
// one block of 1024 a sample (256 left a 1044-token fold's 25k atoms at 85 us a step), a fixed
// reduction order so the centre is the same every run
__global__ void __launch_bounds__(1024) centroidK(const float* x, const float* mask, size_t atoms, float* c) {
  __shared__ float s[4][32];
  const float* xs = x + (size_t)blockIdx.x * atoms * 3;         // one block a sample
  float a[4] = {0, 0, 0, 0};
  for (size_t i = threadIdx.x; i < atoms; i += blockDim.x)
    if (mask[i]) { a[0] += xs[i * 3]; a[1] += xs[i * 3 + 1]; a[2] += xs[i * 3 + 2]; a[3] += 1; }
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int k = 0; k < 4; ++k) {
    for (int o = 16; o; o >>= 1) a[k] += __shfl_xor_sync(~0u, a[k], o);
    if (lane == 0) s[k][warp] = a[k];
  }
  __syncthreads();
  if (warp == 0) {
    int nw = blockDim.x >> 5;
    for (int k = 0; k < 4; ++k) {
      float v = lane < nw ? s[k][lane] : 0.f;
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      a[k] = v;
    }
    if (lane < 3) c[blockIdx.x * 3 + lane] = (lane == 0 ? a[0] : lane == 1 ? a[1] : a[2]) / (a[3] + 1e-6f);
  }
}
// rot = e0, e1, e2 (rows) and the translation, 12 a sample; noisy = augmented x + injected * noise
__global__ void augmentNoiseK(float* x, float* noisy, const float* mask, const float* c, const float* rot,
                              const uint64_t* seeds, uint32_t step, float injected, size_t atoms, size_t total) {
  size_t a = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= total) return;
  size_t k = a / atoms; const float* r = rot + k * 12; const float* ck = c + k * 3;
  float p[3] = {x[a * 3] - ck[0], x[a * 3 + 1] - ck[1], x[a * 3 + 2] - ck[2]};
  bool live = mask[a % atoms] != 0;
  for (int d = 0; d < 3; ++d) {
    float v = live ? p[0] * r[d] + p[1] * r[3 + d] + p[2] * r[6 + d] + r[9 + d] : 0.f;
    x[a * 3 + d] = v;
    noisy[a * 3 + d] = v + injected * gaussian(seeds[k], step, (a % atoms) * 3 + d);
  }
}
__global__ void eulerK(float* x, const float* noisy, const float* denoised, float scale, size_t n3) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n3) x[i] = noisy[i] + scale * (noisy[i] - denoised[i]);
}
// --flow: the page's Flow (src/af3/diffusion/diffusion-sampler-webgpu.js flowOnGpu, step "replace") in
// place of AF3's diffusion - one draw at the top of a schedule that starts at 160 A (sigmaMax 10 sigma_data,
// AF3's own sigmaMin and rho, whatever the model's dialect), then the state REPLACED by each prediction:
// no centring, no rotation, no injected noise
inline bool SAMPLER_FLOW = false;
// --sigma-max=X: where the diffusion schedule starts, in sigma_data units (AF3's own is 160) - the page's
// short schedule (web/af3-model.js diffusionScheduleFor: 80 for a plain protein, 40 for a ligand job, on
// the families its measurements favour), handed to a CUDA fold so it samples what the page samples. 0 is
// the model's own. Flow keeps its own start.
inline double SAMPLER_SIGMA_MAX = 0;
// what a sampler step's prediction is handed to (FrameStreamer, below): the denoised positions on the device
// and the step, called on the host between steps - it must not wait on the GPU
inline std::function<void(const float*, int, int)> FRAME_HOOK;
// Returns every sample's positions, sample-major [ns][atoms][3].
inline std::vector<float> sample(int steps, const std::vector<uint64_t>& seeds, const std::vector<float>& mask,
                                 const std::function<const float*(const float*, float, const float*)>& denoiseFn,
                                 double gamma0 = 0.8, double gammaMin = 1.0, double noiseScale = 1.003,
                                 double stepScale = 1.5,
                                 const std::function<void(const std::vector<float>&)>& onLevels = {}) {
  int ns = (int)seeds.size();
  size_t atoms = mask.size(), n3 = atoms * 3, all3 = n3 * ns;
  // a model's own EDM constants where its dialect carries them (boltz2)
  double sigmaMin = 0.0004, sigmaMax = 160, rho = 7;
  const std::string S = "trunk.dialect.sampler.";
  if (M.has(S + "gamma0")) {
    gamma0 = M.meta(S + "gamma0"); gammaMin = M.meta(S + "gammaMin"); noiseScale = M.meta(S + "noiseScale");
    stepScale = M.meta(S + "stepScale"); rho = M.meta(S + "rho"); sigmaMin = M.meta(S + "sigmaMin"); sigmaMax = M.meta(S + "sigmaMax");
  }
  std::vector<double> levels(steps + 1);
  if (SAMPLER_FLOW) {
    for (int k = 0; k <= steps; ++k) levels[k] = noiseSchedule((double)k / steps, 16, 0.0004, 10, 7);
    std::vector<float> at(levels.begin(), levels.begin() + steps);
    if (onLevels) onLevels(at);
    float* dX; CK(cudaMalloc(&dX, all3 * 4));
    uint64_t* dSeeds; CK(cudaMalloc(&dSeeds, ns * 8));
    CK(cudaMemcpyAsync(dSeeds, seeds.data(), ns * 8, cudaMemcpyHostToDevice, STREAM));
    initialNoiseK<<<blocks(all3), 256, 0, STREAM>>>(dX, n3, dSeeds, (float)levels[0], all3);
    float* dLevels = upload(at.data(), steps);
    for (int step = 1; step <= steps; ++step) {
      const float* d = denoiseFn(dX, (float)levels[step - 1], dLevels + step - 1);
      if (step == 1) { CK(cudaStreamSynchronize(STREAM)); STAGE_MS.clear(); }
      if (FRAME_HOOK) FRAME_HOOK(d, step, steps);
      CK(cudaMemcpyAsync(dX, d, all3 * 4, cudaMemcpyDeviceToDevice, STREAM));
    }
    std::vector<float> out = download(dX, all3);
    for (float* p : {dX, dLevels}) CK(cudaFree(p));
    CK(cudaFree(dSeeds));
    return out;
  }
  if (SAMPLER_SIGMA_MAX > 0) sigmaMax = SAMPLER_SIGMA_MAX;
  for (int k = 0; k <= steps; ++k) levels[k] = noiseSchedule((double)k / steps, 16, sigmaMin, sigmaMax, rho);
  std::vector<float> rot((size_t)steps * ns * 12), tHats(steps);
  for (int k = 0; k < ns; ++k) {
    Normal normal(seeds[k]);
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
  uint64_t* dSeeds; CK(cudaMalloc(&dSeeds, ns * 8));
  CK(cudaMemcpyAsync(dSeeds, seeds.data(), ns * 8, cudaMemcpyHostToDevice, STREAM));
  initialNoiseK<<<blocks(all3), 256, 0, STREAM>>>(dX, n3, dSeeds, (float)levels[0], all3);
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
    centroidK<<<ns, 1024, 0, STREAM>>>(dX, dMask, atoms, dC);
    augmentNoiseK<<<blocks(atoms * ns), 256, 0, STREAM>>>(dX, dNoisy, dMask, dC, dRot + (size_t)(step - 1) * ns * 12,
                                                         dSeeds, (uint32_t)step, (float)injected, atoms, atoms * ns);
    const float* d = denoiseFn(dNoisy, (float)tHat, dLevels + step - 1);
    if (step == 1) { CK(cudaStreamSynchronize(STREAM)); STAGE_MS.clear(); }
    if (FRAME_HOOK) FRAME_HOOK(d, step, steps);
    eulerK<<<blocks(all3), 256, 0, STREAM>>>(dX, dNoisy, d, (float)(stepScale * (level - tHat) / tHat), all3);
  }
  std::vector<float> out = download(dX, all3);
  for (float* p : {dX, dRot, dMask, dLevels}) CK(cudaFree(p));
  CK(cudaFree(dSeeds));
  return out;
}

// A float32 array as a NumPy .npy file (format 1.0: magic, header dict padded to 64 bytes, data)
inline void writeNpy(const std::string& path, const std::vector<float>& data, const std::vector<size_t>& shape) {
  std::string dims;
  for (size_t k = 0; k < shape.size(); ++k) dims += std::to_string(shape[k]) + (shape.size() == 1 || k + 1 < shape.size() ? "," : "");
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
  size_t total = 10 + header.size() + 1;
  header += std::string((64 - total % 64) % 64, ' ') + "\n";
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
  const char magic[] = "\x93NUMPY\x01\x00";
  fwrite(magic, 1, 8, f);
  uint16_t len = (uint16_t)header.size(); fwrite(&len, 2, 1, f);
  fwrite(header.data(), 1, header.size(), f);
  fwrite(data.data(), 4, data.size(), f);
  fclose(f);
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

// --frames=DIR: each sampler step's prediction (the denoised positions, the picture the page draws) written to
// DIR/frame-NNNN.pdb WITHOUT the fold waiting for it. The compute stream only snapshots the positions into a
// free slot of a small device ring (a device-to-device copy of a few hundred KB at most) and records an
// event; a second stream - the copy engine - waits on that event and copies the slot to pinned host memory;
// a host thread waits for THAT copy, writes the PDB through the input's template and frees the slot. When
// no slot is free (the writer has fallen behind) the frame is DROPPED: a frame is a picture, and the fold
// never waits for one. Frames are written whole (a temporary file renamed into place).
struct FrameStreamer {
  // 🔴 THE FRAMES ARE PLANNED, NOT CAUGHT: the host queues a fold's steps far ahead of the GPU (a 25-step
  // diffusion is enqueued in a few ms and runs for 50), so a slot cannot free before the next step is
  // offered - a ring of three caught 4 frames of 25. So at most MAX frames, evenly spaced and always the
  // last step, each with its own slot of the tap (common.cuh's AsyncTap, reserved before the fold).
  static constexpr int MAX = 25;
  std::string dir; size_t n3 = 0; int stride = 1, steps = 0, written = 0;
  std::vector<std::string> lines; std::vector<size_t> slots;     // the input's template.pdb, read once
  std::vector<double> reference; double refCentre[3] = {};
  static bool isAtom(const std::string& l) { return l.size() >= 66 && (!l.compare(0, 4, "ATOM") || !l.compare(0, 6, "HETATM")); }
  static int planned(int steps) { return std::min(steps, MAX) + 1; }
  bool start(const std::string& d, size_t atoms3, int totalSteps) {
    std::ifstream tf(DATA_DIR + "/template.pdb");
    if (!tf) { fprintf(stderr, "frames: no %s/template.pdb, none written\n", DATA_DIR.c_str()); return false; }
    for (std::string l; std::getline(tf, l);) {
      lines.push_back(l);
      if (isAtom(l)) slots.push_back((size_t)std::lround(std::stod(l.substr(30, 8))));
    }
    dir = d; n3 = atoms3; steps = totalSteps;
    stride = std::max(1, (steps + MAX - 1) / MAX);
    FRAME_HOOK = [this](const float* p, int step, int) {
      if (step % stride == 0 || step == steps)
        TAP().offer({{p, n3 * 4}}, [this, step](const char* host, const std::vector<size_t>&) { write((const float*)host, step); });
    };
    return true;
  }
  // (the tap's thread) the template's atoms, superposed onto the first frame's - the page draws its own
  // sampler frames fitted to the first: the prediction moves as it settles, not as the walk rotates it
  void write(const float* x, int step) {
    std::vector<double> pts(slots.size() * 3);
    double c[3] = {};
    for (size_t a = 0; a < slots.size(); ++a)
      for (int d = 0; d < 3; ++d) { pts[a * 3 + d] = x[slots[a] * 3 + d]; c[d] += pts[a * 3 + d] / slots.size(); }
    for (size_t a = 0; a < slots.size(); ++a) for (int d = 0; d < 3; ++d) pts[a * 3 + d] -= c[d];
    if (reference.empty()) { reference = pts; for (int d = 0; d < 3; ++d) refCentre[d] = c[d]; }
    double R[9]; bestRotation(pts, reference, R);
    std::string out; out.reserve(lines.size() * 82);
    size_t a = 0; char buf[32];
    for (auto& line : lines) {
      if (a < slots.size() && isAtom(line)) {
        const double* p = &pts[a * 3];
        out += line.substr(0, 30);
        for (int d = 0; d < 3; ++d) {
          snprintf(buf, sizeof buf, "%8.3f", R[d * 3] * p[0] + R[d * 3 + 1] * p[1] + R[d * 3 + 2] * p[2] + refCentre[d]);
          out += buf;
        }
        out += line.substr(54); out += '\n'; ++a;
      } else { out += line; out += '\n'; }
    }
    char name[64]; snprintf(name, sizeof name, "/frame-%04d.pdb", step);
    writeWhole(dir + name, out.data(), out.size());
    ++written;
  }
  void finish() { FRAME_HOOK = nullptr; TAP().drain(); }
};

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
  std::string stem = pdbPath.size() > 4 && (pdbPath.substr(pdbPath.size() - 4) == ".pdb" ||
                                            pdbPath.substr(pdbPath.size() - 4) == ".cif")
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
