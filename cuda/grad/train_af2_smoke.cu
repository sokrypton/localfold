// End to end: the rematerialised AF2 stack, AdamW, a loss. The weights of a dump (cuda/grad/oracle_af2_block.py)
// are perturbed by 10% and trained back to reproduce the unperturbed stack's outputs on the dump's input
// (0.5 mean squared error over msa and pair). The loss must fall; it is the whole loop - forward, replayed
// backward into every parameter, clip, update - and not a gradient check (test_af2_block.cu is that).
//   cuda/grad/train-af2-smoke cuda/grad/data/af2-stack3 [--steps=60] [--lr=3e-4] [--tf32]
#include <cstring>
#include <curand.h>
#include <fstream>
#include <map>
#include <sstream>
#include "adam.cuh"
#include "af2.cuh"
using namespace grad;

static std::map<std::string, std::vector<float>> readDump(const std::string& dir) {
  std::map<std::string, std::vector<float>> t;
  std::ifstream idx(dir + "/model.idx"), bin(dir + "/model.bin", std::ios::binary);
  std::string line;
  while (std::getline(idx, line)) {
    std::istringstream ls(line); std::string kind, name; ls >> kind >> name;
    if (kind != "t") continue;
    size_t off, len; ls >> off >> len;
    std::vector<float> v(len); bin.seekg((std::streamoff)off * 4); bin.read((char*)v.data(), (std::streamsize)len * 4);
    t[name] = std::move(v);
  }
  return t;
}
static float* up(const std::vector<float>& h) { float* d; GK(cudaMalloc(&d, h.size() * 4)); GK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return d; }

// d = (y - target) / n ; loss partials
__global__ void mseK(const float* y, const float* tgt, float* d, size_t n, float inv, double* loss) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  double s = 0;
  if (t < n) { float e = y[t] - tgt[t]; d[t] = e * inv; s = 0.5 * (double)e * e * inv; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(loss, s);
}
__global__ void perturbK(float* w, const float* z, size_t n, float sd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) w[t] *= 1.f + sd * z[t];
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: train-af2-smoke <dump> [--steps=60] [--lr=3e-4] [--tf32]\n"); return 2; }
  int steps = 60; float lr = 3e-4f;
  for (int i = 2; i < argc; ++i) {
    if (!strncmp(argv[i], "--steps=", 8)) steps = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--lr=", 5)) lr = atof(argv[i] + 5);
    else if (!strcmp(argv[i], "--tf32")) TF32 = true;
  }
  init();
  auto d = readDump(argv[1]);
  const int L = (int)d["meta/L"][0], S = (int)d["meta/S"][0], Cm = (int)d["meta/cm"][0];
  const int nb = d.count("meta/nblocks") ? (int)d["meta/nblocks"][0] : 1;
  af2::Stack st; st.blocks.resize(nb);
  Params all;
  for (int k = 0; k < nb; ++k) {
    auto& b = st.blocks[k];
    b.extra = d["meta/extra"][0] > 0.5f; b.opmFirst = d["meta/opm_first"][0] > 0.5f; b.Cm = Cm;
    std::string pre = nb == 1 ? "" : "b" + std::to_string(k) + "/";
    af2::forEachParam(b, [&](const std::string& n, Param& p, size_t len) {
      allocParam(p, pre + n, len); GK(cudaMemcpy(p.w, d["p/" + pre + n].data(), len * 4, cudaMemcpyHostToDevice)); all.add(p); });
  }
  float *msa = up(d["in/msa"]), *pair = up(d["in/pair"]), *mm_ = up(d["in/msa_mask"]), *pm = up(d["in/pair_mask"]);
  size_t mr = (size_t)S * L * Cm, pr = (size_t)L * L * 128;
  // the target: the unperturbed stack's outputs
  float *tm, *tp; GK(cudaMalloc(&tm, mr * 4)); GK(cudaMalloc(&tp, pr * 4));
  st.forward(msa, pair, mm_, pm, S, L);
  GK(cudaMemcpy(tm, st.msaIn.back(), mr * 4, cudaMemcpyDeviceToDevice));
  GK(cudaMemcpy(tp, st.pairIn.back(), pr * 4, cudaMemcpyDeviceToDevice));
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 11);
  for (auto* p : all.all) {
    if (p->name.find("/scale") != std::string::npos || p->name.find("/offset") != std::string::npos) continue;
    float* z; size_t n2 = (p->n + 1) & ~1; GK(cudaMalloc(&z, n2 * 4)); curandGenerateNormal(gen, z, n2, 0.f, 1.f);
    perturbK<<<blocks(p->n), 256>>>(p->w, z, p->n, 0.1f); GK(cudaFree(z));
  }
  AdamW opt; opt.lr = lr; opt.clip = 1.f;
  float *dm, *dp; GK(cudaMalloc(&dm, mr * 4)); GK(cudaMalloc(&dp, pr * 4));
  double* loss; GK(cudaMalloc(&loss, 8));
  double first = 0, last = 0;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  float ms = 0;
  for (int it = 0; it <= steps; ++it) {
    cudaEventRecord(e0, STREAM);
    all.zeroGrad();
    st.forward(msa, pair, mm_, pm, S, L);
    GK(cudaMemsetAsync(loss, 0, 8, STREAM));
    mseK<<<blocks(mr), 256, 0, STREAM>>>(st.msaIn.back(), tm, dm, mr, 1.f / (mr + pr), loss);
    mseK<<<blocks(pr), 256, 0, STREAM>>>(st.pairIn.back(), tp, dp, pr, 1.f / (mr + pr), loss);
    double h; GK(cudaMemcpyAsync(&h, loss, 8, cudaMemcpyDeviceToHost, STREAM)); GK(cudaStreamSynchronize(STREAM));
    if (it == 0) first = h;
    last = h;
    if (it == steps) break;
    st.backward(mm_, pm, dm, dp);
    opt.update(all);
    cudaEventRecord(e1, STREAM); GK(cudaStreamSynchronize(STREAM));
    float a; cudaEventElapsedTime(&a, e0, e1); ms += a;
    if (it % 10 == 0) printf("  step %3d  loss %.6e  |g| %.3e\n", it, h, opt.lastNorm);
  }
  GK(cudaGetLastError());
  printf("%s: loss %.4e -> %.4e (%.1fx) in %d steps, %.1f ms a step (%d blocks, L %d, S %d)\n",
         last < 0.5 * first ? "PASS" : "FAIL", first, last, first / last, steps, ms / steps, nb, L, S);
  return last < 0.5 * first ? 0 : 1;
}
