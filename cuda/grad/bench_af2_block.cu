// The AF2 training block's cost at a training shape: forward with saves, backward, and what the arena holds -
// against a stack's input saves. Random weights and inputs (no oracle: cuda/grad/test_af2_block.cu is the gate).
//   cuda/grad/bench-af2-block [--L=256] [--S=128] [--extra] [--tf32] [--blocks=48]
#include <cstring>
#include <curand.h>
#include <cuda_profiler_api.h>
#include "af2.cuh"
using namespace grad;
int main(int argc, char** argv) {
  int L = 256, S = 128, nblocks = 48; bool extra = false;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--L=", 4)) L = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--S=", 4)) S = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--blocks=", 9)) nblocks = atoi(argv[i] + 9);
    else if (!strcmp(argv[i], "--extra")) extra = true;
    else if (!strcmp(argv[i], "--tf32")) TF32 = true;
  }
  init();
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 7);
  auto randn = [&](size_t n, float sd) { float* p; GK(cudaMalloc(&p, ((n + 1) & ~1) * 4)); curandGenerateNormal(gen, p, (n + 1) & ~1, 0.f, sd); return p; };
  af2::Block b; b.extra = extra; b.Cm = extra ? 64 : 256;
  af2::forEachParam(b, [&](const std::string& n, Param& p, size_t len) {
    p.name = n; p.n = len; p.w = randn(len, 0.05f); GK(cudaMalloc(&p.g, len * 4)); b.params.add(p); });
  for (auto* p : b.params.all) if (p->name.find("/scale") != std::string::npos) { std::vector<float> one(p->n, 1.f); GK(cudaMemcpy(p->w, one.data(), p->n * 4, cudaMemcpyHostToDevice)); }
  size_t mr = (size_t)S * L * b.Cm, pr = (size_t)L * L * 128;
  float *msa = randn(mr, 1.f), *pair = randn(pr, 1.f), *ctm = randn(mr, 1.f), *ctp = randn(pr, 1.f);
  std::vector<float> ones(std::max((size_t)S * L, (size_t)L * L), 1.f);
  float *mm_, *pm; GK(cudaMalloc(&mm_, (size_t)S * L * 4)); GK(cudaMalloc(&pm, (size_t)L * L * 4));
  GK(cudaMemcpy(mm_, ones.data(), (size_t)S * L * 4, cudaMemcpyHostToDevice)); GK(cudaMemcpy(pm, ones.data(), (size_t)L * L * 4, cudaMemcpyHostToDevice));
  float *mo, *po, *dm, *dp; GK(cudaMalloc(&mo, mr * 4)); GK(cudaMalloc(&po, pr * 4)); GK(cudaMalloc(&dm, mr * 4)); GK(cudaMalloc(&dp, pr * 4));
  cudaEvent_t e0, e1, e2, e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
  float f = 0, fs = 0, bw = 0;
  for (int r = 0; r < 4; ++r) {
    b.params.zeroGrad(); ARENA.reset(); ARENA.high = 0;
    cudaEventRecord(e0, STREAM);
    { af2::BlockSave s = af2::blockForward(b, msa, pair, mm_, pm, mo, po, S, L); (void)s; }  // the stack's plain forward
    cudaEventRecord(e1, STREAM);
    ARENA.reset();
    af2::BlockSave s = af2::blockForward(b, msa, pair, mm_, pm, mo, po, S, L);              // the replay, with saves
    cudaEventRecord(e2, STREAM);
    if (r == 3) cudaProfilerStart();                                                        // (nsys --capture-range=cudaProfilerApi)
    af2::blockBackward(b, s, mm_, pm, ctm, ctp, dm, dp);
    cudaEventRecord(e3, STREAM);
    if (r == 3) { GK(cudaStreamSynchronize(STREAM)); cudaProfilerStop(); }
    GK(cudaStreamSynchronize(STREAM));
    cudaEventElapsedTime(&f, e0, e1); cudaEventElapsedTime(&fs, e1, e2); cudaEventElapsedTime(&bw, e2, e3);
  }
  GK(cudaGetLastError());
  double inBytes = (double)(mr + pr) * 4;
  printf("L %d S %d Cm %d (%s): forward %.1f ms, replay %.1f ms, backward %.1f ms -> %.1f ms a block trained (%.2f s for %d blocks)\n",
         L, S, b.Cm, extra ? "extra" : "main", f, fs, bw, f + fs + bw, (f + fs + bw) * nblocks / 1000, nblocks);
  printf("  one block's activations (arena high water) %.0f MiB; the stack's saved inputs %.0f MiB (%d blocks)\n",
         ARENA.high / 1048576.0, inBytes * (nblocks + 1) / 1048576.0, nblocks);
}
