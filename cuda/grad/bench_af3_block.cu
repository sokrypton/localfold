// The AF3 training pairformer block's cost at a training shape (384 tokens, AF3's widths: Cz 128, Cs 384, pair
// attention 4 x 32, single attention 16 x 24): forward, the replay with saves, the backward. Random weights and
// inputs, all masks one (cuda/grad/test_af3.cu is the gate). bench_torch_af3_block.py is the same block in PyTorch.
//   cuda/grad/bench-af3-block [--n=384] [--tf32]
#include <cstring>
#include <curand.h>
#include <cuda_profiler_api.h>
#include "af3.cuh"
using namespace grad;
int main(int argc, char** argv) {
  int n = 384;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--n=", 4)) n = atoi(argv[i] + 4);
    else if (!strcmp(argv[i], "--tf32")) TF32 = true;
    else if (!strcmp(argv[i], "--bf16")) BF16 = true;
  }
  init();
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 7);
  auto randn = [&](size_t m, float sd) { float* p; GK(cudaMalloc(&p, ((m + 1) & ~1) * 4)); curandGenerateNormal(gen, p, (m + 1) & ~1, 0.f, sd); return p; };
  const int Cz = 128, Cs = 384, Hp = 4, Dp = 32, Hs = 16, Ds = 24;
  af3::Pairformer b; af3::Dialect d; Params ps;
  af3::forEachParam(b, "blk", Cz, Cs, Hp, Dp, Hs, Ds, 4, 4, false, true, [&](const std::string& nm, Param& p, size_t len) {
    p.name = nm; p.n = len; p.w = randn(len, 0.05f); GK(cudaMalloc(&p.g, len * 4)); ps.add(p); });
  for (auto* p : ps.all) if (p->name.find("Scale") != std::string::npos) { std::vector<float> one(p->n, 1.f); GK(cudaMemcpy(p->w, one.data(), p->n * 4, cudaMemcpyHostToDevice)); }
  size_t nn = (size_t)n * n;
  float *pair = randn(nn * Cz, 1.f), *sing = randn((size_t)n * Cs, 1.f), *ctp = randn(nn * Cz, 1.f), *cts = randn((size_t)n * Cs, 1.f);
  std::vector<float> ones(nn, 1.f);
  float *pm, *seq; GK(cudaMalloc(&pm, nn * 4)); GK(cudaMalloc(&seq, n * 4));
  GK(cudaMemcpy(pm, ones.data(), nn * 4, cudaMemcpyHostToDevice)); GK(cudaMemcpy(seq, ones.data(), n * 4, cudaMemcpyHostToDevice));
  float *po, *so, *dp, *ds; GK(cudaMalloc(&po, nn * Cz * 4)); GK(cudaMalloc(&so, (size_t)n * Cs * 4)); GK(cudaMalloc(&dp, nn * Cz * 4)); GK(cudaMalloc(&ds, (size_t)n * Cs * 4));
  cudaEvent_t e0, e1, e2, e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
  std::vector<float> fv, rv, bv;
  for (int r = 0; r < 7; ++r) {
    ps.zeroGrad(); ARENA.reset(); ARENA.high = 0;
    cudaEventRecord(e0, STREAM);
    { af3::pairformerForward(b, d, pair, sing, pm, pm, seq, nullptr, po, so, n); }   // the stack's plain forward
    cudaEventRecord(e1, STREAM);
    ARENA.reset();
    auto s = af3::pairformerForward(b, d, pair, sing, pm, pm, seq, nullptr, po, so, n);   // the replay, with saves
    cudaEventRecord(e2, STREAM);
    if (r == 6) cudaProfilerStart();
    af3::pairformerBackward(b, d, s, pm, pm, seq, ctp, cts, dp, ds, nullptr);
    cudaEventRecord(e3, STREAM);
    GK(cudaStreamSynchronize(STREAM));
    if (r == 6) cudaProfilerStop();
    float f, fs, bw; cudaEventElapsedTime(&f, e0, e1); cudaEventElapsedTime(&fs, e1, e2); cudaEventElapsedTime(&bw, e2, e3);
    if (r >= 2) { fv.push_back(f); rv.push_back(fs); bv.push_back(bw); }
  }
  GK(cudaGetLastError());
  auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  printf("n %d (AF3 pairformer block%s): forward %.1f ms, replay %.1f ms, backward %.1f ms; arena high water %.0f MiB\n",
         n, BF16 ? ", bf16" : TF32 ? ", tf32" : "", med(fv), med(rv), med(bv), ARENA.high / 1048576.0);
}
