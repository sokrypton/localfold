// The AF2 training block's cost at a training shape: forward with saves, backward, and what the arena holds -
// against a stack's input saves. Random weights and inputs (no oracle: cuda/grad/test_af2_block.cu is the gate).
//   cuda/grad/bench-af2-block [--L=256] [--S=128] [--extra] [--tf32] [--bf16] [--blocks=48] [--graph]
//   ... --stack [--keep-mb=N] [--carry] [--no-graphs] [--no-graded] [--bf16-res]   (a whole stack trained, keep tiers off and on)
#include <cstring>
#include <curand.h>
#include <cuda_profiler_api.h>
#include "af2.cuh"
using namespace grad;
int main(int argc, char** argv) {
  bool stack = false, carry = false, noGraphs = false, noGraded = false, bfRes = false;
  size_t keepMb = SIZE_MAX;
  int L = 256, S = 128, nblocks = 48; bool extra = false, graph = false;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--L=", 4)) L = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--S=", 4)) S = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--blocks=", 9)) nblocks = atoi(argv[i] + 9);
    else if (!strcmp(argv[i], "--extra")) extra = true;
    else if (!strcmp(argv[i], "--tf32")) TF32 = true;
    else if (!strcmp(argv[i], "--bf16")) BF16 = true;
    else if (!strcmp(argv[i], "--graph")) graph = true;   // the backward captured once as a CUDA graph and replayed
    else if (!strcmp(argv[i], "--stack")) stack = true;   // a whole --blocks stack trained, with keep tiers and without
    else if (!strncmp(argv[i], "--keep-mb=", 10)) keepMb = (size_t)atol(argv[i] + 10) << 20;   // (--stack: the tiers' budget)
    else if (!strcmp(argv[i], "--carry")) carry = true;          // (--stack: the bf16 carry between blocks)
    else if (!strcmp(argv[i], "--no-graphs")) noGraphs = true;
    else if (!strcmp(argv[i], "--no-graded")) noGraded = true;
    else if (!strcmp(argv[i], "--bf16-res")) bfRes = true;   // (the blocks' residual states in bf16: Block::bf16Res)   // (--stack: keep whole blocks only, no graded keep)   // (--stack --graph: the stream, not the graphs)
  }
  if (graph) GK(cudaStreamCreateWithFlags(&STREAM, cudaStreamNonBlocking));   // (the legacy stream cannot be captured)
  init();
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 7);
  auto randn = [&](size_t n, float sd) { float* p; GK(cudaMalloc(&p, ((n + 1) & ~1) * 4)); curandGenerateNormal(gen, p, (n + 1) & ~1, 0.f, sd); return p; };
  if (stack) {   // a stack of --blocks blocks: forward, then backward, keep tiers on (the default) and off
    af2::Stack st; st.blocks.resize(nblocks);
    for (auto& b : st.blocks) {
      b.bf16Res = bfRes;
      b.extra = extra; b.Cm = extra ? 64 : 256;
      af2::forEachParam(b, [&](const std::string& n, Param& p, size_t len) {
        p.name = n; p.n = len; p.w = randn(len, 0.05f); GK(cudaMalloc(&p.g, len * 4)); b.params.add(p); });
      for (auto* p : b.params.all) if (p->name.find("/scale") != std::string::npos) { std::vector<float> one(p->n, 1.f); GK(cudaMemcpy(p->w, one.data(), p->n * 4, cudaMemcpyHostToDevice)); }
    }
    const int Cm = st.blocks[0].Cm;
    size_t mr = (size_t)S * L * Cm, pr = (size_t)L * L * 128;
    float *msa = randn(mr, 1.f), *pair = randn(pr, 1.f), *ctm = randn(mr, 1.f), *ctp = randn(pr, 1.f);
    std::vector<float> ones(std::max((size_t)S * L, (size_t)L * L), 1.f);
    float *mm_, *pm, *dm, *dp; GK(cudaMalloc(&mm_, (size_t)S * L * 4)); GK(cudaMalloc(&pm, (size_t)L * L * 4));
    GK(cudaMalloc(&dm, mr * 4)); GK(cudaMalloc(&dp, pr * 4));
    GK(cudaMemcpy(mm_, ones.data(), (size_t)S * L * 4, cudaMemcpyHostToDevice)); GK(cudaMemcpy(pm, ones.data(), (size_t)L * L * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1, e2; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    for (int arm = 0; arm < 4; ++arm) {   // off, on, off, on
      st.keepBytes = arm % 2 ? keepMb : 0;
      st.graphs = !noGraphs;   // (on a --graph stream: each block's replay and backward as one CUDA graph)
      st.bf16Carry = carry;
      for (auto& b : st.blocks) b.params.zeroGrad();
      ARENA.reset();
      GK(cudaMemcpyAsync(dm, ctm, mr * 4, cudaMemcpyDeviceToDevice, STREAM));   // (on STREAM: a legacy-stream
      GK(cudaMemcpyAsync(dp, ctp, pr * 4, cudaMemcpyDeviceToDevice, STREAM));   // copy does not order against it)
      st.gradedKeep = !noGraded;
      cudaEventRecord(e0, STREAM);
      st.forward(msa, pair, mm_, pm, S, L);
      cudaEventRecord(e1, STREAM);
      const size_t k = st.keptBlocks(), g = st.graded;
      st.backward(mm_, pm, dm, dp);
      cudaEventRecord(e2, STREAM);
      GK(cudaStreamSynchronize(STREAM)); GK(cudaGetLastError());
      float f, bw; cudaEventElapsedTime(&f, e0, e1); cudaEventElapsedTime(&bw, e1, e2);
      size_t fr, tot; cudaMemGetInfo(&fr, &tot);
      printf("stack of %d (L %d S %d%s), keep tiers %s: forward %.1f ms, backward %.1f ms -> %.1f ms; %zu blocks kept, %zu "
             "graded (%.0f MiB), %.1f GiB free after\n", nblocks, L, S, BF16 ? ", bf16" : "", arm % 2 ? "on " : "off", f, bw,
             f + bw, k, g, st.keep.held() / 1048576.0, fr / 1073741824.0);
    }
    return 0;
  }
  af2::Block b; b.extra = extra; b.Cm = extra ? 64 : 256; b.bf16Res = bfRes;
  af2::forEachParam(b, [&](const std::string& n, Param& p, size_t len) {
    p.name = n; p.n = len; p.w = randn(len, 0.05f); GK(cudaMalloc(&p.g, len * 4)); b.params.add(p); });
  for (auto* p : b.params.all) if (p->name.find("/scale") != std::string::npos) { std::vector<float> one(p->n, 1.f); GK(cudaMemcpy(p->w, one.data(), p->n * 4, cudaMemcpyHostToDevice)); }
  size_t mr = (size_t)S * L * b.Cm, pr = (size_t)L * L * 128;
  float *msa = randn(mr, 1.f), *pair = randn(pr, 1.f), *ctm = randn(mr, 1.f), *ctp = randn(pr, 1.f);
  std::vector<float> ones(std::max((size_t)S * L, (size_t)L * L), 1.f);
  float *mm_, *pm; GK(cudaMalloc(&mm_, (size_t)S * L * 4)); GK(cudaMalloc(&pm, (size_t)L * L * 4));
  GK(cudaMemcpy(mm_, ones.data(), (size_t)S * L * 4, cudaMemcpyHostToDevice)); GK(cudaMemcpy(pm, ones.data(), (size_t)L * L * 4, cudaMemcpyHostToDevice));
  float *mo, *po, *dm, *dp; GK(cudaMalloc(&mo, mr * 4)); GK(cudaMalloc(&po, pr * 4)); GK(cudaMalloc(&dm, mr * 4)); GK(cudaMalloc(&dp, pr * 4));
  cudaEvent_t e0, e1, e2, e3, e4; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3); cudaEventCreate(&e4);
  float f = 0, fs = 0, bw = 0, bwF = 0;
  // BF16: the backward as a stack runs it, on bf16 running gradients from block to block (blockBackwardRun);
  // bwF is blockBackward's own, converting a float cotangent in and the gradients out
  bf *bm = nullptr, *bp = nullptr, *cbm = nullptr, *cbp = nullptr;
  if (BF16) {
    GK(cudaMalloc(&bm, mr * 2)); GK(cudaMalloc(&bp, pr * 2)); GK(cudaMalloc(&cbm, mr * 2)); GK(cudaMalloc(&cbp, pr * 2));
    toBf(ctm, cbm, mr); toBf(ctp, cbp, pr);
  }
  for (int r = 0; r < 4; ++r) {
    b.params.zeroGrad(); ARENA.reset(); ARENA.high = 0;
    cudaEventRecord(e0, STREAM);
    { af2::BlockSave s = af2::blockForward(b, msa, pair, mm_, pm, mo, po, S, L); (void)s; }  // the stack's plain forward
    cudaEventRecord(e1, STREAM);
    ARENA.reset();
    af2::BlockSave s = af2::blockForward(b, msa, pair, mm_, pm, mo, po, S, L);              // the replay, with saves
    cudaEventRecord(e2, STREAM);
    if (BF16) {
      GK(cudaMemcpyAsync(bm, cbm, mr * 2, cudaMemcpyDeviceToDevice, STREAM)); GK(cudaMemcpyAsync(bp, cbp, pr * 2, cudaMemcpyDeviceToDevice, STREAM));
      cudaEventRecord(e4, STREAM);
    }
    if (r == 3) cudaProfilerStart();                                                        // (nsys --capture-range=cudaProfilerApi)
    if (BF16) af2::blockBackwardRun<bf>(b, s, mm_, pm, bm, bp);
    else af2::blockBackward(b, s, mm_, pm, ctm, ctp, dm, dp);
    cudaEventRecord(e3, STREAM);
    if (r == 3) { GK(cudaStreamSynchronize(STREAM)); cudaProfilerStop(); }
    GK(cudaStreamSynchronize(STREAM));
    cudaEventElapsedTime(&f, e0, e1); cudaEventElapsedTime(&fs, e1, e2); cudaEventElapsedTime(&bw, BF16 ? e4 : e2, e3);
    if (BF16) {
      b.params.zeroGrad(); ARENA.reset();
      af2::BlockSave s2 = af2::blockForward(b, msa, pair, mm_, pm, mo, po, S, L);
      cudaEventRecord(e2, STREAM);
      af2::blockBackward(b, s2, mm_, pm, ctm, ctp, dm, dp);
      cudaEventRecord(e3, STREAM);
      GK(cudaStreamSynchronize(STREAM));
      cudaEventElapsedTime(&bwF, e2, e3);
    }
  }
  GK(cudaGetLastError());
  if (graph) {   // the arena is where the last replay left it, so the captured addresses hold for every replay
    b.params.zeroGrad(); ARENA.reset();
    af2::BlockSave s = af2::blockForward(b, msa, pair, mm_, pm, mo, po, S, L);
    cudaGraph_t gr; cudaGraphExec_t ge;
    GK(cudaStreamBeginCapture(STREAM, cudaStreamCaptureModeThreadLocal));
    af2::blockBackward(b, s, mm_, pm, ctm, ctp, dm, dp);
    GK(cudaStreamEndCapture(STREAM, &gr));
    GK(cudaGraphInstantiate(&ge, gr, 0));
    size_t nodes = 0; GK(cudaGraphGetNodes(gr, nullptr, &nodes));
    float tg = 1e9, td = 1e9;
    for (int r = 0; r < 6; ++r) {
      float x;
      cudaEventRecord(e0, STREAM); GK(cudaGraphLaunch(ge, STREAM)); cudaEventRecord(e1, STREAM);
      GK(cudaStreamSynchronize(STREAM)); cudaEventElapsedTime(&x, e0, e1); if (r) tg = std::min(tg, x);
      cudaEventRecord(e0, STREAM); af2::blockBackward(b, s, mm_, pm, ctm, ctp, dm, dp); cudaEventRecord(e1, STREAM);
      GK(cudaStreamSynchronize(STREAM)); cudaEventElapsedTime(&x, e0, e1); if (r) td = std::min(td, x);
    }
    GK(cudaGetLastError());
    printf("  backward as a CUDA graph (%zu nodes): %.2f ms, launched directly %.2f ms (min of 5, interleaved)\n", nodes, tg, td);
  }
  double inBytes = (double)(mr + pr) * 4;
  printf("L %d S %d Cm %d (%s%s): forward %.2f ms, replay %.2f ms, backward %.2f ms -> %.1f ms a block trained (%.2f s for %d blocks)\n",
         L, S, b.Cm, extra ? "extra" : "main", BF16 ? ", bf16" : TF32 ? ", tf32" : "", f, fs, bw, f + fs + bw, (f + fs + bw) * nblocks / 1000, nblocks);
  if (BF16) printf("  (blockBackward with a float cotangent in and float gradients out: %.1f ms)\n", bwF);
  printf("  one block's activations (arena high water) %.0f MiB; the stack's saved inputs %.0f MiB (%d blocks)\n",
         ARENA.high / 1048576.0, inBytes * (nblocks + 1) / 1048576.0, nblocks);
}
