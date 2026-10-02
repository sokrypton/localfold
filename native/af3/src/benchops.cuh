// --bench-ops: each operation of one denoiser transformer block timed alone, 500 back-to-back
// launches with CUDA events, so the per-op cost includes the launch gap but no host sync.
#pragma once
#include "diffusion.cuh"
inline void benchOps(int n) {
  const std::string Tn = "diffusion.transformer";
  int C = (int)M.meta(Tn + ".channels"), Cc = (int)M.meta(Tn + ".condChannels");
  int heads = (int)M.meta(Tn + ".heads"), D = (int)M.meta(Tn + ".dimension"), Wd = heads * D, I = C * 2;
  std::string B = Tn + ".superBlocks.0.blocks.0";
  float* act = dalloc((size_t)n * C); CK(cudaMemset(act, 0, (size_t)n * C * 4));
  float* g = dalloc((size_t)n * 4 * C); CK(cudaMemset(g, 0, (size_t)n * 4 * C * 4));
  half* x = dallocT<half>((size_t)n * C); half* qkvg = dallocT<half>((size_t)(n + 128) * 4 * Wd);
  half* o = dallocT<half>((size_t)n * Wd); float* att = dalloc((size_t)n * C);
  half* wide = dallocT<half>((size_t)n * 2 * I); half* gated = dallocT<half>((size_t)n * I);
  half* bias = dallocT<half>((size_t)heads * n * ((n + 7) / 8 * 8));
  float* mask = dalloc(n); { std::vector<float> m(n, 1.f); CK(cudaMemcpy(mask, m.data(), n * 4, cudaMemcpyHostToDevice)); }
  CK(cudaMemset(x, 0, (size_t)n * C * 2)); CK(cudaMemset(qkvg, 0, (size_t)(n + 128) * 4 * Wd * 2)); CK(cudaMemset(bias, 0, (size_t)heads * n * ((n + 7) / 8 * 8) * 2));
  std::string qw = qkvgWeight(B, C, Wd, false);
  linear<half, half>(x, qkvg, n, C, 4 * Wd, qw);   // uploads
  linear<half, float>(o, att, n, Wd, C, B + ".Transition2");
  linear<half, half>(x, wide, n, C, 2 * I, B + ".ffwTransition1");
  linear<half, float>(gated, att, n, I, C, B + ".ffwTransition2");
  auto time = [&](const char* name, const std::function<void()>& f) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int k = 0; k < 20; ++k) f();
    cudaEventRecord(a, STREAM);
    for (int k = 0; k < 500; ++k) f();
    cudaEventRecord(b, STREAM); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("  %-22s %7.2f us\n", name, ms * 1000 / 500);
  };
  printf("one transformer block's operations at %d tokens:\n", n);
  float* yb = dalloc((size_t)n * C); CK(cudaMemset(yb, 0, (size_t)n * C * 4));
  time("gatedAddAdaLn block/row", [&] { gatedAddAdaLnK<half><<<n, 256, C * 4, STREAM>>>(act, yb, g, 4 * C, g, g + C, 4 * C, x, C, n); });
  time("gatedAddAdaLn vec/row", [&] { gatedAddAdaLnVecK<half, float><<<n, C / 4, 0, STREAM>>>(act, yb, g, 4 * C, g, g + C, 4 * C, x, C, n); });
  time("gatedAddAdaLn warp/row", [&] { gatedAddAdaLnWarpK<half><<<(n + 3) / 4, 128, 0, STREAM>>>(act, yb, g, 4 * C, g, g + C, 4 * C, x, n, C); });
  time("adaLN (strided)", [&] { adaLnStridedTK<half><<<(unsigned)((n + 7) / 8), 256, 0, STREAM>>>(act, g, g + C, 4 * C, x, n, C); });
  time("qkvg GEMM 768->3072", [&] { linear<half, half>(x, qkvg, n, C, 4 * Wd, qw); });
  time("q bias", [&] { addQBiasTK<half><<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(qkvg, W(B + ".qBias"), n, Wd); });
  for (int wv : {0, 1, 2, 4, 8}) {
    FLASH_WARPS_OVERRIDE = wv;
    std::string label = "flash attention w" + std::to_string(wv);
    time(label.c_str(), [&] { flashGrid<half>(qkvg, bias, (n + 7) / 8 * 8, mask, o, n, heads, D, 0, 1, false, 0.1f); });
  }
  FLASH_WARPS_OVERRIDE = 0;
  time("flash split 2", [&] { flashSplitHalfAt<48, 2>(qkvg, bias, (n + 7) / 8 * 8, mask, o, n, heads, 0, 1, false, 0.1f, nullptr); });
  time("flash split 4", [&] { flashSplitHalfAt<48, 4>(qkvg, bias, (n + 7) / 8 * 8, mask, o, n, heads, 0, 1, false, 0.1f, nullptr); });
  time("flash split 8", [&] { flashSplitHalfAt<48, 8>(qkvg, bias, (n + 7) / 8 * 8, mask, o, n, heads, 0, 1, false, 0.1f, nullptr); });
  time("T2 GEMM 768->768", [&] { linear<half, float>(o, att, n, Wd, C, B + ".Transition2"); });
  time("add gated", [&] { addGatedStridedK<float><<<blocks((size_t)n * C), 256, 0, STREAM>>>(act, att, g, 2 * C, n, C, n); });
  time("ffw1 GEMM 768->3072", [&] { linear<half, half>(x, wide, n, C, 2 * I, B + ".ffwTransition1"); });
  time("swiglu", [&] { swiglu<half>(wide, gated, n, I); });
  time("ffw2 GEMM 1536->768", [&] { linear<half, float>(gated, att, n, I, C, B + ".ffwTransition2"); });
  time("empty kernel", [&] { addK<<<1, 32, 0, STREAM>>>(att, att, 0); });
}
// --bench-grid=N: the pair track's grid attention alone at N tokens (4 heads of 32, every row,
// no mask), the arms interleaved and each the median of several rounds of 10 launches
inline void benchGrid(int n) {
  const int heads = 4, D = 32, Wd = heads * D, stride = (n + 7) / 8 * 8;
  size_t rows = n;
  half* qkvg = dallocT<half>(rows * n * 4 * Wd); half* out = dallocT<half>(rows * n * Wd);
  half* bias = dallocT<half>((size_t)heads * n * stride);
  { std::vector<half> h(std::max(rows * n * 4 * Wd, (size_t)heads * n * stride));
    uint64_t s = 1; for (auto& v : h) { s = s * 6364136223846793005ull + 1442695040888963407ull; v = __float2half(((s >> 40) / 16777216.f - 0.5f)); }
    CK(cudaMemcpy(qkvg, h.data(), rows * n * 4 * Wd * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bias, h.data(), (size_t)heads * n * stride * 2, cudaMemcpyHostToDevice)); }
  std::vector<std::pair<std::string, std::function<void()>>> arms = {
    {"grid w4", [&] { flashGridHalfAt<32, 4>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w8", [&] { flashGridHalfAt<32, 8>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
  };
  std::vector<std::vector<float>> t(arms.size());
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int round = 0; round < 7; ++round)
    for (size_t k = 0; k < arms.size(); ++k) {
      arms[k].second();
      cudaEventRecord(a, STREAM);
      for (int i = 0; i < 10; ++i) arms[k].second();
      cudaEventRecord(b, STREAM); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); t[k].push_back(ms / 10);
    }
  double flops = (double)n * n * n * heads * D * 4;
  for (size_t k = 0; k < arms.size(); ++k) {
    std::sort(t[k].begin(), t[k].end());
    printf("  %-16s %7.3f ms  %5.1f TFLOP/s\n", arms[k].first.c_str(), t[k][3], flops / t[k][3] / 1e9);
  }
}
