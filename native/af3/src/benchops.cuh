// --bench-ops: each operation of one denoiser transformer block timed alone, 500 back-to-back
// launches with CUDA events, so the per-op cost includes the launch gap but no host sync.
#pragma once
#include "diffusion.cuh"
inline void benchOps(int nTok, int S = 1) {
  // --bench-ops=N or NxS: S samples as the attention's rows, every GEMM over N*S rows
  int n = nTok * S;
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
  // inside a CUDA graph of 200 launches, as the denoiser step runs (a bare launch is ~2 us)
  auto time = [&](const char* name, const std::function<void()>& f) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int k = 0; k < 20; ++k) f();
    CK(cudaStreamSynchronize(STREAM));
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(STREAM, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < 200; ++k) f();
    CK(cudaStreamEndCapture(STREAM, &gr)); CK(cudaGraphInstantiate(&ge, gr, 0));
    CK(cudaGraphLaunch(ge, STREAM)); CK(cudaStreamSynchronize(STREAM));
    float best = 1e9f;
    for (int r = 0; r < 5; ++r) {
      cudaEventRecord(a, STREAM); CK(cudaGraphLaunch(ge, STREAM)); cudaEventRecord(b, STREAM); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms);
    }
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr));
    printf("  %-22s %7.2f us\n", name, best * 1000 / 200);
  };
  printf("one transformer block's operations at %d tokens x %d samples:\n", nTok, S);
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
    time(label.c_str(), [&] { flashGrid<half>(qkvg, bias, (nTok + 7) / 8 * 8, mask, o, nTok, heads, D, 0, S, false, 0.1f); });
  }
  FLASH_WARPS_OVERRIDE = 0;
  int st8 = (nTok + 7) / 8 * 8;
  time("flash w4 bk48 nomask", [&] { flashGridHalfAt<48, 4, 48>(qkvg, bias, st8, nullptr, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash w4 bk32 nomask", [&] { flashGridHalfAt<48, 4, 32>(qkvg, bias, st8, nullptr, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash w4 bk64 nomask", [&] { flashGridHalfAt<48, 4, 64>(qkvg, bias, st8, nullptr, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash w8 bk48 nomask", [&] { flashGridHalfAt<48, 8, 48>(qkvg, bias, st8, nullptr, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash w8 bk32 nomask", [&] { flashGridHalfAt<48, 8, 32>(qkvg, bias, st8, nullptr, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash split 2", [&] { flashSplitHalfAt<48, 2>(qkvg, bias, (nTok + 7) / 8 * 8, mask, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash split 4", [&] { flashSplitHalfAt<48, 4>(qkvg, bias, (nTok + 7) / 8 * 8, mask, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash split 8", [&] { flashSplitHalfAt<48, 8>(qkvg, bias, (nTok + 7) / 8 * 8, mask, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash split 4 nomask", [&] { flashSplitHalfAt<48, 4>(qkvg, bias, (nTok + 7) / 8 * 8, nullptr, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
  time("flash split 8 nomask", [&] { flashSplitHalfAt<48, 8>(qkvg, bias, (nTok + 7) / 8 * 8, nullptr, o, nTok, heads, 0, S, false, 0.1f, nullptr); });
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
  // the two forms of the kernel (cp.async double-buffered; REG, register-staged - a T4's) against each
  // other, unmasked and with ~10% of keys masked: their outputs must be identical
  float* mask = dalloc((size_t)n * n);
  { std::vector<float> hm((size_t)n * n); uint64_t s = 7;
    for (auto& v : hm) { s = s * 6364136223846793005ull + 1442695040888963407ull; v = (s >> 40) % 10 ? 1.f : 0.f; }
    CK(cudaMemcpy(mask, hm.data(), hm.size() * 4, cudaMemcpyHostToDevice)); }
  half* out2 = dallocT<half>(rows * n * Wd);
  for (const float* mk : {(const float*)nullptr, (const float*)mask}) {
    flashGridHalfRun<32, 4, FA_BK, false>(qkvg, bias, stride, mk, out, n, heads, 0, rows, false, 0.17f, nullptr);
    flashGridHalfRun<32, 4, FA_BK, true>(qkvg, bias, stride, mk, out2, n, heads, 0, rows, false, 0.17f, nullptr);
    std::vector<half> a(rows * n * Wd), b2(rows * n * Wd);
    CK(cudaMemcpy(a.data(), out, a.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b2.data(), out2, b2.size() * 2, cudaMemcpyDeviceToHost));
    size_t differ = 0; for (size_t i = 0; i < a.size(); ++i) differ += memcmp(&a[i], &b2[i], 2) != 0;
    printf("  cp.async against reg%s: %zu of %zu outputs differ\n", mk ? ", masked" : "", differ, a.size());
  }
  std::vector<std::pair<std::string, std::function<void()>>> arms = {
    {"grid w4 cp.async", [&] { flashGridHalfRun<32, 4, FA_BK, false>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4 reg", [&] { flashGridHalfRun<32, 4, FA_BK, true>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4 reg masked", [&] { flashGridHalfRun<32, 4, FA_BK, true>(qkvg, bias, stride, mask, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4", [&] { flashGridHalfAt<32, 4>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w8", [&] { flashGridHalfAt<32, 8>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4 bk32", [&] { flashGridHalfAt<32, 4, 32>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4 bk48", [&] { flashGridHalfAt<32, 4, 48>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4 bk96", [&] { flashGridHalfAt<32, 4, 96>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
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
// --bench-pair=N: the pair track's six hot pair kernels at N tokens with the pair stored in f32 and in bf16
// (branch bf16-pair) - what a bf16 pair is worth, kernel by kernel, on pairformer block 0's weights. Each the
// median of five rounds of ten launches (the pair is read-modify-written: its values drift, its cost does not).
inline void benchPair(int n) {
  using bf = __nv_bfloat16;
  const std::string B = "trunk.pairformerBlocks.0", tri = B + ".triangleMultiplicationOutgoing", ga = B + ".pairAttention1";
  const int C = 128, np = (n + 7) / 8 * 8;
  size_t pairs = (size_t)n * n, cs = (size_t)np * np;
  int heads = (int)M.meta(ga + ".heads"), D = (int)M.meta(ga + ".dimension"), Wd = heads * D;
  int I = (int)(lenW(B + ".pairTransition.transition1") / (2 * (size_t)C));
  float* pf = dalloc(pairs * C); bf* pb = dallocT<bf>(pairs * C);
  { std::vector<float> h(pairs * C); uint64_t s = 1;
    for (auto& v : h) { s = s * 6364136223846793005ull + 1442695040888963407ull; v = ((s >> 40) / 16777216.f - 0.5f); }
    CK(cudaMemcpy(pf, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    std::vector<bf> hb(h.size()); for (size_t i = 0; i < h.size(); ++i) hb[i] = __float2bfloat16(h[i]);
    CK(cudaMemcpy(pb, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice)); }
  float* mask = dalloc(pairs); { std::vector<float> m(pairs, 1.f); CK(cudaMemcpy(mask, m.data(), pairs * 4, cudaMemcpyHostToDevice)); }
  bf *a = dallocT<bf>(cs * C), *b = dallocT<bf>(cs * C), *prod = dallocT<bf>(cs * C);
  half* t2 = dallocT<half>(cs * C);
  CK(cudaMemset(prod, 0, cs * C * 2)); CK(cudaMemset(t2, 0, cs * C * 2));
  half* qo = dallocT<half>((pairs + 128) * 4 * Wd); half* gathered = dallocT<half>(pairs * Wd);
  CK(cudaMemset(gathered, 0, pairs * Wd * 2));
  float* h16 = dalloc(pairs * 16);
  std::string pg = projectionGate(tri, C), qkvg = qkvgWeight(ga, C, Wd, true);
  auto time = [](const std::function<void()>& f) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int k = 0; k < 3; ++k) f();
    std::vector<float> r;
    for (int round = 0; round < 5; ++round) {
      cudaEventRecord(e0, STREAM); for (int k = 0; k < 10; ++k) f(); cudaEventRecord(e1, STREAM); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); r.push_back(ms / 10);
    }
    std::sort(r.begin(), r.end()); return r[2];
  };
  struct Arm { const char* name; std::function<void()> f32, b16; };
  std::vector<Arm> arms = {
    {"triangle in (LN + projections)", [&] { triIn128<bf>(pf, mask, tri, pg, a, b, t2, n, np, cs); },
                                       [&] { triIn128<bf, bf>(pb, mask, tri, pg, a, b, t2, n, np, cs); }},
    {"triangle out (pair += ...)", [&] { triOut128<bf>(prod, tri, t2, pf, n, np, cs); },
                                   [&] { triOut128<bf, bf>(prod, tri, t2, pb, n, np, cs); }},
    {"grid in (LN + q/k/v/g)", [&] { gridIn128(pf, ga, qkvg, qo, n, 0, pairs, false); },
                               [&] { gridIn128<bf>(pb, ga, qkvg, qo, n, 0, pairs, false); }},
    {"grid out, column (pair += ...)", [&] { gridOut128(gathered, ga + ".outputProjection", pf, n, 0, pairs, true); },
                                       [&] { gridOut128<bf>(gathered, ga + ".outputProjection", pb, n, 0, pairs, true); }},
    {"pair transition (fused)", [&] { fusedTransition(pf, pairs, C, I, B + ".pairTransition"); },
                                [&] { fusedTransition<bf>(pb, pairs, C, I, B + ".pairTransition"); }},
    {"single track's pair bias", [&] { lnHeads128<16>(pf, B + ".singlePairLogitsNormScale", B + ".singlePairLogitsNormOffset",
                                                      B + ".singlePairLogitsProjection", h16, pairs); },
                                 [&] { lnHeads128<16, bf>(pb, B + ".singlePairLogitsNormScale", B + ".singlePairLogitsNormOffset",
                                                          B + ".singlePairLogitsProjection", h16, pairs); }},
  };
  printf("bench-pair at %d tokens (ms a launch, median of five rounds of ten)\n", n);
  double tf = 0, tb = 0;
  for (auto& arm : arms) {
    float x = time(arm.f32), y = time(arm.b16);
    tf += x; tb += y;
    printf("  %-34s f32 %7.3f  bf16 %7.3f  saves %6.3f (%4.1f%%)\n", arm.name, x, y, x - y, 100 * (x - y) / x);
  }
  printf("  %-34s f32 %7.3f  bf16 %7.3f  saves %6.3f\n", "all six once", tf, tb, tf - tb);
}
