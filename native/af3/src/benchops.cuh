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
  {   // flashGrid2R (f16 scores, two tiles a warp) against the f32-score kernel, unmasked: how far its output moves
    flashGridHalfRun<32, 4, 48, false>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr);
    flashGrid2RRun<32, 4, 48, 2>(qkvg, bias, stride, out2, n, heads, rows, 0.17f, nullptr);
    std::vector<half> a(rows * n * Wd), b2(rows * n * Wd);
    CK(cudaMemcpy(a.data(), out, a.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b2.data(), out2, b2.size() * 2, cudaMemcpyDeviceToHost));
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      double x = __half2float(a[i]), y = __half2float(b2[i]);
      num += (x - y) * (x - y); den += x * x; mx = std::max(mx, std::fabs(x - y));
    }
    printf("  2R against the f32-score kernel: relRMS %.3e, max |d| %.3e\n", std::sqrt(num / std::max(den, 1e-30)), mx);
  }
  {   // the no-bias form (AF2's column attention) against the biased one fed zeros: the same bytes
    half* zb = dallocT<half>((size_t)heads * n * stride); CK(cudaMemset(zb, 0, (size_t)heads * n * stride * 2));
    flashGrid2RRun<32, 2, 48, 2, 2>(qkvg, zb, stride, out, n, heads, rows, 0.17f, nullptr);
    flashGrid2RRun<32, 2, 48, 2, 2, true>(qkvg, nullptr, stride, out2, n, heads, rows, 0.17f, nullptr);
    std::vector<half> a(rows * n * Wd), b2(rows * n * Wd);
    CK(cudaMemcpy(a.data(), out, a.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b2.data(), out2, b2.size() * 2, cudaMemcpyDeviceToHost));
    size_t differ = 0; for (size_t i = 0; i < a.size(); ++i) differ += memcmp(&a[i], &b2[i], 2) != 0;
    printf("  2R no-bias against zero bias: %zu of %zu outputs differ\n", differ, a.size());
    CK(cudaFree(zb));
  }
  for (int rr : {2, 3}) {   // RR rows a block share one bias tile: the same arithmetic, so the same bytes
    flashGrid2RRun<32, 4, 48, 2>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr);
    if (rr == 2) flashGrid2RRun<32, 4, 48, 2, 2>(qkvg, bias, stride, out2, n, heads, rows, 0.17f, nullptr);
    else flashGrid2RRun<32, 4, 48, 2, 3>(qkvg, bias, stride, out2, n, heads, rows, 0.17f, nullptr);
    std::vector<half> a(rows * n * Wd), b2(rows * n * Wd);
    CK(cudaMemcpy(a.data(), out, a.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b2.data(), out2, b2.size() * 2, cudaMemcpyDeviceToHost));
    size_t differ = 0; for (size_t i = 0; i < a.size(); ++i) differ += memcmp(&a[i], &b2[i], 2) != 0;
    printf("  2R rr%d against rr1: %zu of %zu outputs differ\n", rr, differ, a.size());
  }
  std::vector<std::pair<std::string, std::function<void()>>> arms = {
    {"grid w4 cp.async", [&] { flashGridHalfRun<32, 4, FA_BK, false>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4 reg", [&] { flashGridHalfRun<32, 4, FA_BK, true>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4 reg masked", [&] { flashGridHalfRun<32, 4, FA_BK, true>(qkvg, bias, stride, mask, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4", [&] { flashGridHalfAt<32, 4>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"grid w4 bk48", [&] { flashGridHalfAt<32, 4, 48>(qkvg, bias, stride, nullptr, out, n, heads, 0, rows, false, 0.17f, nullptr); }},
    {"2R w4 bk48 mt2", [&] { flashGrid2RRun<32, 4, 48, 2>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr); }},
    {"2R w4 bk48 rr2", [&] { flashGrid2RRun<32, 4, 48, 2, 2>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr); }},
    {"2R w4 bk48 rr3", [&] { flashGrid2RRun<32, 4, 48, 2, 3>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr); }},
    {"2R w4 bk32 rr2", [&] { flashGrid2RRun<32, 4, 32, 2, 2>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr); }},
    {"2R w4 bk64 rr2", [&] { flashGrid2RRun<32, 4, 64, 2, 2>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr); }},
    {"2R w2 bk48 rr2", [&] { flashGrid2RRun<32, 2, 48, 2, 2>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr); }},
    {"2R w2 bk48 rr3", [&] { flashGrid2RRun<32, 2, 48, 2, 3>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr); }},
    {"2R w8 bk48 mt1", [&] { flashGrid2RRun<32, 8, 48, 1>(qkvg, bias, stride, out, n, heads, rows, 0.17f, nullptr); }},
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
// --bench-trans=N: the pair transition (C 128, I 512) alone over N*N rows, each form against the
// shipped one on the same input (relRMS of the update), then the forms timed interleaved
inline void benchTrans(int n) {
  const int C = 128, I = 512;
  size_t rows = (size_t)n * n;
  float* x0 = dalloc(rows * C); float* x = dalloc(rows * C); float* ref = dalloc(rows * C);
  half* W1 = dallocT<half>((size_t)C * 2 * I); half* W2 = dallocT<half>((size_t)I * C);
  float* sc = dalloc(C); float* of = dalloc(C);
  uint64_t s = 1; auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  { std::vector<float> h(rows * C); for (auto& v : h) v = rnd(); CK(cudaMemcpy(x0, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
  { std::vector<half> h((size_t)C * 2 * I); for (auto& v : h) v = __float2half(rnd() * 0.2f); CK(cudaMemcpy(W1, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    std::vector<half> h2((size_t)I * C); for (auto& v : h2) v = __float2half(rnd() * 0.2f); CK(cudaMemcpy(W2, h2.data(), h2.size() * 2, cudaMemcpyHostToDevice)); }
  { std::vector<float> h(C); for (auto& v : h) v = 1.f + rnd() * 0.2f; CK(cudaMemcpy(sc, h.data(), C * 4, cudaMemcpyHostToDevice));
    for (auto& v : h) v = rnd() * 0.2f; CK(cudaMemcpy(of, h.data(), C * 4, cudaMemcpyHostToDevice)); }
  std::vector<std::pair<std::string, std::function<void()>>> arms = {
    {"w8 mt1 nc32", [&] { fusedTransitionAt<8, false, 1, 32>(x, rows, I, sc, of, W1, W2, nullptr, nullptr); }},
    {"w4 mt1 nc32", [&] { fusedTransitionAt<4, false, 1, 32>(x, rows, I, sc, of, W1, W2, nullptr, nullptr); }},
    {"w4 mt2 nc16", [&] { fusedTransitionAt<4, false, 2, 16>(x, rows, I, sc, of, W1, W2, nullptr, nullptr); }},
  };
  auto delta = [&](float* out) {   // the update, out - x0, on the host
    std::vector<float> a(rows * C), b(rows * C);
    CK(cudaMemcpy(a.data(), out, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), x0, b.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < a.size(); ++i) a[i] -= b[i];
    return a;
  };
  CK(cudaMemcpy(ref, x0, rows * C * 4, cudaMemcpyDeviceToDevice));
  std::swap(x, ref); arms[0].second(); std::swap(x, ref); CK(cudaStreamSynchronize(STREAM));
  auto dr = delta(ref);
  for (size_t k = 1; k < arms.size(); ++k) {
    CK(cudaMemcpy(x, x0, rows * C * 4, cudaMemcpyDeviceToDevice)); arms[k].second(); CK(cudaStreamSynchronize(STREAM));
    auto d = delta(x); double num = 0, den = 0;
    for (size_t i = 0; i < d.size(); ++i) { num += (d[i] - dr[i]) * (double)(d[i] - dr[i]); den += (double)dr[i] * dr[i]; }
    printf("  %-14s against %s: relRMS %.3e\n", arms[k].first.c_str(), arms[0].first.c_str(), std::sqrt(num / den));
  }
  std::vector<std::vector<float>> t(arms.size());
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int round = 0; round < 7; ++round)
    for (size_t k = 0; k < arms.size(); ++k) {
      CK(cudaMemcpy(x, x0, rows * C * 4, cudaMemcpyDeviceToDevice));
      arms[k].second();
      cudaEventRecord(a, STREAM);
      for (int i = 0; i < 10; ++i) arms[k].second();
      cudaEventRecord(b, STREAM); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); t[k].push_back(ms / 10);
    }
  double flops = (double)rows * (C * 2 * I + I * C) * 2;
  for (size_t k = 0; k < arms.size(); ++k) {
    std::sort(t[k].begin(), t[k].end());
    printf("  %-14s %7.3f ms  %5.1f TFLOP/s\n", arms[k].first.c_str(), t[k][3], flops / t[k][3] / 1e9);
  }
}
// --bench-tri=N: the fused triangle's output kernel alone at N tokens (bf16 product, C 128), on random
// inputs: a checksum of the updated pair (to compare two builds bit for bit) and the median of 7 x 10 calls
inline void benchTri(int n) {
  const int C = 128, np = (n + 7) / 8 * 8;
  size_t pp = (size_t)np * np, cs = pp, rows = (size_t)n * n;
  uint64_t s = 1; auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  auto fill = [&](auto* d, size_t count, float scale, float base) {
    using T = std::remove_pointer_t<decltype(d)>;
    std::vector<T> h(count); for (auto& v : h) v = (T)(base + rnd() * scale);
    CK(cudaMemcpy(d, h.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  };
  __nv_bfloat16* prod = dallocT<__nv_bfloat16>((size_t)C * cs); half* t2 = dallocT<half>(pp * C); half* Wo = dallocT<half>((size_t)C * C);
  float* p0 = dalloc(rows * C); float* pair = dalloc(rows * C); float* sc = dalloc(C); float* of = dalloc(C);
  fill(prod, (size_t)C * cs, 4.f, 0.f); fill(t2, pp * C, 4.f, 0.f); fill(Wo, (size_t)C * C, 0.2f, 0.f);
  fill(p0, rows * C, 2.f, 0.f); fill(sc, C, 0.2f, 1.f); fill(of, C, 0.2f, 0.f);
  CK(cudaMemcpy(pair, p0, rows * C * 4, cudaMemcpyDeviceToDevice));
  triOutRaw<__nv_bfloat16>(prod, sc, of, Wo, nullptr, t2, pair, n, np, cs);
  std::vector<uint32_t> h(rows * C); CK(cudaMemcpy(h.data(), pair, h.size() * 4, cudaMemcpyDeviceToHost));
  uint64_t sum = 1469598103934665603ull; for (uint32_t v : h) sum = (sum ^ v) * 1099511628211ull;
  std::vector<float> t;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int round = 0; round < 7; ++round) {
    cudaEventRecord(a, STREAM);
    for (int i = 0; i < 10; ++i) triOutRaw<__nv_bfloat16>(prod, sc, of, Wo, nullptr, t2, pair, n, np, cs);
    cudaEventRecord(b, STREAM); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms / 10);
  }
  std::sort(t.begin(), t.end());
  printf("triOut %d tokens: checksum %016llx  %.4f ms\n", n, (unsigned long long)sum, t[3]);
  // the input kernel: LN(pair) -> a, b (channel-major, bf16) and t2
  half* Wpg = dallocT<half>((size_t)C * 4 * C); half* Wg = dallocT<half>((size_t)C * C); float* mask = dalloc(rows);
  fill(Wpg, (size_t)C * 4 * C, 0.2f, 0.f); fill(Wg, (size_t)C * C, 0.2f, 0.f); fill(mask, rows, 0.f, 1.f);
  __nv_bfloat16* a2 = dallocT<__nv_bfloat16>((size_t)C * cs); __nv_bfloat16* b2 = dallocT<__nv_bfloat16>((size_t)C * cs);
  auto runIn = [&] { triInRaw<__nv_bfloat16>(p0, mask, sc, of, Wpg, Wg, nullptr, a2, b2, t2, n, np, cs); };
  runIn(); CK(cudaStreamSynchronize(STREAM));
  sum = 1469598103934665603ull;
  std::vector<std::pair<const void*, size_t>> outs = {{a2, (size_t)C * cs * 2}, {b2, (size_t)C * cs * 2}, {t2, pp * C * 2}};
  for (auto [ptr, bytes] : outs) {
    std::vector<uint16_t> hh(bytes / 2); CK(cudaMemcpy(hh.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    for (uint16_t v : hh) sum = (sum ^ v) * 1099511628211ull;
  }
  t.clear();
  for (int round = 0; round < 7; ++round) {
    cudaEventRecord(a, STREAM);
    for (int i = 0; i < 10; ++i) runIn();
    cudaEventRecord(b, STREAM); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms / 10);
  }
  std::sort(t.begin(), t.end());
  printf("triIn  %d tokens: checksum %016llx  %.4f ms\n", n, (unsigned long long)sum, t[3]);
  // the 256-channel output kernel (fused256.cuh: protenix2, ESMFold2), its bf16 product in the padded rows
  {
    const int C2 = 256;
    __nv_bfloat16* pr2 = dallocT<__nv_bfloat16>((size_t)C2 * cs); half* t22 = dallocT<half>(pp * C2); half* Wo2 = dallocT<half>((size_t)C2 * C2);
    float* q0 = dalloc(rows * C2); float* q = dalloc(rows * C2); float* sc2 = dalloc(C2); float* of2 = dalloc(C2);
    fill(pr2, (size_t)C2 * cs, 4.f, 0.f); fill(t22, pp * C2, 4.f, 0.f); fill(Wo2, (size_t)C2 * C2, 0.2f, 0.f);
    fill(q0, rows * C2, 2.f, 0.f); fill(sc2, C2, 0.2f, 1.f); fill(of2, C2, 0.2f, 0.f);
    CK(cudaMemcpy(q, q0, rows * C2 * 4, cudaMemcpyDeviceToDevice));
    auto run = [&] { triangleOutRun<256, 4, __nv_bfloat16>(pr2, sc2, of2, Wo2, t22, q, n, np); };
    run();
    std::vector<uint32_t> hq(rows * C2); CK(cudaMemcpy(hq.data(), q, hq.size() * 4, cudaMemcpyDeviceToHost));
    sum = 1469598103934665603ull; for (uint32_t v : hq) sum = (sum ^ v) * 1099511628211ull;
    t.clear();
    for (int round = 0; round < 7; ++round) {
      cudaEventRecord(a, STREAM);
      for (int i = 0; i < 10; ++i) run();
      cudaEventRecord(b, STREAM); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms / 10);
    }
    std::sort(t.begin(), t.end());
    printf("triOut256 %d tokens: checksum %016llx  %.4f ms\n", n, (unsigned long long)sum, t[3]);
  }
}
