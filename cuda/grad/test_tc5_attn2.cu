// The grid attention's forward on tcgen05 (flash_tc5.cuh's gridFwdTc2, the persistent form) against gridFwdBf, on the same inputs as the
// pairformer hands them (af3.cuh's gridForwardBf: q, k, v and the raw gate side by side in one [n n, 4 H D] bf16
// buffer, a pair bias [H, n, n] every row shares, a key mask a row) - every output: o, the gate's sigmoid written
// back over it, o g, and the log-sum-exp pair (gridFwdTc2 takes 768 tokens and up: the smaller cases hold the gate's fallback).
// B200 only (a binary with sm_100a code):
//
//   nvcc -std=c++17 -arch=sm_100a -O2 cuda/grad/test_tc5_attn2.cu -o test-tc5-attn2 -lcublas -lcublasLt -lcurand -ldl && ./test-tc5-attn2
#include "../af3/src/common.cuh"
#include "../af3/src/flash.cuh"         // (with its flash_tc5.cuh: tc5a, gridFwdTc2's helpers, as train_of3.cu has them)
#include "ops.cuh"
#include <curand.h>
using grad::Sdpa; using grad::Bias; namespace flash = grad::flash;
static uint16_t toBf(float x) { uint32_t u; memcpy(&u, &x, 4); return (uint16_t)(u >> 16); }   // (truncated: any bf16 input)
static float fromBf(uint16_t x) { uint32_t u = (uint32_t)x << 16; float f; memcpy(&f, &u, 4); return f; }

int main() {
  grad::init();
  if (!tc5a::ready()) { printf("this device or binary has no tcgen05 (sm_100a): SKIP\n"); return 0; }
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 5);
  struct Case { int n; float sd; bool holes, deadRow; };
  int fails = 0;
  for (Case c : {Case{64, 1.f, false, false}, Case{128, 1.f, true, false}, Case{192, 2.f, true, true}, Case{256, 1.f, true, false},
                 Case{384, 3.f, true, true}, Case{512, 1.f, true, false}, Case{768, 1.f, true, false}, Case{1024, 2.f, true, true}}) {
    const int n = c.n, H = 4, D = 32, HD = H * D, P = 4 * HD;
    const size_t nn = (size_t)n * n;
    // inputs: qkvg (bf16), the pair bias (float) and the key mask
    float* f; GK(cudaMalloc(&f, nn * P * 4)); curandGenerateNormal(gen, f, nn * P, 0.f, c.sd);
    std::vector<float> hf(nn * P); GK(cudaMemcpy(hf.data(), f, nn * P * 4, cudaMemcpyDeviceToHost));
    std::vector<uint16_t> hq(nn * P); for (size_t i = 0; i < hq.size(); ++i) hq[i] = toBf(hf[i]);
    float* nb; GK(cudaMalloc(&nb, (size_t)H * nn * 4)); curandGenerateNormal(gen, nb, (size_t)H * nn, 0.f, c.sd);
    std::vector<float> hm(nn, 1.f);
    for (int b = 0; b < n; ++b)
      for (int k = 0; k < n; ++k) {
        if (c.holes && (b * 7 + k * 13) % 11 == 0) hm[(size_t)b * n + k] = 0.f;
        if (c.deadRow && b == n / 2) hm[(size_t)b * n + k] = 0.f;
      }
    float* m; GK(cudaMalloc(&m, nn * 4)); GK(cudaMemcpy(m, hm.data(), nn * 4, cudaMemcpyHostToDevice));
    Sdpa a; a.B = n; a.H = H; a.Q = n; a.K = n; a.D = D; a.scale = 1 / sqrtf((float)D); a.keyMask = m;
    a.bias[0] = Bias{nb, 0, (long long)nn, n, nullptr};
    // the two forms, each on its own copy (the gate's sigmoid is written over the raw gate)
    struct Out { std::vector<uint16_t> o, og, g; std::vector<float> lse; };
    auto run = [&](bool tc) {
      flash::TC2 = tc;
      uint16_t *qkvg, *o, *og; float* lse;
      GK(cudaMalloc(&qkvg, nn * P * 2)); GK(cudaMalloc(&o, nn * HD * 2)); GK(cudaMalloc(&og, nn * HD * 2));
      GK(cudaMalloc(&lse, 2 * nn * H * 4));
      GK(cudaMemcpy(qkvg, hq.data(), nn * P * 2, cudaMemcpyHostToDevice));
      auto bp = [&](uint16_t* p) { return reinterpret_cast<__nv_bfloat16*>(p); };
      const bool ok = flash::gridForwardBf(a, bp(qkvg), P, bp(qkvg + HD), P, bp(qkvg + 2 * HD), P, bp(o), HD, lse,
                                           bp(qkvg + 3 * HD), P, bp(og), HD, false);
      GK(cudaDeviceSynchronize());
      if (!ok) { printf("gridForwardBf declined the shape\n"); exit(1); }
      Out r; r.o.resize(nn * HD); r.og.resize(nn * HD); r.g.resize(nn * P); r.lse.resize(2 * nn * H);
      GK(cudaMemcpy(r.o.data(), o, nn * HD * 2, cudaMemcpyDeviceToHost)); GK(cudaMemcpy(r.og.data(), og, nn * HD * 2, cudaMemcpyDeviceToHost));
      GK(cudaMemcpy(r.g.data(), qkvg, nn * P * 2, cudaMemcpyDeviceToHost)); GK(cudaMemcpy(r.lse.data(), lse, 2 * nn * H * 4, cudaMemcpyDeviceToHost));
      GK(cudaFree(qkvg)); GK(cudaFree(o)); GK(cudaFree(og)); GK(cudaFree(lse));
      return r;
    };
    const Out A = run(false), B = run(true);
    {   // the two forms timed, interleaved (medians of 9)
      uint16_t *qkvg, *o, *og; float* lse;
      GK(cudaMalloc(&qkvg, nn * P * 2)); GK(cudaMalloc(&o, nn * HD * 2)); GK(cudaMalloc(&og, nn * HD * 2)); GK(cudaMalloc(&lse, 2 * nn * H * 4));
      GK(cudaMemcpy(qkvg, hq.data(), nn * P * 2, cudaMemcpyHostToDevice));
      auto bp = [&](uint16_t* p) { return reinterpret_cast<__nv_bfloat16*>(p); };
      cudaEvent_t e0, e1; GK(cudaEventCreate(&e0)); GK(cudaEventCreate(&e1));
      std::vector<float> t[2];
      for (int rep = 0; rep < 9; ++rep)
        for (int arm = 0; arm < 2; ++arm) {
          flash::TC2 = arm;
          GK(cudaEventRecord(e0, grad::STREAM));
          flash::gridForwardBf(a, bp(qkvg), P, bp(qkvg + HD), P, bp(qkvg + 2 * HD), P, bp(o), HD, lse, nullptr, 0, nullptr, 0, false);
          GK(cudaEventRecord(e1, grad::STREAM)); GK(cudaEventSynchronize(e1));
          float ms; GK(cudaEventElapsedTime(&ms, e0, e1)); t[arm].push_back(ms);
        }
      for (auto& v : t) std::sort(v.begin(), v.end());
      printf("  n %d: gridFwdBf %.3f ms, gridFwdTc2 %.3f ms (%.2fx)\n", n, t[0][4], t[1][4], t[0][4] / t[1][4]);
      GK(cudaFree(qkvg)); GK(cudaFree(o)); GK(cudaFree(og)); GK(cudaFree(lse));
    }
    auto bfv = fromBf;
    auto worst = [&](const std::vector<uint16_t>& x, const std::vector<uint16_t>& y, size_t stride, size_t off, size_t w) {
      double e = 0, mx = 0;
      for (size_t r = 0; r < x.size() / stride; ++r)
        for (size_t j = 0; j < w; ++j) {
          const size_t i = r * stride + off + j;
          e = std::max(e, (double)std::fabs(bfv(x[i]) - bfv(y[i]))); mx = std::max(mx, (double)std::fabs(bfv(x[i])));
        }
      return e / std::max(mx, 1e-30);
    };
    const double eo = worst(A.o, B.o, HD, 0, HD), eog = worst(A.og, B.og, HD, 0, HD), eg = worst(A.g, B.g, P, 3 * HD, HD);
    // (the log-sum-exp as the backward reads it, the pair's sum: gridFwdTc's max is the one O was last rescaled at,
    // within 2^8 of the true one, and its log-sum moves with it)
    double el = 0;
    const size_t half = A.lse.size() / 2;
    for (size_t i = 0; i < half; ++i) {
      const float x = A.lse[i] + A.lse[half + i], y = B.lse[i] + B.lse[half + i];
      if (std::isinf(x) || std::isinf(y)) { if (x != y) el = INFINITY; continue; }
      el = std::max(el, (double)std::fabs(x - y));
    }
    // (bf16 outputs: the two forms round P to bf16 alike, and sum in a different order - a few bf16 ulps of the max)
    const bool ok = eo < 2e-2 && eog < 2e-2 && eg == 0 && el < 1e-3;
    printf("n %4d sd %.0f%s%s: o %.2e, o g %.2e, gate %.1e (relative to the max), lse %.2e  %s\n", n, c.sd,
           c.holes ? " masked keys" : "", c.deadRow ? " + a row masked whole" : "", eo, eog, eg, el, ok ? "PASS" : "FAIL");
    fails += !ok;
    GK(cudaFree(f)); GK(cudaFree(nb)); GK(cudaFree(m));
  }
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
