// The grid attention's backward on tcgen05 (flash_tc5.cuh's bwdTc) against bwdBf, on the inputs the pairformer hands it
// (af3.cuh's gridBackwardBf: q, k, v bf16 rows of one [n n, 4 H D] buffer, dO bf16 [n n, H D], a pair bias [H, n, n] every
// row shares with its float gradient, a key mask a row, dk and dv written in bf16 back into the packed buffer, dQ in float
// and narrowed into it) - every output compared, the two timed interleaved. B200 only (a binary with sm_100a code):
//
//   nvcc -std=c++17 -arch=sm_100a -O2 cuda/grad/test_tc5_bwd.cu -o test-tc5-bwd -lcublas -lcublasLt -lcurand -ldl && ./test-tc5-bwd [nodb]
#include "../af3/src/common.cuh"
#include "../af3/src/flash.cuh"         // (with its flash_tc5.cuh: tc5a, bwdTc's helpers, as train_of3.cu has them)
#include "ops.cuh"
#include <algorithm>
#include <curand.h>
using grad::Sdpa; using grad::Bias; namespace flash = grad::flash;
static float fromBf(uint16_t x) { uint32_t u = (uint32_t)x << 16; float f; memcpy(&f, &u, 4); return f; }
__global__ void toBfK(const float* x, __nv_bfloat16* y, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(x[i]);
}

int main(int argc, char** argv) {
  const bool noDb = argc > 1 && !strcmp(argv[1], "nodb");   // (nodb: no bias gradient - the atomics' share)
  grad::init();
  if (!tc5a::ready()) { printf("this device or binary has no tcgen05 (sm_100a): SKIP\n"); return 0; }
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 11);
  struct Case { int n; float sd; bool holes, deadRow; };
  int fails = 0;
  for (Case c : {Case{128, 1.f, false, false}, Case{256, 1.f, true, false}, Case{384, 2.f, true, true}, Case{512, 1.f, true, false},
                 Case{768, 1.f, true, false}}) {
    const int n = c.n, H = 4, D = 32, HD = H * D, P = 4 * HD;
    const size_t nn = (size_t)n * n;
    auto randBf = [&](size_t len, float sd) {
      float* f; GK(cudaMalloc(&f, len * 4)); curandGenerateNormal(gen, f, len + (len & 1), 0.f, sd);
      __nv_bfloat16* b; GK(cudaMalloc(&b, len * 2)); toBfK<<<(unsigned)((len + 255) / 256), 256>>>(f, b, len);
      GK(cudaDeviceSynchronize()); GK(cudaFree(f)); return b;
    };
    __nv_bfloat16* qkvg = randBf(nn * P, c.sd);
    __nv_bfloat16* dO = randBf(nn * HD, 1.f);
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
    // the forward's log-sum-exp (and o, for D_i = rowsum(dO o))
    __nv_bfloat16 *o, *og; float* lse;
    GK(cudaMalloc(&o, nn * HD * 2)); GK(cudaMalloc(&og, nn * HD * 2)); GK(cudaMalloc(&lse, 2 * nn * H * 4));
    {
      __nv_bfloat16* tmp; GK(cudaMalloc(&tmp, nn * P * 2)); GK(cudaMemcpy(tmp, qkvg, nn * P * 2, cudaMemcpyDeviceToDevice));
      flash::TC2 = false;
      if (!flash::gridForwardBf(a, tmp, P, tmp + HD, P, tmp + 2 * HD, P, o, HD, lse, tmp + 3 * HD, P, og, HD, false)) { printf("forward declined\n"); return 1; }
      GK(cudaDeviceSynchronize()); GK(cudaFree(tmp));
    }
    std::vector<uint16_t> ho(nn * HD), hdo(nn * HD);
    GK(cudaMemcpy(ho.data(), o, nn * HD * 2, cudaMemcpyDeviceToHost)); GK(cudaMemcpy(hdo.data(), dO, nn * HD * 2, cudaMemcpyDeviceToHost));
    std::vector<float> hDi(nn * H);
    for (size_t r = 0; r < nn; ++r)
      for (int h = 0; h < H; ++h) {
        double s = 0; for (int d = 0; d < D; ++d) s += (double)fromBf(ho[r * HD + h * D + d]) * fromBf(hdo[r * HD + h * D + d]);
        hDi[((r / n) * H + h) * n + r % n] = (float)s;   // ([b][h][q])
      }
    float* Di; GK(cudaMalloc(&Di, nn * H * 4)); GK(cudaMemcpy(Di, hDi.data(), nn * H * 4, cudaMemcpyHostToDevice));
    struct Out { std::vector<float> dq, dnb; std::vector<uint16_t> pk; };
    float *dq, *dnb; __nv_bfloat16* dpk;
    GK(cudaMalloc(&dq, nn * HD * 4)); GK(cudaMalloc(&dnb, (size_t)H * nn * 4)); GK(cudaMalloc(&dpk, nn * P * 2));
    auto run = [&](bool tc) {
      flash::TCB = tc;
      GK(cudaMemset(dq, 0, nn * HD * 4)); GK(cudaMemset(dnb, 0, (size_t)H * nn * 4)); GK(cudaMemset(dpk, 0, nn * P * 2));
      GK(cudaDeviceSynchronize());
      Sdpa b = a; b.bias[0].grad = noDb ? nullptr : dnb;
      flash::backwardBf<32>(b, qkvg, P, qkvg + HD, P, qkvg + 2 * HD, P, dO, HD, lse, Di, dq, HD, dpk + HD, P, dpk + 2 * HD, P,
                            false, false, dpk, P);
      GK(cudaDeviceSynchronize());
    };
    auto grab = [&]() {
      Out r; r.dq.resize(nn * HD); r.dnb.resize((size_t)H * nn); r.pk.resize(nn * P);
      GK(cudaMemcpy(r.dq.data(), dq, nn * HD * 4, cudaMemcpyDeviceToHost)); GK(cudaMemcpy(r.dnb.data(), dnb, (size_t)H * nn * 4, cudaMemcpyDeviceToHost));
      GK(cudaMemcpy(r.pk.data(), dpk, nn * P * 2, cudaMemcpyDeviceToHost));
      return r;
    };
    run(false); const Out A = grab();
    run(true); const Out B = grab();
    auto rel = [](auto ga, auto gb, size_t len) {
      double num = 0, den = 0;
      for (size_t i = 0; i < len; ++i) { const double x = ga(i), y = gb(i); num += (x - y) * (x - y); den += y * y; }
      return std::sqrt(num / std::max(den, 1e-30));
    };
    const double eq = rel([&](size_t i) { return B.dq[i]; }, [&](size_t i) { return A.dq[i]; }, nn * HD);
    const double eb = rel([&](size_t i) { return B.dnb[i]; }, [&](size_t i) { return A.dnb[i]; }, (size_t)H * nn);
    double er[3];   // dqo, dk, dv in the packed buffer
    for (int role = 0; role < 3; ++role)
      er[role] = rel([&](size_t i) { return fromBf(B.pk[(i / HD) * P + role * HD + i % HD]); },
                     [&](size_t i) { return fromBf(A.pk[(i / HD) * P + role * HD + i % HD]); }, nn * HD);
    // timing, interleaved: medians of 7
    std::vector<float> ta, tb;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int it = 0; it < 14; ++it) {
      const bool tc = it & 1;
      flash::TCB = tc;
      GK(cudaMemsetAsync(dq, 0, nn * HD * 4, grad::STREAM)); GK(cudaMemsetAsync(dnb, 0, (size_t)H * nn * 4, grad::STREAM));
      Sdpa b = a; b.bias[0].grad = noDb ? nullptr : dnb;
      cudaEventRecord(e0, grad::STREAM);
      flash::backwardBf<32>(b, qkvg, P, qkvg + HD, P, qkvg + 2 * HD, P, dO, HD, lse, Di, dq, HD, dpk + HD, P, dpk + 2 * HD, P,
                            false, false, dpk, P);
      cudaEventRecord(e1, grad::STREAM); GK(cudaEventSynchronize(e1));
      float ms; cudaEventElapsedTime(&ms, e0, e1); (tc ? tb : ta).push_back(ms);
    }
    std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
    const double bound = 2e-2;
    const bool ok = eq < bound && eb < bound && er[0] < bound && er[1] < bound && er[2] < bound;
    fails += !ok;
    printf("n %4d: relRMS dq %.2e dnb %.2e dqo %.2e dk %.2e dv %.2e | bwdBf %.3f ms, bwdTc %.3f ms (%.2fx)  %s\n", n, eq, eb, er[0], er[1],
           er[2], ta[ta.size() / 2], tb[tb.size() / 2], ta[ta.size() / 2] / tb[tb.size() / 2], ok ? "PASS" : "FAIL");
    for (void* p : {(void*)qkvg, (void*)dO, (void*)nb, (void*)m, (void*)o, (void*)og, (void*)lse, (void*)Di, (void*)dq, (void*)dnb, (void*)dpk}) GK(cudaFree(p));
  }
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails != 0;
}
