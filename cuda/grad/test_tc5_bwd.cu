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

// bwdTc's other paths, against bwdBf: Q != K, dk / dv in float accumulated onto what is there (accK, accV), two biases
// with gradients (one a head, one every head shares), a bias tile every head shares (bt.sh 0); q, k, v their own rows
static int variants(curandGenerator_t gen) {
  int fails = 0;
  struct V { int B, Q, K; bool twoBias, headShared; const char* name; };
  for (V v : {V{64, 256, 384, false, false, "Q 256 K 384, float dk / dv accumulated"},
              V{48, 384, 256, true, false, "Q 384 K 256, two biases (one head-shared) with gradients"},
              V{32, 256, 256, false, true, "a head-shared bias tile"}}) {
    const int H = 4, D = 32, HD = H * D;
    const size_t nq = (size_t)v.B * v.Q, nk = (size_t)v.B * v.K, qk = (size_t)v.Q * v.K;
    auto randBf = [&](size_t len, float sd) {
      float* f; GK(cudaMalloc(&f, (len + 1) * 4)); curandGenerateNormal(gen, f, len + (len & 1), 0.f, sd);
      __nv_bfloat16* b; GK(cudaMalloc(&b, len * 2)); toBfK<<<(unsigned)((len + 255) / 256), 256>>>(f, b, len);
      GK(cudaDeviceSynchronize()); GK(cudaFree(f)); return b;
    };
    auto randF = [&](size_t len, float sd) { float* f; GK(cudaMalloc(&f, (len + 1) * 4)); curandGenerateNormal(gen, f, len + (len & 1), 0.f, sd); GK(cudaDeviceSynchronize()); return f; };
    __nv_bfloat16 *q = randBf(nq * HD, 1.f), *k = randBf(nk * HD, 1.f), *vv = randBf(nk * HD, 1.f), *dO = randBf(nq * HD, 1.f);
    float* b0 = randF((v.headShared ? 1 : H) * qk, 1.f);
    float* b1 = v.twoBias ? randF(qk, 1.f) : nullptr;
    std::vector<float> hm(nk, 1.f);
    for (size_t i = 0; i < nk; ++i) if (i % 13 == 5) hm[i] = 0.f;
    float* m; GK(cudaMalloc(&m, nk * 4)); GK(cudaMemcpy(m, hm.data(), nk * 4, cudaMemcpyHostToDevice));
    Sdpa a; a.B = v.B; a.H = H; a.Q = v.Q; a.K = v.K; a.D = D; a.scale = 1 / sqrtf((float)D); a.keyMask = m;
    a.bias[0] = Bias{b0, 0, v.headShared ? 0 : (long long)qk, v.K, nullptr};
    if (b1) a.bias[1] = Bias{b1, 0, 0, v.K, nullptr};
    float *o, *lse; GK(cudaMalloc(&o, nq * HD * 4)); GK(cudaMalloc(&lse, 2 * nq * H * 4));
    if (!flash::runForwardBfIn(a, q, HD, k, HD, vv, HD, o, HD, lse)) { printf("forward declined\n"); return 1; }
    GK(cudaDeviceSynchronize());
    std::vector<float> ho(nq * HD); std::vector<uint16_t> hdo(nq * HD);
    GK(cudaMemcpy(ho.data(), o, nq * HD * 4, cudaMemcpyDeviceToHost)); GK(cudaMemcpy(hdo.data(), dO, nq * HD * 2, cudaMemcpyDeviceToHost));
    std::vector<float> hDi(nq * H);
    for (size_t r = 0; r < nq; ++r)
      for (int h = 0; h < H; ++h) {
        double sum = 0; for (int d = 0; d < D; ++d) sum += (double)ho[r * HD + h * D + d] * fromBf(hdo[r * HD + h * D + d]);
        hDi[((r / v.Q) * H + h) * v.Q + r % v.Q] = (float)sum;
      }
    float* Di; GK(cudaMalloc(&Di, nq * H * 4)); GK(cudaMemcpy(Di, hDi.data(), nq * H * 4, cudaMemcpyHostToDevice));
    float* init = randF(nk * HD, 1.f);   // (what dk and dv hold before: accumulated onto)
    float *dq, *dk, *dv, *g0, *g1;
    GK(cudaMalloc(&dq, nq * HD * 4)); GK(cudaMalloc(&dk, nk * HD * 4)); GK(cudaMalloc(&dv, nk * HD * 4));
    const size_t n0 = (v.headShared ? 1 : H) * qk;
    GK(cudaMalloc(&g0, n0 * 4)); GK(cudaMalloc(&g1, qk * 4));
    const bool acc = v.Q != v.K || v.twoBias;
    auto run = [&](bool tc) {
      flash::TCB = tc;
      GK(cudaMemset(dq, 0, nq * HD * 4)); GK(cudaMemset(g0, 0, n0 * 4)); GK(cudaMemset(g1, 0, qk * 4));
      GK(cudaMemcpy(dk, init, nk * HD * 4, cudaMemcpyDeviceToDevice)); GK(cudaMemcpy(dv, init, nk * HD * 4, cudaMemcpyDeviceToDevice));
      GK(cudaDeviceSynchronize());
      Sdpa b = a; b.bias[0].grad = g0; if (b1) b.bias[1].grad = g1;
      flash::backwardBf<32>(b, q, HD, k, HD, vv, HD, dO, HD, lse, Di, dq, HD, dk, HD, dv, HD, acc, acc);
      GK(cudaDeviceSynchronize());
      std::vector<float> out(nq * HD + 2 * nk * HD + n0 + qk);
      float* w = out.data();
      GK(cudaMemcpy(w, dq, nq * HD * 4, cudaMemcpyDeviceToHost)); w += nq * HD;
      GK(cudaMemcpy(w, dk, nk * HD * 4, cudaMemcpyDeviceToHost)); w += nk * HD;
      GK(cudaMemcpy(w, dv, nk * HD * 4, cudaMemcpyDeviceToHost)); w += nk * HD;
      GK(cudaMemcpy(w, g0, n0 * 4, cudaMemcpyDeviceToHost)); w += n0;
      GK(cudaMemcpy(w, g1, qk * 4, cudaMemcpyDeviceToHost));
      return out;
    };
    const auto A = run(false), Bv = run(true);
    const size_t seg[5] = {nq * HD, nk * HD, nk * HD, n0, qk};
    const char* nm[5] = {"dq", "dk", "dv", "dbias0", "dbias1"};
    size_t off = 0; bool ok = true; printf("  %s:", v.name);
    for (int t = 0; t < 5; ++t) {
      double num = 0, den = 0;
      for (size_t i = off; i < off + seg[t]; ++i) { const double d = Bv[i] - A[i]; num += d * d; den += (double)A[i] * A[i]; }
      off += seg[t];
      if (t == 4 && !b1) continue;
      const double e = std::sqrt(num / std::max(den, 1e-30));
      ok &= e < 2e-2 && std::isfinite(e);
      printf(" %s %.2e", nm[t], e);
    }
    printf("  %s\n", ok ? "PASS" : "FAIL");
    fails += !ok;
    for (void* p : {(void*)q, (void*)k, (void*)vv, (void*)dO, (void*)b0, (void*)m, (void*)o, (void*)lse, (void*)Di, (void*)init, (void*)dq,
                    (void*)dk, (void*)dv, (void*)g0, (void*)g1}) GK(cudaFree(p));
    if (b1) GK(cudaFree(b1));
  }
  return fails;
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
  printf("variants (bwdTc against bwdBf):\n");
  fails += variants(gen);
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails != 0;
}
