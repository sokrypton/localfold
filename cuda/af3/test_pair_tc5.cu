// pair_tc5.cuh's gridInTc against gridInK (mma.sync) on the same pair: q and gate (f16), k and v (e4m3), the bias heads -
// both rows and columns (tr), with the bias and without, and a row window; then the two timed, interleaved.
//   nvcc -std=c++17 -arch=sm_100a -O2 --default-stream per-thread cuda/af3/test_pair_tc5.cu -o test-pair-tc5 -lcublas -lcublasLt -ldl
//   ./test-pair-tc5 [n=512] [reps=20]
#include "src/common.cuh"
#include "src/fusedtriangle.cuh"

int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 512, reps = argc > 2 ? atoi(argv[2]) : 20, C = 128, NQ = 512, heads = 4, stride = (n + 7) / 8 * 8;
  if (!tc5a::ready()) { fprintf(stderr, "no sm_100a tcgen05 here\n"); return 1; }
  CK(cudaStreamCreate(&STREAM));
  PAIR16 = true;
  const size_t pairs = (size_t)n * n;
  uint64_t s = 5;
  auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  std::vector<__nv_bfloat16> hp(pairs * C); for (auto& v : hp) v = __float2bfloat16(3.f * rnd() + 0.3f);
  std::vector<float> hs(C), ho(C), hqb(C); for (auto& v : hs) v = 1.f + 0.3f * rnd(); for (auto& v : ho) v = 0.2f * rnd(); for (auto& v : hqb) v = rnd();
  std::vector<half> hw((size_t)C * NQ), hb((size_t)C * 16); for (auto& v : hw) v = __float2half(0.15f * rnd()); for (auto& v : hb) v = __float2half(0.3f * rnd());
  __nv_bfloat16* pair; float *sc, *of, *qb; half *w, *wb;
  CK(cudaMalloc(&pair, pairs * C * 2)); CK(cudaMalloc(&sc, C * 4)); CK(cudaMalloc(&of, C * 4)); CK(cudaMalloc(&qb, C * 4));
  CK(cudaMalloc(&w, hw.size() * 2)); CK(cudaMalloc(&wb, hb.size() * 2));
  CK(cudaMemcpy(pair, hp.data(), pairs * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sc, hs.data(), C * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(of, ho.data(), C * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(qb, hqb.data(), C * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(wb, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
  half *o1, *o2, *b1, *b2; uint8_t *k1, *k2;
  CK(cudaMalloc(&o1, pairs * NQ * 2)); CK(cudaMalloc(&o2, pairs * NQ * 2)); CK(cudaMalloc(&k1, pairs * 2 * C)); CK(cudaMalloc(&k2, pairs * 2 * C));
  CK(cudaMalloc(&b1, (size_t)heads * n * stride * 2)); CK(cudaMalloc(&b2, (size_t)heads * n * stride * 2));
  int bad = 0;
  for (int tr = 0; tr < 2; ++tr)
    for (int withBias = 0; withBias < 2; ++withBias)
      for (int win = 0; win < 2; ++win) {
        const size_t wlo = win ? pairs / 3 : 0, whi = win ? pairs / 3 + pairs / 2 : SIZE_MAX, wr = win ? pairs / 2 : pairs;
        for (auto o : {o1, o2}) CK(cudaMemset(o, 0, pairs * NQ * 2));
        for (auto k : {k1, k2}) CK(cudaMemset(k, 0, pairs * 2 * C));
        for (auto b : {b1, b2}) CK(cudaMemset(b, 0, (size_t)heads * n * stride * 2));
        ptc::PAIR_TC = false;
        gridInRaw<128>((const float*)pair, sc, of, w, qb - 3 * C, o1, n, 0, pairs, tr, withBias ? wb : nullptr, withBias ? b1 : nullptr, heads, stride, tr, k1, false, nullptr, wlo, whi);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        ptc::PAIR_TC = true;
        gridInRaw<128>((const float*)pair, sc, of, w, qb - 3 * C, o2, n, 0, pairs, tr, withBias ? wb : nullptr, withBias ? b2 : nullptr, heads, stride, tr, k2, false, nullptr, wlo, whi);
        CK(cudaDeviceSynchronize());
        auto cmpH = [&](const half* a, const half* b, size_t cnt, const char* nm) {
          std::vector<half> x(cnt), y(cnt); CK(cudaMemcpy(x.data(), a, cnt * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y.data(), b, cnt * 2, cudaMemcpyDeviceToHost));
          double num = 0, den = 0; for (size_t i = 0; i < cnt; ++i) { double p = __half2float(x[i]), q = __half2float(y[i]); num += (p - q) * (p - q); den += p * p; }
          const double r = sqrt(num / std::max(den, 1e-30));
          printf("  tr %d bias %d window %d %-6s relRMS %.3e\n", tr, withBias, win, nm, r);
          bad += !(r < 2e-3);
        };
        auto cmp8 = [&](const uint8_t* a, const uint8_t* b, size_t cnt, const char* nm) {
          std::vector<uint8_t> x(cnt), y(cnt); CK(cudaMemcpy(x.data(), a, cnt, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y.data(), b, cnt, cudaMemcpyDeviceToHost));
          auto e4 = [](uint8_t v) { __nv_fp8_e4m3 f; *reinterpret_cast<uint8_t*>(&f) = v; return (double)(float)f; };
          double num = 0, den = 0; size_t diff = 0;
          for (size_t i = 0; i < cnt; ++i) { double p = e4(x[i]), q = e4(y[i]); num += (p - q) * (p - q); den += p * p; diff += x[i] != y[i]; }
          const double r = sqrt(num / std::max(den, 1e-30));
          printf("  tr %d bias %d window %d %-6s relRMS %.3e (%zu of %zu bytes differ)\n", tr, withBias, win, nm, r, diff, cnt);
        };
        {   // k against a host reference on sampled rows, each arm
          std::vector<uint8_t> x(wr * 2 * C), y(wr * 2 * C);
          CK(cudaMemcpy(x.data(), k1, x.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y.data(), k2, y.size(), cudaMemcpyDeviceToHost));
          auto e4 = [](uint8_t v) { __nv_fp8_e4m3 f; *reinterpret_cast<uint8_t*>(&f) = v; return (double)(float)f; };
          double n1 = 0, n2 = 0, den = 0;
          for (int t = 0; t < 64; ++t) {
            const size_t r = (size_t)t * 977 % wr, Q = wlo + r, P = tr ? (Q % n) * n + Q / n : Q;
            double m = 0, v = 0; for (int c = 0; c < C; ++c) m += __bfloat162float(hp[P * C + c]); m /= C;
            for (int c = 0; c < C; ++c) { double d = __bfloat162float(hp[P * C + c]) - m; v += d * d; } v = 1 / sqrt(v / C + 1e-5);
            for (int col = 0; col < 2 * C; ++col) {
              double acc = 0; for (int c = 0; c < C; ++c) acc += (double)__half2float(__float2half((float)((__bfloat162float(hp[P * C + c]) - m) * v * hs[c] + ho[c]))) * __half2float(hw[(size_t)c * NQ + C + col]);
              const size_t at = col < C ? r * C + col : wr * C + r * C + col - C;
              n1 += (e4(x[at]) - acc) * (e4(x[at]) - acc); n2 += (e4(y[at]) - acc) * (e4(y[at]) - acc); den += acc * acc;
            }
          }
          printf("  tr %d bias %d window %d k,v from a host reference: gridInK %.3e, gridInTc %.3e\n", tr, withBias, win, sqrt(n1 / den), sqrt(n2 / den));
          bad += !(sqrt(n2 / den) < 4e-2);
        }
        {   // q against a host reference on sampled rows (gridInTc)
          std::vector<half> y(wr * NQ); CK(cudaMemcpy(y.data(), o2, y.size() * 2, cudaMemcpyDeviceToHost));
          double n2 = 0, den = 0;
          for (int t = 0; t < 64; ++t) {
            const size_t r = (size_t)t * 977 % wr, Q = wlo + r, P = tr ? (Q % n) * n + Q / n : Q;
            double m = 0, v = 0; for (int c = 0; c < C; ++c) m += __bfloat162float(hp[P * C + c]); m /= C;
            for (int c = 0; c < C; ++c) { double d = __bfloat162float(hp[P * C + c]) - m; v += d * d; } v = 1 / sqrt(v / C + 1e-5);
            for (int col = 0; col < C; ++col) {
              double acc = 0; for (int c = 0; c < C; ++c) acc += (double)__half2float(__float2half((float)((__bfloat162float(hp[P * C + c]) - m) * v * hs[c] + ho[c]))) * __half2float(hw[(size_t)c * NQ + col]);
              const double p2 = __half2float(y[r * NQ + col]); n2 += (p2 - acc) * (p2 - acc); den += acc * acc;
            }
          }
          printf("  tr %d bias %d window %d q from a host reference: gridInTc %.3e\n", tr, withBias, win, sqrt(n2 / den));
          bad += !(sqrt(n2 / den) < 2e-3);
        }
        cmpH(o1, o2, wr * NQ, "q,gate");
        cmp8(k1, k2, wr * 2 * C, "k,v");   // (reported: gridInK's own kv8 is what moves in some arms - see the host reference)
        if (withBias) {   // the bias against a host reference, each arm
          std::vector<half> x((size_t)heads * n * stride), y(x.size());
          CK(cudaMemcpy(x.data(), b1, x.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y.data(), b2, y.size() * 2, cudaMemcpyDeviceToHost));
          double n1 = 0, n2 = 0, den = 0;
          for (int t = 0; t < 64; ++t) {
            const size_t Q = (size_t)t * 977 % pairs, P = tr ? (Q % n) * n + Q / n : Q;
            const size_t a = tr ? Q % n : Q / n, bb = tr ? Q / n : Q % n, i = tr ? bb : a, jj = tr ? a : bb;   // (swap = tr)
            double m = 0, v = 0; for (int c = 0; c < C; ++c) m += __bfloat162float(hp[P * C + c]); m /= C;
            for (int c = 0; c < C; ++c) { double d = __bfloat162float(hp[P * C + c]) - m; v += d * d; } v = 1 / sqrt(v / C + 1e-5);
            for (int h = 0; h < heads; ++h) {
              double acc = 0; for (int c = 0; c < C; ++c) acc += (double)__half2float(__float2half((float)((__bfloat162float(hp[P * C + c]) - m) * v * hs[c] + ho[c]))) * __half2float(hb[(size_t)c * 16 + h]);
              acc *= LOG2E;
              const size_t at = ((size_t)h * n + i) * stride + jj;
              const double p1 = __half2float(x[at]), p2 = __half2float(y[at]);
              n1 += (p1 - acc) * (p1 - acc); n2 += (p2 - acc) * (p2 - acc); den += acc * acc;
              if (fabs(p2 - acc) > 0.05 * fabs(acc) + 0.01 && t < 64) printf("    bad: Q %zu (%s window) h %d got %g want %g\n", Q, Q >= wlo && Q < (whi < pairs ? whi : pairs) ? "in" : "out of", h, p2, acc);
            }
          }
          printf("  tr %d bias %d window %d bias from a host reference: gridInK %.3e, gridInTc %.3e\n", tr, withBias, win, sqrt(n1 / den), sqrt(n2 / den));
          bad += !(sqrt(n2 / den) < 2e-3);
        }
      }
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> t1, t2;
  for (int r = 0; r < reps; ++r)
    for (int arm = 0; arm < 2; ++arm) {
      ptc::PAIR_TC = arm == 1;
      CK(cudaEventRecord(e0, STREAM));
      gridInRaw<128>((const float*)pair, sc, of, w, qb - 3 * C, arm ? o2 : o1, n, 0, pairs, false, wb, arm ? b2 : b1, heads, stride, false, arm ? k2 : k1);
      CK(cudaEventRecord(e1, STREAM)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); (arm ? t2 : t1).push_back(ms);
    }
  std::sort(t1.begin(), t1.end()); std::sort(t2.begin(), t2.end());
  printf("n %d (%zu pairs): gridInK %.3f ms, gridInTc %.3f ms (%.2fx)%s\n", n, pairs, t1[reps / 2], t2[reps / 2], t1[reps / 2] / t2[reps / 2],
         bad ? "  FAILED" : "");
  return bad ? 1 : 0;
}
