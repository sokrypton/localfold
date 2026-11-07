// pair_tc5.cuh's gridInTc against gridInK (mma.sync) on the same pair: q and gate (f16), k and v (e4m3), the bias heads -
// both rows and columns (tr), with the bias and without, and a row window; then the two timed, interleaved.
//   nvcc -std=c++17 -arch=sm_100a -O2 --default-stream per-thread cuda/af3/test_pair_tc5.cu -o test-pair-tc5 -lcublas -lcublasLt -ldl
//   ./test-pair-tc5 [n=512] [reps=20]
#include "src/common.cuh"
#include "src/fusedtransition.cuh"
#include "src/fused256.cuh"

// triInTc against a host reference: a, b (e4m3 planes) and t2 on sampled rows, both directions, a mask with zeros
static int mainTri(int argc, char** argv) {
  const int n = argc > 2 ? atoi(argv[2]) : 300, reps = argc > 3 ? atoi(argv[3]) : 10, C = 128, np = (n + 15) / 16 * 16;
  const size_t cs = (size_t)np * np, pairs = (size_t)n * n;
  CK(cudaStreamCreate(&STREAM)); PAIR16 = true;
  uint64_t s = 9;
  auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  std::vector<__nv_bfloat16> hp(pairs * C); for (auto& v : hp) v = __float2bfloat16(3.f * rnd() + 0.3f);
  std::vector<float> hs(C), ho(C), hm(pairs); for (auto& v : hs) v = 1.f + 0.3f * rnd(); for (auto& v : ho) v = 0.2f * rnd();
  for (auto& v : hm) v = rnd() > -0.4f ? 1.f : 0.f;
  std::vector<half> wp((size_t)C * 2 * C), wg((size_t)C * 2 * C), wl((size_t)C * C);
  for (auto& v : wp) v = __float2half(0.15f * rnd()); for (auto& v : wg) v = __float2half(0.3f * rnd()); for (auto& v : wl) v = __float2half(0.15f * rnd());
  __nv_bfloat16* pair; float *sc, *of, *mk; half *dwp, *dwg, *dwl, *t2; uint8_t *a8, *b8;
  CK(cudaMalloc(&pair, pairs * C * 2)); CK(cudaMalloc(&sc, C * 4)); CK(cudaMalloc(&of, C * 4)); CK(cudaMalloc(&mk, pairs * 4));
  CK(cudaMalloc(&dwp, wp.size() * 2)); CK(cudaMalloc(&dwg, wg.size() * 2)); CK(cudaMalloc(&dwl, wl.size() * 2));
  CK(cudaMalloc(&t2, cs * C * 2)); CK(cudaMalloc(&a8, cs * C)); CK(cudaMalloc(&b8, cs * C));
  CK(cudaMemcpy(pair, hp.data(), pairs * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sc, hs.data(), C * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(of, ho.data(), C * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(mk, hm.data(), pairs * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwp, wp.data(), wp.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dwg, wg.data(), wg.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwl, wl.data(), wl.size() * 2, cudaMemcpyHostToDevice));
  auto e4 = [](uint8_t v) { __nv_fp8_e4m3 f; *reinterpret_cast<uint8_t*>(&f) = v; return (double)(float)f; };
  int bad = 0;
  const int Tlo = argc > 4 ? atoi(argv[4]) : 0, Thi = argc > 4 ? Tlo + 1 : 2;
  for (int T = Tlo; T < Thi; ++T) {
    CK(cudaMemset(a8, 0x7f, cs * C)); CK(cudaMemset(b8, 0x7f, cs * C)); CK(cudaMemset(t2, 0x7f, cs * C * 2));
    CK(cudaDeviceSynchronize());   // (the memsets ran on the per-thread default stream, which does not order with STREAM)
    if (!ptc::triInTcRun(true, (const float*)pair, mk, sc, of, dwp, dwg, dwl, a8, b8, t2, n, np, cs, T)) { fprintf(stderr, "not taken\n"); return 1; }
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> ha(cs * C), hb(cs * C); std::vector<half> ht(cs * C);
    CK(cudaMemcpy(ha.data(), a8, ha.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), b8, hb.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ht.data(), t2, ht.size() * 2, cudaMemcpyDeviceToHost));
    double na = 0, da = 0, nt = 0, dt = 0; size_t padBad = 0;
    for (int t = 0; t < 96; ++t) {
      const size_t q = (size_t)t * 7919 % cs; const int i = (int)(q / np), j = (int)(q % np);
      const bool real = i < n && j < n;
      const size_t P = real ? (T ? (size_t)j * n + i : (size_t)i * n + j) : 0;
      std::vector<double> x(C, 0.0);
      if (real) {
        double m = 0, v = 0; for (int c = 0; c < C; ++c) m += __bfloat162float(hp[P * C + c]); m /= C;
        for (int c = 0; c < C; ++c) { double d = __bfloat162float(hp[P * C + c]) - m; v += d * d; } v = 1 / sqrt(v / C + 1e-5);
        for (int c = 0; c < C; ++c) x[c] = __half2float(__float2half((float)((__bfloat162float(hp[P * C + c]) - m) * v * hs[c] + ho[c])));
      } else {
        for (int c = 0; c < C; ++c) x[c] = __half2float(__float2half(ho[c]));   // (a zero row's LN: the offset)
      }
      const double mm = real ? hm[P] : 0.0;
      for (int ch = 0; ch < C; ++ch) {
        double pa = 0, pb = 0, ga = 0, gb = 0;
        for (int c = 0; c < C; ++c) {
          pa += x[c] * __half2float(wp[(size_t)c * 2 * C + 2 * ch]); pb += x[c] * __half2float(wp[(size_t)c * 2 * C + 2 * ch + 1]);
          ga += x[c] * __half2float(wg[(size_t)c * 2 * C + 2 * ch]); gb += x[c] * __half2float(wg[(size_t)c * 2 * C + 2 * ch + 1]);
        }
        const double va = pa / (1 + exp(-ga)) * mm, vb = pb / (1 + exp(-gb)) * mm;
        const double ga8 = e4(ha[(size_t)ch * cs + q]), gb8 = e4(hb[(size_t)ch * cs + q]);
        na += (ga8 - va) * (ga8 - va) + (gb8 - vb) * (gb8 - vb); da += va * va + vb * vb;
        if (!real && (ga8 != 0.0 || gb8 != 0.0)) ++padBad;   // (by value: -0, 0x80, is a padding zero too)
        double tl = 0; for (int c = 0; c < C; ++c) tl += x[c] * __half2float(wl[(size_t)c * C + ch]);
        const size_t tr = T ? (size_t)j * np + i : q;
        const double got = __half2float(ht[tr * C + ch]);
        nt += (got - tl) * (got - tl); dt += tl * tl;
      }
    }
    printf("tri n %d T %d: a,b relRMS %.3e (e4m3), t2 relRMS %.3e, padding nonzero %zu\n", n, T, sqrt(na / da), sqrt(nt / dt), padBad);
    bad += !(sqrt(na / da) < 4e-2) || !(sqrt(nt / dt) < 2e-3) || padBad;
  }
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> tt;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0, STREAM)); ptc::triInTcRun(true, (const float*)pair, mk, sc, of, dwp, dwg, dwl, a8, b8, t2, n, np, cs, false);
    CK(cudaEventRecord(e1, STREAM)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); tt.push_back(ms);
  }
  std::sort(tt.begin(), tt.end());
  printf("tri n %d: triInTc %.3f ms%s\n", n, tt[reps / 2], bad ? "  FAILED" : "");
  return bad ? 1 : 0;
}
// transTc against a host reference (x += (silu(LN x W1a) LN x W1b) W2) on sampled rows, rows not a multiple of the block
static int mainTrans(int argc, char** argv) {
  const size_t rows = argc > 2 ? atol(argv[2]) : 100000; const int reps = argc > 3 ? atoi(argv[3]) : 10, C = 128, I = 512;
  CK(cudaStreamCreate(&STREAM)); PAIR16 = true;
  uint64_t s = 11;
  auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  std::vector<__nv_bfloat16> hx(rows * C); for (auto& v : hx) v = __float2bfloat16(3.f * rnd() + 0.3f);
  std::vector<float> hs(C), ho(C); for (auto& v : hs) v = 1.f + 0.3f * rnd(); for (auto& v : ho) v = 0.2f * rnd();
  std::vector<half> w1((size_t)C * 2 * I), w2((size_t)I * C); for (auto& v : w1) v = __float2half(0.15f * rnd()); for (auto& v : w2) v = __float2half(0.08f * rnd());
  __nv_bfloat16* x; float *sc, *of; half *d1, *d2;
  CK(cudaMalloc(&x, (rows + 256) * C * 2)); CK(cudaMalloc(&sc, C * 4)); CK(cudaMalloc(&of, C * 4)); CK(cudaMalloc(&d1, w1.size() * 2)); CK(cudaMalloc(&d2, w2.size() * 2));
  std::vector<__nv_bfloat16> guard(256 * C, __float2bfloat16(7.f));
  CK(cudaMemcpy(x, hx.data(), rows * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(x + rows * C, guard.data(), guard.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(sc, hs.data(), C * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(of, ho.data(), C * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d1, w1.data(), w1.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d2, w2.data(), w2.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
  // the old kernel on a copy, scored against the same reference
  __nv_bfloat16* x0; CK(cudaMalloc(&x0, rows * C * 2)); CK(cudaMemcpy(x0, x, rows * C * 2, cudaMemcpyDeviceToDevice));
  CK(cudaDeviceSynchronize());
  ptc::PAIR_TC = false; fusedTransitionRaw((float*)x0, rows, C, I, sc, of, d1, d2); ptc::PAIR_TC = true;
  if (!ptc::transTcRun(true, (float*)x, rows, C, I, sc, of, d1, d2)) { fprintf(stderr, "not taken\n"); return 1; }
  CK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> hy((rows + 256) * C), h0(rows * C); CK(cudaMemcpy(hy.data(), x, hy.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h0.data(), x0, h0.size() * 2, cudaMemcpyDeviceToHost));
  double num0 = 0;
  double num = 0, den = 0, dnum = 0; size_t guardBad = 0;
  for (size_t i = rows * C; i < (rows + 256) * C; ++i) guardBad += __bfloat162float(hy[i]) != 7.f;
  for (int t = 0; t < 64; ++t) {
    const size_t r = (t == 63) ? rows - 1 : (size_t)t * 7919 % rows;
    double m = 0, v = 0; for (int c = 0; c < C; ++c) m += __bfloat162float(hx[r * C + c]); m /= C;
    for (int c = 0; c < C; ++c) { double d = __bfloat162float(hx[r * C + c]) - m; v += d * d; } v = 1 / sqrt(v / C + 1e-5);
    std::vector<double> xn(C), h(I);
    for (int c = 0; c < C; ++c) xn[c] = __half2float(__float2half((float)((__bfloat162float(hx[r * C + c]) - m) * v * hs[c] + ho[c])));
    for (int j = 0; j < I; ++j) {
      double a = 0, b = 0; for (int c = 0; c < C; ++c) { a += xn[c] * __half2float(w1[(size_t)c * 2 * I + j]); b += xn[c] * __half2float(w1[(size_t)c * 2 * I + I + j]); }
      h[j] = __half2float(__float2half((float)(a / (1 + exp(-a)) * b)));
    }
    for (int o = 0; o < C; ++o) {
      double u = 0; for (int j = 0; j < I; ++j) u += h[j] * __half2float(w2[(size_t)j * C + o]);
      const double want = __bfloat162float(hx[r * C + o]) + u, got = __bfloat162float(hy[r * C + o]);
      num += (got - want) * (got - want); den += want * want; dnum += u * u;
      const double g0 = __bfloat162float(h0[r * C + o]); num0 += (g0 - want) * (g0 - want);
    }
  }
  const double rel = sqrt(num / den), relU = sqrt(num / dnum);
  const double relU0 = sqrt(num0 / dnum);
  printf("trans rows %zu: relRMS %.3e of x + update (%.3e of the update alone; fusedTransitionK %.3e); guard rows changed %zu\n", rows, rel, relU, relU0, guardBad);
  const int bad = !(relU < 1.25 * relU0 + 1e-3) || guardBad;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> t1, t2;
  for (int r = 0; r < reps; ++r)
    for (int arm = 0; arm < 2; ++arm) {
      ptc::PAIR_TC = arm;
      CK(cudaEventRecord(e0, STREAM));
      if (arm) ptc::transTcRun(true, (float*)x, rows, C, I, sc, of, d1, d2);
      else fusedTransitionRaw((float*)x, rows, C, I, sc, of, d1, d2);
      CK(cudaEventRecord(e1, STREAM)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); (arm ? t2 : t1).push_back(ms);
    }
#ifdef TC_PROF
  { unsigned long long hp[16][8]; CK(cudaMemcpyFromSymbol(hp, tc5a::tcProf, sizeof hp));
    for (int w = 0; w < 16; ++w) { if (!hp[w][7]) continue; printf("  warp %2d:", w); for (int k = 0; k < 8; ++k) printf(" %5.1f%%", 100.0 * hp[w][k] / hp[w][7]); printf("\n"); } }
#endif
  std::sort(t1.begin(), t1.end()); std::sort(t2.begin(), t2.end());
  printf("trans rows %zu: fusedTransitionK %.3f ms, transTc %.3f ms (%.2fx)%s\n", rows, t1[reps / 2], t2[reps / 2], t1[reps / 2] / t2[reps / 2], bad ? "  FAILED" : "");
  return bad;
}
// gridOutTc against gridOutK and a host reference on sampled rows, both directions, a bias
static int mainOut(int argc, char** argv) {
  const int n = argc > 2 ? atoi(argv[2]) : 300, reps = argc > 3 ? atoi(argv[3]) : 10, C = 128;
  const size_t pairs = (size_t)n * n;
  CK(cudaStreamCreate(&STREAM)); PAIR16 = true;
  uint64_t s = 13;
  auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  std::vector<__nv_bfloat16> hp(pairs * C); for (auto& v : hp) v = __float2bfloat16(3.f * rnd());
  std::vector<half> hg(pairs * C), hw((size_t)C * C); for (auto& v : hg) v = __float2half(rnd()); for (auto& v : hw) v = __float2half(0.15f * rnd());
  std::vector<float> hob(C); for (auto& v : hob) v = 0.3f * rnd();
  __nv_bfloat16 *p1, *p2; half *g, *w; float* ob;
  CK(cudaMalloc(&p1, pairs * C * 2)); CK(cudaMalloc(&p2, pairs * C * 2)); CK(cudaMalloc(&g, pairs * C * 2)); CK(cudaMalloc(&w, C * C * 2)); CK(cudaMalloc(&ob, C * 4));
  CK(cudaMemcpy(g, hg.data(), pairs * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(w, hw.data(), C * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ob, hob.data(), C * 4, cudaMemcpyHostToDevice));
  int bad = 0;
  for (int tr = 0; tr < 2; ++tr) {
    CK(cudaMemcpy(p1, hp.data(), pairs * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(p2, hp.data(), pairs * C * 2, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    ptc::PAIR_TC = false; gridOutRaw<128>(g, w, ob, (float*)p1, n, 0, pairs, tr);
    ptc::PAIR_TC = true; if (!ptc::gridOutTcRun(true, g, w, ob, (float*)p2, n, 0, pairs, tr)) { fprintf(stderr, "not taken\n"); return 1; }
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> y1(pairs * C), y2(pairs * C);
    CK(cudaMemcpy(y1.data(), p1, y1.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y2.data(), p2, y2.size() * 2, cudaMemcpyDeviceToHost));
    double n1 = 0, n2 = 0, den = 0, d12 = 0;
    for (int t = 0; t < 128; ++t) {
      const size_t q = (size_t)t * 7919 % pairs, pe = tr ? (q % n) * n + q / n : q;
      for (int o = 0; o < C; ++o) {
        double u = hob[o]; for (int c = 0; c < C; ++c) u += (double)__half2float(hg[q * C + c]) * __half2float(hw[(size_t)c * C + o]);
        const double want = __bfloat162float(hp[pe * C + o]) + u, a = __bfloat162float(y1[pe * C + o]), b = __bfloat162float(y2[pe * C + o]);
        n1 += (a - want) * (a - want); n2 += (b - want) * (b - want); den += u * u; d12 += (a - b) * (a - b);
      }
    }
    printf("out n %d tr %d: relRMS of the update from a host reference: gridOutK %.3e, gridOutTc %.3e (between them %.3e)\n", n, tr, sqrt(n1 / den), sqrt(n2 / den), sqrt(d12 / den));
    bad += !(sqrt(n2 / den) < 1.25 * sqrt(n1 / den) + 1e-3);
  }
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> t1, t2;
  for (int r = 0; r < reps; ++r)
    for (int arm = 0; arm < 2; ++arm) {
      ptc::PAIR_TC = arm;
      CK(cudaEventRecord(e0, STREAM)); gridOutRaw<128>(g, w, ob, (float*)(arm ? p2 : p1), n, 0, pairs, true);
      CK(cudaEventRecord(e1, STREAM)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); (arm ? t2 : t1).push_back(ms);
    }
  std::sort(t1.begin(), t1.end()); std::sort(t2.begin(), t2.end());
  printf("out n %d: gridOutK %.3f ms, gridOutTc %.3f ms (%.2fx)%s\n", n, t1[reps / 2], t2[reps / 2], t1[reps / 2] / t2[reps / 2], bad ? "  FAILED" : "");
  return bad;
}
// triOutTc against triangleOutRun's mma.sync kernel and a host reference on sampled pairs
static int mainTout(int argc, char** argv) {
  const int L = argc > 2 ? atoi(argv[2]) : 300, reps = argc > 3 ? atoi(argv[3]) : 10, C = 128, Lp = (L + 15) / 16 * 16;
  const size_t P = (size_t)L * L, plane = (size_t)Lp * Lp;
  CK(cudaStreamCreate(&STREAM)); PAIR16 = true;
  uint64_t s = 17;
  auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  std::vector<__nv_bfloat16> hpr(plane * C), hp(P * C); for (auto& v : hpr) v = __float2bfloat16(2.f * rnd() + 0.2f); for (auto& v : hp) v = __float2bfloat16(3.f * rnd());
  std::vector<half> ht(plane * C), hw((size_t)C * C); for (auto& v : ht) v = __float2half(2.f * rnd()); for (auto& v : hw) v = __float2half(0.15f * rnd());
  std::vector<float> hs(C), ho(C); for (auto& v : hs) v = 1.f + 0.3f * rnd(); for (auto& v : ho) v = 0.2f * rnd();
  __nv_bfloat16 *prod, *p1, *p2; half *t2, *w; float *sc, *of;
  CK(cudaMalloc(&prod, plane * C * 2)); CK(cudaMalloc(&p1, P * C * 2)); CK(cudaMalloc(&p2, P * C * 2)); CK(cudaMalloc(&t2, plane * C * 2)); CK(cudaMalloc(&w, C * C * 2));
  CK(cudaMalloc(&sc, C * 4)); CK(cudaMalloc(&of, C * 4));
  CK(cudaMemcpy(prod, hpr.data(), plane * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(t2, ht.data(), plane * C * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(w, hw.data(), C * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sc, hs.data(), C * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(of, ho.data(), C * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(p1, hp.data(), P * C * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(p2, hp.data(), P * C * 2, cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
  ptc::PAIR_TC = false; triangleOutRun<128, 4, __nv_bfloat16>(prod, sc, of, w, t2, (float*)p1, L, Lp);
  ptc::PAIR_TC = true; triangleOutRun<128, 4, __nv_bfloat16>(prod, sc, of, w, t2, (float*)p2, L, Lp);
  CK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> y1(P * C), y2(P * C); CK(cudaMemcpy(y1.data(), p1, y1.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y2.data(), p2, y2.size() * 2, cudaMemcpyDeviceToHost));
  double n1 = 0, n2 = 0, den = 0;
  for (int t = 0; t < 128; ++t) {
    const size_t pr = (size_t)t * 7919 % P, i = pr / L, j = pr % L, q = i * Lp + j;
    double m = 0, v = 0; for (int c = 0; c < C; ++c) m += __bfloat162float(hpr[(size_t)c * plane + q]); m /= C;
    for (int c = 0; c < C; ++c) { double d = __bfloat162float(hpr[(size_t)c * plane + q]) - m; v += d * d; } v = 1 / sqrt(v / C + 1e-5);
    std::vector<double> x(C); for (int c = 0; c < C; ++c) x[c] = __half2float(__float2half((float)((__bfloat162float(hpr[(size_t)c * plane + q]) - m) * v * hs[c] + ho[c])));
    for (int o = 0; o < C; ++o) {
      double u = 0; for (int c = 0; c < C; ++c) u += x[c] * __half2float(hw[(size_t)c * C + o]);
      u /= 1 + exp(-(double)__half2float(ht[q * C + o]));
      const double want = __bfloat162float(hp[pr * C + o]) + u, a = __bfloat162float(y1[pr * C + o]), b = __bfloat162float(y2[pr * C + o]);
      n1 += (a - want) * (a - want); n2 += (b - want) * (b - want); den += u * u;
    }
  }
  printf("tout L %d: relRMS of the update from a host reference: triangleOutK %.3e, triOutTc %.3e\n", L, sqrt(n1 / den), sqrt(n2 / den));
  const int bad = !(sqrt(n2 / den) < 1.25 * sqrt(n1 / den) + 1e-3);
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> t1, tt2;
  for (int r = 0; r < reps; ++r)
    for (int arm = 0; arm < 2; ++arm) {
      ptc::PAIR_TC = arm;
      CK(cudaEventRecord(e0, STREAM)); triangleOutRun<128, 4, __nv_bfloat16>(prod, sc, of, w, t2, (float*)(arm ? p2 : p1), L, Lp);
      CK(cudaEventRecord(e1, STREAM)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); (arm ? tt2 : t1).push_back(ms);
    }
  std::sort(t1.begin(), t1.end()); std::sort(tt2.begin(), tt2.end());
  printf("tout L %d: triangleOutK %.3f ms, triOutTc %.3f ms (%.2fx)%s\n", L, t1[reps / 2], tt2[reps / 2], t1[reps / 2] / tt2[reps / 2], bad ? "  FAILED" : "");
  return bad;
}
int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "tout")) return mainTout(argc, argv);
  if (argc > 1 && !strcmp(argv[1], "out")) return mainOut(argc, argv);
  if (argc > 1 && !strcmp(argv[1], "trans")) return mainTrans(argc, argv);
  if (argc > 1 && !strcmp(argv[1], "tri")) return mainTri(argc, argv);
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
        CK(cudaDeviceSynchronize());   // (the memsets ran on the per-thread default stream, which does not order with STREAM)
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
          printf("  tr %d bias %d window %d %-6s relRMS %.3e (against gridInK)\n", tr, withBias, win, nm, r);
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
