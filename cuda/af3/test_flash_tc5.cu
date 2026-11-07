// flash_tc5.cuh's tcgen05 attention (flashTc) against flash8 (mma.sync) on the same e4m3 inputs, and against a float32
// reference over a sample of rows: relRMS from the reference for both, and the two kernels' times interleaved.
//   nvcc -std=c++17 -arch=sm_100a -O2 --default-stream per-thread cuda/af3/test_flash_tc5.cu -o test-flash-tc5 -lcublas -lcublasLt -ldl
//   ./test-flash-tc5 h [n=512] [rows=5] [reps=20]   (the diffusion's 48-wide form, flashTcH)
//   ./test-flash-tc5 [n=1024] [rows=256] [reps=20] [skew=1: the bias's spread, which sets how peaked the softmax is] [rows a block=3]
#include "src/common.cuh"
#include "src/flash.cuh"

__global__ void kv8K(const half* qkvg, uint8_t* kv8, size_t prs, int W) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= prs * 2 * W) return;
  size_t which = t / (prs * W), e = t % (prs * W), row = e / W; int c = (int)(e % W);
  kv8[t] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<const __half_raw*>(qkvg + row * 4 * W + (1 + which) * W + c), __NV_SATFINITE, __NV_E4M3);
}
// the diffusion's 48-wide form: flashTcH against flashGridHalfLaunch<48> and a host f32 reference (16 heads, the
// samples as rows, a bias shared by them, a query bias)
static int mainH(int argc, char** argv) {
  const int n = argc > 2 ? atoi(argv[2]) : 512, heads = 16, D = 48, W = heads * D, stride = (n + 7) / 8 * 8;
  const size_t rows = argc > 3 ? atoi(argv[3]) : 5;
  const int reps = argc > 4 ? atoi(argv[4]) : 20;
  const float scale = 1.f / sqrtf(48.f);
  if (!tc5a::ready()) { fprintf(stderr, "no sm_100a tcgen05 here\n"); return 1; }
  CK(cudaStreamCreate(&STREAM));
  const size_t nq = rows * n * 4 * W + 128 * 4 * W, nb = (size_t)heads * n * stride;
  std::vector<half> hq(nq), hb(nb); std::vector<float> hqb(W);
  uint64_t s = 3;
  auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  for (auto& v : hq) v = __float2half(2.f * rnd());
  for (auto& v : hqb) v = 0.5f * rnd();
  for (size_t h = 0; h < (size_t)heads; ++h)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < stride; ++j) hb[(h * n + i) * stride + j] = __float2half(j < n ? 4.f * rnd() : 0.f);
  half *qkvg, *bias, *o1, *o2; float* qb;
  CK(cudaMalloc(&qkvg, nq * 2)); CK(cudaMalloc(&bias, nb * 2)); CK(cudaMalloc(&qb, W * 4));
  CK(cudaMalloc(&o1, rows * n * W * 2)); CK(cudaMalloc(&o2, rows * n * W * 2));
  CK(cudaMemcpy(qkvg, hq.data(), nq * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(bias, hb.data(), nb * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(qb, hqb.data(), W * 4, cudaMemcpyHostToDevice));
  auto mma = [&] { flashGridHalfLaunch<48>(qkvg, bias, stride, nullptr, o1, n, heads, 0, rows, false, scale, qb); };
  auto tc = [&] { tc5a::flashTcHRun(qkvg, bias, stride, o2, n, heads, rows, scale, qb); };
  mma(); tc(); CK(cudaStreamSynchronize(STREAM));
  std::vector<half> a(rows * n * W), b(rows * n * W);
  CK(cudaMemcpy(a.data(), o1, a.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), o2, b.size() * 2, cudaMemcpyDeviceToHost));
  double n1 = 0, n2 = 0, d12 = 0, den = 0;
  for (int t = 0; t < 512; ++t) {
    const size_t r = (size_t)(t * 2654435761u) % rows; const int i = (t * 40503) % n, h = t % heads;
    std::vector<double> sc(n); double mx = -1e30;
    for (int j = 0; j < n; ++j) {
      double acc = 0;
      for (int d = 0; d < D; ++d) acc += ((double)__half2float(hq[(r * n + i) * 4 * W + h * D + d]) + hqb[h * D + d]) * __half2float(hq[(r * n + j) * 4 * W + W + h * D + d]);
      sc[j] = acc * scale + (double)__half2float(hb[((size_t)h * n + i) * stride + j]) / LOG2E;
      mx = std::max(mx, sc[j]);
    }
    double sum = 0; for (int j = 0; j < n; ++j) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
    for (int d = 0; d < D; ++d) {
      double o = 0; for (int j = 0; j < n; ++j) o += sc[j] * __half2float(hq[(r * n + j) * 4 * W + 2 * W + h * D + d]);
      o /= sum * (1 + exp(-(double)__half2float(hq[(r * n + i) * 4 * W + 3 * W + h * D + d])));
      const size_t k = (r * n + i) * W + h * D + d;
      const double x = __half2float(a[k]), y = __half2float(b[k]);
      n1 += (x - o) * (x - o); n2 += (y - o) * (y - o); d12 += (x - y) * (x - y); den += o * o;
    }
  }
  size_t nonfin = 0; for (auto& v : b) nonfin += !std::isfinite(__half2float(v));
  printf("H n %d rows %zu: relRMS from the f32 reference: flashGridHalf %.3e, tcgen05 %.3e; between them %.3e; non-finite %zu\n",
         n, rows, sqrt(n1 / den), sqrt(n2 / den), sqrt(d12 / den), nonfin);
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const double flop = 4.0 * D * (double)n * n * rows * heads;
  std::vector<float> t1, t2;
  for (int k = 0; k < reps; ++k)
    for (int w = 0; w < 2; ++w) {
      CK(cudaEventRecord(e0, STREAM)); if (w) tc(); else mma(); CK(cudaEventRecord(e1, STREAM)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); (w ? t2 : t1).push_back(ms);
    }
#ifdef TC_PROF
  { unsigned long long hp[16][8]; CK(cudaMemcpyFromSymbol(hp, tc5a::tcProf, sizeof hp));
    for (int w = 0; w < 16; ++w) { if (!hp[w][7]) continue; printf("  warp %2d:", w); for (int k = 0; k < 8; ++k) printf(" %5.1f%%", 100.0 * hp[w][k] / hp[w][7]); printf("\n"); } }
#endif
  std::sort(t1.begin(), t1.end()); std::sort(t2.begin(), t2.end());
  printf("  median: flashGridHalf %.3f ms (%.0f TFLOP/s), tcgen05 %.3f ms (%.0f TFLOP/s)\n", t1[reps / 2], flop / t1[reps / 2] * 1e-9,
         t2[reps / 2], flop / t2[reps / 2] * 1e-9);
  return 0;
}
int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "h")) return mainH(argc, argv);
  const int n = argc > 1 ? atoi(argv[1]) : 1024, heads = 4, D = 32, W = heads * D, stride = (n + 7) / 8 * 8;
  const size_t rows = argc > 2 ? atoi(argv[2]) : 256;
  const int reps = argc > 3 ? atoi(argv[3]) : 20;
  const float skew = argc > 4 ? atof(argv[4]) : 1.f, scale = 0.17f;
  const int rpb = argc > 5 ? atoi(argv[5]) : 3;      // rows a block (flashTcRun's arm)
  if (!tc5a::ready()) { fprintf(stderr, "no sm_100a tcgen05 here\n"); return 1; }
  CK(cudaStreamCreate(&STREAM));
  const size_t nq = rows * n * 4 * W + 128 * 4 * W, nb = (size_t)heads * n * stride;
  std::vector<half> hq(nq), hb(nb);
  uint64_t s = 1;
  auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 40) / 16777216.f - 0.5f; };
  for (auto& v : hq) v = __float2half(2.f * rnd());
  for (size_t h = 0; h < (size_t)heads; ++h)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < stride; ++j) hb[(h * n + i) * stride + j] = __float2half(j < n ? skew * 4.f * rnd() : 0.f);
  half *qkvg, *bias, *o1, *o2; uint8_t* kv8;
  CK(cudaMalloc(&qkvg, nq * 2)); CK(cudaMalloc(&bias, nb * 2));
  CK(cudaMalloc(&o1, rows * n * W * 2)); CK(cudaMalloc(&o2, rows * n * W * 2)); CK(cudaMalloc(&kv8, rows * n * 2 * W));
  CK(cudaMemcpy(qkvg, hq.data(), nq * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(bias, hb.data(), nb * 2, cudaMemcpyHostToDevice));
  CK(cudaMemset(o1, 0, rows * n * W * 2)); CK(cudaMemset(o2, 0, rows * n * W * 2));
  kv8K<<<(unsigned)((rows * n * 2 * W + 255) / 256), 256, 0, STREAM>>>(qkvg, kv8, rows * n, W);
  auto f8 = [&] { tc5a::TC_ATTN = false; flash8Run(qkvg, kv8, kv8 + rows * n * W, bias, stride, o1, n, heads, rows, scale, nullptr); };
  // (fp8VtK as flash8Run runs it, then the tcgen05 kernel)
  uint8_t* vt8; const int NK = (n + 63) / 64 * 64; CK(cudaMalloc(&vt8, rows * (size_t)W * NK));
  CK(cudaFuncSetAttribute(fp8VtK, cudaFuncAttributeMaxDynamicSharedMemorySize, 65 * W));
  auto tc = [&] {
    fp8VtK<<<dim3(NK / 64, (unsigned)rows), 256, 65 * W, STREAM>>>(kv8 + rows * n * W, vt8, n, NK, W);
    tc5a::flashTcRun(qkvg, kv8, vt8, bias, stride, o2, n, heads, rows, scale, nullptr, rpb);
  };
  f8(); tc(); CK(cudaStreamSynchronize(STREAM));
  std::vector<half> a(rows * n * W), b(rows * n * W);
  CK(cudaMemcpy(a.data(), o1, a.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), o2, b.size() * 2, cudaMemcpyDeviceToHost));
  // the reference on the host, float32 from the same e4m3-rounded k and v and the f16 q, over a sample of (row, query)
  std::vector<uint8_t> hk(rows * n * 2 * W); CK(cudaMemcpy(hk.data(), kv8, hk.size(), cudaMemcpyDeviceToHost));
  auto e4 = [](uint8_t x) { __nv_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v) = x; return (float)v; };
  double n1 = 0, n2 = 0, d12 = 0, den = 0; size_t bad = 0;
  for (int t = 0; t < 512; ++t) {
    const size_t r = (size_t)(t * 2654435761u) % rows; const int i = (t * 40503) % n, h = t % heads;
    std::vector<double> sc(n); double mx = -1e30;
    for (int j = 0; j < n; ++j) {
      double acc = 0;
      for (int d = 0; d < D; ++d) acc += (double)__half2float(hq[(r * n + i) * 4 * W + h * D + d]) * e4(hk[(r * n + j) * W + h * D + d]);
      sc[j] = acc * scale + (double)__half2float(hb[((size_t)h * n + i) * stride + j]) / LOG2E;
      mx = std::max(mx, sc[j]);
    }
    double sum = 0; for (int j = 0; j < n; ++j) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
    for (int d = 0; d < D; ++d) {
      double o = 0; for (int j = 0; j < n; ++j) o += sc[j] * e4(hk[rows * n * W + (r * n + j) * W + h * D + d]);
      o /= sum;
      const double g = __half2float(hq[(r * n + i) * 4 * W + 3 * W + h * D + d]);
      o /= 1 + exp(-g);
      const size_t k = (r * n + i) * W + h * D + d;
      const double x = __half2float(a[k]), y = __half2float(b[k]);
      n1 += (x - o) * (x - o); n2 += (y - o) * (y - o); d12 += (x - y) * (x - y); den += o * o;
      bad += !std::isfinite(y);
    }
  }
  printf("n %d rows %zu skew %.1f: relRMS from the f32 reference: flash8 %.3e, tcgen05 %.3e; between them %.3e; non-finite %zu\n",
         n, rows, skew, sqrt(n1 / den), sqrt(n2 / den), sqrt(d12 / den), bad);
  size_t nonfin = 0; for (auto& v : b) nonfin += !std::isfinite(__half2float(v));
  printf("  tcgen05 non-finite outputs over all: %zu of %zu\n", nonfin, b.size());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const double flop = 4.0 * D * (double)n * n * rows * heads;
  std::vector<float> t1, t2;
  for (int k = 0; k < reps; ++k) {
    for (int w = 0; w < 2; ++w) {
      CK(cudaEventRecord(e0, STREAM)); if (w) tc(); else f8(); CK(cudaEventRecord(e1, STREAM)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); (w ? t2 : t1).push_back(ms);
    }
  }
#ifdef TC_PROF
  { unsigned long long hp[16][8]; CK(cudaMemcpyFromSymbol(hp, tc5a::tcProf, sizeof hp));
    for (int w = 0; w < 16; ++w) { if (!hp[w][7]) continue; printf("  warp %2d:", w); for (int k = 0; k < 8; ++k) printf(" %5.1f%%", 100.0 * hp[w][k] / hp[w][7]); printf("\n"); } }
#endif
  std::sort(t1.begin(), t1.end()); std::sort(t2.begin(), t2.end());
  printf("  median: flash8 %.3f ms (%.0f TFLOP/s), tcgen05 %.3f ms (%.0f TFLOP/s)\n", t1[reps / 2], flop / t1[reps / 2] * 1e-9,
         t2[reps / 2], flop / t2[reps / 2] * 1e-9);
  return 0;
}
