#include <cstring>
// flash.cuh against the bmm path it replaces: sdpaForward and sdpaBackward on random inputs, every mask and bias form,
// both precisions (3xTF32 and TF32), shapes that do and do not fill a 64-row tile.
//   cuda/grad/test-flash
#include <curand.h>
#include "ops.cuh"
using namespace grad;
static curandGenerator_t gen;
static float* randn(size_t n, float sd = 1.f) { float* p; GK(cudaMalloc(&p, ((n + 1) & ~1) * 4)); curandGenerateNormal(gen, p, (n + 1) & ~1, 0.f, sd); return p; }
static float* zeros(size_t n) { float* p; GK(cudaMalloc(&p, n * 4 + 16)); GK(cudaMemset(p, 0, n * 4 + 16)); return p; }
static std::vector<float> hostv(const float* d, size_t n) { std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h; }
static double rel(const float* a, const float* b, size_t n) {
  auto x = hostv(a, n), y = hostv(b, n); double e = 0, m = 0;
  for (size_t i = 0; i < n; ++i) { e = std::max(e, (double)fabs(x[i] - y[i])); m = std::max(m, (double)fabs(y[i])); }
  return e / (m + 1e-30);
}
static double relRmsHost(const float* a, const float* b, size_t n) {
  auto x = hostv(a, n), y = hostv(b, n); double e = 0, m = 0;
  for (size_t i = 0; i < n; ++i) { e += (double)(x[i] - y[i]) * (x[i] - y[i]); m += (double)y[i] * y[i]; }
  return sqrt(e / (m + 1e-30));
}
int main() {
  init(); curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 3);
  struct Case { int B, H, Q, K, D; bool mask, bias, bias2, clip; float sd = 1.f; bool fullMask = false; };
  std::vector<Case> cases = {{48, 8, 7, 7, 32, true, false, false, true}, {7, 8, 48, 48, 32, true, true, false, true},
                             {48, 4, 48, 48, 32, true, true, false, true}, {3, 2, 130, 70, 16, false, true, true, false},
                             {5, 3, 64, 64, 8, true, false, false, false}, {2, 2, 100, 129, 64, true, true, false, true},
                             {4, 4, 33, 65, 24, true, true, true, true}, {2, 4, 77, 77, 48, false, false, false, false},
                             {7, 8, 48, 48, 32, true, true, false, true, 6.f, true},     // large logits, a row masked whole
                             {5, 8, 37, 37, 32, true, true, false, true, 1.f, true}};
  int fails = 0;
  for (int mode : {0, 1, 2}) {           // float32 (3xTF32), TF32, bf16
    const bool tf = mode > 0;
    setTf32(mode == 1); setBf16(mode == 2);
    int ci = -1;
    for (auto c : cases) {
      ++ci;
      int HD = c.H * c.D, ld = HD + 8;    // a leading dimension past the heads, as a packed qkv's
      size_t nq = (size_t)c.B * c.Q * ld, nk = (size_t)c.B * c.K * ld;
      float *q = randn(nq, c.sd), *k = randn(nk, c.sd), *v = randn(nk), *dO = randn(nq);
      Sdpa a; a.B = c.B; a.H = c.H; a.Q = c.Q; a.K = c.K; a.D = c.D; a.scale = 1.f / sqrtf((float)c.D); a.clip = c.clip;
      float* m = nullptr;
      if (c.mask) { std::vector<float> h((size_t)c.B * c.K); for (size_t i = 0; i < h.size(); ++i) h[i] = (i * 7 % 11) && !(c.fullMask && i / c.K == 1) ? 1.f : 0.f;
                    m = zeros(h.size()); GK(cudaMemcpy(m, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
      a.keyMask = m;
      size_t nb = (size_t)c.H * c.Q * c.K, nb2 = (size_t)c.B * c.K;
      float *b0 = c.bias ? randn(nb) : nullptr, *b1 = c.bias2 ? randn(nb2) : nullptr;
      float *gb[2][2] = {{zeros(nb), zeros(nb2)}, {zeros(nb), zeros(nb2)}};
      float *o[2], *dq[2], *dk[2], *dv[2];
      for (int f = 0; f < 2; ++f) {
        FLASH = f == 1;
        // bf16 against the float32 bmm path: against its own bf16 bmm the reference's rounding is most of the
        // difference, and a dK 1% wrong read the same as a right one
        if (mode == 2) setBf16(f == 1);
        a.bias[0] = c.bias ? Bias{b0, 0, (long long)c.Q * c.K, c.K, gb[f][0]} : Bias{};
        a.bias[1] = c.bias2 ? Bias{b1, (long long)c.K, 0, 0, gb[f][1]} : Bias{};
        o[f] = zeros(nq); dq[f] = randn(nq); dk[f] = zeros(nk); dv[f] = zeros(nk);
        float* lse = zeros((size_t)2 * c.B * c.H * c.Q);
        if (mode == 2 && f == 1 && ci % 2) {
          // every other case with q, k, v already bf16 (a module on bf16 storage): read in place at ld, zero past D
          const bf *qb = toBf(q, nq), *kb = toBf(k, nk), *vb = toBf(v, nk);
          sdpaForwardBf(a, qb, ld, kb, ld, vb, ld, o[f], ld, lse);
          sdpaBackwardBf(a, qb, ld, kb, ld, vb, ld, dO, ld, dq[f], ld, dk[f], ld, dv[f], ld, o[f], ld, lse);
        } else {
          sdpaForward(a, q, ld, k, ld, v, ld, o[f], ld, lse);
          sdpaBackward(a, q, ld, k, ld, v, ld, dO, ld, dq[f], ld, dk[f], ld, dv[f], ld, false, false, false,
                       f ? o[f] : nullptr, ld, f ? lse : nullptr);
        }
        GK(cudaStreamSynchronize(STREAM)); GK(cudaGetLastError());
      }
      // compare only the head columns (the pad columns are never written by either)
      auto cmp = [&](const char* n, float* x, float* y, size_t rows) {
        auto hx = hostv(x, rows * ld), hy = hostv(y, rows * ld); double e = 0, mm = 0, se = 0, sy = 0;
        for (size_t r = 0; r < rows; ++r) for (int j = 0; j < HD; ++j) {
          const double d = hx[r * ld + j] - hy[r * ld + j];
          e = std::max(e, fabs(d)); mm = std::max(mm, (double)fabs(hy[r * ld + j])); se += d * d; sy += (double)hy[r * ld + j] * hy[r * ld + j];
        }
        // bf16 by relative rms: its rounding is ~0.5% of the largest value, which a max-abs bar cannot tell from a
        // 1% defect; spread over every element it is ~0.3%, and a 1% defect is 1%
        double rr = mode == 2 ? sqrt(se / (sy + 1e-30)) : e / (mm + 1e-30);
        double bar = mode == 2 ? (c.sd > 1 ? 0.15 : 6e-3) : tf ? 5e-3 : c.sd > 1 ? 3e-5 : 2e-6;   // (logits ~ +-36: float32's own conditioning)
        printf("  %-4s %.2e%s", n, rr, rr > bar ? " <-- ABOVE" : ""); fails += rr > bar;
      };
      printf("%s B %d H %d Q %d K %d D %d%s%s%s%s:", mode == 2 ? (ci % 2 ? "bf16 in" : "bf16") : tf ? "tf32" : "fp32", c.B, c.H, c.Q, c.K, c.D, c.mask ? " mask" : "", c.bias ? " bias" : "", c.bias2 ? " bias2" : "", c.clip ? " clip" : "");
      cmp("o", o[1], o[0], (size_t)c.B * c.Q); cmp("dq", dq[1], dq[0], (size_t)c.B * c.Q); cmp("dk", dk[1], dk[0], (size_t)c.B * c.K); cmp("dv", dv[1], dv[0], (size_t)c.B * c.K);
      const double bb = mode == 2 ? (c.sd > 1 ? 0.15 : 6e-3) : tf ? 5e-3 : c.sd > 1 ? 3e-5 : 2e-6;
      if (c.bias) { double r = (mode == 2 ? relRmsHost : rel)(gb[1][0], gb[0][0], nb); printf("  db %.2e%s", r, r > bb ? " <-- ABOVE" : ""); fails += r > bb; }
      if (c.bias2) { double r = (mode == 2 ? relRmsHost : rel)(gb[1][1], gb[0][1], nb2); printf("  db2 %.2e%s", r, r > bb ? " <-- ABOVE" : ""); fails += r > bb; }
      printf("\n");
    }
  }
  // with Sdpa::deterministic the bf16 backward's bias gradient is summed in fixed point: bitwise the same on every
  // run (float atomics are not)
  {
    setTf32(false); setBf16(true); FLASH = true;
    const int B = 64, H = 4, Q = 96, D = 32, HD = H * D;
    const size_t n = (size_t)B * Q * HD;
    float *q = randn(n, 1.f), *k = randn(n, 1.f), *v = randn(n, 1.f), *dO = randn(n, 1.f), *bb = randn((size_t)H * Q * Q, 1.f);
    float *o = zeros(n), *dq = zeros(n), *dk = zeros(n), *dv = zeros(n), *lse = zeros((size_t)2 * B * H * Q);
    std::vector<float> g[2];
    for (int r = 0; r < 2; ++r) {
      float* db = zeros((size_t)H * Q * Q);
      Sdpa a; a.B = B; a.H = H; a.Q = a.K = Q; a.D = D; a.scale = 1.f / sqrtf((float)D); a.deterministic = true;
      a.bias[0] = Bias{bb, 0, (long long)Q * Q, Q, db};
      sdpaForward(a, q, HD, k, HD, v, HD, o, HD, lse);
      sdpaBackward(a, q, HD, k, HD, v, HD, dO, HD, dq, HD, dk, HD, dv, HD, false, false, false, o, HD, lse);
      GK(cudaStreamSynchronize(STREAM));
      g[r] = hostv(db, (size_t)H * Q * Q);
    }
    const bool same = !memcmp(g[0].data(), g[1].data(), g[0].size() * 4);
    printf("bf16 bias gradient over %d batch rows, twice: %s\n", B, same ? "bitwise identical" : "DIFFERS  <-- ABOVE");
    fails += !same;
    setBf16(false);
  }
  // both paths against a float64 forward on the host: which is closer to the truth, not only to each other
  {
    setTf32(false); setBf16(false);
    const int B = 4, H = 4, Q = 96, K = 96, D = 32, HD = H * D;
    std::vector<float> hq((size_t)B * Q * HD), hk((size_t)B * K * HD), hv((size_t)B * K * HD), hb((size_t)H * Q * K);
    unsigned st = 5; auto rnd = [&] { st = st * 1664525u + 1013904223u; return ((st >> 8) / 16777216.0f - 0.5f) * 4.f; };
    for (auto* v : {&hq, &hk, &hv, &hb}) for (auto& x : *v) x = rnd();
    float *q, *k, *v, *bb, *o0, *o1, *lse;
    for (auto pr : {std::make_pair(&q, &hq), std::make_pair(&k, &hk), std::make_pair(&v, &hv), std::make_pair(&bb, &hb)}) {
      GK(cudaMalloc(pr.first, pr.second->size() * 4)); GK(cudaMemcpy(*pr.first, pr.second->data(), pr.second->size() * 4, cudaMemcpyHostToDevice)); }
    GK(cudaMalloc(&o0, hq.size() * 4)); GK(cudaMalloc(&o1, hq.size() * 4)); GK(cudaMalloc(&lse, (size_t)2 * B * H * Q * 4));
    Sdpa a; a.B = B; a.H = H; a.Q = Q; a.K = K; a.D = D; a.scale = 1.f / sqrtf((float)D); a.clip = true;
    a.bias[0] = Bias{bb, 0, (long long)Q * K, K, nullptr};
    FLASH = false; sdpaForward(a, q, HD, k, HD, v, HD, o0, HD, lse);
    FLASH = true; sdpaForward(a, q, HD, k, HD, v, HD, o1, HD, lse);
    auto r0 = hostv(o0, hq.size()), r1 = hostv(o1, hq.size());
    double e0 = 0, e1 = 0, mx = 0;
    std::vector<double> lg(K);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) for (int i = 0; i < Q; ++i) {
      double m = -1e300, sum = 0;
      for (int j = 0; j < K; ++j) {
        double d = 0; for (int c = 0; c < D; ++c) d += (double)hq[((size_t)b * Q + i) * HD + h * D + c] * hk[((size_t)b * K + j) * HD + h * D + c];
        lg[j] = d / sqrt((double)D) + hb[((size_t)h * Q + i) * K + j]; m = std::max(m, lg[j]); }
      for (int j = 0; j < K; ++j) { lg[j] = exp(lg[j] - m); sum += lg[j]; }
      for (int c = 0; c < D; ++c) {
        double t = 0; for (int j = 0; j < K; ++j) t += lg[j] * hv[((size_t)b * K + j) * HD + h * D + c];
        t /= sum; size_t ix = ((size_t)b * Q + i) * HD + h * D + c;
        e0 = std::max(e0, fabs(r0[ix] - t)); e1 = std::max(e1, fabs(r1[ix] - t)); mx = std::max(mx, fabs(t));
      }
    }
    printf("against float64 (logits ~ +-10): bmm path %.2e, flash %.2e (max abs, relative to max |o|)\n", e0 / mx, e1 / mx);
    fails += e1 > 3 * e0 + 1e-7;
  }
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
