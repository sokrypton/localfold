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
  for (bool tf : {false, true}) {
    setTf32(tf);
    for (auto c : cases) {
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
        a.bias[0] = c.bias ? Bias{b0, 0, (long long)c.Q * c.K, c.K, gb[f][0]} : Bias{};
        a.bias[1] = c.bias2 ? Bias{b1, (long long)c.K, 0, 0, gb[f][1]} : Bias{};
        o[f] = zeros(nq); dq[f] = randn(nq); dk[f] = zeros(nk); dv[f] = zeros(nk);
        float* lse = zeros((size_t)2 * c.B * c.H * c.Q);
        sdpaForward(a, q, ld, k, ld, v, ld, o[f], ld, lse);
        GK(cudaMemcpy(dk[f], dq[f], 0, cudaMemcpyDeviceToDevice));
        sdpaBackward(a, q, ld, k, ld, v, ld, dO, ld, dq[f], ld, dk[f], ld, dv[f], ld, false, false, false,
                     f ? o[f] : nullptr, ld, f ? lse : nullptr);
        GK(cudaStreamSynchronize(STREAM)); GK(cudaGetLastError());
      }
      // compare only the head columns (the pad columns are never written by either)
      auto cmp = [&](const char* n, float* x, float* y, size_t rows) {
        auto hx = hostv(x, rows * ld), hy = hostv(y, rows * ld); double e = 0, mm = 0;
        for (size_t r = 0; r < rows; ++r) for (int j = 0; j < HD; ++j) { e = std::max(e, (double)fabs(hx[r * ld + j] - hy[r * ld + j])); mm = std::max(mm, (double)fabs(hy[r * ld + j])); }
        double rr = e / (mm + 1e-30), bar = tf ? 5e-3 : c.sd > 1 ? 3e-5 : 2e-6;   // (logits ~ +-36: float32's own conditioning)
        printf("  %-4s %.2e%s", n, rr, rr > bar ? " <-- ABOVE" : ""); fails += rr > bar;
      };
      printf("%s B %d H %d Q %d K %d D %d%s%s%s%s:", tf ? "tf32" : "fp32", c.B, c.H, c.Q, c.K, c.D, c.mask ? " mask" : "", c.bias ? " bias" : "", c.bias2 ? " bias2" : "", c.clip ? " clip" : "");
      cmp("o", o[1], o[0], (size_t)c.B * c.Q); cmp("dq", dq[1], dq[0], (size_t)c.B * c.Q); cmp("dk", dk[1], dk[0], (size_t)c.B * c.K); cmp("dv", dv[1], dv[0], (size_t)c.B * c.K);
      const double bb = tf ? 5e-3 : c.sd > 1 ? 3e-5 : 2e-6;
      if (c.bias) { double r = rel(gb[1][0], gb[0][0], nb); printf("  db %.2e%s", r, r > bb ? " <-- ABOVE" : ""); fails += r > bb; }
      if (c.bias2) { double r = rel(gb[1][1], gb[0][1], nb2); printf("  db2 %.2e%s", r, r > bb ? " <-- ABOVE" : ""); fails += r > bb; }
      printf("\n");
    }
  }
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
