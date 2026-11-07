// sdpaForward / sdpaBackward at AF2's training shapes (L 256, 128 MSA rows): the flash kernels against the bmm path.
//   cuda/grad/bench-flash [--tf32]
#include <cstring>
#include <curand.h>
#include "ops.cuh"
using namespace grad;
int main(int argc, char** argv) {
  bool noDb = false, bfIn = false, gridFwd = false, tc = false;   // (--tc: the grid forward on tcgen05 where the binary has sm_100a code)
  const char* only = nullptr;   // --shape=<name>: that shape alone   // --bf16-in: q, k, v bf16 in a packed [rows, 4 HD] (what the modules hand it)   // --no-db: the bias without its gradient (the atomics' share of the backward)
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--tf32")) TF32 = true; else if (!strcmp(argv[i], "--bf16")) BF16 = true; else if (!strcmp(argv[i], "--no-db")) noDb = true; else if (!strcmp(argv[i], "--bf16-in")) bfIn = BF16 = true; else if (!strncmp(argv[i], "--shape=", 8)) only = argv[i] + 8; else if (!strcmp(argv[i], "--grid")) gridFwd = bfIn = BF16 = true; else if (!strcmp(argv[i], "--tc")) tc = true;   // --grid: the forward by flash::gridForwardBf, the triangle attention's (bf16 in)
  init();
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 1);
  auto randn = [&](size_t n) { float* p; GK(cudaMalloc(&p, ((n + 1) & ~1) * 4)); curandGenerateNormal(gen, p, (n + 1) & ~1, 0.f, 1.f); return p; };
  struct Shape { const char* name; int B, H, Q, D; bool bias; };
  for (Shape c : {Shape{"row", 128, 8, 256, 32, true}, Shape{"col", 256, 8, 128, 32, false}, Shape{"tri", 256, 4, 256, 32, true}, Shape{"tri300", 300, 4, 300, 32, true}, Shape{"xrow", 1024, 8, 256, 8, true}, Shape{"af3tri", 384, 4, 384, 32, true}, Shape{"af3tri768", 768, 4, 768, 32, true}, Shape{"diff512", 26, 16, 512, 48, true}, Shape{"grid512", 512, 4, 512, 32, true}, Shape{"grid768", 768, 4, 768, 32, true}}) {
    if (only && strcmp(only, c.name)) continue;
    int HD = c.H * c.D; size_t n = (size_t)c.B * c.Q * HD;
    float *q = randn(n), *k = randn(n), *v = randn(n), *dO = randn(n), *o = randn(n), *dq = randn(n), *dk = randn(n), *dv = randn(n);
    float *lse; GK(cudaMalloc(&lse, (size_t)2 * c.B * c.H * c.Q * 4));
    float* m; GK(cudaMalloc(&m, (size_t)c.B * c.Q * 4)); std::vector<float> ones((size_t)c.B * c.Q, 1.f);
    GK(cudaMemcpy(m, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
    size_t nb = (size_t)c.H * c.Q * c.Q; float *b = randn(nb), *db = randn(nb);
    Sdpa a; a.B = c.B; a.H = c.H; a.Q = a.K = c.Q; a.D = c.D; a.scale = 1.f / sqrtf((float)c.D); a.keyMask = m; a.clip = strncmp(c.name, "diff", 4) && strncmp(c.name, "grid", 4);
    if (c.bias) a.bias[0] = Bias{b, 0, (long long)c.Q * c.Q, c.Q, noDb ? nullptr : db};
    cudaEvent_t e[3]; for (auto& x : e) cudaEventCreate(&x);
    for (int f = 0; f < 2; ++f) {
      FLASH = f;
      float tf = 1e9, tb = 1e9;
      for (int r = 0; r < 5; ++r) {
        ARENA.reset();
        cudaEventRecord(e[0], STREAM);
        if (bfIn && f) {
          static bf* pk = nullptr; static bf* dpk = nullptr; static size_t cap = 0;
          if (cap < n) {   // (this shape's packed buffers: a later shape is larger)
            if (pk) { GK(cudaFree(pk)); GK(cudaFree(dpk)); }
            GK(cudaMalloc(&pk, n * 4 * 2)); GK(cudaMalloc(&dpk, n * 4 * 2)); cap = n;
            float* r4 = randn(n * 4);   // (random, not zero: zero v makes every dS 0, and the bias atomics vanish)
            toBf(r4, pk, n * 4); GK(cudaStreamSynchronize(STREAM)); GK(cudaFree(r4));
          }
          const int P = 4 * HD;
          if (!gridFwd || !flash::gridForwardBf(a, pk, P, pk + HD, P, pk + 2 * HD, P, dpk, HD, lse, nullptr, 0, nullptr, 0, tc))
            sdpaForwardBf(a, pk, P, pk + HD, P, pk + 2 * HD, P, o, HD, lse);
          cudaEventRecord(e[1], STREAM);
          sdpaBackwardBf(a, pk, P, pk + HD, P, pk + 2 * HD, P, dO, HD, dq, HD, dpk + HD, P, dpk + 2 * HD, P, o, HD, lse);
        } else {
        sdpaForward(a, q, HD, k, HD, v, HD, o, HD, lse);
        cudaEventRecord(e[1], STREAM);
        sdpaBackward(a, q, HD, k, HD, v, HD, dO, HD, dq, HD, dk, HD, dv, HD, false, false, false, f ? o : nullptr, HD, f ? lse : nullptr);
        }
        cudaEventRecord(e[2], STREAM);
        GK(cudaStreamSynchronize(STREAM));
        float x, y; cudaEventElapsedTime(&x, e[0], e[1]); cudaEventElapsedTime(&y, e[1], e[2]);
        tf = std::min(tf, x); tb = std::min(tb, y);
      }
      printf("%s B %d H %d Q=K %d D %d%s %s: forward %.2f ms, backward %.2f ms\n", c.name, c.B, c.H, c.Q, c.D,
             c.bias ? " +bias" : "", f ? "flash" : "bmm  ", tf, tb);
    }
  }
}
