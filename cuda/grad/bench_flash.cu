// sdpaForward / sdpaBackward at AF2's training shapes (L 256, 128 MSA rows): the flash kernels against the bmm path.
//   cuda/grad/bench-flash [--tf32]
#include <cstring>
#include <curand.h>
#include "ops.cuh"
using namespace grad;
int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "--tf32")) TF32 = true; else if (!strcmp(argv[i], "--bf16")) BF16 = true;
  init();
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 1);
  auto randn = [&](size_t n) { float* p; GK(cudaMalloc(&p, ((n + 1) & ~1) * 4)); curandGenerateNormal(gen, p, (n + 1) & ~1, 0.f, 1.f); return p; };
  struct Shape { const char* name; int B, H, Q, D; bool bias; };
  for (Shape c : {Shape{"row", 128, 8, 256, 32, true}, Shape{"col", 256, 8, 128, 32, false}, Shape{"tri", 256, 4, 256, 32, true}, Shape{"xrow", 1024, 8, 256, 8, true}}) {
    int HD = c.H * c.D; size_t n = (size_t)c.B * c.Q * HD;
    float *q = randn(n), *k = randn(n), *v = randn(n), *dO = randn(n), *o = randn(n), *dq = randn(n), *dk = randn(n), *dv = randn(n);
    float *lse; GK(cudaMalloc(&lse, (size_t)2 * c.B * c.H * c.Q * 4));
    float* m; GK(cudaMalloc(&m, (size_t)c.B * c.Q * 4)); std::vector<float> ones((size_t)c.B * c.Q, 1.f);
    GK(cudaMemcpy(m, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
    size_t nb = (size_t)c.H * c.Q * c.Q; float *b = randn(nb), *db = randn(nb);
    Sdpa a; a.B = c.B; a.H = c.H; a.Q = a.K = c.Q; a.D = c.D; a.scale = 1.f / sqrtf((float)c.D); a.keyMask = m; a.clip = true;
    if (c.bias) a.bias[0] = Bias{b, 0, (long long)c.Q * c.Q, c.Q, db};
    cudaEvent_t e[3]; for (auto& x : e) cudaEventCreate(&x);
    for (int f = 0; f < 2; ++f) {
      FLASH = f;
      float tf = 1e9, tb = 1e9;
      for (int r = 0; r < 5; ++r) {
        ARENA.reset();
        cudaEventRecord(e[0], STREAM);
        sdpaForward(a, q, HD, k, HD, v, HD, o, HD, lse);
        cudaEventRecord(e[1], STREAM);
        sdpaBackward(a, q, HD, k, HD, v, HD, dO, HD, dq, HD, dk, HD, dv, HD, false, false, false, f ? o : nullptr, HD, f ? lse : nullptr);
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
