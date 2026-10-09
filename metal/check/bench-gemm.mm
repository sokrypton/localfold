// The runtime's GEMM (metal/shim/shim.metal) on its own: the shapes a fold runs, timed, and checked against the
// host. A development tool for tuning the tiles.
//   metal/check/bench-gemm [m,n,k,ta,tb ...]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include "../shim/lfcuda.h"
namespace lf { extern const char* SHIM_SOURCE; void profileStart(); void profileStop(int); }
int main(int argc, char** argv) { @autoreleasepool {
  struct Case { int m, n, k, ta, tb; };
  std::vector<Case> cases = {{3072, 80, 768, 0, 0}, {768, 80, 1536, 0, 0}, {73728, 68, 392, 0, 0}, {768, 80, 768, 0, 0},
                             {512, 4624, 128, 0, 0}, {128, 4624, 128, 0, 0}, {72, 72, 72, 1, 0}, {4096, 4096, 4096, 0, 0}};
  for (int i = 1; i < argc; ++i) { Case c; sscanf(argv[i], "%d,%d,%d,%d,%d", &c.m, &c.n, &c.k, &c.ta, &c.tb); if (i == 1) cases.clear(); cases.push_back(c); }
  std::mt19937 rng(1);
  for (auto c : cases) {
    size_t na = (size_t)c.m * c.k, nb = (size_t)c.k * c.n, nc = (size_t)c.m * c.n;
    std::vector<__half> A(na), B(nb);
    for (auto& x : A) x = __half(std::uniform_real_distribution<float>(-1, 1)(rng));
    for (auto& x : B) x = __half(std::uniform_real_distribution<float>(-1, 1)(rng));
    __half *dA, *dB, *dC;
    cudaMalloc(&dA, na * 2); cudaMalloc(&dB, nb * 2); cudaMalloc(&dC, nc * 2);
    cudaMemcpy(dA, A.data(), na * 2, cudaMemcpyHostToDevice); cudaMemcpy(dB, B.data(), nb * 2, cudaMemcpyHostToDevice);
    cublasHandle_t h; cublasCreate(&h);
    float one = 1, zero = 0;
    int lda = c.ta ? c.k : c.m, ldb = c.tb ? c.n : c.k;
    auto run = [&] { cublasGemmEx(h, c.ta ? CUBLAS_OP_T : CUBLAS_OP_N, c.tb ? CUBLAS_OP_T : CUBLAS_OP_N, c.m, c.n, c.k, &one, dA, CUDA_R_16F, lda,
                                  dB, CUDA_R_16F, ldb, &zero, dC, CUDA_R_16F, c.m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); };
    run(); cudaDeviceSynchronize();
    std::vector<__half> C(nc); cudaMemcpy(C.data(), dC, nc * 2, cudaMemcpyDeviceToHost);
    double err = 0, nrm = 0;
    for (int s = 0; s < 64; ++s) {
      int i = (int)(rng() % c.m), j = (int)(rng() % c.n); double want = 0;
      for (int k = 0; k < c.k; ++k)
        want += (double)(float)(c.ta ? A[k + (size_t)i * lda] : A[i + (size_t)k * lda]) * (double)(float)(c.tb ? B[j + (size_t)k * ldb] : B[k + (size_t)j * ldb]);
      double got = (float)C[i + (size_t)j * c.m]; err += (got - want) * (got - want); nrm += want * want;
    }
    int reps = std::max(3, (int)(2e9 / (2.0 * c.m * c.n * c.k)));
    reps = std::min(reps, 200);
    if (getenv("GPU_TIME")) lf::profileStart();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) run();
    if (getenv("GPU_TIME")) lf::profileStop(3);
    cudaDeviceSynchronize();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    printf("%6d x %6d x %5d  %s%s  %8.3f ms  %5.2f TFLOP/s  relRMS %.1e\n", c.m, c.n, c.k, c.ta ? "T" : "N", c.tb ? "T" : "N",
           ms, 2.0 * c.m * c.n * c.k / (ms * 1e-3) / 1e12, std::sqrt(err / nrm));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }
}}
