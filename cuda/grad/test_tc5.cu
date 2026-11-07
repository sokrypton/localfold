// The B200's tcgen05 tensor-core path, at its smallest: one 128 x N tile of C = A B^T (bf16 in, fp32 accumulated in
// tensor memory), K a multiple of 16, both operands K-major in shared memory without swizzle - the instruction and
// shared-memory descriptors, the tensor-memory allocation, the commit to an mbarrier and the load back, checked against
// the host. sm_100a only:
//
//   nvcc -std=c++17 -arch=sm_100a -O2 cuda/grad/test_tc5.cu -o test-tc5 && ./test-tc5
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

namespace tc5 {
__device__ __forceinline__ uint32_t smemAddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// the instruction descriptor (kind::f16): bf16 A and B, K-major both, f32 accumulator; M 64 or 128, N a multiple of 16
__host__ __device__ constexpr uint32_t idescBf16(int M, int N) {
  return (1u << 4)                     // c_format F32
       | (1u << 7) | (1u << 10)        // a_format, b_format BF16
       | ((uint32_t)(N >> 3) << 17)    // n_dim
       | ((uint32_t)(M >> 4) << 24);   // m_dim
}
// a K-major operand without swizzle: core matrices of 8 rows x 16 bytes, 128 bytes each; lbo the bytes between the two
// core matrices a 16-wide K step spans, sbo the bytes between core matrices 8 rows apart
__device__ __forceinline__ uint64_t sdescNoSwizzle(const void* p, uint32_t lbo, uint32_t sbo) {
  const uint64_t a = smemAddr(p);
  return ((a >> 4) & 0x3fff) | ((uint64_t)((lbo >> 4) & 0x3fff) << 16) | ((uint64_t)((sbo >> 4) & 0x3fff) << 32) |
         (1ull << 46);   // version 1 (Blackwell); base offset 0, lbo mode 0, layout SWIZZLE_NONE (0)
}
__device__ __forceinline__ void mma(uint32_t tmemC, uint64_t a, uint64_t b, uint32_t idesc, bool accumulate) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
               :: "r"(tmemC), "l"(a), "l"(b), "r"(idesc), "r"((uint32_t)accumulate));
}
__device__ __forceinline__ void commit(uint64_t* mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smemAddr(mbar)) : "memory");
}
__device__ __forceinline__ void mbarInit(uint64_t* m, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smemAddr(m)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbarWait(uint64_t* m, uint32_t parity) {
  asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra W;\n\t}\n"
               :: "r"(smemAddr(m)), "r"(parity) : "memory");
}
}  // namespace tc5

// one CTA, 128 threads: C[128, N] = A[128, K] B[N, K]^T
template <int N, int K>
__global__ void tileK(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C) {
  __shared__ __align__(1024) __nv_bfloat16 As[128 * K], Bs[N * K];
  __shared__ __align__(8) uint64_t mbar;
  __shared__ uint32_t tmemBase;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // the operands into the core-matrix layout: element (r, k) at core matrix (r / 8, k / 8), row r % 8, column k % 8,
  // core matrices K-adjacent first ((r / 8) (K / 8) + k / 8) - so lbo 128 bytes, sbo (K / 8) 128
  auto place = [&](__nv_bfloat16* s, const __nv_bfloat16* g, int rows) {
    for (int i = tid; i < rows * K; i += blockDim.x) {
      const int r = i / K, k = i % K;
      s[((r / 8) * (K / 8) + k / 8) * 64 + (r % 8) * 8 + k % 8] = g[i];
    }
  };
  place(As, A, 128); place(Bs, B, N);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // (the generic proxy's stores, seen by the async proxy)
  if (warp == 0) {
    constexpr uint32_t cols = N < 32 ? 32 : N;   // (a power of two, at least 32)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(tc5::smemAddr(&tmemBase)), "r"(cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) tc5::mbarInit(&mbar, 1);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tm = tmemBase;
  if (tid == 0) {
    constexpr uint32_t idesc = tc5::idescBf16(128, N);
    for (int k = 0; k < K / 16; ++k) {   // (each instruction a K of 16: two core matrices along K, 256 bytes on)
      const uint64_t a = tc5::sdescNoSwizzle(As + k * 128, 128, (K / 8) * 128);
      const uint64_t b = tc5::sdescNoSwizzle(Bs + k * 128, 128, (K / 8) * 128);
      tc5::mma(tm, a, b, idesc, k > 0);
    }
    tc5::commit(&mbar);
  }
  tc5::mbarWait(&mbar, 0);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  // each warp its 32 lanes (rows 32 warp ..), 8 columns a load
  for (int c0 = 0; c0 < N; c0 += 8) {
    uint32_t r[8];
    const uint32_t addr = tm + ((uint32_t)(warp * 32) << 16) + c0;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    for (int j = 0; j < 8; ++j) C[(warp * 32 + lane) * N + c0 + j] = __uint_as_float(r[j]);
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 0) {
    constexpr uint32_t cols = N < 32 ? 32 : N;
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tm), "r"(cols));
  }
}

template <int N, int K>
bool check() {
  std::vector<__nv_bfloat16> a(128 * K), b(N * K);
  std::vector<float> af(a.size()), bf(b.size()), c(128 * N);
  srand(N * 1000 + K);
  for (size_t i = 0; i < a.size(); ++i) { a[i] = __float2bfloat16(rand() / (float)RAND_MAX - 0.5f); af[i] = __bfloat162float(a[i]); }
  for (size_t i = 0; i < b.size(); ++i) { b[i] = __float2bfloat16(rand() / (float)RAND_MAX - 0.5f); bf[i] = __bfloat162float(b[i]); }
  __nv_bfloat16 *dA, *dB; float* dC;
  CK(cudaMalloc(&dA, a.size() * 2)); CK(cudaMalloc(&dB, b.size() * 2)); CK(cudaMalloc(&dC, c.size() * 4));
  CK(cudaMemcpy(dA, a.data(), a.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
  tileK<N, K><<<1, 128>>>(dA, dB, dC);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(c.data(), dC, c.size() * 4, cudaMemcpyDeviceToHost));
  double worst = 0, mx = 0;
  for (int m = 0; m < 128; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0; for (int k = 0; k < K; ++k) s += (double)af[m * K + k] * bf[n * K + k];
      worst = std::max(worst, std::fabs(s - c[m * N + n])); mx = std::max(mx, std::fabs(s));
    }
  const bool ok = worst <= 1e-4 * std::max(1.0, mx);
  printf("tcgen05 128 x %d x %d: max |error| %.2e (max |c| %.2f)  %s\n", N, K, worst, mx, ok ? "PASS" : "FAIL");
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
  return ok;
}

int main() {
  bool ok = check<32, 16>();
  ok &= check<32, 32>();
  ok &= check<64, 32>();
  ok &= check<128, 64>();
  ok &= check<256, 32>();
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
