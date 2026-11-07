// cuda/grad: the shared reverse-mode core every model's TRAINING path is built on - parameters with gradients,
// a per-block activation arena, and the GEMMs a Linear's forward and two backward products need.
//
// Kept apart from every inference port on purpose: nothing under cuda/af2, cuda/af3, cuda/ef2 ... includes this,
// so a forward fold compiles and runs exactly as before. A training graph is its own graph, held to the JAX
// reference's jax.grad (cuda/grad/oracle_*.py) and, through that, to the inference port's forward.
//
// Memory: a stack keeps each block's INPUT only and the backward replays one block's forward (with its saves)
// immediately before that block's backward - AF2's own use_remat - so what is held is one block's activations plus
// N block inputs, never N blocks of activations. Arena::reset() between blocks returns the saves.
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <map>
#include <tuple>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <nvtx3/nvToolsExt.h>

namespace grad {
// an NVTX range for nsys (nsys stats -r nvtx_gpu_proj_sum): free when no profiler is attached
struct Range { explicit Range(const char* n) { nvtxRangePushA(n); } ~Range() { nvtxRangePop(); } };


// The whole library is written on `real`: float32 for training and for the oracle gates, float64 under
// -DGRAD_F64 for the reference-free gradient check (gradcheck.cuh), where a central difference is exact to ~1e-8.
#ifdef GRAD_F64
typedef double real;
#else
typedef float real;
#endif
// BF16: float32 storage, every float GEMM on the bf16 tensor cores (cuBLAS rounds the operands, float32
// accumulation) and the flash attention on bf16 mma - the precision a bf16 training run computes in. Off by default.
inline bool BF16 = false;
inline cublasStatus_t xgemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k,
                            const float* al, const float* A, int la, const float* B, int lb, const float* be, float* C, int lc) {
  if (BF16)
    return cublasGemmEx(h, a, b, m, n, k, al, A, CUDA_R_32F, la, B, CUDA_R_32F, lb, be, C, CUDA_R_32F, lc,
                        CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cublasSgemm(h, a, b, m, n, k, al, A, la, B, lb, be, C, lc);
}
inline cublasStatus_t xgemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k,
                            const double* al, const double* A, int la, const double* B, int lb, const double* be, double* C, int lc) {
  return cublasDgemm(h, a, b, m, n, k, al, A, la, B, lb, be, C, lc);
}
inline cublasStatus_t xgemmB(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k, const float* al,
                             const float* A, int la, long long sa, const float* B, int lb, long long sb, const float* be,
                             float* C, int lc, long long sc, int count) {
  if (BF16)
    return cublasGemmStridedBatchedEx(h, a, b, m, n, k, al, A, CUDA_R_32F, la, sa, B, CUDA_R_32F, lb, sb, be, C,
                                      CUDA_R_32F, lc, sc, count, CUBLAS_COMPUTE_32F_FAST_16BF,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cublasSgemmStridedBatched(h, a, b, m, n, k, al, A, la, sa, B, lb, sb, be, C, lc, sc, count);
}
inline cublasStatus_t xgemmB(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k, const double* al,
                             const double* A, int la, long long sa, const double* B, int lb, long long sb, const double* be,
                             double* C, int lc, long long sc, int count) {
  return cublasDgemmStridedBatched(h, a, b, m, n, k, al, A, la, sa, B, lb, sb, be, C, lc, sc, count);
}

#define GK(x)                                                                                              \
  do {                                                                                                     \
    cudaError_t e_ = (x);                                                                                  \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } \
  } while (0)
#define GB(x)                                                                                              \
  do {                                                                                                     \
    cublasStatus_t s_ = (x);                                                                               \
    if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d cublas %d\n", __FILE__, __LINE__, (int)s_); exit(1); } \
  } while (0)

inline cudaStream_t STREAM = 0;
inline cublasHandle_t BLAS = nullptr;
// TF32 in the GEMMs: off for the gates (plain float32 against a float64 reference), on for a training run.
inline bool TF32 = false;

inline void init() {
  if (BLAS) return;
  GB(cublasCreate(&BLAS));
  GB(cublasSetStream(BLAS, STREAM));
  GB(cublasSetMathMode(BLAS, TF32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
  // the stream-ordered pool keeps what it has handed out (by default it gives it back at every synchronisation, and
  // the next cudaMallocAsync of the same bytes goes to the driver again: 0.9 ms on AF2's block forward)
  int dev; GK(cudaGetDevice(&dev));
  cudaMemPool_t pool; GK(cudaDeviceGetDefaultMemPool(&pool, dev));
  uint64_t keepAll = UINT64_MAX;
  GK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keepAll));
}
inline void setBf16(bool on) { BF16 = on; }
inline void setTf32(bool on) {
  TF32 = on;
  if (BLAS) GB(cublasSetMathMode(BLAS, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
}

inline unsigned blocks(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// ---------------------------------------------------------------- device memory
// The arena holds one block's saves and scratch; reset() between blocks. Chunks are kept, so after the first
// block nothing is allocated again.
// growable(): one contiguous range of virtual addresses instead, physical memory mapped onto its end as the high-water
// mark rises - a stack that never fragments, so what it holds is its peak. (Chunks fragment: a request larger than a
// chunk's tail skips it, and after a rewind the next pattern of sizes skips others - 13 GB held for a 9.1 GB peak in
// train-of3's 384-token step.) The driver's virtual memory calls come through the runtime's entry points, so nothing
// links libcuda.
struct VmCalls {
  CUresult (*reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
  CUresult (*create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
  CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
  CUresult (*access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
  CUresult (*granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
  CUresult (*unmap)(CUdeviceptr, size_t);
  CUresult (*release)(CUmemGenericAllocationHandle);
};
inline const VmCalls& vmCalls() {
  static VmCalls c = [] {
    VmCalls v{};
    auto get = [](const char* name, void** fn) {
      cudaDriverEntryPointQueryResult q;
      GK(cudaGetDriverEntryPoint(name, fn, cudaEnableDefault, &q));
      if (q != cudaDriverEntryPointSuccess || !*fn) { fprintf(stderr, "the driver has no %s\n", name); exit(1); }
    };
    get("cuMemAddressReserve", (void**)&v.reserve); get("cuMemCreate", (void**)&v.create); get("cuMemMap", (void**)&v.map);
    get("cuMemSetAccess", (void**)&v.access); get("cuMemGetAllocationGranularity", (void**)&v.granularity);
    get("cuMemUnmap", (void**)&v.unmap); get("cuMemRelease", (void**)&v.release);
    return v;
  }();
  return c;
}
#define GV(x)                                                                                              \
  do {                                                                                                     \
    CUresult r_ = (x);                                                                                     \
    if (r_ != CUDA_SUCCESS) { fprintf(stderr, "%s:%d %s (driver %d)\n", __FILE__, __LINE__, r_ == CUDA_ERROR_OUT_OF_MEMORY ? "out of memory" : "virtual memory call failed", (int)r_); exit(1); } \
  } while (0)
struct Arena {
  std::vector<std::pair<char*, size_t>> chunks;
  size_t chunk = 0, at = 0, used = 0, high = 0;
  // growable(): base .. base + reserved addressed, base .. base + mapped backed
  char* base = nullptr;
  size_t reserved = 0, mapped = 0, step = 0;
  CUmemAllocationProp prop{};
  void growable() {
    if (base || !chunks.empty()) { fprintf(stderr, "Arena::growable: the arena is already in use\n"); exit(1); }
    int dev; GK(cudaGetDevice(&dev));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = dev;
    size_t gran; GV(vmCalls().granularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t freeB, total; GK(cudaMemGetInfo(&freeB, &total));
    step = ((size_t)64 << 20) / gran * gran;           // (mapped 64 MiB at a time)
    reserved = (total + step - 1) / step * step;       // (more addresses than the device has memory)
    CUdeviceptr p; GV(vmCalls().reserve(&p, reserved, 0, 0, 0));
    base = (char*)p;
  }
  void mapTo(size_t end) {
    if (end > reserved) { fprintf(stderr, "Arena: %zu bytes asked of a %zu-byte reservation\n", end, reserved); exit(1); }
    const size_t want = (end + step - 1) / step * step;
    if (want <= mapped) return;
    const size_t bytes = want - mapped;
    CUmemGenericAllocationHandle h; GV(vmCalls().create(&h, bytes, &prop, 0));
    GV(vmCalls().map((CUdeviceptr)(base + mapped), bytes, 0, h, 0));
    CUmemAccessDesc acc{}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    GV(vmCalls().access((CUdeviceptr)(base + mapped), bytes, &acc, 1));
    mapped = want;
  }
  real* get(size_t n) {
    size_t bytes = (n * sizeof(real) + 255) & ~(size_t)255;
    if (base) {
      mapTo(at + bytes);
      real* p = (real*)(base + at);
      at += bytes; used += bytes; high = std::max(high, used);
      return p;
    }
    while (chunk < chunks.size() && at + bytes > chunks[chunk].second) { ++chunk; at = 0; }
    if (chunk == chunks.size()) {
      size_t size = std::max(bytes, (size_t)256 << 20);
      char* p; GK(cudaMalloc(&p, size));
      chunks.push_back({p, size});
      at = 0;
    }
    real* p = (real*)(chunks[chunk].first + at);
    at += bytes; used += bytes; high = std::max(high, used);
    return p;
  }
  real* zeros(size_t n) { real* p = get(n); GK(cudaMemsetAsync(p, 0, n * sizeof(real), STREAM)); return p; }
  void reset() { chunk = 0; at = 0; used = 0; }
  struct Mark { size_t chunk, at, used; };
  Mark mark() const { return {chunk, at, used}; }
  void rewind(const Mark& m) { chunk = m.chunk; at = m.at; used = m.used; }
  size_t held() const { size_t s = mapped; for (auto& c : chunks) s += c.second; return s; }
  void release() {
    GK(cudaStreamSynchronize(STREAM));
    if (base) { reset(); return; }   // (a growable arena keeps its mapping: its peak is what the next step needs)
    for (auto& c : chunks) GK(cudaFree(c.first));
    chunks.clear(); reset();
  }
};
inline Arena ARENA;
// Scratch for the rest of a scope: everything ARENA hands out after this is returned when it ends. A module's
// backward runs in one, so only the saves and the running gradients outlive it.
struct Scratch {
  Arena::Mark m = ARENA.mark();
  ~Scratch() { ARENA.rewind(m); }
};

// A trainable tensor: float32 value and gradient, both persistent. AdamW's moments live with the optimizer.
struct Param {
  std::string name;
  real* w = nullptr;
  real* g = nullptr;
  size_t n = 0;
  int fanIn = 0;      // a linear weight's input width (set where the module knows it; 0 elsewhere) - what an initialiser scales by
};
struct Params {
  std::vector<Param*> all;
  void add(Param& p) { all.push_back(&p); }
  void zeroGrad() { for (auto* p : all) GK(cudaMemsetAsync(p->g, 0, p->n * sizeof(real), STREAM)); }
  size_t count() const { size_t s = 0; for (auto* p : all) s += p->n; return s; }
};
inline void allocParam(Param& p, const std::string& name, size_t n) {
  p.name = name; p.n = n;
  GK(cudaMalloc(&p.w, n * sizeof(real)));
  GK(cudaMalloc(&p.g, n * sizeof(real)));
  GK(cudaMemset(p.g, 0, n * sizeof(real)));
}

// ---------------------------------------------------------------- row-major GEMMs on cuBLAS's column-major
// y[M,N] = x[M,K] w[K,N] (+ beta y)
inline void mm(const real* x, const real* w, real* y, int M, int K, int N, real beta = 0.f, real alpha = 1.f) {
  GB(xgemm(BLAS, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, w, N, x, K, &beta, y, N));
}
// dx[M,K] (+)= dy[M,N] w[K,N]^T
inline void mmBT(const real* dy, const real* w, real* dx, int M, int K, int N, real beta = 0.f, real alpha = 1.f) {
  GB(xgemm(BLAS, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, w, N, dy, N, &beta, dx, K));
}
// dw[K,N] (+)= x[M,K]^T dy[M,N]. A weight gradient is a tall reduction into a small output (65,536 pair rows into
// 128 x 128), which cuBLAS gives one CTA a 128 x 128 tile and does not split: 0.43 ms where the arithmetic is 0.03.
// Where the output has too few tiles to fill the device, M is cut into equal chunks (one strided-batched GEMM into
// partials) and the partials summed - in a fixed order, so it is as deterministic as the single GEMM.
__global__ void sumPartialsK(real* dw, const real* p, int parts, size_t n, real alpha, real beta) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  real s = 0;
  for (int k = 0; k < parts; ++k) s += p[k * n + i];
  dw[i] = beta == 0 ? alpha * s : beta * dw[i] + alpha * s;
}
inline void mmAT(const real* x, const real* dy, real* dw, int M, int K, int N, real beta = 1.f, real alpha = 1.f) {
  // measured on the A100 (cuda/grad/README.md): only TF32 with a one-tile output gains - 65,536 x 128 x 128 0.42 ms
  // -> 0.07 at 32 chunks - while pedantic float32's strided-batched kernels are 10-50x slower than its single GEMM
  // and a larger output already fills the device
  const bool oneTile = K <= 128 && N <= 128 && (size_t)K * N >= 8192;
  const int splits = TF32 && oneTile ? std::min(32, M / 2048) : 1;
  if (splits <= 1) {
    GB(xgemm(BLAS, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, dy, N, x, K, &beta, dw, N));
    return;
  }
  Scratch scratch;
  const int c = M / splits, rest = M - c * splits;
  const size_t n = (size_t)K * N;
  real* p = ARENA.get(n * splits);
  const real one = 1, zero = 0;
  GB(xgemmB(BLAS, CUBLAS_OP_N, CUBLAS_OP_T, N, K, c, &one, dy, N, (long long)c * N, x, K, (long long)c * K, &zero, p,
            N, (long long)n, splits));
  if (rest)
    GB(xgemm(BLAS, CUBLAS_OP_N, CUBLAS_OP_T, N, K, rest, &one, dy + (size_t)c * splits * N, N,
             x + (size_t)c * splits * K, K, &one, p, N));
  sumPartialsK<<<blocks(n), 256, 0, STREAM>>>(dw, p, splits, n, alpha, beta);
}
// batched over `count` matrices at fixed strides: C_b = op(A_b) op(B_b), all row-major
// (row-major C = A B is column-major C^T = B^T A^T)
inline void bmm(bool ta, bool tb, int M, int N, int K, const real* A, long long sa, const real* B, long long sb,
                real* C, long long sc, int count, real beta = 0.f, real alpha = 1.f) {
  int lda = ta ? M : K, ldb = tb ? K : N;
  GB(xgemmB(BLAS, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                               B, ldb, sb, A, lda, sa, &beta, C, N, sc, count));
}

// ---------------------------------------------------------------- small elementwise kernels
__global__ void addK(real* y, const real* x, size_t n) {           // y += x
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += x[t];
}
inline void add(real* y, const real* x, size_t n) { if (n) addK<<<blocks(n), 256, 0, STREAM>>>(y, x, n); }
// y = a + b, out of place: a residual link in three passes where a copy and then an add were five
__global__ void add3K(real* __restrict__ y, const real* __restrict__ a, const real* __restrict__ b, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = a[t] + b[t];
}
__global__ void add3V4K(float4* __restrict__ y, const float4* __restrict__ a, const float4* __restrict__ b, size_t n4) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n4) { const float4 u = a[t], v = b[t]; y[t] = make_float4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w); }
}
inline void add3(real* y, const real* a, const real* b, size_t n) {
  if (!n) return;
  if (sizeof(real) == 4 && n % 4 == 0 && !(((uintptr_t)y | (uintptr_t)a | (uintptr_t)b) & 15))
    add3V4K<<<blocks(n / 4), 256, 0, STREAM>>>(reinterpret_cast<float4*>(y), reinterpret_cast<const float4*>(a),
                                               reinterpret_cast<const float4*>(b), n / 4);
  else add3K<<<blocks(n), 256, 0, STREAM>>>(y, a, b, n);
}

__global__ void biasK(real* y, const real* b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[t] += b[t % C];
}
// g[c] += sum over rows of d[r, c]; a block a column tile, rows strided, one atomic a column a block
__global__ void colSumK(real* g, const real* d, size_t rows, int C, int ld) {
  int c = blockIdx.x * 32 + threadIdx.x;
  if (c >= C) return;
  real s = 0;
  for (size_t r = blockIdx.y * blockDim.y + threadIdx.y; r < rows; r += (size_t)gridDim.y * blockDim.y) s += d[r * ld + c];
  atomicAdd(&g[c], s);
}
// the same four columns a thread (float4 loads) where C allows: a block 32 x 4 = 128 columns by 8 row lanes, the
// row lanes summed in shared memory and one atomic a column a block (atomics from every thread onto 128 addresses
// were 5.6 ms of a 23 ms backward)
__global__ void colSum4K(float* g, const float* d, size_t rows, int C, int ld) {
  __shared__ float4 part[8][32];
  int c = (blockIdx.x * 32 + threadIdx.x) * 4;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  if (c < C)
    for (size_t r = blockIdx.y * blockDim.y + threadIdx.y; r < rows; r += (size_t)gridDim.y * blockDim.y) {
      float4 v = *reinterpret_cast<const float4*>(d + r * ld + c);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
  part[threadIdx.y][threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.y || c >= C) return;
  for (int y = 1; y < 8; ++y) { float4 v = part[y][threadIdx.x]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
  atomicAdd(&g[c], s.x); atomicAdd(&g[c + 1], s.y); atomicAdd(&g[c + 2], s.z); atomicAdd(&g[c + 3], s.w);
}
// (d at row stride ld: a column block of a wider matrix)
inline void colSum(real* g, const real* d, size_t rows, int C, int ld = 0) {
  if (!rows) return;
  if (!ld) ld = C;
  if (sizeof(real) == 4 && C % 4 == 0 && ld % 4 == 0 && ((uintptr_t)d & 15) == 0) {
    const int C4 = C / 4;
    const unsigned xb = (C4 + 31) / 32;
    dim3 b(32, 8), gr(xb, (unsigned)std::min<size_t>((rows + 7) / 8, std::max(1u, 432 / xb)));
    colSum4K<<<gr, b, 0, STREAM>>>((float*)g, (const float*)d, rows, C, ld);
    return;
  }
  dim3 b(32, 8), gr((C + 31) / 32, (unsigned)std::min<size_t>((rows + 7) / 8, 256));
  colSumK<<<gr, b, 0, STREAM>>>(g, d, rows, C, ld);
}

// ---------------------------------------------------------------- a Linear: y = x W + b, W [K, N] haiku's layout
// W is [K, N] (in, out: haiku's layout), or [N, K] with `trans` (AF3's transpose_weights, torch's (out, in))
struct Linear {
  Param w, b;
  int K = 0, N = 0;
  bool hasBias = true, trans = false;
  void forward(const real* x, real* y, size_t rows, real beta = 0) const {
    if (trans) mmBT(x, w.w, y, (int)rows, N, K, beta);
    else mm(x, w.w, y, (int)rows, K, N, beta);
    if (hasBias) biasK<<<blocks(rows * N), 256, 0, STREAM>>>(y, b.w, rows, N);
  }
  // dW += x^T dy, db += colsum dy; dx (+)= dy W^T when dx is not null
  void backward(const real* x, const real* dy, real* dx, size_t rows, bool accumulate = false) {
    if (trans) mmAT(dy, x, w.g, (int)rows, N, K, 1.f);
    else mmAT(x, dy, w.g, (int)rows, K, N, 1.f);
    if (hasBias) colSum(b.g, dy, rows, N);
    if (dx) {
      if (trans) mm(dy, w.w, dx, (int)rows, N, K, accumulate ? 1.f : 0.f);
      else mmBT(dy, w.w, dx, (int)rows, K, N, accumulate ? 1.f : 0.f);
    }
  }
};

#ifndef GRAD_F64
// ---------------------------------------------------------------- bf16 storage (BF16 mode)
// Where a module keeps its large activations in bf16 - half the bytes of every read and write, which is most of what
// a pair-track GEMM (an inner width of 128) and every elementwise pass cost - its GEMMs take bf16 operands and
// accumulate in float: what a bf16 run's autocast computes. Parameters and their gradients stay float.
typedef __nv_bfloat16 bf;
template <class T> constexpr cudaDataType dtype() { return sizeof(T) == 2 ? CUDA_R_16BF : CUDA_R_32F; }
__global__ void toBfK(const float* __restrict__ x, __nv_bfloat162* __restrict__ y, size_t n2) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n2) { const float2 v = reinterpret_cast<const float2*>(x)[i]; y[i] = __floats2bfloat162_rn(v.x, v.y); }
}
__global__ void toBfTailK(const float* x, bf* y, size_t n) { if (n & 1) y[n - 1] = __float2bfloat16(x[n - 1]); }
inline void toBf(const float* x, bf* y, size_t n) {   // (x 8-byte aligned)
  if (n / 2) toBfK<<<blocks(n / 2), 256, 0, STREAM>>>(x, reinterpret_cast<__nv_bfloat162*>(y), n / 2);
  if (n & 1) toBfTailK<<<1, 1, 0, STREAM>>>(x, y, n);
}
inline bf* bfGet(size_t n) { return reinterpret_cast<bf*>(ARENA.get((n + 1) / 2)); }
inline bf* toBf(const float* x, size_t n) { bf* y = bfGet(n); toBf(x, y, n); return y; }
inline bf* bfZeros(size_t n) { bf* y = bfGet(n); GK(cudaMemsetAsync(y, 0, n * 2, STREAM)); return y; }
inline cublasLtHandle_t LT = nullptr;
constexpr size_t LT_WS = 32u << 20;
inline void* ltWorkspace() {
  static void* ws = nullptr;
  if (!LT) { cublasLtCreate(&LT); GK(cudaMalloc(&ws, LT_WS)); }
  return ws;
}
// cublasGemmEx's contract (column-major, alpha 1, float compute) on cuBLASLt, the algorithm TIMED: the first call of
// each shape runs cuBLASLt's 16 best-ranked candidates on a scratch output and keeps the fastest. cuBLAS's own pick
// for a bf16 output is often an sm_75-style s1688 kernel with one pipeline stage - 186 us against 95 for the
// timed choice on the OPM's 65,536 x 1,024 x 128 backward.
inline void gemmEx(cublasOperation_t a, cublasOperation_t b, int m, int n, int k, const void* A, cudaDataType tA,
                   int la, const void* B, cudaDataType tB, int lb, float beta, void* C, cudaDataType tC, int lc) {
  struct Plan { cublasLtMatmulDesc_t d; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; };
  static std::map<std::vector<long long>, Plan> plans;
  void* ws = ltWorkspace();
  auto align = [](const void* p) { const uintptr_t u = (uintptr_t)p; return u % 16 == 0 ? 16 : u % 8 == 0 ? 8 : u % 4 == 0 ? 4 : 2; };
  const std::vector<long long> key{a, b, m, n, k, tA, la, tB, lb, tC, lc, beta != 0.f, align(A), align(B), align(C)};
  auto it = plans.find(key);
  if (it == plans.end()) {
    Plan p;
    GB(cublasLtMatmulDescCreate(&p.d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    GB(cublasLtMatmulDescSetAttribute(p.d, CUBLASLT_MATMUL_DESC_TRANSA, &a, sizeof a));
    GB(cublasLtMatmulDescSetAttribute(p.d, CUBLASLT_MATMUL_DESC_TRANSB, &b, sizeof b));
    GB(cublasLtMatrixLayoutCreate(&p.a, tA, a == CUBLAS_OP_N ? m : k, a == CUBLAS_OP_N ? k : m, la));
    GB(cublasLtMatrixLayoutCreate(&p.b, tB, b == CUBLAS_OP_N ? k : n, b == CUBLAS_OP_N ? n : k, lb));
    GB(cublasLtMatrixLayoutCreate(&p.c, tC, m, n, lc));
    cublasLtMatmulPreference_t pr;
    GB(cublasLtMatmulPreferenceCreate(&pr));
    GB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &LT_WS, sizeof LT_WS));
    const uint32_t al[3] = {(uint32_t)align(A), (uint32_t)align(B), (uint32_t)align(C)};
    GB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &al[0], 4));
    GB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &al[1], 4));
    GB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &al[2], 4));
    GB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &al[2], 4));
    cublasLtMatmulHeuristicResult_t h[16]; int nh = 0;
    GB(cublasLtMatmulAlgoGetHeuristic(LT, p.d, p.a, p.b, p.c, p.c, pr, 16, h, &nh));
    cublasLtMatmulPreferenceDestroy(pr);
    if (!nh) { fprintf(stderr, "gemmEx: no cuBLASLt algorithm for %d x %d x %d\n", m, n, k); exit(1); }
    p.algo = h[0].algo;
    if (nh > 1) {   // time them on a scratch C (offset as C is, for its alignment): two rounds, the minimum
      const size_t es = tC == CUDA_R_32F ? 4 : 2, bytes = ((size_t)lc * (n - 1) + m) * es + 16;
      char* sc; GK(cudaMallocAsync(&sc, bytes, STREAM));
      void* tc = sc + ((uintptr_t)C % 16);
      GK(cudaMemsetAsync(sc, 0, bytes, STREAM));
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      const float one = 1.f;
      float best = INFINITY;
      for (int round = 0; round < 2; ++round)
        for (int i = 0; i < nh; ++i) {
          cudaEventRecord(e0, STREAM);
          const cublasStatus_t st = cublasLtMatmul(LT, p.d, &one, A, p.a, B, p.b, &beta, tc, p.c, tc, p.c, &h[i].algo, ws, LT_WS, STREAM);
          cudaEventRecord(e1, STREAM);
          GK(cudaEventSynchronize(e1));
          float t; cudaEventElapsedTime(&t, e0, e1);
          if (st == CUBLAS_STATUS_SUCCESS && round == 1 && t < best) { best = t; p.algo = h[i].algo; }
        }
      cudaEventDestroy(e0); cudaEventDestroy(e1);
      GK(cudaFreeAsync(sc, STREAM));
    }
    it = plans.emplace(key, p).first;
  }
  Plan& p = it->second;
  const float one = 1.f;
  const cublasStatus_t st = cublasLtMatmul(LT, p.d, &one, A, p.a, B, p.b, &beta, C, p.c, C, p.c, &p.algo, ws, LT_WS, STREAM);
  if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "gemmEx %d x %d x %d: cublasLt %d\n", m, n, k, (int)st); exit(1); }
}
// mm, mmBT, mmAT on bf16 operands (row-major, as above); y and dx float or bf16, dw float
template <class Y> inline void mmBf(const bf* x, const bf* w, Y* y, int M, int K, int N, float beta = 0) {
  gemmEx(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, w, CUDA_R_16BF, N, x, CUDA_R_16BF, K, beta, y, dtype<Y>(), N);
}
template <class Y> inline void mmBTBf(const bf* dy, const bf* w, Y* dx, int M, int K, int N, float beta = 0) {
  gemmEx(CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, w, CUDA_R_16BF, N, dy, CUDA_R_16BF, N, beta, dx, dtype<Y>(), K);
}
// (a weight gradient's tall reduction into a few output tiles: cuBLAS's own pick ran 147,456 pair rows into 128 x 256
// as two CTAs, 0.8 ms a call, so it was cut into row chunks summed in a fixed order; gemmEx's timed cuBLASLt choice
// splits K itself, its partials reduced in a workspace - 2.99 ms of GEMMs in AF2's backward against 3.11)
inline void mmATBf(const bf* x, const bf* dy, float* dw, int M, int K, int N, float beta = 1) {
  gemmEx(CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, dy, CUDA_R_16BF, N, x, CUDA_R_16BF, K, beta, dw, CUDA_R_32F, N);
}
// y[M, N] = x[M, K] W + b, (relu), on cuBLASLt with the bias - and the relu - in the GEMM's epilogue (bindcraft2's
// lt.cuh): W [K, N], or [N, K] with trans; bf16 operands, y bf16 (then the bias must be bf16: cuBLASLt has no
// float bias for a bf16 output) or float. Plans are cached by shape; one workspace serves every call.
template <class Y>
inline void ltLinear(const bf* x, const bf* w, bool trans, const void* bias, bool relu, Y* y, int M, int K, int N,
                     float beta) {
  struct Plan { cublasLtMatmulDesc_t d; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; };
  static std::map<std::tuple<int, int, int, bool, bool, int>, Plan> plans;
  void* ws = ltWorkspace();
  constexpr size_t WS = LT_WS;
  const auto key = std::make_tuple(M, K, N, trans, relu, (int)sizeof(Y));
  auto it = plans.find(key);
  if (it == plans.end()) {
    Plan p;
    GB(cublasLtMatmulDescCreate(&p.d, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    GB(cublasLtMatmulDescSetAttribute(p.d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    const cublasLtEpilogue_t ep = relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
    GB(cublasLtMatmulDescSetAttribute(p.d, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof ep));
    const cudaDataType bt = dtype<Y>();
    GB(cublasLtMatmulDescSetAttribute(p.d, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof bt));
    // column-major: y^T [N, M] = op(W)^T [N, K] x^T [K, M]
    GB(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, trans ? K : N, trans ? N : K, trans ? K : N));
    GB(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, K));
    GB(cublasLtMatrixLayoutCreate(&p.c, dtype<Y>(), N, M, N));
    cublasLtMatmulPreference_t pr;
    GB(cublasLtMatmulPreferenceCreate(&pr));
    GB(cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
    cublasLtMatmulHeuristicResult_t h; int n = 0;
    GB(cublasLtMatmulAlgoGetHeuristic(LT, p.d, p.a, p.b, p.c, p.c, pr, 1, &h, &n));
    if (!n) { fprintf(stderr, "ltLinear: no algorithm for %d x %d x %d\n", M, K, N); exit(1); }
    p.algo = h.algo;
    cublasLtMatmulPreferenceDestroy(pr);
    it = plans.emplace(key, p).first;
  }
  Plan& p = it->second;
  GB(cublasLtMatmulDescSetAttribute(p.d, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias));
  const float one = 1.f;
  const cublasStatus_t st = cublasLtMatmul(LT, p.d, &one, w, p.a, x, p.b, &beta, y, p.c, y, p.c, &p.algo, ws, WS, STREAM);
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "ltLinear %d x %d x %d trans %d relu %d out %zu beta %g: cublas %d (x %p w %p y %p b %p)\n", M, K, N,
            trans, relu, sizeof(Y), beta, (int)st, (const void*)x, (const void*)w, (void*)y, bias);
    exit(1);
  }
}
// up to four Linears on one input packed side by side: W [K, sum N] bf16 (each role [N, K] when transposed, [K, N]
// otherwise) and its bias (zero where a role has none); back: dW [K, sum N] (float) added into each role's gradient
struct Pack4 { const float* w[4]; float* g[4]; bool trans[4]; int off[5]; int n; };
__global__ void packLinK(Pack4 p, bf* W, const float* dW, int K, bool back) {
  const int Nt = p.off[p.n];
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)K * Nt) return;
  const int col = (int)(i % Nt), k = (int)(i / Nt);
  int r = 0;
  while (col >= p.off[r + 1]) ++r;
  const int j = col - p.off[r], N = p.off[r + 1] - p.off[r];
  const size_t at = p.trans[r] ? (size_t)j * K + k : (size_t)k * N + j;
  if (back) p.g[r][at] += dW[i];
  else W[i] = __float2bfloat16(p.w[r][at]);
}
__global__ void packBiasK(Pack4 p, const float* b0, const float* b1, const float* b2, const float* b3, bf* B) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= p.off[p.n]) return;
  int r = 0;
  while (c >= p.off[r + 1]) ++r;
  const float* b = r == 0 ? b0 : r == 1 ? b1 : r == 2 ? b2 : b3;
  B[c] = __float2bfloat16(b ? b[c - p.off[r]] : 0.f);
}
inline Pack4 pack4(const Linear* const* L, int n) {
  Pack4 p{};
  p.n = n;
  for (int r = 0; r < n; ++r) { p.w[r] = L[r]->w.w; p.g[r] = L[r]->w.g; p.trans[r] = L[r]->trans; p.off[r + 1] = p.off[r] + L[r]->N; }
  return p;
}
// the packed weight (and bias, null when no role has one)
inline bf* packLinears(const Linear* const* L, int n, bf** bias) {
  const Pack4 p = pack4(L, n);
  const int K = L[0]->K, Nt = p.off[n];
  bf* W = bfGet((size_t)K * Nt);
  packLinK<<<blocks((size_t)K * Nt), 256, 0, STREAM>>>(p, W, nullptr, K, false);
  bool any = false;
  for (int r = 0; r < n; ++r) any |= L[r]->hasBias;
  *bias = nullptr;
  if (any) {
    *bias = bfGet(Nt);
    const float* b[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int r = 0; r < n; ++r) if (L[r]->hasBias) b[r] = L[r]->b.w;
    packBiasK<<<blocks(Nt), 256, 0, STREAM>>>(p, b[0], b[1], b[2], b[3], *bias);
  }
  return W;
}
inline void unpackLinearGrads(const Linear* const* L, int n, const float* dW) {
  const Pack4 p = pack4(L, n);
  packLinK<<<blocks((size_t)L[0]->K * p.off[n]), 256, 0, STREAM>>>(p, nullptr, dW, L[0]->K, true);
}

// Linear on bf16 operands; a bf16 output's bias in the GEMM's epilogue (ltLinear), a float one's added after
// (biasAddK), its gradient summed in float (colSumBf: ops.cuh)
template <class Y> __global__ void biasAddK(Y* y, const float* b, size_t rows, int C);
inline void colSumBf(float* g, const bf* d, size_t rows, int C, int ld);
template <class Y>
inline void linearFwdBf(const Linear& l, const bf* x, Y* y, size_t rows, float beta = 0, bool relu = false) {
  {
    Scratch scratch;
    const bf* w = toBf(l.w.w, (size_t)l.K * l.N);
    if (sizeof(Y) == 2 && (l.hasBias || relu)) {   // (a float output's float bias: cuBLASLt plans it, then refuses it)
      ltLinear(x, w, l.trans, l.hasBias ? toBf(l.b.w, l.N) : bfZeros(l.N), relu, y, (int)rows, l.K, l.N, beta);
      return;
    }
    if (l.trans) mmBTBf(x, w, y, (int)rows, l.N, l.K, beta);
    else mmBf(x, w, y, (int)rows, l.K, l.N, beta);
  }
  if (l.hasBias) biasAddK<<<blocks(rows * l.N), 256, 0, STREAM>>>(y, l.b.w, rows, l.N);
  if (relu) { fprintf(stderr, "linearFwdBf: relu on a float output\n"); exit(1); }
}
template <class DX>
inline void linearBwdBf(Linear& l, const bf* x, const bf* dy, DX* dx, size_t rows, bool accumulate = false) {
  if (l.trans) mmATBf(dy, x, l.w.g, (int)rows, l.N, l.K);
  else mmATBf(x, dy, l.w.g, (int)rows, l.K, l.N);
  if (l.hasBias) colSumBf(l.b.g, dy, rows, l.N, l.N);
  if (!dx) return;
  Scratch scratch;
  const bf* w = toBf(l.w.w, (size_t)l.K * l.N);
  if (l.trans) mmBf(dy, w, dx, (int)rows, l.N, l.K, accumulate ? 1.f : 0.f);
  else mmBTBf(dy, w, dx, (int)rows, l.K, l.N, accumulate ? 1.f : 0.f);
}
#endif

// ---------------------------------------------------------------- a device dot product, in double
__global__ void dotK(const real* a, const real* b, size_t n, double* out) {
  double s = 0;
  for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x)
    s += (double)a[t] * b[t];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}
inline double dot(const real* a, const real* b, size_t n) {
  static double* acc = nullptr;
  if (!acc) GK(cudaMalloc(&acc, 8));
  GK(cudaMemsetAsync(acc, 0, 8, STREAM));
  dotK<<<(unsigned)std::min<size_t>(blocks(n), 1024), 256, 0, STREAM>>>(a, b, n, acc);
  double h; GK(cudaMemcpyAsync(&h, acc, 8, cudaMemcpyDeviceToHost, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  return h;
}
__global__ void axpyR(real* y, const real* x, real a, size_t n) {     // y += a x
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += a * x[t];
}

// ---------------------------------------------------------------- comparisons for the gates
template <class A, class B> inline double relRms(const std::vector<A>& a, const std::vector<B>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
  if (!(den > 0)) return num > 0 ? INFINITY : 0;
  return std::sqrt(num / den);
}
inline std::vector<real> host(const real* d, size_t n) {
  std::vector<real> h(n);
  GK(cudaMemcpyAsync(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  return h;
}

}  // namespace grad
