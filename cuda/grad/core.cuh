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
#include <cublas_v2.h>
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
struct Arena {
  std::vector<std::pair<char*, size_t>> chunks;
  size_t chunk = 0, at = 0, used = 0, high = 0;
  real* get(size_t n) {
    size_t bytes = (n * sizeof(real) + 255) & ~(size_t)255;
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
  size_t held() const { size_t s = 0; for (auto& c : chunks) s += c.second; return s; }
  void release() {
    GK(cudaStreamSynchronize(STREAM));
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
