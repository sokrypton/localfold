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
#include <string>
#include <vector>

namespace grad {

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
  float* get(size_t n) {
    size_t bytes = (n * 4 + 255) & ~(size_t)255;
    while (chunk < chunks.size() && at + bytes > chunks[chunk].second) { ++chunk; at = 0; }
    if (chunk == chunks.size()) {
      size_t size = std::max(bytes, (size_t)256 << 20);
      char* p; GK(cudaMalloc(&p, size));
      chunks.push_back({p, size});
      at = 0;
    }
    float* p = (float*)(chunks[chunk].first + at);
    at += bytes; used += bytes; high = std::max(high, used);
    return p;
  }
  float* zeros(size_t n) { float* p = get(n); GK(cudaMemsetAsync(p, 0, n * 4, STREAM)); return p; }
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
  float* w = nullptr;
  float* g = nullptr;
  size_t n = 0;
};
struct Params {
  std::vector<Param*> all;
  void add(Param& p) { all.push_back(&p); }
  void zeroGrad() { for (auto* p : all) GK(cudaMemsetAsync(p->g, 0, p->n * 4, STREAM)); }
  size_t count() const { size_t s = 0; for (auto* p : all) s += p->n; return s; }
};
inline void allocParam(Param& p, const std::string& name, size_t n) {
  p.name = name; p.n = n;
  GK(cudaMalloc(&p.w, n * 4));
  GK(cudaMalloc(&p.g, n * 4));
  GK(cudaMemset(p.g, 0, n * 4));
}

// ---------------------------------------------------------------- row-major GEMMs on cuBLAS's column-major
// y[M,N] = x[M,K] w[K,N] (+ beta y)
inline void mm(const float* x, const float* w, float* y, int M, int K, int N, float beta = 0.f, float alpha = 1.f) {
  GB(cublasSgemm(BLAS, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, w, N, x, K, &beta, y, N));
}
// dx[M,K] (+)= dy[M,N] w[K,N]^T
inline void mmBT(const float* dy, const float* w, float* dx, int M, int K, int N, float beta = 0.f, float alpha = 1.f) {
  GB(cublasSgemm(BLAS, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, w, N, dy, N, &beta, dx, K));
}
// dw[K,N] (+)= x[M,K]^T dy[M,N]
inline void mmAT(const float* x, const float* dy, float* dw, int M, int K, int N, float beta = 1.f, float alpha = 1.f) {
  GB(cublasSgemm(BLAS, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, dy, N, x, K, &beta, dw, N));
}
// batched over `count` matrices at fixed strides: C_b = op(A_b) op(B_b), all row-major
// (row-major C = A B is column-major C^T = B^T A^T)
inline void bmm(bool ta, bool tb, int M, int N, int K, const float* A, long long sa, const float* B, long long sb,
                float* C, long long sc, int count, float beta = 0.f, float alpha = 1.f) {
  int lda = ta ? M : K, ldb = tb ? K : N;
  GB(cublasSgemmStridedBatched(BLAS, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                               B, ldb, sb, A, lda, sa, &beta, C, N, sc, count));
}

// ---------------------------------------------------------------- small elementwise kernels
__global__ void addK(float* y, const float* x, size_t n) {           // y += x
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += x[t];
}
inline void add(float* y, const float* x, size_t n) { if (n) addK<<<blocks(n), 256, 0, STREAM>>>(y, x, n); }

__global__ void biasK(float* y, const float* b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[t] += b[t % C];
}
// g[c] += sum over rows of d[r, c]; a block a column tile, rows strided, one atomic a column a block
__global__ void colSumK(float* g, const float* d, size_t rows, int C) {
  int c = blockIdx.x * 32 + threadIdx.x;
  if (c >= C) return;
  float s = 0;
  for (size_t r = blockIdx.y * blockDim.y + threadIdx.y; r < rows; r += (size_t)gridDim.y * blockDim.y) s += d[r * C + c];
  atomicAdd(&g[c], s);
}
inline void colSum(float* g, const float* d, size_t rows, int C) {
  if (!rows) return;
  dim3 b(32, 8), gr((C + 31) / 32, (unsigned)std::min<size_t>((rows + 7) / 8, 256));
  colSumK<<<gr, b, 0, STREAM>>>(g, d, rows, C);
}

// ---------------------------------------------------------------- a Linear: y = x W + b, W [K, N] haiku's layout
struct Linear {
  Param w, b;
  int K = 0, N = 0;
  bool hasBias = true;
  void forward(const float* x, float* y, size_t rows) const {
    mm(x, w.w, y, (int)rows, K, N);
    if (hasBias) biasK<<<blocks(rows * N), 256, 0, STREAM>>>(y, b.w, rows, N);
  }
  // dW += x^T dy, db += colsum dy; dx (+)= dy W^T when dx is not null
  void backward(const float* x, const float* dy, float* dx, size_t rows, bool accumulate = false) {
    mmAT(x, dy, w.g, (int)rows, K, N, 1.f);
    if (hasBias) colSum(b.g, dy, rows, N);
    if (dx) mmBT(dy, w.w, dx, (int)rows, K, N, accumulate ? 1.f : 0.f);
  }
};

// ---------------------------------------------------------------- comparisons for the gates
inline double relRms(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
  if (!(den > 0)) return num > 0 ? INFINITY : 0;
  return std::sqrt(num / den);
}
inline std::vector<float> host(const float* d, size_t n) {
  std::vector<float> h(n);
  GK(cudaMemcpyAsync(h.data(), d, n * 4, cudaMemcpyDeviceToHost, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  return h;
}

}  // namespace grad
