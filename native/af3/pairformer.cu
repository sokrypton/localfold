// One AlphaFold 3 pairformer block in CUDA + cuBLAS, transcribed from
// src/af3/trunk/pairformer-reference.js (the spec), checked against its output.
//
//   block <data-dir> [--tokens=N] [--tf32] [--repeat=K]
//
// With the exported tokens it runs the exported input and compares; with any
// other --tokens it runs a seeded random input at that size, for timing.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// -DUSE_FP16 builds the same low-precision path in IEEE half: 10 mantissa bits to bf16's 7,
// the same tensor-core rate, a narrower range.
#ifdef USE_FP16
#define __nv_bfloat16 __half
#define __nv_bfloat162 __half2
#define __float2bfloat16 __float2half
#define __floats2bfloat162_rn __floats2half2_rn
#define __bfloat162float __half2float
#define LP_CUDA CUDA_R_16F
#define LP_MMA "f16.f16"
#else
#define LP_CUDA CUDA_R_16BF
#define LP_MMA "bf16.bf16"
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

static cublasHandle_t H;
static double FLASH_MS = 0; static bool BF16 = false; static bool TF_LINEAR = false, TF_TRI = false, TF_ATT = false, FLASH = true, TC = true, FUSED = true, MMA = true, LEAN = true, GRAPH = false;
static void mode(bool tf) { CB(cublasSetMathMode(H, tf ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH)); }
static const float EPS = 1e-5f;

// ---------------------------------------------------------------- data
struct Data {
  std::map<std::string, std::pair<size_t, size_t>> index;  // name -> offset, length
  std::map<std::string, double> meta;
  std::vector<float> all;
  bool has(const std::string& k) const { return index.count(k) > 0; }
  const float* host(const std::string& k) const { return all.data() + index.at(k).first; }
  size_t len(const std::string& k) const { return index.at(k).second; }
};
static Data load(const std::string& dir) {
  Data d;
  std::ifstream idx(dir + "/block.idx");
  std::string kind, name; double a, b;
  while (idx >> kind >> name) {
    if (kind == "t") { idx >> a >> b; d.index[name] = {(size_t)a, (size_t)b}; }
    else { idx >> a; d.meta[name] = a; }
  }
  std::ifstream bin(dir + "/block.bin", std::ios::binary | std::ios::ate);
  size_t bytes = bin.tellg(); bin.seekg(0);
  d.all.resize(bytes / 4);
  bin.read((char*)d.all.data(), bytes);
  return d;
}
static std::map<std::string, float*> W;  // weights on the device
static std::map<const float*, size_t> LENGTH;          // weight pointer -> elements
static std::map<const float*, __nv_bfloat16*> AS_BF16;  // weight pointer -> its bf16 copy
static float* upload(const float* h, size_t n) {
  float* p; CK(cudaMalloc(&p, n * 4)); CK(cudaMemcpy(p, h, n * 4, cudaMemcpyHostToDevice));
  LENGTH[p] = n; return p;
}
static std::string PREFIX;
static bool STAGES = false; static int FLASH_WARPS = 4; static bool ASYNC = true;
static std::map<std::string, double> STAGE_MS;
static void stageMark(const char* name) {      // sums time since the last mark into `name`
  static auto last = std::chrono::steady_clock::now();
  if (!STAGES) return;
  CK(cudaDeviceSynchronize());
  auto now = std::chrono::steady_clock::now();
  if (name) STAGE_MS[name] += std::chrono::duration<double, std::milli>(now - last).count();
  last = now;
}   // "b<k>." in a multi-block export, empty in a one-block one
static float* w(const std::string& k0) {
  std::string k = PREFIX + k0;
  auto it = W.find(k); if (it == W.end()) { fprintf(stderr, "missing %s\n", k.c_str()); exit(1); }
  return it->second;
}
static float* wOpt(const std::string& k) { auto it = W.find(PREFIX + k); return it == W.end() ? nullptr : it->second; }
static float* dalloc(size_t n) { float* p; CK(cudaMalloc(&p, n * 4)); return p; }
// Buffers kept for the life of the process, one per slot, grown when asked for more.
static float* slot(int id, size_t n) {
  static std::vector<std::pair<float*, size_t>> slots(64, {nullptr, 0});
  auto& [p, have] = slots[id];
  if (have < n) { if (p) CK(cudaFree(p)); p = dalloc(n); have = n; }
  return p;
}

// ---------------------------------------------------------------- kernels
__device__ inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }

// LayerNorm over the last axis, AF3's fast variance E[x^2]-E[x]^2. One warp a row.
__global__ void layerNormK(const float* in, float* out, size_t rows, int C,
                           const float* scale, const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* x = in + row * C;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) { float v = x[c]; s += v; ss += v * v; }
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
  float mean = s / C, inv = rsqrtf(ss / C - mean * mean + EPS);
  for (int c = lane; c < C; c += 32) out[row * C + c] = (x[c] - mean) * inv * scale[c] + offset[c];
}
static void layerNorm(const float* in, float* out, size_t rows, int C, const float* s, const float* o) {
  int wpb = 8; layerNormK<<<(unsigned)((rows + wpb - 1) / wpb), 32 * wpb>>>(in, out, rows, C, s, o);
}

__global__ void toBf16K(const float* x, __nv_bfloat16* y, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __float2bfloat16(x[i]);
}
static float* slot(int id, size_t n);
static __nv_bfloat16* bf16Of(const float* weight) {
  auto it = AS_BF16.find(weight);
  if (it != AS_BF16.end()) return it->second;
  size_t n = LENGTH.at(weight);
  __nv_bfloat16* p; CK(cudaMalloc(&p, n * 2));
  toBf16K<<<(unsigned)((n + 255) / 256), 256>>>(weight, p, n);
  return AS_BF16[weight] = p;
}
// Row-major Y[rows x out] = X[rows x in] * W, W (in,out) or (out,in) if transposed.
static void linear(const float* X, float* Y, size_t rows, int in, int out, const float* Wt,
                   bool transposed = false, float beta = 0.f) {
  const float one = 1.f;
  if (BF16) {
    size_t count = rows * (size_t)in;
    auto* xb = (__nv_bfloat16*)slot(20, (count + 1) / 2);
    toBf16K<<<(unsigned)((count + 255) / 256), 256>>>(X, xb, count);
    CB(cublasGemmEx(H, transposed ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one,
                    bf16Of(Wt), LP_CUDA, transposed ? in : out, xb, LP_CUDA, in, &beta,
                    Y, CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return;
  }
  mode(TF_LINEAR);
  if (!transposed) CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one,
                                  Wt, out, X, in, &beta, Y, out));
  else CB(cublasSgemm(H, CUBLAS_OP_T, CUBLAS_OP_N, out, (int)rows, in, &one,
                      Wt, in, X, in, &beta, Y, out));
}
__global__ void addBiasK(float* y, const float* b, size_t rows, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * C) y[i] += b[i % C];
}
static unsigned grid(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

// Triangle: a, b channel-major (c, n*n) for pair rows [r0, r0+rows).
__global__ void triGateK(const float* proj, const float* gate, const float* mask, float* a, float* b,
                         size_t r0, size_t rows, int C, size_t pairs) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  size_t local = t / C; int c = (int)(t % C);
  size_t index = r0 + local;
  float m = mask[index];
  const float* p = proj + local * 2 * C; const float* g = gate + local * 2 * C;
  a[(size_t)c * pairs + index] = p[c * 2] * m * sigm(g[c * 2]);
  b[(size_t)c * pairs + index] = p[c * 2 + 1] * m * sigm(g[c * 2 + 1]);
}
// center_norm over channels of a (c, pairs) array, for pair rows [r0, r0+rows) -> row-major chunk.
__global__ void centerNormK(const float* prod, float* out, size_t r0, size_t rows, int C, size_t pairs,
                            const float* scale, const float* offset) {
  size_t local = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (local >= rows) return;
  size_t index = r0 + local;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) { float v = prod[(size_t)c * pairs + index]; s += v; ss += v * v; }
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
  float mean = s / C, inv = rsqrtf(ss / C - mean * mean + EPS);
  for (int c = lane; c < C; c += 32)
    out[local * C + c] = (prod[(size_t)c * pairs + index] - mean) * inv * scale[c] + offset[c];
}
// pair[r0..] += proj * sigmoid(gate)
__global__ void gatedAddK(float* pair, const float* proj, const float* gate, size_t count) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) pair[i] += proj[i] * sigm(gate[i]);
}

// Grid attention helpers.
// bias[h][i][j] from rawBias[(pair) * heads + h], swapped for the transposed direction if asked.
__global__ void biasLayoutK(const float* raw, float* bias, int n, int heads, bool swap) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t pairs = (size_t)n * n;
  if (t >= pairs * heads) return;
  int h = (int)(t / pairs); size_t ij = t % pairs; size_t i = ij / n, j = ij % n;
  size_t src = swap ? (j * n + i) : ij;
  bias[t] = raw[src * heads + h];
}
// act rows [r0, r0+R) of the (possibly transposed) normalised pair.
__global__ void gatherActK(const float* norm, float* act, int n, int C, size_t r0, size_t R, bool tr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * C) return;
  int c = (int)(t % C); size_t rest = t / C; size_t j = rest % n; size_t r = r0 + rest / n;
  size_t from = tr ? (j * n + r) : (r * n + j);
  act[t] = norm[from * C + c];
}
// [r][j][h*d+e] -> [r][h][j][e]
__global__ void toHeadsK(const float* in, float* out, size_t R, int n, int heads, int d) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = R * n * heads * d;
  if (t >= total) return;
  int e = (int)(t % d); size_t rest = t / d; int j = (int)(rest % n); rest /= n;
  int h = (int)(rest % heads); size_t r = rest / heads;
  out[t] = in[((r * n + j) * heads + h) * d + e];
}
// [r][h][i][e] -> [r][i][h*d+e], times sigmoid(gate)
__global__ void fromHeadsGateK(const float* in, const float* gate, float* out, size_t R, int n,
                               int heads, int d) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = R * n * heads * d;
  if (t >= total) return;
  int e = (int)(t % d); size_t rest = t / d; int h = (int)(rest % heads); rest /= heads;
  int i = (int)(rest % n); size_t r = rest / n;
  out[t] = in[((r * heads + h) * n + i) * d + e] * sigm(gate[t]);
}
// Softmax over keys of logits[b=(r,h)][i][j] with bias[h][i][j] and the key mask.
// One block per (b, i) row.
__global__ void gridSoftmaxK(float* logits, const float* bias, const float* mask, int n, int heads,
                             size_t r0, bool tr, float scale) {
  size_t rowId = blockIdx.x;                 // over R*heads*n
  int i = (int)(rowId % n); size_t b = rowId / n; int h = (int)(b % heads); size_t r = r0 + b / heads;
  float* L = logits + rowId * n;
  const float* B = bias + ((size_t)h * n + i) * n;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    float m = mask[tr ? ((size_t)j * n + r) : (r * n + j)];
    float v = L[j] * scale + B[j] + (m > 0 ? 0.f : -1e9f);
    L[j] = v; mx = fmaxf(mx, v);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  mx = red[0];
  __syncthreads();
  float s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) { float e = expf(L[j] - mx); L[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x) L[j] *= inv;
}
// pair += out, chunk rows [r0, r0+R), transposed back for the column direction.
__global__ void addGridK(float* pair, const float* out, int n, int C, size_t r0, size_t R, bool tr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * C) return;
  int c = (int)(t % C); size_t rest = t / C; size_t j = rest % n; size_t r = r0 + rest / n;
  size_t to = tr ? (j * n + r) : (r * n + j);
  pair[to * C + c] += out[t];
}
__global__ void swigluK(const float* wide, float* gated, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = (int)(t % I);
  float g = wide[r * 2 * I + i];
  gated[t] = g * sigm(g) * wide[r * 2 * I + I + i];
}
__global__ void addK(float* y, const float* x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += x[i];
}
// pairLogits [h][ij] from flat [ij][h]
__global__ void logitsLayoutK(const float* flat, float* out, size_t pairs, int heads) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * heads) return;
  int h = (int)(t / pairs); size_t ij = t % pairs;
  out[t] = flat[ij * heads + h];
}
__global__ void singleSoftmaxK(float* logits, const float* pairLogits, const float* seqMask, int n,
                               float scale) {
  size_t rowId = blockIdx.x;                 // over heads*n
  float* L = logits + rowId * n;
  const float* B = pairLogits + rowId * n;   // [h][i][j] matches
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    float v = L[j] * scale + B[j] + 1e9f * (seqMask[j] - 1.f); L[j] = v; mx = fmaxf(mx, v);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads(); mx = red[0]; __syncthreads();
  float s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) { float e = expf(L[j] - mx); L[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x) L[j] *= inv;
}


// ---- coalesced triangle gate: a tile of 32 pair rows x 32 channels through shared memory.
template <typename T>
__global__ void triGateTiledK(const float* proj, const float* gate, const float* mask, T* a, T* b,
                              size_t r0, size_t rows, int C, size_t pairs) {
  __shared__ float A[32][33], B[32][33];
  size_t row0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;          // 32 x 8
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = row0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (local < rows && c < C) {
      float m = mask[r0 + local];
      const float* p = proj + local * 2 * C; const float* g = gate + local * 2 * C;
      va = p[c * 2] * m * sigm(g[c * 2]);
      vb = p[c * 2 + 1] * m * sigm(g[c * 2 + 1]);
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t local = row0 + tx; int c = c0 + cy;
    if (local < rows && c < C) {
      a[(size_t)c * pairs + r0 + local] = (T)A[tx][cy];
      b[(size_t)c * pairs + r0 + local] = (T)B[tx][cy];
    }
  }
}
// ---- coalesced centre norm: 32 pair indices, all C channels staged (C <= 256).
__global__ void centerNormTiledK(const float* prod, float* out, size_t r0, size_t rows, int C,
                                 size_t pairs, const float* scale, const float* offset) {
  extern __shared__ float T[];                     // C x 33
  __shared__ float mean[32], inv[32];
  size_t i0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;   // 32 x 8
  for (int c = ty; c < C; c += 8) {
    size_t local = i0 + tx;
    T[c * 33 + tx] = local < rows ? prod[(size_t)c * pairs + r0 + local] : 0.f;
  }
  __syncthreads();
  if (ty == 0) {
    float s = 0, ss = 0;
    for (int c = 0; c < C; ++c) { float v = T[c * 33 + tx]; s += v; ss += v * v; }
    float m = s / C; mean[tx] = m; inv[tx] = rsqrtf(ss / C - m * m + EPS);
  }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = i0 + ry; if (local >= rows) continue;
    for (int c = tx; c < C; c += 32)
      out[local * C + c] = (T[c * 33 + ry] - mean[ry]) * inv[ry] * scale[c] + offset[c];
  }
}
// ---- fused grid attention: one thread a query row, 64 queries a block, key tiles of 64 in
// shared memory, online softmax once per tile. q4/k4/v4 [b][n][D], out [b][n][D].
template <int D>
__global__ void flashGridK(const float* q4, const float* k4, const float* v4, const float* bias,
                           const float* mask, float* out, int n, int heads, size_t r0, bool tr,
                           float scale) {
  constexpr int BQ = 64, BK = 64;
  __shared__ float Ks[BK][D + 1], Vs[BK][D + 1], Ms[BK];
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t r = r0 + b / heads;
  int i = blockIdx.x * BQ + threadIdx.x;
  bool live = i < n;
  float q[D], o[D];
  const float* qp = q4 + (b * n + (live ? i : 0)) * D;
  for (int e = 0; e < D; ++e) { q[e] = qp[e] * scale; o[e] = 0.f; }
  float m = -INFINITY, l = 0.f;
  const float* brow = bias + ((size_t)h * n + (live ? i : 0)) * n;
  for (int j0 = 0; j0 < n; j0 += BK) {
    __syncthreads();
    for (int t = threadIdx.x; t < BK * D; t += BQ) {
      int jj = t / D, e = t % D; int j = j0 + jj;
      Ks[jj][e] = j < n ? k4[(b * n + j) * D + e] : 0.f;
      Vs[jj][e] = j < n ? v4[(b * n + j) * D + e] : 0.f;
    }
    if (threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Ms[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f)
                              : -INFINITY;
    }
    __syncthreads();
    float sv[BK]; float tmax = -INFINITY;
#pragma unroll 8
    for (int jj = 0; jj < BK; ++jj) {
      float dot = 0.f;
#pragma unroll
      for (int e = 0; e < D; ++e) dot += q[e] * Ks[jj][e];
      int j = j0 + jj;
      float v = dot + (j < n ? brow[j] : 0.f) + Ms[jj];
      sv[jj] = v; tmax = fmaxf(tmax, v);
    }
    float mNew = fmaxf(m, tmax);
    float corr = expf(m - mNew);
    l *= corr;
#pragma unroll
    for (int e = 0; e < D; ++e) o[e] *= corr;
#pragma unroll 8
    for (int jj = 0; jj < BK; ++jj) {
      float pj = expf(sv[jj] - mNew); l += pj;
#pragma unroll
      for (int e = 0; e < D; ++e) o[e] += pj * Vs[jj][e];
    }
    m = mNew;
  }
  if (live) {
    float invl = 1.f / l;
    float* op = out + (b * n + i) * D;
    for (int e = 0; e < D; ++e) op[e] = o[e] * invl;
  }
}


// ---- fused grid attention on the tensor cores (TF32 WMMA, 16x16x8), D = 32.
// 128 threads = 4 warps, each warp 16 queries; key tiles of 64 staged in shared memory.
// S = Q K^T into shared memory, bias + mask + online softmax by lane pairs, then O += P V
// with O itself held in shared memory and loaded as the accumulator.
__global__ void flashGridTC(const float* qkvg, const float* bias, const float* mask, float* out,
                            int n, int heads, size_t r0, bool tr, float scale) {
  using namespace nvcuda;
  constexpr int D = 32, BQ = 64, BK = 32, LD = D + 4, LS = BK + 4;
  __shared__ __align__(32) float Qs[BQ * LD];
  __shared__ __align__(32) float Ks[BK * LD];
  __shared__ __align__(32) float Vs[BK * LD];
  __shared__ __align__(32) float Ss[4][16 * LS];
  __shared__ __align__(32) float Os[4][16 * LD];
  __shared__ float Ms[BK], rowM[4][16], rowL[4][16], rowC[4][16];
  __shared__ float Bs[BQ][BK + 1];
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t rl = b / heads, r = r0 + rl;
  const int W = heads * D, W4 = 4 * W;
  auto at = [&](int i, int role, int e) { return qkvg[(rl * n + i) * W4 + role * W + h * D + e]; };
  int q0 = blockIdx.x * BQ;
  for (int t = threadIdx.x; t < BQ * D; t += 128) {
    int qi = t / D, e = t % D; int i = q0 + qi;
    Qs[qi * LD + e] = i < n ? at(i, 0, e) * scale : 0.f;
  }
  for (int t = lane; t < 16 * LD; t += 32) Os[warp][t] = 0.f;
  if (lane < 16) { rowM[warp][lane] = -INFINITY; rowL[warp][lane] = 0.f; }
  int myRow = lane >> 1, half = lane & 1;
  int iq = q0 + warp * 16 + myRow;
  for (int j0 = 0; j0 < n; j0 += BK) {
    __syncthreads();
    for (int t = threadIdx.x; t < BK * D; t += 128) {
      int jj = t / D, e = t % D; int j = j0 + jj;
      Ks[jj * LD + e] = j < n ? at(j, 1, e) : 0.f;
      Vs[jj * LD + e] = j < n ? at(j, 2, e) : 0.f;
    }
    // the bias tile, read along keys so neighbouring threads read neighbouring floats
    for (int t = threadIdx.x; t < BQ * BK; t += 128) {
      int qi = t / BK, jj = t % BK; int i = q0 + qi, j = j0 + jj;
      Bs[qi][jj] = (i < n && j < n) ? bias[((size_t)h * n + i) * n + j] : 0.f;
    }
    if (threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Ms[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f)
                              : -INFINITY;
    }
    __syncthreads();
    // S (16 x 64) = Q_w (16 x 32) K^T
    for (int jt = 0; jt < BK / 16; ++jt) {
      wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
      wmma::fill_fragment(acc, 0.f);
      for (int k = 0; k < D / 8; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, Qs + (warp * 16) * LD + k * 8, LD);
        wmma::load_matrix_sync(fb, Ks + (jt * 16) * LD + k * 8, LD);
        for (int x = 0; x < fa.num_elements; ++x) fa.x[x] = wmma::__float_to_tf32(fa.x[x]);
        for (int x = 0; x < fb.num_elements; ++x) fb.x[x] = wmma::__float_to_tf32(fb.x[x]);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(Ss[warp] + jt * 16, acc, LS, wmma::mem_row_major);
    }
    __syncwarp();
    // bias, mask, online softmax: two lanes a row, 32 keys each
    float* srow = Ss[warp] + myRow * LS + half * (BK / 2);
    float tmax = -INFINITY;
    for (int x = 0; x < BK / 2; ++x) {
      int jj = half * (BK / 2) + x, j = j0 + jj;
      float v = srow[x] + Bs[warp * 16 + myRow][jj] + Ms[jj];
      srow[x] = v; tmax = fmaxf(tmax, v);
    }
    tmax = fmaxf(tmax, __shfl_xor_sync(~0u, tmax, 1));
    float mOld = rowM[warp][myRow], mNew = fmaxf(mOld, tmax);
    float sum = 0.f;
    for (int x = 0; x < BK / 2; ++x) { float pj = __expf(srow[x] - mNew); srow[x] = pj; sum += pj; }
    sum += __shfl_xor_sync(~0u, sum, 1);
    float corr = __expf(mOld - mNew);
    __syncwarp();
    if (half == 0) { rowM[warp][myRow] = mNew; rowL[warp][myRow] = rowL[warp][myRow] * corr + sum;
                     rowC[warp][myRow] = corr; }
    __syncwarp();
    for (int e = half * 16; e < half * 16 + 16; ++e) Os[warp][myRow * LD + e] *= rowC[warp][myRow];
    __syncwarp();
    // O (16 x 32) += P (16 x 64) V (64 x 32)
    for (int et = 0; et < D / 16; ++et) {
      wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
      wmma::load_matrix_sync(acc, Os[warp] + et * 16, LD, wmma::mem_row_major);
      for (int k = 0; k < BK / 8; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, Ss[warp] + k * 8, LS);
        wmma::load_matrix_sync(fb, Vs + (k * 8) * LD + et * 16, LD);
        for (int x = 0; x < fa.num_elements; ++x) fa.x[x] = wmma::__float_to_tf32(fa.x[x]);
        for (int x = 0; x < fb.num_elements; ++x) fb.x[x] = wmma::__float_to_tf32(fb.x[x]);
        wmma::mma_sync(acc, fa, fb, acc);
      }
      wmma::store_matrix_sync(Os[warp] + et * 16, acc, LD, wmma::mem_row_major);
    }
    __syncwarp();
  }
  if (iq < n) {
    float invl = 1.f / rowL[warp][myRow];
    float* op = out + (rl * n + iq) * W + h * D;
    for (int e = half * 16; e < half * 16 + 16; ++e)
      op[e] = Os[warp][myRow * LD + e] * invl * sigm(at(iq, 3, e));
  }
}


// ---- fused grid attention, bf16 tensor cores (16x16x16, f32 accumulate), D = 32.
// 256 threads = 8 warps x 16 queries = 128 queries a block, key tiles of 64.
// qkvg is bf16 [prs][4W] straight out of the projection GEMM; bias is bf16 [h][i][j].
// Shared memory (dynamic): K, V tiles bf16; per warp S f32, P bf16, O f32; the bias tile bf16.
__global__ void flashGridBF(const __nv_bfloat16* qkvg, const __nv_bfloat16* bias, const float* mask,
                            float* out, int n, int heads, size_t r0, bool tr, float scale) {
  using namespace nvcuda;
  constexpr int D = 32, BQ = 128, BK = 64, LDK = D + 8, LS = BK + 4, LP = BK + 8, LO = D + 4,
                LB = BK + 8;
  extern __shared__ __align__(32) unsigned char smem[];
  __nv_bfloat16* Ks = (__nv_bfloat16*)smem;                       // BK x LDK
  __nv_bfloat16* Vs = Ks + BK * LDK;                              // BK x LDK
  float* Ss = (float*)(Vs + BK * LDK);                            // 8 x 16 x LS
  __nv_bfloat16* Ps = (__nv_bfloat16*)(Ss + 8 * 16 * LS);         // 8 x 16 x LP
  float* Os = (float*)(Ps + 8 * 16 * LP);                         // 8 x 16 x LO
  __nv_bfloat16* Bs = (__nv_bfloat16*)(Os + 8 * 16 * LO);         // BQ x LB
  float* Ms = (float*)(Bs + BQ * LB);                             // BK
  float* rowM = Ms + BK; float* rowL = rowM + BQ; float* rowC = rowL + BQ;
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t rl = b / heads, r = r0 + rl;
  const int W = heads * D, W4 = 4 * W;
  int q0 = blockIdx.x * BQ, wq = q0 + warp * 16;
  const __nv_bfloat16* base = qkvg + rl * (size_t)n * W4 + h * D;
  // Q in registers: two k-slices of 16. Rows past n read padding the caller allocated.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fq[2];
  for (int k = 0; k < 2; ++k) wmma::load_matrix_sync(fq[k], base + (size_t)wq * W4 + k * 16, W4);
  float* So = Ss + warp * 16 * LS; __nv_bfloat16* Po = Ps + warp * 16 * LP; float* Oo = Os + warp * 16 * LO;
  for (int t = lane; t < 16 * LO; t += 32) Oo[t] = 0.f;
  if (lane < 16) { rowM[warp * 16 + lane] = -INFINITY; rowL[warp * 16 + lane] = 0.f; }
  int myRow = lane >> 1, half = lane & 1, qr = warp * 16 + myRow, iq = q0 + qr;
  for (int j0 = 0; j0 < n; j0 += BK) {
    __syncthreads();
    // K and V tiles: 64 keys x 32 dims, 8 bf16 (16 bytes) a load
    for (int t = threadIdx.x; t < BK * (D / 8) * 2; t += 256) {
      int which = t / (BK * (D / 8)), u = t % (BK * (D / 8)); int jj = u / (D / 8), e8 = (u % (D / 8)) * 8;
      int j = j0 + jj;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (j < n) v = *(const uint4*)(base + (size_t)j * W4 + (which + 1) * W + e8);
      *(uint4*)((which ? Vs : Ks) + jj * LDK + e8) = v;
    }
    for (int t = threadIdx.x; t < BQ * BK; t += 256) {
      int qi = t / BK, jj = t % BK; int i = q0 + qi, j = j0 + jj;
      Bs[qi * LB + jj] = (i < n && j < n) ? bias[((size_t)h * n + i) * n + j] : __float2bfloat16(0.f);
    }
    if (threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Ms[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f)
                              : -INFINITY;
    }
    __syncthreads();
    for (int jt = 0; jt < BK / 16; ++jt) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
      for (int k = 0; k < 2; ++k) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fk;
        wmma::load_matrix_sync(fk, Ks + (jt * 16) * LDK + k * 16, LDK);
        wmma::mma_sync(acc, fq[k], fk, acc);
      }
      wmma::store_matrix_sync(So + jt * 16, acc, LS, wmma::mem_row_major);
    }
    __syncwarp();
    float* srow = So + myRow * LS + half * (BK / 2);
    float tmax = -INFINITY;
    for (int x = 0; x < BK / 2; ++x) {
      int jj = half * (BK / 2) + x;
      float v = srow[x] * scale + __bfloat162float(Bs[qr * LB + jj]) + Ms[jj];
      srow[x] = v; tmax = fmaxf(tmax, v);
    }
    tmax = fmaxf(tmax, __shfl_xor_sync(~0u, tmax, 1));
    float mOld = rowM[qr], mNew = fmaxf(mOld, tmax), sum = 0.f;
    __nv_bfloat16* prow = Po + myRow * LP + half * (BK / 2);
    for (int x = 0; x < BK / 2; ++x) {
      float pj = __expf(srow[x] - mNew); sum += pj; prow[x] = __float2bfloat16(pj);
    }
    sum += __shfl_xor_sync(~0u, sum, 1);
    float corr = __expf(mOld - mNew);
    __syncwarp();
    if (half == 0) { rowM[qr] = mNew; rowL[qr] = rowL[qr] * corr + sum; rowC[qr] = corr; }
    __syncwarp();
    for (int e = half * 16; e < half * 16 + 16; ++e) Oo[myRow * LO + e] *= rowC[qr];
    __syncwarp();
    for (int et = 0; et < D / 16; ++et) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::load_matrix_sync(acc, Oo + et * 16, LO, wmma::mem_row_major);
      for (int k = 0; k < BK / 16; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fp;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fv;
        wmma::load_matrix_sync(fp, Po + k * 16, LP);
        wmma::load_matrix_sync(fv, Vs + (k * 16) * LDK + et * 16, LDK);
        wmma::mma_sync(acc, fp, fv, acc);
      }
      wmma::store_matrix_sync(Oo + et * 16, acc, LO, wmma::mem_row_major);
    }
    __syncwarp();
  }
  if (iq < n) {
    float invl = 1.f / rowL[qr];
    float* op = out + (rl * n + iq) * W + h * D;
    const __nv_bfloat16* gp = base + (size_t)iq * W4 + 3 * W;
    for (int e = half * 16; e < half * 16 + 16; ++e)
      op[e] = Oo[myRow * LO + e] * invl * sigm(__bfloat162float(gp[e]));
  }
}
static size_t flashGridBFSmem() {
  constexpr int D = 32, BQ = 128, BK = 64, LDK = D + 8, LS = BK + 4, LP = BK + 8, LO = D + 4, LB = BK + 8;
  return 2 * BK * LDK * 2 + 8 * 16 * LS * 4 + 8 * 16 * LP * 2 + 8 * 16 * LO * 4 + BQ * LB * 2
         + (BK + 3 * BQ) * 4;
}
__global__ void toBf16Bias(const float* x, __nv_bfloat16* y, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __float2bfloat16(x[i]);
}


// ---- FlashAttention-2 style grid attention: mma.sync m16n8k16 bf16, f32 accumulate, D = 32.
// 4 warps x 16 queries = 64 a block; key tiles of 64. S, P and O never leave registers:
// the softmax reduces over the 4 threads of a quad, and P's accumulators ARE the A
// operand of P V (two adjacent n-tiles of S make one k-slice).
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32." LP_MMA ".f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
template <typename TO, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) flashGridMMA(const __nv_bfloat16* qkvg, const __nv_bfloat16* bias,
                             const float* mask, TO* out, int n, int heads, size_t r0, bool tr,
                             float scale) {
  constexpr int D = 32, BQ = 16 * WARPS, BK = 64, LK = D + 8, LV = BK + 2, NT = WARPS * 32;
  __shared__ __align__(16) __nv_bfloat16 Ks[BK * LK];
  __shared__ __align__(16) __nv_bfloat16 Vt[D * LV];
  __shared__ float Ms[BK];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t rl = b / heads, r = r0 + rl;
  const int W = heads * D, W4 = 4 * W;
  const __nv_bfloat16* base = qkvg + rl * (size_t)n * W4 + h * D;
  int i0 = blockIdx.x * BQ + warp * 16 + g, i1 = i0 + 8;       // this thread's two query rows
  auto q2 = [&](int i, int e) -> uint32_t {                     // two adjacent q elements
    if (i >= n) return 0u;
    return *reinterpret_cast<const uint32_t*>(base + (size_t)i * W4 + e);
  };
  uint32_t qa[2][4];
  for (int ks = 0; ks < 2; ++ks) {
    int e = ks * 16 + tig * 2;
    qa[ks][0] = q2(i0, e); qa[ks][1] = q2(i1, e); qa[ks][2] = q2(i0, e + 8); qa[ks][3] = q2(i1, e + 8);
  }
  float o[D / 8][4] = {};
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const __nv_bfloat16* b0row = bias + ((size_t)h * n + (i0 < n ? i0 : 0)) * n;
  const __nv_bfloat16* b1row = bias + ((size_t)h * n + (i1 < n ? i1 : 0)) * n;
  for (int j0 = 0; j0 < n; j0 += BK) {
    __syncthreads();
    for (int t = threadIdx.x; t < BK * (D / 8); t += NT) {     // K rows, 16 bytes a load
      int jj = t / (D / 8), e8 = (t % (D / 8)) * 8, j = j0 + jj;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (j < n) v = *reinterpret_cast<const uint4*>(base + (size_t)j * W4 + W + e8);
      *reinterpret_cast<uint4*>(Ks + jj * LK + e8) = v;
    }
    for (int t = threadIdx.x; t < BK * D; t += NT) {           // V, transposed to [e][j]
      int jj = t / D, e = t % D, j = j0 + jj;
      Vt[e * LV + jj] = j < n ? base[(size_t)j * W4 + 2 * W + e] : __float2bfloat16(0.f);
    }
    if (threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Ms[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f)
                              : -INFINITY;
    }
    __syncthreads();
    float sv[BK / 8][4];
    for (int nt = 0; nt < BK / 8; ++nt) {
      sv[nt][0] = sv[nt][1] = sv[nt][2] = sv[nt][3] = 0.f;
      for (int ks = 0; ks < 2; ++ks) {
        const __nv_bfloat16* kp = Ks + (nt * 8 + g) * LK + ks * 16 + tig * 2;
        mma16816(sv[nt], qa[ks], *reinterpret_cast<const uint32_t*>(kp),
                 *reinterpret_cast<const uint32_t*>(kp + 8));
      }
    }
    float t0 = -INFINITY, t1 = -INFINITY;
    for (int nt = 0; nt < BK / 8; ++nt) {
      int jj = nt * 8 + tig * 2, j = j0 + jj;
      float bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;
      if (j + 1 < n) {
        __nv_bfloat162 u = *reinterpret_cast<const __nv_bfloat162*>(b0row + j);
        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(b1row + j);
        bx0 = __low2float(u); by0 = __high2float(u); bx1 = __low2float(v); by1 = __high2float(v);
      } else if (j < n) { bx0 = __bfloat162float(b0row[j]); bx1 = __bfloat162float(b1row[j]); }
      sv[nt][0] = sv[nt][0] * scale + bx0 + Ms[jj];
      sv[nt][1] = sv[nt][1] * scale + by0 + Ms[jj + 1];
      sv[nt][2] = sv[nt][2] * scale + bx1 + Ms[jj];
      sv[nt][3] = sv[nt][3] * scale + by1 + Ms[jj + 1];
      t0 = fmaxf(t0, fmaxf(sv[nt][0], sv[nt][1])); t1 = fmaxf(t1, fmaxf(sv[nt][2], sv[nt][3]));
    }
    t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
    t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
    float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);
    float c0 = __expf(m0 - n0), c1 = __expf(m1 - n1);
    m0 = n0; m1 = n1; l0 *= c0; l1 *= c1;
    for (int et = 0; et < D / 8; ++et) { o[et][0] *= c0; o[et][1] *= c0; o[et][2] *= c1; o[et][3] *= c1; }
    for (int nt = 0; nt < BK / 8; ++nt) {
      sv[nt][0] = __expf(sv[nt][0] - n0); sv[nt][1] = __expf(sv[nt][1] - n0);
      sv[nt][2] = __expf(sv[nt][2] - n1); sv[nt][3] = __expf(sv[nt][3] - n1);
      l0 += sv[nt][0] + sv[nt][1]; l1 += sv[nt][2] + sv[nt][3];
    }
    for (int t = 0; t < BK / 16; ++t) {
      uint32_t pa[4] = { pack2(sv[2 * t][0], sv[2 * t][1]), pack2(sv[2 * t][2], sv[2 * t][3]),
                         pack2(sv[2 * t + 1][0], sv[2 * t + 1][1]), pack2(sv[2 * t + 1][2], sv[2 * t + 1][3]) };
      for (int et = 0; et < D / 8; ++et) {
        const __nv_bfloat16* vp = Vt + (et * 8 + g) * LV + t * 16 + tig * 2;
        mma16816(o[et], pa, *reinterpret_cast<const uint32_t*>(vp), *reinterpret_cast<const uint32_t*>(vp + 8));
      }
    }
  }
  l0 += __shfl_xor_sync(~0u, l0, 1); l0 += __shfl_xor_sync(~0u, l0, 2);
  l1 += __shfl_xor_sync(~0u, l1, 1); l1 += __shfl_xor_sync(~0u, l1, 2);
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    if (i0 < n) {
      const __nv_bfloat16* gp = base + (size_t)i0 * W4 + 3 * W + e;
      TO* op = out + (rl * n + i0) * W + h * D + e;
      op[0] = (TO)(o[et][0] / l0 * sigm(__bfloat162float(gp[0])));
      op[1] = (TO)(o[et][1] / l0 * sigm(__bfloat162float(gp[1])));
    }
    if (i1 < n) {
      const __nv_bfloat16* gp = base + (size_t)i1 * W4 + 3 * W + e;
      TO* op = out + (rl * n + i1) * W + h * D + e;
      op[0] = (TO)(o[et][2] / l1 * sigm(__bfloat162float(gp[0])));
      op[1] = (TO)(o[et][3] / l1 * sigm(__bfloat162float(gp[1])));
    }
  }
}


// ---- the bf16 path: kernels that write bf16 so no separate conversion pass is needed.
template <typename TO>
__global__ void layerNormKT(const float* in, TO* out, size_t rows, int C, const float* scale,
                            const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* x = in + row * C;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) { float v = x[c]; s += v; ss += v * v; }
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
  float mean = s / C, inv = rsqrtf(ss / C - mean * mean + EPS);
  for (int c = lane; c < C; c += 32) out[row * C + c] = (TO)((x[c] - mean) * inv * scale[c] + offset[c]);
}
template <typename TO>
static void layerNormT(const float* in, TO* out, size_t rows, int C, const float* s, const float* o) {
  layerNormKT<TO><<<(unsigned)((rows + 7) / 8), 256>>>(in, out, rows, C, s, o);
}
// Row-major Y = X W with bf16 X and W, f32 accumulation, Y in `yt` (f32 or bf16).
static void gemmBF(const __nv_bfloat16* X, void* Y, cudaDataType yt, size_t rows, int in, int out,
                   const float* Wt, bool transposed = false, float beta = 0.f) {
  const float one = 1.f;
  CB(cublasGemmEx(H, transposed ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, out, (int)rows, in, &one,
                  bf16Of(Wt), LP_CUDA, transposed ? in : out, X, LP_CUDA, in, &beta,
                  Y, yt, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
// a, b from one bf16 [rows][4C] = [projection (2C) | gate (2C)], interleaved split, channel-major.
__global__ void triGateBF(const __nv_bfloat16* pg, const float* mask, __nv_bfloat16* a,
                          __nv_bfloat16* b, size_t r0, size_t rows, int C, size_t pairs) {
  __shared__ float A[32][33], B[32][33];
  size_t row0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = row0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (local < rows && c < C) {
      float m = mask[r0 + local];
      const __nv_bfloat16* p = pg + local * 4 * C; const __nv_bfloat16* g = p + 2 * C;
      va = __bfloat162float(p[c * 2]) * m * sigm(__bfloat162float(g[c * 2]));
      vb = __bfloat162float(p[c * 2 + 1]) * m * sigm(__bfloat162float(g[c * 2 + 1]));
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t local = row0 + tx; int c = c0 + cy;
    if (local < rows && c < C) {
      a[(size_t)c * pairs + r0 + local] = __float2bfloat16(A[tx][cy]);
      b[(size_t)c * pairs + r0 + local] = __float2bfloat16(B[tx][cy]);
    }
  }
}
template <typename TO>
__global__ void centerNormKT(const float* prod, TO* out, size_t r0, size_t rows, int C, size_t pairs,
                             const float* scale, const float* offset) {
  extern __shared__ float T[];
  __shared__ float mean[32], inv[32];
  size_t i0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  for (int c = ty; c < C; c += 8) {
    size_t local = i0 + tx;
    T[c * 33 + tx] = local < rows ? prod[(size_t)c * pairs + r0 + local] : 0.f;
  }
  __syncthreads();
  if (ty == 0) {
    float s = 0, ss = 0;
    for (int c = 0; c < C; ++c) { float v = T[c * 33 + tx]; s += v; ss += v * v; }
    float m = s / C; mean[tx] = m; inv[tx] = rsqrtf(ss / C - m * m + EPS);
  }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = i0 + ry; if (local >= rows) continue;
    for (int c = tx; c < C; c += 32)
      out[local * C + c] = (TO)((T[c * 33 + ry] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}
__global__ void swigluBF(const __nv_bfloat16* wide, __nv_bfloat16* gated, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = (int)(t % I);
  float g = __bfloat162float(wide[r * 2 * I + i]);
  gated[t] = __float2bfloat16(g * sigm(g) * __bfloat162float(wide[r * 2 * I + I + i]));
}
__global__ void gatherActBF(const __nv_bfloat16* norm, __nv_bfloat16* act, int n, int C, size_t r0,
                            size_t R) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * C) return;
  int c = (int)(t % C); size_t rest = t / C; size_t j = rest % n; size_t r = r0 + rest / n;
  act[t] = norm[(j * n + r) * C + c];
}
__global__ void biasLayoutBF(const float* raw, __nv_bfloat16* bias, int n, int heads, bool swap) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t pairs = (size_t)n * n;
  if (t >= pairs * heads) return;
  int h = (int)(t / pairs); size_t ij = t % pairs; size_t i = ij / n, j = ij % n;
  bias[t] = __float2bfloat16(raw[(swap ? (j * n + i) : ij) * heads + h]);
}

// ---------------------------------------------------------------- the block
struct Shape { int n, C, S, gh, gd, sh, sd; bool swap, divide; };
static size_t CHUNK_ELEMS = (size_t)64 << 20;   // ~256 MiB of f32 per chunk tensor

// Scratch, sized once for the largest chunk.
struct Scratch { float *t0, *t1, *t2, *t3; size_t elems; };

static void triangle(float* pair, const float* mask, const Shape& s, const std::string& pre,
                     bool outgoing, float* a, float* b, float* prod, Scratch& sc) {
  int n = s.n, C = s.C; size_t pairs = (size_t)n * n;
  size_t rowsPer = std::max<size_t>(1, sc.elems / (2 * C));
  // pass 1: a, b for every pair row
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    layerNorm(pair + r0 * C, sc.t0, rows, C, w(pre + ".leftNormInputScale"), w(pre + ".leftNormInputOffset"));
    linear(sc.t0, sc.t1, rows, C, 2 * C, w(pre + ".projection"));
    linear(sc.t0, sc.t2, rows, C, 2 * C, w(pre + ".gate"));
    dim3 gg((unsigned)((rows + 31) / 32), (C + 31) / 32), bb(32, 8);
    if (BF16) triGateTiledK<__nv_bfloat16><<<gg, bb>>>(sc.t1, sc.t2, mask, (__nv_bfloat16*)a,
                                                        (__nv_bfloat16*)b, r0, rows, C, pairs);
    else triGateTiledK<float><<<gg, bb>>>(sc.t1, sc.t2, mask, a, b, r0, rows, C, pairs);
  }
  // the contraction, one n x n GEMM per channel
  float alpha = s.divide ? 1.f / n : 1.f, zero = 0.f;
  if (BF16) {
    // the same two products with bf16 operands and an f32 accumulator and result
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, b, LP_CUDA, n,
        pairs, a, LP_CUDA, n, pairs, &zero, prod, CUDA_R_32F, n, pairs, C, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, a, LP_CUDA, n,
        pairs, b, LP_CUDA, n, pairs, &zero, prod, CUDA_R_32F, n, pairs, C, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
  mode(TF_TRI);
  if (outgoing)   // P = A B^T  ->  col-major P^T = arrB^T arrA
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, b, n, pairs,
                                 a, n, pairs, &zero, prod, n, pairs, C));
  else            // P = B^T A  ->  col-major P^T = arrA arrB^T
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, a, n, pairs,
                                 b, n, pairs, &zero, prod, n, pairs, C));
  }
  // pass 2: centre norm, output projection, gate from the (unchanged) rows, residual
  rowsPer = std::max<size_t>(1, sc.elems / C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    centerNormTiledK<<<(unsigned)((rows + 31) / 32), dim3(32, 8), C * 33 * 4>>>(prod, sc.t0, r0, rows,
      C, pairs, w(pre + ".centerNormScale"), w(pre + ".centerNormOffset"));
    linear(sc.t0, sc.t1, rows, C, C, w(pre + ".outputProjection"));
    layerNorm(pair + r0 * C, sc.t0, rows, C, w(pre + ".leftNormInputScale"), w(pre + ".leftNormInputOffset"));
    linear(sc.t0, sc.t2, rows, C, C, w(pre + ".gatingLinear"));
    gatedAddK<<<grid(rows * C), 256>>>(pair + r0 * C, sc.t1, sc.t2, rows * C);
  }
}

static void gridAttention(float* pair, const float* mask, const Shape& s, const std::string& pre,
                          bool tr, float* norm, float* bias, float* logits, size_t logitsElems,
                          Scratch& sc) {
  int n = s.n, C = s.C, heads = s.gh, d = s.gd, Wd = heads * d; size_t pairs = (size_t)n * n;
  layerNorm(pair, norm, pairs, C, w(pre + ".actNormScale"), w(pre + ".actNormOffset"));
  linear(norm, sc.t0, pairs, C, heads, w(pre + ".pairBiasProjection"));   // pairs*heads fits t0
  biasLayoutK<<<grid(pairs * heads), 256>>>(sc.t0, bias, n, heads, tr && s.swap);
  // rows of the attention at a time: limited by the chunk tensors and the logits
  size_t R = std::max<size_t>(1, std::min(sc.elems / ((size_t)n * std::max(C, Wd)),
                                          FLASH ? (size_t)n : logitsElems / ((size_t)heads * n * n)));
  // four chunk tensors: act(t0) q(t1) k(t2) v(t3), plus heads-major copies in logits' tail? keep simple:
  size_t need = std::min(sc.elems, R * (size_t)n * Wd);
  float *q4 = slot(0, need), *k4 = slot(1, need), *v4 = slot(2, need), *g = slot(3, need);
  float scale = 1.f / sqrtf((float)d), one = 1.f, zero = 0.f;
  for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
    size_t rows = std::min(R, (size_t)n - r0), prs = rows * n;
    if (FUSED) {
      const float* act = norm + r0 * n * C;
      if (tr) { gatherActK<<<grid(prs * C), 256>>>(norm, sc.t0, n, C, r0, rows, tr); act = sc.t0; }
      if (BF16 && d == 32) {
        // the projection writes bf16 directly; 128 rows of padding for the last query block
        auto* qkvgB = (__nv_bfloat16*)slot(16, ((prs + 128) * 4 * Wd + 1) / 2);
        auto* xb = (__nv_bfloat16*)slot(20, (prs * C + 1) / 2);
        toBf16K<<<grid(prs * C), 256>>>(act, xb, prs * C);
        const float one = 1.f, zero = 0.f;
        CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, 4 * Wd, (int)prs, C, &one, bf16Of(w(pre + ".qkvg")),
                        LP_CUDA, 4 * Wd, xb, LP_CUDA, C, &zero, qkvgB, LP_CUDA, 4 * Wd,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if (wOpt(pre + ".gatingQueryBias")) { fprintf(stderr, "bf16 grid: gate bias not wired\n"); exit(1); }
        auto* biasB = (__nv_bfloat16*)slot(17, ((size_t)heads * n * n + 1) / 2);
        if (r0 == 0) toBf16Bias<<<grid((size_t)heads * n * n), 256>>>(bias, biasB, (size_t)heads * n * n);
        static bool attr = false;
        if (!attr) { CK(cudaFuncSetAttribute(flashGridBF, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             (int)flashGridBFSmem())); attr = true; }
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
        if (MMA) flashGridMMA<float, 4><<<dim3((n + 63) / 64, (unsigned)(rows * heads)), 128>>>(
          qkvgB, biasB, mask, sc.t1, n, heads, r0, tr, scale);
        else flashGridBF<<<dim3((n + 127) / 128, (unsigned)(rows * heads)), 256, flashGridBFSmem()>>>(
          qkvgB, biasB, mask, sc.t1, n, heads, r0, tr, scale);
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
        FLASH_MS += ms; cudaEventDestroy(e0); cudaEventDestroy(e1);
      } else {
      float* qkvg = slot(15, prs * 4 * Wd);
      linear(act, qkvg, prs, C, 4 * Wd, w(pre + ".qkvg"));
      if (float* gb = wOpt(pre + ".gatingQueryBias")) addBiasK<<<grid(prs * 4 * Wd), 256>>>(qkvg, wOpt(pre + ".qkvgBias"), prs, 4 * Wd);
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
      flashGridTC<<<dim3((n + 63) / 64, (unsigned)(rows * heads)), 128>>>(qkvg, bias, mask, sc.t1, n,
                                                                          heads, r0, tr, scale);
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
      FLASH_MS += ms; cudaEventDestroy(e0); cudaEventDestroy(e1);
      }
      linear(sc.t1, sc.t2, prs, Wd, C, w(pre + ".outputProjection"));
      if (float* ob = wOpt(pre + ".outputProjectionBias")) addBiasK<<<grid(prs * C), 256>>>(sc.t2, ob, prs, C);
      addGridK<<<grid(prs * C), 256>>>(pair, sc.t2, n, C, r0, rows, tr);
      continue;
    }
    gatherActK<<<grid(prs * C), 256>>>(norm, sc.t0, n, C, r0, rows, tr);
    linear(sc.t0, sc.t1, prs, C, Wd, w(pre + ".qProjection"), true);
    linear(sc.t0, sc.t2, prs, C, Wd, w(pre + ".kProjection"), true);
    linear(sc.t0, sc.t3, prs, C, Wd, w(pre + ".vProjection"), false);
    linear(sc.t0, g, prs, C, Wd, w(pre + ".gatingQuery"), true);
    if (float* gb = wOpt(pre + ".gatingQueryBias")) addBiasK<<<grid(prs * Wd), 256>>>(g, gb, prs, Wd);
    toHeadsK<<<grid(prs * Wd), 256>>>(sc.t1, q4, rows, n, heads, d);
    toHeadsK<<<grid(prs * Wd), 256>>>(sc.t2, k4, rows, n, heads, d);
    toHeadsK<<<grid(prs * Wd), 256>>>(sc.t3, v4, rows, n, heads, d);
    int batch = (int)(rows * heads);
    if (FLASH && d == 32) {
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
      flashGridK<32><<<dim3((n + 63) / 64, batch), 64>>>(q4, k4, v4, bias, mask, sc.t3, n, heads,
                                                         r0, tr, scale);
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
      FLASH_MS += ms; cudaEventDestroy(e0); cudaEventDestroy(e1);
      fromHeadsGateK<<<grid(prs * Wd), 256>>>(sc.t3, g, sc.t1, rows, n, heads, d);
    } else {
    mode(TF_ATT);
    // L = Q K^T  -> col-major L^T = arrK^T arrQ
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &one, k4, d, (size_t)n * d,
                                 q4, d, (size_t)n * d, &zero, logits, n, (size_t)n * n, batch));
    gridSoftmaxK<<<(unsigned)(rows * heads * n), 256>>>(logits, bias, mask, n, heads, r0, tr, scale);
    // O = P V -> col-major O^T = arrV arrP
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, d, n, n, &one, v4, d, (size_t)n * d,
                                 logits, n, (size_t)n * n, &zero, q4, d, (size_t)n * d, batch));
    fromHeadsGateK<<<grid(prs * Wd), 256>>>(q4, g, sc.t1, rows, n, heads, d);
    }
    linear(sc.t1, sc.t2, prs, Wd, C, w(pre + ".outputProjection"));
    if (float* ob = wOpt(pre + ".outputProjectionBias")) addBiasK<<<grid(prs * C), 256>>>(sc.t2, ob, prs, C);
    addGridK<<<grid(prs * C), 256>>>(pair, sc.t2, n, C, r0, rows, tr);
  }
}

static void transition(float* x, size_t rows, int C, const std::string& pre, Scratch& sc) {
  int I = C * 4;
  size_t rowsPer = std::max<size_t>(1, sc.elems / (2 * I));
  for (size_t r0 = 0; r0 < rows; r0 += rowsPer) {
    size_t r = std::min(rowsPer, rows - r0);
    layerNorm(x + r0 * C, sc.t0, r, C, w(pre + ".inputLayerNormScale"), w(pre + ".inputLayerNormOffset"));
    linear(sc.t0, sc.t1, r, C, 2 * I, w(pre + ".transition1"));
    swigluK<<<grid(r * I), 256>>>(sc.t1, sc.t2, r, I);
    linear(sc.t2, sc.t3, r, I, C, w(pre + ".transition2"));
    addK<<<grid(r * C), 256>>>(x + r0 * C, sc.t3, r * C);
  }
}



// ---- the same attention, pipelined: cp.async double-buffers K, V and a padded bias tile into
// shared memory while the previous tile computes; K and V fragments come in through ldmatrix
// (V with .trans, so it is stored as it arrives). 8 warps, 128 queries, key tiles of 64.
__device__ __forceinline__ void cpAsync16(void* dst, const void* src, bool valid) {
  uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
constexpr int FA_D = 32, FA_BK = 64, FA_LDK = FA_D + 8, FA_LDB = FA_BK + 8;
template <int WARPS> __host__ __device__ constexpr size_t faStage() {
  return (size_t)2 * FA_BK * FA_LDK * 2 + (size_t)(16 * WARPS) * FA_LDB * 2 + FA_BK * 4;
}
// Scores in the log2 domain: Q carries scale * log2(e) and the bias log2(e), so a probability
// is one exp2 and the scale costs nothing per key.
constexpr float LOG2E = 1.4426950408889634f;
template <typename TO, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) flashGridAsync(const __nv_bfloat16* qkvg, const __nv_bfloat16* bias,
    int biasStride, const float* mask, TO* out, int n, int heads, size_t r0, bool tr, float scale) {
  constexpr int D = FA_D, BQ = 16 * WARPS, BK = FA_BK, LDK = FA_LDK, LDB = FA_LDB, NT = WARPS * 32;
  constexpr size_t FA_STAGE = faStage<WARPS>();
  extern __shared__ __align__(16) unsigned char smem[];
  auto Kst = [&](int s) { return (__nv_bfloat16*)(smem + s * FA_STAGE); };
  auto Vst = [&](int s) { return Kst(s) + BK * LDK; };
  auto Bst = [&](int s) { return Vst(s) + BK * LDK; };
  auto Mst = [&](int s) { return (float*)(Bst(s) + BQ * LDB); };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t rl = b / heads, r = r0 + rl;
  const int W = heads * D, W4 = 4 * W;
  const __nv_bfloat16* base = qkvg + rl * (size_t)n * W4 + h * D;
  int q0 = blockIdx.x * BQ;
  auto issue = [&](int j0, int st) {
    __nv_bfloat16 *K = Kst(st), *V = Vst(st), *B = Bst(st);
    for (int t = threadIdx.x; t < BK * 4 * 2; t += NT) {          // K, V: 64 rows x 4 chunks
      int which = t / (BK * 4), u = t % (BK * 4), jj = u >> 2, c = (u & 3) * 8, j = j0 + jj;
      const __nv_bfloat16* src = base + (size_t)(j < n ? j : 0) * W4 + (which + 1) * W + c;
      cpAsync16((which ? V : K) + jj * LDK + c, src, j < n);
    }
    for (int t = threadIdx.x; t < BQ * (BK / 8); t += NT) {        // bias: 128 rows x 8 chunks
      int qi = t / (BK / 8), c = (t % (BK / 8)) * 8, i = q0 + qi, j = j0 + c;
      bool ok = i < n && j < n;     // a chunk straddling n reads the zero padding past it
      const __nv_bfloat16* src = bias + ((size_t)h * n + (i < n ? i : 0)) * biasStride + (ok ? j : 0);
      cpAsync16(B + qi * LDB + c, src, ok);
    }
    if (threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Mst(st)[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f)
                                   : -INFINITY;
    }
    asm volatile("cp.async.commit_group;");
  };
  int i0 = q0 + warp * 16 + g, i1 = i0 + 8;
  auto q2 = [&](int i, int e) -> uint32_t {
    if (i >= n) return 0u;
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(base + (size_t)i * W4 + e);
    v = __floats2bfloat162_rn(__low2float(v) * scale * LOG2E, __high2float(v) * scale * LOG2E);
    return *reinterpret_cast<uint32_t*>(&v);
  };
  uint32_t qa[2][4];
  for (int ks = 0; ks < 2; ++ks) {
    int e = ks * 16 + tig * 2;
    qa[ks][0] = q2(i0, e); qa[ks][1] = q2(i1, e); qa[ks][2] = q2(i0, e + 8); qa[ks][3] = q2(i1, e + 8);
  }
  float o[D / 8][4] = {};
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  int tiles = (n + BK - 1) / BK;
  issue(0, 0);
  for (int tile = 0; tile < tiles; ++tile) {
    int st = tile & 1, j0 = tile * BK;
    if (tile + 1 < tiles) { issue(j0 + BK, st ^ 1); asm volatile("cp.async.wait_group 1;"); }
    else asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    const __nv_bfloat16 *K = Kst(st), *V = Vst(st), *B = Bst(st); const float* M = Mst(st);
    float sv[BK / 8][4];
    for (int nt = 0; nt < BK / 8; ++nt) {
      uint32_t kb[4];
      ldsm4(kb, K + (nt * 8 + (lane & 7)) * LDK + (lane >> 3) * 8);
      sv[nt][0] = sv[nt][1] = sv[nt][2] = sv[nt][3] = 0.f;
      mma16816(sv[nt], qa[0], kb[0], kb[1]);
      mma16816(sv[nt], qa[1], kb[2], kb[3]);
    }
    const __nv_bfloat16* br0 = B + (warp * 16 + g) * LDB;
    const __nv_bfloat16* br1 = br0 + 8 * LDB;
    float t0 = -INFINITY, t1 = -INFINITY;
    for (int nt = 0; nt < BK / 8; ++nt) {
      int jj = nt * 8 + tig * 2;
      __nv_bfloat162 u = *reinterpret_cast<const __nv_bfloat162*>(br0 + jj);
      __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(br1 + jj);
      sv[nt][0] += __low2float(u) + M[jj];
      sv[nt][1] += __high2float(u) + M[jj + 1];
      sv[nt][2] += __low2float(v) + M[jj];
      sv[nt][3] += __high2float(v) + M[jj + 1];
      t0 = fmaxf(t0, fmaxf(sv[nt][0], sv[nt][1])); t1 = fmaxf(t1, fmaxf(sv[nt][2], sv[nt][3]));
    }
    t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
    t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
    float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);
    float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
    m0 = n0; m1 = n1; l0 *= c0; l1 *= c1;
    for (int et = 0; et < D / 8; ++et) { o[et][0] *= c0; o[et][1] *= c0; o[et][2] *= c1; o[et][3] *= c1; }
    for (int nt = 0; nt < BK / 8; ++nt) {
      sv[nt][0] = exp2f(sv[nt][0] - n0); sv[nt][1] = exp2f(sv[nt][1] - n0);
      sv[nt][2] = exp2f(sv[nt][2] - n1); sv[nt][3] = exp2f(sv[nt][3] - n1);
      l0 += sv[nt][0] + sv[nt][1]; l1 += sv[nt][2] + sv[nt][3];
    }
    for (int t = 0; t < BK / 16; ++t) {
      uint32_t pa[4] = { pack2(sv[2 * t][0], sv[2 * t][1]), pack2(sv[2 * t][2], sv[2 * t][3]),
                         pack2(sv[2 * t + 1][0], sv[2 * t + 1][1]), pack2(sv[2 * t + 1][2], sv[2 * t + 1][3]) };
      for (int et = 0; et < D / 8; et += 2) {
        uint32_t vb[4];
        ldsm4t(vb, V + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (et + (lane >> 4)) * 8);
        mma16816(o[et], pa, vb[0], vb[1]);
        mma16816(o[et + 1], pa, vb[2], vb[3]);
      }
    }
    __syncthreads();
  }
  l0 += __shfl_xor_sync(~0u, l0, 1); l0 += __shfl_xor_sync(~0u, l0, 2);
  l1 += __shfl_xor_sync(~0u, l1, 1); l1 += __shfl_xor_sync(~0u, l1, 2);
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    if (i0 < n) {
      const __nv_bfloat16* gp = base + (size_t)i0 * W4 + 3 * W + e;
      TO* op = out + (rl * n + i0) * W + h * D + e;
      op[0] = (TO)(o[et][0] / l0 * sigm(__bfloat162float(gp[0])));
      op[1] = (TO)(o[et][1] / l0 * sigm(__bfloat162float(gp[1])));
    }
    if (i1 < n) {
      const __nv_bfloat16* gp = base + (size_t)i1 * W4 + 3 * W + e;
      TO* op = out + (rl * n + i1) * W + h * D + e;
      op[0] = (TO)(o[et][2] / l1 * sigm(__bfloat162float(gp[0])));
      op[1] = (TO)(o[et][3] / l1 * sigm(__bfloat162float(gp[1])));
    }
  }
}
// The bias with each row padded to a multiple of 8 and the padding zeroed, so a 16-byte
// copy is always aligned and a chunk past the last key reads zeros.
__global__ void biasLayoutPadded(const float* raw, __nv_bfloat16* bias, int n, int stride, int heads,
                                 bool swap) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)heads * n * stride;
  if (t >= total) return;
  int j = (int)(t % stride); size_t rest = t / stride; int i = (int)(rest % n); int h = (int)(rest / n);
  bias[t] = j < n ? __float2bfloat16(LOG2E * raw[(swap ? ((size_t)j * n + i) : ((size_t)i * n + j)) * heads + h])
                  : __float2bfloat16(0.f);
}


__global__ void addBiasHalfK(__nv_bfloat16* y, const float* b, size_t rows, int ld, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * C) { size_t r = i / C; int c = (int)(i % C);
    y[r * ld + c] = __float2bfloat16(__bfloat162float(y[r * ld + c]) + b[c]); }
}
// softmax rows of logits [h][i][j] with the pair logits and the sequence mask; writes P as 16-bit
__global__ void singleSoftmaxHalfK(const float* logits, const float* pairLogits, const float* seqMask,
                                   __nv_bfloat16* P, int n, float scale) {
  size_t rowId = blockIdx.x;
  const float* L = logits + rowId * n; const float* B = pairLogits + rowId * n;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    mx = fmaxf(mx, L[j] * scale + B[j] + 1e9f * (seqMask[j] - 1.f));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  float sum = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    sum += __expf(L[j] * scale + B[j] + 1e9f * (seqMask[j] - 1.f) - mx);
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    P[rowId * n + j] = __float2bfloat16(__expf(L[j] * scale + B[j] + 1e9f * (seqMask[j] - 1.f) - mx) * inv);
}
// gathered [i][h*d+e] (16-bit) times sigmoid(gate), gate read out of qkvg
__global__ void gateHalfK(__nv_bfloat16* o, const __nv_bfloat16* qkvg, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  size_t i = t / Wd; int c = (int)(t % Wd);
  o[t] = __float2bfloat16(__bfloat162float(o[t]) * sigm(__bfloat162float(qkvg[i * 4 * Wd + 3 * Wd + c])));
}
__global__ void logitsLayoutHalfK(const float* flat, float* out, size_t pairs, int heads) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * heads) return;
  out[t] = flat[(t % pairs) * heads + t / pairs];
}

using bf16 = __nv_bfloat16;
static void triangleBF(float* pair, const float* mask, const Shape& s, const std::string& pre,
                       bool outgoing, bf16* a, bf16* b, float* prod, Scratch& sc) {
  int n = s.n, C = s.C; size_t pairs = (size_t)n * n;
  bf16* norm = (bf16*)slot(21, (pairs * C + 1) / 2);            // kept for pass 2's gate
  size_t rowsPer = std::max<size_t>(1, sc.elems / (2 * C));     // [rows][4C] bf16 in a chunk tensor
  bf16* pg = (bf16*)sc.t1;
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    layerNormT<bf16>(pair + r0 * C, norm + r0 * C, rows, C, w(pre + ".leftNormInputScale"),
                     w(pre + ".leftNormInputOffset"));
    gemmBF(norm + r0 * C, pg, LP_CUDA, rows, C, 4 * C, w(pre + ".projectionGate"));
    triGateBF<<<dim3((unsigned)((rows + 31) / 32), (C + 31) / 32), dim3(32, 8)>>>(
      pg, mask, a, b, r0, rows, C, pairs);
  }
  float alpha = s.divide ? 1.f / n : 1.f, zero = 0.f;
  if (outgoing)
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, b, LP_CUDA, n,
      pairs, a, LP_CUDA, n, pairs, &zero, prod, CUDA_R_32F, n, pairs, C, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, a, LP_CUDA, n,
      pairs, b, LP_CUDA, n, pairs, &zero, prod, CUDA_R_32F, n, pairs, C, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  rowsPer = std::max<size_t>(1, sc.elems / C);
  bf16* centred = (bf16*)sc.t0;
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    centerNormKT<bf16><<<(unsigned)((rows + 31) / 32), dim3(32, 8), C * 33 * 4>>>(prod, centred, r0,
      rows, C, pairs, w(pre + ".centerNormScale"), w(pre + ".centerNormOffset"));
    gemmBF(centred, sc.t1, CUDA_R_32F, rows, C, C, w(pre + ".outputProjection"));
    gemmBF(norm + r0 * C, sc.t2, CUDA_R_32F, rows, C, C, w(pre + ".gatingLinear"));
    gatedAddK<<<grid(rows * C), 256>>>(pair + r0 * C, sc.t1, sc.t2, rows * C);
  }
}
static void transitionBF(float* x, size_t rows, int C, const std::string& pre, Scratch& sc) {
  int I = C * 4;
  size_t rowsPer = std::max<size_t>(1, sc.elems * 2 / (2 * I));  // [rows][2I] bf16 in a chunk tensor
  for (size_t r0 = 0; r0 < rows; r0 += rowsPer) {
    size_t r = std::min(rowsPer, rows - r0);
    bf16* xb = (bf16*)sc.t0; bf16* wide = (bf16*)sc.t1; bf16* gated = (bf16*)sc.t2;
    layerNormT<bf16>(x + r0 * C, xb, r, C, w(pre + ".inputLayerNormScale"), w(pre + ".inputLayerNormOffset"));
    gemmBF(xb, wide, LP_CUDA, r, C, 2 * I, w(pre + ".transition1"));
    swigluBF<<<grid(r * I), 256>>>(wide, gated, r, I);
    gemmBF(gated, x + r0 * C, CUDA_R_32F, r, I, C, w(pre + ".transition2"), false, 1.f);
  }
}
static void gridAttentionBF(float* pair, const float* mask, const Shape& s, const std::string& pre,
                            bool tr, Scratch& sc) {
  int n = s.n, C = s.C, heads = s.gh, d = s.gd, Wd = heads * d; size_t pairs = (size_t)n * n;
  if (wOpt(pre + ".gatingQueryBias")) { fprintf(stderr, "bf16 grid: gate bias not wired\n"); exit(1); }
  bf16* norm = (bf16*)slot(21, (pairs * C + 1) / 2);
  layerNormT<bf16>(pair, norm, pairs, C, w(pre + ".actNormScale"), w(pre + ".actNormOffset"));
  gemmBF(norm, sc.t0, CUDA_R_32F, pairs, C, heads, w(pre + ".pairBiasProjection"));
  int stride = (n + 7) / 8 * 8;
  bf16* bias = (bf16*)slot(17, ((size_t)heads * n * stride + 1) / 2);
  if (ASYNC) biasLayoutPadded<<<grid((size_t)heads * n * stride), 256>>>(sc.t0, bias, n, stride, heads, tr && s.swap);
  else biasLayoutBF<<<grid(pairs * heads), 256>>>(sc.t0, bias, n, heads, tr && s.swap);
  size_t R = std::max<size_t>(1, sc.elems / ((size_t)n * std::max(C, Wd)));
  float scale = 1.f / sqrtf((float)d);
  for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
    size_t rows = std::min(R, (size_t)n - r0), prs = rows * n;
    const bf16* act = norm + r0 * n * C;
    if (tr) { gatherActBF<<<grid(prs * C), 256>>>(norm, (bf16*)sc.t0, n, C, r0, rows); act = (bf16*)sc.t0; }
    bf16* qkvg = (bf16*)slot(16, ((prs + 128) * 4 * Wd + 1) / 2);
    gemmBF(act, qkvg, LP_CUDA, prs, C, 4 * Wd, w(pre + ".qkvg"));
    bf16* gathered = (bf16*)sc.t1;
    stageMark("grid.pre");
    if (ASYNC) {
      // 128 queries a block unless that pads more than 32 rows
      bool wide = ((n + 127) / 128) * 128 - n <= 32;
      static bool attr = false;
      if (!attr) {
        CK(cudaFuncSetAttribute(flashGridAsync<bf16, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2 * faStage<8>())));
        CK(cudaFuncSetAttribute(flashGridAsync<bf16, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2 * faStage<4>())));
        attr = true;
      }
      if (wide) flashGridAsync<bf16, 8><<<dim3((n + 127) / 128, (unsigned)(rows * heads)), 256, 2 * faStage<8>()>>>(
        qkvg, bias, stride, mask, gathered, n, heads, r0, tr, scale);
      else flashGridAsync<bf16, 4><<<dim3((n + 63) / 64, (unsigned)(rows * heads)), 128, 2 * faStage<4>()>>>(
        qkvg, bias, stride, mask, gathered, n, heads, r0, tr, scale);
    } else if (FLASH_WARPS == 8)
      flashGridMMA<bf16, 8><<<dim3((n + 127) / 128, (unsigned)(rows * heads)), 256>>>(
        qkvg, bias, mask, gathered, n, heads, r0, tr, scale);
    else
      flashGridMMA<bf16, 4><<<dim3((n + 63) / 64, (unsigned)(rows * heads)), 128>>>(
        qkvg, bias, mask, gathered, n, heads, r0, tr, scale);
    stageMark("grid.flash");
    float* ob = wOpt(pre + ".outputProjectionBias");
    if (!tr && ob == nullptr) {
      gemmBF(gathered, pair + r0 * n * C, CUDA_R_32F, prs, Wd, C, w(pre + ".outputProjection"), false, 1.f);
      continue;
    }
    gemmBF(gathered, sc.t2, CUDA_R_32F, prs, Wd, C, w(pre + ".outputProjection"));
    if (ob) addBiasK<<<grid(prs * C), 256>>>(sc.t2, ob, prs, C);
    addGridK<<<grid(prs * C), 256>>>(pair, sc.t2, n, C, r0, rows, tr);
  }
}

static void singleTrackBF(float* single, const float* pair, const float* seqMask, const Shape& s,
                          Scratch& sc) {
  int n = s.n, C = s.C, S = s.S, heads = s.sh, d = s.sd, Wd = heads * d; size_t pairs = (size_t)n * n;
  // pair logits [h][i][j]
  float* flat = slot(4, pairs * heads); float* pl = slot(5, pairs * heads);
  size_t rowsPer = std::max<size_t>(1, sc.elems * 2 / C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t r = std::min(rowsPer, pairs - r0);
    layerNormT<bf16>(pair + r0 * C, (bf16*)sc.t0, r, C, w("singlePairLogitsNormScale"),
                     w("singlePairLogitsNormOffset"));
    gemmBF((bf16*)sc.t0, flat + r0 * heads, CUDA_R_32F, r, C, heads, w("singlePairLogitsProjection"));
  }
  logitsLayoutHalfK<<<grid(pairs * heads), 256>>>(flat, pl, pairs, heads);
  // q, k, v, gate in one GEMM; the heads are read with strides, never reshuffled
  bf16* nrm = (bf16*)slot(6, ((size_t)n * S + 1) / 2);
  bf16* qkvg = (bf16*)slot(7, ((size_t)n * 4 * Wd + 1) / 2);
  layerNormT<bf16>(single, nrm, n, S, w("singleAttention.layerNormScale"), w("singleAttention.layerNormOffset"));
  gemmBF(nrm, qkvg, LP_CUDA, n, S, 4 * Wd, w("singleAttention.qkvg"));
  addBiasHalfK<<<grid((size_t)n * Wd), 256>>>(qkvg, w("singleAttention.qBias"), n, 4 * Wd, Wd);
  float* logits = slot(8, (size_t)heads * n * n);
  bf16* P = (bf16*)slot(9, ((size_t)heads * n * n + 1) / 2);
  bf16* o = (bf16*)slot(10, ((size_t)n * Wd + 1) / 2);
  const float one = 1.f, zero = 0.f;
  // L[h] (n x n, row-major, i over rows) = Q_h K_h^T: col-major L^T = K_h^T(op T) . Q_h
  CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &one,
     qkvg + Wd, LP_CUDA, 4 * Wd, d, qkvg, LP_CUDA, 4 * Wd, d, &zero, logits, CUDA_R_32F, n,
     (size_t)n * n, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  singleSoftmaxHalfK<<<(unsigned)(heads * n), 128>>>(logits, pl, seqMask, P, n, 1.f / sqrtf((float)d));
  // O_h (n x d, row-major [i][h*d+e]) = P_h V_h: col-major O^T = V_h^T . P_h^T
  CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, d, n, n, &one,
     qkvg + 2 * Wd, LP_CUDA, 4 * Wd, d, P, LP_CUDA, n, (size_t)n * n, &zero, o, LP_CUDA, Wd, d,
     heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  gateHalfK<<<grid((size_t)n * Wd), 256>>>(o, qkvg, n, Wd);
  gemmBF(o, single, CUDA_R_32F, n, Wd, S, w("singleAttention.outputProjection"), false, 1.f);
  transitionBF(single, n, S, "singleTransition", sc);
}

static void singleTrack(float* single, const float* pair, const float* seqMask, const Shape& s,
                        float* logits, Scratch& sc) {
  int n = s.n, C = s.C, S = s.S, heads = s.sh, d = s.sd, Wd = heads * d; size_t pairs = (size_t)n * n;
  // pair logits [h][i][j]
  float* flat = slot(4, pairs * heads);
  float* pl = slot(5, pairs * heads);
  size_t rowsPer = std::max<size_t>(1, sc.elems / C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t r = std::min(rowsPer, pairs - r0);
    layerNorm(pair + r0 * C, sc.t0, r, C, w("singlePairLogitsNormScale"), w("singlePairLogitsNormOffset"));
    linear(sc.t0, flat + r0 * heads, r, C, heads, w("singlePairLogitsProjection"));
  }
  logitsLayoutK<<<grid(pairs * heads), 256>>>(flat, pl, pairs, heads);
  // attention over n tokens
  float* nrm = slot(6, (size_t)n * S); float* q = slot(7, (size_t)n * Wd); float* k = slot(8, (size_t)n * Wd);
  float* v = slot(9, (size_t)n * Wd); float* g = slot(10, (size_t)n * Wd);
  float* q4 = slot(11, (size_t)n * Wd); float* k4 = slot(12, (size_t)n * Wd); float* v4 = slot(13, (size_t)n * Wd);
  float* o = slot(14, (size_t)n * S);
  layerNorm(single, nrm, n, S, w("singleAttention.layerNormScale"), w("singleAttention.layerNormOffset"));
  linear(nrm, q, n, S, Wd, w("singleAttention.qProjection"));
  addBiasK<<<grid((size_t)n * Wd), 256>>>(q, w("singleAttention.qBias"), n, Wd);
  linear(nrm, k, n, S, Wd, w("singleAttention.kProjection"));
  linear(nrm, v, n, S, Wd, w("singleAttention.vProjection"));
  linear(nrm, g, n, S, Wd, w("singleAttention.gatingQuery"));
  toHeadsK<<<grid((size_t)n * Wd), 256>>>(q, q4, 1, n, heads, d);
  toHeadsK<<<grid((size_t)n * Wd), 256>>>(k, k4, 1, n, heads, d);
  toHeadsK<<<grid((size_t)n * Wd), 256>>>(v, v4, 1, n, heads, d);
  float one = 1.f, zero = 0.f;
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &one, k4, d, (size_t)n * d,
                               q4, d, (size_t)n * d, &zero, logits, n, (size_t)n * n, heads));
  singleSoftmaxK<<<(unsigned)(heads * n), 256>>>(logits, pl, seqMask, n, 1.f / sqrtf((float)d));
  CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, d, n, n, &one, v4, d, (size_t)n * d,
                               logits, n, (size_t)n * n, &zero, q4, d, (size_t)n * d, heads));
  fromHeadsGateK<<<grid((size_t)n * Wd), 256>>>(q4, g, q, 1, n, heads, d);
  linear(q, o, n, Wd, S, w("singleAttention.outputProjection"));
  addK<<<grid((size_t)n * S), 256>>>(single, o, (size_t)n * S);
  transition(single, n, S, "singleTransition", sc);
}

int main(int argc, char** argv) {
  std::string dir = argv[1];
  Data D = load(dir);
  int exported = (int)D.meta["tokens"];
  int n = exported; bool tf32 = false; int repeat = 1; bool stages = false;
  for (int i = 2; i < argc; ++i) {
    if (!strncmp(argv[i], "--tokens=", 9)) n = atoi(argv[i] + 9);
    else if (!strcmp(argv[i], "--tf32")) tf32 = TF_LINEAR = TF_TRI = TF_ATT = true;
    else if (!strcmp(argv[i], "--tf32-linear")) TF_LINEAR = true;
    else if (!strcmp(argv[i], "--tf32-tri")) TF_TRI = true;
    else if (!strcmp(argv[i], "--tf32-att")) TF_ATT = true;
    else if (!strcmp(argv[i], "--stages")) stages = STAGES = true;
    else if (!strcmp(argv[i], "--no-flash")) FLASH = false;
    else if (!strcmp(argv[i], "--no-tc")) TC = false;
    else if (!strcmp(argv[i], "--no-fused")) FUSED = false;
    else if (!strcmp(argv[i], "--bf16")) BF16 = true;
    else if (!strcmp(argv[i], "--no-mma")) MMA = false;
    else if (!strcmp(argv[i], "--no-lean")) LEAN = false;
    else if (!strcmp(argv[i], "--graph")) GRAPH = true;
    else if (!strncmp(argv[i], "--warps=", 8)) FLASH_WARPS = atoi(argv[i] + 8);
    else if (!strcmp(argv[i], "--no-async")) ASYNC = false;
    else if (!strncmp(argv[i], "--repeat=", 9)) repeat = atoi(argv[i] + 9);
  }
  Shape s{n, (int)D.meta["pairChannels"], (int)D.meta["singleChannels"], (int)D.meta["gridHeads"],
          (int)D.meta["gridDim"], (int)D.meta["singleHeads"], (int)D.meta["singleDim"],
          D.meta["swapTransposedBias"] != 0, D.meta["divideByLength"] != 0};
  CB(cublasCreate(&H));
  for (auto& [name, ol] : D.index)
    if (name.rfind("input.", 0) != 0 && name.rfind("expected.", 0) != 0 && name.rfind("check.", 0) != 0)
      W[name] = upload(D.all.data() + ol.first, ol.second);

  int blocks = D.meta.count("blocks") ? (int)D.meta["blocks"] : 1;
  auto blockPrefix = [&](int k) { return blocks == 1 && !D.has("b0.singlePairLogitsNormScale")
                                         ? std::string() : "b" + std::to_string(k) + "."; };
  for (int blk = 0; blk < blocks; ++blk) {
  std::string bp = blockPrefix(blk);
  // q, k, v and the gate as one (C, 4W) matrix: q, k and the gate are stored (out, in).
  for (const char* g : {"pairAttention1", "pairAttention2"}) {
    std::string pre = bp + g; int Cc = s.C, Wd = s.gh * s.gd;
    std::vector<float> cat((size_t)Cc * 4 * Wd), catBias(4 * Wd, 0.f);
    const char* roles[4] = {".qProjection", ".kProjection", ".vProjection", ".gatingQuery"};
    bool trans[4] = {true, true, false, true};
    for (int role = 0; role < 4; ++role) {
      const float* src = D.host(pre + roles[role]);
      for (int c = 0; c < Cc; ++c) for (int o = 0; o < Wd; ++o)
        cat[(size_t)c * 4 * Wd + role * Wd + o] = trans[role] ? src[(size_t)o * Cc + c] : src[(size_t)c * Wd + o];
    }
    W[pre + ".qkvg"] = upload(cat.data(), cat.size());
    if (D.has(pre + ".gatingQueryBias")) {
      const float* gb = D.host(pre + ".gatingQueryBias");
      for (int o = 0; o < Wd; ++o) catBias[3 * Wd + o] = gb[o];
      W[pre + ".qkvgBias"] = upload(catBias.data(), catBias.size());
    }
  }
  {
    std::string pre = bp + "singleAttention"; int Sc = s.S, Wd = s.sh * s.sd;
    std::vector<float> cat((size_t)Sc * 4 * Wd);
    const char* roles[4] = {".qProjection", ".kProjection", ".vProjection", ".gatingQuery"};
    for (int role = 0; role < 4; ++role) {
      const float* src = D.host(pre + roles[role]);
      for (int c = 0; c < Sc; ++c) for (int o = 0; o < Wd; ++o)
        cat[(size_t)c * 4 * Wd + role * Wd + o] = src[(size_t)c * Wd + o];
    }
    W[pre + ".qkvg"] = upload(cat.data(), cat.size());
  }
  for (const char* t : {"triOutgoing", "triIncoming"}) {
    std::string pre = bp + t; int Cc = s.C;
    std::vector<float> cat((size_t)Cc * 4 * Cc);
    const float* proj = D.host(pre + ".projection"); const float* gate = D.host(pre + ".gate");
    for (int c = 0; c < Cc; ++c) for (int o = 0; o < 2 * Cc; ++o) {
      cat[(size_t)c * 4 * Cc + o] = proj[(size_t)c * 2 * Cc + o];
      cat[(size_t)c * 4 * Cc + 2 * Cc + o] = gate[(size_t)c * 2 * Cc + o];
    }
    W[pre + ".projectionGate"] = upload(cat.data(), cat.size());
  }
  }
  size_t pairs = (size_t)n * n, C = s.C;
  std::vector<float> hPair(pairs * C), hSingle((size_t)n * s.S), hSeq(n), hMask(pairs);
  bool check = n == exported;
  if (check) {
    memcpy(hPair.data(), D.host("input.pair"), hPair.size() * 4);
    memcpy(hSingle.data(), D.host("input.single"), hSingle.size() * 4);
    memcpy(hSeq.data(), D.host("input.seqMask"), n * 4);
    memcpy(hMask.data(), D.host("input.pairMask"), pairs * 4);
  } else {
    unsigned st = 7; auto rnd = [&]() { st = st * 1103515245u + 12345u; return ((st >> 8) & 0xffff) / 65536.f - 0.5f; };
    for (auto& x : hPair) x = rnd() * 3.4f;
    for (auto& x : hSingle) x = rnd() * 3.4f;
    for (int i = 0; i < n; ++i) hSeq[i] = 1.f;
    for (auto& x : hMask) x = 1.f;
  }
  float* pair0 = upload(hPair.data(), hPair.size());
  float* pair = dalloc(hPair.size());
  float* single0 = upload(hSingle.data(), hSingle.size());
  float* single = dalloc(hSingle.size());
  float* seq = upload(hSeq.data(), n);
  float* mask = upload(hMask.data(), pairs);
  // the big tensors: a, b, product (C x pairs) and the grid's normalised pair, aliased
  float* a = dalloc(pairs * C); float* b = dalloc(pairs * C); float* prod = dalloc(pairs * C);
  float* norm = prod;                       // triangle is done with it by the grid stage
  float* bias = dalloc(pairs * std::max(s.gh, 1));
  // every row of the grid attention at once when that fits in two chunks' worth
  size_t logitsElems = std::max((size_t)s.sh * n * n,
                                std::min<size_t>(CHUNK_ELEMS * 2, (size_t)s.gh * n * n * n));
  float* logits = dalloc(logitsElems);
  Scratch sc{dalloc(CHUNK_ELEMS), dalloc(CHUNK_ELEMS), dalloc(CHUNK_ELEMS), dalloc(CHUNK_ELEMS), CHUNK_ELEMS};
  size_t freeB, totalB; CK(cudaMemGetInfo(&freeB, &totalB));
  printf("tokens %d  pair %.0f MiB  device used %.0f MiB  tf32 %d\n", n, pairs * C * 4 / 1048576.0,
         (totalB - freeB) / 1048576.0, (int)tf32);

  // --graph: run once to allocate every slot and bf16 weight copy, then capture the block
  // into a CUDA graph and replay it - one launch instead of ~60.
  CB(cublasSetStream(H, cudaStreamPerThread));
  void* workspace; CK(cudaMalloc(&workspace, 32 << 20));
  CB(cublasSetWorkspace(H, workspace, 32 << 20));
  cudaGraphExec_t exec = nullptr;
  auto body = [&]() {
    stageMark(nullptr);
    for (int blk = 0; blk < blocks; ++blk) {
      PREFIX = blockPrefix(blk);
      if (BF16 && LEAN) {
        triangleBF(pair, mask, s, "triOutgoing", true, (bf16*)a, (bf16*)b, prod, sc); stageMark("tri.out");
        triangleBF(pair, mask, s, "triIncoming", false, (bf16*)a, (bf16*)b, prod, sc); stageMark("tri.in");
        gridAttentionBF(pair, mask, s, "pairAttention1", false, sc); stageMark("grid.row");
        gridAttentionBF(pair, mask, s, "pairAttention2", true, sc); stageMark("grid.col");
        transitionBF(pair, pairs, s.C, "pairTransition", sc); stageMark("transition");
      } else {
        triangle(pair, mask, s, "triOutgoing", true, a, b, prod, sc);
        triangle(pair, mask, s, "triIncoming", false, a, b, prod, sc);
        gridAttention(pair, mask, s, "pairAttention1", false, norm, bias, logits, logitsElems, sc);
        gridAttention(pair, mask, s, "pairAttention2", true, norm, bias, logits, logitsElems, sc);
        transition(pair, pairs, s.C, "pairTransition", sc);
      }
      if (BF16 && LEAN) singleTrackBF(single, pair, seq, s, sc);
      else singleTrack(single, pair, seq, s, logits, sc);
      stageMark("single");
    }
    PREFIX.clear();
  };
  if (GRAPH) {
    CK(cudaMemcpy(pair, pair0, pairs * C * 4, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(single, single0, hSingle.size() * 4, cudaMemcpyDeviceToDevice));
    body(); CK(cudaDeviceSynchronize());
    cudaGraph_t graph;
    CK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    body();
    CK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));
  }
  double best = 1e30;
  for (int it = 0; it < repeat && GRAPH; ++it) {
    CK(cudaMemcpy(pair, pair0, pairs * C * 4, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(single, single0, hSingle.size() * 4, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    CK(cudaGraphLaunch(exec, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, ms);
  }
  for (int it = 0; it < repeat && !GRAPH; ++it) {
    CK(cudaMemcpy(pair, pair0, pairs * C * 4, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(single, single0, hSingle.size() * 4, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    body();
    CK(cudaDeviceSynchronize());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    best = std::min(best, ms);
    printf("pass %.1f ms\n", ms);
    if (it == 0) STAGE_MS.clear();     // the first pass converts weights; not part of a profile
  }
  if (STAGES) {
    double total = 0; for (auto& [k, v] : STAGE_MS) total += v;
    for (auto& [k, v] : STAGE_MS) printf("  %-12s %8.1f ms  %4.1f%%\n", k.c_str(), v / std::max(1, repeat - 1), 100 * v / total);
  }
  CK(cudaMemGetInfo(&freeB, &totalB));
  printf("best %.1f ms  device used %.0f MiB\n", best, (totalB - freeB) / 1048576.0);
  if (check) {
    auto rel = [](const float* x, const float* y, size_t m) {
      double num = 0, den = 0; for (size_t i = 0; i < m; ++i) { double e = x[i] - y[i]; num += e * e; den += (double)y[i] * y[i]; }
      return std::sqrt(num / den); };
    std::vector<float> outP(pairs * C), outS(hSingle.size());
    CK(cudaMemcpy(outP.data(), pair, outP.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outS.data(), single, outS.size() * 4, cudaMemcpyDeviceToHost));
    printf("relRMS pair %.3e  single %.3e\n", rel(outP.data(), D.host("expected.pair"), outP.size()),
           rel(outS.data(), D.host("expected.single"), outS.size()));
  }
  return 0;
}
