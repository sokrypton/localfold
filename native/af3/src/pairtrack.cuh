// The pair track - triangle multiplication, grid ("triangle") attention, transition -
// shared by the pairformer, the MSA stack and the template stack. Transcribed from
// src/af3/trunk/pairformer-reference.js; generic in channels, heads and head width.
#pragma once
#include "common.cuh"

// ---------------------------------------------------------------- elementwise kernels
// LayerNorm over the last axis with AF3's fast variance E[x^2] - E[x]^2. One warp a row.
template <class TI, class TO>
__global__ void layerNormK(const TI* in, TO* out, size_t rows, int C, const float* scale,
                           const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const TI* x = in + row * C;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) { float v = toF(x[c]); s += v; ss += v * v; }
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
  float mean = s / C, inv = rsqrtf(ss / C - mean * mean + 1e-5f);
  for (int c = lane; c < C; c += 32)
    out[row * C + c] = fromF<TO>((toF(x[c]) - mean) * inv * scale[c] + offset[c]);
}
template <class TI, class TO>
void layerNorm(const TI* in, TO* out, size_t rows, int C, const std::string& w) {
  layerNormK<TI, TO><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(
    in, out, rows, C, W(w + "Scale"), W(w + "Offset"));
}
template <class TI, class TO>
void layerNorm2(const TI* in, TO* out, size_t rows, int C, const std::string& scale,
                const std::string& offset) {
  layerNormK<TI, TO><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(in, out, rows, C, W(scale), W(offset));
}
template <class T>
__global__ void gatedAddK(float* pair, const T* proj, const T* gate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pair[i] += toF(proj[i]) * sigm(toF(gate[i]));
}
// gated = swish(a) * b (* u with `up`: boltz2's conditioned transition up-gate), a row of wide
// being [a | b] or [a | b | u], each I wide
template <class T>
__global__ void swigluK(const T* wide, T* gated, size_t rows, int I, bool up) {
  int L = up ? 3 * I : 2 * I;
  if constexpr (std::is_same_v<T, half>) {
    if (I % 8 == 0) {                 // eight halves (16 bytes) a thread
      size_t t = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
      if (t >= rows * I) return;
      size_t r = t / I; int i = (int)(t % I);
      uint4 a = *(const uint4*)(wide + r * L + i), b = *(const uint4*)(wide + r * L + I + i), o, u;
      if (up) u = *(const uint4*)(wide + r * L + 2 * I + i);
      const half2* a2 = (const half2*)&a; const half2* b2 = (const half2*)&b; half2* o2 = (half2*)&o;
      const half2* u2 = (const half2*)&u;
      for (int k = 0; k < 4; ++k) {
        float2 g = __half22float2(a2[k]), v = __half22float2(b2[k]);
        float x = g.x * sigm(g.x) * v.x, y = g.y * sigm(g.y) * v.y;
        if (up) { float2 w = __half22float2(u2[k]); x *= w.x; y *= w.y; }
        o2[k] = __floats2half2_rn(x, y);
      }
      *(uint4*)(gated + t) = o;
      return;
    }
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int i = (int)(t % I);
  float g = toF(wide[r * L + i]);
  float v = g * sigm(g) * toF(wide[r * L + I + i]);
  if (up) v *= toF(wide[r * L + 2 * I + i]);
  gated[t] = fromF<T>(v);
}
// the launch: an eighth of the threads for the vector path
template <class T>
void swiglu(const T* wide, T* gated, size_t rows, int I, bool up = false) {
  size_t work = std::is_same_v<T, half> && I % 8 == 0 ? rows * I / 8 : rows * I;
  swigluK<T><<<blocks(work), 256, 0, STREAM>>>(wide, gated, rows, I, up);
}
// a conditioned transition's first weight: transition1, or [transition1 | ffwAToB] where the
// bundle carries boltz2's up-gate (one GEMM for both)
inline std::string upGatedTransition1(const std::string& B, int C, int I, bool& up) {
  up = hasW(B + ".ffwAToB");
  if (!up) return B + ".ffwTransition1";
  return concatColumns(B + ".ffwTransition1|aToB~", C, {{B + ".ffwTransition1", 2 * I, false}, {B + ".ffwAToB", I, false}});
}

// ---------------------------------------------------------------- fused weights
// The triangle's projection and gate as one (C, 4C) matrix: [projection (2C) | gate (2C)].
inline std::string projectionGate(const std::string& pre, int C) {
  return concatColumns(pre + ".projectionGate~", C, {{pre + ".projection", 2 * C, false}, {pre + ".gate", 2 * C, false}});
}
// q, k, v and the gate as one (C, 4W) matrix. The grid attention stores q, k and the gate
// (out, in) and v (in, out); `transposedQkg` says whether that holds.
inline std::string qkvgWeight(const std::string& pre, int C, int Wd, bool transposedQkg) {
  return concatColumns(pre + ".qkvg~", C, {{pre + ".qProjection", Wd, transposedQkg}, {pre + ".kProjection", Wd, transposedQkg},
                                          {pre + ".vProjection", Wd, false}, {pre + ".gatingQuery", Wd, transposedQkg}});
}

// ---------------------------------------------------------------- triangle multiplication
// a, b from [rows][4C] = [projection | gate], the INTERLEAVED split (channel ch's halves
// are 2ch and 2ch+1), masked, written channel-major (c, pairs) through shared memory.
template <class T>
__global__ void triGateK(const T* pg, const float* mask, T* a, T* b, size_t r0, size_t rows, int C,
                         size_t pairs, int n, int np) {
  __shared__ float A[32][33], B[32][33];
  size_t row0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = row0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (local < rows && c < C) {
      float m = mask[r0 + local];
      const T* p = pg + local * 4 * C; const T* g = p + 2 * C;
      va = toF(p[c * 2]) * m * sigm(toF(g[c * 2]));
      vb = toF(p[c * 2 + 1]) * m * sigm(toF(g[c * 2 + 1]));
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t local = row0 + tx; int c = c0 + cy;
    if (local < rows && c < C) {
      unsigned p = (unsigned)(r0 + local), i = p / (unsigned)n;
      size_t q = (size_t)i * np + (p - i * (unsigned)n);     // the padded (np x np) position
      a[(size_t)c * pairs + q] = fromF<T>(A[tx][cy]);
      b[(size_t)c * pairs + q] = fromF<T>(B[tx][cy]);
    }
  }
}
// center_norm over the channels of a (c, pairs) array -> row-major, 32 pairs a block.
template <class TO>
__global__ void centerNormK(const float* prod, TO* out, size_t r0, size_t rows, int C, size_t pairs,
                            const float* scale, const float* offset, int n, int np) {
  extern __shared__ float T_[];
  __shared__ float mean[32], inv[32];
  size_t i0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  for (int c = ty; c < C; c += 8) {
    size_t local = i0 + tx;
    unsigned p = (unsigned)(r0 + local), i = p / (unsigned)n;
    T_[c * 33 + tx] = local < rows ? prod[(size_t)c * pairs + (size_t)i * np + (p - i * (unsigned)n)] : 0.f;
  }
  __syncthreads();
  if (ty == 0) {
    float s = 0, ss = 0;
    for (int c = 0; c < C; ++c) { float v = T_[c * 33 + tx]; s += v; ss += v * v; }
    float m = s / C; mean[tx] = m; inv[tx] = rsqrtf(ss / C - m * m + 1e-5f);
  }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t local = i0 + ry; if (local >= rows) continue;
    for (int c = tx; c < C; c += 32)
      out[local * C + c] = fromF<TO>((T_[c * 33 + ry] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}

// the same in two passes over prod (the statistics, then the normalised rows) when its C x 33 tile does not
// fit (IntelliFold-2's 512 channels on a T4: 67.5 KB)
template <class TO>
__global__ void centerNormStreamK(const float* prod, TO* out, size_t r0, size_t rows, int C, size_t pairs,
                                  const float* scale, const float* offset, int n, int np) {
  __shared__ float ps[8][33], pss[8][33], mean[32], inv[32];
  size_t i0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  size_t local = i0 + tx;
  unsigned p = (unsigned)(r0 + local), i = p / (unsigned)n;
  size_t at = (size_t)i * np + (p - i * (unsigned)n);
  float s = 0, ss = 0;
  if (local < rows) for (int c = ty; c < C; c += 8) { float v = prod[(size_t)c * pairs + at]; s += v; ss += v * v; }
  ps[ty][tx] = s; pss[ty][tx] = ss;
  __syncthreads();
  if (ty == 0) {
    float a = 0, b = 0;
    for (int k = 0; k < 8; ++k) { a += ps[k][tx]; b += pss[k][tx]; }
    float m = a / C; mean[tx] = m; inv[tx] = rsqrtf(b / C - m * m + 1e-5f);
  }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t lr = i0 + ry; if (lr >= rows) continue;
    unsigned q = (unsigned)(r0 + lr), qi = q / (unsigned)n;
    size_t qa = (size_t)qi * np + (q - qi * (unsigned)n);
    for (int c = tx; c < C; c += 32)
      out[lr * C + c] = fromF<TO>((prod[(size_t)c * pairs + qa] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}

inline size_t CHUNK = (size_t)64 << 20;   // elements in a chunk tensor
inline bool FUSED_GRID = true;
inline bool TRI_BF16 = true;
inline int TRI_PAD = 8;           // the triangle's padded size is a multiple of this (0: none)
#include "fusedtriangle.cuh"
#include "fused256.cuh"
// The 256-channel pair track's fused kernels (fused256.cuh, ESMFold2's): native/af3's at 128 channels hold
// a whole output tile or weight on the chip, which at 256 is past the registers and shared memory a block
// gets; these stream their weights in narrower steps, two blocks an SM. Below ~80 tokens their tiles leave
// the device idle (ESMFold2's measurement), and a T4's 64 KB fits none of them.
inline bool FUSED_WIDE = true;
inline int FUSED_WIDE_MIN_TOKENS = 80;
// ...at 256 channels only: at OpenDDE's 384 and IntelliFold-2's 512 the same kernels fit one block an SM and
// lose to the unfused path (262 tokens: trunk 2160 against 1925 ms, 3360 against 2821)
constexpr size_t wideTriInSmem(int C) { return (size_t)128 * (C + 8) * 2; }                       // 8 warps
constexpr size_t wideTriOutSmem(int C) { return (size_t)C * 65 * 4 + 2 * 64 * 4; }               // 4 warps
constexpr size_t wideUpSmem(int C) { return std::max((size_t)128 * (C + 8) * 2, (size_t)2 * 2 * C * (32 + 8) * 2); }
inline bool wideFits(int C) {
  return C == 256 && fitsSmem(std::max({wideTriInSmem(C), wideTriOutSmem(C), wideUpSmem(C)}));
}
template <class F> void wideWidth(int, F f) { f(std::integral_constant<int, 256>{}); }

// The bf16 contraction, one np x np GEMM per channel. cuBLAS's default picks a 64x256 tile from
// np 152 to 392 (cuBLAS 12, A100), where a 128x128 tile is up to a quarter faster - 66.5 against
// 84.7 us at np 264 (261 tokens), 83 against 111 at 336, 31 against 39 at 200 - and level at 360
// and up, where the default takes it itself; below ~192 the default wins. So there, cuBLASLt with
// the heuristic list's first 128x128 candidate: a fixed rule, never a timing, so the choice (and
// the output) is the same every run.
inline bool TRI_LT_TILE = true;
inline void triContractBf16(bool outgoing, int np, size_t cs, int C, float alpha, const __nv_bfloat16* a,
                            const __nv_bfloat16* b, __nv_bfloat16* p) {
  const float zero = 0.f;
  if (TRI_LT_TILE && np >= 200 && np <= 352) {
    struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t l; cublasLtMatmulAlgo_t algo; bool ok; };
    static cublasLtHandle_t lt = nullptr;
    static std::map<std::pair<int, bool>, Plan> plans;
    if (!lt) CB(cublasLtCreate(&lt));
    auto it = plans.find({np, outgoing});
    if (it == plans.end()) {
      Plan pl{}; 
      CB(cublasLtMatmulDescCreate(&pl.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t ta = outgoing ? CUBLAS_OP_T : CUBLAS_OP_N, tb = outgoing ? CUBLAS_OP_N : CUBLAS_OP_T;
      CB(cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
      CB(cublasLtMatmulDescSetAttribute(pl.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
      CB(cublasLtMatrixLayoutCreate(&pl.l, CUDA_R_16BF, np, np, np));
      int batch = C; long long stride = (long long)cs;
      CB(cublasLtMatrixLayoutSetAttribute(pl.l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
      CB(cublasLtMatrixLayoutSetAttribute(pl.l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
      cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
      size_t ws = 0;     // no workspace: the plan is captured in the trunk's CUDA graph as it is
      CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      cublasLtMatmulHeuristicResult_t res[32]; int got = 0;
      CB(cublasLtMatmulAlgoGetHeuristic(lt, pl.op, pl.l, pl.l, pl.l, pl.l, pref, 32, res, &got));
      CB(cublasLtMatmulPreferenceDestroy(pref));
      for (int i = 0; i < got && !pl.ok; ++i) {
        int tile = 0, splitk = 1; size_t sz;
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &sz);
        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &sz);
        if (res[i].state == CUBLAS_STATUS_SUCCESS && tile == CUBLASLT_MATMUL_TILE_128x128 && splitk <= 1) {
          pl.algo = res[i].algo; pl.ok = true;
        }
      }
      it = plans.emplace(std::make_pair(np, outgoing), pl).first;
    }
    if (it->second.ok) {          // (a cuBLAS without that candidate keeps its own pick, below)
      const Plan& pl = it->second;
      CB(cublasLtMatmul(lt, pl.op, &alpha, outgoing ? b : a, pl.l, outgoing ? a : b, pl.l, &zero, p, pl.l, p, pl.l,
                        &pl.algo, nullptr, 0, STREAM));
      return;
    }
  }
  if (outgoing)
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, np, np, np, &alpha, b, CUDA_R_16BF, np, cs, a,
      CUDA_R_16BF, np, cs, &zero, p, CUDA_R_16BF, np, cs, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, np, np, np, &alpha, a, CUDA_R_16BF, np, cs, b,
      CUDA_R_16BF, np, cs, &zero, p, CUDA_R_16BF, np, cs, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


// ---------------------------------------------------------------- the triangle multiplication in blocks
// A RECTANGLE of the padded pair space - rows [i0, i0 + I), columns [j0, j0 + J) - with q = (i - i0) J
// + (j - j0) its own index: the operands of one block of the contraction and its output live there.
// A pair held in host memory (tier 2) is read through a WINDOW on the device: rows [wi0, ...) by columns
// [wj0, wj0 + ws), row-major - ws 0 for the pair itself.
struct TriRect {
  int i0, I, j0, J; int wi0 = 0, wj0 = 0, ws = 0;
  size_t size() const { return (size_t)I * J; }
};
// the pair at q (SIZE_MAX for padding), and its place in the window
__device__ __forceinline__ size_t rectPair(const TriRect& r, size_t q, int n, size_t* win = nullptr) {
  size_t i = r.i0 + q / r.J, j = r.j0 + q % r.J;
  if (i >= (size_t)n || j >= (size_t)n) return SIZE_MAX;
  if (win) *win = r.ws ? (i - r.wi0) * r.ws + (j - r.wj0) : i * n + j;
  return i * n + j;
}
// LayerNorm of rows [q0, q0 + cnt) of a rectangle, read from the pair where they lie (zeros for padding),
// and their mask - a warp a row
template <class TO>
__global__ void rectLayerNormK(const float* pair, const float* mask, TO* out, float* m, TriRect r, size_t q0, size_t cnt,
                               int n, int C, const float* scale, const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= cnt) return;
  size_t pw = 0, p = rectPair(r, q0 + row, n, &pw);
  const float* x = p != SIZE_MAX ? pair + pw * C : nullptr;
  float s = 0, ss = 0;
  for (int c = lane; c < C; c += 32) { float v = x ? x[c] : 0.f; s += v; ss += v * v; }
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(~0u, s, o); ss += __shfl_xor_sync(~0u, ss, o); }
  float mean = s / C, inv = rsqrtf(ss / C - mean * mean + 1e-5f);
  for (int c = lane; c < C; c += 32) out[row * C + c] = fromF<TO>(((x ? x[c] : 0.f) - mean) * inv * scale[c] + offset[c]);
  if (lane == 0 && m) m[row] = p != SIZE_MAX ? mask[p] : 0.f;
}
// one operand from a chunk's (rows, 2C) [projection | gate] rows, channel-major into the rectangle's
// buffer at q0 + row - through a 32 x 32 tile, so the reads run along the channels and the writes along
// the rows
template <class T>
__global__ void rectGateK(const T* pg, const float* m, T* out, size_t q0, size_t cnt, int C, size_t size) {
  __shared__ float A[32][33];
  size_t row0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t row = row0 + ry; int c = c0 + tx;
    float v = 0;
    if (row < cnt && c < C) { const T* p = pg + row * 2 * C; v = toF(p[c]) * m[row] * sigm(toF(p[C + c])); }
    A[ry][tx] = v;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t row = row0 + tx; int c = c0 + cy;
    if (row < cnt && c < C) out[(size_t)c * size + q0 + row] = fromF<T>(A[tx][cy]);
  }
}
// center_norm of rows [q0, q0 + cnt) of a rectangle's product (channel-major), row-major out: the
// channel-major tile read along the rows, 32 rows a block
template <class TO>
__global__ void rectCenterNormK(const float* prod, TO* out, size_t q0, size_t cnt, int C, size_t size,
                                const float* scale, const float* offset) {
  __shared__ float ps[8][33], pss[8][33], mean[32], inv[32];
  size_t row0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  size_t row = row0 + tx;
  float s = 0, ss = 0;
  if (row < cnt) for (int c = ty; c < C; c += 8) { float v = prod[(size_t)c * size + q0 + row]; s += v; ss += v * v; }
  ps[ty][tx] = s; pss[ty][tx] = ss;
  __syncthreads();
  if (ty == 0) {
    float a = 0, b = 0;
    for (int k = 0; k < 8; ++k) { a += ps[k][tx]; b += pss[k][tx]; }
    float mu = a / C; mean[tx] = mu; inv[tx] = rsqrtf(b / C - mu * mu + 1e-5f);
  }
  __syncthreads();
  // out[row][c]: each warp row of the block writes a row's channels; the reads of prod are strided by
  // `size`, a channel a lane - the norm's pass above was the coalesced one
  for (int ry = ty; ry < 32; ry += 8) {
    size_t lr = row0 + ry; if (lr >= cnt) continue;
    for (int c = tx; c < C; c += 32)
      out[lr * C + c] = fromF<TO>((prod[(size_t)c * size + q0 + lr] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}
// pair[at q] += t1 * sigmoid(t2) for the chunk's real rows
template <class T>
__global__ void rectGatedAddK(float* pair, const T* t1, const T* t2, TriRect r, size_t q0, size_t cnt, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  size_t row = t / C; int c = (int)(t % C);
  size_t pw = 0, p = rectPair(r, q0 + row, n, &pw);
  if (p != SIZE_MAX) pair[pw * C + c] += toF(t1[t]) * sigm(toF(t2[t]));
}

// [projection | gate] of ONE operand (side 0 = a, 1 = b) as a (C, 2C) matrix, from the interleaved (C, 2C)
// projection and gate (column 2ch is a's channel ch, 2ch + 1 b's)
__global__ void operandWeightK(const float* proj, const float* gate, float* out, int C, int side) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * 2 * C) return;
  int k = (int)(t / (2 * C)), o = (int)(t % (2 * C));
  out[t] = o < C ? proj[(size_t)k * 2 * C + 2 * o + side] : gate[(size_t)k * 2 * C + 2 * (o - C) + side];
}
inline std::string operandWeight(const std::string& pre, int C, int side) {
  std::string key = pre + ".operand" + std::to_string(side) + "~";
  if (!WF.count(key) && !WH.count(key)) {
    float* d = dalloc((size_t)C * 2 * C);
    operandWeightK<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(W(pre + ".projection"), W(pre + ".gate"), d, C, side);
    deviceWeight(key, d, (size_t)C * 2 * C);
  }
  return key;
}
// The whole triangle multiplication in BLOCKS, for a card short of room: the fixed operand b built whole
// (np^2 x C), then per block of OUTPUT rows (outgoing) or columns (incoming) the free operand a, the
// contraction and the output on that block alone - never a LayerNorm'd plane, a whole a or a whole
// product. Outgoing P[i, j] = sum_k a[i, k] b[j, k]: a row block reads pair rows the earlier blocks did
// not write. Incoming P[i, j] = sum_k a[k, j] b[k, i]: a COLUMN block reads pair columns the earlier
// blocks did not write - a row block would read rows they had. The reduction over k is never split.
// Tier 2: the fixed operand b's rows [i0, i0 + I) - the last window's padding rows too - from a window of whole
// rows, into b (np^2 x C, channel-major). On its own pass, or riding on the pass before (pairUpdates).
inline std::string HOST_B_READY;       // the triangle whose b is already built in trib.b (pairUpdates)
template <class T>
void triOperandRows(const float* w, size_t i0, size_t I, int n, int C, int np, const float* mask, const std::string& pre,
                    T* b) {
  size_t cs = (size_t)np * np, per = std::max<size_t>(32, CHUNK / (4 * C));
  std::string pg = operandWeight(pre, C, 1);
  float* m = scratch<float>("trib.mask", per);
  T* ln = scratch<T>("trib.ln", per * C); T* pgOut = scratch<T>("trib.pg", per * 2 * C);
  int rows = i0 + I == (size_t)n ? np - (int)i0 : (int)I;
  TriRect r{(int)i0, rows, 0, np, (int)i0, 0, n};
  for (size_t q0 = 0; q0 < r.size(); q0 += per) {
    size_t cnt = std::min(per, r.size() - q0);
    rectLayerNormK<T><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(w, mask, ln, m, r, q0, cnt, n, C,
      W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset"));
    linear<T, T>(ln, pgOut, cnt, C, 2 * C, pg);
    rectGateK<T><<<dim3((unsigned)((cnt + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(pgOut, m, b, i0 * np + q0,
                                                                                             cnt, C, cs);
  }
}
template <class T>
void triangleBlocked(float* pair, const float* mask, int n, int C, const std::string& pre, bool outgoing,
                     bool divideByLength, int np, const Pair* hp = nullptr, bool bBuilt = false,
                     const std::function<void(float*, size_t, size_t)>& after = nullptr) {
  size_t cs = (size_t)np * np;
  // each operand's own half of [projection | gate], so the two passes together project once
  std::string pgOf[2] = { operandWeight(pre, C, 0), operandWeight(pre, C, 1) };
  float alpha = divideByLength ? 1.f / n : 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  size_t per = std::max<size_t>(32, CHUNK / (4 * C));            // rows a chunk of the row-wise steps
  float* m = scratch<float>("trib.mask", per);
  T* ln = scratch<T>("trib.ln", per * C); T* pgOut = scratch<T>("trib.pg", per * 2 * C);
  // LN(pair) -> [projection | gate] -> one operand (side 0 = a, 1 = b), over a rectangle in chunks, into
  // `out` at base + q of a buffer `size` long per channel (the rectangle's own size unless it is a part of b)
  auto operands = [&](const float* src, const TriRect& r, T* out, int side, size_t size, size_t base) {
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      rectLayerNormK<T><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(src, mask, ln, m, r, q0, cnt, n, C,
        W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset"));
      linear<T, T>(ln, pgOut, cnt, C, 2 * C, pgOf[side]);
      rectGateK<T><<<dim3((unsigned)((cnt + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(pgOut, m, out, base + q0,
                                                                                               cnt, C, size);
    }
  };
  T* b = scratch<T>("trib.b", cs * C);
  // a pair in host memory: b from each window of rows in turn - unless the pass before built it already
  if (hp && bBuilt) {
    if (SCRATCH["trib.b"].second < cs * C * sizeof(T)) { fprintf(stderr, "%s: its fixed operand was not kept\n", pre.c_str()); exit(1); }
  } else if (hp) forRowWindows(*hp, false, [&](float* w, size_t i0, size_t I) { triOperandRows<T>(w, i0, I, n, C, np, mask, pre, b); });
  else operands(pair, {0, np, 0, np}, b, 1, cs, 0);
  // a block of the free operand and its product: at least CHUNK / C rows of the rectangle, and as many more as
  // what is free beside b allows (in tier 2 a window - hostWindowRows budgets for it - and on the card up to
  // 1024 rows) - every block reads the whole
  // of b (np^2 x C), so at 10761 tokens 48-row blocks read 29.6 GB 224 times a triangle, 4.4 s of memory traffic
  // for 1.1 s of arithmetic
  int width = (int)std::max<size_t>(8, std::min<size_t>(np, (CHUNK / C) / np / 8 * 8));
  if (hp) width = (int)std::max<size_t>(width, std::min<size_t>(np, (hostWindowRows(n, C) + 7) / 8 * 8));   // (its budget)
  else {
    size_t f, t; CK(cudaMemGetInfo(&f, &t));
    size_t perRow = (size_t)np * C * (sizeof(T) + 4), spare = f > t / 16 ? f - t / 16 : 0;    // a and prod, a row each
    width = (int)std::max<size_t>(width, std::min<size_t>(np, std::min<size_t>(spare / perRow, 1024) / 8 * 8));
  }
  T* a = scratch<T>("trib.a", (size_t)width * np * C);
  float* prod = scratch<float>("trib.prod", (size_t)width * np * C);
  T* t1 = scratch<T>("trib.t1", per * C); T* t2 = scratch<T>("trib.t2", per * C);
  // the output block at rows (outgoing) or columns (incoming) [k0, k0 + w), of `src` - the pair, or a window
  // whose rows begin at wi0 (columns at wj0) and are ws wide
  auto block = [&](float* src, int k0, int w, int wi0, int wj0, int ws) {
    TriRect r = outgoing ? TriRect{k0, w, 0, np, wi0, wj0, ws} : TriRect{0, np, k0, w, wi0, wj0, ws};
    operands(src, r, a, 0, r.size(), 0);
    if (outgoing)          // (as the whole contraction, with w output rows)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, np, w, np, &alpha, b, cudaType<T>(), np, cs, a,
        cudaType<T>(), np, r.size(), &zero, prod, CUDA_R_32F, np, r.size(), C, CUBLAS_COMPUTE_32F, algo));
    else                   // (with w output columns)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, w, np, np, &alpha, a, cudaType<T>(), w, r.size(), b,
        cudaType<T>(), np, cs, &zero, prod, CUDA_R_32F, w, r.size(), C, CUBLAS_COMPUTE_32F, algo));
    for (size_t q0 = 0; q0 < r.size(); q0 += per) {
      size_t cnt = std::min(per, r.size() - q0);
      rectCenterNormK<T><<<(unsigned)((cnt + 31) / 32), dim3(32, 8), 0, STREAM>>>(prod, ln, q0, cnt, C, r.size(),
        W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"));
      linear<T, T>(ln, t1, cnt, C, C, pre + ".outputProjection");
      rectLayerNormK<T><<<(unsigned)((cnt + 7) / 8), 256, 0, STREAM>>>(src, mask, ln, nullptr, r, q0, cnt, n, C,
        W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset"));
      linear<T, T>(ln, t2, cnt, C, C, pre + ".gatingLinear");
      rectGatedAddK<T><<<blocks(cnt * C), 256, 0, STREAM>>>(src, t1, t2, r, q0, cnt, n, C);
    }
  };
  if (!hp) for (int k0 = 0; k0 < n; k0 += width) block(pair, k0, std::min(width, np - k0), 0, 0, 0);
  // ...in host memory each window of output rows (columns) in turn, its blocks inside it: a block reads only
  // its own rows (columns), which no earlier block wrote
  // (`after` sees each window once its blocks are done, before it is written back)
  else if (outgoing) forRowWindows(*hp, true, [&](float* w, size_t i0, size_t I) {
    for (int k0 = (int)i0; k0 < (int)(i0 + I); k0 += width) block(w, k0, std::min(width, (int)(i0 + I) - k0), (int)i0, 0, n);
    if (after) after(w, i0, I);
  });
  else forColWindows(*hp, true, [&](float* w, size_t j0, size_t J) {
    for (int k0 = (int)j0; k0 < (int)(j0 + J); k0 += width) block(w, k0, std::min(width, (int)(j0 + J) - k0), 0, (int)j0, (int)J);
    if (after) after(w, j0, J);
  });
  // given back at once: the fixed operand is a plane, and the next stage (the MSA attention, the grid
  // attention) peaks beside the pair too - at these sizes a reallocation a call is nothing
  releaseScratch({ "trib." });
}
template <class T>
void triangle(float* pair, const float* mask, int n, int C, const std::string& pre, bool outgoing,
              bool divideByLength) {
  size_t pairs = (size_t)n * n;
  // a, b and their product live in a PADDED np x np space per channel, np a multiple of 8, the
  // padding zero: the contraction is unchanged and cuBLAS's GEMM runs twice as fast on an aligned
  // size (1044: 5.4 against 2.7 ms the pair of them; 522 the same). A multiple of 8 is enough for
  // the GEMM, and 32 padded the fused kernels' rows too (68 tokens: 96^2 against 72^2): trunk
  // 76.6 -> 72.3 ms at 68 tokens, 442 -> 438 at 261, 1617 -> 1606 at 522, flat at 150 and 1044
  int np = TRI_PAD ? (n + TRI_PAD - 1) / TRI_PAD * TRI_PAD : n;
  size_t cs = (size_t)np * np;
  // on a card short of room, in blocks of the output (triangleBlocked): no whole a, product or
  // LayerNorm'd plane - whichever kernels the device would otherwise run
  // (blocking costs time - a fifth of a trunk at 2096 tokens - so only where the whole form's operands,
  // product and gate, five planes, would not fit with room to spare)
  // (whichever form runs gives back the other's buffers first: scratch outlives the call, so a whole form
  // taken while there was room would otherwise sit beside the blocks of the next call, which had none)
  if (shortPair(pairs, C)) {
    if (!roomFor(5 * cs * C * 2, { "tri.a", "tri.b", "tri.prod", "tri.norm", "tri.abf", "tri.bbf", "tri.pbf", "tri.t2whole" })) {
      releaseScratch({ "tri.a", "tri.b", "tri.prod", "tri.norm", "tri.abf", "tri.bbf", "tri.pbf", "tri.t2whole" });
      triangleBlocked<T>(pair, mask, n, C, pre, outgoing, divideByLength, np);
      return;
    }
    releaseScratch({ "trib." });
  }
  std::string pg = projectionGate(pre, C);
  T *a = nullptr, *b = nullptr;
  float* prod = nullptr;          // f32: in f16 the contraction (a sum over n of products) overflows
  auto buffers = [&]() {
    a = scratch<T>("tri.a", cs * C); b = scratch<T>("tri.b", cs * C); prod = scratch<float>("tri.prod", cs * C);
    if (np != n) {                // the padding is written by nothing here
      CK(cudaMemsetAsync(a, 0, cs * C * sizeof(T), STREAM)); CK(cudaMemsetAsync(b, 0, cs * C * sizeof(T), STREAM));
    }
  };
  float alpha = divideByLength ? 1.f / n : 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  auto contract = [&]() {         // one n x n GEMM per channel: outgoing P = A B^T, incoming P = B^T A
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, np, np, np, &alpha, b, cudaType<T>(), np,
        cs, a, cudaType<T>(), np, cs, &zero, prod, CUDA_R_32F, np, cs, C, CUBLAS_COMPUTE_32F, algo));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, np, np, np, &alpha, a, cudaType<T>(), np,
        cs, b, cudaType<T>(), np, cs, &zero, prod, CUDA_R_32F, np, cs, C, CUBLAS_COMPUTE_32F, algo));
  };
  if constexpr (std::is_same_v<T, half>) {
    if (FUSED_WIDE && FUSED_TRIANGLE && n >= FUSED_WIDE_MIN_TOKENS && wideFits(C)) {
      // LN, the projection, the gate and the gating linear in one kernel (writing the padding), the f16
      // contraction into f32, then the centre norm, the output projection, the gate and the residual
      a = scratch<T>("tri.a", cs * C); b = scratch<T>("tri.b", cs * C); prod = scratch<float>("tri.prod", cs * C);
      half* t2 = scratch<half>("tri.t2whole", cs * C);
      wideWidth(C, [&](auto width) {
        constexpr int CC = decltype(width)::value, WI = 8, WO = 4;
        static bool attr = false;
        if (!attr) {
          smemAttr((triIn256K<CC, WI>), (int)wideTriInSmem(CC));
          attr = true;
        }
        triIn256K<CC, WI><<<(unsigned)((cs + 16 * WI - 1) / (16 * WI)), 32 * WI, wideTriInSmem(CC), STREAM>>>(
          pair, mask, W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset"), Wh(pg), Wh(pre + ".gatingLinear"),
          a, b, t2, n, np, cs);
        contract();
        triangleOutRun<CC, WO>(prod, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"), Wh(pre + ".outputProjection"), t2, pair, n, np);
      });
      return;
    }
    if (FUSED_TRIANGLE && C == 128 && (TRI_BF16 ? triFusedFits<__nv_bfloat16>() : triFusedFits<float>())) {   // see fusedtriangle.cuh
      half* t2 = scratch<half>("tri.t2whole", cs * C);
      if (TRI_BF16) {
        // a, b and the contraction's product in bf16: f32's range at half the bytes (f16's
        // range is what overflowed), AF3's own activation precision
        __nv_bfloat16* ab = scratch<__nv_bfloat16>("tri.abf", cs * C);
        __nv_bfloat16* bb = scratch<__nv_bfloat16>("tri.bbf", cs * C);
        __nv_bfloat16* pb = scratch<__nv_bfloat16>("tri.pbf", cs * C);
        triIn128(pair, mask, pre, pg, ab, bb, t2, n, np, cs);
        triContractBf16(outgoing, np, cs, C, alpha, ab, bb, pb);
        triOut128(pb, pre, t2, pair, n, np, cs);
        return;
      }
      a = scratch<T>("tri.a", cs * C); b = scratch<T>("tri.b", cs * C); prod = scratch<float>("tri.prod", cs * C);
      triIn128(pair, mask, pre, pg, a, b, t2, n, np, cs);   // writes the padding itself
      contract();
      triOut128(prod, pre, t2, pair, n, np, cs);
      return;
    }
  }
  buffers();
  T* norm = scratch<T>("tri.norm", pairs * C);
  size_t rowsPer = std::max<size_t>(1, CHUNK / (4 * C));
  T* pgOut = scratch<T>("tri.pg", rowsPer * 4 * C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    layerNorm2<float, T>(pair + r0 * C, norm + r0 * C, rows, C, pre + ".leftNormInputScale",
                         pre + ".leftNormInputOffset");
    linear<T, T>(norm + r0 * C, pgOut, rows, C, 4 * C, pg);
    triGateK<T><<<dim3((unsigned)((rows + 31) / 32), (C + 31) / 32), dim3(32, 8), 0, STREAM>>>(
      pgOut, mask, a, b, r0, rows, C, cs, n, np);
  }
  contract();
  rowsPer = std::max<size_t>(1, CHUNK / C);
  T* centred = scratch<T>("tri.centred", std::min(rowsPer, pairs) * C);
  T* t1 = scratch<T>("tri.t1", std::min(rowsPer, pairs) * C);
  T* t2 = scratch<T>("tri.t2", std::min(rowsPer, pairs) * C);
  for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
    size_t rows = std::min(rowsPer, pairs - r0);
    if (fitsSmem((size_t)C * 33 * 4)) {   // C * 33 floats of shared memory: past the 48 KB default from C = 373 (IntelliFold-2's 512)
      static int granted = 0;
      if (C * 33 * 4 > granted) {
        smemAttr((centerNormK<T>), C * 33 * 4);
        granted = C * 33 * 4;
      }
      centerNormK<T><<<(unsigned)((rows + 31) / 32), dim3(32, 8), C * 33 * 4, STREAM>>>(prod, centred, r0,
        rows, C, cs, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"), n, np);
    } else {
      centerNormStreamK<T><<<(unsigned)((rows + 31) / 32), dim3(32, 8), 0, STREAM>>>(prod, centred, r0,
        rows, C, cs, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"), n, np);
    }
    linear<T, T>(centred, t1, rows, C, C, pre + ".outputProjection");
    linear<T, T>(norm + r0 * C, t2, rows, C, C, pre + ".gatingLinear");
    gatedAddK<T><<<blocks(rows * C), 256, 0, STREAM>>>(pair + r0 * C, t1, t2, rows * C);
  }
}

// ---------------------------------------------------------------- transition
#include "fusedtransition.cuh"
template <class T>
void transition(float* x, size_t rows, int C, int factor, const std::string& pre) {
  // the width is the weight's (OpenDDE's refiner is factor 2 where its trunk is 4)
  (void)factor;
  size_t w1 = lenW(pre + ".transition1");
  if (w1 % (2 * (size_t)C)) { fprintf(stderr, "%s.transition1 has %zu elements, not C %d x 2I\n", pre.c_str(), w1, C); exit(1); }
  int I = (int)(w1 / (2 * (size_t)C));
  if constexpr (std::is_same_v<T, half>) if (fusedTransition(x, rows, C, I, pre)) return;
  if constexpr (std::is_same_v<T, half>) {
    // 256 channels: LN, the widening and SwiGLU in one kernel (fused256.cuh), then the second GEMM with the
    // residual as its beta - the [rows, 2I] widening never written
    if (FUSED_WIDE && FUSED_TRANSITION && rows >= (size_t)FUSED_WIDE_MIN_TOKENS * FUSED_WIDE_MIN_TOKENS && wideFits(C)) {
      constexpr int WU = 8, R = 16 * WU;
      // whole waves of transitionUpK inside the same budget (transitionUpChunkRows)
      size_t rowsPer = transitionUpChunkRows<256, WU>(wideUpSmem(256), std::max<size_t>(R, CHUNK / (2 * I)));
      half* gated = scratch<half>("tr.gated", std::min(rowsPer, rows) * I);
      wideWidth(C, [&](auto width) {
        constexpr int CC = decltype(width)::value;
        static bool attr = false;
        if (!attr) { smemAttr((transitionUpK<CC, WU>), (int)wideUpSmem(CC)); attr = true; }
        for (size_t r0 = 0; r0 < rows; r0 += rowsPer) {
          size_t r = std::min(rowsPer, rows - r0);
          transitionUpK<CC, WU><<<(unsigned)((r + R - 1) / R), 32 * WU, wideUpSmem(CC), STREAM>>>(
            x + r0 * C, W(pre + ".inputLayerNormScale"), W(pre + ".inputLayerNormOffset"), Wh(pre + ".transition1"), gated, r, I);
          linear<half, float>(gated, x + r0 * C, r, I, C, pre + ".transition2", false, 1.f);
        }
      });
      return;
    }
  }
  size_t rowsPer = std::max<size_t>(1, CHUNK / (2 * I));
  T* xn = scratch<T>("tr.x", std::min(rowsPer, rows) * C);
  T* wide = scratch<T>("tr.wide", std::min(rowsPer, rows) * 2 * I);
  T* gated = scratch<T>("tr.gated", std::min(rowsPer, rows) * I);
  for (size_t r0 = 0; r0 < rows; r0 += rowsPer) {
    size_t r = std::min(rowsPer, rows - r0);
    layerNorm2<float, T>(x + r0 * C, xn, r, C, pre + ".inputLayerNormScale", pre + ".inputLayerNormOffset");
    linear<T, T>(xn, wide, r, C, 2 * I, pre + ".transition1");
    swiglu<T>(wide, gated, r, I);
    linear<T, float>(gated, x + r0 * C, r, I, C, pre + ".transition2", false, 1.f);
  }
}

// ---------------------------------------------------------------- grid attention kernels
#include "flash.cuh"

// act[r][j] = norm[j][r] for the column direction, 16 bytes a thread (C * elem a multiple of 16)
__global__ void gatherTransposedK(const void* normV, void* actV, int n, int C, size_t r0, size_t R,
                                  int elem) {
  int chunks = C * elem / 16;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * chunks) return;
  int c = (int)(t % chunks); size_t rest = t / chunks; size_t j = rest % n; size_t r = r0 + rest / n;
  reinterpret_cast<uint4*>(actV)[t] = reinterpret_cast<const uint4*>(normV)[(j * n + r) * chunks + c];
}
// bias[h][i][stride] (j padded to a multiple of 8, zeros past n), scaled by `scale`
template <class TB>
__global__ void biasLayoutK(const float* raw, TB* bias, int n, int stride, int heads, bool swap,
                            float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)heads * n * stride;
  if (t >= total) return;
  int j = (int)(t % stride); size_t rest = t / stride; int i = (int)(rest % n); int h = (int)(rest / n);
  float v = j < n ? scale * raw[(swap ? ((size_t)j * n + i) : ((size_t)i * n + j)) * heads + h] : 0.f;
  bias[t] = fromF<TB>(v);
}
// the same from head-major raw logits [hp][pairs] (hp >= heads, the padded projection's width)
template <class TB>
__global__ void biasLayoutHeadMajorK(const float* raw, TB* bias, int n, int stride, int heads, bool swap,
                                     float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)heads * n * stride, pairs = (size_t)n * n;
  if (t >= total) return;
  int j = (int)(t % stride); size_t rest = t / stride; int i = (int)(rest % n); int h = (int)(rest / n);
  float v = j < n ? scale * raw[(size_t)h * pairs + (swap ? ((size_t)j * n + i) : ((size_t)i * n + j))] : 0.f;
  bias[t] = fromF<TB>(v);
}
// the same from a chunk of pairs [r0, r0 + cnt) of raw (head-major over the chunk: [heads'][cnt]) - the
// padding columns are the caller's to zero
template <class TB>
__global__ void biasFromRawRowsK(const float* raw, TB* bias, size_t r0, size_t cnt, int n, int stride, int heads,
                                 bool swap, float scale, size_t wc = 0, size_t j0 = 0) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * cnt) return;
  // pair p is (a, b) - p counted over a column window of wc columns from j0 where wc is given (tier 2)
  size_t h = t / cnt, q = t % cnt, p = r0 + q, w = wc ? wc : n, a = p / w, b = j0 + p % w;
  size_t i = swap ? b : a, j = swap ? a : b;
  bias[(h * n + i) * stride + j] = fromF<TB>(scale * raw[h * cnt + q]);
}
// ...and from a chunk's row-major raw logits [cnt][heads] (a plain projection's output)
template <class TB>
__global__ void biasFromFlatRowsK(const float* raw, TB* bias, size_t r0, size_t cnt, int n, int stride, int heads,
                                  bool swap, float scale, size_t wc = 0, size_t j0 = 0) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)heads * cnt) return;
  size_t h = t % heads, q = t / heads, p = r0 + q, w = wc ? wc : n, a = p / w, b = j0 + p % w;
  size_t i = swap ? b : a, j = swap ? a : b;
  bias[(h * n + i) * stride + j] = fromF<TB>(scale * raw[q * heads + h]);
}
// a (C, k) weight zero-padded to (C, kp) columns
inline std::string paddedColumns(const std::string& w, int C, int k, int kp) {
  return concatColumns(w + "~pad" + std::to_string(kp), C, {{w, k, false}, {"", kp - k, false}});
}
// pair[(r, j) or (j, r)] += out[r][j], four channels a thread (C a multiple of 4)
__global__ void addGridK(float* pair, const float* out, int n, int C, size_t r0, size_t R, bool tr) {
  int c4 = C / 4;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * n * c4) return;
  int c = (int)(t % c4); size_t rest = t / c4; size_t j = rest % n; size_t r = r0 + rest / n;
  size_t to = tr ? (j * n + r) : (r * n + j);
  float4* p = reinterpret_cast<float4*>(pair) + to * c4 + c;
  float4 v = *p, o = reinterpret_cast<const float4*>(out)[t];
  v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
  *p = v;
}
template <class T>
__global__ void addGateBiasK(T* qkvg, const float* bias, size_t rows, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * Wd) return;
  size_t r = t / Wd; int c = (int)(t % Wd);
  T& g = qkvg[r * 4 * Wd + 3 * Wd + c];
  g = fromF<T>(toF(g) + bias[c]);
}

// Grid attention over the pair, rows (tr = false) or columns (tr = true), residual added.
template <class T>
void gridAttention(float* pair, const float* mask, int n, int C, int heads, int D,
                   const std::string& pre, bool tr, bool swapBias) {
  size_t pairs = (size_t)n * n; int Wd = heads * D;
  if constexpr (std::is_same_v<T, half>) {
    // three fused kernels (fusedtriangle.cuh): LN + the bias projection (head-major), then per
    // chunk LN + q/k/v/gate (reading the column direction's rows transposed in place), the flash
    // kernel, and the output projection added into the pair
    if (FUSED_GRID && C == 128 && Wd == 128 && heads <= 16 && gridFusedFits() && !hasW(pre + ".gatingQueryBias") &&
        !hasW(pre + ".outputProjectionBias")) {
      int stride = (n + 7) / 8 * 8;
      half* bias = scratch<half>("grid.bias", (size_t)heads * n * stride);
      std::string qkvg = qkvgWeight(pre, C, Wd, true);
      std::string wb = paddedColumns(pre + ".pairBiasProjection", C, heads, 16);
      float scale = 1.f / sqrtf((float)D);
      // 🔴 every row in one pass only while its q/k/v/gate are a 32nd of the card: whole is 3.3% of
      // the trunk faster at 262 tokens and 4.8% at 1048 (465 against 481 ms, 7.82 against 8.20 s on an
      // A100), and at 1048 it is 1.13 GB the chunks do not hold (trunk peak 12.88 against 11.73 GB) -
      // nothing on 40 GB, the difference between fitting and not near a T4's 15
      static const size_t gridWhole = [] { size_t f, t; CK(cudaMemGetInfo(&f, &t)); return t / 32; }();
      if (!BIG_FORCED && pairs * 4 * Wd * 2 <= gridWhole) {
        // every row in one pass (this card has the memory): the bias written by the same kernel
        CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * 2, STREAM));    // the padding columns
        half* qkvgOut = scratch<half>("grid.qkvg", (pairs + 128) * 4 * Wd);
        gridIn128(pair, pre, qkvg, qkvgOut, n, 0, pairs, tr, Wh(wb), bias, heads, stride, tr && swapBias);
        half* gathered = scratch<half>("grid.gathered", pairs * Wd);
        flashGrid<half>(qkvgOut, bias, stride, MASK_ALL_ONES ? nullptr : mask, gathered, n, heads, D, 0, n, tr, scale);
        if (!tr) linear<half, float>(gathered, pair, pairs, Wd, C, pre + ".outputProjection", false, 1.f);
        else gridOut128(gathered, pre + ".outputProjection", pair, n, 0, pairs, tr);
        return;
      }
      if (shortPair(pairs, C)) {
        // on a card short of room the 16-column projection in chunks of pairs, each laid into the bias
        size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / 16));
        float* raw = scratch<float>("grid.raw16", per * 16);
        CK(cudaMemsetAsync(bias, 0, (size_t)heads * n * stride * 2, STREAM));     // the padding columns
        for (size_t r0 = 0; r0 < pairs; r0 += per) {
          size_t r = std::min(per, pairs - r0);
          lnHeads128<16>(pair + r0 * C, pre + ".actNormScale", pre + ".actNormOffset", wb, raw, r);
          biasFromRawRowsK<half><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, r0, r, n, stride, heads,
                                                                              tr && swapBias, LOG2E);
        }
      } else {
      float* raw = scratch<float>("grid.raw16", pairs * 16);
      lnHeads128<16>(pair, pre + ".actNormScale", pre + ".actNormOffset", wb, raw, pairs);
      biasLayoutHeadMajorK<half><<<blocks((size_t)heads * n * stride), 256, 0, STREAM>>>(
        raw, bias, n, stride, heads, tr && swapBias, LOG2E);
      }
      size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * 4 * Wd)));
      for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
        size_t rows = std::min(R, (size_t)n - r0), prs = rows * n;
        half* qkvgOut = scratch<half>("grid.qkvg", (prs + 128) * 4 * Wd);   // padding: the last query block
        gridIn128(pair, pre, qkvg, qkvgOut, n, r0 * n, prs, tr);
        half* gathered = scratch<half>("grid.gathered", prs * Wd);
        flashGrid<half>(qkvgOut, bias, stride, MASK_ALL_ONES ? nullptr : mask, gathered, n, heads, D, r0, rows, tr, scale);
        if (!tr) linear<half, float>(gathered, pair + r0 * n * C, prs, Wd, C, pre + ".outputProjection", false, 1.f);
        else gridOut128(gathered, pre + ".outputProjection", pair, n, r0 * n, prs, tr);
      }
      return;
    }
  }
  // On a card short of room the LayerNorm'd pair is not kept: it is per pair position, so the bias pass and
  // each chunk's q/k/v/gate take it again from the pair (a column chunk gathered transposed first). In
  // place that is safe: a row chunk writes only its own rows, a column chunk only its own columns, and the
  // bias is all taken before any is written.
  const bool streamNorm = shortPair(pairs, C);
  T* norm = streamNorm ? nullptr : scratch<T>("grid.norm", pairs * C);
  float* raw = scratch<float>("grid.rawbias", pairs * heads);
  if (!streamNorm) {
    layerNorm2<float, T>(pair, norm, pairs, C, pre + ".actNormScale", pre + ".actNormOffset");
    linear<T, float>(norm, raw, pairs, C, heads, pre + ".pairBiasProjection");
  } else {
    size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / C));
    T* lnc = scratch<T>("grid.normChunk", per * C);
    for (size_t r0 = 0; r0 < pairs; r0 += per) {
      size_t r = std::min(per, pairs - r0);
      layerNorm2<float, T>(pair + r0 * C, lnc, r, C, pre + ".actNormScale", pre + ".actNormOffset");
      linear<T, float>(lnc, raw + r0 * heads, r, C, heads, pre + ".pairBiasProjection");
    }
  }
  int stride = (n + 7) / 8 * 8;
  constexpr bool fast = std::is_same_v<T, half>;
  T* bias = scratch<T>("grid.bias", (size_t)heads * n * stride);
  biasLayoutK<T><<<blocks((size_t)heads * n * stride), 256, 0, STREAM>>>(
    raw, bias, n, stride, heads, tr && swapBias, fast ? LOG2E : 1.f);
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  const float* gateBias = hasW(pre + ".gatingQueryBias") ? W(pre + ".gatingQueryBias") : nullptr;
  const float* outBias = hasW(pre + ".outputProjectionBias") ? W(pre + ".outputProjectionBias") : nullptr;
  size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * 4 * Wd)));
  float scale = 1.f / sqrtf((float)D);
  for (size_t r0 = 0; r0 < (size_t)n; r0 += R) {
    size_t rows = std::min(R, (size_t)n - r0), prs = rows * n;
    const T* act = streamNorm ? nullptr : norm + r0 * n * C;
    if (streamNorm) {
      T* g = scratch<T>("grid.act", prs * C);
      const float* src = pair + r0 * n * C;
      if (tr) {
        float* g32 = scratch<float>("grid.act32", prs * C);
        if (C * 4 % 16) { fprintf(stderr, "grid attention: %d channels are not 16-byte rows\n", C); exit(1); }
        gatherTransposedK<<<blocks(prs * C * 4 / 16), 256, 0, STREAM>>>(pair, g32, n, C, r0, rows, 4);
        src = g32;
      }
      layerNorm2<float, T>(src, g, prs, C, pre + ".actNormScale", pre + ".actNormOffset");
      act = g;
    } else if (tr) {
      T* g = scratch<T>("grid.act", prs * C);
      if (C * sizeof(T) % 16) { fprintf(stderr, "grid attention: %d channels are not 16-byte rows\n", C); exit(1); }
      gatherTransposedK<<<blocks(prs * C * sizeof(T) / 16), 256, 0, STREAM>>>(norm, g, n, C, r0, rows, sizeof(T));
      act = g;
    }
    T* qkvgOut = scratch<T>("grid.qkvg", (prs + 128) * 4 * Wd);   // padding: the last query block
    linear<T, T>(act, qkvgOut, prs, C, 4 * Wd, qkvg);
    if (gateBias) addGateBiasK<T><<<blocks(prs * Wd), 256, 0, STREAM>>>(qkvgOut, gateBias, prs, Wd);
    T* gathered = scratch<T>("grid.gathered", prs * Wd);
    flashGrid<T>(qkvgOut, bias, stride, MASK_ALL_ONES && std::is_same_v<T, half> ? nullptr : mask, gathered, n, heads, D,
                 r0, rows, tr, scale);       // (no mask when every token is real: the unmasked kernel)
    if (!tr && !outBias) {
      linear<T, float>(gathered, pair + r0 * n * C, prs, Wd, C, pre + ".outputProjection", false, 1.f);
      continue;
    }
    float* o = scratch<float>("grid.out", prs * C);
    linear<T, float>(gathered, o, prs, Wd, C, pre + ".outputProjection");
    if (outBias) addBiasK<<<blocks(prs * C), 256, 0, STREAM>>>(o, outBias, prs, C);
    addGridK<<<blocks(prs * C / 4), 256, 0, STREAM>>>(pair, o, n, C, r0, rows, tr);
  }
}

// dst[b][a] = src[a][b], rows of `chunks` 16-byte pieces: a column window [n][J][C] <-> [J][n][C]
__global__ void transposeWindowK(const uint4* src, uint4* dst, size_t A, size_t B, int chunks) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= A * B * chunks) return;
  int c = (int)(t % chunks); size_t rest = t / chunks, a = rest % A, b = rest / A;
  dst[t] = src[(a * B + b) * chunks + c];
}
// whether tier 2's grid attention takes the fused kernels (gridAttention's own test)
template <class T> bool gridHostFused(int C, int heads, int D, const std::string& pre) {
  if constexpr (std::is_same_v<T, half>)
    return FUSED_GRID && C == 128 && heads * D == 128 && heads <= 16 && gridFusedFits() && !hasW(pre + ".gatingQueryBias") &&
           !hasW(pre + ".outputProjectionBias");
  return false;
}
// The grid attention's bias from `cnt` positions of a window - the pair positions from p0 (wc 0), or a column
// window's, counted from p0 over its wc columns from j0 - into the flash kernel's layout (the padding columns
// are the caller's to zero)
template <class T>
void gridBiasPart(const float* w, size_t cnt, size_t p0, size_t wc, size_t j0, T* bias, int n, int C, int heads, int D,
                  const std::string& pre, bool tr, bool swapBias) {
  bool fused = gridHostFused<T>(C, heads, D, pre);
  constexpr bool fast = std::is_same_v<T, half>;
  int stride = (n + 7) / 8 * 8;
  std::string wb = fused ? paddedColumns(pre + ".pairBiasProjection", C, heads, 16) : "";
  size_t per = std::max<size_t>(1, std::min((size_t)n * n, CHUNK / std::max(C, 16)));
  float* raw = scratch<float>("grid.raw16", per * 16);
  for (size_t q = 0; q < cnt; q += per) {
    size_t r = std::min(per, cnt - q);
    if (fused) {
      if constexpr (std::is_same_v<T, half>) lnHeads128<16>(w + q * C, pre + ".actNormScale", pre + ".actNormOffset", wb, raw, r);
      biasFromRawRowsK<T><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, p0 + q, r, n, stride, heads,
                                                                        tr && swapBias, LOG2E, wc, j0);
    } else {
      T* lnc = scratch<T>("grid.normChunk", per * C);
      layerNorm2<float, T>(w + q * C, lnc, r, C, pre + ".actNormScale", pre + ".actNormOffset");
      linear<T, float>(lnc, raw, r, C, heads, pre + ".pairBiasProjection");
      biasFromFlatRowsK<T><<<blocks((size_t)heads * r), 256, 0, STREAM>>>(raw, bias, p0 + q, r, n, stride, heads,
                                                                         tr && swapBias, fast ? LOG2E : 1.f, wc, j0);
    }
  }
}
// Grid attention over a pair in host memory (tier 2), its bias built on an earlier pass (pairUpdates): each
// window of rows - or of columns, turned on the device so its columns are rows and the row-direction kernels
// run on it (the flash kernel keeps `tr`, which reads only the mask). `after` sees each window of rows once
// attended, before it is written back.
template <class T>
void gridAttentionHost(const Pair& P, const float* mask, int n, int C, int heads, int D, const std::string& pre,
                       bool tr, T* bias, const std::function<void(float*, size_t, size_t)>& after = nullptr) {
  int Wd = heads * D, stride = (n + 7) / 8 * 8;
  float scale = 1.f / sqrtf((float)D);
  bool fused = gridHostFused<T>(C, heads, D, pre);
  constexpr bool fast = std::is_same_v<T, half>;
  std::string qkvg = qkvgWeight(pre, C, Wd, true);
  const float* gateBias = hasW(pre + ".gatingQueryBias") ? W(pre + ".gatingQueryBias") : nullptr;
  const float* outBias = hasW(pre + ".outputProjectionBias") ? W(pre + ".outputProjectionBias") : nullptr;
  size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * 4 * Wd)));
  // rows [a0, a0 + A) of the attention's own order (pair rows, or pair columns), in `rows` - a window holding
  // them from its row 0 at attention row w0
  auto attend = [&](float* rows, size_t w0, size_t A) {
    for (size_t r0 = w0; r0 < w0 + A; r0 += R) {
      size_t cnt = std::min(R, w0 + A - r0), prs = cnt * n;
      float* local = rows + (r0 - w0) * n * C;
      T* qkvgOut = scratch<T>("grid.qkvg", (prs + 128) * 4 * Wd);    // padding: the last query block
      if (fused) {
        if constexpr (std::is_same_v<T, half>) gridIn128(local, pre, qkvg, qkvgOut, n, 0, prs, false);
      } else {
        T* g = scratch<T>("grid.act", prs * C);
        layerNorm2<float, T>(local, g, prs, C, pre + ".actNormScale", pre + ".actNormOffset");
        linear<T, T>(g, qkvgOut, prs, C, 4 * Wd, qkvg);
        if (gateBias) addGateBiasK<T><<<blocks(prs * Wd), 256, 0, STREAM>>>(qkvgOut, gateBias, prs, Wd);
      }
      T* gathered = scratch<T>("grid.gathered", prs * Wd);
      flashGrid<T>(qkvgOut, bias, stride, MASK_ALL_ONES && fast ? nullptr : mask, gathered, n, heads, D, r0, cnt, tr, scale);
      if (!outBias) { linear<T, float>(gathered, local, prs, Wd, C, pre + ".outputProjection", false, 1.f); continue; }
      float* o = scratch<float>("grid.out", prs * C);
      linear<T, float>(gathered, o, prs, Wd, C, pre + ".outputProjection");
      addBiasK<<<blocks(prs * C), 256, 0, STREAM>>>(o, outBias, prs, C);
      addK<<<blocks(prs * C), 256, 0, STREAM>>>(local, o, prs * C);
    }
  };
  if (!tr) { forRowWindows(P, true, [&](float* w, size_t i0, size_t I) { attend(w, i0, I); if (after) after(w, i0, I); }); return; }
  if (C * 4 % 16) { fprintf(stderr, "grid attention: %d channels are not 16-byte rows\n", C); exit(1); }
  forColWindows(P, true, [&](float* w, size_t j0, size_t J) {
    float* turned = scratch<float>("hp.turned", J * n * C);
    int chunks = C * 4 / 16;
    transposeWindowK<<<blocks(J * n * chunks), 256, 0, STREAM>>>((const uint4*)w, (uint4*)turned, n, J, chunks);
    attend(turned, j0, J);
    transposeWindowK<<<blocks(J * n * chunks), 256, 0, STREAM>>>((const uint4*)turned, (uint4*)w, J, n, chunks);
  });
}

// The five pair updates of a pairformer/MSA/template block, in AF3's order.
template <class T>
void pairUpdates(float* pair, const float* mask, int n, int C, const std::string& pre, bool swap,
                 bool divide, int transitionFactor) {
  triangle<T>(pair, mask, n, C, pre + ".triangleMultiplicationOutgoing", true, divide); stage("tri.out");
  triangle<T>(pair, mask, n, C, pre + ".triangleMultiplicationIncoming", false, divide); stage("tri.in");
  int heads = (int)M.meta(pre + ".pairAttention1.heads"), D = (int)M.meta(pre + ".pairAttention1.dimension");
  gridAttention<T>(pair, mask, n, C, heads, D, pre + ".pairAttention1", false, swap); stage("grid.row");
  gridAttention<T>(pair, mask, n, C, heads, D, pre + ".pairAttention2", true, swap); stage("grid.col");
  transition<T>(pair, (size_t)n * n, C, transitionFactor, pre + ".pairTransition"); stage("transition");
}
// ...on a pair in host memory (tier 2): every update window by window, and each light pass - a fixed operand,
// an attention's bias, the single track's rows - riding on the window pass before it, whose rows (or columns)
// are on the card already: the row attention's bias off the incoming triangle's column windows, the column
// attention's off the row attention's windows, and (from the transition's windows) `finalRows` - the single
// track - and the next block's outgoing fixed operand, `nextOut`, where nothing touches the pair in between.
// Ten pair reads a pairformer block became six.
template <class T>
void pairUpdates(const Pair& P, const float* mask, const std::string& pre, bool swap, bool divide, int transitionFactor,
                 const std::function<void(const float*, size_t, size_t)>& finalRows = nullptr, const std::string& nextOut = "") {
  if (!P.host) {
    pairUpdates<T>(P.dev, mask, P.n, P.C, pre, swap, divide, transitionFactor);
    if (finalRows) finalRows(P.dev, 0, P.n);
    return;
  }
  int n = P.n, C = P.C, np = TRI_PAD ? (n + TRI_PAD - 1) / TRI_PAD * TRI_PAD : n, stride = (n + 7) / 8 * 8;
  std::string out = pre + ".triangleMultiplicationOutgoing", in = pre + ".triangleMultiplicationIncoming";
  std::string a1 = pre + ".pairAttention1", a2 = pre + ".pairAttention2";
  int h1 = (int)M.meta(a1 + ".heads"), D1 = (int)M.meta(a1 + ".dimension");
  int h2 = (int)M.meta(a2 + ".heads"), D2 = (int)M.meta(a2 + ".dimension");
  bool built = HOST_B_READY == out; HOST_B_READY.clear();
  triangleBlocked<T>(nullptr, mask, n, C, out, true, divide, np, &P, built); stage("tri.out");
  T* rowBias = scratch<T>("grid.biasRow", (size_t)h1 * n * stride);
  CK(cudaMemsetAsync(rowBias, 0, (size_t)h1 * n * stride * sizeof(T), STREAM));
  triangleBlocked<T>(nullptr, mask, n, C, in, false, divide, np, &P, false, [&](float* w, size_t j0, size_t J) {
    gridBiasPart<T>(w, (size_t)n * J, 0, J, j0, rowBias, n, C, h1, D1, a1, false, swap);
  }); stage("tri.in");
  T* colBias = scratch<T>("grid.biasCol", (size_t)h2 * n * stride);
  CK(cudaMemsetAsync(colBias, 0, (size_t)h2 * n * stride * sizeof(T), STREAM));
  gridAttentionHost<T>(P, mask, n, C, h1, D1, a1, false, rowBias, [&](float* w, size_t i0, size_t I) {
    gridBiasPart<T>(w, I * n, i0 * n, 0, 0, colBias, n, C, h2, D2, a2, true, swap);
  }); stage("grid.row");
  gridAttentionHost<T>(P, mask, n, C, h2, D2, a2, true, colBias); stage("grid.col");
  // (the column windows' turned copy and both attentions' biases: the next fixed operand wants the room)
  releaseScratch({ "hp.turned", "grid.biasRow", "grid.biasCol" });
  T* bNext = nextOut.empty() ? nullptr : scratch<T>("trib.b", (size_t)np * np * C);
  forRowWindows(P, true, [&](float* w, size_t i0, size_t I) {
    transition<T>(w, I * n, C, transitionFactor, pre + ".pairTransition");
    if (finalRows) finalRows(w, i0, I);
    if (bNext) triOperandRows<T>(w, i0, I, n, C, np, mask, nextOut, bNext);
  });
  if (bNext) HOST_B_READY = nextOut;
  stage("transition");
}
