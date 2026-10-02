// The --fast path's pieces: f16 activations into tensor-core GEMMs (f32 accumulation) with the bias,
// the ReLU and the residual add in cuBLASLt's epilogue, and weights re-laid once for them.
#pragma once
#include "ops.cuh"

// ---------------------------------------------------------------- cuBLASLt, row-major
// Y[rows, out] (f32 or f16) = X[rows, in] (f16) W[in, out] (f16) (+ bias[out] f32) (ReLU) (+ beta Y)
inline cublasLtHandle_t LT = nullptr;
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; bool ok; };
inline const void* biasFor(const float* bias, int n, bool asHalf) {
  if (!bias || !asHalf) return bias;
  static std::map<const float*, half*> copies;
  auto it = copies.find(bias);
  if (it != copies.end()) return it->second;
  half* h = dallocT<half>(n);
  toHalfK<<<blocks(n), 256, 0, STREAM>>>(bias, h, n);
  return copies[bias] = h;
}
inline void ltGemm(const half* X, const half* Wt, void* Y, bool yHalf, size_t rows, int in, int out, const float* biasF,
                   bool relu, float beta) {
  if (!LT) CB(cublasLtCreate(&LT));
  const void* bias = biasFor(biasF, out, yHalf);
  static std::map<std::tuple<size_t, int, int, bool, int, bool>, LtPlan> plans;
  int epi = bias ? (relu ? 2 : 1) : (relu ? 3 : 0);
  auto key = std::make_tuple(rows, in, out, yHalf, epi, beta != 0.f);
  auto it = plans.find(key);
  if (it == plans.end()) {
    LtPlan p{};
    CB(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtEpilogue_t e = epi == 1 ? CUBLASLT_EPILOGUE_BIAS : epi == 2 ? CUBLASLT_EPILOGUE_RELU_BIAS
                         : epi == 3 ? CUBLASLT_EPILOGUE_RELU : CUBLASLT_EPILOGUE_DEFAULT;
    CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
    // (the bias takes the output's type: this cuBLASLt refuses an f32 bias beside an f16 output, so an
    // f16 copy of it is handed over below)
    // col-major: C^T (out x rows) = W^T (out x in) X^T (in x rows)
    CB(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, out, in, out));
    CB(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, in, (uint64_t)rows, in));
    CB(cublasLtMatrixLayoutCreate(&p.c, yHalf ? CUDA_R_16F : CUDA_R_32F, out, (uint64_t)rows, out));
    cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 0;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    // no split-K with a separate reduction: with a bias and a beta it ran as a second kernel over the
    // whole output (cublasLt's epilogue globalKernel, ~5% of a 261-residue pass)
    uint32_t noReduction = CUBLASLT_REDUCTION_SCHEME_NONE;
    CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &noReduction, sizeof(noReduction)));
    cublasLtMatmulHeuristicResult_t res[1]; int got = 0;
    // the bias pointer is part of the descriptor, set per call below; the heuristic needs it present
    if (bias) CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(LT, p.op, p.a, p.b, p.c, p.c, pref, 1, res, &got);
    if (hs != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLASLt heuristic %d for %zu x %d x %d, f16 out %d, epilogue %d\n", (int)hs, rows, in, out, yHalf, epi);
      exit(1);
    }
    CB(cublasLtMatmulPreferenceDestroy(pref));
    if (got == 0) { fprintf(stderr, "no cuBLASLt algorithm for %zu x %d x %d\n", rows, in, out); exit(1); }
    p.algo = res[0].algo; p.ok = true;
    it = plans.emplace(key, p).first;
  }
  LtPlan& p = it->second;
  if (bias) CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  const float one = 1.f;
  CB(cublasLtMatmul(LT, p.op, &one, Wt, p.a, X, p.b, &beta, Y, p.c, Y, p.c, &p.algo, nullptr, 0, STREAM));
}
// a weight slice's f16 copy: haiku name, stacked block (or -1)
inline const half* PH(const std::string& name, int block = -1) {
  const half* base = Wh("w/" + name);
  if (block < 0) return base;
  return base + M.len("w/" + name) / dimW(name, 0) * block;
}

// ---------------------------------------------------------------- re-laid weights
// q, k, v, gate of one attention block as ONE [C, 4 * H * Dp] f16 matrix - each head padded to Dp
// columns (zeros: an 8-wide head padded to 16 changes no dot product) - and its bias (the gate's,
// zeros elsewhere); the output weight [H * Dp, C] likewise padded with zero rows
__global__ void packQkvgWeightK(const float* q, const float* k, const float* v, const float* g, half* out, int C, int H,
                                int D, int Dp) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int Wp = H * Dp;
  if (t >= (size_t)C * 4 * Wp) return;
  int col = (int)(t % (4 * Wp)), c = (int)(t / (4 * Wp));
  int role = col / Wp, hd = col % Wp, h = hd / Dp, d = hd % Dp;
  float val = 0;
  if (d < D) {
    const float* src = role == 0 ? q : role == 1 ? k : role == 2 ? v : g;
    val = src[(size_t)c * H * D + h * D + d];
  }
  out[t] = __float2half(val);
}
__global__ void packGateBiasK(const float* gb, float* out, int H, int D, int Dp) {
  int t = blockIdx.x * blockDim.x + threadIdx.x, Wp = H * Dp;
  if (t >= 4 * Wp) return;
  int role = t / Wp, hd = t % Wp, h = hd / Dp, d = hd % Dp;
  out[t] = (role == 3 && d < D) ? gb[h * D + d] : 0.f;
}
__global__ void packOutputWeightK(const float* o, half* out, int C, int H, int D, int Dp) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)H * Dp * C) return;
  int c = (int)(t % C); int hd = (int)(t / C), h = hd / Dp, d = hd % Dp;
  out[t] = __float2half(d < D ? o[((size_t)h * D + d) * C + c] : 0.f);
}
struct AttnW { const half* qkvg; const float* qkvgBias; const half* out; int H, D, Dp; };
inline AttnW attnWeights(const std::string& A, int blk, int C) {
  static std::map<std::pair<std::string, int>, AttnW> cache;
  auto it = cache.find({A, blk});
  if (it != cache.end()) return it->second;
  int H = (int)dimW(A + "/query_w", blk < 0 ? 1 : 2), D = (int)dimW(A + "/query_w", blk < 0 ? 2 : 3);
  int Dp = D < 16 ? 16 : D;
  AttnW w{}; w.H = H; w.D = D; w.Dp = Dp;
  half* qk = dallocT<half>((size_t)C * 4 * H * Dp);
  packQkvgWeightK<<<blocks((size_t)C * 4 * H * Dp), 256, 0, STREAM>>>(P(A + "/query_w", blk), P(A + "/key_w", blk),
    P(A + "/value_w", blk), P(A + "/gating_w", blk), qk, C, H, D, Dp);
  float* gb = dalloc(4 * H * Dp);
  packGateBiasK<<<blocks(4 * H * Dp), 256, 0, STREAM>>>(P(A + "/gating_b", blk), gb, H, D, Dp);
  half* ow = dallocT<half>((size_t)H * Dp * C);
  packOutputWeightK<<<blocks((size_t)H * Dp * C), 256, 0, STREAM>>>(P(A + "/output_w", blk), ow, C, H, D, Dp);
  w.qkvg = qk; w.qkvgBias = gb; w.out = ow;
  return cache[{A, blk}] = w;
}

// ---------------------------------------------------------------- f16 elementwise
template <class TO>
__global__ void layerNormTK(const float* x, TO* y, size_t rows, int C, const float* scale, const float* offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) y[row * C + c] = (TO)((xr[c] - mean) * inv * scale[c] + offset[c]);
}
inline void layerNormH(const float* x, half* y, size_t rows, int C, const std::string& w, int block = -1) {
  layerNormTK<half><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(x, y, rows, C, P(w + "/scale", block), P(w + "/offset", block));
}
// out [rows] of a [Bt, n, ...] attention -> y[...] += out transposed back ([n, Bt] -> [Bt, n])
__global__ void swapAddK(float* y, const float* x, int A, int B, int C) {     // y[b][a] += x[a][b]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * B * C) return;
  int c = (int)(t % C); size_t r = t / C; int b = (int)(r % B), a = (int)(r / B);
  y[((size_t)b * A + a) * C + c] += x[t];
}
// o [rows, H*Dp] f16 from the flash kernel, a padded head's first D columns kept: the output projection
// reads the padded layout directly (its weight has zero rows there), so nothing is copied

// ---------------------------------------------------------------- triangle multiplication, --fast
// the three projections the normalised input feeds - projection (2C) | gate (2C) | gating_linear (C) -
// as one [C, 5C] f16 matrix and its bias
__global__ void concat3K(const float* a, const float* b, const float* c, half* out, int C, int wa, int wb, int wc) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; int Wt = wa + wb + wc;
  if (t >= (size_t)C * Wt) return;
  int col = (int)(t % Wt), r = (int)(t / Wt);
  float v = col < wa ? a[(size_t)r * wa + col] : col < wa + wb ? b[(size_t)r * wb + col - wa] : c[(size_t)r * wc + col - wa - wb];
  out[t] = __float2half(v);
}
__global__ void concat3BiasK(const float* a, const float* b, const float* c, float* out, int wa, int wb, int wc) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= wa + wb + wc) return;
  out[t] = t < wa ? a[t] : t < wa + wb ? b[t - wa] : c[t - wa - wb];
}
struct TriW { const half* w5; const float* b5; const half* out; };
inline TriW triWeights(const std::string& T, int blk, int C) {
  static std::map<std::pair<std::string, int>, TriW> cache;
  auto it = cache.find({T, blk});
  if (it != cache.end()) return it->second;
  half* w = dallocT<half>((size_t)C * 5 * C);
  concat3K<<<blocks((size_t)C * 5 * C), 256, 0, STREAM>>>(P(T + "/projection/weights", blk), P(T + "/gate/weights", blk),
    P(T + "/gating_linear/weights", blk), w, C, 2 * C, 2 * C, C);
  float* b = dalloc(5 * C);
  concat3BiasK<<<blocks(5 * C), 256, 0, STREAM>>>(P(T + "/projection/bias", blk), P(T + "/gate/bias", blk),
    P(T + "/gating_linear/bias", blk), b, 2 * C, 2 * C, C);
  TriW tw{w, b, PH(T + "/output_projection/weights", blk)};
  return cache[{T, blk}] = tw;
}
// pg [pairs, 5C] (projection | gate | output gate) -> a, b [C][pairs] f16 = proj * mask * sigmoid(gate),
// a the projection's first C columns; 32 rows x 32 channels a block through shared memory
__global__ void triGateTK(const half* pg, const float* mask, half* a, half* b, size_t pairs, int C) {
  __shared__ float A[32][33], B[32][33];
  size_t r0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t r = r0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (r < pairs) {
      const half* p = pg + r * 5 * C; float m = mask[r];
      va = __half2float(p[c]) * m / (1.f + __expf(-__half2float(p[2 * C + c])));
      vb = __half2float(p[C + c]) * m / (1.f + __expf(-__half2float(p[3 * C + c])));
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t r = r0 + tx; int c = c0 + cy;
    if (r < pairs) { a[(size_t)c * pairs + r] = __float2half(A[tx][cy]); b[(size_t)c * pairs + r] = __float2half(B[tx][cy]); }
  }
}
// prod [C][pairs] f32 -> LN over C -> [pairs, C] f16; 32 rows a block, staged through shared memory
__global__ void centerNormTK(const float* prod, half* out, size_t pairs, int C, const float* scale, const float* offset) {
  extern __shared__ float tile[];               // [C][33]
  size_t r0 = (size_t)blockIdx.x * 32;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int c = warp; c < C; c += nw) {
    size_t r = r0 + lane;
    tile[c * 33 + lane] = r < pairs ? prod[(size_t)c * pairs + r] : 0.f;
  }
  __syncthreads();
  for (int row = warp; row < 32; row += nw) {
    size_t r = r0 + row;
    if (r >= pairs) break;
    float s = 0;
    for (int c = lane; c < C; c += 32) s += tile[c * 33 + row];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, v = 0;
    for (int c = lane; c < C; c += 32) { float d = tile[c * 33 + row] - mean; v += d * d; }
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    float inv = rsqrtf(v / C + 1e-5f);
    for (int c = lane; c < C; c += 32) out[r * C + c] = __float2half((tile[c * 33 + row] - mean) * inv * scale[c] + offset[c]);
  }
}
// pair += out * sigmoid(gate), the gate the [pairs, 5C] block's last C columns
__global__ void gateMulAddHK(float* pair, const float* out, const half* pg, size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  size_t r = t / C; int c = (int)(t % C);
  pair[t] += out[t] / (1.f + __expf(-__half2float(pg[r * 5 * C + 4 * C + c])));
}

// ---------------------------------------------------------------- outer product mean, --fast
__global__ void scaleRowsHK(half* x, const float* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] = __float2half(__half2float(x[t]) * mask[t / C]);
}
__global__ void opmPermuteHK(const half* Pm, half* X, int bi, int L, int O) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)bi * L * O * O) return;
  int e = (int)(t % O); size_t r = t / O; int c = (int)(r % O); r /= O; int j = (int)(r % L), i = (int)(r / L);
  X[t] = Pm[((size_t)i * O + c) * ((size_t)L * O) + (size_t)j * O + e];
}
