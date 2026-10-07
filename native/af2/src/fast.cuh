// The --fast path's pieces: f16 activations into tensor-core GEMMs (f32 accumulation) with the bias,
// the ReLU and the residual add in cuBLASLt's epilogue, and weights re-laid once for them.
#pragma once
#include "ops.cuh"
#include "../../af3/src/fusedtransition.cuh"     // (and fusedtriangle.cuh)
#include "../../af3/src/fused256.cuh"          // (its streaming triangle kernels: a T4's, below)

// ---------------------------------------------------------------- cuBLASLt, row-major
// Y[rows, out] (f32 or f16) = X[rows, in] (f16) W[in, out] (f16) (+ bias[out] f32) (ReLU) (+ beta Y)
inline cublasLtHandle_t LT = nullptr;
// weights re-laid for --fast (one set a block) live for the process: carved out of 64 MiB chunks rather
// than a cudaMalloc each - hundreds of them, each a host round trip, were most of a first pass's warm-up
template <class T> T* wpool(size_t n) {
  static char* chunk = nullptr; static size_t used = 0, have = 0;
  size_t bytes = (n * sizeof(T) + 255) / 256 * 256;
  if (used + bytes > have) {
    have = std::max<size_t>(bytes, (size_t)64 << 20);
    CK(devMalloc(&chunk, have)); used = 0;
  }
  T* p = (T*)(chunk + used); used += bytes; return p;
}
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; bool ok; };
inline const void* biasFor(const float* bias, int n, bool asHalf) {
  if (!bias || !asHalf) return bias;
  static std::map<const float*, half*> copies;
  auto it = copies.find(bias);
  if (it != copies.end()) return it->second;
  half* h = wpool<half>(n);
  toHalfK<<<blocks(n), 256, 0, STREAM>>>(bias, h, n);
  return copies[bias] = h;
}
inline void ltGemm(const half* X, const half* Wt, void* Y, bool yHalf, size_t rows, int in, int out, const float* biasF,
                   bool relu, float beta, int ldx = 0, int ldy = 0) {
  if (ldx == 0) ldx = in;
  if (ldy == 0) ldy = out;
  if (!LT) CB(cublasLtCreate(&LT));
  const void* bias = biasFor(biasF, out, yHalf);
  if (getenv("AF2_LT_SHAPES") && beta != 0.f && biasF) {
    static std::map<std::tuple<size_t, int, int>, int> seen;
    if (seen[std::make_tuple(rows, in, out)]++ == 0) fprintf(stderr, "beta+bias GEMM %zu x %d x %d\n", rows, in, out);
  }
  static std::map<std::tuple<size_t, int, int, bool, int, bool, int, int>, LtPlan> plans;
  int epi = bias ? (relu ? 2 : 1) : (relu ? 3 : 0);
  auto key = std::make_tuple(rows, in, out, yHalf, epi, beta != 0.f, ldx, ldy);
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
    CB(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, in, (uint64_t)rows, ldx));
    CB(cublasLtMatrixLayoutCreate(&p.c, yHalf ? CUDA_R_16F : CUDA_R_32F, out, (uint64_t)rows, ldy));
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
  // AF2_GEMM_TIMES: every call timed by events, summed by shape and printed at exit (an analysis aid:
  // the events serialise nothing but cost a little)
  struct Timed { std::tuple<size_t, int, int, bool, int> key; cudaEvent_t a, b; };
  static std::vector<Timed>* timed = nullptr;
  static bool timing = getenv("AF2_GEMM_TIMES") != nullptr;
  cudaEvent_t ea = nullptr, eb = nullptr;
  if (timing) {
    if (!timed) {
      timed = new std::vector<Timed>;
      atexit([] {
        std::map<std::tuple<size_t, int, int, bool, int>, std::pair<double, int>> sum;
        for (auto& t : *timed) { float ms; cudaEventElapsedTime(&ms, t.a, t.b); auto& s = sum[t.key]; s.first += ms; s.second++; }
        std::vector<std::pair<double, std::tuple<size_t, int, int, bool, int>>> order;
        for (auto& [k, v] : sum) order.push_back({v.first, k});
        std::sort(order.rbegin(), order.rend());
        for (auto& [ms, k] : order) {
          auto [r, i, o, h, e] = k; double tf = 2.0 * r * i * o * sum[k].second / (ms * 1e-3) / 1e12;
          fprintf(stderr, "  %8.2f ms %5d x  %7zu x %5d x %5d  f16out %d epi %d  %6.1f TFLOP/s\n", ms, sum[k].second, r, i, o, h, e, tf);
        }
      });
    }
    cudaEventCreate(&ea); cudaEventCreate(&eb); cudaEventRecord(ea, STREAM);
  }
  CB(cublasLtMatmul(LT, p.op, &one, Wt, p.a, X, p.b, &beta, Y, p.c, Y, p.c, &p.algo, nullptr, 0, STREAM));
  if (timing) { cudaEventRecord(eb, STREAM); timed->push_back({std::make_tuple(rows, in, out, yHalf, epi), ea, eb}); }
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
  half* qk = wpool<half>((size_t)C * 4 * H * Dp);
  packQkvgWeightK<<<blocks((size_t)C * 4 * H * Dp), 256, 0, STREAM>>>(P(A + "/query_w", blk), P(A + "/key_w", blk),
    P(A + "/value_w", blk), P(A + "/gating_w", blk), qk, C, H, D, Dp);
  float* gb = wpool<float>(4 * H * Dp);
  packGateBiasK<<<blocks(4 * H * Dp), 256, 0, STREAM>>>(P(A + "/gating_b", blk), gb, H, D, Dp);
  half* ow = wpool<half>((size_t)H * Dp * C);
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
// (the same at a width known at compile time: layerNormVK, elementwise.cuh)
inline void layerNormH(const float* x, half* y, size_t rows, int C, const std::string& w, int block = -1) {
  const float *sc = P(w + "/scale", block), *of = P(w + "/offset", block);
  unsigned grid = (unsigned)((rows + 7) / 8);
  if (C == 64) layerNormVK<64><<<grid, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else if (C == 128) layerNormVK<128><<<grid, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else if (C == 256) layerNormVK<256><<<grid, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else layerNormTK<half><<<grid, 256, 0, STREAM>>>(x, y, rows, C, sc, of);
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
  half* w = wpool<half>((size_t)C * 5 * C);
  concat3K<<<blocks((size_t)C * 5 * C), 256, 0, STREAM>>>(P(T + "/projection/weights", blk), P(T + "/gate/weights", blk),
    P(T + "/gating_linear/weights", blk), w, C, 2 * C, 2 * C, C);
  float* b = wpool<float>(5 * C);
  concat3BiasK<<<blocks(5 * C), 256, 0, STREAM>>>(P(T + "/projection/bias", blk), P(T + "/gate/bias", blk),
    P(T + "/gating_linear/bias", blk), b, 2 * C, 2 * C, C);
  TriW tw{w, b, PH(T + "/output_projection/weights", blk)};
  return cache[{T, blk}] = tw;
}
// pg [pairs, 5C] (projection | gate | output gate) -> a, b [C][pairs] f16 = proj * mask * sigmoid(gate),
// a the projection's first C columns; 32 rows x 32 channels a block through shared memory
// a and b channel-major, each plane [Lp][Lp] (Lp a multiple of 8, the pad zero): the batched
// contraction then runs on aligned tensor-core tiles - at L odd it fell to cutlass's align1 kernels
// (sm75's), 82 of a 5CAJ fold's 1700 ms
// pg holds rows p0 .. p0 + pairs of the projection (a chunk of them); mask and the planes are whole
__global__ void triGateTK(const half* pg, const float* mask, half* a, half* b, size_t pairs, int C, int L, int Lp,
                          size_t p0 = 0) {
  __shared__ float A[32][33], B[32][33];
  size_t r0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t r = r0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (r < pairs) {
      const half* p = pg + r * 5 * C; float m = mask[p0 + r];
      va = __half2float(p[c]) * m / (1.f + __expf(-__half2float(p[2 * C + c])));
      vb = __half2float(p[C + c]) * m / (1.f + __expf(-__half2float(p[3 * C + c])));
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t r = r0 + tx; int c = c0 + cy;
    if (r < pairs) {
      size_t at = (size_t)c * Lp * Lp + ((p0 + r) / L) * Lp + (p0 + r) % L;
      a[at] = __float2half(A[tx][cy]); b[at] = __float2half(B[tx][cy]);
    }
  }
}
// ...and at 128 channels native/af3's FUSED triangle (fusedtriangle.cuh: triInK, triOutPK - the LayerNorm'd
// rows, the 4C projection and the centred rows never written), the same computation as AlphaFold 3's with
// biases: AF3's kernels interleave a and b (column 2ch is a's channel ch, 2ch+1 b's) where AF2 stores the
// halves one after the other, so the weights are re-laid once, and the biases ride along (BIAS)
__global__ void interleaveTriK(const float* proj, const float* gate, const float* pb, const float* gb, const float* lb,
                               half* wpg, float* bias, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)C * 4 * C) {
    int k = (int)(t / (4 * C)), col = (int)(t % (4 * C)), part = col / (2 * C), cc = col % (2 * C), ch = cc / 2;
    const float* src = part ? gate : proj;
    wpg[t] = __float2half(src[(size_t)k * 2 * C + (cc & 1) * C + ch]);
  }
  if (t < (size_t)5 * C) {
    int col = (int)t;
    if (col < 4 * C) { int part = col / (2 * C), cc = col % (2 * C), ch = cc / 2; bias[t] = (part ? gb : pb)[(cc & 1) * C + ch]; }
    else bias[t] = lb[col - 4 * C];
  }
}
struct TriFused { const half* wpg; const float* bias; };
inline TriFused triFusedWeights(const std::string& T, int blk, int C) {
  static std::map<std::pair<std::string, int>, TriFused> cache;
  auto it = cache.find({T, blk});
  if (it != cache.end()) return it->second;
  half* w = wpool<half>((size_t)C * 4 * C); float* b = wpool<float>((size_t)5 * C);
  interleaveTriK<<<blocks((size_t)C * 4 * C), 256, 0, STREAM>>>(P(T + "/projection/weights", blk), P(T + "/gate/weights", blk),
    P(T + "/projection/bias", blk), P(T + "/gate/bias", blk), P(T + "/gating_linear/bias", blk), w, b, C);
  return cache[{T, blk}] = TriFused{w, b};
}

// ---------------------------------------------------------------- outer product mean, --fast
__global__ void scaleRowsHK(half* x, const float* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] = __float2half(__half2float(x[t]) * mask[t / C]);
}

// ---------------------------------------------------------------- a bias carried by the product
// W [K, N] (f16) with its bias as row K and zero rows to K+8: an input whose row carries a 1 at
// column K (and zeros to K+8) then adds the bias inside the GEMM - no epilogue, no separate pass
__global__ void augmentWeightK(const half* w, const float* b, half* out, int K, int N) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)(K + 8) * N) return;
  int r = (int)(t / N), c = (int)(t % N);
  out[t] = r < K ? w[(size_t)r * N + c] : r == K ? __float2half(b[c]) : __float2half(0.f);
}
inline const half* augmentedWeight(const std::string& name, int blk, const float* bias, int K, int N) {
  static std::map<std::pair<std::string, int>, half*> cache;
  auto it = cache.find({name, blk});
  if (it != cache.end()) return it->second;
  half* w = wpool<half>((size_t)(K + 8) * N);
  augmentWeightK<<<blocks((size_t)(K + 8) * N), 256, 0, STREAM>>>(PH(name, blk), bias, w, K, N);
  return cache[{name, blk}] = w;
}
__global__ void onesColumnK(half* x, size_t rows, int K) {     // x [rows, K+8]: column K = 1, K+1.. = 0
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * 8) return;
  x[(t / 8) * (K + 8) + K + t % 8] = __float2half(t % 8 == 0 ? 1.f : 0.f);
}
// a [rows, K+8] f16 buffer whose last 8 columns are (1, 0, ...), kept per (name, size)
inline half* augmentedInput(const std::string& name, size_t rows, int K) {
  static std::map<std::string, std::pair<half*, size_t>> bufs;
  auto& [p, have] = bufs[name];
  if (have < rows) {
    if (p) CK(cudaFree(p));
    p = dallocT<half>(rows * (K + 8));
    onesColumnK<<<blocks(rows * 8), 256, 0, STREAM>>>(p, rows, K);
    have = rows;
  }
  return p;
}
