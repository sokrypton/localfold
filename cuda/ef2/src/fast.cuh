// --fast: f16 activations into tensor-core GEMMs (f32 accumulation) through cuBLASLt, the triangle's
// contraction on zero-padded f16 planes, residual streams kept f32 - cuda/af2's recipe, the bundle's
// own layout (a triangle's a and b interleaved per channel, a transition's widening [gate | value]).
#pragma once
#include "ops.cuh"
#include "../../af3/src/fusedtransition.cuh"     // cuda/af3's fused kernels, templated on the width

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
inline std::map<const float*, half*> BIAS_HALF;     // f16 copies of biases (forgotten with the derived weights)
inline const bool BIAS_HALF_HOOK = (FORGET_HOOKS.push_back([] { BIAS_HALF.clear(); }), true);
inline const void* biasFor(const float* bias, int n, bool asHalf) {
  if (!bias || !asHalf) return bias;
  auto& copies = BIAS_HALF;
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
  CB(cublasLtMatmul(LT, p.op, &one, Wt, p.a, X, p.b, &beta, Y, p.c, Y, p.c, &p.algo, nullptr, 0, STREAM));
}

// (the same at a width known at compile time: layerNormVK, elementwise.cuh)
// any width, a warp a row (MSA Pairformer's 464-wide MSA track, which no vector width divides)
__global__ void layerNormAnyHK(const float* x, half* y, size_t rows, int C, const float* sc, const float* of) {
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
  for (int c = lane; c < C; c += 32) y[row * C + c] = __float2half((xr[c] - mean) * inv * sc[c] + of[c]);
}
// any width that is a multiple of 4 up to 512, a warp a row held in registers (float4s, read once): the generic form
// above reads the row three times, 335 ms at MSA Pairformer's 464 channels and 2,048 x 1,023 rows
template <int C>
__global__ void layerNormRegHK(const float* __restrict__ x, half* __restrict__ y, size_t rows, const float* __restrict__ sc,
                               const float* __restrict__ of) {
  constexpr int V = C / 4, STEPS = (V + 31) / 32;
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float4* xr = reinterpret_cast<const float4*>(x + row * C);
  float4 v[STEPS]; float s = 0;
#pragma unroll
  for (int k = 0; k < STEPS; ++k) {
    int q = k * 32 + lane;
    v[k] = q < V ? xr[q] : make_float4(0, 0, 0, 0);
    s += v[k].x + v[k].y + v[k].z + v[k].w;
  }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  const float mean = s / C; float q2 = 0;
#pragma unroll
  for (int k = 0; k < STEPS; ++k) if (k * 32 + lane < V) {
    float a = v[k].x - mean, b = v[k].y - mean, c = v[k].z - mean, d = v[k].w - mean; q2 += a * a + b * b + c * c + d * d;
  }
  for (int o = 16; o; o >>= 1) q2 += __shfl_xor_sync(~0u, q2, o);
  const float inv = rsqrtf(q2 / C + 1e-5f);
#pragma unroll
  for (int k = 0; k < STEPS; ++k) {
    int q = k * 32 + lane;
    if (q >= V) continue;
    float4 g = reinterpret_cast<const float4*>(sc)[q], o = reinterpret_cast<const float4*>(of)[q];
    __half2 lo = __floats2half2_rn((v[k].x - mean) * inv * g.x + o.x, (v[k].y - mean) * inv * g.y + o.y);
    __half2 hi = __floats2half2_rn((v[k].z - mean) * inv * g.z + o.z, (v[k].w - mean) * inv * g.w + o.w);
    uint2 out; out.x = *reinterpret_cast<uint32_t*>(&lo); out.y = *reinterpret_cast<uint32_t*>(&hi);
    reinterpret_cast<uint2*>(y + row * C)[q] = out;
  }
}
// LayerNorm f32 -> f16, by pointers (scale/offset f32)
inline void layerNormH(const float* x, half* y, size_t rows, int C, const float* sc, const float* of) {
  unsigned grid = (unsigned)((rows + 7) / 8);
  if (C == 128) layerNormVK<128><<<grid, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else if (C == 256) layerNormVK<256><<<grid, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else if (C == 464) layerNormRegHK<464><<<grid, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else layerNormAnyHK<<<grid, 256, 0, STREAM>>>(x, y, rows, C, sc, of);
}
// a folding-bundle tensor's f16 copy (the file's half mirror)
inline const half* Fh(const std::string& name) { return Wh("f/" + name); }

inline bool FUSED = true;     // cuda/af3's fused triangle input (--no-fused: the unfused f16 path)
// ...from 80 tokens only: below, its few row tiles each stream the whole 256 x 1280 weight and leave the
// card idle - the trunk of 4 passes at 68 tokens is 42.7 ms fused against 38.9 unfused. (From 80 tokens
// the 256-channel kernels of fused256.cuh take the block, so this one serves --no-fused256.)
inline int FUSED_MIN_TOKENS = 80;
// ---------------------------------------------------------------- the triangle, f16
// pg [P, 5C] = [projection 2C | gate 2C | gatingLinear C] (f16); a, b channel-major planes [C][Lp][Lp]
// (the pad zero), interleaved: a = proj[2c] * mask * sigmoid(gate[2c]), b likewise at 2c + 1. A 32-pair x
// 32-channel tile goes through shared memory so both the read and the write are coalesced.
__global__ void triSplitHK(const half* __restrict__ pg, const float* __restrict__ mask, half* __restrict__ a,
                           half* __restrict__ b, size_t plane, int C, int L, int Lp) {
  __shared__ __half A[32][33], B[32][33];
  size_t p0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  int cLoad = c0 + tx;
  #pragma unroll
  for (int ry = ty; ry < 32; ry += 8) {
    size_t pIdx = p0 + ry;
    __half ha = __float2half(0.f), hb = __float2half(0.f);
    if (pIdx < plane) {
      unsigned up = (unsigned)pIdx, uLp = (unsigned)Lp, i = up / uLp, j = up - i * uLp;
      if ((int)i < L && (int)j < L) {
        size_t r = (size_t)i * L + j;
        const half* p = pg + r * 5 * C; float m = __ldg(mask + r);
        half2 pr = __ldg(reinterpret_cast<const half2*>(p) + cLoad);
        half2 gt = __ldg(reinterpret_cast<const half2*>(p + 2 * C) + cLoad);
        float2 pf = __half22float2(pr), gf = __half22float2(gt);
        ha = __float2half(pf.x * m / (1.f + __expf(-gf.x)));
        hb = __float2half(pf.y * m / (1.f + __expf(-gf.y)));
      }
    }
    A[ry][tx] = ha; B[ry][tx] = hb;
  }
  __syncthreads();
  size_t pStore = p0 + tx;
  if (pStore < plane) {
    #pragma unroll
    for (int cy = ty; cy < 32; cy += 8) {
      size_t at = (size_t)(c0 + cy) * plane + pStore;
      a[at] = A[tx][cy]; b[at] = B[tx][cy];
    }
  }
}
inline void triangleFusedIn(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing);
inline void triangleFast(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing) {
  size_t P = (size_t)L * L;
  // (its 4-warp form takes 64 rows and two weight stages - over a T4's 64 KB a block: the plain path there)
  if (FUSED && C == 256 && L >= FUSED_MIN_TOKENS && fitsSmem((size_t)64 * (256 + 8) * 2 + 2 * tiStage(256))) {
    triangleFusedIn(pair, mask, L, C, Tn, outgoing); return;
  }
  half* xn = scratch<half>("ftri.xn", P * C);
  layerNormH(pair, xn, P, C, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"));
  half* pg = scratch<half>("ftri.pg", P * 5 * C);
  std::string pgg = concatColumns("f/" + Tn + "projGateGating~", C,
                                  {{"f/" + Tn + "projection", 2 * C, false},
                                   {"f/" + Tn + "gate", 2 * C, false},
                                   {"f/" + Tn + "gatingLinear", C, false}});
  ltGemm(xn, Wh(pgg), pg, true, P, C, 5 * C, nullptr, false, 0.f);
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
  triSplitHK<<<dim3((unsigned)((plane + 31) / 32), C / 32), dim3(32, 8), 0, STREAM>>>(pg, mask, a, b, plane, C, L, Lp);
  float* prod = scratch<float>("ftri.prod", plane * C);
  const float one = 1.f, zero = 0.f;
  if (outgoing)
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                  plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                  plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  half* cn = scratch<half>("ftri.cn", P * C);
  centerNormHK<<<(unsigned)((P + 31) / 32), 256, (size_t)C * 33 * 4, STREAM>>>(prod, cn, P, C, F(Tn + "centerNormScale"),
                                                                              F(Tn + "centerNormOffset"), L, Lp);
  float* out = scratch<float>("ftri.out", P * C);
  ltGemm(cn, Fh(Tn + "outputProjection"), out, false, P, C, C, nullptr, false, 0.f);
  gateMulAddHK<<<blocks((C & 3) == 0 ? P * (size_t)(C >> 2) : P * C), 256, 0, STREAM>>>(pair, out, pg + 4 * C, P, C, 5 * C);
}
__global__ void swigluHK(const half* h, half* g, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int c = (int)(t % I);
  float x = __half2float(h[r * 2 * I + c]);
  g[t] = __float2half(x / (1.f + __expf(-x)) * __half2float(h[r * 2 * I + I + c]));
}
// the same, eight elements a thread in 16-byte loads and stores (I a multiple of 8), a row's index from a 32-bit
// division: the one above runs at half the memory's bandwidth (MSA Pairformer's 464-wide MSA, 1,040 columns:
// 211 ms of a 1.97 s pass). Each element's arithmetic is the one above's, so the output is the same bytes
__global__ void swigluH8K(const half* __restrict__ h, half* __restrict__ g, unsigned rows, unsigned I8) {
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I8) return;
  unsigned r = t / I8, c = t - r * I8;
  uint4 xv = reinterpret_cast<const uint4*>(h)[(size_t)r * 2 * I8 + c];
  uint4 yv = reinterpret_cast<const uint4*>(h)[(size_t)r * 2 * I8 + I8 + c];
  const half *xs = reinterpret_cast<const half*>(&xv), *ys = reinterpret_cast<const half*>(&yv);
  uint4 ov; half* os = reinterpret_cast<half*>(&ov);
#pragma unroll
  for (int e = 0; e < 8; ++e) { float x = __half2float(xs[e]); os[e] = __float2half(x / (1.f + __expf(-x)) * __half2float(ys[e])); }
  reinterpret_cast<uint4*>(g)[t] = ov;
}
inline void swigluH(const half* h, half* g, size_t rows, int I) {
  if (I % 8 == 0 && rows * (I / 8) < (1ull << 31))
    swigluH8K<<<blocks(rows * (I / 8)), 256, 0, STREAM>>>(h, g, (unsigned)rows, (unsigned)(I / 8));
  else swigluHK<<<blocks(rows * I), 256, 0, STREAM>>>(h, g, rows, I);
}
inline void transitionFast(float* pair, size_t P, int C, const std::string& Tn) {
  int I = (int)dimOf("f/" + Tn + "transition2", 0);
  size_t chunk = std::max<size_t>(1, ((size_t)128 << 20) / (3 * (size_t)I));
  // the LayerNorm whole, or - on a card short of room - a chunk of rows at a time, as the GEMMs already go: it
  // is per row, so the same values (the whole f16 normalised pair was 2.95 GB at 2400 tokens, where a simulated
  // T4 ran out)
  bool chunked = shortPair(P, C);
  half* xn = scratch<half>("ftr.xn", (chunked ? std::min(P, chunk) : P) * C);
  if (!chunked) layerNormH(pair, xn, P, C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"));
  half* h = scratch<half>("ftr.h", std::min(P, chunk) * 2 * I); half* g = scratch<half>("ftr.g", std::min(P, chunk) * I);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    if (chunked) layerNormH(pair + r0 * C, xn, r, C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"));
    ltGemm(xn + (chunked ? 0 : r0 * C), Fh(Tn + "transition1"), h, true, r, C, 2 * I, nullptr, false, 0.f);
    swigluH(h, g, r, I);
    ltGemm(g, Fh(Tn + "transition2"), pair + r0 * C, false, r, I, C, nullptr, false, 1.f);
  }
}

// ---------------------------------------------------------------- cuda/af3's fused kernels at this width
// (fusedTransitionK<256> was tried and is SLOWER here, 504 against 466 ms of trunk at 261 tokens: at 256
// channels it takes 255 registers a thread and 150 KB of shared memory, four warps an SM)
// the triangle's input half in one kernel (LN, projection and gate, the interleaved split into padded
// planes, and the gating linear's raw output t2), cuda/af3's triInK at this width
template <int WARPS>
void triInFused(const float* pair, const float* mask, const std::string& Tn, half* a, half* b, half* t2, int n, int np,
                size_t cs) {
  constexpr int CW = 256, R = 16 * WARPS;
  size_t pp = (size_t)np * np;
  size_t smem = (size_t)R * (CW + 8) * 2 + 2 * tiStage(CW);
  static bool attr = false;
  if (!attr) { smemAttr((triInK<CW, WARPS, half>), (int)smem); attr = true; }
  std::string pg = concatColumns("f/" + Tn + "projectionGate~", CW, {{"f/" + Tn + "projection", 2 * CW, false},
                                                                    {"f/" + Tn + "gate", 2 * CW, false}});
  triInK<CW, WARPS, half><<<(unsigned)((pp + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    pair, mask, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"), Wh(pg), Fh(Tn + "gatingLinear"),
    a, b, t2, n, np, cs);
}
__global__ void gateMulAddPaddedK(float* pair, const float* out, const half* t2, int L, int Lp, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C) return;
  size_t r = t / C; int c = (int)(t % C);
  size_t padded = (r / L) * Lp + r % L;
  pair[t] += out[t] / (1.f + __expf(-__half2float(t2[padded * C + c])));
}
// the triangle with its input half fused: triInK -> the f16 contraction -> centre norm -> output GEMM -> gate
inline void triangleFusedIn(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing) {
  size_t P = (size_t)L * L;
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
  half* t2 = scratch<half>("ftri.t2", plane * C);
  if (plane / 64 >= 54 * 8 && fitsSmem((size_t)128 * (256 + 8) * 2 + 2 * tiStage(256)))
    triInFused<8>(pair, mask, Tn, a, b, t2, L, Lp, plane);     // writes the padding itself
  else triInFused<4>(pair, mask, Tn, a, b, t2, L, Lp, plane);
  float* prod = scratch<float>("ftri.prod", plane * C);
  const float one = 1.f, zero = 0.f;
  if (outgoing)
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                  plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                  plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  half* cn = scratch<half>("ftri.cn", P * C);
  centerNormHK<<<(unsigned)((P + 31) / 32), 256, (size_t)C * 33 * 4, STREAM>>>(prod, cn, P, C, F(Tn + "centerNormScale"),
                                                                              F(Tn + "centerNormOffset"), L, Lp);
  float* out = scratch<float>("ftri.out", P * C);
  ltGemm(cn, Fh(Tn + "outputProjection"), out, false, P, C, C, nullptr, false, 0.f);
  gateMulAddPaddedK<<<blocks(P * C), 256, 0, STREAM>>>(pair, out, t2, L, Lp, C);
}
