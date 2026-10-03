// --fast: f16 activations into tensor-core GEMMs (f32 accumulation) through cuBLASLt, the triangle's
// contraction on zero-padded f16 planes, residual streams kept f32 - native/af2's recipe, the bundle's
// own layout (a triangle's a and b interleaved per channel, a transition's widening [gate | value]).
#pragma once
#include "ops.cuh"
#include "../../af3/src/fusedtransition.cuh"     // native/af3's fused kernels, templated on the width

inline cublasLtHandle_t LT = nullptr;
// weights re-laid for --fast (one set a block) live for the process: carved out of 64 MiB chunks rather
// than a cudaMalloc each - hundreds of them, each a host round trip, were most of a first pass's warm-up
template <class T> T* wpool(size_t n) {
  static char* chunk = nullptr; static size_t used = 0, have = 0;
  size_t bytes = (n * sizeof(T) + 255) / 256 * 256;
  if (used + bytes > have) {
    have = std::max<size_t>(bytes, (size_t)64 << 20);
    CK(cudaMalloc(&chunk, have)); used = 0;
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

// the same at a width known at compile time: the row read once into registers, 16-byte (or 8-byte)
// loads, a warp a row; VEC floats a lane per step, C / (32 VEC) steps
template <int C>
__global__ void layerNormVK(const float* __restrict__ x, half* __restrict__ y, size_t rows, const float* __restrict__ scale,
                            const float* __restrict__ offset) {
  constexpr int VEC = C >= 128 ? 4 : C / 32, STEPS = C / (32 * VEC);
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  float v[STEPS][VEC];
  float s = 0;
#pragma unroll
  for (int k = 0; k < STEPS; ++k) {
    int c = (k * 32 + lane) * VEC;
    if constexpr (VEC == 4) { float4 q = *reinterpret_cast<const float4*>(xr + c); v[k][0] = q.x; v[k][1] = q.y; v[k][2] = q.z; v[k][3] = q.w; }
    else { float2 q = *reinterpret_cast<const float2*>(xr + c); v[k][0] = q.x; v[k][1] = q.y; }
#pragma unroll
    for (int u = 0; u < VEC; ++u) s += v[k][u];
  }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, q2 = 0;
#pragma unroll
  for (int k = 0; k < STEPS; ++k)
#pragma unroll
    for (int u = 0; u < VEC; ++u) { float d = v[k][u] - mean; q2 += d * d; }
  for (int o = 16; o; o >>= 1) q2 += __shfl_xor_sync(~0u, q2, o);
  float inv = rsqrtf(q2 / C + 1e-5f);
#pragma unroll
  for (int k = 0; k < STEPS; ++k) {
    int c = (k * 32 + lane) * VEC;
#pragma unroll
    for (int u = 0; u < VEC; u += 2)
      *reinterpret_cast<half2*>(y + row * C + c + u) =
          __floats2half2_rn((v[k][u] - mean) * inv * scale[c + u] + offset[c + u],
                            (v[k][u + 1] - mean) * inv * scale[c + u + 1] + offset[c + u + 1]);
  }
}
// LayerNorm f32 -> f16, by pointers (scale/offset f32)
inline void layerNormH(const float* x, half* y, size_t rows, int C, const float* sc, const float* of) {
  unsigned grid = (unsigned)((rows + 7) / 8);
  if (C == 128) layerNormVK<128><<<grid, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else if (C == 256) layerNormVK<256><<<grid, 256, 0, STREAM>>>(x, y, rows, sc, of);
  else { fprintf(stderr, "layerNormH: no %d-wide kernel\n", C); exit(1); }
}
// a folding-bundle tensor's f16 copy (the file's half mirror)
inline const half* Fh(const std::string& name) { return Wh("f/" + name); }

inline bool FUSED = true;     // native/af3's fused triangle input (--no-fused: the unfused f16 path)
// ...from 80 tokens only: below, its few row tiles each stream the whole 256 x 1280 weight and leave the
// card idle - the trunk of 4 passes at 68 tokens is 42.7 ms fused against 38.9 unfused. (From 80 tokens
// the 256-channel kernels of fused256.cuh take the block, so this one serves --no-fused256.)
inline int FUSED_MIN_TOKENS = 80;
// ---------------------------------------------------------------- the triangle, f16
// pg [P, 5C] = [projection 2C | gate 2C | gatingLinear C] (f16); a, b channel-major planes [C][Lp][Lp]
// (the pad zero), interleaved: a = proj[2c] * mask * sigmoid(gate[2c]), b likewise at 2c + 1. A 32-pair x
// 32-channel tile goes through shared memory so both the read and the write are coalesced.
__global__ void triSplitHK(const half* pg, const float* mask, half* a, half* b, size_t pairs, int C, int L, int Lp) {
  __shared__ float A[32][33], B[32][33];
  size_t r0 = (size_t)blockIdx.x * 32; int c0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  for (int ry = ty; ry < 32; ry += 8) {
    size_t r = r0 + ry; int c = c0 + tx;
    float va = 0, vb = 0;
    if (r < pairs) {
      const half* p = pg + r * 5 * C; float m = mask[r];
      va = __half2float(p[2 * c]) * m / (1.f + __expf(-__half2float(p[2 * C + 2 * c])));
      vb = __half2float(p[2 * c + 1]) * m / (1.f + __expf(-__half2float(p[2 * C + 2 * c + 1])));
    }
    A[ry][tx] = va; B[ry][tx] = vb;
  }
  __syncthreads();
  for (int cy = ty; cy < 32; cy += 8) {
    size_t r = r0 + tx; int c = c0 + cy;
    if (r < pairs) {
      size_t at = (size_t)c * Lp * Lp + (r / L) * Lp + r % L;
      a[at] = __float2half(A[tx][cy]); b[at] = __float2half(B[tx][cy]);
    }
  }
}
// prod [C][Lp][Lp] f32 -> LN over C -> [P, C] f16; 32 pairs a block through shared memory
__global__ void centerNormHK(const float* prod, half* out, size_t pairs, int C, const float* scale, const float* offset,
                             int L, int Lp) {
  extern __shared__ float tile[];               // [C][33]
  size_t r0 = (size_t)blockIdx.x * 32;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int c = warp; c < C; c += nw) {
    size_t r = r0 + lane;
    tile[c * 33 + lane] = r < pairs ? prod[(size_t)c * Lp * Lp + (r / L) * Lp + r % L] : 0.f;
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
__global__ void gateMulAddHK(float* pair, const float* out, const half* pg, size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  size_t r = t / C; int c = (int)(t % C);
  pair[t] += out[t] / (1.f + __expf(-__half2float(pg[r * 5 * C + 4 * C + c])));
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
  ltGemm(xn, Fh(Tn + "projection"), pg, true, P, C, 2 * C, nullptr, false, 0.f, 0, 5 * C);
  ltGemm(xn, Fh(Tn + "gate"), pg + 2 * C, true, P, C, 2 * C, nullptr, false, 0.f, 0, 5 * C);
  ltGemm(xn, Fh(Tn + "gatingLinear"), pg + 4 * C, true, P, C, C, nullptr, false, 0.f, 0, 5 * C);
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
  // the pad, written once for a buffer and a padded size (nothing else writes it - but a buffer reused at
  // another size has data where this layout's pad is)
  static half* zeroed = nullptr; static int zeroedLp = 0;
  if (Lp != L && (a != zeroed || Lp != zeroedLp)) {
    CK(cudaMemsetAsync(a, 0, plane * C * 2, STREAM)); CK(cudaMemsetAsync(b, 0, plane * C * 2, STREAM));
    zeroed = a; zeroedLp = Lp;
  }
  triSplitHK<<<dim3((unsigned)((P + 31) / 32), C / 32), dim3(32, 8), 0, STREAM>>>(pg, mask, a, b, P, C, L, Lp);
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
  gateMulAddHK<<<blocks(P * C), 256, 0, STREAM>>>(pair, out, pg, P, C);
}
__global__ void swigluHK(const half* h, half* g, size_t rows, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * I) return;
  size_t r = t / I; int c = (int)(t % I);
  float x = __half2float(h[r * 2 * I + c]);
  g[t] = __float2half(x / (1.f + __expf(-x)) * __half2float(h[r * 2 * I + I + c]));
}
inline void transitionFast(float* pair, size_t P, int C, const std::string& Tn) {
  int I = (int)dimOf("f/" + Tn + "transition2", 0);
  half* xn = scratch<half>("ftr.xn", P * C);
  layerNormH(pair, xn, P, C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"));
  size_t chunk = std::max<size_t>(1, ((size_t)128 << 20) / (3 * (size_t)I));
  half* h = scratch<half>("ftr.h", std::min(P, chunk) * 2 * I); half* g = scratch<half>("ftr.g", std::min(P, chunk) * I);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    ltGemm(xn + r0 * C, Fh(Tn + "transition1"), h, true, r, C, 2 * I, nullptr, false, 0.f);
    swigluHK<<<blocks(r * I), 256, 0, STREAM>>>(h, g, r, I);
    ltGemm(g, Fh(Tn + "transition2"), pair + r0 * C, false, r, I, C, nullptr, false, 1.f);
  }
}

// ---------------------------------------------------------------- native/af3's fused kernels at this width
// (fusedTransitionK<256> was tried and is SLOWER here, 504 against 466 ms of trunk at 261 tokens: at 256
// channels it takes 255 registers a thread and 150 KB of shared memory, four warps an SM)
// the triangle's input half in one kernel (LN, projection and gate, the interleaved split into padded
// planes, and the gating linear's raw output t2), native/af3's triInK at this width
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
