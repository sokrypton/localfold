// z_init, the recycle and the 24-block pair trunk, and the distogram (webgpu/esmfold2/fold.js and
// cpu/esmfold2/pair-features.js are the reading):
//
//   z_init = z_init_1(s)[i] + z_init_2(s)[j] + rel_pos + token_bonds + lm_z
//   z = 0;  num_loops + 1 times:  z = z_init + Linear(LN(z));  z = trunk(z)
//   trunk block: z += triangle out(z); z += triangle in(z); z += transition(z)      (no biases anywhere)
//   distogram = (z + z^T) @ W + b
// The trunk is AF3's pairformer block with its grid attentions and single track removed, and the
// bundle carries it in AF3's shapes: a triangle's projection and gate interleave a and b per channel,
// the transition's widening is [gate | value].
#pragma once
#include "fused256.cuh"

// relative position bins (residue 66, token 66, same entity 1, chain 6 = 139), one-hot rows summed;
// computed where it is read (z_init, the diffusion's conditioning, the confidence head), never stored
struct RelIdx { const int *ri, *asym, *sym, *ent, *ti; const float* Wt; };
inline RelIdx relIdx() {
  return {Idev("residue_index"), Idev("asym_id"), Idev("sym_id"), Idev("entity_id"), Idev("token_index"), F("featuriser/relPos")};
}
__device__ __forceinline__ float relPosAt(const RelIdx& r, int i, int j, int c, int C) {
  const int rb = 32, cb = 2;
  auto clip = [](int v, int hi) { return v < 0 ? 0 : v > hi ? hi : v; };
  bool sameChain = r.asym[i] == r.asym[j], sameRes = r.ri[i] == r.ri[j];
  int b0 = sameChain ? clip(r.ri[i] - r.ri[j] + rb, 2 * rb) : 2 * rb + 1;
  int b1 = sameChain && sameRes ? clip(r.ti[i] - r.ti[j] + rb, 2 * rb) : 2 * rb + 1;
  int b3 = sameChain ? 2 * cb + 1 : clip(r.sym[i] - r.sym[j] + cb, 2 * cb);
  const int w = 2 * rb + 2;
  float v = r.Wt[(size_t)b0 * C + c] + r.Wt[(size_t)(w + b1) * C + c] + r.Wt[(size_t)(2 * w + 1 + b3) * C + c];
  if (r.ent[i] == r.ent[j]) v += r.Wt[(size_t)(2 * w) * C + c];
  return v;
}
__global__ void relPosK(RelIdx r, float* out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // one thread per (pair, channel)
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C;
  out[t] = relPosAt(r, (int)(ij / T), (int)(ij % T), c, C);
}
// z_init = rows[i] + cols[j] + relpos + bonds * w_bond + lm_z
__global__ void zInitK(const float* rows, const float* cols, RelIdx rel, const float* bonds, const float* wBond,
                       const float* lmZ, float* z, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  z[t] = rows[(size_t)i * C + c] + cols[(size_t)j * C + c] + relPosAt(rel, i, j, c, C) + bonds[ij] * wBond[c] + (lmZ ? lmZ[t] : 0.f);
}

// a STREAMED z_init (a card short of room): neither it nor the language model's pair is kept - each loop
// makes its rows a block at a time from the per-token states and adds them in (zInitK's sum, in its order)
struct ZInitStream { const float* rows; const float* cols; const float* s; int P; };
inline ZInitStream ZINIT_STREAM{};
__global__ void zInitAddRowsK(float* z, const float* rows, const float* cols, RelIdx rel, const float* bonds, const float* wBond,
                              const float* lm, int T, int C, size_t r0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = r0 + t / C; int i = (int)(ij / T), j = (int)(ij % T);
  z[r0 * C + t] += rows[(size_t)i * C + c] + cols[(size_t)j * C + c] + relPosAt(rel, i, j, c, C) + bonds[ij] * wBond[c] + lm[t];
}
inline void zInitAddStreamed(float* z, int T, int C) {
  const ZInitStream& zs = ZINIT_STREAM;
  int bi = lmPairBlock(T, zs.P);
  float* lm = scratch<float>("zi.lmRows", (size_t)bi * T * zs.P);
  for (int i0 = 0; i0 < T; i0 += bi) {
    int b = std::min(bi, T - i0); size_t cells = (size_t)b * T;
    lmPairRows(zs.s, T, zs.P, i0, b, lm);
    zInitAddRowsK<<<blocks(cells * C), 256, 0, STREAM>>>(z, zs.rows, zs.cols, relIdx(), W("token_bonds"), F("featuriser/tokenBonds"),
                                                         lm, T, C, (size_t)i0 * T, cells);
  }
}
inline void zInit(int T, int C, const float* sInputs, int Si, const float* lmZ, float* z, bool check) {
  size_t P = (size_t)T * T;
  float* rows = scratch<float>("zi.rows", (size_t)T * C); float* cols = scratch<float>("zi.cols", (size_t)T * C);
  gemm(sInputs, F("featuriser/zInit1"), rows, T, Si, C);
  gemm(sInputs, F("featuriser/zInit2"), cols, T, Si, C);
  if (check) {
    checkOracle("z_init_1", rows, (size_t)T * C, "o/z_init_1");
    checkOracle("z_init_2", cols, (size_t)T * C, "o/z_init_2");
    float* rel = scratch<float>("zi.rel", P * C);
    relPosK<<<blocks(P * C), 256, 0, STREAM>>>(relIdx(), rel, T, C);
    checkOracle("rel_pos", rel, P * C, "o/rel_pos");
  }
  zInitK<<<blocks(P * C), 256, 0, STREAM>>>(rows, cols, relIdx(), W("token_bonds"), F("featuriser/tokenBonds"), lmZ, z, T, C);
}

// ---------------------------------------------------------------- the trunk, float32
__global__ void triSplitInterleavedK(const float* proj, const float* gate, const float* mask, float* a, float* b,
                                     size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * 2 * C) return;
  int c2 = (int)(t % (2 * C)); size_t ij = t / (2 * C);
  float v = proj[t] * mask[ij] / (1.f + expf(-gate[t]));
  (c2 & 1 ? b : a)[(size_t)(c2 >> 1) * pairs + ij] = v;
}
__global__ void fillK(float* out, float v, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = v;
}

inline void triangle(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing) {
  size_t P = (size_t)L * L;
  float* xn = scratch<float>("tri.xn", P * C);
  layerNorm(pair, xn, P, C, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"));
  float* proj = scratch<float>("tri.proj", P * 2 * C); float* gate = scratch<float>("tri.gate", P * 2 * C);
  gemm(xn, F(Tn + "projection"), proj, P, C, 2 * C);
  gemm(xn, F(Tn + "gate"), gate, P, C, 2 * C);
  float* a = scratch<float>("tri.a", P * C); float* b = scratch<float>("tri.b", P * C);
  triSplitInterleavedK<<<blocks(P * 2 * C), 256, 0, STREAM>>>(proj, gate, mask, a, b, P, C);
  float* prod = scratch<float>("tri.prod", P * C);
  const float one = 1.f, zero = 0.f;
  // per channel (row-major [i][k] planes): outgoing out[i,j] = sum_k a[i,k] b[j,k]; incoming sum_k a[k,j] b[k,i]
  if (outgoing)
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, L, L, L, &one, b, L, P, a, L, P, &zero, prod, L, P, C));
  else
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, L, &one, a, L, P, b, L, P, &zero, prod, L, P, C));
  float* rowsP = scratch<float>("tri.rows", P * C);
  channelMajorToRowsK<<<blocks(P * C), 256, 0, STREAM>>>(prod, rowsP, P, C);
  layerNorm(rowsP, rowsP, P, C, F(Tn + "centerNormScale"), F(Tn + "centerNormOffset"));
  float* out = scratch<float>("tri.out", P * C);
  gemm(rowsP, F(Tn + "outputProjection"), out, P, C, C);
  float* g = scratch<float>("tri.g", P * C);
  gemm(xn, F(Tn + "gatingLinear"), g, P, C, C);
  gateMulAddK<<<blocks(P * C), 256, 0, STREAM>>>(pair, out, g, P * C);
}
// the SwiGLU transition over any rows (the MSA encoder's MSA track too), residual added
inline void rowsTransition(float* x, size_t P, int C, const std::string& Tn) {
  if (FAST) { transitionFast(x, P, C, Tn); return; }
  float* pair = x; int I = (int)dimOf("f/" + Tn + "transition2", 0);
  float* xn = scratch<float>("tr.xn", P * C);
  layerNorm(pair, xn, P, C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"));
  // a chunk of rows at a time: the widened rows are 2I per pair
  size_t chunk = std::max<size_t>(1, ((size_t)64 << 20) / (2 * (size_t)I));
  float* h = scratch<float>("tr.h", std::min(P, chunk) * 2 * I); float* g = scratch<float>("tr.g", std::min(P, chunk) * I);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    gemm(xn + r0 * C, F(Tn + "transition1"), h, r, C, 2 * I);
    swigluK<<<blocks(r * I), 256, 0, STREAM>>>(h, g, r, I);
    gemm(g, F(Tn + "transition2"), pair + r0 * C, r, I, C, 1.f);
  }
}
inline void pairTransition(float* pair, int L, int C, const std::string& Tn) { rowsTransition(pair, (size_t)L * L, C, Tn); }
// the 256-channel block on the fused kernels (fused256.cuh): triInK -> the f16 contraction -> triangleOutK,
// and transitionUpK -> the second GEMM (cuBLASLt, the residual as beta)
#include "../../af3/src/tricontract.cuh"
// a and b in bf16 and the product written bf16 (AF3's triangle: TRI_BF16), half the bytes of the f32 product
// both ways; LOCALFOLD_EF2_TRI_F16=1 keeps f16 operands and the f32 product
inline bool EF2_TRI_BF16 = !getenv("LOCALFOLD_EF2_TRI_F16");
template <class TA, class TP>
inline void triangle256As(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing,
                          cudaDataType ta, cudaDataType tp) {
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  TA* a = scratch<TA>("ftri.a", plane * C); TA* b = scratch<TA>("ftri.b", plane * C);
  half* t2 = scratch<half>("ftri.t2", plane * C);
  triIn256<TA>(pair, mask, Tn, a, b, t2, L, Lp, plane);
  TP* prod = scratch<TP>("ftri.prod", plane * C);
  if constexpr (std::is_same_v<TA, __nv_bfloat16>) {   // cuda/af3's contraction, a cached plan at every size
    triContractBf16(outgoing, Lp, plane, C, 1.f, a, b, prod, true);
  } else {
  const float one = 1.f, zero = 0.f;
  if (outgoing)
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, ta, Lp, plane, a, ta, Lp,
                                  plane, &zero, prod, tp, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  else
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, ta, Lp, plane, b, ta, Lp,
                                  plane, &zero, prod, tp, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  triangleOut<4>(prod, F(Tn + "centerNormScale"), F(Tn + "centerNormOffset"), Fh(Tn + "outputProjection"), t2, pair, L, Lp);
}
inline bool bf16Mma() {          // bf16 MMA: Ampere on (a T4's contraction stays f16 into f32)
  static int major = [] { int d, m; CK(cudaGetDevice(&d)); CK(cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMajor, d)); return m; }();
  return major >= 8;
}
inline void triangle256(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing) {
  if (EF2_TRI_BF16 && bf16Mma()) triangle256As<__nv_bfloat16, __nv_bfloat16>(pair, mask, L, C, Tn, outgoing, CUDA_R_16BF, CUDA_R_16BF);
  else triangle256As<half, float>(pair, mask, L, C, Tn, outgoing, CUDA_R_16F, CUDA_R_32F);
}
inline void transition256(float* pair, size_t P, int C, const std::string& Tn) {
  int I = (int)dimOf("f/" + Tn + "transition2", 0);
  if (!fused256Big()) {            // a T4: the 16-warp form, its rows normed in rounds (byte-identical)
    constexpr int WU = 16, R = 16 * WU;
    constexpr size_t smem = transitionUpSmem<256, WU, 16, 8>();
    size_t chunk = std::max<size_t>(R, ((size_t)64 << 20) / (2 * (size_t)I) / R * R);
    half* g = scratch<half>("ftr.g", std::min(P, chunk) * I);
    half* w1t = scratch<half>("ftr.w1t", (size_t)2 * C * I);
    tileTransitionUp(Fh(Tn + "transition1"), C, I, w1t);
    static bool attr = false;
    if (!attr) { smemAttr((transitionUpK<256, WU, 16, 8>), (int)smem); attr = true; }
    for (size_t r0 = 0; r0 < P; r0 += chunk) {
      size_t r = std::min(chunk, P - r0);
      transitionUpK<256, WU, 16, 8><<<(unsigned)((r + R - 1) / R), 32 * WU, smem, STREAM>>>(
        pair + r0 * C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"), w1t, g, r, I);
      ltGemm(g, Fh(Tn + "transition2"), pair + r0 * C, false, r, I, C, nullptr, false, 1.f);
    }
    return;
  }
  // whole waves of transitionUpK within the same ~64 MB of widened rows (see transitionUpWaveRows)
  size_t chunk = transitionUpChunkRows<256, 8>(transitionUpSmem<256, 8>(), ((size_t)64 << 20) / (2 * (size_t)I));
  half* w1t = scratch<half>("ftr.w1t", (size_t)2 * C * I);
  tileTransitionUp(Fh(Tn + "transition1"), C, I, w1t);
  if (PAIR16) {
    // a bf16 pair: the gated rows in bf16 and the second GEMM bf16 throughout, accumulating straight into the pair
    using B16 = __nv_bfloat16;
    constexpr int WU = 8, R = 16 * WU;
    constexpr size_t smem = transitionUpSmem<256, WU>();
    B16* g = scratch<B16>("ftr.gbf", std::min(P, chunk) * I);
    static bool attr = false;
    if (!attr) { smemAttr((transitionUpK<256, WU, 32, 1, B16, B16>), (int)smem); attr = true; }
    const float one = 1.f;
    for (size_t r0 = 0; r0 < P; r0 += chunk) {
      size_t r = std::min(chunk, P - r0);
      B16* rows = reinterpret_cast<B16*>(pair) + r0 * C;
      transitionUpK<256, WU, 32, 1, B16, B16><<<(unsigned)((r + R - 1) / R), 32 * WU, smem, STREAM>>>(
        reinterpret_cast<float*>(rows), F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"), w1t, g, r, I);
      CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, C, (int)r, I, &one, Wbf("f/" + Tn + "transition2"), CUDA_R_16BF, C, g,
                      CUDA_R_16BF, I, &one, rows, CUDA_R_16BF, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    return;
  }
  half* g = scratch<half>("ftr.g", std::min(P, chunk) * I);
  for (size_t r0 = 0; r0 < P; r0 += chunk) {
    size_t r = std::min(chunk, P - r0);
    transitionUp<8>(pair + r0 * C, F(Tn + "inputLayerNormScale"), F(Tn + "inputLayerNormOffset"), w1t, g, r, I);
    ltGemm(g, Fh(Tn + "transition2"), pair + r0 * C, false, r, I, C, nullptr, false, 1.f);
  }
}
#include "../../af3/src/triblocked.cuh"
// one operand's [projection | gate] (C, 2C) f16 from the interleaved projection and gate (column 2ch is a's
// channel ch, 2ch + 1 b's), side 0 = a
__global__ void operandInterleavedK(const float* proj, const float* gate, half* w, int C, int side) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * 2 * C) return;
  int k = (int)(t / (2 * C)), o = (int)(t % (2 * C));
  w[t] = __float2half(o < C ? proj[(size_t)k * 2 * C + 2 * o + side] : gate[(size_t)k * 2 * C + 2 * (o - C) + side]);
}
// the triangle multiplication in output blocks (cuda/af3/src/triblocked.cuh), for a card short of room
inline void triangleBlockedEf2(float* pair, const float* mask, int L, int C, const std::string& Tn, bool outgoing) {
  static std::map<std::pair<std::string, int>, half*> ops;     // (a derived weight: forgotten with the others)
  static bool hooked = false;
  if (!hooked) { FORGET_HOOKS.push_back([] { for (auto& [k, p] : ops) CK(cudaFree(p)); ops.clear(); }); hooked = true; }
  auto opOf = [&](int side) {
    auto key = std::make_pair(Tn, side);
    auto it = ops.find(key);
    if (it != ops.end()) return it->second;
    half* w = dallocT<half>((size_t)C * 2 * C);
    operandInterleavedK<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(F(Tn + "projection"), F(Tn + "gate"), w, C, side);
    return ops[key] = w;
  };
  TriBlockedW w{ F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"), opOf(0), opOf(1), nullptr, nullptr,
                 F(Tn + "centerNormScale"), F(Tn + "centerNormOffset"), Fh(Tn + "outputProjection"), nullptr,
                 Fh(Tn + "gatingLinear"), nullptr };
  triangleBlockedHalf(pair, mask, L, C, w, outgoing, (size_t)64 << 20);
}
inline void trunkBlock(float* pair, const float* mask, int L, int C, const std::string& prefix, int b) {
  std::string B = prefix + "/" + std::to_string(b) + "/";
  // on a card short of room, the triangles in output blocks where the whole forms would not fit with room
  // to spare (the transitions already go a chunk of rows at a time)
  if (FAST && shortPair((size_t)L * L, C)) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    if (!roomFor(5 * plane * C * 2, { "ftri.a", "ftri.b", "ftri.t2", "ftri.prod", "ftri.xn", "ftri.pg", "ftri.cn", "ftri.out" })) {
      releaseScratch({ "ftri." });
      triangleBlockedEf2(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true);
      triangleBlockedEf2(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false);
      if (FUSED256 && C == 256 && L >= FUSED256_MIN_TOKENS && fused256Fits()) transition256(pair, (size_t)L * L, C, B + "pairTransition/");
      else transitionFast(pair, (size_t)L * L, C, B + "pairTransition/");
      return;
    }
  }
  // the fused kernels' 64-128-pair tiles leave a small pair track's device idle: measured, warm, trunk of
  // 4 passes - 68 tokens 43.2 ms unfused against 52.7 fused; 92: 71.5 / 58.3; 195: 201 / 190;
  // 261: 368 / 320; 476: 1131 / 953
  if (FAST && FUSED256 && C == 256 && L >= FUSED256_MIN_TOKENS && fused256Fits()) {
    triangle256(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true);
    triangle256(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false);
    transition256(pair, (size_t)L * L, C, B + "pairTransition/");
    return;
  }
  if (FAST) {
    triangleFast(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true);
    triangleFast(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false);
    transitionFast(pair, (size_t)L * L, C, B + "pairTransition/");
    return;
  }
  triangle(pair, mask, L, C, B + "triangleMultiplicationOutgoing/", true);
  triangle(pair, mask, L, C, B + "triangleMultiplicationIncoming/", false);
  pairTransition(pair, L, C, B + "pairTransition/");
}

// the recycle loop: z (out) = the trunk's last pass
// ---------------------------------------------------------------- the released models' recycle (parcae)
// counter-based randoms: a uniform in [0, 1) from (seed, stream, index), splitmix64 twice - so a fold is the
// same fold for the same seed, whatever the launch shape
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ float uniform01(uint64_t seed, uint64_t stream, uint64_t i) {
  return (float)(mix64(seed ^ mix64(stream * 0x100000001B3ull + i)) >> 40) * (1.f / 16777216.f);
}
// the initial pair state: a normal of std sqrt(2 / (5 C)) truncated at 3 std (biohub's _init_pair_state -
// trunc_normal_, which redraws past the bounds; here a Box-Muller draw redrawn the same way)
__global__ void truncNormalK(float* z, size_t n, float std, uint64_t seed) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  float v = 0.f;
  for (int k = 0; k < 64; ++k) {
    float u1 = fmaxf(uniform01(seed, 2 * k, t), 1e-7f), u2 = uniform01(seed, 2 * k + 1, t);
    v = sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
    if (fabsf(v) <= 3.f) break;
  }
  z[t] = fminf(fmaxf(v, -3.f), 3.f) * std;
}
// out = dropout(in, p): zeroed with probability p, else scaled by 1 / (1 - p) (F.dropout, training=True)
__global__ void dropoutK(const float* in, float* out, size_t n, float p, uint64_t seed, uint64_t stream) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  out[t] = uniform01(seed, stream, t) < p ? 0.f : in[t] / (1.f - p);
}
// z = a * z + y, a per channel
__global__ void decayUpdateK(float* z, const float* y, const float* a, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * C) return;
  z[t] = a[t % C] * z[t] + y[t];
}
inline bool parcaeRecycle() { return M.has("f/recycle/decay"); }
// ---------------------------------------------------------------- the pair in bf16 through the blocks (EF2_P16)
// Where every block takes the fused 256-channel path on Ampere or later (and the card has room), a run of blocks
// works on a bf16 copy of the pair - the triangle's input and output kernels and the transition read and write
// half the bytes, the transition's second GEMM accumulating into it in bf16 - converted in before the run and out
// after it (the recycle's own arithmetic on z stays f32). LOCALFOLD_PAIR_F32=1 keeps the f32 pair.
__global__ void ef2ToBf16K(const float* in, __nv_bfloat16* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}
__global__ void ef2FromBf16K(const __nv_bfloat16* in, float* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}
inline bool ef2Pair16(int L, int C) {
  static const bool off = getenv("LOCALFOLD_PAIR_F32") != nullptr;
  return !off && FAST && FUSED256 && C == 256 && L >= FUSED256_MIN_TOKENS && fused256Big() && bf16Mma() &&
         !shortPair((size_t)L * L, C);
}
inline void trunkBlocks(float* pair, const float* mask, int L, int C, const std::string& prefix, int from, int to) {
  if (from >= to) return;
  if (!ef2Pair16(L, C)) { for (int b = from; b < to; ++b) trunkBlock(pair, mask, L, C, prefix, b); return; }
  size_t n = (size_t)L * L * C;
  __nv_bfloat16* p16 = scratch<__nv_bfloat16>("ef2.p16", n);
  ef2ToBf16K<<<blocks(n), 256, 0, STREAM>>>(pair, p16, n);
  PAIR16 = true;
  for (int b = from; b < to; ++b) trunkBlock(reinterpret_cast<float*>(p16), mask, L, C, prefix, b);
  PAIR16 = false;
  ef2FromBf16K<<<blocks(n), 256, 0, STREAM>>>(p16, pair, n);
}

inline void foldingTrunk(int T, int C, const float* zInitP, float* z, int loops, bool check,
                         const float* lmZ = nullptr, uint64_t seed = 0, const float* lmHost = nullptr) {
  size_t P = (size_t)T * T;
  float* mask = scratch<float>("trunk.mask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);      // every token is real (the token mask is all ones)
  if (parcaeRecycle()) {
    // the released models (ESMFold2, ESMFold2-Fast): z starts as noise; each pass refines a dropped-out copy of
    // the language model's pair through its own pair-only blocks, injects z_init plus that, LayerNorm'd and
    // projected, into a per-channel decay of z, and runs the trunk; after the last pass the readout and the
    // coda (biohub's modeling_esmfold2.py _run_one_loop and forward)
    if (!zInitP || (!lmZ && !lmHost)) { fprintf(stderr, "the parcae recycle needs z_init and the language model's pair whole\n"); exit(1); }
    // EF2_DETERMINISTIC=1: no dropout and a zero initial state - af3-any-model's reading (its recycle starts from
    // zeros), and with its LM_PAIR_DROPOUT set to 0 a fold the two can compare tensor for tensor
    static const bool deterministic = getenv("EF2_DETERMINISTIC") != nullptr;
    const float p = deterministic ? 0.f : (float)M.meta("meta/lmDropout");
    const int blocksN = (int)M.meta("meta/blocks"), lmBlocks = (int)M.meta("meta/lmEncoderBlocks"),
              codaBlocks = (int)M.meta("meta/codaBlocks");
    if (deterministic) CK(cudaMemsetAsync(z, 0, P * C * 4, STREAM));
    else truncNormalK<<<blocks(P * C), 256, 0, STREAM>>>(z, P * C, sqrtf(2.f / (5.f * C)), seed ^ 0x5eedull);
    float* inject = scratch<float>("trunk.inject", P * C);
    size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));
    float* xn = scratch<float>("trunk.xn", chunk * C); float* y = scratch<float>("trunk.y", chunk * C);
    for (int loop = 0; loop < loops; ++loop) {
      if (lmHost) {                                 // (parked in host memory: copied in, then dropped out in place)
        CK(cudaMemcpyAsync(inject, lmHost, P * C * 4, cudaMemcpyHostToDevice, STREAM));
        if (p > 0.f) dropoutK<<<blocks(P * C), 256, 0, STREAM>>>(inject, inject, P * C, p, seed, 1000 + loop);
      } else if (p > 0.f) dropoutK<<<blocks(P * C), 256, 0, STREAM>>>(lmZ, inject, P * C, p, seed, 1000 + loop);
      else CK(cudaMemcpyAsync(inject, lmZ, P * C * 4, cudaMemcpyDeviceToDevice, STREAM));
      trunkBlocks(inject, mask, T, C, "lmEncoder/blocks", 0, lmBlocks);
      addK<<<blocks(P * C), 256, 0, STREAM>>>(inject, zInitP, P * C);
      for (size_t r0 = 0; r0 < P; r0 += chunk) {
        size_t r = std::min(chunk, P - r0);
        layerNorm(inject + r0 * C, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
        gemm(xn, F("recycle/projection"), y, r, C, C);
        decayUpdateK<<<blocks(r * C), 256, 0, STREAM>>>(z + r0 * C, y, F("recycle/decay"), r, C);
      }
      trunkBlocks(z, mask, T, C, "blocks", 0, blocksN);
    }
    auto dumpPair = [&](const char* env) {             // (comparison aids: the pair at a seam, raw float32)
      if (!getenv(env)) return;
      std::vector<float> h(P * C); CK(cudaMemcpy(h.data(), z, P * C * 4, cudaMemcpyDeviceToHost));
      FILE* f = fopen(getenv(env), "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f);
    };
    dumpPair("EF2_SAVE_PRE_READOUT");
    for (size_t r0 = 0; r0 < P; r0 += chunk) {          // the readout, a chunk of rows at a time, in place
      size_t r = std::min(chunk, P - r0);
      gemm(z + r0 * C, F("readout"), y, r, C, C);
      CK(cudaMemcpyAsync(z + r0 * C, y, r * C * 4, cudaMemcpyDeviceToDevice, STREAM));
    }
    dumpPair("EF2_SAVE_PRE_CODA");
    trunkBlocks(z, mask, T, C, "coda/blocks", 0, codaBlocks);
    return;
  }
  CK(cudaMemsetAsync(z, 0, P * C * 4, STREAM));
  size_t chunk = std::min<size_t>(P, ((size_t)64 << 20) / (4 * (size_t)C));   // the recycle row by row, 64 MB at a time
  float* xn = scratch<float>("trunk.xn", chunk * C);
  int blocksN = (int)M.meta("meta/blocks");
  for (int loop = 0; loop < loops; ++loop) {
    for (size_t r0 = 0; r0 < P; r0 += chunk) {
      size_t r = std::min(chunk, P - r0);
      layerNorm(z + r0 * C, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
      gemm(xn, F("recycle/projection"), z + r0 * C, r, C, C);
    }
    if (zInitP) addK<<<blocks(P * C), 256, 0, STREAM>>>(z, zInitP, P * C);
    else zInitAddStreamed(z, T, C);
    if (check) checkOracle(("trunk pass " + std::to_string(loop) + " in").c_str(), z, P * C, "o/loop" + std::to_string(loop) + "/in");
    // (a CUDA graph of the 24 blocks, replayed for passes after the first, measured no faster: 453
    // against 455 ms of trunk at 261 tokens; the GPU is busy, the launches are not the cost)
    if (check) for (int b = 0; b < blocksN; ++b) trunkBlock(z, mask, T, C, "blocks", b);   // (an oracle reads f32)
    else trunkBlocks(z, mask, T, C, "blocks", 0, blocksN);
    if (check) checkOracle(("trunk pass " + std::to_string(loop) + " out").c_str(), z, P * C, "o/loop" + std::to_string(loop) + "/out");
    if (getenv("EF2_PASS_TIMES")) {
      static auto t0 = std::chrono::steady_clock::now();
      CK(cudaStreamSynchronize(STREAM));
      printf("    pass %d done at %.1f ms\n", loop, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
  }
}

inline void distogram(const float* z, int T, int C, float* logits) {
  size_t P = (size_t)T * T; int Bn = (int)dimOf("f/distogram/weights", 1);
  float* zs = scratch<float>("dg.sym", P * C);
  symmetriseK<<<blocks(P * C), 256, 0, STREAM>>>(z, zs, T, C);
  gemm(zs, F("distogram/weights"), logits, P, C, Bn);
  addBias(logits, F("distogram/bias"), P, Bn);
}
