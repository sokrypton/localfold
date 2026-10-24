// --fast: BioEmu2's pair track on cuda/ef2's fused 256-channel kernels (cuda/af3's fused256.cuh, tricontract.cuh -
// the f16/bf16 forms ESMFold2's trunk runs, through ef2's triangle256 / transition256 / triangleFast /
// transitionFast). Boltz-2's TriangleMultiplicationFull and ESMFold2's trunk triangle are one computation in two
// layouts, so each block's weights are re-laid once, as derived device weights under "f/be2/<block>/...":
//   projection, gate   [C, 2C] (in, out), a and b INTERLEAVED per channel (column 2ch is a's ch, 2ch + 1 b's), from
//                      p_in / g_in [2C, C] (out, in) whose rows [0, C) are a and [C, 2C) b
//   outputProjection, gatingLinear [C, C] (in, out): p_out^T, g_out^T;  LN in / out as leftNormInput / centerNorm
//   transition1 [C, 2I] = [fc1^T | fc2^T] (the SwiGLU's [gate | value]), transition2 [I, C] = fc3^T
// 🔴 THE INCOMING TRIANGLE SWAPS a AND b: ef2's incoming is sum_k a[k][j] b[k][i] and Boltz's sum_k a[k][i] b[k][j],
// so the incoming block's "a" is BioEmu2's b. The outgoing one (sum_k a[i][k] b[j][k]) is the same in both.
#pragma once
#include "trunk.cuh"
// (declared for ef2's streamed z_init, which this port never calls: ESMFold2's language-model shim is not here)
int lmPairBlock(int T, int P);
void lmPairRows(const float* s, int T, int P, int i0, int b, float* o);
#include "../../ef2/src/trunk.cuh"

namespace be2 {

// dst[k][2ch + s] = __float2half(src[(sideOf(s)) * C + ch][k])   (src [2C, C] torch-layout; sideOf swaps for the incoming block)
__global__ void interleaveHalfK(const float* __restrict__ src, half* __restrict__ dst, int C, bool swap) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * 2 * C) return;
  int k = (int)(t / (2 * C)), o = (int)(t % (2 * C)), ch = o >> 1, s = o & 1;
  int side = swap ? 1 - s : s;
  dst[t] = __float2half(src[((size_t)side * C + ch) * C + k]);
}
// dst[k][o] (in, out) = __float2half(src[o][k]) (out, in), into a column slice [col0, col0 + out) of a row `ld` wide
__global__ void transposeColsHalfK(const float* __restrict__ src, half* __restrict__ dst, int out, int in, int ld, int col0) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)out * in) return;
  int k = (int)(t / out), o = (int)(t % out);
  dst[(size_t)k * ld + col0 + o] = __float2half(src[(size_t)o * in + k]);
}
// dst[k][0..I-1] = src1[0..I-1][k], dst[k][I..2I-1] = src2[0..I-1][k] in one pass
__global__ void transposeCols2HalfK(const float* __restrict__ src1, const float* __restrict__ src2,
                                    half* __restrict__ dst, int I, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * 2 * I) return;
  int k = (int)(t / (2 * I)), o = (int)(t % (2 * I));
  const float* src = (o < I) ? src1 : src2;
  int oi = (o < I) ? o : (o - I);
  dst[t] = __float2half(src[(size_t)oi * C + k]);
}
// Allocate derived trunk weights from wpool<half> directly into WH (instead of 676 individual cudaMalloc + toHalfK calls)
// and alias 1D LayerNorm scale/offset vectors directly to their existing device pointers without copying.
inline void deriveHalfWeight(const std::string& key, const std::function<void(half*)>& fill, size_t n) {
  std::string fk = "f/" + key;
  if (WH.count(fk)) return;
  half* d = wpool<half>(n);
  fill(d);
  WH[fk] = d;
  WLEN[fk] = n;
}
inline void aliasWeight(const std::string& key, const float* src, size_t n) {
  std::string fk = "f/" + key;
  if (WF.count(fk)) return;
  deviceWeight(fk, const_cast<float*>(src), n);
}
// the ef2-layout name prefix of a BioEmu2 triangle (prefix e.g. ".../pairmixer_module/layers/tri_mul_out/", layer l)
inline std::string ef2Triangle(const std::string& p, int l, bool incoming) {
  const int C = 256;
  const std::string Tn = "be2/" + p + std::to_string(l) + "/";
  auto w = [&](const std::string& n) { return PL(p + n, l); };
  deriveHalfWeight(Tn + "projection", [&](half* d) { interleaveHalfK<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(w("p_in/weight"), d, C, incoming); }, (size_t)C * 2 * C);
  deriveHalfWeight(Tn + "gate", [&](half* d) { interleaveHalfK<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(w("g_in/weight"), d, C, incoming); }, (size_t)C * 2 * C);
  deriveHalfWeight(Tn + "outputProjection", [&](half* d) { transposeColsHalfK<<<blocks((size_t)C * C), 256, 0, STREAM>>>(w("p_out/weight"), d, C, C, C, 0); }, (size_t)C * C);
  deriveHalfWeight(Tn + "gatingLinear", [&](half* d) { transposeColsHalfK<<<blocks((size_t)C * C), 256, 0, STREAM>>>(w("g_out/weight"), d, C, C, C, 0); }, (size_t)C * C);
  for (auto [mine, theirs] : std::initializer_list<std::pair<const char*, const char*>>{std::pair<const char*, const char*>{"norm_in/weight", "leftNormInputScale"}, {"norm_in/bias", "leftNormInputOffset"},
                              {"norm_out/weight", "centerNormScale"}, {"norm_out/bias", "centerNormOffset"}}) {
    aliasWeight(Tn + theirs, w(mine), C);
  }
  return Tn;
}
inline std::string ef2Transition(const std::string& p, int l) {
  const int C = 256;
  const int I = dimP(p + "fc1/weight", 1);
  const std::string Tn = "be2/" + p + std::to_string(l) + "/";
  auto w = [&](const std::string& n) { return PL(p + n, l); };
  deriveHalfWeight(Tn + "transition1", [&](half* d) {
    transposeCols2HalfK<<<blocks((size_t)C * 2 * I), 256, 0, STREAM>>>(w("fc1/weight"), w("fc2/weight"), d, I, C);
  }, (size_t)C * 2 * I);
  deriveHalfWeight(Tn + "transition2", [&](half* d) { transposeColsHalfK<<<blocks((size_t)C * I), 256, 0, STREAM>>>(w("fc3/weight"), d, C, I, C, 0); }, (size_t)I * C);
  // (ef2 reads the widths off the tensors' dims)
  M.index["f/" + Tn + "transition2#0"] = Entry{'m', 0, 0, (double)I, 0};
  M.index["f/" + Tn + "transition2#1"] = Entry{'m', 0, 0, (double)C, 0};
  M.index["f/" + Tn + "transition1#0"] = Entry{'m', 0, 0, (double)C, 0};
  M.index["f/" + Tn + "transition1#1"] = Entry{'m', 0, 0, (double)(2 * I), 0};
  for (auto [mine, theirs] : std::initializer_list<std::pair<const char*, const char*>>{std::pair<const char*, const char*>{"norm/weight", "inputLayerNormScale"}, {"norm/bias", "inputLayerNormOffset"}}) {
    aliasWeight(Tn + theirs, w(mine), C);
  }
  return Tn;
}
// Persistent dedicated device buffer (not in scratchPool, so releaseScratch() before the denoiser never reclaims or
// aliases it) for triangleFast's all-ones pair mask; triangle256 takes mask = nullptr directly (triIn256K treats a
// null mask as all-ones without global memory loads).
__global__ void fillOnesK(float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = 1.f;
}
inline const float* pairOnes(size_t P) {
  static float* ones = nullptr;
  static size_t cap = 0;
  if (P > cap) {
    if (ones) CK(cudaFree(ones));
    cap = std::max<size_t>(P, 256 * 1024);
    ones = dalloc(cap);
    fillOnesK<<<blocks(cap), 256, 0, STREAM>>>(ones, cap);
  }
  return ones;
}
// the fast block pieces (ef2's own choice of form by size)
inline void trimulFast(float* z, const std::string& p, int l, bool outgoing) {
  const int T = D_.T, C = 256;
  const std::string Tn = ef2Triangle(p, l, !outgoing);
  if (shortPair((size_t)T * T, C)) {
    int Lp = (T + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    if (!roomFor(5 * plane * C * 2, { "ftri.a", "ftri.b", "ftri.t2", "ftri.prod", "ftri.xn", "ftri.pg", "ftri.cn", "ftri.out" })) {
      releaseScratch({ "ftri.", "ftr.", "be2.pwa.", "be2.opm.", "be2.tr.", "be2.ie.", "be2.cl.", "be2.pt.", "be2.zi." });
      static const bool fusedBlocks = !getenv("LOCALFOLD_TRIB_FUSED") || atoi(getenv("LOCALFOLD_TRIB_FUSED"));
      if (fusedBlocks && FUSED256 && T >= FUSED256_MIN_TOKENS && fused256Fits() && EF2_TRI_BF16 && bf16Mma()) {
        triangleBlockedFusedEf2(z, nullptr, T, C, Tn, outgoing);
      } else {
        triangleBlockedEf2(z, nullptr, T, C, Tn, outgoing);
      }
      return;
    }
  }
  if (FUSED256 && T >= FUSED256_MIN_TOKENS && fused256Fits()) triangle256(z, nullptr, T, C, Tn, outgoing);
  else triangleFast(z, pairOnes((size_t)T * T), T, C, Tn, outgoing);
}
inline void pairTransitionFast(float* z, const std::string& p, int l) {
  const int T = D_.T, C = 256;
  const std::string Tn = ef2Transition(p, l);
  if (FUSED256 && T >= FUSED256_MIN_TOKENS && fused256Fits()) transition256(z, (size_t)T * T, C, Tn);
  else transitionFast(z, (size_t)T * T, C, Tn);
}

}  // namespace be2
