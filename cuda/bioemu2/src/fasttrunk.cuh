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

// dst[k][2ch + s] = src[(sideOf(s)) * C + ch][k]   (src [2C, C] torch-layout; sideOf swaps for the incoming block)
__global__ void interleaveK(const float* src, float* dst, int C, bool swap) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * 2 * C) return;
  int k = (int)(t / (2 * C)), o = (int)(t % (2 * C)), ch = o >> 1, s = o & 1;
  int side = swap ? 1 - s : s;
  dst[t] = src[((size_t)side * C + ch) * C + k];
}
// dst[k][o] (in, out) = src[o][k] (out, in), into a column slice [col0, col0 + out) of a row `ld` wide
__global__ void transposeColsK(const float* src, float* dst, int out, int in, int ld, int col0) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)out * in) return;
  int k = (int)(t / out), o = (int)(t % out);
  dst[(size_t)k * ld + col0 + o] = src[(size_t)o * in + k];
}
inline void deriveWeight(const std::string& key, const std::function<void(float*)>& fill, size_t n) {
  if (WF.count("f/" + key)) return;
  float* d = dalloc(n);
  fill(d);
  deviceWeight("f/" + key, d, n);
}
// the ef2-layout name prefix of a BioEmu2 triangle (prefix e.g. ".../pairmixer_module/layers/tri_mul_out/", layer l)
inline std::string ef2Triangle(const std::string& p, int l, bool incoming) {
  const int C = 256;
  const std::string Tn = "be2/" + p + std::to_string(l) + "/";
  auto w = [&](const std::string& n) { return PL(p + n, l); };
  deriveWeight(Tn + "projection", [&](float* d) { interleaveK<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(w("p_in/weight"), d, C, incoming); }, (size_t)C * 2 * C);
  deriveWeight(Tn + "gate", [&](float* d) { interleaveK<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(w("g_in/weight"), d, C, incoming); }, (size_t)C * 2 * C);
  deriveWeight(Tn + "outputProjection", [&](float* d) { transposeColsK<<<blocks((size_t)C * C), 256, 0, STREAM>>>(w("p_out/weight"), d, C, C, C, 0); }, (size_t)C * C);
  deriveWeight(Tn + "gatingLinear", [&](float* d) { transposeColsK<<<blocks((size_t)C * C), 256, 0, STREAM>>>(w("g_out/weight"), d, C, C, C, 0); }, (size_t)C * C);
  for (auto [mine, theirs] : std::initializer_list<std::pair<const char*, const char*>>{std::pair<const char*, const char*>{"norm_in/weight", "leftNormInputScale"}, {"norm_in/bias", "leftNormInputOffset"},
                              {"norm_out/weight", "centerNormScale"}, {"norm_out/bias", "centerNormOffset"}}) {
    const float* src = w(mine);
    deriveWeight(Tn + theirs, [&](float* d) { CK(cudaMemcpyAsync(d, src, C * 4, cudaMemcpyDeviceToDevice, STREAM)); }, C);
  }
  return Tn;
}
inline std::string ef2Transition(const std::string& p, int l) {
  const int C = 256;
  const int I = dimP(p + "fc1/weight", 1);
  const std::string Tn = "be2/" + p + std::to_string(l) + "/";
  auto w = [&](const std::string& n) { return PL(p + n, l); };
  deriveWeight(Tn + "transition1", [&](float* d) {
    transposeColsK<<<blocks((size_t)I * C), 256, 0, STREAM>>>(w("fc1/weight"), d, I, C, 2 * I, 0);
    transposeColsK<<<blocks((size_t)I * C), 256, 0, STREAM>>>(w("fc2/weight"), d, I, C, 2 * I, I);
  }, (size_t)C * 2 * I);
  deriveWeight(Tn + "transition2", [&](float* d) { transposeColsK<<<blocks((size_t)C * I), 256, 0, STREAM>>>(w("fc3/weight"), d, C, I, C, 0); }, (size_t)I * C);
  // (ef2 reads the widths off the tensors' dims)
  M.index["f/" + Tn + "transition2#0"] = Entry{'m', 0, 0, (double)I, 0};
  M.index["f/" + Tn + "transition2#1"] = Entry{'m', 0, 0, (double)C, 0};
  M.index["f/" + Tn + "transition1#0"] = Entry{'m', 0, 0, (double)C, 0};
  M.index["f/" + Tn + "transition1#1"] = Entry{'m', 0, 0, (double)(2 * I), 0};
  for (auto [mine, theirs] : std::initializer_list<std::pair<const char*, const char*>>{std::pair<const char*, const char*>{"norm/weight", "inputLayerNormScale"}, {"norm/bias", "inputLayerNormOffset"}}) {
    const float* src = w(mine);
    deriveWeight(Tn + theirs, [&](float* d) { CK(cudaMemcpyAsync(d, src, C * 4, cudaMemcpyDeviceToDevice, STREAM)); }, C);
  }
  return Tn;
}
// 🔴 FILLED EVERY CALL, ON THE DEVICE. It was filled once and remembered by (size, address) - and releaseScratch()
// frees it before the denoiser, whose buffers then write over that memory, and the NEXT job's scratch of the same size
// comes back at the same address: the resident server's second fold of a same-length input ran every triangle under a
// mask of garbage (median CA-CA 7-9 A, any mode). A T^2 fill beside an O(T^3) triangle is nothing.
__global__ void fillOnesK(float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = 1.f;
}
inline const float* pairOnes(size_t P) {
  float* ones = scratch<float>("be2.ft.ones", P);
  fillOnesK<<<blocks(P), 256, 0, STREAM>>>(ones, P);
  return ones;
}
// the fast block pieces (ef2's own choice of form by size)
inline void trimulFast(float* z, const std::string& p, int l, bool outgoing) {
  const int T = D_.T, C = 256;
  const std::string Tn = ef2Triangle(p, l, !outgoing);
  const float* mask = pairOnes((size_t)T * T);
  if (FUSED256 && T >= FUSED256_MIN_TOKENS && fused256Fits()) triangle256(z, mask, T, C, Tn, outgoing);
  else triangleFast(z, mask, T, C, Tn, outgoing);
}
inline void pairTransitionFast(float* z, const std::string& p, int l) {
  const int T = D_.T, C = 256;
  const std::string Tn = ef2Transition(p, l);
  if (FUSED256 && T >= FUSED256_MIN_TOKENS && fused256Fits()) transition256(z, (size_t)T * T, C, Tn);
  else transitionFast(z, (size_t)T * T, C, Tn);
}

}  // namespace be2
