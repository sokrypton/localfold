// ESMFold2's launchers for the 256-channel fused kernels (native/af3/src/fused256.cuh, shared with the AF3
// lineage's 256-channel models)
#pragma once
#include "fast.cuh"
#include "../../af3/src/fused256.cuh"

// ---------------------------------------------------------------- launchers
inline bool FUSED256 = true;      // --no-fused256: the cuBLASLt path these replace
inline int FUSED256_MIN_TOKENS = 80;
// their shared memory: 66-80 KB a block, beyond a T4's 64 KB (there the unfused f16 path runs)
constexpr size_t TRI_IN256_SMEM = (size_t)128 * (256 + 8) * 2, TRI_OUT256_SMEM = (size_t)256 * 65 * 4 + 2 * 64 * 4,
                 TRANS_UP256_SMEM = (size_t)2 * 2 * 256 * (32 + 8) * 2;
inline bool fused256Fits() { return fitsSmem(std::max({TRI_IN256_SMEM, TRI_OUT256_SMEM, TRANS_UP256_SMEM})); }
template <int WARPS>
void transitionUp(const float* x, const float* sc, const float* of, const half* W1, half* gated, size_t rows, int I) {
  constexpr int C = 256, R = 16 * WARPS;
  size_t smem = std::max((size_t)R * (C + 8) * 2, 2 * (size_t)2 * C * (32 + 8) * 2);   // 82 KB: two blocks an SM
  static bool attr = false;
  if (!attr) { smemAttr((transitionUpK<C, WARPS>), (int)smem); attr = true; }
  transitionUpK<C, WARPS><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(x, sc, of, W1, gated, rows, I);
}
template <int WARPS>
void triangleOut(const float* prod, const float* sc, const float* of, const half* Wout, const half* t2, float* pair,
                 int L, int Lp) {
  triangleOutRun<256, WARPS>(prod, sc, of, Wout, t2, pair, L, Lp);
}
template <int WARPS>
void triIn256(const float* pair, const float* mask, const std::string& Tn, half* a, half* b, half* t2, int n, int np, size_t cs) {
  constexpr int C = 256, R = 16 * WARPS;
  size_t pp = (size_t)np * np, smem = (size_t)R * (C + 8) * 2;
  static bool attr = false;
  if (!attr) { smemAttr((triIn256K<C, WARPS>), (int)smem); attr = true; }
  std::string pg = concatColumns("f/" + Tn + "projectionGate~", C, {{"f/" + Tn + "projection", 2 * C, false},
                                                                   {"f/" + Tn + "gate", 2 * C, false}});
  triIn256K<C, WARPS><<<(unsigned)((pp + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    pair, mask, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"), Wh(pg), Fh(Tn + "gatingLinear"), a, b, t2, n, np, cs);
}
