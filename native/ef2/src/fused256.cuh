// ESMFold2's launchers for the 256-channel fused kernels (native/af3/src/fused256.cuh, shared with the AF3
// lineage's 256-channel models)
#pragma once
#include "fast.cuh"
#include "../../af3/src/fused256.cuh"

// ---------------------------------------------------------------- launchers
inline bool FUSED256 = true;      // --no-fused256: the cuBLASLt path these replace
inline int FUSED256_MIN_TOKENS = 80;
// their shared memory: 66-80 KB a block, beyond a T4's 64 KB - which takes the T4 forms below instead
constexpr size_t TRI_IN256_SMEM = (size_t)128 * (256 + 8) * 2, TRI_OUT256_SMEM = (size_t)256 * 65 * 4 + 2 * 64 * 4,
                 TRANS_UP256_SMEM = transitionUpSmem<256, 8>();
inline bool fused256Big() { return fitsSmem(std::max({TRI_IN256_SMEM, TRI_OUT256_SMEM, TRANS_UP256_SMEM})); }
// ...or their T4 forms (native/af3's rounds: the input kernel and the transition at 16 warps, their rows normed
// 32 at a time; the output kernel's bf16 tile), byte-identical to the above
constexpr size_t TRI_IN256_T4 = triIn256Smem<half>(256, 16, 8), TRANS_UP256_T4 = transitionUpSmem<256, 16, 16, 8>(),
                 TRI_OUT256_T4 = std::max(triangleOutSmem<256, 4, __nv_bfloat16, 16, __nv_bfloat16>(),
                                          triangleOutSmem<256, 4, __nv_bfloat16, 16, float>());
inline bool fused256Fits() { return fused256Big() || fitsSmem(std::max({TRI_IN256_T4, TRANS_UP256_T4, TRI_OUT256_T4})); }
template <int WARPS>
void transitionUp(const float* x, const float* sc, const float* of, const half* W1, half* gated, size_t rows, int I) {
  constexpr int C = 256, R = 16 * WARPS;
  size_t smem = transitionUpSmem<C, WARPS>();   // 66 KB: two blocks an SM
  static bool attr = false;
  if (!attr) { smemAttr((transitionUpK<C, WARPS>), (int)smem); attr = true; }
  transitionUpK<C, WARPS><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(x, sc, of, W1, gated, rows, I);
}
template <int WARPS, class TP>
void triangleOut(const TP* prod, const float* sc, const float* of, const half* Wout, const half* t2, float* pair,
                 int L, int Lp) {
  triangleOutRun<256, WARPS, TP>(prod, sc, of, Wout, t2, pair, L, Lp);
}
template <int WARPS, class TA, int XROUNDS = 1>
void triIn256Form(const float* pair, const float* mask, const std::string& Tn, TA* a, TA* b, half* t2, int n, int np, size_t cs) {
  constexpr int C = 256, R = 16 * WARPS;
  size_t pp = (size_t)np * np, smem = XROUNDS == 1 ? (size_t)R * (C + 8) * 2 : triIn256Smem<TA>(C, WARPS, XROUNDS);
  static bool attr = false;
  if (!attr) { smemAttr((triIn256K<C, WARPS, TA, XROUNDS>), (int)smem); attr = true; }
  std::string pg = concatColumns("f/" + Tn + "projectionGate~", C, {{"f/" + Tn + "projection", 2 * C, false},
                                                                   {"f/" + Tn + "gate", 2 * C, false}});
  half* wt = scratch<half>("ftri.wt", triInTileHalves(C));
  tileTriIn(Wh(pg), Fh(Tn + "gatingLinear"), C, 16, wt);
  triIn256K<C, WARPS, TA, XROUNDS><<<(unsigned)((pp + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    pair, mask, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"), wt, a, b, t2, n, np, cs);
}
// the 8-warp form where it fits, else the T4's
template <class TA>
void triIn256(const float* pair, const float* mask, const std::string& Tn, TA* a, TA* b, half* t2, int n, int np, size_t cs) {
  if (fused256Big()) triIn256Form<8, TA>(pair, mask, Tn, a, b, t2, n, np, cs);
  else triIn256Form<16, TA, 8>(pair, mask, Tn, a, b, t2, n, np, cs);
}
