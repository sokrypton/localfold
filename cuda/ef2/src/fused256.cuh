// ESMFold2's launchers for the 256-channel fused kernels (cuda/af3/src/fused256.cuh, shared with the AF3
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
// ...or their T4 forms (cuda/af3's rounds: the input kernel and the transition at 16 warps, their rows normed
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
                 int L, int Lp, RectMap rm = {}) {
  triangleOutRun<256, WARPS, TP>(prod, sc, of, Wout, t2, pair, L, Lp, nullptr, rm);
}
template <int WARPS, class TA, int XROUNDS = 1, int MT = 1>
void triIn256Form(const float* pair, const float* mask, const std::string& Tn, TA* a, TA* b, half* t2, int n, int np, size_t cs,
                  RectMap rm = {}) {
  constexpr int C = 256, R = 16 * WARPS * MT;
  size_t pp = rm.J ? rm.size : (size_t)np * np, smem = XROUNDS == 1 ? std::max((size_t)R * (C + 8) * 2, triIn256Smem<TA>(C, WARPS * MT)) : triIn256Smem<TA>(C, WARPS, XROUNDS);
  std::string pg = concatColumns("f/" + Tn + "projectionGate~", C, {{"f/" + Tn + "projection", 2 * C, false},
                                                                   {"f/" + Tn + "gate", 2 * C, false}});
  half* wt = scratch<half>("ftri.wt", triInTileHalves(C));
  tileTriIn(Wh(pg), Fh(Tn + "gatingLinear"), C, 16, wt);
  WITH_PAIR_T(                    // (the pair f32, or bf16 under PAIR16: see trunk.cuh's EF2_P16)
    static bool attr = false;
    if (!attr) { smemAttr((triIn256K<C, WARPS, TA, XROUNDS, false, PT, MT>), (int)smem); attr = true; }
    triIn256K<C, WARPS, TA, XROUNDS, false, PT, MT><<<(unsigned)((pp + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
      pair, mask, F(Tn + "leftNormInputScale"), F(Tn + "leftNormInputOffset"), wt, a, b, t2, n, np, cs, nullptr, rm));
}
// the 8-warp form where it fits, else the T4's - and where an SM holds two blocks of it, 4 warps of two 16-row tiles
// each (the same 128 rows, shared memory and grid): each weight fragment then feeds two warps' worth of MMAs, the
// kernel being shared-memory bound (Nsight Compute at 988 tokens: the shared wavefronts 68% -> 40% of peak).
// triIn256K 742 -> 714 ms of a 988-token fold, 185.7 -> 178.0 at 494, 61.3 -> 59.8 at 261, flat at 195;
// byte-identical (each output's k order unchanged). 8 warps of two tiles - one block an SM - lost (786): nothing
// then overlaps a block's LN prologue. A part holding one block (an L4, an RTX card: ~100 KB an SM) would lose
// half its warps, unmeasured, so it keeps the one-tile form. LOCALFOLD_TRIIN_FORM=1/2 forces it.
template <class TA>
void triIn256(const float* pair, const float* mask, const std::string& Tn, TA* a, TA* b, half* t2, int n, int np, size_t cs,
              RectMap rm = {}) {
  static const bool twoTiles = [] {
    if (const char* e = getenv("LOCALFOLD_TRIIN_FORM")) return atoi(e) == 2;
    int dev, perSm = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    return (size_t)perSm >= 2 * (TRI_IN256_SMEM + 1024);          // (1 KB a block the driver reserves)
  }();
  // ...and where an SM holds ONE 8-warp block (~100 KB: an RTX PRO 6000, an L4), the T4's 16-warp form, its rows normed
  // in rounds, which fits two: measured on Colab's RTX PRO 6000 - 457 -> 428 ms of ESMFold2's 988-token fold, 4490 ->
  // 3882 at 2,964 tokens (the 8-warp form at one block an SM, Nsight Compute: the tensor pipe at 60%); byte-identical.
  // LOCALFOLD_TRIIN_FORM=1 forces the 8-warp form, 2 the two-tile one
  static const bool form1 = getenv("LOCALFOLD_TRIIN_FORM") && atoi(getenv("LOCALFOLD_TRIIN_FORM")) == 1;
  if (fused256Big() && twoTiles) triIn256Form<4, TA, 1, 2>(pair, mask, Tn, a, b, t2, n, np, cs, rm);
  else if (fused256Big() && form1) triIn256Form<8, TA>(pair, mask, Tn, a, b, t2, n, np, cs, rm);
  else triIn256Form<16, TA, 8>(pair, mask, Tn, a, b, t2, n, np, cs, rm);
}
