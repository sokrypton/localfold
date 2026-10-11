#pragma once
// ESMFold2's trunk across several GPUs (--gpus=N): cuda/af3's ranks and shared buffers (multigpu.cuh) and its sharded
// triangle (sharded.cuh's triangleOn, ESMFold2's weight names through sh::WNAME). A trunk block is two triangle
// multiplications and a transition - no attention - so each rank holds its rows of the pair (bf16, as EF2_Z16 holds it
// whole), the recycle and the streamed z_init are row-local, the transition runs on the rank's own rows, and only the
// triangles exchange. After the last pass the pair goes onto rank 0, which folds on alone as one GPU does (the others
// leave): the trunk is the fold's time and, on one card, its memory.
#include "trunk.cuh"
#include "../../af3/src/multigpu.cuh"
#include "../../af3/src/sharded.cuh"

inline std::string ef2WeightName(const std::string& pre, const char* leaf) { return "f/" + pre + "/" + leaf; }

// the trunk on this rank's rows; rank 0 returns the whole pair (bf16, its own allocation), the other ranks leave
inline float* shardedTrunkEf2(int T, int C, int loops) {
  using B16 = __nv_bfloat16;
  sh::WNAME = ef2WeightName;
  const size_t P = (size_t)T * T;
  int lo, hi; sh::rowsOf(T, mg::RANK, lo, hi);
  const int rows = sh::storedRows(T, mg::RANK);
  const size_t held = (size_t)rows * T;                          // this rank's pairs, from pair lo * T
  mg::Shared& zS = mg::shared("sh.z", std::max<size_t>((size_t)sh::maxStored(T) * T * C * 2, 16));
  B16* zl = (B16*)zS.local;
  B16* base = zl - (size_t)lo * T * C;                           // (indexed by the global pair)
  CK(cudaMemsetAsync(zl, 0, std::max<size_t>(held * C * 2, 1), STREAM));
  float* mask = scratch<float>("trunk.mask", P);
  fillK<<<blocks(P), 256, 0, STREAM>>>(mask, 1.f, P);
  const size_t chunk = std::min<size_t>(std::max<size_t>(held, 1), ((size_t)64 << 20) / (4 * (size_t)C));
  float* xn = scratch<float>("trunk.xn", chunk * C); float* zc = scratch<float>("trunk.zc", chunk * C);
  const ZInitStream& zs = ZINIT_STREAM;
  const int bi = lmPairBlock(T, zs.P);
  float* lm = scratch<float>("zi.lmRows", (size_t)bi * T * zs.P);
  const int blocksN = (int)M.meta("meta/blocks");
  for (int loop = 0; loop < loops; ++loop) {
    // the recycle, f32 a chunk of this rank's rows at a time
    for (size_t r0 = 0; r0 < held; r0 += chunk) {
      size_t r = std::min(chunk, held - r0);
      ef2FromBf16K<<<blocks(r * C), 256, 0, STREAM>>>(zl + r0 * C, zc, r * C);
      layerNorm(zc, xn, r, C, F("recycle/norm/scale"), F("recycle/norm/offset"));
      gemm(xn, F("recycle/projection"), zc, r, C, C);
      ef2ToBf16K<<<blocks(r * C), 256, 0, STREAM>>>(zc, zl + r0 * C, r * C);
    }
    // z_init on this rank's token rows (zInitAddStreamed's, the language model's pair made a block of rows at a time)
    for (int i0 = lo; i0 < lo + rows; i0 += bi) {
      int b = std::min(bi, lo + rows - i0); size_t cells = (size_t)b * T;
      lmPairRows(zs.s, T, zs.P, i0, b, lm);
      zInitAddRowsK<<<blocks(cells * C), 256, 0, STREAM>>>(base, zs.rows, zs.cols, relIdx(), W("token_bonds"),
                                                           F("featuriser/tokenBonds"), lm, T, C, (size_t)i0 * T, cells);
    }
    PAIR16 = true;
    for (int b = 0; b < blocksN; ++b) {
      const std::string B = "blocks/" + std::to_string(b) + "/";
      sh::triangleSharded<half>(zS, mask, T, C, B + "triangleMultiplicationOutgoing", false, false);
      sh::triangleSharded<half>(zS, mask, T, C, B + "triangleMultiplicationIncoming", false, true);
      if (held) transition256(reinterpret_cast<float*>(zl), held, C, B + "pairTransition/");
      mg::fence();                                   // (the next block's triangle reads every rank's rows)
    }
    PAIR16 = false;
    mg::tick("trunk pass");
  }
  // the whole pair onto rank 0 (a copy from each rank's buffer), every shared buffer given back, the others gone
  releaseScratch();
  B16* whole = mg::RANK == 0 ? dallocT<B16>(P * C) : nullptr;
  mg::fence();
  if (mg::RANK == 0)
    for (int r = 0; r < mg::WORLD; ++r) {
      int a, e; sh::rowsOf(T, r, a, e); const int rn = sh::storedRows(T, r);
      if (rn) CK(cudaMemcpyAsync(whole + (size_t)a * T * C, zS.peer[r], (size_t)rn * T * C * 2, cudaMemcpyDefault, STREAM));
    }
  mg::fence();
  mg::release({ "" }, mg::FINAL);
  if (mg::RANK != 0) mg::leave();
  mg::finish();
  mg::WORLD = 1; SHORT_PAIR_SHARE = 1;
  return reinterpret_cast<float*>(whole);
}
