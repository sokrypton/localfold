// AlphaFold 3's pTM conventions every confidence path here shares: TM-score's d0, and the token frames' validity that
// keeps an ion's row from being a pTM anchor (and hostSeqMask, the host's view of the batch's sequence mask)
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "common.cuh"
#include "../../featurise/ptm.h"

// (the definitions are cuda/featurise/ptm.h's, every port's; this reads the batch for them)
inline double tmD0(int tokens) { return ptm::d0(tokens); }
// AlphaFold 3's token frames' validity, the pTM anchors' mask (ptm::frameMask), off the batch: a ligand token's
// position its first live dense slot's reference coordinate
inline std::vector<char> tokenFrameMask(int n) {
  if (!M.has("batch.isLigand") || !M.has("batch.refPos")) return std::vector<char>(n, 1);
  const float* pos = M.f("batch.refPos"); const float* rm = M.f("batch.refMask");
  const int dense = (int)M.meta("batch.dense");
  std::vector<float> at((size_t)n * 3, 0.f); std::vector<char> live(n, 0);
  for (int t = 0; t < n; ++t)
    for (int s = 0; s < dense; ++s)
      if (rm[(size_t)t * dense + s] > 0) { for (int d = 0; d < 3; ++d) at[(size_t)t * 3 + d] = pos[((size_t)t * dense + s) * 3 + d]; live[t] = 1; break; }
  return ptm::frameMask(n, M.i("batch.isLigand"), M.i("batch.asymId"), M.i("batch.residueIndex"), at.data(), live.data());
}
inline const float* hostSeqMask(const float* seqMask, int n, std::vector<float>& buf) {
  if (M.has("batch.seqMask") && M.len("batch.seqMask") == (size_t)n && seqMask == Fdev("batch.seqMask"))
    return M.f("batch.seqMask");
  buf = download(seqMask, n);
  return buf.data();
}
