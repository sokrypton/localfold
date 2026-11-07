// AlphaFold 3's pTM conventions, read off this port's batch - the definitions are cuda/featurise/ptm.h's, every port's
// (cuda/af3/src/frames.cuh is the CUDA twin). Reported by Yeqing Lin: the chain-level scores were reduced from the
// GLOBAL TM term (d0 from every token) where AF3 reduces chain_pair_iptm - and chain_ptm, its diagonal - from its
// INTERFACE term (d0 from the pair's own chains); and pTM's anchors are AF3's tokens with frames (pae_single_mask).
// The interface term is taken here on the host off the PAE logits (unified memory: the logits are the host's to read).
#pragma once
#include <vector>
#include "ptm.h"

inline double tmD0(int tokens) { return ptm::d0(tokens); }

// the interface term off PAE logits [n x n, bins]: chainOf any id a token (< 0: none), live its mask; countOf /
// countLive the tokens d0 counts (OpenDDE: the residues, where the pairs are structural); 0 where either has no chain
inline std::vector<float> interfaceTmTerm(const float* logits, int n, int bins, const float* centres,
                                          const int* chainOf, const float* live, const int* countOf, const float* countLive,
                                          int countN) {
  const ptm::Layout L = ptm::layout(chainOf, live, n, countOf, countLive, countN);
  return ptm::interfaceTerm(logits, n, bins, L, ptm::tables(L, centres, bins));
}

// AlphaFold 3's token frames' validity (ptm::frameMask), the pTM anchors, off the batch: a ligand token's position its
// first live dense slot's reference coordinate
inline std::vector<char> tokenFrameMask(int n) {
  if (!M.has("batch.isLigand") || !M.has("batch.refPos")) return std::vector<char>(n, 1);
  const float* pos = M.hostF("batch.refPos"); const float* rm = M.hostF("batch.refMask");
  const int dense = (int)M.meta("batch.dense");
  std::vector<float> at((size_t)n * 3, 0.f); std::vector<char> live(n, 0);
  for (int t = 0; t < n; ++t)
    for (int s = 0; s < dense; ++s)
      if (rm[(size_t)t * dense + s] > 0) { for (int d = 0; d < 3; ++d) at[(size_t)t * 3 + d] = pos[((size_t)t * dense + s) * 3 + d]; live[t] = 1; break; }
  return ptm::frameMask(n, M.hostI("batch.isLigand"), M.hostI("batch.asymId"), M.hostI("batch.residueIndex"), at.data(), live.data());
}
