// AlphaFold 3's pTM conventions every confidence path here shares: TM-score's d0, and the token frames' validity that
// keeps an ion's row from being a pTM anchor (and hostSeqMask, the host's view of the batch's sequence mask)
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "common.cuh"

// TM-score's d0 for a token count (clipped at 19, as AF3's - below 15 it is undefined)
inline double tmD0(int tokens) { return 1.24 * std::cbrt(std::max(tokens, 19) - 15.0) - 1.8; }
// AlphaFold 3's token frames' validity (features.py, Frames.compute_features), which the pTM scores take as their
// anchors' mask (model.py: pae_single_mask, the rows alone): every polymer token has a frame; a ligand token one only
// where its residue has three tokens or more and the token with its two nearest (by the reference conformer) is not
// colinear (within 25 degrees, 1 - |cos| < 0.0937, or two at one place). An ion, a two-atom ligand: no anchor
inline std::vector<char> tokenFrameMask(int n) {
  std::vector<char> ok(n, 1);
  if (!M.has("batch.isLigand") || !M.has("batch.refPos")) return ok;
  const int* lig = M.i("batch.isLigand"); const int* asym = M.i("batch.asymId"); const int* res = M.i("batch.residueIndex");
  const float* pos = M.f("batch.refPos"); const float* rm = M.f("batch.refMask");
  const int dense = (int)M.meta("batch.dense");
  auto at = [&](int t, double* x) {   // (a ligand token's one atom: its first live slot)
    for (int s = 0; s < dense; ++s) if (rm[(size_t)t * dense + s] > 0) { for (int d = 0; d < 3; ++d) x[d] = pos[((size_t)t * dense + s) * 3 + d]; return true; }
    return false;
  };
  for (int t = 0; t < n; ++t) {
    if (!lig[t]) continue;
    std::vector<int> local;
    for (int u = 0; u < n; ++u) if (lig[u] && asym[u] == asym[t] && res[u] == res[t]) local.push_back(u);
    if (local.size() < 3) { ok[t] = 0; continue; }
    double xt[3]; const bool liveT = at(t, xt);
    // the two nearest of the residue's other live tokens (a dead one, or the token itself, at +1e8 as AF3 costs them)
    std::vector<std::pair<double, int>> cost;
    for (int u : local) {
      double xu[3]; const bool live = at(u, xu);
      double d = live && liveT ? std::sqrt((xu[0] - xt[0]) * (xu[0] - xt[0]) + (xu[1] - xt[1]) * (xu[1] - xt[1]) + (xu[2] - xt[2]) * (xu[2] - xt[2])) : 0.0;
      if (!live) d += 1e8;
      if (u == t) d += 1e8;
      cost.push_back({d, u});
    }
    std::stable_sort(cost.begin(), cost.end(), [](auto& x, auto& y) { return x.first < y.first; });
    double xa[3] = {0, 0, 0}, xc[3] = {0, 0, 0};
    at(cost[0].second, xa); at(cost[1].second, xc);
    double v1[3], v2[3], n1 = 0, n2 = 0, dot = 0;
    for (int d = 0; d < 3; ++d) { v1[d] = xa[d] - xt[d]; v2[d] = xc[d] - xt[d]; n1 += v1[d] * v1[d]; n2 += v2[d] * v2[d]; }
    n1 = std::sqrt(n1); n2 = std::sqrt(n2);
    if (n1 < 1e-8 || n2 < 1e-8) { ok[t] = 0; continue; }   // (np.isclose(norm, 0): identical coordinates, colinear)
    for (int d = 0; d < 3; ++d) dot += v1[d] / n1 * v2[d] / n2;
    ok[t] = !(1 - std::fabs(dot) < 0.0937);
  }
  return ok;
}
inline const float* hostSeqMask(const float* seqMask, int n, std::vector<float>& buf) {
  if (M.has("batch.seqMask") && M.len("batch.seqMask") == (size_t)n && seqMask == Fdev("batch.seqMask"))
    return M.f("batch.seqMask");
  buf = download(seqMask, n);
  return buf.data();
}
