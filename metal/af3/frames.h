// AlphaFold 3's pTM conventions every confidence path here shares - cuda/af3/src/frames.cuh's twin, and the interface
// TM term that cuda/af3's tmChainK computes on the device, taken here on the host off the PAE logits (unified memory:
// the logits are the host's to read). Reported by Yeqing Lin: the chain-level scores were reduced from the GLOBAL TM
// term (d0 from every token) where AF3 reduces chain_pair_iptm - and chain_ptm, its diagonal - from its INTERFACE
// term (confidence_head.py's tmscore_adjusted_pae: d0 from the pair's own chains); and pTM's anchors are AF3's tokens
// with frames (model.py's pae_single_mask).
#pragma once
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// TM-score's d0 for a token count (clipped at 19, as AF3's - below 15 it is undefined)
inline double tmD0(int tokens) { return 1.24 * std::cbrt(std::max(tokens, 19) - 15.0) - 1.8; }

// AF3's interface TM term off PAE logits [n x n, bins]: each pair's softmax weighted by its chain pair's table - d0
// from the chain's tokens on the diagonal, the two chains' summed off it (num_interface_tokens). chainOf: any id a
// token (< 0: none), live its mask; countOf / countLive the tokens d0 counts (OpenDDE: the residues, where the pairs are
// structural); 0 where either token has no chain
inline std::vector<float> interfaceTmTerm(const float* logits, int n, int bins, const float* centres,
                                          const int* chainOf, const float* live, const int* countOf, const float* countLive,
                                          int countN) {
  std::vector<int> ids;
  for (int i = 0; i < countN; ++i)
    if (countLive[i] > 0 && countOf[i] >= 0 && std::find(ids.begin(), ids.end(), countOf[i]) == ids.end()) ids.push_back(countOf[i]);
  std::sort(ids.begin(), ids.end());
  const int nc = (int)ids.size();
  std::vector<int> count(nc, 0), of(n, -1);
  for (int i = 0; i < countN; ++i)
    if (countLive[i] > 0 && countOf[i] >= 0) count[std::lower_bound(ids.begin(), ids.end(), countOf[i]) - ids.begin()]++;
  for (int i = 0; i < n; ++i) {
    if (!(live[i] > 0) || chainOf[i] < 0) continue;
    auto it = std::lower_bound(ids.begin(), ids.end(), chainOf[i]);
    if (it != ids.end() && *it == chainOf[i]) of[i] = (int)(it - ids.begin());
  }
  std::vector<float> tab((size_t)std::max(nc, 1) * std::max(nc, 1) * bins, 0.f);
  for (int a = 0; a < nc; ++a)
    for (int b = 0; b < nc; ++b) {
      const double d0 = tmD0(a == b ? count[a] : count[a] + count[b]);
      for (int k = 0; k < bins; ++k) tab[((size_t)a * nc + b) * bins + k] = (float)(1 / (1 + (double)centres[k] * centres[k] / (d0 * d0)));
    }
  std::vector<float> out((size_t)n * n, 0.f);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const int a = of[i], b = of[j];
      if (a < 0 || b < 0) continue;
      const float* l = logits + ((size_t)i * n + j) * bins;
      const float* t = tab.data() + ((size_t)a * nc + b) * bins;
      float mx = -INFINITY;
      for (int k = 0; k < bins; ++k) mx = std::max(mx, l[k]);
      float total = 0.f, w = 0.f;
      for (int k = 0; k < bins; ++k) { const float e = std::exp(l[k] - mx); total += e; w += e * t[k]; }
      out[(size_t)i * n + j] = w / total;
    }
  return out;
}

// AlphaFold 3's token frames' validity (features.py, Frames.compute_features), pTM's anchors: every polymer token has
// a frame; a ligand token one only where its residue has three tokens or more and the token with its two nearest (by
// the reference conformer) is not within 25 degrees of colinear (1 - |cos| < 0.0937, or two at one place)
inline std::vector<char> tokenFrameMask(int n) {
  std::vector<char> ok(n, 1);
  if (!M.has("batch.isLigand") || !M.has("batch.refPos")) return ok;
  const int* lig = M.hostI("batch.isLigand"); const int* asym = M.hostI("batch.asymId"); const int* res = M.hostI("batch.residueIndex");
  const float* pos = M.hostF("batch.refPos"); const float* rm = M.hostF("batch.refMask");
  const int dense = (int)M.meta("batch.dense");
  auto at = [&](int t, double* x) {   // (a ligand token's one atom: its first live slot)
    for (int s = 0; s < dense; ++s)
      if (rm[(size_t)t * dense + s] > 0) { for (int d = 0; d < 3; ++d) x[d] = pos[((size_t)t * dense + s) * 3 + d]; return true; }
    return false;
  };
  for (int t = 0; t < n; ++t) {
    if (!lig[t]) continue;
    std::vector<int> local;
    for (int u = 0; u < n; ++u) if (lig[u] && asym[u] == asym[t] && res[u] == res[t]) local.push_back(u);
    if (local.size() < 3) { ok[t] = 0; continue; }
    double xt[3] = {0, 0, 0}; const bool liveT = at(t, xt);
    std::vector<std::pair<double, int>> cost;
    for (int u : local) {
      double xu[3] = {0, 0, 0}; const bool live = at(u, xu);
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
    if (n1 < 1e-8 || n2 < 1e-8) { ok[t] = 0; continue; }
    for (int d = 0; d < 3; ++d) dot += v1[d] / n1 * v2[d] / n2;
    ok[t] = !(1 - std::fabs(dot) < 0.0937);
  }
  return ok;
}
