// The pTM family as AlphaFold 3 defines it - ONE definition for every native port (cuda/af3, cuda/ef2, metal/af3,
// metal/ef2, ...), so a chain's pTM means the same thing whichever model and backend wrote it. Host C++ over plain
// arrays: nothing here knows a model, a bundle or a device. shared/heads/tm-score.js is the page's twin.
//
// Reported by Yeqing Lin (2026-11-07): the chain-level scores were reduced from the GLOBAL TM term, d0 from every token,
// so a short binder in a large complex read ~0.10 high. AF3 (af3-any-model: confidence_head.py, confidences.py,
// model.py) takes
//   * pTM / ipTM off the global term (d0 from every token), anchored on the tokens WITH FRAMES (pae_single_mask);
//   * chain_pair_iptm off the INTERFACE term (tmscore_adjusted_pae with num_interface_tokens: d0 from the chain's own
//     tokens on the diagonal, the two chains' summed off it), the same anchors; chain_ptm its diagonal;
//   * chain_iptm as iptm_xchain: the mean of a chain's row of chain_pair_iptm off the diagonal.
// A port whose model's vendor defines pTM / ipTM otherwise (ESMFold2: every token an anchor) keeps the vendor's two
// global numbers; the chain-level fields - AF3's names - are AF3's everywhere.
#pragma once
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace ptm {

// TM-score's d0 for a token count (clipped at 19, as AF3's - below 15 it is undefined)
inline double d0(int tokens) { return 1.24 * std::cbrt(std::max(tokens, 19) - 15.0) - 1.8; }

// the chains of a fold: their ids sorted, each token's dense index among them (-1: no chain, or not live), and how many
// tokens each holds - counted over `countOf`/`countLive` where d0 counts other tokens than the ones indexed (OpenDDE:
// the scores' pairs are structural tokens, its d0 the residues'); nullptr: the same tokens
struct Layout { std::vector<int> ids, of, count; int nc() const { return (int)ids.size(); } };
inline Layout layout(const int* chainOf, const float* live, int n, const int* countOf = nullptr,
                     const float* countLive = nullptr, int countN = -1) {
  if (!countOf) { countOf = chainOf; countLive = live; countN = n; }
  auto on = [](const float* m, int i) { return !m || m[i] > 0; };
  Layout L;
  for (int i = 0; i < countN; ++i)
    if (on(countLive, i) && countOf[i] >= 0 && std::find(L.ids.begin(), L.ids.end(), countOf[i]) == L.ids.end()) L.ids.push_back(countOf[i]);
  std::sort(L.ids.begin(), L.ids.end());
  L.count.assign(L.ids.size(), 0);
  for (int i = 0; i < countN; ++i)
    if (on(countLive, i) && countOf[i] >= 0) L.count[std::lower_bound(L.ids.begin(), L.ids.end(), countOf[i]) - L.ids.begin()]++;
  L.of.assign(n, -1);
  for (int i = 0; i < n; ++i) {
    if (!on(live, i) || chainOf[i] < 0) continue;
    auto it = std::lower_bound(L.ids.begin(), L.ids.end(), chainOf[i]);
    if (it != L.ids.end() && *it == chainOf[i]) L.of[i] = (int)(it - L.ids.begin());
  }
  return L;
}

// the interface term's weights [nc][nc][bins]: 1 / (1 + centre^2 / d0^2) at d0 of the chain pair's tokens
inline std::vector<float> tables(const Layout& L, const float* centres, int bins) {
  const int nc = L.nc();
  std::vector<float> tab((size_t)std::max(nc, 1) * std::max(nc, 1) * bins, 0.f);
  for (int a = 0; a < nc; ++a)
    for (int b = 0; b < nc; ++b) {
      const double d = d0(a == b ? L.count[a] : L.count[a] + L.count[b]);
      for (int k = 0; k < bins; ++k) tab[((size_t)a * nc + b) * bins + k] = (float)(1 / (1 + (double)centres[k] * centres[k] / (d * d)));
    }
  return tab;
}

// the interface term on the host off PAE logits [n x n, bins] (each pair's softmax against its chain pair's table);
// 0 where either token has no chain. The devices' kernels compute the same (cuda/af3 tmChainK, cuda/ef2 tmInterfaceK)
inline std::vector<float> interfaceTerm(const float* logits, int n, int bins, const Layout& L, const std::vector<float>& tab) {
  const int nc = L.nc();
  std::vector<float> out((size_t)n * n, 0.f);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const int a = L.of[i], b = L.of[j];
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

// AlphaFold 3's token frames' validity (features.py, Frames.compute_features), the pTM anchors' mask: every polymer
// token has a frame; a ligand token one only where its residue (asym, residue index) has three tokens or more and the
// token with its two nearest (by the reference conformer) is not within 25 degrees of colinear (1 - |cos| < 0.0937, or
// two at one place). pos [n][3]: a token's reference position, live whether it has one (a ligand token is one atom)
inline std::vector<char> frameMask(int n, const int* isLigand, const int* asym, const int* res, const float* pos, const char* live) {
  std::vector<char> ok(n, 1);
  for (int t = 0; t < n; ++t) {
    if (!isLigand[t]) continue;
    std::vector<int> local;
    for (int u = 0; u < n; ++u) if (isLigand[u] && asym[u] == asym[t] && res[u] == res[t]) local.push_back(u);
    if (local.size() < 3) { ok[t] = 0; continue; }
    auto at = [&](int u, double* x) { for (int d = 0; d < 3; ++d) x[d] = live[u] ? pos[(size_t)u * 3 + d] : 0.0; return (bool)live[u]; };
    double xt[3]; const bool liveT = at(t, xt);
    // the two nearest of the residue's other live tokens (a dead one, or the token itself, at +1e8 as AF3 costs them)
    std::vector<std::pair<double, int>> cost;
    for (int u : local) {
      double xu[3]; const bool lu = at(u, xu);
      double d = lu && liveT ? std::sqrt((xu[0] - xt[0]) * (xu[0] - xt[0]) + (xu[1] - xt[1]) * (xu[1] - xt[1]) + (xu[2] - xt[2]) * (xu[2] - xt[2])) : 0.0;
      if (!lu) d += 1e8;
      if (u == t) d += 1e8;
      cost.push_back({d, u});
    }
    std::stable_sort(cost.begin(), cost.end(), [](auto& x, auto& y) { return x.first < y.first; });
    double xa[3], xc[3];
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

// The chain-level scores, in L's chain order: pair [nc][nc] (chain_pair_iptm: on the diagonal the chain's pTM, off it
// the pair's ipTM - anchors in either chain over the other), ptm its diagonal (chain_ptm), iptm (chain_iptm,
// iptm_xchain) the mean of a chain's row off the diagonal; NaN where nothing is selected. Every one is a max over
// anchors i of the mean of the term over the j it selects, so each anchor's row summed per chain - rowSum [n][nc], in
// j order - answers them all (a device can hand back just that: cuda/ef2's tmChainRowsK). frame: the anchors (nullptr:
// every live token)
struct Chains { std::vector<double> pair, ptm, iptm; };
inline Chains reduceRows(const double* rowSum, int n, const Layout& L, const char* frame) {
  const int nc = L.nc();
  auto best = [&](int from, int to) {   // anchors of chain `from`, their mean over chain `to`'s tokens
    double top = -1e30; bool any = false;
    for (int i = 0; i < n; ++i) {
      if (L.of[i] != from || (frame && !frame[i]) || !L.count[to]) continue;
      any = true; top = std::max(top, rowSum[(size_t)i * nc + to] / L.count[to]);
    }
    return any ? top : NAN;
  };
  auto maxNan = [](double x, double y) { return std::isnan(x) ? y : std::isnan(y) ? x : std::max(x, y); };
  Chains c;
  c.pair.assign((size_t)nc * nc, NAN); c.ptm.assign(nc, NAN); c.iptm.assign(nc, NAN);
  for (int a = 0; a < nc; ++a)
    for (int b = 0; b < nc; ++b) c.pair[(size_t)a * nc + b] = a == b ? best(a, a) : maxNan(best(a, b), best(b, a));
  for (int a = 0; a < nc; ++a) {
    c.ptm[a] = c.pair[(size_t)a * nc + a];
    double sum = 0; int cnt = 0;
    for (int b = 0; b < nc; ++b) if (b != a && std::isfinite(c.pair[(size_t)a * nc + b])) { sum += c.pair[(size_t)a * nc + b]; ++cnt; }
    c.iptm[a] = cnt ? sum / cnt : NAN;
  }
  return c;
}
// ...off the whole interface term [n x n]
inline std::vector<double> rowSums(const float* term, int n, const Layout& L) {
  const int nc = L.nc();
  std::vector<double> rows((size_t)n * nc, 0.0);
  for (int i = 0; i < n; ++i) {
    if (L.of[i] < 0) continue;
    double* r = rows.data() + (size_t)i * nc;
    for (int j = 0; j < n; ++j) if (L.of[j] >= 0) r[L.of[j]] += term[(size_t)i * n + j];
  }
  return rows;
}
inline Chains chainScores(const float* term, int n, const Layout& L, const char* frame) {
  const std::vector<double> rows = rowSums(term, n, L);
  return reduceRows(rows.data(), n, L, frame);
}

}  // namespace ptm
