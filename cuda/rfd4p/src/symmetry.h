#pragma once
// Chemically equivalent atoms and identical subunits, resolved as RFD4-Proteina's
// `StructureComparisonMetric` resolves them before a motif RMSD: the C++ twin of
// shared/rfd4p/symmetry.js, which has the notes (atomworks' automorphism units, RDKit's
// conjugation and `fix_mol`'s charge repair behind the bond orders, the 1000-automorphism cap).
// Every loop runs in the page's order, so a tie breaks the same way and gate.py can hold the two
// metrics to the same bits.
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "superpose.h"

namespace rfd4p {
namespace symmetry {

// biotite's `BondType`, as shared/rfd4p/pipeline/cif.js numbers it.
enum BondType { ANY = 0, SINGLE = 1, DOUBLE = 2, TRIPLE = 3, QUADRUPLE = 4, AROMATIC_SINGLE = 5,
                AROMATIC_DOUBLE = 6, AROMATIC_TRIPLE = 7, COORDINATION = 8, AROMATIC = 9 };
constexpr double RESONANT = 1.5;

struct Bond { int i, j, type; };

/** A structure the searches read: an array slice's atoms and its bonds, re-indexed. */
struct View {
  int n = 0;
  std::vector<std::string> element, chainId, chainIid, insCode, resName, atomName, pnUnit;
  std::vector<int> charge, resId, molIid, molEntity;
  std::vector<uint8_t> isPolymer;
  std::vector<Bond> bonds;
};

inline bool aromaticType(int t) {
  return t == AROMATIC || t == AROMATIC_SINGLE || t == AROMATIC_DOUBLE || t == AROMATIC_TRIPLE;
}
inline double orderByType(int t, double fallback) {
  switch (t) {
    case SINGLE: return 1;
    case DOUBLE: return 2;
    case TRIPLE: return 3;
    case QUADRUPLE: return 4;
    default: return fallback;
  }
}
/** `_bond_order`. */
inline double bondOrder(int t) { return aromaticType(t) ? RESONANT : orderByType(t, -1.0 - t); }

struct Periodic { int z, dv; std::vector<int> valences; int nouter; };
inline Periodic periodic(const std::string& el) {
  static const std::map<std::string, Periodic> table = {
      {"H", {1, 1, {1}, 1}},       {"B", {5, 3, {3}, 3}},       {"C", {6, 4, {4}, 4}},
      {"N", {7, 3, {3}, 5}},       {"O", {8, 2, {2}, 6}},       {"F", {9, 1, {1}, 7}},
      {"NA", {11, 1, {1, -1}, 1}}, {"MG", {12, 2, {2, -1}, 2}}, {"AL", {13, 3, {3}, 3}},
      {"SI", {14, 4, {4}, 4}},     {"P", {15, 3, {3, 5}, 5}},   {"S", {16, 2, {2, 4, 6}, 6}},
      {"CL", {17, 1, {1}, 7}},     {"K", {19, 1, {1, -1}, 1}},  {"CA", {20, 2, {2, -1}, 2}},
      {"AS", {33, 3, {3, 5}, 5}},  {"SE", {34, 2, {2, 4, 6}, 6}}, {"BR", {35, 1, {1}, 7}},
      {"I", {53, 1, {1, 3, 5}, 7}}};
  const auto it = table.find(el);
  return it == table.end() ? Periodic{0, -1, {-1}, 0} : it->second;
}
inline bool earlyAtom(const std::string& el) {
  return el == "LI" || el == "BE" || el == "B" || el == "NA" || el == "MG" || el == "AL" ||
         el == "K" || el == "CA";
}

/** `conjugatedBonds` (symmetry.js): RDKit's conjugation over a unit's Kekulé bonds. */
inline std::vector<uint8_t> conjugatedBonds(const std::vector<std::string>& element,
                                            std::vector<int> charge, const std::vector<Bond>& bonds) {
  const int n = (int)element.size();
  std::vector<double> contrib(bonds.size());
  for (size_t k = 0; k < bonds.size(); ++k) {
    const int t = bonds[k].type;
    contrib[k] = t == AROMATIC ? 1.5 : t == AROMATIC_SINGLE ? 1 : t == AROMATIC_DOUBLE ? 2
               : t == AROMATIC_TRIPLE ? 3 : t == COORDINATION ? 0 : orderByType(t, 1);
  }
  std::vector<std::vector<int>> incident(n);
  for (size_t k = 0; k < bonds.size(); ++k) {
    incident[bonds[k].i].push_back((int)k);
    incident[bonds[k].j].push_back((int)k);
  }
  std::vector<int> degree(n), explicitV(n);
  for (int i = 0; i < n; ++i) {
    double s = 0;
    for (const int k : incident[i]) {
      if (bonds[k].type != COORDINATION) ++degree[i];
      s += contrib[k];
    }
    explicitV[i] = (int)std::round(s + 0.1);
  }
  auto implicit = [&](int i, int chgIn) {
    const Periodic p = periodic(element[i]);
    if (p.dv == -1 || p.valences.back() == -1) return 0;
    int chg = chgIn;
    if (earlyAtom(element[i])) chg = -chg;
    if (p.z == 6 && chg > 0) chg = -chg;
    for (const int v : p.valences) if (explicitV[i] <= v + chg) return v + chg - explicitV[i];
    return -1;
  };
  std::vector<int> violators;
  for (int i = 0; i < n; ++i) if (implicit(i, charge[i]) < 0) violators.push_back(i);
  if (!violators.empty()) {
    std::vector<int> fixed = charge;
    for (const int i : violators) fixed[i] = explicitV[i] - periodic(element[i]).dv;
    for (int i = 0; i < n; ++i) if (implicit(i, fixed[i]) < 0) return std::vector<uint8_t>(bonds.size(), 0);
    charge = fixed;
  }
  std::vector<int> hydrogens(n);
  for (int i = 0; i < n; ++i) hydrogens[i] = implicit(i, charge[i]);
  auto electrons = [&](int i) {
    const Periodic p = periodic(element[i]);
    if (p.dv <= 1) return -1;
    const int deg = degree[i] + hydrogens[i];
    if (deg > 3) return -1;
    const int nlp = std::max(p.nouter - p.dv - charge[i], 0);
    int res = (p.dv - deg) + nlp;
    if (res > 1 && explicitV[i] - degree[i] > 1) res = 1;
    return res;
  };
  auto candidate = [&](int i) {
    const Periodic p = periodic(element[i]);
    return (p.z <= 10 || (p.nouter != 5 && p.nouter != 6) || (p.nouter == 6 && p.z < 52)) && electrons(i) > 0;
  };
  std::vector<uint8_t> conj(bonds.size());
  for (size_t k = 0; k < bonds.size(); ++k) conj[k] = aromaticType(bonds[k].type);
  for (int i = 0; i < n; ++i) {
    if (!candidate(i)) continue;
    const int sbo = degree[i] + hydrogens[i];
    if (sbo < 2 || sbo > 3) continue;
    for (const int b1 : incident[i]) {
      if (contrib[b1] < 1.5) continue;
      for (const int b2 : incident[i]) {
        if (b1 == b2) continue;
        const int other = bonds[b2].i == i ? bonds[b2].j : bonds[b2].i;
        if (degree[other] + hydrogens[other] > 3) continue;
        if (candidate(other)) { conj[b1] = 1; conj[b2] = 1; }
      }
    }
  }
  return conj;
}

/** `_resonance_normalized_bond_orders`. */
inline std::vector<double> normalizedBondOrders(const std::vector<std::string>& element,
                                                const std::vector<int>& charge,
                                                const std::vector<Bond>& bonds,
                                                const std::vector<int>& nHeavy) {
  std::vector<double> raw(bonds.size()), orders(bonds.size());
  for (size_t k = 0; k < bonds.size(); ++k) raw[k] = orders[k] = bondOrder(bonds[k].type);
  const std::vector<uint8_t> conj = conjugatedBonds(element, charge, bonds);
  for (size_t k = 0; k < bonds.size(); ++k) {
    if (conj[k] && (raw[k] == 1 || raw[k] == 2 || aromaticType(bonds[k].type))) orders[k] = RESONANT;
  }
  std::map<int, std::vector<std::pair<int, int>>> neighbours;
  for (size_t k = 0; k < bonds.size(); ++k) {
    neighbours[bonds[k].i].push_back({bonds[k].j, (int)k});
    neighbours[bonds[k].j].push_back({bonds[k].i, (int)k});
  }
  for (const auto& [centre, adjacent] : neighbours) {
    (void)centre;
    std::map<std::string, std::vector<int>> byElement;
    for (const auto& [nb, k] : adjacent) if (nHeavy[nb] == 1) byElement[element[nb]].push_back(k);
    for (const auto& [el, ks] : byElement) {
      bool triple = false;
      for (const int k : ks) triple = triple || raw[k] == 3;
      if (ks.size() < 2 || el == "C" || triple) continue;
      for (const int k : ks) orders[k] = RESONANT;
    }
  }
  return orders;
}

/** `automorphisms` (symmetry.js): identity first, at most `max` mappings examined. */
inline std::vector<std::vector<int>> automorphisms(const std::vector<std::string>& colour,
                                                   const std::vector<Bond>& edges,
                                                   const std::vector<double>& labels, int max = 1000) {
  const int n = (int)colour.size();
  const double NONE = 1e300;
  std::vector<double> label((size_t)n * n, NONE);
  std::vector<std::vector<int>> adj(n);
  for (size_t k = 0; k < edges.size(); ++k) {
    label[(size_t)edges[k].i * n + edges[k].j] = labels[k];
    label[(size_t)edges[k].j * n + edges[k].i] = labels[k];
    adj[edges[k].i].push_back(edges[k].j);
    adj[edges[k].j].push_back(edges[k].i);
  }
  std::vector<std::string> cls = colour;
  auto distinct = [](const std::vector<std::string>& v) { return std::set<std::string>(v.begin(), v.end()).size(); };
  for (int round = 0; round < n; ++round) {
    std::vector<std::string> sig(n);
    for (int i = 0; i < n; ++i) {
      std::vector<std::string> parts;
      for (const int j : adj[i]) {
        char buf[64];
        snprintf(buf, sizeof buf, "~%.17g", label[(size_t)i * n + j]);
        parts.push_back(cls[j] + buf);
      }
      std::sort(parts.begin(), parts.end());
      std::string s = cls[i] + "|";
      for (size_t k = 0; k < parts.size(); ++k) s += (k ? "," : "") + parts[k];
      sig[i] = s;
    }
    std::map<std::string, std::string> ids;
    std::vector<std::string> next(n);
    for (int i = 0; i < n; ++i) {
      auto it = ids.find(sig[i]);
      if (it == ids.end()) it = ids.emplace(sig[i], std::to_string(ids.size())).first;
      next[i] = it->second;
    }
    const size_t before = distinct(cls);
    cls = next;
    if (distinct(cls) == before) break;
  }
  std::vector<int> order;
  std::vector<uint8_t> seen(n, 0);
  for (int s = 0; s < n; ++s) {
    if (seen[s]) continue;
    seen[s] = 1;
    std::vector<int> queue = {s};
    for (size_t q = 0; q < queue.size(); ++q) {
      const int v = queue[q];
      order.push_back(v);
      for (const int w : adj[v]) if (!seen[w]) { seen[w] = 1; queue.push_back(w); }
    }
  }
  std::vector<std::vector<int>> out;
  std::vector<int> map(n, -1);
  std::vector<uint8_t> used(n, 0);
  int examined = 0;
  std::vector<int> identity(n);
  for (int i = 0; i < n; ++i) identity[i] = i;
  std::function<void(int)> search = [&](int depth) {
    if (examined >= max) return;
    if (depth == n) {
      ++examined;
      if (map != identity) out.push_back(map);
      return;
    }
    const int v = order[depth];
    for (int t = -1; t < n && examined < max; ++t) {
      const int w = t < 0 ? v : t;
      if ((t >= 0 && t == v) || used[w] || cls[w] != cls[v]) continue;
      bool ok = true;
      for (int d = 0; d < depth && ok; ++d) {
        const int u = order[d];
        ok = label[(size_t)v * n + u] == label[(size_t)w * n + map[u]];
      }
      if (!ok) continue;
      map[v] = w;
      used[w] = 1;
      search(depth + 1);
      used[w] = 0;
      map[v] = -1;
    }
  };
  search(0);
  out.insert(out.begin(), identity);
  return out;
}

/** `count_heavy_neighbors`. */
inline std::vector<int> heavyNeighbours(const View& v) {
  std::vector<int> c(v.n, 0);
  auto hydrogen = [](const std::string& e) { return e == "H" || e == "D" || e == "T"; };
  for (const Bond& b : v.bonds) {
    if (hydrogen(v.element[b.i]) || hydrogen(v.element[b.j])) continue;
    ++c[b.i];
    ++c[b.j];
  }
  return c;
}

inline std::vector<int> residueIndex(const View& v) {
  std::vector<int> idx(v.n, 0);
  int r = 0;
  for (int i = 1; i < v.n; ++i) {
    if (v.chainId[i] != v.chainId[i - 1] || v.resId[i] != v.resId[i - 1] || v.insCode[i] != v.insCode[i - 1]) ++r;
    idx[i] = r;
  }
  return idx;
}

/** `find_automorphisms_with_networkx`: per unit, rows of global atoms (row 0 the identity). */
inline std::vector<std::vector<std::vector<int>>> automorphismUnits(const View& v, int max = 1000) {
  const int n = v.n;
  std::vector<std::vector<std::vector<int>>> out;
  if (n == 0) return out;
  const std::vector<int> heavy = heavyNeighbours(v);
  const std::vector<int> res = residueIndex(v);
  std::set<std::string> nonPolymer;
  for (int i = 0; i < n; ++i) if (!v.isPolymer[i]) nonPolymer.insert(v.pnUnit[i]);
  std::vector<std::string> pnIds(v.pnUnit.begin(), v.pnUnit.end());
  std::sort(pnIds.begin(), pnIds.end());
  pnIds.erase(std::unique(pnIds.begin(), pnIds.end()), pnIds.end());
  std::vector<long> unitOf(n);
  for (int i = 0; i < n; ++i) {
    const long pn = std::lower_bound(pnIds.begin(), pnIds.end(), v.pnUnit[i]) - pnIds.begin();
    unitOf[i] = nonPolymer.count(v.pnUnit[i]) ? (long)n + 1 + pn : res[i];
  }
  std::vector<int> sorted(n);
  for (int i = 0; i < n; ++i) sorted[i] = i;
  std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) { return unitOf[a] < unitOf[b]; });
  std::vector<std::vector<int>> units;
  for (size_t k = 0; k < sorted.size(); ++k) {
    if (k == 0 || unitOf[sorted[k]] != unitOf[sorted[k - 1]]) units.push_back({});
    units.back().push_back(sorted[k]);
  }
  for (const std::vector<int>& indices : units) {
    std::map<int, int> local;
    for (size_t k = 0; k < indices.size(); ++k) local[indices[k]] = (int)k;
    std::vector<int> ures;
    for (const int g : indices) ures.push_back(res[g]);
    const bool single = ures.front() == ures.back();
    std::vector<int> ranks(ures);
    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
    std::vector<Bond> bonds;
    for (const Bond& b : v.bonds) {
      const auto p = local.find(b.i), q = local.find(b.j);
      if (p != local.end() && q != local.end()) bonds.push_back({p->second, q->second, b.type});
    }
    std::vector<std::string> element, colour;
    std::vector<int> charge, nHeavy;
    for (size_t k = 0; k < indices.size(); ++k) {
      const int g = indices[k];
      element.push_back(v.element[g]);
      charge.push_back(v.charge[g]);
      nHeavy.push_back(heavy[g]);
      const int rank = v.isPolymer[g]
          ? (single ? 0 : (int)(std::lower_bound(ranks.begin(), ranks.end(), ures[k]) - ranks.begin()))
          : -1;
      colour.push_back(v.element[g] + "/" + std::to_string(heavy[g]) + "/" + std::to_string(rank));
    }
    const std::vector<double> orders = normalizedBondOrders(element, charge, bonds, nHeavy);
    std::vector<std::vector<int>> perms = automorphisms(colour, bonds, orders, max);
    for (auto& p : perms) for (int& q : p) q = indices[q];
    out.push_back(perms);
  }
  return out;
}

inline bool finiteAt(const std::vector<double>& c, int i) {
  return std::isfinite(c[i * 3]) && std::isfinite(c[i * 3 + 1]) && std::isfinite(c[i * 3 + 2]);
}

/** Fit `pred` onto `gt` over `frame` and return every pred atom moved by that fit. */
inline std::vector<double> superimposeOn(const std::vector<double>& pred, const std::vector<double>& gt,
                                         const std::vector<uint8_t>& frame) {
  std::vector<double> fixed, mobile;
  for (size_t i = 0; i < frame.size(); ++i) {
    if (!frame[i]) continue;
    for (int c = 0; c < 3; ++c) { fixed.push_back(gt[i * 3 + c]); mobile.push_back(pred[i * 3 + c]); }
  }
  const superpose::Fit f = superpose::superimpose(fixed, mobile);
  const double* R = f.R;
  std::vector<double> out(pred.size());
  for (size_t i = 0; i < pred.size() / 3; ++i) {
    const double x = pred[i * 3] - f.mobileCenter[0];
    const double y = pred[i * 3 + 1] - f.mobileCenter[1];
    const double z = pred[i * 3 + 2] - f.mobileCenter[2];
    out[i * 3] = x * R[0] + y * R[1] + z * R[2] + f.fixedCenter[0];
    out[i * 3 + 1] = x * R[3] + y * R[4] + z * R[5] + f.fixedCenter[1];
    out[i * 3 + 2] = x * R[6] + y * R[7] + z * R[8] + f.fixedCenter[2];
  }
  return out;
}

/** `atom_permutation`. */
inline std::vector<int> atomPermutation(const std::vector<double>& pred, const std::vector<double>& gt,
                                        const std::vector<uint8_t>& gtMask,
                                        const std::vector<std::vector<std::vector<int>>>& units,
                                        const std::vector<int>* atomClasses,
                                        const std::vector<uint8_t>* frameMask) {
  const int n = (int)gtMask.size();
  std::vector<int> perm(n);
  for (int i = 0; i < n; ++i) perm[i] = i;
  std::vector<uint8_t> frame(n);
  for (int i = 0; i < n; ++i) {
    const bool f = frameMask ? (*frameMask)[i] : (gtMask[i] && (!atomClasses || (*atomClasses)[i] >= 0));
    frame[i] = f && gtMask[i];
  }
  int nFrame = 0;
  for (const uint8_t f : frame) nFrame += f;
  const std::vector<double> x = nFrame >= 3 ? superimposeOn(pred, gt, frame) : pred;
  for (const auto& rows : units) {
    if (rows.size() <= 1 || rows[0].empty()) continue;
    const std::vector<int>& dest = rows[0];
    const int m = (int)dest.size();
    std::vector<uint8_t> active(m);
    int nActive = 0;
    for (int k = 0; k < m; ++k) {
      bool moves = false;
      for (const auto& r : rows) moves = moves || r[k] != dest[k];
      active[k] = gtMask[dest[k]] && moves && (!atomClasses || (*atomClasses)[dest[k]] >= 0);
      nActive += active[k];
    }
    nActive = std::max(nActive, 1);
    size_t best = 0;
    double bestErr = INFINITY;
    for (size_t c = 0; c < rows.size(); ++c) {
      const std::vector<int>& src = rows[c];
      bool legal = true;
      double err = 0;
      for (int k = 0; k < m; ++k) {
        if (!active[k]) continue;
        if (!gtMask[src[k]] || (atomClasses && (*atomClasses)[src[k]] != (*atomClasses)[dest[k]])) {
          legal = false;
          break;
        }
        const int d = dest[k], s = src[k];
        const double dx = x[d * 3] - gt[s * 3];
        const double dy = x[d * 3 + 1] - gt[s * 3 + 1];
        const double dz = x[d * 3 + 2] - gt[s * 3 + 2];
        err += dx * dx + dy * dy + dz * dz;
      }
      err = legal && std::isfinite(err) ? err / nActive : INFINITY;
      if (err < bestErr) { bestErr = err; best = c; }
    }
    const std::vector<int>& chosen = rows[best];
    for (int k = 0; k < m; ++k) if (active[k]) perm[dest[k]] = chosen[k];
  }
  return perm;
}

/** `permute_gt_atoms`. */
inline std::vector<double> permuteGtAtoms(const std::vector<double>& pred, const std::vector<double>& gt,
                                          const View& view, const std::vector<int>* atomClasses,
                                          const std::vector<uint8_t>* frameMask) {
  const int n = view.n;
  if (atomClasses) {
    std::map<int, int> counts;
    for (const int c : *atomClasses) if (c >= 0) ++counts[c];
    int most = 0;
    for (const auto& [c, k] : counts) { (void)c; most = std::max(most, k); }
    if (most < 2) return gt;
  }
  const auto units = automorphismUnits(view);
  bool any = false;
  for (const auto& u : units) any = any || u.size() > 1;
  if (!any) return gt;
  std::vector<uint8_t> gtMask(n);
  bool observed = false;
  for (int i = 0; i < n; ++i) observed |= (gtMask[i] = finiteAt(gt, i));
  if (!observed) return gt;
  const std::vector<int> perm = atomPermutation(pred, gt, gtMask, units, atomClasses, frameMask);
  std::vector<double> out(gt.size());
  for (int i = 0; i < n; ++i) for (int c = 0; c < 3; ++c) out[i * 3 + c] = gt[perm[i] * 3 + c];
  return out;
}

inline void permutationsOf(const std::vector<int>& items, std::vector<std::vector<int>>& out) {
  if (items.size() <= 1) { out.push_back(items); return; }
  for (size_t k = 0; k < items.size(); ++k) {
    std::vector<int> rest(items);
    rest.erase(rest.begin() + k);
    std::vector<std::vector<int>> sub;
    permutationsOf(rest, sub);
    for (auto& p : sub) {
      p.insert(p.begin(), items[k]);
      out.push_back(p);
    }
  }
}

/** `_subunit_candidates`. */
inline std::vector<std::vector<int>> subunitCandidates(const View& v, const std::vector<int>& classes, int limit) {
  const int n = v.n;
  std::vector<int> identity(n);
  for (int i = 0; i < n; ++i) identity[i] = i;
  std::vector<int> molOrder;
  std::map<int, std::vector<int>> molecules;
  for (int i = 0; i < n; ++i) {
    if (!molecules.count(v.molIid[i])) molOrder.push_back(v.molIid[i]);
    molecules[v.molIid[i]].push_back(i);
  }
  const std::vector<int> res = residueIndex(v);
  std::vector<std::string> groupOrder;
  std::map<std::string, std::vector<std::vector<int>>> groups;
  for (const int mol : molOrder) {
    const std::vector<int>& indices = molecules[mol];
    bool anyClass = false;
    for (const int i : indices) anyClass = anyClass || classes[i];
    if (!anyClass) continue;
    std::string key = std::to_string(v.molEntity[indices[0]]) + "\x1f";
    for (size_t k = 0; k < indices.size(); ++k) {
      const int g = indices[k];
      if (k == 0 || v.chainIid[g] != v.chainIid[indices[k - 1]]) key += "c" + std::to_string(k);
      if (k == 0 || res[g] != res[indices[k - 1]]) key += "r" + std::to_string(k);
    }
    for (const int g : indices) {
      key += "\x1e" + v.resName[g] + "\x1d" + v.atomName[g] + "\x1d" + v.element[g] + "\x1d" +
             std::to_string(classes[g]);
    }
    if (!groups.count(key)) groupOrder.push_back(key);
    groups[key].push_back(indices);
  }
  long count = 1;
  for (const std::string& key : groupOrder) {
    for (size_t f = 2; f <= groups[key].size(); ++f) {
      count *= (long)f;
      if (count > limit) {
        throw std::runtime_error("Subunit symmetry above " + std::to_string(limit) +
                                 " permutations is not supported, as upstream's is not");
      }
    }
  }
  if (count == 1) return {identity};
  std::vector<std::vector<std::vector<int>>> assignments = {{}};
  for (const std::string& key : groupOrder) {
    const auto& g = groups[key];
    std::vector<int> ks(g.size());
    for (size_t k = 0; k < g.size(); ++k) ks[k] = (int)k;
    std::vector<std::vector<int>> orders;
    permutationsOf(ks, orders);
    std::vector<std::vector<std::vector<int>>> next;
    for (const auto& o : orders) {
      for (const auto& a : assignments) {
        std::vector<std::vector<int>> b = a;
        for (const int k : o) b.push_back(g[k]);
        next.push_back(b);
      }
    }
    assignments = next;
  }
  std::vector<std::vector<int>> slots;
  for (const std::string& key : groupOrder) for (const auto& m : groups[key]) slots.push_back(m);
  std::vector<std::vector<int>> out;
  for (const auto& assignment : assignments) {
    std::vector<int> cand = identity;
    for (size_t s = 0; s < assignment.size(); ++s) {
      for (size_t k = 0; k < slots[s].size(); ++k) cand[slots[s][k]] = assignment[s][k];
    }
    out.push_back(cand);
  }
  return out;
}

/** `permute_gt_subunits`. */
inline std::vector<double> permuteGtSubunits(const std::vector<double>& pred, const std::vector<double>& gt,
                                             const View& v, const std::vector<uint8_t>& frame,
                                             const std::vector<uint8_t>& score, bool align = true,
                                             int maxPermutations = 720) {
  const int n = v.n;
  std::vector<int> classes(n);
  for (int i = 0; i < n; ++i) classes[i] = (align && frame[i] ? 1 : 0) + 2 * (score[i] ? 1 : 0);
  const std::vector<std::vector<int>> candidates = subunitCandidates(v, classes, maxPermutations);
  if (candidates.size() == 1) return gt;
  std::vector<uint8_t> valid(n), fit(n), scored(n);
  bool anyScored = false, anyFit = false;
  for (int i = 0; i < n; ++i) {
    valid[i] = finiteAt(gt, i);
    const bool observed = valid[i] && finiteAt(pred, i);
    fit[i] = frame[i] && observed;
    scored[i] = score[i] && observed;
    anyScored |= scored[i];
    anyFit |= fit[i];
  }
  if (!anyScored || (align && !anyFit)) return gt;
  const std::vector<int>* best = &candidates[0];
  double bestErr = INFINITY;
  for (const auto& cand : candidates) {
    bool same = true;
    for (int i = 0; i < n && same; ++i) same = valid[cand[i]] == valid[i];
    if (!same) continue;
    std::vector<double> target((size_t)n * 3);
    for (int i = 0; i < n; ++i) for (int c = 0; c < 3; ++c) target[i * 3 + c] = gt[cand[i] * 3 + c];
    const std::vector<double> fitted = align ? superimposeOn(pred, target, fit) : pred;
    double err = 0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
      if (!scored[i]) continue;
      ++count;
      for (int c = 0; c < 3; ++c) {
        const double d = fitted[i * 3 + c] - target[i * 3 + c];
        err += d * d;
      }
    }
    err /= count;
    if (err < bestErr) { bestErr = err; best = &cand; }
  }
  std::vector<double> out(gt.size());
  for (int i = 0; i < n; ++i) for (int c = 0; c < 3; ++c) out[i * 3 + c] = gt[(*best)[i] * 3 + c];
  return out;
}

}  // namespace symmetry
}  // namespace rfd4p
