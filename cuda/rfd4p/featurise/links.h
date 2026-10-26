// atomworks' link chemistry, the twin of shared/rfd4p/pipeline/links.js: which upstream runs when
// it PREPARES a raw structure (`RCSB_WITH_METALS`, `add_missing_atoms`) - an authored inter-residue
// bond turns two dictionary reactants into one product, and the product is built and checked.
//
// 🔴 IT CHANGES WHAT THE MODEL SEES. 5O45's CCS CE takes a link where its dictionary entry has no
// leaving atom, so its carboxylate's C=O becomes C-O(-); 1DPN's TAF phosphate moves its P=O onto
// OP1; 1QFE's linked amide drops a hydrogen and a charge; a product with no legal valence (1U9X,
// 2ZZ6, 8QIA) is refused. The functions follow links.js one for one, which follows upstream.
//
// Templates over the atom site and the mmCIF block, so cif.h can include this before it defines
// them: a site needs `n`, `element` (upper case), `atomName`, `resName`, `chainId`, `resId`,
// `insCode`, `charge`, `x`/`y`/`z` and `chainType`.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include "tables.inc"

namespace rfd4p {
namespace links {
using std::string;
using std::vector;

constexpr int ANY = 0, SINGLE = 1, DOUBLE = 2, COORDINATION = 8;

/** atomworks `BIOTITE_BOND_TYPE_TO_BOND_ORDER` with `.get(type, 1)`. */
inline int bondOrder(int t) {
  static const int order[9] = {1, 1, 2, 3, 4, 1, 2, 3, 0};
  return t >= 0 && t < 9 ? order[t] : 1;
}
/** `_BOND_ORDERS`, which `get_bond_degree_per_atom` sums: `.get(type, int(type))`. */
inline double degreeWeight(int t) {
  static const int order[9] = {1, 1, 2, 3, 4, 1, 2, 3, 0};
  return t >= 0 && t < 9 ? order[t] : t;
}
/** atomworks `DEFAULT_VALENCE` by an UPPER CASE element: its `Cl` and `Br` never match. */
inline std::optional<int> defaultValence(const string& el) {
  static const std::map<string, int> v = {{"H", 1}, {"C", 4}, {"N", 3}, {"O", 2}, {"F", 1}, {"B", 3}};
  auto it = v.find(el);
  return it == v.end() ? std::nullopt : std::optional<int>(it->second);
}
inline bool isHydrogen(const string& e) { return e == "H" || e == "H2" || e == "D" || e == "T"; }
inline string upper(string s) {
  for (char& c : s) c = (char)toupper((unsigned char)c);
  return s;
}

[[noreturn]] inline void refuse(const string& why) {
  fprintf(stderr, "%s\n", why.c_str());
  exit(1);
}

/** `_is_valid_rdkit_valence`, read off the generated table (valence.js). */
inline bool isValidRdkitValence(const string& element, int charge, int valence) {
  auto it = tables::RDKIT_VALENCE.find(upper(element));
  if (it == tables::RDKIT_VALENCE.end()) return false;
  if (charge < -4 || charge > 4 || valence < 0 || valence > 16) {
    refuse("RDKit valence probe [" + element + "H" + std::to_string(valence) + std::to_string(charge) +
           "] is outside the generated table (valence.js)");
  }
  return (it->second[charge + 4] >> valence) & 1u;
}
inline int rdkitDefaultValence(const string& element) {
  auto it = tables::RDKIT_DEFAULT_VALENCE.find(upper(element));
  if (it == tables::RDKIT_DEFAULT_VALENCE.end()) refuse("Element '" + element + "' not found");
  return it->second;
}

/** A component as `atom_array_from_ccd_code` reads it: every atom in order, and its bonds. */
struct Comp {
  vector<string> names, element;
  vector<int> charge;
  vector<bool> leaving;
  vector<std::array<int, 3>> bonds;
};
using Groups = std::map<string, vector<vector<string>>>;

/** A component's atom's bonds, `{neighbour, type}` in the template's row order. */
inline vector<std::pair<int, int>> templateBonds(const Comp& c, int k) {
  vector<std::pair<int, int>> out;
  for (const auto& b : c.bonds) {
    if (b[0] == k) out.push_back({b[1], b[2]});
    else if (b[1] == k) out.push_back({b[0], b[2]});
  }
  return out;
}

/** `_leaving_groups_from_topology` (links.js `leavingGroups`). */
inline Groups leavingGroups(const Comp* comp) {
  Groups out;
  if (!comp || std::none_of(comp->leaving.begin(), comp->leaving.end(), [](bool b) { return b; })) return out;
  const int n = (int)comp->names.size();
  vector<vector<int>> adj(n);
  vector<std::set<int>> seen(n);
  for (const auto& b : comp->bonds) {
    if (seen[b[0]].insert(b[1]).second) adj[b[0]].push_back(b[1]);
    if (seen[b[1]].insert(b[0]).second) adj[b[1]].push_back(b[0]);
  }
  for (int atom = 0; atom < n; ++atom) {
    vector<vector<int>> groups;
    // A set iterated in insertion order: the neighbours as they were added, minus those reached.
    vector<int> unvisited = adj[atom];
    while (!unvisited.empty()) {
      const int start = unvisited.front();
      unvisited.erase(unvisited.begin());
      std::set<int> visited{atom, start};
      vector<int> queue{start}, component;
      for (size_t q = 0; q < queue.size(); ++q) {
        const int node = queue[q];
        component.push_back(node);
        for (const int nb : adj[node]) if (visited.insert(nb).second) queue.push_back(nb);
      }
      std::set<int> inComponent(component.begin(), component.end());
      unvisited.erase(std::remove_if(unvisited.begin(), unvisited.end(), [&](int x) { return inComponent.count(x) > 0; }),
                      unvisited.end());
      vector<int> heavy;
      for (const int x : component) if (comp->element[x] != "H") heavy.push_back(x);
      const vector<int>& judged = heavy.empty() ? component : heavy;
      const bool leaving = std::all_of(judged.begin(), judged.end(), [&](int x) { return (bool)comp->leaving[x]; });
      if (leaving) {
        std::sort(component.begin(), component.end());
        groups.push_back(component);
      }
    }
    if (groups.empty()) continue;
    std::stable_sort(groups.begin(), groups.end(), [](const vector<int>& a, const vector<int>& b) { return a[0] < b[0]; });
    vector<vector<string>>& named = out[comp->names[atom]];
    for (const auto& g : groups) {
      vector<string> names;
      for (const int x : g) names.push_back(comp->names[x]);
      named.push_back(names);
    }
  }
  return out;
}

/**
 * The dictionary lookups link chemistry makes, cached for one structure (links.js
 * `linkDictionary`): the generated templates, else a fetched entry; `polymerizationAtoms` is
 * cif.h's.
 */
template <class Block>
struct Dictionary {
  const vector<Block>* ccdBlocks = nullptr;
  std::function<std::pair<string, string>(const string&, int)> polymerizationAtoms;
  std::map<string, std::optional<Comp>> comps;
  std::map<string, Groups> groups;

  static const vector<string>* col(const Block& b, const char* cat, const char* key) {
    auto c = b.find(cat);
    if (c == b.end()) return nullptr;
    auto it = c->second.find(key);
    return it == c->second.end() ? nullptr : &it->second;
  }
  static string cell(const vector<string>* v, size_t i, const string& fallback = string()) {
    return v && i < v->size() ? (*v)[i] : fallback;
  }
  static bool unset(const string& v) { return v.empty() || v == "." || v == "?"; }

  static std::optional<Comp> fromBlock(const Block& b, const string& code) {
    const vector<string>* ids = col(b, "chem_comp_atom", "comp_id");
    if (!ids) return std::nullopt;
    Comp c;
    std::map<string, int> index;
    for (size_t r = 0; r < ids->size(); ++r) {
      if ((*ids)[r] != code) continue;
      const string name = cell(col(b, "chem_comp_atom", "atom_id"), r);
      index[name] = (int)c.names.size();
      c.names.push_back(name);
      c.element.push_back(upper(cell(col(b, "chem_comp_atom", "type_symbol"), r)));
      const string charge = cell(col(b, "chem_comp_atom", "charge"), r);
      c.charge.push_back(unset(charge) ? 0 : (int)strtol(charge.c_str(), nullptr, 10));
      c.leaving.push_back(cell(col(b, "chem_comp_atom", "pdbx_leaving_atom_flag"), r) == "Y");
    }
    if (c.names.empty()) return std::nullopt;
    if (const vector<string>* bc = col(b, "chem_comp_bond", "comp_id")) {
      for (size_t r = 0; r < bc->size(); ++r) {
        if ((*bc)[r] != code) continue;
        auto i = index.find(cell(col(b, "chem_comp_bond", "atom_id_1"), r));
        auto j = index.find(cell(col(b, "chem_comp_bond", "atom_id_2"), r));
        if (i == index.end() || j == index.end()) continue;
        string order = cell(col(b, "chem_comp_bond", "value_order"), r, "SING");
        string arom = cell(col(b, "chem_comp_bond", "pdbx_aromatic_flag"), r, "N");
        if (order.empty()) order = "SING";
        if (arom.empty()) arom = "N";
        auto t = tables::COMP_BOND_ORDER_TO_TYPE.find(upper(order) + ":" + upper(arom));
        c.bonds.push_back({std::min(i->second, j->second), std::max(i->second, j->second),
                           t == tables::COMP_BOND_ORDER_TO_TYPE.end() ? ANY : t->second});
      }
    }
    return c;
  }

  const Comp* comp(const string& code) {
    auto hit = comps.find(code);
    if (hit == comps.end()) {
      std::optional<Comp> out;
      auto t = tables::CCD_TEMPLATES.find(code);
      if (t != tables::CCD_TEMPLATES.end()) {
        Comp c;
        for (const auto& a : t->second.atoms) {
          c.names.push_back(a.name);
          c.element.push_back(upper(a.element));
          c.charge.push_back(a.charge);
          c.leaving.push_back(a.leaving);
        }
        c.bonds = t->second.bonds;
        out = c;
      } else if (ccdBlocks) {
        for (const Block& b : *ccdBlocks) {
          if ((out = fromBlock(b, code))) break;
        }
      }
      hit = comps.emplace(code, out).first;
    }
    return hit->second ? &*hit->second : nullptr;
  }

  const Groups& leaving(const string& code) {
    auto hit = groups.find(code);
    if (hit != groups.end()) return hit->second;
    const bool skip = code == "UNL" ||
        std::find(tables::WATER_LIKE_CCDS.begin(), tables::WATER_LIKE_CCDS.end(), code) != tables::WATER_LIKE_CCDS.end();
    return groups.emplace(code, skip ? Groups{} : leavingGroups(comp(code))).first->second;
  }

  /** `get_leaving_atom_bond_type`: -1 for none. */
  int leavingBondType(const string& code, const string& atomName) {
    const Groups& g = leaving(code);
    auto it = g.find(atomName);
    const Comp* c = comp(code);
    if (it == g.end() || !c) return -1;
    const auto k = std::find(c->names.begin(), c->names.end(), atomName);
    if (k == c->names.end()) return -1;
    std::set<string> set;
    for (const auto& grp : it->second) set.insert(grp.begin(), grp.end());
    for (const auto& [nb, t] : templateBonds(*c, (int)(k - c->names.begin()))) {
      if (set.count(c->names[nb]) && t != SINGLE) return t;
    }
    return -1;
  }
};

/** `infer_link_order` (links.js `inferLinkOrder`). */
template <class Site, class Dict>
int inferLinkOrder(const Site& a, int a1, int a2, Dict& dict, bool allowMissingTemplates) {
  const std::pair<string, string> partners[2] = {{a.resName[a1], a.atomName[a1]}, {a.resName[a2], a.atomName[a2]}};
  const string show = "[('" + partners[0].first + "', '" + partners[0].second + "'), ('" + partners[1].first + "', '" +
                      partners[1].second + "')]";
  const bool sameResidue = a.chainId[a1] == a.chainId[a2] && a.resId[a1] == a.resId[a2] &&
                           a.insCode[a1] == a.insCode[a2] && a.resName[a1] == a.resName[a2];
  int order = 1;
  for (const auto& p : partners) {
    const int t = dict.leavingBondType(p.first, p.second);
    order = std::max(order, t > 0 ? t : 1);
  }
  for (const auto& [resName, atomName] : partners) {
    const Comp* t = dict.comp(resName);
    if (!t) {
      if (allowMissingTemplates) return ANY;
      refuse("Cannot infer link order for " + show + ": missing CCD template for " + resName);
    }
    vector<int> matches;
    for (size_t k = 0; k < t->names.size(); ++k) if (t->names[k] == atomName) matches.push_back((int)k);
    if (matches.size() != 1) refuse("Cannot infer link order for " + show + ": missing CCD atom/bonds for " + resName + "/" + atomName);
    const vector<std::pair<int, int>> nbrs = templateBonds(*t, matches[0]);
    if (sameResidue) {
      for (const auto& [nb, type] : nbrs) if (t->names[nb] == partners[1].second) return type;
    }
    if (order == 1) continue;
    std::set<string> leaving;
    auto g = dict.leaving(resName).find(atomName);
    if (g != dict.leaving(resName).end()) for (const auto& grp : g->second) leaving.insert(grp.begin(), grp.end());
    int all = 0, retained = 0;
    for (const auto& [nb, type] : nbrs) {
      all += bondOrder(type);
      if (!isHydrogen(t->element[nb]) && !leaving.count(t->names[nb])) retained += bondOrder(type);
    }
    const string& element = t->element[matches[0]];
    int capacity = std::max(all, rdkitDefaultValence(element)) - retained;
    if (capacity < order && isValidRdkitValence(element, t->charge[matches[0]], retained + order)) capacity = order;
    order = std::min(order, capacity);
  }
  if (order < 1) {
    refuse("Cannot infer link order for " + show +
           ": retained heavy-atom bonds leave no valence; provide explicit product connectivity");
  }
  return order;
}

/** `_validate_inferred_polymer_bond`: each end of a backbone bond must have room for it. */
template <class Site, class Dict, class HasH>
void validateInferredPolymerBond(const Site& a, int i, int j, const vector<double>& degree, HasH hasH, Dict& dict) {
  for (const int idx : {i, j}) {
    const Groups& groups = dict.leaving(a.resName[idx]);
    const int expected = defaultValence(a.element[idx]).value_or(0) + a.charge[idx];
    auto g = groups.find(a.atomName[idx]);
    const bool declared = g != groups.end() && !g->second.empty();
    if (degree[idx] >= expected && !declared && !hasH(idx)) {
      refuse("Cannot infer polymer bond at " + a.resName[idx] + "/" + a.atomName[idx] +
             ": no free valence or declared leaving group");
    }
  }
}

/**
 * biotite's BondList as link chemistry uses it (links.js `BondRows`): rows kept in order, an update
 * keeping its row and a remove-then-add going last.
 */
struct BondRows {
  struct Row { int i, j, t; long long seq; };
  std::map<long long, Row> rows;
  vector<std::set<long long>> adj;
  long long next = 0, version = 0;
  int n = 0;

  explicit BondRows(int atoms) : adj(atoms), n(atoms) {}
  long long key(int i, int j) const { return (long long)std::min(i, j) * n + std::max(i, j); }
  void add(int i, int j, int t) {
    const long long k = key(i, j);
    ++version;
    auto it = rows.find(k);
    if (it != rows.end()) {
      it->second.t = t;
      return;
    }
    rows[k] = {std::min(i, j), std::max(i, j), t, next++};
    adj[i].insert(k);
    adj[j].insert(k);
  }
  void remove(int i, int j) {
    const long long k = key(i, j);
    auto it = rows.find(k);
    if (it == rows.end()) return;
    ++version;
    rows.erase(it);
    adj[i].erase(k);
    adj[j].erase(k);
  }
  int type(int i, int j) const {
    auto it = rows.find(key(i, j));
    return it == rows.end() ? -1 : it->second.t;
  }
  vector<Row> list() const {
    vector<Row> out;
    for (const auto& [k, r] : rows) out.push_back(r);
    std::sort(out.begin(), out.end(), [](const Row& a, const Row& b) { return a.seq < b.seq; });
    return out;
  }
  vector<Row> rowsOf(int i) const {
    vector<Row> out;
    for (const long long k : adj[i]) out.push_back(rows.at(k));
    std::sort(out.begin(), out.end(), [](const Row& a, const Row& b) { return a.seq < b.seq; });
    return out;
  }
  /** `get_bonds(i)`: `{neighbour, type}` in row order. */
  vector<std::pair<int, int>> of(int i) const {
    vector<std::pair<int, int>> out;
    for (const Row& r : rowsOf(i)) out.push_back({r.i == i ? r.j : r.i, r.t});
    return out;
  }
  /** `_find_bonded_hydrogens`: the first column's rows first, then the second's. */
  vector<int> hydrogensOf(int i, const vector<string>& element) const {
    const vector<Row> rs = rowsOf(i);
    vector<int> out;
    for (const Row& r : rs) if (r.i == i && isHydrogen(element[r.j])) out.push_back(r.j);
    for (const Row& r : rs) if (r.j == i && isHydrogen(element[r.i])) out.push_back(r.i);
    return out;
  }
  vector<double> degrees() const {
    vector<double> d(n, 0.0);
    for (const auto& [k, r] : rows) {
      d[r.i] += degreeWeight(r.t);
      d[r.j] += degreeWeight(r.t);
    }
    return d;
  }
};

/** What `resolveLinkChemistry` hands back: who stays, their charges, and the bonds in row order. */
struct Product {
  vector<bool> alive;
  vector<int> charge;
  vector<std::array<int, 3>> bonds;
};

/** The working state (links.js `LinkState`). */
template <class Site, class Dict>
struct LinkState {
  const Site& a;
  Dict& dict;
  int n;
  BondRows bonds;
  vector<int> charge;
  vector<bool> alive;
  vector<int> residue;

  LinkState(const Site& site, Dict& d, const vector<std::array<int, 3>>& rows)
      : a(site), dict(d), n((int)site.n), bonds((int)site.n), charge(site.charge), alive(site.n, true), residue(site.n, 0) {
    for (const auto& r : rows) bonds.add(r[0], r[1], r[2]);
    int r = 0;
    for (int i = 1; i < n; ++i) {
      if (a.chainId[i] != a.chainId[i - 1] || a.resId[i] != a.resId[i - 1] || a.insCode[i] != a.insCode[i - 1] ||
          a.resName[i] != a.resName[i - 1]) {
        ++r;
      }
      residue[i] = r;
    }
  }

  const string& el(int i) const { return a.element[i]; }
  bool finite(int i) const { return std::isfinite(a.x[i]) && std::isfinite(a.y[i]) && std::isfinite(a.z[i]); }
  bool anyNan(int i) const { return std::isnan(a.x[i]) || std::isnan(a.y[i]) || std::isnan(a.z[i]); }
  void kill(int i) {
    alive[i] = false;
    for (const auto& [nb, t] : bonds.of(i)) bonds.remove(i, nb);
  }
  int index(int i) const {
    int k = 0;
    for (int j = 0; j < i; ++j) k += alive[j] ? 1 : 0;
    return k;
  }
  bool validValence(int i) const {
    int v = 0;
    for (const auto& [nb, t] : bonds.of(i)) v += bondOrder(t);
    return isValidRdkitValence(el(i), charge[i], v);
  }
  bool overvalent(int i, const vector<double>& degree) const {
    std::optional<int> expected = defaultValence(el(i));
    if (!expected) return false;
    int e = *expected;
    if (el(i) != "C") e += charge[i];
    return degree[i] > e;
  }
  vector<BondRows::Row> interResidueBonds() const {
    vector<BondRows::Row> out;
    for (const BondRows::Row& r : bonds.list()) {
      if (r.t == COORDINATION) continue;
      if (a.chainId[r.i] != a.chainId[r.j] || a.resId[r.i] != a.resId[r.j] || a.insCode[r.i] != a.insCode[r.j]) {
        out.push_back(r);
      }
    }
    return out;
  }
  vector<bool> interMask() const {
    vector<bool> m(n, false);
    for (const auto& r : interResidueBonds()) m[r.i] = m[r.j] = true;
    return m;
  }

  /** `_displace_hydrogens`, unresolved ones first. */
  int displaceHydrogens(int i, int count, std::set<int>& removed) {
    const vector<int> explicitH = bonds.hydrogensOf(i, a.element);
    vector<int> ordered;
    for (const int h : explicitH) if (!finite(h)) ordered.push_back(h);
    for (const int h : explicitH) if (finite(h)) ordered.push_back(h);
    if ((int)ordered.size() > std::max(0, count)) ordered.resize(std::max(0, count));
    for (const int h : ordered) {
      bonds.remove(i, h);
      removed.insert(h);
    }
    return (int)ordered.size();
  }

  /** `_remove_ccd_leaving_atoms`, resolved atoms preserved. */
  void removeCcdLeavingAtoms(const vector<BondRows::Row>& inter) {
    std::set<int> withInterSet;
    for (const auto& r : inter) withInterSet.insert({r.i, r.j});
    const vector<int> withInter(withInterSet.begin(), withInterSet.end());
    vector<bool> interM(n, false), polymerLink(n, false);
    for (const int i : withInter) interM[i] = true;
    for (const auto& r : inter) {
      const auto s0 = dict.polymerizationAtoms(a.resName[r.i], a.chainType[r.i]);
      const auto s1 = dict.polymerizationAtoms(a.resName[r.j], a.chainType[r.j]);
      const string& p0 = a.atomName[r.i];
      const string& p1 = a.atomName[r.j];
      if ((p0 == s0.first && p1 == s1.second) || (p0 == s0.second && p1 == s1.first)) {
        polymerLink[r.i] = polymerLink[r.j] = true;
      }
    }
    vector<double> displaceable(n, 0.0);
    for (const auto& r : inter) {
      displaceable[r.i] += bondOrder(r.t);
      displaceable[r.j] += bondOrder(r.t);
    }
    std::map<int, vector<int>> resAtoms;
    for (int i = 0; i < n; ++i) if (alive[i]) resAtoms[residue[i]].push_back(i);
    vector<bool> leaving(n, false);
    int current = -1;
    std::set<string> present, preserved, linked;
    for (const int i : withInter) {
      const Groups& g = dict.leaving(a.resName[i]);
      auto it = g.find(a.atomName[i]);
      if (it == g.end() || it->second.empty()) continue;
      const vector<vector<string>>& groups = it->second;
      const vector<int>& members = resAtoms[residue[i]];
      if (residue[i] != current) {
        present.clear();
        preserved.clear();
        linked.clear();
        for (const int x : members) {
          present.insert(a.atomName[x]);
          if (finite(x)) preserved.insert(a.atomName[x]);
          if (interM[x]) linked.insert(a.atomName[x]);
        }
        current = residue[i];
      }
      auto touches = [](const vector<string>& grp, const std::set<string>& s) {
        return std::any_of(grp.begin(), grp.end(), [&](const string& nm) { return s.count(nm) > 0; });
      };
      vector<const vector<string>*> attached, removable;
      for (const auto& grp : groups) if (touches(grp, present)) attached.push_back(&grp);
      for (const auto* grp : attached) {
        if ((polymerLink[i] || !touches(*grp, preserved)) && !touches(*grp, linked)) removable.push_back(grp);
      }
      const double budget = std::max(0.0, displaceable[i] - (double)(groups.size() - attached.size()));
      for (size_t k = 0; k < removable.size() && (double)k < budget; ++k) {
        const std::set<string> set(removable[k]->begin(), removable[k]->end());
        for (const int x : members) if (set.count(a.atomName[x])) leaving[x] = true;
      }
    }
    if (std::none_of(leaving.begin(), leaving.end(), [](bool b) { return b; })) return;
    const vector<BondRows::Row> rows = bonds.list();
    for (int pass = 0; pass < 2; ++pass) {
      for (const auto& r : rows) {
        const int l = pass == 0 ? r.i : r.j, o = pass == 0 ? r.j : r.i;
        if (!leaving[l] || interM[o] || leaving[o]) continue;
        refuse("Leaving atom '" + a.atomName[l] + "' in residue " + a.resName[l] + " (index " + std::to_string(index(l)) +
               ") is bonded to '" + a.atomName[o] + "' (index " + std::to_string(index(o)) +
               "), which is neither a parent atom making inter-residue bonds nor another leaving atom. Cannot safely remove.");
      }
    }
    for (int i = 0; i < n; ++i) if (leaving[i]) kill(i);
  }

  /** `_restore_leaving_bond_orders_or_raise`. */
  void restoreLeavingBondOrdersOrRaise() {
    vector<double> gained(n, 0.0);
    for (const auto& r : interResidueBonds()) {
      gained[r.i] += bondOrder(r.t);
      gained[r.j] += bondOrder(r.t);
    }
    vector<double> degree = bonds.degrees();
    std::set<int> removed;
    for (int i = 0; i < n; ++i) {
      if (!gained[i]) continue;
      const int type = dict.leavingBondType(a.resName[i], a.atomName[i]);
      const int lost = type < 0 ? 0 : bondOrder(type);
      if (lost <= gained[i]) continue;
      const Comp* t = dict.comp(a.resName[i]);
      const int k = (int)(std::find(t->names.begin(), t->names.end(), a.atomName[i]) - t->names.begin());
      double expected = 0;
      for (const auto& [nb, bt] : templateBonds(*t, k)) expected += degreeWeight(bt);
      const string where = a.resName[i] + "/" + a.atomName[i];
      const double times = std::min(lost - gained[i], expected - degree[i]);
      for (int step = 0; step < times; ++step) {
        struct Cand { string state; int nb; bool hasH; int charge; };
        vector<Cand> candidates;
        for (const auto& [nb, bt] : bonds.of(i)) {
          if (bt != SINGLE || residue[nb] != residue[i] || (el(nb) != "O" && el(nb) != "S" && el(nb) != "N")) continue;
          vector<int> heavy;
          for (const auto& [x, xt] : bonds.of(nb)) if (!isHydrogen(el(x))) heavy.push_back(x);
          const bool hasH = !bonds.hydrogensOf(nb, a.element).empty();
          const int c = charge[nb] == -1 ? 0 : charge[nb];
          const int valence = (int)degree[nb] + 1 - (hasH ? 1 : 0);
          if (heavy.size() == 1 && heavy[0] == i && (hasH || charge[nb] == -1) && isValidRdkitValence(el(nb), c, valence)) {
            candidates.push_back({el(nb) + "," + std::to_string(c) + "," + std::to_string(valence), nb, hasH, c});
          }
        }
        if (candidates.empty()) refuse("Unsupported bond-order rearrangement at " + where);
        for (const Cand& c : candidates) if (c.state != candidates[0].state) refuse("Ambiguous bond-order rearrangement at " + where);
        const Cand* pick = &candidates[0];
        for (const Cand& c : candidates) if (!c.hasH) { pick = &c; break; }
        if (pick->hasH) displaceHydrogens(pick->nb, 1, removed);
        bonds.remove(i, pick->nb);
        bonds.add(i, pick->nb, DOUBLE);
        charge[pick->nb] = pick->charge;
        degree[i] += 1;
        degree[pick->nb] += pick->hasH ? 0 : 1;
      }
      if (degree[i] < expected) {
        refuse("Unsupported bond-order rearrangement at " + where +
               ": provide explicit product bond orders instead of inferring a replacement double bond");
      }
    }
    for (const int h : removed) kill(h);
  }

  /** `_lower_multiple_bond` (without implicit hydrogens no donor hands one over, as upstream). */
  bool lowerMultipleBond(int i, vector<double>& degree, bool requireOvervalentPartner, vector<bool>& impacted) {
    for (const auto& [p, bt] : bonds.of(i)) {
      const int order = bondOrder(bt);
      const bool over = overvalent(p, degree);
      if (order <= 1 || (el(p) != "C" && el(p) != "N") || residue[p] != residue[i] || (requireOvervalentPartner && !over)) continue;
      if (!over) continue;
      bonds.remove(i, p);
      bonds.add(i, p, order - 1);
      degree[i] -= 1;
      degree[p] -= 1;
      impacted[i] = impacted[p] = true;
      return true;
    }
    return false;
  }

  /** `_maybe_fix_overvalent_carbon`. */
  vector<int> maybeFixOvervalentCarbon(int i, vector<double>& degree, std::set<int>& removed, vector<bool>& impacted) {
    const vector<std::pair<int, int>> partners = bonds.of(i);
    if (displaceHydrogens(i, 1, removed)) {
      degree[i] -= 1;
      return {};
    }
    bool hasCarbonyl = false, foreign = false;
    for (const auto& [p, t] : partners) {
      const bool own = residue[p] == residue[i];
      if (own && el(p) == "O" && t == DOUBLE) hasCarbonyl = true;
      if (!own && t != COORDINATION) foreign = true;
    }
    if (hasCarbonyl && foreign) {
      for (const auto& [p, t] : partners) {
        if (residue[p] != residue[i] || t != SINGLE) continue;
        int heavyNbrs = 0;
        for (const auto& [x, xt] : bonds.of(p)) heavyNbrs += isHydrogen(el(x)) ? 0 : 1;
        if (el(p) == "N" && anyNan(p) && heavyNbrs == 1) {
          vector<int> out{p};
          for (const int h : bonds.hydrogensOf(p, a.element)) out.push_back(h);
          return out;
        }
      }
    }
    for (const auto& [o, t] : partners) {
      if (el(o) == "O" && t == DOUBLE) {
        bonds.remove(i, o);
        bonds.add(i, o, SINGLE);
        charge[o] = -1;
        degree[i] -= 1;
        degree[o] -= 1;
        impacted[o] = true;
        return {};
      }
    }
    lowerMultipleBond(i, degree, false, impacted);
    return {};
  }

  /** `_rebalance_terminal_heteroatoms`. */
  void rebalanceTerminalHeteroatoms(vector<bool>& impacted, vector<double>& degree, std::set<int>& removed) {
    vector<int> linkedAtoms;
    for (int i = 0; i < n; ++i) if (impacted[i] && alive[i]) linkedAtoms.push_back(i);
    for (const int l : linkedAtoms) {
      const string element = el(l);
      if (validValence(l)) continue;
      const vector<std::pair<int, int>> nbrs = bonds.of(l);
      if (element == "P" || element == "S") {
        vector<std::pair<int, int>> candidates;
        for (const auto& [o, t] : nbrs) {
          if (t != DOUBLE || el(o) != "O") continue;
          const vector<std::pair<int, int>> on = bonds.of(o);
          vector<int> covalent;
          int coordination = 0;
          for (const auto& [x, xt] : on) {
            if (xt != COORDINATION) covalent.push_back(x);
            else ++coordination;
          }
          if (residue[o] == residue[l] && covalent.size() == 1 && covalent[0] == l &&
              isValidRdkitValence(element, charge[l], (int)degree[l] - 1)) {
            candidates.push_back({coordination, o});
          }
        }
        if (!candidates.empty()) {
          const int o = std::min_element(candidates.begin(), candidates.end())->second;
          bonds.remove(l, o);
          bonds.add(l, o, SINGLE);
          charge[o] = -1;
          degree[l] -= 1;
          degree[o] -= 1;
          impacted[o] = true;
        }
        continue;
      }
      if (element != "O" && element != "S" && element != "N") continue;
      struct Cand { int coordination; string signature; int center, s; bool hasH; int charge; };
      vector<Cand> candidates;
      for (const auto& [center, t] : nbrs) {
        if (t != DOUBLE || residue[center] != residue[l]) continue;
        for (const auto& [s, st] : bonds.of(center)) {
          if (st != SINGLE || residue[s] != residue[l] || el(s) != element) continue;
          const vector<std::pair<int, int>> sn = bonds.of(s);
          vector<int> heavyCovalent;
          int coordination = 0;
          for (const auto& [x, xt] : sn) {
            if (xt != COORDINATION && !isHydrogen(el(x))) heavyCovalent.push_back(x);
            if (xt == COORDINATION) ++coordination;
          }
          if (heavyCovalent.size() != 1 || heavyCovalent[0] != center) continue;
          const int explicitH = (int)bonds.hydrogensOf(s, a.element).size();
          const bool hasH = explicitH > 0;
          const int c = charge[s] == -1 ? 0 : charge[s];
          if (isValidRdkitValence(element, c, (int)degree[s] + 1 - (hasH ? 1 : 0))) {
            candidates.push_back({coordination, std::to_string(charge[s]) + ",0," + std::to_string(explicitH), center, s, hasH, c});
          }
        }
      }
      if (candidates.empty()) continue;
      int least = candidates[0].coordination;
      for (const Cand& c : candidates) least = std::min(least, c.coordination);
      vector<Cand> kept;
      for (const Cand& c : candidates) if (c.coordination == least) kept.push_back(c);
      bool same = true;
      for (const Cand& c : kept) same = same && c.signature == kept[0].signature;
      if (!same) continue;
      const Cand c = kept[0];
      if (c.hasH) displaceHydrogens(c.s, 1, removed);
      bonds.remove(l, c.center);
      bonds.add(l, c.center, SINGLE);
      bonds.remove(c.center, c.s);
      bonds.add(c.center, c.s, DOUBLE);
      charge[c.s] = c.charge;
      degree[l] -= 1;
      degree[c.s] += c.hasH ? 0 : 1;
      impacted[c.s] = true;
    }
  }

  /** `_resolve_overvalent_atoms`. */
  vector<bool> resolveOvervalentAtoms() {
    vector<bool> impacted = interMask();
    vector<int> carbons, others;
    for (int i = 0; i < n; ++i) {
      if (!impacted[i]) continue;
      (el(i) == "C" ? carbons : others).push_back(i);
    }
    vector<double> degree = bonds.degrees();
    long long seen = bonds.version;
    std::set<int> removed;
    for (const int i : carbons) if (overvalent(i, degree)) lowerMultipleBond(i, degree, true, impacted);
    rebalanceTerminalHeteroatoms(impacted, degree, removed);
    vector<int> order = carbons;
    order.insert(order.end(), others.begin(), others.end());
    for (const int i : order) {
      if (removed.count(i)) continue;
      const string element = el(i);
      if (element == "S" && degree[i] == 3 && displaceHydrogens(i, 1, removed)) {
        degree[i] -= 1;
        continue;
      }
      if (!overvalent(i, degree)) continue;
      vector<int> displace;
      if (element == "C") displace = maybeFixOvervalentCarbon(i, degree, removed, impacted);
      else if (element == "N" || element == "O") {
        displaceHydrogens(i, (int)(degree[i] - *defaultValence(element) - charge[i]), removed);
      }
      for (const int gone : displace) {
        for (const auto& [nb, t] : bonds.of(gone)) bonds.remove(gone, nb);
        removed.insert(gone);
      }
      // Upstream recounts every atom's degree here; the count only moves when a bond did.
      if (bonds.version != seen) {
        degree = bonds.degrees();
        seen = bonds.version;
      }
    }
    for (const int x : removed) {
      kill(x);
      impacted[x] = false;
    }
    return impacted;
  }

  /** `apply_link_product_charges`. */
  void applyLinkProductCharges(const vector<bool>& toUpdate) {
    bool anyH = false;
    for (int i = 0; i < n; ++i) anyH = anyH || (alive[i] && isHydrogen(el(i)));
    if (!anyH) return;  // upstream warns and returns
    static const std::map<string, int> PRODUCT = {
        {"B,0,4", -1}, {"C,1,4", 0}, {"N,0,4", 1}, {"O,-1,2", 0}, {"O,0,1", -1},
        {"P,0,4", 1},  {"S,0,3", 1}, {"SE,0,1", -1}, {"SE,0,3", 1}};
    const vector<double> degree = bonds.degrees();
    const vector<bool> inter = interMask();
    for (int i = 0; i < n; ++i) {
      if (!toUpdate[i] || !alive[i] || degree[i] != 1 || !inter[i]) continue;
      int size = 0;
      for (int j = 0; j < n; ++j) size += alive[j] && residue[j] == residue[i] ? 1 : 0;
      if (size > 1) {
        refuse("Atom " + std::to_string(index(i)) + " (" + a.element[i] + ", atom_name=" + a.atomName[i] + ", res_name=" +
               a.resName[i] + ", chain_id=" + a.chainId[i] + ", res_id=" + std::to_string(a.resId[i]) +
               ") has inter-residue bonds but no intra-residue bonds \xe2\x80\x94 usually a missing CCD template. Ensure the "
               "residue is not an unknown ligand (UNL) or otherwise lacking bonding information.");
      }
    }
    for (int i = 0; i < n; ++i) {
      if (!toUpdate[i] || !alive[i]) continue;
      char deg[32];
      snprintf(deg, sizeof deg, "%g", degree[i]);
      const string key = upper(el(i)) + "," + std::to_string(charge[i]) + "," + deg;
      auto it = PRODUCT.find(key);
      if (it != PRODUCT.end()) charge[i] = it->second;
      else if (key == "O,0,3") {
        bool oxygen = false;
        for (const auto& [x, t] : bonds.of(i)) oxygen = oxygen || (t != COORDINATION && el(x) == "O");
        if (!oxygen) charge[i] = 1;
      }
    }
  }

  /** `_has_amide_bond`. */
  bool hasAmideBond(int nIdx) const {
    for (const auto& [c, ct] : bonds.of(nIdx)) {
      if (el(c) != "C") continue;
      for (const auto& [o, ot] : bonds.of(c)) if (el(o) == "O" && bonds.type(c, o) == DOUBLE) return true;
    }
    return false;
  }

  /** `_normalize_linked_amides`. */
  void normalizeLinkedAmides(vector<bool>& toUpdate) {
    std::set<int> removed;
    for (int i = 0; i < n; ++i) {
      if (!toUpdate[i] || !alive[i] || el(i) != "N" || charge[i] != 1) continue;
      if (!hasAmideBond(i)) continue;
      if (!displaceHydrogens(i, 1, removed)) {
        refuse("Cannot neutralize amide nitrogen " + std::to_string(index(i)) + ": no removable hydrogen");
      }
      charge[i] -= 1;
    }
    for (const int h : removed) {
      kill(h);
      toUpdate[h] = false;
    }
  }

  /** `_validate_link_valence`. */
  void validateLinkValence(const vector<bool>& impacted) {
    const vector<double> degree = bonds.degrees();
    for (int i = 0; i < n; ++i) {
      if (!impacted[i] || !alive[i] || validValence(i)) continue;
      bool completable = false;
      for (int h = 1; h <= 4 && !completable; ++h) completable = isValidRdkitValence(el(i), charge[i], (int)degree[i] + h);
      if (completable) continue;
      char deg[32];
      snprintf(deg, sizeof deg, "%g", degree[i]);
      refuse("Unresolved link valence at " + a.chainId[i] + "/" + std::to_string(a.resId[i]) + "/" + a.resName[i] + "/" +
             a.atomName[i] + " (element=" + a.element[i] + ", charge=" + std::to_string(charge[i]) +
             ", bond-order sum=" + deg + ")");
    }
  }

  Product result() const {
    Product p{alive, charge, {}};
    for (const auto& r : bonds.list()) p.bonds.push_back({r.i, r.j, r.t});
    return p;
  }
};

/** `resolve_link_chemistry` (links.js `resolveLinkChemistry`) over bonds in upstream's row order. */
template <class Site, class Dict>
Product resolveLinkChemistry(const Site& a, Dict& dict, const vector<std::array<int, 3>>& rows) {
  LinkState<Site, Dict> L(a, dict, rows);
  const auto inter = L.interResidueBonds();
  if (inter.empty()) return L.result();
  L.removeCcdLeavingAtoms(inter);
  L.restoreLeavingBondOrdersOrRaise();
  vector<bool> impacted = L.resolveOvervalentAtoms();
  bool any = false;
  for (int i = 0; i < L.n; ++i) any = any || (impacted[i] && L.alive[i]);
  if (!any) return L.result();
  L.applyLinkProductCharges(impacted);
  L.normalizeLinkedAmides(impacted);
  L.validateLinkValence(impacted);
  return L.result();
}

}  // namespace links
}  // namespace rfd4p
