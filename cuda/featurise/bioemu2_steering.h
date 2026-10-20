// shared/bioemu2/{graph,steering-tables,ligand-tables}.js and potentials.js's hostTables, natively: BioEmu2's bonded
// graph and its steering potentials' tables, byte for byte the JavaScript's (tools/check-native-bioemu2.mjs). The
// conventions are documented there; this transcribes. The integer tables are the JavaScript's arrays; the bounds its
// Float32Arrays; the references (bond_ref, angle_ref) its doubles.
#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "bioemu2_features.h"

namespace lf::bioemu2 {

constexpr int PROTEIN = 0, DNA = 1, RNA = 2, NONPOLYMER = 3;
inline std::string tokenName(int t) {
  static const char* names[] = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK_PROT"};
  return t >= 2 && t <= 22 ? names[t - 2] : "";
}
inline std::string nucleicToCcd(int t) {
  switch (t) { case 28: return "DA"; case 30: return "DC"; case 29: return "DG"; case 31: return "DT";
               case 23: return "A"; case 25: return "C"; case 24: return "G"; case 26: return "U"; }
  return "";
}
inline std::string dAminoToL(const std::string& code) {
  static const std::map<std::string, std::string> m = {
      {"DAL", "ALA"}, {"DAR", "ARG"}, {"DAS", "ASP"}, {"DCY", "CYS"}, {"DGL", "GLU"}, {"DGN", "GLN"}, {"DHI", "HIS"},
      {"DIL", "ILE"}, {"DLE", "LEU"}, {"DLY", "LYS"}, {"DPN", "PHE"}, {"DPR", "PRO"}, {"DSG", "ASN"}, {"DSN", "SER"},
      {"DTH", "THR"}, {"DTR", "TRP"}, {"DTY", "TYR"}, {"DVA", "VAL"}, {"MED", "MET"}};
  auto it = m.find(code);
  return it == m.end() ? "" : it->second;
}
inline bool isStandard(const std::string& label) {
  static const std::set<std::string> s = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                                          "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
                                          "A", "C", "G", "U", "DA", "DC", "DG", "DT"};
  return s.count(label) > 0;
}
inline bool isCanonicalProtein(int t) { return t >= 2 && t <= 21; }

inline std::vector<std::pair<int, int>> residueSpans(const std::vector<int>& res) {
  std::vector<std::pair<int, int>> out;
  int lo = 0;
  for (int i = 1; i <= (int)res.size(); ++i)
    if (i == (int)res.size() || res[i] != res[lo]) { out.push_back({lo, i}); lo = i; }
  return out;
}

// BioEmu2Graph
struct Graph {
  std::vector<std::pair<int, int>> bonds;          // (lower, higher), sorted
  std::set<std::pair<int, int>> bondSet;
  std::vector<std::pair<std::string, std::string>> labels;
  std::vector<std::pair<int, int>> residue;
  std::vector<int> mol;
  std::set<int> ordinaryX;
  std::map<int, std::set<int>> neighbors;          // keyed in insertion order of the JavaScript Map below
  std::vector<int> neighborOrder;                  // ...which is that order
  std::vector<std::array<int, 3>> angles;
  std::set<std::pair<int, int>> peptideEdges;

  explicit Graph(const Features& F) {
    const auto& names = F.names;
    int n = (int)names.size();
    std::map<std::pair<int, int>, bool> set;
    auto add = [&](int a, int b) { set[{std::min(a, b), std::max(a, b)}] = true; };
    auto residues = residueSpans(F.residueIdx);
    for (auto [lo, hi] : residues) {
      if (!isStandard(F.labels[lo])) continue;
      std::map<std::string, int> where;
      for (int i = lo; i < hi; ++i) where[names[i]] = i;
      CifLoop loop = loopOf(standardComponentTexts().at(F.labels[lo]), "_chem_comp_bond");
      auto c1 = std::find(loop.columns.begin(), loop.columns.end(), "atom_id_1") - loop.columns.begin();
      auto c2 = std::find(loop.columns.begin(), loop.columns.end(), "atom_id_2") - loop.columns.begin();
      for (auto& row : loop.rows)
        if (where.count(row[c1]) && where.count(row[c2])) add(where[row[c1]], where[row[c2]]);
    }
    for (auto& [a, b] : F.bondIndex) add(a, b);
    std::set<int> nucleicChains;
    for (auto [lo, hi] : residues) if (F.molType[lo] == DNA || F.molType[lo] == RNA) nucleicChains.insert(F.atomAsym[lo]);
    std::pair<int, int> prev{-1, -1};
    for (auto [lo, hi] : residues) {
      if (prev.first >= 0 && F.atomAsym[prev.first] == F.atomAsym[lo]) {
        std::map<std::string, int> pa, ca;
        for (int i = prev.first; i < prev.second; ++i) pa[names[i]] = i;
        for (int i = lo; i < hi; ++i) ca[names[i]] = i;
        if (nucleicChains.count(F.atomAsym[lo]) && pa.count("O3'") && ca.count("P")) add(pa["O3'"], ca["P"]);
        if (F.molType[prev.first] == PROTEIN && F.molType[lo] == PROTEIN && pa.count("N") && pa.count("C") && ca.count("N"))
          add(pa["C"], ca["N"]);
      }
      prev = {lo, hi};
    }
    for (auto& [k, v] : set) bonds.push_back(k);              // std::map orders as the JavaScript's sort does
    bondSet = std::set<std::pair<int, int>>(bonds.begin(), bonds.end());
    labels.resize(n);
    residue.resize(n);
    mol = F.molType;
    for (auto [lo, hi] : residues) {
      bool protein = F.molType[lo] == PROTEIN;
      int token = protein ? F.residueType[lo] : -1;
      std::set<int> tokens;
      for (int i = lo; i < hi; ++i) tokens.insert(F.tokenIdxAtAtom[i]);
      bool isX = protein && token == UNK_PROT && tokens.size() == 1;
      std::string name = nucleicToCcd(F.residueType[lo]);
      if (name.empty()) name = dAminoToL(F.labels[lo]);
      if (name.empty()) name = F.labels[lo];
      for (int i = lo; i < hi; ++i) {
        labels[i] = {isX ? tokenName(token) : name, names[i]};
        residue[i] = {F.atomAsym[lo], F.residueIdx[lo]};
        if (isX) ordinaryX.insert(i);
      }
    }
    for (auto& [i, j] : bonds) {
      if (!neighbors.count(i)) { neighbors[i]; neighborOrder.push_back(i); }
      if (!neighbors.count(j)) { neighbors[j]; neighborOrder.push_back(j); }
      neighbors[i].insert(j);
      neighbors[j].insert(i);
    }
    std::vector<int> centres(neighborOrder);
    std::sort(centres.begin(), centres.end());
    for (int c : centres) {
      std::vector<int> b(neighbors[c].begin(), neighbors[c].end());
      for (size_t x = 0; x < b.size(); ++x)
        for (size_t y = x + 1; y < b.size(); ++y) angles.push_back({b[x], b[y], c});
    }
    for (auto& [i, j] : bonds) {
      auto& ni = labels[i].second;
      auto& nj = labels[j].second;
      if (!sameResidue(i, j) && residue[i].first == residue[j].first && mol[i] == PROTEIN && mol[j] == PROTEIN && ni != nj
          && (ni == "C" || nj == "C") && (ni == "N" || nj == "N") && std::abs(residue[i].second - residue[j].second) == 1)
        peptideEdges.insert({i, j});
    }
  }
  bool sameResidue(int i, int j) const { return residue[i] == residue[j]; }
  bool hasFollowing(int i, int j, std::string& out) const {
    if (!peptideEdges.count({std::min(i, j), std::max(i, j)})) return false;
    out = labels[i].second == "N" ? labels[i].first : labels[j].first;
    return true;
  }
  bool angleFollowing(int oi, int c, int ok, std::string& out) const {
    bool found = false;
    if (sameResidue(c, ok)) found = hasFollowing(oi, c, out);
    if (!found && sameResidue(c, oi)) found = hasFollowing(c, ok, out);
    return found;
  }
  bool isNaLinkBond(int i, int j) const {
    auto& a = labels[i].second;
    auto& b = labels[j].second;
    return !sameResidue(i, j) && mol[i] == mol[j] && (mol[i] == DNA || mol[i] == RNA) && a != b
           && (a == "O3'" || b == "O3'") && (a == "P" || b == "P") && residue[i].first == residue[j].first
           && std::abs(residue[i].second - residue[j].second) == 1;
  }
  bool isDisulfide(int i, int j) const {
    return labels[i] == std::make_pair(std::string("CYS"), std::string("SG")) && labels[j] == labels[i] && mol[i] == PROTEIN && mol[j] == PROTEIN;
  }
  bool isNaLinkAngle(int oi, int c, int ok) const {
    if (!(mol[oi] == mol[c] && mol[c] == mol[ok])) return false;
    return (sameResidue(c, ok) && isNaLinkBond(oi, c)) || (sameResidue(c, oi) && isNaLinkBond(ok, c));
  }
  std::vector<std::pair<int, int>> nonbondedExclusions(int maxBonds = 3) const {
    std::set<std::pair<int, int>> out;
    for (int source : neighborOrder) {
      std::map<int, int> depth{{source, 0}};
      std::deque<int> queue{source};
      while (!queue.empty()) {
        int node = queue.front();
        queue.pop_front();
        if (depth[node] >= maxBonds) continue;
        for (int other : neighbors.at(node))
          if (!depth.count(other)) { depth[other] = depth[node] + 1; queue.push_back(other); }
      }
      for (auto& [o, d] : depth) if (source < o) out.insert({source, o});
    }
    return std::vector<std::pair<int, int>>(out.begin(), out.end());
  }
};

// --- the stereochemistry tables (parseStereoTables) ------------------------------------------------------------------

inline std::string canonicalKey(const std::vector<std::string>& atoms) {
  std::vector<std::string> rev(atoms.rbegin(), atoms.rend());
  const auto& pick = atoms < rev ? atoms : rev;           // tuple comparison, element by element
  std::string out;
  for (size_t i = 0; i < pick.size(); ++i) out += (i ? "_" : "") + pick[i];
  return out;
}
inline std::vector<std::string> splitUnderscore(const std::string& s) {
  std::vector<std::string> out;
  size_t at = 0, d;
  while ((d = s.find('_', at)) != std::string::npos) { out.push_back(s.substr(at, d - at)); at = d + 1; }
  out.push_back(s.substr(at));
  return out;
}
using Reference = std::array<double, 2>;
using StereoTable = std::map<std::string, std::map<std::string, Reference>>;
struct Stereo { StereoTable bonds, angles, linkBonds, linkAngles; };

inline Stereo parseStereoTables(const std::string& text) {
  // two levels of mapping over `atom_atom: [mean, sigma]` lists - all the file holds
  Stereo out;
  std::string section, data, residue, key;
  std::vector<double> list;
  StereoTable* table = nullptr;
  auto flush = [&]() {
    if (table && !key.empty() && list.size() == 2) (*table)[residue][canonicalKey(splitUnderscore(key))] = {list[0], list[1]};
    list.clear();
  };
  for (auto& raw : splitLines(text)) {
    std::string line = trimWs(raw);
    if (line.empty() || line[0] == '#') continue;
    size_t indent = raw.find_first_not_of(' ');
    if (line.compare(0, 2, "- ") == 0) { list.push_back(std::strtod(line.c_str() + 2, nullptr)); continue; }
    flush();
    size_t colon = line.rfind(':');
    std::string k = line.substr(0, colon);
    if (!k.empty() && (k[0] == '\'' || k[0] == '"')) k = k.substr(1, k.size() - 2);
    if (indent == 0) { section = k; table = nullptr; }
    else if (indent == 2) {
      data = k;
      bool link = section == "monlib_link";
      table = data == "bond_data" ? (link ? &out.linkBonds : &out.bonds) : data == "angle_data" ? (link ? &out.linkAngles : &out.angles) : nullptr;
    } else if (indent == 4) { residue = k; key.clear(); }
    else key = k;
  }
  flush();
  return out;
}

inline const Reference* lookup(const StereoTable& table, const std::string& residue, const std::vector<std::string>& atoms) {
  auto r = table.find(residue);
  if (r == table.end()) return nullptr;
  auto e = r->second.find(canonicalKey(atoms));
  return e == r->second.end() ? nullptr : &e->second;
}
inline std::string peptideLinkKey(const std::string& following) { return following == "GLY" ? "GLY" : following == "PRO" ? "PRO_TRANS" : "PEPTIDE"; }
using Ref3 = std::array<double, 3>;
inline bool pooledOf(const StereoTable& table, const std::vector<std::string>& names, bool peptide, Ref3& out) {
  std::vector<std::string> ids;
  if (peptide) ids = {"PEPTIDE", "GLY", "PRO_TRANS", "PRO_CIS"};
  else for (int t = 2; t <= 21; ++t) ids.push_back(tokenName(t));
  std::vector<Reference> entries;
  for (auto& i : ids) if (auto r = lookup(table, i, names)) entries.push_back(*r);
  if (entries.empty()) return false;
  double lo = INFINITY, hi = -INFINITY, s = -INFINITY;
  for (auto& e : entries) { lo = std::min(lo, e[0]); hi = std::max(hi, e[0]); s = std::max(s, e[1]); }
  out = {lo, hi, s};
  return true;
}

inline Ref3 bondReference(const Graph& g, int i, int j, const Stereo& t, std::map<std::string, std::pair<bool, Ref3>>& pooled) {
  static const Ref3 fallback = {1.1, 2.2, 0.03};
  std::vector<std::string> names = {g.labels[i].second, g.labels[j].second};
  bool same = g.sameResidue(i, j);
  std::string following;
  bool hasF = g.hasFollowing(i, j, following);
  if (g.ordinaryX.count(i) || g.ordinaryX.count(j)) {
    if (same || hasF) {
      std::string k = canonicalKey(names) + "|" + (hasF ? "true" : "false");
      if (!pooled.count(k)) { Ref3 r; bool ok = pooledOf(hasF ? t.linkBonds : t.bonds, names, hasF, r); pooled[k] = {ok, r}; }
      return pooled[k].first ? pooled[k].second : fallback;
    }
    return fallback;
  }
  const Reference* ref = nullptr;
  if (hasF) ref = lookup(t.linkBonds, peptideLinkKey(following), names);
  else if (g.isNaLinkBond(i, j)) ref = lookup(t.linkBonds, "NA", names);
  else if (same) ref = lookup(t.bonds, g.labels[i].first, names);
  else if (g.isDisulfide(i, j)) ref = lookup(t.linkBonds, "PEPTIDE", names);
  return ref ? Ref3{(*ref)[0], (*ref)[0], (*ref)[1]} : fallback;
}

inline Ref3 angleReference(const Graph& g, int oi, int c, int ok, const Stereo& t, std::map<std::string, std::pair<bool, Ref3>>& pooled) {
  static const Ref3 fallback = {60.0, 180.0, 3.0};
  std::vector<std::string> names = {g.labels[oi].second, g.labels[c].second, g.labels[ok].second};
  bool same = g.sameResidue(oi, c) && g.sameResidue(c, ok);
  std::string following;
  bool peptide = g.angleFollowing(oi, c, ok, following);
  if (g.ordinaryX.count(oi) || g.ordinaryX.count(c) || g.ordinaryX.count(ok)) {
    if (same || peptide) {
      std::string k = canonicalKey(names) + "|" + (peptide ? "true" : "false");
      if (!pooled.count(k)) { Ref3 r; bool ok2 = pooledOf(peptide ? t.linkAngles : t.angles, names, peptide, r); pooled[k] = {ok2, r}; }
      return pooled[k].first ? pooled[k].second : fallback;
    }
    return fallback;
  }
  const Reference* ref = nullptr;
  static const Reference disulfide = {104.2, 2.1};
  if (peptide) ref = lookup(t.linkAngles, peptideLinkKey(following), names);
  else if (g.isNaLinkAngle(oi, c, ok)) ref = lookup(t.linkAngles, "NA", names);
  else if (same) ref = lookup(t.angles, g.labels[c].first, names);
  else if (canonicalKey(names) == "CB_SG_SG" && (g.isDisulfide(c, oi) || g.isDisulfide(c, ok))) ref = &disulfide;
  return ref ? Ref3{(*ref)[0], (*ref)[0], (*ref)[1]} : fallback;
}

// geometryBounds, in float32 as the vendor's torch arithmetic
inline std::pair<float, float> geometryBounds(const Ref3& reference, double tolerance, bool angle) {
  double lo = reference[0], hi = reference[1], sigma = reference[2];
  if (angle) { lo = (lo * M_PI) / 180; hi = (hi * M_PI) / 180; sigma = (sigma * M_PI) / 180; }
  float width = (float)((double)(float)sigma * (double)(float)tolerance);
  float lower = (float)((double)(float)lo - (double)width);
  float upper = (float)((double)(float)hi + (double)width);
  return {(float)std::max(0.0, (double)lower), (float)std::min(angle ? M_PI : INFINITY, (double)upper)};
}

// --- Tier 2 from the CCD (ligand-tables.js) --------------------------------------------------------------------------

struct LigandTables {
  std::vector<int> boundsI, boundsJ, bondMask, angleMask;
  std::vector<float> upper, lower;
  std::vector<std::array<int, 4>> chiral;
  std::vector<int> chiralOrientation;
  std::vector<std::array<int, 4>> stereo;
  std::vector<int> stereoOrientation;
  std::vector<std::array<int, 6>> planar;
};

namespace tier2 {
inline double rvdw(const std::string& e) {
  static const std::map<std::string, double> r = {{"H", 1.2}, {"B", 1.8}, {"C", 1.7}, {"N", 1.6}, {"O", 1.55}, {"F", 1.5},
                                                  {"SI", 2.1}, {"P", 1.8}, {"S", 1.8}, {"CL", 1.8}, {"BR", 1.9}, {"I", 2.1}, {"SE", 1.9}};
  auto it = r.find(e);
  return it == r.end() ? 1.8 : it->second;
}
using V3 = std::array<double, 3>;
inline V3 sub(const V3& p, const V3& q) { return {p[0] - q[0], p[1] - q[1], p[2] - q[2]}; }
inline V3 cross(const V3& a, const V3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
inline double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline double dist(const V3& p, const V3& q) { return jsmath::hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); }
inline double dihedral(const V3& p0, const V3& p1, const V3& p2, const V3& p3) {
  V3 rij = sub(p0, p1), rkj = sub(p2, p1), rkl = sub(p2, p3);
  V3 n1 = cross(rij, rkj), n2 = cross(rkj, rkl);
  return jsmath::atan2(dot(rkj, cross(n1, n2)) / std::sqrt(dot(rkj, rkj)), dot(n1, n2));
}
struct Comp {
  std::vector<std::string> names, element;
  std::vector<V3> xyz;
  std::vector<bool> chiral;
  struct Bond { int a, b; std::string order; bool aromatic, stereo; };
  std::vector<Bond> bonds;
  std::vector<std::vector<int>> neighbours;
};
inline Comp component(const std::string& text) {
  Comp c;
  CifLoop atoms = loopOf(text, "_chem_comp_atom");
  auto col = [&](const std::string& name) { auto it = std::find(atoms.columns.begin(), atoms.columns.end(), name);
                                            return it == atoms.columns.end() ? -1 : (int)(it - atoms.columns.begin()); };
  auto get = [&](const std::vector<std::string>& row, const std::string& name) { int at = col(name); return at < 0 || at >= (int)row.size() ? std::string() : row[at]; };
  bool ideal = true;
  for (auto& r : atoms.rows) for (const char* a : {"x", "y", "z"}) {
    std::string v = get(r, std::string("pdbx_model_Cartn_") + a + "_ideal");
    char* end = nullptr;
    double d = v.empty() ? NAN : std::strtod(v.c_str(), &end);
    ideal = ideal && !v.empty() && end && *end == '\0' && std::isfinite(d);
  }
  std::map<std::string, int> index;
  for (auto& r : atoms.rows) {
    index[get(r, "atom_id")] = (int)c.names.size();
    c.names.push_back(get(r, "atom_id"));
    std::string e = get(r, "type_symbol");
    c.element.push_back(e.empty() ? "C" : upper(e));
    V3 p;
    int k = 0;
    for (const char* a : {"x", "y", "z"}) p[k++] = std::strtod(get(r, ideal ? std::string("pdbx_model_Cartn_") + a + "_ideal" : std::string("model_Cartn_") + a).c_str(), nullptr);
    c.xyz.push_back(p);
    std::string s = get(r, "pdbx_stereo_config");
    c.chiral.push_back(s == "R" || s == "S");
  }
  CifLoop bonds = loopOf(text, "_chem_comp_bond");
  auto bcol = [&](const std::string& name) { auto it = std::find(bonds.columns.begin(), bonds.columns.end(), name);
                                             return it == bonds.columns.end() ? -1 : (int)(it - bonds.columns.begin()); };
  auto bget = [&](const std::vector<std::string>& row, const std::string& name, const std::string& fallback) {
    int at = bcol(name); return at < 0 || at >= (int)row.size() ? fallback : row[at]; };
  for (auto& r : bonds.rows) {
    auto a = index.find(bget(r, "atom_id_1", ""));
    auto b = index.find(bget(r, "atom_id_2", ""));
    if (a == index.end() || b == index.end()) continue;
    std::string st = bget(r, "pdbx_stereo_config", "N");
    c.bonds.push_back({a->second, b->second, bget(r, "value_order", "SING"), bget(r, "pdbx_aromatic_flag", "N") == "Y", st == "E" || st == "Z"});
  }
  c.neighbours.resize(c.names.size());
  for (auto& b : c.bonds) { c.neighbours[b.a].push_back(b.b); c.neighbours[b.b].push_back(b.a); }
  return c;
}
inline std::vector<std::vector<int>> topological(const std::vector<std::vector<int>>& nb) {
  std::vector<std::vector<int>> out;
  for (size_t s = 0; s < nb.size(); ++s) {
    std::vector<int> d(nb.size(), -1);
    d[s] = 0;
    std::vector<int> queue{(int)s};
    for (size_t q = 0; q < queue.size(); ++q)
      for (int o : nb[queue[q]]) if (d[o] < 0) { d[o] = d[queue[q]] + 1; queue.push_back(o); }
    out.push_back(d);
  }
  return out;
}
inline std::pair<double, double> cisTrans(double l1, double l2, double l3, double a1, double a2) {
  V3 a = {l1 * jsmath::cos(a1), l1 * jsmath::sin(a1), 0};
  auto at = [&](double phi) { return V3{l2 - l3 * jsmath::cos(a2), l3 * jsmath::sin(a2) * jsmath::cos(phi), l3 * jsmath::sin(a2) * jsmath::sin(phi)}; };
  return {dist(a, at(0)), dist(a, at(M_PI))};
}
}  // namespace tier2

inline LigandTables ligandTables(const Features& F, const CcdReader& ccd) {
  using namespace tier2;
  LigandTables out;
  const auto& names = F.names;
  int lo = 0;
  while (lo < (int)names.size()) {
    int hi = lo + 1;
    while (hi < (int)names.size() && F.residueIdx[hi] == F.residueIdx[lo] && F.atomAsym[hi] == F.atomAsym[lo]) ++hi;
    auto code = F.molType[lo] == NONPOLYMER ? F.ligandCcd.find(F.atomAsym[lo]) : F.ligandCcd.end();
    if (code != F.ligandCcd.end()) {
      Comp c = component(componentText(ccd, code->second));
      std::map<std::string, int> where;
      for (int k = lo; k < hi; ++k) where[trimWs(names[k])] = k;
      std::map<int, int> map;
      for (size_t i = 0; i < c.names.size(); ++i) if (where.count(c.names[i])) map[(int)i] = where[c.names[i]];
      if (!map.empty()) {
        int n = (int)c.names.size();
        auto topo = topological(c.neighbours);
        std::vector<std::vector<double>> lower(n, std::vector<double>(n, 0.0)), upper(n, std::vector<double>(n, 1000.0));
        auto angleAt = [&](int i, int j, int k) {
          V3 u = sub(c.xyz[i], c.xyz[j]), v = sub(c.xyz[k], c.xyz[j]);
          return jsmath::acos(std::max(-1.0, std::min(1.0, dot(u, v) / std::sqrt(dot(u, u) * dot(v, v)))));
        };
        for (int i = 0; i < n; ++i)
          for (int j = i + 1; j < n; ++j) {
            double d = dist(c.xyz[i], c.xyz[j]), l, h;
            if (topo[i][j] == 1) { l = d - 0.01; h = d + 0.01; }
            else if (topo[i][j] == 2) { l = d - 0.04; h = d + 0.04; }
            else if (topo[i][j] == 3) {
              int b = -1, cc = -1;
              for (int x : c.neighbours[i]) if (topo[x][j] == 2) { b = x; break; }
              if (b >= 0) for (int x : c.neighbours[j]) if (topo[x][i] == 2 && std::find(c.neighbours[b].begin(), c.neighbours[b].end(), x) != c.neighbours[b].end()) { cc = x; break; }
              if (b < 0 || cc < 0) { l = d; h = d; }
              else {
                auto [cis, trans] = cisTrans(dist(c.xyz[i], c.xyz[b]), dist(c.xyz[b], c.xyz[cc]), dist(c.xyz[cc], c.xyz[j]),
                                             angleAt(i, b, cc), angleAt(b, cc, j));
                l = std::min(cis, trans);
                h = std::max(cis, trans);
              }
            } else {
              double vdw = rvdw(c.element[i]) + rvdw(c.element[j]);
              l = topo[i][j] == 4 ? vdw * 0.7 : vdw;
              l = std::min(l, d);
              h = 1000;
            }
            lower[i][j] = lower[j][i] = l;
            upper[i][j] = upper[j][i] = h;
          }
        for (int k = 0; k < n; ++k)
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
              if (i == j) continue;
              double u = upper[i][k] + upper[k][j];
              if (u < upper[i][j]) upper[i][j] = u;
              double l = std::max(lower[i][k] - upper[k][j], lower[k][j] - upper[i][k]);
              if (l > lower[i][j]) lower[i][j] = l;
            }
        for (int i = 0; i < n; ++i)
          for (int j = i + 1; j < n; ++j) {
            if (!map.count(i) || !map.count(j)) continue;
            out.boundsI.push_back(map[i]); out.boundsJ.push_back(map[j]);
            out.bondMask.push_back(topo[i][j] == 1); out.angleMask.push_back(topo[i][j] == 2);
            out.upper.push_back((float)upper[i][j]); out.lower.push_back((float)lower[i][j]);
          }
        auto hydrogenLast = [&](int p, int q) {
          int hp = c.element[p] == "H", hq = c.element[q] == "H";
          return hp != hq ? hp < hq : p < q;
        };
        for (int centre = 0; centre < n; ++centre) {
          int deg = (int)c.neighbours[centre].size();
          if (!c.chiral[centre] || deg > 4 || deg < 3) continue;
          std::vector<int> nb = c.neighbours[centre];
          std::stable_sort(nb.begin(), nb.end(), hydrogenLast);
          std::vector<std::array<int, 4>> tuples = {{nb[0], nb[1], nb[2], centre}};
          if (nb.size() == 4)
            for (int skip = 0; skip < 3; ++skip) {
              std::vector<int> cs;
              for (int i = 0; i < 4; ++i) if (i != skip) cs.push_back(nb[i]);
              if (skip % 2 == 0) tuples.push_back({cs[2], cs[1], cs[0], centre});
              else tuples.push_back({cs[0], cs[1], cs[2], centre});
            }
          for (auto& ids : tuples) {
            bool all = true;
            for (int i : ids) all = all && map.count(i);
            if (!all) continue;
            double phi = dihedral(c.xyz[ids[0]], c.xyz[ids[1]], c.xyz[ids[2]], c.xyz[ids[3]]);
            out.chiral.push_back({map[ids[0]], map[ids[1]], map[ids[2]], map[ids[3]]});
            out.chiralOrientation.push_back(phi > 0);
          }
        }
        for (auto& bond : c.bonds) {
          if (!bond.stereo) continue;
          std::vector<int> sn, en;
          for (int x : c.neighbours[bond.a]) if (x != bond.b) sn.push_back(x);
          for (int x : c.neighbours[bond.b]) if (x != bond.a) en.push_back(x);
          std::stable_sort(sn.begin(), sn.end(), hydrogenLast);
          std::stable_sort(en.begin(), en.end(), hydrogenLast);
          std::vector<std::pair<int, int>> pairs;
          if (!sn.empty() && !en.empty()) pairs.push_back({sn[0], en[0]});
          if (sn.size() == 2 && en.size() == 2) pairs.push_back({sn[1], en[1]});
          for (auto [p, q] : pairs) {
            std::array<int, 4> ids = {p, bond.a, bond.b, q};
            bool all = true;
            for (int i : ids) all = all && map.count(i);
            if (!all) continue;
            double phi = dihedral(c.xyz[ids[0]], c.xyz[ids[1]], c.xyz[ids[2]], c.xyz[ids[3]]);
            out.stereo.push_back({map[ids[0]], map[ids[1]], map[ids[2]], map[ids[3]]});
            out.stereoOrientation.push_back(std::fabs(phi) > M_PI / 2);
          }
        }
        for (auto& bond : c.bonds) {
          if (bond.order != "DOUB" || bond.aromatic) continue;
          int a = bond.a, b = bond.b;
          if (c.element[a] != "C" || c.element[b] != "C" || c.neighbours[a].size() != 3 || c.neighbours[b].size() != 3) continue;
          std::vector<int> ids = {a};
          for (int x : c.neighbours[a]) if (x != b) ids.push_back(x);
          ids.push_back(b);
          for (int x : c.neighbours[b]) if (x != a) ids.push_back(x);
          bool all = true;
          for (int i : ids) all = all && map.count(i);
          if (all) out.planar.push_back({map[ids[0]], map[ids[1]], map[ids[2]], map[ids[3]], map[ids[4]], map[ids[5]]});
        }
      }
    }
    lo = hi;
  }
  return out;
}

// --- steeringTables, then hostTables for the default specs (BondLength spec 8, BondAngle spec 9) --------------------

inline const std::vector<double>& vdwRadii() {
  static const std::vector<double> r = {
      1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54, 2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4,
      2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3,
      2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25, 2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45,
      2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28, 2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05,
      2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  return r;
}

struct SteeringTables {
  std::map<std::string, std::vector<int>> ints;    // integer tables, (k, n) flattened row-major where the vendor's are 2-D
  std::map<std::string, std::vector<float>> floats;
  std::map<std::string, std::vector<double>> doubles;   // the references, which the JavaScript keeps as doubles
};

inline SteeringTables steeringTables(const Features& F, const Stereo& stereo, const CcdReader* ccd, double sigmaTolerance = 10.0) {
  SteeringTables T;
  int n = (int)F.names.size(), nPad = F.atomsPadded;
  Graph g(F);
  std::vector<int> chain(nPad, 0), pad(nPad, 0);
  std::vector<float> radius(nPad, 0.f);
  for (int a = 0; a < n; ++a) {
    chain[a] = F.atomAsym[a];
    pad[a] = 1;
    int z = std::min(F.element[a], 118);
    radius[a] = (float)(z == 0 ? 0.0 : vdwRadii()[z - 1]);
  }
  const auto& asym = F.ints.at("asym_id");
  const auto& entity = F.ints.at("entity_id");
  int nChains = *std::max_element(asym.begin(), asym.end()) + 1;
  std::vector<int> sizes(nChains, 0);
  for (int a = 0; a < nPad; ++a) if (pad[a]) sizes[chain[a]] += 1;
  T.ints["atom_pad_mask"] = pad;
  T.ints["atom_chain_id"] = chain;
  T.floats["atom_vdw_radius"] = radius;
  T.ints["chain_sizes"] = sizes;
  std::set<std::pair<int, int>> pairSet;
  for (size_t t = 0; t < asym.size(); ++t) pairSet.insert({asym[t], entity[t]});
  std::vector<std::pair<int, int>> pairs(pairSet.begin(), pairSet.end());
  std::vector<int> symA, symB;
  for (size_t x = 0; x < pairs.size(); ++x)
    for (size_t y = x + 1; y < pairs.size(); ++y)
      if (pairs[x].second == pairs[y].second) { symA.push_back(pairs[x].first); symB.push_back(pairs[y].first); }
  auto rows2 = [](const std::vector<int>& a, const std::vector<int>& b) { std::vector<int> o(a); o.insert(o.end(), b.begin(), b.end()); return o; };
  T.ints["symmetric_chain_index"] = rows2(symA, symB);
  auto excl = g.nonbondedExclusions();
  std::vector<int> cross, ikey;
  for (auto& [a, b] : excl) if (chain[a] != chain[b]) cross.push_back(a * nPad + b);
  T.ints["interchain_bonded_pair_key"] = cross;
  std::vector<int> interMask(nPad, 0);
  for (int a = 0; a < nPad; ++a) interMask[a] = pad[a] && sizes[chain[a]] > 1 ? 1 : 0;
  T.ints["interchain_steric_atom_mask"] = interMask;
  std::vector<int> eligible(nPad, 0);
  for (int a = 0; a < n; ++a) {
    int mol = F.molType[a], rt = F.residueType[a];
    bool protein = mol == PROTEIN && (isCanonicalProtein(rt) || g.ordinaryX.count(a));
    bool nucleic = (mol == DNA || mol == RNA) && !nucleicToCcd(rt).empty();
    eligible[a] = protein || nucleic ? 1 : 0;
  }
  T.ints["polymer_steric_atom_mask"] = eligible;
  for (auto& [a, b] : excl) ikey.push_back(a * nPad + b);
  T.ints["polymer_bonded_pair_key"] = ikey;
  std::vector<int> la, lb;
  for (auto& [a, b] : excl)
    if (!g.bondSet.count({a, b}) && eligible[a] && eligible[b] && F.residueIdx[a] != F.residueIdx[b]) { la.push_back(a); lb.push_back(b); }
  T.ints["polymer_local_clash_index"] = rows2(la, lb);
  std::map<std::string, std::pair<bool, Ref3>> pooled;
  std::vector<double> bondRef, angleRef;
  std::vector<float> bondLower, bondUpper, angleLower, angleUpper;
  std::vector<int> bi, bj;
  for (auto& [i, j] : g.bonds) {
    Ref3 r = bondReference(g, i, j, stereo, pooled);
    bondRef.insert(bondRef.end(), r.begin(), r.end());
    auto [lo2, hi2] = geometryBounds(r, sigmaTolerance, false);
    bondLower.push_back(lo2); bondUpper.push_back(hi2);
    bi.push_back(i); bj.push_back(j);
  }
  pooled.clear();
  std::vector<int> ai, ak, ac;
  for (auto& [i, k, c] : g.angles) {
    Ref3 r = angleReference(g, i, c, k, stereo, pooled);
    angleRef.insert(angleRef.end(), r.begin(), r.end());
    auto [lo2, hi2] = geometryBounds(r, sigmaTolerance, true);
    angleLower.push_back(lo2); angleUpper.push_back(hi2);
    ai.push_back(i); ak.push_back(k); ac.push_back(c);
  }
  T.doubles["bond_ref"] = bondRef;
  T.doubles["angle_ref"] = angleRef;
  T.ints["bond_index"] = rows2(bi, bj);
  T.floats["bond_lower"] = bondLower;
  T.floats["bond_upper"] = bondUpper;
  std::vector<int> angleIndex(ai);
  angleIndex.insert(angleIndex.end(), ak.begin(), ak.end());
  angleIndex.insert(angleIndex.end(), ac.begin(), ac.end());
  T.ints["angle_index"] = angleIndex;
  T.floats["angle_lower"] = angleLower;
  T.floats["angle_upper"] = angleUpper;
  LigandTables L;
  if (ccd && !F.ligandCcd.empty()) L = ligandTables(F, *ccd);
  T.ints["rdkit_bounds_index"] = rows2(L.boundsI, L.boundsJ);
  T.ints["rdkit_bounds_bond_mask"] = L.bondMask;
  T.ints["rdkit_bounds_angle_mask"] = L.angleMask;
  T.floats["rdkit_upper_bounds"] = L.upper;
  T.floats["rdkit_lower_bounds"] = L.lower;
  auto rowsK = [](const auto& tuples, int k) {
    std::vector<int> out;
    for (int r = 0; r < k; ++r) for (auto& t : tuples) out.push_back(t[r]);
    return out;
  };
  T.ints["chiral_atom_index"] = rowsK(L.chiral, 4);
  T.ints["chiral_atom_orientations"] = L.chiralOrientation;
  T.ints["stereo_bond_index"] = rowsK(L.stereo, 4);
  T.ints["stereo_bond_orientations"] = L.stereoOrientation;
  T.ints["planar_bond_index"] = rowsK(L.planar, 6);
  // hostTables (the default specs): the impropers, the chain-pair filter, spec 8's and 9's bounds
  int K = (int)L.planar.size();
  static const int improper[4][2] = {{1, 4}, {2, 5}, {3, 0}, {0, 3}};
  std::vector<int> imp;
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < K; ++k)
      for (int m = 0; m < 2; ++m) imp.push_back(L.planar[k][improper[r][m]]);
  T.ints["planar_improper_index"] = imp;
  std::vector<int> pa, pb;
  for (size_t k = 0; k < symA.size(); ++k)
    if (sizes[symA[k]] > 1 && sizes[symB[k]] > 1) { pa.push_back(symA[k]); pb.push_back(symB[k]); }
  T.ints["symmetric_chain_pairs"] = rows2(pa, pb);
  T.floats["bond_lower.8"] = bondLower;
  T.floats["bond_upper.8"] = bondUpper;
  T.floats["angle_lower.9"] = angleLower;
  T.floats["angle_upper.9"] = angleUpper;
  return T;
}

}  // namespace lf::bioemu2
