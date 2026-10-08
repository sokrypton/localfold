// A ligand from its SMILES, as the page builds one: shared/chem - smiles.js (the graph), kekulize.js (integer
// orders, resonance), rings.js (the smallest set of smallest rings), stereo.js (tetrahedral centres),
// geometry-tables.js (lengths and angles), conformer.js (distance bounds, smoothing, embedding, refinement) and
// component.js (sixteen seeded starts, the best kept, named and rounded). Every step in the JavaScript's order and
// arithmetic, its Math through jsmath.h, so the conformer is the page's to the last bit; every refusal in its words.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <cstdint>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd.h"
#include "jsmath.h"
#include "tables.h"

namespace lf::chem {

// JavaScript's Math.max / Math.min of two: NaN wins, and +0 is larger than -0
inline double jmax(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return NAN;
  if (a == 0 && b == 0) return std::signbit(a) ? b : a;
  return a > b ? a : b;
}
inline double jmin(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return NAN;
  if (a == 0 && b == 0) return std::signbit(a) ? a : b;
  return a < b ? a : b;
}
inline double hypotOf(const std::vector<double>& v) {   // Math.hypot(...v)
  if (v.empty()) return 0;
  double max = 0;
  bool nan = false;
  for (double x : v) { if (std::isnan(x)) nan = true; else if (std::fabs(x) > max) max = std::fabs(x); }
  if (max == INFINITY) return INFINITY;
  if (nan) return NAN;
  if (max == 0) return 0;
  double sum = 0, compensation = 0;
  for (double x : v) {
    double n = std::fabs(x) / max, summand = n * n - compensation, preliminary = sum + summand;
    compensation = (preliminary - sum) - summand;
    sum = preliminary;
  }
  return std::sqrt(sum) * max;
}

struct Atom {
  std::string symbol;
  int element = 0;
  bool aromatic = false;
  int charge = 0;
  int isotope = 0;
  bool hasHydrogens = false;   // null hydrogens: "derive it"
  int hydrogens = 0;
  int chirality = 0;           // 0 none, 1 "@", 2 "@@"
  int mapClass = 0;
  bool bracket = false;
};
struct Bond {
  int from, to;
  double order;
  char direction = 0;          // '/' or '\\', 0 for none
  bool ring = false;
  bool aromatic = false;
  bool hasGeometryOrder = false;
  double geometryOrder = 0;
  double geo() const { return hasGeometryOrder ? geometryOrder : order; }
};
struct Graph {
  std::vector<Atom> atoms;
  std::vector<Bond> bonds;
  std::vector<std::vector<int>> writtenNeighbours;   // bond indices, in the order written
  std::vector<int> components;
};
struct Step { int atom; double order; int bond; };

inline std::vector<std::vector<Step>> adjacency(const Graph& g) {
  std::vector<std::vector<Step>> lists(g.atoms.size());
  for (size_t i = 0; i < g.bonds.size(); ++i) {
    lists[g.bonds[i].from].push_back({g.bonds[i].to, g.bonds[i].order, (int)i});
    lists[g.bonds[i].to].push_back({g.bonds[i].from, g.bonds[i].order, (int)i});
  }
  return lists;
}

inline int elementIndex(const std::string& symbol) {   // ELEMENT_SYMBOLS.indexOf(symbol)
  const auto& s = elementSymbols();
  for (size_t i = 0; i < s.size(); ++i) if (s[i] == symbol) return (int)i;
  return -1;
}

// ---------------------------------------------------------------- kekulize.js
inline void kekulize(Graph& g) {
  static const std::map<std::string, int> AROMATIC_VALENCE = {{"B", 3}, {"C", 4}, {"N", 3}, {"O", 2}, {"P", 3}, {"S", 2}, {"SE", 2}, {"AS", 3}};
  std::vector<int> aromaticBonds;
  for (size_t i = 0; i < g.bonds.size(); ++i) if (g.bonds[i].order == 1.5) aromaticBonds.push_back((int)i);
  if (aromaticBonds.empty()) return;
  for (int i : aromaticBonds) g.bonds[i].aromatic = true;
  std::vector<int> inAromatic;                 // a Set, in insertion order
  std::vector<char> seen(g.atoms.size(), 0);
  for (int i : aromaticBonds)
    for (int a : {g.bonds[i].from, g.bonds[i].to}) if (!seen[a]) { seen[a] = 1; inAromatic.push_back(a); }
  std::vector<int> needsDouble;
  for (int atom : inAromatic) {
    const Atom& info = g.atoms[atom];
    auto v = AROMATIC_VALENCE.find(info.symbol);
    if (v == AROMATIC_VALENCE.end()) continue;
    double used = 0;
    int aromaticNeighbours = 0;
    for (auto& b : g.bonds) {
      int other = b.from == atom ? b.to : b.to == atom ? b.from : -1;
      if (other < 0) continue;
      if (b.order == 1.5) { used += 1; ++aromaticNeighbours; }
      else used += b.order;
    }
    used += info.bracket ? (info.hasHydrogens ? info.hydrogens : 0) : 0;
    double target = v->second + (info.symbol == "B" ? -info.charge : info.charge);
    if (used < target && aromaticNeighbours > 0) needsDouble.push_back(atom);
  }
  // perfectMatching
  std::set<int> matched;
  bool ok;
  if (needsDouble.empty()) ok = true;
  else if (needsDouble.size() % 2 == 1) ok = false;
  else {
    std::map<int, std::vector<int>> options;
    std::set<int> need(needsDouble.begin(), needsDouble.end());
    for (int a : needsDouble) options[a];
    for (int i : aromaticBonds) {
      int f = g.bonds[i].from, t = g.bonds[i].to;
      if (need.count(f) && need.count(t)) { options[f].push_back(i); options[t].push_back(i); }
    }
    std::set<int> used;
    std::function<bool()> search = [&]() -> bool {
      int next = -1;
      double fewest = INFINITY;
      for (int atom : needsDouble) {
        if (used.count(atom)) continue;
        int open = 0;
        for (int i : options[atom]) if (!used.count(g.bonds[i].from) && !used.count(g.bonds[i].to)) ++open;
        if (open < fewest) { fewest = open; next = atom; }
      }
      if (next < 0) return true;
      if (fewest == 0) return false;
      for (int i : options[next]) {
        int f = g.bonds[i].from, t = g.bonds[i].to;
        if (used.count(f) || used.count(t)) continue;
        used.insert(f); used.insert(t); matched.insert(i);
        if (search()) return true;
        used.erase(f); used.erase(t); matched.erase(i);
      }
      return false;
    };
    ok = search();
  }
  if (!ok) {
    std::string symbols;
    for (size_t k = 0; k < needsDouble.size(); ++k)
      symbols += (k ? ", " : "") + g.atoms[needsDouble[k]].symbol + std::to_string(needsDouble[k]);
    throw std::runtime_error("this aromatic ring system cannot be kekulised: no way to give each of " + symbols
                             + " exactly one double bond. An aromatic atom written lower case must be able to take one, or be"
                               " written with its hydrogen in brackets (`[nH]`)");
  }
  for (int i : aromaticBonds) g.bonds[i].order = matched.count(i) ? 2 : 1;
}

inline void delocalizeCharges(Graph& g) {
  std::vector<int> degree(g.atoms.size(), 0);
  for (auto& b : g.bonds) { ++degree[b.from]; ++degree[b.to]; }
  for (int centre = 0; centre < (int)g.atoms.size(); ++centre) {
    const std::string& s = g.atoms[centre].symbol;
    if (s != "C" && s != "N" && s != "S" && s != "P") continue;
    struct Arm { int index; double order; };
    std::vector<Arm> arms;
    for (size_t i = 0; i < g.bonds.size(); ++i) {
      const Bond& b = g.bonds[i];
      int other = b.from == centre ? b.to : b.to == centre ? b.from : -1;
      if (other < 0 || degree[other] != 1) continue;
      if (g.atoms[other].symbol != "O" && g.atoms[other].symbol != "S") continue;
      if (g.atoms[other].hasHydrogens && g.atoms[other].hydrogens > 0) continue;
      arms.push_back({(int)i, b.order});
    }
    if (arms.size() < 2) continue;
    bool hasDouble = false, hasSingle = false;
    for (auto& a : arms) { if (a.order >= 2) hasDouble = true; if (a.order == 1) hasSingle = true; }
    if (!hasDouble || !hasSingle) continue;
    double total = 0;
    for (auto& a : arms) total += a.order;
    double mean = total / arms.size();
    for (auto& a : arms) { g.bonds[a.index].hasGeometryOrder = true; g.bonds[a.index].geometryOrder = mean; }
  }
}

// ---------------------------------------------------------------- smiles.js
inline Graph parseSmiles(const std::string& text) {
  std::string smiles = trimWs(text);
  if (smiles.empty()) throw std::runtime_error("an empty SMILES");
  if (smiles.find('>') != std::string::npos) throw std::runtime_error("reaction SMILES (`>`) is not a ligand");
  static const std::map<char, double> BOND_SYMBOLS = {{'-', 1}, {'=', 2}, {'#', 3}, {'$', 4}, {':', 1.5}, {'/', 1}, {'\\', 1}};
  static const std::set<std::string> ORGANIC = {"B", "C", "N", "O", "P", "S", "F", "CL", "BR", "I"};
  static const std::string AROMATIC_ORGANIC = "bcnops";
  Graph g;
  struct Open { std::string label; int atom; char symbol; size_t slot; };
  std::vector<Open> openRings;       // a Map, in insertion order
  std::vector<int> branches;
  int previous = -1;
  char pending = 0;
  size_t position = 0;
  auto fail = [&](const std::string& message) -> void {
    throw std::runtime_error(message + " at position " + std::to_string(position) + " of \"" + smiles + "\"");
  };
  auto isDirection = [](char s) { return s == '/' || s == '\\'; };
  auto addBond = [&](int from, int to, double order, char direction, bool ring) {
    if (from == to) fail("an atom bonded to itself");
    for (auto& b : g.bonds)
      if ((b.from == from && b.to == to) || (b.from == to && b.to == from))
        fail("atoms " + std::to_string(from) + " and " + std::to_string(to) + " are bonded twice");
    Bond b{from, to, order};
    b.direction = direction;
    b.ring = ring;
    g.bonds.push_back(b);
  };
  auto makeAtom = [&](const std::string& symbol, bool aromatic, int charge, int isotope, bool hasH, int hydrogens, int chirality,
                      int mapClass, bool bracket) {
    Atom a;
    a.symbol = symbol; a.element = elementIndex(symbol) + 1; a.aromatic = aromatic; a.charge = charge; a.isotope = isotope;
    a.hasHydrogens = hasH; a.hydrogens = hydrogens; a.chirality = chirality; a.mapClass = mapClass; a.bracket = bracket;
    return a;
  };
  auto readAtom = [&](size_t start, size_t& next) -> Atom {
    if (smiles[start] != '[') {
      for (const char* two : {"Cl", "Br"})
        if (smiles.compare(start, 2, two) == 0) { next = start + 2; return makeAtom(upper(two), false, 0, 0, false, 0, 0, 0, false); }
      char one = smiles[start];
      std::string up(1, (char)std::toupper((unsigned char)one));
      if (ORGANIC.count(up) && std::string(1, one) == up) { next = start + 1; return makeAtom(up, false, 0, 0, false, 0, 0, 0, false); }
      if (AROMATIC_ORGANIC.find(one) != std::string::npos) { next = start + 1; return makeAtom(up, true, 0, 0, false, 0, 0, 0, false); }
      if (one == '*') fail("`*` (any atom) has no element and cannot be folded");
      fail("\"" + std::string(1, one) + "\" is not an organic-subset atom; bracket it");
    }
    size_t close = smiles.find(']', start);
    if (close == std::string::npos) fail("a `[` with no `]`");
    std::string body = smiles.substr(start + 1, close - start - 1);
    static const std::regex BRACKET(R"(^(\d*)([A-Za-z][a-z]?)(@{1,2}(?:TH|AL|SP|TB|OH)?\d*)?(H\d*)?((?:[+-]\d*|\++|-+)?)(?::(\d+))?$)",
                                    std::regex::ECMAScript);
    std::smatch m;
    if (!std::regex_match(body, m, BRACKET)) fail("cannot read the bracket atom \"[" + body + "]\"");
    std::string isotope = m[1].str(), written = m[2].str();
    bool aromatic = std::islower((unsigned char)written[0]) != 0;
    std::string symbol = upper(written);
    if (elementIndex(symbol) < 0) fail("\"" + written + "\" is not an element");
    int chirality = 0;
    if (m[3].matched) {
      std::string raw = m[3].str();
      if (raw == "@") chirality = 1;
      else if (raw == "@@") chirality = 2;
      else fail("stereo class \"" + raw + "\" is not supported; only @ and @@ are");
    }
    int hydrogens = 0;
    if (m[4].matched) hydrogens = m[4].str() == "H" ? 1 : std::stoi(m[4].str().substr(1));
    int charge = 0;
    std::string chargeRaw = m[5].matched ? m[5].str() : "";
    if (!chargeRaw.empty()) {
      static const std::regex SIGNED(R"(^[+-]\d+$)");
      if (std::regex_match(chargeRaw, SIGNED)) charge = std::stoi(chargeRaw);
      else charge = (chargeRaw[0] == '+' ? 1 : -1) * (int)chargeRaw.size();
    }
    next = close + 1;
    return makeAtom(symbol, aromatic, charge, isotope.empty() ? 0 : std::stoi(isotope), true, hydrogens, chirality,
                    m[6].matched ? std::stoi(m[6].str()) : 0, true);
  };

  while (position < smiles.size()) {
    char c = smiles[position];
    if (c == '(') {
      if (previous < 0) fail("a branch before any atom");
      branches.push_back(previous);
      ++position;
      continue;
    }
    if (c == ')') {
      if (branches.empty()) fail("a `)` with no `(`");
      previous = branches.back();
      branches.pop_back();
      ++position;
      continue;
    }
    if (c == '.') { previous = -1; pending = 0; ++position; continue; }
    if (BOND_SYMBOLS.count(c)) {
      if (pending != 0) fail("two bond symbols in a row");
      pending = c;
      ++position;
      continue;
    }
    if (c == '%' || (c >= '0' && c <= '9')) {
      if (previous < 0) fail("a ring closure before any atom");
      std::string label;
      if (c == '%') {
        std::string digits = smiles.substr(position + 1, 2);
        if (digits.size() != 2 || !std::isdigit((unsigned char)digits[0]) || !std::isdigit((unsigned char)digits[1]))
          fail("`%` needs two digits");
        label = digits;
        position += 3;
      } else {
        label = std::string(1, c);
        position += 1;
      }
      auto open = std::find_if(openRings.begin(), openRings.end(), [&](const Open& o) { return o.label == label; });
      if (open == openRings.end()) {
        size_t slot = g.writtenNeighbours[previous].size();
        g.writtenNeighbours[previous].push_back(-1);    // reserved here, filled when the ring closes
        openRings.push_back({label, previous, pending, slot});
        pending = 0;
        continue;
      }
      Open o = *open;
      openRings.erase(open);
      char here = pending;
      pending = 0;
      char symbol = o.symbol != 0 ? o.symbol : here;
      if (o.symbol != 0 && here != 0 && o.symbol != here) {
        if (!(isDirection(o.symbol) && isDirection(here)))
          fail("ring bond " + label + " is " + std::string(1, o.symbol) + " at one end and " + std::string(1, here) + " at the other");
        symbol = o.symbol;
      }
      bool bothAromatic = g.atoms[o.atom].aromatic && g.atoms[previous].aromatic;
      double order = symbol == 0 ? (bothAromatic ? 1.5 : 1) : BOND_SYMBOLS.at(symbol);
      char direction = isDirection(o.symbol) ? o.symbol : (isDirection(here) ? (here == '/' ? '\\' : '/') : 0);
      addBond(o.atom, previous, order, direction, true);
      int created = (int)g.bonds.size() - 1;
      g.writtenNeighbours[o.atom][o.slot] = created;
      g.writtenNeighbours[previous].push_back(created);
      continue;
    }
    size_t next = 0;
    Atom atom = readAtom(position, next);
    position = next;
    int index = (int)g.atoms.size();
    g.atoms.push_back(atom);
    g.writtenNeighbours.emplace_back();
    if (previous >= 0) {
      bool bothAromatic = g.atoms[previous].aromatic && atom.aromatic;
      double order = pending == 0 ? (bothAromatic ? 1.5 : 1) : BOND_SYMBOLS.at(pending);
      addBond(previous, index, order, isDirection(pending) ? pending : 0, false);
      g.writtenNeighbours[previous].push_back((int)g.bonds.size() - 1);
      g.writtenNeighbours[index].push_back((int)g.bonds.size() - 1);
    } else if (pending != 0) {
      fail("a bond symbol at the start of a fragment");
    }
    pending = 0;
    previous = index;
  }
  if (!branches.empty()) fail("a `(` with no `)`");
  if (!openRings.empty()) {
    std::string labels;
    for (size_t i = 0; i < openRings.size(); ++i) labels += (i ? ", " : "") + openRings[i].label;
    throw std::runtime_error(std::string("ring closure") + (openRings.size() == 1 ? "" : "s") + " " + labels + " never closed in \"" + smiles + "\"");
  }
  if (pending != 0) fail("a bond symbol at the end");
  if (g.atoms.empty()) throw std::runtime_error("no atoms in \"" + smiles + "\"");

  // mergeExplicitHydrogens: a [H] with one non-hydrogen neighbour folds into its count
  {
    auto lists = adjacency(g);
    std::vector<char> drop(g.atoms.size(), 0);
    bool any = false;
    for (size_t index = 0; index < g.atoms.size(); ++index) {
      const Atom& a = g.atoms[index];
      if (a.symbol != "H" || lists[index].size() != 1) continue;
      int host = lists[index][0].atom;
      if (g.atoms[host].symbol == "H" || a.charge != 0) continue;
      g.atoms[host].hydrogens = (g.atoms[host].hasHydrogens ? g.atoms[host].hydrogens : 0) + 1;
      g.atoms[host].hasHydrogens = true;
      drop[index] = 1;
      any = true;
    }
    if (any) {
      std::vector<int> moved(g.atoms.size(), -1), bondMoved;
      int next = 0;
      for (size_t i = 0; i < g.atoms.size(); ++i) if (!drop[i]) moved[i] = next++;
      std::vector<Bond> kept;
      for (auto& b : g.bonds) {
        if (drop[b.from] || drop[b.to]) { bondMoved.push_back(-1); continue; }
        bondMoved.push_back((int)kept.size());
        Bond k = b; k.from = moved[b.from]; k.to = moved[b.to];
        kept.push_back(k);
      }
      std::vector<std::vector<int>> written;
      std::vector<Atom> atoms;
      for (size_t i = 0; i < g.atoms.size(); ++i) {
        if (drop[i]) continue;
        std::vector<int> list;
        for (int e : g.writtenNeighbours[i]) { int m = bondMoved[e]; if (m != -1) list.push_back(m); }
        written.push_back(list);
        atoms.push_back(g.atoms[i]);
      }
      g.atoms = atoms; g.bonds = kept; g.writtenNeighbours = written;
    }
  }
  kekulize(g);
  // fillImplicitHydrogens
  {
    static const std::map<std::string, std::vector<int>> IMPLICIT = {{"B", {3}}, {"C", {4}}, {"N", {3, 5}}, {"O", {2}}, {"P", {3, 5}},
      {"S", {2, 4, 6}}, {"F", {1}}, {"CL", {1}}, {"BR", {1}}, {"I", {1}}};
    std::vector<double> order(g.atoms.size(), 0);
    for (auto& b : g.bonds) { order[b.from] += b.order; order[b.to] += b.order; }
    for (size_t i = 0; i < g.atoms.size(); ++i) {
      Atom& a = g.atoms[i];
      if (a.hasHydrogens) continue;
      a.hasHydrogens = true;
      auto v = IMPLICIT.find(a.symbol);
      if (v == IMPLICIT.end()) { a.hydrogens = 0; continue; }
      double used = std::ceil(order[i] - 1e-9);
      int shift = a.symbol == "B" ? -a.charge : a.charge;
      int target = INT32_MIN;
      for (int valence : v->second) if (valence + shift >= used) { target = valence; break; }
      a.hydrogens = target == INT32_MIN ? 0 : (int)jmax(0, target + shift - used);
    }
  }
  delocalizeCharges(g);
  // fragmentsOf
  {
    int n = (int)g.atoms.size();
    g.components.assign(n, -1);
    std::vector<std::vector<int>> nb(n);
    for (auto& b : g.bonds) { nb[b.from].push_back(b.to); nb[b.to].push_back(b.from); }
    int next = 0;
    for (int start = 0; start < n; ++start) {
      if (g.components[start] >= 0) continue;
      std::vector<int> stack = {start};
      g.components[start] = next;
      while (!stack.empty()) {
        int atom = stack.back(); stack.pop_back();
        for (int o : nb[atom]) if (g.components[o] < 0) { g.components[o] = next; stack.push_back(o); }
      }
      ++next;
    }
  }
  return g;
}

// ---------------------------------------------------------------- rings.js (a BigInt bond vector as words)
struct Bits {
  std::vector<uint64_t> w;
  explicit Bits(size_t bits = 0) : w((bits + 63) / 64, 0) {}
  void set(size_t i) { w[i / 64] |= 1ull << (i % 64); }
  bool zero() const { for (auto x : w) if (x) return false; return true; }
  Bits operator^(const Bits& o) const { Bits r = *this; for (size_t i = 0; i < w.size(); ++i) r.w[i] ^= o.w[i]; return r; }
  bool operator<(const Bits& o) const {
    for (size_t i = w.size(); i-- > 0;) if (w[i] != o.w[i]) return w[i] < o.w[i];
    return false;
  }
  bool operator>(const Bits& o) const { return o < *this; }
};

inline std::vector<std::vector<int>> smallestRings(const Graph& g) {
  auto lists = adjacency(g);
  int n = (int)g.atoms.size();
  auto edge = [](int a, int b) { return a < b ? std::make_pair(a, b) : std::make_pair(b, a); };
  auto bondVector = [&](const std::vector<int>& ring) {
    std::set<std::pair<int, int>> inRing;
    for (size_t i = 0; i < ring.size(); ++i) inRing.insert(edge(ring[i], ring[(i + 1) % ring.size()]));
    Bits v(g.bonds.size());
    for (size_t i = 0; i < g.bonds.size(); ++i) if (inRing.count(edge(g.bonds[i].from, g.bonds[i].to))) v.set(i);
    return v;
  };
  struct Candidate { std::vector<int> atoms; Bits vector; };
  std::vector<Candidate> candidates;
  std::set<std::vector<int>> found;
  for (int root = 0; root < n; ++root) {
    std::map<int, int> previous = {{root, -1}};
    std::vector<int> depth(n, -1);
    depth[root] = 0;
    std::vector<int> queue = {root};
    for (size_t head = 0; head < queue.size(); ++head) {
      int atom = queue[head];
      for (auto& s : lists[atom]) {
        if (previous.count(s.atom)) continue;
        previous[s.atom] = atom;
        depth[s.atom] = depth[atom] + 1;
        queue.push_back(s.atom);
      }
    }
    auto pathTo = [&](int atom) {
      std::vector<int> path;
      for (int at = atom; at != -1;) {
        path.push_back(at);
        auto it = previous.find(at);
        if (it == previous.end()) break;
        at = it->second;
      }
      std::reverse(path.begin(), path.end());
      return path;
    };
    for (auto& b : g.bonds) {
      if (depth[b.from] < 0 || depth[b.to] < 0) continue;
      auto left = pathTo(b.from), right = pathTo(b.to);
      std::set<int> seen(left.begin(), left.end());
      int shared = 0;
      for (int a : right) if (seen.count(a)) ++shared;
      if (shared != 1) continue;
      std::vector<int> atoms(left.rbegin(), left.rend());
      atoms.insert(atoms.end(), right.begin() + 1, right.end());
      if (atoms.size() < 3) continue;
      std::vector<int> sorted = atoms;
      std::sort(sorted.begin(), sorted.end());
      if (found.count(sorted)) continue;
      found.insert(sorted);
      candidates.push_back({atoms, bondVector(atoms)});
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.atoms.size() < b.atoms.size(); });
  int fragments = 0;
  {
    std::set<int> labels(g.components.begin(), g.components.end());
    fragments = g.components.empty() ? (n > 0 ? 1 : 0) : (int)labels.size();
  }
  int rank = (int)g.bonds.size() - n + fragments;
  struct Row { Bits vector; size_t size; };
  std::vector<Row> basis;
  std::vector<std::vector<int>> chosen;
  int independent = 0;
  auto reduce = [&](const Bits& start, double limit) {
    Bits v = start;
    for (auto& row : basis) {
      if ((double)row.size > limit) continue;
      Bits next = v ^ row.vector;
      if (next < v) v = next;
    }
    return v;
  };
  auto sameRing = [](const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    std::set<int> sa(a.begin(), a.end()), both(a.begin(), a.end());
    both.insert(b.begin(), b.end());
    return sa.size() == both.size();
  };
  for (auto& c : candidates) {
    size_t size = c.atoms.size();
    Bits full = reduce(c.vector, INFINITY);
    if (!full.zero()) {
      if (independent >= rank) continue;
      basis.push_back({full, size});
      std::stable_sort(basis.begin(), basis.end(), [](const Row& a, const Row& b) { return a.vector > b.vector; });
      chosen.push_back(c.atoms);
      ++independent;
      continue;
    }
    if (reduce(c.vector, (double)size - 1).zero()) continue;
    bool dup = false;
    for (auto& r : chosen) if (sameRing(r, c.atoms)) dup = true;
    if (dup) continue;
    chosen.push_back(c.atoms);
  }
  return chosen;
}

inline std::vector<bool> bondsInRings(const Graph& g, const std::vector<std::vector<int>>& rings) {
  std::set<std::pair<int, int>> inRing;
  for (auto& r : rings)
    for (size_t i = 0; i < r.size(); ++i) {
      int a = r[i], b = r[(i + 1) % r.size()];
      inRing.insert(a < b ? std::make_pair(a, b) : std::make_pair(b, a));
    }
  std::vector<bool> out;
  for (auto& b : g.bonds) out.push_back(inRing.count(b.from < b.to ? std::make_pair(b.from, b.to) : std::make_pair(b.to, b.from)) > 0);
  return out;
}

// ---------------------------------------------------------------- stereo.js
struct Centre { int atom; int neighbours[4]; int sign; };
inline std::vector<Centre> chiralCentres(const Graph& g) {
  auto lists = adjacency(g);
  std::vector<Centre> centres;
  for (int index = 0; index < (int)g.atoms.size(); ++index) {
    const Atom& atom = g.atoms[index];
    if (atom.chirality == 0) continue;
    std::vector<int> written;
    for (int e : g.writtenNeighbours[index]) if (e >= 0) written.push_back(e);
    if (written.size() != lists[index].size()) continue;
    std::vector<int> order;
    for (int bond : written) order.push_back(g.bonds[bond].from == index ? g.bonds[bond].to : g.bonds[bond].from);
    int hydrogens = atom.hasHydrogens ? atom.hydrogens : 0;
    if (hydrogens > 0) {
      bool hasPreceding = !order.empty() && order[0] < index;
      order.insert(order.begin() + (hasPreceding ? 1 : 0), -1);
    }
    if (order.size() != 4 || hydrogens > 1) continue;
    Centre c;
    c.atom = index;
    for (int k = 0; k < 4; ++k) c.neighbours[k] = order[k] == -1 ? index : order[k];
    c.sign = atom.chirality == 1 ? 1 : -1;
    centres.push_back(c);
  }
  return centres;
}

// ---------------------------------------------------------------- geometry-tables.js
inline double radiusIn(const std::map<std::string, double>& t, const std::string& s, bool& found) {
  auto it = t.find(s);
  found = it != t.end();
  return found ? it->second : 0;
}
inline const std::map<std::string, double>& singleRadii() {
  static const std::map<std::string, double> t = {{"H", 0.31}, {"HE", 0.28}, {"LI", 1.28}, {"BE", 0.96}, {"B", 0.84}, {"C", 0.76},
    {"N", 0.71}, {"O", 0.66}, {"F", 0.57}, {"NE", 0.58}, {"NA", 1.66}, {"MG", 1.41}, {"AL", 1.21}, {"SI", 1.11}, {"P", 1.07},
    {"S", 1.05}, {"CL", 1.02}, {"AR", 1.06}, {"K", 2.03}, {"CA", 1.76}, {"SC", 1.70}, {"TI", 1.60}, {"V", 1.53}, {"CR", 1.39},
    {"MN", 1.39}, {"FE", 1.32}, {"CO", 1.26}, {"NI", 1.24}, {"CU", 1.32}, {"ZN", 1.22}, {"GA", 1.22}, {"GE", 1.20}, {"AS", 1.19},
    {"SE", 1.20}, {"BR", 1.20}, {"KR", 1.16}, {"RB", 2.20}, {"SR", 1.95}, {"Y", 1.90}, {"ZR", 1.75}, {"NB", 1.64}, {"MO", 1.54},
    {"TC", 1.47}, {"RU", 1.46}, {"RH", 1.42}, {"PD", 1.39}, {"AG", 1.45}, {"CD", 1.44}, {"IN", 1.42}, {"SN", 1.39}, {"SB", 1.39},
    {"TE", 1.38}, {"I", 1.39}, {"XE", 1.40}, {"CS", 2.44}, {"BA", 2.15}, {"PT", 1.36}, {"AU", 1.36}, {"HG", 1.32}, {"TL", 1.45},
    {"PB", 1.46}, {"BI", 1.48}};
  return t;
}
inline double covalentRadius(const std::string& symbol, double order, bool aromatic) {
  static const std::map<std::string, double> DOUBLE = {{"B", 0.75}, {"C", 0.67}, {"N", 0.60}, {"O", 0.57}, {"F", 0.59}, {"SI", 1.07},
    {"P", 1.02}, {"S", 0.94}, {"CL", 0.95}, {"AS", 1.06}, {"SE", 1.07}, {"BR", 1.09}, {"I", 1.25}, {"FE", 1.16}, {"ZN", 1.18}};
  static const std::map<std::string, double> TRIPLE = {{"B", 0.73}, {"C", 0.60}, {"N", 0.54}, {"O", 0.53}, {"SI", 1.00}, {"P", 0.94},
    {"S", 0.95}, {"AS", 1.14}, {"SE", 1.07}, {"BR", 1.10}, {"I", 1.25}};
  bool f;
  double single = radiusIn(singleRadii(), symbol, f);
  if (!f) return 1.5;
  double dbl = radiusIn(DOUBLE, symbol, f);
  if (!f) dbl = single;
  if (aromatic) return (single + dbl) / 2;
  if (order >= 3) { double t = radiusIn(TRIPLE, symbol, f); return f ? t : dbl; }
  if (order >= 2) return dbl;
  if (order > 1) return single + (dbl - single) * (order - 1);
  return single;
}
inline std::string isoelectronic(const std::string& symbol, int charge) {
  if (charge == 0) return symbol;
  static const std::vector<std::string> PERIOD = {"B", "C", "N", "O", "F"};
  int at = (int)(std::find(PERIOD.begin(), PERIOD.end(), symbol) - PERIOD.begin());
  if (at >= (int)PERIOD.size()) return symbol;
  int moved = at - charge;
  return moved >= 0 && moved < (int)PERIOD.size() ? PERIOD[moved] : symbol;
}
inline double bondLength(const std::string& a, const std::string& b, double order, bool aromatic, int chargeA, int chargeB) {
  static const std::map<std::string, double> MEASURED = {{"O-P:s", 1.593}, {"O-P:p", 1.540}, {"O-P:d", 1.491}, {"O-S:s", 1.520},
    {"O-S:p", 1.470}, {"O-S:d", 1.455}, {"C-O:p", 1.263}, {"C-N:s", 1.429}, {"C-F:s", 1.363}, {"N-O:p", 1.240}, {"C-C:a", 1.398},
    {"C-N:a", 1.354}, {"C-O:a", 1.367}, {"C-S:a", 1.714}, {"N-N:a", 1.340}, {"C-SE:a", 1.855}};
  bool charged = chargeA != 0 || chargeB != 0;
  if (!charged) {
    std::string pair = a < b ? a + "-" + b : b + "-" + a;
    std::string bucket = aromatic ? "a" : order < 1.25 ? "s" : order < 1.75 ? "p" : "d";
    auto it = MEASURED.find(pair + ":" + bucket);
    if (it != MEASURED.end()) return it->second;
  }
  std::string ia = isoelectronic(a, chargeA), ib = isoelectronic(b, chargeB);
  if (ia != a || ib != b) return covalentRadius(ia, order, aromatic) + covalentRadius(ib, order, aromatic);
  double sum = covalentRadius(a, order, aromatic) + covalentRadius(b, order, aromatic);
  return sum - 0 * 0;    // POLARITY is 0: sum - 0 * |electronegativity difference|, which is sum exactly
}
inline double vanDerWaalsRadius(const std::string& s) {
  static const std::map<std::string, double> VDW = {{"H", 1.10}, {"C", 1.70}, {"N", 1.55}, {"O", 1.52}, {"F", 1.47}, {"SI", 2.10},
    {"P", 1.80}, {"S", 1.80}, {"CL", 1.75}, {"SE", 1.90}, {"BR", 1.83}, {"I", 1.98}, {"B", 1.92}, {"NA", 2.27}, {"MG", 1.73},
    {"K", 2.75}, {"CA", 2.31}, {"ZN", 1.39}, {"FE", 2.00}};
  auto it = VDW.find(s);
  return it == VDW.end() ? 1.8 : it->second;
}
// pi: 0 false, 1 true, 2 "two"
inline double idealAngle(int sigmaCount, const std::string& symbol, int pi) {
  auto degrees = [](double v) { return (v * M_PI) / 180; };
  if (sigmaCount >= 4) return degrees(109.47);
  if (sigmaCount == 3) {
    if (symbol == "S" || symbol == "SE" || symbol == "P" || symbol == "AS") return degrees(106.0);
    if (pi == 1) return degrees(120);
    if (symbol == "N" || symbol == "P" || symbol == "AS") return degrees(107.0);
    return degrees(109.47);
  }
  if (sigmaCount == 2) {
    if (pi == 2) return degrees(180);
    if (pi == 1) return degrees(120);
    if (symbol == "O" || symbol == "S" || symbol == "SE") return degrees(104.5);
    return degrees(109.47);
  }
  return M_PI;
}
inline double polygonAngle(int size) { return ((size - 2) * M_PI) / size; }
inline double lawOfCosines(double a, double b, double angle) { return std::sqrt(a * a + b * b - 2 * a * b * jsmath::cos(angle)); }
inline double substituentWeight(double order, bool aromatic) {
  if (aromatic) return 0.5;
  if (order >= 3) return 1.5;
  if (order > 1) return order - 1;
  return 0;
}

// ---------------------------------------------------------------- conformer.js
struct Bounds {
  std::vector<double> lower, upper;
  int n = 0;
  std::vector<std::vector<int>> rings;
  std::vector<int> sigma, pi;
};

inline int smallestSharedRing(const std::vector<std::vector<int>>& rings, int centre, int first, int second) {
  int best = -1;
  for (auto& r : rings) {
    auto has = [&](int a) { return std::find(r.begin(), r.end(), a) != r.end(); };
    if (!has(centre) || !has(first) || !has(second)) continue;
    if (best < 0 || (int)r.size() < best) best = (int)r.size();
  }
  return best;
}
inline bool ringBond(const std::vector<std::vector<int>>& rings, int a, int b) {
  for (auto& r : rings) {
    auto it = std::find(r.begin(), r.end(), a);
    if (it == r.end()) continue;
    int i = (int)(it - r.begin()), m = (int)r.size();
    if (r[(i + 1) % m] == b || r[(i - 1 + m) % m] == b) return true;
  }
  return false;
}
inline bool isPlanarRing(const Graph& g, const std::vector<int>& ring) {
  for (int atom : ring) {
    if (g.atoms[atom].aromatic) continue;
    int sigma = 0;
    bool hasDouble = false;
    for (auto& b : g.bonds) if (b.from == atom || b.to == atom) { ++sigma; if (b.geo() >= 2) hasDouble = true; }
    sigma += g.atoms[atom].hasHydrogens ? g.atoms[atom].hydrogens : 0;
    if (!(sigma == 3 && hasDouble)) return false;
  }
  return true;
}

inline Bounds distanceBounds(const Graph& g) {
  const double BOND_SLACK = 0.01, ANGLE_SLACK = 0.04, CHAIN_ANGLE_SLACK = (2.5 * M_PI) / 180, RING_ANGLE_SLACK = (1.2 * M_PI) / 180,
               RING_SLACK = 0.10, CLASH_FRACTION = 0.8, VSEPR_STRENGTH = 11;
  int n = (int)g.atoms.size();
  Bounds B;
  B.n = n;
  B.lower.assign((size_t)n * n, 0);
  B.upper.assign((size_t)n * n, 1000);
  auto lists = adjacency(g);
  B.rings = smallestRings(g);
  const auto& rings = B.rings;
  auto setBound = [&](int i, int j, double low, double high) {
    size_t at = (size_t)i * n + j, back = (size_t)j * n + i;
    double newLow = jmax(B.lower[at], low), newHigh = jmin(B.upper[at], high);
    B.lower[at] = newLow; B.lower[back] = newLow;
    B.upper[at] = newHigh; B.upper[back] = newHigh;
  };
  for (int i = 0; i < n; ++i) { B.lower[(size_t)i * n + i] = 0; B.upper[(size_t)i * n + i] = 0; }
  // topologicalDistance
  std::vector<int> hops((size_t)n * n, 1000000);
  for (int start = 0; start < n; ++start) {
    hops[(size_t)start * n + start] = 0;
    std::vector<int> queue = {start};
    for (size_t head = 0; head < queue.size(); ++head) {
      int atom = queue[head], next = hops[(size_t)start * n + atom] + 1;
      for (auto& s : lists[atom]) {
        if (hops[(size_t)start * n + s.atom] <= next) continue;
        hops[(size_t)start * n + s.atom] = next;
        queue.push_back(s.atom);
      }
    }
  }
  auto key = [](int a, int b) { return a < b ? std::make_pair(a, b) : std::make_pair(b, a); };
  std::map<std::pair<int, int>, double> bondDistance;
  for (auto& b : g.bonds) {
    double length = bondLength(g.atoms[b.from].symbol, g.atoms[b.to].symbol, b.geo(), b.aromatic, g.atoms[b.from].charge, g.atoms[b.to].charge);
    setBound(b.from, b.to, length - BOND_SLACK, length + BOND_SLACK);
    bondDistance[key(b.from, b.to)] = length;
  }
  auto distance = [&](int a, int b, double& out) {
    auto it = bondDistance.find(key(a, b));
    if (it == bondDistance.end()) return false;
    out = it->second;
    return true;
  };
  B.sigma.resize(n); B.pi.resize(n);
  for (int i = 0; i < n; ++i) {
    B.sigma[i] = (int)lists[i].size() + (g.atoms[i].hasHydrogens ? g.atoms[i].hydrogens : 0);
    if (g.atoms[i].aromatic) { B.pi[i] = 1; continue; }
    int extra = 0;
    for (auto& s : lists[i]) {
      double order = g.bonds[s.bond].geo();
      if (order >= 3) extra += 2;
      else if (order > 1) extra += 1;
    }
    B.pi[i] = extra >= 2 ? 2 : extra == 1 ? 1 : 0;
  }
  const auto& sigma = B.sigma;
  const auto& pi = B.pi;
  // cyclicRingAngles: the ring as a cyclic polygon of its own sides
  std::map<std::pair<int, int>, double> ringAngles;   // (ring size, atom) -> angle, the first ring that sets it
  for (auto& ring : rings) {
    int m = (int)ring.size();
    if (m < 3 || m > 6) continue;
    std::vector<double> sides;
    bool missing = false;
    for (int i = 0; i < m; ++i) { double d; if (!distance(ring[i], ring[(i + 1) % m], d)) { missing = true; break; } sides.push_back(d); }
    if (missing) continue;
    double longest = -INFINITY;
    for (double s : sides) longest = jmax(longest, s);
    double low = longest / 2, high = longest * m;
    auto turn = [&](double radius) {
      double total = 0;
      for (double side : sides) total = total + 2 * jsmath::asin(jmin(1, side / (2 * radius)));
      return total;
    };
    if (turn(high) > 2 * M_PI) continue;
    for (int step = 0; step < 60; ++step) {
      double middle = (low + high) / 2;
      if (turn(middle) > 2 * M_PI) low = middle; else high = middle;
    }
    double radius = (low + high) / 2;
    std::vector<double> half;
    for (double side : sides) half.push_back(jsmath::asin(jmin(1, side / (2 * radius))));
    for (int i = 0; i < m; ++i) {
      double before = half[(i - 1 + m) % m], after = half[i];
      auto k = std::make_pair(m, ring[i]);
      if (!ringAngles.count(k)) ringAngles[k] = M_PI - before - after;
    }
  }
  auto vseprAngle = [&](int centre, int first, int second) {
    double base = idealAngle(sigma[centre], g.atoms[centre].symbol, pi[centre]);
    const auto& nb = lists[centre];
    if (nb.size() < 3) return base;
    auto weightOf = [&](const Step& s) { return substituentWeight(g.bonds[s.bond].geo(), g.bonds[s.bond].aromatic); };
    double total = 0;
    for (auto& s : nb) total += weightOf(s);
    double mean = total / nb.size();
    const Step *a = nullptr, *b = nullptr;
    for (auto& s : nb) if (s.atom == first) { a = &s; break; }
    for (auto& s : nb) if (s.atom == second) { b = &s; break; }
    if (!a || !b) return base;
    double shift = (weightOf(*a) + weightOf(*b) - 2 * mean) * VSEPR_STRENGTH;
    double deg = jmax(60, jmin(180, (base * 180) / M_PI + shift));
    return (deg * M_PI) / 180;
  };
  struct Estimate { int first, second; double low, high; int shared; double da, db; };
  std::vector<Estimate> estimates;
  std::map<std::pair<int, int>, size_t> estimateAt;
  for (int centre = 0; centre < n; ++centre) {
    const auto& nb = lists[centre];
    for (size_t a = 0; a < nb.size(); ++a)
      for (size_t b = a + 1; b < nb.size(); ++b) {
        int first = nb[a].atom, second = nb[b].atom;
        double da, db;
        if (!distance(centre, first, da) || !distance(centre, second, db)) continue;
        int shared = smallestSharedRing(rings, centre, first, second);
        double angle;
        if (shared >= 0 && shared <= 5) {
          auto it = ringAngles.find({shared, centre});
          angle = it != ringAngles.end() ? it->second : polygonAngle(shared);
        } else {
          angle = vseprAngle(centre, first, second);
        }
        double d = lawOfCosines(da, db, angle);
        auto pair = key(first, second);
        auto it = estimateAt.find(pair);
        if (it == estimateAt.end()) {
          estimateAt[pair] = estimates.size();
          estimates.push_back({first, second, d, d, shared, da, db});
        } else {
          Estimate& s = estimates[it->second];
          s.low = jmin(s.low, d);
          s.high = jmax(s.high, d);
          s.shared = s.shared < 0 ? shared : shared < 0 ? s.shared : std::min(s.shared, shared);
        }
      }
  }
  for (auto& e : estimates) {
    double tolerance = e.shared >= 0 ? RING_ANGLE_SLACK : CHAIN_ANGLE_SLACK;
    auto spread = [&](double d, int sign) {
      double a = e.da, b = e.db;
      double cosine = (a * a + b * b - d * d) / (2 * a * b);
      double angle = jsmath::acos(jmax(-1, jmin(1, cosine)));
      double widened = jmax(0, jmin(M_PI, angle + sign * tolerance));
      return lawOfCosines(a, b, widened);
    };
    bool linear = e.high > 0.999 * (e.da + e.db);
    if (linear) setBound(e.first, e.second, e.da + e.db - 2 * BOND_SLACK, e.da + e.db + 2 * BOND_SLACK);
    else setBound(e.first, e.second, spread(e.low, -1), spread(e.high, 1));
  }
  for (auto& ring : rings) {
    int m = (int)ring.size();
    if (m > 7 || !isPlanarRing(g, ring)) continue;
    double total = 0;
    bool missing = false;
    for (int i = 0; i < m; ++i) { double d; if (!distance(ring[i], ring[(i + 1) % m], d)) { missing = true; break; } total += d; }
    if (missing) continue;
    double side = total / m, radius = side / (2 * jsmath::sin(M_PI / m));
    for (int a = 0; a < m; ++a)
      for (int b = a + 1; b < m; ++b) {
        int separation = std::min(b - a, m - (b - a));
        if (separation < 3) continue;
        double angle = (2 * M_PI * separation) / m, d = 2 * radius * jsmath::sin(angle / 2);
        setBound(ring[a], ring[b], d - RING_SLACK, d + RING_SLACK);
      }
  }
  auto ringAwareAngle = [&](int centre, int first, int second) {
    int shared = smallestSharedRing(rings, centre, first, second);
    return shared >= 0 && shared <= 5 ? polygonAngle(shared) : idealAngle(sigma[centre], g.atoms[centre].symbol, pi[centre]);
  };
  auto distanceAt = [&](int i, int j, int k, int l, double dihedral, double& out) {
    double dij, djk, dkl;
    if (!distance(i, j, dij) || !distance(j, k, djk) || !distance(k, l, dkl)) return false;
    double angleIJK = ringAwareAngle(j, i, k), angleJKL = ringAwareAngle(k, j, l);
    double kp0 = djk;
    double ip0 = dij * jsmath::cos(angleIJK), ip1 = dij * jsmath::sin(angleIJK), ip2 = 0;
    double lx = kp0 - dkl * jsmath::cos(angleJKL), radial = dkl * jsmath::sin(angleJKL);
    double lp0 = lx, lp1 = radial * jsmath::cos(dihedral), lp2 = radial * jsmath::sin(dihedral);
    out = jsmath::hypot(ip0 - lp0, ip1 - lp1, ip2 - lp2);
    return true;
  };
  for (auto& bond : g.bonds) {
    bool inRing = ringBond(rings, bond.from, bond.to);
    bool rotatable = bond.geo() < 1.5 && !inRing;
    for (auto& first : lists[bond.from]) {
      if (first.atom == bond.to) continue;
      for (auto& second : lists[bond.to]) {
        if (second.atom == bond.from || second.atom == first.atom) continue;
        double cis, trans;
        if (!distanceAt(first.atom, bond.from, bond.to, second.atom, 0, cis)) continue;
        if (!distanceAt(first.atom, bond.from, bond.to, second.atom, M_PI, trans)) continue;
        if (rotatable || !inRing) setBound(first.atom, second.atom, jmin(cis, trans) - ANGLE_SLACK, jmax(cis, trans) + ANGLE_SLACK);
      }
    }
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      if (hops[(size_t)i * n + j] < 4) continue;
      double clash = CLASH_FRACTION * (vanDerWaalsRadius(g.atoms[i].symbol) + vanDerWaalsRadius(g.atoms[j].symbol));
      double capped = jmin(clash, B.upper[(size_t)i * n + j]);
      if (capped > B.lower[(size_t)i * n + j]) { B.lower[(size_t)i * n + j] = capped; B.lower[(size_t)j * n + i] = capped; }
    }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      if (B.lower[(size_t)i * n + j] > 0.1) continue;
      double floor = jmin(1.0, B.upper[(size_t)i * n + j] * 0.5);
      B.lower[(size_t)i * n + j] = floor; B.lower[(size_t)j * n + i] = floor;
    }
  return B;
}

inline int smoothBounds(Bounds& B) {
  int n = B.n;
  auto& lower = B.lower;
  auto& upper = B.upper;
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < n; ++i) {
      double ik = upper[(size_t)i * n + k];
      for (int j = i + 1; j < n; ++j) {
        double through = ik + upper[(size_t)k * n + j];
        if (through < upper[(size_t)i * n + j]) { upper[(size_t)i * n + j] = through; upper[(size_t)j * n + i] = through; }
      }
    }
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        double a = lower[(size_t)i * n + k] - upper[(size_t)k * n + j], b = lower[(size_t)k * n + j] - upper[(size_t)i * n + k];
        double best = jmax(a, b);
        if (best > lower[(size_t)i * n + j]) { lower[(size_t)i * n + j] = best; lower[(size_t)j * n + i] = best; }
      }
  int contradictions = 0;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (lower[(size_t)i * n + j] > upper[(size_t)i * n + j] + 1e-9) ++contradictions;
  return contradictions;
}

inline std::vector<std::array<int, 4>> planarQuadruples(const Graph& g, const std::vector<std::vector<int>>& rings) {
  std::vector<std::array<int, 4>> q;
  for (auto& r : rings) {
    int m = (int)r.size();
    if (m < 4 || m > 7 || !isPlanarRing(g, r)) continue;
    for (int i = 0; i < m; ++i) q.push_back({r[i], r[(i + 1) % m], r[(i + 2) % m], r[(i + 3) % m]});
  }
  auto lists = adjacency(g);
  for (int index = 0; index < (int)g.atoms.size(); ++index) {
    const Atom& a = g.atoms[index];
    int sigma = (int)lists[index].size() + (a.hasHydrogens ? a.hydrogens : 0);
    if (sigma != 3 || lists[index].size() != 3) continue;
    if (a.symbol == "S" || a.symbol == "SE" || a.symbol == "P" || a.symbol == "AS") continue;
    bool hasPi = a.aromatic;
    for (auto& s : lists[index]) if (g.bonds[s.bond].geo() > 1) hasPi = true;
    if (!hasPi) continue;
    q.push_back({lists[index][0].atom, lists[index][1].atom, lists[index][2].atom, index});
  }
  return q;
}

inline std::vector<std::array<int, 3>> linearTriples(const Graph& g, const std::vector<int>& sigma, const std::vector<int>& pi) {
  auto lists = adjacency(g);
  std::vector<std::array<int, 3>> t;
  for (int index = 0; index < (int)g.atoms.size(); ++index) {
    if (lists[index].size() != 2) continue;
    if (idealAngle(sigma[index], g.atoms[index].symbol, pi[index]) < M_PI - 0.01) continue;
    t.push_back({lists[index][0].atom, index, lists[index][1].atom});
  }
  return t;
}

// generator(seed): xorshift - with JavaScript's ARITHMETIC `>>` in its middle step, mirrored
struct Xorshift {
  uint32_t state;
  explicit Xorshift(double seed) {
    uint32_t s = (uint32_t)(int64_t)std::fmod(std::trunc(seed), 4294967296.0);
    state = s ? s : 1;
  }
  double operator()() {
    state ^= state << 13;
    state ^= (uint32_t)((int32_t)state >> 17);
    state ^= state << 5;
    return state / 4294967296.0;
  }
};

inline std::vector<double> embedBounds(const Bounds& B, double seed) {
  int n = B.n;
  Xorshift random(seed);
  if (n == 1) return std::vector<double>(3, 0);
  std::vector<double> chosen((size_t)n * n, 0);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      double low = B.lower[(size_t)i * n + j], high = jmin(B.upper[(size_t)i * n + j], low + 50);
      double value = low + (high - low) * random();
      chosen[(size_t)i * n + j] = value; chosen[(size_t)j * n + i] = value;
    }
  auto squared = [&](int i, int j) { return chosen[(size_t)i * n + j] * chosen[(size_t)i * n + j]; };
  double total = 0;
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) total += squared(i, j);
  total /= 2.0 * n * n;
  std::vector<double> toCentre(n);
  for (int i = 0; i < n; ++i) {
    double row = 0;
    for (int j = 0; j < n; ++j) row += squared(i, j);
    toCentre[i] = row / n - total;
  }
  std::vector<double> metric((size_t)n * n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) metric[(size_t)i * n + j] = (toCentre[i] + toCentre[j] - squared(i, j)) / 2;
  double shift = 0;
  for (int i = 0; i < n; ++i) {
    double row = 0;
    for (int j = 0; j < n; ++j) row += std::fabs(metric[(size_t)i * n + j]);
    if (row > shift) shift = row;
  }
  std::vector<double> coordinates(3 * (size_t)n, 0), work((size_t)n * n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) work[(size_t)i * n + j] = metric[(size_t)i * n + j] + (i == j ? shift : 0);
  for (int axis = 0; axis < 3; ++axis) {
    // dominantEigenvector
    std::vector<double> vector(n);
    for (int i = 0; i < n; ++i) vector[i] = random() - 0.5;
    { double length = hypotOf(vector); if (length > 0) for (auto& v : vector) v /= length; }
    double value = 0;
    for (int step = 0; step < 200; ++step) {
      std::vector<double> next(n);
      for (int i = 0; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < n; ++j) sum += work[(size_t)i * n + j] * vector[j];
        next[i] = sum;
      }
      double length = hypotOf(next);
      if (length < 1e-12) break;
      for (auto& v : next) v /= length;
      double change = 0;
      for (int i = 0; i < n; ++i) change = jmax(change, std::fabs(next[i] - vector[i]));
      vector = next;
      value = length;
      if (change < 1e-10) break;
    }
    (void)value;
    double quotient = 0;
    for (int i = 0; i < n; ++i) {
      double sum = 0;
      for (int j = 0; j < n; ++j) sum += work[(size_t)i * n + j] * vector[j];
      quotient += vector[i] * sum;
    }
    double original = quotient - shift, scale = original > 0 ? std::sqrt(original) : 0;
    for (int i = 0; i < n; ++i) coordinates[(size_t)i * 3 + axis] = scale > 0 ? vector[i] * scale : (random() - 0.5) * 0.5;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) work[(size_t)i * n + j] -= quotient * vector[i] * vector[j];
  }
  return coordinates;
}

inline double projectOntoBounds(std::vector<double>& p, const Bounds& B, int rounds) {
  int n = B.n;
  double worst = 0;
  for (int round = 0; round < rounds; ++round) {
    worst = 0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        double dx = p[i * 3] - p[j * 3], dy = p[i * 3 + 1] - p[j * 3 + 1], dz = p[i * 3 + 2] - p[j * 3 + 2];
        double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        double low = B.lower[(size_t)i * n + j], high = B.upper[(size_t)i * n + j], target;
        if (distance < low) target = low;
        else if (distance > high) target = high;
        else continue;
        double violation = std::fabs(target - distance);
        if (violation > worst) worst = violation;
        if (distance < 1e-9) { p[i * 3] += 0.01; continue; }
        double scale = (0.5 * (target - distance)) / distance;
        p[i * 3] += scale * dx; p[i * 3 + 1] += scale * dy; p[i * 3 + 2] += scale * dz;
        p[j * 3] -= scale * dx; p[j * 3 + 1] -= scale * dy; p[j * 3 + 2] -= scale * dz;
      }
    if (worst < 1e-6) break;
  }
  return worst;
}

inline double signedVolume(const std::vector<double>& p, int a, int b, int c, int d) {
  double ax = p[a * 3] - p[d * 3], ay = p[a * 3 + 1] - p[d * 3 + 1], az = p[a * 3 + 2] - p[d * 3 + 2];
  double bx = p[b * 3] - p[d * 3], by = p[b * 3 + 1] - p[d * 3 + 1], bz = p[b * 3 + 2] - p[d * 3 + 2];
  double cx = p[c * 3] - p[d * 3], cy = p[c * 3 + 1] - p[d * 3 + 1], cz = p[c * 3 + 2] - p[d * 3 + 2];
  return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
}
inline void cross3(const double* p, const double* q, double* r) {
  r[0] = p[1] * q[2] - p[2] * q[1]; r[1] = p[2] * q[0] - p[0] * q[2]; r[2] = p[0] * q[1] - p[1] * q[0];
}
inline void addVolumeGradient(const std::vector<double>& p, int a, int b, int c, int d, double push, std::vector<double>& gradient) {
  double u[3], v[3], w[3];
  for (int k = 0; k < 3; ++k) { u[k] = p[a * 3 + k] - p[d * 3 + k]; v[k] = p[b * 3 + k] - p[d * 3 + k]; w[k] = p[c * 3 + k] - p[d * 3 + k]; }
  double da[3], db[3], dc[3];
  cross3(v, w, da); cross3(w, u, db); cross3(u, v, dc);
  for (int k = 0; k < 3; ++k) {
    gradient[a * 3 + k] += push * da[k];
    gradient[b * 3 + k] += push * db[k];
    gradient[c * 3 + k] += push * dc[k];
    gradient[d * 3 + k] -= push * (da[k] + db[k] + dc[k]);
  }
}

struct Restraints { std::vector<std::array<int, 4>> planar; std::vector<std::array<int, 3>> linear; std::vector<Centre> chiral; };

inline double boundsError(const std::vector<double>& p, const Bounds& B, const Restraints& R, std::vector<double>* gradient) {
  const double PLANARITY_WEIGHT = 0.5, LINEARITY_WEIGHT = 0.5;
  int n = B.n;
  if (gradient) std::fill(gradient->begin(), gradient->end(), 0.0);
  double error = 0;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      double dx = p[i * 3] - p[j * 3], dy = p[i * 3 + 1] - p[j * 3 + 1], dz = p[i * 3 + 2] - p[j * 3 + 2];
      double squared = dx * dx + dy * dy + dz * dz;
      double high = B.upper[(size_t)i * n + j], low = B.lower[(size_t)i * n + j];
      double scale = 0;
      if (squared > high * high) {
        double over = squared / (high * high) - 1;
        error += over * over;
        scale = (4 * over) / (high * high);
      } else if (squared < low * low) {
        double ratio = (2 * low * low) / (low * low + squared), over = ratio - 1;
        error += over * over;
        scale = (-2 * over * ratio * ratio) / (low * low);
      }
      if (gradient && scale != 0) {
        auto& gr = *gradient;
        gr[i * 3] += scale * dx; gr[i * 3 + 1] += scale * dy; gr[i * 3 + 2] += scale * dz;
        gr[j * 3] -= scale * dx; gr[j * 3 + 1] -= scale * dy; gr[j * 3 + 2] -= scale * dz;
      }
    }
  for (auto& q : R.planar) {
    double volume = signedVolume(p, q[0], q[1], q[2], q[3]);
    error += volume * volume * PLANARITY_WEIGHT;
    if (gradient) addVolumeGradient(p, q[0], q[1], q[2], q[3], 2 * volume * PLANARITY_WEIGHT, *gradient);
  }
  for (auto& t : R.linear) {
    int a = t[0], centre = t[1], b = t[2];
    double u[3], v[3], w[3];
    for (int k = 0; k < 3; ++k) { u[k] = p[a * 3 + k] - p[centre * 3 + k]; v[k] = p[b * 3 + k] - p[centre * 3 + k]; }
    cross3(u, v, w);
    error += (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) * LINEARITY_WEIGHT;
    if (gradient) {
      double du[3], dv[3];
      cross3(v, w, du); cross3(w, u, dv);
      for (int k = 0; k < 3; ++k) {
        double pushU = 2 * LINEARITY_WEIGHT * du[k], pushV = 2 * LINEARITY_WEIGHT * dv[k];
        (*gradient)[a * 3 + k] += pushU;
        (*gradient)[b * 3 + k] += pushV;
        (*gradient)[centre * 3 + k] -= pushU + pushV;
      }
    }
  }
  for (auto& c : R.chiral) {
    double volume = signedVolume(p, c.neighbours[0], c.neighbours[1], c.neighbours[2], c.neighbours[3]);
    double wanted = c.sign, target = 0.4;
    if (volume * wanted >= target) continue;
    double shortBy = target - volume * wanted;
    error += shortBy * shortBy * 4;
    if (gradient) addVolumeGradient(p, c.neighbours[0], c.neighbours[1], c.neighbours[2], c.neighbours[3], -8 * shortBy * wanted, *gradient);
  }
  return error;
}

struct Refined { std::vector<double> coordinates; double error; int steps; };
inline Refined refineCoordinates(const std::vector<double>& coordinates, const Bounds& B, const Restraints& R) {
  int n = B.n, maxSteps = 400;
  std::vector<double> point = coordinates;
  projectOntoBounds(point, B, 1000);
  std::vector<double> gradient(3 * (size_t)n, 0);
  double step = 0.05;
  double error = boundsError(point, B, R, &gradient);
  int taken = 0, restarts = 0;
  for (; taken < maxSteps; ++taken) {
    double size = hypotOf(gradient);
    if (size < 1e-9) break;
    std::vector<double> trial(3 * (size_t)n, 0);
    bool improved = false;
    for (int attempt = 0; attempt < 20; ++attempt) {
      for (int i = 0; i < 3 * n; ++i) trial[i] = point[i] - (step / size) * gradient[i];
      double next = boundsError(trial, B, R, nullptr);
      if (next < error) { point = trial; error = next; step *= 1.3; improved = true; break; }
      step *= 0.4;
    }
    if (!improved) {
      if (restarts >= 4) break;
      ++restarts;
      step = 0.05;
      continue;
    }
    boundsError(point, B, R, &gradient);
    if (error < 1e-8) break;
  }
  return {point, error, taken};
}

// ---------------------------------------------------------------- component.js
inline uint32_t hashOf(const std::string& text) {      // FNV-1a over the UTF-16 code units (ASCII here)
  uint32_t hash = 0x811c9dc5u;
  for (unsigned char c : text) { hash ^= c; hash *= 0x01000193u; }
  return hash ? hash : 1;
}

inline Component smilesComponent(const std::string& smiles, const std::string& code) {
  Graph g = parseSmiles(smiles);
  Bounds bounds = distanceBounds(g);
  int contradictions = smoothBounds(bounds);
  if (contradictions > 0)
    throw std::runtime_error(code + ": " + std::to_string(contradictions) + " distance bound" + (contradictions == 1 ? "" : "s")
                             + " contradict themselves after smoothing; this molecule's geometry rules disagree");
  Restraints R;
  R.chiral = chiralCentres(g);
  R.planar = planarQuadruples(g, bounds.rings);
  R.linear = linearTriples(g, bounds.sigma, bounds.pi);
  double seed = hashOf(smiles);
  const int attempts = 16;
  std::vector<std::vector<double>> starts;
  for (int attempt = 0; attempt < attempts; ++attempt) starts.push_back(embedBounds(bounds, seed + attempt * 7919.0));
  bool have = false;
  Refined best;
  for (auto& start : starts) {
    Refined one = refineCoordinates(start, bounds, R);
    if (!have || one.error < best.error) { best = one; have = true; }
    if (best.error < 1e-3) break;
  }
  Component c;
  c.code = code;
  c.smiles = true;
  std::map<std::string, int> counts;
  const auto& symbols = elementSymbols();
  for (size_t i = 0; i < g.atoms.size(); ++i) {
    CompAtom a;
    a.element = g.atoms[i].element;
    a.charge = g.atoms[i].charge;
    auto round = [](double v) { return jsmath::round(v * 1000) / 1000; };
    a.x = round(best.coordinates[i * 3]); a.y = round(best.coordinates[i * 3 + 1]); a.z = round(best.coordinates[i * 3 + 2]);
    a.leaving = false;
    int e = g.atoms[i].element;
    std::string symbol = upper(e >= 1 && e <= (int)symbols.size() ? symbols[e - 1] : "X");
    a.name = symbol + std::to_string(++counts[symbol]);
    c.atoms.push_back(a);
  }
  for (auto& b : g.bonds)
    c.bonds.push_back({b.ring ? b.to : b.from, b.ring ? b.from : b.to, b.aromatic ? 4 : (int)jsmath::round(b.order)});
  return c;
}

}  // namespace lf::chem
