// Templates, as the page builds them: web/template-source.js buildTemplate (a PDB or mmCIF chain, aligned to the
// query by web/align.js's local alignment, or mapped by the job's own indices), shared/af3/featurise/
// template-input.js (the dense-24 and atom37 slots, merging), template-features.js (AF3's template geometry -
// pseudo-beta distogram, backbone frames, unit vectors - chai's distogram, rf3's CA histogram, boltz2's 109
// columns) and template-fused-features.js (protenix's 108, and the sparse rows every fused embedder reads).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd.h"
#include "featurise.h"
#include "jsmath.h"
#include "jsnum.h"
#include "tables.h"

namespace lf {

constexpr int GAP_AATYPE = 21, NUM_DENSE = 24;

inline const std::vector<std::string>& atom37Names() {
  static const std::vector<std::string> a = {"N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
    "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ",
    "CZ2", "CZ3", "NZ", "OXT"};
  return a;
}
inline char oneLetter(const std::string& three) {
  static const std::map<std::string, char> m = {{"ALA", 'A'}, {"ARG", 'R'}, {"ASN", 'N'}, {"ASP", 'D'}, {"CYS", 'C'}, {"GLN", 'Q'},
    {"GLU", 'E'}, {"GLY", 'G'}, {"HIS", 'H'}, {"ILE", 'I'}, {"LEU", 'L'}, {"LYS", 'K'}, {"MET", 'M'}, {"PHE", 'F'}, {"PRO", 'P'},
    {"SER", 'S'}, {"THR", 'T'}, {"TRP", 'W'}, {"TYR", 'Y'}, {"VAL", 'V'}};
  auto it = m.find(three);
  return it == m.end() ? 0 : it->second;
}

// ---------------------------------------------------------------- the structure's residues
struct TemplateResidue {
  std::string number;
  bool hasLabel = false; int labelSeq = 0;
  char code = 'X';
  std::vector<std::pair<std::string, std::array<double, 3>>> atoms;   // a Map: the first atom of a name kept
  double confidence = 0;
  int atomCount = 0;
  const std::array<double, 3>* atom(const std::string& name) const {
    for (auto& [n, p] : atoms) if (n == name) return &p;
    return nullptr;
  }
};
struct TemplateStructure { bool hasChain = false; std::string chain; std::vector<TemplateResidue> residues; std::string sequence; };

// chosenAltLocs: each residue's alternate location of highest mean occupancy (ties to the first in name order)
struct AltAtom { std::string key, altLoc; double occupancy; };
inline std::map<std::string, std::string> chosenAltLocs(const std::vector<AltAtom>& atoms) {
  std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::array<double, 2>>>>> totals;
  for (auto& a : atoms) {
    if (a.altLoc.empty()) continue;
    auto t = std::find_if(totals.begin(), totals.end(), [&](auto& e) { return e.first == a.key; });
    if (t == totals.end()) { totals.push_back({a.key, {}}); t = totals.end() - 1; }
    auto p = std::find_if(t->second.begin(), t->second.end(), [&](auto& e) { return e.first == a.altLoc; });
    if (p == t->second.end()) t->second.push_back({a.altLoc, {a.occupancy, 1}});
    else { p->second[0] = p->second[0] + a.occupancy; p->second[1] += 1; }
  }
  std::map<std::string, std::string> chosen;
  for (auto& [key, per] : totals) {
    auto sorted = per;
    std::stable_sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first < b.first; });
    std::string best;
    double bestOccupancy = -INFINITY;
    for (auto& [alt, sc] : sorted) if (sc[0] / sc[1] > bestOccupancy) { best = alt; bestOccupancy = sc[0] / sc[1]; }
    chosen[key] = best;
  }
  return chosen;
}

// chainResidues over shared/heads/superpose-pdb.js coordinateAtoms: a PDB's ATOM/HETATM records by fixed column
inline TemplateStructure chainResidues(const std::string& text, const std::string* chain) {
  struct A { std::array<double, 3> p; std::string name; bool hasChain; char chain; std::string resName, residue; double b; bool hetero; std::string alt; double occ; };
  std::vector<A> atoms;
  size_t start = 0;
  while (start <= text.size()) {
    size_t end = text.find('\n', start);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(start, end - start);
    start = end + 1;
    auto slice = [&](size_t a, size_t b) { return a >= line.size() ? std::string() : line.substr(a, b - a); };
    std::string record = slice(0, 6);
    if (record != "ATOM  " && record != "HETATM") { if (end == text.size()) break; continue; }
    double x = jsNumberOf(slice(30, 38)), y = jsNumberOf(slice(38, 46)), z = jsNumberOf(slice(46, 54));
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) { if (end == text.size()) break; continue; }
    A a;
    a.p = {x, y, z};
    a.name = trimWs(slice(12, 16));
    a.hasChain = line.size() > 21; a.chain = a.hasChain ? line[21] : 0;
    a.resName = trimWs(slice(17, 20));
    a.residue = trimWs(slice(22, 27));
    double b = jsNumberOf(slice(60, 66));
    a.b = std::isfinite(b) ? b : 0;
    a.hetero = record == "HETATM";
    a.alt = line.size() > 16 ? trimWs(std::string(1, line[16])) : "";
    double occ = jsNumberOf(slice(54, 60));
    a.occ = std::isfinite(occ) ? occ : 1;
    atoms.push_back(a);
    if (end == text.size()) break;
  }
  TemplateStructure s;
  bool hasWanted = false;
  char wanted = 0;
  if (chain) { hasWanted = true; wanted = chain->empty() ? 0 : (*chain)[0]; }
  else if (!atoms.empty()) {
    int first = -1;
    for (size_t i = 0; i < atoms.size(); ++i) if (!atoms[i].hetero) { first = (int)i; break; }
    const A& w = atoms[first >= 0 ? first : 0];
    hasWanted = w.hasChain; wanted = w.chain;
  }
  bool wantedNamesOne = !chain || chain->size() == 1;      // a chain id of two letters matches no PDB column
  // (a record too short for the column has no chain - undefined - which matches an undefined wanted chain)
  auto inChain = [&](const A& a) { return wantedNamesOne && hasWanted == a.hasChain && (!hasWanted || a.chain == wanted); };
  std::vector<AltAtom> alts;
  for (auto& a : atoms) alts.push_back({a.residue, inChain(a) ? a.alt : "", a.occ});
  auto keep = chosenAltLocs(alts);
  std::vector<std::string> order;
  for (auto& a : atoms) {
    if (!inChain(a)) continue;
    if (!a.alt.empty()) { auto k = keep.find(a.residue); if (k == keep.end() || k->second != a.alt) continue; }
    if (a.hetero && a.resName != "MSE") continue;
    auto r = std::find_if(s.residues.begin(), s.residues.end(), [&](const TemplateResidue& x) { return x.number == a.residue; });
    if (r == s.residues.end()) {
      TemplateResidue nr;
      nr.number = a.residue;
      char one = oneLetter(a.resName);
      nr.code = one ? one : (a.resName == "MSE" ? 'M' : 'X');
      s.residues.push_back(nr);
      r = s.residues.end() - 1;
    }
    if (!r->atom(a.name)) {
      r->atoms.push_back({a.name, a.p});
      r->atomCount += 1;
      r->confidence += (a.b - r->confidence) / r->atomCount;
    }
  }
  s.hasChain = hasWanted;
  s.chain = hasWanted ? std::string(1, wanted) : "";
  if (chain) s.chain = *chain;
  for (auto& r : s.residues) s.sequence += r.code;
  return s;
}

// cpu/design/mpnn/pdb.js parseCIFAtoms: the first _atom_site loop's first model
struct CifAtom { bool hetero; std::string name, altLoc, resName, chain; double resSeq, labelSeq; std::string iCode; double x, y, z, occupancy; };
inline std::vector<CifAtom> parseCifAtoms(const std::string& text) {
  std::vector<std::string> lines;
  { size_t s = 0, e; while ((e = text.find('\n', s)) != std::string::npos) { lines.push_back(text.substr(s, e - s)); s = e + 1; } lines.push_back(text.substr(s)); }
  std::vector<CifAtom> atoms;
  static const std::regex TOKEN(R"('[^']*'|"[^"]*"|\S+)");
  size_t i = 0;
  while (i < lines.size()) {
    if (trimWs(lines[i]) != "loop_") { ++i; continue; }
    size_t j = i + 1;
    std::vector<std::string> columns;
    while (j < lines.size() && trimWs(lines[j]).compare(0, 1, "_") == 0) {
      std::string t = trimWs(lines[j]);
      columns.push_back(t.substr(0, t.find_first_of(" \t\r\n\f\v")));
      ++j;
    }
    if (columns.empty() || columns[0].compare(0, 11, "_atom_site.") != 0) { i = j; continue; }
    std::map<std::string, size_t> col;
    for (size_t k = 0; k < columns.size(); ++k) {
      std::string c = columns[k];
      size_t at = c.find("_atom_site.");
      if (at != std::string::npos) c.erase(at, 11);
      col[c] = k;
    }
    bool haveFirst = false;
    std::string firstModel;
    for (; j < lines.size(); ++j) {
      std::string line = trimWs(lines[j]);
      if (line.empty() || line[0] == '#' || line.compare(0, 5, "loop_") == 0) break;
      std::vector<std::string> row;
      for (auto it = std::sregex_iterator(line.begin(), line.end(), TOKEN); it != std::sregex_iterator(); ++it) {
        std::string t = it->str();
        if (!t.empty() && (t[0] == '\'' || t[0] == '"')) t.erase(0, 1);
        if (!t.empty() && (t.back() == '\'' || t.back() == '"')) t.pop_back();
        row.push_back(t);
      }
      if (row.empty() || row.size() < columns.size()) continue;
      auto pick = [&](const std::string& key, const std::string& fallback) { auto it = col.find(key); return it == col.end() ? fallback : row[it->second]; };
      std::string model = pick("pdbx_PDB_model_num", "1");
      if (!haveFirst) { firstModel = model; haveFirst = true; }
      if (model != firstModel) break;
      CifAtom a;
      std::string iCode = pick("pdbx_PDB_ins_code", ""), altLoc = pick("label_alt_id", "");
      a.hetero = pick("group_PDB", "ATOM") == "HETATM";
      a.name = pick("auth_atom_id", pick("label_atom_id", ""));
      a.altLoc = altLoc == "." || altLoc == "?" ? "" : altLoc;
      a.resName = upper(pick("auth_comp_id", pick("label_comp_id", "")));
      a.chain = pick("auth_asym_id", pick("label_asym_id", "A"));
      a.resSeq = jsParseInt(pick("auth_seq_id", pick("label_seq_id", "0")));
      a.labelSeq = jsParseInt(pick("label_seq_id", "."));
      a.iCode = iCode == "." || iCode == "?" ? "" : iCode;
      a.x = jsParseFloat(pick("Cartn_x", "0")); a.y = jsParseFloat(pick("Cartn_y", "0")); a.z = jsParseFloat(pick("Cartn_z", "0"));
      double occ = jsParseFloat(pick("occupancy", "1"));
      a.occupancy = std::isnan(occ) || occ == 0 ? 0 : occ;
      atoms.push_back(a);
    }
    i = j;
  }
  return atoms;
}

inline TemplateStructure residuesFromCif(const std::string& text, const std::string* chain) {
  std::vector<CifAtom> atoms;
  for (auto& a : parseCifAtoms(text)) if ((!a.hetero || a.resName == "MSE") && a.occupancy > 0 && std::isfinite(a.x)) atoms.push_back(a);
  TemplateStructure s;
  bool hasWanted = chain != nullptr || !atoms.empty();
  std::string wanted = chain ? *chain : (atoms.empty() ? "" : atoms[0].chain);
  auto keyOf = [](const CifAtom& a) { return jsNumber(a.resSeq) + a.iCode; };
  std::vector<AltAtom> alts;
  for (auto& a : atoms) if (hasWanted && a.chain == wanted) alts.push_back({keyOf(a), a.altLoc, a.occupancy});
  auto keep = chosenAltLocs(alts);
  for (auto& a : atoms) {
    if (!hasWanted || a.chain != wanted) continue;
    if (!a.altLoc.empty()) { auto k = keep.find(keyOf(a)); if (k == keep.end() || k->second != a.altLoc) continue; }
    std::string number = keyOf(a);
    auto r = std::find_if(s.residues.begin(), s.residues.end(), [&](const TemplateResidue& x) { return x.number == number; });
    if (r == s.residues.end()) {
      TemplateResidue nr;
      nr.number = number;
      nr.hasLabel = std::isfinite(a.labelSeq) && std::floor(a.labelSeq) == a.labelSeq; nr.labelSeq = nr.hasLabel ? (int)a.labelSeq : 0;
      char one = oneLetter(a.resName);
      nr.code = one ? one : (a.resName == "MSE" ? 'M' : 'X');
      s.residues.push_back(nr);
      r = s.residues.end() - 1;
    }
    if (!r->atom(a.name)) r->atoms.push_back({a.name, {a.x, a.y, a.z}});
  }
  s.hasChain = hasWanted;
  s.chain = wanted;
  for (auto& r : s.residues) s.sequence += r.code;
  return s;
}

// ---------------------------------------------------------------- the query map
// web/align.js alignPositions: Smith-Waterman, match 2 / mismatch -1 / gap -2, ties to the diagonal, then up
inline std::vector<std::pair<int, int>> alignPositions(const std::string& a, const std::string& b) {
  int n = (int)a.size(), m = (int)b.size();
  if (n == 0 || m == 0) return {};
  std::vector<double> score((size_t)(n + 1) * (m + 1), 0);
  std::vector<uint8_t> from((size_t)(n + 1) * (m + 1), 0);
  auto at = [&](int i, int j) { return (size_t)i * (m + 1) + j; };
  double bestScore = 0;
  int bestI = 0, bestJ = 0;
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= m; ++j) {
      double diagonal = score[at(i - 1, j - 1)] + (a[i - 1] == b[j - 1] ? 2 : -1);
      double up = score[at(i - 1, j)] - 2, left = score[at(i, j - 1)] - 2;
      double best = 0;
      int direction = 3;
      if (diagonal > best) { best = diagonal; direction = 0; }
      if (up > best) { best = up; direction = 1; }
      if (left > best) { best = left; direction = 2; }
      score[at(i, j)] = best;
      from[at(i, j)] = (uint8_t)direction;
      if (best > bestScore) { bestScore = best; bestI = i; bestJ = j; }
    }
  std::vector<std::pair<int, int>> pairs;
  int i = bestI, j = bestJ;
  while (i > 0 && j > 0) {
    int d = from[at(i, j)];
    if (d == 3) break;
    if (d == 0) { pairs.push_back({i - 1, j - 1}); --i; --j; }
    else if (d == 1) --i;
    else --j;
  }
  std::reverse(pairs.begin(), pairs.end());
  return pairs;
}

// a JS Map<number, number>, in insertion order (a re-set key keeps its place)
struct IndexMap {
  std::vector<std::pair<int, int>> e;
  void set(int k, int v) {
    for (auto& p : e) if (p.first == k) { p.second = v; return; }
    e.push_back({k, v});
  }
};

struct TemplateSlot {
  int slots = NUM_DENSE;
  std::vector<int> aatype; std::vector<float> atomPositions, atomMask;
  int covered = 0, atoms = 0;
};

// templateSlot / templateSlotAtom37: each mapped residue's atoms in the layout - dense-24 by the conformer's own atom
// order, atom37 by name - the aatype the residue's letter, the gap where nothing maps
inline TemplateSlot buildSlot(const TemplateStructure& s, int tokens, const IndexMap& map, int offset,
                              const std::function<int(int)>* tokenOf, bool atom37) {
  TemplateSlot t;
  t.slots = atom37 ? 37 : NUM_DENSE;
  t.aatype.assign(tokens, GAP_AATYPE);
  t.atomPositions.assign((size_t)tokens * t.slots * 3, 0);
  t.atomMask.assign((size_t)tokens * t.slots, 0);
  for (auto& [q, ti] : map.e) {
    int token = tokenOf ? (*tokenOf)(q) : q + offset;
    if (token < 0 || token >= tokens || ti < 0 || ti >= (int)s.residues.size()) continue;
    const TemplateResidue& r = s.residues[ti];
    t.aatype[token] = aatypeFor(r.code);
    auto place = [&](int slot, const std::string& name) {
      const auto* p = r.atom(name);
      if (!p) return;
      size_t base = ((size_t)token * t.slots + slot) * 3;
      t.atomPositions[base] = (float)(*p)[0]; t.atomPositions[base + 1] = (float)(*p)[1]; t.atomPositions[base + 2] = (float)(*p)[2];
      t.atomMask[(size_t)token * t.slots + slot] = 1;
      ++t.atoms;
    };
    if (atom37) for (int slot = 0; slot < 37; ++slot) place(slot, atom37Names()[slot]);
    else {
      const auto& layout = conformerFor(r.code, false);
      for (int slot = 0; slot < (int)layout.size() && slot < NUM_DENSE; ++slot) place(slot, layout[slot].name);
    }
    ++t.covered;
  }
  return t;
}

struct BuildOptions {
  std::string text;
  const std::string* chain = nullptr;
  std::string query;
  int tokens = 0, offset = 0;
  const std::function<int(int)>* tokenOf = nullptr;
  const std::vector<std::pair<int, int>>* mapping = nullptr;
  bool atom37 = false;
};
struct Built { TemplateSlot slot; int residues = 0, of = 0; std::string chain; };

inline bool looksLikeCif(const std::string& text) {
  static const std::regex HEAD(R"(^\s*(data_|#|loop_|_))", std::regex::multiline);
  std::string head = text.substr(0, 4096);
  return std::regex_search(head, HEAD) && text.find("_atom_site.") != std::string::npos;
}

inline Built buildTemplate(const BuildOptions& o) {
  TemplateStructure s = looksLikeCif(o.text) ? residuesFromCif(o.text, o.chain) : chainResidues(o.text, o.chain);
  if (s.residues.empty())
    throw std::runtime_error(o.chain == nullptr ? "that structure has no protein chain this can read" : "that structure has no chain " + *o.chain);
  IndexMap map;
  if (o.mapping) {
    std::map<int, int> byLabel;
    // (a JS Map: a later residue of the same label overwrites)
    for (size_t at = 0; at < s.residues.size(); ++at) if (s.residues[at].hasLabel) byLabel[s.residues[at].labelSeq - 1] = (int)at;
    bool labelled = !byLabel.empty();
    for (auto& [q, ti] : *o.mapping) {
      if (!o.query.empty() && q >= (int)o.query.size())
        throw std::runtime_error("template mapping: query index " + std::to_string(q) + " is past the " + std::to_string(o.query.size()) + "-residue chain");
      if (labelled) { auto it = byLabel.find(ti); if (it != byLabel.end()) map.set(q, it->second); continue; }
      if (ti >= (int)s.residues.size())
        throw std::runtime_error("template mapping: template index " + std::to_string(ti) + " is past its " + std::to_string(s.residues.size()) + " residues");
      map.set(q, ti);
    }
  } else if (o.query.empty() || o.query == s.sequence) {
    for (int i = 0; i < (int)s.residues.size(); ++i) map.set(i, i);
  } else {
    for (auto& [q, ti] : alignPositions(o.query, s.sequence)) map.set(q, ti);
  }
  Built b;
  b.slot = buildSlot(s, o.tokens, map, o.offset, o.tokenOf, o.atom37);
  if (b.slot.covered == 0) throw std::runtime_error("that template covers none of the chain");
  b.residues = b.slot.covered;
  b.of = !o.query.empty() ? (int)o.query.size() : (int)s.residues.size();
  b.chain = s.chain;
  return b;
}

// mergeTemplateSlots: several chains' dense slots as one, each token from the one part that covers it
inline TemplateSlot mergeTemplateSlots(const std::vector<TemplateSlot>& slots) {
  if (slots.empty()) throw std::runtime_error("no template slots to merge");
  if (slots.size() == 1) return slots[0];
  int tokens = (int)slots[0].aatype.size();
  TemplateSlot m;
  m.aatype.assign(tokens, GAP_AATYPE);
  m.atomPositions.assign((size_t)tokens * NUM_DENSE * 3, 0);
  m.atomMask.assign((size_t)tokens * NUM_DENSE, 0);
  std::vector<char> taken(tokens, 0);
  for (auto& s : slots) {
    if ((int)s.aatype.size() != tokens)
      throw std::runtime_error("template slots disagree on token count: " + std::to_string(s.aatype.size()) + " against " + std::to_string(tokens));
    for (int t = 0; t < tokens; ++t) {
      bool any = false;
      for (int k = 0; k < NUM_DENSE; ++k) if (s.atomMask[(size_t)t * NUM_DENSE + k] > 0) { any = true; break; }
      if (!any) continue;
      if (taken[t]) throw std::runtime_error("two template slots both cover token " + std::to_string(t));
      taken[t] = 1;
      m.aatype[t] = s.aatype[t];
      for (int k = 0; k < NUM_DENSE; ++k) {
        size_t at = (size_t)t * NUM_DENSE + k;
        m.atomMask[at] = s.atomMask[at];
        for (int x = 0; x < 3; ++x) m.atomPositions[at * 3 + x] = s.atomPositions[at * 3 + x];
        if (s.atomMask[at] > 0) ++m.atoms;
      }
      ++m.covered;
    }
  }
  return m;
}

// mergeAtom37Templates: AF2's one slot over a complex
inline TemplateSlot mergeAtom37Templates(const std::vector<Built>& built, int tokens) {
  TemplateSlot m;
  m.slots = 37;
  m.aatype.assign(tokens, GAP_AATYPE);
  m.atomPositions.assign((size_t)tokens * 37 * 3, 0);
  m.atomMask.assign((size_t)tokens * 37, 0);
  for (auto& b : built)
    for (int t = 0; t < tokens; ++t) {
      if (b.slot.aatype[t] == GAP_AATYPE) continue;
      if (m.aatype[t] != GAP_AATYPE) throw std::runtime_error("two templates cover token " + std::to_string(t));
      m.aatype[t] = b.slot.aatype[t];
      std::copy(b.slot.atomMask.begin() + (size_t)t * 37, b.slot.atomMask.begin() + (size_t)(t + 1) * 37, m.atomMask.begin() + (size_t)t * 37);
      std::copy(b.slot.atomPositions.begin() + (size_t)t * 111, b.slot.atomPositions.begin() + (size_t)(t + 1) * 111, m.atomPositions.begin() + (size_t)t * 111);
    }
  for (auto& b : built) { m.covered += b.slot.covered; m.atoms += b.slot.atoms; }
  return m;
}

// ---------------------------------------------------------------- template-features.js
inline int pseudoBetaSlot(int code) {
  static const int T[31] = {4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 22, 23, 14, 14, 21, 22, 13, 13, 0};
  return code >= 0 && code < 31 ? T[code] : T[20];
}
inline std::array<int, 3> backboneSlots(int code) {
  if (code < 0 || code >= 31) code = 20;
  if (code < 20) return {2, 1, 0};
  if (code == 20 || code == 21 || code == 30) return {0, 0, 0};
  return {12, 8, 6};
}

inline std::vector<float> multichainMaskFor(const std::vector<int>& asymId, int tokens, const std::vector<float>* coverage, bool spanChains) {
  std::vector<float> mask((size_t)tokens * tokens);
  for (int i = 0; i < tokens; ++i)
    for (int j = 0; j < tokens; ++j) {
      bool same = asymId[i] == asymId[j];
      bool spanned = spanChains && coverage && (*coverage)[i] > 0 && (*coverage)[j] > 0;
      mask[(size_t)i * tokens + j] = same || spanned ? 1 : 0;
    }
  return mask;
}
inline std::vector<float> coverageOf(const TemplateSlot& t, int tokens) {
  std::vector<float> c(tokens, 0);
  for (int token = 0; token < tokens; ++token)
    for (int k = 0; k < t.slots; ++k) if (t.atomMask[(size_t)token * t.slots + k] > 0) { c[token] = 1; break; }
  return c;
}

// pseudoBeta (dense layout): each token's pseudo-beta atom and whether it is there
inline void pseudoBeta(const std::vector<int>& aatype, const std::vector<float>& positions, const std::vector<float>& mask, int tokens,
                       std::vector<float>& out, std::vector<float>& present) {
  out.assign((size_t)tokens * 3, 0);
  present.assign(tokens, 0);
  for (int t = 0; t < tokens; ++t) {
    int slot = pseudoBetaSlot(aatype[t]);
    size_t base = ((size_t)t * NUM_DENSE + slot) * 3;
    out[t * 3] = positions[base]; out[t * 3 + 1] = positions[base + 1]; out[t * 3 + 2] = positions[base + 2];
    present[t] = mask[(size_t)t * NUM_DENSE + slot] > 0 ? 1 : 0;
  }
}

struct Geometry { std::vector<int> bin; int bins = 39; std::vector<float> pseudoBetaMask2d, unitVector, backboneMask2d; };

// templateGeometry (dense): each pair's distogram bin (-1 for none), masks and unit vector; chai's own distogram
inline Geometry templateGeometry(const TemplateSlot& t, const std::vector<float>& multichain, int tokens, bool chai) {
  const int DGRAM_BINS = 39;
  const double DGRAM_MIN = 3.25, DGRAM_MAX = 50.75;
  std::vector<float> positions((size_t)tokens * NUM_DENSE * 3);
  for (size_t slot = 0; slot < (size_t)tokens * NUM_DENSE; ++slot) {
    float keep = t.atomMask[slot] > 0 ? 1 : 0;
    for (int x = 0; x < 3; ++x) positions[slot * 3 + x] = (float)((double)t.atomPositions[slot * 3 + x] * keep);
  }
  std::vector<float> beta, betaMask;
  pseudoBeta(t.aatype, positions, t.atomMask, tokens, beta, betaMask);
  // backboneFrames
  std::vector<float> rot((size_t)tokens * 9), trans((size_t)tokens * 3), frameMask(tokens);
  for (int token = 0; token < tokens; ++token) {
    auto bb = backboneSlots(t.aatype[token]);
    int c = bb[0], b = bb[1], a = bb[2];
    auto at = [&](int slot, int axis) { return (double)positions[((size_t)token * NUM_DENSE + slot) * 3 + axis]; };
    frameMask[token] = (t.atomMask[(size_t)token * NUM_DENSE + a] > 0 && t.atomMask[(size_t)token * NUM_DENSE + b] > 0
                        && t.atomMask[(size_t)token * NUM_DENSE + c] > 0) ? 1 : 0;
    for (int x = 0; x < 3; ++x) trans[token * 3 + x] = (float)at(b, x);
    double e1[3] = {at(c, 0) - at(b, 0), at(c, 1) - at(b, 1), at(c, 2) - at(b, 2)};
    double v2[3] = {at(a, 0) - at(b, 0), at(a, 1) - at(b, 1), at(a, 2) - at(b, 2)};
    double n1 = jsmath::hypot(e1[0], e1[1], e1[2]);
    if (n1 == 0 || std::isnan(n1)) n1 = 1;
    for (double& v : e1) v /= n1;
    double dot = e1[0] * v2[0] + e1[1] * v2[1] + e1[2] * v2[2];
    double e2[3] = {v2[0] - dot * e1[0], v2[1] - dot * e1[1], v2[2] - dot * e1[2]};
    double n2 = jsmath::hypot(e2[0], e2[1], e2[2]);
    if (n2 == 0 || std::isnan(n2)) n2 = 1;
    for (double& v : e2) v /= n2;
    double e3[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    for (int x = 0; x < 3; ++x) {
      rot[token * 9 + x * 3] = (float)e1[x]; rot[token * 9 + x * 3 + 1] = (float)e2[x]; rot[token * 9 + x * 3 + 2] = (float)e3[x];
    }
  }
  Geometry g;
  size_t pairs = (size_t)tokens * tokens;
  g.pseudoBetaMask2d.assign(pairs, 0); g.backboneMask2d.assign(pairs, 0); g.unitVector.assign(pairs * 3, 0); g.bin.assign(pairs, -1);
  for (int i = 0; i < tokens; ++i)
    for (int j = 0; j < tokens; ++j) {
      size_t pair = (size_t)i * tokens + j;
      double chain = multichain[pair];
      g.pseudoBetaMask2d[pair] = (float)((double)betaMask[i] * betaMask[j] * chain);
      float backbone = (float)((double)frameMask[i] * frameMask[j] * chain);
      g.backboneMask2d[pair] = backbone;
      double dx = (double)trans[j * 3] - trans[i * 3], dy = (double)trans[j * 3 + 1] - trans[i * 3 + 1], dz = (double)trans[j * 3 + 2] - trans[i * 3 + 2];
      size_t row = (size_t)i * 9;
      double x = (double)rot[row] * dx + (double)rot[row + 3] * dy + (double)rot[row + 6] * dz;
      double y = (double)rot[row + 1] * dx + (double)rot[row + 4] * dy + (double)rot[row + 7] * dz;
      double z = (double)rot[row + 2] * dx + (double)rot[row + 5] * dy + (double)rot[row + 8] * dz;
      double norm = std::max(std::sqrt(x * x + y * y + z * z), 1e-6);
      g.unitVector[pair * 3] = (float)((x / norm) * backbone);
      g.unitVector[pair * 3 + 1] = (float)((y / norm) * backbone);
      g.unitVector[pair * 3 + 2] = (float)((z / norm) * backbone);
    }
  if (chai) {
    for (int i = 0; i < tokens; ++i)
      for (int j = 0; j < tokens; ++j) {
        size_t pair = (size_t)i * tokens + j;
        int cls = 38;
        if (g.pseudoBetaMask2d[pair] > 0) {
          double sq = 1e-10;
          for (int k = 0; k < 3; ++k) { double d = (double)beta[i * 3 + k] - beta[j * 3 + k]; sq += d * d; }
          float distance = (float)std::sqrt(sq);
          cls = 0;
          for (int e = 1; e < 38; ++e) if (distance > (float)(3.25 + (47.5 * e) / 37)) ++cls;
        }
        g.bin[pair] = cls;
      }
    return g;
  }
  double lower[39];
  for (int bin = 0; bin < DGRAM_BINS; ++bin) {
    double edge = DGRAM_MIN + (DGRAM_MAX - DGRAM_MIN) * ((double)bin / (DGRAM_BINS - 1));
    lower[bin] = edge * edge;
  }
  for (int i = 0; i < tokens; ++i)
    for (int j = 0; j < tokens; ++j) {
      size_t pair = (size_t)i * tokens + j;
      if (g.pseudoBetaMask2d[pair] == 0) continue;
      double dx = (double)beta[i * 3] - beta[j * 3], dy = (double)beta[i * 3 + 1] - beta[j * 3 + 1], dz = (double)beta[i * 3 + 2] - beta[j * 3 + 2];
      double d2 = dx * dx + dy * dy + dz * dz;
      for (int bin = 0; bin < DGRAM_BINS; ++bin) {
        double upperEdge = bin + 1 < DGRAM_BINS ? lower[bin + 1] : 1e8;
        if (d2 > lower[bin] && d2 < upperEdge) { g.bin[pair] = bin; break; }
      }
    }
  return g;
}

// A fused embedder's feature row for one pair (dense, `width` columns) - protenix's 108, boltz2's 109, rf3's 66
struct FusedRows {
  int width = 0, tokens = 0;
  std::function<void(int i, int j, std::vector<float>& row)> fill;   // row zeroed by the caller
};

inline FusedRows rosettafold3Rows(const TemplateSlot& t, const std::vector<float>& asymMask, int tokens) {
  std::vector<double> bounds;
  for (int at = 0; at < 30; ++at) bounds.push_back(1.0 + at * 0.1);
  for (int at = 0; at < 33; ++at) bounds.push_back(4.0 + at * 0.5);
  auto ca = std::make_shared<std::vector<float>>((size_t)tokens * 3);
  auto caMask = std::make_shared<std::vector<float>>(tokens);
  for (int token = 0; token < tokens; ++token) {
    int slot = backboneSlots(t.aatype[token])[1];
    (*caMask)[token] = t.atomMask[(size_t)token * NUM_DENSE + slot];
    for (int x = 0; x < 3; ++x) (*ca)[token * 3 + x] = t.atomPositions[((size_t)token * NUM_DENSE + slot) * 3 + x];
  }
  const double noiseLevel = -7.1886193961439764;   // (Math.log(1e-4 / 16) + 1.2) / 1.5, as V8 computes it
  FusedRows r;
  r.width = 66; r.tokens = tokens;
  r.fill = [=, &asymMask](int i, int j, std::vector<float>& row) {
    double has = (double)(*caMask)[i] * (*caMask)[j] * asymMask[(size_t)i * tokens + j];
    if (!(has > 0)) return;
    double squared = 1e-10;
    for (int x = 0; x < 3; ++x) { double d = (double)(*ca)[i * 3 + x] - (*ca)[j * 3 + x]; squared += d * d; }
    double distance = std::isfinite(squared) ? std::sqrt(squared) : 1e9;
    int bin = 0;
    for (double b : bounds) if (distance > b) ++bin;
    row[bin] = (float)has;
    row[64] = (float)has;
    row[65] = (float)(noiseLevel * has);
  };
  return r;
}

inline FusedRows boltz2Rows(const TemplateSlot& t, const std::vector<float>& asymMask, int tokens) {
  const int BINS = 38, RESTYPES = 33, WIDTH = BINS + 1 + 3 + 1 + RESTYPES * 2, restypeAt = BINS + 1 + 3 + 1;
  auto cb = std::make_shared<std::vector<float>>(), cbMask = std::make_shared<std::vector<float>>();
  pseudoBeta(t.aatype, t.atomPositions, t.atomMask, tokens, *cb, *cbMask);
  auto rotation = std::make_shared<std::vector<float>>((size_t)tokens * 9), translation = std::make_shared<std::vector<float>>((size_t)tokens * 3);
  auto frameMask = std::make_shared<std::vector<float>>(tokens), covered = std::make_shared<std::vector<float>>(tokens);
  auto ca = std::make_shared<std::vector<float>>((size_t)tokens * 3);
  for (int token = 0; token < tokens; ++token) {
    double any = 0;
    for (int k = 0; k < NUM_DENSE; ++k) any += t.atomMask[(size_t)token * NUM_DENSE + k];
    (*covered)[token] = any > 0 ? 1 : 0;
    auto bb = backboneSlots(t.aatype[token]);
    auto at = [&](int slot, int axis) { return (double)t.atomPositions[((size_t)token * NUM_DENSE + slot) * 3 + axis]; };
    (*frameMask)[token] = (float)((double)t.atomMask[(size_t)token * NUM_DENSE + bb[0]] * t.atomMask[(size_t)token * NUM_DENSE + bb[1]]
                                  * t.atomMask[(size_t)token * NUM_DENSE + bb[2]]);
    double v1[3], v2[3];
    for (int x = 0; x < 3; ++x) { v1[x] = at(bb[0], x) - at(bb[1], x); v2[x] = at(bb[2], x) - at(bb[1], x); }
    auto norm = [](const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 1e-10; };   // (x ** 2 is x * x in V8)
    double e1[3];
    for (int x = 0; x < 3; ++x) e1[x] = v1[x] / norm(v1);
    double dot = e1[0] * v2[0] + e1[1] * v2[1] + e1[2] * v2[2];
    double u2[3];
    for (int x = 0; x < 3; ++x) u2[x] = v2[x] - e1[x] * dot;
    double e2[3];
    for (int x = 0; x < 3; ++x) e2[x] = u2[x] / norm(u2);
    double e3[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    for (int d = 0; d < 3; ++d) {
      (*rotation)[token * 9 + d * 3] = (float)e1[d]; (*rotation)[token * 9 + d * 3 + 1] = (float)e2[d]; (*rotation)[token * 9 + d * 3 + 2] = (float)e3[d];
    }
    for (int x = 0; x < 3; ++x) { (*translation)[token * 3 + x] = (float)at(bb[1], x); (*ca)[token * 3 + x] = (float)at(bb[1], x); }
  }
  std::vector<double> edges;
  for (int i = 0; i < 37; ++i) edges.push_back(3.25 + (50.75 - 3.25) * i / 36);
  auto aatype = std::make_shared<std::vector<int>>(t.aatype);
  FusedRows r;
  r.width = WIDTH; r.tokens = tokens;
  r.fill = [=, &asymMask](int i, int j, std::vector<float>& row) {
    size_t pair = (size_t)i * tokens + j;
    double asym = asymMask[pair];
    double squared = 1e-10;
    for (int x = 0; x < 3; ++x) { double d = (double)(*cb)[i * 3 + x] - (*cb)[j * 3 + x]; squared += d * d; }
    double distance = std::sqrt(squared);
    int bin = 0;
    for (double e : edges) if (distance > e) ++bin;
    row[bin] = (float)asym;
    row[BINS] = (float)((double)(*cbMask)[i] * (*cbMask)[j] * asym);
    for (int k = 0; k < 3; ++k) {
      double value = 0;
      for (int d = 0; d < 3; ++d) value += (double)(*rotation)[j * 9 + d * 3 + k] * ((double)(*ca)[i * 3 + d] - (*translation)[j * 3 + d]);
      double sign = value > 0 ? 1 : value < 0 ? -1 : value;
      row[BINS + 1 + k] = (float)(sign * asym);
    }
    row[BINS + 4] = (float)((double)(*frameMask)[i] * (*frameMask)[j] * asym);
    int ti = (*covered)[i] != 0 ? (*aatype)[i] + 2 : 0, tj = (*covered)[j] != 0 ? (*aatype)[j] + 2 : 0;
    if (ti >= 0 && ti < RESTYPES) row[restypeAt + ti] = 1;
    if (tj >= 0 && tj < RESTYPES) row[restypeAt + RESTYPES + tj] = 1;
  };
  return r;
}

// protenix's 108: the nine-projection embedder's own features, concatenated
inline FusedRows protenixRows(const TemplateSlot& t, const std::vector<float>& mask, int tokens, int distogramBins, int restypes, int width) {
  if (distogramBins != 39) throw std::runtime_error("this dialect wants " + std::to_string(distogramBins) + " distogram bins and templateGeometry computes 39");
  auto g = std::make_shared<Geometry>(templateGeometry(t, mask, tokens, false));
  auto aatype = std::make_shared<std::vector<int>>(t.aatype);
  FusedRows r;
  r.width = width; r.tokens = tokens;
  int restypeI = distogramBins + 1, restypeJ = restypeI + restypes, vectorAt = restypeJ + restypes;
  r.fill = [=](int i, int j, std::vector<float>& row) {
    size_t pair = (size_t)i * tokens + j;
    if (g->bin[pair] >= 0) row[g->bin[pair]] = 1;
    row[distogramBins] = g->pseudoBetaMask2d[pair];
    int ci = (*aatype)[j], cj = (*aatype)[i];
    if (ci >= 0 && ci < restypes) row[restypeI + ci] = 1;
    if (cj >= 0 && cj < restypes) row[restypeJ + cj] = 1;
    for (int x = 0; x < 3; ++x) row[vectorAt + x] = g->unitVector[pair * 3 + x];
    row[vectorAt + 3] = g->backboneMask2d[pair];
  };
  return r;
}

// sparseTemplateFeatures over rows: K the most nonzeros any row has (at least 1), each row's (column, value) in column
// order, padding -1 / 0
inline void sparseRows(const std::function<void(size_t row, std::vector<float>& dense)>& rowAt, size_t rows, int width,
                       std::vector<int>& idx, std::vector<float>& val, int& K) {
  std::vector<float> dense(width);
  K = 1;
  for (size_t r = 0; r < rows; ++r) {
    std::fill(dense.begin(), dense.end(), 0.0f);
    rowAt(r, dense);
    int nz = 0;
    for (float v : dense) if (v != 0) ++nz;
    K = std::max(K, nz);
  }
  idx.assign(rows * K, -1);
  val.assign(rows * K, 0.0f);
  for (size_t r = 0; r < rows; ++r) {
    std::fill(dense.begin(), dense.end(), 0.0f);
    rowAt(r, dense);
    int k = 0;
    for (int c = 0; c < width; ++c) if (dense[c] != 0) { idx[r * K + k] = c; val[r * K + k] = dense[c]; ++k; }
  }
}

}  // namespace lf
