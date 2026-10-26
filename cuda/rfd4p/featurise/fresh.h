// A raw structure, PREPARED as upstream prepares a condition spec's `input:` that is not an
// annotated CIF (atomworks' `parse` with RCSB_WITH_METALS and no assembly) - a line-for-line port
// of shared/rfd4p/pipeline/fresh.js, which carries the account of why each step exists.
//
// 🔴 THE MODEL SEES WHAT THIS DOES, SO IT IS NOT A CLEAN-UP: unmodeled residues and truncated side
// chains added without coordinates, one alternate conformer per residue chosen by upstream's seeded
// solver, waters, hydrogens and unbonded crystallization aids dropped, MSE as MET, each residue in
// the dictionary's atom order with the dictionary's charges. It rewrites the `_atom_site` rows of a
// block and hands the block back, so the reader that follows derives the chemistry as it does for
// any other file. tools/check-rfd4p-load.py holds the page to atomworks; gate.py holds this to the
// page byte for byte.
#pragma once
#include "cif.h"
#include "../../featurise/ccd.h"
#include <charconv>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include "../../featurise/jsmath.h"

namespace rfd4p {
namespace fresh {

using std::string;
using std::vector;

/** Python's `random.Random(seed)`, as far as atomworks draws from it (pyrandom.js). */
struct PyRandom {
  uint32_t mt[624];
  int mti = 625;
  explicit PyRandom(uint64_t seed) {
    vector<uint32_t> key;
    do { key.push_back((uint32_t)(seed & 0xffffffffu)); seed >>= 32; } while (seed);
    initByArray(key);
  }
  void initGenrand(uint32_t s) {
    mt[0] = s;
    for (int i = 1; i < 624; ++i) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint32_t)i;
    mti = 624;
  }
  void initByArray(const vector<uint32_t>& key) {
    initGenrand(19650218u);
    int i = 1, j = 0;
    for (int k = std::max<int>(624, (int)key.size()); k > 0; --k) {
      mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) + key[j] + (uint32_t)j;
      ++i; ++j;
      if (i >= 624) { mt[0] = mt[623]; i = 1; }
      if (j >= (int)key.size()) j = 0;
    }
    for (int k = 623; k > 0; --k) {
      mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) - (uint32_t)i;
      ++i;
      if (i >= 624) { mt[0] = mt[623]; i = 1; }
    }
    mt[0] = 0x80000000u;
  }
  uint32_t genrand() {
    if (mti >= 624) {
      auto mag = [](uint32_t y) { return (y & 1u) ? 0x9908b0dfu : 0u; };
      int kk = 0;
      for (; kk < 624 - 397; ++kk) {
        const uint32_t y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
        mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag(y);
      }
      for (; kk < 623; ++kk) {
        const uint32_t y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
        mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag(y);
      }
      const uint32_t y = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
      mt[623] = mt[396] ^ (y >> 1) ^ mag(y);
      mti = 0;
    }
    uint32_t y = mt[mti++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
  }
  uint32_t getrandbits(int k) { return genrand() >> (32 - k); }
  uint32_t randbelow(uint32_t n) {
    int k = 0;
    for (uint32_t v = n; v; v >>= 1) ++k;
    uint32_t r = getrandbits(k);
    while (r >= n) r = getrandbits(k);
    return r;
  }
  template <class T>
  void shuffle(vector<T>& x) {
    for (int i = (int)x.size() - 1; i > 0; --i) std::swap(x[i], x[randbelow((uint32_t)i + 1)]);
  }
};

struct TAtom { string name, alt, element; int charge = 0; bool leaving = false; };
struct Template { string type; vector<TAtom> atoms; };

inline const vector<string>* col(const cif::Block& b, const string& cat, const string& key) {
  auto c = b.find(cat);
  if (c == b.end()) return nullptr;
  auto it = c->second.find(key);
  return it == c->second.end() ? nullptr : &it->second;
}
inline string cell(const vector<string>* v, size_t i, const string& fallback = string()) {
  return v && i < v->size() ? (*v)[i] : fallback;
}
inline bool unsetValue(const string& v) { return v.empty() || v == "." || v == "?"; }
inline string upperOf(string s) { for (char& c : s) c = (char)toupper((unsigned char)c); return s; }
inline string lowerOf(string s) { for (char& c : s) c = (char)tolower((unsigned char)c); return s; }

/** A component's template: the generated standard set, else a fetched or the file's `_chem_comp_atom`. */
struct Lookup {
  std::map<string, Template> fromCat;
  std::map<string, string> types;
  void add(const cif::Block& b) {
    const vector<string>* ids = col(b, "chem_comp_atom", "comp_id");
    if (!ids) return;
    std::set<string> taken;
    for (const auto& [k, v] : fromCat) taken.insert(k);
    for (size_t r = 0; r < ids->size(); ++r) {
      const string& id = (*ids)[r];
      if (tables::CCD_TEMPLATES.count(id) || taken.count(id)) continue;
      const string element = upperOf(cell(col(b, "chem_comp_atom", "type_symbol"), r));
      if (element == "H" || element == "D") continue;
      Template& t = fromCat[id];
      const string atom = cell(col(b, "chem_comp_atom", "atom_id"), r);
      const vector<string>* alts = col(b, "chem_comp_atom", "alt_atom_id");
      const string charge = cell(col(b, "chem_comp_atom", "charge"), r);
      t.atoms.push_back({atom, alts ? cell(alts, r) : atom, element,
                         unsetValue(charge) ? 0 : (int)strtol(charge.c_str(), nullptr, 10),
                         cell(col(b, "chem_comp_atom", "pdbx_leaving_atom_flag"), r) == "Y"});
    }
  }
  void addTypes(const cif::Block& b) {
    const vector<string>* ids = col(b, "chem_comp", "id");
    for (size_t r = 0; ids && r < ids->size(); ++r) types[(*ids)[r]] = upperOf(cell(col(b, "chem_comp", "type"), r));
  }
  std::optional<Template> operator()(const string& code) const {
    auto st = tables::CCD_TEMPLATES.find(code);
    if (st != tables::CCD_TEMPLATES.end()) {
      Template t{st->second.type, {}};
      for (const auto& a : st->second.atoms) t.atoms.push_back({a.name, a.alt, a.element, a.charge, a.leaving});
      return t;
    }
    auto it = fromCat.find(code);
    if (it == fromCat.end()) return std::nullopt;
    Template t = it->second;
    if (t.type.empty()) {
      auto ty = types.find(code);
      if (ty != types.end()) t.type = ty->second;
    }
    return t;
  }
};

inline std::pair<string, string> polymerizationAtoms(const string& type) {
  if (type.find("PEPTIDE") != string::npos) return {"C", "N"};
  if (type.find("DNA") != string::npos || type.find("RNA") != string::npos) return {"O3'", "P"};
  return {"", ""};
}

struct Row {
  string group, element, atom, alt, comp, chain, entity, auth, occupancy, b, charge;
  int seq = 0;
  bool labelled = false;
  double xyz[3];
};

[[noreturn]] inline void refuse(const string& what) {
  fprintf(stderr, "%s\n", what.c_str());
  exit(1);
}

/** `_validate_label_seq_ids`. */
inline void validateLabelSeqIds(const cif::Block& b) {
  const vector<string>* asym = col(b, "atom_site", "label_asym_id");
  const vector<string>* seq = col(b, "atom_site", "label_seq_id");
  const vector<string>* ent = col(b, "atom_site", "label_entity_id");
  if (!asym || !seq) return;
  struct E { bool missing = false, present = false; std::set<string> entities; };
  std::map<string, E> byAsym;
  for (size_t i = 0; i < asym->size(); ++i) {
    E& e = byAsym[(*asym)[i]];
    const string s = cell(seq, i);
    if (s == "." || s == "?") e.missing = true; else e.present = true;
    if (ent) e.entities.insert(cell(ent, i));
  }
  for (const auto& [a, e] : byAsym) {
    if (e.missing && e.present) {
      refuse("CIF label_asym_id '" + a + "' mixes defined and missing label_seq_id values. Polymer members, "
             "including noncanonical monomers, require label_seq_id; standalone non-polymers must use a separate label_asym_id.");
    }
    if (ent && e.entities.size() != 1) refuse("CIF label_asym_id '" + a + "' maps to multiple label_entity_id values.");
  }
  std::map<string, string> types;
  const vector<string>* ids = col(b, "entity", "id");
  for (size_t r = 0; ids && r < ids->size(); ++r) {
    string t = lowerOf(cell(col(b, "entity", "type"), r));
    while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
    while (!t.empty() && isspace((unsigned char)t.front())) t.erase(t.begin());
    types[(*ids)[r]] = t;
  }
  if (!ent || types.empty()) return;
  for (size_t i = 0; i < asym->size(); ++i) {
    const string s = cell(seq, i);
    if ((s == "." || s == "?") && types[cell(ent, i)] == "polymer") {
      refuse("CIF polymer label_asym_id '" + (*asym)[i] + "' has missing label_seq_id values. Every polymer member, "
             "including noncanonical monomers, must have a sequence ID.");
    }
  }
}

/** `select_altlocs_clash_aware(seed=0)` (fresh.js `selectAltlocsClashAware`). */
inline vector<Row> selectAltlocsClashAware(const vector<Row>& rows, const cif::Block& b, const Lookup& lookup) {
  auto isDefault = [](const string& a) { return a.empty() || a == "." || a == "?" || a == " "; };
  vector<char> isAlt(rows.size());
  bool any = false;
  for (size_t i = 0; i < rows.size(); ++i) { isAlt[i] = !isDefault(rows[i].alt); any = any || isAlt[i]; }
  if (!any) return rows;
  struct Group { string chain; int seq; vector<string> letterOrder; std::map<string, vector<size_t>> letters; };
  vector<Group> residues;
  std::map<std::pair<string, int>, size_t> index;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (!isAlt[i]) continue;
    auto key = std::make_pair(rows[i].chain, rows[i].seq);
    auto it = index.find(key);
    if (it == index.end()) { index[key] = residues.size(); residues.push_back({rows[i].chain, rows[i].seq, {}, {}}); it = index.find(key); }
    Group& g = residues[it->second];
    if (!g.letters.count(rows[i].alt)) g.letterOrder.push_back(rows[i].alt);
    g.letters[rows[i].alt].push_back(i);
  }
  std::set<string> scBonds;
  if (const vector<string>* a1 = col(b, "struct_conn", "ptnr1_label_asym_id")) {
    for (size_t k = 0; k < a1->size(); ++k) {
      auto side = [&](int p) {
        const string pre = "ptnr" + std::to_string(p);
        string s = cell(col(b, "struct_conn", pre + "_label_seq_id"), k);
        if (s == ".") { const vector<string>* au = col(b, "struct_conn", pre + "_auth_seq_id"); if (au) s = cell(au, k); }
        return cell(col(b, "struct_conn", pre + "_label_asym_id"), k) + '\x00' + s + '\x00' +
               cell(col(b, "struct_conn", pre + "_label_atom_id"), k);
      };
      scBonds.insert(side(1) + '\x01' + side(2));
      scBonds.insert(side(2) + '\x01' + side(1));
    }
  }
  auto typeOf = [&](const string& comp) { auto t = lookup(comp); return t ? t->type : string(); };
  auto bonded = [&](size_t i, size_t j) {
    const Row& a = rows[i];
    const Row& c = rows[j];
    const auto [outA, inA] = polymerizationAtoms(typeOf(a.comp));
    const auto [outB, inB] = polymerizationAtoms(typeOf(c.comp));
    if ((!outA.empty() && a.atom == outA && c.atom == inB) || (!outB.empty() && c.atom == outB && a.atom == inA)) return true;
    return scBonds.count(a.chain + '\x00' + std::to_string(a.seq) + '\x00' + a.atom + '\x01' + c.chain + '\x00' +
                         std::to_string(c.seq) + '\x00' + c.atom) > 0;
  };
  struct Item { size_t i, g; string letter; };
  vector<Item> valid;
  for (size_t g = 0; g < residues.size(); ++g) {
    for (const string& l : residues[g].letterOrder) {
      for (size_t i : residues[g].letters[l]) {
        if (std::isfinite(rows[i].xyz[0]) && std::isfinite(rows[i].xyz[1]) && std::isfinite(rows[i].xyz[2])) valid.push_back({i, g, l});
      }
    }
  }
  std::map<std::array<long, 3>, vector<size_t>> grid;
  auto cellOf = [](const double* c) { return std::array<long, 3>{(long)std::floor(c[0] / 5), (long)std::floor(c[1] / 5), (long)std::floor(c[2] / 5)}; };
  for (size_t k = 0; k < valid.size(); ++k) grid[cellOf(rows[valid[k].i].xyz)].push_back(k);
  struct Clash { size_t gi; string li; size_t gj; string lj; };
  vector<Clash> clashList;
  std::set<string> seenClash, prefSeen;
  vector<std::pair<size_t, size_t>> prefs;
  for (size_t k = 0; k < valid.size(); ++k) {
    const double* c = rows[valid[k].i].xyz;
    const auto base = cellOf(c);
    for (long dx = -1; dx <= 1; ++dx) for (long dy = -1; dy <= 1; ++dy) for (long dz = -1; dz <= 1; ++dz) {
      auto it = grid.find({base[0] + dx, base[1] + dy, base[2] + dz});
      if (it == grid.end()) continue;
      for (size_t m : it->second) {
        if (m <= k) continue;
        const Item& v = valid[k];
        const Item& w = valid[m];
        if (w.g == v.g) continue;
        const double* d2 = rows[w.i].xyz;
        const double d = lf::jsmath::hypot(c[0] - d2[0], c[1] - d2[1], c[2] - d2[2]);
        if (d <= 1.5 && !bonded(v.i, w.i)) {
          const string e1 = std::to_string(v.g) + "|" + v.letter + "|" + std::to_string(w.g) + "|" + w.letter;
          const string e2 = std::to_string(w.g) + "|" + w.letter + "|" + std::to_string(v.g) + "|" + v.letter;
          if (!seenClash.count(e1) && !seenClash.count(e2)) { seenClash.insert(e1); clashList.push_back({v.g, v.letter, w.g, w.letter}); }
        }
        if (d <= 5) {
          const Group& gv = residues[v.g];
          const Group& gw = residues[w.g];
          const bool less = gv.chain < gw.chain || (gv.chain == gw.chain && gv.seq < gw.seq);
          const size_t a = less ? v.g : w.g, c2 = less ? w.g : v.g;
          const string pk = std::to_string(a) + "|" + std::to_string(c2);
          if (!prefSeen.count(pk)) { prefSeen.insert(pk); prefs.push_back({a, c2}); }
        }
      }
    }
  }
  std::set<std::pair<size_t, size_t>> prefPairs;
  for (auto [a, c2] : prefs) { prefPairs.insert({a, c2}); prefPairs.insert({c2, a}); }
  std::map<std::pair<size_t, string>, std::set<std::pair<size_t, string>>> clashes;
  for (const Clash& c : clashList) {
    if (prefPairs.count({c.gi, c.gj})) continue;
    clashes[{c.gi, c.li}].insert({c.gj, c.lj});
    clashes[{c.gj, c.lj}].insert({c.gi, c.li});
  }
  PyRandom rng(0);
  vector<vector<string>> options(residues.size());
  for (size_t g = 0; g < residues.size(); ++g) { options[g] = residues[g].letterOrder; rng.shuffle(options[g]); }
  vector<size_t> parent(residues.size());
  for (size_t g = 0; g < residues.size(); ++g) parent[g] = g;
  std::function<size_t(size_t)> find = [&](size_t x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  for (auto [a, c2] : prefs) parent[find(a)] = find(c2);
  vector<size_t> rootOrder;
  std::map<size_t, vector<size_t>> groupMap;
  for (size_t g = 0; g < residues.size(); ++g) {
    const size_t r = find(g);
    if (!groupMap.count(r)) rootOrder.push_back(r);
    groupMap[r].push_back(g);
  }
  vector<vector<size_t>> unions;
  for (size_t r : rootOrder) unions.push_back(groupMap[r]);
  rng.shuffle(unions);
  std::map<size_t, string> assignment;
  auto countClashes = [&](size_t g, const string& l) {
    int n = 0;
    auto it = clashes.find({g, l});
    if (it == clashes.end()) return 0;
    for (const auto& [og, ol] : it->second) {
      auto a = assignment.find(og);
      if (a != assignment.end() && a->second == ol) ++n;
    }
    return n;
  };
  for (auto& group : unions) {
    rng.shuffle(group);
    if (group.size() == 1) {
      const size_t g = group[0];
      string best = options[g][0];
      int bestScore = countClashes(g, best);
      for (size_t q = 1; q < options[g].size(); ++q) {
        const int s = countClashes(g, options[g][q]);
        if (s < bestScore) { best = options[g][q]; bestScore = s; }
      }
      assignment[g] = best;
    } else {
      std::set<string> all;
      for (size_t g : group) for (const string& l : options[g]) all.insert(l);
      vector<string> candidates(all.begin(), all.end());
      rng.shuffle(candidates);
      auto score = [&](const string& l) {
        int supporters = 0, total = 0;
        for (size_t g : group) {
          if (std::find(options[g].begin(), options[g].end(), l) != options[g].end()) { ++supporters; total += countClashes(g, l); }
        }
        return std::make_pair(-supporters, total);
      };
      string best = candidates[0];
      auto bestScore = score(best);
      for (size_t q = 1; q < candidates.size(); ++q) {
        const auto s = score(candidates[q]);
        if (s < bestScore) { best = candidates[q]; bestScore = s; }
      }
      for (size_t g : group) {
        if (std::find(options[g].begin(), options[g].end(), best) != options[g].end()) assignment[g] = best;
      }
    }
  }
  int unresolved = 0;
  for (const auto& [g, l] : assignment) unresolved += countClashes(g, l);
  if (unresolved > 0) {
    refuse("No fully clash-free altloc assignment found for " + std::to_string(residues.size()) + " residues. " +
           std::to_string(unresolved / 2) + " unresolved clashes remain.");
  }
  vector<char> keep(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) keep[i] = !isAlt[i];
  for (const auto& [g, l] : assignment) for (size_t i : residues[g].letters.at(l)) keep[i] = 1;
  vector<Row> out;
  for (size_t i = 0; i < rows.size(); ++i) if (keep[i]) out.push_back(rows[i]);
  return out;
}

/** `prepareFreshBlock`: the block with its `_atom_site` rewritten as upstream prepares a raw structure. */
inline cif::Block prepareFreshBlock(const cif::Block& block, const vector<cif::Block>& ccdBlocks) {
  if (!block.count("atom_site")) refuse("CIF block does not contain _atom_site category");
  validateLabelSeqIds(block);
  Lookup lookup;
  for (const cif::Block& b : ccdBlocks) lookup.add(b);
  lookup.add(block);
  lookup.addTypes(block);
  for (const cif::Block& b : ccdBlocks) lookup.addTypes(b);

  auto site = [&](const char* k) { return col(block, "atom_site", k); };
  const vector<string>* label = site("label_atom_id");
  const size_t nRows = label ? label->size() : (site("auth_atom_id") ? site("auth_atom_id")->size() : 0);
  const vector<string>* models = site("pdbx_PDB_model_num");
  vector<Row> rows;
  for (size_t r = 0; r < nRows; ++r) {
    if (models && !models->empty() && cell(models, r) != (*models)[0]) continue;
    auto get = [&](const char* k, const string& fallback) { const vector<string>* v = site(k); return v ? cell(v, r, fallback) : fallback; };
    Row w;
    w.group = get("group_PDB", "ATOM");
    w.element = upperOf(get("type_symbol", "C"));
    w.atom = get("label_atom_id", "");
    if (w.atom.empty()) w.atom = get("auth_atom_id", "");
    w.alt = get("label_alt_id", ".");
    w.comp = get("label_comp_id", "");
    if (w.comp.empty()) w.comp = get("auth_comp_id", "");
    w.chain = get("label_asym_id", "");
    if (w.chain.empty()) w.chain = get("auth_asym_id", "A");
    w.entity = get("label_entity_id", "?");
    const string seqText = get("label_seq_id", ".");
    w.auth = get("auth_seq_id", "?");
    w.labelled = !unsetValue(seqText);
    w.seq = (int)strtol((w.labelled ? seqText : w.auth).c_str(), nullptr, 10);
    const char* xyz[3] = {"Cartn_x", "Cartn_y", "Cartn_z"};
    for (int c = 0; c < 3; ++c) {
      const string v = get(xyz[c], "?");
      w.xyz[c] = unsetValue(v) ? NAN : strtod(v.c_str(), nullptr);
    }
    w.occupancy = get("occupancy", "1");
    w.b = get("B_iso_or_equiv", "?");
    w.charge = get("pdbx_formal_charge", "?");
    rows.push_back(w);
  }
  if (site("label_alt_id")) rows = selectAltlocsClashAware(rows, block, lookup);
  {
    vector<Row> kept;
    for (const Row& w : rows) {
      if (w.element == "H" || w.element == "D") continue;
      if (std::find(tables::WATER_LIKE_CCDS.begin(), tables::WATER_LIKE_CCDS.end(), w.comp) != tables::WATER_LIKE_CCDS.end()) continue;
      kept.push_back(w);
    }
    rows = std::move(kept);
  }

  std::map<string, string> entityPolyType, entityType;
  if (const vector<string>* ids = col(block, "entity_poly", "entity_id")) {
    for (size_t r = 0; r < ids->size(); ++r) entityPolyType[(*ids)[r]] = lowerOf(cell(col(block, "entity_poly", "type"), r));
  }
  if (const vector<string>* ids = col(block, "entity", "id")) {
    for (size_t r = 0; r < ids->size(); ++r) {
      string t = lowerOf(cell(col(block, "entity", "type"), r));
      while (!t.empty() && isspace((unsigned char)t.back())) t.pop_back();
      while (!t.empty() && isspace((unsigned char)t.front())) t.erase(t.begin());
      entityType[(*ids)[r]] = t;
    }
  }
  vector<string> chainOrder;
  std::map<string, vector<Row>> chainRows;
  for (const Row& w : rows) {
    if (!chainRows.count(w.chain)) chainOrder.push_back(w.chain);
    chainRows[w.chain].push_back(w);
  }
  struct Mon { int num; string mon; };
  std::map<string, vector<Mon>> polySeq;
  if (const vector<string>* ents = col(block, "entity_poly_seq", "entity_id")) {
    for (size_t k = 0; k < ents->size(); ++k) {
      polySeq[(*ents)[k]].push_back({(int)strtol(cell(col(block, "entity_poly_seq", "num"), k).c_str(), nullptr, 10),
                                     cell(col(block, "entity_poly_seq", "mon_id"), k)});
    }
  }

  struct Out { string group, element, atom, comp, chain, entity, auth, occupancy, b, charge; int seq; double xyz[3]; };
  vector<Out> out;
  struct Residue { int seq; string comp, auth, group; vector<Row> atoms; };
  for (const string& chain : chainOrder) {
    const vector<Row>& list = chainRows[chain];
    const string entity = list[0].entity;
    const string polyType = entityPolyType.count(entity) ? entityPolyType[entity] : string();
    const bool isPolymer = entityPolyType.count(entity) || (entityType.count(entity) && entityType[entity] == "polymer") ||
                           (!entityType.count(entity) && list[0].labelled);
    auto firstType = [&]() { auto t = lookup(list[0].comp); return t ? t->type : string(); };
    const bool isProtein = polyType.find("peptide") != string::npos ||
                           (polyType.empty() && isPolymer && firstType().find("PEPTIDE") != string::npos);
    const bool isNucleic = polyType.find("nucleotide") != string::npos ||
                           (polyType.empty() && isPolymer &&
                            (firstType().find("DNA") != string::npos || firstType().find("RNA") != string::npos));
    vector<Residue> residues;
    for (const Row& w : list) {
      if (!residues.empty() && residues.back().seq == w.seq && residues.back().comp == w.comp) residues.back().atoms.push_back(w);
      else residues.push_back({w.seq, w.comp, w.auth, w.group, {w}});
    }
    struct Entry { const Residue* res; int num; string mon; };
    vector<Entry> sequence;
    for (const Residue& r : residues) sequence.push_back({&r, r.seq, string()});
    if (isPolymer && polySeq.count(entity)) {
      std::set<int> have;
      for (const Residue& r : residues) have.insert(r.seq);
      vector<Entry> missing;
      for (const Mon& m : polySeq[entity]) if (!have.count(m.num)) missing.push_back({nullptr, m.num, m.mon});
      if (!missing.empty()) {
        sequence.insert(sequence.end(), missing.begin(), missing.end());
        std::stable_sort(sequence.begin(), sequence.end(), [](const Entry& a, const Entry& c) { return a.num < c.num; });
      }
    }
    for (size_t k = 0; k < sequence.size(); ++k) {
      const Residue* res = sequence[k].res;
      string comp = res ? res->comp : sequence[k].mon;
      if (comp == "MSE") comp = "MET";
      const auto tmpl = lookup(comp);
      const bool hasPrev = isPolymer && k > 0;
      const bool hasNext = isPolymer && k + 1 < sequence.size();
      auto leaves = [&](const string& name) { return (isProtein && hasNext && name == "OXT") || (isNucleic && hasPrev && name == "OP3"); };
      const int resId = res ? res->seq : sequence[k].num;
      const string auth = res ? res->auth : std::to_string(sequence[k].num);
      const string group = res ? (res->comp == "MSE" ? string("ATOM") : res->group) : string("ATOM");
      auto emit = [&](const string& atom, const string& element, const double* xyz, const string& occ, const string& b, const string& charge) {
        Out o{group, element, atom, comp, chain, entity, auth, occ, b, charge, isPolymer ? resId : -1, {xyz[0], xyz[1], xyz[2]}};
        out.push_back(o);
      };
      if (res && (hasPrev || hasNext)) {
        for (const Row& a : res->atoms) {
          if (a.element == "X") {
            refuse("Unresolved link valence at " + chain + "/" + std::to_string(res->seq) + "/" + res->comp + "/" + a.atom +
                   " (element=X, charge=0): a raw structure cannot carry an atom of no element inside a polymer - load an "
                   "annotated CIF with input_as_annotated_cif: true");
          }
        }
      }
      if (!tmpl) {
        if (!res) continue;
        for (const Row& a : res->atoms) {
          if (leaves(a.atom)) continue;
          emit(a.atom, a.element, a.xyz, a.occupancy, a.b, unsetValue(a.charge) ? "0" : a.charge);
        }
        continue;
      }
      std::map<string, const Row*> byName;
      vector<const Row*> extra;
      if (res) {
        for (const Row& a : res->atoms) {
          string name = a.atom;
          if (res->comp == "MSE" && name == "SE") name = "SD";
          const TAtom* t = nullptr;
          for (const TAtom& x : tmpl->atoms) if (x.name == name) { t = &x; break; }
          if (!t) for (const TAtom& x : tmpl->atoms) if (x.alt == name) { t = &x; break; }
          if (t) { if (!byName.count(t->name)) byName[t->name] = &a; } else extra.push_back(&a);
        }
      }
      const double none[3] = {NAN, NAN, NAN};
      for (const TAtom& t : tmpl->atoms) {
        if (leaves(t.name)) continue;
        auto it = byName.find(t.name);
        if (it != byName.end()) {
          const Row& a = *it->second;
          const string el = res->comp == "MSE" && a.atom == "SE" ? string("S") : a.element;
          emit(t.name, el, a.xyz, a.occupancy, a.b, std::to_string(t.charge));
        } else {
          emit(t.name, t.element, none, "0", "?", std::to_string(t.charge));
        }
      }
      for (const Row* a : extra) {
        if (leaves(a->atom)) continue;
        emit(a->atom, a->element, a->xyz, a->occupancy, a->b, unsetValue(a->charge) ? "0" : a->charge);
      }
    }
  }

  // Crystallization aids go, unless covalently joined to something that stays.
  const std::set<string> aids(tables::CRYSTALLIZATION_AIDS.begin(), tables::CRYSTALLIZATION_AIDS.end());
  vector<Out> prepared = out;
  bool anyAid = false;
  for (const Out& o : out) anyAid = anyAid || aids.count(o.comp);
  if (anyAid) {
    auto resKey = [](const Out& o) { return o.chain + '\x00' + std::to_string(o.seq == -1 ? (int)strtol(o.auth.c_str(), nullptr, 10) : o.seq); };
    std::map<string, string> parentOf;
    std::function<string(string)> find = [&](string x) { while (parentOf[x] != x) x = parentOf[x]; return x; };
    for (const Out& o : out) parentOf[resKey(o)] = resKey(o);
    if (const vector<string>* types = col(block, "struct_conn", "conn_type_id")) {
      for (size_t k = 0; k < types->size(); ++k) {
        if (lowerOf((*types)[k]) != "covale") continue;
        auto key = [&](int p) {
          const string pre = "ptnr" + std::to_string(p);
          string s = cell(col(block, "struct_conn", pre + "_label_seq_id"), k);
          if (s == ".") s = cell(col(block, "struct_conn", pre + "_auth_seq_id"), k);
          return cell(col(block, "struct_conn", pre + "_label_asym_id"), k) + '\x00' +
                 std::to_string((int)strtol(s.c_str(), nullptr, 10));
        };
        const string a = key(1), c = key(2);
        if (parentOf.count(a) && parentOf.count(c)) parentOf[find(a)] = find(c);
      }
    }
    std::set<string> keepRoot;
    for (const Out& o : out) if (!aids.count(o.comp)) keepRoot.insert(find(resKey(o)));
    prepared.clear();
    for (const Out& o : out) if (!aids.count(o.comp) || keepRoot.count(find(resKey(o)))) prepared.push_back(o);
  }

  // `String(x)` of a parsed coordinate: the shortest text that reads back as the same double.
  auto fmt = [](double x) {
    if (!std::isfinite(x)) return string("?");
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, x);
    return string(buf, r.ptr);
  };
  std::map<string, vector<string>> as;
  for (size_t k = 0; k < prepared.size(); ++k) {
    const Out& o = prepared[k];
    as["group_PDB"].push_back(o.group);
    as["id"].push_back(std::to_string(k + 1));
    as["type_symbol"].push_back(o.element);
    as["label_atom_id"].push_back(o.atom);
    as["label_alt_id"].push_back(".");
    as["label_comp_id"].push_back(o.comp);
    as["label_asym_id"].push_back(o.chain);
    as["label_entity_id"].push_back(o.entity);
    as["label_seq_id"].push_back(o.seq == -1 ? string(".") : std::to_string(o.seq));
    as["pdbx_PDB_ins_code"].push_back("?");
    as["Cartn_x"].push_back(fmt(o.xyz[0]));
    as["Cartn_y"].push_back(fmt(o.xyz[1]));
    as["Cartn_z"].push_back(fmt(o.xyz[2]));
    as["occupancy"].push_back(o.occupancy);
    as["B_iso_or_equiv"].push_back(o.b);
    as["pdbx_formal_charge"].push_back(o.charge);
    as["auth_seq_id"].push_back(o.auth);
    as["auth_comp_id"].push_back(o.comp);
    as["auth_asym_id"].push_back(o.chain);
    as["auth_atom_id"].push_back(o.atom);
    as["pdbx_PDB_model_num"].push_back("1");
  }
  cif::Block result = block;
  result["atom_site"] = as;
  return result;
}

/**
 * `ccdBlocksForRawStructure` (ccd.js): every dictionary entry preparing this structure needs - each
 * component outside the generated templates that is not water - from a local CCD when one is
 * given (`--ccd=`), else from RCSB through the shared cache (cuda/featurise/ccd.h). A component
 * that cannot be had is a refusal, never a structure prepared without it.
 */
inline vector<cif::Block> ccdBlocksForRawStructure(const string& text, const string& ccdFile) {
  const cif::Block b = cif::parseBlock(text);
  const vector<string>* names = col(b, "atom_site", "label_comp_id");
  if (!names) names = col(b, "atom_site", "auth_comp_id");
  std::set<string> ids;
  for (size_t i = 0; names && i < names->size(); ++i) {
    const string& n = (*names)[i];
    if (n.empty() || tables::CCD_TEMPLATES.count(n)) continue;
    if (std::find(tables::WATER_LIKE_CCDS.begin(), tables::WATER_LIKE_CCDS.end(), n) != tables::WATER_LIKE_CCDS.end()) continue;
    bool ok = n.size() <= 5;
    for (const char c : n) ok = ok && isalnum((unsigned char)c);
    if (ok) ids.insert(n);
  }
  vector<cif::Block> blocks;
  vector<string> failed;
  for (const string& id : ids) {
    try {
      cif::Block entry = cif::parseBlock(ccdFile.empty() ? lf::ccdText(id) : lf::ccdFileText(ccdFile, id));
      if (!entry.count("chem_comp_atom")) throw std::runtime_error("no _chem_comp_atom");
      blocks.push_back(std::move(entry));
    } catch (const std::exception&) {
      failed.push_back(id);
    }
  }
  if (!failed.empty()) {
    string list;
    for (const string& f : failed) list += (list.empty() ? "" : ", ") + f;
    refuse(string("could not fetch the Chemical Component Dictionary entr") + (failed.size() == 1 ? "y" : "ies") +
           " for " + list + " from RCSB, which preparing this structure needs");
  }
  return blocks;
}

}  // namespace fresh
}  // namespace rfd4p
