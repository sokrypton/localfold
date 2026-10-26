// `ValidateBatchInvariants` (rfproteina/transforms/invariants.py), at inference: the native twin
// of shared/rfd4p/pipeline/invariants.js, which has the notes. A violation is refused in
// upstream's words, every check it names listed; tools/check-rfd4p-invariants.py holds the
// verdicts to upstream's.
#pragma once
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "atoms.h"

namespace rfd4p {
namespace invariants {

using std::string;
using std::vector;

struct Violation { string check, message; long count; };

template <class BatchT>
struct Context {
  atoms::AtomArray& a;
  const BatchT& b;
  int nTok;
};

inline string fmt(const char* f, double v) {
  char buf[64];
  snprintf(buf, sizeof buf, f, v);
  return buf;
}

inline bool finite3(const double* v) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }
inline bool anyFinite3(const double* v) { return std::isfinite(v[0]) || std::isfinite(v[1]) || std::isfinite(v[2]); }
inline bool allNaN3(const double* v) { return std::isnan(v[0]) && std::isnan(v[1]) && std::isnan(v[2]); }

inline const vector<int>* ints(const atoms::AtomArray& a, const char* k) {
  auto it = a.ann.ints.find(k);
  return it == a.ann.ints.end() || it->second.size() != a.n() ? nullptr : &it->second;
}
inline const vector<double>* floats3(const atoms::AtomArray& a, const char* k) {
  auto it = a.ann.floats.find(k);
  return it == a.ann.floats.end() || it->second.size() != a.n() * 3 ? nullptr : &it->second;
}
inline const vector<double>* floats1(const atoms::AtomArray& a, const string& k) {
  auto it = a.ann.floats.find(k);
  return it == a.ann.floats.end() || it->second.size() != a.n() ? nullptr : &it->second;
}
inline const string& chainOf(const atoms::AtomArray& a, size_t i) {
  return a.ids.chainIid.empty() ? a.site.chainId[i] : a.ids.chainIid[i];
}

/** atomworks' `get_residue_starts` (the transformation is "1" throughout here). */
inline vector<int> residueStarts(const atoms::AtomArray& a) {
  vector<int> out;
  for (size_t i = 0; i < a.n(); ++i) {
    if (i == 0 || a.site.chainId[i] != a.site.chainId[i - 1] || a.site.resId[i] != a.site.resId[i - 1] ||
        a.site.insCode[i] != a.site.insCode[i - 1]) {
      out.push_back((int)i);
    }
  }
  return out;
}

inline const char* const SEQ_PROB_TOKENS[28] = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL", "A", "C", "G", "U", "DA", "DC", "DG", "DT"};

template <class B>
inline void moleculesOnSeparateChains(Context<B>& c, vector<Violation>& out) {
  std::map<string, std::set<int>> byChain;
  for (size_t i = 0; i < c.a.n(); ++i) byChain[chainOf(c.a, i)].insert(c.a.ids.moleculeIid[i]);
  for (const auto& [chain, mols] : byChain) {
    if (mols.size() > 1) {
      out.push_back({"multiple_molecules_on_chain",
                     "Chain " + chain + " contains " + std::to_string(mols.size()) + " different molecules", 1});
    }
  }
}

template <class B>
inline void standardAnnotationLevels(Context<B>& c, vector<Violation>& out) {
  const atoms::AtomArray& a = c.a;
  vector<int> residue = residueStarts(a);
  residue.push_back((int)a.n());
  const vector<int> token = atoms::tokenStarts(a);
  vector<std::pair<string, string>> levelled = {
      {"annotation_expsegmax_1_residue", "residue"}, {"annotation_expsegmin_1_residue", "residue"},
      {"condition_c-terminus_1_residue", "residue"}, {"condition_cath-label_1_residue", "residue"},
      {"condition_index-group_1_token", "token"},     {"condition_index_1_residue", "residue"},
      {"condition_n-terminus_1_residue", "residue"}, {"condition_na-ss-state_1_residue", "residue"},
      {"condition_secondary-structure_1_residue", "residue"}};
  for (const char* t : SEQ_PROB_TOKENS) levelled.push_back({string("condition_sequence-probability-") + t + "_1_residue", "residue"});
  levelled.push_back({"condition_sequence_1_residue", "residue"});
  for (const auto& [name, level] : levelled) {
    const vector<int>& starts = level == "residue" ? residue : token;
    // An annotation is a column of ints, strings, or floats (one or several a row).
    std::function<bool(int, int)> same;
    auto iit = a.ann.ints.find(name);
    auto sit = a.ann.strings.find(name);
    auto fit = a.ann.floats.find(name);
    if (iit != a.ann.ints.end()) {
      same = [&, v = &iit->second](int i, int j) { return (*v)[i] == (*v)[j]; };
    } else if (sit != a.ann.strings.end()) {
      same = [&, v = &sit->second](int i, int j) { return (*v)[i] == (*v)[j]; };
    } else if (fit != a.ann.floats.end() && a.n() > 0) {
      const size_t k = fit->second.size() / a.n();
      same = [&, v = &fit->second, k](int i, int j) {
        for (size_t d = 0; d < k; ++d) {
          const double x = (*v)[i * k + d], y = (*v)[j * k + d];
          if (!(x == y || (std::isnan(x) && std::isnan(y)))) return false;
        }
        return true;
      };
    } else {
      continue;
    }
    long bad = 0;
    for (size_t s = 0; s + 1 < starts.size(); ++s) {
      for (int i = starts[s] + 1; i < starts[s + 1]; ++i) {
        if (!same(i, starts[s])) { ++bad; break; }
      }
    }
    if (bad) {
      out.push_back({"standard_annotation_level_violated",
                     "'" + name + "' (level=" + level + ") has inconsistent values in " + std::to_string(bad) + " segment(s)",
                     bad});
    }
  }
}

template <class B>
inline void globalTokenIdConsistent(Context<B>& c, vector<Violation>& out) {
  atoms::AtomArray copy = c.a;
  atoms::addGlobalTokenIdAnnotation(copy);
  long bad = 0;
  for (size_t i = 0; i < c.a.n(); ++i) bad += copy.tokenId[i] != c.a.tokenId[i];
  if (bad) {
    out.push_back({"global_token_id_inconsistent",
                   "'token_id' annotation is inconsistent with recomputed values in " + std::to_string(bad) + " atom(s)", bad});
  }
}

template <class B>
inline void centroidAllOrNothing(Context<B>& c, vector<Violation>& out) {
  const vector<double>* centroid = floats3(c.a, "condition_ChainCentroid_1_atom");
  if (!centroid) return;
  std::map<string, std::pair<bool, bool>> chains;   // some, all
  for (size_t i = 0; i < c.a.n(); ++i) {
    auto& [some, all] = chains.try_emplace(chainOf(c.a, i), false, true).first->second;
    const bool m = anyFinite3(&(*centroid)[i * 3]);
    some = some || m;
    all = all && m;
  }
  long conditioned = 0;
  vector<string> partial;
  for (const auto& [chain, sa] : chains) {
    if (!sa.first) continue;
    ++conditioned;
    if (!sa.second) partial.push_back(chain);
  }
  if (!partial.empty()) {
    string list;
    for (size_t k = 0; k < partial.size(); ++k) list += (k ? ", '" : "'") + partial[k] + "'";
    out.push_back({"centroid_conditioning_partial_chain",
                   "chain(s) [" + list + "] have a centroid on only some of their atoms; the centroid is a per-chain "
                   "value and must cover every atom of the chain", (long)partial.size()});
  }
  if (!chains.empty() && conditioned > 0 && conditioned < (long)chains.size()) {
    out.push_back({"centroid_conditioning_not_all_or_nothing",
                   std::to_string(conditioned) + "/" + std::to_string(chains.size()) +
                       " chain(s) have centroid conditioning; must be applied to all chains or none",
                   (long)chains.size()});
  }
}

template <class B>
inline void conditioningCoordsNotNan(Context<B>& c, vector<Violation>& out) {
  const vector<int>* groups = ints(c.a, "condition_CoordinateGroup_1_atom");
  const vector<double>* intcrd = floats3(c.a, "condition_InternalCoordinate_1_atom");
  const vector<double>* crd = floats3(c.a, "condition_coordinate_1_atom");
  auto internal = [&](size_t i) { return groups && (*groups)[i] != -1 && (*groups)[i] != 0; };
  long orphan = 0, badCrd = 0, badInt = 0;
  for (size_t i = 0; i < c.a.n(); ++i) {
    if (intcrd && anyFinite3(&(*intcrd)[i * 3]) && !internal(i)) ++orphan;
    if (crd && anyFinite3(&(*crd)[i * 3]) && !finite3(&(*crd)[i * 3])) ++badCrd;
    if (internal(i) && !(intcrd && finite3(&(*intcrd)[i * 3]))) ++badInt;
  }
  if (orphan) {
    out.push_back({"internal_coords_require_coord_group",
                   std::to_string(orphan) + " atom(s) have a C_INTCRD payload but no internal C_CRDGRP; coordinate "
                   "groups are the source of truth for internal-coordinate conditioning", orphan});
  }
  if (badCrd) {
    out.push_back({"conditioning_coord_not_finite",
                   std::to_string(badCrd) + " C_CRD-conditioned atom(s) have non-finite conditioning coordinates", badCrd});
  }
  if (badInt) {
    out.push_back({"conditioning_coord_not_finite",
                   std::to_string(badInt) + " C_INTCRD-conditioned atom(s) have non-finite conditioning coordinates", badInt});
  }
}

template <class B>
inline void expandableSegmentConditioning(Context<B>& c, vector<Violation>& out) {
  const vector<int>* segMin = ints(c.a, "annotation_expsegmin_1_residue");
  if (!segMin) return;
  bool anyExp = false;
  for (const int v : *segMin) anyExp = anyExp || v != -1;
  if (!anyExp) return;
  struct Cond { const char* name; int def; bool coord; };
  const Cond conds[] = {{"condition_coordinate_1_atom", 0, true}, {"condition_InternalCoordinate_1_atom", 0, true},
                        {"condition_index_1_residue", 1, false}, {"condition_CoordinateGroup_1_atom", -1, false},
                        {"condition_hotspot_1_atom", 0, false},  {"condition_rasa-binned_1_atom", 0, false},
                        {"condition_hbond_1_atom", 0, false},    {"condition_pistack_1_atom", 0, false},
                        {"condition_pication_1_atom", 0, false}, {"condition_saltbridge_1_atom", 0, false},
                        {"condition_hydrophobic_1_atom", 0, false}, {"condition_metal-coord_1_atom", 0, false}};
  vector<string> names;
  std::set<int> tokens;
  for (const Cond& cd : conds) {
    const vector<double>* f = cd.coord ? floats3(c.a, cd.name) : nullptr;
    const vector<int>* iv = cd.coord ? nullptr : ints(c.a, cd.name);
    if (!f && !iv) continue;
    bool any = false;
    for (size_t i = 0; i < c.a.n(); ++i) {
      if ((*segMin)[i] == -1) continue;
      const bool nondefault = f ? !allNaN3(&(*f)[i * 3]) : (*iv)[i] != cd.def;
      if (nondefault) { any = true; tokens.insert(c.a.tokenId[i]); }
    }
    if (any) names.push_back(cd.name);
  }
  if (tokens.empty()) return;
  string list;
  for (size_t k = 0; k < names.size(); ++k) list += (k ? ", " : "") + names[k];
  out.push_back({"expandable_segment_atom37_conditioning",
                 std::to_string(tokens.size()) + " expandable-segment token(s) have atom37 conditioning that would be "
                 "discarded: " + list + ". Remove the conditioning or provide ordinary polymer residues.",
                 (long)tokens.size()});
}

/** bonds.js `getPolymerizationAtomNamesForToken`, with no chain type. */
inline std::pair<string, string> polymerizationNames(const string& resName, int tokenType) {
  std::pair<string, string> p = cif::polymerizationAtoms(resName, -1);
  if (p.first.empty() && p.second.empty()) {
    if (tokenType == tables::TT_PROTEIN || tokenType == tables::TT_NCAA) p = {"C", "N"};
    else if (tokenType == tables::TT_RNA || tokenType == tables::TT_DNA) p = {"O3'", "P"};
  }
  return p;
}

template <class B>
inline void bondLengths(Context<B>& c, vector<Violation>& out) {
  const atoms::AtomArray& a = c.a;
  if (a.bonds.empty()) return;
  const size_t n = a.n();
  const vector<int>* segMin = ints(a, "annotation_expsegmin_1_residue");
  auto resolved = [&](int i) {
    return std::isfinite(a.site.x[i]) && std::isfinite(a.site.y[i]) && std::isfinite(a.site.z[i]) &&
           !(segMin && (*segMin)[i] != -1) && a.site.occupancy[i] > 0;
  };
  auto tokenType = [&](int i) { return c.b.tokenType[a.tokenId[i]]; };
  auto inter = [&](const cif::Bond& bd) {
    return chainOf(a, bd.u) != chainOf(a, bd.v) || a.site.resId[bd.u] != a.site.resId[bd.v] ||
           a.site.insCode[bd.u] != a.site.insCode[bd.v];
  };
  vector<uint8_t> outgoing(n, 0), incoming(n, 0);
  std::set<std::pair<string, int>> kinds;
  for (const cif::Bond& bd : a.bonds) {
    if (!resolved(bd.u) || !resolved(bd.v) || !inter(bd)) continue;
    kinds.insert({a.site.resName[bd.u], tokenType(bd.u)});
    kinds.insert({a.site.resName[bd.v], tokenType(bd.v)});
  }
  for (const auto& [resName, rt] : kinds) {
    const auto [exitAtom, entryAtom] = polymerizationNames(resName, rt);
    for (size_t i = 0; i < n; ++i) {
      if (a.site.resName[i] != resName || tokenType((int)i) != rt) continue;
      if (!exitAtom.empty() && a.site.atomName[i] == exitAtom) outgoing[i] = 1;
      if (!entryAtom.empty() && a.site.atomName[i] == entryAtom) incoming[i] = 1;
    }
  }
  long count = 0;
  double worst = 0;
  string details;
  int shown = 0;
  for (const cif::Bond& bd : a.bonds) {
    if (!resolved(bd.u) || !resolved(bd.v)) continue;
    const int i = bd.u, j = bd.v;
    const double d = std::hypot(a.site.x[i] - a.site.x[j], a.site.y[i] - a.site.y[j], a.site.z[i] - a.site.z[j]);
    const bool polymer = inter(bd) && ((outgoing[i] && incoming[j]) || (incoming[i] && outgoing[j]));
    const double limit = polymer ? 2.0 : 4.0;
    if (!(d > limit)) continue;
    ++count;
    worst = std::max(worst, d);
    if (shown++ < 5) {
      auto label = [&](int k) {
        return chainOf(a, k) + ":" + a.site.resName[k] + std::to_string(a.site.resId[k]) + a.site.insCode[k] + ":" +
               a.site.atomName[k];
      };
      details += (shown > 1 ? "; " : "") + label(i) + " -- " + label(j) + " = " + fmt("%.2f", d) + " Å (limit " +
                 fmt("%.1f", limit) + " Å)";
    }
  }
  if (!count) return;
  out.push_back({"bond_length",
                 std::to_string(count) + " bond(s) exceed the polymerization limit of 2.0 Å or general limit of 4.0 Å; "
                 "maximum length is " + fmt("%.2f", worst) + " Å. First " + std::to_string(std::min<long>(count, 5)) +
                 ": " + details + ". Check source coordinates and residue numbering; preserve missing-residue gaps "
                 "instead of bonding across them.", count});
}

template <class B>
inline void unindexedCoordConditioning(Context<B>& c, vector<Violation>& out) {
  const atoms::AtomArray& a = c.a;
  const vector<int>* idx = ints(a, "condition_index_1_residue");
  if (!idx) return;
  const vector<double>* crd = floats3(a, "condition_coordinate_1_atom");
  const vector<double>* intcrd = floats3(a, "condition_InternalCoordinate_1_atom");
  vector<int> starts = residueStarts(a);
  starts.push_back((int)a.n());
  vector<int> bad;
  for (size_t r = 0; r + 1 < starts.size(); ++r) {
    bool unindexed = false, coord = false;
    for (int i = starts[r]; i < starts[r + 1]; ++i) {
      unindexed = unindexed || !(*idx)[i];
      coord = coord || (crd && anyFinite3(&(*crd)[i * 3])) || (intcrd && anyFinite3(&(*intcrd)[i * 3]));
    }
    if (unindexed && !coord) bad.push_back(starts[r]);
  }
  if (bad.empty()) return;
  string labels;
  for (size_t k = 0; k < bad.size() && k < 10; ++k) {
    labels += (k ? ", " : "") + a.site.chainId[bad[k]] + "/" + a.site.resName[bad[k]] + "/" + std::to_string(a.site.resId[bad[k]]);
  }
  out.push_back({"unindexed_must_be_coord_conditioned",
                 std::to_string(bad.size()) + " unindexed residue(s) have no coordinate conditioning on any atom: " +
                 labels + (bad.size() > 10 ? " ..." : "") + ". To pad an index group without conditioning it, leave a "
                 "res_id gap instead of including an unconditioned spacer residue.", (long)bad.size()});
}

template <class B>
inline void conditioning(Context<B>& c, vector<Violation>& out) {
  const int M = tables::TOKEN_TO_IDX.at("<M>");
  long atomizedNotSeq = 0, nonBb = 0;
  for (int t = 0; t < c.nTok; ++t) {
    if (c.b.tokenType[t] == tables::TT_ATOMIZED && c.b.conditionSequence[t] == M) ++atomizedNotSeq;
  }
  if (atomizedNotSeq) {
    out.push_back({"atomized_must_be_seq_conditioned",
                   std::to_string(atomizedNotSeq) + " atomized token(s) lack sequence conditioning", atomizedNotSeq});
  }
  vector<uint8_t> backbone((size_t)c.nTok * 37, 0);
  const vector<int> slots = atoms::buildAtom37Slots(c.a);
  for (size_t i = 0; i < c.a.n(); ++i) {
    if (slots[i] < 0) continue;
    backbone[(size_t)c.a.tokenId[i] * 37 + slots[i]] =
        (c.a.isProteinBackbone[i] || c.a.isRnaBackbone[i] || c.a.isDnaBackbone[i]) ? 1 : 0;
  }
  for (int t = 0; t < c.nTok; ++t) {
    bool has = false;
    for (int s = 0; s < 37 && !has; ++s) {
      const size_t at = (size_t)t * 37 + s;
      const bool cc = std::isfinite(c.b.condCoord[at * 3]) || std::isfinite(c.b.condCoord[at * 3 + 1]) ||
                      std::isfinite(c.b.condCoord[at * 3 + 2]) || (c.b.coordGroup[at] != -1 && c.b.coordGroup[at] != 0);
      has = cc && c.b.atom37Mask[at] && !backbone[at];
    }
    if (has && c.b.conditionSequence[t] == M) ++nonBb;
  }
  if (nonBb) {
    out.push_back({"coord_conditioned_must_be_seq_conditioned",
                   std::to_string(nonBb) + " token(s) with non-backbone coordinate conditioning lack sequence conditioning",
                   nonBb});
  }
}

template <class B>
inline void atomizedIndexed(Context<B>& c, vector<Violation>& out) {
  long bad = 0;
  for (int t = 0; t < c.nTok; ++t) {
    if (c.b.tokenType[t] != tables::TT_ATOMIZED) continue;
    bool indexed = false;
    for (int s = 0; s < 37; ++s) indexed = indexed || c.b.conditionIndex[(size_t)t * 37 + s];
    if (!indexed) ++bad;
  }
  if (bad) out.push_back({"atomized_must_be_indexed", std::to_string(bad) + " atomized token(s) are unindexed", bad});
}

template <class B>
inline void indexGroupConnectivity(Context<B>& c, vector<Violation>& out) {
  const atoms::AtomArray& a = c.a;
  if (c.b.tokenBondPathLength.empty()) return;
  const vector<int>* groups = ints(a, "condition_index-group_1_token");
  const vector<int>* idx = ints(a, "condition_index_1_residue");
  const vector<int>* segMin = ints(a, "annotation_expsegmin_1_residue");
  auto group = [&](int i) { return groups ? (*groups)[i] : 0; };
  auto indexed = [&](int i) { return (idx ? (*idx)[i] != 0 : true) && !(a.atomize.size() == a.n() && a.atomize[i]); };
  auto link = [&](int i) { return polymerizationNames(a.site.resName[i], c.b.tokenType[a.tokenId[i]]); };
  auto standsIn = [&](int i) { return segMin && (*segMin)[i] != -1; };
  auto leaving = [&](int i) { return a.site.atomName[i] == link(i).first || standsIn(i); };
  auto entering = [&](int i) { return a.site.atomName[i] == link(i).second || standsIn(i); };
  auto offending = [&](const cif::Bond& bd) {
    const int i = bd.u, j = bd.v;
    return ((leaving(i) && entering(j)) || (entering(i) && leaving(j))) && indexed(i) && indexed(j) && group(i) != group(j);
  };
  for (const cif::Bond& bd : a.bonds) {
    if (!offending(bd)) continue;
    long count = 0;
    for (const cif::Bond& o : a.bonds) count += offending(o);
    out.push_back({"backbone_bond_index_group",
                   "Backbone-bonded " + a.site.chainId[bd.u] + std::to_string(a.site.resId[bd.u]) + "-" +
                       a.site.chainId[bd.v] + std::to_string(a.site.resId[bd.v]) + " have different index groups (" +
                       std::to_string(group(bd.u)) + ", " + std::to_string(group(bd.v)) +
                       "), hiding connectivity from the model. Use one index group for the continuous indexed scaffold.",
                   count});
    return;
  }
  const vector<int> starts = atoms::tokenStarts(a);
  vector<std::pair<int, int>> pairs;
  for (int x = 0; x < c.nTok; ++x) {
    for (int y = x + 1; y < c.nTok; ++y) {
      if (group(starts[x]) == group(starts[y]) && a.ids.moleculeIid[starts[x]] == a.ids.moleculeIid[starts[y]] &&
          c.b.tokenBondPathLength[(size_t)x * c.nTok + y] == 0) {
        pairs.push_back({x, y});
      }
    }
  }
  if (pairs.empty()) return;
  string labels;
  for (size_t k = 0; k < pairs.size() && k < 10; ++k) {
    const int p = starts[pairs[k].first], q = starts[pairs[k].second];
    labels += (k ? ", " : "") + a.site.chainId[p] + "/" + std::to_string(a.site.resId[p]) + "-" + a.site.chainId[q] +
              "/" + std::to_string(a.site.resId[q]);
  }
  out.push_back({"index_group_molecule_must_be_connected",
                 std::to_string(pairs.size()) + " token pair(s) of one index group have no path between them in the "
                 "bond graph though they are one molecule: " + labels + (pairs.size() > 10 ? " ..." : ""),
                 (long)pairs.size()});
}

template <class B>
inline void coordinatesNotNan(Context<B>& c, vector<Violation>& out) {
  bool nan = false, inf = false;
  long nonzero = 0;
  for (int t = 0; t < c.nTok; ++t) {
    for (int s = 0; s < 37; ++s) {
      const size_t at = (size_t)t * 37 + s;
      const float* v = &c.b.coords[at * 3];
      if (!c.b.atom37Mask[at]) {
        if (v[0] != 0 || v[1] != 0 || v[2] != 0) ++nonzero;
        continue;
      }
      const bool conditioned = std::isfinite(c.b.condCoord[at * 3]) || std::isfinite(c.b.condCoord[at * 3 + 1]) ||
                               std::isfinite(c.b.condCoord[at * 3 + 2]) || (s == 1 && c.b.tokenType[t] == tables::TT_ATOMIZED);
      if (!conditioned) continue;
      for (int k = 0; k < 3; ++k) {
        nan = nan || std::isnan(v[k]);
        inf = inf || std::isinf(v[k]);
      }
    }
  }
  if (nan) out.push_back({"nan_in_valid_coords", "NaN found in atom coordinates", 1});
  if (inf) out.push_back({"inf_in_valid_coords", "Inf found in atom coordinates", 1});
  if (nonzero) {
    out.push_back({"nonzero_padding_coords", std::to_string(nonzero) + " invalid atom slot(s) have non-zero coordinates",
                   nonzero});
  }
}

template <class B>
inline void elementEncoding(Context<B>& c, vector<Violation>& out) {
  const int M = tables::TOKEN_TO_IDX.at("<M>");
  long leaked = 0, missing = 0;
  for (int t = 0; t < c.nTok; ++t) {
    for (int s = 0; s < 37; ++s) {
      const size_t at = (size_t)t * 37 + s;
      if (c.b.conditionSequence[t] == M) leaked += c.b.element[at] != -1;
      else if (c.b.atom37Mask[at] && c.b.element[at] < 0) ++missing;
    }
  }
  if (leaked) {
    out.push_back({"element_leakage",
                   std::to_string(leaked) + " atom slot(s) in non-seq-conditioned tokens expose element info", leaked});
  }
  if (missing) {
    out.push_back({"missing_element_in_seq_conditioned",
                   std::to_string(missing) + " valid atom(s) in seq-conditioned tokens lack element info", missing});
  }
}

template <class B>
inline void sequenceConditioning(Context<B>& c, vector<Violation>& out) {
  const int M = tables::TOKEN_TO_IDX.at("<M>"), A = tables::TOKEN_TO_IDX.at("<A>");
  const int nTypes = (int)tables::TOKEN_TO_IDX.size();
  bool range = false;
  long mismatch = 0, notA = 0;
  for (int t = 0; t < c.nTok; ++t) {
    if (c.b.residueType[t] < 0 || c.b.residueType[t] >= nTypes) range = true;
    if (c.b.conditionSequence[t] != M && c.b.residueType[t] != c.b.conditionSequence[t]) ++mismatch;
    if (c.b.tokenType[t] == tables::TT_ATOMIZED && c.b.conditionSequence[t] != A) ++notA;
  }
  if (range) out.push_back({"residue_type_out_of_range", "Ground-truth residue_type contains out-of-range indices", 1});
  if (mismatch) {
    out.push_back({"seq_conditioned_residue_type_mismatch",
                   std::to_string(mismatch) + " seq-conditioned token(s): condition_sequence != residue_type", mismatch});
  }
  if (notA) {
    out.push_back({"atomized_condition_sequence_not_A",
                   std::to_string(notA) + " atomized token(s) have condition_sequence != <A>", notA});
  }
}

inline bool isClose(double a, double b, bool equalNan = false) {
  if (std::isnan(a) || std::isnan(b)) return equalNan && std::isnan(a) && std::isnan(b);
  return std::fabs(a - b) <= 1e-8 + 1e-5 * std::fabs(b);
}

template <class B>
inline void sequenceProbabilities(Context<B>& c, vector<Violation>& out) {
  const atoms::AtomArray& a = c.a;
  vector<int> starts = residueStarts(a);
  const size_t R = starts.size();
  starts.push_back((int)a.n());
  vector<std::array<double, 28>> probs(R);
  vector<std::array<bool, 28>> spec(R);
  for (auto& p : probs) p.fill(0);
  for (auto& s : spec) s.fill(false);
  for (int ch = 0; ch < 28; ++ch) {
    const string tok = SEQ_PROB_TOKENS[ch];
    const vector<double>* values = floats1(a, "condition_sequence-probability-" + tok + "_1_residue");
    if (!values) continue;
    long badMask = 0, badValue = 0;
    for (size_t r = 0; r < R; ++r) {
      const bool m0 = !std::isnan((*values)[starts[r]]);
      bool mb = false, vb = false;
      for (int i = starts[r]; i < starts[r + 1]; ++i) {
        const bool m = !std::isnan((*values)[i]);
        mb = mb || m != m0;
        vb = vb || (m && !isClose((*values)[i], (*values)[starts[r]], true));
      }
      badMask += mb;
      badValue += vb;
      probs[r][ch] = (*values)[starts[r]];
      spec[r][ch] = m0;
    }
    if (badMask) {
      out.push_back({"sequence_probability_nonuniform_residue_mask",
                     "sequence-probability-" + tok + " has a nonuniform mask in " + std::to_string(badMask) + " residue(s)", badMask});
    }
    if (badValue) {
      out.push_back({"sequence_probability_nonuniform_residue_value",
                     "sequence-probability-" + tok + " has nonuniform values in " + std::to_string(badValue) + " residue(s)", badValue});
    }
  }
  auto sit = a.ann.strings.find("condition_sequence_1_residue");
  const vector<string>* cSeq = sit == a.ann.strings.end() ? nullptr : &sit->second;
  vector<uint8_t> seqConditioned(R, 0);
  bool anySpec = false, anySeq = false;
  for (size_t r = 0; r < R; ++r) {
    seqConditioned[r] = cSeq && (*cSeq)[starts[r]] != "<M>";
    anySeq = anySeq || seqConditioned[r];
    for (int ch = 0; ch < 28; ++ch) anySpec = anySpec || spec[r][ch];
  }
  if (!anySpec && !anySeq) return;
  static const std::set<string> AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                                      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"};
  static const std::set<string> RNA = {"A", "C", "G", "U"}, DNA = {"DA", "DC", "DG", "DT"};
  long nonfinite = 0, outOfRange = 0, wrongType = 0, exceeds = 0, below = 0, incomplete = 0, conflicting = 0;
  for (size_t r = 0; r < R; ++r) {
    const string& res = a.site.resName[starts[r]];
    const bool aa = AA.count(res) || res == "UNK" || res == "<M>";
    const bool rna = RNA.count(res) || res == "N" || res == "<M>";
    const bool dna = DNA.count(res) || res == "DN" || res == "<M>";
    std::array<bool, 28> compatible{};
    bool anyCompatible = false, fully = true, all = true;
    double mass = 0;
    for (int ch = 0; ch < 28; ++ch) {
      compatible[ch] = ch < 20 ? aa : ch < 24 ? rna : dna;
      anyCompatible = anyCompatible || compatible[ch];
      fully = fully && (spec[r][ch] || !compatible[ch]);
      all = all && spec[r][ch];
      const double p = probs[r][ch];
      const bool fin = std::isfinite(p);
      if (spec[r][ch] && !fin) ++nonfinite;
      if (spec[r][ch] && fin && (p < 0 || p > 1)) ++outOfRange;
      if (spec[r][ch] && fin && p > 0 && !compatible[ch]) ++wrongType;
      if (spec[r][ch] && fin) mass += p;
    }
    if (mass > 1 + 1e-6) ++exceeds;
    const string target = cSeq ? (*cSeq)[starts[r]] : "<M>";
    bool canonical = false;
    for (const char* t : SEQ_PROB_TOKENS) canonical = canonical || target == t;
    const bool noncanonical = seqConditioned[r] && !canonical;
    if (anyCompatible && fully && !noncanonical && mass < 1 - 1e-6) ++below;
    if (seqConditioned[r] && !all) ++incomplete;
    if (seqConditioned[r]) {
      bool conflict = false;
      for (int ch = 0; ch < 28; ++ch) {
        conflict = conflict || (spec[r][ch] && !isClose(probs[r][ch], target == SEQ_PROB_TOKENS[ch] ? 1.0 : 0.0));
      }
      conflicting += conflict;
    }
  }
  auto push = [&](const char* check, long n, const string& msg) { if (n) out.push_back({check, msg, n}); };
  push("sequence_probability_nonfinite", nonfinite, std::to_string(nonfinite) + " specified sequence probability value(s) are not finite");
  push("sequence_probability_out_of_range", outOfRange, std::to_string(outOfRange) + " specified sequence probability value(s) fall outside [0, 1]");
  push("sequence_probability_wrong_token_type", wrongType, std::to_string(wrongType) +
       " positive sequence probability value(s) target identities from a different polymer type");
  push("sequence_probability_sum_exceeds_one", exceeds, "specified sequence probabilities sum above 1 on " + std::to_string(exceeds) + " residue(s)");
  push("sequence_probability_sum_below_one", below, "fully specified sequence probabilities sum below 1 on " + std::to_string(below) + " residue(s)");
  push("sequence_probability_incomplete_for_c_seq", incomplete, std::to_string(incomplete) +
       " C_SEQ-conditioned residue(s) do not have all canonical-polymer probabilities explicitly specified");
  push("sequence_probability_conflicts_with_c_seq", conflicting, std::to_string(conflicting) +
       " residue(s) have sequence probabilities that conflict with C_SEQ");
}

template <class B>
inline void centering(Context<B>& c, vector<Violation>& out) {
  const atoms::AtomArray& a = c.a;
  const vector<double>* crd = floats3(a, "condition_coordinate_1_atom");
  if (!crd || a.coordToBeNoisedX.size() != a.n()) return;
  double sum[3] = {0, 0, 0};
  long count = 0;
  for (size_t i = 0; i < a.n(); ++i) {
    if (!anyFinite3(&(*crd)[i * 3])) continue;
    sum[0] += a.coordToBeNoisedX[i];
    sum[1] += a.coordToBeNoisedY[i];
    sum[2] += a.coordToBeNoisedZ[i];
    ++count;
  }
  if (!count) return;
  const double norm = std::hypot(sum[0] / count, sum[1] / count, sum[2] / count);
  if (!(norm < 1e-3)) {
    out.push_back({"coordinates_not_centered", "Atom-array centroid norm " + fmt("%.2e", norm) + " (threshold 1e-3)", 1});
  }
  double fsum[3] = {0, 0, 0};
  long fn = 0;
  for (size_t at = 0; at < (size_t)c.nTok * 37; ++at) {
    if (!c.b.atom37Mask[at]) continue;
    if (!(std::isfinite(c.b.condCoord[at * 3]) || std::isfinite(c.b.condCoord[at * 3 + 1]) ||
          std::isfinite(c.b.condCoord[at * 3 + 2]))) {
      continue;
    }
    for (int k = 0; k < 3; ++k) fsum[k] += c.b.coords[at * 3 + k];
    ++fn;
  }
  if (fn) {
    const double fnorm = std::hypot(fsum[0] / fn, fsum[1] / fn, fsum[2] / fn);
    if (!(fnorm < 1e-3)) {
      out.push_back({"feats_coordinates_not_centered", "Feats centroid norm " + fmt("%.2e", fnorm) + " (threshold 1e-3)", 1});
    }
  }
}

template <class B>
inline void supportsLocalLatent(Context<B>& c, vector<Violation>& out) {
  const atoms::AtomArray& a = c.a;
  const vector<int> starts = atoms::tokenStarts(a);
  const vector<int>* idx = ints(a, "condition_index_1_residue");
  const vector<int>* segMin = ints(a, "annotation_expsegmin_1_residue");
  long missingRef = 0, invalid = 0, missing = 0;
  for (size_t t = 0; t + 1 < starts.size(); ++t) {
    const int s = starts[t];
    bool inContext = false;
    for (int i = s; i < starts[t + 1]; ++i) inContext = inContext || (idx && !(*idx)[i]);
    const bool atomizedTok = a.atomize.size() == a.n() && a.atomize[s];
    const bool expected = atoms::localLatentResidues().count(a.site.resName[s]) && !atomizedTok && !inContext;
    const bool has = c.b.supportsLocalLatent[t] != 0;
    if (has && !c.b.atom37Mask[t * 37 + 1] && !(segMin && (*segMin)[s] != -1)) ++missingRef;
    if (has && !expected) ++invalid;
    if (!has && expected) ++missing;
  }
  if (missingRef) {
    out.push_back({"local_latent_reference_atom_missing",
                   std::to_string(missingRef) + " local-latent token(s) lack their atom37 slot-1 reference atom", missingRef});
  }
  if (invalid) {
    out.push_back({"supports_local_latent_ineligible",
                   std::to_string(invalid) + " token(s) have supports_local_latent=True but are not eligible", invalid});
  }
  if (missing) {
    out.push_back({"supports_local_latent_missing",
                   std::to_string(missing) + " eligible token(s) have supports_local_latent=False", missing});
  }
}

template <class B>
inline void noOxtOnLocalLatent(Context<B>& c, vector<Violation>& out) {
  const atoms::AtomArray& a = c.a;
  std::map<string, int> residues;
  for (const int s : residueStarts(a)) ++residues[chainOf(a, s)];
  long bad = 0;
  for (size_t i = 0; i < a.n(); ++i) {
    if (a.site.atomName[i] != "OXT" || (a.atomize.size() == a.n() && a.atomize[i])) continue;
    if (!c.b.supportsLocalLatent[a.tokenId[i]]) continue;
    const int ct = a.site.chainType[i];
    const bool protein = ct == tables::CT_POLYPEPTIDE_D || ct == tables::CT_POLYPEPTIDE_L || ct == tables::CT_CYCLIC_PSEUDO_PEPTIDE;
    if (protein && residues[chainOf(a, i)] > 1) ++bad;
  }
  if (bad) {
    out.push_back({"oxt_on_local_latent_token",
                   std::to_string(bad) + " OXT atom(s) found on supports_local_latent non-atomized token(s)", bad});
  }
}

/** `ValidateBatchInvariants(on_violation="raise")`: refused in upstream's words. */
template <class B>
inline void assertBatchInvariants(atoms::AtomArray& a, const B& b) {
  Context<B> c{a, b, b.n};
  vector<Violation> v;
  moleculesOnSeparateChains(c, v);
  standardAnnotationLevels(c, v);
  globalTokenIdConsistent(c, v);
  centroidAllOrNothing(c, v);
  conditioningCoordsNotNan(c, v);
  expandableSegmentConditioning(c, v);
  bondLengths(c, v);
  unindexedCoordConditioning(c, v);
  conditioning(c, v);
  atomizedIndexed(c, v);
  indexGroupConnectivity(c, v);
  coordinatesNotNan(c, v);
  elementEncoding(c, v);
  sequenceConditioning(c, v);
  sequenceProbabilities(c, v);
  centering(c, v);
  supportsLocalLatent(c, v);
  noOxtOnLocalLatent(c, v);
  if (v.empty()) return;
  fprintf(stderr, "Batch invariant violation(s) for example 'design':\n");
  for (const Violation& x : v) fprintf(stderr, "  [%s] %s\n", x.check.c_str(), x.message.c_str());
  exit(1);
}

}  // namespace invariants
}  // namespace rfd4p
