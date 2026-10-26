#pragma once
/**
 * A design's metrics and its pre-oracle verdict, as shared/rfd4p/metrics.js computes them - which
 * tools/check-rfd4p-metrics.py holds to upstream's own metric manager, and gate.py holds this to,
 * value for value, on the same dumped samples. Read that file for what each metric is and why;
 * this is its port over the native writer's records (`buildRecords`) and the input tables the
 * featuriser writes beside them (`g/m*`, `outputTables`).
 *
 * The arithmetic is float32 where upstream's is (a distance is `np.linalg.norm` of a float32
 * difference), and P-SEA's angles go through V8's acos and atan2 (`lf::jsmath`), as the page's do.
 */
#include "generate.cuh"
#include "metrics_tags.h"
#include "superpose.h"
#include "symmetry.h"
#include "../../featurise/jsmath.h"
#include <array>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <map>
#include <memory>
#include <set>

namespace rfd4p {
namespace metrics {

/** The input tables `--metrics=` reads, per token (and per token x 37 slot for the atom ones). */
struct Inputs {
  std::vector<int> tok;        // isPolymer, isProtein, C_SEC
  std::vector<int> chainIid;   // 4 words a token
  std::vector<int> compType;   // 8 words a token
  std::vector<int> seq;        // 2 words a token
  std::vector<float> seqProb;  // 28 a token
  std::vector<int> hot;        // a token x 37
  std::vector<int> resName;    // 2 words a token, the input's residue name
  const int* condIndex = nullptr;   // a token x 37
  int nTokens = 0;
  // For the motif RMSDs: per token its pn_unit (4 words) and molecule and entity (2); every input
  // atom (`IN_ATOM_WORDS` each, export_input.mjs has the layout); the input's bonds as (atom,
  // atom, type); and the conditioning a token x 37: C_CRD, C_INTCRD and the coordinate group.
  std::vector<int> pnUnit, tokMol, inAtom, inBonds;
  const float* condCoord = nullptr;
  const float* condIntCrd = nullptr;
  const int* coordGroup = nullptr;
};
constexpr int IN_ATOM_WORDS = 24;

static const char* MASKED = "<M>";
static const char* STANDARD_AA[] = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                                     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"};
enum { SSE_NONE = 0, SSE_HELIX = 1, SSE_SHEET = 2, SSE_LOOP = 3 };

inline bool isStandardAa(const std::string& r) {
  for (const char* s : STANDARD_AA) if (r == s) return true;
  return false;
}
/** biotite's `link_type` for the residues a design is made of; "" where the table has none. */
inline std::string standardLink(const std::string& r) {
  if (r == "GLY") return "PEPTIDE LINKING";
  if (isStandardAa(r) || r == "UNK") return "L-PEPTIDE LINKING";
  if (r == "A" || r == "C" || r == "G" || r == "U" || r == "N") return "RNA LINKING";
  if (r == "DA" || r == "DC" || r == "DG" || r == "DT" || r == "DN") return "DNA LINKING";
  return "";
}
inline bool peptideLink(const std::string& t) {
  return t == "PEPTIDE LINKING" || t == "L-PEPTIDE LINKING" || t == "D-PEPTIDE LINKING";
}
inline bool nucleicLink(const std::string& t) { return t == "RNA LINKING" || t == "DNA LINKING"; }
inline bool aminoAcidType(const std::string& t) {
  static const std::set<std::string> types = {
      "D-BETA-PEPTIDE, C-GAMMA LINKING", "D-GAMMA-PEPTIDE, C-DELTA LINKING", "D-PEPTIDE LINKING",
      "D-PEPTIDE NH3 AMINO TERMINUS", "L-BETA-PEPTIDE, C-GAMMA LINKING",
      "L-GAMMA-PEPTIDE, C-DELTA LINKING", "L-PEPTIDE COOH CARBOXY TERMINUS", "L-PEPTIDE LINKING",
      "L-PEPTIDE NH3 AMINO TERMINUS", "PEPTIDE LINKING"};
  return types.count(t) > 0;
}

/** The array the metrics read: one entry per written atom. */
struct Array {
  int n = 0;
  std::vector<float> coord;
  std::vector<std::string> resName, atomName, element, chainId, insCode, chainIid, compType, cSeq;
  std::vector<int> resId, tokenId, cSec;
  std::vector<uint8_t> isPolymer, isProtein, atomize, crd, internal, hot, uii, hasInput;
  std::vector<std::array<float, 28>> seqProb;
  std::vector<std::pair<int, int>> bonds;
  std::vector<int> bondType;   // beside `bonds`
  // The motif RMSDs' (metrics.js has each): charge, live coordinate group, `SA_CRDSRC`, the
  // pn_unit / molecule / entity, whether grafted (and from which input atom), and the C_CRD and
  // C_INTCRD values (NaN where none).
  std::vector<int> charge, coordGroup, molIid, molEntity, condAt;
  std::vector<uint8_t> crdsrc, grafted;
  std::vector<std::string> pnUnit;
  std::vector<std::array<double, 3>> crdValue, intCrd;
};

inline float dist32(const std::vector<float>& c, int i, int j) {
  const float dx = c[i * 3] - c[j * 3], dy = c[i * 3 + 1] - c[j * 3 + 1], dz = c[i * 3 + 2] - c[j * 3 + 2];
  const float s = (dx * dx + dy * dy) + dz * dz;
  return (float)std::sqrt((double)s);
}

inline std::vector<int> segmentStarts(const Array& a, bool withInsCode) {
  std::vector<int> out;
  for (int i = 0; i < a.n; ++i) {
    if (i == 0 || a.chainId[i] != a.chainId[i - 1] || a.resId[i] != a.resId[i - 1] ||
        (withInsCode && a.insCode[i] != a.insCode[i - 1])) {
      out.push_back(i);
    }
  }
  return out;
}
// atomworks' residue starts (chain, residue, insertion code - the transformation is "1"
// throughout), and biotite's, which are the same here.
inline std::vector<int> residueStarts(const Array& a) { return segmentStarts(a, true); }
inline std::vector<int> tokenStarts(const Array& a) {
  std::vector<int> out;
  for (int i = 0; i < a.n; ++i) if (i == 0 || a.tokenId[i] != a.tokenId[i - 1]) out.push_back(i);
  return out;
}

inline std::string linkType(const Array& a, int i) {
  const std::string s = standardLink(a.resName[i]);
  return s.empty() ? a.compType[i] : s;
}

/** The array's bonds: the input's between surviving atoms, plus biotite's residue bonds and links. */
inline void designBonds(Array& a, const std::vector<AtomRecord>& rec, const Inputs& in) {
  std::map<std::pair<int, int>, size_t> seen;
  // A bond both lists carry takes the second list's type (biotite's `BondList.merge`).
  auto add = [&](int i, int j, int type) {
    if (i == j) return;
    const std::pair<int, int> key = i < j ? std::make_pair(i, j) : std::make_pair(j, i);
    const auto it = seen.find(key);
    if (it != seen.end()) { a.bondType[it->second] = type; return; }
    seen[key] = a.bonds.size();
    a.bonds.push_back(key);
    a.bondType.push_back(type);
  };
  std::map<int, int> outOf;
  for (int i = 0; i < a.n; ++i) if (rec[i].inAt >= 0) outOf[rec[i].inAt] = i;
  for (size_t k = 0; k + 2 < in.inBonds.size(); k += 3) {
    const int sp = in.inAtom[(size_t)in.inBonds[k] * IN_ATOM_WORDS];
    const int sq = in.inAtom[(size_t)in.inBonds[k + 1] * IN_ATOM_WORDS];
    if (sp < 0 || sq < 0) continue;
    const auto p = outOf.find(sp), q = outOf.find(sq);
    if (p != outOf.end() && q != outOf.end()) add(p->second, q->second, in.inBonds[k + 2]);
  }
  bool inserted = false;
  for (int i = 0; i < a.n; ++i) if (rec[i].inAt < 0 && !a.atomize[i]) inserted = true;
  if (!inserted) return;
  std::vector<int> sub;
  for (int i = 0; i < a.n; ++i) if (!a.atomize[i]) sub.push_back(i);
  std::vector<int> st;
  for (size_t k = 0; k < sub.size(); ++k) {
    const int i = sub[k];
    if (k == 0 || a.chainId[i] != a.chainId[sub[k - 1]] || a.resId[i] != a.resId[sub[k - 1]] ||
        a.insCode[i] != a.insCode[sub[k - 1]]) {
      st.push_back((int)k);
    }
  }
  st.push_back((int)sub.size());
  for (size_t r = 0; r + 1 < st.size(); ++r) {
    const auto it = tables::STD_RESIDUE_BONDS.find(a.resName[sub[st[r]]]);
    if (it == tables::STD_RESIDUE_BONDS.end()) continue;
    for (const auto& [pair, type] : it->second) {
      const size_t colon = pair.find(':');
      const std::string p = pair.substr(0, colon), q = pair.substr(colon + 1);
      for (int x = st[r]; x < st[r + 1]; ++x) {
        if (a.atomName[sub[x]] != p) continue;
        for (int y = st[r]; y < st[r + 1]; ++y) if (a.atomName[sub[y]] == q) add(sub[x], sub[y], type);
      }
    }
  }
  for (size_t r = 0; r + 2 < st.size(); ++r) {
    const int cur = sub[st[r]], nxt = sub[st[r + 1]];
    if (a.chainId[nxt] != a.chainId[cur] || a.resId[nxt] - a.resId[cur] > 1) continue;
    const std::string lc = linkType(a, cur), ln = linkType(a, nxt);
    std::string c, nn;
    if (peptideLink(lc) && peptideLink(ln)) { c = "C"; nn = "N"; }
    else if (nucleicLink(lc) && nucleicLink(ln)) { c = "O3'"; nn = "P"; }
    else continue;
    int i = -1, j = -1;
    for (int x = st[r]; x < st[r + 1] && i < 0; ++x) if (a.atomName[sub[x]] == c) i = sub[x];
    for (int x = st[r + 1]; x < st[r + 2] && j < 0; ++x) if (a.atomName[sub[x]] == nn) j = sub[x];
    if (i >= 0 && j >= 0) add(i, j, 1);
  }
}

inline Array designArray(const std::vector<AtomRecord>& rec, const Inputs& in) {
  Array a;
  a.n = (int)rec.size();
  a.coord.resize((size_t)a.n * 3);
  for (int i = 0; i < a.n; ++i) {
    const AtomRecord& r = rec[i];
    const int t = r.resTok;
    a.coord[i * 3] = r.x;
    a.coord[i * 3 + 1] = r.y;
    a.coord[i * 3 + 2] = r.z;
    a.resName.push_back(r.resName.empty() ? "UNK" : r.resName);
    a.atomName.push_back(r.atomName);
    std::string el = r.element.empty() ? inferElement(r.atomName) : r.element;
    for (char& ch : el) ch = (char)toupper((unsigned char)ch);
    a.element.push_back(el);
    a.chainId.push_back(r.chain.empty() ? "A" : r.chain);
    a.insCode.push_back(r.insCode);
    a.resId.push_back(r.resId);
    a.tokenId.push_back(r.token);
    a.chainIid.push_back(unpackString(&in.chainIid[(size_t)r.token * 4], 4));
    a.isPolymer.push_back((uint8_t)in.tok[(size_t)r.token * 3]);
    a.isProtein.push_back((uint8_t)in.tok[(size_t)r.token * 3 + 1]);
    a.compType.push_back(unpackString(&in.compType[(size_t)r.token * 8], 8));
    a.atomize.push_back(r.atomized);
    a.crd.push_back(r.crd);
    a.internal.push_back(r.internal);
    a.hot.push_back(r.condAt >= 0 ? (uint8_t)in.hot[r.condAt] : 0);
    a.cSeq.push_back(unpackString(&in.seq[(size_t)t * 2], 2));
    a.cSec.push_back(in.tok[(size_t)t * 3 + 2]);
    std::array<float, 28> p{};
    for (int k = 0; k < 28; ++k) p[k] = in.seqProb[(size_t)t * 28 + k];
    a.seqProb.push_back(p);
    a.uii.push_back(r.unindexedInsertion);
    a.hasInput.push_back(r.inAt >= 0);
    a.charge.push_back(r.charge);
    a.coordGroup.push_back(r.group);
    a.grafted.push_back(r.grafted);
    a.condAt.push_back(r.condAt);
    // `SA_CRDSRC` (metrics.js): a grafted motif atom's region, else its own residue's component
    // where it is coordinate-conditioned.
    const std::string region = r.grafted && r.srcComp != "." ? r.srcComp : (r.crd || r.internal ? r.ownSrc : ".");
    a.crdsrc.push_back(region != ".");
    a.pnUnit.push_back(in.pnUnit.empty() ? a.chainIid.back() : unpackString(&in.pnUnit[(size_t)r.token * 4], 4));
    a.molIid.push_back(in.tokMol.empty() ? 0 : in.tokMol[(size_t)r.token * 2]);
    a.molEntity.push_back(in.tokMol.empty() ? 0 : in.tokMol[(size_t)r.token * 2 + 1]);
    auto value = [&](const float* t, int at) {
      if (!t || at < 0) return std::array<double, 3>{NAN, NAN, NAN};
      return std::array<double, 3>{(double)t[(size_t)at * 3], (double)t[(size_t)at * 3 + 1], (double)t[(size_t)at * 3 + 2]};
    };
    a.crdValue.push_back(value(in.condCoord, r.crdAt));
    a.intCrd.push_back(value(in.condIntCrd, r.intAt));
  }
  designBonds(a, rec, in);
  return a;
}

using Values = std::vector<std::pair<std::string, double>>;

inline Values aminoAcidComposition(const Array& a) {
  int unknown = 0;
  std::vector<std::string> standard;
  for (const int s : residueStarts(a)) {
    if (a.resName[s] == "UNK" && !(a.cSeq[s] != MASKED && a.cSeq[s] == "UNK")) ++unknown;
    if (isStandardAa(a.resName[s])) standard.push_back(a.resName[s]);
  }
  auto frac = [&](const char* r) {
    if (standard.empty()) return (double)NAN;
    int c = 0;
    for (const auto& x : standard) c += x == r;
    return (double)c / standard.size();
  };
  return {{"n_generated_unknown_residues", (double)unknown}, {"n_standard_residues", (double)standard.size()},
          {"ala_fraction", frac("ALA")}, {"val_fraction", frac("VAL")}, {"gly_fraction", frac("GLY")}};
}

inline Values sequenceAdherence(const Array& a) {
  int total = 0, bad = 0;
  for (const int s : residueStarts(a)) {
    if (a.cSeq[s] == MASKED) continue;
    ++total;
    bad += a.cSeq[s] != a.resName[s];
  }
  if (total == 0) return {{"sequence_adherence", NAN}, {"n_violations", 0.0}};
  return {{"sequence_adherence", 1.0 - (double)bad / total}, {"n_violations", (double)bad}};
}

inline Values sequenceProbabilityAdherence(const Array& a) {
  int nZero = 0, nOne = 0, zeroBad = 0, oneBad = 0;
  for (const int s : residueStarts(a)) {
    for (int k = 0; k < 28; ++k) {
      const float v = a.seqProb[s][k];
      if (std::isnan(v)) continue;
      const bool is = a.resName[s] == tables::STANDARD_POLYMER_RESIDUES[k];
      if (v == 0) { ++nZero; zeroBad += is; }
      if (v == 1) { ++nOne; oneBad += !is; }
    }
  }
  return {{"zero_adherence", nZero ? 1.0 - (double)zeroBad / nZero : NAN},
          {"one_adherence", nOne ? 1.0 - (double)oneBad / nOne : NAN},
          {"n_zero_conditions", (double)nZero}, {"n_zero_violations", (double)zeroBad},
          {"n_one_conditions", (double)nOne}, {"n_one_violations", (double)oneBad}};
}

/** Over the PROPOSED mapping for mismatches, and the grafted tokens for the unmapped count. */
inline Values unindexedInsertionMapping(const Array& a, const Inputs& in, const Insertion& proposed) {
  std::vector<int> tokens;
  if (in.condIndex) {
    for (int tk = 0; tk < in.nTokens; ++tk) if (in.condIndex[(size_t)tk * 37 + 1] == 0) tokens.push_back(tk);
  }
  if (tokens.empty()) return {{"n_unmapped_tokens", 0.0}, {"n_sequence_mismatches", 0.0}};
  std::set<int> mapped;
  for (int i = 0; i < a.n; ++i) if (a.uii[i]) mapped.insert(a.tokenId[i]);
  std::map<int, std::string> generated;
  for (const int s : tokenStarts(a)) generated[a.tokenId[s]] = a.resName[s];
  std::map<int, int> target;
  for (size_t k = 0; k < proposed.source.size(); ++k) {
    if (proposed.target[k] >= 0) target[proposed.source[k]] = proposed.target[k];
  }
  int mismatches = 0;
  for (const int u : tokens) {
    if (unpackString(&in.seq[(size_t)u * 2], 2) == MASKED) continue;
    const auto t = target.find(u);
    const auto g = t == target.end() ? generated.end() : generated.find(t->second);
    if (g == generated.end() || g->second != unpackString(&in.resName[(size_t)u * 2], 2)) ++mismatches;
  }
  return {{"n_unmapped_tokens", (double)tokens.size() - (double)mapped.size()},
          {"n_sequence_mismatches", (double)mismatches}};
}

inline Values clashMetrics(const Array& a, double threshold = 1.5) {
  const int n = a.n;
  std::set<std::pair<int, int>> bonded(a.bonds.begin(), a.bonds.end());
  std::map<std::array<long, 3>, std::vector<int>> grid;
  auto cellOf = [&](int i) {
    return std::array<long, 3>{(long)std::floor(a.coord[i * 3] / threshold),
                               (long)std::floor(a.coord[i * 3 + 1] / threshold),
                               (long)std::floor(a.coord[i * 3 + 2] / threshold)};
  };
  for (int i = 0; i < n; ++i) grid[cellOf(i)].push_back(i);
  struct Pair { int i, j; float d; };
  std::vector<Pair> pairs;
  for (int i = 0; i < n; ++i) {
    const auto c = cellOf(i);
    for (long dx = -1; dx <= 1; ++dx) for (long dy = -1; dy <= 1; ++dy) for (long dz = -1; dz <= 1; ++dz) {
      const auto it = grid.find({c[0] + dx, c[1] + dy, c[2] + dz});
      if (it == grid.end()) continue;
      for (const int j : it->second) {
        if (j <= i || bonded.count({i, j}) || (a.crd[i] && a.crd[j])) continue;
        const float d = dist32(a.coord, i, j);
        if (d <= threshold) pairs.push_back({i, j, d});
      }
    }
  }
  auto sameRes = [&](int i, int j) { return a.chainIid[i] == a.chainIid[j] && a.resId[i] == a.resId[j]; };
  auto category = [&](int i, int j) {
    const bool pi = a.isPolymer[i], pj = a.isPolymer[j];
    if (sameRes(i, j)) return pi && pj ? 0 : (!pi && !pj ? 1 : -1);
    return pi && pj ? 2 : (pi != pj ? 3 : 4);
  };
  auto fraction = [&](int cat) {
    std::vector<double> best(n, INFINITY);
    for (const Pair& p : pairs) {
      if (cat >= 0 && category(p.i, p.j) != cat) continue;
      best[p.i] = std::min(best[p.i], (double)p.d);
      best[p.j] = std::min(best[p.j], (double)p.d);
    }
    int count = 0;
    for (double b : best) count += b < threshold;
    return (double)count / n;
  };
  Values out = {{"frac_atoms_clashing_1p5A", fraction(-1)}};
  const char* names[] = {"intra_residue_polymer", "intra_residue_ligand", "inter_residue_polymer_polymer",
                         "inter_residue_polymer_ligand", "inter_residue_ligand_ligand"};
  for (int c = 0; c < 5; ++c) out.push_back({std::string("frac_atoms_clashing_") + names[c] + "_1p5A", fraction(c)});
  return out;
}

inline Values bondLengthMetrics(const Array& a) {
  if (a.bonds.empty()) return {};
  auto chno = [](const std::string& e) { return e == "C" || e == "H" || e == "N" || e == "O"; };
  auto chnops = [&](const std::string& e) { return chno(e) || e == "P" || e == "S"; };
  std::vector<bool> longBond;
  int bad = 0;
  for (const auto& [i, j] : a.bonds) {
    const std::string &e1 = a.element[i], &e2 = a.element[j];
    const double limit = chno(e1) && chno(e2) ? 1.7 : (chnops(e1) && chnops(e2) ? 2.4 : 3.6);
    const bool l = dist32(a.coord, i, j) > limit;
    longBond.push_back(l);
    bad += l;
  }
  Values out = {{"bond_length.frac_unreasonable", (double)bad / a.bonds.size()}};
  auto cat = [&](const char* name, auto test) {
    int total = 0, b = 0;
    for (size_t k = 0; k < a.bonds.size(); ++k) {
      if (!test(a.bonds[k].first, a.bonds[k].second)) continue;
      ++total;
      b += longBond[k];
    }
    out.push_back({std::string("bond_length.frac_unreasonable.") + name, total ? (double)b / total : NAN});
  };
  cat("within_residue", [&](int i, int j) { return a.chainIid[i] == a.chainIid[j] && a.resId[i] == a.resId[j]; });
  cat("between_residue.within_chain", [&](int i, int j) { return a.chainIid[i] == a.chainIid[j] && a.resId[i] != a.resId[j]; });
  cat("between_chain.polymer_polymer", [&](int i, int j) { return a.chainIid[i] != a.chainIid[j] && a.isPolymer[i] && a.isPolymer[j]; });
  cat("between_chain.nonpolymer_to_polymer", [&](int i, int j) { return a.chainIid[i] != a.chainIid[j] && a.isPolymer[i] != a.isPolymer[j]; });
  return out;
}

// ---- P-SEA, float32 as biotite computes it ---------------------------------------------------
using V3 = std::array<float, 3>;
inline V3 sub3(const V3& p, const V3& q) { return {q[0] - p[0], q[1] - p[1], q[2] - p[2]}; }
inline float dot3(const V3& a, const V3& b) { return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; }
inline V3 unit3(const V3& v) { const float l = (float)std::sqrt((double)dot3(v, v)); return {v[0] / l, v[1] / l, v[2] / l}; }
inline V3 cross3(const V3& a, const V3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline bool bad3(const V3& v) { return std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2]); }
inline float distance(const V3& p, const V3& q) { const V3 d = sub3(p, q); return (float)std::sqrt((double)dot3(d, d)); }
inline float angleAt(const V3& a, const V3& b, const V3& c) {
  return (float)lf::jsmath::acos((double)dot3(unit3(sub3(b, a)), unit3(sub3(b, c))));
}
inline float dihedral(const V3& a, const V3& b, const V3& c, const V3& d) {
  const V3 v1 = unit3(sub3(a, b)), v2 = unit3(sub3(b, c)), v3 = unit3(sub3(c, d));
  const V3 n1 = cross3(v1, v2), n2 = cross3(v2, v3);
  return (float)lf::jsmath::atan2((double)dot3(cross3(n1, n2), v2), (double)dot3(n1, n2));
}

inline std::vector<bool> maskConsecutive(const std::vector<bool>& m, int number) {
  std::vector<bool> out(m.size(), false);
  for (size_t s = 0; s + number <= m.size(); ++s) {
    bool all = true;
    for (int k = 0; k < number; ++k) all = all && m[s + k];
    if (all) for (int k = 0; k < number; ++k) out[s + k] = true;
  }
  return out;
}
inline std::vector<bool> extendRegion(const std::vector<bool>& base, const std::vector<bool>& ext) {
  const size_t n = base.size();
  std::vector<bool> change(n), left(n, false), out(n);
  for (size_t i = 0; i < n; ++i) change[i] = base[i] != (i == 0 ? false : (bool)base[i - 1]);
  for (size_t i = 0; i + 1 < n; ++i) left[i] = change[i + 1] && base[i + 1];
  for (size_t i = 0; i < n; ++i) out[i] = base[i] || ((left[i] || (change[i] && !base[i])) && ext[i]);
  return out;
}
inline std::vector<bool> maskRegionsWithContacts(const std::vector<V3>& coord, const std::vector<bool>& cand,
                                                 int minContacts, double minD, double maxD) {
  std::vector<V3> ok;
  for (const V3& c : coord) if (!bad3(c)) ok.push_back(c);
  if (ok.empty()) return std::vector<bool>(cand.size(), false);
  std::vector<int> contacts(coord.size(), 0);
  for (size_t i = 0; i < cand.size(); ++i) {
    if (!cand[i]) continue;
    for (const V3& q : ok) {
      const float d = distance(coord[i], q);
      if (d <= maxD && d > minD) ++contacts[i];
    }
  }
  std::vector<bool> out(cand.size(), false);
  std::vector<size_t> edges;
  for (size_t i = 0; i < cand.size(); ++i) if (cand[i] != (i == 0 ? false : (bool)cand[i - 1])) edges.push_back(i);
  edges.push_back(coord.size());
  for (size_t r = 0; r + 1 < edges.size(); ++r) {
    int total = 0;
    for (size_t i = edges[r]; i < edges[r + 1]; ++i) total += contacts[i];
    if (total >= minContacts) for (size_t i = edges[r]; i < edges[r + 1]; ++i) out[i] = true;
  }
  return out;
}

/** biotite's `annotate_sse`: 'a', 'b', 'c', or 0 for a residue with no CA. */
inline std::vector<char> annotateSse(const Array& a) {
  const std::vector<int> st = residueStarts(a);
  const V3 nan3 = {NAN, NAN, NAN};
  auto residueOf = [&](int i) {
    int r = (int)st.size() - 1;
    while (st[r] > i) --r;
    return r;
  };
  std::vector<V3> ca(st.size(), nan3);
  for (int i = 0; i < a.n; ++i) {
    const std::string link = standardLink(a.resName[i]);
    const bool aa = !link.empty() ? peptideLink(link) : aminoAcidType(a.compType[i]);
    if (!aa || a.atomName[i] != "CA") continue;
    ca[residueOf(i)] = {a.coord[i * 3], a.coord[i * 3 + 1], a.coord[i * 3 + 2]};
  }
  if (ca.size() <= 5) {
    std::vector<char> out;
    for (const V3& c : ca) out.push_back(bad3(c) ? 0 : 'c');
    return out;
  }
  std::vector<int> breaks;
  for (int i = 1; i < a.n; ++i) {
    const int d = a.resId[i] - a.resId[i - 1];
    if (d != 0 && d != 1) breaks.push_back(residueOf(i));
  }
  std::vector<V3> c;
  std::vector<bool> real;
  size_t b = 0;
  for (size_t r = 0; r < ca.size(); ++r) {
    while (b < breaks.size() && breaks[b] == (int)r) { c.push_back(nan3); real.push_back(false); ++b; }
    c.push_back(ca[r]);
    real.push_back(true);
  }
  const int L = (int)c.size();
  std::vector<double> d2(L, NAN), d3(L, NAN), d4(L, NAN), ri(L, NAN), ai(L, NAN);
  for (int i = 1; i < L - 1; ++i) d2[i] = distance(c[i - 1], c[i + 1]);
  for (int i = 1; i < L - 2; ++i) d3[i] = distance(c[i - 1], c[i + 2]);
  for (int i = 1; i < L - 3; ++i) d4[i] = distance(c[i - 1], c[i + 3]);
  for (int i = 1; i < L - 1; ++i) ri[i] = angleAt(c[i - 1], c[i], c[i + 1]);
  for (int i = 1; i < L - 2; ++i) ai[i] = dihedral(c[i - 1], c[i], c[i + 1], c[i + 2]);
  const double deg = M_PI / 180;
  auto in = [](double v, double lo, double hi) { return v >= lo && v <= hi; };
  std::vector<bool> relaxedHelix(L), strictHelix(L), relaxedStrand(L), strictStrand(L);
  for (int i = 0; i < L; ++i) {
    const bool d3h = in(d3[i], 5.3 - 0.5, 5.3 + 0.5), rh = in(ri[i], (89 - 12) * deg, (89 + 12) * deg);
    relaxedHelix[i] = d3h || rh;
    strictHelix[i] = (d3h && in(d4[i], 6.4 - 0.6, 6.4 + 0.6)) || (rh && in(ai[i], (50 - 20) * deg, (50 + 20) * deg));
    relaxedStrand[i] = in(d3[i], 9.9 - 0.9, 9.9 + 0.9);
    strictStrand[i] = (in(d2[i], 6.7 - 0.6, 6.7 + 0.6) && relaxedStrand[i] && in(d4[i], 12.4 - 1.1, 12.4 + 1.1)) ||
                      (in(ri[i], (124 - 14) * deg, (124 + 14) * deg) &&
                       (in(ai[i], -180 * deg, -125 * deg) || in(ai[i], 145 * deg, 180 * deg)));
  }
  const std::vector<bool> helix = extendRegion(maskConsecutive(strictHelix, 5), relaxedHelix);
  const std::vector<bool> shortStrand = maskRegionsWithContacts(c, maskConsecutive(strictStrand, 3), 5, 4.2, 5.2);
  std::vector<bool> strandBase = maskConsecutive(strictStrand, 4);
  for (int i = 0; i < L; ++i) strandBase[i] = strandBase[i] || shortStrand[i];
  const std::vector<bool> strand = extendRegion(strandBase, relaxedStrand);
  std::vector<char> out;
  for (int i = 0; i < L; ++i) {
    if (!real[i]) continue;
    out.push_back(bad3(c[i]) ? 0 : strand[i] ? 'b' : helix[i] ? 'a' : 'c');
  }
  return out;
}

struct Secondary {
  std::vector<int> sse;
  std::vector<bool> valid;
};
inline Secondary secondaryStructure(const Array& a) {
  const std::vector<int> st = residueStarts(a);
  std::vector<int> sse(st.size(), SSE_NONE);
  std::vector<bool> valid(st.size(), false);
  const std::vector<char> codes = annotateSse(a);
  if (codes.size() == st.size()) {
    for (size_t r = 0; r < st.size(); ++r) {
      sse[r] = codes[r] == 'a' ? SSE_HELIX : codes[r] == 'b' ? SSE_SHEET : codes[r] == 'c' ? SSE_LOOP : SSE_NONE;
      valid[r] = codes[r] != 0;
    }
  }
  Secondary s{std::vector<int>(a.n), std::vector<bool>(a.n)};
  for (size_t r = 0; r < st.size(); ++r) {
    const int end = r + 1 < st.size() ? st[r + 1] : a.n;
    for (int i = st[r]; i < end; ++i) { s.sse[i] = sse[r]; s.valid[i] = valid[r]; }
  }
  return s;
}

inline std::string chainOfInterest(const Array& a) {
  std::set<std::string> chainSet(a.chainIid.begin(), a.chainIid.end());
  std::vector<std::string> chains(chainSet.begin(), chainSet.end());
  if (chains.size() < 2) return chains.empty() ? std::string() : chains[0];
  std::vector<std::string> protein;
  for (const auto& ch : chains) {
    bool all = true;
    for (int i = 0; i < a.n; ++i) if (a.chainIid[i] == ch && !a.isProtein[i]) all = false;
    if (all) protein.push_back(ch);
  }
  if (protein.size() == 1) return protein[0];
  if (protein.empty()) return chains[0];
  auto mean = [&](const std::string& ch, auto test) {
    int total = 0, hit = 0;
    for (int i = 0; i < a.n; ++i) if (a.chainIid[i] == ch) { ++total; hit += test(i); }
    return (double)hit / total;
  };
  std::vector<double> seq;
  for (const auto& ch : protein) seq.push_back(mean(ch, [&](int i) { return a.cSeq[i] != MASKED; }));
  const double m = *std::min_element(seq.begin(), seq.end());
  std::vector<std::string> least;
  for (size_t k = 0; k < protein.size(); ++k) if (seq[k] == m) least.push_back(protein[k]);
  if (least.size() == 1) return least[0];
  std::vector<double> crd;
  for (const auto& ch : least) {
    crd.push_back(mean(ch, [&](int i) { return (bool)a.crd[i]; }) + mean(ch, [&](int i) { return (bool)a.internal[i]; }));
  }
  const double mc = *std::min_element(crd.begin(), crd.end());
  for (size_t k = 0; k < least.size(); ++k) if (crd[k] == mc) return least[k];
  return least[0];
}

inline Values sseFractions(const std::vector<int>& v, const std::string& suffix) {
  if (v.empty()) {
    return {{"helix_fraction" + suffix, NAN}, {"sheet_fraction" + suffix, NAN},
            {"loop_fraction" + suffix, NAN}, {"num_ss_elements" + suffix, NAN}};
  }
  auto frac = [&](int c) { int k = 0; for (int x : v) k += x == c; return (double)k / v.size(); };
  int elements = 0;
  for (size_t i = 0; i < v.size(); ++i) if (i == 0 || v[i] != v[i - 1]) ++elements;
  return {{"helix_fraction" + suffix, frac(SSE_HELIX)}, {"sheet_fraction" + suffix, frac(SSE_SHEET)},
          {"loop_fraction" + suffix, frac(SSE_LOOP)}, {"num_ss_elements" + suffix, (double)elements}};
}

inline Values sseDistribution(const Array& a, const Secondary& ss) {
  const std::string coi = chainOfInterest(a);
  std::vector<int> all, own;
  for (const int s : tokenStarts(a)) {
    if (!ss.valid[s]) continue;
    all.push_back(ss.sse[s]);
    if (a.chainIid[s] == coi) own.push_back(ss.sse[s]);
  }
  Values out = sseFractions(all, "");
  for (const auto& kv : sseFractions(own, "_coi")) out.push_back(kv);
  return out;
}

/** No C_SEC column and none conditioned are the same answer, so the column's presence is not read. */
inline Values sseAccuracy(const Array& a, const Secondary& ss) {
  int conditioned = 0;
  std::vector<int> at;
  for (int i = 0; i < a.n; ++i) {
    if (a.cSec[i] == SSE_NONE) continue;
    ++conditioned;
    if (ss.valid[i]) at.push_back(i);
  }
  if (conditioned == 0) return {};
  if (at.empty()) {
    return {{"overall_accuracy", NAN}, {"helix_recall", NAN}, {"sheet_recall", NAN},
            {"loop_recall", NAN}, {"n_conditioned_atoms", (double)conditioned}};
  }
  auto recall = [&](int cls) {
    int of = 0, hit = 0;
    for (int i : at) if (a.cSec[i] == cls) { ++of; hit += ss.sse[i] == cls; }
    return of ? (double)hit / of : NAN;
  };
  int right = 0;
  for (int i : at) right += a.cSec[i] == ss.sse[i];
  return {{"overall_accuracy", (double)right / at.size()}, {"helix_recall", recall(SSE_HELIX)},
          {"sheet_recall", recall(SSE_SHEET)}, {"loop_recall", recall(SSE_LOOP)},
          {"n_conditioned_atoms", (double)at.size()}};
}

inline Values hotspotObedience(const Array& a, double limit = 4.5) {
  std::vector<int> hot;
  for (int i = 0; i < a.n; ++i) if (a.hot[i]) hot.push_back(i);
  if (hot.empty()) return {{"hotspot_obedience", NAN}};
  std::set<std::string> chains;
  for (int i : hot) chains.insert(a.chainIid[i]);
  int obeyed = 0;
  for (const auto& ch : chains) {
    std::vector<int> binder;
    for (int j = 0; j < a.n; ++j) if (a.chainIid[j] != ch && !a.crd[j]) binder.push_back(j);
    for (int i : hot) {
      if (a.chainIid[i] != ch) continue;
      for (int j : binder) if (dist32(a.coord, i, j) < limit) { ++obeyed; break; }
    }
  }
  return {{"hotspot_obedience", (double)obeyed / hot.size()}};
}

/** Protein residue identity against the indexed input's own residue names. */
inline Values sequenceSimilarityToInput(const Array& a, const Inputs& in) {
  std::vector<std::string> generated, given;
  for (const int s : residueStarts(a)) if (a.isProtein[s]) generated.push_back(a.resName[s]);
  for (int tk = 0; tk < in.nTokens; ++tk) {
    if (in.condIndex && in.condIndex[(size_t)tk * 37 + 1] == 0) continue;
    if (!in.tok[(size_t)tk * 3 + 1]) continue;
    given.push_back(unpackString(&in.resName[(size_t)tk * 2], 2));
  }
  if (generated.empty() || given.empty() || generated.size() != given.size()) {
    return {{"sequence_identity_to_input", NAN}};
  }
  int same = 0;
  for (size_t k = 0; k < given.size(); ++k) same += generated[k] == given[k];
  return {{"sequence_identity_to_input", (double)same / generated.size()}};
}

// ---- The motif RMSDs (metrics.js has every note) ----------------------------------------------

inline bool internalGroup(int g) { return g != -1 && g != 0; }
inline bool finite3d(const std::array<double, 3>& v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

/** `fitOnto`: biotite's `superimpose(fixed, mobile)` over two index lists, as a map. */
struct FitMap {
  superpose::Fit f;
  std::array<double, 3> operator()(double x, double y, double z) const {
    const double a = x - f.mobileCenter[0], b = y - f.mobileCenter[1], c = z - f.mobileCenter[2];
    const double* R = f.R;
    return {a * R[0] + b * R[1] + c * R[2] + f.fixedCenter[0], a * R[3] + b * R[4] + c * R[5] + f.fixedCenter[1],
            a * R[6] + b * R[7] + c * R[8] + f.fixedCenter[2]};
  }
};
inline FitMap fitOnto(const std::vector<double>& fixed, const std::vector<double>& mobile,
                      const std::vector<int>& fixedIdx, const std::vector<int>& mobileIdx) {
  std::vector<double> f, m;
  for (const int i : fixedIdx) for (int c = 0; c < 3; ++c) f.push_back(fixed[(size_t)i * 3 + c]);
  for (const int i : mobileIdx) for (int c = 0; c < 3; ++c) m.push_back(mobile[(size_t)i * 3 + c]);
  return FitMap{superpose::superimpose(f, m)};
}

inline void stageGroup(const Array& a, std::vector<double>& cwi, const std::vector<int>& atoms) {
  std::vector<double> coord(a.coord.begin(), a.coord.end());
  std::vector<double> ref((size_t)a.n * 3, 0.0);
  for (const int i : atoms) for (int c = 0; c < 3; ++c) ref[(size_t)i * 3 + c] = a.intCrd[i][c];
  const FitMap fit = fitOnto(coord, ref, atoms, atoms);
  for (const int i : atoms) {
    const auto v = fit(ref[(size_t)i * 3], ref[(size_t)i * 3 + 1], ref[(size_t)i * 3 + 2]);
    for (int c = 0; c < 3; ++c) cwi[(size_t)i * 3 + c] = v[c];
  }
}

template <class Pred>
inline std::vector<std::vector<int>> groupsBy(const Array& a, Pred keep) {
  std::map<int, std::vector<int>> groups;
  for (int i = 0; i < a.n; ++i) if (keep(i)) groups[a.coordGroup[i]].push_back(i);
  std::vector<std::vector<int>> out;
  for (auto& [g, v] : groups) { (void)g; out.push_back(v); }
  return out;
}

/** `coordsWithInsertion`: `SA_CWI`, or nothing where upstream stages none. */
inline bool coordsWithInsertion(const Array& a, const Inputs& in, const Insertion& proposed,
                                std::vector<double>& cwi) {
  const int n = a.n;
  // The input's unindexed atoms, and how many of them carry a coordinate (flag bits 1 and 16).
  int unindexedAtoms = 0, expected = 0;
  for (size_t i = 0; i < in.inAtom.size() / IN_ATOM_WORDS; ++i) {
    const int flags = in.inAtom[i * IN_ATOM_WORDS + 1];
    if (flags & 1) continue;
    ++unindexedAtoms;
    if (flags & 16) ++expected;
  }
  auto ownCrd = [&](int i) { return !a.grafted[i] && a.crd[i]; };
  auto ownInternal = [&](int i) { return !a.grafted[i] && a.internal[i]; };
  bool indexed = false;
  for (int i = 0; i < n; ++i) indexed = indexed || ownCrd(i) || ownInternal(i);
  if (!indexed && unindexedAtoms == 0) return false;
  cwi.assign((size_t)n * 3, NAN);
  if (indexed) {
    for (int i = 0; i < n * 3; ++i) cwi[i] = a.coord[i];
    for (int i = 0; i < n; ++i) if (ownCrd(i)) for (int c = 0; c < 3; ++c) cwi[(size_t)i * 3 + c] = a.crdValue[i][c];
    for (const auto& atoms : groupsBy(a, ownInternal)) {
      for (const int i : atoms) {
        if (!finite3d(a.intCrd[i])) {
          throw std::runtime_error("NaN values detected in input or generated coords for coord group " +
                                   std::to_string(a.coordGroup[atoms[0]]) + ".");
        }
      }
      stageGroup(a, cwi, atoms);
    }
  }
  if (unindexedAtoms == 0) return true;
  bool anyUii = false, anyProposed = false;
  for (int i = 0; i < n; ++i) anyUii = anyUii || a.uii[i];
  for (const int t : proposed.target) anyProposed = anyProposed || t >= 0;
  if (anyProposed && !anyUii) return true;
  auto motif = [&](int i) {
    return a.grafted[i] && (finite3d(a.crdValue[i]) || (in.coordGroup && internalGroup(in.coordGroup[a.condAt[i]])));
  };
  int placed = 0;
  for (int i = 0; i < n; ++i) {
    if (!motif(i)) continue;
    ++placed;
    if (finite3d(a.crdValue[i])) for (int c = 0; c < 3; ++c) cwi[(size_t)i * 3 + c] = a.crdValue[i][c];
  }
  for (const auto& atoms : groupsBy(a, [&](int i) { return a.crdsrc[i] && internalGroup(a.coordGroup[i]); })) {
    bool complete = true;
    for (const int i : atoms) complete = complete && finite3d(a.intCrd[i]);
    if (!complete) continue;
    stageGroup(a, cwi, atoms);
  }
  if (expected > 0 && placed != expected) {
    for (int i = 0; i < n; ++i) if (motif(i)) for (int c = 0; c < 3; ++c) cwi[(size_t)i * 3 + c] = NAN;
  }
  return true;
}

/** `sliceView`. */
inline symmetry::View sliceView(const Array& a, const std::vector<int>& atoms) {
  symmetry::View v;
  v.n = (int)atoms.size();
  std::map<int, int> local;
  for (size_t k = 0; k < atoms.size(); ++k) local[atoms[k]] = (int)k;
  for (const int g : atoms) {
    v.element.push_back(a.element[g]);
    v.charge.push_back(a.charge[g]);
    v.isPolymer.push_back(a.isPolymer[g]);
    v.chainId.push_back(a.chainId[g]);
    v.chainIid.push_back(a.chainIid[g]);
    v.resId.push_back(a.resId[g]);
    v.insCode.push_back(a.insCode[g]);
    v.resName.push_back(a.resName[g]);
    v.atomName.push_back(a.atomName[g]);
    v.pnUnit.push_back(a.pnUnit[g]);
    v.molIid.push_back(a.molIid[g]);
    v.molEntity.push_back(a.molEntity[g]);
  }
  for (size_t k = 0; k < a.bonds.size(); ++k) {
    const auto p = local.find(a.bonds[k].first), q = local.find(a.bonds[k].second);
    if (p != local.end() && q != local.end()) v.bonds.push_back({p->second, q->second, a.bondType[k]});
  }
  return v;
}

struct Comparison { bool ok = false; double rmsd = NAN; int n = 0; };

/** `structureComparison`. */
inline Comparison structureComparison(const std::vector<double>& gt, const std::vector<double>& pred,
                                      const std::vector<uint8_t>& gtAlign, const std::vector<uint8_t>& predAlign,
                                      const std::vector<uint8_t>& gtSubset, const std::vector<uint8_t>& predSubset) {
  auto pick = [](const std::vector<double>& c, const std::vector<uint8_t>& mask) {
    std::vector<int> out;
    for (size_t i = 0; i < mask.size(); ++i) if (mask[i] && symmetry::finiteAt(c, (int)i)) out.push_back((int)i);
    return out;
  };
  Comparison r;
  const std::vector<int> ga = pick(gt, gtAlign), pa = pick(pred, predAlign);
  if (ga.size() != pa.size() || ga.empty()) return r;
  const FitMap fit = fitOnto(gt, pred, ga, pa);
  std::vector<double> moved(pred.size());
  for (size_t i = 0; i < pred.size() / 3; ++i) {
    const auto v = fit(pred[i * 3], pred[i * 3 + 1], pred[i * 3 + 2]);
    for (int c = 0; c < 3; ++c) moved[i * 3 + c] = v[c];
  }
  const std::vector<int> gs = pick(gt, gtSubset), ps = pick(moved, predSubset);
  if (gs.size() != ps.size()) return r;
  r.ok = true;
  if (gs.empty()) return r;
  double sq = 0;
  for (size_t k = 0; k < gs.size(); ++k) {
    for (int c = 0; c < 3; ++c) {
      const double d = gt[(size_t)gs[k] * 3 + c] - moved[(size_t)ps[k] * 3 + c];
      sq += d * d;
    }
  }
  r.rmsd = std::sqrt(sq / gs.size());
  r.n = (int)gs.size();
  return r;
}

inline std::vector<int> classesOf(const std::vector<uint8_t>& subset) {
  bool all = true;
  for (const uint8_t m : subset) all = all && m;
  if (all) return {};
  std::vector<int> c(subset.size());
  for (size_t i = 0; i < subset.size(); ++i) c[i] = subset[i] ? 0 : -1;
  return c;
}

/** `motifRmsd`. */
template <class Sel>
inline Values motifRmsd(const Array& a, const std::vector<double>& cwi, Sel selected) {
  std::set<int> values(a.coordGroup.begin(), a.coordGroup.end());
  int total = 0;
  double weighted = 0;
  for (const int value : values) {
    std::vector<int> atoms;
    for (int i = 0; i < a.n; ++i) if (a.coordGroup[i] == value) atoms.push_back(i);
    const symmetry::View view = sliceView(a, atoms);
    std::vector<double> pred(atoms.size() * 3), gt(atoms.size() * 3);
    std::vector<uint8_t> mask(atoms.size());
    for (size_t k = 0; k < atoms.size(); ++k) {
      for (int c = 0; c < 3; ++c) {
        pred[k * 3 + c] = a.coord[(size_t)atoms[k] * 3 + c];
        gt[k * 3 + c] = cwi[(size_t)atoms[k] * 3 + c];
      }
      mask[k] = selected(atoms[k]);
    }
    gt = symmetry::permuteGtSubunits(pred, gt, view, mask, mask, true);
    const std::vector<int> classes = classesOf(mask);
    gt = symmetry::permuteGtAtoms(pred, gt, view, classes.empty() ? nullptr : &classes, &mask);
    const Comparison r = structureComparison(gt, pred, mask, mask, mask, mask);
    if (!r.ok || r.n == 0) continue;
    total += r.n;
    weighted += r.rmsd * r.rmsd * r.n;
  }
  if (total == 0) return {};
  return {{"rmsd", std::sqrt(weighted / total)}};
}

/** `inputView`: the input's indexed atoms, multi-residue polymers' terminal OXT/OP3 dropped. */
inline symmetry::View inputView(const Inputs& in, std::vector<double>& coord, std::vector<uint8_t>& isProtein) {
  const int total = (int)(in.inAtom.size() / IN_ATOM_WORDS);
  auto row = [&](int i) { return &in.inAtom[(size_t)i * IN_ATOM_WORDS]; };
  auto text = [&](int i, int at, int words) { return unpackString(row(i) + at, words); };
  std::vector<int> kept;
  for (int i = 0; i < total; ++i) if (row(i)[1] & 1) kept.push_back(i);
  // Words: 0 slot, 1 flags, 2 chain type, 3 charge, 4 residue id, 5-7 coordinates, 8 atom name,
  // 9-10 residue name, 11 element, 12 chain, 13 insertion code, 14-17 chain instance,
  // 18-21 pn_unit, 22 molecule, 23 entity.
  std::map<std::string, int> residues;
  for (size_t k = 0; k < kept.size(); ++k) {
    const int i = kept[k];
    if (k == 0 || text(i, 12, 1) != text(kept[k - 1], 12, 1) || row(i)[4] != row(kept[k - 1])[4] ||
        text(i, 13, 1) != text(kept[k - 1], 13, 1)) {
      ++residues[text(i, 14, 4)];
    }
  }
  auto protein = [](int t) { return t == 5 || t == 6 || t == 0; };
  auto nucleic = [](int t) { return t == 3 || t == 7 || t == 4; };
  std::vector<int> atoms;
  for (const int i : kept) {
    const std::string name = text(i, 8, 1);
    const int type = row(i)[2];
    const bool terminal = (name == "OXT" && protein(type)) || (name == "OP3" && nucleic(type));
    if (terminal && residues[text(i, 14, 4)] > 1 && !(row(i)[1] & 2)) continue;
    atoms.push_back(i);
  }
  symmetry::View v;
  v.n = (int)atoms.size();
  std::map<int, int> local;
  coord.assign(atoms.size() * 3, NAN);
  isProtein.assign(atoms.size(), 0);
  for (size_t k = 0; k < atoms.size(); ++k) {
    const int i = atoms[k];
    const int* r = row(i);
    local[i] = (int)k;
    for (int c = 0; c < 3; ++c) {
      float f;
      memcpy(&f, &r[5 + c], 4);
      coord[k * 3 + c] = f;
    }
    isProtein[k] = (r[1] & 4) != 0;
    v.element.push_back(text(i, 11, 1));
    v.charge.push_back(r[3]);
    v.isPolymer.push_back((r[1] & 8) != 0);
    v.chainId.push_back(text(i, 12, 1));
    v.chainIid.push_back(text(i, 14, 4));
    v.resId.push_back(r[4]);
    v.insCode.push_back(text(i, 13, 1));
    v.resName.push_back(text(i, 9, 2));
    v.atomName.push_back(text(i, 8, 1));
    v.pnUnit.push_back(text(i, 18, 4));
    v.molIid.push_back(r[22]);
    v.molEntity.push_back(r[23]);
  }
  for (size_t k = 0; k + 2 < in.inBonds.size(); k += 3) {
    const auto p = local.find(in.inBonds[k]), q = local.find(in.inBonds[k + 1]);
    if (p != local.end() && q != local.end()) v.bonds.push_back({p->second, q->second, in.inBonds[k + 2]});
  }
  return v;
}

/** `ca_rmsd_to_input`. */
inline Values caRmsdToInput(const Array& a, const Inputs& in) {
  std::vector<double> gt;
  std::vector<uint8_t> gtProtein;
  const symmetry::View gtView = inputView(in, gt, gtProtein);
  std::vector<uint8_t> gtCa(gtView.n), predCa(a.n);
  for (int i = 0; i < gtView.n; ++i) gtCa[i] = gtProtein[i] && gtView.atomName[i] == "CA";
  for (int i = 0; i < a.n; ++i) predCa[i] = a.isProtein[i] && a.atomName[i] == "CA";
  const std::vector<double> pred(a.coord.begin(), a.coord.end());
  if (gtView.n == a.n) {
    gt = symmetry::permuteGtSubunits(pred, gt, gtView, gtCa, gtCa, true);
    const std::vector<int> classes = classesOf(gtCa);
    gt = symmetry::permuteGtAtoms(pred, gt, gtView, classes.empty() ? nullptr : &classes, &gtCa);
  }
  const Comparison r = structureComparison(gt, pred, gtCa, predCa, gtCa, predCa);
  if (!r.ok) return {};
  return {{"rmsd", r.rmsd}};
}

struct Gate { std::vector<std::string> any, all, no; };
inline bool applies(const Gate& g, const std::vector<std::string>& tags) {
  auto has = [&](const std::string& t) { return std::find(tags.begin(), tags.end(), t) != tags.end(); };
  bool ok = true;
  for (const auto& t : g.all) ok = ok && has(t);
  if (!g.any.empty()) {
    bool one = false;
    for (const auto& t : g.any) one = one || has(t);
    ok = ok && one;
  }
  for (const auto& t : g.no) ok = ok && !has(t);
  return ok;
}

struct Result {
  Values metrics;
  struct Filter { std::string name; bool pass, present; };
  std::vector<Filter> filters;
  bool pass = true;
};

/** `designMetrics`: every metric the tags admit, keyed `<metric>.<value>`, and the verdict. */
inline Result designMetrics(const std::vector<AtomRecord>& rec, const Inputs& in, const Insertion& proposed,
                            const std::vector<std::string>& tags) {
  const Array a = designArray(rec, in);
  Result out;
  std::unique_ptr<Secondary> ss;
  auto sse = [&]() -> const Secondary& { if (!ss) ss.reset(new Secondary(secondaryStructure(a))); return *ss; };
  std::vector<double> cwi;
  bool staged = false;
  auto motifReference = [&]() -> const std::vector<double>& {
    if (!staged) {
      if (!coordsWithInsertion(a, in, proposed, cwi)) {
        throw std::runtime_error(
            "a motif RMSD needs the design's coordinate-conditioned atoms, and this job has none: upstream "
            "raises here too ('coords-with-insertion' is never staged), so its '3d_coordinate_conditioning' / "
            "'internal_coordinate_conditioning' / 'unindexed' tag does not apply");
      }
      staged = true;
    }
    return cwi;
  };
  struct Entry { const char* name; Gate gate; std::function<Values()> fn; };
  const std::vector<Entry> entries = {
      {"amino_acid_composition", {}, [&] { return aminoAcidComposition(a); }},
      {"bond_length_metrics", {{"design"}, {}, {"small_molecule_only", "structure_prediction"}}, [&] { return bondLengthMetrics(a); }},
      {"clash_metrics", {{"design", "structure_prediction"}, {}, {"small_molecule_only"}}, [&] { return clashMetrics(a); }},
      {"hotspot_obedience", {{"hotspot_conditioning"}, {}, {}}, [&] { return hotspotObedience(a); }},
      {"sequence_adherence", {{"design"}, {}, {}}, [&] { return sequenceAdherence(a); }},
      {"sequence_probability_adherence", {{"design"}, {}, {}}, [&] { return sequenceProbabilityAdherence(a); }},
      {"sse_distribution", {{"design"}, {}, {"small_molecule_only", "structure_prediction"}}, [&] { return sseDistribution(a, sse()); }},
      {"sse_accuracy", {{"design"}, {}, {"small_molecule_only", "structure_prediction"}}, [&] { return sseAccuracy(a, sse()); }},
      {"unindexed_insertion_mapping", {{}, {"unindexed"}, {}}, [&] { return unindexedInsertionMapping(a, in, proposed); }},
      {"sequence_similarity_to_input", {{"partial_diffusion"}, {}, {}}, [&] { return sequenceSimilarityToInput(a, in); }},
      {"design_motif_allatom_rmsd", {{"3d_coordinate_conditioning", "unindexed", "internal_coordinate_conditioning"}, {}, {}},
       [&] { return motifRmsd(a, motifReference(), [&](int i) { return a.crdsrc[i] != 0; }); }},
      {"design_unindexed_motif_allatom_rmsd", {{}, {"unindexed"}, {}},
       [&] { return motifRmsd(a, motifReference(), [&](int i) { return a.crdsrc[i] && a.uii[i]; }); }},
      {"ca_rmsd_to_input", {{"partial_diffusion"}, {}, {}}, [&] { return caRmsdToInput(a, in); }},
  };
  for (const Entry& e : entries) {
    if (!applies(e.gate, tags)) continue;
    for (const auto& [k, v] : e.fn()) out.metrics.push_back({std::string(e.name) + "." + k, v});
  }
  struct FilterSpec { const char* name; Gate gate; std::vector<std::pair<std::string, double>> max; };
  const std::vector<FilterSpec> filters = {
      {"unknown_residue_filter", {}, {{"amino_acid_composition.n_generated_unknown_residues", 0}}},
      {"sequence_adherence_filter", {{"design"}, {}, {}}, {{"sequence_adherence.n_violations", 0}}},
      {"unindexed_insertion_mapping_filter", {{}, {"unindexed"}, {}},
       {{"unindexed_insertion_mapping.n_unmapped_tokens", 0}, {"unindexed_insertion_mapping.n_sequence_mismatches", 0}}},
      {"insertion_rmsd_filter", {{}, {"unindexed", "insertion_rmsd_filter"}, {}}, {{"unindexed_metrics.insertion_rmsd", 1.0}}},
      {"motif_rmsd_filter", {{}, {"motif_rmsd_filter"}, {}}, {{"motif_rmsd.rmsd", 1.0}}},
      {"ligand_clash_filter", {{}, {"ligand_clash_filter"}, {"no_ligand"}},
       {{"clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A", 0}}},
  };
  for (const FilterSpec& f : filters) {
    if (!applies(f.gate, tags)) continue;
    bool ok = true, present = true;
    for (const auto& bound : f.max) {
      const auto it = std::find_if(out.metrics.begin(), out.metrics.end(),
                                   [&](const auto& kv) { return kv.first == bound.first; });
      if (it == out.metrics.end()) { present = false; ok = false; continue; }
      if (!(it->second <= bound.second)) ok = false;
    }
    out.filters.push_back({f.name, ok, present});
    out.pass = out.pass && ok;
  }
  return out;
}

/** The result as design-metrics.mjs writes it: NaN as the string "NaN". */
inline std::string toJson(const Result& r) {
  auto num = [](double v) { return std::isnan(v) ? std::string("\"NaN\"") : jsNumber(v); };
  std::string s = "{\"metrics\":{";
  for (size_t k = 0; k < r.metrics.size(); ++k) {
    s += (k ? "," : "") + std::string("\"") + r.metrics[k].first + "\":" + num(r.metrics[k].second);
  }
  s += "},\"preOracle\":{\"pass\":" + std::string(r.pass ? "true" : "false") + ",\"filters\":{";
  for (size_t k = 0; k < r.filters.size(); ++k) {
    s += (k ? "," : "") + std::string("\"") + r.filters[k].name + "\":{\"pass\":" +
         (r.filters[k].pass ? "true" : "false") + ",\"allMetricsPresent\":" +
         (r.filters[k].present ? "true" : "false") + "}";
  }
  return s + "}}}";
}

}  // namespace metrics
}  // namespace rfd4p
