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
#include "../../featurise/jsmath.h"
#include <functional>
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
  std::vector<int> bonds;      // (token * 37 + slot) pairs
  std::vector<int> resName;    // 2 words a token, the input's residue name
  const int* condIndex = nullptr;   // a token x 37
  int nTokens = 0;
};

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
  std::set<std::pair<int, int>> seen;
  auto add = [&](int i, int j) {
    if (i == j) return;
    const std::pair<int, int> key = i < j ? std::make_pair(i, j) : std::make_pair(j, i);
    if (seen.insert(key).second) a.bonds.push_back(key);
  };
  std::map<int, int> outOf;
  for (int i = 0; i < a.n; ++i) if (rec[i].inAt >= 0) outOf[rec[i].inAt] = i;
  for (size_t k = 0; k + 1 < in.bonds.size(); k += 2) {
    const auto p = outOf.find(in.bonds[k]), q = outOf.find(in.bonds[k + 1]);
    if (p != outOf.end() && q != outOf.end()) add(p->second, q->second);
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
      (void)type;
      const size_t colon = pair.find(':');
      const std::string p = pair.substr(0, colon), q = pair.substr(colon + 1);
      for (int x = st[r]; x < st[r + 1]; ++x) {
        if (a.atomName[sub[x]] != p) continue;
        for (int y = st[r]; y < st[r + 1]; ++y) if (a.atomName[sub[y]] == q) add(sub[x], sub[y]);
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
    if (i >= 0 && j >= 0) add(i, j);
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
