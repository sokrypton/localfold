// The atom array the pipeline MUTATES, as one object.
//
// 🔴 `cif.h` HANDS BACK FOUR PIECES AND A PIPELINE NEEDS ONE. Reading an mmCIF naturally produces
// a column set, an id set, an annotation map and a stereochemistry record; every stage of
// `transforms.js` then changes several of them TOGETHER - `realizeExpandableSegments` inserts
// atoms, which moves every column, every annotation and every bond index at once. Four parallel
// objects is four places to forget one.
//
// This is the container the rest of the structure path is written against. It owns what
// `parseCifToAtomArray` produces and adds the columns the transforms attach.
#pragma once
#include "chirality.h"
#include "cif.h"
#include <type_traits>

namespace rfd4p {
namespace atoms {

using std::string;
using std::vector;

/**
 * The PIPELINE's random source (`createSeededRng`, mulberry32) - not the sampler's xorshift128+.
 * Both are seeded from the job's seed and neither consumes the other's stream.
 */
struct PipelineRng {
  uint32_t a;
  explicit PipelineRng(uint32_t seed) : a(seed) {}
  double nextFloat() {
    a = a + 0x6d2b79f5u;
    uint32_t t = (a ^ (a >> 15)) * (1u | a);
    t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
    return (double)((t ^ (t >> 14))) / 4294967296.0;
  }
  int nextInt(int lo, int hiExclusive) {
    return lo + (int)std::floor(nextFloat() * (hiExclusive - lo));
  }
  /**
   * A uniformly random rotation: nine Box-Muller normals, then Gram-Schmidt.
   *
   * 🔴 EIGHTEEN DRAWS, ROW BY ROW, AND THE ORDER IS THE CONTRACT. Each of the nine entries takes
   * `u1` then `u2` in that order, and `u1` is floored at 1e-12 BEFORE the log - so a port that
   * draws both up front, or fills column-major, produces a perfectly valid rotation from a
   * different point in the stream and every later draw in the pipeline moves with it.
   */
  void randomSO3(double R[3][3]) {
    double m[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const double u1 = std::max(1e-12, nextFloat());
        const double u2 = nextFloat();
        m[i][j] = std::sqrt(-2.0 * lf::jsmath::log(u1)) * lf::jsmath::cos(2.0 * M_PI * u2);
      }
    }
    double n0 = lf::jsmath::hypot(m[0][0], m[0][1], m[0][2]);
    if (n0 == 0.0) n0 = 1.0;
    for (int k = 0; k < 3; ++k) R[0][k] = m[0][k] / n0;
    const double d10 = m[1][0] * R[0][0] + m[1][1] * R[0][1] + m[1][2] * R[0][2];
    double v1[3];
    for (int k = 0; k < 3; ++k) v1[k] = m[1][k] - d10 * R[0][k];
    double n1 = lf::jsmath::hypot(v1[0], v1[1], v1[2]);
    if (n1 == 0.0) n1 = 1.0;
    for (int k = 0; k < 3; ++k) R[1][k] = v1[k] / n1;
    R[2][0] = R[0][1] * R[1][2] - R[0][2] * R[1][1];
    R[2][1] = R[0][2] * R[1][0] - R[0][0] * R[1][2];
    R[2][2] = R[0][0] * R[1][1] - R[0][1] * R[1][0];
  }
};

struct AtomArray {
  cif::AtomSite site;
  cif::Ids ids;
  cif::Annotations ann;
  vector<cif::Bond> bonds;
  chirality::Stereo stereo;

  // `ensureChemicalFlags`: seventeen booleans a column each, all derived from the residue name,
  // the atom name and the chain type.
  vector<uint8_t> isProtein, isRna, isDna, isNucleicAcid;
  vector<uint8_t> isProteinBackbone, isRnaBackbone, isDnaBackbone, isNucleicAcidBackbone;
  vector<uint8_t> isStandardAa, isStandardOrUnknownAa, isStandardUnknownOrMaskedAa;
  vector<uint8_t> isStandardRna, isStandardOrUnknownRna, isStandardUnknownOrMaskedRna;
  vector<uint8_t> isStandardDna, isStandardOrUnknownDna, isStandardUnknownOrMaskedDna;

  // Added by the later stages.
  vector<uint8_t> atomize, isCovalentModification, isUnindexedSource;
  vector<int> atomId, tokenId, withinChainSourceResIdx, insertionSourceIdx;

  size_t n() const { return site.n; }
};

/**
 * Every per-atom column, handed to `visit` one at a time.
 *
 * 🔴 ONE LIST, BECAUSE SLICING AND CONCATENATING MUST NOT FORGET A COLUMN. The reference gets
 * this for free - it walks the object's own keys and takes every array whose length is the atom
 * count - and a C++ struct cannot. A column added to `AtomArray` and not added here is a column
 * that silently keeps its OLD values through an insertion, which is a wrong answer in a field
 * nothing else checks. `site.selected` is deliberately absent: it indexes the FILE's rows, not
 * the atoms, and must not be resliced with them.
 */
template <class F>
inline void forEachColumn(AtomArray& a, F visit) {
  for (vector<string>* v : {&a.site.element, &a.site.atomName, &a.site.resName, &a.site.chainId,
                            &a.site.insCode, &a.site.labelAltId, &a.site.chemCompType,
                            &a.ids.pnUnitId, &a.ids.chainIid, &a.ids.pnUnitIid}) {
    visit(*v);
  }
  for (vector<int>* v : {&a.site.labelEntityId, &a.site.resId, &a.site.authSeqId, &a.site.atomId,
                         &a.site.charge, &a.site.atomicNumber, &a.site.chainType,
                         &a.ids.moleculeId, &a.ids.chainEntity, &a.ids.pnUnitEntity,
                         &a.ids.moleculeEntity, &a.ids.moleculeIid, &a.stereo.chiralType,
                         &a.atomId, &a.tokenId, &a.withinChainSourceResIdx, &a.insertionSourceIdx}) {
    visit(*v);
  }
  for (vector<double>* v : {&a.site.x, &a.site.y, &a.site.z, &a.site.bFactor, &a.site.occupancy}) {
    visit(*v);
  }
  for (vector<uint8_t>* v : {&a.site.hetero, &a.site.isPolymer, &a.site.isAromatic,
                             &a.site.isLeavingAtom, &a.stereo.stereogenicBondAtoms,
                             &a.isProtein, &a.isRna, &a.isDna, &a.isNucleicAcid,
                             &a.isProteinBackbone, &a.isRnaBackbone, &a.isDnaBackbone,
                             &a.isNucleicAcidBackbone, &a.isStandardAa, &a.isStandardOrUnknownAa,
                             &a.isStandardUnknownOrMaskedAa, &a.isStandardRna,
                             &a.isStandardOrUnknownRna, &a.isStandardUnknownOrMaskedRna,
                             &a.isStandardDna, &a.isStandardOrUnknownDna,
                             &a.isStandardUnknownOrMaskedDna, &a.atomize,
                             &a.isCovalentModification, &a.isUnindexedSource}) {
    visit(*v);
  }
}

/**
 * `sliceAtoms`: the given atoms, in the given order, with the bonds that survive RE-INDEXED.
 *
 * 🔴 A BOND SURVIVES ONLY IF BOTH ENDS DO, and comes back with `min, max` rather than its
 * original direction - which is the reference's, and which the bond list's consumers rely on.
 */
inline AtomArray slice(const AtomArray& a, const vector<int>& indices) {
  AtomArray out = a;
  out.site.selected.clear();
  const size_t nOrig = a.n();
  auto take = [&](auto& dst) {
    using T = typename std::decay_t<decltype(dst)>::value_type;
    if (dst.size() != nOrig) { dst.clear(); return; }
    vector<T> picked;
    picked.reserve(indices.size());
    for (int idx : indices) picked.push_back(dst[idx]);
    dst = std::move(picked);
  };
  forEachColumn(out, take);
  out.site.n = indices.size();
  for (auto& [name, v] : out.ann.ints) take(v);
  for (auto& [name, v] : out.ann.strings) take(v);
  for (auto& [name, v] : out.ann.floats) {
    // A three-wide annotation is `n * 3` long, so it is resliced a ROW at a time.
    if (v.size() == nOrig) { take(v); continue; }
    if (v.size() != nOrig * 3) { v.clear(); continue; }
    vector<double> picked;
    picked.reserve(indices.size() * 3);
    for (int idx : indices) {
      for (int k = 0; k < 3; ++k) picked.push_back(v[(size_t)idx * 3 + k]);
    }
    v = std::move(picked);
  }
  std::map<int, int> oldToNew;
  for (size_t i = 0; i < indices.size(); ++i) oldToNew[indices[i]] = (int)i;
  out.bonds.clear();
  for (const cif::Bond& b : a.bonds) {
    auto u = oldToNew.find(b.u), v = oldToNew.find(b.v);
    if (u == oldToNew.end() || v == oldToNew.end()) continue;
    out.bonds.push_back({std::min(u->second, v->second), std::max(u->second, v->second), b.type});
  }
  return out;
}

inline AtomArray slice(const AtomArray& a, int from, int to) {
  vector<int> indices;
  for (int i = from; i < to; ++i) indices.push_back(i);
  return slice(a, indices);
}

/**
 * `concatAtoms`: the arrays end to end, with each one's bonds shifted by the atoms before it.
 *
 * 🔴 AN ANNOTATION MISSING FROM ONE PART TAKES ITS DEFAULT THERE, not zero and not the previous
 * part's values - which is what makes an inserted stub carry `<M>` for its sequence condition and
 * -1 for its expandable-segment bounds rather than inheriting whatever it was spliced next to.
 */
/**
 * The same columns of two arrays, in lockstep - which is how `concat` appends without a
 * per-column name anywhere. A single list (`forEachColumn`) would need the caller to know WHICH
 * column it was looking at; this just hands over both.
 */
template <class F>
inline void forEachColumnPair(AtomArray& dst, const AtomArray& src, F visit) {
  AtomArray& s2 = const_cast<AtomArray&>(src);
  vector<vector<string>*> ds, ss;
  vector<vector<int>*> di, si;
  vector<vector<double>*> dd, sd;
  vector<vector<uint8_t>*> db, sb;
  auto collect = [](AtomArray& a, vector<vector<string>*>& cs, vector<vector<int>*>& ci,
                    vector<vector<double>*>& cd, vector<vector<uint8_t>*>& cb) {
    forEachColumn(a, [&](auto& column) {
      using V = std::decay_t<decltype(column)>;
      if constexpr (std::is_same_v<V, vector<string>>) cs.push_back(&column);
      else if constexpr (std::is_same_v<V, vector<int>>) ci.push_back(&column);
      else if constexpr (std::is_same_v<V, vector<double>>) cd.push_back(&column);
      else cb.push_back(&column);
    });
  };
  collect(dst, ds, di, dd, db);
  collect(s2, ss, si, sd, sb);
  for (size_t i = 0; i < ds.size(); ++i) visit(*ds[i], *ss[i]);
  for (size_t i = 0; i < di.size(); ++i) visit(*di[i], *si[i]);
  for (size_t i = 0; i < dd.size(); ++i) visit(*dd[i], *sd[i]);
  for (size_t i = 0; i < db.size(); ++i) visit(*db[i], *sb[i]);
}

/**
 * `concatAtoms`: the arrays end to end, with each one's bonds shifted by the atoms before it.
 *
 * 🔴 AN ANNOTATION MISSING FROM ONE PART TAKES ITS DEFAULT THERE, not zero and not the previous
 * part's values - which is what makes an inserted stub carry `<M>` for its sequence condition and
 * -1 for its expandable-segment bounds rather than inheriting whatever it was spliced next to.
 */
inline AtomArray concat(const vector<AtomArray>& parts) {
  if (parts.empty()) return {};
  AtomArray out = parts[0];
  out.site.selected.clear();
  auto fallback = [](const string& name) -> std::pair<double, string> {
    if (name == "_expseg_uid") return {-1, ""};
    auto it = tables::ANNOTATIONS_1BODY.find(name);
    if (it != tables::ANNOTATIONS_1BODY.end()) return {it->second.fallback, it->second.text};
    if (name.rfind("condition_sequence-probability-", 0) == 0) return {NAN, ""};
    return {0, ""};
  };
  // Every key any part has, so a column one part lacks is filled with its default there.
  std::set<string> intKeys, strKeys, floatKeys;
  std::map<string, int> floatCols;
  for (const AtomArray& p : parts) {
    for (const auto& [k, v] : p.ann.ints) intKeys.insert(k);
    for (const auto& [k, v] : p.ann.strings) strKeys.insert(k);
    for (const auto& [k, v] : p.ann.floats) {
      floatKeys.insert(k);
      if (p.n() && v.size() == p.n() * 3) floatCols[k] = 3;
      else if (!floatCols.count(k)) floatCols[k] = 1;
    }
  }
  cif::Annotations joined;
  for (const string& k : intKeys) joined.ints[k];
  for (const string& k : strKeys) joined.strings[k];
  for (const string& k : floatKeys) joined.floats[k];
  for (const AtomArray& p : parts) {
    for (const string& k : intKeys) {
      auto it = p.ann.ints.find(k);
      for (size_t i = 0; i < p.n(); ++i) {
        joined.ints[k].push_back(it == p.ann.ints.end() ? (int)fallback(k).first : it->second[i]);
      }
    }
    for (const string& k : strKeys) {
      auto it = p.ann.strings.find(k);
      for (size_t i = 0; i < p.n(); ++i) {
        joined.strings[k].push_back(it == p.ann.strings.end() ? fallback(k).second : it->second[i]);
      }
    }
    for (const string& k : floatKeys) {
      auto it = p.ann.floats.find(k);
      const int cols = floatCols[k];
      for (size_t i = 0; i < p.n(); ++i) {
        for (int c = 0; c < cols; ++c) {
          joined.floats[k].push_back(it == p.ann.floats.end() ? fallback(k).first
                                                              : it->second[i * cols + c]);
        }
      }
    }
  }

  for (size_t part = 1; part < parts.size(); ++part) {
    forEachColumnPair(out, parts[part], [](auto& dst, const auto& src) {
      dst.insert(dst.end(), src.begin(), src.end());
    });
  }
  size_t total = 0;
  for (const AtomArray& p : parts) total += p.n();
  out.site.n = total;
  out.ann = std::move(joined);

  out.bonds.clear();
  int offset = 0;
  for (const AtomArray& p : parts) {
    for (const cif::Bond& b : p.bonds) out.bonds.push_back({b.u + offset, b.v + offset, b.type});
    offset += (int)p.n();
  }
  return out;
}

/** Everything `parseCifToAtomArray` produces, in one object. */
inline AtomArray fromCif(const string& text) {
  AtomArray a;
  const cif::Block block = cif::parseBlock(text);
  a.site = cif::readAtomSite(block);
  cif::deriveChemistry(a.site, block);
  a.ann = cif::readAnnotations(a.site, block, a.site.selected);
  a.bonds = cif::buildBonds(a.site, block);
  a.ids = cif::computeIds(a.site, a.bonds);
  a.stereo = chirality::annotate(a.site, a.bonds, a.ann, {});
  return a;
}

/**
 * `ensureChemicalFlags`. Called three times by the pipeline, because the columns it derives go
 * stale whenever residues are inserted or dropped.
 *
 * 🔴 A MASKED RESIDUE COUNTS AS ITS CHAIN'S KIND. `<M>` is `is_standard_unknown_or_masked_aa` on
 * a polypeptide chain and the RNA or DNA equivalent on a nucleic one - which is how a contig's
 * stubs are protein at all, and why the chain type has to be right before this runs.
 */
inline void ensureChemicalFlags(AtomArray& a) {
  const size_t n = a.n();
  auto fill = [n](vector<uint8_t>& v) { v.assign(n, 0); };
  for (vector<uint8_t>* v : {&a.isProtein, &a.isRna, &a.isDna, &a.isNucleicAcid,
                             &a.isProteinBackbone, &a.isRnaBackbone, &a.isDnaBackbone,
                             &a.isNucleicAcidBackbone, &a.isStandardAa, &a.isStandardOrUnknownAa,
                             &a.isStandardUnknownOrMaskedAa, &a.isStandardRna,
                             &a.isStandardOrUnknownRna, &a.isStandardUnknownOrMaskedRna,
                             &a.isStandardDna, &a.isStandardOrUnknownDna,
                             &a.isStandardUnknownOrMaskedDna}) {
    fill(*v);
  }
  for (size_t i = 0; i < n; ++i) {
    const string& res = a.site.resName[i];
    const string& atom = a.site.atomName[i];
    const int ct = a.site.chainType[i];
    const bool prot = ct == tables::CT_POLYPEPTIDE_L || ct == tables::CT_POLYPEPTIDE_D;
    const bool rna = ct == tables::CT_RNA;
    const bool dna = ct == tables::CT_DNA;
    const bool na = rna || dna || ct == tables::CT_DNA_RNA_HYBRID;
    a.isProtein[i] = prot;
    a.isRna[i] = rna;
    a.isDna[i] = dna;
    a.isNucleicAcid[i] = na;
    a.isProteinBackbone[i] = prot && tables::PROTEIN_BACKBONE_ATOMS.count(atom) > 0;
    a.isRnaBackbone[i] = rna && tables::RNA_BACKBONE_ATOMS.count(atom) > 0;
    a.isDnaBackbone[i] = dna && tables::DNA_BACKBONE_ATOMS.count(atom) > 0;
    a.isNucleicAcidBackbone[i] = na && tables::NA_BACKBONE_ATOMS.count(atom) > 0;
    const bool stdAa = tables::STANDARD_AA.count(res) > 0;
    const bool stdUnkAa = stdAa || res == "UNK";
    a.isStandardAa[i] = stdAa;
    a.isStandardOrUnknownAa[i] = stdUnkAa;
    a.isStandardUnknownOrMaskedAa[i] = stdUnkAa || (res == "<M>" && prot);
    const bool stdRna = tables::STANDARD_RNA.count(res) > 0;
    const bool stdUnkRna = stdRna || res == "N";
    a.isStandardRna[i] = stdRna;
    a.isStandardOrUnknownRna[i] = stdUnkRna;
    a.isStandardUnknownOrMaskedRna[i] = stdUnkRna || (res == "<M>" && rna);
    const bool stdDna = tables::STANDARD_DNA.count(res) > 0;
    const bool stdUnkDna = stdDna || res == "DN";
    a.isStandardDna[i] = stdDna;
    a.isStandardOrUnknownDna[i] = stdUnkDna;
    a.isStandardUnknownOrMaskedDna[i] = stdUnkDna || (res == "<M>" && dna);
  }
}

/**
 * `addSrcComponentAnnotation`: the name of the component each atom CAME FROM - an expandable
 * segment's length range where it has one, otherwise the chain and residue it was read at.
 */
inline void addSrcComponentAnnotation(AtomArray& a) {
  const vector<int> starts = cif::residueSegments(a.site);
  auto segMin = a.ann.ints.find("annotation_expsegmin_1_residue");
  auto segMax = a.ann.ints.find("annotation_expsegmax_1_residue");
  vector<string> src(a.n());
  for (size_t r = 0; r + 1 < starts.size(); ++r) {
    const int s = starts[r], e = starts[r + 1];
    const bool expandable = segMin != a.ann.ints.end() && segMin->second[s] != -1;
    const string label =
        expandable ? std::to_string(segMin->second[s]) + "-" +
                         std::to_string(segMax == a.ann.ints.end() ? -1 : segMax->second[s])
                   : a.site.chainId[s] + std::to_string(a.site.resId[s]);
    for (int i = s; i < e; ++i) src[i] = label;
  }
  a.ann.strings["annotation_src-component_1_atom"] = src;
}

/** `computeWithinChainSourceResIdx`: each residue's ordinal within its own chain. */
inline void computeWithinChainSourceResIdx(AtomArray& a) {
  const vector<int> starts = cif::residueSegments(a.site);
  a.withinChainSourceResIdx.assign(a.n(), 0);
  string chain;
  int within = -1;
  for (size_t r = 0; r + 1 < starts.size(); ++r) {
    const int s = starts[r], e = starts[r + 1];
    if (a.site.chainId[s] != chain) { chain = a.site.chainId[s]; within = 0; }
    else ++within;
    for (int i = s; i < e; ++i) a.withinChainSourceResIdx[i] = within;
  }
}

/**
 * `realizeExpandableSegments`: each sentinel becomes the run of masked residues its length range
 * asked for, and everything after it shifts.
 *
 * 🔴 RIGHT TO LEFT, so an earlier sentinel's index is still its index when its turn comes.
 * Expanding left to right would need every later index remapped after every expansion, and the
 * reference does not - it relies on the order.
 *
 * 🔴 AND THE SUFFIX'S RESIDUE NUMBERS SHIFT ONLY WITHIN THE SENTINEL'S OWN CHAIN. A segment that
 * grows by ten residues moves the ten after it in that chain and nothing in any other, which is
 * what `chain_iid` is compared for rather than `chain_id`: two chains can share a letter across
 * transformations.
 *
 * 🔴 AND THE LENGTH IS DRAWN EVEN WHEN THE RANGE IS A POINT - see the spec path, where skipping
 * that draw moved every later one and relabelled the chains.
 */
inline void realizeExpandableSegments(AtomArray& a, PipelineRng& prng,
                                      const vector<int>& given = {}) {
  if (a.n() == 0) return;
  auto segMin = a.ann.ints.find("annotation_expsegmin_1_residue");
  auto segMax = a.ann.ints.find("annotation_expsegmax_1_residue");
  if (segMin == a.ann.ints.end() || segMax == a.ann.ints.end()) return;
  vector<int> sentinels;
  for (size_t i = 0; i < a.n(); ++i) {
    if (segMin->second[i] != -1 && a.site.resName[i] != "<M>") sentinels.push_back((int)i);
  }
  if (sentinels.empty()) return;

  vector<int> lengths;
  for (size_t k = 0; k < sentinels.size(); ++k) {
    const int lo = segMin->second[sentinels[k]], hi = segMax->second[sentinels[k]];
    if (k < given.size()) {
      if (given[k] < lo || given[k] > hi) {
        fprintf(stderr, "segment length %d outside [%d, %d] for sentinel %zu\n", given[k], lo, hi,
                k);
        exit(1);
      }
      lengths.push_back(given[k]);
    } else {
      lengths.push_back(prng.nextInt(lo, hi + 1));
    }
  }
  a.ann.ints["_expseg_uid"].assign(a.n(), -1);

  for (int k = (int)sentinels.size() - 1; k >= 0; --k) {
    const int idx = sentinels[k], segLen = lengths[k];
    int prevNeighbour = -1, nextNeighbour = -1;
    for (const cif::Bond& b : a.bonds) {
      const int other = b.u == idx ? b.v : (b.v == idx ? b.u : -1);
      if (other < 0) continue;
      if (other < idx) prevNeighbour = other;
      else nextNeighbour = other;
    }
    const int baseResId = a.site.resId[idx];
    const string chainIid =
        a.ids.chainIid.empty() ? a.site.chainId[idx] : a.ids.chainIid[idx];

    vector<AtomArray> parts;
    if (idx > 0) parts.push_back(slice(a, 0, idx));
    AtomArray single = slice(a, idx, idx + 1);
    single.site.resName[0] = "<M>";
    single.site.atomName[0] = "UNK";
    single.site.element[0] = "X";
    single.site.atomicNumber[0] = 0;
    single.site.x[0] = single.site.y[0] = single.site.z[0] = NAN;
    single.site.occupancy[0] = 0.0;
    single.ann.ints["_expseg_uid"][0] = k;
    if (segLen > 0) {
      vector<AtomArray> stubs;
      for (int sIdx = 0; sIdx < segLen; ++sIdx) {
        AtomArray st = single;
        st.site.resId[0] = baseResId + sIdx;
        st.site.authSeqId[0] = baseResId + sIdx;
        stubs.push_back(std::move(st));
      }
      AtomArray run = concat(stubs);
      run.bonds.clear();
      for (int sIdx = 0; sIdx + 1 < segLen; ++sIdx) {
        run.bonds.push_back({sIdx, sIdx + 1, tables::BT_SINGLE});
      }
      parts.push_back(std::move(run));
    }
    if (idx + 1 < (int)a.n()) {
      AtomArray suffix = slice(a, idx + 1, (int)a.n());
      if (segLen != 1) {
        for (size_t i = 0; i < suffix.n(); ++i) {
          const string cid =
              suffix.ids.chainIid.empty() ? suffix.site.chainId[i] : suffix.ids.chainIid[i];
          if (cid != chainIid) continue;
          suffix.site.resId[i] += segLen - 1;
          suffix.site.authSeqId[i] += segLen - 1;
        }
      }
      parts.push_back(std::move(suffix));
    }
    a = parts.empty() ? slice(a, 0, 0) : concat(parts);

    if (segLen > 0) {
      if (prevNeighbour != -1) a.bonds.push_back({prevNeighbour, idx, tables::BT_SINGLE});
      if (nextNeighbour != -1) {
        a.bonds.push_back({idx + segLen - 1, nextNeighbour + segLen - 1, tables::BT_SINGLE});
      }
    } else if (prevNeighbour != -1 && nextNeighbour != -1) {
      a.bonds.push_back({prevNeighbour, nextNeighbour - 1, tables::BT_SINGLE});
    }
  }
  std::stable_sort(a.bonds.begin(), a.bonds.end(), [](const cif::Bond& x, const cif::Bond& y) {
    return x.u != y.u ? x.u < y.u : x.v < y.v;
  });
  computeWithinChainSourceResIdx(a);
  ensureChemicalFlags(a);
}

/**
 * `filterUnwantedAtoms`: hydrogens, a protein's LEAVING `OXT` and a nucleic acid's `OP3` go.
 *
 * 🔴 THE `OXT` RULE READS `is_leaving_atom` AND THE `OP3` ONE DOES NOT. A C-terminal OXT is a
 * real atom and stays; only the one the CCD marks as leaving - the one a peptide bond would have
 * taken - is dropped. `OP3` is dropped from every nucleic residue whatever the flag says, which
 * is the reference's asymmetry and not an oversight of it.
 */
inline void filterUnwantedAtoms(AtomArray& a) {
  ensureChemicalFlags(a);
  vector<int> keep;
  keep.reserve(a.n());
  for (size_t i = 0; i < a.n(); ++i) {
    string elem = a.site.element[i];
    for (char& c : elem) c = (char)std::toupper((unsigned char)c);
    if (elem == "H" || elem == "D") continue;
    const string& atomName = a.site.atomName[i];
    if (a.isProtein[i] && atomName == "OXT" && a.site.isLeavingAtom[i]) continue;
    if (a.isNucleicAcid[i] && atomName == "OP3") continue;
    keep.push_back((int)i);
  }
  if (keep.size() != a.n()) a = slice(a, keep);
  ensureChemicalFlags(a);
}

/** The residues that carry a local latent, as a set - `LOCAL_LATENT_RESIDUES` is kept in the
 *  order its feature column wants, and this stage asks only whether a name is in it. */
inline const std::unordered_set<string>& localLatentResidues() {
  static const std::unordered_set<string> s(tables::LOCAL_LATENT_RESIDUES.begin(),
                                            tables::LOCAL_LATENT_RESIDUES.end());
  return s;
}

/**
 * `annotateCovalentModificationsAndAtomize`: a polymer residue reached by a bond that is not its
 * own backbone link is a covalent modification, and anything with no local latent is atomised.
 *
 * 🔴 THE TEST IS "NOT A STANDARD BACKBONE BOND", NOT "IS A STRANGE BOND". A peptide C-N or a
 * phosphodiester O3'-P between CONSECUTIVE residues of ONE chain is the polymer doing its job;
 * every other inter-residue bond marks both polymer ends it touches. So a disulfide marks two
 * cysteines, a glycan marks the asparagine, and a chain break - `res_id` two apart - marks a pair
 * that looks like an ordinary peptide bond and is not one.
 *
 * 🔴 AND `atomize` IS STICKY. A residue already flagged by an earlier stage stays flagged however
 * ordinary its name is; the set membership can only ADD to it.
 */
inline void annotateCovalentModificationsAndAtomize(AtomArray& a) {
  const size_t n = a.n();
  const vector<int> starts = cif::residueSegments(a.site);
  const size_t nRes = starts.empty() ? 0 : starts.size() - 1;
  vector<int> atomToRes(n, 0);
  for (size_t r = 0; r < nRes; ++r) {
    for (int i = starts[r]; i < starts[r + 1]; ++i) atomToRes[i] = (int)r;
  }

  vector<uint8_t> resIsCovMod(nRes, 0);
  for (const cif::Bond& b : a.bonds) {
    const int r1 = atomToRes[b.u], r2 = atomToRes[b.v];
    if (r1 == r2) continue;
    const string& an1 = a.site.atomName[b.u];
    const string& an2 = a.site.atomName[b.v];
    const bool sameChain = a.ids.chainIid[b.u] == a.ids.chainIid[b.v];
    const bool adjacent = std::abs(a.site.resId[b.u] - a.site.resId[b.v]) == 1;
    const bool peptide = sameChain && adjacent && a.isProtein[b.u] && a.isProtein[b.v] &&
                         ((an1 == "C" && an2 == "N") || (an1 == "N" && an2 == "C"));
    const bool phosphodiester =
        sameChain && adjacent && a.isNucleicAcid[b.u] && a.isNucleicAcid[b.v] &&
        ((an1 == "O3'" && an2 == "P") || (an1 == "P" && an2 == "O3'"));
    if (peptide || phosphodiester) continue;
    if (a.site.isPolymer[b.u]) resIsCovMod[r1] = 1;
    if (a.site.isPolymer[b.v]) resIsCovMod[r2] = 1;
  }

  a.isCovalentModification.assign(n, 0);
  a.atomize.resize(n, 0);
  for (size_t r = 0; r < nRes; ++r) {
    const int s = starts[r], e = starts[r + 1];
    const bool atomise = !localLatentResidues().count(a.site.resName[s]) || a.atomize[s];
    for (int i = s; i < e; ++i) {
      a.isCovalentModification[i] = resIsCovMod[r];
      a.atomize[i] = atomise ? 1 : 0;
    }
  }
}

/** `addGlobalTokenIdAnnotation`: one token a residue, or one token an ATOM where it is atomised. */
inline void addGlobalTokenIdAnnotation(AtomArray& a) {
  const vector<int> starts = cif::residueSegments(a.site);
  a.tokenId.assign(a.n(), 0);
  int next = 0;
  for (size_t r = 0; r + 1 < starts.size(); ++r) {
    const int s = starts[r], e = starts[r + 1];
    if (a.atomize[s]) {
      for (int i = s; i < e; ++i) a.tokenId[i] = next++;
    } else {
      for (int i = s; i < e; ++i) a.tokenId[i] = next;
      ++next;
    }
  }
}

/**
 * `getTokenStarts(atoms, true)`: where each token begins, plus `n`.
 *
 * 🔴 IT READS `token_id` WHERE THERE IS ONE AND FALLS BACK TO THE RESIDUES WHERE THERE IS NOT,
 * which is the reference's own two-branch shape and not a convenience: the stages before
 * `addGlobalTokenIdAnnotation` call this too, and on an atomised residue the two branches give
 * different answers.
 */
inline vector<int> tokenStarts(const AtomArray& a) {
  const size_t n = a.n();
  if (n == 0) return {0};
  if (a.tokenId.size() == n) {
    vector<int> starts{0};
    for (size_t i = 1; i < n; ++i) {
      if (a.tokenId[i] != a.tokenId[i - 1]) starts.push_back((int)i);
    }
    starts.push_back((int)n);
    return starts;
  }
  const vector<int> res = cif::residueSegments(a.site);
  vector<int> starts;
  for (size_t r = 0; r + 1 < res.size(); ++r) {
    const int s = res[r], e = res[r + 1];
    if (a.atomize.size() == n && a.atomize[s]) {
      for (int i = s; i < e; ++i) starts.push_back(i);
    } else {
      starts.push_back(s);
    }
  }
  starts.push_back((int)n);
  return starts;
}

/** `addGlobalAtomIdAnnotation`: 0..n-1, re-run whenever atoms are inserted or dropped. */
inline void addGlobalAtomIdAnnotation(AtomArray& a) {
  a.atomId.resize(a.n());
  for (size_t i = 0; i < a.n(); ++i) a.atomId[i] = (int)i;
}

/** The residues whose sequence identity may be CONDITIONED - the amino acids and the RNA bases,
 *  the two token types whose name the conditioning vocabulary can express. */
inline const std::unordered_set<string>& conditionableResidues() {
  static const std::unordered_set<string> s = [] {
    std::unordered_set<string> out;
    for (const char* key : {"aa", "rna"}) {
      auto it = tables::RESIDUES_BY_TOKEN_TYPE.find(key);
      if (it != tables::RESIDUES_BY_TOKEN_TYPE.end()) out.insert(it->second.begin(),
                                                                 it->second.end());
    }
    return out;
  }();
  return s;
}

/**
 * `finalizeAndCanonicalizeSequenceConditioning`: the sequence condition is settled, then the
 * per-residue probability columns are rewritten to agree with it.
 *
 * 🔴 A RESIDUE THE VOCABULARY CANNOT NAME IS CONDITIONED BY BEING ITSELF. `conditioned` is
 * `existing || !allowed`, so a ligand or a modified residue counts as conditioned without any
 * annotation asking for it - it has no `<M>` to hide behind, and the model is told what it is.
 * That `|| !allowed` is PROVABLY INERT and is kept because the reference has it: `conditioned`
 * is read only to decide whether the spread overwrites `sequence` with the residue's own name,
 * and for a residue the vocabulary cannot name the first loop has already written exactly that.
 * It cannot be falsified by any input, so it is recorded here rather than left looking untested.
 *
 * 🔴 AND THE REQUIREMENT SPREADS OVER A TOKEN, NOT A RESIDUE. One atom with conditioned
 * coordinates off the backbone makes its WHOLE token required, which for an atomised residue is
 * that one atom and for an ordinary one is all of it - so the same geometry means different
 * things either side of `atomize`, and reading residues here would be wrong in exactly the
 * atomised case the stage before this one creates.
 *
 * 🔴 AND A PROBABILITY COLUMN THAT ENDS UP ENTIRELY `NaN` IS DELETED RATHER THAN KEPT. An empty
 * column and an absent one are different inputs downstream; writing 28 columns of NaN where the
 * reference writes none is a difference no per-atom comparison of the columns that DO exist can
 * see.
 */
inline void finalizeAndCanonicalizeSequenceConditioning(AtomArray& a) {
  const size_t n = a.n();
  vector<string> seqIn(n, "<M>");
  auto cSeqIt = a.ann.strings.find("condition_sequence_1_residue");
  if (cSeqIt != a.ann.strings.end() && cSeqIt->second.size() == n) seqIn = cSeqIt->second;
  auto crdIt = a.ann.floats.find("condition_coordinate_1_atom");
  const vector<double>* crd =
      crdIt != a.ann.floats.end() && crdIt->second.size() == n * 3 ? &crdIt->second : nullptr;
  auto grpIt = a.ann.ints.find("condition_CoordinateGroup_1_atom");
  const vector<int>* grp =
      grpIt != a.ann.ints.end() && grpIt->second.size() == n ? &grpIt->second : nullptr;

  vector<string> sequence(n);
  vector<uint8_t> conditioned(n, 0), required(n, 0);
  for (size_t i = 0; i < n; ++i) {
    const bool allowed = conditionableResidues().count(a.site.resName[i]) > 0;
    const bool existing = seqIn[i] != "<M>";
    sequence[i] = allowed ? seqIn[i] : a.site.resName[i];
    conditioned[i] = (existing || !allowed) ? 1 : 0;
    bool req = conditioned[i] || (a.atomize.size() == n && a.atomize[i]);
    const bool hasCrd =
        (crd && std::isfinite((*crd)[i * 3]) && std::isfinite((*crd)[i * 3 + 1]) &&
         std::isfinite((*crd)[i * 3 + 2])) ||
        (grp && (*grp)[i] > 0);
    if (hasCrd) {
      const bool backbone = a.isProteinBackbone[i] || a.isRnaBackbone[i] || a.isDnaBackbone[i];
      if (!backbone) req = true;
    }
    required[i] = req ? 1 : 0;
  }

  const vector<int> tokens = tokenStarts(a);
  for (size_t t = 0; t + 1 < tokens.size(); ++t) {
    const int s = tokens[t], e = tokens[t + 1];
    bool any = false;
    for (int i = s; i < e && !any; ++i) any = required[i] != 0;
    if (!any) continue;
    for (int i = s; i < e; ++i) {
      if (!conditioned[i]) sequence[i] = a.site.resName[i];
    }
  }
  a.ann.strings["condition_sequence_1_residue"] = sequence;

  const vector<int> res = cif::residueSegments(a.site);
  vector<uint8_t> seqAtoms(n, 0);
  for (size_t r = 0; r + 1 < res.size(); ++r) {
    const int s = res[r], e = res[r + 1];
    bool any = false;
    for (int i = s; i < e && !any; ++i) any = sequence[i] != "<M>";
    if (any) for (int i = s; i < e; ++i) seqAtoms[i] = 1;
  }

  for (const char* tok : tables::STANDARD_POLYMER_RESIDUES) {
    const string key = string("condition_sequence-probability-") + tok + "_1_residue";
    auto it = a.ann.floats.find(key);
    vector<double> prob = it != a.ann.floats.end() && it->second.size() == n
                              ? it->second
                              : vector<double>(n, NAN);
    bool anyMask = false;
    for (size_t i = 0; i < n; ++i) {
      if (seqAtoms[i]) {
        prob[i] = sequence[i] == tok ? 1.0 : 0.0;
        anyMask = true;
      } else if (!std::isfinite(prob[i])) {
        prob[i] = NAN;
      } else {
        anyMask = true;
      }
    }
    if (anyMask) a.ann.floats[key] = prob;
    else a.ann.floats.erase(key);
  }
}

/**
 * `featurizeExpandedSegments`: at inference an expanded segment nobody indexed is DELETED, and
 * whatever expanded segment is left has its coordinates zeroed.
 *
 * 🔴 THE DELETION IS INFERENCE-ONLY AND THE ZEROING IS NOT. Training keeps the unindexed
 * segments and still zeroes them, so the two halves are not one switch - reading the stage as
 * "drop the expanded segments" would zero nothing on the training path and delete too much on
 * neither.
 *
 * 🔴 AND AN ATOM IS UNINDEXED WHEN `condition_index_1_residue` IS FALSEY, WHICH INCLUDES ZERO.
 * The column is an int and the reference tests `!cIdx[i]`, so index 0 is unindexed - not a
 * missing value, an actual zero, which is what an unindexed motif carries.
 */
inline void featurizeExpandedSegments(AtomArray& a, bool isInference = true) {
  auto segIt = a.ann.ints.find("annotation_expsegmin_1_residue");
  if (segIt == a.ann.ints.end()) return;
  bool hasExp = false;
  for (size_t i = 0; i < a.n() && !hasExp; ++i) hasExp = segIt->second[i] != -1;
  if (!hasExp) return;

  auto idxIt = a.ann.ints.find("condition_index_1_residue");
  if (isInference && idxIt != a.ann.ints.end()) {
    const vector<int>& seg = segIt->second;
    const vector<int>& idx = idxIt->second;
    vector<int> keep;
    keep.reserve(a.n());
    for (size_t i = 0; i < a.n(); ++i) {
      if (!(seg[i] != -1 && idx[i] == 0)) keep.push_back((int)i);
    }
    if (keep.size() < a.n()) a = slice(a, keep);
  }
  const vector<int>& seg = a.ann.ints["annotation_expsegmin_1_residue"];
  for (size_t i = 0; i < a.n(); ++i) {
    if (seg[i] != -1) { a.site.x[i] = 0.0; a.site.y[i] = 0.0; a.site.z[i] = 0.0; }
  }
}

/**
 * `materializeInternalCoords`: every positive coordinate GROUP whose internal coordinates are
 * not all set gets its members' positions, centred on the group and turned by a random rotation.
 *
 * 🔴 THE DRAW HAPPENS PER QUALIFYING GROUP, AFTER THE `continue`, IN ASCENDING GROUP ORDER - and
 * NOT AT ALL where a rotation is supplied, because the reference writes
 * `(rotations && rotations[gid]) || prng.randomSO3()` and `||` short-circuits. Three ways to get
 * the stream wrong without getting the geometry wrong: drawing for every group rather than the
 * ones that need it, drawing before the `allFinite` test, and drawing even when a rotation was
 * given. This port has already been bitten twice by a draw that happens where it looks as
 * though it should not.
 *
 * 🔴 AND THE GROUP COLUMN IS CREATED HERE WHERE IT IS ABSENT, as 0 for an atom with finite
 * conditioned coordinates and -1 for one without - so a structure that never mentions groups
 * still gets the column, and every group in it is 0, which is not positive and draws nothing.
 */
inline void materializeInternalCoords(AtomArray& a, PipelineRng& prng,
                                      const std::map<int, std::array<std::array<double, 3>, 3>>&
                                          rotations = {}) {
  const size_t n = a.n();
  if (!a.ann.ints.count("condition_CoordinateGroup_1_atom")) {
    auto crdIt = a.ann.floats.find("condition_coordinate_1_atom");
    const vector<double>* crd =
        crdIt != a.ann.floats.end() && crdIt->second.size() == n * 3 ? &crdIt->second : nullptr;
    vector<int>& g = a.ann.ints["condition_CoordinateGroup_1_atom"];
    g.assign(n, -1);
    for (size_t i = 0; i < n; ++i) {
      const bool known = crd && std::isfinite((*crd)[i * 3]) &&
                         std::isfinite((*crd)[i * 3 + 1]) && std::isfinite((*crd)[i * 3 + 2]);
      g[i] = known ? 0 : -1;
    }
  }
  if (!a.ann.floats.count("condition_InternalCoordinate_1_atom")) {
    a.ann.floats["condition_InternalCoordinate_1_atom"].assign(n * 3, NAN);
  }
  const vector<int>& groups = a.ann.ints["condition_CoordinateGroup_1_atom"];
  vector<double>& payload = a.ann.floats["condition_InternalCoordinate_1_atom"];

  std::set<int> positive;
  for (size_t i = 0; i < n; ++i) if (groups[i] > 0) positive.insert(groups[i]);
  if (positive.empty()) return;

  for (const int gid : positive) {
    vector<int> members;
    bool allFinite = true;
    for (size_t i = 0; i < n; ++i) {
      if (groups[i] != gid) continue;
      members.push_back((int)i);
      if (!std::isfinite(payload[i * 3]) || !std::isfinite(payload[i * 3 + 1]) ||
          !std::isfinite(payload[i * 3 + 2])) {
        allFinite = false;
      }
    }
    if (allFinite || members.empty()) continue;

    double mx = 0, my = 0, mz = 0;
    for (const int i : members) { mx += a.site.x[i]; my += a.site.y[i]; mz += a.site.z[i]; }
    mx /= (double)members.size();
    my /= (double)members.size();
    mz /= (double)members.size();

    double R[3][3];
    auto given = rotations.find(gid);
    if (given != rotations.end()) {
      for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) R[r][c] = given->second[r][c];
    } else {
      prng.randomSO3(R);
    }
    for (const int i : members) {
      const double cx = a.site.x[i] - mx, cy = a.site.y[i] - my, cz = a.site.z[i] - mz;
      payload[i * 3] = cx * R[0][0] + cy * R[0][1] + cz * R[0][2];
      payload[i * 3 + 1] = cx * R[1][0] + cy * R[1][1] + cz * R[1][2];
      payload[i * 3 + 2] = cx * R[2][0] + cy * R[2][1] + cz * R[2][2];
    }
  }
}

/**
 * `getDefaultIndexGroups`: one group a MOLECULE for the indexed atoms, then one group per
 * contiguous RUN for the unindexed ones, numbered after every molecule.
 */
inline vector<int> defaultIndexGroups(const AtomArray& a) {
  const size_t n = a.n();
  if (n == 0) return {};
  const vector<int>& mol = a.ids.moleculeIid;
  auto idxIt = a.ann.ints.find("condition_index_1_residue");
  vector<int> isIndexed(n, 1);
  if (idxIt != a.ann.ints.end() && idxIt->second.size() == n) {
    for (size_t i = 0; i < n; ++i) isIndexed[i] = idxIt->second[i] ? 1 : 0;
  }
  std::set<int> uniqueMols(mol.begin(), mol.end());
  std::map<int, int> dense;
  for (const int m : uniqueMols) { const int k = (int)dense.size(); dense[m] = k; }
  const int nMolecules = (int)uniqueMols.size();

  vector<int> groups(n, 0);
  for (size_t i = 0; i < n; ++i) if (isIndexed[i]) groups[i] = dense[mol[i]];

  bool anyUn = false;
  for (size_t i = 0; i < n && !anyUn; ++i) anyUn = !isIndexed[i];
  if (!anyUn) return groups;

  vector<int> runIds(n, 0);
  int r = 0;
  for (size_t i = 1; i < n; ++i) {
    if (mol[i] != mol[i - 1] || isIndexed[i] != isIndexed[i - 1]) ++r;
    runIds[i] = r;
  }
  std::set<int> unindexedRuns;
  for (size_t i = 0; i < n; ++i) if (!isIndexed[i]) unindexedRuns.insert(runIds[i]);
  std::map<int, int> runDense;
  for (const int rid : unindexedRuns) { const int k = (int)runDense.size(); runDense[rid] = k; }
  for (size_t i = 0; i < n; ++i) {
    if (!isIndexed[i]) groups[i] = runDense[runIds[i]] + nMolecules;
  }
  return groups;
}

/**
 * `bondFlankingResidues`: join two residues that are now neighbours, if each offers exactly one
 * polymerization atom.
 *
 * 🔴 EXACTLY ONE. `findPolymerizationAtomIdx` counts the matches and returns nothing when there
 * are two - an altloc or a duplicated name makes the join ambiguous, and the reference declines
 * rather than taking the first. A `<M>` stub residue of one atom IS that atom, whatever it is
 * called.
 */
inline void bondFlankingResidues(AtomArray& a, int prevRes, int nextRes,
                                 const vector<int>& bounds) {
  const int sPrev = bounds[prevRes], ePrev = bounds[prevRes + 1];
  const int sNext = bounds[nextRes], eNext = bounds[nextRes + 1];
  if (a.atomize.size() == a.n() && (a.atomize[sPrev] || a.atomize[sNext])) return;
  // `token_type` is not set until the batch is built, so both ends take the reference's
  // PROTEIN default and with it the C/N fallback where the residue names say nothing.
  auto names = [&](int s) {
    std::pair<string, string> p = cif::polymerizationAtoms(a.site.resName[s], a.site.chainType[s]);
    if (p.first.empty() && p.second.empty()) p = {"C", "N"};
    return p;
  };
  auto find = [&](int s, int e, bool exiting) -> int {
    if (e - s == 1 && a.site.resName[s] == "<M>") return s;
    const std::pair<string, string> p = names(s);
    const string& target = exiting ? p.first : p.second;
    int match = -1, count = 0;
    for (int i = s; i < e; ++i) {
      if (a.site.atomName[i] == target) { match = i; ++count; }
    }
    return count == 1 ? match : -1;
  };
  const int left = find(sPrev, ePrev, true);
  const int right = find(sNext, eNext, false);
  if (left < 0 || right < 0) return;
  const int u = std::min(left, right), v = std::max(left, right);
  for (const cif::Bond& b : a.bonds) {
    if (std::min(b.u, b.v) == u && std::max(b.u, b.v) == v) return;
  }
  a.bonds.push_back(cif::Bond{u, v, tables::BT_SINGLE});
}

/**
 * `unindexFlaggedTokens`: an unindexed motif is COPIED to the end of the array under fresh
 * residue numbers, and at inference its original is removed.
 *
 * 🔴 A REMOVED RUN ONLY COLLAPSES THE NUMBERING AT A CHAIN'S EDGE OR BESIDE AN EXPANDABLE
 * SEGMENT. An interior run with ordinary residues on both sides leaves a HOLE in `res_id` on
 * purpose - the chain really is broken there and the numbering says so - so "renumber what is
 * left" is the wrong reading and produces a chain that claims to be continuous.
 *
 * 🔴 AND THE RESIDUES THAT BECOME NEIGHBOURS ARE RE-BONDED ONLY WHERE SOMETHING COLLAPSED
 * BETWEEN THEM. `collapsedKept[next] > collapsedKept[prev]` is the test, and it is why the
 * number of collapsed residues before each atom has to survive the slice. Measured: removing
 * the re-bond entirely turns the gate red, so the call is live - but forcing the test TRUE
 * changes nothing on any case built, because the extra pairs it admits are consecutive
 * residues that the polymer bonds already joined and `bondFlankingResidues` declines a bond it
 * finds present. The guard saves work rather than changing the answer.
 *
 * 🔴 AND THE COPY'S TOKEN IDS ARE TRANSIENT. `addGlobalTokenIdAnnotation` runs again at the end
 * and renumbers everything, so `nextCopyTokenId` only ever reaches the output through
 * `insertion_source_idx` - which is -1 at inference anyway.
 */
inline void unindexFlaggedTokens(AtomArray& a, bool isInference = true) {
  const size_t n = a.n();
  auto idxIt = a.ann.ints.find("condition_index_1_residue");
  vector<int> cIdx(n, 1);
  if (idxIt != a.ann.ints.end() && idxIt->second.size() == n) cIdx = idxIt->second;
  bool anyUnindexed = false;
  for (size_t i = 0; i < n && !anyUnindexed; ++i) anyUnindexed = !cIdx[i];

  if (!anyUnindexed) {
    a.insertionSourceIdx.assign(n, -1);
    a.isUnindexedSource.assign(n, 0);
    addGlobalTokenIdAnnotation(a);
    addGlobalAtomIdAnnotation(a);
    return;
  }

  if (!a.ann.ints.count("condition_index-group_1_token")) {
    a.ann.ints["condition_index-group_1_token"] = defaultIndexGroups(a);
  }
  const vector<int> indexGroups = a.ann.ints["condition_index-group_1_token"];

  std::map<string, int> nextFreeResId;
  for (size_t i = 0; i < n; ++i) {
    const string& cid = a.ids.chainIid[i];
    auto it = nextFreeResId.find(cid);
    if (it == nextFreeResId.end() || a.site.resId[i] + 1 > it->second) {
      nextFreeResId[cid] = a.site.resId[i] + 1;
    }
  }
  int nextCopyTokenId = 0;
  for (size_t i = 0; i < n; ++i) nextCopyTokenId = std::max(nextCopyTokenId, a.tokenId[i] + 1);

  std::set<int> unindexedGroupIds;
  for (size_t i = 0; i < n; ++i) if (!cIdx[i]) unindexedGroupIds.insert(indexGroups[i]);

  vector<AtomArray> slices;
  for (const int gid : unindexedGroupIds) {
    vector<int> members;
    for (size_t i = 0; i < n; ++i) if (!cIdx[i] && indexGroups[i] == gid) members.push_back((int)i);
    AtomArray motif = slice(a, members);

    vector<string> motifChains;
    for (size_t i = 0; i < motif.n(); ++i) {
      if (std::find(motifChains.begin(), motifChains.end(), motif.ids.chainIid[i]) ==
          motifChains.end()) {
        motifChains.push_back(motif.ids.chainIid[i]);
      }
    }
    for (const string& cid : motifChains) {
      int minRes = INT32_MAX, maxRes = INT32_MIN;
      for (size_t i = 0; i < motif.n(); ++i) {
        if (motif.ids.chainIid[i] != cid) continue;
        minRes = std::min(minRes, motif.site.resId[i]);
        maxRes = std::max(maxRes, motif.site.resId[i]);
      }
      const int offset = nextFreeResId[cid] - minRes;
      for (size_t i = 0; i < motif.n(); ++i) {
        if (motif.ids.chainIid[i] == cid) motif.site.resId[i] += offset;
      }
      nextFreeResId[cid] = maxRes + offset + 2;
    }

    const vector<int> rb = cif::residueSegments(motif.site);
    motif.insertionSourceIdx.assign(motif.n(), -1);
    vector<int> newTok(motif.n(), 0);
    for (size_t r = 0; r + 1 < rb.size(); ++r) {
      const int s = rb[r], e = rb[r + 1];
      const int srcTok = isInference ? -1 : motif.tokenId[s];
      for (int i = s; i < e; ++i) { motif.insertionSourceIdx[i] = srcTok; newTok[i] = nextCopyTokenId; }
      ++nextCopyTokenId;
    }
    motif.tokenId = newTok;
    slices.push_back(std::move(motif));
  }
  const AtomArray unindexed = concat(slices);

  const vector<int> rbOrig = cif::residueSegments(a.site);
  const size_t nResOrig = rbOrig.size() - 1;
  vector<uint8_t> isRemovedRes(nResOrig, 0), isExpandableRes(nResOrig, 0);
  vector<string> chainOfRes(nResOrig);
  auto segIt = a.ann.ints.find("annotation_expsegmin_1_residue");
  for (size_t r = 0; r < nResOrig; ++r) {
    const int s = rbOrig[r], e = rbOrig[r + 1];
    chainOfRes[r] = a.ids.chainIid[s];
    isExpandableRes[r] = segIt != a.ann.ints.end() && segIt->second[s] != -1;
    for (int i = s; i < e; ++i) if (!cIdx[i]) { isRemovedRes[r] = 1; break; }
  }

  vector<int> nCollapsedBefore(nResOrig, 0);
  vector<string> uniqueChains;
  for (const string& c : chainOfRes) {
    if (std::find(uniqueChains.begin(), uniqueChains.end(), c) == uniqueChains.end()) {
      uniqueChains.push_back(c);
    }
  }
  for (const string& cid : uniqueChains) {
    vector<int> inChain;
    for (size_t r = 0; r < nResOrig; ++r) if (chainOfRes[r] == cid) inChain.push_back((int)r);
    vector<uint8_t> collapse(inChain.size());
    for (size_t m = 0; m < inChain.size(); ++m) collapse[m] = isRemovedRes[inChain[m]];
    size_t k = 0;
    while (k < inChain.size()) {
      if (!isRemovedRes[inChain[k]]) { ++k; continue; }
      const size_t runStart = k;
      while (k < inChain.size() && isRemovedRes[inChain[k]]) ++k;
      const size_t runStop = k;
      const bool atBoundary = runStart == 0 || runStop == inChain.size();
      const bool adjExpandable = !atBoundary && (isExpandableRes[inChain[runStart - 1]] ||
                                                 isExpandableRes[inChain[runStop]]);
      if (!atBoundary && !adjExpandable) {
        for (size_t m = runStart; m < runStop; ++m) collapse[m] = 0;
      }
    }
    int cum = 0;
    for (size_t m = 0; m < inChain.size(); ++m) {
      nCollapsedBefore[inChain[m]] = cum;
      if (collapse[m]) ++cum;
    }
  }

  vector<int> collapsedBeforeAtom(n, 0);
  vector<uint8_t> tokenToUnindex(n, 0);
  for (size_t r = 0; r < nResOrig; ++r) {
    const int s = rbOrig[r], e = rbOrig[r + 1];
    for (int i = s; i < e; ++i) {
      collapsedBeforeAtom[i] = nCollapsedBefore[r];
      tokenToUnindex[i] = isRemovedRes[r];
      a.site.resId[i] -= nCollapsedBefore[r];
    }
  }

  vector<int> indexedKeep, collapsedKept;
  for (size_t i = 0; i < n; ++i) {
    if (!tokenToUnindex[i]) { indexedKeep.push_back((int)i); collapsedKept.push_back(collapsedBeforeAtom[i]); }
  }
  AtomArray indexedArr = slice(a, indexedKeep);
  const vector<int> rb = cif::residueSegments(indexedArr.site);
  for (size_t r = 0; r + 1 < rb.size() - 1; ++r) {
    const int sPrev = rb[r], sNext = rb[r + 1];
    if (indexedArr.ids.chainIid[sPrev] == indexedArr.ids.chainIid[sNext] &&
        indexedArr.site.resId[sNext] - indexedArr.site.resId[sPrev] == 1 &&
        collapsedKept[sNext] > collapsedKept[sPrev]) {
      bondFlankingResidues(indexedArr, (int)r, (int)r + 1, rb);
    }
  }
  computeWithinChainSourceResIdx(indexedArr);
  indexedArr.insertionSourceIdx.assign(indexedArr.n(), -1);

  a = concat({indexedArr, unindexed});
  vector<int>& fullGrp = a.ann.ints["condition_index-group_1_token"];
  const vector<int>& fullIdx = a.ann.ints["condition_index_1_residue"];
  int maxIdxGrp = INT32_MIN, minUnidxGrp = INT32_MAX;
  bool sawIdx = false;
  for (size_t i = 0; i < a.n(); ++i) {
    if (fullIdx[i]) { maxIdxGrp = std::max(maxIdxGrp, fullGrp[i]); sawIdx = true; }
    else minUnidxGrp = std::min(minUnidxGrp, fullGrp[i]);
  }
  const int shift = (sawIdx ? maxIdxGrp : -1) - minUnidxGrp + 1;
  for (size_t i = 0; i < a.n(); ++i) if (!fullIdx[i]) fullGrp[i] += shift;

  a.isUnindexedSource.assign(a.n(), 0);
  addGlobalTokenIdAnnotation(a);
  addGlobalAtomIdAnnotation(a);
  ensureChemicalFlags(a);
}

}  // namespace atoms
}  // namespace rfd4p
