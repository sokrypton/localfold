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
  vector<uint8_t> atomize, isCovalentModification;
  vector<int> atomId, tokenId, withinChainSourceResIdx;

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
                         &a.atomId, &a.tokenId, &a.withinChainSourceResIdx}) {
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
                             &a.isCovalentModification}) {
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

}  // namespace atoms
}  // namespace rfd4p
