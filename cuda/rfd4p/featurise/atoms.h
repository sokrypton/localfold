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

namespace rfd4p {
namespace atoms {

using std::string;
using std::vector;

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

/** `addGlobalAtomIdAnnotation`: 0..n-1, re-run whenever atoms are inserted or dropped. */
inline void addGlobalAtomIdAnnotation(AtomArray& a) {
  a.atomId.resize(a.n());
  for (size_t i = 0; i < a.n(); ++i) a.atomId[i] = (int)i;
}

}  // namespace atoms
}  // namespace rfd4p
