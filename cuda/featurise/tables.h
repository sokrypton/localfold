// The featuriser's data: reference conformers, element symbols, each family's conventions - generated from the
// JavaScript it mirrors (tools/gen-native-featuriser-tables.mjs -> tables.inc).
#pragma once
#include <stdexcept>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lf {

struct ConformerAtom { int slot; const char* name; int element; double charge, x, y, z; };
struct ConformerEntry { char code; int aatype; std::vector<ConformerAtom> internal, terminal; };

// featuriserDialect(dialect) plus the dialect fields the exporter reads itself
struct FamilyFlags {
  const char* name;
  bool symmetriseBonds, centreRefConformers, paddedAtomKeys, qblockAtomKeys;
  int dropTerminalAtoms;            // 0 keep, 1 drop (terminalAtoms: false), 2 the protein's only ("nucleic")
  int dedupeSelfMsa;                // -1 undefined, 0 false (duplicate the query row), 1 true
  bool atomizedElementNames; int atomizedUnknownRestype;   // 0 false, 1 true (an amino acid's), 2 "nucleic" (a base's too), 3 "chai" (a base's 30)
  bool atomizedUnknownMsa;
  int atomizedBackboneBonds;        // 0 none, 1 to every neighbour (rf3), 2 "both": between two atomised residues alone (openbind0)
  bool modifiedAsOneToken, chaiMsaFeatures;
  int fusedDistogramBins, fusedRestypes;   // fusedTemplateLayout, 0 when absent
  bool boltz2TemplateFeatures, rosettafold3TemplateFeatures, templateFeatureMeanOnePass;
  int emptyTemplateAatype;          // -1 when absent
  bool templateVisibilityByCoverage, chaiTemplates, templateStackOuterResidual, chaiTokenEmbedding;
  bool structuralTokens, chiralCentres;
  int emptyColumnsState;            // emptyTemplateRestypeColumns: 0 undefined, 1 null, 2 the list below
  std::vector<int> emptyColumns;
  bool rawLastConformer;            // boltz2: the last conformer group in its stored raw frame (boltz2Raw...)
  int declaredPolymerBonds;         // 0 null, 1 "all", 2 "discontinuous", 3 "ligand"
  bool templateGapWhereNoAtoms;     // openbind0: every (slot, token) without template atoms reads GAP
  bool ccdUnknownResidue;           // openbind0: an unknown amino acid as the CCD's UNK (unkCcdConformer)
  int mainMsaCap;                   // openbind0: each chain's first N unpaired rows, deduplicated (0 none)
  bool declaredBondFeature;         // chai1: every declared bond as its own directed token matrix (declaredBondMatrix)
  bool chaiConformers;              // chai1: standard residues on chai-lab's conformers (chai1Conformers)
  bool nativeNonproteinMsa;         // opendde, protenix: a ligand UNK in the MSA, a nucleic chain's gap rows its query
  bool translateConformers;         // opendde, protenix: each centred conformer moved by U(-1, 1) per axis (the fold's seed)
  bool declaredBondLeavingAtoms;    // opendde, protenix: the CCD leaving group at each declared bond dropped (declaredBondLeaving)
  bool keepLigandLeavingAtoms;      // boltz2, intellifold2, chai1: a ligand's glycan keeps its O1 (glycanO1Leaving)
  bool dnaGapMsa;                   // openbind0: a DNA chain gap in every MSA row, DNA and ligand profiles zero
  bool modifiedLeavingAtEnds;       // openbind0: a modified residue's CCD leaving atoms at the chain ends too
  bool modifiedNucleotideKeepsOp3;  // intellifold2, chai1: a modified nucleotide keeps its OP3 mid-chain
  bool modifiedProteinLetter;       // intellifold2: a modified residue's restype its CCD one-letter code
  bool unresolvedLeavingAtoms;      // chai1: a ligand's leaving group at a declared bond a token it does not place
  bool ligandMsaUnk;                // boltz2: a ligand token's MSA query row the protein UNK, not the gap
  bool proteinOnlyMsa, zeroDeletions, profileOverAllRows;   // boltz2 (rf3: the first): its alignment, featurise.js
  bool protenixUnpairedMsa;         // protenix2, opendde: a protein chain's paired rows in its unpaired MSA (pairedIntoUnpaired)
  bool atomizeLigandBonded, declaredBondCharge, rf3ProfileMsa;   // rosettafold3 (atomworks)
  bool if2ProfileMsa;               // intellifold2: the profile over each chain's own sequences (msa-features.js)
  int templateModified;             // a template's modified residue: 0 parent, 1 unk_n, 2 drop_hit, 3 unk_restype
  bool templateAltlocFirst;         // boltz2, rf3: a template residue's first altloc
};
// shared/af3/featurise/boltz2-conformers.js: a residue's stored atoms in Boltz-2's raw frame
struct RawAtom { const char* name; double x, y, z; };
struct RawConformer { const char* residue; std::vector<RawAtom> atoms; };

#include "tables.inc"

// shared/af3/featurise/reference-conformers.js conformerFor / aatypeFor
inline const ConformerEntry& proteinEntry(char code) {
  for (auto& e : proteinConformers()) if (e.code == code) return e;
  for (auto& e : proteinConformers()) if (e.code == 'X') return e;
  throw std::logic_error("no X conformer");
}
inline const std::vector<ConformerAtom>& conformerFor(char code, bool cTerminal) {
  const auto& e = proteinEntry(code);
  return cTerminal ? e.terminal : e.internal;
}
inline int aatypeFor(char code) { return proteinEntry(code).aatype; }
// reference-conformers-nucleic.js: nullptr / -1 for anything that is not a nucleotide of that kind
inline const ConformerEntry* nucleicEntry(const std::string& kind, char code) {
  const std::vector<ConformerEntry>* t = kind == "dna" ? &dnaConformers() : kind == "rna" ? &rnaConformers() : nullptr;
  if (!t) return nullptr;
  for (auto& e : *t) if (e.code == code) return &e;
  return nullptr;
}
inline int nucleicAatypeFor(const std::string& kind, char code) {
  const auto* e = nucleicEntry(kind, code);
  return e ? e->aatype : -1;
}

// shared/af3/dialect.js dialectFor: an unknown name raises
inline const FamilyFlags& familyFor(std::string model) {
  for (auto& [alias, name] : familyAliases()) if (alias == model) { model = name; break; }
  for (auto& f : familyFlags()) if (model == f.name) return f;
  std::string known;
  for (auto& f : familyFlags()) known += (known.empty() ? "" : ", ") + std::string(f.name);
  throw std::runtime_error("no AF3 dialect for model \"" + model + "\"; known: " + known);
}

}  // namespace lf
