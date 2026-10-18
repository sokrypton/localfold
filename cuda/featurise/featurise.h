// AF3's batch, as shared/af3/featurise/featurise.js featuriseProtein builds it and batch.js af3BatchFromA3m
// wraps it: polymers (protein, DNA, RNA), modified residues and bases, ligands (one component or a chain of
// them), declared bonds, the alignment and its profile - every per-family convention the JavaScript has, in its
// order. Float32 values are rounded at the points the JavaScript rounds them (its Float32Array writes), so the
// result is byte-identical; tools/check-native-featuriser.py holds it to that.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "ccd.h"
#include "msa.h"
#include "tables.h"

namespace lf {

constexpr int DENSE = 24, QUERIES = 32, KEYS = 128, RESTYPES = 31, UNK_AATYPE = 20, MSA_GAP = 21;

struct Gather { std::vector<int> indices; std::vector<float> mask; int count = 0; };
inline Gather gather(int count) { Gather g; g.indices.assign(count, 0); g.mask.assign(count, 0); g.count = count; return g; }

struct LigandSpan { int from, count; std::string code; std::vector<CompBond> bonds; };
struct ModifiedSpan {
  int from, count; std::string code; int residue;
  std::vector<CompAtom> atoms; std::vector<CompBond> bonds; bool oneToken = false;
};

struct FeaturiseOptions {
  std::vector<Component> ligands;                 // resolved components (a chain of several: ligandChain)
  struct Mod { int chain; int position; Component component; };
  std::vector<Mod> modifications;
  std::vector<DeclaredBond> bonds;
  std::vector<std::string> chainKinds;
  bool symmetriseBonds = false, centreRefConformers = false, paddedAtomKeys = false, qblockAtomKeys = false;
  bool rawLastConformer = false;
  int declaredPolymerBonds = 0;
  bool ccdUnknownResidue = false;                 // openbind0: an unknown amino acid as the CCD's UNK
  bool declaredBondFeature = false;               // chai1: every declared bond, [from][to], in declaredBondMatrix
  bool chaiConformers = false;                    // chai1: standard residues on chai-lab's conformers
  bool translateConformers = false; double conformerSeed = 0;
  bool declaredBondLeavingAtoms = false;          // opendde, protenix: the leaving group at each declared bond dropped   // opendde, protenix: each conformer moved by U(-1, 1)
  int terminalAtoms = 0;                         // 0 the AF3 rule, 1 none (false), 2 a nucleotide's only ("nucleic")
  bool duplicateQueryRow = false, msaCoversNucleic = false;
  bool queryOnce = false;                         // dedupeSelfMsa true: an MSA that is all the query, one row (featurise.js)
  bool nativeNonproteinMsa = false;               // opendde, protenix: a ligand UNK in the MSA and profile, a nucleic chain's gap rows its query
  bool atomizedElementNames = false, atomizedUnknownMsa = false;
  int atomizedUnknownRestype = 0;                 // 0 false, 1 an amino acid's, 2 "nucleic" (a base's too), 3 "chai" (a base's 30)
  bool atomizedBackboneBonds = false, modifiedAsOneToken = false;
  // the alignment's rows (msa-features.js)
  std::vector<std::vector<int>> msa;
  std::vector<std::vector<float>> deletionMatrix;
  bool hasUnpairedFrom = false; int unpairedFrom = 0;
  std::vector<std::vector<int>> profileMsa;
  std::vector<std::vector<float>> profileDeletionMatrix;
};

struct Batch {
  std::string sequence;
  std::vector<std::string> chains, chainKinds;
  std::vector<float> chainLengths;
  int tokens = 0, dense = DENSE, subsets = 0, atomCount = 0, sequences = 0, keys = 0;
  int pairedRows = 0;                              // the paired block's row count (chai-1's IsPairedMSA)
  std::vector<int> aatype; std::vector<float> profile, deletionMean;
  std::vector<int> msa; std::vector<float> msaMask, deletionMatrix;
  std::vector<int> residueIndex, tokenIndex, asymId, entityId, symId; std::vector<float> seqMask;
  std::vector<float> refPos, refMask; std::vector<int> refElement; std::vector<float> refCharge;
  std::vector<int> refAtomNameChars, refSpaceUid, displayAtomNameChars;
  bool hasBonds = false; std::vector<float> bondMatrix, bondOrderMatrix;
  std::vector<float> declaredBondMatrix;           // chai1's declared-bond feature, empty when there is none
  std::vector<LigandSpan> ligandSpans; std::vector<ModifiedSpan> modifiedSpans;
  std::vector<int> residueOfToken; std::vector<float> chainOfResidue;
  std::vector<int> isDna, isRna, isLigand, isModified;
  Gather tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms, tokensToQueries, tokensToKeys, tokenAtomsToPseudoBeta;
};

inline void writeAtomName(std::vector<int>& target, int flat, const std::string& name, int element, bool elementNames) {
  std::string text = name;
  if (elementNames) {
    const auto& s = elementSymbols();
    text = element >= 1 && element <= (int)s.size() ? s[element - 1] : "C";
  }
  for (int c = 0; c < 4; ++c) target[flat * 4 + c] = c < (int)text.size() ? (unsigned char)text[c] - 32 : 0;
}

inline void atomGathers(Batch& b, const std::vector<int>& realAtoms, const std::vector<int>& pseudoBetaSlot,
                        bool paddedKeys, bool qblockKeys) {
  int tokens = b.tokens, dense = b.dense, atomCount = (int)realAtoms.size();
  int subsets = std::max(1, (atomCount + QUERIES - 1) / QUERIES);
  b.tokenAtomsToQueries = gather(subsets * QUERIES);
  for (int q = 0; q < atomCount; ++q) { b.tokenAtomsToQueries.indices[q] = realAtoms[q]; b.tokenAtomsToQueries.mask[q] = 1; }
  b.queriesToTokenAtoms = gather(tokens * dense);
  for (int q = 0; q < atomCount; ++q) { b.queriesToTokenAtoms.indices[realAtoms[q]] = q; b.queriesToTokenAtoms.mask[realAtoms[q]] = 1; }
  int keys = std::min(KEYS, atomCount);
  b.queriesToKeys = gather(subsets * keys);
  b.tokensToQueries = gather(subsets * QUERIES);
  b.tokensToKeys = gather(subsets * keys);
  std::vector<int> tokenOfQuery(atomCount);
  for (int q = 0; q < atomCount; ++q) tokenOfQuery[q] = realAtoms[q] / dense;
  int edge = qblockKeys ? ((atomCount + QUERIES - 1) / QUERIES) * QUERIES : atomCount;
  int lastStart = std::max(0, edge - (qblockKeys ? KEYS : keys));   // (the full 128-key window: featurise.js)
  for (int s = 0; s < subsets; ++s) {
    // (keys - QUERIES) / 2 in JavaScript is a double: with keys < 32 it can be fractional, and min/max keep it
    double startD = paddedKeys ? (double)(s * QUERIES + (QUERIES >> 1) - (keys >> 1))
                               : std::min(std::max(s * QUERIES - (keys - QUERIES) / 2.0, 0.0), (double)lastStart);
    for (int k = 0; k < keys; ++k) {
      double queryD = startD + k;
      int at = s * keys + k;
      if (queryD < 0 || queryD >= atomCount || std::floor(queryD) != queryD) continue;
      int query = (int)queryD;
      b.queriesToKeys.indices[at] = query; b.queriesToKeys.mask[at] = 1;
      b.tokensToKeys.indices[at] = tokenOfQuery[query]; b.tokensToKeys.mask[at] = 1;
    }
    for (int slot = 0; slot < QUERIES; ++slot) {
      int query = s * QUERIES + slot;
      if (query >= atomCount) continue;
      b.tokensToQueries.indices[query] = tokenOfQuery[query]; b.tokensToQueries.mask[query] = 1;
    }
  }
  b.tokenAtomsToPseudoBeta = gather(tokens);
  for (int t = 0; t < tokens; ++t) {
    b.tokenAtomsToPseudoBeta.indices[t] = t * dense + pseudoBetaSlot[t];
    b.tokenAtomsToPseudoBeta.mask[t] = pseudoBetaSlot[t] >= 0 ? 1 : 0;
  }
  b.subsets = subsets;
  b.keys = keys;
  b.atomCount = atomCount;
}

// protenix's and OpenDDE's leaving groups at the declared bonds (shared/af3/featurise/declared-leaving.js): the atoms
// (indices into the component) of the first `count` leaving groups at `centre`, within [from, from + length)
inline std::vector<int> leavingGroups(const Component& c, int from, int length, const std::string& centre, int count) {
  int at = -1;
  for (int k = from; k < from + length; ++k) if (c.atoms[k].name == centre) { at = k; break; }
  if (at < 0) return {};
  std::map<int, std::vector<int>> nbrs;
  for (auto& b : c.bonds) { nbrs[b.from].push_back(b.to); nbrs[b.to].push_back(b.from); }
  std::vector<int> starts;
  for (int m : nbrs[at]) if (c.atoms[m].leaving) starts.push_back(m);
  std::sort(starts.begin(), starts.end());
  std::vector<std::set<int>> groups;
  for (int start : starts) {
    std::set<int> group; std::vector<int> todo{start};
    while (!todo.empty()) {
      int k = todo.back(); todo.pop_back();
      if (group.count(k)) continue;
      group.insert(k);
      for (int m : nbrs[k]) if (c.atoms[m].leaving && m != at) todo.push_back(m);
    }
    if (std::find(groups.begin(), groups.end(), group) == groups.end()) groups.push_back(group);
  }
  std::vector<int> out;
  for (size_t g = 0; g < groups.size() && (int)g < count; ++g) out.insert(out.end(), groups[g].begin(), groups[g].end());
  return out;
}
inline Component withoutAtoms(const Component& l, const std::set<int>& drop) {
  if (drop.empty()) return l;
  Component out = l;
  out.atoms.clear(); out.bonds.clear();
  std::map<int, int> renumber;
  for (int k = 0; k < (int)l.atoms.size(); ++k) if (!drop.count(k)) { renumber[k] = (int)out.atoms.size(); out.atoms.push_back(l.atoms[k]); }
  for (auto& b : l.bonds)
    if (renumber.count(b.from) && renumber.count(b.to)) out.bonds.push_back({renumber[b.from], renumber[b.to], b.order});
  for (size_t p = 0; p < l.residues.size(); ++p) {
    int kept = 0, first = -1;
    for (int k = l.residues[p].from; k < l.residues[p].from + l.residues[p].count; ++k) {
      auto it = renumber.find(k);
      if (it == renumber.end()) continue;
      if (first < 0) first = it->second;
      ++kept;
    }
    out.residues[p].from = first < 0 ? (int)out.atoms.size() : first; out.residues[p].count = kept;
  }
  return out;
}
// -> the ligands without their leaving groups, and the (chain, index) of each chain-end residue whose terminal atom goes
inline std::vector<Component> declaredBondLeaving(const std::vector<std::string>& chains, const std::vector<std::string>& kinds,
                                                  const FeaturiseOptions& o, std::set<std::pair<int, int>>& terminalDropped) {
  std::vector<std::pair<std::tuple<int, int, std::string>, int>> count;     // (in first-seen order, as a Map)
  for (auto& bond : o.bonds)
    for (const BondEnd* e : {&bond.from, &bond.to}) {
      if (!e->hasAtom || e->atom.empty()) continue;
      auto key = std::make_tuple(e->asym, e->residue, e->atom);
      bool found = false;
      for (auto& [k, n] : count) if (k == key) { ++n; found = true; }
      if (!found) count.push_back({key, 1});
    }
  std::vector<std::set<int>> dropOf(o.ligands.size());
  for (auto& [key, n] : count) {
    auto [asym, residue, atom] = key;
    if (asym >= (int)chains.size()) {
      int li = asym - (int)chains.size();
      if (li >= (int)o.ligands.size()) continue;
      const Component& l = o.ligands[li];
      CompPart whole{l.code, 0, (int)l.atoms.size()};
      const CompPart* part = l.residues.empty() ? &whole
        : (residue - 1 >= 0 && residue - 1 < (int)l.residues.size() ? &l.residues[residue - 1] : nullptr);
      if (!part) continue;
      for (int k : leavingGroups(l, part->from, part->count, atom, n)) dropOf[li].insert(k);
      continue;
    }
    const std::string kind = asym < (int)kinds.size() ? kinds[asym] : "protein";
    int at = residue - 1, length = asym >= 0 && asym < (int)chains.size() ? (int)chains[asym].size() : 0;
    bool end = kind == "protein" ? at == length - 1 : at == 0;
    if (!end) continue;
    const std::string terminalAtom = kind == "protein" ? "OXT" : "OP3";
    const FeaturiseOptions::Mod* mod = nullptr;
    for (auto& m : o.modifications) if (m.chain == asym && m.position == residue) { mod = &m; break; }
    bool drops = false;
    if (mod) {
      for (int k : leavingGroups(mod->component, 0, (int)mod->component.atoms.size(), atom, n))
        if (mod->component.atoms[k].name == terminalAtom) drops = true;
    } else {
      drops = atom == (kind == "protein" ? "C" : "P");
    }
    if (drops) terminalDropped.insert({asym, at});
  }
  std::vector<Component> out;
  for (size_t k = 0; k < o.ligands.size(); ++k) out.push_back(withoutAtoms(o.ligands[k], dropOf[k]));
  return out;
}

inline Batch featuriseProtein(const std::string& sequence, const FeaturiseOptions& o) {
  Batch b;
  {
    size_t s = 0, d;
    std::vector<std::string> parts;
    while ((d = sequence.find(':', s)) != std::string::npos) { parts.push_back(sequence.substr(s, d - s)); s = d + 1; }
    parts.push_back(sequence.substr(s));
    for (auto& p : parts) if (!p.empty()) b.chains.push_back(p);
  }
  const auto& chains = b.chains;
  for (auto& c : chains) b.sequence += c;
  int residueCount = (int)b.sequence.size();
  for (size_t i = 0; i < chains.size(); ++i) b.chainKinds.push_back(i < o.chainKinds.size() ? o.chainKinds[i] : "protein");
  const auto& kinds = b.chainKinds;
  // (protenix and OpenDDE: the leaving group at each declared bond goes before tokenising)
  std::set<std::pair<int, int>> terminalDropped;
  const std::vector<Component> leftLigands = o.declaredBondLeavingAtoms ? declaredBondLeaving(chains, kinds, o, terminalDropped)
                                                                        : std::vector<Component>{};
  const auto& ligands = o.declaredBondLeavingAtoms ? leftLigands : o.ligands;
  int ligandTokens = 0;
  for (auto& l : ligands) ligandTokens += (int)l.atoms.size();
  auto parentAatype = [&](const std::string& kind, char code) {
    if (kind == "protein") return aatypeFor(code);
    int a = nucleicAatypeFor(kind, code);
    return a >= 0 ? a : UNK_AATYPE;
  };
  std::map<std::pair<int, int>, const FeaturiseOptions::Mod*> modificationOf;
  for (auto& m : o.modifications) modificationOf[{m.chain, m.position}] = &m;

  struct Residue { char code; std::string kind; bool terminal; bool modified; Component modification; int tokens; };
  std::vector<Residue> residues;
  std::vector<int> chainOfResidue;
  for (int ci = 0; ci < (int)chains.size(); ++ci) {
    const std::string& chain = chains[ci];
    bool protein = kinds[ci] == "protein";
    for (int at = 0; at < (int)chain.size(); ++at) {
      auto asked = modificationOf.find({ci, at + 1});
      Residue r;
      r.kind = kinds[ci];
      r.modified = asked != modificationOf.end();
      const bool dropsTerminal = terminalDropped.count({ci, at}) > 0;
      if (r.modified)
        r.modification = polymerResidue(asked->second->component, (protein ? at == (int)chain.size() - 1 : at == 0) && !dropsTerminal
                                        && !o.modifiedAsOneToken, protein ? "OXT" : "OP3");   // (boltz2: no terminal atom - featurise.js)
      char parent = r.modified ? parentLetter(r.modification.parent, kinds[ci]) : 0;
      r.code = parent ? parent : chain[at];
      r.terminal = (o.terminalAtoms == 1 || (o.terminalAtoms == 2 && protein) || dropsTerminal) ? false
                 : (protein ? at == (int)chain.size() - 1 : at == 0);
      r.tokens = !r.modified || o.modifiedAsOneToken ? 1 : (int)r.modification.atoms.size();
      residues.push_back(std::move(r));
      chainOfResidue.push_back(ci);
    }
  }
  std::vector<int> msaColumnOfResidue(residueCount, -1);
  {
    int column = 0;
    for (int r = 0; r < residueCount; ++r) {
      if (residues[r].kind != "protein" && !o.msaCoversNucleic) continue;
      msaColumnOfResidue[r] = column++;
    }
  }
  int polymerTokens = 0;
  for (auto& r : residues) polymerTokens += r.tokens;
  int tokens = polymerTokens + ligandTokens;
  if (tokens == 0) throw std::runtime_error("featuriseProtein: empty sequence");
  for (auto& c : chains) b.chainLengths.push_back((float)c.size());
  // chainIdentity / residueIndexPerChain (shared/input/chains.js), over residues, counted from zero
  std::vector<int> idAsym(residueCount), idEntity(residueCount), idSym(residueCount), withinChain(residueCount);
  {
    std::vector<std::string> entityKeys;
    std::map<int, int> copiesSeen;
    int residue = 0;
    for (size_t c = 0; c < chains.size(); ++c) {
      int entity = -1;
      for (size_t k = 0; k < entityKeys.size(); ++k) if (entityKeys[k] == chains[c]) entity = (int)k;
      if (entity < 0) { entity = (int)entityKeys.size(); entityKeys.push_back(chains[c]); }
      int copy = copiesSeen[entity]++;
      for (size_t w = 0; w < chains[c].size(); ++w) {
        idAsym[residue] = (int)c; idEntity[residue] = entity; idSym[residue] = copy; withinChain[residue] = (int)w;
        ++residue;
      }
    }
  }
  b.tokens = tokens;
  b.aatype.assign(tokens, 0); b.refPos.assign(tokens * DENSE * 3, 0); b.refMask.assign(tokens * DENSE, 0);
  b.refElement.assign(tokens * DENSE, 0); b.refCharge.assign(tokens * DENSE, 0); b.refAtomNameChars.assign(tokens * DENSE * 4, 0);
  b.refSpaceUid.assign(tokens * DENSE, 0); b.residueIndex.assign(tokens, 0); b.tokenIndex.assign(tokens, 0);
  b.asymId.assign(tokens, 0); b.entityId.assign(tokens, 0); b.symId.assign(tokens, 0); b.seqMask.assign(tokens, 0);
  std::vector<int> realAtoms, pseudoBetaSlot(tokens, -1);
  b.residueOfToken.assign(tokens, -1);
  auto setAtom = [&](int flat, int element, double charge, double x, double y, double z) {
    b.refMask[flat] = 1; b.refElement[flat] = element; b.refCharge[flat] = (float)charge;
    b.refPos[flat * 3] = (float)x; b.refPos[flat * 3 + 1] = (float)y; b.refPos[flat * 3 + 2] = (float)z;
  };
  auto setToken = [&](int token, int aatype, int number, int asym, int entity, int sym, int residue) {
    b.aatype[token] = aatype; b.residueIndex[token] = number; b.tokenIndex[token] = token + 1;
    b.asymId[token] = asym; b.entityId[token] = entity; b.symId[token] = sym; b.seqMask[token] = 1;
    b.residueOfToken[token] = residue;
  };
  int token = 0, space = 0;
  for (int residue = 0; residue < residueCount; ++residue) {
    const Residue& R = residues[residue];
    int asym = idAsym[residue] + 1, entity = idEntity[residue] + 1, sym = idSym[residue] + 1, number = withinChain[residue] + 1;
    if (!R.modified) {
      int aatype = R.kind == "protein" ? aatypeFor(R.code) : parentAatype(R.kind, R.code);
      setToken(token, aatype, number, asym, entity, sym, residue);
      const std::vector<ConformerAtom>* atoms;
      if (R.kind == "protein") {
        bool standard = R.code != 'X';
        if (standard) { standard = false; for (auto& e : proteinConformers()) if (e.code == R.code) standard = true; }
        atoms = o.ccdUnknownResidue && !standard ? (R.terminal ? &unkCcdConformer().terminal : &unkCcdConformer().internal)
                                                 : &conformerFor(R.code, R.terminal);
      }
      else {
        const ConformerEntry* e = nucleicEntry(R.kind, R.code);
        atoms = e ? (R.terminal ? &e->terminal : &e->internal) : &conformerFor('X', false);
      }
      for (auto& a : *atoms) {
        int flat = token * DENSE + a.slot;
        setAtom(flat, a.element, a.charge, a.x, a.y, a.z);
        std::string name = a.name;
        for (int c = 0; c < 4; ++c) b.refAtomNameChars[flat * 4 + c] = c < (int)name.size() ? (unsigned char)name[c] - 32 : 0;
        realAtoms.push_back(flat);
      }
      bool purine = R.code == 'A' || R.code == 'G';
      const ConformerAtom* beta = nullptr;
      if (R.kind == "protein") {
        for (auto& a : *atoms) if (std::string(a.name) == "CB") { beta = &a; break; }
        if (!beta) for (auto& a : *atoms) if (std::string(a.name) == "CA") { beta = &a; break; }
      } else {
        for (auto& a : *atoms) if (std::string(a.name) == (purine ? "C4" : "C2")) { beta = &a; break; }
      }
      if (beta) pseudoBetaSlot[token] = beta->slot;
      else if (!atoms->empty() && !(R.kind != "protein" && number == 1 && o.terminalAtoms == 1))
        pseudoBetaSlot[token] = (*atoms)[0].slot;      // (AF3's first held atom; masked at a dropped 5' OP3: featurise.js)
      for (int s = 0; s < DENSE; ++s) b.refSpaceUid[token * DENSE + s] = space;
      ++space; ++token;
      continue;
    }
    const Component& M = R.modification;
    int uid = space++;
    if (o.modifiedAsOneToken) {
      ModifiedSpan span{token, 1, M.code, residue, M.atoms, M.bonds, true};
      b.modifiedSpans.push_back(span);
      // (the chain's unknown, as genuine Boltz-2's get_unk_token: a nucleic chain's 30, a DNA one's DN row added - featurise.js)
      setToken(token, o.atomizedUnknownRestype != 0 ? (R.kind != "protein" ? 30 : UNK_AATYPE) : parentAatype(R.kind, R.code), number, asym, entity, sym, residue);
      // (its atoms packed where its component slots run past the token's, as genuine Boltz-2 keeps them all: featurise.js)
      bool packed = false;
      for (size_t k = 0; k < M.atoms.size(); ++k) if ((M.atoms[k].componentSlot >= 0 ? M.atoms[k].componentSlot : (int)k) >= DENSE) packed = true;
      auto slotOf = [&](size_t k) { return packed ? (int)k : (M.atoms[k].componentSlot >= 0 ? M.atoms[k].componentSlot : (int)k); };
      for (size_t atom = 0; atom < M.atoms.size(); ++atom) {
        const CompAtom& s = M.atoms[atom];
        int slotHere = slotOf(atom);
        if (slotHere >= DENSE) {
          fprintf(stderr, "%s: atom %s is past the token's %d slots and is dropped\n",
                  M.code.c_str(), s.name.c_str(), DENSE);
          continue;
        }
        int flat = token * DENSE + slotHere;
        setAtom(flat, s.element, s.charge, s.x, s.y, s.z);
        writeAtomName(b.refAtomNameChars, flat, s.name, s.element, o.atomizedElementNames);
        realAtoms.push_back(flat);
      }
      // (its representative atom is its first, as genuine Boltz-2's atom_center/atom_disto 0: featurise.js)
      int firstHeld = 0;
      while (firstHeld < DENSE - 1 && !b.refMask[token * DENSE + firstHeld]) ++firstHeld;
      pseudoBetaSlot[token] = firstHeld;
      for (int s = 0; s < DENSE; ++s) b.refSpaceUid[token * DENSE + s] = uid;
      ++token;
      continue;
    }
    b.modifiedSpans.push_back({token, (int)M.atoms.size(), M.code, residue, M.atoms, M.bonds, false});
    for (size_t atom = 0; atom < M.atoms.size(); ++atom) {
      const CompAtom& s = M.atoms[atom];
      setToken(token, o.atomizedUnknownRestype == 2 || ((o.atomizedUnknownRestype == 1 || o.atomizedUnknownRestype == 3) && R.kind == "protein") ? UNK_AATYPE
                      : o.atomizedUnknownRestype == 3 ? 30 : parentAatype(R.kind, R.code),
               number, asym, entity, sym, residue);
      int flat = token * DENSE;
      setAtom(flat, s.element, s.charge, s.x, s.y, s.z);
      writeAtomName(b.refAtomNameChars, flat, s.name, s.element, o.atomizedElementNames);
      realAtoms.push_back(flat);
      pseudoBetaSlot[token] = 0;
      for (int k = 0; k < DENSE; ++k) b.refSpaceUid[token * DENSE + k] = uid;
      ++token;
    }
  }

  int asym = (int)chains.size();
  int polymerEntities = 0;
  for (int e : idEntity) polymerEntities = std::max(polymerEntities, e + 1);
  std::vector<std::string> entityOfLigand;
  std::map<int, int> copiesOfEntity;
  int ligandToken = polymerTokens;
  std::vector<int> ligandStart;
  auto partsOf = [](const Component& l) {
    return l.residues.empty() ? std::vector<CompPart>{{l.code, 0, (int)l.atoms.size()}} : l.residues;
  };
  for (const auto& ligand : ligands) {
    ++asym;
    ligandStart.push_back(ligandToken);
    auto parts = partsOf(ligand);
    for (auto& part : parts) {
      LigandSpan span{ligandToken + part.from, part.count, part.code, {}};
      for (auto& bd : ligand.bonds)
        if (bd.from >= part.from && bd.from < part.from + part.count) span.bonds.push_back({bd.from - part.from, bd.to - part.from, bd.order});
      b.ligandSpans.push_back(span);
    }
    std::string identity = ligand.code + "|" + std::to_string(ligand.atoms.size()) + "|";
    for (size_t k = 0; k < ligand.atoms.size(); ++k)
      identity += (k ? "," : "") + std::to_string(ligand.atoms[k].element) + ":" + jsNumber(ligand.atoms[k].charge);
    identity += "|";
    for (size_t k = 0; k < ligand.bonds.size(); ++k)
      identity += (k ? "," : "") + std::to_string(ligand.bonds[k].from) + "-" + std::to_string(ligand.bonds[k].to) + ":" + std::to_string(ligand.bonds[k].order);
    int entity = -1;
    for (size_t k = 0; k < entityOfLigand.size(); ++k) if (entityOfLigand[k] == identity) entity = (int)k;
    if (entity < 0) { entity = (int)entityOfLigand.size(); entityOfLigand.push_back(identity); }
    int copy = ++copiesOfEntity[entity];
    std::vector<int> uidOfPart;
    for (size_t p = 0; p < parts.size(); ++p) uidOfPart.push_back(space++);
    for (size_t atom = 0; atom < ligand.atoms.size(); ++atom) {
      int t = ligandToken + (int)atom;
      const CompAtom& s = ligand.atoms[atom];
      int part = 0;
      for (size_t p = 0; p < parts.size(); ++p)
        if ((int)atom >= parts[p].from && (int)atom < parts[p].from + parts[p].count) { part = (int)p; break; }
      b.aatype[t] = UNK_AATYPE; b.residueIndex[t] = part + 1; b.tokenIndex[t] = t + 1;
      b.asymId[t] = asym; b.entityId[t] = polymerEntities + entity + 1; b.symId[t] = copy; b.seqMask[t] = 1;
      int flat = t * DENSE;
      setAtom(flat, s.element, s.charge, s.x, s.y, s.z);
      writeAtomName(b.refAtomNameChars, flat, s.name, s.element, o.atomizedElementNames);
      realAtoms.push_back(flat);
      pseudoBetaSlot[t] = 0;
      for (int k = 0; k < DENSE; ++k) b.refSpaceUid[t * DENSE + k] = uidOfPart[part];
    }
    ligandToken += (int)ligand.atoms.size();
  }

  struct Group { int base; const std::vector<CompBond>* bonds; };
  std::vector<Group> bondedGroups;
  for (auto& span : b.modifiedSpans) if (!span.oneToken) bondedGroups.push_back({span.from, &span.bonds});
  {
    int base = polymerTokens;
    for (auto& l : ligands) { bondedGroups.push_back({base, &l.bonds}); base += (int)l.atoms.size(); }
  }
  int chainCount = (int)chains.size();
  auto tokenOfEndpoint = [&](const BondEnd& e, const std::string& where) -> int {
    if (e.asym < 0) throw std::runtime_error(where + ": chain " + std::to_string(e.asym) + " is not one this batch has");
    if (e.asym >= chainCount) {
      int li = e.asym - chainCount;
      if (li >= (int)ligands.size()) throw std::runtime_error(where + ": no chain " + std::to_string(e.asym));
      const Component& ligand = ligands[li];
      auto parts = partsOf(ligand);
      const CompPart* part = parts.size() == 1 ? &parts[0] : (e.residue - 1 >= 0 && e.residue - 1 < (int)parts.size() ? &parts[e.residue - 1] : nullptr);
      if (!part) throw std::runtime_error(where + ": " + ligand.code + " has no residue " + std::to_string(e.residue));
      for (int k = 0; k < part->count; ++k)
        if (e.hasAtom && ligand.atoms[part->from + k].name == e.atom) return ligandStart[li] + part->from + k;
      throw std::runtime_error(where + ": " + part->code + " has no atom " + (e.hasAtom ? e.atom : "undefined"));
    }
    int global = e.residue - 1;
    for (int before = 0; before < e.asym; ++before) global += (int)chains[before].size();
    if (e.residue < 1 || global < 0 || global >= residueCount)
      throw std::runtime_error(where + ": residue " + std::to_string(e.residue) + " is past the end of chain " + std::to_string(e.asym));
    const ModifiedSpan* span = nullptr;
    for (auto& s : b.modifiedSpans) if (s.residue == global) { span = &s; break; }
    if (!span) {
      for (int t = 0; t < tokens; ++t) if (b.residueOfToken[t] == global) {
        // ...and a named atom must be one the token has (featurise.js)
        if (e.hasAtom && !e.atom.empty()) {
          bool found = false;
          for (int k = 0; k < DENSE && !found; ++k) {
            const int flat = t * DENSE + k;
            if (b.refMask[flat] == 0) continue;
            std::string name;
            for (int c = 0; c < 4; ++c) { int v = b.refAtomNameChars[flat * 4 + c]; if (v >= 0 && v < 64) name += (char)(v + 32); }
            while (!name.empty() && name.back() == ' ') name.pop_back();
            while (!name.empty() && name.front() == ' ') name.erase(name.begin());
            found = name == e.atom;
          }
          if (!found) throw std::runtime_error(where + ": " + std::string(1, residues[global].code) + " " + std::to_string(e.residue) + " has no atom " + e.atom);
        }
        return t;
      }
      throw std::runtime_error(where + ": residue " + std::to_string(e.residue) + " has no token");
    }
    if (span->oneToken) return span->from;
    for (size_t k = 0; k < span->atoms.size(); ++k) if (e.hasAtom && span->atoms[k].name == e.atom) return span->from + (int)k;
    throw std::runtime_error(where + ": " + span->code + " has no atom " + (e.hasAtom ? e.atom : "undefined"));
  };
  // each family's rule for the declared bonds (featurise.js, af3-any-model 964990e)
  struct Resolved { int from, to, order; bool bothWays; };
  std::vector<Resolved> declared;
  auto isLigandEnd = [&](const BondEnd& e) { return e.asym >= chainCount; };
  auto isModifiedEnd = [&](const BondEnd& e) {
    int global = e.residue - 1;
    for (int before = 0; before < e.asym; ++before) global += (int)chains[before].size();
    for (auto& s : b.modifiedSpans) if (s.residue == global) return true;
    return false;
  };
  for (size_t i = 0; i < o.bonds.size(); ++i) {
    std::string where = "bond " + std::to_string(i + 1);
    const BondEnd &f = o.bonds[i].from, &t = o.bonds[i].to;
    const int from = tokenOfEndpoint(f, where + " from"), to = tokenOfEndpoint(t, where + " to");
    if (o.declaredBondFeature) {
      if (b.declaredBondMatrix.empty()) b.declaredBondMatrix.assign((size_t)tokens * tokens, 0);
      b.declaredBondMatrix[(size_t)from * tokens + to] = 1;
    }
    const bool fl = isLigandEnd(f), tl = isLigandEnd(t);
    bool added;
    if (fl && tl) added = false;
    else if (fl || tl) added = o.declaredPolymerBonds != 0 && isModifiedEnd(fl ? t : f);   // (AF3's own; a rule adds it both ways)
    else if (o.declaredPolymerBonds == 1) added = true;
    else if (o.declaredPolymerBonds == 2 && !(f.asym == t.asym && std::abs(f.residue - t.residue) <= 1)) added = true;
    else continue;
    if (added && from == to) continue;
    declared.push_back({from, to, 5, added});
  }
  bool anyBond = !declared.empty();
  for (auto& g : bondedGroups) if (!g.bonds->empty()) anyBond = true;
  if (anyBond) {
    b.hasBonds = true;
    b.bondMatrix.assign((size_t)tokens * tokens, 0);
    b.bondOrderMatrix.assign((size_t)tokens * tokens, 0);
    auto set = [&](int i, int j, int order) {
      b.bondMatrix[(size_t)i * tokens + j] = 1;
      b.bondOrderMatrix[(size_t)i * tokens + j] = (float)order;
    };
    for (auto& g : bondedGroups)
      for (auto& bd : *g.bonds) {
        set(g.base + bd.from, g.base + bd.to, bd.order);
        if (o.symmetriseBonds) set(g.base + bd.to, g.base + bd.from, bd.order);
      }
    for (auto& d : declared) {
      set(d.from, d.to, d.order);
      if (o.symmetriseBonds || d.bothWays) set(d.to, d.from, d.order);
    }
    if (o.atomizedBackboneBonds) {
      for (auto& span : b.modifiedSpans) {
        auto slotOf = [&](const std::string& name) {
          for (size_t k = 0; k < span.atoms.size(); ++k) if (span.atoms[k].name == name) return (int)k;
          return -1;
        };
        bool nucleic = residues[span.residue].kind != "protein";
        int nitrogen = slotOf(nucleic ? "P" : "N"), carbon = slotOf(nucleic ? "O3'" : "C");
        auto sameChain = [&](int r) { return r >= 0 && r < residueCount && chainOfResidue[r] == chainOfResidue[span.residue]; };
        auto link = [&](int a, int c) {
          if (a < 0 || c < 0 || a >= tokens || c >= tokens) return;
          b.bondMatrix[(size_t)a * tokens + c] = 1; b.bondMatrix[(size_t)c * tokens + a] = 1;
          b.bondOrderMatrix[(size_t)a * tokens + c] = 1; b.bondOrderMatrix[(size_t)c * tokens + a] = 1;
        };
        auto spanOf = [&](int r) -> const ModifiedSpan* {
          for (auto& s : b.modifiedSpans) if (s.residue == r) return &s;
          return nullptr;
        };
        if (nitrogen >= 0 && sameChain(span.residue - 1) && spanOf(span.residue - 1) == nullptr) link(span.from - 1, span.from + nitrogen);
        if (carbon >= 0 && sameChain(span.residue + 1)) {
          const ModifiedSpan* next = spanOf(span.residue + 1);
          int into = 0;
          if (next) {
            into = -1;
            for (size_t k = 0; k < next->atoms.size(); ++k) if (next->atoms[k].name == (nucleic ? "P" : "N")) { into = (int)k; break; }
          }
          if (into >= 0) link(span.from + carbon, span.from + span.count + into);
        }
      }
    }
    b.bondMatrix[0] = 0;
    b.bondOrderMatrix[0] = 0;
  }

  atomGathers(b, realAtoms, pseudoBetaSlot, o.paddedAtomKeys, o.qblockAtomKeys);

  // ---- the MSA
  const auto& extra = o.msa;
  bool duplicateQuery = extra.empty() && o.duplicateQueryRow;
  int sequences = 1 + (int)extra.size() + (duplicateQuery ? 1 : 0);
  b.sequences = sequences;
  b.msa.assign((size_t)sequences * tokens, 0);
  b.msaMask.assign((size_t)sequences * tokens, 1);
  b.deletionMatrix.assign((size_t)sequences * tokens, 0);
  std::vector<int> queryRow = b.aatype;
  if ((o.atomizedUnknownRestype == 1 || o.atomizedUnknownRestype == 3) && !o.atomizedUnknownMsa)
    for (auto& span : b.modifiedSpans) {
      // (genuine Boltz-2's one-token residue reads the protein unknown in the query row: featurise.js)
      int parent = span.oneToken ? UNK_AATYPE : parentAatype(residues[span.residue].kind, residues[span.residue].code);
      for (int at = 0; at < span.count; ++at) queryRow[span.from + at] = parent;
    }
  for (int t = polymerTokens; t < tokens; ++t) queryRow[t] = MSA_GAP;
  std::copy(queryRow.begin(), queryRow.end(), b.msa.begin());
  if (duplicateQuery) std::copy(queryRow.begin(), queryRow.end(), b.msa.begin() + tokens);
  std::vector<int> msaColumnOfToken(tokens, -1);
  std::vector<char> nucleicToken(tokens, 0);
  for (int t = 0; t < tokens; ++t) {
    int r = b.residueOfToken[t];
    if (r < 0) continue;
    msaColumnOfToken[t] = msaColumnOfResidue[r];
    if (residues[r].kind != "protein") nucleicToken[t] = 1;
  }
  int unpairedFrom = o.hasUnpairedFrom ? o.unpairedFrom : (extra.empty() ? 0 : 1);
  b.pairedRows = unpairedFrom;
  std::vector<char> unknownEverywhere(tokens, 0);
  if (o.atomizedUnknownMsa)
    for (auto& span : b.modifiedSpans) std::fill(unknownEverywhere.begin() + span.from, unknownEverywhere.begin() + span.from + span.count, 1);
  int nucleicRow = unpairedFrom;
  for (int row = 0; row < (int)extra.size(); ++row) {
    size_t base = (size_t)(row + 1) * tokens;
    const std::vector<float>* deletions = row < (int)o.deletionMatrix.size() ? &o.deletionMatrix[row] : nullptr;
    bool own = row + 1 == nucleicRow;
    for (int t = 0; t < tokens; ++t) {
      int column = msaColumnOfToken[t];
      if (column < 0 && nucleicToken[t]) { b.msa[base + t] = own ? b.aatype[t] : MSA_GAP; continue; }
      b.msa[base + t] = unknownEverywhere[t] ? queryRow[t]
                      : column < 0 ? MSA_GAP : (column < (int)extra[row].size() ? extra[row][column] : MSA_GAP);
      if (deletions) b.deletionMatrix[base + t] = column < 0 ? 0 : (column < (int)deletions->size() ? (*deletions)[column] : 0);
    }
  }
  bool useProfileRows = !o.profileMsa.empty();
  b.profile.assign((size_t)tokens * RESTYPES, 0);
  b.deletionMean.assign(tokens, 0);
  int profileDepth = !useProfileRows ? std::max(1, sequences - unpairedFrom) : std::max(1, (int)o.profileMsa.size());
  auto codeAt = [&](int row, int t) -> int {
    if (!useProfileRows) return b.msa[(size_t)(unpairedFrom + row) * tokens + t];
    int column = msaColumnOfToken[t];
    if (column < 0) return MSA_GAP;
    if (row >= (int)o.profileMsa.size()) return -1;
    return unknownEverywhere[t] ? queryRow[t] : o.profileMsa[row][column];
  };
  auto deletionAt = [&](int row, int t) -> double {
    if (!useProfileRows) return b.deletionMatrix[(size_t)(unpairedFrom + row) * tokens + t];
    int column = msaColumnOfToken[t];
    if (column < 0) return 0;
    if (row >= (int)o.profileDeletionMatrix.size() || column >= (int)o.profileDeletionMatrix[row].size()) return 0;
    return o.profileDeletionMatrix[row][column];
  };
  auto ownProfile = [&](int t) { return nucleicToken[t] && msaColumnOfToken[t] < 0; };
  std::map<int, int> depthOfChain;
  for (int t = 0; t < polymerTokens; ++t) {
    if (ownProfile(t) || msaColumnOfToken[t] < 0 || unknownEverywhere[t]) continue;
    int chain = chainOfResidue[b.residueOfToken[t]];
    int deepest = depthOfChain.count(chain) ? depthOfChain[chain] : 1;
    for (int row = profileDepth - 1; row >= deepest; --row) {
      int code = codeAt(row, t);
      if (code >= 0 && code != MSA_GAP) { deepest = row + 1; break; }
    }
    depthOfChain[chain] = deepest;
  }
  auto depthOf = [&](int t) {
    if (t < polymerTokens && b.residueOfToken[t] >= 0) {
      auto it = depthOfChain.find(chainOfResidue[b.residueOfToken[t]]);
      return it == depthOfChain.end() ? profileDepth : it->second;
    }
    return profileDepth;
  };
  for (int t = 0; t < tokens; ++t) {
    if (ownProfile(t)) continue;
    int depth = depthOf(t);
    for (int row = 0; row < depth; ++row) {
      int code = codeAt(row, t);
      float& p = b.profile[(size_t)t * RESTYPES + (code >= 0 && code < RESTYPES ? code : 0)];
      if (code >= 0 && code < RESTYPES) p = (float)((double)p + 1.0 / depth);
      b.deletionMean[t] = (float)((double)b.deletionMean[t] + deletionAt(row, t) / depth);
    }
  }
  for (int t = 0; t < tokens; ++t) if (ownProfile(t)) b.profile[(size_t)t * RESTYPES + queryRow[t]] = 1;
  // a self-MSA is the query once, for the dedupeSelfMsa families: every row the query, one row (featurise.js)
  if (o.queryOnce && sequences > 1) {
    bool self = true;
    for (size_t at = tokens; at < (size_t)sequences * tokens && self; ++at) self = b.msa[at] == b.msa[at % tokens];
    if (self) {
      sequences = 1; b.sequences = 1;
      b.msa.resize(tokens); b.msaMask.resize(tokens); b.deletionMatrix.resize(tokens);
    }
  }
  // (a one-token modified residue whose profile is just the parent's one-hot reads UNK's, as genuine Boltz-2: featurise.js)
  for (auto& span : b.modifiedSpans) {
    if (!span.oneToken) continue;
    int parent = parentAatype(residues[span.residue].kind, residues[span.residue].code);
    size_t row = (size_t)span.from * RESTYPES;
    float total = 0;
    for (int k = 0; k < RESTYPES; ++k) total += b.profile[row + k];
    if (b.profile[row + parent] == 1.f && total == 1.f) { b.profile[row + parent] = 0; b.profile[row + UNK_AATYPE] = 1; }
  }
  // opendde's and protenix's MSA on a non-protein token: a ligand UNK in every row and its profile, a nucleic chain's
  // all-gap rows its query (featurise.js, af3-any-model's _native_nonprotein_msa)
  if (o.nativeNonproteinMsa) {
    for (int row = 0; row < sequences; ++row)
      std::fill(b.msa.begin() + (size_t)row * tokens + polymerTokens, b.msa.begin() + (size_t)(row + 1) * tokens, UNK_AATYPE);
    for (int t = polymerTokens; t < tokens; ++t) {
      std::fill(b.profile.begin() + (size_t)t * RESTYPES, b.profile.begin() + (size_t)(t + 1) * RESTYPES, 0.f);
      b.profile[(size_t)t * RESTYPES + UNK_AATYPE] = 1;
    }
    std::vector<int> chainOrder; std::map<int, std::vector<int>> nucleicChains;
    for (int t = 0; t < polymerTokens; ++t) {
      int r = b.residueOfToken[t];
      if (r < 0 || residues[r].kind == "protein") continue;
      int chain = chainOfResidue[r];
      if (!nucleicChains.count(chain)) chainOrder.push_back(chain);
      nucleicChains[chain].push_back(t);
    }
    for (int chain : chainOrder) {
      const auto& ts = nucleicChains[chain];
      for (int row = 1; row < sequences; ++row) {
        size_t base = (size_t)row * tokens;
        bool gap = true;
        for (int t : ts) if (b.msa[base + t] != MSA_GAP) { gap = false; break; }
        if (gap) for (int t : ts) b.msa[base + t] = b.msa[t];
      }
    }
  }

  b.displayAtomNameChars = b.refAtomNameChars;
  if (o.atomizedElementNames) {
    auto restore = [&](int base, const std::vector<CompAtom>& atoms) {
      for (size_t at = 0; at < atoms.size(); ++at) writeAtomName(b.displayAtomNameChars, (base + (int)at) * DENSE, atoms[at].name, atoms[at].element, false);
    };
    for (auto& span : b.modifiedSpans) restore(span.from, span.atoms);
    int base = polymerTokens;
    for (auto& l : ligands) { restore(base, l.atoms); base += (int)l.atoms.size(); }
  }

  if (o.centreRefConformers) {
    std::map<int, std::array<double, 4>> sums;
    std::vector<int> order;
    for (int slot = 0; slot < tokens * DENSE; ++slot) {
      if (b.refMask[slot] == 0) continue;
      int uid = b.refSpaceUid[slot];
      auto& e = sums[uid];
      e[0] += b.refPos[slot * 3]; e[1] += b.refPos[slot * 3 + 1]; e[2] += b.refPos[slot * 3 + 2]; e[3] += 1;
    }
    for (int slot = 0; slot < tokens * DENSE; ++slot) {
      if (b.refMask[slot] == 0) continue;
      auto it = sums.find(b.refSpaceUid[slot]);
      if (it == sums.end() || it->second[3] == 0) continue;
      auto& e = it->second;
      b.refPos[slot * 3] = (float)((double)b.refPos[slot * 3] - e[0] / e[3]);
      b.refPos[slot * 3 + 1] = (float)((double)b.refPos[slot * 3 + 1] - e[1] / e[3]);
      b.refPos[slot * 3 + 2] = (float)((double)b.refPos[slot * 3 + 2] - e[2] / e[3]);
    }
  }

  // a token that is not atomised takes the table's position of each atom its residue's entry names (featurise.js)
  static const char* NAMES[] = { "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
                                 "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK", "-", "A", "G", "C", "U", "DA", "DG",
                                 "DC", "DT", "N" };
  std::map<std::pair<int, int>, int> share;
  for (int t = 0; t < tokens; ++t) share[{ b.asymId[t], b.residueIndex[t] }] += 1;
  auto overrideConformer = [&](int t, const std::vector<RawConformer>& table) {
    if (share[{ b.asymId[t], b.residueIndex[t] }] > 1) return;
    if (b.aatype[t] < 0 || b.aatype[t] > 30) return;
    const RawConformer* entry = nullptr;
    for (auto& e : table) if (std::string(e.residue) == NAMES[b.aatype[t]]) entry = &e;
    if (!entry) return;
    for (int k = 0; k < DENSE; ++k) {
      const int flat = t * DENSE + k;
      if (b.refMask[flat] == 0) continue;
      std::string name;
      for (int c = 0; c < 4; ++c) { int v = b.refAtomNameChars[flat * 4 + c]; if (v >= 0 && v < 64) name += (char)(v + 32); }
      while (!name.empty() && name.back() == ' ') name.pop_back();
      while (!name.empty() && name.front() == ' ') name.erase(name.begin());
      for (auto& a : entry->atoms) if (name == a.name) {
        b.refPos[flat * 3] = (float)a.x; b.refPos[flat * 3 + 1] = (float)a.y; b.refPos[flat * 3 + 2] = (float)a.z;
        break;
      }
    }
  };
  // chai1: every standard residue on chai-lab's conformer; boltz2: the last conformer group's on Boltz-2's stored one, raw
  if (o.chaiConformers) for (int t = 0; t < tokens; ++t) overrideConformer(t, chai1Conformers());
  if (o.rawLastConformer) {
    bool any = false; int last = 0;
    for (int slot = 0; slot < tokens * DENSE; ++slot)
      if (b.refMask[slot] != 0) { last = any ? std::max(last, b.refSpaceUid[slot]) : b.refSpaceUid[slot]; any = true; }
    for (int t = 0; any && t < tokens; ++t) {
      bool inLast = false;
      for (int k = 0; k < DENSE && !inLast; ++k) inLast = b.refMask[t * DENSE + k] != 0 && b.refSpaceUid[t * DENSE + k] == last;
      if (inLast) overrideConformer(t, boltz2RawConformers());
    }
  }

  // opendde, protenix: each centred conformer moved by U(-1, 1) per axis, three uniforms a reference space in ascending
  // ref_space_uid from the fold's seed (featurise.js)
  if (o.translateConformers) {
    std::set<int> uids;
    for (int slot = 0; slot < tokens * DENSE; ++slot) if (b.refMask[slot] != 0) uids.insert(b.refSpaceUid[slot]);
    Uniform uniform(o.conformerSeed);
    std::map<int, std::array<double, 3>> shift;
    for (int uid : uids) { double x = 2 * uniform() - 1, y = 2 * uniform() - 1, z = 2 * uniform() - 1; shift[uid] = {x, y, z}; }
    for (int slot = 0; slot < tokens * DENSE; ++slot) {
      if (b.refMask[slot] == 0) continue;
      auto& s = shift[b.refSpaceUid[slot]];
      for (int k = 0; k < 3; ++k) b.refPos[slot * 3 + k] = (float)((double)b.refPos[slot * 3 + k] + s[k]);
    }
  }

  b.isDna.assign(tokens, 0); b.isRna.assign(tokens, 0); b.isLigand.assign(tokens, 0); b.isModified.assign(tokens, 0);
  for (int t = 0; t < tokens; ++t) {
    int r = b.residueOfToken[t];
    if (r < 0) continue;
    const std::string& kind = kinds[chainOfResidue[r]];
    b.isDna[t] = kind == "dna"; b.isRna[t] = kind == "rna";
  }
  for (auto& s : b.ligandSpans) std::fill(b.isLigand.begin() + s.from, b.isLigand.begin() + s.from + s.count, 1);
  for (auto& s : b.modifiedSpans) std::fill(b.isModified.begin() + s.from, b.isModified.begin() + s.from + s.count, 1);
  for (int r : chainOfResidue) b.chainOfResidue.push_back((float)r);
  return b;
}

}  // namespace lf
