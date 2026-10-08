// What the AF3 exporter adds beside the batch, per family: each token's contact class
// (shared/af3/featurise/contact-classes.js), chai-1's ESM2 inputs (esm2-input.js), rf3's stereocentres
// (template-features.js chiralCentres) and OpenDDE's structural tokens (structural-tokens.js structuralLayout /
// structuralBatch, cpu/af3/structure/structural-expander.js structuralPairFeatures).
#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>

#include "entries.h"
#include "featurise.h"

namespace lf {

// af3ContactClasses: CLASS_NUCLEIC 0, CLASS_LIGAND 1, CLASS_AMINO 2 + restype, CLASS_PROTEIN 22
inline std::vector<int> af3ContactClasses(const Batch& b) {
  std::vector<int> classes(b.tokens, 22);
  for (int t = 0; t < b.tokens; ++t) {
    int restype = b.aatype[t];
    if (restype >= 22) classes[t] = 0;
    else if (restype < 20) classes[t] = 2 + restype;
  }
  for (auto& s : b.ligandSpans)
    for (int at = 0; at < s.count; ++at) if (s.from + at < b.tokens) classes[s.from + at] = 1;
  return classes;
}

// esm2Inputs: each protein chain's [BOS, residues, EOS] in ESM2's vocabulary, and each token's row (-1: none)
inline void addEsm2Inputs(Entries& e, const Batch& b) {
  static const std::string LETTERS = "ARNDCQEGHILKMFPSTWYV";
  static const std::map<char, int> VOCAB = {{'L', 4}, {'A', 5}, {'G', 6}, {'V', 7}, {'S', 8}, {'E', 9}, {'R', 10},
    {'T', 11}, {'I', 12}, {'D', 13}, {'P', 14}, {'K', 15}, {'Q', 16}, {'N', 17}, {'F', 18}, {'Y', 19}, {'M', 20},
    {'H', 21}, {'W', 22}, {'C', 23}, {'X', 24}};
  std::map<int, char> letterOf;
  for (int t = 0; t < b.tokens; ++t) {
    int r = b.residueOfToken[t];
    if (r < 0 || letterOf.count(r)) continue;
    letterOf[r] = b.aatype[t] >= 0 && b.aatype[t] < 20 ? LETTERS[b.aatype[t]] : 'X';
  }
  std::vector<int> ids, chainLengths, tokenRow(b.tokens, -1);
  std::map<int, int> rowOfResidue;
  int row = 0;
  for (int chain = 0; chain < (int)b.chainKinds.size(); ++chain) {
    if (b.chainKinds[chain] != "protein") continue;
    std::vector<int> residues;
    for (int r = 0; r < (int)b.chainOfResidue.size(); ++r) if ((int)b.chainOfResidue[r] == chain) residues.push_back(r);
    if (residues.empty()) continue;
    ids.push_back(0);
    for (int r : residues) {
      auto l = letterOf.find(r);
      ids.push_back(VOCAB.at(l == letterOf.end() ? 'X' : l->second));
      rowOfResidue[r] = row++;
    }
    ids.push_back(2);
    chainLengths.push_back((int)residues.size());
  }
  for (int t = 0; t < b.tokens; ++t) {
    auto it = rowOfResidue.find(b.residueOfToken[t]);
    if (b.residueOfToken[t] >= 0 && it != rowOfResidue.end()) tokenRow[t] = it->second;
  }
  e.i("esm.ids", ids); e.i("esm.chainLengths", chainLengths); e.i("esm.tokenRow", tokenRow);
}

// chiralCentres: every CA (N, C, CB) and ILE/THR's CB, three impropers each, all four atoms real
inline void addChiralCentres(Entries& e, const Batch& b) {
  const double CHIRAL_ANGLE = 0.61547970867038737;    // Math.asin(1 / Math.sqrt(3)), as V8 computes it
  std::vector<int> centers;
  std::vector<float> angles;
  auto emit = [&](int base, int pivot, int x, int y, int z) {
    for (int slot : {pivot, x, y, z}) if (!(b.refMask[base + slot] > 0)) return;
    int triples[3][4] = {{x, y, z, 1}, {x, z, y, -1}, {y, z, x, 1}};
    for (auto& tr : triples) {
      centers.insert(centers.end(), {base + pivot, base + tr[0], base + tr[1], base + tr[2]});
      angles.push_back((float)(tr[3] * CHIRAL_ANGLE));
    }
  };
  for (int t = 0; t < b.tokens; ++t) {
    int base = t * b.dense;
    emit(base, 1, 0, 2, 4);
    if (b.aatype[t] == 9 || b.aatype[t] == 16) emit(base, 4, 1, 5, 6);
  }
  e.i("chiral.centers", centers);
  e.t("chiral.angles", angles);
  e.m("chiral.count", (double)angles.size());
}

// ---------------------------------------------------------------- OpenDDE's structural tokens
struct StructuralLayout {
  int tokens = 0;
  std::vector<int> parent, role, twin, prevParent, nextParent, pseudoBetaSlot, residueAtomGather, residueRepToken;
  std::vector<std::vector<int>> sources;
};

inline std::string atomNameAt(const Batch& b, int token, int slot) {
  std::string name;
  for (int c = 0; c < 4; ++c) { int v = b.refAtomNameChars[(token * b.dense + slot) * 4 + c]; if (v > 0) name += (char)(v + 32); }
  return name;
}

inline StructuralLayout structuralLayout(const Batch& b) {
  static const std::set<std::string> PROTEIN_BACKBONE = {"N", "CA", "C", "O", "OXT"};
  static const std::set<std::string> NUCLEIC_BACKBONE = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "O5'", "C5'",
    "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "O5*", "C5*", "C4*", "O4*", "C3*", "O3*", "C2*", "O2*", "C1*", "O5T", "O3T"};
  static const std::vector<std::string> PBB = {"CA", "N", "C"}, PSC = {"CB"}, NBB = {"C4'", "C4*", "C1'", "C1*"},
    PUR = {"N9", "C4", "C8", "N7", "C5"}, PYR = {"N1", "C2", "C6", "C5", "C4"}, OTH = {"C1'", "C1*", "N9", "N1"};
  StructuralLayout L;
  int dense = b.dense, residueTokens = b.tokens;
  for (int token = 0; token < residueTokens; ++token) {
    std::vector<int> live;
    for (int s = 0; s < dense; ++s) if (b.refMask[token * dense + s] != 0) live.push_back(s);
    if (live.empty()) continue;
    std::vector<std::string> names;
    for (int s : live) names.push_back(atomNameAt(b, token, s));
    int r = b.residueOfToken[token];
    std::string kind = "protein";
    if (r >= 0 && r < (int)b.chainOfResidue.size()) {
      int chain = (int)b.chainOfResidue[r];
      if (chain >= 0 && chain < (int)b.chainKinds.size()) kind = b.chainKinds[chain];
    }
    int restype = b.aatype[token];
    bool polymer = kind == "protein" ? restype <= 20 : (kind == "dna" || kind == "rna") ? (restype >= 22 && restype <= 30) : false;
    struct G { int role; std::vector<int> indices; std::vector<std::string> preference; };
    std::vector<G> groups;
    if (polymer && live.size() > 1) {
      const auto& set = kind == "protein" ? PROTEIN_BACKBONE : NUCLEIC_BACKBONE;
      std::vector<int> backbone, child;
      for (size_t i = 0; i < live.size(); ++i) (set.count(names[i]) ? backbone : child).push_back((int)i);
      int bbRole = kind == "protein" ? 1 : kind == "dna" ? 3 : 5, chRole = kind == "protein" ? 2 : kind == "dna" ? 4 : 6;
      const auto& bbCentre = kind == "protein" ? PBB : NBB;
      if (backbone.empty() || child.empty()) {
        std::vector<int> all;
        for (size_t i = 0; i < live.size(); ++i) all.push_back((int)i);
        groups.push_back({bbRole, all, bbCentre});
      } else {
        bool purine = restype == 22 || restype == 23 || restype == 26 || restype == 27;
        bool pyrimidine = restype == 24 || restype == 25 || restype == 28 || restype == 29;
        const auto& chCentre = kind == "protein" ? PSC : purine ? PUR : pyrimidine ? PYR : OTH;
        groups.push_back({bbRole, backbone, bbCentre});
        groups.push_back({chRole, child, chCentre});
      }
    } else {
      for (size_t i = 0; i < live.size(); ++i) groups.push_back({0, {(int)i}, {names[i]}});
    }
    for (auto& g : groups) {
      L.parent.push_back(token);
      L.role.push_back(g.role);
      std::vector<int> src;
      for (int i : g.indices) src.push_back(live[i]);
      L.sources.push_back(src);
      int chosen = 0;
      bool found = false;
      for (auto& wanted : g.preference) {
        for (size_t at = 0; at < g.indices.size(); ++at)
          if (names[g.indices[at]] == wanted) { chosen = (int)at; found = true; break; }
        if (found) break;
      }
      L.pseudoBetaSlot.push_back(chosen);
    }
  }
  int n = L.tokens = (int)L.parent.size();
  L.twin.assign(n, -1);
  std::map<int, std::vector<int>> members;
  for (int i = 0; i < n; ++i) members[L.parent[i]].push_back(i);
  for (auto& [p, g] : members) if (g.size() == 2) { L.twin[g[0]] = g[1]; L.twin[g[1]] = g[0]; }
  L.prevParent.assign(n, -1); L.nextParent.assign(n, -1);
  for (int i = 0; i < n; ++i) {
    int token = L.parent[i];
    if (token - 1 >= 0 && b.asymId[token - 1] == b.asymId[token]) L.prevParent[i] = token - 1;
    if (token + 1 < residueTokens && b.asymId[token + 1] == b.asymId[token]) L.nextParent[i] = token + 1;
  }
  L.residueAtomGather.assign(residueTokens * dense, -1);
  for (int i = 0; i < n; ++i)
    for (size_t at = 0; at < L.sources[i].size(); ++at) L.residueAtomGather[L.parent[i] * dense + L.sources[i][at]] = i * dense + (int)at;
  L.residueRepToken.assign(residueTokens, 0);
  std::vector<char> seen(residueTokens, 0);
  for (int i = 0; i < n; ++i) if (!seen[L.parent[i]]) { L.residueRepToken[L.parent[i]] = i; seen[L.parent[i]] = 1; }
  return L;
}

// structuralBatch's fields the exporter keeps (its `keep` list, in that order) as sbatch.*, then the layout and the
// expander's pair features
inline void addStructural(Entries& e, const Batch& b) {
  StructuralLayout L = structuralLayout(b);
  int dense = b.dense, n = L.tokens, size = n * dense;
  Batch s;
  s.tokens = n; s.dense = dense;
  s.refPos.assign(size * 3, 0); s.refMask.assign(size, 0); s.refElement.assign(size, 0); s.refCharge.assign(size, 0);
  s.refAtomNameChars.assign(size * 4, 0); s.refSpaceUid.assign(size, 0);
  for (int t = 0; t < n; ++t) {
    int from = L.parent[t];
    for (size_t at = 0; at < L.sources[t].size(); ++at) {
      int source = from * dense + L.sources[t][at], target = t * dense + (int)at;
      for (int k = 0; k < 3; ++k) s.refPos[target * 3 + k] = b.refPos[source * 3 + k];
      s.refMask[target] = b.refMask[source]; s.refElement[target] = b.refElement[source]; s.refCharge[target] = b.refCharge[source];
      for (int c = 0; c < 4; ++c) s.refAtomNameChars[target * 4 + c] = b.refAtomNameChars[source * 4 + c];
      s.refSpaceUid[target] = b.refSpaceUid[source];
    }
  }
  auto take = [&](const std::vector<int>& src) { std::vector<int> o(n); for (int i = 0; i < n; ++i) o[i] = src[L.parent[i]]; return o; };
  s.residueIndex = take(b.residueIndex);
  s.tokenIndex.resize(n);
  for (int i = 0; i < n; ++i) s.tokenIndex[i] = i + 1;
  s.asymId = take(b.asymId); s.entityId = take(b.entityId); s.symId = take(b.symId); s.aatype = take(b.aatype);
  s.residueOfToken = take(b.residueOfToken);
  s.seqMask.resize(n);
  for (int i = 0; i < n; ++i) s.seqMask[i] = b.seqMask[L.parent[i]];
  std::vector<int> realAtoms;
  for (int i = 0; i < size; ++i) if (s.refMask[i] != 0) realAtoms.push_back(i);
  atomGathers(s, realAtoms, L.pseudoBetaSlot, false, false);
  std::vector<float> bondMatrix((size_t)n * n, 0);
  if (b.hasBonds) {
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (b.bondMatrix[(size_t)L.parent[i] * b.tokens + L.parent[j]] != 0) bondMatrix[(size_t)i * n + j] = 1;
    bondMatrix[0] = 0;
  }
  const std::string p = "sbatch.";
  e.m(p + "tokens", n); e.m(p + "dense", dense); e.m(p + "subsets", s.subsets); e.m(p + "atomCount", s.atomCount);
  e.m(p + "shape.tokens", n); e.m(p + "shape.dense", dense); e.m(p + "shape.subsets", s.subsets);
  e.m(p + "shape.queries", QUERIES); e.m(p + "shape.keys", s.keys);
  e.i(p + "aatype", s.aatype); e.i(p + "residueIndex", s.residueIndex); e.i(p + "tokenIndex", s.tokenIndex);
  e.i(p + "asymId", s.asymId); e.i(p + "entityId", s.entityId); e.i(p + "symId", s.symId); e.t(p + "seqMask", s.seqMask);
  e.t(p + "refPos", s.refPos); e.t(p + "refMask", s.refMask); e.i(p + "refElement", s.refElement); e.t(p + "refCharge", s.refCharge);
  e.i(p + "refAtomNameChars", s.refAtomNameChars); e.i(p + "refSpaceUid", s.refSpaceUid); e.t(p + "predDenseAtomMask", s.refMask);
  e.t(p + "bondMatrix", bondMatrix); e.i(p + "residueOfToken", s.residueOfToken);
  addGather(e, p + "tokenAtomsToQueries", s.tokenAtomsToQueries);
  addGather(e, p + "queriesToTokenAtoms", s.queriesToTokenAtoms);
  addGather(e, p + "queriesToKeys", s.queriesToKeys);
  addGather(e, p + "tokensToQueries", s.tokensToQueries);
  addGather(e, p + "tokensToKeys", s.tokensToKeys);
  addGather(e, p + "tokenAtomsToPseudoBeta", s.tokenAtomsToPseudoBeta);
  e.i(p + "features.residueIndex", s.residueIndex); e.i(p + "features.tokenIndex", s.tokenIndex);
  e.i(p + "features.asymId", s.asymId); e.i(p + "features.entityId", s.entityId); e.i(p + "features.symId", s.symId);
  e.i("structural.parent", L.parent); e.i("structural.role", L.role);
  e.i("structural.residueAtomGather", L.residueAtomGather); e.i("structural.residueRepToken", L.residueRepToken);
  // structuralPairFeatures(layout, batch.asymId)
  std::vector<int> sameParent((size_t)n * n), twin((size_t)n * n), prevBackbone((size_t)n * n), nextBackbone((size_t)n * n), rolePair((size_t)n * n);
  auto bb = [&](int i) { return L.role[i] == 1 || L.role[i] == 3 || L.role[i] == 5; };
  auto sc = [&](int i) { return L.role[i] == 2; };
  auto base = [&](int i) { return L.role[i] == 4 || L.role[i] == 6; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      size_t at = (size_t)i * n + j;
      bool same = L.parent[i] == L.parent[j];
      sameParent[at] = same;
      bool sameChain = b.asymId[L.parent[i]] == b.asymId[L.parent[j]];
      twin[at] = same && ((bb(i) && (sc(j) || base(j))) || (bb(j) && (sc(i) || base(i))));
      prevBackbone[at] = bb(i) && bb(j) && sameChain && L.prevParent[i] == L.parent[j];
      nextBackbone[at] = bb(i) && bb(j) && sameChain && L.nextParent[i] == L.parent[j];
      int type = 7;
      if (bb(i) && bb(j)) type = 0;
      else if (bb(i) && sc(j)) type = 1;
      else if (sc(i) && bb(j)) type = 2;
      else if (sc(i) && sc(j)) type = 3;
      else if (bb(i) && base(j)) type = 4;
      else if (base(i) && bb(j)) type = 5;
      else if (base(i) && base(j)) type = 6;
      rolePair[at] = type;
    }
  e.i("structural.sameParent", sameParent); e.i("structural.twin", twin); e.i("structural.prevBackbone", prevBackbone);
  e.i("structural.nextBackbone", nextBackbone); e.i("structural.rolePairType", rolePair);
}

}  // namespace lf
