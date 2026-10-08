// The exporters' output: <dir>/model.bin (raw little-endian 4-byte elements) and <dir>/model.idx, one line an
// entry - `t <name> <offset> <length>` float32, `i ...` int32, `m <name> <value>` a number - written in the order
// cuda/af3/export-model.mjs's walk writes them (its `add`: a typed array always, a plain number array as float32 -
// an EMPTY array of anything as a float32 entry of length 0 - a number with JavaScript's String(), a boolean 0/1,
// strings skipped, an object's keys in their insertion order). The index goes last and atomically, since a binary
// may be waiting for it.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "featurise.h"
#include "jsnum.h"

namespace lf {

struct Entries {
  struct E { char kind; std::string name; std::string value; std::vector<uint32_t> words; };
  std::vector<E> list;
  void m(const std::string& name, double v) { list.push_back({'m', name, jsNumber(v), {}}); }
  void flag(const std::string& name, bool v) { list.push_back({'m', name, v ? "1" : "0", {}}); }
  void text(const std::string& name, const std::string& v) { list.push_back({'m', name, v, {}}); }   // AF2's meta/model
  void t(const std::string& name, const std::vector<float>& v) {
    E e{'t', name, "", std::vector<uint32_t>(v.size())};
    if (!v.empty()) std::memcpy(e.words.data(), v.data(), v.size() * 4);
    list.push_back(std::move(e));
  }
  void t(const std::string& name, const std::vector<double>& v) {   // a JS number array: Float32Array.from
    std::vector<float> f(v.begin(), v.end());
    t(name, f);
  }
  void i(const std::string& name, const std::vector<int>& v) {
    E e{'i', name, "", std::vector<uint32_t>(v.size())};
    if (!v.empty()) std::memcpy(e.words.data(), v.data(), v.size() * 4);
    list.push_back(std::move(e));
  }
  void empty(const std::string& name) { list.push_back({'t', name, "", {}}); }   // [] -> a float32 entry of length 0

  void write(const std::string& dir) const {
    if (system(("mkdir -p '" + dir + "'").c_str()) != 0) throw std::runtime_error("cannot create " + dir);
    FILE* bin = fopen((dir + "/model.bin").c_str(), "wb");
    if (!bin) throw std::runtime_error("cannot write " + dir + "/model.bin");
    std::string idx;
    size_t offset = 0;
    for (auto& e : list) {
      if (e.kind == 'm') { idx += "m " + e.name + " " + e.value + "\n"; continue; }
      idx += std::string(1, e.kind) + " " + e.name + " " + std::to_string(offset) + " " + std::to_string(e.words.size()) + "\n";
      if (!e.words.empty() && fwrite(e.words.data(), 4, e.words.size(), bin) != e.words.size())
        throw std::runtime_error("short write to " + dir + "/model.bin");
      offset += e.words.size();
    }
    fclose(bin);
    std::string tmp = dir + "/model.idx.tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    fwrite(idx.data(), 1, idx.size(), f);
    fclose(f);
    if (rename(tmp.c_str(), (dir + "/model.idx").c_str()) != 0) throw std::runtime_error("cannot rename " + tmp);
  }
  size_t words() const { size_t n = 0; for (auto& e : list) n += e.words.size(); return n; }
};

inline void addGather(Entries& e, const std::string& name, const Gather& g) {
  e.i(name + ".indices", g.indices);
  e.t(name + ".mask", g.mask);
  e.m(name + ".count", g.count);
}

inline void addBonds(Entries& e, const std::string& name, const std::vector<CompBond>& bonds) {
  if (bonds.empty()) { e.empty(name); return; }
  for (size_t j = 0; j < bonds.size(); ++j) {
    std::string p = name + "." + std::to_string(j);
    e.m(p + ".from", bonds[j].from); e.m(p + ".to", bonds[j].to); e.m(p + ".order", bonds[j].order);
  }
}

// featuriseProtein's returned object, walked (`withBonds` false: chai-1's `delete batch.bondMatrix`)
inline void addBatch(Entries& e, const Batch& b, bool withBondMatrix = true) {
  const std::string p = "batch.";
  e.t(p + "chainLengths", b.chainLengths);
  e.m(p + "tokens", b.tokens); e.m(p + "dense", b.dense); e.m(p + "subsets", b.subsets);
  e.m(p + "atomCount", b.atomCount); e.m(p + "sequences", b.sequences);
  e.m(p + "shape.tokens", b.tokens); e.m(p + "shape.dense", b.dense); e.m(p + "shape.subsets", b.subsets);
  e.m(p + "shape.queries", QUERIES); e.m(p + "shape.keys", b.keys);
  e.i(p + "aatype", b.aatype); e.t(p + "profile", b.profile); e.t(p + "deletionMean", b.deletionMean);
  e.i(p + "msa", b.msa); e.t(p + "msaMask", b.msaMask); e.t(p + "deletionMatrix", b.deletionMatrix);
  e.i(p + "residueIndex", b.residueIndex); e.i(p + "tokenIndex", b.tokenIndex); e.i(p + "asymId", b.asymId);
  e.i(p + "entityId", b.entityId); e.i(p + "symId", b.symId); e.t(p + "seqMask", b.seqMask);
  e.t(p + "refPos", b.refPos); e.t(p + "refMask", b.refMask); e.i(p + "refElement", b.refElement);
  e.t(p + "refCharge", b.refCharge); e.i(p + "refAtomNameChars", b.refAtomNameChars); e.i(p + "refSpaceUid", b.refSpaceUid);
  e.i(p + "displayAtomNameChars", b.displayAtomNameChars);
  e.t(p + "predDenseAtomMask", b.refMask);
  if (b.hasBonds) {
    if (withBondMatrix) e.t(p + "bondMatrix", b.bondMatrix);
    e.t(p + "bondOrderMatrix", b.bondOrderMatrix);
  }
  if (b.ligandSpans.empty()) e.empty(p + "ligandSpans");
  for (size_t k = 0; k < b.ligandSpans.size(); ++k) {
    const auto& s = b.ligandSpans[k];
    std::string q = p + "ligandSpans." + std::to_string(k);
    e.m(q + ".from", s.from); e.m(q + ".count", s.count);
    addBonds(e, q + ".bonds", s.bonds);
  }
  if (b.modifiedSpans.empty()) e.empty(p + "modifiedSpans");
  for (size_t k = 0; k < b.modifiedSpans.size(); ++k) {
    const auto& s = b.modifiedSpans[k];
    std::string q = p + "modifiedSpans." + std::to_string(k);
    e.m(q + ".from", s.from); e.m(q + ".count", s.count); e.m(q + ".residue", s.residue);
    if (s.atoms.empty()) e.empty(q + ".atoms");
    for (size_t j = 0; j < s.atoms.size(); ++j) {
      const auto& a = s.atoms[j];
      std::string r = q + ".atoms." + std::to_string(j);
      e.m(r + ".element", a.element); e.m(r + ".charge", a.charge); e.flag(r + ".leaving", a.leaving);
      e.m(r + ".x", a.x); e.m(r + ".y", a.y); e.m(r + ".z", a.z);
      if (a.componentSlot >= 0) e.m(r + ".componentSlot", a.componentSlot);
    }
    addBonds(e, q + ".bonds", s.bonds);
    if (s.oneToken) e.flag(q + ".oneToken", true);
  }
  e.i(p + "residueOfToken", b.residueOfToken);
  e.t(p + "chainOfResidue", b.chainOfResidue);
  e.i(p + "isDna", b.isDna); e.i(p + "isRna", b.isRna); e.i(p + "isLigand", b.isLigand); e.i(p + "isModified", b.isModified);
  addGather(e, p + "tokenAtomsToQueries", b.tokenAtomsToQueries);
  addGather(e, p + "queriesToKeys", b.queriesToKeys);
  addGather(e, p + "queriesToTokenAtoms", b.queriesToTokenAtoms);
  addGather(e, p + "tokensToQueries", b.tokensToQueries);
  addGather(e, p + "tokensToKeys", b.tokensToKeys);
  addGather(e, p + "tokenAtomsToPseudoBeta", b.tokenAtomsToPseudoBeta);
  e.i(p + "features.residueIndex", b.residueIndex); e.i(p + "features.tokenIndex", b.tokenIndex);
  e.i(p + "features.asymId", b.asymId); e.i(p + "features.entityId", b.entityId); e.i(p + "features.symId", b.symId);
}

}  // namespace lf
