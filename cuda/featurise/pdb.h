// The PDB the native writer fills: shared/af3/structure/pdb.js toPdb over a batch, with each atom's dense slot
// as its x coordinate (the exporter's template.pdb), so the native writer knows which coordinates go where.
#pragma once
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "featurise.h"

namespace lf {

inline std::string padStart(const std::string& s, size_t n) { return s.size() >= n ? s : std::string(n - s.size(), ' ') + s; }
inline std::string padEnd(const std::string& s, size_t n) { return s.size() >= n ? s : s + std::string(n - s.size(), ' '); }
// Number.prototype.toFixed: the double's EXACT decimal value rounded at `digits`, a tie going up (away from zero in
// magnitude, which is what "the larger n" means for the absolute value) - printf rounds a tie to even
inline std::string toFixed(double v, int digits) {
  if (!std::isfinite(v)) return std::isnan(v) ? "NaN" : (v > 0 ? "Infinity" : "-Infinity");
  bool negative = v < 0;
  char buf[512];
  snprintf(buf, sizeof buf, "%.80f", std::fabs(v));     // glibc prints the exact binary value
  std::string s(buf);
  size_t dot = s.find('.');
  std::string whole = s.substr(0, dot), frac = s.substr(dot + 1);
  bool up = frac[digits] >= '5';
  std::string kept = whole + frac.substr(0, digits);
  if (up) {
    int i = (int)kept.size() - 1;
    while (i >= 0) { if (kept[i] == '9') { kept[i] = '0'; --i; } else { ++kept[i]; break; } }
    if (i < 0) kept = "1" + kept;
  }
  std::string intPart = kept.substr(0, kept.size() - digits), fracPart = kept.substr(kept.size() - digits);
  std::string out = intPart + (digits > 0 ? "." + fracPart : "");
  return (negative ? "-" : "") + out;     // (-0.0001).toFixed(3) is "-0.000" in JavaScript, and (-0).toFixed(3) "0.000"
}

// shared/af3/structure/pdb.js linkLine: the LINK record joining two atom records (or their heads) - name 13-16,
// residue name 18-20, chain 22, residue number 23-26, the second end from column 43
inline std::string linkLine(const std::string& a, const std::string& b) {
  auto end = [](const std::string& r) { return r.substr(12, 4) + " " + r.substr(17, 3) + " " + r.substr(21, 5) + " "; };
  return "LINK        " + end(a) + std::string(15, ' ') + end(b);
}

// shared/af3/structure/pdb.js pdbLinkRecords: LINK records for declared bonds over a PDB's own atom records, each end
// (chain letter, residue, atom) as the records spell it - BioEmu2's writer, which has no batch
struct LinkEnd { std::string chain; int residue; std::string atom; };
inline std::vector<std::string> pdbLinkRecords(const std::vector<std::string>& records,
                                               const std::vector<std::pair<LinkEnd, LinkEnd>>& bonds) {
  std::vector<std::string> lines;
  if (bonds.empty()) return lines;
  std::map<std::string, std::string> byAtom;
  for (auto& r : records) {
    if (r.rfind("ATOM", 0) != 0 && r.rfind("HETATM", 0) != 0) continue;
    std::string key = r.substr(21, 1) + ":" + std::to_string(std::stoi(r.substr(22, 4))) + ":" + trimWs(r.substr(12, 4));
    byAtom.emplace(key, r);
  }
  for (auto& [from, to] : bonds) {
    auto a = byAtom.find(from.chain + ":" + std::to_string(from.residue) + ":" + from.atom);
    auto c = byAtom.find(to.chain + ":" + std::to_string(to.residue) + ":" + to.atom);
    if (a != byAtom.end() && c != byAtom.end()) lines.push_back(linkLine(a->second, c->second));
  }
  return lines;
}

// toPdb's options.links: a LINK record per declared bond both of whose ends name an atom the batch wrote, keyed by
// (asymId - 1, residue_index, atom name) as shared/af3/structure/pdb.js linkRecords keys them; first, before the atoms
inline std::string pdbText(const Batch& b, const std::vector<float>& positions,
                           const std::vector<DeclaredBond>* links = nullptr) {
  static const std::map<char, std::string> THREE = {{'A', "ALA"}, {'R', "ARG"}, {'N', "ASN"}, {'D', "ASP"}, {'C', "CYS"},
    {'Q', "GLN"}, {'E', "GLU"}, {'G', "GLY"}, {'H', "HIS"}, {'I', "ILE"}, {'L', "LEU"}, {'K', "LYS"}, {'M', "MET"},
    {'F', "PHE"}, {'P', "PRO"}, {'S', "SER"}, {'T', "THR"}, {'W', "TRP"}, {'Y', "TYR"}, {'V', "VAL"}};
  static const std::string CHAINS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const auto& symbols = elementSymbols();
  std::map<int, std::string> componentOf;
  for (auto& s : b.ligandSpans) for (int k = 0; k < s.count; ++k) componentOf[s.from + k] = s.code;
  for (auto& s : b.modifiedSpans) for (int k = 0; k < s.count; ++k) componentOf[s.from + k] = s.code;
  auto atomName = [&](int slot) {
    std::string name;
    for (int c = 0; c < 4; ++c) { int code = b.displayAtomNameChars[slot * 4 + c]; if (code > 0) name += (char)(code + 32); }
    return trimWs(name);
  };
  std::vector<std::string> lines;
  std::map<std::string, std::string> headOf;     // (asym, residue, atom) -> its record's head, for the LINK records
  std::map<int, int> serialOfToken;
  int serial = 1;
  for (int token = 0; token < b.tokens; ++token) {
    auto ligand = componentOf.find(token);
    for (int atom = 0; atom < b.dense; ++atom) {
      int slot = token * b.dense + atom;
      if (!b.predDenseAtomMask[slot]) continue;
      if (ligand != componentOf.end() && atom == 0) serialOfToken[token] = serial;
      std::string name = atomName(slot);
      int residue = b.residueOfToken[token];
      std::string resName;
      if (ligand != componentOf.end()) resName = ligand->second;
      else {
        const std::string* kind = residue >= 0 ? &b.chainKinds[(int)b.chainOfResidue[residue]] : nullptr;
        char code = residue >= 0 && residue < (int)b.sequence.size() ? b.sequence[residue] : 0;
        if (kind && (*kind == "dna" || *kind == "rna") && code) resName = padStart(*kind == "dna" ? std::string("D") + code : std::string(1, code), 3);
        else {
          auto it = residue >= 0 && residue < (int)b.sequence.size() ? THREE.find(b.sequence[residue]) : THREE.end();
          resName = it == THREE.end() ? "UNK" : it->second;
        }
      }
      int element = b.refElement[slot];
      std::string symbol = element >= 1 && element <= (int)symbols.size() ? symbols[element - 1] : "C";
      std::string head = std::string(ligand == componentOf.end() ? "ATOM  " : "HETATM") + padStart(std::to_string(serial % 100000), 5) + " "
        + (name.size() < 4 ? padEnd(" " + name, 4) : name.substr(0, 4)) + " " + padEnd(resName, 3).substr(0, 3) + " "   // three columns (shared/af3/structure/pdb.js)
        + CHAINS[(b.asymId[token] - 1) % (int)CHAINS.size()] + padStart(std::to_string(b.residueIndex[token]), 4) + "    ";
      if (links) headOf.emplace(std::to_string(b.asymId[token] - 1) + ":" + std::to_string(b.residueIndex[token]) + ":" + name, head);
      lines.push_back(head + padStart(toFixed(positions[slot * 3], 3), 8) + padStart(toFixed(positions[slot * 3 + 1], 3), 8)
                      + padStart(toFixed(positions[slot * 3 + 2], 3), 8)
                      + "  1.00" + padStart(toFixed(0, 2), 6) + "          " + padStart(symbol, 2));
      ++serial;
    }
    if (token + 1 < b.tokens && b.asymId[token + 1] != b.asymId[token]) lines.push_back("TER");
  }
  std::map<int, std::vector<int>> partners;
  auto bondsOf = [&](int from, const std::vector<CompBond>& bonds) {
    for (auto& bd : bonds) {
      auto a = serialOfToken.find(from + bd.from), c = serialOfToken.find(from + bd.to);
      if (a == serialOfToken.end() || c == serialOfToken.end()) continue;
      partners[a->second].push_back(c->second);
      partners[c->second].push_back(a->second);
    }
  };
  for (auto& s : b.ligandSpans) bondsOf(s.from, s.bonds);
  for (auto& s : b.modifiedSpans) bondsOf(s.from, s.bonds);
  for (auto& [atom, bonded] : partners)
    for (size_t start = 0; start < bonded.size(); start += 4) {
      // (a serial is five digits: past 99,999 atoms it wraps - as every serial field here - so the columns stay where
      // readers look; mmCIF holds such a fold whole)
      std::string line = "CONECT" + padStart(std::to_string(atom % 100000), 5);
      for (size_t k = start; k < std::min(bonded.size(), start + 4); ++k) line += padStart(std::to_string(bonded[k] % 100000), 5);
      lines.push_back(line);
    }
  lines.push_back("END");
  if (links) {
    std::vector<std::string> linkLines;
    for (auto& bond : *links) {
      if (!bond.from.hasAtom || bond.from.atom.empty() || !bond.to.hasAtom || bond.to.atom.empty()) continue;
      auto a = headOf.find(std::to_string(bond.from.asym) + ":" + std::to_string(bond.from.residue) + ":" + bond.from.atom);
      auto c = headOf.find(std::to_string(bond.to.asym) + ":" + std::to_string(bond.to.residue) + ":" + bond.to.atom);
      if (a != headOf.end() && c != headOf.end()) linkLines.push_back(linkLine(a->second, c->second));
    }
    lines.insert(lines.begin(), linkLines.begin(), linkLines.end());
  }
  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) out += (i ? "\n" : "") + lines[i];
  return out + "\n";
}

// the exporter's template.pdb: each atom's dense slot as its x coordinate
inline std::string pdbTemplateText(const Batch& b, const std::vector<DeclaredBond>* links = nullptr) {
  std::vector<float> slots((size_t)b.tokens * b.dense * 3, 0.0f);
  for (size_t i = 0; i < (size_t)b.tokens * b.dense; ++i) slots[i * 3] = (float)i;
  return pdbText(b, slots, links);
}

}  // namespace lf
