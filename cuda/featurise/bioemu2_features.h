// shared/bioemu2/featurise.js, natively: BioEmu2's own feature dict from an AlphaFold 3 job, byte for byte the
// JavaScript's (tools/check-native-bioemu2.mjs holds the two together). Every convention is documented there; this
// file only transcribes. Doubles where the JavaScript computes in doubles, a float rounding where it stores into a
// Float32Array, and jsmath's V8 atan where it calls Math.atan.
#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd.h"
#include "json.h"
#include <array>
#include <functional>
#include <tuple>
#include "jsmath.h"
#include "featurise.h"   // (standardComponentTexts, in tables.inc, which featurise.h includes inside lf)

namespace lf::bioemu2 {

constexpr int MAX_SEQS = 4092, MAX_PAIRED_SEQS = 8192, NUM_TOKENS = 33, GAP = 1, UNK_PROT = 22, WINDOW = 32;
inline int methodOf(const std::string& mode) {
  if (mode == "equilibrium") return 4;
  if (mode == "structure") return 7;
  throw std::runtime_error("BioEmu2 mode " + mode + ": want structure or equilibrium");
}
enum { CONTACT_UNSPECIFIED = 0, CONTACT_UNSELECTED = 1, CONTACT_CONTACT = 4 };
enum { MOL_PROTEIN = 0, MOL_DNA = 1, MOL_RNA = 2, MOL_LIGAND = 3 };

inline int protCode(char c) {
  static const std::string letters = "ARNDCEQGHILKMFPSTWYV";
  static const int codes[] = {2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  auto at = letters.find(c);
  if (at != std::string::npos) return codes[at];
  if (std::string("XJBZOU").find(c) != std::string::npos) return UNK_PROT;
  if (c == '-') return GAP;
  return -1;
}
inline int rnaCode(char c) { switch (c) { case 'A': return 23; case 'G': return 24; case 'C': return 25; case 'U': return 26; case 'N': return 27; } return -1; }
inline int dnaCode(char c) { switch (c) { case 'A': return 28; case 'G': return 29; case 'C': return 30; case 'T': return 31; case 'N': return 32; } return -1; }
inline std::string threeOf(char c) {
  static const std::map<char, std::string> t = {
      {'A', "ALA"}, {'R', "ARG"}, {'N', "ASN"}, {'D', "ASP"}, {'C', "CYS"}, {'E', "GLU"}, {'Q', "GLN"}, {'G', "GLY"},
      {'H', "HIS"}, {'I', "ILE"}, {'L', "LEU"}, {'K', "LYS"}, {'M', "MET"}, {'F', "PHE"}, {'P', "PRO"}, {'S', "SER"},
      {'T', "THR"}, {'W', "TRP"}, {'Y', "TYR"}, {'V', "VAL"}};
  auto it = t.find(c);
  return it == t.end() ? "" : it->second;
}
inline std::string unknownCcd(const std::string& kind) { return kind == "protein" ? "UNK" : kind == "dna" ? "DN" : "N"; }

struct Template {
  std::vector<std::string> names;
  std::vector<int> element;
  std::vector<double> charge;
  std::vector<float> coords;          // a Float32Array in the JavaScript
  std::vector<std::pair<int, int>> bonds;
};

inline int atomicNumber(const std::string& symbol) {
  std::string u = upper(symbol);
  if (u == "X") return 0;
  int z = elementNumber(u);
  if (z <= 0) throw std::runtime_error("BioEmu2: no atomic number for element " + symbol);
  return z;
}

inline bool finiteNumber(const std::string& v) {      // Number.isFinite(Number(v)) && v !== "?" && v !== "."
  if (v.empty() || v == "?" || v == ".") return false;
  char* end = nullptr;
  double d = std::strtod(v.c_str(), &end);
  return end && *end == '\0' && std::isfinite(d);
}

inline Template ccdTemplate(const std::string& text, const std::string& code, bool removeLeaving = false) {
  CifLoop loop = loopOf(text, "_chem_comp_atom");
  auto col = [&](const std::string& name) {
    auto it = std::find(loop.columns.begin(), loop.columns.end(), name);
    return it == loop.columns.end() ? -1 : (int)(it - loop.columns.begin());
  };
  auto get = [&](const std::vector<std::string>& row, const std::string& name) -> std::string {
    int at = col(name);
    return at < 0 || at >= (int)row.size() ? std::string() : row[at];
  };
  if (loop.rows.empty()) throw std::runtime_error(code + " has no atoms in the CCD");
  bool ideal = true;
  for (auto& row : loop.rows)
    for (const char* axis : {"x", "y", "z"}) ideal = ideal && finiteNumber(get(row, std::string("pdbx_model_Cartn_") + axis + "_ideal"));
  Template t;
  std::map<std::string, int> index;
  std::vector<std::array<double, 3>> xyz;
  for (auto& row : loop.rows) {
    std::string symbol = upper(get(row, "type_symbol"));
    if (symbol == "H" || symbol == "D") continue;
    std::string flag = get(row, "pdbx_leaving_atom_flag");
    if (removeLeaving && flag == "Y") continue;
    std::string name = get(row, "atom_id");
    index[name] = (int)t.names.size();
    t.names.push_back(name);
    std::string ts = get(row, "type_symbol");
    t.element.push_back(atomicNumber(ts.empty() ? "X" : ts));
    std::string q = get(row, "charge");
    t.charge.push_back(q.empty() || q == "?" || q == "." ? 0.0 : std::strtod(q.c_str(), nullptr));
    std::array<double, 3> p{};
    int k = 0;
    for (const char* axis : {"x", "y", "z"}) {
      std::string v = get(row, std::string(ideal ? "pdbx_model_Cartn_" : "model_Cartn_") + axis + (ideal ? "_ideal" : ""));
      if (!finiteNumber(v)) throw std::runtime_error(code + ": atom " + name + " has no usable coordinates");
      p[k++] = std::strtod(v.c_str(), nullptr);
    }
    xyz.push_back(p);
  }
  double mean[3];
  for (int a = 0; a < 3; ++a) {
    double s = 0;
    for (auto& p : xyz) s += p[a];
    mean[a] = s / (double)xyz.size();
  }
  for (auto& p : xyz)
    for (int a = 0; a < 3; ++a) t.coords.push_back((float)(p[a] - mean[a]));
  CifLoop bonds = loopOf(text, "_chem_comp_bond");
  auto b1 = std::find(bonds.columns.begin(), bonds.columns.end(), "atom_id_1") - bonds.columns.begin();
  auto b2 = std::find(bonds.columns.begin(), bonds.columns.end(), "atom_id_2") - bonds.columns.begin();
  for (auto& row : bonds.rows) {
    if ((size_t)b1 >= row.size() || (size_t)b2 >= row.size()) continue;
    auto a = index.find(row[b1]);
    auto b = index.find(row[b2]);
    if (a != index.end() && b != index.end()) t.bonds.push_back({a->second, b->second});
  }
  return t;
}

inline Template dropAtoms(const Template& t, const std::vector<std::string>& drop) {
  Template out;
  std::vector<int> map;
  for (size_t i = 0; i < t.names.size(); ++i) {
    if (std::find(drop.begin(), drop.end(), t.names[i]) != drop.end()) { map.push_back(-1); continue; }
    map.push_back((int)out.names.size());
    out.names.push_back(t.names[i]);
    out.element.push_back(t.element[i]);
    out.charge.push_back(t.charge[i]);
    for (int a = 0; a < 3; ++a) out.coords.push_back(t.coords[i * 3 + a]);
  }
  for (auto& [a, b] : t.bonds)
    if (map[a] >= 0 && map[b] >= 0) out.bonds.push_back({map[a], map[b]});
  return out;
}

inline bool isLeavingAtomDonor(const std::string& text, const std::vector<std::string>& bondAtomNames) {
  CifLoop loop = loopOf(text, "_chem_comp_atom");
  auto col = [&](const std::string& name) { return std::find(loop.columns.begin(), loop.columns.end(), name) - loop.columns.begin(); };
  std::set<std::string> leaving;
  for (auto& row : loop.rows) {
    std::string sym = upper(row[col("type_symbol")]);
    if (row[col("pdbx_leaving_atom_flag")] == "Y" && sym != "H" && sym != "D") leaving.insert(row[col("atom_id")]);
  }
  CifLoop bonds = loopOf(text, "_chem_comp_bond");
  auto b1 = std::find(bonds.columns.begin(), bonds.columns.end(), "atom_id_1") - bonds.columns.begin();
  auto b2 = std::find(bonds.columns.begin(), bonds.columns.end(), "atom_id_2") - bonds.columns.begin();
  std::set<std::string> names(bondAtomNames.begin(), bondAtomNames.end());
  for (auto& row : bonds.rows)
    if ((names.count(row[b1]) && leaving.count(row[b2])) || (names.count(row[b2]) && leaving.count(row[b1]))) return true;
  return false;
}

// a component's text: the standard table first, then the caller's dictionary (user CCD, a CCD file, the RCSB cache)
using CcdReader = std::function<std::string(const std::string&)>;
inline std::string componentText(const CcdReader& ccd, const std::string& code) {
  auto& standard = standardComponentTexts();
  auto it = standard.find(code);
  if (it != standard.end()) return it->second;
  return ccd(code);
}

struct Residue { Template t; int restype; bool perAtom; std::string centre; std::string label; };
struct Modification { std::string code; int position; };

inline std::vector<Residue> polymerResidues(const CcdReader& ccd, const std::string& kind, const std::string& sequence,
                                            const std::vector<Modification>& modifications) {
  bool hasMods = !modifications.empty();
  std::map<int, std::string> mods;
  for (auto& m : modifications) mods[m.position] = m.code;
  char unknownLetter = kind == "protein" ? 'X' : 'N';
  std::vector<Residue> out;
  for (size_t at = 0; at < sequence.size(); ++at) {
    int position = (int)at + 1;
    char letter = sequence[at];
    auto mod = mods.find(position);
    std::string code = mod == mods.end() ? "" : mod->second;
    bool haveCode = mod != mods.end();
    if (!haveCode && letter == unknownLetter) { code = unknownCcd(kind); haveCode = true; }
    if (haveCode && code != unknownCcd(kind)) {
      out.push_back({dropAtoms(ccdTemplate(componentText(ccd, code), code), {"OXT", "OP3"}), UNK_PROT, true, "", code});
      continue;
    }
    if (kind == "protein") {
      std::string ccdCode = haveCode || threeOf(letter).empty() ? "ALA" : threeOf(letter);
      Template t = dropAtoms(ccdTemplate(componentText(ccd, ccdCode), ccdCode), {"OXT"});
      int p = protCode(letter);
      int restype = !haveCode ? (p < 0 ? UNK_PROT : p) : UNK_PROT;
      out.push_back({t, restype, false, "CA", restype == UNK_PROT ? "UNK" : ccdCode});
    } else {
      std::string ccdCode = kind == "rna" ? std::string(1, letter) : "D" + std::string(1, letter);
      if (haveCode) ccdCode = unknownCcd(kind);
      Template t = dropAtoms(ccdTemplate(componentText(ccd, ccdCode), ccdCode), {"OP3"});
      bool unknown = haveCode || letter == 'N';
      int restype = kind == "rna" ? rnaCode(unknown ? 'N' : letter) : dnaCode(unknown ? 'N' : letter);
      if (restype < 0) throw std::runtime_error(std::string("BioEmu2: ") + letter + " is not a " + upper(kind) + " residue");
      std::string label = unknown ? (hasMods ? ccdCode : "UNK") : ccdCode;
      out.push_back({t, restype, false, "C1'", label});
    }
  }
  return out;
}

struct Chain {
  std::string id, kind, sequence, smiles;
  bool hasSmiles = false;
  std::vector<std::string> ccdCodes;
  std::vector<Modification> modifications;
  bool hasUnpaired = false, hasPaired = false;
  std::string unpairedMsa, pairedMsa;
};
struct AtomRef { std::string chain; int residue; std::string atom; };
struct Job { std::vector<Chain> chains; std::vector<std::pair<AtomRef, AtomRef>> bonds; };
struct ContactToken { std::string chain; int residue; bool hasAtom; std::string atom; };
struct Contact { ContactToken a, b; double distance; };

// bioemu2JobFromAf3Json
inline Job jobFromAf3Json(const Json& json) {
  Job job;
  auto text = [](const Json* v) { return Json::absent(v) ? std::string() : v->s; };
  const Json* sequences = json.get("sequences");
  if (sequences == nullptr || !sequences->isArray()) throw std::runtime_error("BioEmu2: the job has no sequences");
  for (auto& entry : sequences->a) {
    const std::string& kind = entry.o.at(0).first;
    const Json& e = entry.o.at(0).second;
    std::vector<std::string> ids;
    const Json* id = e.get("id");
    if (id && id->isArray()) for (auto& i : id->a) ids.push_back(i.s);
    else if (id) ids.push_back(id->s);
    for (auto& cid : ids) {
      Chain c;
      c.id = cid;
      c.kind = kind;
      if (kind == "protein" || kind == "rna" || kind == "dna") {
        c.sequence = text(e.get("sequence"));
        if (kind == "protein") {
          if (!Json::absent(e.get("unpairedMsa"))) { c.hasUnpaired = true; c.unpairedMsa = e.get("unpairedMsa")->s; }
          if (!Json::absent(e.get("pairedMsa"))) { c.hasPaired = true; c.pairedMsa = e.get("pairedMsa")->s; }
        }
        const Json* mods = e.get("modifications");
        if (!Json::absent(mods))
          for (auto& m : mods->a)
            c.modifications.push_back(kind == "protein"
                ? Modification{text(m.get("ptmType")), (int)m.get("ptmPosition")->n}
                : Modification{text(m.get("modificationType")), (int)m.get("basePosition")->n});
      } else if (kind == "ligand") {
        const Json* codes = e.get("ccdCodes");
        if (!Json::absent(codes) && codes->isArray()) for (auto& k : codes->a) c.ccdCodes.push_back(k.s);
        if (!Json::absent(e.get("smiles"))) { c.hasSmiles = true; c.smiles = e.get("smiles")->s; }
      } else {
        throw std::runtime_error("BioEmu2: a " + kind + " entry is not an input it takes");
      }
      job.chains.push_back(c);
    }
  }
  const Json* bonds = json.get("bondedAtomPairs");
  if (!Json::absent(bonds))
    for (auto& pair : bonds->a) {
      auto end = [](const Json& p) { return AtomRef{p.a.at(0).s, (int)p.a.at(1).n, p.a.at(2).s}; };
      job.bonds.push_back({end(pair.a.at(0)), end(pair.a.at(1))});
    }
  return job;
}

struct Features {
  int tokens = 0, atoms = 0, atomsPadded = 0, msaRows = 0;
  std::map<std::string, std::vector<int>> ints;     // the JavaScript's Int32Arrays
  std::map<std::string, std::vector<float>> floats; // ...and its Float32Arrays
  double dt = 0;
  int isDynamicsConditioned = 0;
  std::vector<std::tuple<std::string, int, std::string>> atomKeys;
  // the topology (featuriseBioemu2's `topology`)
  std::vector<std::string> names, labels;
  std::vector<int> element, residueIdx, molType, residueType, tokenIdxAtAtom, atomAsym;
  std::vector<std::pair<int, int>> bondIndex;
  std::map<int, std::string> ligandCcd;
};

namespace detail {
inline std::vector<std::string> a3mRows(const std::string& a3m) {
  std::vector<std::string> out;
  for (auto& line : splitLines(trimWs(a3m))) {
    std::string l = trimWs(line);
    if (l.empty() || l[0] == '>' || l[0] == '#') continue;
    out.push_back(l);
  }
  return out;
}
inline std::string queryOf(const std::string& row) {
  std::string out;
  for (char c : row) if (!(c >= 'a' && c <= 'z') && c != '-' && c != '.') out += c;
  return upper(out);
}
inline std::string noGaps(std::string s) { s.erase(std::remove(s.begin(), s.end(), '-'), s.end()); return s; }
struct Row { std::vector<int> res; std::map<int, int> dels; };
inline std::vector<Row> encodeRows(const std::vector<std::string>& rows) {
  std::vector<Row> out;
  for (auto& row : rows) {
    Row r;
    int count = 0;
    for (char c : row) {
      if (c != '-' && c >= 'a' && c <= 'z') { count += 1; continue; }
      if (count) { r.dels[(int)r.res.size()] = count; count = 0; }
      int code = protCode(c);
      if (code < 0) throw std::runtime_error(std::string("BioEmu2: ") + c + " is not an MSA residue");
      r.res.push_back(code);
    }
    out.push_back(r);
  }
  return out;
}
struct ChainInfo { Chain chain; std::string kind; std::string payload; std::vector<std::string> codes; std::vector<Residue> residues; int from, to; };
}  // namespace detail

inline void msaFeatures(Features& F, const std::vector<detail::ChainInfo>& info, int T) {
  using namespace detail;
  std::vector<int> protein;
  for (size_t i = 0; i < info.size(); ++i) if (info[i].kind == "protein") protein.push_back((int)i);
  std::map<int, std::string> seqs;
  for (int i : protein) seqs[i] = info[i].payload;
  auto isProtein = [&](int c) { return std::find(protein.begin(), protein.end(), c) != protein.end(); };
  auto unpairedRows = [&](int i) {
    std::vector<std::string> rows = info[i].chain.hasUnpaired && !info[i].chain.unpairedMsa.empty() ? a3mRows(info[i].chain.unpairedMsa)
                                                                                                    : std::vector<std::string>();
    if (rows.empty()) rows = {seqs[i]};
    if (queryOf(rows[0]) != upper(seqs[i])) throw std::runtime_error("MSA query of chain " + info[i].chain.id + " is not its sequence");
    return rows;
  };
  std::set<std::string> distinct;
  for (auto& [i, s] : seqs) distinct.insert(s);
  bool heteromeric = distinct.size() > 1;
  bool pairedGiven = heteromeric;
  for (int i : protein) pairedGiven = pairedGiven && info[i].chain.hasPaired && !info[i].chain.pairedMsa.empty();
  int nChains = (int)info.size();
  std::map<int, std::vector<Row>> msas;
  std::vector<std::vector<int>> pairing, isPaired;
  auto all = [&](int v) { return std::vector<int>(nChains, v); };
  if (pairedGiven) {
    std::map<int, std::vector<std::string>> pairedRows, unpaired;
    int depth = MAX_SEQS - 1;
    for (int i : protein) {
      auto rows = a3mRows(info[i].chain.pairedMsa);
      if ((int)rows.size() > MAX_PAIRED_SEQS + 1) rows.resize(MAX_PAIRED_SEQS + 1);
      std::vector<std::string> kept;
      for (size_t k = 1; k < rows.size(); ++k) if (!noGaps(rows[k]).empty()) kept.push_back(rows[k]);
      if ((int)kept.size() > MAX_PAIRED_SEQS) kept.resize(MAX_PAIRED_SEQS);
      pairedRows[i] = kept;
      auto u = unpairedRows(i);
      if ((int)u.size() > MAX_SEQS) u.resize(MAX_SEQS);
      unpaired[i] = u;
    }
    for (int i : protein) depth = std::min(depth, (int)pairedRows[i].size());
    for (int i : protein) {
      std::vector<std::string> combined = {seqs[i]};
      for (int k = 0; k < depth; ++k) combined.push_back(pairedRows[i][k]);
      auto& u = unpaired[i];
      int start = !u.empty() && upper(noGaps(u[0])) == upper(seqs[i]) ? 1 : 0;
      int take = MAX_SEQS - (int)combined.size();
      for (int k = start; k < (int)u.size() && k < start + take; ++k) combined.push_back(u[k]);
      msas[i] = encodeRows(combined);
    }
    pairing.push_back(all(0));
    isPaired.push_back(all(1));
    for (int k = 1; k <= depth; ++k) {
      std::vector<int> p(nChains), q(nChains);
      for (int c = 0; c < nChains; ++c) { p[c] = isProtein(c) ? k : -1; q[c] = isProtein(c) ? 1 : 0; }
      pairing.push_back(p);
      isPaired.push_back(q);
    }
    int offset = depth + 1;
    int maxUnpaired = INT32_MIN;
    for (int c : protein) maxUnpaired = std::max(maxUnpaired, (int)msas[c].size() - offset);
    for (int k = 0; k < maxUnpaired; ++k) {
      std::vector<int> p(nChains);
      for (int c = 0; c < nChains; ++c) p[c] = isProtein(c) && offset + k < (int)msas[c].size() ? offset + k : -1;
      pairing.push_back(p);
      isPaired.push_back(all(0));
      if ((int)pairing.size() >= MAX_SEQS) break;
    }
  } else {
    for (int i : protein) {
      auto rows = unpairedRows(i);
      if ((int)rows.size() > MAX_SEQS) rows.resize(MAX_SEQS);
      msas[i] = encodeRows(rows);
    }
    bool homomeric = distinct.size() == 1 && protein.size() > 1;
    int nSeqs = 1;
    if (!protein.empty()) { nSeqs = INT32_MAX; for (int c : protein) nSeqs = std::min(nSeqs, (int)msas[c].size()); }
    nSeqs = std::min(nSeqs, MAX_SEQS);
    for (int k = 0; k < nSeqs; ++k) {
      int flag = homomeric ? 1 : (k == 0 ? 1 : 0);
      isPaired.push_back(all(flag));
      std::vector<int> p(nChains);
      for (int c = 0; c < nChains; ++c) p[c] = isProtein(c) ? k : (k == 0 ? 0 : -1);
      pairing.push_back(p);
    }
  }
  for (int c = 0; c < nChains; ++c) {
    if (msas.count(c)) continue;
    Row r;
    for (auto& res : info[c].residues) r.res.push_back(res.restype);
    msas[c] = {r};
  }
  std::vector<int> resAtToken(T, 0);
  for (size_t a = 0; a < F.tokenIdxAtAtom.size(); ++a) resAtToken[F.tokenIdxAtAtom[a]] = F.residueIdx[a];
  std::vector<int> asym(T), local(T);
  for (int c = 0; c < nChains; ++c)
    for (int t = info[c].from; t < info[c].to; ++t) { asym[t] = c; local[t] = resAtToken[t] - resAtToken[info[c].from]; }
  int S = (int)pairing.size();
  std::vector<int> msa((size_t)S * T, GAP), paired((size_t)S * T, 0);
  std::vector<float> deletion((size_t)S * T, 0.f);
  for (int t = 0; t < T; ++t) {
    int c = asym[t], r = local[t];
    for (int s = 0; s < S; ++s) {
      paired[(size_t)s * T + t] = isPaired[s][c];
      int k = pairing[s][c];
      if (k != -1) {
        auto& row = msas[c][k];
        msa[(size_t)s * T + t] = row.res[r];
        auto d = row.dels.find(r);
        deletion[(size_t)s * T + t] = (float)(d == row.dels.end() ? 0 : d->second);
      }
    }
  }
  std::vector<float> value((size_t)S * T);
  float half = (float)(M_PI / 2);
  for (size_t i = 0; i < value.size(); ++i)
    value[i] = (float)((double)half * (double)(float)jsmath::atan((double)(float)((double)deletion[i] / 3)));
  std::vector<int> hasDeletion(value.size());
  for (size_t i = 0; i < value.size(); ++i) hasDeletion[i] = deletion[i] > 0 ? 1 : 0;
  std::vector<float> deletionMean(T), profile((size_t)T * NUM_TOKENS);
  for (int t = 0; t < T; ++t) {
    double sum = 0;
    std::vector<double> counts(NUM_TOKENS, 0.0);
    for (int s = 0; s < S; ++s) { sum += value[(size_t)s * T + t]; counts[msa[(size_t)s * T + t]] += 1; }
    deletionMean[t] = (float)(sum / S);
    for (int k = 0; k < NUM_TOKENS; ++k) profile[(size_t)t * NUM_TOKENS + k] = (float)(counts[k] / S);
  }
  F.ints["msa"] = msa;
  F.ints["msa_paired"] = paired;
  F.floats["deletion_value"] = value;
  F.ints["has_deletion"] = hasDeletion;
  F.floats["deletion_mean"] = deletionMean;
  F.floats["profile"] = profile;
  F.ints["msa_mask"] = std::vector<int>((size_t)S * T, 1);
  F.msaRows = S;
}

inline void contactFeatures(Features& F, const std::vector<Contact>& contacts, const std::vector<detail::ChainInfo>& info,
                            const std::vector<int>& centre, int T) {
  std::vector<int> resAtToken(T, 0), local(T, 0);
  for (size_t a = 0; a < F.tokenIdxAtAtom.size(); ++a) resAtToken[F.tokenIdxAtAtom[a]] = F.residueIdx[a];
  std::vector<std::string> names(T);
  for (int t = 0; t < T; ++t) names[t] = F.names[centre[t]];
  std::map<std::string, const detail::ChainInfo*> chainOf;
  for (auto& c : info) {
    chainOf[c.chain.id] = &c;
    for (int t = c.from; t < c.to; ++t) local[t] = resAtToken[t] - resAtToken[c.from];
  }
  auto describe = [](const ContactToken& t) {
    return "[\"" + t.chain + "\"," + std::to_string(t.residue) + "," + (t.hasAtom ? "\"" + t.atom + "\"" : "null") + "]";
  };
  auto resolve = [&](const ContactToken& token) {
    auto it = chainOf.find(token.chain);
    if (it == chainOf.end()) throw std::runtime_error("contact token " + describe(token) + ": no chain " + token.chain);
    const auto& c = *it->second;
    if (c.kind == "ccd" || c.kind == "smiles") {
      if (!token.hasAtom) throw std::runtime_error("contact token " + describe(token) + " addresses a ligand and must name an atom");
      if (token.residue != 1) throw std::runtime_error("contact token " + describe(token) + ": a ligand has one residue");
    } else {
      if (c.kind == "branched" && !token.hasAtom)
        throw std::runtime_error("contact token " + describe(token) + " addresses a branched entity and must name an atom");
      int length = c.kind == "branched" ? (int)c.codes.size() : (int)c.payload.size();
      if (token.residue > length)
        throw std::runtime_error("contact token " + describe(token) + ": res_id exceeds the chain length " + std::to_string(length));
    }
    std::vector<int> sel;
    for (int t = c.from; t < c.to; ++t) if (local[t] == token.residue - 1) sel.push_back(t);
    if (token.hasAtom) { std::vector<int> k; for (int t : sel) if (names[t] == token.atom) k.push_back(t); sel = k; }
    else if (sel.size() > 1) { std::vector<int> k; for (int t : sel) if (names[t] == "CA" || names[t] == "C1'") k.push_back(t); sel = k; }
    if (sel.size() != 1) throw std::runtime_error("contact token " + describe(token) + " does not match exactly one token");
    return sel[0];
  };
  std::vector<int> cls((size_t)T * T, CONTACT_UNSELECTED);
  std::vector<float> threshold((size_t)T * T, 0.f);
  for (auto& contact : contacts) {
    int i = resolve(contact.a), j = resolve(contact.b);
    cls[(size_t)i * T + j] = cls[(size_t)j * T + i] = CONTACT_CONTACT;
    threshold[(size_t)i * T + j] = threshold[(size_t)j * T + i] = (float)contact.distance;
  }
  std::vector<int> out((size_t)T * T * 5, 0);
  for (size_t p = 0; p < (size_t)T * T; ++p) out[p * 5 + cls[p]] = 1;
  F.ints["contact_conditioning"] = out;
  F.floats["contact_threshold"] = threshold;
}

// parseContacts
inline std::vector<Contact> parseContacts(const Json& raw) {
  std::vector<Contact> out;
  for (auto& c : raw.a) {
    for (auto& [k, v] : c.o)
      if (k != "token1" && k != "token2" && k != "max_distance") throw std::runtime_error("unknown contact field(s) " + k);
    auto token = [](const Json* t) {
      if (t == nullptr) throw std::runtime_error("a contact needs token1 and token2");
      for (auto& [k, v] : t->o)
        if (k != "asym_id" && k != "res_id" && k != "atom_name") throw std::runtime_error("unknown contact token field(s) in " + k);
      bool hasAtom = !Json::absent(t->get("atom_name"));
      bool hasRes = t->get("res_id") != nullptr;
      if (!hasAtom && !hasRes) throw std::runtime_error("res_id must be explicitly set when atom_name is None");
      int resId = Json::absent(t->get("res_id")) ? 1 : (int)t->get("res_id")->n;
      if (resId < 1) throw std::runtime_error("res_id is 1-based");
      return ContactToken{t->get("asym_id")->s, resId, hasAtom, hasAtom ? t->get("atom_name")->s : ""};
    };
    double d = Json::absent(c.get("max_distance")) ? 6.0 : c.get("max_distance")->n;
    if (!(d >= 4.0 && d <= 20.0)) throw std::runtime_error("max_distance is outside [4, 20] A, the range BioEmu2 was trained on");
    out.push_back({token(c.get("token1")), token(c.get("token2")), d});
  }
  return out;
}

// featuriseBioemu2. `smiles` is a SMILES chain's template, by chain id.
inline Features featurise(const Job& job, const CcdReader& ccd, const std::string& mode, const std::vector<Contact>* contacts,
                          const std::map<std::string, Template>& smiles) {
  using namespace detail;
  int method = methodOf(mode);
  Features F;
  std::set<std::string> bonded;
  for (auto& [a, b] : job.bonds) { bonded.insert(a.chain); bonded.insert(b.chain); }
  struct EntityGroup { std::string kind, payload; std::vector<std::string> codes; std::vector<Modification> mods; std::vector<Chain> chains; };
  std::vector<EntityGroup> entities;
  std::map<std::string, int> index;
  for (auto& chain : job.chains) {
    std::vector<std::pair<std::string, int>> mods;
    for (auto& m : chain.modifications) mods.push_back({m.code, m.position});
    std::sort(mods.begin(), mods.end());
    EntityGroup g;
    std::string key;
    if (chain.kind == "protein" || chain.kind == "rna" || chain.kind == "dna") {
      g.kind = chain.kind; g.payload = chain.sequence;
      for (auto& [c, p] : mods) g.mods.push_back({c, p});
      key = chain.kind + "\x1f" + chain.sequence;
      for (auto& [c, p] : mods) key += "\x1f" + c + "@" + std::to_string(p);
    } else if (chain.hasSmiles) {
      g.kind = "smiles"; g.payload = chain.smiles; key = "smiles\x1f" + chain.smiles;
    } else if (chain.ccdCodes.size() == 1 && !bonded.count(chain.id)) {
      g.kind = "ccd"; g.payload = chain.ccdCodes[0]; key = "ccd\x1f" + chain.ccdCodes[0];
    } else {
      g.kind = "branched"; g.codes = chain.ccdCodes; key = "branched";
      for (auto& c : chain.ccdCodes) key += "\x1f" + c;
    }
    auto it = index.find(key);
    if (it == index.end()) { index[key] = (int)entities.size(); entities.push_back(g); it = index.find(key); }
    entities[it->second].chains.push_back(chain);
  }
  std::vector<double> coords, charge;
  std::vector<int> restypeOfAtom;
  std::vector<int> tokAsym, tokEntity, tokSym, tokCentre;
  std::vector<ChainInfo> info;
  int nAtoms = 0, nTokens = 0, nRes = 0, chainCount = 0;
  for (size_t entityId = 0; entityId < entities.size(); ++entityId) {
    auto& g = entities[entityId];
    for (size_t symId = 0; symId < g.chains.size(); ++symId) {
      const Chain& chain = g.chains[symId];
      chainCount += 1;
      std::vector<Residue> residues;
      int mol;
      if (g.kind == "protein" || g.kind == "rna" || g.kind == "dna") {
        residues = polymerResidues(ccd, g.kind, g.payload, g.mods);
        mol = g.kind == "protein" ? MOL_PROTEIN : g.kind == "dna" ? MOL_DNA : MOL_RNA;
      } else if (g.kind == "ccd") {
        residues = {{ccdTemplate(componentText(ccd, g.payload), g.payload), UNK_PROT, true, "", g.payload}};
        mol = MOL_LIGAND;
        F.ligandCcd[chainCount - 1] = g.payload;
      } else if (g.kind == "smiles") {
        auto it = smiles.find(chain.id);
        if (it == smiles.end()) throw std::runtime_error("BioEmu2: SMILES chain " + chain.id + " needs a template");
        residues = {{it->second, UNK_PROT, true, "", "UNK"}};
        mol = MOL_LIGAND;
      } else {
        std::set<std::string> entityIds;
        for (auto& c : g.chains) entityIds.insert(c.id);
        std::map<int, std::vector<std::string>> atomsByRes;
        for (auto& [a, b] : job.bonds)
          for (const AtomRef* end : {&a, &b})
            if (entityIds.count(end->chain)) atomsByRes[end->residue - 1].push_back(end->atom);
        for (size_t i = 0; i < g.codes.size(); ++i) {
          std::string text = componentText(ccd, g.codes[i]);
          bool remove = atomsByRes.count((int)i) ? isLeavingAtomDonor(text, atomsByRes[(int)i]) : false;
          residues.push_back({ccdTemplate(text, g.codes[i], remove), UNK_PROT, true, "", g.codes[i]});
        }
        mol = MOL_LIGAND;
      }
      int chainTokenStart = nTokens;
      for (size_t rLocal = 0; rLocal < residues.size(); ++rLocal) {
        const Residue& res = residues[rLocal];
        const Template& t = res.t;
        int n = (int)t.names.size();
        for (int i = 0; i < n; ++i) {
          F.labels.push_back(res.label);
          for (int a = 0; a < 3; ++a) coords.push_back(t.coords[i * 3 + a]);
          F.element.push_back(t.element[i]);
          charge.push_back(t.charge[i]);
          F.names.push_back(t.names[i]);
          F.residueIdx.push_back(nRes);
          F.molType.push_back(mol);
          F.residueType.push_back(res.restype);
          F.tokenIdxAtAtom.push_back(res.perAtom ? nTokens + i : nTokens);
          F.atomKeys.push_back({chain.id, (int)rLocal + 1, t.names[i]});
        }
        if (res.perAtom) for (auto& [a, b] : t.bonds) F.bondIndex.push_back({a + nAtoms, b + nAtoms});
        if (res.perAtom) {
          for (int i = 0; i < n; ++i) tokCentre.push_back(nAtoms + i);
          nTokens += n;
        } else {
          auto it = std::find(t.names.begin(), t.names.end(), res.centre);
          if (it == t.names.end()) throw std::runtime_error("BioEmu2: " + res.label + " has no " + res.centre);
          tokCentre.push_back(nAtoms + (int)(it - t.names.begin()));
          nTokens += 1;
        }
        nAtoms += n;
        nRes += 1;
      }
      for (int i = 0; i < nTokens - chainTokenStart; ++i) {
        tokAsym.push_back(chainCount - 1);
        tokEntity.push_back((int)entityId);
        tokSym.push_back((int)symId);
      }
      info.push_back({chain, g.kind, g.payload, g.codes, residues, chainTokenStart, nTokens});
    }
  }
  for (size_t a = 0; a < F.tokenIdxAtAtom.size(); ++a) F.atomAsym.push_back(tokAsym[F.tokenIdxAtAtom[a]]);
  // declared covalent bonds
  std::map<std::string, int> keyIndex;
  for (size_t i = 0; i < F.atomKeys.size(); ++i) {
    auto& [c, r, n] = F.atomKeys[i];
    keyIndex[c + "\x1f" + std::to_string(r) + "\x1f" + n] = (int)i;
  }
  for (auto& [a, b] : job.bonds) {
    auto ia = keyIndex.find(a.chain + "\x1f" + std::to_string(a.residue) + "\x1f" + a.atom);
    auto ib = keyIndex.find(b.chain + "\x1f" + std::to_string(b.residue) + "\x1f" + b.atom);
    if (ia == keyIndex.end() || ib == keyIndex.end()) throw std::runtime_error("BioEmu2: a declared bond names an atom that is not there");
    F.bondIndex.push_back({ia->second, ib->second});
  }
  int T = nTokens;
  F.tokens = T;
  F.atoms = nAtoms;
  std::vector<float> tokenBonds((size_t)T * T, 0.f);
  for (auto& [a, b] : F.bondIndex) {
    int i = F.tokenIdxAtAtom[a], j = F.tokenIdxAtAtom[b];
    tokenBonds[(size_t)i * T + j] = tokenBonds[(size_t)j * T + i] = 1.f;
  }
  std::vector<int> resType(T), resIndex(T);
  for (int t = 0; t < T; ++t) { resType[t] = F.residueType[tokCentre[t]]; resIndex[t] = F.residueIdx[tokCentre[t]]; }
  F.ints["res_type"] = resType;
  F.ints["residue_index"] = resIndex;
  F.ints["asym_id"] = tokAsym;
  F.ints["entity_id"] = tokEntity;
  F.ints["sym_id"] = tokSym;
  F.ints["token_center_atom_index"] = tokCentre;
  F.ints["token_pad_mask"] = std::vector<int>(T, 1);
  F.floats["token_bonds"] = tokenBonds;
  F.ints["method_feature"] = std::vector<int>(T, method);
  F.ints["pocket_feature"] = std::vector<int>(T, 0);
  std::vector<int> cc((size_t)T * T * 5, 0);
  for (size_t p = 0; p < (size_t)T * T; ++p) cc[p * 5] = 1;
  F.ints["contact_conditioning"] = cc;
  F.floats["contact_threshold"] = std::vector<float>((size_t)T * T, 0.f);
  msaFeatures(F, info, T);
  if (contacts) contactFeatures(F, *contacts, info, tokCentre, T);
  int nPad = (nAtoms + WINDOW - 1) / WINDOW * WINDOW;
  F.atomsPadded = nPad;
  auto padInts = [&](const std::vector<int>& v, int width, int fill) {
    std::vector<int> out((size_t)nPad * width, fill);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
  };
  auto padFloats = [&](const std::vector<double>& v, int width) {
    std::vector<float> out((size_t)nPad * width, 0.f);
    for (size_t i = 0; i < v.size(); ++i) out[i] = (float)v[i];
    return out;
  };
  std::vector<int> nameEncoding;
  for (auto& name : F.names)
    for (int i = 0; i < 4; ++i) nameEncoding.push_back(i < (int)name.size() ? (int)(unsigned char)name[i] - 32 : 0);
  F.floats["template_coords"] = padFloats(coords, 3);
  F.ints["element"] = padInts(F.element, 1, 0);
  F.floats["charge"] = padFloats(charge, 1);
  F.ints["name_encoding"] = padInts(nameEncoding, 4, 0);
  F.ints["token_idx_at_atom"] = padInts(F.tokenIdxAtAtom, 1, -1);
  F.ints["ref_space_uid"] = padInts(F.residueIdx, 1, -1);
  F.ints["atom_pad_mask"] = padInts(std::vector<int>(nAtoms, 1), 1, 0);
  F.ints["atom_mask"] = padInts(std::vector<int>(nAtoms, 1), 1, 0);
  F.floats["coords_cond"] = std::vector<float>((size_t)nPad * 3, 0.f);
  return F;
}

// addPropagation: the frame centred, dt in ps (a float32, as Math.fround), and the flag
inline void addPropagation(Features& F, const std::vector<double>& x0, double lagNs) {
  if (!(lagNs > 0 && std::isfinite(lagNs))) throw std::runtime_error("the lag must be finite and positive");
  int n = (int)x0.size() / 3;
  double mean[3] = {0, 0, 0};
  for (int a = 0; a < 3; ++a) { double s = 0; for (int i = 0; i < n; ++i) s += x0[i * 3 + a]; mean[a] = s / n; }
  std::vector<float> cond((size_t)F.atomsPadded * 3, 0.f);
  for (int i = 0; i < n; ++i) for (int a = 0; a < 3; ++a) cond[i * 3 + a] = (float)(x0[i * 3 + a] - mean[a]);
  F.floats["coords_cond"] = cond;
  F.dt = (double)(float)(lagNs * 1000);
  F.isDynamicsConditioned = 1;
}

}  // namespace lf::bioemu2
