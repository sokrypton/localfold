// The fold's files: the structure (through the featuriser's template.pdb, or as mmCIF), AlphaFold 3's confidence
// files beside it, and the ranking score's structure terms. Host code: cuda/af3/src/scores.cuh (the clash and disorder
// terms and the mmCIF writer) is the same file, read here against a shim that serves the input's host bytes; the PDB
// and confidence writers are cuda/af3/src/sampler.cuh's.
#include "af3.h"
#include "output.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace af3out {
struct HostModel {        // what scores.cuh reads off `M`: the input's bytes on the host
  const int* i(const std::string& k) const { return mt::M.hostI(k); }
  const float* f(const std::string& k) const { return mt::M.hostF(k); }
  double meta(const std::string& k) const { return mt::M.meta(k); }
};
HostModel M;
std::string DATA_DIR;
std::vector<size_t> writePdb(const std::string& path, const std::vector<float>& x, const float* bfactors);
#include "../../cuda/af3/src/scores.cuh"

// The PDB: through the featuriser's template.pdb (the page's own records, the slot in the x column), else one residue a
// token named here. Returns the dense slot of every ATOM/HETATM record, in file order.
std::vector<size_t> writePdb(const std::string& path, const std::vector<float>& x, const float* bfactors) {
  std::vector<size_t> order;
  std::ifstream tf(DATA_DIR + "/template.pdb");
  if (tf) {
    FILE* f = fopen(path.c_str(), "w");
    std::string line;
    while (std::getline(tf, line)) {
      if (line.size() >= 66 && (!line.compare(0, 4, "ATOM") || !line.compare(0, 6, "HETATM"))) {
        size_t slot = (size_t)std::lround(std::stod(line.substr(30, 8)));
        order.push_back(slot);
        char coords[64];
        snprintf(coords, sizeof coords, "%8.3f%8.3f%8.3f%6.2f%6.2f", x[slot * 3], x[slot * 3 + 1], x[slot * 3 + 2], 1.0,
                 bfactors ? bfactors[slot] : 0.0);
        line = line.substr(0, 30) + coords + line.substr(66);
      }
      fprintf(f, "%s\n", line.c_str());
    }
    fclose(f);
    return order;
  }
  static const char* RES[20] = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"};
  static const char* ELEM[] = {"X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl"};
  int n = (int)M.meta("batch.tokens"), dense = (int)M.meta("batch.dense");
  const int* aatype = M.i("batch.aatype"); const int* names = M.i("batch.refAtomNameChars");
  const int* elem = M.i("batch.refElement"); const float* mask = M.f("batch.refMask");
  const int* chain = M.i("batch.asymId"); const int* resIdx = M.i("batch.residueIndex");
  FILE* f = fopen(path.c_str(), "w");
  int serial = 1;
  for (int t = 0; t < n; ++t)
    for (int a = 0; a < dense; ++a) {
      size_t i = (size_t)t * dense + a;
      if (!mask[i]) continue;
      order.push_back(i);
      char name[5] = {0};
      for (int k = 0; k < 4; ++k) { int c = names[i * 4 + k]; name[k] = c > 0 ? (char)(c + 32) : 0; }
      int z = elem[i]; const char* el = z >= 0 && z < 18 ? ELEM[z] : "X";
      const char* res = aatype[t] >= 0 && aatype[t] < 20 ? RES[aatype[t]] : "UNK";
      fprintf(f, "ATOM  %5d %-4s %3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n", serial++,
              strlen(name) < 4 ? (std::string(" ") + name).c_str() : name, res, 'A' + (chain[t] - 1) % 26, resIdx[t], x[i * 3],
              x[i * 3 + 1], x[i * 3 + 2], 1.0, bfactors ? bfactors[i] : 0.0, el);
    }
  fprintf(f, "END\n");
  fclose(f);
  return order;
}

// <stem>_confidences.json (atom pLDDTs in the structure's atom order, contact probabilities, PAE, the token layout)
// and <stem>_summary_confidences.json (the scalar scores, per chain and per chain pair), as the page's archive writes
void writeConfidences(const std::string& pdbPath, const std::vector<size_t>& order, int n, int dense, const std::vector<float>& plddt,
                      const std::vector<float>& pae, const std::vector<float>& tmTerm, const std::vector<float>& contact, double ptm,
                      double iptm, double ranking, double meanPlddt, bool hasClashed, double fractionDisorder) {
  std::string stem = pdbPath.size() > 4 && (pdbPath.substr(pdbPath.size() - 4) == ".pdb" || pdbPath.substr(pdbPath.size() - 4) == ".cif")
    ? pdbPath.substr(0, pdbPath.size() - 4) : pdbPath;
  const int* asym = M.i("batch.asymId"); const int* res = M.i("batch.residueIndex");
  const float* seq = M.f("batch.seqMask");
  auto chainId = [](int a) {
    std::string id; for (a = a - 1; ; a = a / 26 - 1) { id.insert(id.begin(), (char)('A' + a % 26)); if (a < 26) break; }
    return id;
  };
  std::string j;
  j.reserve((size_t)n * n * 14 + order.size() * 16 + 4096);
  auto fixed2 = [&](float v) { char b[64]; int k = snprintf(b, sizeof b, "%.2f", v); j.append(b, k); };
  auto matrix = [&](const std::vector<float>& m) {
    j += "[";
    for (int i = 0; i < n; ++i) {
      j += i ? ",\n  [" : "[";
      for (int c = 0; c < n; ++c) { if (c) j += ", "; fixed2(m[(size_t)i * n + c]); }
      j += "]";
    }
    j += "]";
  };
  j += "{\"atom_chain_ids\": [";
  for (size_t k = 0; k < order.size(); ++k) { if (k) j += ", "; j += "\"" + chainId(asym[order[k] / dense]) + "\""; }
  j += "],\n \"atom_plddts\": [";
  for (size_t k = 0; k < order.size(); ++k) { if (k) j += ", "; fixed2(plddt[order[k]]); }
  j += "],\n";
  if (!contact.empty()) { j += " \"contact_probs\": "; matrix(contact); j += ",\n"; }
  j += " \"pae\": "; matrix(pae);
  j += ",\n \"token_chain_ids\": [";
  for (int i = 0; i < n; ++i) { if (i) j += ", "; j += "\"" + chainId(asym[i]) + "\""; }
  j += "],\n \"token_res_ids\": [";
  for (int i = 0; i < n; ++i) { if (i) j += ", "; j += std::to_string(res[i]); }
  j += "]}\n";
  FILE* f = fopen((stem + "_confidences.json").c_str(), "w");
  fwrite(j.data(), 1, j.size(), f);
  fclose(f);
  std::vector<int> chains;
  for (int i = 0; i < n; ++i) if (seq[i] > 0 && std::find(chains.begin(), chains.end(), asym[i]) == chains.end()) chains.push_back(asym[i]);
  std::sort(chains.begin(), chains.end());
  int nc = (int)chains.size();
  auto chainOf = [&](int a) { return (int)(std::find(chains.begin(), chains.end(), a) - chains.begin()); };
  auto reduce = [&](const std::function<bool(int, int)>& selects) {
    double best = -1e30; bool any = false;
    for (int i = 0; i < n; ++i) {
      double total = 0; int count = 0;
      for (int k = 0; k < n; ++k) {
        if (!(seq[i] > 0 && seq[k] > 0) || !selects(i, k)) continue;
        total += tmTerm[(size_t)i * n + k]; ++count;
      }
      if (count) { any = true; best = std::max(best, total / count); }
    }
    return any ? best : NAN;
  };
  auto numOut = [](FILE* f, double v) { if (std::isfinite(v)) fprintf(f, "%.2f", v); else fprintf(f, "null"); };
  std::vector<double> chainPtm(nc, NAN), chainIptm(nc, NAN);
  for (int a = 0; a < nc; ++a) {
    int c = chains[a];
    chainPtm[a] = reduce([&](int i, int k) { return asym[i] == c && asym[k] == c; });
    chainIptm[a] = reduce([&](int i, int k) { return (asym[i] == c) != (asym[k] == c); });
  }
  f = fopen((stem + "_summary_confidences.json").c_str(), "w");
  fprintf(f, "{\n  \"chain_ids\": [");
  for (int i = 0; i < n; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", chainId(asym[i]).c_str());
  fprintf(f, "],\n");
  if (!tmTerm.empty()) {
    fprintf(f, "  \"chain_pair_iptm\": [");
    for (int a = 0; a < nc; ++a) {
      fprintf(f, "%s[", a ? ", " : "");
      for (int b = 0; b < nc; ++b) {
        if (b) fprintf(f, ", ");
        if (a == b) { numOut(f, chainPtm[a]); continue; }
        int ca = chains[a], cb = chains[b];
        numOut(f, reduce([&](int i, int k) { return (asym[i] == ca && asym[k] == cb) || (asym[i] == cb && asym[k] == ca); }));
      }
      fprintf(f, "]");
    }
    fprintf(f, "],\n");
  }
  if (!contact.empty()) {
    std::vector<double> mx((size_t)nc * nc, -1);
    for (int i = 0; i < n; ++i)
      for (int k = i + 1; k < n; ++k) {
        int a = chainOf(asym[i]), b = chainOf(asym[k]);
        if (a >= nc || b >= nc || (a == b && std::abs(res[i] - res[k]) <= 6)) continue;
        double v = contact[(size_t)i * n + k];
        if (v > mx[(size_t)a * nc + b]) mx[(size_t)a * nc + b] = mx[(size_t)b * nc + a] = v;
      }
    fprintf(f, "  \"chain_pair_max_contact\": [");
    for (int a = 0; a < nc; ++a) {
      fprintf(f, "%s[", a ? ", " : "");
      for (int b = 0; b < nc; ++b) { if (b) fprintf(f, ", "); numOut(f, mx[(size_t)a * nc + b] < 0 ? NAN : mx[(size_t)a * nc + b]); }
      fprintf(f, "]");
    }
    fprintf(f, "],\n");
  }
  std::vector<double> sum(nc, 0), cnt(nc, 0);
  for (size_t k = 0; k < order.size(); ++k) {
    int a = chainOf(asym[order[k] / dense]);
    if (a < nc) { sum[a] += plddt[order[k]]; cnt[a] += 1; }
  }
  fprintf(f, "  \"chain_plddt\": [");
  for (int a = 0; a < nc; ++a) { if (a) fprintf(f, ", "); numOut(f, cnt[a] ? sum[a] / cnt[a] : NAN); }
  fprintf(f, "],\n");
  if (!tmTerm.empty()) {
    fprintf(f, "  \"chain_ptm\": [");
    for (int a = 0; a < nc; ++a) { if (a) fprintf(f, ", "); numOut(f, chainPtm[a]); }
    fprintf(f, "],\n");
    if (nc > 1) {
      fprintf(f, "  \"chain_iptm\": [");
      for (int a = 0; a < nc; ++a) { if (a) fprintf(f, ", "); numOut(f, chainIptm[a]); }
      fprintf(f, "],\n");
    }
  }
  std::vector<double> mn((size_t)nc * nc, INFINITY);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      int a = chainOf(asym[i]), b = chainOf(asym[k]);
      if (a < nc && b < nc) mn[(size_t)a * nc + b] = std::min(mn[(size_t)a * nc + b], (double)pae[(size_t)i * n + k]);
    }
  fprintf(f, "  \"chain_pair_pae_min\": [");
  for (int a = 0; a < nc; ++a) {
    fprintf(f, "%s[", a ? ", " : "");
    for (int b = 0; b < nc; ++b) { if (b) fprintf(f, ", "); numOut(f, mn[(size_t)a * nc + b]); }
    fprintf(f, "]");
  }
  fprintf(f, "],\n");
  if (std::isfinite(iptm)) fprintf(f, "  \"iptm\": %.2f,\n", iptm);
  fprintf(f, "  \"ptm\": %.2f,\n  \"ranking_score\": %.2f,\n", ptm, ranking);
  fprintf(f, "  \"fraction_disordered\": %.2f,\n  \"has_clash\": %.1f,\n  \"mean_plddt\": %.2f\n}\n", fractionDisorder,
          hasClashed ? 1.0 : 0.0, meanPlddt);
  fclose(f);
}
}  // namespace af3out

void setOutputInput(const std::string& dir) { af3out::DATA_DIR = dir; ++af3out::SCORE_ATOMS_GEN; }
std::vector<size_t> writeStructure(const std::string& path, const std::vector<float>& x, const float* bfactors) {
  return af3out::writeStructure(path, x, bfactors);
}
void writeConfidenceFiles(const std::string& path, const std::vector<size_t>& order, int n, int dense, const ConfidenceOut& c,
                          const std::vector<float>& contact, double ranking, const Scores& s) {
  af3out::writeConfidences(path, order, n, dense, c.plddt, c.pae, c.tmTerm, contact, c.ptm, c.iptm, ranking, c.meanPlddt, s.clash,
                           s.disordered);
}
Scores structureScores(const std::vector<float>& x) {
  af3out::foldAtoms();
  af3out::StructureScores s = af3out::structureScores(x);
  return {s.clash, s.disordered};
}
double rankingScore(double ptm, double iptm, const Scores& s) { return af3out::rankingScore(ptm, iptm, {s.clash, s.disordered}); }
int scorePdbMain(const std::string& path) { return af3out::scorePdbMain(path); }
