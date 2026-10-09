// A fold's files: the page's PDB records with these coordinates, and AlphaFold 3's confidence files beside them
// (<stem>_confidences.json, <stem>_summary_confidences.json) in cuda/af3's layout, so one reader takes every port's.
#include "ef2.h"
#include <cmath>
#include <fstream>

void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x, const std::vector<float>* bfactor) {
  std::ifstream in(templatePath);
  if (!in) die("no %s (the featuriser writes it)", templatePath.c_str());
  FILE* f = fopen(out.c_str(), "w");
  if (!f) die("cannot write %s", out.c_str());
  std::string line;
  while (std::getline(in, line)) {
    if ((line.rfind("ATOM", 0) == 0 || line.rfind("HETATM", 0) == 0) && line.size() >= 66) {
      int atom = (int)lround(atof(line.substr(30, 8).c_str())) * 1000 + (int)lround(atof(line.substr(38, 8).c_str()));
      char coords[64];
      snprintf(coords, sizeof coords, "%8.3f%8.3f%8.3f  1.00%6.2f", x[atom * 3], x[atom * 3 + 1], x[atom * 3 + 2],
               bfactor ? (*bfactor)[atom] : 0.0);
      line = line.substr(0, 30) + coords + line.substr(66);
    }
    fprintf(f, "%s\n", line.c_str());
  }
  fclose(f);
}

// a square matrix as the confidence files write it: rows "[a, b, ...]" joined by ",\n  ", two decimals
static void appendMatrix2(std::string& j, const float* m, int L) {
  j.reserve(j.size() + (size_t)L * L * 7 + 16);
  j += "[";
  for (int i = 0; i < L; ++i) {
    j += i ? ",\n  [" : "[";
    for (int c = 0; c < L; ++c) {
      if (c) j += ", ";
      char b[32]; int n = snprintf(b, sizeof b, "%.2f", m[(size_t)i * L + c]); j.append(b, n);
    }
    j += "]";
  }
  j += "]";
}
void writeConfidences(const std::string& pdb, int T, const Confidence& conf, const std::vector<float>& contacts) {
  std::string stem = pdb.size() > 4 && pdb.substr(pdb.size() - 4) == ".pdb" ? pdb.substr(0, pdb.size() - 4) : pdb;
  std::vector<int> asym(M.hostI("asym_id"), M.hostI("asym_id") + T), res(M.hostI("residue_index"), M.hostI("residue_index") + T);
  auto chainId = [](int a) {
    std::string id; for (a = a - 1; ; a = a / 26 - 1) { id.insert(id.begin(), (char)('A' + a % 26)); if (a < 26) break; }
    return id;
  };
  bool chains = false; for (int t = 1; t < T; ++t) chains |= asym[t] != asym[0];
  int first = *std::min_element(asym.begin(), asym.end());
  FILE* f = fopen((stem + "_confidences.json").c_str(), "w");
  {
    std::string j = "{\"pae\": ";
    appendMatrix2(j, conf.pae.data(), T);
    if (!contacts.empty()) { j += ",\n \"contact_probs\": "; appendMatrix2(j, contacts.data(), T); }
    fwrite(j.data(), 1, j.size(), f);
  }
  fprintf(f, ",\n \"token_plddts\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s%.2f", i ? ", " : "", 100.f * conf.plddtToken[i]);
  fprintf(f, "],\n \"token_chain_ids\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", chainId(asym[i] - first + 1).c_str());
  fprintf(f, "],\n \"token_res_ids\": [");
  for (int i = 0; i < T; ++i) fprintf(f, "%s%d", i ? ", " : "", res[i] + 1);
  fprintf(f, "]}\n");
  fclose(f);
  f = fopen((stem + "_summary_confidences.json").c_str(), "w");
  if (chains) fprintf(f, "{\"ptm\": %.4f, \"iptm\": %.4f, \"mean_plddt\": %.2f}\n", conf.ptm, conf.iptm, 100 * conf.meanPlddt);
  else fprintf(f, "{\"ptm\": %.4f, \"iptm\": null, \"mean_plddt\": %.2f}\n", conf.ptm, 100 * conf.meanPlddt);
  fclose(f);
}

