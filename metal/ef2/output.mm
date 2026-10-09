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

// the rotation taking `moving` onto `fixed` (both centred), Horn's quaternion: the frames' superposition
void bestRotation(const std::vector<double>& moving, const std::vector<double>& fixed, double R[9]) {
  double S[3][3] = {};
  for (size_t i = 0; i + 2 < moving.size(); i += 3)
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) S[a][b] += moving[i + a] * fixed[i + b];
  double N[4][4] = {
    {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
    {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
    {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
    {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
  double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0; for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) off += N[p][q] * N[p][q];
    if (off < 1e-22) break;
    for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) {
      if (std::fabs(N[p][q]) < 1e-300) continue;
      double theta = (N[q][q] - N[p][p]) / (2 * N[p][q]);
      double t = (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(theta * theta + 1)), c = 1 / std::sqrt(t * t + 1), sn = t * c;
      for (int k = 0; k < 4; ++k) { double a = N[k][p], b = N[k][q]; N[k][p] = c * a - sn * b; N[k][q] = sn * a + c * b; }
      for (int k = 0; k < 4; ++k) { double a = N[p][k], b = N[q][k]; N[p][k] = c * a - sn * b; N[q][k] = sn * a + c * b; }
      for (int k = 0; k < 4; ++k) { double a = V[k][p], b = V[k][q]; V[k][p] = c * a - sn * b; V[k][q] = sn * a + c * b; }
    }
  }
  int best = 0; for (int k = 1; k < 4; ++k) if (N[k][k] > N[best][best]) best = k;
  double w = V[0][best], x = V[1][best], y = V[2][best], z = V[3][best];
  double R0[9] = {w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z};
  for (int k = 0; k < 9; ++k) R[k] = R0[k];
}
