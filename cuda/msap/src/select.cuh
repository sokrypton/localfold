// MSA Pairformer's input: an A3M read, filtered and subsampled to the model's depth, tokenised. Host C++, no CUDA.
//
// The reference (MSA_Pairformer.dataset.MSA) removes insertions (lowercase, '.', '*'), keeps the query first, then
// subsamples with hhfilter -id 90 -diff N -cov 70 -qid 30 -qsc -20 and greedily to N by the mean Hamming distance
// to the rows already chosen. Here the hhfilter stage is msa-tools' (github.com/sokrypton/msa-tools) filter -
// coverage and identity to the query (filterRows) - and the greedy stage is the reference's own (msap.cu:
// greedyRows, on the device).
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Alignment {
  std::vector<std::string> rows;      // insertions removed, every row the query's length
  std::vector<int> chainBreaks;       // the residue each chain after the first starts at (an A3M's '#' header)
};
// the reference's tokens (aa2tok_d): 20 amino acids, X B Z U O, '-'; anything else reads as X
inline int msapToken(char c) {
  static const char* A = "ARNDCEQGHILKMFPSTWYVXBZUO-";
  const char* p = strchr(A, c);
  return p && c ? (int)(p - A) : 20;
}
inline Alignment readA3m(const std::string& path) {
  std::ifstream in(path);
  if (!in) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
  Alignment a; std::string line, cur; bool inRecord = false;
  auto flush = [&]() {
    if (!inRecord) return;
    if (!a.rows.empty() && cur.size() != a.rows[0].size()) {
      fprintf(stderr, "%s: row %zu is %zu columns against the query's %zu\n", path.c_str(), a.rows.size(), cur.size(), a.rows[0].size());
      exit(1);
    }
    a.rows.push_back(cur); cur.clear();
  };
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '#') {               // ColabFold's paired header: "#len1,len2 card1,card2"
      if (a.rows.empty() && !inRecord) {
        int at = 0; const char* p = line.c_str() + 1;
        while (*p && *p != ' ' && *p != '\t') {
          int n = (int)strtol(p, (char**)&p, 10);
          if (at > 0) a.chainBreaks.push_back(at);
          at += n;
          if (*p == ',') ++p; else break;
        }
      }
      continue;
    }
    if (line[0] == '>') { flush(); inRecord = true; continue; }
    if (!inRecord) continue;
    for (char c : line) if (!(c >= 'a' && c <= 'z') && c != '.' && c != '*') cur.push_back((char)toupper(c));
  }
  flush();
  if (a.rows.empty()) { fprintf(stderr, "%s holds no sequences\n", path.c_str()); exit(1); }
  return a;
}

// The rows the greedy stage chooses from: coverage of the query's columns and identity to the query over the columns
// both cover - hhfilter's -cov 70 -qid 30 (msa-tools' filter -cov -qid), the query always first. hhfilter's -id 90
// and -diff are left out: measured with the reference model on 5CAJ, 1TIM and 5K9P (Cb contacts, top-L long range
// against the crystal) the reference's subsample reads 0.885 / 0.891 / 0.853 and this one with the greedy stage
// after it 0.854 / 0.895 / 0.813 (top-L/2: 0.946 / 0.959 / 0.892 against 0.954 / 0.943 / 0.919), where the
// alignment's first 512 rows read 0.724 / 0.781 / 0.240 - the greedy stage does the selecting, and a redundancy cut
// before it is quadratic in the depth (10,836 rows of 1TIM's 16,469 kept, 134 s in numpy)
inline std::vector<int> filterRows(const std::vector<std::string>& rows, float cov, float qid) {
  const int n = (int)rows.size(), L = (int)rows[0].size();
  const std::string& q = rows[0];
  std::vector<int> keep{0};
  for (int k = 1; k < n; ++k) {
    const std::string& s = rows[k];
    int nongap = 0, both = 0, same = 0;
    for (int i = 0; i < L; ++i) {
      if (s[i] == '-') continue;
      ++nongap;
      if (q[i] != '-') { ++both; same += s[i] == q[i]; }
    }
    if (nongap < cov * L || same < qid * std::max(both, 1)) continue;
    keep.push_back(k);
  }
  return keep;
}
