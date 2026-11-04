// A sampled trajectory turned into a structure, and the structure written out.
//
// `cpu/rfd4p/decoder.js`'s tail is the spec: `sequenceForDecoder`, `decoderPostprocess` and
// `formatSampleLocalLatents`, then the records. All host code - the arrays are [n, 147] and [n, 111] and the work
// is an argmax and a table lookup, which `cuda/rfd4p/src/generate.cuh` says of its own copy for the same reason.
//
// 🔴 WHAT IS HERE IS THE COMMON PATH AND IT SAYS SO. A de-novo design and a motif-scaffolded one come out right:
// every legal and present atom slot of each emitted token, named from the vocabulary, under the input's own
// residue numbering and chain letters where there was an input. What is NOT here, and is in cuda/rfd4p: an
// atomised token written as the input's own atom, a grafted motif atom naming the component it came from,
// b-factors and occupancies carried over, the unindexed-insertion mapping, and mmCIF output. Those are
// bookkeeping over the same coordinates rather than a different structure, and each needs the tables this port
// does not read yet.
#include "rfd4p.h"
#include "rfd4p/featurise/tables.inc"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

namespace {

std::string trimName(const char* s) { return s ? std::string(s) : std::string(); }

// a PDB atom name's column rule: a one-letter element starts in column 14, anything else in 13
std::string pdbAtomName(const std::string& name) {
  if (name.size() >= 4) return name;
  return " " + name + std::string(3 - name.size(), ' ');
}

std::string padLeft(const std::string& s, int w) {
  return s.size() >= (size_t)w ? s : std::string((size_t)(w - s.size()), ' ') + s;
}

std::string element(const std::string& atomName) {
  // the vocabulary's names are element-plus-counter, so the first character is the element but for the
  // two-letter metals a token never carries in atom37
  if (atomName.empty()) return "C";
  return std::string(1, atomName[0]);
}

}  // namespace

int sequenceForDecoder(int n, const float* logits, const int* condSeq, const int* tokenType, const float* emit,
                       int* seq) {
  int filled = 0;
  for (int i = 0; i < n; ++i) {
    seq[i] = condSeq[i];
    // 🔴 ONLY AN EMITTED TOKEN WITH NO CONDITIONED RESIDUE TAKES THE ARGMAX, and the argmax is over the token
    // TYPE's allowed residues - a protein token cannot come out a nucleotide because its logit happened to win.
    if (emit[i] == 0 || seq[i] != 0) continue;
    const int tt = std::max(0, std::min(rfd4p::tables::N_TOKEN_TYPES - 2, tokenType[i]));
    int best = 0;
    float bestV = -INFINITY;
    for (int r = 0; r < rfd4p::tables::N_TOKENS; ++r) {
      if (!rfd4p::tables::ALLOWED_RESIDUE_TYPES[(size_t)tt * rfd4p::tables::N_TOKENS + r]) continue;
      const float v = logits[(size_t)i * rfd4p::tables::N_TOKENS + r];
      if (v > bestV) { bestV = v; best = r; }
    }
    seq[i] = best;
    filled += 1;
  }
  return filled;
}

void decoderPostprocess(int n, const float* caNm, const int* seq, const float* logits, const float* coorsFlat,
                        const float* presence, const float* emit, const int* coordGroup, const int* attnIdx, int na,
                        float* coors, uint8_t* atomMask, int* residueType) {
  // the decoder's rows are the ATTENDED ones; its outputs are read here against the token they came from
  std::vector<int> rowOf((size_t)n, -1);
  for (int k = 0; k < na; ++k) rowOf[(size_t)attnIdx[k]] = k;
  for (int i = 0; i < n; ++i) {
    residueType[i] = -1;
    if (emit[i] == 0 || rowOf[(size_t)i] < 0) {
      // a padded-away token keeps its sampled alpha carbon and nothing else (`formatSampleLocalLatents`)
      atomMask[(size_t)i * 37 + 1] = 1;
      for (int k = 0; k < 3; ++k) coors[(size_t)i * 111 + 3 + k] = (float)(caNm[i * 3 + k] * 10);
      continue;
    }
    const int row = rowOf[(size_t)i];
    const int tt = std::max(0, std::min(rfd4p::tables::N_TOKEN_TYPES - 2, M.hostI("in/tokenType")[i]));
    int best = 0;
    float bestV = -INFINITY;
    for (int r = 0; r < rfd4p::tables::N_TOKENS; ++r) {
      if (!rfd4p::tables::ALLOWED_RESIDUE_TYPES[(size_t)tt * rfd4p::tables::N_TOKENS + r]) continue;
      const float v = logits[(size_t)row * rfd4p::tables::N_TOKENS + r];
      if (v > bestV) { bestV = v; best = r; }
    }
    const int rt = seq[i] != 0 ? seq[i] : best;
    residueType[i] = rt;
    for (int a = 0; a < 37; ++a) {
      const bool legal = rfd4p::tables::RESTYPE_ATOM37_MASK[(size_t)rt * 37 + a] != 0;
      // 🔴 SLOT 1 IS ALWAYS PRESENT AND A COORDINATE-GROUP ATOM IS REQUIRED, whatever the head says: the alpha
      // carbon is the token's position and a conditioned group's atoms were asked for by the job.
      const bool required = coordGroup && coordGroup[(size_t)i * 37 + a] >= 0;
      const bool present = presence[(size_t)row * 37 + a] > 0 || a == 1 || required;
      if (!(legal && present)) continue;
      atomMask[(size_t)i * 37 + a] = 1;
      for (int k = 0; k < 3; ++k) {
        const float rel = a == 1 ? 0.f : coorsFlat[(size_t)row * 111 + a * 3 + k];
        coors[(size_t)i * 111 + a * 3 + k] = (float)((double)(rel + caNm[i * 3 + k]) * 10);
      }
    }
  }
}

int writePdb(const std::string& path, int n, const float* coors, const uint8_t* atomMask, const int* residueType,
             const int* outResId, const int* outChain, const int* chainIid, double* caSpacing) {
  std::string out;
  int serial = 0;
  std::vector<float> prevCa;
  double worstGap = 0;
  std::vector<double> gaps;
  for (int i = 0; i < n; ++i) {
    const int rt = residueType[i];
    const std::string resName = rt >= 0 ? trimName(rfd4p::tables::TOKENS[(size_t)rt]) : "UNK";
    // the input's own numbering and chain letters where it had them: a motif's residues are deposited under
    // their own numbers, and writing the token index renumbers them out of their target's frame
    const int resId = outResId ? outResId[i] : i + 1;
    std::string chain = "A";
    if (outChain) {
      std::string c;
      for (int k = 0; k < 4; ++k) {
        const char ch = (char)((outChain[i] >> (8 * k)) & 0xff);
        if (ch) c += ch;
      }
      if (!c.empty()) chain = c;
    } else if (chainIid) {
      chain = std::string(1, (char)('A' + (chainIid[i] % 26)));
    }
    for (int a = 0; a < 37; ++a) {
      if (!atomMask[(size_t)i * 37 + a]) continue;
      const std::string atomName = rt >= 0 ? trimName(rfd4p::tables::TOKEN_ATOMS[(size_t)rt][(size_t)a])
                                           : trimName(rfd4p::tables::ATOM37_NAMES[(size_t)a]);
      if (atomName.empty()) continue;
      const float x = coors[(size_t)i * 111 + a * 3], y = coors[(size_t)i * 111 + a * 3 + 1],
                  zc = coors[(size_t)i * 111 + a * 3 + 2];
      char line[96];
      snprintf(line, sizeof line, "ATOM  %5d %s %-3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n",
               (serial + 1) % 100000, pdbAtomName(atomName).c_str(), resName.c_str(), chain.substr(0, 1).c_str(),
               resId, x, y, zc, padLeft(element(atomName), 2).c_str());
      out += line;
      serial += 1;
      if (a == 1) {
        if (!prevCa.empty()) {
          const double d = std::sqrt((double)(x - prevCa[0]) * (x - prevCa[0]) + (double)(y - prevCa[1]) * (y - prevCa[1]) +
                                     (double)(zc - prevCa[2]) * (zc - prevCa[2]));
          gaps.push_back(d);
          worstGap = std::max(worstGap, d);
        }
        prevCa = {x, y, zc};
      }
    }
  }
  out += "END\n";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("cannot write %s", path.c_str());
  fwrite(out.data(), 1, out.size(), f);
  fclose(f);
  if (caSpacing && !gaps.empty()) {
    std::vector<double> s = gaps;
    std::sort(s.begin(), s.end());
    caSpacing[0] = s[s.size() / 2];
    caSpacing[1] = s.front();
    caSpacing[2] = s.back();
  }
  (void)worstGap;
  return serial;
}
