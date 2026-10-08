// Alignments, as the page reads them: shared/input/a3m.js parseA3m, shared/input/chains.js's row-aligned merges,
// shared/af3/featurise/msa-features.js af3MsaFromA3m (the seeded row choice and the profile's rows) and the
// seeded generator it draws from (shared/af3/random.js uniformFrom).
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccd.h"
#include "tables.h"

namespace lf {

// uniformFrom(seed): an LCG, exact in 64-bit integers as JavaScript's doubles are exact below 2^53
struct Uniform {
  uint64_t state;
  explicit Uniform(double seed) : state((uint64_t)(int64_t)seed & 0xFFFFFFFFu) {}
  double operator()() {
    state = (state * 1664525u + 1013904223u) & 0xFFFFFFFFu;
    return (double)(state + 1) / 4294967297.0;
  }
};

struct A3m {
  std::vector<std::string> descriptions, raw, sequences;
  std::vector<std::vector<int>> deletions;
  int depth = 0, length = 0;
};

inline A3m parseA3m(const std::string& text, bool anyLetter = false) {
  auto allowed = [&](unsigned char c) {
    if (anyLetter) return (c >= 'A' && c <= 'Z') || c == '-';
    return std::string("ACDEFGHIKLMNPQRSTVWYX-").find((char)c) != std::string::npos && c != 0;
  };
  A3m a;
  int current = -1;
  for (auto& source : splitLines(text)) {
    std::string line = trimWs(source);
    if (line.empty() || line[0] == '#') continue;
    if (line[0] == '>') {
      std::string d = trimWs(line.substr(1));
      if (d.empty()) throw std::runtime_error("A3M contains an empty FASTA header");
      a.descriptions.push_back(d);
      a.raw.push_back("");
      ++current;
      continue;
    }
    if (current < 0) throw std::runtime_error("A3M sequence data appears before the first FASTA header");
    for (char c : line)
      if (std::isspace((unsigned char)c)) throw std::runtime_error("A3M sequence " + a.descriptions[current] + " contains whitespace");
    a.raw[current] += line;
  }
  if (a.raw.empty()) throw std::runtime_error("A3M contains no sequences");
  for (size_t row = 0; row < a.raw.size(); ++row) {
    const std::string& r = a.raw[row];
    if (r.empty()) throw std::runtime_error("A3M sequence " + a.descriptions[row] + " is empty");
    std::string seq;
    std::vector<int> del;
    int insertions = 0;
    for (size_t i = 0; i < r.size(); ++i) {
      unsigned char c = (unsigned char)r[i];
      if (c >= 'a' && c <= 'z') { ++insertions; continue; }
      if (c > 127 || !allowed(c)) {
        throw std::runtime_error("A3M sequence " + a.descriptions[row] + " contains invalid residue \"" + std::string(1, (char)c) + "\"");
      }
      seq += (char)c;
      del.push_back(insertions);
      insertions = 0;
    }
    a.sequences.push_back(seq);
    a.deletions.push_back(del);
  }
  a.length = (int)a.sequences[0].size();
  if (a.length == 0 || a.sequences[0].find('-') != std::string::npos)
    throw std::runtime_error("the first A3M sequence must be a non-empty, ungapped query");
  for (size_t row = 0; row < a.sequences.size(); ++row)
    if ((int)a.sequences[row].size() != a.length)
      throw std::runtime_error("A3M row " + a.descriptions[row] + " has aligned length " + std::to_string(a.sequences[row].size())
                               + "; expected " + std::to_string(a.length));
  a.depth = (int)a.sequences.size();
  return a;
}

// mergeRowAlignedChainA3ms: each chain's rows side by side, row k beside row k, a short one padded with gaps
inline std::string mergeRowAlignedChainA3ms(const std::vector<std::string>& texts, bool anyLetter = false) {
  if (texts.empty()) throw std::runtime_error("at least one chain A3M is required");
  std::vector<A3m> al;
  for (auto& t : texts) al.push_back(parseA3m(t, anyLetter));
  int depth = 0;
  for (auto& a : al) depth = std::max(depth, a.depth);
  std::string out;
  for (int row = 0; row < depth; ++row) {
    std::string label, parts;
    for (size_t c = 0; c < al.size(); ++c) {
      parts += row < al[c].depth ? al[c].raw[row] : std::string(al[c].length, '-');
      if (row > 0) label += (c ? " " : "") + std::to_string(c + 1) + ":" + (row < al[c].depth ? al[c].descriptions[row] : "-");
    }
    out += ">" + (row == 0 ? std::string("query") : label) + "\n" + parts + "\n";
  }
  return out;
}

// The alignment af3MsaFromA3m reads: one text, or a paired and an unpaired block (each may be absent)
struct Alignment {
  bool present = false;            // null alignment: no rows at all
  bool blocks = false;             // {paired, unpaired} rather than one text
  std::string single;
  bool hasPaired = false, hasUnpaired = false, hasUnpairedProfile = false;
  std::string paired, unpaired, unpairedProfile;   // the profile's own rows: the search's block before deduplication
};

struct MsaRows {
  std::vector<std::vector<int>> msa;
  std::vector<std::vector<float>> deletionMatrix;
  std::vector<std::vector<int>> profileMsa;
  std::vector<std::vector<float>> profileDeletionMatrix;
  bool hasProfile = false;
  int unpairedFrom = 0;
};

// AF3_RNA_MSA_CODES / AF3_DNA_MSA_CODES: every letter 30 but the kind's four, the gap 21
inline int nucleicMsaCode(const std::string& kind, char c) {
  if (c == '-') return 21;
  if (kind == "rna") { if (c == 'A') return 22; if (c == 'G') return 23; if (c == 'C') return 24; if (c == 'U') return 25; }
  else { if (c == 'A') return 26; if (c == 'G') return 27; if (c == 'C') return 28; if (c == 'T') return 29; }
  return (c >= 'A' && c <= 'Z') ? 30 : -1;
}

inline MsaRows af3MsaFromA3m(const Alignment& alignment, int maxSequences, const std::vector<std::string>* columnKinds,
                             Uniform* random) {
  auto codeOf = [&](const std::string& aligned, int column) {
    char ch = aligned[column];
    if (columnKinds == nullptr || (*columnKinds)[column] == "protein" ||
        ((*columnKinds)[column] != "rna" && (*columnKinds)[column] != "dna")) {
      int c = af3MsaCode(ch);
      return c >= 0 ? c : 20;                          // ?? AF3_MSA_CODES.X
    }
    int c = nucleicMsaCode((*columnKinds)[column], ch);
    return c >= 0 ? c : 30;
  };
  struct Parsed { bool ok = false; A3m a; };
  auto parse = [&](bool has, const std::string& text) {
    Parsed p;
    if (!has || trimWs(text).empty()) return p;
    p.a = parseA3m(text, columnKinds != nullptr);
    if (columnKinds && p.a.length != (int)columnKinds->size())
      throw std::runtime_error("the alignment is " + std::to_string(p.a.length) + " columns and its chains " + std::to_string(columnKinds->size()));
    p.ok = true;
    return p;
  };
  Parsed paired = alignment.blocks ? parse(alignment.hasPaired, alignment.paired) : Parsed{};
  Parsed unpaired = alignment.blocks ? parse(alignment.hasUnpaired, alignment.unpaired) : parse(true, alignment.single);
  Parsed profileBlock = alignment.blocks ? parse(alignment.hasUnpairedProfile, alignment.unpairedProfile) : Parsed{};
  const Parsed& unpairedProfile = profileBlock.ok ? profileBlock : unpaired;
  int pairedBlock = !paired.ok ? 1 : paired.a.depth + 1;
  int pairedCrop = std::min(pairedBlock, std::max(1, maxSequences / 2));
  int unpairedCrop = !unpaired.ok ? 0 : std::min(unpaired.a.depth, maxSequences - pairedCrop);
  auto chooseRows = [&](std::vector<int> available, int count) {
    if (count >= (int)available.size()) return available;
    if (random == nullptr) { available.resize(std::max(0, count)); return available; }
    for (int index = (int)available.size() - 1; index > 0; --index) {
      int swap = (int)std::floor((*random)() * (index + 1));
      std::swap(available[index], available[swap]);
    }
    available.resize(std::max(0, count));
    std::sort(available.begin(), available.end());
    return available;
  };
  MsaRows out;
  auto append = [&](const Parsed& p, int from, int upTo) {
    if (!p.ok) return;
    std::vector<int> span;
    for (int row = from; row < p.a.depth; ++row) span.push_back(row);
    for (int row : chooseRows(span, upTo - from)) {
      std::vector<int> codes(p.a.length);
      for (int c = 0; c < p.a.length; ++c) codes[c] = codeOf(p.a.sequences[row], c);
      out.msa.push_back(codes);
      out.deletionMatrix.emplace_back(p.a.deletions[row].begin(), p.a.deletions[row].end());
    }
  };
  append(paired, 0, pairedCrop - 1);
  int unpairedFrom = (int)out.msa.size() + 1;
  int unpairedStart = !paired.ok ? 0 : 1;
  append(unpaired, unpairedStart, !unpaired.ok ? 0 : std::min(unpaired.a.depth, unpairedCrop + unpairedStart));
  if (unpairedProfile.ok) {
    for (int row = 0; row < unpairedProfile.a.depth; ++row) {
      std::vector<int> codes(unpairedProfile.a.length);
      for (int c = 0; c < unpairedProfile.a.length; ++c) codes[c] = codeOf(unpairedProfile.a.sequences[row], c);
      out.profileMsa.push_back(codes);
      out.profileDeletionMatrix.emplace_back(unpairedProfile.a.deletions[row].begin(), unpairedProfile.a.deletions[row].end());
    }
  }
  out.hasProfile = true;
  out.unpairedFrom = unpairedCrop == 0 ? 0 : unpairedFrom;
  return out;
}

}  // namespace lf
