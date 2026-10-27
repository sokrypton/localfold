// Alignments, as the page reads them: shared/input/a3m.js parseA3m, shared/input/chains.js's row-aligned merges,
// shared/af3/featurise/msa-features.js af3MsaFromA3m (the seeded row choice and the profile's rows) and the
// seeded generator it draws from (shared/af3/random.js uniformFrom).
#pragma once
#include <set>
#include <map>
#include <array>
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
  // (a table: the string this was built from a character at a time was 0.5 s of a 28,000-row alignment)
  static const std::array<bool, 256> RESIDUE = [] {
    std::array<bool, 256> t{}; for (char c : std::string("ACDEFGHIKLMNPQRSTVWYX-")) t[(unsigned char)c] = true; return t;
  }();
  auto allowed = [&](unsigned char c) {
    if (anyLetter) return (c >= 'A' && c <= 'Z') || c == '-';
    return RESIDUE[c];
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
    static const std::array<bool, 256> SPACE = [] { std::array<bool, 256> t{}; for (int c = 0; c < 256; ++c) t[c] = std::isspace(c); return t; }();
    for (char c : line)
      if (SPACE[(unsigned char)c]) throw std::runtime_error("A3M sequence " + a.descriptions[current] + " contains whitespace");
    a.raw[current] += line;
  }
  if (a.raw.empty()) throw std::runtime_error("A3M contains no sequences");
  for (size_t row = 0; row < a.raw.size(); ++row) {
    const std::string& r = a.raw[row];
    if (r.empty()) throw std::runtime_error("A3M sequence " + a.descriptions[row] + " is empty");
    std::string seq(r.size(), '\0');
    std::vector<int> del(r.size());
    size_t n = 0;
    int insertions = 0;
    for (size_t i = 0; i < r.size(); ++i) {
      unsigned char c = (unsigned char)r[i];
      if (c >= 'a' && c <= 'z') { ++insertions; continue; }
      if (!anyLetter) {   // a3m.js AMBIGUOUS: B, Z to the acid, U to cysteine, J, O to unknown (AF2's and AF3's maps)
        unsigned char to = c == 'B' ? 'D' : c == 'Z' ? 'E' : c == 'U' ? 'C' : (c == 'J' || c == 'O') ? 'X' : 0;
        if (to) { c = to; a.raw[row][i] = (char)to; }
      }
      if (c > 127 || !allowed(c)) {
        throw std::runtime_error("A3M sequence " + a.descriptions[row] + " contains invalid residue \"" + std::string(1, (char)c) + "\"");
      }
      seq[n] = (char)c; del[n++] = insertions;
      insertions = 0;
    }
    seq.resize(n); del.resize(n);
    a.sequences.push_back(std::move(seq));
    a.deletions.push_back(std::move(del));
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

// chains.js uniprotSpecies: AF3's ExtractSpeciesIds - `sp|ID|NAME_SPECIES` / `tr|...`, "" for anything else
inline std::string uniprotSpecies(const std::string& description) {
  std::string part = trimWs(description);
  part = part.substr(0, part.find(' '));
  if (part.rfind("sp|", 0) != 0 && part.rfind("tr|", 0) != 0) return "";
  std::vector<std::string> fields;
  for (size_t at = 0;;) {
    size_t bar = part.find('|', at);
    fields.push_back(part.substr(at, bar == std::string::npos ? std::string::npos : bar - at));
    if (bar == std::string::npos) break;
    at = bar + 1;
  }
  if (fields.size() != 3 || fields[1].find('-') != std::string::npos) return "";
  size_t underscore = fields[2].find('_');
  if (underscore == std::string::npos) return "";
  std::string species = fields[2].substr(underscore + 1);
  species = species.substr(0, species.find('_'));
  return species.substr(0, species.find('/'));
}

// chains.js pairA3msBySpecies: AF3's create_paired_features over per-chain paired A3Ms; empty (no value) when no
// chain names a species
inline bool pairA3msBySpecies(const std::vector<std::string>& texts, bool anyLetter, const std::vector<bool>& pairable,
                              std::string& out) {
  const int maxHits = 600, maxRows = 8192;
  std::vector<A3m> al;
  for (auto& t : texts) al.push_back(parseA3m(t, anyLetter));
  std::vector<std::map<std::string, std::vector<int>>> speciesRows(al.size());
  bool any = false;
  for (size_t c = 0; c < al.size(); ++c) {
    if (!pairable.empty() && !pairable[c]) continue;
    for (int row = 0; row < al[c].depth; ++row) {
      std::string sp = uniprotSpecies(al[c].descriptions[row]);
      if (sp.empty()) continue;
      speciesRows[c][sp].push_back(row);
      any = true;
    }
  }
  if (!any) return false;
  std::vector<std::string> order;   // chain by chain, each chain's species sorted (the JS Map's insertion order)
  std::map<std::string, int> chainsOf, minHits;
  for (auto& rows : speciesRows)
    for (auto& [sp, list] : rows) {
      if (!chainsOf.count(sp)) { order.push_back(sp); chainsOf[sp] = 0; minHits[sp] = maxHits; }
      chainsOf[sp] += 1;
      minHits[sp] = std::min(minHits[sp], (int)list.size());
    }
  std::map<int, std::vector<std::string>, std::greater<int>> byCount;
  for (auto& sp : order) if (chainsOf[sp] > 1) byCount[chainsOf[sp]].push_back(sp);
  std::vector<std::vector<int>> selected{std::vector<int>(al.size(), 0)};
  int seen = 0;
  for (auto& [count, list] : byCount) {
    std::vector<std::vector<int>> block;
    for (auto& sp : list)
      for (int k = 0; k < minHits[sp]; ++k) {
        std::vector<int> row;
        for (auto& rows : speciesRows) { auto f = rows.find(sp); row.push_back(f == rows.end() ? -1 : f->second[k]); }
        block.push_back(row);
      }
    std::vector<double> key;
    for (auto& row : block) { double p = 1; for (int i : row) p = p * (double)std::abs(i); key.push_back(p); }
    std::vector<int> at(block.size());
    for (size_t i = 0; i < at.size(); ++i) at[i] = (int)i;
    std::stable_sort(at.begin(), at.end(), [&](int a, int b) { return key[a] < key[b]; });
    for (int i : at) selected.push_back(block[i]);
    seen += (int)block.size();
    if (seen >= maxRows) break;
  }
  if ((int)selected.size() > maxRows) selected.resize(maxRows);
  out.clear();
  bool first = true;
  for (auto& row : selected) {
    std::string parts, label;
    for (size_t c = 0; c < al.size(); ++c) {
      parts += row[c] < 0 ? std::string(al[c].length, '-') : al[c].raw[row[c]];
      label += (c ? " " : "") + std::to_string(c + 1) + ":" + (row[c] < 0 ? std::string("-") : al[c].descriptions[row[c]]);
    }
    if (!first && std::none_of(parts.begin(), parts.end(), [](char ch) { return ch >= 'A' && ch <= 'Z'; })) continue;
    out += ">" + (first ? std::string("query") : label) + "\n" + parts + "\n";
    first = false;
  }
  return true;
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
  std::vector<std::vector<int8_t>> msa;
  std::vector<std::vector<uint8_t>> deletionMatrix;
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

// msa-features.js capMainMsa: per chain, the first `cap` rows (the chain's query first), deduplicated on the aligned row
inline A3m capMainMsa(const A3m& parsed, int cap, const std::vector<int>* chainOfColumn) {
  auto chainOf = [&](int row) {
    const std::string& aligned = parsed.sequences[row];
    for (size_t c = 0; c < aligned.size(); ++c) if (aligned[c] != '-') return chainOfColumn ? (*chainOfColumn)[c] : 0;
    return chainOfColumn ? (*chainOfColumn)[0] : 0;
  };
  std::map<int, int> counted; std::map<int, std::set<std::string>> seen;
  std::vector<int> keep{0};
  for (int row = 1; row < parsed.depth; ++row) {
    int chain = chainOf(row);
    if (!seen.count(chain)) { seen[chain].insert(parsed.sequences[0]); counted[chain] = 1; }
    int n = counted[chain]++;
    if (n >= cap || seen[chain].count(parsed.sequences[row])) continue;
    seen[chain].insert(parsed.sequences[row]);
    keep.push_back(row);
  }
  if ((int)keep.size() == parsed.depth) return parsed;
  A3m out; out.length = parsed.length; out.depth = (int)keep.size();
  for (int r : keep) {
    out.descriptions.push_back(parsed.descriptions[r]); out.raw.push_back(parsed.raw[r]);
    out.sequences.push_back(parsed.sequences[r]); out.deletions.push_back(parsed.deletions[r]);
  }
  return out;
}

// msa-features.js pairedIntoUnpaired: protenix's unpaired MSA - each protein chain's paired rows, then its unpaired
// rows (a pad past the query skipped), deduplicated on the aligned row, the chains stacked side by side again
inline A3m pairedIntoUnpaired(const A3m* unpaired, const A3m* paired, const std::vector<int>* chainOfColumn,
                              const std::vector<std::string>* columnKinds, int dedupe = 1) {   // 0 none, 1 rows, 2 letters
  const A3m& base = unpaired ? *unpaired : *paired;
  int length = base.length;
  struct Range { int chain, from, to; };
  std::vector<Range> ranges;
  for (int column = 0; column < length; ++column) {
    int chain = chainOfColumn ? (*chainOfColumn)[column] : 0;
    if (ranges.empty() || ranges.back().chain != chain) ranges.push_back({chain, column, column});
    ranges.back().to = column + 1;
  }
  struct Row { std::string sequence; std::vector<int> deletions; };
  std::vector<std::vector<Row>> lists;
  for (auto& range : ranges) {
    bool protein = !columnKinds || range.from >= (int)columnKinds->size() || (*columnKinds)[range.from] == "protein";
    std::vector<Row> rows;
    auto take = [&](const A3m& a, int row) {
      return Row{a.sequences[row].substr(range.from, range.to - range.from),
                 std::vector<int>(a.deletions[row].begin() + range.from, a.deletions[row].begin() + range.to)};
    };
    if (protein && paired) for (int row = 0; row < paired->depth; ++row) rows.push_back(take(*paired, row));
    if (unpaired)   // (without the dedupe - rf3's - the unpaired block's own query is skipped after the paired rows)
      for (int row = dedupe || !protein || !paired ? 0 : 1; row < unpaired->depth; ++row) {
        Row one = take(*unpaired, row);
        if (row > 0 && one.sequence.find_first_not_of('-') == std::string::npos) continue;
        rows.push_back(one);
      }
    if (protein && dedupe) {
      std::set<std::string> seen;
      std::vector<Row> kept;
      for (auto& one : rows) {
        std::string key = one.sequence;
        if (dedupe == 2) key.erase(std::remove(key.begin(), key.end(), '-'), key.end());   // ("letters": IntelliFold's)
        if (seen.insert(key).second) kept.push_back(one);
      }
      rows.swap(kept);
    }
    lists.push_back(rows);
  }
  int depth = 0;
  for (auto& rows : lists) depth = std::max(depth, (int)rows.size());
  A3m out;
  out.length = length; out.depth = depth;
  for (int row = 0; row < depth; ++row) {
    std::string sequence; std::vector<int> deletions;
    for (size_t k = 0; k < ranges.size(); ++k) {
      int width = ranges[k].to - ranges[k].from;
      if (row < (int)lists[k].size()) {
        sequence += lists[k][row].sequence;
        deletions.insert(deletions.end(), lists[k][row].deletions.begin(), lists[k][row].deletions.end());
      } else {
        sequence += std::string(width, '-');
        deletions.insert(deletions.end(), width, 0);
      }
    }
    out.descriptions.push_back("row " + std::to_string(row));
    out.raw.push_back(sequence); out.sequences.push_back(sequence); out.deletions.push_back(deletions);
  }
  return out;
}

inline MsaRows af3MsaFromA3m(const Alignment& alignment, const std::vector<std::string>* columnKinds,
                             int mainMsaCap = 0, const std::vector<int>* chainOfColumn = nullptr,
                             bool protenixUnpairedMsa = false, bool rf3ProfileMsa = false, bool if2ProfileMsa = false) {
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
  if (mainMsaCap > 0) {                           // (openbind0's main MSA: the unpaired block and the profile's rows)
    if (unpaired.ok) unpaired.a = capMainMsa(unpaired.a, mainMsaCap, chainOfColumn);
    if (profileBlock.ok) profileBlock.a = capMainMsa(profileBlock.a, mainMsaCap, chainOfColumn);
  }
  if (protenixUnpairedMsa && (unpaired.ok || paired.ok)) {   // (protenix, opendde: pairedIntoUnpaired)
    const A3m* p = paired.ok ? &paired.a : nullptr;
    unpaired.a = pairedIntoUnpaired(unpaired.ok ? &unpaired.a : nullptr, p, chainOfColumn, columnKinds); unpaired.ok = true;
    if (profileBlock.ok) profileBlock.a = pairedIntoUnpaired(&profileBlock.a, p, chainOfColumn, columnKinds);
  }
  if (rf3ProfileMsa && paired.ok) {   // (rf3: the profile over each chain's query, paired and unpaired rows - msa-features.js)
    profileBlock.a = pairedIntoUnpaired(profileBlock.ok ? &profileBlock.a : (unpaired.ok ? &unpaired.a : nullptr), &paired.a,
                                        chainOfColumn, columnKinds, 0);
    profileBlock.ok = true;
  }
  if (if2ProfileMsa && paired.ok) {   // (intellifold2: each chain's own sequences, repeats gone - msa-features.js)
    profileBlock.a = pairedIntoUnpaired(profileBlock.ok ? &profileBlock.a : (unpaired.ok ? &unpaired.a : nullptr), &paired.a,
                                        chainOfColumn, columnKinds, 2);
    profileBlock.ok = true;
  }
  const Parsed& unpairedProfile = profileBlock.ok ? profileBlock : unpaired;
  // (msa-features.js: the rows are AF3's whole 16384-row MSA - features.py's crop, paired to half - and the trunk
  // draws num_msa of them afresh every pass, shared/af3/msa-draw.js and msa_draw.h)
  int pairedBlock = !paired.ok ? 1 : paired.a.depth + 1;
  const int msaSize = 16384;
  int pairedPool = !paired.ok ? 0 : std::min(paired.a.depth, std::min(pairedBlock, msaSize / 2) - 1);
  int unpairedPool = !unpaired.ok ? 0 : std::min(unpaired.a.depth, msaSize - std::min(pairedBlock, msaSize / 2));
  MsaRows out;
  auto append = [&](const Parsed& p, int from, int upTo) {
    if (!p.ok) return;
    for (int row = from; row < upTo; ++row) {
      std::vector<int8_t> codes(p.a.length);
      for (int c = 0; c < p.a.length; ++c) codes[c] = (int8_t)codeOf(p.a.sequences[row], c);
      out.msa.push_back(std::move(codes));
      // (a deletion count is AF3's int8, clipped at 127 - msa-features.js; the profile's rows keep the true count)
      std::vector<uint8_t> del(p.a.deletions[row].size());
      for (size_t c = 0; c < del.size(); ++c) del[c] = (uint8_t)std::min((float)p.a.deletions[row][c], 127.f);
      out.deletionMatrix.push_back(std::move(del));
    }
  };
  append(paired, 0, pairedPool);
  int unpairedFrom = (int)out.msa.size() + 1;
  int unpairedStart = !paired.ok ? 0 : 1;
  append(unpaired, unpairedStart, unpairedPool);
  int unpairedRows = (int)out.msa.size() + 1 - unpairedFrom;
  if (unpairedProfile.ok) {
    for (int row = 0; row < unpairedProfile.a.depth; ++row) {
      std::vector<int> codes(unpairedProfile.a.length);
      for (int c = 0; c < unpairedProfile.a.length; ++c) codes[c] = codeOf(unpairedProfile.a.sequences[row], c);
      out.profileMsa.push_back(codes);
      out.profileDeletionMatrix.emplace_back(unpairedProfile.a.deletions[row].begin(), unpairedProfile.a.deletions[row].end());
    }
  }
  out.hasProfile = true;
  out.unpairedFrom = unpairedRows == 0 ? 0 : unpairedFrom;
  return out;
}

}  // namespace lf
