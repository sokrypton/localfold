// chai-1's MSA inputs where AlphaFold 3's differ, once there is an alignment: shared/af3/featurise/chai-msa.js, whose
// header says what and why (af3-any-model 1f9b3cf _chai_msa; a query-only alignment is none, ColabFold ebbd90d).
// test:featurise holds the two to the same bytes.
#pragma once
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "featurise.h"

namespace lf {

struct ChaiRow { std::vector<int> tokens, deletions; };

// chai's a3m tokeniser over one row (width < 0: every column)
inline ChaiRow chaiTokenise(const std::string& row, long width = -1) {
  static const std::string PROTEIN = "ARNDCQEGHILKMFPSTWYV";
  ChaiRow out;
  int lower = 0;
  for (unsigned char ch : row) {
    if ((ch >= 'A' && ch <= 'Z') || ch == '-') {
      if (width >= 0 && (long)out.tokens.size() >= width) break;
      size_t index = PROTEIN.find((char)ch);
      out.tokens.push_back(ch == '-' ? MSA_GAP : index != std::string::npos ? (int)index : 20);
      out.deletions.push_back(std::min(lower, 255));
      lower = 0;
    } else if (ch >= 'a' && ch <= 'z') {
      ++lower;
    }
  }
  while (width >= 0 && (long)out.tokens.size() < width) { out.tokens.push_back(MSA_GAP); out.deletions.push_back(0); }
  return out;
}

inline std::vector<std::string> chaiRecords(const std::string& text) {
  std::vector<std::string> out;
  bool open = false;
  std::string current;
  size_t at = 0;
  while (at <= text.size()) {
    size_t end = text.find('\n', at);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(at, end - at);
    size_t a = line.find_first_not_of(" \t\r\f\v"), z = line.find_last_not_of(" \t\r\f\v");
    line = a == std::string::npos ? "" : line.substr(a, z - a + 1);
    if (!line.empty() && line[0] == '>') { if (open) out.push_back(current); current.clear(); open = true; }
    else { open = true; current += line; }
    at = end + 1;
  }
  if (open) out.push_back(current);
  std::vector<std::string> kept;
  for (auto& r : out) if (!r.empty()) kept.push_back(r);
  return kept;
}

// an A3M's records with their headers (chai-msa.js headedRecords)
struct HeadedRecord { std::string header, row; };
inline std::vector<HeadedRecord> chaiHeadedRecords(const std::string& text) {
  std::vector<HeadedRecord> out;
  bool open = false;
  HeadedRecord current;
  size_t at = 0;
  while (at <= text.size()) {
    size_t end = text.find('\n', at);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(at, end - at);
    size_t a = line.find_first_not_of(" \t\r\f\v"), z = line.find_last_not_of(" \t\r\f\v");
    line = a == std::string::npos ? "" : line.substr(a, z - a + 1);
    if (!line.empty() && line[0] == '>') { if (open) out.push_back(current); current = {line.substr(1), ""}; open = true; }
    else { open = true; current.row += line; }
    at = end + 1;
  }
  if (open) out.push_back(current);
  std::vector<HeadedRecord> kept;
  for (auto& r : out) if (!r.row.empty()) kept.push_back(r);
  return kept;
}
// chain `chain`'s (1-based) own header out of a merged one (chai-msa.js chainHeader)
inline std::string chaiChainHeader(const std::string& header, int chain) {
  if (header.rfind("chain_", 0) == 0) {
    size_t bar = header.find('|');
    bool digits = bar != std::string::npos && bar > 6;
    for (size_t k = 6; digits && k < bar; ++k) digits = std::isdigit((unsigned char)header[k]) || header[k] == '+';
    if (digits) return header.substr(bar + 1);
  }
  size_t colon = header.find(':');
  bool numbered = colon != std::string::npos && colon > 0;
  for (size_t k = 0; numbered && k < colon; ++k) numbered = std::isdigit((unsigned char)header[k]);
  if (numbered) {
    // header.split(/ (?=\d+:)/): a piece starts at the beginning or after a space followed by digits and a colon
    std::vector<std::string> parts;
    size_t from = 0;
    for (size_t k = 0; k < header.size(); ++k) {
      if (header[k] != ' ') continue;
      size_t d = k + 1;
      while (d < header.size() && std::isdigit((unsigned char)header[d])) ++d;
      if (d > k + 1 && d < header.size() && header[d] == ':') { parts.push_back(header.substr(from, k - from)); from = k + 1; }
    }
    parts.push_back(header.substr(from));
    const std::string want = std::to_string(chain) + ":";
    for (auto& p : parts) if (p.rfind(want, 0) == 0) return p.substr(want.size());
    return "";
  }
  return header;
}

struct ChaiChain { std::string sequence, kind; std::vector<std::pair<int, std::string>> modifications; };

// false, the batch untouched, where no chain has a row but its own query
inline bool applyChaiMsa(Batch& b, const std::vector<ChaiChain>& chains, const std::string& paired,
                         const std::string& unpaired, bool coversNucleic) {
  const int tokens = b.tokens, sequences = b.sequences;
  std::vector<int> columnChains, offsets, lengths;
  int column = 0;
  for (size_t c = 0; c < chains.size(); ++c)
    if (coversNucleic || chains[c].kind == "protein") {
      columnChains.push_back((int)c); offsets.push_back(column); lengths.push_back((int)chains[c].sequence.size());
      column += (int)chains[c].sequence.size();
    }
  auto chainRows = [&](int at) {
    long width = offsets.back() + lengths.back();
    std::vector<ChaiRow> out;
    auto slice = [&](const std::string& row) {
      ChaiRow full = chaiTokenise(row, width), one;
      one.tokens.assign(full.tokens.begin() + offsets[at], full.tokens.begin() + offsets[at] + lengths[at]);
      one.deletions.assign(full.deletions.begin() + offsets[at], full.deletions.begin() + offsets[at] + lengths[at]);
      return one;
    };
    // (a chain's paired rows only where it has a paired alignment: chai-msa.js)
    std::vector<ChaiRow> pairedRows;
    for (auto& row : chaiRecords(paired)) pairedRows.push_back(slice(row));
    bool own = false;
    for (size_t r = 1; r < pairedRows.size() && !own; ++r)
      own = std::any_of(pairedRows[r].tokens.begin(), pairedRows[r].tokens.end(), [](int t) { return t != MSA_GAP; });
    if (own) out.insert(out.end(), pairedRows.begin(), pairedRows.end());
    for (auto& row : chaiRecords(unpaired)) {
      ChaiRow one = slice(row);
      if (std::any_of(one.tokens.begin(), one.tokens.end(), [](int t) { return t != MSA_GAP; })) out.push_back(one);
    }
    return out;
  };
  std::vector<std::vector<ChaiRow>> rowsOf(chains.size());
  std::vector<char> homologs(chains.size(), 0);
  bool any = false;
  for (size_t c = 0; c < chains.size(); ++c) {
    auto at = std::find(columnChains.begin(), columnChains.end(), (int)c);
    if (chains[c].kind == "protein" && at != columnChains.end()) rowsOf[c] = chainRows((int)(at - columnChains.begin()));
    std::vector<int> query = chaiTokenise(chains[c].sequence).tokens;
    for (auto& r : rowsOf[c]) if (r.tokens != query) { homologs[c] = 1; any = true; break; }
  }
  if (!any) return false;
  // each row's database by its a3m header (chai-msa.js msaSource): 2 UniRef, 0 otherwise, the query 4
  b.msaSource.assign(sequences, 2);
  b.msaSource[0] = 4;
  {
    long width = offsets.back() + lengths.back();
    for (size_t c = 0; c < chains.size(); ++c) {
      auto it = std::find(columnChains.begin(), columnChains.end(), (int)c);
      if (!homologs[c] || it == columnChains.end()) continue;
      int at = (int)(it - columnChains.begin());
      std::map<std::vector<int>, int> want;
      for (const std::string* block : {&paired, &unpaired}) {
        auto records = chaiHeadedRecords(*block);
        for (size_t r = 1; r < records.size(); ++r) {
          ChaiRow full = chaiTokenise(records[r].row, width);
          std::vector<int> key(full.tokens.begin() + offsets[at], full.tokens.begin() + offsets[at] + lengths[at]);
          if (!want.count(key)) want[key] = chaiChainHeader(records[r].header, (int)c + 1).rfind("UniRef", 0) == 0 ? 2 : 0;
        }
      }
      std::vector<int> columns;
      for (int t = 0; t < tokens; ++t) {
        if (b.asymId[t] != (int)c + 1) continue;
        if (columns.empty() || b.residueIndex[columns.back()] != b.residueIndex[t]) columns.push_back(t);
      }
      bool first = true;
      for (int row = 0; row < sequences; ++row) {
        bool live = false;
        for (int t : columns) live = live || b.msaMask[(size_t)row * tokens + t] != 0;
        if (!live) continue;
        if (first) { first = false; continue; }
        std::vector<int> key; bool residue = false;
        for (int t : columns) { int v = b.msa[(size_t)row * tokens + t]; key.push_back(v); residue = residue || v != MSA_GAP; }
        auto hit = want.find(key);
        if (hit != want.end() && residue) b.msaSource[row] = hit->second;
      }
    }
  }
  for (size_t c = 0; c < chains.size(); ++c) {
    std::vector<int> selected;
    for (int t = 0; t < tokens; ++t) if (b.asymId[t] == (int)c + 1) selected.push_back(t);
    if (selected.empty()) continue;
    ChaiRow query = chaiTokenise(chains[c].sequence);
    if (homologs[c]) {
      std::set<std::vector<int>> seen;
      std::vector<const ChaiRow*> kept;
      if (seen.insert(query.tokens).second) kept.push_back(&query);
      for (auto& r : rowsOf[c]) if (seen.insert(r.tokens).second) kept.push_back(&r);
      size_t length = query.tokens.size();
      std::vector<std::vector<float>> profile(length, std::vector<float>(RESTYPES));
      std::vector<float> deletionMean(length);
      for (size_t col = 0; col < length; ++col) {
        std::vector<int> counts(RESTYPES, 0);
        long deletions = 0;
        for (auto* r : kept) { counts[r->tokens[col]] += 1; deletions += r->deletions[col]; }
        long total = 0;
        for (auto& k : counts) { k %= 256; total += k; }
        for (int k = 0; k < RESTYPES; ++k) profile[col][k] = (float)((double)counts[k] / (double)std::max(total, 1L));
        deletionMean[col] = (float)((double)deletions / (double)kept.size());
      }
      for (int t : selected) {
        int col = b.residueIndex[t] - 1;
        for (int k = 0; k < RESTYPES; ++k) b.profile[(size_t)t * RESTYPES + k] = profile[col][k];
        b.deletionMean[t] = deletionMean[col];
      }
      continue;
    }
    std::string text;
    for (size_t i = 0; i < chains[c].sequence.size(); ++i) {
      const std::string* code = nullptr;
      for (auto& m : chains[c].modifications) if (m.first == (int)i + 1) code = &m.second;
      text += code ? "(" + *code + ")" : std::string(1, chains[c].sequence[i]);
    }
    std::vector<int> context = chaiTokenise(text).tokens;
    for (int t : selected) {
      int q = context[std::min(b.residueIndex[t] - 1, (int)context.size() - 1)];
      b.msa[t] = q;
      for (int row = 1; row < sequences; ++row) {
        b.msa[(size_t)row * tokens + t] = 31;
        b.msaMask[(size_t)row * tokens + t] = 0;
      }
      for (int row = 0; row < sequences; ++row) b.deletionMatrix[(size_t)row * tokens + t] = 0;
      for (int k = 0; k < RESTYPES; ++k) b.profile[(size_t)t * RESTYPES + k] = k == q ? 1.f : 0.f;
      b.deletionMean[t] = 0;
    }
  }
  // a ligand token: UNK in the query row, masked padding in the others, UNK's profile (chai-msa.js)
  for (int t = 0; t < tokens; ++t) {
    if (t >= (int)b.isLigand.size() || !b.isLigand[t] || !b.seqMask[t]) continue;
    b.msa[t] = 20;
    for (int row = 1; row < sequences; ++row) b.msa[(size_t)row * tokens + t] = 31;
    for (int k = 0; k < RESTYPES; ++k) b.profile[(size_t)t * RESTYPES + k] = k == 20 ? 1.f : 0.f;
  }
  std::set<std::vector<int>> seen;
  for (int row = 0; row < sequences; ++row) {
    bool live = false;
    for (int t = 0; t < tokens && !live; ++t) live = b.msaMask[(size_t)row * tokens + t] != 0;
    if (!live) continue;
    std::vector<int> key(b.msa.begin() + (size_t)row * tokens, b.msa.begin() + (size_t)(row + 1) * tokens);
    if (!seen.insert(key).second) std::fill(b.msaMask.begin() + (size_t)row * tokens, b.msaMask.begin() + (size_t)(row + 1) * tokens, 0.f);
  }
  return true;
}

}  // namespace lf
