// `applyContig`: a contig against a real input structure.
//
// 🔴 A CONTIG HAS THREE KINDS OF TOKEN AND THE SPEC PATH ONLY EVER SAW ONE. `name:min-maxP` is a
// named expandable segment and is all a pure design job contains; `A16-35` PRESERVES residues of
// an input structure, and `/0` breaks the chain. The preserved kind is what needs the mmCIF
// reader, which is why it was refused by name until the reader existed.
//
// 🔴 AND THE TWO INSERTION RULES ARE NOT SYMMETRIC. An expandable token with a preserved token
// after it is inserted BEFORE that token's first atom; one with nothing after it is appended
// after the run's last atom; and one in a run with no preserved tokens at all starts a NEW
// chain. Reading it as "insert where the token sits" gets the first two the same and the third
// wrong.
#pragma once
#include "atoms.h"

namespace rfd4p {
namespace contig {

using std::string;
using std::vector;

struct Token {
  enum Kind { CHAINBREAK, EXPANDABLE, PRESERVED } kind = EXPANDABLE;
  string name;                 // a named expandable segment, else empty
  int segMin = 0, segMax = 0;
  int chainType = -1;          // -1 where the token names none
  string chainId;              // PRESERVED
  int start = 0, end = 0;      // PRESERVED
};

inline int suffixChainType(char c) {
  switch (c) {
    case 'P': return tables::CT_POLYPEPTIDE_L;
    case 'R': return tables::CT_RNA;
    case 'D': return tables::CT_DNA;
    default: return -2;
  }
}

inline void parseLengthAndSuffix(const string& body, const string& token, Token& out) {
  string b = body;
  if (!b.empty()) {
    const int ct = suffixChainType(b.back());
    if (ct != -2) { out.chainType = ct; b.pop_back(); }
  }
  if (b.empty()) {
    fprintf(stderr, "Malformed expandable-segment token: %s\n", token.c_str());
    exit(1);
  }
  for (const char c : b) {
    if (!isdigit((unsigned char)c) && c != '-') {
      fprintf(stderr, "Malformed expandable-segment token: %s\n", token.c_str());
      exit(1);
    }
  }
  const size_t dash = b.find('-');
  out.segMin = atoi(b.substr(0, dash).c_str());
  out.segMax = dash == string::npos ? out.segMin : atoi(b.substr(dash + 1).c_str());
}

inline Token parseToken(const string& tok) {
  Token out;
  if (tok.empty()) {
    fprintf(stderr, "Empty contig token (check for stray/trailing commas).\n");
    exit(1);
  }
  if (tok == "/0") { out.kind = Token::CHAINBREAK; return out; }
  const size_t colon = tok.find(':');
  if (colon != string::npos) {
    out.kind = Token::EXPANDABLE;
    out.name = tok.substr(0, colon);
    if (out.name.empty() || out.name == "<M>") {
      fprintf(stderr, "Invalid named expandable-segment token: %s\n", tok.c_str());
      exit(1);
    }
    parseLengthAndSuffix(tok.substr(colon + 1), tok, out);
    return out;
  }
  if (isalpha((unsigned char)tok[0])) {
    out.kind = Token::PRESERVED;
    size_t i = 0;
    while (i < tok.size() && isalpha((unsigned char)tok[i])) ++i;
    out.chainId = tok.substr(0, i);
    const size_t dash = tok.find('-', i);
    const string startStr = tok.substr(i, dash == string::npos ? string::npos : dash - i);
    if (startStr.empty()) {
      fprintf(stderr, "Malformed preserved-segment token: %s\n", tok.c_str());
      exit(1);
    }
    for (const char c : startStr) {
      if (!isdigit((unsigned char)c)) {
        fprintf(stderr, "Malformed preserved-segment token: %s\n", tok.c_str());
        exit(1);
      }
    }
    out.start = atoi(startStr.c_str());
    out.end = out.start;
    if (dash != string::npos) {
      const string endStr = tok.substr(dash + 1);
      if (endStr.empty()) {
        fprintf(stderr, "Malformed preserved-segment token: %s\n", tok.c_str());
        exit(1);
      }
      for (const char c : endStr) {
        if (!isdigit((unsigned char)c)) {
          fprintf(stderr, "Malformed preserved-segment token: %s\n", tok.c_str());
          exit(1);
        }
      }
      out.end = atoi(endStr.c_str());
    }
    if (out.end < out.start) {
      fprintf(stderr, "Preserved segment %s has end < start (%d < %d).\n", tok.c_str(), out.end,
              out.start);
      exit(1);
    }
    return out;
  }
  out.kind = Token::EXPANDABLE;
  parseLengthAndSuffix(tok, tok, out);
  return out;
}

inline vector<Token> parseContig(const string& contig) {
  vector<Token> out;
  size_t at = 0;
  while (at <= contig.size()) {
    const size_t comma = contig.find(',', at);
    string tok = contig.substr(at, comma == string::npos ? string::npos : comma - at);
    at = comma == string::npos ? contig.size() + 1 : comma + 1;
    while (!tok.empty() && isspace((unsigned char)tok.front())) tok.erase(tok.begin());
    while (!tok.empty() && isspace((unsigned char)tok.back())) tok.pop_back();
    if (tok.empty()) continue;
    out.push_back(parseToken(tok));
  }
  return out;
}

/**
 * `findFlankingBondAtomIdxs`: the atom a sentinel's previous residue leaves from and the one its
 * next residue enters by.
 *
 * 🔴 A ONE-ATOM NEIGHBOUR THAT IS ITSELF EXPANDABLE COUNTS WHATEVER IT IS CALLED. Two adjacent
 * sentinels have no `C` or `N` between them - they are `UNK` - so the rule is "the
 * polymerization atom, OR the single atom of a one-atom expandable residue", and without the
 * second half a run of two expandable segments comes out unbonded.
 */
inline std::pair<int, int> flankingBondAtoms(const atoms::AtomArray& a, int sentinelIdx) {
  const int ct = a.site.chainType[sentinelIdx];
  const std::pair<string, string> p = cif::polymerizationAtoms("", ct);
  if (p.first.empty() || p.second.empty()) return {-1, -1};
  const string& termName = p.first;
  const string& startName = p.second;
  const string& chainVal = a.ids.chainIid.empty() ? a.site.chainId[sentinelIdx]
                                                  : a.ids.chainIid[sentinelIdx];
  auto chainAt = [&](int i) -> const string& {
    return a.ids.chainIid.empty() ? a.site.chainId[i] : a.ids.chainIid[i];
  };
  const vector<int> rb = cif::residueSegments(a.site);
  int sentinelRes = 0;
  for (size_t r = 0; r + 1 < rb.size(); ++r) {
    if (sentinelIdx >= rb[r] && sentinelIdx < rb[r + 1]) { sentinelRes = (int)r; break; }
  }
  auto segIt = a.ann.ints.find("annotation_expsegmin_1_residue");
  auto expandable = [&](int i) {
    return segIt != a.ann.ints.end() && i < (int)segIt->second.size() && segIt->second[i] != -1;
  };
  int prevTerminal = -1;
  if (sentinelRes > 0) {
    const int ps = rb[sentinelRes - 1], pe = rb[sentinelRes];
    if (chainAt(ps) == chainVal) {
      for (int i = ps; i < pe; ++i) {
        if (a.site.atomName[i] == termName || (pe - ps == 1 && expandable(i))) {
          prevTerminal = i;
          break;
        }
      }
    }
  }
  int nextStart = -1;
  if (sentinelRes < (int)rb.size() - 2) {
    const int ns = rb[sentinelRes + 1], ne = rb[sentinelRes + 2];
    if (chainAt(ns) == chainVal) {
      for (int i = ns; i < ne; ++i) {
        if (a.site.atomName[i] == startName || (ne - ns == 1 && expandable(i))) {
          nextStart = i;
          break;
        }
      }
    }
  }
  return {prevTerminal, nextStart};
}

/** `installSentinelBonds`: the sentinel takes over the bond its two neighbours had. */
inline void installSentinelBonds(atoms::AtomArray& a, int sentinelIdx) {
  const auto [prevIdx, nextIdx] = flankingBondAtoms(a, sentinelIdx);
  if (prevIdx < 0 && nextIdx < 0) return;
  if (prevIdx >= 0 && nextIdx >= 0) {
    const int u0 = std::min(prevIdx, nextIdx), v0 = std::max(prevIdx, nextIdx);
    a.bonds.erase(std::remove_if(a.bonds.begin(), a.bonds.end(), [&](const cif::Bond& b) {
                    return std::min(b.u, b.v) == u0 && std::max(b.u, b.v) == v0;
                  }),
                  a.bonds.end());
  }
  auto hasBond = [&](int x, int y) {
    const int u0 = std::min(x, y), v0 = std::max(x, y);
    for (const cif::Bond& b : a.bonds) {
      if (std::min(b.u, b.v) == u0 && std::max(b.u, b.v) == v0) return true;
    }
    return false;
  };
  if (prevIdx >= 0 && !hasBond(prevIdx, sentinelIdx)) {
    a.bonds.push_back(cif::Bond{std::min(prevIdx, sentinelIdx), std::max(prevIdx, sentinelIdx),
                                tables::BT_SINGLE});
  }
  if (nextIdx >= 0 && !hasBond(sentinelIdx, nextIdx)) {
    a.bonds.push_back(cif::Bond{std::min(sentinelIdx, nextIdx), std::max(sentinelIdx, nextIdx),
                                tables::BT_SINGLE});
  }
}

struct InsertWhere {
  int atomIdx = -1;
  bool insertBefore = true;
  bool newChain = false;
  string newChainId;
  int newChainType = -1;
};

/**
 * `insertExpandableSegment`: a one-atom `<M>`-to-be stub spliced into the array.
 *
 * 🔴 THE SENTINEL'S RESIDUE NUMBER IS CHOSEN TO FIT A HOLE WHERE THERE IS ONE. If the residues
 * either side are already two apart the sentinel takes the number between them and NOTHING
 * shifts; only where they are adjacent does it take `prev + 1` and push every later residue of
 * that chain up by one. A port that always shifted would renumber a chain that had room.
 * Measured: that branch is NOT reachable through `applyContig`, because `buildRunPreserved`
 * renumbers every preserved segment contiguously from 1 before anything is inserted, so there
 * is never a hole to fit - forcing the always-shift arm leaves all seventeen cases green. It
 * matters to a caller that splices into an array with its own numbering, which is why it is
 * ported rather than simplified away, and recorded here rather than left looking checked.
 *
 * 🔴 AND THE BONDS ARE RE-SPLICED BY HAND, NOT BY `slice`. Slicing drops a bond whose other end
 * is in the other piece, which is every bond across the insertion point - so the result's bond
 * list is rebuilt from the ORIGINAL with every index at or past the insertion shifted by one.
 */
inline atoms::AtomArray insertExpandableSegment(const atoms::AtomArray* in, int segMin, int segMax,
                                                const InsertWhere& where) {
  const size_t n = in ? in->n() : 0;
  if (n == 0 && !where.newChain) {
    fprintf(stderr, "Cannot insert an expandable segment into an empty array without a new chain\n");
    exit(1);
  }
  atoms::AtomArray atoms_ = in ? *in : atoms::AtomArray{};
  const bool haveIid = !atoms_.ids.chainIid.empty();
  auto chainAt = [&](int i) -> const string& {
    return haveIid ? atoms_.ids.chainIid[i] : atoms_.site.chainId[i];
  };

  int insertPosition = 0, sourceAtomIdx = -1, sentinelResId = 1, chainType = -1;
  string chainId, chainIdentifier;
  bool needsShift = false;
  if (where.newChain) {
    insertPosition = (int)n;
    sourceAtomIdx = n > 0 ? (int)n - 1 : -1;
    chainId = where.newChainId;
    sentinelResId = 1;
    chainIdentifier = chainId;
    chainType = where.newChainType >= 0 ? where.newChainType : tables::CT_POLYPEPTIDE_L;
  } else {
    insertPosition = where.insertBefore ? where.atomIdx : where.atomIdx + 1;
    sourceAtomIdx = where.atomIdx;
    chainId = atoms_.site.chainId[where.atomIdx];
    chainIdentifier = chainAt(where.atomIdx);
    vector<int> chainIndices;
    for (size_t i = 0; i < n; ++i) if (chainAt((int)i) == chainIdentifier) chainIndices.push_back((int)i);
    int nBefore = 0;
    for (const int idx : chainIndices) if (idx < insertPosition) ++nBefore;
    const int nAfter = (int)chainIndices.size() - nBefore;
    if (nBefore > 0) {
      const int prevResId = atoms_.site.resId[chainIndices[nBefore - 1]];
      if (nAfter > 0) {
        const int nextResId = atoms_.site.resId[chainIndices[nBefore]];
        if (nextResId > prevResId + 1) { sentinelResId = nextResId - 1; needsShift = false; }
        else { sentinelResId = prevResId + 1; needsShift = true; }
      } else { sentinelResId = prevResId + 1; needsShift = true; }
    } else {
      const int firstResId = atoms_.site.resId[chainIndices[0]];
      if (firstResId > 0) { sentinelResId = firstResId - 1; needsShift = false; }
      else { sentinelResId = firstResId; needsShift = true; }
    }
  }

  for (const char* key : {"annotation_expsegmin_1_residue", "annotation_expsegmax_1_residue",
                          "_expseg_uid"}) {
    if (!atoms_.ann.ints.count(key)) atoms_.ann.ints[key].assign(n, -1);
  }
  int maxUid = -1;
  for (const int v : atoms_.ann.ints["_expseg_uid"]) maxUid = std::max(maxUid, v);
  const int segUid = maxUid + 1;

  atoms::AtomArray sentinel;
  if (sourceAtomIdx >= 0) {
    sentinel = atoms::slice(atoms_, vector<int>{sourceAtomIdx});
  } else {
    sentinel.site.n = 1;
    atoms::forEachColumn(sentinel, [](auto& column) {
      using V = std::decay_t<decltype(column)>;
      if constexpr (std::is_same_v<V, vector<string>>) column.assign(1, string());
      else if constexpr (std::is_same_v<V, vector<double>>) column.assign(1, 0.0);
      else column.assign(1, 0);
    });
    sentinel.site.labelEntityId[0] = 1;
    sentinel.site.atomId[0] = 1;
    sentinel.site.bFactor[0] = NAN;
    sentinel.site.labelAltId[0] = ".";
  }
  sentinel.site.n = 1;
  sentinel.site.chainId[0] = chainId;
  sentinel.site.resName[0] = std::to_string(segMin) + "-" + std::to_string(segMax);
  sentinel.site.atomName[0] = "UNK";
  sentinel.site.element[0] = "X";
  sentinel.site.x[0] = sentinel.site.y[0] = sentinel.site.z[0] = NAN;
  sentinel.site.hetero[0] = 1;
  sentinel.site.resId[0] = sentinelResId;
  sentinel.site.authSeqId[0] = sentinelResId;
  sentinel.site.insCode[0] = "";
  sentinel.site.occupancy[0] = 0.0;
  sentinel.site.bFactor[0] = NAN;
  sentinel.site.charge[0] = 0;
  sentinel.site.atomicNumber[0] = 0;
  sentinel.site.isAromatic[0] = 0;
  sentinel.site.isLeavingAtom[0] = 0;
  sentinel.site.chemCompType[0] = "NON-POLYMER";
  if (chainType >= 0) {
    sentinel.site.chainType[0] = chainType;
    sentinel.site.isPolymer[0] = chainType <= tables::CT_RNA;
  }
  if (sourceAtomIdx >= 0 && chainId != atoms_.site.chainId[sourceAtomIdx]) {
    int maxEid = 0;
    for (const int v : atoms_.site.labelEntityId) maxEid = std::max(maxEid, v);
    sentinel.site.labelEntityId[0] = maxEid + 1;
  }
  sentinel.bonds.clear();

  // 🔴 EVERY OTHER ANNOTATION THE SENTINEL INHERITED IS RESET TO ITS DEFAULT. It was sliced from
  // a real atom, so without this it carries that atom's conditions - a stub standing in for
  // residues nobody has specified would arrive conditioned like its neighbour.
  for (auto& [key, v] : sentinel.ann.ints) {
    if (key == "annotation_expsegmin_1_residue" || key == "annotation_expsegmax_1_residue" ||
        key == "_expseg_uid") {
      continue;
    }
    auto spec = tables::ANNOTATIONS_1BODY.find(key);
    v.assign(spec == tables::ANNOTATIONS_1BODY.end() ? 1 : (size_t)spec->second.cols,
             spec == tables::ANNOTATIONS_1BODY.end() ? 0 : (int)spec->second.fallback);
  }
  for (auto& [key, v] : sentinel.ann.floats) {
    auto spec = tables::ANNOTATIONS_1BODY.find(key);
    const double fill = spec == tables::ANNOTATIONS_1BODY.end() ? NAN : spec->second.fallback;
    v.assign(spec == tables::ANNOTATIONS_1BODY.end() ? 1 : (size_t)spec->second.cols, fill);
  }
  for (auto& [key, v] : sentinel.ann.strings) {
    auto spec = tables::ANNOTATIONS_1BODY.find(key);
    v.assign(1, spec == tables::ANNOTATIONS_1BODY.end() ? string() : string(spec->second.text));
  }
  sentinel.ann.ints["annotation_expsegmin_1_residue"] = {segMin};
  sentinel.ann.ints["annotation_expsegmax_1_residue"] = {segMax};
  sentinel.ann.ints["_expseg_uid"] = {segUid};

  vector<atoms::AtomArray> pieces;
  if (insertPosition > 0) pieces.push_back(atoms::slice(atoms_, 0, insertPosition));
  pieces.push_back(sentinel);
  if (insertPosition < (int)n) pieces.push_back(atoms::slice(atoms_, insertPosition, (int)n));
  atoms::AtomArray result = atoms::concat(pieces);

  result.bonds.clear();
  for (const cif::Bond& b : atoms_.bonds) {
    result.bonds.push_back(cif::Bond{b.u >= insertPosition ? b.u + 1 : b.u,
                                     b.v >= insertPosition ? b.v + 1 : b.v, b.type});
  }
  if (!where.newChain) {
    if (needsShift) {
      const bool haveIid2 = !result.ids.chainIid.empty();
      for (size_t i = insertPosition + 1; i < result.n(); ++i) {
        const string& cv = haveIid2 ? result.ids.chainIid[i] : result.site.chainId[i];
        if (cv == chainIdentifier && result.site.resId[i] >= sentinelResId) ++result.site.resId[i];
      }
    }
    installSentinelBonds(result, insertPosition);
  }
  result.ids = cif::computeIds(result.site, result.bonds);
  return result;
}

/**
 * The residue id an expandable segment's atoms carry in `originalResId` - they came from no
 * residue of the input.
 *
 * 🔴 IT IS `-(2**31)`, NOT -1. A real mmCIF can and does carry negative residue numbers, so a
 * sentinel of -1 would collide with one - and the collision is silent, because a selection
 * written against the input's numbering would then match the stub. Guessed as -1 here first and
 * the gate caught it on fifteen structures, one entry each.
 */
inline constexpr int EXPANDABLE_SEGMENT_RES_ID = INT32_MIN;

struct Applied {
  atoms::AtomArray atomArray;
  vector<int> originalResId;
};

/**
 * `applyContig`: the input structure subsetted to the preserved segments, with the expandable
 * ones spliced in.
 *
 * 🔴 A RUN WITH NO PRESERVED SEGMENT IS HELD BACK UNTIL ONE WITH ONE HAS BEEN PLACED, AND THEN
 * MOVED IN FRONT OF IT. `insertExpandableSegment` cannot start a chain in an empty array except
 * through its `newChain` path, so a leading all-expandable run is queued, processed after the
 * first run that brings atoms, and then the whole array is REORDERED to put it back where the
 * contig asked. Dropping the reorder gives the right atoms in the wrong chains.
 *
 * 🔴 AND A PRESERVED SEGMENT IS RENUMBERED FROM 1 WITHIN ITS RUN while `originalResId` keeps
 * what it had - that is the only record of where a motif came from, and `applyConditions` reads
 * it to resolve a selection written against the input's own numbering.
 */
inline Applied applyContig(const atoms::AtomArray& in, const string& contigStr) {
  const vector<Token> tokens = parseContig(contigStr);
  vector<vector<Token>> runs{{}};
  for (const Token& t : tokens) {
    if (t.kind == Token::CHAINBREAK) runs.push_back({});
    else runs.back().push_back(t);
  }
  vector<vector<Token>> nonEmpty;
  for (auto& r : runs) if (!r.empty()) nonEmpty.push_back(r);
  if (nonEmpty.empty()) {
    fprintf(stderr, "Contig string produced no chain runs.\n");
    exit(1);
  }
  // 🔴 UPSTREAM'S CHECKS, IN UPSTREAM'S WORDS (`contig_spec.py`; spec.js `applyContig` has the
  // same): a duplicated segment name, overlapping preserved segments, a range past its chain, a
  // suffix disagreeing with its anchor, and an input with nothing to preserve.
  {
    std::map<string, int> seenNames;
    std::set<string> dupes;
    for (const Token& t : tokens) {
      if (t.kind == Token::EXPANDABLE && !t.name.empty() && seenNames[t.name]++) dupes.insert(t.name);
    }
    if (!dupes.empty()) {
      string list;
      for (const string& d : dupes) list += (list.empty() ? "" : ",") + ("\"" + d + "\"");
      fprintf(stderr, "Duplicate expandable-segment name(s) in contig: [%s]\n", list.c_str());
      exit(1);
    }
    std::map<string, vector<const Token*>> byChain;
    for (const auto& run : nonEmpty) {
      for (const Token& t : run) if (t.kind == Token::PRESERVED) byChain[t.chainId].push_back(&t);
    }
    for (auto& [chainId, segs] : byChain) {
      std::stable_sort(segs.begin(), segs.end(),
                       [](const Token* a, const Token* b) { return a->start < b->start; });
      for (size_t k = 1; k < segs.size(); ++k) {
        if (segs[k]->start <= segs[k - 1]->end) {
          fprintf(stderr, "Preserved segments on chain '%s' overlap: %s%d-%d and %s%d-%d both "
                          "reference some of the same residues.\n", chainId.c_str(),
                  chainId.c_str(), segs[k - 1]->start, segs[k - 1]->end, chainId.c_str(),
                  segs[k]->start, segs[k]->end);
          exit(1);
        }
      }
    }
    // Upstream requires an input; with none this port realises the contig into an empty
    // structure, so only an input with nothing preserved is refused.
    if (in.n() > 0 && byChain.empty()) {
      fprintf(stderr, "Contig has no preserved segments -- condition_spec's `input` structure "
                      "exists only to be referenced by preserved segments, so a contig that "
                      "generates everything from scratch doesn't need an `input` file at all.\n");
      exit(1);
    }
  }

  std::set<string> claimed;
  for (const auto& run : nonEmpty) {
    for (const Token& t : run) if (t.kind == Token::PRESERVED) claimed.insert(t.chainId);
  }
  // `createChainIdGenerator`: A..Z then AA..ZZ, skipping anything a preserved segment claims.
  vector<string> synth;
  for (char c = 'A'; c <= 'Z'; ++c) {
    const string s(1, c);
    if (!claimed.count(s)) synth.push_back(s);
  }
  for (char c = 'A'; c <= 'Z' && synth.size() < 702; ++c) {
    for (char d = 'A'; d <= 'Z'; ++d) {
      const string s = string(1, c) + d;
      if (!claimed.count(s)) synth.push_back(s);
    }
  }
  size_t synthAt = 0;

  struct Run { vector<Token> tokens; string chainId; };
  vector<Run> chainRuns;
  std::set<string> seen;
  for (const auto& runTokens : nonEmpty) {
    string chainId;
    std::set<string> cids;
    for (const Token& t : runTokens) if (t.kind == Token::PRESERVED) cids.insert(t.chainId);
    if (!cids.empty()) {
      if (cids.size() > 1) {
        fprintf(stderr, "An output chain mixes preserved segments from multiple chains\n");
        exit(1);
      }
      chainId = *cids.begin();
    } else {
      if (synthAt >= synth.size()) { fprintf(stderr, "ran out of chain ids\n"); exit(1); }
      chainId = synth[synthAt++];
    }
    if (seen.count(chainId)) {
      fprintf(stderr, "Contig produces two separate output chains both named %s\n", chainId.c_str());
      exit(1);
    }
    seen.insert(chainId);
    chainRuns.push_back({runTokens, chainId});
  }

  struct Prepared {
    bool have = false;
    atoms::AtomArray preserved;
    vector<int> originalResId;
    std::map<int, std::pair<int, int>> tokenOffsets;
  };
  auto buildRunPreserved = [&](const Run& run) {
    Prepared out;
    vector<atoms::AtomArray> pieces;
    int cursor = 0, nextResId = 1;
    for (size_t tokIdx = 0; tokIdx < run.tokens.size(); ++tokIdx) {
      const Token& tok = run.tokens[tokIdx];
      if (tok.kind != Token::PRESERVED) continue;
      vector<int> selected;
      for (size_t i = 0; i < in.n(); ++i) {
        if (in.site.chainId[i] == tok.chainId && in.site.resId[i] >= tok.start &&
            in.site.resId[i] <= tok.end) {
          selected.push_back((int)i);
        }
      }
      if (selected.empty()) {
        fprintf(stderr, "Preserved segment '%s%d-%d' matched no atoms.\n", tok.chainId.c_str(),
                tok.start, tok.end);
        exit(1);
      }
      int chainMin = INT32_MAX, chainMax = INT32_MIN;
      for (size_t i = 0; i < in.n(); ++i) {
        if (in.site.chainId[i] != tok.chainId) continue;
        chainMin = std::min(chainMin, in.site.resId[i]);
        chainMax = std::max(chainMax, in.site.resId[i]);
      }
      if (tok.start < chainMin || tok.end > chainMax) {
        fprintf(stderr, "Preserved segment '%s%d-%d' runs outside chain %s, whose residues are "
                        "%d-%d.\n", tok.chainId.c_str(), tok.start, tok.end, tok.chainId.c_str(),
                chainMin, chainMax);
        exit(1);
      }
      atoms::AtomArray segment = atoms::slice(in, selected);
      const vector<int> rb = cif::residueSegments(segment.site);
      for (size_t r = 0; r + 1 < rb.size(); ++r) {
        for (int i = rb[r]; i < rb[r + 1]; ++i) segment.site.resId[i] = nextResId + (int)r;
      }
      for (const int idx : selected) out.originalResId.push_back(in.site.resId[idx]);
      out.tokenOffsets[(int)tokIdx] = {cursor, cursor + (int)segment.n()};
      cursor += (int)segment.n();
      nextResId += (int)rb.size() - 1;
      pieces.push_back(std::move(segment));
    }
    if (!pieces.empty()) { out.preserved = atoms::concat(pieces); out.have = true; }
    return out;
  };

  auto resolveRunChainType = [&](const Run& run) {
    int anchor = -1;
    for (const Token& t : run.tokens) {
      if (t.kind != Token::PRESERVED) continue;
      std::set<int> types;
      for (size_t i = 0; i < in.n(); ++i) {
        if (in.site.chainId[i] == t.chainId) types.insert(in.site.chainType[i]);
      }
      if (types.size() > 1) {
        fprintf(stderr, "Input chain '%s' contains multiple ChainTypes -- a contig run anchored on "
                        "this chain can't unambiguously default an unsuffixed expandable segment's "
                        "type. Split this chain into separately-named chains upstream.\n",
                t.chainId.c_str());
        exit(1);
      }
      if (types.size() == 1) anchor = *types.begin();
      break;
    }
    std::set<int> requested;
    for (const Token& t : run.tokens) {
      if (t.kind == Token::EXPANDABLE && t.chainType >= 0) requested.insert(t.chainType);
    }
    if (requested.size() > 1) {
      fprintf(stderr, "Chain run requests conflicting chain types.\n");
      exit(1);
    }
    if (requested.empty()) return anchor >= 0 ? anchor : (int)tables::CT_POLYPEPTIDE_L;
    const int chainType = *requested.begin();
    if (anchor >= 0 && chainType != anchor) {
      fprintf(stderr, "An expandable segment explicitly specifies chain type %d, which disagrees "
                      "with its run's anchor chain type %d.\n", chainType, anchor);
      exit(1);
    }
    return chainType;
  };

  bool haveAssembled = false;
  atoms::AtomArray assembled;
  vector<int> originalResId;

  auto processRun = [&](const Run& run, const Prepared& prep) {
    const int chainType = resolveRunChainType(run);
    const int runStart = haveAssembled ? (int)assembled.n() : 0;
    if (prep.have) {
      atoms::AtomArray add = prep.preserved;
      if (haveAssembled && assembled.ann.ints.count("_expseg_uid") &&
          !add.ann.ints.count("_expseg_uid")) {
        add.ann.ints["_expseg_uid"].assign(add.n(), -1);
      }
      assembled = haveAssembled ? atoms::concat({assembled, add}) : add;
      haveAssembled = true;
      originalResId.insert(originalResId.end(), prep.originalResId.begin(),
                           prep.originalResId.end());
    }
    int runTail = runStart + (prep.have ? (int)prep.preserved.n() : 0);
    std::map<int, std::pair<int, int>> live;
    for (const auto& [idx, se] : prep.tokenOffsets) {
      live[idx] = {runStart + se.first, runStart + se.second};
    }
    bool firstExpandable = true;
    for (size_t tokIdx = 0; tokIdx < run.tokens.size(); ++tokIdx) {
      const Token& tok = run.tokens[tokIdx];
      if (tok.kind != Token::EXPANDABLE) continue;
      int nextIdx = -1;
      for (size_t j = tokIdx + 1; j < run.tokens.size(); ++j) {
        if (run.tokens[j].kind == Token::PRESERVED) { nextIdx = (int)j; break; }
      }
      int insertPosition = 0;
      if (nextIdx >= 0) {
        const int atomIdx = live[nextIdx].first;
        assembled = insertExpandableSegment(&assembled, tok.segMin, tok.segMax,
                                            InsertWhere{atomIdx, true, false, "", -1});
        insertPosition = atomIdx;
      } else if (!live.empty()) {
        const int atomIdx = runTail - 1;
        assembled = insertExpandableSegment(&assembled, tok.segMin, tok.segMax,
                                            InsertWhere{atomIdx, false, false, "", -1});
        insertPosition = atomIdx + 1;
      } else if (firstExpandable) {
        assembled = insertExpandableSegment(haveAssembled ? &assembled : nullptr, tok.segMin,
                                            tok.segMax,
                                            InsertWhere{-1, true, true, run.chainId, chainType});
        haveAssembled = true;
        insertPosition = runTail;
      } else {
        assembled = insertExpandableSegment(&assembled, tok.segMin, tok.segMax,
                                            InsertWhere{runTail - 1, false, false, "", -1});
        insertPosition = runTail;
      }
      firstExpandable = false;
      originalResId.insert(originalResId.begin() + insertPosition, EXPANDABLE_SEGMENT_RES_ID);
      for (auto& [k, se] : live) {
        if (se.first >= insertPosition) ++se.first;
        if (se.second >= insertPosition) ++se.second;
      }
      ++runTail;
      assembled.site.resName[insertPosition] =
          tok.name.empty() ? std::to_string(tok.segMin) + "-" + std::to_string(tok.segMax)
                           : tok.name;
    }
  };

  vector<std::pair<Run, Prepared>> pending;
  for (const Run& run : chainRuns) {
    Prepared prep = buildRunPreserved(run);
    if (!haveAssembled && !prep.have) { pending.push_back({run, prep}); continue; }
    processRun(run, prep);
    const int unblocked = (int)assembled.n();
    for (auto& [prun, pprep] : pending) processRun(prun, pprep);
    if (!pending.empty()) {
      vector<int> reorder;
      for (int i = unblocked; i < (int)assembled.n(); ++i) reorder.push_back(i);
      for (int i = 0; i < unblocked; ++i) reorder.push_back(i);
      assembled = atoms::slice(assembled, reorder);
      vector<int> moved(reorder.size());
      for (size_t i = 0; i < reorder.size(); ++i) moved[i] = originalResId[reorder[i]];
      originalResId = moved;
    }
    pending.clear();
  }
  if (!haveAssembled && !pending.empty()) {
    for (auto& [prun, pprep] : pending) processRun(prun, pprep);
    pending.clear();
  }

  // Every chain's lowest residue number becomes 1.
  vector<string> uniqueChains;
  for (const string& c : assembled.site.chainId) {
    if (std::find(uniqueChains.begin(), uniqueChains.end(), c) == uniqueChains.end()) {
      uniqueChains.push_back(c);
    }
  }
  for (const string& cid : uniqueChains) {
    int minResId = INT32_MAX;
    for (size_t i = 0; i < assembled.n(); ++i) {
      if (assembled.site.chainId[i] == cid) minResId = std::min(minResId, assembled.site.resId[i]);
    }
    const int offset = 1 - minResId;
    if (offset == 0) continue;
    for (size_t i = 0; i < assembled.n(); ++i) {
      if (assembled.site.chainId[i] == cid) assembled.site.resId[i] += offset;
    }
  }
  return {assembled, originalResId};
}

}  // namespace contig
}  // namespace rfd4p
