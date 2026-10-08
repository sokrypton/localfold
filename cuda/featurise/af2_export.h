// cuda/af2/export_input.mjs, natively: AlphaFold 2's input for the CUDA port - the page's own features
// (shared/input/a3m-features.js makeA3mFeatures: the deduplicated alignment, a seeded choice of cluster centres and
// extra rows each pass, the masked centres, nearest-centre assignment, cluster profiles; query-only-features.js for
// the residue numbering and chain identity), one set a pass, and the atom37 template the page builds.
//
//   af2-featurise <out dir> (--bundle=<page bundle dir> | --weights=<export dir>) (--job=<AF3 job> | --sequence=A:B)
//                 [--a3m=<one or one a chain> [--paired-a3m=...] | --search] [--recycles=3] [--max-msa=512]
//                 [--max-extra=1024] [--seed=0] [--template=<file>:<chain>[@<chain>][+...]] [--template-search-chains=]
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "af3_export.h"
#include "jsmath.h"
#include "search.h"
#include "templates.h"

namespace lf {

// generator(seed) / randomGenerator: mulberry32
struct Mulberry {
  uint32_t state;
  explicit Mulberry(uint32_t seed) : state(seed) {}
  double operator()() {
    state += 0x6d2b79f5u;
    uint32_t v = state;
    v = (uint32_t)((int64_t)(int32_t)(v ^ (v >> 15)) * (int64_t)(int32_t)(v | 1u));
    v ^= v + (uint32_t)((int64_t)(int32_t)(v ^ (v >> 7)) * (int64_t)(int32_t)(v | 61u));
    return (double)((v ^ (v >> 14))) / 4294967296.0;
  }
};

struct Af2Pass {
  int centres = 0, extras = 0;      // msaSequences, extraSequences
  std::vector<float> msaFeat, msaMask, extraMsa, extraHasDeletion, extraDeletionValue, extraMsaMask;
};
struct Af2Features {
  int length = 0;
  std::vector<int> aatype, residueIndex, asymId, entityId, symId;
  std::vector<float> seqMask;
  bool chainAware = false;
  std::vector<Af2Pass> passes;
};

struct Af2Options {
  bool chainAware = false;
  std::vector<int> chainLengths;
  std::vector<std::string> chainSequences;
  int recycles = 3, maxMsa = 508, maxExtra = 1024;
  double seed = 0;
};

inline Af2Features makeA3mFeatures(const std::string& a3mText, const Af2Options& o) {
  A3m parsed = parseA3m(a3mText);
  // the alignment deduplicated, the query's first appearance kept
  std::vector<int> keep;
  {
    std::set<std::string> seen;
    for (int row = 0; row < parsed.depth; ++row) if (seen.insert(parsed.sequences[row]).second) keep.push_back(row);
  }
  int length = parsed.length, depth = (int)keep.size();
  static const std::string RESTYPES = "ARNDCQEGHILKMFPSTWYV";
  std::vector<uint8_t> encoded((size_t)depth * length);
  std::vector<float> deletions((size_t)depth * length);
  for (int row = 0; row < depth; ++row) {
    const std::string& seq = parsed.sequences[keep[row]];
    for (int r = 0; r < length; ++r) {
      char c = seq[r];
      size_t at = RESTYPES.find(c);
      encoded[(size_t)row * length + r] = c == '-' ? 21 : (at != std::string::npos && c != 0 ? (uint8_t)at : 20);
      deletions[(size_t)row * length + r] = (float)parsed.deletions[keep[row]][r];
    }
  }
  Af2Features F;
  F.length = length;
  // makeQueryOnlyFeatures(alignment.query): the numbering and the identity
  {
    std::string query = trimWs(parsed.sequences[0]);
    for (auto& c : query) c = (char)std::toupper((unsigned char)c);
    for (char c : query) if (c != 'X' && RESTYPES.find(c) == std::string::npos)
      throw std::runtime_error("query must contain only the 20 standard amino acids or X");
    for (char c : query) { size_t at = RESTYPES.find(c); F.aatype.push_back(at == std::string::npos ? 20 : (int)at); }
    F.seqMask.assign(length, 1);
    std::vector<int> lengths = o.chainLengths.empty() ? std::vector<int>{length} : o.chainLengths;
    int sum = 0;
    for (int l : lengths) { if (l <= 0) throw std::runtime_error("chain lengths must be positive integers"); sum += l; }
    if (sum != length) throw std::runtime_error("chain lengths sum to " + std::to_string(sum) + "; expected " + std::to_string(length));
    F.chainAware = o.chainAware && o.chainLengths.size() > 1;
    int residue = 0;
    std::vector<std::string> entities;
    std::map<int, int> copies;
    for (size_t chain = 0; chain < lengths.size(); ++chain) {
      std::string key = o.chainSequences.empty() ? "chain-" + std::to_string(chain) : o.chainSequences[chain];
      int entity = (int)(std::find(entities.begin(), entities.end(), key) - entities.begin());
      if (entity == (int)entities.size()) entities.push_back(key);
      int copy = copies[entity]++;
      for (int w = 0; w < lengths[chain]; ++w, ++residue) {
        F.residueIndex.push_back(F.chainAware ? w : residue + (int)chain * 200);
        if (F.chainAware) { F.asymId.push_back((int)chain); F.entityId.push_back(entity); F.symId.push_back(copy); }
      }
    }
  }
  // the whole alignment's profile, float32 as the JavaScript holds it
  std::vector<float> msaProfile((size_t)length * 23, 0.0f);
  for (int row = 0; row < depth; ++row)
    for (int r = 0; r < length; ++r) msaProfile[(size_t)r * 23 + encoded[(size_t)row * length + r]] += 1.0f;
  for (auto& v : msaProfile) v = (float)((double)v / depth);
  auto sampleProfile = [&](int residue, double uniform) {
    double cumulative = 0;
    for (int code = 0; code < 23; ++code) {
      cumulative += msaProfile[(size_t)residue * 23 + code];
      if (uniform < cumulative) return code;
    }
    return 20;
  };
  int maxMsa = std::min(o.maxMsa, depth);
  struct Plan { std::vector<int> centers, extras; std::vector<uint8_t> centerCodes; };
  std::vector<Plan> plans;
  for (int recycle = 0; recycle <= o.recycles; ++recycle) {
    uint32_t seed = (uint32_t)(int64_t)o.seed ^ (uint32_t)((uint64_t)(uint32_t)(recycle + 1) * 0x9e3779b9u);
    Mulberry random(seed);
    auto shuffle = [&](std::vector<int>& v) {
      for (int i = (int)v.size() - 1; i > 0; --i) { int other = (int)std::floor(random() * (i + 1)); std::swap(v[i], v[other]); }
    };
    std::vector<int> remainder;
    for (int i = 1; i < depth; ++i) remainder.push_back(i);
    shuffle(remainder);
    Plan p;
    int take = std::max(0, maxMsa - 1);
    p.centers.push_back(0);
    for (int i = 0; i < std::min(take, (int)remainder.size()); ++i) p.centers.push_back(remainder[i]);
    std::vector<int> pool(remainder.begin() + std::min(take, (int)remainder.size()), remainder.end());
    shuffle(pool);
    p.extras.assign(pool.begin(), pool.begin() + std::min((int)pool.size(), o.maxExtra));
    p.centerCodes.resize(p.centers.size() * length);
    for (size_t c = 0; c < p.centers.size(); ++c)
      std::copy(encoded.begin() + (size_t)p.centers[c] * length, encoded.begin() + (size_t)(p.centers[c] + 1) * length, p.centerCodes.begin() + c * length);
    for (size_t index = 0; index < p.centerCodes.size(); ++index) {
      if (random() >= 0.15) continue;
      uint8_t original = p.centerCodes[index];
      double draw = random();
      if (draw < 0.7) p.centerCodes[index] = 22;
      else if (draw < 0.8) p.centerCodes[index] = (uint8_t)sampleProfile((int)(index % length), random());
      else if (draw < 0.9) p.centerCodes[index] = original;
      else p.centerCodes[index] = (uint8_t)std::floor(random() * 20);
    }
    plans.push_back(std::move(p));
  }
  auto deletionValue = [](double v) { return jsmath::atan(v / 3) * 2 / M_PI; };
  F.passes.resize(plans.size());
  auto finish = [&](size_t k) {
    const Plan& p = plans[k];
    int centres = (int)p.centers.size(), extras = (int)p.extras.size();
    // nearestCentres: the centre agreeing with the row at the most residues (a masked centre residue agrees with none)
    // (eight residues a 64-bit word, as paddedCodeWords lays them out: a centre's masked residue and its padding are
    // 255, a row's padding 254, so neither ever matches - and a word's matching bytes counted at once)
    int stride = (length + 7) / 8 * 8, words = stride / 8;
    std::vector<uint64_t> centreWords((size_t)centres * words), extraWords((size_t)std::max(1, extras) * words);
    {
      std::vector<uint8_t> padded((size_t)centres * stride, 255);
      for (int c = 0; c < centres; ++c)
        for (int r = 0; r < length; ++r) { uint8_t code = p.centerCodes[(size_t)c * length + r]; if (code <= 20) padded[(size_t)c * stride + r] = code; }
      std::memcpy(centreWords.data(), padded.data(), padded.size());
      std::vector<uint8_t> rows((size_t)extras * stride, 254);
      for (int e = 0; e < extras; ++e) std::memcpy(&rows[(size_t)e * stride], &encoded[(size_t)p.extras[e] * length], length);
      if (extras) std::memcpy(extraWords.data(), rows.data(), rows.size());
    }
    std::vector<int> assignments(extras);
    for (int e = 0; e < extras; ++e) {
      const uint64_t* row = &extraWords[(size_t)e * words];
      int best = 0, bestScore = -1;
      for (int c = 0; c < centres; ++c) {
        const uint64_t* centre = &centreWords[(size_t)c * words];
        int score = 0;
        for (int w = 0; w < words; ++w) {
          uint64_t d = centre[w] ^ row[w];
          uint64_t zeros = ~(((d & 0x7f7f7f7f7f7f7f7fULL) + 0x7f7f7f7f7f7f7f7fULL) | d) & 0x8080808080808080ULL;
          score += __builtin_popcountll(zeros);
        }
        if (score > bestScore) { bestScore = score; best = c; }
      }
      assignments[e] = best;
    }
    std::vector<float> profile((size_t)centres * length * 23, 0.0f), deletionSums((size_t)centres * length, 0.0f);
    std::vector<float> counts((size_t)centres * length, (float)(1 + 1e-6));
    for (int c = 0; c < centres; ++c)
      for (int r = 0; r < length; ++r) {
        profile[((size_t)c * length + r) * 23 + p.centerCodes[(size_t)c * length + r]] = 1;
        deletionSums[(size_t)c * length + r] = deletions[(size_t)p.centers[c] * length + r];
      }
    for (int e = 0; e < extras; ++e) {
      int row = p.extras[e], c = assignments[e];
      for (int r = 0; r < length; ++r) {
        size_t slot = (size_t)c * length + r;
        counts[slot] = counts[slot] + 1.0f;
        size_t ps = slot * 23 + encoded[(size_t)row * length + r];
        profile[ps] = profile[ps] + 1.0f;
        deletionSums[slot] = deletionSums[slot] + deletions[(size_t)row * length + r];
      }
    }
    Af2Pass& out = F.passes[k];
    out.centres = centres;
    out.msaFeat.assign((size_t)centres * length * 49, 0.0f);
    for (int c = 0; c < centres; ++c)
      for (int r = 0; r < length; ++r) {
        size_t slot = (size_t)c * length + r, o49 = slot * 49;
        out.msaFeat[o49 + p.centerCodes[slot]] = 1;
        double deletion = deletions[(size_t)p.centers[c] * length + r];
        out.msaFeat[o49 + 23] = (float)std::min(deletion, 1.0);
        out.msaFeat[o49 + 24] = (float)deletionValue(deletion);
        for (int code = 0; code < 23; ++code) out.msaFeat[o49 + 25 + code] = (float)((double)profile[slot * 23 + code] / counts[slot]);
        out.msaFeat[o49 + 48] = (float)deletionValue((double)deletionSums[slot] / counts[slot]);
      }
    out.msaMask.assign((size_t)centres * length, 1.0f);
    int extraSequences = std::max(1, extras);
    out.extras = extraSequences;
    out.extraMsa.assign((size_t)extraSequences * length, 0.0f);
    out.extraHasDeletion.assign((size_t)extraSequences * length, 0.0f);
    out.extraDeletionValue.assign((size_t)extraSequences * length, 0.0f);
    out.extraMsaMask.assign((size_t)extraSequences * length, 0.0f);
    for (int e = 0; e < extras; ++e)
      for (int r = 0; r < length; ++r) {
        size_t slot = (size_t)e * length + r;
        int row = p.extras[e];
        double deletion = deletions[(size_t)row * length + r];
        out.extraMsa[slot] = encoded[(size_t)row * length + r];
        out.extraHasDeletion[slot] = (float)std::min(deletion, 1.0);
        out.extraDeletionValue[slot] = (float)deletionValue(deletion);
        out.extraMsaMask[slot] = 1;
      }
  };
  // each pass on a thread of its own (no pass reads another's)
  std::vector<std::thread> threads;
  for (size_t k = 0; k < plans.size(); ++k) threads.emplace_back(finish, k);
  for (auto& t : threads) t.join();
  return F;
}

struct Af2Export { Entries entries; std::vector<std::string> said; std::string searchA3m; };

inline Af2Export exportAf2(const Args& args) {
  Af2Export out;
  std::string bundleDir = args.option("bundle"), weightsDir = args.option("weights");
  if (bundleDir.empty() == weightsDir.empty()) throw std::runtime_error("--bundle=<page bundle dir> or --weights=<export dir>");
  bool isMultimer;
  if (!bundleDir.empty()) {
    Json manifest = parseJson(readFile(bundleDir + "/manifest.json"));
    const Json* model = manifest.get("model");
    const Json* name = model ? model->get("name") : nullptr;
    isMultimer = name && name->isString() && name->s.find("multimer") != std::string::npos;
  } else {
    isMultimer = readFile(weightsDir + "/model.idx").find("m meta/multimer 1") != std::string::npos;
  }
  std::string sequence = upper(trimWs(args.option("sequence")));
  if (!args.option("job").empty()) {
    if (!sequence.empty()) throw std::runtime_error("--job and --sequence both name the input");
    Job job = jobFromJson(readFile(args.option("job")));
    for (auto& note : job.notes) out.said.push_back("job: " + note);
    Expanded request = expandEntities(job.entities);
    std::vector<std::string> other;
    for (auto& k : request.chainKinds) if (k != "protein" && std::find(other.begin(), other.end(), k) == other.end()) other.push_back(k);
    if (!other.empty()) {
      std::string list;
      for (size_t i = 0; i < other.size(); ++i) list += (i ? ", " : "") + other[i];
      throw std::runtime_error("AlphaFold 2 folds protein chains only; this job has " + list);
    }
    if (!request.ligands.empty()) throw std::runtime_error("AlphaFold 2 folds protein chains only; this job has a ligand");
    if (!request.modifications.empty()) throw std::runtime_error("AlphaFold 2 folds the standard residues only; this job has a modified residue");
    if (!request.bonds.empty()) throw std::runtime_error("AlphaFold 2 takes no declared bond; this job has one");
    sequence = request.sequence;
  }
  std::string a3mPath = args.option("a3m");
  if (sequence.empty() && a3mPath.empty()) throw std::runtime_error("--sequence or --a3m names the input");
  if (args.has("search") && sequence.empty()) throw std::runtime_error("--search needs --sequence");
  std::vector<std::string> chains = splitNonEmpty(sequence, ':');
  std::string a3m;
  search::Hits hits;
  bool searched = false;
  if (args.has("search")) {
    if (!a3mPath.empty()) throw std::runtime_error("--search and --a3m both name the alignment");
    auto started = std::chrono::steady_clock::now();
    auto say = [&](const std::string& m) { fprintf(stderr, "search: %s\n", m.c_str()); };
    if (chains.size() == 1) { auto s = search::searchOne(chains[0], say); a3m = s.a3m; hits = s.hits; }
    else { auto s = search::searchComplex(chains, isMultimer ? "multimer" : "monomer", say); a3m = s.merged.a3m; hits = s.hits; }
    searched = true;
    char took[64];
    snprintf(took, sizeof took, "%.1f", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    out.said.push_back("search: " + std::to_string(chains.size()) + " chain(s) from api.colabfold.com in " + took + " s");
    out.searchA3m = a3m;
  } else {
    auto paths = splitNonEmpty(a3mPath, ',');
    if (paths.size() > 1 || !args.option("paired-a3m").empty()) {
      if (paths.size() != chains.size())
        throw std::runtime_error(std::to_string(paths.size()) + " alignments for " + std::to_string(chains.size()) + " chains");
      auto pairedPaths = splitOn(args.option("paired-a3m"), ',');
      std::vector<std::string> paired, chainA3ms;
      for (size_t i = 0; i < chains.size(); ++i) paired.push_back(i < pairedPaths.size() && !pairedPaths[i].empty() ? readFile(pairedPaths[i]) : "");
      for (auto& p : paths) chainA3ms.push_back(readFile(p));
      std::map<std::string, std::string> pairedBySequence;
      bool any = false;
      for (auto& t : paired) if (!trimWs(t).empty()) any = true;
      if (any) for (size_t i = 0; i < chains.size(); ++i) pairedBySequence[chains[i]] = paired[i];
      a3m = search::mergeSearchedChains(chains, chainA3ms, pairedBySequence, isMultimer ? "multimer" : "monomer").a3m;
    } else {
      std::string joined;
      for (auto& c : chains) joined += c;
      a3m = a3mPath.empty() ? ">query\n" + joined + "\n" : readFile(a3mPath);
    }
  }
  Af2Options o;
  if (chains.size() > 1) {
    o.chainAware = true;
    for (auto& c : chains) o.chainLengths.push_back((int)c.size());
    o.chainSequences = chains;
  }
  o.recycles = (int)jsNumberOf(args.option("recycles", "3"));
  o.maxMsa = (int)jsNumberOf(args.option("max-msa", "512"));
  o.maxExtra = (int)jsNumberOf(args.option("max-extra", "1024"));
  o.seed = jsNumberOf(args.option("seed", "0"));
  Af2Features F = makeA3mFeatures(a3m, o);
  int L = F.length;
  Entries& E = out.entries;
  E.i("aatype", F.aatype);
  E.i("residue_index", F.residueIndex);
  E.t("seq_mask", F.seqMask);
  E.i("asym_id", F.chainAware ? F.asymId : std::vector<int>(L, 0));
  E.i("entity_id", F.chainAware ? F.entityId : std::vector<int>(L, 0));
  E.i("sym_id", F.chainAware ? F.symId : std::vector<int>(L, 0));
  for (size_t k = 0; k < F.passes.size(); ++k) {
    const Af2Pass& p = F.passes[k];
    std::string f = "f" + std::to_string(k) + "/";
    E.t(f + "msa_feat", p.msaFeat);
    E.t(f + "msa_mask", p.msaMask);
    std::vector<int> extra(p.extraMsa.begin(), p.extraMsa.end());
    E.i(f + "extra_msa", extra);
    E.t(f + "extra_has_deletion", p.extraHasDeletion);
    E.t(f + "extra_deletion_value", p.extraDeletionValue);
    E.t(f + "extra_msa_mask", p.extraMsaMask);
  }
  // --template: one slot, its parts aligned to their chains in the atom37 layout; --template-search-chains: the search's
  // best hit for each listed chain, in the same slot
  auto specs = splitNonEmpty(args.option("template"), ',');
  auto searchChains = splitNonEmpty(args.option("template-search-chains"), ',');
  if (!searchChains.empty() && !searched) throw std::runtime_error("--template-search-chains needs --search: the hits come from that search");
  if (specs.size() > 1) throw std::runtime_error("AlphaFold 2 takes one template slot: join its parts with '+'");
  if (!specs.empty() || !searchChains.empty()) {
    std::vector<int> offsets;
    { int at = 0; for (auto& c : chains) { offsets.push_back(at); at += (int)c.size(); } }
    struct Part { std::string text; bool hasChain; std::string chain; int at; std::string label; };
    std::vector<Part> searchParts;
    for (auto& c : searchChains) {
      int at = (int)jsNumberOf(c);
      auto it = hits.find(at);
      if (it == hits.end() || it->second.empty()) throw std::runtime_error("the search found no template for chain " + std::to_string(at + 1));
      const search::Hit& best = it->second[0];
      auto structures = search::fetchTemplates({best.target});
      auto st = structures.find(best.id);
      if (st == structures.end()) throw std::runtime_error("no structure came back for " + best.target);
      searchParts.push_back({st->second, true, best.chain, at, best.target});
    }
    std::string spec = specs.empty() ? "" : specs[0];
    std::vector<Part> parts;
    if (!spec.empty()) {
      if (spec.find('@') != std::string::npos) {
        for (auto& part : splitOn(spec, '+')) {
          auto pieces = splitOn(part, '@');
          std::string where = pieces[0];
          size_t cut = where.rfind(':');
          std::string path = cut == std::string::npos ? where.substr(0, where.empty() ? 0 : where.size() - 1) : where.substr(0, cut);
          std::string chain = cut == std::string::npos ? where : where.substr(cut + 1);
          parts.push_back({readFile(path), !chain.empty(), chain, (int)jsNumberOf(pieces.size() > 1 ? pieces[1] : ""), ""});
        }
      } else {
        auto halves = splitOn(spec, ':');
        std::string path0 = halves[0], chains0 = halves.size() > 1 ? halves[1] : "";
        auto list = splitOn(chains0, '+');
        std::string text = readFile(path0);
        for (size_t at = 0; at < list.size(); ++at) parts.push_back({text, !list[at].empty(), list[at], (int)at, ""});
      }
    }
    for (auto& p : searchParts) parts.push_back(p);
    std::set<int> taken;
    std::vector<Built> built;
    std::string joined;
    for (auto& c : chains) joined += c;
    for (auto& p : parts) {
      if (!isMultimer && p.at != 0) throw std::runtime_error("AlphaFold 2's monomer takes a template on its one chain only");
      if (p.at >= (int)chains.size()) throw std::runtime_error("template 0: no query chain " + std::to_string(p.at));
      if (taken.count(p.at)) throw std::runtime_error("AlphaFold 2 takes one template a chain; chain " + std::to_string(p.at + 1) + " has two");
      taken.insert(p.at);
      BuildOptions bo;
      bo.text = p.text;
      bo.chain = p.hasChain ? &p.chain : nullptr;
      bo.query = isMultimer ? chains[p.at] : joined;
      bo.offset = isMultimer ? offsets[p.at] : 0;
      bo.tokens = L;
      bo.atom37 = true;
      built.push_back(buildTemplate(bo));
    }
    TemplateSlot slot = built.size() == 1 ? built[0].slot : mergeAtom37Templates(built, L);
    std::string described = spec;
    for (auto& p : searchParts) described += (described.empty() ? "" : " + ") + ("search hit " + p.label);
    out.said.push_back("template: " + described + ", " + std::to_string(slot.covered) + " residues, " + std::to_string(slot.atoms) + " atoms");
    E.i("t/aatype", slot.aatype);
    E.t("t/positions", slot.atomPositions);
    E.t("t/mask", slot.atomMask);
    E.m("meta/templates", 1);
  }
  if (!weightsDir.empty()) {
    std::string w = weightsDir;
    while (!w.empty() && w.back() == '/') w.pop_back();
    std::string base = w.substr(w.rfind('/') == std::string::npos ? 0 : w.rfind('/') + 1);
    if (base.compare(0, 8, "weights-") == 0) base = base.substr(8);
    E.text("meta/model", base);
  }
  E.m("meta/tokens", L);
  E.m("meta/msa_rows", F.passes[0].centres);
  E.m("meta/extra_rows", F.passes[0].extras);
  E.m("meta/passes", (double)F.passes.size());
  out.said.push_back(std::to_string(L) + " residues, " + std::to_string(F.passes[0].centres) + " MSA rows, " + std::to_string(F.passes[0].extras)
                     + " extra, " + std::to_string(F.passes.size()) + " passes -> " + args.positional);
  return out;
}

}  // namespace lf
