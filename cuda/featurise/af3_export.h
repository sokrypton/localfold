// cuda/af3/export-model.mjs --no-weights, natively: a job (or a sequence) in, the AF3-lineage input out - the
// batch, the family's extras, the template passes and the PDB records - byte-identical to the JavaScript exporter
// (tools/check-native-featuriser.py holds it to that).
//
//   af3-featurise <out dir> --family=<model> (--job=<AF3 job.json> | --sequence=<A:B> [--kinds= --ligands= --modify=])
//                 [--max-msa=1024] [--seed=20260831] [--a3m=<one a chain> [--paired-a3m=...]]
#pragma once
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "af3_extras.h"
#include "ccd.h"
#include "entries.h"
#include "featurise.h"
#include "job.h"
#include "msa.h"
#include "pdb.h"
#include "tables.h"
#include "search.h"
#include "templates.h"

namespace lf {


struct Args {
  std::vector<std::string> list;
  std::string positional;
  explicit Args(const std::vector<std::string>& a) : list(a) {
    for (auto& s : a) if (s.compare(0, 2, "--") != 0) { positional = s; break; }
  }
  std::string option(const std::string& name, const std::string& fallback = "") const {
    std::string key = "--" + name + "=";
    for (auto& s : list) if (s.compare(0, key.size(), key) == 0) return s.substr(key.size());
    return fallback;
  }
  bool has(const std::string& flag) const {
    for (auto& s : list) if (s == "--" + flag) return true;
    return false;
  }
};

inline std::vector<std::string> splitOn(const std::string& s, char sep) {   // s.split(sep)
  std::vector<std::string> out;
  size_t start = 0, at;
  while ((at = s.find(sep, start)) != std::string::npos) { out.push_back(s.substr(start, at - start)); start = at + 1; }
  out.push_back(s.substr(start));
  return out;
}
inline std::vector<std::string> splitNonEmpty(const std::string& s, char sep) {   // .split(sep).filter(Boolean)
  std::vector<std::string> out;
  for (auto& p : splitOn(s, sep)) if (!p.empty()) out.push_back(p);
  return out;
}

// the job's userCCD (each data_ block a component), then a local CCD when one is named (--ccd), else the RCSB
// through the cache
struct ComponentSource {
  std::map<std::string, Component> user;
  std::string file;                  // --ccd=<components.cif[.gz]>
  explicit ComponentSource(const Args& args) : file(args.option("ccd")) {}
  void addUserCcd(const std::string& text) {
    std::vector<size_t> starts;
    for (size_t at = 0; at < text.size(); at = text.find('\n', at), at = at == std::string::npos ? text.size() : at + 1)
      if (text.compare(at, 5, "data_") == 0) starts.push_back(at);
    // text.split(/^(?=data_)/m): the piece before the first data_ is a block too, kept only if it starts with data_
    for (size_t i = 0; i < starts.size(); ++i) {
      std::string block = text.substr(starts[i], (i + 1 < starts.size() ? starts[i + 1] : text.size()) - starts[i]);
      Component c = parseCcdComponent(block);
      user[upper(c.code)] = c;
    }
  }
  Component get(const std::string& code) const {
    auto it = user.find(upper(code));
    if (it != user.end()) return it->second;
    return parseCcdComponent(file.empty() ? ccdText(code) : ccdFileText(file, code));
  }
};

struct Af3Export {
  Entries entries;
  std::string pdb;          // template.pdb ("" when the writer could not)
  int tokens = 0;
  std::vector<std::string> said;   // what the JavaScript exporter prints (the worker reads template coverage off it)
  std::string searchA3m;           // one chain's searched alignment, as the search returned it (search.a3m)
};

// The exporter's template stage: --template slots and the job's own templates, each part by buildTemplate (every
// chain's k-th in slot k, merged), rf3's stereocentres, then the embedder's passes - a repeat and an aatype each, and
// either a fused embedder's sparse feature rows or the nine-projection embedder's geometry - and their totals
// what --search found: the template hits by search chain, and which fold chain each search chain is
struct SearchState { bool searched = false; search::Hits hits; std::vector<int> proteinAt; };

struct TemplatePart { std::string text; bool hasChain = false; std::string chain; int queryChain = 0; std::string label;
                      bool hasMapping = false; std::vector<std::pair<int, int>> mapping; };
inline void addTemplates(Af3Export& out, const Args& args, const Batch& batch, const FamilyFlags& D, const std::vector<std::string>& chains,
                         const std::vector<std::vector<JobTemplate>>& jobTemplates, const Expanded* request, const SearchState& found) {
  Entries& E = out.entries;
  const int TEMPLATES = 4;
  int tokens = batch.tokens;
  // which token each chain's residue occupies (a modified residue or ligand shifts them)
  std::vector<int> tokenOfResidue(batch.chainOfResidue.size(), -1);
  for (int t = 0; t < tokens; ++t) {
    int r = batch.residueOfToken[t];
    if (r >= 0 && tokenOfResidue[r] == -1) tokenOfResidue[r] = t;
  }
  std::vector<std::vector<int>> residuesOfChain;
  for (size_t r = 0; r < batch.chainOfResidue.size(); ++r) {
    int c = (int)batch.chainOfResidue[r];
    if ((int)residuesOfChain.size() <= c) residuesOfChain.resize(c + 1);
    residuesOfChain[c].push_back((int)r);
  }
  struct SlotMask { TemplateSlot slot; std::vector<float> mask; };
  std::vector<SlotMask> slots;
  auto buildPart = [&](const TemplatePart& p, int k) {
    std::function<int(int)> tokenOf = [&, chain = p.queryChain](int residue) {
      if (chain < 0 || chain >= (int)residuesOfChain.size() || residue < 0 || residue >= (int)residuesOfChain[chain].size()) return -1;
      int r = residuesOfChain[chain][residue];
      return r >= 0 && r < (int)tokenOfResidue.size() ? tokenOfResidue[r] : -1;
    };
    BuildOptions o;
    o.text = p.text;
    o.chain = p.hasChain ? &p.chain : nullptr;
    o.query = p.queryChain >= 0 && p.queryChain < (int)chains.size() ? chains[p.queryChain] : "";
    o.tokens = tokens;
    o.tokenOf = &tokenOf;
    o.mapping = p.hasMapping ? &p.mapping : nullptr;
    Built b = buildTemplate(o);
    out.said.push_back("template " + std::to_string(k) + ": " + p.label + " -> query chain " + std::to_string(p.queryChain) + ", "
                       + std::to_string(b.residues) + "/" + std::to_string(b.of) + " residues" + (p.hasMapping ? " (the job's mapping)" : ""));
    return b.slot;
  };
  // --template=<file>:<chain>[@<query chain>], comma-separated slots, "+"-joined parts sharing one
  auto specs = splitNonEmpty(args.option("template"), ',');
  if ((int)specs.size() > TEMPLATES) throw std::runtime_error("at most four template slots");
  for (size_t k = 0; k < specs.size(); ++k) {
    std::vector<TemplateSlot> parts;
    for (auto& part : splitOn(specs[k], '+')) {
      auto at = part.find('@');
      std::string where = part.substr(0, at), target = at == std::string::npos ? "0" : part.substr(at + 1);
      size_t cut = where.rfind(':');
      TemplatePart p;
      // (as the JavaScript slices it: with no ":", the path loses its last character and the chain is all of it)
      std::string path = cut == std::string::npos ? where.substr(0, where.empty() ? 0 : where.size() - 1) : where.substr(0, cut);
      std::string chainId = cut == std::string::npos ? where : where.substr(cut + 1);
      p.text = readFile(path);
      p.hasChain = !chainId.empty(); p.chain = chainId;
      p.queryChain = (int)jsNumberOf(target);
      p.label = path + " chain " + (p.hasChain ? chainId : "undefined");
      parts.push_back(buildPart(p, (int)k));
    }
    TemplateSlot slot = parts.size() == 1 ? parts[0] : mergeTemplateSlots(parts);
    bool spanChains = parts.size() > 1 && !args.has("no-span-chains");
    std::vector<float> coverage = coverageOf(slot, tokens);
    slots.push_back({slot, multichainMaskFor(batch.asymId, tokens, &coverage, spanChains)});
  }
  // a job's own templates: every chain's k-th in slot k, each with the job's mapping where it gives one
  std::vector<std::vector<TemplatePart>> extra;
  if (!jobTemplates.empty()) {
    if (!specs.empty()) throw std::runtime_error("the job carries its templates; --template would replace them");
    for (auto& list : jobTemplates) {
      std::vector<TemplatePart> parts;
      for (auto& t : list) {
        TemplatePart p;
        p.text = t.text; p.queryChain = t.chain; p.label = t.label; p.hasMapping = t.hasMapping; p.mapping = t.mapping;
        parts.push_back(p);
      }
      extra.push_back(parts);
    }
  }
  if (request)
    for (auto& [chain, t] : request->templates)
      if (t.kind != "upload" && !args.has("search-templates"))
        throw std::runtime_error("chain " + std::to_string(chain) + ": the job asks for a template search - run with --search-templates,"
                                 " or give the structure with --template=<file>:<chain>@<query chain>");
  // --search-templates: each protein chain's best four hits from the same search, every chain's k-th in slot k
  if (args.has("search-templates")) {
    if (!specs.empty() || !extra.empty()) throw std::runtime_error("--search-templates and other templates both name the slots");
    std::vector<std::pair<int, std::vector<search::Hit>>> perChain;
    std::vector<std::string> targets;
    for (auto& [at, hits] : found.hits) {
      std::vector<search::Hit> best(hits.begin(), hits.begin() + std::min<size_t>(TEMPLATES, hits.size()));
      perChain.push_back({at < (int)found.proteinAt.size() ? found.proteinAt[at] : at, best});
      for (auto& h : best) targets.push_back(h.target);
    }
    auto structures = search::fetchTemplates(targets);
    for (int k = 0; k < TEMPLATES; ++k) {
      std::vector<TemplatePart> parts;
      for (auto& [chain, hits] : perChain) {
        if (k >= (int)hits.size()) continue;
        auto it = structures.find(hits[k].id);
        if (it == structures.end()) throw std::runtime_error("no structure came back for " + hits[k].target);
        TemplatePart p;
        p.text = it->second; p.hasChain = true; p.chain = hits[k].chain; p.queryChain = chain; p.label = "search hit " + hits[k].target;
        parts.push_back(p);
      }
      if (!parts.empty()) extra.push_back(parts);
    }
    if (extra.empty()) out.said.push_back("search: no template hits");
  }
  // --template-search-chains: the page's "from the MSA search" - each listed chain's BEST hit, a slot of its own
  auto searchChains = splitNonEmpty(args.option("template-search-chains"), ',');
  if (!searchChains.empty()) {
    if (!found.searched) throw std::runtime_error("--template-search-chains needs --search: the hits come from that search");
    std::map<int, std::vector<search::Hit>> byChain;
    for (auto& [at, hits] : found.hits) byChain[at < (int)found.proteinAt.size() ? found.proteinAt[at] : at] = hits;
    for (auto& c : searchChains) {
      int chain = (int)jsNumberOf(c);
      auto it = byChain.find(chain);
      if (it == byChain.end() || it->second.empty()) throw std::runtime_error("the search found no template for chain " + std::to_string(chain + 1));
      const search::Hit& best = it->second[0];
      auto structures = search::fetchTemplates({best.target});
      auto st = structures.find(best.id);
      if (st == structures.end()) throw std::runtime_error("no structure came back for " + best.target);
      TemplatePart p;
      p.text = st->second; p.hasChain = true; p.chain = best.chain; p.queryChain = chain; p.label = "search hit " + best.target;
      extra.push_back({p});
    }
  }
  for (size_t k = 0; k < extra.size(); ++k) {
    std::vector<TemplateSlot> parts;
    for (auto& p : extra[k]) parts.push_back(buildPart(p, (int)k));
    TemplateSlot slot = parts.size() == 1 ? parts[0] : mergeTemplateSlots(parts);
    std::vector<float> coverage = coverageOf(slot, tokens);
    slots.push_back({slot, multichainMaskFor(batch.asymId, tokens, &coverage, false)});
  }
  if (D.chiralCentres) addChiralCentres(E, batch);

  bool fused = D.fusedDistogramBins > 0 || D.boltz2TemplateFeatures || D.rosettafold3TemplateFeatures;
  int width = D.boltz2TemplateFeatures ? 109 : D.rosettafold3TemplateFeatures ? 66
            : D.fusedDistogramBins > 0 ? D.fusedDistogramBins + 1 + 2 * D.fusedRestypes + 4 : 0;
  if ((int)slots.size() > TEMPLATES)
    throw std::runtime_error(std::to_string(slots.size()) + " template slots; every family folds with at most " + std::to_string(TEMPLATES));
  // a real slot's dense feature rows (fusedTemplateFeatures with a template)
  auto rowsFor = [&](const SlotMask& sm) {
    if (D.boltz2TemplateFeatures) return boltz2Rows(sm.slot, sm.mask, tokens);
    if (D.rosettafold3TemplateFeatures) return rosettafold3Rows(sm.slot, sm.mask, tokens);
    if (D.fusedDistogramBins <= 0) throw std::runtime_error("dialect.fusedTemplateLayout has no default");
    return protenixRows(sm.slot, sm.mask, tokens, D.fusedDistogramBins, D.fusedRestypes, width);
  };
  // the columns an empty slot sets (emptyTemplateColumns)
  auto emptyColumns = [&](bool useGap) {
    std::vector<int> columns;
    if (useGap) {
      if (D.emptyColumnsState != 2) throw std::runtime_error("dialect.emptyTemplateRestypeColumns has no default");
      columns = D.emptyColumns;
    } else if (D.fusedDistogramBins > 0 && D.emptyColumnsState != 1) {
      columns = {D.fusedDistogramBins + 1, D.fusedDistogramBins + 1 + D.fusedRestypes};
    }
    return columns;
  };
  struct Pass { int repeat; bool real = false; size_t slot = 0; int emptyAatype = 0; bool mean = false; };
  std::vector<Pass> passes;
  std::vector<float> meanFeatures;            // the one-pass mean (rf3), dense, float32 as the JavaScript holds it
  size_t pairs = (size_t)tokens * tokens;
  if (D.templateFeatureMeanOnePass) {
    std::vector<size_t> present;
    for (size_t k = 0; k < slots.size(); ++k)
      for (float v : slots[k].slot.atomMask) if (v > 0) { present.push_back(k); break; }
    meanFeatures.assign(pairs * width, 0.0f);
    if (present.empty()) {
      for (int c : emptyColumns(false)) for (size_t r = 0; r < pairs; ++r) meanFeatures[r * width + c] = 1;
    } else {
      std::vector<float> row(width);
      for (size_t at = 0; at < present.size(); ++at) {
        FusedRows rows = rowsFor(slots[present[at]]);
        for (size_t r = 0; r < pairs; ++r) {
          std::fill(row.begin(), row.end(), 0.0f);
          rows.fill((int)(r / tokens), (int)(r % tokens), row);
          for (int c = 0; c < width; ++c)
            meanFeatures[r * width + c] = at == 0 ? row[c] : (float)((double)meanFeatures[r * width + c] + row[c]);
        }
      }
      for (auto& v : meanFeatures) v = (float)((double)v / present.size());
    }
    passes.push_back({TEMPLATES, false, 0, 0, true});
  } else {
    for (size_t k = 0; k < slots.size(); ++k) passes.push_back({1, true, k, 0, false});
    int empty = TEMPLATES - (int)slots.size();
    if (empty > 0 && D.emptyTemplateAatype >= 0) {
      passes.push_back({1, false, 0, D.emptyTemplateAatype, false});
      if (empty > 1) passes.push_back({empty - 1, false, 0, 0, false});
    } else if (empty > 0) passes.push_back({empty, false, 0, 0, false});
  }
  for (size_t k = 0; k < passes.size(); ++k) {
    const Pass& p = passes[k];
    std::string q = "template." + std::to_string(k);
    bool covered = !((D.templateVisibilityByCoverage || D.chaiTemplates) && !p.real && !p.mean);
    E.m(q + ".repeat", covered ? p.repeat : 0);
    if (p.real) {
      const TemplateSlot& slot = slots[p.slot].slot;
      std::vector<int> aatype = slot.aatype;
      if (D.chaiTemplates)
        for (int t = 0; t < tokens; ++t) {
          bool any = false;
          for (int a = 0; a < batch.dense; ++a) if (slot.atomMask[(size_t)t * batch.dense + a] > 0) { any = true; break; }
          if (!any) aatype[t] = 21;
        }
      E.i(q + ".aatype", aatype);
    } else {
      E.i(q + ".aatype", std::vector<int>(tokens, p.mean ? 0 : p.emptyAatype));
    }
    if (fused) {
      std::vector<int> idx;
      std::vector<float> val;
      int K = 1;
      if (p.mean) {
        sparseRows([&](size_t r, std::vector<float>& d) { std::copy(meanFeatures.begin() + r * width, meanFeatures.begin() + (r + 1) * width, d.begin()); },
                   pairs, width, idx, val, K);
      } else if (p.real) {
        FusedRows rows = rowsFor(slots[p.slot]);
        sparseRows([&](size_t r, std::vector<float>& d) { rows.fill((int)(r / tokens), (int)(r % tokens), d); }, pairs, width, idx, val, K);
      } else {
        // fusedTemplateFeaturesSparse(undefined, ...): every row the empty columns, sorted, value 1
        std::vector<int> columns = emptyColumns(p.emptyAatype != 0);
        std::sort(columns.begin(), columns.end());
        K = std::max(1, (int)columns.size());
        idx.assign(pairs * K, -1);
        val.assign(pairs * K, 0.0f);
        for (size_t r = 0; r < pairs; ++r)
          for (int c = 0; c < (int)columns.size(); ++c) { idx[r * K + c] = columns[c]; val[r * K + c] = 1.0f; }
      }
      E.i(q + ".featuresIdx", idx);
      E.t(q + ".featuresVal", val);
      E.m(q + ".featuresK", K);
    } else if (p.real) {
      Geometry g = templateGeometry(slots[p.slot].slot, slots[p.slot].mask, tokens, D.chaiTemplates);
      E.i(q + ".distogramBin", g.bin);
      E.m(q + ".distogramBins", g.bins);
      E.t(q + ".pseudoBetaMask2d", g.pseudoBetaMask2d);
      E.t(q + ".unitVector", g.unitVector);
      E.t(q + ".backboneMask2d", g.backboneMask2d);
    }
  }
  E.m("template.passes", (double)passes.size());
  E.m("template.templates", D.chaiTemplates ? std::max(1, (int)slots.size()) : TEMPLATES);
  E.m("template.featureWidth", width);
  E.flag("template.outerResidual", D.templateStackOuterResidual);
}

inline Af3Export exportAf3(const Args& args) {
  Af3Export out;
  Entries& E = out.entries;
  if (!args.has("no-weights")) throw std::runtime_error("the native featuriser writes the input alone: pass --no-weights");
  std::string familyName = args.option("family");
  if (familyName.empty()) throw std::runtime_error("--family=<model> names the dialect (a native export reads no manifest)");
  const FamilyFlags& D = familyFor(familyName);

  std::string sequence = args.option("sequence");
  bool haveJob = !args.option("job").empty();
  Expanded request;
  std::vector<std::vector<JobTemplate>> jobTemplates;
  ComponentSource components(args);
  Alignment alignment;
  bool jobAlignment = false;
  std::vector<std::string> msaColumnKinds;
  if (haveJob) {
    if (!sequence.empty()) throw std::runtime_error("--job and --sequence both name the input");
    std::string text = readFile(args.option("job"));
    Job job = jobFromJson(text, &jobTemplates);
    // every modelSeed, where the page folds the first: af3 runs each (--seeds)
    std::vector<double> seeds;
    {
      Json raw = parseJson(text);
      const Json* first = raw.isArray() ? (raw.a.empty() ? nullptr : &raw.a[0]) : &raw;
      const Json* ms = first ? first->get("modelSeeds") : nullptr;
      if (ms != nullptr && !ms->isNull()) {
        if (ms->isArray()) for (auto& v : ms->a) seeds.push_back(jsToNumber(&v));
        else seeds.push_back(jsToNumber(ms));
      }
    }
    for (double s : seeds) if (!jsIsInteger(s) || s < 0) throw std::runtime_error("modelSeeds: " + jsNumber(s) + " is not a seed");
    for (auto& note : job.notes) {
      if (seeds.size() > 1 && note.find("seeds in the file; folding the first") != std::string::npos)
        out.said.push_back("job: " + std::to_string(seeds.size()) + " seeds, each folded");
      else out.said.push_back("job: " + note);
    }
    if (seeds.size() > 1) {
      E.m("job.seeds.count", (double)seeds.size());
      for (size_t i = 0; i < seeds.size(); ++i) E.m("job.seeds." + std::to_string(i), seeds[i]);
    }
    request = expandEntities(job.entities);
    if (!job.userCcd.empty()) components.addUserCcd(job.userCcd);
    sequence = request.sequence;
    if (job.hasAlignments) {
      // mergeJobAlignments (shared/input/chains.js)
      if (job.unpaired.size() != request.chains.size() || job.paired.size() != request.chains.size())
        throw std::runtime_error("the job's alignments name " + std::to_string(job.unpaired.size()) + " chains and the fold has "
                                 + std::to_string(request.chains.size()));
      bool anyLetter = false;
      for (auto& k : request.chainKinds) if (k != "protein") anyLetter = true;
      auto merged = [&](const std::vector<OptText>& list, bool& has, std::string& text) {
        bool any = false;
        for (auto& t : list) if (t.present && !t.text.empty()) any = true;
        has = any;
        if (!any) return;
        std::vector<std::string> filled;
        for (size_t i = 0; i < list.size(); ++i)
          filled.push_back(list[i].present && !list[i].text.empty() ? list[i].text : ">query\n" + request.chains[i] + "\n");
        text = filled.size() == 1 ? filled[0] : mergeRowAlignedChainA3ms(filled, anyLetter);
      };
      alignment.present = true;
      alignment.blocks = true;
      merged(job.unpaired, alignment.hasUnpaired, alignment.unpaired);
      merged(job.paired, alignment.hasPaired, alignment.paired);
      for (size_t i = 0; i < request.chains.size(); ++i)
        for (size_t r = 0; r < request.chains[i].size(); ++r) msaColumnKinds.push_back(request.chainKinds[i]);
      jobAlignment = true;
      int u = 0, p = 0;
      for (auto& t : job.unpaired) if (t.present && !t.text.empty()) ++u;
      for (auto& t : job.paired) if (t.present && !t.text.empty()) ++p;
      out.said.push_back("job: inline alignments for " + std::to_string(u) + " chains (unpaired), " + std::to_string(p) + " (paired)");
    }
    if (job.hasSeed) E.m("job.seed", job.seed);
  }
  if (sequence.empty()) throw std::runtime_error("the native featuriser needs --job or --sequence (a batch dump is the JavaScript exporter's)");

  // --a3m / --paired-a3m: one path a chain, merged row by row; a single path is a monomer's
  {
    auto texts = [&](const std::string& spec, bool& has) {
      std::vector<std::string> list;
      has = !spec.empty();
      if (has) for (auto& path : splitOn(spec, ',')) list.push_back(readFile(trimWs(path)));
      return list;
    };
    bool hasU, hasP;
    auto unpaired = texts(args.option("a3m"), hasU), paired = texts(args.option("paired-a3m"), hasP);
    if (jobAlignment && (hasU || hasP)) throw std::runtime_error("the job carries its alignments; --a3m would replace them");
    if (!jobAlignment && (hasU || hasP)) {
      alignment.present = true;
      if (!hasP && unpaired.size() == 1) alignment.single = unpaired[0];
      else {
        alignment.blocks = true;
        alignment.hasPaired = hasP; alignment.hasUnpaired = hasU;
        if (hasP) alignment.paired = paired.size() == 1 ? paired[0] : mergeRowAlignedChainA3ms(paired);
        if (hasU) alignment.unpaired = unpaired.size() == 1 ? unpaired[0] : mergeRowAlignedChainA3ms(unpaired);
      }
    }
  }
  // --search: the protein chains' alignments from the ColabFold MMseqs2 server, the page's client and merge
  SearchState found;
  if (args.has("search") || args.has("search-templates")) {
    if (alignment.present) throw std::runtime_error("--search and an alignment both name the MSA");
    std::vector<std::string> allChains = splitOn(sequence, ':'), allKinds;
    if (haveJob) allKinds = request.chainKinds;
    else if (!args.option("kinds").empty()) allKinds = splitOn(args.option("kinds"), ',');
    else allKinds.assign(allChains.size(), "protein");
    std::vector<std::string> proteins;
    for (size_t i = 0; i < allChains.size(); ++i)
      if (i < allKinds.size() && allKinds[i] == "protein") { proteins.push_back(allChains[i]); found.proteinAt.push_back((int)i); }
    if (proteins.empty()) throw std::runtime_error("--search: no protein chain to search for");
    auto started = std::chrono::steady_clock::now();
    auto say = [&](const std::string& m) { fprintf(stderr, "search: %s\n", m.c_str()); };
    alignment.present = true;
    if (proteins.size() == 1) {
      search::Searched one = search::searchOne(proteins[0], say);
      alignment.single = one.a3m;
      found.hits = one.hits;
      out.searchA3m = one.a3m;
    } else {
      search::ComplexSearched many = search::searchComplex(proteins, "af3", say);
      alignment.blocks = true;
      alignment.hasPaired = many.merged.hasPaired; alignment.paired = many.merged.paired;
      alignment.hasUnpaired = true; alignment.unpaired = many.merged.unpaired;
      alignment.hasUnpairedProfile = true; alignment.unpairedProfile = many.merged.unpairedProfile;
      found.hits = many.hits;
    }
    found.searched = true;
    char took[64];
    snprintf(took, sizeof took, "%.1f", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    out.said.push_back("search: " + std::to_string(proteins.size()) + " protein chain(s) from api.colabfold.com in " + took + " s");
  }

  // the components: the request's (the page's own resolution), then --ligands / --modify
  FeaturiseOptions o;
  std::string kinds = args.option("kinds");
  if (haveJob) {
    for (auto& l : request.ligands) {
      if (l.kind == LigandRef::Code) o.ligands.push_back(components.get(l.code));
      else if (l.kind == LigandRef::Chain) {
        std::vector<Component> parts;
        for (auto& c : l.codes) parts.push_back(components.get(c));
        o.ligands.push_back(ligandChain(parts));
      } else {
        o.ligands.push_back(chem::smilesComponent(l.smiles, l.code.empty() ? "LIG" : l.code));
      }
    }
    for (auto& m : request.modifications) o.modifications.push_back({m.chain, (int)m.position, components.get(m.code)});
    std::string k;
    for (size_t i = 0; i < request.chainKinds.size(); ++i) k += (i ? "," : "") + request.chainKinds[i];
    kinds = k;
    o.bonds = request.bonds;
  }
  for (auto& code : splitNonEmpty(args.option("ligands"), ',')) o.ligands.push_back(components.get(code));
  if (!args.option("smiles").empty()) {
    auto list = splitNonEmpty(args.option("smiles"), '|');
    std::vector<std::string> seen;
    for (auto& s : list) {        // nameSmilesLigands: one name a distinct SMILES
      size_t at = std::find(seen.begin(), seen.end(), s) - seen.begin();
      if (at == seen.size()) seen.push_back(s);
      o.ligands.push_back(chem::smilesComponent(s, ligandName(at)));
    }
  }
  for (auto& spec : splitNonEmpty(args.option("modify"), ',')) {
    auto parts = splitOn(spec, '@');
    o.modifications.push_back({parts.size() > 2 ? (int)jsParseFloat(parts[2]) : 0, (int)jsParseFloat(parts.size() > 1 ? parts[1] : ""),
                               components.get(parts[0])});
  }
  if (!kinds.empty()) o.chainKinds = splitOn(kinds, ',');

  // af3BatchFromA3m: the rows (seeded from --seed), then featuriseProtein with the dialect's conventions
  if (alignment.present) {
    Uniform random(jsParseFloat(args.option("seed", "20260831")));
    MsaRows rows = af3MsaFromA3m(alignment, (int)jsParseFloat(args.option("max-msa", "1024")),
                                 msaColumnKinds.empty() ? nullptr : &msaColumnKinds, &random);
    o.msa = rows.msa; o.deletionMatrix = rows.deletionMatrix;
    o.profileMsa = rows.profileMsa; o.profileDeletionMatrix = rows.profileDeletionMatrix;
    o.hasUnpairedFrom = true; o.unpairedFrom = rows.unpairedFrom;
  } else {
    o.hasUnpairedFrom = true; o.unpairedFrom = 0;
  }
  o.symmetriseBonds = D.symmetriseBonds; o.centreRefConformers = D.centreRefConformers;
  o.paddedAtomKeys = D.paddedAtomKeys; o.qblockAtomKeys = D.qblockAtomKeys;
  o.atomizedElementNames = D.atomizedElementNames; o.atomizedUnknownRestype = D.atomizedUnknownRestype;
  o.atomizedUnknownMsa = D.atomizedUnknownMsa; o.atomizedBackboneBonds = D.atomizedBackboneBonds;
  o.modifiedAsOneToken = D.modifiedAsOneToken;
  o.terminalAtoms = D.dropTerminalAtoms == 1 ? 1 : D.dropTerminalAtoms == 2 ? 2 : 0;
  o.duplicateQueryRow = D.dedupeSelfMsa == 0;
  for (auto& k : msaColumnKinds) if (k != "protein") o.msaCoversNucleic = true;
  Batch batch = featuriseProtein(sequence, o);
  if (D.chaiMsaFeatures && !alignment.present) {
    std::fill(batch.msaMask.begin(), batch.msaMask.end(), 0.0f);
    std::fill(batch.profile.begin(), batch.profile.end(), 0.0f);
    std::fill(batch.deletionMean.begin(), batch.deletionMean.end(), 0.0f);
  }
  if (D.chaiTokenEmbedding && !o.bonds.empty())
    throw std::runtime_error("Chai-1 does not take declared covalent bonds yet (bondedAtomPairs, a glycan's links): the"
                             " trunk half of its bond projection is not in the bundle");
  addBatch(E, batch, !D.chaiTokenEmbedding);
  if (D.chaiTokenEmbedding) addEsm2Inputs(E, batch);
  E.i("batch.contactClasses", af3ContactClasses(batch));
  if (D.structuralTokens) addStructural(E, batch);
  out.tokens = batch.tokens;

  // the template slots (templates.h, each part by the page's buildTemplate), rf3's stereocentres, then the embedder's
  // passes - in the exporter's order
  std::vector<std::string> chains = splitOn(sequence, ':');
  addTemplates(out, args, batch, D, chains, jobTemplates, haveJob ? &request : nullptr, found);
  try { out.pdb = pdbTemplateText(batch); }
  catch (const std::exception& e) { out.said.push_back(std::string("no PDB template (") + e.what() + "); the native writer will name residues itself"); }
  return out;
}

}  // namespace lf
