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

namespace lf {

// the template stage (templates.h): fills the passes; unset, a job with a template is refused
struct Af3TemplateInput;

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

// the job's userCCD (each data_ block a component), then the RCSB through the cache
struct ComponentSource {
  std::map<std::string, Component> user;
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
    return parseCcdComponent(ccdText(code));
  }
};

struct Af3Export {
  Entries entries;
  std::string pdb;          // template.pdb ("" when the writer could not)
  int tokens = 0;
  std::vector<std::string> said;   // what the JavaScript exporter prints (the worker reads template coverage off it)
};

inline Af3Export exportAf3(const Args& args,
                           const std::function<void(Af3Export&, const Batch&, const FamilyFlags&, const std::vector<std::string>& chains,
                                                    const std::vector<std::vector<JobTemplate>>& jobTemplates,
                                                    const Expanded* request)>& templates = nullptr) {
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
  ComponentSource components;
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
  if (args.has("search") || args.has("search-templates"))
    throw std::runtime_error("the native featuriser's MSA search is not built yet");

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

  // the template slots and the embedder's passes (templates.h), then rf3's stereocentres - in the exporter's order
  std::vector<std::string> chains = splitOn(sequence, ':');
  bool anyTemplate = !args.option("template").empty() || !jobTemplates.empty() || !args.option("template-search-chains").empty();
  for (auto& t : request.templates) if (t.second.kind != "upload") anyTemplate = true;
  if (templates) templates(out, batch, D, chains, jobTemplates, haveJob ? &request : nullptr);
  else if (anyTemplate) throw std::runtime_error("the native featuriser's templates are not built yet");
  else {
    // no template: every slot empty (the exporter's passes for slots = [])
    const int TEMPLATES = 4;
    if (D.chiralCentres) addChiralCentres(E, batch);
    bool fused = D.fusedDistogramBins > 0 || D.boltz2TemplateFeatures || D.rosettafold3TemplateFeatures;
    int width = D.boltz2TemplateFeatures ? 109 : D.rosettafold3TemplateFeatures ? 66
              : D.fusedDistogramBins > 0 ? D.fusedDistogramBins + 1 + 2 * D.fusedRestypes + 4 : 0;
    struct Pass { int repeat; int emptyAatype; bool meanFeatures; };
    std::vector<Pass> passes;
    if (D.templateFeatureMeanOnePass) passes.push_back({TEMPLATES, 0, true});
    else if (D.emptyTemplateAatype >= 0) { passes.push_back({1, D.emptyTemplateAatype, false}); passes.push_back({3, 0, false}); }
    else passes.push_back({TEMPLATES, 0, false});
    size_t pairs = (size_t)batch.tokens * batch.tokens;
    for (size_t k = 0; k < passes.size(); ++k) {
      const Pass& p = passes[k];
      std::string q = "template." + std::to_string(k);
      bool covered = !((D.templateVisibilityByCoverage || D.chaiTemplates) && !p.meanFeatures);
      E.m(q + ".repeat", covered ? p.repeat : 0);
      E.i(q + ".aatype", std::vector<int>(batch.tokens, p.meanFeatures ? 0 : p.emptyAatype));
      if (!fused) continue;
      // fusedTemplateFeaturesSparse(undefined, ...) / sparseTemplateFeatures of the mean pass's (all-empty) features
      std::vector<int> columns;
      if (p.meanFeatures) {
        // fusedTemplateFeatures(undefined, ..., useGap false): emptyTemplateColumns(dialect, false)
        if (D.fusedDistogramBins > 0 && D.emptyColumnsState != 1)
          columns = {D.fusedDistogramBins + 1, D.fusedDistogramBins + 1 + D.fusedRestypes};
      } else {
        bool gap = p.emptyAatype != 0;
        if (gap) {
          if (D.emptyColumnsState != 2) throw std::runtime_error("dialect.emptyTemplateRestypeColumns has no default");
          columns = D.emptyColumns;
        } else if (D.fusedDistogramBins > 0 && D.emptyColumnsState != 1) {
          columns = {D.fusedDistogramBins + 1, D.fusedDistogramBins + 1 + D.fusedRestypes};
        }
        std::sort(columns.begin(), columns.end());
      }
      int K = std::max(1, (int)columns.size());
      std::vector<int> idx(pairs * K);
      std::vector<float> val(pairs * K);
      for (size_t r = 0; r < pairs; ++r)
        for (int c = 0; c < K; ++c) {
          bool real = c < (int)columns.size();
          idx[r * K + c] = real ? columns[c] : -1;
          val[r * K + c] = real ? 1.0f : 0.0f;
        }
      if (p.meanFeatures && !columns.empty()) {
        // the dense mean of identical empty rows, re-sparsed: each listed column, in column order, value 1
        std::vector<int> sorted = columns;
        std::sort(sorted.begin(), sorted.end());
        for (size_t r = 0; r < pairs; ++r) for (int c = 0; c < K; ++c) idx[r * K + c] = sorted[c];
      }
      E.i(q + ".featuresIdx", idx);
      E.t(q + ".featuresVal", val);
      E.m(q + ".featuresK", K);
    }
    E.m("template.passes", (double)passes.size());
    E.m("template.templates", D.chaiTemplates ? 1 : TEMPLATES);
    E.m("template.featureWidth", width);
    E.flag("template.outerResidual", D.templateStackOuterResidual);
  }
  try { out.pdb = pdbTemplateText(batch); }
  catch (const std::exception& e) { out.said.push_back(std::string("no PDB template (") + e.what() + "); the native writer will name residues itself"); }
  return out;
}

}  // namespace lf
