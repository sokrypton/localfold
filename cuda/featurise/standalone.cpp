// A port as one command - see standalone.h. Compiled by g++ with the featuriser's own flags (cuda/build.sh), so the
// input a standalone fold reads is byte for byte what cuda/featurise/<port>-featurise writes.
#include "standalone.h"
#include "standalone_api.h"

namespace lf::standalone {

// the featuriser's command line: the input directory first (its positional, as cuda/featurise/<port>-featurise
// takes it), then the command's flags and the port's own
static std::function<Args(const std::string&)> with(const std::vector<std::string>& list, std::initializer_list<std::string> extra) {
  std::vector<std::string> flags = list;
  for (auto& e : extra) flags.push_back(e);
  return [flags](const std::string& dir) {
    std::vector<std::string> l = {dir};
    l.insert(l.end(), flags.begin(), flags.end());
    return Args(l);
  };
}

static const char* USAGE =
    "usage: %s --job=<job.json> | --sequence=<SEQ>[:<SEQ>...] [input flags] [--model=<model>] [--out=fold.pdb] [fold flags]\n"
    "  input flags: --a3m=, --paired-a3m=, --search, --template=, --search-templates, --ligands=, --smiles=, --modify=, ...\n"
    "  the weights are fetched once into %s (LOCALFOLD_HOME sets it)\n";

static int af3(Run& run, int (*fold)(int, char**), const char* argv0) {
  static const std::set<std::string> MODELS = {"af3", "openbind0", "opendde", "boltz2", "protenix2", "intellifold2", "rosettafold3", "chai1"};
  if (!MODELS.count(run.model))
    throw std::runtime_error("no model " + run.model + " (af3, openbind0, opendde, boltz2, protenix2, intellifold2, rosettafold3, chai1)");
  std::vector<std::string> foldArgs = {"--bundle=" + fetch::blob(run.home, run.model), "--family=" + run.model};
  if (run.model == "chai1") foldArgs.push_back("--esm-bundle=" + fetch::blob(run.home, "esm2"));
  foldArgs.push_back("--fold");
  foldArgs.push_back("--fast");
  for (auto& f : run.foldFlags()) foldArgs.push_back(f);
  auto args = with(run.list, {"--no-weights", "--family=" + run.model});
  return featuriseAndFold(run, [args](const std::string& dir) { featuriseAf3Into(args(dir), dir); }, foldArgs, fold, argv0);
}

static int af2(Run& run, int (*fold)(int, char**), const char* argv0) {
  const std::string& m = run.model;
  bool monomer = m.size() == 11 && m.compare(0, 6, "model_") == 0 && m.compare(7, 4, "_ptm") == 0;
  bool multimer = m.size() == 19 && m.compare(0, 6, "model_") == 0 && m.compare(7, 12, "_multimer_v3") == 0;
  if ((!monomer && !multimer) || m[6] < '1' || m[6] > '5')
    throw std::runtime_error("no published bundle for " + m + " (model_1_ptm ... model_5_ptm, model_1_multimer_v3 ... model_5_multimer_v3)");
  std::string family = monomer ? "monomer" : "multimer", bundle = fetch::bundle(run.home, family);
  std::vector<std::string> foldArgs = {"--bundle=" + bundle, "--fast"};
  // models 2-5 of each are published as deltas on model 1, read as the page reads them
  if (m[6] != '1') foldArgs.push_back("--delta=" + fetch::bundle(run.home, family + "-" + m[6]));
  std::string warm = af2WarmShape(run.args);
  if (!warm.empty()) foldArgs.push_back("--warm=" + warm);
  for (auto& f : run.foldFlags({"recycles"})) if (flagName(f) != "seed") foldArgs.push_back(f);   // (--seed: the alignment's)
  auto args = with(run.list, {"--bundle=" + bundle});
  return featuriseAndFold(run, [args](const std::string& dir) { featuriseAf2Into(args(dir), dir); }, foldArgs, fold, argv0);
}

static int esmfold2(Run& run, int (*fold)(int, char**), const char* argv0) {
  std::string trunk, tower;
  if (run.model == "esmfold2-fast-600m") { trunk = fetch::bundle(run.home, "ef2-fast-600m"); tower = fetch::bundle(run.home, "esmc"); }
  else if (run.model == "esmfold2-fast-300m") { trunk = fetch::bundle(run.home, "ef2-fast-300m"); tower = fetch::bundle(run.home, "esmc-300m"); }
  else if (run.model == "esmfold2" || run.model == "esmfold2-fast") {       // the released models: local exports
    trunk = run.home + "/model-" + run.model + "-f32"; tower = run.home + "/model-esmc-6b-int8";
    for (auto& b : {trunk, tower})
      if (!fetch::exists(b + "/manifest.json")) throw std::runtime_error("no " + b + ": export it first (cuda/esmfold2/README.md, \"The released models\")");
  } else {
    throw std::runtime_error("--model=" + run.model + ": esmfold2-fast-600m, esmfold2-fast-300m, esmfold2-fast or esmfold2");
  }
  std::vector<std::string> foldArgs = {"--fold-bundle=" + trunk, "--esmc-bundle=" + tower, "--fast"};
  std::string warm = esmfold2WarmShape(run.args);
  if (!warm.empty()) foldArgs.push_back("--warm=" + warm);
  for (auto& f : run.foldFlags()) foldArgs.push_back(f);
  auto args = with(run.list, {"--fold-bundle=" + trunk});
  return featuriseAndFold(run, [args](const std::string& dir) { featuriseEsmfold2Into(args(dir), dir); }, foldArgs, fold, argv0);
}

// the input flags each port's featuriser reads (cuda/featurise/<port>_export.h): any other input flag is refused by
// name rather than dropped - a template handed to ESMFold2 must not fold as though it were not there
static const std::set<std::string>& readsInput(const std::string& port) {
  static const std::set<std::string> AF3 = {"job", "sequence", "a3m", "paired-a3m", "search", "search-templates", "template",
                                            "template-search-chains", "kinds", "ligands", "smiles", "modify", "max-msa", "no-span-chains"};
  static const std::set<std::string> AF2 = {"job", "sequence", "a3m", "paired-a3m", "search", "template", "template-search-chains",
                                            "max-msa", "max-extra"};
  static const std::set<std::string> EF2 = {"job", "sequence", "a3m", "kinds", "ligands", "smiles", "modify"};
  return port == "af3" ? AF3 : port == "af2" ? AF2 : EF2;
}

int main(const char* port, int argc, char** argv, int (*fold)(int, char**)) {
  std::string name = port;
  try {
    if (argc < 2 || !strcmp(argv[1], "--help")) {
      fprintf(stderr, USAGE, name.c_str(), fetch::home().c_str());
      return argc < 2 ? 1 : 0;
    }
    Run run(argc, argv, name == "af3" ? "af3" : name == "af2" ? "model_1_ptm" : "esmfold2-fast-600m");
    for (auto& a : run.list) {
      std::string n = flagName(a);
      if (inputFlags().count(n) && !readsInput(name).count(n)) throw std::runtime_error(name + " takes no --" + n);
    }
    if (name == "af3") return af3(run, fold, argv[0]);
    if (name == "af2") return af2(run, fold, argv[0]);
    return esmfold2(run, fold, argv[0]);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}

}  // namespace lf::standalone
