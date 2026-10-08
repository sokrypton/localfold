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

// what each port takes: the short form with no arguments, the whole reference with --help (cuda/<port>/README.md,
// "Guide", says the same at more length)
static const char* AF3_HELP = R"(af3 - AlphaFold 3 and the seven models of its lineage, natively: a job in, a structure out.

usage: af3 (--job=<job.json> | --sequence=<SEQ>[:<SEQ>...]) [--model=<model>] [--out=<path>] [options]

  af3 --job=kras.json --out=kras.pdb
  af3 --sequence=MKV...:GSH... --model=boltz2 --search --samples=5 --out=complex.cif

The weights are downloaded the first time, the input featurised in this process while the GPU starts, and the
fold writes the structure and its confidences.

MODELS (--model=)
  af3            AlphaFold 3 (default) - Google DeepMind's parameters, academic non-commercial use only: the
                 first download asks you to accept the terms, or set LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3
  boltz2  chai1  protenix2  intellifold2  rosettafold3  opendde  openbind0
                 (chai1 also downloads ESM2 3B, 2.5 GB; rosettafold3 and chai1 have no Flow sampler)

INPUT
  --job=<job.json>         an AlphaFold 3 job file, either dialect: proteins, DNA, RNA, ligands (CCD or SMILES),
                           ions, glycans, modified residues and bases, covalent bonds, inline alignments and
                           templates, userCCD, modelSeeds
  --sequence=<A>:<B>       the chains, joined by ':'
  --kinds=protein,dna,rna  each chain's kind, in order (protein when absent)
  --ligands=GOL,ATP        ligands by CCD code
  --smiles='CCO|c1ccccc1'  ligands by SMILES, '|'-separated (the conformer is built here)
  --modify=SEP@3[@<chain>] modified residues or bases: CODE@position (1-based), comma-separated

ALIGNMENT (none by default: single sequence)
  --search                 the protein chains' alignments from the ColabFold MMseqs2 server, paired for a
                           complex - SENDS YOUR SEQUENCES TO api.colabfold.com; kept as <out>.a3m
  --a3m=<a.a3m>[,<b.a3m>]  one alignment per protein chain
  --paired-a3m=<...>       a complex's paired rows
  --max-msa=1024           rows kept

TEMPLATES (up to four slots)
  --template=<file>:<chain>[@<query chain>]
                           a PDB or mmCIF chain aligned to a query chain (0-based); '+' joins parts into one
                           slot spanning chains, ',' separates slots:
                           --template=1brs.pdb:A@0+1brs.pdb:D@1
  --search-templates       each protein chain's best four hits from the MMseqs2 search

FOLD
  --out=<path>             fold.pdb by default; a .cif name writes mmCIF
  --samples=N              diffusion samples per seed, ranked by AlphaFold 3's ranking score (1)
  --seed=N | --seeds=a,b   the job's modelSeeds, else 42
  --recycles=N             trunk recycles, N+1 passes (3)
  --recycle-tolerance=<A>  stop recycling once the prediction moves less than this
  --steps=N                diffusion steps (200)
  --flow                   the Flow sampler instead of diffusion
  --af3-defaults           AlphaFold 3's own run settings: 10 recycles, 5 samples
  --frames=<dir>           each intermediate result as it lands: contacts-PP-of-NN.u8 after every trunk
                           pass, frame-SSSS.pdb every diffusion step
  --save-embeddings  --save-distogram

OUTPUT (for --out=fold.pdb)
  fold.pdb                        the structure (the best sample), B-factors pLDDT
  fold_confidences.json           per-atom pLDDT, PAE, contact probabilities (AlphaFold 3's layout)
  fold_summary_confidences.json   pTM, ipTM, per-chain scores, has_clash, ranking_score
  fold_sample<k>.*, fold_ranking_scores.csv   with several samples or seeds
  fold.a3m                        the searched alignment

ENVIRONMENT
  LOCALFOLD_HOME                  where the weights are kept (the checkout by default)
  LOCALFOLD_ACCEPT_MODEL_TERMS    alphafold3: accept DeepMind's terms without the prompt

A refusal is one line, 'Error: ...', and a nonzero exit. More: cuda/af3/README.md.
)";

static const char* AF2_HELP = R"(af2 - AlphaFold 2, monomer and multimer, natively: a job in, a structure out.

usage: af2 (--job=<job.json> | --sequence=<SEQ>[:<SEQ>...]) [--model=<model>] [--out=<path>] [options]

  af2 --sequence=MKV... --search --out=fold.pdb
  af2 --sequence=<A>:<B> --model=model_1_multimer_v3 --search --out=complex.pdb

The weights are downloaded the first time, the input featurised in this process while the GPU starts, and the
fold prints a line per pass (pLDDT, pTM, ipTM, how far the structure moved) and writes the structure and its
confidences.

PROTEIN CHAINS ONLY: no ligands, nucleic acids, modified residues or covalent bonds - a job or flag carrying
one is refused, not folded without it (cuda/af3 and cuda/esmfold2 take them).

MODELS (--model=)
  model_1_ptm (default), model_2_ptm      the monomer, with a template embedder
  model_3_ptm, model_4_ptm, model_5_ptm   the monomer's template-free models
  model_1_multimer_v3 ... model_5_multimer_v3
  (model 1 of each is downloaded whole, 74 MB; models 2-5 as 43 MB differences on it)

INPUT
  --job=<job.json>         an AlphaFold 3 job file whose sequences are all proteins
  --sequence=<A>:<B>       the chains, joined by ':'

ALIGNMENT (none by default: single sequence)
  --search                 from the ColabFold MMseqs2 server, paired for a complex - SENDS YOUR SEQUENCES TO
                           api.colabfold.com; kept as <out>.a3m
  --a3m=<a.a3m>[,<b.a3m>]  one alignment per chain
  --paired-a3m=<...>       a complex's paired rows
  --max-msa=512            cluster centres
  --max-extra=1024         extra rows
  --seed=N                 seeds which rows are sampled (0)

TEMPLATES (one slot)
  --template=<file>:<chain>[@<query chain>][+...]
                           a PDB or mmCIF chain aligned to a query chain (0-based), '+' adding parts:
                           --template=1brs.pdb:A@0+1brs.pdb:D@1  (not on monomer models 3-5)
  --template-search-chains=0,1
                           with --search: the best hit for each listed chain

FOLD
  --out=<path>             fold.pdb by default
  --recycles=N             N+1 passes (3)
  --tolerance=<A>          stop early once a pass moves the structure less than this
  --frames=<dir>           each pass as it lands: pass-PP-of-NN.pdb and .json, pae-PP-of-NN.u8,
                           contacts-PP-of-NN.u8

OUTPUT (for --out=fold.pdb)
  fold.pdb                        the structure, B-factors pLDDT
  fold_confidences.json           per-atom pLDDT, PAE, contact probabilities (AlphaFold 3's layout)
  fold_summary_confidences.json   pTM, ipTM, mean pLDDT
  fold.a3m                        the searched alignment

ENVIRONMENT
  LOCALFOLD_HOME                  where the weights are kept (the checkout by default)

A refusal is one line, 'Error: ...', and a nonzero exit. More: cuda/af2/README.md.
)";

static const char* EF2_HELP = R"(esmfold2 - ESMFold2, natively: a sequence (or job) in, a structure out, no alignment needed.

usage: esmfold2 (--job=<job.json> | --sequence=<SEQ>[:<SEQ>...]) [--model=<model>] [--out=<path>] [options]

  esmfold2 --sequence=MKV... --out=fold.pdb
  esmfold2 --job=calmodulin_4calcium.json --model=esmfold2-fast-300m --out=cam.pdb

Its language model stands in for the alignment, so nothing leaves the machine. The weights are downloaded the
first time, the input featurised in this process while the GPU starts, and the fold writes the structure and
its confidences.

MODELS (--model=)
  esmfold2-fast-600m   the website's ESMFold2 (default): ESM-C 600M, 129 MB + 224 MB
  esmfold2-fast-300m   its 300M sibling, 129 MB + 130 MB
  esmfold2-fast, esmfold2
                       the released checkpoints on ESM-C 6B, from local exports only (cuda/esmfold2/README.md)

INPUT
  --job=<job.json>         an AlphaFold 3 job file: proteins, DNA, RNA, ligands (CCD or SMILES), ions, modified
                           residues, covalent bonds
  --sequence=<A>:<B>       the chains, joined by ':'
  --kinds=protein,dna,rna  each chain's kind, in order (protein when absent)
  --ligands=GOL,ATP        ligands by CCD code
  --smiles='CCO|c1ccccc1'  ligands by SMILES, '|'-separated
  --modify=SEP@3[@<chain>] modified residues: CODE@position (1-based), comma-separated
  --a3m=<a.a3m>[,<b.a3m>]  the released esmfold2-fast and esmfold2 only; the fast 600M/300M read no alignment
  (no templates)

FOLD
  --out=<path>             fold.pdb by default
  --seed=N                 the sampler's noise (0)
  --steps=N                scheduled sampler steps: 15 (11 run), raised to 64 when the input has a ligand,
                           ion or modified residue, which fewer steps tear apart
  --frames=<dir>           intermediate results as they land: contacts-00-of-01.u8 after the trunk,
                           frame-SSSS-NNNN.pdb every sampler step

OUTPUT (for --out=fold.pdb)
  fold.pdb                        the structure, B-factors pLDDT
  fold_confidences.json           per-atom pLDDT, PAE, contact probabilities (AlphaFold 3's layout)
  fold_summary_confidences.json   pTM, ipTM, mean pLDDT

ENVIRONMENT
  LOCALFOLD_HOME                  where the weights are kept (the checkout by default)

A refusal is one line, 'Error: ...', and a nonzero exit. More: cuda/esmfold2/README.md.
)";

static void usage(const std::string& port, bool full) {
  const char* help = port == "af3" ? AF3_HELP : port == "af2" ? AF2_HELP : EF2_HELP;
  if (full) {
    fputs(help, stdout);
    printf("\nThe weights for this binary live under %s.\n", fetch::home().c_str());
    return;
  }
  // the short form: the help's first lines, up to the examples
  std::string h = help;
  size_t cut = h.find("\n\n", h.find("usage:"));
  cut = h.find("\n\n", cut + 2);
  fprintf(stderr, "%s\n\nrun %s --help for every option\n", h.substr(0, cut).c_str(), port.c_str());
}

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
  // (the experimental tier zeroes the alignment's features and has no MSA encoder: an alignment there is read by
  // nothing, so it is refused rather than dropped; the released esmfold2-fast reads its profile, esmfold2 also encodes it)
  if (!run.args.option("a3m").empty() && (run.model == "esmfold2-fast-600m" || run.model == "esmfold2-fast-300m"))
    throw std::runtime_error(run.model + " folds from the sequence alone (it reads no alignment): --a3m is for the released esmfold2-fast and esmfold2");
  std::vector<std::string> foldArgs = {"--fold-bundle=" + trunk, "--esmc-bundle=" + tower, "--fast"};
  std::string warm = esmfold2WarmShape(run.args);
  if (!warm.empty()) foldArgs.push_back("--warm=" + warm);
  for (auto& f : run.foldFlags()) foldArgs.push_back(f);
  // 🔴 THE PAGE'S FLOOR FOR PER-ATOM TOKENS: a ligand or a modified residue is torn at the checkpoint's 15 scheduled
  // steps (11 run) and whole at 64 (web/esmfold2-model.js ESMFOLD2_ATOMISED_STEPS; cuda/worker.py applies it too)
  if (run.args.option("steps").empty()) {
    bool atomised = !run.args.option("ligands").empty() || !run.args.option("smiles").empty() || !run.args.option("modify").empty();
    if (!run.args.option("job").empty()) {
      Expanded request = expandEntities(jobFromJson(readFile(run.args.option("job"))).entities);
      atomised = atomised || !request.ligands.empty() || !request.modifications.empty();
    }
    if (atomised) foldArgs.push_back("--steps=64");
  }
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
    if (argc < 2) { usage(name, false); return 1; }
    for (int i = 1; i < argc; ++i)
      if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(name, true); return 0; }
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
