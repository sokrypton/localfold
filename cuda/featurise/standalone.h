// One port, one command: a job in, a structure out, nothing else to run.
//
//   af3      --job=kras.json --out=kras.pdb [--model=boltz2] [--samples=5 ...]
//   af2      --sequence=<A>:<B> --model=model_1_multimer_v3 --search --out=1brs.pdb
//   esmfold2 --job=calmodulin.json --out=cam.pdb [--model=esmfold2-fast-300m]
//
// Each port's main hands a command whose first argument is a flag here (its old first argument, a featurised
// input directory, is still taken as it was - the resident server and the tests use it). This:
//   1. fetches the model's published weights if they are not on disk (fetch.h: the page's own bytes) - into
//      --weights-dir=<dir> when given, else the checkout the binary was built in (~/.cache/localfold out of one),
//   2. featurises the input IN THIS PROCESS on a thread - the page's featuriser, byte for byte (the same code as
//      cuda/featurise/<port>-featurise, which tools/check-native-featuriser.py holds to the JavaScript) - into a
//      temporary directory, while the port starts CUDA and puts the weights on the device (--wait-input),
//   3. folds, with the port's own flags, and keeps a searched alignment beside the structure (<out>.a3m).
//
// Input flags (--job, --sequence, --a3m, --search, --template, ...) go to the featuriser and the rest to the fold;
// a flag both read (af2's --recycles) goes to both. A refusal is the page's own sentence, as `Error: <sentence>`.
#pragma once
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "af2_export.h"
#include "esmfold2_export.h"
#include "fetch.h"

namespace lf {

// ---------------------------------------------------------------- each featuriser's output, written to a directory
// (shared with cuda/featurise/featurise.cpp, so the standalone fold and the featuriser binary write the same files)

inline void writeText(const std::string& path, const std::string& text) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  fwrite(text.data(), 1, text.size(), f);
  fclose(f);
}

inline void featuriseAf3Into(const Args& args, const std::string& dir) {
  Af3Export out = exportAf3(args);
  for (auto& line : out.said) printf("%s\n", line.c_str());
  fetch::makeDirs(dir);
  if (!out.pdb.empty()) writeText(dir + "/template.pdb", out.pdb);
  if (!out.searchA3m.empty()) writeText(dir + "/search.a3m", out.searchA3m);
  out.entries.write(dir);
  printf("%zu entries, %.0f MiB, tokens %d\n", out.entries.list.size(), out.entries.words() * 4 / 1048576.0, out.tokens);
  fflush(stdout);
}

inline void featuriseAf2Into(const Args& args, const std::string& dir) {
  Af2Export out = exportAf2(args);
  fetch::makeDirs(dir);
  if (!out.searchA3m.empty()) writeText(dir + "/search.a3m", out.searchA3m);
  out.entries.write(dir);
  for (auto& line : out.said) printf("%s\n", line.c_str());
  fflush(stdout);
}

inline void featuriseEsmfold2Into(const Args& args, const std::string& dir) {
  Esmfold2Export out = exportEsmfold2(args);
  fetch::makeDirs(dir);
  out.entries.write(dir);
  writeText(dir + "/pdb.template", out.pdb);
  for (auto& line : out.said) printf("%s\n", line.c_str());
  fflush(stdout);
}

namespace standalone {

// the flags only a featuriser reads (cuda/featurise/*_export.h's args.option/has), which a port's own parser refuses
inline const std::set<std::string>& inputFlags() {
  static const std::set<std::string> s = {"job", "sequence", "a3m", "paired-a3m", "search", "search-templates", "template",
                                          "template-search-chains", "kinds", "ligands", "smiles", "modify", "max-msa",
                                          "max-extra", "no-span-chains"};
  return s;
}
inline std::string flagName(const std::string& a) {
  if (a.compare(0, 2, "--") != 0) return "";
  size_t eq = a.find('=');
  return a.substr(2, eq == std::string::npos ? std::string::npos : eq - 2);
}

struct Run {
  std::vector<std::string> list;    // the command's flags
  Args args{std::vector<std::string>{}};
  std::string model, home, out, dir;
  explicit Run(int argc, char** argv, const std::string& defaultModel) : list(argv + 1, argv + argc), args(list) {
    model = args.option("model", defaultModel);
    out = args.option("out", "fold.pdb");
    home = args.option("weights-dir");          // where the weights are (fetched into it when absent)
    if (home.empty()) home = fetch::home();
  }
  // the flags for the fold: the command's own, minus the featuriser's and this file's (--model, --weights-dir; and the shared
  // ones the caller names)
  std::vector<std::string> foldFlags(const std::set<std::string>& alsoInput = {}) const {
    std::vector<std::string> f;
    for (auto& a : list) {
      std::string n = flagName(a);
      if (n == "model" || n == "weights-dir" || (inputFlags().count(n) && !alsoInput.count(n))) continue;
      f.push_back(a);
    }
    return f;
  }
};

inline std::string tempDir() {
  struct stat st;
  std::string base = stat("/dev/shm", &st) == 0 && access("/dev/shm", W_OK) == 0 ? "/dev/shm" : "/tmp";
  std::string tmpl = base + "/localfold-in-XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end()); buf.push_back(0);
  if (!mkdtemp(buf.data())) throw std::runtime_error("cannot make a temporary directory under " + base);
  return std::string(buf.data()) + "/in";
}

// what the exit leaves behind: the searched alignment beside the structure, and no temporary directory
// (an atexit hook, because a port ends with std::exit; it waits for the featuriser to stop writing)
inline std::string CLEAN_DIR, KEEP_A3M;
inline std::atomic<bool> FEATURISED{false};
inline void cleanUp() {
  for (int k = 0; !FEATURISED && k < 20000; ++k) usleep(1000);
  if (CLEAN_DIR.empty()) return;
  if (!KEEP_A3M.empty() && fetch::exists(CLEAN_DIR + "/search.a3m")) {
    std::string cp = "cp " + search::shellQuote(CLEAN_DIR + "/search.a3m") + " " + search::shellQuote(KEEP_A3M);
    if (system(cp.c_str()) == 0) fprintf(stderr, "the searched alignment -> %s\n", KEEP_A3M.c_str());
  }
  std::string parent = CLEAN_DIR.substr(0, CLEAN_DIR.rfind('/'));
  if (system(("rm -rf " + search::shellQuote(parent)).c_str())) {}
}

// featurise on a thread into `run.dir`, then fold: `fold` is the port's own main, handed `foldArgs` with the
// input directory first and --wait-input (it starts CUDA and uploads the weights meanwhile)
inline int featuriseAndFold(Run& run, const std::function<void(const std::string&)>& featurise,
                            std::vector<std::string> foldArgs, int (*fold)(int, char**), const char* argv0) {
  run.dir = tempDir();
  CLEAN_DIR = run.dir;
  std::string o = run.out;
  KEEP_A3M = (o.size() > 4 && o.substr(o.size() - 4) == ".pdb" ? o.substr(0, o.size() - 4) : o) + ".a3m";
  std::atexit(cleanUp);
  std::string dir = run.dir;
  std::thread([featurise, dir] {
    try {
      featurise(dir);
    } catch (const std::exception& e) {
      fprintf(stderr, "Error: %s\n", e.what()); fflush(stderr);
      try { fetch::makeDirs(dir); writeText(dir + "/model.failed", std::string(e.what()) + "\n"); } catch (...) {}
    }
    FEATURISED = true;
  }).detach();
  std::vector<std::string> a = {argv0, run.dir};
  for (auto& f : foldArgs) a.push_back(f);
  a.push_back("--wait-input");
  std::vector<char*> argv;
  for (auto& s : a) argv.push_back(s.data());
  argv.push_back(nullptr);
  return fold((int)a.size(), argv.data());
}

// an af2 warm-up shape from the command, as cuda/af2/fold guessed it: L,N,E,T from the sequence, the alignment's
// depth and the templates (a wrong guess warms less and never changes the fold; a job's size is unknown: none)
inline std::string af2WarmShape(const Args& args) {
  std::string seq = args.option("sequence"), a3m = args.option("a3m");
  std::string residues;
  for (char c : seq) if (c != ':') residues += c;
  int depth = 1;
  if (!a3m.empty()) {
    std::string text = readFile(a3m.substr(0, a3m.find(',')));
    depth = 0;
    std::string first;
    for (size_t at = 0; at < text.size();) {
      size_t end = text.find('\n', at); if (end == std::string::npos) end = text.size();
      std::string line = text.substr(at, end - at);
      if (!line.empty() && line[0] == '>') ++depth;
      else if (depth == 1 && residues.empty()) for (char c : line) if (c >= 'A' && c <= 'Z') first += c;
      at = end + 1;
    }
    if (residues.empty()) residues = first;
  }
  if (residues.empty()) return "";
  int maxMsa = atoi(args.option("max-msa", "512").c_str()), maxExtra = atoi(args.option("max-extra", "1024").c_str());
  if (args.has("search")) depth = maxMsa + maxExtra;          // (a search is usually deep)
  int n = std::min(depth, maxMsa), e = std::max(1, std::min(depth - n, maxExtra));
  std::string t = args.option("template");
  int templates = t.empty() ? 0 : 1 + (int)std::count(t.begin(), t.end(), ',');
  return std::to_string(residues.size()) + "," + std::to_string(n) + "," + std::to_string(e) + "," + std::to_string(templates);
}

// an esmfold2 warm-up shape: T,A - tokens = residues, atoms = their heavy atoms
inline std::string esmfold2WarmShape(const Args& args) {
  std::string seq = args.option("sequence");
  static const std::string HEAVY = "A5R11N8D8C6Q9E9G4H10I8L8K9M8F11P7S6T7W14Y12V7";
  int tokens = 0, atoms = 1;
  for (char c : seq) {
    if (c == ':') continue;
    ++tokens;
    size_t at = HEAVY.find(c);
    atoms += at == std::string::npos ? 8 : atoi(HEAVY.c_str() + at + 1);
  }
  return tokens ? std::to_string(tokens) + "," + std::to_string(atoms) : "";
}

}  // namespace standalone
}  // namespace lf
