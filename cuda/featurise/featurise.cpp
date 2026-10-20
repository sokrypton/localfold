// The native featurisers, one binary named four ways - cuda/build.sh links each name to it:
//
//   af3-featurise <out dir> --no-weights --family=<model> (--job=<job.json> | --sequence=...)    (af3_export.h)
//   af2-featurise <out dir> (--bundle=<dir> | --weights=<dir>) (--job=<job.json> | --sequence=...)   (af2_export.h)
//   ef2-featurise <out dir> (--job=<job.json> | --sequence=...)                              (esmfold2_export.h)
//   bioemu2-featurise <out dir> (--job=<job.json> | --sequence=...)                           (bioemu2_export.h)
//   resolve-templates <request.json> <out dir>                                                   (below)
//   fetch-weights [--weights-dir=<dir>] <model>...          what each model's binary reads (fetch.h), by its --model
//                                                            name (boltz2, model_3_ptm, ef2-fast-600m, ...), and
//                                                            `ccd`: wwPDB's whole component dictionary
//   chem-probe < smiles.txt         each SMILES's component, one line, as tools/chem-probe.mjs prints the page's
//
// each the JavaScript it replaces (cuda/af3/export-model.mjs --no-weights, cuda/af2/export_input.mjs,
// cuda/ef2/export_input.mjs, cuda/resolve_templates.mjs), byte for byte - tools/check-native-featuriser.py.
// One binary because the four share almost every header: compiled four times they were ~60 s of CPU here and twice
// that on a two-core Colab runtime, beside the ports' nvcc.
//
//   g++ -std=c++17 -O2 -ffp-contract=off -pthread -o featurise featurise.cpp
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "standalone.h"
#include "json.h"
#include "search.h"
#include "templates.h"

static int af3Main(int argc, char** argv) {
  std::vector<std::string> list(argv + 1, argv + argc);
  lf::Args args(list);
  if (args.positional.empty()) {
    fprintf(stderr, "usage: af3-featurise <out dir> --no-weights --family=<model> (--job=<job.json> | --sequence=...)\n");
    return 2;
  }
  try {
    lf::featuriseAf3Into(args, args.positional);
  } catch (const lf::Refusal& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}

static int af2Main(int argc, char** argv) {
  std::vector<std::string> list(argv + 1, argv + argc);
  lf::Args args(list);
  if (args.positional.empty()) {
    fprintf(stderr, "usage: af2-featurise <out dir> (--bundle=<dir> | --weights=<dir>) (--job=<job.json> | --sequence=...)\n");
    return 1;
  }
  try {
    lf::featuriseAf2Into(args, args.positional);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}

static int esmfold2Main(int argc, char** argv) {
  std::vector<std::string> list(argv + 1, argv + argc);
  lf::Args args(list);
  try {
    lf::featuriseEsmfold2Into(args, args.positional);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}

static int bioemu2Main(int argc, char** argv) {
  std::vector<std::string> list(argv + 1, argv + argc);
  lf::Args args(list);
  try {
    lf::featuriseBioemu2Into(args, args.positional);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}

namespace {

std::string jsonString(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out += (char)c;
    }
  }
  return out + "\"";
}

std::string field(const lf::Json* object, const char* key) {
  const lf::Json* v = object ? object->get(key) : nullptr;
  return v && v->isString() ? v->s : "";
}

}  // namespace

static int resolveMain(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: resolve-templates <request.json> <out dir>\n"); return 1; }
  try {
    lf::Json request = lf::parseJson(lf::readFile(argv[1]));
    std::string out = argv[2];
    if (system(("mkdir -p " + lf::search::shellQuote(out)).c_str()) != 0) throw std::runtime_error("cannot create " + out);
    const lf::Json* entities = request.get("entities");
    struct Row { int chain; const lf::Json* t; };
    std::vector<Row> rows;
    int chains = 0;
    static const std::vector<std::string> KINDS = {"none", "pdb", "afdb", "search", "upload"};
    if (entities && entities->isArray())
      for (auto& e : entities->a) {
        std::string type = field(&e, "type");
        if (type == "contact") continue;
        const lf::Json* c = e.get("copies");
        int copies = c && c->isNumber() ? (int)c->n : 1;
        for (int copy = 0; copy < copies; ++copy) {
          if (type != "protein" && type != "dna" && type != "rna") continue;
          const lf::Json* t = e.get("template");
          std::string kind = field(t, "kind");
          if (std::find(KINDS.begin(), KINDS.end(), kind) == KINDS.end()) kind = "none";
          bool asked = kind == "search" || (kind == "upload" && !field(t, "text").empty())
                       || (kind != "none" && kind != "upload" && !lf::trimWs(field(t, "source")).empty());
          if (asked && type == "protein") rows.push_back({chains, t});
          ++chains;
        }
      }
    std::string printed = "[";
    for (size_t k = 0; k < rows.size(); ++k) {
      const lf::Json* t = rows[k].t;
      std::string kind = field(t, "kind"), source = lf::trimWs(field(t, "source"));
      std::string entry;
      if (kind == "search") {
        entry = "{\"chain\": " + std::to_string(rows[k].chain) + ", \"kind\": \"search\"}";
      } else {
        std::string text, chainId, named;
        bool hasChainId = false;
        if (kind == "upload") {
          text = field(t, "text");
          hasChainId = !source.empty(); chainId = source;
          named = field(t, "filename").empty() ? "the uploaded structure" : field(t, "filename");
        } else {
          if (kind == "none" || source.empty()) continue;
          // parseSource: "1QYS_A", "1QYS:A" or an accession; a four-character id is the PDB's
          size_t cut = source.find_first_of("_:");
          std::string id = lf::upper(source.substr(0, cut));
          std::string chain = cut == std::string::npos ? "" : source.substr(cut + 1, source.find_first_of("_:", cut + 1) - cut - 1);
          std::vector<std::string> urls;
          if (kind == "pdb") {
            urls = {"https://files.rcsb.org/download/" + id + ".pdb", "https://files.rcsb.org/download/" + id + ".cif"};
          } else {
            lf::search::Response r = lf::search::httpOnce("https://alphafold.ebi.ac.uk/api/prediction/" + id, nullptr);
            if (r.status < 200 || r.status >= 300) throw std::runtime_error(std::to_string(r.status) + " asking AlphaFold DB about " + id);
            lf::Json entries = lf::parseJson(r.body);
            const lf::Json* first = entries.isArray() && !entries.a.empty() ? &entries.a[0] : nullptr;
            std::string url = field(first, "pdbUrl");
            if (url.empty()) throw std::runtime_error("AlphaFold DB has no structure for " + id);
            urls = {url};
          }
          std::string last;
          bool got = false;
          for (auto& url : urls) {
            lf::search::Response r = lf::search::httpOnce(url, nullptr);
            if (r.status >= 200 && r.status < 300) { text = r.body; got = true; break; }
            last = std::to_string(r.status) + " for " + url;
          }
          if (!got) throw std::runtime_error(last.empty() ? "could not fetch " + source : last);
          hasChainId = !chain.empty(); chainId = chain;
          named = source;
        }
        std::string file = out + "/template-" + std::to_string(k) + (lf::looksLikeCif(text) ? ".cif" : ".pdb");
        FILE* f = fopen(file.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + file);
        fwrite(text.data(), 1, text.size(), f);
        fclose(f);
        entry = "{\"chain\": " + std::to_string(rows[k].chain) + ", \"kind\": " + jsonString(kind) + ", \"file\": " + jsonString(file)
              + ", \"chainId\": " + (hasChainId ? jsonString(chainId) : "null") + ", \"source\": " + jsonString(named) + "}";
      }
      printed += (printed.size() > 1 ? ", " : "") + entry;
    }
    printf("%s]\n", printed.c_str());
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}

static int chemMain() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    try {
      lf::Component c = lf::chem::smilesComponent(line, "LIG");
      std::cout << "OK";
      for (auto& a : c.atoms) std::cout << " " << a.name << ":" << a.element << ":" << lf::jsNumber(a.charge) << ":" << lf::jsNumber(a.x) << ":" << lf::jsNumber(a.y) << ":" << lf::jsNumber(a.z);
      std::cout << " |";
      for (auto& b : c.bonds) std::cout << " " << b.from << "-" << b.to << ":" << b.order;
      std::cout << "\n";
    } catch (const std::exception& e) { std::cout << "ERR " << e.what() << "\n"; }
  }
  return 0;
}

static int fetchMain(int argc, char** argv) {
  std::string root;
  std::vector<std::string> names;
  bool help = false;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights-dir=", 14)) root = argv[i] + 14;
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) help = true;
    else if (argv[i][0] == '-') { fprintf(stderr, "Error: unknown flag %s (--weights-dir=<dir> takes an =)\n", argv[i]); return 2; }
    else names.push_back(argv[i]);
  }
  if (names.empty() || help) {
    fprintf(stderr, "usage: fetch-weights [--weights-dir=<dir>] <model>|ccd ...\n  models: %s\n  ccd: wwPDB's whole chemical component dictionary, for folding offline\n",
            lf::fetch::modelNames().c_str());
    return help ? 0 : 2;
  }
  try {
    if (root.empty()) root = lf::fetch::home();
    for (auto& n : names) {
      if (n == "ccd") { printf("ccd: %s\n", lf::fetch::ccd(root).c_str()); continue; }      // (the dictionary, whole)
      for (auto& d : lf::fetch::model(root, n).dirs) printf("%s: %s\n", n.c_str(), d.c_str());
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}

// `msa-search <out.json> --sequence=<A>:<B>:... [--name=<name>]`: the ColabFold MMseqs2 search alone - each distinct
// chain searched once, the distinct chains paired - written as an AlphaFold 3 job (the alphafold3 dialect) carrying
// every protein's unpairedMsa and pairedMsa inline, identical chains one entity with several ids. Any port folds it with
// --job=, so a search runs once, ahead of the GPUs, and the job is the input as published. SENDS THE SEQUENCES TO
// api.colabfold.com
static std::string jsonQuoted(const std::string& s) {
  std::string o = "\"";
  o.reserve(s.size() + 2);
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o + "\"";
}
// a window's A3M - its query the residues [off, off + seg.size()) of a chain of `len` - re-written over the whole chain:
// every row padded with gaps outside the window (its insertions kept), in its order; a row that is the window's query
// becomes the whole query (dropQuery: dropped - an unpaired block opens with its query, which the chain's own first row
// already is; a paired block keeps it, its rows being row-aligned across the chains)
static std::string widenA3m(const std::string& a3m, const std::string& seg, const std::string& full, size_t off, bool dropQuery) {
  std::string out, header, row;
  auto flush = [&] {
    if (header.empty()) return;
    size_t aligned = 0; std::string upper;
    for (char c : row) if (!(c >= 'a' && c <= 'z')) { ++aligned; upper += c; }
    if (aligned != seg.size()) throw std::runtime_error("a window's A3M row is " + std::to_string(aligned) + " columns, the window " + std::to_string(seg.size()));
    if (upper == seg) { if (!dropQuery) out += header + "\n" + full + "\n"; }
    else out += header + "\n" + std::string(off, '-') + row + std::string(full.size() - off - seg.size(), '-') + "\n";
    header.clear(); row.clear();
  };
  size_t at = 0;
  while (at < a3m.size()) {
    size_t end = a3m.find('\n', at); if (end == std::string::npos) end = a3m.size();
    std::string line = a3m.substr(at, end - at);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty() && line[0] == '>') { flush(); header = line; }
    else row += line;
    at = end + 1;
  }
  flush();
  return out;
}
// a chain longer than the server searches whole (it ended TycC's 6,486 residues and BacA's 5,255 with ERROR): windows of
// WINDOW residues overlapping by OVERLAP, each searched alone and widened back over the chain
constexpr size_t WINDOW = 3000, OVERLAP = 300;
static std::vector<size_t> windowsOf(size_t len) {
  std::vector<size_t> starts;
  for (size_t s = 0; ; s += WINDOW - OVERLAP) { starts.push_back(s); if (s + WINDOW >= len) break; }
  return starts;
}

// ...and the paired search, every chain's first window at once, takes no more than PAIRED_RESIDUES in all: the server
// paired 8,607 (bacitracin's three) and ended 11,273 (gramicidin's four) with ERROR - each chain's window cut to one
// length where the sum would pass it
constexpr size_t PAIRED_RESIDUES = 8000;
static size_t pairedCut(const std::vector<std::string>& unique) {
  auto total = [&](size_t cut) { size_t t = 0; for (auto& q : unique) t += std::min(q.size(), cut); return t; };
  size_t lo = 1, hi = WINDOW;
  if (total(hi) <= PAIRED_RESIDUES) return hi;
  while (lo < hi) { const size_t mid = (lo + hi + 1) / 2; if (total(mid) <= PAIRED_RESIDUES) lo = mid; else hi = mid - 1; }
  return lo;
}
// every search's answer kept in <out>.searches/, named by its query (FNV-1a), so a rerun after a failure - an hour of
// windows lost to the last step, once - takes what is already there
static std::string keptSearch(const std::string& dir, const std::string& key, const std::function<std::string()>& run) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : key) { h ^= c; h *= 1099511628211ull; }
  char name[32]; snprintf(name, sizeof name, "/%016llx.a3m", (unsigned long long)h);
  const std::string path = dir + name;
  if (lf::fetch::exists(path)) return lf::readFile(path);
  std::string text = run();
  lf::writeText(path + ".part", text);
  if (rename((path + ".part").c_str(), path.c_str())) throw std::runtime_error("cannot write " + path);
  return text;
}

static int msaMain(int argc, char** argv) {
  std::vector<std::string> list(argv + 1, argv + argc);
  lf::Args args(list);
  if (args.positional.empty() || args.option("sequence").empty()) {
    fprintf(stderr, "usage: msa-search <out.json> --sequence=<A>:<B>:... [--name=<name>]\n");
    return 1;
  }
  try {
    std::vector<std::string> chains;
    for (auto& c : lf::splitNonEmpty(args.option("sequence"), ':')) chains.push_back(lf::search::normalizedSequence(c));
    std::vector<std::string> unique;
    for (auto& c : chains) if (std::find(unique.begin(), unique.end(), c) == unique.end()) unique.push_back(c);
    auto say = [](const std::string& m) { fprintf(stderr, "%s\n", m.c_str()); fflush(stderr); };
    const std::string kept = args.positional + ".searches";
    lf::fetch::makeDirs(kept);
    auto one = [&](const std::string& q, const std::string& tag) {
      return keptSearch(kept, "unpaired:" + q, [&] { return lf::search::searchOne(q, [&](const std::string& m) { say(tag + ": " + m); }).a3m; });
    };
    std::vector<std::string> unpaired, paired(unique.size());
    for (size_t i = 0; i < unique.size(); ++i) {
      const std::string& q = unique[i];
      const std::string tag = "chain " + std::to_string(i + 1) + " of " + std::to_string(unique.size()) + " (" + std::to_string(q.size()) + " residues)";
      if (q.size() <= WINDOW) {
        unpaired.push_back(one(q, tag));
        continue;
      }
      std::string a = ">101\n" + q + "\n";
      const auto starts = windowsOf(q.size());
      for (size_t w = 0; w < starts.size(); ++w) {
        const std::string seg = q.substr(starts[w], std::min(WINDOW, q.size() - starts[w]));
        const std::string wtag = tag + ", window " + std::to_string(w + 1) + " of " + std::to_string(starts.size());
        a += widenA3m(one(seg, wtag), seg, q, starts[w], true);
      }
      unpaired.push_back(a);
    }
    // the paired search on each chain's first window (pairing is by organism, and the organisms' full-length proteins
    // carry the window), its rows widened back over the chains
    if (unique.size() > 1) {
      const size_t cut = pairedCut(unique);
      std::vector<std::string> heads;
      std::string key = "paired";
      for (auto& q : unique) { heads.push_back(q.substr(0, std::min(cut, q.size()))); key += ":" + heads.back(); }
      if (cut < WINDOW) say("paired: each chain's first " + std::to_string(cut) + " residues (" + std::to_string(PAIRED_RESIDUES) + " in all at most)");
      const std::string joined = keptSearch(kept, key, [&] {
        std::string j;
        for (auto& t : lf::search::searchPaired(heads, [&](const std::string& m) { say("paired: " + m); })) j += t + '\x01';
        return j;
      });
      for (size_t i = 0, at = 0; i < unique.size(); ++i) {
        const size_t end = joined.find('\x01', at);
        if (end == std::string::npos) throw std::runtime_error("a kept paired search has too few chains: " + kept);
        paired[i] = joined.substr(at, end - at); at = end + 1;
        if (unique[i].size() > heads[i].size()) paired[i] = widenA3m(paired[i], heads[i], unique[i], 0, false);
      }
    }
    // chain ids in input order, mmCIF's way: A..Z, AA, AB, ...
    auto chainId = [](int a) { std::string id; for (a = a; ; a = a / 26 - 1) { id.insert(id.begin(), (char)('A' + a % 26)); if (a < 26) break; } return id; };
    std::string j = "{\n \"name\": " + jsonQuoted(args.option("name", "complex")) + ",\n \"modelSeeds\": [1],\n \"dialect\": \"alphafold3\",\n \"version\": 2,\n \"sequences\": [";
    for (size_t u = 0; u < unique.size(); ++u) {
      std::string ids;
      for (size_t c = 0; c < chains.size(); ++c) if (chains[c] == unique[u]) ids += std::string(ids.empty() ? "" : ", ") + jsonQuoted(chainId((int)c));
      j += std::string(u ? "," : "") + "\n  {\"protein\": {\"id\": [" + ids + "], \"sequence\": " + jsonQuoted(unique[u])
         + ", \"unpairedMsa\": " + jsonQuoted(unpaired[u]) + ", \"pairedMsa\": " + jsonQuoted(paired[u]) + ", \"templates\": []}}";
    }
    j += "\n ]\n}\n";
    lf::writeText(args.positional, j);
    fprintf(stderr, "%zu chains (%zu distinct), %zu residues -> %s (%.1f MB)\n", chains.size(), unique.size(),
            [&] { size_t n = 0; for (auto& c : chains) n += c.size(); return n; }(), args.positional.c_str(), j.size() / 1e6);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  std::string self = argv[0];
  self = self.substr(self.rfind('/') == std::string::npos ? 0 : self.rfind('/') + 1);
  // (or the tool as the first argument: `featurise af3-featurise <out dir> ...`)
  if (self == "featurise" && argc > 1) { self = argv[1]; --argc; ++argv; }
  if (self == "af3-featurise") return af3Main(argc, argv);
  if (self == "af2-featurise") return af2Main(argc, argv);
  if (self == "ef2-featurise") return esmfold2Main(argc, argv);
  if (self == "bioemu2-featurise") return bioemu2Main(argc, argv);
  if (self == "resolve-templates") return resolveMain(argc, argv);
  if (self == "chem-probe") return chemMain();
  if (self == "fetch-weights" || self == "localfold-fetch") return fetchMain(argc, argv);   // (the wheel's name for it)
  if (self == "msa-search") return msaMain(argc, argv);
  fprintf(stderr, "featurise: run as af3-featurise, af2-featurise, ef2-featurise, bioemu2-featurise, resolve-templates, fetch-weights or msa-search (not %s)\n", self.c_str());
  return 2;
}
