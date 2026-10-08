// The native featurisers, one binary named four ways - cuda/build.sh links each name to it:
//
//   af3-featurise <out dir> --no-weights --family=<model> (--job=<job.json> | --sequence=...)    (af3_export.h)
//   af2-featurise <out dir> (--bundle=<dir> | --weights=<dir>) (--job=<job.json> | --sequence=...)   (af2_export.h)
//   esmfold2-featurise <out dir> (--job=<job.json> | --sequence=...)                              (esmfold2_export.h)
//   resolve-templates <request.json> <out dir>                                                   (below)
//   fetch-weights [--weights-dir=<dir>] <model>...          what each model's binary reads (fetch.h), by its --model
//                                                            name (boltz2, model_3_ptm, esmfold2-fast-600m, ...)
//   chem-probe < smiles.txt         each SMILES's component, one line, as tools/chem-probe.mjs prints the page's
//
// each the JavaScript it replaces (cuda/af3/export-model.mjs --no-weights, cuda/af2/export_input.mjs,
// cuda/esmfold2/export_input.mjs, cuda/resolve_templates.mjs), byte for byte - tools/check-native-featuriser.py.
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
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights-dir=", 14)) root = argv[i] + 14;
    else names.push_back(argv[i]);
  }
  if (names.empty()) {
    fprintf(stderr, "usage: fetch-weights [--weights-dir=<dir>] <model>...\n  models: %s\n", lf::fetch::modelNames().c_str());
    return 2;
  }
  try {
    if (root.empty()) root = lf::fetch::home();
    for (auto& n : names) for (auto& d : lf::fetch::model(root, n).dirs) printf("%s: %s\n", n.c_str(), d.c_str());
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
  if (self == "esmfold2-featurise") return esmfold2Main(argc, argv);
  if (self == "resolve-templates") return resolveMain(argc, argv);
  if (self == "chem-probe") return chemMain();
  if (self == "fetch-weights") return fetchMain(argc, argv);
  fprintf(stderr, "featurise: run as af3-featurise, af2-featurise, esmfold2-featurise, resolve-templates or fetch-weights (not %s)\n", self.c_str());
  return 2;
}
