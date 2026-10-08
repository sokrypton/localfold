// resolve-templates: cuda/resolve_templates.mjs without Node - the page's template rows, resolved as the page
// resolves them, for a native fold.
//
//   resolve-templates <request.json> <out dir>
//
// <request.json> is what the page sends a remote fold ({entities, ...}). Its rows are expanded as web/entities.js
// expandEntities expands them - one template per chain COPY, numbered over every polymer - and each structure is
// fetched as web/template-source.js fetchStructure fetches it (the RCSB's PDB file, then its mmCIF; the AlphaFold DB
// by accession) or taken as the uploaded text. Each is written to <out dir> and one JSON line printed:
// [{chain, kind, file, chainId, source}] in the page's order; a "from the MSA search" row has no file.
#include <cstdio>
#include <string>
#include <vector>

#include "json.h"
#include "search.h"
#include "templates.h"

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

int main(int argc, char** argv) {
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
