// af3-featurise: cuda/af3/export-model.mjs --no-weights without Node - see af3_export.h.
//
//   g++ -std=c++17 -O2 -ffp-contract=off -o cuda/featurise/af3-featurise cuda/featurise/af3_featurise.cpp
#include <cstdio>
#include <string>
#include <vector>

#include "af3_export.h"

int main(int argc, char** argv) {
  std::vector<std::string> list(argv + 1, argv + argc);
  lf::Args args(list);
  if (args.positional.empty()) {
    fprintf(stderr, "usage: af3-featurise <out dir> --no-weights --family=<model> (--job=<job.json> | --sequence=...)\n");
    return 2;
  }
  try {
    lf::Af3Export out = lf::exportAf3(args);
    for (auto& line : out.said) printf("%s\n", line.c_str());
    if (system(("mkdir -p '" + args.positional + "'").c_str()) != 0) throw std::runtime_error("cannot create " + args.positional);
    if (!out.pdb.empty()) {
      if (system(("mkdir -p '" + args.positional + "'").c_str()) != 0) throw std::runtime_error("cannot create " + args.positional);
      FILE* f = fopen((args.positional + "/template.pdb").c_str(), "wb");
      fwrite(out.pdb.data(), 1, out.pdb.size(), f);
      fclose(f);
    }
    if (!out.searchA3m.empty()) {
      FILE* f = fopen((args.positional + "/search.a3m").c_str(), "wb");
      fwrite(out.searchA3m.data(), 1, out.searchA3m.size(), f);
      fclose(f);
    }
    out.entries.write(args.positional);
    printf("%zu entries, %.0f MiB, tokens %d\n", out.entries.list.size(), out.entries.words() * 4 / 1048576.0, out.tokens);
  } catch (const lf::Refusal& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
