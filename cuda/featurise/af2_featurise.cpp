// af2-featurise: cuda/af2/export_input.mjs without Node - see af2_export.h.
//
//   g++ -std=c++17 -O2 -ffp-contract=off -pthread -o cuda/featurise/af2-featurise cuda/featurise/af2_featurise.cpp
#include <cstdio>
#include <string>
#include <vector>

#include "af2_export.h"

int main(int argc, char** argv) {
  std::vector<std::string> list(argv + 1, argv + argc);
  lf::Args args(list);
  if (args.positional.empty()) {
    fprintf(stderr, "usage: af2-featurise <out dir> (--bundle=<dir> | --weights=<dir>) (--job=<job.json> | --sequence=...)\n");
    return 1;
  }
  try {
    lf::Af2Export out = lf::exportAf2(args);
    if (system(("mkdir -p '" + args.positional + "'").c_str()) != 0) throw std::runtime_error("cannot create " + args.positional);
    if (!out.searchA3m.empty()) {
      FILE* f = fopen((args.positional + "/search.a3m").c_str(), "wb");
      fwrite(out.searchA3m.data(), 1, out.searchA3m.size(), f);
      fclose(f);
    }
    out.entries.write(args.positional);
    for (auto& line : out.said) printf("%s\n", line.c_str());
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
