// esmfold2-featurise: cuda/esmfold2/export_input.mjs without Node - see esmfold2_export.h.
#include <cstdio>
#include <string>
#include <vector>

#include "esmfold2_export.h"

int main(int argc, char** argv) {
  std::vector<std::string> list(argv + 1, argv + argc);
  lf::Args args(list);
  try {
    lf::Esmfold2Export out = lf::exportEsmfold2(args);
    if (system(("mkdir -p '" + args.positional + "'").c_str()) != 0) throw std::runtime_error("cannot create " + args.positional);
    out.entries.write(args.positional);
    FILE* f = fopen((args.positional + "/pdb.template").c_str(), "wb");
    fwrite(out.pdb.data(), 1, out.pdb.size(), f);
    fclose(f);
    for (auto& line : out.said) printf("%s\n", line.c_str());
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
