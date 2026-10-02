// AlphaFold 3 in CUDA - the driver.
//
//   af3 <data-dir> [--fast] [--stages] [--repeat=N]
//
// Reads <data-dir>/model.{idx,bin} (export-model.mjs). Without --fast it runs the precise
// path (f32 throughout) and checks every seam the oracle recorded; with --fast the f16 path.
#include "trunk.cuh"
#include "atom.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: af3 <data-dir> [--fast] [--stages] [--repeat=N]\n"); return 1; }
  bool fast = false; int repeat = 1, msaCap = 1024;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--fast")) fast = true;
    else if (!strcmp(argv[i], "--stages")) STAGES = true;
    else if (!strncmp(argv[i], "--repeat=", 9)) repeat = atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--msa=", 6)) msaCap = atoi(argv[i] + 6);
  }
  auto t0 = std::chrono::steady_clock::now();
  M.load(argv[1]);
  CB(cublasCreate(&H));
  CB(cublasSetStream(H, STREAM));
  printf("loaded %zu entries in %.1f s; %d tokens\n", M.index.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
         (int)M.meta("batch.tokens"));

  // target_feat from the batch: per-atom conditioning and the atom cross-attention encoder.
  int tokens = (int)M.meta("batch.tokens");
  float* tfDev = buildTargetFeat();
  check("target_feat", tfDev, (size_t)tokens * 447, "oracle.trunk.stages.target_feat");
  std::vector<float> targetFeat = download(tfDev, (size_t)tokens * 447);
  bool oracleTargetFeat = false;
  for (int i = 2; i < argc; ++i) if (!strcmp(argv[i], "--oracle-target-feat")) oracleTargetFeat = true;
  if (oracleTargetFeat)
    targetFeat.assign(M.f("oracle.trunk.stages.target_feat"), M.f("oracle.trunk.stages.target_feat") + (size_t)tokens * 447);
  Trunk t = makeTrunk(targetFeat.data(), msaCap);
  printf("trunk: %d tokens, %d MSA rows, pair %d, single %d, msa %d; %s path\n", t.n, t.S, t.C, t.Cs, t.Cm,
         fast ? "f16" : "f32");
  std::function<void(const char*, const float*, size_t)> seam = [&](const char* name, const float* d, size_t n) {
    std::string tap = std::string("oracle.trunk.stages.tap.") + name;
    std::string plain = std::string("oracle.trunk.stages.") + name;
    check(name, d, n, M.has(tap) ? tap : plain);
  };
  std::function<void(const char*, const float*, size_t)> quiet = [](const char*, const float*, size_t) {};
  for (int it = 0; it < repeat; ++it) {
    CK(cudaDeviceSynchronize());
    if (it == 1) STAGE_MS.clear();
    auto s = std::chrono::steady_clock::now();
    stage(nullptr);
    if (fast) runTrunk<half>(t, it == 0 ? seam : quiet);
    else runTrunk<float>(t, it == 0 ? seam : quiet);
    CK(cudaDeviceSynchronize());
    printf("trunk pass %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count());
  }
  if (STAGES) {
    double total = 0; for (auto& [k, v] : STAGE_MS) total += v;
    for (auto& [k, v] : STAGE_MS) printf("  %-16s %9.1f ms  %4.1f%%\n", k.c_str(), v / std::max(1, repeat - 1), 100 * v / total);
  }
  return 0;
}
