// AlphaFold 3 in CUDA - the driver.
//
//   af3 <data-dir> [--fast] [--stages] [--repeat=N]
//
// Reads <data-dir>/model.{idx,bin} (export-model.mjs). Without --fast it runs the precise
// path (f32 throughout) and checks every seam the oracle recorded; with --fast the f16 path.
#include "trunk.cuh"
#include "atom.cuh"
#include "diffusion.cuh"
#include "sampler.cuh"
#include "confidence.cuh"
#include "benchops.cuh"
#include "profile.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: af3 <data-dir> [--fast] [--stages] [--repeat=N]\n"); return 1; }
  bool fast = false, doFold = false, profile = false; int repeat = 1, msaCap = 1024, steps = 200, recycles = 0, folds = 1;
  uint64_t seed = 42; std::string out = "fold.pdb";
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--fast")) fast = DIFF_HALF = ATOM_HALF = CONF_HALF = true;
    else if (!strcmp(argv[i], "--stages")) STAGES = true;
    else if (!strncmp(argv[i], "--repeat=", 9)) repeat = atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--msa=", 6)) msaCap = atoi(argv[i] + 6);
    else if (!strcmp(argv[i], "--fold")) doFold = true;
    else if (!strcmp(argv[i], "--no-graphs")) GRAPHS = false;
    else if (!strcmp(argv[i], "--profile")) profile = true;
    else if (!strcmp(argv[i], "--no-flash-split")) FLASH_SPLIT = false;
    else if (!strncmp(argv[i], "--folds=", 8)) folds = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--steps=", 8)) steps = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--recycles=", 11)) recycles = atoi(argv[i] + 11);
    else if (!strncmp(argv[i], "--seed=", 7)) seed = strtoull(argv[i] + 7, nullptr, 10);
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
  }
  auto t0 = std::chrono::steady_clock::now();
  M.load(argv[1]);
  if (profile) prof::init();
  CB(cublasCreate(&H));
  CB(cublasSetStream(H, STREAM));
  { void* ws; CK(cudaMalloc(&ws, 64 << 20)); CB(cublasSetWorkspace(H, ws, 64 << 20)); }   // graph capture needs it
  printf("loaded %zu entries in %.1f s; %d tokens\n", M.index.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
         (int)M.meta("batch.tokens"));

  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-ops=", 12)) { benchOps(atoi(argv[i] + 12)); return 0; }
  // One denoiser call on AF3's own inputs, against AF3's own output.
  for (const char* which : {"denoise", "realdenoise"}) {
    std::string O = std::string("oracle.") + which + ".";
    if (!M.has(O + "output")) continue;
    int n = (int)M.meta("batch.tokens");
    for (const char* g : {"token_atoms_to_queries", "queries_to_keys", "queries_to_token_atoms", "tokens_to_queries", "tokens_to_keys"}) {
      std::string o = O + "inputs." + g + ":gather_idxs";
      std::string mine = std::string(g) == "token_atoms_to_queries" ? "batch.tokenAtomsToQueries.indices"
        : std::string(g) == "queries_to_keys" ? "batch.queriesToKeys.indices"
        : std::string(g) == "queries_to_token_atoms" ? "batch.queriesToTokenAtoms.indices"
        : std::string(g) == "tokens_to_queries" ? "batch.tokensToQueries.indices" : "batch.tokensToKeys.indices";
      size_t len = M.len(mine), diff = 0;
      if (M.len(o) != len) { printf("  gather %s: length %zu vs %zu\n", g, M.len(o), len); continue; }
      for (size_t i = 0; i < len; ++i) diff += (int)M.f(o)[i] != M.i(mine)[i];
      printf("  gather %-24s %zu of %zu differ from the batch\n", g, diff, len);
    }
    float* single = upload(M.f(O + "inputs.single"), M.len(O + "inputs.single"));
    float* pair = upload(M.f(O + "inputs.pair"), M.len(O + "inputs.pair"));
    float* sIn = upload(M.f(O + "inputs.sInputs"), M.len(O + "inputs.sInputs"));
    float* pos = upload(M.f(O + "inputs.posNoisy"), M.len(O + "inputs.posNoisy"));
    float* seqm = upload(M.f(O + "inputs.seq_mask"), n);
    float noise = (float)M.meta(O + "noise");
    auto s0 = std::chrono::steady_clock::now();
    float* out = denoise(single, pair, sIn, seqm, pos, noise);
    CK(cudaDeviceSynchronize());
    printf("denoise (sigma %.3f) %.1f ms\n", noise, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count());
    check(which, out, M.len(O + "output"), O + "output");
    DCACHE.ready = false;
  }

  // The confidence head on AF3's own inputs, against AF3's own outputs.
  if (M.has("oracle.confidence.stages.out.full_pae")) {
    int n = (int)M.meta("batch.tokens");
    const std::string I = "oracle.confidence.stages.in.";
    float* pair = upload(M.f(I + "pair"), M.len(I + "pair"));
    float* single = upload(M.f(I + "single"), M.len(I + "single"));
    float* tf = upload(M.f(I + "targetFeat"), M.len(I + "targetFeat"));
    float* beta = upload(M.f(I + "pseudoBeta"), M.len(I + "pseudoBeta"));
    std::vector<float> seq(M.f(I + "seqMask"), M.f(I + "seqMask") + n), pm((size_t)n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pm[(size_t)i * n + j] = seq[i] * seq[j];
    float* seqm = upload(seq.data(), n); float* pairm = upload(pm.data(), pm.size());
    ConfidenceOut c = confidenceHead(pair, single, tf, beta, seqm, pairm, n);
    auto cmp = [&](const char* label, const std::vector<float>& mine, const std::string& o) {
      if (!M.has(o) || M.len(o) != mine.size()) { printf("  %-24s (no oracle)\n", label); return; }
      printf("  %-24s relRMS %.3e\n", label, relRms(mine.data(), M.f(o), mine.size()));
    };
    // 🔴 pLDDT ON THIS DUMP'S RANDOM INPUTS IS ILL-CONDITIONED: a 1e-6 relative change to the
    // input single moves it by 7.2e-3 relRMS, so ~1e-2 is all this comparison can resolve.
    // PAE and PDE are well-conditioned and are the check; a fold's mean pLDDT is the other.
    cmp("confidence pLDDT", c.plddt, "oracle.confidence.stages.out.predicted_lddt");
    {
      std::vector<float> masked = c.plddt, theirs(M.f("oracle.confidence.stages.out.predicted_lddt"),
        M.f("oracle.confidence.stages.out.predicted_lddt") + c.plddt.size());
      const float* am = M.f("batch.refMask");
      for (size_t i = 0; i < masked.size(); ++i) if (!am[i]) masked[i] = theirs[i] = 0;
      printf("  %-24s relRMS %.3e\n", "  over real atoms", relRms(masked.data(), theirs.data(), masked.size()));
    }
    cmp("confidence PAE", c.pae, "oracle.confidence.stages.out.full_pae");
    cmp("confidence PDE", c.pde, "oracle.confidence.stages.out.full_pde");
  }

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
  for (int fi = 0; doFold && fi < folds; ++fi) {
    if (fi > 0) {   // a fresh fold: the trunk restarts from zero recycled state
      size_t pp = (size_t)t.n * t.n * t.C;
      CK(cudaMemset(t.prevPair, 0, pp * 4)); CK(cudaMemset(t.prevSingle, 0, (size_t)t.n * t.Cs * 4));
    }
    std::function<void(const char*, const float*, size_t)> none = [](const char*, const float*, size_t) {};
    auto clock = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    size_t pairs = (size_t)t.n * t.n;
    bool profiling = profile && fi + 1 == folds;      // the last (warm) fold
    if (profiling) prof::start();
    auto f0 = clock();
    for (int pass = 0; pass <= recycles; ++pass) {
      if (pass > 0) {
        CK(cudaMemcpyAsync(t.prevPair, t.pair, pairs * t.C * 4, cudaMemcpyDeviceToDevice, STREAM));
        CK(cudaMemcpyAsync(t.prevSingle, t.single, (size_t)t.n * t.Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
      }
      if (fast) runTrunk<half>(t, none); else runTrunk<float>(t, none);
    }
    CK(cudaDeviceSynchronize());
    auto f1 = clock();
    int dense = (int)M.meta("batch.dense");
    std::vector<float> mask(M.f("batch.refMask"), M.f("batch.refMask") + (size_t)t.n * dense);
    DiffusionFold df = prepareDiffusion(t.single, t.pair, t.targetFeat, t.seqMask, t.n);
    std::vector<float> x = sample(steps, seed, mask, [&](const float* noisy, float tHat, const float* dLevel) {
      return (const float*)denoiseStep(df, noisy, tHat, dLevel);
    });
    auto f2 = clock();
    // pseudo-beta off the structure, then the confidence head
    std::vector<float> beta((size_t)t.n * 3);
    const int* pbIdx = M.i("batch.tokenAtomsToPseudoBeta.indices"); const float* pbMask = M.f("batch.tokenAtomsToPseudoBeta.mask");
    for (int k = 0; k < t.n; ++k) for (int a = 0; a < 3; ++a) beta[k * 3 + a] = pbMask[k] ? x[(size_t)pbIdx[k] * 3 + a] : 0.f;
    float* dBeta = upload(beta.data(), beta.size());
    ConfidenceOut conf = confidenceHead(t.pair, t.single, t.targetFeat, dBeta, t.seqMask, t.pairMask, t.n);
    auto f3 = clock();
    std::vector<float> perToken(t.n, 0.f);
    for (int k = 0; k < t.n; ++k) {
      double s2 = 0, c2 = 0;
      for (int a = 0; a < dense; ++a) if (mask[(size_t)k * dense + a]) { s2 += conf.plddt[(size_t)k * dense + a]; c2 += 1; }
      perToken[k] = (float)(s2 / std::max(c2, 1.0));
    }
    writePdb(out, x, perToken.data());
    if (const char* pp = getenv("AF3_PAE_OUT")) {   // the raw PAE/PDE/pLDDT, for comparing two arms
      FILE* pf = fopen(pp, "wb");
      fwrite(conf.pae.data(), 4, conf.pae.size(), pf); fwrite(conf.pde.data(), 4, conf.pde.size(), pf);
      fwrite(conf.plddt.data(), 4, conf.plddt.size(), pf); fclose(pf);
    }
    if (STAGES) {
      double total = 0; for (auto& [k, v] : STAGE_MS) total += v;
      for (auto& [k, v] : STAGE_MS) printf("  %-16s %9.1f ms  %4.1f%%\n", k.c_str(), v, 100 * v / total);
    }
    printf("mean pLDDT %.2f  pTM %.4f  ipTM %.4f  -> %s\n", conf.meanPlddt, conf.ptm, conf.iptm, out.c_str());
    printf("fold %d: trunk %.1f ms (%d passes), diffusion %.1f ms (%d steps), confidence %.1f ms, total %.1f ms\n", fi + 1,
           ms(f0, f1), recycles + 1, ms(f1, f2), steps, ms(f2, f3), ms(f0, f3));
    if (profiling) prof::stop(40);
    if (fi + 1 == folds) return 0;
  }
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
