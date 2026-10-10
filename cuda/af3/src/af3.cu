// AlphaFold 3 in CUDA - the driver.
//
//   af3 <data-dir> [--fast] [--stages] [--repeat=N]
//
// Reads <data-dir>/model.{idx,bin} (export-model.mjs). Without --fast it runs the precise
// path (f32 throughout) and checks every seam the oracle recorded; with --fast the f16 path.
#include "../../featurise/standalone_api.h"
#include <dirent.h>
#include <future>
#include "trunk.cuh"
#include "atom.cuh"
#include "diffusion.cuh"
#include "sampler.cuh"
#include "scores.cuh"
#include "confidence.cuh"
#include "structural.cuh"
#include "benchops.cuh"
#include "../../featurise/af3_weights.h"   // the weight walk: the bundle read as published, no map
#include "profile.cuh"

// Whether the fold's pair stays bf16 PAST the trunk too (af3.cu's TRUNK_PAIR16): on a card short of room, where
// every reader after the trunk takes bf16 rows - the distogram's contacts, the streamed diffusion preparation and the
// confidence head - so the f32 pair is never made. Not for OpenDDE (its expander reads f32). foldFits sizes a
// fold by it.
// LOCALFOLD_WIDEN_PAIR=1: widened as before (the control arm: the structure must not move - tools/check-standalone.py)
inline bool pairStays16(int n, int C, bool fast) {
  static const bool widen = getenv("LOCALFOLD_WIDEN_PAIR") != nullptr;
  if (widen) return false;
  size_t pairs = (size_t)n * n;
  Trunk probe{}; probe.n = n; probe.C = C;
  return fast && DIFF_HALF && CONF_HALF && pair16Eligible(probe) && shortPair(pairs, C) &&
         !M.flag("trunk.dialect.structuralTokens") &&
         hasW("diffusion.encoder.embedTrunkPairCond") &&
         shortPair(pairs, (int)M.meta("diffusion.conditioning.pairChannels")) && (int)M.meta("confidence.pairChannels") == C;
}

// several GPUs on a sharded pair, at the trunk's or the sampler's end: every rank's rows onto rank 0 into a whole bf16
// pair, every shared buffer given back (collectively), the other ranks gone; rank 0 folds on alone
static void pairOntoRank0(Trunk& t) {
  const size_t row = (size_t)t.n * t.C * 2;
  releaseScratch();                  // (what the trunk or the sampler held: the whole pair is rank 0's largest tensor)
  float* whole = mg::RANK == 0 ? reinterpret_cast<float*>(dallocT<__nv_bfloat16>((size_t)t.n * t.n * t.C)) : nullptr;
  mg::fence();
  if (mg::RANK == 0)
    for (int r = 0; r < mg::WORLD; ++r) {
      int lo, hi; sh::rowsOf(t.n, r, lo, hi); const int rn = sh::storedRows(t.n, r);
      if (rn) CK(cudaMemcpyAsync((char*)whole + (size_t)lo * row, t.zS->peer[r], (size_t)rn * row, cudaMemcpyDefault, STREAM));
    }
  mg::fence();
  mg::release({ "" });               // (every shared buffer, collectively: rank 0's pair is whole and its own now)
  if (mg::RANK != 0) mg::leave();
  mg::finish();
  mg::WORLD = 1;
  t.pair = whole; t.shardLo = -1; t.shardRows = 0; t.zS = t.zTS = nullptr; t.msa = nullptr; t.inPlaceRecycle = true;
  sh::DLO = -1; sh::DROWS = 0;
}
static int foldMain(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: af3 <data-dir> [--fast] [--stages] [--repeat=N]\n"); return 1; }
  // the big-input paths when 18x the f32 pair does not fit the room (common.cuh, shortPair);
  // LOCALFOLD_SHORT_PAIR_TIMES=0 is the old 64th-of-the-card rule
  SHORT_PAIR_TIMES = getenv("LOCALFOLD_SHORT_PAIR_TIMES") ? atof(getenv("LOCALFOLD_SHORT_PAIR_TIMES")) : 18;
  // LOCALFOLD_UNFUSED=grid,triangle,transition: those pair-track families through their unfused kernels (to
  // measure the two on a device - the fused ones are each a block's worth of shared memory)
  if (const char* u = getenv("LOCALFOLD_UNFUSED")) {
    std::string un = std::string(",") + u + ",";
    if (un.find(",grid,") != std::string::npos) FUSED_GRID = false;
    if (un.find(",triangle,") != std::string::npos) FUSED_TRIANGLE = false;
    if (un.find(",transition,") != std::string::npos) FUSED_TRANSITION = false;
  }
  bool noGraphs = false;     // (--no-graphs: GRAPHS stays off where several GPUs turn it back on)
  bool fast = false, doFold = false, profile = false; int repeat = 1, msaCap = 1024, steps = 200, recycles = 3, folds = 1, samples = 1; double recycleTolerance = 0;   // 3 recycles: the page's default
  // --af3-defaults: AlphaFold 3's own run_alphafold.py settings - 10 recycles (11 trunk passes) and
  // 5 diffusion samples - where the command does not set them; the plain defaults are the page's
  bool af3Defaults = false, saveEmbeddings = false, saveDistogram = false;
  uint64_t seed = 42; std::string out = "fold.pdb", weightsDir, bundleDir, family, seedsArg, framesDir, esmBundle;
  bool waitInput = false;   // start up (CUDA, the weights on the device) while the input is still being exported
  bool waitForever = false; // ...for as long as it takes: --wait-input=0, which the standalone mode passes - its featuriser
                            // is a thread of this process that always ends in model.idx or model.failed, and an MMseqs2
                            // queue can outlast any timeout
  std::string serveDir;     // --serve=DIR: stay up, the weights resident, folding each job dropped in DIR
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--fast")) fast = DIFF_HALF = ATOM_HALF = CONF_HALF = F32_TF32 = true;
    else if (!strcmp(argv[i], "--no-tf32")) F32_TF32 = false;
    else if (!strcmp(argv[i], "--no-tri-bf16")) TRI_BF16 = false;
    else if (!strcmp(argv[i], "--no-tri-lt")) TRI_LT_TILE = false;
    else if (!strcmp(argv[i], "--stages")) STAGES = true;
    else if (!strncmp(argv[i], "--repeat=", 9)) repeat = atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--msa=", 6)) msaCap = atoi(argv[i] + 6);
    else if (!strcmp(argv[i], "--fold")) doFold = true;
    else if (!strcmp(argv[i], "--no-graphs")) { GRAPHS = false; noGraphs = true; }
    else if (!strcmp(argv[i], "--profile")) profile = true;
    else if (!strcmp(argv[i], "--chai-second-order")) CHAI_SECOND_ORDER = true;     // chai-lab's two calls a step
    else if (!strcmp(argv[i], "--no-flash-split")) FLASH_SPLIT = false;
    else if (!strncmp(argv[i], "--folds=", 8)) folds = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--steps=", 8)) steps = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--recycles=", 11)) recycles = atoi(argv[i] + 11);
    else if (!strncmp(argv[i], "--recycle-tolerance=", 20)) recycleTolerance = atof(argv[i] + 20);   // angstroms; 0 off
    else if (!strncmp(argv[i], "--samples=", 10)) samples = atoi(argv[i] + 10);
    else if (!strcmp(argv[i], "--af3-defaults")) af3Defaults = true;
    else if (!strcmp(argv[i], "--flow")) SAMPLER_FLOW = true;     // the page's Flow sampler (sampler.cuh)
    else if (!strncmp(argv[i], "--sigma-max=", 12)) SAMPLER_SIGMA_MAX = atof(argv[i] + 12);   // where diffusion starts
    else if (!strncmp(argv[i], "--frames=", 9)) framesDir = argv[i] + 9;   // each step's prediction, streamed (FrameStreamer)
    else if (!strcmp(argv[i], "--save-embeddings")) saveEmbeddings = true;
    else if (!strcmp(argv[i], "--save-distogram")) saveDistogram = true;
    else if (!strncmp(argv[i], "--seed=", 7)) seed = strtoull(argv[i] + 7, nullptr, 10);
    else if (!strncmp(argv[i], "--seeds=", 8)) seedsArg = argv[i] + 8;
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
    else if (!strncmp(argv[i], "--weights=", 10)) weightsDir = argv[i] + 10;
    else if (!strncmp(argv[i], "--bundle=", 9)) bundleDir = argv[i] + 9;       // a published bundle, read as it is,
    else if (!strncmp(argv[i], "--family=", 9)) family = argv[i] + 9;          // whose bundle it is: its weight walk and dialect
    else if (!strncmp(argv[i], "--esm-bundle=", 13)) esmBundle = argv[i] + 13; // chai-1's ESM2 3B (af3-any-model's lm/esm2.bin.zst)
    else if (!strncmp(argv[i], "--score-pdb=", 12)) return scorePdbMain(argv[i] + 12);
    else if (!strcmp(argv[i], "--wait-input")) waitInput = true;
    else if (!strcmp(argv[i], "--wait-input=0")) { waitInput = true; waitForever = true; }   // (standalone: its own featuriser)
    else if (!strncmp(argv[i], "--serve=", 8)) serveDir = argv[i] + 8;
    else if (!strcmp(argv[i], "--detach-output") || !strcmp(argv[i], "--oracle-target-feat") ||
             !strncmp(argv[i], "--bench-", 8)) {}                                 // (read by their own loops below)
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }     // (as af2 and ef2 refuse one)
  }
  auto t0 = std::chrono::steady_clock::now();
  // a batch: `af3 dir1,dir2,... --out=a.pdb,b.pdb` folds each input in this one process, the weights
  // loaded once and every kernel, weight copy and cuBLAS plan warm after the first
  auto splitList = [](const std::string& text) {
    std::vector<std::string> parts; std::string part; std::istringstream in(text);
    while (std::getline(in, part, ',')) if (!part.empty()) parts.push_back(part);
    return parts;
  };
  std::vector<std::string> inputs = splitList(argv[1]), outs = splitList(out);
  if (inputs.size() > 1 && outs.size() != inputs.size()) {
    fprintf(stderr, "%zu inputs and %zu --out paths: a batch names one output per input\n", inputs.size(), outs.size()); return 1;
  }
  if (!bundleDir.empty() != !family.empty()) { fprintf(stderr, "--bundle and --family go together\n"); return 1; }
  // 🔴 A BUNDLE THAT NAMES ITS MODEL DECIDES ITS DIALECT, as the page's af3Dialect does: several families' bundles
  // walk cleanly under another's conventions (af3, openbind0 and intellifold2 each walk the other two), and a fold
  // through the wrong one returns a structure, not an error. af3-any-model's blobs carry no manifest: --family is
  // all there is for them, and the binaries and the worker pass the family the blob belongs to.
  if (!bundleDir.empty()) {
    std::ifstream mf(bundleDir + "/manifest.json");
    if (mf) {
      std::string text((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
      Json manifest = Json::parse(text);
      const Json* model = manifest.get("model");
      const Json* name = model ? model->get("name") : nullptr;
      try {
        if (!name || name->t != Json::STR) throw std::runtime_error(bundleDir + "/manifest.json does not name its model, so its dialect cannot be derived");
        std::string bundleIs = lf::weights::dialectNamed(name->str).t->name, asked = lf::weights::dialectNamed(family).t->name;
        if (bundleIs != asked) throw std::runtime_error(bundleDir + " is a " + bundleIs + " bundle, and --family names " + asked);
      } catch (const std::exception& e) { fprintf(stderr, "Error: %s\n", e.what()); return 1; }
    }
  }
  if (!weightsDir.empty()) M.load(weightsDir);      // the weights exported once (--weights-only)
  else if (!bundleDir.empty())                      // ...or read as published, through the family's weight walk
    M.loadBundle(bundleDir, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
      lf::weights::Shapes S; S.shape = shapes;
      return lf::weights::af3WeightLines(family, S);
    });
  // (its matrices stay resident as int8 codes and expand a layer at a time: cuda/plm/esm2.cuh)
  const int esmSeg = (int)M.segs.size();
  if (!esmBundle.empty()) M.loadBundle(esmBundle, "e", nullptr, "", "esm2/blocks/");   // (af3-any-model's lm/esm2.bin.zst)
  const bool haveWeights = !weightsDir.empty() || !bundleDir.empty();
  bool seedGiven = false;
  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--seed=", 7)) seedGiven = true;
  if (af3Defaults) {
    bool setR = false, setS = false;
    for (int i = 2; i < argc; ++i) {
      setR |= !strncmp(argv[i], "--recycles=", 11);
      setS |= !strncmp(argv[i], "--samples=", 10);
    }
    if (!setR) recycles = 10;
    if (!setS) samples = 5;
  }
  const uint64_t seedArg = seed;
  const std::string seedsArg0 = seedsArg;
  std::vector<uint64_t> seedList;
  if (getenv("FLASH_WARPS")) FLASH_WARPS_OVERRIDE = atoi(getenv("FLASH_WARPS"));   // experiments
  if (getenv("FT_WARPS")) FT_WARPS = atoi(getenv("FT_WARPS"));
  if (getenv("TRI_PAD")) TRI_PAD = atoi(getenv("TRI_PAD"));
  if (getenv("MIN_BLOCKS")) MIN_BLOCKS = strtoull(getenv("MIN_BLOCKS"), nullptr, 10);
  if (getenv("TRI_OUT_PERSISTENT")) TRI_OUT_PERSISTENT = atoi(getenv("TRI_OUT_PERSISTENT"));
  if (getenv("LOCALFOLD_GRID_STRIDED")) GRID_STRIDED = atoi(getenv("LOCALFOLD_GRID_STRIDED"));
  if (getenv("LOCALFOLD_LN_NORM_HEADS")) LN_NORM_HEADS = atoi(getenv("LOCALFOLD_LN_NORM_HEADS"));
  if (profile) prof::init();
  CB(cublasCreate(&H));
  CB(cublasSetStream(H, STREAM));
  { void* ws; CK(cudaMalloc(&ws, 64 << 20)); CB(cublasSetWorkspace(H, ws, 64 << 20)); }   // graph capture needs it
  // cuBLAS's first GEMM costs ~70 ms (its library loads lazily) and each new kernel family a few
  // more: a thread pays that with a few small GEMMs of the kinds a fold runs, while this one puts
  // the weights on the device
  std::thread cublasWarm([] {
    cublasHandle_t h; cudaStream_t s;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS || cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return;
    cublasSetStream(h, s);
    void* buf; if (cudaMalloc(&buf, (size_t)3 << 20) != cudaSuccess) return;
    char* b = (char*)buf; const float one = 1.f, zero = 0.f;
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 512, 512, 128, &one, b, CUDA_R_16F, 512, b + (1 << 20), CUDA_R_16F, 128, &zero,
                 b + (2 << 20), CUDA_R_16F, 512, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 256, 576, 128, &one, b, CUDA_R_16F, 256, b + (1 << 20), CUDA_R_16F, 128, &zero,
                 b + (2 << 20), CUDA_R_32F, 256, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 512, 256, 128, &one, b, CUDA_R_32F, 512, b + (1 << 20), CUDA_R_32F, 128, &zero,
                 b + (2 << 20), CUDA_R_32F, 512, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (ccMajor() >= 8)      // (bf16: the fused triangle's contraction, Ampere on - a T4 runs the unfused path)
      cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 96, 96, 96, &one, b, CUDA_R_16BF, 96, 9216, b + (1 << 20),
                                 CUDA_R_16BF, 96, 9216, &zero, b + (2 << 20), CUDA_R_16BF, 96, 9216, 8, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaStreamSynchronize(s);
    cudaFree(buf); cublasDestroy(h); cudaStreamDestroy(s);
  });
  struct JoinAtExit { std::thread& t; ~JoinAtExit() { if (t.joinable()) t.join(); } } joinWarm{cublasWarm};
  Trunk t{};
  if (haveWeights) M.upload(0);      // now, beside the cuBLAS warm-up (both are needed before any fold)
  if (!esmBundle.empty()) M.upload(esmSeg);
  if (cublasWarm.joinable()) cublasWarm.join();
  auto runInput = [&](size_t which) -> int {
  if (waitInput) {          // the exporter writes model.idx last, by a rename
    std::string idx = inputs[which] + "/model.idx";
    std::string failed = inputs[which] + "/model.failed";     // the wrapper's word that the export died
    for (int k = 0; access(idx.c_str(), R_OK) != 0; ++k) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "af3: the input's export failed\n"); return 1; }
      if (k > 600000 && !waitForever) { fprintf(stderr, "no %s after ten minutes\n", idx.c_str()); return 1; }
      usleep(1000);
    }
  }
  M.load(inputs[which]); DATA_DIR = inputs[which];
  if (inputs.size() > 1) out = outs[which];
  { const float* sm = M.f("batch.seqMask"); size_t k = M.len("batch.seqMask"); MASK_ALL_ONES = true;
    for (size_t i = 0; i < k; ++i) if (!(sm[i] > 0)) MASK_ALL_ONES = false; }
  seed = !seedGiven && M.has("job.seed") ? (uint64_t)M.meta("job.seed") : seedArg;   // the job's own modelSeeds[0]
  // --seeds=a,b,c, else the job's modelSeeds, else the one seed: AlphaFold 3 runs every seed, each
  // with its samples, and ranks them all (one trunk serves every seed - the features do not depend
  // on it - so each seed costs a diffusion and a confidence)
  seedList.clear();
  if (!seedsArg.empty()) {
    for (size_t p = 0; p < seedsArg.size();) {
      size_t q = seedsArg.find(',', p); if (q == std::string::npos) q = seedsArg.size();
      seedList.push_back(strtoull(seedsArg.substr(p, q - p).c_str(), nullptr, 10)); p = q + 1;
    }
  } else if (!seedGiven && M.has("job.seeds.count")) {
    for (int k = 0; k < (int)M.meta("job.seeds.count"); ++k) seedList.push_back((uint64_t)M.meta("job.seeds." + std::to_string(k)));
  } else {
    seedList.push_back(seed);
  }
  setAdaMode(M.flag("trunk.dialect.chaiAtomStack"));     // (chai-1's adaptive LayerNorm form, for every atom and token stack)
  printf("loaded %zu entries in %.1f s; %d tokens\n", M.index.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
         (int)M.meta("batch.tokens"));

  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-ops=", 12)) { const char* a = argv[i] + 12; const char* x = strchr(a, 'x'); benchOps(atoi(a), x ? atoi(x + 1) : 1); return 0; }
  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-grid=", 13)) { benchGrid(atoi(argv[i] + 13)); return 0; }
  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-grid16=", 15)) { benchGrid16(atoi(argv[i] + 15)); return 0; }
  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-tri=", 12)) { benchTri(atoi(argv[i] + 12)); return 0; }
  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-trans=", 14)) { benchTrans(atoi(argv[i] + 14)); return 0; }
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
  if (M.has("oracle.confidence.stages.out.full_pae") && M.flag("trunk.dialect.structuralTokens")
      && !getenv("SKIP_CONFIDENCE_ORACLE")) {
    // OpenDDE's own head, on the dump's own (synthesised) inputs - token-agnostic, no extra bias
    const std::string I = "oracle.confidence.stages.in.";
    int n = (int)M.meta("oracle.confidence.tokens"), dense = (int)M.meta("oracle.confidence.slots");
    float* pair = upload(M.f(I + "pair"), M.len(I + "pair"));
    float* single = upload(M.f(I + "single"), M.len(I + "single"));
    float* tf = upload(M.f(I + "targetFeat"), M.len(I + "targetFeat"));
    std::vector<float> hc(M.f(I + "coordinates"), M.f(I + "coordinates") + M.len(I + "coordinates"));
    float* coords = upload(hc.data(), hc.size());
    std::vector<float> seq(M.f(I + "seqMask"), M.f(I + "seqMask") + n), pm((size_t)n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pm[(size_t)i * n + j] = seq[i] * seq[j];
    float* seqm = upload(seq.data(), n); float* pairm = upload(pm.data(), pm.size());
    DdeConfidence c = ddeConfidence(pair, single, tf, coords, seqm, pairm, nullptr, n, dense, n);
    auto cmp = [&](const char* label, const std::vector<float>& mine, const std::string& o) {
      if (!M.has(o) || M.len(o) != mine.size()) { printf("  %-24s (no oracle)\n", label); return; }
      printf("  %-24s relRMS %.3e\n", label, relRms(mine.data(), M.f(o), mine.size()));
    };
    cmp("confidence pLDDT", c.plddt, "oracle.confidence.stages.out.predicted_lddt");
    cmp("confidence PAE", c.pae, "oracle.confidence.stages.out.full_pae");
    cmp("confidence PDE", c.pde, "oracle.confidence.stages.out.full_pde");
  } else if (M.has("oracle.confidence.stages.out.full_pae") && !getenv("SKIP_CONFIDENCE_ORACLE")) {
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

  // OpenDDE's expander and refiner on the structural oracle's own residue-level inputs
  if (M.has("oracle.structural.stages.in.single") && M.flag("trunk.dialect.structuralTokens")) {
    const std::string I = "oracle.structural.stages.in.";
    int nRes = (int)M.meta(I + "single.shape0");
    float* single = upload(M.f(I + "single"), M.len(I + "single"));
    float* pair = upload(M.f(I + "pair"), M.len(I + "pair"));
    float* tf = upload(M.f(I + "targetFeat"), M.len(I + "targetFeat"));
    printf("structural stage on the oracle's inputs:\n");
    Structural s = expandStructural(single, pair, tf, nRes, false);
    for (float* p : {s.single, s.pair, s.targetFeat, s.bias, s.seqMask, s.pairMask, single, pair, tf}) CK(cudaFree(p));
  }

  // target_feat from the batch: per-atom conditioning and the atom cross-attention encoder.
  int tokens = (int)M.meta("batch.tokens");
  bool tf32 = F32_TF32; F32_TF32 = false;          // target_feat once, in full f32: it feeds everything
  float* tfDev = buildTargetFeat();
  int tfWidth = (int)M.meta("trunk.embedder.targetFeatWidth");
  F32_TF32 = tf32;
  check("target_feat", tfDev, (size_t)tokens * tfWidth, "oracle.trunk.stages.target_feat");
  std::vector<float> targetFeat = download(tfDev, (size_t)tokens * tfWidth);
  bool oracleTargetFeat = false;
  for (int i = 2; i < argc; ++i) if (!strcmp(argv[i], "--oracle-target-feat")) oracleTargetFeat = true;
  if (oracleTargetFeat)
    targetFeat.assign(M.f("oracle.trunk.stages.target_feat"), M.f("oracle.trunk.stages.target_feat") + (size_t)tokens * tfWidth);
  if (!(mg::WORLD > 1 && sh::ON) && !foldFits(tokens, (int)M.meta("trunk.embedder.pairChannels"),
                doFold && pairStays16(tokens, (int)M.meta("trunk.embedder.pairChannels"), fast))) return 1;
  // several GPUs on a sharded pair: the diffusion runs whole on every rank (oneCardPast) where the fold would fit one
  // of them (asked before anything is allocated, so every rank answers alike) AND it is short or batches several
  // samples; else sharded - its tokens split over the ranks with one sample (diffusionTransformer), which on fences
  // that stay on the device costs a block one exchange of k and v: a simulated rank of 8 runs 2,964 tokens' 25 steps
  // in 207 ms against 525 whole, and 988 tokens' 200 steps in 555 against 1,116 (on a box the fences add a few us
  // each, 24 a step - hence not below 512). (LOCALFOLD_MG_SHARD_DIFFUSION=1 or 0 forces either.)
  const char* shardDiff = getenv("LOCALFOLD_MG_SHARD_DIFFUSION");
  const bool oneSample = seedList.size() * samples == 1;
  const bool oneCardPast = mg::WORLD > 1 && sh::ON && (shardDiff ? !atoi(shardDiff) : tokens < 512 || !oneSample) &&
    foldFits(tokens, (int)M.meta("trunk.embedder.pairChannels"), doFold && pairStays16(tokens, (int)M.meta("trunk.embedder.pairChannels"), fast), true);
  // (a confidence head with no sharded form - boltz2's, rf3's - runs on rank 0 with the pair gathered: refused up front
  // where that pair would not fit one GPU)
  const bool confOnRank0 = mg::WORLD > 1 && sh::ON && doFold && !confidenceShardable();
  if (confOnRank0 && !foldFits(tokens, (int)M.meta("trunk.embedder.pairChannels"),
                               pairStays16(tokens, (int)M.meta("trunk.embedder.pairChannels"), fast), true)) {
    fprintf(stderr, "this model's confidence head has no sharded form yet, and a %d-token pair does not fit one GPU\n", tokens);
    return 1;
  }
  t = makeTrunk(targetFeat.data(), msaCap, doFold && fast);
  if (mg::WORLD > 1) {
    // several GPUs (LOCALFOLD_GPUS, multigpu.cuh): the trunk pair in a buffer every rank can read, the pair updates'
    // work split between them; the rest of the fold on rank 0
    if (M.flag("trunk.dialect.parallelPairformer") || M.flag("trunk.dialect.structuralTokens") || !serveDir.empty() ||
        folds != 1 || repeat != 1 || !doFold) {
      fprintf(stderr, "several GPUs fold one job at a time, and not chai-1 or OpenDDE yet\n"); return 1;
    }
    GRAPHS = false;                                             // (a capture cannot hold a host barrier)
    if (sh::ON) {
      // phase 2: the pair sharded by rows (makeTrunk gave each rank its slab); the trunk on slabs, its pair gathered
      // onto rank 0 after the last pass
      if (!framesDir.empty() || recycleTolerance > 0) { fprintf(stderr, "a sharded pair: no --frames or --recycle-tolerance yet\n"); return 1; }
      printf("trunk: the pair sharded over %d GPUs, %d rows here\n", mg::WORLD, t.shardRows);
    } else {
    const size_t pc = (size_t)t.n * t.n * t.C;
    mg::Shared& sp = mg::shared("trunk.pair", pc * 4);          // (f32's room: the pair may be held either way)
    CK(cudaFree(t.pair)); t.pair = (float*)sp.local;
    CK(cudaMemset(t.pair, 0, pc * 4));
    mg::SPLIT = &sp;
    printf("trunk: the pair updates over %d GPUs\n", mg::WORLD);
    }
  }
  memReport("trunk built");
  printf("trunk: %d tokens, %d MSA rows, pair %d, single %d, msa %d; %s path\n", t.n, t.S, t.C, t.Cs, t.Cm,
         fast ? "f16" : "f32");
  // the fold's pair in bf16 where every stack takes it (pair16Eligible; LOCALFOLD_PAIR_F32=1 keeps f32)
  const bool want16 = doFold && fast && pair16Eligible(t);
  if (doFold && want16) printf("trunk: the pair in bf16\n");
  for (int fi = 0; doFold && fi < folds; ++fi) {
    if (fi > 0 || want16) {   // a fresh fold: the trunk restarts from zero recycled state
      size_t pp = (size_t)t.n * t.n * t.C;
      if (!t.pair && !want16) t.pair = dalloc(pp);     // (left parked by the last fold)
      usePair16(t, want16);   // (in place, the pair is the recycled one; zeroed)
      CK(cudaMemset(t.prevSingle, 0, (size_t)t.n * t.Cs * 4));
      t.pass = 0;
    }
    std::function<void(const char*, const float*, size_t)> none = [](const char*, const float*, size_t) {};
    auto clock = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    size_t pairs = (size_t)t.n * t.n;
    bool profiling = profile && fi + 1 == folds;      // the last (warm) fold
    if (profiling) prof::start();
    ++SCORE_ATOMS_GEN;                                // (this fold's input: its atoms are read once, scores.cuh)
    auto f0 = clock();
    // A recycle pass is ~1000 launches with identical shapes and pointers, so from the second pass on
    // it replays as one CUDA graph, captured from that pass (the first has sized every scratch buffer)
    auto recyclePass = [&]() {
      if (!t.inPlaceRecycle)        // (in place, the pair already is the recycled pair: see embed)
        CK(cudaMemcpyAsync(t.prevPair, t.pair, pairs * t.C * (t.p16 ? 2 : 4), cudaMemcpyDeviceToDevice, STREAM));
      CK(cudaMemcpyAsync(t.prevSingle, t.single, (size_t)t.n * t.Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
      if (fast) runTrunk<half>(t, none); else runTrunk<float>(t, none);
    };
    cudaGraphExec_t trunkGraph = nullptr;
    // --frames: each pass's contact map too (the page shows the trunk's after every recycle, before the
    // sampler has a structure), computed on the device, quantised to a byte a pair and tapped (AsyncTap):
    // contacts-PP-of-NN.u8, n*n bytes, probability * 255
    const bool tapContacts = !framesDir.empty() && M.has("batch.contactClasses");
    // chai-1 counts its recycles as TOTAL passes (chai-lab's num_trunk_recycles, af3-any-model's num_trunk_passes),
    // where AlphaFold 3's are passes after the first: the page's 3 is chai-lab's own default of 3 passes
    const int lastPass = M.flag("trunk.dialect.recycleFromInit") ? std::max(1, recycles) - 1 : recycles;
    if (tapContacts) TAP().reserve(lastPass + 1, (size_t)t.n * t.n);
    auto afterPass = [&](int pass) {
      if (!tapContacts) return;
      int bins = (int)M.meta("trunk.distogram.bins");
      size_t pairs = (size_t)t.n * t.n;
      float* logits = scratch<float>("disto.logits", pairs * bins);
      distogram(t, logits);
      float* probs = scratch<float>("disto.contact", pairs);
      contactProbsK<<<blocks(pairs), 256, 0, STREAM>>>(logits, contactBinsDevice(t.n, bins), t.pairMask, probs, pairs, bins);
      unsigned char* bytes = scratch<unsigned char>("disto.contact8", pairs);
      quantiseK<<<blocks(pairs), 256, 0, STREAM>>>(probs, bytes, pairs, 1.f / 255);
      std::string path = framesDir + "/contacts-" + (pass < 10 ? "0" : "") + std::to_string(pass) + "-of-"
                         + (lastPass + 1 < 10 ? "0" : "") + std::to_string(lastPass + 1) + ".u8";
      TAP().offer({{bytes, pairs}}, [path, pairs](const char* host, const std::vector<size_t>&) { writeWhole(path, host, pairs); });
    };
    // --recycle-tolerance: stop once two consecutive passes moved the distogram's predicted distances less than it
    // (the page's rule, shared/af3/feature-convergence.js shouldStopRecycling - one crossing is not enough, GB1's trunk
    // dips under 0.5 A and then moves 1.09 A); off (0) by default, as on the page
    std::vector<double> changes;
    int passesRun = lastPass + 1;
    auto converged = [&](int pass) {
      if (recycleTolerance <= 0 || pass == lastPass) return false;
      double c = distogramChange(t, pass);
      changes.push_back(c);
      size_t k = changes.size();
      if (k < 3 || changes[k - 1] >= recycleTolerance || changes[k - 2] >= recycleTolerance) return false;
      printf("trunk: converged at %.2f A after %d passes\n", changes[k - 1], pass + 1);
      passesRun = pass + 1;
      return true;
    };
    for (int pass = 0; pass <= lastPass; ++pass) {
      if (pass == 0) {
        if (fast) runTrunk<half>(t, none); else runTrunk<float>(t, none);
        afterPass(pass); if (converged(pass)) break; continue;
      }
      // (capturing and instantiating costs ~15 ms and a replayed pass saves ~2 ms at 68 tokens, more
      // as the launches grow: a first fold breaks even at 7 recycles there - AF3's 10 gain 6 ms - and
      // at 3 from ~200 tokens, so the graph is taken where it measured a gain)
      // ...and not where a pass gives its stages' scratch back (runTrunk, shortPair), which a capture
      // cannot do
      if (!GRAPHS || STAGES || !(lastPass >= 7 || t.n >= 200) || shortPair((size_t)t.n * t.n, t.C)) {
        recyclePass(); afterPass(pass); if (converged(pass)) break; continue;
      }
      if (!trunkGraph) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(STREAM, cudaStreamCaptureModeThreadLocal));
        recyclePass();
        CK(cudaStreamEndCapture(STREAM, &g));
        CK(cudaGraphInstantiate(&trunkGraph, g, 0));
        CK(cudaGraphDestroy(g));
      }
      CK(cudaGraphLaunch(trunkGraph, STREAM));
      afterPass(pass); if (converged(pass)) break;
    }
    if (trunkGraph) CK(cudaGraphExecDestroy(trunkGraph));
    CK(cudaDeviceSynchronize());
    if (mg::WORLD > 1 && !sharded(t)) {   // the trunk done on every rank: the others leave, rank 0 folds on alone
      mg::SPLIT = nullptr;
      mg::barrier();
      if (mg::RANK != 0) mg::leave();
      mg::finish();
      mg::WORLD = 1;
    }
    memReport("trunk: passes done");
    // 🔴 THE PAIR STAYS bf16 PAST THE TRUNK on a card short of room, where every reader after it takes bf16 rows
    // (the distogram's contacts, the streamed diffusion preparation, the confidence head): the f32 pair it was widened
    // into was the fold's largest tensor past the trunk - 18.4 GB at 6,000 tokens, 51 at 10,000 - and the widening
    // held both at once. Not for OpenDDE (its expander reads f32), nor --save-embeddings
    if (sharded(t)) {     // what only the trunk read, given back before the diffusion: z^T, the template stack's slabs, the
                          // exchange buffers, the MSA (the slab itself the diffusion reads, and rank 0 gathers after it)
      mg::release({ "sh.zT", "sh.tz", "sh.bmine", "sh.bg", "sh.bias", "sh.st.o", "trunk.msa",
                    "sh.ta", "sh.tb", "sh.tp" });     // (the channel-split triangle's: 9.9 GB a rank at 10,127 tokens on 8)
      t.zTS = nullptr; t.msa = nullptr;
      mg::tick("trunk");
    }
    // ...and where the whole fold fits one GPU, every rank gathers the whole pair and runs the diffusion alone - the same
    // fold on each, nothing exchanged, its steps' graphs on - then frees it and takes its rows of the confidence head
    // (the sharded diffusion meets at every block of every step: at 988 tokens on 8 x A100 158 ms became 2.0 s)
    int keptLo = -1, keptRows = 0;
    if (sharded(t) && oneCardPast) {
      const size_t row = (size_t)t.n * t.C * 2;
      float* whole = reinterpret_cast<float*>(dallocT<__nv_bfloat16>((size_t)t.n * t.n * t.C));
      mg::fence();
      for (int r = 0; r < mg::WORLD; ++r) {
        int lo, hi; sh::rowsOf(t.n, r, lo, hi); const int rn = sh::storedRows(t.n, r);
        if (rn) CK(cudaMemcpyAsync((char*)whole + (size_t)lo * row, t.zS->peer[r], (size_t)rn * row, cudaMemcpyDefault, STREAM));
      }
      mg::fence();
      keptLo = t.shardLo; keptRows = t.shardRows;
      t.pair = whole; t.shardLo = -1; t.shardRows = 0;        // (the whole pair's view; the slab stays in t.zS)
      GRAPHS = !noGraphs;
      mg::tick("whole pair gathered");
    }
    TRUNK_PAIR16 = t.p16 && !saveEmbeddings && (pairStays16(t.n, t.C, fast) || sharded(t));
    if (sharded(t) && (saveEmbeddings || saveDistogram)) { fprintf(stderr, "a sharded pair: no --save-embeddings or --save-distogram yet\n"); return 1; }
    if (!TRUNK_PAIR16) pairToF32(t);  // (a bf16 trunk's pair, for the heads, the sampler and the confidence head)
    else if (t.prevPair) { CK(cudaFree(t.prevPair)); t.prevPair = nullptr; }   // (pairToF32's other half)
    releaseConcatCopies(); memReport("trunk");
    auto f1 = clock();
    if (STAGES) {     // the trunk's stages, then the diffusion's below
      double total = 0; for (auto& [k, v] : STAGE_MS) total += v;
      printf("trunk stages:\n");
      for (auto& [k, v] : STAGE_MS) printf("  %-16s %9.1f ms  %4.1f%%\n", k.c_str(), v, 100 * v / total);
      STAGE_MS.clear();
    }
    // the distogram's contact probabilities, for the confidences file (off the residue batch)
    // (a sharded pair: after the diffusion, once rank 0 holds the whole pair)
    std::vector<float> contact = sharded(t) ? std::vector<float>() : contactProbabilities(t);
    // --save-embeddings / --save-distogram: what AF3's --save_embeddings and --save_distogram write -
    // the trunk's final single (tokens x 384) and pair (tokens x tokens x 128) representations, and
    // the distogram head's probabilities (tokens x tokens x bins) - as .npy files beside the structure
    if ((saveEmbeddings || saveDistogram) && out != "/dev/null") {
      std::string base = out.size() > 4 && (out.substr(out.size() - 4) == ".pdb" || out.substr(out.size() - 4) == ".cif")
        ? out.substr(0, out.size() - 4) : out;
      size_t n = t.n;
      if (saveEmbeddings) {
        writeNpy(base + "_single_embeddings.npy", download(t.single, n * t.Cs), { n, (size_t)t.Cs });
        writeNpy(base + "_pair_embeddings.npy", download(t.pair, n * n * t.C), { n, n, (size_t)t.C });
      }
      if (saveDistogram) {
        int bins = (int)M.meta("trunk.distogram.bins");
        float* logits = scratch<float>("disto.logits", n * n * bins);
        distogram(t, logits);
        std::vector<float> p = download(logits, n * n * bins);
        for (size_t q = 0; q < n * n; ++q) {          // softmax over the bins
          float* r = p.data() + q * bins; float mx = r[0];
          for (int b = 1; b < bins; ++b) mx = std::max(mx, r[b]);
          double sum = 0; for (int b = 0; b < bins; ++b) { r[b] = std::exp(r[b] - mx); sum += r[b]; }
          for (int b = 0; b < bins; ++b) r[b] = (float)(r[b] / sum);
        }
        writeNpy(base + "_distogram.npy", p, { n, n, (size_t)bins });
      }
    }
    // OpenDDE: the expander and refiner, then everything after runs on the structural tokens
    bool structural = M.flag("trunk.dialect.structuralTokens");
    Structural st;
    // (chai-1's diffusion reads its own projection of the token features, not the trunk's: SEPARATE_STRUCTURE_TARGET_FEAT)
    const float *dS = t.single, *dP = t.pair, *dSeq = t.seqMask;
    const float* dTf = M.flag("trunk.dialect.chaiTokenEmbedding") ? TARGET_FEAT_STRUCTURE : t.targetFeat;
    int nD = t.n;
    std::vector<int> resAsym(M.i("batch.asymId"), M.i("batch.asymId") + t.n);
    if (structural) {
      // (a large input's trunk scratch given back BEFORE the expansion, not after: its structural pair - ~2 tokens a
      // residue - takes two f32 [pairs, C] buffers of its own, 6.8 GB each at 1080 residues, which beside the trunk's
      // ordinary-path scratch ran out of a 40 GB card)
      if (tightPair(pairs, t.C)) releaseScratch();
      st = expandStructural(t.single, t.pair, t.targetFeat, t.n, fast);
      swapBatch();
      dS = st.single; dP = st.pair; dTf = st.targetFeat; dSeq = st.seqMask; nD = st.n;
    }
    int dense = (int)M.meta("batch.dense");
    std::vector<float> mask(M.f("batch.refMask"), M.f("batch.refMask") + (size_t)nD * dense);
    // a large input gives each phase the whole card: a pair over 128 MB, 512 tokens at 128 channels
    // (at 1 GB, 1044 tokens peaked at 21.3 GB with the trunk's 8 GB of scratch held to the end)
    bool tight = tightPair(pairs, t.C);
    if (tight) releaseScratch();
    // on a card short of room the trunk's pair waits in host memory from here to the confidence head: the
    // streamed preparation reads it a chunk of rows at a time and the sampler not at all
    // (only where the preparation streams: the f16 path, and an encoder that takes the pair's projection)
    if (!structural && !sharded(t) && keptLo < 0 && DIFF_HALF && hasW("diffusion.encoder.embedTrunkPairCond") && shortPair(pairs, t.C) &&
        parkWorthIt(pairs * t.C * (TRUNK_PAIR16 ? 2 : 4) + diffusionPrepBytes(pairs))) {
      // (the room asked for is the pair's AND what the preparation will hold beside it - asked for the pair alone, a
      // 6,916-token fold kept its 12.2 GB bf16 pair on the device and ran out in the preparation)
      parkToHost(t.pair, pairs * t.C * (TRUNK_PAIR16 ? 2 : 4)); dP = nullptr;
    }
    if (sharded(t)) {                    // the diffusion's conditioning and token attention on this rank's rows
      if (structural) { fprintf(stderr, "a sharded pair: not OpenDDE\n"); return 1; }
      sh::DLO = t.shardLo; sh::DROWS = t.shardRows; dP = pairBase(t);
      GRAPHS = !noGraphs && !mg::SHARED_DEVICE;   // (the steps' graphs: their fences count generations on the device)
      if (DIFF_HALF && oneSample && !M.flag("diffusion.transformer.noResidual")) diffusionSharedBuffers(nD);
    }
    mg::tick("to the diffusion");
    DiffusionFold df = prepareDiffusion(dS, dP, dTf, dSeq, nD);
    mg::tick("diffusion prepared");
    // (several GPUs, the diffusion run whole on each: the gathered pair is read by nothing past its preparation - the
    // confidence head reads this rank's slab - so it goes now, not parked to the host where the card is short: on 8
    // ranks that was 8 x 9 GB pinned at once at 5928 tokens)
    if (keptLo >= 0) { CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(t.pair)); t.pair = nullptr; df.trunkPair = nullptr; }
    // ...and the pair-sized tensors only the preparation reads, given back before the steps: the
    // transformer's and the encoder's pair LayerNorms and the per-super-block logits (1.4 GB at 1048
    // tokens, held through every step). Not the conditioning's chunk buffers (dc.f2*, pt.*): they are
    // CHUNK-sized whatever the length, and giving them back cost 16 ms of a 100 ms diffusion at 525
    // ...and the encoder's per-token-pair projection and its pair MLP's temporaries, which only its preparation
    // reads (folded into the atom pairs' conditioning): held through every step they were 8.8 GB of the
    // sampler's peak at 10761 tokens (branch tier2-host-pair)
    if (tight) releaseScratch({ "dt.pn", "dt.flat", "enc.tpln", "enc.tp", "enc.h1", "enc.h2" });
    // ...and on a card short of room the conditioning pair itself: the steps read the precomputed
    // single conditioning, the prepared encoder and the cached logits, never the pair (DCACHE stays
    // ready - its single base is what a later batch's precompute reads)
    if (shortPair((size_t)nD * nD, (int)M.meta("diffusion.conditioning.pairChannels"))) {
      releaseScratch({ "dc.pair" }); DCACHE.pair = nullptr;
      // ...the preparation's chunk buffers (a fixed cost that matters only here), and the trunk's pair: the
      // sampler never reads it, so on a card short of room it waits in host memory for the confidence head
      releaseScratch({ "dc.f2", "dc.f2n", "dc.pairChunk", "dc.rel", "dc.relProj", "dc.tln", "dc.tproj", "pt." });
      if (!structural && !sharded(t) && keptLo < 0 && t.pair && parkWorthIt(pairs * t.C * (TRUNK_PAIR16 ? 2 : 4))) {
        parkToHost(t.pair, pairs * t.C * (TRUNK_PAIR16 ? 2 : 4)); df.trunkPair = nullptr;
      }
    }
    releaseConcatCopies(); memReport("diffusion prepared");
    // --samples=N: N diffusion samples off one trunk (AF3 runs five) for every seed, each through the
    // confidence head and ranked by AF3's ranking score (src/scores.cuh). A seed's samples run as one
    // batch through the denoiser, sample k of seed s seeded sampleSeed(s, k); the best of them all is
    // written to --out, every one to <out>_sample<k>.pdb (<out>_seed<s>_sample<k>.pdb with several
    // seeds), and the scores to <out>_ranking_scores.csv.
    double diffMs = 0, confMs = 0, bestScore = -1e30; int best = 0; uint64_t bestSeed = seedList[0];
    StructureScores bestSS{ false, 0.0 };
    ConfidenceOut conf;
    std::vector<float> x;
    std::string ext = cifPath(out) ? ".cif" : ".pdb";     // --out=*.cif: mmCIF, as AlphaFold 3 writes
    std::string stem = out.size() > 4 && out.substr(out.size() - 4) == ext ? out.substr(0, out.size() - 4) : out;
    bool many = seedList.size() > 1 || samples > 1;
    std::vector<std::string> ranking;
    // every (seed, sample) through the denoiser together, up to ten a batch: a batch's GEMMs cost far
    // less than its rows (five samples' diffusion is 1.6x one's at 68 tokens)
    std::vector<std::pair<uint64_t, int>> runs;
    for (uint64_t s : seedList) for (int k = 0; k < samples; ++k) runs.push_back({ s, k });
    const size_t perBatch = std::max<size_t>(samples, 10);
    if (sharded(t) && runs.size() > 10) { fprintf(stderr, "a sharded pair: ten seed x sample runs at most yet\n"); return 1; }
    FrameStreamer frames;
    for (size_t c0 = 0; c0 < runs.size(); c0 += perBatch) {
    const size_t cn = std::min(perBatch, runs.size() - c0);
    if (structural && c0 > 0) swapBatch();     // the structural tokens again, for this batch's diffusion
    auto s0 = clock();
    NS = (int)cn;
    std::vector<uint64_t> seeds;
    for (size_t k = 0; k < cn; ++k) seeds.push_back(sampleSeed(runs[c0 + k].first, runs[c0 + k].second));
    if (SAMPLER_FLOW && M.flag("trunk.dialect.noFlowSampler")) {
      // the page's own rule (noFlowSampler, shared/af3/dialect.js): this checkpoint's walk collapses the
      // backbone while its pLDDT reads as if nothing were wrong
      fprintf(stderr, "this checkpoint has no working flow sampler - fold it with diffusion\n"); return 1;
    }
    if (!framesDir.empty() && c0 == 0) {          // (the first batch's first sample)
      TAP().reserve(FrameStreamer::planned(steps), mask.size() * 3 * 4);
      frames.start(framesDir, mask.size() * 3, steps);
    }
    // several GPUs with the diffusion whole on every rank and several samples a batch: the samples DEALT, sample k on
    // rank k % N (each its own seed, so the same walk as in the batch), the coordinates gathered after - every rank
    // ran every sample before, the whole batch's diffusion on each
    const bool dealt = keptLo >= 0 && mg::WORLD > 1 && cn > 1;
    std::vector<uint64_t> mySeeds;
    for (size_t k = 0; k < cn; ++k) if (!dealt || (int)(k % mg::WORLD) == mg::RANK) mySeeds.push_back(seeds[k]);
    NS = (int)mySeeds.size();
    std::vector<float> xs = NS ? sample(steps, mySeeds, mask, [&](const float* noisy, float tHat, const float* dLevel) {
      return (const float*)denoiseStep(df, noisy, tHat, dLevel);
    }, 0.8, 1.0, 1.003, 1.5, [&](const std::vector<float>& levels) { precomputeConditioning(df, levels); }) : std::vector<float>();
    if (dealt) {
      const size_t per = mask.size() * 3;
      mg::Shared& xS = mg::shared("sh.xs", perBatch * per * 4);
      for (size_t k = 0, j = 0; k < cn; ++k)
        if ((int)(k % mg::WORLD) == mg::RANK)
          CK(cudaMemcpyAsync((float*)xS.local + k * per, xs.data() + (j++) * per, per * 4, cudaMemcpyHostToDevice, STREAM));
      mg::fence(); mg::hostFence();
      xs.assign(cn * per, 0.f);
      for (size_t k = 0; k < cn; ++k)
        CK(cudaMemcpy(xs.data() + k * per, (const float*)xS.peer[k % mg::WORLD] + k * per, per * 4, cudaMemcpyDefault));
      mg::hostFence();           // (no rank writes its slots again before every rank has read them)
      NS = (int)cn;
    }
    mg::tick("sampler");
    FRAME_HOOK = nullptr;      // (the writer finishes the last frames while the confidence head runs)
    // several GPUs on a sharded pair, past the sampler: the confidence head of every sample and the contacts on held
    // rows (collective, rank 0's results); then the others leave and rank 0 writes the fold with the whole pair never
    // held anywhere
    std::vector<ConfidenceOut> shardedConf;
    if (keptLo >= 0) {                 // (the diffusion ran whole on every rank: its pair given back, the slab's view again)
      t.pair = (float*)t.zS->local; t.shardLo = keptLo; t.shardRows = keptRows; t.p16 = true;
      GRAPHS = false;
      mg::tick("whole pair freed");
    }
    // (a confidence head with no sharded form: the pair onto rank 0, which runs the one-GPU head - where it fits)
    if (sharded(t) && confOnRank0) {
      GRAPHS = false;
      df.trunkPair = nullptr;
      pairOntoRank0(t);
      // (then the one-GPU rule for the pair's width: boltz2's and rf3's heads read it f32)
      TRUNK_PAIR16 = t.p16 && pairStays16(t.n, t.C, fast);
      if (!TRUNK_PAIR16) pairToF32(t);
      if (contact.empty()) contact = contactProbabilities(t);
    }
    if (sharded(t)) {
      GRAPHS = false;
      releaseScratch();                // (the sampler's: the heads allocate theirs beside the slab)
      mg::tick("scratch released");
      df.trunkPair = nullptr;
      std::vector<int> pbI(M.i("batch.tokenAtomsToPseudoBeta.indices"), M.i("batch.tokenAtomsToPseudoBeta.indices") + nD);
      std::vector<float> pbM(M.f("batch.tokenAtomsToPseudoBeta.mask"), M.f("batch.tokenAtomsToPseudoBeta.mask") + nD);
      const bool ca = M.flag("trunk.dialect.confidenceCaDgram");
      auto sc0 = clock();
      for (size_t k = 0; k < cn; ++k) {
        const float* xk = xs.data() + k * mask.size() * 3;
        std::vector<float> beta((size_t)nD * 3);
        for (int r = 0; r < nD; ++r) for (int c3 = 0; c3 < 3; ++c3)
          beta[r * 3 + c3] = ca ? xk[((size_t)r * dense + 1) * 3 + c3] : pbM[r] ? xk[(size_t)pbI[r] * 3 + c3] : 0.f;
        float* dBeta = upload(beta.data(), beta.size());
        shardedConf.push_back(confidenceSharded(*t.zS, t.shardLo, t.shardRows, t.single, t.targetFeat, dBeta, t.seqMask,
                                                t.pairMask, t.n));
        CK(cudaFree(dBeta));
      }
      if (contact.empty()) contact = contactProbabilitiesSharded(t);    // (made off the whole pair where it was gathered)
      mg::tick("contacts");
      confMs += ms(sc0, clock()); s0 += clock() - sc0;   // (the confidence head's time its own, not the sampler's)
      mg::release({ "" });
      mg::tick("shared buffers released");
      if (mg::RANK != 0) mg::leave();
      mg::finish();
      mg::WORLD = 1;
      mg::tick("the other ranks exited");
      t.pair = nullptr; t.shardLo = -1; t.shardRows = 0; t.zS = t.zTS = nullptr; t.msa = nullptr;
      sh::DLO = -1; sh::DROWS = 0;
    }
    // the steps' precomputed conditioning (steps x tokens rows, 5 GB at 100 steps and 10761 tokens) is read by
    // nothing after the sampler - the next batch makes its own - and held to the end it left the confidence
    // head short of room
    if (df.preSingle) { CK(cudaFree(df.preSingle)); CK(cudaFree(df.preSnProj)); df.preSingle = df.preSnProj = nullptr; }
    NS = 1;
    releaseConcatCopies(); memReport("diffusion");
    diffMs += ms(s0, clock());
    if (tight) releaseScratch();
    size_t atoms3 = mask.size() * 3;
    // the confidence head reads the pseudo-beta of the token space it runs in
    std::vector<int> pbIdx(M.i("batch.tokenAtomsToPseudoBeta.indices"), M.i("batch.tokenAtomsToPseudoBeta.indices") + nD);
    std::vector<float> pbMask(M.f("batch.tokenAtomsToPseudoBeta.mask"), M.f("batch.tokenAtomsToPseudoBeta.mask") + nD);
    if (structural) swapBatch();     // back to the residues, for the confidence's layout and the structure
    // each sample's host work - its clash and disorder scores and, with several samples, its files - runs on a
    // thread beside the next sample's confidence head, joined at the batch's end (before the next batch's
    // swapBatch can change what the model's batch entries mean): 5 x ~23 ms of a 5-sample fold
    struct SampleDone { ConfidenceOut ck; std::vector<float> xk; uint64_t sd; int sk, k; std::string path;
                        StructureScores ss; double score; };
    std::vector<std::unique_ptr<SampleDone>> done;
    std::vector<std::future<void>> hostWork;
    foldAtoms();                     // (built here, before any thread reads it)
    for (int k = 0; k < (int)cn; ++k) {
      const uint64_t sd = runs[c0 + k].first; const int sk = runs[c0 + k].second;
      std::vector<float> xk(xs.begin() + k * atoms3, xs.begin() + (k + 1) * atoms3);
      auto s1 = clock();
      // pseudo-beta off the structure, then the confidence head
      std::vector<float> beta((size_t)nD * 3);
      // rf3's head reads the token-centre CA (dense slot 1), not the pseudo-beta
      bool ca = M.flag("trunk.dialect.confidenceCaDgram");
      for (int r = 0; r < nD; ++r) for (int a = 0; a < 3; ++a)
        beta[r * 3 + a] = ca ? xk[((size_t)r * dense + 1) * 3 + a] : pbMask[r] ? xk[(size_t)pbIdx[r] * 3 + a] : 0.f;
      float* dBeta = upload(beta.data(), beta.size());
      ConfidenceOut ck;
      if (structural) {
        // OpenDDE's own head on the structural tokens, mapped back: atoms through residueAtomGather,
        // pairs through each residue's representative (backbone) subtoken
        const bool lastHead = c0 + cn == runs.size() && k + 1 == (int)cn;   // (no later sample or batch reads st.pair)
        DdeConfidence dc = ddeConfidence(st.pair, st.single, st.targetFeat, dBeta, st.seqMask, st.pairMask, st.bias,
                                         nD, dense, t.n, lastHead);
        const int* gather = M.i("structural.residueAtomGather"); const int* rep = M.i("structural.residueRepToken");
        std::vector<float> xr((size_t)t.n * dense * 3, 0.f);
        ck.plddt.assign((size_t)t.n * dense, 0.f);
        for (size_t a = 0; a < (size_t)t.n * dense; ++a) {
          if (gather[a] < 0) continue;
          for (int d = 0; d < 3; ++d) xr[a * 3 + d] = xk[(size_t)gather[a] * 3 + d];
          ck.plddt[a] = dc.plddt[gather[a]];
        }
        xk = std::move(xr);
        ck.pae.resize((size_t)t.n * t.n); ck.pde.resize((size_t)t.n * t.n);
        std::vector<float> term((size_t)t.n * t.n);
        for (int i = 0; i < t.n; ++i) for (int j = 0; j < t.n; ++j) {
          size_t from = (size_t)rep[i] * nD + rep[j], to = (size_t)i * t.n + j;
          ck.pae[to] = dc.pae[from]; ck.pde[to] = dc.pde[from]; term[to] = dc.tmTerm[from];
        }
        auto reduce = [&](bool interOnly) {
          double bestTm = -1e30; bool any = false;
          for (int i = 0; i < t.n; ++i) {
            double tot = 0; int cnt = 0;
            for (int j = 0; j < t.n; ++j) {
              if (interOnly && resAsym[i] == resAsym[j]) continue;
              tot += term[(size_t)i * t.n + j]; ++cnt;
            }
            if (cnt) { any = true; bestTm = std::max(bestTm, tot / cnt); }
          }
          return any ? bestTm : NAN;
        };
        ck.ptm = reduce(false); ck.iptm = reduce(true);
        ck.tmTerm = term;
        const float* am = M.f("batch.refMask");
        double sum = 0, count = 0;
        for (size_t a = 0; a < ck.plddt.size(); ++a) if (am[a]) { sum += ck.plddt[a]; count += 1; }
        ck.meanPlddt = sum / std::max(count, 1.0);
      } else {
        if (!shardedConf.empty()) { ck = std::move(shardedConf[k]); }      // (made on every GPU above)
        else {
        if (!t.pair) unparkFromHost(t.pair, (size_t)t.n * t.n * t.C * (TRUNK_PAIR16 ? 2 : 4));    // (parked for the sampler)
        // the last confidence call of a fold short of room works in the trunk's pair (nothing reads it after)
        bool last = c0 + k + 1 == runs.size();
        ck = confidenceHead(t.pair, t.single, t.targetFeat, dBeta, t.seqMask, t.pairMask, t.n,
                            last && shortPair((size_t)t.n * t.n, t.C));
        releaseConcatCopies(); memReport("confidence");
        }
      }
      CK(cudaFree(dBeta));
      confMs += ms(s1, clock());
      auto d = std::make_unique<SampleDone>();
      d->ck = std::move(ck); d->xk = std::move(xk); d->sd = sd; d->sk = sk; d->k = k;
      if (many) {
        std::string tag = (seedList.size() > 1 ? "_seed" + std::to_string(sd) : std::string()) + "_sample" + std::to_string(sk);
        d->path = out == "/dev/null" ? out : stem + tag + ext;
      }
      SampleDone* dp = d.get(); done.push_back(std::move(d));
      hostWork.push_back(std::async(std::launch::async, [dp, many, n = t.n, dense, &contact] {
        dp->ss = structureScores(dp->xk);              // AF3's clash and disorder terms, this sample's
        dp->score = rankingScore(dp->ck.ptm, dp->ck.iptm, dp->ss);
        if (!many) return;
        auto order = writeStructure(dp->path, dp->xk, dp->ck.plddt.data());
        if (dp->path != "/dev/null")
          writeConfidences(dp->path, order, n, dense, dp->ck.plddt, dp->ck.pae, dp->ck.tmTerm, contact, dp->ck.ptm,
                           dp->ck.iptm, dp->score, dp->ck.meanPlddt, dp->ss.clash, dp->ss.disordered);
      }));
    }
    for (auto& w : hostWork) w.get();
    for (auto& dp : done) {          // (in sample order: the lines, the ranking and the best, as before)
      if (many) {
        printf("  seed %llu sample %d: mean pLDDT %.2f  pTM %.4f  ipTM %.4f  ranking %.4f -> %s\n", (unsigned long long)dp->sd,
               dp->sk, dp->ck.meanPlddt, dp->ck.ptm, dp->ck.iptm, dp->score, dp->path.c_str());
        char row[96]; snprintf(row, sizeof row, "%llu,%d,%.17g", (unsigned long long)dp->sd, dp->sk, dp->score);
        ranking.push_back(row);
      }
      if (!std::isfinite(dp->score)) { fprintf(stderr, "sample %d: ranking score %f is not finite\n", dp->k, dp->score); exit(1); }
      if (dp->score > bestScore) {
        bestScore = dp->score; best = dp->sk; bestSeed = dp->sd; conf = std::move(dp->ck); x = std::move(dp->xk); bestSS = dp->ss;
      }
    }
    }
    if (!framesDir.empty()) {
      frames.finish();
      printf("frames: %d written, %d dropped\n", frames.written, TAP().dropped);
    }
    if (many && out != "/dev/null") {        // AlphaFold 3's ranking_scores.csv
      FILE* rf = fopen((stem + "_ranking_scores.csv").c_str(), "w");
      fprintf(rf, "seed,sample,ranking_score\n");
      for (auto& r : ranking) fprintf(rf, "%s\n", r.c_str());
      fclose(rf);
    }
    if (structural) {
      for (float* p : {st.single, st.pair, st.targetFeat, st.bias, st.seqMask, st.pairMask}) CK(cudaFree(p));
    }
    auto f2 = clock();
    auto f3 = f2;
    auto order = writeStructure(out, x, conf.plddt.data());     // per-atom pLDDT in the B-factor column
    if (out != "/dev/null") writeConfidences(out, order, t.n, dense, conf.plddt, conf.pae, conf.tmTerm, contact, conf.ptm,
                                             conf.iptm, bestScore, conf.meanPlddt, bestSS.clash, bestSS.disordered);
    if (many) printf("  best: seed %llu sample %d\n", (unsigned long long)bestSeed, best);
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
    printf("fold %d: trunk %.1f ms (%d passes), diffusion %.1f ms (%d steps x %d%s), confidence %.1f ms, total %.1f ms\n",
           fi + 1, ms(f0, f1), passesRun, diffMs, steps, samples,
           seedList.size() > 1 ? (" x " + std::to_string(seedList.size()) + " seeds").c_str() : "", confMs, ms(f0, f3));
    if (profiling) prof::stop(40);
    if (fi == 0 && which == 0 && serveDir.empty() && getenv("LOCALFOLD_UNREAD")) unreadWeights();
    if (df.graph) CK(cudaGraphExecDestroy(df.graph));
    if (df.preSingle) { CK(cudaFree(df.preSingle)); CK(cudaFree(df.preSnProj)); }
    if (tight) releaseScratch();      // the next fold's trunk starts from the card it had
    if (fi + 1 == folds) return 0;
  }
  std::function<void(const char*, const float*, size_t)> seam = [&](const char* name, const float* d, size_t n) {
    std::string tap = std::string("oracle.trunk.stages.tap.") + name;
    std::string plain = std::string("oracle.trunk.stages.") + name;
    check(name, d, n, M.has(tap) ? tap : plain);
    // LOCALFOLD_SAVE_SEAMS=<dir>: each seam as <dir>/<name>.npy too, for an oracle too large to load here
    if (const char* dir = getenv("LOCALFOLD_SAVE_SEAMS")) writeNpy(std::string(dir) + "/" + name + ".npy", download(d, n), { n });
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
  };
  // --detach-output: the last line is "af3: done" and stdout closes, so a caller reading it to its
  // end can return while the driver releases this process's device memory (0.25 s, the rest of the
  // exit); cuda/af3/fold does
  if (!serveDir.empty()) {
    // the job's flags: --out, --samples, --steps, --flow, --sigma-max, --frames, --recycles, --seed, --seeds
    const int steps0 = steps, recycles0 = recycles, samples0 = samples, folds0 = folds;
    const bool flow0 = SAMPLER_FLOW; const double sigmaMax0 = SAMPLER_SIGMA_MAX;
    serveJobs(serveDir, "af3", [&](const std::string& input, const std::vector<std::string>& flags) {
      steps = steps0; recycles = recycles0; samples = samples0; folds = folds0; out = "fold.pdb"; SAMPLER_FLOW = flow0; SAMPLER_SIGMA_MAX = sigmaMax0;
      framesDir.clear();
      bool jobSeed = false; seed = seedArg; seedsArg = seedsArg0;
      for (auto& f : flags) {
        if (!f.compare(0, 6, "--out=")) out = f.substr(6);
        else if (!f.compare(0, 10, "--samples=")) samples = atoi(f.c_str() + 10);
        else if (!f.compare(0, 8, "--steps=")) steps = atoi(f.c_str() + 8);
        else if (f == "--flow") SAMPLER_FLOW = true;
        else if (!f.compare(0, 12, "--sigma-max=")) SAMPLER_SIGMA_MAX = atof(f.c_str() + 12);
        else if (!f.compare(0, 9, "--frames=")) framesDir = f.substr(9);
        else if (!f.compare(0, 11, "--recycles=")) recycles = atoi(f.c_str() + 11);
        else if (!f.compare(0, 7, "--seed=")) { seed = strtoull(f.c_str() + 7, nullptr, 10); jobSeed = true; }
        else if (!f.compare(0, 8, "--seeds=")) seedsArg = f.substr(8);
      }
      seedGiven = jobSeed;
      inputs = {input}; outs = {out};
      int seg = (int)M.segs.size();
      int code = runInput(0);
      CK(cudaDeviceSynchronize());
      freeTrunk(t); CHIRALITY = Chirality{};
      forgetEntries(M.unload(seg));
      return code;
    });
    return 0;
  }
  bool detach = false;
  for (int i = 2; i < argc; ++i) if (!strcmp(argv[i], "--detach-output")) detach = true;
  for (size_t which = 0; which < inputs.size(); ++which) {
    int seg = (int)M.segs.size();
    int code = runInput(which);
    if (code) return code;
    if (which + 1 < inputs.size()) {       // the next input reads its own fields afresh
      CK(cudaDeviceSynchronize());
      freeTrunk(t);
      CHIRALITY = Chirality{};
      forgetEntries(M.unload(seg));
    }
  }
  if (detach) { printf("af3: done\n"); fflush(stdout); fflush(stderr); fclose(stdout); }
  return 0;
}

// `af3 --job=<job.json> --out=<pdb>` (any first argument that is a flag): the whole protocol in this one process -
// the weights fetched, the input featurised in-process while the device starts, the fold (cuda/featurise/standalone.h);
// `af3 <featurised dir> ...` and `af3 - --serve=<dir>` as before
int main(int argc, char** argv) {
  {
    const int world = mg::gpusArg(argc, argv);   // --gpus=N|all: one fold across several GPUs (before any CUDA call: multigpu.cuh)
    if (world > 1)
      for (int i = 1; i < argc; ++i)
        if (!strncmp(argv[i], "--serve", 7) || !strncmp(argv[i], "--folds=", 8) || !strncmp(argv[i], "--repeat=", 9) ||
            !strncmp(argv[i], "--save-", 7) || !strncmp(argv[i], "--frames", 8) || !strncmp(argv[i], "--recycle-tolerance", 19)) {
          fprintf(stderr, "several GPUs fold one job at a time: no %s\n", argv[i]); return 1;
        }
    mg::launch(world);
  }
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) return lf::standalone::main("af3", argc, argv, foldMain);
  return foldMain(argc, argv);
}
