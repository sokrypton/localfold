// AlphaFold 3 in CUDA - the driver.
//
//   af3 <data-dir> [--fast] [--stages] [--repeat=N]
//
// Reads <data-dir>/model.{idx,bin} (export-model.mjs). Without --fast it runs the precise
// path (f32 throughout) and checks every seam the oracle recorded; with --fast the f16 path.
#include <dirent.h>
#include "trunk.cuh"
#include "atom.cuh"
#include "diffusion.cuh"
#include "sampler.cuh"
#include "scores.cuh"
#include "confidence.cuh"
#include "structural.cuh"
#include "benchops.cuh"
#include "profile.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: af3 <data-dir> [--fast] [--stages] [--repeat=N]\n"); return 1; }
  bool fast = false, doFold = false, profile = false; int repeat = 1, msaCap = 1024, steps = 200, recycles = 3, folds = 1, samples = 1;   // 3 recycles: the page's default
  // --af3-defaults: AlphaFold 3's own run_alphafold.py settings - 10 recycles (11 trunk passes) and
  // 5 diffusion samples - where the command does not set them; the plain defaults are the page's
  bool af3Defaults = false, saveEmbeddings = false, saveDistogram = false;
  uint64_t seed = 42; std::string out = "fold.pdb", weightsDir, bundleDir, mapFile, seedsArg, framesDir;
  bool waitInput = false;   // start up (CUDA, the weights on the device) while the input is still being exported
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
    else if (!strcmp(argv[i], "--no-graphs")) GRAPHS = false;
    else if (!strcmp(argv[i], "--profile")) profile = true;
    else if (!strcmp(argv[i], "--no-flash-split")) FLASH_SPLIT = false;
    else if (!strncmp(argv[i], "--folds=", 8)) folds = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--steps=", 8)) steps = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--recycles=", 11)) recycles = atoi(argv[i] + 11);
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
    else if (!strncmp(argv[i], "--map=", 6)) mapFile = argv[i] + 6;            // through the port's map (maps/<family>.map)
    else if (!strncmp(argv[i], "--score-pdb=", 12)) return scorePdbMain(argv[i] + 12);
    else if (!strcmp(argv[i], "--wait-input")) waitInput = true;
    else if (!strncmp(argv[i], "--serve=", 8)) serveDir = argv[i] + 8;
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
  if (!bundleDir.empty() != !mapFile.empty()) { fprintf(stderr, "--bundle and --map go together\n"); return 1; }
  if (!weightsDir.empty()) M.load(weightsDir);      // the weights exported once (--weights-only)
  else if (!bundleDir.empty()) M.loadBundle(bundleDir, "", mapFile);
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
    int cc = 0; cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, 0);
    if (cc >= 8)      // (bf16: the fused triangle's contraction, Ampere on - a T4 runs the unfused path)
      cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 96, 96, 96, &one, b, CUDA_R_16BF, 96, 9216, b + (1 << 20),
                                 CUDA_R_16BF, 96, 9216, &zero, b + (2 << 20), CUDA_R_16BF, 96, 9216, 8, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaStreamSynchronize(s);
    cudaFree(buf); cublasDestroy(h); cudaStreamDestroy(s);
  });
  struct JoinAtExit { std::thread& t; ~JoinAtExit() { if (t.joinable()) t.join(); } } joinWarm{cublasWarm};
  Trunk t{};
  if (haveWeights) M.upload(0);      // now, beside the cuBLAS warm-up (both are needed before any fold)
  if (cublasWarm.joinable()) cublasWarm.join();
  auto runInput = [&](size_t which) -> int {
  if (waitInput) {          // the exporter writes model.idx last, by a rename
    std::string idx = inputs[which] + "/model.idx";
    std::string failed = inputs[which] + "/model.failed";     // the wrapper's word that the export died
    for (int k = 0; access(idx.c_str(), R_OK) != 0; ++k) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "af3: the input's export failed\n"); return 1; }
      if (k > 600000) { fprintf(stderr, "no %s after ten minutes\n", idx.c_str()); return 1; }
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
  printf("loaded %zu entries in %.1f s; %d tokens\n", M.index.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
         (int)M.meta("batch.tokens"));

  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-ops=", 12)) { const char* a = argv[i] + 12; const char* x = strchr(a, 'x'); benchOps(atoi(a), x ? atoi(x + 1) : 1); return 0; }
  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-grid=", 13)) { benchGrid(atoi(argv[i] + 13)); return 0; }
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
  t = makeTrunk(targetFeat.data(), msaCap);
  memReport("trunk built");
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
    // A recycle pass is ~1000 launches with identical shapes and pointers, so from the second pass on
    // it replays as one CUDA graph, captured from that pass (the first has sized every scratch buffer)
    auto recyclePass = [&]() {
      CK(cudaMemcpyAsync(t.prevPair, t.pair, pairs * t.C * 4, cudaMemcpyDeviceToDevice, STREAM));
      CK(cudaMemcpyAsync(t.prevSingle, t.single, (size_t)t.n * t.Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
      if (fast) runTrunk<half>(t, none); else runTrunk<float>(t, none);
    };
    cudaGraphExec_t trunkGraph = nullptr;
    // --frames: each pass's contact map too (the page shows the trunk's after every recycle, before the
    // sampler has a structure), computed on the device, quantised to a byte a pair and tapped (AsyncTap):
    // contacts-PP-of-NN.u8, n*n bytes, probability * 255
    const bool tapContacts = !framesDir.empty() && M.has("batch.contactBins");
    if (tapContacts) TAP().reserve(recycles + 1, (size_t)t.n * t.n);
    auto afterPass = [&](int pass) {
      if (!tapContacts) return;
      int bins = (int)M.meta("trunk.distogram.bins");
      size_t pairs = (size_t)t.n * t.n;
      float* logits = scratch<float>("disto.logits", pairs * bins);
      distogram(t, logits);
      float* probs = scratch<float>("disto.contact", pairs);
      contactProbsK<<<blocks(pairs), 256, 0, STREAM>>>(logits, Idev("batch.contactBins"), t.pairMask, probs, pairs, bins);
      unsigned char* bytes = scratch<unsigned char>("disto.contact8", pairs);
      quantiseK<<<blocks(pairs), 256, 0, STREAM>>>(probs, bytes, pairs, 1.f / 255);
      std::string path = framesDir + "/contacts-" + (pass < 10 ? "0" : "") + std::to_string(pass) + "-of-"
                         + (recycles + 1 < 10 ? "0" : "") + std::to_string(recycles + 1) + ".u8";
      TAP().offer({{bytes, pairs}}, [path, pairs](const char* host, const std::vector<size_t>&) { writeWhole(path, host, pairs); });
    };
    for (int pass = 0; pass <= recycles; ++pass) {
      if (pass == 0) { if (fast) runTrunk<half>(t, none); else runTrunk<float>(t, none); afterPass(pass); continue; }
      // (capturing and instantiating costs ~15 ms and a replayed pass saves ~2 ms at 68 tokens, more
      // as the launches grow: a first fold breaks even at 7 recycles there - AF3's 10 gain 6 ms - and
      // at 3 from ~200 tokens, so the graph is taken where it measured a gain)
      if (!GRAPHS || STAGES || !(recycles >= 7 || t.n >= 200)) { recyclePass(); afterPass(pass); continue; }
      if (!trunkGraph) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(STREAM, cudaStreamCaptureModeThreadLocal));
        recyclePass();
        CK(cudaStreamEndCapture(STREAM, &g));
        CK(cudaGraphInstantiate(&trunkGraph, g, 0));
        CK(cudaGraphDestroy(g));
      }
      CK(cudaGraphLaunch(trunkGraph, STREAM));
      afterPass(pass);
    }
    if (trunkGraph) CK(cudaGraphExecDestroy(trunkGraph));
    CK(cudaDeviceSynchronize());
    memReport("trunk");
    auto f1 = clock();
    if (STAGES) {     // the trunk's stages, then the diffusion's below
      double total = 0; for (auto& [k, v] : STAGE_MS) total += v;
      printf("trunk stages:\n");
      for (auto& [k, v] : STAGE_MS) printf("  %-16s %9.1f ms  %4.1f%%\n", k.c_str(), v, 100 * v / total);
      STAGE_MS.clear();
    }
    // the distogram's contact probabilities, for the confidences file (off the residue batch)
    std::vector<float> contact = contactProbabilities(t);
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
    const float *dS = t.single, *dP = t.pair, *dTf = t.targetFeat, *dSeq = t.seqMask;
    int nD = t.n;
    std::vector<int> resAsym(M.i("batch.asymId"), M.i("batch.asymId") + t.n);
    if (structural) {
      st = expandStructural(t.single, t.pair, t.targetFeat, t.n, fast);
      swapBatch();
      dS = st.single; dP = st.pair; dTf = st.targetFeat; dSeq = st.seqMask; nD = st.n;
    }
    int dense = (int)M.meta("batch.dense");
    std::vector<float> mask(M.f("batch.refMask"), M.f("batch.refMask") + (size_t)nD * dense);
    // a large input gives each phase the whole card: a pair over 128 MB, 512 tokens at 128 channels
    // (at 1 GB, 1044 tokens peaked at 21.3 GB with the trunk's 8 GB of scratch held to the end)
    bool tight = pairs * t.C * 4 > ((size_t)128 << 20);
    if (tight) releaseScratch();
    DiffusionFold df = prepareDiffusion(dS, dP, dTf, dSeq, nD);
    memReport("diffusion prepared");
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
    FrameStreamer frames;
    for (size_t c0 = 0; c0 < runs.size(); c0 += perBatch) {
    const size_t cn = std::min(perBatch, runs.size() - c0);
    if (structural && c0 > 0) swapBatch();     // the structural tokens again, for this batch's diffusion
    auto s0 = clock();
    NS = (int)cn;
    std::vector<uint64_t> seeds;
    for (size_t k = 0; k < cn; ++k) seeds.push_back(sampleSeed(runs[c0 + k].first, runs[c0 + k].second));
    if (SAMPLER_FLOW && M.flag("trunk.dialect.noFlowSampler")) {
      // the page's own rule (noFlowSampler, src/af3/dialect.js): this checkpoint's walk collapses the
      // backbone while its pLDDT reads as if nothing were wrong
      fprintf(stderr, "this checkpoint has no working flow sampler - fold it with diffusion\n"); return 1;
    }
    if (!framesDir.empty() && c0 == 0) {          // (the first batch's first sample)
      TAP().reserve(FrameStreamer::planned(steps), mask.size() * 3 * 4);
      frames.start(framesDir, mask.size() * 3, steps);
    }
    std::vector<float> xs = sample(steps, seeds, mask, [&](const float* noisy, float tHat, const float* dLevel) {
      return (const float*)denoiseStep(df, noisy, tHat, dLevel);
    }, 0.8, 1.0, 1.003, 1.5, [&](const std::vector<float>& levels) { precomputeConditioning(df, levels); });
    FRAME_HOOK = nullptr;      // (the writer finishes the last frames while the confidence head runs)
    NS = 1;
    memReport("diffusion");
    diffMs += ms(s0, clock());
    if (tight) releaseScratch();
    size_t atoms3 = mask.size() * 3;
    // the confidence head reads the pseudo-beta of the token space it runs in
    std::vector<int> pbIdx(M.i("batch.tokenAtomsToPseudoBeta.indices"), M.i("batch.tokenAtomsToPseudoBeta.indices") + nD);
    std::vector<float> pbMask(M.f("batch.tokenAtomsToPseudoBeta.mask"), M.f("batch.tokenAtomsToPseudoBeta.mask") + nD);
    if (structural) swapBatch();     // back to the residues, for the confidence's layout and the structure
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
        DdeConfidence dc = ddeConfidence(st.pair, st.single, st.targetFeat, dBeta, st.seqMask, st.pairMask, st.bias,
                                         nD, dense, t.n);
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
        ck = confidenceHead(t.pair, t.single, t.targetFeat, dBeta, t.seqMask, t.pairMask, t.n);
        memReport("confidence");
      }
      CK(cudaFree(dBeta));
      confMs += ms(s1, clock());
      StructureScores ssk = structureScores(xk);       // AF3's clash and disorder terms, this sample's
      double score = rankingScore(ck.ptm, ck.iptm, ssk);
      if (many) {
        std::string tag = (seedList.size() > 1 ? "_seed" + std::to_string(sd) : std::string()) + "_sample" + std::to_string(sk);
        std::string path = out == "/dev/null" ? out : stem + tag + ext;
        auto order = writeStructure(path, xk, ck.plddt.data());
        if (path != "/dev/null") writeConfidences(path, order, t.n, dense, ck.plddt, ck.pae, ck.tmTerm, contact, ck.ptm, ck.iptm,
                                                  score, ck.meanPlddt, ssk.clash, ssk.disordered);
        printf("  seed %llu sample %d: mean pLDDT %.2f  pTM %.4f  ipTM %.4f  ranking %.4f -> %s\n", (unsigned long long)sd, sk,
               ck.meanPlddt, ck.ptm, ck.iptm, score, path.c_str());
        char row[96]; snprintf(row, sizeof row, "%llu,%d,%.17g", (unsigned long long)sd, sk, score); ranking.push_back(row);
      }
      if (!std::isfinite(score)) { fprintf(stderr, "sample %d: ranking score %f is not finite\n", k, score); exit(1); }
      if (score > bestScore) { bestScore = score; best = sk; bestSeed = sd; conf = std::move(ck); x = std::move(xk); bestSS = ssk; }
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
           fi + 1, ms(f0, f1), recycles + 1, diffMs, steps, samples,
           seedList.size() > 1 ? (" x " + std::to_string(seedList.size()) + " seeds").c_str() : "", confMs, ms(f0, f3));
    if (profiling) prof::stop(40);
    if (fi == 0 && which == 0 && serveDir.empty()) unreadWeights();
    if (df.graph) CK(cudaGraphExecDestroy(df.graph));
    if (df.preSingle) { CK(cudaFree(df.preSingle)); CK(cudaFree(df.preSnProj)); }
    if (df.preG) { CK(cudaFree(df.preG)); CK(cudaFree(df.preR)); }
    PRE_ADA = false;
    if (tight) releaseScratch();      // the next fold's trunk starts from the card it had
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
  };
  // --detach-output: the last line is "af3: done" and stdout closes, so a caller reading it to its
  // end can return while the driver releases this process's device memory (0.25 s, the rest of the
  // exit); native/af3/fold does
  if (!serveDir.empty()) {
    // a job is DIR/<id>.job (renamed into place): its first line the input's directory, then one
    // flag a line (--out, --samples, --steps, --recycles, --seed); its output goes to <id>.log and
    // its exit status to <id>.done. A job reading "quit" stops the server.
    const int steps0 = steps, recycles0 = recycles, samples0 = samples, folds0 = folds;
    const bool flow0 = SAMPLER_FLOW; const double sigmaMax0 = SAMPLER_SIGMA_MAX;
    printf("af3: serving %s\n", serveDir.c_str()); fflush(stdout);
    for (;;) {
      std::string id;
      if (DIR* d = opendir(serveDir.c_str())) {
        std::vector<std::string> jobs;
        while (dirent* e = readdir(d)) {
          std::string name = e->d_name;
          if (name.size() > 4 && name.substr(name.size() - 4) == ".job") jobs.push_back(name.substr(0, name.size() - 4));
        }
        closedir(d);
        if (!jobs.empty()) { std::sort(jobs.begin(), jobs.end()); id = jobs[0]; }
      }
      if (id.empty()) { usleep(2000); continue; }
      std::string base = serveDir + "/" + id;
      std::ifstream job(base + ".job");
      std::string input, line; std::getline(job, input);
      std::vector<std::string> flags; while (std::getline(job, line)) if (!line.empty()) flags.push_back(line);
      job.close(); unlink((base + ".job").c_str());
      if (input == "quit") { printf("af3: stopped\n"); return 0; }
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
      fflush(stdout);
      int saved = dup(1), log = open((base + ".log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      dup2(log, 1); close(log);
      inputs = {input}; outs = {out};
      int seg = (int)M.segs.size();
      int code = runInput(0);
      fflush(stdout); dup2(saved, 1); close(saved);
      CK(cudaDeviceSynchronize());
      freeTrunk(t); CHIRALITY = Chirality{};
      forgetEntries(M.unload(seg));
      FILE* df = fopen((base + ".done.tmp").c_str(), "w"); fprintf(df, "%d\n", code); fclose(df);
      rename((base + ".done.tmp").c_str(), (base + ".done").c_str());
    }
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
