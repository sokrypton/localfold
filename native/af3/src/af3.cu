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
#include "structural.cuh"
#include "benchops.cuh"
#include "profile.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: af3 <data-dir> [--fast] [--stages] [--repeat=N]\n"); return 1; }
  bool fast = false, doFold = false, profile = false; int repeat = 1, msaCap = 1024, steps = 200, recycles = 3, folds = 1, samples = 1;   // 3 recycles: the page's default
  uint64_t seed = 42; std::string out = "fold.pdb", weightsDir;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--fast")) fast = DIFF_HALF = ATOM_HALF = CONF_HALF = F32_TF32 = true;
    else if (!strcmp(argv[i], "--no-tf32")) F32_TF32 = false;
    else if (!strcmp(argv[i], "--no-tri-bf16")) TRI_BF16 = false;
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
    else if (!strncmp(argv[i], "--seed=", 7)) seed = strtoull(argv[i] + 7, nullptr, 10);
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
    else if (!strncmp(argv[i], "--weights=", 10)) weightsDir = argv[i] + 10;
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
  if (!weightsDir.empty()) M.load(weightsDir);      // the weights exported once (--weights-only)
  bool seedGiven = false;
  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--seed=", 7)) seedGiven = true;
  const uint64_t seedArg = seed;
  if (getenv("FLASH_WARPS")) FLASH_WARPS_OVERRIDE = atoi(getenv("FLASH_WARPS"));   // experiments
  if (getenv("FT_WARPS")) FT_WARPS = atoi(getenv("FT_WARPS"));
  if (getenv("TRI_PAD")) TRI_PAD = atoi(getenv("TRI_PAD"));
  if (profile) prof::init();
  CB(cublasCreate(&H));
  CB(cublasSetStream(H, STREAM));
  { void* ws; CK(cudaMalloc(&ws, 64 << 20)); CB(cublasSetWorkspace(H, ws, 64 << 20)); }   // graph capture needs it
  Trunk t{};
  auto runInput = [&](size_t which) -> int {
  M.load(inputs[which]); DATA_DIR = inputs[which];
  if (inputs.size() > 1) out = outs[which];
  { const float* sm = M.f("batch.seqMask"); size_t k = M.len("batch.seqMask"); MASK_ALL_ONES = true;
    for (size_t i = 0; i < k; ++i) if (!(sm[i] > 0)) MASK_ALL_ONES = false; }
  seed = !seedGiven && M.has("job.seed") ? (uint64_t)M.meta("job.seed") : seedArg;   // the job's own modelSeeds[0]
  printf("loaded %zu entries in %.1f s; %d tokens\n", M.index.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
         (int)M.meta("batch.tokens"));

  for (int i = 2; i < argc; ++i) if (!strncmp(argv[i], "--bench-ops=", 12)) { benchOps(atoi(argv[i] + 12)); return 0; }
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
    if (STAGES) {     // the trunk's stages, then the diffusion's below
      double total = 0; for (auto& [k, v] : STAGE_MS) total += v;
      printf("trunk stages:\n");
      for (auto& [k, v] : STAGE_MS) printf("  %-16s %9.1f ms  %4.1f%%\n", k.c_str(), v, 100 * v / total);
      STAGE_MS.clear();
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
    // a large input gives each phase the whole card (a pair over 1 GB: about 1450 tokens)
    bool tight = pairs * t.C * 4 > ((size_t)1 << 30);
    if (tight) releaseScratch();
    DiffusionFold df = prepareDiffusion(dS, dP, dTf, dSeq, nD);
    // --samples=N: N diffusion samples off one trunk (AF3 runs five), each through the confidence
    // head and ranked by AF3's ranking score without its disorder and clash terms - 0.8 ipTM +
    // 0.2 pTM, or pTM for one chain. The samples run as one batch through the denoiser, sample k
    // seeded `seed + k` (what a one-sample run with that seed draws); the best is written to --out
    // and every one to <out>_sample<k>.pdb.
    double diffMs = 0, confMs = 0, bestScore = -1e30; int best = 0;
    ConfidenceOut conf;
    std::vector<float> x;
    // the samples as one batch through the denoiser
    auto s0 = clock();
    NS = samples;
    std::vector<float> xs = sample(steps, seed, mask, [&](const float* noisy, float tHat, const float* dLevel) {
      return (const float*)denoiseStep(df, noisy, tHat, dLevel);
    }, samples);
    NS = 1;
    diffMs = ms(s0, clock());
    if (tight) releaseScratch();
    size_t atoms3 = mask.size() * 3;
    // the confidence head reads the pseudo-beta of the token space it runs in
    std::vector<int> pbIdx(M.i("batch.tokenAtomsToPseudoBeta.indices"), M.i("batch.tokenAtomsToPseudoBeta.indices") + nD);
    std::vector<float> pbMask(M.f("batch.tokenAtomsToPseudoBeta.mask"), M.f("batch.tokenAtomsToPseudoBeta.mask") + nD);
    if (structural) swapBatch();     // back to the residues, for the confidence's layout and the structure
    for (int k = 0; k < samples; ++k) {
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
        const float* am = M.f("batch.refMask");
        double sum = 0, count = 0;
        for (size_t a = 0; a < ck.plddt.size(); ++a) if (am[a]) { sum += ck.plddt[a]; count += 1; }
        ck.meanPlddt = sum / std::max(count, 1.0);
      } else {
        ck = confidenceHead(t.pair, t.single, t.targetFeat, dBeta, t.seqMask, t.pairMask, t.n);
      }
      CK(cudaFree(dBeta));
      confMs += ms(s1, clock());
      double score = std::isnan(ck.iptm) ? ck.ptm : 0.8 * ck.iptm + 0.2 * ck.ptm;
      if (samples > 1) {
        std::string path = out.size() > 4 && out.substr(out.size() - 4) == ".pdb"
          ? out.substr(0, out.size() - 4) + "_sample" + std::to_string(k) + ".pdb" : out + "_sample" + std::to_string(k);
        auto order = writePdb(path, xk, ck.plddt.data());
        if (path != "/dev/null") writeConfidences(path, order, ck.plddt, ck.pae, t.n, ck.ptm, ck.iptm, score);
        printf("  sample %d: mean pLDDT %.2f  pTM %.4f  ipTM %.4f  ranking %.4f -> %s\n", k, ck.meanPlddt, ck.ptm,
               ck.iptm, score, path.c_str());
      }
      if (!std::isfinite(score)) { fprintf(stderr, "sample %d: ranking score %f is not finite\n", k, score); exit(1); }
      if (score > bestScore) { bestScore = score; best = k; conf = std::move(ck); x = std::move(xk); }
    }
    if (structural) {
      for (float* p : {st.single, st.pair, st.targetFeat, st.bias, st.seqMask, st.pairMask}) CK(cudaFree(p));
    }
    auto f2 = clock();
    auto f3 = f2;
    auto order = writePdb(out, x, conf.plddt.data());     // per-atom pLDDT in the B-factor column
    if (out != "/dev/null") writeConfidences(out, order, conf.plddt, conf.pae, t.n, conf.ptm, conf.iptm, bestScore);
    if (samples > 1) printf("  best: sample %d\n", best);
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
    printf("fold %d: trunk %.1f ms (%d passes), diffusion %.1f ms (%d steps x %d), confidence %.1f ms, total %.1f ms\n",
           fi + 1, ms(f0, f1), recycles + 1, diffMs, steps, samples, confMs, ms(f0, f3));
    if (profiling) prof::stop(40);
    if (fi == 0 && which == 0) unreadWeights();
    if (df.graph) CK(cudaGraphExecDestroy(df.graph));
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
  return 0;
}
