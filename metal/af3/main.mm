// localfold-af3 on Metal: the AlphaFold 3 lineage natively over cuda/featurise's input, a bundle read through its
// weight walk (cuda/featurise/af3_weights.h).
//
//   localfold-af3 <input dir> --bundle=<dir> --family=<model> [--out=fold.pdb] [--steps=200] [--recycles=3]
//                 [--samples=1] [--seed=N | --seeds=a,b] [--flow] [--sigma-max=X] [--af3-defaults]
//   localfold-af3 --job=<job.json> --out=<pdb> [--model=boltz2] ...      (featurised in this process)
#include "af3.h"
#include "af3_weights.h"
#include "output.h"
#include "standalone_api.h"
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

extern const char* PORT_SOURCE;

namespace {
double ms(double since) { return (now() - since) * 1e3; }
void saveSeam(const char* name, const float* d, size_t n) {
  const char* dir = getenv("LOCALFOLD_SAVE_SEAMS");
  if (!dir) return;
  std::vector<float> h = download(d, n);
  FILE* f = fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb");
  fwrite(h.data(), 4, h.size(), f); fclose(f);
}
uint64_t sampleSeed(uint64_t seed, int k) { return seed + ((uint64_t)k << 32); }

struct Options {
  std::string out = "fold.pdb", seedsArg;
  int steps = 200, recycles = 3, samples = 1, msaCap = 1024;
  uint64_t seed = 42; bool seedGiven = false;
};

// OpenDDE's head on the structural tokens, mapped back to the residues: atoms through residueAtomGather, pairs through
// each residue's representative (backbone) subtoken. xk (the structural atoms) is replaced by the residues' atoms.
ConfidenceOut structuralConfidence(const Structural& st, const float* coords, std::vector<float>& xk, int nRes, int dense,
                                   const std::vector<int>& resAsym) {
  DdeConfidence dc = ddeConfidence(st, coords, dense, nRes);
  int nD = st.n;
  const int* gather = M.hostI("structural.residueAtomGather"); const int* rep = M.hostI("structural.residueRepToken");
  ConfidenceOut ck;
  std::vector<float> xr((size_t)nRes * dense * 3, 0.f);
  ck.plddt.assign((size_t)nRes * dense, 0.f);
  for (size_t a = 0; a < (size_t)nRes * dense; ++a) {
    if (gather[a] < 0) continue;
    for (int d = 0; d < 3; ++d) xr[a * 3 + d] = xk[(size_t)gather[a] * 3 + d];
    ck.plddt[a] = dc.plddt[gather[a]];
  }
  xk = std::move(xr);
  ck.pae.resize((size_t)nRes * nRes); ck.pde.resize((size_t)nRes * nRes);
  std::vector<float> term((size_t)nRes * nRes);
  for (int i = 0; i < nRes; ++i)
    for (int j = 0; j < nRes; ++j) {
      size_t from = (size_t)rep[i] * nD + rep[j], to = (size_t)i * nRes + j;
      ck.pae[to] = dc.pae[from]; ck.pde[to] = dc.pde[from]; term[to] = dc.tmTerm[from];
    }
  auto reduce = [&](bool interOnly) {
    double bestTm = -1e30; bool any = false;
    for (int i = 0; i < nRes; ++i) {
      double tot = 0; int cnt = 0;
      for (int j = 0; j < nRes; ++j) { if (interOnly && resAsym[i] == resAsym[j]) continue; tot += term[(size_t)i * nRes + j]; ++cnt; }
      if (cnt) { any = true; bestTm = std::max(bestTm, tot / cnt); }
    }
    return any ? bestTm : NAN;
  };
  ck.ptm = reduce(false); ck.iptm = reduce(true);
  ck.tmTerm = term;
  const float* am = M.hostF("batch.refMask");
  double sum = 0, count = 0;
  for (size_t a = 0; a < ck.plddt.size(); ++a) if (am[a]) { sum += ck.plddt[a]; count += 1; }
  ck.meanPlddt = sum / std::max(count, 1.0);
  return ck;
}

int foldInput(const std::string& dir, Options o) {
  double t0 = now();
  M.loadInput(dir);
  setOutputInput(dir);
  int n = (int)M.meta("batch.tokens"), dense = metaI("batch.dense");
  ADA_RAW = flag("trunk.dialect.chaiAtomStack");
  // the seeds: --seeds, else the job's modelSeeds, else the one seed
  std::vector<uint64_t> seeds;
  uint64_t seed = !o.seedGiven && M.has("job.seed") ? (uint64_t)M.meta("job.seed") : o.seed;
  if (!o.seedsArg.empty()) {
    std::stringstream ss(o.seedsArg); std::string part;
    while (std::getline(ss, part, ',')) if (!part.empty()) seeds.push_back(strtoull(part.c_str(), nullptr, 10));
  } else if (!o.seedGiven && M.has("job.seeds.count")) {
    for (int k = 0; k < metaI("job.seeds.count"); ++k) seeds.push_back((uint64_t)M.meta("job.seeds." + num(k)));
  } else seeds.push_back(seed);
  printf("input: %d tokens\n", n);
  // target_feat, then the trunk's passes
  double f0 = now();
  float* tf = buildTargetFeat();
  int F = metaI("trunk.embedder.targetFeatWidth");
  saveSeam("target_feat", tf, (size_t)n * F);
  Trunk t = makeTrunk(tf, o.msaCap);
  releaseScratch({"targetFeat"});
  for (int pass = 0; pass <= o.recycles; ++pass) {
    if (pass == 0 && profiling()) profileStart();
    runTrunk(t);
    if (pass == 0) {
      saveSeam("trunk_out_pair", t.pair, (size_t)n * n * t.C);
      saveSeam("single", t.single, (size_t)n * t.Cs);
      if (profiling()) profileReport("trunk pass", 30);
    }
  }
  releaseScratch({"tri.", "grid.", "tr.", "st."});
  std::vector<float> contact = contactProbabilities(t);
  mt::sync();
  double trunkMs = ms(f0);
  // OpenDDE: the expander and refiner, then the diffusion and its confidence head on the structural tokens (the
  // structural batch swapped in for them, and back out for the files)
  const bool structural = flag("trunk.dialect.structuralTokens");
  Structural st{};
  int nD = n;
  std::vector<int> resAsym(M.hostI("batch.asymId"), M.hostI("batch.asymId") + n);
  if (structural) { st = expandStructural(t); M.swapPrefix("batch.", "sbatch."); nD = st.n; }
  // the diffusion: every (seed, sample) through the denoiser together, up to ten a batch
  double d0 = now();
  if (structural) prepareDiffusion(st.single, st.pair, st.targetFeat, st.masks, nD);
  else prepareDiffusion(t.single, t.pair, t.targetFeat, t.masks, n);
  std::vector<float> mask(M.hostF("batch.refMask"), M.hostF("batch.refMask") + (size_t)nD * dense);
  std::vector<int> pbIdx(M.hostI("batch.tokenAtomsToPseudoBeta.indices"), M.hostI("batch.tokenAtomsToPseudoBeta.indices") + nD);
  std::vector<float> pbMask(M.hostF("batch.tokenAtomsToPseudoBeta.mask"), M.hostF("batch.tokenAtomsToPseudoBeta.mask") + nD);
  bool caDgram = flag("trunk.dialect.confidenceCaDgram");
  if (SAMPLER.flow && flag("trunk.dialect.noFlowSampler")) die("this checkpoint has no working flow sampler - fold it with diffusion");
  std::vector<std::pair<uint64_t, int>> runs;
  for (uint64_t s : seeds) for (int k = 0; k < o.samples; ++k) runs.push_back({s, k});
  const size_t perBatch = std::max<size_t>(o.samples, 10);
  bool many = runs.size() > 1;
  std::string ext = o.out.size() > 4 && o.out.substr(o.out.size() - 4) == ".cif" ? ".cif" : ".pdb";
  std::string stem = o.out.size() > 4 && o.out.substr(o.out.size() - 4) == ext ? o.out.substr(0, o.out.size() - 4) : o.out;
  double diffMs = 0, confMs = 0, bestScore = -1e30;
  ConfidenceOut best; std::vector<float> bestX; Scores bestS{false, 0}; uint64_t bestSeed = 0; int bestSample = 0;
  std::vector<std::string> ranking;
  size_t atoms3 = mask.size() * 3;
  for (size_t c0 = 0; c0 < runs.size(); c0 += perBatch) {
    size_t cn = std::min(perBatch, runs.size() - c0);
    std::vector<uint64_t> batch;
    for (size_t k = 0; k < cn; ++k) batch.push_back(sampleSeed(runs[c0 + k].first, runs[c0 + k].second));
    double s0 = now();
    if (profiling()) profileStart();
    if (structural && c0 > 0) M.swapPrefix("batch.", "sbatch.");      // (the structural tokens again, for this batch)
    std::vector<float> xs = sample(o.steps, batch, mask);
    if (structural) M.swapPrefix("batch.", "sbatch.");                 // (back to the residues, for the files)
    if (profiling()) profileReport("diffusion", 30);
    if (getenv("AF3_STAGES")) reportStages();
    diffMs += ms(s0);
    for (size_t k = 0; k < cn; ++k) {
      double s1 = now();
      std::vector<float> xk(xs.begin() + k * atoms3, xs.begin() + (k + 1) * atoms3);
      std::vector<float> beta((size_t)nD * 3);
      for (int r = 0; r < nD; ++r)
        for (int a = 0; a < 3; ++a)
          beta[r * 3 + a] = caDgram ? xk[((size_t)r * dense + 1) * 3 + a] : pbMask[r] ? xk[(size_t)pbIdx[r] * 3 + a] : 0.f;
      float* dBeta = scratch<float>("main.beta", beta.size());
      upload(dBeta, beta.data(), beta.size() * 4);
      ConfidenceOut ck = structural ? structuralConfidence(st, dBeta, xk, t.n, dense, resAsym) : confidenceHead(t, dBeta);
      confMs += ms(s1);
      Scores ss = structureScores(xk);
      double score = rankingScore(ck.ptm, ck.iptm, ss);
      uint64_t sd = runs[c0 + k].first; int sk = runs[c0 + k].second;
      if (many) {
        std::string path = o.out == "/dev/null" ? o.out
          : stem + (seeds.size() > 1 ? "_seed" + std::to_string(sd) : std::string()) + "_sample" + std::to_string(sk) + ext;
        auto order = writeStructure(path, xk, ck.plddt.data());
        if (path != "/dev/null") writeConfidenceFiles(path, order, n, dense, ck, contact, score, ss);
        printf("  seed %llu sample %d: mean pLDDT %.2f  pTM %.4f  ipTM %.4f  ranking %.4f -> %s\n", (unsigned long long)sd, sk,
               ck.meanPlddt, ck.ptm, ck.iptm, score, path.c_str());
        char row[96]; snprintf(row, sizeof row, "%llu,%d,%.17g", (unsigned long long)sd, sk, score);
        ranking.push_back(row);
      }
      if (!std::isfinite(score) || !std::isfinite(ck.meanPlddt) || ck.ptm < -1)
        die("sample %d: the fold is not finite (pLDDT %f, pTM %f) - a NaN upstream", sk, ck.meanPlddt, ck.ptm);
      if (score > bestScore) { bestScore = score; best = std::move(ck); bestX = std::move(xk); bestS = ss; bestSeed = sd; bestSample = sk; }
    }
  }
  if (many && o.out != "/dev/null") {
    FILE* rf = fopen((stem + "_ranking_scores.csv").c_str(), "w");
    fprintf(rf, "seed,sample,ranking_score\n");
    for (auto& r : ranking) fprintf(rf, "%s\n", r.c_str());
    fclose(rf);
    printf("  best: seed %llu sample %d\n", (unsigned long long)bestSeed, bestSample);
  }
  auto order = writeStructure(o.out, bestX, best.plddt.data());
  if (o.out != "/dev/null") writeConfidenceFiles(o.out, order, n, dense, best, contact, bestScore, bestS);
  printf("mean pLDDT %.2f  pTM %.4f  ipTM %.4f  -> %s\n", best.meanPlddt, best.ptm, best.iptm, o.out.c_str());
  printf("fold: trunk %.1f ms (%d passes), diffusion %.1f ms (%d steps x %d), confidence %.1f ms, total %.1f ms\n", trunkMs,
         o.recycles + 1, diffMs, o.steps, (int)runs.size(), confMs, ms(t0));
  (void)d0;
  mt::sync();
  freeDiffusion();
  if (structural) freeStructural(st);
  freeTrunk(t);
  releaseScratch();
  M.unloadInput();
  return 0;
}
}  // namespace

int foldMain(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: localfold-af3 <input dir> --bundle=<dir> --family=<model> [--out=fold.pdb]\n"); return 1; }
  std::string bundleDir, family;
  Options o;
  bool af3Defaults = false, waitInput = false, detach = false, setR = false, setS = false;
  for (int i = 2; i < argc; ++i) {
    const char* a = argv[i];
    if (!strncmp(a, "--bundle=", 9)) bundleDir = a + 9;
    else if (!strncmp(a, "--family=", 9)) family = a + 9;
    else if (!strncmp(a, "--out=", 6)) o.out = a + 6;
    else if (!strncmp(a, "--steps=", 8)) o.steps = atoi(a + 8);
    else if (!strncmp(a, "--recycles=", 11)) { o.recycles = atoi(a + 11); setR = true; }
    else if (!strncmp(a, "--samples=", 10)) { o.samples = atoi(a + 10); setS = true; }
    else if (!strncmp(a, "--msa=", 6)) o.msaCap = atoi(a + 6);
    else if (!strncmp(a, "--seed=", 7)) { o.seed = strtoull(a + 7, nullptr, 10); o.seedGiven = true; }
    else if (!strncmp(a, "--seeds=", 8)) o.seedsArg = a + 8;
    else if (!strcmp(a, "--flow")) SAMPLER.flow = true;
    else if (!strncmp(a, "--sigma-max=", 12)) SAMPLER.sigmaMax = atof(a + 12);
    else if (!strcmp(a, "--af3-defaults")) af3Defaults = true;
    else if (!strncmp(a, "--score-pdb=", 12)) return scorePdbMain(a + 12);
    else if (!strncmp(a, "--wait-input", 12)) waitInput = true;
    else if (!strcmp(a, "--detach-output")) detach = true;
    else if (!strcmp(a, "--profile")) setenv("LOCALFOLD_PROFILE", "1", 1);
    else if (!strcmp(a, "--fast") || !strcmp(a, "--fold")) {}
    else { fprintf(stderr, "unknown flag %s\n", a); return 1; }
  }
  if (af3Defaults) { if (!setR) o.recycles = 10; if (!setS) o.samples = 5; }
  if (bundleDir.empty() || family.empty()) { fprintf(stderr, "--bundle=<dir> --family=<model>\n"); return 1; }
  double t0 = now();
  setSource("af3", PORT_SOURCE);
  {   // the bundle read as published, through the family's weight walk
    lf::weights::Shapes S; S.shape = Model::bundleShapes(bundleDir);
    std::vector<std::string> lines = lf::weights::af3WeightLines(family, S);
    M.loadBundleWalk(bundleDir, lines, "", [](const std::string&, size_t elements) { return elements >= 16384; });
  }
  printf("weights: %.0f ms\n", ms(t0));
  if (waitInput) {      // the featuriser writes model.idx last (by a rename), or model.failed
    std::string idx = std::string(argv[1]) + "/model.idx", failed = std::string(argv[1]) + "/model.failed";
    while (access(idx.c_str(), R_OK) != 0) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "af3: the input's export failed\n"); return 1; }
      usleep(1000);
    }
  }
  if (getenv("LOCALFOLD_SAVE_SEAMS")) SEAM = saveSeam;
  int code = foldInput(argv[1], o);
  if (detach) { printf("af3: done\n"); fflush(stdout); fflush(stderr); fclose(stdout); }
  return code;
}

int main(int argc, char** argv) {
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) return lf::standalone::main("af3", argc, argv, foldMain);
  return foldMain(argc, argv);
}
