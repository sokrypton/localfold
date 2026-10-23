// localfold-ef2 on Metal: ESMFold2 (ef2-fast-600m, ef2-fast-300m) natively - ESM-C and its shim, the inputs
// embedder, z_init and the pair trunk, the contact map, the diffusion sampler and the confidence head - over
// cuda/featurise's input (the same featuriser, compiled in: `localfold-ef2 --job=... --out=...`).
//
//   localfold-ef2 <input dir> --fold-bundle=<dir> --esmc-bundle=<dir> [--out=fold.pdb] [--seed=N] [--steps=N]
#include "ef2.h"
#include "host.h"
#include "standalone_api.h"
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <fstream>

extern const char* PORT_SOURCE;

namespace {
struct Opts { std::string dir, out = "fold.pdb", frames; uint64_t seed = 0; SamplerSettings sampler; bool stepsGiven = false, profile = false; };

double ms(double since) { return (now() - since) * 1e3; }
void mem(const char* at) {
  if (!getenv("LOCALFOLD_MEM")) return;
  mt::sync();
  printf("  memory %-22s %6.2f GB held (scratch %.2f, weights %.2f)\n", at, allocated() / 1e9, scratchHeld() / 1e9, M.weightBytes() / 1e9);
}

// the fit check's factors, measured on the M5 at 522 and 1044 residues: the language model's stage holds the tower and
// its f32 pair (1.25 a pair byte and 0.1 GB), the trunk's has given the tower back (where its blocks run half 3.56 a pair
// byte since each update emits its next norm over its own input - 4.06 with a second buffer for it -, 5.14 where they run
// float: LOCALFOLD_HALF_PAIR=0) - the larger stage is the peak
static const double EF2_PAIR_SLOW = 5.3;
static double ef2Peak(double held, double tower, double pair, bool parcae) {
  // (the released models' recycle: the float pair's blocks, and two pairs more - the language model's, kept for every
  // pass, and its injection - not measured, counted)
  static const bool halfPair = !getenv("LOCALFOLD_HALF_PAIR") || atoi(getenv("LOCALFOLD_HALF_PAIR")) != 0;
  return std::max(held + 1.25 * pair + 0.1e9, held - tower + (parcae ? EF2_PAIR_SLOW + 2 : halfPair ? 3.7 : EF2_PAIR_SLOW) * pair);
}
bool KEEP_TOWER = false;      // (--serve: the next job reads it again)
int foldInput(const Opts& o) {
  SamplerSettings sampler = o.sampler;
  if (M.has("meta/samplerSteps")) {          // (a bundle that states its sampler; the command's --steps still wins)
    if (!o.stepsGiven) sampler.steps = (int)M.meta("meta/samplerSteps");
    sampler.gamma0 = M.meta("meta/samplerGamma0"); sampler.gammaMin = M.meta("meta/samplerGammaMin");
    sampler.noiseScale = M.meta("meta/samplerNoiseScale"); sampler.stepScale = M.meta("meta/samplerStepScale");
    sampler.p = M.meta("meta/samplerRho"); sampler.sMin = M.meta("meta/samplerSigmaMin");
    sampler.sMax = M.meta("meta/samplerSigmaMax"); sampler.maxSigma = M.meta("meta/samplerMaxSigma");
  }
  int T = (int)M.meta("meta/tokens");
  // the residual scale is the architecture's, sqrt(layers / 36) (esm's EsmcTransformerStack; biohub's 300M runs 0.912871):
  // derived, not read - tools/export_esmc_model.py wrote 1.0 for every tower, and the 300M bundle the registry pins
  // carries it (its language-model pair 8.1e-3 from biohub's with 1.0, 2.2e-3 derived: tools/esmc/probe-esmfold2-lm-pair.py)
  const int layers = (int)M.meta("meta/layers");
  const float residualScale = (float)std::sqrt(layers / 36.0);
  if (std::fabs(M.meta("meta/residualScale", residualScale) - residualScale) > 1e-4) {
    static bool said = false;
    if (!said) fprintf(stderr, "ef2: the tower's manifest says residualScale %g; %d layers make it %.6f, which is used\n",
                       M.meta("meta/residualScale"), layers, residualScale);
    said = true;
  }
  Esmc e{(int)M.meta("meta/lm_rows"), (int)M.meta("meta/width"), (int)M.meta("meta/heads"), dimOf("c/blocks/0/fc2/weights", 0),
         layers, (int)M.meta("meta/pairChannels"), residualScale};
  printf("ESMFold2: %d tokens, %d atoms, %d tower rows\n", T, (int)M.meta("meta/atoms"), e.rows);
  int C = e.pair;
  // the fit check (core.h foldFits): what is held now - the weights and the input - and the f32 pair times a factor
  // measured at 522 and 1044 residues (metal/README.md)
  const bool parcae = M.has("f/recycle/decay");      // (the released models: the language model's pair read every pass)
  // (held: the allocations and a resident tower's mapped codes, which are no allocation - 2.46 GB at 6B int3 - and
  // which bundleWeightBytes counts, so the trunk's stage subtracts what was counted here)
  if (!foldFits("ef2", ef2Peak((double)allocated() + (double)M.residentBytes(), KEEP_TOWER ? 0. : (double)M.bundleWeightBytes("c"),
                               (double)T * T * C * 4, parcae))) return 1;
  // the language model
  float* lmZ = allocT<float>((size_t)T * T * C);
  double t0 = now();
  if (o.profile) profileStart();
  languageModel(e, Ii("lm/ids"), Ii("lm/sequence_id"), Ii("lm/token_to_row"), T, lmZ);
  mt::sync();
  if (o.profile) profileReport("language model");
  printf("language model %.1f ms\n", ms(t0));
  releaseScratch({"esmc.", "shim.", "tower."});      // (tower.: a resident tower's expanded matrices, ~140 MB at 6B)
  if (!KEEP_TOWER) M.unloadBundle("c");      // (read once: a process that folds once gives it back - 1.2 GB of ESM-C 600M)
  if (getenv("EF2_SAVE_LMZ")) {
    auto h = download(lmZ, (size_t)T * T * C); FILE* f = fopen(getenv("EF2_SAVE_LMZ"), "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f);
  }
  // the inputs embedder
  int A = (int)M.meta("meta/atoms"), Si = (int)M.meta("meta/singleInputs");
  float* sInputs = allocT<float>((size_t)T * Si);
  t0 = now();
  inputsEmbedder(T, A, sInputs, Si);
  printf("inputs embedder %.1f ms\n", ms(t0));
  releaseScratch({"atom.", "embed."});
  mem("language model");
  // z_init and the trunk
  float* zi = allocT<float>((size_t)T * T * C);
  zInit(T, C, sInputs, Si, parcae ? nullptr : lmZ, zi);
  auto save = [](const char* path, const float* x, size_t n) {      // (comparison aids: raw float32)
    auto h = download(x, n); FILE* f = fopen(path, "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f);
  };
  if (getenv("EF2_SAVE_ZINIT")) {               // z_init, and beside it (".s") s_inputs - the MSA encoder's two inputs
    save(getenv("EF2_SAVE_ZINIT"), zi, (size_t)T * T * C);
    save((std::string(getenv("EF2_SAVE_ZINIT")) + ".s").c_str(), sInputs, (size_t)T * Si);
  }
  if (M.meta("meta/msaBlocks", 0) > 0) {        // the full ESMFold2: the alignment's encoder over z_init, which it replaces
    double m0 = now();
    msaEncode(zi, sInputs, T, C, Si, o.seed);
    mt::sync();
    printf("msa encoder %.1f ms (%d rows)\n", ms(m0), std::min((int)M.meta("meta/msa_depth"), (int)M.meta("meta/msaMaxDepth")));
    if (getenv("EF2_SAVE_INJECT")) save(getenv("EF2_SAVE_INJECT"), zi, (size_t)T * T * C);
  }
  if (!parcae) { release(lmZ); lmZ = nullptr; }
  float* z = allocT<float>((size_t)T * T * C);
  t0 = now();
  if (o.profile) profileStart();
  int loops = getenv("EF2_PASSES") ? atoi(getenv("EF2_PASSES")) : M.has("meta/loops") ? (int)M.meta("meta/loops") + 1 : 4;
  foldingTrunk(T, C, zi, z, loops, lmZ, o.seed);
  if (lmZ) { release(lmZ); lmZ = nullptr; }
  mt::sync();
  if (o.profile) profileReport("trunk", 25);
  printf("trunk %.1f ms\n", ms(t0));
  mem("trunk");
  release(zi);
  releaseScratch({"ftri.", "ftr.", "frec.", "eln.", "trunk.", "zi."});     // (eln.: the emitted norm, held through the sampler before)
  if (getenv("EF2_SAVE_PAIR")) {
    auto h = download(z, (size_t)T * T * C); FILE* f = fopen(getenv("EF2_SAVE_PAIR"), "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f);
  }
  // the contact map, then the sampler
  std::vector<float> contacts = contactMap(z, T, C);
  releaseScratch({"dg."});
  if (!o.frames.empty() && !contacts.empty()) {
    std::vector<unsigned char> bytes(contacts.size());
    for (size_t k = 0; k < contacts.size(); ++k) bytes[k] = (unsigned char)std::min(255.f, std::max(0.f, std::round(contacts[k] * 255.f)));
    std::string path = o.frames + "/contacts-00-of-01.u8";
    FILE* f = fopen((path + ".tmp").c_str(), "wb");      // (its directory is the worker's: gone, it is said, not a crash)
    if (!f) fprintf(stderr, "ef2: cannot write %s - its directory is gone\n", path.c_str());
    else { fwrite(bytes.data(), 1, bytes.size(), f); fclose(f); }
    rename((path + ".tmp").c_str(), path.c_str());
  }
  t0 = now();
  Denoiser dn = makeDenoiser(T, A, z, sInputs);
  mem("denoiser built");
  int stepsRun = 0;
  std::function<void(const float*, int, int)> onStep;
  std::vector<double> frameRef; double frameCentre[3] = {};
  std::vector<float> frameMask;
  if (!o.frames.empty()) {
    frameMask = download(dn.atoms.ctx.mask, A);
    onStep = [&](const float* dd, int step, int steps) {     // each step's prediction, superposed onto the first
      std::vector<float> x = download(dd, (size_t)A * 3);
      std::vector<double> pts; std::vector<int> live;
      for (int a = 0; a < A; ++a) if (frameMask[a] > 0) { live.push_back(a); for (int k = 0; k < 3; ++k) pts.push_back(x[a * 3 + k]); }
      double c[3] = {};
      for (size_t q = 0; q < live.size(); ++q) for (int k = 0; k < 3; ++k) c[k] += pts[q * 3 + k] / live.size();
      for (size_t q = 0; q < live.size(); ++q) for (int k = 0; k < 3; ++k) pts[q * 3 + k] -= c[k];
      if (frameRef.empty()) { frameRef = pts; for (int k = 0; k < 3; ++k) frameCentre[k] = c[k]; }
      double R[9]; bestRotation(pts, frameRef, R);
      std::vector<float> moved = x;
      for (size_t q = 0; q < live.size(); ++q) {
        const double* p = &pts[q * 3];
        for (int k = 0; k < 3; ++k) moved[live[q] * 3 + k] = (float)(R[k * 3] * p[0] + R[k * 3 + 1] * p[1] + R[k * 3 + 2] * p[2] + frameCentre[k]);
      }
      char name[64]; snprintf(name, sizeof name, "/frame-%04d-%04d.pdb", step, steps);
      std::string path = o.frames + name;
      writePdb(o.dir + "/pdb.template", path + ".tmp", moved);
      rename((path + ".tmp").c_str(), path.c_str());
    };
  }
  if (o.profile) profileStart();
  std::vector<float> coords = sample(dn, sampler, o.seed, &stepsRun, onStep);
  if (o.profile) profileReport("sampler", 30);
  mem("sampler");
  freeDenoiser(dn);
  releaseScratch({"dc.", "dtr.", "tb.", "dn.", "atom.", "ada."});
  printf("sampler %.1f ms (%d steps)\n", ms(t0), stepsRun);
  if (!M.has("f/confidence/pae")) die("the weights carry no confidence head");
  t0 = now();
  float* xd = uploadNew(coords.data(), (size_t)A * 3);
  if (o.profile) profileStart();
  Confidence conf = confidenceHead(T, A, z, sInputs, Si, xd);
  if (o.profile) profileReport("confidence", 25);
  mem("confidence");
  printf("confidence %.1f ms\n", ms(t0));
  std::vector<float> bf(A);
  for (int a = 0; a < A; ++a) bf[a] = 100.f * conf.plddtAtom[a];
  printf("mean pLDDT %.2f  pTM %.4f", 100 * conf.meanPlddt, conf.ptm);
  { const int* asym = M.hostI("asym_id"); bool chains = false; for (int t = 1; t < T; ++t) chains |= asym[t] != asym[0];
    if (chains) printf("  ipTM %.4f", conf.iptm); }
  printf("\n");
  writePdb(o.dir + "/pdb.template", o.out, coords, &bf);
  if (o.out != "/dev/null") writeConfidences(o.out, T, conf, contacts);   // (as af3's: no <stem>_*.json beside /dev/null)
  printf("-> %s\n", o.out.c_str());
  mt::sync();
  for (const void* p : {(const void*)z, (const void*)sInputs, (const void*)xd}) release(p);
  releaseScratch();
  return 0;
}


bool DETACH = false;
int foldMain(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: localfold-ef2 <input dir> --fold-bundle=<dir> --esmc-bundle=<dir> [--out=fold.pdb]\n"); return 1; }
  std::string foldBundle, esmcBundle, serveDir;
  bool waitInput = false;
  Opts o; o.dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const char* a = argv[i];
    if (!strncmp(a, "--fold-bundle=", 14)) foldBundle = a + 14;
    else if (!strncmp(a, "--esmc-bundle=", 14)) esmcBundle = a + 14;
    else if (!strncmp(a, "--out=", 6)) o.out = a + 6;
    else if (!strncmp(a, "--seed=", 7)) o.seed = strtoull(a + 7, nullptr, 10);
    else if (!strncmp(a, "--steps=", 8)) { o.sampler.steps = atoi(a + 8); o.stepsGiven = true; }
    else if (samplerFlag(a, o.sampler)) {}                       // --schedule=, --update=, --no-churn (diffusion.mm)
    else if (!strncmp(a, "--frames=", 9)) o.frames = a + 9;
    else if (!strncmp(a, "--serve=", 8)) serveDir = a + 8;
    else if (!strncmp(a, "--wait-input", 12)) waitInput = true;
    else if (!strcmp(a, "--profile")) o.profile = true;
    else if (!strcmp(a, "--detach-output")) DETACH = true;
    else if (!strcmp(a, "--atom-f32")) ATOM_BF16 = false;
    else if (!strncmp(a, "--inputs-window=", 16)) { int w = atoi(a + 16); INPUTS_HALF_WINDOW = w > 0 ? w / 2 : 1 << 30; }
    else if (!strcmp(a, "--fast") || !strncmp(a, "--warm=", 7) || !strcmp(a, "--no-tower16") || !strcmp(a, "--no-sampler16")) {}
    else if (!strncmp(a, "--weights=", 10)) die("--weights (cuda/ef2's exported file) is not read here: --fold-bundle and --esmc-bundle");
    else { fprintf(stderr, "unknown flag %s\n", a); return 1; }
  }
  if (foldBundle.empty() || esmcBundle.empty()) { fprintf(stderr, "--fold-bundle=<dir> and --esmc-bundle=<dir>\n"); return 1; }
  if (o.profile && !profiling()) setenv("LOCALFOLD_PROFILE", "1", 1);
  double tStart = now();
  setSource("ef2", PORT_SOURCE);
  shareScratch({{"ftri.big", "tri"}, {"ftr.g", "tr"}, {"frec.", "rec"}}, {});
  // the weights, decoded on the device: every large tensor straight to float16, but the confidence head's own
  // projections, which stay float32
  auto asHalf = [](const std::string& n, size_t elements) {
    if (n == "c/embed/weights" || n == "f/featuriser/relPos" || n == "f/msaEncoder/embed" ||
        n.find("/outerProductMean/output") != std::string::npos)
      return false;
    if (n.rfind("f/confidence/", 0) == 0 && n.rfind("f/confidence/blocks/", 0) != 0) return false;
    if (n == "f/diffusionAtomEncoder/coordsLinear" || n == "f/diffusionAtomDecoder/outputLinear") return true;
    return elements >= 2048;
  };
  auto isSwiglu = [](const std::string& n) {
    auto ends = [&](const char* s) { size_t k = strlen(s); return n.size() >= k && !n.compare(n.size() - k, k, s); };
    return ends("/fc1/weights") || ends("/transition1") || ends("/swishWeights") || ends("/ffnUp");
  };
  M.loadBundle(foldBundle, "f", asHalf, nullptr, isSwiglu);
  prepareTrunkWeights();
  prepareAtomWeights();
  prepareDiffusionWeights();
  // the tower's block matrices kept as their codes and expanded a GEMM at a time where the tower as float16 would pass
  // 4 GB (ESM-C 6B: 12.3 GB; 600M is 1.2) - LOCALFOLD_TOWER_RESIDENT=1 forces it, =0 refuses it
  auto residentTower = [](const std::string& n, size_t) {
    static const int forced = getenv("LOCALFOLD_TOWER_RESIDENT") ? atoi(getenv("LOCALFOLD_TOWER_RESIDENT")) : -1;
    if (n.rfind("c/blocks/", 0) != 0) return false;
    if (forced >= 0) return forced > 0;
    double width = M.meta("meta/width", 0), layers = M.meta("meta/layers", 0);
    return layers * 12 * width * width * 2 > 4e9;      // (qkv 3, attn_out 1, fc1 ~5.3, fc2 ~2.7: ~12 C^2 a block)
  };
  if (getenv("EF2_STARTUP")) { mt::sync(); printf("folding model up %.0f ms (%.2f GB)\n", ms(tStart), M.weightBytes() / 1e9); }
  M.loadBundle(esmcBundle, "c", asHalf, residentTower, isSwiglu);
  if (getenv("EF2_STARTUP") && M.residentBytes()) printf("tower: %.2f GB resident as codes\n", M.residentBytes() / 1e9);
  if (getenv("EF2_STARTUP")) { mt::sync(); printf("weights up %.0f ms\n", ms(tStart)); }
  if (!serveDir.empty()) {
    KEEP_TOWER = true;
    serveJobs("ef2", serveDir, [&](const std::string& input, const std::vector<std::string>& flags) {
      Opts j = o; j.dir = input; j.out = "fold.pdb"; j.frames.clear();
      bool badFlag = false;
      for (auto& f : flags) {
        if (!f.compare(0, 6, "--out=")) j.out = f.substr(6);
        else if (!f.compare(0, 7, "--seed=")) j.seed = strtoull(f.c_str() + 7, nullptr, 10);
        else if (!f.compare(0, 8, "--steps=")) { j.sampler.steps = atoi(f.c_str() + 8); j.stepsGiven = true; }
        else if (samplerFlag(f.c_str(), j.sampler, &badFlag)) {}
        else if (!f.compare(0, 9, "--frames=")) j.frames = f.substr(9);
      }
      if (badFlag) return 1;               // (the job fails; the server stays up)
      M.loadInput(input);
      int code = foldInput(j);
      M.unloadInput();
      return code;
    });
    exit(0);
  }
  if (waitInput) {          // the featuriser writes model.idx last, by a rename
    std::string idx = o.dir + "/model.idx", failed = o.dir + "/model.failed";
    while (access(idx.c_str(), R_OK) != 0) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "ef2: the input's export failed\n"); return 1; }
      usleep(1000);
    }
  }
  M.loadInput(o.dir);
  printf("loaded in %.2f s\n", now() - tStart);
  int rc = foldInput(o);
  if (getenv("LOCALFOLD_METAL_STATS") || getenv("LOCALFOLD_PEAK")) printStats();
  if (DETACH && rc == 0) { printf("ef2: done\n"); fflush(stdout); fflush(stderr); fclose(stdout); }
  fflush(stdout);
  exit(rc);
}
}  // namespace

int main(int argc, char** argv) {
  if (getenv("LOCALFOLD_METAL_SPECS_NAME")) { printf("%s\n", specsName("ef2").c_str()); return 0; }
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) return lf::standalone::main("ef2", argc, argv, foldMain);
  return foldMain(argc, argv);
}
