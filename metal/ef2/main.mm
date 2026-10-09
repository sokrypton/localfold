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

int foldInput(const Opts& o) {
  SamplerSettings sampler = o.sampler;
  if (M.has("meta/samplerSteps")) {          // (a bundle that states its sampler; the command's --steps still wins)
    if (!o.stepsGiven) sampler.steps = (int)M.meta("meta/samplerSteps");
    sampler.gamma0 = M.meta("meta/samplerGamma0"); sampler.gammaMin = M.meta("meta/samplerGammaMin");
    sampler.noiseScale = M.meta("meta/samplerNoiseScale"); sampler.stepScale = M.meta("meta/samplerStepScale");
    sampler.p = M.meta("meta/samplerRho"); sampler.sMin = M.meta("meta/samplerSigmaMin");
    sampler.sMax = M.meta("meta/samplerSigmaMax"); sampler.maxSigma = M.meta("meta/samplerMaxSigma");
  }
  if (M.meta("meta/msaBlocks", 0) > 0) die("this model's MSA encoder (the released ESMFold2) is not in the native port yet");
  int T = (int)M.meta("meta/tokens");
  Esmc e{(int)M.meta("meta/lm_rows"), (int)M.meta("meta/width"), (int)M.meta("meta/heads"), dimOf("c/blocks/0/fc2/weights", 0),
         (int)M.meta("meta/layers"), (int)M.meta("meta/pairChannels"), (float)M.meta("meta/residualScale", 1.0)};
  printf("ESMFold2: %d tokens, %d atoms, %d tower rows\n", T, (int)M.meta("meta/atoms"), e.rows);
  int C = e.pair;
  // the language model
  float* lmZ = allocT<float>((size_t)T * T * C);
  double t0 = now();
  if (o.profile) profileStart();
  languageModel(e, Ii("lm/ids"), Ii("lm/sequence_id"), Ii("lm/token_to_row"), T, lmZ);
  mt::sync();
  if (o.profile) profileReport("language model");
  printf("language model %.1f ms\n", ms(t0));
  releaseScratch({"esmc.", "shim."});
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
  zInit(T, C, sInputs, Si, lmZ, zi);
  if (getenv("EF2_SAVE_ZINIT")) {
    auto h = download(zi, (size_t)T * T * C); FILE* f = fopen(getenv("EF2_SAVE_ZINIT"), "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f);
  }
  release(lmZ);
  float* z = allocT<float>((size_t)T * T * C);
  t0 = now();
  if (o.profile) profileStart();
  int loops = getenv("EF2_PASSES") ? atoi(getenv("EF2_PASSES")) : M.has("meta/loops") ? (int)M.meta("meta/loops") + 1 : 4;
  foldingTrunk(T, C, zi, z, loops);
  mt::sync();
  if (o.profile) profileReport("trunk", 25);
  printf("trunk %.1f ms\n", ms(t0));
  mem("trunk");
  release(zi);
  releaseScratch({"ftri.", "ftr.", "trunk.", "zi."});
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
    FILE* f = fopen((path + ".tmp").c_str(), "wb"); fwrite(bytes.data(), 1, bytes.size(), f); fclose(f);
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
  writeConfidences(o.out, T, conf, contacts);
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
  // the weights, decoded on the device: every large tensor straight to float16, but the confidence head's own
  // projections, which stay float32
  auto asHalf = [](const std::string& n, size_t elements) {
    if (n.rfind("f/confidence/", 0) == 0 && n.rfind("f/confidence/blocks/", 0) != 0) return false;
    return elements >= 65536;
  };
  M.loadBundle(foldBundle, "f", asHalf);
  M.loadBundle(esmcBundle, "c", asHalf);
  if (getenv("EF2_STARTUP")) { mt::sync(); printf("weights up %.0f ms\n", ms(tStart)); }
  if (!serveDir.empty()) {
    serveJobs("ef2", serveDir, [&](const std::string& input, const std::vector<std::string>& flags) {
      Opts j = o; j.dir = input; j.out = "fold.pdb"; j.frames.clear();
      for (auto& f : flags) {
        if (!f.compare(0, 6, "--out=")) j.out = f.substr(6);
        else if (!f.compare(0, 7, "--seed=")) j.seed = strtoull(f.c_str() + 7, nullptr, 10);
        else if (!f.compare(0, 8, "--steps=")) { j.sampler.steps = atoi(f.c_str() + 8); j.stepsGiven = true; }
        else if (!f.compare(0, 9, "--frames=")) j.frames = f.substr(9);
      }
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
  if (getenv("LOCALFOLD_METAL_STATS")) printStats();
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
