// BioEmu2 (microsoft/bioemu2, checkpoint bioemu2-preview) natively on Metal: the windowed atom encoder, the PairMixer
// trunk (4 MSA layers, 48 triangle-multiplication blocks, 4 passes), the distogram, and the dual-stream denoiser under
// its EDM sampler. cuda/bioemu2 is the reading; the weights are af3-any-model's int8 blob read as the page's bundle, or that bundle.
//
//   localfold-bioemu2 --job=<job.json> | --sequence=<SEQ> [--out=fold.pdb] [...]   the weights fetched, the input
//       featurised in this process, the fold (cuda/featurise/standalone.h - every port's command line)
//   localfold-bioemu2 <input dir> (--weights=<af3am-bioemu2> | --bundle=<model-bioemu2-int5>) [--oracle=<dir>]
//       [--out=fold.pdb] [--samples=5] [--steps=200] [--seed=N] [--step-scale=S] [--passes=4] [--f32] [--profile]
// <input dir> is cuda/featurise's bioemu2-featurise output; its mode (structure, equilibrium, propagation) is the input's.
// --weights is af3-any-model's int8 blob - what fetch.h fetches, cuda/bioemu2 and the worker fold with - read as the
// bundle (weights.mm); --bundle the page's int5 bundle, which --oracle's WebGPU dumps are made from.
//
// --oracle=<dir>: tools/dump-bioemu2-webgpu.mjs's dump of webgpu/bioemu2 on the same bundle, and every stage held to it
// - EACH FED THE ORACLE'S OWN INPUT (as tools/gpu/check-bioemu2-network.js holds webgpu/bioemu2 to the colab dumps), so a
// residual is that stage's alone and a bug cannot hide behind, or be blamed on, an upstream one.
#include "bioemu2.h"
#include "standalone_api.h"
#include <chrono>
#include <cmath>
#include <map>
#include <unistd.h>

extern const char* PORT_SOURCE;

namespace {

double ms(double t0) { return (now() - t0) * 1e3; }

struct Opts { std::string dir, oracle, out = "fold.pdb", frames, sampler = "edm", potentials; uint64_t seed = 0;
              int samples = 5, steps = 200, passes = 4;
              double stepScale = -1; bool profile = false; };

double relRms(const float* a, const float* b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - b[i];
    if (!std::isfinite(d)) return INFINITY;
    num += d * d; den += (double)b[i] * b[i];
  }
  return sqrt(num / std::max(den, 1e-30));
}

std::map<std::string, double> WORST;
// ours (on the device, n values) against the oracle's "o/<name>"
double check(const char* label, const std::string& name, const float* ours, size_t n) {
  const std::string key = "o/" + name;
  if (!M.has(key)) { printf("  %-34s (not in the oracle)\n", label); return -1; }
  if (M.len(key) != n) die("the oracle's %s has %zu values, this port's %zu", key.c_str(), M.len(key), n);
  mt::sync();
  const double r = relRms((const float*)host(ours), M.hostF(key), n);
  printf("  %-34s relRMS %.3e\n", label, r);
  WORST[label] = r;
  return r;
}
float* fresh(const std::string& key) {        // a writable copy of an oracle tensor
  float* p = allocT<float>(M.len(key));
  copy(p, M.f(key), M.len(key) * 4);
  return p;
}

int oracleArm(const Opts& o, int Si) {
  const int T = D_.T, N = D_.N, Cs = dimW("sequence_embedder/s_init/weight", 0), C = dimW("sequence_embedder/z_init_1/weight", 0);
  const size_t P = (size_t)T * T;
  if ((int)M.meta("o/tokens") != T) die("the oracle is for %d tokens, the input %d", (int)M.meta("o/tokens"), T);
  printf("held to %s (webgpu/bioemu2, %s), each stage from the oracle's input:\n", o.oracle.c_str(),
         M.meta("o/half", 1) != 0 ? "its shipped arm: float16 weights" : "its float32 arm");
  double t0 = now();
  float* sIn = allocT<float>((size_t)T * Si);
  inputEmbedder(sIn, Si, nullptr);
  check("s_inputs", "s_inputs", sIn, (size_t)T * Si);
  const float* sInputs = M.f("o/s_inputs");
  float* sInit = allocT<float>((size_t)T * Cs); float* zInit = allocT<float>(P * C);
  trunkInit(sInputs, Si, sInit, zInit, nullptr);
  check("s_init", "s_init", sInit, (size_t)T * Cs);
  check("z_init", "z_init", zInit, P * C);
  for (int k = 1; k <= o.passes; ++k) {
    float* s = k == 1 ? allocT<float>((size_t)T * Cs) : fresh("o/pass" + num(k - 1) + ".single");
    float* z = k == 1 ? allocT<float>(P * C) : fresh("o/pass" + num(k - 1) + ".pair");
    trunkPass(sInputs, Si, M.f("o/s_init"), M.f("o/z_init"), s, z);
    check(("pass" + num(k) + ".single").c_str(), "pass" + num(k) + ".single", s, (size_t)T * Cs);
    check(("pass" + num(k) + ".pair").c_str(), "pass" + num(k) + ".pair", z, P * C);
    release(s); release(z);
  }
  printf("  (the trunk %.0f ms)\n", ms(t0));
  const std::string last = "o/pass" + num(o.passes);
  const int B = dimW("distogram_module/distogram/weight", 0);
  float* logits = allocT<float>(P * B);
  distogram(M.f(last + ".pair"), logits);
  check("distogram", "distogram", logits, P * B);
  Denoiser d = staticContext(M.f(last + ".single"), M.f(last + ".pair"), sInputs, Si, nullptr);
  check("ctx.z", "ctx.z", d.z, P * C);
  check("ctx.q0", "ctx.q0", d.q0, (size_t)N * 128);
  check("ctx.p", "ctx.p", d.p, (size_t)N * WK * 16);
  // the denoiser at each recorded level, on the oracle's coordinates (its real atoms; the padding after them zero)
  const int real = (int)M.meta("o/real_atoms");
  float* x = allocT<float>((size_t)N * 3); float* den = allocT<float>((size_t)N * 3);
  for (int k = 0; M.has("o/denoise" + num(k) + ".x"); ++k) {
    const std::string key = "o/denoise" + num(k);
    fill(x, 0, (size_t)N * 3 * 4);
    copy(x, M.f(key + ".x"), (size_t)real * 3 * 4);
    t0 = now();
    denoise(d, x, (float)M.meta(key + ".sigma"), den);
    mt::sync();
    const double took = ms(t0);
    char label[64]; snprintf(label, sizeof label, "denoise sigma %g", M.meta(key + ".sigma"));
    check(label, "denoise" + num(k) + ".out", den, (size_t)real * 3);
    printf("    (%.1f ms)\n", took);
  }
  freeDenoiser(d);
  double w = 0; std::string wn;
  for (auto& [k, v] : WORST) if (v > w) { w = v; wn = k; }
  printf("worst stage relRMS %.3e (%s)\n", w, wn.c_str());
  return 0;
}

int foldInput(const Opts& o) {
  double t0 = now();
  const int T = (int)M.meta("meta/tokens"), N = (int)M.meta("meta/atoms"), S = (int)M.meta("meta/msa_rows");
  D_ = {T, N, S, (int)M.meta("meta/n_valid_rows")};
  const bool equilibrium = M.meta("meta/mode") == 1;
  const int chain = (int)M.meta("meta/propagation_steps", 0);
  const int real = (int)M.meta("meta/real_atoms");
  {   // the real atoms first, the padding after them - what the oracle's and the output's atom order assume
    const int* pad = M.hostI("atom_pad_mask");
    for (int a = 0; a < N; ++a) if ((a < real) != (pad[a] != 0)) die("BioEmu2: atom %d breaks the real-atoms-first layout", a);
  }
  forgetRowMaps();
  printf("BioEmu2: %d tokens, %d atoms (%d padded), %d MSA rows, %s%s\n", T, real, N, S,
         equilibrium || chain ? "equilibrium" : "structure",
         chain ? (", propagating " + std::to_string(chain) + " step(s)").c_str() : "");
  buildTokenStarts(T, N);
  const int Si = dimW("sequence_embedder/s_init/weight", 1), Cs = dimW("sequence_embedder/s_init/weight", 0);
  const int C = dimW("sequence_embedder/z_init_1/weight", 0);
  if (!o.oracle.empty()) return oracleArm(o, Si);
  const size_t P = (size_t)T * T;
  float* sInputs = allocT<float>((size_t)T * Si);
  inputEmbedder(sInputs, Si, nullptr);
  float* sInit = allocT<float>((size_t)T * Cs); float* zInit = allocT<float>(P * C);
  trunkInit(sInputs, Si, sInit, zInit, nullptr);
  float* s = allocT<float>((size_t)T * Cs); float* z = allocT<float>(P * C);
  mt::sync();
  printf("inputs %.1f ms\n", ms(t0)); t0 = now();
  if (o.profile) profileStart();
  for (int k = 0; k < o.passes; ++k) trunkPass(sInputs, Si, sInit, zInit, s, z);
  mt::sync();
  printf("trunk %.1f ms (%d passes)\n", ms(t0), o.passes); t0 = now();
  if (o.profile) profileReport("trunk", 2000);
  release(sInit); release(zInit);
  const int B = dimW("distogram_module/distogram/weight", 0);
  float* logits = scratch<float>("be2.dg.logits", P * B);
  distogram(z, logits);
  const std::vector<float> contacts = contactProbs(logits, T, B);
  if (!o.frames.empty()) {      // (the page's live preview, python/localfold/worker.py's on_file: the trunk's contact map)
    std::vector<unsigned char> bytes(contacts.size());
    for (size_t k = 0; k < contacts.size(); ++k) bytes[k] = (unsigned char)std::min(255.f, std::max(0.f, std::round(contacts[k] * 255.f)));
    writeWhole(o.frames + "/contacts-00-of-01.u8", bytes.data(), bytes.size());
  }
  releaseScratch();      // (the trunk's buffers are the denoiser's room)
  Denoiser d = staticContext(s, z, sInputs, Si, nullptr);
  mt::sync();
  printf("static context %.1f ms\n", ms(t0));
  if (o.samples > 0) {
    t0 = now();
    if (o.profile) profileStart();
    SamplerSettings ss; ss.steps = o.steps;
    ss.stepScale = o.stepScale > 0 ? o.stepScale : equilibrium || chain ? 1.0 : 1.5;
    std::vector<std::vector<float>> out;
    if (o.sampler != "edm") {
      if (chain) { fprintf(stderr, "Error: BioEmu2 propagation runs unsteered here; use the edm sampler to propagate\n"); return 1; }
      try {
        out = steeredFold(d, ss, o.sampler, o.potentials, o.seed, o.samples);
      } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        freeDenoiser(d);
        return 1;
      }
    } else if (chain) {
      std::vector<float> x0(M.hostF("x0"), M.hostF("x0") + (size_t)N * 3);
      auto frames = propagate(d, ss, o.seed, o.samples, chain, x0);
      for (auto& tr : frames) for (auto& f : tr) out.push_back(f);    // (trajectory i, step k: sample i * chain + k)
    } else {
      // the page's live preview: each step's estimate of the first sample, superposed onto the first step's (as
      // metal/ef2 writes them: frame-SSSS-NNNN.pdb, written whole then renamed)
      StepHook onStep;
      std::vector<double> ref; double centre[3] = {};
      std::vector<int> live;
      if (!o.frames.empty()) {
        const int* m = M.hostI("atom_mask");
        for (int a = 0; a < N; ++a) if (m[a]) live.push_back(a);
        onStep = [&](int step, int steps, const std::vector<float>& x) {
          std::vector<double> pts;
          for (int a : live) for (int k = 0; k < 3; ++k) pts.push_back(x[a * 3 + k]);
          double c[3] = {};
          for (size_t q = 0; q < live.size(); ++q) for (int k = 0; k < 3; ++k) c[k] += pts[q * 3 + k] / live.size();
          for (size_t q = 0; q < live.size(); ++q) for (int k = 0; k < 3; ++k) pts[q * 3 + k] -= c[k];
          if (ref.empty()) { ref = pts; for (int k = 0; k < 3; ++k) centre[k] = c[k]; }
          double R[9]; bestRotation(pts, ref, R);
          std::vector<float> moved = x;
          for (size_t q = 0; q < live.size(); ++q) {
            const double* p = &pts[q * 3];
            for (int k = 0; k < 3; ++k) moved[live[q] * 3 + k] = (float)(R[k * 3] * p[0] + R[k * 3 + 1] * p[1] + R[k * 3 + 2] * p[2] + centre[k]);
          }
          char name[64]; snprintf(name, sizeof name, "/frame-%04d-%04d.pdb", step, steps);
          writePdb(o.dir + "/pdb.template", o.frames + name, moved);
        };
      }
      out = sample(d, ss, o.seed, o.samples, {}, true, onStep);
    }
    printf("sampler %.1f ms (%d sample(s), %d steps)\n", ms(t0), (int)out.size(), ss.steps);
    if (o.profile) profileReport("sampler", 2000);
    const std::string stem = o.out.size() > 4 && o.out.substr(o.out.size() - 4) == ".pdb" ? o.out.substr(0, o.out.size() - 4) : o.out;
    for (size_t k = 0; k < out.size(); ++k)
      writePdb(o.dir + "/pdb.template", k == 0 ? o.out : stem + "_sample-" + std::to_string(k) + ".pdb", out[k]);
    writeConfidences(o.dir, o.out, T, contacts);
    printf("-> %s%s\n", o.out.c_str(), out.size() > 1 ? (" (+" + std::to_string(out.size() - 1) + " more samples)").c_str() : "");
  }
  freeDenoiser(d);
  for (float* p : {sInputs, s, z}) release(p);
  releaseScratch();
  return 0;
}

// a fold's own flags, the command line's and a served job's alike (false: not one of them)
bool flagInto(const std::string& f, Opts& o) {
  if (!f.compare(0, 6, "--out=")) o.out = f.substr(6);
  else if (!f.compare(0, 7, "--seed=")) o.seed = strtoull(f.c_str() + 7, nullptr, 10);
  else if (!f.compare(0, 8, "--steps=")) o.steps = atoi(f.c_str() + 8);
  else if (!f.compare(0, 10, "--samples=")) o.samples = atoi(f.c_str() + 10);
  else if (!f.compare(0, 13, "--step-scale=")) o.stepScale = atof(f.c_str() + 13);
  else if (!f.compare(0, 9, "--passes=")) o.passes = atoi(f.c_str() + 9);
  else if (!f.compare(0, 10, "--sampler=")) o.sampler = f.substr(10);
  else if (!f.compare(0, 13, "--potentials=")) o.potentials = f.substr(13);
  else if (!f.compare(0, 9, "--frames=")) o.frames = f.substr(9);
  else return false;
  return true;
}
// a sampler this port has, said by name otherwise (refused, not folded as though it had not been asked)
bool samplerRefused(const Opts& o) {
  if (o.sampler == "edm" || o.sampler == "edm_gd" || o.sampler == "edm_fk" || o.sampler == "edm_fkc") return false;
  fprintf(stderr, "Error: BioEmu2 sampler %s: edm, edm_gd, edm_fk or edm_fkc\n", o.sampler.c_str());
  return true;
}

}  // namespace

int foldMain(int argc, char** argv);

int main(int argc, char** argv) {
  if (getenv("LOCALFOLD_METAL_SPECS_NAME")) { printf("%s\n", specsName("bioemu2").c_str()); return 0; }
  if (argc == 2 && !strncmp(argv[1], "--list-weights=", 15)) {     // (every tensor of a weights directory, and its shape)
    for (auto& [n, shape] : Model::bundleShapes(argv[1] + 15)) {
      printf("%s [", n.c_str());
      for (size_t k = 0; k < shape.size(); ++k) printf("%s%lld", k ? ", " : "", shape[k]);
      printf("]\n");
    }
    return 0;
  }
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) return lf::standalone::main("bioemu2", argc, argv, foldMain);
  return foldMain(argc, argv);
}

int foldMain(int argc, char** argv) {
  if (argc < 2 || !strncmp(argv[1], "--", 2)) {
    fprintf(stderr, "usage: localfold-bioemu2 <input dir> --bundle=<model-bioemu2-int5> [--oracle=<dir>] [--out=fold.pdb]"
                    " [--samples=5] [--steps=200] [--seed=N] [--step-scale=S] [--passes=4] [--f32] [--profile]\n");
    return 1;
  }
  Opts o; o.dir = argv[1];
  std::string bundle, blob, compareWith, serveDir;
  bool waitInput = false;
  for (int i = 2; i < argc; ++i) {
    const char* a = argv[i];
    if (!strncmp(a, "--bundle=", 9)) bundle = a + 9;
    else if (!strncmp(a, "--weights=", 10)) blob = a + 10;
    else if (!strncmp(a, "--compare-weights=", 18)) compareWith = a + 18;
    else if (!strcmp(a, "--fast")) {}                        // (the standalone path's; the fast arm is the default here)
    else if (!strncmp(a, "--wait-input", 12)) waitInput = true;   // (the featuriser still writing the input: standalone.h)
    else if (!strncmp(a, "--serve=", 8)) serveDir = a + 8;
    else if (!strncmp(a, "--oracle=", 9)) o.oracle = a + 9;
    else if (flagInto(a, o)) {}
    else if (!strcmp(a, "--f32")) HALF = false;            // (the float32 reference arm: what --oracle's float32 dump holds)
    else if (!strcmp(a, "--profile")) o.profile = true;     // (every labelled dispatch timed: proportions, not speed)
    else { fprintf(stderr, "unknown flag %s\n", a); return 1; }
  }
  if (bundle.empty() == blob.empty()) {
    fprintf(stderr, "one of --weights=<af3-any-model's bioemu2 blob directory> or --bundle=<the int5 bundle's directory>\n");
    return 1;
  }
  if (samplerRefused(o)) return 1;
  if (o.steps < 2) { fprintf(stderr, "BioEmu2 samples with at least 2 steps, got %d\n", o.steps); return 1; }
  if (o.profile && !profiling()) setenv("LOCALFOLD_PROFILE", "1", 1);
  const double tStart = now();
  setSource("bioemu2", PORT_SOURCE);
  // the weights decoded on the device: the large matrices straight to float16 on the fast arm (a vector, a table, a norm
  // stays float32; anything an elementwise kernel reads as float gets a float32 copy on demand), every tensor float32
  // on the reference arm
  auto asHalf = [](const std::string&, size_t elements) { return HALF && elements >= 65536; };
  if (!bundle.empty()) M.loadBundle(bundle, "w", asHalf);
  else loadBlobAsBundle(blob, asHalf);
  mt::sync();
  printf("weights up in %.2f s (%.2f GB)\n", now() - tStart, M.weightBytes() / 1e9);
  if (!serveDir.empty()) {
    // the resident server (metal/core/host.cpp's serveJobs, python/localfold/worker.py's): the weights stay on the GPU,
    // each job its input directory and its flags
    serveJobs("bioemu2", serveDir, [&](const std::string& input, const std::vector<std::string>& flags) {
      Opts j = o; j.dir = input; j.out = "fold.pdb"; j.oracle.clear(); j.frames.clear();
      for (auto& f : flags)
        if (!flagInto(f, j)) { fprintf(stderr, "Error: a BioEmu2 job's flag %s is not one this port reads\n", f.c_str()); return 1; }
      if (samplerRefused(j)) return 1;
      if (j.steps < 2) { fprintf(stderr, "Error: BioEmu2 samples with at least 2 steps, got %d\n", j.steps); return 1; }
      M.loadInput(input);
      const int code = foldInput(j);
      mt::sync();
      M.unloadInput();
      return code;
    });
    return 0;
  }
  if (waitInput) {          // the featuriser writes model.idx last, by a rename
    const std::string idx = o.dir + "/model.idx", failed = o.dir + "/model.failed";
    while (access(idx.c_str(), R_OK) != 0) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "bioemu2: the input's export failed\n"); return 1; }
      usleep(1000);
    }
  }
  if (!compareWith.empty()) {
    // every tensor these weights give the port against another set's (the blob walked into the bundle's layout against
    // the int5 bundle itself): a right layout differs by the two quantisations - a few percent - and a wrong one (a
    // transpose missed, an interleave or the incoming triangle's swap undone) is uncorrelated, relRMS ~1.4
    M.loadBundle(compareWith, "v");
    mt::sync();
    double worst = 0; std::string worstName; size_t n = 0, bad = 0;
    for (auto& [name, shape] : Model::bundleShapes(compareWith)) {
      const std::string w = "w/" + name, v = "v/" + name;
      if (!M.has(w)) { printf("  MISSING %s\n", name.c_str()); ++bad; continue; }
      if (M.len(w) != M.len(v)) { printf("  LENGTH %s %zu against %zu\n", name.c_str(), M.len(w), M.len(v)); ++bad; continue; }
      const double r = relRms(M.hostF(w), M.hostF(v), M.len(v));
      if (r > 0.1) { printf("  %-100s %.3e\n", name.c_str(), r); ++bad; }
      if (r > worst) { worst = r; worstName = name; }
      ++n;
    }
    printf("%zu tensors, %zu past 0.1, worst %.3e (%s)\n", n, bad, worst, worstName.c_str());
    // the reshaped ones by name, and a NEGATIVE CONTROL - two different tensors of one shape - that must read ~1.4, or
    // this comparison could not tell a wrong layout from a right one
    for (const char* t : {"msa/0/triangleMultiplicationOutgoing/projection", "msa/0/triangleMultiplicationIncoming/projection",
                          "pairmixer/47/triangleMultiplicationIncoming/gate", "pairmixer/5/triangleMultiplicationOutgoing/outputProjection",
                          "pairmixer/5/pairTransition/transition1", "pairmixer/5/pairTransition/transition2"})
      printf("  %-60s %.3e\n", t, relRms(M.hostF(std::string("w/") + t), M.hostF(std::string("v/") + t), M.len(std::string("v/") + t)));
    const std::string a = "msa/0/triangleMultiplicationOutgoing/projection", b = "msa/0/triangleMultiplicationIncoming/projection";
    printf("  negative control (%s against the bundle's incoming one) %.3e\n", a.c_str(),
           relRms(M.hostF("w/" + a), M.hostF("v/" + b), M.len("v/" + b)));
    return bad ? 1 : 0;
  }
  M.loadInput(o.dir);
  if (!o.oracle.empty()) M.loadInput(o.oracle);
  const int rc = foldInput(o);
  fflush(stdout);
  return rc;
}
