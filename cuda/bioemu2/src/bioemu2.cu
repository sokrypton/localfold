// BioEmu2 (microsoft/bioemu2, checkpoint bioemu2-preview) in CUDA: the windowed atom encoder, the PairMixer trunk
// (4 MSA layers, 48 triangle-multiplication blocks, 4 passes), the distogram, and the dual-stream denoiser under its
// EDM sampler - each stage held to af3-any-model's port (network/bioemu2.py, itself held to the vendor at 0.000 A)
// on the same input (cuda/bioemu2/oracle.py).
//
//   localfold-bioemu2 <input dir> --weights=<dir with bioemu2*.bin.zst> [--oracle=<dir>] [--out=fold.pdb]
//       [--samples=5] [--steps=200] [--seed=N] [--fast] [--serve=<dir>]
// <input dir> is cuda/bioemu2/export_input.mjs's; its mode (structure, equilibrium, propagation) is the input's.
#include "sampler.cuh"
#include <chrono>

using namespace be2;

struct Opts { std::string dir, oracle, out = "fold.pdb"; uint64_t seed = 0; int samples = 5; int steps = 200;
              bool stepsGiven = false; double stepScale = -1; int passes = 4; std::string frames; };

static inline double ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static std::string stemOf(const std::string& pdb) {
  return pdb.size() > 4 && pdb.substr(pdb.size() - 4) == ".pdb" ? pdb.substr(0, pdb.size() - 4) : pdb;
}

// <stem>_confidences.json and <stem>_summary_confidences.json, AlphaFold 3's layout as the other ports write it:
// the contact probabilities (the distogram's P(d < 8 A)), the token ids, in the job's token order - and NO pLDDT,
// PAE or pTM, because the checkpoint has no confidence head; they are null, not invented
static void writeConfidences(const std::string& dir, const std::string& pdb, int T, const std::vector<float>& contacts) {
  std::ifstream in(dir + "/confidence.template.json");
  if (!in) { fprintf(stderr, "no %s/confidence.template.json (export_input.mjs writes it)\n", dir.c_str()); exit(1); }
  Json j = Json::parse(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
  std::vector<int> order; for (auto& v : j.get("tokens")->arr) order.push_back((int)v.num);
  const int n = (int)order.size();
  if (n != T) { fprintf(stderr, "confidence.template.json names %d tokens of %d\n", n, T); exit(1); }
  std::vector<float> c((size_t)n * n);
  for (int i = 0; i < n; ++i) for (int k = 0; k < n; ++k) c[(size_t)i * n + k] = contacts[(size_t)order[i] * T + order[k]];
  const std::string stem = stemOf(pdb);
  FILE* f = fopen((stem + "_confidences.json").c_str(), "w");
  std::string s = "{\"contact_probs\": ";
  appendMatrix2(s, c.data(), n);
  fwrite(s.data(), 1, s.size(), f);
  fprintf(f, ",\n \"pae\": null,\n \"token_plddts\": null,\n \"atom_plddts\": null,\n \"token_chain_ids\": [");
  auto& ids = j.get("token_chain_ids")->arr; auto& res = j.get("token_res_ids")->arr;
  for (int i = 0; i < n; ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", ids[i].str.c_str());
  fprintf(f, "],\n \"token_res_ids\": [");
  for (int i = 0; i < n; ++i) fprintf(f, "%s%d", i ? ", " : "", (int)res[i].num);
  fprintf(f, "],\n \"note\": \"BioEmu2 has no confidence head: pLDDT, PAE and pTM are not predicted\"}\n");
  fclose(f);
  f = fopen((stem + "_summary_confidences.json").c_str(), "w");
  fprintf(f, "{\"ptm\": null, \"iptm\": null, \"mean_plddt\": null, \"ranking_score\": null,"
             " \"note\": \"BioEmu2 has no confidence head\"}\n");
  fclose(f);
}

// the contact map: the softmax mass of the first bins of each pair's 64 (Boltz's, over 2-22 A), up to 8 A
__global__ void contactProbsK(const float* logits, float* out, size_t pairs, int bins, int near) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= pairs) return;
  const float* l = logits + ij * bins;
  float mx = -INFINITY; for (int b = 0; b < bins; ++b) mx = fmaxf(mx, l[b]);
  float tot = 0, c = 0;
  for (int b = 0; b < bins; ++b) { float e = expf(l[b] - mx); tot += e; if (b < near) c += e; }
  out[ij] = c / tot;
}

static int foldInput(const Opts& o) {
  auto t0 = std::chrono::steady_clock::now();
  const int T = (int)M.meta("meta/tokens"), N = (int)M.meta("meta/atoms"), S = (int)M.meta("meta/msa_rows");
  D_ = {T, N, S, (int)M.meta("meta/n_valid_rows")};
  const bool check = !o.oracle.empty();
  const bool equilibrium = M.meta("meta/mode") == 1;
  const int chain = (int)M.meta("meta/propagation_steps", 0);
  printf("BioEmu2: %d tokens, %d atoms (%d padded), %d MSA rows, %s%s\n", T, (int)M.meta("meta/real_atoms"), N, S,
         equilibrium || chain ? "equilibrium" : "structure",
         chain ? (", propagating " + std::to_string(chain) + " step(s)").c_str() : "");
  buildTokenStarts(T, N);
  std::map<std::string, double> worst;
  auto tap = [&](const char* name, const float* d, size_t n) {
    if (!check) return;
    double r = checkOracle(name, d, n, std::string("o/") + name);
    if (r >= 0) worst[name] = r;
  };
  const int Si = 455, Cs = 384, C = 256;
  size_t Pn = (size_t)T * T;
  float* sInputs = dalloc((size_t)T * Si);
  inputEmbedder(sInputs, Si, tap);
  float* sInit = dalloc((size_t)T * Cs); float* zInit = dalloc(Pn * C);
  trunkInit(sInputs, Si, sInit, zInit, tap);
  float* s = dalloc((size_t)T * Cs); float* z = dalloc(Pn * C);
  CK(cudaMemsetAsync(s, 0, (size_t)T * Cs * 4, STREAM)); CK(cudaMemsetAsync(z, 0, Pn * C * 4, STREAM));
  CK(cudaStreamSynchronize(STREAM));
  printf("inputs %.1f ms\n", ms(t0)); t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < o.passes; ++k) {
    trunkPass(sInputs, Si, sInit, zInit, s, z);
    std::string ks = "pass" + std::to_string(k + 1);
    tap((ks + ".single").c_str(), s, (size_t)T * Cs);
    tap((ks + ".pair").c_str(), z, Pn * C);
  }
  CK(cudaStreamSynchronize(STREAM));
  printf("trunk %.1f ms (%d passes)\n", ms(t0), o.passes); t0 = std::chrono::steady_clock::now();
  CK(cudaFree(sInit)); CK(cudaFree(zInit));
  const int B = dimP("distogram_module/distogram/weight", 0);
  std::vector<float> contacts;
  {
    float* logits = scratch<float>("be2.dg.logits", Pn * B);
    distogram(z, logits);
    tap("distogram", logits, Pn * B);
    // Boltz's 64 bins over 2-22 A: the bins whose lower edges (linspace(2, 22, 63)) are at most 8 A
    int near = 0; for (int e = 0; e < B - 1; ++e) if (2.0 + 20.0 * e / (B - 2) <= 8.0) ++near;
    float* probs = scratch<float>("be2.dg.contacts", Pn);
    contactProbsK<<<blocks(Pn), 256, 0, STREAM>>>(logits, probs, Pn, B, near);
    contacts = download(probs, Pn);
  }
  Denoiser d = staticContext(s, z, sInputs, Si, tap);
  CK(cudaStreamSynchronize(STREAM));
  printf("static context %.1f ms\n", ms(t0));
  if (check && M.has("o/denoise0.x")) {
    // each recorded level, from OUR trunk and context, then from the reference's own (the denoiser alone)
    float* den = dalloc((size_t)N * 3);
    auto levels = [&](const Denoiser& dn, const char* tag) {
      for (int k = 0; M.has("o/denoise" + std::to_string(k) + ".x"); ++k) {
        std::string key = "o/denoise" + std::to_string(k);
        float* x = upload(M.f(key + ".x"), (size_t)N * 3);
        denoise(dn, x, (float)M.meta(key + ".sigma"), den);
        char label[64]; snprintf(label, sizeof label, "denoise sigma %g (%s)", M.meta(key + ".sigma"), tag);
        double r = checkOracle(label, den, (size_t)N * 3, key + ".out");
        worst[label] = r;
        CK(cudaFree(x));
      }
    };
    levels(d, "own trunk");
    float* so = upload(M.f("o/pass" + std::to_string(o.passes) + ".single"), (size_t)T * Cs);
    float* zo = upload(M.f("o/pass" + std::to_string(o.passes) + ".pair"), Pn * C);
    Denoiser dr = staticContext(so, zo, sInputs, Si, nullptr);
    levels(dr, "reference trunk");
    if (M.has("o/denoise_dyn.x")) {
      float* cc = upload(M.f("o/denoise_dyn.coords_cond"), (size_t)N * 3);
      setDynamics(dr, (float)M.meta("o/denoise_dyn.dt"), cc);
      float* x = upload(M.f("o/denoise_dyn.x"), (size_t)N * 3);
      denoise(dr, x, (float)M.meta("o/denoise_dyn.sigma"), den);
      double r = checkOracle("denoise, propagation (reference trunk)", den, (size_t)N * 3, "o/denoise_dyn.out");
      worst["denoise_dyn"] = r;
      CK(cudaFree(x)); CK(cudaFree(cc));
    }
    freeDenoiser(dr); CK(cudaFree(so)); CK(cudaFree(zo)); CK(cudaFree(den));
    double w = 0; for (auto& [k, v] : worst) w = std::max(w, v);
    printf("worst stage relRMS %.3e\n", w);
  }
  if (o.samples > 0) {
    t0 = std::chrono::steady_clock::now();
    SamplerSettings ss; ss.steps = o.steps;
    ss.stepScale = o.stepScale > 0 ? o.stepScale : equilibrium || chain ? 1.0 : 1.5;
    std::vector<std::vector<float>> out;
    if (chain) {
      std::vector<float> x0(M.f("x0"), M.f("x0") + (size_t)N * 3);
      auto frames = propagate(d, ss, o.seed, o.samples, chain, x0);
      for (auto& tr : frames) for (auto& f : tr) out.push_back(f);    // (trajectory i, step k: sample i * chain + k)
    } else {
      out = sample(d, ss, o.seed, o.samples);
    }
    CK(cudaStreamSynchronize(STREAM));
    printf("sampler %.1f ms (%d sample(s), %d steps)\n", ms(t0), (int)out.size(), ss.steps);
    const std::string stem = stemOf(o.out);
    for (size_t k = 0; k < out.size(); ++k) {
      std::string path = k == 0 ? o.out : stem + "_sample-" + std::to_string(k) + ".pdb";
      writePdb(o.dir + "/pdb.template", path, out[k]);
    }
    writeConfidences(o.dir, o.out, T, contacts);
    printf("-> %s%s\n", o.out.c_str(), out.size() > 1 ? (" (+" + std::to_string(out.size() - 1) + " more samples)").c_str() : "");
  }
  freeDenoiser(d);
  for (float* p : {sInputs, s, z}) CK(cudaFree(p));
  releaseScratch();
  return 0;
}

static bool flagInto(const std::string& f, Opts& o) {
  if (!f.compare(0, 6, "--out=")) o.out = f.substr(6);
  else if (!f.compare(0, 7, "--seed=")) o.seed = strtoull(f.c_str() + 7, nullptr, 10);
  else if (!f.compare(0, 8, "--steps=")) { o.steps = atoi(f.c_str() + 8); o.stepsGiven = true; }
  else if (!f.compare(0, 10, "--samples=")) o.samples = atoi(f.c_str() + 10);
  else if (!f.compare(0, 13, "--step-scale=")) o.stepScale = atof(f.c_str() + 13);
  else if (!f.compare(0, 9, "--passes=")) o.passes = atoi(f.c_str() + 9);
  else return false;
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2 || !strncmp(argv[1], "--", 2)) {
    fprintf(stderr, "usage: localfold-bioemu2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--samples=5]"
                    " [--steps=200] [--seed=N] [--step-scale=S] [--fast] [--serve=<dir>]\n");
    return 1;
  }
  Opts o; o.dir = argv[1];
  std::string weights, serveDir, compareWith;
  bool fast = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (!a.compare(0, 10, "--weights=")) weights = a.substr(10);
    else if (!a.compare(0, 9, "--oracle=")) o.oracle = a.substr(9);
    else if (a == "--fast") fast = true;
    else if (!a.compare(0, 18, "--compare-weights=")) compareWith = a.substr(18);
    else if (!a.compare(0, 8, "--serve=")) serveDir = a.substr(8);
    else if (!flagInto(a, o)) { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (weights.empty()) { fprintf(stderr, "--weights=<dir holding bioemu2*.bin.zst>\n"); return 1; }
  if (o.steps < 2) { fprintf(stderr, "BioEmu2 samples with at least 2 steps, got %d\n", o.steps); return 1; }
  auto tStart = std::chrono::steady_clock::now();
  M.loadBundle(weights, "w");
  for (int sgi = 0; sgi < (int)M.segs.size(); ++sgi) M.upload(sgi);
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  CB(cublasSetMathMode(H, fast ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
  printf("weights up in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count());
  if (!compareWith.empty()) {
    // every tensor of these weights against another blob's (the int8 decode against float32): relRMS per tensor
    M.loadBundle(compareWith, "f");
    double worstR = 0; std::string worstName; size_t n = 0;
    for (auto& [k, e] : M.index) {
      if (e.kind == 'm' || k.compare(0, 2, "w/")) continue;
      std::string f = "f/" + k.substr(2);
      if (!M.has(f)) continue;
      auto a = download(W(k), M.len(k)); auto b = download(W(f), M.len(f));
      double r = relRms(a.data(), b.data(), a.size());
      printf("  %-110s %.3e\n", k.c_str() + SCOPE.size(), r);
      if (r > worstR) { worstR = r; worstName = k; }
      ++n;
    }
    printf("%zu tensors, worst %.3e (%s)\n", n, worstR, worstName.c_str());
    finish(0);
  }
  const int weightSegs = (int)M.segs.size();
  if (!serveDir.empty()) {
    serveJobs(serveDir, "bioemu2", [&](const std::string& input, const std::vector<std::string>& flags) {
      Opts j = o; j.dir = input; j.out = "fold.pdb"; j.oracle.clear();
      for (auto& f : flags) flagInto(f, j);
      int seg = (int)M.segs.size();
      M.load(input);
      int code = foldInput(j);
      CK(cudaStreamSynchronize(STREAM));
      forgetEntries(M.unload(seg));
      return code;
    });
    finish(0);
  }
  M.load(o.dir);
  if (!o.oracle.empty()) M.load(o.oracle);
  for (int sgi = weightSegs; sgi < (int)M.segs.size(); ++sgi) M.upload(sgi);
  finish(foldInput(o));
}
