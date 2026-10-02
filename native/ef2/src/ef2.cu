// ESMFold2 (esmfold2-fast-600m) in CUDA: ESM-C, the shim, the inputs embedder, the 24-block pair
// trunk (AF3's pair track, minus its grid attention and single track), the distogram and the
// diffusion sampler - each stage held to biohub's own forward on the same input (oracle.py).
//
//   ef2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--fast]
#include "esmc.cuh"
#include "atoms.cuh"
#include "trunk.cuh"
#include "sampler.cuh"
#include "confidence.cuh"
#include "../../af3/src/profile.cuh"

__global__ void gatherStateK(const float* x, const int* tokenToRow, float* out, int T, int states, int k, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int token = (int)(t / C), c = (int)(t % C), r = tokenToRow[token];
  out[((size_t)token * states + k) * C + c] = r < 0 ? 0.f : x[(size_t)r * C + c];
}

// a synthetic input of T tokens and A atoms (what export_input.mjs writes, its values arbitrary but
// valid: one chain of alanines, A / T atoms a token), in /dev/shm, for --warm
static std::string writeWarmInput(int T, int A) {
  std::string dir = "/dev/shm/ef2-warm-" + std::to_string(getpid());
  mkdir(dir.c_str(), 0755);
  FILE* bin = fopen((dir + "/model.bin").c_str(), "wb");
  std::string idx; size_t offset = 0;
  auto put = [&](char kind, const std::string& name, const std::vector<float>& f, const std::vector<int>& i) {
    size_t n = kind == 'i' ? i.size() : f.size();
    if (kind == 'i') fwrite(i.data(), 4, n, bin); else fwrite(f.data(), 4, n, bin);
    idx += std::string(1, kind) + " " + name + " " + std::to_string(offset) + " " + std::to_string(n) + "\n";
    offset += n;
  };
  auto ints = [&](const std::string& name, std::vector<int> v) { put('i', name, {}, v); };
  auto flts = [&](const std::string& name, std::vector<float> v) { put('t', name, v, {}); };
  std::vector<int> seq(T), zeros(T, 0), restype(T, 2), ids(T, 5);
  for (int t = 0; t < T; ++t) seq[t] = t;
  int K = 33;
  ints("residue_index", seq); ints("token_index", seq); ints("asym_id", zeros); ints("entity_id", zeros); ints("sym_id", zeros);
  ints("mol_type", zeros); ints("res_type", restype); ints("input_ids", ids);
  std::vector<int> a2t(A), dg(T), uid(A), el(A, 6), names(A * 4, 0);
  for (int a = 0; a < A; ++a) { a2t[a] = std::min(T - 1, (int)((long)a * T / A)); uid[a] = a2t[a]; names[a * 4] = 35; }
  for (int a = A - 1; a >= 0; --a) dg[a2t[a]] = a;
  ints("distogram_atom_idx", dg);
  std::vector<float> aat((size_t)T * K, 0.f), pos((size_t)A * 3), ones(A, 1.f), zA(A, 0.f), zT(T, 0.f);
  for (int t = 0; t < T; ++t) aat[(size_t)t * K + 2] = 1.f;
  for (int a = 0; a < A; ++a) { pos[a * 3] = 1.5f * (a % 7); pos[a * 3 + 1] = 1.1f * (a % 5); pos[a * 3 + 2] = 0.9f * (a % 3); }
  flts("aatype", aat); flts("profile", std::vector<float>((size_t)T * K, 0.f)); flts("deletion_mean", zT);
  flts("token_bonds", std::vector<float>((size_t)T * T, 0.f));
  flts("ref_pos", pos); flts("ref_charge", zA); flts("atom_mask", ones);
  ints("ref_element", el); ints("ref_atom_name_chars", names); ints("ref_space_uid", uid); ints("atom_to_token", a2t);
  std::vector<int> lm(T + 2, 5), lmSeq(T + 2, 0), t2r(T);
  lm[0] = 0; lm[T + 1] = 2;
  for (int t = 0; t < T; ++t) t2r[t] = t + 1;
  ints("lm/ids", lm); ints("lm/sequence_id", lmSeq); ints("lm/token_to_row", t2r);
  idx += "m meta/tokens " + std::to_string(T) + "\nm meta/atoms " + std::to_string(A) + "\nm meta/lm_rows " +
         std::to_string(T + 2) + "\nm meta/classes " + std::to_string(K) + "\n";
  fclose(bin);
  FILE* f = fopen((dir + "/model.idx").c_str(), "w"); fputs(idx.c_str(), f); fclose(f);
  return dir;
}

// one input, loaded: the whole fold. warm: the same launches on a synthetic input, nothing printed or
// written - every weight conversion, cuBLAS plan and kernel module loaded while the real input is exported
struct Opts { std::string dir, oracle, out; uint64_t seed; SamplerSettings sampler; bool profile; };
static int foldInput(const Opts& o, bool warm) {
  const std::string& oracle = o.oracle; const std::string& out = o.out; uint64_t seed = o.seed;
  const SamplerSettings& sampler = o.sampler; bool profile = o.profile && !warm;
  auto say = [&](const char* fmt, auto... v) { if (!warm) printf(fmt, v...); };
  CB(cublasSetMathMode(H, FAST ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
  int T = (int)M.meta("meta/tokens");
  Esmc e{(int)M.meta("meta/lm_rows"), (int)M.meta("meta/width"), (int)M.meta("meta/heads"),
         (int)dimOf("c/blocks/0/fc2/weights", 0), (int)M.meta("meta/layers"), (int)M.meta("meta/pairChannels")};
  say("ESMFold2: %d tokens, %d atoms, %d tower rows\n", T, (int)M.meta("meta/atoms"), e.rows);
  bool check = !oracle.empty() && !warm;
  int states = e.layers + 1;
  float* hidden = check ? dalloc((size_t)T * states * e.model) : nullptr;
  float* lmZ = dalloc((size_t)T * T * e.pair);
  auto t0 = std::chrono::steady_clock::now();
  languageModel(e, Idev("lm/ids"), Idev("lm/sequence_id"), Idev("lm/token_to_row"), T, lmZ,
                [&](int k, const float* x) {
                  if (check) gatherStateK<<<blocks((size_t)T * e.model), 256, 0, STREAM>>>(x, Idev("lm/token_to_row"),
                                                                                         hidden, T, states, k, e.model);
                });
  CK(cudaStreamSynchronize(STREAM));
  say("language model %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  if (check) {
    checkOracle("lm hidden states (37)", hidden, (size_t)T * states * e.model, "o/lm_hidden");
    if (getenv("EF2_PER_STATE")) {
      auto h = download(hidden, (size_t)T * states * e.model); const float* o = M.f("o/lm_hidden");
      for (int k = 0; k < states; ++k) {
        double num = 0, den = 0;
        for (int t = 0; t < T; ++t) for (int c = 0; c < e.model; ++c) {
          size_t at = ((size_t)t * states + k) * e.model + c; double d = h[at] - o[at]; num += d * d; den += (double)o[at] * o[at]; }
        say("    state %2d  %.3e\n", k, sqrt(num / den));
      }
    }
    checkOracle("lm pair", lmZ, (size_t)T * T * e.pair, "o/lm_z");
  }
  int A = (int)M.meta("meta/atoms"), Si = (int)M.meta("meta/singleInputs");
  float* sInputs = dalloc((size_t)T * Si);
  t0 = std::chrono::steady_clock::now();
  inputsEmbedder(T, A, sInputs, Si, check);
  CK(cudaStreamSynchronize(STREAM));
  say("inputs embedder %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  if (check) checkOracle("s_inputs", sInputs, (size_t)T * Si, "o/s_inputs");
  int C = e.pair;
  float* zi = dalloc((size_t)T * T * C); float* z = dalloc((size_t)T * T * C);
  zInit(T, C, sInputs, Si, lmZ, zi, check);
  t0 = std::chrono::steady_clock::now();
  if (profile) { prof::init(); prof::start(); }
  foldingTrunk(T, C, zi, z, 4, check);
  if (profile) { CK(cudaStreamSynchronize(STREAM)); prof::stop(25); }
  CK(cudaStreamSynchronize(STREAM));
  say("trunk %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  float* dg = dalloc((size_t)T * T * (int)M.meta("meta/distogramBins"));
  distogram(z, T, C, dg);
  if (check) checkOracle("distogram", dg, (size_t)T * T * (int)M.meta("meta/distogramBins"), "o/distogram");
  float* relPos = scratch<float>("zi.rel", (size_t)T * T * C);     // zInit's relative position encoding, kept
  Denoiser dn = makeDenoiser(T, A, z, relPos, sInputs, check);
  if (check && M.has("o/step0/x_noisy")) {
    float* xn0 = upload(M.f("o/step0/x_noisy"), (size_t)A * 3); float* xd0 = dalloc((size_t)A * 3);
    float t = (float)M.meta("o/step0/t_hat");
    denoise(dn, xn0, t, xd0, true);
    checkOracle("denoiser, step 0", xd0, (size_t)A * 3, "o/step0/x_denoised");
    int last = (int)M.meta("o/steps") - 1;
    std::string k = "o/step" + std::to_string(last);
    CK(cudaMemcpy(xn0, M.f(k + "/x_noisy"), (size_t)A * 12, cudaMemcpyHostToDevice));
    denoise(dn, xn0, (float)M.meta(k + "/t_hat"), xd0);
    checkOracle(("denoiser, step " + std::to_string(last)).c_str(), xd0, (size_t)A * 3, k + "/x_denoised");
  }
  t0 = std::chrono::steady_clock::now();
  int stepsRun = 0;
  std::vector<float> coords = sample(dn, sampler, seed, &stepsRun);
  say("sampler %.1f ms (%d steps)\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), stepsRun);
  if (!M.has("f/confidence/pae")) {
    fprintf(stderr, "the weights carry no confidence head: export them from model-esmfold2-conf-f32 (see native/ef2/README.md)\n");
    return 1;
  }
  if (check && M.has("o/conf/plddt_per_atom")) {        // the head on the reference's own coordinates
    float* xo = upload(M.f("o/coords"), (size_t)A * 3);
    confidenceHead(T, A, z, sInputs, Si, relPos, xo, true);
  }
  t0 = std::chrono::steady_clock::now();
  float* xd = upload(coords.data(), (size_t)A * 3);
  Confidence conf = confidenceHead(T, A, z, sInputs, Si, relPos, xd, false);
  say("confidence %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  std::vector<float> bf(A);
  for (int a = 0; a < A; ++a) bf[a] = 100.f * conf.plddtAtom[a];
  say("mean pLDDT %.2f  pTM %.4f", 100 * conf.meanPlddt, conf.ptm);
  { bool chains = false; std::vector<int> asym(T); CK(cudaMemcpy(asym.data(), Idev("asym_id"), T * 4, cudaMemcpyDeviceToHost));
    for (int t = 1; t < T; ++t) chains |= asym[t] != asym[0];
    if (chains) say("  ipTM %.4f", conf.iptm); }
  say("\n");
  if (!warm) writePdb(o.dir + "/pdb.template", out, coords, &bf);
  say("-> %s\n", out.c_str());
  if (getenv("EF2_DUMP")) { auto h = download(sInputs, (size_t)T * Si); FILE* f = fopen(getenv("EF2_DUMP"), "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f); }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ef2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--fast]\n"); return 1; }
  std::string weights, oracle, out = "fold.pdb"; uint64_t seed = 0; SamplerSettings sampler; bool waitInput = false, profile = false; std::string warmShape;
  for (int i = 2; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--oracle=", 9)) oracle = argv[i] + 9;
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
    else if (!strcmp(argv[i], "--fast")) FAST = true;
    else if (!strcmp(argv[i], "--atom-f32")) ATOM_BF16 = false;
    else if (!strcmp(argv[i], "--wait-input")) waitInput = true;     // start up while the input is still being exported
    else if (!strncmp(argv[i], "--warm=", 7)) warmShape = argv[i] + 7;    // T,A: fold a synthetic input of that size meanwhile
    else if (!strcmp(argv[i], "--profile")) profile = true;
    else if (!strcmp(argv[i], "--no-fused")) FUSED = false;
    else if (!strcmp(argv[i], "--no-fused256")) FUSED256 = false;
    else if (!strncmp(argv[i], "--seed=", 7)) seed = strtoull(argv[i] + 7, nullptr, 10);
    else if (!strncmp(argv[i], "--steps=", 8)) sampler.steps = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--inputs-window=", 16)) {           // 128: biohub's (the default); 0: dense
      int w = atoi(argv[i] + 16);
      INPUTS_HALF_WINDOW = w > 0 ? w / 2 : 1 << 30;
    }
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (weights.empty()) { fprintf(stderr, "--weights=<dir> (native/ef2/export_weights.mjs)\n"); return 1; }
  auto tStart = std::chrono::steady_clock::now();
  M.load(weights);
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  M.upload(0);
  Opts o{argv[1], oracle, out, seed, sampler, profile};
  if (!warmShape.empty()) {
    int wt = 0, wa = 0;
    if (sscanf(warmShape.c_str(), "%d,%d", &wt, &wa) != 2 || wt < 1 || wa < 1) { fprintf(stderr, "--warm=T,A\n"); return 1; }
    std::string dir = writeWarmInput(wt, wa);
    int seg = (int)M.segs.size();
    M.load(dir);
    Opts w = o; w.dir = dir; w.oracle = ""; w.sampler.steps = 2;
    foldInput(w, true);
    CK(cudaStreamSynchronize(STREAM));
    forgetEntries(M.unload(seg));
    std::string rm = "rm -rf '" + dir + "'"; if (system(rm.c_str())) {}
  }
  if (waitInput) {          // the exporter writes model.idx last, by a rename
    std::string idx = std::string(argv[1]) + "/model.idx", failed = std::string(argv[1]) + "/model.failed";
    for (int k = 0; access(idx.c_str(), R_OK) != 0; ++k) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "ef2: the input's export failed\n"); return 1; }
      if (k > 600000) { fprintf(stderr, "no %s after ten minutes\n", idx.c_str()); return 1; }
      usleep(1000);
    }
  }
  M.load(argv[1]);
  if (!oracle.empty()) M.load(oracle);
  printf("loaded in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count());
  return foldInput(o, false);
}
