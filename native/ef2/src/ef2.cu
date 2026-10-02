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

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ef2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--fast]\n"); return 1; }
  std::string weights, oracle, out = "fold.pdb"; uint64_t seed = 0; SamplerSettings sampler; bool waitInput = false, profile = false;
  for (int i = 2; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--oracle=", 9)) oracle = argv[i] + 9;
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
    else if (!strcmp(argv[i], "--fast")) FAST = true;
    else if (!strcmp(argv[i], "--atom-f32")) ATOM_BF16 = false;
    else if (!strcmp(argv[i], "--wait-input")) waitInput = true;
    else if (!strcmp(argv[i], "--profile")) profile = true;
    else if (!strcmp(argv[i], "--no-fused")) FUSED = false;     // start up while the input is still being exported
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
  CB(cublasSetMathMode(H, FAST ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
  int T = (int)M.meta("meta/tokens");
  Esmc e{(int)M.meta("meta/lm_rows"), (int)M.meta("meta/width"), (int)M.meta("meta/heads"),
         (int)dimOf("c/blocks/0/fc2/weights", 0), (int)M.meta("meta/layers"), (int)M.meta("meta/pairChannels")};
  printf("ESMFold2: %d tokens, %d atoms, %d tower rows\n", T, (int)M.meta("meta/atoms"), e.rows);
  bool check = !oracle.empty();
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
  printf("language model %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  if (check) {
    checkOracle("lm hidden states (37)", hidden, (size_t)T * states * e.model, "o/lm_hidden");
    if (getenv("EF2_PER_STATE")) {
      auto h = download(hidden, (size_t)T * states * e.model); const float* o = M.f("o/lm_hidden");
      for (int k = 0; k < states; ++k) {
        double num = 0, den = 0;
        for (int t = 0; t < T; ++t) for (int c = 0; c < e.model; ++c) {
          size_t at = ((size_t)t * states + k) * e.model + c; double d = h[at] - o[at]; num += d * d; den += (double)o[at] * o[at]; }
        printf("    state %2d  %.3e\n", k, sqrt(num / den));
      }
    }
    checkOracle("lm pair", lmZ, (size_t)T * T * e.pair, "o/lm_z");
  }
  int A = (int)M.meta("meta/atoms"), Si = (int)M.meta("meta/singleInputs");
  float* sInputs = dalloc((size_t)T * Si);
  t0 = std::chrono::steady_clock::now();
  inputsEmbedder(T, A, sInputs, Si, check);
  CK(cudaStreamSynchronize(STREAM));
  printf("inputs embedder %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  if (check) checkOracle("s_inputs", sInputs, (size_t)T * Si, "o/s_inputs");
  int C = e.pair;
  float* zi = dalloc((size_t)T * T * C); float* z = dalloc((size_t)T * T * C);
  zInit(T, C, sInputs, Si, lmZ, zi, check);
  t0 = std::chrono::steady_clock::now();
  if (profile) { prof::init(); prof::start(); }
  foldingTrunk(T, C, zi, z, 4, check);
  if (profile) { CK(cudaStreamSynchronize(STREAM)); prof::stop(25); }
  CK(cudaStreamSynchronize(STREAM));
  printf("trunk %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
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
  printf("sampler %.1f ms (%d steps)\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), stepsRun);
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
  printf("confidence %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  std::vector<float> bf(A);
  for (int a = 0; a < A; ++a) bf[a] = 100.f * conf.plddtAtom[a];
  printf("mean pLDDT %.2f  pTM %.4f", 100 * conf.meanPlddt, conf.ptm);
  { bool chains = false; std::vector<int> asym(T); CK(cudaMemcpy(asym.data(), Idev("asym_id"), T * 4, cudaMemcpyDeviceToHost));
    for (int t = 1; t < T; ++t) chains |= asym[t] != asym[0];
    if (chains) printf("  ipTM %.4f", conf.iptm); }
  printf("\n");
  writePdb(std::string(argv[1]) + "/pdb.template", out, coords, &bf);
  printf("-> %s\n", out.c_str());
  if (getenv("EF2_DUMP")) { auto h = download(sInputs, (size_t)T * Si); FILE* f = fopen(getenv("EF2_DUMP"), "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f); }
  return 0;
}
