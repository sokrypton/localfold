// ESMFold2 (esmfold2-fast-600m) in CUDA: ESM-C, the shim, the inputs embedder, the 24-block pair
// trunk (AF3's pair track, minus its grid attention and single track), the distogram and the
// diffusion sampler - each stage held to biohub's own forward on the same input (oracle.py).
//
//   ef2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--fast]
#include "esmc.cuh"
#include "atoms.cuh"
#include "trunk.cuh"

__global__ void gatherStateK(const float* x, const int* tokenToRow, float* out, int T, int states, int k, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int token = (int)(t / C), c = (int)(t % C), r = tokenToRow[token];
  out[((size_t)token * states + k) * C + c] = r < 0 ? 0.f : x[(size_t)r * C + c];
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ef2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--fast]\n"); return 1; }
  std::string weights, oracle, out = "fold.pdb";
  for (int i = 2; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--oracle=", 9)) oracle = argv[i] + 9;
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
    else if (!strcmp(argv[i], "--fast")) FAST = true;
    else if (!strcmp(argv[i], "--atom-f32")) ATOM_BF16 = false;
    else if (!strncmp(argv[i], "--inputs-window=", 16)) INPUTS_HALF_WINDOW = atoi(argv[i] + 16) / 2;   // 128: biohub's
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (weights.empty()) { fprintf(stderr, "--weights=<dir> (native/ef2/export_weights.mjs)\n"); return 1; }
  M.load(weights); M.load(argv[1]);
  if (!oracle.empty()) M.load(oracle);
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
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
  foldingTrunk(T, C, zi, z, 4, check);
  CK(cudaStreamSynchronize(STREAM));
  printf("trunk %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  float* dg = dalloc((size_t)T * T * (int)M.meta("meta/distogramBins"));
  distogram(z, T, C, dg);
  if (check) checkOracle("distogram", dg, (size_t)T * T * (int)M.meta("meta/distogramBins"), "o/distogram");
  if (getenv("EF2_DUMP")) { auto h = download(sInputs, (size_t)T * Si); FILE* f = fopen(getenv("EF2_DUMP"), "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f); }
  return 0;
}
