// ESMFold2 (esmfold2-fast-600m) in CUDA: ESM-C, the shim, the inputs embedder, the 24-block pair
// trunk (AF3's pair track, minus its grid attention and single track), the distogram and the
// diffusion sampler - each stage held to biohub's own forward on the same input (oracle.py).
//
//   ef2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--fast]
//   ef2 <input dir> --fold-bundle=<dir> --esmc-bundle=<dir> ...   (the page's bundles, read as they are)
#include "../../featurise/standalone_api.h"
#include "shim.cuh"         // (ESM-C's tower is cuda/plm/esmc.cuh)
#include "atoms.cuh"
#include "trunk.cuh"
#include "msa.cuh"
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
  flts("aatype", aat); flts("profile", aat); flts("deletion_mean", zT);
  ints("msa/rows", restype); flts("msa/deletion", zT);     // a one-row alignment: the query
  flts("token_bonds", std::vector<float>((size_t)T * T, 0.f));
  flts("ref_pos", pos); flts("ref_charge", zA); flts("atom_mask", ones);
  ints("ref_element", el); ints("ref_atom_name_chars", names); ints("ref_space_uid", uid); ints("atom_to_token", a2t);
  std::vector<int> lm(T + 2, 5), lmSeq(T + 2, 0), t2r(T);
  lm[0] = 0; lm[T + 1] = 2;
  for (int t = 0; t < T; ++t) t2r[t] = t + 1;
  ints("lm/ids", lm); ints("lm/sequence_id", lmSeq); ints("lm/token_to_row", t2r);
  idx += "m meta/tokens " + std::to_string(T) + "\nm meta/atoms " + std::to_string(A) + "\nm meta/lm_rows " +
         std::to_string(T + 2) + "\nm meta/classes " + std::to_string(K) + "\nm meta/msa_depth 1\n";
  fclose(bin);
  FILE* f = fopen((dir + "/model.idx").c_str(), "w"); fputs(idx.c_str(), f); fclose(f);
  return dir;
}

// one input, loaded: the whole fold. warm: the same launches on a synthetic input, nothing printed or
// written - every weight conversion, cuBLAS plan and kernel module loaded while the real input is exported
struct Opts { std::string dir, oracle, out; uint64_t seed; SamplerSettings sampler; bool profile; bool stepsGiven = false; };
// --frames=DIR: the trunk's contact map (contacts-00-of-01.u8, a byte a pair) and every sampler step's
// prediction (frame-SSSS-NNNN.pdb), written through common.cuh's AsyncTap so the fold does not wait for them
static std::string FRAMES_DIR;
// the softmax mass of each pair's first contact_bins bins (the bias is already in the logits: distogram())
// pair positions [p0, p0 + cnt) of z + z^T (symmetriseK's arithmetic, a block at a time)
__global__ void symmetriseRowsK(const float* z, float* out, size_t p0, size_t cnt, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = p0 + t / C, i = ij / T, j = ij % T;
  out[t] = z[ij * C + c] + z[(j * T + i) * C + c];
}
__global__ void contactsK(const float* logits, const int* contactBins, float* out, size_t pairs, int bins) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= pairs) return;
  const float* l = logits + ij * bins;
  float mx = -INFINITY; for (int b = 0; b < bins; ++b) mx = fmaxf(mx, l[b]);
  float total = 0, near = 0;
  for (int b = 0; b < bins; ++b) { float w = expf(l[b] - mx); total += w; if (b < contactBins[ij]) near += w; }
  out[ij] = near / fmaxf(total, 1e-30f);
}
static int foldInput(const Opts& o, bool warm) {
  const std::string& oracle = o.oracle; const std::string& out = o.out; uint64_t seed = o.seed;
  // a bundle that states its sampler (the released models: tools/export_esmfold2_trunk.py) is sampled so - the
  // command's or the job's --steps still wins
  SamplerSettings sampler = o.sampler;
  if (M.has("meta/samplerSteps")) {
    if (!o.stepsGiven && !warm) sampler.steps = (int)M.meta("meta/samplerSteps");
    sampler.gamma0 = M.meta("meta/samplerGamma0"); sampler.gammaMin = M.meta("meta/samplerGammaMin");
    sampler.noiseScale = M.meta("meta/samplerNoiseScale"); sampler.stepScale = M.meta("meta/samplerStepScale");
    sampler.p = M.meta("meta/samplerRho"); sampler.sMin = M.meta("meta/samplerSigmaMin");
    sampler.sMax = M.meta("meta/samplerSigmaMax"); sampler.maxSigma = M.meta("meta/samplerMaxSigma");
  }
  bool profile = o.profile && !warm;
  auto say = [&](const char* fmt, auto... v) { if (!warm) printf(fmt, v...); };
  // EF2_MEM: device memory in use at each phase boundary
  auto mem = [&](const char* at) {
    if (!warm && getenv("LOCALFOLD_MEM")) { memReport(at); return; }   // (common.cuh's, with the largest holders)
    if (warm || !getenv("EF2_MEM")) return;
    CK(cudaDeviceSynchronize()); size_t fr, tot; deviceMemInfo(&fr, &tot);
    size_t held = 0; for (auto& [k, v] : SCRATCH) held += v.second;
    printf("  memory %-22s %6.2f GB in use (scratch %.2f)\n", at, (tot - fr) / 1e9, held / 1e9);
  };
  CB(cublasSetMathMode(H, FAST ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
  int T = (int)M.meta("meta/tokens");
  const bool residentTower = M.has("c/blocks/0/fc2/weightsT");      // (ESM-C 6B: [out, in], resident int8)
  Esmc e{(int)M.meta("meta/lm_rows"), (int)M.meta("meta/width"), (int)M.meta("meta/heads"),
         residentTower ? (int)dimOf("c/blocks/0/fc2/weightsT", 1) : (int)dimOf("c/blocks/0/fc2/weights", 0),
         (int)M.meta("meta/layers"), (int)M.meta("meta/pairChannels"), (float)M.meta("meta/residualScale", 1.0)};
  say("ESMFold2: %d tokens, %d atoms, %d tower rows\n", T, (int)M.meta("meta/atoms"), e.rows);
  bool check = !oracle.empty() && !warm;
  int states = e.layers + 1;
  float* hidden = check ? dalloc((size_t)T * states * e.model) : nullptr;
  // on a card short of room z_init is streamed (ZINIT_STREAM): the language model's pair is never made whole
  const bool parcae = parcaeRecycle();           // (the released models keep the language model's pair: their loop reads it)
  const bool streamZ = !check && !parcae && shortPair((size_t)T * T, e.pair);
  float* lmZ = streamZ ? nullptr : dalloc((size_t)T * T * e.pair);
  auto t0 = std::chrono::steady_clock::now();
  if (residentTower && M.residentParked()) {      // (the last fold parked the tower: back from its shards)
    M.unparkResident();
    say("tower back on the device in %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    t0 = std::chrono::steady_clock::now();
  }
  bool profLm = profile && getenv("EF2_PROFILE") && std::string(getenv("EF2_PROFILE")) == "lm";
  if (profLm) { prof::init(); prof::start(); }
  languageModel(e, Idev("lm/ids"), Idev("lm/sequence_id"), Idev("lm/token_to_row"), T, lmZ,
                [&](int k, const float* x) {
                  if (check) gatherStateK<<<blocks((size_t)T * e.model), 256, 0, STREAM>>>(x, Idev("lm/token_to_row"),
                                                                                         hidden, T, states, k, e.model);
                });
  CK(cudaStreamSynchronize(STREAM));
  if (profLm) prof::stop(15);
  say("language model %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  // the resident tower is idle until the next fold: on a card without room beside it for the trunk's three pairs
  // (z_init, z and the injection) it leaves the device, and the next fold reads it back (1,500 tokens on a T4 ran out
  // with it held: 6.4 GB of codes beside four 2.3 GB pairs)
  // (never in the warm-up: it runs while the tower is still being copied up, and freeing the copy under the upload
  // killed the process - "cannot put a model.bin on the device" - whenever room was short and the upload slow)
  if (residentTower && !warm && !roomFor(3 * (size_t)T * T * e.pair * 4, { "tower.w16" })) {
    releaseScratch({ "tower.", "esmc." });
    if (!streamZ) releaseScratch({ "shim." });     // (a streamed z_init reads shim.tokens later)
    say("tower parked off the device: %.2f GB\n", M.parkResident() / 1e9);
  }
  // the language model's pair is read once a pass, to seed the injection: on a card without room beside it for the
  // trunk's three pairs it waits in pinned host memory and is copied in each pass (four 3.1 GB pairs at 1,750 tokens
  // did not fit a T4 beside the folding weights)
  const float* lmHost = nullptr;
  if (parcae && lmZ && !roomFor(3 * (size_t)T * T * e.pair * 4)) {
    parkToHost(lmZ, (size_t)T * T * e.pair * 4); lmHost = PARK_HOST;
    say("language model's pair parked in host memory\n");
  }
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
  if (getenv("EF2_SAVE_LMZ") && lmZ && !warm) {   // the language model's pair, raw float32 [T, T, P] (a comparison aid)
    auto h = download(lmZ, (size_t)T * T * e.pair); FILE* f = fopen(getenv("EF2_SAVE_LMZ"), "wb");
    fwrite(h.data(), 4, h.size(), f); fclose(f);
  }
  int A = (int)M.meta("meta/atoms"), Si = (int)M.meta("meta/singleInputs");
  float* sInputs = dalloc((size_t)T * Si);
  t0 = std::chrono::steady_clock::now();
  bool profIn = profile && getenv("EF2_PROFILE") && std::string(getenv("EF2_PROFILE")) == "inputs";
  if (profIn) { prof::init(); prof::start(); }
  inputsEmbedder(T, A, sInputs, Si, check);
  CK(cudaStreamSynchronize(STREAM));
  if (profIn) prof::stop(15);
  say("inputs embedder %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  if (check) checkOracle("s_inputs", sInputs, (size_t)T * Si, "o/s_inputs");
  int C = e.pair;
  mem("language model");
  float* zi = nullptr; float* z = nullptr;          // (z after the MSA encoder: one pair fewer while it runs)
  float *ziRows = nullptr, *ziCols = nullptr, *sTok = nullptr;
  if (streamZ) {
    // what the streamed z_init reads, kept past the scratch release below: the per-token states and the
    // row and column projections - [T, C] each
    sTok = dalloc((size_t)T * e.pair);
    CK(cudaMemcpyAsync(sTok, scratch<float>("shim.tokens", (size_t)T * e.pair), (size_t)T * e.pair * 4, cudaMemcpyDeviceToDevice, STREAM));
    ziRows = dalloc((size_t)T * C); ziCols = dalloc((size_t)T * C);
    gemm(sInputs, F("featuriser/zInit1"), ziRows, T, Si, C);
    gemm(sInputs, F("featuriser/zInit2"), ziCols, T, Si, C);
    ZINIT_STREAM = { ziRows, ziCols, sTok, e.pair };
    CK(cudaStreamSynchronize(STREAM));
  } else {
    zi = dalloc((size_t)T * T * C);
    zInit(T, C, sInputs, Si, parcae ? nullptr : lmZ, zi, check);
    if (hasMsaEncoder()) {            // the full ESMFold2: the alignment's encoder over z_init, which it replaces
      auto m0 = std::chrono::steady_clock::now();
      if (getenv("EF2_SAVE_ZINIT") && !warm) {       // (comparison aids: the encoder's two inputs, raw float32)
        auto h = download(zi, (size_t)T * T * C); FILE* f = fopen(getenv("EF2_SAVE_ZINIT"), "wb");
        fwrite(h.data(), 4, h.size(), f); fclose(f);
        auto s = download(sInputs, (size_t)T * Si); f = fopen((std::string(getenv("EF2_SAVE_ZINIT")) + ".s").c_str(), "wb");
        fwrite(s.data(), 4, s.size(), f); fclose(f);
      }
      msaEncode(zi, sInputs, T, C, Si, seed);
      CK(cudaStreamSynchronize(STREAM));
      say("msa encoder %.1f ms (%d rows)\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m0).count(),
          std::min((int)M.meta("meta/msa_depth"), (int)M.meta("meta/msaMaxDepth")));
      if (getenv("EF2_SAVE_INJECT") && !warm) {      // (a comparison aid: the encoder's pair, raw float32)
        auto h = download(zi, (size_t)T * T * C); FILE* f = fopen(getenv("EF2_SAVE_INJECT"), "wb");
        fwrite(h.data(), 4, h.size(), f); fclose(f);
      }
    }
    if (!parcae) { CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(lmZ)); lmZ = nullptr; }
  }
  // a large input gives each phase the whole card: its predecessor's scratch released (a pair over
  // 128 MB, ~350 tokens). Below that the scratch is kept - re-allocating it cost every phase cudaMallocs
  z = dalloc((size_t)T * T * C);
  bool tight = (size_t)T * T * C * 4 > ((size_t)128 << 20);
  if (tight) releaseScratch();
  t0 = std::chrono::steady_clock::now();
  // --profile times one stage's kernels: EF2_PROFILE=trunk (the default), lm, inputs, sampler or confidence
  std::string profStage = getenv("EF2_PROFILE") ? getenv("EF2_PROFILE") : "trunk";
  bool profTrunk = profile && profStage == "trunk";
  if (profTrunk) { prof::init(); prof::start(); }
  foldingTrunk(T, C, zi, z, getenv("EF2_PASSES") ? atoi(getenv("EF2_PASSES")) : M.has("meta/loops") ? (int)M.meta("meta/loops") + 1 : 4, check, lmZ, seed, lmHost);
  if (lmZ) { CK(cudaStreamSynchronize(STREAM)); CK(cudaFree(lmZ)); lmZ = nullptr; }
  if (getenv("EF2_SAVE_PAIR") && !warm) {         // the trunk's final pair, raw float32 [T, T, C] (a comparison aid)
    auto h = download(z, (size_t)T * T * C); FILE* f = fopen(getenv("EF2_SAVE_PAIR"), "wb");
    fwrite(h.data(), 4, h.size(), f); fclose(f);
  }
  if (profTrunk) { CK(cudaStreamSynchronize(STREAM)); prof::stop(25); }
  CK(cudaStreamSynchronize(STREAM));
  say("trunk %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  mem("trunk");
  if (zi) { CK(cudaFree(zi)); zi = nullptr; }
  for (float* p : { ziRows, ziCols, sTok }) if (p) CK(cudaFree(p));
  if (tight) releaseScratch();
  // the page's contact map off the trunk's distogram (webgpu/esmfold2/distogram.js: the softmax mass
  // under each pair's threshold, the bins the exporter counted) - for the confidences file, and with
  // --frames tapped to the page before the sampler runs, as the page shows its own
  std::vector<float> contacts;
  if (!warm && M.has("contact_bins")) {
    int bins = (int)M.meta("meta/distogramBins");
    if ((int)M.meta("meta/contactBinsFor") != bins) {
      fprintf(stderr, "the input's contact bins were counted for a %d-bin distogram and this one has %d\n",
              (int)M.meta("meta/contactBinsFor"), bins);
      return 1;
    }
    size_t P = (size_t)T * T;
    float* probs = scratch<float>("dg.contacts", P);
    if (tight) {
      // on a card short of room a block of pair positions at a time: z + z^T, the projection, the contact mass -
      // never the symmetrised pair (5.9 GB at 2400 tokens) or the logits (2.95 GB) whole; the same arithmetic
      int Bn = (int)dimOf("f/distogram/weights", 1);
      size_t per = std::max<size_t>(1, std::min(P, ((size_t)64 << 20) / std::max(C, Bn)));
      float* zs = scratch<float>("dg.symRows", per * C); float* dg = scratch<float>("dg.logitRows", per * bins);
      for (size_t p0 = 0; p0 < P; p0 += per) {
        size_t n = std::min(per, P - p0);
        symmetriseRowsK<<<blocks(n * C), 256, 0, STREAM>>>(z, zs, p0, n, T, C);
        gemm(zs, F("distogram/weights"), dg, n, C, Bn);
        addBias(dg, F("distogram/bias"), n, Bn);
        contactsK<<<blocks(n), 256, 0, STREAM>>>(dg, Idev("contact_bins") + p0, probs + p0, n, bins);
      }
    } else {
      float* dg = scratch<float>("dg.logits", P * bins);
      distogram(z, T, C, dg);
      contactsK<<<blocks(P), 256, 0, STREAM>>>(dg, Idev("contact_bins"), probs, P, bins);
    }
    if (!FRAMES_DIR.empty()) {
      TAP().reserve(1, P);
      unsigned char* bytes = scratch<unsigned char>("dg.contacts8", P);
      quantiseK<<<blocks(P), 256, 0, STREAM>>>(probs, bytes, P, 1.f / 255);
      std::string path = FRAMES_DIR + "/contacts-00-of-01.u8";
      TAP().offer({{bytes, P}}, [path, P](const char* host, const std::vector<size_t>&) { writeWhole(path, host, P); });
    }
    contacts = download(probs, P);
    if (tight) releaseScratch();
  }
  if (check) {                                   // (nothing else reads the distogram)
    float* dg = dalloc((size_t)T * T * (int)M.meta("meta/distogramBins"));
    distogram(z, T, C, dg);
    checkOracle("distogram", dg, (size_t)T * T * (int)M.meta("meta/distogramBins"), "o/distogram");
    CK(cudaFree(dg)); if (tight) releaseScratch();
  }
  Denoiser dn = makeDenoiser(T, A, z, sInputs, check);
  mem("denoiser built");
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
  bool profSampler = profile && profStage == "sampler";
  if (profSampler) { prof::init(); prof::start(); }
  // --frames: each sampler step's prediction, superposed onto the first (as the page draws its frames),
  // written off the fold's path by the tap's thread
  std::vector<double> frameRef; double frameCentre[3] = {};
  std::vector<float> frameMask;
  if (!FRAMES_DIR.empty() && !warm) {
    frameMask = download(dn.atoms.ctx.mask, A);
    TAP().reserve(64, (size_t)A * 12);
    FRAME_HOOK = [&, A](const float* dd, int step, int steps) {
      char name[64]; snprintf(name, sizeof name, "/frame-%04d-%04d.pdb", step, steps);
      std::string path = FRAMES_DIR + name;
      TAP().offer({{dd, (size_t)A * 12}}, [&, A, path](const char* host, const std::vector<size_t>&) {
        const float* x = (const float*)host;
        std::vector<double> pts; std::vector<int> live;
        for (int a = 0; a < A; ++a) if (frameMask[a] > 0) { live.push_back(a); for (int k = 0; k < 3; ++k) pts.push_back(x[a * 3 + k]); }
        double c[3] = {};
        for (size_t q = 0; q < live.size(); ++q) for (int k = 0; k < 3; ++k) c[k] += pts[q * 3 + k] / live.size();
        for (size_t q = 0; q < live.size(); ++q) for (int k = 0; k < 3; ++k) pts[q * 3 + k] -= c[k];
        if (frameRef.empty()) { frameRef = pts; for (int k = 0; k < 3; ++k) frameCentre[k] = c[k]; }
        double R[9]; bestRotation(pts, frameRef, R);
        std::vector<float> moved(x, x + (size_t)A * 3);
        for (size_t q = 0; q < live.size(); ++q) {
          const double* p = &pts[q * 3];
          for (int k = 0; k < 3; ++k) moved[live[q] * 3 + k] = (float)(R[k * 3] * p[0] + R[k * 3 + 1] * p[1] + R[k * 3 + 2] * p[2] + frameCentre[k]);
        }
        writePdb(o.dir + "/pdb.template", path + ".tmp", moved);
        rename((path + ".tmp").c_str(), path.c_str());
      });
    };
  }
  std::vector<float> coords = sample(dn, sampler, seed, &stepsRun);
  FRAME_HOOK = nullptr;
  if (!FRAMES_DIR.empty()) TAP().drain();
  if (profSampler) { CK(cudaStreamSynchronize(STREAM)); prof::stop(30); }
  mem("sampler");
  freeDenoiser(dn); if (tight) releaseScratch();
  say("sampler %.1f ms (%d steps)\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), stepsRun);
  if (!M.has("f/confidence/pae")) {
    fprintf(stderr, "the weights carry no confidence head: export them from model-esmfold2-conf-f32 (see cuda/ef2/README.md)\n");
    return 1;
  }
  if (check && M.has("o/conf/plddt_per_atom")) {        // the head on the reference's own coordinates
    float* xo = upload(M.f("o/coords"), (size_t)A * 3);
    confidenceHead(T, A, z, sInputs, Si, xo, true);
  }
  t0 = std::chrono::steady_clock::now();
  float* xd = upload(coords.data(), (size_t)A * 3);
  bool profConf = profile && profStage == "confidence";
  if (profConf) { prof::init(); prof::start(); }
  Confidence conf = confidenceHead(T, A, z, sInputs, Si, xd, false, true);   // (z is read by nothing after)
  if (profConf) { CK(cudaStreamSynchronize(STREAM)); prof::stop(25); }
  mem("confidence");
  say("confidence %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  std::vector<float> bf(A);
  for (int a = 0; a < A; ++a) bf[a] = 100.f * conf.plddtAtom[a];
  say("mean pLDDT %.2f  pTM %.4f", 100 * conf.meanPlddt, conf.ptm);
  { bool chains = false; std::vector<int> asym(T); CK(cudaMemcpy(asym.data(), Idev("asym_id"), T * 4, cudaMemcpyDeviceToHost));
    for (int t = 1; t < T; ++t) chains |= asym[t] != asym[0];
    if (chains) say("  ipTM %.4f", conf.iptm); }
  say("\n");
  if (!warm) { writePdb(o.dir + "/pdb.template", out, coords, &bf); writeConfidences(out, T, conf, contacts); }
  say("-> %s\n", out.c_str());
  if (getenv("EF2_DUMP")) { auto h = download(sInputs, (size_t)T * Si); FILE* f = fopen(getenv("EF2_DUMP"), "wb"); fwrite(h.data(), 4, h.size(), f); fclose(f); }
  // everything given back, so a warm-up leaves the real fold the card it had
  CK(cudaStreamSynchronize(STREAM));
  for (float* p : {z, sInputs, xd, hidden}) if (p) CK(cudaFree(p));
  if (tight) releaseScratch();
  return 0;
}

// --detach-output: on success the last line is "ef2: done" and stdout closes, so a caller reading it to
// its end returns while the driver releases this process's device (0.14 s of exit; cuda/ef2/fold does)
static bool DETACH = false;
static int foldMain(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ef2 <input dir> --weights=<dir> [--oracle=<dir>] [--out=fold.pdb] [--fast]\n"); return 1; }
  // the big-input paths when SHORT_PAIR_TIMES x the f32 pair does not fit the room (cuda/af3's common.cuh,
  // shortPair); LOCALFOLD_SHORT_PAIR_TIMES=0 is the old 64th-of-the-card rule
  SHORT_PAIR_TIMES = getenv("LOCALFOLD_SHORT_PAIR_TIMES") ? atof(getenv("LOCALFOLD_SHORT_PAIR_TIMES")) : 18;
  std::string weights, foldBundle, esmcBundle, oracle, out = "fold.pdb"; uint64_t seed = 0; SamplerSettings sampler; bool waitInput = false, waitForever = false, profile = false, stepsGiven = false; std::string warmShape, serveDir;
  for (int i = 2; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--fold-bundle=", 14)) foldBundle = argv[i] + 14;
    else if (!strncmp(argv[i], "--esmc-bundle=", 14)) esmcBundle = argv[i] + 14;
    else if (!strncmp(argv[i], "--oracle=", 9)) oracle = argv[i] + 9;
    else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
    else if (!strcmp(argv[i], "--fast")) FAST = true;
    else if (!strcmp(argv[i], "--no-sampler16")) SAMPLER16 = false;
    else if (!strcmp(argv[i], "--no-tower16")) TOWER16 = false;
    else if (!strcmp(argv[i], "--no-sampler-graph")) SAMPLER_GRAPH = false;
    else if (!strcmp(argv[i], "--no-token-flash")) TOKEN_FLASH = false;
    else if (!strcmp(argv[i], "--atom-f32")) ATOM_BF16 = false;
    else if (!strcmp(argv[i], "--wait-input")) waitInput = true;     // start up while the input is still being exported
    else if (!strcmp(argv[i], "--wait-input=0")) waitInput = waitForever = true;   // ...with no timeout (the standalone mode: its own featuriser)
    else if (!strncmp(argv[i], "--warm=", 7)) warmShape = argv[i] + 7;    // T,A: fold a synthetic input of that size meanwhile
    else if (!strcmp(argv[i], "--profile")) profile = true;
    else if (!strcmp(argv[i], "--detach-output")) DETACH = true;
    else if (!strncmp(argv[i], "--frames=", 9)) FRAMES_DIR = argv[i] + 9;   // stream the trunk's contacts and the sampler's frames
    else if (!strcmp(argv[i], "--no-fused")) FUSED = false;
    else if (!strcmp(argv[i], "--no-fused256")) FUSED256 = false;
    else if (!strncmp(argv[i], "--seed=", 7)) seed = strtoull(argv[i] + 7, nullptr, 10);
    else if (!strncmp(argv[i], "--steps=", 8)) { sampler.steps = atoi(argv[i] + 8); stepsGiven = true; }
    else if (!strncmp(argv[i], "--serve=", 8)) serveDir = argv[i] + 8;   // stay up, folding each job dropped there
    else if (!strncmp(argv[i], "--inputs-window=", 16)) {           // 128: biohub's (the default); 0: dense
      int w = atoi(argv[i] + 16);
      INPUTS_HALF_WINDOW = w > 0 ? w / 2 : 1 << 30;
    }
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (weights.empty() == (foldBundle.empty() || esmcBundle.empty())) {
    fprintf(stderr, "--weights=<dir> (cuda/ef2/export_weights.mjs), or --fold-bundle=<dir> and --esmc-bundle=<dir>\n");
    return 1;
  }
  auto tStart = std::chrono::steady_clock::now();
  // the weights: one exported file, or the two bundles read as they are (decoded on the device)
  if (!weights.empty()) M.load(weights);
  else { M.loadBundle(foldBundle, "f"); M.loadBundle(esmcBundle, "c", nullptr, "", "blocks/"); }   // (int8 tower blocks stay resident)
  const int weightSegs = (int)M.segs.size();
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  { void* ws; CK(cudaMalloc(&ws, 64 << 20)); CB(cublasSetWorkspace(H, ws, 64 << 20)); }   // graph capture needs it
  auto tCtx = std::chrono::steady_clock::now();
  Opts o{argv[1], oracle, out, seed, sampler, profile, stepsGiven};
  // The warm-up runs WHILE the weights go up (0.24 s for 2.9 GB, the GPU otherwise idle): its kernels
  // read a copy still arriving, so its answers are garbage, and everything derived from the weights
  // is forgotten after it. The warm fold needs only the shapes - it loads every kernel module and
  // cuBLAS plan. 6MRR's cold trunk is 128 ms against 39 warm.
  bool warming = !warmShape.empty();
  for (int sgi = 0; sgi < weightSegs; ++sgi) { if (warming) M.uploadAsync(sgi); else M.upload(sgi); }
  if (warming) {
    int wt = 0, wa = 0;
    if (sscanf(warmShape.c_str(), "%d,%d", &wt, &wa) != 2 || wt < 1 || wa < 1) { fprintf(stderr, "--warm=T,A\n"); return 1; }
    // at most 96 tokens (past FUSED256_MIN_TOKENS, so a large input's kernels are the ones warmed): a warm
    // fold the input's size outlasted the upload it hides behind - 5CAJ's 261 made the fold 0.2 s slower
    if (wt > 96) { wa = std::max(1, (int)((long)wa * 96 / wt)); wt = 96; }
    std::string dir = writeWarmInput(wt, wa);
    int seg = (int)M.segs.size();
    M.load(dir);
    Opts w = o; w.dir = dir; w.oracle = ""; w.sampler.steps = 2;
    foldInput(w, true);
    CK(cudaStreamSynchronize(STREAM));
    forgetEntries(M.unload(seg));
    std::string rm = "rm -rf '" + dir + "'"; if (system(rm.c_str())) {}
    M.waitUploads();
    forgetDerivedWeights();
  }
  // --fast: the f16 mirrors made now (the ESM-C tower's, the folding bundle's), and the tower's matrices'
  // f32 copy dropped from the device - 2.2 GB, read only through the mirror on this path
  size_t dropped = 0;
  if (FAST) {
    Wh("f/blocks/0/pairTransition/transition1");
    if (M.has("c/blocks/0/qkv/weights")) {           // (a resident tower has no float32 copy to drop)
      Wh("c/blocks/0/qkv/weights");
      dropped = compactWeights(M.at("c/blocks/0/qkv/weights").seg, towerHalf);
    }
  }
  if (getenv("EF2_STARTUP")) {
    auto now = std::chrono::steady_clock::now();
    printf("tower f32 copies dropped: %.2f GB\n", dropped / 1e9);
    printf("context %.0f ms, weights up%s %.0f ms\n", std::chrono::duration<double, std::milli>(tCtx - tStart).count(),
           warming ? " and warm-up" : "", std::chrono::duration<double, std::milli>(now - tCtx).count());
  }
  if (!serveDir.empty()) {
    // (the input argument is unused: each job names its own) - the job's flags: --out, --seed, --steps, --frames
    serveJobs(serveDir, "ef2", [&](const std::string& input, const std::vector<std::string>& flags) {
      Opts j = o; j.dir = input; j.out = "fold.pdb"; FRAMES_DIR.clear();
      for (auto& f : flags) {
        if (!f.compare(0, 6, "--out=")) j.out = f.substr(6);
        else if (!f.compare(0, 7, "--seed=")) j.seed = strtoull(f.c_str() + 7, nullptr, 10);
        else if (!f.compare(0, 8, "--steps=")) { j.sampler.steps = atoi(f.c_str() + 8); j.stepsGiven = true; }
        else if (!f.compare(0, 9, "--frames=")) FRAMES_DIR = f.substr(9);
      }
      int seg = (int)M.segs.size();
      M.load(input);
      int code = foldInput(j, false);
      CK(cudaStreamSynchronize(STREAM));
      forgetEntries(M.unload(seg));
      return code;
    });
    finish(0);
  }
  if (waitInput) {          // the exporter writes model.idx last, by a rename
    std::string idx = std::string(argv[1]) + "/model.idx", failed = std::string(argv[1]) + "/model.failed";
    for (int k = 0; access(idx.c_str(), R_OK) != 0; ++k) {
      if (access(failed.c_str(), F_OK) == 0) { fprintf(stderr, "ef2: the input's export failed\n"); return 1; }
      if (k > 600000 && !waitForever) { fprintf(stderr, "no %s after ten minutes\n", idx.c_str()); return 1; }
      usleep(1000);
    }
  }
  M.load(argv[1]);
  if (!oracle.empty()) M.load(oracle);
  printf("loaded in %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count());
  int rc = foldInput(o, false);
  if (DETACH && rc == 0) { printf("ef2: done\n"); fflush(stdout); fflush(stderr); fclose(stdout); }
  finish(rc);
}

// `esmfold2 --job=<job.json> --out=<pdb>` (any first argument that is a flag): the whole protocol in this one process -
// the weights fetched, the input featurised in-process while the device starts, the fold (cuda/featurise/standalone.h);
// `esmfold2 <featurised dir> ...` and `esmfold2 - --serve=<dir>` as before
int main(int argc, char** argv) {
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) return lf::standalone::main("ef2", argc, argv, foldMain);
  return foldMain(argc, argv);
}
