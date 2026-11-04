// RFD4-Proteina on Metal: a job file in, a design out, in one process with no JavaScript.
//
//   bash metal/build.sh rfd4p
//   ./metal/rfd4p/localfold-rfd4p --job=<spec.json> --bundle=model-rfd4p-int8 [--out=design.pdb] [--seed=N]
//
// The spec is read and featurised by `cuda/rfd4p/featurise/featurise.cpp` LINKED IN (the same object cuda/rfd4p
// links, and byte for byte what `rfd4p-featurise` writes - measured: the whole input directory identical and all
// 44 tensors of the generation export exact against `cuda/rfd4p/export_input.mjs`, the replayed Box-Muller stream
// among them). Then the trunk, the autoencoder's decoder and the flow sampler run on this Mac's GPU, and the
// design is written. A 54-residue unconditional design is 60 steps in 8.7 s; the whole job, featurisation
// included, is 13.
//
// WHAT RUNS: unconditional, motif scaffolding, protein-protein interfaces, small molecules (whose chirality is
// re-aligned every step), partial diffusion and inverse folding. Measured, consecutive CA-CA within a chain:
// unconditional 3.81 A median over [3.73, 3.90], motif 3.79, ppi 3.80, small molecule 3.81, partial 3.74 - and
// inverse folding returns the input's own backbone to 0.0005 A, which is the nm-to-angstrom rounding.
// 🔴 READ THAT AT A CONVERGED STEP COUNT: the same code at 2 steps gives 11 A and at 60 gives 3.81. The
// trajectory gate below runs at 2 steps and says nothing about geometry; they are different questions.
//
// REFUSED BY NAME rather than run as something else: a job with two or more coordinate groups (their labels are
// permuted every step and this port does not do it - and neither does cuda/rfd4p, which freezes them silently),
// and `center_every_step`.
//
// THE GATE, over an exported input directory rather than a job:
//   localfold-rfd4p <input dir> --bundle=<model-rfd4p-int8> [--oracle=<dir>] [--encoder=<dir>] [--gen=<dir>-gen]
//     [--out=design.pdb] [--f16]
// built by `cuda/rfd4p/export_input.mjs <in> --oracle=<o> [--encoder] [--generate --gen-oracle] --steps=N
// --checkpoint=$PWD/model-rfd4p-int8/model.safetensors` (that file from `tools/rfd4p-bundle-to-safetensors.mjs`:
// the int8 shards dequantised, so the oracle is built from the weights this port runs and the quantisation
// cancels). Run BOTH recycle counts - `--recycles=2` is the only arm that exercises the cycle projections.
//
// FOUR ARMS, and the first two are the ones that hold the kernels:
//   * every trunk stage FROM THE ORACLE'S OWN INPUT at 1e-6 - each kernel answering for its own arithmetic;
//   * every autoencoder layer the same way at 1e-5 (it amplifies: see the bounds below);
//   * the whole trunk and the decoder CHAINED, reported at 2e-6 and 1e-4;
//   * the sampler against the reference's own per-step trajectory at 1e-4.
#include "rfd4p.h"
#include <cmath>
#include <cstring>
#include <map>
#include <vector>
#include <fstream>
#include <filesystem>

extern const char* PORT_SOURCE;

// RFD4-Proteina's own featuriser, linked in rather than run as a second process
// (cuda/rfd4p/featurise/featurise.cpp, the same object cuda/rfd4p links).
namespace rfd4p { namespace featurise { int run(int argc, char** argv); } }

namespace {

double WORST_ABS = 0;
double relRms(const float* a, const float* b, size_t n) {
  double num = 0, den = 0, worst = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - b[i];
    if (!std::isfinite(d)) return INFINITY;
    num += d * d;
    den += (double)b[i] * b[i];
    worst = std::max(worst, d < 0 ? -d : d);
  }
  WORST_ABS = worst;
  return sqrt(num / std::max(den, 1e-30));
}

int FAILED = 0;
// ours (on the device, n values) against the oracle's "o/<name>". `bound` 0 or less REPORTS the number without
// gating on it - for a stage whose residual is its INPUT's, measured, and which has its own arm that is gated.
double check(const char* label, const std::string& name, const float* ours, size_t n, double bound) {
  // a name with its own prefix (the decoder's "d/...") is taken as it is; a bare one is the trunk's "o/..."
  const std::string key = name.find('/') == std::string::npos ? "o/" + name : name;
  if (!M.has(key)) { printf("  %-24s (not in the oracle)\n", label); return -1; }
  if (M.len(key) != n) die("the oracle's %s has %zu values, this port's %zu", key.c_str(), M.len(key), n);
  sync();
  const double r = relRms((const float*)host(ours), M.hostF(key), n);
  // maxAbs beside it, because that is what `tools/gpu/check-decoder.js` holds the page's own decoder to (1e-4 on
  // the float32 arm) and a number nobody can compare with the other ports' is a number that says less.
  if (bound <= 0) { printf("  %-24s relRMS %.3e  maxAbs %.2e  (reported)\n", label, r, WORST_ABS); return r; }
  const bool ok = r <= bound;
  if (!ok) FAILED += 1;
  printf("  %-24s relRMS %.3e  maxAbs %.2e  %s\n", label, r, WORST_ABS, ok ? "ok" : "FAILED");
  return r;
}

double checkFn(const char* label, const std::string& key, const float* ours, size_t n, double bound) {
  return check(label, key, ours, n, bound);
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, bundle, oracle, encDir, genDir, outPath, job, workDir, seed;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!strncmp(a, "--bundle=", 9)) bundle = a + 9;
    else if (!strncmp(a, "--oracle=", 9)) oracle = a + 9;
    else if (!strncmp(a, "--encoder=", 10)) encDir = a + 10;
    else if (!strncmp(a, "--gen=", 6)) genDir = a + 6;
    else if (!strncmp(a, "--out=", 6)) outPath = a + 6;
    else if (!strncmp(a, "--job=", 6)) job = a + 6;
    else if (!strncmp(a, "--work-dir=", 11)) workDir = a + 11;
    else if (!strncmp(a, "--seed=", 7)) seed = a + 7;
    else if (!strcmp(a, "--f16")) HALF = true;
    else if (a[0] != '-') input = a;
    else die("unknown flag %s", a);
  }
  // 🔴 `--job=<spec>` IS ONE PROCESS AND NO JAVASCRIPT. The featuriser is linked in, so a condition spec becomes
  // this port's input here - by the same code and the same bytes `rfd4p-featurise` writes, which
  // `npm run test:featurise`-style gates hold to the JavaScript's to the last place. The directory is still
  // written to disk, because that is what the port reads and what a second run reuses; what is gone is the
  // second process and the Node it needed. cuda/rfd4p's `jobMain` is the reading.
  if (!job.empty()) {
    if (bundle.empty()) { fprintf(stderr, "--job= needs --bundle=<model-rfd4p-int8>\n"); return 2; }
    if (workDir.empty()) workDir = job + ".featurised";
    // 🔴 THE FEATURISER READS THE PUBLISHED LAYOUT (`<dir>/shards/<file>`) AND A LOCAL EXPORT IS FLAT.
    // `tools/export_rfd4p_model.py` writes manifest.json beside its shards; the deployed bundle the port fetches
    // keeps them under `shards/`. Rather than teach one side the other's layout, a run stages a directory of
    // symlinks when it needs to - the bytes are not copied, and the bundle this port reads is untouched.
    std::string fbundle = bundle;
    {
      std::error_code ec;
      if (!std::filesystem::exists(bundle + "/shards", ec) && std::filesystem::exists(bundle + "/shard_000.bin", ec)) {
        const std::string stage = workDir + "-weights";
        std::filesystem::create_directories(stage + "/shards", ec);
        const std::filesystem::path src = std::filesystem::absolute(bundle);
        for (const char* f : {"manifest.json", "config.json"}) {
          std::filesystem::remove(stage + "/" + f, ec);
          std::filesystem::create_symlink(src / f, stage + "/" + f, ec);
        }
        for (const auto& e : std::filesystem::directory_iterator(src, ec)) {
          const std::string name = e.path().filename().string();
          if (name.size() < 4 || name.compare(name.size() - 4, 4, ".bin")) continue;
          std::filesystem::remove(stage + "/shards/" + name, ec);
          std::filesystem::create_symlink(e.path(), stage + "/shards/" + name, ec);
        }
        fbundle = stage;
      }
    }
    std::vector<std::string> fargs = {"rfd4p-featurise", workDir, "--spec=" + job, "--generate", "--int8=" + fbundle};
    if (!seed.empty()) fargs.push_back("--seed=" + seed);
    std::vector<char*> fargv;
    for (std::string& f : fargs) fargv.push_back(f.data());
    const double tf = now();
    if (const int rc = rfd4p::featurise::run((int)fargv.size(), fargv.data())) return rc;
    printf("featurised %s in %.2f s\n", job.c_str(), now() - tf);
    input = workDir;
    genDir = workDir + "-gen";
    if (outPath.empty()) outPath = "design.pdb";
    // 🔴 THE ENCODER'S INPUTS FOR A PARTIAL RUN AND ONLY FOR ONE: the featuriser writes `<work>-enc` when the
    // job is one and says so in the generation's metadata, and handing `--encoder=` to an ordinary run would
    // read whatever a previous job left in the same work directory.
    std::ifstream idx(genDir + "/model.idx");
    std::string kind, name;
    while (idx >> kind >> name) {
      if (kind != "m") { std::string rest; std::getline(idx, rest); continue; }
      double v = 0;
      idx >> v;
      if (name == "meta/partial" && v != 0) encDir = workDir + "-enc";
    }
  }

  if (input.empty() || bundle.empty()) {
    fprintf(stderr, "usage: localfold-rfd4p --job=<spec.json|.yaml> --bundle=<model-rfd4p-int8> [--out=design.pdb]\n"
                    "       localfold-rfd4p <input dir> --bundle=<model-rfd4p-int8> [--oracle=<dir>] "
                    "[--encoder=<dir>] [--gen=<dir>-gen] [--out=design.pdb] [--f16]\n");
    return 2;
  }

  setSource("rfd4p", PORT_SOURCE);
  const double t0 = now();
  loadWeights(bundle);
  checkDims();
  M.loadInput(input);
  if (!oracle.empty()) M.loadInput(oracle);
  // the encoder's own input directory and the reference's answer to it (`export_input.mjs --encoder`)
  // the encoder's input, and the reference's answer to it only where that was written (a job run has none)
  if (!encDir.empty()) {
    M.loadInput(encDir);
    if (FILE* probe = fopen((encDir + "-oracle/model.idx").c_str(), "r")) { fclose(probe); M.loadInput(encDir + "-oracle"); }
  }
  // a generation's exported inputs, and the reference's own trajectory over them where it was written
  if (!genDir.empty()) {
    M.loadInput(genDir);
    const std::string go = genDir + "-oracle";
    FILE* probe = fopen((go + "/model.idx").c_str(), "r");
    if (probe) { fclose(probe); M.loadInput(go); }
  }
  D_.n = (int)M.meta("meta/n");
  D_.nRecycle = (int)M.meta("meta/nRecycle", 1);
  const int n = D_.n;
  const size_t nn = (size_t)n * n;
  printf("rfd4p: %d tokens, %d recycle%s, bundle %s (%.2f GB of weights in %.2f s)\n", n, D_.nRecycle,
         D_.nRecycle == 1 ? "" : "s", bundle.c_str(), M.weightBytes() / 1e9, now() - t0);

  // 🔴 THE FEATURES ARE THE ORACLE'S OWN, which is what makes a residual this port's. The featuriser is
  // `shared/rfd4p/features.js` through the input directory; nothing about a molecule is decided here yet.
  const float* seqF = M.f("in/seq");
  const float* cond0 = M.f("in/cond0");
  const float* pairF = M.f("in/pair");
  const float* pairCondF = M.f("in/pairCond");

  // the attended tokens, as indices: `attn = padding_mask & ~is_in_context_conditioning`, which the input dir
  // carries ready-made. The autoencoder runs over these rows and nothing else.
  int attnCount = 0;
  int* attnIdx = nullptr;
  if (M.has("in/attnMask")) {
    const float* am = M.hostF("in/attnMask");
    std::vector<int> idx;
    for (int i = 0; i < n; ++i) if (am[i] != 0) idx.push_back(i);
    attnCount = (int)idx.size();
    attnIdx = uploadNew(idx.data(), idx.size());
  }

  float* c = allocT<float>((size_t)n * D_.C);
  float* sInit = allocT<float>((size_t)n * D_.D);
  float* zInit = allocT<float>(nn * D_.P);
  float* c1 = allocT<float>((size_t)n * D_.C);
  float* bbCaS = allocT<float>((size_t)n * 3);
  float* latentsS = allocT<float>((size_t)n * 32);
  float* logitsS = allocT<float>((size_t)n * 147);
  float* insertionS = allocT<float>(nn);

  condC(cond0, c, n);
  initSingle(seqF, sInit);
  initPair(pairF, pairCondF, zInit);

  float* s = allocT<float>((size_t)n * D_.D);

  if (!oracle.empty()) {
    printf("held to %s (cpu/rfd4p/reference.js over these same weights):\n", oracle.c_str());
    // 🔴 1e-6 FOR A FLOAT32 ARM, WHICH IS A REORDERING BOUND AND NOT A TOLERANCE ANYBODY PICKED: the reference
    // sums a row in source order on the host and the GEMM sums it in tiles, so the two differ by float
    // non-associativity and nothing else. `--f16` is a different question and will carry its own bound.
    const double bound = HALF ? 3e-3 : 1e-6;
    // 🔴 THE CHAINED ARM'S BOUND IS THE MEASURED ACCUMULATION, AND IT IS NOT A WIDER TOLERANCE FOR THE SAME
    // QUESTION. Arm 1 holds every kernel at `bound` on the reference's own input; this arm asks a different
    // thing - does fourteen layers of float32 drift stay small - and the answer is monotone in depth (1.0e-6 at
    // z10 over both recycle counts). 2e-6 is that with a factor of two, and a failure here with arm 1 green is
    // drift, where a failure in arm 1 is a kernel.
    const double chainedBound = HALF ? 6e-3 : 2e-6;
    // 🔴 THE AUTOENCODER'S PER-STAGE BOUND IS THE TRUNK'S, AND ITS CHAINED ONE IS NOT - BECAUSE THE DECODER
    // AMPLIFIES. Measured over 54 tokens, every decoder layer fed the reference's own input reads 7.4e-8 to
    // 9.9e-7 and is FLAT in depth, exactly as the trunk's are; chained, the same layers read 6.9e-7 at s0,
    // 6.7e-6 at s2 and 3.4e-5 by s4, then settle back to 4.5e-6 by s11. So the stack's own gain is above one on
    // this input and nothing is wrong with it: the per-stage arm is the gate and the chained arm is the report.
    // 🔴 AND THE AMPLIFICATION IS INPUT-DEPENDENT, which is the trap this nearly became: the same code on the
    // one-recycle input chains to 2.9e-6 and on the two-recycle input to 3.4e-5, from latents that differ by
    // 0.2 and a C_SEQ that differs in most positions while every magnitude stays the same to three figures. A
    // bound fitted on one input would have called the other a regression. Only the per-stage arm is stable
    // enough to hold a kernel to.
    // 🔴 AND ONE SUM IS WORTH COMPENSATING, measured rather than assumed: the attention's score dot product
    // runs over q and k that carry a LayerNorm's learned BIAS, so it is a product of vectors with a large
    // common component and what the softmax reads is the cancellation. Kahan-compensating that one sum took the
    // attention output from 3.43e-6 to 1.53e-6 and every stage below it with it (s0 1.26e-6 -> 5.9e-7, coors
    // 1.83e-6 -> 9.0e-7), while the same compensation on the softmax and the context sums moved NOTHING. The
    // trunk's q and k are RMSNormed and carry no bias, which is why its attention never needed it.
    // 🔴 1e-5 AND NOT THE TRUNK'S 1e-6, FROM TWO INPUTS RATHER THAN ONE. Over 54 tokens every decoder layer on
    // the reference's own input reads 7.4e-8 to 9.9e-7; over the 67-token partial-diffusion input the same
    // layers read ~4e-7 except LAYER 2, which reads 3.4e-6 - and layer 2 is the largest of the 54-token set
    // too (9.9e-7), so what is near-degenerate is that layer's input and not its code. A bug in a layer would
    // move every layer, since there is one `aeLayer`; this moves one, on both inputs, by the same ratio.
    const double decBound = HALF ? 3e-3 : 1e-5;
    // the chained decoder, reported at the level measured above with room for another machine and another input
    const double decChained = HALF ? 1e-2 : 1e-4;
    check("c", "c", c, (size_t)n * D_.C, bound);
    check("s_init", "sInit", sInit, (size_t)n * D_.D, bound);
    check("z_init", "zInit", zInit, nn * D_.P, bound);
    // 🔴 EVERY STAGE COMPARED THE MOMENT IT IS PRODUCED, so the first one that moves is the one named - and the
    // pair track is compared after each update rather than only at the end, which is how `cuda/rfd4p` found three
    // of its bugs (its README: "o/s0 rather than a finished trunk that was merely wrong").
    float* z = allocT<float>(nn * D_.P);

    // ---- arm 1: EVERY STAGE FROM THE ORACLE'S OWN INPUT ----------------------------------------------------
    // 🔴 THIS IS THE ARM THAT HOLDS THE KERNELS, AND THE CHAINED ONE BELOW CANNOT. Chained, a stage is handed
    // what the stage before it computed, so what reaches layer 13 is fourteen layers of accumulated float
    // reordering - the reference accumulates every dot product in float64 (`linearLayer` sums into a JS number
    // and stores to a Float32Array) where the GEMM accumulates in float32 and in tiles, so the two drift apart
    // monotonically with depth: measured 2.9e-7 after layer 0 and 1.0e-6 on the pair track after layer 10.
    // Fed the reference's own input, each stage answers for its own arithmetic alone - which is how a defect in
    // layer 11 stops being indistinguishable from drift, and is what metal/bioemu2's oracle arm does.
    {
      const float* cc = D_.nRecycle > 1 ? (condC(M.f("in/cond1"), c1, n), c1) : c;
      const char* zFrom = "zInit";
      // 🔴 LAYER 0 IS HELD ONLY AT ONE RECYCLE, AND THE REASON IS THE ORACLE'S AND NOT THIS PORT'S: the stages
      // it records are the LAST cycle's, and that cycle's layer 0 reads s_init + Lin(RMSNorm(s of the cycle
      // before)) rather than s_init itself. Nothing in the dump is that tensor, so feeding this arm `o/sInit` at
      // two recycles compares the right kernel against the wrong input - measured 1.8e-1, which is what said so.
      // The chained arm holds it instead, and s1 onwards are held here either way.
      const int first = D_.nRecycle > 1 ? 1 : 0;
      if (first) printf("  %-24s (the oracle records the last cycle only - held by the chained arm)\n", "stage s0, z0");
      if (first) { zFrom = "z0"; }
      for (int i = first; i < D_.layers; ++i) {
        const std::string sFrom = i == 0 ? "o/sInit" : "o/s" + std::to_string(i - 1);
        copy(s, M.f(sFrom), (size_t)n * D_.D * 4);
        copy(z, M.f(std::string("o/") + zFrom), nn * D_.P * 4);
        tokenLayer(i, s, z, cc);
        check(("stage s" + std::to_string(i)).c_str(), "s" + std::to_string(i), s, (size_t)n * D_.D, bound);
        if (i < D_.layers - 1 && i % 2 == 0) {
          copy(s, M.f("o/s" + std::to_string(i)), (size_t)n * D_.D * 4);
          pairUpdate(i, s, z);
          check(("stage z" + std::to_string(i)).c_str(), "z" + std::to_string(i), z, nn * D_.P, bound);
          zFrom = strdup(("z" + std::to_string(i)).c_str());
        }
      }
      copy(s, M.f("o/s"), (size_t)n * D_.D * 4);
      heads(s, bbCaS, latentsS, logitsS, insertionS);
      check("stage bb_ca", "bbCa", bbCaS, (size_t)n * 3, bound);
      check("stage local_latents", "localLatents", latentsS, (size_t)n * 32, bound);
      check("stage logits", "logits", logitsS, (size_t)n * 147, bound);
      check("stage insertion", "insertion", insertionS, nn, bound);
    }

    // ---- arm 2: the whole trunk, chained -------------------------------------------------------------------
    // 🔴 EVERY CYCLE BUT THE LAST IS RUN WITHOUT COMPARING ANYTHING, because the oracle records the LAST cycle's
    // stages: a two-recycle run writes one s0..s13 and it is the second pass's. So the stages of the earlier
    // cycles are held only by the one they produce - which is why `--recycles=2` is the arm that exercises the
    // cycle projections and the second conditioning at all, and why cuda/rfd4p runs both.
    // 🔴 ITS OWN BUFFERS, ZEROED BY THE ALLOCATOR, AND NOT ARM 1'S ZEROED AGAIN. The first cycle's answer is a
    // property of the state being zero (RMSNorm(0) is 0 and the projections have no bias, so the cycle's input
    // IS the init product), and arm 1 leaves its own s and z in those buffers - which is how this read relRMS
    // 1.9e-1 once, a plausible structure with every stage wrong and nothing in the recycle code touched.
    // 🔴 AND ZEROING THEM WITH `fill` IS NOT THE FIX, WHICH COST AN HOUR TO SEE: `fill` memsets on the HOST
    // when the device looks idle, so it can land while the dispatches that wrote those buffers are still in
    // flight - the GPU's writes then arrive AFTER the zeroing and the state is not zero at all. It showed as a
    // chained trunk that was right seven runs in eight and 7.2e-5 in the eighth, from the same bytes: a
    // difference between runs is a race, never a precision question. A fresh allocation cannot be raced.
    float* sc = allocT<float>((size_t)n * D_.D);
    float* zc = allocT<float>(nn * D_.P);
    s = sc;
    z = zc;
    for (int cycle = 0; cycle < D_.nRecycle; ++cycle) {
      recycleIn(sInit, zInit, s, z);
      // the conditioning of every cycle after the first is built from the `recycleRemaining` features
      const float* cc = c;
      if (cycle > 0) {
        if (!M.has("in/cond1")) die("a %d-recycle input with no in/cond1", D_.nRecycle);
        condC(M.f("in/cond1"), c1, n);
        cc = c1;
      }
      const bool last = cycle + 1 == D_.nRecycle;
      for (int i = 0; i < D_.layers; ++i) {
        tokenLayer(i, s, z, cc);
        if (last) check(("chained s" + std::to_string(i)).c_str(), "s" + std::to_string(i), s, (size_t)n * D_.D, chainedBound);
        if (i < D_.layers - 1 && i % 2 == 0) {
          pairUpdate(i, s, z);
          if (last) check(("chained z" + std::to_string(i)).c_str(), "z" + std::to_string(i), z, nn * D_.P, chainedBound);
        }
      }
    }
    float* bbCa = allocT<float>((size_t)n * 3);
    float* latents = allocT<float>((size_t)n * 32);
    float* logits = allocT<float>((size_t)n * 147);
    float* insertion = allocT<float>(nn);
    heads(s, bbCa, latents, logits, insertion);
    check("chained bb_ca", "bbCa", bbCa, (size_t)n * 3, chainedBound);
    check("chained local_latents", "localLatents", latents, (size_t)n * 32, chainedBound);
    check("chained logits", "logits", logits, (size_t)n * 147, chainedBound);
    // 🔴 REPORTED, NOT GATED, AND THE ARM BELOW IS WHAT GATES IT. Measured: fed the reference's own single
    // track this head reads 4.8e-7, and fed ours it reads 1.5e-6 - a dot product over 768 channels of a track
    // carrying 6.3e-7, with the cancellation a difference of large sums brings, so it amplifies its input about
    // 2.4x. Holding the end-to-end number to the per-stage bound would be asking this kernel to be more exact
    // than its input, and widening the bound for every stage to let it through would hide the stages that
    // SHOULD hold. So the amplification is stated and the head is gated where it answers for itself.
    check("chained insertion", "insertion", insertion, nn, 0);
    check("chained s", "s", s, (size_t)n * D_.D, chainedBound);
    check("chained z", "z", z, nn * D_.P, chainedBound);
    // ---- the autoencoder's decoder ------------------------------------------------------------------------
    // 🔴 FED THE ORACLE'S OWN LATENTS AND ALPHA CARBONS (`in/zLatent`, `in/caNm`), not the trunk's, for the same
    // reason every stage above is: the decoder is a separate network and its residual should be its own. The
    // trunk's own latents reach it in a generation, where the sampler closes the loop.
    if (M.has("d/feats") && M.has("in/zLatent")) {
      checkDecoderDims();
      const int na = attnCount;
      float* feats = allocT<float>((size_t)n * DD_.feat);
      decoderFeatures(M.hostF("in/caNm"), M.hostF("in/zLatent"), M.hostI("in/tokenType"), M.hostI("in/decSeq"),
                      M.hostF("in/mask"), feats);
      check("decoder feats", "d/feats", feats, (size_t)n * DD_.feat, decBound);

      float* sd = allocT<float>((size_t)na * DD_.D);
      float* sInitD = allocT<float>((size_t)na * DD_.D);
      float* perLayer = allocT<float>((size_t)DD_.layers * na * DD_.D);
      float* ada0 = allocT<float>((size_t)na * DD_.D);
      float* attn0 = allocT<float>((size_t)na * DD_.D);
      float* dLogits = allocT<float>((size_t)na * DD_.types);
      float* dCoors = allocT<float>((size_t)na * 111);
      float* dPresence = allocT<float>((size_t)na * 37);
      DEC_TAP_ADA = ada0;
      DEC_TAP_ATTN = attn0;
      decode(feats, attnIdx, na, sd, dLogits, dCoors, dPresence, sInitD, perLayer);
      DEC_TAP_ADA = DEC_TAP_ATTN = nullptr;

      // 🔴 SCATTERED BACK BEFORE COMPARING, because the oracle's rows are the INPUT's and this port's are the
      // ATTENDED ones. They coincide whenever nothing is in context - which is every case this preset has - and
      // a gate that only ever ran there would not notice the day they stop.
      // 🔴 AND EVERY DECODER LAYER FROM THE ORACLE'S OWN INPUT, for the reason the trunk has the same arm: fed
      // what the stage before it computed, a layer's residual is its own plus everything above it, and this
      // stack amplifies. Measured here, it is what separates the two.
      {
        float* sl = scratch<float>("gate.sl", (size_t)na * DD_.D);
        float* cz = scratch<float>("gate.cz", (size_t)na * DD_.C);
        fill(cz, 0, (size_t)na * DD_.C * 4);
        for (int l = 0; l < DD_.layers; ++l) {
          const std::string from = l == 0 ? "d/sInit" : "d/s" + std::to_string(l - 1);
          gatherRows(M.f(from), sl, attnIdx, na, DD_.D);
          decoderLayerOne(l, sl, cz, na);
          float* out = scratch<float>("gate.scatter2", (size_t)n * DD_.D);
          fill(out, 0, (size_t)n * DD_.D * 4);
          scatterRows(sl, out, attnIdx, na, DD_.D);
          check(("decoder stage s" + std::to_string(l)).c_str(), "d/s" + std::to_string(l), out,
                (size_t)n * DD_.D, decBound);
        }
      }

      float* full = allocT<float>((size_t)n * DD_.D);
      auto scattered = [&](const float* src, int width) {
        float* out = scratch<float>("gate.scatter", (size_t)n * width);
        fill(out, 0, (size_t)n * width * 4);
        scatterRows(src, out, attnIdx, na, width);
        return out;
      };
      (void)full;
      check("decoder s_init", "d/sInit", scattered(sInitD, DD_.D), (size_t)n * DD_.D, decBound);
      // layer 0's two internals: its input is the exact s_init, so these are per-stage numbers too
      check("decoder ada0", "d/ada0", scattered(ada0, DD_.D), (size_t)n * DD_.D, decBound);
      check("decoder attn0", "d/attn0", scattered(attn0, DD_.D), (size_t)n * DD_.D, decChained);
      for (int l = 0; l < DD_.layers; ++l)
        check(("decoder chained s" + std::to_string(l)).c_str(), "d/s" + std::to_string(l),
              scattered(perLayer + (size_t)l * na * DD_.D, DD_.D), (size_t)n * DD_.D, decChained);
      check("decoder chained logits", "d/logits", scattered(dLogits, DD_.types), (size_t)n * DD_.types, decChained);
      check("decoder chained coors", "d/coors", scattered(dCoors, 111), (size_t)n * 111, decChained);
      check("decoder chained presence", "d/presence", scattered(dPresence, 37), (size_t)n * 37, decChained);
    }

    // ---- the autoencoder's encoder ------------------------------------------------------------------------
    if (M.has("e/feats")) {
      checkEncoderDims();
      const int na = attnCount;
      float* ef = allocT<float>((size_t)n * ED_.feat);
      encoderFeatures(n, M.hostF("e/coorsNm"), M.hostF("e/atomMask"), M.hostF("e/chainIid"), M.hostF("e/bondCounts"),
                      M.hostI("e/element"), M.hostI("e/charge"), ef);
      check("encoder feats", "e/feats", ef, (size_t)n * ED_.feat, bound);
      float* mean = allocT<float>((size_t)na * ED_.latent);
      float* logScale = allocT<float>((size_t)na * ED_.latent);
      encode(ef, attnIdx, na, mean, logScale);
      // 🔴 THE LATENTS ARE EMITTED ROWS ONLY, so the comparison is against the oracle's own emit mask and not
      // the attention's: `emit = attn & supports_local_latent`, and a token that supports no latent has zeros
      // there in the reference. Scattering by `emit` is what makes the two tables the same table.
      const float* emit = M.hostF("e/emit");
      std::vector<int> eidx;
      for (int i = 0; i < n; ++i) if (emit[i] != 0) eidx.push_back(i);
      int* ei = uploadNew(eidx.data(), eidx.size());
      auto scatterEmitC = [&](const float* src, const char* name) {
        float* out = scratch<float>(std::string("gate.") + name, (size_t)n * ED_.latent);
        fill(out, 0, (size_t)n * ED_.latent * 4);
        // the emitted rows are a subset of the attended ones, in the same order
        float* gathered = scratch<float>("gate.eg", eidx.size() * ED_.latent);
        std::vector<int> within;
        {
          const float* am = M.hostF("in/attnMask");
          int k = 0;
          for (int i = 0; i < n; ++i) { if (am[i] == 0) continue; if (emit[i] != 0) within.push_back(k); ++k; }
        }
        int* wi = uploadNew(within.data(), within.size());
        gatherRows(src, gathered, wi, within.size(), ED_.latent);
        scatterRows(gathered, out, ei, eidx.size(), ED_.latent);
        return out;
      };
      // 🔴 THE ENCODER'S OUTPUT IS A TWELVE-LAYER CHAIN AND IS HELD AS ONE. There are no per-layer encoder
      // stages in the dump, and there do not need to be: its layers ARE the decoder's (one `aeLayer`, the same
      // prefix away), which the decoder's per-stage arm gates layer by layer. What is left for this arm to say
      // is whether the chain and the head are wired right, at the chained level the decoder measures.
      check("encoder mean", "e/mean", scatterEmitC(mean, "mean"), (size_t)n * ED_.latent, decChained);
      check("encoder logScale", "e/logScale", scatterEmitC(logScale, "ls"), (size_t)n * ED_.latent, decChained);
      // the deterministic latent IS the mean, which is what `e/zLatent` holds in this arm
      check("encoder zLatent", "e/zLatent", scatterEmitC(mean, "zl"), (size_t)n * ED_.latent, decChained);
    }

    printf("%s so far (%.2f s)\n", FAILED ? "FAILED" : "every stage in band", now() - t0);
  }

  // ---- the flow sampler ---------------------------------------------------------------------------------
  if (!genDir.empty()) {
    checkDecoderDims();
    GATE_CHECK = checkFn;
    // 🔴 THE TRAJECTORY'S BOUND IS NOT A STAGE'S. Every step feeds the next, and the backbone's first step
    // divides by t = 1e-3, so what a stage holds to 1e-6 arrives here multiplied; the per-step record is what
    // says whether that is amplification (the numbers grow smoothly) or a wiring fault (one step jumps).
    GATE_TRAJ_BOUND = HALF ? 1e-2 : 1e-4;
    const int ran = generate(genDir, outPath);
    printf("  the sampler ran %d steps\n", ran);
    GATE_CHECK = nullptr;
  }


  if (!oracle.empty() || !genDir.empty()) {
    printf("%s (%.2f s)\n", FAILED ? "FAILED" : "all green", now() - t0);
    return FAILED ? 1 : 0;
  }
  printf("the trunk's init products built in %.2f s (no --oracle= to hold them to)\n", now() - t0);
  return 0;
}
