// RFD4-Proteina on Metal: the trunk, held stage by stage to the CPU reference over the same weights.
//
//   bash metal/build.sh rfd4p
//   node tools/rfd4p-bundle-to-safetensors.mjs                      # the int8 shards as float32, once
//   node cuda/rfd4p/export_input.mjs <in> --oracle=<oracle> --recycles=1 \
//        --checkpoint=$PWD/model-rfd4p-int8/model.safetensors        # the input and the reference's 37 stages
//   ./metal/rfd4p/localfold-rfd4p <in> --bundle=model-rfd4p-int8 --oracle=<oracle>
//
// <input dir> is `cuda/rfd4p/export_input.mjs`'s output (model.idx / model.bin: the trunk's features and the
// padding mask); --oracle= is that run's `--oracle=` directory. Run BOTH recycle counts: `--recycles=2` is the
// only arm that exercises the cycle projections and the second conditioning at all.
//
// TWO ARMS, and the first is the one that holds the kernels:
//   * every stage FROM THE ORACLE'S OWN INPUT, at 1e-6 - each kernel answering for its own arithmetic;
//   * the whole trunk CHAINED, at 2e-6 - fourteen layers of float32 against a float64-accumulating reference.
// Measured on an M5, 54 tokens: per stage 2.1e-7 to 5.8e-7 and FLAT in depth, chained 2.9e-7 growing to 1.0e-6
// at z10. A failure in the first arm is a kernel; a failure in the second with the first green is drift.
#include "rfd4p.h"
#include <cmath>
#include <cstring>
#include <map>

extern const char* PORT_SOURCE;

namespace {

double relRms(const float* a, const float* b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - b[i];
    if (!std::isfinite(d)) return INFINITY;
    num += d * d;
    den += (double)b[i] * b[i];
  }
  return sqrt(num / std::max(den, 1e-30));
}

int FAILED = 0;
// ours (on the device, n values) against the oracle's "o/<name>". `bound` 0 or less REPORTS the number without
// gating on it - for a stage whose residual is its INPUT's, measured, and which has its own arm that is gated.
double check(const char* label, const std::string& name, const float* ours, size_t n, double bound) {
  const std::string key = "o/" + name;
  if (!M.has(key)) { printf("  %-24s (not in the oracle)\n", label); return -1; }
  if (M.len(key) != n) die("the oracle's %s has %zu values, this port's %zu", key.c_str(), M.len(key), n);
  sync();
  const double r = relRms((const float*)host(ours), M.hostF(key), n);
  if (bound <= 0) { printf("  %-24s relRMS %.3e  (reported)\n", label, r); return r; }
  const bool ok = r <= bound;
  if (!ok) FAILED += 1;
  printf("  %-24s relRMS %.3e  %s\n", label, r, ok ? "ok" : "FAILED");
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, bundle, oracle;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!strncmp(a, "--bundle=", 9)) bundle = a + 9;
    else if (!strncmp(a, "--oracle=", 9)) oracle = a + 9;
    else if (!strcmp(a, "--f16")) HALF = true;
    else if (a[0] != '-') input = a;
    else die("unknown flag %s", a);
  }
  if (input.empty() || bundle.empty()) {
    fprintf(stderr, "usage: localfold-rfd4p <input dir> --bundle=<model-rfd4p-int8> [--oracle=<dir>] [--f16]\n");
    return 2;
  }

  setSource("rfd4p", PORT_SOURCE);
  const double t0 = now();
  loadWeights(bundle);
  checkDims();
  M.loadInput(input);
  if (!oracle.empty()) M.loadInput(oracle);
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
    // 🔴 ZEROED EXPLICITLY, BECAUSE THE FIRST CYCLE'S ANSWER IS A PROPERTY OF THE STATE BEING ZERO. `alloc`
    // hands back zeroed memory, so this was invisible until arm 1 above ran first and left its own s and z in
    // these buffers: the cycle projection then had something to project and the whole trunk came back at relRMS
    // 1.9e-1 - a plausible structure, every stage wrong, and nothing in the recycle code touched. A fold that
    // depends on a buffer starting at zero says so.
    fill(s, 0, (size_t)n * D_.D * 4);
    fill(z, 0, nn * D_.P * 4);
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
    printf("%s (%.2f s)\n", FAILED ? "FAILED" : "all stages in band", now() - t0);
    return FAILED ? 1 : 0;
  }
  printf("the trunk's init products built in %.2f s (no --oracle= to hold them to)\n", now() - t0);
  return 0;
}
