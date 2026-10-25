// The native CUDA port of RFD4-Proteina.
//
//   localfold-rfd4p <input dir> --weights=<dir> [--oracle=<dir>] [--out=<path>]
//
// `<input dir>` is `cuda/rfd4p/export_input.mjs`'s `model.idx`/`model.bin`: the trunk's features for
// one sampler step. `--weights=` is `export_weights.mjs`'s float32 directory (the int8 bundle through
// `Model::loadBundle` is the shipping path and is not wired yet). `--oracle=` is the CPU reference's
// answer stage by stage, and turns this into the differential that gates the port.
//
// 🔴 WHAT THIS IS NOT, YET: a fold. There is no sampler, no decoder and no output structure - this
// runs ONE trunk pass and compares it. The flow loop (`shared/rfd4p/flow.js`), the autoencoder decoder
// and the all-atom output come next, and each wants its own stage-by-stage bar before it is believed.
#include "trunk.cuh"
#include "decoder.cuh"
#include "sampler.cuh"
#include "encoder.cuh"
#include "generate.cuh"
#include "bundle.cuh"
#include <fstream>
#include <chrono>

namespace {

struct Opts {
  std::string dir, weights, oracle, gen, genOracle, enc, encOracle, int8, out = "design.pdb";
  int recycles = 0;  // 0 = whatever the input directory says
  int repeat = 0;    // time the forward this many times after a warm-up
  int designs = 1;   // generate this many times, so a WARM one can be timed
  bool fast = false; // TF32 on the tensor cores, as every other port's --fast does
  bool insertion = true;
};

int generate(const Opts& o, const rfd4p::Dims& d, rfd4p::Trunk& t);

int run(const Opts& o) {
  if (!o.weights.empty()) M.load(o.weights);
  M.load(o.dir);
  if (!o.oracle.empty()) M.load(o.oracle);
  if (!o.gen.empty()) M.load(o.gen);
  if (!o.genOracle.empty()) M.load(o.genOracle);
  if (!o.enc.empty()) M.load(o.enc);
  if (!o.encOracle.empty()) M.load(o.encOracle);
  for (int seg = 0; seg < (int)M.segs.size(); ++seg) M.upload(seg);
  CB(cublasCreate(&H));
  CB(cublasSetStream(H, STREAM));
  // The published int8 bundle, decoded on the device into the same names the float32 export writes.
  if (!o.int8.empty()) rfd4p::loadInt8Bundle(o.int8, o.int8 + "/shards");
  F32_TF32 = o.fast;
  CB(cublasSetMathMode(H, o.fast ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));

  rfd4p::Dims d;
  d.n = (int)M.meta("meta/n");
  d.D = (int)M.meta("meta/tokenDim");
  d.P = (int)M.meta("meta/pairDim");
  d.C = (int)M.meta("meta/condDim");
  d.PC = (int)M.meta("meta/pairCondDim");
  d.TRI = (int)M.meta("meta/triMultC");
  d.H = (int)M.meta("meta/nheads");
  d.nlayers = (int)M.meta("meta/nlayers");
  d.everyN = (int)M.meta("meta/updatePairEveryN");
  d.useTriMult = M.meta("meta/useTriMult") != 0;
  d.updatePairRepr = M.meta("meta/updatePairRepr") != 0;
  d.classes = (int)M.meta("meta/seqReadoutClasses");
  d.latent = (int)M.meta("meta/latentDim");
  d.seqF = (int)M.meta("meta/seqFeatDim");
  d.condF = (int)M.meta("meta/condFeatDim");
  d.pairF = (int)M.meta("meta/pairFeatDim");
  d.pairCondF = (int)M.meta("meta/pairCondFeatDim");
  const int nRecycle = o.recycles > 0 ? o.recycles : (int)M.meta("meta/nRecycle");
  const size_t n = d.n, pairs = d.pairs();
  printf("rfd4p: %d tokens, %d layers, %d recycle%s\n", d.n, d.nlayers, nRecycle, nRecycle == 1 ? "" : "s");

  rfd4p::Trunk t;
  t.d = d;
  t.mask = Fdev("in/mask");
  float* pairMask = scratch<float>("rfd4p.pairMask", pairs);
  rfd4p::pairMaskK<<<blocks(pairs), 256, 0, STREAM>>>(t.mask, pairMask, d.n);
  t.pairMask = pairMask;
  t.s = scratch<float>("rfd4p.s", n * d.D);
  // The pair tensor is half under --fast; `forward` picks and casts.
  t.z = o.fast ? (void*)scratch<half>("rfd4p.z16", pairs * d.P)
               : (void*)scratch<float>("rfd4p.z", pairs * d.P);
  t.bbCa = scratch<float>("rfd4p.bbCa", n * 3);
  t.localLatents = scratch<float>("rfd4p.ll", n * d.latent);
  t.logits = scratch<float>("rfd4p.logits", n * d.classes);
  t.insertion = scratch<float>("rfd4p.insertion", pairs);

  // Every stage is compared the moment it is produced, so the first one that moves is the one named -
  // a later stage's residual is its own fault plus everything upstream of it.
  // 🔴 THE BAR FOLLOWS THE ARITHMETIC. TF32 carries ten mantissa bits, so the exact path's 1e-5 would
  // fail a --fast run that is working perfectly; these are the bounds the other ports' gates use.
  //
  // 🔴 AND THE DECODER'S FLOOR IS HIGHER THAN THE TRUNK'S, FOR A REASON THAT WAS MEASURED RATHER
  // THAN ASSUMED. Its stages read up to 3.1e-05 where the trunk reads 5.8e-07, and three
  // measurements say that is the comparison's floor and not a defect: its features are BIT-identical
  // (relRMS exactly 0), its adaLN is 1.3e-07, and the error enters at the attention output, 4.3e-06.
  // A one-ulp nudge of the decoder's latent comes out at x1, so nothing here amplifies - what
  // differs is that the reference computes its attention in JS doubles with a two-pass softmax where
  // flash is f32 with an online rescale, and the decoder's softmax is NEAR-UNIFORM over every key
  // where the trunk's is sharpened by its pair bias. 4e-6 is what a 64-dim f32 dot and a 54-key
  // average should differ by. Same shape as docs/PARITY.md's note that an f64 reference cannot agree
  // with an f32 oracle past a model-dependent floor. The decoder therefore runs in f32 on BOTH arms
  // (see decoder.cuh) and takes one bound.
  auto boundFor = [&](const char* name) {
    // 🔴 THE int8 BUNDLE IS HELD TO ITS OWN QUANTISATION, which is upstream's choice and not this
    // port's arithmetic: read against an oracle computed from the FLOAT checkpoint it lands at
    // 1.6e-02, and the structure it generates is 0.15 A of CA from the float one at the same seed.
    if (!o.int8.empty()) return 2.5e-2;
    // The autoencoder's two halves run f32 on both arms (decoder.cuh, encoder.cuh), so one
    // bound serves them and it does not follow --fast.
    const bool autoenc = (name[0] == 'd' || name[0] == 'e') && name[1] == '/';
    if (autoenc) return 1e-4;
    return o.fast ? 5e-3 : 1e-5;
  };
  const double bound = o.fast ? 5e-3 : 1e-5;
  int bad = 0;
  double worst = 0;
  std::string worstAt;
  std::function<void(const char*, const float*, size_t)> tap;
  if (!o.oracle.empty() || !o.encOracle.empty()) {
    tap = [&](const char* name, const float* p, size_t count) {
      // 🔴 relRMS ALONE CANNOT TELL A GROWING ERROR FROM A SHRINKING SIGNAL, and the decoder's
      // stages bulge in the middle of the stack and come back - which is one or the other. The
      // oracle's own RMS is printed beside it, so the absolute error is visible.
      double r = checkOracle(name, p, count, name);
      if (r >= 0 && getenv("LOCALFOLD_RFD4P_ABS")) {
        const float* ref = M.f(name);
        double sq = 0;
        for (size_t i = 0; i < count; ++i) sq += (double)ref[i] * ref[i];
        const double rms = sqrt(sq / count);
        printf("      |oracle| rms %.4e, absolute error %.3e\n", rms, r * rms);
      }
      if (r >= 0 && r > worst) { worst = r; worstAt = name; }
      if (r > boundFor(name)) ++bad;
    };
  }

  // The autoencoder ENCODER, when the input directory carries its tensors: clean all-atom
  // coordinates into the local latents a partial-diffusion or inverse-folding run starts from.
  // 🔴 ITS ORACLE IS THE DETERMINISTIC ARM (z is the mean), because the sampled one would compare
  // two Box-Muller implementations rather than the encoder - see export_input.mjs.
  if (!o.enc.empty()) {
    rfd4p::EncDims ed;
    ed.n = d.n;
    ed.D = (int)M.meta("meta/encTokenDim");
    ed.C = (int)M.meta("meta/encCondDim");
    ed.H = (int)M.meta("meta/encHeads");
    ed.nlayers = (int)M.meta("meta/encLayers");
    ed.latent = d.latent;
    ed.fdim = (int)M.meta("meta/encFeatDim");
    ed.ttDim = (int)M.meta("meta/encTtDim");
    ed.chargeDim = (int)M.meta("meta/encChargeDim");
    const int offsets[7] = {
        (int)M.meta("meta/encOffAbs"),   (int)M.meta("meta/encOffRel"),
        (int)M.meta("meta/encOffChain"), (int)M.meta("meta/encOffBonds"),
        (int)M.meta("meta/encOffElCh"),  (int)M.meta("meta/encTtCreator"),
        (int)M.meta("meta/encElChCreator")};
    float* mean = scratch<float>("rfd4p.enc.mean", (size_t)ed.n * ed.latent);
    float* logScale = scratch<float>("rfd4p.enc.ls", (size_t)ed.n * ed.latent);
    float* zLat = scratch<float>("rfd4p.enc.z", (size_t)ed.n * ed.latent);
    rfd4p::encode(ed, Fdev("e/coorsNm"), Fdev("e/atomMask"), Idev("in/tokenType"),
                  Fdev("e/chainIid"), Fdev("e/bondCounts"), Idev("e/element"), Idev("e/charge"),
                  Fdev("in/mask"), Fdev("in/attnMask"), Fdev("e/emit"), nullptr, mean, logScale,
                  zLat, offsets, o.encOracle.empty() ? nullptr : tap);
    CK(cudaStreamSynchronize(STREAM));
    if (!o.encOracle.empty()) {
      printf("encoder: %s, worst %.3e at %s\n", bad ? "FAILED" : "ok", worst, worstAt.c_str());
      return bad ? 1 : 0;
    }
  }

  if (!o.gen.empty()) return generate(o, d, t);

  rfd4p::forward(t, Fdev("in/seq"), Fdev("in/cond0"), Fdev("in/cond1"), Fdev("in/pair"),
                 Fdev("in/pairCond"), nRecycle, o.insertion, tap);

  // The decoder, when the input directory carries its tensors. It takes the trunk's latent and the
  // sequence that trunk's logits chose - both exported here so the two sides decode the SAME input
  // rather than each its own.
  if (M.has("in/caNm") && M.has("in/zLatent")) {
    rfd4p::DecDims dd;
    dd.n = d.n;
    dd.D = (int)M.meta("meta/decTokenDim");
    dd.C = (int)M.meta("meta/decCondDim");
    dd.H = (int)M.meta("meta/decHeads");
    dd.nlayers = (int)M.meta("meta/decLayers");
    dd.nTypes = (int)M.meta("meta/nResidueTypes");
    dd.latent = d.latent;
    dd.fdim = (int)M.meta("meta/decFeatDim");
    float* dLogits = scratch<float>("rfd4p.out.logits", n * (size_t)dd.nTypes);
    float* dCoors = scratch<float>("rfd4p.out.coors", n * 111);
    float* dPresence = scratch<float>("rfd4p.out.presence", n * 37);
    rfd4p::decode(dd, Fdev("in/caNm"), Fdev("in/zLatent"), Idev("in/tokenType"), Idev("in/decSeq"),
                  Fdev("in/attnMask"), dLogits, dCoors, dPresence, tap);
  }

  if (tap) {
    // o/z is emitted inside forward(), where the pair tensor's type is known.
    for (const char* name : {"o/bbCa", "o/localLatents", "o/logits", "o/insertion", "o/s"}) {
      const float* p = !strcmp(name, "o/bbCa") ? t.bbCa
                     : !strcmp(name, "o/localLatents") ? t.localLatents
                     : !strcmp(name, "o/logits") ? t.logits
                     : !strcmp(name, "o/insertion") ? t.insertion
                     : t.s;
      size_t count = !strcmp(name, "o/bbCa") ? n * 3
                   : !strcmp(name, "o/localLatents") ? n * d.latent
                   : !strcmp(name, "o/logits") ? n * (size_t)d.classes
                   : !strcmp(name, "o/insertion") ? pairs
                   : n * d.D;
      tap(name, p, count);
    }
  }
  CK(cudaStreamSynchronize(STREAM));

  // 🔴 TIME THE STEADY STATE, NOT THE FIRST PASS. The run above has already paid for the weights,
  // every scratch buffer and cuBLAS's own first-call setup; a number that includes those describes
  // starting the process, which is not what a sampler's sixtieth step costs.
  if (o.repeat > 0) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, STREAM));
    for (int i = 0; i < o.repeat; ++i) {
      rfd4p::forward(t, Fdev("in/seq"), Fdev("in/cond0"), Fdev("in/cond1"), Fdev("in/pair"),
                     Fdev("in/pairCond"), nRecycle, o.insertion, nullptr);
    }
    CK(cudaEventRecord(b, STREAM));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    printf("trunk: %.3f ms a pass (%d tokens, %d recycle%s, %d passes)\n", ms / o.repeat, d.n,
           nRecycle, nRecycle == 1 ? "" : "s", o.repeat);
  }
  if (!o.oracle.empty()) {
    printf("worst stage %s at relRMS %.2e; %d stage%s past their bound (trunk %.0e, decoder %.0e)\n",
           worstAt.empty() ? "(none)" : worstAt.c_str(), worst, bad, bad == 1 ? "" : "s", bound,
           boundFor("d/"));
  }
  return bad == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- a whole generation
//
// The flow loop: sixty steps of trunk, each advancing x by one Euler-Maruyama step, then one
// decode of the finished state into atoms.
int generate(const Opts& o, const rfd4p::Dims& d, rfd4p::Trunk& t) {
  const int n = d.n;
  const int steps = (int)M.meta("meta/steps");
  const int condDim = d.condF, pairDim = d.pairF;
  const std::vector<float> emitBb = download(Fdev("g/emitBb"), n);
  const std::vector<float> emitLl = download(Fdev("g/emitLl"), n);
  const std::vector<float> noise = download(Fdev("g/noise"), M.len("g/noise"));
  const int* depths = M.i("g/depths");
  const int* stochastic = M.i("g/stochastic");
  const int* coordCond = M.i("g/coordCond");
  const int* moleculeIid = M.i("g/moleculeIid");

  // `sample_noise` for both modalities: standard normal on the emitting tokens, the backbone
  // centred and then offset once a molecule - all of it replayed from the exporter's own draws.
  std::vector<float> bb(noise.begin(), noise.begin() + (size_t)n * 3);
  size_t at = (size_t)n * 3;
  double mean[3] = {0, 0, 0};
  int live = 0;
  for (int i = 0; i < n; ++i) {
    if (!emitBb[i]) { bb[i * 3] = bb[i * 3 + 1] = bb[i * 3 + 2] = 0.f; continue; }
    ++live;
    for (int k = 0; k < 3; ++k) mean[k] += bb[i * 3 + k];
  }
  for (int k = 0; k < 3; ++k) mean[k] = (float)(mean[k] / std::max(live, 1));
  for (int i = 0; i < n; ++i) {
    if (!emitBb[i]) continue;
    for (int k = 0; k < 3; ++k) bb[i * 3 + k] = (float)(bb[i * 3 + k] - (float)mean[k]);
  }
  const float scale = (float)M.meta("meta/centeringScale");
  std::vector<int> groups;
  for (int i = 0; i < n; ++i) {
    if (emitBb[i] && !coordCond[i] &&
        std::find(groups.begin(), groups.end(), moleculeIid[i]) == groups.end()) {
      groups.push_back(moleculeIid[i]);
    }
  }
  std::sort(groups.begin(), groups.end());
  for (int g : groups) {
    const float off[3] = {noise[at], noise[at + 1], noise[at + 2]};
    at += 3;
    for (int i = 0; i < n; ++i) {
      if (!emitBb[i] || coordCond[i] || moleculeIid[i] != g) continue;
      for (int k = 0; k < 3; ++k) bb[i * 3 + k] = (float)(bb[i * 3 + k] + (float)(scale * off[k]));
    }
  }
  std::vector<float> ll(noise.begin() + at, noise.begin() + at + (size_t)n * 32);
  at += (size_t)n * 32;
  for (int i = 0; i < n; ++i) {
    if (!emitLl[i]) for (int k = 0; k < 32; ++k) ll[(size_t)i * 32 + k] = 0.f;
  }

  float* xBb = upload(bb.data(), bb.size());
  float* xLl = upload(ll.data(), ll.size());
  float* x1Bb = scratch<float>("gen.x1bb", (size_t)n * 3);
  float* x1Ll = scratch<float>("gen.x1ll", (size_t)n * 32);
  float* epsBuf = scratch<float>("gen.eps", (size_t)n * 32);
  // The features are the exporter's, with the two state blocks overwritten each step.
  float* seqF = scratch<float>("gen.seq", (size_t)n * d.seqF);
  float* pairF = scratch<float>("gen.pair", (size_t)n * n * pairDim);
  CK(cudaMemcpyAsync(seqF, Fdev("in/seq"), (size_t)n * d.seqF * sizeof(float),
                     cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemcpyAsync(pairF, Fdev("g/pairNoDist"), (size_t)n * n * pairDim * sizeof(float),
                     cudaMemcpyDeviceToDevice, STREAM));

  // 🔴 A SECOND DESIGN IS THE WARM ONE, which is how PyTorch's own warm number is taken (its first
  // pays for torch.compile). Ours pays for cuBLAS's heuristic picks, the lazy f16 weight mirrors
  // and every first-touch allocation; small, but a comparison of warm against cold is not a
  // comparison. The noise is replayed from the start each time, so every design is the same one -
  // this measures time, not variety.
  const size_t noiseStart = at;
  std::vector<float> bb0 = bb, ll0 = ll;
  double loopMs = 0;
  int passes = 0;
  for (int design = 0; design < std::max(1, o.designs); ++design) {
  at = noiseStart;
  passes = 0;
  if (design) {
    CK(cudaMemcpyAsync(xBb, bb0.data(), bb0.size() * sizeof(float), cudaMemcpyHostToDevice, STREAM));
    CK(cudaMemcpyAsync(xLl, ll0.data(), ll0.size() * sizeof(float), cudaMemcpyHostToDevice, STREAM));
    CK(cudaMemcpyAsync(seqF, Fdev("in/seq"), (size_t)n * d.seqF * sizeof(float),
                       cudaMemcpyDeviceToDevice, STREAM));
  }
  CK(cudaStreamSynchronize(STREAM));
  const auto t0 = std::chrono::steady_clock::now();
  const rfd4p::Schedule schedBb = rfd4p::bbCaSchedule(steps);
  const rfd4p::Schedule schedLl = rfd4p::latentSchedule(steps);

  const float* emitBbDev = Fdev("g/emitBb");
  const float* emitLlDev = Fdev("g/emitLl");

  for (int s = 0; s < steps; ++s) {
    rfd4p::writeStateK<<<(unsigned)n, 256, 0, STREAM>>>(seqF, xBb, xLl, n, d.seqF, d.latent);
    rfd4p::pairDistBinsK<<<blocks((size_t)n * n), 256, 0, STREAM>>>(
        pairF, xBb, Fdev("in/mask"), n, pairDim, (int)M.meta("meta/distDim"),
        (float)M.meta("meta/distMin"),
        (float)M.meta("meta/distMax"));
    // The loop's INPUT, before anything in the loop can have moved it: the replayed noise against
    // the reference's own draw, and the features this port assembles from it against the ones the
    // reference builds from scratch.
    if (s == 0 && !o.genOracle.empty() && design + 1 == std::max(1, o.designs) &&
        M.has("gen/x0Bb")) {
      checkOracle("gen/x0Bb", xBb, (size_t)n * 3, "gen/x0Bb");
      checkOracle("gen/x0Ll", xLl, (size_t)n * 32, "gen/x0Ll");
      checkOracle("gen/f0Seq", seqF, (size_t)n * d.seqF, "gen/f0Seq");
      checkOracle("gen/f0Pair", pairF, (size_t)n * n * pairDim, "gen/f0Pair");
      checkOracle("gen/f0Cond", Fdev("g/cond"), (size_t)n * condDim, "gen/f0Cond");
      checkOracle("gen/f0Cond1", Fdev("g/cond1"), (size_t)n * condDim, "gen/f0Cond1");
      checkOracle("gen/f0PairCond", Fdev("g/pairCond"), (size_t)d.pairCondF, "gen/f0PairCond");
    }
    rfd4p::forward(t, seqF, Fdev("g/cond") + (size_t)s * n * condDim,
                   Fdev("g/cond1") + (size_t)s * n * condDim, pairF,
                   Fdev("g/pairCond") + (size_t)s * d.pairCondF, depths[s], s == steps - 1);
    passes += depths[s];
    rfd4p::maskEmitK<<<blocks((size_t)n * 3), 256, 0, STREAM>>>(t.bbCa, emitBbDev, n, 3);
    rfd4p::maskEmitK<<<blocks((size_t)n * 32), 256, 0, STREAM>>>(t.localLatents, emitLlDev, n, 32);

    for (int dm = 0; dm < 2; ++dm) {
      const rfd4p::Schedule& sc = dm == 0 ? schedBb : schedLl;
      const int dim = dm == 0 ? 3 : 32;
      const double tt = (float)sc.ts[s], dt = sc.ts[s + 1] - sc.ts[s];
      const size_t count = (size_t)n * dim;
      const float* eps = nullptr;
      if (stochastic[s * 2 + dm]) {
        CK(cudaMemcpyAsync(epsBuf, noise.data() + at, count * sizeof(float),
                           cudaMemcpyHostToDevice, STREAM));
        at += count;
        eps = epsBuf;
      }
      rfd4p::simulationStep(dm == 0 ? xBb : xLl, dm == 0 ? t.bbCa : t.localLatents,
                            dm == 0 ? x1Bb : x1Ll, eps, dm == 0 ? emitBbDev : emitLlDev, count, dim,
                            tt, dt, sc.gt[s], sc.step);
      CK(cudaStreamSynchronize(STREAM));   // epsBuf is reused by the next modality
    }
    // 🔴 EVERY STEP, SO A DIVERGENCE HAS A STEP NUMBER. The backbone's first step divides by
    // t = 0.001, so a difference the trunk makes in the sixth digit leaves it in the third - and
    // the end alone cannot tell that from the loop being wired wrong.
    if (!o.genOracle.empty() && design + 1 == std::max(1, o.designs)) {
      char key[32];
      snprintf(key, sizeof key, "gen/s%dBb", s);
      if (M.has(key)) {
        const double rb = checkOracle(key, xBb, (size_t)n * 3, key);
        snprintf(key, sizeof key, "gen/s%dLl", s);
        const double rl = checkOracle(key, xLl, (size_t)n * 32, key);
        snprintf(key, sizeof key, "gen/s%dX1Bb", s);
        const double qb = M.has(key) ? checkOracle(key, x1Bb, (size_t)n * 3, key) : -1;
        snprintf(key, sizeof key, "gen/s%dX1Ll", s);
        const double ql = M.has(key) ? checkOracle(key, x1Ll, (size_t)n * 32, key) : -1;
        printf("  step %d: x %.3e / %.3e   x1 %.3e / %.3e  (bb / ll)\n", s, rb, rl, qb, ql);
      }
    }
  }

  CK(cudaStreamSynchronize(STREAM));
  loopMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (o.designs > 1) printf("  design %d: sampling %.2f s\n", design + 1, loopMs / 1000);
  }

  // 🔴 THE WHOLE LOOP AGAINST THE REFERENCE'S OWN, which is the one thing every per-stage gate
  // cannot say: the trunk, the decoder and the schedules are each held to `cpu/rfd4p`, and what is
  // left over is whether they are WIRED together the way `generateRfd4pCpu` wires them - the order
  // of the noise draws, which features move between steps, when the emit masks apply, whether the
  // step advances x or x1. Two steps exercise all of that as well as sixty and cost minutes
  // instead of hours.
  if (!o.genOracle.empty()) {
    int bad = 0;
    for (const auto& [name, ptr, count] : std::initializer_list<std::tuple<const char*, const float*, size_t>>{
             {"gen/xBb", xBb, (size_t)n * 3}, {"gen/xLl", xLl, (size_t)n * 32}}) {
      const double r = checkOracle(name, ptr, count, name);
      if (r > (o.fast ? 5e-3 : 1e-5)) ++bad;
    }
    printf("  the trajectory %s the reference's over %d steps\n",
           bad ? "DIVERGES from" : "matches", steps);
    if (bad) return 1;
  }

  // `sequenceForDecoder`: a conditioned residue stays, an emitted unconditioned one takes the
  // argmax of the trunk's ALLOWED logits.
  const int nTypes = d.classes;
  const std::vector<float> logits = download(t.logits, (size_t)n * nTypes);
  const std::vector<int> allowed(M.i("g/allowedTypes"), M.i("g/allowedTypes") + 7 * nTypes);
  const int* tokenType = M.i("in/tokenType");
  std::vector<int> seq(M.i("g/condSeq"), M.i("g/condSeq") + n);
  for (int i = 0; i < n; ++i) {
    if (!emitLl[i] || seq[i] != 0) continue;
    const int tt = std::max(0, std::min(6, tokenType[i]));
    int best = 0;
    float bestV = -3.4028234663852886e38f;
    for (int r = 0; r < nTypes; ++r) {
      const float v = allowed[(size_t)tt * nTypes + r] ? logits[(size_t)i * nTypes + r]
                                                       : -3.4028234663852886e38f;
      if (v > bestV) { bestV = v; best = r; }
    }
    seq[i] = best;
  }
  int* seqDev = upload(seq.data(), seq.size());

  rfd4p::DecDims dd;
  dd.n = n;
  dd.D = (int)M.meta("meta/decTokenDim");
  dd.C = (int)M.meta("meta/decCondDim");
  dd.H = (int)M.meta("meta/decHeads");
  dd.nlayers = (int)M.meta("meta/decLayers");
  dd.nTypes = nTypes;
  dd.latent = d.latent;
  dd.fdim = (int)M.meta("meta/decFeatDim");
  float* dLogits = scratch<float>("gen.dlogits", (size_t)n * nTypes);
  float* dCoors = scratch<float>("gen.dcoors", (size_t)n * 111);
  float* dPresence = scratch<float>("gen.dpresence", (size_t)n * 37);
  rfd4p::decode(dd, xBb, xLl, M.i("in/tokenType") ? Idev("in/tokenType") : nullptr, seqDev,
                Fdev("in/attnMask"), dLogits, dCoors, dPresence);
  CK(cudaStreamSynchronize(STREAM));

  const rfd4p::GenOut g = rfd4p::postprocess(
      n, nTypes, download(dLogits, (size_t)n * nTypes), download(dCoors, (size_t)n * 111),
      download(dPresence, (size_t)n * 37), download(xBb, (size_t)n * 3), seq,
      std::vector<int>(tokenType, tokenType + n),
      std::vector<int>(M.i("g/coordGroup"), M.i("g/coordGroup") + (size_t)n * 37), allowed,
      std::vector<int>(M.i("g/atom37Legal"), M.i("g/atom37Legal") + (size_t)nTypes * 37), emitLl,
      download(t.mask, n));

  // The atom and residue names, written beside the tensors by the exporter: the page's own tables.
  std::vector<std::string> atomNames, resNames;
  {
    std::ifstream in(o.gen + "/names.txt");
    std::string line, word;
    for (int which = 0; which < 2 && std::getline(in, line); ++which) {
      std::istringstream ss(line);
      while (ss >> word) (which == 0 ? atomNames : resNames).push_back(word);
    }
  }
  if (atomNames.size() != 37 || resNames.size() != (size_t)nTypes) {
    fprintf(stderr, "%s/names.txt has %zu atom and %zu residue names\n", o.gen.c_str(),
            atomNames.size(), resNames.size());
    return 1;
  }
  const std::string pdb = rfd4p::toPdb(g, n, atomNames, resNames);
  std::ofstream(o.out) << pdb;
  int atoms = 0, residues = 0;
  for (int i = 0; i < n; ++i) {
    if (g.residueType[i] >= 0) ++residues;
    for (int a = 0; a < 37; ++a) atoms += g.atomMask[(size_t)i * 37 + a];
  }
  printf("generated %d residues, %d atoms over %d steps (%d trunk passes) -> %s\n", residues, atoms,
         steps, passes, o.out.c_str());
  printf("  sampling %.2f s\n", loopMs / 1000);
  return 0;
}

int foldMain(int argc, char** argv) {
  Opts o;
  o.dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string f = argv[i];
    auto value = [&](const char* name) { return f.substr(strlen(name) + 3); };
    if (!f.compare(0, 10, "--weights=")) o.weights = value("weights");
    else if (!f.compare(0, 9, "--oracle=")) o.oracle = value("oracle");
    else if (!f.compare(0, 11, "--generate=")) o.gen = f.substr(11);
    else if (!f.compare(0, 7, "--int8=")) o.int8 = f.substr(7);
    else if (!f.compare(0, 13, "--gen-oracle=")) o.genOracle = f.substr(13);
    else if (!f.compare(0, 13, "--enc-oracle=")) o.encOracle = f.substr(13);
    else if (!f.compare(0, 10, "--encoder=")) o.enc = f.substr(10);
    else if (!f.compare(0, 6, "--out=")) o.out = value("out");
    else if (!f.compare(0, 11, "--recycles=")) o.recycles = atoi(f.c_str() + 11);
    else if (!f.compare(0, 9, "--repeat=")) o.repeat = atoi(f.c_str() + 9);
    else if (!f.compare(0, 10, "--designs=")) o.designs = atoi(f.c_str() + 10);
    else if (f == "--fast") o.fast = true;
    else if (f == "--no-insertion") o.insertion = false;
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (o.weights.empty() && o.int8.empty()) {
    fprintf(stderr, "one of --weights=<dir> or --int8=<dir> is required\n");
    return 1;
  }
  return run(o);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) {
    fprintf(stderr, "usage: localfold-rfd4p <input dir> --weights=<dir> [--oracle=<dir>]\n");
    return 1;
  }
  return foldMain(argc, argv);
}
