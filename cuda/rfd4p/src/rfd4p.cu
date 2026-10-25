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

namespace {

struct Opts {
  std::string dir, weights, oracle, out = "trunk.bin";
  int recycles = 0;  // 0 = whatever the input directory says
  int repeat = 0;    // time the forward this many times after a warm-up
  bool fast = false; // TF32 on the tensor cores, as every other port's --fast does
  bool insertion = true;
};

int run(const Opts& o) {
  M.load(o.weights);
  M.load(o.dir);
  if (!o.oracle.empty()) M.load(o.oracle);
  for (int seg = 0; seg < (int)M.segs.size(); ++seg) M.upload(seg);
  CB(cublasCreate(&H));
  CB(cublasSetStream(H, STREAM));
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
    const bool decoder = name[0] == 'd' && name[1] == '/';
    if (decoder) return 1e-4;   // the decoder is f32 on both arms
    return o.fast ? 5e-3 : 1e-5;
  };
  const double bound = o.fast ? 5e-3 : 1e-5;
  int bad = 0;
  double worst = 0;
  std::string worstAt;
  std::function<void(const char*, const float*, size_t)> tap;
  if (!o.oracle.empty()) {
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

int foldMain(int argc, char** argv) {
  Opts o;
  o.dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string f = argv[i];
    auto value = [&](const char* name) { return f.substr(strlen(name) + 3); };
    if (!f.compare(0, 10, "--weights=")) o.weights = value("weights");
    else if (!f.compare(0, 9, "--oracle=")) o.oracle = value("oracle");
    else if (!f.compare(0, 6, "--out=")) o.out = value("out");
    else if (!f.compare(0, 11, "--recycles=")) o.recycles = atoi(f.c_str() + 11);
    else if (!f.compare(0, 9, "--repeat=")) o.repeat = atoi(f.c_str() + 9);
    else if (f == "--fast") o.fast = true;
    else if (f == "--no-insertion") o.insertion = false;
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (o.weights.empty()) { fprintf(stderr, "--weights=<dir> is required\n"); return 1; }
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
