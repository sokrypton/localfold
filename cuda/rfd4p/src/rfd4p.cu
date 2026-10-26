// The native CUDA port of RFD4-Proteina.
//
//   localfold-rfd4p <input dir> --weights=<dir> [--oracle=<dir>] [--out=<path>]
//
//   localfold-rfd4p --job=<spec.json> --weights=<dir> [--int8=<bundle>] [--out=<design.pdb>]
//
// `<input dir>` is `cuda/rfd4p/export_input.mjs`'s `model.idx`/`model.bin`: the trunk's features for
// one sampler step, which is the differential path. `--job=` is the whole thing in one process -
// the spec read, the mmCIF featurised (`featurise/featurise.cpp`, linked in), the flow loop run and
// a structure written. `--weights=` is `export_weights.mjs`'s float32 directory and `--int8=` the
// published bundle through `Model::loadBundle`; `--oracle=` is the CPU reference's answer stage by
// stage, and turns the first form into the differential that gates the port.
//
// 🔴 THIS HEADER SAID "WHAT THIS IS NOT, YET: A FOLD" LONG AFTER IT WAS ONE, and said the int8
// bundle "is not wired yet" while `gate.py`'s `int8 job` arm folded from it alone. A file's own
// first paragraph is the one thing nobody re-reads; both claims are gone rather than softened.
#include "trunk.cuh"
#include "decoder.cuh"
#include "sampler.cuh"
#include "encoder.cuh"
#include "chirality.cuh"
#include "partial.cuh"
#include "generate.cuh"
#include "metrics.h"
#include "oracle.h"
#include "bundle.cuh"
#include <fstream>
#include <sys/stat.h>
#include <chrono>

// 🔴 OUTSIDE THE ANONYMOUS NAMESPACE. Declared inside it this is `<unnamed>::rfd4p`, which
// shadows the real namespace and makes every `rfd4p::Dims` in the file fail to resolve.
// The featuriser, linked in rather than run as a second process (featurise/featurise.cpp).
namespace rfd4p { namespace featurise { int run(int argc, char** argv); } }

namespace {

struct Opts {
  std::string dir, weights, oracle, gen, genOracle, enc, encOracle, int8, job, dumpSample, metrics,
      out = "design.pdb";
  // `--refold[=<localfold-af3>]`: refold each design that passes the pre-oracle filters with our
  // own AlphaFold 3, as upstream's oracle does with DeepMind's (src/oracle.h).
  std::string refold;
  int seed = 0;      // a later design draws its own noise from seed + design
  int recycles = 0;  // 0 = whatever the input directory says
  int repeat = 0;    // time the forward this many times after a warm-up
  int designs = 1;   // generate this many times, so a WARM one can be timed
  bool fast = false; // TF32 on the tensor cores, as every other port's --fast does
  bool insertion = true;
};

int generate(const Opts& o, const rfd4p::Dims& d, rfd4p::Trunk& t);

/** cuda/af3/localfold-af3 beside this binary's cuda/rfd4p. */
std::string defaultAf3() {
  char self[4096];
  const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
  std::string dir = n > 0 ? std::string(self, (size_t)n) : std::string("./localfold-rfd4p");
  dir = dir.substr(0, dir.rfind('/'));
  return dir + "/../af3/localfold-af3";
}

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
// The sampler the featuriser resolved from the job (flow.js `samplingMeta`), or the released one
// for a directory written before it carried one.
static rfd4p::SamplerCfg samplerFromMeta() {
  rfd4p::SamplerCfg sampler = rfd4p::defaultSampler();
  if (!M.has("meta/sampBbMode")) return sampler;
  static const char* MODES[] = {"vf_tsr", "vf", "sc"};
  static const char* SCHEDS[] = {"uniform", "power", "log"};
  static const char* GTS[] = {"1/t", "1-t/t", "tan", "ode"};
  for (int dm = 0; dm < 2; ++dm) {
    const std::string tag = dm ? "meta/sampLl" : "meta/sampBb";
    auto get = [&](const char* k) { return M.meta(tag + k, 0); };
    rfd4p::ModalityCfg& m = sampler.m[dm];
    m.step.mode = MODES[(int)get("Mode")];
    m.schedule.mode = SCHEDS[(int)get("Sched")];
    m.schedule.p = get("P");
    m.schedule.tailSteps = (int)get("Tail");
    m.schedule.tailMult = (int)get("TailMult");
    m.gt.mode = GTS[(int)get("Gt")];
    m.gt.p = get("GtP");
    m.gt.clamp = get("HasClamp") != 0;
    m.gt.clampVal = get("Clamp");
    m.step.scScaleNoise = get("Noise");
    m.step.tLimOde = get("Ode");
    m.step.tLimOdeBelow = get("OdeBelow");
    m.step.tsrK = get("TsrK");
    m.step.tsrSigma = get("TsrSigma");
    m.step.center = get("Center") != 0;
  }
  return sampler;
}

int generate(const Opts& o, const rfd4p::Dims& d, rfd4p::Trunk& t) {
  const int n = d.n;
  const rfd4p::SamplerCfg sampler = samplerFromMeta();
  const int steps = (int)M.meta("meta/steps");
  const int condDim = d.condF, pairDim = d.pairF;
  const std::vector<float> emitBb = download(Fdev("g/emitBb"), n);
  const std::vector<float> emitLl = download(Fdev("g/emitLl"), n);
  std::vector<float> noise = download(Fdev("g/noise"), M.len("g/noise"));
  const int* depths = M.i("g/depths");
  const int* stochastic = M.i("g/stochastic");
  const int* coordCond = M.i("g/coordCond");
  const int* moleculeIid = M.i("g/moleculeIid");

  const float scale = (float)M.meta("meta/centeringScale");
  const int latent = d.latent;
  const std::vector<int> groups = rfd4p::offsetGroups(n, emitBb.data(), coordCond, moleculeIid);
  const bool partial = M.meta("meta/partial", 0) != 0;
  const bool inverseFolding = M.meta("meta/inverseFolding", 0) != 0;
  size_t at = 0;
  std::vector<float> bb, ll, targetBb;

  if (partial) {
    // 🔴 A PARTIAL RUN STARTS FROM A STRUCTURE, NOT FROM NOISE. The encoder turns the input's
    // all-atom coordinates into the latent half of a CLEAN sample; its CA is the other half.
    // `corrupt_batch` then puts both part-way back along the flow.
    if (o.enc.empty()) {
      fprintf(stderr, "this input is a partial-diffusion or inverse-folding run and needs "
                      "--encoder=<dir> beside --generate\n");
      return 1;
    }
    rfd4p::EncDims ed;
    ed.n = n;
    ed.D = (int)M.meta("meta/encTokenDim");
    ed.C = (int)M.meta("meta/encCondDim");
    ed.H = (int)M.meta("meta/encHeads");
    ed.nlayers = (int)M.meta("meta/encLayers");
    ed.latent = latent;
    ed.fdim = (int)M.meta("meta/encFeatDim");
    ed.ttDim = (int)M.meta("meta/encTtDim");
    ed.chargeDim = (int)M.meta("meta/encChargeDim");
    const int offsets[7] = {
        (int)M.meta("meta/encOffAbs"),   (int)M.meta("meta/encOffRel"),
        (int)M.meta("meta/encOffChain"), (int)M.meta("meta/encOffBonds"),
        (int)M.meta("meta/encOffElCh"),  (int)M.meta("meta/encTtCreator"),
        (int)M.meta("meta/encElChCreator")};
    float* mean = scratch<float>("rfd4p.gen.mean", (size_t)n * latent);
    float* logScale = scratch<float>("rfd4p.gen.ls", (size_t)n * latent);
    float* zLat = scratch<float>("rfd4p.gen.z", (size_t)n * latent);
    // The reparameterisation's own draw comes FIRST in the stream, because `encodeLatentsCpu` is
    // handed the generator's rng before `corrupt_batch` touches it.
    float* encEps = upload(noise.data() + at, (size_t)n * latent);
    at += (size_t)n * latent;
    rfd4p::encode(ed, Fdev("e/coorsNm"), Fdev("e/atomMask"), Idev("in/tokenType"),
                  Fdev("e/chainIid"), Fdev("e/bondCounts"), Idev("e/element"), Idev("e/charge"),
                  Fdev("in/mask"), Fdev("in/attnMask"), Fdev("g/emitLl"), encEps, mean, logScale,
                  zLat, offsets);
    CK(cudaStreamSynchronize(STREAM));
    const std::vector<float> z = download(zLat, (size_t)n * latent);

    // x1's backbone is the input's own CA, in nanometres. A slot that is not there is zero, as the
    // reference's `Number.isFinite(v) ? v : 0` makes it.
    const std::vector<float> coorsNm = download(Fdev("e/coorsNm"), (size_t)n * 37 * 3);
    std::vector<float> x1Bb((size_t)n * 3, 0.f);
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        const float v = coorsNm[((size_t)i * 37 + 1) * 3 + k];
        x1Bb[i * 3 + k] = std::isfinite(v) ? v : 0.f;
      }
    }
    rfd4p::PartialT pt;
    pt.set = true;
    pt.bbCa = M.meta("meta/partialTBb", 0);
    pt.localLatents = M.meta("meta/partialTLl", 0);
    const rfd4p::Corrupted c =
        rfd4p::corruptBatch(x1Bb, z, pt, n, latent, emitBb.data(), emitLl.data(), coordCond,
                            moleculeIid, groups, noise.data(), at, scale);
    bb = c.bb;
    ll = c.ll;
    if (inverseFolding) {
      // `resampled_forward_marginal`: the CLEAN target is kept and the state is redrawn from its
      // forward marginal at every step, so the backbone never drifts while the sequence is designed.
      targetBb = x1Bb;
      for (int i = 0; i < n; ++i) {
        if (!emitBb[i]) targetBb[i * 3] = targetBb[i * 3 + 1] = targetBb[i * 3 + 2] = 0.f;
      }
      bb = rfd4p::noisedTargetBackbone(targetBb, rfd4p::partialSchedule(0, pt.bbCa, steps,
                                                                        (int)M.meta("meta/fullNsteps", 0),
                                                                        sampler).ts[0],
                                       n, emitBb.data(), coordCond, moleculeIid, groups,
                                       noise.data(), at, scale);
    }
  } else {
    // `sample_noise` for both modalities: standard normal on the emitting tokens, the backbone
    // centred and then offset once a molecule - all of it replayed from the exporter's own draws.
    bb = rfd4p::zeroComBbNoise(noise.data(), n, emitBb.data());
    at = (size_t)n * 3;
    rfd4p::applyPerGroupOffset(bb, n, emitBb.data(), coordCond, moleculeIid, groups, noise.data(),
                               at, scale);
    ll.assign(noise.begin() + at, noise.begin() + at + (size_t)n * latent);
    at += (size_t)n * latent;
    for (int i = 0; i < n; ++i) {
      if (!emitLl[i]) for (int k = 0; k < latent; ++k) ll[(size_t)i * latent + k] = 0.f;
    }
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
  const size_t noiseFloats = noise.size();
  double loopMs = 0;
  int passes = 0;
  // 🔴 EACH DESIGN IS WRITTEN, NOT ONLY THE LAST. `--designs=N` used to replay one noise stream
  // N times so a warm fold could be timed against a cold one; it draws a FRESH stream for every
  // design after the first now, which is what a campaign wants and costs the same. Design 1 keeps
  // the exporter's replayed stream, so the trajectory gate is unchanged: an oracle comparison
  // needs the reference's own draws and nobody else's.
  auto finishDesign = [&](int design) -> int {

    // 🔴 THE WHOLE LOOP AGAINST THE REFERENCE'S OWN, which is the one thing every per-stage gate
    // cannot say: the trunk, the decoder and the schedules are each held to `cpu/rfd4p`, and what is
    // left over is whether they are WIRED together the way `generateRfd4pCpu` wires them - the order
    // of the noise draws, which features move between steps, when the emit masks apply, whether the
    // step advances x or x1. Two steps exercise all of that as well as sixty and cost minutes
    // instead of hours.
    if (!o.genOracle.empty() && design == 0) {
      int bad = 0;
      for (const auto& [name, ptr, count] : std::initializer_list<std::tuple<const char*, const float*, size_t>>{
               {"gen/xBb", xBb, (size_t)n * 3}, {"gen/xLl", xLl, (size_t)n * 32}}) {
        const double r = checkOracle(name, ptr, count, name);
        // 🔴 A TRAJECTORY COMPOUNDS, SO ITS --fast BOUND IS NOT A STAGE'S. The trunk's f16 and TF32
        // residual is ~1e-3 a pass; the step at t = 0.001 amplifies it and the next step starts from
        // the result, so two steps read 1.6e-2 with every input exact. That is arithmetic, not a
        // defect, and the number that says so is the STRUCTURE: a finished 60-step generation is
        // 0.020 A of CA from the exact arm, worst atom 0.034 - far inside the seed-to-seed spread.
        // The gate measures that directly beside this.
        if (r > (o.fast ? 5e-2 : 1e-5)) ++bad;
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

    rfd4p::GenOut g = rfd4p::postprocess(
        n, nTypes, download(dLogits, (size_t)n * nTypes), download(dCoors, (size_t)n * 111),
        download(dPresence, (size_t)n * 37), download(xBb, (size_t)n * 3), seq,
        std::vector<int>(tokenType, tokenType + n),
        std::vector<int>(M.i("g/coordGroup"), M.i("g/coordGroup") + (size_t)n * 37), allowed,
        std::vector<int>(M.i("g/atom37Legal"), M.i("g/atom37Legal") + (size_t)nTypes * 37), emitLl,
        download(t.mask, n));

    // 🔴 THE SAMPLE, BEFORE ANY OF THE WRITER'S OWN STEPS, so the writer can be held to the page's.
    // `processSampleToAtoms` takes exactly these - the decoded coordinates, the atom mask, the
    // residue types, the padding and the insertion head's logits - and does the grafting, the
    // alignment and the naming itself; cuda/rfd4p/check-writer.mjs runs it on this dump and
    // requires the SAME TEXT as the file written below.
    if (!o.dumpSample.empty() && design == 0) {
      mkdir(o.dumpSample.c_str(), 0755);
      auto raw = [&](const char* name, const void* p, size_t bytes) {
        FILE* f = fopen((o.dumpSample + "/" + name).c_str(), "wb");
        fwrite(p, 1, bytes, f);
        fclose(f);
      };
      raw("coors.f32", g.coors.data(), g.coors.size() * 4);
      raw("atomMask.u8", g.atomMask.data(), g.atomMask.size());
      raw("residueType.i32", g.residueType.data(), g.residueType.size() * 4);
      const std::vector<float> padHost = download(t.mask, n);
      raw("pad.f32", padHost.data(), padHost.size() * 4);
      if (t.insertion) {
        const std::vector<float> ins = download(t.insertion, (size_t)n * n);
        raw("insertion.f32", ins.data(), ins.size() * 4);
      }
    }
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
    // 🔴 THE FORMAT FOLLOWS THE NAME, because mmCIF is what RFD4-Proteina itself writes and a PDB
    // cannot carry a long chain id or more than 99,999 atoms. `--out=design.cif` gets a CIF.
    const std::vector<int> chainOf =
        M.has("g/chainIid")
            ? std::vector<int>(M.i("g/chainIid"), M.i("g/chainIid") + n)
            : std::vector<int>(n, 0);
    const bool wantCif = o.out.size() > 4 && o.out.compare(o.out.size() - 4, 4, ".cif") == 0;
    // One file a design: `design.pdb`, then `design_2.pdb`, `design_3.pdb`. The first keeps the
    // name it was asked for, so a single-design run is unchanged.
    std::string path = o.out;
    if (design > 0) {
      const size_t dot = path.rfind('.');
      const std::string stem = dot == std::string::npos ? path : path.substr(0, dot);
      const std::string ext = dot == std::string::npos ? "" : path.substr(dot);
      path = stem + "_" + std::to_string(design + 1) + ext;
    }
    // The input's own numbering and chain letters where a job named a structure; absent for a
    // contig, where the token index IS the residue number. And what the page's writer reads off
    // the input atoms (`outputTables` in the featuriser), which the grafting needs too.
    rfd4p::OutNaming naming;
    if (M.has("g/outResId") && M.has("g/outChain")) {
      naming.resId.assign(M.i("g/outResId"), M.i("g/outResId") + n);
      naming.chain.assign(M.i("g/outChain"), M.i("g/outChain") + n);
    }
    if (M.has("g/inName")) {
      auto ints = [&](const char* key, size_t count, std::vector<int>& into) {
        into.assign(M.i(key), M.i(key) + count);
      };
      const size_t slots = (size_t)n * 37;
      ints("g/outAtomize", n, naming.in.atomize);
      ints("g/outHetero", n, naming.in.hetero);
      ints("g/outEntity", n, naming.in.entity);
      ints("g/outInsCode", n, naming.in.insCode);
      ints("g/outResName", (size_t)n * 2, naming.in.resName);
      ints("g/outSrcComp", (size_t)n * 4, naming.in.srcComp);
      ints("g/inName", slots, naming.in.inName);
      ints("g/inElem", slots, naming.in.inElem);
      ints("g/inCharge", slots, naming.in.inCharge);
      ints("g/inB", slots * 2, naming.in.inB);
      ints("g/inOcc", slots * 2, naming.in.inOcc);
    }
    // Only the INDEXED tokens are written; an unindexed motif was grafted onto one of them.
    if (M.has("g/condIndex")) {
      const int* ci = M.i("g/condIndex");
      naming.emit.assign((size_t)n, 1);
      for (int i = 0; i < n; ++i) naming.emit[i] = ci[(size_t)i * 37 + 1] != 0;
    }
    std::vector<int> crdSource, srcAt;
    rfd4p::Insertion proposed;   // the head's mapping as proposed, which the metrics count against
    // 🔴 THE UNINDEXED MOTIFS FIRST, BECAUSE THEY SUPPLY ANCHORS THE ALIGNMENT THEN USES. An
    // unindexed token is in-context conditioning and is never emitted, so its conditioned
    // coordinates reach the output only by being grafted onto a token that was.
    std::vector<uint8_t> isInsertion;
    std::vector<float> condCoordRW, condIntCrdRW;
    std::vector<int> coordGroupRW;
    if (M.has("g/condCoord")) {
      const size_t slots = (size_t)n * 37;
      condCoordRW.assign(M.f("g/condCoord"), M.f("g/condCoord") + slots * 3);
      if (M.has("g/condIntCrd")) {
        condIntCrdRW.assign(M.f("g/condIntCrd"), M.f("g/condIntCrd") + slots * 3);
      }
      if (M.has("g/coordGroup")) {
        coordGroupRW.assign(M.i("g/coordGroup"), M.i("g/coordGroup") + slots);
      }
      if (M.has("g/condIndex")) {
        std::vector<uint8_t> emitted((size_t)n, 1);
        // The insertion head's own logits from the LAST step, which is the only pass that
        // asked for them.
        const std::vector<float> insLogits =
            t.insertion ? download(t.insertion, (size_t)n * n) : std::vector<float>();
        proposed = rfd4p::predictedInsertionMapping(
            n, insLogits.empty() ? nullptr : insLogits.data(), M.i("g/condIndex"), emitted,
            naming.chain);
        const rfd4p::Insertion ins = rfd4p::resolveInsertion(
            g, n, insLogits.empty() ? nullptr : insLogits.data(), M.i("g/condIndex"),
            M.has("g/resType") ? M.i("g/resType") : nullptr,
            M.has("g/coords") ? M.f("g/coords") : nullptr, condCoordRW.data(),
            condIntCrdRW.empty() ? nullptr : condIntCrdRW.data(), emitted, naming);
        if (!ins.source.empty()) {
          printf("  %zu unindexed token%s grafted by the insertion head's mapping\n",
                 ins.source.size(), ins.source.size() == 1 ? "" : "s");
        } else {
          // Upstream warns and writes the design without the motif's annotations; so does this.
          bool anyUnindexed = false;
          for (int i = 0; i < n; ++i) anyUnindexed |= M.i("g/condIndex")[(size_t)i * 37 + 1] == 0;
          if (anyUnindexed) {
            printf("  WARNING: the insertion head's mapping could not be applied to this design; "
                   "the unindexed motif is written without its annotations\n");
          }
        }
        rfd4p::applyInsertion(g, n, ins, naming, M.has("g/resType") ? M.i("g/resType") : nullptr,
                              M.has("g/coords") ? M.f("g/coords") : nullptr, condCoordRW.data(),
                              condIntCrdRW.empty() ? nullptr : condIntCrdRW.data(),
                              coordGroupRW.empty() ? nullptr : coordGroupRW.data(), isInsertion,
                              &crdSource, false, nullptr, &srcAt);
      }
    }
    // 🔴 INTO THE INPUT'S FRAME BEFORE ANYTHING IS WRITTEN. A contig has no conditioned
    // coordinates, so this is a no-op there; a structure job's output is otherwise in the
    // sampler's centred frame and superposes on nothing.
    rfd4p::alignToInputFrame(g, n, condCoordRW.empty() ? nullptr : condCoordRW.data(),
                             condIntCrdRW.empty() ? nullptr : condIntCrdRW.data(),
                             coordGroupRW.empty() ? nullptr : coordGroupRW.data(), naming);
    // The records the page's `processSampleToAtoms` builds, from the INPUT's conditioning as it
    // was before any grafting - which atom carried a conditioned coordinate is a fact about the
    // input, and the grafted ones are named by `crdSource`.
    const std::vector<rfd4p::AtomRecord> records = rfd4p::buildRecords(
        g, n, chainOf, naming, M.has("g/condCoord") ? M.f("g/condCoord") : nullptr,
        M.has("g/condIntCrd") ? M.f("g/condIntCrd") : nullptr, isInsertion, crdSource, srcAt,
        M.has("g/coordGroup") ? M.i("g/coordGroup") : nullptr);
    std::ofstream(path) << (wantCif ? rfd4p::toCif(records) : rfd4p::toPdb(records));
    // 🔴 AND WHAT UPSTREAM'S EVALUATION WOULD SAY ABOUT IT (src/metrics.h): its design metrics
    // under the job's `metrics_tags`, and the pre-oracle verdict - over the records just written.
    if (!o.metrics.empty()) {
      if (!M.has("g/mTok")) {
        fprintf(stderr, "--metrics needs a work directory featurised with the metric tables\n");
        return 1;
      }
      rfd4p::metrics::Inputs in;
      in.nTokens = n;
      auto ints = [&](const char* key, size_t count) {
        return std::vector<int>(M.i(key), M.i(key) + count);
      };
      in.tok = ints("g/mTok", (size_t)n * 3);
      in.chainIid = ints("g/mChainIid", (size_t)n * 4);
      in.compType = ints("g/mCompType", (size_t)n * 8);
      in.seq = ints("g/mSeq", (size_t)n * 2);
      in.hot = ints("g/mHot", (size_t)n * 37);
      in.pnUnit = ints("g/mPnUnit", (size_t)n * 4);
      in.tokMol = ints("g/mTokMol", (size_t)n * 2);
      in.inAtom = ints("g/mInAtom", M.len("g/mInAtom"));
      if (M.len("g/mInBonds") > 0) in.inBonds = ints("g/mInBonds", M.len("g/mInBonds"));
      in.condCoord = M.has("g/condCoord") ? M.f("g/condCoord") : nullptr;
      in.condIntCrd = M.has("g/condIntCrd") ? M.f("g/condIntCrd") : nullptr;
      in.coordGroup = M.has("g/coordGroup") ? M.i("g/coordGroup") : nullptr;
      in.resName = naming.in.resName;
      {
        const std::vector<int> bitsOf = ints("g/mSeqProb", (size_t)n * 28);
        in.seqProb.resize(bitsOf.size());
        memcpy(in.seqProb.data(), bitsOf.data(), bitsOf.size() * 4);
      }
      in.condIndex = M.has("g/condIndex") ? M.i("g/condIndex") : nullptr;
      rfd4p::metrics::Result r;
      try {
        r = rfd4p::metrics::designMetrics(records, in, proposed,
                                          rfd4p::metricsTagsFromBits((int)M.meta("meta/metricsTags", 0)));
      } catch (const std::exception& e) {
        fprintf(stderr, "--metrics: %s\n", e.what());
        return 1;
      }
      std::string mpath = o.metrics;
      if (design > 0) {
        const size_t dot = mpath.rfind('.');
        mpath = (dot == std::string::npos ? mpath : mpath.substr(0, dot)) + "_" +
                std::to_string(design + 1) + (dot == std::string::npos ? "" : mpath.substr(dot));
      }
      std::ofstream(mpath) << rfd4p::metrics::toJson(r) << "\n";
      printf("  metrics -> %s (%s the pre-oracle filters)\n", mpath.c_str(), r.pass ? "passes" : "fails");
      // The oracle runs on what survives the pre-oracle filters, as upstream's does.
      if (!o.refold.empty() && r.pass) {
        const std::vector<std::string> tags = rfd4p::metricsTagsFromBits((int)M.meta("meta/metricsTags", 0));
        // Named after the design it scores: `<design>.refold.json`, the folds under `<design>_refold/`.
        const size_t dot = path.rfind('.');
        const std::string base = dot == std::string::npos || dot < path.rfind('/') + 1 ? path : path.substr(0, dot);
        const size_t slash = base.rfind('/');
        rfd4p::oracle::Run run;
        run.binary = o.refold;
        run.dir = base + "_refold";
        run.name = slash == std::string::npos ? base : base.substr(slash + 1);
        run.seed = (uint32_t)(o.seed + design);
        run.components = rfd4p::oracle::readComponents(o.gen + "/input-components.cif");
        try {
          const rfd4p::metrics::Array full = rfd4p::metrics::designArray(records, in);
          std::vector<double> cwi;
          if (!rfd4p::metrics::coordsWithInsertion(full, in, proposed, cwi)) cwi.clear();
          std::string coi;
          const rfd4p::metrics::Values v = rfd4p::oracle::refoldDesign(full, cwi, run, tags, r.metrics, coi);
          if (!v.empty()) {
            std::ofstream(base + ".refold.json") << rfd4p::oracle::toJson(v, coi) << "\n";
            printf("  refolded -> %s.refold.json\n", base.c_str());
          }
        } catch (const std::exception& e) {
          fprintf(stderr, "--refold: %s\n", e.what());
          return 1;
        }
      }
    }
    // 🔴 COUNTED OVER WHAT IS WRITTEN, NOT OVER EVERY TOKEN. The unindexed tokens are grafted
    // rather than emitted, and counting them made this line disagree with its own file twice:
    // "72 residues" over 112, then "602 atoms" over 562. A count that does not match the
    // output is worse than no count - it is what made the residue bug look like a non-issue.
    int atoms = (int)records.size(), residues = 0;
    for (int i = 0; i < n; ++i) {
      if (naming.skip(i)) continue;
      if (g.residueType[i] >= 0) ++residues;
    }
    printf("generated %d residues, %d atoms over %d steps (%d trunk passes) -> %s\n", residues, atoms,
           steps, passes, path.c_str());
    printf("  sampling %.2f s\n", loopMs / 1000);
    return 0;
  };

  for (int design = 0; design < std::max(1, o.designs); ++design) {
  at = noiseStart;
  passes = 0;
  if (design) {
    // 🔴 A FRESH STREAM, NOT THE SAME ONE AGAIN. The port's Box-Muller is bit-identical to the
    // page's now, so a later design can DRAW rather than replay - seeded `seed + design`, which
    // is how a campaign gets variety out of one process with the weights already up. Design 1
    // keeps the exporter's recorded stream so the trajectory gate still compares two sides that
    // drew the same numbers.
    noise.resize(noiseFloats);
    rfd4p::Rng g2((uint64_t)(o.seed + design));
    g2.randn(noise.data(), noise.size());
    const std::vector<float> nbb = rfd4p::zeroComBbNoise(noise.data(), n, emitBb.data());
    std::vector<float> nb = nbb;
    size_t a2 = (size_t)n * 3;
    rfd4p::applyPerGroupOffset(nb, n, emitBb.data(), coordCond, moleculeIid, groups, noise.data(),
                               a2, scale);
    std::vector<float> nl(noise.begin() + a2, noise.begin() + a2 + (size_t)n * latent);
    for (int i = 0; i < n; ++i) {
      if (!emitLl[i]) for (int k = 0; k < latent; ++k) nl[(size_t)i * latent + k] = 0.f;
    }
    at = a2 + (size_t)n * latent;
    CK(cudaMemcpyAsync(xBb, nb.data(), nb.size() * sizeof(float), cudaMemcpyHostToDevice, STREAM));
    CK(cudaMemcpyAsync(xLl, nl.data(), nl.size() * sizeof(float), cudaMemcpyHostToDevice, STREAM));
    CK(cudaMemcpyAsync(seqF, Fdev("in/seq"), (size_t)n * d.seqF * sizeof(float),
                       cudaMemcpyDeviceToDevice, STREAM));
  }
  CK(cudaStreamSynchronize(STREAM));
  const auto t0 = std::chrono::steady_clock::now();
  // 🔴 A PARTIAL RUN'S SCHEDULE IS NOT THE ORDINARY ONE WITH ITS HEAD CUT OFF - see partial.cuh.
  const int fullNsteps = (int)M.meta("meta/fullNsteps", 0);
  const rfd4p::Schedule schedBb =
      partial ? rfd4p::partialSchedule(0, M.meta("meta/partialTBb", 0), steps, fullNsteps, sampler)
              : rfd4p::buildSchedule(sampler.m[0], steps);
  const rfd4p::Schedule schedLl =
      partial ? rfd4p::partialSchedule(1, M.meta("meta/partialTLl", 0), steps, fullNsteps, sampler)
              : rfd4p::buildSchedule(sampler.m[1], steps);

  const float* emitBbDev = Fdev("g/emitBb");
  const float* emitLlDev = Fdev("g/emitLl");

  const bool alignsChirality = M.meta("meta/alignChirality", 0) != 0;
  const int chirOff = (int)M.meta("meta/chirOff", -1);
  const int chiralF = (int)M.meta("meta/chiralFrameWidth", 0);

  for (int s = 0; s < steps; ++s) {
    rfd4p::writeStateK<<<(unsigned)n, 256, 0, STREAM>>>(seqF, xBb, xLl, n, d.seqF, d.latent);
    // 🔴 THE CHIRALITY IS THE OTHER FEATURE THAT MOVES. Its packed vectors are written in the
    // CLEAN structure's frame; each step refits a rotation against the noisy CA and rotates them,
    // so freezing them at step zero tells the model a handedness measured against coordinates the
    // sampler no longer has.
    if (alignsChirality) {
      rfd4p::alignChirality(Fdev("g/chir"), Idev("g/chiralFrames"), Idev("g/chiralOwners"), xBb,
                            Fdev("in/mask"), seqF, n, chiralF, d.seqF, chirOff);
    }
    rfd4p::pairDistBinsK<<<blocks((size_t)n * n), 256, 0, STREAM>>>(
        pairF, xBb, Fdev("in/mask"), n, pairDim, (int)M.meta("meta/distDim"),
        (float)M.meta("meta/distMin"),
        (float)M.meta("meta/distMax"));
    // The loop's INPUT, before anything in the loop can have moved it: the replayed noise against
    // the reference's own draw, and the features this port assembles from it against the ones the
    // reference builds from scratch.
    // 🔴 DESIGN 0, NOT THE LAST ONE. The oracle's draws are the exporter's, which only the FIRST
    // design replays - every later one draws its own, so comparing it would be comparing two
    // different noise streams and calling the difference a port defect.
    if (s == 0 && !o.genOracle.empty() && design == 0 && M.has("gen/x0Bb")) {
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
    // The resampled target comes after BOTH modalities have stepped, and the last step takes the
    // clean target as it is rather than a noised draw of it.
    if (inverseFolding) {
      std::vector<float> next =
          s == steps - 1
              ? targetBb
              : rfd4p::noisedTargetBackbone(targetBb, schedBb.ts[s + 1], n, emitBb.data(),
                                            coordCond, moleculeIid, groups, noise.data(), at,
                                            scale);
      CK(cudaMemcpyAsync(xBb, next.data(), next.size() * sizeof(float), cudaMemcpyHostToDevice,
                         STREAM));
      CK(cudaMemcpyAsync(x1Bb, targetBb.data(), targetBb.size() * sizeof(float),
                         cudaMemcpyHostToDevice, STREAM));
      CK(cudaStreamSynchronize(STREAM));
    }
    // 🔴 EVERY STEP, SO A DIVERGENCE HAS A STEP NUMBER. The backbone's first step divides by
    // t = 0.001, so a difference the trunk makes in the sixth digit leaves it in the third - and
    // the end alone cannot tell that from the loop being wired wrong.
    if (!o.genOracle.empty() && design == 0) {
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
  if (int rc = finishDesign(design)) return rc;
  }
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
    else if (!f.compare(0, 6, "--job=")) o.job = f.substr(6);
    else if (!f.compare(0, 7, "--seed=")) o.seed = atoi(f.c_str() + 7);
    else if (!f.compare(0, 6, "--out=")) o.out = value("out");
    else if (!f.compare(0, 11, "--recycles=")) o.recycles = atoi(f.c_str() + 11);
    else if (!f.compare(0, 9, "--repeat=")) o.repeat = atoi(f.c_str() + 9);
    else if (!f.compare(0, 10, "--designs=")) o.designs = atoi(f.c_str() + 10);
    else if (f == "--fast") o.fast = true;
    else if (f == "--no-insertion") o.insertion = false;
    else if (!f.compare(0, 14, "--dump-sample=")) o.dumpSample = f.substr(14);
    else if (!f.compare(0, 10, "--metrics=")) o.metrics = f.substr(10);
    else if (f == "--refold") o.refold = defaultAf3();
    else if (!f.compare(0, 9, "--refold=")) o.refold = f.substr(9);
    else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
  }
  if (o.weights.empty() && o.int8.empty()) {
    fprintf(stderr, "one of --weights=<dir> or --int8=<dir> is required\n");
    return 1;
  }
  return run(o);
}

}  // namespace

// 🔴 `--job=<spec>` IS ONE PROCESS AND NO JAVASCRIPT. The featuriser is linked in, so a condition
// spec is turned into the input directory here - by the same code, with the same bytes, that
// `rfd4p-featurise` writes - and the fold reads it straight after. The directory is still written
// to disk because that is what the port reads and what a second run can reuse; what is gone is
// the second process and the Node it needed.
int jobMain(int argc, char** argv) {
  std::string job, weights, int8, outDir;
  std::vector<std::string> keep;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    if (!f.compare(0, 6, "--job=")) job = f.substr(6);
    else if (!f.compare(0, 10, "--weights=")) { weights = f.substr(10); keep.push_back(f); }
    else if (!f.compare(0, 7, "--int8=")) { int8 = f.substr(7); keep.push_back(f); }
    else if (!f.compare(0, 11, "--work-dir=")) outDir = f.substr(11);
    else keep.push_back(f);
  }
  if (outDir.empty()) outDir = job + ".featurised";
  // 🔴 THE FEATURISER READS THE PUBLISHED BUNDLE TOO, so `--int8=` alone is a whole design with no
  // float checkpoint - which is the difference between a run anybody can make and one that needs
  // NVIDIA's weight terms accepted. Every table it wants is float32 in that bundle by the
  // bundle's own rule: only the matrices are coded.
  std::vector<std::string> fargs = {"rfd4p-featurise", outDir, "--spec=" + job, "--generate",
                                    int8.empty() ? "--weights=" + weights : "--int8=" + int8};
  // A seed given to the fold is the job's seed, and both halves need it.
  for (const std::string& f : keep) {
    if (!f.compare(0, 7, "--seed=") || !f.compare(0, 9, "--config=")) fargs.push_back(f);
  }
  std::vector<char*> fargv;
  for (std::string& f : fargs) fargv.push_back(f.data());
  if (int rc = rfd4p::featurise::run((int)fargv.size(), fargv.data())) return rc;

  // 🔴 A FLAG BELONGS TO ONE HALF OR THE OTHER, and handing the fold a featuriser's flag is how
  // `--job --config=` died in silence: the featuriser ran, wrote its directory, and the fold
  // refused with "unknown flag" after the only lines anybody reads. The featuriser-only ones are
  // named here; everything else goes to the fold, so a new fold flag needs no change.
  static const char* FEATURISER_ONLY[] = {"--config=", "--spec=", "--length=", "--step="};
  std::vector<std::string> own = {argv[0], outDir, "--generate=" + outDir + "-gen"};
  // 🔴 A PARTIAL RUN NEEDS THE ENCODER'S INPUTS, AND ONLY A PARTIAL RUN. The featuriser writes
  // `<work>-enc` for one and says so in the generation's meta; handing `--encoder=` to an ordinary
  // run would encode a stale directory a previous job left in the same work dir.
  {
    std::ifstream idx(outDir + "-gen/model.idx");
    std::string kind, name;
    double value = 0;
    bool partial = false;
    while (idx >> kind >> name) {
      if (kind == "m") {
        idx >> value;
        if (name == "meta/partial") partial = value != 0;
      } else {
        std::string rest;
        std::getline(idx, rest);
      }
    }
    if (partial) own.push_back("--encoder=" + outDir + "-enc");
  }
  for (const std::string& f : keep) {
    if (f.compare(0, 2, "--") != 0) continue;
    bool theirs = false;
    for (const char* k : FEATURISER_ONLY) theirs = theirs || f.compare(0, strlen(k), k) == 0;
    if (!theirs) own.push_back(f);
  }
  std::vector<char*> ownv;
  for (std::string& f : own) ownv.push_back(f.data());
  return foldMain((int)ownv.size(), ownv.data());
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--job=", 6)) return jobMain(argc, argv);
  }
  if (argc < 2 || !strncmp(argv[1], "--", 2) || !strcmp(argv[1], "-h")) {
    fprintf(stderr, "usage: localfold-rfd4p <input dir> --weights=<dir> [--oracle=<dir>]\n"
                    "       localfold-rfd4p --job=<spec.json> --weights=<dir> [--out=<pdb>]\n");
    return 1;
  }
  return foldMain(argc, argv);
}
