// The AF3 lineage's template embedder for training (cuda/grad/af3tmpl.cuh), two ways:
//   1. float32: the template term against cuda/af3's embed<float>() on pass 0 - its z_init_generic seam in, its
//      z_after_template seam out - on a family's bundle and an input featurised with templates (6MRR and its own
//      crystal, two slots);
//   2. -DGRAD_F64: the backward against central differences on the same bundle and input, a random pair in.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/test-af3tmpl cuda/grad/test_af3tmpl.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af3tmpl [--bundle=model-af3-full-f32] [--family=af3] [--input=cuda/grad/data/af3-tmpl-af3]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 --default-stream per-thread -o cuda/grad/test-af3tmpl-f64 cuda/grad/test_af3tmpl.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af3/src/trunk.cuh"
#include "../af3/src/diffusion.cuh"
#include "../featurise/af3_weights.h"
#include "af3tmpl.cuh"
#include "gradcheck.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static g::real* devReal(const float* dev, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost));
  return devReal(std::vector<double>(h.begin(), h.end()));
}
static std::vector<float> hostF(const float* d, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h;
}
static std::vector<double> gauss(size_t n, unsigned seed, double sd) {
  std::vector<double> v(n);
  unsigned long long x = seed * 2654435761ull + 1;
  for (auto& e : v) {
    double u1, u2;
    x = x * 6364136223846793005ull + 1442695040888963407ull; u1 = ((x >> 11) + 0.5) / 9007199254740992.0;
    x = x * 6364136223846793005ull + 1442695040888963407ull; u2 = ((x >> 11) + 0.5) / 9007199254740992.0;
    e = sd * std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
  return v;
}

int main(int argc, char** argv) {
  int failed = 0;
  std::string bundle = "model-af3-full-f32", family = "af3", input = "cuda/grad/data/af3-tmpl-af3";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--family=", 9)) family = argv[i] + 9;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines(family, S); });
  M.load(input);
  setAdaMode(M.flag("trunk.dialect.chaiAtomStack"));
  AtomShape sh = atomShape();
  const int n = sh.tokens; const size_t nn = (size_t)n * n;
  auto has = [&](const std::string& k) { return M.has(k); };
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), W(nm), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  };
  // the embedder, sized from the bundle and the input
  const std::string P = "trunk.template.";
  g::af3::TemplateEmbedder m;
  m.fused = M.flag(P + "fused"); m.outer = M.flag("template.outerResidual"); m.chaiMask = M.flag("trunk.dialect.chaiTemplates");
  m.d = g::af3::Dialect{M.flag("trunk.dialect.swapTransposedBias"), M.flag("trunk.dialect.triangleMulDivideByLength")};
  const int Ct = (int)M.meta(P + "channels"), Cq = (int)(M.at(P + "queryEmbeddingNormScale").length);
  const int width = (int)M.meta("template.featureWidth");
  const int bins = m.fused ? 0 : (int)(M.at(P + "templatePairEmbedding0").length / Ct);
  int nb = 0; while (M.has(P + "blocks." + std::to_string(nb) + ".pairTransition.transition1")) ++nb;
  const std::string B0 = P + "blocks.0";
  const int Hp = (int)(M.at(B0 + ".pairAttention1.pairBiasProjection").length / Ct);
  const int Dp = (int)(M.at(B0 + ".pairAttention1.qProjection").length / Ct / Hp);
  const int pf = (int)(M.at(B0 + ".pairTransition.transition1").length / (2 * (size_t)Ct * Ct));
  const bool parallel = M.flag("trunk.dialect.parallelPairformer");
  g::af3::forEachParam(m, Cq, Ct, width, bins, nb, Hp, Dp, pf, M.has(B0 + ".pairAttention1.gatingQueryBias"), parallel, has, load);
  // the slots
  const int passes = (int)M.meta("template.passes"), templates = (int)M.meta("template.templates");
  std::vector<g::af3::TemplateSlot> slots(passes);
  int live = 0;
  for (int k = 0; k < passes; ++k) {
    const std::string S = "template." + std::to_string(k) + ".";
    g::af3::TemplateSlot& s = slots[k];
    s.repeat = (g::real)M.meta(S + "repeat");
    live += s.repeat != 0;
    if (m.fused) {
      const int K = (int)M.meta(S + "featuresK");
      std::vector<int> idx(nn * K); GK(cudaMemcpy(idx.data(), Idev(S + "featuresIdx"), idx.size() * 4, cudaMemcpyDeviceToHost));
      auto val = hostF(Fdev(S + "featuresVal"), nn * K);
      std::vector<double> dense(nn * width, 0);
      for (size_t r = 0; r < nn; ++r) for (int q = 0; q < K; ++q) if (idx[r * K + q] >= 0) dense[r * width + idx[r * K + q]] = val[r * K + q];
      s.dense = devReal(dense);
    } else {
      s.aatype = Idev(S + "aatype");
      if (M.has(S + "distogramBin")) {
        s.bin = Idev(S + "distogramBin");
        s.pb = devReal(Fdev(S + "pseudoBetaMask2d"), nn); s.uv = devReal(Fdev(S + "unitVector"), nn * 3);
        s.bb = devReal(Fdev(S + "backboneMask2d"), nn);
      }
    }
    if (m.chaiMask && M.has(S + "pseudoBetaMask2d")) s.coverage = devReal(Fdev(S + "pseudoBetaMask2d"), nn);
  }
  printf("%s (%s), %d tokens: %d slots (%d weighed, divisor %d), %s, %d blocks (%d heads of %d)%s%s%s\n", family.c_str(),
         bundle.c_str(), n, passes, live, templates, m.fused ? "fused features" : "nine projections", nb, Hp, Dp,
         m.outer ? ", outer residual" : "", parallel ? ", parallel" : "", m.chaiMask ? ", coverage-masked" : "");
  if (!live) { fprintf(stderr, "no slot weighs anything: featurise with --template\n"); return 1; }
  // the port's pass 0: the pair before and after its template term
  // (chai-1's target_feat reads ESM2 3B, which the template term never does: there the port's templateEmbedding is
  // called alone, on a random pair, its trunk made over a stand-in target_feat)
  MSA_DRAW = "prefix";
  const bool direct = M.flag("trunk.dialect.chaiTokenEmbedding");
  float *before, *after; CK(cudaMalloc(&before, nn * Cq * 4)); CK(cudaMalloc(&after, nn * Cq * 4));
  if (direct) {
    const int F = (int)M.meta("trunk.embedder.targetFeatWidth");
    std::vector<float> zt(nn * Cq); { auto gz = gauss(nn * Cq, 2, 1); for (size_t i = 0; i < zt.size(); ++i) zt[i] = (float)gz[i]; }
    float* tfStand; CK(cudaMalloc(&tfStand, (size_t)n * F * 4)); CK(cudaMemset(tfStand, 0, (size_t)n * F * 4));
    Trunk t = makeTrunk(nullptr, 64, false, tfStand);
    CK(cudaMemcpy(before, zt.data(), nn * Cq * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(t.pair, zt.data(), nn * Cq * 4, cudaMemcpyHostToDevice));
    templateEmbedding<float>(t, t.pair);
    CK(cudaMemcpyAsync(after, t.pair, nn * Cq * 4, cudaMemcpyDeviceToDevice, STREAM));
  } else {
    const float* refTf = buildTargetFeat();
    Trunk t = makeTrunk(nullptr, 64, false, refTf);
    embed<float>(t, [&](const char* seam, const float* p, size_t len) {
      if (!strcmp(seam, "z_init_generic")) CK(cudaMemcpyAsync(before, p, len * 4, cudaMemcpyDeviceToDevice, STREAM));
      if (!strcmp(seam, "z_after_template")) CK(cudaMemcpyAsync(after, p, len * 4, cudaMemcpyDeviceToDevice, STREAM));
    });
  }
  CK(cudaStreamSynchronize(STREAM));
  g::real* mask = devReal(std::vector<double>(nn, 1));
  g::real *pair = devReal(before, nn * Cq), *out;
  GK(cudaMalloc(&out, nn * Cq * sizeof(g::real)));
#ifndef GRAD_F64
  {
    GK(cudaMemcpy(out, pair, nn * Cq * sizeof(g::real), cudaMemcpyDeviceToDevice));
    g::ARENA.reset();
    g::af3::templateForward(m, slots, templates, pair, mask, n, out);
    GK(cudaStreamSynchronize(STREAM));
    auto mine = g::host(out, nn * Cq); auto port = hostF(after, nn * Cq), b0 = hostF(before, nn * Cq);
    std::vector<double> a, r;      // the template term itself, after - before
    for (size_t i = 0; i < nn * Cq; ++i) { a.push_back((double)mine[i] - b0[i]); r.push_back((double)port[i] - b0[i]); }
    double e = g::relRms(a, r);
    printf("  the template term against cuda/af3's %s %.3e\n", direct ? "templateEmbedding" : "embed (z_after_template - z_init_generic)", e);
    if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  }
#else
  {
    g::real* p0 = devReal(gauss(nn * Cq, 3, 1));
    g::real *ct = devReal(gauss(nn * Cq, 4, 1)), *dp;
    GK(cudaMalloc(&dp, nn * Cq * sizeof(g::real))); GK(cudaMemset(dp, 0, nn * Cq * sizeof(g::real)));
    auto fwd = [&] {
      g::ARENA.reset();
      GK(cudaMemsetAsync(out, 0, nn * Cq * sizeof(g::real), STREAM));
      g::af3::templateForward(m, slots, templates, p0, mask, n, out);
    };
    ps.zeroGrad();
    g::ARENA.reset();
    GK(cudaMemsetAsync(out, 0, nn * Cq * sizeof(g::real), STREAM));
    auto s = g::af3::templateForward(m, slots, templates, p0, mask, n, out);
    g::af3::templateBackward(m, slots, s, mask, ct, dp);
    g::GradCheck gc;
    gc.verbose = getenv("GRAD_VERBOSE") != nullptr;
    if (getenv("GRAD_REL")) gc.rel = atof(getenv("GRAD_REL"));
    gc.add("in/pair", p0, dp, nn * Cq);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(out, ct, nn * Cq);
    failed += gc.run(fwd);
  }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
