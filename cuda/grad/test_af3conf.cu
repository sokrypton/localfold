// The AF3 lineage's confidence head for training (cuda/grad/af3conf.cuh), two ways:
//   1. float32: its PDE, PAE and pLDDT logits - taken to the expectations cuda/af3's confidenceHead() returns - against
//      that head on a family's bundle, a real featurised input's target_feat (buildTargetFeat) and random trunk outputs
//      and pseudo-beta positions;
//   2. -DGRAD_F64: the backward against central differences on the same head and input.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/test-af3conf cuda/grad/test_af3conf.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af3conf [--bundle=model-af3-full-f32] [--family=af3] [--input=cuda/grad/data/af3-in-af3]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 --default-stream per-thread -o cuda/grad/test-af3conf-f64 cuda/grad/test_af3conf.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af3/src/trunk.cuh"
#include "../af3/src/diffusion.cuh"
#include "../af3/src/sampler.cuh"
#include "../af3/src/scores.cuh"
#include "../af3/src/confidence.cuh"
#include "../featurise/af3_weights.h"
#include "af3conf.cuh"
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
static float* devF(const std::vector<double>& h) {
  std::vector<float> r(h.begin(), h.end());
  float* d; GK(cudaMalloc(&d, r.size() * 4)); GK(cudaMemcpy(d, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
  return d;
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
// softmax(row) . centres
static std::vector<double> expect(const std::vector<double>& lg, size_t rows, int bins, const std::vector<double>& centres, double scale) {
  std::vector<double> o(rows);
  for (size_t r = 0; r < rows; ++r) {
    double mx = -1e300, s = 0, w = 0;
    for (int b = 0; b < bins; ++b) mx = std::max(mx, lg[r * bins + b]);
    for (int b = 0; b < bins; ++b) { double e = std::exp(lg[r * bins + b] - mx); s += e; w += e * centres[b]; }
    o[r] = scale * w / s;
  }
  return o;
}

int main(int argc, char** argv) {
  int failed = 0;
  std::string bundle = "model-af3-full-f32", family = "af3", input = "cuda/grad/data/af3-in-af3";
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
  if (M.flag("trunk.dialect.structuralTokens")) { fprintf(stderr, "%s: OpenDDE's own head is not af3conf.cuh's\n", family.c_str()); return 1; }
  const bool chai = M.flag("trunk.dialect.chaiConfidence");
  const int n = (int)M.meta("batch.tokens"), dense = chai ? 37 : (int)M.meta("batch.dense");   // (chai-1: its ATOM37 slots)
  const size_t nn = (size_t)n * n;
  const std::string P = "confidence";
  g::af3::ConfidenceHead h;
  h.C = (int)M.meta(P + ".pairChannels"); h.Cs = (int)M.meta(P + ".singleChannels"); h.F = (int)M.meta(P + ".targetFeatWidth");
  h.reembed = M.flag("trunk.dialect.reembedConfidencePair"); h.globalNorm = M.flag("trunk.dialect.confidenceGlobalNorm");
  h.binMode = chai ? g::af3::BIN_CHAI : M.flag("trunk.dialect.confidenceCaDgram") ? g::af3::BIN_CA : g::af3::BIN_AF3;
  h.unmasked = chai;
  h.bins = h.reembed ? 64 : (int)(M.at(P + ".distogramFeatProject").length / h.C); h.dense = dense;
  h.preSym = M.flag("trunk.dialect.preSymmetrisedPde");
  h.d = g::af3::Dialect{M.flag("trunk.dialect.swapTransposedBias"), M.flag("trunk.dialect.triangleMulDivideByLength")};
  int nb = 0; while (M.has(P + ".blocks." + std::to_string(nb) + ".singleChannels")) ++nb;
  const std::string B0 = P + ".blocks.0";
  const bool hasGrid = M.has(B0 + ".pairAttention1.heads"), parallel = M.flag("trunk.dialect.parallelPairformer");
  const int Hp = hasGrid ? (int)M.meta(B0 + ".pairAttention1.heads") : 0, Dp = hasGrid ? (int)M.meta(B0 + ".pairAttention1.dimension") : 0;
  const int Hs = (int)M.meta(B0 + ".singleAttention.heads"), Ds = (int)M.meta(B0 + ".singleAttention.dimension");
  const int pf = (int)(M.at(B0 + ".pairTransition.transition1").length / (2 * (size_t)h.C * h.C));
  const int sf = (int)(M.at(B0 + ".singleTransition.transition1").length / (2 * (size_t)h.Cs * h.Cs));
  const bool gridBiases = M.has(B0 + ".pairAttention1.gatingQueryBias");
  auto has = [&](const std::string& k) { return M.has(k); };
  g::Params ps;
  g::af3::forEachParam(h, nb, Hp, Dp, Hs, Ds, pf, sf, gridBiases, hasGrid, parallel, has, [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    std::vector<float> hh(len); GK(cudaMemcpy(hh.data(), W(nm), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(hh.begin(), hh.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  });
  setAdaMode(M.flag("trunk.dialect.chaiAtomStack"));
  // (chai-1's target_feat reads ESM2 3B, which the head need not: a random one there)
  const float* tfF = chai ? nullptr : buildTargetFeat();
  if (chai) { auto r = gauss((size_t)n * h.F, 4, 1); std::vector<float> f(r.begin(), r.end()); float* d = dalloc(f.size()); CK(cudaMemcpy(d, f.data(), f.size() * 4, cudaMemcpyHostToDevice)); tfF = d; }
  CK(cudaStreamSynchronize(STREAM));
  // what the re-embedding and the split heads read
  g::af3::ConfidenceBatch cb{Idev("batch.features.residueIndex"), Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"),
                             Idev("batch.features.entityId"), Idev("batch.features.symId"),
                             M.has("batch.bondMatrix") ? devReal(Fdev("batch.bondMatrix"), nn) : nullptr,
                             M.has("batch.bondOrderMatrix") ? devReal(Fdev("batch.bondOrderMatrix"), nn) : nullptr};
  auto zH = gauss(nn * h.C, 1, 1), sH = gauss((size_t)n * h.Cs, 2, 1), bH = gauss((size_t)n * 3, 3, 10);
  std::vector<double> smH(n, 1), pmH(nn, 1);
  g::real *z = devReal(zH), *sg = devReal(sH), *beta = devReal(bH), *tf = devReal(tfF, (size_t)n * h.F),
          *sm = devReal(smH), *pm = devReal(pmH);
  printf("%s (%s), %d tokens: %d blocks, pair %d, single %d, target_feat %d, %d distance bins%s%s%s%s%s%s\n", family.c_str(), bundle.c_str(),
         n, nb, h.C, h.Cs, h.F, h.bins, h.preSym ? ", pre-symmetrised PDE" : "", h.singleNorm ? ", clamped single norm" : "",
         h.reembed ? ", re-embedded" : "", h.split ? ", split chain heads" : "", h.globalNorm ? (h.absent ? ", global norms (absent columns)" : ", global norms") : "",
         chai ? ", chai-1's distance embedding and ATOM37 pLDDT" : "");
#ifndef GRAD_F64
  g::ARENA.reset();
  g::af3::ConfidenceLogits o{};
  g::af3::confidenceForward(h, z, sg, tf, beta, sm, pm, pm, n, o, cb);
  ConfidenceOut ref = confidenceHead(devF(zH), devF(sH), tfF, devF(bH), devF(smH), devF(pmH), n);
  CK(cudaStreamSynchronize(STREAM));
  const int NB = 64; const double step = 31.0 / (NB - 2);
  std::vector<double> dc(NB), pc(50);
  for (int b = 0; b < NB - 1; ++b) dc[b] = b * step + step / 2;
  dc[NB - 1] = dc[NB - 2] + step;
  for (int b = 0; b < 50; ++b) pc[b] = 0.5 / 50 + (double)b / 50;
  auto hv = [](const g::real* d, size_t len) { auto v = g::host(d, len); return std::vector<double>(v.begin(), v.end()); };
  auto pae = expect(hv(o.pae, nn * NB), nn, NB, dc, 1), pde = expect(hv(o.pde, nn * NB), nn, NB, dc, 1);
  auto pl = expect(hv(o.plddt, (size_t)n * dense * 50), (size_t)n * dense, 50, pc, 100);
  if (chai) {     // each dense slot's from its ATOM37 slot (the port's atom37Index)
    const int D = (int)M.meta("batch.dense");
    std::vector<int> idx((size_t)n * D); CK(cudaMemcpy(idx.data(), atom37Index(n, D), idx.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<double> g((size_t)n * D);
    for (int t = 0; t < n; ++t) for (int k = 0; k < D; ++k) g[(size_t)t * D + k] = pl[(size_t)t * 37 + idx[(size_t)t * D + k]];
    pl.swap(g);
  }
  if (M.flag("trunk.dialect.perAtomHeadsByRank")) {       // each atom's by its rank among the token's real atoms (the port's)
    const int* chars = M.i("batch.refAtomNameChars");
    std::vector<double> ranked(pl.size());
    for (int t = 0; t < n; ++t) {
      int rank = -1;
      for (int s = 0; s < dense; ++s) {
        size_t flat = (size_t)t * dense + s;
        if (chars[flat * 4] || chars[flat * 4 + 1] || chars[flat * 4 + 2] || chars[flat * 4 + 3]) ++rank;
        ranked[flat] = pl[(size_t)t * dense + std::min(std::max(rank, 0), dense - 1)];
      }
    }
    pl.swap(ranked);
  }
  // pLDDT's bar is the port's own conditioning: its single track reads the pair through the single attention's bias,
  // and on random inputs a 1e-7 move of the pair moves intellifold2's pLDDT logits 2e-4 (openbind0's 4e-3) - the
  // port against itself, both inputs moved by float32's last place, measured here every run
  double sens = 0;
  {
    std::vector<float> keep(ref.plddt);
    auto n1 = gauss(zH.size(), 92, 1e-7), n2 = gauss(sH.size(), 91, 1e-7);
    auto z2 = zH, s2 = sH;
    for (size_t i = 0; i < z2.size(); ++i) z2[i] *= 1 + n1[i];
    for (size_t i = 0; i < s2.size(); ++i) s2[i] *= 1 + n2[i];
    ConfidenceOut r2 = confidenceHead(devF(z2), devF(s2), tfF, devF(bH), devF(smH), devF(pmH), n);
    sens = g::relRms(r2.plddt, keep);
  }
  auto cmp = [&](const char* what, const std::vector<double>& mine, const std::vector<float>& port, double bar) {
    double e = g::relRms(mine, port);
    printf("  %-20s %.3e  (bar %.1e)\n", what, e, bar);
    if (!(e < bar)) { printf("  <-- ABOVE ITS BAR\n"); ++failed; }
  };
  cmp("PAE (expected)", pae, ref.pae, 1e-4);
  cmp("PDE (expected)", pde, ref.pde, 1e-4);
  printf("  (the port's pLDDT against itself, its inputs moved 1e-7: %.3e)\n", sens);
  cmp("pLDDT (expected)", pl, ref.plddt, std::max(1e-4, 4 * sens));
#else
  auto ctD = gauss(nn * 64, 81, 1), ctA = gauss(nn * 64, 82, 1), ctL = gauss((size_t)n * dense * 50, 83, 1);
  g::real *ctPde = devReal(ctD), *ctPae = devReal(ctA), *ctPl = devReal(ctL);
  g::real *oPde, *oPae, *oPl, *dz, *ds, *dtf;
  GK(cudaMalloc(&oPde, nn * 64 * sizeof(g::real))); GK(cudaMalloc(&oPae, nn * 64 * sizeof(g::real)));
  GK(cudaMalloc(&oPl, (size_t)n * dense * 50 * sizeof(g::real)));
  GK(cudaMalloc(&dz, nn * h.C * sizeof(g::real))); GK(cudaMalloc(&ds, (size_t)n * h.Cs * sizeof(g::real)));
  GK(cudaMalloc(&dtf, (size_t)n * h.F * sizeof(g::real)));
  GK(cudaMemset(dz, 0, nn * h.C * sizeof(g::real))); GK(cudaMemset(ds, 0, (size_t)n * h.Cs * sizeof(g::real)));
  GK(cudaMemset(dtf, 0, (size_t)n * h.F * sizeof(g::real)));
  auto fwd = [&] {
    g::ARENA.reset();
    g::af3::ConfidenceLogits o{};
    g::af3::confidenceForward(h, z, sg, tf, beta, sm, pm, pm, n, o, cb);
    GK(cudaMemcpyAsync(oPde, o.pde, nn * 64 * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
    GK(cudaMemcpyAsync(oPae, o.pae, nn * 64 * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
    GK(cudaMemcpyAsync(oPl, o.plddt, (size_t)n * dense * 50 * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
  };
  ps.zeroGrad();
  g::ARENA.reset();
  g::af3::ConfidenceLogits o{};
  auto s = g::af3::confidenceForward(h, z, sg, tf, beta, sm, pm, pm, n, o, cb);
  g::af3::confidenceBackward(h, s, z, sg, tf, sm, pm, pm, g::af3::ConfidenceLogits{ctPde, ctPae, ctPl}, dz, ds, dtf, cb);
  g::GradCheck gc;
  gc.add("in/trunkPair", z, dz, nn * h.C);
  gc.add("in/trunkSingle", sg, ds, (size_t)n * h.Cs);
  gc.add("in/targetFeat", tf, dtf, (size_t)n * h.F);
  for (auto* p : ps.all) gc.add(*p);
  gc.output(oPde, ctPde, nn * 64); gc.output(oPae, ctPae, nn * 64); gc.output(oPl, ctPl, (size_t)n * dense * 50);
  failed += gc.run(fwd);
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
