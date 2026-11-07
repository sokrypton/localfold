// OpenDDE's structural-token stage for training (cuda/grad/opendde.cuh), two ways:
//   1. float32: the expander and the refiner against cuda/af3's expandStructural (the refined single and pair, the
//      expanded s_inputs, the attention bias), the confidence head against its ddeConfidence (PAE, PDE, pLDDT
//      expectations) - on OpenDDE's bundle, a real input's structural layout (6MRR: 68 residues, 130 subtokens) and
//      random trunk outputs and coordinates;
//   2. -DGRAD_F64: the backward against central differences through all three, the refiner and the head one block
//      each.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/test-opendde cuda/grad/test_opendde.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-opendde [--bundle=model-opendde-full-f32] [--input=cuda/grad/data/af3-in-opendde]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 --default-stream per-thread -o cuda/grad/test-opendde-f64 cuda/grad/test_opendde.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af3/src/trunk.cuh"
#include "../af3/src/diffusion.cuh"
#include "../af3/src/sampler.cuh"
#include "../af3/src/scores.cuh"
#include "../af3/src/structural.cuh"
#include "../featurise/af3_weights.h"
#include "opendde.cuh"
#include "gradcheck.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static std::vector<double> hostD(const float* d, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return std::vector<double>(h.begin(), h.end());
}
static float* devF(const std::vector<double>& h) {
  std::vector<float> r(h.begin(), h.end());
  float* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * 4)); GK(cudaMemcpy(d, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
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
// expectations of softmax(logits) . centres, row by row
static std::vector<double> expect(const std::vector<double>& l, size_t rows, int bins, const std::vector<double>& c, double scale) {
  std::vector<double> out(rows);
  for (size_t r = 0; r < rows; ++r) {
    double mx = -1e300; for (int b = 0; b < bins; ++b) mx = std::max(mx, l[r * bins + b]);
    double s = 0, e = 0; for (int b = 0; b < bins; ++b) { double w = std::exp(l[r * bins + b] - mx); s += w; e += w * c[b]; }
    out[r] = scale * e / s;
  }
  return out;
}

int main(int argc, char** argv) {
  int failed = 0;
  std::string bundle = "model-opendde-full-f32", input = "cuda/grad/data/af3-in-opendde";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines("opendde", S); });
  M.load(input);
  if (!M.flag("trunk.dialect.structuralTokens")) { fprintf(stderr, "%s is not OpenDDE's\n", bundle.c_str()); return 1; }
  const int nRes = (int)M.meta("batch.tokens"), n = (int)M.meta("sbatch.tokens");
  const size_t nn = (size_t)n * n, rr = (size_t)nRes * nRes;
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), W(nm), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  };
  const g::af3::Dialect d{M.flag("trunk.dialect.swapTransposedBias"), M.flag("trunk.dialect.triangleMulDivideByLength")};
  // the expander
  const std::string E = "expander.";
  g::dde::Expander x;
  const int F = (int)M.meta(E + "singleInputChannels"), Cs = (int)M.meta(E + "singleChannels"), C = (int)M.meta(E + "pairChannels");
  x.flagRows = (int)(M.at(E + "sameParentEmbedding").length / C);
  g::dde::forEachParam(x, F, Cs, C, (int)(M.at(E + "rolePairTypeEmbedding").length / C), load);
  // a stack of pairformer blocks under `pre`, sized from the bundle
  auto blocksAt = [&](std::vector<g::af3::Pairformer>& bs, const std::string& pre, int Cz, int Csg, int count) {
    const std::string B0 = pre + ".0";
    const bool hasGrid = M.has(B0 + ".pairAttention1.heads");
    const int Hp = hasGrid ? (int)M.meta(B0 + ".pairAttention1.heads") : 0, Dp = hasGrid ? (int)M.meta(B0 + ".pairAttention1.dimension") : 0;
    const int Hs = (int)M.meta(B0 + ".singleAttention.heads"), Ds = (int)M.meta(B0 + ".singleAttention.dimension");
    const int pf = (int)(M.at(B0 + ".pairTransition.transition1").length / (2 * (size_t)Cz * Cz));
    const int sf = (int)(M.at(B0 + ".singleTransition.transition1").length / (2 * (size_t)Csg * Csg));
    bs.assign(count, g::af3::Pairformer{});
    for (int k = 0; k < count; ++k)
      g::af3::forEachParam(bs[k], pre + "." + std::to_string(k), Cz, Csg, Hp, Dp, Hs, Ds, pf, sf, M.has(B0 + ".pairAttention1.gatingQueryBias"),
                           true, load, hasGrid);
    return std::make_tuple(Hp, Dp, Hs, Ds, pf, sf);
  };
  int nRef = 0; while (M.has("refiner.blocks." + std::to_string(nRef) + ".singleChannels")) ++nRef;
  const std::string P = "ddeConfidence.";
  int nConf = 0; while (M.has(P + "blocks." + std::to_string(nConf) + ".singleChannels")) ++nConf;
#ifdef GRAD_F64
  nRef = std::min(nRef, 1); nConf = std::min(nConf, 1);
#endif
  std::vector<g::af3::Pairformer> refiner; blocksAt(refiner, "refiner.blocks", C, Cs, nRef);
  g::dde::Confidence h;
  h.C = (int)M.meta(P + "pairChannels"); h.Cs = (int)M.meta(P + "singleChannels"); h.F = (int)M.meta(P + "singleInputChannels");
  h.bins = (int)M.meta(P + "distanceBins"); h.paeBins = (int)M.meta(P + "paeBins"); h.pdeBins = (int)M.meta(P + "pdeBins");
  h.slots = (int)M.meta(P + "denseSlots"); h.PB = (int)M.meta(P + "plddtBins");
  {
    const std::string B0 = P + "blocks.0";
    const bool hasGrid = M.has(B0 + ".pairAttention1.heads");
    const int Hp = hasGrid ? (int)M.meta(B0 + ".pairAttention1.heads") : 0, Dp = hasGrid ? (int)M.meta(B0 + ".pairAttention1.dimension") : 0;
    g::dde::forEachParam(h, nConf, Hp, Dp, (int)M.meta(B0 + ".singleAttention.heads"), (int)M.meta(B0 + ".singleAttention.dimension"),
                         (int)(M.at(B0 + ".pairTransition.transition1").length / (2 * (size_t)h.C * h.C)),
                         (int)(M.at(B0 + ".singleTransition.transition1").length / (2 * (size_t)h.Cs * h.Cs)),
                         M.has(B0 + ".pairAttention1.gatingQueryBias"), load);
  }
  // the layout
  g::dde::Layout L;
  L.n = n; L.nRes = nRes;
  L.parent = Idev("structural.parent"); L.role = Idev("structural.role");
  L.sameParent = Idev("structural.sameParent"); L.twin = Idev("structural.twin"); L.prev = Idev("structural.prevBackbone");
  L.next = Idev("structural.nextBackbone"); L.type = Idev("structural.rolePairType");
  g::dde::buildOrder(L, M.i("structural.role"));
  printf("OpenDDE (%s): %d residues, %d structural tokens; expander s_inputs %d, single %d, pair %d; refiner %d blocks, confidence %d "
         "(pair %d, %d distance bins, %d slots)\n", bundle.c_str(), nRes, n, F, Cs, C, nRef, nConf, h.C, h.bins, h.slots);
  // the inputs: random trunk outputs and per-token coordinates; the structural batch's masks
  auto tsH = gauss((size_t)nRes * Cs, 1, 1), tpH = gauss(rr * C, 2, 1), tfH = gauss((size_t)nRes * F, 3, 1), xH = gauss((size_t)n * 3, 4, 8);
  std::vector<double> smH(M.f("sbatch.seqMask"), M.f("sbatch.seqMask") + n), pmH(nn);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pmH[(size_t)i * n + j] = smH[i] * smH[j];
  g::real *ts = devReal(tsH), *tp = devReal(tpH), *tf = devReal(tfH), *xc = devReal(xH), *sm = devReal(smH), *pm = devReal(pmH);
  g::real *sIn, *single, *pair, *bias;
  GK(cudaMalloc(&sIn, (size_t)n * F * sizeof(g::real))); GK(cudaMalloc(&single, (size_t)n * Cs * sizeof(g::real)));
  GK(cudaMalloc(&pair, nn * C * sizeof(g::real))); GK(cudaMalloc(&bias, nn * sizeof(g::real)));
  auto forward = [&](g::dde::ExpandSave& es, g::dde::RefineRun& rr_, g::dde::ConfSave& cs, g::dde::ConfLogits& o) {
    es = g::dde::expandForward(x, L, ts, tp, tf, sIn, single, pair, bias);
    g::dde::refineForward(refiner, d, rr_, pair, single, pm, sm, bias, n);
    cs = g::dde::confForward(h, d, pair, single, sIn, xc, sm, pm, bias, n, o);
  };
#ifndef GRAD_F64
  g::ARENA.reset();
  g::dde::ExpandSave es; g::dde::RefineRun run; g::dde::ConfSave cs; g::dde::ConfLogits o{};
  forward(es, run, cs, o);
  // the port's
  Structural st = expandStructural(devF(tsH), devF(tpH), devF(tfH), nRes, false);
  float *pmF = devF(pmH), *smF = devF(smH);
  DdeConfidence ref = ddeConfidence(st.pair, st.single, st.targetFeat, devF(xH), smF, pmF, st.bias, n, h.slots, nRes);
  CK(cudaStreamSynchronize(STREAM));
  auto hv = [](const g::real* p, size_t len) { auto v = g::host(p, len); return std::vector<double>(v.begin(), v.end()); };
  auto cmp = [&](const char* what, const std::vector<double>& mine, const std::vector<double>& port, double bar) {
    double e = g::relRms(mine, port);
    printf("  %-44s %.3e%s\n", what, e, e < bar ? "" : "  <-- ABOVE BAR");
    if (!(e < bar)) ++failed;
  };
  cmp("the expanded s_inputs", hv(sIn, (size_t)n * F), hostD(st.targetFeat, (size_t)n * F), 1e-5);
  cmp("the attention bias", hv(bias, nn), hostD(st.bias, nn), 1e-5);
  cmp("the refined single", hv(single, (size_t)n * Cs), hostD(st.single, (size_t)n * Cs), 1e-4);
  cmp("the refined pair", hv(pair, nn * C), hostD(st.pair, nn * C), 1e-4);
  auto centres = [](int nb) { std::vector<double> c(nb); for (int b = 0; b < nb; ++b) c[b] = 32.0 / nb * (b + 0.5); return c; };
  std::vector<double> pc(h.PB); for (int b = 0; b < h.PB; ++b) pc[b] = (b + 0.5) / h.PB;
  cmp("PAE (expected)", expect(hv(o.pae, nn * h.paeBins), nn, h.paeBins, centres(h.paeBins), 1),
      std::vector<double>(ref.pae.begin(), ref.pae.end()), 1e-4);
  cmp("PDE (expected)", expect(hv(o.pde, nn * h.pdeBins), nn, h.pdeBins, centres(h.pdeBins), 1),
      std::vector<double>(ref.pde.begin(), ref.pde.end()), 1e-4);
  cmp("pLDDT (expected)", expect(hv(o.plddt, (size_t)n * h.slots * h.PB), (size_t)n * h.slots, h.PB, pc, 100),
      std::vector<double>(ref.plddt.begin(), ref.plddt.end()), 1e-4);
#else
  g::real *ctPde = devReal(gauss(nn * h.pdeBins, 81, 1)), *ctPae = devReal(gauss(nn * h.paeBins, 82, 1)),
          *ctPl = devReal(gauss((size_t)n * h.slots * h.PB, 83, 1)), *ctS = devReal(gauss((size_t)n * Cs, 84, 1));
  g::real *oPde, *oPae, *oPl, *oS;
  GK(cudaMalloc(&oPde, nn * h.pdeBins * sizeof(g::real))); GK(cudaMalloc(&oPae, nn * h.paeBins * sizeof(g::real)));
  GK(cudaMalloc(&oPl, (size_t)n * h.slots * h.PB * sizeof(g::real))); GK(cudaMalloc(&oS, (size_t)n * Cs * sizeof(g::real)));
  auto fwd = [&] {
    g::ARENA.reset();
    g::dde::ExpandSave es; g::dde::RefineRun run; g::dde::ConfSave cs; g::dde::ConfLogits o{};
    forward(es, run, cs, o);
    GK(cudaMemcpyAsync(oPde, o.pde, nn * h.pdeBins * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(oPae, o.pae, nn * h.paeBins * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(oPl, o.plddt, (size_t)n * h.slots * h.PB * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaMemcpyAsync(oS, single, (size_t)n * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
  };
  ps.zeroGrad();
  g::ARENA.reset();
  g::dde::ExpandSave es; g::dde::RefineRun run; g::dde::ConfSave cs; g::dde::ConfLogits o{};
  forward(es, run, cs, o);
  g::real *dts, *dtp, *dtf, *dsIn, *dpair, *dsingle, *dbias;
  const std::vector<std::pair<g::real**, size_t>> grads{{&dts, (size_t)nRes * Cs}, {&dtp, rr * C}, {&dtf, (size_t)nRes * F},
                        {&dsIn, (size_t)n * F}, {&dpair, nn * C}, {&dsingle, (size_t)n * Cs}, {&dbias, nn}};
  for (auto [p, len] : grads) {
    GK(cudaMalloc(p, len * sizeof(g::real))); GK(cudaMemset(*p, 0, len * sizeof(g::real)));
  }
  GK(cudaMemcpy(dsingle, ctS, (size_t)n * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice));     // (the refined single is an output)
  g::dde::confBackward(h, d, cs, single, sIn, sm, pm, bias, g::dde::ConfLogits{ctPde, ctPae, ctPl}, dpair, dsingle, dsIn, dbias);
  g::dde::refineBackward(refiner, d, run, dpair, dsingle, pm, sm, bias, dbias);
  g::dde::expandBackward(x, L, es, tp, dsIn, dsingle, dpair, dbias, dts, dtp, dtf);
  GK(cudaStreamSynchronize(STREAM));
  g::GradCheck gc;
  gc.verbose = getenv("GRAD_VERBOSE") != nullptr;
  gc.add("in/trunkSingle", ts, dts, (size_t)nRes * Cs);
  gc.add("in/trunkPair", tp, dtp, rr * C);
  gc.add("in/targetFeat", tf, dtf, (size_t)nRes * F);
  for (auto* p : ps.all) gc.add(*p);
  gc.output(oPde, ctPde, nn * h.pdeBins); gc.output(oPae, ctPae, nn * h.paeBins);
  gc.output(oPl, ctPl, (size_t)n * h.slots * h.PB); gc.output(oS, ctS, (size_t)n * Cs);
  failed += gc.run(fwd);
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
