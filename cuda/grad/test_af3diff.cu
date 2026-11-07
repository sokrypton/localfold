// The AF3 lineage's diffusion transformer for training (cuda/grad/af3diff.cuh), two ways:
//   1. float32: the whole token transformer's forward against cuda/af3's prepareTransformer + diffusionTransformer<float>
//      on a family's bundle (read through its weight walk);
//   2. -DGRAD_F64: the backward against central differences, on small random transformers under every switch
//      (q/k LayerNorm, Boltz-2's up-gate, rf3's noResidual, chai-1's AdaLN).
//
//   nvcc -O1 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/test-af3diff cuda/grad/test_af3diff.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af3diff [--bundle=model-af3-full-f32] [--family=af3] [--n=29]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-af3diff-f64 cuda/grad/test_af3diff.cu -lcublas -lcurand
#ifndef GRAD_F64
#include "../af3/src/trunk.cuh"
#include "../af3/src/atom.cuh"
#include "../af3/src/diffusion.cuh"
#include "../featurise/af3_weights.h"
#endif
#include "af3diff.cuh"
#include "gradcheck.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, r.size() * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
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

int main(int argc, char** argv) {
  int failed = 0;
#ifdef GRAD_F64
  g::init();
  struct Case { const char* what; bool kq, up, noRes, chai; };
  for (Case c : {Case{"AlphaFold 3", false, false, false, false}, Case{"q/k LayerNorm", true, false, false, false},
                 Case{"Boltz-2's up-gate", false, true, false, false}, Case{"rf3's noResidual", false, false, true, false},
                 Case{"chai-1's AdaLN", false, false, false, true}}) {
    const int n = 9, pad = 2;
    g::af3::DiffTransformer t;
    t.C = 24; t.Cc = 16; t.Cz = 8; t.H = 2; t.D = 6; t.perSuper = 2; t.factor = 2; t.noResidual = c.noRes; t.chai = c.chai;
    g::Params ps;
    g::af3::forEachParam(t, "dt", 2, c.up, c.kq, [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); });
    unsigned k = 31;
    for (auto* p : ps.all) {
      bool scale = p->name.find("Scale") != std::string::npos && p->name.find("ScaleWeights") == std::string::npos
                   && p->name.find("ScaleBias") == std::string::npos;
      auto v = gauss(p->n, ++k, scale ? 0.1 : 0.3);
      if (scale) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    std::vector<double> mk(n, 1); for (int i = n - pad; i < n; ++i) mk[i] = 0;
    auto ctH = gauss((size_t)n * t.C, 77, 1);
    for (int i = n - pad; i < n; ++i) for (int ch = 0; ch < t.C; ++ch) ctH[(size_t)i * t.C + ch] = 0;
    g::real *act = devReal(gauss((size_t)n * t.C, 71, 1)), *cond = devReal(gauss((size_t)n * t.Cc, 72, 1)),
            *pc = devReal(gauss((size_t)n * n * t.Cz, 73, 1)), *mask = devReal(mk), *ct = devReal(ctH);
    g::real *out, *dact, *dcond, *dpc;
    GK(cudaMalloc(&out, (size_t)n * t.C * sizeof(g::real))); GK(cudaMalloc(&dact, (size_t)n * t.C * sizeof(g::real)));
    GK(cudaMalloc(&dcond, (size_t)n * t.Cc * sizeof(g::real))); GK(cudaMalloc(&dpc, (size_t)n * n * t.Cz * sizeof(g::real)));
    GK(cudaMemset(dcond, 0, (size_t)n * t.Cc * sizeof(g::real))); GK(cudaMemset(dpc, 0, (size_t)n * n * t.Cz * sizeof(g::real)));
    ps.zeroGrad();
    g::af3::DiffRun run;
    g::af3::transformerForward(t, run, act, cond, mask, pc, out, n);
    g::af3::transformerBackward(t, run, cond, mask, pc, ct, dact, dcond, dpc);
    g::GradCheck gc;
    gc.add("in/act", act, dact, (size_t)n * t.C);
    gc.add("in/cond", cond, dcond, (size_t)n * t.Cc);
    gc.add("in/pairCond", pc, dpc, (size_t)n * n * t.Cz);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(out, ct, (size_t)n * t.C);
    printf("diffusion transformer, %s:\n", c.what);
    failed += gc.run([&] { g::af3::transformerForward(t, run, act, cond, mask, pc, out, n); });
  }
#else
  std::string bundle = "model-af3-full-f32", family = "af3";
  int n = 29;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--family=", 9)) family = argv[i] + 9;
    else if (!strncmp(argv[i], "--n=", 4)) n = atoi(argv[i] + 4);
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines(family, S); });
  const std::string Tn = "diffusion.transformer";
  g::af3::DiffTransformer t;
  t.C = (int)M.meta(Tn + ".channels"); t.Cc = (int)M.meta(Tn + ".condChannels"); t.Cz = (int)M.meta(Tn + ".pairChannels");
  t.H = (int)M.meta(Tn + ".heads"); t.D = (int)M.meta(Tn + ".dimension"); t.perSuper = (int)M.meta(Tn + ".blocksPerSuperBlock");
  t.factor = (int)M.meta(Tn + ".transitionFactor"); t.noResidual = M.flag(Tn + ".noResidual"); t.chai = M.flag(Tn + ".chaiAdaLn");
  int nSuper = 0; while (M.has(Tn + ".superBlocks." + std::to_string(nSuper) + ".pairLogitsProjection")) ++nSuper;
  const bool up = M.has(Tn + ".superBlocks.0.blocks.0.ffwAToB"), kq = M.has(Tn + ".superBlocks.0.blocks.0.queryLayerNormScale");
  g::Params ps;
  g::af3::forEachParam(t, Tn, nSuper, up, kq, [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
  });
  setAdaMode(t.chai);
  const int pad = 3;
  std::vector<double> mk(n, 1); for (int i = n - pad; i < n; ++i) mk[i] = 0;
  auto actH = gauss((size_t)n * t.C, 1, 1), condH = gauss((size_t)n * t.Cc, 2, 1), pcH = gauss((size_t)n * n * t.Cz, 3, 1);
  g::real *act = devReal(actH), *cond = devReal(condH), *pc = devReal(pcH), *mask = devReal(mk), *out;
  GK(cudaMalloc(&out, (size_t)n * t.C * 4));
  g::af3::DiffRun run;
  g::af3::transformerForward(t, run, act, cond, mask, pc, out, n);
  float* ref; GK(cudaMalloc(&ref, (size_t)n * t.C * 4));
  GK(cudaMemcpy(ref, act, (size_t)n * t.C * 4, cudaMemcpyDeviceToDevice));
  prepareTransformer(pc, n);
  diffusionTransformer<float>(ref, cond, mask, n);
  CK(cudaStreamSynchronize(STREAM));
  auto a = g::host(out, (size_t)(n - pad) * t.C), r = g::host(ref, (size_t)(n - pad) * t.C);
  double e = g::relRms(a, r);
  printf("%s (%s): %d blocks (%d x %d), C %d, cond %d, pair %d, %dx%d heads, transition x%d%s%s%s%s\n", family.c_str(),
         bundle.c_str(), (int)t.blocks.size(), nSuper, t.perSuper, t.C, t.Cc, t.Cz, t.H, t.D, t.factor, kq ? ", q/k norm" : "",
         up ? ", up-gate" : "", t.noResidual ? ", noResidual" : "", t.chai ? ", chai's AdaLN" : "");
  printf("  against cuda/af3's diffusionTransformer<float>: %.3e over %d real tokens\n", e, n - pad);
  if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
