// MSA Pairformer's layer for training (cuda/grad/msap.cuh), two ways:
//   1. float32: a layer's forward (pair-weighted averaging, transition, the query-biased outer product, the pair
//      block) against cuda/msap's own float32 path - msaPairWeightedAveraging, rowsTransition, qOpmF32, trunkBlock -
//      on the exported weights (cuda/msap/weights);
//   2. -DGRAD_F64: the backward against central differences on a small random layer.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-msap cuda/grad/test_msap.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-msap [--weights=cuda/msap/weights] [--layer=0] [--S=9] [--L=31]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-msap-f64 cuda/grad/test_msap.cu -lcublas -lcurand
#ifndef GRAD_F64
#define main msap_main_
#include "../msap/src/msap.cu"
#undef main
#endif
#include "gradcheck.cuh"
#include "msap.cuh"

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
  std::string weights = "cuda/msap/weights";
  int layer = 0, S = 9, L = 31;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--layer=", 8)) layer = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--S=", 4)) S = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--L=", 4)) L = atoi(argv[i] + 4);
  }
#ifdef GRAD_F64
  g::init();
  {
    const int Cm = 20, Cz = 12, H = 2, D = 4, I = 24;
    S = 6; L = 7;
    g::msap::Layer l;
    g::Params ps;
    g::msap::forEachParam(l, "L", Cm, Cz, H, D, I, 0.3, [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); });
    unsigned k = 41;
    for (auto* p : ps.all) {
      bool scale = p->name.find("Scale") != std::string::npos;
      auto v = gauss(p->n, ++k, p->name.find("lambda") != std::string::npos ? 0.1 : scale ? 0.1 : 0.3);
      if (scale) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    const size_t R = (size_t)S * L, P = (size_t)L * L;
    std::vector<double> ones(std::max(R, P), 1.0);
    g::real *m = devReal(gauss(R * Cm, 51, 1)), *pair = devReal(gauss(P * Cz, 52, 1)), *one = devReal(ones);
    g::real *ctm = devReal(gauss(R * Cm, 53, 1)), *ctp = devReal(gauss(P * Cz, 54, 1));
    g::real *mo, *po, *dm, *dp;
    GK(cudaMalloc(&mo, R * Cm * sizeof(g::real))); GK(cudaMalloc(&po, P * Cz * sizeof(g::real)));
    GK(cudaMalloc(&dm, R * Cm * sizeof(g::real))); GK(cudaMalloc(&dp, P * Cz * sizeof(g::real)));
    ps.zeroGrad();
    g::ARENA.reset();
    auto sv = g::msap::layerForward(l, m, pair, one, one, one, mo, po, S, L, Cz);
    g::msap::layerBackward(l, sv, one, one, one, ctm, ctp, dm, dp, S, L, Cz);
    g::GradCheck gc;
    gc.add("in/msa", m, dm, R * Cm);
    gc.add("in/pair", pair, dp, P * Cz);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(mo, ctm, R * Cm);
    gc.output(po, ctp, P * Cz);
    printf("MSA Pairformer layer (small, random):\n");
    failed += gc.run([&] { g::ARENA.reset(); g::msap::layerForward(l, m, pair, one, one, one, mo, po, S, L, Cz); });
  }
#else
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  FAST = false;
  M.load(weights);
  const int Cm = (int)M.meta("meta/msaChannels"), Cz = (int)M.meta("meta/pairChannels");
  const int Hh = (int)M.meta("meta/msaHeads"), D = (int)M.meta("meta/msaHeadWidth");
  const std::string pre = "f/msap/blocks/" + std::to_string(layer);
  const int I = (int)(M.at(pre + "/msaTransition/transition2").length / Cm);
  g::msap::Layer l;
  g::Params ps;
  g::msap::forEachParam(l, pre, Cm, Cz, Hh, D, I, M.meta("meta/opmLambdaInit/" + std::to_string(layer)),
                        [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the weights, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
  });
  const size_t R = (size_t)S * L, P = (size_t)L * L;
  auto mH = gauss(R * Cm, 1, 1), pH = gauss(P * Cz, 2, 1);
  std::vector<double> ones(std::max(R, P), 1.0);
  g::real *m = devReal(mH), *pair = devReal(pH), *one = devReal(ones), *mo, *po;
  GK(cudaMalloc(&mo, R * Cm * 4)); GK(cudaMalloc(&po, P * Cz * 4));
  g::ARENA.reset();
  g::msap::layerForward(l, m, pair, one, one, one, mo, po, S, L, Cz);
  float *rm, *rp, *w;
  GK(cudaMalloc(&rm, R * Cm * 4)); GK(cudaMalloc(&rp, P * Cz * 4)); GK(cudaMalloc(&w, S * 4));
  GK(cudaMemcpy(rm, m, R * Cm * 4, cudaMemcpyDeviceToDevice)); GK(cudaMemcpy(rp, pair, P * Cz * 4, cudaMemcpyDeviceToDevice));
  const std::string B = "msap/blocks/" + std::to_string(layer) + "/";
  msaPairWeightedAveraging(rm, rp, S, L, Cm, Cz, B + "pwa/");
  rowsTransition(rm, R, Cm, B + "msaTransition/");
  qOpmF32(rp, rm, S, L, layer, w);
  trunkBlock(rp, one, L, Cz, "msap/blocks", layer);
  CK(cudaStreamSynchronize(STREAM));
  double em = g::relRms(g::host(mo, R * Cm), g::host(rm, R * Cm)), ep = g::relRms(g::host(po, P * Cz), g::host(rp, P * Cz));
  printf("MSA Pairformer layer %d (%s), S %d L %d: Cm %d, Cz %d, PWA %dx%d, transition %d\n", layer, weights.c_str(), S, L, Cm,
         Cz, Hh, D, I);
  printf("  against cuda/msap's float32 path: msa %.3e, pair %.3e\n", em, ep);
  if (!(em < 1e-4 && ep < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
