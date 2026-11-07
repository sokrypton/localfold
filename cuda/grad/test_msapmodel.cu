// MSA Pairformer's whole model for training (cuda/grad/msapmodel.cuh), two ways:
//   1. float32: the new pieces against cuda/msap's own on its exported weights - the MSA embedding (msaEmbedK), the
//      initial pair (pairInitK, with a chain break), both contact heads (contacts()) and the LM head (lmHead()) on
//      random inputs; the layers between are test_msap's;
//   2. -DGRAD_F64: the backward against central differences on a small random model (two layers, a contact head after
//      each, the LM head), the embeddings' tables included.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-msapmodel cuda/grad/test_msapmodel.cu -lcublas -lcublasLt -ldl -lcurand
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-msapmodel-f64 cuda/grad/test_msapmodel.cu -lcublas -lcurand
#ifndef GRAD_F64
#define main msap_main_
#include "../msap/src/msap.cu"
#undef main
#endif
#include "gradcheck.cuh"
#include "msapmodel.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static int* devI(const std::vector<int>& h) {
  int* d; GK(cudaMalloc(&d, h.size() * 4)); GK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return d;
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
  for (int i = 1; i < argc; ++i) if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
  const int S = 5, L = 23, brk = 15;            // (a chain break at 15: the shifted index and the bond across it)
  std::vector<int> shifted(L);
  for (int l = 0; l < L; ++l) shifted[l] = l + (l >= brk ? 100 : 0);
#ifdef GRAD_F64
  g::init();
  {
    g::msap::Model mo;
    mo.Cm = 20; mo.C = 12; mo.V = 7; mo.depth = 2; mo.cbLayer = 0; mo.cfLayer = 1; mo.lm.V = 5;
    g::Params ps;
    g::msap::forEachParam(mo, 2, 4, 24, [](int) { return 0.3; }, [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); });
    unsigned k = 7;
    for (auto* p : ps.all) {
      bool scale = p->name.find("Scale") != std::string::npos;
      auto v = gauss(p->n, ++k, p->name.find("lambda") != std::string::npos ? 0.1 : scale ? 0.1 : 0.3);
      if (scale) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    const size_t R = (size_t)S * L, P = (size_t)L * L;
    std::vector<int> tok(R); for (size_t r = 0; r < R; ++r) tok[r] = (int)((r * 7 + r / L) % mo.V);
    int *tokD = devI(tok), *shD = devI(shifted);
    std::vector<double> ones(std::max(R, P), 1.0);
    g::real* one = devReal(ones);
    g::real *ctCb = devReal(gauss(P, 31, 1)), *ctCf = devReal(gauss(P, 32, 1)), *ctLg = devReal(gauss((size_t)L * mo.lm.V, 33, 1));
    g::real *oCb, *oCf, *oLg;
    GK(cudaMalloc(&oCb, P * sizeof(g::real))); GK(cudaMalloc(&oCf, P * sizeof(g::real))); GK(cudaMalloc(&oLg, (size_t)L * mo.lm.V * sizeof(g::real)));
    auto fwd = [&] {
      g::ARENA.reset();
      auto r = g::msap::modelForward(mo, tokD, shD, one, one, one, S, L);
      GK(cudaMemcpyAsync(oCb, r.cbOut, P * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
      GK(cudaMemcpyAsync(oCf, r.cfOut, P * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
      GK(cudaMemcpyAsync(oLg, r.logits, (size_t)L * mo.lm.V * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
    };
    ps.zeroGrad();
    g::ARENA.reset();
    auto r = g::msap::modelForward(mo, tokD, shD, one, one, one, S, L);
    g::msap::modelBackward(mo, r, tokD, shD, one, one, one, S, L, ctCb, ctCf, ctLg);
    g::GradCheck gc;
    for (auto* p : ps.all) gc.add(*p);
    gc.output(oCb, ctCb, P); gc.output(oCf, ctCf, P); gc.output(oLg, ctLg, (size_t)L * mo.lm.V);
    printf("MSA Pairformer model (small, random: two layers, both contact heads, the LM head):\n");
    failed += gc.run(fwd);
  }
#else
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  FAST = false;
  M.load(weights);
  g::msap::Model mo;
  mo.Cm = (int)M.meta("meta/msaChannels"); mo.C = (int)M.meta("meta/pairChannels"); mo.depth = 0;
  mo.V = (int)(M.at("f/msap/msaInit").length / mo.Cm); mo.lm.V = (int)M.at("f/msap/lm/logitsBias").length;
  g::Params ps;
  g::msap::forEachParam(mo, 1, 1, 1, [](int) { return 0.0; }, [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the weights, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
  });
  const size_t R = (size_t)S * L, P = (size_t)L * L;
  std::vector<int> tok(R); for (size_t r = 0; r < R; ++r) tok[r] = (int)((r * 7 + r / L) % mo.V);
  int *tokD = devI(tok), *shD = devI(shifted);
  g::ARENA.reset();
  g::real *m = g::ARENA.get(R * mo.Cm), *z = g::ARENA.get(P * mo.C);
  g::msap::embedRowsK<<<g::blocks(R * mo.Cm), 256, 0, g::STREAM>>>(tokD, mo.msaInit.w, m, R, mo.Cm);
  g::msap::pairInitK2<<<g::blocks(P * mo.C), 256, 0, g::STREAM>>>(shD, mo.relpos.w, mo.bond.w, z, L, mo.C);
  float *rm = dalloc(R * mo.Cm), *rz = dalloc(P * mo.C);
  msaEmbedK<<<blocks(R * mo.Cm), 256, 0, STREAM>>>(tokD, F("msap/msaInit"), rm, R, mo.Cm);
  pairInitK<float><<<blocks(P * mo.C), 256, 0, STREAM>>>(shD, F("msap/relpos"), F("msap/bond"), rz, L, mo.C);
  // the heads on random inputs
  auto zH = gauss(P * mo.C, 1, 1), mH = gauss((size_t)L * mo.Cm, 2, 1);
  g::real *zr = devReal(zH), *mq = devReal(mH), *cb = g::ARENA.get(P), *cf = g::ARENA.get(P), *lg = g::ARENA.get((size_t)L * mo.lm.V);
  g::msap::contactForward(mo.cb, zr, L, cb);
  g::msap::contactForward(mo.cf, zr, L, cf);
  g::msap::lmForward(mo.lm, mq, L, lg);
  float *rcb = dalloc(P), *rcf = dalloc(P), *rlg = dalloc((size_t)L * 26);
  contacts(zr, L, "msap/cbContact/", rcb);
  contacts(zr, L, "msap/confindContact/", rcf);
  lmHead(mq, L, rlg);
  CK(cudaStreamSynchronize(STREAM));
  auto cmp = [&](const char* what, const g::real* mine, const float* port, size_t n) {
    auto a = g::host(mine, n), b = g::host(port, n);
    double e = g::relRms(a, b), rms = 0;
    for (float v : b) rms += (double)v * v;
    rms = std::sqrt(rms / n);
    printf("  %-26s %.3e  (rms %.3e)\n", what, e, rms);
    if (!(rms > 0)) { printf("  <-- THE REFERENCE IS ZERO\n"); ++failed; }
    if (!(e < 1e-5)) { printf("  <-- ABOVE 1e-5\n"); ++failed; }
  };
  printf("MSA Pairformer's embeddings and heads (%s), %d rows x %d, a chain break at %d\n", weights.c_str(), S, L, brk);
  cmp("msa embedding", m, rm, R * mo.Cm);
  cmp("initial pair", z, rz, P * mo.C);
  cmp("Cb contacts", cb, rcb, P);
  cmp("ConFind contacts", cf, rcf, P);
  cmp("LM logits", lg, rlg, (size_t)L * mo.lm.V);
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
