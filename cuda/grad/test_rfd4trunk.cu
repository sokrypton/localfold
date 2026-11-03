// RFD4-Proteina's whole trunk for training (cuda/grad/rfd4trunk.cuh), two ways:
//   1. float32: one cycle against cuda/rfd4p's forward() (FP32, one recycle) on its exported weights and a real input
//      (cuda/rfd4p/data-n54): c, s_init, z_init, every layer's s and z, the CA coordinates, local latents, sequence
//      logits and insertion logits;
//   2. -DGRAD_F64: the backward against central differences on a small random trunk (two layers, an update between),
//      every head a cotangent.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-rfd4trunk cuda/grad/test_rfd4trunk.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-rfd4trunk [--weights=cuda/rfd4p/weights-f32] [--input=cuda/rfd4p/data-n54]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-rfd4trunk-f64 cuda/grad/test_rfd4trunk.cu -lcublas -lcurand
#ifndef GRAD_F64
#include "../rfd4p/src/ops.cuh"
#include "../rfd4p/src/trunk.cuh"
#endif
#include "gradcheck.cuh"
#include "rfd4trunk.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
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
  std::string weights = "cuda/rfd4p/weights-f32", input = "cuda/rfd4p/data-n54";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
#ifdef GRAD_F64
  g::init();
  {
    const int n = 7, pad = 1;
    g::rfd4p::Trunk t;
    t.D = 24; t.P = 12; t.C = 16; t.PC = 8; t.TRI = 10; t.H = 2; t.Hd = 20; t.nlayers = 2; t.everyN = 1;
    t.seqF = 9; t.condF = 11; t.pairF = 7; t.pairCondF = 5; t.classes = 6; t.latent = 4;
    g::Params ps;
    g::rfd4p::forEachParam(t, [&](const std::string& nm, g::Param& p, size_t len) {
      // the triangle's stacked projections arrive as "a|b": one tensor here
      g::allocParam(p, nm, len); ps.add(p);
    });
    unsigned k = 41;
    for (auto* p : ps.all) {
      bool norm = p->name.find("norm") != std::string::npos || p->name.find("ln_out") != std::string::npos ||
                  p->name.find("ca_linear.0") != std::string::npos || p->name.find("latents_linear.0") != std::string::npos ||
                  p->name.find("readout.0") != std::string::npos;
      auto v = gauss(p->n, ++k, norm ? 0.1 : 0.3);
      if (norm) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    const size_t nn = (size_t)n * n;
    std::vector<double> m(n, 1), pmH(nn);
    for (int i = n - pad; i < n; ++i) m[i] = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pmH[(size_t)i * n + j] = m[i] * m[j];
    g::real *seq = devReal(gauss((size_t)n * t.seqF, 51, 1)), *cond = devReal(gauss((size_t)n * t.condF, 52, 1)),
            *pf = devReal(gauss(nn * t.pairF, 53, 1)), *pcf = devReal(gauss(t.pairCondF, 54, 1)), *mk = devReal(m), *pm = devReal(pmH);
    auto zp = [&](std::vector<double> v, size_t per) { for (int i = n - pad; i < n; ++i) for (size_t c = 0; c < per; ++c) v[(size_t)i * per + c] = 0; return v; };
    auto zpp = [&](std::vector<double> v) { for (size_t r = 0; r < nn; ++r) if (pmH[r] == 0) v[r] = 0; return v; };
    g::real *ctCa = devReal(zp(gauss((size_t)n * 3, 61, 1), 3)), *ctLa = devReal(zp(gauss((size_t)n * t.latent, 62, 1), t.latent)),
            *ctLo = devReal(zp(gauss((size_t)n * t.classes, 63, 1), t.classes)), *ctIn = devReal(zpp(gauss(nn, 64, 1)));
    g::real *oCa, *oLa, *oLo, *oIn;
    GK(cudaMalloc(&oCa, (size_t)n * 3 * sizeof(g::real))); GK(cudaMalloc(&oLa, (size_t)n * t.latent * sizeof(g::real)));
    GK(cudaMalloc(&oLo, (size_t)n * t.classes * sizeof(g::real))); GK(cudaMalloc(&oIn, nn * sizeof(g::real)));
    auto fwd = [&] {
      g::ARENA.reset();
      g::rfd4p::TrunkOut o{};
      g::rfd4p::trunkForward(t, seq, cond, pf, pcf, mk, pm, n, o);
      auto cp = [&](g::real* d, const g::real* s, size_t len) { GK(cudaMemcpyAsync(d, s, len * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM)); };
      cp(oCa, o.ca, (size_t)n * 3); cp(oLa, o.latents, (size_t)n * t.latent); cp(oLo, o.logits, (size_t)n * t.classes); cp(oIn, o.insertion, nn);
    };
    ps.zeroGrad();
    g::ARENA.reset();
    g::rfd4p::TrunkOut o{};
    auto s = g::rfd4p::trunkForward(t, seq, cond, pf, pcf, mk, pm, n, o);
    g::rfd4p::trunkBackward(t, s, seq, cond, pf, pcf, mk, pm, g::rfd4p::TrunkOut{ctCa, ctLa, ctLo, ctIn});
    g::GradCheck gc;
    for (auto* p : ps.all) gc.add(*p);
    gc.output(oCa, ctCa, (size_t)n * 3); gc.output(oLa, ctLa, (size_t)n * t.latent);
    gc.output(oLo, ctLo, (size_t)n * t.classes); gc.output(oIn, ctIn, nn);
    printf("RFD4-Proteina trunk (small, random: two layers, an update between):\n");
    failed += gc.run(fwd);
  }
#else
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  CB(cublasSetMathMode(H, CUBLAS_PEDANTIC_MATH));
  g::STREAM = STREAM;
  g::init();
  M.load(weights);
  M.load(input);
  rfd4p::Dims d;
  d.n = (int)M.meta("meta/n"); d.D = (int)M.meta("meta/tokenDim"); d.P = (int)M.meta("meta/pairDim");
  d.C = (int)M.meta("meta/condDim"); d.PC = (int)M.meta("meta/pairCondDim"); d.TRI = (int)M.meta("meta/triMultC");
  d.H = (int)M.meta("meta/nheads"); d.nlayers = (int)M.meta("meta/nlayers"); d.everyN = (int)M.meta("meta/updatePairEveryN");
  d.useTriMult = M.meta("meta/useTriMult") != 0; d.updatePairRepr = M.meta("meta/updatePairRepr") != 0;
  d.classes = (int)M.meta("meta/seqReadoutClasses"); d.latent = (int)M.meta("meta/latentDim");
  d.seqF = (int)M.meta("meta/seqFeatDim"); d.condF = (int)M.meta("meta/condFeatDim");
  d.pairF = (int)M.meta("meta/pairFeatDim"); d.pairCondF = (int)M.meta("meta/pairCondFeatDim");
  const int n = d.n; const size_t nn = d.pairs();
  g::rfd4p::Trunk t;
  t.D = d.D; t.P = d.P; t.C = d.C; t.PC = d.PC; t.TRI = d.TRI; t.H = d.H; t.nlayers = d.nlayers; t.everyN = d.everyN;
  t.updatePair = d.updatePairRepr; t.seqF = d.seqF; t.condF = d.condF; t.pairF = d.pairF; t.pairCondF = d.pairCondF;
  t.classes = d.classes; t.latent = d.latent; t.insertion = M.has("w/nn.insertion_q_proj.weight");
  t.Hd = (int)M.at("w/nn.pair_update_layers.0.transition_out.linear_1.bias").length;
  if (!d.useTriMult) { fprintf(stderr, "a checkpoint without triangle multiplications is not here\n"); return 1; }
  g::Params ps;
  g::rfd4p::forEachParam(t, [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    size_t bar = nm.find('|');
    if (bar == std::string::npos) {
      if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
      GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
    } else {
      std::string a = nm.substr(0, bar) + ".weight", b = nm.substr(bar + 1); b = b.substr(0, b.size() - 7) + ".weight";
      size_t la = M.at(a).length;
      GK(cudaMemcpy(p.w, W(a), la * 4, cudaMemcpyDeviceToDevice));
      GK(cudaMemcpy(p.w + la, W(b), (len - la) * 4, cudaMemcpyDeviceToDevice));
    }
  });
  float* pmD; GK(cudaMalloc(&pmD, nn * 4));
  rfd4p::pairMaskK<<<blocks(nn), 256, 0, STREAM>>>(Fdev("in/mask"), pmD, n);
  g::ARENA.reset();
  g::rfd4p::TrunkOut o{};
  auto s = g::rfd4p::trunkForward(t, Fdev("in/seq"), Fdev("in/cond0"), Fdev("in/pair"), Fdev("in/pairCond"), Fdev("in/mask"), pmD, n, o);
  // the port, one cycle, every stage tapped
  rfd4p::Trunk pt; pt.d = d; pt.mask = Fdev("in/mask"); pt.pairMask = pmD;
  GK(cudaMalloc(&pt.s, (size_t)n * d.D * 4)); GK(cudaMalloc(&pt.z, nn * d.P * 4));
  GK(cudaMalloc(&pt.bbCa, (size_t)n * 3 * 4)); GK(cudaMalloc(&pt.localLatents, (size_t)n * d.latent * 4));
  GK(cudaMalloc(&pt.logits, (size_t)n * d.classes * 4)); GK(cudaMalloc(&pt.insertion, nn * 4));
  std::map<std::string, std::vector<float>> taps;
  rfd4p::forward(pt, Fdev("in/seq"), Fdev("in/cond0"), Fdev("in/cond1"), Fdev("in/pair"), Fdev("in/pairCond"), 1, t.insertion,
                 [&](const char* nm, const float* p, size_t len) { taps[nm] = g::host(p, len); });
  CK(cudaStreamSynchronize(STREAM));
  auto cmp = [&](const std::string& what, const g::real* mine, const std::vector<float>& port) {
    double e = g::relRms(g::host(mine, port.size()), port);
    printf("  %-22s %.3e\n", what.c_str(), e);
    if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  };
  printf("RFD4-Proteina trunk (%s, %s), %d tokens, %d layers, one cycle\n", weights.c_str(), input.c_str(), n, t.nlayers);
  cmp("c", s.c, taps["o/c"]); cmp("s_init", s.sInit, taps["o/sInit"]); cmp("z_init", s.zInit, taps["o/zInit"]);
  for (int i = 0; i < t.nlayers; i += 3) cmp("s" + std::to_string(i), s.s[i + 1], taps["o/s" + std::to_string(i)]);
  cmp("s (last layer)", s.s.back(), taps["o/s" + std::to_string(t.nlayers - 1)]);
  cmp("z (last layer)", s.z.back(), taps["o/z"]);
  cmp("CA", o.ca, g::host(pt.bbCa, (size_t)n * 3));
  cmp("local latents", o.latents, g::host(pt.localLatents, (size_t)n * d.latent));
  cmp("sequence logits", o.logits, g::host(pt.logits, (size_t)n * d.classes));
  if (t.insertion) cmp("insertion logits", o.insertion, g::host(pt.insertion, nn));
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
