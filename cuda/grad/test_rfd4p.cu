// RFD4-Proteina's trunk for training (cuda/grad/rfd4p.cuh), two ways:
//   1. float32: a token layer and a pair update, forward, against cuda/rfd4p's tokenLayer<float> and
//      pairReprUpdate<float> on its exported weights (cuda/rfd4p/weights-f32), over the valid tokens and pairs (the
//      reference zeroes a masked pair, the port leaves it; neither is read);
//   2. -DGRAD_F64: the backward against central differences on a small random token layer and pair update.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-rfd4p cuda/grad/test_rfd4p.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-rfd4p [--weights=cuda/rfd4p/weights-f32] [--n=37] [--layer=0]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-rfd4p-f64 cuda/grad/test_rfd4p.cu -lcublas -lcurand
#ifndef GRAD_F64
#include "../rfd4p/src/ops.cuh"
#include "../rfd4p/src/trunk.cuh"
#endif
#include "gradcheck.cuh"
#include "rfd4p.cuh"

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
static void masks(int n, int pad, std::vector<double>& m, std::vector<double>& pm) {
  m.assign(n, 1); for (int i = n - pad; i < n; ++i) m[i] = 0;
  pm.resize((size_t)n * n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pm[(size_t)i * n + j] = m[i] * m[j];
}

int main(int argc, char** argv) {
  int failed = 0;
  std::string weights = "cuda/rfd4p/weights-f32";
  int n = 37, layer = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--n=", 4)) n = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--layer=", 8)) layer = atoi(argv[i] + 8);
  }
#ifdef GRAD_F64
  g::init();
  {
    const int D = 24, C = 16, P = 12, H = 2, TRI = 10, Hd = 20, pad = 2;
    n = 9;
    g::rfd4p::TokenLayer tl;
    g::rfd4p::PairUpdate pu;
    g::Params ps;
    auto f = [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); };
    g::rfd4p::tokenParams(tl, "L", D, C, P, H, f);
    g::rfd4p::pairParams(pu, "U", D, P, TRI, Hd, f);
    unsigned k = 81;
    for (auto* p : ps.all) {
      bool norm = p->name.find("norm") != std::string::npos && p->name.find(".weight") != std::string::npos;
      auto v = gauss(p->n, ++k, norm ? 0.1 : 0.3);
      if (norm) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    std::vector<double> m, pm; masks(n, pad, m, pm);
    const size_t nn = (size_t)n * n;
    auto ctsH = gauss((size_t)n * D, 91, 1), ctzH = gauss(nn * P, 92, 1);
    for (int i = n - pad; i < n; ++i) for (int c = 0; c < D; ++c) ctsH[(size_t)i * D + c] = 0;
    for (size_t r = 0; r < nn; ++r) if (pm[r] == 0) for (int c = 0; c < P; ++c) ctzH[r * P + c] = 0;
    g::real *s = devReal(gauss((size_t)n * D, 93, 1)), *z = devReal(gauss(nn * P, 94, 1)), *c = devReal(gauss((size_t)n * C, 95, 1));
    g::real *mk = devReal(m), *pmk = devReal(pm), *cts = devReal(ctsH), *ctz = devReal(ctzH);
    g::real *s1, *zo, *ds, *dz, *dc;
    GK(cudaMalloc(&s1, (size_t)n * D * sizeof(g::real))); GK(cudaMalloc(&zo, nn * P * sizeof(g::real)));
    GK(cudaMalloc(&ds, (size_t)n * D * sizeof(g::real))); GK(cudaMalloc(&dz, nn * P * sizeof(g::real)));
    GK(cudaMalloc(&dc, (size_t)n * C * sizeof(g::real)));
    auto fwd = [&] {
      g::ARENA.reset();
      g::rfd4p::tokenForward(tl, s, z, c, mk, s1, n);
      g::rfd4p::puForward(pu, s1, z, mk, pmk, zo, n);
    };
    // the loss reads s1 (cts) and zo (ctz); z feeds both the attention's bias and the update
    ps.zeroGrad();
    GK(cudaMemset(dc, 0, (size_t)n * C * sizeof(g::real)));
    g::ARENA.reset();
    auto ts = g::rfd4p::tokenForward(tl, s, z, c, mk, s1, n);
    auto us = g::rfd4p::puForward(pu, s1, z, mk, pmk, zo, n);
    g::real* ds1 = g::af3::cloneR(cts, (size_t)n * D);
    g::rfd4p::puBackward(pu, us, mk, pmk, ctz, dz, ds1);      // dz <- d z through the update, ds1 += its route
    g::real* dzt = g::ARENA.zeros(nn * P);
    g::rfd4p::tokenBackward(tl, ts, mk, ds1, ds, dzt, dc, n);
    g::add(dz, dzt, nn * P);
    g::GradCheck gc;
    gc.add("in/s", s, ds, (size_t)n * D);
    gc.add("in/z", z, dz, nn * P);
    gc.add("in/c", c, dc, (size_t)n * C);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(s1, cts, (size_t)n * D);
    gc.output(zo, ctz, nn * P);
    printf("RFD4-Proteina token layer + pair update (small, random):\n");
    failed += gc.run(fwd);
  }
#else
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.load(weights);
  rfd4p::Dims d;
  d.n = n; d.D = (int)M.meta("meta/tokenDim"); d.P = (int)M.meta("meta/pairDim"); d.C = (int)M.meta("meta/condDim");
  d.TRI = (int)M.meta("meta/triMultC"); d.H = (int)M.meta("meta/nheads"); d.nlayers = (int)M.meta("meta/nlayers");
  d.useTriMult = M.meta("meta/useTriMult") != 0;
  const std::string L = "w/nn.transformer_layers." + std::to_string(layer), U = "w/nn.pair_update_layers." + std::to_string(layer);
  const int Hd = (int)M.at(U + ".transition_out.linear_1.bias").length;
  g::rfd4p::TokenLayer tl;
  g::rfd4p::PairUpdate pu;
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len);
    size_t bar = nm.find('|');
    if (bar == std::string::npos) {
      if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
      GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
    } else {        // two (out, in) matrices stacked along out
      std::string a = nm.substr(0, bar), b = nm.substr(bar + 1);
      a = a.substr(0, a.size()) + ".weight"; b = b.substr(0, b.size() - 7) + ".weight";
      size_t la = M.at(a).length;
      GK(cudaMemcpy(p.w, W(a), la * 4, cudaMemcpyDeviceToDevice));
      GK(cudaMemcpy(p.w + la, W(b), (len - la) * 4, cudaMemcpyDeviceToDevice));
    }
  };
  g::rfd4p::tokenParams(tl, L, d.D, d.C, d.P, d.H, load);
  g::rfd4p::pairParams(pu, U, d.D, d.P, d.TRI, Hd, load);
  std::vector<double> m, pm; const int pad = 3; masks(n, pad, m, pm);
  const size_t nn = (size_t)n * n;
  auto sH = gauss((size_t)n * d.D, 1, 1), zH = gauss(nn * d.P, 2, 1), cH = gauss((size_t)n * d.C, 3, 1);
  g::real *s = devReal(sH), *z = devReal(zH), *c = devReal(cH), *mk = devReal(m), *pmk = devReal(pm), *s1, *zo;
  GK(cudaMalloc(&s1, (size_t)n * d.D * 4)); GK(cudaMalloc(&zo, nn * d.P * 4));
  g::ARENA.reset();
  g::rfd4p::tokenForward(tl, s, z, c, mk, s1, n);
  g::rfd4p::puForward(pu, s1, z, mk, pmk, zo, n);
  float *rs, *rz;
  GK(cudaMalloc(&rs, (size_t)n * d.D * 4)); GK(cudaMalloc(&rz, nn * d.P * 4));
  GK(cudaMemcpy(rs, s, (size_t)n * d.D * 4, cudaMemcpyDeviceToDevice)); GK(cudaMemcpy(rz, z, nn * d.P * 4, cudaMemcpyDeviceToDevice));
  rfd4p::CondPack pack;
  rfd4p::buildCondPack(c, d, "grad.cond", pack);
  rfd4p::tokenLayer<float>(rs, rz, pack, layer, mk, pmk, d, "nn.transformer_layers." + std::to_string(layer));
  rfd4p::pairReprUpdate<float>(rs, rz, mk, pmk, d, "nn.pair_update_layers." + std::to_string(layer));
  CK(cudaStreamSynchronize(STREAM));
  auto oS = g::host(s1, (size_t)n * d.D), rS = g::host(rs, (size_t)n * d.D), oZ = g::host(zo, nn * d.P), rZ = g::host(rz, nn * d.P);
  std::vector<float> a, r, a2, r2;
  for (int i = 0; i < n - pad; ++i) {
    for (int ch = 0; ch < d.D; ++ch) { a.push_back(oS[(size_t)i * d.D + ch]); r.push_back(rS[(size_t)i * d.D + ch]); }
    for (int j = 0; j < n - pad; ++j) for (int ch = 0; ch < d.P; ++ch) { size_t q = ((size_t)i * n + j) * d.P + ch; a2.push_back(oZ[q]); r2.push_back(rZ[q]); }
  }
  double es = g::relRms(a, r), ez = g::relRms(a2, r2);
  printf("RFD4-Proteina token layer %d + pair update %d (%s), n %d: D %d, P %d, cond %d, tri %d, %d heads\n", layer, layer,
         weights.c_str(), n, d.D, d.P, d.C, d.TRI, d.H);
  printf("  against cuda/rfd4p's tokenLayer<float> + pairReprUpdate<float>: single %.3e, pair %.3e\n", es, ez);
  if (!(es < 1e-4 && ez < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
