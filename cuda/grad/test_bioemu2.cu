// BioEmu2's trunk for training (cuda/grad/bioemu2.cuh), two ways:
//   1. float32: an MSA layer and a PairMixer layer, forward, against cuda/bioemu2's own pairWeightedAveraging,
//      transition, outerProductMean and trimul on the published bundle (model-bioemu2-int5: both sides read the same
//      decoded weights);
//   2. -DGRAD_F64: the backward against central differences on small random layers.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-bioemu2 cuda/grad/test_bioemu2.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-bioemu2 [--bundle=<dir with bioemu2.bin.zst: af3-any-model's float32 blob, ~/ported/bioemu2>] [--S=5] [--T=29] [--layer=0]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-bioemu2-f64 cuda/grad/test_bioemu2.cu -lcublas -lcurand
#ifndef GRAD_F64
#include "../bioemu2/src/layers.cuh"
#include "../bioemu2/src/trunk.cuh"
#endif
#include "bioemu2.cuh"
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
__global__ void transposeIntoK(const float* in, g::real* out, int R, int C) {   // in [R, C] -> out [C, R]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)R * C) return;
  int r = t / C, c = t % C;
  out[(size_t)c * R + r] = (g::real)in[t];
}
__global__ void toRealK(const float* x, g::real* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = (g::real)x[t];
}

int main(int argc, char** argv) {
  int failed = 0;
  std::string bundle = getenv("BIOEMU2_WEIGHTS") ? getenv("BIOEMU2_WEIGHTS") : std::string(getenv("HOME")) + "/ported/bioemu2";
  int S = 5, T = 29, layer = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--S=", 4)) S = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--T=", 4)) T = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--layer=", 8)) layer = atoi(argv[i] + 8);
  }
#ifdef GRAD_F64
  g::init();
  for (int mixer = 0; mixer < 2; ++mixer) {
    const int Cm = 12, C = 16, H = 2, D = 4, Co = 4, Im = 20, Iz = 24;
    S = 4; T = 7;
    g::bioemu2::MsaLayer ml;
    g::bioemu2::PairHalf ph;
    g::Params ps;
    auto f = [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); };
    if (mixer) g::bioemu2::mixerParams(ph, "pm/", 0, C, Iz, f);
    else g::bioemu2::msaLayerParams(ml, "ml/", 0, Cm, C, H, D, Co, Im, Iz, (g::real)S, f);
    unsigned k = 61;
    for (auto* p : ps.all) {
      bool scale = p->name.find("norm") != std::string::npos && p->name.find("/weight") != std::string::npos;
      auto v = gauss(p->n, ++k, scale ? 0.1 : 0.3);
      if (scale) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    const size_t R = (size_t)S * T, P = (size_t)T * T;
    std::vector<double> ones(P, 1.0);
    g::real *m = devReal(gauss(R * Cm, 71, 1)), *z = devReal(gauss(P * C, 72, 1)), *one = devReal(ones);
    g::real *ctm = devReal(gauss(R * Cm, 73, 1)), *ctz = devReal(gauss(P * C, 74, 1));
    g::real *mo, *zo, *dm, *dz;
    GK(cudaMalloc(&mo, R * Cm * sizeof(g::real))); GK(cudaMalloc(&zo, P * C * sizeof(g::real)));
    GK(cudaMalloc(&dm, R * Cm * sizeof(g::real))); GK(cudaMalloc(&dz, P * C * sizeof(g::real)));
    ps.zeroGrad();
    g::ARENA.reset();
    g::GradCheck gc;
    if (mixer) {
      auto sv = g::bioemu2::pairHalfForward(ph, z, one, zo, T);
      g::bioemu2::pairHalfBackward(ph, sv, one, ctz, dz);
    } else {
      auto sv = g::bioemu2::msaLayerForward(ml, m, z, one, mo, zo, S, T);
      g::bioemu2::msaLayerBackward(ml, sv, one, ctm, ctz, dm, dz, S, T);
      gc.add("in/msa", m, dm, R * Cm);
      gc.output(mo, ctm, R * Cm);
    }
    gc.add("in/pair", z, dz, P * C);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(zo, ctz, P * C);
    printf("BioEmu2 %s (small, random):\n", mixer ? "PairMixer layer" : "MSA layer");
    failed += gc.run([&] {
      g::ARENA.reset();
      if (mixer) g::bioemu2::pairHalfForward(ph, z, one, zo, T);
      else g::bioemu2::msaLayerForward(ml, m, z, one, mo, zo, S, T);
    });
  }
#else
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  FAST = false;
  M.loadBundle(bundle, "w");
  using namespace be2;
  const std::string ml = SE + "msa_module/layers/", pm = SE + "pairmixer_module/layers/";
  const int Cm = 128, C = 256, Hh = 8, Dd = 32;
  const int Co = dimP(ml + "outer_product_mean/proj_ab/weight", 1) / 2;
  const int Im = dimP(ml + "msa_transition/fc1/weight", 1), Iz = dimP(ml + "z_transition/fc1/weight", 1);
  const int Izm = dimP(pm + "transition_z/fc1/weight", 1);
  D_.T = T; D_.S = S; D_.nValidRows = S;
  // a slice of a stacked tensor "<name>@<l>" into the parameter: (out, in) as it is, but the outer product's output
  // transposed into [Co Co, C]
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len);
    size_t at = nm.rfind('@');
    std::string base = nm.substr(0, at); int l = atoi(nm.c_str() + at + 1);
    if (lenP(base) / dimP(base, 0) != len) { fprintf(stderr, "%s: %zu a layer, %zu expected\n", base.c_str(), lenP(base) / dimP(base, 0), len); exit(1); }
    if (base.find("outer_product_mean/proj_o/weight") != std::string::npos)
      transposeIntoK<<<g::blocks(len), 256, 0, STREAM>>>(PL(base, l), p.w, C, Co * Co);
    else toRealK<<<g::blocks(len), 256, 0, STREAM>>>(PL(base, l), p.w, len);
  };
  const size_t R = (size_t)S * T, P = (size_t)T * T;
  std::vector<double> ones(P, 1.0);
  auto mH = gauss(R * Cm, 1, 1), zH = gauss(P * C, 2, 1);
  g::real *m = devReal(mH), *z = devReal(zH), *one = devReal(ones), *mo, *zo;
  GK(cudaMalloc(&mo, R * Cm * 4)); GK(cudaMalloc(&zo, P * C * 4));
  {
    g::bioemu2::MsaLayer l;
    g::bioemu2::msaLayerParams(l, ml, layer, Cm, C, Hh, Dd, Co, Im, Iz, (g::real)S, load);
    g::ARENA.reset();
    g::bioemu2::msaLayerForward(l, m, z, one, mo, zo, S, T);
    float *rm, *rz;
    GK(cudaMalloc(&rm, R * Cm * 4)); GK(cudaMalloc(&rz, P * C * 4));
    GK(cudaMemcpy(rm, m, R * Cm * 4, cudaMemcpyDeviceToDevice)); GK(cudaMemcpy(rz, z, P * C * 4, cudaMemcpyDeviceToDevice));
    pairWeightedAveraging(rm, rz, layer);
    transition(rm, rm, R, Cm, ml + "msa_transition/", layer);
    outerProductMean(rm, rz, layer);
    trimul(rz, ml + "tri_mul_out/", layer, true);
    trimul(rz, ml + "tri_mul_in/", layer, false);
    pairTransitionF32(rz, ml + "z_transition/", layer);
    CK(cudaStreamSynchronize(STREAM));
    double em = g::relRms(g::host(mo, R * Cm), g::host(rm, R * Cm)), ez = g::relRms(g::host(zo, P * C), g::host(rz, P * C));
    printf("BioEmu2 MSA layer %d (%s), S %d T %d: against cuda/bioemu2's float32 path: msa %.3e, pair %.3e\n", layer,
           bundle.c_str(), S, T, em, ez);
    if (!(em < 1e-4 && ez < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  }
  {
    g::bioemu2::PairHalf ph;
    g::bioemu2::mixerParams(ph, pm, layer, C, Izm, load);
    g::ARENA.reset();
    g::bioemu2::pairHalfForward(ph, z, one, zo, T);
    float* rz; GK(cudaMalloc(&rz, P * C * 4));
    GK(cudaMemcpy(rz, z, P * C * 4, cudaMemcpyDeviceToDevice));
    trimul(rz, pm + "tri_mul_out/", layer, true);
    trimul(rz, pm + "tri_mul_in/", layer, false);
    pairTransitionF32(rz, pm + "transition_z/", layer);
    CK(cudaStreamSynchronize(STREAM));
    double ez = g::relRms(g::host(zo, P * C), g::host(rz, P * C));
    printf("BioEmu2 PairMixer layer %d: against cuda/bioemu2's float32 path: pair %.3e\n", layer, ez);
    if (!(ez < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
