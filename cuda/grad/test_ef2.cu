// ESMFold2's pair trunk for training: its block is the AF3 lineage's pairformer without triangle attention and
// without a single track (cuda/grad/af3.cuh's Pairformer, hasGrid and hasSingle off, the bundle's '/' names).
//   float32: a block's forward against cuda/ef2's own trunkBlock (its float32 path) on the ESMFold2 bundle, for each
//            stack that runs that block - the folding trunk, the coda, the language-model and MSA encoders;
//   the backward is the pairformer's, held to differences under that switch by test_af3.cu's float64 build.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-ef2 cuda/grad/test_ef2.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-ef2 [--bundle=model-esmfold2-f32] [--n=37]
#include "../ef2/src/shim.cuh"
#include "../ef2/src/atoms.cuh"
#include "../ef2/src/trunk.cuh"
#include "af3.cuh"

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
  std::string bundle = "model-esmfold2-f32";
  int n = 37;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--n=", 4)) n = atoi(argv[i] + 4);
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "f");
  int failed = 0;
  for (std::string prefix : {"blocks", "coda/blocks", "lmEncoder/blocks", "msaEncoder/blocks"}) {
    const std::string pre = "f/" + prefix + "/0";
    if (!M.has(pre + "/pairTransition/transition1")) { printf("%s: not in this bundle\n", prefix.c_str()); continue; }
    const int Cz = (int)M.at(pre + "/triangleMultiplicationOutgoing/leftNormInputScale").length;
    const int pf = (int)(M.at(pre + "/pairTransition/transition1").length / (2 * (size_t)Cz * Cz));
    g::af3::Pairformer b;
    g::Params ps;
    g::af3::forEachParam(b, pre, Cz, 0, 0, 0, 0, 0, pf, 0, false, false, [&](const std::string& nm, g::Param& p, size_t len) {
      g::allocParam(p, nm, len); ps.add(p);
      if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
      GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
    }, false, false, "/");
    size_t nn = (size_t)n * n;
    const int pad = 3;
    std::vector<double> pm(nn, 1), pt(nn, 1);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) if (i >= n - pad || j >= n - pad) pm[(size_t)i * n + j] = pt[(size_t)j * n + i] = 0;
    auto pairH = gauss(nn * Cz, 5, 1);
    g::real *pair = devReal(pairH), *dPm = devReal(pm), *dPt = devReal(pt), *po;
    GK(cudaMalloc(&po, nn * Cz * 4));
    g::ARENA.reset();
    g::af3::pairformerForward(b, {}, pair, nullptr, dPm, dPt, nullptr, nullptr, po, nullptr, n);
    float* ref; GK(cudaMalloc(&ref, nn * Cz * 4));
    GK(cudaMemcpy(ref, pair, nn * Cz * 4, cudaMemcpyDeviceToDevice));
    trunkBlock(ref, dPm, n, Cz, prefix, 0);
    CK(cudaStreamSynchronize(STREAM));
    auto ours = g::host(po, nn * Cz), theirs = g::host(ref, nn * Cz);
    std::vector<float> a, r;
    for (int i = 0; i < n - pad; ++i) for (int j = 0; j < n - pad; ++j) for (int c = 0; c < Cz; ++c) {
      size_t q = ((size_t)i * n + j) * Cz + c; a.push_back(ours[q]); r.push_back(theirs[q]);
    }
    double e = g::relRms(a, r);
    printf("ESMFold2 %s block 0: Cz %d, transition x%d: against cuda/ef2's trunkBlock %.3e\n", prefix.c_str(), Cz, pf, e);
    if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  }
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
