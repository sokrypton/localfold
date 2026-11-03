// The AF3 lineage's training layers (cuda/grad/af3.cuh), two ways:
//   1. float32: a pairformer block's forward against cuda/af3's own pairformerBlockAt<float> - the port's reference
//      arm, held to af3-any-model - on a real bundle read through the port's weight walk, any family;
//   2. -DGRAD_F64: the backward against central differences of the same forward, on small random blocks under
//      every dialect switch the block reads (swapTransposedBias, divideByLength, the grid attention's biases, an
//      extra pair-logit bias).
//
//   nvcc -O1 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/test-af3 cuda/grad/test_af3.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af3 [--bundle=model-af3-full-f32] [--family=af3] [--block=0] [--n=37]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-af3-f64 cuda/grad/test_af3.cu -lcublas -lcurand
#ifndef GRAD_F64
#include "../af3/src/trunk.cuh"
#include "../featurise/af3_weights.h"
#endif
#include "af3.cuh"
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
// residues [0, n - pad) real; the pair mask their outer product
static void masks(int n, int pad, std::vector<double>& seq, std::vector<double>& pairM, std::vector<double>& pairT) {
  seq.assign(n, 1); for (int i = n - pad; i < n; ++i) seq[i] = 0;
  pairM.resize((size_t)n * n); pairT.resize((size_t)n * n);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { pairM[(size_t)i * n + j] = seq[i] * seq[j]; pairT[(size_t)j * n + i] = pairM[(size_t)i * n + j]; }
  // and one masked entry inside the real block, so a key mask that is not a residue mask is exercised
  if (n > 4) { pairM[1 * n + 2] = 0; pairT[2 * n + 1] = 0; }
}

#ifdef GRAD_F64
static int checkBlock(const char* what, g::af3::Dialect d, bool gridBiases, bool extra, bool single, unsigned seed,
                      bool hasGrid = true, bool parallel = false) {
  const int n = 13, pad = 2, Cz = 16, Cs = 24, Hp = 2, Dp = 8, Hs = 2, Ds = 12;
  g::af3::Pairformer b;
  g::Params ps;
  g::af3::forEachParam(b, "blk", Cz, Cs, Hp, Dp, Hs, Ds, 4, 4, gridBiases, single,
                       [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); },
                       hasGrid, parallel);
  unsigned k = seed;
  for (auto* p : ps.all) {
    bool scale = p->name.find("Scale") != std::string::npos;
    auto v = gauss(p->n, ++k, scale ? 0.1 : 0.3);
    if (scale) for (auto& x : v) x += 1;
    std::vector<g::real> r(v.begin(), v.end());
    GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
  }
  size_t nn = (size_t)n * n;
  std::vector<double> sm, pm, pt;
  masks(n, pad, sm, pm, pt);
  g::real *pair = devReal(gauss(nn * Cz, 101 + seed, 1)), *sing = devReal(gauss((size_t)n * Cs, 102 + seed, 1));
  g::real *dSeq = devReal(sm), *dPm = devReal(pm), *dPt = devReal(pt);
  g::real* ex = extra ? devReal(gauss(nn, 103 + seed, 0.5)) : nullptr;
  // the loss reads real residues only: a padded query's row is masked in EVERY key, so its logits are q.k - 1e9 and
  // even float64 rounds q.k there to 1e-7 - a padded output is defined by nothing, and a difference of it is noise
  auto ctpH = gauss(nn * Cz, 104 + seed, 1), ctsH = gauss((size_t)n * Cs, 105 + seed, 1);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) if (sm[i] == 0 || sm[j] == 0)
    for (int c = 0; c < Cz; ++c) ctpH[((size_t)i * n + j) * Cz + c] = 0;
  for (int i = n - pad; i < n; ++i) for (int c = 0; c < Cs; ++c) ctsH[(size_t)i * Cs + c] = 0;
  g::real *ctp = devReal(ctpH), *cts = devReal(ctsH);
  g::real *po, *so, *dp, *dsg, *dex = nullptr;
  GK(cudaMalloc(&po, nn * Cz * sizeof(g::real))); GK(cudaMalloc(&so, (size_t)n * Cs * sizeof(g::real)));
  GK(cudaMalloc(&dp, nn * Cz * sizeof(g::real))); GK(cudaMalloc(&dsg, (size_t)n * Cs * sizeof(g::real)));
  if (extra) { GK(cudaMalloc(&dex, nn * sizeof(g::real))); GK(cudaMemset(dex, 0, nn * sizeof(g::real))); }
  ps.zeroGrad();
  g::ARENA.reset();
  auto s = g::af3::pairformerForward(b, d, pair, sing, dPm, dPt, dSeq, ex, po, so, n);
  g::af3::pairformerBackward(b, d, s, dPm, dPt, dSeq, ctp, cts, dp, dsg, dex);
  g::GradCheck gc;
  gc.add("in/pair", pair, dp, nn * Cz);
  if (single) gc.add("in/single", sing, dsg, (size_t)n * Cs);
  if (extra) gc.add("in/extra", ex, dex, nn);
  for (auto* p : ps.all) gc.add(*p);
  gc.output(po, ctp, nn * Cz);
  if (single) gc.output(so, cts, (size_t)n * Cs);
  printf("%s:\n", what);
  return gc.run([&] { g::ARENA.reset(); g::af3::pairformerForward(b, d, pair, sing, dPm, dPt, dSeq, ex, po, so, n); });
}
#endif

int main(int argc, char** argv) {
  int failed = 0;
#ifdef GRAD_F64
  g::init();
  failed += checkBlock("AlphaFold 3 (no switches)", {}, false, false, true, 1);
  failed += checkBlock("swapTransposedBias (openfold3, opendde, boltz2, protenix2)", {true, false}, false, false, true, 2);
  failed += checkBlock("RoseTTAFold3 (divideByLength, the grid attention's biases)", {false, true}, true, false, true, 3);
  failed += checkBlock("an extra pair-logit bias (OpenDDE's refiner)", {true, false}, false, true, true, 4);
  failed += checkBlock("no single track (the template stack)", {}, false, false, false, 5);
  failed += checkBlock("no triangle attention (Mol-1's PairMixer)", {}, false, false, true, 6, false);
  failed += checkBlock("parallel (chai-1)", {}, false, false, true, 7, true, true);
#else
  std::string bundle = "model-af3-full-f32", family = "af3";
  int blk = 0, n = 37;
  std::string prefix = "trunk.pairformerBlocks.";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--family=", 9)) family = argv[i] + 9;
    else if (!strncmp(argv[i], "--block=", 8)) blk = atoi(argv[i] + 8);
    else if (!strncmp(argv[i], "--n=", 4)) n = atoi(argv[i] + 4);
    else if (!strncmp(argv[i], "--prefix=", 9)) prefix = argv[i] + 9;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines(family, S); });
  const std::string pre = prefix + std::to_string(blk);
  const int Cz = (int)M.meta(pre + ".pairChannels"), Cs = (int)M.meta(pre + ".singleChannels");
  const bool hasGrid = M.has(pre + ".pairAttention1.heads"), parallel = M.flag("trunk.dialect.parallelPairformer");
  const int Hp = hasGrid ? (int)M.meta(pre + ".pairAttention1.heads") : 0, Dp = hasGrid ? (int)M.meta(pre + ".pairAttention1.dimension") : 0;
  const int Hs = (int)M.meta(pre + ".singleAttention.heads"), Ds = (int)M.meta(pre + ".singleAttention.dimension");
  const int pf = (int)(M.at(pre + ".pairTransition.transition1").length / (2 * (size_t)Cz * Cz));
  const int sf = (int)(M.at(pre + ".singleTransition.transition1").length / (2 * (size_t)Cs * Cs));
  const bool gridBiases = M.has(pre + ".pairAttention1.gatingQueryBias");
  g::af3::Dialect d{M.flag("trunk.dialect.swapTransposedBias"), M.flag("trunk.dialect.triangleMulDivideByLength")};
  g::af3::Pairformer b;
  g::Params ps;
  g::af3::forEachParam(b, pre, Cz, Cs, Hp, Dp, Hs, Ds, pf, sf, gridBiases, true,
                       [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
  }, hasGrid, parallel);
  size_t nn = (size_t)n * n;
  std::vector<double> sm, pm, pt;
  masks(n, 3, sm, pm, pt);
  auto pairH = gauss(nn * Cz, 11, 1), singleH = gauss((size_t)n * Cs, 12, 1);
  g::real *pair = devReal(pairH), *sing = devReal(singleH), *dSeq = devReal(sm), *dPm = devReal(pm), *dPt = devReal(pt);
  g::real *po, *so;
  GK(cudaMalloc(&po, nn * Cz * 4)); GK(cudaMalloc(&so, (size_t)n * Cs * 4));
  g::af3::pairformerForward(b, d, pair, sing, dPm, dPt, dSeq, nullptr, po, so, n);
  // the port, on copies (it updates in place)
  float *rp, *rs;
  GK(cudaMalloc(&rp, nn * Cz * 4)); GK(cudaMalloc(&rs, (size_t)n * Cs * 4));
  GK(cudaMemcpy(rp, pair, nn * Cz * 4, cudaMemcpyDeviceToDevice)); GK(cudaMemcpy(rs, sing, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice));
  pairformerBlockAt<float>(rp, rs, dPm, dSeq, n, Cz, Cs, pre, d.swapTransposedBias, d.divideByLength);
  CK(cudaStreamSynchronize(STREAM));
  // compare over the real residues only (a padded row's attention is masked in every key)
  auto ourP = g::host(po, nn * Cz), refP = g::host(rp, nn * Cz), ourS = g::host(so, (size_t)n * Cs), refS = g::host(rs, (size_t)n * Cs);
  std::vector<float> a, r, a2, r2;
  for (int i = 0; i < n - 3; ++i) {
    for (int j = 0; j < n - 3; ++j) for (int c = 0; c < Cz; ++c) { size_t q = ((size_t)i * n + j) * Cz + c; a.push_back(ourP[q]); r.push_back(refP[q]); }
    for (int c = 0; c < Cs; ++c) { a2.push_back(ourS[(size_t)i * Cs + c]); r2.push_back(refS[(size_t)i * Cs + c]); }
  }
  double ep = g::relRms(a, r), es = g::relRms(a2, r2);
  printf("%s (%s) %s, n %d: Cz %d Cs %d, grid %dx%d, single %dx%d, transitions x%d/x%d%s%s%s\n", family.c_str(),
         bundle.c_str(), pre.c_str(), n, Cz, Cs, Hp, Dp, Hs, Ds, pf, sf, d.swapTransposedBias ? ", swapTransposedBias" : "",
         d.divideByLength ? ", divideByLength" : "", gridBiases ? ", grid biases" : "");
  if (!hasGrid) printf("  (no triangle attention)\n");
  if (parallel) printf("  (parallel)\n");
  printf("  against cuda/af3's pairformerBlockAt<float>: pair %.3e, single %.3e\n", ep, es);
  if (!(ep < 1e-4 && es < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
