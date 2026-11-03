// The AF3 lineage's atom transformer for training (cuda/grad/af3diff.cuh's AtomStack), two ways:
//   1. float32: a whole atom stack (its pair logits and every block) against cuda/af3's atomPairLogits +
//      prepareAtomBlock + crossAttentionBlock on a family's bundle, with a real featurised input's windows
//      (af3-featurise --no-weights: queriesToKeys and the atom mask), random activations and conditioning;
//   2. -DGRAD_F64: the backward against central differences on a small random stack under every switch.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/test-atom cuda/grad/test_atom.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-atom --input=cuda/grad/data/af3-in-af3 [--bundle=model-af3-full-f32] [--family=af3] [--stack=diffusion.encoder]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-atom-f64 cuda/grad/test_atom.cu -lcublas -lcurand
#ifndef GRAD_F64
#include "../af3/src/trunk.cuh"
#include "../af3/src/atom.cuh"
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
  struct Case { const char* what; bool keyMasked, chained, noRes, perBlock, chai, up, maskAct; bool kq = false, noTr2 = false; };
  for (Case cs : {Case{"AlphaFold 3", false, false, false, false, false, false, false},
                  Case{"key-masked, chained (OpenDDE, protenix2), per-block pair norms", true, true, false, true, false, false, false},
                  Case{"rf3's noResidual", false, false, true, false, false, false, false},
                  Case{"Boltz-2's up-gate, the act masked per block", false, false, false, false, false, true, true},
                  Case{"chai-1's atom stack (raw AdaLN, no Transition2)", false, false, false, false, true, false, false, false, true},
                  Case{"rf3's q/k LayerNorm", true, false, true, true, false, false, false, true}}) {
    const int subsets = 3, Q = 4, K = 6, C = 8, Cp = 4, H = 2, D = 4, nb = 2;
    const size_t Rq = (size_t)subsets * Q, Rk = (size_t)subsets * K;
    g::af3::AtomStack st;
    st.C = C; st.Cp = Cp; st.H = H; st.D = D;
    g::Params ps;
    g::af3::forEachParam(st, "enc", nb, cs.perBlock, cs.chai, cs.up, [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); }, cs.kq, cs.noTr2);
    for (auto& b : st.blocks) { b.keyMasked = cs.keyMasked; b.chained = cs.chained; b.noResidual = cs.noRes; b.maskActPerBlock = cs.maskAct; }
    unsigned k = 111;
    for (auto* p : ps.all) {
      bool scale = p->name.find("LayerNormScale") != std::string::npos;
      auto v = gauss(p->n, ++k, scale ? 0.1 : 0.3);
      if (scale) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    // a window layout: subset s's keys are queries s*Q - 1 .. s*Q + K - 2 (clipped), with the last query of all padded
    // (a padded query only where the mask is key-masked: under AF3's PRODUCT form a padded query meets a padded key at
    // +1e9 - (0 - 1)(0 - 1) - so its logits sit at 1e9, where even float64's last place is 1e-7, and its output is a key
    // of the next block's real queries: that is the model, and it has no derivative at this resolution)
    std::vector<int> idx(Rk); std::vector<double> km(Rk), qm(Rq, 1);
    if (cs.keyMasked) qm[Rq - 1] = 0;
    for (int s = 0; s < subsets; ++s) for (int kk = 0; kk < K; ++kk) {
      int qi = s * Q - 1 + kk; bool ok = qi >= 0 && qi < (int)Rq;
      idx[(size_t)s * K + kk] = ok ? qi : 0; km[(size_t)s * K + kk] = ok ? 1 : 0;
    }
    int* dIdx; GK(cudaMalloc(&dIdx, Rk * 4)); GK(cudaMemcpy(dIdx, idx.data(), Rk * 4, cudaMemcpyHostToDevice));
    g::real *gm = devReal(km), *qmd = devReal(qm);
    std::vector<double> kmEff(Rk);
    for (size_t r = 0; r < Rk; ++r) kmEff[r] = km[r] ? qm[idx[r]] : 0;
    g::real* kmd = devReal(kmEff);
    g::real *act = devReal(gauss(Rq * C, 121, 1)), *qc = devReal(gauss(Rq * C, 122, 1)), *pair = devReal(gauss(Rq * K * Cp, 123, 1));
    g::real* kc; GK(cudaMalloc(&kc, Rk * C * sizeof(g::real)));
    g::real* mb; GK(cudaMalloc(&mb, (size_t)subsets * Q * K * sizeof(g::real)));
    g::af3::atomMaskBiasK<<<g::blocks((size_t)subsets * Q * K), 256>>>(qmd, kmd, mb, subsets, Q, K, cs.keyMasked);
    g::af3::AtomCtx c{{dIdx, gm, (int)Rk}, qmd, kmd, qc, kc, mb, subsets, Q, K};
    auto ctH = gauss(Rq * C, 124, 1);
    for (int ch = 0; ch < C; ++ch) ctH[(Rq - 1) * C + ch] = 0;       // (the padded query's output is not read)
    g::real* ct = devReal(ctH);
    g::real *out, *dact, *dqc, *dkc, *dpair;
    GK(cudaMalloc(&out, Rq * C * sizeof(g::real))); GK(cudaMalloc(&dact, Rq * C * sizeof(g::real)));
    GK(cudaMalloc(&dqc, Rq * C * sizeof(g::real))); GK(cudaMalloc(&dkc, Rk * C * sizeof(g::real)));
    GK(cudaMalloc(&dpair, Rq * K * Cp * sizeof(g::real)));
    auto fwd = [&] {
      g::ARENA.reset();
      g::af3::gather(c.toKeys, qc, kc, C);
      g::af3::StackRun r;
      g::af3::stackForward(st, c, pair, act, out, r);
    };
    ps.zeroGrad();
    GK(cudaMemset(dqc, 0, Rq * C * sizeof(g::real))); GK(cudaMemset(dkc, 0, Rk * C * sizeof(g::real)));
    GK(cudaMemset(dpair, 0, Rq * K * Cp * sizeof(g::real)));
    g::ARENA.reset();
    g::af3::gather(c.toKeys, qc, kc, C);
    g::af3::StackRun r;
    g::af3::stackForward(st, c, pair, act, out, r);
    g::af3::stackBackward(st, c, pair, r, ct, dact, dqc, dkc, dpair);
    g::af3::scatter(c.toKeys, dkc, dqc, C);              // kCond is the gathered qCond
    g::GradCheck gc;
    gc.add("in/act", act, dact, Rq * C);
    gc.add("in/qCond", qc, dqc, Rq * C);
    gc.add("in/pair", pair, dpair, Rq * K * Cp);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(out, ct, Rq * C);
    printf("atom stack, %s:\n", cs.what);
    failed += gc.run(fwd);
  }
#else
  std::string bundle = "model-af3-full-f32", family = "af3", input = "cuda/grad/data/af3-in-af3", E = "diffusion.encoder";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--family=", 9)) family = argv[i] + 9;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
    else if (!strncmp(argv[i], "--stack=", 8)) E = argv[i] + 8;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines(family, S); });
  M.load(input);
  setAdaMode(M.flag("trunk.dialect.chaiAtomStack"));
  AtomShape sh = atomShape();
  const int C = (int)M.meta(E + ".channels"), Cp = (int)M.meta(E + ".pairChannels"), Hh = (int)M.meta(E + ".heads"),
            Dd = (int)M.meta(E + ".dimension");
  int nb = 0; while (M.has(E + ".blocks." + std::to_string(nb) + ".qProjection")) ++nb;
  const bool perBlock = M.flag(E + ".pairNormPerBlock"), chai = M.flag("trunk.dialect.chaiAtomStack");
  const bool up = M.has(E + ".blocks.0.ffwAToB"), kq = M.has(E + ".blocks.0.queryLayerNormScale"),
             noTr2 = !M.has(E + ".blocks.0.Transition2");
  g::af3::AtomStack st;
  st.C = C; st.Cp = Cp; st.H = Hh; st.D = Dd;
  g::Params ps;
  g::af3::forEachParam(st, E, nb, perBlock, chai, up, [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
  }, kq, noTr2);
  const std::string B0 = E + ".blocks.0";
  for (auto& b : st.blocks) {
    b.keyMasked = M.flag(B0 + ".keyMaskedAtomAttention"); b.chained = M.flag(B0 + ".chainedAtomLayerNorm");
    b.noResidual = M.flag(B0 + ".diffusionNoResidual"); b.maskActPerBlock = M.flag(B0 + ".maskAtomActPerBlock");
  }
  const size_t Rq = (size_t)sh.subsets * sh.queries, Rk = (size_t)sh.subsets * sh.keys, rows = Rq * sh.keys;
  ::Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2k = gatherOf("batch.queriesToKeys");
  float *qm, *km; GK(cudaMalloc(&qm, Rq * 4)); GK(cudaMalloc(&km, Rk * 4));
  convert(t2q, Fdev("batch.predDenseAtomMask"), qm, 1);
  convert(q2k, qm, km, 1);
  auto actH = gauss(Rq * C, 1, 1), qcH = gauss(Rq * C, 2, 1), pairH = gauss(rows * Cp, 3, 1);
  g::real *act = devReal(actH), *qc = devReal(qcH), *pair = devReal(pairH), *kc, *mb, *out;
  GK(cudaMalloc(&kc, Rk * C * 4)); GK(cudaMalloc(&mb, rows * 4)); GK(cudaMalloc(&out, Rq * C * 4));
  g::af3::AtomCtx c{{q2k.idx, q2k.mask, q2k.count}, qm, km, qc, kc, mb, sh.subsets, sh.queries, sh.keys};
  g::af3::gather(c.toKeys, qc, kc, C);
  g::af3::atomMaskBiasK<<<g::blocks(rows), 256, 0, STREAM>>>(qm, km, mb, sh.subsets, sh.queries, sh.keys, st.blocks[0].keyMasked);
  g::af3::StackRun r;
  g::ARENA.reset();
  g::af3::stackForward(st, c, pair, act, out, r);
  // the port
  float* ref; GK(cudaMalloc(&ref, Rq * C * 4));
  GK(cudaMemcpy(ref, act, Rq * C * 4, cudaMemcpyDeviceToDevice));
  std::vector<float*> pls = atomPairLogits(E, pair, rows, Cp, nb, Hh, sh);
  AtomStep stp{ q2k, qm, km, st.blocks[0].keyMasked, st.blocks[0].noResidual };
  for (int b = 0; b < nb; ++b) {
    const std::string B = E + ".blocks." + std::to_string(b);
    AtomBlockCache bc = prepareAtomBlock(B, qc, Rq, C, pls[b]);
    if (st.blocks[0].maskActPerBlock) { /* the port masks inside the stack's loop; one block alone: mask here */
      scaleRows<float>(ref, qm, Rq, C);
    }
    crossAttentionBlock(ref, stp, bc, sh, C, Hh, Dd, B);
  }
  CK(cudaStreamSynchronize(STREAM));
  auto o = g::host(out, Rq * C), rf = g::host(ref, Rq * C), qmh = g::host(qm, Rq);
  std::vector<float> a, b;
  for (size_t i = 0; i < Rq; ++i) if (qmh[i] != 0) for (int ch = 0; ch < C; ++ch) { a.push_back(o[i * C + ch]); b.push_back(rf[i * C + ch]); }
  double e = g::relRms(a, b);
  printf("%s (%s) %s: %d blocks, C %d, pair %d, %dx%d heads, %d subsets of %d x %d%s%s%s%s%s\n", family.c_str(), bundle.c_str(),
         E.c_str(), nb, C, Cp, Hh, Dd, sh.subsets, sh.queries, sh.keys, st.blocks[0].keyMasked ? ", key-masked" : "",
         st.blocks[0].chained ? ", chained" : "", st.blocks[0].noResidual ? ", noResidual" : "", perBlock ? ", per-block pair norms" : "",
         up ? ", up-gate" : "");
  printf("  against cuda/af3's crossAttentionBlock: %.3e over the real atoms\n", e);
  if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
