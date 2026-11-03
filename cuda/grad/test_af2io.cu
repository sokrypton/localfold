// AlphaFold 2's embedder and heads for training (cuda/grad/af2io.cuh), two ways:
//   1. float32: against af3-any-model's JAX itself - cuda/af2/oracle.py's dump of data-test59 (pass 0, the recycled
//      inputs zero): the embedded msa and pair; the heads from the reference's own evoformer and structure-module
//      outputs (single, pLDDT, PAE, distogram, masked MSA logits); the extra MSA's embedding against the port's kernel;
//   2. -DGRAD_F64: the backward against central differences, small random tables, a recycled position that lands in
//      the distogram's bins, both relative encodings.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-af2io cuda/grad/test_af2io.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af2io [--weights=cuda/af2/weights-model_1_ptm] [--input=cuda/af2/data-test59]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-af2io-f64 cuda/grad/test_af2io.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af2/src/evoformer.cuh"
#include "af2io.cuh"
#include "gradcheck.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static g::real* devReal(const float* dev, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost));
  return devReal(std::vector<double>(h.begin(), h.end()));
}
static std::vector<float> hostF(const float* d, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h;
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
static int* devInt(const std::vector<int>& h) {
  int* d; GK(cudaMalloc(&d, h.size() * 4)); GK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return d;
}

int main(int argc, char** argv) {
  int failed = 0;
  std::string weights = "cuda/af2/weights-model_1_ptm", input = "cuda/af2/data-test59";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
#ifndef GRAD_F64
  M.load(weights);
  M.load(input);
  M.load(input + "/oracle");
  const int L = (int)M.len("aatype"), N = (int)(M.len("f0/msa_mask") / L), E = (int)(M.len("f0/extra_msa_mask") / L);
  g::af2::Embedder e; e.multimer = M.flag("meta/multimer");
  g::af2::Heads h;
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.len("w/" + nm) != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), M.len("w/" + nm), len); exit(1); }
    GK(cudaMemcpy(p.w, P(nm), len * 4, cudaMemcpyDeviceToDevice));
  };
  g::af2::forEachParam(e, load);
  g::af2::forEachParam(h, load);
  const size_t pairs = (size_t)L * L;
  g::af2::EmbedInputs in{};
  in.L = L; in.N = N; in.E = E;
  in.msaFeat = devReal(W("f0/msa_feat"), M.len("f0/msa_feat")); in.aatype = Idev("aatype");
  in.ri = Idev("residue_index"); in.asym = Idev("asym_id"); in.entity = Idev("entity_id"); in.sym = Idev("sym_id");
  in.extraMsa = Idev("f0/extra_msa"); in.hasDel = devReal(W("f0/extra_has_deletion"), (size_t)E * L);
  in.delVal = devReal(W("f0/extra_deletion_value"), (size_t)E * L);
  in.prevRow = devReal(std::vector<double>((size_t)L * 256, 0)); in.prevPair = devReal(std::vector<double>(pairs * 128, 0));
  in.prevPos = devReal(std::vector<double>((size_t)L * 111, 0));
  g::real *msa, *pair, *extra;
  GK(cudaMalloc(&msa, (size_t)N * L * 256 * 4)); GK(cudaMalloc(&pair, pairs * 128 * 4)); GK(cudaMalloc(&extra, (size_t)E * L * 64 * 4));
  g::ARENA.reset();
  g::af2::embedForward(e, in, msa, pair, extra);
  float* refExtra; GK(cudaMalloc(&refExtra, (size_t)E * L * 64 * 4));
  extraEmbedK<<<blocks((size_t)E * L * 64), 256, 0, STREAM>>>(Idev("f0/extra_msa"), W("f0/extra_has_deletion"), W("f0/extra_deletion_value"),
      P("evoformer/extra_msa_activations/weights"), P("evoformer/extra_msa_activations/bias"), refExtra, (size_t)E * L);
  // the heads, on the reference's own representations
  g::real *fMsa = devReal(W("o/full/msa"), (size_t)N * L * 256), *fPair = devReal(W("o/full/pair"), pairs * 128),
          *fAct = devReal(W("o/full/structure_act"), (size_t)L * 384);
  g::af2::HeadOut o{};
  g::af2::headsForward(h, fMsa, fPair, fAct, L, N, o);
  CK(cudaStreamSynchronize(STREAM));
  auto cmp = [&](const char* what, const g::real* mine, const float* ref, size_t n) {
    auto a = g::host(mine, n); auto b = hostF(ref, n);
    double er = g::relRms(a, b), rms = 0, diff = 0;
    for (size_t i = 0; i < n; ++i) { rms += (double)b[i] * b[i]; diff = std::max(diff, std::fabs((double)a[i] - b[i])); }
    printf("  %-34s %.3e  (rms %.3e over %zu, max |diff| %.2e)\n", what, er, std::sqrt(rms / n), n, diff);
    if (!(rms > 0)) { printf("  <-- THE REFERENCE IS ZERO\n"); ++failed; }
    if (!(er < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  };
  printf("AF2 embedder and heads (%s), %d residues, %d MSA rows, %d extra; against af3-any-model's JAX (%s/oracle)\n",
         weights.c_str(), L, N, E, input.c_str());
  cmp("embedded msa", msa, W("o/embed/msa"), (size_t)N * L * 256);
  cmp("embedded pair", pair, W("o/embed/pair"), pairs * 128);
  cmp("extra msa (the port's kernel)", extra, refExtra, (size_t)E * L * 64);
  cmp("single (msa row 0)", o.single, W("o/full/single"), (size_t)L * 384);
  cmp("pLDDT logits", o.plddt, W("o/full/plddt_logits"), (size_t)L * 50);
  cmp("PAE logits", o.pae, W("o/full/pae_logits"), pairs * 64);
  cmp("distogram logits", o.distogram, W("o/full/distogram_logits"), pairs * 64);
  cmp("masked MSA logits", o.maskedMsa, W("o/full/masked_msa_logits"), (size_t)N * L * 22);
#else
  for (bool multimer : {false, true}) {
    const int L = 6, N = 3, E = 4;
    g::af2::Embedder e; e.C = 8; e.Cz = 6; e.Ce = 4; e.F = 5; e.multimer = multimer;
    g::af2::Heads h; h.Cm = 8; h.Cz = 6; h.Cs = 10; h.msaClasses = 5; h.dgBins = 7; h.paeBins = 4; h.plddtBins = 5; h.plddtHidden = 6;
    g::Params ps;
    auto alloc = [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); };
    g::af2::forEachParam(e, alloc);
    g::af2::forEachParam(h, alloc);
    unsigned k = 3;
    for (auto* p : ps.all) {
      bool scale = p->name.find("/scale") != std::string::npos;
      auto v = gauss(p->n, ++k, scale ? 0.1 : 0.3);
      if (scale) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    const size_t pairs = (size_t)L * L;
    g::af2::EmbedInputs in{};
    in.L = L; in.N = N; in.E = E;
    in.msaFeat = devReal(gauss((size_t)N * L * e.F, 41, 1));
    in.aatype = devInt({0, 7, 3, 19, 7, 12});
    in.ri = devInt({0, 1, 2, 10, 11, 12}); in.asym = devInt({1, 1, 1, 2, 2, 2});
    in.entity = devInt({1, 1, 1, 1, 1, 1}); in.sym = devInt({1, 1, 1, 2, 2, 2});
    in.extraMsa = devInt({0, 3, 22, 5, 1, 7, 2, 2, 9, 21, 4, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    in.hasDel = devReal(gauss((size_t)E * L, 42, 1)); in.delVal = devReal(gauss((size_t)E * L, 43, 1));
    in.prevRow = devReal(gauss((size_t)L * e.C, 44, 1)); in.prevPair = devReal(gauss(pairs * e.Cz, 45, 1));
    in.prevPos = devReal(gauss((size_t)L * 111, 46, 6));    // pseudo-beta distances across the distogram's 3.25-20.75 A
    g::real *msa, *pair, *extra, *act;
    GK(cudaMalloc(&msa, (size_t)N * L * e.C * sizeof(g::real))); GK(cudaMalloc(&pair, pairs * e.Cz * sizeof(g::real)));
    GK(cudaMalloc(&extra, (size_t)E * L * e.Ce * sizeof(g::real)));
    act = devReal(gauss((size_t)L * h.Cs, 47, 1));
    g::real *ctMsa = devReal(gauss((size_t)N * L * e.C, 51, 1)), *ctPair = devReal(gauss(pairs * e.Cz, 52, 1)),
            *ctExtra = devReal(gauss((size_t)E * L * e.Ce, 53, 1));
    // the heads read the embedded msa and pair (as though the trunk were the identity) and a random act
    g::real *ctS = devReal(gauss((size_t)L * h.Cs, 61, 1)), *ctPl = devReal(gauss((size_t)L * h.plddtBins, 62, 1)),
            *ctPae = devReal(gauss(pairs * h.paeBins, 63, 1)), *ctDg = devReal(gauss(pairs * h.dgBins, 64, 1)),
            *ctMm = devReal(gauss((size_t)N * L * h.msaClasses, 65, 1)), *ctRes = devReal(gauss((size_t)L * 37, 66, 1));
    g::real *oS, *oPl, *oPae, *oDg, *oMm, *oRes;
    for (auto pp : {std::make_pair(&oS, (size_t)L * h.Cs), std::make_pair(&oPl, (size_t)L * h.plddtBins), std::make_pair(&oPae, pairs * h.paeBins),
                    std::make_pair(&oDg, pairs * h.dgBins), std::make_pair(&oMm, (size_t)N * L * h.msaClasses), std::make_pair(&oRes, (size_t)L * 37)})
      GK(cudaMalloc(pp.first, pp.second * sizeof(g::real)));
    auto fwd = [&] {
      g::ARENA.reset();
      g::af2::embedForward(e, in, msa, pair, extra);
      g::af2::HeadOut o{};
      g::af2::headsForward(h, msa, pair, act, L, N, o);
      auto cp = [&](g::real* d, const g::real* s, size_t n) { GK(cudaMemcpyAsync(d, s, n * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM)); };
      cp(oS, o.single, (size_t)L * h.Cs); cp(oPl, o.plddt, (size_t)L * h.plddtBins); cp(oPae, o.pae, pairs * h.paeBins);
      cp(oDg, o.distogram, pairs * h.dgBins); cp(oMm, o.maskedMsa, (size_t)N * L * h.msaClasses); cp(oRes, o.resolved, (size_t)L * 37);
    };
    ps.zeroGrad();
    g::ARENA.reset();
    auto es = g::af2::embedForward(e, in, msa, pair, extra);
    g::af2::HeadOut o{};
    auto hs = g::af2::headsForward(h, msa, pair, act, L, N, o);
    g::real *dmsa = g::cloneR(ctMsa, (size_t)N * L * e.C), *dpair = g::cloneR(ctPair, pairs * e.Cz), *dact = g::ARENA.zeros((size_t)L * h.Cs);
    g::af2::HeadOut d{nullptr, ctPl, ctPae, ctDg, ctMm, ctRes};
    g::af2::headsBackward(h, hs, msa, pair, act, L, N, d, ctS, dmsa, dpair, dact);
    g::af2::embedBackward(e, in, es, dmsa, dpair, ctExtra);
    g::GradCheck gc;
    gc.add("in/act", act, dact, (size_t)L * h.Cs);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(msa, ctMsa, (size_t)N * L * e.C); gc.output(pair, ctPair, pairs * e.Cz); gc.output(extra, ctExtra, (size_t)E * L * e.Ce);
    gc.output(oS, ctS, (size_t)L * h.Cs); gc.output(oPl, ctPl, (size_t)L * h.plddtBins); gc.output(oPae, ctPae, pairs * h.paeBins);
    gc.output(oDg, ctDg, pairs * h.dgBins); gc.output(oMm, ctMm, (size_t)N * L * h.msaClasses); gc.output(oRes, ctRes, (size_t)L * 37);
    printf("AF2 embedder and heads (small, random; %s relative encoding):\n", multimer ? "multimer" : "monomer");
    failed += gc.run(fwd);
  }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
