// AlphaFold 2's template embedders for training (cuda/grad/af2tmpl.cuh), two ways:
//   1. float32: the template term against cuda/af2's templateEmbedding (multimer) or templateEmbeddingMonomer on a real
//      template (data-1brs-t, data-5caj-t), and the template rows against its templateRows;
//   2. -DGRAD_F64: the backward against central differences on the same weights, the template and a second one (the
//      same chain moved by noise, so the monomer's softmax over templates is not trivial), a random pair.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-af2tmpl cuda/grad/test_af2tmpl.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af2tmpl [--weights=cuda/af2/weights-model_1_multimer_v3] [--input=cuda/af2/data-1brs-t]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-af2tmpl-f64 cuda/grad/test_af2tmpl.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af2/src/evoformer.cuh"
#include "../af2/src/templates.cuh"
#include "af2tmpl.cuh"
#include "gradcheck.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static std::vector<double> hostD(const float* d, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return std::vector<double>(h.begin(), h.end());
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
  std::string weights = "cuda/af2/weights-model_1_multimer_v3", input = "cuda/af2/data-1brs-t";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.load(weights);
  M.load(input);
  const bool multimer = M.flag("meta/multimer");
  const int L = (int)M.len("aatype"), T = (int)M.meta("meta/templates", 0);
  if (T < 1) { fprintf(stderr, "%s has no template\n", input.c_str()); return 1; }
  const size_t pr = (size_t)L * L;
  const std::string TE = "evoformer/template_embedding/", S = TE + "single_template_embedding/";
  const std::string stack = multimer ? S + "template_embedding_iteration/" : S + "template_pair_stack/__layer_stack_no_state/";
  const int nb = (int)dimW(stack + "pair_transition/transition1/weights", 0);
  const int Hh = (int)dimW(stack + "triangle_attention_starting_node/feat_2d_weights", 2);
  const int factor = (int)(dimW(stack + "pair_transition/transition1/weights", 2) / 64);
  const int attH = multimer ? 0 : (int)dimW(TE + "attention/query_w", 1), attD = multimer ? 0 : (int)dimW(TE + "attention/query_w", 2);
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    std::string key = nm; int block = -1;
    size_t at = nm.find('@'); if (at != std::string::npos) { key = nm.substr(0, at); block = atoi(nm.c_str() + at + 1); }
    g::allocParam(p, nm, len); ps.add(p);
    const size_t have = M.len("w/" + key) / (block >= 0 ? (size_t)dimW(key, 0) : 1);
    if (have != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), have, len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), P(key, block), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  };
  g::af2::Templates m;
  g::af2::forEachParam(m, multimer, nb, Hh, factor, attH, attD, load);
  g::af2::TemplateRows rw;
  const int RF = multimer ? 34 : 57;
  g::af2::forEachParam(rw, RF, load);
  printf("AF2 %s templates (%s, %s): %d residues, %d template(s), %d blocks (%d heads), transition x%d%s\n",
         multimer ? "multimer" : "monomer", weights.c_str(), input.c_str(), L, T, nb, Hh, factor,
         multimer ? "" : ", the pointwise attention over templates");
  auto features = [&](const int* aat, const g::real* pos, const g::real* msk) {
    g::real* F; GK(cudaMalloc(&F, pr * 88 * sizeof(g::real)));
    g::af2::tmplPairFeatK<<<g::blocks(pr), 256, 0, STREAM>>>(aat, pos, msk, F, L, multimer);
    return (const g::real*)F;
  };
  auto posH = hostD(W("t/positions"), (size_t)T * L * 111), mskH = hostD(W("t/mask"), (size_t)T * L * 37);
  g::real* mask = devReal(std::vector<double>(pr, 1));
#ifndef GRAD_F64
  {
    std::vector<const g::real*> F;
    g::real *pos = devReal(posH), *msk = devReal(mskH);
    for (int t = 0; t < T; ++t) F.push_back(features(Idev("t/aatype") + (size_t)t * L, pos + (size_t)t * L * 111, msk + (size_t)t * L * 37));
    auto pairH = gauss(pr * 128, 1, 1);
    g::real* pair = devReal(pairH); g::real* out = devReal(std::vector<double>(pr * 128, 0));
    g::ARENA.reset();
    g::af2::templatesForward(m, F, pair, mask, L, out);
    std::vector<float> pf(pairH.begin(), pairH.end());
    float* pp = dalloc(pr * 128); CK(cudaMemcpy(pp, pf.data(), pr * 128 * 4, cudaMemcpyHostToDevice));
    float* pm = dalloc(pr); { std::vector<float> one(pr, 1.f); CK(cudaMemcpy(pm, one.data(), pr * 4, cudaMemcpyHostToDevice)); }
    if (multimer) templateEmbedding(pp, pm, L); else templateEmbeddingMonomer(pp, pm, L, T);
    CK(cudaStreamSynchronize(STREAM));
    auto mine = g::host(out, pr * 128); auto port = hostD(pp, pr * 128);
    std::vector<double> a(mine.begin(), mine.end()), r(pr * 128);
    for (size_t i = 0; i < r.size(); ++i) r[i] = port[i] - pairH[i];
    double e = g::relRms(a, r);
    printf("  the template term against cuda/af2's %s %.3e\n", multimer ? "templateEmbedding" : "templateEmbeddingMonomer", e);
    if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
    // the rows: the port's own features (its torsion kernels), its rows
    float* rows = dalloc((size_t)T * L * 256); float* rmask = dalloc((size_t)T * L);
    templateRows(L, T, multimer, rows, rmask);
    CK(cudaStreamSynchronize(STREAM));
    g::real* feat = devReal(hostD(scratch<float>("trow.feat", (size_t)T * L * RF), (size_t)T * L * RF));
    g::real* ro = devReal(std::vector<double>((size_t)T * L * 256, 0));
    g::af2::rowsForward(rw, feat, ro, (size_t)T * L);
    auto mr = g::host(ro, (size_t)T * L * 256);
    double er = g::relRms(std::vector<double>(mr.begin(), mr.end()), hostD(rows, (size_t)T * L * 256));
    printf("  the template rows against cuda/af2's templateRows %.3e\n", er);
    if (!(er < 1e-5)) { printf("  <-- ABOVE 1e-5\n"); ++failed; }
  }
#else
  {
    // the template and a second: the same chain, its atoms moved by 0.7 A
    auto noise = gauss((size_t)L * 111, 9, 0.7);
    std::vector<double> pos2(posH.begin(), posH.begin() + (size_t)L * 111), msk2(mskH.begin(), mskH.begin() + (size_t)L * 37);
    for (size_t i = 0; i < pos2.size(); ++i) pos2[i] += noise[i];
    std::vector<int> aat(L); GK(cudaMemcpy(aat.data(), Idev("t/aatype"), L * 4, cudaMemcpyDeviceToHost));
    int* aatD; GK(cudaMalloc(&aatD, L * 4)); GK(cudaMemcpy(aatD, aat.data(), L * 4, cudaMemcpyHostToDevice));
    g::real *pos = devReal(posH), *msk = devReal(mskH), *p2 = devReal(pos2), *m2 = devReal(msk2);
    std::vector<const g::real*> F{features(aatD, pos, msk), features(aatD, p2, m2)};
    g::real *pair = devReal(gauss(pr * 128, 1, 1)), *out, *ct = devReal(gauss(pr * 128, 2, 1)), *dp;
    GK(cudaMalloc(&out, pr * 128 * sizeof(g::real))); GK(cudaMalloc(&dp, pr * 128 * sizeof(g::real)));
    GK(cudaMemset(dp, 0, pr * 128 * sizeof(g::real)));
    auto fwd = [&] { g::ARENA.reset(); GK(cudaMemsetAsync(out, 0, pr * 128 * sizeof(g::real), STREAM)); g::af2::templatesForward(m, F, pair, mask, L, out); };
    // the rows: random torsion features of the right width
    const size_t R = (size_t)2 * L;
    g::real *feat = devReal(gauss(R * RF, 5, 1)), *ro, *ctr = devReal(gauss(R * 256, 6, 1));
    GK(cudaMalloc(&ro, R * 256 * sizeof(g::real)));
    ps.zeroGrad();
    g::ARENA.reset();
    GK(cudaMemsetAsync(out, 0, pr * 128 * sizeof(g::real), STREAM));
    auto s = g::af2::templatesForward(m, F, pair, mask, L, out);
    g::af2::templatesBackward(m, F, s, pair, mask, ct, dp);
    auto rs = g::af2::rowsForward(rw, feat, ro, R);
    g::af2::rowsBackward(rw, rs, feat, ctr, R);
    g::GradCheck gc;
    gc.verbose = getenv("GRAD_VERBOSE") != nullptr;
    gc.rel = 1e-8;      // (the stack's and the rows' ReLUs)
    gc.add("in/pair", pair, dp, pr * 128);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(out, ct, pr * 128);
    gc.output(ro, ctr, R * 256);
    failed += gc.run([&] { fwd(); g::af2::rowsForward(rw, feat, ro, R); });
  }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
