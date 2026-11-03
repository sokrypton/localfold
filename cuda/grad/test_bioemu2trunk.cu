// BioEmu2's trunk for training (cuda/grad/bioemu2trunk.cuh), two ways:
//   1. float32: s_inputs, s_init, z_init, each pass's single and pair and the distogram against cuda/bioemu2's
//      inputEmbedder, trunkInit, trunkPass and distogram (their float32 arm), on the float32 blob and a real input
//      (6MRR, export_input.mjs: 68 tokens, 576 atoms);
//   2. -DGRAD_F64: the backward against central differences end to end (atoms -> s_inputs -> two passes -> the distogram
//      and the single), the MSA module one layer and the PairMixer two, a contact class on some pairs.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-bioemu2trunk cuda/grad/test_bioemu2trunk.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-bioemu2trunk [--weights=~/ported/bioemu2] [--input=cuda/grad/data/be2-6mrr] [--passes=2]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-bioemu2trunk-f64 cuda/grad/test_bioemu2trunk.cu -lcublas -lcublasLt -ldl -lcurand
#include "../bioemu2/src/fasttrunk.cuh"
#include "../bioemu2/src/vjp.cuh"
#include "bioemu2trunk.cuh"
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
static std::vector<double> hostD(const float* d, size_t n) {
  std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return std::vector<double>(h.begin(), h.end());
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
  std::string weights = std::string(getenv("HOME")) + "/ported/bioemu2", input = "cuda/grad/data/be2-6mrr";
  int passes = 2;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
    else if (!strncmp(argv[i], "--passes=", 9)) passes = atoi(argv[i] + 9);
  }
  M.loadBundle(weights, "w");
  for (int s = 0; s < (int)M.segs.size(); ++s) M.upload(s);
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  CB(cublasSetMathMode(H, CUBLAS_PEDANTIC_MATH));
  g::STREAM = STREAM;
  g::init();
  int seg = (int)M.segs.size();
  M.load(input);
  for (int s = seg; s < (int)M.segs.size(); ++s) M.upload(s);
  const int T = (int)M.meta("meta/tokens"), N = (int)M.meta("meta/atoms"), S = (int)M.meta("meta/msa_rows"), Si = 455, Cs = 384, C = 256;
  be2::D_ = {T, N, S, (int)M.meta("meta/n_valid_rows")};
  const size_t P = (size_t)T * T;
  g::be2::Weights W;
  W.has = [](const std::string& n) { return be2::hasP(n); };
  W.dim = [](const std::string& n, int k) { return be2::dimP(n, k); };
  W.fetch = [](const std::string& n, int l) {
    if (!be2::hasP(n)) { fprintf(stderr, "no %s\n", n.c_str()); exit(1); }
    size_t len = l < 0 ? be2::lenP(n) : be2::lenP(n) / (size_t)be2::dimP(n, 0);
    return download(l < 0 ? be2::P(n) : be2::PL(n, l), len);
  };
  // the batch
  g::be2::Batch b;
  b.T = T; b.N = N; b.Si = Si;
  b.atomPad = Idev("atom_pad_mask"); b.tokAtAtom = Idev("token_idx_at_atom"); b.centre = Idev("token_center_atom_index");
  b.tokPad = Idev("token_pad_mask"); b.element = Idev("element"); b.names = Idev("name_encoding"); b.uid = Idev("ref_space_uid");
  b.asym = Idev("asym_id"); b.ent = Idev("entity_id"); b.res = Idev("residue_index"); b.sym = Idev("sym_id");
  b.ref = devReal(Fdev("template_coords"), (size_t)N * 3); b.charge = devReal(Fdev("charge"), N);
  {
    const int* tok = M.i("token_idx_at_atom"); const int* apad = M.i("atom_pad_mask");
    std::vector<int> row(N), at(N);
    for (int i = 0; i < N; ++i) { row[i] = apad[i] ? tok[i] : T; at[i] = apad[i] ? tok[i] : -1; }
    b.atomRowCond = devI(row); b.atomTok = devI(at);
  }
  be2::buildTokenStarts(T, N);
  std::vector<int> startH(T + 1); GK(cudaMemcpy(startH.data(), be2::TOK_START, (T + 1) * 4, cudaMemcpyDeviceToHost));
  g::be2::InputBatch ib{devI(startH), Idev("res_type"), Idev("pocket_feature"), devReal(Fdev("profile"), (size_t)T * 33),
                        devReal(Fdev("deletion_mean"), T)};
  // the contact classes: the input's, or (the float64 check) a class on some pairs so the conditioning's encoder is read
  std::vector<int> cls(M.i("contact_class"), M.i("contact_class") + P);
  std::vector<double> thr(M.f("contact_threshold"), M.f("contact_threshold") + P);
#ifdef GRAD_F64
  for (size_t ij = 0; ij < P; ij += 37) { cls[ij] = 1 + (int)(ij % 4); thr[ij] = 6 + (double)(ij % 11); }
#endif
  g::be2::TrunkBatch tb{Idev("method_feature"), b.asym, b.ent, b.res, b.sym, devI(cls), devReal(Fdev("token_bonds"), P), devReal(thr),
                        Idev("msa"), Idev("has_deletion"), Idev("msa_paired"), devReal(Fdev("deletion_value"), (size_t)S * T), T, S,
                        (g::real)be2::D_.nValidRows};
  // the trunk's parameters
  const std::string ml = g::be2::SE + "msa_module/layers/", pm = g::be2::SE + "pairmixer_module/layers/";
  int msaLayers = be2::dimP(ml + "z_transition/fc1/weight", 0), mixers = be2::dimP(pm + "transition_z/fc1/weight", 0);
#ifdef GRAD_F64
  msaLayers = 1; mixers = 2;
#endif
  const int Co = be2::dimP(ml + "outer_product_mean/proj_ab/weight", 1) / 2;
  const int Im = be2::dimP(ml + "msa_transition/fc1/weight", 1), Iz = be2::dimP(ml + "z_transition/fc1/weight", 1);
  const int Izm = be2::dimP(pm + "transition_z/fc1/weight", 1);
  g::Params tps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    size_t at = nm.rfind('@');
    std::string base = at == std::string::npos ? nm : nm.substr(0, at); int l = at == std::string::npos ? -1 : atoi(nm.c_str() + at + 1);
    std::vector<float> h = W.fetch(base, l);
    if (h.size() != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), h.size(), len); exit(1); }
    std::vector<g::real> r(len);
    if (base.find("outer_product_mean/proj_o/weight") != std::string::npos) {     // (out, in) [C, Co Co] -> [Co Co, C]
      for (int i = 0; i < C; ++i) for (int k = 0; k < Co * Co; ++k) r[(size_t)k * C + i] = h[(size_t)i * Co * Co + k];
    } else std::copy(h.begin(), h.end(), r.begin());
    g::allocParam(p, nm, len); tps.add(p);
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  };
  g::be2::Trunk m;
  m.B = be2::dimP("distogram_module/distogram/weight", 0);
  g::be2::forEachParam(m, msaLayers, mixers, 8, 32, Co, Im, Iz, Izm, (g::real)be2::D_.nValidRows,
                       be2::dimP(g::be2::SE + "method_conditioning/embedding/weight", 0), W.dim, load);
  printf("BioEmu2 trunk (%s, %s): %d tokens, %d atoms, %d MSA rows, %d passes of %d MSA layers and %d PairMixer layers, %d bins\n",
         weights.c_str(), input.c_str(), T, N, S, passes, msaLayers, mixers, m.B);
  g::real *sInit, *zInit, *s, *z, *lg, *ones, *F;
  GK(cudaMalloc(&sInit, (size_t)T * Cs * sizeof(g::real))); GK(cudaMalloc(&zInit, P * C * sizeof(g::real)));
  GK(cudaMalloc(&s, (size_t)T * Cs * sizeof(g::real))); GK(cudaMalloc(&z, P * C * sizeof(g::real)));
  GK(cudaMalloc(&lg, P * m.B * sizeof(g::real)));
  ones = devReal(std::vector<double>(P, 1));
  GK(cudaMalloc(&F, (size_t)S * T * 36 * sizeof(g::real)));
  g::be2::msaFeatK<<<g::blocks((size_t)S * T), 256, 0, STREAM>>>(tb.msa, tb.hasDel, tb.delVal, tb.paired, F, (size_t)S * T);
  struct Run { g::tp::Tape t; g::tp::Node* sIn; g::be2::TrunkRun r; std::vector<g::be2::PassSave> ps; };
  auto forward = [&](Run& run, bool keep) {
    g::be2::Graph gr{W, b, run.t};
    run.sIn = g::be2::inputEmbedder(gr, ib, Si);
    g::be2::initForward(m, tb, run.sIn->v, sInit, zInit);
    run.r = g::be2::TrunkRun{sInit, zInit, ones, F, {}, {}, passes};
    GK(cudaMemsetAsync(s, 0, (size_t)T * Cs * sizeof(g::real), STREAM)); GK(cudaMemsetAsync(z, 0, P * C * sizeof(g::real), STREAM));
    run.ps.clear();
    for (int k = 0; k < passes; ++k) {
      if (keep) { run.r.sPrev.push_back(g::cloneR(s, (size_t)T * Cs)); run.r.zPrev.push_back(g::cloneR(z, P * C)); }
      run.ps.push_back(g::be2::passForward(m, tb, run.r, run.sIn->v, s, z, false));
    }
    g::be2::distogramForward(m, z, T, lg);
  };
#ifndef GRAD_F64
  g::ARENA.reset();
  Run run; forward(run, false);
  // the port's
  float *pIn = dalloc((size_t)T * Si), *pSi = dalloc((size_t)T * Cs), *pZi = dalloc(P * C), *pS = dalloc((size_t)T * Cs), *pZ = dalloc(P * C);
  be2::inputEmbedder(pIn, Si, nullptr);
  be2::trunkInit(pIn, Si, pSi, pZi, nullptr);
  CK(cudaMemsetAsync(pS, 0, (size_t)T * Cs * 4, STREAM)); CK(cudaMemsetAsync(pZ, 0, P * C * 4, STREAM));
  for (int k = 0; k < passes; ++k) be2::trunkPass(pIn, Si, pSi, pZi, pS, pZ);
  float* pL = dalloc(P * m.B); be2::distogram(pZ, pL);
  CK(cudaStreamSynchronize(STREAM));
  auto hv = [](const g::real* p, size_t len) { auto v = g::host(p, len); return std::vector<double>(v.begin(), v.end()); };
  auto cmp = [&](const char* what, const std::vector<double>& a, const std::vector<double>& r, double bar) {
    double e = g::relRms(a, r);
    printf("  %-40s %.3e%s\n", what, e, e < bar ? "" : "  <-- ABOVE BAR");
    if (!(e < bar)) ++failed;
  };
  cmp("s_inputs", hv(run.sIn->v, (size_t)T * Si), hostD(pIn, (size_t)T * Si), 1e-5);
  cmp("s_init", hv(sInit, (size_t)T * Cs), hostD(pSi, (size_t)T * Cs), 1e-5);
  cmp("z_init", hv(zInit, P * C), hostD(pZi, P * C), 1e-5);
  cmp("the single after the passes", hv(s, (size_t)T * Cs), hostD(pS, (size_t)T * Cs), 1e-4);
  cmp("the pair after the passes", hv(z, P * C), hostD(pZ, P * C), 1e-4);
  cmp("the distogram", hv(lg, P * m.B), hostD(pL, P * m.B), 1e-4);
#else
  g::real *ctL = devReal(gauss(P * m.B, 7, 1)), *ctS = devReal(gauss((size_t)T * Cs, 8, 1)), *oS;
  GK(cudaMalloc(&oS, (size_t)T * Cs * sizeof(g::real)));
  auto fwd = [&] {
    g::ARENA.reset();
    Run run; forward(run, false);
    GK(cudaMemcpyAsync(oS, s, (size_t)T * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice, STREAM));
    GK(cudaStreamSynchronize(STREAM));
  };
  tps.zeroGrad(); W.all.zeroGrad();
  g::ARENA.reset();
  Run run; forward(run, true);
  g::real *ds, *dz, *dsInit, *dzInit, *dsIn;
  for (auto [p, len] : std::vector<std::pair<g::real**, size_t>>{{&ds, (size_t)T * Cs}, {&dz, P * C}, {&dsInit, (size_t)T * Cs},
                                                                 {&dzInit, P * C}, {&dsIn, (size_t)T * Si}}) {
    GK(cudaMalloc(p, len * sizeof(g::real))); GK(cudaMemset(*p, 0, len * sizeof(g::real)));
  }
  GK(cudaMemcpy(ds, ctS, (size_t)T * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice));
  g::be2::distogramBackward(m, z, T, ctL, dz);
  for (int k = passes; k-- > 0;) {
    // the pass replayed from its incoming s and z (the saved layer inputs are its own)
    GK(cudaMemcpy(s, run.r.sPrev[k], (size_t)T * Cs * sizeof(g::real), cudaMemcpyDeviceToDevice));
    GK(cudaMemcpy(z, run.r.zPrev[k], P * C * sizeof(g::real), cudaMemcpyDeviceToDevice));
    g::be2::PassSave p = g::be2::passForward(m, tb, run.r, run.sIn->v, s, z, true);
    g::be2::passBackward(m, tb, run.r, p, run.sIn->v, ds, dz, dsInit, dzInit, dsIn);
  }
  g::be2::initBackward(m, tb, run.sIn->v, dsInit, dzInit, dsIn);
  run.t.backward(run.sIn, dsIn);
  GK(cudaStreamSynchronize(STREAM));
  g::GradCheck gc;
  gc.verbose = getenv("GRAD_VERBOSE") != nullptr;
  gc.rel = getenv("GRAD_REL") ? atof(getenv("GRAD_REL")) : 1e-8;       // (the atom pair MLP's ReLUs)
  for (auto* p : tps.all) gc.add(*p);
  for (auto* p : W.all.all) gc.add(*p);
  gc.output(lg, ctL, P * m.B); gc.output(oS, ctS, (size_t)T * Cs);
  failed += gc.run(fwd);
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
