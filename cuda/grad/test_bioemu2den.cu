// BioEmu2's denoiser for training (cuda/grad/bioemu2den.cuh on cuda/grad/tape.cuh), two ways:
//   1. float32: the static context and one call D(x, sigma) against cuda/bioemu2's staticContext + denoise() (its float32
//      arm), and the gradient into x against the port's own vjp tape (denoiseVjp: held to af3-any-model's jax.grad) - on
//      the float32 blob and a real input's atoms (export_input.mjs), random trunk outputs and noisy atoms;
//   2. -DGRAD_F64: the backward against central differences on the same denoiser and input - every weight, the static
//      context's included, and the trunk's three inputs and x.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-bioemu2den cuda/grad/test_bioemu2den.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-bioemu2den [--weights=~/ported/bioemu2] [--input=cuda/grad/data/be2-6mrr] [--sigma=5]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-bioemu2den-f64 cuda/grad/test_bioemu2den.cu -lcublas -lcublasLt -ldl -lcurand
#include "../bioemu2/src/fasttrunk.cuh"
#include "../bioemu2/src/vjp.cuh"
#include "bioemu2den.cuh"
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
static float* devF(const std::vector<double>& h) {
  std::vector<float> r(h.begin(), h.end());
  float* d; GK(cudaMalloc(&d, r.size() * 4)); GK(cudaMemcpy(d, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
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
  std::string weights = std::string(getenv("HOME")) + "/ported/bioemu2", input = "cuda/grad/data/be2-6mrr";
  double sigma = 5;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
    else if (!strncmp(argv[i], "--sigma=", 8)) sigma = atof(argv[i] + 8);
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
  if (M.flag("meta/is_dynamics_conditioned")) { fprintf(stderr, "a dynamics-conditioned input is not here\n"); return 1; }
  const int T = (int)M.meta("meta/tokens"), N = (int)M.meta("meta/atoms"), Si = 455, Cs = 384, C = 256;
  be2::D_ = {T, N, (int)M.meta("meta/msa_rows"), (int)M.meta("meta/n_valid_rows")};
  const size_t P = (size_t)T * T;
  g::be2::Weights W;
  W.has = [](const std::string& n) { return be2::hasP(n); };
  W.dim = [](const std::string& n, int k) { return be2::dimP(n, k); };
  W.fetch = [](const std::string& n, int l) {
    if (!be2::hasP(n)) { fprintf(stderr, "no %s\n", n.c_str()); exit(1); }
    size_t len = l < 0 ? be2::lenP(n) : be2::lenP(n) / (size_t)be2::dimP(n, 0);
    return download(l < 0 ? be2::P(n) : be2::PL(n, l), len);
  };
  const int layers = be2::dimP(be2::SM + "layers/atom_block/transformer_layer/attention/attention_layer/pair_bias_attn/proj_z/weight", 0);
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
  auto sH = gauss((size_t)T * Cs, 1, 1), zH = gauss(P * C, 2, 1), iH = gauss((size_t)T * Si, 3, 1), xH = gauss((size_t)N * 3, 4, sigma);
  g::real *sT = devReal(sH), *zT = devReal(zH), *sI = devReal(iH), *x = devReal(xH), *out;
  GK(cudaMalloc(&out, (size_t)N * 3 * sizeof(g::real)));
  printf("BioEmu2 denoiser (%s, %s), %d tokens, %d atoms, sigma %g, %d layers\n", weights.c_str(), input.c_str(), T, N, sigma, layers);
  // one forward on a fresh tape: the leaves, the context, the call; returns the denoised node
  struct Run { g::tp::Tape t; g::tp::Node *s, *z, *i, *x, *den; g::be2::Ctx c; };
  auto forward = [&](Run& r) {
    g::ARENA.reset();
    r.t = g::tp::Tape();
    r.s = r.t.leaf(sT, T, Cs); r.z = r.t.leaf(zT, P, C); r.i = r.t.leaf(sI, T, Si); r.x = r.t.leaf(x, N, 3);
    g::be2::Graph gr{W, b, r.t};
    r.c = g::be2::staticContext(gr, r.s, r.z, r.i, layers);
    r.den = g::be2::denoise(gr, r.c, r.x, sigma, layers);
    GK(cudaMemcpyAsync(out, r.den->v, (size_t)N * 3 * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
  };
  auto mk = M.i("atom_pad_mask");
  auto ctH = gauss((size_t)N * 3, 9, 1);
  for (int i = 0; i < N; ++i) if (!mk[i]) for (int k = 0; k < 3; ++k) ctH[i * 3 + k] = 0;
  if (true) {
  Run r; forward(r);
  r.t.backward(r.den, devReal(ctH));
  GK(cudaStreamSynchronize(g::STREAM));
  // the port: its float32 context and call; its vjp tape for d/dx
  be2::GEMM_HALF = false;
  be2::Denoiser d = be2vjp::exactContext(devF(sH), devF(zH), devF(iH), Si);
  float* den = dalloc((size_t)N * 3);
  be2::denoise(d, devF(xH), (float)sigma, den);
  CK(cudaStreamSynchronize(STREAM));
  auto vj = be2vjp::denoiseVjp(d, devF(xH), sigma, devF(ctH));
  auto mine = g::host(out, (size_t)N * 3), gx = g::host(r.x->g, (size_t)N * 3);
  auto port = download(den, (size_t)N * 3);
  const double sd = 16, cSkip = sd * sd / (sigma * sigma + sd * sd);
  std::vector<double> a, p, ga, gp;
  for (int i = 0; i < N; ++i) if (mk[i]) for (int k = 0; k < 3; ++k) {
    a.push_back(mine[i * 3 + k] - cSkip * xH[i * 3 + k]); p.push_back(port[i * 3 + k] - cSkip * xH[i * 3 + k]);
    ga.push_back(gx[i * 3 + k] - cSkip * ctH[i * 3 + k]); gp.push_back(vj.gx[i * 3 + k] - cSkip * ctH[i * 3 + k]);
  }
  if (getenv("GRAD_LAYERS")) {     // both sides cut to the same depth: where a difference starts
    for (int Lc : {0, 1, 2, 4, 16}) {
      be2::Denoiser dc = d; dc.layers = Lc;
      be2::denoise(dc, devF(xH), (float)sigma, den);
      CK(cudaStreamSynchronize(STREAM));
      auto pc = download(den, (size_t)N * 3);
      Run rc; g::ARENA.reset(); rc.t = g::tp::Tape();
      rc.s = rc.t.leaf(sT, T, Cs); rc.z = rc.t.leaf(zT, P, C); rc.i = rc.t.leaf(sI, T, Si); rc.x = rc.t.leaf(x, N, 3);
      g::be2::Graph gr{W, b, rc.t};
      rc.c = g::be2::staticContext(gr, rc.s, rc.z, rc.i, layers);
      rc.den = g::be2::denoise(gr, rc.c, rc.x, sigma, Lc);
      GK(cudaStreamSynchronize(g::STREAM));
      auto mc = g::host(rc.den->v, (size_t)N * 3);
      std::vector<double> u, v;
      for (int i = 0; i < N; ++i) if (mk[i]) for (int k = 0; k < 3; ++k) { u.push_back(mc[i * 3 + k] - cSkip * xH[i * 3 + k]); v.push_back(pc[i * 3 + k] - cSkip * xH[i * 3 + k]); }
      printf("  %2d layers: %.3e\n", Lc, g::relRms(u, v));
    }
  }
  if (getenv("GRAD_SENS")) {        // the port against itself, x moved by float32's last place
    auto nz = gauss(xH.size(), 77, 1e-7); auto x2 = xH; for (size_t k = 0; k < x2.size(); ++k) x2[k] *= 1 + nz[k];
    be2::denoise(d, devF(x2), (float)sigma, den);
    CK(cudaStreamSynchronize(STREAM));
    auto p2 = download(den, (size_t)N * 3);
    auto vj2 = be2vjp::denoiseVjp(d, devF(x2), sigma, devF(ctH));
    std::vector<double> u, v, gu, gv;
    for (int i = 0; i < N; ++i) if (mk[i]) for (int k = 0; k < 3; ++k) {
      u.push_back(p2[i * 3 + k] - cSkip * x2[i * 3 + k]); v.push_back(port[i * 3 + k] - cSkip * xH[i * 3 + k]);
      gu.push_back(vj2.gx[i * 3 + k] - cSkip * ctH[i * 3 + k]); gv.push_back(vj.gx[i * 3 + k] - cSkip * ctH[i * 3 + k]);
    }
    printf("  (the port against itself, x moved 1e-7: its update %.3e, the network's d/dx %.3e)\n", g::relRms(u, v), g::relRms(gu, gv));
  }
  {   // the static context, tensor by tensor
    auto cmpD = [&](const char* nm, const g::tp::Node* mine, const float* port, size_t n) {
      printf("  context %-6s %.3e\n", nm, g::relRms(g::host(mine->v, n), download(port, n)));
    };
    cmpD("z", r.c.z, d.z, P * C); cmpD("q0", r.c.q0, d.q0, (size_t)N * 128); cmpD("p", r.c.p, d.p, (size_t)N * 128 * 16);
    cmpD("sc0", r.c.sc0, d.sc0, (size_t)T * 768); cmpD("a0", r.c.a0, d.a0, (size_t)T * 768);
    cmpD("wbias0", r.c.wbias[0], d.wbias, (size_t)N * 128 * 4);
    if (d.tbias) cmpD("tbias0", r.c.tbias[0], d.tbias, (size_t)16 * P);
  }
  {   // the vjp tape's own forward against denoise(), and against this one
    std::vector<double> vd, pd, md;
    for (int i = 0; i < N; ++i) if (mk[i]) for (int k = 0; k < 3; ++k) {
      vd.push_back(vj.den[i * 3 + k] - cSkip * xH[i * 3 + k]); pd.push_back(port[i * 3 + k] - cSkip * xH[i * 3 + k]);
      md.push_back(mine[i * 3 + k] - cSkip * xH[i * 3 + k]);
    }
    printf("  (the vjp tape's forward against denoise(): %.3e; against this one: %.3e)\n", g::relRms(vd, pd), g::relRms(md, vd));
    double nn = 0, nw = 0; for (size_t k = 0; k < gp.size(); ++k) { nn += gp[k] * gp[k]; } for (double v : ctH) nw += v * v;
    printf("  (|network part of d/dx| %.3e against |c_skip w| %.3e)\n", std::sqrt(nn), cSkip * std::sqrt(nw));
  }
  if (getenv("GRAD_SAVE")) { FILE* fp = fopen(getenv("GRAD_SAVE"), "wb"); fwrite(a.data(), 8, a.size(), fp); fwrite(ga.data(), 8, ga.size(), fp); fclose(fp); }
  if (getenv("GRAD_AGAINST")) {       // this denoiser's float32 run (GRAD_SAVE) against this one
    std::vector<double> oa(a.size()), og(ga.size()); FILE* fp = fopen(getenv("GRAD_AGAINST"), "rb");
    if (fp && fread(oa.data(), 8, oa.size(), fp) == oa.size() && fread(og.data(), 8, og.size(), fp) == og.size())
      printf("  against the float32 run: its update %.3e, the network's d/dx %.3e\n", g::relRms(oa, a), g::relRms(og, ga));
    if (fp) fclose(fp);
  }
  double e = g::relRms(a, p), eg = g::relRms(ga, gp);
  printf("  against cuda/bioemu2's denoise() (float32): its update %.3e over the real atoms\n", e);
  printf("  d/dx against the port's vjp tape (jax.grad's): the network's part %.3e\n", eg);
#ifndef GRAD_F64
  if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  // d/dx's bar is float32's own floor on it: the network's part of d/dx nearly cancels and float32 resolves it to ~1e-2
  // on this random input - this denoiser's float32 is 1.0e-2 from its own float64 (GRAD_SAVE / GRAD_AGAINST), and that
  // float64 is 1.45e-2 from the port's float32 vjp (the port's two float32 paths agree closer, 8e-6 on D, sharing
  // cuBLAS's rounding). The float64 gradcheck is the backward's gate; this arm catches a d/dx that is wrong outright.
  if (!(eg < 5e-2)) { printf("  <-- ABOVE 5e-2\n"); ++failed; }
#endif
  }
#ifdef GRAD_F64
  W.all.zeroGrad();
  Run r; forward(r);
  r.t.backward(r.den, devReal(ctH));
  g::real *ds = g::cloneR(r.s->g, (size_t)T * Cs), *dz = g::cloneR(r.z->g, P * C), *di = g::cloneR(r.i->g, (size_t)T * Si),
          *dx = g::cloneR(r.x->g, (size_t)N * 3);
  // (kept out of the arena the gradcheck's forwards reset)
  auto keep = [](g::real* d, size_t n) { g::real* k; GK(cudaMalloc(&k, n * sizeof(g::real))); GK(cudaMemcpy(k, d, n * sizeof(g::real), cudaMemcpyDeviceToDevice)); return k; };
  ds = keep(ds, (size_t)T * Cs); dz = keep(dz, P * C); di = keep(di, (size_t)T * Si); dx = keep(dx, (size_t)N * 3);
  g::GradCheck gc;
  gc.rel = 1e-8;
  gc.add("in/sTrunk", sT, ds, (size_t)T * Cs); gc.add("in/zTrunk", zT, dz, P * C);
  gc.add("in/sInputs", sI, di, (size_t)T * Si); gc.add("in/x", x, dx, (size_t)N * 3);
  for (auto* p : W.all.all) gc.add(*p);
  g::real* ct = devReal(ctH);
  gc.output(out, ct, (size_t)N * 3);
  failed += gc.run([&] { Run rr; forward(rr); });
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
