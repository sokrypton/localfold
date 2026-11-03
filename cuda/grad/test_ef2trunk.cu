// ESMFold2's trunk glue for training (cuda/grad/ef2trunk.cuh), three ways:
//   1. float32, the experimental tier (cuda/ef2/weights, data-6mrr): the shim from the vendor's 37 hidden states
//      against its lm_z and the pair MLP against cuda/ef2's lmPairRows; z_init's seams against the vendor's and the
//      port's zInit; the four recycle passes' in and out against the vendor's; the trunk against cuda/ef2's
//      foldingTrunk, the distogram against both. Then a whole backward, timed;
//   2. float32, --parcae=<bundle> (ESMFold2-Fast's, ESMFold2's): the parcae recycle - lmEncoder, the decay, the
//      readout and the coda - against foldingTrunk under EF2_DETERMINISTIC on a random z_init and language-model
//      pair;
//   3. -DGRAD_F64: the backward against central differences end to end (37 states -> shim -> z_init -> trunk ->
//      distogram), on a small random input with truncated stacks; --parcae=<bundle> the parcae form with its
//      dropout on and a random initial state.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-ef2trunk cuda/grad/test_ef2trunk.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-ef2trunk [--weights=cuda/ef2/weights] [--input=cuda/ef2/data-6mrr] [--parcae=model-esmfold2-fast-f32]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-ef2trunk-f64 cuda/grad/test_ef2trunk.cu -lcublas -lcublasLt -ldl -lcurand
#include <chrono>
#include "../ef2/src/shim.cuh"
#include "../ef2/src/atoms.cuh"
#include "../ef2/src/trunk.cuh"
#include "ef2trunk.cuh"
#include "gradcheck.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
  return d;
}
static g::real* devReal(const float* host, size_t n) { return devReal(std::vector<double>(host, host + n)); }
static float* devF(const std::vector<double>& h) {
  std::vector<float> r(h.begin(), h.end());
  float* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * 4)); GK(cudaMemcpy(d, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
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
static std::vector<double> hostD(const g::real* d, size_t n) { auto h = g::host(d, n); return std::vector<double>(h.begin(), h.end()); }
#ifdef GRAD_F64
static std::vector<double> hostD(const float* d, size_t n) { std::vector<float> h(n); GK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return std::vector<double>(h.begin(), h.end()); }
#endif
static int* devI(const std::vector<int>& h) {
  int* d; GK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * 4)); GK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  return d;
}

static int failed = 0;
static void expect(const char* what, double e, double bar) {
  printf("  %-70s %.3e%s\n", what, e, e < bar ? "" : "  <-- ABOVE BAR");
  if (!(e < bar)) ++failed;
}

// every parameter of a module from the loaded weights, under `prefix`
struct Loader {
  g::Params ps;
  std::string prefix;
  std::function<void(const std::string&, g::Param&, size_t)> fn() {
    return [this](const std::string& nm, g::Param& p, size_t len) {
      std::string key = prefix + nm;
      if (!nm.rfind("lm/", 0)) key = shimKey(nm);
      if (!M.has(key)) { fprintf(stderr, "%s: not in the weights\n", key.c_str()); exit(1); }
      if (M.at(key).length != len) { fprintf(stderr, "%s: %zu in the weights, %zu expected\n", key.c_str(), (size_t)M.at(key).length, len); exit(1); }
      g::allocParam(p, key, len); ps.add(p);
      std::vector<float> h(len); GK(cudaMemcpy(h.data(), W(key), len * 4, cudaMemcpyDeviceToHost));
      std::vector<g::real> r(h.begin(), h.end());
      GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
    };
  }
};

int main(int argc, char** argv) {
  std::string weights = "cuda/ef2/weights", input = "cuda/ef2/data-6mrr", parcae;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
    else if (!strncmp(argv[i], "--parcae=", 9)) parcae = argv[i] + 9;
  }
  setenv("EF2_DETERMINISTIC", "1", 1);     // (the port's recycle: no dropout, a zero initial state)
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  FAST = false; GEMM16 = false; ATOM_BF16 = false;
  if (parcae.empty()) M.load(weights); else M.loadBundle(parcae, "f");
  const int C = (int)M.meta("meta/pairChannels");
  const int pf = (int)(M.at("f/blocks/0/pairTransition/transition1").length / (2 * (size_t)C * C));
  const int nBlocks = (int)M.meta("meta/blocks");
  const int nLm = parcae.empty() ? 0 : (int)M.meta("meta/lmEncoderBlocks");
  const int nCoda = parcae.empty() ? 0 : (int)M.meta("meta/codaBlocks");
  const int loops = parcae.empty() ? 4 : (int)M.meta("meta/loops") + 1;
  Loader L; L.prefix = "f/";

#ifndef GRAD_F64
  if (!parcae.empty()) {
    // ---------------------------------------------------------------- 2. the parcae recycle, against foldingTrunk
    const int n = 37; const size_t P = (size_t)n * n * C;
    printf("ESMFold2 parcae recycle (%s): %d blocks, lmEncoder %d, coda %d, %d passes, %d tokens\n", parcae.c_str(), nBlocks, nLm, nCoda, loops, n);
    g::ef2::Trunk t; g::ef2::forEachParam(t, C, pf, nBlocks, nLm, nCoda, true, L.fn());
    auto ziH = gauss(P, 3, 1), lmH = gauss(P, 4, 1);
    g::real *zi = devReal(ziH), *lm = devReal(lmH), *out; GK(cudaMalloc(&out, P * sizeof(g::real)));
    g::ef2::TrunkRun r;
    g::ARENA.reset();
    g::ef2::trunkForward(t, r, zi, lm, nullptr, n, loops, 0, 0, out);
    float* z = dalloc(P);
    foldingTrunk(n, C, devF(ziH), z, loops, false, devF(lmH), 0);
    CK(cudaStreamSynchronize(STREAM));
    // the port's own float32 sensitivity: its inputs nudged by 1e-7 (random inputs drive the 30 blocks harder than
    // a fold's z_init does)
    std::vector<double> ziN(ziH), lmN(lmH);
    auto nz = gauss(P, 11, 1e-7), nl = gauss(P, 12, 1e-7);
    for (size_t i = 0; i < P; ++i) { ziN[i] += nz[i]; lmN[i] += nl[i]; }
    float* zs = dalloc(P);
    foldingTrunk(n, C, devF(ziN), zs, loops, false, devF(lmN), 0);
    CK(cudaStreamSynchronize(STREAM));
    const double sens = g::relRms(hostD(zs, P), hostD(z, P));
    printf("  (the port against itself, its inputs nudged by 1e-7: %.3e)\n", sens);
    expect("the trunk's pair against cuda/ef2's foldingTrunk (bar: max(1e-4, 4x that))", g::relRms(hostD(out, P), hostD(z, P)),
           std::max(1e-4, 4 * sens));
    printf("%s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
  }
  // ---------------------------------------------------------------- 1. the experimental tier, against the vendor
  M.load(input);
  M.load(input + "/oracle");
  const int T = (int)M.meta("meta/tokens"), Si = (int)M.meta("meta/singleInputs");
  const size_t PT = (size_t)T * T, P = PT * C;
  const int Cm = (int)M.len("c/lm/norm/scale"), S = (int)M.len("c/lm/combine"), B = (int)dimOf("f/distogram/weights", 1);
  printf("ESMFold2's trunk glue (%s, %s): %d tokens, %d states of %d, pair %d, %d blocks, %d passes\n", weights.c_str(),
         input.c_str(), T, S, Cm, C, nBlocks, loops);
  g::ef2::Shim sh; g::ef2::forEachParam(sh, Cm, C, S, L.fn());
  g::ef2::ZInit zm; g::ef2::forEachParam(zm, Si, C, L.fn());
  g::ef2::Trunk tk; g::ef2::forEachParam(tk, C, pf, nBlocks, 0, 0, false, L.fn());
  g::ef2::Distogram dg; g::ef2::forEachParam(dg, C, B, L.fn());
  // the vendor's states, per token (o/lm_hidden is [T, S, Cm], gathered by token_to_row)
  const float* hid = M.f("o/lm_hidden");
  std::vector<g::real*> st(S); std::vector<const g::real*> stc(S);
  for (int k = 0; k < S; ++k) {
    std::vector<double> h((size_t)T * Cm);
    for (int tt = 0; tt < T; ++tt) for (int c = 0; c < Cm; ++c) h[(size_t)tt * Cm + c] = hid[((size_t)tt * S + k) * Cm + c];
    stc[k] = st[k] = devReal(h);
  }
  std::vector<int> ident(T); for (int i = 0; i < T; ++i) ident[i] = i;
  int* t2r = devI(ident);
  g::ARENA.reset();
  g::real* lmZ = g::ARENA.get(P);
  auto ss = g::ef2::shimForward(sh, stc, T, t2r, T, lmZ);
  expect("the shim's lm_z against the vendor's (o/lm_z)", g::relRms(hostD(lmZ, P), std::vector<double>(M.f("o/lm_z"), M.f("o/lm_z") + P)), 1e-4);
  {
    float* pr = dalloc(P); auto sTok = hostD(ss.s, (size_t)T * C);
    float* sF = devF(sTok);
    for (int i0 = 0, bi = lmPairBlock(T, C); i0 < T; i0 += bi) lmPairRows(sF, T, C, i0, std::min(bi, T - i0), pr + (size_t)i0 * T * C);
    CK(cudaStreamSynchronize(STREAM));
    expect("its pair MLP against cuda/ef2's lmPairRows", g::relRms(hostD(lmZ, P), hostD(pr, P)), 1e-5);
  }
  // z_init from the vendor's s_inputs and lm_z
  g::ef2::Tokens tok{Idev("residue_index"), Idev("asym_id"), Idev("sym_id"), Idev("entity_id"), Idev("token_index"),
                     devReal(M.f("token_bonds"), PT), T};
  g::real* sIn = devReal(M.f("o/s_inputs"), (size_t)T * Si);
  g::real* lmO = devReal(M.f("o/lm_z"), P);
  g::real* zi = g::ARENA.get(P);
  g::ef2::zinitForward(zm, tok, sIn, lmO, zi);
  {
    // its seams, the vendor's (z_init_1 [i] + z_init_2 [j] + rel_pos + token_bonds + lm_z)
    std::vector<double> ref(P);
    const float *a = M.f("o/z_init_1"), *b = M.f("o/z_init_2"), *rp = M.f("o/rel_pos"), *bo = M.f("o/token_bonds"), *lo = M.f("o/lm_z");
    for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) for (int c = 0; c < C; ++c) {
      size_t q = ((size_t)i * T + j) * C + c;
      ref[q] = (double)a[i * C + c] + b[j * C + c] + rp[q] + bo[q] + lo[q];
    }
    expect("z_init against the vendor's five terms", g::relRms(hostD(zi, P), ref), 1e-5);
    buildBonds(T);
    float* zp = dalloc(P);
    zInit(T, C, devF(std::vector<double>(M.f("o/s_inputs"), M.f("o/s_inputs") + (size_t)T * Si)), Si, devF(std::vector<double>(lo, lo + P)), zp, false);
    CK(cudaStreamSynchronize(STREAM));
    expect("z_init against cuda/ef2's zInit", g::relRms(hostD(zi, P), hostD(zp, P)), 1e-5);
  }
  // the trunk, from the vendor's z_init
  g::real* ziO = g::ARENA.get(P);
  {
    std::vector<double> ref(P);
    const float *a = M.f("o/z_init_1"), *b = M.f("o/z_init_2"), *rp = M.f("o/rel_pos"), *bo = M.f("o/token_bonds"), *lo = M.f("o/lm_z");
    for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) for (int c = 0; c < C; ++c) {
      size_t q = ((size_t)i * T + j) * C + c;
      ref[q] = (double)a[i * C + c] + b[j * C + c] + rp[q] + bo[q] + lo[q];
    }
    std::vector<g::real> r(ref.begin(), ref.end());
    GK(cudaMemcpy(ziO, r.data(), P * sizeof(g::real), cudaMemcpyHostToDevice));
  }
  g::ef2::TrunkRun tr;
  g::real* zt = g::ARENA.get(P);
  g::ef2::trunkForward(tk, tr, ziO, nullptr, nullptr, T, loops, 0, 0, zt);
  for (int l = 0; l < loops; ++l) {
    std::string in = "o/loop" + std::to_string(l) + "/in", outK = "o/loop" + std::to_string(l) + "/out";
    const g::real* mo = l + 1 < loops ? tr.zPrev[l + 1] : tr.zLast;
    char what[128];
    snprintf(what, sizeof what, "pass %d in against the vendor's", l);
    expect(what, g::relRms(hostD(tr.zIn[l], P), std::vector<double>(M.f(in), M.f(in) + P)), 1e-4);
    snprintf(what, sizeof what, "pass %d out against the vendor's", l);
    expect(what, g::relRms(hostD(mo, P), std::vector<double>(M.f(outK), M.f(outK) + P)), 1e-5);
  }
  {
    float* zp = dalloc(P);
    foldingTrunk(T, C, devF(hostD(ziO, P)), zp, loops, false);
    CK(cudaStreamSynchronize(STREAM));
    expect("the trunk's pair against cuda/ef2's foldingTrunk", g::relRms(hostD(zt, P), hostD(zp, P)), 1e-4);
    g::real* lg = g::ARENA.get(PT * B);
    g::ef2::distogramForward(dg, zt, T, lg);
    float* lp = dalloc(PT * B);
    distogram(zp, T, C, lp);
    CK(cudaStreamSynchronize(STREAM));
    expect("the distogram against cuda/ef2's", g::relRms(hostD(lg, PT * B), hostD(lp, PT * B)), 1e-4);
    expect("the distogram against the vendor's", g::relRms(hostD(lg, PT * B), std::vector<double>(M.f("o/distogram"), M.f("o/distogram") + PT * B)), 1e-5);
  }
  // a whole step, end to end: the states -> the distogram's logits, timed
  {
    g::ARENA.release();
    auto ctH = gauss(PT * B, 7, 1); g::real* ct = devReal(ctH);
    std::vector<g::real*> dst(S);
    for (auto& p : dst) GK(cudaMalloc(&p, (size_t)T * Cm * sizeof(g::real)));
    g::real *dsIn; GK(cudaMalloc(&dsIn, (size_t)T * Si * sizeof(g::real)));
    auto step = [&](bool timed) {
      auto t0 = std::chrono::steady_clock::now();
      g::ARENA.reset();
      g::real* lm = g::ARENA.get(P);
      auto s1 = g::ef2::shimForward(sh, stc, T, t2r, T, lm);
      g::real* z0 = g::ARENA.get(P); g::ef2::zinitForward(zm, tok, sIn, lm, z0);
      g::ef2::TrunkRun r; g::real* z = g::ARENA.get(P);
      g::ef2::trunkForward(tk, r, z0, nullptr, nullptr, T, loops, 0, 0, z);
      g::real* lg = g::ARENA.get(PT * B); g::ef2::distogramForward(dg, z, T, lg);
      GK(cudaStreamSynchronize(STREAM));
      auto t1 = std::chrono::steady_clock::now();
      g::real* dz = g::ARENA.zeros(P); g::ef2::distogramBackward(dg, z, T, ct, dz);
      g::real* dzi = g::ARENA.zeros(P); g::ef2::trunkBackward(tk, r, z0, nullptr, dz, dzi, nullptr, nullptr);
      g::real* dlm = g::ARENA.zeros(P);
      GK(cudaMemsetAsync(dsIn, 0, (size_t)T * Si * sizeof(g::real), STREAM));
      g::ef2::zinitBackward(zm, tok, sIn, dzi, dsIn, dlm);
      g::ef2::shimBackward(sh, s1, dlm, nullptr, dst);
      GK(cudaStreamSynchronize(STREAM));
      auto t2 = std::chrono::steady_clock::now();
      if (timed) printf("  a step at %d tokens, %d passes of %d blocks: forward %.1f ms, backward %.1f ms; arena high %.2f GB\n", T, loops,
                        nBlocks, std::chrono::duration<double, std::milli>(t1 - t0).count(),
                        std::chrono::duration<double, std::milli>(t2 - t1).count(), g::ARENA.high / 1e9);
    };
    step(false); step(true);
    auto d0 = hostD(dst[0], (size_t)T * Cm); double s2 = 0; for (double v : d0) s2 += v * v;
    if (!std::isfinite(s2) || s2 == 0) { printf("  <-- the first state's gradient is %g\n", s2); ++failed; }
  }
#else
  // ---------------------------------------------------------------- 3. the backward against central differences
  const int T = 12, R = T - 1, nb = 2, nl = 1, nc = 1, lp = 2;
  const size_t PT = (size_t)T * T, P = PT * C;
  const bool par = !parcae.empty();
  const int Si = (int)M.meta("meta/singleInputs");
  const int B = (int)dimOf("f/distogram/weights", 1);
  const bool hasShim = M.has(shimKey("lm/combine"));
  printf("ESMFold2's trunk glue backward (%s): %d tokens, %d blocks%s, %d passes\n", par ? "parcae" : "experimental", T, nb,
         par ? ", lmEncoder 1, coda 1" : "", lp);
  g::ef2::Shim sh; const int Cm = hasShim ? (int)M.len(shimKey("lm/norm/scale")) : 0, S = hasShim ? (int)M.len(shimKey("lm/combine")) : 0;
  if (hasShim) g::ef2::forEachParam(sh, Cm, C, S, L.fn());
  g::ef2::ZInit zm; g::ef2::forEachParam(zm, Si, C, L.fn());
  g::ef2::Trunk tk; g::ef2::forEachParam(tk, C, pf, nb, par ? nl : 0, par ? nc : 0, par, L.fn());
  g::ef2::Distogram dg; g::ef2::forEachParam(dg, C, B, L.fn());
  // a token without a tower row (the zero state), rows out of order
  std::vector<int> t2rH(T); for (int i = 0; i < T; ++i) t2rH[i] = i == 5 ? -1 : (i < 5 ? i : R - 1 - (i - 6));
  int* t2r = devI(t2rH);
  std::vector<g::real*> st(S), dst(S); std::vector<const g::real*> stc(S);
  for (int k = 0; k < S; ++k) { stc[k] = st[k] = devReal(gauss((size_t)R * Cm, 100 + k, 1)); GK(cudaMalloc(&dst[k], (size_t)R * Cm * sizeof(g::real))); }
  std::vector<int> ri(T), asym(T), sym(T), ent(T), ti(T);
  for (int i = 0; i < T; ++i) { ri[i] = i < 8 ? i : i - 8; asym[i] = i < 8 ? 0 : 1; sym[i] = asym[i]; ent[i] = 0; ti[i] = i; }
  std::vector<double> bonds(PT, 0); bonds[1 * T + 2] = bonds[2 * T + 1] = 1; bonds[3 * T + 9] = bonds[9 * T + 3] = 1;
  g::ef2::Tokens tok{devI(ri), devI(asym), devI(sym), devI(ent), devI(ti), devReal(bonds), T};
  g::real* sIn = devReal(gauss((size_t)T * Si, 5, 1));
  g::real* z0 = devReal(gauss(P, 6, 0.1));
  g::real *lm, *zi, *z, *lg;
  GK(cudaMalloc(&lm, P * sizeof(g::real))); GK(cudaMalloc(&zi, P * sizeof(g::real)));
  GK(cudaMalloc(&z, P * sizeof(g::real))); GK(cudaMalloc(&lg, PT * B * sizeof(g::real)));
  const g::real p = par ? (g::real)0.25 : 0;
  g::ef2::ShimSave ss; g::ef2::TrunkRun tr;
  auto fwd = [&] {
    g::ARENA.reset();
    ss = g::ef2::shimForward(sh, stc, R, t2r, T, lm);
    g::ef2::zinitForward(zm, tok, sIn, par ? nullptr : lm, zi);
    g::ef2::trunkForward(tk, tr, zi, par ? lm : nullptr, par ? z0 : nullptr, T, lp, p, 17, z);
    g::ef2::distogramForward(dg, z, T, lg);
  };
  g::ARENA.reset();
  ss = g::ef2::shimForward(sh, stc, R, t2r, T, lm);
  g::ef2::zinitForward(zm, tok, sIn, par ? nullptr : lm, zi);
  g::ef2::trunkForward(tk, tr, zi, par ? lm : nullptr, par ? z0 : nullptr, T, lp, p, 17, z);
  g::ef2::distogramForward(dg, z, T, lg);
  auto ctD = gauss(PT * B, 9, 1), ctZ = gauss(P, 10, 0.3);
  g::real *ct = devReal(ctD), *ctz = devReal(ctZ);
  L.ps.zeroGrad();
  g::real *dz, *dzi, *dlm, *dsIn, *dz0;
  for (g::real** q : {&dz, &dzi, &dlm, &dz0}) { GK(cudaMalloc(q, P * sizeof(g::real))); GK(cudaMemset(*q, 0, P * sizeof(g::real))); }
  GK(cudaMalloc(&dsIn, (size_t)T * Si * sizeof(g::real))); GK(cudaMemset(dsIn, 0, (size_t)T * Si * sizeof(g::real)));
  GK(cudaMemcpy(dz, ctz, P * sizeof(g::real), cudaMemcpyDeviceToDevice));     // (the trunk's pair is an output too)
  g::ef2::distogramBackward(dg, z, T, ct, dz);
  g::ef2::trunkBackward(tk, tr, zi, par ? lm : nullptr, dz, dzi, par ? dlm : nullptr, par ? dz0 : nullptr);
  g::ef2::zinitBackward(zm, tok, sIn, dzi, dsIn, par ? nullptr : dlm);
  g::ef2::shimBackward(sh, ss, dlm, nullptr, dst);
  GK(cudaStreamSynchronize(STREAM));
  g::GradCheck gc;
  gc.rel = getenv("GRAD_REL") ? atof(getenv("GRAD_REL")) : 1e-7;
  gc.verbose = getenv("GRAD_VERBOSE") != nullptr;
  for (int k = 0; k < S; k += 9) gc.add("in/state" + std::to_string(k), st[k], dst[k], (size_t)R * Cm);
  gc.add("in/sInputs", sIn, dsIn, (size_t)T * Si);
  if (par) gc.add("in/z0", z0, dz0, P);
  for (auto* q : L.ps.all) gc.add(*q);
  gc.output(lg, ct, PT * B);
  gc.output(z, ctz, P);
  failed += gc.run(fwd);
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
