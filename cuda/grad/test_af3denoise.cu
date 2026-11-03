// The AF3 lineage's whole denoiser for training (cuda/grad/af3denoise.cuh), two ways:
//   1. float32: D(x, sigma) against cuda/af3's denoise() on a family's bundle and a real featurised input
//      (af3-featurise --no-weights: the atom layout, the reference features, the relative-position features), with a
//      random trunk single, trunk pair, target_feat and noisy positions;
//   2. -DGRAD_F64: the same denoiser on the same input and the same bundle's widths, shrunk - the bundle's weights
//      perturbed are what the gradcheck differentiates - against central differences. (Its shape is the bundle's, so
//      it is slow: a handful of tensors are checked, each a few entries.)
//
//   nvcc -O1 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/test-af3denoise cuda/grad/test_af3denoise.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af3denoise [--bundle=model-af3-full-f32] [--family=af3] [--input=cuda/grad/data/af3-in-af3] [--sigma=5]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 --default-stream per-thread -o cuda/grad/test-af3denoise-f64 cuda/grad/test_af3denoise.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af3/src/trunk.cuh"
#include "../af3/src/diffusion.cuh"
#include "../af3/src/sampler.cuh"
#include "../af3/src/scores.cuh"
#include "../af3/src/structural.cuh"
#include "../featurise/af3_weights.h"
#include "af3denoise.cuh"
#include "gradcheck.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, r.size() * sizeof(g::real)));
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
static float* devF(const std::vector<double>& h) {
  std::vector<float> r(h.begin(), h.end());
  float* d; GK(cudaMalloc(&d, r.size() * 4));
  GK(cudaMemcpy(d, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
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
static g::af3::Gather gatherG(const ::Gather& q) { return g::af3::Gather{q.idx, devReal(q.mask, q.count), q.count}; }

int main(int argc, char** argv) {
  int failed = 0;
  std::string bundle = "model-af3-full-f32", family = "af3", input = "cuda/grad/data/af3-in-af3";
  double sigma = 5;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--family=", 9)) family = argv[i] + 9;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
    else if (!strncmp(argv[i], "--sigma=", 8)) sigma = atof(argv[i] + 8);
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines(family, S); });
  M.load(input);
  const bool chai = M.flag("trunk.dialect.chaiAtomStack");
  setAdaMode(chai);
  // OpenDDE's diffusion runs on its structural tokens: the structural batch in place of the residue one (cuda/af3's)
  if (M.flag("trunk.dialect.structuralTokens")) swapBatch();
  AtomShape sh = atomShape();
  const int n = sh.tokens;
  const std::string P = "diffusion.conditioning", E = "diffusion.encoder", Dd = "diffusion.decoder", Tn = "diffusion.transformer";
  auto has = [&](const std::string& k) { return M.has(k); };
  g::af3::Denoiser m;
  g::af3::DiffCond& c = m.cond;
  c.Cz = (int)M.meta(P + ".pairChannels"); c.Cs = (int)M.meta(P + ".seqChannels");
  c.Czt = (int)M.meta(P + ".trunkPairChannels"); c.Cst = (int)M.meta(P + ".trunkSingleChannels");
  c.F = (int)M.meta(P + ".targetFeatWidth"); c.rel = (int)M.meta(P + ".relativeWidth");
  c.nc = (int)M.at(P + ".fourierWeight").length; c.pad = M.flag("trunk.dialect.padSingleCondUnknownDna");
  c.chai = M.flag("trunk.dialect.chaiDiffusionConditioning");
  c.split = !c.chai && has(P + ".zTrunkProjection"); c.relpe = !c.chai && !c.split && has(P + ".relpeProjection");
  g::af3::DiffEncoder& e = m.enc;
  e.chai = chai; m.decChai = chai;
  e.C = (int)M.meta(E + ".channels"); e.Cp = (int)M.meta(E + ".pairChannels"); e.H = (int)M.meta(E + ".heads");
  e.D = (int)M.meta(E + ".dimension"); e.perToken = (int)M.meta(E + ".perTokenChannels");
  e.Cst = (int)M.meta(E + ".trunkSingleChannels"); e.Cz = (int)M.meta(E + ".trunkPairChannels");
  e.rawCharge = M.flag("trunk.dialect.rawRefCharge"); e.preTrunkQuery = M.flag("trunk.dialect.preTrunkQuery");
  e.keysUidFromQueries = M.flag("trunk.dialect.keysUidFromQueriesLayout"); e.maskPaddedKeys = M.flag("trunk.dialect.maskPaddedKeys");
  int nbE = 0; while (M.has(E + ".blocks." + std::to_string(nbE) + ".qProjection")) ++nbE;
  int nbD = 0; while (M.has(Dd + ".blocks." + std::to_string(nbD) + ".qProjection")) ++nbD;
  m.decStack.C = (int)M.meta(Dd + ".channels"); m.decStack.Cp = (int)M.meta(Dd + ".pairChannels");
  m.decStack.H = (int)M.meta(Dd + ".heads"); m.decStack.D = (int)M.meta(Dd + ".dimension");
  g::af3::DiffTransformer& t = m.tr;
  t.C = (int)M.meta(Tn + ".channels"); t.Cc = (int)M.meta(Tn + ".condChannels"); t.Cz = (int)M.meta(Tn + ".pairChannels");
  t.H = (int)M.meta(Tn + ".heads"); t.D = (int)M.meta(Tn + ".dimension"); t.perSuper = (int)M.meta(Tn + ".blocksPerSuperBlock");
  t.factor = (int)M.meta(Tn + ".transitionFactor"); t.noResidual = M.flag(Tn + ".noResidual"); t.chai = chai;
  int nSuper = 0; while (M.has(Tn + ".superBlocks." + std::to_string(nSuper) + ".pairLogitsProjection")) ++nSuper;
  const bool perBlock = M.flag(E + ".pairNormPerBlock"), up = M.has(E + ".blocks.0.ffwAToB"),
             kq = M.has(E + ".blocks.0.queryLayerNormScale"), noTr2 = !M.has(E + ".blocks.0.Transition2");
  const bool tUp = M.has(Tn + ".superBlocks.0.blocks.0.ffwAToB"), tKq = M.has(Tn + ".superBlocks.0.blocks.0.queryLayerNormScale");
  g::Params ps;
  g::af3::forEachParam(m, nbE, nbD, nSuper, perBlock, up, kq, noTr2, tUp, tKq, has, [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), W(nm), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  });
  for (auto* st : {&e.stack, &m.decStack}) {
    const std::string B0 = (st == &e.stack ? E : Dd) + ".blocks.0";
    for (auto& b : st->blocks) {
      b.keyMasked = M.flag(B0 + ".keyMaskedAtomAttention"); b.chained = M.flag(B0 + ".chainedAtomLayerNorm");
      b.noResidual = M.flag(B0 + ".diffusionNoResidual"); b.maskActPerBlock = M.flag(B0 + ".maskAtomActPerBlock");
    }
  }
  if (getenv("GRAD_FALSIFY")) {       // a convention flipped: the gate must go red
    e.preTrunkQuery = !e.preTrunkQuery; e.maskPaddedKeys = !e.maskPaddedKeys;
    printf("  FALSIFIED: preTrunkQuery %d, maskPaddedKeys %d\n", e.preTrunkQuery, e.maskPaddedKeys);
  }
  // the batch
  const size_t atoms = (size_t)n * sh.dense, nn = (size_t)n * n;
  g::af3::AtomBatch b;
  b.tokens = n; b.dense = sh.dense; b.subsets = sh.subsets; b.Q = sh.queries; b.K = sh.keys;
  b.t2q = gatherG(gatherOf("batch.tokenAtomsToQueries")); b.q2k = gatherG(gatherOf("batch.queriesToKeys"));
  b.q2t = gatherG(gatherOf("batch.queriesToTokenAtoms")); b.tq = gatherG(gatherOf("batch.tokensToQueries"));
  b.tk = gatherG(gatherOf("batch.tokensToKeys"));
  b.refPos = devReal(Fdev("batch.refPos"), atoms * 3); b.refMask = devReal(Fdev("batch.refMask"), atoms);
  b.refCharge = devReal(Fdev("batch.refCharge"), atoms); b.atomMask = devReal(Fdev("batch.predDenseAtomMask"), atoms);
  {
    std::vector<int> u(M.len("batch.refSpaceUid")); GK(cudaMemcpy(u.data(), Idev("batch.refSpaceUid"), u.size() * 4, cudaMemcpyDeviceToHost));
    b.uid = devReal(std::vector<double>(u.begin(), u.end()));
  }
  b.refElement = Idev("batch.refElement"); b.nameChars = Idev("batch.refAtomNameChars");
  const int *ri = Idev("batch.features.residueIndex"), *ti = Idev("batch.features.tokenIndex"), *asym = Idev("batch.features.asymId"),
            *ent = Idev("batch.features.entityId"), *sym = Idev("batch.features.symId");
  if (c.chai) {      // chai-1's structure pair: the dense ranks of the entity and sym ids, the declared bonds
    auto rank = [&](const char* k) {
      const int* h = M.i(k); size_t len = M.len(k);
      std::vector<int> sorted(h, h + len); std::sort(sorted.begin(), sorted.end()); sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
      std::vector<int> r(len); for (size_t q = 0; q < len; ++q) r[q] = (int)(std::lower_bound(sorted.begin(), sorted.end(), h[q]) - sorted.begin());
      int* d; GK(cudaMalloc(&d, len * 4)); GK(cudaMemcpy(d, r.data(), len * 4, cudaMemcpyHostToDevice)); return (const int*)d;
    };
    c.entRank = rank("batch.features.entityId"); c.symRank = rank("batch.features.symId");
    if (M.has("batch.declaredBondMatrix")) c.declBonds = devReal(Fdev("batch.declaredBondMatrix"), nn);
  }
  // random trunk outputs and noisy positions (the positions at sigma's scale)
  auto tsH = gauss((size_t)n * c.Cst, 1, 1), tpH = gauss(nn * c.Czt, 2, 1), tfH = gauss((size_t)n * c.F, 3, 0.5),
       xH = gauss(atoms * 3, 4, sigma);
  std::vector<double> smH(n, 1);
  g::real *ts = devReal(tsH), *tp = devReal(tpH), *tf = devReal(tfH), *x = devReal(xH), *sm = devReal(smH), *out;
  GK(cudaMalloc(&out, atoms * 3 * sizeof(g::real)));
  printf("%s (%s), %d tokens, sigma %g: encoder %d blocks, decoder %d, transformer %d; pair %d, single %d, atoms C %d%s%s%s\n",
         family.c_str(), bundle.c_str(), n, sigma, nbE, nbD, (int)t.blocks.size(), c.Cz, c.Cs, e.C, c.split ? ", split pair" : "",
         c.relpe ? ", projected relpe" : "", e.preTrunkQuery ? ", pre-trunk queries" : "");
  {
  g::ARENA.reset();
  GRAPHS = false;
  const float* ref = denoise(devF(tsH), devF(tpH), devF(tfH), devF(smH), devF(xH), (float)sigma);
  CK(cudaStreamSynchronize(STREAM));
  if (getenv("GRAD_SENS")) {          // the port against itself, its inputs moved by float32's last place
    std::vector<float> r0 = hostF(ref, atoms * 3);
    auto nudge = [](std::vector<double> v, unsigned seed) { auto e = gauss(v.size(), seed, 1e-7); for (size_t i = 0; i < v.size(); ++i) v[i] *= 1 + e[i]; return v; };
    const float* r1 = denoise(devF(nudge(tsH, 31)), devF(nudge(tpH, 32)), devF(nudge(tfH, 33)), devF(smH), devF(nudge(xH, 34)), (float)sigma);
    CK(cudaStreamSynchronize(STREAM));
    auto h1 = hostF(r1, atoms * 3);
    auto mkh = hostF(Fdev("batch.predDenseAtomMask"), atoms);
    std::vector<float> a0, a1;
    for (size_t i = 0; i < atoms; ++i) if (mkh[i] != 0) for (int k = 0; k < 3; ++k) { a0.push_back(r0[i * 3 + k]); a1.push_back(h1[i * 3 + k]); }
    printf("  the port against itself, inputs moved 1e-7: %.3e\n", g::relRms(a1, a0));
    ref = denoise(devF(tsH), devF(tpH), devF(tfH), devF(smH), devF(xH), (float)sigma);
    CK(cudaStreamSynchronize(STREAM));
  }
  // rf3's chirality term: the port's, of these positions, handed to this denoiser as the constant it takes
  if (e.hasChiral) b.chiral = devReal(scratch<float>("enc.chiralGrads", atoms * 3), atoms * 3);
  auto ds = g::af3::denoiseForward(m, b, ts, tp, tf, sm, x, (g::real)sigma, ri, ti, asym, ent, sym, out);
  CK(cudaStreamSynchronize(STREAM));
  if (getenv("GRAD_STAGES")) {      // where a disagreement starts
    auto cmp = [&](const char* what, const g::real* mine, const float* port, size_t len) {
      double e = g::relRms(g::host(mine, len), hostF(port, len));
      printf("  stage %-22s %.3e\n", what, e);
    };
    cmp("conditioning.single", ds.cs.sFinal, scratch<float>("dc.single", (size_t)n * c.Cs), (size_t)n * c.Cs);
    cmp("conditioning.pair", ds.cs.pairOut, DCACHE.pair, nn * c.Cz);
    cmp("encoder.qCond", ds.es.qCond, scratch<float>(E + ".qCond", (size_t)sh.subsets * sh.queries * e.C), (size_t)sh.subsets * sh.queries * e.C);
    cmp("encoder.pair", ds.es.pair, scratch<float>(E + ".pair", (size_t)sh.subsets * sh.queries * sh.keys * e.Cp), (size_t)sh.subsets * sh.queries * sh.keys * e.Cp);
    cmp("encoder.skip", ds.es.skip, scratch<float>(E + ".act", (size_t)sh.subsets * sh.queries * e.C), (size_t)sh.subsets * sh.queries * e.C);
    cmp("transformer.out (LN)", ds.actn, scratch<float>(E + ".tokenAct", (size_t)n * e.perToken), (size_t)n * e.perToken);
    cmp("decoder.update", ds.ds.upd, scratch<float>("dec.upd", (size_t)sh.subsets * sh.queries * 3), (size_t)sh.subsets * sh.queries * 3);
  }
  auto o = g::host(out, atoms * 3);
  auto r = hostF(ref, atoms * 3), mk = hostF(Fdev("batch.predDenseAtomMask"), atoms);
  std::vector<float> a, bb;
  for (size_t i = 0; i < atoms; ++i) if (mk[i] != 0) for (int k = 0; k < 3; ++k) { a.push_back(o[i * 3 + k]); bb.push_back(r[i * 3 + k]); }
  // the update alone, D - skip x: what the network computed (the skip term dominates at a small sigma)
  const double d = sigma * sigma + 256, sk = 256 / d;
  std::vector<float> ua, ub;
  for (size_t i = 0, j = 0; i < atoms; ++i) if (mk[i] != 0) for (int k = 0; k < 3; ++k, ++j) {
    ua.push_back(a[j] - sk * xH[i * 3 + k]); ub.push_back(bb[j] - sk * xH[i * 3 + k]);
  }
  if (getenv("GRAD_SAVE")) { FILE* fp = fopen(getenv("GRAD_SAVE"), "wb"); fwrite(ua.data(), 4, ua.size(), fp); fclose(fp); }
  if (getenv("GRAD_AGAINST")) {       // this denoiser's float32 run (GRAD_SAVE) against this one
    std::vector<float> other(ua.size()); FILE* fp = fopen(getenv("GRAD_AGAINST"), "rb");
    if (fp && fread(other.data(), 4, other.size(), fp) == other.size()) printf("  against %s: %.3e\n", getenv("GRAD_AGAINST"), g::relRms(ua, other));
    if (fp) fclose(fp);
  }
  double eo = g::relRms(a, bb), eu = g::relRms(ua, ub);
  printf("  against cuda/af3's denoise(): D %.3e, its update %.3e, over %zu real atoms\n", eo, eu, a.size() / 3);
#ifdef GRAD_F64
  printf("  (float64: the port's own float32 rounding is what is left)\n");
#else
  // 1e-3: the 24-block transformer's float32 is all that is left - this denoiser's float64 is as far from the port as
  // its float32 (boltz2 4.0e-4 / 4.1e-4), and the transformer alone reads 7e-5..1.5e-4 against the port on random
  // inputs (test-af3diff --n=68); a convention flipped (GRAD_FALSIFY) reads 1e-2 and up
  if (!(eu < 1e-3)) { printf("  <-- ABOVE 1e-3\n"); ++failed; }
#endif
  }
#ifdef GRAD_F64
  // the gradcheck: every input and a sample of the parameters, each at a few entries
  auto ctH = gauss(atoms * 3, 9, 1);
  for (size_t i = 0; i < atoms; ++i) if (xH.size() && g::host(b.atomMask + i, 1)[0] == 0) for (int k = 0; k < 3; ++k) ctH[i * 3 + k] = 0;
  g::real* ct = devReal(ctH);
  g::real *dts, *dtp, *dtf, *dx;
  GK(cudaMalloc(&dts, (size_t)n * c.Cst * sizeof(g::real))); GK(cudaMalloc(&dtp, nn * c.Czt * sizeof(g::real)));
  GK(cudaMalloc(&dtf, (size_t)n * c.F * sizeof(g::real))); GK(cudaMalloc(&dx, atoms * 3 * sizeof(g::real)));
  for (auto* p : {dts, dtf}) GK(cudaMemset(p, 0, (p == dts ? (size_t)n * c.Cst : (size_t)n * c.F) * sizeof(g::real)));
  GK(cudaMemset(dtp, 0, nn * c.Czt * sizeof(g::real))); GK(cudaMemset(dx, 0, atoms * 3 * sizeof(g::real)));
  ps.zeroGrad();
  g::ARENA.reset();
  auto s = g::af3::denoiseForward(m, b, ts, tp, tf, sm, x, (g::real)sigma, ri, ti, asym, ent, sym, out);
  g::af3::denoiseBackward(m, b, s, ts, sm, ct, dts, dtp, dtf, dx);
  if (getenv("GRAD_KINK")) {          // the aggregation's ReLU inputs on real queries: how many at or near zero
    const size_t Rq = (size_t)sh.subsets * sh.queries;
    auto pj = g::host(s.es.proj, Rq * e.perToken); auto qm = g::host(s.es.qMask, Rq);
    double rms = 0; size_t real_ = 0, zero = 0, near = 0;
    for (size_t q = 0; q < Rq; ++q) if (qm[q] != 0) for (int c2 = 0; c2 < e.perToken; ++c2) { rms += pj[q * e.perToken + c2] * pj[q * e.perToken + c2]; ++real_; }
    rms = std::sqrt(rms / real_);
    for (size_t q = 0; q < Rq; ++q) if (qm[q] != 0) for (int c2 = 0; c2 < e.perToken; ++c2) {
      double v = std::fabs(pj[q * e.perToken + c2]); if (v == 0) ++zero; else if (v < 1e-6 * rms) ++near;
    }
    printf("  aggregation ReLU inputs: %zu real, %zu exactly 0, %zu within 1e-6 rms\n", real_, zero, near);
    auto sk = g::host(s.es.skip, Rq * e.C);
    std::vector<int> t2qi(Rq); GK(cudaMemcpy(t2qi.data(), b.t2q.idx, Rq * 4, cudaMemcpyDeviceToHost));
    auto pm = hostF(Fdev("batch.predDenseAtomMask"), atoms);
    int shown = 0; size_t rowsNear = 0, chNear[2048] = {0};
    for (size_t q = 0; q < Rq; ++q) if (qm[q] != 0) {
      int cnt = 0; double sn = 0;
      for (int c2 = 0; c2 < e.perToken; ++c2) if (std::fabs(pj[q * e.perToken + c2]) < 1e-6 * rms) { ++cnt; ++chNear[c2]; }
      for (int c2 = 0; c2 < e.C; ++c2) sn += sk[q * e.C + c2] * sk[q * e.C + c2];
      if (cnt) { ++rowsNear; if (shown++ < 4) printf("    query %zu (atom %d, pred mask %g): %d channels near 0, |skip| %.3e\n", q, t2qi[q], pm[t2qi[q]], cnt, std::sqrt(sn)); }
    }
    int chans = 0; for (int c2 = 0; c2 < e.perToken; ++c2) chans += chNear[c2] > 0;
    printf("    %zu rows touched, %d channels touched\n", rowsNear, chans);
    auto Wa = g::host(e.aggr.w.w, (size_t)e.C * e.perToken);
    double wr = 0; for (double v : Wa) wr += v * v; wr = std::sqrt(wr / Wa.size());
    int dead = 0; double deadMax = 0;
    for (int c2 = 0; c2 < e.perToken; ++c2) if (chNear[c2]) {
      double cn = 0; for (int k2 = 0; k2 < e.C; ++k2) cn = std::max(cn, std::fabs((double)Wa[(size_t)k2 * e.perToken + c2]));
      ++dead; deadMax = std::max(deadMax, cn);
    }
    printf("    those channels' largest weight %.3e against the tensor's rms %.3e (%d channels)\n", deadMax, wr, dead);
  }
  g::GradCheck gc;
  // a step 10x under the default: the encoder's pair MLP has ~4M ReLU inputs, and at 1e-7 a few sit inside +-2h of
  // their kink (pairMlp1 read 2e-4; at 1e-8 the worst tensor of all 688 is 1.1e-5)
  // (h = 1e-8 for the ReLU kinks of the atom pair MLP; OpenDDE's 130 structural tokens take 1e-7, where its zero-gate
  // biases' directional derivatives - 2.7e-9 - sit at 1e-8's difference floor: 1.06e-4 there, 1.2e-5 worst at 1e-7)
  gc.rel = getenv("GRAD_REL") ? atof(getenv("GRAD_REL")) : M.flag("trunk.dialect.structuralTokens") ? 1e-7 : 1e-8;
  gc.add("in/trunkSingle", ts, dts, (size_t)n * c.Cst);
  gc.add("in/trunkPair", tp, dtp, nn * c.Czt);
  gc.add("in/targetFeat", tf, dtf, (size_t)n * c.F);
  gc.add("in/x", x, dx, atoms * 3);
  for (auto* p : ps.all) gc.add(*p);
  gc.output(out, ct, atoms * 3);
  failed += gc.run([&] {
    g::ARENA.reset();
    g::af3::denoiseForward(m, b, ts, tp, tf, sm, x, (g::real)sigma, ri, ti, asym, ent, sym, out);
  });
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
