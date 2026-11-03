// RFD4-Proteina's autoencoder for training (cuda/grad/rfd4ae.cuh), two ways:
//   1. float32: the decoder against cuda/rfd4p's decode() on its exported weights and a real input (data-r2: the CA
//      positions, a latent and the chosen sequence) - its features, s_init, every layer, the three heads;
//   2. -DGRAD_F64: the backward against central differences on a small random decoder (two layers, every head), the
//      feature embeddings and the latent included.
//
//   nvcc -O1 -std=c++17 -arch=sm_80 -o cuda/grad/test-rfd4ae cuda/grad/test_rfd4ae.cu -lcublas -lcublasLt -ldl -lcurand
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 -o cuda/grad/test-rfd4ae-f64 cuda/grad/test_rfd4ae.cu -lcublas -lcurand
#ifndef GRAD_F64
#include "../rfd4p/src/ops.cuh"
#include "../rfd4p/src/encoder.cuh"
#endif
#include "gradcheck.cuh"
#include "rfd4ae.cuh"

namespace g = grad;

static g::real* devReal(const std::vector<double>& h) {
  std::vector<g::real> r(h.begin(), h.end());
  g::real* d; GK(cudaMalloc(&d, std::max<size_t>(r.size(), 1) * sizeof(g::real)));
  GK(cudaMemcpy(d, r.data(), r.size() * sizeof(g::real), cudaMemcpyHostToDevice));
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
static const char* DEC = "autoencoder.decoder";

int main(int argc, char** argv) {
  int failed = 0;
  std::string weights = "cuda/rfd4p/weights-f32", input = "cuda/rfd4p/data-r2";
  bool encoder = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--encoder")) encoder = true;
    if (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
#ifdef GRAD_F64
  g::init();
  {
    const int n = 7, pad = 1, latent = 4, nTypes = 6, types = 3;
    g::rfd4p::AeStack a; a.D = 24; a.C = 8; a.H = 2; a.nlayers = 2; a.fdim = 3 + latent + 32 + nTypes;
    g::Params ps;
    auto f = [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); };
    g::rfd4p::aeStackParams(a, DEC, {{"logit_linear", nTypes}, {"struct_linear", 111}, {"atom_presence_linear", 37}}, f);
    g::Param typeEmb, seqEmb;
    f("typeEmb", typeEmb, (size_t)types * 32); f("seqEmb", seqEmb, (size_t)nTypes * nTypes);
    unsigned k = 17;
    for (auto* p : ps.all) {
      bool norm = p->name.find("norm") != std::string::npos && p->name.find(".weight") != std::string::npos;
      auto v = gauss(p->n, ++k, norm ? 0.1 : 0.3);
      if (norm) for (auto& x : v) x += 1;
      std::vector<g::real> r(v.begin(), v.end());
      GK(cudaMemcpy(p->w, r.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    std::vector<double> m(n, 1); for (int i = n - pad; i < n; ++i) m[i] = 0;
    g::real *ca = devReal(gauss((size_t)n * 3, 21, 1)), *z = devReal(gauss((size_t)n * latent, 22, 1)), *mk = devReal(m);
    int *tt = devI({0, 1, 2, 1, 0, 2, 1}), *seq = devI({0, 3, 5, 1, 0, 2, 4});
    auto zp = [&](std::vector<double> v, size_t per) { for (int i = n - pad; i < n; ++i) for (size_t c = 0; c < per; ++c) v[(size_t)i * per + c] = 0; return v; };
    g::real *c0 = devReal(zp(gauss((size_t)n * nTypes, 31, 1), nTypes)), *c1 = devReal(zp(gauss((size_t)n * 111, 32, 1), 111)),
            *c2 = devReal(zp(gauss((size_t)n * 37, 33, 1), 37));
    g::real *o0, *o1, *o2, *dz;
    GK(cudaMalloc(&o0, (size_t)n * nTypes * sizeof(g::real))); GK(cudaMalloc(&o1, (size_t)n * 111 * sizeof(g::real)));
    GK(cudaMalloc(&o2, (size_t)n * 37 * sizeof(g::real))); GK(cudaMalloc(&dz, (size_t)n * latent * sizeof(g::real)));
    GK(cudaMemset(dz, 0, (size_t)n * latent * sizeof(g::real)));
    auto forward = [&](g::real*& feats) {
      feats = g::ARENA.get((size_t)n * a.fdim);
      g::rfd4p::decFeatsK<<<g::blocks((size_t)n * a.fdim), 256, 0, g::STREAM>>>(ca, z, tt, seq, typeEmb.w, seqEmb.w, mk, feats, n, latent, nTypes, a.fdim);
      return g::rfd4p::aeStackForward(a, feats, mk, n);
    };
    auto fwd = [&] {
      g::ARENA.reset(); g::real* feats; auto r = forward(feats);
      GK(cudaMemcpyAsync(o0, r.out[0], (size_t)n * nTypes * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
      GK(cudaMemcpyAsync(o1, r.out[1], (size_t)n * 111 * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
      GK(cudaMemcpyAsync(o2, r.out[2], (size_t)n * 37 * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM));
    };
    ps.zeroGrad();
    g::ARENA.reset();
    g::real* feats; auto r = forward(feats);
    g::real* df = g::ARENA.get((size_t)n * a.fdim);
    g::rfd4p::aeStackBackward(a, r, feats, mk, n, {c0, c1, c2}, df);
    g::rfd4p::decFeatsBwdK<<<g::blocks((size_t)n * a.fdim), 256, 0, g::STREAM>>>(df, tt, seq, mk, dz, typeEmb.g, seqEmb.g, n, latent, nTypes, a.fdim);
    g::GradCheck gc;
    gc.add("in/latent", z, dz, (size_t)n * latent);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(o0, c0, (size_t)n * nTypes); gc.output(o1, c1, (size_t)n * 111); gc.output(o2, c2, (size_t)n * 37);
    printf("RFD4-Proteina autoencoder decoder (small, random: two layers, every head):\n");
    failed += gc.run(fwd);
  }
  {
    const int n = 5, ttDim = 4, chargeDim = 3, sd = 8 + chargeDim;
    g::rfd4p::EncLayout EL{ttDim, chargeDim, ttDim, ttDim + 148, ttDim + 296, ttDim + 297, ttDim + 297 + 370};
    const int fdim = ttDim + 297 + 370 + 37 * sd;
    g::rfd4p::AeStack a; a.D = 16; a.C = 8; a.H = 2; a.nlayers = 1; a.fdim = fdim;
    g::Params ps;
    auto f = [&](const std::string& nm, g::Param& p, size_t len) { g::allocParam(p, nm, len); ps.add(p); };
    g::rfd4p::aeStackParams(a, "autoencoder.encoder", {{"latent_decoder_mean_n_log_scale", 6}}, f);
    g::Param ttE, elE, chE;
    f("ttEmb", ttE, (size_t)3 * ttDim); f("elEmb", elE, (size_t)6 * 8); f("chEmb", chE, (size_t)11 * chargeDim);
    unsigned k = 71;
    for (auto* p : ps.all) {
      bool norm = p->name.find("norm") != std::string::npos && p->name.find(".weight") != std::string::npos;
      auto v = gauss(p->n, ++k, norm ? 0.1 : 0.3);
      if (norm) for (auto& x : v) x += 1;
      std::vector<g::real> rr(v.begin(), v.end());
      GK(cudaMemcpy(p->w, rr.data(), p->n * sizeof(g::real), cudaMemcpyHostToDevice));
    }
    std::vector<double> am((size_t)n * 37);
    for (size_t s2 = 0; s2 < am.size(); ++s2) am[s2] = (s2 % 37 < 9 || s2 % 5 == 0) && s2 % 37 != 1 + 37 * 0 ? 1 : 0;   // (token 0 with no CA)
    std::vector<int> el((size_t)n * 37), chg((size_t)n * 37);
    for (size_t s2 = 0; s2 < el.size(); ++s2) { el[s2] = (int)(s2 % 6); chg[s2] = (int)(s2 % 5) - 2; }
    g::real *co = devReal(gauss((size_t)n * 111, 81, 1)), *amD = devReal(am), *chn = devReal(gauss(n, 82, 1)),
            *bd = devReal(gauss((size_t)n * 370, 83, 1)), *pd = devReal(std::vector<double>(n, 1));
    int *ttD = devI({0, 2, 1, 1, 0}), *elD = devI(el), *chD = devI(chg);
    g::real* ct = devReal(gauss((size_t)n * 6, 84, 1));
    g::real* o; GK(cudaMalloc(&o, (size_t)n * 6 * sizeof(g::real)));
    auto forward = [&](g::real*& feats) {
      feats = g::ARENA.get((size_t)n * fdim);
      g::rfd4p::encFeatsK<<<g::blocks((size_t)n * fdim), 256, 0, g::STREAM>>>(co, amD, ttD, chn, bd, elD, chD, pd, ttE.w, elE.w, chE.w, EL, feats, n, fdim);
      return g::rfd4p::aeStackForward(a, feats, pd, n);
    };
    auto fwd = [&] { g::ARENA.reset(); g::real* ft; auto r = forward(ft); GK(cudaMemcpyAsync(o, r.out[0], (size_t)n * 6 * sizeof(g::real), cudaMemcpyDeviceToDevice, g::STREAM)); };
    ps.zeroGrad();
    g::ARENA.reset();
    g::real* ft; auto r = forward(ft);
    g::real* df = g::ARENA.get((size_t)n * fdim);
    g::rfd4p::aeStackBackward(a, r, ft, pd, n, {ct}, df);
    g::rfd4p::encFeatsBwdK<<<g::blocks((size_t)n * fdim), 256, 0, g::STREAM>>>(df, co, amD, ttD, chn, bd, elD, chD, pd, ttE.w, elE.w, chE.w, EL,
                                                                            ttE.g, elE.g, chE.g, n, fdim);
    g::GradCheck gc;
    for (auto* p : ps.all) gc.add(*p);
    gc.output(o, ct, (size_t)n * 6);
    printf("RFD4-Proteina autoencoder encoder (small, random: its features' three tables):\n");
    failed += gc.run(fwd);
  }
#else
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  CB(cublasSetMathMode(H, CUBLAS_PEDANTIC_MATH));
  g::STREAM = STREAM;
  g::init();
  M.load(weights);
  if (encoder) { M.load("cuda/rfd4p/data-pd"); M.load("cuda/rfd4p/data-pd-enc"); }
  else M.load(input);
  auto cmp = [&](const std::string& what, const g::real* mine, const std::vector<float>& port) {
    double e = g::relRms(g::host(mine, port.size()), port);
    printf("  %-18s %.3e\n", what.c_str(), e);
    if (!(e < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
  };
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    GK(cudaMemcpy(p.w, W(nm), len * 4, cudaMemcpyDeviceToDevice));
  };
  std::map<std::string, std::vector<float>> taps;
  if (!encoder) {
  const int n = (int)M.meta("meta/n");
  rfd4p::DecDims dd;
  dd.n = n; dd.D = (int)M.meta("meta/decTokenDim"); dd.C = (int)M.meta("meta/decCondDim"); dd.H = (int)M.meta("meta/decHeads");
  dd.nlayers = (int)M.meta("meta/decLayers"); dd.nTypes = (int)M.meta("meta/nResidueTypes");
  dd.latent = (int)M.meta("meta/latentDim"); dd.fdim = (int)M.meta("meta/decFeatDim");
  g::rfd4p::AeStack a; a.D = dd.D; a.C = dd.C; a.H = dd.H; a.nlayers = dd.nlayers; a.fdim = dd.fdim;
  g::rfd4p::aeStackParams(a, DEC, {{"logit_linear", dd.nTypes}, {"struct_linear", 111}, {"atom_presence_linear", 37}}, load);
  g::real* feats = g::ARENA.get((size_t)n * dd.fdim);
  g::rfd4p::decFeatsK<<<g::blocks((size_t)n * dd.fdim), 256, 0, g::STREAM>>>(Fdev("in/caNm"), Fdev("in/zLatent"), Idev("in/tokenType"),
      Idev("in/decSeq"), W("w/autoencoder.decoder.init_repr_factory.feat_creators.2.embedding.weight"),
      W("w/autoencoder.decoder.init_repr_factory.feat_creators.3.embedding.weight"), Fdev("in/attnMask"), feats, n, dd.latent, dd.nTypes, dd.fdim);
  auto r = g::rfd4p::aeStackForward(a, feats, Fdev("in/attnMask"), n);
  float *pl = dalloc((size_t)n * dd.nTypes), *pc = dalloc((size_t)n * 111), *pp = dalloc((size_t)n * 37);
  rfd4p::decode(dd, Fdev("in/caNm"), Fdev("in/zLatent"), Idev("in/tokenType"), Idev("in/decSeq"), Fdev("in/attnMask"), pl, pc, pp,
                [&](const char* nm, const float* p, size_t len) { taps[nm] = g::host(p, len); });
  CK(cudaStreamSynchronize(STREAM));
  printf("RFD4-Proteina autoencoder decoder (%s, %s), %d tokens, %d layers\n", weights.c_str(), input.c_str(), n, dd.nlayers);
  cmp("features", feats, taps["d/feats"]); cmp("s_init", r.s[0], taps["d/sInit"]);
  for (int i = 0; i < dd.nlayers; ++i) cmp("s" + std::to_string(i), r.s[i + 1], taps["d/s" + std::to_string(i)]);
  cmp("logits", r.out[0], taps["d/logits"]); cmp("coordinates", r.out[1], taps["d/coors"]); cmp("presence", r.out[2], taps["d/presence"]);
  } else {
    // the encoder, on its own input (data-pd with data-pd-enc's all-atom tensors)
    const int ne = (int)M.meta("meta/n");
    rfd4p::EncDims ed;
    ed.n = ne; ed.D = (int)M.meta("meta/encTokenDim"); ed.C = (int)M.meta("meta/encCondDim"); ed.H = (int)M.meta("meta/encHeads");
    ed.nlayers = (int)M.meta("meta/encLayers"); ed.latent = (int)M.meta("meta/latentDim"); ed.fdim = (int)M.meta("meta/encFeatDim");
    ed.ttDim = (int)M.meta("meta/encTtDim"); ed.chargeDim = (int)M.meta("meta/encChargeDim");
    const int off[7] = {(int)M.meta("meta/encOffAbs"), (int)M.meta("meta/encOffRel"), (int)M.meta("meta/encOffChain"),
                        (int)M.meta("meta/encOffBonds"), (int)M.meta("meta/encOffElCh"), (int)M.meta("meta/encTtCreator"),
                        (int)M.meta("meta/encElChCreator")};
    g::rfd4p::AeStack e; e.D = ed.D; e.C = ed.C; e.H = ed.H; e.nlayers = ed.nlayers; e.fdim = ed.fdim;
    g::rfd4p::aeStackParams(e, "autoencoder.encoder", {{"latent_decoder_mean_n_log_scale", 2 * ed.latent}}, load);
    const std::string FC = "w/autoencoder.encoder.init_repr_factory.feat_creators.";
    g::rfd4p::EncLayout EL{ed.ttDim, ed.chargeDim, off[0], off[1], off[2], off[3], off[4]};
    g::real* ef = g::ARENA.get((size_t)ne * ed.fdim);
    g::rfd4p::encFeatsK<<<g::blocks((size_t)ne * ed.fdim), 256, 0, g::STREAM>>>(Fdev("e/coorsNm"), Fdev("e/atomMask"), Idev("in/tokenType"),
        Fdev("e/chainIid"), Fdev("e/bondCounts"), Idev("e/element"), Idev("e/charge"), Fdev("in/mask"),
        W(FC + std::to_string(off[5]) + ".embedding.weight"), W(FC + std::to_string(off[6]) + ".element_embedding.weight"),
        W(FC + std::to_string(off[6]) + ".charge_embedding.weight"), EL, ef, ne, ed.fdim);
    auto er = g::rfd4p::aeStackForward(e, ef, Fdev("in/attnMask"), ne);
    float *mean = dalloc((size_t)ne * ed.latent), *ls = dalloc((size_t)ne * ed.latent), *zl = dalloc((size_t)ne * ed.latent);
    taps.clear();
    rfd4p::encode(ed, Fdev("e/coorsNm"), Fdev("e/atomMask"), Idev("in/tokenType"), Fdev("e/chainIid"), Fdev("e/bondCounts"),
                  Idev("e/element"), Idev("e/charge"), Fdev("in/mask"), Fdev("in/attnMask"), Fdev("e/emit"), nullptr, mean, ls, zl, off,
                  [&](const char* nm, const float* p, size_t len) { taps[nm] = g::host(p, len); });
    CK(cudaStreamSynchronize(STREAM));
    printf("RFD4-Proteina autoencoder encoder (data-pd + data-pd-enc), %d tokens, %d layers\n", ne, ed.nlayers);
    cmp("features", ef, taps["e/feats"]); cmp("s_init", er.s[0], taps["e/sInit"]);
    cmp("s (last layer)", er.s.back(), taps["e/s" + std::to_string(ed.nlayers - 1)]);
    cmp("mean | log scale", er.out[0], taps["e/flat"]);
  }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
