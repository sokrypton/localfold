// The AF3 lineage's input embedder and distogram head for training (cuda/grad/af3embed.cuh), two ways:
//   1. float32: target_feat against cuda/af3's buildTargetFeat(); z_init (before the template term), the single and
//      the MSA against its embed<float>() on pass 0, on the same MSA rows; the distogram against its distogramHalf +
//      symmetrise - on a family's bundle and a real featurised input;
//   2. -DGRAD_F64: the backward against central differences, the same input and bundle, random recycled pair and
//      single (so their LayerNorms are not at the zero input pass 0 has).
//
//   nvcc -O1 -std=c++17 -arch=sm_80 --default-stream per-thread -o cuda/grad/test-af3embed cuda/grad/test_af3embed.cu -lcublas -lcublasLt -ldl -lcurand
//   cuda/grad/test-af3embed [--bundle=model-af3-full-f32] [--family=af3] [--input=cuda/grad/data/af3-in-af3]
//   nvcc -O2 -std=c++17 -arch=sm_80 -DGRAD_F64 --default-stream per-thread -o cuda/grad/test-af3embed-f64 cuda/grad/test_af3embed.cu -lcublas -lcublasLt -ldl -lcurand
#include "../af3/src/trunk.cuh"
#include "../af3/src/diffusion.cuh"
#include "../featurise/af3_weights.h"
#include "af3embed.cuh"
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
static g::af3::Gather gatherG(const ::Gather& q) { return g::af3::Gather{q.idx, devReal(q.mask, q.count), q.count}; }

int main(int argc, char** argv) {
  int failed = 0;
  std::string bundle = "model-af3-full-f32", family = "af3", input = "cuda/grad/data/af3-in-af3", esmBundle = "af3am-esm2";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--esm-bundle=", 13)) esmBundle = argv[i] + 13;     // (chai-1's ESM2 3B)
    if (!strncmp(argv[i], "--bundle=", 9)) bundle = argv[i] + 9;
    else if (!strncmp(argv[i], "--family=", 9)) family = argv[i] + 9;
    else if (!strncmp(argv[i], "--input=", 8)) input = argv[i] + 8;
  }
  CB(cublasCreate(&H)); CB(cublasSetStream(H, STREAM));
  g::STREAM = STREAM;
  g::init();
  M.loadBundle(bundle, "", [&](const std::map<std::string, std::vector<long long>>& shapes) {
    lf::weights::Shapes S; S.shape = shapes; return lf::weights::af3WeightLines(family, S); });
  M.load(input);
  const bool chai = M.flag("trunk.dialect.chaiTokenEmbedding");
  if (chai) {     // its target_feat reads ESM2 3B's embeddings (af3.cu's --esm-bundle)
    const int seg = (int)M.segs.size();
    M.loadBundle(esmBundle, "e", nullptr, "", "esm2/blocks/");
    M.upload(seg);
  }
  setAdaMode(M.flag("trunk.dialect.chaiAtomStack"));
  AtomShape sh = atomShape();
  const int n = sh.tokens; const size_t nn = (size_t)n * n, atoms = (size_t)n * sh.dense;
  auto has = [&](const std::string& k) { return M.has(k); };
  // target_feat
  const std::string E = "targetFeat.encoder";
  g::af3::TargetFeat tfm;
  g::af3::DiffEncoder& e = tfm.enc;
  e.C = (int)M.meta(E + ".channels"); e.Cp = (int)M.meta(E + ".pairChannels"); e.H = (int)M.meta(E + ".heads");
  e.D = (int)M.meta(E + ".dimension"); e.perToken = (int)M.meta(E + ".perTokenChannels");
  // (the bundle may carry trunk-conditioning weights this encoder never reads - no trunk input: they load and stay at zero
  // gradient)
  e.Cst = has(E + ".embedTrunkSingleCond") ? (int)(M.at(E + ".embedTrunkSingleCond").length / e.C) : 0;
  e.Cz = has(E + ".embedTrunkPairCond") ? (int)(M.at(E + ".embedTrunkPairCond").length / e.Cp) : 0;
  e.chai = M.flag("trunk.dialect.chaiAtomStack");
  e.rawCharge = M.flag("trunk.dialect.rawRefCharge"); e.preTrunkQuery = M.flag("trunk.dialect.preTrunkQuery");
  e.keysUidFromQueries = M.flag("trunk.dialect.keysUidFromQueriesLayout"); e.maskPaddedKeys = M.flag("trunk.dialect.maskPaddedKeys");
  int nbE = 0; while (M.has(E + ".blocks." + std::to_string(nbE) + ".qProjection")) ++nbE;
  const bool perBlock = M.flag(E + ".pairNormPerBlock"), up = M.has(E + ".blocks.0.ffwAToB"),
             kq = M.has(E + ".blocks.0.queryLayerNormScale"), noTr2 = !M.has(E + ".blocks.0.Transition2");
  g::Params ps;
  auto load = [&](const std::string& nm, g::Param& p, size_t len) {
    g::allocParam(p, nm, len); ps.add(p);
    if (M.at(nm).length != len) { fprintf(stderr, "%s: %zu in the bundle, %zu expected\n", nm.c_str(), M.at(nm).length, len); exit(1); }
    std::vector<float> h(len); GK(cudaMemcpy(h.data(), W(nm), len * 4, cudaMemcpyDeviceToHost));
    std::vector<g::real> r(h.begin(), h.end());
    GK(cudaMemcpy(p.w, r.data(), len * sizeof(g::real), cudaMemcpyHostToDevice));
  };
  g::af3::forEachParam(e, E, "targetFeat.reference", nbE, perBlock, up, kq, noTr2, has, load);
  tfm.chai = chai;
  if (chai) {
    const std::string S = E + ".chaiToken.";
    const int Ct = (int)M.at(S + "tokenFeatureBias").length, P = e.perToken;
    g::af3::forEachParam(tfm.ct, P, Ct, (int)(M.at(S + "esmWeights").length / Ct), (int)(M.at(S + "singleProjInTrunk").length / (P + Ct)),
                         (int)(M.at(S + "singleProjInStructure").length / (P + Ct)), has(S + "dx"), load);
    tfm.flags.isDna = M.has("batch.isDna") ? Idev("batch.isDna") : nullptr;
  }
  tfm.atomOnly = M.flag("trunk.dialect.targetFeatAtomOnly");       // (boltz2: the encoder's columns plus six token terms)
  if (tfm.atomOnly) {
    const std::string S = E + ".targetFeatSum.";
    g::af3::forEachParam(tfm.sum, e.perToken, (int)(M.at(S + "method").length / e.perToken), has(S + "dnResType"), load);
    auto flag = [&](const char* k) { return M.has(k) ? Idev(k) : nullptr; };
    tfm.flags = g::af3::TokenFlags{flag("batch.isDna"), flag("batch.isRna"), flag("batch.isLigand"), flag("batch.isModified")};
  }
  for (auto& b : e.stack.blocks) {
    const std::string B0 = E + ".blocks.0";
    b.keyMasked = M.flag(B0 + ".keyMaskedAtomAttention"); b.chained = M.flag(B0 + ".chainedAtomLayerNorm");
    b.noResidual = M.flag(B0 + ".diffusionNoResidual"); b.maskActPerBlock = M.flag(B0 + ".maskAtomActPerBlock");
  }
  // the trunk embedder and the distogram
  const std::string TE = "trunk.embedder.";
  g::af3::TrunkEmbedder te;
  te.C = (int)M.meta(TE + "pairChannels"); te.Cs = (int)M.meta(TE + "singleChannels"); te.Cm = (int)M.meta(TE + "msaChannels");
  te.F = (int)M.meta(TE + "targetFeatWidth"); te.fromSingle = M.flag("trunk.dialect.pairInitFromSingle");
  te.msaWidth = (int)(M.at(TE + "msaActivations").length / te.Cm);
  te.pairedQuery = M.flag("trunk.dialect.msaPairedQueryRow");
  te.chai = M.flag("trunk.dialect.recycleFromInit");
  g::af3::forEachParam(te, has, load);
  g::af3::DistogramHead dh; dh.C = te.C; dh.bins = (int)M.meta("trunk.distogram.bins");
  dh.scale = M.flag("trunk.dialect.mlpDistogram") ? 0.5f : 1.f;
  if (M.has("trunk.distogram.hidden")) dh.Hd = (int)M.at("trunk.distogram.hiddenBias").length;
  g::af3::forEachParam(dh, has, load);
  // the batch
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
  // the port's trunk (pass 0), whose MSA rows both sides embed
  MSA_DRAW = "prefix";
  const float* refTf = buildTargetFeat();
  Trunk t = makeTrunk(nullptr, 64, false, refTf);
  te.pairedQuery = M.flag("trunk.dialect.msaPairedQueryRow") || (M.flag("trunk.dialect.msaPairedQueryRowAligned") && t.S > 1);
  float* zInit; CK(cudaMalloc(&zInit, nn * te.C * 4));
  embed<float>(t, [&](const char* seam, const float* p, size_t len) {
    if (!strcmp(seam, "z_init_generic")) CK(cudaMemcpyAsync(zInit, p, len * 4, cudaMemcpyDeviceToDevice, STREAM));
  });
  CK(cudaStreamSynchronize(STREAM));
  g::af3::EmbedBatch eb{Idev("batch.features.residueIndex"), Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"),
                        Idev("batch.features.entityId"), Idev("batch.features.symId"),
                        M.has("batch.bondMatrix") ? devReal(Fdev("batch.bondMatrix"), nn) : nullptr,
                        M.has("batch.bondOrderMatrix") ? devReal(Fdev("batch.bondOrderMatrix"), nn) : nullptr,
                        t.msaRows, devReal(t.deletion, (size_t)t.S * n), n, t.S};
  // chai-1's: its ESM2 rows, its declared bonds, its MSA's 41 columns (the port's chain hits)
  g::real *esm = nullptr, *tfS = nullptr; int esmE = 0, outS = 0;
  if (chai) {
    const float* ef = esmEmbeddings(n, esmE);
    esm = devReal(ef, (size_t)n * esmE);
    outS = tfm.ct.structure.N; GK(cudaMalloc(&tfS, (size_t)n * outS * sizeof(g::real)));
    if (M.has("batch.declaredBondMatrix")) eb.declaredBonds = devReal(Fdev("batch.declaredBondMatrix"), nn);
    int chains = 1; { const int* asym = M.i("batch.features.asymId"); for (int k = 0; k < n; ++k) chains = std::max(chains, asym[k] + 1); }
    const size_t rows = (size_t)t.S * n;
    int* chainHit; CK(cudaMalloc(&chainHit, (size_t)t.S * chains * sizeof(int))); CK(cudaMemset(chainHit, 0, (size_t)t.S * chains * sizeof(int)));
    chaiChainHitK<<<blocks(rows), 256, 0, STREAM>>>(t.msaRows, t.msaMask, Idev("batch.features.asymId"), chainHit, rows, n, chains);
    g::real* F; GK(cudaMalloc(&F, rows * 41 * sizeof(g::real)));
    g::af3::chaiMsaFeatK<<<g::blocks(rows), 256, 0, STREAM>>>(t.msaRows, t.deletion, t.msaMask, Idev("batch.features.asymId"), chainHit,
      chains, M.has("batch.pairedRows") ? (int)M.meta("batch.pairedRows") : 0, M.has("batch.isLigand") ? Idev("batch.isLigand") : nullptr,
      M.has("batch.msaSource") ? Idev("batch.msaSource") : nullptr, F, rows, n);
    eb.chaiMsa = F;
  }
  const int* aatype = Idev("batch.aatype");
  g::real *profile = devReal(Fdev("batch.profile"), (size_t)n * 31), *delMean = devReal(Fdev("batch.deletionMean"), n);
  g::real *tf, *pair, *single, *msa, *logits;
  GK(cudaMalloc(&tf, (size_t)n * te.F * sizeof(g::real))); GK(cudaMalloc(&pair, nn * te.C * sizeof(g::real)));
  GK(cudaMalloc(&single, (size_t)n * te.Cs * sizeof(g::real))); GK(cudaMalloc(&msa, (size_t)t.S * n * te.Cm * sizeof(g::real)));
  GK(cudaMalloc(&logits, nn * dh.bins * sizeof(g::real)));
  printf("%s (%s), %d tokens, %d MSA rows: target_feat %d wide (encoder %d blocks), pair %d, single %d, msa %d x %d%s\n",
         family.c_str(), bundle.c_str(), n, t.S, te.F, nbE, te.C, te.Cs, te.Cm, te.msaWidth, te.fromSingle ? ", pair from s_init" : "");
#ifndef GRAD_F64
  {
    g::real *pp = devReal(std::vector<double>(nn * te.C, 0)), *pss = devReal(std::vector<double>((size_t)n * te.Cs, 0));
    g::ARENA.reset();
    if (!chai && te.F != (tfm.atomOnly ? 0 : 63) + e.perToken) { fprintf(stderr, "target_feat %d wide, 63 + %d expected\n", te.F, e.perToken); return 1; }
    g::af3::targetFeatForward(tfm, b, aatype, profile, delMean, tf, esm, tfS);
    g::af3::trunkEmbedForward(te, eb, tf, pp, pss, pair, single, msa);
    // the distogram on the port's z_init (a pair any head could read)
    float* half_ = scratch<float>("test.half", nn * dh.bins); float* ref = scratch<float>("test.logits", nn * dh.bins);
    distogramHalf(zInit, half_, nn, te.C, dh.bins);
    symmetriseK<<<blocks(nn * dh.bins), 256, 0, STREAM>>>(half_, ref, n, dh.bins);
    if (dh.scale != 1) scaleK<<<blocks(nn * dh.bins), 256, 0, STREAM>>>(ref, dh.scale, nn * dh.bins);
    g::af3::distogramForward(dh, zInit, n, logits);
    CK(cudaStreamSynchronize(STREAM));
    auto cmp = [&](const char* what, const g::real* mine, const float* port, size_t len) {
      double er = g::relRms(g::host(mine, len), hostF(port, len));
      printf("  %-30s %.3e\n", what, er);
      if (!(er < 1e-4)) { printf("  <-- ABOVE 1e-4\n"); ++failed; }
    };
    cmp("target_feat", tf, refTf, (size_t)n * te.F);
    if (chai) cmp("target_feat (the structure's)", tfS, TARGET_FEAT_STRUCTURE, (size_t)n * outS);
    cmp("z_init (before templates)", pair, zInit, nn * te.C);
    cmp("single", single, t.single, (size_t)n * te.Cs);
    cmp("msa", msa, t.msa, (size_t)t.S * n * te.Cm);
    cmp("distogram logits", logits, ref, nn * dh.bins);
  }
#else
  {
    g::real *pp = devReal(gauss(nn * te.C, 71, 1)), *pss = devReal(gauss((size_t)n * te.Cs, 72, 1));
    g::real *ctP = devReal(gauss(nn * te.C, 81, 1)), *ctS = devReal(gauss((size_t)n * te.Cs, 82, 1)),
            *ctM = devReal(gauss((size_t)t.S * n * te.Cm, 83, 1)), *ctD = devReal(gauss(nn * dh.bins, 84, 1));
    g::real *dpp, *dps;
    GK(cudaMalloc(&dpp, nn * te.C * sizeof(g::real))); GK(cudaMalloc(&dps, (size_t)n * te.Cs * sizeof(g::real)));
    GK(cudaMemset(dpp, 0, nn * te.C * sizeof(g::real))); GK(cudaMemset(dps, 0, (size_t)n * te.Cs * sizeof(g::real)));
    auto fwd = [&] {
      g::ARENA.reset();
      g::af3::targetFeatForward(tfm, b, aatype, profile, delMean, tf, esm, tfS);
      g::af3::trunkEmbedForward(te, eb, tf, pp, pss, pair, single, msa);
      g::af3::distogramForward(dh, pair, n, logits);
    };
    ps.zeroGrad();
    g::ARENA.reset();
    auto ts = g::af3::targetFeatForward(tfm, b, aatype, profile, delMean, tf, esm, tfS);
    auto es = g::af3::trunkEmbedForward(te, eb, tf, pp, pss, pair, single, msa);
    g::af3::distogramForward(dh, pair, n, logits);
    g::real* dpair = g::cloneR(ctP, nn * te.C);
    g::af3::distogramBackward(dh, pair, n, ctD, dpair);
    g::real* dtf = g::ARENA.zeros((size_t)n * te.F);
    g::af3::trunkEmbedBackward(te, eb, es, tf, dpair, ctS, ctM, dtf, dpp, dps);
    g::real *ctT = chai ? devReal(gauss((size_t)n * outS, 85, 1)) : nullptr, *desm = nullptr;
    if (chai) { GK(cudaMalloc(&desm, (size_t)n * esmE * sizeof(g::real))); GK(cudaMemset(desm, 0, (size_t)n * esmE * sizeof(g::real))); }
    g::af3::targetFeatBackward(tfm, b, ts, dtf, ctT, desm);
    g::GradCheck gc;
    gc.rel = 1e-8;      // (the target_feat encoder's pair MLP: ReLU inputs near their kink at 1e-7, as the denoiser's)
    gc.add("in/prevPair", pp, dpp, nn * te.C);
    gc.add("in/prevSingle", pss, dps, (size_t)n * te.Cs);
    for (auto* p : ps.all) gc.add(*p);
    gc.output(pair, ctP, nn * te.C); gc.output(single, ctS, (size_t)n * te.Cs);
    gc.output(msa, ctM, (size_t)t.S * n * te.Cm); gc.output(logits, ctD, nn * dh.bins);
    if (chai) { gc.add("in/esm", esm, desm, (size_t)n * esmE); gc.output(tfS, ctT, (size_t)n * outS); }
    failed += gc.run(fwd);
  }
#endif
  printf("%s\n", failed ? "FAIL" : "PASS");
  return failed ? 1 : 0;
}
