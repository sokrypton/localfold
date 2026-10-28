// The trunk: the embedder, the template stack, the MSA stack, the pairformer and the distogram (cuda/af3/src/trunk.cuh
// is the reading). The pair, single and MSA stay float32; every GEMM is half with float accumulation.
#include "af3.h"
#include <cmath>
#include "msa_draw.h"
#include "constraint_tables.h"

static const int* Ib(const std::string& k) { return M.i(k); }
static const float* Fb(const std::string& k) { return M.f(k); }
static RelIdx relIdx() {
  return {Ib("batch.features.residueIndex"), Ib("batch.features.tokenIndex"), Ib("batch.features.asymId"),
          Ib("batch.features.entityId"), Ib("batch.features.symId")};
}

Trunk makeTrunk(const float* targetFeat, int msaCap) {
  Trunk t{};
  t.n = (int)M.meta("batch.tokens");
  t.C = metaI("trunk.embedder.pairChannels"); t.Cs = metaI("trunk.embedder.singleChannels");
  t.Cm = metaI("trunk.embedder.msaChannels"); t.F = metaI("trunk.embedder.targetFeatWidth");
  // S of the pool's P rows a pass: --msa=, and the featuriser's own batch.numMsa (AF3's num_msa, the page's depth)
  t.P = (int)M.meta("batch.sequences");
  t.S = std::min(t.P, msaCap);
  if (M.has("batch.numMsa")) t.S = std::min(t.S, (int)M.meta("batch.numMsa"));
  // 🔴 A FRESH DRAW OF THOSE S EVERY PASS (AF3's shuffle_msa in each pass; cuda/af3 trunk.cuh drawMsa) - not for
  // chai-1, which does not shuffle, nor a batch read from an AF3 dump (no batch.numMsa): those keep the first S rows
  t.drawMode = MSA_DRAW;
  t.drawn = t.P > t.S && M.has("batch.numMsa") && !flag("trunk.dialect.chaiMsaFeatures") && t.drawMode != "prefix";
  // ...and protenix's own: a random COUNT of the whole pool every pass, no num_msa cap (shared/af3/msa-draw.js)
  t.native = flag("trunk.dialect.msaNativeSample") && M.has("batch.numMsa") && t.drawMode != "prefix";
  if (t.native) { t.S = t.P; t.drawn = t.P > 1; }
  t.Smax = t.S;
  if (M.meta("trunk.dialect.recycleMcDropout", 0) > 0) {
    const double rate = M.meta("trunk.dialect.recycleMcDropout");
    t.mcThreshold = lf::mcDropoutThreshold(rate); t.mcScale = (float)(1.0 / (1.0 - rate));
  }
  size_t pairs = (size_t)t.n * t.n;
  size_t np = t.n <= 128 ? (size_t)((t.n + 15) & ~15) : (t.n <= 256 ? (size_t)((t.n + 31) & ~31) : (size_t)t.n);
  t.pair = allocT<float>(pairs * t.C); t.single = allocT<float>(np * t.Cs);
  t.msa = nullptr;
  if (np > (size_t)t.n) fill(t.single + (size_t)t.n * t.Cs, 0, (np - t.n) * t.Cs * 4);
  t.prevSingle = allocT<float>(np * t.Cs);
  fill(t.prevSingle, 0, np * t.Cs * 4);
  t.targetFeat = allocT<float>((size_t)t.n * t.F);
  copy(t.targetFeat, targetFeat, (size_t)t.n * t.F * 4);
  std::vector<float> seq(M.hostF("batch.seqMask"), M.hostF("batch.seqMask") + t.n), pm(pairs);
  bool ones = true;
  for (int i = 0; i < t.n; ++i) { ones &= seq[i] > 0; for (int j = 0; j < t.n; ++j) pm[(size_t)i * t.n + j] = seq[i] * seq[j]; }
  t.seqMask = uploadNew(seq.data(), t.n); t.pairMask = uploadNew(pm.data(), pairs);
  // the trunk's own S rows, widened from the pool (batch.msaPacked, msa_draw.h): the first S here, a pass's own draw
  // from drawMsa where the rows are drawn
  size_t rows = (size_t)t.S * t.n;
  t.msaRows = allocT<int>(rows); t.deletion = allocT<float>(rows); t.msaMask = allocT<float>(rows);
  t.masks = {t.pairMask, t.seqMask, ones};
  std::vector<int> first(t.S);
  for (int k = 0; k < t.S; ++k) first[k] = k;
  fillMsaRows(t, first);
  return t;
}
std::string MSA_DRAW = "pass";
// the trunk's S rows from these pool rows (upload is in stream order: the last pass has read its rows before these land)
void fillMsaRows(Trunk& t, const std::vector<int>& chosen) {
  lf::MsaPoolView pool;
  if (M.has("batch.msaPacked")) pool.packed = M.hostI("batch.msaPacked");
  else { pool.codes = M.hostI("batch.msa"); pool.deletion = M.hostF("batch.deletionMatrix"); pool.mask = M.hostF("batch.msaMask"); }
  t.S = (int)chosen.size();          // (protenix's count changes pass to pass; every other draw is S rows)
  const size_t n = (size_t)t.n, rows = (size_t)t.S * n;
  std::vector<int> codes(rows); std::vector<float> del(rows), mask(rows);
  for (int k = 0; k < t.S; ++k) lf::unpackMsaRow(pool, (size_t)chosen[k], n, codes.data() + k * n, del.data() + k * n, mask.data() + k * n);
  upload(t.msaRows, codes.data(), rows * 4); upload(t.deletion, del.data(), rows * 4); upload(t.msaMask, mask.data(), rows * 4);
  t.queryDrawn = chosen[0] == 0;
}
void drawMsa(Trunk& t) {
  if (!t.drawn) return;
  fillMsaRows(t, t.native ? lf::msaNativeRows(t.drawSeed, t.drawMode == "once" ? 0 : t.pass, t.P)
                          : lf::msaDrawRows(t.drawMode, t.drawSeed, t.pass, t.P, t.S));
}
void freeTrunk(Trunk& t) {
  for (const void* p : {(const void*)t.pair, (const void*)t.single, (const void*)t.msa, (const void*)t.targetFeat, (const void*)t.pairMask,
                        (const void*)t.seqMask, (const void*)t.msaMask, (const void*)t.prevSingle,
                        (const void*)t.msaRows, (const void*)t.deletion})
    if (p) release(p);
  t = Trunk{};
}

// ---------------------------------------------------------------- the template stack
// Each pass's input is the query term (LN(pair) projected) plus - the nine-projection embedder - its aatype one-hot
// projected along each axis and its geometry, or - the fused embedder (protenix2, boltz2, rf3) - its feature columns
// projected; then the stack's two blocks (wrapped in a residual under boltz2), the output LayerNorm, summed with the
// pass's repeat weight; the sum divided by every slot, relu, projected into the pair.
static void templateEmbedding(Trunk& t) {
  int n = t.n, Cq = t.C; size_t pairs = (size_t)n * n;
  const std::string P = "trunk.template.";
  int Ct = metaI(P + "channels");
  bool fused = flag(P + "fused");
  if (!M.has("template.passes")) die("this input was exported before template passes: export it again");
  int passes = metaI("template.passes"), templates = metaI("template.templates");
  bool anyRepeat = false;
  for (int k = 0; k < passes; ++k)
    if ((float)M.meta("template." + num(k) + ".repeat") != 0.f) { anyRepeat = true; break; }
  if (!anyRepeat) return;
  int width = (int)M.meta("template.featureWidth", 0);
  bool outer = flag("template.outerResidual");
  // The stage's room, the trunk's highest after the MSA stack's (1044 tokens: 4.19 GB, five pair-sized tensors): the
  // query's normalised input given back once it is projected, each pass's output normalised in place - 1.4 -> 0.84 GB
  // of the stage's own (docs/CUDA-TO-METAL.md, 2). (The query held in half saved 0.14 GB more and moved AF3's kitchen
  // sink 74.50 -> 75.24 pLDDT, past the gate's bar: float.)
  const bool reclaim = pairs * Cq * 4 >= (64u << 20) || getenv("LOCALFOLD_PEAK") || getenv("LOCALFOLD_SCRIBBLE");
  float* query = scratch<float>("tmpl.query", pairs * Ct);
  {
    half* qn = scratch<half>("tmpl.qn", pairs * Cq);
    ln(t.pair, qn, pairs, Cq, P + "queryEmbeddingNormScale", P + "queryEmbeddingNormOffset");
    lin(qn, P + (fused ? "zProjection" : "templatePairEmbedding8"), query, pairs, Cq, Ct);
    if (reclaim) releaseScratch({"tmpl.qn"});
  }
  float* act = scratch<float>("tmpl.act", pairs * Ct);
  float* before = outer ? scratch<float>("tmpl.before", pairs * Ct) : nullptr;
  float* summed = scratch<float>("tmpl.summed", pairs * Ct);
  fill(summed, 0, pairs * Ct * 4);
  float* oh = scratch<float>("tmpl.onehot", (size_t)n * 31);
  float* row = scratch<float>("tmpl.row", (size_t)n * Ct); float* col = scratch<float>("tmpl.col", (size_t)n * Ct);
  int nb = 0; while (M.has(P + "blocks." + num(nb) + ".pairTransition.transition1")) ++nb;
  for (int k = 0; k < passes; ++k) {
    std::string S = "template." + num(k) + ".";
    float repeat = (float)M.meta(S + "repeat");
    if (repeat == 0.f) continue;                // a slot weighed zero (boltz2's empty ones)
    if (fused) {
      int K = metaI(S + "featuresK");
      if (M.len(S + "featuresIdx") != pairs * K) die("%sfeatures: %zu entries, not %zu x %d", S.c_str(), M.len(S + "featuresIdx"), pairs, K);
      if (lenW(P + "aProjection") != (size_t)width * Ct) die("%saProjection is not %d x %d", P.c_str(), width, Ct);
      run1d("af3_template_sparse_proj", (Ct & 3) ? pairs * Ct : pairs * (Ct >> 2),
            SparseProjArgs{act, query, Ib(S + "featuresIdx"), Fb(S + "featuresVal"), Wh(P + "aProjection"),
                           Wopt(P + "templateFeatureBias"), pairs, (uint)K, (uint)width, (uint)Ct, 0});
    } else {
      run1d("af3_onehot", (size_t)n * 31, OnehotArgs{Ib(S + "aatype"), oh, (uint)n, 31});
      half* ohh = scratch<half>("tmpl.ohh", (size_t)n * 31);
      toHalf(oh, ohh, (size_t)n * 31);
      lin(ohh, P + "templatePairEmbedding2", row, n, 31, Ct);
      lin(ohh, P + "templatePairEmbedding3", col, n, 31, Ct);
      run1d("af3_add_row_col", (Ct & 3) ? pairs * Ct : pairs * (Ct >> 2),
            AddRowColArgs{act, query, row, col, Wopt(P + "templateFeatureBias"), (uint)n, (uint)Ct});
      if (M.has(S + "distogramBin")) {
        int bins = (int)(lenW(P + "templatePairEmbedding0") / Ct);
        if (M.len(S + "distogramBin") != pairs || metaI(S + "distogramBins") != bins) die("%sdistogram: the wrong shape", S.c_str());
        run1d("af3_template_geometry", (Ct & 3) ? pairs * Ct : pairs * (Ct >> 2),
              TmplGeomArgs{act, Ib(S + "distogramBin"), Fb(S + "pseudoBetaMask2d"), Fb(S + "unitVector"), Fb(S + "backboneMask2d"),
                           W(P + "templatePairEmbedding0"), W(P + "templatePairEmbedding1"), W(P + "templatePairEmbedding4"),
                           W(P + "templatePairEmbedding5"), W(P + "templatePairEmbedding6"), W(P + "templatePairEmbedding7"),
                           pairs, (uint)Ct, 0});
      }
    }
    if (outer) copy(before, act, pairs * Ct * 4);
    for (int b = 0; b < nb; ++b)
      pairUpdates(act, t.masks, n, Ct, P + "blocks." + num(b),
                  b + 1 < nb ? P + "blocks." + num(b + 1) + ".triangleMultiplicationOutgoing.leftNormInput" : "");
    if (outer) add(act, before, pairs * Ct);
    ln(act, act, pairs, Ct, P + "outputLayerNormScale", P + "outputLayerNormOffset");     // (a row read whole before it is written)
    if (flag("trunk.dialect.chaiTemplates") && M.has(S + "pseudoBetaMask2d"))      // chai: masked by its own coverage
      scaleRows(act, M.f(S + "pseudoBetaMask2d"), pairs, Ct, pairs);
    add(summed, act, pairs * Ct, repeat);
  }
  // divided by template.templates, then relu: every slot for AF3, the templates PRESENT for chai-1 and boltz2 (the
  // featuriser's divisor, cuda/featurise/af3_export.h - by four, boltz2's one template reached the trunk at a quarter)
  half* summedH = scratch<half>("tmpl.summedH", pairs * Ct);
  scale(summed, pairs * Ct, 1.f / (1e-7f + templates), true, summedH);
  const std::string nextPn = flag("trunk.dialect.msaUpdateBeforeOuterProduct") ? "trunk.msaBlocks.0.msaAttention1.pairNorm" : "";
  if (!nextPn.empty() && canEmitPairLn(Cq) && hasW(nextPn + "Scale") && lenW(P + "outputLinear") == (size_t)Ct * Cq) {
    half* lnOut = scratch<half>("mnorm.p1", pairs * Cq);
    Gemm g{}; g.X = summedH; g.tx = F16; g.W = Wh(P + "outputLinear"); g.tw = F16; g.Y = t.pair; g.rows = pairs;
    g.in = Ct; g.out = Cq; g.beta = 1.f; g.label = "template output, next norm";
    g.lnOut = lnOut; g.lnScale = W(nextPn + "Scale"); g.lnOffset = Wopt(nextPn + "Offset");
    if (gemm(g)) emitPairLn(0, t.pair, nextPn, lnOut);
  } else lin(summedH, P + "outputLinear", t.pair, pairs, Ct, Cq, 1.f);
  if (reclaim) releaseScratch({"tmpl."});
}

// ---------------------------------------------------------------- the embedder
static void embed(Trunk& t) {
  int n = t.n, C = t.C; size_t pairs = (size_t)n * n;
  size_t np = n <= 128 ? (size_t)((n + 15) & ~15) : (n <= 256 ? (size_t)((n + 31) & ~31) : (size_t)n);
  const std::string E = "trunk.embedder.";
  float* left = scratch<float>("emb.left", np * C); float* right = scratch<float>("emb.right", np * C);
  half* tfh = scratch<half>("emb.tfh", np * t.F);
  if (np > (size_t)n) fill(tfh + (size_t)n * t.F, 0, (np - n) * t.F * sizeof(half));
  toHalf(t.targetFeat, tfh, (size_t)n * t.F);
  if (flag("trunk.dialect.pairInitFromSingle")) {        // OpenDDE: the pair from s_init
    lin(tfh, E + "singleActivations", t.single, np, t.F, t.Cs);
    lin(t.single, E + "leftSingle", left, n, t.Cs, C);
    lin(t.single, E + "rightSingle", right, n, t.Cs, C);
  } else {
    lin(tfh, E + "leftSingle", left, np, t.F, C);
    lin(tfh, E + "rightSingle", right, np, t.F, C);
  }
  // pair = left[i] + right[j] + prevEmbedding(LN(prev pair)) - not zero on the first pass. chai-1's first pass recycles
  // z_init itself (left + right + its relative encoding)
  const bool chai = flag("trunk.dialect.recycleFromInit");
  // (protenix's MC dropout on the recycled term, element by element inside af3_outer_sum: this pass's key)
  const uint dropKey = t.mcOn ? lf::mcDropoutKey(t.drawSeed, t.pass) : 0u;
  auto chaiRelEnc = [&]() {
    run1d("af3_chai_relenc", (C & 3) ? pairs * C : pairs * (C >> 2),
          ChaiRelEncArgs{relIdx(), W(E + "positionActivations"), W(E + "positionActivationsBias"),
                         hasW(E + "chaiChainEntity") ? W(E + "chaiChainEntity") : nullptr,
                         M.has("batch.declaredBondMatrix") ? M.f("batch.declaredBondMatrix") : nullptr,
                         hasW(E + "chaiBondTrunk") ? W(E + "chaiBondTrunk") : nullptr, t.pair, (uint)n, (uint)C});
    // ...and a job's pocket and contact constraints, through the trunk half of chai's restraint columns
    addChai1Constraints(t.pair, pairs, 0, C, C, 0, "chai.trunk", E + "chaiRestraintTrunk", 0);
  };
  if (chai && t.pass == 0) {
    half* pn = scratch<half>("emb.prevln", pairs * C);
    run1d("af3_outer_sum", (C & 3) ? pairs * C : pairs * (C >> 2), OuterSumArgs{left, right, nullptr, t.pair, (uint)n, (uint)C, 0, 0});
    chaiRelEnc();
    ln(t.pair, pn, pairs, C, E + "prevEmbeddingNormScale", E + "prevEmbeddingNormOffset");
    lin(pn, E + "prevEmbedding", t.pair, pairs, C, C, 1.f);
  } else if (t.pass == 0) {
    // (before the first pass the recycled pair is zero on every row: project 16 zero rows once and add row 0 in af3_outer_sum)
    float* z0 = scratch<float>("emb.z0", (size_t)16 * C);
    half* pn0 = scratch<half>("emb.prevln0", (size_t)16 * C);
    float* prev0 = scratch<float>("emb.prev0", (size_t)16 * C);
    fill(z0, 0, (size_t)16 * C * 4);
    ln(z0, pn0, 16, C, E + "prevEmbeddingNormScale", E + "prevEmbeddingNormOffset");
    lin(pn0, E + "prevEmbedding", prev0, 16, C, C);
    run1d("af3_outer_sum", (C & 3) ? pairs * C : pairs * (C >> 2), OuterSumArgs{left, right, prev0, t.pair, (uint)n, (uint)C, 1u, 0,
                                                                                dropKey, t.mcThreshold, t.mcScale, t.mcOn ? 1u : 0u});
  } else {
    // (the recycled pair is t.pair itself: its term projected back over it - a row reads only
    // its own normalised row - and the outer sum added in place; a copy of it and an f32 term were two pairs more)
    half* pn = scratch<half>("emb.prevln", pairs * C);
    ln(t.pair, pn, pairs, C, E + "prevEmbeddingNormScale", E + "prevEmbeddingNormOffset");
    lin(pn, E + "prevEmbedding", t.pair, pairs, C, C);
    run1d("af3_outer_sum", (C & 3) ? pairs * C : pairs * (C >> 2), OuterSumArgs{left, right, t.pair, t.pair, (uint)n, (uint)C, 0, 0,
                                                                                dropKey, t.mcThreshold, t.mcScale, t.mcOn ? 1u : 0u});
  }
  if (chai) { if (t.pass > 0) chaiRelEnc(); }      // (the first pass added it before the recycle term)
  else run1d("af3_relenc", (C & 3) ? pairs * C : pairs * (C >> 2), RelEncArgs{relIdx(), W(E + "positionActivations"), t.pair, (uint)n, (uint)C});
  const bool bonds = M.has("batch.bondMatrix") && hasW(E + "bondEmbedding"), types = hasW(E + "tokenBondsTypeEmbed");
  if (bonds || types)
    run1d("af3_bond_embed", (C & 3) ? pairs * C : pairs * (C >> 2),
          BondEmbedArgs{t.pair, bonds ? Fb("batch.bondMatrix") : nullptr, bonds ? W(E + "bondEmbedding") : nullptr,
                        types && M.has("batch.bondOrderMatrix") ? Fb("batch.bondOrderMatrix") : nullptr,
                        types ? W(E + "tokenBondsTypeEmbed") : nullptr, types ? W(E + "contactEncodingUnspecified") : nullptr,
                        pairs, (uint)C, 0});
  // ...and boltz2's: a job's pocket and contact constraints, each pair's term less the unspecified constant just added
  if (types) addBoltz2Constraints(t.pair, pairs, C, E);
  releaseScratch({"emb.prevln"});
  auto buildSingle = [&]() {      // the single: target_feat projected, plus the recycled single's (chai's first: s_init's)
    if (!flag("trunk.dialect.pairInitFromSingle")) lin(tfh, E + "singleActivations", t.single, np, t.F, t.Cs);
    half* sln = scratch<half>("emb.prevsln", np * t.Cs);
    if (np > (size_t)n) fill(sln + (size_t)n * t.Cs, 0, (np - n) * t.Cs * sizeof(half));
    ln(chai && t.pass == 0 ? t.single : t.prevSingle, sln, n, t.Cs, E + "prevSingleEmbeddingNormScale", E + "prevSingleEmbeddingNormOffset");
    // (the first block's norms: Mol-1's first is its PairMixer's)
    const std::string B0 = M.has("trunk.pairMixerBlocks.0.singleChannels") ? "trunk.pairMixerBlocks.0" : "trunk.pairformerBlocks.0";
    const std::string sn1 = B0 + ".singleAttention.layerNorm";
    const std::string sn2 = flag("trunk.dialect.parallelPairformer") ? B0 + ".singleTransition.inputLayerNorm" : "";
    if (canEmitPairLn(t.Cs, n) && hasW(sn1 + "Scale") && lenW(E + "prevSingleEmbedding") == (size_t)t.Cs * t.Cs) {
      half* nrm = scratch<half>("trunk.sn1", np * t.Cs);
      half* txn = (!sn2.empty() && hasW(sn2 + "Scale")) ? scratch<half>("trunk.sn2", np * t.Cs) : nullptr;
      if (np > (size_t)n) {
        fill(nrm + (size_t)n * t.Cs, 0, (np - n) * t.Cs * sizeof(half));
        if (txn) fill(txn + (size_t)n * t.Cs, 0, (np - n) * t.Cs * sizeof(half));
      }
      Gemm g{}; g.X = sln; g.tx = F16; g.W = Wh(E + "prevSingleEmbedding"); g.tw = F16; g.Y = t.single; g.rows = n;
      g.in = t.Cs; g.out = t.Cs; g.beta = 1.f; g.label = "prevSingleEmbedding, next norm";
      g.lnOut = nrm; g.lnScale = W(sn1 + "Scale"); g.lnOffset = Wopt(sn1 + "Offset");
      if (txn) { g.lnOut2 = txn; g.lnScale2 = W(sn2 + "Scale"); g.lnOffset2 = Wopt(sn2 + "Offset"); }
      if (gemm(g)) {
        emitPairLn(2, t.single, sn1, nrm);
        if (txn) emitPairLn(3, t.single, sn2, txn);
      }
    } else lin(sln, E + "prevSingleEmbedding", t.single, np, t.Cs, t.Cs, 1.f);
  };
  if (!t.msa) t.msa = allocT<float>((size_t)t.Smax * n * t.Cm);     // (the most rows any pass reads: protenix's vary)
  if (flag("trunk.dialect.chaiMsaFeatures")) {
    // chai-1: the single first, then the MSA's 41 features through a biased linear plus the single's projection
    buildSingle();
    float* fromSingle = scratch<float>("emb.fromTarget", (size_t)n * t.Cm);
    lin(t.single, E + "extraMsaTargetFeat", fromSingle, n, t.Cs, t.Cm);
    size_t rows = (size_t)t.S * n;
    if (lenW(E + "msaActivations") != (size_t)41 * t.Cm) die("chai's msa features are not 41 wide");
    int chains = 1;
    { const int* asym = M.hostI("batch.features.asymId"); for (int k = 0; k < n; ++k) chains = std::max(chains, asym[k] + 1); }
    int* chainHit = scratch<int>("emb.chainHit", (size_t)t.S * chains);
    fill(chainHit, 0, (size_t)t.S * chains * sizeof(int));
    run1d("af3_chai_chain_hit", rows, ChaiChainHitArgs{t.msaRows, t.msaMask, M.i("batch.features.asymId"), chainHit, rows, (uint)n, (uint)chains});
    // (the paired block's row count: a batch without it - a dump's - has none)
    const uint pairedRows = M.has("batch.pairedRows") ? (uint)M.meta("batch.pairedRows") : 0u;
    run1d("af3_chai_msa_embed", (t.Cm & 3) ? rows * t.Cm : rows * (t.Cm >> 2),
          ChaiMsaEmbedArgs{t.msaRows, t.deletion, t.msaMask, M.i("batch.features.asymId"), M.has("batch.isLigand") ? M.i("batch.isLigand") : nullptr,
                           chainHit, W(E + "msaActivations"), W(E + "msaActivationsBias"), fromSingle, t.msa,
                           M.has("batch.msaSource") ? M.i("batch.msaSource") : nullptr, rows, (uint)n, (uint)t.Cm,
                           (uint)chains, pairedRows});
  } else {
    // the MSA: its features projected, plus the target's projection broadcast over the rows
    float* fromTarget = scratch<float>("emb.fromTarget", np * t.Cm);
    lin(tfh, E + "extraMsaTargetFeat", fromTarget, np, t.F, t.Cm);
    size_t rows = (size_t)t.S * n;
    int msaWidth = (int)(lenW(E + "msaActivations") / t.Cm);
    if (msaWidth != 34 && msaWidth != 35) die("msa feature width %d", msaWidth);
    run1d("af3_msa_embed", (t.Cm & 3) ? rows * t.Cm : rows * (t.Cm >> 2),
          MsaEmbedArgs{t.msaRows, t.deletion, W(E + "msaActivations"), fromTarget, t.msa,
                       hasW(E + "boltz2DnMsa") ? W(E + "boltz2DnMsa") : nullptr,
                       M.has("batch.isDna") ? M.i("batch.isDna") : nullptr, rows, (uint)n,
                       (uint)t.Cm, (uint)msaWidth,
                       (flag("trunk.dialect.msaPairedQueryRow")
                         || (flag("trunk.dialect.msaPairedQueryRowAligned") && t.S > 1)) && t.queryDrawn ? 1u : 0u});
    buildSingle();
  }
  if (t.pass == 0) seam("z_init_generic", t.pair, pairs * C);
  templateEmbedding(t);
  if (t.pass == 0) seam("z_after_template", t.pair, pairs * C);
  ++t.pass;
}

// ---------------------------------------------------------------- the MSA stack
// the outer product mean: pair[i][j] += (sum_s l[s][i] (x) r[s][j] W + b) / (1e-3 + norm[i][j])
static void outerProductMean(Trunk& t, const std::string& pre, const std::string& nextNorm = "", const std::string& nextNorm2 = "",
                             const std::string& dualMsaNorm = "", bool computeMaskNorm = true) {
  int L = t.n, S = t.S, Cm = t.Cm, C = t.C;
  int O = metaI(pre + ".outerChannels");
  size_t rows = (size_t)S * L, pairs = (size_t)L * L;
  half* xn = takePairLn(t.msa, pre + ".layerNormInput");
  if (!xn) {
    xn = scratch<half>("opm.xn", rows * Cm);
    if (!dualMsaNorm.empty() && hasW(dualMsaNorm + "Scale")) {
      half* xn2 = scratch<half>("mnorm.m2", rows * Cm);
      layerNormDual(t.msa, xn, W(pre + ".layerNormInputScale"), Wopt(pre + ".layerNormInputOffset"),
                    xn2, W(dualMsaNorm + "Scale"), Wopt(dualMsaNorm + "Offset"), rows, Cm);
      emitPairLn(5, t.msa, dualMsaNorm, xn2);
    } else {
      ln(t.msa, xn, rows, Cm, pre + ".layerNormInputScale", pre + ".layerNormInputOffset");
    }
  }
  half* lt = scratch<half>("opm.left", rows * O); half* rt = scratch<half>("opm.right", rows * O);
  linH(xn, pre + ".leftProjection", lt, rows, Cm, O, Wopt(pre + ".leftProjectionBias"));      // (rf3's biases, before the mask)
  linH(xn, pre + ".rightProjection", rt, rows, Cm, O, Wopt(pre + ".rightProjectionBias"));
  // 🔴 PAST 1024 ROWS THE SUM BELOW IS HELD AT 1024 ROWS' MAGNITUDE (cuda/af3's outerProductMean, where protenix2's
  // 128-channel MSA overflowed float16 at ~4,700 rows of 8S83 - one token's sums inf, its pair row NaN): the left operand
  // scaled down by a power of two, the projection's alpha back up. Exact, and nothing at or under 1024 rows moves.
  const int shift = S > 1024 ? (int)std::ceil(std::log2((double)S / 1024)) : 0;
  const float up = std::ldexp(1.f, shift);
  run1d("af3_scale_rows_h", (O & 3) ? rows * O : rows * (O >> 2), ScaleRowsHArgs{lt, rt, t.msaMask, rows, (uint)O, (uint)shift});
  float* norm = scratch<float>("mnorm.opm", pairs);
  if (computeMaskNorm) run1d("af3_mask_norm", pairs, MaskNormArgs{t.msaMask, norm, (uint)S, (uint)L});
  const half* Wout = Wh(pre + ".outputW");          // [O * O][C]
  const uint after = flag("trunk.dialect.opmBiasAfterNorm") ? 1u : 0u;
  static const bool lnEmit = !getenv("LOCALFOLD_LN_EMIT") || atoi(getenv("LOCALFOLD_LN_EMIT")) != 0;
  const bool opmEmit = S <= 128 ? (C == 128 || C == 256) : canEmitPairLn(C);
  half* lnOut = (opmEmit && !nextNorm.empty() && lnEmit && matrixUnits()) ? scratch<half>("mnorm.p1", pairs * C) : nullptr;
  const float* lnScale = lnOut ? W(nextNorm + "Scale") : nullptr;
  const float* lnOffset = lnOut ? Wopt(nextNorm + "Offset") : nullptr;
  half* lnOut2 = (lnOut && !nextNorm2.empty()) ? scratch<half>("mnorm.p2", pairs * C) : nullptr;
  const float* lnScale2 = lnOut2 ? W(nextNorm2 + "Scale") : nullptr;
  const float* lnOffset2 = lnOut2 ? Wopt(nextNorm2 + "Offset") : nullptr;
  if (S <= 128) {
    // the SHALLOW form: the output projection folded into the right operand first - T[c][(s, j)][f] = sum_e r[s][j][e]
    // W[c O + e][f], a GEMM batched over c - then Y[i][(j, f)] = sum_{c, s} l[s][i][c] T[(c, s)][(j, f)], K = O S
    half* T = scratch<half>("opm.T", (size_t)O * rows * C);
    { Gemm g{}; g.X = rt; g.tx = F16; g.W = Wout; g.tw = F16; g.sw = (int64_t)O * C; g.Y = T; g.ty = F16; g.sy = (int64_t)rows * C;
      g.rows = rows; g.in = O; g.out = C; g.batch = O; g.accFloat = true; g.label = "opm shallow T"; gemm(g); }
    half* Lt = scratch<half>("opm.Lt", rows * O);
    run1d("af3_opm_left", rows * O, OpmLeftArgs{lt, Lt, (uint)S, (uint)L, (uint)O, 1.f});
    int K = O * S;
    int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / ((size_t)L * C)));
    float* Y = nullptr;
    bool allEmitted = lnOut != nullptr;
    for (int i0 = 0; i0 < L; i0 += Bi) {
      int bi = std::min(Bi, L - i0);
      half* lnChunk = lnOut ? lnOut + (size_t)i0 * L * C : nullptr;
      half* lnChunk2 = lnOut2 ? lnOut2 + (size_t)i0 * L * C : nullptr;
      if (C % 128 == 0 && C <= 512 &&
          gemmOpmOut(Lt + (size_t)i0 * K, T, t.pair + (size_t)i0 * L * C, W(pre + ".outputB"),
                     norm + (size_t)i0 * L, bi, K, L * C, 1.f, lnChunk, lnScale, lnOffset, 1e-5f,
                     lnChunk2, lnScale2, lnOffset2, after != 0, C))
        continue;
      allEmitted = false;
      if (!Y) Y = scratch<float>("opm.Y", (size_t)Bi * L * C);
      Gemm g{}; g.X = Lt + (size_t)i0 * K; g.tx = F16; g.W = T; g.tw = F16; g.Y = Y; g.rows = bi; g.in = K; g.out = L * C; g.label = "opm shallow";
      gemm(g);
      run1d("af3_opm_add", (size_t)bi * L * C, OpmAddArgs{t.pair, Y, W(pre + ".outputB"), norm, (u64)i0, (uint)bi, (uint)L, (uint)C, after});
    }
    if (allEmitted) {
      emitPairLn(0, t.pair, nextNorm, lnOut);
      if (lnOut2) emitPairLn(1, t.pair, nextNorm2, lnOut2);
    }
    return;
  }
  // the standard form: P[(i, c)][(j, e)] = sum_s l[s][i][c] r[s][j][e] a block of rows i at a time, permuted to
  // [(i, j)][(c, e)], then the output projection
  // 🔴 STORED IN HALF UNSCALED, AND MEASURED SAFE - unlike AF2's (metal/af2's outerProductMean, cuda/af2's a04ed7e), which
  // overflowed past ~2,000 extra rows. Scaling this sum by 2^-k changes nothing until it overflows, so a fold that differs
  // between k = 0 and a small k is the overflow's mark: boltz2 on tools/fixtures/test.a3m at all 8,076 rows, and at 16,151
  // (the alignment doubled, every row sum twice as large), reads byte for byte the same at k = 0, 1, 2 and 3; protenix2
  // too at 8,076. AF2's own k (~rows / 64, 7 there) instead MOVED the fold - small entries pushed into float16's
  // subnormals - so the AF2 fix copied here would cost precision for nothing.
  // 🔴 ...BUT THAT WAS ONE 59-RESIDUE ALIGNMENT, AND 8S83 OVERFLOWED: cuda/af3's protenix2 went NaN at ~4,700 rows there
  // (2026-10-27), once protenix's native draw read past 1024 rows. So past 1024 rows only, and by 2^k with k at most 4 -
  // the k = 1-3 that read byte for byte the same above - the sum is held at 1024 rows' magnitude (the shift above).
  size_t per = (size_t)L * O * O;
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / per));
  half* X = scratch<half>("opm.X", (size_t)Bi * per);
  float* Y = nullptr;
  bool allEmitted = lnOut != nullptr;
  for (int i0 = 0; i0 < L; i0 += Bi) {
    int bi = std::min(Bi, L - i0);
    // (on the matrix units the product's epilogue stores it permuted - no Pm, no permute pass; core.h)
    if (!gemmOpmPermuted(lt + (size_t)i0 * O, L * O, rt, L * O, X, bi, L, O, S)) {
      half* Pm = scratch<half>("opm.P", (size_t)Bi * per);
      { Gemm g{}; g.X = lt + (size_t)i0 * O; g.tx = F16; g.transX = true; g.ldx = L * O; g.W = rt; g.tw = F16; g.ldw = L * O; g.Y = Pm;
        g.ty = F16; g.rows = (size_t)bi * O; g.in = S; g.out = L * O; g.accFloat = true; g.label = "opm product"; gemm(g); }
      run1d("af3_opm_permute", (size_t)bi * L * O * (O / 8), OpmPermuteArgs{Pm, X, (uint)bi, (uint)L, (uint)O, 0});
    }
    half* lnChunk = lnOut ? lnOut + (size_t)i0 * L * C : nullptr;
    half* lnChunk2 = lnOut2 ? lnOut2 + (size_t)i0 * L * C : nullptr;
    if (C % 128 == 0 && C <= 512 &&
        gemmOpmOut(X, Wout, t.pair + (size_t)i0 * L * C, W(pre + ".outputB"),
                   norm + (size_t)i0 * L, (size_t)bi * L, O * O, C, up, lnChunk, lnScale, lnOffset, 1e-5f,
                   lnChunk2, lnScale2, lnOffset2, after != 0, C)) {
      continue;
    }
    allEmitted = false;
    if (!Y) Y = scratch<float>("opm.Y", (size_t)Bi * L * C);
    { Gemm g{}; g.X = X; g.tx = F16; g.W = Wout; g.tw = F16; g.Y = Y; g.rows = (size_t)bi * L; g.in = O * O; g.out = C; g.alpha = up; g.label = "opm output"; gemm(g); }
    run1d("af3_opm_add", (size_t)bi * L * C, OpmAddArgs{t.pair, Y, W(pre + ".outputB"), norm, (u64)i0, (uint)bi, (uint)L, (uint)C, after});
  }
  if (allEmitted) {
    emitPairLn(0, t.pair, nextNorm, lnOut);
    if (lnOut2) emitPairLn(1, t.pair, nextNorm2, lnOut2);
  }
}
// AF3's MSA "attention": per head, weights softmax_j over LN(pair) projected (no queries or keys), averaging every
// row's values; gated and projected back into the MSA
// `parallelTransition`: chai-1's MSA transition, which reads the same MSA as the weighted averaging (chai-lab's
// msa + transition(msa) + pwa(msa); af3-any-model 7cc3165) - in place once the whole MSA's norm is taken, before any of
// the averaging is added. Its scratch is msatr.*, the msa family's (main.mm), so it never shares a slot with msaatt.ln.
static void msaAttention(Trunk& t, const std::string& pre, const std::string& parallelTransition = "",
                         const std::string& nextPairNorm = "", const std::string& nextMsaNorm = "", bool preNormed = false,
                         bool computeKeyMask = true) {
  int n = t.n, S = t.S, Cm = t.Cm, C = t.C;
  int heads = metaI(pre + ".heads"), d = metaI(pre + ".dimension"), Wd = heads * d;
  size_t rows = (size_t)S * n, pairs = (size_t)n * n;
  half* xn = takePairLn(t.msa, pre + ".actNorm");
  if (!xn) {
    xn = scratch<half>("msaatt.ln", rows * Cm);
    if (!preNormed && !parallelTransition.empty()) {
      half* txn = scratch<half>("msatr.xn", rows * Cm);
      layerNormDual(t.msa, xn, W(pre + ".actNormScale"), Wopt(pre + ".actNormOffset"),
                    txn, W(parallelTransition + ".inputLayerNormScale"), Wopt(parallelTransition + ".inputLayerNormOffset"), rows, Cm);
      emitPairLn(4, t.msa, parallelTransition + ".inputLayerNorm", txn);
    } else if (!preNormed) ln(t.msa, xn, rows, Cm, pre + ".actNormScale", pre + ".actNormOffset");
  }
  if (!parallelTransition.empty()) transition(t.msa, rows, Cm, parallelTransition, nullptr, "", "", "msatr.");
  half* pln = pairLn(t.pair, pairs, C, pre + ".pairNorm", nextPairNorm);
  float* flat = scratch<float>("msaatt.flat", pairs * heads);
  lin(pln, pre + ".pairLogits", flat, pairs, C, heads);
  // chai: the logits masked by the TOKEN mask, not by the alignment's coverage; the values zeroed instead
  const bool chaiMask = flag("trunk.dialect.chaiMsaFeatures");
  const float* keyMask = t.seqMask;
  if (!chaiMask) {
    float* km = scratch<float>("mnorm.keymask", n);
    if (computeKeyMask) run1d("af3_key_mask", n, KeyMaskArgs{t.msaMask, km, (uint)S, (uint)n});
    keyMask = km;
  }
  int ldw = (n + 7) / 8 * 8;
  half* w = scratch<half>("msaatt.w", (size_t)heads * n * ldw);
  run("af3_msa_weights", Grid{(uint32_t)(heads * n), 1, 1}, 256, MsaWeightsArgs{flat, keyMask, w, (uint)n, (uint)heads, (uint)ldw, 0});
  half* v = scratch<half>("msaatt.v", rows * Wd);
  linH(xn, pre + ".vProjection", v, rows, Cm, Wd);
  half* vh = scratch<half>("msaatt.vh", (size_t)S * ldw * Wd);
  const bool v8 = d % 8 == 0 && (size_t)S * ldw * Wd < ((size_t)1 << 32);     // (16 bytes a thread)
  const uint d8 = (uint)(d / 8);
  const bool tiled8 = v8 && d8 >= 1 && (16 % d8) == 0 && (d8 & (d8 - 1)) == 0 &&
                      ((size_t)ldw * heads * d8 + 15) / 16 <= 65535 && ((size_t)S + 15) / 16 <= 65535;
  const float* vMask = chaiMask ? t.msaMask : nullptr;
  if (tiled8) {
    run("af3_msa_v_heads8_tiled", Grid{(uint32_t)(((size_t)ldw * heads * d8 + 15) / 16), (uint32_t)((S + 15) / 16), 1}, 256,
        MsaVHeadsArgs{v, vh, vMask, (uint)S, (uint)n, (uint)heads, (uint)d, (uint)ldw, 0});
  } else if (v8) run1d("af3_msa_v_heads8", (size_t)S * ldw * Wd / 8, MsaVHeadsArgs{v, vh, vMask, (uint)S, (uint)n, (uint)heads, (uint)d, (uint)ldw, 0});
  else run1d("af3_msa_v_heads", (size_t)S * ldw * Wd, MsaVHeadsArgs{v, vh, vMask, (uint)S, (uint)n, (uint)heads, (uint)d, (uint)ldw, 0});
  // per head: O_h [n][S d] = W_h [n][ldw] V_h [ldw][S d]
  half* oh = scratch<half>("msaatt.oh", rows * Wd);
  { Gemm g{}; g.X = w; g.tx = F16; g.sx = (int64_t)n * ldw; g.W = vh; g.tw = F16; g.sw = (int64_t)ldw * S * d; g.Y = oh; g.ty = F16;
    g.sy = (int64_t)n * S * d; g.rows = n; g.in = ldw; g.out = S * d; g.batch = heads; g.accFloat = true; g.label = "msa average"; gemm(g); }
  half* gate = scratch<half>("msaatt.gate", rows * Wd);
  linH(xn, pre + ".gatingQuery", gate, rows, Cm, Wd);
  half* gated = scratch<half>("msaatt.gated", rows * Wd);
  if (tiled8) {
    run("af3_msa_from_heads8_tiled", Grid{(uint32_t)(((size_t)n * heads * d8 + 15) / 16), (uint32_t)((S + 15) / 16), 1}, 256,
        MsaFromHeadsArgs{oh, gate, gated, (uint)S, (uint)n, (uint)heads, (uint)d});
  } else if (v8) run1d("af3_msa_from_heads8", rows * Wd / 8, MsaFromHeadsArgs{oh, gate, gated, (uint)S, (uint)n, (uint)heads, (uint)d});
  else run1d("af3_msa_from_heads", rows * Wd, MsaFromHeadsArgs{oh, gate, gated, (uint)S, (uint)n, (uint)heads, (uint)d});
  if (!nextMsaNorm.empty() && (Cm == 64 || Cm == 128) && hasW(nextMsaNorm + "Scale") && lenW(pre + ".outputProjection") == (size_t)Wd * Cm) {
    half* msaLnOut = scratch<half>("msaatt.ln", rows * Cm);
    Gemm g{}; g.X = gated; g.tx = F16; g.W = Wh(pre + ".outputProjection"); g.tw = F16; g.Y = t.msa; g.rows = rows;
    g.in = Wd; g.out = Cm; g.beta = 1.f; g.label = "msa output, next norm";
    g.lnOut = msaLnOut; g.lnScale = W(nextMsaNorm + "Scale"); g.lnOffset = Wopt(nextMsaNorm + "Offset");
    if (gemm(g)) emitPairLn(4, t.msa, nextMsaNorm, msaLnOut);
    return;
  }
  lin(gated, pre + ".outputProjection", t.msa, rows, Wd, Cm, 1.f);
}
// chai-1's GROUPED outer product: L, R = LN(m) Wl, Wr reshaped [S][n][G][K], masked; P[i,j,g,k,l] = sum_s L[s,i,g,k]
// R[s,j,g,l] - NOT divided by any count; P = LN(P; eps 0.1) over its G K K; pair += P Wout + b. float32 throughout: the
// sum grows with the alignment's depth before the norm
static void linF(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, const float* bias = nullptr, float beta = 0.f) {
  if (lenW(w) != (size_t)in * out) die("%s has %zu elements, not %d x %d", w.c_str(), lenW(w), in, out);
  Gemm g{}; g.X = X; g.W = W(w); g.Y = Y; g.rows = rows; g.in = in; g.out = out; g.bias = bias; g.beta = beta; g.label = w.c_str();
  gemm(g);
}
static void groupedOuterProduct(Trunk& t, const std::string& pre) {
  int n = t.n, S = t.S, Cm = t.Cm, C = t.C;
  int O = metaI(pre + ".outerChannels"), G = metaI(pre + ".groups"), K = O / G, per = G * K * K;
  size_t rows = (size_t)S * n;
  float* lnM = scratch<float>("gopm.ln", rows * Cm);
  ln(t.msa, lnM, rows, Cm, pre + ".layerNormInputScale", pre + ".layerNormInputOffset");
  float* L = scratch<float>("gopm.left", rows * O); float* R = scratch<float>("gopm.right", rows * O);
  linF(lnM, pre + ".leftProjection", L, rows, Cm, O);
  linF(lnM, pre + ".rightProjection", R, rows, Cm, O);
  float* Lg = scratch<float>("gopm.lg", rows * O); float* Rg = scratch<float>("gopm.rg", rows * O);
  if (G == 8 && K == 8 && (rows + 31) / 32 <= 65535) {
    run("af3_group_major8_tiled", Grid{(uint32_t)((rows + 31) / 32), 2, 1}, 256,
        GroupMajorArgs{L, R, t.msaMask, Lg, Rg, (uint)S, (uint)n, (uint)G, (uint)K});
  } else {
    size_t gmThreads = (K & 7) == 0 ? rows * (O >> 3) : ((K & 3) == 0 ? rows * (O >> 2) : rows * O);
    run1d("af3_group_major", gmThreads, GroupMajorArgs{L, R, t.msaMask, Lg, Rg, (uint)S, (uint)n, (uint)G, (uint)K});
  }
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(n, ((size_t)32 << 20) / ((size_t)n * per * 2)));
  float* Pm = scratch<float>("gopm.P", (size_t)Bi * n * per); float* Pp = scratch<float>("gopm.Pp", (size_t)Bi * n * per);
  for (int i0 = 0; i0 < n; i0 += Bi) {
    int bi = std::min(Bi, n - i0);
    // per group: P_g [(i, k)][(j, l)] = sum_s Lg[s][(i, k)] Rg[s][(j, l)]
    { Gemm g{}; g.X = Lg + (size_t)i0 * K; g.transX = true; g.ldx = n * K; g.sx = (int64_t)S * n * K; g.W = Rg; g.ldw = n * K;
      g.sw = (int64_t)S * n * K; g.Y = Pm; g.sy = (int64_t)bi * K * n * K; g.rows = (size_t)bi * K; g.in = S; g.out = n * K; g.batch = G;
      g.label = "grouped opm"; gemm(g); }
    size_t pRows = (size_t)bi * n, sgs = pRows <= 256 ? 2 : 8, groups = (pRows + sgs - 1) / sgs;
    Grid gGrid = groups <= 65535 ? Grid{(uint32_t)groups, 1, 1} : Grid{65535, (uint32_t)((groups + 65534) / 65535), 1};
    run("af3_grouped_permute_ln", gGrid, (uint32_t)(sgs * 32),
        GroupedPermuteLnArgs{Pm, Pp, nullptr, W(pre + ".productNormScale"), W(pre + ".productNormOffset"), (uint)bi, (uint)n, (uint)G, (uint)K, 0.1f, 0});
    linF(Pp, pre + ".outputW", t.pair + (size_t)i0 * n * C, pRows, per, C, W(pre + ".outputB"), 1.f);
  }
}
static void msaBlock(Trunk& t, int k, int msaBlocks) {
  std::string B = "trunk.msaBlocks." + num(k);
  if (flag("trunk.dialect.groupedOuterProduct")) {
    // chai-1: z += grouped OPM(m); m += attention(m, z) + transition(m), both off the same m; then the pair track in two
    // parallel stages
    groupedOuterProduct(t, B + ".outerProductMean");
    msaAttention(t, B + ".msaAttention1", B + ".msaTransition", "__xhat__", "", false, k == 0);
    parallelPairUpdates(t.pair, t.masks, t.n, t.C, B, "oit", true);
    parallelPairUpdates(t.pair, t.masks, t.n, t.C, B, "rc", false);
    return;
  }
  bool updateFirst = flag("trunk.dialect.msaUpdateBeforeOuterProduct");     // (OpenDDE, boltz2)
  if (!updateFirst)
    outerProductMean(t, B + ".outerProductMean", B + ".msaAttention1.pairNorm",
                     B + ".triangleMultiplicationOutgoing.leftNormInput", B + ".msaAttention1.actNorm", k == 0);
  msaAttention(t, B + ".msaAttention1", "", !updateFirst ? B + ".triangleMultiplicationOutgoing.leftNormInput" : "",
               B + ".msaTransition.inputLayerNorm", !updateFirst || k > 0, k == 0);
  const std::string nextMsa1 = updateFirst ? B + ".outerProductMean.layerNormInput"
                                           : ((k + 1 < msaBlocks) ? "trunk.msaBlocks." + num(k + 1) + ".outerProductMean.layerNormInput" : "");
  const std::string nextMsa2 = (k + 1 < msaBlocks) ? "trunk.msaBlocks." + num(k + 1) + ".msaAttention1.actNorm" : "";
  transition(t.msa, (size_t)t.S * t.n, t.Cm, B + ".msaTransition", nullptr, nextMsa1, nextMsa2, "msatr.");
  if (updateFirst)
    outerProductMean(t, B + ".outerProductMean", B + ".triangleMultiplicationOutgoing.leftNormInput", "",
                     (k + 1 < msaBlocks) ? "trunk.msaBlocks." + num(k + 1) + ".msaAttention1.actNorm" : "", k == 0);
  pairUpdates(t.pair, t.masks, t.n, t.C, B,
              (k + 1 < msaBlocks && updateFirst) ? "trunk.msaBlocks." + num(k + 1) + ".msaAttention1.pairNorm" : "");
}

// a half tensor's seam (LOCALFOLD_SAVE_SEAMS): through a float copy, only when one is taken
static void seamHalf(const char* name, const half* h, size_t n) {
  if (!SEAM) return;
  float* f = allocT<float>(n);
  toFloat(h, f, n);
  seam(name, f, n);
  release(f);
}

// ---------------------------------------------------------------- one pass
void runTrunk(Trunk& t) {
  size_t pairs = (size_t)t.n * t.n;
  const bool reclaim = pairs * t.C * 4 >= (64u << 20) || getenv("LOCALFOLD_PEAK") || getenv("LOCALFOLD_SCRIBBLE");
  // the recycled state: the last pass's pair and single (zero before the first)
  if (t.pass > 0) copy(t.prevSingle, t.single, (size_t)t.n * t.Cs * 4);     // (the pair is recycled in place: embed())
  embed(t);
  if (reclaim) releaseScratch({"emb."});
  int msaBlocks = 0; while (M.has("trunk.msaBlocks." + num(msaBlocks) + ".pairChannels")) ++msaBlocks;
  float* zIn = nullptr;
  if (flag("trunk.dialect.msaDoubleAddPair")) {     // boltz2 adds the pre-MSA pair back
    zIn = scratch<float>("trunk.zBeforeMsa", pairs * t.C);
    copy(zIn, t.pair, pairs * t.C * 4);
  }
  for (int k = 0; k < msaBlocks; ++k) msaBlock(t, k, msaBlocks);
  if (zIn) add(t.pair, zIn, pairs * t.C);
  int mixer = 0; while (M.has("trunk.pairMixerBlocks." + num(mixer) + ".singleChannels")) ++mixer;
  // (Mol-1's trunk_in_single is after its PairMixer, where the reference taps it)
  if (t.pass == 1) { seam("z_after_msa", t.pair, pairs * t.C); if (!mixer) seam("trunk_in_single", t.single, (size_t)t.n * t.Cs); }
  // (and the MSA stack's pair-track scratch and MSA tensor: the stack's half pair is made beside the f32 one before its blocks make theirs)
  if (reclaim) {
    if (t.msa) { release(t.msa); t.msa = nullptr; }
    releaseScratch({"msaatt.", "msatr.", "opm.", "gopm.", "mnorm.", "trunk.zBeforeMsa", "pr.", "grid.", "tr.", "par."});
  }
  int blocks = 0; while (M.has("trunk.pairformerBlocks." + num(blocks) + ".singleChannels")) ++blocks;
  // The stack over a half pair: its pair updates are bandwidth-bound GEMMs that read and write the pair, and half moves
  // half the bytes (128 x 65025 x 128 adding into the pair: 0.70 ms float, 0.47 half). Every residual is summed in
  // float and rounded once, every norm reads the half values. LOCALFOLD_HALF_PAIR=0 the control.
  // (Pair values seen: up to ~13000 - openbind0 - against half's 65504; the stores saturate rather than overflow.)
  static const bool halfWanted = !getenv("LOCALFOLD_HALF_PAIR") || atoi(getenv("LOCALFOLD_HALF_PAIR")) != 0;
  // (the f32 pair given back while the stack runs on the half one, and made again once the pair track's scratch is:
  // the trunk's top stage, 1044 tokens 4.30 -> 3.74 GB)
  t.pairHalfExact = false;
  auto stack = [&](auto* z, int C, const std::string& S, int count) {
    for (int k = 0; k < count; ++k)
      pairformerBlock(z, t.single, t.masks, t.n, C, t.Cs, S + num(k), nullptr, k + 1 < count ? S + num(k + 1) : "");
  };
  const int Cn = mixer ? metaI("trunk.trunkPairChannels") : 0;
  if (mixer && halfWanted && t.C % 8 == 0 && Cn % 8 == 0) {
    // Mol-1: eight PairMixer blocks at the pair's width, then the pairformer on a narrower pair between two bias-free
    // adapters (trunk_pair_in, trunk_pair_out) - the narrow pair a buffer of its own, the full pair rewritten from it
    half* zh = scratch<half>("trunk.zh", pairs * t.C);
    toHalf(t.pair, zh, pairs * t.C);
    if (reclaim) { release(t.pair); t.pair = nullptr; }
    stack(zh, t.C, "trunk.pairMixerBlocks.", mixer);
    half* nz = scratch<half>("trunk.narrow", pairs * Cn);
    linH(zh, "trunk.trunkPairIn", nz, pairs, t.C, Cn);
    if (reclaim) releaseScratch({"trunk.zh", "pr.", "grid.", "tr.", "st.", "par.", "trunk.sn"});
    if (t.pass == 1) { seamHalf("trunk_in_pair", nz, pairs * Cn); seam("trunk_in_single", t.single, (size_t)t.n * t.Cs); }
    stack(nz, Cn, "trunk.pairformerBlocks.", blocks);
    if (t.pass == 1) seamHalf("trunk_out_pair", nz, pairs * Cn);
    if (reclaim) {
      releaseScratch({"pr.", "grid.", "tr.", "st.", "par.", "trunk.sn"});
      t.pair = allocT<float>(pairs * t.C);
    }
    lin(nz, "trunk.trunkPairOut", t.pair, pairs, Cn, t.C);
    if (reclaim) releaseScratch({"trunk.narrow"});
  } else if (mixer) {      // (the control: LOCALFOLD_HALF_PAIR=0, every pair in float)
    stack(t.pair, t.C, "trunk.pairMixerBlocks.", mixer);
    float* nz = scratch<float>("trunk.narrow", pairs * Cn);
    lin(t.pair, "trunk.trunkPairIn", nz, pairs, t.C, Cn);
    if (t.pass == 1) { seam("trunk_in_pair", nz, pairs * Cn); seam("trunk_in_single", t.single, (size_t)t.n * t.Cs); }
    stack(nz, Cn, "trunk.pairformerBlocks.", blocks);
    if (t.pass == 1) seam("trunk_out_pair", nz, pairs * Cn);
    lin(nz, "trunk.trunkPairOut", t.pair, pairs, Cn, t.C);
    if (reclaim) releaseScratch({"trunk.narrow"});
  } else if (halfWanted && t.C % 8 == 0) {
    half* zh = scratch<half>("trunk.zh", pairs * t.C);
    toHalf(t.pair, zh, pairs * t.C);
    if (reclaim) { release(t.pair); t.pair = nullptr; }
    for (int k = 0; k < blocks; ++k)
      pairformerBlock(zh, t.single, t.masks, t.n, t.C, t.Cs, "trunk.pairformerBlocks." + num(k), nullptr,
                      k + 1 < blocks ? "trunk.pairformerBlocks." + num(k + 1) : "");
    if (reclaim) {
      releaseScratch({"pr.", "grid.", "tr.", "st.", "par.", "trunk.sn"});
      t.pair = allocT<float>(pairs * t.C);
    }
    toFloat(zh, t.pair, pairs * t.C);
    if (reclaim) releaseScratch({"trunk.zh"});
    t.pairHalfExact = true;
  } else
    for (int k = 0; k < blocks; ++k)
      pairformerBlock(t.pair, t.single, t.masks, t.n, t.C, t.Cs, "trunk.pairformerBlocks." + num(k), nullptr,
                      k + 1 < blocks ? "trunk.pairformerBlocks." + num(k + 1) : "");
  if (reclaim) releaseScratch({"pr.", "grid.", "tr.", "st.", "par.", "trunk.sn"});     // (the next pass's embedder and MSA stack would hold theirs beside it)
}

// ---------------------------------------------------------------- the distogram
static void distogramHalf(const float* pair, float* half_, size_t rows, int C, int bins) {
  const std::string D = "trunk.distogram.";
  if (hasW(D + "hidden")) {      // chai-1's MLP head: LN, gelu(z W1 + b1), W2 + b2
    int Hd = (int)lenW(D + "hiddenBias");
    half* lnP = scratch<half>("disto.ln", rows * C); half* h = scratch<half>("disto.hidden", rows * Hd);
    ln(pair, lnP, rows, C, D + "inputLayerNormScale", D + "inputLayerNormOffset");
    Gemm g{}; g.X = lnP; g.tx = F16; g.W = Wh(D + "hidden"); g.tw = F16; g.Y = h; g.ty = F16; g.rows = rows; g.in = C; g.out = Hd;
    g.bias = W(D + "hiddenBias"); g.gelu = true; g.accFloat = true; gemm(g);
    lin(h, D + "halfLogits", half_, rows, Hd, bins, 0.f, Wopt(D + "halfLogitsBias"));
    return;
  }
  lin(pair, D + "halfLogits", half_, rows, C, bins, 0.f, Wopt(D + "halfLogitsBias"));
}
void distogram(Trunk& t, float* logits) {
  int bins = metaI("trunk.distogram.bins");
  size_t pairs = (size_t)t.n * t.n;
  float* h = scratch<float>("disto.half", pairs * bins);
  distogramHalf(t.pair, h, pairs, t.C, bins);
  run1d("af3_symmetrise", (bins & 3) ? pairs * bins : pairs * (bins >> 2),
        SymmetriseArgs{h, nullptr, logits, (uint)t.n, (uint)bins, flag("trunk.dialect.mlpDistogram") ? 0.5f : 1.f, 0});
}
// the contact thresholds (shared/heads/contact-threshold.js), cut against this model's own distogram bins
static float contactAngstroms(int a, int b) {
  static const float LIGAND_PROTEIN[20] = {5, 8, 7, 7, 6, 7, 7, 5, 8, 6, 7, 7, 8, 8, 6, 6, 6, 7, 8, 6};
  auto kind = [](int c) { return c == 0 ? 0 : c == 1 ? 1 : 2; };       // nucleic, ligand, protein
  int ka = kind(a), kb = kind(b);
  if (ka == 1 && kb == 2 && b >= 2 && b < 22) return LIGAND_PROTEIN[b - 2];
  if (kb == 1 && ka == 2 && a >= 2 && a < 22) return LIGAND_PROTEIN[a - 2];
  static const float BY_KIND[3][3] = {{9, 7, 10}, {7, 5, 7}, {10, 7, 8}};
  return BY_KIND[ka][kb];
}
static const int* contactBins(int n, int bins) {
  const int CLASSES = 23;
  std::vector<float> breaks(bins - 1);
  for (int i = 0; i < bins - 1; ++i) breaks[i] = (float)(2.3125 + (21.6875 - 2.3125) * i / (double)(bins - 2));
  double spacing = (double)breaks[bins - 2] - (double)breaks[bins - 3];
  auto top = [&](int bin) { return bin < bins - 1 ? (double)breaks[bin] : (double)breaks[bins - 2] + spacing; };
  std::vector<int> table(CLASSES * CLASSES);
  for (int a = 0; a < CLASSES; ++a)
    for (int b = 0; b < CLASSES; ++b) {
      double angstroms = contactAngstroms(a, b);
      int count = 0;
      while (count <= bins - 1 && top(count) <= angstroms + 1e-3) ++count;
      table[a * CLASSES + b] = count;
    }
  int* tableDev = scratch<int>("contact.table", table.size());
  upload(tableDev, table.data(), table.size() * 4);
  int* out = scratch<int>("contact.bins", (size_t)n * n);
  run1d("af3_contact_bins", (size_t)n * n, ContactBinsArgs{Ib("batch.contactClasses"), tableDev, out, (uint)n, 0});
  return out;
}
std::vector<float> contactProbabilities(Trunk& t) {
  if (!M.has("batch.contactClasses")) return {};
  int bins = metaI("trunk.distogram.bins");
  size_t pairs = (size_t)t.n * t.n;
  float* logits = scratch<float>("disto.logits", pairs * bins);
  distogram(t, logits);
  float* out = scratch<float>("disto.contact", pairs);
  run1d("af3_contact_probs", pairs, ContactProbsArgs{logits, contactBins(t.n, bins), t.pairMask, out, pairs, (uint)bins, 0});
  std::vector<float> c = download(out, pairs);
  releaseScratch({"disto.", "contact."});
  return c;
}
// the distance each pair's distogram predicts: the expectation over the bin centres, the open first and last bins at
// their breaks (shared/af3/feature-convergence.js expectedDistances) - --recycle-tolerance's measure
std::vector<float> expectedDistances(Trunk& t) {
  int bins = metaI("trunk.distogram.bins");
  size_t pairs = (size_t)t.n * t.n;
  float* logits = scratch<float>("disto.logits", pairs * bins);
  distogram(t, logits);
  std::vector<float> l = download(logits, pairs * bins), out(pairs);
  const float fb = 2.3125f, lb = 21.6875f;
  for (size_t ij = 0; ij < pairs; ++ij) {
    const float* r = &l[ij * bins];
    float mx = -INFINITY; for (int b = 0; b < bins; ++b) mx = std::max(mx, r[b]);
    double total = 0, weighted = 0;
    for (int b = 0; b < bins; ++b) {
      float centre = b == 0 ? fb : b == bins - 1 ? lb : fb + (lb - fb) * (b - 0.5f) / (bins - 2);
      double p = std::exp(r[b] - mx); total += p; weighted += p * centre;
    }
    out[ij] = total > 0 ? (float)(weighted / total) : 0.f;
  }
  releaseScratch({"disto."});
  return out;
}
