// The trunk: the embedder, the template stack, the MSA stack, the pairformer and the distogram (cuda/af3/src/trunk.cuh
// is the reading). The pair, single and MSA stay float32; every GEMM is half with float accumulation.
#include "af3.h"
#include <cmath>

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
  t.S = std::min((int)M.meta("batch.sequences"), msaCap);
  size_t pairs = (size_t)t.n * t.n;
  t.pair = allocT<float>(pairs * t.C); t.single = allocT<float>((size_t)t.n * t.Cs);
  t.msa = allocT<float>((size_t)t.S * t.n * t.Cm);
  t.prevPair = allocT<float>(pairs * t.C); t.prevSingle = allocT<float>((size_t)t.n * t.Cs);
  t.targetFeat = allocT<float>((size_t)t.n * t.F);
  copy(t.targetFeat, targetFeat, (size_t)t.n * t.F * 4);
  std::vector<float> seq(M.hostF("batch.seqMask"), M.hostF("batch.seqMask") + t.n), pm(pairs);
  bool ones = true;
  for (int i = 0; i < t.n; ++i) { ones &= seq[i] > 0; for (int j = 0; j < t.n; ++j) pm[(size_t)i * t.n + j] = seq[i] * seq[j]; }
  t.seqMask = uploadNew(seq.data(), t.n); t.pairMask = uploadNew(pm.data(), pairs);
  size_t rows = (size_t)t.S * t.n;           // the first S rows (AF3 keeps the first num_msa after its identity shuffle)
  t.msaRows = uploadNew(M.hostI("batch.msa"), rows);
  t.deletion = uploadNew(M.hostF("batch.deletionMatrix"), rows);
  t.msaMask = uploadNew(M.hostF("batch.msaMask"), rows);
  t.masks = {t.pairMask, t.seqMask, ones};
  return t;
}
void freeTrunk(Trunk& t) {
  for (const void* p : {(const void*)t.pair, (const void*)t.single, (const void*)t.msa, (const void*)t.targetFeat, (const void*)t.pairMask,
                        (const void*)t.seqMask, (const void*)t.msaMask, (const void*)t.prevPair, (const void*)t.prevSingle,
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
  int width = (int)M.meta("template.featureWidth", 0);
  bool outer = flag("template.outerResidual");
  half* qn = scratch<half>("tmpl.qn", pairs * Cq);
  ln(t.pair, qn, pairs, Cq, P + "queryEmbeddingNormScale", P + "queryEmbeddingNormOffset");
  float* query = scratch<float>("tmpl.query", pairs * Ct);
  lin(qn, P + (fused ? "zProjection" : "templatePairEmbedding8"), query, pairs, Cq, Ct);
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
    copy(act, query, pairs * Ct * 4);
    if (fused) {
      int K = metaI(S + "featuresK");
      if (M.len(S + "featuresIdx") != pairs * K) die("%sfeatures: %zu entries, not %zu x %d", S.c_str(), M.len(S + "featuresIdx"), pairs, K);
      float* dense = scratch<float>("tmpl.features", pairs * width);
      fill(dense, 0, pairs * width * 4);
      run1d("af3_scatter_rows", pairs * K, ScatterRowsArgs{Ib(S + "featuresIdx"), Fb(S + "featuresVal"), dense, pairs, (uint)K, (uint)width});
      lin(dense, P + "aProjection", act, pairs, width, Ct, 1.f);
    } else {
      run1d("af3_onehot", (size_t)n * 31, OnehotArgs{Ib(S + "aatype"), oh, (uint)n, 31});
      lin(oh, P + "templatePairEmbedding2", row, n, 31, Ct);
      lin(oh, P + "templatePairEmbedding3", col, n, 31, Ct);
      run1d("af3_add_row_col", pairs * Ct, AddRowColArgs{act, row, col, (uint)n, (uint)Ct});
      if (M.has(S + "distogramBin")) {
        int bins = (int)(lenW(P + "templatePairEmbedding0") / Ct);
        if (M.len(S + "distogramBin") != pairs || metaI(S + "distogramBins") != bins) die("%sdistogram: the wrong shape", S.c_str());
        run1d("af3_template_geometry", pairs * Ct,
              TmplGeomArgs{act, Ib(S + "distogramBin"), Fb(S + "pseudoBetaMask2d"), Fb(S + "unitVector"), Fb(S + "backboneMask2d"),
                           W(P + "templatePairEmbedding0"), W(P + "templatePairEmbedding1"), W(P + "templatePairEmbedding4"),
                           W(P + "templatePairEmbedding5"), W(P + "templatePairEmbedding6"), W(P + "templatePairEmbedding7"),
                           pairs, (uint)Ct, 0});
      }
    }
    if (hasW(P + "templateFeatureBias")) addBias(act, W(P + "templateFeatureBias"), pairs, Ct);
    if (outer) copy(before, act, pairs * Ct * 4);
    for (int b = 0; b < nb; ++b) pairUpdates(act, t.masks, n, Ct, P + "blocks." + num(b));
    if (outer) add(act, before, pairs * Ct);
    float* actn = scratch<float>("tmpl.actn", pairs * Ct);
    ln(act, actn, pairs, Ct, P + "outputLayerNormScale", P + "outputLayerNormOffset");
    add(summed, actn, pairs * Ct, repeat);
  }
  scale(summed, pairs * Ct, 1.f / (1e-7f + templates), true);     // divided by every slot (not the real ones), relu
  lin(summed, P + "outputLinear", t.pair, pairs, Ct, Cq, 1.f);
  releaseScratch({"tmpl."});
}

// ---------------------------------------------------------------- the embedder
static void embed(Trunk& t) {
  int n = t.n, C = t.C; size_t pairs = (size_t)n * n;
  const std::string E = "trunk.embedder.";
  if (flag("trunk.dialect.recycleFromInit") || flag("trunk.dialect.chaiMsaFeatures")) die("chai-1 is not in the native port yet");
  float* left = scratch<float>("emb.left", (size_t)n * C); float* right = scratch<float>("emb.right", (size_t)n * C);
  const float* pairSource = t.targetFeat; int sourceWidth = t.F;
  if (flag("trunk.dialect.pairInitFromSingle")) {        // OpenDDE: the pair from s_init
    float* sInit = scratch<float>("emb.sInit", (size_t)n * t.Cs);
    lin(t.targetFeat, E + "singleActivations", sInit, n, t.F, t.Cs);
    pairSource = sInit; sourceWidth = t.Cs;
  }
  lin(pairSource, E + "leftSingle", left, n, sourceWidth, C);
  lin(pairSource, E + "rightSingle", right, n, sourceWidth, C);
  // pair = left[i] + right[j] + prevEmbedding(LN(prev pair)) - not zero on the first pass
  float* prev = scratch<float>("emb.prev", pairs * C);
  {
    half* pn = scratch<half>("emb.prevln", pairs * C);
    ln(t.prevPair, pn, pairs, C, E + "prevEmbeddingNormScale", E + "prevEmbeddingNormOffset");
    lin(pn, E + "prevEmbedding", prev, pairs, C, C);
  }
  if (t.pass == 0) { seam("dbg_left", left, (size_t)n * C); seam("dbg_prev", prev, pairs * C); seam("dbg_tf", t.targetFeat, (size_t)n * t.F); }
  run1d("af3_outer_sum", pairs * C, OuterSumArgs{left, right, prev, t.pair, (uint)n, (uint)C});
  if (t.pass == 0) seam("dbg_outer", t.pair, pairs * C);
  run1d("af3_relenc", pairs * C, RelEncArgs{relIdx(), W(E + "positionActivations"), t.pair, (uint)n, (uint)C});
  const bool bonds = M.has("batch.bondMatrix") && hasW(E + "bondEmbedding"), types = hasW(E + "tokenBondsTypeEmbed");
  if (bonds || types)
    run1d("af3_bond_embed", pairs * C,
          BondEmbedArgs{t.pair, bonds ? Fb("batch.bondMatrix") : nullptr, bonds ? W(E + "bondEmbedding") : nullptr,
                        types && M.has("batch.bondOrderMatrix") ? Fb("batch.bondOrderMatrix") : nullptr,
                        types ? W(E + "tokenBondsTypeEmbed") : nullptr, types ? W(E + "contactEncodingUnspecified") : nullptr,
                        pairs, (uint)C, 0});
  releaseScratch({"emb.prev"});
  if (t.pass == 0) seam("z_init_generic", t.pair, pairs * C);
  templateEmbedding(t);
  if (t.pass == 0) seam("z_after_template", t.pair, pairs * C);
  // the MSA: its features projected, plus the target's projection broadcast over the rows
  float* fromTarget = scratch<float>("emb.fromTarget", (size_t)n * t.Cm);
  lin(t.targetFeat, E + "extraMsaTargetFeat", fromTarget, n, t.F, t.Cm);
  size_t rows = (size_t)t.S * n;
  int msaWidth = (int)(lenW(E + "msaActivations") / t.Cm);
  if (msaWidth != 34 && msaWidth != 35) die("msa feature width %d", msaWidth);
  run1d("af3_msa_embed", rows * t.Cm, MsaEmbedArgs{t.msaRows, t.deletion, W(E + "msaActivations"), fromTarget, t.msa, rows, (uint)n,
                                                   (uint)t.Cm, (uint)msaWidth, flag("trunk.dialect.msaPairedQueryRow") ? 1u : 0u});
  // the single: target_feat projected, plus the recycled single's
  lin(t.targetFeat, E + "singleActivations", t.single, n, t.F, t.Cs);
  half* sln = scratch<half>("emb.prevsln", (size_t)n * t.Cs);
  ln(t.prevSingle, sln, n, t.Cs, E + "prevSingleEmbeddingNormScale", E + "prevSingleEmbeddingNormOffset");
  lin(sln, E + "prevSingleEmbedding", t.single, n, t.Cs, t.Cs, 1.f);
  ++t.pass;
}

// ---------------------------------------------------------------- the MSA stack
// the outer product mean: pair[i][j] += (sum_s l[s][i] (x) r[s][j] W + b) / (1e-3 + norm[i][j])
static void outerProductMean(Trunk& t, const std::string& pre) {
  int L = t.n, S = t.S, Cm = t.Cm, C = t.C;
  int O = metaI(pre + ".outerChannels");
  if (M.has(pre + ".groups")) die("chai-1's grouped outer product is not in the native port yet");
  size_t rows = (size_t)S * L;
  half* xn = scratch<half>("opm.xn", rows * Cm);
  ln(t.msa, xn, rows, Cm, pre + ".layerNormInputScale", pre + ".layerNormInputOffset");
  half* lt = scratch<half>("opm.left", rows * O); half* rt = scratch<half>("opm.right", rows * O);
  linH(xn, pre + ".leftProjection", lt, rows, Cm, O, Wopt(pre + ".leftProjectionBias"));      // (rf3's biases, before the mask)
  linH(xn, pre + ".rightProjection", rt, rows, Cm, O, Wopt(pre + ".rightProjectionBias"));
  run1d("af3_scale_rows_h", rows * O, ScaleRowsHArgs{lt, t.msaMask, rows, (uint)O, 0});
  run1d("af3_scale_rows_h", rows * O, ScaleRowsHArgs{rt, t.msaMask, rows, (uint)O, 0});
  float* norm = scratch<float>("opm.norm", (size_t)L * L);
  run1d("af3_mask_norm", (size_t)L * L, MaskNormArgs{t.msaMask, norm, (uint)S, (uint)L});
  const half* Wout = Wh(pre + ".outputW");          // [O * O][C]
  const uint after = flag("trunk.dialect.opmBiasAfterNorm") ? 1u : 0u;
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
    float* Y = scratch<float>("opm.Y", (size_t)Bi * L * C);
    for (int i0 = 0; i0 < L; i0 += Bi) {
      int bi = std::min(Bi, L - i0);
      Gemm g{}; g.X = Lt + (size_t)i0 * K; g.tx = F16; g.W = T; g.tw = F16; g.Y = Y; g.rows = bi; g.in = K; g.out = L * C; g.label = "opm shallow";
      gemm(g);
      run1d("af3_opm_add", (size_t)bi * L * C, OpmAddArgs{t.pair, Y, W(pre + ".outputB"), norm, (u64)i0, (uint)bi, (uint)L, (uint)C, after});
    }
    return;
  }
  // the standard form: P[(i, c)][(j, e)] = sum_s l[s][i][c] r[s][j][e] a block of rows i at a time, permuted to
  // [(i, j)][(c, e)], then the output projection
  size_t per = (size_t)L * O * O;
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / per));
  half* Pm = scratch<half>("opm.P", (size_t)Bi * per); half* X = scratch<half>("opm.X", (size_t)Bi * per);
  float* Y = scratch<float>("opm.Y", (size_t)Bi * L * C);
  for (int i0 = 0; i0 < L; i0 += Bi) {
    int bi = std::min(Bi, L - i0);
    { Gemm g{}; g.X = lt + (size_t)i0 * O; g.tx = F16; g.transX = true; g.ldx = L * O; g.W = rt; g.tw = F16; g.ldw = L * O; g.Y = Pm;
      g.ty = F16; g.rows = (size_t)bi * O; g.in = S; g.out = L * O; g.accFloat = true; g.label = "opm product"; gemm(g); }
    run1d("af3_opm_permute", (size_t)bi * L * O * (O / 8), OpmPermuteArgs{Pm, X, (uint)bi, (uint)L, (uint)O, 0});
    { Gemm g{}; g.X = X; g.tx = F16; g.W = Wout; g.tw = F16; g.Y = Y; g.rows = (size_t)bi * L; g.in = O * O; g.out = C; g.label = "opm output"; gemm(g); }
    run1d("af3_opm_add", (size_t)bi * L * C, OpmAddArgs{t.pair, Y, W(pre + ".outputB"), norm, (u64)i0, (uint)bi, (uint)L, (uint)C, after});
  }
}
// AF3's MSA "attention": per head, weights softmax_j over LN(pair) projected (no queries or keys), averaging every
// row's values; gated and projected back into the MSA
static void msaAttention(Trunk& t, const std::string& pre) {
  int n = t.n, S = t.S, Cm = t.Cm, C = t.C;
  int heads = metaI(pre + ".heads"), d = metaI(pre + ".dimension"), Wd = heads * d;
  size_t rows = (size_t)S * n, pairs = (size_t)n * n;
  half* xn = scratch<half>("msaatt.ln", rows * Cm);
  ln(t.msa, xn, rows, Cm, pre + ".actNormScale", pre + ".actNormOffset");
  half* pln = scratch<half>("msaatt.pln", pairs * C);
  ln(t.pair, pln, pairs, C, pre + ".pairNormScale", pre + ".pairNormOffset");
  float* flat = scratch<float>("msaatt.flat", pairs * heads);
  lin(pln, pre + ".pairLogits", flat, pairs, C, heads);
  float* keyMask = scratch<float>("msaatt.keymask", n);
  run1d("af3_key_mask", n, KeyMaskArgs{t.msaMask, keyMask, (uint)S, (uint)n});
  int ldw = (n + 7) / 8 * 8;
  half* w = scratch<half>("msaatt.w", (size_t)heads * n * ldw);
  run("af3_msa_weights", Grid{(uint32_t)(heads * n), 1, 1}, 256, MsaWeightsArgs{flat, keyMask, w, (uint)n, (uint)heads, (uint)ldw, 0});
  half* v = scratch<half>("msaatt.v", rows * Wd);
  linH(xn, pre + ".vProjection", v, rows, Cm, Wd);
  half* vh = scratch<half>("msaatt.vh", (size_t)S * ldw * Wd);
  run1d("af3_msa_v_heads", (size_t)S * ldw * Wd, MsaVHeadsArgs{v, vh, (uint)S, (uint)n, (uint)heads, (uint)d, (uint)ldw, 0});
  // per head: O_h [n][S d] = W_h [n][ldw] V_h [ldw][S d]
  half* oh = scratch<half>("msaatt.oh", rows * Wd);
  { Gemm g{}; g.X = w; g.tx = F16; g.sx = (int64_t)n * ldw; g.W = vh; g.tw = F16; g.sw = (int64_t)ldw * S * d; g.Y = oh; g.ty = F16;
    g.sy = (int64_t)n * S * d; g.rows = n; g.in = ldw; g.out = S * d; g.batch = heads; g.accFloat = true; g.label = "msa average"; gemm(g); }
  half* gate = scratch<half>("msaatt.gate", rows * Wd);
  linH(xn, pre + ".gatingQuery", gate, rows, Cm, Wd);
  half* gated = scratch<half>("msaatt.gated", rows * Wd);
  run1d("af3_msa_from_heads", rows * Wd, MsaFromHeadsArgs{oh, gate, gated, (uint)S, (uint)n, (uint)heads, (uint)d});
  lin(gated, pre + ".outputProjection", t.msa, rows, Wd, Cm, 1.f);
}
static void msaBlock(Trunk& t, int k) {
  std::string B = "trunk.msaBlocks." + num(k);
  bool updateFirst = flag("trunk.dialect.msaUpdateBeforeOuterProduct");     // (OpenDDE, boltz2)
  if (!updateFirst) outerProductMean(t, B + ".outerProductMean");
  msaAttention(t, B + ".msaAttention1");
  transition(t.msa, (size_t)t.S * t.n, t.Cm, B + ".msaTransition");
  if (updateFirst) outerProductMean(t, B + ".outerProductMean");
  pairUpdates(t.pair, t.masks, t.n, t.C, B);
}

// ---------------------------------------------------------------- one pass
void runTrunk(Trunk& t) {
  size_t pairs = (size_t)t.n * t.n;
  // the recycled state: the last pass's pair and single (zero before the first)
  if (t.pass > 0) { copy(t.prevPair, t.pair, pairs * t.C * 4); copy(t.prevSingle, t.single, (size_t)t.n * t.Cs * 4); }
  embed(t);
  releaseScratch({"emb."});
  int msaBlocks = 0; while (M.has("trunk.msaBlocks." + num(msaBlocks) + ".pairChannels")) ++msaBlocks;
  float* zIn = nullptr;
  if (flag("trunk.dialect.msaDoubleAddPair")) {     // boltz2 adds the pre-MSA pair back
    zIn = scratch<float>("trunk.zBeforeMsa", pairs * t.C);
    copy(zIn, t.pair, pairs * t.C * 4);
  }
  for (int k = 0; k < msaBlocks; ++k) msaBlock(t, k);
  if (zIn) add(t.pair, zIn, pairs * t.C);
  if (t.pass == 1) { seam("z_after_msa", t.pair, pairs * t.C); seam("trunk_in_single", t.single, (size_t)t.n * t.Cs); }
  releaseScratch({"msaatt.", "opm.", "trunk.zBeforeMsa"});
  int blocks = 0; while (M.has("trunk.pairformerBlocks." + num(blocks) + ".singleChannels")) ++blocks;
  for (int k = 0; k < blocks; ++k) pairformerBlock(t.pair, t.single, t.masks, t.n, t.C, t.Cs, "trunk.pairformerBlocks." + num(k));
}

// ---------------------------------------------------------------- the distogram
static void distogramHalf(const float* pair, float* half_, size_t rows, int C, int bins) {
  const std::string D = "trunk.distogram.";
  if (hasW(D + "hidden")) die("chai-1's distogram head is not in the native port yet");
  lin(pair, D + "halfLogits", half_, rows, C, bins, 0.f, Wopt(D + "halfLogitsBias"));
}
void distogram(Trunk& t, float* logits) {
  int bins = metaI("trunk.distogram.bins");
  size_t pairs = (size_t)t.n * t.n;
  float* h = scratch<float>("disto.half", pairs * bins);
  distogramHalf(t.pair, h, pairs, t.C, bins);
  run1d("af3_symmetrise", pairs * bins, SymmetriseArgs{h, logits, (uint)t.n, (uint)bins, flag("trunk.dialect.mlpDistogram") ? 0.5f : 1.f, 0});
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
