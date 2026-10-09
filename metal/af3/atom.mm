// The per-atom conditioning and the atom cross-attention encoder - which builds target_feat's 384 atom columns and
// runs again inside every denoiser call - and its block, which the decoder shares (cuda/af3/src/atom.cuh is the
// reading). Atoms are attended in subsets: 32 queries against 128 keys each, biased per block by the atom pair.
#include "af3.h"
#include <cmath>

int NS = 1;
bool ADA_RAW = false;

Gather gatherOf(const std::string& name) {
  return {M.i(name + ".indices"), M.f(name + ".mask"), (int)M.len(name + ".indices")};
}
AtomShape atomShape() {
  return {metaI("batch.shape.tokens"), metaI("batch.shape.dense"), metaI("batch.shape.subsets"), metaI("batch.shape.queries"),
          metaI("batch.shape.keys")};
}
void convert(const Gather& g, const float* src, float* out, int C, size_t srcRows, int ns) {
  run1d("af3_convert", (size_t)ns * g.count * C, ConvertArgs{g.idx, g.mask, src, out, srcRows, (uint)g.count, (uint)C, (uint)ns, 0});
}
void adaLn(const float* x, const float* scale, const float* shift, half* out, float* outF, size_t rows, int C, size_t period, int ld) {
  run("af3_ada_ln", grid1d((rows + 7) / 8, 1), 256,
      AdaLnArgs{x, scale, shift, out, outF, rows, (uint)C, (uint)period, (uint)(ld ? ld : C), ADA_RAW ? 1u : 0u, ADA_RAW ? 0.1f : 1e-5f, 0});
}

// ---------------------------------------------------------------- the conditioning
static float* perAtomConditioning(const std::string& ref, int rows) {
  int C = metaI(ref + ".channels");
  float* act = scratch<float>(ref + ".cond", (size_t)rows * C);
  run1d("af3_per_atom", (size_t)rows * C,
        PerAtomArgs{M.f("batch.refPos"), M.f("batch.refMask"), M.i("batch.refElement"), M.f("batch.refCharge"),
                    M.i("batch.refAtomNameChars"), W(ref + ".embedRefPos"), W(ref + ".embedRefMask"), W(ref + ".embedRefElement"),
                    W(ref + ".embedRefCharge"), W(ref + ".embedRefAtomName"), Wopt(ref + ".embedAtomFeaturesBias"), act, (uint)rows,
                    (uint)C, flag("trunk.dialect.rawRefCharge") ? 1u : 0u, 0});
  return act;
}
// scale = LN_s(cond) W + b and shift = LN_s(cond) W' for one adaptive LayerNorm (chai: the conditioning as it is,
// scale = cond W + 1)
static void adaCond(const float* cond, size_t rows, int C, int condC, const std::string& w, float* sc, float* sh) {
  if (ADA_RAW) die("chai-1's adaptive LayerNorm is not in the native port yet");
  float* cn = scratch<float>("ada.cn", rows * condC);
  ln(cond, cn, rows, condC, w + "SingleCondLayerNormScale", "");
  lin(cn, w + "SingleCondScaleWeights", sc, rows, condC, C, 0.f, W(w + "SingleCondScaleBias"));
  lin(cn, w + "SingleCondBias", sh, rows, condC, C);
}
AtomBlockCache prepareAtomBlock(const std::string& B, const float* qCond, size_t qRows, int C, float* pairLogits) {
  AtomBlockCache c{};
  auto a = [&](const char* tag) { return scratch<float>(B + tag, qRows * C); };
  c.qScale = a(".qs"); c.qShift = a(".qh"); c.kScale = a(".ks"); c.kShift = a(".kh");
  c.ffwScale = a(".fs"); c.ffwShift = a(".fh"); c.zg = a(".zg"); c.tg = a(".tg");
  adaCond(qCond, qRows, C, C, B + ".q", c.qScale, c.qShift);
  adaCond(qCond, qRows, C, C, B + ".k", c.kScale, c.kShift);
  adaCond(qCond, qRows, C, C, B + ".ffw", c.ffwScale, c.ffwShift);
  lin(qCond, B + ".AdaptiveZeroCondWeights", c.zg, qRows, C, C, 0.f, W(B + ".AdaptiveZeroCondBias"));
  lin(qCond, B + ".ffwAdaptiveZeroCondWeights", c.tg, qRows, C, C, 0.f, W(B + ".ffwAdaptiveZeroCondBias"));
  c.pairLogits = pairLogits;
  c.chained = flag(B + ".chainedAtomLayerNorm");
  return c;
}

// ---------------------------------------------------------------- one cross-attention block, in place on act
void crossAttentionBlock(float* act, const AtomStep& st, const AtomBlockCache& bc, const AtomShape& sh, int C, int heads, int D,
                         const std::string& B) {
  size_t q1 = (size_t)sh.subsets * sh.queries, qRows = q1 * NS, kRows = (size_t)sh.subsets * sh.keys * NS;
  int Wd = heads * D;
  if (D > 32 || sh.keys > 128) die("atom attention: %d keys of a head %d wide", sh.keys, D);
  float* pre = nullptr;
  if (st.noResidual) { pre = scratch<float>("ab.pre", qRows * C); copy(pre, act, qRows * C * 4); }
  half* xq = scratch<half>("ab.xq", qRows * C); half* xk = scratch<half>("ab.xk", qRows * C);
  if (bc.chained) {         // xk = adaLN_k(adaLN_q(x)): the queries normalised in f32 first (OpenDDE, protenix2)
    float* xqF = scratch<float>("ab.xqF", qRows * C);
    adaLn(act, bc.qScale, bc.qShift, xq, xqF, qRows, C, q1);
    adaLn(xqF, bc.kScale, bc.kShift, xk, nullptr, qRows, C, q1);
  } else {
    adaLn(act, bc.qScale, bc.qShift, xq, nullptr, qRows, C, q1);
    adaLn(act, bc.kScale, bc.kShift, xk, nullptr, qRows, C, q1);
  }
  half* qg = scratch<half>("ab.qg", qRows * 2 * Wd); half* kvAtom = scratch<half>("ab.kvAtom", qRows * 2 * Wd);
  linW(xq, concatColumns(B + ".qg", C, {{B + ".qProjection", Wd, false}, {B + ".gatingQuery", Wd, false}}), qg, qRows, C, 2 * Wd,
       nullptr, "atom q gate");
  linW(xk, concatColumns(B + ".kv", C, {{B + ".kProjection", Wd, false}, {B + ".vProjection", Wd, false}}), kvAtom, qRows, C, 2 * Wd,
       nullptr, "atom k v");
  const float* qBias = W(B + ".qBias");
  if (hasW(B + ".queryLayerNormScale")) {       // rf3: q (its bias inside) and k normalised per atom row
    run("af3_kq_norm", grid1d((qRows + 7) / 8, 1), 256,
        KqNormArgs{qg, kvAtom, qBias, W(B + ".queryLayerNormScale"), W(B + ".queryLayerNormOffset"), W(B + ".keyLayerNormScale"),
                   W(B + ".keyLayerNormOffset"), qRows, (uint)(2 * Wd), (uint)(2 * Wd), (uint)Wd, 0});
    qBias = M.derived<float>("zeros:" + num(Wd), Wd, [](float*) {});
  }
  half* kv = scratch<half>("ab.kv", kRows * 2 * Wd);
  uint chunks = (uint)(2 * Wd * 2 / 16);
  run1d("af3_gather_rows", kRows * chunks, GatherRowsArgs{(const uchar*)kvAtom, st.queriesToKeys.idx, st.queriesToKeys.mask, (uchar*)kv,
                                                         kRows, (u64)st.queriesToKeys.count, q1, chunks, 0});
  half* gathered = scratch<half>("ab.gathered", qRows * Wd);
  static const bool scalarAttention = getenv("AF3_SCALAR_ATOM_ATTENTION") != nullptr;      // (the control arm)
  const bool mma = !scalarAttention && sh.queries == 32 && sh.keys == 128 && (D == 16 || D == 32);
  run(mma ? (D == 16 ? "af3_atom_attention_mma16" : "af3_atom_attention_mma32") : "af3_atom_attention",
      Grid{(uint32_t)(sh.subsets * NS), (uint32_t)heads, 1}, 128,
      AtomAttnArgs{qg, qBias, kv, st.qMask, st.kMask, bc.pairLogits, gathered, (uint)sh.queries, (uint)sh.keys,
                   (uint)heads, (uint)D, (uint)sh.subsets, st.keyMasked ? 1u : 0u});
  float* attention = scratch<float>("ab.attention", qRows * C);
  lin(gathered, B + ".Transition2", attention, qRows, Wd, C);
  run1d("af3_gated_residual", qRows * C, GatedResidualArgs{act, attention, bc.zg, qRows * C, q1 * C});
  half* tn = scratch<half>("ab.tn", qRows * C);
  adaLn(st.noResidual ? pre : act, bc.ffwScale, bc.ffwShift, tn, nullptr, qRows, C, q1);
  int I = C * 2;
  half* gated = scratch<half>("ab.gated", qRows * I);
  gemmSwiglu(tn, swigluPairs(B + ".ffwTransition1", C, I), gated, qRows, C, I);
  if (hasW(B + ".ffwAToB")) {          // boltz2's up-gate: silu(a) b u
    half* u = scratch<half>("ab.up", qRows * I);
    linH(tn, B + ".ffwAToB", u, qRows, C, I);
    run1d("af3_mul_h", qRows * I, MulHArgs{gated, u, qRows * I});
  }
  float* projected = scratch<float>("ab.projected", qRows * C);
  lin(gated, B + ".ffwTransition2", projected, qRows, I, C);
  run1d("af3_gated_residual", qRows * C, GatedResidualArgs{act, projected, bc.tg, qRows * C, q1 * C});
}

// ---------------------------------------------------------------- the encoder
// per-block pair logits off the pair conditioning: LN (scale only), one projection to blocks x heads, laid out per block
std::vector<float*> atomPairLogits(const std::string& P, const float* pair, size_t pairRows, int Cp, int nblocks, int heads,
                                   const AtomShape& sh) {
  std::vector<float*> out;
  size_t per = (size_t)sh.subsets * heads * sh.queries * sh.keys;
  half* pn = scratch<half>("apl.pn", pairRows * Cp);
  if (flag(P + ".pairNormPerBlock")) {      // a block its own LayerNorm scale and projection (OpenDDE, protenix2)
    float* flat = scratch<float>("apl.flat", pairRows * heads);
    for (int b = 0; b < nblocks; ++b) {
      ln(pair, pn, pairRows, Cp, P + ".pairInputLayerNormScales." + num(b), "");
      lin(pn, P + ".pairLogitsProjections." + num(b), flat, pairRows, Cp, heads);
      float* pl = scratch<float>(P + ".pl" + num(b), per);
      run1d("af3_atom_logits", per, AtomLogitsArgs{flat, pl, 0, 1, (uint)sh.subsets, (uint)heads, (uint)sh.queries, (uint)sh.keys});
      out.push_back(pl);
    }
    return out;
  }
  ln(pair, pn, pairRows, Cp, P + ".pairInputLayerNormScale", "");
  float* flat = scratch<float>("apl.flat", pairRows * nblocks * heads);
  lin(pn, P + ".pairLogitsProjection", flat, pairRows, Cp, nblocks * heads);
  for (int b = 0; b < nblocks; ++b) {
    float* pl = scratch<float>(P + ".pl" + num(b), per);
    run1d("af3_atom_logits", per, AtomLogitsArgs{flat, pl, (uint)b, (uint)nblocks, (uint)sh.subsets, (uint)heads, (uint)sh.queries,
                                                 (uint)sh.keys});
    out.push_back(pl);
  }
  return out;
}

// The encoder's per-fold part: everything but the activation. trunkSingle [tokens][Cs] and trunkPair [tokens^2][Cz] may
// be null (target_feat).
EncoderOut prepareEncoder(const std::string& E, const std::string& refPrefix, const float* trunkSingle, const float* trunkPair) {
  AtomShape sh = atomShape();
  EncoderOut o{};
  o.C = metaI(E + ".channels"); int Cp = metaI(E + ".pairChannels");
  o.heads = metaI(E + ".heads"); o.D = metaI(E + ".dimension"); o.perToken = metaI(E + ".perTokenChannels");
  int C = o.C;
  if (flag("trunk.dialect.chaiAtomStack")) die("chai-1's atom stack is not in the native port yet");
  size_t atoms = (size_t)sh.tokens * sh.dense, qRows = (size_t)sh.subsets * sh.queries, kRows = (size_t)sh.subsets * sh.keys;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2k = gatherOf("batch.queriesToKeys");
  if ((size_t)t2q.count != qRows || (size_t)q2k.count != kRows) die("atom gathers %d/%d against %zu/%zu rows", t2q.count, q2k.count, qRows, kRows);
  float* cond = perAtomConditioning(refPrefix, (int)atoms);
  o.qCond = scratch<float>(E + ".qCond", qRows * C);
  convert(t2q, cond, o.qCond, C);
  o.qMask = scratch<float>(E + ".qMask", qRows);
  convert(t2q, M.f("batch.refMask"), o.qMask, 1);
  o.qStart = o.qCond;
  if (flag("trunk.dialect.preTrunkQuery") && trunkSingle) {
    // the queries start from the per-atom features alone (rf3, boltz2); every adaptive LN reads the full conditioning
    o.qStart = scratch<float>(E + ".qStart", qRows * C);
    copy(o.qStart, o.qCond, qRows * C * 4);
    scaleRows(o.qStart, o.qMask, qRows, C, qRows);
  }
  if (trunkSingle) {
    int Cs = metaI(E + ".trunkSingleChannels");
    float* lnS = scratch<float>("enc.tsln", (size_t)sh.tokens * Cs);
    ln(trunkSingle, lnS, sh.tokens, Cs, E + ".lnormTrunkSingleCondScale", E + ".lnormTrunkSingleCondOffset");
    float* proj = scratch<float>("enc.tsproj", (size_t)sh.tokens * C);
    lin(lnS, E + ".embedTrunkSingleCond", proj, sh.tokens, Cs, C);
    float* perQuery = scratch<float>("enc.perQuery", qRows * C);
    convert(gatherOf("batch.tokensToQueries"), proj, perQuery, C);
    add(o.qCond, perQuery, qRows * C);
  }
  scaleRows(o.qCond, o.qMask, qRows, C, qRows);
  o.kCond = scratch<float>(E + ".kCond", kRows * C);
  convert(q2k, o.qCond, o.kCond, C);
  o.kMask = scratch<float>(E + ".kMask", kRows);
  convert(q2k, o.qMask, o.kMask, 1);
  // the pair conditioning
  half* rq = scratch<half>("enc.rq", qRows * C); half* rk = scratch<half>("enc.rk", kRows * C);
  run1d("af3_relu_h", qRows * C, ReluHArgs{o.qCond, rq, qRows * C});
  run1d("af3_relu_h", kRows * C, ReluHArgs{o.kCond, rk, kRows * C});
  float* row = scratch<float>("enc.row", qRows * Cp); float* col = scratch<float>("enc.col", kRows * Cp);
  lin(rq, E + ".singleToPairCondRow", row, qRows, C, Cp);
  lin(rk, E + ".singleToPairCondCol", col, kRows, C, Cp);
  float* tp = nullptr;
  if (trunkPair) {
    int Cz = metaI(E + ".trunkPairChannels");
    size_t pairs = (size_t)sh.tokens * sh.tokens;
    half* lnP = scratch<half>("enc.tpln", pairs * Cz);
    ln(trunkPair, lnP, pairs, Cz, E + ".lnormTrunkPairCondScale", E + ".lnormTrunkPairCondOffset");
    tp = scratch<float>("enc.tp", pairs * Cp);
    lin(lnP, E + ".embedTrunkPairCond", tp, pairs, Cz, Cp);
  }
  float* qPos = scratch<float>("enc.qPos", qRows * 3); float* kPos = scratch<float>("enc.kPos", kRows * 3);
  o.qUid = scratch<float>(E + ".qUid", qRows); o.kUid = scratch<float>(E + ".kUid", kRows);
  convert(t2q, M.f("batch.refPos"), qPos, 3);
  convert(q2k, qPos, kPos, 3);
  {   // the batch's own reference-space uids, as floats
    size_t nu = M.len("batch.refSpaceUid");
    std::vector<float> u(nu);
    const int* hu = M.hostI("batch.refSpaceUid");
    for (size_t i = 0; i < nu; ++i) u[i] = (float)hu[i];
    float* uidF = scratch<float>("enc.uid", nu);
    upload(uidF, u.data(), nu * 4);
    convert(t2q, uidF, o.qUid, 1);
    convert(q2k, o.qUid, o.kUid, 1);
  }
  size_t pairRows = qRows * sh.keys;
  o.pair = scratch<float>(E + ".pair", pairRows * Cp);
  Gather tq = gatherOf("batch.tokensToQueries"), tk = gatherOf("batch.tokensToKeys");
  run1d("af3_atom_pair", pairRows * Cp,
        AtomPairArgs{row, col, qPos, kPos, o.qUid, o.kUid, o.kMask, W(E + ".embedPairOffsets"), W(E + ".embedPairDistances"),
                     W(E + ".embedPairOffsetsValid"), tp, tq.idx, tq.mask, tk.idx, tk.mask, o.pair, (uint)sh.subsets, (uint)sh.queries,
                     (uint)sh.keys, (uint)Cp, (uint)sh.tokens, flag("trunk.dialect.maskPaddedKeys") ? 1u : 0u});
  {   // the pair MLP: pair += mlp3(relu(mlp2(relu(mlp1(relu(pair))))))
    half* h1 = scratch<half>("enc.h1", pairRows * Cp); half* h2 = scratch<half>("enc.h2", pairRows * Cp);
    float* f = scratch<float>("enc.hf", pairRows * Cp);
    run1d("af3_relu_h", pairRows * Cp, ReluHArgs{o.pair, h1, pairRows * Cp});
    lin(h1, E + ".pairMlp1", f, pairRows, Cp, Cp);
    run1d("af3_relu_h", pairRows * Cp, ReluHArgs{f, h2, pairRows * Cp});
    lin(h2, E + ".pairMlp2", f, pairRows, Cp, Cp);
    run1d("af3_relu_h", pairRows * Cp, ReluHArgs{f, h1, pairRows * Cp});
    lin(h1, E + ".pairMlp3", o.pair, pairRows, Cp, Cp, 1.f);
  }
  int nblocks = 0; while (M.has(E + ".blocks." + num(nblocks) + ".qProjection")) ++nblocks;
  std::vector<float*> logits = atomPairLogits(E, o.pair, pairRows, Cp, nblocks, o.heads, sh);
  for (int b = 0; b < nblocks; ++b) o.blocks.push_back(prepareAtomBlock(E + ".blocks." + num(b), o.qCond, qRows, C, logits[b]));
  o.keyMasked = flag(E + ".blocks.0.keyMaskedAtomAttention");
  o.noResidual = flag(E + ".blocks.0.diffusionNoResidual");
  releaseScratch({"enc.", "apl.", "ada."});
  return o;
}
// rf3's chirality centres inverted: each atom's (centre, corner) entries, built once an input
struct ChiralIndex { const int* offsets; const int* entries; const void* input; };
static ChiralIndex CHIRAL{};
static const ChiralIndex& chiralIndex(size_t atoms) {
  const int* centers = M.hostI("chiral.centers");
  if (CHIRAL.input == (const void*)M.i("chiral.centers")) return CHIRAL;
  int count = (int)M.meta("chiral.count");
  std::vector<int> offsets(atoms + 1, 0), entries((size_t)count * 4);
  for (int i = 0; i < count * 4; ++i) offsets[centers[i] + 1]++;
  for (size_t a = 0; a < atoms; ++a) offsets[a + 1] += offsets[a];
  std::vector<int> fillAt(offsets.begin(), offsets.end() - 1);
  for (int i = 0; i < count * 4; ++i) entries[fillAt[centers[i]]++] = i;    // (centre << 2) | corner
  if (CHIRAL.offsets) { release(CHIRAL.offsets); release(CHIRAL.entries); }
  CHIRAL = {uploadNew(offsets.data(), offsets.size()), uploadNew(entries.data(), std::max<size_t>(1, entries.size())), M.i("chiral.centers")};
  return CHIRAL;
}
// The encoder's per-step part: the activation from (scaled) positions, the blocks, the per-token aggregation
void encoderStep(const std::string& E, EncoderOut& o, const float* atomPositions) {
  AtomShape sh = atomShape();
  int C = o.C;
  size_t atoms = (size_t)sh.tokens * sh.dense, q1 = (size_t)sh.subsets * sh.queries, qRows = q1 * NS;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2t = gatherOf("batch.queriesToTokenAtoms");
  float* act = scratch<float>(E + ".act", qRows * C);
  bool maskPerBlock = flag(E + ".blocks.0.maskAtomActPerBlock");
  bool chiral = atomPositions && hasW(E + ".atomChiralToFeatures");
  if (chiral) {
    // rf3's: the positions' projection and the chirality gradient's beside it, masked, added to the start
    for (int k = 0; k < NS; ++k) copy(act + k * q1 * C, o.qStart, q1 * C * 4);
    float* gp = scratch<float>("enc.gp", qRows * 3);
    convert(t2q, atomPositions, gp, 3, atoms, NS);
    float* positional = scratch<float>("enc.positional", qRows * C);
    lin(gp, E + ".atomPositionsToFeatures", positional, qRows, 3, C);
    if (!M.has("chiral.centers")) die("this bundle reads chirality centres the input lacks: export it again");
    const ChiralIndex& ch = chiralIndex(atoms);
    float* grads = scratch<float>("enc.chiralGrads", atoms * NS * 3);
    run1d("af3_chiral_grad", atoms * NS, ChiralGradArgs{atomPositions, M.i("chiral.centers"), M.f("chiral.angles"), ch.offsets, ch.entries,
                                                        grads, atoms, (uint)NS, 0});
    float* gc = scratch<float>("enc.gc", qRows * 3);
    convert(t2q, grads, gc, 3, atoms, NS);
    lin(gc, E + ".atomChiralToFeatures", positional, qRows, 3, C, 1.f);
    scaleRows(positional, o.qMask, qRows, C, q1);
    add(act, positional, qRows * C);
  } else if (atomPositions) {
    if (lenW(E + ".atomPositionsToFeatures") != (size_t)3 * C) die("encoder: the positions projection is not 3 x C");
    run1d("af3_encoder_start", qRows * C, EncoderStartArgs{o.qStart, atomPositions, t2q.idx, t2q.mask, W(E + ".atomPositionsToFeatures"),
                                                          o.qMask, act, qRows, q1, atoms, (uint)C, 0});
  } else {
    for (int k = 0; k < NS; ++k) copy(act + k * q1 * C, o.qStart, q1 * C * 4);
  }
  AtomStep st{gatherOf("batch.queriesToKeys"), o.qMask, o.kMask, o.keyMasked, o.noResidual};
  for (size_t b = 0; b < o.blocks.size(); ++b) {
    if (maskPerBlock) scaleRows(act, o.qMask, qRows, C, q1);
    crossAttentionBlock(act, st, o.blocks[b], sh, C, o.heads, o.D, E + ".blocks." + num(b));
  }
  scaleRows(act, o.qMask, qRows, C, q1);
  o.skip = act;
  float* projected = scratch<float>("enc.aggr", qRows * o.perToken);
  lin(act, E + ".projectAtomFeaturesForAggr", projected, qRows, C, o.perToken);
  o.tokenAct = scratch<float>(E + ".tokenAct", (size_t)sh.tokens * NS * o.perToken);
  run1d("af3_aggregate", (size_t)sh.tokens * NS * o.perToken,
        AggregateArgs{projected, q2t.idx, q2t.mask, M.f("batch.refMask"), o.tokenAct, (u64)sh.tokens * NS, q1, (uint)sh.tokens,
                      (uint)sh.dense, (uint)o.perToken, 0});
}

// ---------------------------------------------------------------- target_feat
float* buildTargetFeat() {
  int tokens = (int)M.meta("batch.tokens");
  if (flag("trunk.dialect.chaiTokenEmbedding")) die("chai-1's token features are not in the native port yet");
  EncoderOut e = prepareEncoder("targetFeat.encoder", "targetFeat.reference", nullptr, nullptr);
  encoderStep("targetFeat.encoder", e, nullptr);
  float* tf;
  if (flag("trunk.dialect.targetFeatAtomOnly")) {      // boltz2: the atom columns plus six summed projections
    const std::string S = "targetFeat.encoder.targetFeatSum.";
    int C = 384;
    tf = scratch<float>("targetFeat", (size_t)tokens * C);
    copy(tf, e.tokenAct, (size_t)tokens * C * 4);
    auto flagged = [](const char* k) -> const int* { return M.has(k) ? M.i(k) : nullptr; };
    run1d("af3_target_feat_sum", (size_t)tokens * C,
          TargetFeatSumArgs{tf, M.i("batch.aatype"), M.f("batch.profile"), M.f("batch.deletionMean"), flagged("batch.isDna"),
                            flagged("batch.isRna"), flagged("batch.isLigand"), flagged("batch.isModified"), W(S + "resType"),
                            W(S + "msaProfile"), W(S + "molType"), W(S + "method"), W(S + "modified"), (uint)tokens, (uint)C});
  } else {
    tf = scratch<float>("targetFeat", (size_t)tokens * 447);
    run1d("af3_target_feat", (size_t)tokens * 447,
          TargetFeatArgs{M.i("batch.aatype"), M.f("batch.profile"), M.f("batch.deletionMean"), e.tokenAct, tf, (uint)tokens, 0});
  }
  releaseScratch({"targetFeat.encoder.", "targetFeat.reference.", "ab.", "enc."});
  return tf;
}
