// The per-atom conditioning and the atom cross-attention encoder - which builds target_feat's 384 atom columns and
// runs again inside every denoiser call - and its block, which the decoder shares (cuda/af3/src/atom.cuh is the
// reading). Atoms are attended in subsets: 32 queries against 128 keys each, biased per block by the atom pair.
#include "af3.h"
#include <cmath>

int NS = 1;
float* TARGET_FEAT_STRUCTURE = nullptr;
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
void adaLn(float* x, const float* scale, const float* shift, half* out, float* outF, size_t rows, int C, size_t period, int ld,
           const float* kScale, const float* kShift, half* kOut, bool chained, const float* resY, const float* resGate, int resLd,
           const half* resYh, float resScale, bool preSigmoid, const float* fScale, const float* fShift, half* fOut,
           const float* rowMask) {
  const uint ldu = (uint)(ld ? ld : C), rld = (uint)(resLd ? resLd : C);
  const bool k128 = !outF && C == 128 && (ldu & 3) == 0 && (rld & 3) == 0;
  const bool k768 = !outF && !chained && C == 768 && (ldu & 3) == 0 && (rld & 3) == 0 && !rowMask;
  const uint32_t sgs = rows <= 256 ? 2 : 8;
  run(k128 ? "af3_ada_ln_128" : k768 ? "af3_ada_ln_768" : "af3_ada_ln", grid1d((rows + sgs - 1) / sgs, 1), sgs * 32,
      AdaLnArgs{x, scale, shift, out, outF, kScale, kShift, kOut, resY, resYh, resGate, fScale, fShift, fOut, rowMask, rows,
                (uint)C, (uint)period, ldu, ((ADA_RAW || preSigmoid) ? 1u : 0u) | (preSigmoid ? 2u : 0u),
                ADA_RAW ? 0.1f : 1e-5f, chained ? 1u : 0u, rld, resScale});
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
  if (ADA_RAW) {      // chai: the conditioning as it is, scale = cond Ws + 1 (no bias), shift = cond Wb
    lin(cond, w + "SingleCondScaleWeights", sc, rows, condC, C);
    run1d("af3_add_const", rows * C, AddConstArgs{sc, rows * C, 1.f, 0});
    lin(cond, w + "SingleCondBias", sh, rows, condC, C);
    return;
  }
  float* cn = scratch<float>("ada.cn", rows * condC);
  ln(cond, cn, rows, condC, w + "SingleCondLayerNormScale", "");
  lin(cn, w + "SingleCondScaleWeights", sc, rows, condC, C, 0.f, W(w + "SingleCondScaleBias"));
  lin(cn, w + "SingleCondBias", sh, rows, condC, C);
}
AtomBlockCache prepareAtomBlock(const std::string& B, const float* qCond, size_t qRows, int C, float* pairLogits) {
  AtomBlockCache c{};
  const size_t n = qRows * C;
  float* s6 = scratch<float>(B + ".s6", 6 * n);
  c.qScale = s6; c.qShift = s6 + n; c.kScale = s6 + 2 * n; c.kShift = s6 + 3 * n;
  c.ffwScale = s6 + 4 * n; c.ffwShift = s6 + 5 * n;
  c.zg = scratch<float>(B + ".zg", n); c.tg = scratch<float>(B + ".tg", n);
  adaCond(qCond, qRows, C, C, B + ".q", c.qScale, c.qShift);
  adaCond(qCond, qRows, C, C, B + ".k", c.kScale, c.kShift);
  adaCond(qCond, qRows, C, C, B + ".ffw", c.ffwScale, c.ffwShift);
  lin(qCond, B + ".AdaptiveZeroCondWeights", c.zg, qRows, C, C, 0.f, W(B + ".AdaptiveZeroCondBias"));
  lin(qCond, B + ".ffwAdaptiveZeroCondWeights", c.tg, qRows, C, C, 0.f, W(B + ".ffwAdaptiveZeroCondBias"));
  run1d("af3_sigmoid_atom", qRows * C, SigmoidAtomArgs{c.qScale, c.kScale, c.ffwScale, c.zg, c.tg, qRows * C, ADA_RAW ? 1u : 0u, 0});
  c.pairLogits = pairLogits;
  c.chained = flag(B + ".chainedAtomLayerNorm");
  return c;
}

// ---------------------------------------------------------------- one cross-attention block, in place on act
void crossAttentionBlock(float* act, const AtomStep& st, const AtomBlockCache& bc, const AtomShape& sh, int C, int heads, int D,
                         const std::string& B, const float** pendY, const float** pendGate, bool deferFfwRes, bool precomputedQk,
                         const AtomBlockCache* nextBc, bool* emittedNextQk, half* finalActH, bool* scaledFinal) {
  if (emittedNextQk) *emittedNextQk = false;
  if (scaledFinal) *scaledFinal = false;
  size_t q1 = (size_t)sh.subsets * sh.queries, qRows = q1 * NS;
  int Wd = heads * D;
  if (D > 32 || sh.keys > 128) die("atom attention: %d keys of a head %d wide", sh.keys, D);
  const float* py = pendY ? *pendY : nullptr; const float* pg = pendGate ? *pendGate : nullptr;
  if (pendY) *pendY = nullptr; if (pendGate) *pendGate = nullptr;
  const bool fuseFfw = st.noResidual && C <= 768;
  float* pre = nullptr;
  if (st.noResidual && !fuseFfw) { pre = scratch<float>("ab.pre", qRows * C); copy(pre, act, qRows * C * 4); }
  half* xqk = scratch<half>("ab.xqk", 2 * qRows * C);
  half* xq = xqk; half* xk = xqk + qRows * C;
  half* tn = scratch<half>("ab.tn", qRows * C);
  const float* rMask = st.maskPerBlock ? st.qMask : nullptr;
  if (!precomputedQk) {
    if (C <= 768) {
      adaLn(act, bc.qScale, bc.qShift, xq, nullptr, qRows, C, q1, 0, bc.kScale, bc.kShift, xk, bc.chained, py, pg, C,
            nullptr, 1.f, true, fuseFfw ? bc.ffwScale : nullptr, fuseFfw ? bc.ffwShift : nullptr, fuseFfw ? tn : nullptr, rMask);
    } else if (bc.chained) {  // xk = adaLN_k(adaLN_q(x)): the queries normalised in f32 first (OpenDDE, protenix2)
      float* xqF = scratch<float>("ab.xqF", qRows * C);
      adaLn(act, bc.qScale, bc.qShift, xq, xqF, qRows, C, q1, 0, nullptr, nullptr, nullptr, false, nullptr, nullptr, 0,
            nullptr, 1.f, true, nullptr, nullptr, nullptr, rMask);
      adaLn(xqF, bc.kScale, bc.kShift, xk, nullptr, qRows, C, q1);
    } else {
      adaLn(act, bc.qScale, bc.qShift, xq, nullptr, qRows, C, q1, 0, nullptr, nullptr, nullptr, false, nullptr, nullptr, 0,
            nullptr, 1.f, true, nullptr, nullptr, nullptr, rMask);
      adaLn(act, bc.kScale, bc.kShift, xk, nullptr, qRows, C, q1);
    }
  }
  half* qgkv = scratch<half>("ab.qgkv", 2 * qRows * 2 * Wd);
  half* qg = qgkv; half* kvAtom = qgkv + qRows * 2 * Wd;
  { Gemm g{}; g.X = xqk; g.tx = F16; g.sx = (int64_t)(qRows * C); g.W = qgkvWeight(B, C, Wd); g.tw = F16; g.sw = (int64_t)C * 2 * Wd;
    g.Y = qgkv; g.ty = F16; g.sy = (int64_t)(qRows * 2 * Wd); g.rows = qRows; g.in = C; g.out = 2 * Wd; g.batch = 2;
    g.accFloat = true; g.label = "atom qg kv"; gemm(g); }
  const float* qBias = W(B + ".qBias");
  if (hasW(B + ".queryLayerNormScale")) {       // rf3: q (its bias inside) and k normalised per atom row
    uint32_t sgs = 2 * qRows <= 512 ? 2 : 8;
    run("af3_kq_norm", grid1d((2 * qRows + sgs - 1) / sgs, 1), sgs * 32,
        KqNormArgs{qg, kvAtom, qBias, W(B + ".queryLayerNormScale"), W(B + ".queryLayerNormOffset"), W(B + ".keyLayerNormScale"),
                   W(B + ".keyLayerNormOffset"), qRows, (uint)(2 * Wd), (uint)(2 * Wd), (uint)Wd, 0});
    qBias = M.derived<float>("zeros:" + num(Wd), Wd, [](float*) {});
  }
  const bool hasT2 = hasW(B + ".Transition2");
  const bool directAct = !hasT2 && st.noResidual && Wd == C;
  half* gathered = directAct ? nullptr : scratch<half>("ab.gathered", qRows * Wd);
  static const bool scalarAttention = (getenv("AF3_SCALAR_ATOM_ATTENTION") && atoi(getenv("AF3_SCALAR_ATOM_ATTENTION")) != 0);      // (the control arm)
  const bool mma = !scalarAttention && sh.queries == 32 && sh.keys <= 128 && (D == 16 || D == 32);
  run(mma ? (D == 16 ? "af3_atom_attention_mma16" : "af3_atom_attention_mma32") : "af3_atom_attention",
      Grid{(uint32_t)(sh.subsets * NS), (uint32_t)heads, 1}, 128,
      AtomAttnArgs{qg, qBias, kvAtom, st.queriesToKeys.idx, st.queriesToKeys.mask, st.qMask, st.kMask, bc.pairLogits, gathered,
                   directAct ? act : nullptr, directAct ? bc.zg : nullptr,
                   (uint)sh.queries, (uint)sh.keys, (uint)heads, (uint)D, (uint)sh.subsets, st.keyMasked ? 1u : 0u});
  if (hasT2) {
    bool emittedTn = false;
    if (!gemmGateResidual(gathered, Wh(B + ".Transition2"), act, qRows, qRows, Wd, C, bc.zg, C, q1, 1.f,
                          "atom output, gated residual", true,
                          !st.noResidual && C == 128 ? tn : nullptr, bc.ffwScale, bc.ffwShift, ADA_RAW ? 0.1f : 1e-5f,
                          nullptr, nullptr, nullptr, false, &emittedTn)) {
      float* attention = scratch<float>("ab.attention", qRows * C);
      lin(gathered, B + ".Transition2", attention, qRows, Wd, C);
      if (!st.noResidual) {
        adaLn(act, bc.ffwScale, bc.ffwShift, tn, nullptr, qRows, C, q1, 0, nullptr, nullptr, nullptr, false, attention, bc.zg, C);
      } else {
        run1d("af3_gated_residual", qRows * C, GatedResidualArgs{act, attention, nullptr, bc.zg, qRows * C, q1 * C, 1.f, 0});
        if (!fuseFfw) adaLn(pre, bc.ffwScale, bc.ffwShift, tn, nullptr, qRows, C, q1);
      }
    } else if (!fuseFfw && !emittedTn) {
      adaLn(st.noResidual ? pre : act, bc.ffwScale, bc.ffwShift, tn, nullptr, qRows, C, q1);
    }
  } else if (!st.noResidual) {
    adaLn(act, bc.ffwScale, bc.ffwShift, tn, nullptr, qRows, C, q1, 0, nullptr, nullptr, nullptr, false, nullptr, bc.zg, C, gathered, 2.f);
  } else {
    if (!directAct) run1d("af3_gated_residual", qRows * C, GatedResidualArgs{act, nullptr, gathered, bc.zg, qRows * C, q1 * C, 2.f, 0});
    if (!fuseFfw) adaLn(pre, bc.ffwScale, bc.ffwShift, tn, nullptr, qRows, C, q1);
  }
  int I = C * 2;
  half* gated = scratch<half>("ab.gated", qRows * I);
  gemmSwiglu(tn, swigluPairs(B + ".ffwTransition1", C, I), gated, qRows, C, I);
  if (hasW(B + ".ffwAToB")) {          // boltz2's up-gate: silu(a) b u
    if (!gemmMulH(tn, Wh(B + ".ffwAToB"), gated, qRows, C, I, "atom up-gate")) {
      half* u = scratch<half>("ab.up", qRows * I);
      linH(tn, B + ".ffwAToB", u, qRows, C, I);
      run1d("af3_mul_h", qRows * I, MulHArgs{gated, u, qRows * I});
    }
  }
  const bool wantNextQk = nextBc && C == 128;
  const float* resMask = wantNextQk ? rMask : (finalActH ? st.qMask : nullptr);
  half* resOut3 = (wantNextQk && st.noResidual) ? tn : ( wantNextQk ? nullptr : finalActH );
  if (!gemmGateResidual(gated, Wh(B + ".ffwTransition2"), act, qRows, qRows, I, C, bc.tg, C, q1, 1.f,
                        "atom transition, gated residual", true,
                        wantNextQk ? xq : nullptr, wantNextQk ? nextBc->qScale : nullptr, wantNextQk ? nextBc->qShift : nullptr,
                        ADA_RAW ? 0.1f : 1e-5f,
                        wantNextQk ? xk : nullptr, wantNextQk ? nextBc->kScale : nullptr, wantNextQk ? nextBc->kShift : nullptr,
                        wantNextQk && nextBc->chained, emittedNextQk, resMask, resOut3)) {
    float* projected = scratch<float>("ab.projected", qRows * C);
    lin(gated, B + ".ffwTransition2", projected, qRows, I, C);
    if (deferFfwRes && !st.noResidual && C <= 768 && pendY && pendGate) {
      *pendY = projected; *pendGate = bc.tg;
    } else {
      run1d("af3_gated_residual", qRows * C, GatedResidualArgs{act, projected, nullptr, bc.tg, qRows * C, q1 * C, 1.f, 0});
    }
  } else if (finalActH && scaledFinal) {
    *scaledFinal = true;
  }
}

// ---------------------------------------------------------------- the encoder
// per-block pair logits off the pair conditioning: LN (scale only), one projection to blocks x heads, laid out per block
std::vector<float*> atomPairLogits(const std::string& P, const float* pair, size_t pairRows, int Cp, int nblocks, int heads,
                                   const AtomShape& sh, const float* qMask, const float* kMask, bool keyMasked, int D,
                                   const float* qUid, const float* kUid) {
  std::vector<float*> out;
  size_t per = (size_t)sh.subsets * heads * sh.queries * sh.keys;
  static const bool scalarAttention = (getenv("AF3_SCALAR_ATOM_ATTENTION") && atoi(getenv("AF3_SCALAR_ATOM_ATTENTION")) != 0);
  const uint log2Scale = (!scalarAttention && sh.queries == 32 && sh.keys <= 128 && (D == 16 || D == 32)) ? 1u : 0u;
  Gather tq = qUid ? gatherOf("batch.tokensToQueries") : Gather{};
  Gather tk = qUid ? gatherOf("batch.tokensToKeys") : Gather{};
  half* pn = scratch<half>("apl.pn", pairRows * Cp);
  if (flag(P + ".pairNormPerBlock")) {      // a block its own LayerNorm scale and projection (OpenDDE, protenix2)
    float* flat = scratch<float>("apl.flat", pairRows * heads);
    for (int b = 0; b < nblocks; ++b) {
      ln(pair, pn, pairRows, Cp, P + ".pairInputLayerNormScales." + num(b), "");
      lin(pn, P + ".pairLogitsProjections." + num(b), flat, pairRows, Cp, heads);
      float* pl = scratch<float>(P + ".pl" + num(b), per);
      run1d("af3_atom_logits", per,
            AtomLogitsArgs{flat, qMask, kMask, qUid, tq.mask, kUid, tk.mask, pl, 0, 1, (uint)sh.subsets, (uint)heads,
                           (uint)sh.queries, (uint)sh.keys, keyMasked ? 1u : 0u, log2Scale});
      out.push_back(pl);
    }
    return out;
  }
  ln(pair, pn, pairRows, Cp, P + ".pairInputLayerNormScale", "");
  float* flat = scratch<float>("apl.flat", pairRows * nblocks * heads);
  lin(pn, P + ".pairLogitsProjection", flat, pairRows, Cp, nblocks * heads);
  for (int b = 0; b < nblocks; ++b) {
    float* pl = scratch<float>(P + ".pl" + num(b), per);
    run1d("af3_atom_logits", per,
          AtomLogitsArgs{flat, qMask, kMask, qUid, tq.mask, kUid, tk.mask, pl, (uint)b, (uint)nblocks, (uint)sh.subsets,
                         (uint)heads, (uint)sh.queries, (uint)sh.keys, keyMasked ? 1u : 0u, log2Scale});
    out.push_back(pl);
  }
  return out;
}

void sameRefMask(float* pl, const float* qUid, const float* kUid, int heads, const AtomShape& sh) {
  Gather tq = gatherOf("batch.tokensToQueries"), tk = gatherOf("batch.tokensToKeys");
  size_t per = (size_t)sh.subsets * heads * sh.queries * sh.keys;
  run1d("af3_same_ref_mask", per, SameRefMaskArgs{pl, qUid, tq.mask, kUid, tk.mask, (uint)sh.subsets, (uint)heads, (uint)sh.queries, (uint)sh.keys});
}
// The encoder's per-fold part: everything but the activation. trunkSingle [tokens][Cs] and trunkPair [tokens^2][Cz] may
// be null (target_feat).
EncoderOut prepareEncoder(const std::string& E, const std::string& refPrefix, const float* trunkSingle, const float* trunkPair) {
  AtomShape sh = atomShape();
  EncoderOut o{};
  o.C = metaI(E + ".channels"); int Cp = metaI(E + ".pairChannels");
  o.heads = metaI(E + ".heads"); o.D = metaI(E + ".dimension"); o.perToken = metaI(E + ".perTokenChannels");
  int C = o.C;
  const bool chai = flag("trunk.dialect.chaiAtomStack");
  size_t atoms = (size_t)sh.tokens * sh.dense, qRows = (size_t)sh.subsets * sh.queries, kRows = (size_t)sh.subsets * sh.keys;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2k = gatherOf("batch.queriesToKeys");
  if ((size_t)t2q.count != qRows || (size_t)q2k.count != kRows) die("atom gathers %d/%d against %zu/%zu rows", t2q.count, q2k.count, qRows, kRows);
  float* cond = perAtomConditioning(refPrefix, (int)atoms);
  o.qCond = scratch<float>(E + ".qCond", qRows * C);
  convert(t2q, cond, o.qCond, C);
  o.qMask = scratch<float>(E + ".qMask", qRows);
  convert(t2q, M.f("batch.refMask"), o.qMask, 1);
  o.qStart = o.qCond;
  if (flag("trunk.dialect.preTrunkQuery") && (trunkSingle || chai)) {
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
  if (chai) {     // chai: cond = LN(a [+ the trunk term]), affine-free, before the mask
    float* nrm = scratch<float>("enc.condLn", qRows * C);
    layerNorm(o.qCond, nrm, qRows, C, nullptr, nullptr);
    copy(o.qCond, nrm, qRows * C * 4);
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
    // (stock AF3: the keys' uid gathered with queries_to_keys from the TOKEN layout - dialect.js keysUidFromQueriesLayout off;
    // the index clamped, as a masked entry may point past the end)
    if (!flag("trunk.dialect.keysUidFromQueriesLayout")) {
      const int* qi = M.hostI("batch.queriesToKeys.indices"); const float* qm = M.hostF("batch.queriesToKeys.mask");
      size_t nk = M.len("batch.queriesToKeys.indices");
      std::vector<float> k(nk);
      for (size_t i = 0; i < nk; ++i) k[i] = qm[i] != 0 ? u[std::min<size_t>((size_t)qi[i], nu - 1)] : 0.f;
      upload(o.kUid, k.data(), nk * 4);
    } else convert(q2k, o.qUid, o.kUid, 1);
  }
  size_t pairRows = qRows * sh.keys;
  o.pair = scratch<float>(E + ".pair", pairRows * Cp);
  Gather tq = gatherOf("batch.tokensToQueries"), tk = gatherOf("batch.tokensToKeys");
  if (chai)
    run1d("af3_chai_atom_pair", pairRows * Cp,
          ChaiAtomPairArgs{row, col, qPos, kPos, o.qUid, o.kUid, W(E + ".embedAtomPairFeat"), W(E + ".embedAtomPairFeatBias"), tp, tq.idx,
                           tq.mask, tk.idx, tk.mask, o.pair, (uint)sh.subsets, (uint)sh.queries, (uint)sh.keys, (uint)Cp, (uint)sh.tokens, 0});
  else
    run1d("af3_atom_pair", pairRows * Cp,
          AtomPairArgs{row, col, qPos, kPos, o.qUid, o.kUid, o.kMask, W(E + ".embedPairOffsets"), W(E + ".embedPairDistances"),
                       W(E + ".embedPairOffsetsValid"), tp, tq.idx, tq.mask, tk.idx, tk.mask, o.pair, (uint)sh.subsets, (uint)sh.queries,
                       (uint)sh.keys, (uint)Cp, (uint)sh.tokens, flag("trunk.dialect.maskPaddedKeys") ? 1u : 0u});
  if (chai) {     // chai's pair MLP: two layers, no relu on its input
    half* h1 = scratch<half>("enc.h1", pairRows * Cp); float* f = scratch<float>("enc.hf", pairRows * Cp);
    half* p16 = scratch<half>("enc.h2", pairRows * Cp);
    toHalf(o.pair, p16, pairRows * Cp);
    lin(p16, E + ".pairMlp1", f, pairRows, Cp, Cp);
    run1d("af3_relu_h", pairRows * Cp, ReluHArgs{f, h1, pairRows * Cp});
    lin(h1, E + ".pairMlp2", o.pair, pairRows, Cp, Cp, 1.f);
  } else {   // the pair MLP: pair += mlp3(relu(mlp2(relu(mlp1(relu(pair))))))
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
  o.keyMasked = flag(E + ".blocks.0.keyMaskedAtomAttention");
  o.noResidual = flag(E + ".blocks.0.diffusionNoResidual");
  std::vector<float*> logits = atomPairLogits(E, o.pair, pairRows, Cp, nblocks, o.heads, sh,
                                              o.qMask, o.kMask, o.keyMasked, o.D, chai ? o.qUid : nullptr, chai ? o.kUid : nullptr);
  for (int b = 0; b < nblocks; ++b) o.blocks.push_back(prepareAtomBlock(E + ".blocks." + num(b), o.qCond, qRows, C, logits[b]));
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
void encoderStep(const std::string& E, EncoderOut& o, const float* atomPositions, const float* atomMask, float posScale,
                 const float* tokenBias, const float* txScale, const float* txShift, half* txOut,
                 const float* txFfwScale, const float* txFfwShift, half* txFfwOut, int txLd, bool txPreSigmoid) {
  AtomShape sh = atomShape();
  int C = o.C;
  size_t atoms = (size_t)sh.tokens * sh.dense, q1 = (size_t)sh.subsets * sh.queries, qRows = q1 * NS;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2t = gatherOf("batch.queriesToTokenAtoms");
  float* act = scratch<float>(E + ".act", qRows * C);
  bool maskPerBlock = flag(E + ".blocks.0.maskAtomActPerBlock");
  bool chiral = atomPositions && hasW(E + ".atomChiralToFeatures");
  const bool fuse0 = atomPositions && C == 128 && !o.blocks.empty();
  if (atomPositions) {
    if (lenW(E + ".atomPositionsToFeatures") != (size_t)3 * C) die("encoder: the positions projection is not 3 x C");
    float* grads = nullptr;
    if (chiral) {
      if (!M.has("chiral.centers")) die("this bundle reads chirality centres the input lacks: export it again");
      const ChiralIndex& ch = chiralIndex(atoms);
      grads = scratch<float>("enc.chiralGrads", atoms * NS * 3);
      run1d("af3_chiral_grad", atoms * NS,
            ChiralGradArgs{atomPositions, atomMask, M.i("chiral.centers"), M.f("chiral.angles"), ch.offsets, ch.entries,
                           grads, atoms, (uint)NS, posScale});
    }
    half* xqk0 = fuse0 ? scratch<half>("ab.xqk", 2 * qRows * C) : nullptr;
    half* xq0 = xqk0;
    half* xk0 = fuse0 ? xqk0 + qRows * C : nullptr;
    half* tn0 = (fuse0 && o.noResidual) ? scratch<half>("ab.tn", qRows * C) : nullptr;
    const AtomBlockCache* b0 = fuse0 ? &o.blocks[0] : nullptr;
    run1d("af3_encoder_start", (C & 3) ? qRows * C : qRows * (C >> 2),
          EncoderStartArgs{o.qStart, atomPositions, atomMask, t2q.idx, t2q.mask, W(E + ".atomPositionsToFeatures"),
                           o.qMask, act,
                           b0 ? b0->qScale : nullptr, b0 ? b0->qShift : nullptr, xq0,
                           b0 ? b0->kScale : nullptr, b0 ? b0->kShift : nullptr, xk0,
                           tn0 ? b0->ffwScale : nullptr, tn0 ? b0->ffwShift : nullptr, tn0,
                           grads, chiral ? W(E + ".atomChiralToFeatures") : nullptr,
                           qRows, q1, atoms, (uint)C, posScale, ADA_RAW ? 0.1f : 1e-5f, b0 && b0->chained ? 1u : 0u});
  } else {
    for (int k = 0; k < NS; ++k) copy(act + k * q1 * C, o.qStart, q1 * C * 4);
  }
  AtomStep st{gatherOf("batch.queriesToKeys"), o.qMask, o.kMask, o.keyMasked, o.noResidual, maskPerBlock};
  const float* pendY = nullptr; const float* pendGate = nullptr;
  bool preQk = fuse0, scaledFinal = false;
  half* actH = scratch<half>("enc.actH", qRows * C);
  for (size_t b = 0; b < o.blocks.size(); ++b) {
    bool nextQk = false;
    const bool last = b + 1 == o.blocks.size();
    crossAttentionBlock(act, st, o.blocks[b], sh, C, o.heads, o.D, E + ".blocks." + num(b),
                        &pendY, &pendGate, true, preQk, last ? nullptr : &o.blocks[b + 1], &nextQk,
                        last ? actH : nullptr, last ? &scaledFinal : nullptr);
    preQk = nextQk;
  }
  if (!scaledFinal) scaleRows(act, o.qMask, qRows, C, q1, actH, pendY, pendGate);
  o.skip = act;
  float* projected = scratch<float>("enc.aggr", qRows * o.perToken);
  lin(actH, E + ".projectAtomFeaturesForAggr", projected, qRows, C, o.perToken);
  const size_t tokRows = (size_t)sh.tokens * NS;
  o.tokenAct = scratch<float>(E + ".tokenAct", tokRows * o.perToken);
  const uint32_t sgs = tokRows <= 256 ? 2 : 8;
  run("af3_aggregate", grid1d((tokRows + sgs - 1) / sgs, 1), sgs * 32,
      AggregateArgs{projected, q2t.idx, q2t.mask, M.f("batch.refMask"), tokenBias, o.tokenAct,
                    txScale, txShift, txOut, txFfwScale, txFfwShift, txFfwOut,
                    (u64)tokRows, q1, (uint)sh.tokens, (uint)sh.dense, (uint)o.perToken,
                    (uint)(txLd ? txLd : o.perToken), ADA_RAW ? 0.1f : 1e-5f, (ADA_RAW || txPreSigmoid) ? 1u : 0u});
}

// ---------------------------------------------------------------- target_feat
float* buildTargetFeat() {
  int tokens = (int)M.meta("batch.tokens");
  if (flag("trunk.dialect.chaiTokenEmbedding")) {
    // chai-1: s_cat = [pooled atoms | token features (+ ESM2 3B's projection)]; the trunk's target_feat = s_cat W_trunk
    // and the diffusion module's = s_cat W_structure (TARGET_FEAT_STRUCTURE)
    const std::string P = "targetFeat.encoder.chaiToken.";
    EncoderOut e = prepareEncoder("targetFeat.encoder", "targetFeat.reference", nullptr, nullptr);
    encoderStep("targetFeat.encoder", e, nullptr);
    int Cp = e.perToken, C = (int)lenW(P + "tokenFeatureBias");
    float* feat = scratch<float>("tf.token", (size_t)tokens * C);
    run1d("af3_chai_token_feat", (size_t)tokens * C,
          ChaiTokenFeatArgs{M.i("batch.aatype"), M.f("batch.profile"), M.f("batch.deletionMean"), W(P + "tokenFeatureWeights"),
                            W(P + "tokenFeatureBias"), W(P + "msaProfileWeights"), hasW(P + "dx") ? W(P + "dx") : nullptr,
                            M.has("batch.isDna") ? M.i("batch.isDna") : nullptr, feat, (uint)tokens, (uint)C});
    int E = 0;
    const float* esm;
    if (M.has("batch.esmEmbeddings")) { esm = M.f("batch.esmEmbeddings"); E = (int)(M.len("batch.esmEmbeddings") / tokens); }
    else {
      if (!M.has("esm.ids")) die("chai-1 reads ESM2 embeddings: the input has neither them nor esm.ids");
      if (!M.hasInt8("e/esm2/blocks/q/0/weights")) die("chai-1 reads ESM2: give --esm-bundle=<a directory holding af3-any-model's lm/esm2.bin.zst>");
      esm = esmEmbeddings(tokens, E);
    }
    lin(esm, P + "esmWeights", feat, tokens, E, C, 1.f);
    float* cat = scratch<float>("tf.cat", (size_t)tokens * (Cp + C));
    run1d("af3_concat_pad", (size_t)tokens * (Cp + C), ConcatPadArgs{e.tokenAct, feat, cat, (uint)tokens, (uint)Cp, (uint)C, -1, -1, 0});
    int out = (int)(lenW(P + "singleProjInTrunk") / (Cp + C));
    float* tf = scratch<float>("targetFeat", (size_t)tokens * out);
    lin(cat, P + "singleProjInTrunk", tf, tokens, Cp + C, out);
    if (TARGET_FEAT_STRUCTURE) release(TARGET_FEAT_STRUCTURE);
    TARGET_FEAT_STRUCTURE = allocT<float>((size_t)tokens * out);      // (read by the diffusion, after the trunk)
    lin(cat, P + "singleProjInStructure", TARGET_FEAT_STRUCTURE, tokens, Cp + C, out);
    releaseScratch({"targetFeat.encoder.", "targetFeat.reference.", "ab.", "enc.", "tf.", "esm."});
    return tf;
  }
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
                            W(S + "msaProfile"), W(S + "molType"), W(S + "method"), W(S + "modified"),
                            hasW(S + "dnResType") ? W(S + "dnResType") : nullptr, hasW(S + "dnMsaProfile") ? W(S + "dnMsaProfile") : nullptr,
                            (uint)tokens, (uint)C});
  } else {
    tf = scratch<float>("targetFeat", (size_t)tokens * 447);
    run1d("af3_target_feat", (size_t)tokens * 447,
          TargetFeatArgs{M.i("batch.aatype"), M.f("batch.profile"), M.f("batch.deletionMean"), e.tokenAct, tf, (uint)tokens, 0});
  }
  releaseScratch({"targetFeat.encoder.", "targetFeat.reference.", "ab.", "enc."});
  return tf;
}
