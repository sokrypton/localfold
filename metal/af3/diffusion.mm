// The diffusion head - its conditioning, the atom encoder, the conditioned token transformer, the atom decoder - and
// AF3's EDM sampler around it (cuda/af3/src/diffusion.cuh and sampler.cuh are the reading). Everything that does not
// depend on the noise level is prepared once a fold; a denoiser call runs every in-flight sample (NS) as one batch.
#include "af3.h"
#include <cmath>
#include <map>
#include <random>

static const size_t CHUNK = (size_t)16 << 20;      // elements of a chunked pair-sized working tensor
static int round8(int n) { return (n + 7) / 8 * 8; }
static RelIdx relIdx() {
  return {M.i("batch.features.residueIndex"), M.i("batch.features.tokenIndex"), M.i("batch.features.asymId"),
          M.i("batch.features.entityId"), M.i("batch.features.symId")};
}
// the conditioning's plain transition: LN (scale, offset), SwiGLU, the projection added
static void plainTransition(float* x, size_t rows, int C, const std::string& P) {
  int I = (int)(lenW(P + ".ffwTransition1") / (2 * (size_t)C));
  size_t chunk = std::min(rows, std::max<size_t>(64, CHUNK / (2 * (size_t)I)));
  half* xn = scratch<half>("pt.xn", chunk * C); half* g = scratch<half>("pt.g", chunk * I);
  const half* wp = swigluPairs(P + ".ffwTransition1", Wh(P + ".ffwTransition1"), C, I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    ln(x + r0 * C, xn, r, C, P + ".ffwLayerNormScale", P + ".ffwLayerNormOffset");
    gemmSwiglu(xn, wp, g, r, C, I);
    lin(g, P + ".ffwTransition2", x + r0 * C, r, I, C, 1.f);
  }
}

struct Diffusion {
  int n;
  const float *trunkSingle, *targetFeat;
  Masks masks;
  float *pairCond, *singleBase;
  EncoderOut enc;
  std::vector<AtomBlockCache> decBlocks;
  int decC, decHeads, decD, decPerToken;
  // the token transformer: every block's pair bias and its conditioning projections folded into two weights
  int nblocks, perSuper, stride, ldn, ldr;
  std::vector<half*> bias;
  const half *wNorm, *wRaw;
  float *bNorm, *bRaw;
};
static Diffusion D;

// ---------------------------------------------------------------- the conditioning
static void prepareConditioning(const float* trunkSingle, const float* trunkPair, const float* targetFeat, int n) {
  const std::string P = "diffusion.conditioning";
  int Cz = metaI(P + ".pairChannels"), Cs = metaI(P + ".seqChannels");
  int Czt = metaI(P + ".trunkPairChannels"), Cst = metaI(P + ".trunkSingleChannels"), F = metaI(P + ".targetFeatWidth");
  const int rel = 139;
  if (flag("trunk.dialect.chaiDiffusionConditioning")) die("chai-1's diffusion conditioning is not in the native port yet");
  size_t pairs = (size_t)n * n;
  // [trunk pair | relative one-hot] under AF3; the relative encoding projected first under some (relpeProjection), and
  // the trunk pair LayerNormed and projected too (zTrunkProjection: OpenDDE, protenix2)
  bool split = hasW(P + ".zTrunkProjection"), relpe = !split && hasW(P + ".relpeProjection");
  int width = split ? 2 * Cz : relpe ? Czt + Cz : Czt + rel;
  size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / std::max(width, rel + Czt)));
  float* f2 = scratch<float>("dc.f2", per * width);
  half* f2n = scratch<half>("dc.f2n", per * width);
  D.pairCond = scratch<float>("dc.pair", pairs * Cz);
  for (size_t p0 = 0; p0 < pairs; p0 += per) {
    size_t r = std::min(per, pairs - p0);
    if (!split && !relpe) {
      run1d("af3_pair_features", r * width, PairFeatArgs{trunkPair, relIdx(), f2, p0, r, (uint)n, (uint)Czt});
    } else {
      float* relRows = scratch<float>("dc.rel", per * rel);
      run1d("af3_pair_features", r * rel, PairFeatArgs{trunkPair, relIdx(), relRows, p0, r, (uint)n, 0});
      float* relProj = scratch<float>("dc.relProj", per * Cz);
      lin(relRows, P + ".relpeProjection", relProj, r, rel, Cz);
      const float* first = trunkPair + p0 * Czt; int firstWidth = Czt;
      if (split) {
        half* tln = scratch<half>("dc.tln", per * Czt);
        ln(trunkPair + p0 * Czt, tln, r, Czt, P + ".zTrunkNormScale", P + ".zTrunkNormOffset");
        float* tproj = scratch<float>("dc.tproj", per * Cz);
        lin(tln, P + ".zTrunkProjection", tproj, r, Czt, Cz);
        first = tproj; firstWidth = Cz;
      }
      run1d("af3_concat_pad", r * width, ConcatPadArgs{first, relProj, f2, (uint)r, (uint)firstWidth, (uint)Cz, -1, -1, 0});
    }
    ln(f2, f2n, r, width, P + ".pairCondInitialNormScale", P + ".pairCondInitialNormOffset");
    lin(f2n, P + ".pairCondInitialProjection", D.pairCond + p0 * Cz, r, width, Cz);
  }
  for (int k = 0; k < 2; ++k) plainTransition(D.pairCond, pairs, Cz, P + ".pairTransitions." + num(k));
  // [trunk single | target_feat]; the openfold3 lineage pads two always-zero columns (unknown DNA, after the restype
  // and profile blocks) - free before a linear, not before this LayerNorm
  bool pad = flag("trunk.dialect.padSingleCondUnknownDna");
  int sw = Cst + F + (pad ? 2 : 0);
  if (lenW(P + ".singleCondInitialNormScale") != (size_t)sw) die("single conditioning: the LayerNorm is %zu wide, the features %d",
                                                                  lenW(P + ".singleCondInitialNormScale"), sw);
  float* f1 = scratch<float>("dc.f1", (size_t)n * sw);
  run1d("af3_concat_pad", (size_t)n * sw, ConcatPadArgs{trunkSingle, targetFeat, f1, (uint)n, (uint)Cst, (uint)F,
                                                        pad ? Cst + 31 : -1, pad ? Cst + 63 : -1, 0});
  half* f1n = scratch<half>("dc.f1n", (size_t)n * sw);
  ln(f1, f1n, n, sw, P + ".singleCondInitialNormScale", P + ".singleCondInitialNormOffset");
  D.singleBase = scratch<float>("dc.singleBase", (size_t)n * Cs);
  lin(f1n, P + ".singleCondInitialProjection", D.singleBase, n, sw, Cs, 0.f, Wopt(P + ".singleCondInitialProjectionBias"));
  releaseScratch({"dc.f", "dc.rel", "dc.tln", "dc.tproj", "pt."});
}
// the single conditioning at one noise level: the base plus the Fourier embedding's projection, two transitions
static float* singleConditioning(float level) {
  const std::string P = "diffusion.conditioning";
  int n = D.n, Cs = metaI(P + ".seqChannels");
  size_t nc = lenW(P + ".fourierWeight");
  float* e = scratch<float>("dc.emb", nc);
  run1d("af3_fourier", nc, FourierArgs{W(P + ".fourierWeight"), W(P + ".fourierBias"), e, (uint)nc, level});
  float* en = scratch<float>("dc.embn", nc);
  ln(e, en, 1, (int)nc, P + ".noiseEmbeddingInitialNormScale", P + ".noiseEmbeddingInitialNormOffset");
  float* proj = scratch<float>("dc.noiseproj", Cs);
  lin(en, P + ".noiseEmbeddingInitialProjection", proj, 1, (int)nc, Cs);
  float* single = scratch<float>("dc.single", (size_t)n * Cs);
  copy(single, D.singleBase, (size_t)n * Cs * 4);
  addBias(single, proj, n, Cs);
  for (int k = 0; k < 2; ++k) plainTransition(single, n, Cs, P + ".singleTransitions." + num(k));
  if (hasW(P + ".singleCondFinalNormScale")) {
    float* tmp = scratch<float>("dc.sfnorm", (size_t)n * Cs);
    ln(single, tmp, n, Cs, P + ".singleCondFinalNormScale", P + ".singleCondFinalNormOffset");
    copy(single, tmp, (size_t)n * Cs * 4);
  }
  return single;
}

// ---------------------------------------------------------------- the token transformer
static std::string blockName(int b) {
  return "diffusion.transformer.superBlocks." + num(b / D.perSuper) + ".blocks." + num(b % D.perSuper);
}
static void prepareTransformer(int n) {
  const std::string T = "diffusion.transformer";
  int C = metaI(T + ".channels"), Cc = metaI(T + ".condChannels"), Cz = metaI(T + ".pairChannels"), heads = metaI(T + ".heads");
  D.perSuper = metaI(T + ".blocksPerSuperBlock");
  int sbs = 0; while (hasW(T + ".superBlocks." + num(sbs) + ".pairLogitsProjection")) ++sbs;
  D.nblocks = sbs * D.perSuper;
  D.ldn = D.nblocks * 4 * C; D.ldr = D.nblocks * 2 * C;
  // every block's adaptive-LN projections folded: LN_s(cond) W = LN0(cond) diag(s) W -> [scale | shift | ffw scale |
  // ffw shift] per block over LN0(cond), and [zero gate | ffw zero gate] per block over cond; the biases as GEMM biases
  if (ADA_RAW) die("chai-1's transformer is not in the native port yet");
  D.wNorm = M.derived<half>("difftx.wNorm", (size_t)Cc * D.ldn, [&](half* out) {
    for (int b = 0; b < D.nblocks; ++b)
      for (int slot = 0; slot < 2; ++slot) {
        std::string pre = blockName(b) + (slot ? ".ffw" : ".");
        const float* s = W(pre + "SingleCondLayerNormScale");
        run1d("af3_fold_cond", (size_t)Cc * C, FoldCondArgs{s, Wh(pre + "SingleCondScaleWeights"), out, (uint)Cc, (uint)C, (uint)D.ldn,
                                                             (uint)(b * 4 * C + slot * 2 * C)});
        run1d("af3_fold_cond", (size_t)Cc * C, FoldCondArgs{s, Wh(pre + "SingleCondBias"), out, (uint)Cc, (uint)C, (uint)D.ldn,
                                                             (uint)(b * 4 * C + slot * 2 * C + C)});
      }
  });
  D.wRaw = M.derived<half>("difftx.wRaw", (size_t)Cc * D.ldr, [&](half* out) {
    for (int b = 0; b < D.nblocks; ++b)
      for (int slot = 0; slot < 2; ++slot) {
        std::string pre = blockName(b) + (slot ? ".ffw" : ".");
        run1d("af3_fold_cond", (size_t)Cc * C, FoldCondArgs{nullptr, Wh(pre + "AdaptiveZeroCondWeights"), out, (uint)Cc, (uint)C,
                                                             (uint)D.ldr, (uint)(b * 2 * C + slot * C)});
      }
  });
  D.bNorm = M.derived<float>("difftx.bNorm", D.ldn, [&](float* out) {
    for (int b = 0; b < D.nblocks; ++b)
      for (int slot = 0; slot < 2; ++slot)
        copy(out + b * 4 * C + slot * 2 * C, W(blockName(b) + (slot ? ".ffw" : ".") + "SingleCondScaleBias"), (size_t)C * 4);
  });
  D.bRaw = M.derived<float>("difftx.bRaw", D.ldr, [&](float* out) {
    for (int b = 0; b < D.nblocks; ++b)
      for (int slot = 0; slot < 2; ++slot)
        copy(out + b * 2 * C + slot * C, W(blockName(b) + (slot ? ".ffw" : ".") + "AdaptiveZeroCondBias"), (size_t)C * 4);
  });
  // every block's pair bias, from the (fold-constant) pair conditioning
  size_t pairs = (size_t)n * n;
  D.stride = round8(n);
  half* pn = scratch<half>("dt.pn", pairs * Cz);
  ln(D.pairCond, pn, pairs, Cz, T + ".pairInputLayerNormScale", "");
  float* flat = scratch<float>("dt.flat", pairs * D.perSuper * heads);
  D.bias.assign(D.nblocks, nullptr);
  for (int b = 0; b < D.nblocks; ++b) {
    if (b % D.perSuper == 0) lin(pn, T + ".superBlocks." + num(b / D.perSuper) + ".pairLogitsProjection", flat, pairs, Cz, D.perSuper * heads);
    D.bias[b] = scratch<half>("dt.bias" + num(b), (size_t)heads * n * D.stride);
    run1d("af3_bias_layout", (size_t)heads * n * D.stride,
          BiasLayoutArgs{flat, D.bias[b], (uint)n, (uint)D.stride, (uint)heads, 0, (uint)(D.perSuper * heads), (uint)((b % D.perSuper) * heads),
                         (float)M_LOG2E, 0});
  }
  releaseScratch({"dt.pn", "dt.flat"});
}
static void transformer(float* act, const float* cond) {
  const std::string Tn = "diffusion.transformer";
  int n = D.n, C = metaI(Tn + ".channels"), Cc = metaI(Tn + ".condChannels");
  int heads = metaI(Tn + ".heads"), Dh = metaI(Tn + ".dimension"), Wd = heads * Dh, factor = metaI(Tn + ".transitionFactor");
  size_t rows = (size_t)n * NS;
  // every block's conditioning, two GEMMs
  // (the conditioning GEMMs too on 16-row tiles: [LN0(cond) | cond] in half, the padding rows zero)
  size_t pn = ((size_t)n + 15) / 16 * 16;
  float* gNorm = scratch<float>("dt.gNorm", pn * D.ldn);
  float* gRaw = scratch<float>("dt.gRaw", pn * D.ldr);
  half* cn = scratch<half>("dt.cn", pn * Cc);
  half* ch = scratch<half>("dt.ch", pn * Cc);
  static const void* zeroedC = nullptr;
  if (pn != (size_t)n && zeroedC != cn) { fill(cn, 0, pn * Cc * 2); fill(ch, 0, pn * Cc * 2); zeroedC = cn; }
  layerNorm(cond, cn, n, Cc, nullptr, nullptr);
  toHalf(cond, ch, (size_t)n * Cc);
  linW(cn, D.wNorm, gNorm, pn, Cc, D.ldn, 0.f, D.bNorm, 1.f, "transformer conditioning");
  linW(ch, D.wRaw, gRaw, pn, Cc, D.ldr, 0.f, D.bRaw, 1.f, "transformer zero gates");
  bool noResidual = flag(Tn + ".noResidual");
  float* pre = noResidual ? scratch<float>("dt.pre", rows * C) : nullptr;
  // the GEMMs run on the rows rounded up to 16: a ragged last tile takes the GEMM's bounds-checked paths (68 rows: 0.31
  // against 0.22 ms for a 768 x 3072 projection). The padding rows compute values nobody reads (zero in, finite out)
  size_t prows = (rows + 15) / 16 * 16;
  half* x = scratch<half>("dt.x", prows * C);
  half* qkvg = scratch<half>("dt.qkvg", prows * 4 * Wd);
  half* o = scratch<half>("dt.o", prows * Wd);
  float* att = scratch<float>("dt.att", prows * C);
  half* tn = scratch<half>("dt.tn", prows * C);
  int I = C * factor;
  half* gated = scratch<half>("dt.gated", prows * I);
  float* proj = scratch<float>("dt.proj", prows * C);
  static const void* zeroed = nullptr; static size_t zeroedRows = 0;
  if (prows != rows && (zeroed != x || zeroedRows != prows)) {
    fill(x, 0, prows * C * 2); fill(o, 0, prows * Wd * 2); fill(tn, 0, prows * C * 2);
    zeroed = x; zeroedRows = prows;
  }
  for (int b = 0; b < D.nblocks; ++b) {
    std::string B = blockName(b);
    const float* g = gNorm + (size_t)b * 4 * C; const float* z = gRaw + (size_t)b * 2 * C;
    if (hasW(B + ".queryLayerNormScale")) die("rf3's transformer q/k LayerNorm is not in the native port yet");
    adaLn(act, g, g + C, x, nullptr, rows, C, n, D.ldn);
    if (noResidual) copy(pre, act, rows * C * 4);
    linW(x, qkvgWeight(B, C, Wd, false), qkvg, prows, C, 4 * Wd, nullptr, "transformer qkvg");
    Attention at{}; at.qkvg = qkvg; at.out = o; at.n = n; at.heads = heads; at.D = Dh; at.rows = NS; at.scale = 1.f / sqrtf((float)Dh);
    at.bias = D.bias[b]; at.biasStride = D.stride; at.qBias = W(B + ".qBias");
    if (!D.masks.ones) { at.mask = D.masks.seq; at.maskB = 0; at.maskK = 1; }
    attention(at);
    lin(o, B + ".Transition2", att, prows, Wd, C);
    run1d("af3_gated_res_strided", rows * C, GatedResStridedArgs{act, att, z, rows, (uint)C, (uint)n, (uint)D.ldr, 0});
    adaLn(noResidual ? pre : act, g + 2 * C, g + 3 * C, tn, nullptr, rows, C, n, D.ldn);
    gemmSwiglu(tn, swigluPairs(B + ".ffwTransition1", Wh(B + ".ffwTransition1"), C, I), gated, prows, C, I);
    if (hasW(B + ".ffwAToB")) {          // boltz2's up-gate
      half* u = scratch<half>("dt.up", rows * I);
      linH(tn, B + ".ffwAToB", u, rows, C, I);
      run1d("af3_mul_h", rows * I, MulHArgs{gated, u, rows * I});
    }
    lin(gated, B + ".ffwTransition2", proj, prows, I, C);
    run1d("af3_gated_res_strided", rows * C, GatedResStridedArgs{act, proj, z + C, rows, (uint)C, (uint)n, (uint)D.ldr, 0});
  }
}

// ---------------------------------------------------------------- the decoder
static void prepareDecoder() {
  const std::string Dd = "diffusion.decoder";
  AtomShape sh = atomShape();
  D.decC = metaI(Dd + ".channels"); int Cp = metaI(Dd + ".pairChannels");
  D.decHeads = metaI(Dd + ".heads"); D.decD = metaI(Dd + ".dimension"); D.decPerToken = metaI(Dd + ".perTokenChannels");
  size_t qRows = (size_t)sh.subsets * sh.queries;
  int nblocks = 0; while (M.has(Dd + ".blocks." + num(nblocks) + ".qProjection")) ++nblocks;
  std::vector<float*> logits = atomPairLogits(Dd, D.enc.pair, qRows * sh.keys, Cp, nblocks, D.decHeads, sh);
  D.decBlocks.clear();
  for (int b = 0; b < nblocks; ++b) D.decBlocks.push_back(prepareAtomBlock(Dd + ".blocks." + num(b), D.enc.qCond, qRows, D.decC, logits[b]));
  releaseScratch({"apl.", "ada."});
}
static float* atomDecoder(const float* tokenAct) {
  const std::string Dd = "diffusion.decoder";
  AtomShape sh = atomShape();
  int C = D.decC;
  size_t q1 = (size_t)sh.subsets * sh.queries, qRows = q1 * NS, atoms = (size_t)sh.tokens * sh.dense;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2t = gatherOf("batch.queriesToTokenAtoms");
  float* proj = scratch<float>("dec.proj", (size_t)sh.tokens * NS * C);
  lin(tokenAct, Dd + ".projectTokenFeaturesForBroadcast", proj, (size_t)sh.tokens * NS, D.decPerToken, C);
  float* act = scratch<float>("dec.act", qRows * C);
  run1d("af3_broadcast_skip", qRows * C, BroadcastSkipArgs{proj, t2q.idx, t2q.mask, D.enc.skip, D.enc.qMask, act, qRows, q1, (uint)C,
                                                          (uint)sh.tokens, (uint)sh.dense, 0});
  AtomStep st{gatherOf("batch.queriesToKeys"), D.enc.qMask, D.enc.kMask, flag(Dd + ".blocks.0.keyMaskedAtomAttention"),
              flag(Dd + ".blocks.0.diffusionNoResidual")};
  bool maskPerBlock = flag(Dd + ".blocks.0.maskAtomActPerBlock");
  for (size_t b = 0; b < D.decBlocks.size(); ++b) {
    if (maskPerBlock) scaleRows(act, D.enc.qMask, qRows, C, q1);
    crossAttentionBlock(act, st, D.decBlocks[b], sh, C, D.decHeads, D.decD, Dd + ".blocks." + num(b));
  }
  if (lenW(Dd + ".atomFeaturesToPositionUpdate") != (size_t)C * 3) die("decoder: the position update is not C x 3");
  float* upd = scratch<float>("dec.upd", qRows * 3);
  run("af3_mask_ln_project3", grid1d((qRows + 7) / 8, 1), 256,
      MaskLnProject3Args{act, D.enc.qMask, W(Dd + ".atomFeaturesLayerNormScale"), Wopt(Dd + ".atomFeaturesLayerNormOffset"),
                         W(Dd + ".atomFeaturesToPositionUpdate"), upd, qRows, q1, (uint)C, 0});
  float* out = scratch<float>("dec.out", atoms * NS * 3);
  convert(q2t, upd, out, 3, q1, NS);
  return out;
}

// ---------------------------------------------------------------- the denoiser
void prepareDiffusion(const float* trunkSingle, const float* trunkPair, const float* targetFeat, const Masks& masks, int n) {
  D.n = n; D.trunkSingle = trunkSingle; D.targetFeat = targetFeat; D.masks = masks;
  prepareConditioning(trunkSingle, trunkPair, targetFeat, n);
  D.enc = prepareEncoder("diffusion.encoder", "atomReference", trunkSingle, D.pairCond);
  prepareDecoder();
  prepareTransformer(n);
  releaseScratch({"dc.pair"});
}
// D(x; sigma): positions [NS][tokens dense][3] in, the denoised positions out
static bool STAGES = getenv("AF3_STAGES") != nullptr;
static std::map<std::string, double> STAGE_MS;
static double stageAt = 0;
static void stage(const char* name) {
  if (!STAGES) return;
  mt::sync();
  double t = now();
  if (name) STAGE_MS[name] += (t - stageAt) * 1e3;
  stageAt = t;
}
void reportStages() {
  for (auto& [k, v] : STAGE_MS) printf("  %-14s %8.1f ms\n", k.c_str(), v);
  STAGE_MS.clear();
}
static float* denoise(const float* x, float level) {
  stage(nullptr);
  int n = D.n, dense = metaI("batch.dense");
  size_t atoms = (size_t)n * dense, total = atoms * NS;
  float d = level * level + 256.f, skip = 256.f / d, outS = level * 16.f / sqrtf(d), in = 1.f / sqrtf(d);
  float* single = singleConditioning(level);
  stage("conditioning");
  const float* atomMask = M.f("batch.refMask");
  float* scaled = scratch<float>("dn.scaled", total * 3);
  run1d("af3_scale_positions", total * 3, ScalePosArgs{x, atomMask, scaled, total, atoms, in, 0});
  encoderStep("diffusion.encoder", D.enc, scaled);
  stage("encoder");
  int Cs = metaI("diffusion.seqChannels"), perToken = metaI("diffusion.perTokenChannels");
  float* snProj = scratch<float>("dn.snProj", (size_t)n * perToken);
  {
    float* sn = scratch<float>("dn.sn", (size_t)n * Cs);
    if (hasW("diffusion.singleCondEmbeddingNormScale"))
      ln(single, sn, n, Cs, "diffusion.singleCondEmbeddingNormScale", "diffusion.singleCondEmbeddingNormOffset");
    else copy(sn, single, (size_t)n * Cs * 4);
    lin(sn, "diffusion.singleCondEmbeddingProjection", snProj, n, Cs, perToken);
  }
  size_t rows = (size_t)n * NS;
  float* act = scratch<float>("dn.act", rows * perToken);
  copy(act, D.enc.tokenAct, rows * perToken * 4);
  run1d("af3_add_broadcast", rows * perToken, AddBroadcastArgs{act, snProj, (u64)n * perToken, rows * perToken});
  stage("snproj");
  transformer(act, single);
  stage("transformer");
  float* actn = scratch<float>("dn.actn", rows * perToken);
  ln(act, actn, rows, perToken, "diffusion.outputNormScale", "diffusion.outputNormOffset");
  float* upd = atomDecoder(actn);
  stage("decoder");
  float* out = scratch<float>("dn.out", total * 3);
  run1d("af3_denoise_out", total * 3, DenoiseOutArgs{x, upd, atomMask, out, total, atoms, skip, outS});
  return out;
}

// ---------------------------------------------------------------- the sampler
static double noiseSchedule(double t, double sigmaData, double sigmaMin, double sigmaMax, double rho) {
  double lo = std::pow(sigmaMin, 1 / rho), hi = std::pow(sigmaMax, 1 / rho);
  return sigmaData * std::pow(hi + t * (lo - hi), rho);
}
SamplerOptions SAMPLER;
// every sample's positions, sample-major [ns][atoms][3]. AF3's sampler: each step centres the real atoms, rotates by a
// random rotation, translates by a unit normal, injects noise and takes the Euler step - the per-atom Gaussians a
// counter hash of (seed, step, element), the augmentation's twelve a step from the host's seeded stream
std::vector<float> sample(int steps, const std::vector<uint64_t>& seeds, const std::vector<float>& mask,
                          const std::function<void(const float*, int, int)>& onStep) {
  int ns = (int)seeds.size();
  size_t atoms = mask.size(), n3 = atoms * 3, all3 = n3 * ns;
  NS = ns;
  double gamma0 = 0.8, gammaMin = 1.0, noiseScale = 1.003, stepScale = 1.5, sigmaMin = 0.0004, sigmaMax = 160, rho = 7;
  const std::string S = "trunk.dialect.sampler.";
  if (M.has(S + "gamma0")) {          // a model's own EDM constants (boltz2)
    gamma0 = M.meta(S + "gamma0"); gammaMin = M.meta(S + "gammaMin"); noiseScale = M.meta(S + "noiseScale");
    stepScale = M.meta(S + "stepScale"); rho = M.meta(S + "rho"); sigmaMin = M.meta(S + "sigmaMin"); sigmaMax = M.meta(S + "sigmaMax");
  }
  if (flag("trunk.dialect.chaiSampler")) die("chai-1's sampler is not in the native port yet");
  float* dX = allocT<float>(all3);
  uint64_t* dSeeds = uploadNew(seeds.data(), ns);
  float* dMask = uploadNew(mask.data(), atoms);
  std::vector<double> levels(steps + 1);
  if (SAMPLER.flow) {
    // the page's Flow: one draw at the top of a schedule from 160 A, then the state REPLACED by each prediction
    for (int k = 0; k <= steps; ++k) levels[k] = noiseSchedule((double)k / steps, 16, 0.0004, 10, 7);
    run1d("af3_initial_noise", all3, InitNoiseArgs{dX, dSeeds, n3, all3, (float)levels[0], 0});
    for (int step = 1; step <= steps; ++step) {
      const float* d = denoise(dX, (float)levels[step - 1]);
      if (onStep) onStep(d, step, steps);
      copy(dX, d, all3 * 4);
    }
  } else {
    if (SAMPLER.sigmaMax > 0) sigmaMax = SAMPLER.sigmaMax;
    for (int k = 0; k <= steps; ++k) levels[k] = noiseSchedule((double)k / steps, 16, sigmaMin, sigmaMax, rho);
    std::vector<float> rot((size_t)steps * ns * 12);
    for (int k = 0; k < ns; ++k) {
      std::mt19937_64 gen(seeds[k]); std::normal_distribution<double> normal(0.0, 1.0);
      for (int s = 0; s < steps; ++s) {
        double v0[3] = {normal(gen), normal(gen), normal(gen)}, v1[3] = {normal(gen), normal(gen), normal(gen)};
        auto norm = [](const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
        double e0[3], e1[3], e2[3], s0 = 1 / std::max(1e-10, norm(v0));
        for (int d = 0; d < 3; ++d) e0[d] = v0[d] * s0;
        double dot = v1[0] * e0[0] + v1[1] * e0[1] + v1[2] * e0[2], w[3];
        for (int d = 0; d < 3; ++d) w[d] = v1[d] - e0[d] * dot;
        double s1 = 1 / std::max(1e-10, norm(w));
        for (int d = 0; d < 3; ++d) e1[d] = w[d] * s1;
        e2[0] = e0[1] * e1[2] - e0[2] * e1[1]; e2[1] = e0[2] * e1[0] - e0[0] * e1[2]; e2[2] = e0[0] * e1[1] - e0[1] * e1[0];
        float* r = &rot[((size_t)s * ns + k) * 12];
        for (int d = 0; d < 3; ++d) { r[d] = (float)e0[d]; r[3 + d] = (float)e1[d]; r[6 + d] = (float)e2[d]; }
        for (int d = 0; d < 3; ++d) r[9 + d] = (float)normal(gen);
      }
    }
    float* dRot = uploadNew(rot.data(), rot.size());
    float* dNoisy = scratch<float>("sample.noisy", all3); float* dC = scratch<float>("sample.centroid", 3 * ns);
    run1d("af3_initial_noise", all3, InitNoiseArgs{dX, dSeeds, n3, all3, (float)levels[0], 0});
    for (int step = 1; step <= steps; ++step) {
      double previous = levels[step - 1], level = levels[step];
      float tHat = (float)(previous * (1 + (level > gammaMin ? gamma0 : 0)));
      double injected = noiseScale * std::sqrt(std::max(0.0, (double)tHat * tHat - previous * previous));
      run("af3_centroid", Grid{(uint32_t)ns, 1, 1}, 1024, CentroidArgs{dX, dMask, dC, atoms});
      run1d("af3_augment", atoms * ns, AugmentArgs{dX, dNoisy, dMask, dC, dRot + (size_t)(step - 1) * ns * 12, dSeeds, atoms, atoms * ns,
                                                   (uint)step, (float)injected});
      const float* d = denoise(dNoisy, tHat);
      if (onStep) onStep(d, step, steps);
      run1d("af3_euler", all3, EulerArgs{dX, dNoisy, d, all3, (float)(stepScale * (level - tHat) / tHat), 0});
    }
    release(dRot);
  }
  std::vector<float> out = download(dX, all3);
  release(dX); release(dSeeds); release(dMask);
  NS = 1;
  return out;
}
void freeDiffusion() {
  releaseScratch({"dt.", "dc.", "dn.", "dec.", "diffusion.", "atomReference.", "sample.", "ab.", "enc."});
  D = Diffusion{};
}
