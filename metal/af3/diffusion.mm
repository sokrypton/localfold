// The diffusion head - its conditioning, the atom encoder, the conditioned token transformer, the atom decoder - and
// AF3's EDM sampler around it (cuda/af3/src/diffusion.cuh and sampler.cuh are the reading). Everything that does not
// depend on the noise level is prepared once a fold; a denoiser call runs every in-flight sample (NS) as one batch.
#include "../../cuda/af3/src/rigid.cuh"   // rigidAlign (plain C++), boltz2's realign
#include "af3.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <random>

static const size_t CHUNK = (size_t)16 << 20;      // elements of a chunked pair-sized working tensor
static int round8(int n) { return (n + 7) / 8 * 8; }
static RelIdx relIdx() {
  return {M.i("batch.features.residueIndex"), M.i("batch.features.tokenIndex"), M.i("batch.features.asymId"),
          M.i("batch.features.entityId"), M.i("batch.features.symId")};
}
// the conditioning's plain transition: LN (scale, offset), SwiGLU, the projection added
static void plainTransition(float* x, size_t rows, int C, const std::string& P, size_t padRows = 0) {
  int I = (int)(lenW(P + ".ffwTransition1") / (2 * (size_t)C));
  size_t rp = padRows > rows ? padRows : rows;
  size_t chunk = std::min(rows, std::max<size_t>(64, CHUNK / (2 * (size_t)I)));
  half* xn = scratch<half>("pt.xn", std::max(chunk, rp) * C); half* g = scratch<half>("pt.g", std::max(chunk, rp) * I);
  if (rp > rows && chunk == rows) fill(xn + rows * C, 0, (rp - rows) * C * sizeof(half));
  const half* wp = swigluPairs(P + ".ffwTransition1", C, I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    size_t rg = (rp > rows && r == rows) ? rp : r;
    ln(x + r0 * C, xn, r, C, P + ".ffwLayerNormScale", P + ".ffwLayerNormOffset");
    gemmSwiglu(xn, wp, g, rg, C, I);
    lin(g, P + ".ffwTransition2", x + r0 * C, rg, I, C, 1.f);
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
  // the transformer's conditioning for upcoming noise levels, batched (conditioningAhead): the sampler's plan of
  // levels, the levels held and their two GEMMs' outputs, slot k at k pn rows
  std::vector<float> plan; size_t planAt = 0;
  std::vector<float> held; float *heldNorm = nullptr, *heldRaw = nullptr, *heldSingle = nullptr, *heldSnProj = nullptr;
  bool heldPermuted = false;
};
static Diffusion D;

// ---------------------------------------------------------------- the conditioning
static void prepareConditioning(const float* trunkSingle, const float* trunkPair, const float* targetFeat, int n) {
  const std::string P = "diffusion.conditioning";
  int Cz = metaI(P + ".pairChannels"), Cs = metaI(P + ".seqChannels");
  int Czt = metaI(P + ".trunkPairChannels"), Cst = metaI(P + ".trunkSingleChannels"), F = metaI(P + ".targetFeatWidth");
  const int rel = 139;
  size_t pairs = (size_t)n * n;
  // [trunk pair | relative one-hot] under AF3; the relative encoding projected first under some (relpeProjection), and
  // the trunk pair LayerNormed and projected too (zTrunkProjection: OpenDDE, protenix2)
  const bool chai = flag("trunk.dialect.chaiDiffusionConditioning");
  bool split = hasW(P + ".zTrunkProjection"), relpe = !split && hasW(P + ".relpeProjection");
  int width = chai ? Czt + Cz : split ? 2 * Cz : relpe ? Czt + Cz : Czt + rel;
  // (chai's: the entity and symmetry ids as dense ranks, torch.unique's inverse)
  auto denseRank = [&](const char* key) {
    const int* h = M.hostI(key); size_t m = M.len(key);
    std::vector<int> sorted(h, h + m); std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    std::vector<int> rank(m);
    for (size_t k = 0; k < m; ++k) rank[k] = (int)(std::lower_bound(sorted.begin(), sorted.end(), h[k]) - sorted.begin());
    int* d = scratch<int>(std::string("dc.rank.") + key, m);
    upload(d, rank.data(), m * 4);
    return (const int*)d;
  };
  const int* entityRank = chai ? denseRank("batch.features.entityId") : nullptr;
  const int* symRank = chai ? denseRank("batch.features.symId") : nullptr;
  size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / std::max(width, rel + Czt)));
  float* f2 = scratch<float>("dc.f2", per * width);
  half* f2n = scratch<half>("dc.f2n", per * width);
  D.pairCond = scratch<float>("dc.pair", pairs * Cz);
  for (size_t p0 = 0; p0 < pairs; p0 += per) {
    size_t r = std::min(per, pairs - p0);
    if (chai) {
      run1d("af3_chai_struct_pair", r * width,
            ChaiStructPairArgs{trunkPair, M.i("batch.features.residueIndex"), M.i("batch.features.tokenIndex"), M.i("batch.features.asymId"),
                               entityRank, symRank, W(P + ".structurePairWeights"), W(P + ".structurePairBias"),
                               M.has("batch.declaredBondMatrix") ? M.f("batch.declaredBondMatrix") : nullptr, W(P + ".structureBondWeights"), f2, p0, r,
                               (uint)n, (uint)Czt, (uint)Cz, 0});
    } else if (!split && !relpe) {
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
  if (chai) {     // chai closes the pair track with an affine LayerNorm
    float* tmp = scratch<float>("dc.fnorm", pairs * Cz);
    ln(D.pairCond, tmp, pairs, Cz, P + ".pairCondFinalNormScale", P + ".pairCondFinalNormOffset");
    copy(D.pairCond, tmp, pairs * Cz * 4);
  }
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
  FourierArgs fa{W(P + ".fourierWeight"), W(P + ".fourierBias"), e, (uint)nc, 1u, {level}};
  run1d("af3_fourier", nc, fa);
  half* en = scratch<half>("dc.embn", nc);
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

// singleConditioning for K levels at once: the K noise projections added to K copies of the base, then the
// transitions and the final norm - row-wise - over all K n rows (at a short n those were K launches of 68-row work)
static void singleConditioningBatch(const float* levels, size_t K, float* out, size_t padRows = 0) {
  const std::string P = "diffusion.conditioning";
  int n = D.n, Cs = metaI(P + ".seqChannels");
  size_t nc = lenW(P + ".fourierWeight");
  float* e = scratch<float>("dc.emb", K * nc);
  half* en = scratch<half>("dc.embn", K * nc);
  float* projs = scratch<float>("dc.noiseprojK", K * Cs);
  FourierArgs fa{W(P + ".fourierWeight"), W(P + ".fourierBias"), e, (uint)nc, (uint)K, {}};
  for (size_t k = 0; k < K; ++k) fa.levels[k] = levels[k];
  run1d("af3_fourier", K * nc, fa);
  ln(e, en, K, (int)nc, P + ".noiseEmbeddingInitialNormScale", P + ".noiseEmbeddingInitialNormOffset");
  lin(en, P + ".noiseEmbeddingInitialProjection", projs, K, (int)nc, Cs);
  run1d("af3_single_batch_init", (Cs & 3) ? K * n * Cs : K * n * (Cs >> 2),
        SingleBatchInitArgs{D.singleBase, projs, out, (uint)K, (uint)n, (uint)Cs, 0});
  for (int t = 0; t < 2; ++t) plainTransition(out, K * n, Cs, P + ".singleTransitions." + num(t), padRows);
  if (hasW(P + ".singleCondFinalNormScale")) {
    float* tmp = scratch<float>("dc.sfnormK", K * n * Cs);
    ln(out, tmp, K * n, Cs, P + ".singleCondFinalNormScale", P + ".singleCondFinalNormOffset");
    copy(out, tmp, K * n * Cs * 4);
  }
}

// ---------------------------------------------------------------- the token transformer
static std::string blockName(int b) {
  return "diffusion.transformer.superBlocks." + num(b / D.perSuper) + ".blocks." + num(b % D.perSuper);
}
void transformerWeights() {
  const std::string T = "diffusion.transformer";
  int C = metaI(T + ".channels"), Cc = metaI(T + ".condChannels");
  D.perSuper = metaI(T + ".blocksPerSuperBlock");
  int sbs = 0; while (hasW(T + ".superBlocks." + num(sbs) + ".pairLogitsProjection")) ++sbs;
  D.nblocks = sbs * D.perSuper;
  D.ldn = D.nblocks * 4 * C; D.ldr = D.nblocks * 2 * C;
  // every block's adaptive-LN projections folded: LN_s(cond) W = LN0(cond) diag(s) W -> [scale | shift | ffw scale |
  // ffw shift] per block over LN0(cond), and [zero gate | ffw zero gate] per block over cond; the biases as GEMM biases
  D.wNorm = M.derived<half>("difftx.wNorm", (size_t)Cc * D.ldn, [&](half* out) {
    for (int b = 0; b < D.nblocks; ++b)
      for (int slot = 0; slot < 2; ++slot) {
        std::string pre = blockName(b) + (slot ? ".ffw" : ".");
        const float* s = ADA_RAW ? nullptr : W(pre + "SingleCondLayerNormScale");     // (chai: the conditioning raw)
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
        if (ADA_RAW) {      // chai: the scale's + 1 as its bias
          std::vector<float> ones((size_t)C, 1.f);
          upload(out + b * 4 * C + slot * 2 * C, ones.data(), (size_t)C * 4);
        } else copy(out + b * 4 * C + slot * 2 * C, W(blockName(b) + (slot ? ".ffw" : ".") + "SingleCondScaleBias"), (size_t)C * 4);
  });
  D.bRaw = M.derived<float>("difftx.bRaw", D.ldr, [&](float* out) {
    for (int b = 0; b < D.nblocks; ++b)
      for (int slot = 0; slot < 2; ++slot)
        copy(out + b * 2 * C + slot * C, W(blockName(b) + (slot ? ".ffw" : ".") + "AdaptiveZeroCondBias"), (size_t)C * 4);
  });
}
static void prepareTransformer(int n) {
  const std::string T = "diffusion.transformer";
  int Cz = metaI(T + ".pairChannels"), heads = metaI(T + ".heads");
  transformerWeights();
  // every block's pair bias, from the (fold-constant) pair conditioning
  size_t pairs = (size_t)n * n;
  D.stride = (n + 63) / 64 * 64;
  half* pn = scratch<half>("dt.pn", pairs * Cz);
  ln(D.pairCond, pn, pairs, Cz, T + ".pairInputLayerNormScale", "");
  // (the pair conditioning's last reader: the encoder's and decoder's preparations ran before - given back before the
  // biases are made, the diffusion's top stage: 1044 tokens 4.06 -> 3.50 GB)
  releaseScratch({"dc.pair"});
  D.pairCond = nullptr;
  float* flat = scratch<float>("dt.flat", pairs * D.perSuper * heads);
  D.bias.assign(D.nblocks, nullptr);
  for (int b = 0; b < D.nblocks; ++b) {
    if (b % D.perSuper == 0) lin(pn, T + ".superBlocks." + num(b / D.perSuper) + ".pairLogitsProjection", flat, pairs, Cz, D.perSuper * heads);
    D.bias[b] = scratch<half>("dt.bias" + num(b), (size_t)heads * n * D.stride);
    run("af3_bias_layout", Grid{(uint32_t)((D.stride + 31) / 32), (uint32_t)((n + 31) / 32), (uint32_t)heads}, 256,
          BiasLayoutArgs{flat, D.bias[b], (uint)n, (uint)D.stride, (uint)heads, 0, (uint)(D.perSuper * heads), (uint)((b % D.perSuper) * heads),
                         (float)M_LOG2E, 0});
  }
  releaseScratch({"dt.pn", "dt.flat"});
}
// The transformer's conditioning depends on the noise level alone, and the sampler knows its levels: at a short n the
// two conditioning GEMMs (73728 and 36864 columns at boltz2's 24 blocks) are bound by reading their 170 MB of weights,
// so the next K levels' conditionings run as ONE pair of GEMMs, K pn rows - the weights read once for K steps. Only
// where the rows are few (pn <= 128: past that the GEMMs are compute-bound) and K x the outputs fit 256 MB.
// AF3_COND_AHEAD=0 the control.
static void conditioningAhead(float level) {
  static const bool on = !getenv("AF3_COND_AHEAD") || atoi(getenv("AF3_COND_AHEAD")) != 0;
  if (!on || D.plan.empty() || std::find(D.held.begin(), D.held.end(), level) != D.held.end()) return;
  auto it = std::find(D.plan.begin() + std::min(D.planAt, D.plan.size()), D.plan.end(), level);
  if (it == D.plan.end()) return;
  D.planAt = (size_t)(it - D.plan.begin());
  const std::string Tn = "diffusion.transformer";
  const int n = D.n, Cc = metaI(Tn + ".condChannels"), C = metaI(Tn + ".channels");
  const size_t pn = ((size_t)n + 15) / 16 * 16, per = (size_t)n * (D.ldn + D.ldr) * 4;
  const size_t rem = D.plan.size() - D.planAt, maxK = std::max<size_t>(1, std::min<size_t>(32, ((size_t)768 << 20) / std::max<size_t>(per, 1)));
  const size_t batches = std::max<size_t>(1, (rem + maxK - 1) / maxK), K = std::min<size_t>(rem, (rem + batches - 1) / batches);
  const size_t maxPn = flag("trunk.dialect.structuralTokens") ? 256 : 128;
  if (pn > maxPn || K < 2 || metaI("diffusion.conditioning.seqChannels") != Cc) return;
  D.held.assign(D.plan.begin() + D.planAt, D.plan.begin() + D.planAt + K);
  const size_t kn = K * n, pkn = (kn + 63) / 64 * 64, cRows = std::max(K * pn, pkn);
  D.heldSingle = scratch<float>("dt.heldSingle", pkn * Cc);
  singleConditioningBatch(D.held.data(), K, D.heldSingle, pkn);      // (the denoiser reads its slot too: heldSingleFor)
  int Cs = metaI("diffusion.seqChannels"), perToken = metaI("diffusion.perTokenChannels");
  D.heldSnProj = scratch<float>("dt.heldSnProj", pkn * perToken);
  half* sn = scratch<half>("dt.heldSn", pkn * Cs);
  if (pkn > kn) fill(sn + kn * Cs, 0, (pkn - kn) * Cs * sizeof(half));
  if (hasW("diffusion.singleCondEmbeddingNormScale"))
    ln(D.heldSingle, sn, kn, Cs, "diffusion.singleCondEmbeddingNormScale", "diffusion.singleCondEmbeddingNormOffset");
  else toHalf(D.heldSingle, sn, kn * Cs);
  lin(sn, "diffusion.singleCondEmbeddingProjection", D.heldSnProj, pkn, Cs, perToken);
  D.heldNorm = scratch<float>("dt.heldNorm", K * pn * D.ldn);
  D.heldRaw = scratch<float>("dt.heldRaw", K * pn * D.ldr);
  half* cn = scratch<half>("dt.heldCn", cRows * Cc);
  half* ch = scratch<half>("dt.heldCh", cRows * Cc);
  run("af3_cond_norm_cast", grid1d((pkn + 7) / 8, 1), 256,
      CondNormCastArgs{D.heldSingle, cn, ch, 1u, (uint)kn, (uint)pkn, (uint)Cc});
  if (gemmCondPermuted(ADA_RAW ? ch : cn, D.wNorm, D.heldNorm, pkn, kn, Cc, D.ldn, D.bNorm, n, 2 * C, ADA_RAW ? 0 : C,
                       "transformer conditioning, ahead") &&
      gemmCondPermuted(ch, D.wRaw, D.heldRaw, pkn, kn, Cc, D.ldr, D.bRaw, n, C, C,
                       "transformer zero gates, ahead")) {
    D.heldPermuted = true;
    return;
  }
  D.heldPermuted = false;
  run("af3_cond_norm_cast", grid1d((K * pn + 7) / 8, 1), 256,
      CondNormCastArgs{D.heldSingle, cn, ch, (uint)K, (uint)n, (uint)pn, (uint)Cc});
  linW(ADA_RAW ? ch : cn, D.wNorm, D.heldNorm, K * pn, Cc, D.ldn, 0.f, D.bNorm, 1.f, "transformer conditioning, ahead");
  linW(ch, D.wRaw, D.heldRaw, K * pn, Cc, D.ldr, 0.f, D.bRaw, 1.f, "transformer zero gates, ahead");
}
static void transformerConditioning(const float* cond, float level, float** gNormOut, float** gRawOut, bool* permutedOut) {
  const std::string Tn = "diffusion.transformer";
  int n = D.n, C = metaI(Tn + ".channels"), Cc = metaI(Tn + ".condChannels");
  size_t pn = ((size_t)n + 15) / 16 * 16;
  auto slot = std::find(D.held.begin(), D.held.end(), level);
  if (slot != D.held.end()) {
    size_t idx = (size_t)(slot - D.held.begin());
    size_t stepRows = D.heldPermuted ? (size_t)n : pn;
    *gNormOut = D.heldNorm + idx * stepRows * D.ldn;
    *gRawOut = D.heldRaw + idx * stepRows * D.ldr;
    *permutedOut = D.heldPermuted;
    return;
  }
  float* gNorm = scratch<float>("dt.gNorm", pn * D.ldn);
  float* gRaw = scratch<float>("dt.gRaw", pn * D.ldr);
  half* cn = scratch<half>("dt.cn", pn * Cc);
  half* ch = scratch<half>("dt.ch", pn * Cc);
  run("af3_cond_norm_cast", grid1d((pn + 7) / 8, 1), 256, CondNormCastArgs{cond, cn, ch, 1u, (uint)n, (uint)pn, (uint)Cc});
  if (gemmCondPermuted(ADA_RAW ? ch : cn, D.wNorm, gNorm, pn, n, Cc, D.ldn, D.bNorm, n, 2 * C, ADA_RAW ? 0 : C,
                       "transformer conditioning") &&
      gemmCondPermuted(ch, D.wRaw, gRaw, pn, n, Cc, D.ldr, D.bRaw, n, C, C,
                       "transformer zero gates")) {
    *permutedOut = true;
  } else {
    linW(ADA_RAW ? ch : cn, D.wNorm, gNorm, pn, Cc, D.ldn, 0.f, D.bNorm, 1.f, "transformer conditioning");    // (chai: raw)
    linW(ch, D.wRaw, gRaw, pn, Cc, D.ldr, 0.f, D.bRaw, 1.f, "transformer zero gates");
    *permutedOut = false;
  }
  *gNormOut = gNorm;
  *gRawOut = gRaw;
}
static void transformer(float* act, const float* gNorm, const float* gRaw, bool permuted, bool preTx0) {
  const std::string Tn = "diffusion.transformer";
  int n = D.n, C = metaI(Tn + ".channels");
  int heads = metaI(Tn + ".heads"), Dh = metaI(Tn + ".dimension"), Wd = heads * Dh, factor = metaI(Tn + ".transitionFactor");
  size_t rows = (size_t)n * NS;
  bool noResidual = flag(Tn + ".noResidual");
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
    fill(x + rows * C, 0, (prows - rows) * C * 2);
    fill(o + rows * Wd, 0, (prows - rows) * Wd * 2);
    fill(tn + rows * C, 0, (prows - rows) * C * 2);
    zeroed = x; zeroedRows = prows;
  }
  const float* pendY = nullptr; const float* pendGate = nullptr;
  for (int b = 0; b < D.nblocks; ++b) {
    std::string B = blockName(b);
    const int ldn = permuted ? 2 * C : D.ldn, ldr = permuted ? C : D.ldr;
    const float* gAtt = permuted ? gNorm + (size_t)(2 * b) * n * 2 * C : gNorm + (size_t)b * 4 * C;
    const float* gFfw = permuted ? gNorm + (size_t)(2 * b + 1) * n * 2 * C : gAtt + 2 * C;
    const float* zAtt = permuted ? gRaw + (size_t)(2 * b) * n * C : gRaw + (size_t)b * 2 * C;
    const float* zFfw = permuted ? gRaw + (size_t)(2 * b + 1) * n * C : zAtt + C;
    if (b > 0 || !preTx0) {
      adaLn(act, gAtt, gAtt + C, x, nullptr, rows, C, n, ldn,
            noResidual ? gFfw : nullptr, noResidual ? gFfw + C : nullptr, noResidual ? tn : nullptr,
            false, pendY, pendGate, ldr, nullptr, 1.f, permuted);
      pendY = nullptr; pendGate = nullptr;
    }
    linW(x, qkvgWeight(B, C, Wd, false), qkvg, prows, C, 4 * Wd, nullptr, "transformer qkvg");
    Attention at{}; at.qkvg = qkvg; at.out = o; at.n = n; at.heads = heads; at.D = Dh; at.rows = NS; at.scale = 1.f / sqrtf((float)Dh);
    at.bias = D.bias[b]; at.biasStride = D.stride; at.biasPadInf = true; at.qBias = W(B + ".qBias");
    if (!D.masks.ones) { at.mask = D.masks.seq; at.maskB = 0; at.maskK = 1; }
    if (hasW(B + ".queryLayerNormScale")) {     // rf3: q (its bias inside) and k normalised per token row
      uint32_t sgs = 2 * rows <= 512 ? 2 : 8;
      run("af3_kq_norm", grid1d((2 * rows + sgs - 1) / sgs, 1), sgs * 32,
          KqNormArgs{qkvg, qkvg + Wd, W(B + ".qBias"), W(B + ".queryLayerNormScale"), W(B + ".queryLayerNormOffset"),
                     W(B + ".keyLayerNormScale"), W(B + ".keyLayerNormOffset"), rows, (uint)(4 * Wd), (uint)(4 * Wd), (uint)Wd, 0});
      at.qBias = nullptr;
    }
    attention(at);
    // (the gated residual in the output projection's epilogue where the matrix units take it: gemmGateResidual)
    if (!gemmGateResidual(o, Wh(B + ".Transition2"), act, prows, rows, Wd, C, zAtt, ldr, n, ADA_RAW ? 2.f : 1.f,
                          "transformer output, gated residual", permuted)) {
      lin(o, B + ".Transition2", att, prows, Wd, C, 0.f, nullptr, ADA_RAW ? 2.f : 1.f);   // (chai's zero gate weights: 0.5 undone)
      if (!noResidual) {
        adaLn(act, gFfw, gFfw + C, tn, nullptr, rows, C, n, ldn, nullptr, nullptr, nullptr, false, att, zAtt, ldr, nullptr, 1.f, permuted);
      } else {
        run1d("af3_gated_res_strided", rows * C, GatedResStridedArgs{act, att, zAtt, rows, (uint)C, (uint)n, (uint)ldr, permuted ? 1u : 0u});
      }
    } else if (!noResidual) {
      adaLn(act, gFfw, gFfw + C, tn, nullptr, rows, C, n, ldn, nullptr, nullptr, nullptr, false, nullptr, nullptr, 0, nullptr, 1.f, permuted);
    }
    gemmSwiglu(tn, swigluPairs(B + ".ffwTransition1", C, I), gated, prows, C, I);
    if (hasW(B + ".ffwAToB")) {          // boltz2's up-gate
      if (!gemmMulH(tn, Wh(B + ".ffwAToB"), gated, prows, C, I, "transformer up-gate")) {
        half* u = scratch<half>("dt.up", rows * I);
        linH(tn, B + ".ffwAToB", u, rows, C, I);
        run1d("af3_mul_h", rows * I, MulHArgs{gated, u, rows * I});
      }
    }
    if (!gemmGateResidual(gated, Wh(B + ".ffwTransition2"), act, prows, rows, I, C, zFfw, ldr, n, 1.f,
                          "transformer transition, gated residual", permuted)) {
      lin(gated, B + ".ffwTransition2", proj, prows, I, C);
      if (b + 1 < D.nblocks) {
        pendY = proj; pendGate = zFfw;
      } else {
        run1d("af3_gated_res_strided", rows * C, GatedResStridedArgs{act, proj, zFfw, rows, (uint)C, (uint)n, (uint)ldr, permuted ? 1u : 0u});
      }
    }
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
  const float* cond = D.enc.qCond;
  if (flag("trunk.dialect.chaiAtomStack")) {
    // chai conditions its decoder on a second, affine LayerNorm of the encoder's conditioning, and restricts its
    // attention to one reference space as the encoder does
    float* c2 = scratch<float>("dec.cond", qRows * D.decC);
    ln(D.enc.qCond, c2, qRows, D.decC, Dd + ".postAtomCondLayerNormScale", Dd + ".postAtomCondLayerNormOffset");
    cond = c2;
    for (float* pl : logits) sameRefMask(pl, D.enc.qUid, D.enc.kUid, D.decHeads, sh);
  }
  D.decBlocks.clear();
  for (int b = 0; b < nblocks; ++b) D.decBlocks.push_back(prepareAtomBlock(Dd + ".blocks." + num(b), cond, qRows, D.decC, logits[b]));
  releaseScratch({"apl.", "ada."});
}
static float* atomDecoder(const half* tokenAct) {
  const std::string Dd = "diffusion.decoder";
  AtomShape sh = atomShape();
  int C = D.decC;
  size_t q1 = (size_t)sh.subsets * sh.queries, qRows = q1 * NS;
  size_t rows = (size_t)sh.tokens * NS, prows = (rows + 15) / 16 * 16;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries");
  float* proj = scratch<float>("dec.proj", prows * C);
  lin(tokenAct, Dd + ".projectTokenFeaturesForBroadcast", proj, prows, D.decPerToken, C);
  float* act = scratch<float>("dec.act", qRows * C);
  bool maskPerBlock = flag(Dd + ".blocks.0.maskAtomActPerBlock");
  AtomStep st{gatherOf("batch.queriesToKeys"), D.enc.qMask, D.enc.kMask, flag(Dd + ".blocks.0.keyMaskedAtomAttention"),
              flag(Dd + ".blocks.0.diffusionNoResidual"), maskPerBlock};
  const bool fuse0 = C == 128 && !D.decBlocks.empty();
  half* xqk0 = fuse0 ? scratch<half>("ab.xqk", 2 * qRows * C) : nullptr;
  half* xq0 = xqk0;
  half* xk0 = fuse0 ? xqk0 + qRows * C : nullptr;
  half* tn0 = (fuse0 && st.noResidual) ? scratch<half>("ab.tn", qRows * C) : nullptr;
  const AtomBlockCache* b0 = fuse0 ? &D.decBlocks[0] : nullptr;
  run1d("af3_broadcast_skip", (C & 3) ? qRows * C : qRows * (C >> 2),
        BroadcastSkipArgs{proj, t2q.idx, t2q.mask, D.enc.skip, D.enc.qMask, act,
                          b0 ? b0->qScale : nullptr, b0 ? b0->qShift : nullptr, xq0,
                          b0 ? b0->kScale : nullptr, b0 ? b0->kShift : nullptr, xk0,
                          tn0 ? b0->ffwScale : nullptr, tn0 ? b0->ffwShift : nullptr, tn0,
                          qRows, q1, (uint)C, (uint)sh.tokens, (uint)sh.dense, 0,
                          ADA_RAW ? 0.1f : 1e-5f, b0 && b0->chained ? 1u : 0u});
  const float* pendY = nullptr; const float* pendGate = nullptr;
  bool preQk = fuse0;
  for (size_t b = 0; b < D.decBlocks.size(); ++b) {
    bool nextQk = false;
    crossAttentionBlock(act, st, D.decBlocks[b], sh, C, D.decHeads, D.decD, Dd + ".blocks." + num(b),
                        &pendY, &pendGate, true, preQk, b + 1 < D.decBlocks.size() ? &D.decBlocks[b + 1] : nullptr, &nextQk);
    preQk = nextQk;
  }
  if (lenW(Dd + ".atomFeaturesToPositionUpdate") != (size_t)C * 3) die("decoder: the position update is not C x 3");
  float* upd = scratch<float>("dec.upd", qRows * 3);
  run("af3_mask_ln_project3", grid1d((qRows + 7) / 8, 1), 256,
      MaskLnProject3Args{act, D.enc.qMask, W(Dd + ".atomFeaturesLayerNormScale"), Wopt(Dd + ".atomFeaturesLayerNormOffset"),
                         W(Dd + ".atomFeaturesToPositionUpdate"), upd, pendY, pendGate, qRows, q1, (uint)C, 0});
  return upd;
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
static float* denoise(const float* x, float level, float* eulerX = nullptr, float eulerScale = 0.f, bool needOut = true) {
  stage(nullptr);
  int n = D.n, dense = metaI("batch.dense"), Cs = metaI("diffusion.seqChannels"), perToken = metaI("diffusion.perTokenChannels");
  size_t atoms = (size_t)n * dense, total = atoms * NS, rows = (size_t)n * NS;
  float d = level * level + 256.f, skip = 256.f / d, outS = level * 16.f / sqrtf(d), in = 1.f / sqrtf(d);
  conditioningAhead(level);
  auto heldAt = std::find(D.held.begin(), D.held.end(), level);
  float* single = nullptr;
  float* snProj = nullptr;
  if (heldAt != D.held.end()) {
    size_t idx = (size_t)(heldAt - D.held.begin());
    single = D.heldSingle + idx * n * metaI("diffusion.conditioning.seqChannels");
    snProj = D.heldSnProj + idx * n * perToken;
  } else {
    single = singleConditioning(level);
    size_t pn = ((size_t)n + 15) / 16 * 16;
    half* sn = scratch<half>("dn.sn", pn * Cs);
    static const half* zeroedSn = nullptr; static size_t zeroedSnRows = 0;
    if (pn > (size_t)n && (sn != zeroedSn || pn != zeroedSnRows)) {
      fill(sn + (size_t)n * Cs, 0, (pn - n) * Cs * sizeof(half));
      zeroedSn = sn; zeroedSnRows = pn;
    }
    if (hasW("diffusion.singleCondEmbeddingNormScale"))
      ln(single, sn, n, Cs, "diffusion.singleCondEmbeddingNormScale", "diffusion.singleCondEmbeddingNormOffset");
    else toHalf(single, sn, (size_t)n * Cs);
    snProj = scratch<float>("dn.snProj", pn * perToken);
    lin(sn, "diffusion.singleCondEmbeddingProjection", snProj, pn, Cs, perToken);
  }
  float* gNorm = nullptr; float* gRaw = nullptr; bool permuted = false;
  transformerConditioning(single, level, &gNorm, &gRaw, &permuted);
  stage("conditioning");
  const int C = metaI("diffusion.transformer.channels");
  const bool noResidual = flag("diffusion.transformer.noResidual");
  const size_t prows = (rows + 15) / 16 * 16;
  const bool fuseTx0 = D.nblocks > 0 && C == 768 && perToken == 768;
  half* tx0 = fuseTx0 ? scratch<half>("dt.x", prows * C) : nullptr;
  half* tn0 = (fuseTx0 && noResidual) ? scratch<half>("dt.tn", prows * C) : nullptr;
  const int ldn0 = permuted ? 2 * C : D.ldn;
  const float* gAtt0 = gNorm;
  const float* gFfw0 = permuted ? gNorm + (size_t)n * 2 * C : gAtt0 + 2 * C;
  const float* atomMask = M.f("batch.predDenseAtomMask");
  encoderStep("diffusion.encoder", D.enc, x, atomMask, in, snProj,
              fuseTx0 ? gAtt0 : nullptr, fuseTx0 ? gAtt0 + C : nullptr, tx0,
              tn0 ? gFfw0 : nullptr, tn0 ? gFfw0 + C : nullptr, tn0, ldn0, permuted);
  stage("encoder");
  float* act = D.enc.tokenAct;
  stage("snproj");
  transformer(act, gNorm, gRaw, permuted, fuseTx0);
  stage("transformer");
  half* actn = scratch<half>("dt.x", prows * perToken);
  ln(act, actn, rows, perToken, "diffusion.outputNormScale", "diffusion.outputNormOffset");
  float* upd = atomDecoder(actn);
  stage("decoder");
  Gather q2t = gatherOf("batch.queriesToTokenAtoms");
  AtomShape sh = atomShape();
  size_t q1 = (size_t)sh.subsets * sh.queries;
  float* out = needOut ? scratch<float>("dn.out", total * 3) : nullptr;
  run1d("af3_denoise_out", total * 3,
        DenoiseOutArgs{x, upd, q2t.idx, q2t.mask, atomMask, out, eulerX, total, atoms, q1, skip, outS, eulerScale, 0});
  return out;
}

// ---------------------------------------------------------------- the sampler
static double noiseSchedule(double t, double sigmaData, double sigmaMin, double sigmaMax, double rho) {
  double lo = std::pow(sigmaMin, 1 / rho), hi = std::pow(sigmaMax, 1 / rho);
  return sigmaData * std::pow(hi + t * (lo - hi), rho);
}
SamplerOptions SAMPLER;
bool samplerFlag(const char* f, SamplerOptions& o, bool* bad) {
  auto refuse = [&]() { if (!bad) exit(1); *bad = true; };
  auto value = [&](const char* name, const char* a, const char* b, bool& set) {
    const size_t n = strlen(name);
    if (strncmp(f, name, n)) return false;
    if (!strcmp(f + n, a)) set = false; else if (!strcmp(f + n, b)) set = true;
    else { fprintf(stderr, "%s takes %s or %s, not %s\n", name, a, b, f + n); refuse(); }
    return true;
  };
  if (value("--schedule=", "karras", "geometric", o.geometric)) { o.scheduleGiven = o.given = true; return true; }
  if (value("--update=", "euler", "exponential", o.exponential)) { o.given = true; return true; }
  if (!strcmp(f, "--no-churn")) { o.churn = false; o.given = true; return true; }
  if (!strncmp(f, "--step-scale=", 13)) {
    o.stepScale = atof(f + 13);
    if (o.stepScale <= 0) { fprintf(stderr, "--step-scale takes a positive number, not %s\n", f + 13); refuse(); }
    o.given = true; return true;
  }
  return false;
}
// the augmentation's rotation and translation for every (step, sample), from each sample's seeded stream
static std::vector<float> rotations(int steps, const std::vector<uint64_t>& seeds) {
  int ns = (int)seeds.size();
  std::vector<float> rot((size_t)std::max(steps, 1) * ns * 12);
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
  return rot;
}
// chai-1's sampler (af3-any-model diffusion_head.py, chai1.py's): its schedule (sigma_max 80, rho 7) at the N MIDPOINTS
// t = (2k + 1) / 2N; per transition the augmentation, churn min(80 / N, sqrt 2 - 1) only where 4e-4 <= sigma_prev <= 80,
// noise 1.003 sqrt(max(1e-6, tHat^2 - sigma_prev^2)), and its second-order step - by default ONE call a step, the
// correction's denoised structure taken as the first's, which makes the step x = noisy + 2 dt g1 (cuda/af3's
// measurement: the same bonds for half the calls); CHAI_SECOND_ORDER=1 is chai-lab's two calls.
// 🔴 AND ITS DEFAULT STEP IS THE EXPONENTIAL ONE (cuda/af3 8469ca1): x' = r^S noisy + (1 - r^S) D, r = sigma / tHat, S its
// own 2 - the one-call step above is Euler at that scale - on its own schedule: at the page's 25 steps its 200 steps'
// bonds (mainchain 0.063 -> 0.034 A, ligands 1/6 torn -> 0/6), crystals level (docs/AF3.md, "chai-1's sampler").
// --update=euler is the sampler before; --schedule=geometric its endpoints at the same midpoints, even in log sigma;
// --no-churn, --step-scale=; CHAI_SECOND_ORDER=1 on the exponential update is the exponential Heun step
static void sampleChai(int steps, int ns, size_t atoms, float* dX, const uint64_t* dSeeds, const float* dMask,
                       const std::vector<uint64_t>& seeds, const std::function<void(const float*, int, int)>& onStep) {
  int N = steps, T = N - 1;
  size_t all3 = atoms * 3 * ns;
  static const bool secondOrder = getenv("CHAI_SECOND_ORDER") && atoi(getenv("CHAI_SECOND_ORDER"));
  const bool geometric = SAMPLER.scheduleGiven && SAMPLER.geometric, exponential = SAMPLER.exponential;
  const double S = SAMPLER.stepScale > 0 ? SAMPLER.stepScale : 2.0;
  std::vector<double> levels(N);
  for (int k = 0; k < N; ++k) {
    const double t = (2.0 * k + 1) / (2.0 * N);
    levels[k] = geometric ? 16 * std::exp(std::log(80.0) + t * (std::log(0.0004) - std::log(80.0))) : noiseSchedule(t, 16, 0.0004, 80, 7);
  }
  const double churn = SAMPLER.churn ? std::min(80.0 / N, std::sqrt(2.0) - 1.0) : 0.0;
  std::vector<float> rot = rotations(T, seeds);
  float* dRot = uploadNew(rot.data(), rot.size());
  float* dIn = scratch<float>("sample.noisy", all3);
  float* dG = scratch<float>("sample.grad", all3); float* dNoisy = scratch<float>("sample.noisyKeep", all3);
  run1d("af3_initial_noise", all3, InitNoiseArgs{dX, dSeeds, atoms * 3, all3, (float)levels[0], 0});
  D.plan.clear(); D.planAt = 0;      // (the levels the denoiser will see, in order, for conditioningAhead)
  for (int s = 1; s <= T; ++s) {
    const double prev = levels[s - 1];
    D.plan.push_back((float)(prev * (1 + (prev >= 4e-4 && prev <= 80 ? churn : 0))));
    if (secondOrder) D.plan.push_back((float)levels[s]);
  }
  for (int s = 1; s <= T; ++s) {
    double prev = levels[s - 1], level = levels[s], tHat = prev * (1 + (prev >= 4e-4 && prev <= 80 ? churn : 0)), dt = level - tHat;
    double injected = 1.003 * std::sqrt(std::max(1e-6, tHat * tHat - prev * prev));
    run("af3_augment", Grid{(uint32_t)ns, 1, 1}, 1024, AugmentArgs{dX, dIn, dMask, nullptr, dRot + (size_t)(s - 1) * ns * 12, dSeeds, atoms, atoms * ns, (uint)s,
                                                                   (float)injected});
    if (!secondOrder && exponential) {          // one call: the correction's D taken as this step's (see above)
      float coef = (float)(std::pow(level / tHat, S) - 1);
      const float* d1 = denoise(dIn, (float)tHat, dX, coef, (bool)onStep);
      if (onStep) onStep(d1, s, T);
      continue;
    }
    copy(dNoisy, dIn, all3 * 4);
    const float* d1 = denoise(dIn, (float)tHat);
    if (onStep) onStep(d1, s, T);
    if (!secondOrder) {
      run1d("af3_chai_euler", all3, ChaiEulerArgs{dX, dG, dNoisy, d1, all3, (float)tHat, (float)(S * dt)});
      continue;
    }
    if (exponential) {                          // the exponential Heun step (dG keeps D1 across the second call)
      run1d("af3_exp_heun1", all3, ExpHeun1Args{dX, dG, dNoisy, d1, all3, (float)(level / tHat - 1), 0});
      copy(dIn, dX, all3 * 4);
      const float* d2 = denoise(dIn, (float)level);
      run1d("af3_exp_heun2", all3, ExpHeun2Args{dX, dNoisy, dG, d2, all3, (float)(std::pow(level / tHat, S) - 1), 0});
      continue;
    }
    run1d("af3_chai_euler", all3, ChaiEulerArgs{dX, dG, dNoisy, d1, all3, (float)tHat, (float)dt});
    copy(dIn, dX, all3 * 4);
    const float* d2 = denoise(dIn, (float)level);
    run1d("af3_chai_correct", all3, ChaiCorrectArgs{dX, d2, dG, all3, (float)level, (float)dt});
  }
  release(dRot);
}
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
  float* dX = allocT<float>(all3);
  uint64_t* dSeeds = uploadNew(seeds.data(), ns);
  float* dMask = uploadNew(mask.data(), atoms);
  std::vector<double> levels(steps + 1);
  if (flag("trunk.dialect.chaiSampler")) {
    sampleChai(steps, ns, atoms, dX, dSeeds, dMask, seeds, onStep);
  } else {
    if (SAMPLER.sigmaMax > 0) sigmaMax = SAMPLER.sigmaMax;
    if (SAMPLER.stepScale > 0) stepScale = SAMPLER.stepScale;
    for (int k = 0; k <= steps; ++k)
      levels[k] = SAMPLER.geometric ? 16 * std::exp(std::log(sigmaMax) + (double)k / steps * (std::log(sigmaMin) - std::log(sigmaMax)))
                                    : noiseSchedule((double)k / steps, 16, sigmaMin, sigmaMax, rho);
    // how the schedule ends (cuda/af3/src/sampler.cuh): 1 N points then a 0, 2 the last 0, 3 a 0 appended
    const int tail = M.has(S + "scheduleTail") ? (int)M.meta(S + "scheduleTail") : 0;
    if (tail != 0) {
      const int points = tail == 1 ? steps : steps + 1;
      levels.assign(points + (tail == 1 || tail == 3 ? 1 : 0), 0.0);
      for (int k = 0; k < points; ++k) {
        const double t = points == 1 ? 0.0 : (double)k / (points - 1);
        levels[k] = SAMPLER.geometric ? 16 * std::exp(std::log(sigmaMax) + t * (std::log(sigmaMin) - std::log(sigmaMax)))
                                      : noiseSchedule(t, 16, sigmaMin, sigmaMax, rho);
      }
      if (tail == 2) levels[points - 1] = 0;
      steps = (int)levels.size() - 1;
    }
    const bool realign = flag("trunk.dialect.realignSampler");
    if (!SAMPLER.churn) gamma0 = 0;
    std::vector<float> rot = rotations(steps, seeds);
    float* dRot = uploadNew(rot.data(), rot.size());
    float* dNoisy = scratch<float>("sample.noisy", all3);
    run1d("af3_initial_noise", all3, InitNoiseArgs{dX, dSeeds, n3, all3, (float)levels[0], 0});
    D.plan.clear(); D.planAt = 0;      // (the levels the denoiser will see, for conditioningAhead)
    for (int step = 1; step <= steps; ++step)
      D.plan.push_back((float)(levels[step - 1] * (1 + (levels[step] > gammaMin ? gamma0 : 0))));
    for (int step = 1; step <= steps; ++step) {
      double previous = levels[step - 1], level = levels[step];
      float tHat = (float)(previous * (1 + (level > gammaMin ? gamma0 : 0)));
      double injected = noiseScale * std::sqrt(std::max(0.0, (double)tHat * tHat - previous * previous));
      run("af3_augment", Grid{(uint32_t)ns, 1, 1}, 1024, AugmentArgs{dX, dNoisy, dMask, nullptr, dRot + (size_t)(step - 1) * ns * 12, dSeeds, atoms, atoms * ns,
                                                                     (uint)step, (float)injected});
      const double coef = SAMPLER.exponential ? std::pow(level / tHat, stepScale) - 1 : stepScale * (level - tHat) / tHat;
      if (!realign) {
        const float* d = denoise(dNoisy, tHat, dX, (float)coef, (bool)onStep);
        if (onStep) onStep(d, step, steps);
      } else {
        const float* d = denoise(dNoisy, tHat);
        if (onStep) onStep(d, step, steps);
        // (boltz2: the noisy copy rigid-aligned onto the answer, each sample - rigid.cuh)
        std::vector<float> noisy = download(dNoisy, all3), answer = download(d, all3);
        for (int k = 0; k < ns; ++k) {
          std::vector<float> x(noisy.begin() + (size_t)k * n3, noisy.begin() + (size_t)(k + 1) * n3),
                             t(answer.begin() + (size_t)k * n3, answer.begin() + (size_t)(k + 1) * n3);
          rigidAlign(x, t, mask, (int)atoms);
          std::copy(x.begin(), x.end(), noisy.begin() + (size_t)k * n3);
        }
        upload(dNoisy, noisy.data(), all3 * sizeof(float));
        run1d("af3_euler", all3, EulerArgs{dX, dNoisy, d, all3, (float)coef, 0});
      }
    }
    release(dRot);
  }
  std::vector<float> out = download(dX, all3);
  release(dX); release(dSeeds); release(dMask);
  D.plan.clear(); D.planAt = 0; D.held.clear();
  NS = 1;
  return out;
}
void freeDiffusion() {
  releaseScratch({"dt.", "dc.", "dn.", "dec.", "diffusion.", "atomReference.", "sample.", "ab.", "enc."});
  D = Diffusion{};
}
