// The trunk: embedder, template stack, MSA stack, pairformer, distogram.
// Transcribed from src/af3/trunk/{embedder,template,msa,pairformer,trunk}-reference.js.
#pragma once
#include "pairtrack.cuh"

// ---------------------------------------------------------------- embedder
// pair[i][j] = left[i] + right[j]
__global__ void outerSumK(const float* left, const float* right, float* pair, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  pair[t] = left[(size_t)i * C + c] + right[(size_t)j * C + c];
}
// rows [r0, r0 + cnt) of the pair (as pair rows i*n+j) = left[i] + right[j] + add[row]
__global__ void outerSumRowsK(const float* left, const float* right, const float* add, float* pair, size_t r0, size_t cnt,
                              int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = r0 + t / C; size_t i = ij / n, j = ij % n;
  pair[r0 * C + t] = left[i * C + c] + right[j * C + c] + add[t];
}
// AF3's relative encoding, 139 one-hot columns, folded straight into its projection: each
// pair adds the four or five weight rows its one-hot selects.
__global__ void relativeEncodingK(const int* residueIndex, const int* tokenIndex, const int* asymId,
                                  const int* entityId, const int* symId, const float* Wpos, float* pair,
                                  int n, int C, int maxIdx, int maxChain, size_t p0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * C) return;
  int c = (int)(t % C); size_t ij = p0 + t / C; int i = (int)(ij / n), j = (int)(ij % n);
  int positionBins = 2 * maxIdx + 2;
  auto clamp = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
  bool sameChain = asymId[i] == asymId[j], sameEntity = entityId[i] == entityId[j];
  int offset = clamp(residueIndex[i] - residueIndex[j] + maxIdx, 2 * maxIdx);
  int c0 = sameChain ? offset : 2 * maxIdx + 1;
  bool sameResidue = sameChain && residueIndex[i] == residueIndex[j];
  int tokenOffset = clamp(tokenIndex[i] - tokenIndex[j] + maxIdx, 2 * maxIdx);
  int c1 = positionBins + (sameResidue ? tokenOffset : 2 * maxIdx + 1);
  int relChain = clamp(symId[i] - symId[j] + maxChain, 2 * maxChain);
  int c3 = positionBins * 2 + 1 + (sameEntity ? relChain : 2 * maxChain + 1);
  float v = Wpos[(size_t)c0 * C + c] + Wpos[(size_t)c1 * C + c] + Wpos[(size_t)c3 * C + c];
  if (sameEntity) v += Wpos[(size_t)(positionBins * 2) * C + c];
  pair[t] += v;
}
// msa = one_hot(32) + clip(deletion) + atan(deletion/3)*2/pi, projected, plus the target
// feature's projection broadcast over rows.
// (width 35: an is_paired column, set on the query row (row < n) only where `pairedQuery` - boltz2
// says yes, rosettafold3 carries the column and leaves it zero)
__global__ void msaEmbedK(const int* rows, const float* deletion, const float* Wmsa, const float* fromTarget,
                          float* msa, size_t count, int n, int C, int width, bool pairedQuery) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count * C) return;
  int c = (int)(t % C); size_t row = t / C; int token = (int)(row % n);
  int code = rows[row]; float d = deletion[row];
  float v = (code >= 0 && code < 32) ? Wmsa[(size_t)code * C + c] : 0.f;
  v += fminf(fmaxf(d, 0.f), 1.f) * Wmsa[(size_t)32 * C + c];
  v += atanf(d / 3.f) * (2.f / 3.14159265358979f) * Wmsa[(size_t)33 * C + c];
  if (width > 34 && pairedQuery && row < (size_t)n) v += Wmsa[(size_t)34 * C + c];
  msa[t] = v + fromTarget[(size_t)token * C + c];
}

struct Trunk {
  int n, S;                 // tokens, MSA rows used
  int C, Cs, Cm, F;         // pair, single, msa channels, target-feature width
  float *pair, *single, *msa, *targetFeat, *pairMask, *seqMask, *msaMask;
  float *prevPair, *prevSingle;
  bool inPlaceRecycle = false;         // the recycled pair is t.pair (makeTrunk)
  float* pairHost = nullptr;           // tier 2: the pair in host memory, t.pair null (makeTrunk)
  int* msaRows; float* deletion;
  bool swap, divide;
  Pair P() const { return Pair{pair, pairHost, n, C}; }
};
// Tier 2 where the pair and what the trunk holds beside it would not fit the card, asked of what is free once
// the weights are up: beside its pair the trunk's scratch measured 0.63x the pair with tier 1's levers taken
// (11.6 GB beside an 18.4 GB pair at 6000 tokens on 40 GB), and more on a card short of shared memory, whose
// unfused kernels hold more - 1.65x put 3500 tokens on a simulated T4 into tier 1, which ran out there; 1.8x
// and a 20th of the card spare do not. Scratch held counts as free (the target feature's encoder holds 3.6 GB
// here, given back as the trunk starts): 6000 tokens on 40 GB stay in tier 1, where they fold.
// ...and whether tier 2 itself fits this machine, asked before anything is pinned or folded: an input past
// it is refused at once, with the longest this machine takes, rather than failing hours in. Host memory: the
// trunk's pair and, beside it, the template stack's (the trunk) or the diffusion's LayerNorm'd pair (after it)
// - ~770 bytes a token pair, 89 GB at 10761 tokens - or the confidence head's copy of the pair where a fold
// has more than one sample (1024 bytes), with 4 GB spare. The card: the triangle's fixed operand
// whole (256 bytes a pair at 128 channels) and ~5 GB of windows, biases and scratch beside it.
inline int FOLD_RUNS = 1;      // samples x seeds of the fold (af3.cu): past one, the confidence head works in a copy
inline void tier2Fits(int n, int C) {
  int Ct = M.has("trunk.template.channels") ? (int)M.meta("trunk.template.channels") : 0;
  int Cz = (int)M.meta("diffusion.conditioning.pairChannels");
  int np = (n + 7) / 8 * 8;
  size_t perPair = (size_t)C * 4 + std::max<size_t>({ (size_t)Ct * 4, (size_t)Cz * 2, FOLD_RUNS > 1 ? (size_t)C * 4 : 0 }),
         spareHost = (size_t)4 << 30;
  size_t host = hostAvailable() + PARK_HAVE, f, t; CK(cudaMemGetInfo(&f, &t));
  for (auto& [name, slot] : SCRATCH) f += slot.second;
  size_t perOperand = (size_t)C * 2, spareCard = (size_t)5 << 30;
  bool hostOk = (size_t)n * n * perPair + spareHost <= host, cardOk = (size_t)np * np * perOperand + spareCard <= f;
  if (hostOk && cardOk) return;
  auto most = [](size_t budget, size_t spare, size_t per) { return budget > spare ? (int)std::sqrt((double)(budget - spare) / per) : 0; };
  int byHost = most(host, spareHost, perPair), byCard = most(f, spareCard, perOperand);
  fprintf(stderr, "%d tokens do not fit this machine: past what the card holds the pair waits in host memory, which "
          "needs %.0f GB of it (%.0f available) and %.0f GB of the card (%.0f free) - this machine folds up to ~%d tokens "
          "(its %s is the limit)\n", n, ((size_t)n * n * perPair + spareHost) / 1e9, host / 1e9,
          ((size_t)np * np * perOperand + spareCard) / 1e9, f / 1e9, std::min(byHost, byCard),
          byHost < byCard ? "host memory" : "card");
  exit(1);
}
inline bool hostTier(size_t pairs, int C) {
  if (HOST_FORCED) return true;
  size_t f, t; CK(cudaMemGetInfo(&f, &t));
  for (auto& [name, slot] : SCRATCH) f += slot.second;
  return pairs * C * 4 * 180 / 100 + t / 20 > f;
}

inline Trunk makeTrunk(const float* targetFeatHost, int msaCap) {
  Trunk t{};
  t.n = (int)M.meta("batch.tokens");
  t.C = (int)M.meta("trunk.embedder.pairChannels");
  t.Cs = (int)M.meta("trunk.embedder.singleChannels");
  t.Cm = (int)M.meta("trunk.embedder.msaChannels");
  t.F = (int)M.meta("trunk.embedder.targetFeatWidth");
  t.S = std::min((int)M.meta("batch.sequences"), msaCap);
  t.swap = M.flag("trunk.dialect.swapTransposedBias");
  t.divide = M.flag("trunk.dialect.triangleMulDivideByLength");
  size_t pairs = (size_t)t.n * t.n;
  // tier 2: the pair lives in host memory (the parking buffer, where diffusion's conditioning already reads a
  // parked pair from), re-embedded in place, and passes through the card a window at a time
  if (hostTier(pairs, t.C)) {
    tier2Fits(t.n, t.C);
    t.pairHost = parkBuffer(pairs * t.C * 4);
    memset(t.pairHost, 0, pairs * t.C * 4);
    size_t f, tot; CK(cudaMemGetInfo(&f, &tot));
    fprintf(stderr, "the pair (%.1f GB, %.1f GB of the card free) does not fit: it stays in host memory and passes "
            "through in windows\n", pairs * t.C * 4 / 1e9, f / 1e9);
  }
  t.pair = t.pairHost ? nullptr : dalloc(pairs * t.C); t.single = dalloc((size_t)t.n * t.Cs);
  t.msa = dalloc((size_t)t.S * t.n * t.Cm);
  // on a card short of room the recycled pair is the pair itself, re-embedded in place (embed): no second
  // pair-sized tensor - 9 GB at 4192 tokens - and the first pass starts from a zeroed pair
  t.inPlaceRecycle = shortPair(pairs, t.C) || t.pairHost;
  t.prevPair = t.inPlaceRecycle ? nullptr : dalloc(pairs * t.C); t.prevSingle = dalloc((size_t)t.n * t.Cs);
  if (!t.pairHost) CK(cudaMemset(t.inPlaceRecycle ? t.pair : t.prevPair, 0, pairs * t.C * 4));
  CK(cudaMemset(t.prevSingle, 0, (size_t)t.n * t.Cs * 4));
  t.targetFeat = upload(targetFeatHost, (size_t)t.n * t.F);
  std::vector<float> seq(M.f("batch.seqMask"), M.f("batch.seqMask") + t.n), pm(pairs);
  for (int i = 0; i < t.n; ++i) for (int j = 0; j < t.n; ++j) pm[(size_t)i * t.n + j] = seq[i] * seq[j];
  t.seqMask = upload(seq.data(), t.n); t.pairMask = upload(pm.data(), pairs);
  // the first S rows: AF3 keeps the first num_msa after its (here identity) shuffle
  size_t rows = (size_t)t.S * t.n;
  t.msaRows = upload(M.i("batch.msa"), rows);
  t.deletion = upload(M.f("batch.deletionMatrix"), rows);
  t.msaMask = upload(M.f("batch.msaMask"), rows);
  return t;
}

inline void freeTrunk(Trunk& t) {
  for (void* p : {(void*)t.pair, (void*)t.single, (void*)t.msa, (void*)t.targetFeat, (void*)t.pairMask, (void*)t.seqMask,
                  (void*)t.msaMask, (void*)t.prevPair, (void*)t.prevSingle, (void*)t.msaRows, (void*)t.deletion})
    if (p) CK(cudaFree(p));
  t = Trunk{};
}

template <class T> void templateEmbedding(Trunk& t, float* pairOut);     // adds its output into pairOut
template <class T> void templateEmbeddingHost(Trunk& t);                 // ...into a pair in host memory
__global__ void onehotK(const int* idx, float* out, int n, int classes) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * classes) return;
  int c = (int)(t % classes); int v = idx[t / classes];
  out[t] = v == c ? 1.f : 0.f;
}
__global__ void bondEmbedK(float* pair, const float* bonds, const float* w, size_t pairs, int C, size_t p0 = 0) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < pairs * C) pair[t] += bonds[p0 + t / C] * w[t % C];
}

// boltz2's two extra z-init terms: token_bonds_type_embed[bond order] (row 0 on an unbonded pair,
// trained non-zero) plus the contact conditioning's unspecified-restraint constant
__global__ void bondTypeEmbedK(float* pair, const float* orders, const float* table, const float* unspecified,
                               size_t pairs, int C, size_t p0 = 0) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  int o = orders ? (int)orders[p0 + t / C] : 0;
  if (o < 0 || o >= 7) o = 0;
  pair[t] += table[o * C + t % C] + unspecified[t % C];
}
inline void bondTypeEmbed(float* pair, size_t pairs, int C, const std::string& pre, size_t p0 = 0) {
  if (!hasW(pre + "tokenBondsTypeEmbed")) return;
  bondTypeEmbedK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, M.has("batch.bondOrderMatrix") ? Fdev("batch.bondOrderMatrix") : nullptr,
    W(pre + "tokenBondsTypeEmbed"), W(pre + "contactEncodingUnspecified"), pairs, C, p0);
}

// The embedder up to and including the template term. `onSeam` sees the pair after each.
template <class T>
void embed(Trunk& t, const std::function<void(const char*, const float*, size_t)>& onSeam) {
  int n = t.n, C = t.C; size_t pairs = (size_t)n * n;
  const std::string E = "trunk.embedder.";
  float* left = scratch<float>("emb.left", (size_t)n * C);
  float* right = scratch<float>("emb.right", (size_t)n * C);
  // the pair from target_feat (AF3), or from s_init = target_feat's single projection (OpenDDE)
  const float* pairSource = t.targetFeat; int sourceWidth = t.F;
  if (M.flag("trunk.dialect.pairInitFromSingle")) {
    float* sInit = scratch<float>("emb.sInit", (size_t)n * t.Cs);
    linear<float, float>(t.targetFeat, sInit, n, t.F, t.Cs, E + "singleActivations");
    pairSource = sInit; sourceWidth = t.Cs;
  }
  linear<float, float>(pairSource, left, n, sourceWidth, C, E + "leftSingle");
  linear<float, float>(pairSource, right, n, sourceWidth, C, E + "rightSingle");
  if (t.pairHost) {
    // tier 2: each window of rows re-embedded in place (as below), then its relative encoding and bonds
    size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / C));
    T* ln = scratch<T>("emb.prevln", per * C);
    float* prev = scratch<float>("emb.prevproj", per * C);
    forRowWindows(t.P(), true, [&](float* w, size_t i0, size_t I) {
      size_t p0 = i0 * n, cnt = I * n;
      for (size_t r0 = 0; r0 < cnt; r0 += per) {
        size_t r = std::min(per, cnt - r0);
        layerNorm2<float, T>(w + r0 * C, ln, r, C, E + "prevEmbeddingNormScale", E + "prevEmbeddingNormOffset");
        linear<T, float>(ln, prev, r, C, C, E + "prevEmbedding");
        outerSumRowsK<<<blocks(r * C), 256, 0, STREAM>>>(left, right, prev, shifted(w, p0, C), p0 + r0, r, n, C);
      }
      relativeEncodingK<<<blocks(cnt * C), 256, 0, STREAM>>>(
        Idev("batch.features.residueIndex"), Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"),
        Idev("batch.features.entityId"), Idev("batch.features.symId"), W(E + "positionActivations"), w, n, C, 32, 2, p0, cnt);
      if (M.has("batch.bondMatrix")) {
        if (lenW(E + "bondEmbedding") != (size_t)C) { fprintf(stderr, "bondEmbedding is not 1 x %d\n", C); exit(1); }
        bondEmbedK<<<blocks(cnt * C), 256, 0, STREAM>>>(w, Fdev("batch.bondMatrix"), W(E + "bondEmbedding"), cnt, C, p0);
      }
      bondTypeEmbed(w, cnt, C, E, p0);
    });
    templateEmbeddingHost<T>(t);
  } else {
  if (t.inPlaceRecycle) {
    // in place, a chunk of rows at a time: each row's new value reads only the same row of the last pass's
    // pair, so the projection of the old rows is taken first and the rows are then overwritten with
    // left + right + it
    size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / C));
    T* ln = scratch<T>("emb.prevln", per * C);
    float* prev = scratch<float>("emb.prevproj", per * C);
    for (size_t r0 = 0; r0 < pairs; r0 += per) {
      size_t r = std::min(per, pairs - r0);
      layerNorm2<float, T>(t.pair + r0 * C, ln, r, C, E + "prevEmbeddingNormScale", E + "prevEmbeddingNormOffset");
      linear<T, float>(ln, prev, r, C, C, E + "prevEmbedding");
      outerSumRowsK<<<blocks(r * C), 256, 0, STREAM>>>(left, right, prev, t.pair, r0, r, n, C);
    }
  } else {
  outerSumK<<<blocks(pairs * C), 256, 0, STREAM>>>(left, right, t.pair, n, C);
  onSeam("z_before_prev", t.pair, pairs * C);
  // the recycled pair: LayerNorm then projection, which is NOT zero on the first pass - on a card short
  // of room in row chunks
  bool tight = shortPair(pairs, C);
  size_t per = tight ? std::max<size_t>(1, std::min(pairs, CHUNK / C)) : pairs;
  T* ln = scratch<T>("emb.prevln", per * C);
  for (size_t r0 = 0; r0 < pairs; r0 += per) {
    size_t r = std::min(per, pairs - r0);
    layerNorm2<float, T>(t.prevPair + r0 * C, ln, r, C, E + "prevEmbeddingNormScale", E + "prevEmbeddingNormOffset");
    linear<T, float>(ln, t.pair + r0 * C, r, C, C, E + "prevEmbedding", false, 1.f);
  }
  }
  onSeam("z_after_prev", t.pair, pairs * C);
  relativeEncodingK<<<blocks(pairs * C), 256, 0, STREAM>>>(
    Idev("batch.features.residueIndex"), Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"),
    Idev("batch.features.entityId"), Idev("batch.features.symId"), W(E + "positionActivations"),
    t.pair, n, C, 32, 2, 0, pairs);
  if (M.has("batch.bondMatrix")) {
    // bond_embedding: bias-free, one input column (the token-pair contact); zero for a polymer
    // without links
    if (lenW(E + "bondEmbedding") != (size_t)C) { fprintf(stderr, "bondEmbedding is not 1 x %d\n", C); exit(1); }
    bondEmbedK<<<blocks(pairs * C), 256, 0, STREAM>>>(t.pair, Fdev("batch.bondMatrix"), W(E + "bondEmbedding"), pairs, C);
  }
  bondTypeEmbed(t.pair, pairs, C, E);
  onSeam("z_init_generic", t.pair, pairs * C);
  templateEmbedding<T>(t, t.pair);     // its projection accumulated into the pair (it reads the pair first)
  onSeam("z_after_template", t.pair, pairs * C);
  }
  // msa and single
  float* fromTarget = scratch<float>("emb.fromTarget", (size_t)n * t.Cm);
  linear<float, float>(t.targetFeat, fromTarget, n, t.F, t.Cm, E + "extraMsaTargetFeat");
  size_t rows = (size_t)t.S * n;
  int msaWidth = (int)(lenW(E + "msaActivations") / t.Cm);
  if (msaWidth != 34 && msaWidth != 35) { fprintf(stderr, "msa feature width %d\n", msaWidth); exit(1); }
  msaEmbedK<<<blocks(rows * t.Cm), 256, 0, STREAM>>>(t.msaRows, t.deletion, W(E + "msaActivations"),
                                                     fromTarget, t.msa, rows, n, t.Cm, msaWidth,
                                                     M.flag("trunk.dialect.msaPairedQueryRow"));
  linear<float, float>(t.targetFeat, t.single, n, t.F, t.Cs, E + "singleActivations");
  T* sln = scratch<T>("emb.prevsln", (size_t)n * t.Cs);
  layerNorm2<float, T>(t.prevSingle, sln, n, t.Cs, E + "prevSingleEmbeddingNormScale",
                       E + "prevSingleEmbeddingNormOffset");
  linear<T, float>(sln, t.single, n, t.Cs, t.Cs, E + "prevSingleEmbedding", false, 1.f);
}

// ---------------------------------------------------------------- template stack
// act[i][j] += row[j] + column[i]   (aatype one-hot projections, per axis)
__global__ void addRowColumnK(float* act, const float* row, const float* col, int n, int C, size_t p0 = 0,
                              size_t cnt = SIZE_MAX) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (cnt < (size_t)n * n ? cnt : (size_t)n * n) * C) return;
  int c = (int)(t % C); size_t ij = p0 + t / C; int i = (int)(ij / n), j = (int)(ij % n);
  act[t] += row[(size_t)j * C + c] + col[(size_t)i * C + c];
}
__global__ void reluScaleK(float* x, float s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = fmaxf(0.f, x[i] * s);
}
// act += the geometry features of a real template slot: the distogram (one-hot bins) projected,
// and five scalar features each times a per-channel weight (AF3's num_input_dims=0).
__global__ void templateGeometryK(float* act, const float* dgram, const float* pb, const float* uv, const float* bb,
                                  const float* W0, const float* W1, const float* W4, const float* W5,
                                  const float* W6, const float* W7, size_t pairs, int C, int bins, size_t p0 = 0) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  int c = (int)(t % C); size_t p = p0 + t / C;
  float v = 0.f;
  for (int b = 0; b < bins; ++b) { float d = dgram[p * bins + b]; if (d != 0.f) v += d * W0[(size_t)b * C + c]; }
  v += pb[p] * W1[c] + uv[p * 3] * W4[c] + uv[p * 3 + 1] * W5[c] + uv[p * 3 + 2] * W6[c] + bb[p] * W7[c];
  act[t] += v;
}
// The template embedding, over the exporter's passes (built as the page's trunk builds them):
// each pass's input is the query term plus - nine-projection embedder - its aatype one-hot
// projected along each axis and, for a real template, its geometry, or - the fused embedder
// (protenix2, boltz2, rf3) - its feature columns projected; then the stack's blocks (wrapped in a
// residual under boltz2), the output LayerNorm, summed with the pass's repeat weight; the sum
// divided by every slot, relu, projected.
__global__ void scaleK(float* x, float s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = s * x[i];
}
__global__ void addScaledK(float* y, const float* x, float s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += s * x[i];
}
template <class T>
void templateEmbedding(Trunk& t, float* out) {
  int n = t.n, Cq = t.C; size_t pairs = (size_t)n * n;
  const std::string P = "trunk.template.";
  int Ct = (int)M.meta(P + "channels");
  bool fused = M.flag(P + "fused");
  if (!M.has("template.passes")) {
    fprintf(stderr, "this input was exported before template passes: export it again with export-model.mjs\n"); exit(1);
  }
  int passes = (int)M.meta("template.passes"), templates = (int)M.meta("template.templates");
  int width = (int)M.meta("template.featureWidth");
  bool outer = M.flag("template.outerResidual");
  if (fused && lenW(P + "aProjection") != (size_t)width * Ct) {
    fprintf(stderr, "template features are %d wide, the projection takes %zu\n", width, lenW(P + "aProjection") / Ct); exit(1);
  }
  // The query term, LN(pair) projected - the same for every pass. On a card short of room it is not
  // kept: each pass recomputes it into its activation in row chunks (the normalised pair is read once,
  // by the projection), which is a pair-sized tensor fewer for a LayerNorm and a narrow GEMM a pass.
  bool tight = shortPair(pairs, Cq);
  size_t per = tight ? std::max<size_t>(1, std::min(pairs, CHUNK / Cq)) : pairs;
  T* ln = scratch<T>("tmpl.ln", per * Cq);
  auto queryInto = [&](float* dst) {
    for (size_t r0 = 0; r0 < pairs; r0 += per) {
      size_t r = std::min(per, pairs - r0);
      layerNorm2<float, T>(t.pair + r0 * Cq, ln, r, Cq, P + "queryEmbeddingNormScale", P + "queryEmbeddingNormOffset");
      linear<T, float>(ln, dst + r0 * Ct, r, Cq, Ct, P + (fused ? "zProjection" : "templatePairEmbedding8"));
    }
  };
  float* query = tight ? nullptr : scratch<float>("tmpl.query", pairs * Ct);
  if (!tight) queryInto(query);
  float* act = scratch<float>("tmpl.act", pairs * Ct);
  float* before = outer ? scratch<float>("tmpl.before", pairs * Ct) : nullptr;
  // one pass that counts (no templates, or one): its normalised activation IS the sum, scaled in place by
  // its repeat - the same products in the same order as adding it to a zeroed sum, without the sum
  int live = 0; for (int k = 0; k < passes; ++k) live += M.meta("template." + std::to_string(k) + ".repeat") != 0.;
  float* summed = live == 1 ? act : scratch<float>("tmpl.summed", pairs * Ct);
  if (live != 1) CK(cudaMemsetAsync(summed, 0, pairs * Ct * 4, STREAM));
  float* oh = scratch<float>("tmpl.onehot", (size_t)n * 31);
  float* row = scratch<float>("tmpl.row", (size_t)n * Ct); float* col = scratch<float>("tmpl.col", (size_t)n * Ct);
  int nb = 0; while (M.has(P + "blocks." + std::to_string(nb) + ".pairTransition.transition1")) ++nb;
  for (int k = 0; k < passes; ++k) {
    std::string S = "template." + std::to_string(k) + ".";
    float repeat = (float)M.meta(S + "repeat");
    if (repeat == 0.f) continue;                // a slot weighed zero (boltz2's empty ones)
    if (tight) queryInto(act);
    else CK(cudaMemcpyAsync(act, query, pairs * Ct * 4, cudaMemcpyDeviceToDevice, STREAM));
    if (fused) {
      linear<float, float>(Fdev(S + "features"), act, pairs, width, Ct, P + "aProjection", false, 1.f);
    } else {
      onehotK<<<blocks((size_t)n * 31), 256, 0, STREAM>>>(Idev(S + "aatype"), oh, n, 31);   // on the device: capturable
      linear<float, float>(oh, row, n, 31, Ct, P + "templatePairEmbedding2");
      linear<float, float>(oh, col, n, 31, Ct, P + "templatePairEmbedding3");
      addRowColumnK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(act, row, col, n, Ct);
      if (M.has(S + "distogram")) {
        int bins = (int)(lenW(P + "templatePairEmbedding0") / Ct);
        if (M.len(S + "distogram") != pairs * bins) { fprintf(stderr, "%sdistogram is not %zu x %d\n", S.c_str(), pairs, bins); exit(1); }
        templateGeometryK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(act, Fdev(S + "distogram"), Fdev(S + "pseudoBetaMask2d"),
          Fdev(S + "unitVector"), Fdev(S + "backboneMask2d"), W(P + "templatePairEmbedding0"), W(P + "templatePairEmbedding1"),
          W(P + "templatePairEmbedding4"), W(P + "templatePairEmbedding5"), W(P + "templatePairEmbedding6"),
          W(P + "templatePairEmbedding7"), pairs, Ct, bins);
      }
    }
    if (outer) CK(cudaMemcpyAsync(before, act, pairs * Ct * 4, cudaMemcpyDeviceToDevice, STREAM));
    // with one pass the trunk's pair is read before the stack (the query) and after it (the output) and
    // not in between: on a card short of room it waits in host memory while the stack runs
    bool parked = live == 1 && tight && out == t.pair && parkWorthIt(pairs * Cq * 4);
    if (parked) { parkToHost(t.pair, pairs * Cq * 4); out = nullptr; }
    for (int b = 0; b < nb; ++b) {
      std::string B = P + "blocks." + std::to_string(b);
      int factor = (int)(lenW(B + ".pairTransition.transition1") / ((size_t)Ct * Ct * 2));
      pairUpdates<T>(act, t.pairMask, n, Ct, B, t.swap, t.divide, factor);
    }
    if (outer) addK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(act, before, pairs * Ct);
    // in place: act is the pass's own and is written afresh by the next (one warp a row, each lane
    // reading an element before writing it)
    layerNorm2<float, float>(act, act, pairs, Ct, P + "outputLayerNormScale", P + "outputLayerNormOffset");
    if (live == 1) scaleK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(act, repeat, pairs * Ct);
    else addScaledK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(summed, act, repeat, pairs * Ct);
    if (parked) { releaseScratch({ "tri.", "trib.", "grid.", "tr.", "st." }); unparkFromHost(t.pair, pairs * Cq * 4); out = t.pair; }
  }
  // divided by every slot (not the real ones), relu, projected
  reluScaleK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(summed, 1.f / (1e-7f + templates), pairs * Ct);
  linear<float, float>(summed, out, pairs, Ct, Cq, P + "outputLinear", false, 1.f);
}

// rows [p0, p0 + cnt) of a pair-shaped input feature `width` wide, from the export's host copy into scratch
// `slot`, the pointer moved back so the feature's absolute pair indices land in it (shifted)
inline float* featureRows(const std::string& key, size_t p0, size_t cnt, size_t width, const std::string& slot) {
  float* d = scratch<float>(slot, cnt * width);
  CK(cudaMemcpyAsync(d, M.f(key) + p0 * width, cnt * width * 4, cudaMemcpyHostToDevice, STREAM));
  return shifted(d, p0, (int)width);
}
// The template embedding in tier 2: its activation (with the sum over passes where more than one counts, and
// boltz2's residual copy) a pair in host memory too, each step a window of rows at a time - the trunk's pair
// rows and the pass's features beside it staged up, never a whole pair-sized feature on the card.
template <class T>
void templateEmbeddingHost(Trunk& t) {
  int n = t.n, Cq = t.C; size_t pairs = (size_t)n * n;
  const std::string P = "trunk.template.";
  int Ct = (int)M.meta(P + "channels");
  bool fused = M.flag(P + "fused");
  if (!M.has("template.passes")) {
    fprintf(stderr, "this input was exported before template passes: export it again with export-model.mjs\n"); exit(1);
  }
  int passes = (int)M.meta("template.passes"), templates = (int)M.meta("template.templates");
  int width = (int)M.meta("template.featureWidth");
  bool outer = M.flag("template.outerResidual");
  if (fused && lenW(P + "aProjection") != (size_t)width * Ct) {
    fprintf(stderr, "template features are %d wide, the projection takes %zu\n", width, lenW(P + "aProjection") / Ct); exit(1);
  }
  int live = 0; for (int k = 0; k < passes; ++k) live += M.meta("template." + std::to_string(k) + ".repeat") != 0.;
  Pair act{nullptr, hostKept("template.act", pairs * Ct), n, Ct};
  Pair summed = live == 1 ? act : Pair{nullptr, hostKept("template.summed", pairs * Ct), n, Ct};
  Pair before{nullptr, outer ? hostKept("template.before", pairs * Ct) : nullptr, n, Ct};
  if (live != 1) { CK(cudaStreamSynchronize(STREAM)); memset(summed.host, 0, pairs * Ct * 4); }   // (host writes: the device done)
  Pair trunk = t.P();
  size_t per = std::max<size_t>(1, std::min(pairs, CHUNK / Cq));
  T* ln = scratch<T>("tmpl.ln", per * Cq);
  float* oh = scratch<float>("tmpl.onehot", (size_t)n * 31);
  float* row = scratch<float>("tmpl.row", (size_t)n * Ct); float* col = scratch<float>("tmpl.col", (size_t)n * Ct);
  int nb = 0; while (M.has(P + "blocks." + std::to_string(nb) + ".pairTransition.transition1")) ++nb;
  for (int k = 0; k < passes; ++k) {
    std::string S = "template." + std::to_string(k) + ".";
    float repeat = (float)M.meta(S + "repeat");
    if (repeat == 0.f) continue;
    int bins = 0;
    if (!fused) {
      onehotK<<<blocks((size_t)n * 31), 256, 0, STREAM>>>(Idev(S + "aatype"), oh, n, 31);
      linear<float, float>(oh, row, n, 31, Ct, P + "templatePairEmbedding2");
      linear<float, float>(oh, col, n, 31, Ct, P + "templatePairEmbedding3");
      bins = M.has(S + "distogram") ? (int)(lenW(P + "templatePairEmbedding0") / Ct) : 0;
      if (bins && M.len(S + "distogram") != pairs * bins) { fprintf(stderr, "%sdistogram is not %zu x %d\n", S.c_str(), pairs, bins); exit(1); }
    }
    // the pass's input: the query term LN(pair) projected, then its features (each window's rows of them)
    forRowWindows(act, true, [&](float* w, size_t i0, size_t I) {
      size_t p0 = i0 * n, cnt = I * n;
      const float* q = stageRows(trunk, i0, I, "hp.win2");
      for (size_t r0 = 0; r0 < cnt; r0 += per) {
        size_t r = std::min(per, cnt - r0);
        layerNorm2<float, T>(q + r0 * Cq, ln, r, Cq, P + "queryEmbeddingNormScale", P + "queryEmbeddingNormOffset");
        linear<T, float>(ln, w + r0 * Ct, r, Cq, Ct, P + (fused ? "zProjection" : "templatePairEmbedding8"));
      }
      if (fused) {
        const float* f = featureRows(S + "features", p0, cnt, width, "tmpl.featRows");
        linear<float, float>(f + p0 * width, w, cnt, width, Ct, P + "aProjection", false, 1.f);
      } else {
        addRowColumnK<<<blocks(cnt * Ct), 256, 0, STREAM>>>(w, row, col, n, Ct, p0, cnt);
        if (bins)
          templateGeometryK<<<blocks(cnt * Ct), 256, 0, STREAM>>>(w, featureRows(S + "distogram", p0, cnt, bins, "tmpl.dgRows"),
            featureRows(S + "pseudoBetaMask2d", p0, cnt, 1, "tmpl.pbRows"), featureRows(S + "unitVector", p0, cnt, 3, "tmpl.uvRows"),
            featureRows(S + "backboneMask2d", p0, cnt, 1, "tmpl.bbRows"), W(P + "templatePairEmbedding0"),
            W(P + "templatePairEmbedding1"), W(P + "templatePairEmbedding4"), W(P + "templatePairEmbedding5"),
            W(P + "templatePairEmbedding6"), W(P + "templatePairEmbedding7"), cnt, Ct, bins, p0);
      }
      if (outer) CK(cudaMemcpyAsync(before.host + p0 * Ct, w, cnt * Ct * 4, cudaMemcpyDeviceToHost, STREAM));
    });
    for (int b = 0; b < nb; ++b) {
      std::string B = P + "blocks." + std::to_string(b);
      int factor = (int)(lenW(B + ".pairTransition.transition1") / ((size_t)Ct * Ct * 2));
      pairUpdates<T>(act, t.pairMask, B, t.swap, t.divide, factor, nullptr,
                     b + 1 < nb ? P + "blocks." + std::to_string(b + 1) + ".triangleMultiplicationOutgoing" : "");
    }
    // (boltz2's residual around the stack,) the output LayerNorm in place, scaled by the pass's repeat - or
    // added so into the sum
    forRowWindows(act, live == 1, [&](float* w, size_t i0, size_t I) {
      size_t cnt = I * n;
      if (outer) addK<<<blocks(cnt * Ct), 256, 0, STREAM>>>(w, stageRows(before, i0, I, "hp.win2"), cnt * Ct);
      layerNorm2<float, float>(w, w, cnt, Ct, P + "outputLayerNormScale", P + "outputLayerNormOffset");
      if (live == 1) { scaleK<<<blocks(cnt * Ct), 256, 0, STREAM>>>(w, repeat, cnt * Ct); return; }
      float* sw = stageRows(summed, i0, I, "hp.win2");
      addScaledK<<<blocks(cnt * Ct), 256, 0, STREAM>>>(sw, w, repeat, cnt * Ct);
      commitRows(summed, i0, I, sw);
    });
  }
  // divided by every slot (not the real ones), relu, projected into the trunk's pair
  forRowWindows(summed, false, [&](float* w, size_t i0, size_t I) {
    size_t cnt = I * n;
    reluScaleK<<<blocks(cnt * Ct), 256, 0, STREAM>>>(w, 1.f / (1e-7f + templates), cnt * Ct);
    float* q = stageRows(trunk, i0, I, "hp.win2");
    linear<float, float>(w, q, cnt, Ct, Cq, P + "outputLinear", false, 1.f);
    commitRows(trunk, i0, I, q);
  });
  releaseScratch({ "hp.win2", "tmpl." });
}

// ---------------------------------------------------------------- MSA stack
template <class T>
__global__ void biasRowsK(T* x, const float* b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] = fromF<T>(toF(x[t]) + b[t % C]);
}
template <class T>
__global__ void scaleRowsK(T* x, const float* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] = fromF<T>(toF(x[t]) * mask[t / C]);
}
// T: the projections and the contraction's inputs (f16 on the fast path, tensor cores, f32
// accumulation); the contraction's output, the mask normaliser and the residual stay f32.
template <class T>
void outerProductMean(Trunk& t, const std::string& pre) {
  int n = t.n, S = t.S, Cm = t.Cm, C = t.C;
  int O = (int)M.meta(pre + ".outerChannels");
  size_t rows = (size_t)S * n;
  T* ln = scratch<T>("opm.ln", rows * Cm);
  layerNorm2<float, T>(t.msa, ln, rows, Cm, pre + ".layerNormInputScale", pre + ".layerNormInputOffset");
  T* L = scratch<T>("opm.left", rows * O); T* R = scratch<T>("opm.right", rows * O);
  linear<T, T>(ln, L, rows, Cm, O, pre + ".leftProjection");
  linear<T, T>(ln, R, rows, Cm, O, pre + ".rightProjection");
  if (hasW(pre + ".leftProjectionBias")) {      // rosettafold3's biased projections, before the mask
    biasRowsK<T><<<blocks(rows * O), 256, 0, STREAM>>>(L, W(pre + ".leftProjectionBias"), rows, O);
    biasRowsK<T><<<blocks(rows * O), 256, 0, STREAM>>>(R, W(pre + ".rightProjectionBias"), rows, O);
  }
  scaleRowsK<T><<<blocks(rows * O), 256, 0, STREAM>>>(L, t.msaMask, rows, O);
  scaleRowsK<T><<<blocks(rows * O), 256, 0, STREAM>>>(R, t.msaMask, rows, O);
  // norm[i][j] = sum_s mask[s][i] mask[s][j]
  float* norm = scratch<float>("opm.norm", (size_t)n * n);
  const float one = 1.f, zero = 0.f;
  CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, n, n, S, &one, t.msaMask, n, t.msaMask, n, &zero, norm, n));
  // in blocks of query rows i: P[(i,c),(j,e)] = sum_s L[s,i,c] R[s,j,e]
  size_t per = (size_t)n * O * O;
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(n, CHUNK / per));
  T* P = scratch<T>("opm.Pt", (size_t)Bi * per);     // in T: the permute rounded it to T anyway
  T* Pp = scratch<T>("opm.Pp", (size_t)Bi * per);
  float* X = scratch<float>("opm.X", (size_t)Bi * n * C);
  bool after = M.flag("trunk.dialect.opmBiasAfterNorm");
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  // (window by window where the pair is in host memory: a block's rows are a window's)
  forRowWindows(t.P(), true, [&](float* base, size_t w0, size_t Wn) {
  for (int i0 = (int)w0; i0 < (int)(w0 + Wn); i0 += Bi) {
    int bi = std::min(Bi, (int)(w0 + Wn) - i0);
    // row-major P (bi*O x n*O) = L_blk^T R where L_blk is [S][bi*O] with row stride n*O.
    // col-major: P^T (n*O x bi*O) = R^T(op N on R as (n*O x S), ld n*O) * L_blk (op T)
    CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_T, n * O, bi * O, S, &one, R, cudaType<T>(), n * O,
                    L + (size_t)i0 * O, cudaType<T>(), n * O, &zero, P, cudaType<T>(), n * O, CUBLAS_COMPUTE_32F, algo));
    // (16 bytes a thread where the type and width allow: native/af2's opmPermuteHK)
    if constexpr (std::is_same_v<T, half>) {
      if (O % 8 == 0) opmPermuteHK<<<blocks((size_t)bi * per / 8), 256, 0, STREAM>>>(P, Pp, bi, n, O);
      else opmPermuteK<T><<<blocks((size_t)bi * per), 256, 0, STREAM>>>(P, Pp, bi, n, O);
    } else {
      opmPermuteK<T><<<blocks((size_t)bi * per), 256, 0, STREAM>>>(P, Pp, bi, n, O);
    }
    linear<T, float>(Pp, X, (size_t)bi * n, O * O, C, pre + ".outputW");
    opmAddK<<<blocks((size_t)bi * n * C), 256, 0, STREAM>>>(shifted(base, w0 * n, C), X, W(pre + ".outputB"), norm, i0,
                                                           bi, n, C, after);
  }
  });
}
// logits[h][i][j] from [ij][h], key mask, softmax over j, in place
__global__ void msaWeightsK(const float* flat, const float* keyMask, float* w, int n, int heads) {
  size_t rowId = blockIdx.x; int h = (int)(rowId / n), i = (int)(rowId % n);
  float* out = w + rowId * n;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    float v = flat[((size_t)i * n + j) * heads + h] + 1e9f * (keyMask[j] - 1.f);
    out[j] = v; mx = fmaxf(mx, v);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  float s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) { float e = expf(out[j] - mx); out[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x) out[j] *= inv;
}
__global__ void keyMaskK(const float* msaMask, float* keyMask, int S, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= n) return;
  float m = 0; for (int s = 0; s < S; ++s) m = fmaxf(m, msaMask[(size_t)s * n + j]);
  keyMask[j] = m;
}
// v [s][j][h*d+e] -> [h][j][s][e]
template <class T>
__global__ void castK(const float* in, T* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = fromF<T>(in[i]);
}
template <class T>
__global__ void msaVToHeadsK(const T* v, T* out, int S, int n, int heads, int d) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * n * heads * d) return;
  int e = (int)(t % d); size_t rest = t / d; int s = (int)(rest % S); rest /= S;
  int j = (int)(rest % n); int h = (int)(rest / n);
  out[t] = v[((size_t)s * n + j) * heads * d + h * d + e];
}
// o [h][i][s][e] -> [s][i][h*d+e], times sigmoid(gate)
template <class T>
__global__ void msaFromHeadsK(const T* o, const T* gate, T* out, int S, int n, int heads, int d) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * n * heads * d) return;
  int c = (int)(t % (heads * d)); size_t si = t / (heads * d); int i = (int)(si % n), s = (int)(si / n);
  int h = c / d, e = c % d;
  out[t] = fromF<T>(toF(o[(((size_t)h * n + i) * S + s) * d + e]) * (1.f / (1.f + expf(-toF(gate[t])))));
}
// T: the activations and the weighted sum's inputs (f16 on the fast path); the softmax in f32.
template <class T>
void msaAttention(Trunk& t, const std::string& pre) {
  int n = t.n, S = t.S, Cm = t.Cm, C = t.C;
  int heads = (int)M.meta(pre + ".heads"), d = (int)M.meta(pre + ".dimension"), Wd = heads * d;
  size_t rows = (size_t)S * n, pairs = (size_t)n * n;
  T* ln = scratch<T>("msaatt.ln", rows * Cm);
  layerNorm2<float, T>(t.msa, ln, rows, Cm, pre + ".actNormScale", pre + ".actNormOffset");
  // the pair's LayerNorm feeds only the bias projection: on a card short of room it is taken in row chunks
  size_t per = shortPair(pairs, C) || t.pairHost ? std::max<size_t>(1, std::min(pairs, CHUNK / C)) : pairs;
  T* pln = scratch<T>("msaatt.pln", per * C);
  float* flat = scratch<float>("msaatt.flat", pairs * heads);
  forRowWindows(t.P(), false, [&](float* w, size_t i0, size_t I) {
    size_t p0 = i0 * n, cnt = I * n;
    for (size_t r0 = 0; r0 < cnt; r0 += per) {
      size_t r = std::min(per, cnt - r0);
      layerNorm2<float, T>(w + r0 * C, pln, r, C, pre + ".pairNormScale", pre + ".pairNormOffset");
      linear<T, float>(pln, flat + (p0 + r0) * heads, r, C, heads, pre + ".pairLogits");
    }
  });
  float* keyMask = scratch<float>("msaatt.keymask", n);
  keyMaskK<<<blocks(n, 128), 128, 0, STREAM>>>(t.msaMask, keyMask, S, n);
  float* w = scratch<float>("msaatt.w", (size_t)heads * pairs);
  msaWeightsK<<<(unsigned)(heads * n), 128, 0, STREAM>>>(flat, keyMask, w, n, heads);
  const T* wT;
  if constexpr (std::is_same_v<T, float>) wT = w;
  else {
    T* wh = scratch<T>("msaatt.wh", (size_t)heads * pairs);
    castK<T><<<blocks((size_t)heads * pairs), 256, 0, STREAM>>>(w, wh, (size_t)heads * pairs);
    wT = wh;
  }
  // the values, the per-head weighted sums, the gate and the output - each MSA row's own, so on a card short of
  // room they run in blocks of rows (a 1024-row alignment at 10761 tokens held five 5.6 GB tensors whole)
  // (under LOCALFOLD_BIG a third of the rows, so a small input crosses blocks)
  int Sc = shortPair(pairs, C) ? (int)std::max<size_t>(1, std::min<size_t>(S, CHUNK / ((size_t)n * Wd))) : S;
  if (BIG_FORCED) Sc = std::max(1, std::min(Sc, (S + 2) / 3));
  T* v = scratch<T>("msaatt.v", (size_t)Sc * n * Wd);
  T* vh = scratch<T>("msaatt.vh", (size_t)Sc * n * Wd);
  T* oh = scratch<T>("msaatt.oh", (size_t)Sc * n * Wd);
  T* gate = scratch<T>("msaatt.gate", (size_t)Sc * n * Wd);
  T* gated = scratch<T>("msaatt.gated", (size_t)Sc * n * Wd);
  const float one = 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  for (int s0 = 0; s0 < S; s0 += Sc) {
    int sc = std::min(Sc, S - s0); size_t cr = (size_t)sc * n;
    const T* lnc = ln + (size_t)s0 * n * Cm;
    linear<T, T>(lnc, v, cr, Cm, Wd, pre + ".vProjection");
    msaVToHeadsK<T><<<blocks(cr * Wd), 256, 0, STREAM>>>(v, vh, sc, n, heads, d);
    // per head: O_h (n x sc*d) = W_h (n x n) V_h (n x sc*d); col-major O^T = V^T W^T
    CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, sc * d, n, n, &one, vh, cudaType<T>(), sc * d,
       (size_t)n * sc * d, wT, cudaType<T>(), n, pairs, &zero, oh, cudaType<T>(), sc * d, (size_t)n * sc * d, heads,
       CUBLAS_COMPUTE_32F, algo));
    linear<T, T>(lnc, gate, cr, Cm, Wd, pre + ".gatingQuery");
    msaFromHeadsK<T><<<blocks(cr * Wd), 256, 0, STREAM>>>(oh, gate, gated, sc, n, heads, d);
    linear<T, float>(gated, t.msa + (size_t)s0 * n * Cm, cr, Wd, Cm, pre + ".outputProjection", false, 1.f);
  }
  // (on a card short of room its pair-sized buffers go now, before the block's pair track allocates)
  if (shortPair(pairs, C)) releaseScratch({ "msaatt." });
}

template <class T>
void msaBlock(Trunk& t, int k) {
  std::string B = "trunk.msaBlocks." + std::to_string(k);
  // the outer product off the pre-update MSA (AF3), or off the updated one (OpenDDE, boltz2)
  bool updateFirst = M.flag("trunk.dialect.msaUpdateBeforeOuterProduct");
  if (!updateFirst) { outerProductMean<T>(t, B + ".outerProductMean"); stage("msa.opm"); }
  msaAttention<T>(t, B + ".msaAttention1"); stage("msa.attention");
  transition<T>(t.msa, (size_t)t.S * t.n, t.Cm, 4, B + ".msaTransition"); stage("msa.transition");
  if (updateFirst) { outerProductMean<T>(t, B + ".outerProductMean"); stage("msa.opm"); }
  pairUpdates<T>(t.P(), t.pairMask, B, t.swap, t.divide, 4);
}

// ---------------------------------------------------------------- pairformer single track
__global__ void logitsLayoutK(const float* flat, float* out, size_t pairs, int heads) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < pairs * heads) out[t] = flat[(t % pairs) * heads + t / pairs];
}
template <class T>
__global__ void singleSoftmaxK(const float* logits, const float* pairLogits, const float* seqMask, T* P,
                               int n, float scale) {
  size_t rowId = blockIdx.x;
  const float* L = logits + rowId * n; const float* B = pairLogits + rowId * n;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, L[j] * scale + B[j] + 1e9f * (seqMask[j] - 1.f));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  float sum = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) sum += expf(L[j] * scale + B[j] + 1e9f * (seqMask[j] - 1.f) - mx);
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    P[rowId * n + j] = fromF<T>(expf(L[j] * scale + B[j] + 1e9f * (seqMask[j] - 1.f) - mx) * inv);
}
template <class T>
__global__ void addQBiasK(T* qkvg, const float* b, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  size_t i = t / Wd; int c = (int)(t % Wd);
  qkvg[i * 4 * Wd + c] = fromF<T>(toF(qkvg[i * 4 * Wd + c]) + b[c]);
}
template <class T>
__global__ void gateK(T* o, const T* qkvg, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  size_t i = t / Wd; int c = (int)(t % Wd);
  o[t] = fromF<T>(toF(o[t]) * sigm(toF(qkvg[i * 4 * Wd + 3 * Wd + c])));
}
// OpenDDE's refiner and confidence blocks add one precomputed [i][j] bias to every head's logits
__global__ void addBiasHeadsK(float* pl, const float* bias, size_t pairs, int heads) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < pairs * heads) pl[t] += bias[t % pairs];
}
// The single track's attention in blocks of query rows - singleTrack's form on a card short of room, each
// block's pair logits from its own pair rows, its scores, softmax and values; the [heads, n, n] logits,
// probabilities and pair logits never whole (6.9 GB at 6000 tokens). The rows come a window at a time (rows),
// none crossing one - in tier 2 inside another pass over the pair's windows (pairUpdates) - then end().
template <class T>
struct SingleRows {
  float* single; const float* seqMask; int n, C, Cs; std::string B, A;
  int heads, d, Wd, R;
  float *pl, *logits, *flat = nullptr; T *P, *nrm, *qkvg, *o, *ln = nullptr;
  SingleRows(float* single_, const float* seqMask_, int n_, int C_, int Cs_, const std::string& B_)
      : single(single_), seqMask(seqMask_), n(n_), C(C_), Cs(Cs_), B(B_), A(B_ + ".singleAttention") {
    heads = (int)M.meta(A + ".heads"); d = (int)M.meta(A + ".dimension"); Wd = heads * d;
    R = (int)std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)heads * n)));
    pl = scratch<float>("st.pl", (size_t)heads * R * n);
    logits = scratch<float>("st.logits", (size_t)heads * R * n);
    P = scratch<T>("st.P", (size_t)heads * R * n);
    nrm = scratch<T>("st.nrm", (size_t)n * Cs);
    qkvg = scratch<T>("st.qkvg", (size_t)n * 4 * Wd);
    o = scratch<T>("st.o", (size_t)n * Wd);
    layerNorm2<float, T>(single, nrm, n, Cs, A + ".layerNormScale", A + ".layerNormOffset");
    linear<T, T>(nrm, qkvg, n, Cs, 4 * Wd, qkvgWeight(A, Cs, Wd, false));
    addQBiasK<T><<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(qkvg, W(A + ".qBias"), n, Wd);
  }
  // query rows [w0, w0 + Wn), their pair rows at base
  void rows(const float* base, size_t w0, size_t Wn) {
    const float one = 1.f, zero = 0.f;
    auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    for (int i0 = (int)w0; i0 < (int)(w0 + Wn); i0 += R) {
      int r = std::min(R, (int)(w0 + Wn) - i0); size_t rows = (size_t)r * n;
      const float* prow = base + (size_t)(i0 - w0) * n * C;
      bool fusedHeads = false;
      if constexpr (std::is_same_v<T, half>)
        if (C == 128 && heads == 16) {
          lnHeads128<16>(prow, B + ".singlePairLogitsNormScale", B + ".singlePairLogitsNormOffset",
                         B + ".singlePairLogitsProjection", pl, rows);
          fusedHeads = true;
        }
      if (!fusedHeads) {
        if (!flat) { flat = scratch<float>("st.flat", (size_t)heads * R * n); ln = scratch<T>("st.ln", (size_t)R * n * C); }
        layerNorm2<float, T>(prow, ln, rows, C, B + ".singlePairLogitsNormScale", B + ".singlePairLogitsNormOffset");
        linear<T, float>(ln, flat, rows, C, heads, B + ".singlePairLogitsProjection");
        logitsLayoutK<<<blocks(rows * heads), 256, 0, STREAM>>>(flat, pl, rows, heads);
      }
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, n, r, d, &one,
         qkvg + Wd, cudaType<T>(), 4 * Wd, d, qkvg + (size_t)i0 * 4 * Wd, cudaType<T>(), 4 * Wd, d, &zero, logits, CUDA_R_32F, n,
         rows, heads, CUBLAS_COMPUTE_32F, algo));
      singleSoftmaxK<T><<<(unsigned)(heads * r), 128, 0, STREAM>>>(logits, pl, seqMask, P, n, 1.f / sqrtf((float)d));
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, d, r, n, &one,
         qkvg + 2 * Wd, cudaType<T>(), 4 * Wd, d, P, cudaType<T>(), n, rows, &zero, o + (size_t)i0 * Wd, cudaType<T>(),
         Wd, d, heads, CUBLAS_COMPUTE_32F, algo));
    }
  }
  void end() {
    gateK<T><<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(o, qkvg, n, Wd);
    linear<T, float>(o, single, n, Wd, Cs, A + ".outputProjection", false, 1.f);
    transition<T>(single, n, Cs, 4, B + ".singleTransition");
  }
};
template <class T>
void singleTrack(float* single, const float* pair, const float* seqMask, int n, int C, int Cs,
                 const std::string& B, const float* extraBias = nullptr) {
  size_t pairs = (size_t)n * n;
  std::string A = B + ".singleAttention";
  int heads = (int)M.meta(A + ".heads"), d = (int)M.meta(A + ".dimension"), Wd = heads * d;
  // On a card short of room, in blocks of query rows: each block's pair logits from its own pair rows, its
  // scores, softmax and values - the [heads, n, n] logits, probabilities and pair logits never whole
  // (6.9 GB at 6000 tokens). The softmax is a row's alone and the projection is per pair position.
  if (shortPair(pairs, C) && !extraBias) {
    SingleRows<T> rows(single, seqMask, n, C, Cs, B);
    rows.rows(pair, 0, n);
    rows.end();
    return;
  }
  float* pl = scratch<float>("st.pl", pairs * heads);
  bool fused = false;
  if constexpr (std::is_same_v<T, half>)
    if (C == 128 && heads == 16) {     // one kernel: LN, the projection, the head-major layout
      lnHeads128<16>(pair, B + ".singlePairLogitsNormScale", B + ".singlePairLogitsNormOffset",
                     B + ".singlePairLogitsProjection", pl, pairs);
      fused = true;
    }
  if (!fused) {
    float* flat = scratch<float>("st.flat", pairs * heads);
    size_t rowsPer = std::max<size_t>(1, CHUNK / C);
    T* ln = scratch<T>("st.ln", std::min(rowsPer, pairs) * C);
    for (size_t r0 = 0; r0 < pairs; r0 += rowsPer) {
      size_t r = std::min(rowsPer, pairs - r0);
      layerNorm2<float, T>(pair + r0 * C, ln, r, C, B + ".singlePairLogitsNormScale", B + ".singlePairLogitsNormOffset");
      linear<T, float>(ln, flat + r0 * heads, r, C, heads, B + ".singlePairLogitsProjection");
    }
    logitsLayoutK<<<blocks(pairs * heads), 256, 0, STREAM>>>(flat, pl, pairs, heads);
  }
  if (extraBias) addBiasHeadsK<<<blocks(pairs * heads), 256, 0, STREAM>>>(pl, extraBias, pairs, heads);
  T* nrm = scratch<T>("st.nrm", (size_t)n * Cs);
  T* qkvg = scratch<T>("st.qkvg", (size_t)n * 4 * Wd);
  layerNorm2<float, T>(single, nrm, n, Cs, A + ".layerNormScale", A + ".layerNormOffset");
  linear<T, T>(nrm, qkvg, n, Cs, 4 * Wd, qkvgWeight(A, Cs, Wd, false));
  addQBiasK<T><<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(qkvg, W(A + ".qBias"), n, Wd);
  float* logits = scratch<float>("st.logits", (size_t)heads * n * n);
  T* P = scratch<T>("st.P", (size_t)heads * n * n);
  T* o = scratch<T>("st.o", (size_t)n * Wd);
  const float one = 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &one,
     qkvg + Wd, cudaType<T>(), 4 * Wd, d, qkvg, cudaType<T>(), 4 * Wd, d, &zero, logits, CUDA_R_32F, n,
     (size_t)n * n, heads, CUBLAS_COMPUTE_32F, algo));
  singleSoftmaxK<T><<<(unsigned)(heads * n), 128, 0, STREAM>>>(logits, pl, seqMask, P, n, 1.f / sqrtf((float)d));
  CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, d, n, n, &one,
     qkvg + 2 * Wd, cudaType<T>(), 4 * Wd, d, P, cudaType<T>(), n, (size_t)n * n, &zero, o, cudaType<T>(),
     Wd, d, heads, CUBLAS_COMPUTE_32F, algo));
  gateK<T><<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(o, qkvg, n, Wd);
  linear<T, float>(o, single, n, Wd, Cs, A + ".outputProjection", false, 1.f);
  transition<T>(single, n, Cs, 4, B + ".singleTransition");
}

// (tier 2: the single track's rows ride on the pair updates' last pass, and the next block's first fixed
// operand with them - `next` is that block's name, empty for the last)
template <class T>
void pairformerBlockAt(const Pair& P, float* single, const float* pairMask, const float* seqMask, int Cs,
                       const std::string& B, bool swap, bool divide, const float* extraBias = nullptr,
                       const std::string& next = "") {
  if (!P.host) {
    pairUpdates<T>(P, pairMask, B, swap, divide, 4);
    singleTrack<T>(single, P.dev, seqMask, P.n, P.C, Cs, B, extraBias); stage("single");
    return;
  }
  if (extraBias) { fprintf(stderr, "a pair in host memory takes no extra single-attention bias\n"); exit(1); }
  std::unique_ptr<SingleRows<T>> rows;      // (made at the last pass, so its buffers are not held through the rest)
  pairUpdates<T>(P, pairMask, B, swap, divide, 4, [&](const float* w, size_t i0, size_t I) {
    if (!rows) rows = std::make_unique<SingleRows<T>>(single, seqMask, P.n, P.C, Cs, B);
    rows->rows(w, i0, I);
  }, next.empty() ? "" : next + ".triangleMultiplicationOutgoing");
  rows->end(); stage("single");
}
template <class T>
void pairformerBlockAt(float* pair, float* single, const float* pairMask, const float* seqMask, int n, int C,
                       int Cs, const std::string& B, bool swap, bool divide, const float* extraBias = nullptr) {
  pairformerBlockAt<T>(Pair{pair, nullptr, n, C}, single, pairMask, seqMask, Cs, B, swap, divide, extraBias);
}
template <class T>
void pairformerBlock(Trunk& t, int k) {
  std::string next = "trunk.pairformerBlocks." + std::to_string(k + 1);
  pairformerBlockAt<T>(t.P(), t.single, t.pairMask, t.seqMask, t.Cs, "trunk.pairformerBlocks." + std::to_string(k),
                       t.swap, t.divide, nullptr, M.has(next + ".singleChannels") ? next : "");
}

// ---------------------------------------------------------------- distogram
// logits[i][j] = half[i][j] + half[j][i] (symmetriseK, elementwise.cuh)
inline void distogram(Trunk& t, float* logits) {
  int bins = (int)M.meta("trunk.distogram.bins");
  size_t pairs = (size_t)t.n * t.n;
  float* half_ = scratch<float>("disto.half", pairs * bins);
  linear<float, float>(t.pair, half_, pairs, t.C, bins, "trunk.distogram.halfLogits");
  // a trained bias (OpenDDE, boltz2, ...) is in each half, so twice in the symmetrised logit -
  // what those checkpoints were trained with (src/af3/trunk/trunk-webgpu.js)
  if (hasW("trunk.distogram.halfLogitsBias"))
    addBiasK<<<blocks(pairs * bins), 256, 0, STREAM>>>(half_, W("trunk.distogram.halfLogitsBias"), pairs, bins);
  symmetriseK<<<blocks(pairs * bins), 256, 0, STREAM>>>(half_, logits, t.n, bins);
}
// P(distance under the pair's contact threshold): the softmax mass of the first contactBins[ij]
// bins (src/af3/featurise/contact-classes.js), masked
__global__ void contactProbsK(const float* logits, const int* contactBins, const float* pairMask, float* out,
                              size_t pairs, int bins) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= pairs) return;
  const float* l = logits + ij * bins;
  float mx = -INFINITY;
  for (int b = 0; b < bins; ++b) mx = fmaxf(mx, l[b]);
  float total = 0.f, contact = 0.f;
  for (int b = 0; b < bins; ++b) { float p = expf(l[b] - mx); total += p; if (b < contactBins[ij]) contact += p; }
  out[ij] = pairMask[ij] * contact / total;
}
// ...on the device (disto.contact)
inline float* contactProbsDevice(Trunk& t) {
  int bins = (int)M.meta("trunk.distogram.bins");
  size_t pairs = (size_t)t.n * t.n;
  float* out = scratch<float>("disto.contact", pairs);
  if (shortPair(pairs, t.C) || t.pairHost) {
    // on a card short of room in blocks of rows: a row's symmetrised logit is its own half-logit plus the
    // transposed pair's, so each block projects its rows and the column entries gathered from the pair -
    // never the [pairs, bins] logits whole (9.2 GB at 6000 tokens, twice)
    int n = t.n, C = t.C;
    size_t R = std::max<size_t>(1, std::min<size_t>(n, CHUNK / ((size_t)n * std::max(C, bins))));
    float* rowsT = scratch<float>("disto.rowsT", R * n * C);
    float* a = scratch<float>("disto.half", R * n * bins); float* b = scratch<float>("disto.halfT", R * n * bins);
    bool bias = hasW("trunk.distogram.halfLogitsBias");
    // (where the pair is in host memory, each block's rows from a window of rows and its columns from a
    // window of the same columns, staged beside it)
    Pair P = t.P();
    auto rowsOf = [&](const float* base, size_t w0, size_t Wn) {
    for (size_t r0 = w0; r0 < w0 + Wn; r0 += R) {
      size_t r = std::min(R, w0 + Wn - r0), rows = r * n;
      linear<float, float>(base + (r0 - w0) * n * C, a, rows, C, bins, "trunk.distogram.halfLogits");
      if (P.host) {
        float* cols = scratch<float>("hp.win2", rows * C);
        CK(cudaMemcpy2DAsync(cols, r * C * 4, P.host + r0 * C, (size_t)n * C * 4, r * C * 4, n, cudaMemcpyHostToDevice, STREAM));
        transposeWindowK<<<blocks(rows * C * 4 / 16), 256, 0, STREAM>>>((const uint4*)cols, (uint4*)rowsT, n, r, C * 4 / 16);
      } else {
        gatherTransposedK<<<blocks(rows * C * 4 / 16), 256, 0, STREAM>>>(t.pair, rowsT, n, C, r0, r, 4);
      }
      linear<float, float>(rowsT, b, rows, C, bins, "trunk.distogram.halfLogits");
      if (bias) {
        addBiasK<<<blocks(rows * bins), 256, 0, STREAM>>>(a, W("trunk.distogram.halfLogitsBias"), rows, bins);
        addBiasK<<<blocks(rows * bins), 256, 0, STREAM>>>(b, W("trunk.distogram.halfLogitsBias"), rows, bins);
      }
      addK<<<blocks(rows * bins), 256, 0, STREAM>>>(a, b, rows * bins);
      contactProbsK<<<blocks(rows), 256, 0, STREAM>>>(a, Idev("batch.contactBins") + r0 * n, t.pairMask + r0 * n,
                                                      out + r0 * n, rows, bins);
    }
    };
    if (P.host) { forRowWindows(P, false, rowsOf); releaseScratch({ "hp.win2" }); }
    else rowsOf(t.pair, 0, n);
    return out;
  }
  float* logits = scratch<float>("disto.logits", pairs * bins);
  distogram(t, logits);
  contactProbsK<<<blocks(pairs), 256, 0, STREAM>>>(logits, Idev("batch.contactBins"), t.pairMask, out, pairs, bins);
  return out;
}
inline std::vector<float> contactProbabilities(Trunk& t) {
  if (!M.has("batch.contactBins")) return {};
  return download(contactProbsDevice(t), (size_t)t.n * t.n);
}

// The whole trunk pass. `onSeam(name, ptr, n)` sees the oracle's seams.
template <class T>
void runTrunk(Trunk& t, const std::function<void(const char*, const float*, size_t)>& onSeam) {
  // (tier 2 on a card short of room: what the target feature's atom encoder left goes first)
  if (t.pairHost && shortPair((size_t)t.n * t.n, t.C)) releaseScratch({ "targetFeat.", "enc.", "apl." });
  embed<T>(t, onSeam); stage("embed"); memReport("  trunk: embedded");
  size_t pairs = (size_t)t.n * t.n;
  // on a card short of room, the embedder's and template stack's scratch given back before the MSA
  // stack, and the MSA stack's before the pairformer (see shortPair)
  bool tight = shortPair(pairs, t.C);
  // ...and the template stack's own triangle buffers: it runs 64 channels through the unfused path,
  // whose names the 128-channel pairformer never asks for again (4.9 GB at 2620 tokens)
  if (tight) releaseScratch({ "emb.", "tmpl.", "trib.", "grid.", "tri.a", "tri.b", "tri.prod", "tri.norm", "tri.pg", "tri.centred",
                              "tri.t1", "tri.t2" });
  int msaBlocks = 0; while (M.has("trunk.msaBlocks." + std::to_string(msaBlocks) + ".pairChannels")) ++msaBlocks;
  // boltz2 adds the pre-MSA pair back: its MSA module returns the updated z and the caller adds z
  float* zIn = nullptr;
  if (M.flag("trunk.dialect.msaDoubleAddPair") && t.pairHost) {
    fprintf(stderr, "a pair this card cannot hold does not fold with boltz2's MSA module (it keeps a second pair)\n"); exit(1);
  }
  if (M.flag("trunk.dialect.msaDoubleAddPair")) {
    zIn = scratch<float>("trunk.zBeforeMsa", pairs * t.C);
    CK(cudaMemcpyAsync(zIn, t.pair, pairs * t.C * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  for (int k = 0; k < msaBlocks; ++k) msaBlock<T>(t, k);
  if (zIn) addK<<<blocks(pairs * t.C), 256, 0, STREAM>>>(t.pair, zIn, pairs * t.C);
  memReport("  trunk: MSA stack");
  if (tight) releaseScratch({ "msaatt.", "opm.", "trunk.zBeforeMsa", "trib." });
  onSeam("z_after_msa", t.pair, pairs * t.C);
  onSeam("trunk_in_single", t.single, (size_t)t.n * t.Cs);
  int blocks_ = 0; while (M.has("trunk.pairformerBlocks." + std::to_string(blocks_) + ".singleChannels")) ++blocks_;
  for (int k = 0; k < blocks_; ++k) pairformerBlock<T>(t, k);
  memReport("  trunk: pairformer");
  // ...and the pair track's own, before the next pass's template stack allocates beside it: held, they
  // put a recycle pass's peak at its embedding (17.71 against 12.88 GB at 1572 tokens)
  if (tight) releaseScratch({ "tri.", "trib.", "grid.", "tr.", "st." });
  onSeam("trunk_out_pair", t.pair, pairs * t.C);
  onSeam("single", t.single, (size_t)t.n * t.Cs);
  // (tier 2: the template stage's host pairs go once the fold's last pass is done with them - af3.cu)
}
inline void freeTemplateHost() { for (const char* k : { "template.act", "template.summed", "template.before" }) hostKeptFree(k); }
