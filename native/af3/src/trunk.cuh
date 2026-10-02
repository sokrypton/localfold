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
// AF3's relative encoding, 139 one-hot columns, folded straight into its projection: each
// pair adds the four or five weight rows its one-hot selects.
__global__ void relativeEncodingK(const int* residueIndex, const int* tokenIndex, const int* asymId,
                                  const int* entityId, const int* symId, const float* Wpos, float* pair,
                                  int n, int C, int maxIdx, int maxChain) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
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
  int* msaRows; float* deletion;
  bool swap, divide;
};

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
  t.pair = dalloc(pairs * t.C); t.single = dalloc((size_t)t.n * t.Cs);
  t.msa = dalloc((size_t)t.S * t.n * t.Cm);
  t.prevPair = dalloc(pairs * t.C); t.prevSingle = dalloc((size_t)t.n * t.Cs);
  CK(cudaMemset(t.prevPair, 0, pairs * t.C * 4)); CK(cudaMemset(t.prevSingle, 0, (size_t)t.n * t.Cs * 4));
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

template <class T> void templateEmbedding(Trunk& t, float* pairOut);
__global__ void bondEmbedK(float* pair, const float* bonds, const float* w, size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < pairs * C) pair[t] += bonds[t / C] * w[t % C];
}

// boltz2's two extra z-init terms: token_bonds_type_embed[bond order] (row 0 on an unbonded pair,
// trained non-zero) plus the contact conditioning's unspecified-restraint constant
__global__ void bondTypeEmbedK(float* pair, const float* orders, const float* table, const float* unspecified,
                               size_t pairs, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  int o = orders ? (int)orders[t / C] : 0;
  if (o < 0 || o >= 7) o = 0;
  pair[t] += table[o * C + t % C] + unspecified[t % C];
}
inline void bondTypeEmbed(float* pair, size_t pairs, int C, const std::string& pre) {
  if (!hasW(pre + "tokenBondsTypeEmbed")) return;
  bondTypeEmbedK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, M.has("batch.bondOrderMatrix") ? Fdev("batch.bondOrderMatrix") : nullptr,
    W(pre + "tokenBondsTypeEmbed"), W(pre + "contactEncodingUnspecified"), pairs, C);
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
  outerSumK<<<blocks(pairs * C), 256, 0, STREAM>>>(left, right, t.pair, n, C);
  onSeam("z_before_prev", t.pair, pairs * C);
  // the recycled pair: LayerNorm then projection, which is NOT zero on the first pass
  T* ln = scratch<T>("emb.prevln", pairs * C);
  layerNorm2<float, T>(t.prevPair, ln, pairs, C, E + "prevEmbeddingNormScale", E + "prevEmbeddingNormOffset");
  linear<T, float>(ln, t.pair, pairs, C, C, E + "prevEmbedding", false, 1.f);
  onSeam("z_after_prev", t.pair, pairs * C);
  relativeEncodingK<<<blocks(pairs * C), 256, 0, STREAM>>>(
    Idev("batch.features.residueIndex"), Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"),
    Idev("batch.features.entityId"), Idev("batch.features.symId"), W(E + "positionActivations"),
    t.pair, n, C, 32, 2);
  if (M.has("batch.bondMatrix")) {
    // bond_embedding: bias-free, one input column (the token-pair contact); zero for a polymer
    // without links
    if (lenW(E + "bondEmbedding") != (size_t)C) { fprintf(stderr, "bondEmbedding is not 1 x %d\n", C); exit(1); }
    bondEmbedK<<<blocks(pairs * C), 256, 0, STREAM>>>(t.pair, Fdev("batch.bondMatrix"), W(E + "bondEmbedding"), pairs, C);
  }
  bondTypeEmbed(t.pair, pairs, C, E);
  onSeam("z_init_generic", t.pair, pairs * C);
  float* tmpl = scratch<float>("emb.template", pairs * C);
  templateEmbedding<T>(t, tmpl);
  addK<<<blocks(pairs * C), 256, 0, STREAM>>>(t.pair, tmpl, pairs * C);
  onSeam("z_after_template", t.pair, pairs * C);
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
__global__ void addRowColumnK(float* act, const float* row, const float* col, int n, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  act[t] += row[(size_t)j * C + c] + col[(size_t)i * C + c];
}
__global__ void reluScaleK(float* x, float s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = fmaxf(0.f, x[i] * s);
}
// act += the geometry features of a real template slot: the distogram (one-hot bins) projected,
// and five scalar features each times a per-channel weight (AF3's num_input_dims=0).
__global__ void templateGeometryK(float* act, const float* dgram, const float* pb, const float* uv, const float* bb,
                                  const float* W0, const float* W1, const float* W4, const float* W5,
                                  const float* W6, const float* W7, size_t pairs, int C, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  int c = (int)(t % C); size_t p = t / C;
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
  T* ln = scratch<T>("tmpl.ln", pairs * Cq);
  layerNorm2<float, T>(t.pair, ln, pairs, Cq, P + "queryEmbeddingNormScale", P + "queryEmbeddingNormOffset");
  float* query = scratch<float>("tmpl.query", pairs * Ct);
  linear<T, float>(ln, query, pairs, Cq, Ct, P + (fused ? "zProjection" : "templatePairEmbedding8"));
  float* act = scratch<float>("tmpl.act", pairs * Ct);
  float* before = outer ? scratch<float>("tmpl.before", pairs * Ct) : nullptr;
  float* normed = scratch<float>("tmpl.normed", pairs * Ct);
  float* summed = scratch<float>("tmpl.summed", pairs * Ct);
  CK(cudaMemsetAsync(summed, 0, pairs * Ct * 4, STREAM));
  float* oh = scratch<float>("tmpl.onehot", (size_t)n * 31);
  float* row = scratch<float>("tmpl.row", (size_t)n * Ct); float* col = scratch<float>("tmpl.col", (size_t)n * Ct);
  int nb = 0; while (M.has(P + "blocks." + std::to_string(nb) + ".pairTransition.transition1")) ++nb;
  for (int k = 0; k < passes; ++k) {
    std::string S = "template." + std::to_string(k) + ".";
    float repeat = (float)M.meta(S + "repeat");
    if (repeat == 0.f) continue;                // a slot weighed zero (boltz2's empty ones)
    CK(cudaMemcpyAsync(act, query, pairs * Ct * 4, cudaMemcpyDeviceToDevice, STREAM));
    if (fused) {
      linear<float, float>(Fdev(S + "features"), act, pairs, width, Ct, P + "aProjection", false, 1.f);
    } else {
      std::vector<float> onehot((size_t)n * 31, 0.f);
      const int* aatype = M.i(S + "aatype");
      for (int i = 0; i < n; ++i) if (aatype[i] >= 0 && aatype[i] < 31) onehot[(size_t)i * 31 + aatype[i]] = 1.f;
      CK(cudaMemcpy(oh, onehot.data(), onehot.size() * 4, cudaMemcpyHostToDevice));
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
    for (int b = 0; b < nb; ++b) {
      std::string B = P + "blocks." + std::to_string(b);
      int factor = (int)(lenW(B + ".pairTransition.transition1") / ((size_t)Ct * Ct * 2));
      pairUpdates<T>(act, t.pairMask, n, Ct, B, t.swap, t.divide, factor);
    }
    if (outer) addK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(act, before, pairs * Ct);
    layerNorm2<float, float>(act, normed, pairs, Ct, P + "outputLayerNormScale", P + "outputLayerNormOffset");
    addScaledK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(summed, normed, repeat, pairs * Ct);
  }
  // divided by every slot (not the real ones), relu, projected
  reluScaleK<<<blocks(pairs * Ct), 256, 0, STREAM>>>(summed, 1.f / (1e-7f + templates), pairs * Ct);
  linear<float, float>(summed, out, pairs, Ct, Cq, P + "outputLinear");
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
// [(bi, c), (j, e)] -> [(bi, j), (c, e)]
template <class T>
__global__ void opmPermuteK(const float* in, T* out, int Bi, int n, int O) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)Bi * n * O * O;
  if (t >= total) return;
  int e = (int)(t % O); size_t rest = t / O; int c = (int)(rest % O); rest /= O;
  int j = (int)(rest % n); int bi = (int)(rest / n);
  out[t] = fromF<T>(in[((size_t)bi * O + c) * ((size_t)n * O) + (size_t)j * O + e]);
}
// pair[i][j] += (bias + x) / (1e-3 + norm[i][j])   (AF3: the bias inside the scale)
__global__ void opmAddK(float* pair, const float* x, const float* bias, const float* norm, size_t i0,
                        int Bi, int n, int C, bool biasAfterNorm) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)Bi * n * C) return;
  int f = (int)(t % C); size_t ij = t / C; size_t i = i0 + ij / n, j = ij % n;
  float nv = norm[i * n + j];
  float v = biasAfterNorm ? x[t] / fmaxf(nv, 1.f) + bias[f] : (bias[f] + x[t]) / (1e-3f + nv);
  pair[(i * n + j) * C + f] += v;
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
  float* P = scratch<float>("opm.P", (size_t)Bi * per);
  T* Pp = scratch<T>("opm.Pp", (size_t)Bi * per);
  float* X = scratch<float>("opm.X", (size_t)Bi * n * C);
  bool after = M.flag("trunk.dialect.opmBiasAfterNorm");
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  for (int i0 = 0; i0 < n; i0 += Bi) {
    int bi = std::min(Bi, n - i0);
    // row-major P (bi*O x n*O) = L_blk^T R where L_blk is [S][bi*O] with row stride n*O.
    // col-major: P^T (n*O x bi*O) = R^T(op N on R as (n*O x S), ld n*O) * L_blk (op T)
    CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_T, n * O, bi * O, S, &one, R, cudaType<T>(), n * O,
                    L + (size_t)i0 * O, cudaType<T>(), n * O, &zero, P, CUDA_R_32F, n * O, CUBLAS_COMPUTE_32F, algo));
    opmPermuteK<T><<<blocks((size_t)bi * per), 256, 0, STREAM>>>(P, Pp, bi, n, O);
    linear<T, float>(Pp, X, (size_t)bi * n, O * O, C, pre + ".outputW");
    opmAddK<<<blocks((size_t)bi * n * C), 256, 0, STREAM>>>(t.pair, X, W(pre + ".outputB"), norm, i0, bi, n,
                                                           C, after);
  }
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
  T* pln = scratch<T>("msaatt.pln", pairs * C);
  layerNorm2<float, T>(t.pair, pln, pairs, C, pre + ".pairNormScale", pre + ".pairNormOffset");
  float* flat = scratch<float>("msaatt.flat", pairs * heads);
  linear<T, float>(pln, flat, pairs, C, heads, pre + ".pairLogits");
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
  T* v = scratch<T>("msaatt.v", rows * Wd);
  linear<T, T>(ln, v, rows, Cm, Wd, pre + ".vProjection");
  T* vh = scratch<T>("msaatt.vh", rows * Wd);
  msaVToHeadsK<T><<<blocks(rows * Wd), 256, 0, STREAM>>>(v, vh, S, n, heads, d);
  // per head: O_h (n x S*d) = W_h (n x n) V_h (n x S*d); col-major O^T = V^T W^T
  T* oh = scratch<T>("msaatt.oh", rows * Wd);
  const float one = 1.f, zero = 0.f;
  auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, S * d, n, n, &one, vh, cudaType<T>(), S * d,
     (size_t)n * S * d, wT, cudaType<T>(), n, pairs, &zero, oh, cudaType<T>(), S * d, (size_t)n * S * d, heads,
     CUBLAS_COMPUTE_32F, algo));
  T* gate = scratch<T>("msaatt.gate", rows * Wd);
  linear<T, T>(ln, gate, rows, Cm, Wd, pre + ".gatingQuery");
  T* gated = scratch<T>("msaatt.gated", rows * Wd);
  msaFromHeadsK<T><<<blocks(rows * Wd), 256, 0, STREAM>>>(oh, gate, gated, S, n, heads, d);
  linear<T, float>(gated, t.msa, rows, Wd, Cm, pre + ".outputProjection", false, 1.f);
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
  pairUpdates<T>(t.pair, t.pairMask, t.n, t.C, B, t.swap, t.divide, 4);
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
template <class T>
void singleTrack(float* single, const float* pair, const float* seqMask, int n, int C, int Cs,
                 const std::string& B, const float* extraBias = nullptr) {
  size_t pairs = (size_t)n * n;
  std::string A = B + ".singleAttention";
  int heads = (int)M.meta(A + ".heads"), d = (int)M.meta(A + ".dimension"), Wd = heads * d;
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

template <class T>
void pairformerBlockAt(float* pair, float* single, const float* pairMask, const float* seqMask, int n, int C,
                       int Cs, const std::string& B, bool swap, bool divide, const float* extraBias = nullptr) {
  pairUpdates<T>(pair, pairMask, n, C, B, swap, divide, 4);
  singleTrack<T>(single, pair, seqMask, n, C, Cs, B, extraBias); stage("single");
}
template <class T>
void pairformerBlock(Trunk& t, int k) {
  pairformerBlockAt<T>(t.pair, t.single, t.pairMask, t.seqMask, t.n, t.C, t.Cs,
                       "trunk.pairformerBlocks." + std::to_string(k), t.swap, t.divide);
}

// ---------------------------------------------------------------- distogram
// logits[i][j] = half[i][j] + half[j][i]
__global__ void symmetriseK(const float* half_, float* logits, int n, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * bins) return;
  int b = (int)(t % bins); size_t ij = t / bins; size_t i = ij / n, j = ij % n;
  logits[t] = half_[ij * bins + b] + half_[(j * n + i) * bins + b];
}
inline void distogram(Trunk& t, float* logits) {
  int bins = (int)M.meta("trunk.distogram.bins");
  size_t pairs = (size_t)t.n * t.n;
  float* half_ = scratch<float>("disto.half", pairs * bins);
  linear<float, float>(t.pair, half_, pairs, t.C, bins, "trunk.distogram.halfLogits");
  symmetriseK<<<blocks(pairs * bins), 256, 0, STREAM>>>(half_, logits, t.n, bins);
}

// The whole trunk pass. `onSeam(name, ptr, n)` sees the oracle's seams.
template <class T>
void runTrunk(Trunk& t, const std::function<void(const char*, const float*, size_t)>& onSeam) {
  embed<T>(t, onSeam); stage("embed");
  size_t pairs = (size_t)t.n * t.n;
  int msaBlocks = 0; while (M.has("trunk.msaBlocks." + std::to_string(msaBlocks) + ".pairChannels")) ++msaBlocks;
  // boltz2 adds the pre-MSA pair back: its MSA module returns the updated z and the caller adds z
  float* zIn = nullptr;
  if (M.flag("trunk.dialect.msaDoubleAddPair")) {
    zIn = scratch<float>("trunk.zBeforeMsa", pairs * t.C);
    CK(cudaMemcpyAsync(zIn, t.pair, pairs * t.C * 4, cudaMemcpyDeviceToDevice, STREAM));
  }
  for (int k = 0; k < msaBlocks; ++k) msaBlock<T>(t, k);
  if (zIn) addK<<<blocks(pairs * t.C), 256, 0, STREAM>>>(t.pair, zIn, pairs * t.C);
  onSeam("z_after_msa", t.pair, pairs * t.C);
  onSeam("trunk_in_single", t.single, (size_t)t.n * t.Cs);
  int blocks_ = 0; while (M.has("trunk.pairformerBlocks." + std::to_string(blocks_) + ".singleChannels")) ++blocks_;
  for (int k = 0; k < blocks_; ++k) pairformerBlock<T>(t, k);
  onSeam("trunk_out_pair", t.pair, pairs * t.C);
  onSeam("single", t.single, (size_t)t.n * t.Cs);
}
