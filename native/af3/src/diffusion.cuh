// The diffusion head - conditioning, the atom encoder, the 24-block conditioned transformer,
// the atom decoder - and the sampler. Transcribed from src/af3/diffusion/diffusion-reference.js
// and diffusion-sampler-reference.js.
#pragma once
#include "atom.cuh"

constexpr float SIGMA_DATA = 16.f;

// features2d = [trunk pair (Cz) | relative one-hot (139)] per pair; one-hot built in place
__global__ void pairFeaturesK(const float* trunkPair, const int* residueIndex, const int* tokenIndex,
                              const int* asymId, const int* entityId, const int* symId, float* out, int n,
                              int Cz, int maxIdx, int maxChain) {
  int rel = (2 * maxIdx + 2) * 2 + 1 + (2 * maxChain + 2), width = Cz + rel;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * width) return;
  int c = (int)(t % width); size_t ij = t / width; int i = (int)(ij / n), j = (int)(ij % n);
  if (c < Cz) { out[t] = trunkPair[ij * Cz + c]; return; }
  int k = c - Cz, positionBins = 2 * maxIdx + 2;
  auto clamp = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
  bool sameChain = asymId[i] == asymId[j], sameEntity = entityId[i] == entityId[j];
  int c0 = sameChain ? clamp(residueIndex[i] - residueIndex[j] + maxIdx, 2 * maxIdx) : 2 * maxIdx + 1;
  bool sameResidue = sameChain && residueIndex[i] == residueIndex[j];
  int c1 = positionBins + (sameResidue ? clamp(tokenIndex[i] - tokenIndex[j] + maxIdx, 2 * maxIdx) : 2 * maxIdx + 1);
  int c2 = positionBins * 2;
  int c3 = positionBins * 2 + 1 + (sameEntity ? clamp(symId[i] - symId[j] + maxChain, 2 * maxChain) : 2 * maxChain + 1);
  out[t] = (k == c0 || k == c1 || k == c3 || (k == c2 && sameEntity)) ? 1.f : 0.f;
}
__global__ void concatK(const float* a, int wa, const float* b, int wb, float* out, int rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int w = wa + wb;
  if (t >= (size_t)rows * w) return;
  int r = (int)(t / w), c = (int)(t % w);
  out[t] = c < wa ? a[(size_t)r * wa + c] : b[(size_t)r * wb + (c - wa)];
}
__global__ void addVectorK(float* x, const float* v, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t < rows * C) x[t] += v[t % C];
}

// The plain gated transition used by the conditioning: LN (scale, offset), SwiGLU, project.
inline void plainTransition(float* x, size_t rows, int C, int factor, const std::string& P) {
  int I = C * factor;
  float* xn = scratch<float>("pt.xn", rows * C);
  float* wide = scratch<float>("pt.wide", rows * 2 * I);
  float* gated = scratch<float>("pt.gated", rows * I);
  layerNormSlow(x, xn, rows, C, W(P + ".ffwLayerNormScale"), Wopt(P + ".ffwLayerNormOffset"));
  linear<float, float>(xn, wide, rows, C, 2 * I, P + ".ffwTransition1");
  swigluK<float><<<blocks(rows * I), 256, 0, STREAM>>>(wide, gated, rows, I);
  linear<float, float>(gated, x, rows, I, C, P + ".ffwTransition2", false, 1.f);
}

struct Conditioning { float *single, *pair; };
// Everything in the conditioning but the noise term is the same at every step; the pair is
// computed once and the single's noise-free part kept, so a step adds one projection.
struct DiffusionCache { bool ready = false; float *pair, *singleBase; };
inline DiffusionCache DCACHE;

inline Conditioning diffusionConditioning(const float* trunkSingle, const float* trunkPair,
                                          const float* targetFeat, float noiseLevel, int n) {
  const std::string P = "diffusion.conditioning";
  int Cz = (int)M.meta(P + ".pairChannels"), Cs = (int)M.meta(P + ".seqChannels");
  int Czt = (int)M.meta(P + ".trunkPairChannels"), Cst = (int)M.meta(P + ".trunkSingleChannels");
  int F = (int)M.meta(P + ".targetFeatWidth"), rel = (int)M.meta(P + ".relativeWidth");
  size_t pairs = (size_t)n * n;
  if (hasW(P + ".zTrunkProjection") || hasW(P + ".relpeProjection")) {
    fprintf(stderr, "split/projected relpos conditioning: not ported\n"); exit(1);
  }
  if (M.flag("trunk.dialect.padSingleCondUnknownDna")) { fprintf(stderr, "padded single cond: not ported\n"); exit(1); }
  if (!DCACHE.ready) {
    int width = Czt + rel;
    float* f2 = scratch<float>("dc.f2", pairs * width);
    pairFeaturesK<<<blocks(pairs * width), 256, 0, STREAM>>>(trunkPair, Idev("batch.features.residueIndex"),
      Idev("batch.features.tokenIndex"), Idev("batch.features.asymId"), Idev("batch.features.entityId"),
      Idev("batch.features.symId"), f2, n, Czt, 32, 2);
    float* f2n = scratch<float>("dc.f2n", pairs * width);
    layerNormSlow(f2, f2n, pairs, width, W(P + ".pairCondInitialNormScale"), Wopt(P + ".pairCondInitialNormOffset"));
    DCACHE.pair = dalloc(pairs * Cz);
    linear<float, float>(f2n, DCACHE.pair, pairs, width, Cz, P + ".pairCondInitialProjection");
    for (int k = 0; k < 2; ++k) plainTransition(DCACHE.pair, pairs, Cz, 2, P + ".pairTransitions." + std::to_string(k));
    int sw = Cst + F;
    float* f1 = scratch<float>("dc.f1", (size_t)n * sw);
    concatK<<<blocks((size_t)n * sw), 256, 0, STREAM>>>(trunkSingle, Cst, targetFeat, F, f1, n);
    float* f1n = scratch<float>("dc.f1n", (size_t)n * sw);
    layerNormSlow(f1, f1n, n, sw, W(P + ".singleCondInitialNormScale"), Wopt(P + ".singleCondInitialNormOffset"));
    DCACHE.singleBase = dalloc((size_t)n * Cs);
    linear<float, float>(f1n, DCACHE.singleBase, n, sw, Cs, P + ".singleCondInitialProjection");
    if (hasW(P + ".singleCondInitialProjectionBias"))
      addVectorK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(DCACHE.singleBase, W(P + ".singleCondInitialProjectionBias"), n, Cs);
    DCACHE.ready = true;
  }
  // the Fourier noise embedding, on the host: 256 cosines
  size_t nc = lenW(P + ".fourierWeight");
  std::vector<float> emb(nc);
  const float* fw = M.f(P + ".fourierWeight"); const float* fb = M.f(P + ".fourierBias");
  double tr = 0.25 * std::log((double)noiseLevel / SIGMA_DATA);
  for (size_t k = 0; k < nc; ++k) emb[k] = (float)std::cos(2 * M_PI * (tr * fw[k] + fb[k]));
  float* e = scratch<float>("dc.emb", nc);
  CK(cudaMemcpyAsync(e, emb.data(), nc * 4, cudaMemcpyHostToDevice, STREAM));
  float* en = scratch<float>("dc.embn", nc);
  layerNormSlow(e, en, 1, (int)nc, W(P + ".noiseEmbeddingInitialNormScale"), Wopt(P + ".noiseEmbeddingInitialNormOffset"));
  float* proj = scratch<float>("dc.noiseproj", Cs);
  linear<float, float>(en, proj, 1, (int)nc, Cs, P + ".noiseEmbeddingInitialProjection");
  float* single = scratch<float>("dc.single", (size_t)n * Cs);
  CK(cudaMemcpyAsync(single, DCACHE.singleBase, (size_t)n * Cs * 4, cudaMemcpyDeviceToDevice, STREAM));
  addVectorK<<<blocks((size_t)n * Cs), 256, 0, STREAM>>>(single, proj, n, Cs);
  for (int k = 0; k < 2; ++k) plainTransition(single, n, Cs, 2, P + ".singleTransitions." + std::to_string(k));
  return { single, DCACHE.pair };
}

// ---------------------------------------------------------------- the transformer
// softmax over keys of logits[h][i][j] * scale + mask + pairLogits[h][i][j], in place
__global__ void tokenSoftmaxK(float* logits, const float* pairLogits, const float* mask, int n, float scale) {
  size_t rowId = blockIdx.x;
  float* L = logits + rowId * n; const float* B = pairLogits + rowId * n;
  __shared__ float red[32];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    float v = L[j] * scale + 1e9f * (mask[j] - 1.f) + B[j]; L[j] = v; mx = fmaxf(mx, v);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads(); mx = red[0]; __syncthreads();
  float s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) { float e = expf(L[j] - mx); L[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) { float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); if (threadIdx.x == 0) red[0] = v; }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < n; j += blockDim.x) L[j] *= inv;
}
// (gathered, gate) -> gathered * sigmoid(gate), both [n][Wd] inside qkvg's [n][4Wd]
__global__ void gateFromQkvgK(float* o, const float* qkvg, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  size_t i = t / Wd; int c = (int)(t % Wd);
  o[t] *= sigm(qkvg[i * 4 * Wd + 3 * Wd + c]);
}
__global__ void addQBiasF32K(float* qkvg, const float* b, int n, int Wd) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * Wd) return;
  qkvg[(t / Wd) * 4 * Wd + (t % Wd)] += b[t % Wd];
}

inline void diffusionTransformer(float* act, const float* cond, const float* pairCond, const float* mask, int n) {
  const std::string T = "diffusion.transformer";
  int C = (int)M.meta(T + ".channels"), Cc = (int)M.meta(T + ".condChannels"), Cz = (int)M.meta(T + ".pairChannels");
  int heads = (int)M.meta(T + ".heads"), D = (int)M.meta(T + ".dimension"), Wd = heads * D;
  int perSuper = (int)M.meta(T + ".blocksPerSuperBlock"), factor = (int)M.meta(T + ".transitionFactor");
  if (M.flag(T + ".pairNormPerBlock") || M.flag(T + ".noResidual")) { fprintf(stderr, "transformer dialect: not ported\n"); exit(1); }
  size_t pairs = (size_t)n * n;
  float* pn = scratch<float>("dt.pn", pairs * Cz);
  layerNormSlow(pairCond, pn, pairs, Cz, W(T + ".pairInputLayerNormScale"), nullptr);
  float* flat = scratch<float>("dt.flat", pairs * perSuper * heads);
  float* pl = scratch<float>("dt.pl", (size_t)heads * pairs);
  float* x = scratch<float>("dt.x", (size_t)n * C);
  float* qkvg = scratch<float>("dt.qkvg", (size_t)n * 4 * Wd);
  float* logits = scratch<float>("dt.logits", (size_t)heads * pairs);
  float* o = scratch<float>("dt.o", (size_t)n * Wd);
  float* att = scratch<float>("dt.att", (size_t)n * C);
  float* zg = scratch<float>("dt.zg", (size_t)n * C);
  float* tn = scratch<float>("dt.tn", (size_t)n * C);
  float* wide = scratch<float>("dt.wide", (size_t)n * 2 * C * factor);
  float* gated = scratch<float>("dt.gated", (size_t)n * C * factor);
  float* proj = scratch<float>("dt.proj", (size_t)n * C);
  float* tg = scratch<float>("dt.tg", (size_t)n * C);
  const float one = 1.f, zero = 0.f;
  for (int sb = 0; sb * perSuper < 1000; ++sb) {
    std::string S = T + ".superBlocks." + std::to_string(sb);
    if (!hasW(S + ".pairLogitsProjection")) break;
    linear<float, float>(pn, flat, pairs, Cz, perSuper * heads, S + ".pairLogitsProjection");
    for (int inner = 0; inner < perSuper; ++inner) {
      std::string B = S + ".blocks." + std::to_string(inner);
      // pair logits [h][i][j] for this block: column inner*heads + h of flat
      atomLogitsLayoutK<<<blocks((size_t)heads * pairs), 256, 0, STREAM>>>(flat, pl, inner, perSuper, 1, heads, n, n);
      adaptiveLayerNorm(act, cond, x, n, C, Cc, B + ".");
      linear<float, float>(x, qkvg, n, C, 4 * Wd, qkvgWeight(B, C, Wd, false));
      addQBiasF32K<<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(qkvg, W(B + ".qBias"), n, Wd);
      CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, n, n, D, &one, qkvg + Wd, 4 * Wd, D,
                                   qkvg, 4 * Wd, D, &zero, logits, n, pairs, heads));
      tokenSoftmaxK<<<(unsigned)(heads * n), 128, 0, STREAM>>>(logits, pl, mask, n, 1.f / sqrtf((float)D));
      CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, D, n, n, &one, qkvg + 2 * Wd, 4 * Wd, D,
                                   logits, n, pairs, &zero, o, Wd, D, heads));
      gateFromQkvgK<<<blocks((size_t)n * Wd), 256, 0, STREAM>>>(o, qkvg, n, Wd);
      linear<float, float>(o, att, n, Wd, C, B + ".Transition2");
      linear<float, float>(cond, zg, n, Cc, C, B + ".AdaptiveZeroCondWeights");
      addVectorK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(zg, W(B + ".AdaptiveZeroCondBias"), n, C);
      mulSigmoidK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(att, zg, (size_t)n * C);
      addK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(act, att, (size_t)n * C);      // after attention
      adaptiveLayerNorm(act, cond, tn, n, C, Cc, B + ".ffw");
      int I = C * factor;
      linear<float, float>(tn, wide, n, C, 2 * I, B + ".ffwTransition1");
      swigluK<float><<<blocks((size_t)n * I), 256, 0, STREAM>>>(wide, gated, n, I);
      if (hasW(B + ".ffwAToB")) { fprintf(stderr, "ffwAToB: not ported\n"); exit(1); }
      linear<float, float>(gated, proj, n, I, C, B + ".ffwTransition2");
      linear<float, float>(cond, tg, n, Cc, C, B + ".ffwAdaptiveZeroCondWeights");
      addVectorK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(tg, W(B + ".ffwAdaptiveZeroCondBias"), n, C);
      mulSigmoidK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(proj, tg, (size_t)n * C);
      addK<<<blocks((size_t)n * C), 256, 0, STREAM>>>(act, proj, (size_t)n * C);
    }
  }
}

// ---------------------------------------------------------------- the decoder
__global__ void broadcastTokensK(const float* proj, float* perAtom, int tokens, int dense, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)tokens * dense * C) return;
  int c = (int)(t % C); size_t ta = t / C; int token = (int)(ta / dense);
  perAtom[t] = proj[(size_t)token * C + c];
}
__global__ void addSkipMaskK(float* act, const float* skip, const float* mask, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) act[t] = (act[t] + skip[t]) * mask[t / C];
}
inline float* atomDecoder(const float* tokenAct, const EncoderOut& enc) {
  const std::string Dd = "diffusion.decoder";
  AtomShape sh{ (int)M.meta("batch.shape.tokens"), (int)M.meta("batch.shape.dense"),
                (int)M.meta("batch.shape.subsets"), (int)M.meta("batch.shape.queries"),
                (int)M.meta("batch.shape.keys") };
  int C = (int)M.meta(Dd + ".channels"), Cp = (int)M.meta(Dd + ".pairChannels");
  int heads = (int)M.meta(Dd + ".heads"), Dh = (int)M.meta(Dd + ".dimension");
  int perToken = (int)M.meta(Dd + ".perTokenChannels");
  size_t atoms = (size_t)sh.tokens * sh.dense, qRows = (size_t)sh.subsets * sh.queries;
  Gather t2q = gatherOf("batch.tokenAtomsToQueries"), q2k = gatherOf("batch.queriesToKeys");
  Gather q2t = gatherOf("batch.queriesToTokenAtoms");
  float* proj = scratch<float>("dec.proj", (size_t)sh.tokens * C);
  linear<float, float>(tokenAct, proj, sh.tokens, perToken, C, Dd + ".projectTokenFeaturesForBroadcast");
  float* perAtom = scratch<float>("dec.perAtom", atoms * C);
  broadcastTokensK<<<blocks(atoms * C), 256, 0, STREAM>>>(proj, perAtom, sh.tokens, sh.dense, C);
  float* act = scratch<float>("dec.act", qRows * C);
  convert(t2q, perAtom, act, C);
  addSkipMaskK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, enc.skip, enc.qMask, qRows, C);
  int nblocks = 0; while (M.has(Dd + ".blocks." + std::to_string(nblocks) + ".qProjection")) ++nblocks;
  if (M.flag(Dd + ".pairNormPerBlock")) { fprintf(stderr, "per-block decoder pair norm: not ported\n"); exit(1); }
  size_t pairRows = qRows * sh.keys;
  float* pn = scratch<float>("dec.pn", pairRows * Cp);
  layerNormSlow(enc.pair, pn, pairRows, Cp, W(Dd + ".pairInputLayerNormScale"), nullptr);
  float* flat = scratch<float>("dec.flat", pairRows * nblocks * heads);
  linear<float, float>(pn, flat, pairRows, Cp, nblocks * heads, Dd + ".pairLogitsProjection");
  for (int b = 0; b < nblocks; ++b) {
    float* pl = scratch<float>("dec.pl", (size_t)sh.subsets * heads * sh.queries * sh.keys);
    atomLogitsLayoutK<<<blocks((size_t)sh.subsets * heads * sh.queries * sh.keys), 256, 0, STREAM>>>(
      flat, pl, b, nblocks, sh.subsets, heads, sh.queries, sh.keys);
    AtomState st{ q2k, enc.qMask, enc.kMask, enc.qCond, enc.kCond, { pl } };
    crossAttentionBlock(act, st, sh, C, heads, Dh, Dd + ".blocks." + std::to_string(b));
  }
  scaleByRowK<<<blocks(qRows * C), 256, 0, STREAM>>>(act, enc.qMask, qRows, C);
  float* ln = scratch<float>("dec.ln", qRows * C);
  layerNormSlow(act, ln, qRows, C, W(Dd + ".atomFeaturesLayerNormScale"), Wopt(Dd + ".atomFeaturesLayerNormOffset"));
  float* upd = scratch<float>("dec.upd", qRows * 3);
  linear<float, float>(ln, upd, qRows, C, 3, Dd + ".atomFeaturesToPositionUpdate");
  float* out = scratch<float>("dec.out", atoms * 3);
  convert(q2t, upd, out, 3);
  return out;
}

// ---------------------------------------------------------------- one denoiser call
__global__ void scalePositionsK(const float* x, const float* mask, float* y, size_t atoms, float s) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < atoms * 3) y[t] = x[t] * mask[t / 3] * s;
}
__global__ void denoiseOutK(const float* x, const float* upd, const float* mask, float* out, size_t atoms,
                            float skip, float outScale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < atoms * 3) out[t] = (skip * x[t] + outScale * upd[t]) * mask[t / 3];
}

// D(x; sigma): positions [tokens*dense][3] in, the denoised positions out.
inline float* denoise(const float* trunkSingle, const float* trunkPair, const float* targetFeat,
                      const float* seqMask, const float* positionsNoisy, float noiseLevel) {
  int n = (int)M.meta("batch.tokens"), dense = (int)M.meta("batch.dense");
  size_t atoms = (size_t)n * dense;
  double denom = (double)noiseLevel * noiseLevel + SIGMA_DATA * SIGMA_DATA;
  float sSkip = (float)(SIGMA_DATA * SIGMA_DATA / denom), sOut = (float)(noiseLevel * SIGMA_DATA / std::sqrt(denom));
  float sIn = (float)(1.0 / std::sqrt(denom));
  Conditioning cond = diffusionConditioning(trunkSingle, trunkPair, targetFeat, noiseLevel, n);
  const float* atomMask = Fdev("batch.refMask");
  float* scaled = scratch<float>("dn.scaled", atoms * 3);
  scalePositionsK<<<blocks(atoms * 3), 256, 0, STREAM>>>(positionsNoisy, atomMask, scaled, atoms, sIn);
  EncoderOut enc = atomEncoder("diffusion.encoder", "atomReference", trunkSingle, cond.pair, scaled);
  int Cs = (int)M.meta("diffusion.seqChannels"), perToken = (int)M.meta("diffusion.perTokenChannels");
  float* sn = scratch<float>("dn.sn", (size_t)n * Cs);
  layerNormSlow(cond.single, sn, n, Cs, W("diffusion.singleCondEmbeddingNormScale"), Wopt("diffusion.singleCondEmbeddingNormOffset"));
  float* act = scratch<float>("dn.act", (size_t)n * perToken);
  CK(cudaMemcpyAsync(act, enc.tokenAct, (size_t)n * perToken * 4, cudaMemcpyDeviceToDevice, STREAM));
  linear<float, float>(sn, act, n, Cs, perToken, "diffusion.singleCondEmbeddingProjection", false, 1.f);
  diffusionTransformer(act, cond.single, cond.pair, seqMask, n);
  float* actn = scratch<float>("dn.actn", (size_t)n * perToken);
  layerNormSlow(act, actn, n, perToken, W("diffusion.outputNormScale"), Wopt("diffusion.outputNormOffset"));
  float* upd = atomDecoder(actn, enc);
  float* out = scratch<float>("dn.out", atoms * 3);
  denoiseOutK<<<blocks(atoms * 3), 256, 0, STREAM>>>(positionsNoisy, upd, atomMask, out, atoms, sSkip, sOut);
  return out;
}
