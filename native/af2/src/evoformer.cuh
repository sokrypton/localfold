// The input embedder and the Evoformer block - af3-any-model's AF2 multimer graph
// (alphafold3/af2/model/modules.py: EmbeddingsAndEvoformer, EvoformerIteration), float32.
#pragma once
#include "fast.cuh"
#include "flash2.cuh"

struct Trunk {
  int L, N, E;                  // residues, MSA rows, extra MSA rows
  int T = 0;                    // template rows appended to the MSA (their single features), after N
  float* msa;                   // [N, L, 256]
  float* extra;                 // [E, L, 64]
  float* pair;                  // [L, L, 128]
  const float* msaMask;         // [N, L]
  const float* extraMask;       // [E, L]
  float* pairMask;              // [L, L]
  bool opmFirst;
  // all ones (no padding, no absent row): the flash kernel then skips the mask altogether
  bool msaOnes = false, extraOnes = false, pairOnes = false;
};

// ---------------------------------------------------------------- the embedder
// relpos [L, L, 73] one-hot, as _relative_encoding builds it
__global__ void relposK(const int* ri, const int* asym, const int* entity, const int* sym, float* out, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L) return;
  int i = (int)(t / L), j = (int)(t % L);
  float* o = out + t * 73;
  for (int c = 0; c < 73; ++c) o[c] = 0;
  int off = ri[i] - ri[j];
  int clipped = min(max(off + 32, 0), 64);
  o[asym[i] == asym[j] ? clipped : 65] = 1;
  bool sameEntity = entity[i] == entity[j];
  o[66] = sameEntity ? 1.f : 0.f;
  int rc = min(max(sym[i] - sym[j] + 2, 0), 4);
  o[67 + (sameEntity ? rc : 5)] = 1;
}
// pseudo-beta (CB; CA for glycine) from atom37 positions, then the 15-bin distogram AF2 recycles
__global__ void prevDgramK(const float* pos37, const int* aatype, float* out, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L) return;
  int i = (int)(t / L), j = (int)(t % L);
  auto pb = [&](int r, int k) { int a = aatype[r] == 7 ? 1 : 3; return pos37[((size_t)r * 37 + a) * 3 + k]; };
  float d2 = 0;
  for (int k = 0; k < 3; ++k) { float d = pb(i, k) - pb(j, k); d2 += d * d; }
  for (int b = 0; b < 15; ++b) {
    float lo = 3.25f + (20.75f - 3.25f) * b / 14.f;
    float lower = lo * lo;
    float upper = b + 1 < 15 ? (3.25f + (20.75f - 3.25f) * (b + 1) / 14.f) * (3.25f + (20.75f - 3.25f) * (b + 1) / 14.f) : 1e8f;
    out[t * 15 + b] = (d2 > lower && d2 < upper) ? 1.f : 0.f;
  }
}
__global__ void targetFeatK(const int* aatype, float* tf, int L) {      // one-hot 20 (X as V), a zero 21st
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * 21) return;
  int i = (int)(t / 21), c = (int)(t % 21);
  int a = min(max(aatype[i], 0), 19);
  tf[t] = c == a ? 1.f : 0.f;
}
__global__ void pairOuterSumK(float* pair, const float* left, const float* right, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / L), j = (int)(ij % L);
  pair[t] = left[(size_t)i * C + c] + right[(size_t)j * C + c];
}
__global__ void broadcastAddRowsK(float* msa, const float* row, size_t rows, int L, int C) {   // msa[s, i] += row[i]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) msa[t] += row[(t / C % L) * C + t % C];
}
__global__ void extraFeatK(const int* codes, const float* hasDel, const float* delVal, float* out, size_t rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * 25) return;
  size_t r = t / 25; int c = (int)(t % 25);
  out[t] = c < 23 ? (codes[r] == c ? 1.f : 0.f) : c == 23 ? hasDel[r] : delVal[r];
}
__global__ void pairMaskK(const float* seqMask, float* out, int L) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)L * L) out[t] = seqMask[t / L] * seqMask[t % L];
}

// prev: prevMsaRow [L, 256], prevPair [L, L, 128], prevPos [L, 37, 3]
inline void embed(Trunk& t, int pass, const float* prevMsaRow, const float* prevPair, const float* prevPos) {
  const std::string E = "evoformer/";
  int L = t.L, N = t.N;
  std::string f = "f" + std::to_string(pass) + "/";
  const int* aatype = Idev("aatype");
  float* tf = scratch<float>("emb.tf", (size_t)L * 21);
  targetFeatK<<<blocks((size_t)L * 21), 256, 0, STREAM>>>(aatype, tf, L);
  // msa = preprocess_1d(target)[None] + preprocess_msa(msa_feat); row 0 += LN(prev msa first row)
  float* p1d = scratch<float>("emb.p1d", (size_t)L * 256);
  linearB(tf, E + "preprocess_1d", -1, p1d, L, 21, 256);
  linearB(W(f + "msa_feat"), E + "preprocess_msa", -1, t.msa, (size_t)N * L, 49, 256);
  broadcastAddRowsK<<<blocks((size_t)N * L * 256), 256, 0, STREAM>>>(t.msa, p1d, (size_t)N * L, L, 256);
  float* prevRowLn = scratch<float>("emb.prevRow", (size_t)L * 256);
  layerNorm(prevMsaRow, prevRowLn, L, 256, E + "prev_msa_first_row_norm");
  addK2<<<blocks((size_t)L * 256), 256, 0, STREAM>>>(t.msa, prevRowLn, (size_t)L * 256);
  // pair = left[i] + right[j] + prev_pos_linear(dgram) + LN(prev pair) + relpos
  float* left = scratch<float>("emb.left", (size_t)L * 128); float* right = scratch<float>("emb.right", (size_t)L * 128);
  linearB(tf, E + "left_single", -1, left, L, 21, 128);
  linearB(tf, E + "right_single", -1, right, L, 21, 128);
  pairOuterSumK<<<blocks((size_t)L * L * 128), 256, 0, STREAM>>>(t.pair, left, right, L, 128);
  size_t pairs = (size_t)L * L;
  float* dgram = scratch<float>("emb.dgram", pairs * 15);
  prevDgramK<<<blocks(pairs), 256, 0, STREAM>>>(prevPos, aatype, dgram, L);
  float* tmp = scratch<float>("emb.tmp", pairs * 128);
  linearB(dgram, E + "prev_pos_linear", -1, tmp, pairs, 15, 128);
  addK2<<<blocks(pairs * 128), 256, 0, STREAM>>>(t.pair, tmp, pairs * 128);
  layerNorm(prevPair, tmp, pairs, 128, E + "prev_pair_norm");
  addK2<<<blocks(pairs * 128), 256, 0, STREAM>>>(t.pair, tmp, pairs * 128);
  float* rel = scratch<float>("emb.rel", pairs * 73);
  relposK<<<blocks(pairs), 256, 0, STREAM>>>(Idev("residue_index"), Idev("asym_id"), Idev("entity_id"), Idev("sym_id"), rel, L);
  linearB(rel, E + "~_relative_encoding/position_activations", -1, tmp, pairs, 73, 128);
  addK2<<<blocks(pairs * 128), 256, 0, STREAM>>>(t.pair, tmp, pairs * 128);
  // the extra MSA's activations
  size_t erows = (size_t)t.E * L;
  float* ef = scratch<float>("emb.extraFeat", erows * 25);
  extraFeatK<<<blocks(erows * 25), 256, 0, STREAM>>>(Idev(f + "extra_msa"), W(f + "extra_has_deletion"),
                                                      W(f + "extra_deletion_value"), ef, erows);
  linearB(ef, E + "extra_msa_activations", -1, t.extra, erows, 25, 64);
  // the masks into fixed buffers: the stacks after this are replayed as one CUDA graph from pass 1 on,
  // and a graph's pointers do not change between passes
  float* mm = scratch<float>("emb.msaMask", (size_t)(N + t.T) * L); float* em = scratch<float>("emb.extraMask", erows);
  CK(cudaMemcpyAsync(mm, W(f + "msa_mask"), (size_t)N * L * 4, cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemcpyAsync(em, W(f + "extra_msa_mask"), erows * 4, cudaMemcpyDeviceToDevice, STREAM));
  t.msaMask = mm; t.extraMask = em;
  auto ones = [](const float* h, size_t n) { for (size_t i = 0; i < n; ++i) if (h[i] != 1.f) return false; return true; };
  t.msaOnes = ones(M.f(f + "msa_mask"), (size_t)N * L) && t.T == 0;     // (a template row's mask is the template's)
  t.extraOnes = ones(M.f(f + "extra_msa_mask"), erows);
  t.pairOnes = ones(M.f("seq_mask"), L);
  pairMaskK<<<blocks(pairs), 256, 0, STREAM>>>(W("seq_mask"), t.pairMask, L);
}

// ---------------------------------------------------------------- the block's modules
// gated self-attention with its own q/k/v/gate/output weights, over [Bt, n, C] rows
inline void gatedAttention(const float* xn, int Bt, int n, int C, const std::string& A, int blk, int H, int D,
                           const float* keyMask, const float* pairBias, float* out) {
  size_t rows = (size_t)Bt * n; int Wd = H * D;
  float* q = scratch<float>("att.q", rows * Wd); float* k = scratch<float>("att.k", rows * Wd);
  float* v = scratch<float>("att.v", rows * Wd); float* g = scratch<float>("att.g", rows * Wd);
  float* o = scratch<float>("att.o", rows * Wd);
  gemm(xn, P(A + "/query_w", blk), q, rows, C, Wd);
  gemm(xn, P(A + "/key_w", blk), k, rows, C, Wd);
  gemm(xn, P(A + "/value_w", blk), v, rows, C, Wd);
  gemm(xn, P(A + "/gating_w", blk), g, rows, C, Wd);
  addStridedBiasK<<<blocks(rows * Wd), 256, 0, STREAM>>>(g, P(A + "/gating_b", blk), rows, Wd, Wd);
  attention(D, q, k, v, g, keyMask, pairBias, o, Bt, n, H);
  gemm(o, P(A + "/output_w", blk), out, rows, Wd, C);
  addBiasK<<<blocks(rows * C), 256, 0, STREAM>>>(out, P(A + "/output_b", blk), rows, C);
}
// [pairs, H] f32 pair-bias projection (+ 1e9 (mask - 1) when a mask is given) -> [H, L, stride] f16 in
// the flash kernel's log2 units
__global__ void biasFromProjK(const float* proj, const float* pairMask, half* out, int L, int H, int stride, bool transposed) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * H) return;
  int h = (int)(t % H); size_t ij = t / H; int i = (int)(ij / L), j = (int)(ij % L);
  float v = proj[t] + (pairMask ? 1e9f * (pairMask[ij] - 1.f) : 0.f);
  // transposed: the ending node's bias, b[h][q][k] = proj[k][q] (it attends along the first axis)
  if (transposed) { int x = i; i = j; j = x; }
  out[((size_t)h * L + i) * stride + j] = __float2half(fmaxf(v * LOG2E, -6e4f));
}
inline const half* zeroBias(int H, int n, int stride) {        // a bias-free attention's bias, once
  static half* z = nullptr; static size_t have = 0;
  size_t need = (size_t)H * n * stride;
  if (need > have) {
    if (z) CK(cudaFree(z));
    z = dallocT<half>(need); CK(cudaMemset(z, 0, need * 2)); have = need;
  }
  return z;
}
// --fast: from the normalised input xn (f16): one q/k/v/gate GEMM straight into the flash kernel's
// layout -> flash attention -> the output projection with its bias, added into `residual` (transposed
// back first when `transposedBack`, xn being [n, Bt] of the residual's [Bt, n])
inline void attentionCore(const half* xn, int Bt, int n, int C, const std::string& A, int blk, const float* keyMask,
                          const half* bias, float* residual, bool transposedBack) {
  size_t rows = (size_t)Bt * n;
  AttnW w = attnWeights(A, blk, C);
  int Wp = w.H * w.Dp, stride = (n + 7) / 8 * 8;
  half* qkvg = scratch<half>("fatt.qkvg", (rows + 128) * 4 * Wp);
  ltGemm(xn, w.qkvg, qkvg, true, rows, C, 4 * Wp, w.qkvgBias, false, 0.f);
  if (!bias) bias = zeroBias(w.H, n, stride);
  half* o = scratch<half>("fatt.o", rows * Wp);
  flashGrid<half>(qkvg, bias, stride, keyMask, o, n, w.H, w.Dp, 0, Bt, false, 1.f / sqrtf((float)w.D));
  if (!transposedBack) {
    ltGemm(o, w.out, residual, false, rows, Wp, C, P(A + "/output_b", blk), false, 1.f);
  } else {
    float* tmp = scratch<float>("fatt.tmp", rows * C);
    ltGemm(o, w.out, tmp, false, rows, Wp, C, P(A + "/output_b", blk), false, 0.f);
    swapAddK<<<blocks(rows * C), 256, 0, STREAM>>>(residual, tmp, Bt, n, C);
  }
}
// the same, attending ACROSS the leading axis of xn [n][Bt][C] (an MSA's columns, the triangle's ending
// node) where it lies: the strided flash kernel reads rows and positions at strides, and the output
// lands in the residual's own layout, so nothing is transposed
inline void attentionCoreAcross(const half* xn, int Bt, int n, int C, const std::string& A, int blk, const half* bias,
                                float* residual) {
  size_t rows = (size_t)Bt * n;
  AttnW w = attnWeights(A, blk, C);
  int Wp = w.H * w.Dp, stride = (n + 7) / 8 * 8;
  half* qkvg = scratch<half>("fatt.qkvg", (rows + 128) * 4 * Wp);
  ltGemm(xn, w.qkvg, qkvg, true, rows, C, 4 * Wp, w.qkvgBias, false, 0.f);
  if (!bias) bias = zeroBias(w.H, n, stride);
  half* o = scratch<half>("fatt.o", rows * Wp);
  flashGridStrided(qkvg, bias, stride, nullptr, o, n, w.H, w.Dp, Bt, 1.f / sqrtf((float)w.D),
                   (size_t)4 * Wp, (size_t)Bt * 4 * Wp, (size_t)Wp, (size_t)Bt * Wp);
  ltGemm(o, w.out, residual, false, rows, Wp, C, P(A + "/output_b", blk), false, 1.f);
}
// the pair bias, --fast: LN(pair) in f16 -> [pairs, H] -> the flash layout
inline const half* pairBiasFast(const half* pn, int L, int C, int H, const float* w, const float* pairMask, bool transposed = false) {
  size_t pairs = (size_t)L * L; int stride = (L + 7) / 8 * 8;
  float* proj = scratch<float>("fbias.proj", pairs * H);
  static std::map<const float*, half*> wh;
  auto it = wh.find(w);
  if (it == wh.end()) {
    half* h = wpool<half>((size_t)C * H);
    toHalfK<<<blocks((size_t)C * H), 256, 0, STREAM>>>(w, h, (size_t)C * H);
    it = wh.emplace(w, h).first;
  }
  ltGemm(pn, it->second, proj, false, pairs, C, H, nullptr, false, 0.f);
  half* bias = scratch<half>("fbias.bias", (size_t)H * L * stride);
  // the pad columns past L are read by the kernel's 16-byte loads beside the last real ones: a stale
  // NaN there (the buffer is shared) poisons the row, so they must hold zeros
  if (stride != L) CK(cudaMemsetAsync(bias, 0, (size_t)H * L * stride * 2, STREAM));
  biasFromProjK<<<blocks(pairs * H), 256, 0, STREAM>>>(proj, pairMask, bias, L, H, stride, transposed);
  return bias;
}

// pair bias [H, L, L] = LN(pair) W (+ 1e9 (pairMask - 1) for the MSA rows)
__global__ void pairBiasK(const float* proj, const float* pairMask, float* bias, int L, int H) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * H) return;
  int h = (int)(t % H); size_t ij = t / H;
  bias[(size_t)h * L * L + ij] = proj[t] + (pairMask ? 1e9f * (pairMask[ij] - 1.f) : 0.f);
}

inline void msaRowAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C, int H, int D,
                            const float* msaMask) {
  int L = t.L; size_t pairs = (size_t)L * L, rows = (size_t)rowsN * L;
  std::string R = S + "msa_row_attention_with_pair_bias";
  if (FAST) {
    half* pn = scratch<half>("frow.pn", pairs * 128);
    layerNormH(t.pair, pn, pairs, 128, R + "/feat_2d_norm", blk);
    const half* bias = pairBiasFast(pn, L, 128, H, P(R + "/feat_2d_weights", blk), t.pairOnes ? nullptr : t.pairMask);
    half* xn = scratch<half>("frow.xn", rows * C);
    layerNormH(msa, xn, rows, C, R + "/query_norm", blk);
    bool ones = msa == t.msa ? t.msaOnes : t.extraOnes;
    attentionCore(xn, rowsN, L, C, R + "/attention", blk, ones ? nullptr : msaMask, bias, msa, false);
    return;
  }
  float* pn = scratch<float>("row.pn", pairs * 128);
  layerNorm(t.pair, pn, pairs, 128, R + "/feat_2d_norm", blk);
  float* proj = scratch<float>("row.proj", pairs * H);
  gemm(pn, P(R + "/feat_2d_weights", blk), proj, pairs, 128, H);
  float* bias = scratch<float>("row.bias", pairs * H);
  pairBiasK<<<blocks(pairs * H), 256, 0, STREAM>>>(proj, t.pairMask, bias, L, H);
  float* xn = scratch<float>("row.xn", rows * C);
  layerNorm(msa, xn, rows, C, R + "/query_norm", blk);
  float* out = scratch<float>("row.out", rows * C);
  gatedAttention(xn, rowsN, L, C, R + "/attention", blk, H, D, msaMask, bias, out);
  addK2<<<blocks(rows * C), 256, 0, STREAM>>>(msa, out, rows * C);
}
inline void msaColumnAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C, int H, int D,
                               const float* msaMask) {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string A = S + "msa_column_attention";
  if (FAST && t.msaOnes) {
    half* xn = scratch<half>("fcol.xn", rows * C);
    layerNormH(msa, xn, rows, C, A + "/query_norm", blk);
    attentionCoreAcross(xn, L, rowsN, C, A + "/attention", blk, nullptr, msa);
    return;
  }
  float* tr = scratch<float>("col.tr", rows * C);
  float* mt = scratch<float>("col.mask", rows);
  swap01(msa, tr, rowsN, L, C);                 // [L, N, C]
  swap01(msaMask, mt, rowsN, L, 1);
  if (FAST) {
    half* xn = scratch<half>("fcol.xn", rows * C);
    layerNormH(tr, xn, rows, C, A + "/query_norm", blk);
    attentionCore(xn, L, rowsN, C, A + "/attention", blk, t.msaOnes ? nullptr : mt, nullptr, msa, true);
    return;
  }
  float* xn = scratch<float>("col.xn", rows * C);
  layerNorm(tr, xn, rows, C, A + "/query_norm", blk);
  float* out = scratch<float>("col.out", rows * C);
  gatedAttention(xn, L, rowsN, C, A + "/attention", blk, H, D, mt, nullptr, out);
  swap01(out, tr, L, rowsN, C);
  addK2<<<blocks(rows * C), 256, 0, STREAM>>>(msa, tr, rows * C);
}
inline void msaColumnGlobalAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C,
                                     const float* msaMask) {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string A = S + "msa_column_global_attention";
  int H = (int)dimW(A + "/attention/query_w", 2), D = (int)dimW(A + "/attention/query_w", 3), Wd = H * D;
  float* tr = scratch<float>("gcol.tr", rows * C); float* xn = scratch<float>("gcol.xn", rows * C);
  float* mt = scratch<float>("gcol.mask", rows);
  swap01(msa, tr, rowsN, L, C);
  swap01(msaMask, mt, rowsN, L, 1);
  layerNorm(tr, xn, rows, C, A + "/query_norm", blk);
  float* avg = scratch<float>("gcol.avg", (size_t)L * Wd);
  size_t smem = (size_t)(C + Wd + H * rowsN + 2 * rowsN * D) * 4;
  if (fitsSmem(smem) || H != 8 || D != 8) {
    static bool attr = false;
    if (!attr) { smemAttr((globalAttentionK), std::min<size_t>(160 * 1024, smemLimit())); attr = true; }
    if (!fitsSmem(smem) || smem > 160 * 1024) { fprintf(stderr, "global attention: %d sequences do not fit one block\n", rowsN); exit(1); }
    globalAttentionK<<<L, 256, smem, STREAM>>>(xn, mt, P(A + "/attention/query_w", blk), P(A + "/attention/key_w", blk),
                                               P(A + "/attention/value_w", blk), avg, rowsN, C, H, D);
  } else {     // a device with less shared memory (a T4): the streamed form
    globalAttentionStreamK<8, 8><<<L, 256, (size_t)(C + Wd + 8 * H * (D + 2)) * 4, STREAM>>>(
      xn, mt, P(A + "/attention/query_w", blk), P(A + "/attention/key_w", blk), P(A + "/attention/value_w", blk), avg, rowsN, C);
  }
  float* gate = scratch<float>("gcol.gate", rows * Wd);
  gemm(xn, P(A + "/attention/gating_w", blk), gate, rows, C, Wd);
  addStridedBiasK<<<blocks(rows * Wd), 256, 0, STREAM>>>(gate, P(A + "/attention/gating_b", blk), rows, Wd, Wd);
  float* gated = scratch<float>("gcol.gated", rows * Wd);
  globalGateK<<<blocks(rows * Wd), 256, 0, STREAM>>>(avg, gate, gated, L, rowsN, Wd);
  float* out = scratch<float>("gcol.out", rows * C);
  gemm(gated, P(A + "/attention/output_w", blk), out, rows, Wd, C);
  addBiasK<<<blocks(rows * C), 256, 0, STREAM>>>(out, P(A + "/attention/output_b", blk), rows, C);
  swap01(out, tr, L, rowsN, C);
  addK2<<<blocks(rows * C), 256, 0, STREAM>>>(msa, tr, rows * C);
}
inline void transition(float* x, size_t rows, int C, const std::string& T, int blk) {
  int I = (int)dimW(T + "/transition1/weights", blk < 0 ? 1 : 2);     // 4C in the stacks, 2C in the template's
  if (FAST) {
    // in row chunks of ~128 MB of the widened rows (the whole widened tensor was 1.26 GB of a pair track
    // at 783 residues; a chunk of 2^15+ rows keeps the GEMMs as fast)
    size_t chunk = std::min(rows, std::max<size_t>(32768, ((size_t)128 << 20) / (2 * (size_t)(I + 8))));
    half* xh = scratch<half>("ftr.xn", chunk * C);
    half* mh = augmentedInput("ftr.mid" + std::to_string(I), chunk, I);     // [rows, I+8], a 1 at column I
    const half* w2 = augmentedWeight(T + "/transition2/weights", blk, P(T + "/transition2/bias", blk), I, C);
    for (size_t r0 = 0; r0 < rows; r0 += chunk) {
      size_t r = std::min(chunk, rows - r0);
      layerNormH(x + r0 * C, xh, r, C, T + "/input_layer_norm", blk);
      ltGemm(xh, PH(T + "/transition1/weights", blk), mh, true, r, C, I, P(T + "/transition1/bias", blk), true, 0.f, 0, I + 8);
      // the second layer's bias carried by the product (its epilogue with a residual add ran as a second
      // kernel over the whole MSA): the 1 at column I picks up the bias row of the augmented weight
      ltGemm(mh, w2, x + r0 * C, false, r, I + 8, C, nullptr, false, 1.f);
    }
    return;
  }
  float* xn = scratch<float>("tr.xn", rows * C);
  float* mid = scratch<float>("tr.mid", rows * I);
  float* out = scratch<float>("tr.out", rows * C);
  layerNorm(x, xn, rows, C, T + "/input_layer_norm", blk);
  linearB(xn, T + "/transition1", blk, mid, rows, C, I, true);
  linearB(mid, T + "/transition2", blk, out, rows, I, C);
  addK2<<<blocks(rows * C), 256, 0, STREAM>>>(x, out, rows * C);
}
inline void outerProductMean(Trunk& t, const std::string& S, int blk, const float* msa, int rowsN, int C,
                             const float* msaMask) {
  int L = t.L; size_t rows = (size_t)rowsN * L; const int O = 32;
  std::string Op = S + "outer_product_mean";
  if (FAST) {
    half* xn = scratch<half>("fopm.xn", rows * C);
    layerNormH(msa, xn, rows, C, Op + "/layer_norm_input", blk);
    half* lt = scratch<half>("fopm.left", rows * O); half* rt = scratch<half>("fopm.right", rows * O);
    ltGemm(xn, PH(Op + "/left_projection/weights", blk), lt, true, rows, C, O, P(Op + "/left_projection/bias", blk), false, 0.f);
    ltGemm(xn, PH(Op + "/right_projection/weights", blk), rt, true, rows, C, O, P(Op + "/right_projection/bias", blk), false, 0.f);
    bool ones = msa == t.msa ? t.msaOnes : t.extraOnes;
    if (!ones) {
      scaleRowsHK<<<blocks(rows * O), 256, 0, STREAM>>>(lt, msaMask, rows, O);
      scaleRowsHK<<<blocks(rows * O), 256, 0, STREAM>>>(rt, msaMask, rows, O);
    }
    float* norm = scratch<float>("opm.norm", (size_t)L * L);
    const float one = 1.f, zero = 0.f;
    CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, rowsN, &one, msaMask, L, msaMask, L, &zero, norm, L));
    size_t per = (size_t)L * O * O;
    int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / per));
    half* Pm = scratch<half>("fopm.P", (size_t)Bi * per);
    half* X = scratch<half>("fopm.X", (size_t)Bi * per);
    float* Y = scratch<float>("fopm.Y", (size_t)Bi * L * 128);
    for (int i0 = 0; i0 < L; i0 += Bi) {
      int bi = std::min(Bi, L - i0);
      CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_T, L * O, bi * O, rowsN, &one, rt, CUDA_R_16F, L * O, lt + (size_t)i0 * O,
                      CUDA_R_16F, L * O, &zero, Pm, CUDA_R_16F, L * O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      opmPermuteHK<<<blocks((size_t)bi * per / 8), 256, 0, STREAM>>>(Pm, X, bi, L, O);
      ltGemm(X, PH(Op + "/output_w", blk), Y, false, (size_t)bi * L, O * O, 128, nullptr, false, 0.f);
      opmAddK2<<<blocks((size_t)bi * L * 128), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L, 128);
    }
    return;
  }
  float* xn = scratch<float>("opm.xn", rows * C);
  layerNorm(msa, xn, rows, C, Op + "/layer_norm_input", blk);
  float* lt = scratch<float>("opm.left", rows * O); float* rt = scratch<float>("opm.right", rows * O);
  linearB(xn, Op + "/left_projection", blk, lt, rows, C, O);
  linearB(xn, Op + "/right_projection", blk, rt, rows, C, O);
  scaleRowsK2<<<blocks(rows * O), 256, 0, STREAM>>>(lt, msaMask, rows, O);
  scaleRowsK2<<<blocks(rows * O), 256, 0, STREAM>>>(rt, msaMask, rows, O);
  float* norm = scratch<float>("opm.norm", (size_t)L * L);
  const float one = 1.f, zero = 0.f;
  // norm[i, j] = sum_s mask[s, i] mask[s, j]
  CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, rowsN, &one, msaMask, L, msaMask, L, &zero, norm, L));
  size_t per = (size_t)L * O * O;
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / per));
  float* Pm = scratch<float>("opm.P", (size_t)Bi * per);
  float* X = scratch<float>("opm.X", (size_t)Bi * per);
  float* Y = scratch<float>("opm.Y", (size_t)Bi * L * 128);
  for (int i0 = 0; i0 < L; i0 += Bi) {
    int bi = std::min(Bi, L - i0);
    // row-major P (bi*O x L*O) = left_blk^T right, as col-major P^T = right * left_blk^T
    CB(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_T, L * O, bi * O, rowsN, &one, rt, L * O, lt + (size_t)i0 * O, L * O,
                   &zero, Pm, L * O));
    opmPermuteK2<<<blocks((size_t)bi * per), 256, 0, STREAM>>>(Pm, X, bi, L, O);
    gemm(X, P(Op + "/output_w", blk), Y, (size_t)bi * L, O * O, 128);
    opmAddK2<<<blocks((size_t)bi * L * 128), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L, 128);
  }
}
// a, b [c][i][k] (channel-major) from the projection's two halves, masked and gated
__global__ void triSplitK(const float* proj, const float* gate, const float* mask, float* a, float* b, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * 2 * C) return;
  int c2 = (int)(t % (2 * C)); size_t ij = t / (2 * C);
  float v = proj[t] * mask[ij] / (1.f + __expf(-gate[t]));
  if (c2 < C) a[(size_t)c2 * L * L + ij] = v; else b[(size_t)(c2 - C) * L * L + ij] = v;
}
__global__ void channelMajorToRowsK(const float* x, float* y, size_t pairs, int C) {   // [c][ij] -> [ij][c]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < pairs * C) y[t] = x[(t % C) * pairs + t / C];
}
__global__ void gateMulAddK(float* pair, const float* out, const float* gate, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) pair[t] += out[t] / (1.f + __expf(-gate[t]));
}
inline void triangleMultiplication(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk,
                                   bool outgoing) {
  size_t pairs = (size_t)L * L;
  std::string T = S + (outgoing ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming");
  if (FAST) {
    TriW w = triWeights(T, blk, C);
    half* xn = scratch<half>("ftri.xn", pairs * C);
    layerNormH(pair, xn, pairs, C, T + "/left_norm_input", blk);
    // planes [Lp][Lp], the pad rows and columns zero (written once: nothing else writes them)
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
    static half* zeroed = nullptr; static size_t zeroedBytes = 0;
    if (Lp != L && (a != zeroed || plane * C * 2 > zeroedBytes)) {
      CK(cudaMemsetAsync(a, 0, plane * C * 2, STREAM)); CK(cudaMemsetAsync(b, 0, plane * C * 2, STREAM));
      zeroed = a; zeroedBytes = plane * C * 2;
    }
    // the five projections in row chunks of ~128 MB (whole, [pairs, 5C] was 0.78 GB at 783 residues):
    // a and b gated into their planes, the output gate kept [pairs, C] for the end
    size_t chunk = std::min(pairs, std::max<size_t>(32768, ((size_t)128 << 20) / (10 * (size_t)C)));
    half* pg = scratch<half>("ftri.pg", chunk * 5 * C);
    half* og = scratch<half>("ftri.og", pairs * C);
    for (size_t p0 = 0; p0 < pairs; p0 += chunk) {
      size_t r = std::min(chunk, pairs - p0);
      ltGemm(xn + p0 * C, w.w5, pg, true, r, C, 5 * C, w.b5, false, 0.f);
      triGateTK<<<dim3((unsigned)((r + 31) / 32), C / 32), dim3(32, 8), 0, STREAM>>>(pg, pairMask, a, b, r, C, L, Lp, p0);
      CK(cudaMemcpy2DAsync(og + p0 * C, C * 2, pg + 4 * C, 5 * C * 2, C * 2, r, cudaMemcpyDeviceToDevice, STREAM));
    }
    float* prod = scratch<float>("ftri.prod", plane * C);
    const float one = 1.f, zero = 0.f;
    // every extent Lp: the pad's zeros add nothing to a sum, and the pad's outputs are never read
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    half* cn = xn;                // (the normalised input is spent)
    centerNormTK<<<(unsigned)((pairs + 31) / 32), 256, (size_t)C * 33 * 4, STREAM>>>(prod, cn, pairs, C,
      P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk), L, Lp);
    float* out = scratch<float>("ftri.out", pairs * C);
    ltGemm(cn, w.out, out, false, pairs, C, C, P(T + "/output_projection/bias", blk), false, 0.f);
    gateMulAddHK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, out, og, pairs, C, C);
    return;
  }
  float* xn = scratch<float>("tri.xn", pairs * C);
  layerNorm(pair, xn, pairs, C, T + "/left_norm_input", blk);
  float* proj = scratch<float>("tri.proj", pairs * 2 * C); float* gate = scratch<float>("tri.gate", pairs * 2 * C);
  linearB(xn, T + "/projection", blk, proj, pairs, C, 2 * C);
  linearB(xn, T + "/gate", blk, gate, pairs, C, 2 * C);
  float* a = scratch<float>("tri.a", pairs * C); float* b = scratch<float>("tri.b", pairs * C);
  triSplitK<<<blocks(pairs * 2 * C), 256, 0, STREAM>>>(proj, gate, pairMask, a, b, L, C);
  float* prod = scratch<float>("tri.prod", pairs * C);
  const float one = 1.f, zero = 0.f;
  // per channel (row-major [i][k] planes): outgoing act[i,j] = sum_k a[i,k] b[j,k] = a b^T;
  // incoming act[i,j] = sum_k a[k,j] b[k,i] = (b^T a)[i,j]
  if (outgoing)
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, L, L, L, &one, b, L, pairs, a, L, pairs, &zero, prod, L,
                                 pairs, C));
  else
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_T, L, L, L, &one, a, L, pairs, b, L, pairs, &zero, prod, L,
                                 pairs, C));
  float* rowsP = scratch<float>("tri.rows", pairs * C);
  channelMajorToRowsK<<<blocks(pairs * C), 256, 0, STREAM>>>(prod, rowsP, pairs, C);
  float* cn = scratch<float>("tri.cn", pairs * C);
  layerNorm(rowsP, cn, pairs, C, T + "/center_norm", blk);
  float* out = scratch<float>("tri.out", pairs * C);
  linearB(cn, T + "/output_projection", blk, out, pairs, C, C);
  float* g = scratch<float>("tri.g", pairs * C);
  linearB(xn, T + "/gating_linear", blk, g, pairs, C, C);
  gateMulAddK<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, out, g, pairs * C);
}
inline void triangleAttention(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk,
                              bool starting, bool pairOnes = false) {
  size_t pairs = (size_t)L * L;
  std::string A = S + (starting ? "triangle_attention_starting_node" : "triangle_attention_ending_node");
  int H = (int)dimW(A + "/attention/query_w", 2), D = (int)dimW(A + "/attention/query_w", 3);
  if (FAST && pairOnes && !starting) {
    half* xn = scratch<half>("ftatt.xn", pairs * C);
    layerNormH(pair, xn, pairs, C, A + "/query_norm", blk);
    const half* bias = pairBiasFast(xn, L, C, H, P(A + "/feat_2d_weights", blk), nullptr, true);
    attentionCoreAcross(xn, L, L, C, A + "/attention", blk, bias, pair);
    return;
  }
  const float* x = pair; const float* mask = pairMask;
  float* tr = starting ? nullptr : scratch<float>("tatt.tr", pairs * C);
  float* mt = starting ? nullptr : scratch<float>("tatt.mask", pairs);
  if (!starting) { swap01(pair, tr, L, L, C); if (!FAST || !pairOnes) swap01(pairMask, mt, L, L, 1); x = tr; mask = mt; }
  if (FAST) {
    half* xn = scratch<half>("ftatt.xn", pairs * C);
    layerNormH(x, xn, pairs, C, A + "/query_norm", blk);
    const half* bias = pairBiasFast(xn, L, C, H, P(A + "/feat_2d_weights", blk), nullptr);
    attentionCore(xn, L, L, C, A + "/attention", blk, pairOnes ? nullptr : mask, bias, pair, !starting);
    return;
  }
  float* xn = scratch<float>("tatt.xn", pairs * C);
  layerNorm(x, xn, pairs, C, A + "/query_norm", blk);
  float* proj = scratch<float>("tatt.proj", pairs * H);
  gemm(xn, P(A + "/feat_2d_weights", blk), proj, pairs, C, H);
  float* bias = scratch<float>("tatt.bias", pairs * H);
  pairBiasK<<<blocks(pairs * H), 256, 0, STREAM>>>(proj, nullptr, bias, L, H);
  float* out = scratch<float>("tatt.out", pairs * C);
  gatedAttention(xn, L, L, C, A + "/attention", blk, H, D, mask, bias, out);
  if (!starting) { swap01(out, tr, L, L, C); out = tr; }
  addK2<<<blocks(pairs * C), 256, 0, STREAM>>>(pair, out, pairs * C);
}

// one Evoformer iteration of a stack: S is "evoformer/evoformer_iteration/" or ".../extra_msa_stack/"
inline void evoformerBlock(Trunk& t, bool extraStack, int blk) {
  const std::string S = extraStack ? "evoformer/extra_msa_stack/" : "evoformer/evoformer_iteration/";
  float* msa = extraStack ? t.extra : t.msa;
  int rowsN = extraStack ? t.E : t.N + t.T, C = extraStack ? 64 : 256;
  const float* mask = extraStack ? t.extraMask : t.msaMask;
  std::string R = S + "msa_row_attention_with_pair_bias/attention/query_w";
  int H = (int)dimW(R, 2), D = (int)dimW(R, 3);
  if (t.opmFirst) outerProductMean(t, S, blk, msa, rowsN, C, mask);
  msaRowAttention(t, S, blk, msa, rowsN, C, H, D, mask);
  if (extraStack) msaColumnGlobalAttention(t, S, blk, msa, rowsN, C, mask);
  else msaColumnAttention(t, S, blk, msa, rowsN, C, H, D, mask);
  transition(msa, (size_t)rowsN * t.L, C, S + "msa_transition", blk);
  if (!t.opmFirst) outerProductMean(t, S, blk, msa, rowsN, C, mask);
  triangleMultiplication(t.pair, t.pairMask, t.L, 128, S, blk, true);
  triangleMultiplication(t.pair, t.pairMask, t.L, 128, S, blk, false);
  triangleAttention(t.pair, t.pairMask, t.L, 128, S, blk, true, t.pairOnes);
  triangleAttention(t.pair, t.pairMask, t.L, 128, S, blk, false, t.pairOnes);
  transition(t.pair, (size_t)t.L * t.L, 128, S + "pair_transition", blk);
}
