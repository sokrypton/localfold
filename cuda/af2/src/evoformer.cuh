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
inline size_t AF2_CHUNK = (size_t)64 << 20;     // elements a row-chunked tensor holds, on a card short of room
// ---------------------------------------------------------------- the pair in bf16 (AF2_P16)
// Where every update the stacks run has a bf16-pair form (af2Pair16Ok), the pair is held in bf16 from the embedder
// to the end of the Evoformer: the stacks' pair kernels take it through cuda/af3's PAIR16 (set around them in
// evoformerBlock), the embedder and the templates add into it here, and after the stacks it is converted once into
// the recycled pair (f32), which the structure module, the heads and the next pass's embedder read.
// LOCALFOLD_PAIR_F32=1 keeps the f32 pair.
inline bool AF2_P16 = false;
template <class PT, class TX>
__global__ void addIntoPairK(float* pair, const TX* x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pairSt<PT>(pair, i, pairLd<PT>(pair, i) + toF(x[i]));
}
__global__ void pairBf16ToF32K(const __nv_bfloat16* in, float* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}
// the pair as f32 for a reader with no bf16 form (the template embedders' query): itself, or a converted copy
inline const float* pairF32(const float* pair, size_t n) {
  if (!AF2_P16) return pair;
  float* f = scratch<float>("p16.view", n);
  pairBf16ToF32K<<<blocks(n), 256, 0, STREAM>>>(reinterpret_cast<const __nv_bfloat16*>(pair), f, n);
  return f;
}
template <class PT = float>
__global__ void pairOuterSumK(float* pair, const float* left, const float* right, int L, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / L), j = (int)(ij % L);
  pairSt<PT>(pair, t, left[(size_t)i * C + c] + right[(size_t)j * C + c]);
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

// a fold short of room (af2.cu): the chunked forms may engage, and no CUDA graph is captured, so a choice
// made from the free memory cannot differ between a pass and its capture
inline bool AF2_TIGHT = false;
// rows [r0, r0 + cnt) of the pair features, for the in-place embedding below: the relative encoding,
// the previous positions' distogram, and the sum that writes the rows
__global__ void relposRowsK(const int* ri, const int* asym, const int* entity, const int* sym, float* out, int L,
                            size_t r0, size_t cnt) {
  size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= cnt) return;
  size_t t = r0 + q; int i = (int)(t / L), j = (int)(t % L);
  float* o = out + q * 73;
  for (int c = 0; c < 73; ++c) o[c] = 0;
  int off = ri[i] - ri[j];
  int clipped = min(max(off + 32, 0), 64);
  o[asym[i] == asym[j] ? clipped : 65] = 1;
  bool sameEntity = entity[i] == entity[j];
  o[66] = sameEntity ? 1.f : 0.f;
  int rc = min(max(sym[i] - sym[j] + 2, 0), 4);
  o[67 + (sameEntity ? rc : 5)] = 1;
}
__global__ void prevDgramRowsK(const float* pos37, const int* aatype, float* out, int L, size_t r0, size_t cnt) {
  size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= cnt) return;
  size_t t = r0 + q; int i = (int)(t / L), j = (int)(t % L);
  auto pb = [&](int r, int k) { int a = aatype[r] == 7 ? 1 : 3; return pos37[((size_t)r * 37 + a) * 3 + k]; };
  float d2 = 0;
  for (int k = 0; k < 3; ++k) { float d = pb(i, k) - pb(j, k); d2 += d * d; }
  for (int b = 0; b < 15; ++b) {
    float lo = 3.25f + (20.75f - 3.25f) * b / 14.f;
    float lower = lo * lo;
    float upper = b + 1 < 15 ? (3.25f + (20.75f - 3.25f) * (b + 1) / 14.f) * (3.25f + (20.75f - 3.25f) * (b + 1) / 14.f) : 1e8f;
    out[q * 15 + b] = (d2 > lower && d2 < upper) ? 1.f : 0.f;
  }
}
// pair rows = left[i] + right[j], + the distogram term, + LN(the old row), + the relative term - the
// whole form's additions in its order
__global__ void embedRowsK(const float* left, const float* right, const float* dg, const float* ln, const float* rel,
                           float* pair, int L, size_t r0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * 128) return;
  int c = (int)(t % 128); size_t ij = r0 + t / 128; size_t i = ij / L, j = ij % L;
  float v = left[i * 128 + c] + right[j * 128 + c];
  v += dg[t]; v += ln[t]; v += rel[t];
  pair[r0 * 128 + t] = v;
}
// prev: prevMsaRow [L, 256], prevPair [L, L, 128] (null: the pair itself, re-embedded in place), prevPos [L, 37, 3]
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
  size_t pairs = (size_t)L * L;
  if (!prevPair) {
    if (AF2_P16) { fprintf(stderr, "the in-place re-embedding has no bf16-pair form\n"); exit(1); }
    // on a card short of room the recycled pair IS the pair, re-embedded in place a chunk of rows at a
    // time - every term of a row reads only that row - so no second pair-sized tensor is kept
    size_t per = std::max<size_t>(1, std::min(pairs, AF2_CHUNK / 128));
    float* dg = scratch<float>("emb.dgram", per * 15); float* dgl = scratch<float>("emb.dgl", per * 128);
    float* ln = scratch<float>("emb.prevln", per * 128);
    float* rel = scratch<float>("emb.rel", per * 73); float* rell = scratch<float>("emb.rell", per * 128);
    for (size_t r0 = 0; r0 < pairs; r0 += per) {
      size_t r = std::min(per, pairs - r0);
      prevDgramRowsK<<<blocks(r), 256, 0, STREAM>>>(prevPos, aatype, dg, L, r0, r);
      linearB(dg, E + "prev_pos_linear", -1, dgl, r, 15, 128);
      layerNorm(t.pair + r0 * 128, ln, r, 128, E + "prev_pair_norm");
      relposRowsK<<<blocks(r), 256, 0, STREAM>>>(Idev("residue_index"), Idev("asym_id"), Idev("entity_id"), Idev("sym_id"),
                                                 rel, L, r0, r);
      linearB(rel, E + "~_relative_encoding/position_activations", -1, rell, r, 73, 128);
      embedRowsK<<<blocks(r * 128), 256, 0, STREAM>>>(left, right, dgl, ln, rell, t.pair, L, r0, r);
    }
  } else {
  WITH_PT(AF2_P16, pairOuterSumK<PT><<<blocks((size_t)L * L * 128), 256, 0, STREAM>>>(t.pair, left, right, L, 128));
  float* dgram = scratch<float>("emb.dgram", pairs * 15);
  prevDgramK<<<blocks(pairs), 256, 0, STREAM>>>(prevPos, aatype, dgram, L);
  float* tmp = scratch<float>("emb.tmp", pairs * 128);
  linearB(dgram, E + "prev_pos_linear", -1, tmp, pairs, 15, 128);
  WITH_PT(AF2_P16, addIntoPairK<PT, float><<<blocks(pairs * 128), 256, 0, STREAM>>>(t.pair, tmp, pairs * 128));
  layerNorm(prevPair, tmp, pairs, 128, E + "prev_pair_norm");
  WITH_PT(AF2_P16, addIntoPairK<PT, float><<<blocks(pairs * 128), 256, 0, STREAM>>>(t.pair, tmp, pairs * 128));
  float* rel = scratch<float>("emb.rel", pairs * 73);
  relposK<<<blocks(pairs), 256, 0, STREAM>>>(Idev("residue_index"), Idev("asym_id"), Idev("entity_id"), Idev("sym_id"), rel, L);
  linearB(rel, E + "~_relative_encoding/position_activations", -1, tmp, pairs, 73, 128);
  WITH_PT(AF2_P16, addIntoPairK<PT, float><<<blocks(pairs * 128), 256, 0, STREAM>>>(t.pair, tmp, pairs * 128));
  }
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
// y[q * A + b] += x[(b - b0) * B + q] for rows b in [b0, b0 + cnt): a chunk of transposedBack's swapAdd
__global__ void swapAddRowsK(float* y, const float* x, int A, int B, int C, size_t b0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * B * C) return;
  int c = (int)(t % C); size_t r = t / C; size_t q = r % B, b = b0 + r / B;
  y[(q * A + b) * C + c] += x[t];
}
inline void attentionCore(const half* xn, int Bt, int n, int C, const std::string& A, int blk, const float* keyMask,
                          const half* bias, float* residual, bool transposedBack) {
  size_t rows = (size_t)Bt * n;
  AttnW w = attnWeights(A, blk, C);
  int Wp = w.H * w.Dp, stride = (n + 7) / 8 * 8;
  if (!bias) bias = zeroBias(w.H, n, stride);
  // where the q/k/v/gate and the output of every row would not fit with room to spare, a chunk of rows at
  // a time (the flash kernel's row offset indexes only the key mask)
  size_t need = (rows + 128) * 4 * Wp * 2 + rows * Wp * 2 + (transposedBack ? rows * C * 4 : 0);
  size_t R = !AF2_TIGHT || roomFor(need, { "fatt.qkvg", "fatt.o", "fatt.tmp" }) ? Bt
           : std::max<size_t>(1, std::min<size_t>(Bt, AF2_CHUNK / ((size_t)n * 4 * Wp)));
  half* qkvg = scratch<half>("fatt.qkvg", (R * n + 128) * 4 * Wp);
  half* o = scratch<half>("fatt.o", R * n * Wp);
  float* tmp = transposedBack ? scratch<float>("fatt.tmp", R * n * C) : nullptr;
  for (size_t b0 = 0; b0 < (size_t)Bt; b0 += R) {
    size_t bc = std::min(R, (size_t)Bt - b0), rc = bc * n;
    ltGemm(xn + b0 * n * C, w.qkvg, qkvg, true, rc, C, 4 * Wp, w.qkvgBias, false, 0.f);
    flashGrid<half>(qkvg, bias, stride, keyMask, o, n, w.H, w.Dp, b0, bc, false, 1.f / sqrtf((float)w.D));
    if (!transposedBack) {
      ltGemm(o, w.out, residual + b0 * n * C, false, rc, Wp, C, P(A + "/output_b", blk), false, 1.f);
    } else if (R == (size_t)Bt) {
      ltGemm(o, w.out, tmp, false, rows, Wp, C, P(A + "/output_b", blk), false, 0.f);
      swapAddK<<<blocks(rows * C), 256, 0, STREAM>>>(residual, tmp, Bt, n, C);
    } else {
      ltGemm(o, w.out, tmp, false, rc, Wp, C, P(A + "/output_b", blk), false, 0.f);
      swapAddRowsK<<<blocks(rc * C), 256, 0, STREAM>>>(residual, tmp, Bt, n, C, b0, bc);
    }
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
  // no bias (the column attention): the grid kernel takes none and loads no zeros; any other form wants them
  if (!bias && !(FLASH_2R && w.Dp == 32 && (!flashRegStaged() || flash2R1Takes()))) bias = zeroBias(w.H, n, stride);
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

// the bias from pair rows [r0, r0 + cnt) (biasFromProjK over a chunk)
__global__ void biasFromProjRowsK(const float* proj, half* out, int L, int H, int stride, bool transposed, size_t r0,
                                  size_t cnt, const float* pairMask = nullptr, bool headMajor = false) {   // proj [cnt][H], or [H][cnt]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * H) return;
  int h = (int)(t % H); size_t ij = r0 + t / H; size_t i = ij / L, j = ij % L;
  float v = proj[headMajor ? (size_t)h * cnt + t / H : t] + (pairMask ? 1e9f * (pairMask[ij] - 1.f) : 0.f);
  if (transposed) { size_t x = i; i = j; j = x; }
  out[((size_t)h * L + i) * stride + j] = __float2half(fmaxf(v * LOG2E, -6e4f));
}
// [C][H] f32 -> [C][16] f16, zero columns past H (gridInK's bias projection reads 16)
__global__ void padHeadsK(const float* w, half* out, int C, int H) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= C * 16) return;
  int k = t / 16, c = t % 16;
  out[t] = __float2half(c < H ? w[k * H + c] : 0.f);
}
inline void msaRowAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C, int H, int D,
                            const float* msaMask) {
  int L = t.L; size_t pairs = (size_t)L * L, rows = (size_t)rowsN * L;
  std::string R = S + "msa_row_attention_with_pair_bias";
  if (FAST) {
    // the pair bias straight from the pair: cuda/af3's lnHeadsK (LN + the projection to 16 columns, H of
    // them live) in chunks, then laid out with the pair mask - no LN'd copy of the pair, no GEMM
    const half* bias;
    {
      if (H > 16) { fprintf(stderr, "msa row attention: %d heads, the fused pair bias takes 16\n", H); exit(1); }
      int stride = (L + 7) / 8 * 8;
      static std::map<const float*, half*> wbCache;
      const float* wf = P(R + "/feat_2d_weights", blk);
      auto it = wbCache.find(wf);
      if (it == wbCache.end()) {
        half* h = wpool<half>((size_t)128 * 16);
        padHeadsK<<<blocks((size_t)128 * 16), 256, 0, STREAM>>>(wf, h, 128, H);
        it = wbCache.emplace(wf, h).first;
      }
      half* b = scratch<half>("fbias.bias", (size_t)H * L * stride);
      if (stride != L) CK(cudaMemsetAsync(b, 0, (size_t)H * L * stride * 2, STREAM));
      size_t per = std::max<size_t>(1, std::min(pairs, AF2_CHUNK / 16));
      float* raw = scratch<float>("fbias.raw16", per * 16);
      const float *sc = P(R + "/feat_2d_norm/scale", blk), *of = P(R + "/feat_2d_norm/offset", blk);
      for (size_t r0 = 0; r0 < pairs; r0 += per) {
        size_t r = std::min(per, pairs - r0);
        lnHeadsRaw<16>(t.pair + r0 * 128, sc, of, it->second, raw, r);
        biasFromProjRowsK<<<blocks(r * H), 256, 0, STREAM>>>(raw, b, L, H, stride, false, r0, r,
                                                             t.pairOnes ? nullptr : t.pairMask, true);
      }
      bias = b;
    }
    half* xn = scratch<half>("frow.xn", rows * C);
    layerNormH(msa, xn, rows, C, R + "/query_norm", blk);
    bool ones = msa == t.msa ? t.msaOnes : t.extraOnes;
    attentionCore(xn, rowsN, L, C, R + "/attention", blk, ones ? nullptr : msaMask, bias, msa, false);
    return;
  }
  needF32Pair("af2 msa row attention (unfused)");
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
  // at 128 channels (the pair stacks) cuda/af3's fused transition, its ReLU-with-bias form: the LayerNorm'd
  // rows and the widened ones never written
  if (FAST && fusedTransitionRaw<true>(x, rows, C, I, P(T + "/input_layer_norm/scale", blk), P(T + "/input_layer_norm/offset", blk),
                                       PH(T + "/transition1/weights", blk), PH(T + "/transition2/weights", blk),
                                       P(T + "/transition1/bias", blk), P(T + "/transition2/bias", blk)))
    return;
  needF32Pair("af2 transition (unfused)");     // (PAIR16 is set only around the pair's own updates)
  if (FAST) {
    // in row chunks of ~128 MB of the widened rows (the whole widened tensor was 1.26 GB of a pair track
    // at 783 residues; a chunk of 2^15+ rows keeps the GEMMs as fast)
    // (at 256 channels - the MSA stacks - the LayerNorm and the widening as one kernel was measured and is
    // slower: cuda/af3's transitionUpK in a ReLU-with-bias form, writing these augmented rows, 182.8 ms
    // over 5CAJ's 576 calls against 117.8 for the LN plus this GEMM it replaced, and 181.6 at two tiles a
    // warp. At 256 channels its m16-a-warp MMAs read each weight fragment for one MMA; cuBLAS does better.)
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
// lt [s][i][c] -> [i][c][s], the left operand of the shallow form's contraction
__global__ void opmLeftToICS(const half* lt, half* out, int S, int L, int O, float scale = 1.f) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)S * L * O) return;
  int s = (int)(t % S); size_t r = t / S; int c = (int)(r % O); int i = (int)(r / O);
  out[t] = scale == 1.f ? lt[((size_t)s * L + i) * O + c] : __float2half(__half2float(lt[((size_t)s * L + i) * O + c]) * scale);
}
// the output bias over a pair row's L positions, scaled: [L][C] of bias[c] * scale
__global__ void tileBiasK(const float* bias, float* out, int L, int C, float scale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)L * C) out[t] = bias[t % C] * scale;
}
// The outer product mean's SHALLOW form, for an alignment of few rows: the output projection folded into the
// right operand first - T[c][s][j][f] = sum_e R[s][j][e] W[c O + e][f], a batched GEMM over c (W's rows for one
// c are contiguous) - then Y[i][j][f] = sum_{c,s} L[s][i][c] T[c][s][j][f], one GEMM with K = O x S. Its work
// is n^2 S O 128 against the standard form's n^2 O^2 (S + 128): a single sequence is ~30x less, and the forms
// cross in FLOPs near S = 43 - but measured on an A100 at 500 residues, whole fold, the shallow form is faster
// to 128 rows (1991 -> 1657 ms at 4, 2071 -> 1784 at 32, 2443 -> 2317 at 128) and slower from 256 (2912 against
// 3063): the standard form's permute and its O^2-wide product are memory, not arithmetic. LOCALFOLD_OPM_SHALLOW
// sets the deepest alignment it takes (0: never).
inline const int OPM_SHALLOW = getenv("LOCALFOLD_OPM_SHALLOW") ? atoi(getenv("LOCALFOLD_OPM_SHALLOW")) : 128;
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
    if (rowsN <= OPM_SHALLOW) {
      const half* W = PH(Op + "/output_w", blk);           // [O*O][128] f16
      half* T = scratch<half>("fopm.T", (size_t)O * rows * 128);
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, 128, (int)rows, O, &one, W, CUDA_R_16F, 128, (long long)O * 128,
                                    rt, CUDA_R_16F, O, 0, &zero, T, CUDA_R_16F, 128, (long long)rows * 128, O,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      half* Lt = scratch<half>("fopm.Lt", rows * O);
      // with every MSA row live the norm is rowsN at every pair, so (Y + b) / (1e-3 + norm) folds into the
      // GEMM: L scaled by the constant, the bias tiled and scaled, accumulated straight into the pair (no
      // f32 Y written and read back, no opmAddK: 1.06 s of a 2,088-residue fold)
      const float inv = ones ? 1.f / (1e-3f + (float)rowsN) : 1.f;
      opmLeftToICS<<<blocks(rows * O), 256, 0, STREAM>>>(lt, Lt, rowsN, L, O, inv);
      int K = O * rowsN;
      if (ones) {
        float* bt = scratch<float>("fopm.btile", (size_t)L * 128);
        tileBiasK<<<blocks((size_t)L * 128), 256, 0, STREAM>>>(P(Op + "/output_b", blk), bt, L, 128, inv);
        int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / ((size_t)L * 128)));
        for (int i0 = 0; i0 < L; i0 += Bi) {
          int bi = std::min(Bi, L - i0);
          if (!PAIR16) ltGemm(Lt + (size_t)i0 * K, T, t.pair + (size_t)i0 * L * 128, false, bi, K, L * 128, bt, false, 1.f);
          else {        // (into an f32 block of rows, then added into the bf16 pair: an f16 product there moved 6MRR
                        // from a single sequence 1.898 -> 2.001 A, pLDDT 84.6 -> 81.1)
            float* Yb = scratch<float>("fopm.Y", (size_t)Bi * L * 128);
            ltGemm(Lt + (size_t)i0 * K, T, Yb, false, bi, K, L * 128, bt, false, 0.f);
            addIntoPairK<__nv_bfloat16, float><<<blocks((size_t)bi * L * 128), 256, 0, STREAM>>>(
              reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(t.pair) + (size_t)i0 * L * 128), Yb, (size_t)bi * L * 128);
          }
        }
        return;
      }
      int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / ((size_t)L * 128)));
      float* Y = scratch<float>("fopm.Y", (size_t)Bi * L * 128);
      for (int i0 = 0; i0 < L; i0 += Bi) {
        int bi = std::min(Bi, L - i0);
        CB(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, L * 128, bi, K, &one, T, CUDA_R_16F, L * 128, Lt + (size_t)i0 * K,
                        CUDA_R_16F, K, &zero, Y, CUDA_R_32F, L * 128, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        WITH_PAIR_T(opmAddK<PT><<<blocks((size_t)bi * L * 128), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L, 128, false));
      }
      return;
    }
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
      WITH_PAIR_T(opmAddK<PT><<<blocks((size_t)bi * L * 128), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L, 128, false));
    }
    return;
  }
  needF32Pair("af2 outer product mean (unfused)");
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
    opmPermuteK<float><<<blocks((size_t)bi * per), 256, 0, STREAM>>>(Pm, X, bi, L, O);
    gemm(X, P(Op + "/output_w", blk), Y, (size_t)bi * L, O * O, 128);
    opmAddK<<<blocks((size_t)bi * L * 128), 256, 0, STREAM>>>(t.pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L, 128, false);
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
#include "../../af3/src/triblocked.cuh"
// one operand's [projection | gate] (C, 2C) f16 and its 2C bias, side 0 = a (the first C columns), 1 = b
__global__ void operand2K(const float* proj, const float* gate, const float* pb, const float* gb, half* w, float* bias,
                          int C, int side) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < (size_t)C * 2 * C) {
    int k = (int)(t / (2 * C)), o = (int)(t % (2 * C));
    w[t] = __float2half(o < C ? proj[(size_t)k * 2 * C + side * C + o] : gate[(size_t)k * 2 * C + side * C + o - C]);
  }
  if (t < (size_t)2 * C) bias[t] = (int)t < C ? pb[side * C + t] : gb[side * C + t - C];
}
inline void triangleBlocked2(float* pair, const float* mask, int L, int C, const std::string& T, int blk, bool outgoing) {
  struct Op { half* w; float* b; };
  static std::map<std::tuple<std::string, int, int>, Op> ops;
  auto opOf = [&](int side) {
    auto key = std::make_tuple(T, blk, side);
    auto it = ops.find(key);
    if (it != ops.end()) return it->second;
    Op o{ wpool<half>((size_t)C * 2 * C), wpool<float>(2 * C) };
    operand2K<<<blocks((size_t)C * 2 * C), 256, 0, STREAM>>>(P(T + "/projection/weights", blk), P(T + "/gate/weights", blk),
      P(T + "/projection/bias", blk), P(T + "/gate/bias", blk), o.w, o.b, C, side);
    return ops[key] = o;
  };
  Op a = opOf(0), b = opOf(1);
  TriBlockedW w{ P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), a.w, b.w, a.b, b.b,
                 P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                 PH(T + "/output_projection/weights", blk), P(T + "/output_projection/bias", blk),
                 PH(T + "/gating_linear/weights", blk), P(T + "/gating_linear/bias", blk) };
  triangleBlockedHalf(pair, mask, L, C, w, outgoing, AF2_CHUNK);
}
// the triangle's whole-form buffers (both precisions'): what a pass already holds counts toward its room
#define TRI_FAST_HELD { "ftri.a", "ftri.b", "ftri.t2", "ftri.prod", "ftri.xn", "ftri.og", "ftri.out", "ftri.abf", "ftri.bbf", "ftri.pbf" }
inline void triangleMultiplication(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk,
                                   bool outgoing) {
  size_t pairs = (size_t)L * L;
  std::string T = S + (outgoing ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming");
  // on a card short of room, in blocks of the output, where the whole form would not fit with room to spare
  if (FAST && shortPair(pairs, C)) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    if (!roomFor(5 * plane * C * 2, TRI_FAST_HELD)) {
      releaseScratch({ "ftri." });
      needF32Pair("af2 blocked triangle");
      triangleBlocked2(pair, pairMask, L, C, T, blk, outgoing);
      return;
    }
  }
  // a, b and the contraction's product in bf16 (f32's range at half the bytes - f16's range is what overflows),
  // as cuda/af3 has them and as AlphaFold 2's own trunk runs (global_config.bfloat16): the output kernel then
  // stages half the product and takes its persistent form. AF2_TRI_F32=1 keeps the f32 product.
  static const bool triBf16 = !getenv("AF2_TRI_F32");
  if (FAST && FUSED_TRIANGLE && C == 128 && triBf16 && triFusedFits<__nv_bfloat16>()) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    TriFused w = triFusedWeights(T, blk, C);
    __nv_bfloat16* a = scratch<__nv_bfloat16>("ftri.abf", plane * C); __nv_bfloat16* b = scratch<__nv_bfloat16>("ftri.bbf", plane * C);
    half* t2 = scratch<half>("ftri.t2", plane * C);
    triInRaw<__nv_bfloat16, true>(pair, pairMask, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk),
                                  w.wpg, PH(T + "/gating_linear/weights", blk), w.bias, a, b, t2, L, Lp, plane);
    __nv_bfloat16* prod = scratch<__nv_bfloat16>("ftri.pbf", plane * C);
    const float one = 1.f, zero = 0.f;
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16BF, Lp, plane, a, CUDA_R_16BF, Lp,
                                    plane, &zero, prod, CUDA_R_16BF, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16BF, Lp, plane, b, CUDA_R_16BF, Lp,
                                    plane, &zero, prod, CUDA_R_16BF, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    triOutRaw<__nv_bfloat16, true>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                                   PH(T + "/output_projection/weights", blk), P(T + "/output_projection/bias", blk), t2, pair, L, Lp,
                                   plane);
    return;
  }
  if (FAST && FUSED_TRIANGLE && C == 128 && triFusedFits<float>()) {
    // cuda/af3's fused kernels (fast.cuh, triFusedWeights): LN, the gated projections and the gating
    // linear in one kernel, AF2's contraction, then the centre norm, the output projection and the gate
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    TriFused w = triFusedWeights(T, blk, C);
    half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
    half* t2 = scratch<half>("ftri.t2", plane * C);
    triInRaw<half, true>(pair, pairMask, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), w.wpg,
                         PH(T + "/gating_linear/weights", blk), w.bias, a, b, t2, L, Lp, plane);   // (writes the padding)
    float* prod = scratch<float>("ftri.prod", plane * C);
    const float one = 1.f, zero = 0.f;
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    triOutRaw<float, true>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                           PH(T + "/output_projection/weights", blk), P(T + "/output_projection/bias", blk), t2, pair, L, Lp, plane);
    return;
  }
  // a T4, where neither fused form above fits (the output kernel holds the whole 128 x 128 weight, 71 KB):
  // cuda/af3's streaming triangle kernels at 128 channels, biased, the weight 16 columns a stage (~35 KB),
  // the contraction f16 into f32 (no bf16 MMA there)
  if (FAST && FUSED_TRIANGLE && C == 128 && L >= 80 &&
      fitsSmem(std::max(triIn256Smem<half>(128, 8), triangleOutSmem<128, 4, float, 32, float>()))) {
    int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
    TriFused w = triFusedWeights(T, blk, C);
    half* wt = scratch<half>("ftri.wt", triInTileHalves(C));
    tileTriIn(w.wpg, PH(T + "/gating_linear/weights", blk), C, 16, wt);
    half* a = scratch<half>("ftri.a", plane * C); half* b = scratch<half>("ftri.b", plane * C);
    half* t2 = scratch<half>("ftri.t2", plane * C);
    constexpr int WI = 8;
    WITH_PAIR_T(
      static bool attr = false;
      if (!attr) { smemAttr((triIn256K<128, WI, half, 1, true, PT>), (int)triIn256Smem<half>(128, WI)); attr = true; }
      triIn256K<128, WI, half, 1, true, PT><<<(unsigned)((plane + 16 * WI - 1) / (16 * WI)), 32 * WI, triIn256Smem<half>(128, WI), STREAM>>>(
        pair, pairMask, P(T + "/left_norm_input/scale", blk), P(T + "/left_norm_input/offset", blk), wt, a, b, t2, L, Lp, plane, w.bias));
    float* prod = scratch<float>("ftri.prod", plane * C);
    const float one = 1.f, zero = 0.f;
    if (outgoing)
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, Lp, Lp, Lp, &one, b, CUDA_R_16F, Lp, plane, a, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
      CB(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_T, Lp, Lp, Lp, &one, a, CUDA_R_16F, Lp, plane, b, CUDA_R_16F, Lp,
                                    plane, &zero, prod, CUDA_R_32F, Lp, plane, C, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    triangleOutRun<128, 4, float>(prod, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                                  PH(T + "/output_projection/weights", blk), t2, pair, L, Lp, P(T + "/output_projection/bias", blk));
    return;
  }
  needF32Pair("af2 triangle multiplication (unfused)");
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
    centerNormHK<<<(unsigned)((pairs + 31) / 32), 256, (size_t)C * 33 * 4, STREAM>>>(prod, cn, pairs, C,
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
// pair columns [b0, b0 + cnt) as rows: out[(b - b0) * L + q] = pair[q * L + b]
__global__ void gatherColumnsK(const float* pair, float* out, int L, int C, size_t b0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * L * C) return;
  int c = (int)(t % C); size_t r = t / C; size_t q = r % L, b = b0 + r / L;
  out[t] = pair[(q * L + b) * C + c];
}
// pair[q * L + b] += x[(b - b0) * L + q], the columns [b0, b0 + cnt)
__global__ void addColumnsK(float* pair, const float* x, int L, int C, size_t b0, size_t cnt) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * L * C) return;
  int c = (int)(t % C); size_t r = t / C; size_t q = r % L, b = b0 + r / L;
  pair[(q * L + b) * C + c] += x[t];
}
// The triangle attention on a card short of room (--fast), in chunks of attention rows: the bias from the
// pair in chunks first, then per chunk its rows' LayerNorm, q/k/v/gate, the flash kernel and the output
// projection added back - never the LayerNorm'd pair, the q/k/v/gate or the output whole, nor the
// ending node's transposed copy of the pair (its rows are gathered from the columns, its mask read
// transposed). In place it is safe: a row (column) chunk reads and writes only its own rows (columns).
inline void triangleAttentionChunked(float* pair, const float* pairMask, int L, int C, const std::string& A, int blk,
                                     bool starting, bool pairOnes) {
  size_t pairs = (size_t)L * L;
  AttnW w = attnWeights(A + "/attention", blk, C);
  int H = w.H, Wp = w.H * w.Dp, stride = (L + 7) / 8 * 8;
  static std::map<const float*, half*> whCache;
  const float* wf = P(A + "/feat_2d_weights", blk);
  auto it = whCache.find(wf);
  if (it == whCache.end()) {
    half* h = wpool<half>((size_t)C * H);
    toHalfK<<<blocks((size_t)C * H), 256, 0, STREAM>>>(wf, h, (size_t)C * H);
    it = whCache.emplace(wf, h).first;
  }
  half* bias = scratch<half>("fbias.bias", (size_t)H * L * stride);
  if (stride != L) CK(cudaMemsetAsync(bias, 0, (size_t)H * L * stride * 2, STREAM));
  size_t per = std::max<size_t>(1, std::min(pairs, AF2_CHUNK / C));
  half* lnc = scratch<half>("ftatt.lnc", per * C);
  float* projc = scratch<float>("fbias.projc", per * H);
  for (size_t r0 = 0; r0 < pairs; r0 += per) {
    size_t r = std::min(per, pairs - r0);
    layerNormH(pair + r0 * C, lnc, r, C, A + "/query_norm", blk);
    ltGemm(lnc, it->second, projc, false, r, C, H, nullptr, false, 0.f);
    biasFromProjRowsK<<<blocks(r * H), 256, 0, STREAM>>>(projc, bias, L, H, stride, !starting, r0, r);
  }
  // attention rows a chunk: a row is L positions, and its q/k/v/gate 4 Wp halves each
  size_t R = std::max<size_t>(1, std::min<size_t>(L, AF2_CHUNK / ((size_t)L * 4 * Wp)));
  half* xn = scratch<half>("ftatt.xn", R * L * C);
  float* gath = starting ? nullptr : scratch<float>("ftatt.cols", R * L * C);
  half* qkvg = scratch<half>("fatt.qkvg", (R * L + 128) * 4 * Wp);
  half* o = scratch<half>("fatt.o", R * L * Wp);
  float* tmp = starting ? nullptr : scratch<float>("fatt.tmp", R * L * C);
  for (size_t b0 = 0; b0 < (size_t)L; b0 += R) {
    size_t bc = std::min(R, (size_t)L - b0), rows = bc * L;
    const float* src = pair + b0 * L * C;
    if (!starting) { gatherColumnsK<<<blocks(rows * C), 256, 0, STREAM>>>(pair, gath, L, C, b0, bc); src = gath; }
    layerNormH(src, xn, rows, C, A + "/query_norm", blk);
    ltGemm(xn, w.qkvg, qkvg, true, rows, C, 4 * Wp, w.qkvgBias, false, 0.f);
    flashGrid<half>(qkvg, bias, stride, pairOnes ? nullptr : pairMask, o, L, H, w.Dp, b0, bc, !starting,
                    1.f / sqrtf((float)w.D));
    if (starting) {
      ltGemm(o, w.out, pair + b0 * L * C, false, rows, Wp, C, P(A + "/attention/output_b", blk), false, 1.f);
    } else {
      ltGemm(o, w.out, tmp, false, rows, Wp, C, P(A + "/attention/output_b", blk), false, 0.f);
      addColumnsK<<<blocks(rows * C), 256, 0, STREAM>>>(pair, tmp, L, C, b0, bc);
    }
  }
}
inline bool FUSED_GRID_AF2 = !getenv("LOCALFOLD_AF2_UNFUSED_GRID");
// The triangle attention on cuda/af3's fused grid kernels: LN + q/k/v/gate (+ the gate bias) + the pair
// bias projection in one kernel reading the ending node's rows transposed where they lie, the flash
// kernel, and the output projection (+ its bias) added into the pair - transposed in place for the ending
// node, so no gather, no scatter and no LN'd copy of the pair. One pass where the q/k/v/gate are a 32nd of
// the card (the bias written by the same kernel); else the bias first (LN + projection, chunked), then the
// rows a chunk at a time.
inline void triangleAttentionGrid(float* pair, const float* pairMask, int L, int C, const std::string& A, int blk,
                                  bool starting, bool pairOnes) {
  size_t pairs = (size_t)L * L;
  AttnW w = attnWeights(A + "/attention", blk, C);
  int H = w.H, Wp = H * w.Dp, stride = (L + 7) / 8 * 8;
  const bool tr = !starting;
  const float *lnS = P(A + "/query_norm/scale", blk), *lnO = P(A + "/query_norm/offset", blk);
  const float* ob = P(A + "/attention/output_b", blk);
  static std::map<const float*, half*> wbCache;
  const float* wf = P(A + "/feat_2d_weights", blk);
  auto it = wbCache.find(wf);
  if (it == wbCache.end()) {
    half* h = wpool<half>((size_t)C * 16);
    padHeadsK<<<blocks((size_t)C * 16), 256, 0, STREAM>>>(wf, h, C, H);
    it = wbCache.emplace(wf, h).first;
  }
  const half* Wb = it->second;
  half* bias = scratch<half>("fbias.bias", (size_t)H * L * stride);
  CK(cudaMemsetAsync(bias, 0, (size_t)H * L * stride * 2, STREAM));                // the padding columns
  const float* mask = pairOnes ? nullptr : pairMask;
  float scale = 1.f / sqrtf((float)w.D);
  // every row in one pass whenever the card has the room for its q/k/v/gate and output (what they already
  // hold counted, so each pass decides alike) - not below a fixed 32nd of the card: at 1,566 residues
  // 17.43 -> 16.43 s, at 2,088 35.80 -> 34.19 s (peak 7.9 -> 10.7 and 12.0 -> 17.1 GB on 40 GB); the
  // chunked form only where the memory is not there (LOCALFOLD_BIG=1 forces it, roomFor).
  // 🔴 The room asked for includes the triangle multiplication's whole form (five planes): these buffers
  // stay held while it runs, and starved of them it blocks - at 2,610 residues the one-pass grid took the
  // memory and the fold went 63.7 -> 104.4 s
  size_t triPlane = (size_t)((L + 7) / 8 * 8) * ((L + 7) / 8 * 8);
  if (roomFor(((pairs + 128) * 4 * Wp + pairs * Wp) * 2 + 5 * triPlane * C * 2,
              {"fatt.qkvg", "fatt.o", "ftri.a", "ftri.b", "ftri.t2", "ftri.prod", "ftri.xn", "ftri.og", "ftri.out", "ftri.abf", "ftri.bbf", "ftri.pbf"})) {
    half* qkvg = scratch<half>("fatt.qkvg", (pairs + 128) * 4 * Wp);
    gridInRaw(pair, lnS, lnO, w.qkvg, w.qkvgBias, qkvg, L, 0, pairs, tr, Wb, bias, H, stride, tr);
    half* o = scratch<half>("fatt.o", pairs * Wp);
    flashGrid<half>(qkvg, bias, stride, mask, o, L, H, w.Dp, 0, L, tr, scale);
    // (the row direction's GEMM accumulates into an f32 pair; a bf16 pair takes gridOutK in both directions -
    // an f16 product and an add pass were 6-10 ms slower a 494-residue fold)
    if (!tr && !PAIR16) ltGemm(o, w.out, pair, false, pairs, Wp, C, ob, false, 1.f);
    else gridOutRaw(o, w.out, ob, pair, L, 0, pairs, tr);
    return;
  }
  {   // the bias, all of it before any row is attended (and before any is written): LN + the 16-column
      // projection in one kernel (lnHeadsK), then laid out head-major
    size_t per = std::max<size_t>(1, std::min(pairs, AF2_CHUNK / 16));
    float* raw = scratch<float>("fbias.raw16", per * 16);
    for (size_t r0 = 0; r0 < pairs; r0 += per) {
      size_t r = std::min(per, pairs - r0);
      lnHeadsRaw<16>(pair + r0 * C, lnS, lnO, Wb, raw, r);
      biasFromProjRowsK<<<blocks(r * H), 256, 0, STREAM>>>(raw, bias, L, H, stride, tr, r0, r, nullptr, true);   // lnHeadsK writes head-major
    }
  }
  size_t R = std::max<size_t>(1, std::min<size_t>(L, AF2_CHUNK / ((size_t)L * 4 * Wp)));
  half* qkvg = scratch<half>("fatt.qkvg", (R * L + 128) * 4 * Wp);
  half* o = scratch<half>("fatt.o", R * L * Wp);
  for (size_t b0 = 0; b0 < (size_t)L; b0 += R) {
    size_t bc = std::min(R, (size_t)L - b0), rows = bc * L;
    gridInRaw(pair, lnS, lnO, w.qkvg, w.qkvgBias, qkvg, L, b0 * L, rows, tr);
    flashGrid<half>(qkvg, bias, stride, mask, o, L, H, w.Dp, b0, bc, tr, scale);
    if (!tr && !PAIR16) ltGemm(o, w.out, pair + b0 * L * C, false, rows, Wp, C, ob, false, 1.f);
    else gridOutRaw(o, w.out, ob, pair, L, b0 * L, rows, tr);
  }
}
inline void triangleAttention(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk,
                              bool starting, bool pairOnes = false) {
  size_t pairs = (size_t)L * L;
  std::string A = S + (starting ? "triangle_attention_starting_node" : "triangle_attention_ending_node");
  int H = (int)dimW(A + "/attention/query_w", 2), D = (int)dimW(A + "/attention/query_w", 3);
  if (FAST && FUSED_GRID_AF2 && C == 128 && H * (D < 16 ? 16 : D) == 128 && H <= 16 && gridFusedFits()) {
    triangleAttentionGrid(pair, pairMask, L, C, A, blk, starting, pairOnes);
    return;
  }
  needF32Pair("af2 triangle attention (unfused)");
  if (FAST && shortPair(pairs, C)) { triangleAttentionChunked(pair, pairMask, L, C, A, blk, starting, pairOnes); return; }
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
  if (t.opmFirst) { PAIR16 = AF2_P16; outerProductMean(t, S, blk, msa, rowsN, C, mask); PAIR16 = false; }
  PAIR16 = AF2_P16; msaRowAttention(t, S, blk, msa, rowsN, C, H, D, mask); PAIR16 = false;
  if (extraStack) msaColumnGlobalAttention(t, S, blk, msa, rowsN, C, mask);
  else msaColumnAttention(t, S, blk, msa, rowsN, C, H, D, mask);
  transition(msa, (size_t)rowsN * t.L, C, S + "msa_transition", blk);
  PAIR16 = AF2_P16;
  if (!t.opmFirst) outerProductMean(t, S, blk, msa, rowsN, C, mask);
  triangleMultiplication(t.pair, t.pairMask, t.L, 128, S, blk, true);
  triangleMultiplication(t.pair, t.pairMask, t.L, 128, S, blk, false);
  triangleAttention(t.pair, t.pairMask, t.L, 128, S, blk, true, t.pairOnes);
  triangleAttention(t.pair, t.pairMask, t.L, 128, S, blk, false, t.pairOnes);
  transition(t.pair, (size_t)t.L * t.L, 128, S + "pair_transition", blk);
  PAIR16 = false;
}
// whether every update both stacks run takes a bf16 pair (each route below has a bf16 form; the f32-only ones
// refuse through needF32Pair): the fused triangle (either fused form, or a T4's streaming one), the fused grid
// attention, the fused transition, and not a card short of room (its blocked and in-place forms are f32)
inline bool af2Pair16Ok(int L) {
  // (Ampere on: a T4 has no f32 -> bf16 conversion instruction, and there the bf16 pair was 1-3% SLOWER - its
  // biased float-tile triangle output 1237 -> 1520 ms a 494-residue fold, gridOutK in both directions 563 -> 780)
  static const int major = [] { int d, m; CK(cudaGetDevice(&d)); CK(cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMajor, d)); return m; }();
  if (major < 8 || !FAST || getenv("LOCALFOLD_PAIR_F32") || shortPair((size_t)L * L, 128) || !FUSED_TRIANGLE || !FUSED_GRID_AF2) return false;
  bool tri = triFusedFits<__nv_bfloat16>() || triFusedFits<float>() ||
             (L >= 80 && fitsSmem(std::max(triIn256Smem<half>(128, 8), triangleOutSmem<128, 4, float, 32, float>())));
  if (!tri || !gridFusedFits()) return false;
  for (const char* S : { "evoformer/extra_msa_stack/", "evoformer/evoformer_iteration/" }) {
    for (const char* A : { "triangle_attention_starting_node", "triangle_attention_ending_node" }) {
      std::string q = std::string(S) + A + "/attention/query_w";
      int H = (int)dimW(q, 2), D = (int)dimW(q, 3);
      if (H * (D < 16 ? 16 : D) != 128 || H > 16) return false;
    }
    int I = (int)dimW(std::string(S) + "pair_transition/transition1/weights", 2);
    if (!fusedTransitionFits((size_t)L * L, 128, I)) return false;
  }
  return true;
}
