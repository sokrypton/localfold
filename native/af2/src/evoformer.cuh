// The input embedder and the Evoformer block - af3-any-model's AF2 multimer graph
// (alphafold3/af2/model/modules.py: EmbeddingsAndEvoformer, EvoformerIteration), float32.
#pragma once
#include "ops.cuh"

struct Trunk {
  int L, N, E;                  // residues, MSA rows, extra MSA rows
  float* msa;                   // [N, L, 256]
  float* extra;                 // [E, L, 64]
  float* pair;                  // [L, L, 128]
  const float* msaMask;         // [N, L]
  const float* extraMask;       // [E, L]
  float* pairMask;              // [L, L]
  bool opmFirst;
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
  t.msaMask = W(f + "msa_mask");
  t.extraMask = W(f + "extra_msa_mask");
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
  float* tr = scratch<float>("col.tr", rows * C); float* xn = scratch<float>("col.xn", rows * C);
  float* mt = scratch<float>("col.mask", rows);
  swap01(msa, tr, rowsN, L, C);                 // [L, N, C]
  swap01(msaMask, mt, rowsN, L, 1);
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
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(globalAttentionK, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024)); attr = true; }
  if (smem > 160 * 1024) { fprintf(stderr, "global attention: %d sequences do not fit one block\n", rowsN); exit(1); }
  globalAttentionK<<<L, 256, smem, STREAM>>>(xn, mt, P(A + "/attention/query_w", blk), P(A + "/attention/key_w", blk),
                                             P(A + "/attention/value_w", blk), avg, rowsN, C, H, D);
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
  float* xn = scratch<float>("tr.xn", rows * C);
  float* mid = scratch<float>("tr.mid", rows * C * 4);
  float* out = scratch<float>("tr.out", rows * C);
  layerNorm(x, xn, rows, C, T + "/input_layer_norm", blk);
  linearB(xn, T + "/transition1", blk, mid, rows, C, 4 * C, true);
  linearB(mid, T + "/transition2", blk, out, rows, 4 * C, C);
  addK2<<<blocks(rows * C), 256, 0, STREAM>>>(x, out, rows * C);
}
inline void outerProductMean(Trunk& t, const std::string& S, int blk, const float* msa, int rowsN, int C,
                             const float* msaMask) {
  int L = t.L; size_t rows = (size_t)rowsN * L; const int O = 32;
  std::string Op = S + "outer_product_mean";
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
inline void triangleMultiplication(Trunk& t, const std::string& S, int blk, bool outgoing) {
  int L = t.L; size_t pairs = (size_t)L * L; const int C = 128;
  std::string T = S + (outgoing ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming");
  float* xn = scratch<float>("tri.xn", pairs * C);
  layerNorm(t.pair, xn, pairs, C, T + "/left_norm_input", blk);
  float* proj = scratch<float>("tri.proj", pairs * 2 * C); float* gate = scratch<float>("tri.gate", pairs * 2 * C);
  linearB(xn, T + "/projection", blk, proj, pairs, C, 2 * C);
  linearB(xn, T + "/gate", blk, gate, pairs, C, 2 * C);
  float* a = scratch<float>("tri.a", pairs * C); float* b = scratch<float>("tri.b", pairs * C);
  triSplitK<<<blocks(pairs * 2 * C), 256, 0, STREAM>>>(proj, gate, t.pairMask, a, b, L, C);
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
  gateMulAddK<<<blocks(pairs * C), 256, 0, STREAM>>>(t.pair, out, g, pairs * C);
}
inline void triangleAttention(Trunk& t, const std::string& S, int blk, bool starting) {
  int L = t.L; size_t pairs = (size_t)L * L; const int C = 128;
  std::string A = S + (starting ? "triangle_attention_starting_node" : "triangle_attention_ending_node");
  int H = (int)dimW(A + "/attention/query_w", 2), D = (int)dimW(A + "/attention/query_w", 3);
  float* x = t.pair; float* mask = t.pairMask;
  float* tr = scratch<float>("tatt.tr", pairs * C); float* mt = scratch<float>("tatt.mask", pairs);
  if (!starting) { swap01(t.pair, tr, L, L, C); swap01(t.pairMask, mt, L, L, 1); x = tr; mask = mt; }
  float* xn = scratch<float>("tatt.xn", pairs * C);
  layerNorm(x, xn, pairs, C, A + "/query_norm", blk);
  float* proj = scratch<float>("tatt.proj", pairs * H);
  gemm(xn, P(A + "/feat_2d_weights", blk), proj, pairs, C, H);
  float* bias = scratch<float>("tatt.bias", pairs * H);
  pairBiasK<<<blocks(pairs * H), 256, 0, STREAM>>>(proj, nullptr, bias, L, H);
  float* out = scratch<float>("tatt.out", pairs * C);
  gatedAttention(xn, L, L, C, A + "/attention", blk, H, D, mask, bias, out);
  if (!starting) { swap01(out, tr, L, L, C); out = tr; }
  addK2<<<blocks(pairs * C), 256, 0, STREAM>>>(t.pair, out, pairs * C);
}

// one Evoformer iteration of a stack: S is "evoformer/evoformer_iteration/" or ".../extra_msa_stack/"
inline void evoformerBlock(Trunk& t, bool extraStack, int blk) {
  const std::string S = extraStack ? "evoformer/extra_msa_stack/" : "evoformer/evoformer_iteration/";
  float* msa = extraStack ? t.extra : t.msa;
  int rowsN = extraStack ? t.E : t.N, C = extraStack ? 64 : 256;
  const float* mask = extraStack ? t.extraMask : t.msaMask;
  std::string R = S + "msa_row_attention_with_pair_bias/attention/query_w";
  int H = (int)dimW(R, 2), D = (int)dimW(R, 3);
  if (t.opmFirst) outerProductMean(t, S, blk, msa, rowsN, C, mask);
  msaRowAttention(t, S, blk, msa, rowsN, C, H, D, mask);
  if (extraStack) msaColumnGlobalAttention(t, S, blk, msa, rowsN, C, mask);
  else msaColumnAttention(t, S, blk, msa, rowsN, C, H, D, mask);
  transition(msa, (size_t)rowsN * t.L, C, S + "msa_transition", blk);
  if (!t.opmFirst) outerProductMean(t, S, blk, msa, rowsN, C, mask);
  triangleMultiplication(t, S, blk, true);
  triangleMultiplication(t, S, blk, false);
  triangleAttention(t, S, blk, true);
  triangleAttention(t, S, blk, false);
  transition(t.pair, (size_t)t.L * t.L, 128, S + "pair_transition", blk);
}
