// The input embedder and the Evoformer block - af3-any-model's AF2 multimer graph (alphafold3/af2/model/modules.py:
// EmbeddingsAndEvoformer, EvoformerIteration); cuda/af2/src/evoformer.cuh is the reading. Every attention runs on the
// core's flash kernel - the MSA's column attention and the triangle's ending node ACROSS the tensor where it lies (its
// strides), so nothing is transposed - and its one q/k/v/gate projection is a GEMM straight into the kernel's layout.
#include "af2.h"
#include <cmath>
#include <map>

void linearB(const float* X, const std::string& w, int block, float* Y, size_t rows, int in, int out, bool relu, float beta) {
  Gemm g{}; g.X = X; g.tx = F32; g.W = PH(w + "/weights", block); g.tw = F16; g.half = true; g.Y = Y; g.rows = rows; g.in = in;
  g.out = out; g.bias = P(w + "/bias", block); g.relu = relu; g.beta = beta; g.label = w.c_str();
  gemm(g);
}
void linearB(const half* X, const std::string& w, int block, float* Y, size_t rows, int in, int out, bool relu, float beta) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = PH(w + "/weights", block); g.tw = F16; g.Y = Y; g.rows = rows; g.in = in; g.out = out;
  g.bias = P(w + "/bias", block); g.relu = relu; g.beta = beta; g.label = w.c_str();
  gemm(g);
}
void layerNormW(const float* x, float* y, size_t rows, int C, const std::string& w, int block) {
  layerNorm(x, y, rows, C, P(w + "/scale", block), P(w + "/offset", block));
}
void layerNormW(const float* x, half* y, size_t rows, int C, const std::string& w, int block) {
  layerNorm(x, y, rows, C, P(w + "/scale", block), P(w + "/offset", block));
}
// a linear whose weight and bias are named apart (an attention's output_w / output_b, gating_w / gating_b)
static void linearWB(const half* X, const std::string& w, const std::string& b, int blk, float* Y, size_t rows, int in, int out,
                     float beta) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = PH(w, blk); g.tw = F16; g.Y = Y; g.rows = rows; g.in = in; g.out = out; g.bias = P(b, blk);
  g.beta = beta; g.label = w.c_str();
  gemm(g);
}
// the pair's next LayerNorm where the previous update's GEMM already wrote it (Gemm::lnOut, gemmGatedAddDual's): the
// pair, the norm (its weight's name and block) and the buffer, taken by pairLn of the same; a miss drops it.
// LOCALFOLD_LN_EMIT=0: none emitted
static struct { const float* x = nullptr; std::string norm; half* buf = nullptr; } emitted;
static std::string normKey(const std::string& w, int blk) { return w + "#" + std::to_string(blk); }
static half* takeEmitted(const float* x, const std::string& w, int blk) {
  const bool hit = emitted.x == x && emitted.norm == normKey(w, blk);
  emitted.x = nullptr;
  return hit ? emitted.buf : nullptr;
}
static half* pairLn(const float* pair, size_t pairs, int C, const std::string& w, int blk, const char* scratchName) {
  if (half* b = takeEmitted(pair, w, blk)) return b;
  half* xn = scratch<half>(scratchName, pairs * C);
  layerNormW(pair, xn, pairs, C, w, blk);
  return xn;
}
// where an update's GEMM emits the next norm: whichever of two buffers its own input is not
static half* lnTarget(size_t pairs, int C, const half* inUse) {
  half* a = scratch<half>("ln.a", pairs * C);
  return inUse == a ? scratch<half>("ln.b", pairs * C) : a;
}
static bool allOnes(const float* d, size_t n) {
  const float* h = (const float*)host(d);
  for (size_t i = 0; i < n; ++i) if (h[i] != 1.f) return false;
  return true;
}

// ---------------------------------------------------------------- the embedder
// prev: prevMsaRow [L, 256], prevPair [L, L, 128], prevPos [L, 37, 3]
void embed(Trunk& t, int pass, const float* prevMsaRow, const float* prevPair, const float* prevPos) {
  const std::string E = "evoformer/";
  int L = t.L, N = t.N;
  std::string f = "f" + std::to_string(pass) + "/";
  const int* aatype = Ii("aatype");
  float* tf = scratch<float>("emb.tf", (size_t)L * 21);
  run1d("af2_target_feat", (size_t)L * 21, TargetFeatArgs{aatype, tf, (uint)L, 0});
  // msa = preprocess_1d(target)[None] + preprocess_msa(msa_feat); row 0 += LN(prev msa first row)
  float* p1d = scratch<float>("emb.p1d", (size_t)L * 256);
  linearB(tf, E + "preprocess_1d", -1, p1d, L, 21, 256);
  linearB(In(f + "msa_feat"), E + "preprocess_msa", -1, t.msa, (size_t)N * L, 49, 256);
  run1d("af2_broadcast_rows", (size_t)N * L * 256, BroadcastRowsArgs{t.msa, p1d, (u64)N * L, (uint)L, 256});
  float* prevRowLn = scratch<float>("emb.prevRow", (size_t)L * 256);
  layerNormW(prevMsaRow, prevRowLn, L, 256, E + "prev_msa_first_row_norm");
  add(t.msa, prevRowLn, (size_t)L * 256);
  // pair = left[i] + right[j] + prev_pos_linear(dgram) + LN(prev pair) + relpos
  float* left = scratch<float>("emb.left", (size_t)L * 128); float* right = scratch<float>("emb.right", (size_t)L * 128);
  linearB(tf, E + "left_single", -1, left, L, 21, 128);
  linearB(tf, E + "right_single", -1, right, L, 21, 128);
  size_t pairs = (size_t)L * L;
  run1d("af2_outer_sum", pairs * 128, OuterSumArgs{t.pair, left, right, (uint)L, 128});
  float* dgram = scratch<float>("emb.dgram", pairs * 15);
  run1d("af2_prev_dgram", pairs, PrevDgramArgs{prevPos, aatype, dgram, (uint)L, 0});
  linearB(dgram, E + "prev_pos_linear", -1, t.pair, pairs, 15, 128, false, 1.f);
  {   // + LN(prev pair)
    float* tmp = scratch<float>("emb.tmp", pairs * 128);
    layerNormW(prevPair, tmp, pairs, 128, E + "prev_pair_norm");
    add(t.pair, tmp, pairs * 128);
  }
  float* rel = scratch<float>("emb.rel", pairs * 73);
  run1d("af2_relpos", pairs, RelposArgs{Ii("residue_index"), Ii("asym_id"), Ii("entity_id"), Ii("sym_id"), rel, (uint)L, 0});
  linearB(rel, E + "~_relative_encoding/position_activations", -1, t.pair, pairs, 73, 128, false, 1.f);
  releaseScratch({"emb.rel", "emb.dgram", "emb.tmp"});
  // the extra MSA's activations
  size_t erows = (size_t)t.E * L;
  float* ef = scratch<float>("emb.extraFeat", erows * 25);
  run1d("af2_extra_feat", erows * 25, ExtraFeatArgs{Ii(f + "extra_msa"), In(f + "extra_has_deletion"), In(f + "extra_deletion_value"), ef, erows});
  linearB(ef, E + "extra_msa_activations", -1, t.extra, erows, 25, 64);
  copy(t.msaMask, In(f + "msa_mask"), (size_t)N * L * 4);
  copy(t.extraMask, In(f + "extra_msa_mask"), erows * 4);
  t.msaOnes = allOnes(In(f + "msa_mask"), (size_t)N * L) && t.T == 0;
  t.extraOnes = allOnes(In(f + "extra_msa_mask"), erows);
  t.pairOnes = allOnes(In("seq_mask"), L);
  run1d("af2_pair_mask", pairs, PairMaskArgs{In("seq_mask"), t.pairMask, (uint)L, 0});
}

// ---------------------------------------------------------------- attention
// one attention block's q | k | v | gate as one [C][4W] f16 weight, and its bias (the gate's, zeros elsewhere)
struct AttnW { const half* qkvg; const float* bias; int H, D; };
static AttnW attnWeights(const std::string& A, int blk, int C) {
  int H = (int)dimW(A + "/query_w", blk < 0 ? 1 : 2), D = (int)dimW(A + "/query_w", blk < 0 ? 2 : 3), W = H * D;
  std::string key = A + "#" + std::to_string(blk);
  const half* w = M.derived<half>("qkvg:" + key, (size_t)C * 4 * W, [&](half* out) {
    run1d("af2_qkvg_weight", (size_t)C * 4 * W, QkvgWArgs{PH(A + "/query_w", blk), PH(A + "/key_w", blk), PH(A + "/value_w", blk),
                                                           PH(A + "/gating_w", blk), out, (uint)C, (uint)W});
  });
  const float* b = M.derived<float>("qkvgb:" + key, (size_t)4 * W, [&](float* out) {
    run1d("af2_gate_bias", (size_t)4 * W, GateBiasArgs{P(A + "/gating_b", blk), out, (uint)W, 0});
  });
  return {w, b, H, D};
}
// xn [Bt][n][C] (or the leading axis attended, `across`: xn [n][Bt][C]) -> the residual += attention's output projection
static void attend(const half* xn, int Bt, int n, int C, const std::string& A, int blk, const half* bias, const float* mask,
                   int64_t maskB, int64_t maskK, float* residual, bool across, const std::string& nextNorm = "", int nextBlk = 0) {
  AttnW w = attnWeights(A, blk, C);
  size_t rows = (size_t)Bt * n; int W = w.H * w.D;
  half* qkvg = scratch<half>("att.qkvg", rows * 4 * W);
  Gemm g{}; g.X = xn; g.tx = F16; g.W = w.qkvg; g.tw = F16; g.Y = qkvg; g.ty = F16; g.rows = rows; g.in = C; g.out = 4 * W;
  g.bias = w.bias; g.label = "attention qkvg";
  gemm(g);
  half* o = scratch<half>("att.o", rows * W);
  Attention at{}; at.qkvg = qkvg; at.out = o; at.n = n; at.heads = w.H; at.D = w.D; at.rows = Bt; at.scale = 1.f / sqrtf((float)w.D);
  at.bias = bias; at.biasStride = n; at.mask = mask; at.maskB = maskB; at.maskK = maskK;
  if (across) { at.rowStride = 4 * W; at.posStride = (int64_t)Bt * 4 * W; at.outRowStride = W; at.outPosStride = (int64_t)Bt * W; }
  attention(at);
  if (!nextNorm.empty() && C == 128) {     // (the residual's next norm emitted by the output projection)
    half* lnOut = lnTarget(rows, C, xn);
    Gemm g{}; g.X = o; g.tx = F16; g.W = PH(A + "/output_w", blk); g.tw = F16; g.Y = residual; g.rows = rows; g.in = W; g.out = C;
    g.bias = P(A + "/output_b", blk); g.beta = 1.f; g.label = "attention output, next norm";
    g.lnOut = lnOut; g.lnScale = P(nextNorm + "/scale", nextBlk); g.lnOffset = P(nextNorm + "/offset", nextBlk);
    if (gemm(g)) emitted = {residual, normKey(nextNorm, nextBlk), lnOut};
    return;
  }
  linearWB(o, A + "/output_w", A + "/output_b", blk, residual, rows, W, C, 1.f);
}
// a pair bias [H][L][L] (log2 units, + 1e9 (mask - 1) where a mask is given) from LN(pair) W
static half* pairBias(const float* pair, int L, int C, const std::string& normW, const std::string& projW, int blk, int H,
                      const float* pairMask, bool transposed, half** normed = nullptr) {
  size_t pairs = (size_t)L * L;
  half* pn = pairLn(pair, pairs, C, normW, blk, "bias.pn");
  if (normed) *normed = pn;
  float* proj = scratch<float>("bias.proj", pairs * H);
  Gemm g{}; g.X = pn; g.tx = F16; g.W = PH(projW, blk); g.tw = F16; g.Y = proj; g.rows = pairs; g.in = C; g.out = H; g.label = "pair bias";
  gemm(g);
  half* b = scratch<half>("bias.b", (size_t)H * pairs);
  run1d("af2_bias_layout", pairs * H, BiasLayoutArgs{proj, pairMask, b, (uint)L, (uint)H, transposed ? 1u : 0u, 0});
  return b;
}

// ---------------------------------------------------------------- the block's modules
static void msaRowAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C, const float* msaMask, bool ones) {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string R = S + "msa_row_attention_with_pair_bias";
  int H = (int)dimW(R + "/attention/query_w", 2);
  half* bias = pairBias(t.pair, L, 128, R + "/feat_2d_norm", R + "/feat_2d_weights", blk, H, t.pairOnes ? nullptr : t.pairMask, false);
  half* xn = scratch<half>("row.xn", rows * C);
  layerNormW(msa, xn, rows, C, R + "/query_norm", blk);
  attend(xn, rowsN, L, C, R + "/attention", blk, bias, ones ? nullptr : msaMask, L, 1, msa, false);
}
static void msaColumnAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C, const float* msaMask, bool ones) {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string A = S + "msa_column_attention";
  half* xn = scratch<half>("col.xn", rows * C);
  layerNormW(msa, xn, rows, C, A + "/query_norm", blk);
  // across the sequences: batch row = the column i, positions the sequences s; the key's mask msaMask[s][i]
  attend(xn, L, rowsN, C, A + "/attention", blk, nullptr, ones ? nullptr : msaMask, 1, L, msa, true);
}
static void msaColumnGlobalAttention(Trunk& t, const std::string& S, int blk, float* msa, int rowsN, int C, const float* msaMask) {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string A = S + "msa_column_global_attention";
  int H = (int)dimW(A + "/attention/query_w", 2), D = (int)dimW(A + "/attention/query_w", 3), W = H * D;
  if (H > 8 || D > 16 || C > 256) die("global attention: %d heads of %d over %d channels", H, D, C);
  half* xn = scratch<half>("gcol.xn", rows * C);
  layerNormW(msa, xn, rows, C, A + "/query_norm", blk);
  // every sequence's key and value (shared by the heads): one GEMM against [key_w | value_w]
  std::string key = A + "#" + std::to_string(blk);
  const half* kvw = M.derived<half>("kv:" + key, (size_t)C * 2 * D, [&](half* out) {
    copy2d(out, 2 * D * 2, PH(A + "/attention/key_w", blk), D * 2, D * 2, C);
    copy2d(out + D, 2 * D * 2, PH(A + "/attention/value_w", blk), D * 2, D * 2, C);
  });
  float* kv = scratch<float>("gcol.kv", rows * 2 * D);
  { Gemm g{}; g.X = xn; g.tx = F16; g.W = kvw; g.tw = F16; g.Y = kv; g.rows = rows; g.in = C; g.out = 2 * D; g.label = "global kv"; gemm(g); }
  float* avg = scratch<float>("gcol.avg", (size_t)L * W);
  run("af2_global_attention", Grid{(uint32_t)L, 1, 1}, 256,
      GlobalAttnArgs{xn, kv, msaMask, PH(A + "/attention/query_w", blk), avg, (uint)rowsN, (uint)L, (uint)C, (uint)H, (uint)D, 0});
  float* gate = scratch<float>("gcol.gate", rows * W);
  linearWB(xn, A + "/attention/gating_w", A + "/attention/gating_b", blk, gate, rows, C, W, 0.f);
  half* gated = scratch<half>("gcol.gated", rows * W);
  run1d("af2_global_gate", rows * W, GlobalGateArgs{avg, gate, gated, (uint)rowsN, (uint)L, (uint)W, 0});
  linearWB(gated, A + "/attention/output_w", A + "/attention/output_b", blk, msa, rows, W, C, 1.f);
}
void transition(float* x, size_t rows, int C, const std::string& T, int blk) {
  half* emittedIn = takeEmitted(x, T + "/input_layer_norm", blk);
  int I = (int)dimW(T + "/transition1/weights", blk < 0 ? 1 : 2);
  size_t chunk = std::min(rows, std::max<size_t>(32768, ((size_t)128 << 20) / (2 * (size_t)I)));
  half* xn = scratch<half>("tr.xn", chunk * C);
  half* mid = scratch<half>("tr.mid", chunk * I);
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    const half* in = r0 == 0 && r == rows && emittedIn ? emittedIn : xn;
    if (in == xn) layerNormW(x + r0 * C, xn, r, C, T + "/input_layer_norm", blk);
    Gemm g{}; g.X = in; g.tx = F16; g.W = PH(T + "/transition1/weights", blk); g.tw = F16; g.Y = mid; g.ty = F16;
    g.rows = r; g.in = C; g.out = I; g.bias = P(T + "/transition1/bias", blk); g.relu = true; g.label = "transition1";
    gemm(g);
    linearB(mid, T + "/transition2", blk, x + r0 * C, r, I, C, false, 1.f);
  }
}
// the outer product mean: pair[i][j] += (sum_s l[s][i] (x) r[s][j] @ W + b) / (1e-3 + norm[i][j])
static void outerProductMean(Trunk& t, const std::string& S, int blk, const float* msa, int rowsN, int C, const float* msaMask,
                             bool ones) {
  int L = t.L; size_t rows = (size_t)rowsN * L; const int O = 32;
  std::string Op = S + "outer_product_mean";
  half* xn = scratch<half>("opm.xn", rows * C);
  layerNormW(msa, xn, rows, C, Op + "/layer_norm_input", blk);
  half* lt = scratch<half>("opm.left", rows * O); half* rt = scratch<half>("opm.right", rows * O);
  for (int side = 0; side < 2; ++side) {
    std::string w = Op + (side ? "/right_projection" : "/left_projection");
    Gemm g{}; g.X = xn; g.tx = F16; g.W = PH(w + "/weights", blk); g.tw = F16; g.Y = side ? rt : lt; g.ty = F16; g.rows = rows;
    g.in = C; g.out = O; g.bias = P(w + "/bias", blk); g.label = "opm projection";
    gemm(g);
  }
  if (!ones) {
    run1d("af2_scale_rows_h", rows * O, ScaleRowsHArgs{lt, msaMask, rows, (uint)O, 0});
    run1d("af2_scale_rows_h", rows * O, ScaleRowsHArgs{rt, msaMask, rows, (uint)O, 0});
  }
  float* norm = scratch<float>("opm.norm", (size_t)L * L);
  run1d("af2_mask_norm", (size_t)L * L, MaskNormArgs{msaMask, norm, (uint)rowsN, (uint)L});
  const half* Wout = PH(Op + "/output_w", blk);        // [O * O][128]
  if (rowsN <= 128) {
    // the SHALLOW form for an alignment of few rows: the output projection folded into the right operand first -
    // T[c][(s, j)][f] = sum_e r[s][j][e] W[c O + e][f], a GEMM batched over c - then Y[i][(j, f)] = sum_{c, s}
    // l[s][i][c] T[(c, s)][(j, f)], one GEMM of K = O S: n^2 S O 128 against the standard form's n^2 O^2 (S + 128)
    half* T = scratch<half>("opm.T", (size_t)O * rows * 128);
    { Gemm g{}; g.X = rt; g.tx = F16; g.W = Wout; g.tw = F16; g.sw = (int64_t)O * 128; g.Y = T; g.ty = F16; g.sy = (int64_t)rows * 128;
      g.rows = rows; g.in = O; g.out = 128; g.batch = O; g.accFloat = true; g.label = "opm shallow T"; gemm(g); }
    half* Lt = scratch<half>("opm.Lt", rows * O);
    const float inv = ones ? 1.f / (1e-3f + (float)rowsN) : 1.f;
    run1d("af2_opm_left", rows * O, OpmLeftArgs{lt, Lt, (uint)rowsN, (uint)L, (uint)O, inv});
    int K = O * rowsN;
    int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / ((size_t)L * 128)));
    if (ones) {     // every row live: the norm is rowsN everywhere, folded into the GEMM (L scaled, the bias tiled)
      float* bt = scratch<float>("opm.btile", (size_t)L * 128);
      run1d("af2_tile_bias", (size_t)L * 128, TileBiasArgs{P(Op + "/output_b", blk), bt, (uint)L, 128, inv, 0});
      for (int i0 = 0; i0 < L; i0 += Bi) {
        int bi = std::min(Bi, L - i0);
        Gemm g{}; g.X = Lt + (size_t)i0 * K; g.tx = F16; g.W = T; g.tw = F16; g.Y = t.pair + (size_t)i0 * L * 128; g.beta = 1.f;
        g.rows = bi; g.in = K; g.out = L * 128; g.bias = bt; g.label = "opm shallow";
        gemm(g);
      }
      return;
    }
    float* Y = scratch<float>("opm.Y", (size_t)Bi * L * 128);
    for (int i0 = 0; i0 < L; i0 += Bi) {
      int bi = std::min(Bi, L - i0);
      Gemm g{}; g.X = Lt + (size_t)i0 * K; g.tx = F16; g.W = T; g.tw = F16; g.Y = Y; g.rows = bi; g.in = K; g.out = L * 128; g.label = "opm shallow";
      gemm(g);
      run1d("af2_opm_add", (size_t)bi * L * 128, OpmAddArgs{t.pair, Y, P(Op + "/output_b", blk), norm, (u64)i0, (uint)bi, (uint)L, 128, 0});
    }
    return;
  }
  // the standard form: P[(i, c)][(j, e)] = sum_s l[s][i][c] r[s][j][e] a block of rows i at a time, permuted to
  // [(i, j)][(c, e)], then the output projection
  size_t per = (size_t)L * O * O;
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / per));
  half* Pm = scratch<half>("opm.P", (size_t)Bi * per); half* X = scratch<half>("opm.X", (size_t)Bi * per);
  float* Y = scratch<float>("opm.Y", (size_t)Bi * L * 128);
  for (int i0 = 0; i0 < L; i0 += Bi) {
    int bi = std::min(Bi, L - i0);
    { Gemm g{}; g.X = lt + (size_t)i0 * O; g.tx = F16; g.transX = true; g.ldx = L * O; g.W = rt; g.tw = F16; g.ldw = L * O; g.Y = Pm;
      g.ty = F16; g.rows = (size_t)bi * O; g.in = rowsN; g.out = L * O; g.accFloat = true; g.label = "opm product"; gemm(g); }
    run1d("af2_opm_permute", (size_t)bi * L * O * (O / 8), OpmPermuteArgs{Pm, X, (uint)bi, (uint)L, (uint)O, 0});
    { Gemm g{}; g.X = X; g.tx = F16; g.W = Wout; g.tw = F16; g.Y = Y; g.rows = (size_t)bi * L; g.in = O * O; g.out = 128; g.label = "opm output"; gemm(g); }
    run1d("af2_opm_add", (size_t)bi * L * 128, OpmAddArgs{t.pair, Y, P(Op + "/output_b", blk), norm, (u64)i0, (uint)bi, (uint)L, 128, 0});
  }
}
// the triangle multiplication: LN -> one GEMM gating a and b into channel-major padded planes (gemmTriGate, with AF2's
// biases) -> the planes' batched product -> the centre norm -> the output projection -> the gate's GEMM adding into
// the pair (gemmGatedAdd)
struct TriW { const half* w4; const float* b4; };
static TriW triWeights(const std::string& T, int blk, int C) {
  std::string key = T + "#" + std::to_string(blk);
  if (triQuartersApply(C)) {    // (the matrix units' quarters, built once through a blocks-of-8 temporary)
    half* w4 = M.derived<half>("tri4q:" + key, (size_t)C * 4 * C, [](half*) {});
    float* b4 = M.derived<float>("tri4qb:" + key, (size_t)4 * C, [&](float* b) {
      half* w8 = allocT<half>((size_t)C * 4 * C); float* b8 = allocT<float>((size_t)4 * C);
      run1d("af2_trigate_weight", (size_t)C * 4 * C, TriGateW2Args{PH(T + "/projection/weights", blk), PH(T + "/gate/weights", blk),
            P(T + "/projection/bias", blk), P(T + "/gate/bias", blk), w8, b8, (uint)C, 0});
      triQuarters(w8, w4, b8, b, C);
      release(w8); release(b8);
    });
    return {w4, b4};
  }
  half* w4 = M.derived<half>("tri4:" + key, (size_t)C * 4 * C, [](half*) {});
  float* b4 = M.derived<float>("tri4b:" + key, (size_t)4 * C, [&](float* b) {
    run1d("af2_trigate_weight", (size_t)C * 4 * C, TriGateW2Args{PH(T + "/projection/weights", blk), PH(T + "/gate/weights", blk),
          P(T + "/projection/bias", blk), P(T + "/gate/bias", blk), w4, b, (uint)C, 0});
  });
  return {w4, b4};
}
void triangleMultiplication(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk, bool outgoing,
                            const std::string& nextNorm) {
  size_t pairs = (size_t)L * L;
  std::string T = S + (outgoing ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming");
  half* xn = pairLn(pair, pairs, C, T + "/left_norm_input", blk, "tri.xn");
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  half* a = scratch<half>("tri.a", plane * C); half* b = scratch<half>("tri.b", plane * C);
  static half *zeroedA = nullptr, *zeroedB = nullptr; static size_t zeroedPlane = 0;
  if (Lp != L && (a != zeroedA || b != zeroedB || plane * C != zeroedPlane)) {
    fill(a, 0, plane * C * 2); fill(b, 0, plane * C * 2);
    zeroedA = a; zeroedB = b; zeroedPlane = plane * C;
  }
  TriW w = triWeights(T, blk, C);
  gemmTriGate(xn, w.w4, pairMask, a, b, 0, pairs, C, plane, L, Lp, w.b4, triQuartersApply(C));
  float* prod = scratch<float>("tri.prod", plane * C);
  {
    Gemm g{}; g.tx = F16; g.tw = F16; g.ty = F32; g.rows = Lp; g.in = Lp; g.out = Lp; g.ldx = g.ldw = g.ldy = Lp;
    g.sx = g.sw = g.sy = (int64_t)plane; g.batch = C; g.Y = prod; g.label = "triangle contraction";
    if (outgoing) { g.X = a; g.W = b; g.transW = true; }
    else { g.X = b; g.transX = true; g.W = a; }
    gemm(g);
  }
  half* cn = scratch<half>("tri.cn", pairs * C);
  centerNorm(prod, cn, L, Lp, C, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk));
  half* outH = scratch<half>("tri.outh", pairs * C);
  half* lnOut = nextNorm.empty() ? nullptr : lnTarget(pairs, C, xn);
  if (gemmGatedAddDual(xn, PH(T + "/gating_linear/weights", blk), cn, PH(T + "/output_projection/weights", blk), pair, pairs, C, C,
                       P(T + "/gating_linear/bias", blk), P(T + "/output_projection/bias", blk), outH, "triangle output and gated add",
                       lnOut, lnOut ? P(nextNorm + "/scale", blk) : nullptr, lnOut ? P(nextNorm + "/offset", blk) : nullptr))
    emitted = {pair, normKey(nextNorm, blk), lnOut};
}
// the triangle attention: the starting node attends along a row, the ending node along a column (across the pair's
// leading axis: its bias the projection transposed, its key's mask pairMask[k][j])
void triangleAttention(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk, bool starting, bool pairOnes,
                       const std::string& nextNorm) {
  std::string A = S + (starting ? "triangle_attention_starting_node" : "triangle_attention_ending_node");
  int H = (int)dimW(A + "/attention/query_w", 2);
  half* xn = nullptr;     // (the bias's LN(pair) is the query's: the same norm)
  half* bias = pairBias(pair, L, C, A + "/query_norm", A + "/feat_2d_weights", blk, H, nullptr, !starting, &xn);
  if (starting) attend(xn, L, L, C, A + "/attention", blk, bias, pairOnes ? nullptr : pairMask, L, 1, pair, false, nextNorm, blk);
  else attend(xn, L, L, C, A + "/attention", blk, bias, pairOnes ? nullptr : pairMask, 1, L, pair, true, nextNorm, blk);
}

// one Evoformer iteration of a stack: S is "evoformer/evoformer_iteration/" or ".../extra_msa_stack/"
void evoformerBlock(Trunk& t, bool extraStack, int blk) {
  const std::string S = extraStack ? "evoformer/extra_msa_stack/" : "evoformer/evoformer_iteration/";
  float* msa = extraStack ? t.extra : t.msa;
  int rowsN = extraStack ? t.E : t.N + t.T, C = extraStack ? 64 : 256;
  const float* mask = extraStack ? t.extraMask : t.msaMask;
  bool ones = extraStack ? t.extraOnes : t.msaOnes;
  if (t.opmFirst) outerProductMean(t, S, blk, msa, rowsN, C, mask, ones);
  msaRowAttention(t, S, blk, msa, rowsN, C, mask, ones);
  if (extraStack) msaColumnGlobalAttention(t, S, blk, msa, rowsN, C, mask);
  else msaColumnAttention(t, S, blk, msa, rowsN, C, mask, ones);
  transition(msa, (size_t)rowsN * t.L, C, S + "msa_transition", blk);
  if (!t.opmFirst) outerProductMean(t, S, blk, msa, rowsN, C, mask, ones);
  // (each update's last GEMM emitting the next one's norm)
  triangleMultiplication(t.pair, t.pairMask, t.L, 128, S, blk, true, S + "triangle_multiplication_incoming/left_norm_input");
  triangleMultiplication(t.pair, t.pairMask, t.L, 128, S, blk, false, S + "triangle_attention_starting_node/query_norm");
  triangleAttention(t.pair, t.pairMask, t.L, 128, S, blk, true, t.pairOnes, S + "triangle_attention_ending_node/query_norm");
  triangleAttention(t.pair, t.pairMask, t.L, 128, S, blk, false, t.pairOnes, S + "pair_transition/input_layer_norm");
  transition(t.pair, (size_t)t.L * t.L, 128, S + "pair_transition", blk);
}
