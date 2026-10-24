// The input embedder and the Evoformer block - af3-any-model's AF2 multimer graph (alphafold3/af2/model/modules.py:
// EmbeddingsAndEvoformer, EvoformerIteration); cuda/af2/src/evoformer.cuh is the reading. Every attention runs on the
// core's flash kernel - the MSA's column attention and the triangle's ending node ACROSS the tensor where it lies (its
// strides), so nothing is transposed - and its one q/k/v/gate projection is a GEMM straight into the kernel's layout.
#include "af2.h"
#include <cmath>
#include <map>
#include <type_traits>

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
void layerNormW(const float* x, float* y, half* yh, size_t rows, int C, const std::string& w, int block) {
  layerNorm(x, y, yh, rows, C, P(w + "/scale", block), P(w + "/offset", block));
}
void layerNormW(const half* x, half* y, size_t rows, int C, const std::string& w, int block) {
  layerNorm(x, y, rows, C, P(w + "/scale", block), P(w + "/offset", block));
}
// the residual's type: the evoformer's activations are float, or half where halfActivations() (main.mm)
template <class T> constexpr DT dtOf() { return std::is_same<T, half>::value ? F16 : F32; }
// a linear whose weight and bias are named apart (an attention's output_w / output_b, gating_w / gating_b), into a float or
// a half Y
template <class T>
static void linearWB(const half* X, const std::string& w, const std::string& b, int blk, T* Y, size_t rows, int in, int out,
                     float beta) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = PH(w, blk); g.tw = F16; g.Y = Y; g.ty = dtOf<T>(); g.rows = rows; g.in = in; g.out = out;
  g.bias = P(b, blk); g.beta = beta; g.label = w.c_str();
  gemm(g);
}
// a named linear adding into a half residual (linearB's half twin: Y += X W + bias)
void linearB(const half* X, const std::string& w, int block, half* Y, size_t rows, int in, int out, bool relu, float beta) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = PH(w + "/weights", block); g.tw = F16; g.Y = Y; g.ty = F16; g.rows = rows; g.in = in;
  g.out = out; g.bias = P(w + "/bias", block); g.relu = relu; g.beta = beta; g.accFloat = true; g.label = w.c_str();
  gemm(g);
}
// the pair's next LayerNorm where the previous update's GEMM already wrote it (Gemm::lnOut, gemmGatedAddDual's): the
// pair, the norm (its weight's name and block) and the buffer, taken by pairLn of the same; a miss drops it.
// LOCALFOLD_LN_EMIT=0: none emitted
static struct { const void* x = nullptr; std::string norm; half* buf = nullptr; } emitted[4];
static void clearEmitted() { for (auto& e : emitted) e.x = nullptr; }
static std::string normKey(const std::string& w, int blk) { return w + "#" + std::to_string(blk); }
static void emitLn(const void* x, const std::string& w, int blk, half* buf) {
  const std::string key = normKey(w, blk);
  for (auto& e : emitted)
    if (e.x == x && e.norm == key) { e.buf = buf; return; }
  for (auto& e : emitted)
    if (!e.x) { e = {x, key, buf}; return; }
  emitted[0] = {x, key, buf};
}
static half* takeEmitted(const void* x, const std::string& w, int blk) {
  const std::string key = normKey(w, blk);
  for (auto& e : emitted)
    if (e.x == x && e.norm == key) { e.x = nullptr; return e.buf; }
  for (auto& e : emitted)
    if (e.x == x) e.x = nullptr;
  return nullptr;
}
template <class T>
static half* pairLn(const T* pair, size_t pairs, int C, const std::string& w, int blk, const char* scratchName,
                    const std::string& nextW = "", int nextBlk = 0, const char* nextScratch = "ln.d2") {
  if (half* b = takeEmitted(pair, w, blk)) return b;
  half* xn = scratch<half>(scratchName, pairs * C);
  if (!nextW.empty()) {
    half* xn2 = scratch<half>(nextScratch, pairs * C);
    layerNormDual(pair, xn, P(w + "/scale", blk), P(w + "/offset", blk),
                  xn2, P(nextW + "/scale", nextBlk), P(nextW + "/offset", nextBlk), pairs, C);
    emitLn(pair, nextW, nextBlk, xn2);
    return xn;
  }
  layerNormW(pair, xn, pairs, C, w, blk);
  return xn;
}
// where an update's GEMM emits the next norm: always ln.a (in the nested family main.mm names, so no module's shared
// scratch lands on it) - in place where the update's own input norm is the one the previous update emitted there. An
// emitting GEMM covers every output column of a row block in one tile, so the block's input rows are read before its
// epilogue writes the norm over them behind a threadgroup barrier (as metal/ef2's lnTarget); an attention's input norm
// is dead once q|k|v|g are made. A second buffer for it (ln.b) was at the evoformer's peak: 279 MB at 1,044 residues
static half* lnTarget(size_t pairs, int C, const half*) {
  return scratch<half>("ln.a", pairs * C);
}
static bool allOnes(const float* d, size_t n) {
  const float* h = (const float*)host(d);
  for (size_t i = 0; i < n; ++i) if (h[i] != 1.f) return false;
  return true;
}

// ---------------------------------------------------------------- the embedder
// prev: prevMsaRow [L, 256], prevPos [L, 37, 3]; the previous pair is t.pair itself (zero before the first pass)
void embed(Trunk& t, int pass, const float* prevMsaRow, const float* prevPos) {
  clearEmitted();
  const std::string E = "evoformer/";
  int L = t.L, N = t.N;
  std::string f = "f" + std::to_string(pass) + "/";
  t.normOf[0] = t.normOf[1] = nullptr;      // (the masks are written for this pass below)
  const int* aatype = Ii("aatype");
  half* tf = scratch<half>("emb.tf", (size_t)L * 21);
  run1d("af2_target_feat", (size_t)L * 21, TargetFeatArgs{aatype, tf, (uint)L, 0});
  // msa = preprocess_1d(target)[None] + preprocess_msa(msa_feat); row 0 += LN(prev msa first row)
  float* p1d = scratch<float>("emb.p1d", (size_t)L * 256);
  linearB(tf, E + "preprocess_1d", -1, p1d, L, 21, 256);
  linearB(In(f + "msa_feat"), E + "preprocess_msa", -1, t.msa, (size_t)N * L, 49, 256);
  run1d("af2_broadcast_rows", (size_t)N * L * 64, BroadcastRowsArgs{t.msa, p1d, (u64)N * L, (uint)L, 256});
  float* prevRowLn = scratch<float>("emb.prevRow", (size_t)L * 256);
  layerNormW(prevMsaRow, prevRowLn, L, 256, E + "prev_msa_first_row_norm");
  add(t.msa, prevRowLn, (size_t)L * 256);
  // pair = left[i] + right[j] + prev_pos_linear(dgram) + LN(prev pair) + relpos
  float* left = scratch<float>("emb.left", (size_t)L * 128); float* right = scratch<float>("emb.right", (size_t)L * 128);
  linearB(tf, E + "left_single", -1, left, L, 21, 128);
  linearB(tf, E + "right_single", -1, right, L, 21, 128);
  size_t pairs = (size_t)L * L;
  // (the recycled pair normalised in place - a row is read whole before it is written - and the outer sum added to it:
  // a copy of the last pass's pair and its norm were two f32 pairs more, 1.1 GB at 1044 residues)
  layerNormW(t.pair, t.pair, pairs, 128, E + "prev_pair_norm");
  run1d("af2_outer_sum", pairs * 32, OuterSumArgs{t.pair, left, right, (uint)L, 128, 1, 0});
  half* dgram = scratch<half>("emb.dgram", pairs * 15);
  run1d("af2_prev_dgram", pairs, PrevDgramArgs{prevPos, aatype, dgram, (uint)L, 0});
  linearB(dgram, E + "prev_pos_linear", -1, t.pair, pairs, 15, 128, false, 1.f);
  half* rel = scratch<half>("emb.rel", pairs * 73);
  run1d("af2_relpos", pairs, RelposArgs{Ii("residue_index"), Ii("asym_id"), Ii("entity_id"), Ii("sym_id"), rel, (uint)L, 0});
  linearB(rel, E + "~_relative_encoding/position_activations", -1, t.pair, pairs, 73, 128, false, 1.f);
  releaseScratch({"emb.rel", "emb.dgram"});
  // the extra MSA's activations
  size_t erows = (size_t)t.E * L;
  half* ef = scratch<half>("emb.extraFeat", erows * 25);
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
  int B = blk < 0 ? 1 : (int)dimW(A + "/query_w", 0);
  int H = (int)dimW(A + "/query_w", blk < 0 ? 1 : 2), D = (int)dimW(A + "/query_w", blk < 0 ? 2 : 3), W = H * D;
  size_t b = blk < 0 ? 0 : (size_t)blk;
  half* w = M.derived<half>("qkvg:" + A, (size_t)B * C * 4 * W, [](half*) {});
  float* bias = M.derived<float>("qkvgb:" + A, (size_t)B * 4 * W, [&](float* bOut) {
    run1d("af2_qkvg_weight", (size_t)B * C * (W / 2),
          QkvgWArgs{PH(A + "/query_w"), PH(A + "/key_w"), PH(A + "/value_w"), PH(A + "/gating_w"), P(A + "/gating_b"),
                    w, bOut, (uint)C, (uint)W, (uint)B, 0});
  });
  return {w + b * C * 4 * W, bias + b * 4 * W, H, D};
}
// a pair bias's row stride: padded with -inf to a whole 64-key tile (af2_bias_layout; the attention's NT kernel, core.mm)
static int biasStride(int L) { return (L + 63) / 64 * 64; }
// xn [Bt][n][C] (or the leading axis attended, `across`: xn [n][Bt][C]) -> the residual += attention's output projection
template <class T>
static void attend(const half* xn, int Bt, int n, int C, const std::string& A, int blk, const half* bias, const float* mask,
                   int64_t maskB, int64_t maskK, T* residual, bool across, const std::string& nextNorm = "", int nextBlk = 0) {
  AttnW w = attnWeights(A, blk, C);
  size_t rows = (size_t)Bt * n; int W = w.H * w.D;
  // q|k|v|g a block of batch rows at a time, 256 MB of them: a triangle attention's whole pair's was 1.1 GB at 1044
  // residues, and an alignment's as much. A batch row attends over its own n positions only, so the blocks are the same
  // arithmetic; below ~500 residues (and most alignments) one block is everything
  const size_t perRow = (size_t)n * (n == 1 ? 2 : 4) * W * 2;
  const int rb = qkvgBlockRows(perRow, Bt);
  half* qkvg = scratch<half>("att.qkvg", (size_t)rb * perRow / 2);
  half* o = scratch<half>("att.o", rows * W);
  if (n == 1) {
    Gemm g{}; g.X = xn; g.tx = F16; g.W = w.qkvg + 2 * W; g.ldw = 4 * W; g.tw = F16;
    g.Y = qkvg; g.ty = F16; g.rows = Bt; g.in = C; g.out = 2 * W; g.bias = w.bias + 2 * W;
    g.label = "attention qkvg";
    gemm(g);
    run1d("af2_vg_gate", (W & 3) ? rows * W : rows * (W >> 2), VgGateArgs{qkvg, o, (uint)Bt, (uint)W});
  } else {
    for (int r0 = 0; r0 < Bt; r0 += rb) {
      const int r = std::min(rb, Bt - r0);
      // (across: batch rows r0.. are r rows at each of the n positions, a GEMM batched over them)
      Gemm g{}; g.W = w.qkvg; g.tw = F16; g.tx = F16; g.Y = qkvg; g.ty = F16; g.in = C; g.out = 4 * W; g.bias = w.bias;
      g.label = "attention qkvg";
      if (!across || r == Bt) { g.X = xn + (size_t)r0 * n * C; g.rows = (size_t)r * n; }
      else { g.X = xn + (size_t)r0 * C; g.rows = r; g.ldx = C; g.sx = (int64_t)Bt * C; g.sy = (int64_t)r * 4 * W; g.batch = n; }
      gemm(g);
      Attention at{}; at.qkvg = qkvg; at.n = n; at.heads = w.H; at.D = w.D; at.rows = r; at.scale = 1.f / sqrtf((float)w.D);
      at.bias = bias; at.biasStride = bias ? biasStride(n) : n; at.biasPadInf = bias != nullptr;
      at.mask = mask; at.maskB = maskB; at.maskK = maskK; at.r0 = r0;
      if (!across) at.out = o + (size_t)r0 * n * W;
      else {
        at.rowStride = 4 * W; at.posStride = (int64_t)r * 4 * W;
        at.out = o + (size_t)r0 * W; at.outRowStride = W; at.outPosStride = (int64_t)Bt * W;
      }
      attention(at);
    }
  }
  if (!nextNorm.empty() && (C == 64 || C == 128 || C == 256)) {     // (the residual's next norm emitted by the output projection)
    half* lnOut = lnTarget(rows, C, xn);
    Gemm g{}; g.X = o; g.tx = F16; g.W = PH(A + "/output_w", blk); g.tw = F16; g.Y = residual; g.ty = dtOf<T>(); g.rows = rows; g.in = W; g.out = C;
    g.bias = P(A + "/output_b", blk); g.beta = 1.f; g.label = "attention output, next norm";
    g.lnOut = lnOut; g.lnScale = P(nextNorm + "/scale", nextBlk); g.lnOffset = P(nextNorm + "/offset", nextBlk);
    if (gemm(g)) emitLn(residual, nextNorm, nextBlk, lnOut);
    return;
  }
  linearWB(o, A + "/output_w", A + "/output_b", blk, residual, rows, W, C, 1.f);
}
// a pair bias [H][L][biasStride(L)] (log2 units, + 1e9 (mask - 1) where a mask is given, -inf past L) from LN(pair) W
template <class T>
static half* pairBias(const T* pair, int L, int C, const std::string& normW, const std::string& projW, int blk, int H,
                      const float* pairMask, bool transposed, half** normed = nullptr,
                      const std::string& nextPairNorm = "", const char* nextScratch = "ln.d2") {
  size_t pairs = (size_t)L * L;
  half* pn = pairLn(pair, pairs, C, normW, blk, "bias.pn", nextPairNorm, blk, nextScratch);
  if (normed) *normed = pn;
  // (rows padded with -inf to a whole 64-key tile: biasStride(L), and the attention's NT kernel - core.mm)
  const int S = biasStride(L);
  half* b = scratch<half>("bias.b", (size_t)H * L * S);
  // (the pair bias projected and laid out per head in one kernel; LOCALFOLD_PAIR_BIAS=0 the control - a GEMM, then the layout)
  static const bool fusedPairBias = !getenv("LOCALFOLD_PAIR_BIAS") || atoi(getenv("LOCALFOLD_PAIR_BIAS")) != 0;
  if (fusedPairBias && C <= 128 && (C & 31) == 0 && (H == 4 || H == 8)) {
    run("af2_pair_bias", Grid{(uint32_t)((L + 7) / 8), (uint32_t)((L + 7) / 8), 1}, 256,
        PairBiasArgs{pn, PH(projW, blk), pairMask, b, (uint)L, (uint)C, (uint)H, transposed ? 1u : 0u, (uint)S, 0});
    return b;
  }
  float* proj = scratch<float>("bias.proj", pairs * H);
  Gemm g{}; g.X = pn; g.tx = F16; g.W = PH(projW, blk); g.tw = F16; g.Y = proj; g.rows = pairs; g.in = C; g.out = H; g.label = "pair bias";
  gemm(g);
  run("af2_bias_layout", Grid{(uint32_t)((S + 31) / 32), (uint32_t)((L + 31) / 32), (uint32_t)H}, 256,
      BiasLayoutArgs{proj, pairMask, b, (uint)L, (uint)H, transposed ? 1u : 0u, (uint)S});
  return b;
}

// ---------------------------------------------------------------- the block's modules
template <class T>
static void msaRowAttention(Trunk& t, const std::string& S, int blk, T* msa, const T* pair, int rowsN, int C, const float* msaMask,
                            bool ones, const std::string& nextNorm = "", const std::string& nextPairNorm = "") {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string R = S + "msa_row_attention_with_pair_bias";
  int H = (int)dimW(R + "/attention/query_w", 2);
  half* bias = pairBias(pair, L, 128, R + "/feat_2d_norm", R + "/feat_2d_weights", blk, H, t.pairOnes ? nullptr : t.pairMask, false,
                        nullptr, nextPairNorm, "ln.p2");
  half* xn = pairLn(msa, rows, C, R + "/query_norm", blk, "row.xn");
  attend(xn, rowsN, L, C, R + "/attention", blk, bias, ones ? nullptr : msaMask, L, 1, msa, false, nextNorm, blk);
}
template <class T>
static void msaColumnAttention(Trunk& t, const std::string& S, int blk, T* msa, int rowsN, int C, const float* msaMask, bool ones,
                               const std::string& nextNorm = "") {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string A = S + "msa_column_attention";
  half* xn = pairLn(msa, rows, C, A + "/query_norm", blk, "col.xn");
  // across the sequences: batch row = the column i, positions the sequences s; the key's mask msaMask[s][i]
  attend(xn, L, rowsN, C, A + "/attention", blk, nullptr, ones ? nullptr : msaMask, 1, L, msa, true, nextNorm, blk);
}
template <class T>
static void msaColumnGlobalAttention(Trunk& t, const std::string& S, int blk, T* msa, int rowsN, int C, const float* msaMask,
                                     const std::string& nextNorm = "") {
  int L = t.L; size_t rows = (size_t)rowsN * L;
  std::string A = S + "msa_column_global_attention";
  int H = (int)dimW(A + "/attention/query_w", 2), D = (int)dimW(A + "/attention/query_w", 3), W = H * D;
  if (H > 8 || D > 16 || C > 256) die("global attention: %d heads of %d over %d channels", H, D, C);
  half* xn = pairLn(msa, rows, C, A + "/query_norm", blk, "gcol.xn");
  float* avg = scratch<float>("gcol.avg", (size_t)L * (rowsN == 1 ? D : W));
  if (rowsN == 1) {
    Gemm g{}; g.X = xn; g.tx = F16; g.W = PH(A + "/attention/value_w", blk); g.tw = F16; g.Y = avg;
    g.rows = L; g.in = C; g.out = D; g.label = "global kv"; gemm(g);
  } else {
    // every sequence's key and value (shared by the heads): one GEMM against [key_w | value_w]
    int B = (int)dimW(A + "/attention/query_w", 0);
    const half* kvw = M.derived<half>("kv:" + A, (size_t)B * C * 2 * D, [&](half* out) {
      copy2d(out, 2 * D * 2, PH(A + "/attention/key_w"), D * 2, D * 2, (size_t)B * C);
      copy2d(out + D, 2 * D * 2, PH(A + "/attention/value_w"), D * 2, D * 2, (size_t)B * C);
    }) + (size_t)blk * C * 2 * D;
    float* kv = scratch<float>("gcol.kv", rows * 2 * D);
    { Gemm g{}; g.X = xn; g.tx = F16; g.W = kvw; g.tw = F16; g.Y = kv; g.rows = rows; g.in = C; g.out = 2 * D; g.label = "global kv"; gemm(g); }
    run("af2_global_attention", Grid{(uint32_t)L, 1, 1}, 256,
        GlobalAttnArgs{xn, kv, msaMask, PH(A + "/attention/query_w", blk), avg, (uint)rowsN, (uint)L, (uint)C, (uint)H, (uint)D, 0});
  }
  float* gate = scratch<float>("gcol.gate", rows * W);
  linearWB(xn, A + "/attention/gating_w", A + "/attention/gating_b", blk, gate, rows, C, W, 0.f);
  half* gated = scratch<half>("gcol.gated", rows * W);
  run1d("af2_global_gate", (W & 3) ? rows * W : rows * (W >> 2),
        GlobalGateArgs{avg, gate, gated, (uint)rowsN, (uint)L, (uint)W, rowsN == 1 ? (uint)D : 0u});
  if (!nextNorm.empty() && (C == 64 || C == 128 || C == 256)) {
    half* lnOut = lnTarget(rows, C, xn);
    Gemm g{}; g.X = gated; g.tx = F16; g.W = PH(A + "/attention/output_w", blk); g.tw = F16; g.Y = msa; g.ty = dtOf<T>();
    g.rows = rows; g.in = W; g.out = C; g.bias = P(A + "/attention/output_b", blk); g.beta = 1.f;
    g.label = "global output, next norm"; g.lnOut = lnOut; g.lnScale = P(nextNorm + "/scale", blk); g.lnOffset = P(nextNorm + "/offset", blk);
    if (gemm(g)) emitLn(msa, nextNorm, blk, lnOut);
    return;
  }
  linearWB(gated, A + "/attention/output_w", A + "/attention/output_b", blk, msa, rows, W, C, 1.f);
}
template <class X>
static void transitionT(X* x, size_t rows, int C, const std::string& T, int blk,
                        const std::string& nextNorm = "", int nextBlk = 0,
                        const std::string& nextNorm2 = "", int nextBlk2 = 0,
                        const char* lnName1 = nullptr) {
  half* emittedIn = takeEmitted(x, T + "/input_layer_norm", blk);
  int I = (int)dimW(T + "/transition1/weights", blk < 0 ? 1 : 2);
  size_t chunk = std::min(rows, std::max<size_t>(32768, ((size_t)128 << 20) / (2 * (size_t)I)));
  half* xn = scratch<half>("tr.xn", chunk * C);
  half* mid = scratch<half>("tr.mid", chunk * I);
  const bool emitNext = !nextNorm.empty() && (C == 64 || C == 128 || C == 256);
  const bool emitNext2 = emitNext && !nextNorm2.empty() && rows <= 16384;
  half* ln1 = emitNext ? (lnName1 ? scratch<half>(lnName1, rows * C) : lnTarget(rows, C, nullptr)) : nullptr;
  half* ln2 = emitNext2 ? scratch<half>("ln.m", rows * C) : nullptr;
  bool emittedAll = emitNext;
  for (size_t r0 = 0; r0 < rows; r0 += chunk) {
    size_t r = std::min(chunk, rows - r0);
    const half* in = emittedIn ? emittedIn + r0 * C : xn;     // (the emitted norm read a chunk at a time)
    if (in == xn) layerNormW(x + r0 * C, xn, r, C, T + "/input_layer_norm", blk);
    Gemm g{}; g.X = in; g.tx = F16; g.W = PH(T + "/transition1/weights", blk); g.tw = F16; g.Y = mid; g.ty = F16;
    g.rows = r; g.in = C; g.out = I; g.bias = P(T + "/transition1/bias", blk); g.relu = true; g.label = "transition1";
    gemm(g);
    if (emittedAll) {
      Gemm G{}; G.X = mid; G.tx = F16; G.W = PH(T + "/transition2/weights", blk); G.tw = F16; G.Y = x + r0 * C; G.ty = dtOf<X>();
      G.rows = r; G.in = I; G.out = C; G.bias = P(T + "/transition2/bias", blk); G.beta = 1.f; G.label = "transition2, next norm";
      G.lnOut = ln1 + r0 * C; G.lnScale = P(nextNorm + "/scale", nextBlk); G.lnOffset = P(nextNorm + "/offset", nextBlk);
      if (ln2) { G.lnOut2 = ln2 + r0 * C; G.lnScale2 = P(nextNorm2 + "/scale", nextBlk2); G.lnOffset2 = P(nextNorm2 + "/offset", nextBlk2); }
      if (gemm(G)) continue;
      if (r0) die("%s: the next norm emitted for some chunks and not others", T.c_str());
      emittedAll = false;
      continue;
    }
    linearB(mid, T + "/transition2", blk, x + r0 * C, r, I, C, false, 1.f);
  }
  if (emittedAll) {
    emitLn(x, nextNorm, nextBlk, ln1);
    if (ln2) emitLn(x, nextNorm2, nextBlk2, ln2);
  }
}
void transition(float* x, size_t rows, int C, const std::string& T, int blk,
                const std::string& nextNorm, int nextBlk) {
  transitionT(x, rows, C, T, blk, nextNorm, nextBlk);
}
// pair[(i0 + i)][j] += (bias + Y) / (1e-3 + norm), float or half, optionally emitting the next norm
static void opmAdd(float* pair, const float* Y, const float* bias, const float* norm, int i0, int bi, int L,
                   half* lnOut = nullptr, const float* lnScale = nullptr, const float* lnOffset = nullptr) {
  run1d("af2_opm_add", (size_t)bi * L * 32, OpmAddArgs{pair, Y, bias, norm, lnOut, lnScale, lnOffset, (u64)i0, (uint)bi, (uint)L, 128, 0});
}
static void opmAdd(half* pair, const float* Y, const float* bias, const float* norm, int i0, int bi, int L,
                   half* lnOut = nullptr, const float* lnScale = nullptr, const float* lnOffset = nullptr) {
  run1d("af2_opm_add_h", (size_t)bi * L * 32, OpmAddHArgs{pair, Y, bias, norm, lnOut, lnScale, lnOffset, (u64)i0, (uint)bi, (uint)L, 128, 0});
}
// the outer product mean: pair[i][j] += (sum_s l[s][i] (x) r[s][j] @ W + b) / (1e-3 + norm[i][j])
struct OpmW { const half* w2; const float* b2; };
static OpmW opmWeights(const std::string& Op, int blk, int C, int O) {
  int B = blk < 0 ? 1 : (int)dimW(Op + "/left_projection/weights", 0);
  size_t b = blk < 0 ? 0 : (size_t)blk;
  half* w2 = M.derived<half>("opm2:" + Op, (size_t)B * C * 2 * O, [&](half* out) {
    copy2d(out, 2 * O * 2, PH(Op + "/left_projection/weights"), O * 2, O * 2, (size_t)B * C);
    copy2d(out + O, 2 * O * 2, PH(Op + "/right_projection/weights"), O * 2, O * 2, (size_t)B * C);
  });
  float* b2 = M.derived<float>("opm2b:" + Op, (size_t)B * 2 * O, [&](float* out) {
    copy2d(out, 2 * O * 4, P(Op + "/left_projection/bias"), O * 4, O * 4, (size_t)B);
    copy2d(out + O, 2 * O * 4, P(Op + "/right_projection/bias"), O * 4, O * 4, (size_t)B);
  });
  return {w2 + b * C * 2 * O, b2 + b * 2 * O};
}
template <class A>
static void outerProductMean(Trunk& t, const std::string& S, int blk, const A* msa, A* pair, int rowsN, int C, const float* msaMask,
                             bool ones, const std::string& nextNorm = "", const std::string& nextNorm2 = "",
                             const std::string& nextMsaNorm = "") {
  int L = t.L; size_t rows = (size_t)rowsN * L; const int O = 32;
  std::string Op = S + "outer_product_mean";
  half* xn = pairLn(msa, rows, C, Op + "/layer_norm_input", blk, "opm.xn", nextMsaNorm, blk, "ln.m");
  // the shallow form reads l | r interleaved, [s][i][l | r] from one GEMM (ldx 2 O); 🔴 THE STANDARD FORM READS EACH AS
  // ITS OWN [s][(i, c)] MATRIX, i O + c, which the interleaved buffer is not (93890570 gave it that buffer: every fold
  // with more than 128 rows read a mix of l and r - a 59-residue query at 8,076 rows pLDDT 96.2 -> 47.6), so past 128
  // rows the two projections stay two GEMMs into two buffers
  const bool merged = rowsN <= 128;
  half* lt; half* rt;
  if (merged) {
    OpmW opm = opmWeights(Op, blk, C, O);
    half* lrt = scratch<half>("opm.left", rows * 2 * O);
    lt = lrt; rt = lrt + O;
    Gemm g{}; g.X = xn; g.tx = F16; g.W = opm.w2; g.tw = F16; g.Y = lrt; g.ty = F16; g.rows = rows;
    g.in = C; g.out = 2 * O; g.bias = opm.b2; g.label = "opm projection";
    gemm(g);
    if (!ones) run1d("af2_scale_rows_h", rows * ((2 * O) >> 2), ScaleRowsHArgs{lrt, nullptr, msaMask, rows, (uint)(2 * O), 0});
  } else {
    lt = scratch<half>("opm.left", rows * O); rt = scratch<half>("opm.right", rows * O);
    for (int side = 0; side < 2; ++side) {
      std::string w = Op + (side ? "/right_projection" : "/left_projection");
      Gemm g{}; g.X = xn; g.tx = F16; g.W = PH(w + "/weights", blk); g.tw = F16; g.Y = side ? rt : lt; g.ty = F16; g.rows = rows;
      g.in = C; g.out = O; g.bias = P(w + "/bias", blk); g.label = "opm projection";
      gemm(g);
    }
    if (!ones) run1d("af2_scale_rows_h", rows * (O >> 2), ScaleRowsHArgs{lt, rt, msaMask, rows, (uint)O, 0});
  }
  // the mask norm, once a stack a pass: every block of a stack reads the same mask (LOCALFOLD_OPM_NORM_ONCE=0: each block)
  static const bool once = !getenv("LOCALFOLD_OPM_NORM_ONCE") || atoi(getenv("LOCALFOLD_OPM_NORM_ONCE")) != 0;
  const int slot = msaMask == t.extraMask;
  // (written once and read by every block after: never in a slot another module's buffers share - main.mm's families
  // leave "mnorm." out; at 2,900 residues it is past the 32 MB a shared buffer starts at)
  float* norm = scratch<float>(slot ? "mnorm.extra" : "mnorm.main", (size_t)L * L);
  if (!once || t.normOf[slot] != msaMask) {
    run1d("af2_mask_norm", (size_t)L * L, MaskNormArgs{msaMask, norm, (uint)rowsN, (uint)L});
    t.normOf[slot] = msaMask;
  }
  const half* Wout = PH(Op + "/output_w", blk);        // [O * O][128]
  static const bool lnEmit = !getenv("LOCALFOLD_LN_EMIT") || atoi(getenv("LOCALFOLD_LN_EMIT")) != 0;
  half* lnOut = (!nextNorm.empty() && lnEmit && matrixUnits()) ? lnTarget((size_t)L * L, 128, nullptr) : nullptr;
  const float* lnScale = lnOut ? P(nextNorm + "/scale", blk) : nullptr;
  const float* lnOffset = lnOut ? P(nextNorm + "/offset", blk) : nullptr;
  half* lnOut2 = (lnOut && !nextNorm2.empty() && L <= 512) ? scratch<half>("ln.p2", (size_t)L * L * 128) : nullptr;
  const float* lnScale2 = lnOut2 ? P(nextNorm2 + "/scale", blk) : nullptr;
  const float* lnOffset2 = lnOut2 ? P(nextNorm2 + "/offset", blk) : nullptr;
  if (rowsN <= 128) {
    // the SHALLOW form for an alignment of few rows: the output projection folded into the right operand first -
    // T[c][(s, j)][f] = sum_e r[s][j][e] W[c O + e][f], a GEMM batched over c - then Y[i][(j, f)] = sum_{c, s}
    // l[s][i][c] T[(c, s)][(j, f)], one GEMM of K = O S: n^2 S O 128 against the standard form's n^2 O^2 (S + 128)
    half* T = scratch<half>("opm.T", (size_t)O * rows * 128);
    { Gemm g{}; g.X = rt; g.ldx = 2 * O; g.tx = F16; g.W = Wout; g.tw = F16; g.sw = (int64_t)O * 128; g.Y = T; g.ty = F16; g.sy = (int64_t)rows * 128;
      g.rows = rows; g.in = O; g.out = 128; g.batch = O; g.accFloat = true; g.label = "opm shallow T"; gemm(g); }
    half* Lt = scratch<half>("opm.Lt", rows * O);
    const bool foldOnes = ones && (!lnOut || (L > 64 && L <= 128));
    const float inv = foldOnes ? 1.f / (1e-3f + (float)rowsN) : 1.f;
    float* bt = foldOnes ? scratch<float>("opm.btile", (size_t)L * 128) : nullptr;
    run1d("af2_opm_left", std::max<size_t>(rows * O, bt ? (size_t)L * 32 : 0),
          OpmLeftArgs{lt, Lt, (uint)rowsN, (uint)L, (uint)O, inv, 2u * (uint)O, bt ? P(Op + "/output_b", blk) : nullptr, bt});
    int K = O * rowsN;
    int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / ((size_t)L * 128)));
    if (foldOnes) {     // every row live: the norm is rowsN everywhere, folded into the GEMM (L scaled, the bias tiled)
      for (int i0 = 0; i0 < L; i0 += Bi) {
        int bi = std::min(Bi, L - i0);
        Gemm g{}; g.X = Lt + (size_t)i0 * K; g.tx = F16; g.W = T; g.tw = F16; g.Y = pair + (size_t)i0 * L * 128; g.ty = dtOf<A>(); g.beta = 1.f;
        g.rows = bi; g.in = K; g.out = L * 128; g.bias = bt; g.label = "opm shallow";
        gemm(g);
      }
      return;
    }
    bool allAdded = lnOut != nullptr;
    for (int i0 = 0; i0 < L; i0 += Bi) {
      int bi = std::min(Bi, L - i0);
      half* lnChunk = lnOut ? lnOut + (size_t)i0 * L * 128 : nullptr;
      half* lnChunk2 = lnOut2 ? lnOut2 + (size_t)i0 * L * 128 : nullptr;
      if (gemmOpmOut(Lt + (size_t)i0 * K, T, pair + (size_t)i0 * L * 128, P(Op + "/output_b", blk), norm + (size_t)i0 * L,
                     bi, K, L * 128, 1.f, lnChunk, lnScale, lnOffset, 1e-5f, lnChunk2, lnScale2, lnOffset2)) {
        continue;
      }
      allAdded = false;
      float* Y = scratch<float>("opm.Y", (size_t)Bi * L * 128);
      Gemm g{}; g.X = Lt + (size_t)i0 * K; g.tx = F16; g.W = T; g.tw = F16; g.Y = Y; g.rows = bi; g.in = K; g.out = L * 128; g.label = "opm shallow";
      gemm(g);
      opmAdd(pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L, lnOut, lnScale, lnOffset);
    }
    if (allAdded) {
      emitLn(pair, nextNorm, blk, lnOut);
      if (lnOut2) emitLn(pair, nextNorm2, blk, lnOut2);
    }
    return;
  }
  // the standard form: P[(i, c)][(j, e)] = sum_s l[s][i][c] r[s][j][e] a block of rows i at a time, permuted to
  // [(i, j)][(c, e)], then the output projection
  size_t per = (size_t)L * O * O;
  int Bi = (int)std::max<size_t>(1, std::min<size_t>(L, ((size_t)64 << 20) / per));
  half* X = scratch<half>("opm.X", (size_t)Bi * per);
  float* Y = nullptr;
  // 🔴 THE SUM OVER THE ROWS SCALED BY 2^-k AS IT IS STORED IN HALF - exact, a power of two - and scaled back in the output
  // (cuda/af2's a04ed7e: unscaled, 7GYV's ColabFold alignment at 1,945 extra rows put an entry past 65,504 and NaN'd every
  // coordinate). 2^k ~ the row count / 64, its k: the largest entry there ~34 a row, so ~2,200 scaled; scaling all the way
  // to the mean reached float16's subnormals and moved 7GYV's fold
  int k = 0; while ((64 << k) < rowsN) ++k;
  const float down = ldexpf(1.f, -k), up = ldexpf(1.f, k);
  bool allAdded = lnOut != nullptr;
  for (int i0 = 0; i0 < L; i0 += Bi) {
    int bi = std::min(Bi, L - i0);
    // (on the matrix units the product's epilogue stores it permuted - no Pm, no permute pass)
    if (!gemmOpmPermuted(lt + (size_t)i0 * O, L * O, rt, L * O, X, bi, L, O, rowsN, down)) {
      half* Pm = scratch<half>("opm.P", (size_t)Bi * per);
      { Gemm g{}; g.X = lt + (size_t)i0 * O; g.tx = F16; g.transX = true; g.ldx = L * O; g.W = rt; g.tw = F16; g.ldw = L * O; g.Y = Pm;
        g.ty = F16; g.rows = (size_t)bi * O; g.in = rowsN; g.out = L * O; g.alpha = down; g.accFloat = true; g.label = "opm product";
        gemm(g); }
      run1d("af2_opm_permute", (size_t)bi * L * O * (O / 8), OpmPermuteArgs{Pm, X, (uint)bi, (uint)L, (uint)O, 0});
    }
    // (on the matrix units the add in the output GEMM's epilogue - no Y)
    half* lnChunk = lnOut ? lnOut + (size_t)i0 * L * 128 : nullptr;
    half* lnChunk2 = lnOut2 ? lnOut2 + (size_t)i0 * L * 128 : nullptr;
    if (gemmOpmOut(X, Wout, pair + (size_t)i0 * L * 128, P(Op + "/output_b", blk), norm + (size_t)i0 * L, (size_t)bi * L, O * O, 128, up,
                   lnChunk, lnScale, lnOffset, 1e-5f, lnChunk2, lnScale2, lnOffset2)) {
      continue;
    }
    if (!Y) Y = scratch<float>("opm.Y", (size_t)Bi * L * 128);
    { Gemm g{}; g.X = X; g.tx = F16; g.W = Wout; g.tw = F16; g.Y = Y; g.rows = (size_t)bi * L; g.in = O * O; g.out = 128; g.alpha = up;
      g.label = "opm output"; gemm(g); }
    opmAdd(pair, Y, P(Op + "/output_b", blk), norm, i0, bi, L, lnOut, lnScale, lnOffset);
  }
  if (allAdded) {
    emitLn(pair, nextNorm, blk, lnOut);
    if (lnOut2) emitLn(pair, nextNorm2, blk, lnOut2);
  }
}
// the triangle multiplication: LN -> one GEMM gating a and b into channel-major padded planes (gemmTriGate, with AF2's
// biases) -> the planes' batched product -> the centre norm -> the output projection -> the gate's GEMM adding into
// the pair (gemmGatedAdd)
struct TriW { const half* w4; const float* b4; };
static TriW triWeights(const std::string& T, int blk, int C) {
  int B = blk < 0 ? 1 : (int)dimW(T + "/projection/weights", 0);
  size_t b = blk < 0 ? 0 : (size_t)blk;
  uint q = triQuartersApply(C) ? 1u : 0u;
  std::string tag = (q ? "tri4q:" : "tri4:") + T;
  half* w4 = M.derived<half>(tag, (size_t)B * C * 4 * C, [](half*) {});
  float* b4 = M.derived<float>(tag + ":b", (size_t)B * 4 * C, [&](float* bOut) {
    run1d("af2_trigate_weight", (size_t)B * C * (C / 2),
          TriGateW2Args{PH(T + "/projection/weights"), PH(T + "/gate/weights"), P(T + "/projection/bias"),
                        P(T + "/gate/bias"), w4, bOut, (uint)C, (uint)B, q, 0});
  });
  return {w4 + b * C * 4 * C, b4 + b * 4 * C};
}
template <class X>
static void triangleMultiplicationT(X* pair, const float* pairMask, int L, int C, const std::string& S, int blk, bool outgoing,
                                    const std::string& nextNorm) {
  size_t pairs = (size_t)L * L;
  std::string T = S + (outgoing ? "triangle_multiplication_outgoing" : "triangle_multiplication_incoming");
  half* xn = pairLn(pair, pairs, C, T + "/left_norm_input", blk, "tri.xn");
  int Lp = (L + 7) / 8 * 8; size_t plane = (size_t)Lp * Lp;
  // the planes, their contraction, the centre norm and the gated add's value in one buffer (core's triContract: 0.63 GB
  // at 1044 residues, against 0.84 for the planes and the product apart)
  void* big = scratchBytes("tri.big", triBytes(C, Lp));
  half* a = triPlanes(big, L, C, Lp);
  TriW w = triWeights(T, blk, C);
  gemmTriGate(xn, w.w4, pairMask, a, a + plane, 0, pairs, C, 2 * plane, L, Lp, w.b4, triQuartersApply(C));
  TriMid mid = triContract(big, L, C, outgoing, 1.f, P(T + "/center_norm/scale", blk), P(T + "/center_norm/offset", blk),
                           halfProductScale() ? nullptr : scratch<half>("tri.cn", pairs * C), "triangle contraction");
  half* cn = mid.cn; half* outH = mid.outH;
  half* lnOut = nextNorm.empty() ? nullptr : lnTarget(pairs, C, xn);
  if (gemmGatedAddDual(xn, PH(T + "/gating_linear/weights", blk), cn, PH(T + "/output_projection/weights", blk), pair, pairs, C, C,
                       P(T + "/gating_linear/bias", blk), P(T + "/output_projection/bias", blk), outH, "triangle output and gated add",
                       lnOut, lnOut ? P(nextNorm + "/scale", blk) : nullptr, lnOut ? P(nextNorm + "/offset", blk) : nullptr))
    emitLn(pair, nextNorm, blk, lnOut);
}
void triangleMultiplication(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk, bool outgoing,
                            const std::string& nextNorm) {
  triangleMultiplicationT(pair, pairMask, L, C, S, blk, outgoing, nextNorm);
}
// the triangle attention: the starting node attends along a row, the ending node along a column (across the pair's
// leading axis: its bias the projection transposed, its key's mask pairMask[k][j])
template <class X>
static void triangleAttentionT(X* pair, const float* pairMask, int L, int C, const std::string& S, int blk, bool starting, bool pairOnes,
                               const std::string& nextNorm) {
  std::string A = S + (starting ? "triangle_attention_starting_node" : "triangle_attention_ending_node");
  int H = (int)dimW(A + "/attention/query_w", 2);
  half* xn = nullptr;     // (the bias's LN(pair) is the query's: the same norm)
  half* bias = pairBias(pair, L, C, A + "/query_norm", A + "/feat_2d_weights", blk, H, nullptr, !starting, &xn);
  if (starting) attend(xn, L, L, C, A + "/attention", blk, bias, pairOnes ? nullptr : pairMask, L, 1, pair, false, nextNorm, blk);
  else attend(xn, L, L, C, A + "/attention", blk, bias, pairOnes ? nullptr : pairMask, 1, L, pair, true, nextNorm, blk);
}
void triangleAttention(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk, bool starting, bool pairOnes,
                       const std::string& nextNorm) {
  triangleAttentionT(pair, pairMask, L, C, S, blk, starting, pairOnes, nextNorm);
}

// one Evoformer iteration of a stack: S is "evoformer/evoformer_iteration/" or ".../extra_msa_stack/"; over the float
// activations, or their half copies where main.mm made them (Trunk::h)
template <class T>
static void evoformerBlockT(Trunk& t, bool extraStack, int blk, T* msa, T* pair,
                            const std::string& nextPairNorm, int nextPairBlk) {
  const std::string S = extraStack ? "evoformer/extra_msa_stack/" : "evoformer/evoformer_iteration/";
  int rowsN = extraStack ? t.E : t.N + t.T, C = extraStack ? 64 : 256;
  const float* mask = extraStack ? t.extraMask : t.msaMask;
  bool ones = extraStack ? t.extraOnes : t.msaOnes;
  bool moreMain = !extraStack && nextPairBlk >= 0;
  if (t.opmFirst) outerProductMean(t, S, blk, (const T*)msa, pair, rowsN, C, mask, ones,
                                   S + "msa_row_attention_with_pair_bias/feat_2d_norm",
                                   S + "triangle_multiplication_outgoing/left_norm_input",
                                   S + "msa_row_attention_with_pair_bias/query_norm");
  msaRowAttention(t, S, blk, msa, (const T*)pair, rowsN, C, mask, ones,
                  S + (extraStack ? "msa_column_global_attention/query_norm" : "msa_column_attention/query_norm"),
                  t.opmFirst ? S + "triangle_multiplication_outgoing/left_norm_input" : "");
  if (extraStack) msaColumnGlobalAttention(t, S, blk, msa, rowsN, C, mask, S + "msa_transition/input_layer_norm");
  else msaColumnAttention(t, S, blk, msa, rowsN, C, mask, ones, S + "msa_transition/input_layer_norm");
  transitionT(msa, (size_t)rowsN * t.L, C, S + "msa_transition", blk,
              (!t.opmFirst || moreMain || (extraStack && nextPairBlk > blk)) ? S + "outer_product_mean/layer_norm_input" : "",
              t.opmFirst ? blk + 1 : blk,
              (moreMain || (extraStack && nextPairBlk > blk)) ? S + "msa_row_attention_with_pair_bias/query_norm" : "", blk + 1,
              t.opmFirst ? "ln.m1" : nullptr);
  if (!t.opmFirst) outerProductMean(t, S, blk, (const T*)msa, pair, rowsN, C, mask, ones, S + "triangle_multiplication_outgoing/left_norm_input");
  // (each update's last GEMM emitting the next one's norm)
  triangleMultiplicationT(pair, t.pairMask, t.L, 128, S, blk, true, S + "triangle_multiplication_incoming/left_norm_input");
  triangleMultiplicationT(pair, t.pairMask, t.L, 128, S, blk, false, S + "triangle_attention_starting_node/query_norm");
  triangleAttentionT(pair, t.pairMask, t.L, 128, S, blk, true, t.pairOnes, S + "triangle_attention_ending_node/query_norm");
  triangleAttentionT(pair, t.pairMask, t.L, 128, S, blk, false, t.pairOnes, S + "pair_transition/input_layer_norm");
  transitionT(pair, (size_t)t.L * t.L, 128, S + "pair_transition", blk, nextPairNorm, nextPairBlk);
}
void evoformerBlock(Trunk& t, bool extraStack, int blk, const std::string& nextPairNorm, int nextPairBlk) {
  if (t.h.pair) evoformerBlockT(t, extraStack, blk, extraStack ? t.h.extra : t.h.msa, t.h.pair, nextPairNorm, nextPairBlk);
  else evoformerBlockT(t, extraStack, blk, extraStack ? t.extra : t.msa, t.pair, nextPairNorm, nextPairBlk);
}
void prepareWeights() {
  for (int ex = 0; ex < 2; ++ex) {
    const std::string S = ex ? "evoformer/extra_msa_stack/" : "evoformer/evoformer_iteration/";
    if (!M.has("w/" + S + "msa_transition/transition1/weights")) continue;
    const int C = ex ? 64 : 256;
    attnWeights(S + "msa_row_attention_with_pair_bias/attention", 0, C);
    if (!ex) attnWeights(S + "msa_column_attention/attention", 0, C);
    opmWeights(S + "outer_product_mean", 0, C, 32);
    attnWeights(S + "triangle_attention_starting_node/attention", 0, 128);
    attnWeights(S + "triangle_attention_ending_node/attention", 0, 128);
    triWeights(S + "triangle_multiplication_outgoing", 0, 128);
    triWeights(S + "triangle_multiplication_incoming", 0, 128);
  }
  for (const char* sub : {"template_embedding_iteration/", "template_pair_stack/__layer_stack_no_state/"}) {
    const std::string IT = std::string("evoformer/template_embedding/single_template_embedding/") + sub;
    if (!M.has("w/" + IT + "pair_transition/transition1/weights")) continue;
    attnWeights(IT + "triangle_attention_starting_node/attention", 0, 64);
    attnWeights(IT + "triangle_attention_ending_node/attention", 0, 64);
    triWeights(IT + "triangle_multiplication_outgoing", 0, 64);
    triWeights(IT + "triangle_multiplication_incoming", 0, 64);
  }
}
