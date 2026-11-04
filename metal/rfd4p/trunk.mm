// The trunk's initial products: the conditioning row, the single track's start, and the pair track's.
//
// `cpu/rfd4p/reference.js`'s `forwardTrunkCpu` is the spec, op for op:
//   c      = transition_c_2(transition_c_1(cond_factory(cond features)))
//   s_init = init_repr_factory(sequence features)
//   z_init = AdaLN(RMSNorm_ln_out(pair_repr_builder(pair features)), RMSNorm_ln_out(cond_factory(pair cond)))
//
// 🔴 NO MASKS, AND THAT IS THE INFERENCE PATH RATHER THAN A SHORTCUT: one design is generated at a time with no
// padding, so the padding mask is all ones and every row is real. webgpu/rfd4p/trunk.js says the same in its header
// and its kernels apply none either; the emit masks belong to the sampler's output, not here.
#include "rfd4p.h"
#include <cmath>

// 🔴 THE FUSED GATED ADD IS SILENTLY DROPPED FOR SOME SHAPES, which cost an hour and a wrong structure. core
// chooses its epilogue by the output width (and the operands' alignment): at `out` a multiple of 128 it applies
// `pair += aux * sigmoid(X W)`, and at 33489 it ran the PLAIN product instead and said so only in the profile's
// kernel name (`e0` where the working calls read `e128`). So the gate was never applied, the structure was
// wrong, and it was slower than what it replaced. Every call here goes through this check, which refuses rather
// than letting a future width lose its gate in silence.
static bool fusedGateFits(int out) { return out == 64 || out % 128 == 0; }


void condC(const float* condFeats, float* c, size_t rows) {
  const int C = D_.C, inner = 2 * C;
  float* tmp = scratch<float>("c.tmp", rows * C);
  float* hid = scratch<float>("c.hid", rows * 2 * inner);
  float* gated = scratch<float>("c.gated", rows * inner);
  lin(condFeats, "nn.cond_factory.linear_out.weight", c, rows, D_.condDim, C);
  transition(c, tmp, rows, C, 2, "nn.transition_c_1", hid, gated);
  transition(tmp, c, rows, C, 2, "nn.transition_c_2", hid, gated);
}

void initSingle(const float* seqFeats, float* sInit) {
  lin(seqFeats, "nn.init_repr_factory.linear_out.weight", sInit, D_.n, D_.seqDim, D_.D);
}

void initPair(const float* pairFeats, const float* pairCondFeats, Z zInit) {
  const int P = D_.P, C = D_.C;
  const size_t nn = (size_t)D_.n * D_.n;
  const std::string PB = "nn.pair_repr_builder.";

  // the pair features projected, then its own RMSNorm (`ln_out`)
  float* raw = scratch<float>("z.raw", nn * P);
  lin(pairFeats, PB + "init_repr_factory.linear_out.weight", raw, nn, D_.pairDim, P);
  float* normed = scratch<float>("z.normed", nn * P);
  rmsNorm(raw, normed, nn, P, M.f("w/" + PB + "init_repr_factory.ln_out.weight"));

  // the pair conditioning: ONE row, projected and normed, then its own norm inside the AdaLN
  float* zc = scratch<float>("z.cond", C);
  lin(pairCondFeats, PB + "cond_factory.linear_out.weight", zc, 1, D_.pairCondDim, C);
  float* zcn = scratch<float>("z.condn", C);
  rmsNorm(zc, zcn, 1, C, M.f("w/" + PB + "cond_factory.ln_out.weight"));
  float* zcn2 = scratch<float>("z.condn2", C);
  rmsNorm(zcn, zcn2, 1, C, M.f("w/" + PB + "adaln.norm_cond.weight"));

  // gamma (with its bias) and beta, one row each, broadcast over every pair row by the kernel
  float* gamma = scratch<float>("z.gamma", P);
  float* beta = scratch<float>("z.beta", P);
  lin(zcn2, PB + "adaln.to_gamma.0.weight", gamma, 1, C, P, M.f("w/" + PB + "adaln.to_gamma.0.bias"));
  lin(zcn2, PB + "adaln.to_beta.weight", beta, 1, C, P);

  // 🔴 THE SECOND NORM OF THE PAIR TRACK IS THE AdaLN's OWN AND HAS NO WEIGHT. `adaptiveLayerNorm` normalises its
  // input again (`applyNorm(x, ..., null, null)`) before scaling it, so z_init is normed TWICE: once by `ln_out`
  // above and once here. Fused into the scale so the second normalised copy is never written.
  RfdAdalnArgs a{};
  a.x = normed; a.y = zInit.f; a.yh = zInit.h; a.gamma = gamma; a.beta = beta;
  a.rows = nn; a.C = P; a.eps = 1e-5f;
  run("rfd4p_adaln_bcast", grid1d((nn + 7) / 8, 1), 256, a);
}

void tokenLayer(int layer, float* s, Z z, const float* c, const half* cH, const float* gb) {
  const int n = D_.n, D = D_.D, P = D_.P, H = D_.H, dimHead = D_.D / D_.H;
  const size_t nn = (size_t)n * n;
  const std::string L = "nn.transformer_layers." + std::to_string(layer) + ".";
  const std::string MHA = L + "mhba.mha.";
  if (n > LF_ATTN_MAX_N) die("rfd4p: %d tokens is past this port's float32 attention (%d)", n, LF_ATTN_MAX_N);

  // ---- 1. the pair-biased gated attention, under its adaLN and output scale
  float* xa = scratch<float>("tl.xa", (size_t)n * D);
  // the layer's window into the precomputed conditioning, where the caller has it
  if (gb) adalnPre(s, gb, layer, 0, D_.layers, xa, n, D);
  else adalnRows(s, c, L + "mhba.adaln", xa, n, D);
  // 🔴 THE NORM WRITES HALF WHERE THE NEXT GEMM WANTS HALF. Handing core an f32 operand with `half` asked for
  // means it stages a converted copy first: 205 such passes were 9.6% of a step.
  float* nodeN = scratch<float>("tl.nodeN", (size_t)n * D);
  half* nodeNh = HALF ? scratch<half>("tl.nodeNh", (size_t)n * D) : nullptr;
  if (HALF) {
    RfdRmsArgs r{};
    r.x = xa; r.y = nullptr; r.yh = nodeNh; r.w = M.f("w/" + MHA + "node_norm.weight");
    r.rows = n; r.C = D; r.eps = 1e-5f;
    run("rfd4p_rms_norm", grid1d((size_t)(n + 7) / 8, 1), 256, r);
  } else {
    rmsNorm(xa, nodeN, n, D, M.f("w/" + MHA + "node_norm.weight"));
  }

  // 🔴 ONE PACKED [n][4 D] BUFFER UNDER `--f16`, BECAUSE THAT IS WHAT metal/core's FLASH ATTENTION READS AND
  // THE PACKING IS FREE. It wants q | k | v | gate at 0, W, 2W, 3W of each position's row with W = heads *
  // dimHead, which here IS D - so `to_qkv`'s three contiguous output blocks ARE q | k | v in that layout, and
  // all the change takes is an output row stride of 4 D and the gate's own product written at 3 D. No repacking
  // pass, and the q and k norms run in place: a lane reads and writes the one channel it reduced.
  half* qkvg = HALF ? scratch<half>("tl.qkvg", (size_t)n * 4 * D) : nullptr;
  float* qkv = HALF ? nullptr : scratch<float>("tl.qkv", (size_t)n * 3 * D);
  float* q = nullptr;
  float* k = nullptr;
  float* g = nullptr;
  if (HALF) {
    linH(nodeNh, MHA + "to_qkv.weight", qkvg, n, D, 3 * D, M.f("w/" + MHA + "to_qkv.bias"), false, 4 * D);
    linH(nodeNh, MHA + "to_g.weight", qkvg + 3 * D, n, D, D, M.f("w/" + MHA + "to_g.bias"), false, 4 * D);
    RfdRmsArgs rq{}; rq.xh = qkvg; rq.yh = qkvg; rq.w = M.f("w/" + MHA + "q_layer_norm.weight");
    rq.rows = n; rq.C = D; rq.ldx = 4 * D; rq.ldy = 4 * D; rq.eps = 1e-5f;
    run("rfd4p_rms_norm", grid1d((size_t)(n + 7) / 8, 1), 256, rq);
    RfdRmsArgs rk = rq; rk.xh = qkvg + D; rk.yh = qkvg + D; rk.w = M.f("w/" + MHA + "k_layer_norm.weight");
    run("rfd4p_rms_norm", grid1d((size_t)(n + 7) / 8, 1), 256, rk);
  } else {
    lin(nodeN, MHA + "to_qkv.weight", qkv, n, D, 3 * D, M.f("w/" + MHA + "to_qkv.bias"));
    // q and k are normed in place out of the packed row (`use_qkln`); v is read where it lies, by stride.
    q = scratch<float>("tl.q", (size_t)n * D);
    k = scratch<float>("tl.k", (size_t)n * D);
    RfdRmsArgs rq{}; rq.x = qkv; rq.y = q; rq.w = M.f("w/" + MHA + "q_layer_norm.weight");
    rq.rows = n; rq.C = D; rq.ldx = 3 * D; rq.ldy = D; rq.eps = 1e-5f;
    run("rfd4p_rms_norm", grid1d((size_t)(n + 7) / 8, 1), 256, rq);
    RfdRmsArgs rk = rq; rk.x = qkv + D; rk.y = k; rk.w = M.f("w/" + MHA + "k_layer_norm.weight");
    run("rfd4p_rms_norm", grid1d((size_t)(n + 7) / 8, 1), 256, rk);
    g = scratch<float>("tl.g", (size_t)n * D);
    lin(nodeN, MHA + "to_g.weight", g, n, D, D, M.f("w/" + MHA + "to_g.bias"));
  }

  // the pair bias: this layer's norm of the pair track, projected to one value a head
  float* bias = HALF ? nullptr : scratch<float>("tl.bias", nn * H);
  half* biasH = HALF ? scratch<half>("tl.biasH", (size_t)H * nn) : nullptr;
  if (HALF) {
    half* znh = scratch<half>("tl.znh", nn * P);
    RfdRmsArgs r{};
    r.x = z.f; r.xh = z.h; r.y = nullptr; r.yh = znh; r.w = M.f("w/" + MHA + "pair_norm.weight");
    r.rows = nn; r.C = P; r.eps = 1e-5f;
    run("rfd4p_rms_norm", grid1d((nn + 7) / 8, 1), 256, r);
    // 🔴 HEAD-MAJOR AND IN LOG2 UNITS STRAIGHT OUT OF THE PROJECTION, BY SWAPPING ITS OPERANDS. core's flash
    // attention reads the bias as [H][n][n] where this writes it pair-major with the head innermost, and the
    // obvious transpose kernel is a disaster: its read is `bias[(q n + k) H + h]`, a gather at a stride of
    // twelve floats, and it cost 22.6 ms a step against the 8.0 ms of attention it was there to save - more
    // than the whole kernel it replaced. The product itself can be written the other way round, because
    // `Y[rows][out] = X[rows][in] W[in][out]` is symmetric in which operand is the weight: X is the projection
    // [H][P] and W is the normed pair track transposed, giving [H][n n] contiguous for nothing. `alpha` carries
    // the log2(e) the kernel's own `scale * log2(e)` expects of an added bias, so that is free too.
    Gemm gb2{};
    gb2.X = M.h("w/" + MHA + "to_bias.weight"); gb2.tx = F16;
    gb2.W = znh; gb2.tw = F16; gb2.transW = true;
    gb2.Y = biasH; gb2.ty = F16;
    gb2.rows = (size_t)H; gb2.in = P; gb2.out = nn;
    gb2.alpha = (float)M_LOG2E; gb2.half = true;
    gb2.label = "to_bias (head-major, log2)";
    gemm(gb2);
  } else {
    float* zn = scratch<float>("tl.zn", nn * P);
    rmsNorm(z.f, zn, nn, P, M.f("w/" + MHA + "pair_norm.weight"));
    lin(zn, MHA + "to_bias.weight", bias, nn, P, H);
  }

  float* ctx = HALF ? nullptr : scratch<float>("tl.ctx", (size_t)n * D);
  half* ctxH = HALF ? scratch<half>("tl.ctxH", (size_t)n * D) : nullptr;
  if (HALF) {
    // 🔴 metal/core's FLASH ATTENTION, ON THE MATRIX UNITS, WHERE THIS PORT HAD ITS OWN SCALAR KERNEL. One
    // simdgroup a query and a Kahan dot product a key is a correct softmax and nowhere near the machine: at 165
    // tokens, 12 heads of 64, measured through `metal/bench attn` with its arms interleaved, core's kernel is
    // 0.108 ms against this port's 0.571 - 5.3x - and the attention is 6.4% of a sampler step. The bias has to
    // be head-major and in log2 units for it, which the projection above writes directly.
    Attention A{};
    A.qkvg = qkvg; A.out = ctxH; A.n = n; A.heads = H; A.D = dimHead; A.rows = 1;
    A.scale = 1.f / sqrtf((float)dimHead);
    A.bias = biasH; A.biasStride = n;
    attention(A);
  } else {
    RfdAttnArgs at{};
    at.q = q; at.k = k; at.v = qkv; at.bias = bias; at.g = g; at.out = ctx; at.outH = ctxH;
    at.n = (uint)n; at.heads = (uint)H; at.dimHead = (uint)dimHead; at.ldv = (uint)(3 * D); at.vOff = 2 * D;
    at.scale = 1.f / sqrtf((float)dimHead);
    dispatch("rfd4p_attention", &at, sizeof at, Grid{(uint32_t)n, (uint32_t)H, 1}, 256, (size_t)n * 4, "rfd4p_attn");
  }

  // 🔴 THE OUTPUT SCALE AND THE RESIDUAL ARE THE GEMM'S EPILOGUE WHERE THE CONDITIONING IS IN HALF: core's
  // `pair += aux * sigmoid(X W)` is `AdaptiveOutputScale` followed by the block's residual, exactly. Two of
  // these a layer, twenty-eight a trunk pass, each otherwise a [n][256]x[256][768] GEMM and a pass over s.
  if (HALF && cH) {
    half* attnOut = scratch<half>("tl.attnOutH", (size_t)n * D);
    linH(ctxH, MHA + "to_out_node.weight", attnOut, n, D, D, M.f("w/" + MHA + "to_out_node.bias"));
    if (!fusedGateFits(D)) die("rfd4p: the output scale cannot be fused at width %d", D);
    gemmGatedAdd(cH, weightT(L + "mhba.scale_output.to_adaln_zero_gamma.0.weight"), attnOut, s, n, D_.C, D,
                 M.f("w/" + L + "mhba.scale_output.to_adaln_zero_gamma.0.bias"));
  } else if (HALF) {
    float* attnOut = scratch<float>("tl.attnOut", (size_t)n * D);
    linHF(ctxH, MHA + "to_out_node.weight", attnOut, n, D, D, M.f("w/" + MHA + "to_out_node.bias"));
    scaleAdd(s, attnOut, c, L + "mhba.scale_output", n, D);
  } else {
    float* attnOut = scratch<float>("tl.attnOut", (size_t)n * D);
    lin(ctx, MHA + "to_out_node.weight", attnOut, n, D, D, M.f("w/" + MHA + "to_out_node.bias"));
    scaleAdd(s, attnOut, c, L + "mhba.scale_output", n, D);
  }

  // ---- 2. the SwiGLU transition, under its own adaLN and output scale
  if (HALF) {
    half* xah = scratch<half>("tl.xah", (size_t)n * D);
    if (gb) adalnPre(s, gb, layer, 1, D_.layers, xa, n, D, xah);
    else adalnRows(s, c, L + "transition.adaln", xa, n, D, 0, false, xah);
    half* hid = scratch<half>("tl.hidh", (size_t)n * 8 * D);
    half* gated = scratch<half>("tl.gatedh", (size_t)n * 4 * D);
    half* trh = scratch<half>("tl.trh", (size_t)n * D);
    transitionH(xah, trh, n, D, 4, L + "transition.transition", hid, gated);
    if (cH) {
      gemmGatedAdd(cH, weightT(L + "transition.scale_output.to_adaln_zero_gamma.0.weight"), trh, s, n, D_.C, D,
                   M.f("w/" + L + "transition.scale_output.to_adaln_zero_gamma.0.bias"));
    } else {
      float* tr = scratch<float>("tl.tr", (size_t)n * D);
      toFloat(trh, tr, (size_t)n * D);
      scaleAdd(s, tr, c, L + "transition.scale_output", n, D);
    }
  } else {
    float* tr = scratch<float>("tl.tr", (size_t)n * D);
    if (gb) adalnPre(s, gb, layer, 1, D_.layers, xa, n, D);
    else adalnRows(s, c, L + "transition.adaln", xa, n, D);
    float* hid = scratch<float>("tl.hid", (size_t)n * 8 * D);
    float* gated = scratch<float>("tl.gated", (size_t)n * 4 * D);
    transition(xa, tr, n, D, 4, L + "transition.transition", hid, gated);
    scaleAdd(s, tr, c, L + "transition.scale_output", n, D);
  }
}

// One triangle multiplication, added into `z` in place. `outgoing`: x[i][j][c] = sum_k a[i][k][c] b[j][k][c];
// incoming: x[i][j][c] = sum_k a[k][i][c] b[k][j][c] - the SAME bytes contracted with two of the GEMM's strides
// exchanged, so neither direction transposes any data.
//
// 🔴 CHANNEL-MAJOR PLANES AND ONE BATCHED GEMM, WHICH IS THE WHOLE REASON THE PAIR TRACK NEVER MATERIALISES
// O(n^3). The operands arrive as pair rows [n*n][T] and the contraction wants [T] separate [n][n] matrices, so
// `rfd4p_tri_ab` applies the gate and transposes in the same pass and `rfd4p_tri_gather` brings the result back.
//
// 🔴 AND IT HOLDS THE WHOLE PAIR TENSOR AT ONCE, which is what webgpu/rfd4p/trunk.js's `chunkPlan` exists not to
// do: T * n^2 floats is 3 MB at n = 54 and 369 MB at n = 600, times the five buffers here. The slicing goes in
// when a length needs it, with the bound measured on this machine rather than copied from the page's.
// `znPre`: the input's LayerNorm, already written by whatever produced `z` (metal/core's gated add can emit the
// next norm as part of its epilogue). `nextScale`/`nextOffset`/`nextOut`: the norm THIS triangle's gated add
// should emit for whatever reads `z` next - returns true where it did, and the caller then skips that norm.
// 🔴 TWO OF A PAIR UPDATE'S THREE LayerNorms ARE SOMEBODY ELSE'S EPILOGUE: the outgoing triangle's add produces
// the pair track that the incoming triangle immediately norms, and the incoming one's produces what the
// transition norms. Emitting them costs the GEMM nothing and saves a pass over 25 MB each.
static bool triangle(const std::string& prefix, Z z, bool outgoing, const half* znPre = nullptr,
                     const float* nextScale = nullptr, const float* nextOffset = nullptr,
                     half* nextOut = nullptr) {
  const int n = D_.n, P = D_.P, T = D_.T;
  const size_t nn = (size_t)n * n;
  const std::string W = "w/" + prefix + ".";

  // 🔴 LayerNorm, NOT RMSNorm, AND WITH ITS BIAS. The trunk's token side is RMSNorm throughout (`norm_type:
  // rmsnorm_torch_fused`) and the triangle's three norms are plain LayerNorms - mixing them up is a 1e-1 error
  // that still produces a plausible structure.
  // 🔴 AND UNDER `--f16` THE WHOLE TRIANGLE IS HALF FROM HERE TO THE GATE, which is where the time is: with an
  // f32 pair track the core staged every operand through `lf_to_half` - 289 conversions and 18.6% of a step -
  // and the contraction ran on the vector units because its operands were floats. Half from the norm down, the
  // conversions are gone and the contraction is what the matrix units take. The norm is computed in float
  // either way (core's LayerNorm reads f32 and writes half), so what is in half is the PRODUCTS, not the sums.
  if (HALF) {
    // 🔴 THE PROJECTIONS WRITE CHANNEL-MAJOR BY SWAPPING THE GEMM'S OPERANDS, so the triangle has no transpose
    // in it at all. The contraction wants [T] planes of n x n; the activations arrive as pair rows [n*n][T], and
    // a kernel that transposes them was the two most expensive dispatches in a step (10.8% and 9.5%, both pure
    // data movement of 17 MB a buffer with the writes a whole row apart and nothing coalescing). The identity is
    // that the same product comes out transposed when the roles swap:
    //     Y[t][r] = sum_p weight[t][p] zn[r][p]
    // is `rows = T, in = P, out = n*n` with the WEIGHT as X and the activations as a transposed W. The GEMM
    // writes the planes the contraction reads, and the gate below is elementwise in that layout.
    // 🔴 AND A BIAS DOES COME THROUGH, IN THE GATE RATHER THAN THE GEMM - which this comment used to deny. core's
    // own bias is per output COLUMN, and the column here is a pair row rather than a channel, so the GEMM cannot
    // carry one; but a per-channel bias in a channel-major layout is a per-ROW constant, and the gate below
    // already touches every element of all four planes. `tri_mult_bias` is false in this checkpoint so the
    // tensors are absent and the ordinary kernel runs; `opt` returning one picks the variant that adds it. The
    // refusal that stood here named a reason that was not true of the triangle.
    const float* baP = opt(prefix + ".linear_a_p.bias");
    const float* baG = opt(prefix + ".linear_a_g.bias");
    const float* bbP = opt(prefix + ".linear_b_p.bias");
    const float* bbG = opt(prefix + ".linear_b_g.bias");
    const bool anyBias = baP || baG || bbP || bbG;
    half* zn = const_cast<half*>(znPre);
    if (!zn) {
      zn = scratch<half>(outgoing ? "tri.znHo" : "tri.znHi", nn * P);
      if (z.h) layerNorm(z.h, zn, nn, P, M.f(W + "layer_norm_in.weight"), opt(prefix + ".layer_norm_in.bias"));
      else layerNorm(z.f, zn, nn, P, M.f(W + "layer_norm_in.weight"), opt(prefix + ".layer_norm_in.bias"));
    }
    // 🔴 AND THE OTHER ORIENTATION MEASURES THE SAME, which is worth recording because the reason to try it was
    // good and wrong. Channel-major has an M dimension of 4T = 1024, so core's 64-row tiles re-read the whole
    // 25.7 MB of `zn` sixteen times; pair-row-major reads it once and re-reads the 786 KB weight instead, at the
    // cost of a tiled transpose in the gate. Built and measured both: 167.1 ms a step against 166.6 - the same.
    // So this GEMM is not limited by its operand reuse (at 0.88 TFLOP/s it is not at the bandwidth of its 68 MB
    // output either, and the remaining 3x is in core's tiling for a K of 384, not in this port's layout).
    // 🔴 THE FOUR PROJECTIONS ARE ONE GEMM, because they share their input and differ only in the weight: the
    // four [T][P] matrices stacked into one [4T][P] (derived once) make a product four times the size and a
    // quarter of the dispatches. At 54 tokens each of them alone was 2916 x 256 x 384 - small enough that the
    // launch was a third of the step - and the port issued 1184 dispatches a step before this.
    half* ab4 = scratch<half>("tri.ab4H", nn * 4 * T);
    {
      Gemm g{};
      g.X = weightStack(prefix, {prefix + ".linear_a_p.weight", prefix + ".linear_a_g.weight",
                                 prefix + ".linear_b_p.weight", prefix + ".linear_b_g.weight"});
      g.tx = F16;                                        // the stacked weight [4T][P] as the row operand
      g.W = zn; g.tw = F16; g.transW = true;             // the activations [n*n][P] read as [out][in]
      g.Y = ab4; g.ty = F16;
      g.rows = (size_t)4 * T; g.in = P; g.out = (int)nn;
      g.accFloat = true;
      g.label = "tri-project";
      gemm(g);
    }
    half* aP = ab4;
    half* aG = ab4 + nn * T;
    half* bP = ab4 + 2 * nn * T;
    half* bG = ab4 + 3 * nn * T;

    half* aT = scratch<half>("tri.aTH", nn * T);
    half* bT = scratch<half>("tri.bTH", nn * T);
    RfdTriAbHArgs ab{};
    ab.aP = aP; ab.aG = aG; ab.bP = bP; ab.bG = bG; ab.aT = aT; ab.bT = bT; ab.rows = nn; ab.T = (uint)T;
    ab.baP = baP; ab.baG = baG; ab.bbP = bbP; ab.bbG = bbG;
    // channel-major in and out: a thread's read and its write are the same index, so both coalesce
    run1d(anyBias ? "rfd4p_tri_ab_hmb" : "rfd4p_tri_ab_hm", nn * T, ab);

    half* x = scratch<half>("tri.xH", nn * T);
    Gemm g{};
    g.X = aT; g.tx = F16; g.W = bT; g.tw = F16; g.Y = x; g.ty = F16;
    g.rows = n; g.in = n; g.out = n;
    g.ldx = n; g.ldw = n; g.ldy = n;
    g.batch = T; g.sx = (int64_t)nn; g.sw = (int64_t)nn; g.sy = (int64_t)nn;
    g.transW = outgoing;
    g.transX = !outgoing;
    // 🔴 ACCUMULATED IN FLOAT EVEN THOUGH THE OPERANDS ARE HALF: the contraction is over n terms of products of
    // gated activations, and a half accumulator loses the small ones.
    g.accFloat = true;
    g.label = outgoing ? "tri-out" : "tri-in";
    gemm(g);

    // 🔴 THE CENTRE NORM TRANSPOSES AND NORMALISES IN ONE PASS - metal/core's own kernel, written for exactly
    // this seam: a channel-major product in, pair rows of half out. It replaces a gather of 17 MB and a
    // LayerNorm over it.
    half* xn = scratch<half>("tri.xnH", nn * T);
    centerNorm(x, 1.f, xn, n, n, T, M.f(W + "layer_norm_out.weight"), opt(prefix + ".layer_norm_out.bias"));

    half* proj = scratch<half>("tri.projH", nn * P);
    linH(xn, prefix + ".linear_z.weight", proj, nn, T, P, opt(prefix + ".linear_z.bias"));
    // 🔴 THE GATE, THE PRODUCT AND THE RESIDUAL IN ONE GEMM - metal/core's own fused epilogue, which is exactly
    // this network's triangle tail: `pair += aux * sigmoid(X W)`. It replaces a [n*n][384] GEMM for the gate and
    // a pass over the pair track to apply it (that pass alone was 9.2% of a step, reading and writing 103 MB a
    // call). The weight goes in as [in][out], which this checkpoint does not store, so a transposed half copy is
    // derived ONCE - 294 KB against the 25 MB a call it saves.
    // 🔴 ITS RETURN VALUE IS NOT "DID IT RUN". `gemmGatedAdd` hands back whether it also emitted the NEXT
    // LayerNorm (metal/core's `Gemm::lnOut`), and reading it as a fallback flag ran the gate twice - the product
    // applied once by the GEMM and once by the pass that was meant to replace it. It read 2.4e-2 against the
    // reference and took longer than what it replaced, which is a useful shape of bug: a fallback that fires
    // when nothing failed. The call always runs; off the matrix units it is lf_gemm's extended instance, which
    // is slower there and not wrong.
    if (!fusedGateFits(P)) die("rfd4p: the triangle's gate cannot be fused at width %d", P);
    if (z.h)
      return gemmGatedAdd(zn, weightT(prefix + ".linear_g.weight"), proj, z.h, nn, P, P,
                          opt(prefix + ".linear_g.bias"), nextOut, nextScale, nextOffset, 1e-5f);
    return gemmGatedAdd(zn, weightT(prefix + ".linear_g.weight"), proj, z.f, nn, P, P,
                        opt(prefix + ".linear_g.bias"), nextOut, nextScale, nextOffset, 1e-5f);
  }

  float* zn = scratch<float>("tri.zn", nn * P);
  layerNorm(z.f, zn, nn, P, M.f(W + "layer_norm_in.weight"), opt(prefix + ".layer_norm_in.bias"));

  float* aP = scratch<float>("tri.aP", nn * T);
  float* aG = scratch<float>("tri.aG", nn * T);
  float* bP = scratch<float>("tri.bP", nn * T);
  float* bG = scratch<float>("tri.bG", nn * T);
  lin(zn, prefix + ".linear_a_p.weight", aP, nn, P, T, opt(prefix + ".linear_a_p.bias"));
  lin(zn, prefix + ".linear_a_g.weight", aG, nn, P, T, opt(prefix + ".linear_a_g.bias"));
  lin(zn, prefix + ".linear_b_p.weight", bP, nn, P, T, opt(prefix + ".linear_b_p.bias"));
  lin(zn, prefix + ".linear_b_g.weight", bG, nn, P, T, opt(prefix + ".linear_b_g.bias"));

  float* aT = scratch<float>("tri.aT", nn * T);
  float* bT = scratch<float>("tri.bT", nn * T);
  RfdTriAbArgs ab{};
  ab.aP = aP; ab.aG = aG; ab.bP = bP; ab.bG = bG; ab.aT = aT; ab.bT = bT; ab.rows = nn; ab.T = (uint)T;
  run1d("rfd4p_tri_ab", nn * T, ab);

  float* x = scratch<float>("tri.x", nn * T);
  Gemm g{};
  g.X = aT; g.tx = F32; g.W = bT; g.tw = F32; g.Y = x; g.ty = F32;
  g.rows = n; g.in = n; g.out = n;
  g.ldx = n; g.ldw = n; g.ldy = n;
  g.batch = T; g.sx = (int64_t)nn; g.sw = (int64_t)nn; g.sy = (int64_t)nn;
  // outgoing: Y[i][j] = sum_k X[i][k] W[j][k] (W read as [out][in]); incoming: Y[i][j] = sum_k X[k][i] W[k][j]
  g.transW = outgoing;
  g.transX = !outgoing;
  g.label = outgoing ? "tri-out" : "tri-in";
  gemm(g);

  float* xr = scratch<float>("tri.xr", nn * T);
  RfdTriGatherArgs ga{};
  ga.x = x; ga.out = xr; ga.rows = nn; ga.T = (uint)T;
  run1d("rfd4p_tri_gather", nn * T, ga);

  float* xn = scratch<float>("tri.xn", nn * T);
  layerNorm(xr, xn, nn, T, M.f(W + "layer_norm_out.weight"), opt(prefix + ".layer_norm_out.bias"));

  float* proj = scratch<float>("tri.proj", nn * P);
  float* gate = scratch<float>("tri.gate", nn * P);
  lin(xn, prefix + ".linear_z.weight", proj, nn, T, P, opt(prefix + ".linear_z.bias"));
  lin(zn, prefix + ".linear_g.weight", gate, nn, P, P, opt(prefix + ".linear_g.bias"));
  RfdMulSigmoidArgs ms{};
  ms.x = proj; ms.g = gate; ms.y = z.f; ms.n = nn * P;
  run1d("rfd4p_mul_sigmoid", nn * P, ms);
  return false;
}

void pairUpdate(int layer, const float* s, Z z) {
  const int n = D_.n, D = D_.D, P = D_.P;
  const size_t nn = (size_t)n * n;
  const std::string Q = "nn.pair_update_layers." + std::to_string(layer) + ".";

  // ---- the pair write: the single track's rows, projected once and written as an outer product
  float* xn = scratch<float>("pu.xn", (size_t)n * D);
  layerNorm(s, xn, n, D, M.f("w/" + Q + "layer_norm_in.weight"), M.f("w/" + Q + "layer_norm_in.bias"));
  float* pij = scratch<float>("pu.pij", (size_t)n * 2 * P);
  lin(xn, Q + "linear_x.weight", pij, n, D, 2 * P);
  // a threadgroup a pair row (see the kernel): the row index is the group's, not a division a thread
  if (z.h) {
    RfdPairWriteHArgs pw{};
    pw.pair = z.h; pw.pij = pij; pw.z = z.h; pw.rows = nn; pw.n = (uint)n; pw.P = (uint)P;
    dispatch("rfd4p_pair_write_h", &pw, sizeof pw, grid1d(nn, 1), 256, 0, "rfd4p_pair_write_h");
  } else {
    RfdPairWriteArgs pw{};
    pw.pair = z.f; pw.pij = pij; pw.z = z.f; pw.rows = nn; pw.n = (uint)n; pw.P = (uint)P;
    dispatch("rfd4p_pair_write", &pw, sizeof pw, grid1d(nn, 1), 256, 0, "rfd4p_pair_write");
  }

  // ---- the two triangles, each added in, the second reading what the first wrote - and each asking the GEMM
  // that writes the pair track to emit the norm of it that the next stage wants
  const std::string TRI_IN = Q + "tri_mult_in", TR_ = Q + "transition_out.";
  half* znIn = HALF ? scratch<half>("pu.znInH", nn * P) : nullptr;
  half* tnBuf = HALF ? scratch<half>("pu.tnh", nn * P) : nullptr;
  const bool haveZnIn = triangle(Q + "tri_mult_out", z, true, nullptr,
                                 HALF ? M.f("w/" + TRI_IN + ".layer_norm_in.weight") : nullptr,
                                 HALF ? opt(TRI_IN + ".layer_norm_in.bias") : nullptr, znIn);
  const bool haveTn = triangle(TRI_IN, z, false, haveZnIn ? znIn : nullptr,
                               HALF ? M.f("w/" + TR_ + "layer_norm.weight") : nullptr,
                               HALF ? M.f("w/" + TR_ + "layer_norm.bias") : nullptr, tnBuf);

  // ---- the pair transition
  const std::string TR = Q + "transition_out.";
  const int hidden = (int)M.len("w/" + TR + "linear_1.bias");
  if (HALF) {
    half* tn = tnBuf;
    if (!haveTn) {
      if (z.h) layerNorm(z.h, tn, nn, P, M.f("w/" + TR + "layer_norm.weight"), M.f("w/" + TR + "layer_norm.bias"));
      else layerNorm(z.f, tn, nn, P, M.f("w/" + TR + "layer_norm.weight"), M.f("w/" + TR + "layer_norm.bias"));
    }
    half* h = scratch<half>("pu.hh", nn * (size_t)hidden);
    linH(tn, TR + "linear_1.weight", h, nn, P, hidden, M.f("w/" + TR + "linear_1.bias"), true);
    float* tr = scratch<float>("pu.tr", nn * P);
    // 🔴 THE RESIDUAL IS THE GEMM'S OWN beta, so the pair track is read and written once rather than twice: the
    // separate add was 9.2% of a step across the updates.
    Gemm g{};
    g.X = h; g.tx = F16; g.W = M.h("w/" + TR + "linear_2.weight"); g.tw = F16; g.transW = true;
    if (z.h) { g.Y = z.h; g.ty = F16; g.Yin = z.h; } else { g.Y = z.f; g.ty = F32; g.Yin = z.f; }
    g.beta = 1.f;
    g.rows = nn; g.in = hidden; g.out = P;
    g.bias = M.f("w/" + TR + "linear_2.bias"); g.tbias = F32;
    g.label = "pair-transition-out";
    gemm(g);
    (void)tr;
    return;
  }
  float* tn = scratch<float>("pu.tn", nn * P);
  layerNorm(z.f, tn, nn, P, M.f("w/" + TR + "layer_norm.weight"), M.f("w/" + TR + "layer_norm.bias"));
  float* h = scratch<float>("pu.h", nn * (size_t)hidden);
  lin(tn, TR + "linear_1.weight", h, nn, P, hidden, M.f("w/" + TR + "linear_1.bias"), true);
  float* tr = scratch<float>("pu.tr", nn * P);
  lin(h, TR + "linear_2.weight", tr, nn, hidden, P, M.f("w/" + TR + "linear_2.bias"));
  add(z.f, tr, nn * P);
}

// The three heads and the insertion logits, each a norm and a projection of the finished single track.
//
// 🔴 THE INSERTION HEAD IS A PLAIN PROJECTION PAIR, `nn.insertion_q_proj.weight` and `..._k_proj.weight`, with no
// norm of any kind. Asking for `nn.insertion_q_proj.0.weight` - a norm/linear pair this checkpoint does not
// contain - is what made the CPU reference skip the head in silence for as long as it existed (upstream's
// 62276d95), which left an unindexed motif with no mapping to place it by. One matrix, read by name.
void heads(const float* s, float* bbCa, float* localLatents, float* logits, float* insertion) {
  const int n = D_.n, D = D_.D;
  float* sn = scratch<float>("hd.sn", (size_t)n * D);
  struct Head { const char* prefix; float* out; int width; };
  const Head hs[] = {{"nn.ca_linear", bbCa, 3}, {"nn.local_latents_linear", localLatents, 32},
                     {"nn.sequence_readout", logits, 147}};
  for (const Head& h : hs) {
    rmsNorm(s, sn, n, D, M.f(std::string("w/") + h.prefix + ".0.weight"));
    lin(sn, std::string(h.prefix) + ".1.weight", h.out, n, D, h.width, opt(std::string(h.prefix) + ".1.bias"));
  }
  if (!insertion) return;
  float* q = scratch<float>("hd.q", (size_t)n * D);
  float* k = scratch<float>("hd.k", (size_t)n * D);
  lin(s, "nn.insertion_q_proj.weight", q, n, D, D);
  lin(s, "nn.insertion_k_proj.weight", k, n, D, D);
  Gemm g{};
  g.X = q; g.tx = F32; g.W = k; g.tw = F32; g.transW = true; g.Y = insertion; g.ty = F32;
  g.rows = n; g.in = D; g.out = n; g.alpha = 1.f / sqrtf((float)D); g.label = "ins_logits";
  gemm(g);
}

// The recycle's input: s = s_init + Lin(RMSNorm(s)), z = z_init + Lin(RMSNorm(z)), in place on s and z.
//
// 🔴 CYCLE ZERO IS THIS SAME ARITHMETIC AND NOT A SPECIAL CASE: s and z start at zero, RMSNorm(0) is 0 (the norm
// has no bias) and the projections have none either, so the delta is zero and the cycle's input is the init
// product itself. Running it anyway is what keeps the two paths one path - and the caller may skip it on the
// first cycle because it is a measured no-op, not because the code would be wrong.
void recycleIn(const float* sInit, Z zInit, float* s, Z z) {
  const int n = D_.n, D = D_.D, P = D_.P;
  const size_t nn = (size_t)n * n;
  float* sn = scratch<float>("rc.sn", (size_t)n * D);
  rmsNorm(s, sn, n, D, M.f("w/nn.s_cycle_norm.weight"));
  float* sd = scratch<float>("rc.sd", (size_t)n * D);
  lin(sn, "nn.s_cycle_proj.weight", sd, n, D, D);
  copy(s, sInit, (size_t)n * D * 4);
  add(s, sd, (size_t)n * D);

  if (z.h) {
    // the cycle's own norm and projection, in half like the track they read and write
    half* zn = scratch<half>("rc.znH", nn * P);
    RfdRmsArgs r{};
    r.xh = z.h; r.yh = zn; r.w = M.f("w/nn.z_cycle_norm.weight");
    r.rows = nn; r.C = P; r.eps = 1e-5f;
    run("rfd4p_rms_norm", grid1d((nn + 7) / 8, 1), 256, r);
    half* zd = scratch<half>("rc.zdH", nn * P);
    linH(zn, "nn.z_cycle_proj.weight", zd, nn, P, P);
    RfdAddHArgs a{};
    a.a = zInit.h; a.b = zd; a.y = z.h; a.n = nn * P;
    run1d("rfd4p_add_h", nn * P, a);
    return;
  }
  float* zn = scratch<float>("rc.zn", nn * P);
  rmsNorm(z.f, zn, nn, P, M.f("w/nn.z_cycle_norm.weight"));
  float* zd = scratch<float>("rc.zd", nn * P);
  lin(zn, "nn.z_cycle_proj.weight", zd, nn, P, P);
  copy(z.f, zInit.f, nn * P * 4);
  add(z.f, zd, nn * P);
}

// The whole trunk: the three init products, `nRecycle` cycles of 14 layers with the pair update after every even
// one but the last, and the heads. `s` and `z` are the caller's state buffers and must start at ZERO (the first
// cycle's input is the init product because RMSNorm(0) is 0 and the projections have no bias); `insertion` may be
// null, which is what every step but the last passes.
void trunkForward(const float* seqF, const float* cond0, const float* cond1, const float* pairF,
                  const float* pairCondF, int nRecycle, float* s, Z z, float* bbCa, float* latents,
                  float* logits, float* insertion) {
  const int n = D_.n;
  const size_t nn = (size_t)n * n;
  float* c = scratch<float>("tf.c", (size_t)n * D_.C);
  float* c1 = scratch<float>("tf.c1", (size_t)n * D_.C);
  float* sInit = scratch<float>("tf.sInit", (size_t)n * D_.D);
  Z zInit;
  if (z.h) zInit.h = scratch<half>("tf.zInitH", nn * D_.P);
  else zInit.f = scratch<float>("tf.zInit", nn * D_.P);
  condC(cond0, c, n);
  if (nRecycle > 1) condC(cond1, c1, n);
  initSingle(seqF, sInit);
  initPair(pairF, pairCondF, zInit);
  // the conditioning in half once a trunk pass, for the fused output scales: n x 256 values, against the
  // twenty-eight GEMMs and twenty-eight passes over `s` it saves
  half* cH = HALF ? scratch<half>("tf.cH", (size_t)n * D_.C) : nullptr;
  half* c1H = HALF && nRecycle > 1 ? scratch<half>("tf.c1H", (size_t)n * D_.C) : nullptr;
  if (cH) toHalf(c, cH, (size_t)n * D_.C);
  if (c1H) toHalf(c1, c1H, (size_t)n * D_.C);
  // every layer's adaLN conditioning, once a cycle rather than six dispatches a layer
  const float* gb = precomputeCond(c, n, D_.layers, D_.D, D_.C);

  for (int cycle = 0; cycle < nRecycle; ++cycle) {
    recycleIn(sInit, zInit, s, z);
    const float* cc = cycle == 0 ? c : c1;
    const half* ccH = cycle == 0 ? cH : c1H;
    // 🔴 THE SECOND CYCLE'S CONDITIONING IS A DIFFERENT `c`, so its precompute is its own: computed on entry to
    // the cycle rather than once, because `c1` is what every later cycle uses.
    const float* ccGb = cycle == 0 ? gb : precomputeCond(c1, n, D_.layers, D_.D, D_.C);
    for (int i = 0; i < D_.layers; ++i) {
      tokenLayer(i, s, z, cc, ccH, ccGb);
      if (i < D_.layers - 1 && i % 2 == 0) pairUpdate(i, s, z);
    }
  }
  heads(s, bbCa, latents, logits, insertion);
}
