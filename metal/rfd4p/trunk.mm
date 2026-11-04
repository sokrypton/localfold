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

void initPair(const float* pairFeats, const float* pairCondFeats, float* zInit) {
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
  a.x = normed; a.y = zInit; a.gamma = gamma; a.beta = beta;
  a.rows = nn; a.C = P; a.eps = 1e-5f;
  run("rfd4p_adaln_bcast", grid1d((nn + 7) / 8, 1), 256, a);
}

void tokenLayer(int layer, float* s, const float* z, const float* c) {
  const int n = D_.n, D = D_.D, P = D_.P, H = D_.H, dimHead = D_.D / D_.H;
  const size_t nn = (size_t)n * n;
  const std::string L = "nn.transformer_layers." + std::to_string(layer) + ".";
  const std::string MHA = L + "mhba.mha.";
  if (n > LF_ATTN_MAX_N) die("rfd4p: %d tokens is past this port's float32 attention (%d)", n, LF_ATTN_MAX_N);

  // ---- 1. the pair-biased gated attention, under its adaLN and output scale
  float* xa = scratch<float>("tl.xa", (size_t)n * D);
  adalnRows(s, c, L + "mhba.adaln", xa, n, D);
  float* nodeN = scratch<float>("tl.nodeN", (size_t)n * D);
  rmsNorm(xa, nodeN, n, D, M.f("w/" + MHA + "node_norm.weight"));

  float* qkv = scratch<float>("tl.qkv", (size_t)n * 3 * D);
  lin(nodeN, MHA + "to_qkv.weight", qkv, n, D, 3 * D, M.f("w/" + MHA + "to_qkv.bias"));
  // q and k are normed in place out of the packed row (`use_qkln`); v is read where it lies, by stride.
  float* q = scratch<float>("tl.q", (size_t)n * D);
  float* k = scratch<float>("tl.k", (size_t)n * D);
  RfdRmsArgs rq{}; rq.x = qkv; rq.y = q; rq.w = M.f("w/" + MHA + "q_layer_norm.weight");
  rq.rows = n; rq.C = D; rq.ldx = 3 * D; rq.ldy = D; rq.eps = 1e-5f;
  run("rfd4p_rms_norm", grid1d((size_t)(n + 7) / 8, 1), 256, rq);
  RfdRmsArgs rk = rq; rk.x = qkv + D; rk.y = k; rk.w = M.f("w/" + MHA + "k_layer_norm.weight");
  run("rfd4p_rms_norm", grid1d((size_t)(n + 7) / 8, 1), 256, rk);

  float* g = scratch<float>("tl.g", (size_t)n * D);
  lin(nodeN, MHA + "to_g.weight", g, n, D, D, M.f("w/" + MHA + "to_g.bias"));

  // the pair bias: this layer's norm of the pair track, projected to one value a head
  float* zn = scratch<float>("tl.zn", nn * P);
  rmsNorm(z, zn, nn, P, M.f("w/" + MHA + "pair_norm.weight"));
  float* bias = scratch<float>("tl.bias", nn * H);
  lin(zn, MHA + "to_bias.weight", bias, nn, P, H);

  float* ctx = scratch<float>("tl.ctx", (size_t)n * D);
  RfdAttnArgs at{};
  at.q = q; at.k = k; at.v = qkv; at.bias = bias; at.g = g; at.out = ctx;
  at.n = (uint)n; at.heads = (uint)H; at.dimHead = (uint)dimHead; at.ldv = (uint)(3 * D); at.vOff = 2 * D;
  at.scale = 1.f / sqrtf((float)dimHead);
  dispatch("rfd4p_attention", &at, sizeof at, Grid{(uint32_t)n, (uint32_t)H, 1}, 256, (size_t)n * 4, "rfd4p_attn");

  float* attnOut = scratch<float>("tl.attnOut", (size_t)n * D);
  lin(ctx, MHA + "to_out_node.weight", attnOut, n, D, D, M.f("w/" + MHA + "to_out_node.bias"));
  scaleAdd(s, attnOut, c, L + "mhba.scale_output", n, D);

  // ---- 2. the SwiGLU transition, under its own adaLN and output scale
  adalnRows(s, c, L + "transition.adaln", xa, n, D);
  float* hid = scratch<float>("tl.hid", (size_t)n * 8 * D);
  float* gated = scratch<float>("tl.gated", (size_t)n * 4 * D);
  float* tr = scratch<float>("tl.tr", (size_t)n * D);
  transition(xa, tr, n, D, 4, L + "transition.transition", hid, gated);
  scaleAdd(s, tr, c, L + "transition.scale_output", n, D);
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
static void triangle(const std::string& prefix, float* z, bool outgoing) {
  const int n = D_.n, P = D_.P, T = D_.T;
  const size_t nn = (size_t)n * n;
  const std::string W = "w/" + prefix + ".";

  // 🔴 LayerNorm, NOT RMSNorm, AND WITH ITS BIAS. The trunk's token side is RMSNorm throughout (`norm_type:
  // rmsnorm_torch_fused`) and the triangle's three norms are plain LayerNorms - mixing them up is a 1e-1 error
  // that still produces a plausible structure.
  float* zn = scratch<float>("tri.zn", nn * P);
  layerNorm(z, zn, nn, P, M.f(W + "layer_norm_in.weight"), opt(prefix + ".layer_norm_in.bias"));

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
  float* outv = scratch<float>("tri.out", nn * P);
  RfdMulSigmoidArgs ms{};
  ms.x = proj; ms.g = gate; ms.y = outv; ms.n = nn * P;
  run1d("rfd4p_mul_sigmoid", nn * P, ms);
  add(z, outv, nn * P);
}

void pairUpdate(int layer, const float* s, float* z) {
  const int n = D_.n, D = D_.D, P = D_.P;
  const size_t nn = (size_t)n * n;
  const std::string Q = "nn.pair_update_layers." + std::to_string(layer) + ".";

  // ---- the pair write: the single track's rows, projected once and written as an outer product
  float* xn = scratch<float>("pu.xn", (size_t)n * D);
  layerNorm(s, xn, n, D, M.f("w/" + Q + "layer_norm_in.weight"), M.f("w/" + Q + "layer_norm_in.bias"));
  float* pij = scratch<float>("pu.pij", (size_t)n * 2 * P);
  lin(xn, Q + "linear_x.weight", pij, n, D, 2 * P);
  float* zw = scratch<float>("pu.zw", nn * P);
  RfdPairWriteArgs pw{};
  pw.pair = z; pw.pij = pij; pw.z = zw; pw.rows = nn; pw.n = (uint)n; pw.P = (uint)P;
  run1d("rfd4p_pair_write", nn * P, pw);
  copy(z, zw, nn * P * 4);

  // ---- the two triangles, each added in, the second reading what the first wrote
  triangle(Q + "tri_mult_out", z, true);
  triangle(Q + "tri_mult_in", z, false);

  // ---- the pair transition
  const std::string TR = Q + "transition_out.";
  const int hidden = (int)M.len("w/" + TR + "linear_1.bias");
  float* tn = scratch<float>("pu.tn", nn * P);
  layerNorm(z, tn, nn, P, M.f("w/" + TR + "layer_norm.weight"), M.f("w/" + TR + "layer_norm.bias"));
  float* h = scratch<float>("pu.h", nn * (size_t)hidden);
  lin(tn, TR + "linear_1.weight", h, nn, P, hidden, M.f("w/" + TR + "linear_1.bias"), true);
  float* tr = scratch<float>("pu.tr", nn * P);
  lin(h, TR + "linear_2.weight", tr, nn, hidden, P, M.f("w/" + TR + "linear_2.bias"));
  add(z, tr, nn * P);
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
void recycleIn(const float* sInit, const float* zInit, float* s, float* z) {
  const int n = D_.n, D = D_.D, P = D_.P;
  const size_t nn = (size_t)n * n;
  float* sn = scratch<float>("rc.sn", (size_t)n * D);
  rmsNorm(s, sn, n, D, M.f("w/nn.s_cycle_norm.weight"));
  float* sd = scratch<float>("rc.sd", (size_t)n * D);
  lin(sn, "nn.s_cycle_proj.weight", sd, n, D, D);
  copy(s, sInit, (size_t)n * D * 4);
  add(s, sd, (size_t)n * D);

  float* zn = scratch<float>("rc.zn", nn * P);
  rmsNorm(z, zn, nn, P, M.f("w/nn.z_cycle_norm.weight"));
  float* zd = scratch<float>("rc.zd", nn * P);
  lin(zn, "nn.z_cycle_proj.weight", zd, nn, P, P);
  copy(z, zInit, nn * P * 4);
  add(z, zd, nn * P);
}

// The whole trunk: the three init products, `nRecycle` cycles of 14 layers with the pair update after every even
// one but the last, and the heads. `s` and `z` are the caller's state buffers and must start at ZERO (the first
// cycle's input is the init product because RMSNorm(0) is 0 and the projections have no bias); `insertion` may be
// null, which is what every step but the last passes.
void trunkForward(const float* seqF, const float* cond0, const float* cond1, const float* pairF,
                  const float* pairCondF, int nRecycle, float* s, float* z, float* bbCa, float* latents,
                  float* logits, float* insertion) {
  const int n = D_.n;
  const size_t nn = (size_t)n * n;
  float* c = scratch<float>("tf.c", (size_t)n * D_.C);
  float* c1 = scratch<float>("tf.c1", (size_t)n * D_.C);
  float* sInit = scratch<float>("tf.sInit", (size_t)n * D_.D);
  float* zInit = scratch<float>("tf.zInit", nn * D_.P);
  condC(cond0, c, n);
  if (nRecycle > 1) condC(cond1, c1, n);
  initSingle(seqF, sInit);
  initPair(pairF, pairCondF, zInit);
  for (int cycle = 0; cycle < nRecycle; ++cycle) {
    recycleIn(sInit, zInit, s, z);
    const float* cc = cycle == 0 ? c : c1;
    for (int i = 0; i < D_.layers; ++i) {
      tokenLayer(i, s, z, cc);
      if (i < D_.layers - 1 && i % 2 == 0) pairUpdate(i, s, z);
    }
  }
  heads(s, bbCa, latents, logits, insertion);
}
