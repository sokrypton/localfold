// BioEmu2's atom side, its two attentions, the conditioned transformer layer and the input embedder
// (cuda/bioemu2/src/layers.cuh, attention.cuh and trunk.cuh's inputEmbedder are the reading).
#include "bioemu2.h"

// ---------------------------------------------------------------- the atoms and their windowed pair
void atomFeatures(const std::string& p, float* c, size_t N) {
  const int C = dimW(p + "proj_atom_continuous/weight", 0);
  run1d("be2_atom_features", N * C,
        Be2AtomFeatArgs{Fin("template_coords"), Fin("charge"), Iin("element"), Iin("name_encoding"), Iin("atom_pad_mask"),
                        Wf(p + "proj_atom_continuous/weight"), Wf(p + "embed_element/weight"),
                        Wf(p + "embed_atom_name_chars_0/weight"), Wf(p + "embed_atom_name_chars_1/weight"),
                        Wf(p + "embed_atom_name_chars_2/weight"), Wf(p + "embed_atom_name_chars_3/weight"), c, N, (uint)C, 0});
}
void pairGeometry(const std::string& p, float* pr, int N) {
  run1d("be2_pair_geometry", (size_t)N * WK,
        Be2PairGeomArgs{Fin("template_coords"), Iin("ref_space_uid"), Iin("atom_pad_mask"), Wf(p + "embed_atompair_ref_pos/weight"),
                        Wf(p + "embed_atompair_ref_dist/weight"), Wf(p + "embed_atompair_mask/weight"), pr, (uint)N,
                        (uint)dimW(p + "embed_atompair_ref_pos/weight", 0)});
}
// _atom_pair_tail: pr += cq[q] + ck[k], then pr += p_mlp(pr)
void pairTail(const std::string& p, float* pr, const float* c, int N) {
  const int C = dimW(p + "c_to_p_trans_q_1/weight", 1), CP = dimW(p + "c_to_p_trans_q_1/weight", 0);
  if (CP != 16) die("BioEmu2: an atom pair of %d channels (this port is written for 16)", CP);
  float* rc = scratch<float>("be2.pt.relu", (size_t)N * C);
  relu(c, rc, (size_t)N * C);
  float* cq = scratch<float>("be2.pt.cq", (size_t)N * CP); float* ck = scratch<float>("be2.pt.ck", (size_t)N * CP);
  linT(rc, p + "c_to_p_trans_q_1/weight", cq, N, C, CP);
  linT(rc, p + "c_to_p_trans_k_1/weight", ck, N, C, CP);
  run1d("be2_pair_tail", (size_t)N * WK, Be2PairTailArgs{pr, cq, ck, Wf(p + "p_mlp_1/weight"), Wf(p + "p_mlp_3/weight"),
                                                         Wf(p + "p_mlp_5/weight"), (uint)N, 0});
}
void windowBias(const float* pr, const float* normW, const float* projZ, float* out, int N, int H) {
  run1d("be2_window_bias", (size_t)N * WK, Be2WindowBiasArgs{pr, normW, projZ, out, (uint)N, (uint)H});
}

// ---------------------------------------------------------------- the conditioned transformer layer
float* layerCond(const CondLayer& L, const float* c, int Rc, int Cc, int D, const char* name) {
  float* cn = scratch<float>("be2.cl.cn", (size_t)Rc * Cc);
  layerNorm(c, cn, Rc, Cc, Wf(L.p + "cond_0/weight"), WfOpt(L.p + "cond_0/bias"));
  silu(cn, cn, (size_t)Rc * Cc);
  float* cond = scratch<float>(name, (size_t)Rc * 6 * D);
  linT(cn, L.p + "cond_2/weight", cond, Rc, Cc, 6 * D, 0.f, 0, 0, false, WfOpt(L.p + "cond_2/bias"));
  return cond;
}
void adaLn(const float* x, float* y, size_t rows, int C, const float* alpha, const float* beta, int ldc, const int* map,
           half* yh) {
  const size_t groups = (rows + 7) / 8;
  run("be2_ada_ln", Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1}, 256,
      Be2AdaLnArgs{x, y, alpha, beta, map, rows, (uint)C, (uint)C, (uint)ldc, 1.f, yh});
}
// cond: [6D] a row - alpha_a, beta_a, gamma_a, alpha_m, beta_m, gamma_m
// the attention's q | k | v | g as one torch weight [4D][D] in half: proj_qkv's rows then proj_g's - the layout
// metal/core's flash attention reads (per position q, k, v, g, each heads x head width)
static const half* qkvgWeight(const std::string& a, int D) {
  return M.derived<half>("qkvg:" + a, (size_t)4 * D * D, [&](half* out) {
    run1d("be2_stack", (size_t)4 * D * D, Be2StackArgs{M.h("w/" + a + "proj_qkv/weight"), M.h("w/" + a + "proj_g/weight"), out,
                                                       (u64)3 * D * D, (u64)D * D});
  });
}
// the attention on the fast arm: adaLN into half, q|k|v|g in one GEMM, either be2_window_attention_h (windowed atoms:
// half K/V in 16 KB shared memory, sigmoid(g) in the epilogue) or metal/core's flash attention (tokens), and the
// output projection from half into x with the gated residual in its epilogue where metal/core takes it
static void attentionHalf(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map,
                          const float* bias, const half* biasH, int perSample) {
  const std::string a = L.p + L.att;
  half* hh = scratch<half>("be2.cl.hh", (size_t)R * D);
  adaLn(x, nullptr, R, D, cond, cond + D, 6 * D, map, hh);
  half* qkvg = scratch<half>("be2.cl.qkvg", (size_t)R * 4 * D);
  Gemm g{}; g.X = hh; g.tx = F16; g.W = qkvgWeight(a, D); g.tw = F16; g.transW = true; g.Y = qkvg; g.ty = F16;
  g.rows = R; g.in = D; g.out = 4 * D; g.accFloat = true; g.label = L.windowed ? "atom qkvg" : "token qkvg";
  gemm(g);
  half* o = scratch<half>("be2.cl.oh", (size_t)R * D);
  if (L.windowed) {
    if (D / L.heads != 32) die("BioEmu2: an atom head of %d (this port is written for 32)", D / L.heads);
    if (R % WQ || perSample % WQ) die("BioEmu2: %d atom rows are not whole windows", R);
    run("be2_window_attention_h", Grid{(uint32_t)(R / WQ), (uint32_t)L.heads, 1}, WQ * 4,
        Be2WindowAttnHArgs{qkvg, Wf(a + "proj_q_bias"), bias, Iin("atom_pad_mask"), o, (uint)perSample, (uint)L.heads});
  } else {
    Attention at{}; at.qkvg = qkvg; at.out = o; at.n = perSample; at.heads = L.heads; at.D = D / L.heads; at.rows = R / perSample;
    at.scale = 1.f / sqrtf((float)(D / L.heads)); at.bias = biasH; at.biasStride = perSample; at.qBias = Wf(a + "proj_q_bias");
    // (every token is real here - no token padding - so no key mask; a padded input would need token_pad_mask as float)
    attention(at);
  }
  Gemm G{}; G.X = o; G.tx = F16; G.W = M.h("w/" + a + "proj_o/weight"); G.tw = F16; G.transW = true; G.Y = x; G.ty = F32;
  G.rows = R; G.in = D; G.out = D; G.label = L.windowed ? "atom proj_o" : "token proj_o";
  if (gemmGatedRows(G, cond + 2 * D, 6 * D, map)) return;
  float* h = scratch<float>("be2.cl.h", (size_t)R * D);
  G.Y = h; gemm(G);
  be2Run1d("be2_gated_add", ((size_t)R * D) >> 2, Be2GatedAddArgs{x, h, cond + 2 * D, map, (size_t)R, (uint)D, (uint)(6 * D)});
}
// the attention half of a layer on the reference arm: adaLN, q|k|v and g, the attention, the gate, the output projection into h
static void attentionF32(const CondLayer& L, const float* x, float* h, int R, int D, const float* cond, const int* map,
                         const float* bias, int perSample) {
  const std::string a = L.p + L.att;
  adaLn(x, h, R, D, cond, cond + D, 6 * D, map);
  float* qkv = scratch<float>("be2.cl.qkv", (size_t)R * 3 * D);
  float* g = scratch<float>("be2.cl.g", (size_t)R * D);
  float* o = scratch<float>("be2.cl.o", (size_t)R * D);
  linT(h, a + "proj_qkv/weight", qkv, R, D, 3 * D);
  linT(h, a + "proj_g/weight", g, R, D, D);
  if (L.windowed) {
    if (D / L.heads != 32) die("BioEmu2: an atom head of %d (this port is written for 32)", D / L.heads);
    if (R % WQ || perSample % WQ) die("BioEmu2: %d atom rows are not whole windows", R);
    run("be2_window_attention", Grid{(uint32_t)(R / WQ), (uint32_t)L.heads, 1}, WQ * 4,
        Be2WindowAttnArgs{qkv, Wf(a + "proj_q_bias"), bias, Iin("atom_pad_mask"), o, (uint)perSample, (uint)L.heads});
  } else {
    if (D / L.heads > 64) die("BioEmu2: a token head of %d (at most 64 here)", D / L.heads);
    const int samples = R / perSample;
    run1d("be2_token_attention", (size_t)samples * L.heads * perSample,
          Be2TokenAttnArgs{qkv, Wf(a + "proj_q_bias"), bias, Iin("token_pad_mask"), o, (uint)perSample, (uint)D,
                           (uint)L.heads, (uint)samples});
  }
  run1d("be2_sigmoid_mul", (size_t)R * D, Be2SigmoidMulArgs{o, g, o, (size_t)R * D});
  linT(o, a + "proj_o/weight", h, R, D, D);
}
void condLayer(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map, const float* bias, int perSample,
               const half* biasH) {
  if (HALF && (L.windowed || biasH)) {
    attentionHalf(L, x, R, D, cond, map, bias, biasH, perSample);
  } else {
    float* h = scratch<float>("be2.cl.h", (size_t)R * D);
    attentionF32(L, x, h, R, D, cond, map, bias, perSample);
    be2Run1d("be2_gated_add", ((size_t)R * D) >> 2, Be2GatedAddArgs{x, h, cond + 2 * D, map, (size_t)R, (uint)D, (uint)(6 * D)});
  }
  if (HALF) {
    half* hh = scratch<half>("be2.cl.hh", (size_t)R * D);
    adaLn(x, nullptr, R, D, cond + 3 * D, cond + 4 * D, 6 * D, map, hh);
    transitionHalf(hh, x, R, D, L.p + "transition/", false, cond + 5 * D, 6 * D, map);    // (x += gamma fc3(...))
  } else {
    float* h = scratch<float>("be2.cl.h", (size_t)R * D);
    float* t = scratch<float>("be2.cl.t", (size_t)R * D);
    adaLn(x, h, R, D, cond + 3 * D, cond + 4 * D, 6 * D, map);
    transition(h, t, R, D, L.p + "transition/", false, false);
    be2Run1d("be2_gated_add", ((size_t)R * D) >> 2, Be2GatedAddArgs{x, t, cond + 5 * D, map, (size_t)R, (uint)D, (uint)(6 * D)});
  }
}

// ---------------------------------------------------------------- the input embedder
// s_inputs [T][455] = [mean_atoms(relu(W q)) | one-hot res_type | profile | deletion_mean | one-hot pocket], q the per-atom
// features c through three windowed conditioned layers (each conditioned on c itself)
void inputEmbedder(float* sInputs, int Si, const Tap& tap) {
  const std::string IE = "input_embedder/atom_attention_encoder/";
  const int T = D_.T, N = D_.N, C = dimW(IE + "proj_atom_continuous/weight", 0);
  float* c = scratch<float>("be2.ie.c", (size_t)N * C);
  atomFeatures(IE, c, N);
  float* pr = scratch<float>("be2.ie.p", (size_t)N * WK * 16);
  pairGeometry(IE, pr, N);
  pairTail(IE, pr, c, N);
  float* q = scratch<float>("be2.ie.q", (size_t)N * C);
  copy(q, c, (size_t)N * C * 4);
  for (int i = 0; ; ++i) {
    const std::string p = IE + "atom_encoder_layers_" + num(i) + "/transformer_layer/";
    const std::string a = "attention/attention_layer/pair_bias_attn/";
    if (!hasW(p + a + "proj_z/weight")) break;
    const int H = dimW(p + a + "proj_z/weight", 0);
    float* bias = scratch<float>("be2.ie.bias", (size_t)N * WK * H);
    windowBias(pr, Wf(p + a + "norm_z/weight"), Wf(p + a + "proj_z/weight"), bias, N, H);
    const CondLayer L{p, a, true, H};
    float* cond = layerCond(L, c, N, C, C, "be2.cl.cond");
    condLayer(L, q, N, C, cond, nullptr, bias, N);
  }
  const int A = dimW(IE + "atom_to_token_trans_0/weight", 0);
  float* h = scratch<float>("be2.ie.h", (size_t)N * A);
  linT(q, IE + "atom_to_token_trans_0/weight", h, N, C, A, 0.f, 0, 0, true);     // (wide: q is raw)
  relu(h, h, (size_t)N * A);
  run1d("be2_mean_atoms", (size_t)T * A, Be2MeanAtomsArgs{h, TOK_START, sInputs, (uint)T, (uint)A, (uint)Si, 0});
  run1d("be2_s_inputs_tail", (size_t)T * 71, Be2SInputsTailArgs{sInputs, Iin("res_type"), Fin("profile"), Fin("deletion_mean"),
                                                                 Iin("pocket_feature"), (uint)T, (uint)A, (uint)Si, 0});
  if (A + 71 != Si) die("BioEmu2: s_inputs is %d + 71 wide, not %d", A, Si);
  if (tap) tap("s_inputs", sInputs, (size_t)T * Si);
}
