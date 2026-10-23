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
  if (HALF && C >= 16 && CP >= 16) {
    half* rch = scratch<half>("be2.pt.rch", (size_t)N * C);
    toHalf(rc, rch, (size_t)N * C);
    const std::string pk = "pt.qk." + p;
    Gemm G{}; G.ptrs = true; G.batch = 2;
    G.X = pointerArray(pk + ".x", {(u64)rch, (u64)rch}); G.tx = F16;
    G.W = pointerArray(pk + ".w", {(u64)M.h("w/" + p + "c_to_p_trans_q_1/weight"), (u64)M.h("w/" + p + "c_to_p_trans_k_1/weight")});
    G.tw = F16; G.transW = true;
    G.Y = pointerArray(pk + ".y", {(u64)cq, (u64)ck}); G.ty = F32;
    G.rows = N; G.in = C; G.out = CP; G.label = "c_to_p_trans_qk";
    gemm(G);
  } else {
    linT(rc, p + "c_to_p_trans_q_1/weight", cq, N, C, CP);
    linT(rc, p + "c_to_p_trans_k_1/weight", ck, N, C, CP);
  }
  run1d("be2_pair_tail", (size_t)N * WK, Be2PairTailArgs{pr, cq, ck, Wf(p + "p_mlp_1/weight"), Wf(p + "p_mlp_3/weight"),
                                                         Wf(p + "p_mlp_5/weight"), (uint)N, 0});
}
void windowBias(const float* pr, const float* normW, const float* projZ, float* out, int N, int H, int layers) {
  run1d("be2_window_bias", (size_t)N * WK, Be2WindowBiasArgs{pr, normW, projZ, out, (uint)N, (uint)H, (uint)layers, 0});
}

// ---------------------------------------------------------------- the conditioned transformer layer
float* layerCond(const CondLayer& L, const float* c, int Rc, int Cc, int D, const char* name) {
  float* cond = scratch<float>(name, (size_t)Rc * 6 * D);
  if (HALF && (Cc == 128 || Cc == 768)) {
    const int pR = Rc <= 128 ? ((Rc + 15) & ~15) : ((Rc + 31) & ~31);
    half* cnh = scratch<half>("be2.cl.cnh", (size_t)pR * Cc);
    if (pR > Rc) fill(cnh + (size_t)Rc * Cc, 0, (size_t)(pR - Rc) * Cc * 2);
    const size_t groups = ((size_t)Rc + 7) / 8;
    run("be2_cond_in", Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1}, 256,
        Be2CondInArgs{c, Wf(L.p + "cond_0/weight"), WfOpt(L.p + "cond_0/bias"), cnh, (u64)Rc, (uint)Rc, (uint)Cc, 0, 1});
    linT(cnh, L.p + "cond_2/weight", cond, pR, Cc, 6 * D);
    if (const float* b2 = WfOpt(L.p + "cond_2/bias"))
      be2Run1d("be2_batch_bias", (size_t)Rc * 6 * D, Be2BatchBiasArgs{cond, b2, (u64)Rc * 6 * D, (uint)Rc, (uint)(6 * D), 0, 0});
    return cond;
  }
  float* cn = scratch<float>("be2.cl.cn", (size_t)Rc * Cc);
  layerNorm(c, cn, Rc, Cc, Wf(L.p + "cond_0/weight"), WfOpt(L.p + "cond_0/bias"));
  silu(cn, cn, (size_t)Rc * Cc);
  linT(cn, L.p + "cond_2/weight", cond, Rc, Cc, 6 * D, 0.f, 0, 0, false, WfOpt(L.p + "cond_2/bias"));
  return cond;
}
void adaLn(float* x, float* y, size_t rows, int C, const float* alpha, const float* beta, int ldc, const int* map,
           half* yh, const float* resH, const float* resGate, int ldrg) {
  const size_t groups = (rows + 7) / 8;
  run("be2_ada_ln", Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1}, 256,
      Be2AdaLnArgs{x, y, alpha, beta, map, rows, (uint)C, (uint)C, (uint)ldc, 1.f, yh, resH, resGate, (uint)(ldrg ? ldrg : ldc), 0});
}
// cond: [6D] a row - alpha_a, beta_a, gamma_a, alpha_m, beta_m, gamma_m
// the attention's q | k | v | g as one torch weight [4D][D] in half: proj_qkv's rows then proj_g's - the layout
// metal/core's flash attention reads (per position q, k, v, g, each heads x head width)
const half* qkvgWeight(const std::string& a, int D) {
  return M.derived<half>("qkvg:" + a, (size_t)4 * D * D, [&](half* out) {
    run1d("be2_stack", ((size_t)4 * D * D) >> 2, Be2StackArgs{M.h("w/" + a + "proj_qkv/weight"), M.h("w/" + a + "proj_g/weight"), out,
                                                              (u64)3 * D * D, (u64)D * D});
  });
}
// the attention on the fast arm: adaLN into half, q|k|v|g in one GEMM, either be2_window_attention_h (windowed atoms:
// half K/V in 16 KB shared memory, sigmoid(g) in the epilogue) or metal/core's flash attention (tokens), and the
// output projection from half into x with the gated residual in its epilogue where metal/core takes it (or returned for
// the transition's adaLn to add in place before its norm)
static const float* attentionHalf(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map,
                                  const float* bias, const half* biasH, int perSample,
                                  const float* inResH, const float* inResGate, int inResLdg, bool preNorm) {
  const int pR = L.windowed ? R : (R <= 128 ? ((R + 15) & ~15) : ((R + 31) & ~31));
  const std::string a = L.p + L.att;
  half* hh = scratch<half>(L.windowed ? "be2.cl.hh" : "be2.cl.thh", (size_t)pR * D);
  if (!preNorm) adaLn(x, nullptr, R, D, cond, cond + D, 6 * D, map, hh, inResH, inResGate, inResLdg);
  half* qkvg = scratch<half>("be2.cl.qkvg", (size_t)pR * 4 * D);
  Gemm g{}; g.X = hh; g.tx = F16; g.W = qkvgWeight(a, D); g.tw = F16; g.transW = true; g.Y = qkvg; g.ty = F16;
  g.rows = pR; g.in = D; g.out = 4 * D; g.accFloat = true; g.label = L.windowed ? "atom qkvg" : "token qkvg";
  gemm(g);
  half* o = scratch<half>(L.windowed ? "be2.cl.oh" : "be2.cl.toh", (size_t)pR * D);
  if (L.windowed) {
    if (D / L.heads != 32) die("BioEmu2: an atom head of %d (this port is written for 32)", D / L.heads);
    if (R % WQ || perSample % WQ) die("BioEmu2: %d atom rows are not whole windows", R);
    run("be2_window_attention_h", Grid{(uint32_t)(R / WQ), (uint32_t)L.heads, 1}, WQ * 4,
        Be2WindowAttnHArgs{qkvg, Wf(a + "proj_q_bias"), bias, Iin("atom_pad_mask"), o, (uint)perSample, (uint)L.heads});
  } else {
    Attention at{}; at.qkvg = qkvg; at.out = o; at.n = perSample; at.heads = L.heads; at.D = D / L.heads; at.rows = R / perSample;
    at.scale = 1.f / sqrtf((float)(D / L.heads)); at.bias = biasH; at.biasStride = (perSample + 63) & ~63; at.biasPadInf = true;
    at.qBias = Wf(a + "proj_q_bias");
    // (every token is real here - no token padding - so no key mask; a padded input would need token_pad_mask as float)
    attention(at);
  }
  Gemm G{}; G.X = o; G.tx = F16; G.W = M.h("w/" + a + "proj_o/weight"); G.tw = F16; G.transW = true; G.Y = x; G.ty = F32;
  G.rows = pR; G.in = D; G.out = D; G.label = L.windowed ? "atom proj_o" : "token proj_o";
  if (gemmGatedRows(G, cond + 2 * D, 6 * D, map)) return nullptr;
  float* h = scratch<float>("be2.cl.h", (size_t)pR * D);
  G.Y = h; gemm(G);
  return h;
}
// the attention half of a layer on the reference arm: adaLN, q|k|v and g, the attention, the gate, the output projection into h
static void attentionF32(const CondLayer& L, float* x, float* h, int R, int D, const float* cond, const int* map,
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
               const half* biasH, const float* inResH, const float* inResGate, int inResLdg, const float** outResH, bool preNorm) {
  const float* resH = nullptr;
  if (HALF && (L.windowed || biasH)) {
    resH = attentionHalf(L, x, R, D, cond, map, bias, biasH, perSample, inResH, inResGate, inResLdg, preNorm);
  } else {
    float* h = scratch<float>("be2.cl.h", (size_t)R * D);
    attentionF32(L, x, h, R, D, cond, map, bias, perSample);
    be2Run1d("be2_gated_add", ((size_t)R * D) >> 2, Be2GatedAddArgs{x, h, cond + 2 * D, map, (size_t)R, (uint)D, (uint)(6 * D)});
  }
  if (HALF) {
    const int pR = L.windowed ? R : (R <= 128 ? ((R + 15) & ~15) : ((R + 31) & ~31));
    half* hh = scratch<half>(L.windowed ? "be2.cl.hh" : "be2.cl.thh", (size_t)pR * D);
    adaLn(x, nullptr, R, D, cond + 3 * D, cond + 4 * D, 6 * D, map, hh, resH, resH ? cond + 2 * D : nullptr, 6 * D);
    transitionHalf(hh, x, pR, D, L.p + "transition/", false, cond + 5 * D, 6 * D, map, outResH);    // (x += gamma fc3(...))
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
  int ieLayers = 0;
  auto iePrefix = [&](int i) { return IE + "atom_encoder_layers_" + num(i) + "/transformer_layer/"; };
  const std::string a = "attention/attention_layer/pair_bias_attn/";
  while (hasW(iePrefix(ieLayers) + a + "proj_z/weight")) ++ieLayers;
  if (ieLayers > 0) {
    const int H = dimW(iePrefix(0) + a + "proj_z/weight", 0);
    const size_t wb = (size_t)N * WK * H;
    const float* wbNorm = M.derived<float>("ie.wb.norm", (size_t)ieLayers * 16, [&](float* out) {
      for (int i = 0; i < ieLayers; ++i) copy(out + (size_t)i * 16, Wf(iePrefix(i) + a + "norm_z/weight"), 64);
    });
    const float* wbProj = M.derived<float>("ie.wb.proj", (size_t)ieLayers * H * 16, [&](float* out) {
      for (int i = 0; i < ieLayers; ++i) copy(out + (size_t)i * H * 16, Wf(iePrefix(i) + a + "proj_z/weight"), (size_t)H * 64);
    });
    float* bias = scratch<float>("be2.ie.bias", wb * ieLayers);
    windowBias(pr, wbNorm, wbProj, bias, N, H, ieLayers);
    if (HALF && C == 128) {
      const float* cw0 = M.derived<float>("ie.cond.w0", (size_t)ieLayers * C, [&](float* out) {
        for (int i = 0; i < ieLayers; ++i) copy(out + (size_t)i * C, Wf(iePrefix(i) + "cond_0/weight"), (size_t)C * 4);
      });
      const float* cb0 = hasW(iePrefix(0) + "cond_0/bias") ? M.derived<float>("ie.cond.b0", (size_t)ieLayers * C, [&](float* out) {
        for (int i = 0; i < ieLayers; ++i) copy(out + (size_t)i * C, Wf(iePrefix(i) + "cond_0/bias"), (size_t)C * 4);
      }) : nullptr;
      const float* cb2 = hasW(iePrefix(0) + "cond_2/bias") ? M.derived<float>("ie.cond.b2", (size_t)ieLayers * 6 * C, [&](float* out) {
        for (int i = 0; i < ieLayers; ++i) copy(out + (size_t)i * 6 * C, Wf(iePrefix(i) + "cond_2/bias"), (size_t)6 * C * 4);
      }) : nullptr;
      half* cnh = scratch<half>("be2.ie.cnh", (size_t)ieLayers * N * C);
      const size_t groups = ((size_t)N + 7) / 8;
      run("be2_cond_in", Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1}, 256,
          Be2CondInArgs{c, cw0, cb0, cnh, (u64)N, (uint)N, (uint)C, 0, (uint)ieLayers});
      float* condAll = scratch<float>("be2.ie.condAll", (size_t)ieLayers * N * 6 * C);
      std::vector<u64> X(ieLayers), Wp(ieLayers), Y(ieLayers);
      for (int i = 0; i < ieLayers; ++i) {
        X[i] = (u64)(cnh + (size_t)i * N * C);
        Wp[i] = (u64)M.h("w/" + iePrefix(i) + "cond_2/weight");
        Y[i] = (u64)(condAll + (size_t)i * N * 6 * C);
      }
      Gemm Gc{}; Gc.ptrs = true; Gc.batch = ieLayers;
      Gc.X = pointerArray("ie.cond.x", X); Gc.tx = F16;
      Gc.W = pointerArray("ie.cond.w", Wp); Gc.tw = F16; Gc.transW = true;
      Gc.Y = pointerArray("ie.cond.y", Y); Gc.ty = F32;
      Gc.rows = N; Gc.in = C; Gc.out = 6 * C; Gc.label = "ie cond_2 batched";
      gemm(Gc);
      if (cb2)
        be2Run1d("be2_batch_bias", (size_t)ieLayers * N * 6 * C,
                 Be2BatchBiasArgs{condAll, cb2, (u64)ieLayers * N * 6 * C, (uint)N, (uint)(6 * C), 0, 0});
      for (int i = 0; i < ieLayers; ++i) {
        const CondLayer L{iePrefix(i), a, true, H};
        condLayer(L, q, N, C, condAll + (size_t)i * N * 6 * C, nullptr, bias + wb * i, N);
      }
    } else {
      for (int i = 0; i < ieLayers; ++i) {
        const CondLayer L{iePrefix(i), a, true, H};
        float* cond = layerCond(L, c, N, C, C, "be2.cl.cond");
        condLayer(L, q, N, C, cond, nullptr, bias + wb * i, N);
      }
    }
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
