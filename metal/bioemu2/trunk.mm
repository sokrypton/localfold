// BioEmu2's trunk (cuda/bioemu2/src/trunk.cuh is the reading):
//
//   s_init = W s_inputs + method[m];  z_init = W1 s_i + W2 s_j + method_pair[m_i] + method_pair[m_j] + relpos
//            + bonds W_b + contact conditioning
//   a pass: s = s_init + W LN(s_prev);  z = z_init + W LN(z_prev);
//           4 MSA layers (pair-weighted averaging, transition, outer product mean, the two triangle multiplications,
//           the pair transition) over m = W msa_feat + W s_inputs, then 48 PairMixer blocks (triangle out, triangle
//           in, transition). The single is never updated inside a pass.
//   distogram = W (z + z^T) + b, 64 bins
#include "bioemu2.h"

static const std::string SE = "sequence_embedder/";

int* TOK_START = nullptr;
void buildTokenStarts(int T, int N) {
  const int* tok = M.hostI("token_idx_at_atom"); const int* pad = M.hostI("atom_pad_mask");
  std::vector<int> start(T + 1, -1);
  int last = -1;
  for (int a = 0; a < N; ++a) {
    if (!pad[a]) continue;
    if (tok[a] < last) die("BioEmu2: a token's atoms are not contiguous");
    if (tok[a] != last) { for (int t = last + 1; t <= tok[a]; ++t) start[t] = a; last = tok[a]; }
  }
  int real = 0; for (int a = 0; a < N; ++a) real += pad[a] != 0;
  for (int t = last + 1; t <= T; ++t) start[t] = real;
  if (TOK_START) release(TOK_START);
  TOK_START = uploadNew(start.data(), start.size());
}

void trunkInit(const float* sInputs, int Si, float* sInit, float* zInit, const Tap& tap) {
  const int T = D_.T, Cs = dimW(SE + "s_init/weight", 0), C = dimW(SE + "z_init_1/weight", 0);
  linT(sInputs, SE + "s_init/weight", sInit, T, Si, Cs);
  {   // s_init += method_conditioning.embedding[method] (no token padding here)
    const int* meth = M.hostI("method_feature");
    const float* emb = M.hostF("w/" + SE + "method_conditioning/embedding/weight");
    std::vector<float> rows((size_t)T * Cs);
    for (int i = 0; i < T; ++i) std::copy(emb + (size_t)meth[i] * Cs, emb + (size_t)(meth[i] + 1) * Cs, rows.begin() + (size_t)i * Cs);
    float* d = uploadNew(rows.data(), rows.size());
    add(sInit, d, rows.size());
    release(d);
  }
  float* rows = scratch<float>("be2.zi.rows", (size_t)T * C); float* cols = scratch<float>("be2.zi.cols", (size_t)T * C);
  linT(sInputs, SE + "z_init_1/weight", rows, T, Si, C);
  linT(sInputs, SE + "z_init_2/weight", cols, T, Si, C);
  const std::string cc = SE + "contact_conditioning/";
  if (dimW(cc + "encoder/weight", 1) != 3 + 1 + C) die("BioEmu2: the contact encoder reads %d, not %d", dimW(cc + "encoder/weight", 1), 4 + C);
  run1d("be2_z_init_terms", (size_t)T * T * C,
        Be2ZInitTermsArgs{zInit, rows, cols, Iin("method_feature"), Wf(SE + "method_conditioning/pair_embedding/weight"),
                          Fin("token_bonds"), Wf(SE + "token_bonds/weight"), Iin("contact_class"), Fin("contact_threshold"),
                          Wf(cc + "encoder/weight"), Wf(cc + "encoder/bias"), Wf(cc + "encoding_unspecified"),
                          Wf(cc + "encoding_unselected"), Wf(cc + "fourier_embedding/proj/weight"),
                          Wf(cc + "fourier_embedding/proj/bias"), (uint)T, (uint)C});
  relPos(SE + "rel_pos/", zInit, T, C, true);
  if (tap) { tap("s_init", sInit, (size_t)T * Cs); tap("z_init", zInit, (size_t)T * T * C); }
}

// ---------------------------------------------------------------- the MSA module
static void pairWeightedAveraging(float* m, const float* z, int l) {
  const int T = D_.T, S = D_.S;
  const std::string p = SE + "msa_module/layers/" + num(l) + "/pair_weighted_averaging/";
  const int Cm = dimW(p + "proj_m/weight", 1), Hh = dimW(p + "proj_z/weight", 0), C = dimW(p + "proj_z/weight", 1);
  const int Ch = dimW(p + "proj_m/weight", 0) / Hh;
  const size_t R = (size_t)S * T, P = (size_t)T * T;
  float* mn = scratch<float>("be2.pwa.mn", R * Cm);
  layerNorm(m, mn, R, Cm, Wf(p + "norm_m/weight"), Wf(p + "norm_m/bias"));
  float* v = scratch<float>("be2.pwa.v", R * Hh * Ch); float* g = scratch<float>("be2.pwa.g", R * Hh * Ch);
  linT(mn, p + "proj_m/weight", v, R, Cm, Hh * Ch);
  linT(mn, p + "proj_g/weight", g, R, Cm, Hh * Ch);
  float* zn = scratch<float>("be2.pwa.zn", P * C);
  layerNorm(z, zn, P, C, Wf(p + "norm_z/weight"), Wf(p + "norm_z/bias"));
  float* b = scratch<float>("be2.pwa.b", P * Hh);
  linT(zn, p + "proj_z/weight", b, P, C, Hh);
  float* w = scratch<float>("be2.pwa.w", (size_t)Hh * P);
  const size_t groups = ((size_t)Hh * T + 7) / 8;
  run("be2_pwa_softmax", Grid{(uint32_t)std::min<size_t>(groups, 65535), (uint32_t)((groups + 65534) / 65535), 1}, 256,
      Be2PwaSoftmaxArgs{b, w, (uint)T, (uint)Hh});
  float* o = scratch<float>("be2.pwa.o", R * Hh * Ch);
  run1d("be2_pwa_apply", R * Hh * Ch, Be2PwaApplyArgs{w, v, o, (u64)S, (uint)T, (uint)Hh, (uint)Ch, 0});
  run1d("be2_sigmoid_mul", R * Hh * Ch, Be2SigmoidMulArgs{o, g, o, R * Hh * Ch});
  linT(o, p + "proj_o/weight", m, R, Hh * Ch, Cm, 1.f);
}
// OuterProductMeanVanilla: the mean is over VALID ROWS (rows with any live entry), not per pair; a block of i at a time
static void outerProductMean(const float* m, float* z, int l) {
  const int T = D_.T, S = D_.S;
  const std::string p = SE + "msa_module/layers/" + num(l) + "/outer_product_mean/";
  const int Cm = dimW(p + "proj_ab/weight", 1), Co = dimW(p + "proj_ab/weight", 0) / 2, C = dimW(p + "proj_o/weight", 0);
  const size_t R = (size_t)S * T;
  float* mn = scratch<float>("be2.opm.mn", R * Cm);
  layerNorm(m, mn, R, Cm, Wf(p + "norm/weight"), Wf(p + "norm/bias"));
  float* proj = scratch<float>("be2.opm.proj", R * 2 * Co);
  linT(mn, p + "proj_ab/weight", proj, R, Cm, 2 * Co);
  const size_t perI = (size_t)T * Co * Co;
  const int bi = (int)std::max<size_t>(1, std::min<size_t>(T, ((size_t)32 << 20) / (4 * perI)));
  float* y = scratch<float>("be2.opm.y", (size_t)bi * perI);
  for (int i0 = 0; i0 < T; i0 += bi) {
    const int n = std::min(bi, T - i0);
    run1d("be2_opm", (size_t)n * perI, Be2OpmArgs{proj, y, (u64)S, (uint)T, (uint)Co, (uint)i0, (uint)n,
                                                  1.f / (float)D_.nValidRows, 0});
    float* zr = z + (size_t)i0 * T * C;
    linT(y, p + "proj_o/weight", zr, (size_t)n * T, Co * Co, C, 1.f);
    addBias(zr, Wf(p + "proj_o/bias"), (size_t)n * T, C);
  }
}

void trunkPass(const float* sInputs, int Si, const float* sInit, const float* zInit, float* s, float* z) {
  const int T = D_.T, S = D_.S, Cs = dimW(SE + "s_recycle/weight", 0), C = dimW(SE + "z_recycle/weight", 0);
  const size_t P = (size_t)T * T;
  // s = s_init + s_recycle(LN(s)); z = z_init + z_recycle(LN(z)), in place
  float* sn = scratch<float>("be2.tp.sn", (size_t)T * Cs);
  layerNorm(s, sn, T, Cs, Wf(SE + "s_norm/weight"), Wf(SE + "s_norm/bias"));
  copy(s, sInit, (size_t)T * Cs * 4);
  linT(sn, SE + "s_recycle/weight", s, T, Cs, Cs, 1.f);
  float* zn = scratch<float>("be2.tri.zn", P * C);       // (shares the triangle's: spent before it runs)
  layerNorm(z, zn, P, C, Wf(SE + "z_norm/weight"), Wf(SE + "z_norm/bias"));
  copy(z, zInit, P * C * 4);
  linT(zn, SE + "z_recycle/weight", z, P, C, C, 1.f);
  // the MSA module
  const std::string mm = SE + "msa_module/";
  const int Cm = dimW(mm + "msa_proj/weight", 0);
  float* m = scratch<float>("be2.msa.m", (size_t)S * T * Cm);
  float* sp = scratch<float>("be2.msa.sp", (size_t)T * Cm);
  linT(sInputs, mm + "s_proj/weight", sp, T, Si, Cm);
  run1d("be2_msa_embed", (size_t)S * T * Cm, Be2MsaEmbedArgs{m, Iin("msa"), Iin("has_deletion"), Fin("deletion_value"),
                                                              Iin("msa_paired"), Wf(mm + "msa_proj/weight"), sp, (u64)S,
                                                              (uint)T, (uint)Cm});
  for (int l = 0; hasW("msa/" + num(l) + "/pairTransition/transition1"); ++l) {
    const std::string B = "msa/" + num(l) + "/";
    pairWeightedAveraging(m, z, l);
    transition(m, m, (size_t)S * T, Cm, mm + "layers/" + num(l) + "/msa_transition/");
    outerProductMean(m, z, l);
    triangle(z, T, C, B + "triangleMultiplicationOutgoing/", true);
    triangle(z, T, C, B + "triangleMultiplicationIncoming/", false);
    pairTransition(z, P, C, B + "pairTransition/");
  }
  for (int l = 0; hasW("pairmixer/" + num(l) + "/pairTransition/transition1"); ++l) {
    const std::string B = "pairmixer/" + num(l) + "/";
    triangle(z, T, C, B + "triangleMultiplicationOutgoing/", true);
    triangle(z, T, C, B + "triangleMultiplicationIncoming/", false);
    pairTransition(z, P, C, B + "pairTransition/");
  }
}

// distogram logits [P][64] = W (z + z^T) + b
void distogram(const float* z, float* logits) {
  const int T = D_.T, C = dimW("distogram_module/distogram/weight", 1), B = dimW("distogram_module/distogram/weight", 0);
  const size_t P = (size_t)T * T;
  float* zs = scratch<float>("be2.dg.zs", P * C);
  run1d("be2_symmetrise", P * C, Be2SymmetriseArgs{z, zs, (uint)T, (uint)C});
  // (wide: the RAW pair - its RMS is ~4.6e3 at 6MRR, and z + z^T passes float16's 65,504 - rounded to half it is inf)
  linT(zs, "distogram_module/distogram/weight", logits, P, C, B, 0.f, 0, 0, true);
  addBias(logits, Wf("distogram_module/distogram/bias"), P, B);
}
// Boltz's 64 bins over 2-22 A: the bins whose lower edges (linspace(2, 22, 63)) are at most 8 A
std::vector<float> contactProbs(const float* logits, int T, int B) {
  int near = 0; for (int e = 0; e < B - 1; ++e) if (2.0 + 20.0 * e / (B - 2) <= 8.0) ++near;
  const size_t P = (size_t)T * T;
  float* probs = scratch<float>("be2.dg.contacts", P);
  run1d("be2_contacts", P, Be2ContactArgs{logits, probs, P, (uint)B, (uint)near});
  return download(probs, P);
}
