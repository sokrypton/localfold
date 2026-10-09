// AlphaFold 2's template embedders (cuda/af2/src/templates.cuh is the reading): the multimer's (TemplateEmbedding: it
// runs whether or not a template is given - one blank template, whose embedding is not zero), the monomer's
// (MonomerTemplateEmbedding: a pair stack a template, then a pointwise attention from the query pair), and the
// templates' single features as MSA rows.
#include "af2.h"
#include <cmath>

void templateEmbedding(float* pair, const float* pairMask, int L) {
  const std::string TE = "evoformer/template_embedding/", S = TE + "single_template_embedding/";
  const std::string IT = S + "template_embedding_iteration/";
  const int C = 64, Cz = 128;
  size_t pairs = (size_t)L * L;
  int T = (int)M.meta("meta/templates", 0);
  const int* tAatype; const float* tPos; const float* tMask;
  if (T == 0) {          // one blank template: aatype 0, no atoms
    tAatype = scratch<int>("tmpl.blankAatype", L); tPos = scratch<float>("tmpl.blankPos", (size_t)L * 37 * 3);
    tMask = scratch<float>("tmpl.blankMask", (size_t)L * 37);
    fill((void*)tAatype, 0, (size_t)L * 4); fill((void*)tPos, 0, (size_t)L * 37 * 3 * 4); fill((void*)tMask, 0, (size_t)L * 37 * 4);
    T = 1;
  } else { tAatype = Ii("t/aatype"); tPos = In("t/positions"); tMask = In("t/mask"); }
  half* qn = scratch<half>("tmpl.qn", pairs * Cz);
  layerNormW(pair, qn, pairs, Cz, S + "query_embedding_norm");
  // the nine linears' biases, summed once
  const float* bsum = M.derived<float>("tmpl.bsum", C, [&](float* out) {
    copy(out, P(S + "template_pair_embedding_0/bias"), C * 4);
    for (int k = 1; k < 9; ++k) add(out, P(S + "template_pair_embedding_" + std::to_string(k) + "/bias"), C);
  });
  float* sum = scratch<float>("tmpl.sum", pairs * C);
  fill(sum, 0, pairs * C * 4);
  float* act = scratch<float>("tmpl.act", pairs * C);
  float* actn = scratch<float>("tmpl.actn", pairs * C);
  auto w = [&](int k) { return P(S + "template_pair_embedding_" + std::to_string(k) + "/weights"); };
  int nb = (int)dimW(IT + "pair_transition/transition1/weights", 0);
  for (int k = 0; k < T; ++k) {
    run("af2_template_pair", grid1d(pairs, 1), 64,
        TmplPairArgs{tAatype + (size_t)k * L, tPos + (size_t)k * L * 37 * 3, tMask + (size_t)k * L * 37, w(0), w(1), w(2), w(3), w(4),
                     w(5), w(6), w(7), bsum, act, (uint)L, (uint)C});
    // + the normalised query's term (template_pair_embedding_8, its bias already in bsum)
    { Gemm g{}; g.X = qn; g.tx = F16; g.W = PH(S + "template_pair_embedding_8/weights"); g.tw = F16; g.Y = act; g.beta = 1.f;
      g.rows = pairs; g.in = Cz; g.out = C; g.label = "template query term"; gemm(g); }
    for (int b = 0; b < nb; ++b) {
      triangleMultiplication(act, pairMask, L, C, IT, b, true);
      triangleMultiplication(act, pairMask, L, C, IT, b, false);
      triangleAttention(act, pairMask, L, C, IT, b, true, false);
      triangleAttention(act, pairMask, L, C, IT, b, false, false);
      transition(act, pairs, C, IT + "pair_transition", b);
    }
    layerNormW(act, actn, pairs, C, S + "output_layer_norm");
    add(sum, actn, pairs * C);
  }
  run1d("af2_relu_scale", pairs * C, ReluScaleArgs{sum, pairs * C, 1.f / T, 0});
  linearB(sum, TE + "output_linear", -1, pair, pairs, C, Cz, false, 1.f);
  releaseScratch({"tmpl."});
}

void templateEmbeddingMonomer(float* pair, const float* pairMask, int L, int T) {
  const std::string TE = "evoformer/template_embedding/", S = TE + "single_template_embedding/";
  const std::string PS = S + "template_pair_stack/__layer_stack_no_state/";
  const int C = 64, Cz = 128;
  size_t pairs = (size_t)L * L;
  if (T > 8) die("at most 8 templates");
  float* reps = scratch<float>("mtmpl.reps", (size_t)T * pairs * C);
  int nb = (int)dimW(PS + "pair_transition/transition1/weights", 0);
  for (int k = 0; k < T; ++k) {
    float* act = reps + (size_t)k * pairs * C;
    run("af2_template_pair_monomer", grid1d(pairs, 1), 64,
        TmplPairMonoArgs{Ii("t/aatype") + (size_t)k * L, In("t/positions") + (size_t)k * L * 37 * 3, In("t/mask") + (size_t)k * L * 37,
                         P(S + "embedding2d/weights"), P(S + "embedding2d/bias"), act, (uint)L, (uint)C});
    for (int b = 0; b < nb; ++b) {        // the MONOMER's order: both attentions, then both multiplications
      triangleAttention(act, pairMask, L, C, PS, b, true, false);
      triangleAttention(act, pairMask, L, C, PS, b, false, false);
      triangleMultiplication(act, pairMask, L, C, PS, b, true);
      triangleMultiplication(act, pairMask, L, C, PS, b, false);
      transition(act, pairs, C, PS + "pair_transition", b);
    }
    float* tmp = scratch<float>("mtmpl.ln", pairs * C);
    layerNormW(act, tmp, pairs, C, S + "output_layer_norm");
    copy(act, tmp, pairs * C * 4);
  }
  const std::string A = TE + "attention/";
  int Hh = (int)dimW(A + "query_w", 1), D = (int)dimW(A + "query_w", 2), W = Hh * D;
  float* q = scratch<float>("mtmpl.q", pairs * W);
  float* kk = scratch<float>("mtmpl.k", (size_t)T * pairs * W); float* vv = scratch<float>("mtmpl.v", (size_t)T * pairs * W);
  { Gemm g{}; g.X = pair; g.W = PH(A + "query_w"); g.tw = F16; g.half = true; g.Y = q; g.rows = pairs; g.in = Cz; g.out = W;
    g.alpha = 1.f / sqrtf((float)D); g.label = "template query"; gemm(g); }
  { Gemm g{}; g.X = reps; g.W = PH(A + "key_w"); g.tw = F16; g.half = true; g.Y = kk; g.rows = (size_t)T * pairs; g.in = C; g.out = W; gemm(g); }
  { Gemm g{}; g.X = reps; g.W = PH(A + "value_w"); g.tw = F16; g.half = true; g.Y = vv; g.rows = (size_t)T * pairs; g.in = C; g.out = W; gemm(g); }
  float* o = scratch<float>("mtmpl.o", pairs * W);
  run1d("af2_point_attention", pairs * Hh, PointAttnArgs{q, kk, vv, o, pairs, (uint)T, (uint)Hh, (uint)D, 0});
  { Gemm g{}; g.X = o; g.W = PH(A + "output_w"); g.tw = F16; g.half = true; g.Y = pair; g.beta = 1.f; g.bias = P(A + "output_b");
    g.rows = pairs; g.in = W; g.out = Cz; g.label = "template output"; gemm(g); }
  releaseScratch({"mtmpl."});
}

// rows [T, L, 256] of the MSA from the templates' single features; their masks into rowMask [T, L]
void templateRows(int L, int T, bool multimer, float* rows, float* rowMask) {
  const std::string E = "evoformer/";
  int F = multimer ? 34 : 57;
  float* feat = scratch<float>("trow.feat", (size_t)T * L * F);
  for (int k = 0; k < T; ++k)
    run1d(multimer ? "af2_template_chi" : "af2_template_torsion", L,
          TorsionFeatArgs{Ii("t/aatype") + (size_t)k * L, In("t/positions") + (size_t)k * L * 37 * 3, In("t/mask") + (size_t)k * L * 37,
                          feat + (size_t)k * L * F, rowMask + (size_t)k * L, (uint)L, 0});
  float* hid = scratch<float>("trow.hid", (size_t)T * L * 256);
  linearB(feat, E + "template_single_embedding", -1, hid, (size_t)T * L, F, 256, true);
  linearB(hid, E + "template_projection", -1, rows, (size_t)T * L, 256, 256);
}
