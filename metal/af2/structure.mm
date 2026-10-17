// The structure module - af3-any-model's AF2 (alphafold3/af2/model/folding.py: generate_monomer_rigids, FoldIteration,
// InvariantPointAttention, MultiRigidSidechain; all_atom.py). cuda/af2/src/structure.cuh is the reading.
#include "af2.h"
#include <cmath>

StructureOut structureModule(const float* single, const float* pair, int L, float positionScale) {
  const std::string S = "structure_module/", F = S + "fold_iteration/", I = F + "invariant_point_attention/";
  const int C = 384, C2 = 128;
  size_t pairs = (size_t)L * L;
  float* initial = scratch<float>("sm.initial", (size_t)L * C);
  layerNormW(single, initial, L, C, S + "single_layer_norm");
  float* act = scratch<float>("sm.act", (size_t)L * C);
  linearB(initial, S + "initial_projection", -1, act, L, C, C);
  float* act2d = scratch<float>("sm.act2d", pairs * C2);
  layerNormW(pair, act2d, pairs, C2, S + "pair_layer_norm");
  int Hh = (int)dimW(I + "q_scalar_projection/weights", 1), Cs = (int)dimW(I + "q_scalar_projection/weights", 2);
  int Pq = (int)dimW(I + "q_point_projection/point_projection/weights", 2) / 3;
  int Pv = (int)dimW(I + "v_point_projection/point_projection/weights", 2) / 3;
  float* b2d = scratch<float>("sm.b2d", pairs * Hh);
  linearB(act2d, I + "attention_2d", -1, b2d, pairs, C2, Hh);       // the same every iteration (shared weights)
  float* pw = scratch<float>("sm.pw", Hh);
  run1d("af2_softplus_scale", Hh, SoftplusArgs{P(I + "trainable_point_weights"), pw, (uint)Hh, sqrtf(1.f / (Pq * 9.f / 2.f))});
  float* rig = scratch<float>("sm.rigid", (size_t)L * 12);
  run1d("af2_identity_rigid", L, IdentityRigidArgs{rig, (uint)L, 0});
  float* qs = scratch<float>("sm.qs", (size_t)L * Hh * Cs); float* ks = scratch<float>("sm.ks", (size_t)L * Hh * Cs);
  float* vs = scratch<float>("sm.vs", (size_t)L * Hh * Cs);
  float* proj = scratch<float>("sm.pproj", (size_t)L * Hh * 3 * std::max(Pq, Pv));
  float* qp = scratch<float>("sm.qp", (size_t)L * Hh * Pq * 3); float* kp = scratch<float>("sm.kp", (size_t)L * Hh * Pq * 3);
  float* vp = scratch<float>("sm.vp", (size_t)L * Hh * Pv * 3);
  float* attn = scratch<float>("sm.attn", (size_t)L * Hh * L);
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  float* fin = scratch<float>("sm.final", (size_t)L * Fw);
  float* t1 = scratch<float>("sm.t1", (size_t)L * C); float* t2 = scratch<float>("sm.t2", (size_t)L * C);
  float* upd = scratch<float>("sm.upd", (size_t)L * C);
  float* rq = scratch<float>("sm.rq", (size_t)L * 6);
  const float* seqMask = In("seq_mask");
  const float qScale = sqrtf(1.f / Cs);
  const float* qBias = M.derived<float>("ipa.qbias", (size_t)Hh * Cs, [&](float* out) {
    const float* b = M.hostF("w/" + I + "q_scalar_projection/bias");
    std::vector<float> v(b, b + (size_t)Hh * Cs);
    for (auto& x : v) x *= qScale;
    upload(out, v.data(), v.size() * 4);
  });
  for (int it = 0; it < 8; ++it) {
    // IPA: q_scalar * sqrt(1 / num_scalar_qk), its bias included - the GEMM's alpha and a scaled copy of the bias
    { Gemm g{}; g.X = act; g.W = PH(I + "q_scalar_projection/weights"); g.tw = F16; g.half = true; g.Y = qs; g.rows = L; g.in = C;
      g.out = Hh * Cs; g.alpha = qScale; g.bias = qBias; g.label = "ipa q"; gemm(g); }
    linearB(act, I + "k_scalar_projection", -1, ks, L, C, Hh * Cs);
    linearB(act, I + "v_scalar_projection", -1, vs, L, C, Hh * Cs);
    linearB(act, I + "q_point_projection/point_projection", -1, proj, L, C, Hh * 3 * Pq);
    run1d("af2_points_global", (size_t)L * Hh * Pq, PointsGlobalArgs{proj, rig, qp, (uint)L, (uint)Hh, (uint)Pq, 0});
    linearB(act, I + "k_point_projection/point_projection", -1, proj, L, C, Hh * 3 * Pq);
    run1d("af2_points_global", (size_t)L * Hh * Pq, PointsGlobalArgs{proj, rig, kp, (uint)L, (uint)Hh, (uint)Pq, 0});
    linearB(act, I + "v_point_projection/point_projection", -1, proj, L, C, Hh * 3 * Pv);
    run1d("af2_points_global", (size_t)L * Hh * Pv, PointsGlobalArgs{proj, rig, vp, (uint)L, (uint)Hh, (uint)Pv, 0});
    run("af2_ipa_weights", grid1d(((size_t)L * Hh + 7) / 8, 1), 256,
        IpaWeightsArgs{qs, ks, qp, kp, b2d, pw, seqMask, attn, (uint)L, (uint)Hh, (uint)Cs, (uint)Pq});
    run("af2_ipa_outputs", Grid{(uint32_t)(L * Hh), 1, 1}, 128,
        IpaOutputsArgs{attn, vs, vp, act2d, rig, fin, (uint)L, (uint)Hh, (uint)Cs, (uint)Pv, (uint)C2, 0});
    linearB(fin, I + "output_projection", -1, act, L, Fw, C, false, 1.f);
    layerNormW(act, t1, L, C, F + "attention_layer_norm");
    // the transition: three layers, ReLU between, a residual round them
    linearB(t1, F + "transition", -1, t2, L, C, C, true);
    linearB(t2, F + "transition_1", -1, upd, L, C, C, true);
    linearB(upd, F + "transition_2", -1, t2, L, C, C);
    add(t2, t1, (size_t)L * C);
    layerNormW(t2, act, L, C, F + "transition_layer_norm");
    linearB(act, F + "quat_rigid/rigid", -1, rq, L, C, 6);
    run1d("af2_rigid_update", L, RigidUpdateArgs{rig, rq, (uint)L, 0});
  }
  // the side chains, from the last iteration
  const std::string R = F + "rigid_sidechain/";
  float* ra = scratch<float>("sm.ra", (size_t)L * C); float* sc = scratch<float>("sm.sc", (size_t)L * 128);
  float* sb = scratch<float>("sm.sb", (size_t)L * 128); float* sc2 = scratch<float>("sm.sc2", (size_t)L * 128);
  run1d("af2_relu_copy", (size_t)L * C, ReluCopyArgs{act, ra, (u64)L * C});
  linearB(ra, R + "input_projection", -1, sc, L, C, 128);
  run1d("af2_relu_copy", (size_t)L * C, ReluCopyArgs{initial, ra, (u64)L * C});
  linearB(ra, R + "input_projection_1", -1, sc, L, C, 128, false, 1.f);
  for (int r = 0; r < 2; ++r) {
    std::string suffix = r == 0 ? "" : "_1";
    run1d("af2_relu_copy", (size_t)L * 128, ReluCopyArgs{sc, sb, (u64)L * 128});
    linearB(sb, R + "resblock1" + suffix, -1, sc2, L, 128, 128, true);
    linearB(sc2, R + "resblock2" + suffix, -1, sc, L, 128, 128, false, 1.f);
  }
  run1d("af2_relu_copy", (size_t)L * 128, ReluCopyArgs{sc, sb, (u64)L * 128});
  float* un = scratch<float>("sm.un", (size_t)L * 14);
  linearB(sb, R + "unnormalized_angles", -1, un, L, 128, 14);
  StructureOut o{act, rig, scratch<float>("sm.pos37", (size_t)L * 37 * 3), scratch<float>("sm.pos14", (size_t)L * 14 * 3),
                 scratch<float>("sm.angles", (size_t)L * 14)};
  run1d("af2_sidechains", L, SidechainArgs{un, rig, Ii("aatype"), In("c/rigid_group_default_frame"), M.i("c/atom14_to_rigid_group"),
                                          In("c/atom14_rigid_group_positions"), In("c/atom14_mask"), M.i("c/atom37_to_atom14"),
                                          In("c/atom37_mask"), seqMask, o.angles, o.pos14, o.pos37, (uint)L, positionScale});
  return o;
}
