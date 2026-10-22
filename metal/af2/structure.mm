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
  half* actH = scratch<half>("sm.actH", (size_t)L * C);
  linearB(initial, S + "initial_projection", -1, act, L, C, C);
  toHalf(act, actH, (size_t)L * C);
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
  const int Ws = Hh * Cs, Wpq = Hh * 3 * Pq, Wpv = Hh * 3 * Pv, Fipa = 3 * Ws + 2 * Wpq + Wpv, Ptot = 2 * Pq + Pv;
  float* ipaProj = scratch<float>("sm.ipaProj", (size_t)L * Fipa);
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
  const half* ipaW = M.derived<half>("ipa.w", (size_t)C * Fipa, [&](half* out) {
    size_t off = 0;
    auto part = [&](const std::string& name, int w) {
      copy2d(out + off, (size_t)Fipa * 2, PH(I + name + "/weights"), (size_t)w * 2, (size_t)w * 2, C);
      off += w;
    };
    part("q_scalar_projection", Ws);
    part("k_scalar_projection", Ws);
    part("v_scalar_projection", Ws);
    part("q_point_projection/point_projection", Wpq);
    part("k_point_projection/point_projection", Wpq);
    part("v_point_projection/point_projection", Wpv);
  });
  const float* ipaB = M.derived<float>("ipa.b", (size_t)Fipa, [&](float* out) {
    size_t off = 0;
    auto part = [&](const std::string& name, int w) {
      copy(out + off, P(I + name + "/bias"), (size_t)w * 4);
      off += w;
    };
    part("q_scalar_projection", Ws);
    part("k_scalar_projection", Ws);
    part("v_scalar_projection", Ws);
    part("q_point_projection/point_projection", Wpq);
    part("k_point_projection/point_projection", Wpq);
    part("v_point_projection/point_projection", Wpv);
  });
  for (int it = 0; it < 8; ++it) {
    { Gemm g{}; g.X = actH; g.tx = F16; g.W = ipaW; g.tw = F16; g.Y = ipaProj; g.rows = L; g.in = C;
      g.out = Fipa; g.bias = ipaB; g.label = "ipa qkv"; gemm(g); }
    run1d("af2_points_global", (size_t)L * Hh * Ptot,
          PointsGlobalArgs{ipaProj + 3 * Ws, rig, qp, kp, vp, (uint)L, (uint)Hh, (uint)Pq, (uint)Pv, (uint)Fipa, 0});
    run("af2_ipa_weights", grid1d(((size_t)L * Hh + 7) / 8, 1), 256,
        IpaWeightsArgs{ipaProj, ipaProj + Ws, qp, kp, b2d, pw, seqMask, attn, (uint)L, (uint)Hh, (uint)Cs, (uint)Pq, (uint)Fipa, qScale});
    run("af2_ipa_outputs", Grid{(uint32_t)L, 2, 1}, 256,
        IpaOutputsArgs{attn, ipaProj + 2 * Ws, vp, act2d, rig, fin, (uint)L, (uint)Hh, (uint)Cs, (uint)Pv, (uint)C2, (uint)Fipa});
    linearB(fin, I + "output_projection", -1, act, L, Fw, C, false, 1.f);
    layerNormW(act, t1, L, C, F + "attention_layer_norm");
    // the transition: three layers, ReLU between, a residual round them
    linearB(t1, F + "transition", -1, t2, L, C, C, true);
    linearB(t2, F + "transition_1", -1, upd, L, C, C, true);
    linearB(upd, F + "transition_2", -1, t1, L, C, C, false, 1.f);
    layerNormW(t1, act, L, C, F + "transition_layer_norm");
    toHalf(act, actH, (size_t)L * C);
    linearB(actH, F + "quat_rigid/rigid", -1, rq, L, C, 6);
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
