#include "structgrad.h"
#include "grad.h"
#include "kernels.h"
#include <cmath>
#include <string>

namespace mt {

void rigidUpdateBwd(const float* rigOld, const float* upd, const float* dRigNew,
                    float* dRigOld, float* dUpd, int L, bool accumulateRigOld) {
  run1d("lf_rigid_upd_bwd", (size_t)L,
        RigidUpdBwdArgs{rigOld, upd, dRigNew, dRigOld, dUpd, (uint32_t)L, accumulateRigOld ? 1u : 0u});
}

void pointsGlobalBwd(const float* proj, const float* rig, const float* dOut,
                     float* dProj, float* dRig, int L, int H, int P,
                     bool accumulateProj, bool accumulateRig) {
  run1d("lf_points_global_bwd", (size_t)L,
        PointsGlobalBwdArgs{proj, rig, dOut, dProj, dRig, (uint32_t)L, (uint32_t)H, (uint32_t)P,
                            accumulateProj ? 1u : 0u, accumulateRig ? 1u : 0u, 0});
}

// ---------------------------------------------------------------- InvariantPointAttention, the core
static float* sbuf(const char* name, size_t n) {
  return (float*)scratchBytes(std::string("design.ipa.") + name, n * sizeof(float));
}

void ipaWeightsBwd(const IpaWeightsSaved& s, const float* dAttn, int L, int H, int Cs, int Pq, float pointBase,
                   IpaWeightsGrads& g, bool accumulateShared) {
  if (!L) return;
  const size_t lh = (size_t)L * H;
  float* dLogit = sbuf("dlogit", lh * (size_t)L);
  float* dpwPart = sbuf("dpwpart", lh);
  run1d("lf_ipa_dlogit", lh, IpaDlogitArgs{s.attn, dAttn, dLogit, (uint32_t)L, (uint32_t)H, 0, 0});
  run1d("lf_ipa_dq", lh,
        IpaDqArgs{dLogit, s.ks, s.qp, s.kp, s.pw, g.dqs, g.dqp, dpwPart,
                  (uint32_t)L, (uint32_t)H, (uint32_t)Cs, (uint32_t)Pq});
  run1d("lf_ipa_dk", lh,
        IpaDkArgs{dLogit, s.qs, s.qp, s.kp, s.pw, g.dks, g.dkp,
                  (uint32_t)L, (uint32_t)H, (uint32_t)Cs, (uint32_t)Pq});
  if (g.db2d)
    run1d("lf_ipa_db2d", lh * (size_t)L,
          IpaDb2dArgs{dLogit, g.db2d, (uint32_t)L, (uint32_t)H, accumulateShared ? 1u : 0u, 0});
  if (g.dPointWeightRaw)
    run1d("lf_ipa_dpw", (size_t)H,
          IpaDpwArgs{dpwPart, s.pointWeightRaw, g.dPointWeightRaw, (uint32_t)L, (uint32_t)H, pointBase,
                     accumulateShared ? 1u : 0u});
}

void ipaOutputsBwd(const IpaOutputsSaved& s, const float* dFinal, int L, int H, int Cs, int Pv, int C2,
                   IpaOutputsGrads& g, bool accumulateRig, bool accumulateAct2d) {
  if (!L) return;
  const size_t lh = (size_t)L * H;
  float* dg = sbuf("dg", lh * (size_t)Pv * 3);
  float* dRigPart = sbuf("drigpart", lh * 12);
  run1d("lf_ipa_out_dq", lh,
        IpaOutDqArgs{s.attn, dFinal, s.vs, s.vp, s.act2d, s.rig, g.dAttn, dg, dRigPart,
                     (uint32_t)L, (uint32_t)H, (uint32_t)Cs, (uint32_t)Pv, (uint32_t)C2, 0});
  if (g.dRig)
    run1d("lf_ipa_out_drig", (size_t)L,
          IpaOutDrigArgs{dRigPart, g.dRig, (uint32_t)L, (uint32_t)H, accumulateRig ? 1u : 0u, 0});
  run1d("lf_ipa_out_dv", lh,
        IpaOutDvArgs{s.attn, dFinal, dg, g.dvs, g.dvp, (uint32_t)L, (uint32_t)H, (uint32_t)Cs, (uint32_t)Pv,
                     (uint32_t)C2, 0});
  if (g.dAct2d)
    run1d("lf_ipa_out_d2d", (size_t)L * L,
          IpaOutD2dArgs{s.attn, dFinal, g.dAct2d, (uint32_t)L, (uint32_t)H, (uint32_t)Cs, (uint32_t)Pv,
                        (uint32_t)C2, accumulateAct2d ? 1u : 0u});
}

// ---------------------------------------------------------------- InvariantPointAttention, the whole module
void invariantPointAttentionBwd(const IpaSaved& s, const float* dOut, const IpaDims& d, const IpaWeightSet& w,
                                IpaGradSet& g) {
  const int L = d.L, H = d.H, Cs = d.Cs, Pq = d.Pq, Pv = d.Pv, C = d.C, C2 = d.C2;
  if (!L) return;
  const int Fw = H * Cs + 4 * H * Pv + H * C2, W = H * Cs;
  const size_t lh = (size_t)L * H;
  float* dFin = sbuf("dfin", (size_t)L * Fw);
  float* dAttn = sbuf("dattn", lh * (size_t)L);
  float* dvs = sbuf("dvs", lh * Cs);
  float* dvp = sbuf("dvp", lh * (size_t)Pv * 3);
  float* dqs = sbuf("dqs", lh * Cs);
  float* dks = sbuf("dks", lh * Cs);
  float* dqp = sbuf("dqp", lh * (size_t)Pq * 3);
  float* dkp = sbuf("dkp", lh * (size_t)Pq * 3);
  float* dProj = sbuf("dproj", lh * 3 * (size_t)(Pq > Pv ? Pq : Pv));

  // the output projection, whose beta is 1 - so dAct STARTS as the cotangent of this module's own output
  linearBwd(s.fin, w.Wo, dOut, dFin, g.dWo, g.dbo, (size_t)L, Fw, C, false, true);
  copy(g.dAct, dOut, (size_t)L * C * sizeof(float));

  IpaOutputsGrads og{dAttn, dvs, dvp, g.dAct2d, g.dRig};
  ipaOutputsBwd({s.attn, s.vs, s.vp, s.act2d, s.rig}, dFin, L, H, Cs, Pv, C2, og, true, true);

  IpaWeightsGrads wg{dqs, dks, dqp, dkp, g.db2d, g.dPointWeightRaw};
  ipaWeightsBwd({s.attn, s.qs, s.ks, s.qp, s.kp, w.pw, w.pointWeightRaw}, dAttn, L, H, Cs, Pq, w.pointBase, wg,
                true);

  // the three point projections: each one's local cotangent, and a fourth contribution to the rigid
  pointsGlobalBwd(s.projQ, s.rig, dqp, dProj, g.dRig, L, H, Pq, false, true);
  linearBwd(s.act, w.Wqp, dProj, g.dAct, g.dWqp, g.dbqp, (size_t)L, C, H * 3 * Pq, true, true);
  pointsGlobalBwd(s.projK, s.rig, dkp, dProj, g.dRig, L, H, Pq, false, true);
  linearBwd(s.act, w.Wkp, dProj, g.dAct, g.dWkp, g.dbkp, (size_t)L, C, H * 3 * Pq, true, true);
  pointsGlobalBwd(s.projV, s.rig, dvp, dProj, g.dRig, L, H, Pv, false, true);
  linearBwd(s.act, w.Wvp, dProj, g.dAct, g.dWvp, g.dbvp, (size_t)L, C, H * 3 * Pv, true, true);

  // the three scalar projections; the query carries the forward's 1/sqrt(Cs)
  run1d("lf_scale", lh * Cs, ScaleArgs{dqs, lh * (size_t)Cs, w.qScale, 0});
  linearBwd(s.act, w.Wq, dqs, g.dAct, g.dWq, g.dbq, (size_t)L, C, W, true, true);
  linearBwd(s.act, w.Wk, dks, g.dAct, g.dWk, g.dbk, (size_t)L, C, W, true, true);
  linearBwd(s.act, w.Wv, dvs, g.dAct, g.dWv, g.dbv, (size_t)L, C, W, true, true);
}

// ---------------------------------------------------------------- FoldIteration, the whole iteration
void foldIterationBwd(const FoldIterSaved& s, const IpaDims& d, const FoldIterWeights& w, FoldIterGrads& g) {
  const int L = d.L, C = d.C;
  if (!L) return;
  const size_t n = (size_t)L * C;
  float* dAct = g.ipa.dAct;      // actOut's cotangent on entry, actIn's on return
  float* dRig = g.ipa.dRig;      // rigOut's on entry, rigIn's on return

  // the backbone update. Its kernel cannot read and write one buffer, so rigOut's cotangent is copied aside.
  float* dRigNext = sbuf("drignext", (size_t)L * 12);
  float* dRq = sbuf("drq", (size_t)L * 6);
  copy(dRigNext, dRig, (size_t)L * 12 * sizeof(float));
  rigidUpdateBwd(s.ipa.rig, s.rq, dRigNext, dRig, dRq, L, false);
  // quat_rigid reads actOut, which the next iteration also read - so this ADDS to the cotangent already there
  linearBwd(s.actOut, w.Wq6, dRq, dAct, g.dWq6, g.dbq6, (size_t)L, C, 6, true, true);

  // transition_layer_norm
  float* dT2 = sbuf("dt2", n);
  layerNormBwd(s.t2, dAct, dT2, (size_t)L, C, w.lnTransScale, g.dLnTransScale, g.dLnTransOffset, 1e-5f, 0, 0, 0,
               true);

  // the transition, backwards, with its residual: t2 = c + t1, so dT1 starts as dT2
  float* dT1 = sbuf("dt1", n);
  float* dB = sbuf("db", n);
  float* dA = sbuf("da", n);
  copy(dT1, dT2, n * sizeof(float));
  linearBwd(s.b, w.W3, dT2, dB, g.dW3, g.db3, (size_t)L, C, C, false, true);
  reluBwd(dB, s.b, dB, n);
  linearBwd(s.a, w.W2, dB, dA, g.dW2, g.db2, (size_t)L, C, C, false, true);
  reluBwd(dA, s.a, dA, n);
  linearBwd(s.t1, w.W1, dA, dT1, g.dW1, g.db1, (size_t)L, C, C, true, true);

  // attention_layer_norm, then the IPA, which writes actIn's cotangent into dAct
  float* dActA = sbuf("dacta", n);
  layerNormBwd(s.actA, dT1, dActA, (size_t)L, C, w.lnAttnScale, g.dLnAttnScale, g.dLnAttnOffset, 1e-5f, 0, 0, 0,
               true);
  invariantPointAttentionBwd(s.ipa, dActA, d, w.ipa, g.ipa);
}

}  // namespace mt
