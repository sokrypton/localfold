#include "structgrad.h"
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
                   IpaWeightsGrads& g) {
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
  if (g.db2d) run1d("lf_ipa_db2d", lh * (size_t)L, IpaDb2dArgs{dLogit, g.db2d, (uint32_t)L, (uint32_t)H, 0, 0});
  if (g.dPointWeightRaw)
    run1d("lf_ipa_dpw", (size_t)H,
          IpaDpwArgs{dpwPart, s.pointWeightRaw, g.dPointWeightRaw, (uint32_t)L, (uint32_t)H, pointBase, 0});
}

void ipaOutputsBwd(const IpaOutputsSaved& s, const float* dFinal, int L, int H, int Cs, int Pv, int C2,
                   IpaOutputsGrads& g, bool accumulateRig) {
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
                        (uint32_t)C2, 0});
}

}  // namespace mt
