#include "structgrad.h"
#include "kernels.h"

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

}  // namespace mt
