// metal/design: the STRUCTURE MODULE's backward, host side - the half of AF2 that turns the single representation
// into coordinates, and the half every remaining boundary array of bindcraft/af/boundary.py comes from (`atoms`,
// `plddt`, `experimentally_resolved_ca`).
//
// The forward is metal/af2/structure.mm. Conventions are grad.h's: a cotangent is d<forward name>, every array is a
// device pointer from metal/core's allocator, and a gradient is written unless a flag says to accumulate.
//
// 🔴 BINDCRAFT2 LETS THE GRADIENT THROUGH THE WHOLE RIGID CHAIN, WHERE AF2 DOES NOT. Upstream AF2 ends each fold
// iteration with `affine.apply_rotation_tensor_fn(jax.lax.stop_gradient)`; in BindCraft2 that line is COMMENTED OUT
// (bindcraft/af/alphafold/model/folding.py:379), so the rigid is a recurrent state whose cotangent flows back
// through all eight iterations rather than being cut at each one. That decides the shape of this backward: the rigid
// carries a gradient between iterations and cannot be treated as a constant input to any of them.
#pragma once
#include "core.h"

namespace mt {

// ---------------------------------------------------------------- the backbone update (Alg. 23)
// The forward (af2_rigid_update): the quaternion (1, upd0, upd1, upd2) normalised, its rotation and the translation
// (upd3, upd4, upd5) composed onto the running rigid - rigNew = rigOld o quat(upd).
//
//   rigOld, upd    the forward's inputs, [L][12] and [L][6]
//   dRigNew        the cotangent of the composed rigid, [L][12]
//   dRigOld        [L][12]; `accumulateRigOld` adds into it, which is what an iteration wants - the previous
//                  rigid also feeds that iteration's point projections and its side chains
//   dUpd           [L][6], always written
void rigidUpdateBwd(const float* rigOld, const float* upd, const float* dRigNew,
                    float* dRigOld, float* dUpd, int L, bool accumulateRigOld = false);

// ---------------------------------------------------------------- the point projections
// The forward (af2_points_global): a head's three blocks of P local coordinates taken into the global frame,
// g = R loc + t. `proj` is [L][H*3*P] (x, then y, then z, a head) and `dOut` is [L][H][P][3], the forward's output
// layout. Both gradients may accumulate: the three point projections of one IPA iteration read ONE rigid, and
// `proj` is a scratch buffer the forward reuses for q, k and v in turn.
void pointsGlobalBwd(const float* proj, const float* rig, const float* dOut,
                     float* dProj, float* dRig, int L, int H, int P,
                     bool accumulateProj = false, bool accumulateRig = false);

}  // namespace mt
