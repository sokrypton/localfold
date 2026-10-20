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

// ---------------------------------------------------------------- InvariantPointAttention, the core (Alg. 22)
// The two halves of the IPA that are not linear layers: the attention weights and the outputs they weight. See
// kernels.h for the forward each is the backward of, and for why each runs in three passes.

// What the weights backward reads: the forward's own saved tensors, nothing recomputed.
//   attn  [L][H][L] the softmax output      qs/ks  [L][H][Cs]      qp/kp  [L][H][Pq][3], in the GLOBAL frame
//   pw    [H] the softplus-scaled point weight the logit uses
//   pointWeightRaw  [H] the parameter behind it, needed only for dPointWeightRaw
struct IpaWeightsSaved { const float* attn; const float* qs; const float* ks; const float* qp; const float* kp;
                         const float* pw; const float* pointWeightRaw; };
// Every one is WRITTEN, not accumulated. db2d is [L][L][H] - the pair bias's own layout, the head axis innermost -
// and dPointWeightRaw may be null; the rest may not.
struct IpaWeightsGrads { float* dqs; float* dks; float* dqp; float* dkp; float* db2d; float* dPointWeightRaw; };
void ipaWeightsBwd(const IpaWeightsSaved& saved, const float* dAttn, int L, int H, int Cs, int Pq,
                   float pointBase, IpaWeightsGrads& grads);

// What the outputs backward reads. `rig` is the forward's rigid at this iteration - the point block comes back into
// the LOCAL frame through it, so the rigid takes a gradient here as well as through the point projections.
struct IpaOutputsSaved { const float* attn; const float* vs; const float* vp; const float* act2d; const float* rig; };
// dAttn, dvs, dvp are written; dAct2d and dRig may be null, and dRig may accumulate (the same rigid feeds the point
// projections of the same iteration).
struct IpaOutputsGrads { float* dAttn; float* dvs; float* dvp; float* dAct2d; float* dRig; };
void ipaOutputsBwd(const IpaOutputsSaved& saved, const float* dFinal, int L, int H, int Cs, int Pv, int C2,
                   IpaOutputsGrads& grads, bool accumulateRig = false);

}  // namespace mt
