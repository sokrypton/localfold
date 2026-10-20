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
// `accumulateShared` adds into db2d and dPointWeightRaw instead of writing them - the pair bias and the point
// weight are ONE set of parameters read by all eight fold iterations.
void ipaWeightsBwd(const IpaWeightsSaved& saved, const float* dAttn, int L, int H, int Cs, int Pq,
                   float pointBase, IpaWeightsGrads& grads, bool accumulateShared = false);

// What the outputs backward reads. `rig` is the forward's rigid at this iteration - the point block comes back into
// the LOCAL frame through it, so the rigid takes a gradient here as well as through the point projections.
struct IpaOutputsSaved { const float* attn; const float* vs; const float* vp; const float* act2d; const float* rig; };
// dAttn, dvs, dvp are written; dAct2d and dRig may be null, and dRig may accumulate (the same rigid feeds the point
// projections of the same iteration).
struct IpaOutputsGrads { float* dAttn; float* dvs; float* dvp; float* dAct2d; float* dRig; };
void ipaOutputsBwd(const IpaOutputsSaved& saved, const float* dFinal, int L, int H, int Cs, int Pv, int C2,
                   IpaOutputsGrads& grads, bool accumulateRig = false, bool accumulateAct2d = false);

// ---------------------------------------------------------------- InvariantPointAttention, the whole module
// The forward, as metal/af2/structure.mm runs it inside each fold iteration:
//
//   qs = qScale (act Wq + bq)                 ks = act Wk + bk            vs = act Wv + bv
//   projQ = act Wqp + bqp   ->  qp = global(projQ, rig)          (and the same for k with Pq, v with Pv)
//   attn = ipaWeights(qs, ks, qp, kp, b2d, pw, seqMask)
//   fin  = ipaOutputs(attn, vs, vp, act2d, rig)
//   out  = fin Wo + bo + act                  <- the output projection's beta is 1: the residual is INSIDE it
//
// qScale is sqrt(1/Cs) and the forward folds it into the GEMM's alpha with a pre-scaled bias; this backward scales
// the cotangent instead, which is the same map.
//
// 🔴 EVERYTHING HERE ACCUMULATES EXCEPT dAct, AND THE CALLER ZEROES. One set of IPA weights runs eight times, one
// pair bias feeds all eight, and the rigid takes a cotangent four times within a single call - through the three
// point projections and again through the outputs' return to the local frame. A written gradient would keep only
// the last of those, so there is no flag to get wrong: zero the destinations once before the loop.
struct IpaDims { int L, C, C2, H, Cs, Pq, Pv; };
struct IpaWeightSet {
  const float* Wq; const float* Wk; const float* bk; const float* Wv; const float* bv;
  const float* Wqp; const float* bqp; const float* Wkp; const float* bkp; const float* Wvp; const float* bvp;
  const float* Wo; const float* bo;
  const float* pw;               // the softplus-scaled point weight the logit reads
  const float* pointWeightRaw;   // the parameter behind it
  float qScale, pointBase;
};
// The forward's own tensors. 🔴 projQ/projK/projV must be THREE buffers: the forward reuses one piece of scratch for
// all three, and dRig needs each projection's LOCAL coordinates, so emitting the save set means keeping them apart.
struct IpaSaved {
  const float* act; const float* act2d; const float* rig;
  const float* qs; const float* ks; const float* vs;
  const float* projQ; const float* projK; const float* projV;
  const float* qp; const float* kp; const float* vp;
  const float* attn; const float* fin;
};
struct IpaGradSet {
  float* dAct;                   // WRITTEN (the residual through the output projection included)
  float* dRig; float* dAct2d; float* db2d; float* dPointWeightRaw;
  float* dWq; float* dbq; float* dWk; float* dbk; float* dWv; float* dbv;
  float* dWqp; float* dbqp; float* dWkp; float* dbkp; float* dWvp; float* dbvp;
  float* dWo; float* dbo;
};
void invariantPointAttentionBwd(const IpaSaved& saved, const float* dOut, const IpaDims& d,
                                const IpaWeightSet& w, IpaGradSet& grads);

// ---------------------------------------------------------------- FoldIteration, the whole iteration
// The forward, as metal/af2/structure.mm runs it eight times:
//
//   actA   = InvariantPointAttention(actIn, act2d, b2d, rigIn)      (its own residual included)
//   t1     = LayerNorm(actA, attention_layer_norm)
//   a      = relu(t1 W1 + b1)      b = relu(a W2 + b2)      c = b W3 + b3        (the three-layer transition)
//   t2     = c + t1                                                              (the transition's residual)
//   actOut = LayerNorm(t2, transition_layer_norm)
//   rq     = actOut Wq6 + bq6                                                    (quat_rigid/rigid)
//   rigOut = rigIn o quat(rq)
//
// `dAct` is BOTH input and output: on entry it holds actOut's cotangent (from the next iteration's input), on
// return actIn's. `dRig` likewise - rigOut's in, rigIn's out - which is what the rigid being a recurrent state
// means. Everything else accumulates and the caller zeroes, as the IPA above.
//
// 🔴 dRig IS READ AND REWRITTEN IN PLACE, which the backbone update's kernel cannot do on one buffer: it writes
// dRigOld[0] before it has read dRigNew[4]. So this copies dRig aside first. Two buffers would be the alternative
// and the caller would then have to swap them eight times.
struct FoldIterSaved {
  IpaSaved ipa;            // the IPA's own, whose `act` is actIn and whose `rig` is rigIn
  const float* actA;       // the IPA's output, before attention_layer_norm
  const float* t1;         // after it
  const float* a;          // relu(t1 W1 + b1)
  const float* b;          // relu(a W2 + b2)
  const float* t2;         // c + t1, before transition_layer_norm
  const float* actOut;     // after it - this iteration's output, and quat_rigid's input
  const float* rq;         // actOut Wq6 + bq6
};
struct FoldIterWeights {
  IpaWeightSet ipa;
  const float* lnAttnScale; const float* lnAttnOffset;
  const float* W1; const float* b1; const float* W2; const float* b2; const float* W3; const float* b3;
  const float* lnTransScale; const float* lnTransOffset;
  const float* Wq6; const float* bq6;
};
struct FoldIterGrads {
  IpaGradSet ipa;          // ipa.dAct is this iteration's dAct in and out; ipa.dRig the shared rigid cotangent
  float* dLnAttnScale; float* dLnAttnOffset;
  float* dW1; float* db1; float* dW2; float* db2; float* dW3; float* db3;
  float* dLnTransScale; float* dLnTransOffset;
  float* dWq6; float* dbq6;
};
void foldIterationBwd(const FoldIterSaved& saved, const IpaDims& d, const FoldIterWeights& w, FoldIterGrads& grads);

}  // namespace mt
