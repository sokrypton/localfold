// AlphaFold 2's kernel arguments (metal/core/args.h conventions: one declaration for the host and the kernels)
#pragma once
#include "args.h"

// ---------------------------------------------------------------- the embedder
struct TargetFeatArgs { CP(int) aatype; DP(float) tf; uint L, pad; };
struct BroadcastRowsArgs { DP(float) msa; CP(float) row; u64 rows; uint L, C; };
struct OuterSumArgs { DP(float) pair; CP(float) left; CP(float) right; uint L, C; };
struct PrevDgramArgs { CP(float) pos37; CP(int) aatype; DP(float) out; uint L, pad; };
struct RelposArgs { CP(int) ri; CP(int) asym; CP(int) entity; CP(int) sym; DP(float) out; uint L, pad; };
struct ExtraFeatArgs { CP(int) codes; CP(float) hasDel; CP(float) delVal; DP(float) out; u64 rows; };
struct PairMaskArgs { CP(float) seqMask; DP(float) out; uint L, pad; };

// ---------------------------------------------------------------- the Evoformer
// a [pairs][H] projection (+ 1e9 (mask - 1)) -> the attention's bias [H][L][L] in log2 units; transposed: b[h][q][k]
// = proj[k][q] (the ending node)
struct BiasLayoutArgs { CP(float) proj; CP(float) pairMask; DP(half) out; uint L, H, transposed, pad; };
// the triangle's projection and gate (AF2's halves: a's C columns, then b's) as gemmTriGate's weight and bias
struct TriGateW2Args { CP(half) proj; CP(half) gate; CP(float) pb; CP(float) gb; DP(half) w4; DP(float) b4; uint C, pad; };
struct QkvgWArgs { CP(half) q; CP(half) k; CP(half) v; CP(half) g; DP(half) out; uint C, W; };
struct GateBiasArgs { CP(float) gb; DP(float) out; uint W, pad; };
struct ScaleRowsHArgs { DP(half) x; CP(float) mask; u64 rows; uint C, pad; };
struct OpmPermuteArgs { CP(half) Pm; DP(half) X; uint bi, L, O, pad; };
struct OpmAddArgs { DP(float) pair; CP(float) Y; CP(float) bias; CP(float) norm; u64 i0; uint bi, L, C, pad; };
struct OpmLeftArgs { CP(half) lt; DP(half) out; uint S, L, O; float scale; };
struct TileBiasArgs { CP(float) bias; DP(float) out; uint L, C; float scale; uint pad; };
struct MaskNormArgs { CP(float) mask; DP(float) norm; uint S, L; };
struct GlobalAttnArgs { CP(half) xn; CP(float) kv; CP(float) mask; CP(half) qw; DP(float) avg; uint S, L, C, H, D, pad; };
struct GlobalGateArgs { CP(float) avg; CP(float) gate; DP(half) out; uint S, L, W, pad; };

// ---------------------------------------------------------------- templates
struct TmplPairArgs { CP(int) aatype; CP(float) pos; CP(float) mask; CP(float) w0; CP(float) w1; CP(float) w2; CP(float) w3;
                      CP(float) w4; CP(float) w5; CP(float) w6; CP(float) w7; CP(float) bsum; DP(float) act; uint L, C; };
struct TmplPairMonoArgs { CP(int) aatype; CP(float) pos; CP(float) mask; CP(float) w; CP(float) b; DP(float) act; uint L, C; };
struct ReluScaleArgs { DP(float) x; u64 n; float s; uint pad; };
struct PointAttnArgs { CP(float) q; CP(float) k; CP(float) v; DP(float) out; u64 pairs; uint T, H, D, pad; };
struct TorsionFeatArgs { CP(int) aatype; CP(float) pos; CP(float) mask; DP(float) feat; DP(float) rowMask; uint L, pad; };

// ---------------------------------------------------------------- the structure module and the heads
struct IdentityRigidArgs { DP(float) r; uint L, pad; };
struct PointsGlobalArgs { CP(float) proj; CP(float) rig; DP(float) out; uint L, H, P, pad; };
struct IpaWeightsArgs { CP(float) qs; CP(float) ks; CP(float) qp; CP(float) kp; CP(float) b2d; CP(float) pw; CP(float) seqMask;
                        DP(float) attn; uint L, H, Cs, Pq; };
struct IpaOutputsArgs { CP(float) attn; CP(float) vs; CP(float) vp; CP(float) act2d; CP(float) rig; DP(float) fin;
                        uint L, H, Cs, Pv, C2, pad; };
struct RigidUpdateArgs { DP(float) rig; CP(float) upd; uint L, pad; };
struct ReluCopyArgs { CP(float) x; DP(float) y; u64 n; };
struct SoftplusArgs { CP(float) raw; DP(float) out; uint H; float base; };
struct SidechainArgs { CP(float) unnorm; CP(float) rig; CP(int) aatype; CP(float) defaultFrames; CP(int) atom14Group;
                       CP(float) litPos; CP(float) atom14Mask; CP(int) atom37To14; CP(float) atom37Mask; CP(float) seqMask;
                       DP(float) angles; DP(float) pos14; DP(float) pos37; uint L; float positionScale; };
struct PaeTmArgs { CP(float) logits; DP(float) pae; DP(float) tm; u64 pairs; float d0; uint pad; };
struct SymmetriseArgs { CP(float) x; DP(float) y; uint L, C; };
struct Contact8Args { CP(float) logits; DP(float) out; u64 pairs; };
