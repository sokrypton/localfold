// ESMFold2's kernel arguments (metal/core/args.h conventions: one declaration for the host and the kernels)
#pragma once
#include "args.h"

// the relative-position features' indices and table (z_init, the diffusion's conditioning, the confidence head)
struct RelIdx { CP(int) ri; CP(int) asym; CP(int) sym; CP(int) ent; CP(int) ti; CP(float) Wt; };

// a weight interleaved for gemmSwiglu: out [rows][2I] in blocks of 8 - a_0..a_7 b_0..b_7 a_8.. - from a and b rows lda apart
struct Interleave8Args { CP(half) a; CP(half) b; DP(half) out; uint rows, I, lda, pad; };
struct StackColsArgs { CP(half) a; CP(half) b; CP(half) c; DP(half) out; uint rows, ca, cb, cc; };

// ---------------------------------------------------------------- the language model
struct EmbedArgs { CP(int) ids; CP(float) table; DP(float) x; uint rows, C; };
struct RopeArgs { DP(float) x; DP(float) y; uint rows, heads, ld, pad; };
struct QkNormRopeArgs { CP(float) qkv; CP(float) qScale; CP(float) kScale; DP(float) q; DP(float) k; uint rows, C; };
struct ShimNormAccArgs {
  CP(float) x; CP(float) shimScale; CP(float) shimOffset; CP(float) attnScale; CP(float) attnOffset; CP(float) finalScale;
  DP(float) mixXn; DP(half) attnXn; uint rows, C, first; float weight;
};
struct SoftmaxSeqArgs { DP(float) S; CP(int) seq; uint rows, total; float scale; uint pad; };
struct EsmcAttendArgs { CP(float) q; CP(float) k; CP(float) v; CP(int) seq; DP(half) ctx; uint rows, C, ldv, pad; };
struct ScatterRowsArgs { CP(float) rows; CP(int) tokenToRow; CP(float) zero; DP(float) out; uint T, C; };
struct PairJoinArgs { CP(float) s; DP(half) out; uint T, i0, bi, C; };
struct PairJoinFArgs { CP(float) s; DP(float) out; uint T, i0, bi, C; };   // (float: a tower whose a * a passes half's range)

// ---------------------------------------------------------------- the trunk
struct ZInitArgs { CP(float) rows; CP(float) cols; RelIdx rel; CP(float) bonds; CP(float) wBond; CP(float) lmZ; DP(float) z; uint T, C; };
// the triangle's projection and gate as gemmTriGate's one weight: channel c's (pa ga pb gb) in blocks of 8 or quarters of 32
struct TriGateWArgs { CP(half) proj; CP(half) gate; DP(half) out; uint rows, C, quarters, pad; };
struct FillFArgs { DP(float) out; u64 n; float v; uint pad; };
// the released models' recycle (parcae): the initial state, the language model's pair dropped out, the decay update
struct TruncNormalArgs { DP(float) z; u64 n; u64 seed; float std; uint pad; };
struct DropoutArgs { CP(float) in; DP(float) out; u64 n; u64 seed; u64 stream; float p; uint pad; };
struct DecayArgs { DP(float) z; CP(float) y; CP(float) decay; u64 n; uint C, pad; };
// the full ESMFold2's MSA encoder (trunk.mm msaEncode; cuda/ef2's msa.cuh)
struct MsaEmbedArgs { CP(int) rows; CP(float) del; CP(float) mask; CP(float) E; CP(float) fromInputs; DP(float) m; u64 count; uint T, C; };
struct OpmSplitArgs { CP(float) x; CP(float) mask; DP(float) a; DP(float) b; u64 rows; uint H, pad; };
struct OpmPermuteArgs { CP(float) P; DP(float) Pp; uint bi, T, H, pad; };
struct OpmAddArgs { DP(float) pair; CP(float) X; CP(float) bias; CP(float) count; uint i0, bi, T, C; };
struct PwaSoftmaxArgs { CP(float) flat; DP(float) w; uint T, heads; };
struct PwaHeadsArgs { CP(float) in; DP(float) out; CP(float) gate; uint S, T, heads, d; };
struct SymRowsArgs { CP(float) z; DP(float) out; u64 p0, cnt; uint T, C; };
struct ContactsArgs { CP(float) logits; CP(int) contactBins; DP(float) out; u64 pairs; uint bins, pad; };
struct QuantiseArgs { CP(float) x; DP(uchar) out; u64 n; float scale; uint pad; };

// ---------------------------------------------------------------- the atoms
struct AtomFeaturesArgs { CP(float) pos; CP(float) charge; CP(float) mask; CP(int) element; CP(int) nameChars; DP(float) out; DP(half) outH; uint A, pad; };
struct RopeTableArgs { CP(float) pos; CP(int) uid; DP(float) cosT; DP(float) sinT; uint A, pad; };
struct SiluArgs { CP(float) x; DP(float) y; DP(half) yh; u64 n; };
struct SwigluArgs { CP(float) h; DP(float) g; uint rows, F; };
struct RmsModArgs { DP(float) x; CP(float) mod; DP(float) y; DP(half) yh; CP(float) resD; uint rows, C, shift, scale, resWhich, pad; };
struct QkvPrepArgs { DP(float) qkv; CP(float) cosT; CP(float) sinT; uint A, C, heads, bf16, ld, pad; };
struct SwaArgs { CP(float) qkv; CP(int) valid; DP(float) ctx; uint nValid, C, halfWindow; float scale; uint ld, pad; DP(half) outH; };
struct GateLiveArgs { DP(float) ctx; CP(float) gate; CP(float) mask; DP(half) outH; uint A, C, ld, pad; };
struct GatedAddArgs { DP(float) x; CP(float) mod; CP(float) d; uint A, C, which, pad; };
struct ScatterMeanArgs { CP(float) v; CP(int) tokenStart; CP(int) tokenAtoms; CP(float) mask; DP(float) out; uint T, C, ld, pad; CP(float) add; };
struct TailInputsArgs { CP(float) aatype; CP(float) profile; CP(float) delMean; DP(float) out; uint T, C, K, ld; };

// ---------------------------------------------------------------- the diffusion module
struct JoinPairRelArgs { CP(float) z; RelIdx rel; DP(float) out; u64 p0, n; uint T, C; CP(float) scale; CP(float) offset; DP(half) outH; };
struct SiluMulArgs { DP(float) a; CP(float) b; u64 n; };
struct FourierArgs { CP(float) w; CP(float) b; CP(float) level; DP(float) out; uint n, pad; };
struct AddRowArgs { CP(float) base; CP(float) row; DP(float) x; u64 rows; uint C, pad; };
struct CoordsInputArgs { CP(float) x; CP(float) level; DP(float) out; DP(half) outH; uint A, C; CP(float) c0; CP(half) wH; };
struct AdaCombineArgs { DP(float) an; CP(float) g; CP(float) gb; CP(float) sh; DP(half) out; CP(float) resY; CP(float) resGate; uint T, C, ld, pad; };
struct SigmoidMulArgs { DP(float) x; CP(float) g; CP(float) gb; DP(float) dst; uint T, C, ld, pad; };
struct ScaleCopiesArgs { CP(float) x; CP(float) scales; DP(float) out; DP(half) outH; uint rows, C, n, pad; CP(float) offsets; };
struct BiasSoftmaxArgs { DP(float) S; CP(half) bias; uint T; float scale; };
struct PairToHeadsArgs { CP(float) pb; DP(half) out; u64 P; uint H, T, stride, blocks; };
struct GatherTokensArgs { CP(float) perToken; CP(int) atomToToken; CP(float) mask; DP(float) q; uint A, C; };
struct EdmArgs { CP(float) xNoisy; CP(float) r; DP(float) out; CP(float) level; uint n, pad; };
struct DecoderOutArgs { CP(float) q; CP(float) scale; CP(float) offset; CP(half) wH; CP(float) xNoisy; CP(float) level;
                        DP(float) out; uint A, C, halfGemm, pad; };

// ---------------------------------------------------------------- the confidence head
struct ConfZArgs { DP(float) z; RelIdx rel; CP(float) bonds; CP(float) wBond; CP(float) rows; CP(float) cols; uint T, C;
                   CP(float) scale; CP(float) offset; };
struct OuterArgs { CP(float) a; CP(float) b; DP(half) out; u64 p0, n; uint T, C; };
struct DistEmbedArgs { DP(float) z; CP(float) x; CP(int) rep; CP(float) edges; CP(float) table; uint nEdges, T, C, pad;
                       DP(half) zh; DP(half) lnOut; CP(float) lnScale; CP(float) lnOffset; };
struct AddHalfArgs { DP(float) z; CP(half) h; u64 n4; };
struct RowPoolArgs { CP(float) z; DP(float) score; DP(float) pooled; uint T, C; };
struct GatherAtomsArgs { CP(float) tok; CP(int) atomToToken; DP(float) out; uint A, C; };
struct PlddtArgs { CP(float) s; CP(int) atomToToken; CP(int) slot; CP(float) table; DP(float) plddt; uint A, C, bins, pad; };
struct PaeArgs { CP(float) logits; DP(float) pae; DP(float) tm; u64 P; uint bins; float width, d0, pad; };
struct TmRowsArgs { CP(float) tm; CP(int) asym; DP(float) rows; uint T, pad; };
