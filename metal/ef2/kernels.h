// ESMFold2's kernel arguments (metal/core/args.h conventions: one declaration for the host and the kernels)
#pragma once
#include "args.h"

// the relative-position features' indices and table (z_init, the diffusion's conditioning, the confidence head)
struct RelIdx { CP(int) ri; CP(int) asym; CP(int) sym; CP(int) ent; CP(int) ti; CP(float) Wt; };

// a weight interleaved for gemmSwiglu: out [rows][2I] in blocks of 8 - a_0..a_7 b_0..b_7 a_8.. - from a and b rows lda apart
struct Interleave8Args { CP(half) a; CP(half) b; DP(half) out; uint rows, I, lda, pad; };

// ---------------------------------------------------------------- the language model
struct EmbedArgs { CP(int) ids; CP(float) table; DP(float) x; uint rows, C; };
struct RopeArgs { DP(float) x; uint rows, heads, ld, pad; };
struct SoftmaxSeqArgs { DP(float) S; CP(int) seq; uint rows; float scale; };
struct ScatterRowsArgs { CP(float) rows; CP(int) tokenToRow; CP(float) zero; DP(float) out; uint T, C; };
struct PairJoinArgs { CP(float) s; DP(half) out; uint T, i0, bi, C; };

// ---------------------------------------------------------------- the trunk
struct ZInitArgs { CP(float) rows; CP(float) cols; RelIdx rel; CP(float) bonds; CP(float) wBond; CP(float) lmZ; DP(float) z; uint T, C; };
// the triangle's projection and gate as gemmTriGate's one weight: channel c's (pa ga pb gb) in blocks of 8
struct TriGateWArgs { CP(half) proj; CP(half) gate; DP(half) out; uint rows, C; };
struct FillFArgs { DP(float) out; u64 n; float v; uint pad; };
struct SymRowsArgs { CP(float) z; DP(float) out; u64 p0, cnt; uint T, C; };
struct ContactsArgs { CP(float) logits; CP(int) contactBins; DP(float) out; u64 pairs; uint bins, pad; };
struct QuantiseArgs { CP(float) x; DP(uchar) out; u64 n; float scale; uint pad; };

// ---------------------------------------------------------------- the atoms
struct AtomFeaturesArgs { CP(float) pos; CP(float) charge; CP(float) mask; CP(int) element; CP(int) nameChars; DP(float) out; uint A, pad; };
struct RopeTableArgs { CP(float) pos; CP(int) uid; DP(float) cosT; DP(float) sinT; uint A, pad; };
struct SiluArgs { CP(float) x; DP(float) y; u64 n; };
struct SwigluArgs { CP(float) h; DP(float) g; uint rows, F; };
struct RmsModArgs { CP(float) x; CP(float) mod; DP(float) y; uint rows, C, shift, scale; };
struct QkvPrepArgs { DP(float) qkv; CP(float) cosT; CP(float) sinT; uint A, C, heads, bf16; };
struct SwaArgs { CP(float) qkv; CP(int) valid; DP(float) ctx; uint nValid, C, halfWindow; float scale; };
struct GateLiveArgs { DP(float) ctx; CP(float) gate; CP(float) mask; uint A, C; };
struct GatedAddArgs { DP(float) x; CP(float) mod; CP(float) d; uint A, C, which, pad; };
struct ScatterMeanArgs { CP(float) v; CP(int) tokenStart; CP(int) tokenAtoms; CP(float) mask; DP(float) out; uint T, C, ld, pad; };
struct TailInputsArgs { CP(float) aatype; CP(float) profile; CP(float) delMean; DP(float) out; uint T, C, K, ld; };

// ---------------------------------------------------------------- the diffusion module
struct JoinPairRelArgs { CP(float) z; RelIdx rel; DP(float) out; u64 p0, n; uint T, C; };
struct SiluMulArgs { DP(float) a; CP(float) b; u64 n; };
struct FourierArgs { CP(float) w; CP(float) b; CP(float) level; DP(float) out; uint n, pad; };
struct AddRowArgs { DP(float) x; CP(float) row; u64 rows; uint C, pad; };
struct CoordsInputArgs { CP(float) x; CP(float) level; DP(float) out; uint A, pad; };
struct AdaCombineArgs { CP(float) an; CP(float) g; CP(float) gb; CP(float) sh; DP(half) out; uint T, C, ld, pad; };
struct SigmoidMulArgs { DP(float) x; CP(float) g; CP(float) gb; uint T, C, ld, pad; };
struct ScaleCopiesArgs { CP(float) x; CP(float) scales; DP(float) out; uint rows, C, n, pad; };
struct BiasSoftmaxArgs { DP(float) S; CP(half) bias; uint T; float scale; };
struct PairToHeadsArgs { CP(float) pb; DP(half) out; u64 P; uint H, T, stride, pad; };
struct GatherTokensArgs { CP(float) perToken; CP(int) atomToToken; CP(float) mask; DP(float) q; uint A, C; };
struct EdmArgs { CP(float) xNoisy; CP(float) r; DP(float) out; CP(float) level; uint n, pad; };

// ---------------------------------------------------------------- the confidence head
struct ConfZArgs { DP(float) z; RelIdx rel; CP(float) bonds; CP(float) wBond; CP(float) rows; CP(float) cols; uint T, C; };
struct OuterArgs { CP(float) a; CP(float) b; DP(half) out; u64 p0, n; uint T, C; };
struct DistEmbedArgs { DP(float) z; CP(float) x; CP(int) rep; CP(float) edges; CP(float) table; uint nEdges, T, C, pad; };
struct RowPoolArgs { CP(float) z; CP(float) score; DP(float) pooled; uint T, C; };
struct GatherAtomsArgs { CP(float) tok; CP(int) atomToToken; DP(float) out; uint A, C; };
struct PlddtArgs { CP(float) s; CP(int) slot; CP(float) table; DP(float) plddt; uint A, C, bins, pad; };
struct PaeArgs { CP(float) logits; DP(float) pae; DP(float) tm; u64 P; uint bins; float width, d0, pad; };
struct TmRowsArgs { CP(float) tm; CP(int) asym; DP(float) rows; uint T, pad; };
