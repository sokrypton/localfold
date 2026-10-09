// The AlphaFold 3 lineage's kernel arguments (metal/core/args.h conventions: one declaration for the host and the
// kernels). Pointers first, then 8-byte scalars, then 4-byte ones in pairs.
#pragma once
#include "args.h"

// ---------------------------------------------------------------- derived weights
// dst[c][off + o] = transposed ? src[o][c] : src[c][o] (c < C, o < width; dst rows `total` wide)
struct ConcatPartArgs { CP(half) src; DP(half) dst; uint C, width, total, off, transposed, pad; };
// a weight interleaved for gemmSwiglu: out [rows][2I] in blocks of 8 - a_0..a_7 b_0..b_7 a_8.. - a and b rows lda apart
struct Interleave8Args { CP(half) a; CP(half) b; DP(half) out; uint rows, I, lda, pad; };
// the triangle's projection and gate (a and b interleaved by channel) as gemmTriGate's weight: (pa ga pb gb) in 8s
struct TriGateWArgs { CP(half) proj; CP(half) gate; DP(half) out; uint rows, C; };
// x[r][c] (*)= scale
struct ScaleArgs { DP(float) x; u64 n; float s; uint relu; };
struct ScaleRowsArgs { DP(float) x; CP(float) mask; u64 rows; uint C, period; };      // x[r] *= mask[r % period]
struct ScaleRowsHArgs { DP(half) x; CP(float) mask; u64 rows; uint C, pad; };
struct CastArgs { CP(float) x; DP(half) y; u64 n; };

// ---------------------------------------------------------------- the embedder
struct RelIdx { CP(int) ri; CP(int) ti; CP(int) asym; CP(int) ent; CP(int) sym; };
// pair[i][j] = left[i] + right[j] (+ add[ij])
struct OuterSumArgs { CP(float) left; CP(float) right; CP(float) add; DP(float) pair; uint n, C; };
// pair += the 139 relative-encoding rows a pair's one-hot selects
struct RelEncArgs { RelIdx r; CP(float) W; DP(float) pair; uint n, C; };
// pair += bonds[ij] w + (boltz2) table[order[ij]] + unspecified
struct BondEmbedArgs { DP(float) pair; CP(float) bonds; CP(float) w; CP(float) orders; CP(float) table; CP(float) unspecified;
                       u64 pairs; uint C, pad; };
struct MsaEmbedArgs { CP(int) rows; CP(float) del; CP(float) W; CP(float) fromTarget; DP(float) msa; u64 count; uint n, C, width, pairedQuery; };
// the template stack's inputs
struct OnehotArgs { CP(int) idx; DP(float) out; uint n, classes; };
struct AddRowColArgs { DP(float) act; CP(float) row; CP(float) col; uint n, C; };
struct TmplGeomArgs { DP(float) act; CP(int) bin; CP(float) pb; CP(float) uv; CP(float) bb; CP(float) W0; CP(float) W1; CP(float) W4;
                      CP(float) W5; CP(float) W6; CP(float) W7; u64 pairs; uint C, pad; };
struct ScatterRowsArgs { CP(int) idx; CP(float) val; DP(float) dense; u64 rows; uint K, width; };

// ---------------------------------------------------------------- attention biases
// raw [pairs][heads] f32 -> bias [heads][n][stride] f16, times scale (log2 e), (i, j) read at (j, i) with swap
struct BiasLayoutArgs { CP(float) raw; DP(half) bias; uint n, stride, heads, swap, ld, off; float scale; uint pad; };   // raw [pairs][ld], heads from off

// ---------------------------------------------------------------- the MSA stack
struct KeyMaskArgs { CP(float) msaMask; DP(float) keyMask; uint S, n; };
// w[h][i][ld] = softmax_j(flat[(i, j)][h] + mask), a threadgroup a (h, i) row, the padding zero
struct MsaWeightsArgs { CP(float) flat; CP(float) keyMask; DP(half) w; uint n, heads, ld, pad; };
// v [s][j][h d + e] -> [h][j][s][e] (j < np, the rows past n zero)
struct MsaVHeadsArgs { CP(half) v; DP(half) out; uint S, n, heads, d, np, pad; };
// o [h][i][s][e] -> [s][i][h d + e], times sigmoid(gate)
struct MsaFromHeadsArgs { CP(half) o; CP(half) gate; DP(half) out; uint S, n, heads, d; };
struct MaskNormArgs { CP(float) mask; DP(float) norm; uint S, L; };
struct OpmLeftArgs { CP(half) lt; DP(half) out; uint S, L, O; float scale; };
struct OpmPermuteArgs { CP(half) Pm; DP(half) X; uint bi, L, O, pad; };
// pair[i][j] += (b + Y) / (1e-3 + norm), or (after) Y / max(norm, 1) + b; rows i0..
struct OpmAddArgs { DP(float) pair; CP(float) Y; CP(float) bias; CP(float) norm; u64 i0; uint bi, L, C, after; };

// ---------------------------------------------------------------- the distogram
struct SymmetriseArgs { CP(float) x; DP(float) y; uint L, C; float scale; uint pad; };    // y = (x + x^T) scale
struct ContactBinsArgs { CP(int) classes; CP(int) table; DP(int) out; uint n, pad; };
struct ContactProbsArgs { CP(float) logits; CP(int) bins; CP(float) pairMask; DP(float) out; u64 pairs; uint nb, pad; };

// ---------------------------------------------------------------- the atoms
// out[(k count + g)][c] = mask[g] ? src[idx[g] + k srcRows][c] : 0, over `ns` samples
struct ConvertArgs { CP(int) idx; CP(float) mask; CP(float) src; DP(float) out; u64 srcRows; uint count, C, ns, pad; };
struct PerAtomArgs { CP(float) pos; CP(float) mask; CP(int) element; CP(float) charge; CP(int) nameChars; CP(float) Wpos; CP(float) Wmask;
                     CP(float) Welem; CP(float) Wcharge; CP(float) Wname; CP(float) bias; DP(float) act; uint rows, C, rawCharge, pad; };
struct ReluHArgs { CP(float) x; DP(half) y; u64 n; };
// the atom pair conditioning: row[q] + col[k] + valid (offsets W + 1 / (1 + |d|^2) + Wvalid) (+ the trunk pair)
struct AtomPairArgs { CP(float) row; CP(float) col; CP(float) qPos; CP(float) kPos; CP(float) qUid; CP(float) kUid; CP(float) kMask;
                      CP(float) Woff; CP(float) Wdist; CP(float) Wvalid; CP(float) tp; CP(int) tqIdx; CP(float) tqMask; CP(int) tkIdx;
                      CP(float) tkMask; DP(float) pair; uint subsets, queries, keys, Cp, tokens, maskPadded; };
// flat [(s, q, k)][blocks heads] -> block b's [s][h][q][k]
struct AtomLogitsArgs { CP(float) flat; DP(float) out; uint block, nblocks, subsets, heads, queries, keys; };
// the adaptive LayerNorm: ADA(scale) LN(x) + shift (scale, shift read at row % period), to half or float
struct AdaLnArgs { CP(float) x; CP(float) scale; CP(float) shift; DP(half) out; DP(float) outF; u64 rows; uint C, period, ld, raw; float eps; uint pad; };
// rows of 16-byte chunks gathered: out[r] = mask[g] ? in[idx[g] + k srcRows] : 0, r = k count + g
struct GatherRowsArgs { CP(uchar) in; CP(int) idx; CP(float) mask; DP(uchar) out; u64 rows, count, srcRows; uint chunks, pad; };
// one subset's attention: 32 queries against 128 keys, q (+ bias) and the gate in qg [rows][2W], k and v in kv
struct AtomAttnArgs { CP(half) qg; CP(float) qBias; CP(half) kv; CP(float) qMask; CP(float) kMask; CP(float) logits; DP(half) out;
                      uint queries, keys, heads, D, subsets, keyMasked; };
// act += y sigmoid(gate[t % period])
struct GatedResidualArgs { DP(float) act; CP(float) y; CP(float) gate; u64 n, period; };
struct MulHArgs { DP(half) x; CP(half) y; u64 n; };
// act[q] = qStart[q] + qMask[q] (positions gathered to q) W, W [3][C]
struct EncoderStartArgs { CP(float) qStart; CP(float) pos; CP(int) idx; CP(float) gmask; CP(float) Wp; CP(float) qMask; DP(float) act;
                          u64 rows, q1, atoms; uint C, pad; };
// per token: the mean over its real atoms of relu(projected), gathered back to the token's atoms
struct AggregateArgs { CP(float) projected; CP(int) idx; CP(float) gmask; CP(float) atomMask; DP(float) out; u64 rows, q1; uint tokens, dense, Cp, pad; };
// target_feat: [aatype one-hot 31 | profile 31 | deletion mean | atom features 384]
struct TargetFeatArgs { CP(int) aatype; CP(float) profile; CP(float) delMean; CP(float) atom; DP(float) out; uint tokens, pad; };
// boltz2's: the atom columns plus six bias-free projections
struct TargetFeatSumArgs { DP(float) tf; CP(int) aatype; CP(float) profile; CP(float) delMean; CP(int) isDna; CP(int) isRna; CP(int) isLigand;
                           CP(int) isModified; CP(float) wRes; CP(float) wProf; CP(float) wMol; CP(float) wMethod; CP(float) wMod; uint tokens, C; };

// ---------------------------------------------------------------- the diffusion module
// rows [p0, p0 + rows) of [trunk pair (Czt) | the relative one-hot (139)]
struct PairFeatArgs { CP(float) trunkPair; RelIdx r; DP(float) out; u64 p0, rows; uint n, Czt; };
// out[r] = [a[r] | b[r]] with zero columns at p0 and p1 (-1: none)
struct ConcatPadArgs { CP(float) a; CP(float) b; DP(float) out; uint rows, wa, wb; int p0, p1; uint pad; };
struct FourierArgs { CP(float) w; CP(float) b; DP(float) out; uint n; float level; };
// out[k][off + c] = scale[k] w[k][c] (scale null: 1), rows ld wide
struct FoldCondArgs { CP(float) scale; CP(half) w; DP(half) out; uint Cc, C, ld, off; };
// act[r][c] += y[r][c] sigmoid(gate[(r % period) ld + c])
struct GatedResStridedArgs { DP(float) act; CP(float) y; CP(float) gate; u64 rows; uint C, period, ld, pad; };
struct AddBroadcastArgs { DP(float) x; CP(float) v; u64 n, total; };            // x[t] += v[t % n]
// the decoder's start: act[q] = (the query's token's projection + skip[q]) qMask
struct BroadcastSkipArgs { CP(float) proj; CP(int) idx; CP(float) gmask; CP(float) skip; CP(float) qMask; DP(float) act; u64 rows, q1;
                           uint C, tokens, dense, pad; };
// upd[row] = LN(act[row] qMask) W, W [C][3]
struct MaskLnProject3Args { CP(float) act; CP(float) qMask; CP(float) scale; CP(float) offset; CP(float) Wp; DP(float) upd; u64 rows, q1;
                            uint C, pad; };
struct ScalePosArgs { CP(float) x; CP(float) mask; DP(float) y; u64 total, atoms; float in; uint pad; };
struct DenoiseOutArgs { CP(float) x; CP(float) upd; CP(float) mask; DP(float) o; u64 total, atoms; float skip, out; };
// the sampler
struct InitNoiseArgs { DP(float) x; CP(u64) seeds; u64 n3, total; float scale; uint pad; };
struct CentroidArgs { CP(float) x; CP(float) mask; DP(float) c; u64 atoms; };
struct AugmentArgs { DP(float) x; DP(float) noisy; CP(float) mask; CP(float) c; CP(float) rot; CP(u64) seeds; u64 atoms, total; uint step;
                     float injected; };
struct EulerArgs { DP(float) x; CP(float) noisy; CP(float) den; u64 n; float scale; uint pad; };

// ---------------------------------------------------------------- the confidence head
// each pair's distogram bin (-1: none) and squared distance, once a pair
struct ConfBinArgs { CP(float) beta; DP(int) bin; DP(float) sq; uint n, bins; float dmin, dmax; uint caBins, pad; };
// pair[i][j] += left[j] + right[i] + Wd[bin] mask (+ protenix2's distance term)
struct ConfPairInitArgs { DP(float) pair; CP(float) left; CP(float) right; CP(int) bin; CP(float) sq; CP(float) pairMask; CP(float) Wd;
                          CP(float) Wdist; uint n, C, caBins, pad; };
// the expectation over bins of softmax(logits) . centres (symmetricN: logits[ij] + logits[ji]), times scale (mask)
struct ExpectationArgs { CP(float) logits; DP(float) out; CP(float) mask; CP(float) centres; u64 rows; uint bins, symmetricN; float scale; uint pad; };
struct InterChainArgs { DP(float) logits; CP(float) inter; CP(int) asym; uint n, bins; };
struct ClampArgs { DP(float) x; u64 n; float limit; uint pad; };
