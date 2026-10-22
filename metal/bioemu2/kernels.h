// BioEmu2's kernel arguments (metal/core/args.h conventions: one declaration for the host and the kernels)
#pragma once
#include "args.h"

// ---------------------------------------------------------------- the atoms and their windowed pair
// (a window is 32 query atoms; its 128 keys are centred on it: key j of window w is atom w*32 - 48 + j)
struct Be2AtomFeatArgs { CP(float) ref; CP(float) charge; CP(int) element; CP(int) names; CP(int) pad; CP(float) wCont;
                         CP(float) wElem; CP(float) wN0; CP(float) wN1; CP(float) wN2; CP(float) wN3; DP(float) c;
                         u64 N; uint C, pad0; };
struct Be2PairGeomArgs { CP(float) ref; CP(int) uid; CP(int) pad; CP(float) wPos; CP(float) wDist; CP(float) wMask;
                         DP(float) pr; uint N, CP_; };
struct Be2PairTokenAddArgs { CP(float) zp; CP(int) atomTok; CP(int) pad; DP(float) pr; uint N, T, CP_, pad0; };
struct Be2PairTailArgs { DP(float) pr; CP(float) cq; CP(float) ck; CP(float) w1; CP(float) w3; CP(float) w5; uint N, pad0; };
// LN(pr) (scale, no offset) -> proj_z: out[w][h][i][j]
struct Be2WindowBiasArgs { CP(float) pr; CP(float) normW; CP(float) projZ; DP(float) out; uint N, H; };
// the sliding-window attention: a threadgroup per (window of a stacked sample, head), a thread per query; qkv [rows][3D]
// (q at 0, k at D, v at 2D), out [rows][D]; N atoms a sample. The fast arm's _h takes qkvg [rows][4D] in half and gates
// its half output in the epilogue
struct Be2WindowAttnArgs { CP(float) qkv; CP(float) qBias; CP(float) bias; CP(int) pad; DP(float) out; uint N, H; };
struct Be2WindowAttnHArgs { CP(half) qkvg; CP(float) qBias; CP(float) bias; CP(int) pad; DP(half) out; uint N, H; };
// the full token attention: a thread per (sample, head, query); bias [H][T][T] shared by the samples
struct Be2TokenAttnArgs { CP(float) qkv; CP(float) qBias; CP(float) bias; CP(int) keyMask; DP(float) out;
                          uint T, D, H, samples; };

// ---------------------------------------------------------------- elementwise
struct Be2UnaryArgs { CP(float) x; DP(float) y; u64 n; uint op, pad0; };            // op 0 relu, 1 silu
struct Be2SiluMulArgs { CP(float) h1; CP(float) h2; DP(float) g; u64 n; };          // g = silu(h1) h2
// the blocked SwiGLU of AF3's transition1: g[r][c] = silu(h[r][c]) h[r][I + c]
struct Be2SwigluBlockedArgs { CP(float) h; DP(float) g; u64 rows; uint I, pad0; };
struct Be2SigmoidMulArgs { CP(float) h; CP(float) g; DP(float) y; u64 n; };         // y = h sigmoid(g)
// x[r][c] += gate[map(r) ldg + c] h[r][c] (map null: the identity)
struct Be2GatedAddArgs { DP(float) x; CP(float) h; CP(float) gate; CP(int) map; u64 rows; uint C, ldg; };
// adaLN: y = scale (LN(x) (1 + alpha) + beta), alpha / beta rows map(r) of the conditioning (stride ldc); a simdgroup a row
// (resH / resGate: an upstream x[r] += resGate[map(r) ldrg] resH[r] fused in place before the norm)
struct Be2AdaLnArgs { DP(float) x; DP(float) y; CP(float) alpha; CP(float) beta; CP(int) map; u64 rows; uint C, ldx, ldc;
                      float scale; DP(half) yh; CP(float) resH; CP(float) resGate; uint ldrg, pad0; };         // (yh: written in half instead of y)
// the coupling's two inputs on the fast arm, normalised directly into half without materialising [q_centre | a] or [a_tok | q]
struct Be2AdaLnQ2SArgs { CP(float) q; DP(float) a; CP(int) centre; CP(float) alpha; CP(float) beta; CP(int) map;
                         DP(half) yh; u64 rows; uint T, N, Ca, Dt, ldc, ldrg; CP(float) resH; CP(float) resGate; };
struct Be2AdaLnS2QArgs { CP(float) a; CP(float) q; CP(int) atomTok; CP(int) pad; CP(float) alpha; CP(float) beta; CP(int) map;
                         DP(half) yh; u64 rows; uint T, N, Dt, Ca, ldc, pad0; };
// the vendor's fc1 / fc2 [I][C] (torch) as gemmSwiglu's one weight [C][2I]: blocks of 8, a_0..a_7 b_0..b_7 a_8.., b = fc2 / S
struct Be2SwigluWArgs { CP(half) w1; CP(half) w2; DP(half) out; uint C, I; float invS; uint pad0; };
// the coupling's atom-side mixer SPLIT (denoise.mm, couplingSplit): a token row's statistics (mean, centred sum of squares,
// spread) and Xa = (1 + alpha_a) (a - mean) / spread in half; an atom row's (inv, da, dq, sq, mq, sa, its sample-token
// row or -1, its conditioning row) and Xq = (1 + alpha_q) (q - mq) / sq as TWO halves, [hi | lo] (Xq ~ hi + lo); gamma (1 + alpha) and beta on the T + 1
// conditioning rows, for Qa | Qq | B. A simdgroup a row for the first two
struct Be2CsTokArgs { CP(float) a; CP(float) cqc; DP(float) tstats; DP(half) xa; u64 rows; uint T, C, ldc, pad0; };
struct Be2CsAtomArgs { CP(float) q; CP(float) tstats; CP(float) cqc; CP(int) tokIdx; CP(int) pad; CP(int) atomTok; DP(float) stats;
                       DP(half) xq; u64 rows; uint N, T, Ca, Dt, ldc, pad0; };   // (xq [hi | lo])
// (R's row [-gamma_a / 6 | gamma_q] and beta, two halves each, [2][R][Dc]: hi, then the remainder - float's precision
// on the matrix units)
struct Be2CsGammaArgs { CP(float) cqc; DP(half) gR; DP(half) beta; u64 n; uint Dt, Ca, ldc, R; };
// the coupling split's atom-half weight in blocks of 2 (columns 4m, 4m+1: gates 2m, 2m+1; 4m+2, 4m+3 their values / S) -
// a thread's four adjacent results of the vector epilogue are then two gates and their values - its K rows the
// fc1 | fc2 columns off..off+K-1, stacked twice ([hi | lo] inputs)
struct Be2CsWqArgs { CP(half) w1; CP(half) w2; DP(half) out; uint C, I, off, K; float invS; uint pad0; };
// the denoiser's conditioning a group of layers at a time (denoise.mm, condGroup): out[j][r] = silu(LN(c[r]) w[l0 + j] +
// b[l0 + j]) in half - every layer conditions on the same rows, only its LayerNorm's affine is its own; w, b stacked
// [layers][C] (b null: none). A simdgroup a row. Then the bias add over the batch: y[j][r][o] += bias[l0 + j][o]
struct Be2CondInArgs { CP(float) c; CP(float) w; CP(float) b; DP(half) out; u64 rows; uint Rc, C, l0, pad0; };
struct Be2BatchBiasArgs { DP(float) y; CP(float) bias; u64 n; uint Rc, O, l0, pad0; };
struct Be2TokensToAtomsArgs { CP(float) tok; CP(int) atomTok; CP(int) pad; DP(float) out; u64 N; uint T, C, ldo, samples;
                              float beta; uint pad0; };
struct Be2CentreAtomsArgs { CP(float) atom; CP(int) centre; DP(float) out; uint T, N, C, ldo, samples, pad0; };
struct Be2MeanAtomsArgs { CP(float) x; CP(int) start; DP(float) out; uint T, C, ldo, pad0; };
struct Be2SInputsTailArgs { DP(float) s; CP(int) resType; CP(float) profile; CP(float) delMean; CP(int) pocket;
                            uint T, C0, ld, pad0; };
struct Be2ZInitTermsArgs { DP(float) z; CP(float) rows; CP(float) cols; CP(int) meth; CP(float) methPair; CP(float) bonds;
                           CP(float) wBond; CP(int) cls; CP(float) thr; CP(float) enc; CP(float) encB; CP(float) unspec;
                           CP(float) unsel; CP(float) fw; CP(float) fb; uint T, C; };
struct Be2RelPosArgs { CP(float) pos; CP(float) tok; CP(float) ent; CP(float) chain; CP(int) asym; CP(int) entity;
                       CP(int) res; CP(int) sym; DP(float) out; uint T, C, add, ldo; };
// the triangle's (AF3-layout) projection and gate [pairs][2C], a and b interleaved by channel -> channel-major a, b [C][pairs]
struct Be2TriSplitArgs { CP(float) proj; CP(float) gate; DP(float) a; DP(float) b; u64 pairs; uint C, pad0; };
struct Be2ChannelToRowsArgs { CP(float) prod; DP(float) rows; u64 pairs; uint C, pad0; };
struct Be2GateMulAddArgs { DP(float) z; CP(float) out; CP(float) g; u64 n; };       // z += out sigmoid(g)
struct Be2MsaEmbedArgs { DP(float) m; CP(int) msa; CP(int) hasDel; CP(float) delVal; CP(int) paired; CP(float) w;
                         CP(float) sp; u64 S; uint T, C; };
// softmax over j of the [pairs][H] bias, heads first [H][T][T]; a simdgroup a row
struct Be2PwaSoftmaxArgs { CP(float) b; DP(float) w; uint T, H; };
// o[s][i][h][:] = sum_j w[h][i][j] v[s][j][h][:]
struct Be2PwaApplyArgs { CP(float) w; CP(float) v; DP(float) o; u64 S; uint T, H, Ch, pad0; };
// the outer product mean: proj [S][T][2 Co] packed into a, b [S][T Co] (its halves); then a GEMM over the rows gives
// x [(i, c)][(j, d)] for a block of i, permuted into y [(i, j)][(c, d)] / n
struct Be2OpmPackArgs { CP(float) proj; DP(float) a; DP(float) b; u64 R; uint Co, pad0; };
struct Be2OpmPermuteArgs { CP(float) x; DP(float) y; uint bi, T, Co; float inv; };
struct Be2SymmetriseArgs { CP(float) z; DP(float) zs; uint T, C; };
struct Be2ContactArgs { CP(float) logits; DP(float) out; u64 pairs; uint bins, near; };
struct Be2TokenBiasArgs { CP(float) proj; DP(float) out; uint T, H; DP(half) outh; float mul; uint pad0; };
                                                    // [pairs][H] -> [H][pairs] (outh: in half, times mul - log2 e)
// the pair track's fast weights (metal/ef2's): the triangle's projection and gate [C][2C] (a, b interleaved by channel) as
// gemmTriGate's one weight [C][4C], channel c's (pa ga pb gb) in blocks of 8; AF3's blocked transition1 [C][2I] as
// gemmSwiglu's blocks of 8, the value half / S
struct Be2TriGateWArgs { CP(half) proj; CP(half) gate; DP(half) out; uint rows, C, quarters, pad0; };
struct Be2Interleave8Args { CP(half) a; CP(half) b; DP(half) out; uint rows, I, lda; float invS; };
// rows of two torch weights stacked into one [(r1 + r2)][C] half weight
struct Be2StackArgs { CP(half) a; CP(half) b; DP(half) out; u64 na, nb; };

// ---------------------------------------------------------------- the denoiser
struct Be2ScaleMaskedArgs { CP(float) x; CP(int) mask; DP(float) r; u64 N, n; float s; uint pad0; };
struct Be2EdmOutArgs { CP(float) x; CP(float) out; DP(float) d; u64 n; float skip, cout; };
struct Be2CosEmbedArgs { CP(float) w; CP(float) b; DP(float) out; float t; uint n; };
struct Be2BroadcastAddArgs { DP(float) y; CP(float) v; u64 rows; uint C, pad0; };    // y[r] += v
struct Be2ScaledAddArgs { DP(float) y; CP(float) x; u64 n; float s; uint pad0; };     // y += s x
struct Be2RepeatArgs { CP(float) src; DP(float) dst; u64 elems, total; };          // dst = src, src, ...
