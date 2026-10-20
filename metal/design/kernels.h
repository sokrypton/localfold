// metal/design's kernel arguments. Read by C++ and by Metal, like metal/core/args.h (which build.sh prepends).
#pragma once

// A LayerNorm's row statistics, as its forward computed them (metal/core/common.metal, lf_layernorm):
//   mean = sum(x) / C,  inv = rsqrt(mean((x - mean)^2) + eps)      (the variance biased, as the forward)
// The forward keeps none, so the backward computes them once here and both consumers below read them. WHY A BUFFER
// AND NOT A RECOMPUTE PER CONSUMER: the column sums need n[r][c] for ONE channel per thread, and recomputing a row's
// statistics there would read the whole row per channel - O(rows * C^2) for the tensor instead of O(rows * C). Two
// floats a row is the cheap way out, and it is also what a training forward would save.
struct LnStatsArgs {
  CP(float) x; CP(half) xh;
  DP(float) mean; DP(float) inv;
  u64 rows; int C, ldx; float eps; uint pad;
};

// LayerNorm's backward. With n[c] = (x[c] - mean) * inv, g[c] = dy[c] * scale[c], and the means over the C channels:
//   dx[c] = inv * (g[c] - mean(g) - n[c] * mean(g * n))
struct LnBwdArgs {
  CP(float) x; CP(half) xh;        // the forward's input, f32 or f16 (as LayerNormArgs)
  CP(float) dy; CP(half) dyh;      // the incoming cotangent
  DP(float) dx; DP(half) dxh;      // where it goes
  CP(float) scale;                 // may be null (then g == dy)
  CP(float) mean; CP(float) inv;   // from lf_ln_stats
  u64 rows; int C, ldx, ldy, lddx; uint pad;
};

// relu's backward: dx = dy * (ref > 0). `ref` may be either side of the activation - relu(h) > 0 exactly where
// h > 0 - so a caller that kept only the OUTPUT (which a transition does, since the output is what its second
// product consumes) needs no second save set.
struct ReluBwdArgs { CP(float) dy; CP(float) ref; DP(float) dx; u64 n; };

// The gate AF2 closes attention and the triangle with: out = y * sigmoid(g), y and g both produced by a linear. So
//   dy = dout * s,  dg = dout * y * s * (1 - s),   s = sigmoid(g)
// `g` is the gate BEFORE the sigmoid (what the linear wrote); dy and dg may alias their inputs.
struct GateBwdArgs { CP(float) dout; CP(float) y; CP(float) g; DP(float) dy; DP(float) dg; u64 n; };

// y *= alpha. AF2 scales q by key_dim^-0.5 inside the projection, so the cotangent of the UNSCALED product is the
// scaled one times alpha; this is that multiply.
struct ScaleArgs { DP(float) y; u64 n; float alpha; uint pad; };

// Swap the two leading axes of a [A][B][C] tensor into [B][A][C]. AF2's column attentions and the triangle's
// per_column orientation are the same module as their per_row twin with the first two axes swapped
// (modules.py:595, :703), so this is how they are served rather than by a second indexing mode in every kernel.
struct Swap01Args { CP(float) x; DP(float) y; u64 n; uint A, B, C, pad; };

// ---------------------------------------------------------------- the heads, and the metrics a loss reads
// y[i][j][b] = x[i][j][b] + x[j][i][b]. The distogram head symmetrises its logits (modules.py:1121), and this is
// both that and its own backward, the operation being self-adjoint.
struct SymAddArgs { CP(float) x; DP(float) y; u64 n; uint N, C, pad, pad2; };

// pTM and ipTM, from the PAE head's logits (confidence.py:113-180, bindcraft/af2.py:146-164). The only
// differentiable path is through `probs`: the bin centres, d0 and the normalisation are all constant with respect to
// the logits, d0 depending on the residue weights' SUM and not on the logits at all.
//   probs[i][j][b] = softmax_b(logits)
//   pae[i][j]      = (sum_b probs c[b] + its transpose) / 2
//   tm[i][j]       = (sum_b probs tmPerBin[b]) * pairMask[i][j]
//   perAlign[i]    = sum_j tm[i][j] normed[i][j]
//   ptm            = max_i (perAlign[i] w[i])
// `pairMask` is everything for pTM and `asym_id[i] != asym_id[j]` for ipTM, so both are computed with one set of
// kernels and two masks.
struct PaeArgs {
  CP(float) logits; CP(float) breaks; CP(float) weights; CP(float) asym;
  DP(float) centers; DP(float) tmPerBin;       // [bins]; filled by lf_pae_consts
  DP(float) tm; DP(float) normed; DP(float) rowsum; DP(float) perAlign;
  CP(float) dPae; CP(float) dPtm; CP(float) dIptm; CP(float) dIptmPer;
  DP(float) dPerAlign; DP(float) dLogits;
  CP(float) tmP; CP(float) normedP; CP(float) perAlignP;      // the pTM set, when the ipTM set is in `tm` etc.
  CP(float) dPerAlignP;
  u64 n; uint N, bins, interface, pad; float eps;
};

// y += x. A block's modules are residual - new = output_act + module(input_act) - so a module's contribution is
// computed into a temporary and added to the running cotangent. One buffer and one cheap pass a module, against
// threading an `accumulate` flag through six modules' final LayerNorm; the blast radius decided it.
struct AccumArgs { DP(float) y; CP(float) x; u64 n; uint pad; };

// y = x * alpha + beta. AF2 turns a 0/1 mask into an additive attention bias with 1e9 * (mask - 1).
struct AffineArgs { CP(float) x; DP(float) y; u64 n; float alpha, beta; uint pad; };

// y = x * sigmoid(g) - the forward of the triangle's two gated projections, recomputed on the way back
struct MulSigmoidArgs { CP(float) x; CP(float) g; DP(float) y; u64 n; };

// The triangle's product is a batched GEMM over CHANNELS, so its operands are held channel-major: a pair tensor
// [n2][C] becomes [C][n2] (pack) and back (unpack). n2 = N * N. This is the same reason metal/core's forward keeps
// channel-major planes for the triangle (gemmTriGate writes them that way); without it each channel's matrix is
// strided by C and the product cannot be one GEMM.
struct TriPackArgs { CP(float) src; DP(float) dst; u64 n; u64 n2; uint C; uint pad; };

// ---------------------------------------------------------------- attention, forward-saved and backward
// The forward, AF2's (modules.py:336-364), with q ALREADY SCALED by key_dim^-0.5 as it scales it:
//   s[b][h][q][k] = dot(q[b][q][h], k[b][k][h]) + maskBias[b][k] + pairBias[h][q][k]
//   out[b][q][h][c] = sum_k softmax_k(s)[k] * v[b][k][h][c]
// `maskBias` is the per-key mask bias and `pairBias` the non-batched bias AF2 adds from the pair representation -
// shared across b, so its gradient is a sum over b. Either may be null.
//
// WHY THIS IS FLASH-SHAPED AND NOT MATERIALISED. The weights are [B][H][Q][K], and for a triangle attention b runs
// over residues too, so that is O(N^3) - 113 MB at N=192 with four heads, and AGENTS.md forbids it outright. So
// nothing here stores a score: every pass recomputes s from q and k, which is what the saved logsumexp L is for.
//
// THREE PASSES, each deterministic and none needing an atomic:
//   dq        a simdgroup a (b, h, q), looping k
//   dk, dv    a simdgroup a (b, h, k), looping q      (dk and dv accumulate over q, so q must be the inner axis)
//   dPairBias a thread a (h, q, k), looping b         (shared across b, so b must be the inner axis)
// The alternative - one pass writing dS for the others to read - is exactly the O(N^3) buffer that is not allowed.
// Three passes recompute the scores three times; a fused pass with a tree reduction over b is the optimisation this
// leaves, and it is not worth doing before the stack runs end to end.
//
// Layouts: q/k/v/out/dOut are [B][pos][H][D] (AF2's 'bqhc'); L and Drow are [B][H][Q]; pairBias is [H][Q][K];
// maskBias is [B][K]. D <= 64 (a lane holds at most two channels).
struct AttnBwdArgs {
  CP(float) q; CP(float) k; CP(float) v;
  CP(float) maskBias; CP(float) pairBias;
  CP(float) out; CP(float) L; CP(float) Drow; CP(float) dOut;
  DP(float) wout; DP(float) wL;                 // the reference forward's outputs
  DP(float) dq; DP(float) dk; DP(float) dv; DP(float) dPairBias;
  u64 n; uint B, Q, K, H, D, pad;
};

// ---------------------------------------------------------------- GlobalAttention (modules.py:432-497)
// The extra-MSA stack's column attention, and a genuinely different shape from every other attention here:
//   qavg[b][a]     = sum_k mask[b][k] qData[b][k][a] / (sum_k mask[b][k] + 1e-10)      ONE query a batch
//   q[b][h][c]     = qavg[b] Wq * D^-0.5          k[b][k][c] = mData[b][k] Wk          k and v have NO head axis
//   v[b][k][c]     = mData[b][k] Wv
//   logits[b][h][k]= dot(q[b][h], k[b][k]) + 1e9 (mask[b][k] - 1)
//   wa[b][h][c]    = sum_k softmax_k(logits) v[b][k][c]
//   out[b][q][o]   = sum_hc (wa[b][h][c] * sigmoid(qData[b][q] Wg + bg)[q][h][c]) Wo[h][c][o] + bo
// So the single query is BROADCAST back over every position by the gate, which is what makes the gate's backward a
// reduction over q rather than elementwise.
//
// NOTHING here needs flash shaping: the weights are [B][H][K] with no query axis - B*H*K floats, kilobytes - so
// they are stored, and the softmax backward reads them instead of recomputing a score. That is the opposite of
// every other attention in this tree, and the reason is the absent axis, not a different policy.
//
// `gating` is required: AF2 configures it on, and the ungated branch (modules.py:495) has a different output shape.
struct GattnArgs {
  CP(float) qData; CP(float) mData; CP(float) mask;
  CP(float) q; CP(float) k; CP(float) v;
  CP(float) weights; CP(float) wa; CP(float) gpre;
  DP(float) wqavg; DP(float) wminv;            // the masked mean, and 1/(sum mask + eps)
  DP(float) wweights; DP(float) wwa;           // the forward's saves
  DP(float) dlogits; DP(float) dwa; DP(float) dgpre;
  DP(float) dq; DP(float) dk; DP(float) dv; DP(float) dqData;
  CP(float) dwag;                              // the cotangent of the gated, broadcast activation
  u64 n; uint B, K, C, H, D, pad;
};

// ---------------------------------------------------------------- OuterProductMean (Alg. 10)
// The forward, as AF2 writes it (modules.py:1157-1212):
//   act    = LayerNorm(msa)                            [S][N][Cm]
//   left   = mask * (act Wl + bl)                      [S][N][O]      (O = num_outer_channel = 32)
//   right  = mask * (act Wr + br)                      [S][N][O]
//   P      [i][j][c][e] = sum_s left[s][i][c] right[s][j][e]
//   pre    [i][j][f]    = sum_ce P[i][j][c][e] Wo[c][e][f] + bo[f]
//   norm   [i][j]       = sum_s mask[s][i] mask[s][j]
//   out    [i][j][f]    = pre[i][j][f] / (1e-3 + norm[i][j])
//
// WHY THE BACKWARD IS TILED OVER i. Both P and its cotangent are [N][N][O][O] - 1024 floats a pair, 151 MB at
// N=192 and 604 MB at N=384 - so neither is materialised whole. A tile of T rows of i holds [T][N][O][O], and the
// gradients that accumulate across tiles (dWo, dright) do so in separate dispatches, which run in stream order, so
// the sum is in a FIXED order and the test stays deterministic. A float atomic would not be.
//
// The alternative considered and rejected: fold Wo into right first, V[s][j][c][f] = sum_e Wo[c][e][f] right[s][j][e],
// which avoids dP entirely. It is S-DEPENDENT - 25 MB at S=8 but 1.6 GB at the extra-MSA stack's S=512 - and costs
// F/O = 4x the multiplies. The tiled form below is independent of the MSA depth, which is the axis that varies most.

// P[ti][j][c][e] = sum_s left[s][i0 + ti][c] * right[s][j][e]
struct OpmPArgs { CP(float) left; CP(float) right; DP(float) P; u64 n; uint S, N, O, i0, T, pad; };
// dWo[c][e][f] += sum over the tile's (ti, j) of P[ti][j][c][e] * dPre[(i0 + ti) N + j][f]
struct OpmDwoArgs { CP(float) P; CP(float) dPre; DP(float) dWo; u64 n; uint N, O, F, i0, T, pad; };
// dP[ti][j][c][e] = sum_f dPre[(i0 + ti) N + j][f] * Wo[(c O + e) F + f]
struct OpmDpArgs { CP(float) dPre; CP(float) Wo; DP(float) dP; u64 n; uint N, O, F, i0, T, pad; };
// dleft[s][i0 + ti][c] += sum over (j, e) of dP[ti][j][c][e] * right[s][j][e]
struct OpmDleftArgs { CP(float) dP; CP(float) right; DP(float) dleft; u64 n; uint S, N, O, i0, T, pad; };
// dright[s][j][e] += sum over the tile's (ti, c) of dP[ti][j][c][e] * left[s][i0 + ti][c]
struct OpmDrightArgs { CP(float) dP; CP(float) left; DP(float) dright; u64 n; uint S, N, O, i0, T, pad; };
// norm[i][j] = sum_s mask[s][i] mask[s][j], then dPre[i][j][f] = dOut[i][j][f] / (eps + norm[i][j])
struct OpmDpreArgs { CP(float) dOut; CP(float) mask; DP(float) dPre; u64 n; uint S, N, F, pad; float eps; uint pad2; };
// y[s][i][c] *= mask[s][i] - the forward's masking of a projection, and the same factor on the way back
struct OpmMaskArgs { DP(float) y; CP(float) mask; u64 n; uint N, O, pad, pad2; };

// The column sums the parameters' gradients are, over all rows:
//   doffset[c] = sum dy[r][c]            (and a linear's dbias, which is the same reduction)
//   dscale[c]  = sum dy[r][c] * n[r][c]
// `x` null (and mean/inv null) asks for doffset alone, so this one kernel serves a bias too.
//
// One threadgroup a channel, the rows strided across its 256 threads, then a fixed-order threadgroup reduction. The
// order is fixed on purpose: AGENTS.md wants a DETERMINISTIC differential test, and a float atomic - the obvious way
// to let every row-block accumulate straight into the result - sums in whatever order the groups happen to retire.
struct ColSumArgs {
  CP(float) dy; CP(half) dyh;
  CP(float) x; CP(half) xh;        // null: doffset/dbias only, no dscale
  DP(float) doffset; DP(float) dscale;
  CP(float) mean; CP(float) inv;   // from lf_ln_stats; needed only for dscale
  u64 rows; int C, ldx, ldy; uint accum, pad2;   // accum: ADD into doffset/dscale (a weight shared across iterations)
};

// ---------------------------------------------------------------- the structure module's rigids
// A rigid is 12 floats, the rotation ROW-MAJOR then the translation - metal/af2/af2.metal's own layout, so a
// backward here reads exactly what that forward wrote. Both kernels below are a THREAD A RESIDUE rather than a
// thread a point: several (head, point) pairs read one residue's rigid, so a per-point kernel would need a float
// atomic to accumulate dRig, and AGENTS.md wants the differential test to be DETERMINISTIC - an atomic sums in
// whatever order the threads happen to retire.

// The backbone update (af2_rigid_update): rigNew = rigOld o quat(upd), the quaternion (1, u0, u1, u2) normalised and
// the translation (u3, u4, u5). Given dRigNew: dRigOld [L][12] and dUpd [L][6].
struct RigidUpdBwdArgs { CP(float) rigOld; CP(float) upd; CP(float) dRigNew; DP(float) dRigOld; DP(float) dUpd;
                         uint L, accum; };

// Points into the global frame (af2_points_global): g = R loc + t, loc read out of proj [L][H*3*P] as the three
// blocks (x[0:P], y[P:2P], z[2P:3P]) a head. Given dOut [L][H][P][3]: dProj [L][H*3*P] and dRig [L][12].
struct PointsGlobalBwdArgs { CP(float) proj; CP(float) rig; CP(float) dOut; DP(float) dProj; DP(float) dRig;
                             uint L, H, P, accumProj; uint accumRig, pad; };

// ---------------------------------------------------------------- InvariantPointAttention, the core
// The forward is af2_ipa_weights and af2_ipa_outputs. Its logit, for a query q, a head h and a key k:
//
//   s = sum_c qs[q][h][c] ks[k][h][c]  -  0.5 w[h] sum_{p,x} (qp[q][h][p][x] - kp[k][h][p][x])^2
//       + b2d[q][k][h]  -  1e5 (1 - mask[q] mask[k])
//   s *= sqrt(1/3)                                then attn[q][h][:] = softmax(s)
//
// w[h] is softplus(raw[h]) * base, so its own gradient goes back through that. The mask term is constant.
//
// THREE PASSES, as the flash-shaped attention backward above is, and for the same reason: dqs and dqp accumulate
// over KEYS while dks and dkp accumulate over QUERIES, so no single thread assignment serves both. dLogit is
// materialised once (the same size as attn, which the forward already holds) so that both passes read it instead of
// recomputing the softmax, and so that the pass order cannot change the answer.
struct IpaDlogitArgs { CP(float) attn; CP(float) dAttn; DP(float) dLogit; uint L, H, pad, pad2; };
// per (query, head): dqs, dqp, and dpw's partial - summed over queries by lf_ipa_dpw in a fixed order
struct IpaDqArgs { CP(float) dLogit; CP(float) ks; CP(float) qp; CP(float) kp; CP(float) pw;
                   DP(float) dqs; DP(float) dqp; DP(float) dpwPart; uint L, H, Cs, Pq; };
// per (key, head): dks, dkp
struct IpaDkArgs { CP(float) dLogit; CP(float) qs; CP(float) qp; CP(float) kp; CP(float) pw;
                   DP(float) dks; DP(float) dkp; uint L, H, Cs, Pq; };
// db2d[q][k][h] = dLogit[q][h][k] - the head axis moves, which is the whole kernel
struct IpaDb2dArgs { CP(float) dLogit; DP(float) db2d; uint L, H, accum, pad2; };
// the trainable point weight, through its softplus: dRaw[h] = sum_q dpwPart[q][h] * base * sigmoid(raw[h])
struct IpaDpwArgs { CP(float) dpwPart; CP(float) raw; DP(float) dRaw; uint L, H; float base; uint accum; };

// The outputs: per (query, head) a scalar block (attn . vs), a point block (attn . vp taken back into the LOCAL
// frame, with its norm) and a pair block (attn . act2d), laid out as
//   [H*Cs | local x | local y | local z | norms | H*C2],  Fw = H*Cs + 4*H*Pv + H*C2
// dRig is accumulated through a per-head partial for the same determinism reason as above.
struct IpaOutDqArgs { CP(float) attn; CP(float) df; CP(float) vs; CP(float) vp; CP(float) act2d; CP(float) rig;
                      DP(float) dAttn; DP(float) dg; DP(float) dRigPart; uint L, H, Cs, Pv; uint C2, pad; };
struct IpaOutDrigArgs { CP(float) dRigPart; DP(float) dRig; uint L, H, accum, pad; };
struct IpaOutDvArgs { CP(float) attn; CP(float) df; CP(float) dg; DP(float) dvs; DP(float) dvp;
                      uint L, H, Cs, Pv; uint C2, pad; };
struct IpaOutD2dArgs { CP(float) attn; CP(float) df; DP(float) dact2d; uint L, H, Cs, Pv; uint C2, accum; };

// ---------------------------------------------------------------- MultiRigidSidechain's geometry
// The forward is af2_sidechains, a thread a residue: seven torsion angles normalised, eight rigid group frames
// built from the residue type's default frames and a rotation about x, four of them CHAINED (5 onto 4, 6 onto 5,
// 7 onto 6), then every atom14 slot placed by composing the backbone rigid onto its group's frame and applying the
// literature position, and atom37 gathered from atom14.
//
// A thread a residue here too, and for a stronger reason than the kernels above: the frame chain is sequential and
// its backward walks it in reverse, so splitting a residue across threads would mean a reduction per link.
struct SidechainBwdArgs {
  CP(float) unnorm; CP(float) rig; CP(int) aatype; CP(float) defaultFrames; CP(int) atom14Group;
  CP(float) litPos; CP(float) atom14Mask; CP(int) atom37To14; CP(float) atom37Mask; CP(float) seqMask;
  CP(float) dAngles; CP(float) dPos14; CP(float) dPos37;    // any of the three may be null
  DP(float) dUnnorm; DP(float) dRig;
  uint L, accumRig; float positionScale; uint pad;
};
