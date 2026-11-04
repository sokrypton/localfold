// RFD4-Proteina's kernel arguments (metal/core/args.h conventions: one declaration read by the host and by Metal).
#pragma once
#include "args.h"

// RMSNorm over a row: y = x / sqrt(mean(x^2) + eps) * w, w optional. x and y float32; yh the same values in float16
// where the next GEMM wants a half operand. `ld` 0: the row stride is C.
struct RfdRmsArgs { CP(float) x; DP(float) y; DP(half) yh; CP(float) w; u64 rows; int C, ldx, ldy; float eps; };
// the SwiGLU transition's middle: g[r][i] = silu(h[r][I + i]) * h[r][i], from the one [rows][2I] product
struct RfdSwigluArgs { CP(float) h; DP(float) g; DP(half) gh; u64 rows; uint I, pad; };
// AdaptiveLayerNorm with ONE conditioning row broadcast over every row (the pair track's z_init): out = RMS(x) *
// sigmoid(gamma) + beta, the norm fused so the normalised x is never written. gamma and beta are [C].
struct RfdAdalnArgs { CP(float) x; DP(float) y; DP(half) yh; CP(float) gamma; CP(float) beta; u64 rows; int C; float eps; };
// AdaptiveLayerNorm with a conditioning row PER ROW (every token layer's two adaLNs): out = RMS(x) *
// sigmoid(gamma[r]) + beta[r], the norm fused so the normalised x is never written
struct RfdAdalnRowsArgs { CP(float) x; DP(float) y; CP(float) gamma; CP(float) beta; u64 rows; int C; float eps; };
// x[r][d] += y[r][d] * sigmoid(gamma[r][d]) - AdaptiveOutputScale and its residual, in one pass
struct RfdScaleAddArgs { DP(float) x; CP(float) y; CP(float) gamma; u64 rows; int C; uint pad; };
// pair-biased gated attention in float32, a threadgroup per (query, head): out[i][h D + d] =
// sigmoid(g[i][h D + d]) * sum_j softmax_j(q.k * scale + bias[(i n + j) H + h]) v[j][h D + d].
// q and k are [n][H D]; v is read with its own leading dimension (the packed to_qkv output's row).
struct RfdAttnArgs { CP(float) q; CP(float) k; CP(float) v; CP(float) bias; CP(float) g; DP(float) out;
                     uint n, heads, dimHead, ldv; u64 vOff; float scale; };
// the pair write (`pair_write: hadamard`): z[i n + j][d] = pair[i n + j][d] + pij[j][d] * pij[i][P + d] - the
// first half of linear_x's output indexed by the COLUMN, the second by the row (pair_update.py's pi / pj)
struct RfdPairWriteArgs { CP(float) pair; CP(float) pij; DP(float) z; u64 rows; uint n, P; };
// the triangle's operands, gated and transposed to channel-major planes in one pass:
// aT[c][r] = aP[r][c] * sigmoid(aG[r][c]), and the same for b - what the batched contraction reads
struct RfdTriAbArgs { CP(float) aP; CP(float) aG; CP(float) bP; CP(float) bG; DP(float) aT; DP(float) bT; u64 rows; uint T, pad; };
// the contraction's result back to pair rows: out[r][c] = x[c][r]
struct RfdTriGatherArgs { CP(float) x; DP(float) out; u64 rows; uint T, pad; };
// y[i] = x[i] * sigmoid(g[i])
struct RfdMulSigmoidArgs { CP(float) x; CP(float) g; DP(float) y; u64 n; };
// the same, with LayerNorm in place of RMSNorm (the autoencoder's two transformers are LayerNorm throughout)
struct RfdAdalnRowsLnArgs { CP(float) x; DP(float) y; CP(float) gamma; CP(float) beta; u64 rows; int C; float eps; };
// rows gathered by index (the attended tokens) and scattered back: out[r][c] = in[idx[r]][c], and the reverse
struct RfdRowsArgs { CP(float) in; DP(float) out; CP(int) idx; u64 rows; int C; uint pad; };
// the pair track's distance bins, rebuilt from the current alpha carbons each step: for every pair row the
// one-hot of `torch.bucketize(|ca_i - ca_j|, limits, right=False)` over `dim` columns at `off` of a `stride`-wide
// row (the 29 limits are dist.min + (dist.max - dist.min) k / (dim - 2))
struct RfdPairDistArgs { CP(float) caNm; DP(float) pair; u64 rows; uint n, dim, off, stride; float dmin, dmax; };
// one Euler(-Maruyama) step of one modality, in place on x, with the clean estimate x1 out (flow.js's
// `simulationStep`): mode 0 vf_tsr, 1 vf, 2 sc; `eps` null on a deterministic step
// `vsCoef` is (1 - r) / t, computed on the host from the tsr ratio's closed form: the kernel's
// vs = x vsCoef + r v is algebraically the reference's (x + (1 - t) score r) / t and avoids its cancellation
struct RfdStepArgs { DP(float) x; DP(float) x1; CP(float) v; CP(float) eps; CP(float) emit; u64 n; uint dim, mode;
                     float t, dt, gt, std, r, vsCoef; uint stochastic, pad; };
