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
  u64 rows; int C, ldx, ldy; uint pad, pad2;
};
