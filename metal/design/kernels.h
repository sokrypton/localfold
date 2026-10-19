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
