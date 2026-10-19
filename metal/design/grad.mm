// metal/design's host half: the dispatches behind grad.h.
#include "grad.h"
#include "kernels.h"
#include <algorithm>

namespace mt {

// a simdgroup a row, eight a threadgroup - metal/core's layerNormRun grid, so the rows map the same way
static Grid rowGroups(size_t rows) {
  size_t groups = (rows + 7) / 8;
  return groups <= 65535 ? Grid{(uint32_t)groups, 1, 1}
                         : Grid{65535, (uint32_t)((groups + 65534) / 65535), 1};
}

void layerNormBwd(const float* x, const float* dy, float* dx, size_t rows, int C, const float* scale,
                  float* dscale, float* doffset, float eps, int ldx, int ldy, int lddx) {
  if (!rows) return;
  ldx = ldx ? ldx : C;
  ldy = ldy ? ldy : C;
  lddx = lddx ? lddx : C;

  // the row statistics, as the forward computed them (two floats a row; see kernels.h for why they are not recomputed
  // inside the column sums)
  float* mean = (float*)scratchBytes("design.ln.mean", rows * sizeof(float));
  float* inv = (float*)scratchBytes("design.ln.inv", rows * sizeof(float));
  LnStatsArgs sa{x, nullptr, mean, inv, rows, C, ldx, eps, 0};
  dispatch("lf_ln_stats", &sa, sizeof sa, rowGroups(rows), 256, 0, "design.ln.stats");

  if (dx) {
    LnBwdArgs a{x, nullptr, dy, nullptr, dx, nullptr, scale, mean, inv, rows, C, ldx, ldy, lddx, 0};
    dispatch("lf_ln_bwd", &a, sizeof a, rowGroups(rows), 256, 0, "design.ln.bwd");
  }
  if (dscale || doffset) {
    // dscale needs n, so it needs x and the statistics; doffset alone does not
    ColSumArgs a{dy, nullptr, dscale ? x : nullptr, nullptr, doffset, dscale,
                 dscale ? mean : nullptr, dscale ? inv : nullptr, rows, C, ldx, ldy, 0, 0};
    dispatch("lf_col_sums", &a, sizeof a, Grid{(uint32_t)C, 1, 1}, 256, 0, "design.ln.params");
  }
}

void columnSum(const float* dy, float* out, size_t rows, int C, int ldy) {
  if (!rows || !C) return;
  ColSumArgs a{dy, nullptr, nullptr, nullptr, out, nullptr, nullptr, nullptr, rows, C, C, ldy ? ldy : C, 0, 0};
  dispatch("lf_col_sums", &a, sizeof a, Grid{(uint32_t)C, 1, 1}, 256, 0, "design.colsum");
}

void reluBwd(const float* dy, const float* ref, float* dx, size_t n) {
  if (!n) return;
  run1d("lf_relu_bwd", n, ReluBwdArgs{dy, ref, dx, n});
}

void gateBwd(const float* dout, const float* y, const float* g, float* dy, float* dg, size_t n) {
  if (!n) return;
  run1d("lf_gate_bwd", n, GateBwdArgs{dout, y, g, dy, dg, n});
}

void linearBwd(const float* X, const float* W, const float* dY, float* dX, float* dW, float* dbias,
               size_t rows, int in, int out, bool accumulateX) {
  if (!rows) return;
  if (dX) {
    // dX[rows][in] = dY[rows][out] * W^T. In Gemm's terms the contraction is over `out`, the result is `in` wide, and
    // the weight wanted is [out][in] - which is W, stored [in][out], read transposed.
    Gemm g{};
    g.X = (void*)dY; g.tx = F32; g.transX = false;
    g.W = (void*)W; g.tw = F32; g.transW = true;
    g.Y = dX; g.ty = F32;
    g.rows = rows; g.in = out; g.out = in;
    g.beta = accumulateX ? 1.f : 0.f;
    g.sx = (int64_t)rows * out; g.sw = (int64_t)in * out; g.sy = (int64_t)rows * in;
    gemm(g);
  }
  if (dW) {
    // dW[in][out] = X^T dY: `in` rows, the contraction over `rows`, and the left operand wanted is [in][rows] - which
    // is X, stored [rows][in], read transposed.
    Gemm g{};
    g.X = (void*)X; g.tx = F32; g.transX = true;
    g.W = (void*)dY; g.tw = F32; g.transW = false;
    g.Y = dW; g.ty = F32;
    g.rows = (size_t)in; g.in = (int)rows; g.out = out;
    g.sx = (int64_t)rows * in; g.sw = (int64_t)rows * out; g.sy = (int64_t)in * out;
    gemm(g);
  }
  if (dbias) columnSum(dY, dbias, rows, out);
}

void transitionBwd(const float* act, const float* dOut, size_t rows, int C, int H,
                   const float* lnScale, const float* lnOffset,
                   const float* W1, const float* b1, const float* W2, const float* b2,
                   float* dAct,
                   float* dLnScale, float* dLnOffset, float* dW1, float* db1, float* dW2, float* db2) {
  if (!rows) return;
  float* ln = (float*)scratchBytes("design.tr.ln", rows * (size_t)C * sizeof(float));
  float* a1 = (float*)scratchBytes("design.tr.a1", rows * (size_t)H * sizeof(float));
  float* dh = (float*)scratchBytes("design.tr.dh", rows * (size_t)H * sizeof(float));
  float* dln = (float*)scratchBytes("design.tr.dln", rows * (size_t)C * sizeof(float));

  // -------- the forward, recomputed: the normalised input and the post-relu activation
  layerNorm(act, ln, rows, C, lnScale, lnOffset);
  {
    Gemm g{};
    g.X = ln; g.W = W1; g.Y = a1;
    g.rows = rows; g.in = C; g.out = H;
    g.bias = b1; g.relu = true;
    gemm(g);
  }

  // -------- the second linear: dW2, db2, and the cotangent of the activation (into dh, then masked in place)
  linearBwd(a1, W2, dOut, dh, dW2, db2, rows, H, C);
  reluBwd(dh, a1, dh, rows * (size_t)H);

  // -------- the first linear: dW1, db1, and the cotangent of the normalised input
  linearBwd(ln, W1, dh, dln, dW1, db1, rows, C, H);

  // -------- and through the LayerNorm, back to the module's input
  layerNormBwd(act, dln, dAct, rows, C, lnScale, dLnScale, dLnOffset);
}

void outerProductMeanBwd(const float* msa, const float* mask, size_t S, int N, int Cm, int O, int F,
                         const float* lnScale, const float* lnOffset,
                         const float* Wl, const float* bl, const float* Wr, const float* br,
                         const float* Wo, const float* dOut,
                         float* dMsa,
                         float* dLnScale, float* dLnOffset, float* dWl, float* dbl, float* dWr, float* dbr,
                         float* dWo, float* dbo, float eps, int tile) {
  if (!S || !N) return;
  const size_t rows = S * (size_t)N;        // the msa as a flat [S * N][Cm]
  const size_t pairs = (size_t)N * N;
  const size_t proj = rows * O;
  const uint OO = (uint)O * O;

  float* act = (float*)scratchBytes("design.opm.act", rows * Cm * sizeof(float));
  float* left = (float*)scratchBytes("design.opm.left", proj * sizeof(float));
  float* right = (float*)scratchBytes("design.opm.right", proj * sizeof(float));
  float* dleft = (float*)scratchBytes("design.opm.dleft", proj * sizeof(float));
  float* dright = (float*)scratchBytes("design.opm.dright", proj * sizeof(float));
  float* dact = (float*)scratchBytes("design.opm.dact", rows * Cm * sizeof(float));
  float* dPre = (float*)scratchBytes("design.opm.dpre", pairs * F * sizeof(float));
  float* P = (float*)scratchBytes("design.opm.P", (size_t)tile * N * OO * sizeof(float));
  float* dP = (float*)scratchBytes("design.opm.dP", (size_t)tile * N * OO * sizeof(float));

  // -------- the forward, recomputed: the normalised msa and the two masked projections
  layerNorm(msa, act, rows, Cm, lnScale, lnOffset);
  {
    Gemm g{}; g.X = act; g.W = Wl; g.Y = left; g.rows = rows; g.in = Cm; g.out = O; g.bias = bl; gemm(g);
  }
  {
    Gemm g{}; g.X = act; g.W = Wr; g.Y = right; g.rows = rows; g.in = Cm; g.out = O; g.bias = br; gemm(g);
  }
  run1d("lf_opm_mask", proj, OpmMaskArgs{left, mask, proj, (uint)N, (uint)O, 0, 0});
  run1d("lf_opm_mask", proj, OpmMaskArgs{right, mask, proj, (uint)N, (uint)O, 0, 0});

  // -------- the division by (eps + norm) comes off first, and dbo is its column sum
  run1d("lf_opm_dpre", pairs, OpmDpreArgs{dOut, mask, dPre, pairs, (uint)S, (uint)N, (uint)F, 0, eps, 0});
  if (dbo) columnSum(dPre, dbo, pairs, F);

  // -------- the tiles. dWo, dleft and dright accumulate, so they start at zero
  if (dWo) fill(dWo, 0, (size_t)OO * F * sizeof(float));
  fill(dleft, 0, proj * sizeof(float));
  fill(dright, 0, proj * sizeof(float));
  for (int i0 = 0; i0 < N; i0 += tile) {
    const uint T = (uint)std::min(tile, N - i0);
    const size_t pn = (size_t)T * N * OO;
    run1d("lf_opm_p", pn, OpmPArgs{left, right, P, pn, (uint)S, (uint)N, (uint)O, (uint)i0, T, 0});
    if (dWo)
      run1d("lf_opm_dwo", (size_t)OO * F,
            OpmDwoArgs{P, dPre, dWo, (size_t)OO * F, (uint)N, (uint)O, (uint)F, (uint)i0, T, 0});
    run1d("lf_opm_dp", pn, OpmDpArgs{dPre, Wo, dP, pn, (uint)N, (uint)O, (uint)F, (uint)i0, T, 0});
    run1d("lf_opm_dleft", S * T * O,
          OpmDleftArgs{dP, right, dleft, S * T * O, (uint)S, (uint)N, (uint)O, (uint)i0, T, 0});
    run1d("lf_opm_dright", proj,
          OpmDrightArgs{dP, left, dright, proj, (uint)S, (uint)N, (uint)O, (uint)i0, T, 0});
  }

  // -------- back through the masks, the two linears (sharing one input), and the LayerNorm
  run1d("lf_opm_mask", proj, OpmMaskArgs{dleft, mask, proj, (uint)N, (uint)O, 0, 0});
  run1d("lf_opm_mask", proj, OpmMaskArgs{dright, mask, proj, (uint)N, (uint)O, 0, 0});
  linearBwd(act, Wl, dleft, dact, dWl, dbl, rows, Cm, O, false);
  linearBwd(act, Wr, dright, dact, dWr, dbr, rows, Cm, O, true);     // accumulate: one input, two paths
  layerNormBwd(msa, dact, dMsa, rows, Cm, lnScale, dLnScale, dLnOffset);
}

}  // namespace mt
