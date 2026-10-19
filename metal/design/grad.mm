// metal/design's host half: the dispatches behind grad.h.
#include "grad.h"
#include "kernels.h"

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

void linearBwd(const float* X, const float* W, const float* dY, float* dX, float* dW, float* dbias,
               size_t rows, int in, int out) {
  if (!rows) return;
  if (dX) {
    // dX[rows][in] = dY[rows][out] * W^T. In Gemm's terms the contraction is over `out`, the result is `in` wide, and
    // the weight wanted is [out][in] - which is W, stored [in][out], read transposed.
    Gemm g{};
    g.X = (void*)dY; g.tx = F32; g.transX = false;
    g.W = (void*)W; g.tw = F32; g.transW = true;
    g.Y = dX; g.ty = F32;
    g.rows = rows; g.in = out; g.out = in;
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

}  // namespace mt
