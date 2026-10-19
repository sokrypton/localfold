// metal/design's kernels: the backward of what the Evoformer is built from.
//
// WHY THIS TREE EXISTS. Every other backend here folds - forward only. Binder design (BindCraft2's trajectory) is
// gradient descent on a sequence through the folding network, so it needs the backward of every stage, which no
// backend has. This is the first of it: LayerNorm's, and the column sums a linear's parameters need. A linear's own
// backward is two GEMMs and lives on the host (grad.mm) because metal/core's GEMM already transposes either operand.

// ---------------------------------------------------------------- a LayerNorm's row statistics
// The forward's layout - a simdgroup a row, eight rows a threadgroup - so a row's reads land the same way.
kernel void lf_ln_stats(constant LnStatsArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                        uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  ulong row = (ulong)(tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  const int C = a.C;
  device const float* x = a.x ? a.x + row * a.ldx : nullptr;
  device const half* xh = a.xh ? a.xh + row * a.ldx : nullptr;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += x ? x[c] : (float)xh[c];
  float mean = simd_sum(s) / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = (x ? x[c] : (float)xh[c]) - mean; v += d * d; }
  float inv = rsqrt(simd_sum(v) / C + a.eps);
  if (lane == 0) { a.mean[row] = mean; a.inv[row] = inv; }
}

// ---------------------------------------------------------------- LayerNorm backward
kernel void lf_ln_bwd(constant LnBwdArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  ulong row = (ulong)(tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  const int C = a.C;
  device const float* x = a.x ? a.x + row * a.ldx : nullptr;
  device const half* xh = a.xh ? a.xh + row * a.ldx : nullptr;
  device const float* dy = a.dy ? a.dy + row * a.ldy : nullptr;
  device const half* dyh = a.dyh ? a.dyh + row * a.ldy : nullptr;
  const float mean = a.mean[row], inv = a.inv[row];

  // the two row means the backward needs: mean(g) and mean(g * n)
  float sg_ = 0, sgn = 0;
  for (int c = lane; c < C; c += 32) {
    float g = (dy ? dy[c] : (float)dyh[c]) * (a.scale ? a.scale[c] : 1.f);
    float n = ((x ? x[c] : (float)xh[c]) - mean) * inv;
    sg_ += g;
    sgn += g * n;
  }
  const float mg = simd_sum(sg_) / C, mgn = simd_sum(sgn) / C;

  for (int c = lane; c < C; c += 32) {
    float g = (dy ? dy[c] : (float)dyh[c]) * (a.scale ? a.scale[c] : 1.f);
    float n = ((x ? x[c] : (float)xh[c]) - mean) * inv;
    float d = inv * (g - mg - n * mgn);
    if (a.dxh) a.dxh[row * a.lddx + c] = (half)d;
    else a.dx[row * a.lddx + c] = d;
  }
}

// ---------------------------------------------------------------- the parameters' column sums
// One threadgroup a channel; the rows strided across its 256 threads, then a fixed-order reduction.
kernel void lf_col_sums(constant ColSumArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                        uint3 lid3 [[thread_position_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partOff[8], partScale[8];
  const uint lid = lid3.x;
  const int c = (int)tg.x;
  const int C = a.C;
  if (c >= C) return;
  const bool wantScale = a.dscale && (a.x || a.xh) && a.mean && a.inv;

  float so = 0, ss = 0;
  for (ulong row = lid; row < a.rows; row += 256) {
    device const float* dy = a.dy ? a.dy + row * a.ldy : nullptr;
    device const half* dyh = a.dyh ? a.dyh + row * a.ldy : nullptr;
    const float g = dy ? dy[c] : (float)dyh[c];
    so += g;
    if (wantScale) {
      device const float* x = a.x ? a.x + row * a.ldx : nullptr;
      device const half* xh = a.xh ? a.xh + row * a.ldx : nullptr;
      ss += g * (((x ? x[c] : (float)xh[c]) - a.mean[row]) * a.inv[row]);
    }
  }
  so = simd_sum(so);
  ss = simd_sum(ss);
  if (lane == 0) { partOff[sg] = so; partScale[sg] = ss; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float to = 0, ts = 0;
    for (int i = 0; i < 8; ++i) { to += partOff[i]; ts += partScale[i]; }
    if (a.doffset) a.doffset[c] = to;
    if (wantScale) a.dscale[c] = ts;
  }
}
