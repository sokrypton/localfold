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

// ---------------------------------------------------------------- the elementwise backwards
kernel void lf_relu_bwd(LF_ARGS(ReluBwdArgs)) {
  ulong i = LF_INDEX;
  if (i >= a.n) return;
  a.dx[i] = a.ref[i] > 0.f ? a.dy[i] : 0.f;
}

kernel void lf_gate_bwd(LF_ARGS(GateBwdArgs)) {
  ulong i = LF_INDEX;
  if (i >= a.n) return;
  const float s = 1.f / (1.f + exp(-a.g[i]));
  const float d = a.dout[i];
  const float y = a.y[i];
  // read both before either store: dy and dg are allowed to alias y and g
  if (a.dy) a.dy[i] = d * s;
  if (a.dg) a.dg[i] = d * y * s * (1.f - s);
}

// ---------------------------------------------------------------- OuterProductMean's backward
// Every kernel here is one thread an output element, looping the contracted axes in a fixed order. The strided reads
// (dP's j stride is O*O) are the optimisation surface and are left alone deliberately: correctness first.

kernel void lf_opm_p(LF_ARGS(OpmPArgs)) {
  ulong idx = LF_INDEX;
  if (idx >= a.n) return;
  const uint OO = a.O * a.O;
  const uint ce = (uint)(idx % (ulong)OO);
  const uint rest = (uint)(idx / (ulong)OO);
  const uint j = rest - lf_udiv(rest, a.N) * a.N;
  const uint ti = lf_udiv(rest, a.N);
  const uint c = lf_udiv(ce, a.O), e = ce - lf_udiv(ce, a.O) * a.O;
  const uint i = a.i0 + ti;
  float acc = 0;
  for (uint s = 0; s < a.S; ++s)
    acc += a.left[((ulong)s * a.N + i) * a.O + c] * a.right[((ulong)s * a.N + j) * a.O + e];
  a.P[idx] = acc;
}

kernel void lf_opm_dwo(LF_ARGS(OpmDwoArgs)) {
  ulong idx = LF_INDEX;
  if (idx >= a.n) return;                      // n = O * O * F
  const uint f = (uint)(idx - lf_udiv((uint)idx, a.F) * a.F);
  const uint ce = lf_udiv((uint)idx, a.F);
  const uint OO = a.O * a.O;
  float acc = 0;
  for (uint ti = 0; ti < a.T; ++ti) {
    const ulong prow = (ulong)(a.i0 + ti) * a.N;
    for (uint j = 0; j < a.N; ++j)
      acc += a.P[((ulong)ti * a.N + j) * OO + ce] * a.dPre[(prow + j) * a.F + f];
  }
  a.dWo[(ulong)ce * a.F + f] += acc;
}

kernel void lf_opm_dp(LF_ARGS(OpmDpArgs)) {
  ulong idx = LF_INDEX;
  if (idx >= a.n) return;
  const uint OO = a.O * a.O;
  const uint ce = (uint)(idx % (ulong)OO);
  const uint rest = (uint)(idx / (ulong)OO);
  const uint j = rest - lf_udiv(rest, a.N) * a.N;
  const uint ti = lf_udiv(rest, a.N);
  const ulong prow = ((ulong)(a.i0 + ti) * a.N + j) * a.F;
  const ulong wrow = (ulong)ce * a.F;
  float acc = 0;
  for (uint f = 0; f < a.F; ++f) acc += a.dPre[prow + f] * a.Wo[wrow + f];
  a.dP[idx] = acc;
}

kernel void lf_opm_dleft(LF_ARGS(OpmDleftArgs)) {
  ulong idx = LF_INDEX;
  if (idx >= a.n) return;                      // n = S * T * O
  const uint c = (uint)(idx - lf_udiv((uint)idx, a.O) * a.O);
  const uint rest = lf_udiv((uint)idx, a.O);
  const uint ti = rest - lf_udiv(rest, a.T) * a.T;
  const uint s = lf_udiv(rest, a.T);
  const uint OO = a.O * a.O;
  float acc = 0;
  for (uint j = 0; j < a.N; ++j) {
    const ulong dprow = ((ulong)ti * a.N + j) * OO + (ulong)c * a.O;
    const ulong rrow = ((ulong)s * a.N + j) * a.O;
    for (uint e = 0; e < a.O; ++e) acc += a.dP[dprow + e] * a.right[rrow + e];
  }
  a.dleft[((ulong)s * a.N + a.i0 + ti) * a.O + c] += acc;
}

kernel void lf_opm_dright(LF_ARGS(OpmDrightArgs)) {
  ulong idx = LF_INDEX;
  if (idx >= a.n) return;                      // n = S * N * O
  const uint e = (uint)(idx - lf_udiv((uint)idx, a.O) * a.O);
  const uint rest = lf_udiv((uint)idx, a.O);
  const uint j = rest - lf_udiv(rest, a.N) * a.N;
  const uint s = lf_udiv(rest, a.N);
  const uint OO = a.O * a.O;
  float acc = 0;
  for (uint ti = 0; ti < a.T; ++ti) {
    const ulong dprow = ((ulong)ti * a.N + j) * OO;
    const ulong lrow = ((ulong)s * a.N + a.i0 + ti) * a.O;
    for (uint c = 0; c < a.O; ++c) acc += a.dP[dprow + (ulong)c * a.O + e] * a.left[lrow + c];
  }
  a.dright[((ulong)s * a.N + j) * a.O + e] += acc;
}

kernel void lf_opm_dpre(LF_ARGS(OpmDpreArgs)) {
  ulong idx = LF_INDEX;
  if (idx >= a.n) return;                      // n = N * N, a thread a pair
  const uint j = (uint)(idx - lf_udiv((uint)idx, a.N) * a.N);
  const uint i = lf_udiv((uint)idx, a.N);
  float norm = 0;
  for (uint s = 0; s < a.S; ++s) norm += a.mask[(ulong)s * a.N + i] * a.mask[(ulong)s * a.N + j];
  const float inv = 1.f / (a.eps + norm);
  const ulong row = idx * a.F;
  for (uint f = 0; f < a.F; ++f) a.dPre[row + f] = a.dOut[row + f] * inv;
}

kernel void lf_opm_mask(LF_ARGS(OpmMaskArgs)) {
  ulong idx = LF_INDEX;
  if (idx >= a.n) return;                      // n = S * N * O
  const uint rest = (uint)lf_udiv((uint)idx, a.O);          // = s * N + i
  a.y[idx] *= a.mask[rest];
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
