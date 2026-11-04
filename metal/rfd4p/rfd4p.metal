// RFD4-Proteina's own kernels. Everything the trunk needs beyond metal/core's GEMM, attention and LayerNorm: the
// RMSNorm this network uses throughout where AlphaFold's stacks use LayerNorm, the SwiGLU transition's middle, and
// the adaptive LayerNorm that conditions the pair track.
//
// 🔴 THE EPSILON IS INSIDE THE MEAN, NOT ON THE NORM: 1 / sqrt(mean(x^2) + eps), which is torch's RMSNorm and
// cpu/rfd4p/reference.js's `rmsNorm`. metal/ef2's own RMS kernels use a different epsilon (1.19e-7, its network's)
// and are not interchangeable with these.

// a simdgroup a row, 8 rows a threadgroup
kernel void rfd4p_rms_norm(constant RfdRmsArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                           uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const ulong row = (ulong)(tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  const int ldx = a.ldx ? a.ldx : a.C, ldy = a.ldy ? a.ldy : a.C;
  device const float* x = a.x + row * ldx;
  float s = 0;
  for (int c = lane; c < a.C; c += 32) { const float v = x[c]; s += v * v; }
  const float inv = rsqrt(simd_sum(s) / (float)a.C + a.eps);
  for (int c = lane; c < a.C; c += 32) {
    const float v = x[c] * inv * (a.w ? a.w[c] : 1.f);
    if (a.y) a.y[row * ldy + c] = v;
    if (a.yh) a.yh[row * ldy + c] = (half)v;
  }
}

kernel void rfd4p_swiglu(LF_ARGS(RfdSwigluArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.rows * a.I) return;
  const uint r = lf_udiv((uint)(t / 1), a.I), i = (uint)(t - (ulong)r * a.I);
  device const float* h = a.h + (ulong)r * 2 * a.I;
  const float v = lf_silu(h[a.I + i]) * h[i];
  if (a.g) a.g[t] = v;
  if (a.gh) a.gh[t] = (half)v;
}

// a simdgroup a row: RMS(x) * sigmoid(gamma) + beta, one conditioning row over every row
kernel void rfd4p_adaln_bcast(constant RfdAdalnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                              uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const ulong row = (ulong)(tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  device const float* x = a.x + row * a.C;
  float s = 0;
  for (int c = lane; c < a.C; c += 32) { const float v = x[c]; s += v * v; }
  const float inv = rsqrt(simd_sum(s) / (float)a.C + a.eps);
  for (int c = lane; c < a.C; c += 32) {
    const float v = x[c] * inv * lf_sigmoid(a.gamma[c]) + a.beta[c];
    if (a.y) a.y[row * a.C + c] = v;
    if (a.yh) a.yh[row * a.C + c] = (half)v;
  }
}

// a simdgroup a row: RMS(x) * sigmoid(gamma[row]) + beta[row]
kernel void rfd4p_adaln_rows(constant RfdAdalnRowsArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                             uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const ulong row = (ulong)(tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  device const float* x = a.x + row * a.C;
  device const float* gm = a.gamma + row * a.C;
  device const float* bt = a.beta + row * a.C;
  float s = 0;
  for (int c = lane; c < a.C; c += 32) { const float v = x[c]; s += v * v; }
  const float inv = rsqrt(simd_sum(s) / (float)a.C + a.eps);
  for (int c = lane; c < a.C; c += 32) a.y[row * a.C + c] = x[c] * inv * lf_sigmoid(gm[c]) + bt[c];
}

kernel void rfd4p_scale_add(LF_ARGS(RfdScaleAddArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.rows * (ulong)a.C) return;
  a.x[t] += a.y[t] * lf_sigmoid(a.gamma[t]);
}

// 🔴 FLOAT32 AND ONE THREADGROUP A (QUERY, HEAD), WHICH IS A CORRECTNESS KERNEL AND NOT THE FAST ONE.
// metal/core's `lf_attention` is the fast path - flash, half, its bias in log2 units - and this port will move
// onto it with the move MEASURED. Until then the trunk is held to a float64 reference at 1e-6, and a half
// attention cannot be: it would put the whole stack's bound at 1e-3 before any of it had been checked at all.
// Scores live in threadgroup memory, so n is bounded by it (the host refuses past LF_ATTN_MAX_N).
kernel void rfd4p_attention(constant RfdAttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]],
                            threadgroup float* sc [[threadgroup(0)]]) {
  const uint i = tg.x, h = tg.y, n = a.n, D = a.dimHead, W = a.heads * D;
  device const float* qi = a.q + (ulong)i * W + h * D;
  // the scores, and their maximum
  float localMax = -INFINITY;
  for (uint j = tid; j < n; j += tpg.x) {
    device const float* kj = a.k + (ulong)j * W + h * D;
    float dot = 0, cc = 0;
    for (uint d = 0; d < D; ++d) { const float p = qi[d] * kj[d] - cc, t = dot + p; cc = (t - dot) - p; dot = t; }
    float s = dot * a.scale;
    if (a.bias) s += a.bias[((ulong)i * n + j) * a.heads + h];
    sc[j] = s;
    localMax = max(localMax, s);
  }
  threadgroup float red[32];
  const uint sgs = tpg.x / 32;
  if (tid % 32 == 0) red[tid / 32] = -INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  localMax = simd_max(localMax);
  if (tid % 32 == 0) red[tid / 32] = localMax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -INFINITY;
  for (uint s = 0; s < sgs; ++s) m = max(m, red[s]);
  // exp and its sum
  float localSum = 0, cs = 0;
  for (uint j = tid; j < n; j += tpg.x) {
    const float e = exp(sc[j] - m);
    sc[j] = e;
    const float p = e - cs, t = localSum + p;
    cs = (t - localSum) - p;
    localSum = t;
  }
  localSum = simd_sum(localSum);
  if (tid % 32 == 0) red[tid / 32] = localSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0;
  for (uint s = 0; s < sgs; ++s) sum += red[s];
  const float invSum = sum > 0 ? 1.f / sum : 0.f;
  // the context, a thread a channel of this head
  device const float* v0 = a.v + a.vOff + h * D;
  for (uint d = tid; d < D; d += tpg.x) {
    float acc = 0, ca = 0;
    for (uint j = 0; j < n; ++j) {
      const float p = sc[j] * invSum * v0[(ulong)j * a.ldv + d] - ca, t = acc + p;
      ca = (t - acc) - p;
      acc = t;
    }
    const ulong at = (ulong)i * W + h * D + d;
    a.out[at] = lf_sigmoid(a.g[at]) * acc;
  }
}

kernel void rfd4p_pair_write(LF_ARGS(RfdPairWriteArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.rows * (ulong)a.P) return;
  const uint P = a.P;
  const ulong row = t / P, chan = t - row * P;
  const uint i = lf_udiv((uint)row, a.n), j = (uint)row - i * a.n;
  a.z[t] = a.pair[t] + a.pij[(ulong)j * 2 * P + chan] * a.pij[(ulong)i * 2 * P + P + chan];
}

// a thread a (pair row, channel): the gate applied and the value transposed into its channel's plane
kernel void rfd4p_tri_ab(LF_ARGS(RfdTriAbArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.rows * (ulong)a.T) return;
  const ulong row = t / a.T, c = t - row * a.T;
  const ulong at = c * a.rows + row;
  a.aT[at] = a.aP[t] * lf_sigmoid(a.aG[t]);
  a.bT[at] = a.bP[t] * lf_sigmoid(a.bG[t]);
}

kernel void rfd4p_tri_gather(LF_ARGS(RfdTriGatherArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.rows * (ulong)a.T) return;
  const ulong row = t / a.T, c = t - row * a.T;
  a.out[t] = a.x[c * a.rows + row];
}

kernel void rfd4p_mul_sigmoid(LF_ARGS(RfdMulSigmoidArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.n) return;
  a.y[t] = a.x[t] * lf_sigmoid(a.g[t]);
}

// a simdgroup a row: LayerNorm(x) * sigmoid(gamma[row]) + beta[row] - the x-norm inside `adaptiveLayerNorm` has
// neither scale nor offset of its own, so this is (x - mean) / sqrt(var + eps) and then the conditioning
kernel void rfd4p_adaln_rows_ln(constant RfdAdalnRowsLnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const ulong row = (ulong)(tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  device const float* x = a.x + row * a.C;
  device const float* gm = a.gamma + row * a.C;
  device const float* bt = a.beta + row * a.C;
  float sum = 0;
  for (int c = lane; c < a.C; c += 32) sum += x[c];
  const float mean = simd_sum(sum) / (float)a.C;
  float var = 0;
  for (int c = lane; c < a.C; c += 32) { const float d = x[c] - mean; var += d * d; }
  const float inv = rsqrt(simd_sum(var) / (float)a.C + a.eps);
  for (int c = lane; c < a.C; c += 32) a.y[row * a.C + c] = (x[c] - mean) * inv * lf_sigmoid(gm[c]) + bt[c];
}

kernel void rfd4p_gather_rows(LF_ARGS(RfdRowsArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.rows * (ulong)a.C) return;
  const ulong r = t / a.C, c = t - r * a.C;
  a.out[t] = a.in[(ulong)a.idx[r] * a.C + c];
}

kernel void rfd4p_scatter_rows(LF_ARGS(RfdRowsArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.rows * (ulong)a.C) return;
  const ulong r = t / a.C, c = t - r * a.C;
  a.out[(ulong)a.idx[r] * a.C + c] = a.in[t];
}

// a thread a pair row: the previous step's one-hot cleared and this step's written
kernel void rfd4p_pair_dists(LF_ARGS(RfdPairDistArgs)) {
  const ulong r = LF_INDEX;
  if (r >= a.rows) return;
  const uint i = lf_udiv((uint)r, a.n), j = (uint)r - i * a.n;
  const float dx = a.caNm[i * 3] - a.caNm[j * 3];
  const float dy = a.caNm[i * 3 + 1] - a.caNm[j * 3 + 1];
  const float dz = a.caNm[i * 3 + 2] - a.caNm[j * 3 + 2];
  const float d = sqrt(dx * dx + dy * dy + dz * dz);
  // bucketize(right=false): the first limit NOT below d
  const uint limits = a.dim - 1;
  const float step = (a.dmax - a.dmin) / (float)(a.dim - 2);
  uint bin = 0;
  while (bin < limits && a.dmin + step * (float)bin < d) bin += 1;
  device float* row = a.pair + r * a.stride + a.off;
  for (uint k = 0; k < a.dim; ++k) row[k] = k == bin ? 1.f : 0.f;
}

kernel void rfd4p_sim_step(LF_ARGS(RfdStepArgs)) {
  const ulong t = LF_INDEX;
  if (t >= a.n) return;
  const uint tok = lf_udiv((uint)(t / a.dim), 1);
  const ulong token = t / a.dim;
  const float live = a.emit[token] != 0.f ? 1.f : 0.f;
  const float x = a.x[t], v = a.v[t];
  a.x1[t] = live != 0.f ? x + (1.f - a.t) * v : 0.f;
  // `vf_tsr` at t = 0 leaves x where it is (the rescaled velocity is zero there)
  if (a.mode == 0u && a.t == 0.f) { if (live == 0.f) a.x[t] = 0.f; return; }
  float delta;
  if (a.mode == 1u) {
    delta = v * a.dt;
  } else {
    const float score = (a.t * v - x) / (1.f - a.t);
    if (a.mode == 2u && a.stochastic) {
      delta = (v + a.gt * score) * a.dt + a.std * a.eps[t];
    } else {
      const float ss = score * a.r;
      // 🔴 NOT `(x + (1 - t) ss) / t`, WHICH IS THE SAME NUMBER AND LOSES IT. At the first step t is 1e-3 and
      // r is 1 to six figures, so `x + (1 - t) score r` subtracts two nearly equal quantities and then
      // multiplies the remainder by 1000: in float32 the answer is rounding. The identity
      //   (x + (1 - t) r (t v - x) / (1 - t)) / t = x (1 - r) / t + r v
      // has no cancellation in it, and (1 - r) / t comes from the tsr ratio's closed form on the host, where
      // 1 - r = A (1 - k) / k / (A / k + 1) with A = snr sigma^2 - exact for small A rather than 1 minus 1.
      // Measured: the backbone's first step went 4.1e-4 against the reference to 5.7e-7.
      const float vs = x * a.vsCoef + a.r * v;
      delta = a.stochastic ? (vs + a.gt * ss) * a.dt + a.std * a.eps[t] : vs * a.dt;
    }
  }
  a.x[t] = live != 0.f ? x + delta : 0.f;
  (void)tok;
}
