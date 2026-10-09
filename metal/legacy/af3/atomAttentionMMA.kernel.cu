// cuda/af3/src/atom.cuh's atomAttentionMMA (the f16 path's atom attention: D 32, 128 keys, 32 queries, two warps)
// on Apple's 8x8 simdgroup matrices: each simdgroup's 16 queries as two 8-row tiles, S = Q K^T and O = P V on the
// matrix units, the pair logits, masks and softmax in registers (thread_elements: a lane holds a row's two adjacent
// columns, the row's four lanes differing in lane bits 0 and 3), the gate on the way out.
{
  threadgroup half Ks[KEYS * D];
  threadgroup half Vs[KEYS * D];
  threadgroup half Qs[32 * D];
  const int s = blockIdx.x, h = blockIdx.y, warp = _c.warp, lane = _c.lane;
  const int Wd = heads * D, W2 = 2 * Wd, ss = s % subsets;
  const float qs = scale * 1.4426950408889634f;
  // (four halves a load: D and the row strides are multiples of 4)
  for (int t = threadIdx.x * 4; t < KEYS * D; t += 64 * 4) {
    int key = t / D, e = t % D;
    size_t row = ((size_t)s * KEYS + key) * W2 + h * D + e;
    *(threadgroup half4*)(Ks + t) = *(device const half4*)(kv + row);
    *(threadgroup half4*)(Vs + t) = *(device const half4*)(kv + row + Wd);
  }
  for (int t = threadIdx.x; t < 32 * D; t += 64) {
    int q = t / D, e = t % D;
    Qs[t] = q < queries ? (half)(((float)qg[((size_t)s * queries + q) * W2 + h * D + e] + qBias[h * D + e]) * qs) : (half)0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  _Pragma("clang loop unroll(full)")
  for (int rt = 0; rt < 2; ++rt) {
    const int q = warp * 16 + rt * 8 + sm;
    const int qc = q < queries ? q : queries - 1;
    simdgroup_half8x8 Qm[D / 8];
    _Pragma("clang loop unroll(full)")
    for (int dt = 0; dt < D / 8; ++dt) simdgroup_load(Qm[dt], Qs + (warp * 16 + rt * 8) * D, D, ulong2(dt * 8, 0));
    const float qm = qMask[(size_t)ss * queries + qc];
    device const float* pl = pairLogits + (((size_t)ss * heads + h) * queries + qc) * KEYS;
    simdgroup_float8x8 S[KEYS / 8];
    float mx = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (int kt = 0; kt < KEYS / 8; ++kt) {
      S[kt] = simdgroup_float8x8(0);
      _Pragma("clang loop unroll(full)")
      for (int dt = 0; dt < D / 8; ++dt) {
        simdgroup_half8x8 Kt;
        simdgroup_load(Kt, Ks, D, ulong2(dt * 8, kt * 8), true);
        simdgroup_multiply_accumulate(S[kt], Qm[dt], Kt, S[kt]);
      }
      thread auto& e = S[kt].thread_elements();
      int key = kt * 8 + sn;
      float2 b = *(device const float2*)(pl + key);
      float km0 = kMask[(size_t)ss * KEYS + key], km1 = kMask[(size_t)ss * KEYS + key + 1];
      float mb0 = keyMasked ? -1e9f * ((1.f - qm) + (1.f - km0)) : 1e9f * (qm - 1.f) * (km0 - 1.f);
      float mb1 = keyMasked ? -1e9f * ((1.f - qm) + (1.f - km1)) : 1e9f * (qm - 1.f) * (km1 - 1.f);
      e[0] += (b.x + mb0) * 1.4426950408889634f;
      e[1] += (b.y + mb1) * 1.4426950408889634f;
      mx = fmax(mx, fmax(e[0], e[1]));
    }
    mx = fmax(mx, simd_shuffle_xor(mx, 1)); mx = fmax(mx, simd_shuffle_xor(mx, 8));
    float l = 0;
    simdgroup_half8x8 P[KEYS / 8];
    _Pragma("clang loop unroll(full)")
    for (int kt = 0; kt < KEYS / 8; ++kt) {
      thread auto& e = S[kt].thread_elements();
      float p0 = exp2(e[0] - mx), p1 = exp2(e[1] - mx);
      l += p0 + p1;
      thread auto& pe = P[kt].thread_elements();
      pe[0] = (half)p0; pe[1] = (half)p1;
    }
    l += simd_shuffle_xor(l, 1); l += simd_shuffle_xor(l, 8);
    _Pragma("clang loop unroll(full)")
    for (int dt = 0; dt < D / 8; ++dt) {
      simdgroup_float8x8 O = simdgroup_float8x8(0);
      _Pragma("clang loop unroll(full)")
      for (int kt = 0; kt < KEYS / 8; ++kt) {
        simdgroup_half8x8 Vm;
        simdgroup_load(Vm, Vs, D, ulong2(dt * 8, kt * 8));
        simdgroup_multiply_accumulate(O, P[kt], Vm, O);
      }
      thread auto& o = O.thread_elements();
      if (q < queries) {
        int e = dt * 8 + sn;
        size_t gi = ((size_t)s * queries + q) * W2 + Wd + h * D + e;
        size_t oi = ((size_t)s * queries + q) * Wd + h * D + e;
        out[oi] = (half)(o[0] / l * sigm((float)qg[gi]));
        out[oi + 1] = (half)(o[1] / l * sigm((float)qg[gi + 1]));
      }
    }
  }
}
