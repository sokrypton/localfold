// ESMFold2's kernels (metal/ef2): the language model's, the trunk's, the atom transformer's, the diffusion module's and
// the confidence head's. cuda/ef2/src is the reading: the same arithmetic, written for Apple's GPUs - 32-bit index
// math (lf_udiv: no integer divider), simdgroup reductions, threadgroup tiles within 32 KB.

// ---------------------------------------------------------------- relative positions (139 bins, summed rows)
inline float4 relPosAt4(constant RelIdx& r, uint i, uint j, uint c, uint C) {
  const int rb = 32, cb = 2;
  bool sameChain = r.asym[i] == r.asym[j], sameRes = r.ri[i] == r.ri[j];
  int b0 = sameChain ? clamp(r.ri[i] - r.ri[j] + rb, 0, 2 * rb) : 2 * rb + 1;
  int b1 = sameChain && sameRes ? clamp(r.ti[i] - r.ti[j] + rb, 0, 2 * rb) : 2 * rb + 1;
  int b3 = sameChain ? 2 * cb + 1 : clamp(r.sym[i] - r.sym[j] + cb, 0, 2 * cb);
  const int w = 2 * rb + 2;
  float4 v = *(device const float4*)(r.Wt + (uint)b0 * C + c) + *(device const float4*)(r.Wt + (uint)(w + b1) * C + c) +
             *(device const float4*)(r.Wt + (uint)(2 * w + 1 + b3) * C + c);
  if (r.ent[i] == r.ent[j]) v += *(device const float4*)(r.Wt + (uint)(2 * w) * C + c);
  return v;
}

kernel void ef2_interleave8(LF_ARGS(Interleave8Args)) {
  uint t = (uint)LF_INDEX, B2 = a.I >> 2;
  if (t >= a.rows * B2) return;
  uint r = lf_udiv(t, B2), c = t - r * B2, m = c >> 1;
  device const uint4* src = (c & 1) ? (device const uint4*)(a.b + (ulong)r * a.lda) : (device const uint4*)(a.a + (ulong)r * a.lda);
  ((device uint4*)a.out)[t] = src[m];
}
kernel void ef2_stack_cols(LF_ARGS(StackColsArgs)) {
  uint t = (uint)LF_INDEX, ca4 = a.ca >> 2, cb4 = a.cb >> 2, cc4 = a.cc >> 2, W4 = ca4 + cb4 + cc4;
  if (t >= a.rows * W4) return;
  uint r = lf_udiv(t, W4), c4 = t - r * W4;
  ((device half4*)a.out)[t] = c4 < ca4 ? ((device const half4*)(a.a + (ulong)r * a.ca))[c4]
                            : c4 < ca4 + cb4 ? ((device const half4*)(a.b + (ulong)r * a.cb))[c4 - ca4]
                                             : ((device const half4*)(a.c + (ulong)r * a.cc))[c4 - ca4 - cb4];
}
kernel void ef2_interleave8_inplace(constant Interleave8Args& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  const uint r = tg.x, B8 = a.I >> 3;
  if (r >= a.rows) return;
  device uint4* row = (device uint4*)(a.out + (ulong)r * 2 * a.I);
  if (B8 <= 256) {
    uint4 va = 0, vb = 0;
    if (tid < B8) {
      va = row[tid];
      vb = row[B8 + tid];
    }
    threadgroup_barrier(mem_flags::mem_none);
    if (tid < B8) {
      row[2 * tid] = va;
      row[2 * tid + 1] = vb;
    }
    return;
  }
  uint4 va0 = row[tid], vb0 = row[B8 + tid], va1 = 0, vb1 = 0;
  const uint m1 = 256 + tid;
  if (m1 < B8) {
    va1 = row[m1];
    vb1 = row[B8 + m1];
  }
  threadgroup_barrier(mem_flags::mem_none);
  row[2 * tid] = va0;
  row[2 * tid + 1] = vb0;
  if (m1 < B8) {
    row[2 * m1] = va1;
    row[2 * m1 + 1] = vb1;
  }
}
kernel void ef2_interleave8_inplace_wide(constant Interleave8Args& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]]) {
  const uint r = tg.x, B8 = a.I >> 3;
  if (r >= a.rows) return;
  device uint4* row = (device uint4*)(a.out + (ulong)r * 2 * a.I);
  uint4 va[4], vb[4];
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; ++k) {
    const uint m = k * 256 + tid;
    if (m < B8) {
      va[k] = row[m];
      vb[k] = row[B8 + m];
    }
  }
  threadgroup_barrier(mem_flags::mem_none);
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; ++k) {
    const uint m = k * 256 + tid;
    if (m < B8) {
      row[2 * m] = va[k];
      row[2 * m + 1] = vb[k];
    }
  }
}

// ---------------------------------------------------------------- the language model
kernel void ef2_embed(LF_ARGS(EmbedArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.rows * a.C) return;
  uint row = lf_udiv((uint)t, a.C), c = (uint)t - row * a.C;
  a.x[t] = a.table[(ulong)a.ids[row] * a.C + c];
}
// rotary, split halves: channel d with d + 32 of each 64-wide head, position = row; sin/cos computed once per (row, d)
kernel void ef2_rope(LF_ARGS(RopeArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.rows * 32) return;
  uint d = t & 31, row = t >> 5;
  float angle = (float)row * precise::pow(10000.f, -(float)(2 * d) / 64.f);
  float s = precise::sin(angle), c = precise::cos(angle);
  device float* px = a.x + (ulong)row * a.ld + d;
  device float* py = a.y ? a.y + (ulong)row * a.ld + d : nullptr;
  for (uint h = 0; h < a.heads; ++h) {
    float x0 = px[h * 64], x1 = px[h * 64 + 32];
    px[h * 64] = x0 * c - x1 * s;
    px[h * 64 + 32] = x0 * s + x1 * c;
    if (py) {
      float y0 = py[h * 64], y1 = py[h * 64 + 32];
      py[h * 64] = y0 * c - y1 * s;
      py[h * 64 + 32] = y0 * s + y1 * c;
    }
  }
}
// q_norm, k_norm and rotary in one pass over qkv [rows, 3C] (C a multiple of 64 up to 1152): tg.y selects q (0) or k (1),
// a simdgroup a row; within each 128-channel step (two 64-wide heads), lane and lane ^ 8 hold the (d, d + 32) rotary pair
kernel void ef2_qk_norm_rope(constant QkNormRopeArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  uint row = tg.x * sgs + sg;
  if (row >= a.rows) return;
  const uint C = a.C, C4 = C >> 2, isK = tg.y;
  device const float4* x4 = (device const float4*)(a.qkv + (ulong)row * 3 * C + isK * C);
  device const float4* sc4 = (device const float4*)(isK ? a.kScale : a.qScale);
  device float4* y4 = (device float4*)((isK ? a.k : a.q) + (ulong)row * C);
  float4 r[9];
  float s = 0;
  const bool exact1152 = C == 1152;
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < 9; ++k) {
    uint c = lane + 32 * k;
    if (exact1152 || c < C4) {
      r[k] = x4[c];
      s += r[k].x + r[k].y + r[k].z + r[k].w;
    }
  }
  float mean = simd_sum(s) / (float)C, v = 0;
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < 9; ++k) {
    if (exact1152 || lane + 32 * k < C4) { float4 d = r[k] - mean; v += dot(d, d); }
  }
  float inv = rsqrt(simd_sum(v) / (float)C + 1e-5f);
  uint d0 = (lane & 7) << 2;
  float4 cs, sn;
  _Pragma("clang loop unroll(full)")
  for (int e = 0; e < 4; ++e) {
    float angle = (float)row * precise::pow(10000.f, -(float)(2 * (d0 + e)) / 64.f);
    sn[e] = precise::sin(angle);
    cs[e] = precise::cos(angle);
  }
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < 9; ++k) {
    uint c = lane + 32 * k;
    if (exact1152 || c < C4) {
      float4 n = (r[k] - mean) * inv * sc4[c];
      float4 p = simd_shuffle_xor(n, 8);
      y4[c] = (lane & 8) ? (p * sn + n * cs) : (n * cs - p * sn);
    }
  }
}
kernel void ef2_shim_norm_acc(constant ShimNormAccArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                              uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint row = (tg.y * ng.x + tg.x) * sgs + sg;
  if (row >= a.rows) return;
  const uint C = a.C, C4 = C >> 2;
  const bool exact1152 = C == 1152;
  device const float4* x4 = (device const float4*)(a.x + (ulong)row * C);
  device const float4* ss4 = (device const float4*)a.shimScale;
  device const float4* so4 = (device const float4*)a.shimOffset;
  device float4* m4 = (device float4*)(a.mixXn + (ulong)row * C);
  float4 r[9];
  float s = 0;
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < 9; ++k) {
    uint c = lane + 32 * k;
    if (exact1152 || c < C4) {
      r[k] = x4[c];
      s += r[k].x + r[k].y + r[k].z + r[k].w;
    }
  }
  float mean = simd_sum(s) / (float)C, v = 0;
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < 9; ++k) {
    if (exact1152 || lane + 32 * k < C4) { float4 d = r[k] - mean; v += dot(d, d); }
  }
  float inv = rsqrt(simd_sum(v) / (float)C + 1e-5f);
  if (a.finalScale) {
    device const float4* fs4 = (device const float4*)a.finalScale;
    float s2 = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 9; ++k) {
      uint c = lane + 32 * k;
      if (exact1152 || c < C4) {
        r[k] = (r[k] - mean) * inv * fs4[c];
        s2 += r[k].x + r[k].y + r[k].z + r[k].w;
      }
    }
    mean = simd_sum(s2) / (float)C; v = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 9; ++k) {
      if (exact1152 || lane + 32 * k < C4) { float4 d = r[k] - mean; v += dot(d, d); }
    }
    inv = rsqrt(simd_sum(v) / (float)C + 1e-5f);
  }
  device const float4* as4 = a.attnScale ? (device const float4*)a.attnScale : nullptr;
  device const float4* ao4 = a.attnOffset ? (device const float4*)a.attnOffset : nullptr;
  device half4* ax4 = a.attnXn ? (device half4*)(a.attnXn + (ulong)row * C) : nullptr;
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < 9; ++k) {
    uint c = lane + 32 * k;
    if (exact1152 || c < C4) {
      float4 n0 = (r[k] - mean) * inv;
      float4 sn = float4(half4(n0 * ss4[c] + so4[c]));
      float4 acc = (a.first ? float4(0.f) : m4[c]) + a.weight * sn;
      if (a.finalScale && ax4) ax4[c] = half4(acc);
      else m4[c] = acc;
      if (!a.finalScale && ax4) ax4[c] = half4(n0 * as4[c] + ao4[c]);
    }
  }
}
// softmax over a score row, keys of another chain (sequence id) excluded; a simdgroup a (head, query) row
kernel void ef2_softmax_seq(constant SoftmaxSeqArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint3 ng [[threadgroups_per_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint row = (tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.total) return;
  uint q = row - lf_udiv(row, a.rows) * a.rows;
  device float* s = a.S + (ulong)row * a.rows;
  int sq = a.seq[q];
  if (a.rows <= 256) {
    float r[8], m = -INFINITY;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 8; ++k) {
      uint j = k * 32 + lane;
      if (j < a.rows) { float v = a.seq[j] == sq ? s[j] * a.scale : -INFINITY; r[k] = v; m = max(m, v); }
    }
    m = simd_max(m);
    float sum = 0;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 8; ++k) {
      if (k * 32 + lane < a.rows) { float e = exp(r[k] - m); r[k] = e; sum += e; }
    }
    float inv = 1.f / simd_sum(sum);
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 8; ++k) {
      uint j = k * 32 + lane;
      if (j < a.rows) s[j] = r[k] * inv;
    }
    return;
  }
  float m = -INFINITY;
  for (uint j = lane; j < a.rows; j += 32) { float v = a.seq[j] == sq ? s[j] * a.scale : -INFINITY; s[j] = v; m = max(m, v); }
  m = simd_max(m);
  float sum = 0;
  for (uint j = lane; j < a.rows; j += 32) { float e = exp(s[j] - m); s[j] = e; sum += e; }
  float inv = 1.f / simd_sum(sum);
  for (uint j = lane; j < a.rows; j += 32) s[j] *= inv;
}
// fused ESM-C self-attention (D = 64, scale = 1/8, chain mask seq[q] == seq[k]): 4 simdgroups (32 queries) per threadgroup
kernel void ef2_esmc_attend(constant EsmcAttendArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tile[32][68];
  threadgroup int k_seq[32];
  const uint r0 = tg.x * 32, h = tg.y, C = a.C;
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  const uint qi = r0 + sg * 8 + sm;
  const int sq = qi < a.rows ? a.seq[qi] : -1;
  _Pragma("clang loop unroll(full)")
  for (uint e = tid; e < 32 * 16; e += 128) {
    uint r = e >> 4, c4 = (e & 15) << 2;
    float4 v = (r0 + r < a.rows) ? *(device const float4*)(a.q + (ulong)(r0 + r) * C + h * 64 + c4) * 0.125f : float4(0.f);
    *(threadgroup float4*)&tile[r][c4] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 qf[8], o[8];
  _Pragma("clang loop unroll(full)")
  for (int dk = 0; dk < 8; ++dk) {
    simdgroup_load(qf[dk], &tile[sg * 8][dk * 8], 68);
    o[dk] = simdgroup_float8x8(0);
  }
  float m = -INFINITY, l = 0.f;
  for (uint k0 = 0; k0 < a.rows; k0 += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) k_seq[tid] = (k0 + tid < a.rows) ? a.seq[k0 + tid] : -2;
    _Pragma("clang loop unroll(full)")
    for (uint e = tid; e < 32 * 16; e += 128) {
      uint r = e >> 4, c4 = (e & 15) << 2;
      float4 v = (k0 + r < a.rows) ? *(device const float4*)(a.k + (ulong)(k0 + r) * C + h * 64 + c4) : float4(0.f);
      *(threadgroup float4*)&tile[r][c4] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 sc[4];
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) sc[j] = simdgroup_float8x8(0);
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j)
      _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 8; ++dk) {
        simdgroup_float8x8 kt;
        simdgroup_load(kt, &tile[j * 8][dk * 8], 68, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sc[j], qf[dk], kt, sc[j]);
      }
    float mx = m;
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) {
      thread auto& e = sc[j].thread_elements();
      _Pragma("clang loop unroll(full)") for (int t = 0; t < 2; ++t) {
        uint kj = j * 8 + sn + t;
        bool ok = (k0 + kj < a.rows) && (qi < a.rows) && (k_seq[kj] == sq);
        float v = ok ? e[t] : -INFINITY;
        e[t] = v;
        mx = max(mx, v);
      }
    }
    mx = max(mx, simd_shuffle_xor(mx, (ushort)1));
    mx = max(mx, simd_shuffle_xor(mx, (ushort)8));
    float corr = (m == -INFINITY || mx == -INFINITY) ? 0.f : exp(m - mx);
    float row_sum = 0.f;
    simdgroup_float8x8 p[4];
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) {
      thread auto& e = sc[j].thread_elements();
      thread auto& pe = p[j].thread_elements();
      _Pragma("clang loop unroll(full)") for (int t = 0; t < 2; ++t) {
        float v = (mx == -INFINITY || e[t] == -INFINITY) ? 0.f : exp(e[t] - mx);
        pe[t] = v;
        row_sum += v;
      }
    }
    row_sum += simd_shuffle_xor(row_sum, (ushort)1);
    row_sum += simd_shuffle_xor(row_sum, (ushort)8);
    l = l * corr + row_sum;
    m = mx;
    _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 8; ++dk) {
      thread auto& oe = o[dk].thread_elements();
      oe[0] *= corr; oe[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (uint e = tid; e < 32 * 16; e += 128) {
      uint r = e >> 4, c4 = (e & 15) << 2;
      float4 v = (k0 + r < a.rows) ? *(device const float4*)(a.v + (ulong)(k0 + r) * a.ldv + h * 64 + c4) : float4(0.f);
      *(threadgroup float4*)&tile[r][c4] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j)
      _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 8; ++dk) {
        simdgroup_float8x8 vf;
        simdgroup_load(vf, &tile[j * 8][dk * 8], 68);
        simdgroup_multiply_accumulate(o[dk], p[j], vf, o[dk]);
      }
  }
  if (qi < a.rows) {
    float inv = l > 0.f ? 1.f / l : 0.f;
    device half* out = a.ctx + (ulong)qi * C + h * 64;
    _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 8; ++dk) {
      thread auto& oe = o[dk].thread_elements();
      *(device half2*)(out + dk * 8 + sn) = half2(half(oe[0] * inv), half(oe[1] * inv));
    }
  }
}
// the tower's rows onto the tokens; a token the tower never saw takes the shim at a zero state
kernel void ef2_scatter_rows(LF_ARGS(ScatterRowsArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.T * a.C) return;
  uint token = lf_udiv(t, a.C), c = t - token * a.C;
  int r = a.tokenToRow[token];
  a.out[t] = r < 0 ? a.zero[c] : a.rows[(uint)r * a.C + c];
}
// [a * b | a - b] for pair rows i0.. (half: the pair MLP's input)
kernel void ef2_pair_join(LF_ARGS(PairJoinArgs)) {
  ulong t = LF_INDEX, C4 = a.C >> 2;
  if (t >= (ulong)a.bi * a.T * C4) return;
  uint ij = lf_udiv((uint)t, (uint)C4), c4 = (uint)t - ij * (uint)C4;
  uint ii = lf_udiv(ij, a.T), j = ij - ii * a.T, i = a.i0 + ii;
  float4 x = ((device const float4*)(a.s + i * a.C))[c4], y = ((device const float4*)(a.s + j * a.C))[c4];
  ((device half4*)(a.out + (ulong)ij * 2 * a.C))[c4] = half4(x * y);
  ((device half4*)(a.out + (ulong)ij * 2 * a.C + a.C))[c4] = half4(x - y);
}
kernel void ef2_pair_join_f(LF_ARGS(PairJoinFArgs)) {
  ulong t = LF_INDEX, C4 = a.C >> 2;
  if (t >= (ulong)a.bi * a.T * C4) return;
  uint ij = lf_udiv((uint)t, (uint)C4), c4 = (uint)t - ij * (uint)C4;
  uint ii = lf_udiv(ij, a.T), j = ij - ii * a.T, i = a.i0 + ii;
  float4 x = ((device const float4*)(a.s + i * a.C))[c4], y = ((device const float4*)(a.s + j * a.C))[c4];
  ((device float4*)(a.out + (ulong)ij * 2 * a.C))[c4] = x * y;
  ((device float4*)(a.out + (ulong)ij * 2 * a.C + a.C))[c4] = x - y;
}

// ---------------------------------------------------------------- the trunk
// z_init = rows[i] + cols[j] + relpos + bonds * w_bond + lm_z: a threadgroup a pair, a thread a channel
kernel void ef2_zinit(constant ZInitArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint ij = tg.y * ng.x + tg.x;
  if (ij >= a.T * a.T) return;
  uint i = lf_udiv(ij, a.T), j = ij - i * a.T;
  float b = a.bonds[ij];
  for (uint c = tid * 4; c < a.C; c += 1024) {
    ulong t = (ulong)ij * a.C + c;
    float4 v = *(device const float4*)(a.rows + i * a.C + c) + *(device const float4*)(a.cols + j * a.C + c) +
               relPosAt4(a.rel, i, j, c, a.C) + b * *(device const float4*)(a.wBond + c);
    if (a.lmZ) v += *(device const float4*)(a.lmZ + t);
    *(device float4*)(a.z + t) = v;
  }
}
kernel void ef2_trigate_weight(LF_ARGS(TriGateWArgs)) {
  uint t = (uint)LF_INDEX, W4 = a.C;
  if (t >= a.rows * W4) return;
  uint r = lf_udiv(t, W4), col = (t - r * W4) << 2;
  uint kind = a.quarters ? (col % 128) / 32 : (col % 32) / 8;
  uint c = a.quarters ? 32 * (col / 128) + col % 32 : 8 * (col / 32) + col % 8;
  uint k = kind == 1 ? 2 : kind == 2 ? 1 : kind;      // (blocks of 8 or quarters of 32: pa ga pb gb; a and b interleaved by channel)
  device const half* src = (k < 2 ? a.proj : a.gate) + (ulong)r * 2 * a.C + 2 * c + (k & 1);
  ((device half4*)a.out)[t] = half4(src[0], src[2], src[4], src[6]);
}
kernel void ef2_fill(LF_ARGS(FillFArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.out[t] = a.v;
}
// ---------------------------------------------------------------- the released models' recycle (parcae)
// counter-based randoms, cuda/ef2's (trunk.cuh): a uniform in [0, 1) from (seed, stream, index), splitmix64 twice
inline ulong ef2_mix64(ulong z) {
  z += 0x9E3779B97F4A7C15ul; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul; z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
  return z ^ (z >> 31);
}
inline float ef2_uniform01(ulong seed, ulong stream, ulong i) {
  return (float)(ef2_mix64(seed ^ ef2_mix64(stream * 0x100000001B3ul + i)) >> 40) * (1.f / 16777216.f);
}
// the initial pair state: a normal of std sqrt(2 / (5 C)) truncated at 3 std (biohub's trunc_normal_, which redraws
// past the bounds; a Box-Muller draw redrawn the same way)
kernel void ef2_trunc_normal(LF_ARGS(TruncNormalArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n) return;
  float v = 0.f;
  for (uint k = 0; k < 64; ++k) {
    float u1 = max(ef2_uniform01(a.seed, 2 * k, t), 1e-7f), u2 = ef2_uniform01(a.seed, 2 * k + 1, t);
    v = sqrt(-2.f * log(u1)) * cospi(2.f * u2);
    if (fabs(v) <= 3.f) break;
  }
  a.z[t] = clamp(v, -3.f, 3.f) * a.std;
}
// out = dropout(in, p): zeroed with probability p, else scaled by 1 / (1 - p) (F.dropout, training=True)
kernel void ef2_dropout(LF_ARGS(DropoutArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.out[t] = ef2_uniform01(a.seed, a.stream, t) < a.p ? 0.f : a.in[t] / (1.f - a.p);
}
// ---------------------------------------------------------------- the full ESMFold2's MSA encoder (cuda/ef2's msa.cuh)
// a row's one-hot where its column is live, the two deletion features always, plus the projected s_inputs
kernel void ef2_msa_embed(LF_ARGS(MsaEmbedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.count * a.C) return;
  uint c = (uint)(t % a.C); ulong row = t / a.C; uint token = (uint)(row % a.T);
  float d = a.del[row];
  float v = a.mask[row] > 0.f ? a.E[(ulong)a.rows[row] * a.C + c] : 0.f;
  if (d > 0.f) v += a.E[33ul * a.C + c];
  v += 1.57079632679489662f * atan(d / 3.f) * a.E[34ul * a.C + c];
  a.m[t] = v + a.fromInputs[(ulong)token * a.C + c];
}
// x [rows][a | b] times the row's mask, split into a [rows][H] and b [rows][H]
kernel void ef2_opm_split(LF_ARGS(OpmSplitArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.H) return;
  ulong r = t / a.H; uint c = (uint)(t % a.H); float k = a.mask[r];
  a.a[t] = a.x[r * 2 * a.H + c] * k; a.b[t] = a.x[r * 2 * a.H + a.H + c] * k;
}
// P [(i, c)][(j, e)] -> Pp [(i, j)][(c, e)]
kernel void ef2_opm_permute(LF_ARGS(OpmPermuteArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.bi * a.T * a.H * a.H) return;
  uint e = (uint)(t % a.H); ulong r = t / a.H; uint c = (uint)(r % a.H); r /= a.H; uint j = (uint)(r % a.T); uint i = (uint)(r / a.T);
  a.Pp[t] = a.P[(((ulong)i * a.H + c) * a.T + j) * a.H + e];
}
// pair[(i0 + i) j] += (X + b) / max(count, 1)
kernel void ef2_opm_add(LF_ARGS(OpmAddArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.bi * a.T * a.C) return;
  uint c = (uint)(t % a.C); ulong at = (ulong)a.i0 * a.T + t / a.C;
  a.pair[at * a.C + c] += (a.X[t] + a.bias[c]) / max(a.count[at], 1.f);
}
// w[h][i][j] = softmax over j of flat[(i j)][h], a threadgroup a (h, i) (every token is real: no mask)
kernel void ef2_pwa_softmax(constant PwaSoftmaxArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint row = tg.y * ng.x + tg.x;
  if (row >= a.heads * a.T) return;
  uint h = row / a.T, i = row - h * a.T;
  device float* out = a.w + (ulong)row * a.T;
  threadgroup float red[8];
  float m = -INFINITY;
  for (uint j = tid; j < a.T; j += 256) { float v = a.flat[((ulong)i * a.T + j) * a.heads + h]; out[j] = v; m = max(m, v); }
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0]; for (int k = 1; k < 8; ++k) m = max(m, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0;
  for (uint j = tid; j < a.T; j += 256) { float e = exp(out[j] - m); out[j] = e; sum += e; }
  sum = simd_sum(sum);
  if (lane == 0) red[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0; for (int k = 0; k < 8; ++k) tot += red[k];
  float inv = 1.f / tot;
  for (uint j = tid; j < a.T; j += 256) out[j] *= inv;
}
// v [s][j][h d + e] -> [h][j][s][e]
kernel void ef2_pwa_to_heads(LF_ARGS(PwaHeadsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.S * a.T * a.heads * a.d) return;
  uint e = (uint)(t % a.d); ulong r = t / a.d; uint s = (uint)(r % a.S); r /= a.S; uint j = (uint)(r % a.T); uint h = (uint)(r / a.T);
  a.out[t] = a.in[((ulong)s * a.T + j) * a.heads * a.d + h * a.d + e];
}
// o [h][i][s][e] -> [s][i][h d + e], times sigmoid(gate)
kernel void ef2_pwa_from_heads(LF_ARGS(PwaHeadsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.S * a.T * a.heads * a.d) return;
  uint c = (uint)(t % (a.heads * a.d)); ulong si = t / (a.heads * a.d); uint i = (uint)(si % a.T), s = (uint)(si / a.T);
  uint h = c / a.d, e = c - h * a.d;
  a.out[t] = a.in[(((ulong)h * a.T + i) * a.S + s) * a.d + e] / (1.f + exp(-a.gate[t]));
}
// z = decay * z + y, the decay per channel
kernel void ef2_decay_update(LF_ARGS(DecayArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.z[t] = a.decay[t % a.C] * a.z[t] + a.y[t];
}
// pair positions [p0, p0 + cnt) of z + z^T
kernel void ef2_sym_rows(LF_ARGS(SymRowsArgs)) {
  ulong t = LF_INDEX, C4 = a.C >> 2;
  if (t >= a.cnt * C4) return;
  uint pc = lf_udiv((uint)t, (uint)C4), c4 = (uint)t - pc * (uint)C4;
  uint ij = (uint)a.p0 + pc, i = lf_udiv(ij, a.T), j = ij - i * a.T;
  ((device float4*)a.out)[t] = ((device const float4*)(a.z + (ulong)ij * a.C))[c4] + ((device const float4*)(a.z + ((ulong)j * a.T + i) * a.C))[c4];
}
// the softmax mass of each pair's first contact_bins bins
kernel void ef2_contacts(LF_ARGS(ContactsArgs)) {
  ulong ij = LF_INDEX;
  if (ij >= a.pairs) return;
  device const float* l = a.logits + ij * a.bins;
  float mx = -INFINITY;
  for (uint b = 0; b < a.bins; ++b) mx = max(mx, l[b]);
  float total = 0, near = 0;
  int cb = a.contactBins[ij];
  for (uint b = 0; b < a.bins; ++b) { float w = exp(l[b] - mx); total += w; if ((int)b < cb) near += w; }
  a.out[ij] = near / max(total, 1e-30f);
}
kernel void ef2_quantise(LF_ARGS(QuantiseArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.out[t] = (uchar)clamp((int)rint(a.x[t] / a.scale), 0, 255);
}

// ---------------------------------------------------------------- the atoms
constant float RMS_EPS = 1.1920928955078125e-7f;
constant uint ATOM_FEATURES = 389;   // 3 + 1 + 1 + 128 + 4 * 64
kernel void ef2_atom_features(LF_ARGS(AtomFeaturesArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.A * ATOM_FEATURES) return;
  uint atom = lf_udiv(t, ATOM_FEATURES), f = t - atom * ATOM_FEATURES;
  bool live = a.mask[atom] != 0.f;
  float v = 0.f;
  if (f < 3) v = a.pos[atom * 3 + f];
  else if (f == 3) v = a.charge[atom];
  else if (f == 4) v = a.mask[atom];
  else if (f < 5 + 128) v = live && a.element[atom] == (int)f - 5 ? 1.f : 0.f;
  else { uint i = (f - 133) / 64, ch = (f - 133) % 64; v = live && a.nameChars[atom * 4 + i] == (int)ch ? 1.f : 0.f; }
  if (a.outH) a.outH[t] = (half)v;
  else a.out[t] = v;
}
kernel void ef2_atom_c0(constant AtomC0Args& a [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  uint atom = tid >> 5, l = tid & 31;
  if (atom >= a.A || a.C != 128) return;
  bool live = a.mask[atom] != 0.f;
  float4 v;
  if (a.wh) {
    float px = (float)(half)a.pos[atom * 3 + 0];
    float py = (float)(half)a.pos[atom * 3 + 1];
    float pz = (float)(half)a.pos[atom * 3 + 2];
    float chg = (float)(half)a.charge[atom];
    float msk = (float)(half)a.mask[atom];
    device const half4* w4 = (device const half4*)a.wh;
    v = float4(w4[0 * 32 + l]) * px + float4(w4[1 * 32 + l]) * py + float4(w4[2 * 32 + l]) * pz +
        float4(w4[3 * 32 + l]) * chg + float4(w4[4 * 32 + l]) * msk;
    if (live) {
      int el = a.element[atom];
      if (el >= 0 && el < 128) v += float4(w4[(5 + el) * 32 + l]);
      for (uint i = 0; i < 4; ++i) {
        int ch = a.nameChars[atom * 4 + i];
        if (ch >= 0 && ch < 64) v += float4(w4[(133 + i * 64 + ch) * 32 + l]);
      }
    }
  } else {
    float px = a.pos[atom * 3 + 0], py = a.pos[atom * 3 + 1], pz = a.pos[atom * 3 + 2];
    float chg = a.charge[atom], msk = a.mask[atom];
    device const float4* w4 = (device const float4*)a.wf;
    v = w4[0 * 32 + l] * px + w4[1 * 32 + l] * py + w4[2 * 32 + l] * pz +
        w4[3 * 32 + l] * chg + w4[4 * 32 + l] * msk;
    if (live) {
      int el = a.element[atom];
      if (el >= 0 && el < 128) v += w4[(5 + el) * 32 + l];
      for (uint i = 0; i < 4; ++i) {
        int ch = a.nameChars[atom * 4 + i];
        if (ch >= 0 && ch < 64) v += w4[(133 + i * 64 + ch) * 32 + l];
      }
    }
  }
  float mean = simd_sum(v.x + v.y + v.z + v.w) * (1.f / 128.f);
  float4 d = v - mean;
  float inv = rsqrt(simd_sum(dot(d, d)) * (1.f / 128.f) + 1e-5f);
  float4 y = d * inv * ((device const float4*)a.scale)[l] + ((device const float4*)a.offset)[l];
  ulong off = (ulong)atom * 32 + l;
  ((device float4*)a.c0)[off] = y;
  if (a.copyOut) ((device float4*)a.copyOut)[off] = y;
  if (a.siluH || a.siluF) {
    float4 sv = float4(lf_silu(y.x), lf_silu(y.y), lf_silu(y.z), lf_silu(y.w));
    if (a.siluH) ((device half4*)a.siluH)[off] = half4(sv);
    else ((device float4*)a.siluF)[off] = sv;
  }
}
// the rotary table [A, 16]: x's two pairs (base 20), y's, z's, then the space uid's ten (base 10000), in bfloat16
kernel void ef2_rope_table(LF_ARGS(RopeTableArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.A * 16) return;
  uint atom = t >> 4, at = t & 15;
  float angle;
  if (at < 6) { uint axis = at / 2, i = at % 2; angle = a.pos[atom * 3 + axis] * (1.f / precise::pow(20.f, (float)i / 2.f)); }
  else { uint i = at - 6; angle = (float)a.uid[atom] * (1.f / precise::pow(10000.f, (float)i / 10.f)); }
  a.cosT[t] = lf_bf16(precise::cos(angle));
  a.sinT[t] = lf_bf16(precise::sin(angle));
}
kernel void ef2_silu(LF_ARGS(SiluArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n) return;
  float v = lf_silu(a.x[t]);
  if (a.yh) a.yh[t] = (half)v;
  else a.y[t] = v;
}
kernel void ef2_swiglu(LF_ARGS(SwigluArgs)) {      // g = silu(h[:, :F]) h[:, F:]
  uint t = (uint)LF_INDEX;
  if (t >= a.rows * a.F) return;
  uint r = lf_udiv(t, a.F), c = t - r * a.F;
  a.g[t] = lf_silu(a.h[(ulong)r * 2 * a.F + c]) * a.h[(ulong)r * 2 * a.F + a.F + c];
}
// rms(x) * (1 + mod[scale]) + mod[shift], a simdgroup a row
kernel void ef2_rms_modulate(constant RmsModArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                             uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  uint row = (tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  device float* xr = a.x + (ulong)row * a.C;
  device const float* m = a.mod + (ulong)row * 6 * a.C;
  device const float* dr = a.resD ? a.resD + (ulong)row * a.C : nullptr;
  if (a.yh && a.C == 128) {
    device float4* x4 = (device float4*)xr;
    float4 x = x4[lane];
    if (dr) {
      x += ((device const float4*)(m + a.resWhich * 128))[lane] * ((device const float4*)dr)[lane];
      x4[lane] = x;
    }
    float inv = rsqrt(simd_sum(dot(x, x)) * (1.f / 128.f) + RMS_EPS);
    float4 sc = ((device const float4*)(m + a.scale * 128))[lane];
    float4 sh = ((device const float4*)(m + a.shift * 128))[lane];
    ((device half4*)(a.yh + (ulong)row * 128))[lane] = half4(x * inv * (1.f + sc) + sh);
    return;
  }
  float s = 0;
  for (uint c = lane; c < a.C; c += 32) {
    float x = xr[c];
    if (dr) { x += m[a.resWhich * a.C + c] * dr[c]; xr[c] = x; }
    s += x * x;
  }
  float inv = rsqrt(simd_sum(s) / a.C + RMS_EPS);
  for (uint c = lane; c < a.C; c += 32) {
    float v = xr[c] * inv * (1.f + m[a.scale * a.C + c]) + m[a.shift * a.C + c];
    if (a.yh) a.yh[(ulong)row * a.C + c] = (half)v;
    else a.y[(ulong)row * a.C + c] = v;
  }
}
// q and k (heads of 32, inside the packed [A, ld] qkv): rms per head, rotate, narrow to bfloat16; v narrowed; a simdgroup an (atom, head)
kernel void ef2_qkv_prepare(constant QkvPrepArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  if (a.C == 128 && a.heads == 4 && (a.ld & 3) == 0) {
    uint atom = (tg.y * ng.x + tg.x) * 8 + sg;
    if (atom >= a.A) return;
    device float4* row4 = (device float4*)(a.qkv + (ulong)atom * a.ld);
    if (a.bf16) {
      float4 vv = row4[64 + lane];
      row4[64 + lane] = float4(lf_bf16(vv.x), lf_bf16(vv.y), lf_bf16(vv.z), lf_bf16(vv.w));
    }
    float4 qv = row4[lane], kv = row4[32 + lane];
    float sq = dot(qv, qv), sk = dot(kv, kv);
    sq += simd_shuffle_xor(sq, (ushort)1); sk += simd_shuffle_xor(sk, (ushort)1);
    sq += simd_shuffle_xor(sq, (ushort)2); sk += simd_shuffle_xor(sk, (ushort)2);
    sq += simd_shuffle_xor(sq, (ushort)4); sk += simd_shuffle_xor(sk, (ushort)4);
    float invQ = rsqrt(sq * (1.f / 32.f) + RMS_EPS);
    float invK = rsqrt(sk * (1.f / 32.f) + RMS_EPS);
    qv *= invQ; kv *= invK;
    uint d4 = lane & 3;
    float4 c = ((device const float4*)(a.cosT + (ulong)atom * 16))[d4];
    float4 sn = ((device const float4*)(a.sinT + (ulong)atom * 16))[d4];
    float4 qPair = simd_shuffle_xor(qv, (ushort)4), kPair = simd_shuffle_xor(kv, (ushort)4);
    float4 qOut = (lane & 4) ? (qv * c + qPair * sn) : (qv * c - qPair * sn);
    float4 kOut = (lane & 4) ? (kv * c + kPair * sn) : (kv * c - kPair * sn);
    if (a.bf16) {
      qOut = float4(lf_bf16(qOut.x), lf_bf16(qOut.y), lf_bf16(qOut.z), lf_bf16(qOut.w));
      kOut = float4(lf_bf16(kOut.x), lf_bf16(kOut.y), lf_bf16(kOut.z), lf_bf16(kOut.w));
    }
    row4[lane] = qOut;
    row4[32 + lane] = kOut;
    return;
  }
  uint ah = (tg.y * ng.x + tg.x) * 8 + sg;
  if (ah >= a.A * a.heads) return;
  uint atom = lf_udiv(ah, a.heads), head = ah - atom * a.heads;
  device float* row = a.qkv + (ulong)atom * a.ld + head * 32;
  if (a.bf16) row[2 * a.C + lane] = lf_bf16(row[2 * a.C + lane]);
  float qv = row[lane], kv = row[a.C + lane];
  float invQ = rsqrt(simd_sum(qv * qv) / 32.f + RMS_EPS);
  float invK = rsqrt(simd_sum(kv * kv) / 32.f + RMS_EPS);
  qv *= invQ; kv *= invK;
  float qHi = simd_shuffle_down(qv, 16), kHi = simd_shuffle_down(kv, 16);
  if (lane < 16) {
    float c = a.cosT[atom * 16 + lane], sn = a.sinT[atom * 16 + lane];
    float qLo = qv * c - qHi * sn, qH = qHi * c + qv * sn;
    float kLo = kv * c - kHi * sn, kH = kHi * c + kv * sn;
    row[lane] = a.bf16 ? lf_bf16(qLo) : qLo;
    row[16 + lane] = a.bf16 ? lf_bf16(qH) : qH;
    row[a.C + lane] = a.bf16 ? lf_bf16(kLo) : kLo;
    row[a.C + 16 + lane] = a.bf16 ? lf_bf16(kH) : kH;
  }
}
// the windowed attention (no [heads, A, A] matrix): a valid query's keys are the valid atoms within halfWindow in rank,
// itself among them. A threadgroup takes 32 consecutive valid queries of one head (four simdgroups, eight queries each)
// and walks the union of their windows in tiles of 32 keys; online softmax, a lane a key for the scores, a lane a
// channel for the output
constant int SWA_KT = 32, SWA_Q = 32;
kernel void ef2_swa(constant SwaArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float Ks[SWA_KT][36];
  threadgroup float Vs[SWA_KT][36];
  const int nValid = (int)a.nValid, hw = (int)a.halfWindow, C = (int)a.C, ld = (int)a.ld;
  const int head = tg.y;
  const int r0 = tg.x * SWA_Q, r1 = min(nValid, r0 + SWA_Q) - 1;
  const int klo = max(0, r0 - hw), khi = min(nValid - 1, r1 + hw);
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  const int rq = r0 + 8 * (int)warp, q_idx = rq + sm;
  _Pragma("clang loop unroll(full)")
  for (int e = tid; e < SWA_Q * 8; e += 128) {
    int u = e >> 3, part = (e & 7) << 2;
    device const float* q = a.qkv + (ulong)a.valid[min(r0 + u, nValid - 1)] * ld + head * 32 + part;
    *(threadgroup float4*)&Ks[u][part] = *(device const float4*)q * a.scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 qf[4], o[4];
  _Pragma("clang loop unroll(full)")
  for (int dk = 0; dk < 4; ++dk) {
    simdgroup_load(qf[dk], &Ks[warp * 8][dk * 8], 36);
    o[dk] = simdgroup_float8x8(0);
  }
  float m = -INFINITY, l = 0.f;
  for (int t0 = klo; t0 <= khi; t0 += SWA_KT) {
    int n = min(SWA_KT, khi - t0 + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (int e = tid; e < SWA_KT * 8; e += 128) {
      int j = e >> 3, part = (e & 7) << 2;
      if (j < n) {
        device const float* row = a.qkv + (ulong)a.valid[t0 + j] * ld + head * 32 + part;
        *(threadgroup float4*)&Ks[j][part] = *(device const float4*)(row + C);
        *(threadgroup float4*)&Vs[j][part] = *(device const float4*)(row + 2 * C);
      } else {
        *(threadgroup float4*)&Ks[j][part] = float4(0.f);
        *(threadgroup float4*)&Vs[j][part] = float4(0.f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rq > r1) continue;
    int lo = max(rq - hw, t0) - t0, hi = min(min(rq + 7, r1) + hw, t0 + n - 1) - t0;
    if (lo > hi) continue;
    simdgroup_float8x8 sc[4];
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) sc[j] = simdgroup_float8x8(0);
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) {
      if (j * 8 <= hi && j * 8 + 7 >= lo) {
        _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 4; ++dk) {
          simdgroup_float8x8 kt;
          simdgroup_load(kt, &Ks[j * 8][dk * 8], 36, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(sc[j], qf[dk], kt, sc[j]);
        }
      }
    }
    float mx = m;
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) {
      thread auto& e = sc[j].thread_elements();
      _Pragma("clang loop unroll(full)") for (int t = 0; t < 2; ++t) {
        int j_local = j * 8 + sn + t, key = t0 + j_local;
        bool ok = (j_local <= hi) && (q_idx <= r1) && (abs(key - q_idx) <= hw);
        float v = ok ? e[t] : -INFINITY;
        e[t] = v;
        mx = max(mx, v);
      }
    }
    mx = max(mx, simd_shuffle_xor(mx, (ushort)1));
    mx = max(mx, simd_shuffle_xor(mx, (ushort)8));
    float corr = (m == -INFINITY || mx == -INFINITY) ? 0.f : exp(m - mx);
    float row_sum = 0.f;
    simdgroup_float8x8 p[4];
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) {
      thread auto& e = sc[j].thread_elements();
      thread auto& pe = p[j].thread_elements();
      _Pragma("clang loop unroll(full)") for (int t = 0; t < 2; ++t) {
        float v = (e[t] == -INFINITY) ? 0.f : exp(e[t] - mx);
        pe[t] = v;
        row_sum += v;
      }
    }
    row_sum += simd_shuffle_xor(row_sum, (ushort)1);
    row_sum += simd_shuffle_xor(row_sum, (ushort)8);
    l = l * corr + row_sum;
    m = mx;
    _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 4; ++dk) {
      thread auto& oe = o[dk].thread_elements();
      oe[0] *= corr; oe[1] *= corr;
    }
    _Pragma("clang loop unroll(full)") for (int j = 0; j < 4; ++j) {
      if (j * 8 <= hi && j * 8 + 7 >= lo) {
        _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 4; ++dk) {
          simdgroup_float8x8 vf;
          simdgroup_load(vf, &Vs[j * 8][dk * 8], 36);
          simdgroup_multiply_accumulate(o[dk], p[j], vf, o[dk]);
        }
      }
    }
  }
  if (q_idx <= r1) {
    ulong atom = (ulong)a.valid[q_idx];
    float inv = l > 0.f ? 1.f / l : 0.f;
    if (a.outH) {
      device const float* g = a.qkv + atom * ld + 3 * C + head * 32;
      device half* out = a.outH + atom * C + head * 32;
      _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 4; ++dk) {
        thread auto& oe = o[dk].thread_elements();
        float2 gv = *(device const float2*)(g + dk * 8 + sn);
        *(device half2*)(out + dk * 8 + sn) = half2(half(oe[0] * inv * lf_sigmoid(gv.x)), half(oe[1] * inv * lf_sigmoid(gv.y)));
      }
    } else {
      device float* out = a.ctx + atom * C + head * 32;
      _Pragma("clang loop unroll(full)") for (int dk = 0; dk < 4; ++dk) {
        thread auto& oe = o[dk].thread_elements();
        *(device float2*)(out + dk * 8 + sn) = float2(oe[0] * inv, oe[1] * inv);
      }
    }
  }
}
kernel void ef2_gate_live(LF_ARGS(GateLiveArgs)) {
  uint t = (uint)LF_INDEX, C4 = a.C >> 2;
  if (t >= a.A * C4) return;
  uint atom = a.C == 128 ? (t >> 5) : lf_udiv(t, C4), c4 = a.C == 128 ? (t & 31) : (t - atom * C4);
  float4 v = float4(0.f);
  if (a.mask[atom] != 0.f) {
    float4 cx = ((device const float4*)a.ctx)[t];
    float4 gt = *(device const float4*)(a.gate + (ulong)atom * a.ld + (c4 << 2));
    v = cx * float4(lf_sigmoid(gt.x), lf_sigmoid(gt.y), lf_sigmoid(gt.z), lf_sigmoid(gt.w));
  }
  if (a.outH) ((device half4*)a.outH)[t] = half4(v);
  else ((device float4*)a.ctx)[t] = v;
}
kernel void ef2_gated_add(LF_ARGS(GatedAddArgs)) {
  uint t = (uint)LF_INDEX, C4 = a.C >> 2;
  if (t >= a.A * C4) return;
  uint atom = a.C == 128 ? (t >> 5) : lf_udiv(t, C4), c4 = a.C == 128 ? (t & 31) : (t - atom * C4);
  ((device float4*)a.x)[t] += ((device const float4*)(a.mod + ((ulong)atom * 6 + a.which) * a.C))[c4] * ((device const float4*)a.d)[t];
}
// the mean over each token's atoms, weighted by the mask, into the first C columns of rows ld apart
kernel void ef2_scatter_mean(LF_ARGS(ScatterMeanArgs)) {
  if (a.outH && a.C == 768 && (a.adaLd & 3) == 0) {
    const uint token = (uint)(LF_INDEX >> 5), lane = _tid & 31;
    if (token >= a.T) return;
    float4 s[6] = {float4(0.f), float4(0.f), float4(0.f), float4(0.f), float4(0.f), float4(0.f)};
    float w = 0.f;
    const int k0 = a.tokenStart[token], k1 = a.tokenStart[token + 1];
    for (int k = k0; k < k1; ++k) {
      const int at = a.tokenAtoms[k];
      const float m = a.mask[at];
      device const float4* vr4 = (device const float4*)(a.v + (ulong)at * 768);
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < 6; ++u) s[u] += vr4[lane + 32 * u] * m;
      w += m;
    }
    const float invW = 1.f / max(w, 1e-9f);
    device const float4* add4 = a.add ? (device const float4*)(a.add + (ulong)token * 768) : nullptr;
    device float4* out4 = (device float4*)(a.out + (ulong)token * a.ld);
    float4 xv[6];
    float sum = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int u = 0; u < 6; ++u) {
      const uint idx = lane + 32 * u;
      const float4 v = s[u] * invW + (add4 ? add4[idx] : float4(0.f));
      out4[idx] = v;
      xv[u] = v;
      sum += v.x + v.y + v.z + v.w;
    }
    const float mean = simd_sum(sum) * (1.f / 768.f);
    float var = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int u = 0; u < 6; ++u) { const float4 d = xv[u] - mean; var += dot(d, d); }
    const float inv = rsqrt(simd_sum(var) * (1.f / 768.f) + 1e-5f);
    device const float4* gr4 = (device const float4*)(a.g + (ulong)token * a.adaLd);
    device const float4* gb4 = (device const float4*)a.gb;
    device const float4* shr4 = (device const float4*)(a.sh + (ulong)token * a.adaLd);
    device half4* yr4 = (device half4*)(a.outH + (ulong)token * 768);
    _Pragma("clang loop unroll(full)")
    for (int u = 0; u < 6; ++u) {
      const uint idx = lane + 32 * u;
      const float4 u4 = gr4[idx] + gb4[idx];
      const float4 sig = float4(lf_sigmoid(u4.x), lf_sigmoid(u4.y), lf_sigmoid(u4.z), lf_sigmoid(u4.w));
      yr4[idx] = half4(((xv[u] - mean) * inv) * sig + shr4[idx]);
    }
    return;
  }
  uint t = (uint)LF_INDEX, C4 = a.C >> 2;
  if (t >= a.T * C4) return;
  uint token = lf_udiv(t, C4), c4 = t - token * C4, c = c4 << 2;
  float4 s = float4(0.f); float w = 0.f;
  for (int k = a.tokenStart[token]; k < a.tokenStart[token + 1]; ++k) {
    int at = a.tokenAtoms[k];
    float m = a.mask[at];
    s += *(device const float4*)(a.v + (ulong)at * a.C + c) * m; w += m;
  }
  float4 v = s / max(w, 1e-9f);
  if (a.add) v += *(device const float4*)(a.add + (ulong)token * a.C + c);
  *(device float4*)(a.out + (ulong)token * a.ld + c) = v;
}
kernel void ef2_tail_inputs(LF_ARGS(TailInputsArgs)) {
  uint t = (uint)LF_INDEX, F = 2 * a.K + 1;
  if (t >= a.T * F) return;
  uint token = lf_udiv(t, F), f = t - token * F;
  float v = f < a.K ? a.aatype[token * a.K + f] : !a.profile ? 0.f : f < 2 * a.K ? a.profile[token * a.K + f - a.K] : a.delMean[token];
  a.out[(ulong)token * a.ld + a.C + f] = v;
}

// ---------------------------------------------------------------- the diffusion module
// rows [p0, p0 + n) of [z | rel_pos] (or fused with zInputNorm into outH when outH is set)
kernel void ef2_join_pair_rel(LF_ARGS(JoinPairRelArgs)) {
  if (a.outH && a.C == 256) {
    const uint row = (_tg.y * _ng.x + _tg.x) * 8 + (_tid >> 5), lane = _tid & 31;
    if (row >= a.n) return;
    const uint p = (uint)a.p0 + row, i = lf_udiv(p, a.T), j = p - i * a.T;
    const int rb = 32, cb = 2, w = 2 * rb + 2;
    const bool sameChain = a.rel.asym[i] == a.rel.asym[j], sameRes = a.rel.ri[i] == a.rel.ri[j], sameEnt = a.rel.ent[i] == a.rel.ent[j];
    const int b0 = sameChain ? clamp(a.rel.ri[i] - a.rel.ri[j] + rb, 0, 2 * rb) : 2 * rb + 1;
    const int b1 = sameChain && sameRes ? clamp(a.rel.ti[i] - a.rel.ti[j] + rb, 0, 2 * rb) : 2 * rb + 1;
    const int b3 = sameChain ? 2 * cb + 1 : clamp(a.rel.sym[i] - a.rel.sym[j] + cb, 0, 2 * cb);
    device const float4* zr = (device const float4*)(a.z + (ulong)p * 256);
    device const float4* w0 = (device const float4*)(a.rel.Wt + (uint)b0 * 256);
    device const float4* w1 = (device const float4*)(a.rel.Wt + (uint)(w + b1) * 256);
    device const float4* w3 = (device const float4*)(a.rel.Wt + (uint)(2 * w + 1 + b3) * 256);
    device const float4* we = sameEnt ? (device const float4*)(a.rel.Wt + (uint)(2 * w) * 256) : nullptr;
    float4 r[4];
    r[0] = zr[lane];
    r[1] = zr[lane + 32];
    r[2] = w0[lane] + w1[lane] + w3[lane] + (we ? we[lane] : float4(0.f));
    r[3] = w0[lane + 32] + w1[lane + 32] + w3[lane + 32] + (we ? we[lane + 32] : float4(0.f));
    float s = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 4; ++k) s += r[k].x + r[k].y + r[k].z + r[k].w;
    const float mean = simd_sum(s) * (1.f / 512.f);
    float v = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 4; ++k) { const float4 d = r[k] - mean; v += dot(d, d); }
    const float inv = rsqrt(simd_sum(v) * (1.f / 512.f) + 1e-5f);
    device const float4* sc4 = (device const float4*)a.scale;
    device const float4* of4 = (device const float4*)a.offset;
    device half4* yh4 = (device half4*)(a.outH + (ulong)row * 512);
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 4; ++k) {
      const uint idx = lane + 32 * k;
      yh4[idx] = half4((r[k] - mean) * inv * sc4[idx] + of4[idx]);
    }
    return;
  }
  ulong t = LF_INDEX, C2_4 = a.C >> 1;
  if (t >= a.n * C2_4) return;
  uint pr = lf_udiv((uint)t, (uint)C2_4), c4 = (uint)t - pr * (uint)C2_4, c = c4 << 2;
  uint p = (uint)a.p0 + pr;
  if (c < a.C) ((device float4*)a.out)[t] = *(device const float4*)(a.z + (ulong)p * a.C + c);
  else { uint i = lf_udiv(p, a.T); ((device float4*)a.out)[t] = relPosAt4(a.rel, i, p - i * a.T, c - a.C, a.C); }
}
kernel void ef2_silu_mul(LF_ARGS(SiluMulArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.a[t] = lf_silu(a.a[t]) * a.b[t];
}
kernel void ef2_fourier(LF_ARGS(FourierArgs)) {
  uint i = (uint)LF_INDEX, K = max(a.pad, 1u);
  if (i >= K * a.n) return;
  uint s = K > 1 ? lf_udiv(i, a.n) : 0, f = i - s * a.n;
  a.out[i] = precise::cos(2.f * M_PI_F * (a.level[s * 4] * a.w[f] + a.b[f]));
}
kernel void ef2_add_row(LF_ARGS(AddRowArgs)) {
  ulong t = LF_INDEX, C4 = a.C >> 2;
  if (t >= a.rows * C4) return;
  uint r = lf_udiv((uint)t, (uint)C4), c4 = (uint)t - r * (uint)C4;
  uint s = a.pad > 0 ? lf_udiv(r, a.pad) : 0, tok = a.pad > 0 ? r - s * a.pad : r;
  ((device float4*)a.x)[t] = ((device const float4*)a.base)[(ulong)tok * C4 + c4] + ((device const float4*)a.row)[(ulong)s * C4 + c4];
}
kernel void ef2_coords_input(LF_ARGS(CoordsInputArgs)) {     // [A, 6] = [x / denom | 0] (or fused with coordsLinear + c0 when wH is set)
  uint t = (uint)LF_INDEX;
  if (a.wH) {
    uint C4 = a.C >> 2;
    if (t >= a.A * C4) return;
    uint at = a.C == 128 ? (t >> 5) : lf_udiv(t, C4), c4 = a.C == 128 ? (t & 31) : (t - at * C4);
    float s = a.level[1];
    float r0 = (float)(half)(a.x[at * 3 + 0] * s);
    float r1 = (float)(half)(a.x[at * 3 + 1] * s);
    float r2 = (float)(half)(a.x[at * 3 + 2] * s);
    float4 w0 = float4(((device const half4*)(a.wH + 0 * a.C))[c4]);
    float4 w1 = float4(((device const half4*)(a.wH + 1 * a.C))[c4]);
    float4 w2 = float4(((device const half4*)(a.wH + 2 * a.C))[c4]);
    float4 qv = ((device const float4*)a.c0)[t] + r0 * w0 + r1 * w1 + r2 * w2;
    ((device float4*)a.out)[t] = qv;
    if (a.xmh && a.C == 128) {
      float inv = rsqrt(simd_sum(dot(qv, qv)) * (1.f / 128.f) + RMS_EPS);
      device const float* m = a.mod + (ulong)at * 6 * 128;
      float4 sc = ((device const float4*)(m + 128))[c4];
      float4 sh = ((device const float4*)m)[c4];
      ((device half4*)(a.xmh + (ulong)at * 128))[c4] = half4(qv * inv * (1.f + sc) + sh);
    }
    return;
  }
  if (t >= a.A * 6) return;
  uint at = t / 6, k = t - at * 6;
  float v = k < 3 ? a.x[at * 3 + k] * a.level[1] : 0.f;
  if (a.outH) a.outH[t] = (half)v;
  else a.out[t] = v;
}
// adaLN: sigmoid(g + gb) * LN(a) + sh, the projections rows ld apart; a simdgroup a row
kernel void ef2_ada_ln(constant AdaCombineArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                       uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint r = (tg.y * ng.x + tg.x) * sgs + sg;
  if (r >= a.T) return;
  device float* xr = a.an + (ulong)r * a.C;
  device const float* ry = a.resY ? a.resY + (ulong)r * a.C : nullptr;
  device const float* rg = a.resGate ? a.resGate + (ulong)r * a.ld : nullptr;
  if (a.C == 768 && (a.ld & 3) == 0) {
    device float4* x4 = (device float4*)xr;
    float4 xv[6];
    float s = 0;
    if (ry) {
      device const float4* ry4 = (device const float4*)ry;
      device const float4* rg4 = (device const float4*)rg;
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) {
        const uint idx = lane + 32 * k;
        const float4 g4 = rg4[idx];
        const float4 sig = float4(lf_sigmoid(g4.x), lf_sigmoid(g4.y), lf_sigmoid(g4.z), lf_sigmoid(g4.w));
        const float4 x = x4[idx] + ry4[idx] * sig;
        x4[idx] = x; xv[k] = x; s += x.x + x.y + x.z + x.w;
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) {
        const float4 x = x4[lane + 32 * k];
        xv[k] = x; s += x.x + x.y + x.z + x.w;
      }
    }
    const float mean = simd_sum(s) * (1.f / 768.f);
    float v = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 6; ++k) { const float4 d = xv[k] - mean; v += dot(d, d); }
    const float inv = rsqrt(simd_sum(v) * (1.f / 768.f) + 1e-5f);
    device const float4* gr4 = (device const float4*)(a.g + (ulong)r * a.ld);
    device const float4* gb4 = (device const float4*)a.gb;
    device const float4* shr4 = (device const float4*)(a.sh + (ulong)r * a.ld);
    device half4* yr4 = (device half4*)(a.out + (ulong)r * 768);
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 6; ++k) {
      const uint idx = lane + 32 * k;
      const float4 u = gr4[idx] + gb4[idx];
      const float4 sig = float4(lf_sigmoid(u.x), lf_sigmoid(u.y), lf_sigmoid(u.z), lf_sigmoid(u.w));
      yr4[idx] = half4(((xv[k] - mean) * inv) * sig + shr4[idx]);
    }
    return;
  }
  float s = 0;
  for (uint c = lane; c < a.C; c += 32) {
    float x = xr[c];
    if (ry) { x += ry[c] * lf_sigmoid(rg[c]); xr[c] = x; }
    s += x;
  }
  float mean = simd_sum(s) / a.C, v = 0;
  for (uint c = lane; c < a.C; c += 32) { float d = xr[c] - mean; v += d * d; }
  float inv = rsqrt(simd_sum(v) / a.C + 1e-5f);
  device const float* gr = a.g + (ulong)r * a.ld;
  device const float* shr = a.sh + (ulong)r * a.ld;
  device half* yr = a.out + (ulong)r * a.C;
  for (uint c = lane; c < a.C; c += 32)
    yr[c] = (half)(((xr[c] - mean) * inv) * lf_sigmoid(gr[c] + a.gb[c]) + shr[c]);
}
kernel void ef2_ada_combine(LF_ARGS(AdaCombineArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.T * a.C) return;
  uint r = lf_udiv(t, a.C), c = t - r * a.C, gi = r * a.ld + c;
  a.out[t] = (half)(a.an[t] * lf_sigmoid(a.g[gi] + a.gb[c]) + a.sh[gi]);
}
kernel void ef2_sigmoid_mul(LF_ARGS(SigmoidMulArgs)) {     // dst += x * sigmoid(g (+ gb)), or x *= sigmoid(g (+ gb))
  uint t = (uint)LF_INDEX;
  if (t >= a.T * a.C) return;
  uint r = lf_udiv(t, a.C), c = t - r * a.C;
  float v = a.x[t] * lf_sigmoid(a.g[r * a.ld + c] + (a.gb ? a.gb[c] : 0.f));
  if (a.dst) a.dst[t] += v; else a.x[t] = v;
}
kernel void ef2_scale_copies(LF_ARGS(ScaleCopiesArgs)) {    // out[j] = x * scale[j], [n, rows, C] (or LN(x) * scale[j] + offset[j] when pad == 1)
  if (a.pad == 1) {
    const uint row = (uint)(LF_INDEX >> 5), lane = _tid & 31;
    if (row >= a.rows) return;
    if (a.C == 256) {
      device const float4* xr = (device const float4*)(a.x + (ulong)row * 256);
      float4 r0 = xr[lane], r1 = xr[lane + 32];
      const float s = r0.x + r0.y + r0.z + r0.w + r1.x + r1.y + r1.z + r1.w;
      const float mean = simd_sum(s) * (1.f / 256.f);
      const float4 d0 = r0 - mean, d1 = r1 - mean;
      const float inv = rsqrt(simd_sum(dot(d0, d0) + dot(d1, d1)) * (1.f / 256.f) + 1e-5f);
      const float4 n0 = d0 * inv, n1 = d1 * inv;
      for (uint j = 0; j < a.n; ++j) {
        device const float4* sc4 = (device const float4*)(a.scales + (ulong)j * 256);
        device const float4* of4 = a.offsets ? (device const float4*)(a.offsets + (ulong)j * 256) : nullptr;
        device half4* yh4 = (device half4*)(a.outH + ((ulong)j * a.rows + row) * 256);
        yh4[lane] = half4(n0 * sc4[lane] + (of4 ? of4[lane] : float4(0.f)));
        yh4[lane + 32] = half4(n1 * sc4[lane + 32] + (of4 ? of4[lane + 32] : float4(0.f)));
      }
      return;
    }
      if (a.C == 768) {
      device const float4* xr = (device const float4*)(a.x + (ulong)row * 768);
      float4 xv[6]; float s = 0;
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) { const float4 x = xr[lane + 32 * k]; xv[k] = x; s += x.x + x.y + x.z + x.w; }
      if (a.offsets) {
        device half4* raw4 = (device half4*)(a.outH + ((ulong)(a.n + 1) * a.rows + row) * 768);
        _Pragma("clang loop unroll(full)")
        for (int k = 0; k < 6; ++k) { const uint idx = lane + 32 * k; raw4[idx] = half4(xv[k]); }
      }
      const float mean = simd_sum(s) * (1.f / 768.f);
      float v = 0;
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) { const float4 d = xv[k] - mean; v += dot(d, d); }
      const float inv = rsqrt(simd_sum(v) * (1.f / 768.f) + 1e-5f);
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) xv[k] = (xv[k] - mean) * inv;
      for (uint j = 0; j < a.n; ++j) {
        device const float4* sc4 = (device const float4*)(a.scales + (ulong)j * 768);
        device half4* yh4 = (device half4*)(a.outH + ((ulong)j * a.rows + row) * 768);
        _Pragma("clang loop unroll(full)")
        for (int k = 0; k < 6; ++k) { const uint idx = lane + 32 * k; yh4[idx] = half4(xv[k] * sc4[idx]); }
      }
      if (a.offsets) {
        device const float4* sc4 = (device const float4*)(a.scales + (ulong)a.n * 768);
        device const float4* of4 = (device const float4*)a.offsets;
        device half4* yh4 = (device half4*)(a.outH + ((ulong)a.n * a.rows + row) * 768);
        _Pragma("clang loop unroll(full)")
        for (int k = 0; k < 6; ++k) { const uint idx = lane + 32 * k; yh4[idx] = half4(xv[k] * sc4[idx] + of4[idx]); }
      }
      return;
    }
  }
  uint t = (uint)LF_INDEX, C4 = a.C >> 2, per4 = a.rows * C4;
  if (t >= a.n * per4) return;
  uint j = lf_udiv(t, per4), r4 = t - j * per4, c4 = r4 - lf_udiv(r4, C4) * C4;
  float4 v = ((device const float4*)a.x)[r4] * ((device const float4*)a.scales)[j * C4 + c4];
  if (a.outH) ((device half4*)a.outH)[t] = half4(v);
  else ((device float4*)a.out)[t] = v;
}
// scores [H, T, T] + bias [H, T, T] (half), softmax per row; a threadgroup a row
kernel void ef2_bias_softmax(constant BiasSoftmaxArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                             uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint row = tg.y * ng.x + tg.x;
  device float* s = a.S + (ulong)row * a.T;
  device const half* b = a.bias + (ulong)row * a.T;
  threadgroup float red[8];
  float m = -INFINITY;
  for (uint j = tid; j < a.T; j += 256) { float v = s[j] * a.scale + (float)b[j]; s[j] = v; m = max(m, v); }
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0]; for (int k = 1; k < 8; ++k) m = max(m, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0;
  for (uint j = tid; j < a.T; j += 256) { float e = exp(s[j] - m); s[j] = e; sum += e; }
  sum = simd_sum(sum);
  if (lane == 0) red[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0; for (int k = 0; k < 8; ++k) tot += red[k];
  float inv = 1.f / tot;
  for (uint j = tid; j < a.T; j += 256) s[j] *= inv;
}
// [blocks, T * T, H] -> the attention's bias [blocks, H, T, stride] in log2 units (the stride padded to 64 with -inf)
kernel void ef2_pair_to_heads(LF_ARGS(PairToHeadsArgs)) {
  ulong t = LF_INDEX;
  const uint B = max(a.blocks, 1u);
  if (a.H == 16) {
    uint H4 = 4, per = a.T * a.stride * H4;
    if (t >= (ulong)B * per) return;
    uint b = B > 1 ? lf_udiv((uint)t, per) : 0, rem = (uint)t - b * per;
    uint pos = rem >> 2, h0 = (rem & 3) << 2, i = lf_udiv(pos, a.stride), j = pos - i * a.stride;
    float4 v = j < a.T ? ((device const float4*)a.pb)[((ulong)b * a.P + (ulong)i * a.T + j) * H4 + (rem & 3)] * M_LOG2E_F : float4(-INFINITY);
    ulong base = (ulong)i * a.stride + j, sH = (ulong)a.T * a.stride, bOff = (ulong)b * a.H * sH;
    a.out[bOff + (ulong)(h0 + 0) * sH + base] = (half)v.x;
    a.out[bOff + (ulong)(h0 + 1) * sH + base] = (half)v.y;
    a.out[bOff + (ulong)(h0 + 2) * sH + base] = (half)v.z;
    a.out[bOff + (ulong)(h0 + 3) * sH + base] = (half)v.w;
    return;
  }
  uint per = a.T * a.stride * a.H;
  if (t >= (ulong)B * per) return;
  uint b = B > 1 ? lf_udiv((uint)t, per) : 0, rem = (uint)t - b * per;
  uint pos = lf_udiv(rem, a.H), h = rem - pos * a.H, i = lf_udiv(pos, a.stride), j = pos - i * a.stride;
  a.out[(((ulong)b * a.H + h) * a.T + i) * a.stride + j] =
      j < a.T ? (half)(a.pb[((ulong)b * a.P + (ulong)i * a.T + j) * a.H + h] * M_LOG2E_F) : (half)-INFINITY;
}
kernel void ef2_gather_tokens(LF_ARGS(GatherTokensArgs)) {
  uint t = (uint)LF_INDEX, C4 = a.C >> 2;
  if (t >= a.A * C4) return;
  uint at = a.C == 128 ? (t >> 5) : lf_udiv(t, C4), c4 = a.C == 128 ? (t & 31) : (t - at * C4);
  int token = a.mask[at] != 0.f ? a.atomToToken[at] : 0;
  float4 qv = ((device const float4*)a.q)[t] + ((device const float4*)(a.perToken + (uint)token * a.C))[c4];
  ((device float4*)a.q)[t] = qv;
  if (a.xmh && a.C == 128) {
    float inv = rsqrt(simd_sum(dot(qv, qv)) * (1.f / 128.f) + RMS_EPS);
    device const float* m = a.mod + (ulong)at * 6 * 128;
    float4 sc = ((device const float4*)(m + 128))[c4];
    float4 sh = ((device const float4*)m)[c4];
    ((device half4*)(a.xmh + (ulong)at * 128))[c4] = half4(qv * inv * (1.f + sc) + sh);
  }
}
kernel void ef2_edm_combine(LF_ARGS(EdmArgs)) {
  uint t = (uint)LF_INDEX;
  if (t < a.n) a.out[t] = a.level[2] * a.xNoisy[t] + a.level[3] * a.r[t];
}
kernel void ef2_decoder_out(constant DecoderOutArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  uint at = (tg.y * ng.x + tg.x) * 8 + sg;
  if (at >= a.A || a.C != 128) return;
  const float4 qv = ((device const float4*)(a.q + (ulong)at * 128))[lane];
  const float mean = simd_sum(qv.x + qv.y + qv.z + qv.w) * (1.f / 128.f);
  const float4 d = qv - mean;
  const float inv = rsqrt(simd_sum(dot(d, d)) * (1.f / 128.f) + 1e-5f);
  const float4 yn = d * inv * ((device const float4*)a.scale)[lane] + ((device const float4*)a.offset)[lane];
  const float4 qn = a.halfGemm ? float4(half4(yn)) : yn;
  const uint c0 = lane << 2;
  float p0 = 0, p1 = 0, p2 = 0;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < 4; ++k) {
    device const half* wk = a.wH + (c0 + k) * 3;
    p0 += qn[k] * (float)wk[0];
    p1 += qn[k] * (float)wk[1];
    p2 += qn[k] * (float)wk[2];
  }
  const float r0 = simd_sum(p0), r1 = simd_sum(p1), r2 = simd_sum(p2);
  if (lane == 0) {
    const float cSkip = a.level[2], cOut = a.level[3];
    const uint xoff = at * 3;
    a.out[xoff + 0] = cSkip * a.xNoisy[xoff + 0] + cOut * r0;
    a.out[xoff + 1] = cSkip * a.xNoisy[xoff + 1] + cOut * r1;
    a.out[xoff + 2] = cSkip * a.xNoisy[xoff + 2] + cOut * r2;
  }
}

// ---------------------------------------------------------------- the confidence head
kernel void ef2_conf_z(constant ConfZArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                       uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (a.scale && a.C == 256) {
    uint ij = (tg.y * ng.x + tg.x) * 8 + sg;
    if (ij >= a.T * a.T) return;
    uint i = lf_udiv(ij, a.T), j = ij - i * a.T;
    float b = a.bonds[ij];
    device float4* zr = (device float4*)(a.z + (ulong)ij * 256);
    float4 z0 = zr[lane], z1 = zr[lane + 32];
    float mean = simd_sum(z0.x + z0.y + z0.z + z0.w + z1.x + z1.y + z1.z + z1.w) * (1.f / 256.f);
    float4 d0 = z0 - mean, d1 = z1 - mean;
    float inv = rsqrt(simd_sum(dot(d0, d0) + dot(d1, d1)) * (1.f / 256.f) + 1e-5f);
    device const float4* sc = (device const float4*)a.scale;
    device const float4* of = (device const float4*)a.offset;
    uint c0 = lane * 4, c1 = (lane + 32) * 4;
    zr[lane] = d0 * inv * sc[lane] + of[lane] + relPosAt4(a.rel, i, j, c0, 256) +
               b * *(device const float4*)(a.wBond + c0) +
               *(device const float4*)(a.rows + i * 256 + c0) + *(device const float4*)(a.cols + j * 256 + c0);
    zr[lane + 32] = d1 * inv * sc[lane + 32] + of[lane + 32] + relPosAt4(a.rel, i, j, c1, 256) +
                    b * *(device const float4*)(a.wBond + c1) +
                    *(device const float4*)(a.rows + i * 256 + c1) + *(device const float4*)(a.cols + j * 256 + c1);
    return;
  }
  uint ij = tg.y * ng.x + tg.x;
  if (ij >= a.T * a.T) return;
  uint i = lf_udiv(ij, a.T), j = ij - i * a.T;
  float b = a.bonds[ij];
  for (uint c = tid * 4; c < a.C; c += 1024)
    *(device float4*)(a.z + (ulong)ij * a.C + c) += relPosAt4(a.rel, i, j, c, a.C) + b * *(device const float4*)(a.wBond + c) +
                                                    *(device const float4*)(a.rows + i * a.C + c) + *(device const float4*)(a.cols + j * a.C + c);
}
kernel void ef2_outer(LF_ARGS(OuterArgs)) {      // a_i * b_j, pair rows p0..
  ulong t = LF_INDEX, C4 = a.C >> 2;
  if (t >= a.n * C4) return;
  uint pr = lf_udiv((uint)t, (uint)C4), c4 = (uint)t - pr * (uint)C4;
  uint ij = (uint)a.p0 + pr, i = lf_udiv(ij, a.T), j = ij - i * a.T;
  ((device half4*)a.out)[t] = half4(((device const float4*)(a.a + i * a.C))[c4] * ((device const float4*)(a.b + j * a.C))[c4]);
}
// a simdgroup a pair: its distance and bucket once, then the row four channels a lane
kernel void ef2_distance_embed(constant DistEmbedArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                               uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  uint ij = (tg.y * ng.x + tg.x) * 8 + sg;
  if (ij >= a.T * a.T) return;
  uint i = lf_udiv(ij, a.T), j = ij - i * a.T;
  device const float* p = a.x + (uint)a.rep[i] * 3; device const float* q = a.x + (uint)a.rep[j] * 3;
  float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
  float d = sqrt(dx * dx + dy * dy + dz * dz);
  int bucket = 0;
  for (uint e0 = 0; e0 < a.nEdges; e0 += 32) bucket += simd_sum((e0 + lane < a.nEdges && d > a.edges[e0 + lane]) ? 1 : 0);
  device float4* row = (device float4*)(a.z + (ulong)ij * a.C);
  device const float4* tb = (device const float4*)(a.table + (ulong)bucket * a.C);
  if ((a.zh || a.lnOut) && a.C == 256) {
    float4 v0 = row[lane] + tb[lane], v1 = row[lane + 32] + tb[lane + 32];
    row[lane] = v0; row[lane + 32] = v1;
    float4 f0 = v0, f1 = v1;
    if (a.zh) {
      half4 h0 = half4(v0), h1 = half4(v1);
      device half4* rh = (device half4*)(a.zh + (ulong)ij * 256);
      rh[lane] = h0; rh[lane + 32] = h1;
      f0 = float4(h0); f1 = float4(h1);
    }
    if (a.lnOut) {
      float mean = simd_sum(f0.x + f0.y + f0.z + f0.w + f1.x + f1.y + f1.z + f1.w) * (1.f / 256.f);
      float4 d0 = f0 - mean, d1 = f1 - mean;
      float inv = rsqrt(simd_sum(dot(d0, d0) + dot(d1, d1)) * (1.f / 256.f) + 1e-5f);
      device const float4* sc = (device const float4*)a.lnScale;
      device const float4* of = (device const float4*)a.lnOffset;
      device half4* lo = (device half4*)(a.lnOut + (ulong)ij * 256);
      lo[lane] = half4(d0 * inv * sc[lane] + of[lane]);
      lo[lane + 32] = half4(d1 * inv * sc[lane + 32] + of[lane + 32]);
    }
    return;
  }
  for (uint c = lane; c < a.C / 4; c += 32) row[c] += tb[c];
}
kernel void ef2_add_half(LF_ARGS(AddHalfArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n4) ((device float4*)a.z)[t] += float4(((device const half4*)a.h)[t]);
}
// row-attention pooling: pooled[i] = sum_j softmax_j(score[i, j]) z[i, j]; a threadgroup a row
kernel void ef2_row_pool(constant RowPoolArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  uint i = tg.x;
  threadgroup float red[8];
  threadgroup float sc_tg[1024];
  bool useTg = a.w && a.C == 256 && a.T <= 1024;
  device float* sc = useTg ? nullptr : a.score + (ulong)i * a.T;
  float mx = -INFINITY;
  if (a.w && a.C == 256) {
    device const float4* w4 = (device const float4*)a.w;
    float4 w0 = w4[lane], w1 = w4[lane + 32];
    for (uint j = sg; j < a.T; j += 8) {
      device const float4* zr = (device const float4*)(a.z + ((ulong)i * a.T + j) * 256);
      float v = simd_sum(dot(zr[lane], w0) + dot(zr[lane + 32], w1));
      if (lane == 0) { if (useTg) sc_tg[j] = v; else sc[j] = v; }
      mx = max(mx, v);
    }
    if (lane == 0) red[sg] = mx;
  } else {
    for (uint j = tid; j < a.T; j += 256) mx = max(mx, sc[j]);
    mx = simd_max(mx);
    if (lane == 0) red[sg] = mx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = red[0]; for (int k = 1; k < 8; ++k) mx = max(mx, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0;
  for (uint j = tid; j < a.T; j += 256) {
    float v = useTg ? sc_tg[j] : sc[j];
    float e = exp(v - mx);
    if (useTg) sc_tg[j] = e; else sc[j] = e;
    sum += e;
  }
  sum = simd_sum(sum);
  if (lane == 0) red[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0; for (int k = 0; k < 8; ++k) tot += red[k];
  for (uint j = tid; j < a.T; j += 256) { if (useTg) sc_tg[j] /= tot; else sc[j] /= tot; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = tid; c < a.C; c += 256) {
    float acc = 0;
    for (uint j = 0; j < a.T; ++j) acc += (useTg ? sc_tg[j] : sc[j]) * a.z[((ulong)i * a.T + j) * a.C + c];
    a.pooled[(ulong)i * a.C + c] = acc;
  }
}
kernel void ef2_gather_atoms(LF_ARGS(GatherAtomsArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.A * a.C) return;
  uint at = lf_udiv(t, a.C), c = t - at * a.C;
  a.out[t] = a.tok[(uint)a.atomToToken[at] * a.C + c];
}
// per atom: logits over bins from its slot's table, then the expectation over [0, 1]
kernel void ef2_plddt_atom(constant PlddtArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  uint at = tg.x;
  device const float* s = a.s + (ulong)(uint)a.atomToToken[at] * a.C;
  device const float* tb = a.table + (ulong)a.slot[at] * a.C * a.bins;
  if (a.bins <= 64) {
    threadgroup float red[4];
    float lg = -INFINITY, mx = -INFINITY;
    if (tid < a.bins) {
      float acc = 0;
      for (uint c = 0; c < a.C; ++c) acc += s[c] * tb[(ulong)c * a.bins + tid];
      lg = acc; mx = acc;
    }
    mx = simd_max(mx);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = max(red[0], red[1]);
    float w = tid < a.bins ? exp(lg - mx) : 0.f;
    float e = w * (float(tid) + 0.5f) / float(a.bins);
    w = simd_sum(w); e = simd_sum(e);
    if (lane == 0) { red[sg] = w; red[sg + 2] = e; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) a.plddt[at] = (red[2] + red[3]) / (red[0] + red[1]);
    return;
  }
  threadgroup float logit[64];
  for (uint b = tid; b < a.bins; b += 64) {
    float acc = 0;
    for (uint c = 0; c < a.C; ++c) acc += s[c] * tb[(ulong)c * a.bins + b];
    logit[b] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float mx = -INFINITY; for (uint b = 0; b < a.bins; ++b) mx = max(mx, logit[b]);
    float tot = 0, e = 0;
    for (uint b = 0; b < a.bins; ++b) { float w = exp(logit[b] - mx); tot += w; e += w * (b + 0.5f) / a.bins; }
    a.plddt[at] = e / tot;
  }
}
// PAE and the pTM term from the logits, a thread a pair
kernel void ef2_pae(LF_ARGS(PaeArgs)) {
  ulong ij = LF_INDEX;
  if (ij >= a.P) return;
  device const float* lg = a.logits + ij * a.bins;
  float mx = -INFINITY;
  for (uint b = 0; b < a.bins; ++b) mx = max(mx, lg[b]);
  float tot = 0, mean = 0, t = 0;
  for (uint b = 0; b < a.bins; ++b) {
    float w = exp(lg[b] - mx), centre = a.width * (b + 0.5f), r = centre / a.d0;
    tot += w; mean += w * centre; t += w / (1 + r * r);
  }
  a.pae[ij] = mean / tot; a.tm[ij] = t / tot;
}
// a threadgroup a row: its mean tm over every column and over the other chains' columns
kernel void ef2_tm_rows(constant TmRowsArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  uint i = tg.x;
  threadgroup float red[3][8];
  float sum = 0, isum = 0, icnt = 0;
  for (uint j = tid; j < a.T; j += 256) {
    float v = a.tm[(ulong)i * a.T + j]; sum += v;
    if (a.asym[i] != a.asym[j]) { isum += v; icnt += 1; }
  }
  sum = simd_sum(sum); isum = simd_sum(isum); icnt = simd_sum(icnt);
  if (lane == 0) { red[0][sg] = sum; red[1][sg] = isum; red[2][sg] = icnt; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float s = 0, is = 0, ic = 0;
    for (int k = 0; k < 8; ++k) { s += red[0][k]; is += red[1][k]; ic += red[2][k]; }
    a.rows[2 * i] = s / (a.T + 1e-8f); a.rows[2 * i + 1] = is / (ic + 1e-8f);
  }
}
