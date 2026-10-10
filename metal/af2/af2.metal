// AlphaFold 2's kernels (metal/af2): the embedder's features, the Evoformer's layouts and its global column attention,
// the template features, and the structure module's frames and side chains. cuda/af2/src is the reading.

// ---------------------------------------------------------------- the embedder
kernel void af2_target_feat(LF_ARGS(TargetFeatArgs)) {      // one-hot 20 (X as V), a zero 21st
  uint t = (uint)LF_INDEX;
  if (t >= a.L * 21) return;
  uint i = t / 21, c = t - i * 21;
  int aa = clamp(a.aatype[i], 0, 19);
  a.tf[t] = (int)c == aa ? 1.f : 0.f;
}
kernel void af2_broadcast_rows(LF_ARGS(BroadcastRowsArgs)) {   // msa[s, i] += row[i]
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  uint r = lf_udiv((uint)t, a.C), c = (uint)t - r * a.C, i = r - lf_udiv(r, a.L) * a.L;
  a.msa[t] += a.row[i * a.C + c];
}
kernel void af2_outer_sum(LF_ARGS(OuterSumArgs)) {      // pair = left[i] + right[j]
  ulong t = LF_INDEX;
  if (t >= (ulong)a.L * a.L * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.L), j = ij - i * a.L;
  a.pair[t] = a.left[i * a.C + c] + a.right[j * a.C + c];
}
// pseudo-beta (CB; CA for glycine) from atom37 positions, then the 15-bin distogram AF2 recycles
kernel void af2_prev_dgram(LF_ARGS(PrevDgramArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.L) return;
  uint i = lf_udiv(t, a.L), j = t - i * a.L;
  int ai = a.aatype[i] == 7 ? 1 : 3, aj = a.aatype[j] == 7 ? 1 : 3;
  float d2 = 0;
  for (int k = 0; k < 3; ++k) { float d = a.pos37[(i * 37 + ai) * 3 + k] - a.pos37[(j * 37 + aj) * 3 + k]; d2 += d * d; }
  for (int b = 0; b < 15; ++b) {
    float lo = 3.25f + (20.75f - 3.25f) * b / 14.f, hi = 3.25f + (20.75f - 3.25f) * (b + 1) / 14.f;
    float upper = b + 1 < 15 ? hi * hi : 1e8f;
    a.out[(ulong)t * 15 + b] = (d2 > lo * lo && d2 < upper) ? 1.f : 0.f;
  }
}
// relpos [L, L, 73] one-hot, as _relative_encoding builds it
kernel void af2_relpos(LF_ARGS(RelposArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.L) return;
  uint i = lf_udiv(t, a.L), j = t - i * a.L;
  device float* o = a.out + (ulong)t * 73;
  for (int c = 0; c < 73; ++c) o[c] = 0;
  int clipped = clamp(a.ri[i] - a.ri[j] + 32, 0, 64);
  o[a.asym[i] == a.asym[j] ? clipped : 65] = 1;
  bool sameEntity = a.entity[i] == a.entity[j];
  o[66] = sameEntity ? 1.f : 0.f;
  int rc = clamp(a.sym[i] - a.sym[j] + 2, 0, 4);
  o[67 + (sameEntity ? rc : 5)] = 1;
}
kernel void af2_extra_feat(LF_ARGS(ExtraFeatArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * 25) return;
  uint r = (uint)(t / 25), c = (uint)(t - (ulong)r * 25);
  a.out[t] = c < 23 ? (a.codes[r] == (int)c ? 1.f : 0.f) : c == 23 ? a.hasDel[r] : a.delVal[r];
}
kernel void af2_pair_mask(LF_ARGS(PairMaskArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.L) return;
  uint i = lf_udiv(t, a.L);
  a.out[t] = a.seqMask[i] * a.seqMask[t - i * a.L];
}

// ---------------------------------------------------------------- the Evoformer
kernel void af2_bias_layout(LF_ARGS(BiasLayoutArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.L * a.H) return;
  uint ij = lf_udiv(t, a.H), h = t - ij * a.H, i = lf_udiv(ij, a.L), j = ij - i * a.L;
  float v = a.proj[t] + (a.pairMask ? 1e9f * (a.pairMask[ij] - 1.f) : 0.f);
  if (a.transposed) { uint x = i; i = j; j = x; }
  a.out[((ulong)h * a.L + i) * a.L + j] = (half)max(v * M_LOG2E_F, -6e4f);
}
// gemmTriGate's weight from AF2's [a | b] halves: channel c's (pa ga pb gb) in blocks of 8, and their biases
kernel void af2_trigate_weight(LF_ARGS(TriGateW2Args)) {
  uint t = (uint)LF_INDEX, W = 4 * a.C;
  if (t >= a.C * W) return;
  uint r = lf_udiv(t, W), col = t - r * W, kind = (col % 32) / 8, c = 8 * (col / 32) + col % 8;
  uint side = kind == 0 || kind == 1 ? 0 : 1;         // pa ga: a's; pb gb: b's
  bool gate = kind == 1 || kind == 3;
  a.w4[t] = gate ? a.gate[r * 2 * a.C + side * a.C + c] : a.proj[r * 2 * a.C + side * a.C + c];
  if (r == 0) a.b4[col] = gate ? a.gb[side * a.C + c] : a.pb[side * a.C + c];
}
// q, k, v, gate [C][W] each -> [C][4W] (the attention's one projection)
kernel void af2_qkvg_weight(LF_ARGS(QkvgWArgs)) {
  uint t = (uint)LF_INDEX, W4 = 4 * a.W;
  if (t >= a.C * W4) return;
  uint c = lf_udiv(t, W4), col = t - c * W4, role = col / a.W, w = col - role * a.W;
  device const half* src = role == 0 ? a.q : role == 1 ? a.k : role == 2 ? a.v : a.g;
  a.out[t] = src[c * a.W + w];
}
kernel void af2_gate_bias(LF_ARGS(GateBiasArgs)) {      // [4W]: the gate's bias, zeros elsewhere
  uint t = (uint)LF_INDEX;
  if (t >= 4 * a.W) return;
  a.out[t] = t >= 3 * a.W ? a.gb[t - 3 * a.W] : 0.f;
}
kernel void af2_scale_rows_h(LF_ARGS(ScaleRowsHArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  a.x[t] = (half)((float)a.x[t] * a.mask[lf_udiv((uint)t, a.C)]);
}
// norm[i][j] = sum_s mask[s][i] mask[s][j]
kernel void af2_mask_norm(LF_ARGS(MaskNormArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.L) return;
  uint i = lf_udiv(t, a.L), j = t - i * a.L;
  float s = 0;
  for (uint q = 0; q < a.S; ++q) s += a.mask[q * a.L + i] * a.mask[q * a.L + j];
  a.norm[t] = s;
}
// Pm [(i, c)][(j, e)] -> X [(i, j)][(c, e)], eight halves a thread
kernel void af2_opm_permute(LF_ARGS(OpmPermuteArgs)) {
  ulong t = LF_INDEX;
  uint O8 = a.O / 8;
  if (t >= (ulong)a.bi * a.L * a.O * O8) return;
  // (32-bit division - the host's block keeps t under 2^32 - the 64-bit one held this to ~16 GB/s)
  uint r = lf_udiv((uint)t, O8), e8 = (uint)t - r * O8;
  uint r2 = lf_udiv(r, a.O), c = r - r2 * a.O;
  uint i = lf_udiv(r2, a.L), j = r2 - i * a.L;
  ((device uint4*)a.X)[t] = ((device const uint4*)a.Pm)[(((ulong)i * a.O + c) * ((ulong)a.L * a.O) + (ulong)j * a.O) / 8 + e8];
}
// pair[i][j] += (bias + Y) / (1e-3 + norm[i][j]), rows i0..
kernel void af2_opm_add(LF_ARGS(OpmAddArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.bi * a.L * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), f = (uint)t - ij * a.C, ii = lf_udiv(ij, a.L), j = ij - ii * a.L;
  ulong i = a.i0 + ii;
  a.pair[(i * a.L + j) * a.C + f] += (a.bias[f] + a.Y[t]) / (1e-3f + a.norm[i * a.L + j]);
}
// (the same into a half pair - the evoformer's half activations: summed in float, rounded once, saturating)
kernel void af2_opm_add_h(LF_ARGS(OpmAddHArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.bi * a.L * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), f = (uint)t - ij * a.C, ii = lf_udiv(ij, a.L), j = ij - ii * a.L;
  ulong i = a.i0 + ii;
  device half* p = a.pair + (i * a.L + j) * a.C + f;
  *p = (half)clamp((float)*p + (a.bias[f] + a.Y[t]) / (1e-3f + a.norm[i * a.L + j]), -65504.f, 65504.f);
}
// lt [s][i][c] -> [i][c][s] (scaled): the shallow outer product's left operand
kernel void af2_opm_left(LF_ARGS(OpmLeftArgs)) {
  ulong t = LF_INDEX;
  const ulong total = (ulong)a.S * a.L * a.O;
  if (t >= total) return;
  uint s, c, i;      // (32-bit division where the index fits: the 64-bit one is slow)
  if (total <= 0xffffffffull) { uint r = lf_udiv((uint)t, a.S); s = (uint)t - r * a.S; i = lf_udiv(r, a.O); c = r - i * a.O; }
  else { s = (uint)(t % a.S); ulong r = t / a.S; c = (uint)(r % a.O); i = (uint)(r / a.O); }
  a.out[t] = (half)((float)a.lt[((ulong)s * a.L + i) * a.O + c] * a.scale);
}
kernel void af2_tile_bias(LF_ARGS(TileBiasArgs)) {      // [L][C] of bias[c] scale
  uint t = (uint)LF_INDEX;
  if (t < a.L * a.C) a.out[t] = a.bias[t - lf_udiv(t, a.C) * a.C] * a.scale;
}
// MSAColumnGlobalAttention for one column i (a threadgroup): the query the masked mean over the S sequences of the
// normalised rows, per head; one key and value a sequence shared by every head (kv [S][L][2D]); an online softmax
// over the sequences, each thread its own, merged through threadgroup memory. avg [L][H][D]
kernel void af2_global_attention(constant GlobalAttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  constexpr int MAXH = 8, MAXD = 16;
  threadgroup float qavg[256];
  threadgroup float qh[MAXH * MAXD];
  threadgroup float part[256 / 32][MAXH][MAXD + 2];
  threadgroup float msum;
  const uint i = tg.x, S = a.S, L = a.L, C = a.C, H = a.H, D = a.D;
  if (tid == 0) { float m = 0; for (uint s = 0; s < S; ++s) m += a.mask[s * L + i]; msum = m; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = tid; c < C; c += 256) {
    float acc = 0;
    for (uint s = 0; s < S; ++s) acc += a.mask[s * L + i] * (float)a.xn[((ulong)s * L + i) * C + c];
    qavg[c] = acc / (msum + 1e-10f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t = tid; t < H * D; t += 256) {
    float acc = 0;
    for (uint c = 0; c < C; ++c) acc += qavg[c] * (float)a.qw[c * H * D + t];
    qh[t] = acc / sqrt((float)D);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m[MAXH], l[MAXH], acc[MAXH][MAXD];
  for (uint h = 0; h < MAXH; ++h) { m[h] = -1e30f; l[h] = 0.f; for (uint d = 0; d < MAXD; ++d) acc[h][d] = 0.f; }
  for (uint s = tid; s < S; s += 256) {
    device const float* kv = a.kv + ((ulong)s * L + i) * 2 * D;
    float bias = 1e9f * (a.mask[s * L + i] - 1.f);
    for (uint h = 0; h < H; ++h) {
      float lg = bias;
      for (uint d = 0; d < D; ++d) lg += qh[h * D + d] * kv[d];
      float mn = max(m[h], lg), cs = exp(m[h] - mn), e = exp(lg - mn);
      l[h] = l[h] * cs + e;
      for (uint d = 0; d < D; ++d) acc[h][d] = acc[h][d] * cs + e * kv[D + d];
      m[h] = mn;
    }
  }
  // merge within the simdgroup, then across the eight
  for (uint h = 0; h < H; ++h) {
    float M = simd_max(m[h]);
    float c = exp(m[h] - M);
    float L_ = simd_sum(l[h] * c);
    for (uint d = 0; d < D; ++d) acc[h][d] = simd_sum(acc[h][d] * c);
    m[h] = M; l[h] = L_;
  }
  if (tid % 32 == 0)
    for (uint h = 0; h < H; ++h) {
      part[tid / 32][h][0] = m[h]; part[tid / 32][h][1] = l[h];
      for (uint d = 0; d < D; ++d) part[tid / 32][h][2 + d] = acc[h][d];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t = tid; t < H * D; t += 256) {
    uint h = t / D, d = t - h * D;
    float M = -1e30f;
    for (int w = 0; w < 8; ++w) M = max(M, part[w][h][0]);
    float Ls = 0.f, A = 0.f;
    for (int w = 0; w < 8; ++w) { float c = exp(part[w][h][0] - M); Ls += part[w][h][1] * c; A += part[w][h][2 + d] * c; }
    a.avg[((ulong)i * H + h) * D + d] = A / Ls;
  }
}
// out[s, i, w] = avg[i, w] sigmoid(gate[s, i, w]) (half: the output projection's input)
kernel void af2_global_gate(LF_ARGS(GlobalGateArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.S * a.L * a.W) return;
  uint r = lf_udiv((uint)t, a.W), w = (uint)t - r * a.W, i = r - lf_udiv(r, a.L) * a.L;
  a.out[t] = (half)(a.avg[(ulong)i * a.W + w] * lf_sigmoid(a.gate[t]));
}

// ---------------------------------------------------------------- templates
// the multimer's template pair inputs, summed through their linears into act [L, L, 64] (a threadgroup a pair):
// 0 dgram(39) . 1 pseudo-beta mask . 2 aatype_j(22) . 3 aatype_i(22) . 4-6 unit vector . 7 backbone mask
inline void af2_frame(device const float* tPos, device const float* tMask, uint r, thread float* R, thread float* t, thread float& m) {
  device const float* p = tPos + (ulong)r * 37 * 3;
  m = tMask[r * 37] * tMask[r * 37 + 1] * tMask[r * 37 + 2];
  float e0[3], e1[3];
  for (int k = 0; k < 3; ++k) { e0[k] = p[6 + k] - p[3 + k]; e1[k] = p[k] - p[3 + k]; t[k] = p[3 + k]; }
  float n0 = rsqrt(max(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2], 1e-12f));
  for (int k = 0; k < 3; ++k) e0[k] *= n0;
  float c = e1[0] * e0[0] + e1[1] * e0[1] + e1[2] * e0[2];
  for (int k = 0; k < 3; ++k) e1[k] -= c * e0[k];
  float n1 = rsqrt(max(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2], 1e-12f));
  for (int k = 0; k < 3; ++k) e1[k] *= n1;
  float e2[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]};
  for (int k = 0; k < 3; ++k) { R[k * 3] = e0[k]; R[k * 3 + 1] = e1[k]; R[k * 3 + 2] = e2[k]; }
}
kernel void af2_template_pair(constant TmplPairArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                              uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint ij = tg.y * ng.x + tg.x;
  if (ij >= a.L * a.L) return;
  uint i = lf_udiv(ij, a.L), j = ij - i * a.L;
  threadgroup float feat[44];
  threadgroup int aij[2];
  if (tid == 0) {
    int pi = a.aatype[i] == 7 ? 1 : 3, pj = a.aatype[j] == 7 ? 1 : 3;
    float pb2d = a.mask[i * 37 + pi] * a.mask[j * 37 + pj];
    float d2 = 0;
    for (int k = 0; k < 3; ++k) { float d = a.pos[(i * 37 + pi) * 3 + k] - a.pos[(j * 37 + pj) * 3 + k]; d2 += d * d; }
    for (int b = 0; b < 39; ++b) {
      float lo = 3.25f + (50.75f - 3.25f) * b / 38.f, hi = 3.25f + (50.75f - 3.25f) * (b + 1) / 38.f;
      feat[b] = ((d2 > lo * lo && d2 < (b + 1 < 39 ? hi * hi : 1e8f)) ? 1.f : 0.f) * pb2d;
    }
    feat[39] = pb2d;
    float Ri[9], ti[3], mi, Rj[9], tj[3], mj;
    af2_frame(a.pos, a.mask, i, Ri, ti, mi); af2_frame(a.pos, a.mask, j, Rj, tj, mj);
    float d[3] = {tj[0] - ti[0], tj[1] - ti[1], tj[2] - ti[2]}, v[3];
    for (int k = 0; k < 3; ++k) v[k] = Ri[k] * d[0] + Ri[3 + k] * d[1] + Ri[6 + k] * d[2];
    float nv = rsqrt(max(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], 1e-12f)), bb = sqrt(mi * mj);
    for (int k = 0; k < 3; ++k) feat[40 + k] = v[k] * nv * bb;
    feat[43] = bb;
    aij[0] = clamp(a.aatype[i], 0, 21); aij[1] = clamp(a.aatype[j], 0, 21);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = tid; c < a.C; c += 64) {
    float acc = a.bsum[c];
    for (int b = 0; b < 39; ++b) acc += feat[b] * a.w0[b * a.C + c];
    acc += feat[39] * a.w1[c];
    acc += a.w2[aij[1] * a.C + c] + a.w3[aij[0] * a.C + c];
    acc += feat[40] * a.w4[c] + feat[41] * a.w5[c] + feat[42] * a.w6[c] + feat[43] * a.w7[c];
    a.act[(ulong)ij * a.C + c] = acc;
  }
}
// the monomer's 88 template pair inputs, all times the backbone mask, through embedding2d (88 -> 64)
kernel void af2_template_pair_monomer(constant TmplPairMonoArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                      uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint ij = tg.y * ng.x + tg.x;
  if (ij >= a.L * a.L) return;
  uint i = lf_udiv(ij, a.L), j = ij - i * a.L;
  threadgroup float feat[40];
  threadgroup int aij[2];
  threadgroup float bbs;
  if (tid == 0) {
    int pi = a.aatype[i] == 7 ? 1 : 3, pj = a.aatype[j] == 7 ? 1 : 3;
    float m2 = a.mask[i * 37 + pi] * a.mask[j * 37 + pj];
    float d2 = 0;
    for (int k = 0; k < 3; ++k) { float d = a.pos[(i * 37 + pi) * 3 + k] - a.pos[(j * 37 + pj) * 3 + k]; d2 += d * d; }
    for (int q = 0; q < 39; ++q) {
      float lo = 3.25f + (50.75f - 3.25f) * q / 38.f, hi = 3.25f + (50.75f - 3.25f) * (q + 1) / 38.f;
      feat[q] = (d2 > lo * lo && d2 < (q + 1 < 39 ? hi * hi : 1e8f)) ? 1.f : 0.f;
    }
    feat[39] = m2;
    float bi = a.mask[i * 37] * a.mask[i * 37 + 1] * a.mask[i * 37 + 2], bj = a.mask[j * 37] * a.mask[j * 37 + 1] * a.mask[j * 37 + 2];
    bbs = bi * bj;
    aij[0] = clamp(a.aatype[i], 0, 21); aij[1] = clamp(a.aatype[j], 0, 21);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = tid; c < a.C; c += 64) {
    float acc = 0;
    for (int q = 0; q < 40; ++q) acc += feat[q] * a.w[q * a.C + c];
    acc += a.w[(40 + aij[1]) * a.C + c] + a.w[(62 + aij[0]) * a.C + c];
    acc += a.w[87 * a.C + c];
    a.act[(ulong)ij * a.C + c] = acc * bbs + a.b[c];
  }
}
kernel void af2_relu_scale(LF_ARGS(ReluScaleArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.x[t] = max(a.x[t] * a.s, 0.f);
}
// the monomer's pointwise attention from the query pair over the templates (no gating, every template present)
kernel void af2_point_attention(LF_ARGS(PointAttnArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.pairs * a.H) return;
  ulong p = t / a.H; uint h = (uint)(t - p * a.H), W = a.H * a.D;
  float logits[8]; float mx = -1e30f;
  for (uint s = 0; s < a.T; ++s) {
    float acc = 0;
    for (uint d = 0; d < a.D; ++d) acc += a.q[p * W + h * a.D + d] * a.k[((ulong)s * a.pairs + p) * W + h * a.D + d];
    logits[s] = acc; mx = max(mx, acc);
  }
  float sum = 0;
  for (uint s = 0; s < a.T; ++s) { logits[s] = exp(logits[s] - mx); sum += logits[s]; }
  for (uint d = 0; d < a.D; ++d) {
    float acc = 0;
    for (uint s = 0; s < a.T; ++s) acc += logits[s] * a.v[((ulong)s * a.pairs + p) * W + h * a.D + d];
    a.out[p * W + h * a.D + d] = acc / sum;
  }
}
// the template torsions' residue tables (all_atom.get_chi_atom_indices, residue_constants' chi_angles_mask and
// chi_pi_periodic, an unknown residue's row of zeros appended)
constant int CHI_ATOM_INDICES[21 * 4 * 4] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 1, 3, 5, 11, 3, 5, 11, 23, 5, 11, 23,
    32, 0, 1, 3, 5, 1, 3, 5, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 1, 3, 5, 16, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 1, 3, 5, 11, 3, 5, 11, 26, 0, 0, 0,
    0, 0, 1, 3, 5, 1, 3, 5, 11, 3, 5, 11, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 3, 5, 1, 3, 5, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 1, 3, 6, 12, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 3, 5, 1, 3, 5, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 1, 3, 5, 11, 3, 5, 11, 19, 5, 11, 19,
    35, 0, 1, 3, 5, 1, 3, 5, 18, 3, 5, 18, 19, 0, 0, 0, 0, 0, 1, 3, 5, 1, 3, 5, 12, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 3, 5, 1, 3, 5, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 3, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 1, 3, 5, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 3, 5, 1, 3, 5, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
constant float CHI_ANGLES_MASK[21 * 4] = {
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0};
constant float CHI_PI_PERIODIC[21 * 4] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0};
// monomer: aatype(22) | torsion sin, cos(14) | alt torsion sin, cos(14) | torsion mask(7) = 57, and the row's mask (psi's)
kernel void af2_template_torsion(LF_ARGS(TorsionFeatArgs)) {
  uint i = (uint)LF_INDEX;
  if (i >= a.L) return;
  int aaRaw = a.aatype[i], aa = min(aaRaw, 20);
  device const float* p = a.pos + (ulong)i * 37 * 3; device const float* m = a.mask + (ulong)i * 37;
  float prev[9], pm[3];
  for (int k = 0; k < 9; ++k) prev[k] = i > 0 ? a.pos[(ulong)(i - 1) * 37 * 3 + k] : 0.f;     // (N, CA, C of the residue before)
  for (int k = 0; k < 3; ++k) pm[k] = i > 0 ? a.mask[(ulong)(i - 1) * 37 + k] : 0.f;
  float at[7][4][3]; float tm[7];
  for (int k = 0; k < 3; ++k) {
    at[0][0][k] = prev[3 + k]; at[0][1][k] = prev[6 + k]; at[0][2][k] = p[k]; at[0][3][k] = p[3 + k];     // pre-omega
    at[1][0][k] = prev[6 + k]; at[1][1][k] = p[k]; at[1][2][k] = p[3 + k]; at[1][3][k] = p[6 + k];        // phi
    at[2][0][k] = p[k]; at[2][1][k] = p[3 + k]; at[2][2][k] = p[6 + k]; at[2][3][k] = p[12 + k];          // psi
  }
  tm[0] = pm[1] * pm[2] * m[0] * m[1];
  tm[1] = pm[2] * m[0] * m[1] * m[2];
  tm[2] = m[0] * m[1] * m[2] * m[4];
  for (int c = 0; c < 4; ++c) {
    float mm = CHI_ANGLES_MASK[aa * 4 + c];
    for (int q = 0; q < 4; ++q) {
      int idx = CHI_ATOM_INDICES[(aa * 4 + c) * 4 + q];
      for (int k = 0; k < 3; ++k) at[3 + c][q][k] = p[idx * 3 + k];
      mm *= m[idx];
    }
    tm[3 + c] = mm;
  }
  device float* f = a.feat + (ulong)i * 57;
  for (int k = 0; k < 22; ++k) f[k] = k == clamp(aaRaw, 0, 21) ? 1.f : 0.f;
  for (int t = 0; t < 7; ++t) {
    float e0[3], e1[3], e2[3], d[3];
    for (int k = 0; k < 3; ++k) { e0[k] = at[t][2][k] - at[t][1][k]; e1[k] = at[t][0][k] - at[t][2][k]; d[k] = at[t][3][k] - at[t][2][k]; }
    float n0 = sqrt(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2] + 1e-8f); for (int k = 0; k < 3; ++k) e0[k] /= n0;
    float c = e1[0] * e0[0] + e1[1] * e0[1] + e1[2] * e0[2]; for (int k = 0; k < 3; ++k) e1[k] -= c * e0[k];
    float n1 = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2] + 1e-8f); for (int k = 0; k < 3; ++k) e1[k] /= n1;
    e2[0] = e0[1] * e1[2] - e0[2] * e1[1]; e2[1] = e0[2] * e1[0] - e0[0] * e1[2]; e2[2] = e0[0] * e1[1] - e0[1] * e1[0];
    float y = e1[0] * d[0] + e1[1] * d[1] + e1[2] * d[2], z = e2[0] * d[0] + e2[1] * d[1] + e2[2] * d[2];
    float nn = sqrt(z * z + y * y + 1e-8f);
    float sn = z / nn, cs = y / nn;
    if (t == 0 && i == 0) sn = cs = 0.f;     // (the first residue's pre-omega: defined as zero)
    if (t == 2) { sn = -sn; cs = -cs; }
    float alt = t >= 3 ? 1.f - 2.f * CHI_PI_PERIODIC[aa * 4 + t - 3] : 1.f;
    f[22 + t * 2] = sn; f[22 + t * 2 + 1] = cs;
    f[36 + t * 2] = sn * alt; f[36 + t * 2 + 1] = cs * alt;
    f[50 + t] = tm[t];
  }
  a.rowMask[i] = tm[2];
}
// multimer: aatype(22) | sin(chi) mask(4) | cos(chi) mask(4) | chi mask(4) = 34, the row's mask chi 1's
kernel void af2_template_chi(LF_ARGS(TorsionFeatArgs)) {
  uint i = (uint)LF_INDEX;
  if (i >= a.L) return;
  int aaRaw = a.aatype[i], aa = clamp(aaRaw, 0, 20);
  device const float* p = a.pos + (ulong)i * 37 * 3; device const float* m = a.mask + (ulong)i * 37;
  device float* f = a.feat + (ulong)i * 34;
  for (int k = 0; k < 22; ++k) f[k] = k == clamp(aaRaw, 0, 21) ? 1.f : 0.f;
  for (int c = 0; c < 4; ++c) {
    float x[4][3]; float mm = CHI_ANGLES_MASK[aa * 4 + c];
    for (int q = 0; q < 4; ++q) { int idx = CHI_ATOM_INDICES[(aa * 4 + c) * 4 + q]; for (int k = 0; k < 3; ++k) x[q][k] = p[idx * 3 + k]; mm *= m[idx]; }
    float3 v1 = float3(x[0][0], x[0][1], x[0][2]) - float3(x[1][0], x[1][1], x[1][2]);
    float3 v2 = float3(x[1][0], x[1][1], x[1][2]) - float3(x[2][0], x[2][1], x[2][2]);
    float3 v3 = float3(x[3][0], x[3][1], x[3][2]) - float3(x[2][0], x[2][1], x[2][2]);
    float3 c1 = cross(v1, v2), c2 = cross(v3, v2), c3 = cross(c2, c1);
    float v2m = sqrt(max(dot(v2, v2), 1e-12f));
    float ang = precise::atan2(dot(c3, v2), v2m * dot(c1, c2));
    f[22 + c] = precise::sin(ang) * mm; f[26 + c] = precise::cos(ang) * mm; f[30 + c] = mm;
    if (c == 0) a.rowMask[i] = mm;
  }
}

// ---------------------------------------------------------------- the structure module
// a rigid is 12 floats: the rotation row-major, then the translation
inline void rapply(thread const float* r, thread const float* p, thread float* o) {
  for (int k = 0; k < 3; ++k) o[k] = r[k * 3] * p[0] + r[k * 3 + 1] * p[1] + r[k * 3 + 2] * p[2] + r[9 + k];
}
inline void rapplyInv(thread const float* r, thread const float* p, thread float* o) {
  float d[3] = {p[0] - r[9], p[1] - r[10], p[2] - r[11]};
  for (int k = 0; k < 3; ++k) o[k] = r[k] * d[0] + r[3 + k] * d[1] + r[6 + k] * d[2];
}
inline void rcompose(thread const float* A, thread const float* B, thread float* C) {
  float R[9], t[3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  float b[3] = {B[9], B[10], B[11]};
  rapply(A, b, t);
  for (int k = 0; k < 9; ++k) C[k] = R[k];
  for (int k = 0; k < 3; ++k) C[9 + k] = t[k];
}
kernel void af2_identity_rigid(LF_ARGS(IdentityRigidArgs)) {
  uint i = (uint)LF_INDEX;
  if (i >= a.L) return;
  device float* x = a.r + i * 12;
  for (int k = 0; k < 12; ++k) x[k] = 0;
  x[0] = x[4] = x[8] = 1;
}
// points [L, H*3*P] (per head: x[0:P] y[P:2P] z[2P:3P]) -> global [L, H, P, 3]
kernel void af2_points_global(LF_ARGS(PointsGlobalArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.H * a.P) return;
  uint p = t % a.P, r = t / a.P, h = r % a.H, i = r / a.H;
  device const float* s = a.proj + (ulong)i * a.H * 3 * a.P + h * 3 * a.P;
  float rig[12]; for (int k = 0; k < 12; ++k) rig[k] = a.rig[i * 12 + k];
  float loc[3] = {s[p], s[a.P + p], s[2 * a.P + p]}, g[3];
  rapply(rig, loc, g);
  for (int k = 0; k < 3; ++k) a.out[(ulong)t * 3 + k] = g[k];
}
// IPA logits -> weights, a simdgroup a (query, head): [q, h, k]
kernel void af2_ipa_weights(constant IpaWeightsArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  uint wid = (tg.y * ng.x + tg.x) * 8 + sg;
  if (wid >= a.L * a.H) return;
  uint h = wid % a.H, q = wid / a.H;
  device float* row = a.attn + ((ulong)q * a.H + h) * a.L;
  float mx = -1e30f, w = a.pw[h];
  for (uint k = lane; k < a.L; k += 32) {
    float d2 = 0;
    for (uint p = 0; p < a.Pq; ++p)
      for (int x = 0; x < 3; ++x) {
        float d = a.qp[(((ulong)q * a.H + h) * a.Pq + p) * 3 + x] - a.kp[(((ulong)k * a.H + h) * a.Pq + p) * 3 + x];
        d2 += d * d;
      }
    float s = -0.5f * w * d2, sc = 0;
    for (uint c = 0; c < a.Cs; ++c) sc += a.qs[((ulong)q * a.H + h) * a.Cs + c] * a.ks[((ulong)k * a.H + h) * a.Cs + c];
    s += sc + a.b2d[((ulong)q * a.L + k) * a.H + h];
    s -= 1e5f * (1.f - a.seqMask[q] * a.seqMask[k]);
    s *= sqrt(1.f / 3.f);
    row[k] = s;
    mx = max(mx, s);
  }
  mx = simd_max(mx);
  float sum = 0;
  for (uint k = lane; k < a.L; k += 32) { float e = exp(row[k] - mx); row[k] = e; sum += e; }
  sum = simd_sum(sum);
  for (uint k = lane; k < a.L; k += 32) row[k] /= sum;
}
// IPA's outputs, a threadgroup a (query, head): [result_scalar | point local x | y | z | norms | over 2d]
kernel void af2_ipa_outputs(constant IpaOutputsArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
  uint q = tg.x / a.H, h = tg.x % a.H;
  device const float* at = a.attn + ((ulong)q * a.H + h) * a.L;
  uint Fw = a.H * a.Cs + 4 * a.H * a.Pv + a.H * a.C2;
  device float* f = a.fin + (ulong)q * Fw;
  for (uint c = tid; c < a.Cs; c += 128) {
    float s = 0;
    for (uint k = 0; k < a.L; ++k) s += at[k] * a.vs[((ulong)k * a.H + h) * a.Cs + c];
    f[h * a.Cs + c] = s;
  }
  float rig[12]; for (int k = 0; k < 12; ++k) rig[k] = a.rig[q * 12 + k];
  for (uint p = tid; p < a.Pv; p += 128) {
    float g[3] = {0, 0, 0};
    for (uint k = 0; k < a.L; ++k)
      for (int x = 0; x < 3; ++x) g[x] += at[k] * a.vp[(((ulong)k * a.H + h) * a.Pv + p) * 3 + x];
    float loc[3];
    rapplyInv(rig, g, loc);
    uint base = a.H * a.Cs, idx = h * a.Pv + p;
    f[base + idx] = loc[0];
    f[base + a.H * a.Pv + idx] = loc[1];
    f[base + 2 * a.H * a.Pv + idx] = loc[2];
    f[base + 3 * a.H * a.Pv + idx] = sqrt(max(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2], 1e-16f));
  }
  for (uint c = tid; c < a.C2; c += 128) {
    float s = 0;
    for (uint k = 0; k < a.L; ++k) s += at[k] * a.act2d[((ulong)q * a.L + k) * a.C2 + c];
    f[a.H * a.Cs + 4 * a.H * a.Pv + h * a.C2 + c] = s;
  }
}
// the backbone update: QuatRigid's (qx qy qz tx ty tz), qw = 1, normalised; rigid = rigid @ update
kernel void af2_rigid_update(LF_ARGS(RigidUpdateArgs)) {
  uint i = (uint)LF_INDEX;
  if (i >= a.L) return;
  device const float* u = a.upd + i * 6;
  float w = 1, x = u[0], y = u[1], z = u[2];
  float inv = rsqrt(max(1e-6f, w * w + x * x + y * y + z * z));
  w *= inv; x *= inv; y *= inv; z *= inv;
  float U[12] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), u[3], u[4], u[5]};
  float R[12], C[12];
  for (int k = 0; k < 12; ++k) R[k] = a.rig[i * 12 + k];
  rcompose(R, U, C);
  for (int k = 0; k < 12; ++k) a.rig[i * 12 + k] = C[k];
}
kernel void af2_relu_copy(LF_ARGS(ReluCopyArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.y[t] = max(a.x[t], 0.f);
}
kernel void af2_softplus_scale(LF_ARGS(SoftplusArgs)) {
  uint h = (uint)LF_INDEX;
  if (h < a.H) a.out[h] = a.base * log(1.f + exp(a.raw[h]));
}
// angles (unnormalised [L, 7, 2]) -> frames -> atom14 -> atom37, a thread a residue
kernel void af2_sidechains(LF_ARGS(SidechainArgs)) {
  uint i = (uint)LF_INDEX;
  if (i >= a.L) return;
  int aa = clamp(a.aatype[i], 0, 19);
  float bb[12];
  for (int k = 0; k < 12; ++k) bb[k] = a.rig[i * 12 + k];
  for (int k = 9; k < 12; ++k) bb[k] *= a.positionScale;
  float sinA[8], cosA[8];
  sinA[0] = 0; cosA[0] = 1;
  for (int q = 0; q < 7; ++q) {
    float s = a.unnorm[(i * 7 + q) * 2], c = a.unnorm[(i * 7 + q) * 2 + 1];
    float inv = 1.f / sqrt(max(s * s + c * c, 1e-12f));
    s *= inv; c *= inv;
    a.angles[(i * 7 + q) * 2] = s; a.angles[(i * 7 + q) * 2 + 1] = c;
    sinA[q + 1] = s; cosA[q + 1] = c;
  }
  float frames[8][12];
  for (int g = 0; g < 8; ++g) {
    device const float* m = a.defaultFrames + ((ulong)aa * 8 + g) * 16;
    float D[12] = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10], m[3], m[7], m[11]};
    float Rx[12] = {1, 0, 0, 0, cosA[g], -sinA[g], 0, sinA[g], cosA[g], 0, 0, 0};
    float R[12];
    rcompose(D, Rx, R);
    for (int k = 9; k < 12; ++k) R[k] = D[k];
    for (int k = 0; k < 12; ++k) frames[g][k] = R[k];
  }
  float tmp[12];
  rcompose(frames[4], frames[5], tmp); for (int k = 0; k < 12; ++k) frames[5][k] = tmp[k];
  rcompose(frames[5], frames[6], tmp); for (int k = 0; k < 12; ++k) frames[6][k] = tmp[k];
  rcompose(frames[6], frames[7], tmp); for (int k = 0; k < 12; ++k) frames[7][k] = tmp[k];
  float pos14[14][3];
  for (int q = 0; q < 14; ++q) {
    int g = a.atom14Group[aa * 14 + q];
    float glob[12];
    rcompose(bb, frames[g], glob);
    float lit[3] = {a.litPos[((ulong)aa * 14 + q) * 3], a.litPos[((ulong)aa * 14 + q) * 3 + 1], a.litPos[((ulong)aa * 14 + q) * 3 + 2]};
    float p[3];
    rapply(glob, lit, p);
    float m = a.atom14Mask[aa * 14 + q];
    for (int x = 0; x < 3; ++x) { pos14[q][x] = p[x] * m; a.pos14[((ulong)i * 14 + q) * 3 + x] = p[x] * m; }
  }
  for (int q = 0; q < 37; ++q) {
    int s = a.atom37To14[aa * 37 + q];
    float m = a.atom37Mask[aa * 37 + q] * a.seqMask[i];
    for (int x = 0; x < 3; ++x) a.pos37[((ulong)i * 37 + q) * 3 + x] = pos14[s][x] * m;
  }
}
// the heads: a pair's expected PAE and its pTM term (64 bins, centres 31/62 apart); P(< 8 A) from the distogram
kernel void af2_pae_tm(LF_ARGS(PaeTmArgs)) {
  ulong ij = LF_INDEX;
  if (ij >= a.pairs) return;
  device const float* l = a.logits + ij * 64;
  const float step = 31.f / 62;
  float mx = -1e30f; for (int b = 0; b < 64; ++b) mx = max(mx, l[b]);
  float s = 0, e = 0, t = 0;
  for (int b = 0; b < 64; ++b) {
    float c = b < 63 ? b * step + step / 2 : 62 * step + step / 2 + step;
    float p = exp(l[b] - mx); s += p; e += p * c; t += p / (1 + (c / a.d0) * (c / a.d0));
  }
  a.pae[ij] = e / s; a.tm[ij] = t / s;
}
kernel void af2_symmetrise(LF_ARGS(SymmetriseArgs)) {   // y[i][j] = x[i][j] + x[j][i]
  ulong t = LF_INDEX;
  if (t >= (ulong)a.L * a.L * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.L), j = ij - i * a.L;
  a.y[t] = a.x[t] + a.x[((ulong)j * a.L + i) * a.C + c];
}
kernel void af2_contact8(LF_ARGS(Contact8Args)) {
  ulong ij = LF_INDEX;
  if (ij >= a.pairs) return;
  device const float* l = a.logits + ij * 64;
  float mx = -1e30f; for (int b = 0; b < 64; ++b) mx = max(mx, l[b]);
  float s = 0, near = 0; for (int b = 0; b < 64; ++b) { float p = exp(l[b] - mx); s += p; if (b <= 18) near += p; }
  a.out[ij] = near / s;
}
