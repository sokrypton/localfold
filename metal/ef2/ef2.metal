// ESMFold2's kernels (metal/ef2): the language model's, the trunk's, the atom transformer's, the diffusion module's and
// the confidence head's. cuda/ef2/src is the reading: the same arithmetic, written for Apple's GPUs - 32-bit index
// math (lf_udiv: no integer divider), simdgroup reductions, threadgroup tiles within 32 KB.

// ---------------------------------------------------------------- relative positions (139 bins, summed rows)
inline float relPosAt(constant RelIdx& r, uint i, uint j, uint c, uint C) {
  const int rb = 32, cb = 2;
  bool sameChain = r.asym[i] == r.asym[j], sameRes = r.ri[i] == r.ri[j];
  int b0 = sameChain ? clamp(r.ri[i] - r.ri[j] + rb, 0, 2 * rb) : 2 * rb + 1;
  int b1 = sameChain && sameRes ? clamp(r.ti[i] - r.ti[j] + rb, 0, 2 * rb) : 2 * rb + 1;
  int b3 = sameChain ? 2 * cb + 1 : clamp(r.sym[i] - r.sym[j] + cb, 0, 2 * cb);
  const int w = 2 * rb + 2;
  float v = r.Wt[(uint)b0 * C + c] + r.Wt[(uint)(w + b1) * C + c] + r.Wt[(uint)(2 * w + 1 + b3) * C + c];
  if (r.ent[i] == r.ent[j]) v += r.Wt[(uint)(2 * w) * C + c];
  return v;
}

kernel void ef2_interleave8(LF_ARGS(Interleave8Args)) {
  uint t = (uint)LF_INDEX, W = 2 * a.I;
  if (t >= a.rows * W) return;
  uint r = lf_udiv(t, W), c = t - r * W, m = c >> 4, q = c & 15;
  a.out[t] = q < 8 ? a.a[r * a.lda + 8 * m + q] : a.b[r * a.lda + 8 * m + q - 8];
}

// ---------------------------------------------------------------- the language model
kernel void ef2_embed(LF_ARGS(EmbedArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.rows * a.C) return;
  uint row = lf_udiv((uint)t, a.C), c = (uint)t - row * a.C;
  a.x[t] = a.table[(ulong)a.ids[row] * a.C + c];
}
// rotary, split halves: channel d with d + 32 of each 64-wide head, position = row
kernel void ef2_rope(LF_ARGS(RopeArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.rows * a.heads * 32) return;
  uint d = t & 31, hr = t >> 5, row = lf_udiv(hr, a.heads), h = hr - row * a.heads;
  float angle = (float)row * precise::pow(10000.f, -(float)(2 * d) / 64.f);
  float s = precise::sin(angle), c = precise::cos(angle);
  device float* p = a.x + (ulong)row * a.ld + h * 64 + d;
  float x0 = p[0], x1 = p[32];
  p[0] = x0 * c - x1 * s;
  p[32] = x0 * s + x1 * c;
}
// softmax over a score row, keys of another chain (sequence id) excluded; a threadgroup a (head, query) row
kernel void ef2_softmax_seq(constant SoftmaxSeqArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint row = tg.y * ng.x + tg.x;
  uint q = row - lf_udiv(row, a.rows) * a.rows;
  device float* s = a.S + (ulong)row * a.rows;
  threadgroup float red[8];
  int sq = a.seq[q];
  float m = -INFINITY;
  for (uint j = tid; j < a.rows; j += 256) { float v = a.seq[j] == sq ? s[j] * a.scale : -INFINITY; s[j] = v; m = max(m, v); }
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0]; for (int k = 1; k < 8; ++k) m = max(m, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0;
  for (uint j = tid; j < a.rows; j += 256) { float e = exp(s[j] - m); s[j] = e; sum += e; }
  sum = simd_sum(sum);
  if (lane == 0) red[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0; for (int k = 0; k < 8; ++k) tot += red[k];
  float inv = 1.f / tot;
  for (uint j = tid; j < a.rows; j += 256) s[j] *= inv;
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
  ulong t = LF_INDEX;
  if (t >= (ulong)a.bi * a.T * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C;
  uint ii = lf_udiv(ij, a.T), j = ij - ii * a.T, i = a.i0 + ii;
  float x = a.s[i * a.C + c], y = a.s[j * a.C + c];
  a.out[(ulong)ij * 2 * a.C + c] = (half)(x * y);
  a.out[(ulong)ij * 2 * a.C + a.C + c] = (half)(x - y);
}

// ---------------------------------------------------------------- the trunk
// z_init = rows[i] + cols[j] + relpos + bonds * w_bond + lm_z: a threadgroup a pair, a thread a channel
kernel void ef2_zinit(constant ZInitArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint ij = tg.y * ng.x + tg.x;
  if (ij >= a.T * a.T) return;
  uint i = lf_udiv(ij, a.T), j = ij - i * a.T;
  for (uint c = tid; c < a.C; c += 256) {
    ulong t = (ulong)ij * a.C + c;
    a.z[t] = a.rows[i * a.C + c] + a.cols[j * a.C + c] + relPosAt(a.rel, i, j, c, a.C) + a.bonds[ij] * a.wBond[c] +
             (a.lmZ ? a.lmZ[t] : 0.f);
  }
}
kernel void ef2_trigate_weight(LF_ARGS(TriGateWArgs)) {
  uint t = (uint)LF_INDEX, W = 4 * a.C;
  if (t >= a.rows * W) return;
  uint r = lf_udiv(t, W), col = t - r * W, kind = (col % 32) / 8, c = 8 * (col / 32) + col % 8;
  uint k = kind == 1 ? 2 : kind == 2 ? 1 : kind;      // (blocks of 8: pa ga pb gb; a and b interleaved by channel)
  a.out[t] = k < 2 ? a.proj[r * 2 * a.C + 2 * c + k] : a.gate[r * 2 * a.C + 2 * c + k - 2];
}
kernel void ef2_fill(LF_ARGS(FillFArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.out[t] = a.v;
}
// the triangle's centre LayerNorm: the channel-major product [C][Lp * Lp] to pair rows [pairs][C] in half. A lane a pair,
// eight simdgroups splitting the channels, every value in registers, the two-pass statistics through a 1 KB tile, the
// rows written coalesced through a half tile (C a multiple of 8, at most 256)
kernel void ef2_center_norm(constant CenterNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[256];
  threadgroup half T[32 * (256 + 8)];
  const uint lane = tid & 31, grp = tid >> 5, C = a.C, nk = C / 8, ld = C + 8;
  const ulong r0 = (ulong)(tg.y * ng.x + tg.x) * 32;
  const uint rr = (uint)(r0 + lane), ii = lf_udiv(rr, a.L);
  const ulong q = (ulong)ii * a.Lp + (rr - ii * a.L), plane = (ulong)a.Lp * a.Lp;
  const bool live = r0 + lane < a.pairs;
  float v[32];
  float s = 0.f;
  for (uint k = 0; k < 32; ++k) if (k < nk) { v[k] = live ? a.prod[(ulong)(grp + 8 * k) * plane + q] : 0.f; s += v[k]; }
  part[grp * 32 + lane] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float S = 0.f;
  for (int g = 0; g < 8; ++g) S += part[g * 32 + lane];
  const float mean = S / C;
  float d2 = 0.f;
  for (uint k = 0; k < 32; ++k) if (k < nk) { float d = v[k] - mean; d2 += d * d; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  part[grp * 32 + lane] = d2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float V = 0.f;
  for (int g = 0; g < 8; ++g) V += part[g * 32 + lane];
  const float inv = rsqrt(V / C + 1e-5f);
  for (uint k = 0; k < 32; ++k) if (k < nk) {
    uint c = grp + 8 * k;
    T[lane * ld + c] = (half)((v[k] - mean) * inv * a.scale[c] + a.offset[c]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint row = grp; row < 32; row += 8) {
    ulong r = r0 + row;
    if (r >= a.pairs) break;
    for (uint c = lane; c < C; c += 32) a.out[r * C + c] = T[row * ld + c];
  }
}
// pair positions [p0, p0 + cnt) of z + z^T
kernel void ef2_sym_rows(LF_ARGS(SymRowsArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.cnt * a.C) return;
  uint pc = lf_udiv((uint)t, a.C), c = (uint)t - pc * a.C;
  uint ij = (uint)a.p0 + pc, i = lf_udiv(ij, a.T), j = ij - i * a.T;
  a.out[t] = a.z[(ulong)ij * a.C + c] + a.z[((ulong)j * a.T + i) * a.C + c];
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
  a.out[t] = v;
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
  if (t < a.n) a.y[t] = lf_silu(a.x[t]);
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
  device const float* xr = a.x + (ulong)row * a.C;
  float s = 0;
  for (uint c = lane; c < a.C; c += 32) s += xr[c] * xr[c];
  float inv = rsqrt(simd_sum(s) / a.C + RMS_EPS);
  device const float* m = a.mod + (ulong)row * 6 * a.C;
  for (uint c = lane; c < a.C; c += 32) a.y[(ulong)row * a.C + c] = xr[c] * inv * (1.f + m[a.scale * a.C + c]) + m[a.shift * a.C + c];
}
// q and k (heads of 32, inside the packed [A, 3C] qkv): rms per head, rotate, narrow to bfloat16; v narrowed
kernel void ef2_qkv_prepare(LF_ARGS(QkvPrepArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.A * 3 * a.heads) return;
  uint ph = lf_udiv(t, a.heads), head = t - ph * a.heads, atom = lf_udiv(ph, 3), part = ph - atom * 3;
  device float* p = a.qkv + (ulong)atom * 3 * a.C + part * a.C + head * 32;
  if (part == 2) { if (a.bf16) for (int d = 0; d < 32; ++d) p[d] = lf_bf16(p[d]); return; }
  float s = 0;
  for (int d = 0; d < 32; ++d) s += p[d] * p[d];
  float inv = rsqrt(s / 32 + RMS_EPS);
  float v[32];
  for (int d = 0; d < 32; ++d) v[d] = p[d] * inv;
  for (int i = 0; i < 16; ++i) {
    float c = a.cosT[atom * 16 + i], sn = a.sinT[atom * 16 + i], x0 = v[i], x1 = v[16 + i];
    float lo = x0 * c - x1 * sn, hi = x1 * c + x0 * sn;
    p[i] = a.bf16 ? lf_bf16(lo) : lo;
    p[16 + i] = a.bf16 ? lf_bf16(hi) : hi;
  }
}
// the windowed attention (no [heads, A, A] matrix): a valid query's keys are the valid atoms within halfWindow in rank,
// itself among them. A threadgroup takes 16 consecutive valid queries of one head (four simdgroups, four queries each)
// and walks the union of their windows in tiles of 96 keys; online softmax, a lane a key for the scores, a lane a
// channel for the output
constant int SWA_KT = 96, SWA_Q = 16;
kernel void ef2_swa(constant SwaArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float Ks[SWA_KT][33];
  threadgroup float Vs[SWA_KT][32];
  const int nValid = (int)a.nValid, hw = (int)a.halfWindow, C = (int)a.C;
  int head = tg.y;
  int r0 = tg.x * SWA_Q, r1 = min(nValid, r0 + SWA_Q) - 1;
  int klo = max(0, r0 - hw), khi = min(nValid - 1, r1 + hw);
  int rq = r0 + 4 * (int)warp;
  float qd[4][32], m[4], l[4], acc[4];
  for (int u = 0; u < 4; ++u) {
    m[u] = -INFINITY; l[u] = 0.f; acc[u] = 0.f;
    device const float* q = a.qkv + (ulong)a.valid[min(rq + u, nValid - 1)] * 3 * C + head * 32;
    for (int d = 0; d < 32; ++d) qd[u][d] = q[d] * a.scale;
  }
  for (int t0 = klo; t0 <= khi; t0 += SWA_KT) {
    int n = min(SWA_KT, khi - t0 + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < n * 8; e += 128) {
      int j = e >> 3, part = e & 7;
      device const float* row = a.qkv + (ulong)a.valid[t0 + j] * 3 * C + head * 32 + part * 4;
      float4 k = *(device const float4*)(row + C), v = *(device const float4*)(row + 2 * C);
      Ks[j][part * 4] = k.x; Ks[j][part * 4 + 1] = k.y; Ks[j][part * 4 + 2] = k.z; Ks[j][part * 4 + 3] = k.w;
      *(threadgroup float4*)&Vs[j][part * 4] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rq > r1) continue;
    int lo = max(rq - hw, t0) - t0, hi = min(min(rq + 3, r1) + hw, t0 + n - 1) - t0;
    for (int j0 = lo; j0 <= hi; j0 += 32) {
      int j = j0 + (int)lane, key = t0 + j;
      float s[4] = {0.f, 0.f, 0.f, 0.f};
      if (j <= hi) {
        for (int d = 0; d < 32; ++d) {
          float k = Ks[j][d];
          for (int u = 0; u < 4; ++u) s[u] += qd[u][d] * k;
        }
      }
      float p[4];
      for (int u = 0; u < 4; ++u) {
        bool ok = j <= hi && abs(key - (rq + u)) <= hw;
        float v = ok ? s[u] : -INFINITY;
        float mn = max(m[u], simd_max(v));
        if (mn == -INFINITY) { p[u] = 0.f; continue; }
        float corr = exp(m[u] - mn);
        p[u] = ok ? exp(v - mn) : 0.f;
        l[u] = l[u] * corr + simd_sum(p[u]); acc[u] *= corr; m[u] = mn;
      }
      int cnt = min(32, hi - j0 + 1);
      for (int k = 0; k < cnt; ++k) {
        float v = Vs[j0 + k][lane];
        for (int u = 0; u < 4; ++u) acc[u] += simd_shuffle(p[u], (ushort)k) * v;
      }
    }
  }
  for (int u = 0; u < 4; ++u)
    if (rq + u <= r1) a.ctx[(ulong)a.valid[rq + u] * C + head * 32 + lane] = acc[u] / l[u];
}
kernel void ef2_gate_live(LF_ARGS(GateLiveArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.A * a.C) return;
  uint atom = lf_udiv(t, a.C);
  a.ctx[t] *= (a.mask[atom] != 0.f ? 1.f : 0.f) * lf_sigmoid(a.gate[t]);
}
kernel void ef2_gated_add(LF_ARGS(GatedAddArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.A * a.C) return;
  uint atom = lf_udiv(t, a.C), c = t - atom * a.C;
  a.x[t] += a.mod[(ulong)atom * 6 * a.C + a.which * a.C + c] * a.d[t];
}
// the mean over each token's atoms, weighted by the mask, into the first C columns of rows ld apart
kernel void ef2_scatter_mean(LF_ARGS(ScatterMeanArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.T * a.C) return;
  uint token = lf_udiv(t, a.C), c = t - token * a.C;
  float s = 0, w = 0;
  for (int k = a.tokenStart[token]; k < a.tokenStart[token + 1]; ++k) {
    int at = a.tokenAtoms[k];
    s += a.v[(ulong)at * a.C + c] * a.mask[at]; w += a.mask[at];
  }
  a.out[(ulong)token * a.ld + c] = s / max(w, 1e-9f);
}
kernel void ef2_tail_inputs(LF_ARGS(TailInputsArgs)) {
  uint t = (uint)LF_INDEX, F = 2 * a.K + 1;
  if (t >= a.T * F) return;
  uint token = lf_udiv(t, F), f = t - token * F;
  float v = f < a.K ? a.aatype[token * a.K + f] : !a.profile ? 0.f : f < 2 * a.K ? a.profile[token * a.K + f - a.K] : a.delMean[token];
  a.out[(ulong)token * a.ld + a.C + f] = v;
}

// ---------------------------------------------------------------- the diffusion module
// rows [p0, p0 + n) of [z | rel_pos]
kernel void ef2_join_pair_rel(LF_ARGS(JoinPairRelArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n * 2 * a.C) return;
  uint C2 = 2 * a.C, pr = lf_udiv((uint)t, C2), c = (uint)t - pr * C2;
  uint p = (uint)a.p0 + pr;
  if (c < a.C) a.out[t] = a.z[(ulong)p * a.C + c];
  else { uint i = lf_udiv(p, a.T); a.out[t] = relPosAt(a.rel, i, p - i * a.T, c - a.C, a.C); }
}
kernel void ef2_silu_mul(LF_ARGS(SiluMulArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.a[t] = lf_silu(a.a[t]) * a.b[t];
}
kernel void ef2_fourier(LF_ARGS(FourierArgs)) {
  uint i = (uint)LF_INDEX;
  if (i < a.n) a.out[i] = precise::cos(2.f * M_PI_F * (a.level[0] * a.w[i] + a.b[i]));
}
kernel void ef2_add_row(LF_ARGS(AddRowArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  uint r = lf_udiv((uint)t, a.C);
  a.x[t] += a.row[(uint)t - r * a.C];
}
kernel void ef2_coords_input(LF_ARGS(CoordsInputArgs)) {     // [A, 6] = [x / denom | 0]
  uint t = (uint)LF_INDEX;
  if (t >= a.A * 6) return;
  uint at = t / 6, k = t - at * 6;
  a.out[t] = k < 3 ? a.x[at * 3 + k] * a.level[1] : 0.f;
}
// adaLN: sigmoid(g + gb) * LN(a) + sh, the projections rows ld apart
kernel void ef2_ada_combine(LF_ARGS(AdaCombineArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.T * a.C) return;
  uint r = lf_udiv(t, a.C), c = t - r * a.C, gi = r * a.ld + c;
  a.out[t] = (half)(a.an[t] * lf_sigmoid(a.g[gi] + a.gb[c]) + a.sh[gi]);
}
kernel void ef2_sigmoid_mul(LF_ARGS(SigmoidMulArgs)) {     // x *= sigmoid(g (+ gb))
  uint t = (uint)LF_INDEX;
  if (t >= a.T * a.C) return;
  uint r = lf_udiv(t, a.C), c = t - r * a.C;
  a.x[t] *= lf_sigmoid(a.g[r * a.ld + c] + (a.gb ? a.gb[c] : 0.f));
}
kernel void ef2_scale_copies(LF_ARGS(ScaleCopiesArgs)) {    // out[j] = x * scale[j], [n, rows, C]
  uint t = (uint)LF_INDEX, per = a.rows * a.C;
  if (t >= a.n * per) return;
  uint j = lf_udiv(t, per), r = t - j * per, c = r - lf_udiv(r, a.C) * a.C;
  a.out[t] = a.x[r] * a.scales[j * a.C + c];
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
kernel void ef2_pair_to_heads(LF_ARGS(PairToHeadsArgs)) {    // [P, H] -> [H, P]
  ulong t = LF_INDEX;
  if (t >= a.P * a.H) return;
  uint p = lf_udiv((uint)t, a.H), h = (uint)t - p * a.H;
  a.out[(ulong)h * a.P + p] = (half)a.pb[t];
}
kernel void ef2_gather_tokens(LF_ARGS(GatherTokensArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.A * a.C) return;
  uint at = lf_udiv(t, a.C), c = t - at * a.C;
  int token = a.mask[at] != 0.f ? a.atomToToken[at] : 0;
  a.q[t] += a.perToken[(uint)token * a.C + c];
}
kernel void ef2_edm_combine(LF_ARGS(EdmArgs)) {
  uint t = (uint)LF_INDEX;
  if (t < a.n) a.out[t] = a.level[2] * a.xNoisy[t] + a.level[3] * a.r[t];
}

// ---------------------------------------------------------------- the confidence head
kernel void ef2_conf_z(constant ConfZArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                       uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint ij = tg.y * ng.x + tg.x;
  if (ij >= a.T * a.T) return;
  uint i = lf_udiv(ij, a.T), j = ij - i * a.T;
  for (uint c = tid; c < a.C; c += 256)
    a.z[(ulong)ij * a.C + c] += relPosAt(a.rel, i, j, c, a.C) + a.bonds[ij] * a.wBond[c] + a.rows[i * a.C + c] + a.cols[j * a.C + c];
}
kernel void ef2_outer(LF_ARGS(OuterArgs)) {      // a_i * b_j, pair rows p0..
  ulong t = LF_INDEX;
  if (t >= a.n * a.C) return;
  uint pr = lf_udiv((uint)t, a.C), c = (uint)t - pr * a.C;
  uint ij = (uint)a.p0 + pr, i = lf_udiv(ij, a.T), j = ij - i * a.T;
  a.out[t] = (half)(a.a[i * a.C + c] * a.b[j * a.C + c]);
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
  for (uint c = lane; c < a.C / 4; c += 32) row[c] += tb[c];
}
// row-attention pooling: pooled[i] = sum_j softmax_j(score[i, j]) z[i, j]; a threadgroup a row
kernel void ef2_row_pool(constant RowPoolArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  uint i = tg.x;
  threadgroup float red[8];
  device const float* sc = a.score + (ulong)i * a.T;
  float mx = -INFINITY;
  for (uint j = tid; j < a.T; j += 256) mx = max(mx, sc[j]);
  mx = simd_max(mx);
  if (lane == 0) red[sg] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = red[0]; for (int k = 1; k < 8; ++k) mx = max(mx, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0;
  for (uint j = tid; j < a.T; j += 256) sum += exp(sc[j] - mx);
  sum = simd_sum(sum);
  if (lane == 0) red[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0; for (int k = 0; k < 8; ++k) tot += red[k];
  for (uint c = tid; c < a.C; c += 256) {
    float acc = 0;
    for (uint j = 0; j < a.T; ++j) acc += exp(sc[j] - mx) / tot * a.z[((ulong)i * a.T + j) * a.C + c];
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
                           uint tid [[thread_index_in_threadgroup]]) {
  uint at = tg.x;
  threadgroup float logit[64];
  device const float* tb = a.table + (ulong)a.slot[at] * a.C * a.bins;
  for (uint b = tid; b < a.bins; b += 64) {
    float acc = 0;
    for (uint c = 0; c < a.C; ++c) acc += a.s[(ulong)at * a.C + c] * tb[(ulong)c * a.bins + b];
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
