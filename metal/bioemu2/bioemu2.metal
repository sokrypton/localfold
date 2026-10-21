// BioEmu2's kernels (cuda/bioemu2/src/*.cuh is the reading: each kernel here is the CUDA kernel of the same name, in
// float32). metal/build.sh prepends core/args.h, this port's kernels.h and core/common.metal.

constant int BE2_WQ = 32, BE2_WK = 128, BE2_CTX = 48;   // atom query / key windows; keys centred on queries
constant float BE2_NEG_INF = -1e6f;                       // the vendor's -INF (1e6): a masked logit, not -inf

// a simdgroup a row, eight rows a threadgroup of 256 (metal/core's LayerNorm's dispatch)
#define BE2_ROW ((ulong)(_tg.y * _ng.x + _tg.x) * 8 + (_tid >> 5))
#define BE2_LANE (_tid & 31)

// ---------------------------------------------------------------- the atoms and their windowed pair
// c = proj_atom_continuous([ref, charge]) + embed_element[element] + sum_q embed_atom_name_chars_q[name[q]], zero on padding
kernel void be2_atom_features(LF_ARGS(Be2AtomFeatArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.N * a.C) return;
  ulong at = t / a.C; uint k = (uint)(t % a.C);
  if (!a.pad[at]) { a.c[t] = 0.f; return; }
  const float in[4] = {a.ref[at * 3], a.ref[at * 3 + 1], a.ref[at * 3 + 2], a.charge[at]};
  float v = 0.f;
  for (int q = 0; q < 4; ++q) v += a.wCont[k * 4 + q] * in[q];
  v += a.wElem[(ulong)a.element[at] * a.C + k];
  v += a.wN0[(ulong)a.names[at * 4 + 0] * a.C + k];
  v += a.wN1[(ulong)a.names[at * 4 + 1] * a.C + k];
  v += a.wN2[(ulong)a.names[at * 4 + 2] * a.C + k];
  v += a.wN3[(ulong)a.names[at * 4 + 3] * a.C + k];
  a.c[t] = v;
}
// windowed_atom_pair_geometry: pr = (W_pos disp + W_dist / (1 + |disp|^2) + W_mask same) same, disp = key - query,
// `same`: both in one reference space and the query real
kernel void be2_pair_geometry(LF_ARGS(Be2PairGeomArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.N * BE2_WK) return;
  int j = (int)(t % BE2_WK); ulong qa = t / BE2_WK;
  int w = (int)(qa / BE2_WQ), ka = w * BE2_WQ - BE2_CTX + j;
  bool inRange = ka >= 0 && ka < (int)a.N;
  float kx = inRange ? a.ref[ka * 3] : 0.f, ky = inRange ? a.ref[ka * 3 + 1] : 0.f, kz = inRange ? a.ref[ka * 3 + 2] : 0.f;
  float d0 = kx - a.ref[qa * 3], d1 = ky - a.ref[qa * 3 + 1], d2 = kz - a.ref[qa * 3 + 2];
  float inv = 1.f / (1.f + d0 * d0 + d1 * d1 + d2 * d2);
  int ku = inRange && a.pad[ka] ? a.uid[ka] : -1;
  float same = a.pad[qa] && a.uid[qa] == ku ? 1.f : 0.f;
  device float* o = a.pr + t * a.CP_;
  for (uint c = 0; c < a.CP_; ++c)
    o[c] = (a.wPos[c * 3] * d0 + a.wPos[c * 3 + 1] * d1 + a.wPos[c * 3 + 2] * d2 + a.wDist[c] * inv + a.wMask[c] * same) * same;
}
// broadcast_token_pair_to_atom_window: pr += zp[tok_q][tok_k], zero where either atom is padding
kernel void be2_pair_token_add(LF_ARGS(Be2PairTokenAddArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.N * BE2_WK) return;
  int j = (int)(t % BE2_WK); ulong qa = t / BE2_WK; int w = (int)(qa / BE2_WQ);
  int ka = w * BE2_WQ - BE2_CTX + j;
  if (ka < 0 || ka >= (int)a.N || !a.pad[ka] || !a.pad[qa]) return;
  device const float* z = a.zp + ((ulong)a.atomTok[qa] * a.T + a.atomTok[ka]) * a.CP_;
  device float* o = a.pr + t * a.CP_;
  for (uint c = 0; c < a.CP_; ++c) o[c] += z[c];
}
// _atom_pair_tail: pr += cq[q] + ck[k] (ck zero past the edge), then pr += p_mlp(pr): three ReLU-Linear layers, 16 wide
kernel void be2_pair_tail(LF_ARGS(Be2PairTailArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.N * BE2_WK) return;
  int j = (int)(t % BE2_WK); ulong qa = t / BE2_WK; int w = (int)(qa / BE2_WQ);
  int ka = w * BE2_WQ - BE2_CTX + j;
  float p[16], h[16], g[16];
  device float* o = a.pr + t * 16;
  for (int c = 0; c < 16; ++c) p[c] = o[c] + a.cq[qa * 16 + c] + (ka >= 0 && ka < (int)a.N ? a.ck[(ulong)ka * 16 + c] : 0.f);
  for (int l = 0; l < 3; ++l) {
    device const float* W = l == 0 ? a.w1 : l == 1 ? a.w3 : a.w5;
    thread float* src = l == 0 ? p : l == 1 ? h : g;
    thread float* dst = l == 0 ? h : l == 1 ? g : h;
    float in[16];
    for (int c = 0; c < 16; ++c) in[c] = max(src[c], 0.f);
    for (int r = 0; r < 16; ++r) { float s = 0; for (int c = 0; c < 16; ++c) s += W[r * 16 + c] * in[c]; dst[r] = s; }
  }
  for (int c = 0; c < 16; ++c) o[c] = p[c] + h[c];
}
// the windowed pair's bias: LN(pr) (scale, no offset) -> proj_z [H][16], heads first: out[w][h][i][j]
kernel void be2_window_bias(LF_ARGS(Be2WindowBiasArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.N * BE2_WK) return;
  int j = (int)(t % BE2_WK); ulong qa = t / BE2_WK; int w = (int)(qa / BE2_WQ), i = (int)(qa % BE2_WQ);
  device const float* x = a.pr + t * 16;
  float mean = 0; for (int c = 0; c < 16; ++c) mean += x[c]; mean /= 16.f;
  float var = 0; for (int c = 0; c < 16; ++c) { float d = x[c] - mean; var += d * d; }
  float inv = rsqrt(var / 16.f + 1e-5f);
  float n[16]; for (int c = 0; c < 16; ++c) n[c] = (x[c] - mean) * inv * a.normW[c];
  for (uint h = 0; h < a.H; ++h) {
    float s = 0; for (int c = 0; c < 16; ++c) s += a.projZ[h * 16 + c] * n[c];
    a.out[(((ulong)w * a.H + h) * BE2_WQ + i) * BE2_WK + j] = s;
  }
}
// AttentionPairBiasSlidingWindow: a threadgroup per (window, head) of 32 threads, a thread per query; online softmax over
// the 128 keys (head width 32). Several samples stacked on the rows, N atoms each; every sample reads one bias
kernel void be2_window_attention(constant Be2WindowAttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint i [[thread_index_in_threadgroup]]) {
  constexpr int HD = 32;
  const uint wg = tg.x, h = tg.y, D = a.H * HD, NW = a.N / BE2_WQ, smp = wg / NW, w = wg % NW;
  const ulong base = (ulong)smp * a.N;
  // (the keys and values are 32 KB, Apple's whole threadgroup allowance: the key mask is read per key from the device)
  threadgroup float sk[BE2_WK][HD], sv[BE2_WK][HD];
  for (uint e = i; e < BE2_WK * HD; e += BE2_WQ) {
    int j = e / HD, d = e % HD, ka = (int)(w * BE2_WQ) - BE2_CTX + j;
    bool ok = ka >= 0 && ka < (int)a.N;
    sk[j][d] = ok ? a.qkv[(base + ka) * 3 * D + D + h * HD + d] : 0.f;
    sv[j][d] = ok ? a.qkv[(base + ka) * 3 * D + 2 * D + h * HD + d] : 0.f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong qa = base + w * BE2_WQ + i;
  float q[HD];
  const float scale = rsqrt((float)HD);
  for (int d = 0; d < HD; ++d) q[d] = (a.qkv[qa * 3 * D + h * HD + d] + a.qBias[h * HD + d]) * scale;
  device const float* b = a.bias + (((ulong)w * a.H + h) * BE2_WQ + i) * BE2_WK;
  float mx = -INFINITY, sum = 0, acc[HD];
  for (int d = 0; d < HD; ++d) acc[d] = 0;
  for (int j = 0; j < BE2_WK; ++j) {
    float l = 0; for (int d = 0; d < HD; ++d) l += q[d] * sk[j][d];
    const int ka = (int)(w * BE2_WQ) - BE2_CTX + j;
    l = ka >= 0 && ka < (int)a.N && a.pad[ka] ? l + b[j] : BE2_NEG_INF;
    float nm = max(mx, l), f = exp(mx - nm), e = exp(l - nm);
    sum = sum * f + e;
    for (int d = 0; d < HD; ++d) acc[d] = acc[d] * f + e * sv[j][d];
    mx = nm;
  }
  for (int d = 0; d < HD; ++d) a.out[qa * D + h * HD + d] = acc[d] / sum;
}
// AttentionPairBias over all T tokens of each stacked sample: a thread per (sample, head, query), online softmax over
// the keys; qkv [samples T][3D] (q, k, v thirds), bias [H][T][T]; out [samples T][D]. (The float32 reference arm: the
// fast path is metal/core's flash attention.)
kernel void be2_token_attention(LF_ARGS(Be2TokenAttnArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.samples * a.H * a.T) return;
  const uint i = (uint)(t % a.T), h = (uint)((t / a.T) % a.H), s = (uint)(t / ((ulong)a.T * a.H));
  const uint D = a.D, HD = D / a.H;
  device const float* qkv = a.qkv + (ulong)s * a.T * 3 * D;
  device const float* qr = qkv + (ulong)i * 3 * D + h * HD;
  const float scale = rsqrt((float)HD);
  device const float* b = a.bias + ((ulong)h * a.T + i) * a.T;
  float mx = -INFINITY, sum = 0, acc[64];
  for (uint d = 0; d < HD; ++d) acc[d] = 0;
  for (uint j = 0; j < a.T; ++j) {
    device const float* kr = qkv + (ulong)j * 3 * D + D + h * HD;
    float l = 0; for (uint d = 0; d < HD; ++d) l += (qr[d] + a.qBias[h * HD + d]) * kr[d];
    l = a.keyMask[j] ? l * scale + b[j] : BE2_NEG_INF;
    float nm = max(mx, l), f = exp(mx - nm), e = exp(l - nm);
    sum = sum * f + e;
    device const float* vr = kr + D;
    for (uint d = 0; d < HD; ++d) acc[d] = acc[d] * f + e * vr[d];
    mx = nm;
  }
  device float* o = a.out + ((ulong)s * a.T + i) * D + h * HD;
  for (uint d = 0; d < HD; ++d) o[d] = acc[d] / sum;
}

// ---------------------------------------------------------------- elementwise
kernel void be2_unary(LF_ARGS(Be2UnaryArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n) return;
  float v = a.x[t];
  a.y[t] = a.op == 0 ? max(v, 0.f) : lf_silu(v);
}
kernel void be2_silu_mul(LF_ARGS(Be2SiluMulArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.g[t] = lf_silu(a.h1[t]) * a.h2[t];
}
kernel void be2_swiglu_blocked(LF_ARGS(Be2SwigluBlockedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.I) return;
  ulong r = t / a.I; uint c = (uint)(t % a.I);
  a.g[t] = lf_silu(a.h[r * 2 * a.I + c]) * a.h[r * 2 * a.I + a.I + c];
}
kernel void be2_sigmoid_mul(LF_ARGS(Be2SigmoidMulArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.y[t] = a.h[t] * lf_sigmoid(a.g[t]);
}
kernel void be2_gated_add(LF_ARGS(Be2GatedAddArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  ulong r = t / a.C; uint c = (uint)(t % a.C);
  ulong g = a.map ? (ulong)a.map[r] : r;
  a.x[t] += a.gate[g * a.ldg + c] * a.h[t];
}
kernel void be2_ada_ln(LF_ARGS(Be2AdaLnArgs)) {
  const ulong row = BE2_ROW; const uint lane = BE2_LANE;
  if (row >= a.rows) return;
  device const float* xr = a.x + row * a.ldx;
  float s = 0; for (uint c = lane; c < a.C; c += 32) s += xr[c];
  s = simd_sum(s);
  float mean = s / a.C, v = 0;
  for (uint c = lane; c < a.C; c += 32) { float d = xr[c] - mean; v += d * d; }
  v = simd_sum(v);
  float inv = rsqrt(v / a.C + 1e-5f);
  const ulong cr = a.map ? (ulong)a.map[row] : row;
  for (uint c = lane; c < a.C; c += 32)
    a.y[row * a.C + c] = a.scale * ((xr[c] - mean) * inv * (1.f + a.alpha[cr * a.ldc + c]) + a.beta[cr * a.ldc + c]);
}
// a token tensor's rows broadcast to its atoms (zero on padding atoms), samples stacked
kernel void be2_tokens_to_atoms(LF_ARGS(Be2TokensToAtomsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.samples * a.N * a.C) return;
  ulong at = t / a.C, s = at / a.N, al = at % a.N; uint c = (uint)(t % a.C);
  float v = a.pad[al] ? a.tok[(s * a.T + a.atomTok[al]) * a.C + c] : 0.f;
  a.out[at * a.ldo + c] = a.beta == 0.f ? v : a.beta * a.out[at * a.ldo + c] + v;
}
// each token's centre atom (every token is real here: no token padding)
kernel void be2_centre_atoms(LF_ARGS(Be2CentreAtomsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.samples * a.T * a.C) return;
  ulong i = t / a.C, s = i / a.T, il = i % a.T; uint c = (uint)(t % a.C);
  a.out[i * a.ldo + c] = a.atom[(s * a.N + a.centre[il]) * a.C + c];
}
// scatter_mean_atoms_to_tokens: per token, the mean over its real atoms (+1e-6 in the count)
kernel void be2_mean_atoms(LF_ARGS(Be2MeanAtomsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.T * a.C) return;
  uint i = (uint)(t / a.C), c = (uint)(t % a.C);
  float s = 0; int n = a.start[i + 1] - a.start[i];
  for (int at = a.start[i]; at < a.start[i + 1]; ++at) s += a.x[(ulong)at * a.C + c];
  a.out[(ulong)i * a.ldo + c] = s / ((float)n + 1e-6f);
}
// s_inputs past the atom half: one-hot res_type (33), profile (33), deletion_mean, one-hot pocket (4)
kernel void be2_s_inputs_tail(LF_ARGS(Be2SInputsTailArgs)) {
  ulong t = LF_INDEX;
  const uint W = 33 + 33 + 1 + 4;
  if (t >= (ulong)a.T * W) return;
  uint i = (uint)(t / W), k = (uint)(t % W);
  float v;
  if (k < 33) v = a.resType[i] == (int)k ? 1.f : 0.f;
  else if (k < 66) v = a.profile[(ulong)i * 33 + (k - 33)];
  else if (k == 66) v = a.delMean[i];
  else v = a.pocket[i] == (int)k - 67 ? 1.f : 0.f;
  a.s[(ulong)i * a.ld + a.C0 + k] = v;
}
// z_init's terms past the two projections: the method pair (both axes), the token bonds, the contact conditioning
kernel void be2_z_init_terms(LF_ARGS(Be2ZInitTermsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.T * a.T * a.C) return;
  uint c = (uint)(t % a.C); ulong ij = t / a.C; uint i = (uint)(ij / a.T), j = (uint)(ij % a.T);
  float v = a.rows[(ulong)i * a.C + c] + a.cols[(ulong)j * a.C + c];
  v += a.methPair[(ulong)a.meth[i] * a.C + c] + a.methPair[(ulong)a.meth[j] * a.C + c];
  v += a.bonds[ij] * a.wBond[c];
  // x = encoder([cc[2:], thr', cos 2 pi (w thr' + b)]) for a class past UNSELECTED; the class's own vector for
  // UNSPECIFIED (0) and UNSELECTED (1)
  int k = a.cls[ij];
  if (k == 0) v += a.unspec[c];
  else if (k == 1) v += a.unsel[c];
  else {
    const uint W = 3 + 1 + a.C;
    float th = (a.thr[ij] - 4.f) / 16.f;
    device const float* row = a.enc + (ulong)c * W;
    float e = a.encB[c] + row[k - 2] + row[3] * th;
    for (uint f = 0; f < a.C; ++f) e += row[4 + f] * cos(2.f * M_PI_F * (th * a.fw[f] + a.fb[f]));
    v += e;
  }
  a.z[t] = v;
}
// FusedRelativePositionEncoder over the token index (+= when add)
kernel void be2_rel_pos(LF_ARGS(Be2RelPosArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.T * a.T * a.C) return;
  uint c = (uint)(t % a.C); ulong ij = t / a.C; int i = (int)(ij / a.T), j = (int)(ij % a.T);
  const int rmax = 32, smax = 2;
  bool sameChain = a.asym[i] == a.asym[j], sameRes = a.res[i] == a.res[j], sameEnt = a.entity[i] == a.entity[j];
  int dRes = sameChain ? clamp(a.res[i] - a.res[j] + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  int dTok = sameChain && sameRes ? clamp(i - j + rmax, 0, 2 * rmax) : 2 * rmax + 1;
  int dChain = sameEnt ? clamp(a.sym[i] - a.sym[j] + smax, 0, 2 * smax) : 2 * smax + 1;
  float v = a.pos[(ulong)dRes * a.C + c] + a.tok[(ulong)dTok * a.C + c] + (sameEnt ? a.ent[c] : 0.f)
            + a.chain[(ulong)dChain * a.C + c];
  device float* o = a.out + ij * a.ldo + c;
  *o = a.add ? *o + v : v;
}
// the triangle's a and b, gated, channel-major: AF3's layout interleaves them by channel (column 2c a's, 2c + 1 b's)
kernel void be2_tri_split(LF_ARGS(Be2TriSplitArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.pairs * 2 * a.C) return;
  uint c2 = (uint)(t % (2 * a.C)); ulong ij = t / (2 * a.C);
  float v = a.proj[t] * lf_sigmoid(a.gate[t]);
  if (c2 & 1) a.b[(ulong)(c2 >> 1) * a.pairs + ij] = v; else a.a[(ulong)(c2 >> 1) * a.pairs + ij] = v;
}
kernel void be2_channel_to_rows(LF_ARGS(Be2ChannelToRowsArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.pairs * a.C) return;
  uint c = (uint)(t % a.C); ulong ij = t / a.C;
  a.rows[t] = a.prod[(ulong)c * a.pairs + ij];
}
kernel void be2_gate_mul_add(LF_ARGS(Be2GateMulAddArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.z[t] += a.out[t] * lf_sigmoid(a.g[t]);
}
// m [S][T][C] = msa_proj([one-hot msa | has_deletion | deletion_value | msa_paired]) + s_proj(s_inputs)[t]
kernel void be2_msa_embed(LF_ARGS(Be2MsaEmbedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.S * a.T * a.C) return;
  uint c = (uint)(t % a.C); ulong st = t / a.C; uint i = (uint)(st % a.T);
  device const float* row = a.w + (ulong)c * 36;
  a.m[t] = row[a.msa[st]] + row[33] * (float)a.hasDel[st] + row[34] * a.delVal[st] + row[35] * (float)a.paired[st]
           + a.sp[(ulong)i * a.C + c];
}
kernel void be2_pwa_softmax(LF_ARGS(Be2PwaSoftmaxArgs)) {
  const ulong row = BE2_ROW; const uint lane = BE2_LANE;
  if (row >= (ulong)a.H * a.T) return;
  uint h = (uint)(row / a.T), i = (uint)(row % a.T);
  float mx = -INFINITY;
  for (uint j = lane; j < a.T; j += 32) mx = max(mx, a.b[((ulong)i * a.T + j) * a.H + h]);
  mx = simd_max(mx);
  float s = 0;
  for (uint j = lane; j < a.T; j += 32) s += exp(a.b[((ulong)i * a.T + j) * a.H + h] - mx);
  s = simd_sum(s);
  for (uint j = lane; j < a.T; j += 32) a.w[row * a.T + j] = exp(a.b[((ulong)i * a.T + j) * a.H + h] - mx) / s;
}
kernel void be2_pwa_apply(LF_ARGS(Be2PwaApplyArgs)) {
  ulong t = LF_INDEX;
  const uint W = a.H * a.Ch;
  if (t >= a.S * a.T * W) return;
  uint d = (uint)(t % a.Ch); ulong r = t / a.Ch; uint h = (uint)(r % a.H); r /= a.H; uint i = (uint)(r % a.T); ulong s = r / a.T;
  device const float* w = a.w + ((ulong)h * a.T + i) * a.T;
  device const float* v = a.v + s * a.T * W + h * a.Ch + d;
  float acc = 0;
  for (uint j = 0; j < a.T; ++j) acc += w[j] * v[(ulong)j * W];
  a.o[t] = acc;
}
// OuterProductMeanVanilla over the rows (the mean over VALID rows): proj [S][T][2 Co], a its first half, b its second
kernel void be2_opm(LF_ARGS(Be2OpmArgs)) {
  ulong t = LF_INDEX;
  const uint Co = a.Co, CC = Co * Co;
  if (t >= (ulong)a.bi * a.T * CC) return;
  uint d = (uint)(t % Co); ulong r = t / Co; uint c = (uint)(r % Co); r /= Co; uint j = (uint)(r % a.T); uint i = a.i0 + (uint)(r / a.T);
  if (i >= a.T) return;
  float acc = 0;
  for (ulong s = 0; s < a.S; ++s) {
    device const float* row = a.proj + s * a.T * 2 * Co;
    acc += row[(ulong)i * 2 * Co + c] * row[(ulong)j * 2 * Co + Co + d];
  }
  a.y[t] = acc * a.inv;
}
kernel void be2_symmetrise(LF_ARGS(Be2SymmetriseArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.T * a.T * a.C) return;
  uint c = (uint)(t % a.C); ulong ij = t / a.C; uint i = (uint)(ij / a.T), j = (uint)(ij % a.T);
  a.zs[t] = a.z[t] + a.z[((ulong)j * a.T + i) * a.C + c];
}
// the contact map: the softmax mass of a pair's first `near` bins
kernel void be2_contacts(LF_ARGS(Be2ContactArgs)) {
  ulong ij = LF_INDEX;
  if (ij >= a.pairs) return;
  device const float* l = a.logits + ij * a.bins;
  float mx = -INFINITY; for (uint b = 0; b < a.bins; ++b) mx = max(mx, l[b]);
  float tot = 0, c = 0;
  for (uint b = 0; b < a.bins; ++b) { float e = exp(l[b] - mx); tot += e; if (b < a.near) c += e; }
  a.out[ij] = c / tot;
}
kernel void be2_token_bias(LF_ARGS(Be2TokenBiasArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.T * a.T * a.H) return;
  uint h = (uint)(t % a.H); ulong ij = t / a.H;
  a.out[(ulong)h * a.T * a.T + ij] = a.proj[t];
}

// ---------------------------------------------------------------- the denoiser
kernel void be2_scale_masked(LF_ARGS(Be2ScaleMaskedArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n * 3) a.r[t] = a.mask[(t / 3) % a.N] ? a.x[t] * a.s : 0.f;
}
kernel void be2_edm_out(LF_ARGS(Be2EdmOutArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.d[t] = a.skip * a.x[t] + a.cout * a.out[t];
}
kernel void be2_cos_embed(LF_ARGS(Be2CosEmbedArgs)) {
  ulong k = LF_INDEX;
  if (k < a.n) a.out[k] = cos(2.f * M_PI_F * (a.t * a.w[k] + a.b[k]));
}
kernel void be2_broadcast_add(LF_ARGS(Be2BroadcastAddArgs)) {
  ulong t = LF_INDEX;
  if (t < a.rows * a.C) a.y[t] += a.v[t % a.C];
}
kernel void be2_scaled_add(LF_ARGS(Be2ScaledAddArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.y[t] += a.s * a.x[t];
}
kernel void be2_repeat(LF_ARGS(Be2RepeatArgs)) {
  ulong t = LF_INDEX;
  if (t < a.total) a.dst[t] = a.src[t % a.elems];
}
