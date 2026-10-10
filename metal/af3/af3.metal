// The AlphaFold 3 lineage's kernels (metal/af3): derived weights, the embedder's features, the attention biases, the
// MSA stack's pair-weighted averaging and outer product, the distogram. cuda/af3/src is the reading.

// ---------------------------------------------------------------- derived weights
kernel void af3_concat_part(LF_ARGS(ConcatPartArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.C * a.width) return;
  uint c = lf_udiv(t, a.width), o = t - c * a.width;
  a.dst[c * a.total + a.off + o] = a.transposed ? a.src[o * a.C + c] : a.src[t];
}
kernel void af3_interleave8(LF_ARGS(Interleave8Args)) {
  uint t = (uint)LF_INDEX, W = 2 * a.I;
  if (t >= a.rows * W) return;
  uint r = lf_udiv(t, W), c = t - r * W, m = c >> 4, q = c & 15;
  a.out[t] = q < 8 ? a.a[r * a.lda + 8 * m + q] : a.b[r * a.lda + 8 * m + q - 8];
}
kernel void af3_trigate_weight(LF_ARGS(TriGateWArgs)) {
  uint t = (uint)LF_INDEX, W = 4 * a.C;
  if (t >= a.rows * W) return;
  uint r = lf_udiv(t, W), col = t - r * W, kind = (col % 32) / 8, c = 8 * (col / 32) + col % 8;
  uint k = kind == 1 ? 2 : kind == 2 ? 1 : kind;      // (blocks of 8: pa ga pb gb; a and b interleaved by channel)
  a.out[t] = k < 2 ? a.proj[r * 2 * a.C + 2 * c + k] : a.gate[r * 2 * a.C + 2 * c + k - 2];
}
kernel void af3_scale(LF_ARGS(ScaleArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n) return;
  float v = a.x[t] * a.s;
  a.x[t] = a.relu ? max(v, 0.f) : v;
}
kernel void af3_scale_rows(LF_ARGS(ScaleRowsArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  uint r = lf_udiv((uint)t, a.C);
  a.x[t] *= a.mask[r - lf_udiv(r, a.period) * a.period];
}
kernel void af3_scale_rows_h(LF_ARGS(ScaleRowsHArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  a.x[t] = (half)((float)a.x[t] * a.mask[lf_udiv((uint)t, a.C)]);
}

// ---------------------------------------------------------------- the embedder
kernel void af3_outer_sum(LF_ARGS(OuterSumArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  a.pair[t] = a.left[i * a.C + c] + a.right[j * a.C + c] + (a.add ? a.add[t] : 0.f);
}
// AF3's relative encoding (139 one-hot columns: residue offset 66, token offset 66, same entity 1, chain offset 6)
kernel void af3_relenc(LF_ARGS(RelEncArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  const int maxIdx = 32, maxChain = 2, positionBins = 2 * maxIdx + 2;
  bool sameChain = a.r.asym[i] == a.r.asym[j], sameEntity = a.r.ent[i] == a.r.ent[j];
  int c0 = sameChain ? clamp(a.r.ri[i] - a.r.ri[j] + maxIdx, 0, 2 * maxIdx) : 2 * maxIdx + 1;
  bool sameResidue = sameChain && a.r.ri[i] == a.r.ri[j];
  int c1 = positionBins + (sameResidue ? clamp(a.r.ti[i] - a.r.ti[j] + maxIdx, 0, 2 * maxIdx) : 2 * maxIdx + 1);
  int c3 = positionBins * 2 + 1 + (sameEntity ? clamp(a.r.sym[i] - a.r.sym[j] + maxChain, 0, 2 * maxChain) : 2 * maxChain + 1);
  float v = a.W[c0 * a.C + c] + a.W[c1 * a.C + c] + a.W[c3 * a.C + c];
  if (sameEntity) v += a.W[(positionBins * 2) * a.C + c];
  a.pair[t] += v;
}
kernel void af3_bond_embed(LF_ARGS(BondEmbedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.pairs * a.C) return;
  uint p = lf_udiv((uint)t, a.C), c = (uint)t - p * a.C;
  float v = 0.f;
  if (a.bonds && a.w) v += a.bonds[p] * a.w[c];
  if (a.table) {
    int o = a.orders ? (int)a.orders[p] : 0;
    if (o < 0 || o >= 7) o = 0;
    v += a.table[o * a.C + c] + a.unspecified[c];
  }
  a.pair[t] += v;
}
// msa = one_hot(32) + clip(deletion) + atan(deletion / 3) 2 / pi (+ is_paired), projected, plus the target's projection
kernel void af3_msa_embed(LF_ARGS(MsaEmbedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.count * a.C) return;
  uint row = lf_udiv((uint)t, a.C), c = (uint)t - row * a.C, token = row - lf_udiv(row, a.n) * a.n;
  int code = a.rows[row]; float d = a.del[row];
  float v = code >= 0 && code < 32 ? a.W[code * a.C + c] : 0.f;
  v += clamp(d, 0.f, 1.f) * a.W[32 * a.C + c];
  v += atan(d / 3.f) * (2.f / M_PI_F) * a.W[33 * a.C + c];
  if (a.width > 34 && a.pairedQuery && row < a.n) v += a.W[34 * a.C + c];
  a.msa[t] = v + a.fromTarget[token * a.C + c];
}
kernel void af3_onehot(LF_ARGS(OnehotArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.n * a.classes) return;
  uint r = lf_udiv(t, a.classes), c = t - r * a.classes;
  a.out[t] = a.idx[r] == (int)c ? 1.f : 0.f;
}
kernel void af3_add_row_col(LF_ARGS(AddRowColArgs)) {      // act[i][j] += row[j] + col[i]
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  a.act[t] += a.row[j * a.C + c] + a.col[i * a.C + c];
}
kernel void af3_template_geometry(LF_ARGS(TmplGeomArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.pairs * a.C) return;
  uint p = lf_udiv((uint)t, a.C), c = (uint)t - p * a.C;
  float v = 0.f;
  int b = a.bin[p];
  if (b >= 0) v += a.W0[b * a.C + c];
  v += a.pb[p] * a.W1[c] + a.uv[p * 3] * a.W4[c] + a.uv[p * 3 + 1] * a.W5[c] + a.uv[p * 3 + 2] * a.W6[c] + a.bb[p] * a.W7[c];
  a.act[t] += v;
}
kernel void af3_scatter_rows(LF_ARGS(ScatterRowsArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.K) return;
  int c = a.idx[t];
  if (c >= 0) a.dense[lf_udiv((uint)t, a.K) * a.width + c] = a.val[t];
}

// ---------------------------------------------------------------- attention biases
// a 32 x 32 tile of (i, j) a threadgroup through threadgroup memory: the raw scores read along their own rows (one
// thread an element, the swapped direction's reads were n x ld floats apart: 140 -> 31 ms a trunk pass at 510 tokens),
// every head in turn (the later heads' lines cached)
kernel void af3_bias_layout(constant BiasLayoutArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float tile[32][33];
  const uint j0 = tg.x * 32, i0 = tg.y * 32, tx = tid & 31, ty = tid >> 5;     // 32 x 8 threads
  for (uint h = 0; h < a.heads; ++h) {
    // read: the source row s0 + r, column c0 + tx - (i, j) unswapped, (j, i) swapped
    const uint s0 = a.swap ? j0 : i0, c0 = a.swap ? i0 : j0;
    for (uint r = ty; r < 32; r += 8) {
      const uint sr = s0 + r, sc = c0 + tx;
      tile[r][tx] = sr < a.n && sc < a.n ? a.raw[((ulong)sr * a.n + sc) * a.ld + a.off + h] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = ty; r < 32; r += 8) {
      const uint i = i0 + r, j = j0 + tx;
      if (i < a.n && j < a.stride) {
        const float v = j < a.n ? (a.swap ? tile[tx][r] : tile[r][tx]) : 0.f;
        a.bias[((ulong)h * a.n + i) * a.stride + j] = (half)(a.scale * v);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---------------------------------------------------------------- the MSA stack
kernel void af3_key_mask(LF_ARGS(KeyMaskArgs)) {
  uint j = (uint)LF_INDEX;
  if (j >= a.n) return;
  float m = 0.f;
  for (uint s = 0; s < a.S; ++s) m = max(m, a.msaMask[s * a.n + j]);
  a.keyMask[j] = m;
}
kernel void af3_msa_weights(constant MsaWeightsArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  uint row = tg.x, h = row / a.n, i = row - h * a.n;
  device half* out = a.w + (ulong)row * a.ld;
  float mx = -1e30f;
  for (uint j = tid; j < a.n; j += 256) mx = max(mx, a.flat[((ulong)i * a.n + j) * a.heads + h] + 1e9f * (a.keyMask[j] - 1.f));
  mx = simd_max(mx);
  if (lane == 0) red[sg] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = red[0]; for (int k = 1; k < 8; ++k) mx = max(mx, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0.f;
  for (uint j = tid; j < a.n; j += 256) s += exp(a.flat[((ulong)i * a.n + j) * a.heads + h] + 1e9f * (a.keyMask[j] - 1.f) - mx);
  s = simd_sum(s);
  if (lane == 0) red[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  s = 0.f; for (int k = 0; k < 8; ++k) s += red[k];
  float inv = 1.f / s;
  for (uint j = tid; j < a.ld; j += 256)
    out[j] = j < a.n ? (half)(exp(a.flat[((ulong)i * a.n + j) * a.heads + h] + 1e9f * (a.keyMask[j] - 1.f) - mx) * inv) : (half)0.f;
}
kernel void af3_msa_v_heads(LF_ARGS(MsaVHeadsArgs)) {
  ulong t = LF_INDEX;
  const ulong total = (ulong)a.S * a.np * a.heads * a.d;
  if (t >= total) return;
  uint e, s, j, h;   // (32-bit division where the index fits: the 64-bit one is slow)
  if (total <= 0xffffffffull) {
    uint r = lf_udiv((uint)t, a.d); e = (uint)t - r * a.d; uint r2 = lf_udiv(r, a.S); s = r - r2 * a.S; h = lf_udiv(r2, a.np); j = r2 - h * a.np;
  } else {
    e = (uint)(t % a.d); ulong rest = t / a.d; s = (uint)(rest % a.S); rest /= a.S; j = (uint)(rest % a.np); h = (uint)(rest / a.np);
  }
  a.out[t] = j < a.n ? a.v[((ulong)s * a.n + j) * a.heads * a.d + h * a.d + e] : (half)0.f;
}
kernel void af3_msa_from_heads(LF_ARGS(MsaFromHeadsArgs)) {
  ulong t = LF_INDEX;
  uint W = a.heads * a.d;
  if (t >= (ulong)a.S * a.n * W) return;
  uint si = lf_udiv((uint)t, W), c = (uint)t - si * W, s = lf_udiv(si, a.n), i = si - s * a.n, h = c / a.d, e = c - h * a.d;
  a.out[t] = (half)((float)a.o[(((ulong)h * a.n + i) * a.S + s) * a.d + e] * lf_sigmoid((float)a.gate[t]));
}
kernel void af3_mask_norm(LF_ARGS(MaskNormArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.L * a.L) return;
  uint i = lf_udiv(t, a.L), j = t - i * a.L;
  float s = 0;
  for (uint q = 0; q < a.S; ++q) s += a.mask[q * a.L + i] * a.mask[q * a.L + j];
  a.norm[t] = s;
}
kernel void af3_opm_left(LF_ARGS(OpmLeftArgs)) {          // lt [s][i][c] -> [i][c][s] (scaled)
  ulong t = LF_INDEX;
  const ulong total = (ulong)a.S * a.L * a.O;
  if (t >= total) return;
  uint s, c, i;      // (32-bit division where the index fits: the 64-bit one is slow)
  if (total <= 0xffffffffull) { uint r = lf_udiv((uint)t, a.S); s = (uint)t - r * a.S; i = lf_udiv(r, a.O); c = r - i * a.O; }
  else { s = (uint)(t % a.S); ulong r = t / a.S; c = (uint)(r % a.O); i = (uint)(r / a.O); }
  a.out[t] = (half)((float)a.lt[((ulong)s * a.L + i) * a.O + c] * a.scale);
}
kernel void af3_opm_permute(LF_ARGS(OpmPermuteArgs)) {    // Pm [(i, c)][(j, e)] -> X [(i, j)][(c, e)], eight halves a thread
  ulong t = LF_INDEX;
  uint O8 = a.O / 8;
  if (t >= (ulong)a.bi * a.L * a.O * O8) return;
  // (32-bit division - the host's block keeps t under 2^32 - the 64-bit one held AF2's twin to ~16 GB/s)
  uint r = lf_udiv((uint)t, O8), e8 = (uint)t - r * O8;
  uint r2 = lf_udiv(r, a.O), c = r - r2 * a.O;
  uint i = lf_udiv(r2, a.L), j = r2 - i * a.L;
  ((device uint4*)a.X)[t] = ((device const uint4*)a.Pm)[(((ulong)i * a.O + c) * ((ulong)a.L * a.O) + (ulong)j * a.O) / 8 + e8];
}
kernel void af3_opm_add(LF_ARGS(OpmAddArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.bi * a.L * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), f = (uint)t - ij * a.C, ii = lf_udiv(ij, a.L), j = ij - ii * a.L;
  ulong i = a.i0 + ii;
  float nv = a.norm[i * a.L + j];
  a.pair[(i * a.L + j) * a.C + f] += a.after ? a.Y[t] / max(nv, 1.f) + a.bias[f] : (a.bias[f] + a.Y[t]) / (1e-3f + nv);
}

// ---------------------------------------------------------------- the distogram
kernel void af3_symmetrise(LF_ARGS(SymmetriseArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.L * a.L * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.L), j = ij - i * a.L;
  a.y[t] = (a.x[t] + a.x[((ulong)j * a.L + i) * a.C + c]) * a.scale;
}
kernel void af3_contact_bins(LF_ARGS(ContactBinsArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.n * a.n) return;
  uint i = lf_udiv(t, a.n), j = t - i * a.n;
  a.out[t] = a.table[a.classes[i] * 23 + a.classes[j]];
}
kernel void af3_contact_probs(LF_ARGS(ContactProbsArgs)) {
  ulong ij = LF_INDEX;
  if (ij >= a.pairs) return;
  device const float* l = a.logits + ij * a.nb;
  float mx = -1e30f;
  for (uint b = 0; b < a.nb; ++b) mx = max(mx, l[b]);
  float total = 0.f, contact = 0.f;
  for (uint b = 0; b < a.nb; ++b) { float p = exp(l[b] - mx); total += p; if ((int)b < a.bins[ij]) contact += p; }
  a.out[ij] = a.pairMask[ij] * contact / total;
}

// ---------------------------------------------------------------- the atoms
kernel void af3_convert(LF_ARGS(ConvertArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.ns * a.count * a.C) return;
  uint i = lf_udiv((uint)t, a.C), c = (uint)t - i * a.C, k = lf_udiv(i, a.count), g = i - k * a.count;
  a.out[t] = a.mask[g] != 0.f ? a.src[((ulong)a.idx[g] + k * a.srcRows) * a.C + c] : 0.f;
}
kernel void af3_per_atom(LF_ARGS(PerAtomArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.rows * a.C) return;
  uint r = lf_udiv((uint)t, a.C), c = (uint)t - r * a.C;
  float m = a.mask[r], ch = a.rawCharge ? a.charge[r] : asinh(a.charge[r]);
  float v = ((a.pos[r * 3] * a.Wpos[c]) + a.pos[r * 3 + 1] * a.Wpos[a.C + c]) + a.pos[r * 3 + 2] * a.Wpos[2 * a.C + c];
  if (a.bias) v += a.bias[c];
  v += m * a.Wmask[c];
  int z = a.element[r];
  if (z >= 0 && z < 128) v += a.Welem[z * a.C + c];
  v += ch * a.Wcharge[c];
  float name = 0.f;
  for (int k = 0; k < 4; ++k) {
    int code = a.nameChars[r * 4 + k];
    if (code >= 0 && code < 64) name += a.Wname[(k * 64 + code) * a.C + c];
  }
  a.act[t] = (v + name) * m;
}
kernel void af3_relu_h(LF_ARGS(ReluHArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.y[t] = (half)max(a.x[t], 0.f);
}
kernel void af3_atom_pair(LF_ARGS(AtomPairArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.subsets * a.queries * a.keys * a.Cp) return;
  ulong rest = t / a.Cp; uint c = (uint)(t - rest * a.Cp);
  uint key = (uint)(rest % a.keys); ulong qi = rest / a.keys;
  uint s = (uint)(qi / a.queries);
  ulong ki = (ulong)s * a.keys + key;
  float v = a.row[qi * a.Cp + c] + a.col[ki * a.Cp + c];
  bool valid = a.qUid[qi] == a.kUid[ki] && (!a.maskPadded || a.kMask[ki] != 0.f);
  float d0 = a.qPos[qi * 3] - a.kPos[ki * 3], d1 = a.qPos[qi * 3 + 1] - a.kPos[ki * 3 + 1], d2 = a.qPos[qi * 3 + 2] - a.kPos[ki * 3 + 2];
  float sq = d0 * d0 + d1 * d1 + d2 * d2;
  float off = d0 * a.Woff[c] + d1 * a.Woff[a.Cp + c] + d2 * a.Woff[2 * a.Cp + c];
  if (valid) v += (off + a.Wdist[c] / (1.f + sq)) + a.Wvalid[c];
  if (a.tp && a.tqMask[qi] != 0.f && a.tkMask[ki] != 0.f) v += a.tp[((ulong)a.tqIdx[qi] * a.tokens + a.tkIdx[ki]) * a.Cp + c];
  a.pair[t] = v;
}
kernel void af3_atom_logits(LF_ARGS(AtomLogitsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.subsets * a.heads * a.queries * a.keys) return;
  uint key = (uint)(t % a.keys); ulong rest = t / a.keys; uint q = (uint)(rest % a.queries); rest /= a.queries;
  uint h = (uint)(rest % a.heads), s = (uint)(rest / a.heads);
  a.out[t] = a.flat[(((ulong)s * a.queries + q) * a.keys + key) * a.nblocks * a.heads + a.block * a.heads + h];
}
// a simdgroup a row, eight a threadgroup; two-pass variance
kernel void af3_ada_ln(constant AdaLnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ulong row = ((ulong)tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  device const float* x = a.x + row * a.C;
  float s = 0.f;
  for (uint c = lane; c < a.C; c += 32) s += x[c];
  float mean = simd_sum(s) / a.C, v = 0.f;
  for (uint c = lane; c < a.C; c += 32) { float d = x[c] - mean; v += d * d; }
  float inv = rsqrt(simd_sum(v) / a.C + a.eps);
  ulong pr = (row % a.period) * a.ld;
  for (uint c = lane; c < a.C; c += 32) {
    float sc = a.scale[pr + c], y = (a.raw ? sc : lf_sigmoid(sc)) * ((x[c] - mean) * inv) + a.shift[pr + c];
    if (a.out) a.out[row * a.C + c] = (half)y;
    if (a.outF) a.outF[row * a.C + c] = y;
  }
}
kernel void af3_gather_rows(LF_ARGS(GatherRowsArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.chunks) return;
  ulong r = t / a.chunks, c = t - r * a.chunks, g = r % a.count, k = r / a.count;
  uint4 v = a.mask[g] != 0.f ? ((device const uint4*)a.in)[((ulong)a.idx[g] + k * a.srcRows) * a.chunks + c] : uint4(0);
  ((device uint4*)a.out)[t] = v;
}
// a threadgroup a (subset, head): the subset's keys and values staged, a simdgroup a query at a time - lanes over the
// keys for the scores and the softmax, then over the head's channels for P V; the gate on the way out
kernel void af3_atom_attention(constant AtomAttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  constexpr int MAXK = 128, MAXD = 32, NSG = 4;
  threadgroup half Ks[MAXK * (MAXD + 2)];
  threadgroup half Vs[MAXK * (MAXD + 2)];
  threadgroup float Qs[NSG][MAXD];
  threadgroup float Ps[NSG][MAXK];
  const uint s = tg.x, h = tg.y, D = a.D, LD = D + 2, Wd = a.heads * D, W2 = 2 * Wd, ss = s % a.subsets;
  for (uint t = tid; t < a.keys * D; t += 32 * NSG) {
    uint key = t / D, e = t - key * D;
    ulong row = (ulong)s * a.keys + key;
    Ks[key * LD + e] = a.kv[row * W2 + h * D + e];
    Vs[key * LD + e] = a.kv[row * W2 + Wd + h * D + e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = rsqrt((float)D);
  for (uint qi = sg; qi < a.queries; qi += NSG) {
    ulong qrow = (ulong)s * a.queries + qi;
    for (uint e = lane; e < D; e += 32) Qs[sg][e] = ((float)a.qg[qrow * W2 + h * D + e] + a.qBias[h * D + e]) * scale;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float qm = a.qMask[(ulong)ss * a.queries + qi];
    device const float* pl = a.logits + (((ulong)ss * a.heads + h) * a.queries + qi) * a.keys;
    float mx = -1e30f;
    for (uint key = lane; key < a.keys; key += 32) {
      float dot = 0.f;
      for (uint e = 0; e < D; ++e) dot += Qs[sg][e] * (float)Ks[key * LD + e];
      float km = a.kMask[(ulong)ss * a.keys + key];
      float maskBias = a.keyMasked ? -1e9f * ((1.f - qm) + (1.f - km)) : 1e9f * (qm - 1.f) * (km - 1.f);
      float l = dot + maskBias + pl[key];
      Ps[sg][key] = l; mx = max(mx, l);
    }
    mx = simd_max(mx);
    float sum = 0.f;
    for (uint key = lane; key < a.keys; key += 32) { float p = exp(Ps[sg][key] - mx); Ps[sg][key] = p; sum += p; }
    sum = simd_sum(sum);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.f / sum;
    for (uint e = lane; e < D; e += 32) {
      float acc = 0.f;
      for (uint key = 0; key < a.keys; ++key) acc += Ps[sg][key] * (float)Vs[key * LD + e];
      float g = (float)a.qg[qrow * W2 + Wd + h * D + e];
      a.out[qrow * Wd + h * D + e] = (half)(acc * inv * lf_sigmoid(g));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
kernel void af3_gated_residual(LF_ARGS(GatedResidualArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.act[t] += a.y[t] * lf_sigmoid(a.gate[t % a.period]);
}
kernel void af3_mul_h(LF_ARGS(MulHArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.x[t] = (half)((float)a.x[t] * (float)a.y[t]);
}
kernel void af3_encoder_start(LF_ARGS(EncoderStartArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  ulong q = t / a.C; uint c = (uint)(t - q * a.C);
  ulong gq = q % a.q1, k = q / a.q1;
  float v = 0.f;
  if (a.gmask[gq] != 0.f) {
    device const float* p = a.pos + ((ulong)a.idx[gq] + k * a.atoms) * 3;
    v = p[0] * a.Wp[c] + p[1] * a.Wp[a.C + c] + p[2] * a.Wp[2 * a.C + c];
  }
  a.act[t] = a.qStart[gq * a.C + c] + v * a.qMask[gq];
}
kernel void af3_aggregate(LF_ARGS(AggregateArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.Cp) return;
  ulong token = t / a.Cp; uint c = (uint)(t - token * a.Cp);
  ulong k = token / a.tokens, g0 = (token % a.tokens) * a.dense;
  float count = 0.f, sum = 0.f;
  for (uint at = 0; at < a.dense; ++at) {
    float m = a.atomMask[g0 + at];
    count += m;
    if (m != 0.f) {
      float v = a.gmask[g0 + at] != 0.f ? a.projected[((ulong)a.idx[g0 + at] + k * a.q1) * a.Cp + c] : 0.f;
      sum += max(v, 0.f);
    }
  }
  a.out[t] = count > 0.f ? sum / count : 0.f;
}
kernel void af3_target_feat(LF_ARGS(TargetFeatArgs)) {
  const uint width = 31 * 2 + 1 + 384;
  ulong t = LF_INDEX;
  if (t >= (ulong)a.tokens * width) return;
  uint token = lf_udiv((uint)t, width), c = (uint)t - token * width;
  float v;
  if (c < 31) v = a.aatype[token] == (int)c ? 1.f : 0.f;
  else if (c < 62) v = a.profile[token * 31 + (c - 31)];
  else if (c == 62) v = a.delMean[token];
  else v = a.atom[(ulong)token * 384 + (c - 63)];
  a.out[t] = v;
}
kernel void af3_target_feat_sum(LF_ARGS(TargetFeatSumArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.tokens * a.C) return;
  uint token = lf_udiv((uint)t, a.C), c = (uint)t - token * a.C;
  float v = 0.f;
  int aa = a.aatype[token];
  if (aa >= 0 && aa < 31) v += a.wRes[aa * a.C + c];
  for (int k = 0; k < 31; ++k) v += a.profile[token * 31 + k] * a.wProf[k * a.C + c];
  v += a.delMean[token] * a.wProf[31 * a.C + c];
  int mol = a.isLigand && a.isLigand[token] ? 3 : a.isRna && a.isRna[token] ? 2 : a.isDna && a.isDna[token] ? 1 : 0;
  v += a.wMol[mol * a.C + c];
  v += a.wMethod[1 * a.C + c];                  // x-ray diffraction, the default method
  v += a.wMod[(a.isModified && a.isModified[token] ? 1 : 0) * a.C + c];
  a.tf[t] += v;
}

// ---------------------------------------------------------------- the diffusion module
kernel void af3_pair_features(LF_ARGS(PairFeatArgs)) {
  const int maxIdx = 32, maxChain = 2, rel = (2 * maxIdx + 2) * 2 + 1 + (2 * maxChain + 2), positionBins = 2 * maxIdx + 2;
  uint width = a.Czt + rel;
  ulong t = LF_INDEX;
  if (t >= a.rows * width) return;
  ulong r = t / width; uint c = (uint)(t - r * width);
  ulong ij = a.p0 + r; uint i = (uint)(ij / a.n), j = (uint)(ij - (ulong)i * a.n);
  if (c < a.Czt) { a.out[t] = a.trunkPair[ij * a.Czt + c]; return; }
  int k = (int)c - (int)a.Czt;
  bool sameChain = a.r.asym[i] == a.r.asym[j], sameEntity = a.r.ent[i] == a.r.ent[j];
  int c0 = sameChain ? clamp(a.r.ri[i] - a.r.ri[j] + maxIdx, 0, 2 * maxIdx) : 2 * maxIdx + 1;
  bool sameResidue = sameChain && a.r.ri[i] == a.r.ri[j];
  int c1 = positionBins + (sameResidue ? clamp(a.r.ti[i] - a.r.ti[j] + maxIdx, 0, 2 * maxIdx) : 2 * maxIdx + 1);
  int c2 = positionBins * 2;
  int c3 = positionBins * 2 + 1 + (sameEntity ? clamp(a.r.sym[i] - a.r.sym[j] + maxChain, 0, 2 * maxChain) : 2 * maxChain + 1);
  a.out[t] = (k == c0 || k == c1 || k == c3 || (k == c2 && sameEntity)) ? 1.f : 0.f;
}
kernel void af3_concat_pad(LF_ARGS(ConcatPadArgs)) {
  uint w = a.wa + a.wb + (a.p0 >= 0) + (a.p1 >= 0);
  ulong t = LF_INDEX;
  if (t >= (ulong)a.rows * w) return;
  uint r = lf_udiv((uint)t, w); int c = (int)((uint)t - r * w);
  if (c == a.p0 || c == a.p1) { a.out[t] = 0.f; return; }
  int src = c - (a.p0 >= 0 && c > a.p0) - (a.p1 >= 0 && c > a.p1);
  a.out[t] = src < (int)a.wa ? a.a[(ulong)r * a.wa + src] : a.b[(ulong)r * a.wb + (src - a.wa)];
}
kernel void af3_fourier(LF_ARGS(FourierArgs)) {
  uint k = (uint)LF_INDEX;
  if (k >= a.n) return;
  float tr = 0.25f * log(a.level / 16.f);
  a.out[k] = cos(6.283185307179586f * (tr * a.w[k] + a.b[k]));
}
kernel void af3_fold_cond(LF_ARGS(FoldCondArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.Cc * a.C) return;
  uint k = lf_udiv(t, a.C), c = t - k * a.C;
  a.out[k * a.ld + a.off + c] = (half)((a.scale ? a.scale[k] : 1.f) * (float)a.w[t]);
}
kernel void af3_gated_res_strided(LF_ARGS(GatedResStridedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  uint r = lf_udiv((uint)t, a.C), c = (uint)t - r * a.C;
  a.act[t] += a.y[t] * lf_sigmoid(a.gate[(ulong)(r - lf_udiv(r, a.period) * a.period) * a.ld + c]);
}
kernel void af3_add_broadcast(LF_ARGS(AddBroadcastArgs)) {
  ulong t = LF_INDEX;
  if (t < a.total) a.x[t] += a.v[t % a.n];
}
kernel void af3_broadcast_skip(LF_ARGS(BroadcastSkipArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  ulong q = t / a.C; uint c = (uint)(t - q * a.C);
  ulong gq = q % a.q1, k = q / a.q1;
  float v = a.gmask[gq] != 0.f ? a.proj[((ulong)a.idx[gq] / a.dense + k * a.tokens) * a.C + c] : 0.f;
  a.act[t] = (v + a.skip[t]) * a.qMask[gq];
}
kernel void af3_mask_ln_project3(constant MaskLnProject3Args& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  ulong row = ((ulong)tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  device const float* x = a.act + row * a.C;
  float m = a.qMask[row % a.q1], s = 0.f;
  for (uint c = lane; c < a.C; c += 32) s += x[c] * m;
  float mean = simd_sum(s) / a.C, v = 0.f;
  for (uint c = lane; c < a.C; c += 32) { float d = x[c] * m - mean; v += d * d; }
  float inv = rsqrt(simd_sum(v) / a.C + 1e-5f);
  float y0 = 0.f, y1 = 0.f, y2 = 0.f;
  for (uint c = lane; c < a.C; c += 32) {
    float l = (x[c] * m - mean) * inv * a.scale[c] + (a.offset ? a.offset[c] : 0.f);
    y0 += l * a.Wp[c * 3]; y1 += l * a.Wp[c * 3 + 1]; y2 += l * a.Wp[c * 3 + 2];
  }
  y0 = simd_sum(y0); y1 = simd_sum(y1); y2 = simd_sum(y2);
  if (lane == 0) { a.upd[row * 3] = y0; a.upd[row * 3 + 1] = y1; a.upd[row * 3 + 2] = y2; }
}
kernel void af3_scale_positions(LF_ARGS(ScalePosArgs)) {
  ulong t = LF_INDEX;
  if (t < a.total * 3) a.y[t] = a.x[t] * a.mask[(t / 3) % a.atoms] * a.in;
}
kernel void af3_denoise_out(LF_ARGS(DenoiseOutArgs)) {
  ulong t = LF_INDEX;
  if (t < a.total * 3) a.o[t] = (a.skip * a.x[t] + a.out * a.upd[t]) * a.mask[(t / 3) % a.atoms];
}

// ---------------------------------------------------------------- the sampler
// a standard normal from (seed, step, index): Box-Muller on two 32-bit uniforms of a counter hash (cuda/af3's)
inline ulong lf_mix64(ulong z) {
  z += 0x9e3779b97f4a7c15ul;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ul;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebul;
  return z ^ (z >> 31);
}
inline float lf_gaussian(ulong seed, uint step, ulong index) {
  ulong h = lf_mix64(seed * 0x2545f4914f6cdd1dul ^ lf_mix64(((ulong)step << 40) ^ index));
  float u1 = ((float)(uint)h + 1.f) * 2.3283064e-10f, u2 = (float)(uint)(h >> 32) * 2.3283064e-10f;
  return sqrt(-2.f * log(u1)) * cospi(2.f * u2);
}
kernel void af3_initial_noise(LF_ARGS(InitNoiseArgs)) {
  ulong i = LF_INDEX;
  if (i < a.total) a.x[i] = a.scale * lf_gaussian(a.seeds[i / a.n3], 0, i % a.n3);
}
// a threadgroup of 1024 a sample, a fixed reduction order
kernel void af3_centroid(constant CentroidArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4][32];
  device const float* xs = a.x + (ulong)tg.x * a.atoms * 3;
  float s[4] = {0, 0, 0, 0};
  for (ulong i = tid; i < a.atoms; i += 1024)
    if (a.mask[i] != 0.f) { s[0] += xs[i * 3]; s[1] += xs[i * 3 + 1]; s[2] += xs[i * 3 + 2]; s[3] += 1.f; }
  for (int k = 0; k < 4; ++k) { float v = simd_sum(s[k]); if (lane == 0) part[k][sg] = v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v[4];
    for (int k = 0; k < 4; ++k) v[k] = simd_sum(part[k][lane]);
    if (lane < 3) a.c[tg.x * 3 + lane] = v[lane] / (v[3] + 1e-6f);
  }
}
kernel void af3_augment(LF_ARGS(AugmentArgs)) {
  ulong at = LF_INDEX;
  if (at >= a.total) return;
  ulong k = at / a.atoms;
  device const float* r = a.rot + k * 12; device const float* ck = a.c + k * 3;
  float p[3] = {a.x[at * 3] - ck[0], a.x[at * 3 + 1] - ck[1], a.x[at * 3 + 2] - ck[2]};
  bool live = a.mask[at % a.atoms] != 0.f;
  for (int d = 0; d < 3; ++d) {
    float v = live ? p[0] * r[d] + p[1] * r[3 + d] + p[2] * r[6 + d] + r[9 + d] : 0.f;
    a.x[at * 3 + d] = v;
    a.noisy[at * 3 + d] = v + a.injected * lf_gaussian(a.seeds[k], a.step, (at % a.atoms) * 3 + d);
  }
}
kernel void af3_euler(LF_ARGS(EulerArgs)) {
  ulong i = LF_INDEX;
  if (i < a.n) a.x[i] = a.noisy[i] + a.scale * (a.noisy[i] - a.den[i]);
}

// ---------------------------------------------------------------- the confidence head
kernel void af3_conf_bin(LF_ARGS(ConfBinArgs)) {
  uint ij = (uint)LF_INDEX;
  if (ij >= a.n * a.n) return;
  uint i = lf_udiv(ij, a.n), j = ij - i * a.n;
  float sq = 0.f;
  for (int k = 0; k < 3; ++k) { float d = a.beta[i * 3 + k] - a.beta[j * 3 + k]; sq += d * d; }
  int bin = -1;
  if (a.chaiBins) {     // chai-1's: 16 bins, how many of 15 evenly spaced bounds from 3.375 to 21.375 the distance is past
    float distance = sqrt(sq + 1e-10f);
    bin = 0;
    for (uint at = 0; at < a.bins - 1; ++at) bin += distance > 3.375f + at * (18.f / (a.bins - 2));
  } else if (a.caBins) {       // rf3's: how many of `bins - 1` evenly spaced bounds the distance is past
    float distance = sqrt(sq + 1e-10f);
    bin = 0;
    for (uint at = 0; at < a.bins - 1; ++at) if (distance > a.dmin + at * ((a.dmax - a.dmin) / (a.bins - 1))) ++bin;
  } else {
    for (uint b = 0; b < a.bins; ++b) {
      float lo = a.dmin + (a.dmax - a.dmin) * b / (a.bins - 1), hi = a.dmin + (a.dmax - a.dmin) * (b + 1) / (a.bins - 1);
      float lower = lo * lo, upper = b + 1 < a.bins ? hi * hi : 1e8f;
      if (sq > lower && sq < upper) { bin = (int)b; break; }
    }
  }
  a.bin[ij] = bin; a.sq[ij] = sq;
}
kernel void af3_conf_pair_init(LF_ARGS(ConfPairInitArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  float v = a.left[j * a.C + c] + a.right[i * a.C + c];
  int bin = a.bin[ij];
  if (bin >= 0) v += a.Wd[bin * a.C + c] * (a.unmasked ? 1.f : a.pairMask[ij]);
  if (!a.caBins && a.Wdist) v += sqrt(a.sq[ij] + 1e-10f) * a.Wdist[c];
  a.pair[t] += v;
}
kernel void af3_expectation(LF_ARGS(ExpectationArgs)) {
  ulong r = LF_INDEX;
  if (r >= a.rows) return;
  ulong rt = r;
  if (a.symmetricN) { ulong i = r / a.symmetricN, j = r - i * a.symmetricN; rt = j * a.symmetricN + i; }
  float mx = -1e30f;
  for (uint b = 0; b < a.bins; ++b) mx = max(mx, a.logits[r * a.bins + b] + (a.symmetricN ? a.logits[rt * a.bins + b] : 0.f));
  float total = 0.f, weighted = 0.f;
  for (uint b = 0; b < a.bins; ++b) {
    float p = exp(a.logits[r * a.bins + b] + (a.symmetricN ? a.logits[rt * a.bins + b] : 0.f) - mx);
    total += p; weighted += p * a.centres[b];
  }
  a.out[r] = weighted / total * a.scale * (a.mask ? a.mask[r] : 1.f);
}
kernel void af3_inter_chain(LF_ARGS(InterChainArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.n * a.bins) return;
  uint ij = lf_udiv((uint)t, a.bins), i = lf_udiv(ij, a.n), j = ij - i * a.n;
  if (a.asym[i] != a.asym[j]) a.logits[t] = a.inter[t];
}
kernel void af3_clamp(LF_ARGS(ClampArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.x[t] = clamp(a.x[t], -a.limit, a.limit);
}
// The same attention on simdgroup matrices, for AF3's shape (32 queries, 128 keys, a head 16 or 32 wide): a threadgroup
// a (subset, head), a simdgroup eight queries - S = Q K^T as 8 x 8 matrices (half in, float accumulated), the pair
// logits and the mask added in place, the softmax over the 128 keys in registers (log2 domain), O = P V, the gate on
// the way out. A lane holds row sm of each 8 x 8 matrix, columns sn and sn + 1; a row's four lanes differ in lane
// bits 0 and 3 (metal/core's flash attention's layout).
template <int D>
kernel void af3_atom_attention_mma(constant AtomAttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  constexpr int Q = 32, K = 128, LD = D + 8, DF = D / 8, KF = K / 8, D8 = D / 8;
  threadgroup half Qs[Q * LD];
  threadgroup half Ks[K * LD];
  threadgroup half Vs[K * LD];
  const uint s = tg.x, h = tg.y, ss = s % a.subsets, Wd = a.heads * D, W2 = 2 * Wd;
  const float qs = rsqrt((float)D) * M_LOG2E_F;
  for (uint e = tid; e < Q * D8; e += 128) {
    uint qi = e / D8, d = (e - qi * D8) * 8;
    device const half* src = a.qg + ((ulong)s * Q + qi) * W2 + h * D + d;
    float4 f0 = float4(*(device const half4*)src) + *(device const float4*)(a.qBias + h * D + d);
    float4 f1 = float4(*(device const half4*)(src + 4)) + *(device const float4*)(a.qBias + h * D + d + 4);
    *(threadgroup half4*)(Qs + qi * LD + d) = half4(f0 * qs);
    *(threadgroup half4*)(Qs + qi * LD + d + 4) = half4(f1 * qs);
  }
  for (uint e = tid; e < K * D8; e += 128) {
    uint kj = e / D8, d = (e - kj * D8) * 8;
    device const half* src = a.kv + ((ulong)s * K + kj) * W2 + h * D + d;
    *(threadgroup half4*)(Ks + kj * LD + d) = *(device const half4*)src;
    *(threadgroup half4*)(Ks + kj * LD + d + 4) = *(device const half4*)(src + 4);
    *(threadgroup half4*)(Vs + kj * LD + d) = *(device const half4*)(src + Wd);
    *(threadgroup half4*)(Vs + kj * LD + d + 4) = *(device const half4*)(src + Wd + 4);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[DF];
  _Pragma("clang loop unroll(full)") for (int dk = 0; dk < DF; ++dk) simdgroup_load(qf[dk], Qs + (sg * 8) * LD + dk * 8, LD);
  simdgroup_float8x8 sc[KF];
  _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j) sc[j] = simdgroup_float8x8(0);
  _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j)
    _Pragma("clang loop unroll(full)") for (int dk = 0; dk < DF; ++dk) {
      simdgroup_half8x8 kt;
      simdgroup_load(kt, Ks + (j * 8) * LD + dk * 8, LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(sc[j], qf[dk], kt, sc[j]);
    }
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  const uint q = sg * 8 + sm;
  const float qm = a.qMask[(ulong)ss * Q + q];
  device const float* pl = a.logits + (((ulong)ss * a.heads + h) * Q + q) * K;
  device const float* km = a.kMask + (ulong)ss * K;
  float mx = -1e30f;
  _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j) {
    thread auto& e = sc[j].thread_elements();
    _Pragma("clang loop unroll(full)") for (int t = 0; t < 2; ++t) {
      const int k = j * 8 + sn + t;
      float kk = km[k];
      float maskBias = a.keyMasked ? -1e9f * ((1.f - qm) + (1.f - kk)) : 1e9f * (qm - 1.f) * (kk - 1.f);
      e[t] += (pl[k] + maskBias) * M_LOG2E_F;
      mx = max(mx, (float)e[t]);
    }
  }
  mx = max(mx, simd_shuffle_xor(mx, (ushort)1));
  mx = max(mx, simd_shuffle_xor(mx, (ushort)8));
  simdgroup_half8x8 p[KF];
  float sum = 0.f;
  _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j) {
    thread auto& e = sc[j].thread_elements();
    thread auto& pe = p[j].thread_elements();
    _Pragma("clang loop unroll(full)") for (int t = 0; t < 2; ++t) { float v = exp2(e[t] - mx); sum += v; pe[t] = (half)v; }
  }
  sum += simd_shuffle_xor(sum, (ushort)1);
  sum += simd_shuffle_xor(sum, (ushort)8);
  simdgroup_float8x8 o[DF];
  _Pragma("clang loop unroll(full)") for (int c = 0; c < DF; ++c) o[c] = simdgroup_float8x8(0);
  _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j)
    _Pragma("clang loop unroll(full)") for (int c = 0; c < DF; ++c) {
      simdgroup_half8x8 vf;
      simdgroup_load(vf, Vs + (j * 8) * LD + c * 8, LD);
      simdgroup_multiply_accumulate(o[c], p[j], vf, o[c]);
    }
  const ulong qrow = (ulong)s * Q + q;
  device const half* g = a.qg + qrow * W2 + Wd + h * D;
  device half* out = a.out + qrow * Wd + h * D;
  const float inv = 1.f / sum;
  _Pragma("clang loop unroll(full)") for (int c = 0; c < DF; ++c) {
    thread auto& oe = o[c].thread_elements();
    const int d = c * 8 + sn;
    half2 gv = *(device const half2*)(g + d);
    *(device half2*)(out + d) = half2(half(oe[0] * inv * lf_sigmoid((float)gv.x)), half(oe[1] * inv * lf_sigmoid((float)gv.y)));
  }
}
template [[host_name("af3_atom_attention_mma16")]] kernel void af3_atom_attention_mma<16>(constant AtomAttnArgs&, uint3, uint, uint, uint);
template [[host_name("af3_atom_attention_mma32")]] kernel void af3_atom_attention_mma<32>(constant AtomAttnArgs&, uint3, uint, uint, uint);
kernel void af3_reembed_bin(LF_ARGS(ReembedBinArgs)) {
  uint ij = (uint)LF_INDEX;
  if (ij >= a.n * a.n) return;
  uint i = lf_udiv(ij, a.n), j = ij - i * a.n;
  float sq = 1e-10f;
  for (int k = 0; k < 3; ++k) { float d = a.beta[i * 3 + k] - a.beta[j * 3 + k]; sq += d * d; }
  float distance = sqrt(sq);
  int bin = 0;
  for (int e = 0; e < 63; ++e) if (distance > 2.f + 20.f * e / 62) ++bin;
  a.bin[ij] = bin;
}
kernel void af3_reembed_pair(LF_ARGS(ReembedPairArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  int o = a.orders ? (int)a.orders[ij] : 0;
  if (o < 0 || o >= 7) o = 0;
  a.pair[t] += a.right[i * a.C + c] + a.left[j * a.C + c] + a.Wd[a.bin[ij] * a.C + c] * a.pairMask[ij] +
               (a.bonds ? a.bonds[ij] * a.wBond[c] : 0.f) + a.wBondType[o * a.C + c] + a.unspecified[c];
}
kernel void af3_outer_prod(LF_ARGS(OuterProdArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.rows * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), e = (uint)t - ij * a.C, ii = lf_udiv(ij, a.n), j = ij - ii * a.n;
  a.out[t] = (half)(a.a[(a.i0 + ii) * a.C + e] * a.b[j * a.C + e]);
}

// ---------------------------------------------------------------- rosettafold3
kernel void af3_kq_norm(constant KqNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ulong row = ((ulong)tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  for (int side = 0; side < 2; ++side) {
    device half* x = side ? a.k + row * a.ldk : a.q + row * a.ldq;
    device const float* sc = side ? a.ks : a.qs; device const float* of = side ? a.ko : a.qo;
    float s = 0.f;
    for (uint c = lane; c < a.Wd; c += 32) s += (float)x[c] + (side || !a.qBias ? 0.f : a.qBias[c]);
    float mean = simd_sum(s) / a.Wd, v = 0.f;
    for (uint c = lane; c < a.Wd; c += 32) { float d = (float)x[c] + (side || !a.qBias ? 0.f : a.qBias[c]) - mean; v += d * d; }
    float inv = rsqrt(simd_sum(v) / a.Wd + 1e-5f);
    for (uint c = lane; c < a.Wd; c += 32)
      x[c] = (half)(((float)x[c] + (side || !a.qBias ? 0.f : a.qBias[c]) - mean) * inv * sc[c] + of[c]);
  }
}
// (float, where the reference's is double - Metal has none; the step is 1e-3 rather than 1e-4 for it)
inline float lf_improper(float3 a, float3 b, float3 c, float3 d) {
  const float eps = 1e-6f;
  float3 b0 = a - b, b1 = c - b, b2 = d - c;
  float3 n = b1 / (length(b1) + eps);
  float3 v = b0 - dot(b0, n) * n, w = b2 - dot(b2, n) * n;
  return atan2(dot(cross(n, v), w) + eps, dot(v, w) + eps);
}
kernel void af3_chiral_grad(LF_ARGS(ChiralGradArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.atoms * a.ns) return;
  ulong k = t / a.atoms, atom = t - k * a.atoms;
  device const float* x = a.positions + k * a.atoms * 3;
  float3 g = float3(0.f);
  const float step = 1e-3f;
  for (int e = a.offsets[atom]; e < a.offsets[atom + 1]; ++e) {
    int centre = a.entries[e] >> 2, corner = a.entries[e] & 3;
    float ideal = a.angles[centre];
    if (ideal == 0.f) continue;
    float3 p[4];
    for (int q = 0; q < 4; ++q) { ulong at = (ulong)a.centers[centre * 4 + q] * 3; p[q] = float3(x[at], x[at + 1], x[at + 2]); }
    for (int d = 0; d < 3; ++d) {
      float keep = p[corner][d];
      p[corner][d] = keep + step; float up = lf_improper(p[0], p[1], p[2], p[3]) - ideal;
      p[corner][d] = keep - step; float down = lf_improper(p[0], p[1], p[2], p[3]) - ideal;
      p[corner][d] = keep;
      float derivative = (up * up - down * down) / (2.f * step);
      if (isfinite(derivative)) g[d] += derivative;
    }
  }
  a.grads[t * 3] = g.x; a.grads[t * 3 + 1] = g.y; a.grads[t * 3 + 2] = g.z;
}
kernel void af3_masked_sum(constant MaskedSumArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[2][8];
  float acc = 0.f, live = 0.f;
  ulong total = a.rows * a.C;
  for (ulong i = (ulong)tg.x * 256 + tid; i < total; i += (ulong)ng.x * 256) {
    ulong r = i / a.C;
    if (!(a.mask[r] > 0.f)) continue;
    float v = a.x[i];
    if (a.pass) { v -= a.stat[0]; v *= v; }
    acc += v;
    if (i - r * a.C == 0) live += 1.f;
  }
  acc = simd_sum(acc); live = simd_sum(live);
  if (lane == 0) { red[0][sg] = acc; red[1][sg] = live; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float s = 0.f, l = 0.f;
    for (int k = 0; k < 8; ++k) { s += red[0][k]; l += red[1][k]; }
    a.partial[tg.x * 2] = s; a.partial[tg.x * 2 + 1] = l;
  }
}
kernel void af3_global_stat(constant GlobalStatArgs& a [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) {
  if (tid) return;
  float total = 0.f, live = 0.f;
  for (uint k = 0; k < a.parts; ++k) { total += a.partial[k * 2]; live += a.partial[k * 2 + 1]; }
  float count = max(live * a.vendorWidth, 1.f);
  if (a.pass == 0) a.stat[0] = total / count;
  else {
    float variance = total + (float)(a.vendorWidth - a.C) * live * a.stat[0] * a.stat[0];
    a.stat[1] = rsqrt(variance / count + 1e-5f);
  }
}
kernel void af3_apply_norm(LF_ARGS(ApplyNormArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.x[t] = (a.x[t] - a.stat[0]) * a.stat[1];
}

// ---------------------------------------------------------------- OpenDDE's structural tokens
kernel void af3_gather_parent(LF_ARGS(GatherParentArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.C) return;
  uint i = lf_udiv((uint)t, a.C), c = (uint)t - i * a.C;
  a.out[t] = a.src[(ulong)a.parent[i] * a.C + c] + (a.roleEmb ? a.roleEmb[a.role[i] * a.C + c] : 0.f);
}
kernel void af3_silu(LF_ARGS(SiluInPlaceArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.x[t] = lf_silu(a.x[t]);
}
kernel void af3_single_struct(LF_ARGS(SingleStructArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.C) return;
  uint i = lf_udiv((uint)t, a.C), c = (uint)t - i * a.C;
  a.out[t] = a.a[t] + a.b[t] + a.roleEmb[a.role[i] * a.C + c];
}
kernel void af3_gather_pair_sorted(LF_ARGS(GatherPairSortedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  ulong r = t / a.C; uint c = (uint)(t - r * a.C);
  int ij = a.order[r], i = ij / (int)a.n, j = ij - i * (int)a.n;
  a.out[t] = (half)a.pair[((ulong)a.parent[i] * a.nRes + a.parent[j]) * a.C + c];
}
kernel void af3_scatter_pair(LF_ARGS(ScatterPairArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  ulong r = t / a.C; uint c = (uint)(t - r * a.C);
  int ij = a.order[r], i = ij / (int)a.n, j = ij - i * (int)a.n;
  a.pair[(ulong)ij * a.C + c] = a.trunkPair[((ulong)a.parent[i] * a.nRes + a.parent[j]) * a.C + c] + a.projected[t] +
                                a.eSame[a.sameParent[ij] * a.C + c] + a.eTwin[a.twin[ij] * a.C + c] + a.ePrev[a.prev[ij] * a.C + c] +
                                a.eNext[a.next[ij] * a.C + c] + a.eType[a.type[ij] * a.C + c];
}
kernel void af3_attn_bias(LF_ARGS(AttnBiasArgs)) {
  ulong ij = LF_INDEX;
  if (ij >= a.pairs) return;
  a.bias[ij] = a.bSame[0] * a.sameParent[ij] + a.bTwin[0] * a.twin[ij] + a.bPrev[0] * a.prev[ij] + a.bNext[0] * a.next[ij] + a.bType[a.type[ij]];
}
kernel void af3_add_bias_heads(LF_ARGS(AddBiasHeadsArgs)) {
  ulong t = LF_INDEX;
  if (t < a.pairs * a.heads) a.raw[t] += a.bias[t / a.heads];
}
kernel void af3_dde_pair_init(LF_ARGS(DdePairInitArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  float sq = 0.f;
  for (int k = 0; k < 3; ++k) { float d = a.coords[i * 3 + k] - a.coords[j * 3 + k]; sq += d * d; }
  float distance = sqrt(max(1e-10f, sq));
  int bin = (int)floor((distance - 3.25f) / 1.25f);
  if (distance < 3.25f) bin = -1;
  if (bin >= (int)a.bins) bin = a.bins - 1;
  a.pair[t] += a.s1[j * a.C + c] + a.s2[i * a.C + c] + (bin >= 0 ? a.Wd[bin * a.C + c] : 0.f) + distance * a.Wraw[c];
}
kernel void af3_slot_major(LF_ARGS(SlotMajorArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.slots * a.C * a.bins) return;
  uint b = (uint)(t % a.bins); ulong r = t / a.bins; uint c = (uint)(r % a.C), slot = (uint)(r / a.C);
  a.out[(ulong)c * a.slots * a.bins + slot * a.bins + b] = (half)a.w[t];
}

// ---------------------------------------------------------------- ESM2 3B
kernel void af3_expand8(LF_ARGS(Expand8Args)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.rows * a.out) return;
  uint row = lf_udiv((uint)t, a.out), col = (uint)t - row * a.out;
  device const uchar* s = a.scales + 4 * ((ulong)lf_udiv(row, a.g) * a.out + col);
  float scale = as_type<float>((uint)s[0] | ((uint)s[1] << 8) | ((uint)s[2] << 16) | ((uint)s[3] << 24));
  a.w[(ulong)row * a.ld + a.col0 + col] = (half)((float)(char)a.codes[t] * scale);
}
kernel void af3_esm_embed(LF_ARGS(EsmEmbedArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.rows * a.C) return;
  uint r = lf_udiv((uint)t, a.C), c = (uint)t - r * a.C;
  a.x[t] = a.table[(ulong)a.ids[r] * a.C + c] * a.scale;
}
kernel void af3_esm_pack(LF_ARGS(EsmPackArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.rows * a.C) return;
  uint r = lf_udiv((uint)t, a.C), c = (uint)t - r * a.C, d = c & 63;
  device const float* row = a.qkv + (ulong)r * 3 * a.C;
  device half* out = a.qkvg + (ulong)r * 4 * a.C;
  for (int which = 0; which < 2; ++which) {       // q and k: rotated
    float x = row[which * a.C + c], y;
    float co = a.cosT[r * 32 + (d & 31)], si = a.sinT[r * 32 + (d & 31)];
    if (d < 32) { y = row[which * a.C + c + 32]; out[which * a.C + c] = (half)(x * co - y * si); }
    else { y = row[which * a.C + c - 32]; out[which * a.C + c] = (half)(y * si + x * co); }
  }
  out[2 * a.C + c] = (half)row[2 * a.C + c];
  out[3 * a.C + c] = (half)30.f;                  // (sigmoid(30) is 1 to within 1e-13: no gate)
}
kernel void af3_gather_esm(LF_ARGS(GatherEsmArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.tokens * a.E) return;
  uint token = lf_udiv((uint)t, a.E), e = (uint)t - token * a.E;
  int r = a.tokenRow[token];
  a.out[t] = r < 0 ? 0.f : a.rows[(ulong)r * a.E + e];
}

// ---------------------------------------------------------------- chai-1
kernel void af3_chai_relenc(LF_ARGS(ChaiRelEncArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  bool sameChain = a.r.asym[i] == a.r.asym[j];
  int rss = sameChain ? clamp(a.r.ri[i] - a.r.ri[j] + 33, 0, 65) : 66;
  int rts = sameChain && a.r.ri[i] == a.r.ri[j] ? clamp(a.r.ti[i] - a.r.ti[j] + 32, 0, 65) : 66;
  a.pair[t] += a.bias[c] + a.W[rss * a.C + c] + a.W[(67 + rts) * a.C + c];
}
kernel void af3_chai_msa_embed(LF_ARGS(ChaiMsaEmbedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.count * a.C) return;
  uint row = lf_udiv((uint)t, a.C), c = (uint)t - row * a.C, s = lf_udiv(row, a.n), token = row - s * a.n;
  int first = -1; bool paired = false;           // (the row covers tokens of more than one chain)
  for (uint k = 0; k < a.n && !paired; ++k)
    if (a.msaMask[s * a.n + k] != 0.f) { if (first < 0) first = a.asym[k]; else if (a.asym[k] != first) paired = true; }
  float d = a.del[row];
  int code = a.rows[row];
  if (a.isLigand && a.isLigand[token]) code = s == 0 ? 20 : 31;      // (chai's query row: unknown, the rest: its mask class)
  float v = a.bias[c] + (paired ? a.W[c] : 0.f) + a.W[(1 + (s == 0 ? 4 : 2)) * a.C + c] +
            atan(d / 3.f) * (2.f / M_PI_F) * a.W[7 * a.C + c] + clamp(d, 0.f, 1.f) * a.W[8 * a.C + c] +
            (code >= 0 && code < 32 ? a.W[(9 + code) * a.C + c] : 0.f);
  a.msa[t] = v + a.fromSingle[token * a.C + c];
}
kernel void af3_group_major(LF_ARGS(GroupMajorArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.S * a.n * a.G * a.K) return;
  uint k = (uint)(t % a.K); ulong r = t / a.K; uint i = (uint)(r % a.n); r /= a.n; uint s = (uint)(r % a.S), g = (uint)(r / a.S);
  a.out[t] = a.x[(((ulong)s * a.n + i) * a.G + g) * a.K + k];
}
kernel void af3_grouped_permute(LF_ARGS(GroupedPermuteArgs)) {
  ulong t = LF_INDEX;
  ulong per = (ulong)a.G * a.K * a.K;
  if (t >= (ulong)a.bi * a.n * per) return;
  uint l = (uint)(t % a.K); ulong r = t / a.K; uint k = (uint)(r % a.K); r /= a.K; uint g = (uint)(r % a.G); r /= a.G;
  uint j = (uint)(r % a.n), i = (uint)(r / a.n);
  a.out[t] = a.P[(ulong)g * ((ulong)a.bi * a.K * a.n * a.K) + (((ulong)i * a.K + k) * a.n + j) * a.K + l];
}
kernel void af3_scale_h(LF_ARGS(ScaleHArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.y[t] = a.s * (float)a.x[t];
}
kernel void af3_chai_atom_pair(LF_ARGS(ChaiAtomPairArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.subsets * a.queries * a.keys * a.Cp) return;
  ulong rest = t / a.Cp; uint c = (uint)(t - rest * a.Cp);
  uint key = (uint)(rest % a.keys); ulong qi = rest / a.keys;
  uint s = (uint)(qi / a.queries);
  ulong ki = (ulong)s * a.keys + key;
  float v = a.row[qi * a.Cp + c] + a.col[ki * a.Cp + c];
  if (a.tp && a.tqMask[qi] != 0.f && a.tkMask[ki] != 0.f) v += a.tp[((ulong)a.tqIdx[qi] * a.tokens + a.tkIdx[ki]) * a.Cp + c];
  bool valid = a.qUid[qi] == a.kUid[ki];
  float d0 = a.qPos[qi * 3] - a.kPos[ki * 3], d1 = a.qPos[qi * 3 + 1] - a.kPos[ki * 3 + 1], d2 = a.qPos[qi * 3 + 2] - a.kPos[ki * 3 + 2];
  float sq = d0 * d0 + d1 * d1 + d2 * d2;
  const float edges[10] = {0.f, 1.f, 4.f, 9.f, 16.f, 25.f, 36.f, 64.f, 144.f, 256.f};
  int idx = 0;
  for (int e = 0; e < 10; ++e) idx += sq > edges[e];
  if (!valid) idx = 11;
  v += a.bf[c] + a.Wf[idx * a.Cp + c] + a.Wf[12 * a.Cp + c] / (1.f + sq) + (valid ? a.Wf[13 * a.Cp + c] : 0.f);
  a.pair[t] = v;
}
kernel void af3_same_ref_mask(LF_ARGS(SameRefMaskArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.subsets * a.heads * a.queries * a.keys) return;
  uint key = (uint)(t % a.keys); ulong rest = t / a.keys; uint q = (uint)(rest % a.queries); rest /= a.queries;
  uint s = (uint)(rest / a.heads);
  ulong qi = (ulong)s * a.queries + q, ki = (ulong)s * a.keys + key;
  if (!(a.tqMask[qi] != 0.f && a.tkMask[ki] != 0.f && a.qUid[qi] == a.kUid[ki])) a.pl[t] = -1e9f;
}
kernel void af3_add_const(LF_ARGS(AddConstArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.x[t] += a.v;
}
kernel void af3_chai_token_feat(LF_ARGS(ChaiTokenFeatArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.tokens * a.C) return;
  uint token = lf_udiv((uint)t, a.C), c = (uint)t - token * a.C;
  int aa = a.aatype[token];
  float v = a.bt[c] + (aa >= 0 && aa < 31 ? a.Wt[aa * a.C + c] : 0.f);
  for (int k = 0; k < 31; ++k) v += a.profile[token * 31 + k] * a.Wp[k * a.C + c];
  v += a.delMean[token] * a.Wp[31 * a.C + c];
  a.out[t] = v;
}
kernel void af3_chai_struct_pair(LF_ARGS(ChaiStructPairArgs)) {
  uint width = a.Czt + a.Cz;
  ulong t = LF_INDEX;
  if (t >= a.rows * width) return;
  ulong row = t / width; uint c = (uint)(t - row * width);
  ulong ij = a.p0 + row;
  if (c < a.Czt) { a.out[t] = a.trunkPair[ij * a.Czt + c]; return; }
  c -= a.Czt;
  uint i = (uint)(ij / a.n), j = (uint)(ij - (ulong)i * a.n);
  bool sameChain = a.asym[i] == a.asym[j];
  int rss = sameChain ? clamp(a.ri[i] - a.ri[j] + 33, 0, 65) : 66;
  int rts = sameChain && a.ri[i] == a.ri[j] ? clamp(a.ti[i] - a.ti[j] + 32, 0, 65) : 66;
  int relEntity = a.entityRank[i] - a.entityRank[j];
  int rchain = relEntity != 0 ? 5 : clamp(a.symRank[i] - a.symRank[j] + 2, 0, 4);
  int rent = clamp(relEntity + 1, 0, 2);
  float v = a.bias[c] + a.Wp[5 * a.Cz + c] + a.Wp[155 * a.Cz + c] + a.Wp[162 * a.Cz + c] + a.Wp[(6 + rchain) * a.Cz + c] +
            a.Wp[(12 + rent) * a.Cz + c] + a.Wp[(15 + rss) * a.Cz + c] + a.Wp[(82 + rts) * a.Cz + c];
  if (a.bonds) v += a.bonds[ij] * a.Wb[c];
  a.out[t] = v;
}
kernel void af3_chai_euler(LF_ARGS(ChaiEulerArgs)) {
  ulong i = LF_INDEX;
  if (i >= a.n) return;
  float g = (a.noisy[i] - a.d1[i]) / a.tHat;
  a.g1[i] = g; a.x[i] = a.noisy[i] + a.dt * g;
}
kernel void af3_chai_correct(LF_ARGS(ChaiCorrectArgs)) {
  ulong i = LF_INDEX;
  if (i < a.n) a.x[i] += a.dt * ((a.x[i] - a.d2[i]) / a.level + a.g1[i]) * 0.5f;
}
kernel void af3_plddt37(LF_ARGS(Plddt37Args)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.dense * a.bins) return;
  uint b = (uint)(t % a.bins); ulong slot = t / a.bins; ulong token = slot / a.dense;
  a.out[t] = a.p37[(token * 37 + a.idx[slot]) * a.bins + b];
}
