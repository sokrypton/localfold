// The AlphaFold 3 lineage's kernels (metal/af3): derived weights, the embedder's features, the attention biases, the
// MSA stack's pair-weighted averaging and outer product, the distogram. cuda/af3/src is the reading.

// ---------------------------------------------------------------- derived weights
kernel void af3_concat_part(LF_ARGS(ConcatPartArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.C * a.width) return;
  uint c = lf_udiv(t, a.width), o = t - c * a.width;
  a.dst[c * a.total + a.off + o] = a.transposed ? a.src[o * a.C + c] : a.src[t];
}
kernel void af3_affine_h(LF_ARGS(AffineHArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n) return;
  // (32-bit: a transition chunk's rows x C stay under 2^32 - a 64-bit modulo is the GPU's slow division)
  const uint u = (uint)t, c = u - lf_udiv(u, a.C) * a.C;
  a.y[t] = (half)((float)a.x[t] * a.scale[c] + (a.offset ? a.offset[c] : 0.f));
}
kernel void af3_fold_rows(LF_ARGS(FoldRowsArgs)) {
  uint t = (uint)LF_INDEX;
  if ((a.out_ & 3) == 0) {
    uint out4 = a.out_ >> 2;
    if (t >= a.in * out4) return;
    uint k = lf_udiv(t, out4);
    ((device half4*)a.out)[t] = half4(a.scale[k] * float4(((device const half4*)a.w)[t]));
    return;
  }
  if (t >= a.in * a.out_) return;
  uint k = lf_udiv(t, a.out_);
  a.out[t] = (half)(a.scale[k] * (float)a.w[t]);
}
kernel void af3_fold_bias(LF_ARGS(FoldBiasArgs)) {
  uint o = (uint)LF_INDEX;
  if ((a.out_ & 3) == 0) {
    uint out4 = a.out_ >> 2;
    if (o >= out4) return;
    float4 s = a.extra ? ((device const float4*)a.extra)[o] : float4(0.f);
    if (a.offset) {
      device const half4* w4 = (device const half4*)a.w + o;
      for (uint k = 0; k < a.in; ++k) s += a.offset[k] * float4(w4[(ulong)k * out4]);
    }
    ((device float4*)a.bias)[o] = s;
    return;
  }
  if (o >= a.out_) return;
  float s = a.extra ? a.extra[o] : 0.f;
  if (a.offset) for (uint k = 0; k < a.in; ++k) s += a.offset[k] * (float)a.w[(ulong)k * a.out_ + o];
  a.bias[o] = s;
}
kernel void af3_interleave8(LF_ARGS(Interleave8Args)) {
  uint t = (uint)LF_INDEX, B2 = a.I >> 2;
  if (t >= a.rows * B2) return;
  uint r = lf_udiv(t, B2), c = t - r * B2, m = c >> 1;
  device const uint4* src = (c & 1) ? (device const uint4*)(a.b + (ulong)r * a.lda) : (device const uint4*)(a.a + (ulong)r * a.lda);
  ((device uint4*)a.out)[t] = src[m];
}
kernel void af3_trigate_weight(LF_ARGS(TriGateWArgs)) {
  uint t = (uint)LF_INDEX;
  if ((a.C & 1) == 0) {
    uint W4 = a.C, total = a.rows * W4;
    if (t >= total) return;
    uint r = lf_udiv(t, W4), col4 = t - r * W4, col = col4 * 4;
    uint kind = a.quarters ? ((col & 127) >> 5) : ((col & 31) >> 3);
    uint c = a.quarters ? (32 * (col >> 7) + (col & 31)) : (8 * (col >> 5) + (col & 7));
    uint k = kind == 1 ? 2 : kind == 2 ? 1 : kind;      // (blocks of 8 or 32: pa ga pb gb; a and b interleaved by channel)
    device const half* src = (k < 2 ? a.proj : a.gate) + (ulong)r * 2 * a.C + 2 * c + (k & 1);
    ((device half4*)a.out)[t] = half4(src[0], src[2], src[4], src[6]);
    return;
  }
  uint W = 4 * a.C;
  if (t >= a.rows * W) return;
  uint r = lf_udiv(t, W), col = t - r * W;
  uint kind = a.quarters ? ((col & 127) >> 5) : ((col & 31) >> 3);
  uint c = a.quarters ? (32 * (col >> 7) + (col & 31)) : (8 * (col >> 5) + (col & 7));
  uint k = kind == 1 ? 2 : kind == 2 ? 1 : kind;      // (blocks of 8 or 32: pa ga pb gb; a and b interleaved by channel)
  a.out[t] = k < 2 ? a.proj[r * 2 * a.C + 2 * c + k] : a.gate[r * 2 * a.C + 2 * c + k - 2];
}
kernel void af3_scale(LF_ARGS(ScaleArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n) return;
  float v = a.x[t] * a.s;
  v = a.relu ? max(v, 0.f) : v;
  if (a.outH) a.outH[t] = (half)v;
  else a.x[t] = v;
}
kernel void af3_scale_rows(LF_ARGS(ScaleRowsArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.rows * C4) return;
    uint r = lf_udiv((uint)t, C4), c4 = (uint)t - r * C4, pr = r - lf_udiv(r, a.period) * a.period;
    float4 v = ((device const float4*)a.x)[t];
    if (a.resY) {
      float4 ry = ((device const float4*)a.resY)[t];
      float4 rg = ((device const float4*)a.resGate)[(ulong)pr * C4 + c4];
      v += ry * rg;
    }
    v *= a.mask[pr];
    ((device float4*)a.x)[t] = v;
    if (a.outH) ((device half4*)a.outH)[t] = half4(v);
    return;
  }
  if (t >= a.rows * a.C) return;
  uint r = lf_udiv((uint)t, a.C), c = (uint)t - r * a.C, pr = r - lf_udiv(r, a.period) * a.period;
  float v = a.x[t] + (a.resY ? a.resY[t] * a.resGate[(ulong)pr * a.C + c] : 0.f);
  v *= a.mask[pr];
  a.x[t] = v;
  if (a.outH) a.outH[t] = (half)v;
}
kernel void af3_scale_rows_h(LF_ARGS(ScaleRowsHArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.rows * C4) return;
    float m = a.mask[lf_udiv((uint)t, C4)];
    ((device half4*)a.x)[t] = half4(float4(((device const half4*)a.x)[t]) * m);
    if (a.y) ((device half4*)a.y)[t] = half4(float4(((device const half4*)a.y)[t]) * m);
    return;
  }
  if (t >= a.rows * a.C) return;
  float m = a.mask[lf_udiv((uint)t, a.C)];
  a.x[t] = (half)((float)a.x[t] * m);
  if (a.y) a.y[t] = (half)((float)a.y[t] * m);
}

// ---------------------------------------------------------------- the embedder
kernel void af3_outer_sum(LF_ARGS(OuterSumArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.n * a.n * C4) return;
    uint ij = lf_udiv((uint)t, C4), c4 = (uint)t - ij * C4, i = lf_udiv(ij, a.n), j = ij - i * a.n;
    ((device float4*)a.pair)[t] = ((device const float4*)a.left)[i * C4 + c4] +
                                  ((device const float4*)a.right)[j * C4 + c4] +
                                  (a.add ? ((device const float4*)a.add)[a.addRow ? c4 : t] : float4(0.f));
    return;
  }
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  a.pair[t] = a.left[i * a.C + c] + a.right[j * a.C + c] + (a.add ? a.add[a.addRow ? c : t] : 0.f);
}
// AF3's relative encoding (139 one-hot columns: residue offset 66, token offset 66, same entity 1, chain offset 6)
kernel void af3_relenc(LF_ARGS(RelEncArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.n * a.n * C4) return;
    uint ij = lf_udiv((uint)t, C4), c4 = (uint)t - ij * C4, i = lf_udiv(ij, a.n), j = ij - i * a.n;
    const int maxIdx = 32, maxChain = 2, positionBins = 2 * maxIdx + 2;
    bool sameChain = a.r.asym[i] == a.r.asym[j], sameEntity = a.r.ent[i] == a.r.ent[j];
    int c0 = sameChain ? clamp(a.r.ri[i] - a.r.ri[j] + maxIdx, 0, 2 * maxIdx) : 2 * maxIdx + 1;
    bool sameResidue = sameChain && a.r.ri[i] == a.r.ri[j];
    int c1 = positionBins + (sameResidue ? clamp(a.r.ti[i] - a.r.ti[j] + maxIdx, 0, 2 * maxIdx) : 2 * maxIdx + 1);
    int c3 = positionBins * 2 + 1 + (sameEntity ? clamp(a.r.sym[i] - a.r.sym[j] + maxChain, 0, 2 * maxChain) : 2 * maxChain + 1);
    device const float4* W4 = (device const float4*)a.W;
    float4 v = W4[c0 * C4 + c4] + W4[c1 * C4 + c4] + W4[c3 * C4 + c4];
    if (sameEntity) v += W4[(positionBins * 2) * C4 + c4];
    ((device float4*)a.pair)[t] += v;
    return;
  }
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
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.pairs * C4) return;
    uint p = lf_udiv((uint)t, C4), c4 = (uint)t - p * C4;
    float4 v = float4(0.f);
    if (a.bonds && a.w) v += a.bonds[p] * ((device const float4*)a.w)[c4];
    if (a.table) {
      int o = a.orders ? (int)a.orders[p] : 0;
      if (o < 0 || o >= 7) o = 0;
      v += ((device const float4*)a.table)[o * C4 + c4] + ((device const float4*)a.unspecified)[c4];
    }
    ((device float4*)a.pair)[t] += v;
    return;
  }
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
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.count * C4) return;
    uint row = lf_udiv((uint)t, C4), c4 = (uint)t - row * C4, token = row - lf_udiv(row, a.n) * a.n;
    int code = a.rows[row]; float d = a.del[row];
    device const float4* W4 = (device const float4*)a.W;
    float4 v = code >= 0 && code < 32 ? W4[code * C4 + c4] : float4(0.f);
    v += clamp(d, 0.f, 1.f) * W4[32 * C4 + c4];
    v += atan(d / 3.f) * (2.f / M_PI_F) * W4[33 * C4 + c4];
    if (a.width > 34 && a.pairedQuery && row < a.n) v += W4[34 * C4 + c4];
    if (a.dnMsa && a.isDna && code == 30 && a.isDna[token] != 0) v += ((device const float4*)a.dnMsa)[c4];      // boltz2's DN row
    ((device float4*)a.msa)[t] = v + ((device const float4*)a.fromTarget)[token * C4 + c4];
    return;
  }
  if (t >= a.count * a.C) return;
  uint row = lf_udiv((uint)t, a.C), c = (uint)t - row * a.C, token = row - lf_udiv(row, a.n) * a.n;
  int code = a.rows[row]; float d = a.del[row];
  float v = code >= 0 && code < 32 ? a.W[code * a.C + c] : 0.f;
  v += clamp(d, 0.f, 1.f) * a.W[32 * a.C + c];
  v += atan(d / 3.f) * (2.f / M_PI_F) * a.W[33 * a.C + c];
  if (a.width > 34 && a.pairedQuery && row < a.n) v += a.W[34 * a.C + c];
  if (a.dnMsa && a.isDna && code == 30 && a.isDna[token] != 0) v += a.dnMsa[c];      // boltz2's DN row
  a.msa[t] = v + a.fromTarget[token * a.C + c];
}
kernel void af3_onehot(LF_ARGS(OnehotArgs)) {
  uint t = (uint)LF_INDEX;
  if (t >= a.n * a.classes) return;
  uint r = lf_udiv(t, a.classes), c = t - r * a.classes;
  a.out[t] = a.idx[r] == (int)c ? 1.f : 0.f;
}
kernel void af3_add_row_col(LF_ARGS(AddRowColArgs)) {      // act[i][j] = (query[i][j] +) row[j] + col[i] (+ bias)
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.n * a.n * C4) return;
    uint ij = lf_udiv((uint)t, C4), c4 = (uint)t - ij * C4, i = lf_udiv(ij, a.n), j = ij - i * a.n;
    float4 v = (a.query ? ((device const float4*)a.query)[t] : ((device const float4*)a.act)[t]) +
               ((device const float4*)a.row)[j * C4 + c4] + ((device const float4*)a.col)[i * C4 + c4];
    if (a.bias) v += ((device const float4*)a.bias)[c4];
    ((device float4*)a.act)[t] = v;
    return;
  }
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  float v = (a.query ? a.query[t] : a.act[t]) + a.row[j * a.C + c] + a.col[i * a.C + c] + (a.bias ? a.bias[c] : 0.f);
  a.act[t] = v;
}
kernel void af3_template_sparse_proj(LF_ARGS(SparseProjArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.pairs * C4) return;
    uint p = lf_udiv((uint)t, C4), c4 = (uint)t - p * C4;
    float4 v = a.query ? ((device const float4*)a.query)[t] : ((device const float4*)a.act)[t];
    device const int* idx = a.idx + (ulong)p * a.K;
    device const float* val = a.val + (ulong)p * a.K;
    device const half4* W4 = (device const half4*)a.W;
    for (uint k = 0; k < a.K; ++k) {
      int col = idx[k];
      if (col >= 0 && (uint)col < a.width) v += (float)(half)val[k] * float4(W4[(uint)col * C4 + c4]);
    }
    if (a.bias) v += ((device const float4*)a.bias)[c4];
    ((device float4*)a.act)[t] = v;
    return;
  }
  if (t >= a.pairs * a.C) return;
  uint p = lf_udiv((uint)t, a.C), c = (uint)t - p * a.C;
  float v = a.query ? a.query[t] : a.act[t];
  for (uint k = 0; k < a.K; ++k) {
    int col = a.idx[(ulong)p * a.K + k];
    if (col >= 0 && (uint)col < a.width) v += (float)(half)a.val[(ulong)p * a.K + k] * (float)a.W[(uint)col * a.C + c];
  }
  if (a.bias) v += a.bias[c];
  a.act[t] = v;
}
kernel void af3_template_geometry(LF_ARGS(TmplGeomArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.pairs * C4) return;
    uint p = lf_udiv((uint)t, C4), c4 = (uint)t - p * C4;
    float4 v = float4(0.f);
    int b = a.bin[p];
    if (b >= 0) v += ((device const float4*)a.W0)[b * C4 + c4];
    v += a.pb[p] * ((device const float4*)a.W1)[c4] + a.uv[p * 3] * ((device const float4*)a.W4)[c4] +
         a.uv[p * 3 + 1] * ((device const float4*)a.W5)[c4] + a.uv[p * 3 + 2] * ((device const float4*)a.W6)[c4] +
         a.bb[p] * ((device const float4*)a.W7)[c4];
    ((device float4*)a.act)[t] += v;
    return;
  }
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
  const uint j0 = tg.x * 32, i0 = tg.y * 32, h = tg.z, tx = tid & 31, ty = tid >> 5;     // 32 x 8 threads
  if (!a.swap) {
    _Pragma("clang loop unroll(full)")
    for (uint r = ty; r < 32; r += 8) {
      const uint i = i0 + r, j = j0 + tx;
      if (i < a.n && j < a.stride) {
        a.bias[((ulong)h * a.n + i) * a.stride + j] =
          j < a.n ? (half)(a.scale * a.raw[((ulong)i * a.n + j) * a.ld + a.off + h]) : -INFINITY;
      }
    }
    return;
  }
  threadgroup float tile[32][33];
  _Pragma("clang loop unroll(full)")
  for (uint r = ty; r < 32; r += 8) {
    const uint sr = j0 + r, sc = i0 + tx;
    tile[r][tx] = sr < a.n && sc < a.n ? a.raw[((ulong)sr * a.n + sc) * a.ld + a.off + h] : 0.f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (uint r = ty; r < 32; r += 8) {
    const uint i = i0 + r, j = j0 + tx;
    if (i < a.n && j < a.stride) {
      // (the padding past n is -inf: a key tile reaching past the last key masks it through the bias alone)
      a.bias[((ulong)h * a.n + i) * a.stride + j] = j < a.n ? (half)(a.scale * tile[tx][r]) : -INFINITY;
    }
  }
}
// Fused pair-bias projection + layout for H in {4, 8, 16} and C in {64, 128}: 8 lanes cooperate per pair row to read
// pn [n*n, C] coalesced (32B/lane), multiply by Ws [C, H] in shared memory, reduce via simd_shuffle_xors, transpose/pad
// in shared memory, and write out [H][n][stride] in log2 units.
kernel void af3_pair_bias(constant PairBiasArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
  const uint j0 = tg.x * 8, i0 = tg.y * 8;
  if (i0 >= a.n || j0 >= a.n) return;
  threadgroup half4 Ws[1024];        // up to 512 * 2 half4s (8.0 KB)
  threadgroup half4 Ws2[1024];       // second 8-head block when H > 8 (8.0 KB)
  threadgroup half tile[8 * 8 * 16]; // [8][8][16] halves (2.0 KB)
  if (a.C <= 128 || (a.C == 256 && a.H <= 4)) {
    if (a.H <= 8) {
      if (tid < a.C * (a.H <= 4 ? 1u : 2u)) Ws[tid] = *(device const half4*)(a.W + tid * 4);
    } else if (tid < a.C * 2) {
      Ws[tid] = *(device const half4*)(a.W + (tid >> 1) * a.H + (tid & 1) * 4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint sr_base = a.swap ? j0 : i0;
  const uint sc_base = a.swap ? i0 : j0;
  const uint tx = tid & 7, g = tid >> 3; // g in 0..31
  const uint r0 = g >> 3, r1 = r0 + 4, m = g & 7;
  const uint sr0 = sr_base + r0, sr1 = sr_base + r1, sc = sc_base + m;
  const bool valid0 = sr0 < a.n && sc < a.n;
  const bool valid1 = sr1 < a.n && sc < a.n;

  if (a.H <= 4 && a.C <= 256) {
    const uint c4_per_lane = a.C >> 5; // 4 when C=128, 2 when C=64
    const uint c4_base = tx * c4_per_lane;
    device const half4* p0 = (device const half4*)(a.pn + ((ulong)sr0 * a.n + sc) * a.C) + c4_base;
    device const half4* p1 = (device const half4*)(a.pn + ((ulong)sr1 * a.n + sc) * a.C) + c4_base;
    float4 acc0(0.f), acc1(0.f);
    for (uint u = 0; u < c4_per_lane; ++u) {
      float4 x0 = valid0 ? float4(p0[u]) : float4(0.f);
      float4 x1 = valid1 ? float4(p1[u]) : float4(0.f);
      uint c = (c4_base + u) * 4;
      float4 w0 = float4(Ws[c]), w1 = float4(Ws[c + 1]), w2 = float4(Ws[c + 2]), w3 = float4(Ws[c + 3]);
      acc0 += x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3;
      acc1 += x1.x * w0 + x1.y * w1 + x1.z * w2 + x1.w * w3;
    }
    float4 v = (tx < 4 ? acc0 : acc1) + simd_shuffle_xor(tx < 4 ? acc1 : acc0, (ushort)4);
    v += simd_shuffle_xor(v, (ushort)2);
    v += simd_shuffle_xor(v, (ushort)1);
    if (tx == 0 || tx == 4) {
      const uint r = tx == 0 ? r0 : r1;
      const uint sr = sr_base + r;
      const bool val_pair = tx == 0 ? valid0 : valid1;
      const uint row_i = a.swap ? m : r, col_j = a.swap ? r : m;
      const bool vj = (j0 + col_j) < a.n;
      const float eb = (a.extraBias && val_pair) ? a.extraBias[sr * a.n + sc] : 0.f;
      const float4 wb = a.wBias ? *(device const float4*)a.wBias : float4(0.f);
      _Pragma("clang loop unroll(full)")
      for (uint h = 0; h < 4; ++h) {
        tile[(h * 8 + row_i) * 16 + col_j] = vj ? (half)((v[h] + wb[h] + eb) * M_LOG2E_F) : (half)(-INFINITY);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < a.H * 8) {
      uint h = tid >> 3, row_i = tid & 7, i = i0 + row_i;
      if (i < a.n) {
        *(device uint4*)(a.out + ((ulong)h * a.n + i) * a.stride + j0) = *(threadgroup const uint4*)(tile + (h * 8 + row_i) * 16);
      }
    }
  } else if (a.C <= 128) {
    const uint c4_per_lane = a.C >> 5;
    const uint c4_base = tx * c4_per_lane;
    device const half4* p0 = (device const half4*)(a.pn + ((ulong)sr0 * a.n + sc) * a.C) + c4_base;
    device const half4* p1 = (device const half4*)(a.pn + ((ulong)sr1 * a.n + sc) * a.C) + c4_base;
    float4 x0_u[4], x1_u[4];
    for (uint u = 0; u < c4_per_lane; ++u) {
      x0_u[u] = valid0 ? float4(p0[u]) : float4(0.f);
      x1_u[u] = valid1 ? float4(p1[u]) : float4(0.f);
    }
    const float eb0 = (a.extraBias && valid0) ? a.extraBias[sr0 * a.n + sc] : 0.f;
    const float eb1 = (a.extraBias && valid1) ? a.extraBias[sr1 * a.n + sc] : 0.f;
    for (uint h8 = 0; h8 < a.H; h8 += 8) {
      const uint remH = min(8u, a.H - h8);
      if (h8 > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < a.C * 2) {
          Ws[tid] = (remH == 8 || (tid & 1) == 0) ? *(device const half4*)(a.W + (tid >> 1) * a.H + h8 + (tid & 1) * 4) : half4(0.h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      float4 acc0_0(0.f), acc0_1(0.f), acc1_0(0.f), acc1_1(0.f);
      for (uint u = 0; u < c4_per_lane; ++u) {
        float4 x0 = x0_u[u], x1 = x1_u[u];
        uint c = (c4_base + u) * 4;
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; ++k) {
          float xv0 = x0[k], xv1 = x1[k];
          float4 wa = float4(Ws[(c + k) * 2]), wb = float4(Ws[(c + k) * 2 + 1]);
          acc0_0 += xv0 * wa; acc0_1 += xv0 * wb;
          acc1_0 += xv1 * wa; acc1_1 += xv1 * wb;
        }
      }
      float4 v0 = (tx < 4 ? acc0_0 : acc0_1) + simd_shuffle_xor(tx < 4 ? acc0_1 : acc0_0, (ushort)4);
      float4 v1 = (tx < 4 ? acc1_0 : acc1_1) + simd_shuffle_xor(tx < 4 ? acc1_1 : acc1_0, (ushort)4);
      v0 += simd_shuffle_xor(v0, (ushort)2); v1 += simd_shuffle_xor(v1, (ushort)2);
      v0 += simd_shuffle_xor(v0, (ushort)1); v1 += simd_shuffle_xor(v1, (ushort)1);
      if (tx == 0 || (tx == 4 && remH > 4)) {
        const uint h_base = tx;
        const uint row_i0 = a.swap ? m : r0, col_j0 = a.swap ? r0 : m;
        const uint row_i1 = a.swap ? m : r1, col_j1 = a.swap ? r1 : m;
        const bool vj0 = (j0 + col_j0) < a.n;
        const bool vj1 = (j0 + col_j1) < a.n;
        const float4 wb = a.wBias ? *(device const float4*)(a.wBias + h8 + h_base) : float4(0.f);
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4; ++t) {
          uint h = h_base + t;
          tile[(h * 8 + row_i0) * 16 + col_j0] = vj0 ? (half)((v0[t] + wb[t] + eb0) * M_LOG2E_F) : (half)(-INFINITY);
          tile[(h * 8 + row_i1) * 16 + col_j1] = vj1 ? (half)((v1[t] + wb[t] + eb1) * M_LOG2E_F) : (half)(-INFINITY);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < remH * 8) {
        uint h = h8 + (tid >> 3), row_i = tid & 7, i = i0 + row_i;
        if (i < a.n) {
          *(device uint4*)(a.out + ((ulong)h * a.n + i) * a.stride + j0) = *(threadgroup const uint4*)(tile + ((tid >> 3) * 8 + row_i) * 16);
        }
      }
    }
  } else if (a.H <= 8) {
    for (uint t = tid; t < a.C * 2; t += 256) {
      Ws[t] = (a.H == 8 || (t & 1) == 0) ? *(device const half4*)(a.W + (t >> 1) * a.H + (t & 1) * 4) : half4(0.h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float eb0 = (a.extraBias && valid0) ? a.extraBias[sr0 * a.n + sc] : 0.f;
    const float eb1 = (a.extraBias && valid1) ? a.extraBias[sr1 * a.n + sc] : 0.f;
    device const half4* p0_base = (device const half4*)(a.pn + ((ulong)sr0 * a.n + sc) * a.C);
    device const half4* p1_base = (device const half4*)(a.pn + ((ulong)sr1 * a.n + sc) * a.C);
    float4 acc0_0(0.f), acc0_1(0.f), acc1_0(0.f), acc1_1(0.f);
    if (valid0 && valid1) {
      for (uint c0 = 0; c0 < a.C; c0 += 128) {
        const uint c4_base = (c0 >> 2) + tx * 4;
        _Pragma("clang loop unroll(full)")
        for (uint u = 0; u < 4; ++u) {
          float4 x0 = float4(p0_base[c4_base + u]);
          float4 x1 = float4(p1_base[c4_base + u]);
          uint c = c0 + (tx * 4 + u) * 4;
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < 4; ++k) {
            float xv0 = x0[k], xv1 = x1[k];
            float4 wa = float4(Ws[(c + k) * 2]), wb = float4(Ws[(c + k) * 2 + 1]);
            acc0_0 += xv0 * wa; acc0_1 += xv0 * wb;
            acc1_0 += xv1 * wa; acc1_1 += xv1 * wb;
          }
        }
      }
    } else {
      for (uint c0 = 0; c0 < a.C; c0 += 128) {
        const uint c4_base = (c0 >> 2) + tx * 4;
        _Pragma("clang loop unroll(full)")
        for (uint u = 0; u < 4; ++u) {
          float4 x0 = valid0 ? float4(p0_base[c4_base + u]) : float4(0.f);
          float4 x1 = valid1 ? float4(p1_base[c4_base + u]) : float4(0.f);
          uint c = c0 + (tx * 4 + u) * 4;
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < 4; ++k) {
            float xv0 = x0[k], xv1 = x1[k];
            float4 wa = float4(Ws[(c + k) * 2]), wb = float4(Ws[(c + k) * 2 + 1]);
            acc0_0 += xv0 * wa; acc0_1 += xv0 * wb;
            acc1_0 += xv1 * wa; acc1_1 += xv1 * wb;
          }
        }
      }
    }
    float4 v0 = (tx < 4 ? acc0_0 : acc0_1) + simd_shuffle_xor(tx < 4 ? acc0_1 : acc0_0, (ushort)4);
    float4 v1 = (tx < 4 ? acc1_0 : acc1_1) + simd_shuffle_xor(tx < 4 ? acc1_1 : acc1_0, (ushort)4);
    v0 += simd_shuffle_xor(v0, (ushort)2); v1 += simd_shuffle_xor(v1, (ushort)2);
    v0 += simd_shuffle_xor(v0, (ushort)1); v1 += simd_shuffle_xor(v1, (ushort)1);
    if (tx == 0 || (tx == 4 && a.H > 4)) {
      const uint h_base = tx;
      const uint row_i0 = a.swap ? m : r0, col_j0 = a.swap ? r0 : m;
      const uint row_i1 = a.swap ? m : r1, col_j1 = a.swap ? r1 : m;
      const bool vj0 = (j0 + col_j0) < a.n;
      const bool vj1 = (j0 + col_j1) < a.n;
      const float4 wb = a.wBias ? *(device const float4*)(a.wBias + h_base) : float4(0.f);
      _Pragma("clang loop unroll(full)")
      for (uint t = 0; t < 4; ++t) {
        uint h = h_base + t;
        tile[(h * 8 + row_i0) * 16 + col_j0] = vj0 ? (half)((v0[t] + wb[t] + eb0) * M_LOG2E_F) : (half)(-INFINITY);
        tile[(h * 8 + row_i1) * 16 + col_j1] = vj1 ? (half)((v1[t] + wb[t] + eb1) * M_LOG2E_F) : (half)(-INFINITY);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < a.H * 8) {
      uint h = tid >> 3, row_i = tid & 7, i = i0 + row_i;
      if (i < a.n) {
        *(device uint4*)(a.out + ((ulong)h * a.n + i) * a.stride + j0) = *(threadgroup const uint4*)(tile + (h * 8 + row_i) * 16);
      }
    }
  } else {
    const uint remH = a.H - 8u;
    if (remH <= 4) {
      for (uint t = tid; t < a.C * 2; t += 256) {
        Ws[t] = *(device const half4*)(a.W + (t >> 1) * a.H + (t & 1) * 4);
      }
      for (uint t = tid; t < a.C; t += 256) {
        Ws2[t] = *(device const half4*)(a.W + t * a.H + 8);
      }
    } else {
      for (uint t = tid; t < a.C * 2; t += 256) {
        Ws[t] = *(device const half4*)(a.W + (t >> 1) * a.H + (t & 1) * 4);
        Ws2[t] = *(device const half4*)(a.W + (t >> 1) * a.H + 8 + (t & 1) * 4);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float eb0 = (a.extraBias && valid0) ? a.extraBias[sr0 * a.n + sc] : 0.f;
    const float eb1 = (a.extraBias && valid1) ? a.extraBias[sr1 * a.n + sc] : 0.f;
    device const half4* p0_base = (device const half4*)(a.pn + ((ulong)sr0 * a.n + sc) * a.C);
    device const half4* p1_base = (device const half4*)(a.pn + ((ulong)sr1 * a.n + sc) * a.C);
    float4 acc0_0(0.f), acc0_1(0.f), acc1_0(0.f), acc1_1(0.f);
    float4 acc0_2(0.f), acc0_3(0.f), acc1_2(0.f), acc1_3(0.f);
    if (remH <= 4) {
      for (uint c0 = 0; c0 < a.C; c0 += 128) {
        const uint c4_base = (c0 >> 2) + tx * 4;
        _Pragma("clang loop unroll(full)")
        for (uint u = 0; u < 4; ++u) {
          float4 x0 = valid0 ? float4(p0_base[c4_base + u]) : float4(0.f);
          float4 x1 = valid1 ? float4(p1_base[c4_base + u]) : float4(0.f);
          uint c = c0 + (tx * 4 + u) * 4;
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < 4; ++k) {
            float xv0 = x0[k], xv1 = x1[k];
            uint wi = (c + k) * 2;
            float4 wa = float4(Ws[wi]), wb = float4(Ws[wi + 1]);
            acc0_0 += xv0 * wa; acc0_1 += xv0 * wb;
            acc1_0 += xv1 * wa; acc1_1 += xv1 * wb;
            float4 wc = float4(Ws2[c + k]);
            acc0_2 += xv0 * wc;
            acc1_2 += xv1 * wc;
          }
        }
      }
    } else {
      for (uint c0 = 0; c0 < a.C; c0 += 128) {
        const uint c4_base = (c0 >> 2) + tx * 4;
        _Pragma("clang loop unroll(full)")
        for (uint u = 0; u < 4; ++u) {
          float4 x0 = valid0 ? float4(p0_base[c4_base + u]) : float4(0.f);
          float4 x1 = valid1 ? float4(p1_base[c4_base + u]) : float4(0.f);
          uint c = c0 + (tx * 4 + u) * 4;
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < 4; ++k) {
            float xv0 = x0[k], xv1 = x1[k];
            uint wi = (c + k) * 2;
            float4 wa = float4(Ws[wi]), wb = float4(Ws[wi + 1]);
            acc0_0 += xv0 * wa; acc0_1 += xv0 * wb;
            acc1_0 += xv1 * wa; acc1_1 += xv1 * wb;
            float4 wc = float4(Ws2[wi]), wd = float4(Ws2[wi + 1]);
            acc0_2 += xv0 * wc; acc0_3 += xv0 * wd;
            acc1_2 += xv1 * wc; acc1_3 += xv1 * wd;
          }
        }
      }
    }
    float4 v0_0 = (tx < 4 ? acc0_0 : acc0_1) + simd_shuffle_xor(tx < 4 ? acc0_1 : acc0_0, (ushort)4);
    float4 v1_0 = (tx < 4 ? acc1_0 : acc1_1) + simd_shuffle_xor(tx < 4 ? acc1_1 : acc1_0, (ushort)4);
    v0_0 += simd_shuffle_xor(v0_0, (ushort)2); v1_0 += simd_shuffle_xor(v1_0, (ushort)2);
    v0_0 += simd_shuffle_xor(v0_0, (ushort)1); v1_0 += simd_shuffle_xor(v1_0, (ushort)1);
    float4 v0_1 = (tx < 4 ? acc0_2 : acc0_3) + simd_shuffle_xor(tx < 4 ? acc0_3 : acc0_2, (ushort)4);
    float4 v1_1 = (tx < 4 ? acc1_2 : acc1_3) + simd_shuffle_xor(tx < 4 ? acc1_3 : acc1_2, (ushort)4);
    v0_1 += simd_shuffle_xor(v0_1, (ushort)2); v1_1 += simd_shuffle_xor(v1_1, (ushort)2);
    v0_1 += simd_shuffle_xor(v0_1, (ushort)1); v1_1 += simd_shuffle_xor(v1_1, (ushort)1);
    if (tx == 0 || tx == 4) {
      const uint row_i0 = a.swap ? m : r0, col_j0 = a.swap ? r0 : m;
      const uint row_i1 = a.swap ? m : r1, col_j1 = a.swap ? r1 : m;
      const bool vj0 = (j0 + col_j0) < a.n;
      const bool vj1 = (j0 + col_j1) < a.n;
      const float4 wb0 = a.wBias ? *(device const float4*)(a.wBias + tx) : float4(0.f);
      _Pragma("clang loop unroll(full)")
      for (uint t = 0; t < 4; ++t) {
        uint h = tx + t;
        tile[(h * 8 + row_i0) * 16 + col_j0] = vj0 ? (half)((v0_0[t] + wb0[t] + eb0) * M_LOG2E_F) : (half)(-INFINITY);
        tile[(h * 8 + row_i1) * 16 + col_j1] = vj1 ? (half)((v1_0[t] + wb0[t] + eb1) * M_LOG2E_F) : (half)(-INFINITY);
      }
      if (tx == 0 || remH > 4) {
        const float4 wb1 = a.wBias ? *(device const float4*)(a.wBias + 8 + tx) : float4(0.f);
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < 4; ++t) {
          uint h = tx + t;
          tile[(h * 8 + row_i0) * 16 + 8 + col_j0] = vj0 ? (half)((v0_1[t] + wb1[t] + eb0) * M_LOG2E_F) : (half)(-INFINITY);
          tile[(h * 8 + row_i1) * 16 + 8 + col_j1] = vj1 ? (half)((v1_1[t] + wb1[t] + eb1) * M_LOG2E_F) : (half)(-INFINITY);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < a.H * 8) {
      uint h = tid >> 3, row_i = tid & 7, i = i0 + row_i;
      if (i < a.n) {
        *(device uint4*)(a.out + ((ulong)h * a.n + i) * a.stride + j0) =
          *(threadgroup const uint4*)(tile + ((h & 7) * 8 + row_i) * 16 + (h & 8));
      }
    }
  }
  if (j0 + 8 >= a.n) {
    uint sub = tid & 127, h = sub >> 3, i = i0 + (sub & 7);
    if (h < a.H && i < a.n) {
      for (uint jp = j0 + 8 + (tid >> 7) * 8; jp < a.stride; jp += 16) {
        device half4* dst = (device half4*)(a.out + ((ulong)h * a.n + i) * a.stride + jp);
        dst[0] = half4(-INFINITY); dst[1] = half4(-INFINITY);
      }
    }
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
// the two above, 8 channels (16 bytes) a thread where d is a multiple of 8 (a half a thread moved ~12 GB/s)
kernel void af3_msa_v_heads8(LF_ARGS(MsaVHeadsArgs)) {
  const uint d8 = a.d / 8;
  const uint t = (uint)LF_INDEX;
  if ((ulong)t >= (ulong)a.S * a.np * a.heads * d8) return;
  uint r = lf_udiv(t, d8), e8 = t - r * d8, r2 = lf_udiv(r, a.S), s = r - r2 * a.S, h = lf_udiv(r2, a.np), j = r2 - h * a.np;
  ((device uint4*)a.out)[t] = j < a.n ? *(device const uint4*)(a.v + ((ulong)s * a.n + j) * a.heads * a.d + h * a.d + e8 * 8) : uint4(0);
}
kernel void af3_msa_from_heads8(LF_ARGS(MsaFromHeadsArgs)) {
  const uint W8 = a.heads * a.d / 8, d8 = a.d / 8;
  const uint t = (uint)LF_INDEX;
  if ((ulong)t >= (ulong)a.S * a.n * W8) return;
  uint si = lf_udiv(t, W8), c8 = t - si * W8, s = lf_udiv(si, a.n), i = si - s * a.n, h = lf_udiv(c8, d8), e8 = c8 - h * d8;
  device const half4* o = (device const half4*)(a.o + (((ulong)h * a.n + i) * a.S + s) * a.d + e8 * 8);
  device const half4* g = (device const half4*)(a.gate + (ulong)t * 8);
  device half4* out = (device half4*)(a.out + (ulong)t * 8);
  for (int k = 0; k < 2; ++k) out[k] = half4(float4(o[k]) * (1.f / (1.f + exp(-float4(g[k])))));
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
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.L * a.L * C4) return;
    uint ij = lf_udiv((uint)t, C4), c4 = (uint)t - ij * C4, i = lf_udiv(ij, a.L), j = ij - i * a.L;
    ulong ji4 = ((ulong)j * a.L + i) * C4 + c4;
    float4 v = a.xh ? float4(((device const half4*)a.xh)[t]) + float4(((device const half4*)a.xh)[ji4])
                    : ((device const float4*)a.x)[t] + ((device const float4*)a.x)[ji4];
    ((device float4*)a.y)[t] = v * a.scale;
    return;
  }
  if (t >= (ulong)a.L * a.L * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.L), j = ij - i * a.L;
  ulong jic = ((ulong)j * a.L + i) * a.C + c;
  float v = a.xh ? (float)a.xh[t] + (float)a.xh[jic] : a.x[t] + a.x[jic];
  a.y[t] = v * a.scale;
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
kernel void af3_sigmoid_atom(LF_ARGS(SigmoidAtomArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n) return;
  if (!a.raw) {
    a.qs[t] = lf_sigmoid(a.qs[t]);
    a.ks[t] = lf_sigmoid(a.ks[t]);
    a.fs[t] = lf_sigmoid(a.fs[t]);
  }
  a.zg[t] = lf_sigmoid(a.zg[t]);
  a.tg[t] = lf_sigmoid(a.tg[t]);
}
// a simdgroup a row, eight a threadgroup; two-pass variance
kernel void af3_ada_ln_128(constant AdaLnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint3 tpg [[threads_per_threadgroup]]) {
  ulong row = ((ulong)tg.y * ng.x + tg.x) * (tpg.x >> 5) + sg;
  if (row >= a.rows) return;
  device float4* x4 = (device float4*)(a.x + row * 128);
  uint r32 = (uint)row, prRow = r32 < a.period ? r32 : r32 - lf_udiv(r32, a.period) * a.period;
  ulong pr = (ulong)prRow * a.ld, gpr = (ulong)prRow * a.resLd;
  const bool rawS = (a.raw & 1) != 0, rawG = (a.raw & 2) != 0;
  float4 xv = x4[lane];
  if (a.resY || a.resYh) {
    float4 ry = a.resY ? ((device const float4*)(a.resY + row * 128))[lane]
                       : float4(((device const half4*)(a.resYh + row * 128))[lane]) * a.resScale;
    float4 rg = ((device const float4*)(a.resGate + gpr))[lane];
    xv += ry * (rawG ? rg : float4(lf_sigmoid(rg.x), lf_sigmoid(rg.y), lf_sigmoid(rg.z), lf_sigmoid(rg.w)));
  }
  if (a.rowMask) xv *= a.rowMask[prRow];
  if (a.resY || a.resYh || a.rowMask) x4[lane] = xv;
  float mean = simd_sum(xv.x + xv.y + xv.z + xv.w) / 128.f;
  float4 d = xv - mean;
  float inv = rsqrt(simd_sum(d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w) / 128.f + a.eps);
  float4 nrm = d * inv;
  float4 sv = ((device const float4*)(a.scale + pr))[lane];
  float4 y = (rawS ? sv : float4(lf_sigmoid(sv.x), lf_sigmoid(sv.y), lf_sigmoid(sv.z), lf_sigmoid(sv.w))) * nrm +
             (a.shift ? ((device const float4*)(a.shift + pr))[lane] : float4(0.f));
  if (a.out) ((device half4*)(a.out + row * 128))[lane] = half4(y);
  if (a.kOut) {
    float4 ksc = ((device const float4*)(a.kScale + pr))[lane];
    float4 ksv = rawS ? ksc : float4(lf_sigmoid(ksc.x), lf_sigmoid(ksc.y), lf_sigmoid(ksc.z), lf_sigmoid(ksc.w));
    float4 ksh = ((device const float4*)(a.kShift + pr))[lane];
    if (!a.chained) {
      ((device half4*)(a.kOut + row * 128))[lane] = half4(ksv * nrm + ksh);
    } else {
      float meank = simd_sum(y.x + y.y + y.z + y.w) / 128.f;
      float4 dk = y - meank;
      float invk = rsqrt(simd_sum(dk.x * dk.x + dk.y * dk.y + dk.z * dk.z + dk.w * dk.w) / 128.f + a.eps);
      ((device half4*)(a.kOut + row * 128))[lane] = half4(ksv * (dk * invk) + ksh);
    }
  }
  if (a.fOut) {
    float4 fsc = ((device const float4*)(a.fScale + pr))[lane];
    float4 fsv = rawS ? fsc : float4(lf_sigmoid(fsc.x), lf_sigmoid(fsc.y), lf_sigmoid(fsc.z), lf_sigmoid(fsc.w));
    float4 fsh = ((device const float4*)(a.fShift + pr))[lane];
    ((device half4*)(a.fOut + row * 128))[lane] = half4(fsv * nrm + fsh);
  }
}
kernel void af3_ada_ln_768(constant AdaLnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint3 tpg [[threads_per_threadgroup]]) {
  ulong row = ((ulong)tg.y * ng.x + tg.x) * (tpg.x >> 5) + sg;
  if (row >= a.rows) return;
  device float4* x4 = (device float4*)(a.x + row * 768);
  uint r32 = (uint)row, prRow = r32 < a.period ? r32 : r32 - lf_udiv(r32, a.period) * a.period;
  ulong pr = (ulong)prRow * a.ld, gpr = (ulong)prRow * a.resLd;
  const bool rawS = (a.raw & 1) != 0, rawG = (a.raw & 2) != 0;
  device const float4* sc4 = (device const float4*)(a.scale + pr);
  if (!a.resY && !a.resYh && a.shift && a.out && rawS && !a.fOut) {
    device const float4* sh4 = (device const float4*)(a.shift + pr);
    device half4* out4 = (device half4*)(a.out + row * 768);
    float4 xv[6]; float s = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 6; ++i) { float4 val = x4[lane + i * 32]; xv[i] = val; s += val.x + val.y + val.z + val.w; }
    float mean = simd_sum(s) / 768.f, v = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 6; ++i) { float4 d = xv[i] - mean; v += dot(d, d); }
    float inv = rsqrt(simd_sum(v) / 768.f + a.eps);
    if (a.kOut) {
      device const float4* ksc4 = (device const float4*)(a.kScale + pr);
      device const float4* ksh4 = (device const float4*)(a.kShift + pr);
      device half4* kout4 = (device half4*)(a.kOut + row * 768);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 6; ++i) {
        uint idx = lane + i * 32;
        float4 nrm = (xv[i] - mean) * inv;
        out4[idx] = half4(sc4[idx] * nrm + sh4[idx]);
        kout4[idx] = half4(ksc4[idx] * nrm + ksh4[idx]);
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 6; ++i) {
        uint idx = lane + i * 32;
        out4[idx] = half4(sc4[idx] * ((xv[i] - mean) * inv) + sh4[idx]);
      }
    }
    return;
  }
  device const float4* sh4 = a.shift ? (device const float4*)(a.shift + pr) : nullptr;
  device half4* out4 = a.out ? (device half4*)(a.out + row * 768) : nullptr;
  device const float4* ksc4 = a.kOut ? (device const float4*)(a.kScale + pr) : nullptr;
  device const float4* ksh4 = a.kOut ? (device const float4*)(a.kShift + pr) : nullptr;
  device half4* kout4 = a.kOut ? (device half4*)(a.kOut + row * 768) : nullptr;
  device const float4* fsc4 = a.fOut ? (device const float4*)(a.fScale + pr) : nullptr;
  device const float4* fsh4 = a.fOut ? (device const float4*)(a.fShift + pr) : nullptr;
  device half4* fout4 = a.fOut ? (device half4*)(a.fOut + row * 768) : nullptr;
  float4 xv[6]; float s = 0.f;
  if (a.resY || a.resYh) {
    device const float4* ry4 = a.resY ? (device const float4*)(a.resY + row * 768) : nullptr;
    device const half4* ryh4 = a.resYh ? (device const half4*)(a.resYh + row * 768) : nullptr;
    device const float4* rg4 = (device const float4*)(a.resGate + gpr);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 6; ++i) {
      uint idx = lane + i * 32;
      float4 rg = rg4[idx];
      float4 ry = ry4 ? ry4[idx] : float4(ryh4[idx]) * a.resScale;
      float4 val = x4[idx] + ry * (rawG ? rg : float4(lf_sigmoid(rg.x), lf_sigmoid(rg.y), lf_sigmoid(rg.z), lf_sigmoid(rg.w)));
      x4[idx] = val; xv[i] = val; s += val.x + val.y + val.z + val.w;
    }
  } else {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 6; ++i) { float4 val = x4[lane + i * 32]; xv[i] = val; s += val.x + val.y + val.z + val.w; }
  }
  float mean = simd_sum(s) / 768.f, v = 0.f;
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 6; ++i) { float4 d = xv[i] - mean; v += dot(d, d); }
  float inv = rsqrt(simd_sum(v) / 768.f + a.eps);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 6; ++i) {
    uint idx = lane + i * 32;
    float4 nrm = (xv[i] - mean) * inv;
    float4 sv = sc4[idx];
    float4 y = (rawS ? sv : float4(lf_sigmoid(sv.x), lf_sigmoid(sv.y), lf_sigmoid(sv.z), lf_sigmoid(sv.w))) * nrm +
               (sh4 ? sh4[idx] : float4(0.f));
    if (out4) out4[idx] = half4(y);
    if (kout4) {
      float4 ksv = ksc4[idx];
      kout4[idx] = half4((rawS ? ksv : float4(lf_sigmoid(ksv.x), lf_sigmoid(ksv.y), lf_sigmoid(ksv.z), lf_sigmoid(ksv.w))) * nrm + ksh4[idx]);
    }
    if (fout4) {
      float4 fsv = fsc4[idx];
      fout4[idx] = half4((rawS ? fsv : float4(lf_sigmoid(fsv.x), lf_sigmoid(fsv.y), lf_sigmoid(fsv.z), lf_sigmoid(fsv.w))) * nrm + fsh4[idx]);
    }
  }
}
kernel void af3_ada_ln(constant AdaLnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint3 tpg [[threads_per_threadgroup]]) {
  ulong row = ((ulong)tg.y * ng.x + tg.x) * (tpg.x >> 5) + sg;
  if (row >= a.rows) return;
  device float* x = a.x + row * a.C;
  uint r32 = (uint)row, prRow = r32 < a.period ? r32 : r32 - lf_udiv(r32, a.period) * a.period;
  ulong pr = (ulong)prRow * a.ld, gpr = (ulong)prRow * a.resLd;
  const bool rawS = (a.raw & 1) != 0, rawG = (a.raw & 2) != 0;
  float s = 0.f;
  const float rm = a.rowMask ? a.rowMask[prRow] : 1.f;
  if (a.resY || a.resYh) {
    device const float* ry = a.resY ? a.resY + row * a.C : nullptr;
    device const half* ryh = a.resYh ? a.resYh + row * a.C : nullptr;
    device const float* rg = a.resGate + gpr;
    for (uint c = lane; c < a.C; c += 32) {
      float yv = ry ? ry[c] : (float)ryh[c] * a.resScale;
      float val = x[c] + yv * (rawG ? rg[c] : lf_sigmoid(rg[c]));
      if (a.rowMask) val *= rm;
      x[c] = val;
      s += val;
    }
  } else if (a.rowMask) {
    for (uint c = lane; c < a.C; c += 32) { float val = x[c] * rm; x[c] = val; s += val; }
  } else {
    for (uint c = lane; c < a.C; c += 32) s += x[c];
  }
  float mean = simd_sum(s) / a.C, v = 0.f;
  for (uint c = lane; c < a.C; c += 32) { float d = x[c] - mean; v += d * d; }
  float inv = rsqrt(simd_sum(v) / a.C + a.eps);
  float sk = 0.f;
  for (uint c = lane; c < a.C; c += 32) {
    float nrm = (x[c] - mean) * inv;
    float sc = a.scale[pr + c], y = (rawS ? sc : lf_sigmoid(sc)) * nrm + (a.shift ? a.shift[pr + c] : 0.f);
    if (a.out) a.out[row * a.C + c] = (half)y;
    if (a.outF) a.outF[row * a.C + c] = y;
    if (a.kOut) {
      if (!a.chained) {
        float ksc = a.kScale[pr + c], ky = (rawS ? ksc : lf_sigmoid(ksc)) * nrm + a.kShift[pr + c];
        a.kOut[row * a.C + c] = (half)ky;
      } else sk += y;
    }
    if (a.fOut) {
      float fsc = a.fScale[pr + c], fy = (rawS ? fsc : lf_sigmoid(fsc)) * nrm + a.fShift[pr + c];
      a.fOut[row * a.C + c] = (half)fy;
    }
  }
  if (a.kOut && a.chained) {
    float meank = simd_sum(sk) / a.C, vk = 0.f;
    for (uint c = lane; c < a.C; c += 32) {
      float nrm = (x[c] - mean) * inv;
      float sc = a.scale[pr + c], y = (rawS ? sc : lf_sigmoid(sc)) * nrm + (a.shift ? a.shift[pr + c] : 0.f);
      float d = y - meank; vk += d * d;
    }
    float invk = rsqrt(simd_sum(vk) / a.C + a.eps);
    for (uint c = lane; c < a.C; c += 32) {
      float nrm = (x[c] - mean) * inv;
      float sc = a.scale[pr + c], y = (rawS ? sc : lf_sigmoid(sc)) * nrm + (a.shift ? a.shift[pr + c] : 0.f);
      float ksc = a.kScale[pr + c], ky = (rawS ? ksc : lf_sigmoid(ksc)) * ((y - meank) * invk) + a.kShift[pr + c];
      a.kOut[row * a.C + c] = (half)ky;
    }
  }
}
kernel void af3_gather_rows(LF_ARGS(GatherRowsArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.chunks) return;
  uint u = (uint)t, chunks = a.chunks, r = lf_udiv(u, chunks), c = u - r * chunks;
  uint cnt = (uint)a.count, k = lf_udiv(r, cnt), g = r - k * cnt;
  uint4 v = a.mask[g] != 0.f ? ((device const uint4*)a.in)[((ulong)a.idx[g] + k * a.srcRows) * chunks + c] : uint4(0);
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
  const uint s = tg.x, h = tg.y, D = a.D, LD = D + 2, Wd = a.heads * D, W2 = 2 * Wd;
  const uint ks = lf_udiv(s, a.subsets), ss = s - ks * a.subsets;
  const ulong srcBase = (ulong)ks * a.subsets * a.queries;
  for (uint t = tid; t < a.keys * D; t += 32 * NSG) {
    uint key = t / D, e = t - key * D;
    if (a.q2kIdx) {
      uint g = ss * a.keys + key;
      if (a.q2kMask[g] != 0.f) {
        ulong row = srcBase + (uint)a.q2kIdx[g];
        Ks[key * LD + e] = a.kv[row * W2 + h * D + e];
        Vs[key * LD + e] = a.kv[row * W2 + Wd + h * D + e];
      } else {
        Ks[key * LD + e] = (half)0;
        Vs[key * LD + e] = (half)0;
      }
    } else {
      ulong row = (ulong)s * a.keys + key;
      Ks[key * LD + e] = a.kv[row * W2 + h * D + e];
      Vs[key * LD + e] = a.kv[row * W2 + Wd + h * D + e];
    }
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
      half yh = (half)(acc * inv * lf_sigmoid(g));
      if (a.act) a.act[qrow * Wd + h * D + e] += (float)yh * 2.f * a.zg[((ulong)ss * a.queries + qi) * Wd + h * D + e];
      else a.out[qrow * Wd + h * D + e] = yh;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
kernel void af3_gated_residual(LF_ARGS(GatedResidualArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n) return;
  uint u = (uint)t, per = (uint)a.period, g = u - lf_udiv(u, per) * per;
  float y = a.yh ? (float)a.yh[t] * a.scale : a.y[t];
  a.act[t] += y * a.gate[g];
}
kernel void af3_mul_h(LF_ARGS(MulHArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.x[t] = (half)((float)a.x[t] * (float)a.y[t]);
}
kernel void af3_encoder_start(LF_ARGS(EncoderStartArgs)) {
  ulong t = LF_INDEX;
  if (a.C == 128) {
    if (t >= a.rows * 32) return;
    uint u = (uint)t, q = u >> 5, c4 = u & 31;
    uint q1 = (uint)a.q1, k = lf_udiv(q, q1), gq = q - k * q1;
    float4 v = float4(0.f);
    if (a.gmask[gq] != 0.f) {
      uint at = (uint)a.idx[gq];
      device const float* p = a.pos + ((ulong)at + k * a.atoms) * 3;
      float s = a.atomMask ? a.atomMask[at] * a.posScale : a.posScale;
      float p0 = p[0] * s, p1 = p[1] * s, p2 = p[2] * s;
      device const float4* Wp4 = (device const float4*)a.Wp;
      float4 w0 = Wp4[c4], w1 = Wp4[32 + c4], w2 = Wp4[64 + c4];
      v = float4(((p0 * w0.x) + p1 * w1.x) + p2 * w2.x,
                 ((p0 * w0.y) + p1 * w1.y) + p2 * w2.y,
                 ((p0 * w0.z) + p1 * w1.z) + p2 * w2.z,
                 ((p0 * w0.w) + p1 * w1.w) + p2 * w2.w);
      if (a.chiralGrads) {
        device const float* g = a.chiralGrads + ((ulong)at + k * a.atoms) * 3;
        float g0 = g[0], g1 = g[1], g2 = g[2];
        device const float4* Wc4 = (device const float4*)a.Wc;
        float4 c0 = Wc4[c4], c1 = Wc4[32 + c4], c2 = Wc4[64 + c4];
        v += float4(((g0 * c0.x) + g1 * c1.x) + g2 * c2.x,
                    ((g0 * c0.y) + g1 * c1.y) + g2 * c2.y,
                    ((g0 * c0.z) + g1 * c1.z) + g2 * c2.z,
                    ((g0 * c0.w) + g1 * c1.w) + g2 * c2.w);
      }
    }
    float4 xv = ((device const float4*)a.qStart)[(ulong)gq * 32 + c4] + v * a.qMask[gq];
    ((device float4*)a.act)[t] = xv;
    if (a.xq) {
      float mean = simd_sum(xv.x + xv.y + xv.z + xv.w) / 128.f;
      float4 d = xv - mean;
      float inv = rsqrt(simd_sum(d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w) / 128.f + a.eps);
      float4 nrm = d * inv;
      ulong pr = (ulong)gq * 128;
      float4 y = ((device const float4*)(a.qScale + pr))[c4] * nrm + ((device const float4*)(a.qShift + pr))[c4];
      ((device half4*)a.xq)[t] = half4(y);
      float4 ksv = ((device const float4*)(a.kScale + pr))[c4];
      float4 ksh = ((device const float4*)(a.kShift + pr))[c4];
      if (!a.chained) {
        ((device half4*)a.xk)[t] = half4(ksv * nrm + ksh);
      } else {
        float meank = simd_sum(y.x + y.y + y.z + y.w) / 128.f;
        float4 dk = y - meank;
        float invk = rsqrt(simd_sum(dk.x * dk.x + dk.y * dk.y + dk.z * dk.z + dk.w * dk.w) / 128.f + a.eps);
        ((device half4*)a.xk)[t] = half4(ksv * (dk * invk) + ksh);
      }
      if (a.fOut) {
        float4 fsv = ((device const float4*)(a.fScale + pr))[c4];
        float4 fsh = ((device const float4*)(a.fShift + pr))[c4];
        ((device half4*)a.fOut)[t] = half4(fsv * nrm + fsh);
      }
    }
    return;
  }
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.rows * C4) return;
    uint u = (uint)t, q = lf_udiv(u, C4), c4 = u - q * C4;
    uint q1 = (uint)a.q1, k = lf_udiv(q, q1), gq = q - k * q1;
    float4 v = float4(0.f);
    if (a.gmask[gq] != 0.f) {
      uint at = (uint)a.idx[gq];
      device const float* p = a.pos + ((ulong)at + k * a.atoms) * 3;
      float s = a.atomMask ? a.atomMask[at] * a.posScale : a.posScale;
      float p0 = p[0] * s, p1 = p[1] * s, p2 = p[2] * s;
      device const float4* Wp4 = (device const float4*)a.Wp;
      float4 w0 = Wp4[c4], w1 = Wp4[C4 + c4], w2 = Wp4[2 * C4 + c4];
      v = float4(((p0 * w0.x) + p1 * w1.x) + p2 * w2.x,
                 ((p0 * w0.y) + p1 * w1.y) + p2 * w2.y,
                 ((p0 * w0.z) + p1 * w1.z) + p2 * w2.z,
                 ((p0 * w0.w) + p1 * w1.w) + p2 * w2.w);
      if (a.chiralGrads) {
        device const float* g = a.chiralGrads + ((ulong)at + k * a.atoms) * 3;
        float g0 = g[0], g1 = g[1], g2 = g[2];
        device const float4* Wc4 = (device const float4*)a.Wc;
        float4 c0 = Wc4[c4], c1 = Wc4[C4 + c4], c2 = Wc4[2 * C4 + c4];
        v += float4(((g0 * c0.x) + g1 * c1.x) + g2 * c2.x,
                    ((g0 * c0.y) + g1 * c1.y) + g2 * c2.y,
                    ((g0 * c0.z) + g1 * c1.z) + g2 * c2.z,
                    ((g0 * c0.w) + g1 * c1.w) + g2 * c2.w);
      }
    }
    ((device float4*)a.act)[t] = ((device const float4*)a.qStart)[(ulong)gq * C4 + c4] + v * a.qMask[gq];
    return;
  }
  if (t >= a.rows * a.C) return;
  uint u = (uint)t, q = lf_udiv(u, a.C), c = u - q * a.C;
  uint q1 = (uint)a.q1, k = lf_udiv(q, q1), gq = q - k * q1;
  float v = 0.f;
  if (a.gmask[gq] != 0.f) {
    uint at = (uint)a.idx[gq];
    device const float* p = a.pos + ((ulong)at + k * a.atoms) * 3;
    float s = a.atomMask ? a.atomMask[at] * a.posScale : a.posScale;
    float p0 = p[0] * s, p1 = p[1] * s, p2 = p[2] * s;
    v = ((p0 * a.Wp[c]) + p1 * a.Wp[a.C + c]) + p2 * a.Wp[2 * a.C + c];
    if (a.chiralGrads) {
      device const float* g = a.chiralGrads + ((ulong)at + k * a.atoms) * 3;
      v += (g[0] * a.Wc[c] + g[1] * a.Wc[a.C + c]) + g[2] * a.Wc[2 * a.C + c];
    }
  }
  a.act[t] = a.qStart[gq * a.C + c] + v * a.qMask[gq];
}
kernel void af3_aggregate(constant AggregateArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint3 tpg [[threads_per_threadgroup]]) {
  ulong token = ((ulong)tg.y * ng.x + tg.x) * (tpg.x >> 5) + sg;
  if (token >= a.rows) return;
  uint t32 = (uint)token, k = lf_udiv(t32, a.tokens), tok = t32 - k * a.tokens, g0 = tok * a.dense;
  if (a.Cp == 768) {
    float count = 0.f;
    float4 sum[6] = {float4(0.f), float4(0.f), float4(0.f), float4(0.f), float4(0.f), float4(0.f)};
    device const float4* proj4 = (device const float4*)a.projected + lane;
    for (uint at = 0; at < a.dense; ++at) {
      float m = a.atomMask[g0 + at];
      count += m;
      if (m != 0.f && a.gmask[g0 + at] != 0.f) {
        device const float4* p4 = proj4 + ((ulong)a.idx[g0 + at] + (ulong)k * a.q1) * 192;
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < 6; ++i) sum[i] += max(p4[i * 32], float4(0.f));
      }
    }
    device const float4* tb4 = a.tokenBias ? (device const float4*)a.tokenBias + (ulong)tok * 192 + lane : nullptr;
    device float4* out4 = (device float4*)a.out + token * 192 + lane;
    float4 xv[6];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 6; ++i) {
      float4 res = count > 0.f ? sum[i] / count : float4(0.f);
      if (tb4) res += tb4[i * 32];
      out4[i * 32] = res;
      xv[i] = res;
    }
    if (a.xOut) {
      float s = 0.f;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 6; ++i) s += xv[i].x + xv[i].y + xv[i].z + xv[i].w;
      float mean = simd_sum(s) / 768.f, v = 0.f;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 6; ++i) { float4 d = xv[i] - mean; v += dot(d, d); }
      float inv = rsqrt(simd_sum(v) / 768.f + a.eps);
      ulong pr = (ulong)tok * a.ld;
      device const float4* sc4 = (device const float4*)(a.scale + pr) + lane;
      device const float4* sh4 = (device const float4*)(a.shift + pr) + lane;
      device half4* xout4 = (device half4*)a.xOut + token * 192 + lane;
      device const float4* fsc4 = a.fOut ? (device const float4*)(a.fScale + pr) + lane : nullptr;
      device const float4* fsh4 = a.fOut ? (device const float4*)(a.fShift + pr) + lane : nullptr;
      device half4* fout4 = a.fOut ? (device half4*)a.fOut + token * 192 + lane : nullptr;
      const bool rawS = a.raw != 0;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < 6; ++i) {
        float4 nrm = (xv[i] - mean) * inv;
        float4 sv = sc4[i * 32];
        xout4[i * 32] = half4((rawS ? sv : float4(lf_sigmoid(sv.x), lf_sigmoid(sv.y), lf_sigmoid(sv.z), lf_sigmoid(sv.w))) * nrm + sh4[i * 32]);
        if (fout4) {
          float4 fsv = fsc4[i * 32];
          fout4[i * 32] = half4((rawS ? fsv : float4(lf_sigmoid(fsv.x), lf_sigmoid(fsv.y), lf_sigmoid(fsv.z), lf_sigmoid(fsv.w))) * nrm + fsh4[i * 32]);
        }
      }
    }
    return;
  }
  if ((a.Cp & 3) == 0) {
    uint Cp4 = a.Cp >> 2;
    device const float4* proj4 = (device const float4*)a.projected;
    for (uint c4 = lane; c4 < Cp4; c4 += 32) {
      float count = 0.f;
      float4 sum = float4(0.f);
      for (uint at = 0; at < a.dense; ++at) {
        float m = a.atomMask[g0 + at];
        count += m;
        if (m != 0.f && a.gmask[g0 + at] != 0.f) {
          float4 v = proj4[((ulong)a.idx[g0 + at] + (ulong)k * a.q1) * Cp4 + c4];
          sum += max(v, float4(0.f));
        }
      }
      float4 res = count > 0.f ? sum / count : float4(0.f);
      if (a.tokenBias) res += ((device const float4*)a.tokenBias)[(ulong)tok * Cp4 + c4];
      ((device float4*)a.out)[token * Cp4 + c4] = res;
    }
    return;
  }
  for (uint c = lane; c < a.Cp; c += 32) {
    float count = 0.f, sum = 0.f;
    for (uint at = 0; at < a.dense; ++at) {
      float m = a.atomMask[g0 + at];
      count += m;
      if (m != 0.f) {
        float v = a.gmask[g0 + at] != 0.f ? a.projected[((ulong)a.idx[g0 + at] + (ulong)k * a.q1) * a.Cp + c] : 0.f;
        sum += max(v, 0.f);
      }
    }
    float res = count > 0.f ? sum / count : 0.f;
    if (a.tokenBias) res += a.tokenBias[(ulong)tok * a.Cp + c];
    a.out[token * a.Cp + c] = res;
  }
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
  // Boltz's DN, the DNA unknown: DN's column less N's on a DNA token, by its class-30 weight
  if (a.dnRes && a.isDna && a.isDna[token] != 0) v += (aa == 30 ? a.dnRes[c] : 0.f) + a.profile[token * 31 + 30] * a.dnProf[c];
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
  uint t = (uint)LF_INDEX;
  if (t >= a.K * a.n) return;
  uint k = lf_udiv(t, a.n), c = t - k * a.n;
  float tr = 0.25f * log(a.levels[k] / 16.f);
  a.out[t] = cos(6.283185307179586f * (tr * a.w[c] + a.b[c]));
}
kernel void af3_single_batch_init(LF_ARGS(SingleBatchInitArgs)) {
  uint t = (uint)LF_INDEX;
  if ((a.Cs & 3) == 0) {
    uint Cs4 = a.Cs >> 2;
    if (t >= a.K * a.n * Cs4) return;
    uint r = lf_udiv(t, Cs4), c4 = t - r * Cs4, k = lf_udiv(r, a.n), i = r - k * a.n;
    ((device float4*)a.out)[t] = ((device const float4*)a.base)[(ulong)i * Cs4 + c4] + ((device const float4*)a.projs)[(ulong)k * Cs4 + c4];
    return;
  }
  if (t >= a.K * a.n * a.Cs) return;
  uint r = lf_udiv(t, a.Cs), c = t - r * a.Cs, k = lf_udiv(r, a.n), i = r - k * a.n;
  a.out[t] = a.base[(ulong)i * a.Cs + c] + a.projs[(ulong)k * a.Cs + c];
}
kernel void af3_cond_norm_cast(constant CondNormCastArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                               uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  ulong row = ((ulong)tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= (ulong)a.K * a.pn) return;
  uint k = lf_udiv((uint)row, a.pn), r = (uint)row - k * a.pn;
  device half* cn = a.cn + row * a.Cc;
  device half* ch = a.ch + row * a.Cc;
  if ((a.Cc & 3) == 0 && a.Cc <= 512) {
    uint C4 = a.Cc >> 2;
    device half4* cn4 = (device half4*)cn;
    device half4* ch4 = (device half4*)ch;
    if (r >= a.n) {
      for (uint c4 = lane; c4 < C4; c4 += 32) { cn4[c4] = half4(0); ch4[c4] = half4(0); }
      return;
    }
    device const float4* x4 = (device const float4*)(a.cond + ((ulong)k * a.n + r) * a.Cc);
    float4 xv[4];
    float s = 0.f;
    for (int i = 0; i < 4; ++i) {
      uint c4 = lane + 32 * i;
      if (c4 < C4) { xv[i] = x4[c4]; s += xv[i].x + xv[i].y + xv[i].z + xv[i].w; }
    }
    float mean = simd_sum(s) / a.Cc, v = 0.f;
    for (int i = 0; i < 4; ++i)
      if (lane + 32 * i < C4) { float4 d = xv[i] - mean; v += dot(d, d); }
    float inv = rsqrt(simd_sum(v) / a.Cc + 1e-5f);
    for (int i = 0; i < 4; ++i) {
      uint c4 = lane + 32 * i;
      if (c4 < C4) {
        cn4[c4] = half4((xv[i] - mean) * inv);
        ch4[c4] = half4(xv[i]);
      }
    }
    return;
  }
  if (r >= a.n) {
    for (uint c = lane; c < a.Cc; c += 32) { cn[c] = (half)0; ch[c] = (half)0; }
    return;
  }
  device const float* x = a.cond + ((ulong)k * a.n + r) * a.Cc;
  float s = 0.f;
  for (uint c = lane; c < a.Cc; c += 32) s += x[c];
  float mean = simd_sum(s) / a.Cc, v = 0.f;
  for (uint c = lane; c < a.Cc; c += 32) { float d = x[c] - mean; v += d * d; }
  float inv = rsqrt(simd_sum(v) / a.Cc + 1e-5f);
  for (uint c = lane; c < a.Cc; c += 32) {
    float xc = x[c];
    cn[c] = (half)((xc - mean) * inv);
    ch[c] = (half)xc;
  }
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
  float g = a.gate[(ulong)(r - lf_udiv(r, a.period) * a.period) * a.ld + c];
  a.act[t] += a.y[t] * (a.pad ? g : lf_sigmoid(g));
}
kernel void af3_add_broadcast(LF_ARGS(AddBroadcastArgs)) {
  ulong t = LF_INDEX;
  if ((a.n & 3) == 0 && (a.total & 3) == 0) {
    uint n4 = (uint)(a.n >> 2);
    if (t < (a.total >> 2)) ((device float4*)a.x)[t] += ((device const float4*)a.v)[(uint)t - lf_udiv((uint)t, n4) * n4];
    return;
  }
  if (t < a.total) a.x[t] += a.v[t % a.n];
}
kernel void af3_broadcast_skip(LF_ARGS(BroadcastSkipArgs)) {
  ulong t = LF_INDEX;
  if (a.C == 128) {
    if (t >= a.rows * 32) return;
    uint u = (uint)t, q = u >> 5, c4 = u & 31;
    uint q1 = (uint)a.q1, k = lf_udiv(q, q1), gq = q - k * q1;
    float4 v = a.gmask[gq] != 0.f ? ((device const float4*)a.proj)[((ulong)lf_udiv((uint)a.idx[gq], a.dense) + k * a.tokens) * 32 + c4] : float4(0.f);
    float4 xv = (v + ((device const float4*)a.skip)[t]) * a.qMask[gq];
    ((device float4*)a.act)[t] = xv;
    if (a.xq) {
      float mean = simd_sum(xv.x + xv.y + xv.z + xv.w) / 128.f;
      float4 d = xv - mean;
      float inv = rsqrt(simd_sum(d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w) / 128.f + a.eps);
      float4 nrm = d * inv;
      ulong pr = (ulong)gq * 128;
      float4 y = ((device const float4*)(a.qScale + pr))[c4] * nrm + ((device const float4*)(a.qShift + pr))[c4];
      ((device half4*)a.xq)[t] = half4(y);
      float4 ksv = ((device const float4*)(a.kScale + pr))[c4];
      float4 ksh = ((device const float4*)(a.kShift + pr))[c4];
      if (!a.chained) {
        ((device half4*)a.xk)[t] = half4(ksv * nrm + ksh);
      } else {
        float meank = simd_sum(y.x + y.y + y.z + y.w) / 128.f;
        float4 dk = y - meank;
        float invk = rsqrt(simd_sum(dk.x * dk.x + dk.y * dk.y + dk.z * dk.z + dk.w * dk.w) / 128.f + a.eps);
        ((device half4*)a.xk)[t] = half4(ksv * (dk * invk) + ksh);
      }
      if (a.fOut) {
        float4 fsv = ((device const float4*)(a.fScale + pr))[c4];
        float4 fsh = ((device const float4*)(a.fShift + pr))[c4];
        ((device half4*)a.fOut)[t] = half4(fsv * nrm + fsh);
      }
    }
    return;
  }
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.rows * C4) return;
    uint u = (uint)t, q = lf_udiv(u, C4), c4 = u - q * C4;
    uint q1 = (uint)a.q1, k = lf_udiv(q, q1), gq = q - k * q1;
    float4 v = a.gmask[gq] != 0.f ? ((device const float4*)a.proj)[((ulong)lf_udiv((uint)a.idx[gq], a.dense) + k * a.tokens) * C4 + c4] : float4(0.f);
    ((device float4*)a.act)[t] = (v + ((device const float4*)a.skip)[t]) * a.qMask[gq];
    return;
  }
  if (t >= a.rows * a.C) return;
  uint u = (uint)t, q = lf_udiv(u, a.C), c = u - q * a.C;
  uint q1 = (uint)a.q1, k = lf_udiv(q, q1), gq = q - k * q1;
  float v = a.gmask[gq] != 0.f ? a.proj[((ulong)lf_udiv((uint)a.idx[gq], a.dense) + k * a.tokens) * a.C + c] : 0.f;
  a.act[t] = (v + a.skip[t]) * a.qMask[gq];
}
kernel void af3_mask_ln_project3(constant MaskLnProject3Args& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  ulong row = ((ulong)tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  device const float* x = a.act + row * a.C;
  uint r32 = (uint)row, q1 = (uint)a.q1, prRow = r32 - lf_udiv(r32, q1) * q1;
  if (a.C == 128) {
    float m = a.qMask[prRow];
    device const float4* x4 = (device const float4*)x;
    float4 xv = x4[lane];
    if (a.resY) {
      float4 ry = ((device const float4*)(a.resY + row * 128))[lane];
      float4 rg = ((device const float4*)(a.resGate + (ulong)prRow * 128))[lane];
      xv += ry * rg;
    }
    xv *= m;
    float mean = simd_sum(xv.x + xv.y + xv.z + xv.w) * (1.f / 128.f);
    float4 d = xv - mean;
    float inv = rsqrt(simd_sum(dot(d, d)) * (1.f / 128.f) + 1e-5f);
    float4 l = d * inv * ((device const float4*)a.scale)[lane] +
               (a.offset ? ((device const float4*)a.offset)[lane] : float4(0.f));
    device const float* wp = a.Wp + lane * 12;
    float y0 = l.x * wp[0] + l.y * wp[3] + l.z * wp[6] + l.w * wp[9];
    float y1 = l.x * wp[1] + l.y * wp[4] + l.z * wp[7] + l.w * wp[10];
    float y2 = l.x * wp[2] + l.y * wp[5] + l.z * wp[8] + l.w * wp[11];
    y0 = simd_sum(y0); y1 = simd_sum(y1); y2 = simd_sum(y2);
    if (lane == 0) { a.upd[row * 3] = y0; a.upd[row * 3 + 1] = y1; a.upd[row * 3 + 2] = y2; }
    return;
  }
  float m = a.qMask[prRow], s = 0.f;
  for (uint c = lane; c < a.C; c += 32) {
    float xc = x[c] + (a.resY ? a.resY[row * a.C + c] * a.resGate[(ulong)prRow * a.C + c] : 0.f);
    s += xc * m;
  }
  float mean = simd_sum(s) / a.C, v = 0.f;
  for (uint c = lane; c < a.C; c += 32) {
    float xc = x[c] + (a.resY ? a.resY[row * a.C + c] * a.resGate[(ulong)prRow * a.C + c] : 0.f);
    float d = xc * m - mean; v += d * d;
  }
  float inv = rsqrt(simd_sum(v) / a.C + 1e-5f);
  float y0 = 0.f, y1 = 0.f, y2 = 0.f;
  for (uint c = lane; c < a.C; c += 32) {
    float xc = x[c] + (a.resY ? a.resY[row * a.C + c] * a.resGate[(ulong)prRow * a.C + c] : 0.f);
    float l = (xc * m - mean) * inv * a.scale[c] + (a.offset ? a.offset[c] : 0.f);
    y0 += l * a.Wp[c * 3]; y1 += l * a.Wp[c * 3 + 1]; y2 += l * a.Wp[c * 3 + 2];
  }
  y0 = simd_sum(y0); y1 = simd_sum(y1); y2 = simd_sum(y2);
  if (lane == 0) { a.upd[row * 3] = y0; a.upd[row * 3 + 1] = y1; a.upd[row * 3 + 2] = y2; }
}
kernel void af3_scale_positions(LF_ARGS(ScalePosArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.total * 3) return;
  uint at = lf_udiv((uint)t, 3u), atoms = (uint)a.atoms;
  a.y[t] = a.x[t] * a.mask[at - lf_udiv(at, atoms) * atoms] * a.in;
}
kernel void af3_denoise_out(LF_ARGS(DenoiseOutArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.total * 3) return;
  uint u = (uint)t, at = lf_udiv(u, 3u), c = u - at * 3u, atoms = (uint)a.atoms;
  uint k = lf_udiv(at, atoms), g = at - k * atoms;
  float xt = a.x[t];
  float upd = a.q2tIdx ? (a.q2tMask[g] != 0.f ? a.upd[((ulong)a.q2tIdx[g] + k * a.q1) * 3 + c] : 0.f) : a.upd[t];
  float den = (a.skip * xt + a.out * upd) * a.mask[g];
  if (a.o) a.o[t] = den;
  if (a.eulerX) a.eulerX[t] = xt + a.eulerScale * (xt - den);
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
kernel void af3_augment(constant AugmentArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4][32];
  threadgroup float ck[3];
  const ulong k = tg.x;
  device float* xs = a.x + k * a.atoms * 3;
  device float* noisy = a.noisy + k * a.atoms * 3;
  float s[4] = {0, 0, 0, 0};
  for (ulong i = tid; i < a.atoms; i += 1024)
    if (a.mask[i] != 0.f) { s[0] += xs[i * 3]; s[1] += xs[i * 3 + 1]; s[2] += xs[i * 3 + 2]; s[3] += 1.f; }
  for (int d = 0; d < 4; ++d) { float v = simd_sum(s[d]); if (lane == 0) part[d][sg] = v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v[4];
    for (int d = 0; d < 4; ++d) v[d] = simd_sum(part[d][lane]);
    if (lane < 3) ck[lane] = v[lane] / (v[3] + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float c0 = ck[0], c1 = ck[1], c2 = ck[2];
  device const float* r = a.rot + k * 12;
  const ulong seed = a.seeds[k];
  for (ulong i = tid; i < a.atoms; i += 1024) {
    float p[3] = {xs[i * 3] - c0, xs[i * 3 + 1] - c1, xs[i * 3 + 2] - c2};
    bool live = a.mask[i] != 0.f;
    for (int d = 0; d < 3; ++d) {
      float v = live ? p[0] * r[d] + p[1] * r[3 + d] + p[2] * r[6 + d] + r[9 + d] : 0.f;
      xs[i * 3 + d] = v;
      noisy[i * 3 + d] = v + a.injected * lf_gaussian(seed, a.step, i * 3 + d);
    }
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
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.n * a.n * C4) return;
    uint ij = lf_udiv((uint)t, C4), c4 = (uint)t - ij * C4, i = lf_udiv(ij, a.n), j = ij - i * a.n;
    float4 v = ((device const float4*)a.left)[j * C4 + c4] + ((device const float4*)a.right)[i * C4 + c4];
    int bin = a.bin[ij];
    if (bin >= 0) v += ((device const float4*)a.Wd)[bin * C4 + c4] * (a.unmasked ? 1.f : a.pairMask[ij]);
    if (!a.caBins && a.Wdist) v += sqrt(a.sq[ij] + 1e-10f) * ((device const float4*)a.Wdist)[c4];
    float4 p = (a.srcPair ? ((device const float4*)a.srcPair)[t] : ((device const float4*)a.pair)[t]) + v;
    if (a.outH) ((device half4*)a.outH)[t] = half4(p);
    else ((device float4*)a.pair)[t] = p;
    return;
  }
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  float v = a.left[j * a.C + c] + a.right[i * a.C + c];
  int bin = a.bin[ij];
  if (bin >= 0) v += a.Wd[bin * a.C + c] * (a.unmasked ? 1.f : a.pairMask[ij]);
  if (!a.caBins && a.Wdist) v += sqrt(a.sq[ij] + 1e-10f) * a.Wdist[c];
  float p = (a.srcPair ? a.srcPair[t] : a.pair[t]) + v;
  if (a.outH) a.outH[t] = (half)p;
  else a.pair[t] = p;
}
kernel void af3_expectation(LF_ARGS(ExpectationArgs)) {
  ulong r = LF_INDEX;
  if (r >= a.rows) return;
  ulong rt = r;
  if (a.symmetricN) { ulong i = r / a.symmetricN, j = r - i * a.symmetricN; rt = j * a.symmetricN + i; }
  float mx = -1e30f;
  for (uint b = 0; b < a.bins; ++b) mx = max(mx, a.logits[r * a.bins + b] + (a.symmetricN ? a.logits[rt * a.bins + b] : 0.f));
  float total = 0.f, weighted = 0.f, weighted2 = 0.f;
  for (uint b = 0; b < a.bins; ++b) {
    float p = exp(a.logits[r * a.bins + b] + (a.symmetricN ? a.logits[rt * a.bins + b] : 0.f) - mx);
    total += p; weighted += p * a.centres[b];
    if (a.out2) weighted2 += p * a.centres2[b];
  }
  a.out[r] = weighted / total * a.scale * (a.mask ? a.mask[r] : 1.f);
  if (a.out2) a.out2[r] = weighted2 / total * a.scale;
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
  threadgroup half KV[K * LD];
  const uint s = tg.x, h = tg.y, ks = lf_udiv(s, a.subsets), ss = s - ks * a.subsets, Wd = a.heads * D, W2 = 2 * Wd;
  const float qs = rsqrt((float)D) * M_LOG2E_F;
  if (D == 32 || tid < Q * D8) {
    uint qi = tid / D8, d = (tid - qi * D8) * 8;
    device const half* src = a.qg + ((ulong)s * Q + qi) * W2 + h * D + d;
    float4 f0 = float4(*(device const half4*)src) + *(device const float4*)(a.qBias + h * D + d);
    float4 f1 = float4(*(device const half4*)(src + 4)) + *(device const float4*)(a.qBias + h * D + d + 4);
    *(threadgroup half4*)(KV + qi * LD + d) = half4(f0 * qs);
    *(threadgroup half4*)(KV + qi * LD + d + 4) = half4(f1 * qs);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[DF];
  _Pragma("clang loop unroll(full)") for (int dk = 0; dk < DF; ++dk) simdgroup_load(qf[dk], KV + (sg * 8) * LD + dk * 8, LD);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong srcBase = (ulong)ks * a.subsets * Q;
  const bool fullK = a.keys == K;
  uint4 vReg[D8];
  if (a.q2kIdx) {
    _Pragma("clang loop unroll(full)")
    for (uint s_step = 0; s_step < D8; ++s_step) {
      uint e = tid + s_step * 128;
      uint kj = e / D8, d = (e - kj * D8) * 8, g = ss * a.keys + kj;
      if ((fullK || kj < a.keys) && a.q2kMask[g] != 0.f) {
        device const half* src = a.kv + (srcBase + (uint)a.q2kIdx[g]) * W2 + h * D + d;
        *(threadgroup uint4*)(KV + kj * LD + d) = *(device const uint4*)src;
        vReg[s_step] = *(device const uint4*)(src + Wd);
      } else {
        *(threadgroup uint4*)(KV + kj * LD + d) = uint4(0);
        vReg[s_step] = uint4(0);
      }
    }
  } else {
    _Pragma("clang loop unroll(full)")
    for (uint s_step = 0; s_step < D8; ++s_step) {
      uint e = tid + s_step * 128;
      uint kj = e / D8, d = (e - kj * D8) * 8;
      if (fullK || kj < a.keys) {
        device const half* src = a.kv + ((ulong)s * a.keys + kj) * W2 + h * D + d;
        *(threadgroup uint4*)(KV + kj * LD + d) = *(device const uint4*)src;
        vReg[s_step] = *(device const uint4*)(src + Wd);
      } else {
        *(threadgroup uint4*)(KV + kj * LD + d) = uint4(0);
        vReg[s_step] = uint4(0);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 sc[KF];
  _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j) sc[j] = simdgroup_float8x8(0);
  _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j)
    _Pragma("clang loop unroll(full)") for (int dk = 0; dk < DF; ++dk) {
      simdgroup_half8x8 kt;
      simdgroup_load(kt, KV + (j * 8) * LD + dk * 8, LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(sc[j], qf[dk], kt, sc[j]);
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (uint s_step = 0; s_step < D8; ++s_step) {
    uint e = tid + s_step * 128;
    uint kj = e / D8, d = (e - kj * D8) * 8;
    *(threadgroup uint4*)(KV + kj * LD + d) = vReg[s_step];
  }
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  const uint q = sg * 8 + sm;
  const float qm = a.qMask[(ulong)ss * Q + q];
  device const float* pl = a.logits + (((ulong)ss * a.heads + h) * Q + q) * a.keys;
  device const float* km = a.kMask + (ulong)ss * a.keys;
  float mx = -1e30f;
  if (fullK) {
    _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j) {
      thread auto& e = sc[j].thread_elements();
      const uint k = j * 8 + sn;
      const float2 pv = *(device const float2*)(pl + k);
      const float2 kv = *(device const float2*)(km + k);
      const float2 mb = a.keyMasked ? -1e9f * ((1.f - qm) + (1.f - kv)) : (1e9f * (qm - 1.f)) * (kv - 1.f);
      e[0] += (pv.x + mb.x) * M_LOG2E_F;
      e[1] += (pv.y + mb.y) * M_LOG2E_F;
      mx = max(mx, max((float)e[0], (float)e[1]));
    }
  } else {
    _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j) {
      thread auto& e = sc[j].thread_elements();
      _Pragma("clang loop unroll(full)") for (int t = 0; t < 2; ++t) {
        const uint k = j * 8 + sn + t;
        if (k < a.keys) {
          float kk = km[k];
          float maskBias = a.keyMasked ? -1e9f * ((1.f - qm) + (1.f - kk)) : 1e9f * (qm - 1.f) * (kk - 1.f);
          e[t] += (pl[k] + maskBias) * M_LOG2E_F;
        } else {
          e[t] = -1e30f;
        }
        mx = max(mx, (float)e[t]);
      }
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 o[DF];
  _Pragma("clang loop unroll(full)") for (int c = 0; c < DF; ++c) o[c] = simdgroup_float8x8(0);
  _Pragma("clang loop unroll(full)") for (int j = 0; j < KF; ++j)
    _Pragma("clang loop unroll(full)") for (int c = 0; c < DF; ++c) {
      simdgroup_half8x8 vf;
      simdgroup_load(vf, KV + (j * 8) * LD + c * 8, LD);
      simdgroup_multiply_accumulate(o[c], p[j], vf, o[c]);
    }
  const ulong qrow = (ulong)s * Q + q;
  device const half* g = a.qg + qrow * W2 + Wd + h * D;
  const float inv = 1.f / sum;
  if (a.act) {
    device float* actOut = a.act + qrow * Wd + h * D;
    device const float* zg = a.zg + ((ulong)ss * Q + q) * Wd + h * D;
    _Pragma("clang loop unroll(full)") for (int c = 0; c < DF; ++c) {
      thread auto& oe = o[c].thread_elements();
      const int d = c * 8 + sn;
      half2 gv = *(device const half2*)(g + d);
      half2 yh(half(oe[0] * inv * lf_sigmoid((float)gv.x)), half(oe[1] * inv * lf_sigmoid((float)gv.y)));
      float2 zv = *(device const float2*)(zg + d);
      *(device float2*)(actOut + d) += float2(yh) * 2.f * zv;
    }
    return;
  }
  device half* out = a.out + qrow * Wd + h * D;
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
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.n * a.n * C4) return;
    uint ij = lf_udiv((uint)t, C4), c4 = (uint)t - ij * C4, i = lf_udiv(ij, a.n), j = ij - i * a.n;
    int o = a.orders ? (int)a.orders[ij] : 0;
    if (o < 0 || o >= 7) o = 0;
    ((device float4*)a.pair)[t] += ((device const float4*)a.right)[i * C4 + c4] +
                                   ((device const float4*)a.left)[j * C4 + c4] +
                                   ((device const float4*)a.Wd)[a.bin[ij] * C4 + c4] * a.pairMask[ij] +
                                   (a.bonds ? a.bonds[ij] * ((device const float4*)a.wBond)[c4] : float4(0.f)) +
                                   ((device const float4*)a.wBondType)[o * C4 + c4] +
                                   ((device const float4*)a.unspecified)[c4];
    return;
  }
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  int o = a.orders ? (int)a.orders[ij] : 0;
  if (o < 0 || o >= 7) o = 0;
  a.pair[t] += a.right[i * a.C + c] + a.left[j * a.C + c] + a.Wd[a.bin[ij] * a.C + c] * a.pairMask[ij] +
               (a.bonds ? a.bonds[ij] * a.wBond[c] : 0.f) + a.wBondType[o * a.C + c] + a.unspecified[c];
}
kernel void af3_outer_prod(LF_ARGS(OuterProdArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.rows * a.n * C4) return;
    uint ij = lf_udiv((uint)t, C4), e4 = (uint)t - ij * C4, ii = lf_udiv(ij, a.n), j = ij - ii * a.n;
    ((device half4*)a.out)[t] = half4(((device const float4*)a.a)[(a.i0 + ii) * C4 + e4] * ((device const float4*)a.b)[j * C4 + e4]);
    return;
  }
  if (t >= (ulong)a.rows * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), e = (uint)t - ij * a.C, ii = lf_udiv(ij, a.n), j = ij - ii * a.n;
  a.out[t] = (half)(a.a[(a.i0 + ii) * a.C + e] * a.b[j * a.C + e]);
}

// ---------------------------------------------------------------- rosettafold3
kernel void af3_kq_norm(constant KqNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint3 tpg [[threads_per_threadgroup]]) {
  ulong rs = ((ulong)tg.y * ng.x + tg.x) * (tpg.x >> 5) + sg;
  if (rs >= 2 * a.rows) return;
  ulong row = rs >> 1;
  uint side = (uint)(rs & 1);
  device half* x = side ? a.k + row * a.ldk : a.q + row * a.ldq;
  device const float* sc = side ? a.ks : a.qs;
  device const float* of = side ? a.ko : a.qo;
  device const float* qb = (side || !a.qBias) ? nullptr : a.qBias;
  if (a.Wd == 768 && (a.ldq & 3) == 0 && (a.ldk & 3) == 0) {
    device half4* x4 = (device half4*)x;
    device const float4* sc4 = (device const float4*)sc;
    device const float4* of4 = (device const float4*)of;
    device const float4* qb4 = (device const float4*)qb;
    float4 xv[6]; float s = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 6; ++i) {
      uint idx = lane + i * 32;
      float4 val = float4(x4[idx]) + (qb4 ? qb4[idx] : float4(0.f));
      xv[i] = val;
      s += val.x + val.y + val.z + val.w;
    }
    float mean = simd_sum(s) / 768.f, v = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 6; ++i) { float4 d = xv[i] - mean; v += dot(d, d); }
    float inv = rsqrt(simd_sum(v) / 768.f + 1e-5f);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < 6; ++i) {
      uint idx = lane + i * 32;
      x4[idx] = half4((xv[i] - mean) * inv * sc4[idx] + of4[idx]);
    }
    return;
  }
  if (a.Wd == 128 && (a.ldq & 3) == 0 && (a.ldk & 3) == 0) {
    device half4* x4 = (device half4*)x;
    device const float4* sc4 = (device const float4*)sc;
    device const float4* of4 = (device const float4*)of;
    device const float4* qb4 = (device const float4*)qb;
    float4 xv = float4(x4[lane]) + (qb4 ? qb4[lane] : float4(0.f));
    float mean = simd_sum(xv.x + xv.y + xv.z + xv.w) / 128.f;
    float4 d = xv - mean;
    float inv = rsqrt(simd_sum(dot(d, d)) / 128.f + 1e-5f);
    x4[lane] = half4(d * inv * sc4[lane] + of4[lane]);
    return;
  }
  float s = 0.f;
  for (uint c = lane; c < a.Wd; c += 32) s += (float)x[c] + (qb ? qb[c] : 0.f);
  float mean = simd_sum(s) / a.Wd, v = 0.f;
  for (uint c = lane; c < a.Wd; c += 32) { float d = (float)x[c] + (qb ? qb[c] : 0.f) - mean; v += d * d; }
  float inv = rsqrt(simd_sum(v) / a.Wd + 1e-5f);
  for (uint c = lane; c < a.Wd; c += 32)
    x[c] = (half)(((float)x[c] + (qb ? qb[c] : 0.f) - mean) * inv * sc[c] + of[c]);
}
// (float, where the reference's is double - Metal has none; the step is 1e-3 rather than 1e-4 for it)
inline float lf_improper(float3 a, float3 b, float3 c, float3 d) {
  const float eps = 1e-6f;
  float3 b0 = a - b, b1 = c - b, b2 = d - c;
  float3 n = b1 / (length(b1) + eps);
  float3 v = b0 - dot(b0, n) * n, w = b2 - dot(b2, n) * n;
  return atan2(dot(cross(n, v), w) + eps, dot(v, w) + eps);
}
inline float3 lf_chiral_pos(device const float* x, constant ChiralGradArgs& a, int idx) {
  uint ca = (uint)idx;
  float s = a.atomMask ? a.atomMask[ca] * a.posScale : a.posScale;
  ulong at = (ulong)ca * 3;
  return float3(x[at] * s, x[at + 1] * s, x[at + 2] * s);
}
kernel void af3_chiral_grad(LF_ARGS(ChiralGradArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.atoms * a.ns) return;
  uint u = (uint)t, atoms = (uint)a.atoms, k = lf_udiv(u, atoms), atom = u - k * atoms;
  device const float* x = a.positions + (ulong)k * a.atoms * 3;
  float3 g = float3(0.f);
  const float step = 1e-3f;
  for (int e = a.offsets[atom]; e < a.offsets[atom + 1]; ++e) {
    int centre = a.entries[e] >> 2, corner = a.entries[e] & 3;
    float ideal = a.angles[centre];
    if (ideal == 0.f) continue;
    int4 c4 = ((device const int4*)a.centers)[centre];
    float3 p0 = lf_chiral_pos(x, a, c4.x), p1 = lf_chiral_pos(x, a, c4.y);
    float3 p2 = lf_chiral_pos(x, a, c4.z), p3 = lf_chiral_pos(x, a, c4.w);
    _Pragma("clang loop unroll(full)")
    for (int d = 0; d < 3; ++d) {
      float3 p0u = p0, p1u = p1, p2u = p2, p3u = p3;
      float3 p0d = p0, p1d = p1, p2d = p2, p3d = p3;
      if (corner == 0) { p0u[d] += step; p0d[d] -= step; }
      else if (corner == 1) { p1u[d] += step; p1d[d] -= step; }
      else if (corner == 2) { p2u[d] += step; p2d[d] -= step; }
      else { p3u[d] += step; p3d[d] -= step; }
      float up = lf_improper(p0u, p1u, p2u, p3u) - ideal;
      float down = lf_improper(p0d, p1d, p2d, p3d) - ideal;
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
  if (t >= a.n) return;
  float v = lf_silu(a.x[t]);
  if (a.outH) a.outH[t] = (half)v;
  else a.x[t] = v;
}
kernel void af3_single_struct(LF_ARGS(SingleStructArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.C) return;
  uint i = lf_udiv((uint)t, a.C), c = (uint)t - i * a.C;
  a.out[t] = a.a[t] + a.b[t] + a.roleEmb[a.role[i] * a.C + c];
}
kernel void af3_gather_pair_sorted(LF_ARGS(GatherPairSortedArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.rows * C4) return;
    ulong r = t / C4; uint c4 = (uint)(t - r * C4);
    int ij = a.order[r], i = ij / (int)a.n, j = ij - i * (int)a.n;
    ((device half4*)a.out)[t] = half4(((device const float4*)a.pair)[((ulong)a.parent[i] * a.nRes + a.parent[j]) * C4 + c4]);
    return;
  }
  if (t >= a.rows * a.C) return;
  ulong r = t / a.C; uint c = (uint)(t - r * a.C);
  int ij = a.order[r], i = ij / (int)a.n, j = ij - i * (int)a.n;
  a.out[t] = (half)a.pair[((ulong)a.parent[i] * a.nRes + a.parent[j]) * a.C + c];
}
kernel void af3_scatter_pair(LF_ARGS(ScatterPairArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.rows * C4) return;
    ulong r = t / C4; uint c4 = (uint)(t - r * C4);
    int ij = a.order[r], i = ij / (int)a.n, j = ij - i * (int)a.n;
    float4 v = ((device const float4*)a.trunkPair)[((ulong)a.parent[i] * a.nRes + a.parent[j]) * C4 + c4] +
               ((device const float4*)a.projected)[t] +
               ((device const float4*)a.eSame)[a.sameParent[ij] * C4 + c4] +
               ((device const float4*)a.eTwin)[a.twin[ij] * C4 + c4] +
               ((device const float4*)a.ePrev)[a.prev[ij] * C4 + c4] +
               ((device const float4*)a.eNext)[a.next[ij] * C4 + c4] +
               ((device const float4*)a.eType)[a.type[ij] * C4 + c4];
    if (a.outH) ((device half4*)a.outH)[(ulong)ij * C4 + c4] = half4(v);
    else ((device float4*)a.pair)[(ulong)ij * C4 + c4] = v;
    return;
  }
  if (t >= a.rows * a.C) return;
  ulong r = t / a.C; uint c = (uint)(t - r * a.C);
  int ij = a.order[r], i = ij / (int)a.n, j = ij - i * (int)a.n;
  float v = a.trunkPair[((ulong)a.parent[i] * a.nRes + a.parent[j]) * a.C + c] + a.projected[t] +
            a.eSame[a.sameParent[ij] * a.C + c] + a.eTwin[a.twin[ij] * a.C + c] + a.ePrev[a.prev[ij] * a.C + c] +
            a.eNext[a.next[ij] * a.C + c] + a.eType[a.type[ij] * a.C + c];
  if (a.outH) a.outH[(ulong)ij * a.C + c] = (half)v;
  else a.pair[(ulong)ij * a.C + c] = v;
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
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.n * a.n * C4) return;
    uint ij = lf_udiv((uint)t, C4), c4 = (uint)t - ij * C4, i = lf_udiv(ij, a.n), j = ij - i * a.n;
    float sq = 0.f;
    for (int k = 0; k < 3; ++k) { float d = a.coords[i * 3 + k] - a.coords[j * 3 + k]; sq += d * d; }
    float distance = sqrt(max(1e-10f, sq));
    int bin = (int)floor((distance - 3.25f) / 1.25f);
    if (distance < 3.25f) bin = -1;
    if (bin >= (int)a.bins) bin = a.bins - 1;
    float4 v = ((device const float4*)a.s1)[j * C4 + c4] + ((device const float4*)a.s2)[i * C4 + c4] +
               (bin >= 0 ? ((device const float4*)a.Wd)[bin * C4 + c4] : float4(0.f)) +
               distance * ((device const float4*)a.Wraw)[c4];
    float4 p = (a.srcPair ? ((device const float4*)a.srcPair)[t] : ((device const float4*)a.pair)[t]) + v;
    if (a.outH) ((device half4*)a.outH)[t] = half4(p);
    else ((device float4*)a.pair)[t] = p;
    return;
  }
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  float sq = 0.f;
  for (int k = 0; k < 3; ++k) { float d = a.coords[i * 3 + k] - a.coords[j * 3 + k]; sq += d * d; }
  float distance = sqrt(max(1e-10f, sq));
  int bin = (int)floor((distance - 3.25f) / 1.25f);
  if (distance < 3.25f) bin = -1;
  if (bin >= (int)a.bins) bin = a.bins - 1;
  float v = a.s1[j * a.C + c] + a.s2[i * a.C + c] + (bin >= 0 ? a.Wd[bin * a.C + c] : 0.f) + distance * a.Wraw[c];
  float p = (a.srcPair ? a.srcPair[t] : a.pair[t]) + v;
  if (a.outH) a.outH[t] = (half)p;
  else a.pair[t] = p;
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
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= (ulong)a.n * a.n * C4) return;
    uint ij = lf_udiv((uint)t, C4), c4 = (uint)t - ij * C4, i = lf_udiv(ij, a.n), j = ij - i * a.n;
    bool sameChain = a.r.asym[i] == a.r.asym[j];
    int rss = sameChain ? clamp(a.r.ri[i] - a.r.ri[j] + 33, 0, 65) : 66;
    int rts = sameChain && a.r.ri[i] == a.r.ri[j] ? clamp(a.r.ti[i] - a.r.ti[j] + 32, 0, 65) : 66;
    device const float4* W4 = (device const float4*)a.W;
    float4 v = ((device const float4*)a.bias)[c4] + W4[rss * C4 + c4] + W4[(67 + rts) * C4 + c4];
    if (a.ce) {
      int relE = a.r.ent[i] - a.r.ent[j];
      int rchain = relE != 0 ? 5 : clamp(a.r.sym[i] - a.r.sym[j] + 2, 0, 4), rent = clamp(relE + 1, 0, 2);
      device const float4* ce4 = (device const float4*)a.ce;
      v += ce4[rchain * C4 + c4] + ce4[(6 + rent) * C4 + c4];
    }
    if (a.bond && a.Wbond) v += a.bond[ij] * ((device const float4*)a.Wbond)[c4];
    ((device float4*)a.pair)[t] += v;
    return;
  }
  if (t >= (ulong)a.n * a.n * a.C) return;
  uint ij = lf_udiv((uint)t, a.C), c = (uint)t - ij * a.C, i = lf_udiv(ij, a.n), j = ij - i * a.n;
  bool sameChain = a.r.asym[i] == a.r.asym[j];
  int rss = sameChain ? clamp(a.r.ri[i] - a.r.ri[j] + 33, 0, 65) : 66;
  int rts = sameChain && a.r.ri[i] == a.r.ri[j] ? clamp(a.r.ti[i] - a.r.ti[j] + 32, 0, 65) : 66;
  float v = a.bias[c] + a.W[rss * a.C + c] + a.W[(67 + rts) * a.C + c];
  if (a.ce) {
    int relE = a.r.ent[i] - a.r.ent[j];
    int rchain = relE != 0 ? 5 : clamp(a.r.sym[i] - a.r.sym[j] + 2, 0, 4), rent = clamp(relE + 1, 0, 2);
    v += a.ce[rchain * a.C + c] + a.ce[(6 + rent) * a.C + c];
  }
  if (a.bond && a.Wbond) v += a.bond[ij] * a.Wbond[c];
  a.pair[t] += v;
}
kernel void af3_chai_chain_hit(LF_ARGS(ChaiChainHitArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.count) return;
  uint s = lf_udiv((uint)t, a.n), k = (uint)t - s * a.n;
  if (a.msaMask[t] != 0.f && a.rows[t] != 21) a.hit[s * a.chains + a.asym[k]] = 1;
}
kernel void af3_chai_msa_embed(LF_ARGS(ChaiMsaEmbedArgs)) {
  ulong t = LF_INDEX;
  if ((a.C & 3) == 0) {
    uint C4 = a.C >> 2;
    if (t >= a.count * C4) return;
    uint row = lf_udiv((uint)t, C4), c4 = (uint)t - row * C4, s = lf_udiv(row, a.n), token = row - s * a.n;
    bool paired = s > 0 && s < a.pairedRows && a.msaMask[row] != 0.f && a.chainHit[s * a.chains + a.asym[token]] != 0
               && a.chainHit[s * a.chains + a.asym[0]] != 0;
    float d = a.del[row];
    int code = a.rows[row];
    if (a.isLigand && a.isLigand[token]) code = s == 0 ? 20 : 31;      // (chai's query row: unknown, the rest: its mask class)
    device const float4* W4 = (device const float4*)a.W;
    float4 v = ((device const float4*)a.bias)[c4] + (paired ? W4[c4] : float4(0.f)) +
               W4[(1 + (s == 0 ? 4 : a.source ? a.source[s] : 2)) * C4 + c4] +
               atan(d / 3.f) * (2.f / M_PI_F) * W4[7 * C4 + c4] + clamp(d, 0.f, 1.f) * W4[8 * C4 + c4] +
               (code >= 0 && code < 32 ? W4[(9 + code) * C4 + c4] : float4(0.f));
    ((device float4*)a.msa)[t] = v + ((device const float4*)a.fromSingle)[token * C4 + c4];
    return;
  }
  if (t >= a.count * a.C) return;
  uint row = lf_udiv((uint)t, a.C), c = (uint)t - row * a.C, s = lf_udiv(row, a.n), token = row - s * a.n;
  bool paired = s > 0 && s < a.pairedRows && a.msaMask[row] != 0.f && a.chainHit[s * a.chains + a.asym[token]] != 0
             && a.chainHit[s * a.chains + a.asym[0]] != 0;
  float d = a.del[row];
  int code = a.rows[row];
  if (a.isLigand && a.isLigand[token]) code = s == 0 ? 20 : 31;      // (chai's query row: unknown, the rest: its mask class)
  float v = a.bias[c] + (paired ? a.W[c] : 0.f) + a.W[(1 + (s == 0 ? 4 : a.source ? a.source[s] : 2)) * a.C + c] +
            atan(d / 3.f) * (2.f / M_PI_F) * a.W[7 * a.C + c] + clamp(d, 0.f, 1.f) * a.W[8 * a.C + c] +
            (code >= 0 && code < 32 ? a.W[(9 + code) * a.C + c] : 0.f);
  a.msa[t] = v + a.fromSingle[token * a.C + c];
}
kernel void af3_group_major(LF_ARGS(GroupMajorArgs)) {
  ulong t = LF_INDEX;
  if ((a.K & 3) == 0) {
    uint K4 = a.K >> 2;
    const ulong total4 = (ulong)a.S * a.n * a.G * K4;
    if (t >= total4) return;
    uint k4, i, s, g;
    if (total4 <= 0xffffffffull) {
      uint r = lf_udiv((uint)t, K4); k4 = (uint)t - r * K4; uint r2 = lf_udiv(r, a.n); i = r - r2 * a.n; g = lf_udiv(r2, a.S); s = r2 - g * a.S;
    } else { k4 = (uint)(t % K4); ulong r = t / K4; i = (uint)(r % a.n); r /= a.n; s = (uint)(r % a.S); g = (uint)(r / a.S); }
    ulong row = (ulong)s * a.n + i;
    ((device float4*)a.out)[t] = ((device const float4*)a.x)[(row * a.G + g) * K4 + k4] * (a.mask ? a.mask[row] : 1.f);
    return;
  }
  const ulong total = (ulong)a.S * a.n * a.G * a.K;
  if (t >= total) return;
  uint k, i, s, g;   // (32-bit division where the index fits: the 64-bit one is slow)
  if (total <= 0xffffffffull) {
    uint r = lf_udiv((uint)t, a.K); k = (uint)t - r * a.K; uint r2 = lf_udiv(r, a.n); i = r - r2 * a.n; g = lf_udiv(r2, a.S); s = r2 - g * a.S;
  } else { k = (uint)(t % a.K); ulong r = t / a.K; i = (uint)(r % a.n); r /= a.n; s = (uint)(r % a.S); g = (uint)(r / a.S); }
  ulong row = (ulong)s * a.n + i;
  a.out[t] = a.x[(row * a.G + g) * a.K + k] * (a.mask ? a.mask[row] : 1.f);
}
kernel void af3_grouped_permute(LF_ARGS(GroupedPermuteArgs)) {
  ulong t = LF_INDEX;
  if ((a.K & 3) == 0) {
    uint K4 = a.K >> 2;
    ulong per4 = (ulong)a.G * a.K * K4;
    const ulong total4 = (ulong)a.bi * a.n * per4;
    if (t >= total4) return;
    uint l4, k, g, j, i;
    if (total4 <= 0xffffffffull) {
      uint r = lf_udiv((uint)t, K4); l4 = (uint)t - r * K4; uint r2 = lf_udiv(r, a.K); k = r - r2 * a.K;
      uint r3 = lf_udiv(r2, a.G); g = r2 - r3 * a.G; i = lf_udiv(r3, a.n); j = r3 - i * a.n;
    } else {
      l4 = (uint)(t % K4); ulong r = t / K4; k = (uint)(r % a.K); r /= a.K; g = (uint)(r % a.G); r /= a.G;
      j = (uint)(r % a.n); i = (uint)(r / a.n);
    }
    ((device float4*)a.out)[t] = ((device const float4*)a.P)[(ulong)g * ((ulong)a.bi * a.K * a.n * K4) + (((ulong)i * a.K + k) * a.n + j) * K4 + l4];
    return;
  }
  ulong per = (ulong)a.G * a.K * a.K;
  const ulong total = (ulong)a.bi * a.n * per;
  if (t >= total) return;
  uint l, k, g, j, i;   // (32-bit division where the index fits: the 64-bit one is slow)
  if (total <= 0xffffffffull) {
    uint r = lf_udiv((uint)t, a.K); l = (uint)t - r * a.K; uint r2 = lf_udiv(r, a.K); k = r - r2 * a.K;
    uint r3 = lf_udiv(r2, a.G); g = r2 - r3 * a.G; i = lf_udiv(r3, a.n); j = r3 - i * a.n;
  } else {
    l = (uint)(t % a.K); ulong r = t / a.K; k = (uint)(r % a.K); r /= a.K; g = (uint)(r % a.G); r /= a.G;
    j = (uint)(r % a.n); i = (uint)(r / a.n);
  }
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
  if (a.dx && a.isDna && a.isDna[token] != 0 && aa == 30) v += a.dx[c];
  for (int k = 0; k < 31; ++k) v += a.profile[token * 31 + k] * a.Wp[k * a.C + c];
  v += a.delMean[token] * a.Wp[31 * a.C + c];
  a.out[t] = v;
}
kernel void af3_chai_struct_pair(LF_ARGS(ChaiStructPairArgs)) {
  uint width = a.Czt + a.Cz;
  ulong t = LF_INDEX;
  if ((a.Czt & 3) == 0 && (a.Cz & 3) == 0) {
    uint width4 = width >> 2, Czt4 = a.Czt >> 2, Cz4 = a.Cz >> 2;
    if (t >= a.rows * width4) return;
    ulong row = t / width4; uint c4 = (uint)(t - row * width4);
    ulong ij = a.p0 + row;
    if (c4 < Czt4) { ((device float4*)a.out)[t] = ((device const float4*)a.trunkPair)[ij * Czt4 + c4]; return; }
    c4 -= Czt4;
    uint i = (uint)(ij / a.n), j = (uint)(ij - (ulong)i * a.n);
    bool sameChain = a.asym[i] == a.asym[j];
    int rss = sameChain ? clamp(a.ri[i] - a.ri[j] + 33, 0, 65) : 66;
    int rts = sameChain && a.ri[i] == a.ri[j] ? clamp(a.ti[i] - a.ti[j] + 32, 0, 65) : 66;
    int relEntity = a.entityRank[i] - a.entityRank[j];
    int rchain = relEntity != 0 ? 5 : clamp(a.symRank[i] - a.symRank[j] + 2, 0, 4);
    int rent = clamp(relEntity + 1, 0, 2);
    device const float4* Wp4 = (device const float4*)a.Wp;
    float4 v = ((device const float4*)a.bias)[c4] + Wp4[5 * Cz4 + c4] + Wp4[155 * Cz4 + c4] + Wp4[162 * Cz4 + c4] +
               Wp4[(6 + rchain) * Cz4 + c4] + Wp4[(12 + rent) * Cz4 + c4] + Wp4[(15 + rss) * Cz4 + c4] + Wp4[(82 + rts) * Cz4 + c4];
    if (a.bonds) v += a.bonds[ij] * ((device const float4*)a.Wb)[c4];
    ((device float4*)a.out)[t] = v;
    return;
  }
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
kernel void af3_exp_heun1(LF_ARGS(ExpHeun1Args)) {
  ulong i = LF_INDEX;
  if (i >= a.n) return;
  a.keep[i] = a.d1[i]; a.x[i] = a.noisy[i] + a.c * (a.noisy[i] - a.d1[i]);
}
kernel void af3_exp_heun2(LF_ARGS(ExpHeun2Args)) {
  ulong i = LF_INDEX;
  if (i < a.n) a.x[i] = a.noisy[i] + a.c * (a.noisy[i] - 0.5f * (a.d1[i] + a.d2[i]));
}
kernel void af3_plddt37(LF_ARGS(Plddt37Args)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.n * a.dense * a.bins) return;
  uint b = (uint)(t % a.bins); ulong slot = t / a.bins; ulong token = slot / a.dense;
  a.out[t] = a.p37[(token * 37 + a.idx[slot]) * a.bins + b];
}
