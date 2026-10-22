// BioEmu2's kernels (cuda/bioemu2/src/*.cuh is the reading: each kernel here is the CUDA kernel of the same name, in
// float32). metal/build.sh prepends core/args.h, this port's kernels.h and core/common.metal.

constant int BE2_WQ = 32, BE2_WK = 128, BE2_CTX = 48;   // atom query / key windows; keys centred on queries
constant float BE2_NEG_INF = -1e6f;                       // the vendor's -INF (1e6): a masked logit, not -inf

// a simdgroup a row, eight rows a threadgroup of 256 (metal/core's LayerNorm's dispatch)
#define BE2_ROW ((ulong)(_tg.y * _ng.x + _tg.x) * 8 + (_tid >> 5))
#define BE2_LANE (_tid & 31)
// an element index split by a runtime width without a 64-bit division (Apple's GPUs have no integer divider: `ulong / d`
// is a long software routine - tokens_to_atoms took 0.94 ms for 2.2 M elements). The host keeps these tensors under
// 2^32 elements (be2Run1d)
#define BE2_SPLIT(t, d, q, r) uint q = lf_udiv((uint)(t), (d)), r = (uint)(t) - q * (d)

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
// AttentionPairBiasSlidingWindow: a threadgroup per (window, head) of 128 threads, FOUR lanes a query, each over a
// quarter of the 128 keys (its own online softmax), merged by shuffles - cuda/bioemu2's windowAttention4K, its float
// path's too. (A thread a query was latency-bound: 360 threadgroups of 32 at 68 tokens, each thread a serial chain over
// every key - 0.74 ms a call, ~130 GFLOP/s.) Head width 32. Several samples stacked on the rows, N atoms each
kernel void be2_window_attention(constant Be2WindowAttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  constexpr int HD = 32, SPLIT = 4, KS = BE2_WK / SPLIT;
  const uint wg = tg.x, h = tg.y, D = a.H * HD, NW = a.N / BE2_WQ, smp = wg / NW, w = wg % NW;
  const uint i = tid / SPLIT, part = tid % SPLIT;
  const ulong base = (ulong)smp * a.N;
  // (the keys and values are 32 KB, Apple's whole threadgroup allowance: the key mask is read per key from the device)
  threadgroup float sk[BE2_WK][HD], sv[BE2_WK][HD];
  for (uint e = tid; e < BE2_WK * HD; e += BE2_WQ * SPLIT) {
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
  for (int jj = 0; jj < KS; ++jj) {
    const int j = jj * SPLIT + (int)part;          // (interleaved: the four lanes read neighbouring keys)
    float l = 0;
    for (int d = 0; d < HD; ++d) l += q[d] * sk[j][d];
    const int ka = (int)(w * BE2_WQ) - BE2_CTX + j;
    l = ka >= 0 && ka < (int)a.N && a.pad[ka] ? l + b[j] : BE2_NEG_INF;
    float nm = max(mx, l), f = exp(mx - nm), e = exp(l - nm);
    sum = sum * f + e;
    for (int d = 0; d < HD; ++d) acc[d] = acc[d] * f + e * sv[j][d];
    mx = nm;
  }
  // the four parts' online softmaxes merged (lanes 4i .. 4i + 3 of a simdgroup)
  for (ushort o = 1; o < SPLIT; o <<= 1) {
    float om = simd_shuffle_xor(mx, o), os = simd_shuffle_xor(sum, o);
    float nm = max(mx, om), f = exp(mx - nm), fo = exp(om - nm);
    sum = sum * f + os * fo;
    for (int d = 0; d < HD; ++d) { float oa = simd_shuffle_xor(acc[d], o); acc[d] = acc[d] * f + oa * fo; }
    mx = nm;
  }
  if (part == 0) for (int d = 0; d < HD; ++d) a.out[qa * D + h * HD + d] = acc[d] / sum;
}
// the fast arm's sliding-window attention: qkvg [rows][4D] in half, K/V staged in half (16 KB, half the threadgroup limit)
// with the key mask staged in sm[128], and the output gated by sigmoid(g) into half
kernel void be2_window_attention_h(constant Be2WindowAttnHArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  constexpr int HD = 32, SPLIT = 4, KS = BE2_WK / SPLIT;
  const uint wg = tg.x, h = tg.y, D = a.H * HD, NW = a.N / BE2_WQ, smp = wg / NW, w = wg % NW;
  const uint i = tid / SPLIT, part = tid % SPLIT;
  const ulong base = (ulong)smp * a.N;
  threadgroup half sk[BE2_WK][HD], sv[BE2_WK][HD];
  threadgroup uchar sm[BE2_WK];
  for (uint e4 = tid; e4 < BE2_WK * (HD / 4); e4 += BE2_WQ * SPLIT) {
    int j = e4 >> 3, d = (e4 & 7) << 2, ka = (int)(w * BE2_WQ) - BE2_CTX + j;
    bool ok = ka >= 0 && ka < (int)a.N;
    const ulong off = (base + ka) * 4 * D + h * HD + d;
    *(threadgroup half4*)(sk[j] + d) = ok ? *(device const half4*)(a.qkvg + off + D) : half4(0.h);
    *(threadgroup half4*)(sv[j] + d) = ok ? *(device const half4*)(a.qkvg + off + 2 * D) : half4(0.h);
  }
  if (tid < (uint)BE2_WK) {
    int ka = (int)(w * BE2_WQ) - BE2_CTX + (int)tid;
    sm[tid] = ka >= 0 && ka < (int)a.N && a.pad[ka] ? 1 : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong qa = base + w * BE2_WQ + i;
  float4 q[8], acc[8];
  const float scale = rsqrt((float)HD);
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < 8; ++k) {
    const int d = k << 2;
    q[k] = (float4(*(device const half4*)(a.qkvg + qa * 4 * D + h * HD + d)) + *(device const float4*)(a.qBias + h * HD + d)) * scale;
    acc[k] = float4(0.f);
  }
  device const float* b = a.bias + (((ulong)w * a.H + h) * BE2_WQ + i) * BE2_WK;
  float mx = -INFINITY, sum = 0;
  for (int jj = 0; jj < KS; ++jj) {
    const int j = jj * SPLIT + (int)part;
    float l = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 8; ++k) l += dot(q[k], float4(*(threadgroup const half4*)(sk[j] + (k << 2))));
    l = sm[j] ? l + b[j] : BE2_NEG_INF;
    float nm = max(mx, l), f = exp(mx - nm), e = exp(l - nm);
    sum = sum * f + e;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 8; ++k) acc[k] = acc[k] * f + e * float4(*(threadgroup const half4*)(sv[j] + (k << 2)));
    mx = nm;
  }
  for (ushort o = 1; o < SPLIT; o <<= 1) {
    float om = simd_shuffle_xor(mx, o), os = simd_shuffle_xor(sum, o);
    float nm = max(mx, om), f = exp(mx - nm), fo = exp(om - nm);
    sum = sum * f + os * fo;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 8; ++k) { const float4 oa = simd_shuffle_xor(acc[k], o); acc[k] = acc[k] * f + oa * fo; }
    mx = nm;
  }
  if (part == 0) {
    const float inv = 1.f / sum;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 8; ++k) {
      const int d = k << 2;
      const float4 gv = float4(*(device const half4*)(a.qkvg + qa * 4 * D + 3 * D + h * HD + d));
      const float4 av = acc[k] * inv / (1.f + exp(-gv));
      *(device half4*)(a.out + qa * D + h * HD + d) = half4(av);
    }
  }
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
  const uint C4 = a.C >> 2;
  if (t >= a.rows * C4) return;
  BE2_SPLIT(t, C4, r, c4);
  uint g = a.map ? (uint)a.map[r] : r;
  ((device float4*)a.x)[t] += ((device const float4*)(a.gate + (ulong)g * a.ldg))[c4] * ((device const float4*)a.h)[t];
}
kernel void be2_ada_ln(LF_ARGS(Be2AdaLnArgs)) {
  const ulong row = BE2_ROW; const uint lane = BE2_LANE;
  if (row >= a.rows) return;
  const ulong cr = a.map ? (ulong)a.map[row] : row;
  device float* xr = a.x + row * a.ldx;
  device const float* al = a.alpha + cr * a.ldc;
  device const float* be = a.beta + cr * a.ldc;
  if (a.yh && (a.C == 128 || a.C == 768)) {
    device float4* x4 = (device float4*)xr;
    device const float4* al4 = (device const float4*)al;
    device const float4* be4 = (device const float4*)be;
    device half4* yh4 = (device half4*)(a.yh + row * a.C);
    if (a.C == 128) {
      float4 x = x4[lane];
      if (a.resH) {
        x += ((device const float4*)(a.resGate + cr * a.ldrg))[lane] * ((device const float4*)(a.resH + row * 128))[lane];
        x4[lane] = x;
      }
      const float mean = simd_sum(x.x + x.y + x.z + x.w) * (1.f / 128.f);
      const float4 d = x - mean;
      const float inv = rsqrt(simd_sum(dot(d, d)) * (1.f / 128.f) + 1e-5f);
      yh4[lane] = half4(a.scale * (d * inv * (1.f + al4[lane]) + be4[lane]));
    } else {
      float4 xv[6]; float s = 0;
      if (a.resH) {
        device const float4* rh4 = (device const float4*)(a.resH + row * 768);
        device const float4* rg4 = (device const float4*)(a.resGate + cr * a.ldrg);
        _Pragma("clang loop unroll(full)")
        for (int k = 0; k < 6; ++k) {
          const uint idx = lane + 32 * k;
          const float4 x = x4[idx] + rg4[idx] * rh4[idx];
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
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) {
        const uint idx = lane + 32 * k;
        yh4[idx] = half4(a.scale * ((xv[k] - mean) * inv * (1.f + al4[idx]) + be4[idx]));
      }
    }
    return;
  }
  float s = 0;
  if (a.resH) {
    device const float* rh = a.resH + row * a.C;
    device const float* rg = a.resGate + cr * a.ldrg;
    for (uint c = lane; c < a.C; c += 32) { const float x = xr[c] + rg[c] * rh[c]; xr[c] = x; s += x; }
  } else {
    for (uint c = lane; c < a.C; c += 32) s += xr[c];
  }
  const float mean = simd_sum(s) / a.C;
  float v = 0;
  for (uint c = lane; c < a.C; c += 32) { const float d = xr[c] - mean; v += d * d; }
  const float inv = rsqrt(simd_sum(v) / a.C + 1e-5f);
  for (uint c = lane; c < a.C; c += 32) {
    const float val = a.scale * ((xr[c] - mean) * inv * (1.f + al[c]) + be[c]);
    if (a.yh) a.yh[row * a.C + c] = (half)val; else a.y[row * a.C + c] = val;
  }
}
kernel void be2_ada_ln_q2s(LF_ARGS(Be2AdaLnQ2SArgs)) {
  const ulong row = BE2_ROW; const uint lane = BE2_LANE;
  if (row >= a.rows) return;
  BE2_SPLIT(row, a.T, s, il);
  device const float* qr = a.q + ((ulong)s * a.N + a.centre[il]) * a.Ca;
  device float* ar = a.a + row * a.Dt;
  const uint C = a.Ca + a.Dt;
  const ulong cr = a.map ? (ulong)a.map[row] : row;
  device const float* al = a.alpha + cr * a.ldc;
  device const float* be = a.beta + cr * a.ldc;
  device half* yr = a.yh + row * C;
  if (a.Ca == 128 && a.Dt == 768) {
    float4 xv[7];
    xv[0] = ((device const float4*)qr)[lane];
    float sum = xv[0].x + xv[0].y + xv[0].z + xv[0].w;
    device float4* ar4 = (device float4*)ar;
    if (a.resH) {
      device const float4* rh4 = (device const float4*)(a.resH + row * 768);
      device const float4* rg4 = (device const float4*)(a.resGate + cr * a.ldrg);
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) {
        const uint idx = lane + 32 * k;
        const float4 x = ar4[idx] + rg4[idx] * rh4[idx];
        ar4[idx] = x; xv[k + 1] = x; sum += x.x + x.y + x.z + x.w;
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) {
        const float4 x = ar4[lane + 32 * k];
        xv[k + 1] = x; sum += x.x + x.y + x.z + x.w;
      }
    }
    const float mean = simd_sum(sum) * (1.f / 896.f);
    float v = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 7; ++k) { const float4 d = xv[k] - mean; v += dot(d, d); }
    const float inv = rsqrt(simd_sum(v) * (1.f / 896.f) + 1e-5f);
    device const float4* al4 = (device const float4*)al;
    device const float4* be4 = (device const float4*)be;
    device half4* yr4 = (device half4*)yr;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 7; ++k) {
      const uint idx = lane + 32 * k;
      yr4[idx] = half4((xv[k] - mean) * inv * (1.f + al4[idx]) + be4[idx]);
    }
    return;
  }
  if (a.resH) {
    device const float* rh = a.resH + row * a.Dt;
    device const float* rg = a.resGate + cr * a.ldrg;
    for (uint c = lane; c < a.Dt; c += 32) ar[c] += rg[c] * rh[c];
  }
  float sum = 0;
  for (uint c = lane; c < C; c += 32) sum += c < a.Ca ? qr[c] : ar[c - a.Ca];
  const float mean = simd_sum(sum) / C;
  float v = 0;
  for (uint c = lane; c < C; c += 32) {
    const float d = (c < a.Ca ? qr[c] : ar[c - a.Ca]) - mean;
    v += d * d;
  }
  const float inv = rsqrt(simd_sum(v) / C + 1e-5f);
  for (uint c = lane; c < C; c += 32) {
    const float x = c < a.Ca ? qr[c] : ar[c - a.Ca];
    yr[c] = (half)((x - mean) * inv * (1.f + al[c]) + be[c]);
  }
}
kernel void be2_ada_ln_s2q(LF_ARGS(Be2AdaLnS2QArgs)) {
  const ulong row = BE2_ROW; const uint lane = BE2_LANE;
  if (row >= a.rows) return;
  BE2_SPLIT(row, a.N, s, alIdx);
  device const float* ar = a.pad[alIdx] ? a.a + ((ulong)s * a.T + a.atomTok[alIdx]) * a.Dt : nullptr;
  device const float* qr = a.q + row * a.Ca;
  const uint C = a.Dt + a.Ca;
  const ulong cr = a.map ? (ulong)a.map[row] : row;
  device const float* al = a.alpha + cr * a.ldc;
  device const float* be = a.beta + cr * a.ldc;
  device half* yr = a.yh + row * C;
  if (a.Dt == 768 && a.Ca == 128) {
    float4 xv[7]; float sum = 0;
    if (ar) {
      device const float4* ar4 = (device const float4*)ar;
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) {
        const float4 x = ar4[lane + 32 * k];
        xv[k] = x; sum += x.x + x.y + x.z + x.w;
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 6; ++k) xv[k] = float4(0.f);
    }
    xv[6] = ((device const float4*)qr)[lane];
    sum += xv[6].x + xv[6].y + xv[6].z + xv[6].w;
    const float mean = simd_sum(sum) * (1.f / 896.f);
    float v = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 7; ++k) { const float4 d = xv[k] - mean; v += dot(d, d); }
    const float inv = rsqrt(simd_sum(v) * (1.f / 896.f) + 1e-5f);
    device const float4* al4 = (device const float4*)al;
    device const float4* be4 = (device const float4*)be;
    device half4* yr4 = (device half4*)yr;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 7; ++k) {
      const uint idx = lane + 32 * k;
      yr4[idx] = half4((xv[k] - mean) * inv * (1.f + al4[idx]) + be4[idx]);
    }
    return;
  }
  float sum = 0;
  for (uint c = lane; c < C; c += 32) sum += c < a.Dt ? (ar ? ar[c] : 0.f) : qr[c - a.Dt];
  const float mean = simd_sum(sum) / C;
  float v = 0;
  for (uint c = lane; c < C; c += 32) {
    const float d = (c < a.Dt ? (ar ? ar[c] : 0.f) : qr[c - a.Dt]) - mean;
    v += d * d;
  }
  const float inv = rsqrt(simd_sum(v) / C + 1e-5f);
  for (uint c = lane; c < C; c += 32) {
    const float x = c < a.Dt ? (ar ? ar[c] : 0.f) : qr[c - a.Dt];
    yr[c] = (half)((x - mean) * inv * (1.f + al[c]) + be[c]);
  }
}
// the coupling split's statistics and normalised halves (Be2CsTokArgs, Be2CsAtomArgs): webgpu/bioemu2's cs-tstats /
// cs-xa and cs-astats / cs-xq, the parallel-variance form (never E[x^2] - mu^2)
kernel void be2_cs_tok(LF_ARGS(Be2CsTokArgs)) {
  const ulong row = BE2_ROW; const uint lane = BE2_LANE;
  if (row >= a.rows) return;
  device const float* xr = a.a + row * a.C;
  const uint t = (uint)row - lf_udiv((uint)row, a.T) * a.T;
  device const float* al = a.cqc + (ulong)t * a.ldc;
  if (a.C == 768) {
    device const float4* xr4 = (device const float4*)xr;
    float4 xv[6]; float s = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 6; ++k) {
      const float4 x = xr4[lane + 32 * k];
      xv[k] = x; s += x.x + x.y + x.z + x.w;
    }
    const float m = simd_sum(s) * (1.f / 768.f);
    float v = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 6; ++k) { const float4 d = xv[k] - m; v += dot(d, d); }
    v = simd_sum(v);
    const float sa = sqrt(v * (1.f / 768.f) + 1e-5f), invSa = 1.f / sa;
    if (lane == 0) { a.tstats[row * 3] = m; a.tstats[row * 3 + 1] = v; a.tstats[row * 3 + 2] = sa; }
    device const float4* al4 = (device const float4*)al;
    device half4* xa4 = (device half4*)(a.xa + row * 768);
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 6; ++k) {
      const uint idx = lane + 32 * k;
      xa4[idx] = half4((1.f + al4[idx]) * (xv[k] - m) * invSa);
    }
    return;
  }
  float s = 0;
  for (uint c = lane; c < a.C; c += 32) s += xr[c];
  const float m = simd_sum(s) / a.C;
  float v = 0; for (uint c = lane; c < a.C; c += 32) { const float d = xr[c] - m; v += d * d; }
  v = simd_sum(v);
  const float sa = sqrt(v / a.C + 1e-5f);
  if (lane == 0) { a.tstats[row * 3] = m; a.tstats[row * 3 + 1] = v; a.tstats[row * 3 + 2] = sa; }
  for (uint c = lane; c < a.C; c += 32) a.xa[row * a.C + c] = (half)((1.f + al[c]) * (xr[c] - m) / sa);
}
kernel void be2_cs_atom(LF_ARGS(Be2CsAtomArgs)) {
  const ulong row = BE2_ROW; const uint lane = BE2_LANE;
  if (row >= a.rows) return;
  device const float* xr = a.q + row * a.Ca;
  float4 x4 = a.Ca == 128 ? ((device const float4*)xr)[lane] : float4(0.f);
  float s = 0;
  if (a.Ca == 128) s = x4.x + x4.y + x4.z + x4.w;
  else for (uint c = lane; c < a.Ca; c += 32) s += xr[c];
  const float mq = simd_sum(s) / a.Ca;
  float v = 0;
  if (a.Ca == 128) { const float4 d = x4 - mq; v = dot(d, d); }
  else for (uint c = lane; c < a.Ca; c += 32) { const float d = xr[c] - mq; v += d * d; }
  const float ssq = simd_sum(v);
  const uint smp = lf_udiv((uint)row, a.N), al = (uint)row - smp * a.N;
  const int st = a.pad[al] ? (int)(smp * a.T) + a.tokIdx[al] : -1;
  float ma = 0, ssa = 0, sa = 0;
  if (st >= 0) { ma = a.tstats[st * 3]; ssa = a.tstats[st * 3 + 1]; sa = a.tstats[st * 3 + 2]; }
  const float W = (float)(a.Dt + a.Ca);
  const float mu = ((float)a.Dt * ma + (float)a.Ca * mq) / W, da = ma - mu, dq = mq - mu;
  const float var = (ssa + ssq + (float)a.Dt * da * da + (float)a.Ca * dq * dq) / W;
  const float sq = sqrt(ssq / a.Ca + 1e-5f);
  const int t = a.atomTok[row];
  if (lane == 0) {
    device float* o = a.stats + row * 8;
    o[0] = rsqrt(var + 1e-5f); o[1] = da; o[2] = dq; o[3] = sq; o[4] = mq; o[5] = sa; o[6] = (float)st; o[7] = (float)t;
  }
  device const float* aq = a.cqc + (ulong)t * a.ldc + a.Dt;
  // (two halves, hi and the remainder: Xq in ONE half cost the split its precision - metal/bioemu2's denoise.mm)
  if (a.Ca == 128) {
    const float4 val = (1.f + ((device const float4*)aq)[lane]) * (x4 - mq) / sq;
    const half4 h = half4(val);
    device half4* xq4 = (device half4*)(a.xq + row * 256);
    xq4[lane] = h;
    xq4[32 + lane] = half4(val - float4(h));
    return;
  }
  for (uint c = lane; c < a.Ca; c += 32) {
    const float val = (1.f + aq[c]) * (xr[c] - mq) / sq;
    const half h = (half)val;
    a.xq[row * 2 * a.Ca + c] = h;
    a.xq[row * 2 * a.Ca + a.Ca + c] = (half)(val - (float)h);
  }
}
kernel void be2_cs_gamma(LF_ARGS(Be2CsGammaArgs)) {
  ulong i = LF_INDEX;
  if (i >= a.n) return;
  const uint Dc = a.Dt + a.Ca;
  BE2_SPLIT(i, Dc, lr, c);
  BE2_SPLIT(lr, a.R, j, t);
  // (R = Qq - Qa / 6: the token half's gamma scaled by da / dq = -Ca / Dt (-1/6), since 768 da + 128 dq = 0)
  device const float* row = a.cqc + ((ulong)j * a.R + t) * a.ldc;
  float g = 1.f + row[c];
  if (c < a.Dt) g *= -(float)a.Ca / (float)a.Dt;
  const float b = row[Dc + c];
  const half gh = (half)g, bh = (half)b;
  const ulong span = (ulong)2 * a.R * Dc;
  device half* dst = a.gR + (ulong)j * 2 * span;
  const ulong ir = (ulong)(2 * t) * Dc + c, ib = (ulong)(2 * t + 1) * Dc + c;
  dst[ir] = gh; dst[span + ir] = (half)(g - (float)gh);
  dst[ib] = bh; dst[span + ib] = (half)(b - (float)bh);
}
kernel void be2_cs_wq2(LF_ARGS(Be2CsWqArgs)) {
  ulong t = LF_INDEX;
  const uint W = 2 * a.I;
  if (t >= (ulong)2 * a.K * W) return;
  BE2_SPLIT(t, W, k, col);
  const uint kk = a.off + (k % a.K), q = col & 3, j = 2 * (col >> 2) + (q & 1);
  a.out[t] = q < 2 ? a.w1[(ulong)j * a.C + kk] : (half)((float)a.w2[(ulong)j * a.C + kk] * a.invS);
}
kernel void be2_cond_in(LF_ARGS(Be2CondInArgs)) {
  const ulong row = BE2_ROW; const uint lane = BE2_LANE;
  if (row >= a.rows) return;
  const uint j = lf_udiv((uint)row, a.Rc), r = (uint)row - j * a.Rc;
  device const float* x = a.c + (ulong)r * a.C;
  device const float* w = a.w + (ulong)(a.l0 + j) * a.C;
  device const float* b = a.b ? a.b + (ulong)(a.l0 + j) * a.C : nullptr;
  if (a.C == 768) {
    device const float4* x4 = (device const float4*)x;
    float4 xv[6]; float s = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 6; ++k) {
      const float4 val = x4[lane + 32 * k];
      xv[k] = val; s += val.x + val.y + val.z + val.w;
    }
    const float mean = simd_sum(s) * (1.f / 768.f);
    float v = 0;
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 6; ++k) { const float4 d = xv[k] - mean; v += dot(d, d); }
    const float inv = rsqrt(simd_sum(v) * (1.f / 768.f) + 1e-5f);
    device const float4* w4 = (device const float4*)w;
    device const float4* b4 = (device const float4*)b;
    device half4* out4 = (device half4*)(a.out + row * 768);
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 6; ++k) {
      const uint idx = lane + 32 * k;
      float4 n = (xv[k] - mean) * inv * w4[idx];
      if (b4) n += b4[idx];
      out4[idx] = half4(n / (1.f + exp(-n)));
    }
    return;
  }
  float s = 0;
  for (uint c = lane; c < a.C; c += 32) s += x[c];
  const float mean = simd_sum(s) / a.C;
  float v = 0; for (uint c = lane; c < a.C; c += 32) { const float d = x[c] - mean; v += d * d; }
  const float inv = rsqrt(simd_sum(v) / a.C + 1e-5f);
  for (uint c = lane; c < a.C; c += 32) {      // (lf_layernorm's order, then be2_unary's silu)
    float n = (x[c] - mean) * inv;
    n *= w[c];
    if (b) n += b[c];
    a.out[row * a.C + c] = (half)lf_silu(n);
  }
}
kernel void be2_batch_bias(LF_ARGS(Be2BatchBiasArgs)) {
  ulong t = LF_INDEX;
  const uint O4 = a.O >> 2;
  if (t >= (a.n >> 2)) return;
  BE2_SPLIT(t, O4, row, o4);
  const uint j = lf_udiv(row, a.Rc);
  ((device float4*)a.y)[t] += ((device const float4*)a.bias)[(ulong)(a.l0 + j) * O4 + o4];
}
kernel void be2_swiglu_weight(LF_ARGS(Be2SwigluWArgs)) {
  ulong t = LF_INDEX;
  const uint W4 = a.I >> 1;
  if (t >= (ulong)a.C * W4) return;
  BE2_SPLIT(t, W4, k, col4);
  const uint m = col4 >> 2, q = col4 & 3, j = 8 * m + ((q & 1) << 2);
  if (q < 2) {
    ((device half4*)a.out)[t] = half4(a.w1[(ulong)j * a.C + k], a.w1[(ulong)(j + 1) * a.C + k],
                                      a.w1[(ulong)(j + 2) * a.C + k], a.w1[(ulong)(j + 3) * a.C + k]);
  } else {
    ((device half4*)a.out)[t] = half4((float)a.w2[(ulong)j * a.C + k] * a.invS, (float)a.w2[(ulong)(j + 1) * a.C + k] * a.invS,
                                      (float)a.w2[(ulong)(j + 2) * a.C + k] * a.invS, (float)a.w2[(ulong)(j + 3) * a.C + k] * a.invS);
  }
}
// a token tensor's rows broadcast to its atoms (zero on padding atoms), samples stacked
kernel void be2_tokens_to_atoms(LF_ARGS(Be2TokensToAtomsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.samples * a.N * a.C) return;
  BE2_SPLIT(t, a.C, at, c);
  BE2_SPLIT(at, (uint)a.N, s, al);
  float v = a.pad[al] ? a.tok[((ulong)s * a.T + a.atomTok[al]) * a.C + c] : 0.f;
  a.out[(ulong)at * a.ldo + c] = a.beta == 0.f ? v : a.beta * a.out[(ulong)at * a.ldo + c] + v;
}
// each token's centre atom (every token is real here: no token padding)
kernel void be2_centre_atoms(LF_ARGS(Be2CentreAtomsArgs)) {
  ulong t = LF_INDEX;
  if (t >= (ulong)a.samples * a.T * a.C) return;
  BE2_SPLIT(t, a.C, i, c);
  BE2_SPLIT(i, a.T, s, il);
  a.out[(ulong)i * a.ldo + c] = a.atom[((ulong)s * a.N + a.centre[il]) * a.C + c];
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
  BE2_SPLIT(t, a.Ch, r, d);
  BE2_SPLIT(r, a.H, r2, h);
  BE2_SPLIT(r2, a.T, s, i);
  device const float* w = a.w + ((ulong)h * a.T + i) * a.T;
  const uint hd = h * a.Ch + d;
  device const float* v = a.v + (ulong)s * a.T * a.ldv + hd;
  float acc = 0;
  for (uint j = 0; j < a.T; ++j) acc += w[j] * v[(ulong)j * a.ldv];
  a.o[t] = acc * lf_sigmoid(a.g[((ulong)s * a.T + i) * a.ldv + hd]);
}
// OuterProductMeanVanilla's pack: proj [R = S T][2 Co] -> a, b [R][Co] (the first half and the second)
kernel void be2_opm_pack(LF_ARGS(Be2OpmPackArgs)) {
  ulong t = LF_INDEX;
  const uint W = 2 * a.Co;
  if (t >= a.R * W) return;
  BE2_SPLIT(t, W, r, c);
  if (c < a.Co) a.a[(ulong)r * a.Co + c] = a.proj[t]; else a.b[(ulong)r * a.Co + c - a.Co] = a.proj[t];
}
// x [(i, c)][(j, d)] for a block of bi rows of i -> y [(i, j)][(c, d)] / n (the mean over VALID rows)
kernel void be2_opm_permute(LF_ARGS(Be2OpmPermuteArgs)) {
  ulong t = LF_INDEX;
  const uint C4 = a.Co >> 2, CC4 = a.Co * C4;
  if (t >= (ulong)a.bi * a.T * CC4) return;
  BE2_SPLIT(t, CC4, ij, cd4);
  BE2_SPLIT(ij, a.T, i, j);
  BE2_SPLIT(cd4, C4, c, d4);
  ((device float4*)a.y)[t] = *(device const float4*)(a.x + ((ulong)i * a.Co + c) * ((ulong)a.T * a.Co) + (ulong)j * a.Co + (d4 << 2)) * a.inv;
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
  if (a.outh) a.outh[(ulong)h * a.T * a.T + ij] = (half)(a.proj[t] * a.mul);
  else a.out[(ulong)h * a.T * a.T + ij] = a.proj[t];
}
kernel void be2_trigate_weight(LF_ARGS(Be2TriGateWArgs)) {
  ulong t = LF_INDEX;
  const uint W4 = a.C;
  if (t >= (ulong)a.rows * W4) return;
  BE2_SPLIT(t, W4, r, col4);
  const uint col = col4 << 2;
  const uint kind = a.quarters ? (col % 128) / 32 : (col % 32) / 8;
  const uint c = a.quarters ? 32 * (col / 128) + col % 32 : 8 * (col / 32) + col % 8;
  const uint k = kind == 1 ? 2 : kind == 2 ? 1 : kind;      // (blocks of 8 or quarters of 32: pa ga pb gb; a and b interleaved by channel)
  device const half* src = (k < 2 ? a.proj : a.gate) + (ulong)r * 2 * a.C + 2 * c + (k & 1);
  ((device half4*)a.out)[t] = half4(src[0], src[2], src[4], src[6]);
}
kernel void be2_interleave8(LF_ARGS(Be2Interleave8Args)) {
  ulong t = LF_INDEX;
  const uint W4 = a.I >> 1;
  if (t >= (ulong)a.rows * W4) return;
  BE2_SPLIT(t, W4, r, c4);
  const uint m = c4 >> 2, q = c4 & 3, off = 8 * m + ((q & 1) << 2);
  if (q < 2) {
    ((device half4*)a.out)[t] = *(device const half4*)(a.a + (ulong)r * a.lda + off);
  } else {
    const float4 v = float4(*(device const half4*)(a.b + (ulong)r * a.lda + off)) * a.invS;
    ((device half4*)a.out)[t] = half4(v);
  }
}
kernel void be2_stack(LF_ARGS(Be2StackArgs)) {
  ulong t = LF_INDEX;
  const ulong na4 = a.na >> 2, nb4 = a.nb >> 2;
  if (t < na4) ((device half4*)a.out)[t] = ((device const half4*)a.a)[t];
  else if (t < na4 + nb4) ((device half4*)a.out)[t] = ((device const half4*)a.b)[t - na4];
}

// ---------------------------------------------------------------- the denoiser
kernel void be2_scale_masked(LF_ARGS(Be2ScaleMaskedArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.n * 3) return;
  BE2_SPLIT(t / 3, (uint)a.N, s, at);
  (void)s;
  a.r[t] = a.mask[at] ? a.x[t] * a.s : 0.f;
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
  if (t >= a.rows * a.C) return;
  BE2_SPLIT(t, a.C, r, c);
  (void)r;
  a.y[t] += a.v[c];
}
kernel void be2_scaled_add(LF_ARGS(Be2ScaledAddArgs)) {
  ulong t = LF_INDEX;
  if (t < a.n) a.y[t] += a.s * a.x[t];
}
kernel void be2_repeat(LF_ARGS(Be2RepeatArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.total) return;
  BE2_SPLIT(t, (uint)a.elems, k, e);
  (void)k;
  a.dst[t] = a.src[e];
}
