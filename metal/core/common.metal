// metal/core's kernels: copies and fills in stream order, the bundle decoder, LayerNorm, conversions, elementwise.
// (args.h is prepended by metal/build.sh)
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// a kernel's arguments and its linear index over a grid folded into y (core.h's grid1d)
#define LF_ARGS(T) constant T& a [[buffer(0)]], uint3 _tg [[threadgroup_position_in_grid]], \
  uint3 _ng [[threadgroups_per_grid]], uint _tid [[thread_index_in_threadgroup]], uint3 _tpg [[threads_per_threadgroup]]
#define LF_INDEX ((ulong)(_tg.y * _ng.x + _tg.x) * _tpg.x + _tid)
// a 32-bit unsigned division by a runtime divisor: Apple's GPUs have no integer divider, and `a / d` is a software
// routine - a float reciprocal and one correction instead, exact for a < 2^24
inline uint lf_udiv(uint a, uint d) {
  if (a >= (1u << 24)) return a / d;
  uint q = (uint)((float)a * (1.f / (float)d));
  if (q * d > a) --q;
  else if ((q + 1) * d <= a) ++q;
  return q;
}
inline float lf_sigmoid(float x) { return 1.f / (1.f + exp(-x)); }
inline float lf_silu(float x) { return x / (1.f + exp(-x)); }
// erf (Metal has none): Abramowitz and Stegun 7.1.26, |error| < 1.5e-7
inline float lf_erf(float x) {
  float s = sign(x), t = 1.f / (1.f + 0.3275911f * fabs(x));
  float y = 1.f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
  return s * y;
}
inline float lf_gelu(float x) { return 0.5f * x * (1.f + lf_erf(x * 0.70710678118654752f)); }
// round to bfloat16 (nearest even) and back
inline float lf_bf16(float v) {
  uint w = as_type<uint>(v);
  w = (w + 0x7fffu + ((w >> 16) & 1u)) & 0xffff0000u;
  return as_type<float>(w);
}

kernel void lf_copy(LF_ARGS(CopyArgs)) {
  ulong i = LF_INDEX;
  if ((((ulong)a.dst | (ulong)a.src) & 15) == 0) {
    ulong n16 = a.bytes / 16;
    if (i < n16) ((device uint4*)a.dst)[i] = ((device const uint4*)a.src)[i];
    ulong tail = n16 * 16 + i;
    if (i < 16 && tail < a.bytes) a.dst[tail] = a.src[tail];
  } else if (i < a.bytes) {
    a.dst[i] = a.src[i];
  }
}
kernel void lf_copy2d(LF_ARGS(Copy2DArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.width * a.height) return;
  ulong y = t / a.width, x = t - y * a.width;
  a.dst[y * a.dpitch + x] = a.src[y * a.spitch + x];
}
kernel void lf_fill(LF_ARGS(FillArgs)) {
  ulong i = LF_INDEX;
  uchar v = (uchar)a.value;
  if ((((ulong)a.dst) & 15) == 0) {
    ulong n16 = a.bytes / 16;
    uint w = (uint)v * 0x01010101u;
    if (i < n16) ((device uint4*)a.dst)[i] = uint4(w);
    ulong tail = n16 * 16 + i;
    if (i < 16 && tail < a.bytes) a.dst[tail] = v;
  } else if (i < a.bytes) {
    a.dst[i] = v;
  }
}

// ---------------------------------------------------------------- the bundle decoder: a table entry a grid row
inline float lf_half_at(device const uchar* p) { return (float)as_type<half>((ushort)(p[0] | (p[1] << 8))); }
kernel void lf_decode(constant DecodeArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  DecodeEntry d = a.table[tg.y];
  for (ulong o = (ulong)tg.x * 256 + tid; o < d.n; o += (ulong)ng.x * 256) {
    ulong i = d.first + o;
    float v;
    if (d.kind == 0) v = as_type<float>(*(device const uint*)(a.raw + d.src + 4 * i));
    else if (d.kind == 1) v = lf_half_at(a.raw + d.src + 2 * i);
    else {
      ulong g = i / d.block, bit = i * d.bits, byte = bit >> 3;
      uint sh = (uint)(bit & 7);
      uint word = a.raw[d.src + byte];
      if (sh + d.bits > 8) word |= (uint)a.raw[d.src + byte + 1] << 8;
      uint code = (word >> sh) & ((1u << d.bits) - 1);
      v = fma((float)code, lf_half_at(a.raw + d.scale + 2 * g), lf_half_at(a.raw + d.zero + 2 * g));
    }
    if (d.flags & 1) v = (float)(half)v;
    if (d.out16) {
      device half* p = (device half*)d.dst + o;
      *p = (d.flags & 2) ? (half)((float)*p + v) : (half)v;
    } else {
      device float* p = (device float*)d.dst + o;
      *p = (d.flags & 2) ? *p + v : v;
    }
  }
}
// a weight walk's parts: a grid row a part
kernel void lf_gather(constant GatherArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  GatherPart p = a.parts[tg.y];
  for (ulong o = (ulong)tg.x * 256 + tid; o < p.n; o += (ulong)ng.x * 256) {
    ulong r = o; long d = 0, x = 0, y = 0;
    for (int k = p.rank - 1; k >= 0; --k) {
      long i = (long)(r % (ulong)p.dims[k]); r /= (ulong)p.dims[k];
      d += i * p.ds[k]; x += i * p.s0[k]; y += i * p.s1[k];
    }
    float v;
    if (p.op == 'o') v = 1.f;
    else if (p.op == 'x') v = ((device const float*)p.src0)[x] * ((device const float*)p.src1)[y];
    else if (p.op == 'i') v = as_type<float>((int)((device const float*)p.src0)[x]);
    else v = ((device const float*)p.src0)[x];
    ((device float*)p.dst)[d] = v;
  }
}

// ---------------------------------------------------------------- LayerNorm: a simdgroup a row, eight rows a threadgroup
kernel void lf_layernorm(constant LayerNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
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
  for (int c = lane; c < C; c += 32) {
    float n = ((x ? x[c] : (float)xh[c]) - mean) * inv;
    if (a.scale) n *= a.scale[c];
    if (a.offset) n += a.offset[c];
    if (a.yh) a.yh[row * a.ldy + c] = (half)n;
    else a.y[row * a.ldy + c] = n;
  }
}
// C a multiple of 4 up to 1536: the row held in registers, read once, four channels a load
kernel void lf_layernorm4(constant LayerNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  ulong row = (ulong)(tg.y * ng.x + tg.x) * 8 + sg;
  if (row >= a.rows) return;
  const int C4 = a.C / 4;
  constexpr int MAXV = 12;
  float4 r[MAXV];
  float s = 0;
  const bool aligned = (a.ldx & 3) == 0;
  for (int k = 0; k < MAXV; ++k) {
    int c = lane + 32 * k;
    if (c < C4) {
      if (a.x) r[k] = aligned ? ((device const float4*)(a.x + row * a.ldx))[c]
                              : float4(a.x[row * a.ldx + 4 * c], a.x[row * a.ldx + 4 * c + 1], a.x[row * a.ldx + 4 * c + 2], a.x[row * a.ldx + 4 * c + 3]);
      else r[k] = float4(((device const half4*)(a.xh + row * a.ldx))[c]);
      s += r[k].x + r[k].y + r[k].z + r[k].w;
    }
  }
  float mean = simd_sum(s) / a.C, v = 0;
  for (int k = 0; k < MAXV; ++k)
    if (lane + 32 * k < (uint)C4) { float4 d = r[k] - mean; v += dot(d, d); }
  float inv = rsqrt(simd_sum(v) / a.C + a.eps);
  for (int k = 0; k < MAXV; ++k) {
    int c = lane + 32 * k;
    if (c >= C4) continue;
    float4 n = (r[k] - mean) * inv;
    if (a.scale) n *= ((device const float4*)a.scale)[c];
    if (a.offset) n += ((device const float4*)a.offset)[c];
    if (a.yh) {
      if ((a.ldy & 3) == 0) ((device half4*)(a.yh + row * a.ldy))[c] = half4(n);
      else for (int e = 0; e < 4; ++e) a.yh[row * a.ldy + 4 * c + e] = (half)n[e];
    } else {
      if ((a.ldy & 3) == 0) ((device float4*)(a.y + row * a.ldy))[c] = n;
      else for (int e = 0; e < 4; ++e) a.y[row * a.ldy + 4 * c + e] = n[e];
    }
  }
}

// ---------------------------------------------------------------- conversions and elementwise (four a thread)
kernel void lf_to_half(LF_ARGS(ConvArgs)) {
  ulong i = LF_INDEX * 4;
  for (int e = 0; e < 4; ++e) if (i + e < a.n) a.y[i + e] = (half)a.x[i + e];
}
kernel void lf_to_float(LF_ARGS(ConvBackArgs)) {
  ulong i = LF_INDEX * 4;
  for (int e = 0; e < 4; ++e) if (i + e < a.n) a.y[i + e] = (float)a.x[i + e];
}
kernel void lf_add(LF_ARGS(AddArgs)) {
  ulong i = LF_INDEX * 4;
  for (int e = 0; e < 4; ++e) if (i + e < a.n) a.y[i + e] += a.a * a.x[i + e];
}
kernel void lf_add_bias(LF_ARGS(BiasArgs)) {
  ulong t = LF_INDEX;
  if (t >= a.rows * a.C) return;
  uint c = t < 0xffffffffu ? (uint)t - lf_udiv((uint)t, (uint)a.C) * (uint)a.C : (uint)(t % (ulong)a.C);
  float v = a.y[t] + a.b[c];
  if (a.act == 1) v = max(v, 0.f);
  else if (a.act == 2) v = lf_gelu(v);
  a.y[t] = v;
}

// ---------------------------------------------------------------- gated flash attention
// A threadgroup is 64 queries of one (row, head), four simdgroups of 16; keys in tiles of 32 staged in threadgroup
// memory. Scores and P V on 8 x 8 simdgroup matrices (half in, float accumulated), the softmax online in the log2
// domain (the query carries scale log2e, the bias is log2-scaled already). A lane holds row sm of each 8 x 8 matrix,
// columns sn and sn + 1; a row's four lanes differ in lane bits 0 and 3.
template <int D>
kernel void lf_attention(constant AttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int QB = 64, KT = 32, LD = D + 8, DF = D / 8;
  threadgroup half Qs[QB * LD];
  threadgroup half Ks[KT * LD];
  threadgroup half Vs[KT * LD];
  const int b = tg.y, h = tg.z, q0 = tg.x * QB, n = a.n, W = a.heads * D;
  device const half* base = a.qkvg + (long)b * a.rowStride + h * D;
  const float qs = a.scale * M_LOG2E_F;
  for (int e = tid; e < QB * D; e += 128) {
    int qi = e / D, d = e - qi * D, q = q0 + qi;
    float v = q < n ? (float)base[(long)q * a.posStride + d] + (a.qBias ? a.qBias[h * D + d] : 0.f) : 0.f;
    Qs[qi * LD + d] = (half)(v * qs);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[2][DF];
  for (int r = 0; r < 2; ++r)
    for (int dk = 0; dk < DF; ++dk) simdgroup_load(qf[r][dk], Qs + (sg * 16 + r * 8) * LD + dk * 8, LD);
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f};
  simdgroup_float8x8 o[2][DF];
  for (int r = 0; r < 2; ++r) for (int c = 0; c < DF; ++c) o[r][c] = simdgroup_float8x8(0);
  const long bq = (long)(a.r0 + b);
  for (int k0 = 0; k0 < n; k0 += KT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < KT * D; e += 128) {
      int kj = e / D, d = e - kj * D, k = k0 + kj;
      bool live = k < n;
      Ks[kj * LD + d] = live ? base[(long)k * a.posStride + W + d] : (half)0;
      Vs[kj * LD + d] = live ? base[(long)k * a.posStride + 2 * W + d] : (half)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 s[2][4];
    for (int r = 0; r < 2; ++r) for (int j = 0; j < 4; ++j) s[r][j] = simdgroup_float8x8(0);
    for (int j = 0; j < 4; ++j)
      for (int dk = 0; dk < DF; ++dk) {
        simdgroup_half8x8 kt;
        simdgroup_load(kt, Ks + (j * 8) * LD + dk * 8, LD, ulong2(0, 0), true);
        for (int r = 0; r < 2; ++r) simdgroup_multiply_accumulate(s[r][j], qf[r][dk], kt, s[r][j]);
      }
    // the bias, the mask and the keys past n; each row's new maximum
    float mk[4][2];
    for (int j = 0; j < 4; ++j)
      for (int t = 0; t < 2; ++t) {
        int k = k0 + j * 8 + sn + t;
        float v = 0.f;
        if (k >= n) v = -1e30f;
        else if (a.mask) v = a.mask[a.maskT ? (long)k * n + bq : bq * n + k] > 0.f ? 0.f : -1e9f;
        mk[j][t] = v;
      }
    simdgroup_half8x8 p[2][4];
    for (int r = 0; r < 2; ++r) {
      const int q = q0 + sg * 16 + r * 8 + sm;
      device const half* brow = a.bias ? a.bias + ((long)h * n + min(q, n - 1)) * a.biasStride : nullptr;
      float rowMax = -1e30f;
      for (int j = 0; j < 4; ++j) {
        thread auto& e = s[r][j].thread_elements();
        for (int t = 0; t < 2; ++t) {
          int k = k0 + j * 8 + sn + t;
          float v = e[t] + mk[j][t] + (brow && k < n ? (float)brow[k] : 0.f);
          e[t] = v;
          rowMax = max(rowMax, v);
        }
      }
      rowMax = max(rowMax, simd_shuffle_xor(rowMax, (ushort)1));
      rowMax = max(rowMax, simd_shuffle_xor(rowMax, (ushort)8));
      float mn = max(m[r], rowMax), corr = exp2(m[r] - mn), sum = 0.f;
      for (int j = 0; j < 4; ++j) {
        thread auto& e = s[r][j].thread_elements();
        thread auto& pe = p[r][j].thread_elements();
        for (int t = 0; t < 2; ++t) { float pv = exp2(e[t] - mn); sum += pv; pe[t] = (half)pv; }
      }
      sum += simd_shuffle_xor(sum, (ushort)1);
      sum += simd_shuffle_xor(sum, (ushort)8);
      l[r] = l[r] * corr + sum; m[r] = mn;
      for (int c = 0; c < DF; ++c) { thread auto& oe = o[r][c].thread_elements(); oe[0] *= corr; oe[1] *= corr; }
    }
    for (int j = 0; j < 4; ++j)
      for (int c = 0; c < DF; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, Vs + (j * 8) * LD + c * 8, LD);
        for (int r = 0; r < 2; ++r) simdgroup_multiply_accumulate(o[r][c], p[r][j], vf, o[r][c]);
      }
  }
  // out = O / l * sigmoid(gate)
  for (int r = 0; r < 2; ++r) {
    const int q = q0 + sg * 16 + r * 8 + sm;
    if (q >= n) continue;
    device const half* g = base + (long)q * a.posStride + 3 * W;
    device half* out = a.out + (long)b * a.outRowStride + (long)q * a.outPosStride + h * D;
    const float inv = 1.f / l[r];
    for (int c = 0; c < DF; ++c) {
      thread auto& oe = o[r][c].thread_elements();
      for (int t = 0; t < 2; ++t) {
        int d = c * 8 + sn + t;
        out[d] = (half)(oe[t] * inv * lf_sigmoid((float)g[d]));
      }
    }
  }
}
template [[host_name("lf_attention_8")]] kernel void lf_attention<8>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_16")]] kernel void lf_attention<16>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_32")]] kernel void lf_attention<32>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_48")]] kernel void lf_attention<48>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_64")]] kernel void lf_attention<64>(constant AttnArgs&, uint3, uint, uint, uint);
