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
    float v;
    if (d.kind == 0) v = as_type<float>(*(device const uint*)(a.raw + d.src + 4 * o));
    else if (d.kind == 1) v = lf_half_at(a.raw + d.src + 2 * o);
    else {
      uint g = (uint)(o / d.block), within = (uint)(o - (ulong)g * d.block);
      ulong bit = (ulong)g * d.block * d.bits + (ulong)within * d.bits, byte = bit >> 3;
      uint sh = (uint)(bit & 7);
      uint word = a.raw[d.src + byte];
      if (sh + d.bits > 8) word |= (uint)a.raw[d.src + byte + 1] << 8;
      uint code = (word >> sh) & ((1u << d.bits) - 1);
      v = fma((float)code, lf_half_at(a.raw + d.scale + 2 * g), lf_half_at(a.raw + d.zero + 2 * g));
    }
    if (d.out16) ((device half*)d.dst)[o] = (half)v;
    else ((device float*)d.dst)[o] = v;
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
