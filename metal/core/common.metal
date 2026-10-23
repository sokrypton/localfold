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
  ulong y, x;      // (32-bit division where it fits: the 64-bit one is slow)
  if (t <= 0xffffffffull && a.width <= 0xffffffffull) { uint y32 = lf_udiv((uint)t, (uint)a.width); y = y32; x = (uint)t - y32 * (uint)a.width; }
  else { y = t / a.width; x = t - y * a.width; }
  a.dst[y * a.dpitch + x] = a.src[y * a.spitch + x];
}
kernel void lf_copy2d16(LF_ARGS(Copy2DArgs)) {     // (pitches and width in 16-byte units)
  ulong t = LF_INDEX;
  if (t >= a.width * a.height) return;
  ulong y, x;
  if (t <= 0xffffffffull && a.width <= 0xffffffffull) { uint y32 = lf_udiv((uint)t, (uint)a.width); y = y32; x = (uint)t - y32 * (uint)a.width; }
  else { y = t / a.width; x = t - y * a.width; }
  ((device uint4*)a.dst)[y * a.dpitch + x] = ((device const uint4*)a.src)[y * a.spitch + x];
}
kernel void lf_pad_zero(LF_ARGS(PadZeroArgs)) {
  // (32-bit index arithmetic: the GPU's 64-bit division is slow - 104 ms a trunk pass at 510 tokens with it)
  const uint i = (uint)LF_INDEX, np = a.np, n = a.n, w = np - n;
  const uint per = np * np - n * n;
  if ((ulong)i >= a.planes * per) return;
  const uint pl = lf_udiv(i, per), r = i - pl * per, tail = w * np;
  uint pos;
  if (r < tail) pos = n * np + r;                                    // rows n..np-1, whole
  else { const uint k = r - tail, kr = lf_udiv(k, w); pos = kr * np + n + (k - kr * w); }   // rows ..n-1, columns n..
  a.dst[(ulong)pl * np * np + pos] = 0.h;
}

kernel void lf_tri_quarters(LF_ARGS(TriQuartersArgs)) {
  ulong i = LF_INDEX;
  const uint W4 = 4 * a.C;
  if (i >= a.rows * W4) return;
  const uint r = lf_udiv((uint)i, W4), nc = (uint)i - r * W4, t = nc / 128, p = (nc % 128) / 32, ch = t * 32 + nc % 32;
  const uint oc = (ch / 8) * 32 + p * 8 + ch % 8;
  a.dst[i] = a.src[i - nc + oc];
  if (a.srcf && i < W4) a.dstf[nc] = a.srcf[oc];
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
// four bytes as a float, at any alignment (an af3-any-model blob's records have headers of any length)
inline float lf_f32_at(device const uchar* p) {
  return as_type<float>((uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24));
}
// (the index arithmetic in 32 bits where the entry's indices fit - always, in practice: the GPU's 64-bit division made
// this kernel ~2 GB/s, 300 ms of boltz2's load)
template <typename IX>
inline float lf_decode_at(constant DecodeArgs& a, thread const DecodeEntry& d, IX i) {
  if (d.kind == 0) return lf_f32_at(a.raw + d.src + 4 * (ulong)i);
  if (d.kind == 1) return lf_half_at(a.raw + d.src + 2 * (ulong)i);
  if (d.kind == 2) return (float)(char)a.raw[d.src + i] * lf_half_at(a.raw + d.scale + 2 * (ulong)(i / (IX)d.block));   // symmetric int8
  if (d.kind == 6) {             // an af3-any-model blob's int8: a float32 scale per channel of the last axis (`block` of
                                 // them) and per block of rows (`bits` blocks of the `zero` rows)
    IX row = i / (IX)d.block, col = i - row * (IX)d.block, g = (IX)((d.zero + d.bits - 1) / d.bits);
    float scale = lf_f32_at(a.raw + d.scale + 4 * (ulong)((row / g) * (IX)d.block + col));
    return (float)(char)a.raw[d.src + i] * scale;
  }
  if (d.kind == 7) {             // bfloat16
    uint h = (uint)a.raw[d.src + 2 * (ulong)i] | ((uint)a.raw[d.src + 2 * (ulong)i + 1] << 8);
    return as_type<float>(h << 16);
  }
  IX g = i / (IX)d.block;
  ulong bit = (ulong)i * d.bits, byte = bit >> 3;
  uint sh = (uint)(bit & 7);
  uint word = a.raw[d.src + byte];
  if (sh + d.bits > 8) word |= (uint)a.raw[d.src + byte + 1] << 8;
  uint code = (word >> sh) & ((1u << d.bits) - 1);
  return fma((float)code, lf_half_at(a.raw + d.scale + 2 * (ulong)g), lf_half_at(a.raw + d.zero + 2 * (ulong)g));
}
kernel void lf_decode(constant DecodeArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  DecodeEntry d = a.table[tg.y];
  const bool small = d.first + d.n <= 0xffffffffull;
  for (ulong o = (ulong)tg.x * 256 + tid; o < d.n; o += (ulong)ng.x * 256) {
    ulong i = d.first + o;
    float v = small ? lf_decode_at<uint>(a, d, (uint)i) : lf_decode_at<ulong>(a, d, i);
    if (d.flags & 1) v = (float)(half)v;
    ulong dstO = o;
    if (d.pad) {
      uint W = 2 * d.pad, r = lf_udiv((uint)o, W), col = (uint)o - r * W;
      uint c = col >= d.pad ? col - d.pad : col;
      dstO = (ulong)r * W + ((c & ~7u) << 1) + (col >= d.pad ? 8u : 0u) + (c & 7u);
    }
    if (d.out16) {
      device half* p = (device half*)d.dst + dstO;
      *p = (d.flags & 2) ? (half)((float)*p + v) : (half)v;
    } else {
      device float* p = (device float*)d.dst + dstO;
      *p = (d.flags & 2) ? *p + v : v;
    }
  }
}
// a weight walk's parts: a grid row a part
kernel void lf_gather(constant GatherArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                      uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  GatherPart p = a.parts[tg.y];
  const bool small = p.n <= 0xffffffffull;     // (the index split in 32 bits where it fits: 64-bit division is slow)
  for (ulong o = (ulong)tg.x * 256 + tid; o < p.n; o += (ulong)ng.x * 256) {
    long d = 0, x = 0, y = 0;
    if (small) {
      uint r = (uint)o;
      for (int k = p.rank - 1; k >= 0; --k) {
        const uint dk = (uint)p.dims[k], q = r / dk; const long i = (long)(r - q * dk); r = q;
        d += i * p.ds[k]; x += i * p.s0[k]; y += i * p.s1[k];
      }
    } else {
      ulong r = o;
      for (int k = p.rank - 1; k >= 0; --k) {
        long i = (long)(r % (ulong)p.dims[k]); r /= (ulong)p.dims[k];
        d += i * p.ds[k]; x += i * p.s0[k]; y += i * p.s1[k];
      }
    }
    float v;
    if (p.op == 'o') v = 1.f;
    else if (p.op == 'x') v = ((device const float*)p.src0)[x] * ((device const float*)p.src1)[y];
    else if (p.op == 'i') v = as_type<float>((int)((device const float*)p.src0)[x]);
    else v = ((device const float*)p.src0)[x];
    ((device float*)p.dst)[d] = v;
  }
}

// ---------------------------------------------------------------- LayerNorm: a simdgroup a row
kernel void lf_layernorm(constant LayerNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ulong row = (ulong)(tg.y * ng.x + tg.x) * sgs + sg;
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
    float n0 = ((x ? x[c] : (float)xh[c]) - mean) * inv;
    float n = n0;
    if (a.scale) n *= a.scale[c];
    if (a.offset) n += a.offset[c];
    if (a.yh) a.yh[row * a.ldy + c] = (half)n;
    if (a.y) a.y[row * a.ldy + c] = n;
    if (a.yh2) {
      float n2 = n0;
      if (a.scale2) n2 *= a.scale2[c];
      if (a.offset2) n2 += a.offset2[c];
      a.yh2[row * a.ldy + c] = (half)n2;
    }
  }
}
// C a multiple of 4 up to 1536: the row held in registers, read once, four channels a load - MAXV of them a lane, the
// fewest that hold the row (an array sized for the widest row cost a 128-channel one its occupancy)
template <int MAXV>
kernel void lf_layernorm4(constant LayerNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint3 ng [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ulong row = (ulong)(tg.y * ng.x + tg.x) * sgs + sg;
  if (row >= a.rows) return;
  const int C4 = a.C / 4;
  float4 r[MAXV];
  float s = 0;
  const bool aligned = (a.ldx & 3) == 0;
  const bool full = C4 == MAXV * 32;
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < MAXV; ++k) {
    int c = lane + 32 * k;
    if (full || c < C4) {
      if (a.x) r[k] = aligned ? ((device const float4*)(a.x + row * a.ldx))[c]
                              : float4(a.x[row * a.ldx + 4 * c], a.x[row * a.ldx + 4 * c + 1], a.x[row * a.ldx + 4 * c + 2], a.x[row * a.ldx + 4 * c + 3]);
      else r[k] = float4(((device const half4*)(a.xh + row * a.ldx))[c]);
      s += r[k].x + r[k].y + r[k].z + r[k].w;
    }
  }
  float mean = simd_sum(s) / a.C, v = 0;
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < MAXV; ++k)
    if (full || lane + 32 * k < (uint)C4) { float4 d = r[k] - mean; v += dot(d, d); }
  float inv = rsqrt(simd_sum(v) / a.C + a.eps);
  const bool outAligned = (a.ldy & 3) == 0;
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < MAXV; ++k) {
    int c = lane + 32 * k;
    if (!full && c >= C4) continue;
    float4 n0 = (r[k] - mean) * inv;
    float4 n = n0;
    if (a.scale) n *= ((device const float4*)a.scale)[c];
    if (a.offset) n += ((device const float4*)a.offset)[c];
    if (a.yh) {
      if (outAligned) ((device half4*)(a.yh + row * a.ldy))[c] = half4(n);
      else for (int e = 0; e < 4; ++e) a.yh[row * a.ldy + 4 * c + e] = (half)n[e];
    }
    if (a.y) {
      if (outAligned) ((device float4*)(a.y + row * a.ldy))[c] = n;
      else for (int e = 0; e < 4; ++e) a.y[row * a.ldy + 4 * c + e] = n[e];
    }
    if (a.yh2) {
      float4 n2 = n0;
      if (a.scale2) n2 *= ((device const float4*)a.scale2)[c];
      if (a.offset2) n2 += ((device const float4*)a.offset2)[c];
      if (outAligned) ((device half4*)(a.yh2 + row * a.ldy))[c] = half4(n2);
      else for (int e = 0; e < 4; ++e) a.yh2[row * a.ldy + 4 * c + e] = (half)n2[e];
    }
  }
}

#define LF_LN4(V) template [[host_name("lf_layernorm4_" #V)]] kernel void lf_layernorm4<V>(constant LayerNormArgs&, uint3, uint3, uint, uint, uint);
LF_LN4(1) LF_LN4(2) LF_LN4(3) LF_LN4(4) LF_LN4(6) LF_LN4(8) LF_LN4(9) LF_LN4(12)

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

// the triangle's centre LayerNorm: the channel-major product [C][Lp * Lp] to pair rows [pairs][C] in half. A lane a pair,
// eight simdgroups splitting the channels, every value in registers, the two-pass statistics through a 1 KB tile, the
// rows written coalesced through a half tile (C a multiple of 8, at most 256)
// (NK: the channels a lane holds, C / 8 - at C 128 the registers and the tile sized for 256 took the core's occupancy)
template <int NK>
kernel void lf_center_norm(constant CenterNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  device const float* prod = a.prod;
  threadgroup half T[32 * 8 * NK];
  threadgroup float* part = (threadgroup float*)T;
  const uint lane = tid & 31, grp = tid >> 5, C = a.C, nk = C / 8;
  const bool exact = C == 8 * NK;
  const uint ld = exact ? (8 * NK) : (C + 4);
  const ulong r0 = (ulong)(tg.y * ng.x + tg.x) * 32;
  const uint rr = (uint)(r0 + lane), ii = lf_udiv(rr, a.L);
  const ulong q = (ulong)ii * a.Lp + (rr - ii * a.L), plane = (ulong)a.Lp * a.Lp;
  const bool live = r0 + lane < a.pairs;
  float v[NK];
  float s = 0.f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < NK; ++k) if (exact || k < nk) {
    uint c = exact ? (4 * grp + 32 * (k >> 2) + (k & 3)) : (grp + 8 * k);
    v[k] = live ? prod[(ulong)c * plane + q] : 0.f;
    s += v[k];
  }
  part[grp * 32 + lane] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float S = 0.f;
  _Pragma("clang loop unroll(full)")
  for (int g = 0; g < 8; ++g) S += part[g * 32 + lane];
  const float mean = S / C;
  float d2 = 0.f;
  _Pragma("clang loop unroll(full)")
  for (uint k = 0; k < NK; ++k) if (exact || k < nk) { float d = v[k] - mean; d2 += d * d; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  part[grp * 32 + lane] = d2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float V = 0.f;
  _Pragma("clang loop unroll(full)")
  for (int g = 0; g < 8; ++g) V += part[g * 32 + lane];
  const float inv = rsqrt(V / C + a.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (exact) {
    const uint swiz = (lane & 15) << 2;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < NK / 4; ++k) {
      uint c0 = 4 * grp + 32 * k;
      float4 sc = *(device const float4*)(a.scale + c0);
      float4 of = *(device const float4*)(a.offset + c0);
      *(threadgroup half4*)(T + lane * (8 * NK) + (c0 ^ swiz)) = half4(
          (half)((v[4 * k + 0] - mean) * inv * sc.x + of.x),
          (half)((v[4 * k + 1] - mean) * inv * sc.y + of.y),
          (half)((v[4 * k + 2] - mean) * inv * sc.z + of.z),
          (half)((v[4 * k + 3] - mean) * inv * sc.w + of.w));
    }
  } else {
    for (uint k = 0; k < nk; ++k) {
      uint c = grp + 8 * k;
      T[lane * ld + c] = (half)((v[k] - mean) * inv * a.scale[c] + a.offset[c]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint row = grp; row < 32; row += 8) {
    ulong r = r0 + row;
    if (r >= a.pairs) break;
    if (exact) {
      const uint swiz = (row & 15) << 2;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < NK / 16; ++k) {
        const uint c = 128 * k + 4 * lane;
        *(device half4*)(a.out + r * C + c) = *(threadgroup const half4*)(T + row * (8 * NK) + (c ^ swiz));
      }
    } else if ((C & 3) == 0) {
      for (uint c = 4 * lane; c < C; c += 128) *(device half4*)(a.out + r * C + c) = *(threadgroup const half4*)(T + row * ld + c);
    } else {
      for (uint c = lane; c < C; c += 32) a.out[r * C + c] = T[row * ld + c];
    }
  }
}
// ...from a half product (scaled; CenterNormArgs.prodh): PP pairs a lane - the product's reads twice as long, where the
// half tile still fits - and the rows written four channels a lane (ESMFold2's 0.82 -> 0.75 ms a call at 256 channels,
// AF3's 0.40 -> 0.38 at 128; the same values)
template <int NK, int PP>
kernel void lf_center_norm_h(constant CenterNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                             uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint R = 32 * PP;
  device const half* prod = a.prodh;
  threadgroup half T[32 * 8 * NK];
  threadgroup float* part = (threadgroup float*)T;
  const uint lane = tid & 31, grp = tid >> 5, C = a.C, nk = C / 8;
  const bool exact = C == 8 * NK;
  const uint ld = exact ? (8 * NK) : (C + 4);
  const ulong r0 = (ulong)(tg.y * ng.x + tg.x) * R, plane = (ulong)a.Lp * a.Lp;
  ulong q[PP]; bool live[PP];
  _Pragma("clang loop unroll(full)")
  for (uint p = 0; p < PP; ++p) {
    const uint rr = (uint)(r0 + PP * lane + p), ii = lf_udiv(rr, a.L);
    q[p] = (ulong)ii * a.Lp + (rr - ii * a.L);
    live[p] = r0 + PP * lane + p < a.pairs;
  }
  half v[PP][NK];
  if (PP == 2 && exact && live[0] && live[1] && q[1] == q[0] + 1 && (q[0] & 1) == 0) {
    float s0 = 0.f, s1 = 0.f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < NK / 4; ++k) {
      _Pragma("clang loop unroll(full)")
      for (uint j = 0; j < 4; ++j) {
        uint c = 4 * grp + 32 * k + j;
        half2 h = *(device const half2*)(prod + (ulong)c * plane + q[0]);
        v[0][4 * k + j] = h.x; v[1][4 * k + j] = h.y;
        s0 += (float)h.x; s1 += (float)h.y;
      }
    }
    part[grp * R + 2 * lane] = s0; part[grp * R + 2 * lane + 1] = s1;
  } else {
    _Pragma("clang loop unroll(full)")
    for (uint p = 0; p < PP; ++p) {
      float s = 0.f;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < NK; ++k) if (exact || k < nk) {
        uint c = exact ? (4 * grp + 32 * (k >> 2) + (k & 3)) : (grp + 8 * k);
        half h = live[p] ? prod[(ulong)c * plane + q[p]] : 0.h;
        v[p][k] = h;
        s += (float)h;
      }
      part[grp * R + PP * lane + p] = s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean[PP];
  _Pragma("clang loop unroll(full)")
  for (uint p = 0; p < PP; ++p) { float S = 0.f; _Pragma("clang loop unroll(full)") for (int g = 0; g < 8; ++g) S += part[g * R + PP * lane + p]; mean[p] = S / C; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (uint p = 0; p < PP; ++p) {
    float d2 = 0.f;
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < NK; ++k) if (exact || k < nk) { float d = (float)v[p][k] - mean[p]; d2 += d * d; }
    part[grp * R + PP * lane + p] = d2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[PP];
  _Pragma("clang loop unroll(full)")
  for (uint p = 0; p < PP; ++p) {
    float V = 0.f;
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < 8; ++g) V += part[g * R + PP * lane + p];
    inv[p] = rsqrt(V / C + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  _Pragma("clang loop unroll(full)")
  for (uint p = 0; p < PP; ++p) {
    if (p > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
    if (exact) {
      const uint swiz = (lane & 15) << 2;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < NK / 4; ++k) {
        uint c0 = 4 * grp + 32 * k;
        float4 sc = *(device const float4*)(a.scale + c0);
        float4 of = *(device const float4*)(a.offset + c0);
        *(threadgroup half4*)(T + lane * (8 * NK) + (c0 ^ swiz)) = half4(
            (half)(((float)v[p][4 * k + 0] - mean[p]) * inv[p] * sc.x + of.x),
            (half)(((float)v[p][4 * k + 1] - mean[p]) * inv[p] * sc.y + of.y),
            (half)(((float)v[p][4 * k + 2] - mean[p]) * inv[p] * sc.z + of.z),
            (half)(((float)v[p][4 * k + 3] - mean[p]) * inv[p] * sc.w + of.w));
      }
    } else {
      for (uint k = 0; k < nk; ++k) {
        uint c = grp + 8 * k;
        T[lane * ld + c] = (half)(((float)v[p][k] - mean[p]) * inv[p] * a.scale[c] + a.offset[c]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (uint row = grp; row < 32; row += 8) {
      ulong r = r0 + PP * row + p;
      if (r >= a.pairs) break;
      if (exact) {
        const uint swiz = (row & 15) << 2;
        if constexpr (NK == 8) {
          if (lane < 16) {
            const uint c = 4 * lane;
            *(device half4*)(a.out + r * C + c) = *(threadgroup const half4*)(T + row * (8 * NK) + (c ^ swiz));
          }
        } else {
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < NK / 16; ++k) {
            const uint c = 128 * k + 4 * lane;
            *(device half4*)(a.out + r * C + c) = *(threadgroup const half4*)(T + row * (8 * NK) + (c ^ swiz));
          }
        }
      } else {
        for (uint c = 4 * lane; c < C; c += 128) *(device half4*)(a.out + r * C + c) = *(threadgroup const half4*)(T + row * ld + c);
      }
    }
  }
}
template [[host_name("lf_center_norm_8h2")]] kernel void lf_center_norm_h<8, 2>(constant CenterNormArgs&, uint3, uint3, uint);
template [[host_name("lf_center_norm_16h2")]] kernel void lf_center_norm_h<16, 2>(constant CenterNormArgs&, uint3, uint3, uint);
template [[host_name("lf_center_norm_32h1")]] kernel void lf_center_norm_h<32, 1>(constant CenterNormArgs&, uint3, uint3, uint);
template [[host_name("lf_center_norm_16")]] kernel void lf_center_norm<16>(constant CenterNormArgs&, uint3, uint3, uint);
template [[host_name("lf_center_norm_32")]] kernel void lf_center_norm<32>(constant CenterNormArgs&, uint3, uint3, uint);
// ...for any width (OpenDDE's 384, IntelliFold-2's 512): the channels read once into registers when C % 128 == 0 && C <= 512,
// and transposed in 128-channel chunks through an 8 KB half tile so global output writes are coalesced half4 stores
template <typename TP>
kernel void lf_center_norm_wide_t(constant CenterNormArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half T[32 * 128];
  threadgroup float* part = (threadgroup float*)T;
  threadgroup float* part2 = part + 512;
  device const TP* prod = sizeof(TP) == 2 ? (device const TP*)a.prodh : (device const TP*)a.prod;
  const uint lane = tid & 31, grp = tid >> 5, C = a.C;
  const ulong r0 = (ulong)(tg.y * ng.x + tg.x) * 32;
  const uint rr = (uint)(r0 + lane), ii = lf_udiv(rr, a.L);
  const ulong q = (ulong)ii * a.Lp + (rr - ii * a.L), plane = (ulong)a.Lp * a.Lp;
  const bool live = r0 + lane < a.pairs;
  if (sizeof(TP) == 2 && (C & 127) == 0 && C <= 512) {
    const ulong r64 = (ulong)(tg.y * ng.x + tg.x) * 64;
    ulong q2[2]; bool lv[2];
    _Pragma("clang loop unroll(full)")
    for (uint p = 0; p < 2; ++p) {
      const uint rrp = (uint)(r64 + 2 * lane + p), iip = lf_udiv(rrp, a.L);
      q2[p] = (ulong)iip * a.Lp + (rrp - iip * a.L);
      lv[p] = r64 + 2 * lane + p < a.pairs;
    }
    const bool pair2 = lv[0] && lv[1] && q2[1] == q2[0] + 1 && (q2[0] & 1) == 0;
    float K0 = lv[0] ? (float)prod[q2[0]] : 0.f;
    float K1 = lv[1] ? (float)prod[q2[1]] : 0.f;
    float s1_0 = 0.f, s2_0 = 0.f, s1_1 = 0.f, s2_1 = 0.f;
    if (pair2) {
      for (uint c = grp; c < C; c += 8) {
        half2 h = *(device const half2*)(prod + (ulong)c * plane + q2[0]);
        float d0 = (float)h.x - K0, d1 = (float)h.y - K1;
        s1_0 += d0; s2_0 += d0 * d0;
        s1_1 += d1; s2_1 += d1 * d1;
      }
    } else {
      for (uint c = grp; c < C; c += 8) {
        float d0 = (lv[0] ? (float)prod[(ulong)c * plane + q2[0]] : 0.f) - K0;
        float d1 = (lv[1] ? (float)prod[(ulong)c * plane + q2[1]] : 0.f) - K1;
        s1_0 += d0; s2_0 += d0 * d0;
        s1_1 += d1; s2_1 += d1 * d1;
      }
    }
    part[grp * 64 + 2 * lane] = s1_0; part[grp * 64 + 2 * lane + 1] = s1_1;
    part2[grp * 64 + 2 * lane] = s2_0; part2[grp * 64 + 2 * lane + 1] = s2_1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean[2], inv[2];
    _Pragma("clang loop unroll(full)")
    for (uint p = 0; p < 2; ++p) {
      float S1 = 0.f, S2 = 0.f;
      _Pragma("clang loop unroll(full)")
      for (int g = 0; g < 8; ++g) {
        S1 += part[g * 64 + 2 * lane + p];
        S2 += part2[g * 64 + 2 * lane + p];
      }
      float m1 = S1 / C;
      mean[p] = (p == 0 ? K0 : K1) + m1;
      inv[p] = rsqrt(max(S2 / C - m1 * m1, 0.f) + a.eps);
    }
    const uint swizL = (lane & 15) << 2;
    const uint nChunks = C >> 7;
    for (uint ch = 0; ch < nChunks; ++ch) {
      uint c0 = ch << 7;
      half4 out4[2][4];
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < 4; ++k) {
        uint cOff = (k << 5) + (grp << 2), cBase = c0 + cOff;
        float4 sc = *(device const float4*)(a.scale + cBase);
        float4 of = *(device const float4*)(a.offset + cBase);
        float4 p0v, p1v;
        if (pair2) {
          half2 h0 = *(device const half2*)(prod + (ulong)(cBase + 0) * plane + q2[0]);
          half2 h1 = *(device const half2*)(prod + (ulong)(cBase + 1) * plane + q2[0]);
          half2 h2 = *(device const half2*)(prod + (ulong)(cBase + 2) * plane + q2[0]);
          half2 h3 = *(device const half2*)(prod + (ulong)(cBase + 3) * plane + q2[0]);
          p0v = float4((float)h0.x, (float)h1.x, (float)h2.x, (float)h3.x);
          p1v = float4((float)h0.y, (float)h1.y, (float)h2.y, (float)h3.y);
        } else {
          p0v = lv[0] ? float4((float)prod[(ulong)(cBase + 0) * plane + q2[0]], (float)prod[(ulong)(cBase + 1) * plane + q2[0]],
                               (float)prod[(ulong)(cBase + 2) * plane + q2[0]], (float)prod[(ulong)(cBase + 3) * plane + q2[0]]) : float4(0.f);
          p1v = lv[1] ? float4((float)prod[(ulong)(cBase + 0) * plane + q2[1]], (float)prod[(ulong)(cBase + 1) * plane + q2[1]],
                               (float)prod[(ulong)(cBase + 2) * plane + q2[1]], (float)prod[(ulong)(cBase + 3) * plane + q2[1]]) : float4(0.f);
        }
        out4[0][k] = half4((p0v - mean[0]) * inv[0] * sc + of);
        out4[1][k] = half4((p1v - mean[1]) * inv[1] * sc + of);
      }
      _Pragma("clang loop unroll(full)")
      for (uint p = 0; p < 2; ++p) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; ++k) {
          uint cOff = (k << 5) + (grp << 2);
          *(threadgroup half4*)(T + lane * 128 + (cOff ^ swizL)) = out4[p][k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (uint row = grp; row < 32; row += 8) {
          ulong r = r64 + 2 * row + p;
          if (r >= a.pairs) break;
          const uint swizR = (row & 15) << 2;
          const uint c = 4 * lane;
          *(device half4*)(a.out + r * C + c0 + c) = *(threadgroup const half4*)(T + row * 128 + (c ^ swizR));
        }
      }
    }
    return;
  }
  // the statistics in ONE read of the product (the output is the second): sums shifted by the pair's channel 0, so the
  // variance is no difference of large numbers - mean = K + s1 / C, var = s2 / C - (s1 / C)^2
  const float K = live ? (float)prod[q] : 0.f;
  float s1 = 0.f, s2 = 0.f;
  for (uint c = grp; c < C; c += 8) { const float d = (live ? (float)prod[c * plane + q] : 0.f) - K; s1 += d; s2 += d * d; }
  part[grp * 32 + lane] = s1; part2[grp * 32 + lane] = s2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float S1 = 0.f, S2 = 0.f;
  for (int g = 0; g < 8; ++g) { S1 += part[g * 32 + lane]; S2 += part2[g * 32 + lane]; }
  const float m1 = S1 / C, mean = K + m1;
  const float inv = rsqrt(max(S2 / C - m1 * m1, 0.f) + a.eps);
  if (!live) return;
  for (uint c = grp; c < C; c += 8) a.out[(r0 + lane) * C + c] = (half)(((float)prod[c * plane + q] - mean) * inv * a.scale[c] + a.offset[c]);
}
template [[host_name("lf_center_norm_wide")]] kernel void lf_center_norm_wide_t<float>(constant CenterNormArgs&, uint3, uint3, uint);
template [[host_name("lf_center_norm_wideh")]] kernel void lf_center_norm_wide_t<half>(constant CenterNormArgs&, uint3, uint3, uint);
// ---------------------------------------------------------------- gated flash attention
// A threadgroup is 32 queries of one (row, head), four simdgroups of 8; keys in tiles of 16 staged in threadgroup memory
// - the smallest of each measured fastest (0.94 TFLOP/s at 16 queries and 32 keys, 0.75 at 32 queries, 0.64 at 64 keys,
// 1.22 here, on a 195-residue triangle attention: the kernel is bound by its registers, not its loads). Scores and P V on 8 x 8 simdgroup matrices (half in, float accumulated), the softmax online in the log2
// domain (the query carries scale log2e, the bias is log2-scaled already). A lane holds row sm of each 8 x 8 matrix,
// columns sn and sn + 1; a row's four lanes differ in lane bits 0 and 3.
#define LF_UNROLL _Pragma("clang loop unroll(full)")
template <int D, int KT = 16, int QR = 1, bool NT = false>
kernel void lf_attention(constant AttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int QB = 32 * QR, LD = D + 8, DF = D / 8, D8 = D / 8, KF = KT / 8;
  constexpr int SMEM = QB * LD > 2 * KT * LD ? QB * LD : 2 * KT * LD;
  threadgroup half smem[SMEM];
  threadgroup half* Qs = smem;
  threadgroup half* Ks = smem;
  threadgroup half* Vs = smem + KT * LD;
  const int b = tg.y, h = tg.z, q0 = tg.x * QB, n = a.n, W = a.heads * D;
  device const half* base = a.qkvg + (long)b * a.rowStride + h * D;
  const float qs = a.scale * M_LOG2E_F;
  // the queries (scaled, the query bias added), eight halves a load
  LF_UNROLL for (int e = tid; e < QB * D8; e += 128) {
    int qi = e / D8, d = (e - qi * D8) * 8, q = q0 + qi;
    half4 v0 = half4(0), v1 = half4(0);
    if (q < n) {
      device const half* src = base + (long)q * a.posStride + d;
      float4 f0 = float4(*(device const half4*)src), f1 = float4(*(device const half4*)(src + 4));
      if (a.qBias) { f0 += *(device const float4*)(a.qBias + h * D + d); f1 += *(device const float4*)(a.qBias + h * D + d + 4); }
      v0 = half4(f0 * qs); v1 = half4(f1 * qs);
    }
    *(threadgroup half4*)(Qs + qi * LD + d) = v0;
    *(threadgroup half4*)(Qs + qi * LD + d + 4) = v1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[QR][DF];
  LF_UNROLL for (int r = 0; r < QR; ++r)
    LF_UNROLL for (int dk = 0; dk < DF; ++dk) simdgroup_load(qf[r][dk], Qs + (sg * 8 * QR + r * 8) * LD + dk * 8, LD);
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  float m[QR], l[QR];
  LF_UNROLL for (int r = 0; r < QR; ++r) { m[r] = -1e30f; l[r] = 0.f; }
  simdgroup_float8x8 o[QR][DF];
  LF_UNROLL for (int r = 0; r < QR; ++r) LF_UNROLL for (int c = 0; c < DF; ++c) o[r][c] = simdgroup_float8x8(0);
  const long bq = (long)(a.r0 + b);
  device const half* brow[QR];
  LF_UNROLL for (int r = 0; r < QR; ++r) {
    const int q = min(q0 + (int)sg * 8 * QR + r * 8 + sm, n - 1);
    brow[r] = a.bias ? a.bias + ((long)h * n + q) * a.biasStride : nullptr;
  }
  const bool sgActive = q0 + (int)sg * 8 * QR < n;
  for (int k0 = 0; k0 < n; k0 += KT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (KT * D8 <= 64) {
      const int e = (int)tid & 63;
      if (e < KT * D8) {
        const int kj = e / D8, d = (e - kj * D8) * 8, k = k0 + kj;
        half4 v0 = half4(0), v1 = half4(0);
        if (k < n) {
          device const half* src = base + (long)k * a.posStride + (tid < 64 ? W : 2 * W) + d;
          v0 = *(device const half4*)src; v1 = *(device const half4*)(src + 4);
        }
        threadgroup half* dst = (tid < 64 ? Ks : Vs) + kj * LD + d;
        *(threadgroup half4*)dst = v0; *(threadgroup half4*)(dst + 4) = v1;
      }
    } else {
      LF_UNROLL for (int e = tid; e < KT * D8; e += 128) {
        int kj = e / D8, d = (e - kj * D8) * 8, k = k0 + kj;
        half4 k0v = half4(0), k1v = half4(0), v0 = half4(0), v1 = half4(0);
        if (k < n) {
          device const half* src = base + (long)k * a.posStride + d;
          k0v = *(device const half4*)(src + W); k1v = *(device const half4*)(src + W + 4);
          v0 = *(device const half4*)(src + 2 * W); v1 = *(device const half4*)(src + 2 * W + 4);
        }
        *(threadgroup half4*)(Ks + kj * LD + d) = k0v; *(threadgroup half4*)(Ks + kj * LD + d + 4) = k1v;
        *(threadgroup half4*)(Vs + kj * LD + d) = v0; *(threadgroup half4*)(Vs + kj * LD + d + 4) = v1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sgActive) continue;
    simdgroup_float8x8 s[QR][KF];
    LF_UNROLL for (int r = 0; r < QR; ++r) LF_UNROLL for (int j = 0; j < KF; ++j) s[r][j] = simdgroup_float8x8(0);
    LF_UNROLL for (int j = 0; j < KF; ++j)
      LF_UNROLL for (int dk = 0; dk < DF; ++dk) {
        simdgroup_half8x8 kt;
        simdgroup_load(kt, Ks + (j * 8) * LD + dk * 8, LD, ulong2(0, 0), true);
        LF_UNROLL for (int r = 0; r < QR; ++r) simdgroup_multiply_accumulate(s[r][j], qf[r][dk], kt, s[r][j]);
      }
    simdgroup_half8x8 p[QR][KF];
    if constexpr (NT) {
      LF_UNROLL for (int r = 0; r < QR; ++r) {
        float rowMax = -1e30f;
        LF_UNROLL for (int j = 0; j < KF; ++j) {
          thread auto& e = s[r][j].thread_elements();
          const float2 bv = float2(*(device const half2*)(brow[r] + k0 + j * 8 + sn));
          e[0] += bv.x; e[1] += bv.y;
          rowMax = max(rowMax, max(e[0], e[1]));
        }
        rowMax = max(rowMax, simd_shuffle_xor(rowMax, (ushort)1));
        rowMax = max(rowMax, simd_shuffle_xor(rowMax, (ushort)8));
        float mn = max(m[r], rowMax), corr = fast::exp2(m[r] - mn), sum = 0.f;
        LF_UNROLL for (int j = 0; j < KF; ++j) {
          thread auto& e = s[r][j].thread_elements();
          thread auto& pe = p[r][j].thread_elements();
          LF_UNROLL for (int t = 0; t < 2; ++t) { float pv = fast::exp2(e[t] - mn); sum += pv; pe[t] = (half)pv; }
        }
        sum += simd_shuffle_xor(sum, (ushort)1);
        sum += simd_shuffle_xor(sum, (ushort)8);
        l[r] = l[r] * corr + sum; m[r] = mn;
        LF_UNROLL for (int c = 0; c < DF; ++c) { thread auto& oe = o[r][c].thread_elements(); oe[0] *= corr; oe[1] *= corr; }
      }
    } else {
      // the bias, the mask and the keys past n; each row's new maximum
      const bool whole = k0 + KT <= n;
      float mk[KF][2];
      LF_UNROLL for (int j = 0; j < KF; ++j)
        LF_UNROLL for (int t = 0; t < 2; ++t) {
          int k = k0 + j * 8 + sn + t;
          float v = 0.f;
          if (!whole && k >= n) v = -1e30f;
          else if (a.mask) v = a.mask[bq * a.maskB + (long)k * a.maskK] > 0.f ? 0.f : -1e9f;
          mk[j][t] = v;
        }
      const bool bias2 = (a.biasStride & 1) == 0;     // (the bias's rows padded even: a lane's two in one load)
      LF_UNROLL for (int r = 0; r < QR; ++r) {
        float rowMax = -1e30f;
        LF_UNROLL for (int j = 0; j < KF; ++j) {
          thread auto& e = s[r][j].thread_elements();
          const int k = k0 + j * 8 + sn;
          float2 bv = float2(0.f);
          if (brow[r]) {
            if (bias2 && (whole || k + 1 < n)) bv = float2(*(device const half2*)(brow[r] + k));
            else { bv.x = whole || k < n ? (float)brow[r][k] : 0.f; bv.y = whole || k + 1 < n ? (float)brow[r][k + 1] : 0.f; }
          }
          e[0] += mk[j][0] + bv.x; e[1] += mk[j][1] + bv.y;
          rowMax = max(rowMax, max(e[0], e[1]));
        }
        rowMax = max(rowMax, simd_shuffle_xor(rowMax, (ushort)1));
        rowMax = max(rowMax, simd_shuffle_xor(rowMax, (ushort)8));
        float mn = max(m[r], rowMax), corr = fast::exp2(m[r] - mn), sum = 0.f;
        LF_UNROLL for (int j = 0; j < KF; ++j) {
          thread auto& e = s[r][j].thread_elements();
          thread auto& pe = p[r][j].thread_elements();
          LF_UNROLL for (int t = 0; t < 2; ++t) { float pv = fast::exp2(e[t] - mn); sum += pv; pe[t] = (half)pv; }
        }
        sum += simd_shuffle_xor(sum, (ushort)1);
        sum += simd_shuffle_xor(sum, (ushort)8);
        l[r] = l[r] * corr + sum; m[r] = mn;
        LF_UNROLL for (int c = 0; c < DF; ++c) { thread auto& oe = o[r][c].thread_elements(); oe[0] *= corr; oe[1] *= corr; }
      }
    }
    LF_UNROLL for (int j = 0; j < KF; ++j)
      LF_UNROLL for (int c = 0; c < DF; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, Vs + (j * 8) * LD + c * 8, LD);
        LF_UNROLL for (int r = 0; r < QR; ++r) simdgroup_multiply_accumulate(o[r][c], p[r][j], vf, o[r][c]);
      }
  }
  // out = O / l * sigmoid(gate)
  LF_UNROLL for (int r = 0; r < QR; ++r) {
    const int q = q0 + sg * 8 * QR + r * 8 + sm;
    if (q >= n) continue;
    device const half* g = base + (long)q * a.posStride + 3 * W;
    device half* out = a.out + (long)b * a.outRowStride + (long)q * a.outPosStride + h * D;
    const float inv = 1.f / l[r];
    LF_UNROLL for (int c = 0; c < DF; ++c) {
      thread auto& oe = o[r][c].thread_elements();
      const int d = c * 8 + sn;
      half2 gv = *(device const half2*)(g + d);
      *(device half2*)(out + d) = half2(half(oe[0] * inv * lf_sigmoid((float)gv.x)), half(oe[1] * inv * lf_sigmoid((float)gv.y)));
    }
  }
}
template [[host_name("lf_attention_8")]] kernel void lf_attention<8>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_16")]] kernel void lf_attention<16>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_24")]] kernel void lf_attention<24>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_32")]] kernel void lf_attention<32>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_48")]] kernel void lf_attention<48>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_64")]] kernel void lf_attention<64>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_nt_8")]] kernel void lf_attention<8, 16, 1, true>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_nt_16")]] kernel void lf_attention<16, 16, 1, true>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_nt_24")]] kernel void lf_attention<24, 16, 1, true>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_nt_32")]] kernel void lf_attention<32, 16, 1, true>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_nt_48")]] kernel void lf_attention<48, 16, 1, true>(constant AttnArgs&, uint3, uint, uint, uint);
template [[host_name("lf_attention_nt_64")]] kernel void lf_attention<64, 16, 1, true>(constant AttnArgs&, uint3, uint, uint, uint);
