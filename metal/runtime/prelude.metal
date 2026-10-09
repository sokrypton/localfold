// CUDA's device vocabulary in Metal Shading Language: what metal/tools/cu2metal.py's translation of the CUDA
// ports compiles against. Every name here is CUDA's, so the translated kernels read as the originals do.
//
// Threads: a kernel's builtins live in a context (`_c`), built at kernel entry from Metal's attributes and handed
// to the device functions that need it, so `threadIdx.x` reads the same in both. A simdgroup is 32 lanes on every
// Apple GPU, a CUDA warp's width, and threads are assigned to simdgroups in linear thread order, as warps are.
#pragma once
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_atomic>
using namespace metal;

// ---------------------------------------------------------------- threads
struct LfCtx {
  uint3 tid, bid, bdim, gdim;
  uint lane, warp, tidx;
  threadgroup char* smem;
};
#define LF_KERNEL_BUILTINS                                                                       \
  uint3 _lf_bid [[threadgroup_position_in_grid]], uint3 _lf_tid [[thread_position_in_threadgroup]], \
  uint3 _lf_bdim [[threads_per_threadgroup]], uint3 _lf_gdim [[threadgroups_per_grid]],            \
  uint _lf_lane [[thread_index_in_simdgroup]], uint _lf_warp [[simdgroup_index_in_threadgroup]],   \
  uint _lf_tidx [[thread_index_in_threadgroup]]
#define LF_KERNEL_CTX const LfCtx _c = {_lf_tid, _lf_bid, _lf_bdim, _lf_gdim, _lf_lane, _lf_warp, _lf_tidx, _lf_smem};
#define threadIdx (_c.tid)
#define blockIdx (_c.bid)
#define blockDim (_c.bdim)
#define gridDim (_c.gdim)
#define warpSize 32

#define __syncthreads() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define __syncwarp(...) simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define __threadfence() atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)
#define __threadfence_block() threadgroup_barrier(mem_flags::mem_threadgroup)
#define __activemask() 0xffffffffu
#define __ldg(p) (*(p))
#define __launch_bounds__(...)
#define __forceinline__
#define __noinline__
#define __restrict__
#define assert(x)
#define lf_printf(...) ((void)0)

// ---------------------------------------------------------------- warp shuffles
// (the mask is ignored: every shuffle here is over a full, converged warp, as in the CUDA ports)
// a 32-bit unsigned division by a runtime divisor, as a float reciprocal and one correction: Apple's GPUs have no integer
// divider - `a / d` is a software routine, and two of them an element made AF2's opmAddK 5.8x its bandwidth. Exact for
// a < 2^24 (the float estimate is then within one either way); past it the plain division.
inline uint lf_udiv(uint a, uint d) {
  if (a >= (1u << 24)) return a / d;
  uint q = (uint)((float)a * (1.f / (float)d));
  if (q * d > a) --q;
  else if ((q + 1) * d <= a) ++q;
  return q;
}
template <typename T> inline T lf_shfl_xor(T v, int m) { return simd_shuffle_xor(v, (ushort)m); }
template <typename T> inline T lf_shfl(T v, int src) { return simd_shuffle(v, (ushort)(src & 31)); }
template <typename T> inline T lf_shfl_down(T v, int d) { return simd_shuffle_down(v, (ushort)d); }
template <typename T> inline T lf_shfl_up(T v, int d) { return simd_shuffle_up(v, (ushort)d); }
// with a width under 32: the shuffle stays inside each width-lane segment, a lane past its end keeps its value
template <typename T> inline T lf_shfl_w(thread const LfCtx& c, T v, int src, int w) {
  return simd_shuffle(v, (ushort)((c.lane & ~(uint)(w - 1)) + ((uint)src & (uint)(w - 1))));
}
template <typename T> inline T lf_shfl_down_w(thread const LfCtx& c, T v, int d, int w) {
  T o = simd_shuffle_down(v, (ushort)d);
  return ((c.lane & (uint)(w - 1)) + (uint)d < (uint)w) ? o : v;
}
template <typename T> inline T lf_shfl_up_w(thread const LfCtx& c, T v, int d, int w) {
  T o = simd_shuffle_up(v, (ushort)d);
  return ((c.lane & (uint)(w - 1)) >= (uint)d) ? o : v;
}
template <typename T> inline T lf_shfl_xor_w(thread const LfCtx& c, T v, int m, int w) { return simd_shuffle_xor(v, (ushort)m); }
#define LF_SEL4(_1, _2, _3, _4, NAME, ...) NAME
#define __shfl_xor_sync(...) LF_SEL4(__VA_ARGS__, LF_XOR4, LF_XOR3, _)(__VA_ARGS__)
#define LF_XOR3(m, v, l) lf_shfl_xor(v, l)
#define LF_XOR4(m, v, l, w) lf_shfl_xor(v, l)
#define __shfl_sync(...) LF_SEL4(__VA_ARGS__, LF_SH4, LF_SH3, _)(__VA_ARGS__)
#define LF_SH3(m, v, s) lf_shfl(v, s)
#define LF_SH4(m, v, s, w) lf_shfl_w(_c, v, s, w)
#define __shfl_down_sync(...) LF_SEL4(__VA_ARGS__, LF_DN4, LF_DN3, _)(__VA_ARGS__)
#define LF_DN3(m, v, d) lf_shfl_down(v, d)
#define LF_DN4(m, v, d, w) lf_shfl_down_w(_c, v, d, w)
#define __shfl_up_sync(...) LF_SEL4(__VA_ARGS__, LF_UP4, LF_UP3, _)(__VA_ARGS__)
#define LF_UP3(m, v, d) lf_shfl_up(v, d)
#define LF_UP4(m, v, d, w) lf_shfl_up_w(_c, v, d, w)
#define __ballot_sync(m, p) ((uint)(ulong)simd_ballot((bool)(p)))
#define __any_sync(m, p) simd_any((bool)(p))
#define __all_sync(m, p) simd_all((bool)(p))
#define __popc(x) popcount((uint)(x))
#define __popcll(x) popcount((ulong)(x))
#define __clz(x) clz((int)(x))
#define __ffs(x) ((x) == 0 ? 0 : (int)ctz((uint)(x)) + 1)
#define __brev(x) reverse_bits((uint)(x))

// ---------------------------------------------------------------- types
#define FLT_MAX MAXFLOAT
#define SIZE_MAX 0xffffffffffffffffUL
#define FLT_MIN 1.17549435e-38f
#define FLT_EPSILON 1.1920928955078125e-07f
#define CUDART_INF_F INFINITY
#define M_PI_F_ 3.14159265358979323846f

// float3 in device memory is CUDA's 12 bytes
#define make_float2(x, y) float2((x), (y))
#define make_float3(x, y, z) float3((x), (y), (z))
#define make_float4(x, y, z, w) float4((x), (y), (z), (w))
#define make_int2(x, y) int2((x), (y))
#define make_int3(x, y, z) int3((x), (y), (z))
#define make_int4(x, y, z, w) int4((x), (y), (z), (w))
#define make_uint2(x, y) uint2((x), (y))
#define make_uint4(x, y, z, w) uint4((x), (y), (z), (w))
#define make_half2(x, y) half2((x), (y))

// A double the HOST reads or writes (a kernel argument, a buffer): CUDA's 8 bytes, kept as the double's bits.
// Metal has no double arithmetic, so a kernel computes in float and converts at the edge (exact from float).
struct alignas(8) lf_f64 {   // (8-aligned: a host double in an argument struct sits on 8 bytes)
  uint lo, hi;
  lf_f64() = default;
  lf_f64(float f) { set(f); }
  void set(float f) thread {
    uint b = as_type<uint>(f), s = b >> 31, e = (b >> 23) & 0xff, m = b & 0x7fffff;
    if (e == 0) { hi = s << 31; lo = 0; return; }               // (a float denormal or zero: zero)
    if (e == 0xff) { hi = (s << 31) | 0x7ff00000u | (m >> 3); lo = m << 29; return; }
    hi = (s << 31) | ((e + 896u) << 20) | (m >> 3); lo = m << 29;
  }
  float get() const thread {
    uint s = hi >> 31, e = (hi >> 20) & 0x7ff, m = ((hi & 0xfffff) << 3) | (lo >> 29);
    if (e == 0) return s ? -0.0f : 0.0f;
    if (e == 0x7ff) return as_type<float>((s << 31) | 0x7f800000u | m);
    int fe = (int)e - 896;
    if (fe <= 0) return s ? -0.0f : 0.0f;
    if (fe >= 255) return s ? -INFINITY : INFINITY;
    return as_type<float>((s << 31) | ((uint)fe << 23) | m);
  }
  float get() const device { lf_f64 t = *this; return t.get(); }
  float get() const constant { lf_f64 t = *this; return t.get(); }
  operator float() const thread { return get(); }
  operator float() const device { return get(); }
  operator float() const constant { return get(); }
  void operator=(float f) device { lf_f64 t; t.set(f); lo = t.lo; hi = t.hi; }
  void operator=(float f) thread { set(f); }
  void operator+=(float f) device { float v = get() + f; lf_f64 t; t.set(v); lo = t.lo; hi = t.hi; }
  void operator+=(float f) thread { set(get() + f); }
};

// ---------------------------------------------------------------- half
#define __float2half(x) half(x)
#define __float2half_rn(x) half(x)
#define __float2half_rz(x) half(x)
#define __half2float(x) float(x)
#define __float2half2_rn(x) half2(half(x))
#define __floats2half2_rn(a, b) half2(half(a), half(b))
#define __float22half2_rn(v) half2(v)
#define __half22float2(v) float2(v)
#define __halves2half2(a, b) half2((a), (b))
#define __half2half2(a) half2(a)
#define __low2half(v) ((v).x)
#define __high2half(v) ((v).y)
#define __low2float(v) float((v).x)
#define __high2float(v) float((v).y)
#define __lows2half2(a, b) half2((a).x, (b).x)
#define __highs2half2(a, b) half2((a).y, (b).y)
#define __hadd(a, b) ((a) + (b))
#define __hsub(a, b) ((a) - (b))
#define __hmul(a, b) ((a) * (b))
#define __hdiv(a, b) ((a) / (b))
#define __hfma(a, b, c) fma((a), (b), (c))
#define __hneg(a) (-(a))
#define __hadd2(a, b) ((a) + (b))
#define __hsub2(a, b) ((a) - (b))
#define __hmul2(a, b) ((a) * (b))
#define __h2div(a, b) ((a) / (b))
#define __hfma2(a, b, c) fma((a), (b), (c))
#define __hmax(a, b) max((a), (b))
#define __hmin(a, b) min((a), (b))
#define __hmax2(a, b) max((a), (b))
#define __hmin2(a, b) min((a), (b))
#define __hgt(a, b) ((a) > (b))
#define __hlt(a, b) ((a) < (b))
#define __hge(a, b) ((a) >= (b))
#define __hle(a, b) ((a) <= (b))
#define __heq(a, b) ((a) == (b))
#define __hisnan(a) isnan(a)
#define hexp(a) exp(a)
#define h2exp(a) exp(a)
#define hexp2(a) exp2(a)
#define h2exp2(a) exp2(a)
#define hsqrt(a) sqrt(a)
#define hrsqrt(a) rsqrt(a)
#define __half_as_ushort(h) as_type<ushort>(h)
#define __ushort_as_half(u) as_type<half>((ushort)(u))
#define __half_as_short(h) as_type<short>(h)
#define __short_as_half(u) as_type<half>((short)(u))
#define __half2_as_uint(h) as_type<uint>(h)
#define __uint_as_half2(u) as_type<half2>((uint)(u))

// ---------------------------------------------------------------- bfloat16, as storage (macOS 13 has no bfloat type)
struct lf_bf16 {
  ushort v;
  lf_bf16() = default;
  lf_bf16(float f) { v = from(f); }
  static ushort from(float f) {
    uint b = as_type<uint>(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return (ushort)((b >> 16) | 0x40);   // NaN stays NaN
    return (ushort)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);                      // round to nearest even
  }
  float get() const thread { return as_type<float>((uint)v << 16); }
  float get() const device { return as_type<float>((uint)v << 16); }
  float get() const threadgroup { return as_type<float>((uint)v << 16); }
  float get() const constant { return as_type<float>((uint)v << 16); }
  operator float() const thread { return get(); }
  operator float() const device { return get(); }
  operator float() const threadgroup { return get(); }
  void operator=(float f) device { v = from(f); }
  void operator=(float f) threadgroup { v = from(f); }
  void operator=(float f) thread { v = from(f); }
};
struct lf_bf162 {
  lf_bf16 x, y;
};
inline lf_bf16 lf_make_bf16(float f) { lf_bf16 r; r.v = lf_bf16::from(f); return r; }
#define __float2bfloat16(f) lf_make_bf16((float)(f))
#define __float2bfloat16_rn(f) lf_make_bf16((float)(f))
#define __bfloat162float(b) ((b).get())
inline lf_bf162 lf_make_bf162(float a, float b) { lf_bf162 r; r.x = lf_make_bf16(a); r.y = lf_make_bf16(b); return r; }
#define __floats2bfloat162_rn(a, b) lf_make_bf162((a), (b))
#define __float22bfloat162_rn(v) lf_make_bf162((v).x, (v).y)
#define __bfloat1622float2(v) float2((v).x.get(), (v).y.get())
#define __bfloat16_as_ushort(b) ((b).v)
inline lf_bf16 lf_ushort_as_bf16(ushort u) { lf_bf16 r; r.v = u; return r; }
#define __ushort_as_bfloat16(u) lf_ushort_as_bf16((ushort)(u))

// ---------------------------------------------------------------- a pointer's element type, whatever its address space
// (an overloaded device function whose overloads differ in a pointer's element type takes a generic pointer
// constrained to that element: LF_IS)
template <typename P> struct lf_elem { typedef void type; };
template <typename T> struct lf_elem<device T*> { typedef metal::remove_cv_t<T> type; };
template <typename T> struct lf_elem<threadgroup T*> { typedef metal::remove_cv_t<T> type; };
template <typename T> struct lf_elem<thread T*> { typedef metal::remove_cv_t<T> type; };
template <typename T> struct lf_elem<constant T*> { typedef metal::remove_cv_t<T> type; };
template <typename T> struct lf_elem<const device T*> { typedef metal::remove_cv_t<T> type; };
template <typename T> struct lf_elem<const threadgroup T*> { typedef metal::remove_cv_t<T> type; };
template <typename T> struct lf_elem<const thread T*> { typedef metal::remove_cv_t<T> type; };
#define LF_IS(P, T) metal::enable_if_t<metal::is_same<typename lf_elem<P>::type, metal::remove_cv_t<T>>::value, int> = 0

// ---------------------------------------------------------------- casts that keep an address space
template <typename T, typename P> inline device T* lf_rcast(device P* p) { return reinterpret_cast<device T*>(p); }
template <typename T, typename P> inline threadgroup T* lf_rcast(threadgroup P* p) { return reinterpret_cast<threadgroup T*>(p); }
template <typename T, typename P> inline thread T* lf_rcast(thread P* p) { return reinterpret_cast<thread T*>(p); }
template <typename T, typename P> inline constant T* lf_rcast(constant P* p) { return reinterpret_cast<constant T*>(p); }
template <typename T, typename P> inline const device T* lf_rcast(const device P* p) { return reinterpret_cast<const device T*>(p); }
template <typename T, typename P> inline const threadgroup T* lf_rcast(const threadgroup P* p) { return reinterpret_cast<const threadgroup T*>(p); }
template <typename T, typename P> inline const thread T* lf_rcast(const thread P* p) { return reinterpret_cast<const thread T*>(p); }

// a value's bits as another type of the same size (what a CUDA cast of an element's address does)
template <typename T, typename V> inline T lf_reinterp(V v) { thread V t = v; return *reinterpret_cast<thread T*>(&t); }

// ---------------------------------------------------------------- bit reinterpretation
#define __float_as_int(x) as_type<int>((float)(x))
#define __float_as_uint(x) as_type<uint>((float)(x))
#define __int_as_float(x) as_type<float>((int)(x))
#define __uint_as_float(x) as_type<float>((uint)(x))

// ---------------------------------------------------------------- math (CUDA's float names; doubles are float here)
#define expf(x) exp((float)(x))
#define exp2f(x) exp2((float)(x))
#define exp10f(x) exp10((float)(x))
#define logf(x) log((float)(x))
#define log2f(x) log2((float)(x))
#define log10f(x) log10((float)(x))
#define sqrtf(x) sqrt((float)(x))
#define rsqrtf(x) rsqrt((float)(x))
#define cbrtf(x) cbrt((float)(x))
#define sinf(x) sin((float)(x))
#define cosf(x) cos((float)(x))
#define tanf(x) tan((float)(x))
#define asinf(x) asin((float)(x))
#define acosf(x) acos((float)(x))
#define atanf(x) atan((float)(x))
#define atan2f(y, x) atan2((float)(y), (float)(x))
#define sinhf(x) sinh((float)(x))
#define coshf(x) cosh((float)(x))
#define tanhf(x) tanh((float)(x))
#define fabsf(x) fabs((float)(x))
#define floorf(x) floor((float)(x))
#define ceilf(x) ceil((float)(x))
#define roundf(x) round((float)(x))
#define rintf(x) rint((float)(x))
#define nearbyintf(x) rint((float)(x))
#define truncf(x) trunc((float)(x))
#define fmodf(x, y) fmod((float)(x), (float)(y))
#define fmaxf(x, y) fmax((float)(x), (float)(y))
#define fminf(x, y) fmin((float)(x), (float)(y))
#define fmaf(x, y, z) fma((float)(x), (float)(y), (float)(z))
#define copysignf(x, y) copysign((float)(x), (float)(y))
#define isnanf(x) isnan(x)
#define __expf(x) fast::exp((float)(x))
#define __exp10f(x) fast::exp10((float)(x))
#define __logf(x) fast::log((float)(x))
#define __log2f(x) fast::log2((float)(x))
#define __sinf(x) fast::sin((float)(x))
#define __cosf(x) fast::cos((float)(x))
#define __tanf(x) fast::tan((float)(x))
#define __powf(x, y) fast::pow((float)(x), (float)(y))
#define __fdividef(x, y) ((float)(x) / (float)(y))
#define __frcp_rn(x) (1.0f / (float)(x))
#define __fsqrt_rn(x) sqrt((float)(x))
#define __frsqrt_rn(x) rsqrt((float)(x))
#define __saturatef(x) saturate((float)(x))
#define __fmul_rn(a, b) ((float)(a) * (float)(b))
#define __fadd_rn(a, b) ((float)(a) + (float)(b))
#define __fsub_rn(a, b) ((float)(a) - (float)(b))
#define __fmaf_rn(a, b, c) fma((float)(a), (float)(b), (float)(c))
#define __fdiv_rn(a, b) ((float)(a) / (float)(b))
#define __dmul_rn(a, b) ((float)(a) * (float)(b))
#define __dadd_rn(a, b) ((float)(a) + (float)(b))
#define __dsub_rn(a, b) ((float)(a) - (float)(b))
#define __fma_rn(a, b, c) fma((float)(a), (float)(b), (float)(c))
#define __float2int_rn(x) ((int)rint((float)(x)))
#define __float2int_rz(x) ((int)(x))
#define __float2int_rd(x) ((int)floor((float)(x)))
#define __float2int_ru(x) ((int)ceil((float)(x)))
#define __float2uint_rn(x) ((uint)rint((float)(x)))
#define __int2float_rn(x) ((float)(x))
#define __double2float_rn(x) ((float)(x))
#define sincosf(x, s, c) (*(s) = sincos((float)(x), *(c)))
#define __sincosf(x, s, c) (*(s) = sincos((float)(x), *(c)))
// powf: C's, defined for a negative base and an integer exponent, where Metal's pow is not
inline float lf_pow(float x, float y) {
  if (x >= 0.f) return pow(x, y);
  float r = pow(-x, y);
  return (fmod(y, 2.f) != 0.f) ? -r : r;
}
#define powf(x, y) lf_pow((float)(x), (float)(y))
// erf/erfc (Metal has neither): W. J. Cody's rational approximations as in most libms, ~1e-7 relative
inline float lf_erf(float x) {
  float ax = fabs(x);
  if (ax < 0.5f) {
    float t = x * x;
    float top = (((0.0185777706f * t + 0.1857777061f) * t + 1.1283791671f));
    // (a short series: erf(x) = 2/sqrt(pi) (x - x^3/3 + x^5/10 - x^7/42 + x^9/216 ...))
    float s = x * (1.1283791671f + t * (-0.3761263890f + t * (0.1128379167f + t * (-0.0268661706f + t * (0.0052239776f + t * (-0.0008548327f))))));
    return s + 0.f * top;
  }
  // Abramowitz-Stegun 7.1.26 refined: erfc via exp, |err| < 1.2e-7
  float t = 1.0f / (1.0f + 0.5f * ax);
  float y = t * exp(-ax * ax - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
            t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
            t * (-0.82215223f + t * 0.17087277f)))))))));
  float r = 1.0f - y;
  return x >= 0 ? r : -r;
}
inline float lf_erfc(float x) {
  float ax = fabs(x);
  if (ax < 0.5f) return 1.0f - lf_erf(x);
  float t = 1.0f / (1.0f + 0.5f * ax);
  float y = t * exp(-ax * ax - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
            t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
            t * (-0.82215223f + t * 0.17087277f)))))))));
  return x >= 0 ? y : 2.0f - y;
}
#define erff(x) lf_erf((float)(x))
#define erfcf(x) lf_erfc((float)(x))
#define erf(x) lf_erf((float)(x))
#define erfc(x) lf_erfc((float)(x))
#define expm1f(x) (exp((float)(x)) - 1.0f)
#define log1pf(x) log(1.0f + (float)(x))
#define asinhf(x) asinh((float)(x))
#define acoshf(x) acosh((float)(x))
#define atanhf(x) atanh((float)(x))
#define cospif(x) cospi((float)(x))
#define sinpif(x) sinpi((float)(x))
#define tanpif(x) tanpi((float)(x))
template <typename S, typename C> inline void lf_sincos(float x, S s, C c) { float cc; *s = sincos(x, cc); *c = cc; }
#define hypotf(a, b) sqrt((float)(a) * (float)(a) + (float)(b) * (float)(b))
#define norm3df(a, b, c) sqrt((float)(a) * (float)(a) + (float)(b) * (float)(b) + (float)(c) * (float)(c))

// ---------------------------------------------------------------- atomics
inline float atomicAdd(device float* p, float v) { return atomic_fetch_add_explicit((device atomic_float*)p, v, memory_order_relaxed); }
inline float atomicAdd(threadgroup float* p, float v) {   // (no threadgroup atomic_float: a compare-and-swap on its bits)
  threadgroup atomic_uint* a = (threadgroup atomic_uint*)p;
  uint old = atomic_load_explicit(a, memory_order_relaxed);
  while (!atomic_compare_exchange_weak_explicit(a, &old, as_type<uint>(as_type<float>(old) + v), memory_order_relaxed, memory_order_relaxed)) {}
  return as_type<float>(old);
}
inline int atomicAdd(device int* p, int v) { return atomic_fetch_add_explicit((device atomic_int*)p, v, memory_order_relaxed); }
inline int atomicAdd(threadgroup int* p, int v) { return atomic_fetch_add_explicit((threadgroup atomic_int*)p, v, memory_order_relaxed); }
inline uint atomicAdd(device uint* p, uint v) { return atomic_fetch_add_explicit((device atomic_uint*)p, v, memory_order_relaxed); }
inline uint atomicAdd(threadgroup uint* p, uint v) { return atomic_fetch_add_explicit((threadgroup atomic_uint*)p, v, memory_order_relaxed); }
inline int atomicMax(device int* p, int v) { return atomic_fetch_max_explicit((device atomic_int*)p, v, memory_order_relaxed); }
inline uint atomicMax(device uint* p, uint v) { return atomic_fetch_max_explicit((device atomic_uint*)p, v, memory_order_relaxed); }
inline int atomicMin(device int* p, int v) { return atomic_fetch_min_explicit((device atomic_int*)p, v, memory_order_relaxed); }
inline uint atomicMin(device uint* p, uint v) { return atomic_fetch_min_explicit((device atomic_uint*)p, v, memory_order_relaxed); }
inline int atomicMax(threadgroup int* p, int v) { return atomic_fetch_max_explicit((threadgroup atomic_int*)p, v, memory_order_relaxed); }
inline int atomicMin(threadgroup int* p, int v) { return atomic_fetch_min_explicit((threadgroup atomic_int*)p, v, memory_order_relaxed); }
inline int atomicExch(device int* p, int v) { return atomic_exchange_explicit((device atomic_int*)p, v, memory_order_relaxed); }
inline uint atomicExch(device uint* p, uint v) { return atomic_exchange_explicit((device atomic_uint*)p, v, memory_order_relaxed); }
inline float atomicExch(device float* p, float v) { return atomic_exchange_explicit((device atomic_float*)p, v, memory_order_relaxed); }
inline int atomicCAS(device int* p, int cmp, int v) {
  atomic_compare_exchange_weak_explicit((device atomic_int*)p, &cmp, v, memory_order_relaxed, memory_order_relaxed); return cmp;
}
inline uint atomicCAS(device uint* p, uint cmp, uint v) {
  atomic_compare_exchange_weak_explicit((device atomic_uint*)p, &cmp, v, memory_order_relaxed, memory_order_relaxed); return cmp;
}
// a float max through the int ordering (non-negative floats order as their bits)
inline float atomicMax(device float* p, float v) {
  device atomic_uint* a = (device atomic_uint*)p;
  uint old = atomic_load_explicit(a, memory_order_relaxed);
  while (as_type<float>(old) < v && !atomic_compare_exchange_weak_explicit(a, &old, as_type<uint>(v), memory_order_relaxed, memory_order_relaxed)) {}
  return as_type<float>(old);
}
// a double accumulator (lf_f64): Apple GPUs have no 64-bit atomic add, so the add is a compare-and-swap on the HIGH
// word alone - the double's sign, exponent and top 20 mantissa bits, ~6 significant digits - the low word kept zero
inline float atomicAdd(device lf_f64* p, float v) {
  device atomic_uint* hi = (device atomic_uint*)&p->hi;
  uint old = atomic_load_explicit(hi, memory_order_relaxed);
  for (;;) {
    lf_f64 cur; cur.hi = old; cur.lo = 0;
    lf_f64 next; next.set(cur.get() + v);
    if (atomic_compare_exchange_weak_explicit(hi, &old, next.hi, memory_order_relaxed, memory_order_relaxed)) {
      p->lo = 0; return cur.get();
    }
  }
}

// ---------------------------------------------------------------- misc
// memcpy (type punning in the ports' decoders): byte by byte, in whatever address spaces the two sides are in
template <typename D, typename S> inline void memcpy(D d, S s, ulong n) {
  auto dd = lf_rcast<uchar>(d); auto ss = lf_rcast<const uchar>(s);
  for (ulong i = 0; i < n; ++i) dd[i] = ss[i];
}
template <typename T> inline T lf_min(T a, T b) { return a < b ? a : b; }
template <typename T> inline T lf_max(T a, T b) { return a > b ? a : b; }
#define __umulhi(a, b) mulhi((uint)(a), (uint)(b))
#define __mul24(a, b) ((int)(a) * (int)(b))
#define __umul24(a, b) ((uint)(a) * (uint)(b))
