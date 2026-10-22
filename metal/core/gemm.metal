// metal/core's GEMM: C = alpha op(A) op(B) (+ beta C) (+ bias) (relu / gelu), and its fused epilogues - the gated
// residual, SwiGLU in blocks of 8 and the triangle's gate. Column-major in cuBLAS's sense; core.mm's gemm() is the
// row-major view (A the weight, B the activations). Instantiated by name on first use (core.mm, gemmRun).
// (args.h is prepended by metal/build.sh)
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
inline float lf_erf(float x) {     // (Abramowitz and Stegun 7.1.26, |error| < 1.5e-7: Metal has no erf)
  float s = sign(x), t = 1.f / (1.f + 0.3275911f * fabs(x));
  return s * (1.f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x));
}
inline uint lf_udiv(uint a, uint d) {
  if (a >= (1u << 24)) return a / d;
  uint q = (uint)((float)a * (1.f / (float)d));
  if (q * d > a) --q;
  else if ((q + 1) * d <= a) ++q;
  return q;
}
inline float lf_ld(device const float* p, ulong i) { return p[i]; }
inline float lf_ld(device const half* p, ulong i) { return (float)p[i]; }
inline void lf_st(device float* p, ulong i, float v) { p[i] = v; }
inline void lf_st(device half* p, ulong i, float v) { p[i] = (half)v; }

// Tiles: BM x BN of C a threadgroup (four simdgroups, 2 x 2), BK of k a step. A and B are staged in threadgroup
// memory in the type the multiply takes (half for f16 inputs, float otherwise), along
// whichever axis is contiguous in memory - 16 bytes a thread where VEC says every row start is aligned - and read
// back as 8 x 8 matrices, transposed where staging was along the other axis. TA/TB: the operand is transposed.
template <typename T> struct lf_tile { typedef float type; };
template <> struct lf_tile<half> { typedef half type; };
template <typename TA, typename TB> struct lf_mma { typedef float type; };
template <> struct lf_mma<half, half> { typedef half type; };
inline float lf_ldf(device const float* p) { return *p; }
inline float lf_ldf(device const half* p) { return (float)*p; }
template <typename S, typename T> inline void lf_stage8(threadgroup T* dst, device const S* src) {
  _Pragma("clang loop unroll(full)")
  for (int e = 0; e < 8; ++e) dst[e] = (T)lf_ldf(src + e);
}
template <> inline void lf_stage8<half, half>(threadgroup half* dst, device const half* src) {
  *(threadgroup half4*)dst = *(device const half4*)src;
  *(threadgroup half4*)(dst + 4) = *(device const half4*)(src + 4);
}
template <> inline void lf_stage8<float, half>(threadgroup half* dst, device const float* src) {
  *(threadgroup half4*)dst = half4(*(device const float4*)src);
  *(threadgroup half4*)(dst + 4) = half4(*(device const float4*)(src + 4));
}
template <> inline void lf_stage8<float, float>(threadgroup float* dst, device const float* src) {
  *(threadgroup float4*)dst = *(device const float4*)src;
  *(threadgroup float4*)(dst + 4) = *(device const float4*)(src + 4);
}
// Computed as C^T = X W, X = op(B)^T (n x k), W = op(A)^T (k x m): cuBLAS's column-major non-transposed product
// is then row-major activations times row-major weights, both staged along their contiguous axis and loaded as
// 8 x 8 matrices WITHOUT the transposing load (a transposed operand stages the other way and takes it). A tile is
// TR rows of X (n) by TC columns of W (m); each lane's two output elements are adjacent in C.
template <typename TA, typename TB, typename TC, int TR, int TC_, bool TRA, bool TRB, int BK = 32, int EP = 0>
kernel void lf_gemm(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  // (the multiply's type: half where both operands are, or where EP bit 32 asks - an f32 operand rounded as it is staged)
  typedef metal::conditional_t<(EP & 32) != 0, half, typename lf_mma<TA, TB>::type> T;
  constexpr int PAD = 8;
  const uint z = grp.z;
  device const TA* A; device const TB* B; device TC* D; device const TC* C;
  if (g.ptrs) {
    // (the arrays themselves are in device memory: the address space goes on the OUTER pointer)
    A = (device const TA*)((device const ulong*)g.A)[z];
    B = (device const TB*)((device const ulong*)g.B)[z];
    D = (device TC*)((device const ulong*)g.D)[z];
    C = (device const TC*)D;
  } else {
    A = (device const TA*)g.A + (long)z * g.sa;
    B = (device const TB*)g.B + (long)z * g.sb;
    D = (device TC*)g.D + (long)z * g.sd;
    C = (device const TC*)g.C + (long)z * g.sc;
  }
  const int j0 = grp.y * TR, i0 = grp.x * TC_;      // (grid x over m, y over n)
  // X tile: [j][k] when B is not transposed (contiguous k), else [k][j]; W tile: [k][i] when A is not transposed
  // (contiguous i), else [i][k]
  constexpr int XROW = TRB ? TR + PAD : BK + PAD, WROW = TRA ? BK + PAD : TC_ + PAD;
  // (EP 1's gated a and b, staged channel-major for the store, reuse the tiles' memory once the k loop is done: in a
  // buffer of their own they cost a 64 x 64 tile its occupancy)
  constexpr int XSZ = (TRB ? BK : TR) * XROW, WSZ = (TRA ? TC_ : BK) * WROW;
  constexpr int SABSZ = (EP & 7) == 1 ? (2 * (TC_ / 4) * TR * 2 + sizeof(T) - 1) / sizeof(T) : 0;
  threadgroup T tgm[XSZ + WSZ > SABSZ ? XSZ + WSZ : SABSZ];
  threadgroup T* Xs = tgm;
  threadgroup T* Ws = tgm + XSZ;
  threadgroup half* Sab = (threadgroup half*)tgm;
  constexpr int WR = TR / 2, WC = TC_ / 2, FR = WR / 8, FC = WC / 8;
  const int sr = (sg / 2) * WR, sc = (sg % 2) * WC;
  // (EP bit 8: accumulated in half - the all-half GEMMs, lfcuda.mm's gemm)
  typedef metal::conditional_t<(EP & 8) != 0, simdgroup_half8x8, simdgroup_float8x8> ACC;
  ACC acc[FR][FC];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) acc[a][b] = ACC(0);
  // (vector staging: the host's word, or for a pointer array - whose pointers the host cannot see - the leading dimensions'
  // (bit 512) and this entry's own pointers' alignment)
  const bool vec = (g.biasType & 256) || ((g.biasType & 512) && ((ulong)A % (8 * sizeof(TA))) == 0 && ((ulong)B % (8 * sizeof(TB))) == 0);
  // a whole tile (the common case): staged without a bounds check an element (6% of a K-256 projection's time)
  const bool whole = vec && j0 + TR <= g.n && i0 + TC_ <= g.m && g.k % BK == 0;
  for (int k0 = 0; k0 < g.k; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (whole) {
      constexpr int XO = TRB ? BK : TR, XI = TRB ? TR : BK, WO = TRA ? TC_ : BK, WI = TRA ? BK : TC_;
      _Pragma("clang loop unroll(full)")
      for (int c = tid; c < XO * (XI / 8); c += 128) {
        int o = c / (XI / 8), in = (c % (XI / 8)) * 8;
        lf_stage8<TB, T>(Xs + o * XROW + in, B + (TRB ? (long)(j0 + in) + (long)(k0 + o) * g.ldb : (long)(k0 + in) + (long)(j0 + o) * g.ldb));
      }
      _Pragma("clang loop unroll(full)")
      for (int c = tid; c < WO * (WI / 8); c += 128) {
        int o = c / (WI / 8), in = (c % (WI / 8)) * 8;
        lf_stage8<TA, T>(Ws + o * WROW + in, A + (TRA ? (long)(k0 + in) + (long)(i0 + o) * g.lda : (long)(i0 + in) + (long)(k0 + o) * g.lda));
      }
    } else {
    {   // X = op(B)^T: element (j, k) = TRB ? B[j + k ldb] : B[k + j ldb]; staged a chunk of 8 along the contiguous axis,
        // 16 bytes where the whole chunk is inside the matrix (and VEC says it is aligned), element by element where not
      constexpr int OUTER = TRB ? BK : TR, INNER = TRB ? TR : BK;
      const long ld = g.ldb;
      _Pragma("clang loop unroll(full)")
      for (int c = tid; c < OUTER * (INNER / 8); c += 128) {
        int o = c / (INNER / 8), in = (c % (INNER / 8)) * 8;
        int ob = TRB ? k0 + o : j0 + o, ib = TRB ? j0 + in : k0 + in;           // (outer, inner) in the matrix
        int olim = TRB ? g.k : g.n, ilim = TRB ? g.n : g.k;
        device const TB* src = B + (long)ib + (long)ob * ld;
        threadgroup T* dst = Xs + o * XROW + in;
        if (vec && ob < olim && ib + 8 <= ilim) lf_stage8<TB, T>(dst, src);
        else for (int e = 0; e < 8; ++e) dst[e] = (ob < olim && ib + e < ilim) ? (T)lf_ldf(src + e) : (T)0;
      }
    }
    {   // W = op(A)^T: element (k, i) = TRA ? A[k + i lda] : A[i + k lda]
      constexpr int OUTER = TRA ? TC_ : BK, INNER = TRA ? BK : TC_;
      const long ld = g.lda;
      _Pragma("clang loop unroll(full)")
      for (int c = tid; c < OUTER * (INNER / 8); c += 128) {
        int o = c / (INNER / 8), in = (c % (INNER / 8)) * 8;
        int ob = TRA ? i0 + o : k0 + o, ib = TRA ? k0 + in : i0 + in;
        int olim = TRA ? g.m : g.k, ilim = TRA ? g.k : g.m;
        device const TA* src = A + (long)ib + (long)ob * ld;
        threadgroup T* dst = Ws + o * WROW + in;
        if (vec && ob < olim && ib + 8 <= ilim) lf_stage8<TA, T>(dst, src);
        else for (int e = 0; e < 8; ++e) dst[e] = (ob < olim && ib + e < ilim) ? (T)lf_ldf(src + e) : (T)0;
      }
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<T, 8, 8> xm[FR], wm[FC];
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        if (TRB) simdgroup_load(xm[a], Xs + kk * XROW + sr + a * 8, XROW, ulong2(0, 0), true);
        else simdgroup_load(xm[a], Xs + (sr + a * 8) * XROW + kk, XROW);
      }
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < FC; ++b) {
        if (TRA) simdgroup_load(wm[b], Ws + (sc + b * 8) * WROW + kk, WROW, ulong2(0, 0), true);
        else simdgroup_load(wm[b], Ws + kk * WROW + sc + b * 8, WROW);
      }
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a)
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) simdgroup_multiply_accumulate(acc[a][b], xm[a], wm[b], acc[a][b]);
    }
  }
  if ((EP & 7) == 1) {   // the triangle's gate: columns in blocks of 8 - 8 channels' pa, then their ga, pb and gb - so a
    // simdgroup's four 8 x 8 accumulators hold one channel's four values at the same lane and element: every lane gates
    // its own (no shuffle, no idle half), and a and b go to Sab channel-major as whole matrices, stored transposed
    threadgroup_barrier(mem_flags::mem_threadgroup);    // (every simdgroup done with the tiles Sab overlays)
    const int sm1 = (lane / 16) * 4 + (lane % 8) / 2, sn1 = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
    device const float* mask = (device const float*)g.aux;
    if constexpr (FC % 4 == 0) {
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        const int jg = j0 + sr + a * 8 + sm1;
        const float m = jg < g.n ? mask[g.tgR0 + jg] : 0.f;
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; b += 4) {
          thread auto& pa = acc[a][b].thread_elements();
          thread auto& ga = acc[a][b + 1].thread_elements();
          thread auto& pb = acc[a][b + 2].thread_elements();
          thread auto& gb = acc[a][b + 3].thread_elements();
          simdgroup_half8x8 ma, mb;
          thread auto& oa = ma.thread_elements();
          thread auto& ob = mb.thread_elements();
          _Pragma("clang loop unroll(full)")
          for (int t = 0; t < 2; ++t) {
            float vpa = pa[t], vga = ga[t], vpb = pb[t], vgb = gb[t];
            if (g.epilogue & 4) {     // (the projection's and gate's biases, f32, in the same order: AF2's)
              device const float* bias = (device const float*)g.bias + i0 + sc + b * 8 + sn1 + t;
              vpa += bias[0]; vga += bias[8]; vpb += bias[16]; vgb += bias[24];
            }
            oa[t] = (half)(vpa * m / (1.f + exp(-vga)));
            ob[t] = (half)(vpb * m / (1.f + exp(-vgb)));
          }
          const int cl0 = (sc + b * 8) / 4;     // (the tile's channel of this block's first: 8 a 32 columns)
          simdgroup_store(ma, Sab + cl0 * TR + sr + a * 8, TR, ulong2(0, 0), true);
          simdgroup_store(mb, Sab + (TC_ / 4 + cl0) * TR + sr + a * 8, TR, ulong2(0, 0), true);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* outA = (device half*)g.aux2; device half* outB = (device half*)g.aux3;
    for (int t = tid; t < (TC_ / 4) * TR; t += 128) {      // channel-major, a channel's rows consecutive
      int cl = t / TR, jl = t % TR, jg = j0 + jl, c = i0 / 4 + cl;
      if (jg >= g.n || c >= g.tgC) continue;
      uint p = (uint)(g.tgR0 + jg), r = lf_udiv(p, (uint)g.tgN);      // (no integer divider: prelude)
      ulong q = (ulong)r * g.tgNp + (p - r * (uint)g.tgN);
      outA[(ulong)c * g.tgPairs + q] = Sab[cl * TR + jl];
      outB[(ulong)c * g.tgPairs + q] = Sab[(TC_ / 4 + cl) * TR + jl];
    }
    return;
  }
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
#define LF_STORE_SWIGLU_8X8(a, b, j, i) do { \
    thread auto& ea = acc[a][b].thread_elements(); \
    thread auto& eb = acc[a][(b) + 1].thread_elements(); \
    simdgroup_matrix<TC, 8, 8> om; \
    thread auto& o = om.thread_elements(); \
    _Pragma("clang loop unroll(full)") \
    for (int t = 0; t < 2; ++t) { \
      float va = g.alpha * (float)ea[t], vb = g.alpha * (float)eb[t]; \
      if (g.epilogue & 4) { \
        const int col = (i) + sn + t; \
        va += ((device const float*)g.bias)[col]; vb += ((device const float*)g.bias)[col + 8]; \
      } \
      o[t] = (TC)(va / (1.f + exp(-va)) * vb); \
    } \
    simdgroup_store(om, D + (ulong)(j) * g.ldd + (i) / 2, (ulong)g.ldd); \
  } while (0)
#define LF_STORE_8X8(a, b, j, i) do { \
    thread auto& e = acc[a][b].thread_elements(); \
    if constexpr ((EP & 64) == 0) { \
      if ((g.epilogue & ~6) == 0 && g.beta == 0.f) { \
        if constexpr (metal::is_same_v<TC, half> == ((EP & 8) != 0)) { \
          if (g.epilogue || g.alpha != 1.f) { \
            _Pragma("clang loop unroll(full)") \
            for (int t = 0; t < 2; ++t) { \
              const int col = (i) + sn + t; \
              const ulong boff = g.ptrs ? (ulong)z * g.m : 0; \
              float v = g.alpha * (float)e[t]; \
              if (g.epilogue & 4) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + col] : ((device const float*)g.bias)[boff + col]; \
              if (g.epilogue & 2) v = max(v, 0.f); \
              e[t] = v; \
            } \
          } \
          simdgroup_store(acc[a][b], D + (ulong)(j) * g.ldd + (i), (ulong)g.ldd); \
        } else { \
          simdgroup_matrix<TC, 8, 8> om; \
          thread auto& o = om.thread_elements(); \
          _Pragma("clang loop unroll(full)") \
          for (int t = 0; t < 2; ++t) { \
            const int col = (i) + sn + t; \
            const ulong boff = g.ptrs ? (ulong)z * g.m : 0; \
            float v = g.alpha * (float)e[t]; \
            if (g.epilogue & 4) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + col] : ((device const float*)g.bias)[boff + col]; \
            if (g.epilogue & 2) v = max(v, 0.f); \
            if constexpr (metal::is_same_v<TC, half>) o[t] = (half)clamp(v, -65504.f, 65504.f); \
            else o[t] = (TC)v; \
          } \
          simdgroup_store(om, D + (ulong)(j) * g.ldd + (i), (ulong)g.ldd); \
        } \
        break; \
      } \
    } \
    simdgroup_matrix<TC, 8, 8> om, cm; \
    simdgroup_matrix<float, 8, 8> cfm; \
    simdgroup_matrix<half, 8, 8> xm; \
    const bool gated = (EP & 64) == 0 && (g.epilogue & 64); \
    const bool cf32 = (g.biasType & 512) != 0; \
    const bool loadC = gated || g.beta != 0.f || ((EP & 64) != 0 && (g.epilogue & (8192 | 1048576))); \
    if (loadC) { \
      if (cf32) simdgroup_load(cfm, (device const float*)g.C + (ulong)(j) * g.ldc + (i), (ulong)g.ldc); \
      else simdgroup_load(cm, C + (ulong)(j) * g.ldc + (i), (ulong)g.ldc); \
    } \
    if (gated) simdgroup_load(xm, (device const half*)g.aux + (ulong)(j) * g.ldaux + (i), (ulong)g.ldaux); \
    thread auto& o = om.thread_elements(); \
    thread auto& c = cm.thread_elements(); \
    thread auto& cf = cfm.thread_elements(); \
    thread auto& x = xm.thread_elements(); \
    _Pragma("clang loop unroll(full)") \
    for (int t = 0; t < 2; ++t) { \
      const int col = (i) + sn + t; \
      float v = g.alpha * (float)e[t]; \
      const float cv = loadC ? (cf32 ? cf[t] : (float)c[t]) : 0.f; \
      if constexpr ((EP & 64) != 0) { \
        if constexpr (metal::is_same_v<TC, half>) { \
          if (g.epilogue & 1048576) v = cv * v; \
        } \
        if (g.epilogue & 8192) { \
          const float b128 = ((device const float*)g.bias)[col & 127]; \
          const float s = 1e-3f + ((device const float*)g.aux)[(uint)((j) + sm) * ((uint)g.m >> 7) + ((uint)(i) >> 7)]; \
          v = cv + (b128 + v) / s; \
        } \
      } else if (gated) { \
        if (g.epilogue & 4) v += ((device const float*)g.bias)[col]; \
        v = cv + (float)x[t] * (1.f / (1.f + exp(-v))); \
      } else { \
        if (g.beta != 0.f) v += g.beta * cv; \
        if (g.epilogue & 4) { \
          const ulong boff = g.ptrs ? (ulong)z * g.m : 0; \
          v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + col] : ((device const float*)g.bias)[boff + col]; \
        } \
        if (g.epilogue & 2) v = max(v, 0.f); \
        if (g.epilogue & 32) v = 0.5f * v * (1.f + lf_erf(v * 0.70710678118654752f)); \
      } \
      if constexpr (metal::is_same_v<TC, half> == ((EP & 8) != 0)) e[t] = v; \
      else if constexpr (metal::is_same_v<TC, half>) o[t] = (half)clamp(v, -65504.f, 65504.f); \
      else o[t] = (TC)v; \
    } \
    if constexpr (metal::is_same_v<TC, half> == ((EP & 8) != 0)) simdgroup_store(acc[a][b], D + (ulong)(j) * g.ldd + (i), (ulong)g.ldd); \
    else simdgroup_store(om, D + (ulong)(j) * g.ldd + (i), (ulong)g.ldd); \
  } while (0)
  if constexpr ((EP & 16) == 0 && (metal::is_same_v<TC, half> || metal::is_same_v<TC, float>)) {
    if (j0 + TR <= g.n && i0 + TC_ <= g.m) {
      if (g.epilogue & 128) {
        if constexpr (FC % 2 == 0) {
          _Pragma("clang loop unroll(full)")
          for (int a = 0; a < FR; ++a)
            _Pragma("clang loop unroll(full)")
            for (int b = 0; b < FC; b += 2) LF_STORE_SWIGLU_8X8(a, b, j0 + sr + a * 8, i0 + sc + b * 8);
          return;
        }
      } else if ((EP & 64) == 0 || (g.epilogue & (8192 | 1048576))) {
        _Pragma("clang loop unroll(full)")
        for (int a = 0; a < FR; ++a)
          _Pragma("clang loop unroll(full)")
          for (int b = 0; b < FC; ++b) LF_STORE_8X8(a, b, j0 + sr + a * 8, i0 + sc + b * 8);
        return;
      }
    }
  }
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) {
      const int j = j0 + sr + a * 8, i = i0 + sc + b * 8;
      if (j >= g.n || i >= g.m) continue;
      thread auto& e = acc[a][b].thread_elements();
      if (g.epilogue & 128) {
        if constexpr (FC % 2 == 0) {
          if ((b & 1) == 0 && j + sm < g.n) {
            thread auto& eb = acc[a][b | 1].thread_elements();
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < 2; ++t) {
              int col = i + sn + t;
              if (col + 8 < g.m) {
                float va = g.alpha * (float)e[t], vb = g.alpha * (float)eb[t];
                if (g.epilogue & 4) { va += ((device const float*)g.bias)[col]; vb += ((device const float*)g.bias)[col + 8]; }
                lf_st(D, (ulong)((col >> 4) * 8 + (col & 7)) + (ulong)(j + sm) * g.ldd, va / (1.f + exp(-va)) * vb);
              }
            }
          }
        }
        continue;
      }
      if constexpr ((EP & 16) == 0 && (EP & 64) != 0 && metal::is_same_v<TC, half>) {
        if ((g.epilogue & 4096) && (g.tgC & 7) == 0 && j + 8 <= g.n && i + 8 <= g.m) {
          const uint O = (uint)g.tgC, L = (uint)g.tgN;
          const uint ii = lf_udiv((uint)j, O), cc = (uint)j - ii * O, jj = lf_udiv((uint)i, O), ee = (uint)i - jj * O;
          simdgroup_matrix<half, 8, 8> om;
          thread auto& o = om.thread_elements();
          _Pragma("clang loop unroll(full)")
          for (int t = 0; t < 2; ++t) o[t] = (half)clamp(g.alpha * (float)e[t], -65504.f, 65504.f);
          simdgroup_store(om, D + (((ulong)ii * L + jj) * O + cc) * O + ee, (ulong)O);
          continue;
        }
      }
#undef LF_STORE_8X8
#undef LF_STORE_SWIGLU_8X8
      const int jr = j + sm;
      if (jr >= g.n) continue;
      _Pragma("clang loop unroll(full)")
      for (int t = 0; t < 2; ++t) {
        const int ic = i + sn + t;
        if (ic >= g.m) continue;
        float v = g.alpha * e[t];
        if constexpr ((EP & 64) != 0) {
          if constexpr (metal::is_same_v<TC, float>) {
            if (g.epilogue & 262144) {
              device const int* mp = (device const int*)g.aux2;
              const ulong row = mp ? (ulong)(uint)mp[jr] : (ulong)jr;
              const float gm = ((device const float*)g.aux)[row * g.ldaux + ic];
              lf_st(D, (ulong)ic + (ulong)jr * g.ldd, lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc) + gm * v);
              continue;
            }
            if (g.epilogue & 16384) {
              if (jr >= g.tgPairs) continue;
              const uint r = (uint)jr - lf_udiv((uint)jr, (uint)g.tgN) * (uint)g.tgN;
              const float gp = ((device const float*)g.aux)[(ulong)r * g.ldaux + ic];
              const float gm = g.tgC ? gp : (1.f / (1.f + exp(-gp)));
              lf_st(D, (ulong)ic + (ulong)jr * g.ldd, lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc) + v * gm);
              continue;
            }
            if (g.epilogue & 524288) {
              if (jr >= g.tgPairs) continue;
              if (g.epilogue & 4) v += ((device const float*)g.bias)[ic];
              const uint W = (uint)g.tgC, N = (uint)g.tgN, BS = (uint)g.m / W;
              const uint k = lf_udiv((uint)jr, N), r = (uint)jr - k * N;
              const uint bs = lf_udiv((uint)ic, W), col = (uint)ic - bs * W;
              if ((int)col < g.tgNp) v = 1.f / (1.f + exp(-v));
              lf_st(D, (((ulong)k * BS + bs) * N + r) * W + col, v);
              continue;
            }
          }
          if constexpr (metal::is_same_v<TC, half>) {
            if (g.epilogue & 1048576) {
              lf_st(D, (ulong)ic + (ulong)jr * g.ldd, lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc) * v);
              continue;
            }
            if (g.epilogue & 4096) {
              const uint O = (uint)g.tgC, L = (uint)g.tgN;
              const uint ii = lf_udiv((uint)jr, O), cc = (uint)jr - ii * O, jj = lf_udiv((uint)ic, O), ee = (uint)ic - jj * O;
              lf_st(D, (((ulong)ii * L + jj) * O + cc) * O + ee, v);
              continue;
            }
          }
          if (g.epilogue & 8192) {
            const float cv = (g.biasType & 512) ? ((device const float*)g.C)[(ulong)ic + (ulong)jr * g.ldc] : lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc);
            const float b128 = ((device const float*)g.bias)[ic & 127];
            const float s = 1e-3f + ((device const float*)g.aux)[(uint)jr * ((uint)g.m >> 7) + ((uint)ic >> 7)];
            lf_st(D, (ulong)ic + (ulong)jr * g.ldd, cv + (b128 + v) / s);
            continue;
          }
        }
        const float cv = (g.biasType & 512) ? ((device const float*)g.C)[(ulong)ic + (ulong)jr * g.ldc] : lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc);
        if (g.epilogue & 64) {      // a gated residual: the GEMM is the gate, aux the gated values, D the residual
          if (g.epilogue & 4) v += ((device const float*)g.bias)[ic];      // (the gate's bias, f32: AF2's)
          v = (float)((device const half*)g.aux)[(ulong)ic + (ulong)jr * g.ldaux] * (1.f / (1.f + exp(-v)));
          lf_st(D, (ulong)ic + (ulong)jr * g.ldd, cv + v);
          continue;
        }
        if (g.beta != 0.f) v += g.beta * cv;
        if (g.epilogue & 4) {
          const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
          v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + ic] : ((device const float*)g.bias)[boff + ic];
        }
        if (g.epilogue & 2) v = max(v, 0.f);
        if (g.epilogue & 32) v = 0.5f * v * (1.f + lf_erf(v * 0.70710678118654752f));   // (GELU, erf's)
        lf_st(D, (ulong)ic + (ulong)jr * g.ldd, v);
      }
    }
}
// (instantiated on first use by the runtime: lf_gemm<TA, TB, TC, TR, TC_, TRA, TRB>, host name lf_gemm_...)

// ---------------------------------------------------------------- an experiment (metal/bench): the plain half GEMM alone,
// D[n][m] = X[n][k] W[k][m] row-major, TR x TC a threadgroup of SGR x SGC simdgroups, BK a step - no epilogues, whole
// tiles only (the bench's shapes divide)
template <int TR, int TC_, int BK, int SGR, int SGC, bool HACC>
kernel void lf_gemm_x(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int NT = 32 * SGR * SGC, PAD = 8, XROW = BK + PAD, WROW = TC_ + PAD;
  constexpr int WR = TR / SGR, WC = TC_ / SGC, FR = WR / 8, FC = WC / 8;
  threadgroup half Xs[TR * XROW];
  threadgroup half Ws[BK * WROW];
  device const half* W = (device const half*)g.A;
  device const half* X = (device const half*)g.B;
  device half* D = (device half*)g.D;
  const int j0 = grp.y * TR, i0 = grp.x * TC_;
  const int sr = (sg / SGC) * WR, sc = (sg % SGC) * WC;
  typedef metal::conditional_t<HACC, simdgroup_half8x8, simdgroup_float8x8> ACC;
  ACC acc[FR][FC];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) acc[a][b] = ACC(0);
  for (int k0 = 0; k0 < g.k; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (int c = tid; c < TR * (BK / 8); c += NT) {
      int o = c / (BK / 8), in = (c % (BK / 8)) * 8;
      device const half* src = X + (long)(j0 + o) * g.ldb + k0 + in;
      *(threadgroup half4*)(Xs + o * XROW + in) = *(device const half4*)src;
      *(threadgroup half4*)(Xs + o * XROW + in + 4) = *(device const half4*)(src + 4);
    }
    _Pragma("clang loop unroll(full)")
    for (int c = tid; c < BK * (TC_ / 8); c += NT) {
      int o = c / (TC_ / 8), in = (c % (TC_ / 8)) * 8;
      device const half* src = W + (long)(k0 + o) * g.lda + i0 + in;
      *(threadgroup half4*)(Ws + o * WROW + in) = *(device const half4*)src;
      *(threadgroup half4*)(Ws + o * WROW + in + 4) = *(device const half4*)(src + 4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_half8x8 xm[FR], wm[FC];
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) simdgroup_load(xm[a], Xs + (sr + a * 8) * XROW + kk, XROW);
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < FC; ++b) simdgroup_load(wm[b], Ws + kk * WROW + sc + b * 8, WROW);
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a)
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) simdgroup_multiply_accumulate(acc[a][b], xm[a], wm[b], acc[a][b]);
    }
  }
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) {
      simdgroup_half8x8 o;
      if constexpr (HACC) o = acc[a][b];
      else { thread auto& e = acc[a][b].thread_elements(); thread auto& f = o.thread_elements(); f[0] = (half)e[0]; f[1] = (half)e[1]; }
      simdgroup_store(o, D + (long)(j0 + sr + a * 8) * g.ldd + i0 + sc + b * 8, (ulong)g.ldd);
    }
}
