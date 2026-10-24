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
  constexpr int SROW = TR + 8;
  constexpr int SABSZ = (EP & 7) == 1 ? (2 * (TC_ / 4) * SROW * 2 + sizeof(T) - 1) / sizeof(T) : 0;
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
  // (vector staging: the host's word, or for a pointer array this entry's own pointers' alignment)
  const bool vec = (g.biasType & 256) && (!g.ptrs || (((ulong)A % (8 * sizeof(TA))) == 0 && ((ulong)B % (8 * sizeof(TB))) == 0));
  // a whole tile (the common case): staged without a bounds check an element (6% of a K-256 projection's time)
  const bool wholeX = vec && j0 + TR <= g.n && g.k % BK == 0;
  const bool wholeW = vec && i0 + TC_ <= g.m && g.k % BK == 0;
  for (int k0 = 0; k0 < g.k; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wholeX) {
      constexpr int XO = TRB ? BK : TR, XI = TRB ? TR : BK;
      _Pragma("clang loop unroll(full)")
      for (int c = tid; c < XO * (XI / 8); c += 128) {
        int o = c / (XI / 8), in = (c % (XI / 8)) * 8;
        lf_stage8<TB, T>(Xs + o * XROW + in, B + (TRB ? (long)(j0 + in) + (long)(k0 + o) * g.ldb : (long)(k0 + in) + (long)(j0 + o) * g.ldb));
      }
    } else if (!TRB && vec && g.k % BK == 0) {
      _Pragma("clang loop unroll(full)")
      for (int c = tid; c < TR * (BK / 8); c += 128) {
        int o = c / (BK / 8), in = (c % (BK / 8)) * 8;
        if (j0 + o < g.n) lf_stage8<TB, T>(Xs + o * XROW + in, B + (long)(k0 + in) + (long)(j0 + o) * g.ldb);
        else if (k0 == 0) { _Pragma("clang loop unroll(full)") for (int e = 0; e < 8; ++e) Xs[o * XROW + in + e] = (T)0; }
      }
    } else {
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
    if (wholeW) {
      constexpr int WO = TRA ? TC_ : BK, WI = TRA ? BK : TC_;
      _Pragma("clang loop unroll(full)")
      for (int c = tid; c < WO * (WI / 8); c += 128) {
        int o = c / (WI / 8), in = (c % (WI / 8)) * 8;
        lf_stage8<TA, T>(Ws + o * WROW + in, A + (TRA ? (long)(k0 + in) + (long)(i0 + o) * g.lda : (long)(i0 + in) + (long)(k0 + o) * g.lda));
      }
    } else {
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
          float2 bpa(0.f), bga(0.f), bpb(0.f), bgb(0.f);
          if (g.epilogue & 4) {     // (the projection's and gate's biases, f32, in the same order: AF2's)
            device const float* bias = (device const float*)g.bias + i0 + sc + b * 8 + sn1;
            bpa = *(device const float2*)(bias);
            bga = *(device const float2*)(bias + 8);
            bpb = *(device const float2*)(bias + 16);
            bgb = *(device const float2*)(bias + 24);
          }
          _Pragma("clang loop unroll(full)")
          for (int t = 0; t < 2; ++t) {
            float vpa = pa[t] + bpa[t], vga = ga[t] + bga[t], vpb = pb[t] + bpb[t], vgb = gb[t] + bgb[t];
            oa[t] = (half)(vpa * m / (1.f + exp(-vga)));
            ob[t] = (half)(vpb * m / (1.f + exp(-vgb)));
          }
          const int cl0 = (sc + b * 8) / 4;     // (the tile's channel of this block's first: 8 a 32 columns)
          simdgroup_store(ma, Sab + cl0 * SROW + sr + a * 8, SROW, ulong2(0, 0), true);
          simdgroup_store(mb, Sab + (TC_ / 4 + cl0) * SROW + sr + a * 8, SROW, ulong2(0, 0), true);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* outA = (device half*)g.aux2; device half* outB = (device half*)g.aux3;
    const bool eqN = g.tgNp == g.tgN;
    if constexpr (128 % TR == 0) {
      const int jl = tid % TR, jg = j0 + jl;
      if (jg < g.n) {
        const uint p = (uint)(g.tgR0 + jg);
        const ulong q = eqN ? (ulong)p : (ulong)lf_udiv(p, (uint)g.tgN) * (uint)(g.tgNp - g.tgN) + p;
        _Pragma("clang loop unroll(full)")
        for (int cl = tid / TR; cl < TC_ / 4; cl += 128 / TR) {
          const int c = i0 / 4 + cl;
          if (c >= g.tgC) break;
          const ulong dst = (ulong)c * g.tgPairs + q;
          outA[dst] = Sab[cl * SROW + jl];
          outB[dst] = Sab[(TC_ / 4 + cl) * SROW + jl];
        }
      }
    } else {
      for (int t = tid; t < (TC_ / 4) * TR; t += 128) {      // channel-major, a channel's rows consecutive
        int cl = t / TR, jl = t % TR, jg = j0 + jl, c = i0 / 4 + cl;
        if (jg >= g.n || c >= g.tgC) continue;
        uint p = (uint)(g.tgR0 + jg);
        ulong q = eqN ? (ulong)p : (ulong)lf_udiv(p, (uint)g.tgN) * (uint)(g.tgNp - g.tgN) + p;
        outA[(ulong)c * g.tgPairs + q] = Sab[cl * SROW + jl];
        outB[(ulong)c * g.tgPairs + q] = Sab[(TC_ / 4 + cl) * SROW + jl];
      }
    }
    return;
  }
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
#define LF_STORE_SWIGLU_8X8(a, b, j, i) do { \
    thread auto& ea = acc[a][b].thread_elements(); \
    thread auto& eb = acc[a][(b) + 1].thread_elements(); \
    simdgroup_matrix<TC, 8, 8> om; \
    thread auto& o = om.thread_elements(); \
    float2 bva(0.f), bvb(0.f); \
    if (g.epilogue & 4) { \
      device const float* bp = (device const float*)g.bias + (i) + sn; \
      bva = *(device const float2*)(bp); bvb = *(device const float2*)(bp + 8); \
    } \
    _Pragma("clang loop unroll(full)") \
    for (int t = 0; t < 2; ++t) { \
      float va = g.alpha * (float)ea[t] + bva[t], vb = g.alpha * (float)eb[t] + bvb[t]; \
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
            float2 bv2(0.f); \
            if (g.epilogue & 4) { \
              const ulong boff = (g.ptrs ? (ulong)z * g.m : 0) + (i) + sn; \
              bv2 = (g.biasType & 255) == 2 ? float2(*(device const half2*)((device const half*)g.bias + boff)) \
                                            : *(device const float2*)((device const float*)g.bias + boff); \
            } \
            _Pragma("clang loop unroll(full)") \
            for (int t = 0; t < 2; ++t) { \
              float v = g.alpha * (float)e[t] + bv2[t]; \
              if (g.epilogue & 2) v = max(v, 0.f); \
              e[t] = v; \
            } \
          } \
          simdgroup_store(acc[a][b], D + (ulong)(j) * g.ldd + (i), (ulong)g.ldd); \
        } else { \
          simdgroup_matrix<TC, 8, 8> om; \
          thread auto& o = om.thread_elements(); \
          float2 bv2(0.f); \
          if (g.epilogue & 4) { \
            const ulong boff = (g.ptrs ? (ulong)z * g.m : 0) + (i) + sn; \
            bv2 = (g.biasType & 255) == 2 ? float2(*(device const half2*)((device const half*)g.bias + boff)) \
                                          : *(device const float2*)((device const float*)g.bias + boff); \
          } \
          _Pragma("clang loop unroll(full)") \
          for (int t = 0; t < 2; ++t) { \
            float v = g.alpha * (float)e[t] + bv2[t]; \
            if (g.epilogue & 2) v = max(v, 0.f); \
            if constexpr (metal::is_same_v<TC, half>) { const half hv = (half)clamp(v, -65504.f, 65504.f); o[t] = hv; if constexpr ((EP & 128) != 0) e[t] = (float)hv; } \
            else { o[t] = (TC)v; if constexpr ((EP & 128) != 0) e[t] = v; } \
          } \
          simdgroup_store(om, D + (ulong)(j) * g.ldd + (i), (ulong)g.ldd); \
        } \
        break; \
      } \
    } \
    simdgroup_matrix<TC, 8, 8> om; \
    simdgroup_matrix<half, 8, 8> xm; \
    const bool gated = (EP & 8) == 0 && (EP & 64) == 0 && (g.epilogue & 64); \
    const bool loadC = ((EP & 8) == 0 && (gated || g.beta != 0.f)) || ((EP & 64) != 0 && (g.epilogue & (8192 | 1048576))); \
    if constexpr ((EP & 8) == 0 && !metal::is_same_v<TC, float>) { \
      if (loadC && !(g.biasType & 512)) simdgroup_load(om, C + (ulong)(j) * g.ldc + (i), (ulong)g.ldc); \
    } else if constexpr ((EP & 8) == 0 || (EP & 64) != 0) { \
      if (loadC) simdgroup_load(om, C + (ulong)(j) * g.ldc + (i), (ulong)g.ldc); \
    } \
    if (gated) simdgroup_load(xm, (device const half*)g.aux + (ulong)(j) * g.ldaux + (i), (ulong)g.ldaux); \
    thread auto& o = om.thread_elements(); \
    thread auto& x = xm.thread_elements(); \
    float2 bv2(0.f); \
    if ((EP & 64) == 0 && (g.epilogue & 4)) { \
      const ulong boff = (g.ptrs ? (ulong)z * g.m : 0) + (i) + sn; \
      bv2 = (!gated && (g.biasType & 255) == 2) ? float2(*(device const half2*)((device const half*)g.bias + boff)) \
                                                : *(device const float2*)((device const float*)g.bias + boff); \
    } \
    _Pragma("clang loop unroll(full)") \
    for (int t = 0; t < 2; ++t) { \
      const int col = (i) + sn + t; \
      float v = g.alpha * (float)e[t]; \
      float cv = loadC ? (float)o[t] : 0.f; \
      if constexpr ((EP & 8) == 0 && !metal::is_same_v<TC, float>) { \
        if (loadC && (g.biasType & 512)) cv = ((device const float*)g.C)[(ulong)((j) + sm) * g.ldc + col]; \
      } \
      if constexpr ((EP & 64) != 0) { \
        if constexpr (metal::is_same_v<TC, half>) { \
          if (g.epilogue & 1048576) v = cv * v; \
        } \
        if (g.epilogue & 8192) { \
          const uint opmC = (uint)g.tgN, u = (uint)col; \
          const uint jTok = opmC == 128 ? (u >> 7) : opmC == 256 ? (u >> 8) : lf_udiv(u, opmC); \
          const float bCol = ((device const float*)g.bias)[u - jTok * opmC]; \
          const float nv = ((device const float*)g.aux)[(uint)((j) + sm) * (uint)g.tgNp + jTok]; \
          v = cv + (g.tgC ? v / max(nv, 1.f) + bCol : (bCol + v) / (1e-3f + nv)); \
        } \
      } else if (gated) { \
        if (g.epilogue & 4) v += bv2[t]; \
        v = cv + (float)x[t] * (1.f / (1.f + exp(-v))); \
      } else { \
        if (g.beta != 0.f) v += g.beta * cv; \
        if (g.epilogue & 4) v += bv2[t]; \
        if (g.epilogue & 2) v = max(v, 0.f); \
        if (g.epilogue & 32) v = 0.5f * v * (1.f + lf_erf(v * 0.70710678118654752f)); \
      } \
      if constexpr (metal::is_same_v<TC, half> == ((EP & 8) != 0)) e[t] = v; \
      else if constexpr (metal::is_same_v<TC, half>) { const half hv = (half)clamp(v, -65504.f, 65504.f); o[t] = hv; if constexpr ((EP & 128) != 0) e[t] = (float)hv; } \
      else { o[t] = (TC)v; if constexpr ((EP & 128) != 0) e[t] = v; } \
    } \
    if constexpr (metal::is_same_v<TC, half> == ((EP & 8) != 0)) simdgroup_store(acc[a][b], D + (ulong)(j) * g.ldd + (i), (ulong)g.ldd); \
    else simdgroup_store(om, D + (ulong)(j) * g.ldd + (i), (ulong)g.ldd); \
  } while (0)
  if constexpr ((EP & 16) == 0 && (EP & 64) != 0 && metal::is_same_v<TC, half>) {
    if (g.epilogue & 32768) {
      int jr_row[FR], st_row[FR];
      uint t_row[FR];
      float inv_row[FR], dq_row[FR], sq_row[FR], sa_row[FR];
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        const int jr = j0 + sr + a * 8 + sm;
        jr_row[a] = jr;
        if (jr < g.n) {
          device const float4* sr4 = (device const float4*)((device const float*)g.aux + (ulong)jr * 8);
          const float4 s0 = sr4[0], s1 = sr4[1];
          inv_row[a] = s0.x; dq_row[a] = s0.z; sq_row[a] = s0.w; sa_row[a] = s1.y;
          st_row[a] = (int)s1.z; t_row[a] = (uint)s1.w;
        } else {
          inv_row[a] = 0.f; dq_row[a] = 0.f; sq_row[a] = 0.f; sa_row[a] = 0.f;
          st_row[a] = -2; t_row[a] = 0xffffffffu;
        }
      }
      if (g.epilogue & 131072) {
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) {
          const int col = i0 + sc + b * 8 + sn;
          if (col >= g.m) break;
          int prev_st = -2;
          uint prev_t = 0xffffffffu;
          float2 pv(0.f), qv(0.f), bv(0.f);
          _Pragma("clang loop unroll(full)")
          for (int a = 0; a < FR; ++a) {
            const int jr = jr_row[a];
            if (jr >= g.n) continue;
            const int st = st_row[a];
            const uint tq = t_row[a];
            if (st != prev_st) {
              pv = st >= 0 ? sa_row[a] * *(device const float2*)((device const float*)g.aux2 + (ulong)st * g.m + col) : float2(0.f);
              prev_st = st;
            }
            if (tq != prev_t) {
              device const float* qb = (device const float*)g.aux3 + (ulong)tq * 2 * g.m + col;
              qv = *(device const float2*)(qb);
              bv = *(device const float2*)(qb + g.m);
              prev_t = tq;
            }
            thread auto& e = acc[a][b].thread_elements();
            const float2 uv = inv_row[a] * (pv + sq_row[a] * float2((float)e[0], (float)e[1]) + dq_row[a] * qv) + bv;
            const float2 u2 = simd_shuffle_xor(uv, 1);
            if ((lane & 1) == 0)
              *(device half2*)(D + (ulong)jr * g.ldd + (col >> 1)) = half2(fast::divide(uv, 1.f + fast::exp(-uv)) * u2);
          }
        }
      } else if constexpr (FC % 2 == 0) {
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; b += 2) {
          const int i = i0 + sc + b * 8, col = i + sn;
          if (col + 8 >= g.m) break;
          _Pragma("clang loop unroll(full)")
          for (int a = 0; a < FR; ++a) {
            const int jr = jr_row[a];
            if (jr >= g.n) continue;
            const int st = st_row[a];
            device const float* pr = (device const float*)g.aux2 + (ulong)max(st, 0) * g.m;
            device const float* qb = (device const float*)g.aux3 + (ulong)t_row[a] * 2 * g.m;
            thread auto& ea = acc[a][b].thread_elements();
            thread auto& eb = acc[a][b + 1].thread_elements();
            const float2 p1 = st >= 0 ? sa_row[a] * *(device const float2*)(pr + col) : float2(0.f);
            const float2 p2 = st >= 0 ? sa_row[a] * *(device const float2*)(pr + col + 8) : float2(0.f);
            const float2 q1 = *(device const float2*)(qb + col), q2 = *(device const float2*)(qb + col + 8);
            const float2 b1 = *(device const float2*)(qb + g.m + col), b2 = *(device const float2*)(qb + g.m + col + 8);
            const float2 u1 = inv_row[a] * (p1 + sq_row[a] * float2((float)ea[0], (float)ea[1]) + dq_row[a] * q1) + b1;
            const float2 u2 = inv_row[a] * (p2 + sq_row[a] * float2((float)eb[0], (float)eb[1]) + dq_row[a] * q2) + b2;
            *(device half2*)(D + (ulong)jr * g.ldd + (i >> 1) + sn) = half2(fast::divide(u1, 1.f + fast::exp(-u1)) * u2);
          }
        }
      }
      return;
    }
    if constexpr ((EP & 16) == 0 && metal::is_same_v<TC, float>) {
      if ((g.epilogue & 16384) && i0 + TC_ <= g.m) {
        const int jLim = min(g.n, (int)g.tgPairs);
        _Pragma("clang loop unroll(full)")
        for (int a = 0; a < FR; ++a) {
          const int j = j0 + sr + a * 8;
          if (j >= jLim) continue;
          const uint jr = (uint)(j + sm);
          const uint r = jr < (uint)g.tgN ? jr : jr - lf_udiv(jr, (uint)g.tgN) * (uint)g.tgN;
          const float rm = g.aux2 ? ((device const float*)g.aux2)[r] : 1.f;
          device const float* gp0 = (device const float*)g.aux + (ulong)r * g.ldaux;
          if (j + 8 <= jLim) {
            _Pragma("clang loop unroll(full)")
            for (int b = 0; b < FC; ++b) {
              const int i = i0 + sc + b * 8;
              thread auto& e = acc[a][b].thread_elements();
              simdgroup_matrix<float, 8, 8> om, cm;
              simdgroup_load(cm, C + (ulong)j * g.ldc + i, (ulong)g.ldc);
              thread auto& o = om.thread_elements();
              thread auto& c = cm.thread_elements();
              const float2 gp = *(device const float2*)(gp0 + i + sn);
              const float2 gm = g.tgC ? gp : (1.f / (1.f + exp(-gp)));
              float r0 = c[0] + g.alpha * (float)e[0] * gm.x;
              float r1 = c[1] + g.alpha * (float)e[1] * gm.y;
              if (g.aux2) { r0 *= rm; r1 *= rm; }
              o[0] = r0; o[1] = r1;
              simdgroup_store(om, D + (ulong)j * g.ldd + i, (ulong)g.ldd);
              if (g.aux3) {
                simdgroup_matrix<half, 8, 8> hm;
                thread auto& h = hm.thread_elements();
                h[0] = (half)clamp(r0, -65504.f, 65504.f); h[1] = (half)clamp(r1, -65504.f, 65504.f);
                simdgroup_store(hm, (device half*)g.aux3 + (ulong)j * g.ldd + i, (ulong)g.ldd);
              }
            }
          } else if ((int)jr < jLim) {
            device const float* cp0 = C + (ulong)jr * g.ldc;
            device float* dp0 = D + (ulong)jr * g.ldd;
            device half* hp0 = g.aux3 ? (device half*)g.aux3 + (ulong)jr * g.ldd : nullptr;
            _Pragma("clang loop unroll(full)")
            for (int b = 0; b < FC; ++b) {
              const int i = i0 + sc + b * 8 + sn;
              thread auto& e = acc[a][b].thread_elements();
              const float2 gp = *(device const float2*)(gp0 + i);
              const float2 gm = g.tgC ? gp : (1.f / (1.f + exp(-gp)));
              const float2 cv = *(device const float2*)(cp0 + i);
              float2 res = cv + g.alpha * float2((float)e[0], (float)e[1]) * gm;
              if (g.aux2) res *= rm;
              *(device float2*)(dp0 + i) = res;
              if (hp0) *(device half2*)(hp0 + i) = half2(clamp(res, -65504.f, 65504.f));
            }
          }
        }
        return;
      }
      if ((g.epilogue & 524288) && (g.tgC & 7) == 0 && i0 + TC_ <= g.m) {
        const uint W = (uint)g.tgC, N = (uint)g.tgN, BS = (uint)g.m / W;
        const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
        const int jLim = min(g.n, (int)g.tgPairs);
        uint bs_col[FC], col_off[FC];
        float2 bv[FC];
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) {
          const uint i = (uint)(i0 + sc + b * 8);
          const uint bsk = lf_udiv(i, W);
          bs_col[b] = bsk;
          col_off[b] = i - bsk * W;
          if (g.epilogue & 4)
            bv[b] = (g.biasType & 255) == 2 ? float2(*(device const half2*)((device const half*)g.bias + boff + i + sn))
                                            : *(device const float2*)((device const float*)g.bias + boff + i + sn);
          else bv[b] = float2(0.f);
        }
        _Pragma("clang loop unroll(full)")
        for (int a = 0; a < FR; ++a) {
          const int j = j0 + sr + a * 8;
          if (j >= jLim) continue;
          const uint k0 = lf_udiv((uint)j, N), r0 = (uint)j - k0 * N;
          if (j + 8 <= jLim && r0 + 8 <= N) {
            _Pragma("clang loop unroll(full)")
            for (int b = 0; b < FC; ++b) {
              thread auto& e = acc[a][b].thread_elements();
              simdgroup_matrix<float, 8, 8> om;
              thread auto& o = om.thread_elements();
              const uint col0 = col_off[b];
              _Pragma("clang loop unroll(full)")
              for (int t = 0; t < 2; ++t) {
                float v = g.alpha * (float)e[t];
                if (g.epilogue & 4) v += bv[b][t];
                if ((int)col0 + sn + t < g.tgNp) v = 1.f / (1.f + exp(-v));
                o[t] = v;
              }
              simdgroup_store(om, D + (((ulong)k0 * BS + bs_col[b]) * N + r0) * W + col0, (ulong)W);
            }
          } else {
            const int jr = j + sm;
            if (jr >= jLim) continue;
            const uint kr = lf_udiv((uint)jr, N), rr = (uint)jr - kr * N;
            _Pragma("clang loop unroll(full)")
            for (int b = 0; b < FC; ++b) {
              thread auto& e = acc[a][b].thread_elements();
              const uint col0 = col_off[b];
              float2 v = g.alpha * float2((float)e[0], (float)e[1]);
              if (g.epilogue & 4) v += bv[b];
              if ((int)col0 + sn < g.tgNp) v.x = 1.f / (1.f + exp(-v.x));
              if ((int)col0 + sn + 1 < g.tgNp) v.y = 1.f / (1.f + exp(-v.y));
              *(device float2*)(D + (((ulong)kr * BS + bs_col[b]) * N + rr) * W + col0 + sn) = v;
            }
          }
        }
        return;
      }
    }
  }
  // A whole tile in range stores in 8 x 8 blocks with both loops unrolled, and everything else takes the loop below
  // element by element. 5646656b gave that loop a per-row 8 x 8 store too, and on Apple8 - registers allocated
  // statically, for the whole kernel - that store's mere presence cost every lf_gemm 15x (the bench's 1024 x 66816 x 256
  // 12.4 -> 189 ms, an AF2 pass at 261 residues and 1536 rows 17 -> 84 s), whether a tile reached it or not. The M5
  // never showed it: it allocates registers dynamically, and runs lf_gemm_tensor anyway.
  bool stored8x8 = false;
  constexpr bool PLAIN_IN_PLACE = (EP & 64) == 0 && metal::is_same_v<TC, half> == ((EP & 8) != 0);
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
      } else if (PLAIN_IN_PLACE && (g.epilogue & ~6) == 0 && g.beta == 0.f) {
        // a plain product (a bias, a ReLU) in the output's type: the accumulators adjusted in place and stored, decided
        // once for the tile - asked inside LF_STORE_8X8, every one of the FR x FC blocks carries the general epilogue
        // beside it, and on Apple8 that is 5% of a plain GEMM (1024 x 66816 x 256: 12.3 -> 12.9 ms at 0de9becb)
        if constexpr (PLAIN_IN_PLACE) {
          _Pragma("clang loop unroll(full)")
          for (int a = 0; a < FR; ++a)
            _Pragma("clang loop unroll(full)")
            for (int b = 0; b < FC; ++b) {
              const int i = i0 + sc + b * 8;
              thread auto& e = acc[a][b].thread_elements();
              if (g.epilogue || g.alpha != 1.f) {
                float2 bv2(0.f);
                if (g.epilogue & 4) {
                  const ulong boff = (g.ptrs ? (ulong)z * g.m : 0) + i + sn;
                  bv2 = (g.biasType & 255) == 2 ? float2(*(device const half2*)((device const half*)g.bias + boff))
                                                : *(device const float2*)((device const float*)g.bias + boff);
                }
                _Pragma("clang loop unroll(full)")
                for (int t = 0; t < 2; ++t) {
                  float v = g.alpha * (float)e[t] + bv2[t];
                  if (g.epilogue & 2) v = max(v, 0.f);
                  e[t] = v;
                }
              }
              simdgroup_store(acc[a][b], D + (ulong)(j0 + sr + a * 8) * g.ldd + i, (ulong)g.ldd);
            }
          if constexpr ((EP & 128) == 0) return;
          stored8x8 = true;
        }
      } else if ((EP & 64) == 0 || (g.epilogue & (8192 | 1048576))) {
        _Pragma("clang loop unroll(full)")
        for (int a = 0; a < FR; ++a)
          _Pragma("clang loop unroll(full)")
          for (int b = 0; b < FC; ++b) LF_STORE_8X8(a, b, j0 + sr + a * 8, i0 + sc + b * 8);
        if constexpr ((EP & 128) == 0) return;
        stored8x8 = true;
      }
    }
  }
  if (!stored8x8) {
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a) {
    const int j = j0 + sr + a * 8;
    if (j >= g.n) continue;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) {
      const int i = i0 + sc + b * 8;
      if (i >= g.m) continue;
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
      if constexpr ((EP & 16) == 0 && (EP & 64) != 0) {
        if constexpr (metal::is_same_v<TC, half>) {
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
        if constexpr (metal::is_same_v<TC, float>) {
          if ((g.epilogue & 16384) && j + 8 <= g.n && j + 8 <= g.tgPairs && i + 8 <= g.m) {
            simdgroup_matrix<float, 8, 8> om, cm;
            simdgroup_load(cm, C + (ulong)j * g.ldc + i, (ulong)g.ldc);
            thread auto& o = om.thread_elements();
            thread auto& c = cm.thread_elements();
            const uint jr = (uint)(j + sm), r = jr < (uint)g.tgN ? jr : jr - lf_udiv(jr, (uint)g.tgN) * (uint)g.tgN;
            const float rm = g.aux2 ? ((device const float*)g.aux2)[r] : 1.f;
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < 2; ++t) {
              const float gp = ((device const float*)g.aux)[(ulong)r * g.ldaux + (i + sn + t)];
              const float gm = g.tgC ? gp : (1.f / (1.f + exp(-gp)));
              float res = c[t] + g.alpha * (float)e[t] * gm;
              if (g.aux2) res *= rm;
              o[t] = res;
            }
            simdgroup_store(om, D + (ulong)j * g.ldd + i, (ulong)g.ldd);
            if (g.aux3) {
              simdgroup_matrix<half, 8, 8> hm;
              thread auto& h = hm.thread_elements();
              h[0] = (half)clamp((float)o[0], -65504.f, 65504.f); h[1] = (half)clamp((float)o[1], -65504.f, 65504.f);
              simdgroup_store(hm, (device half*)g.aux3 + (ulong)j * g.ldd + i, (ulong)g.ldd);
            }
            continue;
          }
          if ((g.epilogue & 262144) && j + 8 <= g.n && i + 8 <= g.m) {
            simdgroup_matrix<float, 8, 8> om, cm;
            simdgroup_load(cm, C + (ulong)j * g.ldc + i, (ulong)g.ldc);
            thread auto& o = om.thread_elements();
            thread auto& c = cm.thread_elements();
            device const int* mp = (device const int*)g.aux2;
            const int jr = j + sm;
            const ulong row = mp ? (ulong)(uint)mp[jr] : (ulong)jr;
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < 2; ++t) {
              const float gm = ((device const float*)g.aux)[row * g.ldaux + (i + sn + t)];
              o[t] = c[t] + gm * (g.alpha * (float)e[t]);
            }
            simdgroup_store(om, D + (ulong)j * g.ldd + i, (ulong)g.ldd);
            continue;
          }
          if ((g.epilogue & 524288) && (g.tgC & 7) == 0 && j + 8 <= g.n && j + 8 <= g.tgPairs && i + 8 <= g.m) {
            const uint W = (uint)g.tgC, N = (uint)g.tgN, BS = (uint)g.m / W;
            const uint k0 = lf_udiv((uint)j, N), r0 = (uint)j - k0 * N;
            if (r0 + 8 <= N) {
              const uint bs = lf_udiv((uint)i, W), col0 = (uint)i - bs * W;
              simdgroup_matrix<float, 8, 8> om;
              thread auto& o = om.thread_elements();
              _Pragma("clang loop unroll(full)")
              for (int t = 0; t < 2; ++t) {
                const int ic = i + sn + t, col = (int)col0 + sn + t;
                float v = g.alpha * (float)e[t];
                if (g.epilogue & 4) v += ((device const float*)g.bias)[ic];
                if (col < g.tgNp) v = 1.f / (1.f + exp(-v));
                o[t] = v;
              }
              simdgroup_store(om, D + (((ulong)k0 * BS + bs) * N + r0) * W + col0, (ulong)W);
              continue;
            }
          }
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
              const uint r = (uint)jr < (uint)g.tgN ? (uint)jr : (uint)jr - lf_udiv((uint)jr, (uint)g.tgN) * (uint)g.tgN;
              const float gp = ((device const float*)g.aux)[(ulong)r * g.ldaux + ic];
              const float gm = g.tgC ? gp : (1.f / (1.f + exp(-gp)));
              float res = lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc) + v * gm;
              if (g.aux2) res *= ((device const float*)g.aux2)[r];
              lf_st(D, (ulong)ic + (ulong)jr * g.ldd, res);
              if (g.aux3) ((device half*)g.aux3)[(ulong)ic + (ulong)jr * g.ldd] = (half)clamp(res, -65504.f, 65504.f);
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
            float cv = lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc);
            if constexpr ((EP & 8) == 0 && !metal::is_same_v<TC, float>) {
              if (g.biasType & 512) cv = ((device const float*)g.C)[(ulong)ic + (ulong)jr * g.ldc];
            }
            const uint opmC = (uint)g.tgN, u = (uint)ic;
            const uint jTok = opmC == 128 ? (u >> 7) : opmC == 256 ? (u >> 8) : lf_udiv(u, opmC);
            const float bCol = ((device const float*)g.bias)[u - jTok * opmC];
            const float nv = ((device const float*)g.aux)[(uint)jr * (uint)g.tgNp + jTok];
            v = cv + (g.tgC ? v / max(nv, 1.f) + bCol : (bCol + v) / (1e-3f + nv));
            lf_st(D, (ulong)ic + (ulong)jr * g.ldd, v);
            if constexpr ((EP & 128) != 0) e[t] = metal::is_same_v<TC, half> ? (float)(half)clamp(v, -65504.f, 65504.f) : v;
            continue;
          }
        }
        if (g.epilogue & 64) {      // a gated residual: the GEMM is the gate, aux the gated values, D the residual
          if (g.epilogue & 4) v += ((device const float*)g.bias)[ic];      // (the gate's bias, f32: AF2's)
          float cv = lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc);
          if constexpr ((EP & 8) == 0 && !metal::is_same_v<TC, float>) {
            if (g.biasType & 512) cv = ((device const float*)g.C)[(ulong)ic + (ulong)jr * g.ldc];
          }
          v = cv + (float)((device const half*)g.aux)[(ulong)ic + (ulong)jr * g.ldaux] * (1.f / (1.f + exp(-v)));
          lf_st(D, (ulong)ic + (ulong)jr * g.ldd, v);
          if constexpr ((EP & 128) != 0) e[t] = metal::is_same_v<TC, half> ? (float)(half)clamp(v, -65504.f, 65504.f) : v;
          continue;
        }
        if (g.beta != 0.f) {
          float cv = lf_ldf(C + (ulong)ic + (ulong)jr * g.ldc);
          if constexpr ((EP & 8) == 0 && !metal::is_same_v<TC, float>) {
            if (g.biasType & 512) cv = ((device const float*)g.C)[(ulong)ic + (ulong)jr * g.ldc];
          }
          v += g.beta * cv;
        }
        if (g.epilogue & 4) {
          const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
          v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + ic] : ((device const float*)g.bias)[boff + ic];
        }
        if (g.epilogue & 2) v = max(v, 0.f);
        if (g.epilogue & 32) v = 0.5f * v * (1.f + lf_erf(v * 0.70710678118654752f));   // (GELU, erf's)
        lf_st(D, (ulong)ic + (ulong)jr * g.ldd, v);
        if constexpr ((EP & 128) != 0) e[t] = metal::is_same_v<TC, half> ? (float)(half)clamp(v, -65504.f, 65504.f) : v;
      }
    }
  }
  }
  if constexpr ((EP & 128) != 0) {
    if (g.lnOut) {
      threadgroup float* part = (threadgroup float*)tgm;
      const bool leader = ((lane & 9) == 0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        float s = 0.f;
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) {
          thread auto& e = acc[a][b].thread_elements();
          s += (float)e[0] + (float)e[1];
        }
        s += simd_shuffle_xor(s, 1);
        s += simd_shuffle_xor(s, 8);
        if (leader) part[(sg & 1) * 64 + sr + a * 8 + sm] = s;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float mean[FR];
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        const int r = sr + a * 8 + sm;
        mean[a] = (part[r] + part[64 + r]) * (1.f / (float)TC_);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        const float m = mean[a];
        float v = 0.f;
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) {
          thread auto& e = acc[a][b].thread_elements();
          const float d0 = (float)e[0] - m, d1 = (float)e[1] - m;
          v += d0 * d0 + d1 * d1;
        }
        v += simd_shuffle_xor(v, 1);
        v += simd_shuffle_xor(v, 8);
        if (leader) part[(sg & 1) * 64 + sr + a * 8 + sm] = v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      device half* lnOut = (device half*)g.lnOut;
      device const float* lnSc = (device const float*)g.lnScale;
      device const float* lnOf = (device const float*)g.lnOffset;
      device half* lnOut2 = (device half*)g.lnOut2;
      device const float* lnSc2 = (device const float*)g.lnScale2;
      device const float* lnOf2 = (device const float*)g.lnOffset2;
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        const int r = sr + a * 8 + sm, jr = j0 + r;
        if (jr >= g.n) continue;
        const float m = mean[a];
        const float inv = rsqrt((part[r] + part[64 + r]) * (1.f / (float)TC_) + g.lnEps);
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) {
          thread auto& e = acc[a][b].thread_elements();
          const int col0 = sc + b * 8 + sn, ic0 = i0 + col0, sc0 = (g.epilogue & 8192) ? (col0 & 127) : col0;
          const float n0 = ((float)e[0] - m) * inv, n1 = ((float)e[1] - m) * inv;
          const float s0 = lnSc ? lnSc[sc0] : 1.f, s1 = lnSc ? lnSc[sc0 + 1] : 1.f;
          const float o0 = lnOf ? lnOf[sc0] : 0.f, o1 = lnOf ? lnOf[sc0 + 1] : 0.f;
          *(device half2*)(lnOut + (ulong)jr * g.ldd + ic0) = half2((half)(n0 * s0 + o0), (half)(n1 * s1 + o1));
          if (lnOut2) {
            const float t0 = lnSc2 ? lnSc2[sc0] : 1.f, t1 = lnSc2 ? lnSc2[sc0 + 1] : 1.f;
            const float p0 = lnOf2 ? lnOf2[sc0] : 0.f, p1 = lnOf2 ? lnOf2[sc0 + 1] : 0.f;
            *(device half2*)(lnOut2 + (ulong)jr * g.ldd + ic0) = half2((half)(n0 * t0 + p0), (half)(n1 * t1 + p1));
          }
        }
      }
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

// Dual gated residual GEMM for simdgroup_matrix (Apple8/Apple9): D += (Xv @ Wv + biasV) * sigmoid(Xg @ Wg + biasG),
// optionally emitting LayerNorm(D) into g.lnOut when TC_ covers all columns (g.m == TC_).
// Runs Xv @ Wv first into threadgroup Vs (single accumulator set acc[FR][FC], <= 14.25 KB threadgroup memory),
// then Xg @ Wg into acc[FR][FC], fusing the gated residual and optional LayerNorm without a global vTmp round-trip.
template <typename TC, int TR, int TC_, int SGR, int SGC, bool LN>
kernel void lf_gemm_dual(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 16, PAD = 8, XROW = BK + PAD, WROW = TC_ + PAD;
  constexpr int WR = TR / SGR, WC = TC_ / SGC, FR = WR / 8, FC = WC / 8;
  threadgroup half Xs[TR * XROW];
  threadgroup half Ws[BK * WROW];
  threadgroup half Vs[TR * WROW];
  device const half* Wg = (device const half*)g.A;
  device const half* Xg = (device const half*)g.B;
  device const half* Xv = (device const half*)g.aux;
  device const half* Wv = (device const half*)g.aux2;
  device const TC* C = (device const TC*)g.C;
  device TC* D = (device TC*)g.D;
  const int j0 = grp.y * TR, i0 = grp.x * TC_;
  if (j0 >= g.n || i0 >= g.m) return;
  const int sr = (sg / SGC) * WR, sc = (sg % SGC) * WC;
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  const int xr = tid >> 2, xc = (tid & 3) * 4;
  const int wr = tid >> 3, wc = (tid & 7) * (TC_ / 8);
  const bool xValid = j0 + xr < g.n;

  simdgroup_float8x8 acc[FR][FC];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) acc[a][b] = simdgroup_float8x8(0.f);

  for (int k0 = 0; k0 < g.k; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup half4*)(Xs + xr * XROW + xc) = xValid ? *(device const half4*)(Xv + (ulong)(j0 + xr) * g.ldaux + k0 + xc) : half4(0);
    _Pragma("clang loop unroll(full)")
    for (int u = 0; u < TC_ / 64; ++u)
      *(threadgroup uint4*)(Ws + wr * WROW + wc + u * 8) = *(device const uint4*)(Wv + (ulong)(k0 + wr) * g.lda + i0 + wc + u * 8);
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
  device const float* biasV = (device const float*)g.aux3;
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < FC; ++b) {
    const float2 bv = biasV ? *(device const float2*)(biasV + i0 + sc + b * 8 + sn) : float2(0.f);
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < FR; ++a) {
      simdgroup_half8x8 vm;
      thread auto& ev = acc[a][b].thread_elements();
      thread auto& v = vm.thread_elements();
      v[0] = (half)clamp(ev[0] + bv[0], -65504.f, 65504.f);
      v[1] = (half)clamp(ev[1] + bv[1], -65504.f, 65504.f);
      simdgroup_store(vm, Vs + (sr + a * 8) * WROW + sc + b * 8, WROW);
      acc[a][b] = simdgroup_float8x8(0.f);
    }
  }

  for (int k0 = 0; k0 < g.k; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup half4*)(Xs + xr * XROW + xc) = xValid ? *(device const half4*)(Xg + (ulong)(j0 + xr) * g.ldb + k0 + xc) : half4(0);
    _Pragma("clang loop unroll(full)")
    for (int u = 0; u < TC_ / 64; ++u)
      *(threadgroup uint4*)(Ws + wr * WROW + wc + u * 8) = *(device const uint4*)(Wg + (ulong)(k0 + wr) * g.lda + i0 + wc + u * 8);
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

  device const float* biasG = (device const float*)g.bias;
  float2 bg2[FC];
  _Pragma("clang loop unroll(full)")
  for (int b = 0; b < FC; ++b) {
    const int col0 = i0 + sc + b * 8 + sn;
    bg2[b] = biasG ? *(device const float2*)(biasG + col0) : float2(0.f);
  }
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a) {
    const int j = j0 + sr + a * 8;
    if (j >= g.n) continue;
    if (j + 8 <= g.n) {
      _Pragma("clang loop unroll(full)")
      for (int b = 0; b < FC; ++b) {
        const int i = i0 + sc + b * 8;
        simdgroup_matrix<TC, 8, 8> om;
        simdgroup_half8x8 vm;
        simdgroup_load(om, C + (ulong)j * g.ldc + i, (ulong)g.ldc);
        simdgroup_load(vm, Vs + (sr + a * 8) * WROW + sc + b * 8, WROW);
        thread auto& o = om.thread_elements();
        thread auto& eg = acc[a][b].thread_elements();
        thread auto& ev = vm.thread_elements();
        _Pragma("clang loop unroll(full)")
        for (int t = 0; t < 2; ++t) {
          const float gval = eg[t] + bg2[b][t];
          const float res = (float)o[t] + (float)ev[t] * (1.f / (1.f + exp(-gval)));
          if constexpr (metal::is_same_v<TC, half>) {
            const half hv = (half)clamp(res, -65504.f, 65504.f);
            o[t] = hv;
            if constexpr (LN) eg[t] = (float)hv;
          } else {
            o[t] = res;
            if constexpr (LN) eg[t] = res;
          }
        }
        simdgroup_store(om, D + (ulong)j * g.ldd + i, (ulong)g.ldd);
      }
      continue;
    }
    const int jr = j + sm;
    if (jr >= g.n) continue;
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) {
      const int i = i0 + sc + b * 8;
      simdgroup_half8x8 vm;
      simdgroup_load(vm, Vs + (sr + a * 8) * WROW + sc + b * 8, WROW);
      thread auto& eg = acc[a][b].thread_elements();
      thread auto& ev = vm.thread_elements();
      _Pragma("clang loop unroll(full)")
      for (int t = 0; t < 2; ++t) {
        const int ic = i + sn + t;
        const float gval = eg[t] + bg2[b][t];
        const float res = lf_ldf(C + (ulong)jr * g.ldc + ic) + (float)ev[t] * (1.f / (1.f + exp(-gval)));
        if constexpr (metal::is_same_v<TC, half>) {
          const half hv = (half)clamp(res, -65504.f, 65504.f);
          D[(ulong)jr * g.ldd + ic] = hv;
          if constexpr (LN) eg[t] = (float)hv;
        } else {
          D[(ulong)jr * g.ldd + ic] = res;
          if constexpr (LN) eg[t] = res;
        }
      }
    }
  }
  if constexpr (LN) {
    if (g.lnOut) {
      threadgroup float* part = (threadgroup float*)Xs;
      const bool leader = ((lane & 9) == 0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        float s = 0.f;
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) {
          thread auto& eg = acc[a][b].thread_elements();
          s += eg[0] + eg[1];
        }
        s += simd_shuffle_xor(s, 1);
        s += simd_shuffle_xor(s, 8);
        if (leader) part[(sg % SGC) * TR + sr + a * 8 + sm] = s;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float mean[FR];
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        const int r = sr + a * 8 + sm;
        float sum = 0.f;
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < SGC; ++c) sum += part[c * TR + r];
        mean[a] = sum * (1.f / (float)TC_);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        const float m = mean[a];
        float v = 0.f;
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) {
          thread auto& eg = acc[a][b].thread_elements();
          const float d0 = eg[0] - m, d1 = eg[1] - m;
          v += d0 * d0 + d1 * d1;
        }
        v += simd_shuffle_xor(v, 1);
        v += simd_shuffle_xor(v, 8);
        if (leader) part[(sg % SGC) * TR + sr + a * 8 + sm] = v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      device half* lnOut = (device half*)g.lnOut;
      device const float* lnSc = (device const float*)g.lnScale;
      device const float* lnOf = (device const float*)g.lnOffset;
      _Pragma("clang loop unroll(full)")
      for (int a = 0; a < FR; ++a) {
        const int r = sr + a * 8 + sm, jr = j0 + r;
        if (jr >= g.n) continue;
        const float m = mean[a];
        float vsum = 0.f;
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < SGC; ++c) vsum += part[c * TR + r];
        const float inv = rsqrt(vsum * (1.f / (float)TC_) + g.lnEps);
        _Pragma("clang loop unroll(full)")
        for (int b = 0; b < FC; ++b) {
          thread auto& eg = acc[a][b].thread_elements();
          const int col0 = sc + b * 8 + sn, ic0 = i0 + col0;
          const float n0 = (eg[0] - m) * inv, n1 = (eg[1] - m) * inv;
          const float s0 = lnSc ? lnSc[col0] : 1.f, s1 = lnSc ? lnSc[col0 + 1] : 1.f;
          const float o0 = lnOf ? lnOf[col0] : 0.f, o1 = lnOf ? lnOf[col0 + 1] : 0.f;
          *(device half2*)(lnOut + (ulong)jr * g.ldd + ic0) = half2((half)(n0 * s0 + o0), (half)(n1 * s1 + o1));
        }
      }
    }
  }
}
