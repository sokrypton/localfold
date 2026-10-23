// metal/core's GEMM on the matrix units (Apple10, the M5's "neural accelerators"): Metal 4's tensor operations
// (MetalPerformancePrimitives' matmul2d), the same arguments and the same epilogues as gemm.metal's lf_gemm. Compiled
// at Metal Shading Language 4.0 in a library of its own (core.mm: tensor instances) and chosen at run time - an older
// GPU or OS keeps lf_gemm. (args.h is prepended by metal/build.sh)
//
// The product is matmul2d's, into a cooperative tensor of the accumulator's type (half where EP bit 8 asks, as lf_gemm's
// all-half instances accumulate, else float). A plain product - a bias, a ReLU, alpha 1 and beta 0, the output in the
// accumulator's type or converted on the way out - is adjusted in place and stored from the tensor; every other epilogue stages the tile in
// threadgroup memory and applies lf_gemm's arithmetic to it an element at a time, coalesced along the output's row.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
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
inline float lf_ldf(device const float* p) { return *p; }
inline float lf_ldf(device const half* p) { return (float)*p; }

// the next LayerNorm, emitted by a tile's epilogue that holds whole rows (EP bit 64; lf_gemm_tensor_dual's lnOut): a
// 64 x 128 or a 32 x 256 float destination over 4 simdgroups, its new values read back from D where each
// thread wrote its own (4 adjacent columns an entry); g.lnOut [rows][TN] half (and g.lnOut2, a second norm of the same rows).
// As probed on the M5: a row is 4 lanes' (lane ^ 1, ^ 8) in each of TN / 64 simdgroups (64 x 128: sg ^ 1; 32 x 256:
// all four), 16 columns a lane; element e's row + 8 at e bit 2, the other row pair at bit 5. Summed in the lane,
// across the 4 lanes by shuffles, then the simdgroups' parts through threadgroup memory in a fixed order -
// deterministic. part: [2 statistics][4 simdgroups][64 rows], 512 floats
// a thread's tile values at element e, read back from D where it just wrote them (cheaper than holding 16 float4s) - in
// D's own type, so a half output (the half pair) is normalised as it was stored
template <bool REG = false, typename CT, typename TD>
inline float4 lf_ln_at(thread CT& cg, thread CT& cv, device const TD* D, constant GemmArgs& g, int i0, int j0, uint16_t e) {
  if constexpr (REG) return float4((float)cg[e], (float)cg[e + 1], (float)cg[e + 2], (float)cg[e + 3]);
  return float4(as_type<float>(half2(cg[e], cv[e])),
                as_type<float>(half2(cg[e + 1], cv[e + 1])),
                as_type<float>(half2(cg[e + 2], cv[e + 2])),
                as_type<float>(half2(cg[e + 3], cv[e + 3])));
}
template <int TN, bool REG = false, int NR = 4, typename CT, typename TD>
inline void lf_emit_ln(thread CT& cg, thread CT& cv, device const TD* D, threadgroup float* part, uint tid, constant GemmArgs& g, int i0, int j0) {
  constexpr uint NS = TN == 64 ? 2 : TN / 64;
  static_assert(TN == 64 || TN == 128 || TN == 256, "the emitted LayerNorm is probed for TN = 64, 128 and 256");
  const uint sg = tid / 32, first = sg & ~(NS - 1), lane = tid & 31;
  auto qOf = [](uint16_t e) { return TN == 64 ? (((e >> 2) & 1) | ((e >> 3) & 2)) : (((e >> 2) & 1) | ((e >> 4) & 2)); };
  auto c0Of = [](uint16_t e) { return TN == 64 ? (((e >> 3) & 1) == 0) : (((e >> 3) & 3) == 0); };
  float s4[NR];
  for (int q = 0; q < NR; ++q) s4[q] = 0;
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); e += 4) { const float4 v = lf_ln_at<REG>(cg, cv, D, g, i0, j0, e); s4[qOf(e)] += v.x + v.y + v.z + v.w; }
  int rowOf[NR];
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); e += 4) if (c0Of(e)) rowOf[qOf(e)] = cg.get_multidimensional_index(e)[1];
  for (int q = 0; q < NR; ++q) {
    s4[q] += simd_shuffle_xor(s4[q], 1);
    s4[q] += simd_shuffle_xor(s4[q], 8);
    if ((lane & 9) == 0) part[(0 * 4 + sg) * 64 + rowOf[q]] = s4[q];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean[NR];
  for (int q = 0; q < NR; ++q) {
    float t = 0;
    for (uint k = 0; k < NS; ++k) t += part[(0 * 4 + first + k) * 64 + rowOf[q]];
    mean[q] = t / TN;
  }
  float v4[NR];
  for (int q = 0; q < NR; ++q) v4[q] = 0;
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); e += 4) {
    const int q = qOf(e);
    const float4 d = lf_ln_at<REG>(cg, cv, D, g, i0, j0, e) - mean[q];
    v4[q] += dot(d, d);
  }
  for (int q = 0; q < NR; ++q) {
    v4[q] += simd_shuffle_xor(v4[q], 1);
    v4[q] += simd_shuffle_xor(v4[q], 8);
    if ((lane & 9) == 0) part[(1 * 4 + sg) * 64 + rowOf[q]] = v4[q];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[NR];
  for (int q = 0; q < NR; ++q) {
    float t = 0;
    for (uint k = 0; k < NS; ++k) t += part[(1 * 4 + first + k) * 64 + rowOf[q]];
    inv[q] = rsqrt(t / TN + g.lnEps);
  }
  device half* out = (device half*)g.lnOut;
  device half* out2 = (device half*)g.lnOut2;
  device const float* sc = (device const float*)g.lnScale; device const float* of = (device const float*)g.lnOffset;
  device const float* sc2 = (device const float*)g.lnScale2; device const float* of2 = (device const float*)g.lnOffset2;
  constexpr int NC = TN == 64 ? 2 : 4;
  constexpr int CSTEP = TN == 256 ? 64 : 32;
  const int col0 = cg.get_multidimensional_index((uint16_t)0)[0];
  float4 sc4[NC], of4[NC], sc2_4[NC], of2_4[NC];
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < NC; ++k) {
    const int col = col0 + k * CSTEP;
    sc4[k] = sc ? *(device const float4*)(sc + col) : float4(1.f);
    of4[k] = of ? *(device const float4*)(of + col) : float4(0.f);
    if (out2) {
      sc2_4[k] = sc2 ? *(device const float4*)(sc2 + col) : float4(1.f);
      of2_4[k] = of2 ? *(device const float4*)(of2 + col) : float4(0.f);
    }
  }
  _Pragma("clang loop unroll(full)")
  for (int q = 0; q < NR; ++q) {
    const int j = j0 + rowOf[q];
    if (j >= g.n) continue;
    const uint16_t e0 = (NR == 4 && TN == 64) ? (uint16_t)((q & 1) * 4 + (q >> 1) * 16)
                                              : (uint16_t)((q & 1) * 4 + (q >> 1) * 32);
    const ulong rowOff = (ulong)j * g.ldd + (ulong)i0 + (ulong)col0;
    const float mq = mean[q], ivq = inv[q];
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < NC; ++k) {
      const uint16_t e = (uint16_t)(e0 + k * 8);
      const float4 x = (lf_ln_at<REG>(cg, cv, D, g, i0, j0, e) - mq) * ivq;
      float4 y = x;
      if (sc) y *= sc4[k];
      if (of) y += of4[k];
      *(device half4*)(out + rowOff + k * CSTEP) = half4(y);
      if (out2) {
        float4 y2 = x;
        if (sc2) y2 *= sc2_4[k];
        if (of2) y2 += of2_4[k];
        *(device half4*)(out2 + rowOff + k * CSTEP) = half4(y2);
      }
    }
  }
}

// a vector epilogue's 4 results stored in the output's type - a half output saturating at half's largest finite value
// (the half pair: a residual past it is clamped, not turned into infinity)
template <typename T> inline vec<T, 4> lf_out4(float4 v) { return vec<T, 4>(v); }
template <> inline half4 lf_out4<half>(float4 v) { return half4(clamp(v, -65504.f, 65504.f)); }

// Operands as matmul2d sees them: left X = op(B)^T (n x k), right W = op(A)^T (k x m), destination D (n x m), each a
// tensor over the GEMM's own pointers - extents innermost first, the leading dimension a stride, a transposed operand
// the descriptor's transpose flag over its stored layout. A tile is TM rows of D (n) by TN columns (m), SG simdgroups.
template <typename TA, typename TB, typename TC, int TM, int TN, bool TRA, bool TRB, int EP = 0>
kernel void lf_gemm_tensor(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
  using namespace mpp::tensor_ops;
  constexpr int SG = 4, NT = 32 * SG;
  typedef metal::conditional_t<(EP & 8) != 0, half, float> TACC;
  const uint z = grp.z;
  device TA* A; device TB* B; device TC* D; device const TC* C;
  if (g.ptrs) {
    A = (device TA*)((device const ulong*)g.A)[z];
    B = (device TB*)((device const ulong*)g.B)[z];
    D = (device TC*)((device const ulong*)g.D)[z];
    C = (device const TC*)D;
  } else {
    A = (device TA*)g.A + (long)z * g.sa;
    B = (device TB*)g.B + (long)z * g.sb;
    D = (device TC*)g.D + (long)z * g.sd;
    C = (device const TC*)g.C + (long)z * g.sc;
  }
  const int j0 = grp.y * TM, i0 = grp.x * TN;
  typedef dextents<int32_t, 2> E2;
  // X: element (j, k) = TRB ? B[j + k ldb] : B[k + j ldb]
  const int nX = (!TRB && g.n <= 128 && (g.k & 15) == 0 && j0 + TM > g.n && (ulong)(j0 + TM - g.n) * (ulong)g.ldb * sizeof(TB) <= 65536u) ? j0 + TM : g.n;
  tensor<device TB, E2, tensor_inline> tX(B, TRB ? E2(g.n, g.k) : E2(g.k, nX), array<int32_t, 2>{1, g.ldb});
  // W: element (k, i) = TRA ? A[k + i lda] : A[i + k lda]
  tensor<device TA, E2, tensor_inline> tW(A, TRA ? E2(g.k, g.m) : E2(g.m, g.k), array<int32_t, 2>{1, g.lda});
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), TRB, TRA, false);
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto mX = TRB ? tX.slice(j0, 0) : tX.slice(0, j0);
  auto mW = TRA ? tW.slice(0, i0) : tW.slice(i0, 0);
  auto c = op.template get_destination_cooperative_tensor<decltype(mX), decltype(mW), TACC>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < c.get_capacity(); ++e) if (c.is_valid_element(e)) c[e] = TACC(0);
  op.run(mX, mW, c);

  // EP bit 16, the triangle's gate in registers: the weight in quarters (core's lf_tri_quarters - a 128-column tile 32
  // channels' pa, ga, pb, gb), and a thread holds 4 adjacent columns every 32 on its rows, element e's column + 32 at
  // e + 8 (matmul2d's 64 x 128 destination over 4 simdgroups, as probed on the M5; the selftest checks it)
  if constexpr ((EP & 16) != 0) {
    static_assert(TM == 64 && TN == 128, "the register gate is laid out for 64 x 128 tiles");
    device const float* mask = (device const float*)g.aux;
    device half* outA = (device half*)g.aux2; device half* outB = (device half*)g.aux3;
    const int col = c.get_multidimensional_index((uint16_t)0)[0], ch0 = (i0 >> 2) + col;
    if (ch0 >= g.tgC) return;
    float4 bpa(0.f), bga(0.f), bpb(0.f), bgb(0.f);
    if (g.epilogue & 4) {
      device const float* bias = (device const float*)g.bias + i0 + col;
      bpa = *(device const float4*)(bias);
      bga = *(device const float4*)(bias + 32);
      bpb = *(device const float4*)(bias + 64);
      bgb = *(device const float4*)(bias + 96);
    }
    const ushort4 bs(0, 4, 32, 36);
    _Pragma("clang loop unroll(full)")
    for (int r_idx = 0; r_idx < 4; ++r_idx) {
      const uint16_t e = bs[r_idx];
      const int jg = j0 + c.get_multidimensional_index(e)[1];
      if (jg >= g.n) continue;
      const float4 vpa = float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]) + bpa;
      const float4 vga = float4((float)c[e + 8], (float)c[e + 9], (float)c[e + 10], (float)c[e + 11]) + bga;
      const float4 vpb = float4((float)c[e + 16], (float)c[e + 17], (float)c[e + 18], (float)c[e + 19]) + bpb;
      const float4 vgb = float4((float)c[e + 24], (float)c[e + 25], (float)c[e + 26], (float)c[e + 27]) + bgb;
      const float m = mask[g.tgR0 + jg];
      const uint p = (uint)(g.tgR0 + jg);
      const ulong q = g.tgNp == g.tgN ? (ulong)p : (ulong)lf_udiv(p, (uint)g.tgN) * (uint)(g.tgNp - g.tgN) + p;
      const half4 ha = half4(fast::divide(vpa * m, 1.f + fast::exp(-vga)));
      const half4 hb = half4(fast::divide(vpb * m, 1.f + fast::exp(-vgb)));
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 4; ++k) {
        const ulong off = (ulong)(ch0 + k) * g.tgPairs + q;
        outA[off] = ha[k];
        outB[off] = hb[k];
      }
    }
    return;
  }
  // EP bit 32, SwiGLU in registers: columns in blocks of 8 (16m..16m+7 are a_8m.., 16m+8..16m+15 their b); e and e + 4
  // hold the same 4 columns on rows ix[1] and ix[1] + 8, while lane ^ 8 holds column ^ 8 (b for a) on the same rows.
  // One butterfly shuffle pairs all 32 lanes: even-block lanes compute row ix[1], odd-block lanes compute row ix[1] + 8
  if constexpr ((EP & 32) != 0) {
    const bool odd = (c.get_multidimensional_index((uint16_t)0)[0] & 8) != 0;
    if (j0 + TM <= g.n && i0 + TN <= g.m && !(g.epilogue & 4) && g.alpha == 1.f) {
      _Pragma("clang loop unroll(full)")
      for (uint16_t e = 0; e < c.get_capacity(); e += 8) {
        if (!c.is_valid_element(e)) continue;
        const auto ix = c.get_multidimensional_index(odd ? (uint16_t)(e + 4) : e);
        const float4 v0((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
        const float4 v1((float)c[e + 4], (float)c[e + 5], (float)c[e + 6], (float)c[e + 7]);
        const float4 partner = simd_shuffle_xor(odd ? v0 : v1, 8);
        const float4 va = odd ? partner : v0;
        const float4 vb = odd ? v1 : partner;
        const int i = i0 + (ix[0] & ~8);
        const int outCol = (i >> 4) * 8 + (i & 7);
        const float4 res = fast::divide(va, 1.f + fast::exp(-va)) * vb;
        *(device vec<TC, 4>*)(D + (ulong)(j0 + ix[1]) * g.ldd + outCol) = lf_out4<TC>(res);
      }
      return;
    }
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < c.get_capacity(); e += 8) {
      if (!c.is_valid_element(e)) continue;
      const auto ix = c.get_multidimensional_index(odd ? (uint16_t)(e + 4) : e);
      float4 v0 = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
      float4 v1 = g.alpha * float4((float)c[e + 4], (float)c[e + 5], (float)c[e + 6], (float)c[e + 7]);
      if ((g.epilogue & 4) && i0 + ix[0] < g.m) {
        const float4 bv = *(device const float4*)((device const float*)g.bias + i0 + ix[0]);
        v0 += bv; v1 += bv;
      }
      const float4 partner = simd_shuffle_xor(odd ? v0 : v1, 8);
      const float4 va = odd ? partner : v0;
      const float4 vb = odd ? v1 : partner;
      const int j = j0 + ix[1];
      const int i = i0 + (ix[0] & ~8);
      if (j < g.n && i + 8 < g.m) {
        const int outCol = (i >> 4) * 8 + (i & 7);
        const float4 res = fast::divide(va, 1.f + fast::exp(-va)) * vb;
        *(device vec<TC, 4>*)(D + (ulong)j * g.ldd + outCol) = lf_out4<TC>(res);
      }
    }
    return;
  }
  // EP bit 128, the general epilogues (alpha, beta, a bias, ReLU, GELU, the gated residual) in registers: a thread holds
  // 4 adjacent columns at e..e+3 (the 64 x 64 and 64 x 128 layouts probed on the M5), read and written 4 at a time -
  // the host's promise: the leading dimensions and pointers 4-element aligned, m a multiple of 4
  if constexpr ((EP & 128) != 0) {
    static_assert((TM % 16 == 0 && TN % 32 == 0),
                  "the vector epilogue requires 16-row by 32-column multiple tiles");
    static_assert((EP & 64) == 0 || ((TM * TN == 8192 || TM * TN == 4096) && (!metal::is_same_v<TACC, half> || (TM == 64 && TN == 128))),
                  "the emitted LayerNorm: 64 x 128, 32 x 256, or 16 x 256");
    if constexpr (metal::is_same_v<TC, float> && metal::is_same_v<TACC, half>) {
      if constexpr (TM == 64 && TN == 128) {
        constexpr int NR = 4;
        const int col0 = i0 + c.get_multidimensional_index((uint16_t)0)[0];
        const bool fullM = i0 + TN <= g.m;
        const int jLim = min(g.n, (int)g.tgPairs);
        const ushort4 bs(0, 4, 32, 36);
        if constexpr ((EP & 64) == 0) {
          _Pragma("clang loop unroll(full)")
          for (int r_idx = 0; r_idx < NR; ++r_idx) {
            const uint16_t er = bs[r_idx];
            const int j = j0 + c.get_multidimensional_index(er)[1];
            if (j >= jLim) continue;
            const uint r = (uint)j < (uint)g.tgN ? (uint)j : (uint)j - lf_udiv((uint)j, (uint)g.tgN) * (uint)g.tgN;
            const float rm = g.aux2 ? ((device const float*)g.aux2)[r] : 1.f;
            device const float* gp0 = (device const float*)g.aux + (ulong)r * g.ldaux;
            device const float* cp0 = C + (ulong)j * g.ldc;
            device float* dp0 = D + (ulong)j * g.ldd;
            device half* hp0 = g.aux3 ? (device half*)g.aux3 + (ulong)j * g.ldd : nullptr;
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < 4; ++k) {
              const int i = col0 + k * 32;
              if (fullM || i < g.m) {
                const uint16_t e = er + k * 8;
                const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
                const float4 gp = *(device const float4*)(gp0 + i);
                const float4 cv = *(device const float4*)(cp0 + i);
                const float4 gm = g.tgC ? gp : fast::divide(1.f, 1.f + fast::exp(-gp));
                float4 res = cv + v * gm;
                if (g.aux2) res *= rm;
                *(device float4*)(dp0 + i) = res;
                if (hp0) *(device half4*)(hp0 + i) = half4(res);
              }
            }
          }
          return;
        } else {
          threadgroup float part[512];
          const uint sg = tid >> 5, first = sg & ~1u, lane = tid & 31;
          float4 xv[NR][4];
          int j_row[NR];
          uint r_row[NR];
          _Pragma("clang loop unroll(full)")
          for (int r_idx = 0; r_idx < NR; ++r_idx) {
            const uint16_t er = bs[r_idx];
            const int j = j0 + c.get_multidimensional_index(er)[1];
            j_row[r_idx] = j;
            float s = 0.f;
            if (j < jLim) {
              const uint r = (uint)j < (uint)g.tgN ? (uint)j : (uint)j - lf_udiv((uint)j, (uint)g.tgN) * (uint)g.tgN;
              r_row[r_idx] = r;
              const float rm = g.aux2 ? ((device const float*)g.aux2)[r] : 1.f;
              device const float* gp0 = (device const float*)g.aux + (ulong)r * g.ldaux;
              device const float* cp0 = C + (ulong)j * g.ldc;
              device float* dp0 = D + (ulong)j * g.ldd;
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < 4; ++k) {
                const int i = col0 + k * 32;
                const uint16_t e = er + k * 8;
                const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
                const float4 gp = *(device const float4*)(gp0 + i);
                const float4 cv = *(device const float4*)(cp0 + i);
                const float4 gm = g.tgC ? gp : fast::divide(1.f, 1.f + fast::exp(-gp));
                float4 res = cv + v * gm;
                if (g.aux2) res *= rm;
                *(device float4*)(dp0 + i) = res;
                xv[r_idx][k] = res;
                s += res.x + res.y + res.z + res.w;
              }
            }
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 8);
            if ((lane & 9) == 0) part[(0 * 4 + sg) * 64 + (j - j0)] = s;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          float mean[NR];
          _Pragma("clang loop unroll(full)")
          for (int r_idx = 0; r_idx < NR; ++r_idx) {
            const int rtile = j_row[r_idx] - j0;
            const float m = (part[(0 * 4 + first) * 64 + rtile] + part[(0 * 4 + first + 1) * 64 + rtile]) * (1.f / 128.f);
            mean[r_idx] = m;
            float v = 0.f;
            if (j_row[r_idx] < jLim) {
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < 4; ++k) { const float4 d = xv[r_idx][k] - m; v += dot(d, d); }
            }
            v += simd_shuffle_xor(v, 1);
            v += simd_shuffle_xor(v, 8);
            if ((lane & 9) == 0) part[(1 * 4 + sg) * 64 + rtile] = v;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          float4 yv[NR][4];
          float sk_row[NR];
          _Pragma("clang loop unroll(full)")
          for (int r_idx = 0; r_idx < NR; ++r_idx) {
            sk_row[r_idx] = 0.f;
            const int j = j_row[r_idx];
            if (j >= jLim) continue;
            const int rtile = j - j0;
            const float inv = rsqrt((part[(1 * 4 + first) * 64 + rtile] + part[(1 * 4 + first + 1) * 64 + rtile]) * (1.f / 128.f) + g.lnEps);
            const ulong pr = (ulong)r_row[r_idx] * 128;
            device const float* sc0 = (device const float*)g.lnScale + pr;
            device const float* of0 = g.lnOffset ? (device const float*)g.lnOffset + pr : nullptr;
            device half* out0 = (device half*)g.lnOut + (ulong)j * 128;
            device const float* ksc0 = g.lnOut2 ? (device const float*)g.lnScale2 + pr : nullptr;
            device const float* kof0 = (g.lnOut2 && g.lnOffset2) ? (device const float*)g.lnOffset2 + pr : nullptr;
            device half* kout0 = g.lnOut2 ? (device half*)g.lnOut2 + (ulong)j * 128 : nullptr;
            const ulong fSpan = 4 * (ulong)g.tgN * 128;
            device const float* fsc0 = g.aux3 ? sc0 + fSpan : nullptr;
            device const float* fof0 = g.aux3 ? sc0 + fSpan + (ulong)g.tgN * 128 : nullptr;
            device half* fout0 = g.aux3 ? (device half*)g.aux3 + (ulong)j * 128 : nullptr;
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < 4; ++k) {
              const int i = col0 + k * 32;
              const float4 nrm = (xv[r_idx][k] - mean[r_idx]) * inv;
              const float4 y = *(device const float4*)(sc0 + i) * nrm + (of0 ? *(device const float4*)(of0 + i) : float4(0.f));
              *(device half4*)(out0 + i) = half4(y);
              if (fout0) {
                const float4 fy = *(device const float4*)(fsc0 + i) * nrm + *(device const float4*)(fof0 + i);
                *(device half4*)(fout0 + i) = half4(fy);
              }
              if (g.lnOut2) {
                if (!g.lnPad) {
                  const float4 ky = *(device const float4*)(ksc0 + i) * nrm + (kof0 ? *(device const float4*)(kof0 + i) : float4(0.f));
                  *(device half4*)(kout0 + i) = half4(ky);
                } else {
                  yv[r_idx][k] = y;
                  sk_row[r_idx] += y.x + y.y + y.z + y.w;
                }
              }
            }
          }
          if (g.lnOut2 && g.lnPad) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            _Pragma("clang loop unroll(full)")
            for (int r_idx = 0; r_idx < NR; ++r_idx) {
              float s = sk_row[r_idx];
              s += simd_shuffle_xor(s, 1);
              s += simd_shuffle_xor(s, 8);
              if ((lane & 9) == 0) part[(0 * 4 + sg) * 64 + (j_row[r_idx] - j0)] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float meank[NR];
            _Pragma("clang loop unroll(full)")
            for (int r_idx = 0; r_idx < NR; ++r_idx) {
              const int rtile = j_row[r_idx] - j0;
              const float mk = (part[(0 * 4 + first) * 64 + rtile] + part[(0 * 4 + first + 1) * 64 + rtile]) * (1.f / 128.f);
              meank[r_idx] = mk;
              float vk = 0.f;
              if (j_row[r_idx] < jLim) {
                _Pragma("clang loop unroll(full)")
                for (int k = 0; k < 4; ++k) { const float4 dk = yv[r_idx][k] - mk; vk += dot(dk, dk); }
              }
              vk += simd_shuffle_xor(vk, 1);
              vk += simd_shuffle_xor(vk, 8);
              if ((lane & 9) == 0) part[(1 * 4 + sg) * 64 + rtile] = vk;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            _Pragma("clang loop unroll(full)")
            for (int r_idx = 0; r_idx < NR; ++r_idx) {
              const int j = j_row[r_idx];
              if (j >= jLim) continue;
              const int rtile = j - j0;
              const float invk = rsqrt((part[(1 * 4 + first) * 64 + rtile] + part[(1 * 4 + first + 1) * 64 + rtile]) * (1.f / 128.f) + g.lnEps);
              const ulong pr = (ulong)r_row[r_idx] * 128;
              device const float* ksc0 = (device const float*)g.lnScale2 + pr;
              device const float* kof0 = g.lnOffset2 ? (device const float*)g.lnOffset2 + pr : nullptr;
              device half* kout0 = (device half*)g.lnOut2 + (ulong)j * 128;
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < 4; ++k) {
                const int i = col0 + k * 32;
                const float4 ky = *(device const float4*)(ksc0 + i) * ((yv[r_idx][k] - meank[r_idx]) * invk) +
                                  (kof0 ? *(device const float4*)(kof0 + i) : float4(0.f));
                *(device half4*)(kout0 + i) = half4(ky);
              }
            }
          }
          return;
        }
      }
      if constexpr (TM == 32 && TN == 64) {
        const int col0 = i0 + c.get_multidimensional_index((uint16_t)0)[0];
        const bool fullM = i0 + TN <= g.m;
        _Pragma("clang loop unroll(full)")
        for (int r_idx = 0; r_idx < 2; ++r_idx) {
          const uint16_t er = (uint16_t)(r_idx * 4);
          const int j = j0 + c.get_multidimensional_index(er)[1];
          if (j >= g.tgPairs) continue;
          const uint r = (uint)j < (uint)g.tgN ? (uint)j : (uint)j - lf_udiv((uint)j, (uint)g.tgN) * (uint)g.tgN;
          const float rm = g.aux2 ? ((device const float*)g.aux2)[r] : 1.f;
          device const float* gp0 = (device const float*)g.aux + (ulong)r * g.ldaux;
          device const float* cp0 = C + (ulong)j * g.ldc;
          device float* dp0 = D + (ulong)j * g.ldd;
          device half* hp0 = g.aux3 ? (device half*)g.aux3 + (ulong)j * g.ldd : nullptr;
          _Pragma("clang loop unroll(full)")
          for (int k = 0; k < 2; ++k) {
            const int i = col0 + k * 32;
            if (fullM || i < g.m) {
              const uint16_t e = er + k * 8;
              const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
              const float4 gp = *(device const float4*)(gp0 + i);
              const float4 cv = *(device const float4*)(cp0 + i);
              const float4 gm = g.tgC ? gp : fast::divide(1.f, 1.f + fast::exp(-gp));
              float4 res = cv + v * gm;
              if (g.aux2) res *= rm;
              *(device float4*)(dp0 + i) = res;
              if (hp0) *(device half4*)(hp0 + i) = half4(res);
            }
          }
        }
        return;
      }
      const bool fullM = i0 + TN <= g.m;
      const int jLim = min(g.n, (int)g.tgPairs);
      int prev_ix1 = -1;
      uint r = 0;
      float rm = 1.f;
      device const float* gp0 = nullptr;
      device const float* cp0 = nullptr;
      device float* dp0 = nullptr;
      device half* hp0 = nullptr;
      _Pragma("clang loop unroll(full)")
      for (int pass = 0; pass < 2; ++pass) {
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = (uint16_t)(pass * 4); e < c.get_capacity(); e += 8) {
          if (!c.is_valid_element(e)) continue;
          const auto ix = c.get_multidimensional_index(e);
          const int i = i0 + ix[0], j = j0 + ix[1];
          if (j >= jLim || (!fullM && i >= g.m)) continue;
          if (ix[1] != prev_ix1) {
            prev_ix1 = ix[1];
            r = (uint)j < (uint)g.tgN ? (uint)j : (uint)j - lf_udiv((uint)j, (uint)g.tgN) * (uint)g.tgN;
            rm = g.aux2 ? ((device const float*)g.aux2)[r] : 1.f;
            gp0 = (device const float*)g.aux + (ulong)r * g.ldaux;
            cp0 = C + (ulong)j * g.ldc;
            dp0 = D + (ulong)j * g.ldd;
            hp0 = g.aux3 ? (device half*)g.aux3 + (ulong)j * g.ldd : nullptr;
          }
          const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
          const float4 gp = *(device const float4*)(gp0 + i);
          const float4 cv = *(device const float4*)(cp0 + i);
          const float4 gm = g.tgC ? gp : fast::divide(1.f, 1.f + fast::exp(-gp));
          float4 res = cv + v * gm;
          if (g.aux2) res *= rm;
          *(device float4*)(dp0 + i) = res;
          if (hp0) *(device half4*)(hp0 + i) = half4(res);
        }
      }
      return;
    }
    if constexpr (metal::is_same_v<TC, half> && !metal::is_same_v<TACC, half> && (EP & 64) == 0) {
      if (g.epilogue & 131072) {   // the coupling combine on a blocks-of-2 weight (core.h's gemmCouplingCombine): gates
        // i / 2 and i / 2 + 1 here with their values; P and R | B also in blocks of 2
        if constexpr (TM == 64 && TN == 128) {
          const int col0 = i0 + c.get_multidimensional_index((uint16_t)0)[0];
          const bool fullM = i0 + TN <= g.m;
          const ushort4 bs(0, 4, 32, 36);
          int j_row[4], st_row[4];
          uint t_row[4];
          float inv_row[4], dq_row[4], sq_row[4], sa_row[4];
          _Pragma("clang loop unroll(full)")
          for (int r_idx = 0; r_idx < 4; ++r_idx) {
            const int j = j0 + c.get_multidimensional_index(bs[r_idx])[1];
            j_row[r_idx] = j;
            if (j < g.n) {
              device const float4* sr4 = (device const float4*)((device const float*)g.aux + (ulong)j * 8);
              const float4 s0 = sr4[0], s1 = sr4[1];
              inv_row[r_idx] = s0.x; dq_row[r_idx] = s0.z; sq_row[r_idx] = s0.w; sa_row[r_idx] = s1.y;
              st_row[r_idx] = (int)s1.z; t_row[r_idx] = (uint)s1.w;
            } else {
              inv_row[r_idx] = 0.f; dq_row[r_idx] = 0.f; sq_row[r_idx] = 0.f; sa_row[r_idx] = 0.f;
              st_row[r_idx] = -2; t_row[r_idx] = 0xffffffffu;
            }
          }
          _Pragma("clang loop unroll(full)")
          for (int k = 0; k < 4; ++k) {
            const int i = col0 + k * 32;
            if (fullM || i < g.m) {
              int prev_st = -2;
              uint prev_t = 0xffffffffu;
              float4 pv(0.f), qv(0.f), bv(0.f);
              _Pragma("clang loop unroll(full)")
              for (int r_idx = 0; r_idx < 4; ++r_idx) {
                const int j = j_row[r_idx];
                if (j >= g.n) continue;
                const int st = st_row[r_idx];
                const uint t = t_row[r_idx];
                if (st != prev_st) {
                  pv = st >= 0 ? sa_row[r_idx] * *(device const float4*)((device const float*)g.aux2 + (ulong)st * g.m + i) : float4(0.f);
                  prev_st = st;
                }
                if (t != prev_t) {
                  device const float* qb = (device const float*)g.aux3 + (ulong)t * 2 * g.m + i;
                  qv = *(device const float4*)(qb);
                  bv = *(device const float4*)(qb + g.m);
                  prev_t = t;
                }
                const uint16_t e = bs[r_idx] + k * 8;
                const float4 cv = float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
                const float4 uv = inv_row[r_idx] * (pv + sq_row[r_idx] * cv + dq_row[r_idx] * qv) + bv;
                *(device half2*)(D + (ulong)(i >> 1) + (ulong)j * g.ldd) = half2(fast::divide(uv.xy, 1.f + fast::exp(-uv.xy)) * uv.zw);
              }
            }
          }
          return;
        }
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
          if (!c.is_valid_element(e)) continue;
          const auto ix = c.get_multidimensional_index(e);
          const int i = i0 + ix[0], j = j0 + ix[1];
          if (j >= g.n || i >= g.m) continue;
          device const float4* sr4 = (device const float4*)((device const float*)g.aux + (ulong)j * 8);
          const float4 s0 = sr4[0], s1 = sr4[1];
          const float inv = s0.x, dq = s0.z, sq = s0.w, sa = s1.y;
          const int st = (int)s1.z;
          device const float* qb = (device const float*)g.aux3 + (ulong)(uint)s1.w * 2 * g.m;
          device const float* pr = (device const float*)g.aux2 + (ulong)max(st, 0) * g.m;
          const float4 pv = st >= 0 ? sa * *(device const float4*)(pr + i) : float4(0.f);
          const float4 qv = *(device const float4*)(qb + i);
          const float4 bv = *(device const float4*)(qb + g.m + i);
          const float4 cv = float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
          const float4 uv = inv * (pv + sq * cv + dq * qv) + bv;
          *(device half2*)(D + (ulong)(i >> 1) + (ulong)j * g.ldd) = half2(fast::divide(uv.xy, 1.f + fast::exp(-uv.xy)) * uv.zw);
        }
        return;
      }
    }
    if constexpr (metal::is_same_v<TC, float> && (EP & 64) == 0) {
      if (g.epilogue & 262144) {   // a gated residual by rows (core.h's gemmGatedRows): D = C + gate[map(j)][i] v
        device const int* mp = (device const int*)g.aux2;
        if constexpr (TM == 64 && TN == 128) {
          const int col0 = i0 + c.get_multidimensional_index((uint16_t)0)[0];
          const bool fullM = i0 + TN <= g.m;
          const ushort4 bs(0, 4, 32, 36);
          _Pragma("clang loop unroll(full)")
          for (int r_idx = 0; r_idx < 4; ++r_idx) {
            const uint16_t er = bs[r_idx];
            const int j = j0 + c.get_multidimensional_index(er)[1];
            if (j >= g.n) continue;
            const ulong row = mp ? (ulong)(uint)mp[j] : (ulong)j;
            device const float* gp0 = (device const float*)g.aux + row * g.ldaux;
            device const float* cp0 = C + (ulong)j * g.ldc;
            device float* dp0 = D + (ulong)j * g.ldd;
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < 4; ++k) {
              const int i = col0 + k * 32;
              if (fullM || i < g.m) {
                const uint16_t e = er + k * 8;
                const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
                const float4 gm = *(device const float4*)(gp0 + i);
                const float4 cv = *(device const float4*)(cp0 + i);
                *(device float4*)(dp0 + i) = cv + gm * v;
              }
            }
          }
          return;
        }
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
          if (!c.is_valid_element(e)) continue;
          const auto ix = c.get_multidimensional_index(e);
          const int i = i0 + ix[0], j = j0 + ix[1];
          if (j >= g.n || i >= g.m) continue;
          const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
          const ulong row = mp ? (ulong)(uint)mp[j] : (ulong)j;
          const float4 gm = *(device const float4*)((device const float*)g.aux + row * g.ldaux + i);
          const float4 cv = *(device const float4*)(C + (ulong)i + (ulong)j * g.ldc);
          *(device float4*)(D + (ulong)i + (ulong)j * g.ldd) = cv + gm * v;
        }
        return;
      }
      if (g.epilogue & 16384) {   // the diffusion's gated residual: D = C + v sigmoid(aux[(j mod tgN) ldaux + i]), j < tgPairs
        if constexpr (TM == 64 && TN == 128) {
          const int col0 = i0 + c.get_multidimensional_index((uint16_t)0)[0];
          const bool fullM = i0 + TN <= g.m;
          const int jLim = min(g.n, (int)g.tgPairs);
          const ushort4 bs(0, 4, 32, 36);
          _Pragma("clang loop unroll(full)")
          for (int r_idx = 0; r_idx < 4; ++r_idx) {
            const uint16_t er = bs[r_idx];
            const int j = j0 + c.get_multidimensional_index(er)[1];
            if (j >= jLim) continue;
            const uint r = (uint)j < (uint)g.tgN ? (uint)j : (uint)j - lf_udiv((uint)j, (uint)g.tgN) * (uint)g.tgN;
            const float rm = g.aux2 ? ((device const float*)g.aux2)[r] : 1.f;
            device const float* gp0 = (device const float*)g.aux + (ulong)r * g.ldaux;
            device const float* cp0 = C + (ulong)j * g.ldc;
            device float* dp0 = D + (ulong)j * g.ldd;
            device half* hp0 = g.aux3 ? (device half*)g.aux3 + (ulong)j * g.ldd : nullptr;
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < 4; ++k) {
              const int i = col0 + k * 32;
              if (fullM || i < g.m) {
                const uint16_t e = er + k * 8;
                const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
                const float4 gp = *(device const float4*)(gp0 + i);
                const float4 cv = *(device const float4*)(cp0 + i);
                const float4 gm = g.tgC ? gp : fast::divide(1.f, 1.f + fast::exp(-gp));
                float4 res = cv + v * gm;
                if (g.aux2) res *= rm;
                *(device float4*)(dp0 + i) = res;
                if (hp0) *(device half4*)(hp0 + i) = half4(res);
              }
            }
          }
          return;
        }
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
          if (!c.is_valid_element(e)) continue;
          const auto ix = c.get_multidimensional_index(e);
          const int i = i0 + ix[0], j = j0 + ix[1];
          if (j >= g.n || j >= g.tgPairs || i >= g.m) continue;
          const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
          const uint r = (uint)j < (uint)g.tgN ? (uint)j : (uint)j - lf_udiv((uint)j, (uint)g.tgN) * (uint)g.tgN;
          const float4 gp = *(device const float4*)((device const float*)g.aux + (ulong)r * g.ldaux + i);
          const float4 cv = *(device const float4*)(C + (ulong)i + (ulong)j * g.ldc);
          const float4 gm = g.tgC ? gp : fast::divide(1.f, 1.f + fast::exp(-gp));
          float4 res = cv + v * gm;
          if (g.aux2) res *= ((device const float*)g.aux2)[r];
          *(device float4*)(D + (ulong)i + (ulong)j * g.ldd) = res;
          if (g.aux3) *(device half4*)((device half*)g.aux3 + (ulong)i + (ulong)j * g.ldd) = half4(res);
        }
        return;
      }
      if (g.epilogue & 524288) {  // conditioning permuted store: [k * n + r][bs * W + col] -> [(k * BS + bs) * n + r][col]
        const uint W = (uint)g.tgC, N = (uint)g.tgN, BS = (uint)g.m / W;
        const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
        if constexpr (TM == 64 && TN == 128) {
          const int laneCol = c.get_multidimensional_index((uint16_t)0)[0];
          const int col0 = i0 + laneCol;
          const int jLim = min(g.n, (int)g.tgPairs);
          if ((W & 127u) == 0 && (g.tgNp & 127) == 0 && i0 + 128 <= g.m) {
            const uint bs_tile = lf_udiv((uint)i0, W);
            const uint tile_off = (uint)i0 - bs_tile * W;
            const uint col0_off = tile_off + (uint)laneCol;
            const bool doSig = (int)tile_off < g.tgNp;
            const bool hasBias = (g.epilogue & 4) != 0, halfBias = (g.biasType & 255) == 2;
            float4 bv[4];
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < 4; ++k) {
              const int i = col0 + k * 32;
              bv[k] = !hasBias ? float4(0.f)
                               : halfBias ? float4(*(device const half4*)((device const half*)g.bias + boff + i))
                                          : *(device const float4*)((device const float*)g.bias + boff + i);
            }
            const ushort4 rbs(0, 4, 32, 36);
            _Pragma("clang loop unroll(full)")
            for (int r_idx = 0; r_idx < 4; ++r_idx) {
              const uint16_t er = rbs[r_idx];
              const int j = j0 + c.get_multidimensional_index(er)[1];
              if (j >= jLim) continue;
              const uint kr = (uint)j < N ? 0u : lf_udiv((uint)j, N), rr = (uint)j - kr * N;
              device float* dp0 = D + (((ulong)kr * BS + bs_tile) * N + rr) * W + col0_off;
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < 4; ++k) {
                const uint16_t e = er + k * 8;
                float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]) + bv[k];
                if (doSig) v = fast::divide(1.f, 1.f + fast::exp(-v));
                *(device float4*)(dp0 + k * 32) = v;
              }
            }
            return;
          }
          uint bs_col[4], col_off[4];
          float4 bv[4];
          _Pragma("clang loop unroll(full)")
          for (int k = 0; k < 4; ++k) {
            const int i = col0 + k * 32;
            if (i < g.m) {
              const uint bsk = lf_udiv((uint)i, W);
              bs_col[k] = bsk;
              col_off[k] = (uint)i - bsk * W;
              if (g.epilogue & 4)
                bv[k] = (g.biasType & 255) == 2 ? float4(*(device const half4*)((device const half*)g.bias + boff + i))
                                                : *(device const float4*)((device const float*)g.bias + boff + i);
              else bv[k] = float4(0.f);
            }
          }
          const ushort4 rbs(0, 4, 32, 36);
          _Pragma("clang loop unroll(full)")
          for (int r_idx = 0; r_idx < 4; ++r_idx) {
            const uint16_t er = rbs[r_idx];
            const int j = j0 + c.get_multidimensional_index(er)[1];
            if (j >= jLim) continue;
            const uint kr = lf_udiv((uint)j, N), rr = (uint)j - kr * N;
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < 4; ++k) {
              const int i = col0 + k * 32;
              if (i < g.m) {
                const uint16_t e = er + k * 8;
                float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
                if (g.epilogue & 4) v += bv[k];
                const uint col = col_off[k];
                if ((int)col < g.tgNp) v = fast::divide(1.f, 1.f + fast::exp(-v));
                *(device float4*)(D + (((ulong)kr * BS + bs_col[k]) * N + rr) * W + col) = v;
              }
            }
          }
          return;
        }
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
          if (!c.is_valid_element(e)) continue;
          const auto ix = c.get_multidimensional_index(e);
          const int i = i0 + ix[0], j = j0 + ix[1];
          if (j >= g.n || j >= g.tgPairs || i >= g.m) continue;
          float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
          if (g.epilogue & 4)
            v += (g.biasType & 255) == 2 ? float4(*(device const half4*)((device const half*)g.bias + boff + i))
                                         : *(device const float4*)((device const float*)g.bias + boff + i);
          const uint k = lf_udiv((uint)j, N), r = (uint)j - k * N;
          const uint bs = lf_udiv((uint)i, W), col = (uint)i - bs * W;
          if ((int)col < g.tgNp) v = fast::divide(1.f, 1.f + fast::exp(-v));
          *(device float4*)(D + (((ulong)k * BS + bs) * N + r) * W + col) = v;
        }
        return;
      }
    }
    if constexpr (metal::is_same_v<TC, half> && (EP & 64) == 0) {
      if (g.epilogue & 1048576) { // in-place multiply: D *= v
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
          if (!c.is_valid_element(e)) continue;
          const auto ix = c.get_multidimensional_index(e);
          const int i = i0 + ix[0], j = j0 + ix[1];
          if (j >= g.n || i >= g.m) continue;
          const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
          const float4 cv = float4(*(device const half4*)(C + (ulong)i + (ulong)j * g.ldc));
          *(device half4*)(D + (ulong)i + (ulong)j * g.ldd) = lf_out4<half>(cv * v);
        }
        return;
      }
      if (g.epilogue & 4096) {    // the outer product's permuted store: D row (i, c), column (j, e) -> [(i, j)][(c, e)]
        const uint O = (uint)g.tgC, L = (uint)g.tgN;
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
          if (!c.is_valid_element(e)) continue;
          const auto ix = c.get_multidimensional_index(e);
          const int i = i0 + ix[0], j = j0 + ix[1];
          if (j >= g.n || i >= g.m) continue;
          const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
          const uint ii = lf_udiv((uint)j, O), cc = (uint)j - ii * O, jj = lf_udiv((uint)i, O), ee = (uint)i - jj * O;
          *(device half4*)(D + ((((ulong)ii * L + jj) * O + cc) * O + ee)) = half4(v);
        }
        return;
      }
      if constexpr ((TM == 64 && TN == 128) || (TM == 32 && TN == 256)) {
        if ((g.m & 3) == 0) {
          constexpr int CSTEP = TN == 256 ? 64 : 32;
          const int col0 = i0 + c.get_multidimensional_index((uint16_t)0)[0];
          const bool fullM = i0 + TN <= g.m;
          const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
          const bool hasBias = (g.epilogue & 4) != 0, halfBias = (g.biasType & 255) == 2;
          float4 b4[4] = {float4(0.f), float4(0.f), float4(0.f), float4(0.f)};
          if (hasBias && !(g.epilogue & 8192)) {
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < 4; ++k) {
              const int col = col0 + k * CSTEP;
              if (fullM || col < g.m)
                b4[k] = halfBias ? float4(*(device const half4*)((device const half*)g.bias + boff + col))
                                 : *(device const float4*)((device const float*)g.bias + boff + col);
            }
          }
          const ushort4 bs(0, 4, 32, 36);
          if (g.epilogue & 8192) {
            const uint opmC = (uint)g.tgN;
            float4 bCol4[4] = {float4(0.f), float4(0.f), float4(0.f), float4(0.f)};
            uint jTok4[4];
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < 4; ++k) {
              const uint u = (uint)(col0 + k * CSTEP);
              const uint jTok = opmC == 128 ? (u >> 7) : opmC == 256 ? (u >> 8) : lf_udiv(u, opmC);
              jTok4[k] = jTok;
              if (fullM || (int)u < g.m) bCol4[k] = *(device const float4*)((device const float*)g.bias + (u - jTok * opmC));
            }
            const bool sameTok = (jTok4[0] == jTok4[3]);
            device const float* auxF = (device const float*)g.aux;
            const bool tgC = g.tgC != 0;
            _Pragma("clang loop unroll(full)")
            for (int r_idx = 0; r_idx < 4; ++r_idx) {
              const uint16_t e0 = bs[r_idx];
              const int j = j0 + c.get_multidimensional_index(e0)[1];
              if (j >= g.n) continue;
              device TC* dp0 = D + (ulong)j * g.ldd + (ulong)col0;
              device const TC* cp0 = C + (ulong)j * g.ldc + (ulong)col0;
              const uint rowBase = (uint)j * (uint)g.tgNp;
              const float nv0 = sameTok ? auxF[rowBase + jTok4[0]] : 0.f;
              const float inv0 = sameTok ? (tgC ? 1.f / max(nv0, 1.f) : 1.f / (1e-3f + nv0)) : 0.f;
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < 4; ++k) {
                const int col = col0 + k * CSTEP;
                if (!fullM && col >= g.m) continue;
                const uint16_t e = (uint16_t)(e0 + k * 8);
                const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
                const float4 cv = float4(*(device const vec<TC, 4>*)(cp0 + k * CSTEP));
                const float nv = sameTok ? nv0 : auxF[rowBase + jTok4[k]];
                const vec<TC, 4> st = sameTok
                  ? lf_out4<TC>(cv + (tgC ? v * inv0 + bCol4[k] : (bCol4[k] + v) * inv0))
                  : lf_out4<TC>(cv + (tgC ? v / max(nv, 1.f) + bCol4[k] : (bCol4[k] + v) / (1e-3f + nv)));
                *(device vec<TC, 4>*)(dp0 + k * CSTEP) = st;
                if constexpr ((EP & 64) != 0) { c[e] = (float)st.x; c[e + 1] = (float)st.y; c[e + 2] = (float)st.z; c[e + 3] = (float)st.w; }
              }
            }
          } else if (g.epilogue & 64) {
            device const half* auxH = (device const half*)g.aux;
            _Pragma("clang loop unroll(full)")
            for (int r_idx = 0; r_idx < 4; ++r_idx) {
              const uint16_t e0 = bs[r_idx];
              const int j = j0 + c.get_multidimensional_index(e0)[1];
              if (j >= g.n) continue;
              device TC* dp0 = D + (ulong)j * g.ldd + (ulong)col0;
              device const TC* cp0 = C + (ulong)j * g.ldc + (ulong)col0;
              device const half* ap0 = auxH + (ulong)j * g.ldaux + (ulong)col0;
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < 4; ++k) {
                const int col = col0 + k * CSTEP;
                if (!fullM && col >= g.m) continue;
                const uint16_t e = (uint16_t)(e0 + k * 8);
                const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]) + b4[k];
                const float4 ax = float4(*(device const half4*)(ap0 + k * CSTEP));
                const float4 cv = float4(*(device const vec<TC, 4>*)(cp0 + k * CSTEP));
                const vec<TC, 4> st = lf_out4<TC>(cv + ax * fast::divide(1.f, 1.f + fast::exp(-v)));
                *(device vec<TC, 4>*)(dp0 + k * CSTEP) = st;
                if constexpr ((EP & 64) != 0) { c[e] = (float)st.x; c[e + 1] = (float)st.y; c[e + 2] = (float)st.z; c[e + 3] = (float)st.w; }
              }
            }
          } else {
            const bool hasBeta = g.beta != 0.f, floatC = (g.biasType & 512) != 0;
            const bool relu = (EP & 64) == 0 && (g.epilogue & 2) != 0;
            const bool gelu = (EP & 64) == 0 && (g.epilogue & 32) != 0;
            device const float* cfBase = (device const float*)g.C + (ulong)z * g.sc;
            _Pragma("clang loop unroll(full)")
            for (int r_idx = 0; r_idx < 4; ++r_idx) {
              const uint16_t e0 = bs[r_idx];
              const int j = j0 + c.get_multidimensional_index(e0)[1];
              if (j >= g.n) continue;
              device TC* dp0 = D + (ulong)j * g.ldd + (ulong)col0;
              device const TC* cp0 = C + (ulong)j * g.ldc + (ulong)col0;
              device const float* cfp0 = cfBase + (ulong)j * g.ldc + (ulong)col0;
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < 4; ++k) {
                const int col = col0 + k * CSTEP;
                if (!fullM && col >= g.m) continue;
                const uint16_t e = (uint16_t)(e0 + k * 8);
                float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
                if (hasBeta) {
                  const float4 cv = floatC ? *(device const float4*)(cfp0 + k * CSTEP)
                                           : float4(*(device const vec<TC, 4>*)(cp0 + k * CSTEP));
                  v += g.beta * cv;
                }
                if (hasBias) v += b4[k];
                if (relu) v = max(v, 0.f);
                if (gelu) for (int t = 0; t < 4; ++t) v[t] = 0.5f * v[t] * (1.f + lf_erf(v[t] * 0.70710678118654752f));
                const vec<TC, 4> st = lf_out4<TC>(v);
                *(device vec<TC, 4>*)(dp0 + k * CSTEP) = st;
                if constexpr ((EP & 64) != 0) { c[e] = (float)st.x; c[e + 1] = (float)st.y; c[e + 2] = (float)st.z; c[e + 3] = (float)st.w; }
              }
            }
          }
          if constexpr ((EP & 64) != 0) { threadgroup float part[512]; lf_emit_ln<TN, true, (TM * (TN < 128 ? 128 : TN)) / 2048>(c, c, (device const TC*)D, part, tid, g, i0, j0); }
          return;
        }
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
      if (!c.is_valid_element(e)) continue;
      const auto ix = c.get_multidimensional_index(e);
      const int i = i0 + ix[0], j = j0 + ix[1];
      if (j >= g.n || i >= g.m) continue;
      // (lf_gemm's order of operations, so the results match the staged path's bit for bit)
      float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
      const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
      const float4 bv = !(g.epilogue & 4) ? float4(0)
                        : (g.biasType & 255) == 2 ? float4(*(device const half4*)((device const half*)g.bias + boff + i))
                                                  : *(device const float4*)((device const float*)g.bias + boff + i);
      device TC* dp = D + (ulong)i + (ulong)j * g.ldd;
      if constexpr (!metal::is_same_v<TACC, half>) {
        if (g.epilogue & 64) {      // a gated residual: D = C + aux * sigmoid(v)
          if (g.epilogue & 4) v += bv;
          const float4 ax = float4(*(device const half4*)((device const half*)g.aux + (ulong)i + (ulong)j * g.ldaux));
          const float4 cv = float4(*(device const vec<TC, 4>*)(C + (ulong)i + (ulong)j * g.ldc));
          const vec<TC, 4> st = lf_out4<TC>(cv + ax * fast::divide(1.f, 1.f + fast::exp(-v)));
          *(device vec<TC, 4>*)dp = st;
          if constexpr ((EP & 64) != 0) { c[e] = (float)st.x; c[e + 1] = (float)st.y; c[e + 2] = (float)st.z; c[e + 3] = (float)st.w; }
          continue;
        }
        if (g.epilogue & 8192) {    // the outer product's normalised add: D = C + (bias + v) / (1e-3 + aux[j]) (af2_opm_add's order)
          const uint opmC = (uint)g.tgN, u = (uint)i;
          const uint jTok = opmC == 128 ? (u >> 7) : opmC == 256 ? (u >> 8) : lf_udiv(u, opmC);
          const float4 bCol = *(device const float4*)((device const float*)g.bias + (u - jTok * opmC));
          const float nv = ((device const float*)g.aux)[(uint)j * (uint)g.tgNp + jTok];
          const float4 cv = float4(*(device const vec<TC, 4>*)(C + (ulong)i + (ulong)j * g.ldc));
          const vec<TC, 4> st = lf_out4<TC>(cv + (g.tgC ? v / max(nv, 1.f) + bCol : (bCol + v) / (1e-3f + nv)));
          *(device vec<TC, 4>*)dp = st;
          if constexpr ((EP & 64) != 0) { c[e] = (float)st.x; c[e + 1] = (float)st.y; c[e + 2] = (float)st.z; c[e + 3] = (float)st.w; }
          continue;
        }
        if (g.beta != 0.f) {
          const float4 cv = (g.biasType & 512)
                              ? *(device const float4*)((device const float*)g.C + (ulong)z * g.sc + (ulong)i + (ulong)j * g.ldc)
                              : float4(*(device const vec<TC, 4>*)(C + (ulong)i + (ulong)j * g.ldc));
          v += g.beta * cv;
        }
      }
      if (g.epilogue & 4) v += bv;
      if constexpr ((EP & 64) == 0) {
        if (g.epilogue & 2) v = max(v, 0.f);
        if (g.epilogue & 32) for (int t = 0; t < 4; ++t) v[t] = 0.5f * v[t] * (1.f + lf_erf(v[t] * 0.70710678118654752f));
      }
      const vec<TC, 4> st = lf_out4<TC>(v);
      *(device vec<TC, 4>*)dp = st;
      if constexpr ((EP & 64) != 0 && !metal::is_same_v<TACC, half>) { c[e] = (float)st.x; c[e + 1] = (float)st.y; c[e + 2] = (float)st.z; c[e + 3] = (float)st.w; }
    }
    if constexpr ((EP & 64) != 0 && !metal::is_same_v<TACC, half>) { threadgroup float part[512]; lf_emit_ln<TN, true, (TM * (TN < 128 ? 128 : TN)) / 2048>(c, c, (device const TC*)D, part, tid, g, i0, j0); }
    return;
  }
  // EP bit 2, a plain product (the host's promise: a bias or a ReLU at most, alpha 1 - or a half output's scale -, beta
  // 0) - adjusted in place and stored by the tensor (bounds against D's extents), no staging tile; a float accumulator
  // into a half output an element at a time, saturating
  if constexpr ((EP & 2) != 0) {
    static_assert((EP & 1) == 0, "a plain product has no triangle gate");
    {
      if constexpr (metal::is_same_v<TC, half> && !metal::is_same_v<TC, TACC>) {
        auto ch = op.template get_destination_cooperative_tensor<decltype(mX), decltype(mW), half>();
        if (!g.epilogue && g.alpha == 1.f) {
          _Pragma("clang loop unroll(full)")
          for (uint16_t e = 0; e < c.get_capacity(); ++e) if (c.is_valid_element(e)) ch[e] = (half)clamp((float)c[e], -65504.f, 65504.f);
        } else if constexpr (TN % 32 == 0) {
          const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
          if ((g.m & 3) == 0) {
            const bool fullM = i0 + TN <= g.m, hasBias = (g.epilogue & 4) != 0, halfBias = (g.biasType & 255) == 2;
            const bool relu = (g.epilogue & 2) != 0;
            _Pragma("clang loop unroll(full)")
            for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
              if (!c.is_valid_element(e)) continue;
              const int col = i0 + c.get_multidimensional_index(e)[0];
              float4 v = float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]) * g.alpha;
              if (hasBias && (fullM || col < g.m))
                v += halfBias ? float4(*(device const half4*)((device const half*)g.bias + boff + col))
                              : *(device const float4*)((device const float*)g.bias + boff + col);
              if (relu) v = max(v, 0.f);
              v = clamp(v, -65504.f, 65504.f);
              ch[e] = (half)v.x; ch[e + 1] = (half)v.y; ch[e + 2] = (half)v.z; ch[e + 3] = (half)v.w;
            }
          } else {
            _Pragma("clang loop unroll(full)")
            for (uint16_t e = 0; e < c.get_capacity(); ++e)
              if (c.is_valid_element(e)) {
                const int col = i0 + c.get_multidimensional_index(e)[0];
                float v = (float)c[e] * g.alpha;
                if ((g.epilogue & 4) && col < g.m) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + col] : ((device const float*)g.bias)[boff + col];
                if (g.epilogue & 2) v = max(v, 0.f);
                ch[e] = (half)clamp(v, -65504.f, 65504.f);
              }
          }
        } else {
          const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
          _Pragma("clang loop unroll(full)")
          for (uint16_t e = 0; e < c.get_capacity(); ++e)
            if (c.is_valid_element(e)) {
              const int col = i0 + c.get_multidimensional_index(e)[0];
              float v = (float)c[e] * g.alpha;
              if ((g.epilogue & 4) && col < g.m) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + col] : ((device const float*)g.bias)[boff + col];
              if (g.epilogue & 2) v = max(v, 0.f);
              ch[e] = (half)clamp(v, -65504.f, 65504.f);
            }
        }
        tensor<device TC, E2, tensor_inline> tD(D, E2(g.m, g.n), array<int32_t, 2>{1, g.ldd});
        auto mD = tD.slice(i0, j0);
        ch.store(mD);
      } else {
        if (g.epilogue || g.alpha != 1.f) {
          const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
          if constexpr (TN % 32 == 0) {
            if ((g.m & 3) == 0) {
              const bool fullM = i0 + TN <= g.m, hasBias = (g.epilogue & 4) != 0, halfBias = (g.biasType & 255) == 2;
              const bool relu = (g.epilogue & 2) != 0;
              _Pragma("clang loop unroll(full)")
              for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
                if (!c.is_valid_element(e)) continue;
                const int col = i0 + c.get_multidimensional_index(e)[0];
                float4 v = float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]) * g.alpha;
                if (hasBias && (fullM || col < g.m))
                  v += halfBias ? float4(*(device const half4*)((device const half*)g.bias + boff + col))
                                : *(device const float4*)((device const float*)g.bias + boff + col);
                if (relu) v = max(v, 0.f);
                c[e] = (TACC)v.x; c[e + 1] = (TACC)v.y; c[e + 2] = (TACC)v.z; c[e + 3] = (TACC)v.w;
              }
            } else {
              _Pragma("clang loop unroll(full)")
              for (uint16_t e = 0; e < c.get_capacity(); ++e)
                if (c.is_valid_element(e)) {
                  const int col = i0 + c.get_multidimensional_index(e)[0];
                  float v = (float)c[e] * g.alpha;
                  if ((g.epilogue & 4) && col < g.m) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + col] : ((device const float*)g.bias)[boff + col];
                  if (g.epilogue & 2) v = max(v, 0.f);
                  c[e] = (TACC)v;
                }
            }
          } else {
            _Pragma("clang loop unroll(full)")
            for (uint16_t e = 0; e < c.get_capacity(); ++e)
              if (c.is_valid_element(e)) {
                const int col = i0 + c.get_multidimensional_index(e)[0];
                float v = (float)c[e] * g.alpha;
                if ((g.epilogue & 4) && col < g.m) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + col] : ((device const float*)g.bias)[boff + col];
                if (g.epilogue & 2) v = max(v, 0.f);
                c[e] = (TACC)v;
              }
          }
        }
        if constexpr (metal::is_same_v<TC, TACC>) {
          tensor<device TC, E2, tensor_inline> tD(D, E2(g.m, g.n), array<int32_t, 2>{1, g.ldd});
          auto mD = tD.slice(i0, j0);
          c.store(mD);
        } else {   // (the tensor stores only its own type: a float accumulator into a half output an element at a time)
          _Pragma("clang loop unroll(full)")
          for (uint16_t e = 0; e < c.get_capacity(); ++e)
            if (c.is_valid_element(e)) {
              const auto ix = c.get_multidimensional_index(e);
              const int i = i0 + ix[0], j = j0 + ix[1];
              if (i < g.m && j < g.n) D[(ulong)i + (ulong)j * g.ldd] = metal::is_same_v<TC, half> ? (TC)clamp((float)c[e], -65504.f, 65504.f) : (TC)c[e];
            }
        }
      }
    }
  } else {
  // every other epilogue: the tile staged, then lf_gemm's arithmetic an element at a time
  threadgroup TACC tile[TM * TN];
  {
    tensor<threadgroup TACC, extents<int32_t, TN, TM>, tensor_inline> tT(tile, extents<int32_t, TN, TM>());
    c.store(tT);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr ((EP & 1) != 0) {   // the triangle's gate: columns in blocks of 8 - 8 channels' pa, then their ga, pb, gb
    device const float* mask = (device const float*)g.aux;
    device half* outA = (device half*)g.aux2; device half* outB = (device half*)g.aux3;
    constexpr int CH = TN / 4;
    for (int t = tid; t < CH * TM; t += NT) {      // channel-major, a channel's rows consecutive
      const int cl = t / TM, jl = t % TM, jg = j0 + jl, ch = i0 / 4 + cl;
      if (jg >= g.n || ch >= g.tgC) continue;
      const int base = (cl / 8) * 32 + cl % 8;
      float vpa = (float)tile[jl * TN + base], vga = (float)tile[jl * TN + base + 8];
      float vpb = (float)tile[jl * TN + base + 16], vgb = (float)tile[jl * TN + base + 24];
      if (g.epilogue & 4) {
        device const float* bias = (device const float*)g.bias + i0 + base;
        vpa += bias[0]; vga += bias[8]; vpb += bias[16]; vgb += bias[24];
      }
      const float m = mask[g.tgR0 + jg];
      uint p = (uint)(g.tgR0 + jg), r = lf_udiv(p, (uint)g.tgN);
      ulong q = (ulong)r * g.tgNp + (p - r * (uint)g.tgN);
      outA[(ulong)ch * g.tgPairs + q] = (half)(vpa * m / (1.f + exp(-vga)));
      outB[(ulong)ch * g.tgPairs + q] = (half)(vpb * m / (1.f + exp(-vgb)));
    }
    return;
  } else {
    if (g.epilogue & 32768) {    // the coupling split's combine and SwiGLU (core.h's gemmCouplingCombine), a float tile
      // (u = inv (sa P[st] + sq Z + dq R[t]) + B[t] for a and for b, columns in gemmSwiglu's blocks of 8 - R = Qq - Qa
      // da / dq, a constant of the row's two widths, folded by the caller)
      device const float* stats = (device const float*)g.aux;
      device const float* Pp = (device const float*)g.aux2;
      device const float* QB = (device const float*)g.aux3;
      const int H2 = g.m;
      for (int t = tid; t < TM * (TN / 2); t += NT) {
        const int jl = t / (TN / 2), h = t % (TN / 2), il = (h / 8) * 16 + h % 8;
        const int j = j0 + jl, i = i0 + il;
        if (j >= g.n || i + 8 >= g.m) continue;
        device const float* sr = stats + (ulong)j * 8;
        const float inv = sr[0], dq = sr[2], sq = sr[3], sa = sr[5];
        const int st = (int)sr[6];
        const uint tq = (uint)sr[7];
        float p1 = 0.f, p2 = 0.f;
        if (st >= 0) { device const float* pr = Pp + (ulong)st * H2; p1 = sa * pr[i]; p2 = sa * pr[i + 8]; }
        device const float* qb = QB + (ulong)tq * 2 * H2;
        const float z1 = (float)tile[jl * TN + il], z2 = (float)tile[jl * TN + il + 8];
        const float u1 = inv * (p1 + sq * z1 + dq * qb[i]) + qb[H2 + i];
        const float u2 = inv * (p2 + sq * z2 + dq * qb[i + 8]) + qb[H2 + i + 8];
        lf_st(D, (ulong)((i >> 4) * 8 + (i & 7)) + (ulong)j * g.ldd, fast::divide(u1, 1.f + fast::exp(-u1)) * u2);
      }
      return;
    }
    if (g.epilogue & 128) {      // SwiGLU in blocks of 8: columns 16m..16m+7 are a_8m.., 16m+8..16m+15 their b
      constexpr int H4 = TN / 8;
      _Pragma("clang loop unroll(full)")
      for (int t = tid; t < TM * H4; t += NT) {
        const int jl = t / H4, h4 = t % H4, il = (h4 >> 1) * 16 + (h4 & 1) * 4;
        const int j = j0 + jl, i = i0 + il, outCol = (i0 >> 1) + h4 * 4;
        if (j >= g.n || i + 8 >= g.m) continue;
        float4 va = g.alpha * float4(*(threadgroup const vec<TACC, 4>*)(tile + jl * TN + il));
        float4 vb = g.alpha * float4(*(threadgroup const vec<TACC, 4>*)(tile + jl * TN + il + 8));
        if (g.epilogue & 4) {
          va += *(device const float4*)((device const float*)g.bias + i);
          vb += *(device const float4*)((device const float*)g.bias + i + 8);
        }
        float4 res = fast::divide(va, 1.f + fast::exp(-va)) * vb;
        if (i + 12 <= g.m && (g.ldd & 3) == 0 && ((ulong)D & 7) == 0) {
          *(device vec<TC, 4>*)(D + (ulong)j * g.ldd + outCol) = vec<TC, 4>(res);
        } else {
          for (int k = 0; k < 4 && i + k + 8 < g.m; ++k) lf_st(D, (ulong)(outCol + k) + (ulong)j * g.ldd, res[k]);
        }
      }
      return;
    }
    for (int t = tid; t < TM * TN; t += NT) {
      const int jl = t / TN, il = t % TN, j = j0 + jl, i = i0 + il;
      if (j >= g.n || i >= g.m) continue;
      float v = g.alpha * (float)tile[t];
      if (g.epilogue & 64) {      // a gated residual: the GEMM is the gate, aux the gated values, D the residual
        if (g.epilogue & 4) v += ((device const float*)g.bias)[i];
        v = (float)((device const half*)g.aux)[(ulong)i + (ulong)j * g.ldaux] * fast::divide(1.f, 1.f + fast::exp(-v));
        lf_st(D, (ulong)i + (ulong)j * g.ldd, lf_ldf(C + (ulong)i + (ulong)j * g.ldc) + v);
        continue;
      }
      if (g.beta != 0.f) {
        const float cv = (g.biasType & 512)
                           ? ((device const float*)g.C)[(ulong)z * g.sc + (ulong)i + (ulong)j * g.ldc]
                           : lf_ldf(C + (ulong)i + (ulong)j * g.ldc);
        v += g.beta * cv;
      }
      if (g.epilogue & 4) {
        const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
        v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[boff + i] : ((device const float*)g.bias)[boff + i];
      }
      if (g.epilogue & 2) v = max(v, 0.f);
      if (g.epilogue & 32) v = 0.5f * v * (1.f + lf_erf(v * 0.70710678118654752f));   // (GELU, erf's)
      lf_st(D, (ulong)i + (ulong)j * g.ldd, v);
    }
  }
  }
}
// (instantiated on first use by the runtime: lf_gemm_tensor<TA, TB, TC, TM, TN, TRA, TRB, EP>, host name gemmt_...)

// The triangle's last two GEMMs in one pass: pair[r][o] += v[r][o] sigmoid(g[r][o]), g = Xg Wg (+ the gate's bias, in
// float) and v = Xv Wv (+ its bias), rounded to half as the projection's half output was - so the value never leaves
// the registers. The gate's product is summed by the matrix units and rounded to half once, as v's is: a float
// accumulator beside v's was 96 registers a thread, and in half the kernel is 15% faster (440 -> 372 ms a 522-token
// trunk pass; CA 0.002 A; a sigmoid's input needs no more). A = Wg, B = Xg, C = D = the pair, bias the gate's; aux = Xv (ldaux), aux2 = Wv
// ([k][m], dense), aux3 its bias or 0. A 64 x 128 tile over 4 simdgroups: a thread's 4 adjacent columns at e..e+3.
// TP the pair's type: float, or half (the pairformer's half pair - the sum in float, rounded once on the store).
template <int TM = 64, int TN = 128, typename TP = float, bool LN = true>
kernel void lf_gemm_tensor_dual(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  using namespace mpp::tensor_ops;
  const int j0 = grp.y * TM, i0 = grp.x * TN;
  typedef dextents<int32_t, 2> E2;
  tensor<device half, E2, tensor_inline> tXg((device half*)g.B, E2(g.k, g.n), array<int32_t, 2>{1, g.ldb});
  tensor<device half, E2, tensor_inline> tWg((device half*)g.A, E2(g.m, g.k), array<int32_t, 2>{1, g.lda});
  tensor<device half, E2, tensor_inline> tXv((device half*)g.aux, E2(g.k, g.n), array<int32_t, 2>{1, g.ldaux});
  tensor<device half, E2, tensor_inline> tWv((device half*)g.aux2, E2(g.m, g.k), array<int32_t, 2>{1, g.m});
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mXg = tXg.slice(0, j0); auto mWg = tWg.slice(i0, 0);
  auto mXv = tXv.slice(0, j0); auto mWv = tWv.slice(i0, 0);
  auto cg = op.template get_destination_cooperative_tensor<decltype(mXg), decltype(mWg), half>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); ++e) if (cg.is_valid_element(e)) cg[e] = 0.h;
  op.run(mXg, mWg, cg);
  auto cv = op.template get_destination_cooperative_tensor<decltype(mXv), decltype(mWv), half>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cv.get_capacity(); ++e) if (cv.is_valid_element(e)) cv[e] = 0.h;
  op.run(mXv, mWv, cv);
  device TP* D = (device TP*)g.D;
  device const float* bg = (g.epilogue & 4) ? (device const float*)g.bias : nullptr;
  device const float* bv = (device const float*)g.aux3;
  constexpr int NC = TN == 64 ? 2 : 4;
  float4 bg4[NC], bv4[NC];
  _Pragma("clang loop unroll(full)")
  for (int k = 0; k < NC; ++k) {
    const int col = i0 + cg.get_multidimensional_index((uint16_t)(k * 8))[0];
    bg4[k] = (bg && col < g.m) ? *(device const float4*)(bg + col) : float4(0.f);
    bv4[k] = (bv && col < g.m) ? *(device const float4*)(bv + col) : float4(0.f);
  }
  if constexpr (!LN && TM == 64 && TN == 128) {
    constexpr int NR = 4;
    const int col0 = i0 + cg.get_multidimensional_index((uint16_t)0)[0];
    _Pragma("clang loop unroll(full)")
    for (int r_idx = 0; r_idx < NR; ++r_idx) {
      const uint16_t er = (uint16_t)((r_idx & 1) * 4 + (r_idx >> 1) * 32);
      const int j = j0 + cg.get_multidimensional_index(er)[1];
      if (j >= g.n) continue;
      device TP* dp0 = D + (ulong)j * g.ldd;
      _Pragma("clang loop unroll(full)")
      for (int k = 0; k < 4; ++k) {
        const uint16_t e = er + k * 8;
        const int i = col0 + k * 32;
        float4 v1 = float4((float)cg[e], (float)cg[e + 1], (float)cg[e + 2], (float)cg[e + 3]);
        if (bg) v1 += bg4[k];
        float4 v2 = float4((float)cv[e], (float)cv[e + 1], (float)cv[e + 2], (float)cv[e + 3]);
        if (bv) v2 = float4(half4(v2 + bv4[k]));
        device vec<TP, 4>* dp = (device vec<TP, 4>*)(dp0 + i);
        const float4 r = float4(*dp) + v2 * fast::divide(1.f, 1.f + fast::exp(-v1));
        *dp = lf_out4<TP>(r);
      }
    }
    return;
  }
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); e += 4) {
    if (!cg.is_valid_element(e)) continue;
    const auto ix = cg.get_multidimensional_index(e);
    const int i = i0 + ix[0], j = j0 + ix[1];
    if (j >= g.n || i >= g.m) {
      if constexpr (LN) {
        if (g.lnOut != 0) {
          cg[e] = 0.h; cv[e] = 0.h; cg[e + 1] = 0.h; cv[e + 1] = 0.h;
          cg[e + 2] = 0.h; cv[e + 2] = 0.h; cg[e + 3] = 0.h; cv[e + 3] = 0.h;
        }
      }
      continue;
    }
    const int k = TN == 64 ? ((e >> 3) & 1) : ((e >> 3) & 3);
    float4 v1 = float4((float)cg[e], (float)cg[e + 1], (float)cg[e + 2], (float)cg[e + 3]);
    if (bg) v1 += bg4[k];
    float4 v2 = float4((float)cv[e], (float)cv[e + 1], (float)cv[e + 2], (float)cv[e + 3]);
    if (bv) v2 = float4(half4(v2 + bv4[k]));
    device vec<TP, 4>* dp = (device vec<TP, 4>*)(D + (ulong)i + (ulong)j * g.ldd);
    const float4 r = float4(*dp) + v2 * (1.f / (1.f + fast::exp(-v1)));
    const vec<TP, 4> st = lf_out4<TP>(r);
    *dp = st;
    if constexpr (LN) {
      if (g.lnOut != 0) {
        if constexpr (metal::is_same_v<TP, half>) {
          cg[e] = st.x; cg[e + 1] = st.y; cg[e + 2] = st.z; cg[e + 3] = st.w;
        } else {
          const half2 hx = as_type<half2>(st.x), hy = as_type<half2>(st.y), hz = as_type<half2>(st.z), hw = as_type<half2>(st.w);
          cg[e] = hx.x; cv[e] = hx.y;
          cg[e + 1] = hy.x; cv[e + 1] = hy.y;
          cg[e + 2] = hz.x; cv[e + 2] = hz.y;
          cg[e + 3] = hw.x; cv[e + 3] = hw.y;
        }
      }
    }
  }
  if constexpr (LN) {
    threadgroup float part[512];
    if (g.lnOut != 0) lf_emit_ln<TN, metal::is_same_v<TP, half>, (TM * (TN < 128 ? 128 : TN)) / 2048>(cg, cv, (device const TP*)D, part, tid, g, i0, j0);
  }
}
// (instantiated on first use: lf_gemm_tensor_dual<64, 128>, host name gemmt_dual_64x128; <64, 128, half> gemmt_dualh_64x128)

// ---------------------------------------------------------------- the gated flash attention on the matrix units
// core/common.metal's lf_attention (the same AttnArgs, layouts, mask and bias conventions, the same arithmetic: queries
// scaled into log2 units in half, logits and the running statistics in float, P in half) with its two products on
// matmul2d. A threadgroup QB queries, a simdgroup QS of them on its own - reduce_rows wants a single simdgroup's
// scope - over key tiles of KT staged by all four. P.V accumulates (mode::multiply_accumulate: matmul2d's default
// mode OVERWRITES its destination, whatever its header's comment says).
template <int D, int QB = 64, int KT = 32>
kernel void lf_attention_tensor(constant AttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  using namespace mpp::tensor_ops;
  constexpr int NT = 128, D8 = D / 8, QS = QB / 4;
  threadgroup half Qs[QB * D], Ks[KT * D], Vs[KT * D], Ps[QB * KT];
  threadgroup float mrow[QB], lrow[QB], crow[QB], red[QB];
  const int b = tg.y, h = tg.z, q0 = tg.x * QB, n = a.n, W = a.heads * D;
  device const half* base = a.qkvg + (long)b * a.rowStride + h * D;
  const float qs = a.scale * M_LOG2E_F;
  for (int e = tid; e < QB * D8; e += NT) {
    int qi = e / D8, d = (e - qi * D8) * 8, q = q0 + qi;
    half4 v0 = half4(0), v1 = half4(0);
    if (q < n) {
      device const half* src = base + (long)q * a.posStride + d;
      float4 f0 = float4(*(device const half4*)src), f1 = float4(*(device const half4*)(src + 4));
      if (a.qBias) { f0 += *(device const float4*)(a.qBias + h * D + d); f1 += *(device const float4*)(a.qBias + h * D + d + 4); }
      v0 = half4(f0 * qs); v1 = half4(f1 * qs);
    }
    *(threadgroup half4*)(Qs + qi * D + d) = v0;
    *(threadgroup half4*)(Qs + qi * D + d + 4) = v1;
  }
  const int r0 = sg * QS;
  threadgroup float* M = mrow + r0; threadgroup float* L = lrow + r0; threadgroup float* Cr = crow + r0; threadgroup float* Rd = red + r0;
  if (lane < (uint)QS) { M[lane] = -1e30f; L[lane] = 0.f; }
  typedef dextents<int32_t, 2> E2;
  threadgroup half* Pm = Ps + r0 * KT;
  tensor<threadgroup half, E2, tensor_inline> tQ(Qs + r0 * D, E2(D, QS), array<int32_t, 2>{1, D});
  tensor<threadgroup half, E2, tensor_inline> tK(Ks, E2(D, KT), array<int32_t, 2>{1, D});
  tensor<threadgroup half, E2, tensor_inline> tV(Vs, E2(D, KT), array<int32_t, 2>{1, D});
  tensor<threadgroup half, E2, tensor_inline> tP(Pm, E2(KT, QS), array<int32_t, 2>{1, KT});
  constexpr auto dS = matmul2d_descriptor(QS, KT, D, false, true, false);
  constexpr auto dO = matmul2d_descriptor(QS, D, KT, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto O = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tV), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < O.get_capacity(); ++e) if (O.is_valid_element(e)) O[e] = 0.f;
  const long bq = (long)(a.r0 + b);
  for (int k0 = 0; k0 < n; k0 += KT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = tid; e < KT * D8; e += NT) {
      int kj = e / D8, d = (e - kj * D8) * 8, k = k0 + kj;
      half4 k0v = half4(0), k1v = half4(0), v0 = half4(0), v1 = half4(0);
      if (k < n) {
        device const half* src = base + (long)k * a.posStride + d;
        k0v = *(device const half4*)(src + W); k1v = *(device const half4*)(src + W + 4);
        v0 = *(device const half4*)(src + 2 * W); v1 = *(device const half4*)(src + 2 * W + 4);
      }
      *(threadgroup half4*)(Ks + kj * D + d) = k0v; *(threadgroup half4*)(Ks + kj * D + d + 4) = k1v;
      *(threadgroup half4*)(Vs + kj * D + d) = v0; *(threadgroup half4*)(Vs + kj * D + d + 4) = v1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
    opS.run(tQ, tK, S);
    // the bias, the mask and the keys past n (lf_attention's sentinels: -1e30 past n, -1e9 a masked key)
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < S.get_capacity(); ++e)
      if (S.is_valid_element(e)) {
        auto ix = S.get_multidimensional_index(e);
        const int k = k0 + ix[0], q = min(q0 + r0 + ix[1], n - 1);
        float v;
        if (k >= n) v = -1e30f;
        else {
          v = a.mask ? (a.mask[bq * a.maskB + (long)k * a.maskK] > 0.f ? 0.f : -1e9f) : 0.f;
          if (a.bias) v += (float)a.bias[((long)h * n + q) * a.biasStride + k];
        }
        S[e] += v;
      }
    auto R = opS.template get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
    reduce_rows(S, R, reduction_operation::max, -1e30f);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < R.get_capacity(); ++e) if (R.is_valid_element(e)) Rd[R.get_multidimensional_index(e)[0]] = R[e];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < (uint)QS) { float mn = max(M[lane], Rd[lane]); Cr[lane] = exp2(M[lane] - mn); M[lane] = mn; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < S.get_capacity(); ++e)
      if (S.is_valid_element(e)) {
        auto ix = S.get_multidimensional_index(e);
        const float pv = exp2(S[e] - M[ix[1]]);
        S[e] = pv;                        // (summed in float, as lf_attention's; P itself half)
        Pm[ix[1] * KT + ix[0]] = (half)pv;
      }
    reduce_rows(S, R, reduction_operation::sum, 0.f);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < O.get_capacity(); ++e) if (O.is_valid_element(e)) O[e] *= Cr[O.get_multidimensional_index(e)[1]];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < R.get_capacity(); ++e) if (R.is_valid_element(e)) Rd[R.get_multidimensional_index(e)[0]] = R[e];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < (uint)QS) L[lane] = L[lane] * Cr[lane] + Rd[lane];
    opO.run(tP, tV, O);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  // out = O / l * sigmoid(gate)
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < O.get_capacity(); ++e)
    if (O.is_valid_element(e)) {
      auto ix = O.get_multidimensional_index(e);
      const int d = ix[0], q = q0 + r0 + ix[1];
      if (q >= n) continue;
      const float g = (float)base[(long)q * a.posStride + 3 * W + d];
      a.out[(long)b * a.outRowStride + (long)q * a.outPosStride + h * D + d] = (half)(O[e] / L[ix[1]] * (1.f / (1.f + exp(-g))));
    }
}
// (instantiated on first use: lf_attention_tensor<D, QB, KT>, host name gemmt_attn_<D>_<QB>x<KT>)

// ---------------------------------------------------------------- v2: simdgroups on their own, little threadgroup memory
// The same arithmetic as lf_attention_tensor, but nothing staged by the threadgroup: each simdgroup's two products read
// Q, K and V straight from the qkvg buffer through device tensors (the caches share them between the simdgroups), the
// scale applied to the logits in float, so the key loop has no threadgroup barrier; and the bias tile goes through the
// same threadgroup half tile P later takes, read 8 keys a lane. Threadgroup memory: QB x KT halves for P and 4 QB floats
// - occupancy is what this kernel runs on (an extra 8 KB float bias tile: 2.90 -> 4.06 ms at n 255 D 32). No q bias
// (the host keeps lf_attention_tensor for those).
template <int D, int QS = 16, int KT = 32>
kernel void lf_attention_tensor2(constant AttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  using namespace mpp::tensor_ops;
  constexpr int QB = 4 * QS, L8 = KT / 8;
  threadgroup half Ps[QB * KT];
  threadgroup float mrow[QB], lrow[QB], crow[QB], red[QB];
  const int b = tg.y, h = tg.z, n = a.n, W = a.heads * D, r0 = sg * QS, qa = tg.x * QB + r0;
  device const half* base = a.qkvg + (long)b * a.rowStride + h * D;
  const float qs = a.scale * M_LOG2E_F;
  threadgroup float* M = mrow + r0; threadgroup float* L = lrow + r0; threadgroup float* Cr = crow + r0; threadgroup float* Rd = red + r0;
  if (lane < (uint)QS) { M[lane] = -1e30f; L[lane] = 0.f; }
  typedef dextents<int32_t, 2> E2;
  threadgroup half* Pm = Ps + r0 * KT;
  const int32_t ps = (int32_t)a.posStride;
  tensor<device half, E2, tensor_inline> tQ((device half*)base, E2(D, n), array<int32_t, 2>{1, ps});
  tensor<device half, E2, tensor_inline> tK((device half*)base + W, E2(D, n), array<int32_t, 2>{1, ps});
  tensor<device half, E2, tensor_inline> tV((device half*)base + 2 * W, E2(D, n), array<int32_t, 2>{1, ps});
  tensor<threadgroup half, E2, tensor_inline> tP(Pm, E2(KT, QS), array<int32_t, 2>{1, KT});
  constexpr auto dS = matmul2d_descriptor(QS, KT, D, false, true, false);
  constexpr auto dO = matmul2d_descriptor(QS, D, KT, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto mQ = tQ.slice(0, qa);
  auto mK0 = tK.slice(0, 0); auto mV0 = tV.slice(0, 0);
  auto O = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(mV0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < O.get_capacity(); ++e) if (O.is_valid_element(e)) O[e] = 0.f;
  const long bq = (long)(a.r0 + b);
  const bool vb = a.bias && (a.biasStride & 7) == 0;
  for (int k0 = 0; k0 < n; k0 += KT) {
    auto mK = tK.slice(0, k0); auto mV = tV.slice(0, k0);
    // the bias, the mask and the keys past n into P's tile, in half (the -1e30 sentinel as -inf, -1e9 as -65504 -
    // either leaves exp2 at zero beside a real logit)
    for (int e = lane; e < QS * L8; e += 32) {
      const int ql = e / L8, kl = (e - ql * L8) * 8, q = min(qa + ql, n - 1), k = k0 + kl;
      device const half* bp = a.bias + ((long)h * n + q) * a.biasStride + k;
      half4 x0 = half4(0), x1 = half4(0);
      if (vb && k + 8 <= n) { x0 = *(device const half4*)bp; x1 = *(device const half4*)(bp + 4); }
      else if (a.bias) for (int t = 0; t < 4; ++t) { x0[t] = k + t < n ? bp[t] : 0.h; x1[t] = k + 4 + t < n ? bp[4 + t] : 0.h; }
      for (int t = 0; t < 4; ++t) {
        if (k + t >= n) x0[t] = -INFINITY;
        else if (a.mask && !(a.mask[bq * a.maskB + (long)(k + t) * a.maskK] > 0.f)) x0[t] = -65504.h;
        if (k + 4 + t >= n) x1[t] = -INFINITY;
        else if (a.mask && !(a.mask[bq * a.maskB + (long)(k + 4 + t) * a.maskK] > 0.f)) x1[t] = -65504.h;
      }
      *(threadgroup half4*)(Pm + ql * KT + kl) = x0; *(threadgroup half4*)(Pm + ql * KT + kl + 4) = x1;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto S = opS.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    opS.run(mQ, mK, S);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < S.get_capacity(); ++e)
      if (S.is_valid_element(e)) {
        auto ix = S.get_multidimensional_index(e);
        S[e] = S[e] * qs + (float)Pm[ix[1] * KT + ix[0]];      // (the bias already in log2 units, as lf_attention adds it)
      }
    auto R = opS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(S, R, reduction_operation::max, -INFINITY);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < R.get_capacity(); ++e) if (R.is_valid_element(e)) Rd[R.get_multidimensional_index(e)[0]] = R[e];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < (uint)QS) { float mn = max(M[lane], Rd[lane]); Cr[lane] = exp2(M[lane] - mn); M[lane] = mn; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < S.get_capacity(); ++e)
      if (S.is_valid_element(e)) {
        auto ix = S.get_multidimensional_index(e);
        const float pv = exp2(S[e] - M[ix[1]]);
        S[e] = pv;
        Pm[ix[1] * KT + ix[0]] = (half)pv;
      }
    reduce_rows(S, R, reduction_operation::sum, 0.f);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < O.get_capacity(); ++e) if (O.is_valid_element(e)) O[e] *= Cr[O.get_multidimensional_index(e)[1]];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < R.get_capacity(); ++e) if (R.is_valid_element(e)) Rd[R.get_multidimensional_index(e)[0]] = R[e];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < (uint)QS) L[lane] = L[lane] * Cr[lane] + Rd[lane];
    opO.run(tP, mV, O);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < O.get_capacity(); ++e)
    if (O.is_valid_element(e)) {
      auto ix = O.get_multidimensional_index(e);
      const int d = ix[0], q = qa + ix[1];
      if (q >= n) continue;
      const float g = (float)base[(long)q * a.posStride + 3 * W + d];
      a.out[(long)b * a.outRowStride + (long)q * a.outPosStride + h * D + d] = (half)(O[e] / L[ix[1]] * (1.f / (1.f + exp(-g))));
    }
}
// (instantiated on first use: lf_attention_tensor2<D, QS, KT>, host name gemmt_attn2_<D>_<QS>x<KT>)

// ---------------------------------------------------------------- v3: the online softmax in registers
// v2's products, with nothing in threadgroup memory: the bias tile loaded straight into a half destination tensor of
// the logits' layout, the row maximum and sum (and the rescale) kept in the logits' row-reduction tensors and mapped
// onto the logits' and the output's elements by their iterators, and P handed to the second product as a cooperative
// left input. (Compatibilities probed on the M5: the half logits as P.V's left input; logits -> rows; output -> the
// logits' rows.) A q bias stages the simdgroup's Q + bias once, in half, in threadgroup memory (QBIAS instances).
// NT: the bias is present and padded with -inf past n to a whole key tile (AttnArgs pad bit 1, the host's promise),
// and no mask: the key tile's logits are S qs + B with no per-element branch - the tail's padding keys come out -inf
// from the bias alone. The branch (tail || mask) in the element loop cost the tile 12% even where it never fires
template <int D, int QS, int KT, class MQ, bool NT = false, bool QBIAS = false>
METAL_FUNC void lf_attn3_body(constant AttnArgs& a, MQ mQ, int b, int h, int qa, device const half* base) {
  using namespace mpp::tensor_ops;
  const int n = a.n, W = a.heads * D;
  const float qs = a.scale * M_LOG2E_F;
  typedef dextents<int32_t, 2> E2;
  const int32_t ps = (int32_t)a.posStride;
  tensor<device half, E2, tensor_inline> tK((device half*)base + W, E2(D, n), array<int32_t, 2>{1, ps});
  tensor<device half, E2, tensor_inline> tV((device half*)base + 2 * W, E2(D, n), array<int32_t, 2>{1, ps});
  tensor<device half, E2, tensor_inline> tB((device half*)(a.bias ? a.bias + (long)h * n * a.biasStride : a.qkvg),
                                            E2(NT ? a.biasStride : n, n), array<int32_t, 2>{1, (int32_t)a.biasStride});
  constexpr auto dS = matmul2d_descriptor(QS, KT, D, false, true, false);
  constexpr auto dO = matmul2d_descriptor(QS, D, KT, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto mK0 = tK.slice(0, 0); auto mV0 = tV.slice(0, 0);
  auto S = opS.template get_destination_cooperative_tensor<MQ, decltype(mK0), float>();
  auto Sh = opS.template get_destination_cooperative_tensor<MQ, decltype(mK0), half>();
  auto Bh = opS.template get_destination_cooperative_tensor<MQ, decltype(mK0), half>();
  auto M = opS.template get_row_reduction_destination_cooperative_tensor<MQ, decltype(mK0), float>();
  auto L = opS.template get_row_reduction_destination_cooperative_tensor<MQ, decltype(mK0), float>();
  auto C = opS.template get_row_reduction_destination_cooperative_tensor<MQ, decltype(mK0), float>();
  auto R = opS.template get_row_reduction_destination_cooperative_tensor<MQ, decltype(mK0), float>();
  auto P0 = opO.template get_left_input_cooperative_tensor<half, half, float>(Sh);
  auto O = opO.template get_destination_cooperative_tensor<decltype(P0), decltype(mV0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < O.get_capacity(); ++e) if (O.is_valid_element(e)) O[e] = 0.f;
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < M.get_capacity(); ++e) if (M.is_valid_element(e)) { M[e] = -1e30f; L[e] = 0.f; }
  const long bq = (long)(a.r0 + b);
  // the simdgroup's queries held as the first product's left input, loaded once rather than every key tile (1-2%)
  auto Q = opS.template get_left_input_cooperative_tensor<half, half, float>();
  Q.load(mQ);
  if constexpr (QBIAS) {
    device const float* qb = a.qBias + h * D;
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < Q.get_capacity(); ++e)
      if (Q.is_valid_element(e)) Q[e] = (half)((float)Q[e] + qb[Q.get_multidimensional_index(e)[0]]);
  }
  for (int k0 = 0; k0 < n; k0 += KT) {
    auto mK = tK.slice(0, k0); auto mV = tV.slice(0, k0);
    opS.run(Q, mK, S);
    if constexpr (NT) {
      auto mb = tB.slice(k0, qa); Bh.load(mb);
      _Pragma("clang loop unroll(full)")
      for (uint16_t e = 0; e < S.get_capacity(); ++e) if (S.is_valid_element(e)) S[e] = S[e] * qs + (float)Bh[e];
    } else {
    if (a.bias) Bh.load(tB.slice(k0, qa));
    const bool tail = k0 + KT > n;
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < S.get_capacity(); ++e)
      if (S.is_valid_element(e)) {
        float v = S[e] * qs + (a.bias ? (float)Bh[e] : 0.f);
        if (tail || a.mask) {
          const int k = k0 + S.get_multidimensional_index(e)[0];
          if (k >= n) v = -INFINITY;
          else if (a.mask && !(a.mask[bq * a.maskB + (long)k * a.maskK] > 0.f)) v += -1e9f;
        }
        S[e] = v;
      }
    }
    reduce_rows(S, R, reduction_operation::max, -INFINITY);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < M.get_capacity(); ++e)
      if (M.is_valid_element(e)) { const float mn = max(M[e], R[e]); C[e] = exp2(M[e] - mn); M[e] = mn; }
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < S.get_capacity(); ++e)
      if (S.is_valid_element(e)) {
        const float pv = fast::exp2(S[e] - *M.map_iterator(S.get_iterator(e)));
        S[e] = pv; Sh[e] = (half)pv;
      }
    reduce_rows(S, R, reduction_operation::sum, 0.f);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < L.get_capacity(); ++e) if (L.is_valid_element(e)) L[e] = L[e] * C[e] + R[e];
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < O.get_capacity(); ++e) if (O.is_valid_element(e)) O[e] *= *C.map_iterator(O.get_iterator(e));
    auto P = opO.template get_left_input_cooperative_tensor<half, half, float>(Sh);
    opO.run(P, mV, O);
  }
  // out = O / l * sigmoid(gate): a thread holds 4 adjacent columns at e..e+3 (probed on the M5 at D 16-64), so the
  // gate is read and the output written 4 at a time where the strides keep them 8-byte aligned
  if (((a.posStride | a.outPosStride | a.outRowStride) & 3) == 0) {
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < O.get_capacity(); e += 4)
      if (O.is_valid_element(e)) {
        auto ix = O.get_multidimensional_index(e);
        const int d = ix[0], q = qa + ix[1];
        if (q >= n) continue;
        const float4 g = float4(*(device const half4*)(base + (long)q * a.posStride + 3 * W + d));
        const float il = 1.f / *L.map_iterator(O.get_iterator(e));
        const float4 o = float4(O[e], O[e + 1], O[e + 2], O[e + 3]) * il * fast::divide(1.f, 1.f + fast::exp(-g));
        *(device half4*)(a.out + (long)b * a.outRowStride + (long)q * a.outPosStride + h * D + d) = half4(o);
      }
    return;
  }
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < O.get_capacity(); ++e)
    if (O.is_valid_element(e)) {
      auto ix = O.get_multidimensional_index(e);
      const int d = ix[0], q = qa + ix[1];
      if (q >= n) continue;
      const float g = (float)base[(long)q * a.posStride + 3 * W + d];
      a.out[(long)b * a.outRowStride + (long)q * a.outPosStride + h * D + d] =
          (half)(O[e] / *L.map_iterator(O.get_iterator(e)) * (1.f / (1.f + exp(-g))));
    }
}
template <int D, int QS = 16, int KT = 64, bool QBIAS = false, bool NT = false>
kernel void lf_attention_tensor3(constant AttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const int b = tg.y, h = tg.z, n = a.n, qa = tg.x * 4 * QS + sg * QS;
  if (qa >= n) return;
  device const half* base = a.qkvg + (long)b * a.rowStride + h * D;
  typedef dextents<int32_t, 2> E2;
  tensor<device half, E2, tensor_inline> tQ((device half*)base, E2(D, n), array<int32_t, 2>{1, (int32_t)a.posStride});
  lf_attn3_body<D, QS, KT, decltype(tQ.slice(0, qa)), NT, QBIAS>(a, tQ.slice(0, qa), b, h, qa, base);
}
// (instantiated on first use: lf_attention_tensor3<D, QS, KT, QBIAS>, host name gemmt_attn3[q]_<D>_<QS>x<KT>)

// The matrix units' issue ceiling (localfold-bench peak): operands in threadgroup memory, R products a threadgroup - over
// four simdgroups (lf_mm_peak) or one each (lf_mm_peak1). On the M5 (8-core GPU): 12.3-12.6 TFLOP/s from 32 x 128 x 64 up,
// half or float accumulators; a simdgroup's own 16 x 64 x 32 12.4 - the GEMMs' 10-11 and attention's products 9.3 are
// against this
template <int TM, int TN, int KK, int R, typename TACC>
kernel void lf_mm_peak(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
  using namespace mpp::tensor_ops;
  threadgroup half xs[TM * KK]; threadgroup half ws[KK * TN];
  for (uint i = tid; i < TM * KK; i += 128) xs[i] = half((i % 7) * 0.01f);
  for (uint i = tid; i < KK * TN; i += 128) ws[i] = half((i % 5) * 0.01f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  typedef dextents<int32_t, 2> E2;
  tensor<threadgroup half, E2, tensor_inline> tX(xs, E2(KK, TM), array<int32_t, 2>{1, KK});
  tensor<threadgroup half, E2, tensor_inline> tW(ws, E2(TN, KK), array<int32_t, 2>{1, TN});
  constexpr auto desc = matmul2d_descriptor(TM, TN, KK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto c = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), TACC>();
  for (uint16_t e = 0; e < c.get_capacity(); ++e) if (c.is_valid_element(e)) c[e] = TACC(0);
  for (int r = 0; r < R; ++r) op.run(tX, tW, c);
  float s = 0;
  for (uint16_t e = 0; e < c.get_capacity(); ++e) if (c.is_valid_element(e)) s += float(c[e]);
  ((device float*)g.D)[grp.x * 128 + tid] = s;
}
template <int TM, int TN, int KK, int R, typename TACC>
kernel void lf_mm_peak1(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  using namespace mpp::tensor_ops;
  threadgroup half xs[4 * TM * KK]; threadgroup half ws[4 * KK * TN];
  for (uint i = tid; i < 4 * TM * KK; i += 128) xs[i] = half((i % 7) * 0.01f);
  for (uint i = tid; i < 4 * KK * TN; i += 128) ws[i] = half((i % 5) * 0.01f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  typedef dextents<int32_t, 2> E2;
  tensor<threadgroup half, E2, tensor_inline> tX(xs + sg * TM * KK, E2(KK, TM), array<int32_t, 2>{1, KK});
  tensor<threadgroup half, E2, tensor_inline> tW(ws + sg * KK * TN, E2(TN, KK), array<int32_t, 2>{1, TN});
  constexpr auto desc = matmul2d_descriptor(TM, TN, KK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto c = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), TACC>();
  for (uint16_t e = 0; e < c.get_capacity(); ++e) if (c.is_valid_element(e)) c[e] = TACC(0);
  for (int r = 0; r < R; ++r) op.run(tX, tW, c);
  float s = 0;
  for (uint16_t e = 0; e < c.get_capacity(); ++e) if (c.is_valid_element(e)) s += float(c[e]);
  ((device float*)g.D)[grp.x * 128 + tid] = s;
}

