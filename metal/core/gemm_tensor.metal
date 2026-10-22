// metal/core's GEMM on the matrix units (Apple10, the M5's "neural accelerators"): Metal 4's tensor operations
// (MetalPerformancePrimitives' matmul2d), the same arguments and the same epilogues as gemm.metal's lf_gemm. Compiled
// at Metal Shading Language 4.0 in a library of its own (core.mm: tensor instances) and chosen at run time - an older
// GPU or OS keeps lf_gemm. (args.h and gemm.metal are prepended by metal/build.sh: lf_erf, lf_udiv, lf_ldf, lf_st)
//
// The product is matmul2d's, into a cooperative tensor of the accumulator's type (half where EP bit 8 asks, as lf_gemm's
// all-half instances accumulate, else float). A plain product - a bias, a ReLU, alpha 1 and beta 0, the output in the
// accumulator's type or converted on the way out - is adjusted in place and stored from the tensor; every other epilogue stages the tile in
// threadgroup memory and applies lf_gemm's arithmetic to it an element at a time, coalesced along the output's row.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

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
inline float4 lf_ln_at(thread CT& cg, device const TD* D, constant GemmArgs& g, int i0, int j0, uint16_t e) {
  if constexpr (REG) return float4((float)cg[e], (float)cg[e + 1], (float)cg[e + 2], (float)cg[e + 3]);
  const auto ix = cg.get_multidimensional_index(e);
  const int j = min(j0 + ix[1], g.n - 1);
  return float4(*(device const vec<TD, 4>*)(D + (ulong)(i0 + ix[0]) + (ulong)j * g.ldd));
}
template <int TN, bool REG = false, typename CT, typename TD>
inline void lf_emit_ln(thread CT& cg, device const TD* D, threadgroup float* part, uint tid, constant GemmArgs& g, int i0, int j0) {
  constexpr uint NS = TN / 64;
  static_assert(TN == 128 || TN == 256, "the emitted LayerNorm is probed for 64 x 128 and 32 x 256");
  const uint sg = tid / 32, first = sg & ~(NS - 1), lane = tid & 31;
  float s4[4] = {0, 0, 0, 0};
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); e += 4) { const float4 v = lf_ln_at<REG>(cg, D, g, i0, j0, e); s4[((e >> 2) & 1) | ((e >> 4) & 2)] += v.x + v.y + v.z + v.w; }
  int rowOf[4];
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < 64; e += 4) if (((e >> 3) & 3) == 0) rowOf[((e >> 2) & 1) | ((e >> 4) & 2)] = cg.get_multidimensional_index(e)[1];
  for (int q = 0; q < 4; ++q) {
    s4[q] += simd_shuffle_xor(s4[q], 1);
    s4[q] += simd_shuffle_xor(s4[q], 8);
    if ((lane & 9) == 0) part[(0 * 4 + sg) * 64 + rowOf[q]] = s4[q];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean[4];
  for (int q = 0; q < 4; ++q) {
    float t = 0;
    for (uint k = 0; k < NS; ++k) t += part[(0 * 4 + first + k) * 64 + rowOf[q]];
    mean[q] = t / TN;
  }
  float v4[4] = {0, 0, 0, 0};
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); e += 4) {
    const int q = ((e >> 2) & 1) | ((e >> 4) & 2);
    const float4 d = lf_ln_at<REG>(cg, D, g, i0, j0, e) - mean[q];
    v4[q] += dot(d, d);
  }
  for (int q = 0; q < 4; ++q) {
    v4[q] += simd_shuffle_xor(v4[q], 1);
    v4[q] += simd_shuffle_xor(v4[q], 8);
    if ((lane & 9) == 0) part[(1 * 4 + sg) * 64 + rowOf[q]] = v4[q];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[4];
  for (int q = 0; q < 4; ++q) {
    float t = 0;
    for (uint k = 0; k < NS; ++k) t += part[(1 * 4 + first + k) * 64 + rowOf[q]];
    inv[q] = rsqrt(t / TN + g.lnEps);
  }
  device half* out = (device half*)g.lnOut;
  device const float* sc = (device const float*)g.lnScale; device const float* of = (device const float*)g.lnOffset;
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); e += 4) {
    if (!cg.is_valid_element(e)) continue;
    const auto ix = cg.get_multidimensional_index(e);
    const int i = i0 + ix[0], j = j0 + ix[1];
    if (j >= g.n) continue;
    const int q = ((e >> 2) & 1) | ((e >> 4) & 2);
    const float4 x = (lf_ln_at<REG>(cg, D, g, i0, j0, e) - mean[q]) * inv[q];
    float4 y = x;
    if (sc) y *= *(device const float4*)(sc + i);
    if (of) y += *(device const float4*)(of + i);
    *(device half4*)(out + (ulong)i + (ulong)j * TN) = half4(y);
    if (g.lnOut2) {
      float4 y2 = x;
      if (g.lnScale2) y2 *= *(device const float4*)((device const float*)g.lnScale2 + i);
      if (g.lnOffset2) y2 += *(device const float4*)((device const float*)g.lnOffset2 + i);
      *(device half4*)((device half*)g.lnOut2 + (ulong)i + (ulong)j * TN) = half4(y2);
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
  tensor<device TB, E2, tensor_inline> tX(B, TRB ? E2(g.n, g.k) : E2(g.k, g.n), array<int32_t, 2>{1, g.ldb});
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
    device const float* bias = (g.epilogue & 4) ? (device const float*)g.bias + i0 : nullptr;
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < c.get_capacity(); ++e) {
      if (((e >> 3) & 3) != 0 || !c.is_valid_element(e)) continue;
      const auto ix = c.get_multidimensional_index(e);
      const int col = ix[0], jg = j0 + ix[1], ch = i0 / 4 + col;
      if (jg >= g.n || ch >= g.tgC) continue;
      float vpa = (float)c[e], vga = (float)c[e + 8], vpb = (float)c[e + 16], vgb = (float)c[e + 24];
      if (bias) { vpa += bias[col]; vga += bias[col + 32]; vpb += bias[col + 64]; vgb += bias[col + 96]; }
      const float m = mask[g.tgR0 + jg];
      uint p = (uint)(g.tgR0 + jg), r = lf_udiv(p, (uint)g.tgN);
      ulong q = (ulong)r * g.tgNp + (p - r * (uint)g.tgN);
      outA[(ulong)ch * g.tgPairs + q] = (half)fast::divide(vpa * m, 1.f + fast::exp(-vga));
      outB[(ulong)ch * g.tgPairs + q] = (half)fast::divide(vpb * m, 1.f + fast::exp(-vgb));
    }
    return;
  }
  // EP bit 32, SwiGLU in registers: columns in blocks of 8 (16m..16m+7 are a_8m.., 16m+8..16m+15 their b); e and e + 4
  // hold the same 4 columns on rows ix[1] and ix[1] + 8, while lane ^ 8 holds column ^ 8 (b for a) on the same rows.
  // One butterfly shuffle pairs all 32 lanes: even-block lanes compute row ix[1], odd-block lanes compute row ix[1] + 8
  if constexpr ((EP & 32) != 0) {
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < c.get_capacity(); e += 8) {
      if (!c.is_valid_element(e)) continue;
      const auto ix0 = c.get_multidimensional_index(e);
      const auto ix1 = c.get_multidimensional_index(e + 4);
      float4 v0 = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
      float4 v1 = g.alpha * float4((float)c[e + 4], (float)c[e + 5], (float)c[e + 6], (float)c[e + 7]);
      if ((g.epilogue & 4) && i0 + ix0[0] < g.m) {
        const float4 bv = *(device const float4*)((device const float*)g.bias + i0 + ix0[0]);
        v0 += bv; v1 += bv;
      }
      const bool odd = (ix0[0] & 8) != 0;
      const float4 partner = simd_shuffle_xor(odd ? v0 : v1, 8);
      const float4 va = odd ? partner : v0;
      const float4 vb = odd ? v1 : partner;
      const int j = j0 + (odd ? ix1[1] : ix0[1]);
      const int i = i0 + (ix0[0] & ~8);
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
    static_assert((EP & 64) == 0 || (TM * TN == 8192 && !metal::is_same_v<TACC, half>), "the emitted LayerNorm: 64 x 128 or 32 x 256, float");
    if constexpr (metal::is_same_v<TC, float> && metal::is_same_v<TACC, half>) {
      _Pragma("clang loop unroll(full)")
      for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
        if (!c.is_valid_element(e)) continue;
        const auto ix = c.get_multidimensional_index(e);
        const int i = i0 + ix[0], j = j0 + ix[1];
        if (j >= g.tgPairs || i >= g.m) continue;
        const float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
        const uint r = (uint)j - lf_udiv((uint)j, (uint)g.tgN) * (uint)g.tgN;
        const float4 gp = *(device const float4*)((device const float*)g.aux + (ulong)r * g.ldaux + i);
        const float4 cv = *(device const float4*)(C + (ulong)i + (ulong)j * g.ldc);
        const float4 gm = g.tgC ? gp : fast::divide(1.f, 1.f + fast::exp(-gp));
        *(device float4*)(D + (ulong)i + (ulong)j * g.ldd) = cv + v * gm;
      }
      return;
    }
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < c.get_capacity(); e += 4) {
      if (!c.is_valid_element(e)) continue;
      const auto ix = c.get_multidimensional_index(e);
      const int i = i0 + ix[0], j = j0 + ix[1];
      if (j >= g.n || i >= g.m) continue;
      if constexpr (metal::is_same_v<TC, half> && !metal::is_same_v<TACC, half> && (EP & 64) == 0) {
        if (g.epilogue & 131072) {   // the coupling combine on a blocks-of-2 weight (core.h's gemmCouplingCombine): gates
          // i / 2 and i / 2 + 1 here with their values; P and R | B in gemmSwiglu's blocks of 8
          device const float4* sr4 = (device const float4*)((device const float*)g.aux + (ulong)j * 8);
          const float4 s0 = sr4[0], s1 = sr4[1];
          const float inv = s0.x, dq = s0.z, sq = s0.w, sa = s1.y;
          const int st = (int)s1.z;
          device const float* qb = (device const float*)g.aux3 + (ulong)(uint)s1.w * 2 * g.m;
          device const float* pr = (device const float*)g.aux2 + (ulong)max(st, 0) * g.m;
          const int x = i >> 1, pc = (i & ~15) | (x & 6);
          const float2 pa = st >= 0 ? sa * *(device const float2*)(pr + pc) : float2(0.f);
          const float2 pb = st >= 0 ? sa * *(device const float2*)(pr + pc + 8) : float2(0.f);
          const float2 qa = *(device const float2*)(qb + pc), qb8 = *(device const float2*)(qb + pc + 8);
          const float2 ba = *(device const float2*)(qb + g.m + pc), bb8 = *(device const float2*)(qb + g.m + pc + 8);
          const float2 ca = float2((float)c[e], (float)c[e + 1]);
          const float2 cb = float2((float)c[e + 2], (float)c[e + 3]);
          const float2 ua = inv * (pa + sq * ca + dq * qa) + ba;
          const float2 ub = inv * (pb + sq * cb + dq * qb8) + bb8;
          *(device half2*)(D + (ulong)x + (ulong)j * g.ldd) = half2(fast::divide(ua, 1.f + fast::exp(-ua)) * ub);
          continue;
        }
      }
      // (lf_gemm's order of operations, so the results match the staged path's bit for bit)
      float4 v = g.alpha * float4((float)c[e], (float)c[e + 1], (float)c[e + 2], (float)c[e + 3]);
      if constexpr (metal::is_same_v<TC, float> && (EP & 64) == 0) {
        if (g.epilogue & 262144) {   // a gated residual by rows (core.h's gemmGatedRows): D = C + gate[map(j)][i] v
          device const int* mp = (device const int*)g.aux2;
          const ulong row = mp ? (ulong)(uint)mp[j] : (ulong)j;
          const float4 gm = *(device const float4*)((device const float*)g.aux + row * g.ldaux + i);
          const float4 cv = float4(*(device const vec<TC, 4>*)(C + (ulong)i + (ulong)j * g.ldc));
          *(device vec<TC, 4>*)(D + (ulong)i + (ulong)j * g.ldd) = lf_out4<TC>(cv + gm * v);
          continue;
        }
      }
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
      }
      if constexpr (metal::is_same_v<TC, float> && (EP & 64) == 0) {
        if (g.epilogue & 16384) {   // the diffusion's gated residual: D = C + v sigmoid(aux[(j mod tgN) ldaux + i]), j < tgPairs
          if (j >= g.tgPairs) continue;
          const uint r = (uint)j - lf_udiv((uint)j, (uint)g.tgN) * (uint)g.tgN;
          const float4 gp = *(device const float4*)((device const float*)g.aux + (ulong)r * g.ldaux + i);
          const float4 cv = float4(*(device const vec<TC, 4>*)(C + (ulong)i + (ulong)j * g.ldc));
          const float4 gm = g.tgC ? gp : fast::divide(1.f, 1.f + fast::exp(-gp));
          *(device vec<TC, 4>*)dp = lf_out4<TC>(cv + v * gm);
          continue;
        }
        if (g.epilogue & 524288) {  // conditioning permuted store: [k * n + r][bs * W + col] -> [(k * BS + bs) * n + r][col]
          if (j >= g.tgPairs) continue;
          if (g.epilogue & 4) v += bv;
          const uint W = (uint)g.tgC, N = (uint)g.tgN, BS = (uint)g.m / W;
          const uint k = lf_udiv((uint)j, N), r = (uint)j - k * N;
          const uint bs = lf_udiv((uint)i, W), col = (uint)i - bs * W;
          if ((int)col < g.tgNp) v = fast::divide(1.f, 1.f + fast::exp(-v));
          *(device vec<TC, 4>*)(D + (((ulong)k * BS + bs) * N + r) * W + col) = lf_out4<TC>(v);
          continue;
        }
      }
      if constexpr (metal::is_same_v<TC, half> && (EP & 64) == 0) {
        if (g.epilogue & 1048576) { // in-place multiply: D *= v
          const float4 cv = float4(*(device const vec<TC, 4>*)(C + (ulong)i + (ulong)j * g.ldc));
          *(device vec<TC, 4>*)dp = lf_out4<TC>(cv * v);
          continue;
        }
        if (g.epilogue & 4096) {    // the outer product's permuted store: D row (i, c), column (j, e) -> [(i, j)][(c, e)]
          const uint O = (uint)g.tgC, L = (uint)g.tgN;
          const uint ii = lf_udiv((uint)j, O), cc = (uint)j - ii * O, jj = lf_udiv((uint)i, O), ee = (uint)i - jj * O;
          *(device vec<TC, 4>*)(D + ((((ulong)ii * L + jj) * O + cc) * O + ee)) = vec<TC, 4>(v);
          continue;
        }
      }
      if constexpr (!metal::is_same_v<TACC, half>) {
        if (g.epilogue & 8192) {    // the outer product's normalised add: D = C + (bias + v) / (1e-3 + aux[j]) (af2_opm_add's order)
          const float4 b128 = *(device const float4*)((device const float*)g.bias + (i & 127));
          const float s = 1e-3f + ((device const float*)g.aux)[(uint)j * ((uint)g.m >> 7) + ((uint)i >> 7)];
          const float4 cv = float4(*(device const vec<TC, 4>*)(C + (ulong)i + (ulong)j * g.ldc));
          const vec<TC, 4> st = lf_out4<TC>(cv + (b128 + v) / s);
          *(device vec<TC, 4>*)dp = st;
          if constexpr ((EP & 64) != 0) { c[e] = (float)st.x; c[e + 1] = (float)st.y; c[e + 2] = (float)st.z; c[e + 3] = (float)st.w; }
          continue;
        }
        if (g.beta != 0.f) v += g.beta * float4(*(device const vec<TC, 4>*)(C + (ulong)i + (ulong)j * g.ldc));
      }
      if (g.epilogue & 4) v += bv;
      if constexpr ((EP & 64) == 0) {
        if (g.epilogue & 2) v = max(v, 0.f);
        if (g.epilogue & 32) for (int t = 0; t < 4; ++t) v[t] = 0.5f * v[t] * (1.f + lf_erf(v[t] * 0.70710678118654752f));
      }
      const vec<TC, 4> st = lf_out4<TC>(v);
      *(device vec<TC, 4>*)dp = st;
      if constexpr ((EP & 64) != 0) { c[e] = (float)st.x; c[e + 1] = (float)st.y; c[e + 2] = (float)st.z; c[e + 3] = (float)st.w; }
    }
    if constexpr ((EP & 64) != 0) { threadgroup float part[512]; lf_emit_ln<TN, true>(c, (device const TC*)D, part, tid, g, i0, j0); }
    return;
  }
  // EP bit 2, a plain product (the host's promise: a bias or a ReLU at most, alpha 1 - or a half output's scale -, beta
  // 0) - adjusted in place and stored by the tensor (bounds against D's extents), no staging tile; a float accumulator
  // into a half output an element at a time, saturating
  if constexpr ((EP & 2) != 0) {
    static_assert((EP & 1) == 0, "a plain product has no triangle gate");
    {
      if (g.epilogue || g.alpha != 1.f) {
        const ulong boff = g.ptrs ? (ulong)z * g.m : 0;
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
      if constexpr (metal::is_same_v<TC, TACC>) {
        tensor<device TC, E2, tensor_inline> tD(D, E2(g.m, g.n), array<int32_t, 2>{1, g.ldd});
        auto mD = tD.slice(i0, j0);
        c.store(mD);
      } else if constexpr (metal::is_same_v<TC, half>) {   // (the tensor stores only its own type: through a half twin, saturating)
        auto ch = op.template get_destination_cooperative_tensor<decltype(mX), decltype(mW), half>();
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = 0; e < c.get_capacity(); ++e) if (c.is_valid_element(e)) ch[e] = (half)clamp((float)c[e], -65504.f, 65504.f);
        tensor<device TC, E2, tensor_inline> tD(D, E2(g.m, g.n), array<int32_t, 2>{1, g.ldd});
        auto mD = tD.slice(i0, j0);
        ch.store(mD);
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
      if (g.beta != 0.f) v += g.beta * lf_ldf(C + (ulong)i + (ulong)j * g.ldc);
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
template <int TM = 64, int TN = 128, typename TP = float>
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
  auto cv = op.template get_destination_cooperative_tensor<decltype(mXv), decltype(mWv), half>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); ++e) if (cg.is_valid_element(e)) cg[e] = 0.h;
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cv.get_capacity(); ++e) if (cv.is_valid_element(e)) cv[e] = 0.h;
  op.run(mXg, mWg, cg);
  op.run(mXv, mWv, cv);
  device TP* D = (device TP*)g.D;
  device const float* bg = (g.epilogue & 4) ? (device const float*)g.bias : nullptr;
  device const float* bv = (device const float*)g.aux3;
  const bool ln = g.lnOut != 0;
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < cg.get_capacity(); e += 4) {
    if (!cg.is_valid_element(e)) continue;
    const auto ix = cg.get_multidimensional_index(e);
    const int i = i0 + ix[0], j = j0 + ix[1];
    if (j >= g.n || i >= g.m) continue;
    float4 v1 = float4((float)cg[e], (float)cg[e + 1], (float)cg[e + 2], (float)cg[e + 3]);
    if (bg) v1 += *(device const float4*)(bg + i);
    float4 v2 = float4((float)cv[e], (float)cv[e + 1], (float)cv[e + 2], (float)cv[e + 3]);
    if (bv) v2 = float4(half4(v2 + *(device const float4*)(bv + i)));
    device vec<TP, 4>* dp = (device vec<TP, 4>*)(D + (ulong)i + (ulong)j * g.ldd);
    const float4 r = float4(*dp) + v2 * (1.f / (1.f + fast::exp(-v1)));
    const vec<TP, 4> st = lf_out4<TP>(r);
    *dp = st;
    if constexpr (metal::is_same_v<TP, half>) {
      if (ln) { cg[e] = st.x; cg[e + 1] = st.y; cg[e + 2] = st.z; cg[e + 3] = st.w; }
    }
  }
  threadgroup float part[512];
  if (ln) lf_emit_ln<TN, metal::is_same_v<TP, half>>(cg, (device const TP*)D, part, tid, g, i0, j0);
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
template <int D, int QS, int KT, class MQ, bool NT = false>
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
  device const half* base = a.qkvg + (long)b * a.rowStride + h * D;
  typedef dextents<int32_t, 2> E2;
  if constexpr (QBIAS) {
    threadgroup half Qs[4 * QS * D];
    threadgroup half* Qm = Qs + sg * QS * D;
    constexpr int D4 = D / 4;
    for (int e = lane; e < QS * D4; e += 32) {
      const int qi = e / D4, d = (e - qi * D4) * 4, q = qa + qi;
      half4 v = half4(0);
      if (q < n) v = half4(float4(*(device const half4*)(base + (long)q * a.posStride + d)) + *(device const float4*)(a.qBias + h * D + d));
      *(threadgroup half4*)(Qm + qi * D + d) = v;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half, E2, tensor_inline> tQ(Qm, E2(D, QS), array<int32_t, 2>{1, D});
    lf_attn3_body<D, QS, KT>(a, tQ, b, h, qa, base);
  } else {
    tensor<device half, E2, tensor_inline> tQ((device half*)base, E2(D, n), array<int32_t, 2>{1, (int32_t)a.posStride});
    lf_attn3_body<D, QS, KT, decltype(tQ.slice(0, qa)), NT>(a, tQ.slice(0, qa), b, h, qa, base);
  }
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

