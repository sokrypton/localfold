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
      outA[(ulong)ch * g.tgPairs + q] = (half)(vpa * m / (1.f + exp(-vga)));
      outB[(ulong)ch * g.tgPairs + q] = (half)(vpb * m / (1.f + exp(-vgb)));
    }
    return;
  }
  // EP bit 2, a plain product (the host's promise: a bias or a ReLU at most, alpha 1, beta 0, the output in the
  // accumulator's type) - adjusted in place and stored by the tensor (bounds against D's extents), no staging tile
  if constexpr ((EP & 2) != 0) {
    static_assert((EP & 1) == 0, "a plain product has no triangle gate");
    {
      if (g.epilogue) {
        _Pragma("clang loop unroll(full)")
        for (uint16_t e = 0; e < c.get_capacity(); ++e)
          if (c.is_valid_element(e)) {
            const int col = i0 + c.get_multidimensional_index(e)[0];
            float v = (float)c[e];
            if ((g.epilogue & 4) && col < g.m) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[col] : ((device const float*)g.bias)[col];
            if (g.epilogue & 2) v = max(v, 0.f);
            c[e] = (TACC)v;
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
            if (i < g.m && j < g.n) D[(ulong)i + (ulong)j * g.ldd] = (TC)c[e];
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
    if (g.epilogue & 128) {      // SwiGLU in blocks of 8: columns 16m..16m+7 are a_8m.., 16m+8..16m+15 their b
      for (int t = tid; t < TM * (TN / 2); t += NT) {
        const int jl = t / (TN / 2), h = t % (TN / 2), il = (h / 8) * 16 + h % 8;
        const int j = j0 + jl, i = i0 + il;
        if (j >= g.n || i + 8 >= g.m) continue;
        float va = g.alpha * (float)tile[jl * TN + il], vb = g.alpha * (float)tile[jl * TN + il + 8];
        lf_st(D, (ulong)((i >> 4) * 8 + (i & 7)) + (ulong)j * g.ldd, va / (1.f + exp(-va)) * vb);
      }
      return;
    }
    for (int t = tid; t < TM * TN; t += NT) {
      const int jl = t / TN, il = t % TN, j = j0 + jl, i = i0 + il;
      if (j >= g.n || i >= g.m) continue;
      float v = g.alpha * (float)tile[t];
      if (g.epilogue & 64) {      // a gated residual: the GEMM is the gate, aux the gated values, D the residual
        if (g.epilogue & 4) v += ((device const float*)g.bias)[i];
        v = (float)((device const half*)g.aux)[(ulong)i + (ulong)j * g.ldaux] * (1.f / (1.f + exp(-v)));
        lf_st(D, (ulong)i + (ulong)j * g.ldd, lf_ldf(C + (ulong)i + (ulong)j * g.ldc) + v);
        continue;
      }
      if (g.beta != 0.f) v += g.beta * lf_ldf(C + (ulong)i + (ulong)j * g.ldc);
      if (g.epilogue & 4) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[i] : ((device const float*)g.bias)[i];
      if (g.epilogue & 2) v = max(v, 0.f);
      if (g.epilogue & 32) v = 0.5f * v * (1.f + lf_erf(v * 0.70710678118654752f));   // (GELU, erf's)
      lf_st(D, (ulong)i + (ulong)j * g.ldd, v);
    }
  }
  }
}
// (instantiated on first use by the runtime: lf_gemm_tensor<TA, TB, TC, TM, TN, TRA, TRB, EP>, host name gemmt_...)

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
// logits' rows.)
template <int D, int QS = 16, int KT = 64>
kernel void lf_attention_tensor3(constant AttnArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  using namespace mpp::tensor_ops;
  constexpr int QB = 4 * QS;
  const int b = tg.y, h = tg.z, n = a.n, W = a.heads * D, qa = tg.x * QB + sg * QS;
  device const half* base = a.qkvg + (long)b * a.rowStride + h * D;
  const float qs = a.scale * M_LOG2E_F;
  typedef dextents<int32_t, 2> E2;
  const int32_t ps = (int32_t)a.posStride;
  tensor<device half, E2, tensor_inline> tQ((device half*)base, E2(D, n), array<int32_t, 2>{1, ps});
  tensor<device half, E2, tensor_inline> tK((device half*)base + W, E2(D, n), array<int32_t, 2>{1, ps});
  tensor<device half, E2, tensor_inline> tV((device half*)base + 2 * W, E2(D, n), array<int32_t, 2>{1, ps});
  tensor<device half, E2, tensor_inline> tB((device half*)(a.bias ? a.bias + (long)h * n * a.biasStride : a.qkvg), E2(n, n),
                                            array<int32_t, 2>{1, (int32_t)a.biasStride});
  constexpr auto dS = matmul2d_descriptor(QS, KT, D, false, true, false);
  constexpr auto dO = matmul2d_descriptor(QS, D, KT, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto mQ = tQ.slice(0, qa);
  auto mK0 = tK.slice(0, 0); auto mV0 = tV.slice(0, 0);
  auto S = opS.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
  auto Sh = opS.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK0), half>();
  auto Bh = opS.template get_destination_cooperative_tensor<decltype(mQ), decltype(mK0), half>();
  auto M = opS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
  auto L = opS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
  auto C = opS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
  auto R = opS.template get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK0), float>();
  auto P0 = opO.template get_left_input_cooperative_tensor<half, half, float>(Sh);
  auto O = opO.template get_destination_cooperative_tensor<decltype(P0), decltype(mV0), float>();
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < O.get_capacity(); ++e) if (O.is_valid_element(e)) O[e] = 0.f;
  _Pragma("clang loop unroll(full)")
  for (uint16_t e = 0; e < M.get_capacity(); ++e) if (M.is_valid_element(e)) { M[e] = -1e30f; L[e] = 0.f; }
  const long bq = (long)(a.r0 + b);
  for (int k0 = 0; k0 < n; k0 += KT) {
    auto mK = tK.slice(0, k0); auto mV = tV.slice(0, k0);
    opS.run(mQ, mK, S);
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
    reduce_rows(S, R, reduction_operation::max, -INFINITY);
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < M.get_capacity(); ++e)
      if (M.is_valid_element(e)) { const float mn = max(M[e], R[e]); C[e] = exp2(M[e] - mn); M[e] = mn; }
    _Pragma("clang loop unroll(full)")
    for (uint16_t e = 0; e < S.get_capacity(); ++e)
      if (S.is_valid_element(e)) {
        const float pv = exp2(S[e] - *M.map_iterator(S.get_iterator(e)));
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
// (instantiated on first use: lf_attention_tensor3<D, QS, KT>, host name gemmt_attn3_<D>_<QS>x<KT>)
