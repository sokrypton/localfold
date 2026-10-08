// The Metal runtime's own kernels (lfcuda.mm): copies and fills in stream order, and cuBLAS's GEMMs.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")

// ---------------------------------------------------------------- copies and fills (cudaMemcpyAsync, cudaMemsetAsync)
struct CopyArgs { device uchar* dst; device const uchar* src; ulong bytes; };
kernel void lf_copy(constant CopyArgs& a [[buffer(0)]], uint3 gid [[thread_position_in_grid]], uint3 gsz [[threads_per_grid]]) {
  ulong i = (ulong)gid.x + (ulong)gid.y * gsz.x;
  // 16 bytes a thread where both ends are aligned, else a byte
  bool aligned = (((ulong)a.dst | (ulong)a.src) & 15) == 0;
  if (aligned) {
    ulong n16 = a.bytes / 16;
    if (i < n16) ((device uint4*)a.dst)[i] = ((device const uint4*)a.src)[i];
    ulong tail = n16 * 16 + i;
    if (i < 16 && tail < a.bytes) a.dst[tail] = a.src[tail];
  } else if (i < a.bytes) {
    a.dst[i] = a.src[i];
  }
}
struct FillArgs { device uchar* dst; ulong bytes; uint value; };
kernel void lf_fill(constant FillArgs& a [[buffer(0)]], uint3 gid [[thread_position_in_grid]], uint3 gsz [[threads_per_grid]]) {
  ulong i = (ulong)gid.x + (ulong)gid.y * gsz.x;
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
struct ScalArgs { device float* x; int n, inc; float alpha; };
kernel void lf_scal(constant ScalArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if ((int)i < a.n) a.x[(ulong)i * a.inc] *= a.alpha;
}

// ---------------------------------------------------------------- GEMM (cuBLAS's, column-major)
// C[i + j ldc] = alpha * sum_k op(A)(i, k) op(B)(k, j) + beta * C[i + j ldc]  (+ bias[i], relu / gelu: cuBLASLt's)
// op(A)(i, k) = A[i + k lda], or A[k + i lda] transposed; op(B)(k, j) = B[k + j ldb], or B[j + k ldb] transposed.
// A batch is the grid's z, by a stride or (ptrs) through device arrays of pointers.
struct lf_bf16s { ushort v; };
inline float lf_ld(device const float* p, ulong i) { return p[i]; }
inline float lf_ld(device const half* p, ulong i) { return (float)p[i]; }
inline float lf_ld(device const lf_bf16s* p, ulong i) { return as_type<float>((uint)p[i].v << 16); }
inline void lf_st(device float* p, ulong i, float v) { p[i] = v; }
inline void lf_st(device half* p, ulong i, float v) { p[i] = (half)v; }
inline void lf_st(device lf_bf16s* p, ulong i, float v) {
  uint b = as_type<uint>(v);
  p[i].v = (b & 0x7fffffffu) > 0x7f800000u ? (ushort)((b >> 16) | 0x40) : (ushort)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
struct GemmArgs {
  ulong A, B, C, D, bias;            // device addresses (D: the output, C: beta's input - the same for cuBLAS)
  long sa, sb, sc, sd;               // batch strides, elements
  int m, n, k, lda, ldb, ldc, ldd;
  int ta, tb, ptrs, epilogue, biasType;
  float alpha, beta;
};
constant int BM = 64, BN = 64, BK = 16;
template <typename TA, typename TB, typename TC>
kernel void lf_gemm(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const uint z = grp.z;
  device const TA* A; device const TB* B; device TC* D; device const TC* C;
  if (g.ptrs) {
    A = ((device const device TA* const*)g.A)[z];
    B = ((device const device TB* const*)g.B)[z];
    D = ((device device TC* const*)g.D)[z];
    C = (device const TC*)D;
  } else {
    A = (device const TA*)g.A + (long)z * g.sa;
    B = (device const TB*)g.B + (long)z * g.sb;
    D = (device TC*)g.D + (long)z * g.sd;
    C = (device const TC*)g.C + (long)z * g.sc;
  }
  const int i0 = grp.x * BM, j0 = grp.y * BN;
  threadgroup float As[BM * (BK + 4)];     // [i][k]
  threadgroup float Bs[BK * (BN + 4)];     // [k][j]
  constexpr int LDA = BK + 4, LDB = BN + 4;
  // four simdgroups, 2 x 2, each a 32 x 32 block: 4 x 4 tiles of 8 x 8
  const int si = (sg / 2) * 32, sj = (sg % 2) * 32;
  simdgroup_float8x8 acc[4][4];
  UNROLL for (int a = 0; a < 4; ++a) UNROLL for (int b = 0; b < 4; ++b) acc[a][b] = simdgroup_float8x8(0);
  for (int k0 = 0; k0 < g.k; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A tile: BM x BK, consecutive threads along the contiguous axis
    for (int e = tid; e < BM * BK; e += 128) {
      int ii, kk;
      if (g.ta) { kk = e % BK; ii = e / BK; } else { ii = e % BM; kk = e / BM; }
      int i = i0 + ii, k = k0 + kk;
      float v = 0;
      if (i < g.m && k < g.k) v = lf_ld(A, g.ta ? (ulong)k + (ulong)i * g.lda : (ulong)i + (ulong)k * g.lda);
      As[ii * LDA + kk] = v;
    }
    for (int e = tid; e < BK * BN; e += 128) {
      int kk, jj;
      if (g.tb) { jj = e % BN; kk = e / BN; } else { kk = e % BK; jj = e / BK; }
      int k = k0 + kk, j = j0 + jj;
      float v = 0;
      if (k < g.k && j < g.n) v = lf_ld(B, g.tb ? (ulong)j + (ulong)k * g.ldb : (ulong)k + (ulong)j * g.ldb);
      Bs[kk * LDB + jj] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    UNROLL for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 am[4], bm[4];
      UNROLL for (int a = 0; a < 4; ++a) simdgroup_load(am[a], As + (si + a * 8) * LDA + kk, LDA);
      UNROLL for (int b = 0; b < 4; ++b) simdgroup_load(bm[b], Bs + kk * LDB + sj + b * 8, LDB);
      UNROLL for (int a = 0; a < 4; ++a) UNROLL for (int b = 0; b < 4; ++b) simdgroup_multiply_accumulate(acc[a][b], am[a], bm[b], acc[a][b]);
    }
  }
  // this lane's elements of each 8 x 8 tile: row sm, columns sn and sn + 1
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  UNROLL for (int a = 0; a < 4; ++a) UNROLL for (int b = 0; b < 4; ++b) {
    thread auto& e = acc[a][b].thread_elements();
    int i = i0 + si + a * 8 + sm;
    if (i >= g.m) continue;
    UNROLL for (int t = 0; t < 2; ++t) {
      int j = j0 + sj + b * 8 + sn + t;
      if (j >= g.n) continue;
      float v = g.alpha * e[t];
      if (g.beta != 0.f) v += g.beta * lf_ld(C, (ulong)i + (ulong)j * g.ldc);
      if (g.epilogue & 4) v += g.biasType == 2 ? (float)((device const half*)g.bias)[i] : ((device const float*)g.bias)[i];
      if (g.epilogue & 2) v = max(v, 0.f);
      if (g.epilogue & 32) v = 0.5f * v * (1.f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
      lf_st(D, (ulong)i + (ulong)j * g.ldd, v);
    }
  }
}
#define LF_GEMM(TA, TB, TC, N) \
  template [[host_name("lf_gemm_" N)]] kernel void lf_gemm<TA, TB, TC>(constant GemmArgs&, uint3, uint, uint, uint);
LF_GEMM(float, float, float, "f32_f32_f32")
LF_GEMM(float, float, half, "f32_f32_f16")
LF_GEMM(float, float, lf_bf16s, "f32_f32_bf16")
LF_GEMM(half, half, float, "f16_f16_f32")
LF_GEMM(half, half, half, "f16_f16_f16")
LF_GEMM(half, half, lf_bf16s, "f16_f16_bf16")
LF_GEMM(lf_bf16s, lf_bf16s, float, "bf16_bf16_f32")
LF_GEMM(lf_bf16s, lf_bf16s, half, "bf16_bf16_f16")
LF_GEMM(lf_bf16s, lf_bf16s, lf_bf16s, "bf16_bf16_bf16")
LF_GEMM(half, float, float, "f16_f32_f32")
LF_GEMM(float, half, float, "f32_f16_f32")
