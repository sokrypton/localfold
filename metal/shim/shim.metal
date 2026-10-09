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
// cudaMemcpy2DAsync / cudaMemset2DAsync: one dispatch, a thread a byte (x) and a row (y) - not a copy a row
struct Copy2DArgs { device uchar* dst; device const uchar* src; ulong dpitch, spitch, width, height; uint value, fill; };
kernel void lf_copy2d(constant Copy2DArgs& a [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
  ulong x = gid.x, y = gid.y + (ulong)gid.z * 65535;
  if (x >= a.width || y >= a.height) return;
  if (a.fill) a.dst[y * a.dpitch + x] = (uchar)a.value;
  else a.dst[y * a.dpitch + x] = a.src[y * a.spitch + x];
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
  ulong aux; int ldaux, auxPad;      // epilogue 64: D += aux * sigmoid(alpha X W) (aux half, D float)
};
// Tiles: BM x BN of C a threadgroup (four simdgroups, 2 x 2), BK of k a step. A and B are staged in threadgroup
// memory in the type the multiply takes (half for f16 inputs, float otherwise - bf16's range is float's), along
// whichever axis is contiguous in memory - 16 bytes a thread where VEC says every row start is aligned - and read
// back as 8 x 8 matrices, transposed where staging was along the other axis. TA/TB: the operand is transposed.
template <typename T> struct lf_tile { typedef float type; };
template <> struct lf_tile<half> { typedef half type; };
template <typename TA, typename TB> struct lf_mma { typedef float type; };
template <> struct lf_mma<half, half> { typedef half type; };
inline float lf_ldf(device const float* p) { return *p; }
inline float lf_ldf(device const half* p) { return (float)*p; }
inline float lf_ldf(device const lf_bf16s* p) { return as_type<float>((uint)p->v << 16); }
template <typename S, typename T> inline void lf_stage8(threadgroup T* dst, device const S* src) {
  _Pragma("clang loop unroll(full)")
  for (int e = 0; e < 8; ++e) dst[e] = (T)lf_ldf(src + e);
}
template <> inline void lf_stage8<half, half>(threadgroup half* dst, device const half* src) {
  *(threadgroup half4*)dst = *(device const half4*)src;
  *(threadgroup half4*)(dst + 4) = *(device const half4*)(src + 4);
}
template <> inline void lf_stage8<float, float>(threadgroup float* dst, device const float* src) {
  *(threadgroup float4*)dst = *(device const float4*)src;
  *(threadgroup float4*)(dst + 4) = *(device const float4*)(src + 4);
}
// Computed as C^T = X W, X = op(B)^T (n x k), W = op(A)^T (k x m): cuBLAS's column-major non-transposed product
// is then row-major activations times row-major weights, both staged along their contiguous axis and loaded as
// 8 x 8 matrices WITHOUT the transposing load (a transposed operand stages the other way and takes it). A tile is
// TR rows of X (n) by TC columns of W (m); each lane's two output elements are adjacent in C.
template <typename TA, typename TB, typename TC, int TR, int TC_, bool TRA, bool TRB, int BK = 32>
kernel void lf_gemm(constant GemmArgs& g [[buffer(0)]], uint3 grp [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  typedef typename lf_mma<TA, TB>::type T;
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
  threadgroup T Xs[(TRB ? BK : TR) * XROW];
  threadgroup T Ws[(TRA ? TC_ : BK) * WROW];
  constexpr int WR = TR / 2, WC = TC_ / 2, FR = WR / 8, FC = WC / 8;
  const int sr = (sg / 2) * WR, sc = (sg % 2) * WC;
  simdgroup_float8x8 acc[FR][FC];
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) acc[a][b] = simdgroup_float8x8(0);
  const bool vec = g.biasType & 256;
  for (int k0 = 0; k0 < g.k; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
  // a lane's elements: X row (j) sm, W columns (i) sn and sn + 1 - adjacent in C
  const int sm = (lane / 16) * 4 + (lane % 8) / 2, sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  _Pragma("clang loop unroll(full)")
  for (int a = 0; a < FR; ++a)
    _Pragma("clang loop unroll(full)")
    for (int b = 0; b < FC; ++b) {
      thread auto& e = acc[a][b].thread_elements();
      int j = j0 + sr + a * 8 + sm;
      if (j >= g.n) continue;
      _Pragma("clang loop unroll(full)")
      for (int t = 0; t < 2; ++t) {
        int i = i0 + sc + b * 8 + sn + t;
        if (i >= g.m) continue;
        float v = g.alpha * e[t];
        if (g.epilogue & 64) {      // a gated residual: the GEMM is the gate, aux the gated values, D the residual
          v = (float)((device const half*)g.aux)[(ulong)i + (ulong)j * g.ldaux] * (1.f / (1.f + exp(-v)));
          lf_st(D, (ulong)i + (ulong)j * g.ldd, lf_ldf(C + (ulong)i + (ulong)j * g.ldc) + v);
          continue;
        }
        if (g.beta != 0.f) v += g.beta * lf_ldf(C + (ulong)i + (ulong)j * g.ldc);
        if (g.epilogue & 4) v += (g.biasType & 255) == 2 ? (float)((device const half*)g.bias)[i] : ((device const float*)g.bias)[i];
        if (g.epilogue & 2) v = max(v, 0.f);
        if (g.epilogue & 32) v = 0.5f * v * (1.f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
        lf_st(D, (ulong)i + (ulong)j * g.ldd, v);
      }
    }
}
// (instantiated on first use by the runtime: lf_gemm<TA, TB, TC, TR, TC_, TRA, TRB>, host name lf_gemm_...)
