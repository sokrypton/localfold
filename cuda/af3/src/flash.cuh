// Grid attention kernels. q, k, v and the gate come from one fused projection,
// [rows of the chunk][query or key position][4W] with W = heads * D, roles in that order.
// Output: the gated attention [row][position][W] in T.
//
//   flashGridHalf<D, WARPS>   f16 tensor cores (mma.sync m16n8k16), FlashAttention-2: S, P and O
//                             in registers, cp.async double-buffered K/V/bias tiles, ldmatrix
//                             (.trans for V), scores in the log2 domain (Q carries
//                             scale*log2e, the bias log2e).
//   flashGridF32<D>           the precise path: one thread a query, f32 throughout.
#pragma once
#include "common.cuh"
#include <cstdint>

constexpr float LOG2E = 1.4426950408889634f;
// every token real (the batch's sequence mask all ones), so attention masks are all ones too and
// the flash kernels can skip them; set from the batch
inline bool MASK_ALL_ONES = false;

// Turing (sm_75, a T4) has no m16n8k16 and no cp.async: there the MMA is two m16n8k8 over the same
// fragments (a0 a1 / b0 the first eight k, a2 a3 / b1 the second - the accumulation order the k16
// instruction uses), and a copy is a 16-byte load and shared store (the kernels' barriers already
// order it: commit and wait become nothing)
// Volta (sm_70, a V100) has neither m16n8k8 nor ldmatrix: the m16n8k8 below is computed from the operand fragments by
// warp shuffles (each thread gathers its two rows of A from its own quad and its two columns of B from the quads that
// hold them, f32 sums) - the same fragments and results as the instruction, without the tensor cores. Not inlined:
// a call each, where inlined into the unrolled kernels the device code grew ~50x and cicc ran past 35 minutes
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
__device__ __noinline__ void mma1688Emu(float* acc, uint32_t a0, uint32_t a1, uint32_t b) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    uint32_t ar0 = __shfl_sync(~0u, a0, g * 4 + s), ar1 = __shfl_sync(~0u, a1, g * 4 + s);
    uint32_t bc0 = __shfl_sync(~0u, b, (2 * t) * 4 + s), bc1 = __shfl_sync(~0u, b, (2 * t + 1) * 4 + s);
    float2 x0 = __half22float2(*reinterpret_cast<__half2*>(&ar0)), x1 = __half22float2(*reinterpret_cast<__half2*>(&ar1));
    float2 y0 = __half22float2(*reinterpret_cast<__half2*>(&bc0)), y1 = __half22float2(*reinterpret_cast<__half2*>(&bc1));
    acc[0] = fmaf(x0.x, y0.x, fmaf(x0.y, y0.y, acc[0])); acc[1] = fmaf(x0.x, y1.x, fmaf(x0.y, y1.y, acc[1]));
    acc[2] = fmaf(x1.x, y0.x, fmaf(x1.y, y0.y, acc[2])); acc[3] = fmaf(x1.x, y1.x, fmaf(x1.y, y1.y, acc[3]));
  }
}
__device__ __noinline__ void mma1688hEmu(uint32_t* d, uint32_t a0, uint32_t a1, uint32_t b) {
  float2 r0 = __half22float2(*reinterpret_cast<__half2*>(&d[0])), r1 = __half22float2(*reinterpret_cast<__half2*>(&d[1]));
  float acc[4] = {r0.x, r0.y, r1.x, r1.y};
  mma1688Emu(acc, a0, a1, b);
  __half2 h0 = __floats2half2_rn(acc[0], acc[1]), h1 = __floats2half2_rn(acc[2], acc[3]);
  d[0] = *reinterpret_cast<uint32_t*>(&h0); d[1] = *reinterpret_cast<uint32_t*>(&h1);
}
#endif
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#elif __CUDA_ARCH__ < 750
  mma1688Emu(d, a[0], a[1], b0); mma1688Emu(d, a[2], a[3], b1);
#else
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
// row sums only (column 0 of m16n8k16): columns 1..7 are identical when B is all ones, so d1 and d3 need not live across tiles
__device__ __forceinline__ void mma16816_lsum(float& d0, float& d2, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  float d1, d3;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,0f00000000,%2,0f00000000};"
               : "+f"(d0), "=f"(d1), "+f"(d2), "=f"(d3)
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  float d[4] = {d0, 0.f, d2, 0.f};
  mma16816(d, a, b0, b1);
  d0 = d[0]; d2 = d[2];
#endif
}
// the same with an f16 accumulator (two half2 registers: row g, row g + 8) - F16S's scores
__device__ __forceinline__ void mma16816h(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#elif __CUDA_ARCH__ < 750
  mma1688hEmu(d, a[0], a[1], b0); mma1688hEmu(d, a[2], a[3], b1);
#else
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(b0));
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
// m16n8k8 with an f16 accumulator (an 8-wide head's Q K^T)
__device__ __forceinline__ void mma1688h(uint32_t* d, uint32_t a0, uint32_t a1, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  mma1688hEmu(d, a0, a1, b);
#else
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a0), "r"(a1), "r"(b));
#endif
}
__device__ __forceinline__ __half2 asH2(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
__device__ __forceinline__ uint32_t asU32(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// every one of these carries a "memory" clobber: without it the compiler takes cp.async, the group wait and
// ldmatrix for instructions that touch no memory and may schedule an ldmatrix ahead of the wait or the
// barrier that makes the tile it reads complete - a race that a neighbouring plain shared load had been
// pinning in place (found when a form of the grid kernel without its bias loads came out nondeterministic)
__device__ __forceinline__ void cpAsync16(void* dst, const void* src, bool valid) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
#else
  *reinterpret_cast<uint4*>(dst) = valid ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
#endif
}
// sm_75 has no cp.async: cpAsync16 there is a load and a store, so a kernel "prefetching" its next stage
// blocks on it right where it meant to overlap. A kernel that cares loads the next stage into registers
// before its compute (RegStage::load) and stores it into the idle stage after (store) - LF_REG_STAGES
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define LF_REG_STAGES 1
#else
#define LF_REG_STAGES 0
#endif
template <int N> struct RegStage {
  uint4 v[N];
  __device__ __forceinline__ void load(int i, const void* src) { v[i] = *reinterpret_cast<const uint4*>(src); }
  __device__ __forceinline__ void store(int i, void* dst) const { *reinterpret_cast<uint4*>(dst) = v[i]; }
};
__device__ __forceinline__ void cpCommit() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void cpWait() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
#endif
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
__device__ __noinline__ void ldsm2Emu(uint32_t* r, uint32_t a) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    uint32_t row = __shfl_sync(~0u, a, 8 * m + (lane >> 2)) + (lane & 3) * 4;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(r[m]) : "r"(row) : "memory");
  }
}
__device__ __noinline__ void ldsm4Emu(uint32_t* r, uint32_t a) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    uint32_t row = __shfl_sync(~0u, a, 8 * m + (lane >> 2)) + (lane & 3) * 4;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(r[m]) : "r"(row) : "memory");
  }
}
__device__ __noinline__ void ldsm4tEmu(uint32_t* r, uint32_t a) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    uint32_t r0 = __shfl_sync(~0u, a, 8 * m + 2 * (lane & 3)) + (lane >> 2) * 2;
    uint32_t r1 = __shfl_sync(~0u, a, 8 * m + 2 * (lane & 3) + 1) + (lane >> 2) * 2;
    unsigned short lo, hi;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(lo) : "r"(r0) : "memory");
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(hi) : "r"(r1) : "memory");
    r[m] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
}
#endif
// (Volta: the rows' addresses shuffled from the lanes that name them, then 32-bit shared loads - or, transposed, two
// 16-bit loads from consecutive rows - giving each thread the registers ldmatrix gives it)
__device__ __forceinline__ void ldsm2(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  ldsm2Emu(r, a);
#else
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
#endif
}
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  ldsm4Emu(r, a);
#else
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
#endif
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  ldsm4tEmu(r, a);
#else
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
#endif
}

constexpr int FA_BK = 64;
template <int D, int WARPS, int BK = FA_BK> __host__ __device__ constexpr size_t faStage() {
  return (size_t)2 * BK * (D + 8) * 2 + (size_t)(16 * WARPS) * (BK + 8) * 2 + BK * 4;
}

// mask[r * n + j] (rows) or mask[j * n + r] (columns): the KEY's mask, the pair mask
// transposed for the column direction; -1e9 where it is zero. MASKED false: every key is real
// (a protein with no padding - the mask is all ones), so no mask is loaded or added and only the
// last tile masks the keys past n (3-7% of this kernel).
__device__ __forceinline__ uint32_t ex2h2(uint32_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750                     // (Volta: no f16 ex2 - two f32 ones)
  float2 v = __half22float2(*reinterpret_cast<__half2*>(&x));
  __half2 h = __floats2half2_rn(exp2f(v.x), exp2f(v.y));
  return *reinterpret_cast<uint32_t*>(&h);
#else
  uint32_t y; asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x)); return y;
#endif
}
constexpr uint32_t ONES_H2 = 0x3C003C00u;      // half2(1, 1)
// REG: the next key tile staged in REGISTERS while this one is computed, then stored into the one shared
// buffer - for a device without cp.async (a T4), where the double buffer's "async" copies are synchronous
// loads that stall the warp before every tile, and its two stages (39 KB at D 32) let one block an SM fit
// in a T4's 64 KB. The same arithmetic in the same order: the output is identical.
// STRIDED: rows and positions at arbitrary strides (elements) - an attention ACROSS a tensor's leading axis
// (an MSA's columns, the triangle's ending node) read where the tensor lies rather than from a transposed
// copy; cuda/af2 runs it. Without it the addressing is the dense [rows][positions][4W] layout's, in the
// same 32-bit arithmetic as always (the strides would cost registers the dense form has none to spare).
// F16S (unmasked only): the scores accumulated in f16 - the bias tile IS the MMA's accumulator input, the row max
// and the subtraction packed half2 ops, the subtraction's result already the exponent's packed input - where
// the f32 scores spent ~140 scalar instructions a tile against 27 MMAs (bias conversions and adds, 24 subtracts,
// 12 pack conversions, the max): the kernel is issue-bound on them. 11 mantissa bits for a logit, where
// AlphaFold 3's own JAX runs these in bf16's 8; the running max is rounded to an f16 value so the exponents and
// the rescale agree.
template <int D, int WARPS, bool MASKED = true, int BK = FA_BK, bool REG = false,
          int MINB = (D == 48 && WARPS == 4 && BK == 48 && !REG) ? 4 : 1, bool STRIDED = false, bool F16S = false>
__global__ void __launch_bounds__(WARPS * 32, MINB) flashGridHalf(const half* __restrict__ qkvg, const half* __restrict__ bias,
    int biasStride, const float* __restrict__ mask, half* __restrict__ out, int n, int heads, size_t r0, bool tr, float scale,
    const float* qBias, size_t rowStride = 0, size_t posStride = 0, size_t outRowStride = 0, size_t outPosStride = 0,
    int qlo = 0, int qhi = 0, int biasRows = 0) {
  // [qlo, qhi): the queries this launch computes (qhi 0: all n) - several GPUs, each its own query rows; biasRows: the
  // rows a head's bias plane holds (0: n), its first being query qlo's
  constexpr int BQ = 16 * WARPS, LDK = D + 8, LDB = BK + 8, NT = WARPS * 32;
  static_assert(BK % 16 == 0, "a key tile is whole k16 steps of the PV product");
  constexpr size_t STAGE = faStage<D, WARPS, BK>();
  extern __shared__ __align__(16) unsigned char smem[];
  auto Kst = [&](int s) { return (half*)(smem + s * STAGE); };
  auto Vst = [&](int s) { return Kst(s) + BK * LDK; };
  auto Bst = [&](int s) { return Vst(s) + BK * LDK; };
  auto Mst = [&](int s) { return (float*)(Bst(s) + BQ * LDB); };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  // the head is the SLOWEST index: CTAs run in roughly linear order, so all rows of one head pass
  // before the next head starts and the pair bias they share is one head's (n x n f16) rather than
  // every head's - at 2088 tokens all four heads' bias is 35 MB against a 40 MB L2
  const size_t rowsHere = gridDim.y / heads;
  size_t b = blockIdx.y; int h = (int)(b / rowsHere); size_t rl = b % rowsHere, r = r0 + rl;
  const int Wd = heads * D, W4 = 4 * Wd;
  // a position's offset in q, k, v and the gate, and an output row's
  auto pos = [&](auto i) { if constexpr (STRIDED) return (size_t)i * posStride; else return i * W4; };
  auto outAt = [&](int i) -> size_t {
    if constexpr (STRIDED) return rl * outRowStride + (size_t)i * outPosStride + h * D;
    else return (rl * n + i) * Wd + h * D;
  };
  const half* base = qkvg + (STRIDED ? rl * rowStride : rl * (size_t)n * W4) + h * D;
  const int QN = qhi ? qhi : n;
  int q0 = qlo + blockIdx.x * BQ;
  const bool warpLive = q0 + warp * 16 < QN;
  // the loads, with compile-time trip counts and 32-bit offsets: written as a strided loop from
  // threadIdx.x the compiler could not unroll it, and the per-tile index arithmetic was ~800
  // integer instructions a thread against 36 tensor-core MMAs
  constexpr int KV_CHUNKS = BK * (D / 8), B_CHUNKS = BQ * (BK / 8);
  static_assert(B_CHUNKS % NT == 0, "a bias tile is a whole number of chunks a thread");
  const half* biasHead = bias + (size_t)h * (biasRows ? biasRows : n) * biasStride - (size_t)qlo * biasStride;
  // per-thread source offsets, advanced by a constant each tile (the bounds matter only on the
  // last tile, which the `last` flag takes through the checked path)
  constexpr int KV_PER = (KV_CHUNKS + NT - 1) / NT, B_PER = B_CHUNKS / NT;
  using SrcOff = std::conditional_t<STRIDED, size_t, int>;
  SrcOff kvBase[KV_PER]; int kvMeta[KV_PER];
#pragma unroll
  for (int k = 0; k < KV_PER; ++k) {
    int u = k * NT + threadIdx.x, jj = u / (D / 8), c = (u % (D / 8)) * 8;
    int kj = (KV_CHUNKS % NT == 0 || u < KV_CHUNKS) ? jj : 0x7FFF;
    kvMeta[k] = (jj * LDK + c) | (kj << 16);
    kvBase[k] = (SrcOff)pos(jj) + Wd + c;
  }
  int bBase[B_PER], bMeta[B_PER];
#pragma unroll
  for (int k = 0; k < B_PER; ++k) {
    int u = k * NT + threadIdx.x, qi = u / (BK / 8), c = (u % (BK / 8)) * 8, i = q0 + qi;
    bMeta[k] = (qi * LDB + c) | ((i < QN ? c : 0x7FFF) << 16);
    bBase[k] = (i < QN ? i : qlo) * biasStride + c;
  }
  auto issue = [&](int j0, int st) {
    half *K = Kst(st), *V = Vst(st), *B = Bst(st);
    bool last = j0 + BK > n;
    if (!last) {
      const half* kvp = base + pos(j0);
      const half* bp = biasHead + j0;
#pragma unroll
      for (int k = 0; k < KV_PER; ++k) {
        if (k + 1 < KV_PER || KV_CHUNKS % NT == 0 || (kvMeta[k] >> 16) < 0x7FFF) {
          const half* src = kvp + kvBase[k];
          cpAsync16(K + (kvMeta[k] & 0xFFFF), src, true);
          cpAsync16(V + (kvMeta[k] & 0xFFFF), src + Wd, true);
        }
      }
#pragma unroll
      for (int k = 0; k < B_PER; ++k) {
        cpAsync16(B + (bMeta[k] & 0xFFFF), bp + bBase[k], (bMeta[k] >> 16) < 0x7FFF);
      }
      if (MASKED && threadIdx.x < BK) {
        int j = j0 + threadIdx.x;
        Mst(st)[threadIdx.x] = mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f;
      }
      cpCommit();
      return;
    }
#pragma unroll
    for (int k = 0; k < KV_PER; ++k) {
      int kj = kvMeta[k] >> 16, koff = kvMeta[k] & 0xFFFF;
      if (kj >= 0x7FFF) continue;
      bool ok = j0 + kj < n;
      const half* src = ok ? base + kvBase[k] + pos(j0) : base;
      cpAsync16(K + koff, src, ok);
      cpAsync16(V + koff, src + Wd, ok);
    }
#pragma unroll
    for (int k = 0; k < B_PER; ++k) {
      int bc = bMeta[k] >> 16, boff = bMeta[k] & 0xFFFF;
      bool ok = bc < 0x7FFF && j0 + bc < n;
      cpAsync16(B + boff, ok ? biasHead + (size_t)bBase[k] + j0 : bias, ok);
    }
    if (MASKED && threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Mst(st)[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f)
                                   : -INFINITY;
    }
    cpCommit();
  };
  constexpr bool GATE_SM = !REG && D % 16 == 0;
  auto issueGate = [&](int st) {
    if (warpLive) {
      half* G = Kst(st) + warp * 16 * LDK;
      constexpr int PER = D / 8, G_PER = D / 16;
#pragma unroll
      for (int k = 0; k < G_PER; ++k) {
        int u = k * 32 + lane, rq = u / PER, c = (u % PER) * 8, i = q0 + warp * 16 + rq;
        bool ok = i < QN;
        cpAsync16(G + rq * LDK + c, ok ? base + (STRIDED ? pos(i) : (size_t)i * W4) + 3 * Wd + c : base, ok);
      }
    }
    cpCommit();
  };
  // ...REG's halves of `issue`: the loads into registers, and the registers into the one stage
  uint4 kr[REG ? KV_PER : 1][2], brg[REG ? B_PER : 1]; float mr = 0.f;
  auto loadR = [&](int j0) {
    if constexpr (REG) {
      bool last = j0 + BK > n;
#pragma unroll
      for (int k = 0; k < KV_PER; ++k) {
        int kj = kvMeta[k] >> 16;
        if (kj >= 0x7FFF) continue;
        bool ok = !last || j0 + kj < n;
        const half* src = base + kvBase[k] + (ok ? pos(j0) : 0);
        kr[k][0] = ok ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
        kr[k][1] = ok ? *reinterpret_cast<const uint4*>(src + Wd) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int k = 0; k < B_PER; ++k) {
        int bc = bMeta[k] >> 16;
        bool ok = bc < 0x7FFF && (!last || j0 + bc < n);
        brg[k] = ok ? *reinterpret_cast<const uint4*>(biasHead + (size_t)bBase[k] + j0) : make_uint4(0, 0, 0, 0);
      }
      if (MASKED && threadIdx.x < BK) {
        int j = j0 + threadIdx.x;
        mr = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f) : -INFINITY;
      }
    }
  };
  auto storeR = [&]() {
    if constexpr (REG) {
      half *K = Kst(0), *V = Vst(0), *B = Bst(0);
#pragma unroll
      for (int k = 0; k < KV_PER; ++k) {
        if ((kvMeta[k] >> 16) >= 0x7FFF) continue;
        *reinterpret_cast<uint4*>(K + (kvMeta[k] & 0xFFFF)) = kr[k][0];
        *reinterpret_cast<uint4*>(V + (kvMeta[k] & 0xFFFF)) = kr[k][1];
      }
#pragma unroll
      for (int k = 0; k < B_PER; ++k) *reinterpret_cast<uint4*>(B + (bMeta[k] & 0xFFFF)) = brg[k];
      if (MASKED && threadIdx.x < BK) Mst(0)[threadIdx.x] = mr;
    }
  };
  int tiles = (n + BK - 1) / BK;
  if constexpr (REG) { loadR(0); storeR(); __syncthreads(); if (tiles > 1) loadR(BK); }
  else issue(0, 0);
  int i0 = q0 + warp * 16 + g, i1 = i0 + 8;
  auto q2 = [&](int i, int e) -> uint32_t {
    if (i >= QN) return 0u;
    half2 v = *reinterpret_cast<const half2*>(base + (STRIDED ? pos(i) : (size_t)i * W4) + e);
    float2 f = __half22float2(v);
    if (qBias) { f.x += qBias[h * D + e]; f.y += qBias[h * D + e + 1]; }   // the query's bias, folded in here
    return pack2(f.x * scale * LOG2E, f.y * scale * LOG2E);
  };
  uint32_t qa[D / 16][4];
#pragma unroll
  for (int ks = 0; ks < D / 16; ++ks) {
    int e = ks * 16 + tig * 2;
    qa[ks][0] = q2(i0, e); qa[ks][1] = q2(i1, e); qa[ks][2] = q2(i0, e + 8); qa[ks][3] = q2(i1, e + 8);
  }
  float o[D / 8][4] = {};
  float m0 = -INFINITY, m1 = -INFINITY, lsum0 = 0.f, lsum2 = 0.f;
  for (int tile = 0; tile < tiles; ++tile) {
    int st = REG ? 0 : tile & 1;
    if constexpr (!REG) {
      cpWait<0>();
      __syncthreads();
      if (tile + 1 < tiles) issue((tile + 1) * BK, st ^ 1);
      else if constexpr (GATE_SM) issueGate(st ^ 1);
    }
    if (warpLive) {
      const half *K = Kst(st), *V = Vst(st), *B = Bst(st); const float* Ms = Mst(st);
      const half* rowp = B + (warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDB + (lane >> 4) * 8;
      if constexpr (F16S && !MASKED) {
        uint32_t sh[BK / 8][2];
#pragma unroll
        for (int nt = 0; nt < BK / 8; nt += 2) {
          uint32_t r4[4];
          ldsm4(r4, rowp + nt * 8);
          if constexpr (D == 48) {
            uint32_t kb0[4], kb1[4], kb2[4];
            ldsm4(kb0, K + (nt * 8 + (lane & 7)) * LDK + (lane >> 3) * 8);
            ldsm4(kb1, K + ((nt + 1) * 8 + (lane & 7)) * LDK + (lane >> 3) * 8);
            ldsm4(kb2, K + ((nt + (lane >> 4)) * 8 + (lane & 7)) * LDK + 32 + ((lane >> 3) & 1) * 8);
            sh[nt][0] = r4[0]; sh[nt][1] = r4[1]; sh[nt + 1][0] = r4[2]; sh[nt + 1][1] = r4[3];
#pragma unroll
            for (int s = 0; s < 2; ++s) {
              if (tile == tiles - 1) {
                int jg = tile * BK + (nt + s) * 8 + tig * 2;
                if (jg >= n) { sh[nt + s][0] = (sh[nt + s][0] & 0xFFFF0000u) | 0xFC00u; sh[nt + s][1] = (sh[nt + s][1] & 0xFFFF0000u) | 0xFC00u; }
                if (jg + 1 >= n) { sh[nt + s][0] = (sh[nt + s][0] & 0xFFFFu) | 0xFC000000u; sh[nt + s][1] = (sh[nt + s][1] & 0xFFFFu) | 0xFC000000u; }
              }
            }
            mma16816h(sh[nt], qa[0], kb0[0], kb0[1]);
            mma16816h(sh[nt], qa[1], kb0[2], kb0[3]);
            mma16816h(sh[nt + 1], qa[0], kb1[0], kb1[1]);
            mma16816h(sh[nt + 1], qa[1], kb1[2], kb1[3]);
            mma16816h(sh[nt], qa[2], kb2[0], kb2[1]);
            mma16816h(sh[nt + 1], qa[2], kb2[2], kb2[3]);
          } else {
            sh[nt][0] = r4[0]; sh[nt][1] = r4[1]; sh[nt + 1][0] = r4[2]; sh[nt + 1][1] = r4[3];
#pragma unroll
            for (int s = 0; s < 2; ++s) {
              int jj = (nt + s) * 8 + tig * 2;
              if (tile == tiles - 1) {                       // keys past n: -inf (f16 0xFC00), both rows
                int jg = tile * BK + jj;
                if (jg >= n) { sh[nt + s][0] = (sh[nt + s][0] & 0xFFFF0000u) | 0xFC00u; sh[nt + s][1] = (sh[nt + s][1] & 0xFFFF0000u) | 0xFC00u; }
                if (jg + 1 >= n) { sh[nt + s][0] = (sh[nt + s][0] & 0xFFFFu) | 0xFC000000u; sh[nt + s][1] = (sh[nt + s][1] & 0xFFFFu) | 0xFC000000u; }
              }
#pragma unroll
              for (int k2 = 0; k2 < D / 32 + (D % 32 ? 1 : 0); ++k2) {
                if constexpr (D % 32 != 0) {
                  if (k2 * 2 + 1 >= D / 16) {
                    uint32_t kb[2];
                    ldsm2(kb, K + ((nt + s) * 8 + (lane & 7)) * LDK + k2 * 32 + ((lane >> 3) & 1) * 8);
                    mma16816h(sh[nt + s], qa[k2 * 2], kb[0], kb[1]);
                    continue;
                  }
                }
                uint32_t kb[4];
                ldsm4(kb, K + ((nt + s) * 8 + (lane & 7)) * LDK + k2 * 32 + (lane >> 3) * 8);
                mma16816h(sh[nt + s], qa[k2 * 2], kb[0], kb[1]);
                if (k2 * 2 + 1 < D / 16) mma16816h(sh[nt + s], qa[k2 * 2 + 1], kb[2], kb[3]);
              }
            }
          }
        }
        __half2 x0 = asH2(sh[0][0]), x1 = asH2(sh[0][1]);
#pragma unroll
        for (int nt = 1; nt < BK / 8; ++nt) { x0 = __hmax2(x0, asH2(sh[nt][0])); x1 = __hmax2(x1, asH2(sh[nt][1])); }
        float t0 = fmaxf(__low2float(x0), __high2float(x0)), t1 = fmaxf(__low2float(x1), __high2float(x1));
        t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
        // (the new maxima rounded to f16 values: the exponents subtract exactly these, and the rescale uses them)
        uint32_t vb[2][4];
        ldsm4t(vb[0], V + (((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (lane >> 4) * 8);
        float n0, n1;
        if (tile == 0) {
          n0 = __half2float(__float2half(t0)); n1 = __half2float(__float2half(t1));
          m0 = n0; m1 = n1;
        } else {
          n0 = __half2float(__float2half(fmaxf(m0, t0))); n1 = __half2float(__float2half(fmaxf(m1, t1)));
          float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
          m0 = n0; m1 = n1;
#pragma unroll
          for (int et = 0; et < D / 8; ++et) { o[et][0] *= c0; o[et][1] *= c0; o[et][2] *= c1; o[et][3] *= c1; }
          lsum0 *= c0; lsum2 *= c1;
        }
        __half2 h0 = __float2half2_rn(n0), h1 = __float2half2_rn(n1);
#pragma unroll
        for (int t = 0; t < BK / 16; ++t) {
          uint32_t pa[4] = { ex2h2(asU32(__hsub2(asH2(sh[2 * t][0]), h0))), ex2h2(asU32(__hsub2(asH2(sh[2 * t][1]), h1))),
                             ex2h2(asU32(__hsub2(asH2(sh[2 * t + 1][0]), h0))), ex2h2(asU32(__hsub2(asH2(sh[2 * t + 1][1]), h1))) };
          mma16816_lsum(lsum0, lsum2, pa, ONES_H2, ONES_H2);
#pragma unroll
          for (int et = 0; et < D / 8; et += 2) {
            const int cur = (t * (D / 16) + (et >> 1)) & 1;
            if (et + 2 < D / 8) ldsm4t(vb[cur ^ 1], V + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (et + 2 + (lane >> 4)) * 8);
            else if (t + 1 < BK / 16) ldsm4t(vb[cur ^ 1], V + ((t + 1) * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (lane >> 4) * 8);
            mma16816(o[et], pa, vb[cur][0], vb[cur][1]);
            mma16816(o[et + 1], pa, vb[cur][2], vb[cur][3]);
          }
        }
      } else {
        // S starts from the bias (and the key mask), and the tensor cores accumulate Q.K onto it
        float sv[BK / 8][4];
#pragma unroll
        for (int nt = 0; nt < BK / 8; nt += 2) {
          uint32_t r4[4];
          ldsm4(r4, rowp + nt * 8);
          if constexpr (D == 48) {
            uint32_t kb0[4], kb1[4], kb2[4];
            ldsm4(kb0, K + (nt * 8 + (lane & 7)) * LDK + (lane >> 3) * 8);
            ldsm4(kb1, K + ((nt + 1) * 8 + (lane & 7)) * LDK + (lane >> 3) * 8);
            ldsm4(kb2, K + ((nt + (lane >> 4)) * 8 + (lane & 7)) * LDK + 32 + ((lane >> 3) & 1) * 8);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
              int jj = (nt + s) * 8 + tig * 2;
              float2 u = __half22float2(asH2(r4[2 * s])), v = __half22float2(asH2(r4[2 * s + 1]));
              if (MASKED) {
                float ma = Ms[jj], mb = Ms[jj + 1];
                u.x += ma; u.y += mb; v.x += ma; v.y += mb;
              } else if (tile == tiles - 1) {
                int jg = tile * BK + jj;
                if (jg >= n) { u.x = v.x = -INFINITY; }
                if (jg + 1 >= n) { u.y = v.y = -INFINITY; }
              }
              sv[nt + s][0] = u.x; sv[nt + s][1] = u.y; sv[nt + s][2] = v.x; sv[nt + s][3] = v.y;
            }
            mma16816(sv[nt], qa[0], kb0[0], kb0[1]);
            mma16816(sv[nt], qa[1], kb0[2], kb0[3]);
            mma16816(sv[nt + 1], qa[0], kb1[0], kb1[1]);
            mma16816(sv[nt + 1], qa[1], kb1[2], kb1[3]);
            mma16816(sv[nt], qa[2], kb2[0], kb2[1]);
            mma16816(sv[nt + 1], qa[2], kb2[2], kb2[3]);
          } else {
#pragma unroll
            for (int s = 0; s < 2; ++s) {
              int jj = (nt + s) * 8 + tig * 2;
              float2 u = __half22float2(asH2(r4[2 * s])), v = __half22float2(asH2(r4[2 * s + 1]));
              if (MASKED) {
                float ma = Ms[jj], mb = Ms[jj + 1];
                u.x += ma; u.y += mb; v.x += ma; v.y += mb;
              } else if (tile == tiles - 1) {
                int jg = tile * BK + jj;
                if (jg >= n) { u.x = v.x = -INFINITY; }
                if (jg + 1 >= n) { u.y = v.y = -INFINITY; }
              }
              sv[nt + s][0] = u.x; sv[nt + s][1] = u.y; sv[nt + s][2] = v.x; sv[nt + s][3] = v.y;
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
              for (int k2 = 0; k2 < D / 32 + (D % 32 ? 1 : 0); ++k2) {
                if constexpr (D % 32 != 0) {
                  if (k2 * 2 + 1 >= D / 16) {
                    uint32_t kb[2];
                    ldsm2(kb, K + ((nt + s) * 8 + (lane & 7)) * LDK + k2 * 32 + ((lane >> 3) & 1) * 8);
                    mma16816(sv[nt + s], qa[k2 * 2], kb[0], kb[1]);
                    continue;
                  }
                }
                uint32_t kb[4];
                ldsm4(kb, K + ((nt + s) * 8 + (lane & 7)) * LDK + k2 * 32 + (lane >> 3) * 8);
                mma16816(sv[nt + s], qa[k2 * 2], kb[0], kb[1]);
                if (k2 * 2 + 1 < D / 16) mma16816(sv[nt + s], qa[k2 * 2 + 1], kb[2], kb[3]);
              }
            }
          }
        }
        // the row maxima as a tree, not a 16-deep chain of dependent max instructions
        float r0m[BK / 8], r1m[BK / 8];
#pragma unroll
        for (int nt = 0; nt < BK / 8; ++nt) { r0m[nt] = fmaxf(sv[nt][0], sv[nt][1]); r1m[nt] = fmaxf(sv[nt][2], sv[nt][3]); }
        // (folding the ends together, so a tile of 6 n8 columns reduces as well as one of 8)
#pragma unroll
        for (int cnt = BK / 8; cnt > 1; cnt = (cnt + 1) / 2) {
#pragma unroll
          for (int nt = 0; nt < cnt / 2; ++nt) { r0m[nt] = fmaxf(r0m[nt], r0m[cnt - 1 - nt]); r1m[nt] = fmaxf(r1m[nt], r1m[cnt - 1 - nt]); }
        }
        float t0 = r0m[0], t1 = r1m[0];
        t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
        uint32_t vb[2][4];
        ldsm4t(vb[0], V + (((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (lane >> 4) * 8);
        float n0, n1;
        if (tile == 0) {
          n0 = t0; n1 = t1;
          m0 = n0; m1 = n1;
        } else {
          n0 = fmaxf(m0, t0); n1 = fmaxf(m1, t1);
          float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
          m0 = n0; m1 = n1;
#pragma unroll
          for (int et = 0; et < D / 8; ++et) { o[et][0] *= c0; o[et][1] *= c0; o[et][2] *= c1; o[et][3] *= c1; }
          lsum0 *= c0; lsum2 *= c1;
        }
        // P = 2^(S - max), two at a time in f16 (ex2.approx.f16x2: half the SFU work, and the result
        // is already the packed A fragment the PV product reads); its row sums on the tensor cores
        // (P . ones), accumulated in f32 like O
#pragma unroll
        for (int t = 0; t < BK / 16; ++t) {
          uint32_t pa[4] = { ex2h2(pack2(sv[2 * t][0] - n0, sv[2 * t][1] - n0)), ex2h2(pack2(sv[2 * t][2] - n1, sv[2 * t][3] - n1)),
                             ex2h2(pack2(sv[2 * t + 1][0] - n0, sv[2 * t + 1][1] - n0)),
                             ex2h2(pack2(sv[2 * t + 1][2] - n1, sv[2 * t + 1][3] - n1)) };
          mma16816_lsum(lsum0, lsum2, pa, ONES_H2, ONES_H2);
#pragma unroll
          for (int et = 0; et < D / 8; et += 2) {
            const int cur = (t * (D / 16) + (et >> 1)) & 1;
            if (et + 2 < D / 8) ldsm4t(vb[cur ^ 1], V + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (et + 2 + (lane >> 4)) * 8);
            else if (t + 1 < BK / 16) ldsm4t(vb[cur ^ 1], V + ((t + 1) * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (lane >> 4) * 8);
            mma16816(o[et], pa, vb[cur][0], vb[cur][1]);
            mma16816(o[et + 1], pa, vb[cur][2], vb[cur][3]);
          }
        }
      }
    }
    // (REG: tile+1, loaded while this one computed, into the stage every warp has finished reading; then
    // tile+2's loads issued, to land while tile+1 computes)
    if constexpr (REG) {
      __syncthreads();
      if (tile + 1 < tiles) { storeR(); __syncthreads(); if (tile + 2 < tiles) loadR((tile + 2) * BK); }
    }
  }
  if (!warpLive) return;
  float inv0 = 1.f / lsum0, inv1 = 1.f / lsum2;
  if constexpr (GATE_SM) {
    cpWait<0>();
    __syncwarp();
    half* G = Kst(tiles & 1) + warp * 16 * LDK;
#pragma unroll
    for (int et = 0; et < D / 8; ++et) {
      int e = et * 8 + tig * 2;
      float2 a = __half22float2(*reinterpret_cast<const half2*>(G + g * LDK + e));
      float2 b2 = __half22float2(*reinterpret_cast<const half2*>(G + (g + 8) * LDK + e));
      *reinterpret_cast<half2*>(G + g * LDK + e) = __floats2half2_rn(o[et][0] * inv0 * sigm(a.x), o[et][1] * inv0 * sigm(a.y));
      *reinterpret_cast<half2*>(G + (g + 8) * LDK + e) = __floats2half2_rn(o[et][2] * inv1 * sigm(b2.x), o[et][3] * inv1 * sigm(b2.y));
    }
    __syncwarp();
    constexpr int PER = D / 8, G_PER = D / 16;
#pragma unroll
    for (int k = 0; k < G_PER; ++k) {
      int u = k * 32 + lane, rq = u / PER, c = (u % PER) * 8, i = q0 + warp * 16 + rq;
      if (i < QN) *reinterpret_cast<uint4*>(out + outAt(i) + c) = *reinterpret_cast<const uint4*>(G + rq * LDK + c);
    }
    return;
  }
  // gates read first and stored as pairs: with a store between every load the compiler
  // cannot reorder (out may alias qkvg) and the epilogue was 24 serial round trips, 10 us
  half2 ga[D / 8], gb[D / 8];
#pragma unroll
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    ga[et] = i0 < QN ? *reinterpret_cast<const half2*>(base + (STRIDED ? pos(i0) : (size_t)i0 * W4) + 3 * Wd + e) : half2{};
    gb[et] = i1 < QN ? *reinterpret_cast<const half2*>(base + (STRIDED ? pos(i1) : (size_t)i1 * W4) + 3 * Wd + e) : half2{};
  }
#pragma unroll
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    float2 a = __half22float2(ga[et]), b2 = __half22float2(gb[et]);
    if (i0 < QN) *reinterpret_cast<half2*>(out + outAt(i0) + e) =
        __floats2half2_rn(o[et][0] * inv0 * sigm(a.x), o[et][1] * inv0 * sigm(a.y));
    if (i1 < QN) *reinterpret_cast<half2*>(out + outAt(i1) + e) =
        __floats2half2_rn(o[et][2] * inv1 * sigm(b2.x), o[et][3] * inv1 * sigm(b2.y));
  }
}

// The same attention for few blocks (a token transformer: one row, 16 heads, a few hundred
// tokens - 80 blocks of 64 queries on 108 SMs). A block is 16 queries and KS warps, each warp
// streaming every KS-th key tile through its own double buffer, merged at the end
// (flash-decoding's split over keys, inside a block).
constexpr int FS_BK = 32;
template <int D> __host__ __device__ constexpr size_t fsStage() {
  return (size_t)2 * FS_BK * (D + 8) * 2 + (size_t)16 * (FS_BK + 8) * 2 + FS_BK * 4;
}
template <int D, int KS, bool MASKED = true>
__global__ void __launch_bounds__(KS * 32, (D == 48 && KS == 4) ? 2 : 1) flashSplitHalf(const half* __restrict__ qkvg, const half* __restrict__ bias,
    int biasStride, const float* __restrict__ mask, half* __restrict__ out, int n, int heads, size_t r0, bool tr, float scale,
    const float* qBias) {
  constexpr int BK = FS_BK, LDK = D + 8, LDB = BK + 8, LDO = D + 4;
  constexpr size_t STAGE = fsStage<D>();
  extern __shared__ __align__(16) unsigned char smem[];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  unsigned char* mine = smem + (size_t)warp * 2 * STAGE;
  auto Kst = [&](int s) { return (half*)(mine + s * STAGE); };
  auto Vst = [&](int s) { return Kst(s) + BK * LDK; };
  auto Bst = [&](int s) { return Vst(s) + BK * LDK; };
  auto Mst = [&](int s) { return (float*)(Bst(s) + 16 * LDB); };
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t rl = b / heads, r = r0 + rl;
  const int Wd = heads * D, W4 = 4 * Wd;
  const half* base = qkvg + rl * (size_t)n * W4 + h * D;
  int q0 = blockIdx.x * 16;
  constexpr int KV_CHUNKS = BK * (D / 8), KV_PER = (KV_CHUNKS + 31) / 32, B_PER = (16 * (BK / 8)) / 32;
  int kvBase[KV_PER], kvOff[KV_PER], kvJ[KV_PER];
#pragma unroll
  for (int k = 0; k < KV_PER; ++k) {
    int u = k * 32 + lane, jj = u / (D / 8), c = (u % (D / 8)) * 8;
    kvJ[k] = (KV_CHUNKS % 32 == 0 || u < KV_CHUNKS) ? jj : 1 << 30;
    kvOff[k] = jj * LDK + c;
    kvBase[k] = jj * W4 + Wd + c;
  }
  const half* biasHead = bias + (size_t)h * n * biasStride;
  int bBase[B_PER], bOff[B_PER], bC[B_PER]; bool bRow[B_PER];
#pragma unroll
  for (int k = 0; k < B_PER; ++k) {
    int u = k * 32 + lane, qi = u / (BK / 8), c = (u % (BK / 8)) * 8, i = q0 + qi;
    bRow[k] = i < n; bC[k] = c; bOff[k] = qi * LDB + c;
    bBase[k] = (i < n ? i : 0) * biasStride + c;
  }
  auto issue = [&](int j0, int st) {
    half *K = Kst(st), *V = Vst(st), *B = Bst(st);
    bool last = j0 + BK > n;
    if (!last) {
      const half* kvp = base + j0 * W4;
      const half* bp = biasHead + j0;
#pragma unroll
      for (int k = 0; k < KV_PER; ++k) {
        if (k + 1 < KV_PER || KV_CHUNKS % 32 == 0 || kvJ[k] < (1 << 30)) {
          const half* src = kvp + kvBase[k];
          cpAsync16(K + kvOff[k], src, true);
          cpAsync16(V + kvOff[k], src + Wd, true);
        }
      }
#pragma unroll
      for (int k = 0; k < B_PER; ++k) {
        cpAsync16(B + bOff[k], bp + bBase[k], bRow[k]);
      }
      if constexpr (MASKED) {
        int j = j0 + lane;
        Mst(st)[lane] = mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f;
      }
      cpCommit();
      return;
    }
#pragma unroll
    for (int k = 0; k < KV_PER; ++k) {
      if (kvJ[k] >= (1 << 30)) continue;
      bool ok = j0 + kvJ[k] < n;
      const half* src = base + (ok ? kvBase[k] + j0 * W4 : Wd);
      cpAsync16(K + kvOff[k], src, ok);
      cpAsync16(V + kvOff[k], src + Wd, ok);
    }
#pragma unroll
    for (int k = 0; k < B_PER; ++k) {
      bool ok = bRow[k] && j0 + bC[k] < n;
      cpAsync16(B + bOff[k], biasHead + (ok ? bBase[k] + j0 : 0), ok);
    }
    if constexpr (MASKED) {
      int j = j0 + lane;
      Mst(st)[lane] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f) : -INFINITY;
    }
    cpCommit();
  };
  int tiles = (n + BK - 1) / BK;
  if (warp < tiles) issue(warp * BK, 0);
  int i0 = q0 + g, i1 = i0 + 8;
  auto q2 = [&](int i, int e) -> uint32_t {
    if (i >= n) return 0u;
    float2 f = __half22float2(*reinterpret_cast<const half2*>(base + (size_t)i * W4 + e));
    if (qBias) { f.x += qBias[h * D + e]; f.y += qBias[h * D + e + 1]; }
    return pack2(f.x * scale * LOG2E, f.y * scale * LOG2E);
  };
  uint32_t qa[D / 16][4];
  if (warp < tiles) {
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      int e = ks * 16 + tig * 2;
      qa[ks][0] = q2(i0, e); qa[ks][1] = q2(i1, e); qa[ks][2] = q2(i0, e + 8); qa[ks][3] = q2(i1, e + 8);
    }
  }
  float o[D / 8][4] = {};
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  for (int tile = warp, it = 0; tile < tiles; tile += KS, ++it) {
    int st = it & 1;
    cpWait<0>();
    __syncwarp();
    if (tile + KS < tiles) issue((tile + KS) * BK, st ^ 1);
    const half *K = Kst(st), *V = Vst(st), *B = Bst(st); const float* Ms = Mst(st);
    const half* rowp = B + ((lane & 7) + ((lane >> 3) & 1) * 8) * LDB + (lane >> 4) * 8;
    float sv[BK / 8][4];
#pragma unroll
    for (int nt = 0; nt < BK / 8; nt += 2) {
      uint32_t r4[4];
      ldsm4(r4, rowp + nt * 8);
      if constexpr (D == 48) {
        uint32_t kb0[4], kb1[4], kb2[4];
        ldsm4(kb0, K + (nt * 8 + (lane & 7)) * LDK + (lane >> 3) * 8);
        ldsm4(kb1, K + ((nt + 1) * 8 + (lane & 7)) * LDK + (lane >> 3) * 8);
        ldsm4(kb2, K + ((nt + (lane >> 4)) * 8 + (lane & 7)) * LDK + 32 + ((lane >> 3) & 1) * 8);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          int jj = (nt + s) * 8 + tig * 2;
          float2 u = __half22float2(asH2(r4[2 * s])), v = __half22float2(asH2(r4[2 * s + 1]));
          if constexpr (MASKED) {
            float m0j = Ms[jj], m1j = Ms[jj + 1];
            u.x += m0j; u.y += m1j; v.x += m0j; v.y += m1j;
          } else if (tile == tiles - 1) {
            int jg = tile * BK + jj;
            if (jg >= n) u.x = v.x = -INFINITY;
            if (jg + 1 >= n) u.y = v.y = -INFINITY;
          }
          sv[nt + s][0] = u.x; sv[nt + s][1] = u.y; sv[nt + s][2] = v.x; sv[nt + s][3] = v.y;
        }
        mma16816(sv[nt], qa[0], kb0[0], kb0[1]);
        mma16816(sv[nt], qa[1], kb0[2], kb0[3]);
        mma16816(sv[nt + 1], qa[0], kb1[0], kb1[1]);
        mma16816(sv[nt + 1], qa[1], kb1[2], kb1[3]);
        mma16816(sv[nt], qa[2], kb2[0], kb2[1]);
        mma16816(sv[nt + 1], qa[2], kb2[2], kb2[3]);
      } else {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          int jj = (nt + s) * 8 + tig * 2;
          float2 u = __half22float2(asH2(r4[2 * s])), v = __half22float2(asH2(r4[2 * s + 1]));
          if constexpr (MASKED) {
            float m0j = Ms[jj], m1j = Ms[jj + 1];
            u.x += m0j; u.y += m1j; v.x += m0j; v.y += m1j;
          } else if (tile == tiles - 1) {
            int jg = tile * BK + jj;
            if (jg >= n) u.x = v.x = -INFINITY;
            if (jg + 1 >= n) u.y = v.y = -INFINITY;
          }
          sv[nt + s][0] = u.x; sv[nt + s][1] = u.y; sv[nt + s][2] = v.x; sv[nt + s][3] = v.y;
        }
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
          for (int k2 = 0; k2 < D / 32 + (D % 32 ? 1 : 0); ++k2) {
            if constexpr (D % 32 != 0) {
              if (k2 * 2 + 1 >= D / 16) {
                uint32_t kb[2];
                ldsm2(kb, K + ((nt + s) * 8 + (lane & 7)) * LDK + k2 * 32 + ((lane >> 3) & 1) * 8);
                mma16816(sv[nt + s], qa[k2 * 2], kb[0], kb[1]);
                continue;
              }
            }
            uint32_t kb[4];
            ldsm4(kb, K + ((nt + s) * 8 + (lane & 7)) * LDK + k2 * 32 + (lane >> 3) * 8);
            mma16816(sv[nt + s], qa[k2 * 2], kb[0], kb[1]);
            if (k2 * 2 + 1 < D / 16) mma16816(sv[nt + s], qa[k2 * 2 + 1], kb[2], kb[3]);
          }
        }
      }
    }
    float t0 = fmaxf(fmaxf(fmaxf(sv[0][0], sv[0][1]), fmaxf(sv[1][0], sv[1][1])),
                     fmaxf(fmaxf(sv[2][0], sv[2][1]), fmaxf(sv[3][0], sv[3][1])));
    float t1 = fmaxf(fmaxf(fmaxf(sv[0][2], sv[0][3]), fmaxf(sv[1][2], sv[1][3])),
                     fmaxf(fmaxf(sv[2][2], sv[2][3]), fmaxf(sv[3][2], sv[3][3])));
    t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
    t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
    uint32_t vb[2][4];
    ldsm4t(vb[0], V + (((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (lane >> 4) * 8);
    float n0 = t0, n1 = t1;
    if (it > 0) {
      n0 = fmaxf(m0, t0); n1 = fmaxf(m1, t1);
      float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
      l0 *= c0; l1 *= c1;
#pragma unroll
      for (int et = 0; et < D / 8; ++et) { o[et][0] *= c0; o[et][1] *= c0; o[et][2] *= c1; o[et][3] *= c1; }
    }
    m0 = n0; m1 = n1;
#pragma unroll
    for (int t = 0; t < BK / 16; ++t) {
      uint32_t pa[4] = { ex2h2(pack2(sv[2 * t][0] - n0, sv[2 * t][1] - n0)), ex2h2(pack2(sv[2 * t][2] - n1, sv[2 * t][3] - n1)),
                         ex2h2(pack2(sv[2 * t + 1][0] - n0, sv[2 * t + 1][1] - n0)),
                         ex2h2(pack2(sv[2 * t + 1][2] - n1, sv[2 * t + 1][3] - n1)) };
      mma16816_lsum(l0, l1, pa, ONES_H2, ONES_H2);
#pragma unroll
      for (int et = 0; et < D / 8; et += 2) {
        const int cur = (t * (D / 16) + (et >> 1)) & 1;
        if (et + 2 < D / 8) ldsm4t(vb[cur ^ 1], V + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (et + 2 + (lane >> 4)) * 8);
        else if (t + 1 < BK / 16) ldsm4t(vb[cur ^ 1], V + ((t + 1) * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (lane >> 4) * 8);
        mma16816(o[et], pa, vb[cur][0], vb[cur][1]);
        mma16816(o[et + 1], pa, vb[cur][2], vb[cur][3]);
      }
    }
  }
  __shared__ float Mx[KS * 16], Ls[KS * 16];
  if (tig == 0) { Mx[warp * 16 + g] = m0; Mx[warp * 16 + g + 8] = m1; Ls[warp * 16 + g] = l0; Ls[warp * 16 + g + 8] = l1; }
  if constexpr (KS == 4 && (D / 2) % 8 == 0) {
    constexpr int GPT2 = (D / 2) / 8;
    int row = threadIdx.x & 15, c0 = (threadIdx.x >> 4) * 2, i = q0 + row;
    half2 gate2[GPT2];
#pragma unroll
    for (int k = 0; k < GPT2; ++k) {
      int e = c0 + k * 16;
      gate2[k] = i < n ? *reinterpret_cast<const half2*>(base + (size_t)i * W4 + 3 * Wd + e) : half2{};
    }
    __syncwarp();
    float* myOs = (float*)mine;
#pragma unroll
    for (int et = 0; et < D / 8; ++et) {
      int e = et * 8 + tig * 2;
      float* row0 = myOs + g * LDO + e;
      *reinterpret_cast<float2*>(row0) = make_float2(o[et][0], o[et][1]);
      *reinterpret_cast<float2*>(row0 + 8 * LDO) = make_float2(o[et][2], o[et][3]);
    }
    __syncthreads();
    if (i >= n) return;
    float M = -INFINITY;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) M = fmaxf(M, Mx[ks * 16 + row]);
    float c[KS], L = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      c[ks] = exp2f(Mx[ks * 16 + row] - M);
      L += Ls[ks * 16 + row] * c[ks];
    }
    float invL = 1.f / L;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) c[ks] *= invL;
#pragma unroll
    for (int k = 0; k < GPT2; ++k) {
      int e = c0 + k * 16;
      float O0 = 0.f, O1 = 0.f;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        const float* ksOs = (const float*)(smem + (size_t)ks * 2 * STAGE);
        float2 ov = *reinterpret_cast<const float2*>(ksOs + row * LDO + e);
        O0 += ov.x * c[ks]; O1 += ov.y * c[ks];
      }
      float2 gv = __half22float2(gate2[k]);
      *reinterpret_cast<half2*>(out + (rl * n + i) * Wd + h * D + e) =
          __floats2half2_rn(O0 * sigm(gv.x), O1 * sigm(gv.y));
    }
    return;
  }
  // merge the KS partial softmaxes: [ks][16 rows][LDO] outputs, [ks][16] maxima and sums
  __syncthreads();
  constexpr int D2 = D / 2, GPT2 = (16 * D2 + KS * 32 - 1) / (KS * 32);
  half2 gate2[GPT2];
#pragma unroll
  for (int k = 0; k < GPT2; ++k) {
    int t2 = threadIdx.x + k * KS * 32, row = t2 / D2, e = (t2 % D2) * 2, i = q0 + row;
    gate2[k] = t2 < 16 * D2 && i < n ? *reinterpret_cast<const half2*>(base + (size_t)i * W4 + 3 * Wd + e) : half2{};
  }
  float* Os = (float*)smem;
#pragma unroll
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    float* row0 = Os + ((size_t)warp * 16 + g) * LDO + e; float* row1 = row0 + 8 * LDO;
    *reinterpret_cast<float2*>(row0) = make_float2(o[et][0], o[et][1]);
    *reinterpret_cast<float2*>(row1) = make_float2(o[et][2], o[et][3]);
  }
  if (threadIdx.x < 16) {
    int row = threadIdx.x;
    float M = -INFINITY;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) M = fmaxf(M, Mx[ks * 16 + row]);
    float L = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      float c = exp2f(Mx[ks * 16 + row] - M);
      Mx[ks * 16 + row] = c;
      L += Ls[ks * 16 + row] * c;
    }
    Ls[row] = 1.f / L;
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < GPT2; ++k) {
    int t2 = threadIdx.x + k * KS * 32, row = t2 / D2, e = (t2 % D2) * 2, i = q0 + row;
    if (t2 >= 16 * D2 || i >= n) continue;
    float invL = Ls[row], O0 = 0.f, O1 = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      float c = Mx[ks * 16 + row];
      float2 ov = *reinterpret_cast<const float2*>(Os + ((size_t)ks * 16 + row) * LDO + e);
      O0 += ov.x * c; O1 += ov.y * c;
    }
    float2 gv = __half22float2(gate2[k]);
    *reinterpret_cast<half2*>(out + (rl * n + i) * Wd + h * D + e) =
        __floats2half2_rn(O0 * invL * sigm(gv.x), O1 * invL * sigm(gv.y));
  }
}
template <int D, int KS>
void flashSplitHalfAt(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                      int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias) {
  size_t smem = std::max((size_t)KS * 2 * fsStage<D>(), (size_t)KS * 16 * (D + 6) * 4);
  if (mask) {
    static bool done = false;
    if (!done) { smemAttr((flashSplitHalf<D, KS, true>), (int)smem); done = true; }
    flashSplitHalf<D, KS, true><<<dim3((n + 15) / 16, (unsigned)(rows * heads)), 32 * KS, smem, STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
  } else {
    static bool done = false;
    if (!done) { smemAttr((flashSplitHalf<D, KS, false>), (int)smem); done = true; }
    flashSplitHalf<D, KS, false><<<dim3((n + 15) / 16, (unsigned)(rows * heads)), 32 * KS, smem, STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
  }
}

// The precise path: one thread a query row, f32 throughout, key tiles of 64 in shared memory.
template <int D>
__global__ void flashGridF32(const float* __restrict__ qkvg, const float* __restrict__ bias, int biasStride,
                             const float* __restrict__ mask, float* __restrict__ out, int n, int heads, size_t r0, bool tr, float scale) {
  constexpr int BQ = 64, BK = 64;
  __shared__ float Ks[BK][D + 1], Vs[BK][D + 1], Ms[BK];
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t rl = b / heads, r = r0 + rl;
  const int Wd = heads * D, W4 = 4 * Wd;
  const float* base = qkvg + rl * (size_t)n * W4 + h * D;
  int i = blockIdx.x * BQ + threadIdx.x;
  bool live = i < n;
  float q[D], o[D];
  for (int e = 0; e < D; ++e) { q[e] = live ? base[(size_t)i * W4 + e] * scale : 0.f; o[e] = 0.f; }
  float m = -INFINITY, l = 0.f;
  const float* brow = bias + ((size_t)h * n + (live ? i : 0)) * biasStride;
  for (int j0 = 0; j0 < n; j0 += BK) {
    __syncthreads();
    for (int t = threadIdx.x; t < BK * D; t += BQ) {
      int jj = t / D, e = t % D, j = j0 + jj;
      Ks[jj][e] = j < n ? base[(size_t)j * W4 + Wd + e] : 0.f;
      Vs[jj][e] = j < n ? base[(size_t)j * W4 + 2 * Wd + e] : 0.f;
    }
    if (threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Ms[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f) : -INFINITY;
    }
    __syncthreads();
    for (int jj = 0; jj < BK && j0 + jj < n; ++jj) {
      float dot = 0.f;
      for (int e = 0; e < D; ++e) dot += q[e] * Ks[jj][e];
      float v = dot + brow[j0 + jj] + Ms[jj];
      float mNew = fmaxf(m, v), c = expf(m - mNew), p = expf(v - mNew);
      l = l * c + p;
      for (int e = 0; e < D; ++e) o[e] = o[e] * c + p * Vs[jj][e];
      m = mNew;
    }
  }
  if (live) {
    const float* gp = base + (size_t)i * W4 + 3 * Wd;
    float* op = out + (rl * n + i) * Wd + h * D;
    for (int e = 0; e < D; ++e) op[e] = o[e] / l * (1.f / (1.f + expf(-gp[e])));
  }
}

// TWO query tiles a warp (32 rows, BQ = 32 x WARPS), f16 scores, unmasked, dense rows, cp.async double-buffered:
// every K and V fragment a warp loads from shared memory feeds both tiles' MMAs. With heads 32 wide the one-tile
// kernel reads ~10 bytes of shared memory a score (K and V 8, the bias 2) against 64 multiply-adds - bound by
// that bandwidth (~240 cycles a block-tile for ~54 of tensor work); this halves the K/V part.
// RR: grid rows a block (WARPS warps each, the same head and queries): the bias tile, the same for every
// row, is read from L2 once for all RR of them - at RR 1 it is two thirds of the block's L2 traffic
// K and V rows: at D 32 unpadded 64-byte rows, each 16-byte chunk XOR-swizzled by (row >> 1) & 3, so the
// cp.async writes (two rows a 128-byte line) and the 8-row ldmatrix reads both hit distinct banks - the
// padded 80-byte rows were conflict-free to read and conflicting to write: --bench-grid 5.65 -> 5.24 ms at
// 1,044 tokens, 0.204 -> 0.191 at 300, bit-identical. Otherwise D + 8 as before. (The same swizzle on the
// bias tile - 128-byte rows, chunk ^ row & 7 - was measured 3.6% slower: wider rows, more address math.)
// At D 8 (AF2's extra stack) a row is one 16-byte chunk, unpadded: eight rows are 128 contiguous bytes, conflict-free
template <int D> __host__ __device__ constexpr int fa2Ldk() { return D == 32 || D == 8 ? D : D + 8; }
template <int D> __device__ __forceinline__ int fa2Kv(int r, int c) {
  if constexpr (D == 32) return r * D + (c ^ (((r >> 1) & 3) << 3));
  else if constexpr (D == 8) return r * 8 + c;
  else return r * (D + 8) + c;
}
template <int D, int WARPS, int BK, int MT = 2, int RR = 1> __host__ __device__ constexpr size_t fa2Stage() {
  return (size_t)RR * 2 * BK * fa2Ldk<D>() * 2 + (size_t)(16 * MT * WARPS) * (BK + 8) * 2;
}
// (4 warps a block - every shipped form - asks the compiler for three blocks an SM, which it reached anyway at 168 registers:
// its schedule moves, --bench-grid 1.20 -> 1.19 ms at 1,000 tokens, 4.60 -> 4.51 at 2,000, 40.1 -> 39.4 at 6,000, byte-
// identical; four blocks spill, 4-8% slower; 32-key tiles at four or five blocks an SM are 3% and 40% behind at 4,000)
// NB: no bias at all (AF2's MSA column attention): the scores start at zero and no bias tile is loaded
// ONE: one stage, the next tile held in registers across the tile's compute and stored between two barriers -
// a T4's form (no cp.async; two stages are 39 KB, one block of 4 warps an SM there, one is three)
// (an FP8 part runs flash8 below instead, through the pair track: e4m3 MMAs, half of these)
// OT: the output's type - f16, or bf16 (cuda/af2's bf16 MSA, whose output projection then reads bf16)
template <int D, int WARPS, int BK, int MT = 2, int RR = 1, bool NB = false, bool ONE = false, class OT = half, bool SPEC = false>
__global__ void __launch_bounds__(WARPS * RR * 32, (!ONE && WARPS * RR == 4 && BK <= 48 && (D <= 32 || (D == 64 && MT == 1))) ? 3 : 1) flashGrid2R(const half* __restrict__ qkvg, const half* __restrict__ bias,
    int biasStride, OT* __restrict__ out, int n, int heads, float scale, const float* qBias, size_t rowsTotal,
    size_t rowStride, size_t posStride, size_t outRowStride, size_t outPosStride, int sw) {
  // strides in elements: a grid row's qkvg, a position's within it, and the output's - the dense layout is
  // (n * 4W, 4W, n * W, W); AF2's attention ACROSS a tensor's leading axis passes its own
  constexpr int BQ = 16 * MT * WARPS, LDK = fa2Ldk<D>(), LDB = BK + 8, NT = WARPS * RR * 32, NTR = WARPS * 32;
  constexpr size_t STAGE = fa2Stage<D, WARPS, BK, MT, RR>();
  extern __shared__ __align__(16) unsigned char smem[];
  // the thread's row of the block's RR (constants at RR 1: a runtime offset in every address costs 18%)
  const int rg = RR == 1 ? 0 : (int)threadIdx.x / NTR, tr = RR == 1 ? (int)threadIdx.x : (int)threadIdx.x % NTR;
  auto Kst = [&](int s) { return (half*)(smem + s * STAGE) + rg * 2 * BK * LDK; };
  auto Vst = [&](int s) { return Kst(s) + BK * LDK; };
  auto Bst = [&](int s) { return (half*)(smem + s * STAGE) + RR * 2 * BK * LDK; };
  int warp = tr >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const size_t rowsHere = rowsTotal, perHead = (rowsHere + RR - 1) / RR;
  // sw > 0: the blocks of one head walk sw row groups across each query tile before the next tile, so the resident
  // blocks share a bias tile (n x 64 halves, re-read from HBM by every row group once a head's bias outgrows L2) as
  // well as their rows' K and V; 0: the query tiles of one row group, then the next (K/V shared, bias not)
  size_t b, qt;
  if (sw > 0) {
    const size_t nq = gridDim.x, L = (size_t)blockIdx.y * nq + blockIdx.x, perHeadBlocks = perHead * nq;
    const size_t r = L % perHeadBlocks, group = r / ((size_t)sw * nq), within = r % ((size_t)sw * nq);
    const size_t left = perHead - group * sw, gsz = left < (size_t)sw ? left : (size_t)sw;
    b = (L / perHeadBlocks) * perHead + group * sw + within % gsz; qt = within / gsz;
  } else { b = blockIdx.y; qt = blockIdx.x; }
  int h = (int)(b / perHead); size_t rl = (b % perHead) * RR + rg;
  const bool live = RR == 1 || rl < rowsHere;                        // a last block's spare row computes, stores nothing
  if (!live) rl = rowsHere - 1;
  const int Wd = heads * D, W4 = 4 * Wd;
  const half* base = qkvg + rl * rowStride + h * D;
  int q0 = (int)qt * BQ;
  constexpr int KV_CHUNKS = BK * (D / 8), B_CHUNKS = BQ * (BK / 8);
  const half* biasHead = bias + (size_t)h * n * biasStride;
  constexpr int KV_PER = (KV_CHUNKS + NTR - 1) / NTR, B_PER = (B_CHUNKS + NT - 1) / NT;
  const half* kvSrc[KV_PER]; int kvOff[KV_PER], kvJ[KV_PER];
#pragma unroll
  for (int k = 0; k < KV_PER; ++k) {
    int u = k * NTR + tr, jj = u / (D / 8), c = (u % (D / 8)) * 8;
    kvJ[k] = (KV_CHUNKS % NTR == 0 || u < KV_CHUNKS) ? jj : 1 << 30;
    kvOff[k] = fa2Kv<D>(jj, c);
    kvSrc[k] = base + (size_t)jj * posStride + Wd + c;
  }
  const half* bSrc[B_PER]; int bOff[B_PER], bC[B_PER]; bool bRow[B_PER];
#pragma unroll
  for (int k = 0; k < B_PER; ++k) {
    int u = k * NT + threadIdx.x, qi = u / (BK / 8), c = (u % (BK / 8)) * 8, i = q0 + qi;
    bool valid = B_CHUNKS % NT == 0 || u < B_CHUNKS;
    bRow[k] = valid && i < n; bC[k] = valid ? c : 1 << 30; bOff[k] = valid ? qi * LDB + c : 0;
    bSrc[k] = biasHead + (size_t)(bRow[k] ? i : 0) * biasStride + c;
  }
  // a whole tile's issue walks running pointers, a key tile on each call: the bounds and the 64-bit offsets
  // are the last tile's alone (they were ~70 of a tile's ~430 instructions a warp, for its 9 copies) -
  // 2274 -> 2211 ms of AF3's grid attention at 1,044 tokens, byte-identical
  const half* kvCur[KV_PER]; const half* bCur[B_PER];
#pragma unroll
  for (int k = 0; k < KV_PER; ++k) kvCur[k] = kvSrc[k];
#pragma unroll
  for (int k = 0; k < B_PER; ++k) bCur[k] = bSrc[k];
  const size_t kvStep = (size_t)BK * posStride;
  auto issue = [&](int j0, int st) {
    half *K = Kst(st), *V = Vst(st), *B = Bst(st);
    bool last = j0 + BK > n;
    if (!last) {
#pragma unroll
      for (int k = 0; k < KV_PER; ++k) {
        if (KV_CHUNKS % NTR == 0 || kvJ[k] < (1 << 30)) {
          cpAsync16(K + kvOff[k], kvCur[k], true);
          cpAsync16(V + kvOff[k], kvCur[k] + Wd, true);
        }
        kvCur[k] += kvStep;
      }
      if constexpr (!NB) {
#pragma unroll
        for (int k = 0; k < B_PER; ++k) {
          if (B_CHUNKS % NT == 0 || bC[k] < (1 << 30)) cpAsync16(B + bOff[k], bRow[k] ? bCur[k] : biasHead, bRow[k]);
          bCur[k] += BK;
        }
      }
      cpCommit();
      return;
    }
#pragma unroll
    for (int k = 0; k < KV_PER; ++k) {
      if (kvJ[k] >= (1 << 30)) continue;
      bool ok = j0 + kvJ[k] < n;
      cpAsync16(K + kvOff[k], ok ? kvCur[k] : base, ok);
      cpAsync16(V + kvOff[k], (ok ? kvCur[k] : base) + Wd, ok);
    }
    if constexpr (!NB) {   // (braced: an unbraced if constexpr over a #pragma'd loop lost the commit below)
#pragma unroll
      for (int k = 0; k < B_PER; ++k) {
        if (B_CHUNKS % NT != 0 && bC[k] >= (1 << 30)) continue;
        bool ok = bRow[k] && j0 + bC[k] < n;
        cpAsync16(B + bOff[k], ok ? bCur[k] : biasHead, ok);
      }
    }
    cpCommit();
  };
  // the first key tile's copies in flight before the queries' loads, which they overlap (they were taken after them, and
  // each block's start waited on both in turn - at 261 tokens, five key tiles a block, a tenth of its stall samples)
  if constexpr (!ONE) issue(0, 0);
  // the gate's rows (this row group's BQ queries, D halves each) copied into the stage the last tile leaves idle, past the
  // output's staging, under that tile's compute - the epilogue read them from global memory after the loop and waited
  constexpr size_t YS_BYTES = (size_t)WARPS * RR * 16 * MT * (D + 8) * 2, G_OFF = (YS_BYTES + 127) / 128 * 128;
  // (only where it measured faster: the 8-wide heads (the extra stack, 68 -> 63 ms of a 261-residue fold) and the
  // bias-free column attention (294 -> 290 at 600); the 64-key 32-wide forms already sit at the register cap and
  // spilled more with the copy loop - the MSA row attention 366 -> 396 ms at 600)
  constexpr bool GATE_SM = !ONE && G_OFF + (size_t)RR * BQ * D * 2 <= STAGE && D % 8 == 0 && (D <= 16 || NB);
  auto issueGate = [&](int s) {
    half* G = (half*)(smem + (size_t)s * STAGE + G_OFF) + (size_t)rg * BQ * D;
    constexpr int CH = BQ * (D / 8);
#pragma unroll
    for (int k = 0; k < (CH + NTR - 1) / NTR; ++k) {
      const int u = k * NTR + tr;
      if (CH % NTR != 0 && u >= CH) break;
      const int qi = u / (D / 8), c = (u % (D / 8)) * 8, i = q0 + qi;
      cpAsync16(G + qi * D + c, i < n ? base + (size_t)i * posStride + 3 * Wd + c : base, i < n);
    }
    cpCommit();
  };
  // ONE's halves of `issue`: its loads into registers (the same pieces, the same running pointers, the same zeros
  // past the end), and the registers into the stage
  uint4 rkv[ONE ? KV_PER : 1][2], rb[ONE && B_PER > 0 ? B_PER : 1];
  auto loadR = [&](int j0) {
    if constexpr (ONE) {
      const uint4 z = make_uint4(0, 0, 0, 0);
      bool last = j0 + BK > n;
#pragma unroll
      for (int k = 0; k < KV_PER; ++k) {
        if (!(KV_CHUNKS % NTR == 0 || kvJ[k] < (1 << 30))) { if (!last) kvCur[k] += kvStep; continue; }
        bool ok = !last || j0 + kvJ[k] < n;
        rkv[k][0] = ok ? *reinterpret_cast<const uint4*>(kvCur[k]) : z;
        rkv[k][1] = ok ? *reinterpret_cast<const uint4*>(kvCur[k] + Wd) : z;
        if (!last) kvCur[k] += kvStep;
      }
      if constexpr (!NB) {
#pragma unroll
        for (int k = 0; k < B_PER; ++k) {
          bool ok = last ? bRow[k] && j0 + bC[k] < n : bRow[k];
          rb[k] = ok ? *reinterpret_cast<const uint4*>(bCur[k]) : z;
          if (!last) bCur[k] += BK;
        }
      }
    }
  };
  auto storeR = [&]() {
    if constexpr (ONE) {
      half *K = Kst(0), *V = Vst(0), *B = Bst(0);
#pragma unroll
      for (int k = 0; k < KV_PER; ++k) {
        if (!(KV_CHUNKS % NTR == 0 || kvJ[k] < (1 << 30))) continue;
        *reinterpret_cast<uint4*>(K + kvOff[k]) = rkv[k][0];
        *reinterpret_cast<uint4*>(V + kvOff[k]) = rkv[k][1];
      }
      if constexpr (!NB) {
#pragma unroll
        for (int k = 0; k < B_PER; ++k) *reinterpret_cast<uint4*>(B + bOff[k]) = rb[k];
      }
    }
  };
  auto q2 = [&](int i, int e) -> uint32_t {
    if (i >= n) return 0u;
    half2 v = *reinterpret_cast<const half2*>(base + (size_t)i * posStride + e);
    float2 f = __half22float2(v);
    if (qBias) { f.x += qBias[h * D + e]; f.y += qBias[h * D + e + 1]; }
    return pack2(f.x * scale * LOG2E, f.y * scale * LOG2E);
  };
  auto ib = [&](int mt) { return q0 + warp * 16 * MT + mt * 16 + g; };
  const bool warpLive = q0 + warp * 16 * MT < n;
  // D 8: one m16n8k8 A fragment a tile (rows g and g + 8, k 2 tig and one more) in qa[mt][0][0..1]
  static_assert(D != 8 || BK % 32 == 0, "an 8-wide head's K and V are read four key tiles an ldmatrix");
  uint32_t qa[MT][(D + 15) / 16][4];
  if constexpr (D == 8) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) { qa[mt][0][0] = q2(ib(mt), tig * 2); qa[mt][0][1] = q2(ib(mt) + 8, tig * 2); }
  } else {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        int e = ks * 16 + tig * 2;
        qa[mt][ks][0] = q2(ib(mt), e); qa[mt][ks][1] = q2(ib(mt) + 8, e);
        qa[mt][ks][2] = q2(ib(mt), e + 8); qa[mt][ks][3] = q2(ib(mt) + 8, e + 8);
      }
    }
  }
  float o[MT][D / 8][4] = {};
  float mx[MT][2], lsum0[MT] = {}, lsum2[MT] = {};
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) mx[mt][0] = mx[mt][1] = -INFINITY;
  int tiles = (n + BK - 1) / BK;
  if constexpr (ONE) { loadR(0); storeR(); }
  for (int tile = 0; tile < tiles; ++tile) {
    int st = ONE ? 0 : tile & 1;
    if constexpr (ONE) {
      if (tile + 1 < tiles) loadR((tile + 1) * BK);
      __syncthreads();
    } else {
      cpWait<0>();
      __syncthreads();
      if (tile + 1 < tiles) issue((tile + 1) * BK, st ^ 1);
      else if constexpr (GATE_SM) issueGate(st ^ 1);
    }
    // a warp whose query rows all lie past n (the last query tile's padding: at 261 tokens one warp-tile in ten) takes
    // the loads and the barriers and nothing else - warp-uniform, so a live warp's arithmetic is the same
    if (warpLive) {
    const half *K = Kst(st), *V = Vst(st), *B = Bst(st);
    uint32_t sh[MT][BK / 8][2];
    // the bias tile straight into the score accumulators' layout by ldmatrix - an 8x8 matrix's thread t holds row t / 4,
    // columns 2 (t % 4) and one more, which is an m16n8 accumulator's - four of them (two key tiles of 8, rows g and
    // g + 8) an instruction, where 32-bit loads took one a pair (24 a warp a tile against 6): --bench-grid 17.80 -> 17.45
    // ms at 4,000 tokens (256 rows), the fold's grid attention -1.1% at 2,964 tokens and +0.7% at 988; byte-identical
    static_assert(BK % 16 == 0, "key tiles in pairs of 8");
    if constexpr (!NB) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const half* rowp = B + (warp * 16 * MT + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDB + (lane >> 4) * 8;
#pragma unroll
        for (int nt = 0; nt < BK / 8; nt += 2) {
          uint32_t r4[4];
          ldsm4(r4, rowp + nt * 8);
          sh[mt][nt][0] = r4[0]; sh[mt][nt][1] = r4[1]; sh[mt][nt + 1][0] = r4[2]; sh[mt][nt + 1][1] = r4[3];
        }
      }
    }
    uint32_t kb[2][4];
    if constexpr (D == 32 || D == 64) ldsm4(kb[0], K + fa2Kv<D>(lane & 7, (lane >> 3) * 8));
    else if constexpr (D == 16) ldsm4(kb[0], K + fa2Kv<D>((lane >> 4) * 8 + (lane & 7), ((lane >> 3) & 1) * 8));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
        int jj = nt * 8 + tig * 2;
        if constexpr (NB) { sh[mt][nt][0] = 0u; sh[mt][nt][1] = 0u; }
        if (tile == tiles - 1) {
          int jg = tile * BK + jj;
          if (jg >= n) { sh[mt][nt][0] = (sh[mt][nt][0] & 0xFFFF0000u) | 0xFC00u; sh[mt][nt][1] = (sh[mt][nt][1] & 0xFFFF0000u) | 0xFC00u; }
          if (jg + 1 >= n) { sh[mt][nt][0] = (sh[mt][nt][0] & 0xFFFFu) | 0xFC000000u; sh[mt][nt][1] = (sh[mt][nt][1] & 0xFFFFu) | 0xFC000000u; }
        }
      }
    }
    if constexpr (D == 8) {
#pragma unroll
      for (int nt = 0; nt < BK / 8; nt += 4) {
        uint32_t kb8[4];                                   // four key tiles' B fragments (k 2 tig, key g)
        ldsm4(kb8, K + fa2Kv<D>((nt + (lane >> 3)) * 8 + (lane & 7), 0));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) mma1688h(sh[mt][nt + q], qa[mt][0][0], qa[mt][0][1], kb8[q]);
        }
      }
    }
    if constexpr (D == 32) {
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
        const int cur = nt & 1;
        if (nt + 1 < BK / 8) ldsm4(kb[cur ^ 1], K + fa2Kv<D>((nt + 1) * 8 + (lane & 7), (lane >> 3) * 8));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816h(sh[mt][nt], qa[mt][0], kb[cur][0], kb[cur][1]);
          mma16816h(sh[mt][nt], qa[mt][1], kb[cur][2], kb[cur][3]);
        }
      }
    } else if constexpr (D == 16) {
#pragma unroll
      for (int nt = 0; nt < BK / 8; nt += 2) {
        const int cur = (nt >> 1) & 1;
        if (nt + 2 < BK / 8) ldsm4(kb[cur ^ 1], K + fa2Kv<D>((nt + 2 + (lane >> 4)) * 8 + (lane & 7), ((lane >> 3) & 1) * 8));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816h(sh[mt][nt], qa[mt][0], kb[cur][0], kb[cur][1]);
          mma16816h(sh[mt][nt + 1], qa[mt][0], kb[cur][2], kb[cur][3]);
        }
      }
    } else if constexpr (D == 64) {
#pragma unroll
      for (int s = 0; s < (BK / 8) * 2; ++s) {
        const int cur = s & 1, nt = s >> 1, k2 = s & 1;
        if (s + 1 < (BK / 8) * 2) {
          const int nnt = (s + 1) >> 1, nk2 = (s + 1) & 1;
          ldsm4(kb[cur ^ 1], K + fa2Kv<D>(nnt * 8 + (lane & 7), nk2 * 32 + (lane >> 3) * 8));
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816h(sh[mt][nt], qa[mt][2 * k2], kb[cur][0], kb[cur][1]);
          mma16816h(sh[mt][nt], qa[mt][2 * k2 + 1], kb[cur][2], kb[cur][3]);
        }
      }
    } else if constexpr (D != 8) {
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
        for (int k2 = 0; k2 < D / 32 + (D % 32 ? 1 : 0); ++k2) {
          if constexpr (D % 32 != 0) {
            if (k2 * 2 + 1 >= D / 16) {
              uint32_t kb2[2];
              ldsm2(kb2, K + fa2Kv<D>(nt * 8 + (lane & 7), k2 * 32 + ((lane >> 3) & 1) * 8));
#pragma unroll
              for (int mt = 0; mt < MT; ++mt) mma16816h(sh[mt][nt], qa[mt][k2 * 2], kb2[0], kb2[1]);
              continue;
            }
          }
          uint32_t kb4[4];
          ldsm4(kb4, K + fa2Kv<D>(nt * 8 + (lane & 7), k2 * 32 + (lane >> 3) * 8));
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma16816h(sh[mt][nt], qa[mt][k2 * 2], kb4[0], kb4[1]);
            if (k2 * 2 + 1 < D / 16) mma16816h(sh[mt][nt], qa[mt][k2 * 2 + 1], kb4[2], kb4[3]);
          }
        }
      }
    }
    // (skipping the rescale when no row of the warp moved its max - a vote, each factor then exactly 1 - is
    // byte-identical and slower: 2211 -> 2251 ms of AF3's grid attention at 1,044 tokens, 111 -> 123 ms of
    // AF2's MSA column attention)
    // SPEC (speculative softmax, BioIR's ClaudeKit kernel's idea): a tile whose every row stays within 2^8 of the warp's
    // running maximum keeps it - its P exponentiated against the stale maximum (at most 2^8, an f16), the cross-lane
    // maximum, the two rescale exponentials and the O and l rescale skipped; any row past it takes the full step for
    // the warp (the first tile always). o / l is the same softmax whichever reference P was taken against
    __half2 hn[MT][2];
    float t0_[MT], t1_[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      __half2 x0 = asH2(sh[mt][0][0]), x1 = asH2(sh[mt][0][1]);
#pragma unroll
      for (int nt = 1; nt < BK / 8; ++nt) { x0 = __hmax2(x0, asH2(sh[mt][nt][0])); x1 = __hmax2(x1, asH2(sh[mt][nt][1])); }
      t0_[mt] = fmaxf(__low2float(x0), __high2float(x0)); t1_[mt] = fmaxf(__low2float(x1), __high2float(x1));
    }
    bool full = true;
    if constexpr (SPEC) {
      if (tile > 0) {
        bool over = false;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) over |= t0_[mt] > mx[mt][0] + 8.f || t1_[mt] > mx[mt][1] + 8.f;
        full = __any_sync(~0u, over);
        if (!full) {
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) { hn[mt][0] = __float2half2_rn(mx[mt][0]); hn[mt][1] = __float2half2_rn(mx[mt][1]); }
        }
      }
    }
    uint32_t v8[4], vb[2][4];
    if constexpr (D != 8) ldsm4t(vb[0], V + fa2Kv<D>(((lane >> 3) & 1) * 8 + (lane & 7), (lane >> 4) * 8));
    if (full) {
  #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        float t0 = t0_[mt], t1 = t1_[mt];
        t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
        if (tile == 0) {
          float n0 = __half2float(__float2half(t0)), n1 = __half2float(__float2half(t1));
          mx[mt][0] = n0; mx[mt][1] = n1;
          hn[mt][0] = __float2half2_rn(n0); hn[mt][1] = __float2half2_rn(n1);
        } else {
          float n0 = __half2float(__float2half(fmaxf(mx[mt][0], t0))), n1 = __half2float(__float2half(fmaxf(mx[mt][1], t1)));
          float c0 = exp2f(mx[mt][0] - n0), c1 = exp2f(mx[mt][1] - n1);
          mx[mt][0] = n0; mx[mt][1] = n1;
  #pragma unroll
          for (int et = 0; et < D / 8; ++et) { o[mt][et][0] *= c0; o[mt][et][1] *= c0; o[mt][et][2] *= c1; o[mt][et][3] *= c1; }
          lsum0[mt] *= c0; lsum2[mt] *= c1;
          hn[mt][0] = __float2half2_rn(n0); hn[mt][1] = __float2half2_rn(n1);
        }
      }
    }
#pragma unroll
    for (int t = 0; t < BK / 16; ++t) {
      uint32_t pa[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        // (a quarter to all of these on the FMA pipe instead of the MUFU - 2^x as a cubic in f16, 2^round(x) laid into
        // the exponent; relRMS 3.5-4.0e-4 from the f32 reference against 3.3e-4 - was measured 1-14% SLOWER at 1,000
        // and 4,000 tokens: the MUFU is busy as often as the tensor pipe, but neither binds, and the cubic's nine
        // instructions a pair cost more issue than the MUFU's waits)
        pa[mt][0] = ex2h2(asU32(__hsub2(asH2(sh[mt][2 * t][0]), hn[mt][0])));
        pa[mt][1] = ex2h2(asU32(__hsub2(asH2(sh[mt][2 * t][1]), hn[mt][1])));
        pa[mt][2] = ex2h2(asU32(__hsub2(asH2(sh[mt][2 * t + 1][0]), hn[mt][0])));
        pa[mt][3] = ex2h2(asU32(__hsub2(asH2(sh[mt][2 * t + 1][1]), hn[mt][1])));
        mma16816_lsum(lsum0[mt], lsum2[mt], pa[mt], ONES_H2, ONES_H2);
      }
      if constexpr (D == 8) {
        if (t % 2 == 0) ldsm4t(v8, V + fa2Kv<D>((t + (lane >> 4)) * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), 0));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma16816(o[mt][0], pa[mt], v8[(t % 2) * 2], v8[(t % 2) * 2 + 1]);
      }
#pragma unroll
      for (int et = 0; et < (D == 8 ? 0 : D / 8); et += 2) {
        const int cur = (t * (D / 16) + (et >> 1)) & 1;
        if (et + 2 < D / 8) ldsm4t(vb[cur ^ 1], V + fa2Kv<D>(t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), (et + 2 + (lane >> 4)) * 8));
        else if (t + 1 < BK / 16) ldsm4t(vb[cur ^ 1], V + fa2Kv<D>((t + 1) * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), (lane >> 4) * 8));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) { mma16816(o[mt][et], pa[mt], vb[cur][0], vb[cur][1]); mma16816(o[mt][et + 1], pa[mt], vb[cur][2], vb[cur][3]); }
      }
    }
    }
    if constexpr (ONE) {
      __syncthreads();                // every warp is done with the stage: the next tile goes in
      if (tile + 1 < tiles) storeR();
      __syncthreads();
    }
  }
  if constexpr (GATE_SM) { cpWait<0>(); __syncthreads(); }
  half2 ga[MT][D / 8], gb[MT][D / 8];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    int i0 = ib(mt), i1 = i0 + 8;
    if constexpr (GATE_SM) {
      const half* G = (const half*)(smem + (size_t)(tiles & 1) * STAGE + G_OFF) + (size_t)rg * BQ * D;
#pragma unroll
      for (int et = 0; et < D / 8; ++et) {
        int e = et * 8 + tig * 2;
        ga[mt][et] = *reinterpret_cast<const half2*>(G + (i0 - q0) * D + e);
        gb[mt][et] = *reinterpret_cast<const half2*>(G + (i1 - q0) * D + e);
      }
    } else {
#pragma unroll
      for (int et = 0; et < D / 8; ++et) {
        int e = et * 8 + tig * 2;
        ga[mt][et] = i0 < n ? *reinterpret_cast<const half2*>(base + (size_t)i0 * posStride + 3 * Wd + e) : half2{};
        gb[mt][et] = i1 < n ? *reinterpret_cast<const half2*>(base + (size_t)i1 * posStride + 3 * Wd + e) : half2{};
      }
    }
  }
  if constexpr (!GATE_SM && !ONE) __syncthreads();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    float inv0 = 1.f / lsum0[mt], inv1 = 1.f / lsum2[mt];
#pragma unroll
    for (int et = 0; et < D / 8; ++et) {
      int e = et * 8 + tig * 2;
      float2 a = __half22float2(ga[mt][et]), b2 = __half22float2(gb[mt][et]);
      // staged in the warp's own slice of the (now idle) stage memory, written below as whole rows
      half* Ys = (half*)smem + (size_t)(threadIdx.x >> 5) * (16 * MT) * (D + 8);
      auto cv = [](float x, float y) {
        if constexpr (std::is_same_v<OT, half>) return __floats2half2_rn(x, y); else return __floats2bfloat162_rn(x, y);
      };
      using OT2 = decltype(cv(0.f, 0.f));
      *reinterpret_cast<OT2*>(Ys + (mt * 16 + g) * (D + 8) + e) =
          cv(o[mt][et][0] * inv0 * sigm(a.x), o[mt][et][1] * inv0 * sigm(a.y));
      *reinterpret_cast<OT2*>(Ys + (mt * 16 + g + 8) * (D + 8) + e) =
          cv(o[mt][et][2] * inv1 * sigm(b2.x), o[mt][et][3] * inv1 * sigm(b2.y));
    }
  }
  // the warp's 16 MT rows of this head, 2 D bytes each, as 16-byte stores (D / 8 lanes a row) rather than
  // 4 bytes in each of 8 rows; every warp is past the loop's last barrier, so the stages are free
  static_assert((size_t)WARPS * RR * 16 * MT * (D + 8) * 2 <= 2 * STAGE, "the output staging fits in the stages");
  __syncwarp();
  {
    const half* Ys = (const half*)smem + (size_t)(threadIdx.x >> 5) * (16 * MT) * (D + 8);
    constexpr int PER = D / 8, ROWS_AT = 32 / PER;
#pragma unroll
    for (int r0 = 0; r0 < 16 * MT; r0 += ROWS_AT) {
      int r = r0 + lane / PER, c = (lane % PER) * 8;
      int i = q0 + warp * 16 * MT + r;
      if (live && i < n) *reinterpret_cast<uint4*>(out + rl * outRowStride + (size_t)i * outPosStride + h * D + c) = *reinterpret_cast<const uint4*>(Ys + r * (D + 8) + c);
    }
  }
}
// the unmasked, dense 32-wide grid attention through flashGrid2R (4 warps, 48-key tiles): --bench-grid on the
// A100 0.884 -> 0.753 ms at 500 tokens, 6.34 -> 5.02 at 1000 (80.7 -> 102 TFLOP/s), output relRMS 3.9e-4 from
// the f32-score kernel's. Three or four tiles a warp spill (0.95-1.0 ms at 500); eight warps of one tile with
// f16 scores reach 0.764. LOCALFOLD_FLASH_2R=0 keeps the one-tile kernel.
// It runs as 2 warps x 2 grid rows a block (RR 2): 64-query tiles pad a row less than 128-query ones (1,044
// queries are 1,088 against 1,152) while the two rows share the bias tile, so K/V and bias traffic stay the
// 4-warp form's - 6.27 -> 5.87 ms at 1,044 tokens, 0.251 -> 0.215 at 300, level at 500 (512 either way);
// bit-identical to RR 1. Software-pipelining the loop (tile t+1's Q K^T beside tile t's softmax and P V,
// three stages of shared memory) was measured and is slower: bit-identical, but 6.29 -> 6.79 ms at 1,044
// tokens with 32-key tiles (the occupancy the 48-key form keeps with two stages), and 5.77 -> 7.58 at 48
// keys where the third stage costs a block an SM - three warps a scheduler already overlap one warp's
// softmax with another's MMAs.
inline bool FLASH_2R = !getenv("LOCALFOLD_FLASH_2R") || atoi(getenv("LOCALFOLD_FLASH_2R"));
// the blocks' order (flashGrid2R's sw): four row groups a query tile once a head's bias (n^2 halves) is past ~24 MB of
// L2 - --bench-grid at 256 rows 75.6 -> 72.3 ms at 8,000 tokens, 42.5 -> 40.3 at 6,000, level at 3,000 and ~1% behind
// at 1,000; byte-identical (the same blocks, another order). LOCALFOLD_GRID_SWIZZLE=<sw> forces one, 0 off
// (three warps a block - 96 queries, K and V a third less L2 traffic a score, still 12 warps an SM - measured level at
// 6,000 tokens and 12% behind at 1,000: past the L2 (66% busy in Nsight Compute) the kernel is latency-bound at 168
// registers a thread, three blocks an SM; key tiles of 32, 64 or 80 are 2-5% behind 48 at 6,000 too)
inline int GRID_SWIZZLE = getenv("LOCALFOLD_GRID_SWIZZLE") ? atoi(getenv("LOCALFOLD_GRID_SWIZZLE")) : -1;
template <int D, int WARPS, int BK, int MT = 2, int RR = 1, bool NB = false, bool ONE = false, class OT = half, bool SPEC = false>
void flashGrid2RRun(const half* qkvg, const half* bias, int stride, OT* out, int n, int heads, size_t rows, float scale,
                    const float* qBias, size_t rowStride = 0, size_t posStride = 0, size_t outRowStride = 0,
                    size_t outPosStride = 0) {
  constexpr int BQ = 16 * MT * WARPS;
  const size_t W = (size_t)heads * D;
  if (!posStride) { rowStride = (size_t)n * 4 * W; posStride = 4 * W; outRowStride = (size_t)n * W; outPosStride = W; }
  // (the output is staged in the stages' memory, and one stage (ONE) can be smaller than its tiles' staging - four
  // tiles a warp overflowed it, an illegal access)
  const int bytes = std::max((ONE ? 1 : 2) * (int)fa2Stage<D, WARPS, BK, MT, RR>(), WARPS * RR * 16 * MT * (D + 8) * 2);
  static bool attr = false;
  if (!attr) { smemAttr((flashGrid2R<D, WARPS, BK, MT, RR, NB, ONE, OT, SPEC>), bytes); attr = true; }
  dim3 grid((n + BQ - 1) / BQ, (unsigned)((rows + RR - 1) / RR * heads));
  flashGrid2R<D, WARPS, BK, MT, RR, NB, ONE, OT, SPEC><<<grid, 32 * WARPS * RR, bytes, STREAM>>>(qkvg, bias, stride, out, n, heads, scale, qBias, rows,
                                                                              rowStride, posStride, outRowStride, outPosStride, GRID_SWIZZLE >= 0 ? GRID_SWIZZLE : n >= 3500 ? 4 : 0);
}
// 64-key tiles for the dense 32-wide grid attention where an SM holds two of its blocks (an A100, a Hopper part): on
// the A100 4-7% faster than 48 at every length measured - 256 to 8,192 tokens, 0.145 against 0.156 ms at 256, 4.59 against
// 4.93 at 1,024, 69.7 against 73.4 at 8,192 (256 rows, --bench-grid) - where the note above had 48 ahead at 6,000 for an
// older kernel; a 100 KB-an-SM part (an A6000, an L4, an RTX PRO 6000) would hold one 64-key block where it holds two
// of 48. (BioIR's sm_80 triangle attention takes 64-key tiles; theirs against ours is cuda/af3/README.md's, 2026-10-12.)
// LOCALFOLD_FLASH_2R_BK=48|64 forces either
// ...and its speculative softmax (flashGrid2R's SPEC; BioIR's ClaudeKit kernel's idea): --bench-grid 9.72 -> 9.36 ms at
// 3,072 tokens, 38.5 -> 37.0 at 6,144 (256 rows); a whole trunk pass at 2,964 tokens its two grid stages 6.85 -> 6.69 s
// each; 5CAJ self-templated 0.005 A from the plain softmax. The 48-key form too (a 100 KB-an-SM card's - an A6000, or an
// L4 with FP8 off): on a Colab L4 17.4-17.9 against 19.1-19.3 ms at 2,048 tokens, level at 512 (three alternations each;
// 256 rows); on the A100 9.65 against 10.19 at 3,072. LOCALFOLD_FLASH_SPEC=0 turns it off
inline bool FLASH_SPEC = !getenv("LOCALFOLD_FLASH_SPEC") || atoi(getenv("LOCALFOLD_FLASH_SPEC"));
inline bool flash2RWideKeys() {
  static const bool v = [] {
    if (const char* e = getenv("LOCALFOLD_FLASH_2R_BK")) return atoi(e) == 64;
    int dev, perSm = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    const size_t block = std::max(2 * fa2Stage<32, 2, 64, 2, 2>(), (size_t)2 * 2 * 16 * 2 * (32 + 8) * 2);
    return (size_t)perSm >= 2 * (block + 1024);            // (1 KB a block the driver reserves)
  }();
  return v;
}
// ...and for n keys: 64-key tiles unless 48-key ones pad n meaningfully less (a 64-key tile is ~2.4% cheaper a key where
// both fit exactly - 384, 768). On the A100, 48 against 64 (two rows a block, SPEC): the grid attention (rows = n)
// 0.182 / 0.191 ms at 261 tokens, 0.294 / 0.318 at 330, level at 600, 4.343 / 4.245 at 1,000; AF2's MSA row attention
// (512 rows, n keys) 0.280 / 0.293 at 261, 0.445 / 0.481 at 330, 0.928 / 0.999 at 600, 1.897 / 1.928 at 900, 1.193 / 1.134
// at 700 (64 there)
inline bool flash2RWideKeysFor(int n) {
  if (!flash2RWideKeys()) return false;
  if (getenv("LOCALFOLD_FLASH_2R_BK")) return true;
  const int p64 = (n + 63) / 64 * 64, p48 = (n + 47) / 48 * 48;
  return p64 <= 1.025 * p48;
}
// (32: the biased form's 32-key tiles, the bias-free form's 48 - Colab T4 --bench-grid, against 48 for both: biased
// 0.79 / 0.83 / 0.93 the time at 261 / 512 / 1,024 tokens, bias-free level. Four warps of one query tile each at 32
// keys measured the same; 16 keys, and two register-staged stages (one barrier a tile), won at one length and lost at
// another)
inline int flash2R1Tile() {
  static const int v = getenv("LOCALFOLD_FLASH_2R1") ? atoi(getenv("LOCALFOLD_FLASH_2R1")) : 32;
  return v;
}
// a T4's (register-staged) unmasked 32-wide grid attention through flashGrid2R's one-stage form - two query tiles a
// warp at a register-staged kernel's footprint (LOCALFOLD_FLASH_2R1: 0 off, 32, 48 or 64 the key tile); false: not taken
// will flash2R1Strided take this device's unmasked 32-wide attention (so a caller may pass no bias at all)
inline bool flash2R1Takes() {
  int bk = flash2R1Tile();
  return (bk == 64 && fitsSmem(fa2Stage<32, 2, 64, 2, 2>())) || ((bk == 48 || bk == 32) && fitsSmem(fa2Stage<32, 2, 48, 2, 2>()));
}
template <bool NB = false>
bool flash2R1Strided(const half* qkvg, const half* bias, int stride, half* out, int n, int heads, size_t rows, float scale,
                     const float* qBias, size_t rowStride = 0, size_t posStride = 0, size_t outRowStride = 0,
                     size_t outPosStride = 0) {
  int bk = flash2R1Tile();
  if (bk == 32 && NB) bk = 48;
  if (bk == 32 && fitsSmem(fa2Stage<32, 2, 32, 2, 2>()))
    flashGrid2RRun<32, 2, 32, 2, 2, NB, true>(qkvg, bias, stride, out, n, heads, rows, scale, qBias, rowStride, posStride, outRowStride, outPosStride);
  else if (bk == 64 && fitsSmem(fa2Stage<32, 2, 64, 2, 2>()))
    flashGrid2RRun<32, 2, 64, 2, 2, NB, true>(qkvg, bias, stride, out, n, heads, rows, scale, qBias, rowStride, posStride, outRowStride, outPosStride);
  else if (bk == 48 && fitsSmem(fa2Stage<32, 2, 48, 2, 2>()))
    flashGrid2RRun<32, 2, 48, 2, 2, NB, true>(qkvg, bias, stride, out, n, heads, rows, scale, qBias, rowStride, posStride, outRowStride, outPosStride);
  else return false;
  return true;
}
template <int D, int WARPS, bool MASKED, int BK = FA_BK, bool REG = false,
          int MINB = (D == 48 && WARPS == 4 && BK == 48 && !REG) ? 4 : 1, bool F16S = false> void setFlashSmem() {
  static bool done = false;
  if (!done) {
    smemAttr((flashGridHalf<D, WARPS, MASKED, BK, REG, MINB, false, F16S>), (int)((REG ? 1 : 2) * faStage<D, WARPS, BK>()));
    done = true;
  }
}
// FP8 (e4m3) grid attention where the card has FP8 tensor instructions (fp8Tensor): the unmasked, dense 32-wide form
// flashGrid2R takes, its Q K^T and P V in m16n8k32 e4m3 MMAs (half the f16 form's MMAs). Q is rounded to e4m3 as it is
// loaded (scale, log2(e) and a query bias folded in first), K and V by the projection (gridInK's kv8) into K8 and V8
// [row][pos][W], V then by fp8VtK into Vt8 [row][W][NK] - transposed, since there is no 8-bit ldmatrix.trans, each 32-key group in vPerm's order so a
// thread's P accumulators are already an m16n8k32 A fragment, the keys padded to 64 with zeros. P = 2^(s - m + 8), in
// [0, 256] (e4m3's range used), rounded to e4m3; l summed from the rounded P by an MMA against ones, so it normalises
// what P V summed. Emulated on the A100 first (q, k, v and P rounded through e4m3: 5CAJ 0.016 A from the f16 fold);
// On Colab's RTX PRO 6000 (--bench-grid, 256 rows): 249 / 301 TFLOP/s at 1,000 / 4,000 tokens against flashGrid2R's
// 207 / 230, output relRMS 3.5e-2 from the f32-score reference (2R's 3.2e-4). In the fold the attention is 262 -> 208 ms
// of a 988-token trunk (+18 of fp8VtK) and 6.43 -> 4.88 s at 2,964 (+0.17): the trunk 750 -> 707 ms and 11.18 -> 9.82
// s; 5CAJ self-templated 0.171 -> 0.170 A, 1BRS 0.485 both, 1TIM 1.007 -> 1.006 (chunked, LOCALFOLD_BIG=1, the same),
// a single-sequence 68-mer 0.054 A from the f16 fold. 🔴 Its block asks for two an SM (__launch_bounds__ ..., 2): Q
// rounded to e4m3 in the kernel took it to 97 registers, ONE 12-warp block an SM, and 30% slower than the same kernel
// reading a pre-rounded Q at 80. LOCALFOLD_FP8_ATTN=0 keeps flashGrid2R. (CUDA 12.8 on: the f16-accumulating e4m3 MMA)
#ifndef LF_FP8_ATTN
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
#define LF_FP8_ATTN 1
#else
#define LF_FP8_ATTN 0
#endif
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && LF_FP8_ATTN
#define LF_FP8_DEVICE 1
#else
#define LF_FP8_DEVICE 0
#endif
__host__ __device__ inline int vPerm(int p) {     // the physical key at logical position p of a 32-key group
  int hi = p >= 16 ? 16 : 0, r = p - hi, t = r / 4, b = r % 4;
  return hi + (b < 2 ? 2 * t + b : 8 + 2 * t + b - 2);
}
#if LF_FP8_DEVICE
__device__ __forceinline__ void mma8h(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
    : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma8f(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t e4m3x2(uint32_t h) {    // two f16 -> two e4m3 in the low 16 bits (lo, hi)
  unsigned short d; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(h)); return d;
}
#endif
// Vt8 [row][W][NK] from V8 [row][pos][W] (both e4m3): a block a 64-key group of one row, the group staged transposed
// (rows of 65 bytes: a lane's sixteen dims land in eight banks) in vPerm's order, padding zero
__global__ void __launch_bounds__(256) fp8VtK(const uint8_t* __restrict__ V8, uint8_t* __restrict__ Vt8, int n, int NK, int W) {
  constexpr int LD = 65;
  extern __shared__ __align__(16) unsigned char vs[];             // [W][LD]
  const size_t r = blockIdx.y; const int k0 = blockIdx.x * 64, W16 = W / 16;
  for (int u = threadIdx.x; u < 64 * W16; u += 256) {
    int lk = u / W16, c = (u % W16) * 16, pos = k0 + lk;
    uint4 v = pos < n ? *reinterpret_cast<const uint4*>(V8 + (r * n + pos) * (size_t)W + c) : make_uint4(0, 0, 0, 0);
    int g = lk & 31, p = (lk & 32) + (g & 16) + ((g & 15) < 8 ? ((g & 7) >> 1) * 4 + (g & 1) : ((g & 7) >> 1) * 4 + 2 + (g & 1));
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
    for (int i = 0; i < 16; ++i) vs[(c + i) * LD + p] = b[i];
  }
  __syncthreads();
  for (int u = threadIdx.x; u < W * 4; u += 256) {
    int hd = u >> 2, p0 = (u & 3) * 16;
    uint32_t w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t x = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) x |= (uint32_t)vs[hd * LD + p0 + q * 4 + b] << (8 * b);
      w[q] = x;
    }
    *reinterpret_cast<uint4*>(Vt8 + (r * W + hd) * (size_t)NK + k0 + p0) = make_uint4(w[0], w[1], w[2], w[3]);
  }
}
template <int WARPS, int BK, int MT, int RR, bool SPEC = false>
__global__ void __launch_bounds__(WARPS * RR * 32, 2) flash8(const half* __restrict__ qkvg, const uint8_t* __restrict__ K8,
    const uint8_t* __restrict__ Vt8, const half* __restrict__ bias, int biasStride, half* __restrict__ out, int n, int NK,
    int heads, size_t rows, float scale, const float* __restrict__ qBias, int sw) {
#if LF_FP8_DEVICE
  constexpr int D = 32, BQ = 16 * MT * WARPS, NT = WARPS * RR * 32, NTR = WARPS * 32, LDB = BK + 8;
  constexpr int KROW = D, VROW = BK;                                  // bytes: a key's row, a dim's row of the tile
  constexpr int KT = BK * KROW, VT = D * VROW;                        // a tile's K and V bytes, per grid row
  constexpr size_t STAGE = (size_t)RR * (KT + VT) + (size_t)BQ * LDB * 2;
  extern __shared__ __align__(16) unsigned char smem[];
  const int rg = (int)threadIdx.x / NTR, tr = (int)threadIdx.x % NTR;
  const int warp = tr >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const size_t perHead = (rows + RR - 1) / RR;
  // sw > 0: flashGrid2R's block order - one head's blocks walk sw row groups across each query tile, so the resident
  // blocks share a bias tile once a head's bias is past the L2 (205 MB a head at 10,127 tokens, the 6000's L2 128):
  // that fold's trunk pass 350 -> 296 s, byte-identical (--bench-grid at 8,000 tokens: sw 4 309 TFLOP/s against 0's 301)
  unsigned b = blockIdx.y, qt = blockIdx.x;                          // (32-bit: a block count fits, and 64-bit spilled)
  if (sw > 0) {
    const unsigned nq = gridDim.x, ph = (unsigned)perHead, L = blockIdx.y * nq + blockIdx.x, phb = ph * nq;
    const unsigned r = L % phb, group = r / (sw * nq), within = r % (sw * nq), left = ph - group * sw;
    const unsigned gsz = left < (unsigned)sw ? left : (unsigned)sw;
    b = (L / phb) * ph + group * sw + within % gsz; qt = within / gsz;
  }
  const int h = (int)(b / (unsigned)perHead); size_t rl = (size_t)(b % (unsigned)perHead) * RR + rg;
  const bool live = rl < rows; if (!live) rl = rows - 1;
  const int W = heads * D;
  const int q0 = (int)qt * BQ;
  auto Kst = [&](int s) { return smem + s * STAGE + (size_t)rg * (KT + VT); };
  auto Vst = [&](int s) { return Kst(s) + KT; };
  auto Bst = [&](int s) { return (half*)(smem + s * STAGE + (size_t)RR * (KT + VT)); };
  // K rows 32 bytes: chunk c (0/1) of row r at c ^ ((r >> 2) & 1); V rows BK bytes: chunk c at c ^ ((r >> 1) & (BK/16 - 1))
  auto kOff = [](int r, int c) { return r * KROW + ((c ^ ((r >> 2) & 1)) << 4); };
  auto vOff = [](int r, int c) { return r * VROW + ((c ^ ((r >> 1) & (BK / 16 - 1))) << 4); };
  const uint8_t* kRow = K8 + (rl * n) * (size_t)W + h * D;              // + key * W
  const uint8_t* vRow = Vt8 + ((rl * heads + h) * (size_t)D) * NK;      // + dim * NK + key
  const half* base = qkvg + (rl * n) * (size_t)(4 * W) + h * D;         // + pos * 4W: q, then (+3W) the gate
  const half* biasHead = bias + (size_t)h * n * biasStride;
  auto issue = [&](int j0, int st) {
    uint8_t* Ks = Kst(st); uint8_t* Vs = Vst(st); half* Bs = Bst(st);
    for (int u = tr; u < BK * 2; u += NTR) {                          // K: BK rows x 2 chunks
      int r = u >> 1, c = u & 1, key = j0 + r;
      cpAsync16(Ks + kOff(r, c), kRow + (size_t)(key < n ? key : 0) * W + c * 16, key < n);
    }
    for (int u = tr; u < D * (BK / 16); u += NTR) {                   // V: D rows x BK/16 chunks (padding is zero)
      int r = u / (BK / 16), c = u % (BK / 16);
      cpAsync16(Vs + vOff(r, c), vRow + (size_t)r * NK + j0 + c * 16, true);
    }
    for (int u = threadIdx.x; u < BQ * (BK / 8); u += NT) {           // bias: BQ rows x BK/8 chunks
      int qi = u / (BK / 8), c = (u % (BK / 8)) * 8, i = q0 + qi;
      bool ok = i < n && j0 + c < n;
      cpAsync16(Bs + qi * LDB + c, ok ? biasHead + (size_t)i * biasStride + j0 + c : bias, ok);
    }
    cpCommit();
  };
  // Q fragments: a0 row g bytes 4tig.., a1 row g+8, a2 row g bytes 16+4tig.., a3 row g+8
  const float qs = scale * LOG2E;
  auto q4 = [&](int i, int e) -> uint32_t {
    if (i >= n) return 0u;
    const half* p = base + (size_t)i * (4 * W) + e;
    float2 a = __half22float2(*reinterpret_cast<const half2*>(p)), c = __half22float2(*reinterpret_cast<const half2*>(p + 2));
    if (qBias) { a.x += qBias[h * D + e]; a.y += qBias[h * D + e + 1]; c.x += qBias[h * D + e + 2]; c.y += qBias[h * D + e + 3]; }
    half2 lo = __floats2half2_rn(a.x * qs, a.y * qs), hi = __floats2half2_rn(c.x * qs, c.y * qs);
    return e4m3x2(*reinterpret_cast<uint32_t*>(&lo)) | e4m3x2(*reinterpret_cast<uint32_t*>(&hi)) << 16;
  };
  uint32_t qa[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    int i0 = q0 + warp * 16 * MT + mt * 16 + g, i1 = i0 + 8;
    qa[mt][0] = q4(i0, 4 * tig); qa[mt][1] = q4(i1, 4 * tig); qa[mt][2] = q4(i0, 16 + 4 * tig); qa[mt][3] = q4(i1, 16 + 4 * tig);
  }
  float o[MT][D / 8][4] = {}, lsum[MT][4] = {};
  float mx[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) mx[mt][0] = mx[mt][1] = -INFINITY;
  const int tiles = (n + BK - 1) / BK;
  issue(0, 0);
  for (int tile = 0; tile < tiles; ++tile) {
    int st = tile & 1;
    if (tile + 1 < tiles) { issue((tile + 1) * BK, st ^ 1); cpWait<1>(); } else cpWait<0>();
    __syncthreads();
    const uint8_t* Ks = Kst(st); const uint8_t* Vs = Vst(st); const half* Bs = Bst(st);
    uint32_t sh[MT][BK / 8][2];                                        // the scores, starting from the bias
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const half* rowp = Bs + (warp * 16 * MT + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LDB + (lane >> 4) * 8;
#pragma unroll
      for (int nt = 0; nt < BK / 8; nt += 2) {
        uint32_t r4[4]; ldsm4(r4, rowp + nt * 8);
        sh[mt][nt][0] = r4[0]; sh[mt][nt][1] = r4[1]; sh[mt][nt + 1][0] = r4[2]; sh[mt][nt + 1][1] = r4[3];
      }
      if (tile == tiles - 1)
#pragma unroll
        for (int nt = 0; nt < BK / 8; ++nt) {
          int jg = tile * BK + nt * 8 + tig * 2;
          if (jg >= n) { sh[mt][nt][0] = (sh[mt][nt][0] & 0xFFFF0000u) | 0xFC00u; sh[mt][nt][1] = (sh[mt][nt][1] & 0xFFFF0000u) | 0xFC00u; }
          if (jg + 1 >= n) { sh[mt][nt][0] = (sh[mt][nt][0] & 0xFFFFu) | 0xFC000000u; sh[mt][nt][1] = (sh[mt][nt][1] & 0xFFFFu) | 0xFC000000u; }
        }
    }
    // S += Q K^T: two 8-key tiles an ldmatrix (keys x 16-byte halves of their 32 dims)
#pragma unroll
    for (int nt = 0; nt < BK / 8; nt += 2) {
      uint32_t kb[4];
      int qm = lane >> 3, key = nt * 8 + (qm >> 1) * 8 + (lane & 7);
      ldsm4(kb, Ks + kOff(key, qm & 1));
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) { mma8h(sh[mt][nt], qa[mt], kb[0], kb[1]); mma8h(sh[mt][nt + 1], qa[mt], kb[2], kb[3]); }
    }
    // the running max (rounded to f16, as the f16 kernel's), the rescale; P = 2^(s - (m - 8))
    // SPEC (flashGrid2R's speculative softmax): a tile whose rows stay within 0.5 of the warp's running maximum keeps it -
    // P then at most 2^8.5 = 362, inside e4m3's 448 (where the f16 kernel's P has room for 2^8 of slack, e4m3's has this)
    __half2 hn[MT][2];
    bool full = true;
    if constexpr (SPEC) {
      bool over = false;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        __half2 x0 = *reinterpret_cast<__half2*>(&sh[mt][0][0]), x1 = *reinterpret_cast<__half2*>(&sh[mt][0][1]);
#pragma unroll
        for (int nt = 1; nt < BK / 8; ++nt) { x0 = __hmax2(x0, *reinterpret_cast<__half2*>(&sh[mt][nt][0])); x1 = __hmax2(x1, *reinterpret_cast<__half2*>(&sh[mt][nt][1])); }
        over |= fmaxf(__low2float(x0), __high2float(x0)) > mx[mt][0] + 0.5f || fmaxf(__low2float(x1), __high2float(x1)) > mx[mt][1] + 0.5f;
      }
      full = __any_sync(~0u, over);
      if (!full) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) { hn[mt][0] = __float2half2_rn(mx[mt][0] - 8.f); hn[mt][1] = __float2half2_rn(mx[mt][1] - 8.f); }
      }
    }
    if (full) {
  #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        __half2 x0 = *reinterpret_cast<__half2*>(&sh[mt][0][0]), x1 = *reinterpret_cast<__half2*>(&sh[mt][0][1]);
  #pragma unroll
        for (int nt = 1; nt < BK / 8; ++nt) { x0 = __hmax2(x0, *reinterpret_cast<__half2*>(&sh[mt][nt][0])); x1 = __hmax2(x1, *reinterpret_cast<__half2*>(&sh[mt][nt][1])); }
        float t0 = fmaxf(__low2float(x0), __high2float(x0)), t1 = fmaxf(__low2float(x1), __high2float(x1));
        t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
        float n0 = __half2float(__float2half(fmaxf(mx[mt][0], t0))), n1 = __half2float(__float2half(fmaxf(mx[mt][1], t1)));
        float c0 = exp2f(mx[mt][0] - n0), c1 = exp2f(mx[mt][1] - n1);
        mx[mt][0] = n0; mx[mt][1] = n1;
  #pragma unroll
        for (int et = 0; et < D / 8; ++et) { o[mt][et][0] *= c0; o[mt][et][1] *= c0; o[mt][et][2] *= c1; o[mt][et][3] *= c1; }
        lsum[mt][0] *= c0; lsum[mt][1] *= c0; lsum[mt][2] *= c1; lsum[mt][3] *= c1;
        hn[mt][0] = __float2half2_rn(n0 - 8.f); hn[mt][1] = __float2half2_rn(n1 - 8.f);
      }
    }
    // O += P V, 32 keys an MMA
#pragma unroll
    for (int t = 0; t < BK / 32; ++t) {
      uint32_t pa[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        auto e = [&](int nt, int hr) {
          __half2 d = __hsub2(*reinterpret_cast<__half2*>(&sh[mt][nt][hr]), hn[mt][hr]);
          return e4m3x2(ex2h2(*reinterpret_cast<uint32_t*>(&d)));
        };
        pa[mt][0] = e(4 * t, 0) | (e(4 * t + 1, 0) << 16);
        pa[mt][1] = e(4 * t, 1) | (e(4 * t + 1, 1) << 16);
        pa[mt][2] = e(4 * t + 2, 0) | (e(4 * t + 3, 0) << 16);
        pa[mt][3] = e(4 * t + 2, 1) | (e(4 * t + 3, 1) << 16);
        mma8f(lsum[mt], pa[mt], 0x38383838u, 0x38383838u);           // (e4m3 1.0 = 0x38)
      }
#pragma unroll
      for (int et = 0; et < D / 8; et += 2) {
        uint32_t vb[4];
        int qm = lane >> 3, dim = et * 8 + (qm >> 1) * 8 + (lane & 7);
        ldsm4(vb, Vs + vOff(dim, t * 2 + (qm & 1)));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) { mma8f(o[mt][et], pa[mt], vb[0], vb[1]); mma8f(o[mt][et + 1], pa[mt], vb[2], vb[3]); }
      }
    }
    __syncthreads();
  }
  // out = O / l * sigmoid(gate)
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    int i0 = q0 + warp * 16 * MT + mt * 16 + g, i1 = i0 + 8;
    float l0 = lsum[mt][0], l1 = lsum[mt][2];
#pragma unroll
    for (int et = 0; et < D / 8; ++et) {
      int e = et * 8 + tig * 2;
      if (live && i0 < n) {
        float2 a = __half22float2(*reinterpret_cast<const half2*>(base + (size_t)i0 * (4 * W) + 3 * W + e));
        *reinterpret_cast<half2*>(out + (rl * n + i0) * (size_t)W + h * D + e) =
          __floats2half2_rn(o[mt][et][0] / l0 / (1.f + __expf(-a.x)), o[mt][et][1] / l0 / (1.f + __expf(-a.y)));
      }
      if (live && i1 < n) {
        float2 a = __half22float2(*reinterpret_cast<const half2*>(base + (size_t)i1 * (4 * W) + 3 * W + e));
        *reinterpret_cast<half2*>(out + (rl * n + i1) * (size_t)W + h * D + e) =
          __floats2half2_rn(o[mt][et][2] / l1 / (1.f + __expf(-a.x)), o[mt][et][3] / l1 / (1.f + __expf(-a.y)));
      }
    }
  }
#endif
}
// will flash8Run take this attention: an FP8 card, a binary whose flash8 was compiled for one (not an older
// architecture's PTX run by the driver, where the kernel is empty), and LOCALFOLD_FP8_ATTN not 0
inline bool fp8Attn() {
  static const bool on = [] {
    if (!LF_FP8_ATTN || !fp8Tensor()) return false;
    if (getenv("LOCALFOLD_FP8_ATTN") && !atoi(getenv("LOCALFOLD_FP8_ATTN"))) return false;
    cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, flash8<4, 64, 1, 3>));
    return a.ptxVersion >= 89;
  }();
  return on;
}
inline int FP8_SWIZZLE = getenv("LOCALFOLD_FP8_SWIZZLE") ? atoi(getenv("LOCALFOLD_FP8_SWIZZLE")) : -1;
// flash8's speculative softmax (flashGrid2R's SPEC) from 1,024 tokens: on a Colab L4 (--bench-grid, 256 rows) 20% slower
// at 512 (3.47 against 2.88 ms), 2.6% faster at 1,024 (20.70 / 21.26) and 3.2% at 2,048 (126.2 / 130.5); 6MRR, 5CAJ
// and 1TIM self-templated 0.014 A from the folds without it, every crystal RMSD within 0.006 A of its own.
// spec: 0 or 1 takes it off or on (--bench-grid's arms), -1 by the length
// the attention from qkvg's q and gate (f16, [row][pos][4W]) and K8, V8 (e4m3, [row][pos][W] - gridInK's kv8)
// (its geometry swept on Colab's RTX PRO 6000, --bench-grid at 1,000 / 2,000 tokens: 4 warps, one query tile, 3 rows
// 2.04 / 14.65 ms; 2 warps two tiles 3 rows 2.24 / 14.20, 4 warps 2 rows 2.18 / 15.74, 8 warps 2.33 / 19.44, 6 rows
// 7.62 / 67.6 - every geometry byte-identical. Kept)
inline void flash8Run(const half* qkvg, const uint8_t* K8, const uint8_t* V8, const half* bias, int stride, half* out, int n,
                      int heads, size_t rows, float scale, const float* qBias, int spec = -1, bool vtDone = false) {
  // vtDone: V8 is flash8's transposed V already (gridInK's vt8), and fp8VtK is skipped
  constexpr int WARPS = 4, BK = 64, MT = 1, RR = 3, BQ = 16 * MT * WARPS;
  const int W = heads * 32, NK = (n + 63) / 64 * 64;
  const uint8_t* Vt8 = V8;
  if (!vtDone) {
    uint8_t* vt = scratch<uint8_t>("attn.vt8", rows * (size_t)W * NK);
    static int vtAttr = 0;
    if (65 * W > vtAttr) { smemAttr(fp8VtK, 65 * W); vtAttr = 65 * W; }
    fp8VtK<<<dim3(NK / 64, (unsigned)rows), 256, 65 * W, STREAM>>>(V8, vt, n, NK, W);
    Vt8 = vt;
  }
  const int bytes = 2 * (RR * (BK * 32 + 32 * BK) + BQ * (BK + 8) * 2);
  static bool attr = false;
  dim3 grid((n + BQ - 1) / BQ, (unsigned)((rows + RR - 1) / RR * heads));
  const int sw = FP8_SWIZZLE >= 0 ? FP8_SWIZZLE : n >= 3500 ? 4 : 0;
  if (spec > 0 || (spec < 0 && n >= 1024)) {
    static bool attrS = false;
    if (!attrS) { smemAttr((flash8<WARPS, BK, MT, RR, true>), bytes); attrS = true; }
    flash8<WARPS, BK, MT, RR, true><<<grid, 32 * WARPS * RR, bytes, STREAM>>>(qkvg, K8, Vt8, bias, stride, out, n, NK, heads, rows, scale, qBias, sw);
    return;
  }
  if (!attr) { smemAttr((flash8<WARPS, BK, MT, RR>), bytes); attr = true; }
  flash8<WARPS, BK, MT, RR><<<grid, 32 * WARPS * RR, bytes, STREAM>>>(qkvg, K8, Vt8, bias, stride, out, n, NK, heads, rows, scale, qBias, sw);
}
inline int FLASH_WARPS_OVERRIDE = 0;
// F16S (flashGridHalf): the unmasked kernel's scores in f16 - LOCALFOLD_FLASH_F16S=1 (being measured)
inline bool FLASH_F16S = getenv("LOCALFOLD_FLASH_F16S") && atoi(getenv("LOCALFOLD_FLASH_F16S"));
inline bool FLASH_SPLIT = true;
// the register-staged form (see flashGridHalf's REG) where the device has no cp.async - before Ampere, a
// T4 - unless LOCALFOLD_FLASH_REG says otherwise (0 or 1: to measure either form on any device)
inline bool flashRegStaged() {
  static int v = [] {
    if (const char* e = getenv("LOCALFOLD_FLASH_REG")) return atoi(e);
    return ccMajor() < 8 ? 1 : 0;
  }();
  return v != 0;
}
// several GPUs on a sharded pair (the diffusion transformer's token attention): this rank's queries [FLASH_QLO,
// FLASH_QHI), its bias planes FLASH_BIAS_ROWS rows from query FLASH_QLO; 0 - every query, the whole bias
inline int FLASH_QLO = 0, FLASH_QHI = 0, FLASH_BIAS_ROWS = 0;
template <int D, int WARPS, int BK = FA_BK, bool REG = false,
          int MINB = (D == 48 && WARPS == 4 && BK == 48 && !REG) ? 4 : 1>
void flashGridHalfRun(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                      int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias) {
  const int qn = FLASH_QHI ? FLASH_QHI - FLASH_QLO : n;
  dim3 grid((qn + 16 * WARPS - 1) / (16 * WARPS), (unsigned)(rows * heads));
  if (FLASH_QHI && (mask || FLASH_2R && D == 32 && WARPS == 4 && BK == 48)) {
    fprintf(stderr, "flashGridHalfRun: a query range takes the unmasked 16/48/64-wide kernel only\n"); exit(1);
  }
  const int bytes = (REG ? 1 : 2) * faStage<D, WARPS, BK>();
  if (mask) {                    // a null mask: every key real (see MASKED)
    setFlashSmem<D, WARPS, true, BK, REG, MINB>();
    flashGridHalf<D, WARPS, true, BK, REG, MINB><<<grid, 32 * WARPS, bytes, STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
    return;
  }
  if constexpr (!REG && D == 32 && WARPS == 4 && BK == 48) {   // (the unmasked kernel reads no r0)
    if (FLASH_2R) {
      if (flash2RWideKeysFor(n) && FLASH_SPEC) flashGrid2RRun<D, 2, 64, 2, 2, false, false, half, true>(qkvg, bias, stride, out, n, heads, rows, scale, qBias);
      else if (flash2RWideKeysFor(n)) flashGrid2RRun<D, 2, 64, 2, 2>(qkvg, bias, stride, out, n, heads, rows, scale, qBias);
      else if (FLASH_SPEC) flashGrid2RRun<D, 2, BK, 2, 2, false, false, half, true>(qkvg, bias, stride, out, n, heads, rows, scale, qBias);
      else flashGrid2RRun<D, 2, BK, 2, 2>(qkvg, bias, stride, out, n, heads, rows, scale, qBias);
      return;
    }
  }
  if (FLASH_F16S) {
    setFlashSmem<D, WARPS, false, BK, REG, MINB, true>();
    flashGridHalf<D, WARPS, false, BK, REG, MINB, false, true><<<grid, 32 * WARPS, bytes, STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias, 0, 0, 0, 0, FLASH_QLO, FLASH_QHI, FLASH_BIAS_ROWS);
  } else {
    setFlashSmem<D, WARPS, false, BK, REG, MINB>();
    flashGridHalf<D, WARPS, false, BK, REG, MINB><<<grid, 32 * WARPS, bytes, STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias, 0, 0, 0, 0, FLASH_QLO, FLASH_QHI, FLASH_BIAS_ROWS);
  }
}
template <int D, int WARPS, int BK = FA_BK>
void flashGridHalfAt(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                     int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias) {
  if (flashRegStaged()) flashGridHalfRun<D, WARPS, BK, true>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
  else flashGridHalfRun<D, WARPS, BK, false>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
}
template <int D>
void flashGridHalfLaunch(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                         int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias = nullptr) {
  // 4 warps (64 queries) a block across both the token transformer and the pair track's grid attention:
  // at D = 48, BK = 48 with __launch_bounds__(128, 4), 4 warps beat 8 warps at every size measured
  // (256 tokens: 10.6 vs 13.5 us; 384: 13.4 vs 16.9 us; 256x5: 19.5 vs 26.6 us; 1024: 55.4 vs 78.4 us).
  int warps = 4;
  if (FLASH_WARPS_OVERRIDE) warps = FLASH_WARPS_OVERRIDE;
  // a query range (several GPUs, the diffusion's tokens split: a rank's few hundred queries over every key) in blocks of
  // 4 warps at most - a simulated rank of 8's 25 steps 89.6 -> 84.1 ms at 988 tokens, 161 -> 149 at 2,964 (2 warps
  // 82.8 / 161, 1 warp 85.9 / 175: narrower blocks each walk every key again)
  if (FLASH_QHI && !FLASH_WARPS_OVERRIDE) warps = std::min(warps, 4);
  // too few blocks to fill the device: split each 16 queries' keys over four warps instead.
  // Measured at 16 heads, D 48: 9.3 against 14.7 us at 68 tokens, 13.4/19.5 at 192, and worse
  // from 256 (22.8/21.5), where the merge outweighs the parallelism.
  // ...and ONE row only: with the samples as rows (--samples=5) the grid kernel has the blocks it
  // lacked, and the split loses at every size measured - 11.1 against 8.2 us at 68 tokens and five
  // rows, 27.0 against 13.4 at 150, 15.7 against 8.8 at 192 and two rows (a tie at 68 and two)
  if (FLASH_QHI && D == 32 && flashRegStaged()) {        // (flash2R1Strided below computes every query)
    fprintf(stderr, "flashGridHalfLaunch: a query range does not take the register-staged 32-wide form\n"); exit(1);
  }
  if (!FLASH_WARPS_OVERRIDE && FLASH_SPLIT && !FLASH_QHI && rows == 1 && n <= 192 && heads * ((n + 63) / 64) < 4 * 108 &&
      fitsSmem(4 * 2 * fsStage<D>() + 4 * 16 * 8)) {        // (the split's four warps' own buffers: 67 KB at D 48, past a T4's 64)
    flashSplitHalfAt<D, 4>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    return;
  }
  // the denoiser's 48-wide heads in 48-key tiles: with 64 a 4-warp block takes 47.6 KB of shared
  // memory and three fit an SM where the registers allow four - 23.1 against 32.8 us at 261 tokens
  // and five samples, 44.0 against 45.9 at 400, level at 150 and one sample
  // ...up to ~1700 tokens, where the longer key loop's per-tile cost overtakes the occupancy (1536: 106.7
  // against 120.7 us; 1800: 162.3 against 154.0; 2088: 194 against 180)
  constexpr int BK = D == 48 ? 48 : FA_BK;
  // no more warps than the device's shared memory allows (a T4's 64 KB)
  if (warps == 8 && !fitsSmem(2 * faStage<D, 8, BK>())) warps = 4;
  if (warps == 4 && !fitsSmem(2 * faStage<D, 4, BK>())) warps = 2;
  // (64-key tiles: their own shared memory checked - 66 KB double-buffered at 4 warps, past a 64 KB device)
  const int stages = flashRegStaged() ? 1 : 2;
  if constexpr (D == 48) {
    if (n >= 1700 && (warps == 4 || warps == 8) &&
        fitsSmem(stages * (warps == 8 ? faStage<D, 8, FA_BK>() : faStage<D, 4, FA_BK>()))) {
      if (warps == 8) flashGridHalfAt<D, 8, FA_BK>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
      else flashGridHalfAt<D, 4, FA_BK>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
      return;
    }
  }
  switch (warps) {
    case 8:
      if constexpr (D == 48)
        flashGridHalfAt<D, 8, BK>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
      break;
    case 4:
      // the pair track's 32-wide heads in 48-key tiles where cp.async double-buffers them: smaller stages, more
      // blocks an SM - --bench-grid on an A100 1.036 against 1.211 ms at 524 tokens, 7.23 against 7.80 at 1044,
      // level at 262; an L4 (99 KB a block) 0.566 against 0.747 at 262. A T4's register-staged form keeps 64
      // (48 is 1.45 against 1.22 ms there)
      // the template stack's 16-wide heads, unmasked (a pair track's rows): flashGrid2R's two tiles a warp and f16 scores,
      // as the 32-wide ones - --bench-grid16 0.737 -> 0.549 ms at 500 tokens, 41.8 -> 30.4 at 2,000, 61.5 -> 35.1 at
      // 6,000 (256 rows); output relRMS 3.7e-4 from the f32-score kernel's (the 32-wide form's was 3.9e-4)
      if constexpr (D == 16) {
        if (!flashRegStaged() && !mask && rows >= 32) {   // (unmasked, r0 and tr unread)
          flashGrid2RRun<16, 2, 48, 2, 2>(qkvg, bias, stride, out, n, heads, rows, scale, qBias); break;
        }
      }

      if constexpr (D == 32) {
        if (!flashRegStaged()) { flashGridHalfAt<D, 4, 48>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias); break; }
        if (!mask && flash2R1Strided(qkvg, bias, stride, out, n, heads, rows, scale, qBias)) break;
      }
      if constexpr (D == 64) {
        if (!flashRegStaged() && !mask && !FLASH_QHI && FLASH_2R && rows >= 16 && fitsSmem(2 * fa2Stage<64, 4, 48, 1, 1>())) {
          flashGrid2RRun<64, 4, 48, 1, 1>(qkvg, bias, stride, out, n, heads, rows, scale, qBias); break;
        }
      }
      flashGridHalfAt<D, 4, BK>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias); break;
    case 2:
      if constexpr (D == 48)
        flashGridHalfAt<D, 2>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
      break;
    default:
      if constexpr (D == 48)
        flashGridHalfAt<D, 1>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
      break;
  }
}
// an 8-wide head unpadded (AF2's extra stack): flashGrid2R's D 8 form, where it takes the call - unmasked, cp.async
// (anywhere else the heads are padded to 16)
inline bool flashNarrow(const float* mask) { return !mask && !flashRegStaged(); }
template <class T>
void flashGrid(const T* qkvg, const T* bias, int stride, const float* mask, T* out, int n, int heads,
               int D, size_t r0, size_t rows, bool tr, float scale, const float* qBias = nullptr) {
  if constexpr (std::is_same_v<T, half>) {
    if (D == 32) flashGridHalfLaunch<32>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    else if (D == 48) flashGridHalfLaunch<48>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    else if (D == 16) flashGridHalfLaunch<16>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    else if (D == 8 && flashNarrow(mask)) {
      // (the 32-wide form's geometry and its speculative softmax: at 1,000 residues and 5,120 rows 704 ms padded to 16,
      // 577 unpadded, 532 with SPEC; 32- 96- and 128-key tiles, four warps a block or four rows were level or slower)
      if (FLASH_SPEC) flashGrid2RRun<8, 2, 64, 2, 2, false, false, half, true>(qkvg, bias, stride, out, n, heads, rows, scale, qBias);
      else flashGrid2RRun<8, 2, 64, 2, 2>(qkvg, bias, stride, out, n, heads, rows, scale, qBias);
    }
    else if (D == 64) flashGridHalfLaunch<64>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    else { fprintf(stderr, "flashGrid: no f16 kernel for head width %d\n", D); exit(1); }
  } else {
    dim3 g((n + 63) / 64, (unsigned)(rows * heads));
    if (D == 32) flashGridF32<32><<<g, 64, 0, STREAM>>>(qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
    else if (D == 16) flashGridF32<16><<<g, 64, 0, STREAM>>>(qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
    else if (D == 24) flashGridF32<24><<<g, 64, 0, STREAM>>>(qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
    else if (D == 64) flashGridF32<64><<<g, 64, 0, STREAM>>>(qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
    else { fprintf(stderr, "flashGrid: no f32 kernel for head width %d\n", D); exit(1); }
  }
}
