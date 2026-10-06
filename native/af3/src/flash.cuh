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
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
// the same with an f16 accumulator (two half2 registers: row g, row g + 8) - F16S's scores
__device__ __forceinline__ void mma16816h(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(b0));
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a[2]), "r"(a[3]), "r"(b1));
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
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
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
  uint32_t y; asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x)); return y;
}
constexpr uint32_t ONES_H2 = 0x3C003C00u;      // half2(1, 1)
// REG: the next key tile staged in REGISTERS while this one is computed, then stored into the one shared
// buffer - for a device without cp.async (a T4), where the double buffer's "async" copies are synchronous
// loads that stall the warp before every tile, and its two stages (39 KB at D 32) let one block an SM fit
// in a T4's 64 KB. The same arithmetic in the same order: the output is identical.
// STRIDED: rows and positions at arbitrary strides (elements) - an attention ACROSS a tensor's leading axis
// (an MSA's columns, the triangle's ending node) read where the tensor lies rather than from a transposed
// copy; native/af2 runs it. Without it the addressing is the dense [rows][positions][4W] layout's, in the
// same 32-bit arithmetic as always (the strides would cost registers the dense form has none to spare).
// F16S (unmasked only): the scores accumulated in f16 - the bias tile IS the MMA's accumulator input, the row max
// and the subtraction packed half2 ops, the subtraction's result already the exponent's packed input - where
// the f32 scores spent ~140 scalar instructions a tile against 27 MMAs (bias conversions and adds, 24 subtracts,
// 12 pack conversions, the max): the kernel is issue-bound on them. 11 mantissa bits for a logit, where
// AlphaFold 3's own JAX runs these in bf16's 8; the running max is rounded to an f16 value so the exponents and
// the rescale agree.
template <int D, int WARPS, bool MASKED = true, int BK = FA_BK, bool REG = false, int MINB = 1, bool STRIDED = false,
          bool F16S = false>
__global__ void __launch_bounds__(WARPS * 32, MINB) flashGridHalf(const half* __restrict__ qkvg, const half* __restrict__ bias,
    int biasStride, const float* __restrict__ mask, half* __restrict__ out, int n, int heads, size_t r0, bool tr, float scale,
    const float* qBias, size_t rowStride = 0, size_t posStride = 0, size_t outRowStride = 0, size_t outPosStride = 0) {
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
  int q0 = blockIdx.x * BQ;
  // the loads, with compile-time trip counts and 32-bit offsets: written as a strided loop from
  // threadIdx.x the compiler could not unroll it, and the per-tile index arithmetic was ~800
  // integer instructions a thread against 36 tensor-core MMAs
  constexpr int KV_CHUNKS = BK * (D / 8), B_CHUNKS = BQ * (BK / 8);
  static_assert(B_CHUNKS % NT == 0, "a bias tile is a whole number of chunks a thread");
  const half* biasHead = bias + (size_t)h * n * biasStride;
  // per-thread source pointers, advanced by a constant each tile (the bounds matter only on the
  // last tile, which the `last` flag takes through the checked path)
  constexpr int KV_PER = (KV_CHUNKS + NT - 1) / NT, B_PER = B_CHUNKS / NT;
  const half* kvSrc[KV_PER]; int kvOff[KV_PER], kvJ[KV_PER];
#pragma unroll
  for (int k = 0; k < KV_PER; ++k) {
    int u = k * NT + threadIdx.x, jj = u / (D / 8), c = (u % (D / 8)) * 8;
    kvJ[k] = (KV_CHUNKS % NT == 0 || u < KV_CHUNKS) ? jj : 1 << 30;
    kvOff[k] = jj * LDK + c;
    kvSrc[k] = base + pos(jj) + Wd + c;
  }
  const half* bSrc[B_PER]; int bOff[B_PER], bC[B_PER]; bool bRow[B_PER];
#pragma unroll
  for (int k = 0; k < B_PER; ++k) {
    int u = k * NT + threadIdx.x, qi = u / (BK / 8), c = (u % (BK / 8)) * 8, i = q0 + qi;
    bRow[k] = i < n; bC[k] = c; bOff[k] = qi * LDB + c;
    bSrc[k] = biasHead + (i < n ? i : 0) * biasStride + c;
  }
  auto issue = [&](int j0, int st) {
    half *K = Kst(st), *V = Vst(st), *B = Bst(st);
    bool last = j0 + BK > n;
#pragma unroll
    for (int k = 0; k < KV_PER; ++k) {
      if (kvJ[k] >= (1 << 30)) continue;
      bool ok = !last || j0 + kvJ[k] < n;
      const half* src = ok ? kvSrc[k] + pos(j0) : base;
      cpAsync16(K + kvOff[k], src, ok);
      cpAsync16(V + kvOff[k], src + Wd, ok);
    }
#pragma unroll
    for (int k = 0; k < B_PER; ++k) {
      bool ok = bRow[k] && (!last || j0 + bC[k] < n);
      cpAsync16(B + bOff[k], ok ? bSrc[k] + j0 : biasHead, ok);
    }
    if (MASKED && threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Mst(st)[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f)
                                   : -INFINITY;
    }
    cpCommit();
  };
  // ...REG's halves of `issue`: the loads into registers, and the registers into the one stage
  uint4 kr[KV_PER][2], brg[B_PER]; float mr = 0.f;
  auto loadR = [&](int j0) {
    bool last = j0 + BK > n;
#pragma unroll
    for (int k = 0; k < KV_PER; ++k) {
      if (kvJ[k] >= (1 << 30)) continue;
      bool ok = !last || j0 + kvJ[k] < n;
      const half* src = kvSrc[k] + (ok ? pos(j0) : 0);
      kr[k][0] = ok ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
      kr[k][1] = ok ? *reinterpret_cast<const uint4*>(src + Wd) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < B_PER; ++k) {
      bool ok = bRow[k] && (!last || j0 + bC[k] < n);
      brg[k] = ok ? *reinterpret_cast<const uint4*>(bSrc[k] + j0) : make_uint4(0, 0, 0, 0);
    }
    if (MASKED && threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      mr = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f) : -INFINITY;
    }
  };
  auto storeR = [&]() {
    half *K = Kst(0), *V = Vst(0), *B = Bst(0);
#pragma unroll
    for (int k = 0; k < KV_PER; ++k) {
      if (kvJ[k] >= (1 << 30)) continue;
      *reinterpret_cast<uint4*>(K + kvOff[k]) = kr[k][0];
      *reinterpret_cast<uint4*>(V + kvOff[k]) = kr[k][1];
    }
#pragma unroll
    for (int k = 0; k < B_PER; ++k) *reinterpret_cast<uint4*>(B + bOff[k]) = brg[k];
    if (MASKED && threadIdx.x < BK) Mst(0)[threadIdx.x] = mr;
  };
  int i0 = q0 + warp * 16 + g, i1 = i0 + 8;
  auto q2 = [&](int i, int e) -> uint32_t {
    if (i >= n) return 0u;
    half2 v = *reinterpret_cast<const half2*>(base + (STRIDED ? pos(i) : (size_t)i * W4) + e);
    float2 f = __half22float2(v);
    if (qBias) { f.x += qBias[h * D + e]; f.y += qBias[h * D + e + 1]; }   // the query's bias, folded in here
    return pack2(f.x * scale * LOG2E, f.y * scale * LOG2E);
  };
  uint32_t qa[D / 16][4];
  for (int ks = 0; ks < D / 16; ++ks) {
    int e = ks * 16 + tig * 2;
    qa[ks][0] = q2(i0, e); qa[ks][1] = q2(i1, e); qa[ks][2] = q2(i0, e + 8); qa[ks][3] = q2(i1, e + 8);
  }
  float o[D / 8][4] = {};
  float m0 = -INFINITY, m1 = -INFINITY, lsum[4] = {};
  int tiles = (n + BK - 1) / BK;
  if constexpr (REG) { loadR(0); storeR(); __syncthreads(); if (tiles > 1) loadR(BK); }
  else issue(0, 0);
  for (int tile = 0; tile < tiles; ++tile) {
    int st = REG ? 0 : tile & 1;
    if constexpr (!REG) {
      if (tile + 1 < tiles) { issue((tile + 1) * BK, st ^ 1); cpWait<1>(); }
      else cpWait<0>();
      __syncthreads();
    }
    const half *K = Kst(st), *V = Vst(st), *B = Bst(st); const float* Ms = Mst(st);
    if constexpr (F16S && !MASKED) {
      const half* hb0 = B + (warp * 16 + g) * LDB;
      const half* hb1 = hb0 + 8 * LDB;
      uint32_t sh[BK / 8][2];
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
        int jj = nt * 8 + tig * 2;
        sh[nt][0] = *reinterpret_cast<const uint32_t*>(hb0 + jj);
        sh[nt][1] = *reinterpret_cast<const uint32_t*>(hb1 + jj);
        if (tile == tiles - 1) {                       // keys past n: -inf (f16 0xFC00), both rows
          int jg = tile * BK + jj;
          if (jg >= n) { sh[nt][0] = (sh[nt][0] & 0xFFFF0000u) | 0xFC00u; sh[nt][1] = (sh[nt][1] & 0xFFFF0000u) | 0xFC00u; }
          if (jg + 1 >= n) { sh[nt][0] = (sh[nt][0] & 0xFFFFu) | 0xFC000000u; sh[nt][1] = (sh[nt][1] & 0xFFFFu) | 0xFC000000u; }
        }
#pragma unroll
        for (int k2 = 0; k2 < D / 32 + (D % 32 ? 1 : 0); ++k2) {
          uint32_t kb[4];
          ldsm4(kb, K + (nt * 8 + (lane & 7)) * LDK + k2 * 32 + (lane >> 3) * 8);
          mma16816h(sh[nt], qa[k2 * 2], kb[0], kb[1]);
          if (k2 * 2 + 1 < D / 16) mma16816h(sh[nt], qa[k2 * 2 + 1], kb[2], kb[3]);
        }
      }
      __half2 x0 = asH2(sh[0][0]), x1 = asH2(sh[0][1]);
#pragma unroll
      for (int nt = 1; nt < BK / 8; ++nt) { x0 = __hmax2(x0, asH2(sh[nt][0])); x1 = __hmax2(x1, asH2(sh[nt][1])); }
      float t0 = fmaxf(__low2float(x0), __high2float(x0)), t1 = fmaxf(__low2float(x1), __high2float(x1));
      t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
      t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
      // (the new maxima rounded to f16 values: the exponents subtract exactly these, and the rescale uses them)
      float n0 = __half2float(__float2half(fmaxf(m0, t0))), n1 = __half2float(__float2half(fmaxf(m1, t1)));
      float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
      m0 = n0; m1 = n1;
#pragma unroll
      for (int et = 0; et < D / 8; ++et) { o[et][0] *= c0; o[et][1] *= c0; o[et][2] *= c1; o[et][3] *= c1; }
      lsum[0] *= c0; lsum[1] *= c0; lsum[2] *= c1; lsum[3] *= c1;
      __half2 h0 = __float2half2_rn(n0), h1 = __float2half2_rn(n1);
#pragma unroll
      for (int t = 0; t < BK / 16; ++t) {
        uint32_t pa[4] = { ex2h2(asU32(__hsub2(asH2(sh[2 * t][0]), h0))), ex2h2(asU32(__hsub2(asH2(sh[2 * t][1]), h1))),
                           ex2h2(asU32(__hsub2(asH2(sh[2 * t + 1][0]), h0))), ex2h2(asU32(__hsub2(asH2(sh[2 * t + 1][1]), h1))) };
        mma16816(lsum, pa, ONES_H2, ONES_H2);
#pragma unroll
        for (int et = 0; et < D / 8; et += 2) {
          uint32_t vb[4];
          ldsm4t(vb, V + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (et + (lane >> 4)) * 8);
          mma16816(o[et], pa, vb[0], vb[1]);
          mma16816(o[et + 1], pa, vb[2], vb[3]);
        }
      }
      __syncthreads();
      if constexpr (REG) {
        if (tile + 1 < tiles) { storeR(); __syncthreads(); if (tile + 2 < tiles) loadR((tile + 2) * BK); }
      }
      continue;
    }
    // S starts from the bias (and the key mask), and the tensor cores accumulate Q.K onto it
    const half* br0 = B + (warp * 16 + g) * LDB;
    const half* br1 = br0 + 8 * LDB;
    float sv[BK / 8][4];
    for (int nt = 0; nt < BK / 8; ++nt) {
      int jj = nt * 8 + tig * 2;
      float2 u = __half22float2(*reinterpret_cast<const half2*>(br0 + jj));
      float2 v = __half22float2(*reinterpret_cast<const half2*>(br1 + jj));
      if (MASKED) {
        float ma = Ms[jj], mb = Ms[jj + 1];
        u.x += ma; u.y += mb; v.x += ma; v.y += mb;
      } else if (tile == tiles - 1) {
        int jg = tile * BK + jj;
        if (jg >= n) { u.x = v.x = -INFINITY; }
        if (jg + 1 >= n) { u.y = v.y = -INFINITY; }
      }
      sv[nt][0] = u.x; sv[nt][1] = u.y; sv[nt][2] = v.x; sv[nt][3] = v.y;
      for (int k2 = 0; k2 < D / 32 + (D % 32 ? 1 : 0); ++k2) {
        uint32_t kb[4];
        ldsm4(kb, K + (nt * 8 + (lane & 7)) * LDK + k2 * 32 + (lane >> 3) * 8);
        mma16816(sv[nt], qa[k2 * 2], kb[0], kb[1]);
        if (k2 * 2 + 1 < D / 16) mma16816(sv[nt], qa[k2 * 2 + 1], kb[2], kb[3]);
      }
    }
    // the row maxima as a tree, not a 16-deep chain of dependent max instructions
    float r0m[BK / 8], r1m[BK / 8];
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) { r0m[nt] = fmaxf(sv[nt][0], sv[nt][1]); r1m[nt] = fmaxf(sv[nt][2], sv[nt][3]); }
    // (folding the ends together, so a tile of 6 n8 columns reduces as well as one of 8)
#pragma unroll
    for (int cnt = BK / 8; cnt > 1; cnt = (cnt + 1) / 2)
#pragma unroll
      for (int nt = 0; nt < cnt / 2; ++nt) { r0m[nt] = fmaxf(r0m[nt], r0m[cnt - 1 - nt]); r1m[nt] = fmaxf(r1m[nt], r1m[cnt - 1 - nt]); }
    float t0 = r0m[0], t1 = r1m[0];
    t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
    t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
    float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);
    float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
    m0 = n0; m1 = n1;
    for (int et = 0; et < D / 8; ++et) { o[et][0] *= c0; o[et][1] *= c0; o[et][2] *= c1; o[et][3] *= c1; }
    lsum[0] *= c0; lsum[1] *= c0; lsum[2] *= c1; lsum[3] *= c1;
    // P = 2^(S - max), two at a time in f16 (ex2.approx.f16x2: half the SFU work, and the result
    // is already the packed A fragment the PV product reads); its row sums on the tensor cores
    // (P . ones), accumulated in f32 like O
    for (int t = 0; t < BK / 16; ++t) {
      uint32_t pa[4] = { ex2h2(pack2(sv[2 * t][0] - n0, sv[2 * t][1] - n0)), ex2h2(pack2(sv[2 * t][2] - n1, sv[2 * t][3] - n1)),
                         ex2h2(pack2(sv[2 * t + 1][0] - n0, sv[2 * t + 1][1] - n0)),
                         ex2h2(pack2(sv[2 * t + 1][2] - n1, sv[2 * t + 1][3] - n1)) };
      mma16816(lsum, pa, ONES_H2, ONES_H2);
      for (int et = 0; et < D / 8; et += 2) {
        uint32_t vb[4];
        ldsm4t(vb, V + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (et + (lane >> 4)) * 8);
        mma16816(o[et], pa, vb[0], vb[1]);
        mma16816(o[et + 1], pa, vb[2], vb[3]);
      }
    }
    __syncthreads();
    // (REG: tile+1, loaded while this one computed, into the stage every warp has finished reading; then
    // tile+2's loads issued, to land while tile+1 computes)
    if constexpr (REG) {
      if (tile + 1 < tiles) { storeR(); __syncthreads(); if (tile + 2 < tiles) loadR((tile + 2) * BK); }
    }
  }
  float l0 = lsum[0], l1 = lsum[2];
  // gates read first and stored as pairs: with a store between every load the compiler
  // cannot reorder (out may alias qkvg) and the epilogue was 24 serial round trips, 10 us
  half2 ga[D / 8], gb[D / 8];
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    ga[et] = i0 < n ? *reinterpret_cast<const half2*>(base + (STRIDED ? pos(i0) : (size_t)i0 * W4) + 3 * Wd + e) : half2{};
    gb[et] = i1 < n ? *reinterpret_cast<const half2*>(base + (STRIDED ? pos(i1) : (size_t)i1 * W4) + 3 * Wd + e) : half2{};
  }
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    float2 a = __half22float2(ga[et]), b2 = __half22float2(gb[et]);
    if (i0 < n) *reinterpret_cast<half2*>(out + outAt(i0) + e) =
        __floats2half2_rn(o[et][0] / l0 * sigm(a.x), o[et][1] / l0 * sigm(a.y));
    if (i1 < n) *reinterpret_cast<half2*>(out + outAt(i1) + e) =
        __floats2half2_rn(o[et][2] / l1 * sigm(b2.x), o[et][3] / l1 * sigm(b2.y));
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
template <int D, int KS>
__global__ void __launch_bounds__(KS * 32) flashSplitHalf(const half* __restrict__ qkvg, const half* __restrict__ bias,
    int biasStride, const float* __restrict__ mask, half* __restrict__ out, int n, int heads, size_t r0, bool tr, float scale,
    const float* qBias) {
  constexpr int BK = FS_BK, LDK = D + 8, LDB = BK + 8;
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
  auto issue = [&](int j0, int st) {
    half *K = Kst(st), *V = Vst(st), *B = Bst(st);
    for (int t = lane; t < BK * (D / 8) * 2; t += 32) {
      int which = t / (BK * (D / 8)), u = t % (BK * (D / 8)), jj = u / (D / 8), c = (u % (D / 8)) * 8;
      int j = j0 + jj;
      cpAsync16((which ? V : K) + jj * LDK + c, base + (size_t)(j < n ? j : 0) * W4 + (which + 1) * Wd + c, j < n);
    }
    for (int t = lane; t < 16 * (BK / 8); t += 32) {
      int qi = t / (BK / 8), c = (t % (BK / 8)) * 8, i = q0 + qi, j = j0 + c;
      bool ok = i < n && j < n;
      cpAsync16(B + qi * LDB + c, bias + ((size_t)h * n + (i < n ? i : 0)) * biasStride + (ok ? j : 0), ok);
    }
    int j = j0 + lane;       // (no mask: every key is real)
    Mst(st)[lane] = j < n ? (!mask || mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f) : -INFINITY;
    cpCommit();
  };
  int i0 = q0 + g, i1 = i0 + 8;
  auto q2 = [&](int i, int e) -> uint32_t {
    if (i >= n) return 0u;
    float2 f = __half22float2(*reinterpret_cast<const half2*>(base + (size_t)i * W4 + e));
    if (qBias) { f.x += qBias[h * D + e]; f.y += qBias[h * D + e + 1]; }
    return pack2(f.x * scale * LOG2E, f.y * scale * LOG2E);
  };
  uint32_t qa[D / 16][4];
  for (int ks = 0; ks < D / 16; ++ks) {
    int e = ks * 16 + tig * 2;
    qa[ks][0] = q2(i0, e); qa[ks][1] = q2(i1, e); qa[ks][2] = q2(i0, e + 8); qa[ks][3] = q2(i1, e + 8);
  }
  float o[D / 8][4] = {};
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  int tiles = (n + BK - 1) / BK;
  if (warp < tiles) issue(warp * BK, 0);
  // the output gates the merge below writes with, loaded now (at the end they were one more round trip)
  constexpr int GPT = (16 * D + KS * 32 - 1) / (KS * 32);
  float gate[GPT];
#pragma unroll
  for (int k = 0; k < GPT; ++k) {
    int t = threadIdx.x + k * KS * 32, row = t / D, e = t % D, i = q0 + row;
    gate[k] = t < 16 * D && i < n ? __half2float(base[(size_t)i * W4 + 3 * Wd + e]) : 0.f;
  }
  for (int tile = warp, it = 0; tile < tiles; tile += KS, ++it) {
    int st = it & 1;
    if (tile + KS < tiles) { issue((tile + KS) * BK, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
    __syncwarp();
    const half *K = Kst(st), *V = Vst(st), *B = Bst(st); const float* Ms = Mst(st);
    float sv[BK / 8][4];
    for (int nt = 0; nt < BK / 8; ++nt) {
      sv[nt][0] = sv[nt][1] = sv[nt][2] = sv[nt][3] = 0.f;
      for (int k2 = 0; k2 < D / 32 + (D % 32 ? 1 : 0); ++k2) {
        uint32_t kb[4];
        ldsm4(kb, K + (nt * 8 + (lane & 7)) * LDK + k2 * 32 + (lane >> 3) * 8);
        mma16816(sv[nt], qa[k2 * 2], kb[0], kb[1]);
        if (k2 * 2 + 1 < D / 16) mma16816(sv[nt], qa[k2 * 2 + 1], kb[2], kb[3]);
      }
    }
    const half* br0 = B + g * LDB;
    const half* br1 = br0 + 8 * LDB;
    float t0 = -INFINITY, t1 = -INFINITY;
    for (int nt = 0; nt < BK / 8; ++nt) {
      int jj = nt * 8 + tig * 2;
      float2 u = __half22float2(*reinterpret_cast<const half2*>(br0 + jj));
      float2 v = __half22float2(*reinterpret_cast<const half2*>(br1 + jj));
      sv[nt][0] += u.x + Ms[jj]; sv[nt][1] += u.y + Ms[jj + 1];
      sv[nt][2] += v.x + Ms[jj]; sv[nt][3] += v.y + Ms[jj + 1];
      t0 = fmaxf(t0, fmaxf(sv[nt][0], sv[nt][1])); t1 = fmaxf(t1, fmaxf(sv[nt][2], sv[nt][3]));
    }
    t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
    t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
    float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);
    float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
    m0 = n0; m1 = n1; l0 *= c0; l1 *= c1;
    for (int et = 0; et < D / 8; ++et) { o[et][0] *= c0; o[et][1] *= c0; o[et][2] *= c1; o[et][3] *= c1; }
    for (int nt = 0; nt < BK / 8; ++nt) {
      sv[nt][0] = exp2f(sv[nt][0] - n0); sv[nt][1] = exp2f(sv[nt][1] - n0);
      sv[nt][2] = exp2f(sv[nt][2] - n1); sv[nt][3] = exp2f(sv[nt][3] - n1);
      l0 += sv[nt][0] + sv[nt][1]; l1 += sv[nt][2] + sv[nt][3];
    }
    for (int t = 0; t < BK / 16; ++t) {
      uint32_t pa[4] = { pack2(sv[2 * t][0], sv[2 * t][1]), pack2(sv[2 * t][2], sv[2 * t][3]),
                         pack2(sv[2 * t + 1][0], sv[2 * t + 1][1]), pack2(sv[2 * t + 1][2], sv[2 * t + 1][3]) };
      for (int et = 0; et < D / 8; et += 2) {
        uint32_t vb[4];
        ldsm4t(vb, V + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDK + (et + (lane >> 4)) * 8);
        mma16816(o[et], pa, vb[0], vb[1]);
        mma16816(o[et + 1], pa, vb[2], vb[3]);
      }
    }
    __syncwarp();
  }
  l0 += __shfl_xor_sync(~0u, l0, 1); l0 += __shfl_xor_sync(~0u, l0, 2);
  l1 += __shfl_xor_sync(~0u, l1, 1); l1 += __shfl_xor_sync(~0u, l1, 2);
  // merge the KS partial softmaxes: [ks][16 rows][D] outputs, [ks][16] maxima and sums
  __syncthreads();
  float* Os = (float*)smem; float* Mx = Os + KS * 16 * D; float* Ls = Mx + KS * 16;
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    float* row0 = Os + ((size_t)warp * 16 + g) * D + e; float* row1 = row0 + 8 * D;
    row0[0] = o[et][0]; row0[1] = o[et][1]; row1[0] = o[et][2]; row1[1] = o[et][3];
  }
  if (tig == 0) { Mx[warp * 16 + g] = m0; Mx[warp * 16 + g + 8] = m1; Ls[warp * 16 + g] = l0; Ls[warp * 16 + g + 8] = l1; }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < GPT; ++k) {
    int t = threadIdx.x + k * KS * 32, row = t / D, e = t % D, i = q0 + row;
    if (t >= 16 * D || i >= n) continue;
    float M = -INFINITY;
    for (int k = 0; k < KS; ++k) M = fmaxf(M, Mx[k * 16 + row]);
    float L = 0.f, O = 0.f;
    for (int k = 0; k < KS; ++k) {
      float c = exp2f(Mx[k * 16 + row] - M);
      L += Ls[k * 16 + row] * c; O += Os[((size_t)k * 16 + row) * D + e] * c;
    }
    out[(rl * n + i) * Wd + h * D + e] = __float2half(O / L * sigm(gate[k]));
  }
}
template <int D, int KS>
void flashSplitHalfAt(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                      int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias) {
  size_t smem = std::max((size_t)KS * 2 * fsStage<D>(), (size_t)KS * 16 * (D + 2) * 4);
  static bool done = false;
  if (!done) { smemAttr((flashSplitHalf<D, KS>), (int)smem); done = true; }
  flashSplitHalf<D, KS><<<dim3((n + 15) / 16, (unsigned)(rows * heads)), 32 * KS, smem, STREAM>>>(
    qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
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
template <int D> __host__ __device__ constexpr int fa2Ldk() { return D == 32 ? D : D + 8; }
template <int D> __device__ __forceinline__ int fa2Kv(int r, int c) {
  if constexpr (D == 32) return r * D + (c ^ (((r >> 1) & 3) << 3));
  else return r * (D + 8) + c;
}
template <int D, int WARPS, int BK, int MT = 2, int RR = 1> __host__ __device__ constexpr size_t fa2Stage() {
  return (size_t)RR * 2 * BK * fa2Ldk<D>() * 2 + (size_t)(16 * MT * WARPS) * (BK + 8) * 2;
}
// NB: no bias at all (AF2's MSA column attention): the scores start at zero and no bias tile is loaded
template <int D, int WARPS, int BK, int MT = 2, int RR = 1, bool NB = false>
__global__ void __launch_bounds__(WARPS * RR * 32) flashGrid2R(const half* __restrict__ qkvg, const half* __restrict__ bias,
    int biasStride, half* __restrict__ out, int n, int heads, float scale, const float* qBias, size_t rowsTotal,
    size_t rowStride, size_t posStride, size_t outRowStride, size_t outPosStride) {
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
  size_t b = blockIdx.y; int h = (int)(b / perHead); size_t rl = (b % perHead) * RR + rg;
  const bool live = RR == 1 || rl < rowsHere;                        // a last block's spare row computes, stores nothing
  if (!live) rl = rowsHere - 1;
  const int Wd = heads * D, W4 = 4 * Wd;
  const half* base = qkvg + rl * rowStride + h * D;
  int q0 = blockIdx.x * BQ;
  constexpr int KV_CHUNKS = BK * (D / 8), B_CHUNKS = BQ * (BK / 8);
  static_assert(B_CHUNKS % NT == 0, "a bias tile is a whole number of chunks a thread");
  const half* biasHead = bias + (size_t)h * n * biasStride;
  constexpr int KV_PER = (KV_CHUNKS + NTR - 1) / NTR, B_PER = B_CHUNKS / NT;
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
    bRow[k] = i < n; bC[k] = c; bOff[k] = qi * LDB + c;
    bSrc[k] = biasHead + (size_t)(i < n ? i : 0) * biasStride + c;
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
        for (int k = 0; k < B_PER; ++k) { cpAsync16(B + bOff[k], bRow[k] ? bCur[k] : biasHead, bRow[k]); bCur[k] += BK; }
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
        bool ok = bRow[k] && j0 + bC[k] < n;
        cpAsync16(B + bOff[k], ok ? bCur[k] : biasHead, ok);
      }
    }
    cpCommit();
  };
  auto q2 = [&](int i, int e) -> uint32_t {
    if (i >= n) return 0u;
    half2 v = *reinterpret_cast<const half2*>(base + (size_t)i * posStride + e);
    float2 f = __half22float2(v);
    if (qBias) { f.x += qBias[h * D + e]; f.y += qBias[h * D + e + 1]; }
    return pack2(f.x * scale * LOG2E, f.y * scale * LOG2E);
  };
  int ib[MT];                                                        // each tile's row g (and g + 8)
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) ib[mt] = q0 + warp * 16 * MT + mt * 16 + g;
  uint32_t qa[MT][D / 16][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      int e = ks * 16 + tig * 2;
      qa[mt][ks][0] = q2(ib[mt], e); qa[mt][ks][1] = q2(ib[mt] + 8, e);
      qa[mt][ks][2] = q2(ib[mt], e + 8); qa[mt][ks][3] = q2(ib[mt] + 8, e + 8);
    }
  float o[MT][D / 8][4] = {};
  float mx[MT][2], lsum[MT][4] = {};
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) mx[mt][0] = mx[mt][1] = -INFINITY;
  int tiles = (n + BK - 1) / BK;
  issue(0, 0);
  for (int tile = 0; tile < tiles; ++tile) {
    int st = tile & 1;
    if (tile + 1 < tiles) { issue((tile + 1) * BK, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
    __syncthreads();
    const half *K = Kst(st), *V = Vst(st), *B = Bst(st);
    uint32_t sh[MT][BK / 8][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const half* hb0 = B + (warp * 16 * MT + mt * 16 + g) * LDB;
      const half* hb1 = hb0 + 8 * LDB;
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
        int jj = nt * 8 + tig * 2;
        if constexpr (NB) { sh[mt][nt][0] = 0u; sh[mt][nt][1] = 0u; }
        else {
          sh[mt][nt][0] = *reinterpret_cast<const uint32_t*>(hb0 + jj);
          sh[mt][nt][1] = *reinterpret_cast<const uint32_t*>(hb1 + jj);
        }
        if (tile == tiles - 1) {
          int jg = tile * BK + jj;
          if (jg >= n) { sh[mt][nt][0] = (sh[mt][nt][0] & 0xFFFF0000u) | 0xFC00u; sh[mt][nt][1] = (sh[mt][nt][1] & 0xFFFF0000u) | 0xFC00u; }
          if (jg + 1 >= n) { sh[mt][nt][0] = (sh[mt][nt][0] & 0xFFFFu) | 0xFC000000u; sh[mt][nt][1] = (sh[mt][nt][1] & 0xFFFFu) | 0xFC000000u; }
        }
      }
    }
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt)
#pragma unroll
      for (int k2 = 0; k2 < D / 32 + (D % 32 ? 1 : 0); ++k2) {
        uint32_t kb[4];
        ldsm4(kb, K + fa2Kv<D>(nt * 8 + (lane & 7), k2 * 32 + (lane >> 3) * 8));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816h(sh[mt][nt], qa[mt][k2 * 2], kb[0], kb[1]);
          if (k2 * 2 + 1 < D / 16) mma16816h(sh[mt][nt], qa[mt][k2 * 2 + 1], kb[2], kb[3]);
        }
      }
    // (skipping the rescale when no row of the warp moved its max - a vote, each factor then exactly 1 - is
    // byte-identical and slower: 2211 -> 2251 ms of AF3's grid attention at 1,044 tokens, 111 -> 123 ms of
    // AF2's MSA column attention)
    __half2 hn[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      __half2 x0 = asH2(sh[mt][0][0]), x1 = asH2(sh[mt][0][1]);
#pragma unroll
      for (int nt = 1; nt < BK / 8; ++nt) { x0 = __hmax2(x0, asH2(sh[mt][nt][0])); x1 = __hmax2(x1, asH2(sh[mt][nt][1])); }
      float t0 = fmaxf(__low2float(x0), __high2float(x0)), t1 = fmaxf(__low2float(x1), __high2float(x1));
      t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 1)); t0 = fmaxf(t0, __shfl_xor_sync(~0u, t0, 2));
      t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 1)); t1 = fmaxf(t1, __shfl_xor_sync(~0u, t1, 2));
      float n0 = __half2float(__float2half(fmaxf(mx[mt][0], t0))), n1 = __half2float(__float2half(fmaxf(mx[mt][1], t1)));
      float c0 = exp2f(mx[mt][0] - n0), c1 = exp2f(mx[mt][1] - n1);
      mx[mt][0] = n0; mx[mt][1] = n1;
#pragma unroll
      for (int et = 0; et < D / 8; ++et) { o[mt][et][0] *= c0; o[mt][et][1] *= c0; o[mt][et][2] *= c1; o[mt][et][3] *= c1; }
      lsum[mt][0] *= c0; lsum[mt][1] *= c0; lsum[mt][2] *= c1; lsum[mt][3] *= c1;
      hn[mt][0] = __float2half2_rn(n0); hn[mt][1] = __float2half2_rn(n1);
    }
#pragma unroll
    for (int t = 0; t < BK / 16; ++t) {
      uint32_t pa[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        pa[mt][0] = ex2h2(asU32(__hsub2(asH2(sh[mt][2 * t][0]), hn[mt][0])));
        pa[mt][1] = ex2h2(asU32(__hsub2(asH2(sh[mt][2 * t][1]), hn[mt][1])));
        pa[mt][2] = ex2h2(asU32(__hsub2(asH2(sh[mt][2 * t + 1][0]), hn[mt][0])));
        pa[mt][3] = ex2h2(asU32(__hsub2(asH2(sh[mt][2 * t + 1][1]), hn[mt][1])));
        mma16816(lsum[mt], pa[mt], ONES_H2, ONES_H2);
      }
#pragma unroll
      for (int et = 0; et < D / 8; et += 2) {
        uint32_t vb[4];
        ldsm4t(vb, V + fa2Kv<D>(t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), (et + (lane >> 4)) * 8));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) { mma16816(o[mt][et], pa[mt], vb[0], vb[1]); mma16816(o[mt][et + 1], pa[mt], vb[2], vb[3]); }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    int i0 = ib[mt], i1 = i0 + 8;
    float l0 = lsum[mt][0], l1 = lsum[mt][2];
    half2 ga[D / 8], gb[D / 8];
    for (int et = 0; et < D / 8; ++et) {
      int e = et * 8 + tig * 2;
      ga[et] = i0 < n ? *reinterpret_cast<const half2*>(base + (size_t)i0 * posStride + 3 * Wd + e) : half2{};
      gb[et] = i1 < n ? *reinterpret_cast<const half2*>(base + (size_t)i1 * posStride + 3 * Wd + e) : half2{};
    }
    for (int et = 0; et < D / 8; ++et) {
      int e = et * 8 + tig * 2;
      float2 a = __half22float2(ga[et]), b2 = __half22float2(gb[et]);
      // staged in the warp's own slice of the (now idle) stage memory, written below as whole rows
      half* Ys = (half*)smem + (size_t)(threadIdx.x >> 5) * (16 * MT) * (D + 8);
      *reinterpret_cast<half2*>(Ys + (mt * 16 + g) * (D + 8) + e) =
          __floats2half2_rn(o[mt][et][0] / l0 * sigm(a.x), o[mt][et][1] / l0 * sigm(a.y));
      *reinterpret_cast<half2*>(Ys + (mt * 16 + g + 8) * (D + 8) + e) =
          __floats2half2_rn(o[mt][et][2] / l1 * sigm(b2.x), o[mt][et][3] / l1 * sigm(b2.y));
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
template <int D, int WARPS, int BK, int MT = 2, int RR = 1, bool NB = false>
void flashGrid2RRun(const half* qkvg, const half* bias, int stride, half* out, int n, int heads, size_t rows, float scale,
                    const float* qBias, size_t rowStride = 0, size_t posStride = 0, size_t outRowStride = 0,
                    size_t outPosStride = 0) {
  constexpr int BQ = 16 * MT * WARPS;
  const size_t W = (size_t)heads * D;
  if (!posStride) { rowStride = (size_t)n * 4 * W; posStride = 4 * W; outRowStride = (size_t)n * W; outPosStride = W; }
  const int bytes = 2 * (int)fa2Stage<D, WARPS, BK, MT, RR>();
  static bool attr = false;
  if (!attr) { smemAttr((flashGrid2R<D, WARPS, BK, MT, RR, NB>), bytes); attr = true; }
  dim3 grid((n + BQ - 1) / BQ, (unsigned)((rows + RR - 1) / RR * heads));
  flashGrid2R<D, WARPS, BK, MT, RR, NB><<<grid, 32 * WARPS * RR, bytes, STREAM>>>(qkvg, bias, stride, out, n, heads, scale, qBias, rows,
                                                                              rowStride, posStride, outRowStride, outPosStride);
}
template <int D, int WARPS, bool MASKED, int BK = FA_BK, bool REG = false, int MINB = 1, bool F16S = false> void setFlashSmem() {
  static bool done = false;
  if (!done) {
    smemAttr((flashGridHalf<D, WARPS, MASKED, BK, REG, MINB, false, F16S>), (int)((REG ? 1 : 2) * faStage<D, WARPS, BK>()));
    done = true;
  }
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
    int dev, major; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    return major < 8 ? 1 : 0;
  }();
  return v != 0;
}
template <int D, int WARPS, int BK = FA_BK, bool REG = false, int MINB = 1>
void flashGridHalfRun(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                      int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias) {
  dim3 grid((n + 16 * WARPS - 1) / (16 * WARPS), (unsigned)(rows * heads));
  const int bytes = (REG ? 1 : 2) * faStage<D, WARPS, BK>();
  if (mask) {                    // a null mask: every key real (see MASKED)
    setFlashSmem<D, WARPS, true, BK, REG, MINB>();
    flashGridHalf<D, WARPS, true, BK, REG, MINB><<<grid, 32 * WARPS, bytes, STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
  } else if (FLASH_2R && !REG && D == 32 && WARPS == 4 && BK == 48) {   // (the unmasked kernel reads no r0)
    flashGrid2RRun<D, 2, BK, 2, 2>(qkvg, bias, stride, out, n, heads, rows, scale, qBias);
  } else if (FLASH_F16S) {
    setFlashSmem<D, WARPS, false, BK, REG, MINB, true>();
    flashGridHalf<D, WARPS, false, BK, REG, MINB, false, true><<<grid, 32 * WARPS, bytes, STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
  } else {
    setFlashSmem<D, WARPS, false, BK, REG, MINB>();
    flashGridHalf<D, WARPS, false, BK, REG, MINB><<<grid, 32 * WARPS, bytes, STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
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
  // 128 queries a block unless that pads >32 rows. Narrower blocks were measured for small n
  // (a token transformer at 68 tokens is only 16 heads x 2 query blocks) and are not faster:
  // 1/2/4/8 warps read 19.6/18.6/17.8/17.6 us at 68 tokens and 32.3/30.3/23.4/26.4 at 261.
  int warps = ((n + 127) / 128) * 128 - n <= 32 ? 8 : 4;
  // the pair track's grid attention (a row per pair row): 4 warps at every size measured
  // (--bench-grid, 261 to 2048 tokens: 0.184 / 0.834 / 6.05 / 20.4 / 54.0 ms against 8 warps'
  // 0.227 / 0.894 / 6.63 / 21.5 / 54.1); the token transformer's one row keeps the rule above
  if (rows >= 32) warps = 4;               // (the denoiser's rows are its samples, a handful)
  if (FLASH_WARPS_OVERRIDE) warps = FLASH_WARPS_OVERRIDE;
  // too few blocks to fill the device: split each 16 queries' keys over four warps instead.
  // Measured at 16 heads, D 48: 9.3 against 14.7 us at 68 tokens, 13.4/19.5 at 192, and worse
  // from 256 (22.8/21.5), where the merge outweighs the parallelism.
  // ...and ONE row only: with the samples as rows (--samples=5) the grid kernel has the blocks it
  // lacked, and the split loses at every size measured - 11.1 against 8.2 us at 68 tokens and five
  // rows, 27.0 against 13.4 at 150, 15.7 against 8.8 at 192 and two rows (a tie at 68 and two)
  if (!FLASH_WARPS_OVERRIDE && FLASH_SPLIT && rows == 1 && n <= 192 && heads * ((n + 63) / 64) < 4 * 108 &&
      fitsSmem(4 * 2 * fsStage<D>())) {        // (the split's four warps' own buffers: 67 KB at D 48, past a T4's 64)
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
  if (D == 48 && n >= 1700 && (warps == 4 || warps == 8) &&
      fitsSmem(stages * (warps == 8 ? faStage<D, 8, FA_BK>() : faStage<D, 4, FA_BK>()))) {
    if (warps == 8) flashGridHalfAt<D, 8, FA_BK>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    else flashGridHalfAt<D, 4, FA_BK>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    return;
  }
  switch (warps) {
    case 8: flashGridHalfAt<D, 8, BK>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias); break;
    case 4:
      // the pair track's 32-wide heads in 48-key tiles where cp.async double-buffers them: smaller stages, more
      // blocks an SM - --bench-grid on an A100 1.036 against 1.211 ms at 524 tokens, 7.23 against 7.80 at 1044,
      // level at 262; an L4 (99 KB a block) 0.566 against 0.747 at 262. A T4's register-staged form keeps 64
      // (48 is 1.45 against 1.22 ms there)
      if constexpr (D == 32) {
        if (!flashRegStaged()) { flashGridHalfAt<D, 4, 48>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias); break; }
      }
      flashGridHalfAt<D, 4, BK>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias); break;
    case 2: flashGridHalfAt<D, 2>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias); break;
    default: flashGridHalfAt<D, 1>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
  }
}
template <class T>
void flashGrid(const T* qkvg, const T* bias, int stride, const float* mask, T* out, int n, int heads,
               int D, size_t r0, size_t rows, bool tr, float scale, const float* qBias = nullptr) {
  if constexpr (std::is_same_v<T, half>) {
    if (D == 32) flashGridHalfLaunch<32>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    else if (D == 48) flashGridHalfLaunch<48>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    else if (D == 16) flashGridHalfLaunch<16>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
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
