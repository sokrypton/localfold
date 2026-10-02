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

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void cpAsync16(void* dst, const void* src, bool valid) {
  uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

constexpr int FA_BK = 64;
template <int D, int WARPS> __host__ __device__ constexpr size_t faStage() {
  return (size_t)2 * FA_BK * (D + 8) * 2 + (size_t)(16 * WARPS) * (FA_BK + 8) * 2 + FA_BK * 4;
}

// mask[r * n + j] (rows) or mask[j * n + r] (columns): the KEY's mask, the pair mask
// transposed for the column direction; -1e9 where it is zero. MASKED false: every key is real
// (a protein with no padding - the mask is all ones), so no mask is loaded or added and only the
// last tile masks the keys past n (3-7% of this kernel).
__device__ __forceinline__ uint32_t ex2h2(uint32_t x) {
  uint32_t y; asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x)); return y;
}
constexpr uint32_t ONES_H2 = 0x3C003C00u;      // half2(1, 1)
template <int D, int WARPS, bool MASKED = true>
__global__ void __launch_bounds__(WARPS * 32) flashGridHalf(const half* __restrict__ qkvg, const half* __restrict__ bias,
    int biasStride, const float* __restrict__ mask, half* __restrict__ out, int n, int heads, size_t r0, bool tr, float scale,
    const float* qBias) {
  constexpr int BQ = 16 * WARPS, BK = FA_BK, LDK = D + 8, LDB = BK + 8, NT = WARPS * 32;
  constexpr size_t STAGE = faStage<D, WARPS>();
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
  const half* base = qkvg + rl * (size_t)n * W4 + h * D;
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
    kvSrc[k] = base + jj * W4 + Wd + c;
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
      const half* src = ok ? kvSrc[k] + j0 * W4 : base;
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
    asm volatile("cp.async.commit_group;");
  };
  int i0 = q0 + warp * 16 + g, i1 = i0 + 8;
  auto q2 = [&](int i, int e) -> uint32_t {
    if (i >= n) return 0u;
    half2 v = *reinterpret_cast<const half2*>(base + (size_t)i * W4 + e);
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
  issue(0, 0);
  for (int tile = 0; tile < tiles; ++tile) {
    int st = tile & 1;
    if (tile + 1 < tiles) { issue((tile + 1) * BK, st ^ 1); asm volatile("cp.async.wait_group 1;"); }
    else asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    const half *K = Kst(st), *V = Vst(st), *B = Bst(st); const float* Ms = Mst(st);
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
#pragma unroll
    for (int w = BK / 16; w >= 1; w >>= 1)
#pragma unroll
      for (int nt = 0; nt < w; ++nt) { r0m[nt] = fmaxf(r0m[nt], r0m[nt + w]); r1m[nt] = fmaxf(r1m[nt], r1m[nt + w]); }
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
  }
  float l0 = lsum[0], l1 = lsum[2];
  // gates read first and stored as pairs: with a store between every load the compiler
  // cannot reorder (out may alias qkvg) and the epilogue was 24 serial round trips, 10 us
  half2 ga[D / 8], gb[D / 8];
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    ga[et] = i0 < n ? *reinterpret_cast<const half2*>(base + (size_t)i0 * W4 + 3 * Wd + e) : half2{};
    gb[et] = i1 < n ? *reinterpret_cast<const half2*>(base + (size_t)i1 * W4 + 3 * Wd + e) : half2{};
  }
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    float2 a = __half22float2(ga[et]), b2 = __half22float2(gb[et]);
    if (i0 < n) *reinterpret_cast<half2*>(out + (rl * n + i0) * Wd + h * D + e) =
        __floats2half2_rn(o[et][0] / l0 * sigm(a.x), o[et][1] / l0 * sigm(a.y));
    if (i1 < n) *reinterpret_cast<half2*>(out + (rl * n + i1) * Wd + h * D + e) =
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
    int j = j0 + lane;
    Mst(st)[lane] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f) : -INFINITY;
    asm volatile("cp.async.commit_group;");
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
  for (int tile = warp, it = 0; tile < tiles; tile += KS, ++it) {
    int st = it & 1;
    if (tile + KS < tiles) { issue((tile + KS) * BK, st ^ 1); asm volatile("cp.async.wait_group 1;"); }
    else asm volatile("cp.async.wait_group 0;");
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
  for (int t = threadIdx.x; t < 16 * D; t += KS * 32) {
    int row = t / D, e = t % D, i = q0 + row;
    if (i >= n) continue;
    float M = -INFINITY;
    for (int k = 0; k < KS; ++k) M = fmaxf(M, Mx[k * 16 + row]);
    float L = 0.f, O = 0.f;
    for (int k = 0; k < KS; ++k) {
      float c = exp2f(Mx[k * 16 + row] - M);
      L += Ls[k * 16 + row] * c; O += Os[((size_t)k * 16 + row) * D + e] * c;
    }
    out[(rl * n + i) * Wd + h * D + e] = __float2half(O / L * sigm(__half2float(base[(size_t)i * W4 + 3 * Wd + e])));
  }
}
template <int D, int KS>
void flashSplitHalfAt(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                      int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias) {
  size_t smem = std::max((size_t)KS * 2 * fsStage<D>(), (size_t)KS * 16 * (D + 2) * 4);
  static bool done = false;
  if (!done) { CK(cudaFuncSetAttribute(flashSplitHalf<D, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); done = true; }
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

template <int D, int WARPS, bool MASKED> void setFlashSmem() {
  static bool done = false;
  if (!done) {
    CK(cudaFuncSetAttribute(flashGridHalf<D, WARPS, MASKED>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)(2 * faStage<D, WARPS>())));
    done = true;
  }
}
inline int FLASH_WARPS_OVERRIDE = 0;
inline bool FLASH_SPLIT = true;
template <int D, int WARPS>
void flashGridHalfAt(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                     int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias) {
  dim3 grid((n + 16 * WARPS - 1) / (16 * WARPS), (unsigned)(rows * heads));
  if (mask) {                    // a null mask: every key real (see MASKED)
    setFlashSmem<D, WARPS, true>();
    flashGridHalf<D, WARPS, true><<<grid, 32 * WARPS, 2 * faStage<D, WARPS>(), STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
  } else {
    setFlashSmem<D, WARPS, false>();
    flashGridHalf<D, WARPS, false><<<grid, 32 * WARPS, 2 * faStage<D, WARPS>(), STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale, qBias);
  }
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
  if (!FLASH_WARPS_OVERRIDE && FLASH_SPLIT && mask && n <= 192 && rows * heads * ((n + 63) / 64) < 4 * 108) {
    flashSplitHalfAt<D, 4>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
    return;
  }
  switch (warps) {
    case 8: flashGridHalfAt<D, 8>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias); break;
    case 4: flashGridHalfAt<D, 4>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias); break;
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
