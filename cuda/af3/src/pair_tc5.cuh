// The pair track's input projections on the B200's tcgen05 tensor cores (sm_100a). gridInK (the grid attention's
// LN + q/k/v/gate projection, and the pair bias) ran on mma.sync, whose peak here is 559 TFLOP/s - an RTX PRO 6000's -
// at ~150 TFLOP/s and 1.3 TB/s; this is the same arithmetic on tcgen05 at the memory's speed.
//
// gridInTc: a persistent block an SM over tiles of 128 output rows. Warps 0-3 lay each row's LayerNorm (gridInK's:
// exact two-pass variance, eps 1e-5, rounded to f16) straight into a 128B-swizzled K-major A tile, double buffered;
// the weight (C x 4C, transposed once to K-major and cached) sits in shared memory for the block's life; one thread
// issues the products - q|k into tensor-memory columns 0..255, v|gate into 256..511, and (with the bias) LN x Wb into
// columns 0..15 once q|k are drained; warps 4-7 drain them: q and gate f16 into `out` (gate + qb), k and v e4m3 into
// kv8, the bias heads f16 (x log2 e) into [h][i][stride].
#pragma once
#include "flash_tc5.cuh"


namespace ptc {
using namespace tc5a;
constexpr int C = 128, NQ = 512, TM = 128;
// shared memory: A [2 buffers][2 K-atoms][128 rows][128 B] 64 KB; B (the weight, K-major) [2 K-atoms][512 rows][128 B]
// 128 KB; Bb (the bias weight) [2][16 rows][128 B] 4 KB; the store staging [2][8 KB]; LN scale/offset 1 KB; barriers

// the K-major weight copies (Wq is C x NQ row-major), made once a weight
inline const half* kMajor(const half* w, int K, int N) {
  // (a derived weight: forgotten with the others - cuda/ef2 warms up on weights still arriving, test/cuda-derived-caches)
  static LF_TLS std::map<const half*, half*> made;
  static LF_TLS bool hooked = false;
  if (!hooked) { FORGET_HOOKS.push_back([] { for (auto& [k, p] : made) CK(cudaFreeAsync(p, STREAM)); made.clear(); }); hooked = true; }
  auto it = made.find(w);
  if (it != made.end()) return it->second;
  std::vector<half> h((size_t)K * N), t((size_t)K * N);
  CK(cudaMemcpy(h.data(), w, h.size() * 2, cudaMemcpyDeviceToHost));
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) t[(size_t)n * K + k] = h[(size_t)k * N + n];
  half* d; CK(cudaMalloc(&d, t.size() * 2)); CK(cudaMemcpy(d, t.data(), t.size() * 2, cudaMemcpyHostToDevice));
  made[w] = d;
  return d;
}
// a K-major row of 128 f16 (256 B) into a 128B-swizzled [2 atoms][rows][128 B] region: 16-byte chunk c of row r
__device__ __forceinline__ uint32_t swOff(int r, int c, int rows) { return (c >> 3) * rows * 128 + r * 128 + (((c & 7) ^ (r & 7)) << 4); }
// LayerNorm of a 128-row tile in place (raw bf16 -> f16, the 128B-swizzled K-major layout both share), a QUAD a row: lanes
// 4r'..4r'+3 hold row r' of the warp's 8, a lane 32 channels (chunks 4q..4q+3), its scale and offset in registers; two-
// pass statistics over the quad (two shuffles each) - gridInK's arithmetic (exact two-pass variance, eps 1e-5). A warp a
// row spent its time in two 5-level shuffle reductions, a thread a row in three passes of dependent loads (transTc:
// the products waited on its LN 38% / 53% of the time)
struct QuadLn {
  float sc[32], of[32];
  __device__ void init(const float* lnS, int lane) {
    const int q = lane & 3;
#pragma unroll
    for (int i = 0; i < 32; ++i) { sc[i] = lnS[32 * q + i]; of[i] = lnS[C + 32 * q + i]; }
  }
  // rows warpIdx*8 + (lane>>2), stepping 8*nWarps, of the tile at R (128 rows)
  __device__ void run(unsigned char* R, int warpIdx, int nWarps, int lane) const {
    const int q = lane & 3;
    for (int r = warpIdx * 8 + (lane >> 2); r < TM; r += 8 * nWarps) {
      float v[32];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint4 w = *reinterpret_cast<const uint4*>(R + swOff(r, 4 * q + j, TM));
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&(&w.x)[e])); v[8 * j + 2 * e] = f.x; v[8 * j + 2 * e + 1] = f.y; }
      }
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int i = 0; i < 32; i += 2) { s0 += v[i]; s1 += v[i + 1]; }
      float sum = s0 + s1;
      sum += __shfl_xor_sync(~0u, sum, 1); sum += __shfl_xor_sync(~0u, sum, 2);
      const float mean = sum / C;
      float q0 = 0.f, q1 = 0.f;
#pragma unroll
      for (int i = 0; i < 32; i += 2) { const float d0 = v[i] - mean, d1 = v[i + 1] - mean; q0 += d0 * d0; q1 += d1 * d1; }
      float qd = q0 + q1;
      qd += __shfl_xor_sync(~0u, qd, 1); qd += __shfl_xor_sync(~0u, qd, 2);
      const float inv = rsqrtf(qd / C + 1e-5f);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint4 w;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = 8 * j + 2 * e;
          (&w.x)[e] = f16x2((v[i] - mean) * inv * sc[i] + of[i], (v[i + 1] - mean) * inv * sc[i + 1] + of[i + 1]);
        }
        *reinterpret_cast<uint4*>(R + swOff(r, 4 * q + j, TM)) = w;
      }
    }
  }
};

// gridInTc: a persistent block an SM over tiles of 128 pair rows - (grid row a, positions b0 .. b0+127), never crossing
// a grid row, so both directions are a TMA box (the column direction's rows are n C apart). Warp 9 loads each raw bf16
// tile by TMA into a 128B-swizzled K-major A buffer (double buffered); warps 0-3 lay each row's LayerNorm (gridInK's:
// exact two-pass variance, eps 1e-5, rounded to f16) IN PLACE - bf16 and f16 are both two bytes, so a row's normalised
// values go back to the addresses its raw ones came from, already in the MMA's layout; warp 8 issues the products (the
// weight, transposed once to K-major and cached, in shared memory for the block's life): q|k into tensor-memory columns
// 0..255, v|gate into 256..511, and with the bias LN x Wb into 0..15 once q|k are drained; warps 4-7 drain them 32
// columns at a time through a swizzled staging buffer that TMA stores - q and gate f16 into `out` (gate + qb), k and v
// e4m3 into kv8 - and write the bias heads f16 (x log2 e) into [h][i][stride]. (Rows past n in a grid row's last tile,
// and rows outside the window, fall outside the stores' boxes, which TMA clips.)
// The first form loaded a row a warp (8 KB in flight an SM, ~0.85 TB/s) and stored 16 bytes a thread a 1 KB-apart row:
// no faster than gridInK (0.62 against 0.57 ms at 1M pairs).
struct Lay { static constexpr uint32_t A = 0, B = 65536, BB = B + 131072, ST = BB + 4096, LN = ST + 16384, BAR = LN + 1024, BYTES = BAR + 256; };
__device__ __forceinline__ void tmaStore3(const CUtensorMap* map, const void* src, int c0, int c1, int c2) {
#if LF_TC5A
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
               :: "l"((uint64_t)map), "r"(sa(src)), "r"(c0), "r"(c1), "r"(c2) : "memory");
#endif
}
__device__ __forceinline__ void bulkCommit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulkWaitRead() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulkWait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void epiSync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }   // (warps 4-7)

__global__ void __launch_bounds__(320, 1) gridInTc(const __grid_constant__ CUtensorMap mapP, const __grid_constant__ CUtensorMap mapO,
    const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ WqT, int n, int a0, int A, int wa0, bool tr, const half* __restrict__ WbT,
    half* __restrict__ bias, int heads, int stride, bool swap, const float* __restrict__ qb, bool qkvg) {
#if LF_TC5A
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(sm + Lay::BAR);
  uint64_t *ldFull = bar, *aFull = bar + 2, *aFree = bar + 4, *dFull = bar + 6, *dFree = bar + 8;
  __shared__ uint32_t tmBase;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nbt = (n + TM - 1) / TM;
  const size_t tiles = (size_t)A * nbt;
  const int mine = tiles > blockIdx.x ? (int)((tiles - blockIdx.x + gridDim.x - 1) / gridDim.x) : 0;
  const bool hasBias = bias != nullptr;
  if (warp == 8) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { barInit(ldFull + i, 1); barInit(aFull + i, 4); barInit(aFree + i, 1); barInit(dFull + i, 1); barInit(dFree + i, 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < NQ * 16; i += 320) {
    const int r = i >> 4, c = i & 15;
    *reinterpret_cast<uint4*>(sm + Lay::B + swOff(r, c, NQ)) = *reinterpret_cast<const uint4*>(WqT + (size_t)r * C + c * 8);
  }
  if (hasBias)
    for (int i = tid; i < 16 * 16; i += 320) {
      const int r = i >> 4, c = i & 15;
      *reinterpret_cast<uint4*>(sm + Lay::BB + swOff(r, c, 16)) = *reinterpret_cast<const uint4*>(WbT + (size_t)r * C + c * 8);
    }
  float* lnS = reinterpret_cast<float*>(sm + Lay::LN);
  for (int i = tid; i < C; i += 320) { lnS[i] = lnScale[i]; lnS[C + i] = lnOffset[i]; }
  fenceProxy();
  fenceBefore();
  __syncthreads();
  fenceAfter();
  const uint32_t tm = tmBase, base = sa(sm);
  // tile k of this block: grid row a (of A, from a0), positions b0 ..; the window's grid rows from wa0 (q, k, v, gate
  // only there - the stores' boxes clip the rest)
  auto tileOf = [&](int k, int& a, int& b0) { const size_t t = (size_t)blockIdx.x + (size_t)k * gridDim.x; a = (int)(t / nbt); b0 = (int)(t % nbt) * TM; };
  if (warp == 9) {
    if (lane == 0) {
      for (int k = 0; k < mine; ++k) {
        int a, b0; tileOf(k, a, b0);
        const int b = k & 1;
        if (k >= 2) barWait(aFree + b, ((k - 2) >> 1) & 1);
        barExpect(ldFull + b, 32768);
        unsigned char* Ab = sm + Lay::A + b * 32768;
        tma3(Ab, &mapP, 0, b0, a0 + a, ldFull + b);
        tma3(Ab + 16384, &mapP, 64, b0, a0 + a, ldFull + b);
      }
    }
  } else if (warp < 4) {
    float sc[4], of[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) { sc[e] = lnS[lane * 4 + e]; of[e] = lnS[C + lane * 4 + e]; }
    for (int k = 0; k < mine; ++k) {
      const int b = k & 1;
      barWait(ldFull + b, (k >> 1) & 1);
      unsigned char* Ab = sm + Lay::A + b * 32768;
      for (int r0 = warp; r0 < TM; r0 += 32) {
        float v[8][4];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const uint2 w = *reinterpret_cast<const uint2*>(Ab + swOff(r0 + 4 * u, lane >> 1, TM) + (lane & 1) * 8);
          const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x)), y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
          v[u][0] = x.x; v[u][1] = x.y; v[u][2] = y.x; v[u][3] = y.y;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          float s = v[u][0] + v[u][1] + v[u][2] + v[u][3];
          for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
          const float mean = s / C;
          float qd = 0.f;
#pragma unroll
          for (int e = 0; e < 4; ++e) { const float d = v[u][e] - mean; qd += d * d; }
          for (int o = 16; o; o >>= 1) qd += __shfl_xor_sync(~0u, qd, o);
          const float inv = rsqrtf(qd / C + 1e-5f);
          uint2 w;
          w.x = f16x2((v[u][0] - mean) * inv * sc[0] + of[0], (v[u][1] - mean) * inv * sc[1] + of[1]);
          w.y = f16x2((v[u][2] - mean) * inv * sc[2] + of[2], (v[u][3] - mean) * inv * sc[3] + of[3]);
          *reinterpret_cast<uint2*>(Ab + swOff(r0 + 4 * u, lane >> 1, TM) + (lane & 1) * 8) = w;
        }
      }
      fenceProxy();
      __syncwarp();
      if (lane == 0) barArrive(aFull + b);
    }
  } else if (warp == 8) {
    if (lane == 0) {
      constexpr uint32_t ID256 = idesc(TM, 256), ID16 = idesc(TM, 16);
      int u0 = 0, u1 = 0;
      auto take = [&](int h, int& u) { if (u > 0) barWait(dFree + h, (u - 1) & 1); fenceAfter(); ++u; };
      for (int k = 0; k < mine; ++k) {
        int a, b0; tileOf(k, a, b0);
        const int b = k & 1;
        barWait(aFull + b, (k >> 1) & 1);
        fenceAfter();
        const uint32_t Ab = base + Lay::A + b * 32768;
        auto prod = [&](uint32_t d, uint32_t bAddr, int bRows, uint32_t id) {
#pragma unroll
          for (int s = 0; s < 8; ++s)
            mmaF16(d, sdesc(Ab + (s >> 2) * TM * 128 + (s & 3) * 32, 16, 1024, SW128),
                   sdesc(bAddr + (s >> 2) * bRows * 128 + (s & 3) * 32, 16, 1024, SW128), id, s > 0);
        };
        if (qkvg) {
          take(0, u0); prod(tm, base + Lay::B, NQ, ID256); commit(dFull + 0);
          take(1, u1); prod(tm + 256, base + Lay::B + 256 * 128, NQ, ID256); commit(dFull + 1);
        }
        if (hasBias) { take(0, u0); prod(tm, base + Lay::BB, 16, ID16); commit(dFull + 0); }
        commit(aFree + b);
      }
    }
  } else {
    // warps 4-7: a thread a row (tensor memory's lane), 32 columns at a time through staging buffer (chunk & 1)
    const int sub = warp - 4, rl = sub * 32 + lane;
    const uint32_t lane0 = (uint32_t)(sub * 32) << 16;
    const bool T0 = rl == 0;
    int u0 = 0, u1 = 0, chunk = 0;
    auto wait = [&](int h, int& u) { barWait(dFull + h, u & 1); fenceAfter(); ++u; };
    auto release = [&](int h) { fenceBefore(); __syncwarp(); if (lane == 0) barArrive(dFree + h); };
    // one 32-column chunk: f16 (64 B a row, 64B swizzle) or e4m3 (32 B, 32B swizzle) into staging, then a TMA store
    auto emit = [&](const uint32_t* x, bool f8, const float* add, const CUtensorMap* map, int col, int a, int b0) {
      unsigned char* S = sm + Lay::ST + (chunk & 1) * 8192;
      if (T0) bulkWaitRead<1>();   // (the store that last read this buffer, two chunks ago)
      epiSync();
      if (!f8) {
        uint32_t w[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          float a0f = __uint_as_float(x[2 * e]), a1f = __uint_as_float(x[2 * e + 1]);
          if (add) { a0f += add[2 * e]; a1f += add[2 * e + 1]; }
          w[e] = f16x2(a0f, a1f);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
          *reinterpret_cast<uint4*>(S + rl * 64 + ((j ^ ((rl >> 1) & 3)) << 4)) = make_uint4(w[4 * j], w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]);
      } else {
        uint32_t w[8];
#pragma unroll
        for (int e = 0; e < 8; ++e)
          w[e] = e4m3x2(__uint_as_float(x[4 * e]), __uint_as_float(x[4 * e + 1])) | (uint32_t)e4m3x2(__uint_as_float(x[4 * e + 2]), __uint_as_float(x[4 * e + 3])) << 16;
#pragma unroll
        for (int j = 0; j < 2; ++j)
          *reinterpret_cast<uint4*>(S + rl * 32 + ((j ^ ((rl >> 2) & 1)) << 4)) = make_uint4(w[4 * j], w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]);
      }
      fenceProxy();
      epiSync();
      if (T0) { tmaStore3(map, S, col, b0, a - wa0); bulkCommit(); }
      ++chunk;
    };
    for (int k = 0; k < mine; ++k) {
      int a, b0; tileOf(k, a, b0);
      if (qkvg) {
        wait(0, u0);
        for (int c = 0; c < 8; ++c) {
          uint32_t x[32];
          ld32(tm + lane0 + c * 32, x);
          ldWait();
          if (c < 4) emit(x, false, nullptr, &mapO, c * 32, a, b0);    // q
          else emit(x, true, nullptr, &mapK, (c - 4) * 32, a, b0);     // k
        }
        release(0);
        wait(1, u1);
        for (int c = 0; c < 8; ++c) {
          uint32_t x[32];
          ld32(tm + lane0 + 256 + c * 32, x);
          ldWait();
          if (c < 4) emit(x, true, nullptr, &mapV, c * 32, a, b0);                                   // v
          else emit(x, false, qb ? qb + (c - 4) * 32 : nullptr, &mapO, 3 * C + (c - 4) * 32, a, b0); // gate + qb
        }
        release(1);
      }
      if (hasBias) {
        wait(0, u0);
        uint32_t x[16];
        ld16(tm + lane0, x);
        ldWait();
        release(0);
        const int bq = b0 + rl;
        if (bq < n) {
          const unsigned ga = (unsigned)(a0 + a), gb = (unsigned)bq;    // the pair (ga, gb) in the call's own order
          const unsigned p = tr ? gb : ga, qq = tr ? ga : gb;            // (row, col) of the pair as stored
          const size_t i = swap ? qq : p, jj = swap ? p : qq;
          for (int h = 0; h < heads; ++h) bias[((size_t)h * n + i) * stride + jj] = __float2half(LOG2E * __uint_as_float(x[h]));
        }
      }
    }
    if (T0) bulkWait<0>();
  }
  fenceBefore();
  __syncthreads();
  if (warp == 8) {
    fenceAfter();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm));
  }
#endif
}
inline LF_TLS bool PAIR_TC = true;   // --no-tc-pair: the mma.sync pair-track kernels on a B200 too
// gridInRaw's contract for its common AF3 form - a bf16 pair, C 128, kv8 (no vt8), calls covering whole grid rows (q0,
// rows and the window multiples of n): true where it ran; false - the caller runs gridInK. qb as gridInRaw takes it:
// indexed by the projected column (its gate quarter's from 3C)
inline bool gridInTcRun(bool pair16, const float* pair, const float* lnScale, const float* lnOffset, const half* Wq, const float* qb,
                        half* out, int n, size_t q0, size_t rows, bool tr, const half* Wb, half* bias, int heads, int stride,
                        bool swap, uint8_t* kv8, uint8_t* vt8, size_t wlo, size_t whi) {
  const size_t wHi = std::min(whi, rows);
  if (!PAIR_TC || !ready() || !pair16 || !kv8 || vt8 || (bias && heads > 16) || q0 % n || rows % n || wlo % n || wHi % n || wHi <= wlo) return false;
  const half* WqT = kMajor(Wq, C, NQ);
  const half* WbT = bias ? kMajor(Wb, C, 16) : nullptr;
  const uint64_t N = (uint64_t)n, wA = (wHi - wlo) / n;
  // the pair as [a][b][C]: (a, b) at (a n + b) C in the row direction, (b n + a) C in the column direction
  const CUtensorMap mp = map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, pair, { (uint64_t)C, N, N },
                              { tr ? N * C * 2 : (uint64_t)C * 2, tr ? (uint64_t)C * 2 : N * C * 2 }, { 64, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap mo = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, out, { (uint64_t)NQ, N, wA }, { (uint64_t)NQ * 2, N * NQ * 2 }, { 32, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_64B);
  const CUtensorMap mk = map3(CU_TENSOR_MAP_DATA_TYPE_UINT8, kv8, { (uint64_t)C, N, wA }, { (uint64_t)C, N * C }, { 32, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_32B);
  const CUtensorMap mv = map3(CU_TENSOR_MAP_DATA_TYPE_UINT8, kv8 + (wHi - wlo) * C, { (uint64_t)C, N, wA }, { (uint64_t)C, N * C }, { 32, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_32B);
  static LF_TLS int sms = 0;
  if (!sms) { int dev; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); }
  const int A = (int)(rows / n), a0 = (int)(q0 / n), nbt = (n + TM - 1) / TM;
  // (a call whose tiles all lie outside the window but which writes the bias: the bias alone)
  const bool qkvg = true;
  const unsigned grid = (unsigned)std::min<size_t>((size_t)A * nbt, sms);
  static LF_TLS bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(gridInTc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)Lay::BYTES)); attr = true; }
  gridInTc<<<grid, 320, Lay::BYTES, STREAM>>>(mp, mo, mk, mv, lnScale, lnOffset, WqT, n, a0, A, (int)(wlo / n), tr, WbT, bias, heads, stride,
                                              swap, qb ? qb + 3 * C : nullptr, qkvg);
  CK(cudaGetLastError());
  return true;
}

// ---- triInTc: the triangle multiplication's input kernel (triIn256K's FP8 form at C 128) on tcgen05: LN(pair) times the
// projection [C, 2C] (column 2c a's channel c, 2c + 1 b's), the gate [C, 2C] and the gating linear [C, C]; a and b =
// projection x sigmoid(gate) x mask, e4m3, channel-major planes [C][np8 np8] (the padding zero: its mask 0); t2 = the
// gating linear's raw output, f16 [np8 np8][C] (its own pair's row - transposed in the incoming direction, rm.T).
// gridInTc's structure: 128-row tiles of the padded grid that never cross a grid row, the raw tile by TMA (the pair's
// rows and columns past n zero-filled), LN in place, the three weights K-major in shared memory for the block's life
// (160 KB), the projection and the gate into the two tensor-memory halves (N 256 each) and the gating linear into the
// first again once drained (N 128); a and b to global a channel a lane-row at a time (a warp's 32 rows are 32
// consecutive bytes of a plane); t2 staged in the tile's own A buffer (its products done) and TMA stored, which then
// frees the buffer for the next load.
namespace tri {
struct Lay { static constexpr uint32_t A = 0, WP = 65536, WG = WP + 65536, WL = WG + 65536, LN = WL + 32768, BAR = LN + 1024, BYTES = BAR + 256; };
}
__global__ void __launch_bounds__(320, 1) triInTc(const __grid_constant__ CUtensorMap mapP, const __grid_constant__ CUtensorMap mapT,
    const float* __restrict__ mask, const float* __restrict__ lnScale, const float* __restrict__ lnOffset, const half* __restrict__ WpT,
    const half* __restrict__ WgT, const half* __restrict__ WlT, uint8_t* __restrict__ a8, uint8_t* __restrict__ b8, int n, int np, size_t cs, bool T) {
#if LF_TC5A
  using L = tri::Lay;
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(sm + L::BAR);
  uint64_t *ldFull = bar, *aFull = bar + 2, *aFree = bar + 4, *dFull = bar + 6, *dFree = bar + 8;
  __shared__ uint32_t tmBase;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nbt = (np + TM - 1) / TM;
  const size_t tiles = (size_t)np * nbt;
  const int mine = tiles > blockIdx.x ? (int)((tiles - blockIdx.x + gridDim.x - 1) / gridDim.x) : 0;
  if (warp == 8) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { barInit(ldFull + i, 1); barInit(aFull + i, 4); barInit(aFree + i, 1); barInit(dFull + i, 1); barInit(dFree + i, 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 256 * 16; i += 320) {
    const int r = i >> 4, c = i & 15;
    *reinterpret_cast<uint4*>(sm + L::WP + swOff(r, c, 256)) = *reinterpret_cast<const uint4*>(WpT + (size_t)r * C + c * 8);
    *reinterpret_cast<uint4*>(sm + L::WG + swOff(r, c, 256)) = *reinterpret_cast<const uint4*>(WgT + (size_t)r * C + c * 8);
    if (r < C) *reinterpret_cast<uint4*>(sm + L::WL + swOff(r, c, C)) = *reinterpret_cast<const uint4*>(WlT + (size_t)r * C + c * 8);
  }
  float* lnS = reinterpret_cast<float*>(sm + L::LN);
  for (int i = tid; i < C; i += 320) { lnS[i] = lnScale[i]; lnS[C + i] = lnOffset[i]; }
  fenceProxy();
  fenceBefore();
  __syncthreads();
  fenceAfter();
  const uint32_t tm = tmBase, base = sa(sm);
  auto tileOf = [&](int k, int& i, int& b0) { const size_t t = (size_t)blockIdx.x + (size_t)k * gridDim.x; i = (int)(t / nbt); b0 = (int)(t % nbt) * TM; };
  if (warp == 9) {
    if (lane == 0) {
      for (int k = 0; k < mine; ++k) {
        int i, b0; tileOf(k, i, b0);
        const int b = k & 1;
        if (k >= 2) barWait(aFree + b, ((k - 2) >> 1) & 1);
        barExpect(ldFull + b, 32768);
        unsigned char* Ab = sm + L::A + b * 32768;
        tma3(Ab, &mapP, 0, b0, i, ldFull + b);
        tma3(Ab + 16384, &mapP, 64, b0, i, ldFull + b);
      }
    }
  } else if (warp < 4) {
    float sc[4], of[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) { sc[e] = lnS[lane * 4 + e]; of[e] = lnS[C + lane * 4 + e]; }
    for (int k = 0; k < mine; ++k) {
      const int b = k & 1;
      barWait(ldFull + b, (k >> 1) & 1);
      unsigned char* Ab = sm + L::A + b * 32768;
      for (int r0 = warp; r0 < TM; r0 += 32) {
        float v[8][4];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const uint2 w = *reinterpret_cast<const uint2*>(Ab + swOff(r0 + 4 * u, lane >> 1, TM) + (lane & 1) * 8);
          const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x)), y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
          v[u][0] = x.x; v[u][1] = x.y; v[u][2] = y.x; v[u][3] = y.y;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          float s = v[u][0] + v[u][1] + v[u][2] + v[u][3];
          for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
          const float mean = s / C;
          float qd = 0.f;
#pragma unroll
          for (int e = 0; e < 4; ++e) { const float d = v[u][e] - mean; qd += d * d; }
          for (int o = 16; o; o >>= 1) qd += __shfl_xor_sync(~0u, qd, o);
          const float inv = rsqrtf(qd / C + 1e-5f);
          uint2 w;
          w.x = f16x2((v[u][0] - mean) * inv * sc[0] + of[0], (v[u][1] - mean) * inv * sc[1] + of[1]);
          w.y = f16x2((v[u][2] - mean) * inv * sc[2] + of[2], (v[u][3] - mean) * inv * sc[3] + of[3]);
          *reinterpret_cast<uint2*>(Ab + swOff(r0 + 4 * u, lane >> 1, TM) + (lane & 1) * 8) = w;
        }
      }
      fenceProxy();
      __syncwarp();
      if (lane == 0) barArrive(aFull + b);
    }
  } else if (warp == 8) {
    if (lane == 0) {
      constexpr uint32_t ID256 = idesc(TM, 256), ID128 = idesc(TM, 128);
      int u0 = 0, u1 = 0;
      auto take = [&](int h, int& u) { if (u > 0) barWait(dFree + h, (u - 1) & 1); fenceAfter(); ++u; };
      for (int k = 0; k < mine; ++k) {
        const int b = k & 1;
        barWait(aFull + b, (k >> 1) & 1);
        fenceAfter();
        const uint32_t Ab = base + L::A + b * 32768;
        auto prod = [&](uint32_t d, uint32_t bAddr, int bRows, uint32_t id) {
#pragma unroll
          for (int s = 0; s < 8; ++s)
            mmaF16(d, sdesc(Ab + (s >> 2) * TM * 128 + (s & 3) * 32, 16, 1024, SW128),
                   sdesc(bAddr + (s >> 2) * bRows * 128 + (s & 3) * 32, 16, 1024, SW128), id, s > 0);
        };
        take(0, u0); prod(tm, base + L::WP, 256, ID256); commit(dFull + 0);         // the projection
        take(1, u1); prod(tm + 256, base + L::WG, 256, ID256); commit(dFull + 1);   // its gate
        take(0, u0); prod(tm, base + L::WL, C, ID128); commit(dFull + 0);           // the gating linear
      }
    }
  } else {
    const int sub = warp - 4, rl = sub * 32 + lane;
    const uint32_t lane0 = (uint32_t)(sub * 32) << 16;
    const bool T0 = rl == 0;
    int u0 = 0, u1 = 0;
    auto wait = [&](int h, int& u) { barWait(dFull + h, u & 1); fenceAfter(); ++u; };
    auto release = [&](int h) { fenceBefore(); __syncwarp(); if (lane == 0) barArrive(dFree + h); };
    for (int k = 0; k < mine; ++k) {
      int i, b0; tileOf(k, i, b0);
      const int j = b0 + rl, b = k & 1;
      const bool rowIn = j < np;
      const size_t q = (size_t)i * np + j;
      float m = 0.f;
      if (i < n && j < n) m = mask ? mask[T ? (size_t)j * n + i : (size_t)i * n + j] : 1.f;
      wait(0, u0); wait(1, u1);
      for (int c = 0; c < 8; ++c) {
        uint32_t p[32], g[32];
        ld32(tm + lane0 + c * 32, p);
        ld32(tm + lane0 + 256 + c * 32, g);
        ldWait();
        if (!rowIn) continue;
#pragma unroll
        for (int e = 0; e < 16; ++e) {      // columns 2e (a's channel 16c + e) and 2e + 1 (b's)
          const float va = __uint_as_float(p[2 * e]) * sigmH(__uint_as_float(g[2 * e])) * m,
                      vb = __uint_as_float(p[2 * e + 1]) * sigmH(__uint_as_float(g[2 * e + 1])) * m;
          const unsigned short ab = e4m3x2(va, vb);
          a8[(size_t)(16 * c + e) * cs + q] = (uint8_t)(ab & 0xFF);
          b8[(size_t)(16 * c + e) * cs + q] = (uint8_t)(ab >> 8);
        }
      }
      release(1);
      release(0);
      // t2: the gating linear's raw output, staged in this tile's A buffer (its products are done) and TMA stored
      wait(0, u0);
      unsigned char* S = sm + L::A + b * 32768;
      for (int c = 0; c < 4; ++c) {
        uint32_t x[32];
        ld32(tm + lane0 + c * 32, x);
        ldWait();
        uint32_t w[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) w[e] = f16x2(__uint_as_float(x[2 * e]), __uint_as_float(x[2 * e + 1]));
#pragma unroll
        for (int e = 0; e < 4; ++e)
          *reinterpret_cast<uint4*>(S + swOff(rl, c * 4 + e, TM)) = make_uint4(w[4 * e], w[4 * e + 1], w[4 * e + 2], w[4 * e + 3]);
      }
      release(0);
      fenceProxy();
      epiSync();
      if (T0) {
        tmaStore3(&mapT, S, 0, b0, i);
        tmaStore3(&mapT, S + 16384, 64, b0, i);
        bulkCommit();
        bulkWaitRead<0>();
        barArrive(aFree + b);
      }
    }
    if (T0) bulkWait<0>();
  }
  fenceBefore();
  __syncthreads();
  if (warp == 8) {
    fenceAfter();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm));
  }
#endif
}
// triIn256K's FP8 form at C 128 for a bf16 pair (the whole grid, rm.T the incoming direction's transposed read): true
// where it ran
inline bool triInTcRun(bool pair16, const float* pair, const float* mask, const float* lnScale, const float* lnOffset, const half* Wp,
                       const half* Wgate, const half* Wg, uint8_t* a8, uint8_t* b8, half* t2, int n, int np, size_t cs, bool T) {
  if (!PAIR_TC || !ready() || !pair16 || cs != (size_t)np * np) return false;
  const half *WpT = kMajor(Wp, C, 2 * C), *WgT = kMajor(Wgate, C, 2 * C), *WlT = kMajor(Wg, C, C);
  const uint64_t N = (uint64_t)n, NP = (uint64_t)np;
  const CUtensorMap mp = map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, pair, { (uint64_t)C, N, N },
                              { T ? N * C * 2 : (uint64_t)C * 2, T ? (uint64_t)C * 2 : N * C * 2 }, { 64, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap mt = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, t2, { (uint64_t)C, NP, NP },
                              { T ? NP * C * 2 : (uint64_t)C * 2, T ? (uint64_t)C * 2 : NP * C * 2 }, { 64, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  static LF_TLS int sms = 0;
  if (!sms) { int dev; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); }
  const unsigned grid = (unsigned)std::min<size_t>((size_t)np * ((np + TM - 1) / TM), sms);
  static LF_TLS bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(triInTc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tri::Lay::BYTES)); attr = true; }
  triInTc<<<grid, 320, tri::Lay::BYTES, STREAM>>>(mp, mt, mask, lnScale, lnOffset, WpT, WgT, WlT, a8, b8, n, np, cs, T);
  CK(cudaGetLastError());
  return true;
}

// ---- transTc: the pair transition (fusedTransitionK's AF3 form: x += (silu(LN(x) W1a) * LN(x) W1b) W2, C 128, I 512, a bf16
// pair) on tcgen05. Its weights (384 KB) do not fit shared memory, so they stream: a block takes 256 rows (two 128-row
// sub-tiles sharing every weight chunk) and walks the 8 chunks of 64 hidden channels - GEMM1 (the chunk's a|b columns,
// N 128) for both sub-tiles into tensor memory, SwiGLU by warps 4-11 (a sub-tile each four) into a 128B-swizzled h
// (f16, GEMM2's A), GEMM2 (h x the chunk's W2, K 64, N 128) accumulating both sub-tiles' outputs - the chunks' weights
// TMA double buffered (48 KB a chunk, from L2). Then x + the output, in f32, rounded to bf16, staged in the A buffer and
// TMA stored back over the rows.
namespace trs {
constexpr int I = 512, HC = 64, NCH = I / HC, BM = 256;
// shared memory: A [2 sub-tiles][2 atoms][128][128 B] 64 KB; weight stages [2][W1 chunk 32 KB, W2 chunk 16 KB]; h [2][16 KB]
// ...and OUT [2 sub-tiles][64 columns][128 rows][128 B] 32 KB, the output staged half its columns at a time - so A is free
// once the block's last GEMM1 is done, and the next block's load and LN run under its last GEMM2 and the store (with the
// output staged in A, the products waited on the next LN 46% of the time)
struct Lay { static constexpr uint32_t A = 0, W = 65536, WSTAGE = 49152, H = W + 2 * WSTAGE, OUT = H + 32768, LN = OUT + 32768, BAR = LN + 1024, BYTES = BAR + 256; };
// W1 as GEMM1's B a chunk at a time: chunk p [128 rows: a's columns 64p.., then b's I + 64p..][K 128]; W2 as GEMM2's B:
// chunk p [128 output columns][the chunk's 64 hidden] - both K-major, made once a weight
inline const half* w1Chunks(const half* W1) {
  static LF_TLS std::map<const half*, half*> made;
  static LF_TLS bool hooked = false;
  if (!hooked) { FORGET_HOOKS.push_back([] { for (auto& [k, p] : made) CK(cudaFreeAsync(p, STREAM)); made.clear(); }); hooked = true; }
  auto it = made.find(W1); if (it != made.end()) return it->second;
  std::vector<half> h((size_t)C * 2 * I), t((size_t)NCH * 128 * C);
  CK(cudaMemcpy(h.data(), W1, h.size() * 2, cudaMemcpyDeviceToHost));
  for (int p = 0; p < NCH; ++p) for (int r = 0; r < 128; ++r) for (int k = 0; k < C; ++k)
    t[((size_t)p * 128 + r) * C + k] = h[(size_t)k * 2 * I + (r < HC ? HC * p + r : I + HC * p + r - HC)];
  half* d; CK(cudaMalloc(&d, t.size() * 2)); CK(cudaMemcpy(d, t.data(), t.size() * 2, cudaMemcpyHostToDevice));
  return made[W1] = d;
}
inline const half* w2Chunks(const half* W2) {
  static LF_TLS std::map<const half*, half*> made;
  static LF_TLS bool hooked = false;
  if (!hooked) { FORGET_HOOKS.push_back([] { for (auto& [k, p] : made) CK(cudaFreeAsync(p, STREAM)); made.clear(); }); hooked = true; }
  auto it = made.find(W2); if (it != made.end()) return it->second;
  std::vector<half> h((size_t)I * C), t((size_t)NCH * C * HC);
  CK(cudaMemcpy(h.data(), W2, h.size() * 2, cudaMemcpyDeviceToHost));
  for (int p = 0; p < NCH; ++p) for (int o = 0; o < C; ++o) for (int k = 0; k < HC; ++k)
    t[((size_t)p * C + o) * HC + k] = h[(size_t)(HC * p + k) * C + o];
  half* d; CK(cudaMalloc(&d, t.size() * 2)); CK(cudaMemcpy(d, t.data(), t.size() * 2, cudaMemcpyHostToDevice));
  return made[W2] = d;
}
}  // namespace trs
__device__ __forceinline__ void epi8Sync() { asm volatile("bar.sync 2, 256;" ::: "memory"); }   // (warps 4-11)
__device__ __forceinline__ uint32_t clusterRank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void clusterSync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
// a TMA load multicast to the cluster's CTAs in mask: the data at dst's offset in each, completing on m's offset in each
__device__ __forceinline__ void tma3mc(void* dst, const CUtensorMap* map, int c0, int c1, int c2, uint64_t* m, uint16_t mask) {
#if LF_TC5A
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6;"
               :: "r"(sa(dst)), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(c2), "r"(sa(m)), "h"(mask) : "memory");
#endif
}
// a commit arriving on m's offset in every CTA of the mask
__device__ __forceinline__ void commitMc(uint64_t* m, uint16_t mask) {
#if LF_TC5A
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(sa(m)), "h"(mask) : "memory");
#endif
}
// CL: the cluster's CTAs, which walk the weight chunks in lockstep - each loads 1/CL of every chunk and multicasts it to
// all (the weights from L2 a CL-th as often: one CTA's 384 KB a 256-row block held the kernel at ~2.9 TB/s of L2), and a
// stage is refilled only once every CTA's GEMM2 has read it (wFree counts CL commits, multicast). Partners walk the
// same number of blocks: one past the rows is a dummy (its loads zero, its stores clipped).
template <int CL>
__global__ void __launch_bounds__(448, 1) transTc(const __grid_constant__ CUtensorMap mapX, const __grid_constant__ CUtensorMap mapW1,
    const __grid_constant__ CUtensorMap mapW2, const __grid_constant__ CUtensorMap mapW1s, const __grid_constant__ CUtensorMap mapW2s, __nv_bfloat16* __restrict__ x, const float* __restrict__ lnScale, const float* __restrict__ lnOffset,
    size_t rows) {
#if LF_TC5A
  using namespace trs;
  using L = trs::Lay;
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(sm + L::BAR);
  uint64_t *xFull = bar, *aFull = bar + 1, *aFree = bar + 2, *wFull = bar + 3, *wFree = bar + 5, *d1Full = bar + 7, *d1Free = bar + 8,
           *hFull = bar + 9, *hFree = bar + 10, *d2Full = bar + 11, *d2Free = bar + 12;
  __shared__ uint32_t tmBase;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const size_t blocks = (rows + BM - 1) / BM;
  const unsigned cl0 = blockIdx.x - blockIdx.x % CL;    // (the cluster's first CTA: every partner walks its count)
  const int mine = blocks > cl0 ? (int)((blocks - cl0 + gridDim.x - 1) / gridDim.x) : 0;
  const uint32_t crank = CL > 1 ? clusterRank() : 0;
  const uint16_t cmask = (uint16_t)((1u << CL) - 1);
  if (warp == 12) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    barInit(xFull, 1); barInit(aFull, 4); barInit(aFree, 1);
    for (int i = 0; i < 2; ++i) { barInit(wFull + i, 1); barInit(wFree + i, CL); }
    barInit(d1Full, 1); barInit(d1Free, 8); barInit(hFull, 8); barInit(hFree, 1); barInit(d2Full, 1); barInit(d2Free, 8);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  float* lnS = reinterpret_cast<float*>(sm + L::LN);
  for (int i = tid; i < C; i += 448) { lnS[i] = lnScale[i]; lnS[C + i] = lnOffset[i]; }
  fenceBefore();
  __syncthreads();
  if constexpr (CL > 1) clusterSync();   // (every partner's barriers initialised before any multicast reaches them)
  fenceAfter();
  const uint32_t tm = tmBase, base = sa(sm);
  auto blockOf = [&](int k) { return (size_t)blockIdx.x + (size_t)k * gridDim.x; };
  if (warp == 13) {
    if (lane == 0) {
      int c = 0;     // (chunks loaded so far: stage c & 1)
      TCP_T(p9);
      for (int k = 0; k < mine; ++k) {
        const int r0 = (int)(blockOf(k) * BM);
        TCP_T(p0); if (k > 0) barWait(aFree, (k - 1) & 1); TCP_ADD(warp, 0, p0);
        barExpect(xFull, 65536);
        for (int s = 0; s < 2; ++s)
          for (int at = 0; at < 2; ++at) tma3(sm + L::A + s * 32768 + at * 16384, &mapX, 64 * at, r0 + 128 * s, 0, xFull);
        for (int p = 0; p < NCH; ++p, ++c) {
          const int st = c & 1;
          TCP_T(p1); if (c >= 2) barWait(wFree + st, ((c - 2) >> 1) & 1); TCP_ADD(warp, 1, p1);
          barExpect(wFull + st, L::WSTAGE);
          unsigned char* W = sm + L::W + st * L::WSTAGE;
          if constexpr (CL == 1) {
            tma3(W, &mapW1, 0, 0, p, wFull + st);
            tma3(W + 16384, &mapW1, 64, 0, p, wFull + st);
            tma3(W + 32768, &mapW2, 0, 0, p, wFull + st);
          } else {
            // the chunk's three 16 KB pieces (W1's two K atoms, W2) in 128/CL-row slices, CTA r its slices of each
            constexpr int SR = 128 / CL;
            tma3mc(W + crank * SR * 128, &mapW1s, 0, crank * SR, p, wFull + st, cmask);
            tma3mc(W + 16384 + crank * SR * 128, &mapW1s, 64, crank * SR, p, wFull + st, cmask);
            tma3mc(W + 32768 + crank * SR * 128, &mapW2s, 0, crank * SR, p, wFull + st, cmask);
          }
        }
      }
      TCP_ADD(warp, 7, p9);
    }
  } else if (warp < 4) {
        QuadLn ln; ln.init(lnS, lane);
    TCP_T(l9);
    for (int k = 0; k < mine; ++k) {
      TCP_T(l0); barWait(xFull, k & 1); if (lane == 0) TCP_ADD(warp, 0, l0);
      for (int s2 = 0; s2 < 2; ++s2) ln.run(sm + L::A + s2 * 32768, warp, 4, lane);
      fenceProxy();
      __syncwarp();
      if (lane == 0) barArrive(aFull);
    }
    if (lane == 0) TCP_ADD(warp, 7, l9);
  } else if (warp == 12) {
    if (lane == 0) {
      constexpr uint32_t ID128 = idesc(TM, 128);
      TCP_T(m9);
      int c = 0;     // chunks consumed so far (global, as the producer counts)
      for (int k = 0; k < mine; ++k) {
        TCP_T(m4); barWait(aFull, k & 1); TCP_ADD(warp, 4, m4);
        fenceAfter();
        // GEMM1 of chunk p + 1 as soon as SwiGLU has chunk p's D1 in registers, then GEMM2 of chunk p (once its h is
        // written): the tensor core runs the next GEMM1 under the SwiGLU - issued after GEMM2 (which waits on h), it sat
        // idle through every SwiGLU (1.65x fusedTransitionK where ~6x is the MMA bound)
        auto gemm1 = [&](int p) {
          const int cc = c + p, st = cc & 1;
          TCP_T(m0); barWait(wFull + st, (cc >> 1) & 1); TCP_ADD(warp, 0, m0);
          TCP_T(m1); if (cc > 0) barWait(d1Free, (cc - 1) & 1); TCP_ADD(warp, 1, m1);
          fenceAfter();
          const uint32_t W = base + L::W + st * L::WSTAGE;
          for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int q = 0; q < 8; ++q)
              mmaF16(tm + 128 * s, sdesc(base + L::A + s * 32768 + (q >> 2) * 16384 + (q & 3) * 32, 16, 1024, SW128),
                     sdesc(W + (q >> 2) * 16384 + (q & 3) * 32, 16, 1024, SW128), ID128, q > 0);
          commit(d1Full);
        };
        gemm1(0);
        for (int p = 0; p < NCH; ++p) {
          if (p + 1 < NCH) gemm1(p + 1);
          if (p + 1 == NCH - 1 || NCH == 1) commit(aFree);     // (the block's last GEMM1: A is free for the next load)
          const int cc = c + p, st = cc & 1;     // GEMM2 of chunk p into D2 (cols 256 + 128 s)
          TCP_T(m2); barWait(hFull, cc & 1); TCP_ADD(warp, 2, m2);
          TCP_T(m3); if (p == 0 && k > 0) barWait(d2Free, (k - 1) & 1); TCP_ADD(warp, 3, m3);
          fenceAfter();
          const uint32_t W2 = base + L::W + st * L::WSTAGE + 32768;
          for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int q = 0; q < 4; ++q)
              mmaF16(tm + 256 + 128 * s, sdesc(base + L::H + s * 16384 + q * 32, 16, 1024, SW128),
                     sdesc(W2 + q * 32, 16, 1024, SW128), ID128, p > 0 || q > 0);
          commit(hFree);
          if constexpr (CL == 1) commit(wFree + st); else commitMc(wFree + st, cmask);
        }
        commit(d2Full);
        c += NCH;
      }
      TCP_ADD(warp, 7, m9);
    }
  } else {
    // warps 4-11: sub-tile (w - 4) / 4, tensor-memory lanes 32 (w % 4) ..: a thread a row
    const int s = (warp - 4) >> 2, quarter = warp & 3, rl = quarter * 32 + lane;
    const uint32_t lane0 = (uint32_t)(quarter * 32) << 16;
    unsigned char* Hs = sm + L::H + s * 16384;
    TCP_T(e9);
    int c = 0;
    for (int k = 0; k < mine; ++k) {
      for (int p = 0; p < NCH; ++p, ++c) {
        TCP_T(e0); barWait(d1Full, c & 1); if (lane == 0) TCP_ADD(warp, 0, e0);
        fenceAfter();
        uint32_t av[32], bv[32], av2[32], bv2[32];
        ld32(tm + lane0 + 128 * s, av); ld32(tm + lane0 + 128 * s + 32, av2);
        ld32(tm + lane0 + 128 * s + 64, bv); ld32(tm + lane0 + 128 * s + 96, bv2);
        ldWait();
        fenceBefore();
        __syncwarp();
        if (lane == 0) barArrive(d1Free);
        uint32_t w[32];
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          const float a0 = __uint_as_float(av[2 * e]), a1 = __uint_as_float(av[2 * e + 1]);
          const float c0 = __uint_as_float(av2[2 * e]), c1 = __uint_as_float(av2[2 * e + 1]);
          w[e] = f16x2(a0 * sigmH(a0) * __uint_as_float(bv[2 * e]), a1 * sigmH(a1) * __uint_as_float(bv[2 * e + 1]));
          w[16 + e] = f16x2(c0 * sigmH(c0) * __uint_as_float(bv2[2 * e]), c1 * sigmH(c1) * __uint_as_float(bv2[2 * e + 1]));
        }
        TCP_T(e1); if (c > 0) barWait(hFree, (c - 1) & 1); if (lane == 0) TCP_ADD(warp, 1, e1);     // (GEMM2 of the last chunk has read h)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          *reinterpret_cast<uint4*>(Hs + rl * 128 + ((j ^ (rl & 7)) << 4)) = make_uint4(w[4 * j], w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]);
        fenceProxy();
        __syncwarp();
        if (lane == 0) barArrive(hFull);
      }
      // x + the output: f32, rounded to bf16, staged in the A buffer (every product of the block is done) and TMA stored
      TCP_T(e2); barWait(d2Full, k & 1); if (lane == 0) TCP_ADD(warp, 2, e2);
      fenceAfter();
      const size_t row = blockOf(k) * BM + s * 128 + rl;
      unsigned char* S = sm + L::OUT + s * 16384;
      for (int hh = 0; hh < 2; ++hh) {           // columns 64 hh ..: through OUT, both sub-tiles, then a TMA store
        uint32_t w[2][16];
        for (int h2 = 0; h2 < 2; ++h2) {
          const int h = 2 * hh + h2;
          uint32_t ov[32];
          ld32(tm + lane0 + 256 + 128 * s + 32 * h, ov);
          ldWait();
          uint4 xr[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) xr[j] = row < rows ? *reinterpret_cast<const uint4*>(x + row * C + 32 * h + 8 * j) : make_uint4(0, 0, 0, 0);
#pragma unroll
          for (int e = 0; e < 16; ++e) {
            const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&(&xr[e >> 2].x)[e & 3]));
            const __nv_bfloat162 o = __floats2bfloat162_rn(xv.x + __uint_as_float(ov[2 * e]), xv.y + __uint_as_float(ov[2 * e + 1]));
            w[h2][e] = *reinterpret_cast<const uint32_t*>(&o);
          }
        }
        if (hh == 1) { fenceBefore(); __syncwarp(); if (lane == 0) barArrive(d2Free); }
        if (warp == 4 && lane == 0) bulkWaitRead<0>();   // (OUT's last store read)
        epi8Sync();
#pragma unroll
        for (int j = 0; j < 8; ++j)
          *reinterpret_cast<uint4*>(S + rl * 128 + ((j ^ (rl & 7)) << 4)) = make_uint4(w[j >> 2][4 * (j & 3)], w[j >> 2][4 * (j & 3) + 1], w[j >> 2][4 * (j & 3) + 2], w[j >> 2][4 * (j & 3) + 3]);
        fenceProxy();
        epi8Sync();
        if (warp == 4 && lane == 0) {
          const int r0 = (int)(blockOf(k) * BM);
          for (int ss = 0; ss < 2; ++ss) tmaStore3(&mapX, sm + L::OUT + ss * 16384, 64 * hh, r0 + 128 * ss, 0);
          bulkCommit();
        }
      }
    }
    if (warp == 4 && lane == 0) bulkWait<0>();
    if (lane == 0) TCP_ADD(warp, 7, e9);
  }
  fenceBefore();
  __syncthreads();
  if constexpr (CL > 1) clusterSync();   // (no CTA leaves while a partner may still multicast into it or arrive on it)
  if (warp == 12) {
    fenceAfter();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm));
  }
#endif
}
// fusedTransitionRaw's AF3 form (C 128, I 512, SwiGLU, no biases) for a bf16 pair: true where it ran
inline bool transTcRun(bool pair16, float* x, size_t rows, int Cin, int I, const float* lnScale, const float* lnOffset, const half* W1,
                       const half* W2) {
  if (!PAIR_TC || !ready() || !pair16 || Cin != C || I != trs::I || rows == 0) return false;
  const half *w1 = trs::w1Chunks(W1), *w2 = trs::w2Chunks(W2);
  const CUtensorMap mx = map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, x, { (uint64_t)C, rows, 1 }, { (uint64_t)C * 2, rows * C * 2 }, { 64, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap m1 = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, w1, { (uint64_t)C, 128, trs::NCH }, { (uint64_t)C * 2, (uint64_t)128 * C * 2 }, { 64, 128, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap m2 = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, w2, { (uint64_t)trs::HC, (uint64_t)C, trs::NCH }, { (uint64_t)trs::HC * 2, (uint64_t)C * trs::HC * 2 }, { 64, 128, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  static LF_TLS int sms = 0;
  if (!sms) { int dev; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); }
  auto launch = [&](auto kern, int cl) {
    const CUtensorMap s1 = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, w1, { (uint64_t)C, 128, trs::NCH }, { (uint64_t)C * 2, (uint64_t)128 * C * 2 }, { 64, (uint32_t)(128 / cl), 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap s2 = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, w2, { (uint64_t)trs::HC, (uint64_t)C, trs::NCH }, { (uint64_t)trs::HC * 2, (uint64_t)C * trs::HC * 2 }, { 64, (uint32_t)(128 / cl), 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
    const size_t blocks = (rows + trs::BM - 1) / trs::BM;
    const unsigned grid = (unsigned)(std::min<size_t>((blocks + cl - 1) / cl * cl, sms / cl * cl));
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)trs::Lay::BYTES));
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(grid); cfg.blockDim = dim3(448); cfg.dynamicSmemBytes = trs::Lay::BYTES; cfg.stream = STREAM;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cl; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, kern, mx, m1, m2, s1, s2, reinterpret_cast<__nv_bfloat16*>(x), lnScale, lnOffset, rows));
  };
  // (clusters of 2 and 4 sharing each weight chunk by multicast measured no faster - 0.717 / 1.360 against 0.720 ms at
  // 1M rows - so one CTA a cluster)
  launch(transTc<1>, 1);
  CK(cudaGetLastError());
  return true;
}

// ---- gridOutTc: the grid attention's output projection and residual (gridOutK at C 128, a bf16 pair) on tcgen05:
// pair[pe(q)] += gathered[q] Wout (+ ob), pe the column direction's transpose. 128-row tiles that never cross a grid
// row: TMA loads the gathered tile (f16, straight into the MMA's A layout) and the pair tile (the 3-d map gridInTc
// uses, transposed strides in the column direction); Wout K-major in shared memory for the block's life; one N=128
// product a tile into a double-buffered tensor-memory D; warps 2-5 add D, ob and the pair in f32, round to bf16 in
// place in the pair tile and TMA store it back.
namespace gout { struct Lay { static constexpr uint32_t A = 0, P = 65536, W = 131072, OB = W + 32768, BAR = OB + 512, BYTES = BAR + 256; }; }
__global__ void __launch_bounds__(192, 1) gridOutTc(const __grid_constant__ CUtensorMap mapG, const __grid_constant__ CUtensorMap mapP,
    const half* __restrict__ WoT, const float* __restrict__ ob, int n, int a0, int A) {
#if LF_TC5A
  using L = gout::Lay;
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(sm + L::BAR);
  uint64_t *ldFull = bar, *bufFree = bar + 2, *dFull = bar + 4, *dFree = bar + 6;
  __shared__ uint32_t tmBase;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nbt = (n + TM - 1) / TM;
  const size_t tiles = (size_t)A * nbt;
  const int mine = tiles > blockIdx.x ? (int)((tiles - blockIdx.x + gridDim.x - 1) / gridDim.x) : 0;
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" :: "r"(sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { barInit(ldFull + i, 1); barInit(bufFree + i, 1); barInit(dFull + i, 1); barInit(dFree + i, 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < C * 16; i += 192) {
    const int r = i >> 4, c = i & 15;
    *reinterpret_cast<uint4*>(sm + L::W + swOff(r, c, C)) = *reinterpret_cast<const uint4*>(WoT + (size_t)r * C + c * 8);
  }
  float* obS = reinterpret_cast<float*>(sm + L::OB);
  for (int i = tid; i < C; i += 192) obS[i] = ob ? ob[i] : 0.f;
  fenceProxy();
  fenceBefore();
  __syncthreads();
  fenceAfter();
  const uint32_t tm = tmBase, base = sa(sm);
  auto tileOf = [&](int k, int& a, int& b0) { const size_t t = (size_t)blockIdx.x + (size_t)k * gridDim.x; a = (int)(t / nbt); b0 = (int)(t % nbt) * TM; };
  if (warp == 0) {
    if (lane == 0) {
      for (int k = 0; k < mine; ++k) {
        int a, b0; tileOf(k, a, b0);
        const int b = k & 1;
        if (k >= 2) barWait(bufFree + b, ((k - 2) >> 1) & 1);
        barExpect(ldFull + b, 65536);
        unsigned char* Ab = sm + L::A + b * 32768; unsigned char* Pb = sm + L::P + b * 32768;
        tma3(Ab, &mapG, 0, b0, a, ldFull + b); tma3(Ab + 16384, &mapG, 64, b0, a, ldFull + b);
        tma3(Pb, &mapP, 0, b0, a0 + a, ldFull + b); tma3(Pb + 16384, &mapP, 64, b0, a0 + a, ldFull + b);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      constexpr uint32_t ID128 = idesc(TM, 128);
      for (int k = 0; k < mine; ++k) {
        const int b = k & 1;
        barWait(ldFull + b, (k >> 1) & 1);
        if (k >= 2) barWait(dFree + b, ((k - 2) >> 1) & 1);
        fenceAfter();
        const uint32_t Ab = base + L::A + b * 32768;
#pragma unroll
        for (int s = 0; s < 8; ++s)
          mmaF16(tm + 128 * b, sdesc(Ab + (s >> 2) * TM * 128 + (s & 3) * 32, 16, 1024, SW128),
                 sdesc(base + L::W + (s >> 2) * C * 128 + (s & 3) * 32, 16, 1024, SW128), ID128, s > 0);
        commit(dFull + b);
      }
    }
  } else {
    const int sub = warp & 3, rl = sub * 32 + lane;
    const uint32_t lane0 = (uint32_t)(sub * 32) << 16;
    const bool T0 = warp == 2 && lane == 0;
    for (int k = 0; k < mine; ++k) {
      int a, b0; tileOf(k, a, b0);
      const int b = k & 1;
      barWait(dFull + b, (k >> 1) & 1);
      fenceAfter();
      unsigned char* Pb = sm + L::P + b * 32768;
      for (int c = 0; c < 4; ++c) {
        uint32_t x[32];
        ld32(tm + lane0 + 128 * b + 32 * c, x);
        ldWait();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint4* pp = reinterpret_cast<uint4*>(Pb + swOff(rl, 4 * c + j, TM));
          uint4 w = *pp;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = 8 * j + 2 * e, ch = 32 * c + i;
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&(&w.x)[e]));
            const __nv_bfloat162 o = __floats2bfloat162_rn(f.x + __uint_as_float(x[i]) + obS[ch], f.y + __uint_as_float(x[i + 1]) + obS[ch + 1]);
            (&w.x)[e] = *reinterpret_cast<const uint32_t*>(&o);
          }
          *pp = w;
        }
      }
      fenceBefore();
      __syncwarp();
      if (lane == 0) barArrive(dFree + b);
      fenceProxy();
      epiSync();
      if (T0) {
        tmaStore3(&mapP, Pb, 0, b0, a0 + a); tmaStore3(&mapP, Pb + 16384, 64, b0, a0 + a);
        bulkCommit();
        bulkWaitRead<0>();
        barArrive(bufFree + b);
      }
    }
    if (T0) bulkWait<0>();
  }
  fenceBefore();
  __syncthreads();
  if (warp == 1) {
    fenceAfter();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tm));
  }
#endif
}
// gridOutRaw's form at C 128 for a bf16 pair, calls covering whole grid rows: true where it ran
inline bool gridOutTcRun(bool pair16, const half* gathered, const half* Wout, const float* ob, float* pair, int n, size_t q0, size_t rows, bool tr) {
  if (!PAIR_TC || !ready() || !pair16 || q0 % n || rows % n || rows == 0) return false;
  const half* WoT = kMajor(Wout, C, C);
  const uint64_t N = (uint64_t)n, A = rows / n;
  const CUtensorMap mg = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, gathered, { (uint64_t)C, N, A }, { (uint64_t)C * 2, N * C * 2 }, { 64, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap mp = map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, pair, { (uint64_t)C, N, N },
                              { tr ? N * C * 2 : (uint64_t)C * 2, tr ? (uint64_t)C * 2 : N * C * 2 }, { 64, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  static LF_TLS int sms = 0;
  if (!sms) { int dev; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); }
  const unsigned grid = (unsigned)std::min<size_t>(A * ((n + TM - 1) / TM), sms);
  static LF_TLS bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(gridOutTc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gout::Lay::BYTES)); attr = true; }
  gridOutTc<<<grid, 192, gout::Lay::BYTES, STREAM>>>(mg, mp, WoT, ob, n, (int)(q0 / n), (int)A);
  CK(cudaGetLastError());
  return true;
}

// ---- triOutTc: the triangle multiplication's output side (triangleOutK's VEC form at C 128: a bf16 product, a bf16 pair)
// on tcgen05: pair[i L + j] += (LN_centre(prod[:, q]) Wout (+ ob)) * sigmoid(t2[q]), q = i Lp + j the padded row, for
// i, j < L. 128-row tiles (i, j0 ..) of real pairs: TMA loads the product tile channel-major ([ch][128 rows], no swizzle),
// t2's rows and the pair's rows (both 128B-swizzled); warps 2-5 take each row's centre-norm statistics from its column
// (a thread a row, its 128 values held packed), then - after a barrier among them - write the normalised f16 row back
// into the same buffer in the MMA's K-major layout (an in-place transpose); one N=128 product a tile into a double-
// buffered D; warps 6-9 add D, ob, the gate and the pair in f32, round to bf16 in place in the pair tile, TMA store it.
namespace tout {
// shared memory: [2 stages] { the product tile, then the normalised A 32 KB; t2 32 KB; the pair tile 32 KB }; Wout 32 KB
struct Lay { static constexpr uint32_t STG = 98304, A = 0, T2 = 32768, PR = 65536, W = 2 * STG, OB = W + 32768, LN = OB + 512, BAR = LN + 1024, BYTES = BAR + 256; };
}
__device__ __forceinline__ void lnSync() { asm volatile("bar.sync 3, 128;" ::: "memory"); }   // (warps 2-5)
__global__ void __launch_bounds__(320, 1) triOutTc(const __grid_constant__ CUtensorMap mapProd, const __grid_constant__ CUtensorMap mapT2,
    const __grid_constant__ CUtensorMap mapP, const float* __restrict__ cnScale, const float* __restrict__ cnOffset, const half* __restrict__ WoT,
    const float* __restrict__ ob, int L, int Lp) {
#if LF_TC5A
  using T = tout::Lay;
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(sm + T::BAR);
  uint64_t *ldFull = bar, *aFull = bar + 2, *bufFree = bar + 4, *dFull = bar + 6, *dFree = bar + 8;
  __shared__ uint32_t tmBase;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nbt = (L + TM - 1) / TM;
  const size_t tiles = (size_t)L * nbt;
  const int mine = tiles > blockIdx.x ? (int)((tiles - blockIdx.x + gridDim.x - 1) / gridDim.x) : 0;
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" :: "r"(sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { barInit(ldFull + i, 1); barInit(aFull + i, 4); barInit(bufFree + i, 1); barInit(dFull + i, 1); barInit(dFree + i, 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < C * 16; i += 320) {
    const int r = i >> 4, c = i & 15;
    *reinterpret_cast<uint4*>(sm + T::W + swOff(r, c, C)) = *reinterpret_cast<const uint4*>(WoT + (size_t)r * C + c * 8);
  }
  float* obS = reinterpret_cast<float*>(sm + T::OB);
  float* lnS = reinterpret_cast<float*>(sm + T::LN);
  for (int i = tid; i < C; i += 320) { obS[i] = ob ? ob[i] : 0.f; lnS[i] = cnScale[i]; lnS[C + i] = cnOffset[i]; }
  fenceProxy();
  fenceBefore();
  __syncthreads();
  fenceAfter();
  const uint32_t tm = tmBase, base = sa(sm);
  auto tileOf = [&](int k, int& i, int& j0) { const size_t t = (size_t)blockIdx.x + (size_t)k * gridDim.x; i = (int)(t / nbt); j0 = (int)(t % nbt) * TM; };
  if (warp == 0) {
    if (lane == 0) {
      for (int k = 0; k < mine; ++k) {
        int i, j0; tileOf(k, i, j0);
        const int b = k & 1;
        if (k >= 2) barWait(bufFree + b, ((k - 2) >> 1) & 1);
        barExpect(ldFull + b, 3 * 32768);
        unsigned char* S = sm + b * T::STG;
        tma3(S + T::A, &mapProd, j0, i, 0, ldFull + b);                                   // [128 ch][128 rows]
        tma3(S + T::T2, &mapT2, 0, j0, i, ldFull + b); tma3(S + T::T2 + 16384, &mapT2, 64, j0, i, ldFull + b);
        tma3(S + T::PR, &mapP, 0, j0, i, ldFull + b); tma3(S + T::PR + 16384, &mapP, 64, j0, i, ldFull + b);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      constexpr uint32_t ID128 = idesc(TM, 128);
      for (int k = 0; k < mine; ++k) {
        const int b = k & 1;
        barWait(aFull + b, (k >> 1) & 1);
        if (k >= 2) barWait(dFree + b, ((k - 2) >> 1) & 1);
        fenceAfter();
        const uint32_t A = base + b * T::STG + T::A;
#pragma unroll
        for (int s = 0; s < 8; ++s)
          mmaF16(tm + 128 * b, sdesc(A + (s >> 2) * TM * 128 + (s & 3) * 32, 16, 1024, SW128),
                 sdesc(base + T::W + (s >> 2) * C * 128 + (s & 3) * 32, 16, 1024, SW128), ID128, s > 0);
        commit(dFull + b);
      }
    }
  } else if (warp < 6) {
    // the centre norm, a thread a row: its column of 128 channels (two bytes each, a warp 64 contiguous bytes) held
    // packed, the statistics, then - once every LN thread holds its row - the normalised row written back over the tile
    const int r = (warp - 2) * 32 + lane;
    for (int k = 0; k < mine; ++k) {
      const int b = k & 1;
      barWait(ldFull + b, (k >> 1) & 1);
      unsigned char* S = sm + b * T::STG + T::A;
      uint32_t v[64];
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int c = 0; c < C; c += 2) {
        const unsigned short lo = *reinterpret_cast<const unsigned short*>(S + c * 256 + r * 2),
                             hi = *reinterpret_cast<const unsigned short*>(S + (c + 1) * 256 + r * 2);
        v[c >> 1] = lo | (uint32_t)hi << 16;
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v[c >> 1]));
        s0 += f.x; s1 += f.y;
      }
      const float mean = (s0 + s1) / C;
      float q0 = 0.f, q1 = 0.f;
#pragma unroll
      for (int c = 0; c < 64; ++c) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v[c]));
        q0 += (f.x - mean) * (f.x - mean); q1 += (f.y - mean) * (f.y - mean);
      }
      const float inv = rsqrtf((q0 + q1) / C + 1e-5f);
      lnSync();          // (every LN thread has its column: the tile may be overwritten)
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        uint4 w;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v[4 * c + e]));
          const int ch = 8 * c + 2 * e;
          (&w.x)[e] = f16x2((f.x - mean) * inv * lnS[ch] + lnS[C + ch], (f.y - mean) * inv * lnS[ch + 1] + lnS[C + ch + 1]);
        }
        *reinterpret_cast<uint4*>(S + swOff(r, c, TM)) = w;
      }
      fenceProxy();
      __syncwarp();
      if (lane == 0) barArrive(aFull + b);
    }
  } else {
    const int sub = warp & 3, rl = sub * 32 + lane;
    const uint32_t lane0 = (uint32_t)(sub * 32) << 16;
    const bool T0 = warp == 6 && lane == 0;
    for (int k = 0; k < mine; ++k) {
      int i, j0; tileOf(k, i, j0);
      const int b = k & 1;
      barWait(dFull + b, (k >> 1) & 1);
      fenceAfter();
      unsigned char* S = sm + b * T::STG;
      for (int c = 0; c < 4; ++c) {
        uint32_t x[32];
        ld32(tm + lane0 + 128 * b + 32 * c, x);
        ldWait();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint4* pp = reinterpret_cast<uint4*>(S + T::PR + swOff(rl, 4 * c + j, TM));
          const uint4 gw = *reinterpret_cast<const uint4*>(S + T::T2 + swOff(rl, 4 * c + j, TM));
          uint4 w = *pp;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int ii = 8 * j + 2 * e, ch = 32 * c + ii;
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&(&w.x)[e]));
            const float2 g = __half22float2(*reinterpret_cast<const half2*>(&(&gw.x)[e]));
            const __nv_bfloat162 o = __floats2bfloat162_rn(f.x + (__uint_as_float(x[ii]) + obS[ch]) * sigmH(g.x),
                                                           f.y + (__uint_as_float(x[ii + 1]) + obS[ch + 1]) * sigmH(g.y));
            (&w.x)[e] = *reinterpret_cast<const uint32_t*>(&o);
          }
          *pp = w;
        }
      }
      fenceBefore();
      __syncwarp();
      if (lane == 0) barArrive(dFree + b);
      fenceProxy();
      epiSync();
      if (T0) {
        tmaStore3(&mapP, S + T::PR, 0, j0, i); tmaStore3(&mapP, S + T::PR + 16384, 64, j0, i);
        bulkCommit();
        bulkWaitRead<0>();
        barArrive(bufFree + b);
      }
    }
    if (T0) bulkWait<0>();
  }
  fenceBefore();
  __syncthreads();
  if (warp == 1) {
    fenceAfter();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tm));
  }
#endif
}
// triangleOutRun's VEC form at C 128 (a bf16 product and pair, the whole plane): true where it ran
inline bool triOutTcRun(bool pair16, const __nv_bfloat16* prod, const float* cnScale, const float* cnOffset, const half* Wout, const half* t2,
                        float* pair, int L, int Lp, const float* ob) {
  if (!PAIR_TC || !ready() || !pair16) return false;
  const half* WoT = kMajor(Wout, C, C);
  const uint64_t NL = (uint64_t)L, NP = (uint64_t)Lp;
  const CUtensorMap mprod = map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, prod, { NP, NP, (uint64_t)C }, { NP * 2, NP * NP * 2 }, { 128, 1, 128 }, CU_TENSOR_MAP_SWIZZLE_NONE);
  const CUtensorMap mt2 = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, t2, { (uint64_t)C, NP, NP }, { (uint64_t)C * 2, NP * C * 2 }, { 64, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap mp = map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, pair, { (uint64_t)C, NL, NL }, { (uint64_t)C * 2, NL * C * 2 }, { 64, TM, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  static LF_TLS int sms = 0;
  if (!sms) { int dev; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); }
  const unsigned grid = (unsigned)std::min<size_t>((size_t)L * ((L + TM - 1) / TM), sms);
  static LF_TLS bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(triOutTc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tout::Lay::BYTES)); attr = true; }
  triOutTc<<<grid, 320, tout::Lay::BYTES, STREAM>>>(mprod, mt2, mp, cnScale, cnOffset, WoT, ob, L, Lp);
  CK(cudaGetLastError());
  return true;
}
}  // namespace ptc
