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
  static std::map<const half*, half*> made;
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
}  // namespace ptc
