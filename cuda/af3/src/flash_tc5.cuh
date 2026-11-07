// flash8's attention on the B200's tcgen05 tensor cores (sm_100a): the grid/triangle attention a fold spends most of its
// trunk in, with flash8's contract - q and the gate f16 out of qkvg, k and v e4m3 (kv8, v transposed by fp8VtK), the
// pair bias f16 (already times log2 e), P e4m3 - and its precision: the same e4m3 operands, P in e4m3, the row sum by
// the MMA from P as rounded. On sm_100 flash8's mma.sync is the legacy path and its issue-bound softmax runs at half
// the RTX PRO 6000's rate (122 against 234 TFLOP/s at 2,048 tokens); here the products cost the softmax no issue at all.
//
// A block: one head, 128 queries, R grid rows sharing the bias tile. Warps 0..4R-1 the R rows' softmaxes (a thread a
// query: tensor memory's lane), warp 4R the products (one thread), warp 4R+1 the loads (TMA, one thread). Per tile of
// 64 keys and row r: S = bias + Q K^T into tensor memory - the bias ADDED BY THE MMA (an identity B over the bias tile
// as A, kind::f16), so the softmax never loads it - then the softmax reads S, P = 2^(s - ref) in e4m3 to shared memory,
// O += P V and the row sum += P 1 (an N=16 product against a ones operand: column 0 the sum). S and P double buffered;
// O rescaled lazily (FA4): ref = max - OFF, moved only where a tile's max passes the old by LAZY, P <= 2^(OFF+LAZY).
#pragma once
#include <array>
#include <cstdint>
#include <cuda.h>      // CUtensorMap (the encoder is reached through the runtime: no -lcuda)

namespace tc5a {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
#define LF_TC5A 1
#else
#define LF_TC5A 0
#endif
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// instruction descriptors: dtype f32 (bit 4), atype/btype 0 (kind::f16: f16; kind::f8f6f4: e4m3), both K-major
__host__ __device__ constexpr uint32_t idesc(int M, int N) { return (1u << 4) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
// a shared-memory operand without swizzle: lbo the bytes between core matrices along K, sbo along M/N
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t layout = 0) {
  return ((uint64_t)(addr >> 4) & 0x3fff) | ((uint64_t)((lbo >> 4) & 0x3fff) << 16) | ((uint64_t)((sbo >> 4) & 0x3fff) << 32) | (1ull << 46) |
         ((uint64_t)layout << 61);
}
// a K-major operand as TMA swizzles it (rows of 128, 64 or 32 bytes, 8-row groups sbo apart): the layout type, and a
// K step inside the swizzle atom is a byte offset added to the start (the atom's pattern follows the address)
constexpr uint32_t SW128 = 2, SW64 = 4, SW32 = 6;
template <int E = 0> __device__ __forceinline__ void mmaF16(uint32_t d, uint64_t a, uint64_t b, uint32_t id, bool acc) {
#if LF_TC5A
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"((uint32_t)acc));
#endif
}
__device__ __forceinline__ void mmaF8(uint32_t d, uint64_t a, uint64_t b, uint32_t id, bool acc) {
#if LF_TC5A
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"((uint32_t)acc));
#endif
}
// A from tensor memory (a: its address - lane the row, 4 e4m3 a column)
__device__ __forceinline__ void mmaF8ts(uint32_t d, uint32_t a, uint64_t b, uint32_t id, bool acc) {
#if LF_TC5A
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n\t}\n"
               :: "r"(d), "r"(a), "l"(b), "r"(id), "r"((uint32_t)acc));
#endif
}
__device__ __forceinline__ void mmaF16ts(uint32_t d, uint32_t a, uint64_t b, uint32_t id, bool acc) {
#if LF_TC5A
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n"
               :: "r"(d), "r"(a), "l"(b), "r"(id), "r"((uint32_t)acc));
#endif
}
__device__ __forceinline__ void commit(uint64_t* m) {
#if LF_TC5A
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(sa(m)) : "memory");
#endif
}
__device__ __forceinline__ void barInit(uint64_t* m, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(m)), "r"(n) : "memory"); }
__device__ __forceinline__ void barArrive(uint64_t* m) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(m)) : "memory"); }
__device__ __forceinline__ void barExpect(uint64_t* m, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(m)), "r"(bytes) : "memory");
}
#ifndef TC_SUSPEND
#define TC_SUSPEND 0x989680
#endif
// (with a suspend-time hint: a waiting warp parks in the barrier unit instead of spinning on try_wait, whose loops took
// issue slots from the warps doing the work)
__device__ __forceinline__ void barWait(uint64_t* m, uint32_t parity) {
#if TC_SUSPEND
  asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t@!p bra W;\n\t}\n"
               :: "r"(sa(m)), "r"(parity), "r"((uint32_t)TC_SUSPEND) : "memory");
#else
  asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra W;\n\t}\n"
               :: "r"(sa(m)), "r"(parity) : "memory");
#endif
}
__device__ __forceinline__ void fenceBefore() {
#if LF_TC5A
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#endif
}
__device__ __forceinline__ void fenceAfter() {
#if LF_TC5A
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}
__device__ __forceinline__ void fenceProxy() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// a 4-d TMA tile load into shared memory, completing on m
__device__ __forceinline__ void tma3(void* dst, const CUtensorMap* map, int c0, int c1, int c2, uint64_t* m) {
#if LF_TC5A
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(sa(dst)), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(c2), "r"(sa(m)) : "memory");
#endif
}
__device__ __forceinline__ void tma4(void* dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, uint64_t* m) {
#if LF_TC5A
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
               :: "r"(sa(dst)), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(sa(m)) : "memory");
#endif
}
#define LF_TC5A_R32(r) "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), \
  "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), \
  "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), \
  "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
// 32 consecutive columns of this warp's 32 lanes (no wait: the caller waits once for several)
__device__ __forceinline__ void ld32(uint32_t addr, uint32_t* r) {
#if LF_TC5A
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" : LF_TC5A_R32(r) : "r"(addr));
#endif
}
__device__ __forceinline__ void ld16(uint32_t addr, uint32_t* r) {
#if LF_TC5A
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(addr));
#endif
}
__device__ __forceinline__ void ld1(uint32_t addr, uint32_t& r) {
#if LF_TC5A
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r) : "r"(addr));
#endif
}
__device__ __forceinline__ void ldWait() {
#if LF_TC5A
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}
__device__ __forceinline__ void st32(uint32_t addr, const uint32_t* r) {
#if LF_TC5A
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
               "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
                  "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]),
                  "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]) : "memory");
#endif
}
__device__ __forceinline__ void st16(uint32_t addr, const uint32_t* r) {
#if LF_TC5A
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
#endif
}
__device__ __forceinline__ void st1(uint32_t addr, uint32_t r) {
#if LF_TC5A
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(addr), "r"(r) : "memory");
#endif
}
__device__ __forceinline__ void stWait() {
#if LF_TC5A
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#endif
}
__device__ __forceinline__ uint32_t f16x2(float lo, float hi) {
  uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float2 sub2(float2 a, float b) {     // FADD2 (sm_100)
  uint64_t r, x = (uint64_t)__float_as_uint(a.y) << 32 | __float_as_uint(a.x), y = (uint64_t)__float_as_uint(-b) << 32 | __float_as_uint(-b);
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(x), "l"(y));
  return make_float2(__uint_as_float((uint32_t)r), __uint_as_float((uint32_t)(r >> 32)));
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  uint64_t r, x = (uint64_t)__float_as_uint(a.y) << 32 | __float_as_uint(a.x), y = (uint64_t)__float_as_uint(b.y) << 32 | __float_as_uint(b.x);
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(x), "l"(y));
  return make_float2(__uint_as_float((uint32_t)r), __uint_as_float((uint32_t)(r >> 32)));
}
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {    // FFMA2 (sm_100)
  uint64_t r, x = (uint64_t)__float_as_uint(a.y) << 32 | __float_as_uint(a.x), y = (uint64_t)__float_as_uint(b.y) << 32 | __float_as_uint(b.x),
              z = (uint64_t)__float_as_uint(c.y) << 32 | __float_as_uint(c.x);
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(x), "l"(y), "l"(z));
  return make_float2(__uint_as_float((uint32_t)r), __uint_as_float((uint32_t)(r >> 32)));
}
// 2^d for a pair on the FMA pipe (FA4's trick, to share the exponentials with the MUFU unit, 16 a clock an SM): d rounded
// to n + f (f in [-1/2, 1/2]) by the 1.5 2^23 shift, 2^f a degree-2 minimax (relative error 1.7e-3 - P is e4m3, whose step
// is 2^-4), n added into the exponent by one IMAD of the shifted value's bits (its low bits are n, the rest shift out)
__device__ __forceinline__ float2 ex2emu2(float2 d) {
  d.x = fmaxf(d.x, -40.f); d.y = fmaxf(d.y, -40.f);
  const float2 t = add2(d, make_float2(12582912.f, 12582912.f));
  const float2 f = fma2(add2(t, make_float2(-12582912.f, -12582912.f)), make_float2(-1.f, -1.f), d);
  const float2 p = fma2(fma2(f, make_float2(0.2384158f, 0.2384158f), make_float2(0.70342298f, 0.70342298f)), f, make_float2(1.00044213f, 1.00044213f));
  return make_float2(__int_as_float(__float_as_int(t.x) * 0x800000 + __float_as_int(p.x)),
                     __int_as_float(__float_as_int(t.y) * 0x800000 + __float_as_int(p.y)));
}
__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// two f32 to an e4m3 pair (lo the low byte)
__device__ __forceinline__ unsigned short e4m3x2(float lo, float hi) {
  unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}
// the position fp8VtK gives key lk of a 64-key group
__host__ __device__ constexpr int vtKeyPos(int lk) {
  return (lk & 32) + (lk & 16) + ((lk & 15) < 8 ? ((lk & 7) >> 1) * 4 + (lk & 1) : ((lk & 7) >> 1) * 4 + 2 + (lk & 1));
}
// which key fp8VtK put at position p of a 64-key group (its V columns run in mma.sync's fragment order; P follows)
__host__ __device__ constexpr int vtKeyAt(int p) {
  for (int lk = 0; lk < 64; ++lk) {
    int g = lk & 31, q = (lk & 32) + (g & 16) + ((g & 15) < 8 ? ((g & 7) >> 1) * 4 + (g & 1) : ((g & 7) >> 1) * 4 + 2 + (g & 1));
    if (q == p) return lk;
  }
  return -1;
}
#ifndef TC_EMU_PAIR
#define TC_EMU_PAIR(c) false   // (pairs whose exponentials go to the FMA pipe: none - measured slower, the softmax being latency-bound)
#endif
constexpr int TQ = 128, TK = 64, OFF = 5;
constexpr float LAZY = 3.f;   // (P <= 2^(OFF + LAZY) = 256, inside e4m3's 448)
#ifdef TC_PROF
__device__ unsigned long long tcProf[16][8];   // [warp][slot] summed over blocks
#define TCP_T(v) long long v = clock64()
#define TCP_ADD(w, slot, v0) atomicAdd(&tcProf[w][slot], (unsigned long long)(clock64() - (v0)))
#else
#define TCP_T(v)
#define TCP_ADD(w, slot, v0)
#endif
// per R (rows a block): TS load stages, SB S buffers. Tensor memory, columns: S [b][r] 64 each, O [r] 64 (48 used: O and
// the sum), P [r][2] 16 each (64 e4m3 a lane) - SB 64R + 64R + 32R within 512: R 2 with SB 2, R 3 with SB 1
template <int R> struct Cfg { static constexpr int SB = R >= 3 ? 1 : 2, TS = R >= 3 ? 5 : 7; };
// shared memory, bytes (every region 1024-aligned): Q [2][R] 4 KB; TS stages of { bias 16 KB, [R] { K 2 KB, V 2 KB, then
// 1 KB of ones rows - V's rows 32..47, row 32 all ones: one N=48 product gives O and the row sum } }; the identity B of
// the bias product (f16, N 64R x K 64: R identities stacked, core (n/8, k/8) at (k/8) 1024R + (n/8) 128) 8R KB; the last
// tile's key mask (A: f16 128 x 16, every row (1, 0, ...) - one core matrix read for every row group, sbo 0 - and B: f16
// 64 x 16, key k's first column 0 where k is a key, -30000 past n) 1 + 2 KB; the barriers
template <int R> struct Lay {
  static constexpr uint32_t Q = 0, STAGE = 16384 + R * 5120, ST = R * 8192, ID = ST + Cfg<R>::TS * STAGE, MA = ID + R * 8192,
                            MB = MA + 1024, BAR = MB + 2048, BYTES = BAR + 512;
};
template <int R> __host__ __device__ constexpr int tcWarps() { return 4 * R + 2; }   // 4R softmax, the products, the loads

// PERSISTENT: a block an SM, each walking its share of the work items (head, query tile, row group) with its pipeline
// running across them - a block a work item paid ~11 us a block in setup, fill and drain that nothing overlapped (one
// block fills an SM). Q double buffered: the softmax threads convert the next item's Q as an item starts (its S products
// wait on qFull), and every barrier phase counts tiles across items.
// THE BIAS IS ADDED BY THE MMA, ONCE FOR ALL R ROWS: a buffer's R S blocks are adjacent columns, so four K16 products at
// N = 64R (A the bias tile, B the stacked identity) lay the bias in every row's S, and each row's Q K^T accumulates on
// top. P GOES TO TENSOR MEMORY (tcgen05.st) and the PV product reads it there as A: no shared-memory P, so no proxy fence
// (MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC on every tile of every warp).
template <int R>
__global__ void __launch_bounds__(32 * tcWarps<R>(), 1) flashTc(const __grid_constant__ CUtensorMap mapB, const __grid_constant__ CUtensorMap mapK,
    const __grid_constant__ CUtensorMap mapV, const half* __restrict__ qkvg, half* __restrict__ out, int n, int heads, int rows,
    float scale, const float* __restrict__ qBias, int G) {
#if LF_TC5A
  using L = Lay<R>;
  constexpr int D = 32, MW = 4 * R, LW = 4 * R + 1, TS = Cfg<R>::TS, SB = Cfg<R>::SB;
  constexpr uint32_t OCOL = 64 * R * SB, PCOL = OCOL + 64 * R;
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* kvFull = reinterpret_cast<uint64_t*>(sm + L::BAR);
  uint64_t *kvFree = kvFull + TS, *sFull = kvFree + TS, *sRead = sFull + R * SB, *pFull = sRead + R * SB, *pFree = pFull + 2 * R,   // [b R + r], [r 2 + b]
           *qFull = pFree + 2 * R;                                                                                                // [item & 1]
  __shared__ uint32_t tmBase;
  const int W = heads * D, nqt = (n + TQ - 1) / TQ, nrg = (rows + R - 1) / R, nk = (n + TK - 1) / TK;
  const unsigned per = (unsigned)nqt * nrg, items = per * heads;
  const int mine = items > blockIdx.x ? (int)((items - blockIdx.x + gridDim.x - 1) / gridDim.x) : 0;
  // item k of this block: a head's items walk G row groups across each query tile, so the resident blocks share bias tiles
  auto decode = [&](int k, int& h, int& q0, int& rg) {
    const unsigned w = blockIdx.x + (unsigned)k * gridDim.x, r0 = w % per;
    h = (int)(w / per);
    const unsigned group = r0 / (G * nqt), within = r0 % (G * nqt), left = nrg - group * G, gsz = left < (unsigned)G ? left : G;
    rg = (int)(group * G + within % gsz); q0 = (int)(within / gsz) * TQ;
  };
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = sa(sm);
  if (warp == MW) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    for (int s = 0; s < TS; ++s) { barInit(kvFull + s, 1); barInit(kvFree + s, 1); }
    for (int i = 0; i < R * SB; ++i) { barInit(sFull + i, 1); barInit(sRead + i, 4); }
    for (int i = 0; i < 2 * R; ++i) { barInit(pFull + i, 4); barInit(pFree + i, 1); }
    for (int i = 0; i < 2; ++i) barInit(qFull + i, 4 * R);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // the ones rows after each stage's V (64-byte swizzled rows: row 0 of the group all ones, unmoved by the swizzle)
  for (int i = tid; i < TS * R * 64; i += blockDim.x) {
    const int sr = i >> 6, u = i & 63;
    *reinterpret_cast<uint4*>(sm + L::ST + (sr / R) * L::STAGE + 16384 + (sr % R) * 5120 + 4096 + u * 16) =
        u < 4 ? make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u) : make_uint4(0, 0, 0, 0);
  }
  // the stacked identity: row n of B is e_(n mod 64)
  for (int i = tid; i < R * 8192 / 16; i += blockDim.x) {
    const int kc = i / (R * 64), g = (i / 8) % (R * 8), rr = i & 7, nn = (g * 8 + rr) & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (nn / 8 == kc) { const int k = nn & 7; (&v.x)[k >> 1] = (k & 1) ? 0x3C000000u : 0x3C00u; }
    *reinterpret_cast<uint4*>(sm + L::ID + i * 16) = v;
  }
  // the mask: A's two core matrices (row r: column 0 one), B's (key u at core (u/8, 0): 1024 kc + 128 (u/8) + 16 (u%8))
  for (int i = tid; i < 16; i += blockDim.x)
    *reinterpret_cast<uint4*>(sm + L::MA + i * 16) = make_uint4(i < 8 ? 0x3C00u : 0u, 0, 0, 0);
  for (int i = tid; i < 2048 / 16; i += blockDim.x) {
    const int kc = i >> 6, u = i & 63, valid = n - (n - 1) / TK * TK;
    *reinterpret_cast<uint4*>(sm + L::MB + i * 16) = make_uint4(kc == 0 && u >= valid ? 0xF753u : 0u, 0, 0, 0);   // (-30000)
  }
  fenceProxy();
  fenceBefore();
  __syncthreads();
  fenceAfter();
  const uint32_t tm = tmBase;
  const int T = mine * nk;      // this block's tiles, every item's in turn
  if (warp == LW) {
    if (lane == 0) {
      int s = 0, u = 0;    // (stage, and the rounds of the ring so far: the kvFree wait's parity is (u - 1) & 1)
      for (int k = 0; k < mine; ++k) {
        int h, q0, rg; decode(k, h, q0, rg);
        int row[R];
        for (int r = 0; r < R; ++r) row[r] = min(rg * R + r, rows - 1);
        for (int j = 0, t = k * nk; j < nk; ++j, ++t) {
          if (u > 0) barWait(kvFree + s, (u - 1) & 1);
          barExpect(kvFull + s, 16384 + R * 4096);
          unsigned char* st = sm + L::ST + s * L::STAGE;
          tma3(st, &mapB, j * TK, q0, h, kvFull + s);                                    // bias [q 128][64 keys], 128B swizzle
          for (int r = 0; r < R; ++r) {
            tma3(st + 16384 + r * 5120, &mapK, h * D, j * TK, row[r], kvFull + s);        // K [key 64][32 dims], 32B swizzle
            tma3(st + 16384 + r * 5120 + 2048, &mapV, j * TK, h * D, row[r], kvFull + s); // V [dim 32][64 keys], 64B swizzle
          }
          if (++s == TS) { s = 0; ++u; }
        }
      }
    }
  } else if (warp == MW) {
    // the products, one thread: S(t+SB) as soon as every row's softmax has S(t) in registers (sRead) - the bias for all R
    // rows at once, then each row's Q K^T on top - and O += P V for each row once its P(t) is written
    if (lane == 0) {
      constexpr uint32_t IDB = idesc(TQ, TK), IDBR = idesc(TQ, TK * R), IDQK = idesc(TQ, TK), IDPV = idesc(TQ, D + 16);
      const uint64_t dBias = sdesc(base + L::ST, 16, 1024, SW128), dId = sdesc(base + L::ID, R * 1024, 128),
                     dQ = sdesc(base + L::Q, 2048, 128), dK = sdesc(base + L::ST + 16384, 16, 256, SW32),
                     dV = sdesc(base + L::ST + 16384 + 2048, 16, 512, SW64), dMA = sdesc(base + L::MA, 128, 0), dMB = sdesc(base + L::MB, 1024, 128);
      // the S side's own counters (tile ts, its item ks and key tile js, stage ss)
      int ts = 0, ks = 0, js = 0, ss = 0;
      auto S = [&]() {
        const int b = ts % SB;
        TCP_T(a0);
        barWait(kvFull + ss, (ts / TS) & 1);
        TCP_ADD(warp, 0, a0);
        TCP_T(a1);
        if (js == 0) barWait(qFull + (ks & 1), (ks >> 1) & 1);
        TCP_ADD(warp, 1, a1);
        TCP_T(a2);
        if (ts >= SB) for (int r = 0; r < R; ++r) barWait(sRead + b * R + r, ((ts - SB) / SB) & 1);
        TCP_ADD(warp, 2, a2);
        fenceAfter();
        const uint32_t so = ss * L::STAGE >> 4, d = tm + b * R * TK;
#pragma unroll
        for (int c = 0; c < 4; ++c)   // S = bias I for every row: A the bias tile (128B-swizzled rows), B the stacked identity
          mmaF16(d, dBias + so + c * 2, dId + c * (2 * R * 1024 >> 4), IDBR, c > 0);
#pragma unroll
        for (int r = 0; r < R; ++r) {   // S_r += Q_r K_r^T (K 32 bytes: one product)
          mmaF8(d + r * TK, dQ + (((ks & 1) * R + r) * 4096 >> 4), dK + so + (r * 5120 >> 4), IDQK, true);
          if (js == nk - 1 && n % TK) mmaF16(d + r * TK, dMA, dMB, IDB, true);   // keys past n
          commit(sFull + b * R + r);
        }
        ++ts; if (++ss == TS) ss = 0; if (++js == nk) { js = 0; ++ks; }
      };
      TCP_T(a9);
      for (int t = 0; t < T && t < SB; ++t) S();
      int j = 0, s = 0;
      for (int t = 0; t < T; ++t) {
        if (t + SB < T) S();
        const uint32_t so = s * L::STAGE >> 4;
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int i = r * 2 + (t & 1);
          TCP_T(a3);
          barWait(pFull + i, (t >> 1) & 1);
          TCP_ADD(warp, 3, a3);
          fenceAfter();
#pragma unroll
          for (int c = 0; c < 2; ++c)   // [O | sum] += P [V | 1]: P from tensor memory (8 columns a K32 step), V 64B-swizzled rows
            mmaF8ts(tm + OCOL + 64 * r, tm + PCOL + 16 * i + 8 * c, dV + so + ((r * 5120 + c * 32) >> 4), IDPV, j > 0 || c > 0);
          commit(pFree + i);
        }
        commit(kvFree + s);
        if (++s == TS) s = 0;
        if (++j == nk) j = 0;
      }
      TCP_ADD(warp, 7, a9);
    }
  } else {
    const int r = warp >> 2, sub = warp & 3, ql = tid & 127;
    const uint32_t lane0 = (uint32_t)(sub * 32) << 16, tmO = tm + OCOL + 64 * r + lane0;
    const float qs = scale * LOG2E;
    // Q of item k: (q + qBias) scale log2e rounded to f16 then e4m3 (as flash8), A's core (q/8, k/16) at (k/16) 2048 + (q/8) 128
    auto convQ = [&](int k) {
      int h, q0, rg; decode(k, h, q0, rg);
      const int q = q0 + ql, row = min(rg * R + r, rows - 1);
      uint32_t w[8];
      const half* p = qkvg + ((size_t)row * n + q) * (size_t)(4 * W) + h * D;
      uint4 raw[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) raw[c] = q < n ? *reinterpret_cast<const uint4*>(p + 8 * c) : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float2 x = __half22float2(*reinterpret_cast<const half2*>(&(&raw[c >> 1].x)[(c & 1) * 2])),
                     y = __half22float2(*reinterpret_cast<const half2*>(&(&raw[c >> 1].x)[(c & 1) * 2 + 1]));
        float a0 = x.x, a1 = x.y, a2 = y.x, a3 = y.y;
        if (qBias && q < n) { a0 += qBias[h * D + 4 * c]; a1 += qBias[h * D + 4 * c + 1]; a2 += qBias[h * D + 4 * c + 2]; a3 += qBias[h * D + 4 * c + 3]; }
        unsigned short lo, hi;
        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(f16x2(a0 * qs, a1 * qs)));
        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(f16x2(a2 * qs, a3 * qs)));
        w[c] = lo | (uint32_t)hi << 16;
      }
      unsigned char* Qs = sm + L::Q + ((k & 1) * R + r) * 4096 + (ql >> 3) * 128 + (ql & 7) * 16;
      *reinterpret_cast<uint4*>(Qs) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(Qs + 2048) = make_uint4(w[4], w[5], w[6], w[7]);
      fenceProxy();
      __syncwarp();
      if (lane == 0) barArrive(qFull + (k & 1));
    };
    TCP_T(b9);
    if (mine > 0) convQ(0);
    for (int k = 0; k < mine; ++k) {
      int h, q0, rg; decode(k, h, q0, rg);
      const int q = q0 + ql, row = rg * R + r;
      if (k + 1 < mine) convQ(k + 1);   // (buffer (k+1)&1 was item k-1's: every S of it is in registers by now)
      float mx = -INFINITY;     // f16-representable: P = 2^(s - (mx - OFF))
      for (int j = 0; j < nk; ++j) {
        const int t = k * nk + j, i = r * 2 + (t & 1), is = (t % SB) * R + r;
        TCP_T(b0);
        barWait(sFull + is, (t / SB) & 1);
        if (lane == 0) TCP_ADD(warp, 0, b0);
        fenceAfter();
        uint32_t xr[TK];
        ld32(tm + is * TK + lane0, xr);
        ld32(tm + is * TK + lane0 + 32, xr + 32);
        ldWait();
        fenceBefore();
        __syncwarp();
        if (lane == 0) barArrive(sRead + is);
        float x[TK];
#pragma unroll
        for (int u = 0; u < TK; ++u) x[u] = __uint_as_float(xr[u]);
        float m8[8];       // (eight independent chains, then a tree)
#pragma unroll
        for (int c = 0; c < 8; ++c) m8[c] = fmaxf(x[c], fmaxf(x[8 + c], x[16 + c]));
#pragma unroll
        for (int u = 24; u < TK; u += 16)
#pragma unroll
          for (int c = 0; c < 8; ++c) m8[c] = fmaxf(m8[c], fmaxf(x[u + c], x[u + 8 + c]));
        const float rmax = fmaxf(fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])), fmaxf(fmaxf(m8[4], m8[5]), fmaxf(m8[6], m8[7])));
        float alpha = 1.f;
        if (rmax > mx + LAZY) {
          const float nm = __half2float(__float2half_rn(rmax));
          alpha = mx == -INFINITY ? 1.f : ex2f(mx - nm);
          mx = nm;
        }
        if (j > 0 && __any_sync(~0u, alpha != 1.f)) {   // O current first: every O += P V of this item so far
          barWait(pFree + r * 2 + ((t - 1) & 1), ((t - 1) >> 1) & 1);
          fenceAfter();
          uint32_t ov[32], sv;
          ld32(tmO, ov); ld1(tmO + 32, sv);
          ldWait();
#pragma unroll
          for (int u = 0; u < 32; ++u) ov[u] = __float_as_uint(__uint_as_float(ov[u]) * alpha);
          sv = __float_as_uint(__uint_as_float(sv) * alpha);
          st32(tmO, ov); st1(tmO + 32, sv);
          stWait();
        }
        const float ref = mx - OFF;
        // P = 2^(s - ref) a key pair at a time (adjacent keys stay adjacent in fp8VtK's order: its pairs move whole), each
        // pair two f32 exponentials and one conversion to an e4m3 pair, laid at its position pair
        unsigned short pe[32];
#pragma unroll
        for (int c = 0; c < 32; ++c) {
          float2 d = sub2(make_float2(x[2 * c], x[2 * c + 1]), ref);
          if (TC_EMU_PAIR(c)) { const float2 e = ex2emu2(d); pe[vtKeyPos(2 * c) / 2] = e4m3x2(e.x, e.y); }
          else pe[vtKeyPos(2 * c) / 2] = e4m3x2(ex2f(d.x), ex2f(d.y));
        }
        uint32_t pw[16];
#pragma unroll
        for (int w = 0; w < 16; ++w) pw[w] = pe[2 * w] | (uint32_t)pe[2 * w + 1] << 16;
        TCP_T(b1);
        if (t >= 2) { barWait(pFree + i, ((t - 2) >> 1) & 1); fenceAfter(); }
        if (lane == 0) TCP_ADD(warp, 1, b1);
        st16(tm + PCOL + 16 * i + lane0, pw);
        stWait();
        fenceBefore();
        __syncwarp();
        if (lane == 0) barArrive(pFull + i);
      }
      const int tl = k * nk + nk - 1;
      barWait(pFree + r * 2 + (tl & 1), (tl >> 1) & 1);
      fenceAfter();
      uint4 gv[4];                      // the gate (its load under the last tile's wait)
      const half* gp = qkvg + ((size_t)min(row, rows - 1) * n + min(q, n - 1)) * (size_t)(4 * W) + 3 * W + h * D;
#pragma unroll
      for (int c = 0; c < 4; ++c) gv[c] = *reinterpret_cast<const uint4*>(gp + 8 * c);
      uint32_t ov[32], sv;
      ld32(tmO, ov); ld1(tmO + 32, sv);
      ldWait();
      if (q < n && row < rows) {
        const float inv = 1.f / __uint_as_float(sv);
        half* o = out + ((size_t)row * n + q) * (size_t)W + h * D;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint32_t res[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 a = __half22float2(*reinterpret_cast<const half2*>(&(&gv[c].x)[e]));
            const float v0 = __uint_as_float(ov[8 * c + 2 * e]) * inv / (1.f + ex2f(-a.x * LOG2E)),
                        v1 = __uint_as_float(ov[8 * c + 2 * e + 1]) * inv / (1.f + ex2f(-a.y * LOG2E));
            const half2 hv = __floats2half2_rn(v0, v1);
            res[e] = *reinterpret_cast<const uint32_t*>(&hv);
          }
          *reinterpret_cast<uint4*>(o + 8 * c) = make_uint4(res[0], res[1], res[2], res[3]);
        }
      }
    }
    if (lane == 0) TCP_ADD(warp, 7, b9);
  }
  fenceBefore();
  __syncthreads();
  if (warp == MW) {
    fenceAfter();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm));
  }
#endif
}

// ---- the diffusion transformer's token attention on tcgen05 (flashTcH): flashGridHalf's contract at 48-wide heads -
// f16 q, k, v and gate out of qkvg ([sample][token][4W]), the f16 pair bias (times log2 e) shared by the samples, f16 P,
// scores in f32 - with flashTc's structure: a persistent block an SM over (head, 128 queries, 2 samples sharing the bias
// tile), the bias laid by the MMA for both, P in tensor memory, the sum by the PV product (ones rows after V^T).
// kind::f16 throughout: Q K^T three K16 steps (Q K-major, written by the softmax threads; K three 32B-swizzled 16-dim
// chunks by TMA), O += P V^T four (P from tensor memory, 8 columns a step; V^T 128B-swizzled rows, made by vtHK).
namespace th {
constexpr int D = 48, R = 2, SB = 2, TS = 3, TV = 3;   // TS stages of the S ring (bias, K), TV of the V ring
constexpr float LAZYH = 8.f;     // (P = 2^(s - max) <= 2^8 in f16; f16 keeps 2^-24 below the max)
// shared memory: Q [2][R] 12 KB; TS stages of the S ring { bias 16 KB, [R] K 3 x 2 KB }, freed once that tile's S products
// complete; TV of the V ring { [R] { V^T 6 KB, ones 2 KB } }, freed once its PV products do (one ring held a stage from
// its load to its PV - with S two tiles ahead, a 3-stage ring left the loads no lead: the products waited on them 36% of
// the time and the softmax on S 45%); the stacked identity 16 KB; the mask 3 KB; the barriers
struct Lay {
  static constexpr uint32_t Q = 0, STAGE = 16384 + R * 6144, VSTAGE = R * 8192, ST = 2 * R * 12288, VT = ST + TS * STAGE,
                            ID = VT + TV * VSTAGE, MA = ID + R * 8192, MB = MA + 1024, BAR = MB + 2048, BYTES = BAR + 512;
};
constexpr int WARPS = 4 * R + 3;   // 4R softmax, the S products, the PV products, the loads
}  // namespace th
// V^T [row][W][NK] f16 from qkvg's v ([row][pos][4W] at 2W): a block 64 keys x 64 dims of one row through shared memory
__global__ void __launch_bounds__(256) vtHK(const half* __restrict__ qkvg, half* __restrict__ vt, int n, int NK, int W) {
  __shared__ half t[64][66];
  const size_t r = blockIdx.z; const int k0 = blockIdx.x * 64, d0 = blockIdx.y * 64;
  for (int u = threadIdx.x; u < 64 * 32; u += 256) {
    const int kk = u / 32, dd = (u % 32) * 2, key = k0 + kk;
    half2 v = __floats2half2_rn(0.f, 0.f);
    if (key < n && d0 + dd < W) v = *reinterpret_cast<const half2*>(qkvg + (r * n + key) * (size_t)(4 * W) + 2 * W + d0 + dd);
    t[dd][kk] = v.x; t[dd + 1][kk] = v.y;
  }
  __syncthreads();
  for (int u = threadIdx.x; u < 64 * 32; u += 256) {
    const int dd = u / 32, kk = (u % 32) * 2;
    if (d0 + dd < W) *reinterpret_cast<half2*>(vt + (r * W + d0 + dd) * (size_t)NK + k0 + kk) = __halves2half2(t[dd][kk], t[dd][kk + 1]);
  }
}
__global__ void __launch_bounds__(32 * th::WARPS, 1) flashTcH(const __grid_constant__ CUtensorMap mapB, const __grid_constant__ CUtensorMap mapK,
    const __grid_constant__ CUtensorMap mapV, const half* __restrict__ qkvg, half* __restrict__ out, int n, int heads, int rows,
    float scale, const float* __restrict__ qBias) {
#if LF_TC5A
  using namespace th;
  using L = th::Lay;
  constexpr int MW = 4 * R, PW = 4 * R + 1, LW = 4 * R + 2;
  constexpr uint32_t OCOL = 64 * R * SB, PCOL = OCOL + 64 * R;     // S 256, O 128 (48 O, 1 sum), P [r][2] 32 each: 512
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* kvFull = reinterpret_cast<uint64_t*>(sm + L::BAR);
  uint64_t *kvFree = kvFull + TS, *sFull = kvFree + TS, *sRead = sFull + R * SB, *pFull = sRead + R * SB, *pFree = pFull + 2 * R,
           *qFull = pFree + 2 * R, *vFull = qFull + 2, *vFree = vFull + TV;
  __shared__ uint32_t tmBase;
  const int W = heads * D, nqt = (n + TQ - 1) / TQ, nrg = (rows + R - 1) / R, nk = (n + TK - 1) / TK;
  const unsigned per = (unsigned)nqt * nrg, items = per * heads;
  const int mine = items > blockIdx.x ? (int)((items - blockIdx.x + gridDim.x - 1) / gridDim.x) : 0;
  auto decode = [&](int k, int& h, int& q0, int& rg) {   // a head's items: every row group of a query tile together
    const unsigned w = blockIdx.x + (unsigned)k * gridDim.x, r0 = w % per;
    h = (int)(w / per); rg = (int)(r0 % nrg); q0 = (int)(r0 / nrg) * TQ;
  };
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = sa(sm);
  if (warp == MW) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    for (int s = 0; s < TS; ++s) { barInit(kvFull + s, 1); barInit(kvFree + s, 1); }
    for (int i = 0; i < R * SB; ++i) { barInit(sFull + i, 1); barInit(sRead + i, 4); }
    for (int i = 0; i < 2 * R; ++i) { barInit(pFull + i, 4); barInit(pFree + i, 1); }
    for (int i = 0; i < 2; ++i) barInit(qFull + i, 4 * R);
    for (int s = 0; s < TV; ++s) { barInit(vFull + s, 1); barInit(vFree + s, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // the ones rows after each stage's V^T (rows 48..63 of the 128B-swizzled tile: row 48 all f16 ones, unmoved by the swizzle)
  for (int i = tid; i < TV * R * 128; i += blockDim.x) {
    const int sr = i >> 7, u = i & 127;
    *reinterpret_cast<uint4*>(sm + L::VT + (sr / R) * L::VSTAGE + (sr % R) * 8192 + 6144 + u * 16) =
        u < 8 ? make_uint4(0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u) : make_uint4(0, 0, 0, 0);
  }
  for (int i = tid; i < R * 8192 / 16; i += blockDim.x) {      // the stacked identity: row n of B is e_(n mod 64)
    const int kc = i / (R * 64), g = (i / 8) % (R * 8), rr = i & 7, nn = (g * 8 + rr) & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (nn / 8 == kc) { const int k = nn & 7; (&v.x)[k >> 1] = (k & 1) ? 0x3C000000u : 0x3C00u; }
    *reinterpret_cast<uint4*>(sm + L::ID + i * 16) = v;
  }
  for (int i = tid; i < 16; i += blockDim.x)
    *reinterpret_cast<uint4*>(sm + L::MA + i * 16) = make_uint4(i < 8 ? 0x3C00u : 0u, 0, 0, 0);
  for (int i = tid; i < 2048 / 16; i += blockDim.x) {
    const int kc = i >> 6, u = i & 63, valid = n - (n - 1) / TK * TK;
    *reinterpret_cast<uint4*>(sm + L::MB + i * 16) = make_uint4(kc == 0 && u >= valid ? 0xF753u : 0u, 0, 0, 0);   // (-30000)
  }
  fenceProxy();
  fenceBefore();
  __syncthreads();
  fenceAfter();
  const uint32_t tm = tmBase;
  const int T = mine * nk;
  if (warp == LW) {
    if (lane == 0) {
      int s = 0, u = 0, sv = 0, uv = 0;
      for (int k = 0; k < mine; ++k) {
        int h, q0, rg; decode(k, h, q0, rg);
        int row[R];
        for (int r = 0; r < R; ++r) row[r] = min(rg * R + r, rows - 1);
        for (int j = 0; j < nk; ++j) {
          if (u > 0) barWait(kvFree + s, (u - 1) & 1);
          barExpect(kvFull + s, 16384 + R * 6144);
          unsigned char* st = sm + L::ST + s * L::STAGE;
          tma3(st, &mapB, j * TK, q0, h, kvFull + s);                                     // bias [q 128][64 keys], 128B swizzle
          for (int r = 0; r < R; ++r)
            for (int c = 0; c < 3; ++c)                                                   // K [key 64][16 dims] x 3, 32B swizzle
              tma3(st + 16384 + r * 6144 + c * 2048, &mapK, W + h * D + 16 * c, j * TK, row[r], kvFull + s);
          if (++s == TS) { s = 0; ++u; }
          if (uv > 0) barWait(vFree + sv, (uv - 1) & 1);
          barExpect(vFull + sv, R * 6144);
          for (int r = 0; r < R; ++r)                                                     // V^T [dim 48][64 keys], 128B swizzle
            tma3(sm + L::VT + sv * L::VSTAGE + r * 8192, &mapV, j * TK, h * D, row[r], vFull + sv);
          if (++sv == TV) { sv = 0; ++uv; }
        }
      }
    }
  } else if (warp == MW || warp == PW) {
    // two issuing threads (a commit tracks its own thread's products): the S side - the bias for both rows, Q K^T, the
    // S ring's stage freed - as soon as both softmaxes have S(t - SB) in registers; the PV side as each P(t) is written.
    // (One thread issuing all 18 products and 7 commits of a tile step was busy 75% of the time and S came late.)
    if (lane == 0 && warp == MW) {
      constexpr uint32_t IDB = idesc(TQ, TK), IDBR = idesc(TQ, TK * R), IDQK = idesc(TQ, TK);
      const uint64_t dBias = sdesc(base + L::ST, 16, 1024, SW128), dId = sdesc(base + L::ID, R * 1024, 128),
                     dQ = sdesc(base + L::Q, 2048, 128), dK = sdesc(base + L::ST + 16384, 16, 256, SW32),
                     dMA = sdesc(base + L::MA, 128, 0), dMB = sdesc(base + L::MB, 1024, 128);
      int ks = 0, js = 0, ss = 0;
      for (int ts = 0; ts < T; ++ts) {
        const int b = ts % SB;
        TCP_T(a0);
        barWait(kvFull + ss, (ts / TS) & 1);
        TCP_ADD(warp, 0, a0);
        TCP_T(a1);
        if (js == 0) barWait(qFull + (ks & 1), (ks >> 1) & 1);
        TCP_ADD(warp, 1, a1);
        TCP_T(a2);
        if (ts >= SB) for (int r = 0; r < R; ++r) barWait(sRead + b * R + r, ((ts - SB) / SB) & 1);
        TCP_ADD(warp, 2, a2);
        fenceAfter();
        const uint32_t so = ss * L::STAGE >> 4, d = tm + b * R * TK;
#pragma unroll
        for (int c = 0; c < 4; ++c) mmaF16(d, dBias + so + c * 2, dId + c * (2 * R * 1024 >> 4), IDBR, c > 0);
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
          for (int c = 0; c < 3; ++c)   // S_r += Q_r K_r^T, 16 dims a step
            mmaF16(d + r * TK, dQ + ((((ks & 1) * R + r) * 12288 + c * 4096) >> 4), dK + so + ((r * 6144 + c * 2048) >> 4), IDQK, true);
          if (js == nk - 1 && n % TK) mmaF16(d + r * TK, dMA, dMB, IDB, true);
          commit(sFull + b * R + r);
        }
        commit(kvFree + ss);     // (the S ring's stage: free once these products complete)
        if (++ss == TS) ss = 0; if (++js == nk) { js = 0; ++ks; }
      }
    }
    if (lane == 0 && warp == PW) {
      constexpr uint32_t IDPV = idesc(TQ, 64);
      const uint64_t dV = sdesc(base + L::VT, 16, 1024, SW128);
      int j = 0, s = 0;
      TCP_T(a9);
      for (int t = 0; t < T; ++t) {
        barWait(vFull + s, (t / TV) & 1);
        fenceAfter();
        const uint32_t so = s * L::VSTAGE >> 4;
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int i = r * 2 + (t & 1);
          TCP_T(a3);
          barWait(pFull + i, (t >> 1) & 1);
          TCP_ADD(warp, 3, a3);
          fenceAfter();
#pragma unroll
          for (int c = 0; c < 4; ++c)   // [O | sum] += P [V^T | 1]^T: P from tensor memory (8 columns a K16 step)
            mmaF16ts(tm + OCOL + 64 * r, tm + PCOL + 32 * i + 8 * c, dV + so + ((r * 8192 + c * 32) >> 4), IDPV, j > 0 || c > 0);
          commit(pFree + i);
        }
        commit(vFree + s);
        if (++s == TV) s = 0;
        if (++j == nk) j = 0;
      }
      TCP_ADD(warp, 7, a9);
    }
  } else {
    const int r = warp >> 2, sub = warp & 3, ql = tid & 127;
    const uint32_t lane0 = (uint32_t)(sub * 32) << 16, tmO = tm + OCOL + 64 * r + lane0;
    const float qs = scale * LOG2E;
    // Q of item k: (q + qBias) scale log2e in f16 (flashGridHalf's), A's core (q/8, k/8) at (k/8) 2048 + (q/8) 128
    auto convQ = [&](int k) {
      int h, q0, rg; decode(k, h, q0, rg);
      const int q = q0 + ql, row = min(rg * R + r, rows - 1);
      const half* p = qkvg + ((size_t)row * n + q) * (size_t)(4 * W) + h * D;
      unsigned char* Qs = sm + L::Q + ((k & 1) * R + r) * 12288 + (ql >> 3) * 128 + (ql & 7) * 16;
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        uint4 raw = q < n ? *reinterpret_cast<const uint4*>(p + 8 * c) : make_uint4(0, 0, 0, 0);
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float2 f = __half22float2(*reinterpret_cast<const half2*>(&(&raw.x)[e]));
          if (qBias && q < n) { f.x += qBias[h * D + 8 * c + 2 * e]; f.y += qBias[h * D + 8 * c + 2 * e + 1]; }
          w[e] = f16x2(f.x * qs, f.y * qs);
        }
        *reinterpret_cast<uint4*>(Qs + c * 2048) = make_uint4(w[0], w[1], w[2], w[3]);
      }
      fenceProxy();
      __syncwarp();
      if (lane == 0) barArrive(qFull + (k & 1));
    };
    TCP_T(b9);
    if (mine > 0) convQ(0);
    for (int k = 0; k < mine; ++k) {
      int h, q0, rg; decode(k, h, q0, rg);
      const int q = q0 + ql, row = rg * R + r;
      if (k + 1 < mine) convQ(k + 1);
      float mx = -INFINITY;     // f16-representable: P = 2^(s - mx)
      for (int j = 0; j < nk; ++j) {
        const int t = k * nk + j, i = r * 2 + (t & 1), is = (t % SB) * R + r;
        TCP_T(b0);
        barWait(sFull + is, (t / SB) & 1);
        if (lane == 0) TCP_ADD(warp, 0, b0);
        fenceAfter();
        uint32_t xr[TK];
        ld32(tm + is * TK + lane0, xr);
        ld32(tm + is * TK + lane0 + 32, xr + 32);
        ldWait();
        fenceBefore();
        __syncwarp();
        if (lane == 0) barArrive(sRead + is);
        float x[TK];
#pragma unroll
        for (int u = 0; u < TK; ++u) x[u] = __uint_as_float(xr[u]);
        float m8[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) m8[c] = fmaxf(x[c], fmaxf(x[8 + c], x[16 + c]));
#pragma unroll
        for (int u = 24; u < TK; u += 16)
#pragma unroll
          for (int c = 0; c < 8; ++c) m8[c] = fmaxf(m8[c], fmaxf(x[u + c], x[u + 8 + c]));
        const float rmax = fmaxf(fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])), fmaxf(fmaxf(m8[4], m8[5]), fmaxf(m8[6], m8[7])));
        float alpha = 1.f;
        if (rmax > mx + LAZYH) {
          const float nm = __half2float(__float2half_rn(rmax));
          alpha = mx == -INFINITY ? 1.f : ex2f(mx - nm);
          mx = nm;
        }
        if (j > 0 && __any_sync(~0u, alpha != 1.f)) {
          barWait(pFree + r * 2 + ((t - 1) & 1), ((t - 1) >> 1) & 1);
          fenceAfter();
          uint32_t ov[32], o2[16], sv;
          ld32(tmO, ov); ld16(tmO + 32, o2); ld1(tmO + 48, sv);
          ldWait();
#pragma unroll
          for (int u = 0; u < 32; ++u) ov[u] = __float_as_uint(__uint_as_float(ov[u]) * alpha);
#pragma unroll
          for (int u = 0; u < 16; ++u) o2[u] = __float_as_uint(__uint_as_float(o2[u]) * alpha);
          sv = __float_as_uint(__uint_as_float(sv) * alpha);
          st32(tmO, ov); st16(tmO + 32, o2); st1(tmO + 48, sv);
          stWait();
        }
        uint32_t pw[32];   // P in f16, key pairs in order
#pragma unroll
        for (int c = 0; c < 32; ++c) {
          const float2 d = sub2(make_float2(x[2 * c], x[2 * c + 1]), mx);
          pw[c] = f16x2(ex2f(d.x), ex2f(d.y));
        }
        TCP_T(b1);
        if (t >= 2) { barWait(pFree + i, ((t - 2) >> 1) & 1); fenceAfter(); }
        if (lane == 0) TCP_ADD(warp, 1, b1);
        st32(tm + PCOL + 32 * i + lane0, pw);
        stWait();
        fenceBefore();
        __syncwarp();
        if (lane == 0) barArrive(pFull + i);
      }
      const int tl = k * nk + nk - 1;
      barWait(pFree + r * 2 + (tl & 1), (tl >> 1) & 1);
      fenceAfter();
      uint4 gv[6];
      const half* gp = qkvg + ((size_t)min(row, rows - 1) * n + min(q, n - 1)) * (size_t)(4 * W) + 3 * W + h * D;
#pragma unroll
      for (int c = 0; c < 6; ++c) gv[c] = *reinterpret_cast<const uint4*>(gp + 8 * c);
      uint32_t ov[48], sv;
      ld32(tmO, ov); ld16(tmO + 32, ov + 32); ld1(tmO + 48, sv);
      ldWait();
      if (q < n && row < rows) {
        const float inv = 1.f / __uint_as_float(sv);
        half* o = out + ((size_t)row * n + q) * (size_t)W + h * D;
#pragma unroll
        for (int c = 0; c < 6; ++c) {
          uint32_t res[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 a = __half22float2(*reinterpret_cast<const half2*>(&(&gv[c].x)[e]));
            res[e] = f16x2(__uint_as_float(ov[8 * c + 2 * e]) * inv / (1.f + __expf(-a.x)), __uint_as_float(ov[8 * c + 2 * e + 1]) * inv / (1.f + __expf(-a.y)));
          }
          *reinterpret_cast<uint4*>(o + 8 * c) = make_uint4(res[0], res[1], res[2], res[3]);
        }
      }
    }
    if (lane == 0) TCP_ADD(warp, 7, b9);
  }
  fenceBefore();
  __syncthreads();
  if (warp == MW) {
    fenceAfter();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm));
  }
#endif
}

// whether this binary carries sm_100a code for the device it runs on (asked once, of the device)
__global__ void probeK(int* o) { *o = LF_TC5A; }
inline bool ready() {
  static int on = -1;
  if (on < 0) {
    int dev = 0, major = 0, minor = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    on = 0;
    if (major == 10 && minor == 0) {
      int* d; CK(cudaMalloc(&d, sizeof(int))); CK(cudaMemset(d, 0, sizeof(int)));
      probeK<<<1, 1>>>(d); CK(cudaGetLastError());
      CK(cudaMemcpy(&on, d, sizeof(int), cudaMemcpyDeviceToHost)); CK(cudaFree(d));
    }
  }
  return on == 1;
}
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
inline EncodeFn encoder() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
#else
    CK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q));
#endif
    if (!p || q != cudaDriverEntryPointSuccess) { fprintf(stderr, "cuTensorMapEncodeTiled: not found\n"); exit(1); }
    fn = (EncodeFn)p;
  }
  return fn;
}
// a 4-d map: dims (innermost first), byte strides of dims 1..3, the box
inline CUtensorMap map4(CUtensorMapDataType t, const void* p, std::array<uint64_t, 4> dims, std::array<uint64_t, 3> strides,
                        std::array<uint32_t, 4> box) {
  CUtensorMap m;
  const cuuint32_t es[4] = { 1, 1, 1, 1 };
  CUresult r = encoder()(&m, t, 4, const_cast<void*>(p), dims.data(), strides.data(), box.data(), es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled failed (%d)\n", (int)r); exit(1); }
  return m;
}
inline CUtensorMap map3(CUtensorMapDataType t, const void* p, std::array<uint64_t, 3> dims, std::array<uint64_t, 2> strides,
                        std::array<uint32_t, 3> box, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  const cuuint32_t es[3] = { 1, 1, 1 };
  CUresult r = encoder()(&m, t, 3, const_cast<void*>(p), dims.data(), strides.data(), box.data(), es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled failed (%d)\n", (int)r); exit(1); }
  return m;
}
inline LF_TLS bool TC_ATTN = true;  // --no-tc-attn: flash8 (mma.sync) on a B200 too
constexpr int TC_GROUP = 8;     // the block order's row groups
// the attention from qkvg's q and gate (f16, [row][pos][4W]), K8 (e4m3 [row][pos][W]) and Vt8 (fp8VtK's [row][W][NK])
template <int R>
inline void flashTcRunR(const half* qkvg, const uint8_t* K8, const uint8_t* Vt8, const half* bias, int stride, half* out, int n,
                        int heads, size_t rows, float scale, const float* qBias) {
  const int W = heads * 32, NK = (n + 63) / 64 * 64;
  if (stride % 8) { fprintf(stderr, "flashTcRun: bias stride %d not a multiple of 8\n", stride); exit(1); }
  const CUtensorMap mb = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, bias, { (uint64_t)n, (uint64_t)n, (uint64_t)heads },
                              { (uint64_t)stride * 2, (uint64_t)n * stride * 2 }, { 64, 128, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap mk = map3(CU_TENSOR_MAP_DATA_TYPE_UINT8, K8, { (uint64_t)W, (uint64_t)n, rows }, { (uint64_t)W, (uint64_t)n * W },
                              { 32, 64, 1 }, CU_TENSOR_MAP_SWIZZLE_32B);
  const CUtensorMap mv = map3(CU_TENSOR_MAP_DATA_TYPE_UINT8, Vt8, { (uint64_t)NK, (uint64_t)W, rows }, { (uint64_t)NK, (uint64_t)W * NK },
                              { 64, 32, 1 }, CU_TENSOR_MAP_SWIZZLE_64B);
  static LF_TLS bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(flashTc<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)Lay<R>::BYTES)); attr = true; }
  const unsigned nqt = (n + TQ - 1) / TQ, nrg = (unsigned)((rows + R - 1) / R), items = nqt * nrg * heads;
  static LF_TLS int sms = 0;
  if (!sms) { int dev; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); }
  flashTc<R><<<std::min<unsigned>(items, sms), 32 * tcWarps<R>(), Lay<R>::BYTES, STREAM>>>(mb, mk, mv, qkvg, out, n, heads, (int)rows, scale, qBias, TC_GROUP);
  CK(cudaGetLastError());
}
// rowsPerBlock: 3 (measured faster than 2 at every length from 256 to 4,096 tokens, test_flash_tc5: 294 against 247
// TFLOP/s at 1,024, 364 against 311 on a real 4,096-token call) or 2 (test_flash_tc5's arm)
inline void flashTcRun(const half* qkvg, const uint8_t* K8, const uint8_t* Vt8, const half* bias, int stride, half* out, int n,
                       int heads, size_t rows, float scale, const float* qBias, int rowsPerBlock = 3) {
  if (rowsPerBlock == 3) flashTcRunR<3>(qkvg, K8, Vt8, bias, stride, out, n, heads, rows, scale, qBias);
  else if (rowsPerBlock == 2) flashTcRunR<2>(qkvg, K8, Vt8, bias, stride, out, n, heads, rows, scale, qBias);
  else { fprintf(stderr, "flashTcRun: %d rows a block (2 or 3)\n", rowsPerBlock); exit(1); }
}
// the diffusion's token attention (flashGridHalf's contract at D 48, no mask, no query range): rows the samples
inline void flashTcHRun(const half* qkvg, const half* bias, int stride, half* out, int n, int heads, size_t rows, float scale,
                        const float* qBias) {
  const int W = heads * th::D, NK = (n + 63) / 64 * 64;
  if (stride % 8) { fprintf(stderr, "flashTcHRun: bias stride %d not a multiple of 8\n", stride); exit(1); }
  half* vt = scratch<half>("attn.vtH", rows * (size_t)W * NK);
  vtHK<<<dim3(NK / 64, (W + 63) / 64, (unsigned)rows), 256, 0, STREAM>>>(qkvg, vt, n, NK, W);
  const CUtensorMap mb = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, bias, { (uint64_t)n, (uint64_t)n, (uint64_t)heads },
                              { (uint64_t)stride * 2, (uint64_t)n * stride * 2 }, { 64, 128, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap mk = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, qkvg, { (uint64_t)4 * W, (uint64_t)n, rows },
                              { (uint64_t)4 * W * 2, (uint64_t)n * 4 * W * 2 }, { 16, 64, 1 }, CU_TENSOR_MAP_SWIZZLE_32B);
  const CUtensorMap mv = map3(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, vt, { (uint64_t)NK, (uint64_t)W, rows },
                              { (uint64_t)NK * 2, (uint64_t)W * NK * 2 }, { 64, 48, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  static LF_TLS bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(flashTcH, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)th::Lay::BYTES)); attr = true; }
  const unsigned nqt = (n + TQ - 1) / TQ, nrg = (unsigned)((rows + th::R - 1) / th::R), items = nqt * nrg * heads;
  static LF_TLS int sms = 0;
  if (!sms) { int dev; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)); }
  flashTcH<<<std::min<unsigned>(items, sms), 32 * th::WARPS, th::Lay::BYTES, STREAM>>>(mb, mk, mv, qkvg, out, n, heads, (int)rows, scale, qBias);
  CK(cudaGetLastError());
}
}  // namespace tc5a
