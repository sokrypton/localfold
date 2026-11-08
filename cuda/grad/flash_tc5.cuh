// The grid attention's forward on the B200's tcgen05 tensor cores (sm_100a): gridFwdBf's contract - its tiles, key
// terms and bias tile, its log-sum-exp pair and fused gate - with the matrix products issued by one thread into
// tensor memory instead of 4,000 mma.sync a tile across the warps. gridFwdBf measured 155 TFLOP/s at 768 tokens on the
// B200 (cuDNN's tcgen05 SDPA 360, without a bias): with 32-wide heads it was bound by issue, ~20 instructions an MMA.
//
// A block: one (row b, head h, 128 queries), 128 threads, a thread a query - tensor memory's lane is the query, so the
// row's max and sum need no shuffles. A tile of 64 keys: S = Q K^T into tensor memory (columns 0..63), each thread its
// row's 64 logits out of it (s scale log2e + bias log2e + key term, gridFwdBf's two FMAs), the online softmax, O
// (columns 64..95) rescaled in place where the max moved, P in bf16 into shared memory, O += P V. K and V double
// buffered (cp.async); the two products' completions on mbarriers. Operands without swizzle: Q, K and P K-major, V
// MN-major (its rows as they are), each in 8 x 16-byte core matrices (test_tc5.cu holds the encodings to the host).
#pragma once
#include <cstdint>

namespace grad {
namespace flash {
namespace tc5 {
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// the instruction descriptor (kind::f16): bf16 A and B, f32 accumulator; bMN: B MN-major (else K-major)
__host__ __device__ constexpr uint32_t idesc(int M, int N, bool bMN) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)bMN << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
// a shared-memory operand without swizzle: lbo the bytes between core matrices along K (K-major) or along K's 8-row
// groups (MN-major), sbo along M/N
__device__ __forceinline__ uint64_t sdesc(const void* p, uint32_t lbo, uint32_t sbo) {
  return ((uint64_t)(sa(p) >> 4) & 0x3fff) | ((uint64_t)((lbo >> 4) & 0x3fff) << 16) |
         ((uint64_t)((sbo >> 4) & 0x3fff) << 32) | (1ull << 46);
}
__device__ __forceinline__ void mma(uint32_t c, uint64_t a, uint64_t b, uint32_t id, bool acc) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
               :: "r"(c), "l"(a), "l"(b), "r"(id), "r"((uint32_t)acc));
#endif
}
__device__ __forceinline__ void commit(uint64_t* m) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(sa(m)) : "memory");
#endif
}
__device__ __forceinline__ void barInit(uint64_t* m) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(m)) : "memory"); }
__device__ __forceinline__ void barWait(uint64_t* m, uint32_t parity) {
  asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra W;\n\t}\n"
               :: "r"(sa(m)), "r"(parity) : "memory");
}
__device__ __forceinline__ void fenceBefore() {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#endif
}
__device__ __forceinline__ void fenceAfter() {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}
__device__ __forceinline__ void cpWait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fenceProxy() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// 32 consecutive columns of this warp's 32 lanes (a thread its lane's), and back
__device__ __forceinline__ void ld32(uint32_t addr, float* v) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  uint32_t r[32];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
#endif
}
__device__ __forceinline__ void st32(uint32_t addr, const float* v) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  uint32_t r[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) r[i] = __float_as_uint(v[i]);
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
               "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
                  "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]),
                  "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]) : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#endif
}
// whether this binary carries sm_100a code (tcgen05) for the device it runs on: asked once, of the device
__global__ void probeK(int* out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  *out = 1;
#else
  *out = 0;
#endif
}
inline bool ready() {
  static int on = -1;
  if (on < 0) {
    int dev = 0, major = 0;
    GK(cudaGetDevice(&dev));
    GK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    on = 0;
    if (major == 10) {
      int* d; GK(cudaMalloc(&d, sizeof(int))); GK(cudaMemset(d, 0, sizeof(int)));
      probeK<<<1, 1>>>(d); GK(cudaGetLastError());
      GK(cudaMemcpy(&on, d, sizeof(int), cudaMemcpyDeviceToHost)); GK(cudaFree(d));
    }
  }
  return on == 1;
}

__device__ __forceinline__ void barInitN(uint64_t* m, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(m)), "r"(n) : "memory"); }
__device__ __forceinline__ void barArrive(uint64_t* m) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(m)) : "memory"); }
// the calling thread's cp.async copies so far, arriving on m once they land (the barrier's count holds the arrival)
__device__ __forceinline__ void cpArrive(uint64_t* m) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(sa(m)) : "memory"); }
}  // namespace tc5

// A block: one (row b, head h, 256 queries) in two query tiles of 128, ten warps with a role each - warps 0-3 and 4-7
// the two tiles' softmaxes (a thread a query: tensor memory's lane), warp 8 the products (one thread issues both tiles'
// S = Q K^T and O += P V), warp 9 the loads (K, V and the key terms, three stages ahead). Each tile's S and P double
// buffered (S in tensor memory, P in shared), so a softmax never waits on a product: S of tile j + 2 is issued while
// tile j's P is being written, and the two tiles' softmaxes overlap the products besides. O is rescaled lazily (as
// FlashAttention-4 does): a row keeps the max it last rescaled at until a tile's max passes it by 2^8, so P stays at
// most 2^8 and the softmax rarely needs O current - the log-sum-exp pair it writes (that max, the log of the sum at it)
// differs from gridFwdBf's (the true max) and their sum, all the backward reads, does not.
// The handoffs on mbarriers: K/V full (landed) and free (both tiles' products of it done); per tile and buffer S full,
// S read (its softmax has it in registers), P full (written), P free (its O += P V done).
constexpr int TQ = 128, TK = 64, TS = 3;
constexpr float LAZY = 8.f;   // (the max's slack, in base 2, before O is rescaled)
// NG query tiles a block (NG = 2: 256 queries; 3: 384 - three softmaxes a scheduler partition, where two left the
// exponential unit idle 3/4 of the time); warps: 4 NG softmax, the products', the loads'
template <int NG> __host__ __device__ constexpr int tcWarps() { return 4 * NG + 2; }
// shared memory: Q [NG][128 x 32], TS stages of K and V [64 x 32], P [NG tiles][2 buffers][128 x 64] (bf16), TS
// stages of the key terms, the barriers
template <int NG>
constexpr size_t gridFwdTcSmem() { return (size_t)NG * TQ * 32 * 2 + TS * 2 * TK * 32 * 2 + (size_t)2 * NG * TQ * TK * 2 + TS * TK * 4 + (2 * TS + 8 * NG) * 8; }

template <int NG>
__global__ void __launch_bounds__(32 * tcWarps<NG>(), 1) gridFwdTc(Sdpa a, BiasTile bt, HSrc qs_, HSrc ks_, HSrc vs_,
                                                         __nv_bfloat16* __restrict__ o, int ldo, float* __restrict__ lse,
                                                         __nv_bfloat16* __restrict__ gate, int ldg, __nv_bfloat16* __restrict__ og,
                                                         int ldog, const float* __restrict__ ka2, int kpad) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  constexpr int D = 32;
  extern __shared__ __align__(1024) unsigned char tsm[];
  uint16_t* Qs = reinterpret_cast<uint16_t*>(tsm);                        // [tile] core matrices ((q / 8) 4 + d / 8)
  constexpr int MW = 4 * NG, LW = 4 * NG + 1, NT = 32 * tcWarps<NG>();   // (the products' warp, the loads')
  uint16_t* KVs = Qs + NG * TQ * D;                                       // [stage][K, V][64 x 32]
  uint16_t* Ps = KVs + TS * 2 * TK * D;                                   // [tile][buffer] core matrices ((q / 8) 8 + k / 8)
  float* KAs = reinterpret_cast<float*>(Ps + 2 * NG * TQ * TK);           // [stage][64]
  uint64_t* kvFull = reinterpret_cast<uint64_t*>(KAs + TS * TK);
  uint64_t *kvFree = kvFull + TS, *sFull = kvFree + TS, *sRead = sFull + 2 * NG, *pFull = sRead + 2 * NG, *pFree = pFull + 2 * NG;   // [tile 2 + buffer]
  __shared__ uint32_t tmBase;
  const int nqb = (a.Q + NG * TQ - 1) / (NG * TQ);
  const int qb = blockIdx.x % nqb, h = (blockIdx.x / nqb) % a.H, b = blockIdx.x / (nqb * a.H);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int q0 = qb * NG * TQ;
  auto kOff = [](int r, int c) { return ((r >> 3) * 4 + c) * 64 + (r & 7) * 8; };          // K-major, 4 chunks a row group
  auto vOff = [](int r, int c) { return (c * (TK / 8) + (r >> 3)) * 64 + (r & 7) * 8; };   // V: MN-major, k-groups adjacent
  for (int i = tid; i < NG * TQ * 4; i += NT) {
    const int r = i >> 2, c = i & 3;
    const bool on = q0 + r < qs_.N && c * 8 < qs_.W;
    cp16(Qs + (r / TQ) * TQ * D + kOff(r % TQ, c), on ? hrow(qs_, b, h, q0 + r) + c * 8 : qs_.p, on);
  }
  cpCommit();
  if (warp == MW) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(tc5::sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    for (int s = 0; s < TS; ++s) { tc5::barInitN(kvFull + s, 32); tc5::barInitN(kvFree + s, 1); }
    for (int i = 0; i < 2 * NG; ++i) { tc5::barInitN(sFull + i, 1); tc5::barInitN(sRead + i, 4); tc5::barInitN(pFull + i, 4); tc5::barInitN(pFree + i, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  tc5::cpWait0();
  tc5::fenceProxy();
  tc5::fenceBefore();
  __syncthreads();
  tc5::fenceAfter();
  const uint32_t tm = tmBase;   // S [tile 2 + buffer] at columns 64 i, O [tile] at 128 NG + 32 tile
  const int nk = (a.K + TK - 1) / TK;
  if (warp == LW) {
    for (int j = 0; j < nk; ++j) {
      const int s = j % TS, k0 = j * TK;
      if (j >= TS) tc5::barWait(kvFree + s, ((j - TS) / TS) & 1);
      uint16_t* Kt = KVs + s * 2 * TK * D; uint16_t* Vt = Kt + TK * D;
#pragma unroll 4
      for (int i = lane; i < TK * 4; i += 32) {
        const int r = i >> 2, c = i & 3;
        const bool on = k0 + r < ks_.N && c * 8 < ks_.W;
        cp16(Kt + kOff(r, c), on ? hrow(ks_, b, h, k0 + r) + c * 8 : ks_.p, on);
        cp16(Vt + vOff(r, c), on ? hrow(vs_, b, h, k0 + r) + c * 8 : vs_.p, on);
      }
      if (lane < TK / 4) cp16(KAs + s * TK + lane * 4, ka2 + (size_t)b * kpad + k0 + lane * 4, true);
      tc5::cpArrive(kvFull + s);
    }
  } else if (warp == MW) {
    if (lane == 0) {
      constexpr uint32_t ID1 = tc5::idesc(TQ, TK, false), ID2 = tc5::idesc(TQ, D, true);
      auto kvReady = [&](int j) {
        tc5::barWait(kvFull + j % TS, (j / TS) & 1);
        tc5::fenceProxy();
        tc5::fenceAfter();
      };
      auto S = [&](int g, int j) {   // S_g(j) = Q_g K(j)^T into buffer j % 2, once that buffer's last S is in registers
        const int i = g * 2 + (j & 1);
        if (j >= 2) { tc5::barWait(sRead + i, ((j - 2) >> 1) & 1); tc5::fenceAfter(); }
        const uint16_t* Kt = KVs + (j % TS) * 2 * TK * D;
#pragma unroll
        for (int k = 0; k < D / 16; ++k)
          tc5::mma(tm + i * TK, tc5::sdesc(Qs + g * TQ * D + k * 128, 128, 512), tc5::sdesc(Kt + k * 128, 128, 512), ID1, k > 0);
        tc5::commit(sFull + i);
      };
      for (int j = 0; j < nk && j < 2; ++j) { kvReady(j); for (int g = 0; g < NG; ++g) S(g, j); }
      for (int j = 0; j < nk; ++j) {
        const uint16_t* Vt = KVs + (j % TS) * 2 * TK * D + TK * D;
        for (int g = 0; g < NG; ++g) {
          const int i = g * 2 + (j & 1);
          tc5::barWait(pFull + i, (j >> 1) & 1);
          tc5::fenceAfter();
#pragma unroll
          for (int k = 0; k < TK / 16; ++k)   // O_g += P_g V: four K-16 steps over the keys
            tc5::mma(tm + 2 * NG * TK + g * D, tc5::sdesc(Ps + i * TQ * TK + k * 128, 128, (TK / 8) * 128),
                     tc5::sdesc(Vt + k * 128, 128, (TK / 8) * 128), ID2, j > 0 || k > 0);
          tc5::commit(pFree + i);
        }
        tc5::commit(kvFree + j % TS);
        if (j + 2 < nk) { kvReady(j + 2); for (int g = 0; g < NG; ++g) S(g, j + 2); }
      }
    }
  } else {
    const int g = warp >> 2, r = tid & 127, q = q0 + g * TQ + r;
    const uint32_t lane0 = (uint32_t)((warp & 3) * 32) << 16, tmO = tm + 2 * NG * TK + g * D;
    const float sl2 = (float)a.scale * LOG2E;
    const uint16_t* brow = bt.p && q < a.Q ? bt.p + h * bt.sh + (size_t)q * bt.ld : nullptr;
    // the row's 64 bias values (bf16, 128 bytes): the next tile's loaded as soon as this tile's are consumed, the rest of
    // the tile (its exponentials, P, the handoffs) covering the load
    uint4 bw[8];
    auto biasLoad = [&](int k0) {
#pragma unroll
      for (int j = 0; j < 8; ++j) bw[j] = brow ? __ldg(reinterpret_cast<const uint4*>(brow + k0) + j) : make_uint4(0, 0, 0, 0);
    };
    biasLoad(0);
    float mx = -INFINITY, sum = 0.f;   // (mx: the max O is scaled to - within 2^8 of the true one)
    for (int j = 0; j < nk; ++j) {
      const int s = j % TS, i = g * 2 + (j & 1);
      tc5::barWait(kvFull + s, (j / TS) & 1);   // (the key terms)
      tc5::barWait(sFull + i, (j >> 1) & 1);
      tc5::fenceAfter();
      float x[TK];
      tc5::ld32(tm + i * TK + lane0, x);
      tc5::ld32(tm + i * TK + lane0 + 32, x + 32);
      tc5::fenceBefore();
      __syncwarp();
      if (lane == 0) tc5::barArrive(sRead + i);
      const float* ka = KAs + s * TK;
      float rmax = -INFINITY;
#pragma unroll
      for (int u = 0; u < TK; ++u) {
        const uint32_t w = (&bw[u >> 3].x)[(u >> 1) & 3];
        const float bias = __uint_as_float(u & 1 ? w & 0xffff0000u : w << 16);
        x[u] = fmaf(x[u], sl2, fmaf(bias, LOG2E, ka[u]));
        rmax = fmaxf(rmax, x[u]);
      }
      if (j + 1 < nk) biasLoad((j + 1) * TK);
      // lazily: a new max only where this tile's passes the old by 2^8 (or where there was none)
      float alpha = 1.f;
      if (rmax > mx + LAZY || (mx == -INFINITY && rmax > -INFINITY)) {
        alpha = mx == -INFINITY ? 1.f : ex2(mx - rmax);
        mx = rmax;
      }
      const float nm = mx == -INFINITY ? 0.f : mx;
      float rs = 0.f;
#pragma unroll
      for (int u = 0; u < TK; ++u) { x[u] = ex2(x[u] - nm); rs += x[u]; }
      sum = sum * alpha + rs;
      if (j > 0 && __any_sync(~0u, alpha != 1.f)) {   // O current first: every O += P V of this tile so far
        const int pi = g * 2 + ((j - 1) & 1);
        tc5::barWait(pFree + pi, ((j - 1) >> 1) & 1);
        tc5::fenceAfter();
        float ov[D];
        tc5::ld32(tmO + lane0, ov);
#pragma unroll
        for (int u = 0; u < D; ++u) ov[u] *= alpha;
        tc5::st32(tmO + lane0, ov);
      }
      if (j >= 2) { tc5::barWait(pFree + i, ((j - 2) >> 1) & 1); tc5::fenceAfter(); }   // (this buffer's last P read)
      uint16_t* P = Ps + i * TQ * TK;
#pragma unroll
      for (int c = 0; c < TK / 8; ++c) {
        uint4 w;
        w.x = bf2(x[c * 8 + 0], x[c * 8 + 1]); w.y = bf2(x[c * 8 + 2], x[c * 8 + 3]);
        w.z = bf2(x[c * 8 + 4], x[c * 8 + 5]); w.w = bf2(x[c * 8 + 6], x[c * 8 + 7]);
        *reinterpret_cast<uint4*>(P + ((r >> 3) * (TK / 8) + c) * 64 + (r & 7) * 8) = w;
      }
      tc5::fenceProxy();
      tc5::fenceBefore();
      __syncwarp();
      if (lane == 0) tc5::barArrive(pFull + i);
    }
    const int li = g * 2 + ((nk - 1) & 1);
    tc5::barWait(pFree + li, ((nk - 1) >> 1) & 1);
    tc5::fenceAfter();
    float ov[D];
    tc5::ld32(tmO + lane0, ov);
    if (q < a.Q) {
      const size_t bh = (size_t)b * a.H + h, N = (size_t)a.B * a.H * a.Q, row = (size_t)b * a.Q + q;
      const float inv = sum > 0.f ? 1.f / sum : 0.f;
#pragma unroll
      for (int n = 0; n < D; n += 2) {
        const int c = h * D + n;
        const uint32_t w = bf2(ov[n] * inv, ov[n + 1] * inv);
        *reinterpret_cast<uint32_t*>(o + row * ldo + c) = w;
        if (gate) {
          uint32_t* gp = reinterpret_cast<uint32_t*>(gate + row * ldg + c);
          const uint32_t gw = *gp;
          const float s0 = __bfloat162float(__float2bfloat16(1.f / (1.f + __expf(-__uint_as_float(gw << 16))))),
                      s1 = __bfloat162float(__float2bfloat16(1.f / (1.f + __expf(-__uint_as_float(gw & 0xffff0000u)))));
          *gp = bf2(s0, s1);
          *reinterpret_cast<uint32_t*>(og + row * ldog + c) = bf2(__uint_as_float(w << 16) * s0, __uint_as_float(w & 0xffff0000u) * s1);
        }
      }
      lse[bh * a.Q + q] = sum > 0.f ? mx / LOG2E : -INFINITY;
      lse[N + bh * a.Q + q] = sum > 0.f ? logf(sum) : 0.f;
    }
  }
  tc5::fenceBefore();
  __syncthreads();
  if (warp == MW) {
    tc5::fenceAfter();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm));
  }
#endif
}
}  // namespace flash
}  // namespace grad

#if defined(LF_TC5A)
namespace grad {
namespace flash {
// ---- bwdTc: the grid attention's backward on tcgen05 - bwdBf's short form (D 32, no clip, chain ids or fixed point; the
// bias tile shared by the rows, any bias gradient a [h][q][k] one shared by the rows; Q and K multiples of 128). bwdBf was
// L1-bound (76% of its throughput: the ldmatrix operands, dS through shared memory, the atomics) at 46% issue.
// A CTA an SM a (row, head, 128-key block), 64-query tiles. S^T = K Q^T and dP^T = V dO^T into tensor memory (lane a key),
// double-buffered with P^T in its 512 columns, so tile i + 1's S and dP are issued before tile i's softmax ends and tile
// i's dV, dK, dQ run under tile i + 1's. The softmax a thread a key in two groups (a tile's queries 0-31, 32-63): p =
// 2^(s scale log2e + bias log2e + key term - c_q) and ds = p (dp - D_q), as bwdBf's - the general form only in a tile
// with a fully masked query (the producer's vote; both forms an element, or a branch an element, broke the overlap of
// the elements) - P^T in bf16 back into tensor memory (dV's A operand), dS^T into shared memory 128B-swizzled, the bias
// gradients by red.f32 from the float ds (a warp's 32 keys one line). dV += P^T dO and dK += dS^T Q (dO, Q the MN-major B
// operands - the same 64B-swizzled tiles S and dP read K-major), and every two tiles dQ = dS K (dS^T read MN-major: M the
// two tiles' 128 queries, K the keys). The loads all asynchronous, four stages (TMA; the raw lse, log-sum, D_i by bulk
// copies - blocking loads of them had starved the products). Needs cuda/af3's flash_tc5.cuh first (tc5a's helpers).
// Warps: 0 the loads, 1 the products, 4-11 the softmax, 12-15 dQ's adds. (Measured: each MMA ~108 cycles here against ~45
// alone - interference from the softmax's traffic on the SM. A two-CTAs-an-SM, single-buffered form with blocking per-query
// loads ran 3.60 ms at 768 in isolation against this one's 3.78-3.84; it was not kept or timed in a training step.)
namespace tcb {
constexpr int QT = 64, KB = 128, D = 32, WARPS = 16;
constexpr int NS = 4;   // (stages of a tile's loads)
struct Lay {   // K, V [128][32] (64B rows); dS^T [2][128][64] (128B rows); a softmax warp's {c_q, D_q, L_q, log2e lg_q} and
               // {c_q, D_q} of its 32 queries; [NS] stages { Q, dO [64][32]; the bias tile [64 q][128 k] as two 64-key boxes
               // (128B rows); the raw lse, log-sum, D_i of the tile's 64 queries }; barriers
  static constexpr uint32_t K = 0, V = 8192, DS = 16384, QW = 49152, QWS = 32 * 16 + 32 * 8, ST = 57344, SQ = 0, SO = 4096,
                            SB = 8192, SR = 24576, STS = 25600, BAR = ST + NS * STS, BYTES = BAR + 256;
};
__host__ __device__ constexpr uint32_t idescBf(int M, int N) { return tc5a::idesc(M, N) | (1u << 7) | (1u << 10); }
constexpr uint32_t A_MN = 1u << 15, B_MN = 1u << 16;   // (the instruction descriptor's operand majors: MN-major)
}
__device__ __forceinline__ void bulk(void* dst, const void* src, uint32_t bytes, uint64_t* m) {   // a 1-d bulk copy
#if LF_TC5A
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(tc5a::sa(dst)), "l"(src), "r"(bytes), "r"(tc5a::sa(m)) : "memory");
#endif
}
__device__ __forceinline__ void red1(float* p, float x) { asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(x) : "memory"); }
__device__ __forceinline__ void st8(uint32_t addr, const uint32_t* r) {
#if LF_TC5A
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
#endif
}
template <class DKV>
__global__ void __launch_bounds__(32 * tcb::WARPS, 1) bwdTc(const __grid_constant__ CUtensorMap mapQ,
    const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV, const __grid_constant__ CUtensorMap mapO,
    const __grid_constant__ CUtensorMap mapB, Sdpa a, bool perH, const float* __restrict__ lse, const float* __restrict__ Di,
    float* __restrict__ dq, int lddq, DKV* __restrict__ dk, int lddk, DKV* __restrict__ dv, int lddv, bool accK, bool accV,
    uint16_t* __restrict__ dqo, int ldqo, unsigned* __restrict__ cnt) {
#if LF_TC5A
  using namespace tc5a;
  constexpr int QT = tcb::QT, KB = tcb::KB, D = tcb::D; using L = tcb::Lay; using tcb::idescBf;
  // tensor memory: S^T [2], dP^T [2], dV, dK, dQ, P^T [2] (bf16 pairs)
  constexpr uint32_t CS = 0, CP = 128, CV = 256, CK = 288, CQ = 320, CT = 352;
  extern __shared__ __align__(1024) unsigned char smB[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(smB + L::BAR);
  constexpr int NS = tcb::NS;
  uint64_t *kvFull = bar, *tFull = bar + 1, *tFree = tFull + NS, *sFull = tFree + NS, *pFull = sFull + 2, *dsFree = pFull + 1,
           *dqFull = dsFree + 2, *dqFree = dqFull + 1, *done = dqFree + 1;
  __shared__ uint32_t tmBase;
  __shared__ int lastBlk;
  const int nkb = a.K / KB, nq = a.Q / QT;
  const int kb = blockIdx.x % nkb, h = (blockIdx.x / nkb) % a.H, b = blockIdx.x / (nkb * a.H), k0 = kb * KB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = sa(smB);
  const size_t bh = (size_t)b * a.H + h;
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) {
    barInit(kvFull, 1);
    for (int s = 0; s < NS; ++s) { barInit(tFull + s, 1); barInit(tFree + s, 9); }
    for (int s = 0; s < 2; ++s) { barInit(sFull + s, 1); barInit(dsFree + s, 1); }
    barInit(pFull, 8); barInit(dqFull, 1); barInit(dqFree, 4);
    barInit(done, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  fenceBefore();
  __syncthreads();
  fenceAfter();
  const uint32_t tm = tmBase;
  if (warp == 0) {   // the loads, all asynchronous (TMA, and bulk copies of the per-query terms)
    if (lane == 0) {
      const float* lseb = lse + bh * a.Q;
      const float* lgb = lseb + (size_t)a.B * a.H * a.Q;
      const float* Dib = Di + bh * a.Q;
      barExpect(kvFull, 16384);
      tma3(smB + L::K, &mapK, h * D, k0, b, kvFull);
      tma3(smB + L::V, &mapV, h * D, k0, b, kvFull);
      for (int i = 0; i < nq; ++i) {
        const int s = i % NS, q0 = i * QT;
        if (i >= NS) barWait(tFree + s, ((i / NS) - 1) & 1);
        unsigned char* st = smB + L::ST + s * L::STS;
        barExpect(tFull + s, 8192 + 16384 + 3 * QT * 4);
        tma3(st + L::SQ, &mapQ, h * D, q0, b, tFull + s);
        tma3(st + L::SO, &mapO, h * D, q0, b, tFull + s);
        tma3(st + L::SB, &mapB, k0, q0, perH ? h : 0, tFull + s);
        tma3(st + L::SB + 8192, &mapB, k0 + 64, q0, perH ? h : 0, tFull + s);
        bulk(st + L::SR, lseb + q0, QT * 4, tFull + s);
        bulk(st + L::SR + QT * 4, lgb + q0, QT * 4, tFull + s);
        bulk(st + L::SR + 2 * QT * 4, Dib + q0, QT * 4, tFull + s);
      }
    }
  } else if (warp == 1) {   // the products
    if (lane == 0) {
      constexpr uint32_t IDS = idescBf(128, QT), IDV = idescBf(128, D) | tcb::B_MN, IDQ = idescBf(128, D) | tcb::A_MN | tcb::B_MN;
      const uint64_t dK = sdesc(base + L::K, 16, 512, SW64), dV = sdesc(base + L::V, 16, 512, SW64);
      const uint64_t dKmn = sdesc(base + L::K, 16, 512, SW64);   // (MN-major, N = d: one 32-wide atom; 8 keys 512 B apart)
      const uint64_t dSmn = sdesc(base + L::DS, 16384, 1024, SW128);   // (MN-major: the two tiles' 64 queries 16 KB apart)
      barWait(kvFull, 0);
      fenceAfter();
      auto issueS = [&](int t) {   // S^T and dP^T of tile t into buffer t & 1 (its previous tile's softmax is done)
        const int s = t & 1, ts = t % NS;
        barWait(tFull + ts, (t / NS) & 1);
        fenceAfter();
        const uint32_t st = base + L::ST + ts * L::STS;
        const uint64_t dQ = sdesc(st + L::SQ, 16, 512, SW64), dO = sdesc(st + L::SO, 16, 512, SW64);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          mmaF16(tm + CS + s * 64, dK + (c * 32 >> 4), dQ + (c * 32 >> 4), IDS, c > 0);
          mmaF16(tm + CP + s * 64, dV + (c * 32 >> 4), dO + (c * 32 >> 4), IDS, c > 0);
        }
        commit(sFull + s);
      };
      issueS(0);
      for (int i = 0; i < nq; ++i) {
        const int s = i & 1;
        if (i + 1 < nq) issueS(i + 1);
        const uint32_t st = base + L::ST + (i % NS) * L::STS;
        const uint64_t dQ = sdesc(st + L::SQ, 16, 512, SW64), dO = sdesc(st + L::SO, 16, 512, SW64);
        barWait(pFull, i & 1);
        fenceAfter();
        const uint64_t dST = sdesc(base + L::DS + s * 16384, 16, 1024, SW128);
#pragma unroll
        for (int kk = 0; kk < QT / 16; ++kk) {
          mmaF16ts(tm + CV, tm + CT + s * 32 + kk * 8, dO + (kk * 1024 >> 4), IDV, i > 0 || kk > 0);
          mmaF16(tm + CK, dST + (kk * 32 >> 4), dQ + (kk * 1024 >> 4), IDV, i > 0 || kk > 0);
        }
        if (i & 1) {
          if (i >= 3) barWait(dqFree, ((i >> 1) - 1) & 1);
          fenceAfter();
#pragma unroll
          for (int kk = 0; kk < KB / 16; ++kk) mmaF16(tm + CQ, dSmn + (kk * 2048 >> 4), dKmn + (kk * 1024 >> 4), IDQ, kk > 0);
          commit(dqFull);
          commit(dsFree);       // (both dS^T regions read: by dK of tiles i - 1, i and by this dQ)
          commit(dsFree + 1);
        }
        commit(tFree + i % NS);
      }
      commit(done);
    }
  } else if (warp >= 4 && warp < 12) {   // the softmax: a thread a key, group g a tile's queries 32 g .. 32 g + 31
    const int g = (warp - 4) >> 2;
    const int kl = (warp & 3) * 32 + lane, kk_ = k0 + kl;
    const uint32_t lt = tm + ((uint32_t)((warp & 3) * 32) << 16);
    const float kaR = keyAdd(a, b, h, kk_), ckL = kaR * LOG2E, sl2 = a.scale * LOG2E, scale = a.scale;
    real* g0 = a.bias[0].grad ? a.bias[0].grad + h * a.bias[0].sh + kk_ : nullptr;
    real* g1 = a.bias[1].grad ? a.bias[1].grad + h * a.bias[1].sh + kk_ : nullptr;
    const long long sq0 = a.bias[0].sq, sq1 = a.bias[1].sq;
    const int kc = (kl & 63) >> 3, ke = kl & 7, sw = kl & 7;
    uint32_t bo[8];   // (the bias tile's swizzled offsets of this key, by query & 7: the reads then take immediates)
#pragma unroll
    for (int r = 0; r < 8; ++r) bo[r] = ((kc ^ r) << 4) + ke * 2;
    for (int i = 0; i < nq; ++i) {
      const int s = i & 1, q0 = i * QT;
      const int ts = i % NS;
      barWait(sFull + s, (i >> 1) & 1);
      barWait(tFull + ts, (i / NS) & 1);
      if (i >= 2) barWait(dsFree + s, ((i >> 1) - 1) & 1);
      fenceAfter();
      const unsigned char* stg = smB + L::ST + ts * L::STS;
      const unsigned char* T = stg + L::SB + (kl >> 6) * 8192;
      // this warp's 32 queries' terms from the raw ones (a query whose every key is out takes +inf: its every p is 0)
      float4* qv = reinterpret_cast<float4*>(smB + L::QW + (warp - 4) * L::QWS) - g * 32;   // (indexed by the tile's query)
      float2* qf = reinterpret_cast<float2*>(qv + g * 32 + 32) - g * 32;
      __syncwarp();   // (the previous tile's reads of them done)
      bool shortForm;
      {
        const int q = g * 32 + lane;
        const float* raw = reinterpret_cast<const float*>(stg + L::SR);
        const float l = raw[q], Lq = l != -INFINITY ? l : INFINITY, Lgq = LOG2E * raw[QT + q], cq = fmaf(Lq, LOG2E, Lgq), Dq = raw[2 * QT + q];
        qv[q] = make_float4(cq, Dq, Lq, Lgq);
        qf[q] = make_float2(cq, Dq);
        shortForm = __all_sync(~0u, Lq > -1e8f);
      }
      __syncwarp();
      unsigned char* DSrow = smB + L::DS + s * 16384 + kl * 128;
      auto tile = [&](auto SF) {
      constexpr bool SHORT = decltype(SF)::value;
#pragma unroll 1
      for (int hh = 0; hh < 2; ++hh) {   // (16 queries at a time; no branch inside: the bias gradients after - a branch an
                                         // element broke the overlap; the general form only in a tile that needs it)
        uint32_t sv[16], dp[16];
        const int qc = g * 32 + hh * 16;   // (the chunk's first query in the tile)
        ld16(lt + CS + s * 64 + qc, sv);
        ld16(lt + CP + s * 64 + qc, dp);
        ldWait();
        uint32_t pp[8], dd[8];
        float dr[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int q = qc + j;
          const uint16_t bb = *reinterpret_cast<const uint16_t*>(T + q * 128 + bo[q & 7]);
          const float bias = __uint_as_float((uint32_t)bb << 16);
          const float sc = __uint_as_float(sv[j]);
          float p, Dq;
          if constexpr (SHORT) {   // (bwdBf's short form: 2^(s scale log2e + (key term + bias) log2e - c_q))
            const float2 v = qf[q];
            p = ex2f(fmaf(sc, sl2, fmaf(bias, LOG2E, ckL - v.x)));
            Dq = v.y;
          } else {   // (a row whose every key is masked, max ~-1e9, takes the general form in natural units)
            const float4 v = qv[q];
            const float xs = fmaf(sc, sl2, fmaf(bias, LOG2E, ckL - v.x)), xg = fmaf(fmaf(sc, scale, kaR + bias) - v.z, LOG2E, -v.w);
            p = ex2f(v.z > -1e8f ? xs : xg);
            Dq = v.y;
          }
          dr[j] = p * (__uint_as_float(dp[j]) - Dq);
          sv[j] = __float_as_uint(p);
        }
#pragma unroll
        for (int j = 0; j < 16; j += 2) { pp[j >> 1] = bf2(__uint_as_float(sv[j]), __uint_as_float(sv[j + 1])); dd[j >> 1] = bf2(dr[j], dr[j + 1]); }
        if (g0) {
          real* gp = g0 + (long long)(q0 + qc) * sq0;
#pragma unroll
          for (int j = 0; j < 16; ++j, gp += sq0) red1(gp, dr[j]);
        }
        if (g1) {
          real* gp = g1 + (long long)(q0 + qc) * sq1;
#pragma unroll
          for (int j = 0; j < 16; ++j, gp += sq1) red1(gp, dr[j]);
        }
        st8(lt + CT + s * 32 + (qc >> 1), pp);   // (P^T into tensor memory: dV's A operand)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int ch = (((qc >> 3) + c) ^ sw) << 4;
          *reinterpret_cast<uint4*>(DSrow + ch) = make_uint4(dd[4 * c], dd[4 * c + 1], dd[4 * c + 2], dd[4 * c + 3]);
        }
      }
      };
      if (shortForm) tile(std::true_type{}); else tile(std::false_type{});
      stWait();
      fenceProxy();   // (dS^T, written by the generic proxy, visible to the tensor cores)
      fenceBefore();
      __syncwarp();
      if (lane == 0) { barArrive(pFull); barArrive(tFree + ts); }
    }
    barWait(done, 0);
    fenceAfter();
    uint32_t x[32];   // (group 0 writes dK, group 1 dV)
    ld32(lt + (g ? CV : CK), x);
    ldWait();
    DKV* pr = g ? dv + ((size_t)b * a.K + kk_) * lddv + h * D : dk + ((size_t)b * a.K + kk_) * lddk + h * D;
    const float m = g ? 1.f : scale;
    const bool acc = g ? accV : accK;
#pragma unroll
    for (int c = 0; c < D; c += 2) {
      float x0 = __uint_as_float(x[c]) * m, x1 = __uint_as_float(x[c + 1]) * m;
      if (acc) { x0 += ldOut(pr + c); x1 += ldOut(pr + c + 1); }
      st2(pr + c, x0, x1);
    }
  } else if (warp >= 12) {   // dQ: a thread a query of the two-tile block, its 32 columns added
    const int r = (warp & 3) * 32 + lane;
    const uint32_t ta = tm + ((uint32_t)((warp & 3) * 32) << 16) + CQ;
    float* dqb = dq + (size_t)b * a.Q * lddq + h * D;
    for (int p = 0; p < nq / 2; ++p) {
      barWait(dqFull, p & 1);
      fenceAfter();
      uint32_t x[32];
      ld32(ta, x);
      ldWait();
      fenceBefore();
      __syncwarp();
      if (lane == 0) barArrive(dqFree);
      float* row = dqb + (size_t)(p * 128 + r) * lddq;
#pragma unroll
      for (int c = 0; c < D; c += 4)
        redV4(row + c, __uint_as_float(x[c]) * a.scale, __uint_as_float(x[c + 1]) * a.scale, __uint_as_float(x[c + 2]) * a.scale,
              __uint_as_float(x[c + 3]) * a.scale);
    }
  }
  fenceBefore();
  __syncthreads();
  fenceAfter();
  if (warp == 1) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm));
  // dqo: the last of a (b, h)'s key blocks narrows its dQ into the bf16 copy (bwdBf's)
  if (dqo) {
    __threadfence();
    __syncthreads();
    if (tid == 0) lastBlk = atomicAdd(cnt + bh, 1u) == (unsigned)(nkb - 1);
    __syncthreads();
    if (lastBlk) {
      __threadfence();
      const float* dqb = dq + (size_t)b * a.Q * lddq + h * D;
      for (int i = tid; i < a.Q * (D / 4); i += blockDim.x) {
        const int qq = i / (D / 4), c = (i % (D / 4)) * 4;
        const float4 v = __ldcg(reinterpret_cast<const float4*>(dqb + (long long)qq * lddq + c));
        *reinterpret_cast<uint2*>(dqo + ((size_t)b * a.Q + qq) * ldqo + h * D + c) = make_uint2(bf2(v.x, v.y), bf2(v.z, v.w));
      }
    }
  }
#endif
}
// ON by default where it applies (--no-tc-bwd: bwdBf)
inline bool TCB = true;
// backwardBf's short form for bwdTc: D 32, bf16 q, k, v, dO rows (ld a multiple of 8), the bias tile shared by the rows,
// Q and K multiples of 128, every bias gradient a q-dependent one shared by the rows
template <class DKV>
inline bool bwdTcRun(const Sdpa& a, const BiasTile& bt, const __nv_bfloat16* q, int ldq, const __nv_bfloat16* k, int ldk,
                     const __nv_bfloat16* v, int ldv, const __nv_bfloat16* dO, int lddo, const float* lse, const float* Di,
                     float* dq, int lddq, DKV* dk, int lddk, DKV* dv, int lddv, bool accK, bool accV, uint16_t* dqo, int ldqo,
                     unsigned* cnt) {
  using tcb::QT; using tcb::KB;
  if (!TCB || !tc5a::ready() || a.D != 32 || a.Q % 128 || a.K % 128 || !bt.p || bt.sb != 0 || ldq % 8 || ldk % 8 || ldv % 8 ||
      lddo % 8 || a.clip || a.seqQ || a.seqK || a.deterministic)
    return false;
  for (int j = 0; j < 2; ++j)
    if (a.bias[j].grad && (a.bias[j].sb != 0 || a.bias[j].sq == 0)) return false;
  if ((dqo && !cnt) || ((uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)dO) & 15) return false;
  const int Npq = a.Q, Npk = a.K;   // (multiples of 128: biasBf's 64-padded tile is exactly Q x K)
  const CUtensorMap mq = tc5a::map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, q, { (uint64_t)ldq, (uint64_t)a.Q, (uint64_t)a.B }, { (uint64_t)ldq * 2, (uint64_t)a.Q * ldq * 2 }, { 32, QT, 1 }, CU_TENSOR_MAP_SWIZZLE_64B);
  const CUtensorMap mo = tc5a::map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, dO, { (uint64_t)lddo, (uint64_t)a.Q, (uint64_t)a.B }, { (uint64_t)lddo * 2, (uint64_t)a.Q * lddo * 2 }, { 32, QT, 1 }, CU_TENSOR_MAP_SWIZZLE_64B);
  const CUtensorMap mk = tc5a::map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, k, { (uint64_t)ldk, (uint64_t)a.K, (uint64_t)a.B }, { (uint64_t)ldk * 2, (uint64_t)a.K * ldk * 2 }, { 32, KB, 1 }, CU_TENSOR_MAP_SWIZZLE_64B);
  const CUtensorMap mv = tc5a::map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, v, { (uint64_t)ldv, (uint64_t)a.K, (uint64_t)a.B }, { (uint64_t)ldv * 2, (uint64_t)a.K * ldv * 2 }, { 32, KB, 1 }, CU_TENSOR_MAP_SWIZZLE_64B);
  const uint64_t Hh = bt.sh ? (uint64_t)a.H : 1;
  const CUtensorMap mb = tc5a::map3(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, bt.p, { (uint64_t)Npk, (uint64_t)Npq, Hh }, { (uint64_t)bt.ld * 2, (uint64_t)Npq * bt.ld * 2 }, { 64, QT, 1 }, CU_TENSOR_MAP_SWIZZLE_128B);
  static bool attr = false;
  if (!attr) { GK(cudaFuncSetAttribute(bwdTc<DKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tcb::Lay::BYTES)); attr = true; }
  const unsigned grid = (unsigned)((size_t)a.B * a.H * (a.K / KB));
  bwdTc<DKV><<<grid, 32 * tcb::WARPS, tcb::Lay::BYTES, STREAM>>>(mq, mk, mv, mo, mb, a, bt.sh != 0, lse, Di, dq, lddq, dk, lddk, dv, lddv,
                                                                accK, accV, dqo, ldqo, cnt);
  GK(cudaGetLastError());
  return true;
}
}  // namespace flash
}  // namespace grad
#endif
