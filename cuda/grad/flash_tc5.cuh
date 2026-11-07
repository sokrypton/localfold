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
}  // namespace tc5

constexpr int TQ = 128, TK = 64;   // queries a block, keys a tile
constexpr int TS = 3;               // K and V stages: tile it + 1 loading from the top of tile it (the stage it takes
                                    // held it - 2, whose O += P V was waited for in it - 1)
// shared memory: Q [128 x 32], TS stages of K and V [64 x 32], P [128 x 64] (bf16), TS stages of the key terms
constexpr size_t gridFwdTcSmem() { return (size_t)TQ * 32 * 2 + TS * 2 * TK * 32 * 2 + (size_t)TQ * TK * 2 + TS * TK * 4 + 64; }

__global__ void __launch_bounds__(128) gridFwdTc(Sdpa a, BiasTile bt, HSrc qs_, HSrc ks_, HSrc vs_,
                                                 __nv_bfloat16* __restrict__ o, int ldo, float* __restrict__ lse,
                                                 __nv_bfloat16* __restrict__ gate, int ldg, __nv_bfloat16* __restrict__ og,
                                                 int ldog, const float* __restrict__ ka2, int kpad) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  constexpr int D = 32;
  extern __shared__ __align__(1024) unsigned char tsm[];
  uint16_t* Qs = reinterpret_cast<uint16_t*>(tsm);                        // core matrices ((q / 8) 4 + d / 8)
  uint16_t* KVs = Qs + TQ * D;                                            // [stage][K, V][64 x 32]
  uint16_t* Ps = KVs + TS * 2 * TK * D;                                   // core matrices ((q / 8) 8 + k / 8)
  float* KAs = reinterpret_cast<float*>(Ps + TQ * TK);                    // [stage][64]
  uint64_t* bar = reinterpret_cast<uint64_t*>(KAs + TS * TK);             // S's, O's
  __shared__ uint32_t tmBase;
  const int nqb = (a.Q + TQ - 1) / TQ;
  const int qb = blockIdx.x % nqb, h = (blockIdx.x / nqb) % a.H, b = blockIdx.x / (nqb * a.H);
  const int tid = threadIdx.x, warp = tid >> 5;
  const int q0 = qb * TQ, q = q0 + tid;
  const float sl2 = (float)a.scale * LOG2E;
  // a 16-byte chunk (row r, columns 8c ..) of a row-major [rows][32] source into a core-matrix layout
  auto kOff = [](int r, int c) { return ((r >> 3) * 4 + c) * 64 + (r & 7) * 8; };          // K-major, 4 chunks a row group
  auto vOff = [](int r, int c) { return (c * (TK / 8) + (r >> 3)) * 64 + (r & 7) * 8; };   // V: MN-major, k-groups adjacent
  // Q: 128 rows x 4 chunks, 4 a thread
  for (int i = tid; i < TQ * 4; i += 128) {
    const int r = i >> 2, c = i & 3;
    const bool on = q0 + r < qs_.N && c * 8 < qs_.W;
    cp16(Qs + kOff(r, c), on ? hrow(qs_, b, h, q0 + r) + c * 8 : qs_.p, on);
  }
  auto issue = [&](int st, int k0) {
    uint16_t* Kt = KVs + st * 2 * TK * D; uint16_t* Vt = Kt + TK * D;   // (st: a stage)
    for (int i = tid; i < TK * 4; i += 128) {
      const int r = i >> 2, c = i & 3;
      const bool on = k0 + r < ks_.N && c * 8 < ks_.W;
      cp16(Kt + kOff(r, c), on ? hrow(ks_, b, h, k0 + r) + c * 8 : ks_.p, on);
      cp16(Vt + vOff(r, c), on ? hrow(vs_, b, h, k0 + r) + c * 8 : vs_.p, on);
    }
    if (tid < TK / 4) cp16(KAs + st * TK + tid * 4, ka2 + (size_t)b * kpad + k0 + tid * 4, true);
  };
  issue(0, 0);
  cpCommit();
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" :: "r"(tc5::sa(&tmBase)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) { tc5::barInit(bar); tc5::barInit(bar + 1); }
  tc5::fenceBefore();
  __syncthreads();
  tc5::fenceAfter();
  const uint32_t tm = tmBase, tmS = tm, tmO = tm + TK, lane0 = (uint32_t)(warp * 32) << 16;
  const uint16_t* brow = bt.p && q < a.Q ? bt.p + h * bt.sh + (size_t)q * bt.ld : nullptr;
  float mx = -INFINITY, sum = 0.f;
  constexpr uint32_t ID1 = tc5::idesc(TQ, TK, false), ID2 = tc5::idesc(TQ, D, true);
  const int nk = (a.K + TK - 1) / TK;
  // the row's 64 bias values (bf16, 128 bytes) a tile ahead: loaded while the tile before computes
  uint4 bw[8], bn[8];
  auto biasLoad = [&](uint4* r, int k0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = brow ? __ldg(reinterpret_cast<const uint4*>(brow + k0) + j) : make_uint4(0, 0, 0, 0);
  };
  biasLoad(bn, 0);
  for (int it = 0; it < nk; ++it) {
    const int st = it % TS, k0 = it * TK;
    tc5::cpWait0();
    tc5::fenceProxy();
    __syncthreads();
    if (it + 1 < nk) issue((it + 1) % TS, k0 + TK);
    cpCommit();
#pragma unroll
    for (int j = 0; j < 8; ++j) bw[j] = bn[j];
    if (it + 1 < nk) biasLoad(bn, k0 + TK);
    const uint16_t* Kt = KVs + st * 2 * TK * D;
    const uint16_t* Vt = Kt + TK * D;
    if (tid == 0) {   // S = Q K^T: two K-16 steps (256 bytes on in both operands)
      tc5::fenceAfter();
#pragma unroll
      for (int s = 0; s < D / 16; ++s)
        tc5::mma(tmS, tc5::sdesc(Qs + s * 128, 128, 512), tc5::sdesc(Kt + s * 128, 128, 512), ID1, s > 0);
      tc5::commit(bar);
    }
    tc5::barWait(bar, it & 1);
    tc5::fenceAfter();
    float x[TK];
    tc5::ld32(tmS + lane0, x);
    tc5::ld32(tmS + lane0 + 32, x + 32);
    const float* ka = KAs + st * TK;
    float rmax = -INFINITY;
#pragma unroll
    for (int j = 0; j < TK; ++j) {
      const uint32_t w = (&bw[j >> 3].x)[(j >> 1) & 3];
      const float bias = __uint_as_float(j & 1 ? w & 0xffff0000u : w << 16);
      x[j] = fmaf(x[j], sl2, fmaf(bias, LOG2E, ka[j]));
      rmax = fmaxf(rmax, x[j]);
    }
    const float m = fmaxf(mx, rmax);
    const float alpha = m == -INFINITY ? 1.f : ex2(mx - m), nm = m == -INFINITY ? 0.f : m;
    mx = m;
    float rs = 0.f;
#pragma unroll
    for (int j = 0; j < TK; ++j) { x[j] = ex2(x[j] - nm); rs += x[j]; }
    sum = sum * alpha + rs;
    // the previous O += P V done: P's buffer and O free
    if (it > 0) { tc5::barWait(bar + 1, (it - 1) & 1); tc5::fenceAfter(); }
    if (it > 0 && __any_sync(~0u, alpha != 1.f)) {
      float ov[D];
      tc5::ld32(tmO + lane0, ov);
#pragma unroll
      for (int j = 0; j < D; ++j) ov[j] *= alpha;
      tc5::st32(tmO + lane0, ov);
    }
    // P in bf16 into its core matrices: 8 chunks of 8 keys, this thread's row
#pragma unroll
    for (int c = 0; c < TK / 8; ++c) {
      uint4 w;
      w.x = bf2(x[c * 8 + 0], x[c * 8 + 1]); w.y = bf2(x[c * 8 + 2], x[c * 8 + 3]);
      w.z = bf2(x[c * 8 + 4], x[c * 8 + 5]); w.w = bf2(x[c * 8 + 6], x[c * 8 + 7]);
      *reinterpret_cast<uint4*>(Ps + ((tid >> 3) * (TK / 8) + c) * 64 + (tid & 7) * 8) = w;
    }
    tc5::fenceProxy();
    tc5::fenceBefore();
    __syncthreads();
    if (tid == 0) {   // O += P V: four K-16 steps over the keys (two k-groups, 256 bytes, on in both)
      tc5::fenceAfter();
#pragma unroll
      for (int s = 0; s < TK / 16; ++s)
        tc5::mma(tmO, tc5::sdesc(Ps + s * 128, 128, (TK / 8) * 128), tc5::sdesc(Vt + s * 128, 128, (TK / 8) * 128), ID2,
                 it > 0 || s > 0);
      tc5::commit(bar + 1);
    }
  }
  tc5::barWait(bar + 1, (nk - 1) & 1);
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
  tc5::fenceBefore();
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(tm));
#endif
}
}  // namespace flash
}  // namespace grad
