// native/af3's FlashAttention-2 grid kernel (flash.cuh: flashGridHalf), copied with its rows and
// positions at arbitrary strides - so an attention ACROSS a tensor's leading axis (the MSA's columns,
// the triangle's ending node) reads the tensor where it lies instead of a transposed copy. Kept apart
// from native/af3's, whose register budget a stride parameter would cost (it sits one register from
// losing a block an SM).
#pragma once
#include "../../af3/src/flash.cuh"

template <int D, int WARPS, bool MASKED = true, int BK = FA_BK>
__global__ void __launch_bounds__(WARPS * 32) flashStrided(const half* __restrict__ qkvg, const half* __restrict__ bias,
    int biasStride, const float* __restrict__ mask, half* __restrict__ out, int n, int heads, float scale,
    size_t rowStride, size_t posStride, size_t outRowStride, size_t outPosStride) {
  const size_t r0 = 0; const bool tr = false; const float* qBias = nullptr;
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
  const half* base = qkvg + rl * rowStride + h * D;
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
    kvSrc[k] = base + (size_t)jj * posStride + Wd + c;
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
      const half* src = ok ? kvSrc[k] + (size_t)j0 * posStride : base;
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
    half2 v = *reinterpret_cast<const half2*>(base + (size_t)i * posStride + e);
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
  }
  float l0 = lsum[0], l1 = lsum[2];
  // gates read first and stored as pairs: with a store between every load the compiler
  // cannot reorder (out may alias qkvg) and the epilogue was 24 serial round trips, 10 us
  half2 ga[D / 8], gb[D / 8];
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    ga[et] = i0 < n ? *reinterpret_cast<const half2*>(base + (size_t)i0 * posStride + 3 * Wd + e) : half2{};
    gb[et] = i1 < n ? *reinterpret_cast<const half2*>(base + (size_t)i1 * posStride + 3 * Wd + e) : half2{};
  }
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    float2 a = __half22float2(ga[et]), b2 = __half22float2(gb[et]);
    if (i0 < n) *reinterpret_cast<half2*>(out + rl * outRowStride + (size_t)i0 * outPosStride + h * D + e) =
        __floats2half2_rn(o[et][0] / l0 * sigm(a.x), o[et][1] / l0 * sigm(a.y));
    if (i1 < n) *reinterpret_cast<half2*>(out + rl * outRowStride + (size_t)i1 * outPosStride + h * D + e) =
        __floats2half2_rn(o[et][2] / l1 * sigm(b2.x), o[et][3] / l1 * sigm(b2.y));
  }
}


template <int D, int WARPS, bool MASKED>
void flashStridedAt(const half* qkvg, const half* bias, int stride, const float* mask, half* out, int n, int heads,
                    size_t rows, float scale, size_t rowStride, size_t posStride, size_t outRowStride, size_t outPosStride) {
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(flashStrided<D, WARPS, MASKED>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(2 * faStage<D, WARPS>())));
    attr = true;
  }
  dim3 grid((n + 16 * WARPS - 1) / (16 * WARPS), (unsigned)(rows * heads));
  flashStrided<D, WARPS, MASKED><<<grid, 32 * WARPS, 2 * faStage<D, WARPS>(), STREAM>>>(
    qkvg, bias, stride, mask, out, n, heads, scale, rowStride, posStride, outRowStride, outPosStride);
}
// rows x (n positions) of a [.., 4W] qkvg at the given strides (elements); out likewise ([.., W])
inline void flashGridStrided(const half* qkvg, const half* bias, int stride, const float* mask, half* out, int n, int heads,
                             int D, size_t rows, float scale, size_t rowStride, size_t posStride, size_t outRowStride,
                             size_t outPosStride) {
  auto go = [&](auto dTag) {
    constexpr int DD = decltype(dTag)::value;
    if (mask) flashStridedAt<DD, 4, true>(qkvg, bias, stride, mask, out, n, heads, rows, scale, rowStride, posStride, outRowStride, outPosStride);
    else flashStridedAt<DD, 4, false>(qkvg, bias, stride, mask, out, n, heads, rows, scale, rowStride, posStride, outRowStride, outPosStride);
  };
  if (D == 32) go(std::integral_constant<int, 32>{});
  else if (D == 16) go(std::integral_constant<int, 16>{});
  else { fprintf(stderr, "flashGridStrided: no kernel for head width %d\n", D); exit(1); }
}
