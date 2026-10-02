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
// transposed for the column direction; -1e9 where it is zero.
template <int D, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) flashGridHalf(const half* qkvg, const half* bias,
    int biasStride, const float* mask, half* out, int n, int heads, size_t r0, bool tr, float scale) {
  constexpr int BQ = 16 * WARPS, BK = FA_BK, LDK = D + 8, LDB = BK + 8, NT = WARPS * 32;
  constexpr size_t STAGE = faStage<D, WARPS>();
  extern __shared__ __align__(16) unsigned char smem[];
  auto Kst = [&](int s) { return (half*)(smem + s * STAGE); };
  auto Vst = [&](int s) { return Kst(s) + BK * LDK; };
  auto Bst = [&](int s) { return Vst(s) + BK * LDK; };
  auto Mst = [&](int s) { return (float*)(Bst(s) + BQ * LDB); };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t rl = b / heads, r = r0 + rl;
  const int Wd = heads * D, W4 = 4 * Wd;
  const half* base = qkvg + rl * (size_t)n * W4 + h * D;
  int q0 = blockIdx.x * BQ;
  auto issue = [&](int j0, int st) {
    half *K = Kst(st), *V = Vst(st), *B = Bst(st);
    for (int t = threadIdx.x; t < BK * (D / 8) * 2; t += NT) {
      int which = t / (BK * (D / 8)), u = t % (BK * (D / 8)), jj = u / (D / 8), c = (u % (D / 8)) * 8;
      int j = j0 + jj;
      cpAsync16((which ? V : K) + jj * LDK + c, base + (size_t)(j < n ? j : 0) * W4 + (which + 1) * Wd + c, j < n);
    }
    for (int t = threadIdx.x; t < BQ * (BK / 8); t += NT) {
      int qi = t / (BK / 8), c = (t % (BK / 8)) * 8, i = q0 + qi, j = j0 + c;
      bool ok = i < n && j < n;
      cpAsync16(B + qi * LDB + c, bias + ((size_t)h * n + (i < n ? i : 0)) * biasStride + (ok ? j : 0), ok);
    }
    if (threadIdx.x < BK) {
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
  issue(0, 0);
  for (int tile = 0; tile < tiles; ++tile) {
    int st = tile & 1;
    if (tile + 1 < tiles) { issue((tile + 1) * BK, st ^ 1); asm volatile("cp.async.wait_group 1;"); }
    else asm volatile("cp.async.wait_group 0;");
    __syncthreads();
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
    const half* br0 = B + (warp * 16 + g) * LDB;
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
    __syncthreads();
  }
  l0 += __shfl_xor_sync(~0u, l0, 1); l0 += __shfl_xor_sync(~0u, l0, 2);
  l1 += __shfl_xor_sync(~0u, l1, 1); l1 += __shfl_xor_sync(~0u, l1, 2);
  for (int et = 0; et < D / 8; ++et) {
    int e = et * 8 + tig * 2;
    if (i0 < n) {
      const half* gp = base + (size_t)i0 * W4 + 3 * Wd + e;
      half* op = out + (rl * n + i0) * Wd + h * D + e;
      op[0] = __float2half(o[et][0] / l0 * sigm(__half2float(gp[0])));
      op[1] = __float2half(o[et][1] / l0 * sigm(__half2float(gp[1])));
    }
    if (i1 < n) {
      const half* gp = base + (size_t)i1 * W4 + 3 * Wd + e;
      half* op = out + (rl * n + i1) * Wd + h * D + e;
      op[0] = __float2half(o[et][2] / l1 * sigm(__half2float(gp[0])));
      op[1] = __float2half(o[et][3] / l1 * sigm(__half2float(gp[1])));
    }
  }
}

// The precise path: one thread a query row, f32 throughout, key tiles of 64 in shared memory.
template <int D>
__global__ void flashGridF32(const float* qkvg, const float* bias, int biasStride, const float* mask,
                             float* out, int n, int heads, size_t r0, bool tr, float scale) {
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

template <int D, int WARPS> void setFlashSmem() {
  static bool done = false;
  if (!done) {
    CK(cudaFuncSetAttribute(flashGridHalf<D, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)(2 * faStage<D, WARPS>())));
    done = true;
  }
}
template <int D>
void flashGridHalfLaunch(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                         int n, int heads, size_t r0, size_t rows, bool tr, float scale) {
  bool wide = ((n + 127) / 128) * 128 - n <= 32;     // 128 queries a block unless that pads >32 rows
  if (wide) {
    setFlashSmem<D, 8>();
    flashGridHalf<D, 8><<<dim3((n + 127) / 128, (unsigned)(rows * heads)), 256, 2 * faStage<D, 8>(), STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
  } else {
    setFlashSmem<D, 4>();
    flashGridHalf<D, 4><<<dim3((n + 63) / 64, (unsigned)(rows * heads)), 128, 2 * faStage<D, 4>(), STREAM>>>(
      qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
  }
}
template <class T>
void flashGrid(const T* qkvg, const T* bias, int stride, const float* mask, T* out, int n, int heads,
               int D, size_t r0, size_t rows, bool tr, float scale) {
  if constexpr (std::is_same_v<T, half>) {
    if (D == 32) flashGridHalfLaunch<32>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale);
    else if (D == 16) flashGridHalfLaunch<16>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale);
    else { fprintf(stderr, "flashGrid: no f16 kernel for head width %d\n", D); exit(1); }
  } else {
    dim3 g((n + 63) / 64, (unsigned)(rows * heads));
    if (D == 32) flashGridF32<32><<<g, 64, 0, STREAM>>>(qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
    else if (D == 16) flashGridF32<16><<<g, 64, 0, STREAM>>>(qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
    else if (D == 24) flashGridF32<24><<<g, 64, 0, STREAM>>>(qkvg, bias, stride, mask, out, n, heads, r0, tr, scale);
    else { fprintf(stderr, "flashGrid: no f32 kernel for head width %d\n", D); exit(1); }
  }
}
