// The structure module and the confidence heads - af3-any-model's AF2 (alphafold3/af2/model/folding.py:
// generate_monomer_rigids, FoldIteration, InvariantPointAttention, MultiRigidSidechain; all_atom.py),
// float32. A rigid is 12 floats: the rotation row-major (xx xy xz yx yy yz zx zy zz), then the
// translation.
#pragma once
#include "ops.cuh"

__device__ inline void rapply(const float* r, const float* p, float* o) {      // R p + t
#pragma unroll
  for (int a = 0; a < 3; ++a)
    o[a] = ptx_fma_f32(r[a * 3], p[0], ptx_fma_f32(r[a * 3 + 1], p[1], ptx_fma_f32(r[a * 3 + 2], p[2], r[9 + a])));
}
__device__ inline void rapplyInv(const float* r, const float* p, float* o) {   // R^T (p - t)
  float d[3] = {ptx_sub_f32(p[0], r[9]), ptx_sub_f32(p[1], r[10]), ptx_sub_f32(p[2], r[11])};
#pragma unroll
  for (int a = 0; a < 3; ++a)
    o[a] = ptx_fma_f32(r[a], d[0], ptx_fma_f32(r[3 + a], d[1], ptx_mul_f32(r[6 + a], d[2])));
}
__device__ inline void rcompose(const float* A, const float* B, float* C) {    // A @ B
  float R[9], t[3];
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      R[i * 3 + j] = ptx_fma_f32(A[i * 3], B[j], ptx_fma_f32(A[i * 3 + 1], B[3 + j], ptx_mul_f32(A[i * 3 + 2], B[6 + j])));
  rapply(A, B + 9, t);
#pragma unroll
  for (int k = 0; k < 9; ++k) C[k] = R[k];
#pragma unroll
  for (int k = 0; k < 3; ++k) C[9 + k] = t[k];
}

__global__ void identityRigidK(float* r, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  float* x = r + i * 12;
  ptx_st_global_v4_f32(x, make_float4(1.f, 0.f, 0.f, 0.f));
  ptx_st_global_v4_f32(x + 4, make_float4(1.f, 0.f, 0.f, 0.f));
  ptx_st_global_v4_f32(x + 8, make_float4(1.f, 0.f, 0.f, 0.f));
}
// points [L, H*3*P] from the projection (per head: x[0:P] y[P:2P] z[2P:3P]) -> global [L, H, P, 3]
__global__ void pointsToGlobalK(const float* proj, const float* rig, float* out, int L, int Hh, int Pp) {
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (unsigned)(L * Hh * Pp)) return;
  int p = (int)(t % (unsigned)Pp); unsigned r = t / (unsigned)Pp; int h = (int)(r % (unsigned)Hh); int i = (int)(r / (unsigned)Hh);
  const float* s = proj + (size_t)i * Hh * 3 * Pp + h * 3 * Pp;
  float loc[3] = {ptx_ldg_f32(s + p), ptx_ldg_f32(s + Pp + p), ptx_ldg_f32(s + 2 * Pp + p)}, g[3];
  float rg[12];
  const float* rbase = rig + i * 12;
  float4 r0 = ptx_ldg_v4_f32(rbase), r1 = ptx_ldg_v4_f32(rbase + 4), r2 = ptx_ldg_v4_f32(rbase + 8);
  rg[0] = r0.x; rg[1] = r0.y; rg[2] = r0.z; rg[3] = r0.w;
  rg[4] = r1.x; rg[5] = r1.y; rg[6] = r1.z; rg[7] = r1.w;
  rg[8] = r2.x; rg[9] = r2.y; rg[10] = r2.z; rg[11] = r2.w;
  rapply(rg, loc, g);
  for (int a = 0; a < 3; ++a) ptx_st_global_f32(out + (size_t)t * 3 + a, g[a]);
}
// IPA logits -> attention weights, a warp a (query, head): logits[q, k, h] then softmax over k
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__global__ void __launch_bounds__(256, 3)
#else
__global__ void __launch_bounds__(256, 4)
#endif
ipaWeightsK(const float* __restrict__ qs, const float* __restrict__ ks, const float* __restrict__ qp,
            const float* __restrict__ kp, const float* __restrict__ b2d, const float* __restrict__ pw,
            const float* __restrict__ seqMask, float* __restrict__ attn, int L, int Hh, int Cs, int Pq) {
  __shared__ float srow[8][512];
  unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (wid >= (unsigned)(L * Hh)) return;
  int h = (int)(wid % (unsigned)Hh), q = (int)(wid / (unsigned)Hh);
  float* row = attn + ((size_t)q * Hh + h) * L;     // [q, h, k]
  float mx = -INFINITY;
  float w = ptx_ldg_f32(pw + h);
  // the query's scalars and points held in registers, read once (they were read again for every key); the
  // keys' scalars as float4s where 16-byte aligned - the same products in the same order
  if (Cs == 16 && Pq == 4) {
    const float4* qsh = reinterpret_cast<const float4*>(qs + ((size_t)q * Hh + h) * 16);
    float4 qv0 = ptx_ldg_v4_f32(qsh), qv1 = ptx_ldg_v4_f32(qsh + 1), qv2 = ptx_ldg_v4_f32(qsh + 2), qv3 = ptx_ldg_v4_f32(qsh + 3);
    const float4* qph = reinterpret_cast<const float4*>(qp + ((size_t)q * Hh + h) * 12);
    float4 qp0 = ptx_ldg_v4_f32(qph), qp1 = ptx_ldg_v4_f32(qph + 1), qp2 = ptx_ldg_v4_f32(qph + 2);
    float sq = ptx_ldg_f32(seqMask + q);
    const float invSqrt3 = 0.5773502691896257645f;
    const float nw2 = ptx_mul_f32(-0.5f, w);
    for (int k = lane; k < L; k += 32) {
      const float4* kpk = reinterpret_cast<const float4*>(kp + ((size_t)k * Hh + h) * 12);
      float4 kp0 = ptx_ldg_v4_f32(kpk), kp1 = ptx_ldg_v4_f32(kpk + 1), kp2 = ptx_ldg_v4_f32(kpk + 2);
      float d2 = 0.f;
      float dx, dy, dz;
      dx = ptx_sub_f32(qp0.x, kp0.x); dy = ptx_sub_f32(qp0.y, kp0.y); dz = ptx_sub_f32(qp0.z, kp0.z);
      d2 = ptx_add_f32(d2, ptx_fma_f32(dz, dz, ptx_fma_f32(dy, dy, ptx_mul_f32(dx, dx))));
      dx = ptx_sub_f32(qp0.w, kp0.w); dy = ptx_sub_f32(qp1.x, kp1.x); dz = ptx_sub_f32(qp1.y, kp1.y);
      d2 = ptx_add_f32(d2, ptx_fma_f32(dz, dz, ptx_fma_f32(dy, dy, ptx_mul_f32(dx, dx))));
      dx = ptx_sub_f32(qp1.z, kp1.z); dy = ptx_sub_f32(qp1.w, kp1.w); dz = ptx_sub_f32(qp2.x, kp2.x);
      d2 = ptx_add_f32(d2, ptx_fma_f32(dz, dz, ptx_fma_f32(dy, dy, ptx_mul_f32(dx, dx))));
      dx = ptx_sub_f32(qp2.y, kp2.y); dy = ptx_sub_f32(qp2.z, kp2.z); dz = ptx_sub_f32(qp2.w, kp2.w);
      d2 = ptx_add_f32(d2, ptx_fma_f32(dz, dz, ptx_fma_f32(dy, dy, ptx_mul_f32(dx, dx))));
      const float4* ksk = reinterpret_cast<const float4*>(ks + ((size_t)k * Hh + h) * 16);
      float sc = 0.f;
      float4 v0 = ptx_ldg_v4_f32(ksk);
      sc = ptx_fma_f32(qv0.x, v0.x, sc); sc = ptx_fma_f32(qv0.y, v0.y, sc); sc = ptx_fma_f32(qv0.z, v0.z, sc); sc = ptx_fma_f32(qv0.w, v0.w, sc);
      float4 v1 = ptx_ldg_v4_f32(ksk + 1);
      sc = ptx_fma_f32(qv1.x, v1.x, sc); sc = ptx_fma_f32(qv1.y, v1.y, sc); sc = ptx_fma_f32(qv1.z, v1.z, sc); sc = ptx_fma_f32(qv1.w, v1.w, sc);
      float4 v2 = ptx_ldg_v4_f32(ksk + 2);
      sc = ptx_fma_f32(qv2.x, v2.x, sc); sc = ptx_fma_f32(qv2.y, v2.y, sc); sc = ptx_fma_f32(qv2.z, v2.z, sc); sc = ptx_fma_f32(qv2.w, v2.w, sc);
      float4 v3 = ptx_ldg_v4_f32(ksk + 3);
      sc = ptx_fma_f32(qv3.x, v3.x, sc); sc = ptx_fma_f32(qv3.y, v3.y, sc); sc = ptx_fma_f32(qv3.z, v3.z, sc); sc = ptx_fma_f32(qv3.w, v3.w, sc);
      float s = ptx_mul_f32(nw2, d2);
      s = ptx_add_f32(s, ptx_add_f32(sc, ptx_ldg_f32(b2d + ((size_t)q * L + k) * Hh + h)));
      s = ptx_fma_f32(-1e5f, ptx_fma_f32(-sq, ptx_ldg_f32(seqMask + k), 1.f), s);
      s = ptx_mul_f32(s, invSqrt3);
      if (L <= 512) ptx_st_shared_f32(&srow[warp][k], s);
      else ptx_st_global_f32(row + k, s);
      mx = ptx_max_f32(mx, s);
    }
    mx = ptx_warp_max_f32(mx);
    float sum = 0.f;
    if (L <= 512) {
      for (int k = lane; k < L; k += 32) {
        float e = ptx_exp_f32(ptx_sub_f32(ptx_ld_shared_f32(&srow[warp][k]), mx));
        ptx_st_shared_f32(&srow[warp][k], e);
        sum = ptx_add_f32(sum, e);
      }
      sum = ptx_warp_sum_f32(sum);
      for (int k = lane; k < L; k += 32) ptx_st_global_f32(row + k, ptx_div_f32(ptx_ld_shared_f32(&srow[warp][k]), sum));
    } else {
      for (int k = lane; k < L; k += 32) {
        float e = ptx_exp_f32(ptx_sub_f32(ptx_ld_global_f32(row + k), mx));
        ptx_st_global_f32(row + k, e);
        sum = ptx_add_f32(sum, e);
      }
      sum = ptx_warp_sum_f32(sum);
      for (int k = lane; k < L; k += 32) ptx_st_global_f32(row + k, ptx_div_f32(ptx_ld_global_f32(row + k), sum));
    }
    return;
  }
  for (int k = lane; k < L; k += 32) {
    float d2 = 0.f, sc = 0.f;
    for (int p = 0; p < Pq; ++p)
      for (int a = 0; a < 3; ++a) {
        float d = ptx_sub_f32(ptx_ldg_f32(qp + (((size_t)q * Hh + h) * Pq + p) * 3 + a),
                              ptx_ldg_f32(kp + (((size_t)k * Hh + h) * Pq + p) * 3 + a));
        d2 = ptx_fma_f32(d, d, d2);
      }
    for (int c = 0; c < Cs; ++c)
      sc = ptx_fma_f32(ptx_ldg_f32(qs + ((size_t)q * Hh + h) * Cs + c), ptx_ldg_f32(ks + ((size_t)k * Hh + h) * Cs + c), sc);
    float s = ptx_mul_f32(ptx_mul_f32(-0.5f, w), d2);
    s = ptx_add_f32(s, ptx_add_f32(sc, ptx_ldg_f32(b2d + ((size_t)q * L + k) * Hh + h)));
    s = ptx_fma_f32(-1e5f, ptx_fma_f32(-ptx_ldg_f32(seqMask + q), ptx_ldg_f32(seqMask + k), 1.f), s);
    s = ptx_mul_f32(s, 0.5773502691896257645f);
    ptx_st_global_f32(row + k, s);
    mx = ptx_max_f32(mx, s);
  }
  mx = ptx_warp_max_f32(mx);
  float sum = 0.f;
  for (int k = lane; k < L; k += 32) {
    float e = ptx_exp_f32(ptx_sub_f32(ptx_ld_global_f32(row + k), mx));
    ptx_st_global_f32(row + k, e);
    sum = ptx_add_f32(sum, e);
  }
  sum = ptx_warp_sum_f32(sum);
  for (int k = lane; k < L; k += 32) ptx_st_global_f32(row + k, ptx_div_f32(ptx_ld_global_f32(row + k), sum));
}
// IPA's scalar and point outputs when the pair term runs in ipaPairTermK: a warp a (query tile of 4, head),
// lanes 0..Cs-1 accumulating vs and lanes Cs..Cs+Pv-1 accumulating vp concurrently in one loop over k
__global__ void __launch_bounds__(128, 8)
ipaOutputsFastK(const float* __restrict__ attn, const float* __restrict__ vs, const float* __restrict__ vp,
                const float* __restrict__ rig, float* __restrict__ final_, int L, int Hh, int Cs, int Pv, int C2) {
  int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  if (Cs == 16 && Pv == 8) {
    int qTiles = (L + 3) >> 2;
    if (wid >= qTiles * Hh) return;
    int h = wid % Hh, q0 = (wid / Hh) << 2;
    int nq = ptx_min_s32(4, L - q0);
    __shared__ float sp[4][24];
    const float* p0 = lane < 16 ? (vs + h * 16 + lane) : (vp + h * 24 + (lane - 16));
    const float* p1 = vp + h * 24 + (lane >= 24 ? (lane - 8) : 0);
    int st0 = lane < 16 ? (Hh * 16) : (Hh * 24), st1 = Hh * 24;
    const size_t aStride = (size_t)Hh * L;
    const float* a0 = attn + ((size_t)q0 * Hh + h) * L;
    const float* a1 = (nq > 1) ? (a0 + aStride) : a0;
    const float* a2 = (nq > 2) ? (a0 + 2 * aStride) : a0;
    const float* a3 = (nq > 3) ? (a0 + 3 * aStride) : a0;
    float g0_0 = 0.f, g0_1 = 0.f, g0_2 = 0.f, g0_3 = 0.f;
    float g1_0 = 0.f, g1_1 = 0.f, g1_2 = 0.f, g1_3 = 0.f;
#pragma unroll 4
    for (int k = 0; k < L; ++k, p0 += st0, p1 += st1) {
      float v0 = ptx_ldg_f32(p0);
      float v1 = lane >= 24 ? ptx_ldg_f32(p1) : 0.f;
      float ak0 = ptx_ldg_f32(a0 + k);
      float ak1 = ptx_ldg_f32(a1 + k);
      float ak2 = ptx_ldg_f32(a2 + k);
      float ak3 = ptx_ldg_f32(a3 + k);
      g0_0 = ptx_fma_f32(ak0, v0, g0_0);
      g0_1 = ptx_fma_f32(ak1, v0, g0_1);
      g0_2 = ptx_fma_f32(ak2, v0, g0_2);
      g0_3 = ptx_fma_f32(ak3, v0, g0_3);
      if (lane >= 24) {
        g1_0 = ptx_fma_f32(ak0, v1, g1_0);
        g1_1 = ptx_fma_f32(ak1, v1, g1_1);
        g1_2 = ptx_fma_f32(ak2, v1, g1_2);
        g1_3 = ptx_fma_f32(ak3, v1, g1_3);
      }
    }
    float g0s[4] = {g0_0, g0_1, g0_2, g0_3};
    float g1s[4] = {g1_0, g1_1, g1_2, g1_3};
    for (int qi = 0; qi < nq; ++qi) {
      int q = q0 + qi;
      float* f = final_ + (size_t)q * Fw;
      if (lane < 16) ptx_st_global_f32(f + h * 16 + lane, g0s[qi]);
      else ptx_st_shared_f32(&sp[warp][lane - 16], g0s[qi]);
      if (lane >= 24) ptx_st_shared_f32(&sp[warp][lane - 8], g1s[qi]);
      ptx_bar_warp();
      if (lane < 8) {
        int p = lane;
        float g[3] = {ptx_ld_shared_f32(&sp[warp][p * 3]),
                      ptx_ld_shared_f32(&sp[warp][p * 3 + 1]),
                      ptx_ld_shared_f32(&sp[warp][p * 3 + 2])}, loc[3];
        float rg[12];
        const float* rbase = rig + q * 12;
        float4 r0 = ptx_ldg_v4_f32(rbase), r1 = ptx_ldg_v4_f32(rbase + 4), r2 = ptx_ldg_v4_f32(rbase + 8);
        rg[0] = r0.x; rg[1] = r0.y; rg[2] = r0.z; rg[3] = r0.w;
        rg[4] = r1.x; rg[5] = r1.y; rg[6] = r1.z; rg[7] = r1.w;
        rg[8] = r2.x; rg[9] = r2.y; rg[10] = r2.z; rg[11] = r2.w;
        rapplyInv(rg, g, loc);
        int base = Hh * 16, idx = h * 8 + p;
        ptx_st_global_f32(f + base + idx, loc[0]);
        ptx_st_global_f32(f + base + Hh * 8 + idx, loc[1]);
        ptx_st_global_f32(f + base + 2 * Hh * 8 + idx, loc[2]);
        ptx_st_global_f32(f + base + 3 * Hh * 8 + idx,
                          ptx_sqrt_f32(ptx_max_f32(ptx_fma_f32(loc[2], loc[2], ptx_fma_f32(loc[1], loc[1], ptx_mul_f32(loc[0], loc[0]))), 1e-16f)));
      }
      ptx_bar_warp();
    }
    return;
  }
  if (wid >= L * Hh) return;
  int q = wid / Hh, h = wid % Hh;
  const float* a = attn + (size_t)wid * L;
  float* f = final_ + (size_t)q * Fw;
  float s = 0.f, g0 = 0.f, g1 = 0.f, g2 = 0.f;
  int p = lane - Cs;
  bool isS = lane < Cs, isP = p >= 0 && p < Pv;
  if (!isS && !isP) return;
  if (isS) {
    const float* vsh = vs + h * Cs + lane;
    int strideS = Hh * Cs;
#pragma unroll 8
    for (int k = 0; k < L; ++k) s = ptx_fma_f32(ptx_ldg_f32(a + k), ptx_ldg_f32(vsh + (size_t)k * strideS), s);
    ptx_st_global_f32(f + h * Cs + lane, s);
  } else {
    const float* vph = vp + (h * Pv + p) * 3;
    int strideP = Hh * Pv * 3;
#pragma unroll 8
    for (int k = 0; k < L; ++k) {
      float ak = ptx_ldg_f32(a + k);
      const float* vpk = vph + (size_t)k * strideP;
      g0 = ptx_fma_f32(ak, ptx_ldg_f32(vpk), g0);
      g1 = ptx_fma_f32(ak, ptx_ldg_f32(vpk + 1), g1);
      g2 = ptx_fma_f32(ak, ptx_ldg_f32(vpk + 2), g2);
    }
    float g[3] = {g0, g1, g2}, loc[3];
    rapplyInv(rig + q * 12, g, loc);
    int base = Hh * Cs, idx = h * Pv + p;
    ptx_st_global_f32(f + base + idx, loc[0]);
    ptx_st_global_f32(f + base + Hh * Pv + idx, loc[1]);
    ptx_st_global_f32(f + base + 2 * Hh * Pv + idx, loc[2]);
    ptx_st_global_f32(f + base + 3 * Hh * Pv + idx,
                      ptx_sqrt_f32(ptx_max_f32(ptx_fma_f32(loc[2], loc[2], ptx_fma_f32(loc[1], loc[1], ptx_mul_f32(loc[0], loc[0]))), 1e-16f)));
  }
}
// IPA's outputs, a block a (query, head): [result_scalar | point local x | y | z | norms | over 2d]
// written into final [L, 2112] at the reference's concatenation offsets
__global__ void ipaOutputsK(const float* attn, const float* vs, const float* vp, const float* act2d, const float* rig,
                            float* final_, int L, int Hh, int Cs, int Pv, int C2) {
  int q = blockIdx.x / Hh, h = blockIdx.x % Hh;
  const float* a = attn + ((size_t)q * Hh + h) * L;
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  float* f = final_ + (size_t)q * Fw;
  for (int c = threadIdx.x; c < Cs; c += blockDim.x) {
    float s = 0.f;
    for (int k = 0; k < L; ++k) s = ptx_fma_f32(ptx_ldg_f32(a + k), ptx_ldg_f32(vs + ((size_t)k * Hh + h) * Cs + c), s);
    ptx_st_global_f32(f + h * Cs + c, s);
  }
  for (int p = threadIdx.x; p < Pv; p += blockDim.x) {
    float g[3] = {0.f, 0.f, 0.f};
    for (int k = 0; k < L; ++k) {
      float ak = ptx_ldg_f32(a + k);
      for (int x = 0; x < 3; ++x) g[x] = ptx_fma_f32(ak, ptx_ldg_f32(vp + (((size_t)k * Hh + h) * Pv + p) * 3 + x), g[x]);
    }
    float loc[3];
    rapplyInv(rig + q * 12, g, loc);
    int base = Hh * Cs, idx = h * Pv + p;
    ptx_st_global_f32(f + base + idx, loc[0]);
    ptx_st_global_f32(f + base + Hh * Pv + idx, loc[1]);
    ptx_st_global_f32(f + base + 2 * Hh * Pv + idx, loc[2]);
    ptx_st_global_f32(f + base + 3 * Hh * Pv + idx,
                      ptx_sqrt_f32(ptx_max_f32(ptx_fma_f32(loc[2], loc[2], ptx_fma_f32(loc[1], loc[1], ptx_mul_f32(loc[0], loc[0]))), 1e-16f)));
  }
  for (int c = threadIdx.x; act2d && c < C2; c += blockDim.x) {     // (no act2d: ipaPairOutputsK writes it)
    float s = 0.f;
    for (int k = 0; k < L; ++k) s = ptx_fma_f32(ptx_ldg_f32(a + k), ptx_ldg_f32(act2d + ((size_t)q * L + k) * C2 + c), s);
    ptx_st_global_f32(f + Hh * Cs + 4 * Hh * Pv + h * C2 + c, s);
  }
}
// ipaOutputsK's pair term, a block a QUERY for every head at once: each act2d element is read once and feeds all Hh
// heads' sums (a block a (query, head) read the query's pair row once per head - 12 times, ~1.5 GB a call at 494
// residues). Each sum runs over k in the same order with the same arithmetic, so the outputs are the same bytes.
constexpr int IPA_MAX_HEADS = 16, IPA_KT = 256;
__global__ void __launch_bounds__(128, 6)
ipaPairTermK(const float* __restrict__ attn, const float* __restrict__ act2d, float* __restrict__ final_,
             int L, int Hh, int Cs, int Pv, int C2) {
  __shared__ float as[IPA_MAX_HEADS][IPA_KT];
  int q = blockIdx.x;
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  if (C2 == 128 && Hh == 12 && blockDim.x == 128) {
    int c = threadIdx.x;
    float acc[12] = {};
    for (int k0 = 0; k0 < L; k0 += IPA_KT) {
      int kt = ptx_min_s32(IPA_KT, L - k0);
#pragma unroll
      for (int h = 0; h < 12; ++h) {
        const float* ah = attn + ((size_t)q * 12 + h) * L + k0;
        if (c < kt) ptx_st_shared_f32(&as[h][c], ptx_ldg_f32(ah + c));
        if (c + 128 < kt) ptx_st_shared_f32(&as[h][c + 128], ptx_ldg_f32(ah + c + 128));
      }
      ptx_bar_sync();
      const float* a2d = act2d + ((size_t)q * L + k0) * 128 + c;
      int k = 0;
      for (; k + 3 < kt; k += 4, a2d += 512) {
        float v0 = ptx_ldg_f32(a2d);
        float v1 = ptx_ldg_f32(a2d + 128);
        float v2 = ptx_ldg_f32(a2d + 256);
        float v3 = ptx_ldg_f32(a2d + 384);
        float vs[4] = {v0, v1, v2, v3};
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
          for (int h = 0; h < 12; ++h) acc[h] = ptx_fma_f32(ptx_ld_shared_f32(&as[h][k + u]), vs[u], acc[h]);
        }
      }
      for (; k < kt; ++k, a2d += 128) {
        float v = ptx_ldg_f32(a2d);
#pragma unroll
        for (int h = 0; h < 12; ++h) acc[h] = ptx_fma_f32(ptx_ld_shared_f32(&as[h][k]), v, acc[h]);
      }
      ptx_bar_sync();
    }
    float* f = final_ + (size_t)q * Fw + 12 * Cs + 48 * Pv + c;
#pragma unroll
    for (int h = 0; h < 12; ++h) ptx_st_global_f32(f + h * 128, acc[h]);
    return;
  }
  int c = threadIdx.x;
  float acc[IPA_MAX_HEADS];
#pragma unroll
  for (int h = 0; h < IPA_MAX_HEADS; ++h) acc[h] = 0.f;
  for (int k0 = 0; k0 < L; k0 += IPA_KT) {
    int kt = ptx_min_s32(IPA_KT, L - k0);
    ptx_bar_sync();
    for (int t = threadIdx.x; t < Hh * kt; t += blockDim.x) {
      int h = t / kt, k = t % kt;
      ptx_st_shared_f32(&as[h][k], ptx_ldg_f32(attn + ((size_t)q * Hh + h) * L + k0 + k));
    }
    ptx_bar_sync();
    if (c < C2)
#pragma unroll 8
      for (int k = 0; k < kt; ++k) {
        float v = ptx_ldg_f32(act2d + ((size_t)q * L + k0 + k) * C2 + c);
#pragma unroll
        for (int h = 0; h < IPA_MAX_HEADS; ++h)
          if (h < Hh) acc[h] = ptx_fma_f32(ptx_ld_shared_f32(&as[h][k]), v, acc[h]);
      }
  }
  if (c < C2) {
    float* f = final_ + (size_t)q * Fw;
    for (int h = 0; h < Hh; ++h) ptx_st_global_f32(f + Hh * Cs + 4 * Hh * Pv + h * C2 + c, acc[h]);
  }
}
// a pair position's LayerNorm statistics, by layerNormK's own arithmetic (two passes, a warp a row)
__global__ void lnStatsK(const float* __restrict__ x, float2* __restrict__ stats, size_t rows, int C) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  if (C == 128) {
    float xv[4], s = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) { float val = ptx_ldg_f32(xr + lane + 32 * k); xv[k] = val; s = ptx_add_f32(s, val); }
    s = ptx_warp_sum_f32(s);
    float mean = ptx_mul_f32(s, 1.f / 128.f), v = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) { float d = ptx_sub_f32(xv[k], mean); v = ptx_fma_f32(d, d, v); }
    v = ptx_warp_sum_f32(v);
    if (lane == 0) ptx_st_global_v2_f32(stats + row, make_float2(mean, ptx_rsqrt_f32(ptx_fma_f32(v, 1.f / 128.f, 1e-5f))));
    return;
  }
  float s = 0.f;
  for (int c = lane; c < C; c += 32) s = ptx_add_f32(s, ptx_ldg_f32(xr + c));
  s = ptx_warp_sum_f32(s);
  float mean = ptx_div_f32(s, (float)C), v = 0.f;
  for (int c = lane; c < C; c += 32) { float d = ptx_sub_f32(ptx_ldg_f32(xr + c), mean); v = ptx_fma_f32(d, d, v); }
  v = ptx_warp_sum_f32(v);
  if (lane == 0) ptx_st_global_v2_f32(stats + row, make_float2(mean, ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(v, (float)C), 1e-5f))));
}
// ipaOutputsK's pair term with act2d made where it is read: (pair - mean) * inv * scale + offset, the
// LayerNorm's own arithmetic, so the same values as the stored act2d - which on a card short of room is
// not stored (a pair-sized f32 tensor)
__global__ void ipaPairOutputsK(const float* attn, const float* pair, const float2* stats, const float* scale,
                                const float* offset, float* final_, int L, int Hh, int Cs, int Pv, int C2) {
  int q = blockIdx.x / Hh, h = blockIdx.x % Hh;
  const float* a = attn + ((size_t)q * Hh + h) * L;
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  float* f = final_ + (size_t)q * Fw;
  for (int c = threadIdx.x; c < C2; c += blockDim.x) {
    float s = 0.f;
    float sc = ptx_ldg_f32(scale + c), of = ptx_ldg_f32(offset + c);
    for (int k = 0; k < L; ++k) {
      size_t at = (size_t)q * L + k;
      float2 st = ptx_ldg_v2_f32(stats + at);
      float a2d = ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(ptx_ldg_f32(pair + at * C2 + c), st.x), st.y), sc, of);
      s = ptx_fma_f32(ptx_ldg_f32(a + k), a2d, s);
    }
    ptx_st_global_f32(f + Hh * Cs + 4 * Hh * Pv + h * C2 + c, s);
  }
}
// the backbone update: QuatRigid's (qx qy qz tx ty tz), qw = 1, normalised; rigid = rigid @ update
__global__ void rigidUpdateK(float* rig, const float* upd, const float* bias, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  const float* u = upd + i * 6;
  float w = 1.f;
  float x = ptx_add_f32(ptx_ldg_f32(u), ptx_ldg_f32(bias));
  float y = ptx_add_f32(ptx_ldg_f32(u + 1), ptx_ldg_f32(bias + 1));
  float z = ptx_add_f32(ptx_ldg_f32(u + 2), ptx_ldg_f32(bias + 2));
  float inv = ptx_rsqrt_f32(ptx_max_f32(1e-6f, ptx_fma_f32(z, z, ptx_fma_f32(y, y, ptx_fma_f32(x, x, 1.f)))));
  w = inv; x = ptx_mul_f32(x, inv); y = ptx_mul_f32(y, inv); z = ptx_mul_f32(z, inv);
  float xx = ptx_mul_f32(x, x), yy = ptx_mul_f32(y, y), zz = ptx_mul_f32(z, z);
  float xy = ptx_mul_f32(x, y), xz = ptx_mul_f32(x, z), yz = ptx_mul_f32(y, z);
  float wx = ptx_mul_f32(w, x), wy = ptx_mul_f32(w, y), wz = ptx_mul_f32(w, z);
  float U[12] = {ptx_fma_f32(-2.f, ptx_add_f32(yy, zz), 1.f), ptx_mul_f32(2.f, ptx_sub_f32(xy, wz)), ptx_mul_f32(2.f, ptx_add_f32(xz, wy)),
                 ptx_mul_f32(2.f, ptx_add_f32(xy, wz)), ptx_fma_f32(-2.f, ptx_add_f32(xx, zz), 1.f), ptx_mul_f32(2.f, ptx_sub_f32(yz, wx)),
                 ptx_mul_f32(2.f, ptx_sub_f32(xz, wy)), ptx_mul_f32(2.f, ptx_add_f32(yz, wx)), ptx_fma_f32(-2.f, ptx_add_f32(xx, yy), 1.f),
                 ptx_add_f32(ptx_ldg_f32(u + 3), ptx_ldg_f32(bias + 3)),
                 ptx_add_f32(ptx_ldg_f32(u + 4), ptx_ldg_f32(bias + 4)),
                 ptx_add_f32(ptx_ldg_f32(u + 5), ptx_ldg_f32(bias + 5))};
  float rg[12], C[12];
  const float* rbase = rig + i * 12;
  float4 r0 = ptx_ld_global_v4_f32(rbase), r1 = ptx_ld_global_v4_f32(rbase + 4), r2 = ptx_ld_global_v4_f32(rbase + 8);
  rg[0] = r0.x; rg[1] = r0.y; rg[2] = r0.z; rg[3] = r0.w;
  rg[4] = r1.x; rg[5] = r1.y; rg[6] = r1.z; rg[7] = r1.w;
  rg[8] = r2.x; rg[9] = r2.y; rg[10] = r2.z; rg[11] = r2.w;
  rcompose(rg, U, C);
  ptx_st_global_v4_f32(rig + i * 12, make_float4(C[0], C[1], C[2], C[3]));
  ptx_st_global_v4_f32(rig + i * 12 + 4, make_float4(C[4], C[5], C[6], C[7]));
  ptx_st_global_v4_f32(rig + i * 12 + 8, make_float4(C[8], C[9], C[10], C[11]));
}
__global__ void reluCopyK(const float* x, float* y, size_t n) {
  if ((n & 3) == 0) {
    size_t t4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t4 < (n >> 2)) {
      float4 v = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(x) + t4);
      ptx_st_global_v4_f32(reinterpret_cast<float4*>(y) + t4,
                           make_float4(ptx_relu_f32(v.x), ptx_relu_f32(v.y), ptx_relu_f32(v.z), ptx_relu_f32(v.w)));
    }
    return;
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) ptx_st_global_f32(y + t, ptx_relu_f32(ptx_ldg_f32(x + t)));
}
__global__ void addBiasIntoK(float* dst, float* y, const float* bias, size_t rows, int C) {
  if ((C & 3) == 0) {
    size_t c4 = (size_t)(C >> 2), t4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t4 < rows * c4) {
      float4 d = ptx_ld_global_v4_f32(reinterpret_cast<const float4*>(dst) + t4);
      float4 v = ptx_ld_global_v4_f32(reinterpret_cast<const float4*>(y) + t4);
      float4 b = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(bias) + (t4 % c4));
      float4 s = make_float4(ptx_add_f32(d.x, ptx_add_f32(v.x, b.x)),
                             ptx_add_f32(d.y, ptx_add_f32(v.y, b.y)),
                             ptx_add_f32(d.z, ptx_add_f32(v.z, b.z)),
                             ptx_add_f32(d.w, ptx_add_f32(v.w, b.w)));
      ptx_st_global_v4_f32(reinterpret_cast<float4*>(dst) + t4, s);
      ptx_st_global_v4_f32(reinterpret_cast<float4*>(y) + t4,
                           make_float4(ptx_relu_f32(s.x), ptx_relu_f32(s.y), ptx_relu_f32(s.z), ptx_relu_f32(s.w)));
    }
    return;
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) {
    float s = ptx_add_f32(ptx_ld_global_f32(dst + t), ptx_add_f32(ptx_ld_global_f32(y + t), ptx_ldg_f32(bias + (t % C))));
    ptx_st_global_f32(dst + t, s);
    ptx_st_global_f32(y + t, ptx_relu_f32(s));
  }
}
__global__ void __launch_bounds__(256, 4)
addBiasLnK(float* __restrict__ residual, const float* __restrict__ y, const float* __restrict__ bias,
           float* __restrict__ out, size_t rows, int C, const float* __restrict__ scale, const float* __restrict__ offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float* rr = residual + row * C;
  const float* yr = y + row * C;
  if (C == 384) {
    float xv[12], yv[12], bv[12], s = 0.f, v = 0.f;
#pragma unroll
    for (int k = 0; k < 12; ++k) {
      int c = lane + k * 32;
      xv[k] = ptx_ld_global_f32(rr + c);
      yv[k] = ptx_ldg_f32(yr + c);
      bv[k] = ptx_ldg_f32(bias + c);
    }
#pragma unroll
    for (int k = 0; k < 12; ++k) {
      int c = lane + k * 32;
      float val = ptx_add_f32(xv[k], ptx_add_f32(yv[k], bv[k]));
      ptx_st_global_f32(rr + c, val);
      xv[k] = val;
      s = ptx_add_f32(s, val);
    }
    s = ptx_warp_sum_f32(s);
    float mean = ptx_mul_f32(s, 1.f / 384.f);
    float sc[12], of[12];
#pragma unroll
    for (int k = 0; k < 12; ++k) {
      int c = lane + k * 32;
      sc[k] = ptx_ldg_f32(scale + c);
      of[k] = ptx_ldg_f32(offset + c);
    }
#pragma unroll
    for (int k = 0; k < 12; ++k) { float d = ptx_sub_f32(xv[k], mean); v = ptx_fma_f32(d, d, v); }
    v = ptx_warp_sum_f32(v);
    float inv = ptx_rsqrt_f32(ptx_fma_f32(v, 1.f / 384.f, 1e-5f));
    float* outr = out + row * 384;
#pragma unroll
    for (int k = 0; k < 12; ++k) {
      int c = lane + k * 32;
      ptx_st_global_f32(outr + c, ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(xv[k], mean), inv), sc[k], of[k]));
    }
    return;
  }
  float s = 0.f;
  for (int c = lane; c < C; c += 32) {
    float v = ptx_add_f32(ptx_ld_global_f32(rr + c), ptx_add_f32(ptx_ldg_f32(yr + c), ptx_ldg_f32(bias + c)));
    ptx_st_global_f32(rr + c, v);
    s = ptx_add_f32(s, v);
  }
  s = ptx_warp_sum_f32(s);
  float mean = ptx_div_f32(s, (float)C), v = 0.f;
  for (int c = lane; c < C; c += 32) { float d = ptx_sub_f32(ptx_ld_global_f32(rr + c), mean); v = ptx_fma_f32(d, d, v); }
  v = ptx_warp_sum_f32(v);
  float inv = ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(v, (float)C), 1e-5f));
  for (int c = lane; c < C; c += 32)
    ptx_st_global_f32(out + row * C + c,
                      ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(ptx_ld_global_f32(rr + c), mean), inv), ptx_ldg_f32(scale + c), ptx_ldg_f32(offset + c)));
}
__global__ void __launch_bounds__(256, 6)
ipaUnpackK(const float* __restrict__ proj, float* __restrict__ qs, float* __restrict__ ks, float* __restrict__ vs,
           const float* __restrict__ bias, const float* __restrict__ rig,
           float* __restrict__ qp, float* __restrict__ kp, float* __restrict__ vp,
           int L, int Hh, int Cs, int Pq, int Pv, float qScale) {
  int Ws = Hh * Cs, Wqp = Hh * 3 * Pq, Wvp = Hh * 3 * Pv, Wipa = 3 * Ws + 2 * Wqp + Wvp;
  int pts = Hh * (2 * Pq + Pv), work = ptx_max_s32(3 * Ws, pts);
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (unsigned)(L * work)) return;
  int i = (int)(t / (unsigned)work), k = (int)(t % (unsigned)work);
  const float* row = proj + (size_t)i * Wipa;
  if (k < 3 * Ws) {
    float v = ptx_add_f32(ptx_ldg_f32(row + k), ptx_ldg_f32(bias + k));
    if (k < Ws) ptx_st_global_f32(qs + (size_t)i * Ws + k, ptx_mul_f32(v, qScale));
    else if (k < 2 * Ws) ptx_st_global_f32(ks + (size_t)i * Ws + (k - Ws), v);
    else ptx_st_global_f32(vs + (size_t)i * Ws + (k - 2 * Ws), v);
  }
  if (k < pts) {
    float rg[12];
    const float* rbase = rig + i * 12;
    float4 r0 = ptx_ldg_v4_f32(rbase), r1 = ptx_ldg_v4_f32(rbase + 4), r2 = ptx_ldg_v4_f32(rbase + 8);
    rg[0] = r0.x; rg[1] = r0.y; rg[2] = r0.z; rg[3] = r0.w;
    rg[4] = r1.x; rg[5] = r1.y; rg[6] = r1.z; rg[7] = r1.w;
    rg[8] = r2.x; rg[9] = r2.y; rg[10] = r2.z; rg[11] = r2.w;
    if (k < Hh * Pq) {
      int h = k / Pq, p = k % Pq, off = h * 3 * Pq, boff = 3 * Ws + off;
      const float* r = row + boff;
      float loc[3] = {ptx_add_f32(ptx_ldg_f32(r + p), ptx_ldg_f32(bias + boff + p)),
                      ptx_add_f32(ptx_ldg_f32(r + Pq + p), ptx_ldg_f32(bias + boff + Pq + p)),
                      ptx_add_f32(ptx_ldg_f32(r + 2 * Pq + p), ptx_ldg_f32(bias + boff + 2 * Pq + p))}, g[3];
      rapply(rg, loc, g);
      for (int a = 0; a < 3; ++a) ptx_st_global_f32(qp + ((size_t)i * Hh * Pq + k) * 3 + a, g[a]);
    } else if (k < 2 * Hh * Pq) {
      int kpIdx = k - Hh * Pq, h = kpIdx / Pq, p = kpIdx % Pq, off = h * 3 * Pq, boff = 3 * Ws + Wqp + off;
      const float* r = row + boff;
      float loc[3] = {ptx_add_f32(ptx_ldg_f32(r + p), ptx_ldg_f32(bias + boff + p)),
                      ptx_add_f32(ptx_ldg_f32(r + Pq + p), ptx_ldg_f32(bias + boff + Pq + p)),
                      ptx_add_f32(ptx_ldg_f32(r + 2 * Pq + p), ptx_ldg_f32(bias + boff + 2 * Pq + p))}, g[3];
      rapply(rg, loc, g);
      for (int a = 0; a < 3; ++a) ptx_st_global_f32(kp + ((size_t)i * Hh * Pq + kpIdx) * 3 + a, g[a]);
    } else {
      int vpIdx = k - 2 * Hh * Pq, h = vpIdx / Pv, p = vpIdx % Pv, off = h * 3 * Pv, boff = 3 * Ws + 2 * Wqp + off;
      const float* r = row + boff;
      float loc[3] = {ptx_add_f32(ptx_ldg_f32(r + p), ptx_ldg_f32(bias + boff + p)),
                      ptx_add_f32(ptx_ldg_f32(r + Pv + p), ptx_ldg_f32(bias + boff + Pv + p)),
                      ptx_add_f32(ptx_ldg_f32(r + 2 * Pv + p), ptx_ldg_f32(bias + boff + 2 * Pv + p))}, g[3];
      rapply(rg, loc, g);
      for (int a = 0; a < 3; ++a) ptx_st_global_f32(vp + ((size_t)i * Hh * Pv + vpIdx) * 3 + a, g[a]);
    }
  }
}
// angles (unnormalised [L, 7, 2]) -> frames -> atom14 -> atom37, a thread a residue
__global__ void sidechainAtomsK(const float* unnorm, const float* rig, float positionScale, const int* aatypeIn,
                                const float* defaultFrames, const int* atom14Group, const float* litPos,
                                const float* atom14Mask, const int* atom37To14, const float* atom37Mask,
                                const float* seqMask, float* angles, float* pos14, float* pos37, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aa = ptx_min_s32(ptx_max_s32(ptx_ldg_s32(aatypeIn + i), 0), 19);    // the reference clips X to the 20 standard types
  float bb[12];
  for (int k = 0; k < 12; ++k) bb[k] = ptx_ldg_f32(rig + i * 12 + k);
  for (int k = 9; k < 12; ++k) bb[k] = ptx_mul_f32(bb[k], positionScale);
  float sinA[8], cosA[8];
  sinA[0] = 0.f; cosA[0] = 1.f;
  for (int a = 0; a < 7; ++a) {
    float s = ptx_ldg_f32(unnorm + (i * 7 + a) * 2), c = ptx_ldg_f32(unnorm + (i * 7 + a) * 2 + 1);
    float inv = ptx_div_f32(1.f, ptx_sqrt_f32(ptx_max_f32(ptx_fma_f32(c, c, ptx_mul_f32(s, s)), 1e-12f)));
    s = ptx_mul_f32(s, inv); c = ptx_mul_f32(c, inv);
    ptx_st_global_f32(angles + (i * 7 + a) * 2, s);
    ptx_st_global_f32(angles + (i * 7 + a) * 2 + 1, c);
    sinA[a + 1] = s; cosA[a + 1] = c;
  }
  // all_frames[g] = default_frame[aa, g] composed with the rotation about x by the g-th torsion;
  // chi2..4 (g = 5..7) chain onto chi1 (g = 4)
  float cur[12];
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const float* m = defaultFrames + ((size_t)aa * 8 + g) * 16;
    float D[12] = {ptx_ldg_f32(m), ptx_ldg_f32(m + 1), ptx_ldg_f32(m + 2),
                   ptx_ldg_f32(m + 4), ptx_ldg_f32(m + 5), ptx_ldg_f32(m + 6),
                   ptx_ldg_f32(m + 8), ptx_ldg_f32(m + 9), ptx_ldg_f32(m + 10),
                   ptx_ldg_f32(m + 3), ptx_ldg_f32(m + 7), ptx_ldg_f32(m + 11)};
    float Rx[12] = {1.f, 0.f, 0.f, 0.f, cosA[g], ptx_neg_f32(sinA[g]), 0.f, sinA[g], cosA[g], 0.f, 0.f, 0.f};
    float R[12];
    rcompose(D, Rx, R);
    for (int k = 9; k < 12; ++k) R[k] = D[k];       // compose_rotation keeps the translation
    if (g >= 5) {
      float tmp[12];
      rcompose(cur, R, tmp);
      for (int k = 0; k < 12; ++k) cur[k] = tmp[k];
    } else {
      for (int k = 0; k < 12; ++k) cur[k] = R[k];
    }
    float glob[12];
    rcompose(bb, cur, glob);
    for (int a = 0; a < 14; ++a) {
      if (ptx_ldg_s32(atom14Group + aa * 14 + a) != g) continue;
      float lp[3] = {ptx_ldg_f32(litPos + ((size_t)aa * 14 + a) * 3),
                     ptx_ldg_f32(litPos + ((size_t)aa * 14 + a) * 3 + 1),
                     ptx_ldg_f32(litPos + ((size_t)aa * 14 + a) * 3 + 2)};
      float p[3];
      rapply(glob, lp, p);
      float mk = ptx_ldg_f32(atom14Mask + aa * 14 + a);
      for (int x = 0; x < 3; ++x) ptx_st_global_f32(pos14 + ((size_t)i * 14 + a) * 3 + x, ptx_mul_f32(p[x], mk));
    }
  }
  float sm = ptx_ldg_f32(seqMask + i);
  for (int a = 0; a < 37; ++a) {
    int s = ptx_ldg_s32(atom37To14 + aa * 37 + a);
    float m = ptx_mul_f32(ptx_ldg_f32(atom37Mask + aa * 37 + a), sm);
    for (int x = 0; x < 3; ++x)
      ptx_st_global_f32(pos37 + ((size_t)i * 37 + a) * 3 + x, ptx_mul_f32(ptx_ld_global_f32(pos14 + ((size_t)i * 14 + s) * 3 + x), m));
  }
}
__global__ void softplusScaleK(const float* raw, float* out, int Hh, float base) {
  int h = threadIdx.x;
  if (h < Hh) ptx_st_global_f32(out + h, ptx_mul_f32(base, ptx_log_f32(ptx_add_f32(1.f, ptx_exp_f32(ptx_ldg_f32(raw + h))))));
}

struct StructureOut { float* act; float* rigid; float* pos37; float* pos14; float* angles; };
inline StructureOut structureModule(const float* single, const float* pair, int L, float positionScale) {
  const std::string S = "structure_module/", F = S + "fold_iteration/", I = F + "invariant_point_attention/";
  const int C = 384, C2 = 128;
  size_t pairs = (size_t)L * L;
  float* initial = scratch<float>("sm.initial", (size_t)L * C);
  layerNorm(single, initial, L, C, S + "single_layer_norm");
  float* act = scratch<float>("sm.act", (size_t)L * C);
  linearB(initial, S + "initial_projection", -1, act, L, C, C);
  const bool lean = AF2_TIGHT && !roomFor(pairs * C2 * 4, { "sm.act2d" });
  float* act2d = lean ? nullptr : scratch<float>("sm.act2d", pairs * C2);
  float2* stats = lean ? scratch<float2>("sm.lnstats", pairs) : nullptr;
  if (lean) { releaseScratch({ "sm.act2d" }); lnStatsK<<<(unsigned)((pairs + 7) / 8), 256, 0, STREAM>>>(pair, stats, pairs, C2); }
  else layerNorm(pair, act2d, pairs, C2, S + "pair_layer_norm");
  int Hh = (int)dimW(I + "q_scalar_projection/weights", 1), Cs = (int)dimW(I + "q_scalar_projection/weights", 2);
  int Pq = (int)dimW(I + "q_point_projection/point_projection/weights", 2) / 3;
  int Pv = (int)dimW(I + "v_point_projection/point_projection/weights", 2) / 3;
  float* b2d = scratch<float>("sm.b2d", pairs * Hh);
  if (!lean) linearB(act2d, I + "attention_2d", -1, b2d, pairs, C2, Hh);    // the same every iteration (shared weights)
  else {                         // (on a card short of room, from the LayerNorm in chunks of pairs)
    size_t per = std::max<size_t>(1, std::min(pairs, AF2_CHUNK / C2));
    float* lnc = scratch<float>("sm.lnc", per * C2);
    for (size_t r0 = 0; r0 < pairs; r0 += per) {
      size_t r = std::min(per, pairs - r0);
      layerNorm(pair + r0 * C2, lnc, r, C2, S + "pair_layer_norm");
      linearB(lnc, I + "attention_2d", -1, b2d + r0 * Hh, r, C2, Hh);
    }
  }
  float* pw = scratch<float>("sm.pw", Hh);
  softplusScaleK<<<1, 32, 0, STREAM>>>(P(I + "trainable_point_weights"), pw, Hh, sqrtf(1.f / (Pq * 9.f / 2.f)));
  float* rig = scratch<float>("sm.rigid", (size_t)L * 12);
  identityRigidK<<<blocks(L), 256, 0, STREAM>>>(rig, L);
  int Ws = Hh * Cs, Wqp = Hh * 3 * Pq, Wvp = Hh * 3 * Pv, Wipa = 3 * Ws + 2 * Wqp + Wvp;
  float* qs = scratch<float>("sm.qs", (size_t)3 * L * Ws);
  float* ks = qs + (size_t)L * Ws;
  float* vs = qs + (size_t)2 * L * Ws;
  int ipaWork = std::max(3 * Ws, Hh * (2 * Pq + Pv));
  struct IpaW { const float* w; const float* b; };
  static std::map<const float*, IpaW> ipaCache;
  const float* qsw = P(I + "q_scalar_projection/weights");
  auto ipaIt = ipaCache.find(qsw);
  if (ipaIt == ipaCache.end()) {
    float* w = wpool<float>((size_t)C * Wipa);
    float* b = wpool<float>(Wipa);
    auto col = [&](int off, int wcol, const std::string& name) {
      CK(cudaMemcpy2DAsync(w + off, (size_t)Wipa * 4, P(name + "/weights"), (size_t)wcol * 4,
                           (size_t)wcol * 4, C, cudaMemcpyDeviceToDevice, STREAM));
      CK(cudaMemcpyAsync(b + off, P(name + "/bias"), wcol * 4, cudaMemcpyDeviceToDevice, STREAM));
    };
    col(0, Ws, I + "q_scalar_projection");
    col(Ws, Ws, I + "k_scalar_projection");
    col(2 * Ws, Ws, I + "v_scalar_projection");
    col(3 * Ws, Wqp, I + "q_point_projection/point_projection");
    col(3 * Ws + Wqp, Wqp, I + "k_point_projection/point_projection");
    col(3 * Ws + 2 * Wqp, Wvp, I + "v_point_projection/point_projection");
    ipaIt = ipaCache.emplace(qsw, IpaW{w, b}).first;
  }
  float* proj = scratch<float>("sm.pproj", (size_t)L * Wipa);
  float* qp = scratch<float>("sm.qp", (size_t)L * Hh * Pq * 3); float* kp = scratch<float>("sm.kp", (size_t)L * Hh * Pq * 3);
  float* vp = scratch<float>("sm.vp", (size_t)L * Hh * Pv * 3);
  float* attn = scratch<float>("sm.attn", (size_t)L * Hh * L);
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  float* fin = scratch<float>("sm.final", (size_t)L * Fw);
  float* upd = scratch<float>("sm.upd", (size_t)L * C);
  float* t1 = scratch<float>("sm.t1", (size_t)L * C); float* t2 = scratch<float>("sm.t2", (size_t)L * C);
  float* rq = scratch<float>("sm.rq", (size_t)L * 6);
  const float* seqMask = W("seq_mask");
  int layers = 8;
  for (int it = 0; it < layers; ++it) {
    // IPA
    gemm(act, ipaIt->second.w, proj, L, C, Wipa);
    ipaUnpackK<<<blocks((size_t)L * ipaWork), 256, 0, STREAM>>>(
      proj, qs, ks, vs, ipaIt->second.b, rig, qp, kp, vp, L, Hh, Cs, Pq, Pv, sqrtf(1.f / Cs));
    ipaWeightsK<<<(unsigned)(((size_t)L * Hh + 7) / 8), 256, 0, STREAM>>>(qs, ks, qp, kp, b2d, pw, seqMask, attn, L, Hh, Cs, Pq);
    // (the pair term apart, a block a query: see ipaPairTermK)
    const bool pairApart = act2d && Hh <= IPA_MAX_HEADS && C2 <= 1024;
    if ((pairApart || lean) && Cs + Pv <= 32) {
      size_t nw = (Cs == 16 && Pv == 8) ? (size_t)((L + 3) >> 2) * Hh : (size_t)L * Hh;
      ipaOutputsFastK<<<(unsigned)((nw + 3) / 4), 128, 0, STREAM>>>(attn, vs, vp, rig, fin, L, Hh, Cs, Pv, C2);
    } else
      ipaOutputsK<<<L * Hh, 128, 0, STREAM>>>(attn, vs, vp, pairApart ? nullptr : act2d, rig, fin, L, Hh, Cs, Pv, C2);
    if (pairApart) ipaPairTermK<<<L, (C2 == 128 && Hh == 12) ? 128 : (C2 + 31) / 32 * 32, 0, STREAM>>>(attn, act2d, fin, L, Hh, Cs, Pv, C2);
    if (lean) ipaPairOutputsK<<<L * Hh, 128, 0, STREAM>>>(attn, pair, stats, P(S + "pair_layer_norm/scale"),
                                                         P(S + "pair_layer_norm/offset"), fin, L, Hh, Cs, Pv, C2);
    gemm(fin, P(I + "output_projection/weights", -1), upd, L, Fw, C);
    addBiasLnK<<<(unsigned)((L + 7) / 8), 256, 0, STREAM>>>(act, upd, P(I + "output_projection/bias", -1), t1, L, C,
                                                            P(F + "attention_layer_norm/scale", -1), P(F + "attention_layer_norm/offset", -1));
    // the transition: three layers, ReLU between, a residual round them
    linearB(t1, F + "transition", -1, t2, L, C, C, true);
    linearB(t2, F + "transition_1", -1, upd, L, C, C, true);
    gemm(upd, P(F + "transition_2/weights", -1), t2, L, C, C);
    addBiasLnK<<<(unsigned)((L + 7) / 8), 256, 0, STREAM>>>(t1, t2, P(F + "transition_2/bias", -1), act, L, C,
                                                            P(F + "transition_layer_norm/scale", -1), P(F + "transition_layer_norm/offset", -1));
    gemm(act, P(F + "quat_rigid/rigid/weights", -1), rq, L, C, 6);
    rigidUpdateK<<<blocks(L), 256, 0, STREAM>>>(rig, rq, P(F + "quat_rigid/rigid/bias", -1), L);
  }
  // the side chains, from the last iteration (the only one the outputs keep)
  const std::string R = F + "rigid_sidechain/";
  float* ra = scratch<float>("sm.ra", (size_t)L * C); float* sc = scratch<float>("sm.sc", (size_t)L * 128);
  float* sb = scratch<float>("sm.sb", (size_t)L * 128); float* sc2 = scratch<float>("sm.sc2", (size_t)L * 128);
  reluCopyK<<<blocks((size_t)L * C >> 2), 256, 0, STREAM>>>(act, ra, (size_t)L * C);
  linearB(ra, R + "input_projection", -1, sc, L, C, 128);
  reluCopyK<<<blocks((size_t)L * C >> 2), 256, 0, STREAM>>>(initial, ra, (size_t)L * C);
  gemm(ra, P(R + "input_projection_1/weights", -1), sb, L, C, 128);
  addBiasIntoK<<<blocks((size_t)L * 32), 256, 0, STREAM>>>(sc, sb, P(R + "input_projection_1/bias", -1), L, 128);
  for (int r = 0; r < 2; ++r) {
    std::string suffix = r == 0 ? "" : "_1";
    linearB(sb, R + "resblock1" + suffix, -1, sc2, L, 128, 128, true);
    gemm(sc2, P(R + "resblock2" + suffix + "/weights", -1), sb, L, 128, 128);
    addBiasIntoK<<<blocks((size_t)L * 32), 256, 0, STREAM>>>(sc, sb, P(R + "resblock2" + suffix + "/bias", -1), L, 128);
  }
  float* un = scratch<float>("sm.un", (size_t)L * 14);
  linearB(sb, R + "unnormalized_angles", -1, un, L, 128, 14);
  StructureOut o{act, rig, scratch<float>("sm.pos37", (size_t)L * 37 * 3), scratch<float>("sm.pos14", (size_t)L * 14 * 3),
                 scratch<float>("sm.angles", (size_t)L * 14)};
  sidechainAtomsK<<<blocks(L, 128), 128, 0, STREAM>>>(un, rig, positionScale, Idev("aatype"), W("c/rigid_group_default_frame"),
                                                 Idev("c/atom14_to_rigid_group"), W("c/atom14_rigid_group_positions"),
                                                 W("c/atom14_mask"), Idev("c/atom37_to_atom14"), W("c/atom37_mask"),
                                                 seqMask, o.angles, o.pos14, o.pos37, L);
  return o;
}
