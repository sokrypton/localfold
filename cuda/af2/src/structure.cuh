// The structure module and the confidence heads - af3-any-model's AF2 (alphafold3/af2/model/folding.py:
// generate_monomer_rigids, FoldIteration, InvariantPointAttention, MultiRigidSidechain; all_atom.py),
// float32. A rigid is 12 floats: the rotation row-major (xx xy xz yx yy yz zx zy zz), then the
// translation.
#pragma once
#include "ops.cuh"

__device__ inline void rapply(const float* r, const float* p, float* o) {      // R p + t
  for (int a = 0; a < 3; ++a) o[a] = r[a * 3] * p[0] + r[a * 3 + 1] * p[1] + r[a * 3 + 2] * p[2] + r[9 + a];
}
__device__ inline void rapplyInv(const float* r, const float* p, float* o) {   // R^T (p - t)
  float d[3] = {p[0] - r[9], p[1] - r[10], p[2] - r[11]};
  for (int a = 0; a < 3; ++a) o[a] = r[a] * d[0] + r[3 + a] * d[1] + r[6 + a] * d[2];
}
__device__ inline void rcompose(const float* A, const float* B, float* C) {    // A @ B
  float R[9], t[3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  rapply(A, B + 9, t);
  for (int k = 0; k < 9; ++k) C[k] = R[k];
  for (int k = 0; k < 3; ++k) C[9 + k] = t[k];
}

__global__ void identityRigidK(float* r, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  float* x = r + i * 12;
  for (int k = 0; k < 12; ++k) x[k] = 0;
  x[0] = x[4] = x[8] = 1;
}
// points [L, H*3*P] from the projection (per head: x[0:P] y[P:2P] z[2P:3P]) -> global [L, H, P, 3]
__global__ void pointsToGlobalK(const float* proj, const float* rig, float* out, int L, int Hh, int Pp) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * Hh * Pp) return;
  int p = (int)(t % Pp); size_t r = t / Pp; int h = (int)(r % Hh); int i = (int)(r / Hh);
  const float* s = proj + (size_t)i * Hh * 3 * Pp + h * 3 * Pp;
  float loc[3] = {s[p], s[Pp + p], s[2 * Pp + p]}, g[3];
  rapply(rig + i * 12, loc, g);
  for (int a = 0; a < 3; ++a) out[t * 3 + a] = g[a];
}
// IPA logits -> attention weights, a warp a (query, head): logits[q, k, h] then softmax over k
__global__ void __launch_bounds__(256, 4)
ipaWeightsK(const float* __restrict__ qs, const float* __restrict__ ks, const float* __restrict__ qp,
            const float* __restrict__ kp, const float* __restrict__ b2d, const float* __restrict__ pw,
            const float* __restrict__ seqMask, float* __restrict__ attn, int L, int Hh, int Cs, int Pq) {
  __shared__ float srow[8][512];
  size_t wid = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (wid >= (size_t)L * Hh) return;
  int h = (int)(wid % Hh), q = (int)(wid / Hh);
  float* row = attn + ((size_t)q * Hh + h) * L;     // [q, h, k]
  float* sr = L <= 512 ? srow[warp] : row;
  float mx = -INFINITY;
  float w = __ldg(pw + h);
  // the query's scalars and points held in registers, read once (they were read again for every key); the
  // keys' scalars as float4s where 16-byte aligned - the same products in the same order
  if (Cs == 16 && Pq == 4) {
    const float4* qsh = reinterpret_cast<const float4*>(qs + ((size_t)q * Hh + h) * 16);
    float4 qv0 = __ldg(qsh), qv1 = __ldg(qsh + 1), qv2 = __ldg(qsh + 2), qv3 = __ldg(qsh + 3);
    const float4* qph = reinterpret_cast<const float4*>(qp + ((size_t)q * Hh + h) * 12);
    float4 qp0 = __ldg(qph), qp1 = __ldg(qph + 1), qp2 = __ldg(qph + 2);
    float sq = __ldg(seqMask + q);
    for (int k = lane; k < L; k += 32) {
      const float4* kpk = reinterpret_cast<const float4*>(kp + ((size_t)k * Hh + h) * 12);
      float4 kp0 = __ldg(kpk), kp1 = __ldg(kpk + 1), kp2 = __ldg(kpk + 2);
      float d2 = 0;
      float dx, dy, dz;
      dx = qp0.x - kp0.x; dy = qp0.y - kp0.y; dz = qp0.z - kp0.z; d2 += dx * dx + dy * dy + dz * dz;
      dx = qp0.w - kp0.w; dy = qp1.x - kp1.x; dz = qp1.y - kp1.y; d2 += dx * dx + dy * dy + dz * dz;
      dx = qp1.z - kp1.z; dy = qp1.w - kp1.w; dz = qp2.x - kp2.x; d2 += dx * dx + dy * dy + dz * dz;
      dx = qp2.y - kp2.y; dy = qp2.z - kp2.z; dz = qp2.w - kp2.w; d2 += dx * dx + dy * dy + dz * dz;
      const float4* ksk = reinterpret_cast<const float4*>(ks + ((size_t)k * Hh + h) * 16);
      float sc = 0;
      float4 v0 = __ldg(ksk); sc += qv0.x * v0.x; sc += qv0.y * v0.y; sc += qv0.z * v0.z; sc += qv0.w * v0.w;
      float4 v1 = __ldg(ksk + 1); sc += qv1.x * v1.x; sc += qv1.y * v1.y; sc += qv1.z * v1.z; sc += qv1.w * v1.w;
      float4 v2 = __ldg(ksk + 2); sc += qv2.x * v2.x; sc += qv2.y * v2.y; sc += qv2.z * v2.z; sc += qv2.w * v2.w;
      float4 v3 = __ldg(ksk + 3); sc += qv3.x * v3.x; sc += qv3.y * v3.y; sc += qv3.z * v3.z; sc += qv3.w * v3.w;
      float s = -0.5f * w * d2;
      s += sc + __ldg(b2d + ((size_t)q * L + k) * Hh + h);
      s -= 1e5f * (1.f - sq * __ldg(seqMask + k));
      s *= sqrtf(1.f / 3.f);
      sr[k] = s;
      mx = fmaxf(mx, s);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    float sum = 0;
    for (int k = lane; k < L; k += 32) { float e = __expf(sr[k] - mx); sr[k] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    for (int k = lane; k < L; k += 32) row[k] = sr[k] / sum;
    return;
  }
  for (int k = lane; k < L; k += 32) {
    float d2 = 0, sc = 0;
    for (int p = 0; p < Pq; ++p)
      for (int a = 0; a < 3; ++a) {
        float d = __ldg(qp + (((size_t)q * Hh + h) * Pq + p) * 3 + a) - __ldg(kp + (((size_t)k * Hh + h) * Pq + p) * 3 + a);
        d2 += d * d;
      }
    for (int c = 0; c < Cs; ++c) sc += __ldg(qs + ((size_t)q * Hh + h) * Cs + c) * __ldg(ks + ((size_t)k * Hh + h) * Cs + c);
    float s = -0.5f * w * d2;
    s += sc + __ldg(b2d + ((size_t)q * L + k) * Hh + h);
    s -= 1e5f * (1.f - __ldg(seqMask + q) * __ldg(seqMask + k));
    s *= sqrtf(1.f / 3.f);
    row[k] = s;
    mx = fmaxf(mx, s);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  float sum = 0;
  for (int k = lane; k < L; k += 32) { float e = __expf(row[k] - mx); row[k] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  for (int k = lane; k < L; k += 32) row[k] /= sum;
}
// IPA's scalar and point outputs when the pair term runs in ipaPairTermK: a warp a (query, head), lanes 0..Cs-1
// accumulating vs and lanes Cs..Cs+Pv-1 accumulating vp concurrently in one loop over k
__global__ void __launch_bounds__(256, 6)
ipaOutputsFastK(const float* __restrict__ attn, const float* __restrict__ vs, const float* __restrict__ vp,
                const float* __restrict__ rig, float* __restrict__ final_, int L, int Hh, int Cs, int Pv, int C2) {
  int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (wid >= L * Hh) return;
  int q = wid / Hh, h = wid % Hh;
  const float* a = attn + (size_t)wid * L;
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  float* f = final_ + (size_t)q * Fw;
  if (Cs == 16 && Pv == 8) {
    __shared__ float sp[8][24];
    const float* p0 = lane < 16 ? (vs + h * 16 + lane) : (vp + h * 24 + (lane - 16));
    const float* p1 = vp + h * 24 + (lane - 8);
    int st0 = lane < 16 ? (Hh * 16) : (Hh * 24), st1 = Hh * 24;
    float g0 = 0.f, g1 = 0.f;
#pragma unroll 8
    for (int k = 0; k < L; ++k) {
      float ak = __ldg(a + k);
      g0 += ak * __ldg(p0 + (size_t)k * st0);
      if (lane >= 24) g1 += ak * __ldg(p1 + (size_t)k * st1);
    }
    if (lane < 16) f[h * 16 + lane] = g0;
    else sp[warp][lane - 16] = g0;
    if (lane >= 24) sp[warp][lane - 8] = g1;
    __syncwarp();
    if (lane < 8) {
      int p = lane;
      float g[3] = {sp[warp][p * 3], sp[warp][p * 3 + 1], sp[warp][p * 3 + 2]}, loc[3];
      rapplyInv(rig + q * 12, g, loc);
      int base = Hh * 16, idx = h * 8 + p;
      f[base + idx] = loc[0];
      f[base + Hh * 8 + idx] = loc[1];
      f[base + 2 * Hh * 8 + idx] = loc[2];
      f[base + 3 * Hh * 8 + idx] = sqrtf(fmaxf(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2], 1e-16f));
    }
    return;
  }
  float s = 0.f, g0 = 0.f, g1 = 0.f, g2 = 0.f;
  int p = lane - Cs;
  bool isS = lane < Cs, isP = p >= 0 && p < Pv;
  if (!isS && !isP) return;
  if (isS) {
    const float* vsh = vs + h * Cs + lane;
    int strideS = Hh * Cs;
#pragma unroll 8
    for (int k = 0; k < L; ++k) s += __ldg(a + k) * __ldg(vsh + (size_t)k * strideS);
    f[h * Cs + lane] = s;
  } else {
    const float* vph = vp + (h * Pv + p) * 3;
    int strideP = Hh * Pv * 3;
#pragma unroll 8
    for (int k = 0; k < L; ++k) {
      float ak = __ldg(a + k);
      const float* vpk = vph + (size_t)k * strideP;
      g0 += ak * __ldg(vpk); g1 += ak * __ldg(vpk + 1); g2 += ak * __ldg(vpk + 2);
    }
    float g[3] = {g0, g1, g2}, loc[3];
    rapplyInv(rig + q * 12, g, loc);
    int base = Hh * Cs, idx = h * Pv + p;
    f[base + idx] = loc[0];
    f[base + Hh * Pv + idx] = loc[1];
    f[base + 2 * Hh * Pv + idx] = loc[2];
    f[base + 3 * Hh * Pv + idx] = sqrtf(fmaxf(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2], 1e-16f));
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
    float s = 0;
    for (int k = 0; k < L; ++k) s += a[k] * vs[((size_t)k * Hh + h) * Cs + c];
    f[h * Cs + c] = s;
  }
  for (int p = threadIdx.x; p < Pv; p += blockDim.x) {
    float g[3] = {0, 0, 0};
    for (int k = 0; k < L; ++k)
      for (int x = 0; x < 3; ++x) g[x] += a[k] * vp[(((size_t)k * Hh + h) * Pv + p) * 3 + x];
    float loc[3];
    rapplyInv(rig + q * 12, g, loc);
    int base = Hh * Cs, idx = h * Pv + p;
    f[base + idx] = loc[0];
    f[base + Hh * Pv + idx] = loc[1];
    f[base + 2 * Hh * Pv + idx] = loc[2];
    f[base + 3 * Hh * Pv + idx] = sqrtf(fmaxf(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2], 1e-16f));
  }
  for (int c = threadIdx.x; act2d && c < C2; c += blockDim.x) {     // (no act2d: ipaPairOutputsK writes it)
    float s = 0;
    for (int k = 0; k < L; ++k) s += a[k] * act2d[((size_t)q * L + k) * C2 + c];
    f[Hh * Cs + 4 * Hh * Pv + h * C2 + c] = s;
  }
}
// ipaOutputsK's pair term, a block a QUERY for every head at once: each act2d element is read once and feeds all Hh
// heads' sums (a block a (query, head) read the query's pair row once per head - 12 times, ~1.5 GB a call at 494
// residues). Each sum runs over k in the same order with the same arithmetic, so the outputs are the same bytes.
constexpr int IPA_MAX_HEADS = 16, IPA_KT = 256;
__global__ void ipaPairTermK(const float* __restrict__ attn, const float* __restrict__ act2d, float* __restrict__ final_,
                             int L, int Hh, int Cs, int Pv, int C2) {
  __shared__ float as[IPA_MAX_HEADS][IPA_KT];
  int q = blockIdx.x;
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  if (C2 == 128 && Hh == 12 && blockDim.x == 128) {
    int c = threadIdx.x;
    float acc[12] = {};
    for (int k0 = 0; k0 < L; k0 += IPA_KT) {
      int kt = min(IPA_KT, L - k0);
      for (int t = c; t < 12 * kt; t += 128) {
        int h = t / kt, k = t - h * kt;
        as[h][k] = __ldg(attn + ((size_t)q * 12 + h) * L + k0 + k);
      }
      __syncthreads();
      const float* a2d = act2d + ((size_t)q * L + k0) * 128 + c;
#pragma unroll 8
      for (int k = 0; k < kt; ++k) {
        float v = __ldg(a2d + (size_t)k * 128);
#pragma unroll
        for (int h = 0; h < 12; ++h) acc[h] += as[h][k] * v;
      }
      __syncthreads();
    }
    float* f = final_ + (size_t)q * Fw + 12 * Cs + 48 * Pv + c;
#pragma unroll
    for (int h = 0; h < 12; ++h) f[h * 128] = acc[h];
    return;
  }
  int c = threadIdx.x;
  float acc[IPA_MAX_HEADS];
#pragma unroll
  for (int h = 0; h < IPA_MAX_HEADS; ++h) acc[h] = 0.f;
  for (int k0 = 0; k0 < L; k0 += IPA_KT) {
    int kt = min(IPA_KT, L - k0);
    __syncthreads();
    for (int t = threadIdx.x; t < Hh * kt; t += blockDim.x) {
      int h = t / kt, k = t % kt;
      as[h][k] = attn[((size_t)q * Hh + h) * L + k0 + k];
    }
    __syncthreads();
    if (c < C2)
#pragma unroll 8
      for (int k = 0; k < kt; ++k) {
        float v = act2d[((size_t)q * L + k0 + k) * C2 + c];
#pragma unroll
        for (int h = 0; h < IPA_MAX_HEADS; ++h) if (h < Hh) acc[h] += as[h][k] * v;
      }
  }
  if (c < C2) {
    float* f = final_ + (size_t)q * Fw;
    for (int h = 0; h < Hh; ++h) f[Hh * Cs + 4 * Hh * Pv + h * C2 + c] = acc[h];
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
    for (int k = 0; k < 4; ++k) { float val = __ldg(xr + lane + 32 * k); xv[k] = val; s += val; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s * (1.f / 128.f), v = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) { float d = xv[k] - mean; v += d * d; }
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    if (lane == 0) stats[row] = make_float2(mean, rsqrtf(v * (1.f / 128.f) + 1e-5f));
    return;
  }
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = xr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  if (lane == 0) stats[row] = make_float2(mean, rsqrtf(v / C + 1e-5f));
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
    float s = 0;
    for (int k = 0; k < L; ++k) {
      size_t at = (size_t)q * L + k; float2 st = stats[at];
      s += a[k] * ((pair[at * C2 + c] - st.x) * st.y * scale[c] + offset[c]);
    }
    f[Hh * Cs + 4 * Hh * Pv + h * C2 + c] = s;
  }
}
// the backbone update: QuatRigid's (qx qy qz tx ty tz), qw = 1, normalised; rigid = rigid @ update
__global__ void rigidUpdateK(float* rig, const float* upd, const float* bias, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  const float* u = upd + i * 6;
  float w = 1, x = u[0] + bias[0], y = u[1] + bias[1], z = u[2] + bias[2];
  float inv = rsqrtf(fmaxf(1e-6f, w * w + x * x + y * y + z * z));
  w *= inv; x *= inv; y *= inv; z *= inv;
  float U[12] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
                 u[3] + bias[3], u[4] + bias[4], u[5] + bias[5]};
  float C[12];
  rcompose(rig + i * 12, U, C);
  for (int k = 0; k < 12; ++k) rig[i * 12 + k] = C[k];
}
__global__ void reluCopyK(const float* x, float* y, size_t n) {
  if ((n & 3) == 0) {
    size_t t4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t4 < (n >> 2)) {
      float4 v = reinterpret_cast<const float4*>(x)[t4];
      reinterpret_cast<float4*>(y)[t4] = make_float4(fmaxf(v.x, 0.f), fmaxf(v.y, 0.f), fmaxf(v.z, 0.f), fmaxf(v.w, 0.f));
    }
    return;
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmaxf(x[t], 0.f);
}
__global__ void addBiasIntoK(float* dst, float* y, const float* bias, size_t rows, int C) {
  if ((C & 3) == 0) {
    size_t c4 = (size_t)(C >> 2), t4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t4 < rows * c4) {
      float4 d = reinterpret_cast<const float4*>(dst)[t4];
      float4 v = reinterpret_cast<const float4*>(y)[t4];
      float4 b = reinterpret_cast<const float4*>(bias)[t4 % c4];
      float4 s = make_float4(d.x + v.x + b.x, d.y + v.y + b.y, d.z + v.z + b.z, d.w + v.w + b.w);
      reinterpret_cast<float4*>(dst)[t4] = s;
      reinterpret_cast<float4*>(y)[t4] = make_float4(fmaxf(s.x, 0.f), fmaxf(s.y, 0.f), fmaxf(s.z, 0.f), fmaxf(s.w, 0.f));
    }
    return;
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) {
    float s = dst[t] + y[t] + bias[t % C];
    dst[t] = s;
    y[t] = fmaxf(s, 0.f);
  }
}
__global__ void __launch_bounds__(256, 6)
addBiasLnK(float* __restrict__ residual, const float* __restrict__ y, const float* __restrict__ bias,
           float* __restrict__ out, size_t rows, int C, const float* __restrict__ scale, const float* __restrict__ offset) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float* rr = residual + row * C;
  const float* yr = y + row * C;
  if (C == 384) {
    float xv[12], s = 0, v = 0;
#pragma unroll
    for (int k = 0; k < 12; ++k) {
      int c = lane + k * 32;
      float val = rr[c] + (__ldg(yr + c) + __ldg(bias + c));
      rr[c] = val;
      xv[k] = val;
      s += val;
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / 384;
#pragma unroll
    for (int k = 0; k < 12; ++k) { float d = xv[k] - mean; v += d * d; }
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    float inv = rsqrtf(v / 384 + 1e-5f);
    float* outr = out + row * 384;
#pragma unroll
    for (int k = 0; k < 12; ++k) {
      int c = lane + k * 32;
      outr[c] = (xv[k] - mean) * inv * __ldg(scale + c) + __ldg(offset + c);
    }
    return;
  }
  float s = 0;
  for (int c = lane; c < C; c += 32) {
    float v = rr[c] + (__ldg(yr + c) + __ldg(bias + c));
    rr[c] = v;
    s += v;
  }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, v = 0;
  for (int c = lane; c < C; c += 32) { float d = rr[c] - mean; v += d * d; }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  float inv = rsqrtf(v / C + 1e-5f);
  for (int c = lane; c < C; c += 32) out[row * C + c] = (rr[c] - mean) * inv * __ldg(scale + c) + __ldg(offset + c);
}
__global__ void ipaUnpackK(const float* __restrict__ proj, float* __restrict__ qs, float* __restrict__ ks, float* __restrict__ vs,
                           const float* __restrict__ bias, const float* __restrict__ rig,
                           float* __restrict__ qp, float* __restrict__ kp, float* __restrict__ vp,
                           int L, int Hh, int Cs, int Pq, int Pv, float qScale) {
  int Ws = Hh * Cs, Wqp = Hh * 3 * Pq, Wvp = Hh * 3 * Pv, Wipa = 3 * Ws + 2 * Wqp + Wvp;
  int pts = Hh * (2 * Pq + Pv), work = max(3 * Ws, pts);
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * work) return;
  int i = (int)(t / work), k = (int)(t % work);
  const float* row = proj + (size_t)i * Wipa;
  if (k < 3 * Ws) {
    float v = row[k] + bias[k];
    if (k < Ws) qs[(size_t)i * Ws + k] = v * qScale;
    else if (k < 2 * Ws) ks[(size_t)i * Ws + (k - Ws)] = v;
    else vs[(size_t)i * Ws + (k - 2 * Ws)] = v;
  }
  if (k < pts) {
    const float* rg = rig + i * 12;
    if (k < Hh * Pq) {
      int h = k / Pq, p = k % Pq, off = h * 3 * Pq, boff = 3 * Ws + off;
      const float* r = row + boff;
      float loc[3] = {r[p] + bias[boff + p], r[Pq + p] + bias[boff + Pq + p], r[2 * Pq + p] + bias[boff + 2 * Pq + p]}, g[3];
      rapply(rg, loc, g);
      for (int a = 0; a < 3; ++a) qp[((size_t)i * Hh * Pq + k) * 3 + a] = g[a];
    } else if (k < 2 * Hh * Pq) {
      int kpIdx = k - Hh * Pq, h = kpIdx / Pq, p = kpIdx % Pq, off = h * 3 * Pq, boff = 3 * Ws + Wqp + off;
      const float* r = row + boff;
      float loc[3] = {r[p] + bias[boff + p], r[Pq + p] + bias[boff + Pq + p], r[2 * Pq + p] + bias[boff + 2 * Pq + p]}, g[3];
      rapply(rg, loc, g);
      for (int a = 0; a < 3; ++a) kp[((size_t)i * Hh * Pq + kpIdx) * 3 + a] = g[a];
    } else {
      int vpIdx = k - 2 * Hh * Pq, h = vpIdx / Pv, p = vpIdx % Pv, off = h * 3 * Pv, boff = 3 * Ws + 2 * Wqp + off;
      const float* r = row + boff;
      float loc[3] = {r[p] + bias[boff + p], r[Pv + p] + bias[boff + Pv + p], r[2 * Pv + p] + bias[boff + 2 * Pv + p]}, g[3];
      rapply(rg, loc, g);
      for (int a = 0; a < 3; ++a) vp[((size_t)i * Hh * Pv + vpIdx) * 3 + a] = g[a];
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
  int aa = min(max(aatypeIn[i], 0), 19);    // the reference clips X to the 20 standard types
  float bb[12];
  for (int k = 0; k < 12; ++k) bb[k] = rig[i * 12 + k];
  for (int k = 9; k < 12; ++k) bb[k] *= positionScale;
  float sinA[8], cosA[8];
  sinA[0] = 0; cosA[0] = 1;
  for (int a = 0; a < 7; ++a) {
    float s = unnorm[(i * 7 + a) * 2], c = unnorm[(i * 7 + a) * 2 + 1];
    float inv = 1.f / sqrtf(fmaxf(s * s + c * c, 1e-12f));
    s *= inv; c *= inv;
    angles[(i * 7 + a) * 2] = s; angles[(i * 7 + a) * 2 + 1] = c;
    sinA[a + 1] = s; cosA[a + 1] = c;
  }
  // all_frames[g] = default_frame[aa, g] composed with the rotation about x by the g-th torsion;
  // chi2..4 (g = 5..7) chain onto chi1 (g = 4)
  float cur[12];
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const float* m = defaultFrames + ((size_t)aa * 8 + g) * 16;
    float D[12] = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10], m[3], m[7], m[11]};
    float Rx[12] = {1, 0, 0, 0, cosA[g], -sinA[g], 0, sinA[g], cosA[g], 0, 0, 0};
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
      if (atom14Group[aa * 14 + a] != g) continue;
      float p[3];
      rapply(glob, litPos + ((size_t)aa * 14 + a) * 3, p);
      float mk = atom14Mask[aa * 14 + a];
      for (int x = 0; x < 3; ++x) pos14[((size_t)i * 14 + a) * 3 + x] = p[x] * mk;
    }
  }
  for (int a = 0; a < 37; ++a) {
    int s = atom37To14[aa * 37 + a];
    float m = atom37Mask[aa * 37 + a] * seqMask[i];
    for (int x = 0; x < 3; ++x) pos37[((size_t)i * 37 + a) * 3 + x] = pos14[((size_t)i * 14 + s) * 3 + x] * m;
  }
}
__global__ void softplusScaleK(const float* raw, float* out, int Hh, float base) {
  int h = threadIdx.x;
  if (h < Hh) out[h] = base * logf(1.f + expf(raw[h]));
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
    if ((pairApart || lean) && Cs + Pv <= 32)
      ipaOutputsFastK<<<(unsigned)(((size_t)L * Hh + 7) / 8), 256, 0, STREAM>>>(attn, vs, vp, rig, fin, L, Hh, Cs, Pv, C2);
    else
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
