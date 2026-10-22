// BioEmu2's two attentions and its conditioned transformer layer (network/bioemu2.py's _window_attention,
// _attention and _conditioned_layer), several samples stacked on the rows.
#pragma once
#include "layers.cuh"
#include "../../af3/src/flash.cuh"

namespace be2 {

// ---------------------------------------------------------------- attentions
// AttentionPairBiasSlidingWindow: a block per (window, head), a thread per query; online softmax over the 128
// keys. qkv [rows, 3D] as the vendor fuses it: q (+ q_bias) at [0, D), then k at [D, 2D) and v at [2D, 3D) - its
// (2, heads, hd) reshape of the key/value half. out [rows, D], head-major. Several samples are stacked on the
// rows, N atoms each: a window's keys stay inside its sample, and every sample reads one bias.
template <int HD>
__global__ void windowAttentionK(const float* qkv, const float* qBias, const float* bias, const int* pad, float* out,
                                 int N, int H) {
  const int wg = blockIdx.x, h = blockIdx.y, i = threadIdx.x, D = H * HD;
  const int NW = N / WQ, smp = wg / NW, w = wg % NW;
  const size_t base = (size_t)smp * N;
  __shared__ float sk[WK][HD], sv[WK][HD];
  __shared__ int sm[WK];
  for (int e = threadIdx.x; e < WK * HD; e += blockDim.x) {
    int j = e / HD, d = e % HD, ka = w * WQ - CTX + j;
    bool ok = ka >= 0 && ka < N;
    sk[j][d] = ok ? qkv[(base + ka) * 3 * D + D + h * HD + d] : 0.f;
    sv[j][d] = ok ? qkv[(base + ka) * 3 * D + 2 * D + h * HD + d] : 0.f;
  }
  for (int j = threadIdx.x; j < WK; j += blockDim.x) { int ka = w * WQ - CTX + j; sm[j] = ka >= 0 && ka < N && pad[ka]; }
  __syncthreads();
  const size_t qa = base + w * WQ + i;
  float q[HD];
  const float scale = rsqrtf((float)HD);
  for (int d = 0; d < HD; ++d) q[d] = (qkv[qa * 3 * D + h * HD + d] + qBias[h * HD + d]) * scale;
  const float* b = bias + (((size_t)w * H + h) * WQ + i) * WK;
  float mx = -INFINITY, sum = 0, acc[HD];
  for (int d = 0; d < HD; ++d) acc[d] = 0;
  for (int j = 0; j < WK; ++j) {
    float l = 0; for (int d = 0; d < HD; ++d) l += q[d] * sk[j][d];
    l = sm[j] ? l + b[j] : NEG_INF;
    float nm = fmaxf(mx, l), f = expf(mx - nm), e = expf(l - nm);
    sum = sum * f + e;
    for (int d = 0; d < HD; ++d) acc[d] = acc[d] * f + e * sv[j][d];
    mx = nm;
  }
  for (int d = 0; d < HD; ++d) out[qa * D + h * HD + d] = acc[d] / sum;
}

// ...the same with four lanes a query, each a quarter of the 128 keys (its own online softmax), merged by shuffles:
// 128 threads a (window, head) block where the form above has 32 - 9.5 ms of an 82 ms step at 494 tokens.
// Takes fused qkvg [rows, 4D] (q at [0, D), k at [D, 2D), v at [2D, 3D), g at [3D, 4D)) and fuses sigm(g) gating + TY output.
template <int HD, class TY = float>
__global__ void __launch_bounds__(128, 3)
windowAttention4K(const float* __restrict__ qkvg, const float* __restrict__ qBias,
                  const float* __restrict__ bias, const int* __restrict__ pad,
                  TY* __restrict__ out, int N, int H) {
  constexpr int SPLIT = 4, BK = 64, KS = BK / SPLIT;
  const int wg = blockIdx.x, h = blockIdx.y, D = H * HD;
  const int NW = N / WQ, smp = wg / NW, w = wg % NW;
  const int i = threadIdx.x / SPLIT, part = threadIdx.x % SPLIT;
  const size_t base = (size_t)smp * N;
  const size_t qa = base + w * WQ + i;
  const float scale = rsqrtf((float)HD);
  const float* b = bias + (((size_t)w * H + h) * WQ + i) * WK;
  if constexpr (HD == 32) {
    __shared__ float4 sk4[BK][8], sv4[BK][8];
    __shared__ int sm[BK];
    float4 q4[8], acc4[8];
    const float4* qp4 = reinterpret_cast<const float4*>(qkvg + qa * 4 * D + h * 32);
    const float4* qb4 = reinterpret_cast<const float4*>(qBias + h * 32);
    #pragma unroll
    for (int d4 = 0; d4 < 8; ++d4) {
      float4 qv = __ldg(qp4 + d4), bv = __ldg(qb4 + d4);
      q4[d4] = make_float4((qv.x + bv.x) * scale, (qv.y + bv.y) * scale,
                           (qv.z + bv.z) * scale, (qv.w + bv.w) * scale);
      acc4[d4] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float mx = -INFINITY, sum = 0.f;
    for (int k0 = 0; k0 < WK; k0 += BK) {
      #pragma unroll
      for (int m = 0; m < 4; ++m) {
        int idx = threadIdx.x + m * 128;
        int j = idx >> 3, d4 = idx & 7;
        int ka = w * WQ - CTX + k0 + j;
        bool ok = ka >= 0 && ka < N;
        const float4* kv4 = ok ? reinterpret_cast<const float4*>(qkvg + (base + ka) * 4 * D + h * 32) : nullptr;
        sk4[j][d4 ^ (j & 3)] = ok ? __ldg(kv4 + (D >> 2) + d4) : make_float4(0.f, 0.f, 0.f, 0.f);
        sv4[j][d4 ^ (j & 3)] = ok ? __ldg(kv4 + (2 * D >> 2) + d4) : make_float4(0.f, 0.f, 0.f, 0.f);
      }
      if (threadIdx.x < BK) {
        int ka = w * WQ - CTX + k0 + threadIdx.x;
        sm[threadIdx.x] = (ka >= 0 && ka < N && __ldg(pad + ka)) ? 1 : 0;
      }
      __syncthreads();
      float logits[KS];
      float nm = mx;
      #pragma unroll
      for (int jj = 0; jj < KS; ++jj) {
        const int j = jj * SPLIT + part;
        float l = 0.f;
        #pragma unroll
        for (int d4 = 0; d4 < 8; ++d4) {
          float4 kv = sk4[j][d4 ^ part];
          l += q4[d4].x * kv.x;
          l += q4[d4].y * kv.y;
          l += q4[d4].z * kv.z;
          l += q4[d4].w * kv.w;
        }
        l = sm[j] ? l + __ldg(b + k0 + j) : NEG_INF;
        logits[jj] = l;
        nm = fmaxf(nm, l);
      }
      if (k0 > 0 && nm > mx) {
        float f = expf(mx - nm);
        sum *= f;
        #pragma unroll
        for (int d4 = 0; d4 < 8; ++d4) {
          acc4[d4].x *= f; acc4[d4].y *= f; acc4[d4].z *= f; acc4[d4].w *= f;
        }
      }
      mx = nm;
      #pragma unroll
      for (int jj = 0; jj < KS; ++jj) {
        const int j = jj * SPLIT + part;
        float e = expf(logits[jj] - mx);
        sum += e;
        #pragma unroll
        for (int d4 = 0; d4 < 8; ++d4) {
          float4 vv = sv4[j][d4 ^ part];
          acc4[d4].x += e * vv.x;
          acc4[d4].y += e * vv.y;
          acc4[d4].z += e * vv.z;
          acc4[d4].w += e * vv.w;
        }
      }
      __syncthreads();
    }
    #pragma unroll
    for (int o = 1; o < SPLIT; o <<= 1) {
      float om = __shfl_xor_sync(~0u, mx, o), os = __shfl_xor_sync(~0u, sum, o);
      float nm = fmaxf(mx, om), f = expf(mx - nm), fo = expf(om - nm);
      sum = sum * f + os * fo;
      #pragma unroll
      for (int d4 = 0; d4 < 8; ++d4) {
        float ox = __shfl_xor_sync(~0u, acc4[d4].x, o);
        float oy = __shfl_xor_sync(~0u, acc4[d4].y, o);
        float oz = __shfl_xor_sync(~0u, acc4[d4].z, o);
        float ow = __shfl_xor_sync(~0u, acc4[d4].w, o);
        acc4[d4].x = acc4[d4].x * f + ox * fo;
        acc4[d4].y = acc4[d4].y * f + oy * fo;
        acc4[d4].z = acc4[d4].z * f + oz * fo;
        acc4[d4].w = acc4[d4].w * f + ow * fo;
      }
      mx = nm;
    }
    if (part == 0) {
      float inv = 1.f / sum;
      const float4* gp4 = reinterpret_cast<const float4*>(qkvg + qa * 4 * D + 3 * D + h * 32);
      TY* outRow = out + qa * D + h * 32;
      #pragma unroll
      for (int d4 = 0; d4 < 8; ++d4) {
        float4 gv = __ldg(gp4 + d4);
        storeVec4<TY>(outRow, d4,
                      (acc4[d4].x * inv) * sigm(gv.x),
                      (acc4[d4].y * inv) * sigm(gv.y),
                      (acc4[d4].z * inv) * sigm(gv.z),
                      (acc4[d4].w * inv) * sigm(gv.w));
      }
    }
  } else {
    __shared__ float sk[WK][HD + 1], sv[WK][HD + 1];
    __shared__ int sm[WK];
    for (int e = threadIdx.x; e < WK * HD; e += blockDim.x) {
      int j = e / HD, d = e % HD, ka = w * WQ - CTX + j;
      bool ok = ka >= 0 && ka < N;
      sk[j][d] = ok ? qkvg[(base + ka) * 4 * D + D + h * HD + d] : 0.f;
      sv[j][d] = ok ? qkvg[(base + ka) * 4 * D + 2 * D + h * HD + d] : 0.f;
    }
    for (int j = threadIdx.x; j < WK; j += blockDim.x) { int ka = w * WQ - CTX + j; sm[j] = ka >= 0 && ka < N && pad[ka]; }
    __syncthreads();
    float q[HD];
    for (int d = 0; d < HD; ++d) q[d] = (qkvg[qa * 4 * D + h * HD + d] + qBias[h * HD + d]) * scale;
    float mx = -INFINITY, sum = 0, acc[HD];
    for (int d = 0; d < HD; ++d) acc[d] = 0;
    for (int jj = 0; jj < WK / SPLIT; ++jj) {
      const int j = jj * SPLIT + part;
      float l = 0;
#pragma unroll
      for (int d = 0; d < HD; ++d) l += q[d] * sk[j][d];
      l = sm[j] ? l + b[j] : NEG_INF;
      float nm = fmaxf(mx, l), f = expf(mx - nm), e = expf(l - nm);
      sum = sum * f + e;
#pragma unroll
      for (int d = 0; d < HD; ++d) acc[d] = acc[d] * f + e * sv[j][d];
      mx = nm;
    }
#pragma unroll
    for (int o = 1; o < SPLIT; o <<= 1) {
      float om = __shfl_xor_sync(~0u, mx, o), os = __shfl_xor_sync(~0u, sum, o);
      float nm = fmaxf(mx, om), f = expf(mx - nm), fo = expf(om - nm);
      sum = sum * f + os * fo;
#pragma unroll
      for (int d = 0; d < HD; ++d) { float oa = __shfl_xor_sync(~0u, acc[d], o); acc[d] = acc[d] * f + oa * fo; }
      mx = nm;
    }
    if (part == 0) {
      for (int d = 0; d < HD; ++d) {
        float v = (acc[d] / sum) * sigm(qkvg[qa * 4 * D + 3 * D + h * HD + d]);
        if constexpr (std::is_same_v<TY, float>) out[qa * D + h * HD + d] = v;
        else out[qa * D + h * HD + d] = toH<TY>(v);
      }
    }
  }
}

// Tensor Core m16n8k16 sliding-window atom attention for HD == 32, WQ == 32, WK == 128:
// 2 warps (64 threads, 16 queries per warp) per (window, head), Ks[128*32] and Vs[128*32] in XOR-swizzled half smem.
__global__ void __launch_bounds__(64, 6)
windowAttentionMMAK(const half* __restrict__ qkvg, const float* __restrict__ qBias,
                    const float* __restrict__ bias, const int* __restrict__ pad,
                    half* __restrict__ out, int N, int H) {
  constexpr int LD = 32;
  __shared__ __align__(16) half Ks[WK * LD];
  __shared__ __align__(16) half Vs[WK * LD];
  const int wg = blockIdx.x, h = blockIdx.y, D = H * 32;
  const unsigned NW = (unsigned)(N / WQ), smp = (unsigned)wg / NW, w = (unsigned)wg - smp * NW;
  const size_t base = (size_t)smp * N;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  auto cl = [](float v) { return fminf(fmaxf(v, -65504.f), 65504.f); };
  auto packCl2 = [&](float a, float b) -> uint32_t { return pack2(cl(a), cl(b)); };
  const int kaBase = (int)w * WQ - CTX;
  #pragma unroll
  for (int i = 0; i < WK / 16; ++i) {
    int key = i * 16 + (threadIdx.x >> 2), c = (threadIdx.x & 3) * 8, cs = c ^ ((key & 3) * 8);
    int ka = kaBase + key;
    bool live = ((unsigned)ka < (unsigned)N && __ldg(pad + ka) != 0);
    const half* kv = qkvg + (base + (live ? ka : 0)) * 4 * D + h * 32 + c;
    cpAsync16(Ks + key * LD + cs, kv + D, live);
    cpAsync16(Vs + key * LD + cs, kv + 2 * D, live);
  }
  cpCommit();
  const int q0 = warp * 16, r0 = q0 + g, r1 = r0 + 8;
  const size_t i0 = base + w * WQ + r0, i1 = base + w * WQ + r1;
  const float qs = (1.f / sqrtf(32.f)) * LOG2E;
  uint32_t qa[2][4];
  #pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    int e = ks * 16 + tig * 2;
    float2 qb0 = __ldg(reinterpret_cast<const float2*>(qBias + h * 32 + e));
    float2 qb1 = __ldg(reinterpret_cast<const float2*>(qBias + h * 32 + e + 8));
    float2 q00 = __half22float2(__ldg(reinterpret_cast<const half2*>(qkvg + i0 * 4 * D + h * 32 + e)));
    float2 q10 = __half22float2(__ldg(reinterpret_cast<const half2*>(qkvg + i1 * 4 * D + h * 32 + e)));
    float2 q01 = __half22float2(__ldg(reinterpret_cast<const half2*>(qkvg + i0 * 4 * D + h * 32 + e + 8)));
    float2 q11 = __half22float2(__ldg(reinterpret_cast<const half2*>(qkvg + i1 * 4 * D + h * 32 + e + 8)));
    qa[ks][0] = packCl2((q00.x + qb0.x) * qs, (q00.y + qb0.y) * qs);
    qa[ks][1] = packCl2((q10.x + qb0.x) * qs, (q10.y + qb0.y) * qs);
    qa[ks][2] = packCl2((q01.x + qb1.x) * qs, (q01.y + qb1.y) * qs);
    qa[ks][3] = packCl2((q11.x + qb1.x) * qs, (q11.y + qb1.y) * qs);
  }
  const float* pl0 = bias + (((size_t)w * H + h) * WQ + r0) * WK;
  const float* pl1 = bias + (((size_t)w * H + h) * WQ + r1) * WK;
  float2 b0 = __ldg(reinterpret_cast<const float2*>(pl0 + tig * 2));
  float2 b1 = __ldg(reinterpret_cast<const float2*>(pl1 + tig * 2));
  int ka0 = kaBase + tig * 2;
  int2 pm = ((unsigned)ka0 < (unsigned)N) ? __ldg(reinterpret_cast<const int2*>(pad + ka0)) : make_int2(0, 0);
  cpWait<0>();
  __syncthreads();
  float sv[WK / 8][4];
  float m0 = -1e30f, m1 = -1e30f;
  uint32_t kb[2][4];
  ldsm4(kb[0], Ks + (lane & 7) * LD + ((lane >> 3) ^ (lane & 3)) * 8);
  #pragma unroll
  for (int nt = 0; nt < WK / 8; ++nt) {
    const int cur = nt & 1;
    sv[nt][0] = (pm.x ? b0.x : NEG_INF) * LOG2E;
    sv[nt][1] = (pm.y ? b0.y : NEG_INF) * LOG2E;
    sv[nt][2] = (pm.x ? b1.x : NEG_INF) * LOG2E;
    sv[nt][3] = (pm.y ? b1.y : NEG_INF) * LOG2E;
    if (nt + 1 < WK / 8) {
      ldsm4(kb[cur ^ 1], Ks + ((nt + 1) * 8 + (lane & 7)) * LD + ((lane >> 3) ^ (lane & 3)) * 8);
      int nkNext = (nt + 1) * 8 + tig * 2;
      b0 = __ldg(reinterpret_cast<const float2*>(pl0 + nkNext));
      b1 = __ldg(reinterpret_cast<const float2*>(pl1 + nkNext));
      int kaNext = kaBase + nkNext;
      pm = ((unsigned)kaNext < (unsigned)N) ? __ldg(reinterpret_cast<const int2*>(pad + kaNext)) : make_int2(0, 0);
    }
    mma16816(sv[nt], qa[0], kb[cur][0], kb[cur][1]);
    mma16816(sv[nt], qa[1], kb[cur][2], kb[cur][3]);
    m0 = fmaxf(m0, fmaxf(sv[nt][0], sv[nt][1]));
    m1 = fmaxf(m1, fmaxf(sv[nt][2], sv[nt][3]));
  }
  half2 ga[4], gb[4];
  #pragma unroll
  for (int et = 0; et < 4; ++et) {
    int e = et * 8 + tig * 2;
    ga[et] = __ldg(reinterpret_cast<const half2*>(qkvg + i0 * 4 * D + 3 * D + h * 32 + e));
    gb[et] = __ldg(reinterpret_cast<const half2*>(qkvg + i1 * 4 * D + 3 * D + h * 32 + e));
  }
  m0 = fmaxf(m0, __shfl_xor_sync(~0u, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(~0u, m0, 2));
  m1 = fmaxf(m1, __shfl_xor_sync(~0u, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(~0u, m1, 2));
  float l0 = 0.f, l1 = 0.f;
  #pragma unroll
  for (int t = 0; t < WK / 16; ++t) {
    #pragma unroll
    for (int s2 = 0; s2 < 2; ++s2) {
      int nt = 2 * t + s2;
      sv[nt][0] = exp2f(sv[nt][0] - m0); sv[nt][1] = exp2f(sv[nt][1] - m0);
      sv[nt][2] = exp2f(sv[nt][2] - m1); sv[nt][3] = exp2f(sv[nt][3] - m1);
      l0 += sv[nt][0] + sv[nt][1]; l1 += sv[nt][2] + sv[nt][3];
    }
    uint32_t p0 = pack2(sv[2 * t][0], sv[2 * t][1]), p1 = pack2(sv[2 * t][2], sv[2 * t][3]);
    uint32_t p2 = pack2(sv[2 * t + 1][0], sv[2 * t + 1][1]), p3 = pack2(sv[2 * t + 1][2], sv[2 * t + 1][3]);
    uint32_t* pa = reinterpret_cast<uint32_t*>(sv[t]);
    pa[0] = p0; pa[1] = p1; pa[2] = p2; pa[3] = p3;
  }
  l0 += __shfl_xor_sync(~0u, l0, 1); l0 += __shfl_xor_sync(~0u, l0, 2);
  l1 += __shfl_xor_sync(~0u, l1, 1); l1 += __shfl_xor_sync(~0u, l1, 2);
  float o[4][4] = {};
  uint32_t vb0[2][4], vb1[2][4];
  ldsm4t(vb0[0], Vs + (((lane >> 3) & 1) * 8 + (lane & 7)) * LD + ((lane >> 4) ^ (lane & 3)) * 8);
  ldsm4t(vb1[0], Vs + (((lane >> 3) & 1) * 8 + (lane & 7)) * LD + ((2 + (lane >> 4)) ^ (lane & 3)) * 8);
  #pragma unroll
  for (int t = 0; t < WK / 16; ++t) {
    const int cur = t & 1;
    if (t + 1 < WK / 16) {
      ldsm4t(vb0[cur ^ 1], Vs + ((t + 1) * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LD + ((lane >> 4) ^ (lane & 3)) * 8);
      ldsm4t(vb1[cur ^ 1], Vs + ((t + 1) * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LD + ((2 + (lane >> 4)) ^ (lane & 3)) * 8);
    }
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(sv[t]);
    mma16816(o[0], pa, vb0[cur][0], vb0[cur][1]);
    mma16816(o[1], pa, vb0[cur][2], vb0[cur][3]);
    mma16816(o[2], pa, vb1[cur][0], vb1[cur][1]);
    mma16816(o[3], pa, vb1[cur][2], vb1[cur][3]);
  }
  float inv0 = 1.f / l0, inv1 = 1.f / l1;
  #pragma unroll
  for (int et = 0; et < 4; ++et) {
    int e = et * 8 + tig * 2;
    float2 g0 = __half22float2(ga[et]), g1 = __half22float2(gb[et]);
    *reinterpret_cast<half2*>(out + i0 * D + h * 32 + e) =
      __floats2half2_rn(cl((o[et][0] * inv0) * fastSigm(g0.x)), cl((o[et][1] * inv0) * fastSigm(g0.y)));
    *reinterpret_cast<half2*>(out + i1 * D + h * 32 + e) =
      __floats2half2_rn(cl((o[et][2] * inv1) * fastSigm(g1.x)), cl((o[et][3] * inv1) * fastSigm(g1.y)));
  }
}

// Tensor Core m16n8k16 token attention for HD == 48: 2 warps (64 threads, BQ == 32 queries per block),
// key tiles of BK == 32 in conflict-free LD == 56 half smem, online softmax across tiles, fused sigm(g) gating.
__global__ void __launch_bounds__(64, 8)
tokenAttention48MMAK(const half* __restrict__ qkvg, const float* __restrict__ qBias,
                     const float* __restrict__ bias, const int* __restrict__ keyMask,
                     half* __restrict__ out, int T, int heads) {
  constexpr int BQ = 32, BK = 32, HD = 48, LD = 56;
  __shared__ __align__(16) half Ks[BK * LD];
  __shared__ __align__(16) half Vs[BK * LD];
  __shared__ uint8_t sm[BK];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = blockIdx.x * BQ + warp * 16, r0 = q0 + g, r1 = r0 + 8;
  const int h = blockIdx.y, smp = blockIdx.z, D = heads * HD;
  const float qs = (1.f / sqrtf(48.f)) * LOG2E;
  auto cl = [](float v) { return fminf(fmaxf(v, -65504.f), 65504.f); };
  auto packCl2 = [&](float a, float b) -> uint32_t { return pack2(cl(a), cl(b)); };
  uint32_t qa[3][4];
  #pragma unroll
  for (int ks = 0; ks < 3; ++ks) {
    int e = ks * 16 + tig * 2;
    float2 qb0 = __ldg(reinterpret_cast<const float2*>(qBias + h * HD + e));
    float2 qb1 = __ldg(reinterpret_cast<const float2*>(qBias + h * HD + e + 8));
    if (r0 < T) {
      const half* qp0 = qkvg + ((size_t)(smp * T + r0) * 4 * D + h * HD + e);
      float2 q00 = __half22float2(__ldg(reinterpret_cast<const half2*>(qp0)));
      float2 q01 = __half22float2(__ldg(reinterpret_cast<const half2*>(qp0 + 8)));
      qa[ks][0] = packCl2((q00.x + qb0.x) * qs, (q00.y + qb0.y) * qs);
      qa[ks][2] = packCl2((q01.x + qb1.x) * qs, (q01.y + qb1.y) * qs);
    } else {
      qa[ks][0] = 0; qa[ks][2] = 0;
    }
    if (r1 < T) {
      const half* qp1 = qkvg + ((size_t)(smp * T + r1) * 4 * D + h * HD + e);
      float2 q10 = __half22float2(__ldg(reinterpret_cast<const half2*>(qp1)));
      float2 q11 = __half22float2(__ldg(reinterpret_cast<const half2*>(qp1 + 8)));
      qa[ks][1] = packCl2((q10.x + qb0.x) * qs, (q10.y + qb0.y) * qs);
      qa[ks][3] = packCl2((q11.x + qb1.x) * qs, (q11.y + qb1.y) * qs);
    } else {
      qa[ks][1] = 0; qa[ks][3] = 0;
    }
  }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float o[6][4] = {};
  const float* bRow0 = (r0 < T) ? (bias + ((size_t)h * T + r0) * T) : nullptr;
  const float* bRow1 = (r1 < T) ? (bias + ((size_t)h * T + r1) * T) : nullptr;
  for (int k0 = 0; k0 < T; k0 += BK) {
    #pragma unroll
    for (int m = 0; m < 3; ++m) {
      int u = m * 64 + threadIdx.x;
      int kj = u / 6, c = (u - kj * 6) * 8;
      int ka = k0 + kj;
      bool live = (ka < T && __ldg(keyMask + ka) != 0);
      if (c == 0) sm[kj] = live ? 1 : 0;
      const half* kv = qkvg + ((size_t)(smp * T + (live ? ka : 0)) * 4 * D + h * HD + c);
      cpAsync16(Ks + kj * LD + c, kv + D, live);
      cpAsync16(Vs + kj * LD + c, kv + 2 * D, live);
    }
    cpCommit();
    cpWait<0>();
    __syncthreads();
    if (q0 < T) {
      float sv[4][4];
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        int nkey = nt * 8 + tig * 2;
        uint8_t m_lo = sm[nkey], m_hi = sm[nkey + 1];
        sv[nt][0] = (m_lo ? (bRow0 ? __ldg(bRow0 + k0 + nkey)     : 0.f) : NEG_INF) * LOG2E;
        sv[nt][1] = (m_hi ? (bRow0 ? __ldg(bRow0 + k0 + nkey + 1) : 0.f) : NEG_INF) * LOG2E;
        sv[nt][2] = (m_lo ? (bRow1 ? __ldg(bRow1 + k0 + nkey)     : 0.f) : NEG_INF) * LOG2E;
        sv[nt][3] = (m_hi ? (bRow1 ? __ldg(bRow1 + k0 + nkey + 1) : 0.f) : NEG_INF) * LOG2E;
      }
      #pragma unroll
      for (int t = 0; t < 2; ++t) {
        #pragma unroll
        for (int ks = 0; ks < 3; ++ks) {
          uint32_t kb[4];
          ldsm4(kb, Ks + (t * 16 + (lane >> 4) * 8 + (lane & 7)) * LD + ks * 16 + ((lane >> 3) & 1) * 8);
          mma16816(sv[2 * t],     qa[ks], kb[0], kb[1]);
          mma16816(sv[2 * t + 1], qa[ks], kb[2], kb[3]);
        }
      }
      float nm0 = m0, nm1 = m1;
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        nm0 = fmaxf(nm0, fmaxf(sv[nt][0], sv[nt][1]));
        nm1 = fmaxf(nm1, fmaxf(sv[nt][2], sv[nt][3]));
      }
      nm0 = fmaxf(nm0, __shfl_xor_sync(~0u, nm0, 1)); nm0 = fmaxf(nm0, __shfl_xor_sync(~0u, nm0, 2));
      nm1 = fmaxf(nm1, __shfl_xor_sync(~0u, nm1, 1)); nm1 = fmaxf(nm1, __shfl_xor_sync(~0u, nm1, 2));
      if (k0 > 0) {
        float f0 = exp2f(m0 - nm0), f1 = exp2f(m1 - nm1);
        l0 *= f0; l1 *= f1;
        #pragma unroll
        for (int et = 0; et < 6; ++et) {
          o[et][0] *= f0; o[et][1] *= f0;
          o[et][2] *= f1; o[et][3] *= f1;
        }
      }
      m0 = nm0; m1 = nm1;
      float s0 = 0.f, s1 = 0.f;
      #pragma unroll
      for (int t = 0; t < 2; ++t) {
        #pragma unroll
        for (int s2 = 0; s2 < 2; ++s2) {
          int nt = 2 * t + s2;
          sv[nt][0] = exp2f(sv[nt][0] - m0); sv[nt][1] = exp2f(sv[nt][1] - m0);
          sv[nt][2] = exp2f(sv[nt][2] - m1); sv[nt][3] = exp2f(sv[nt][3] - m1);
          s0 += sv[nt][0] + sv[nt][1]; s1 += sv[nt][2] + sv[nt][3];
        }
        uint32_t p0 = pack2(sv[2 * t][0], sv[2 * t][1]), p1 = pack2(sv[2 * t][2], sv[2 * t][3]);
        uint32_t p2 = pack2(sv[2 * t + 1][0], sv[2 * t + 1][1]), p3 = pack2(sv[2 * t + 1][2], sv[2 * t + 1][3]);
        uint32_t* pa = reinterpret_cast<uint32_t*>(sv[t]);
        pa[0] = p0; pa[1] = p1; pa[2] = p2; pa[3] = p3;
      }
      s0 += __shfl_xor_sync(~0u, s0, 1); s0 += __shfl_xor_sync(~0u, s0, 2);
      s1 += __shfl_xor_sync(~0u, s1, 1); s1 += __shfl_xor_sync(~0u, s1, 2);
      l0 += s0; l1 += s1;
      #pragma unroll
      for (int t = 0; t < 2; ++t) {
        const uint32_t* pa = reinterpret_cast<const uint32_t*>(sv[t]);
        #pragma unroll
        for (int eg = 0; eg < 3; ++eg) {
          uint32_t vb[4];
          ldsm4t(vb, Vs + (t * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LD + eg * 16 + (lane >> 4) * 8);
          mma16816(o[2 * eg + 0], pa, vb[0], vb[1]);
          mma16816(o[2 * eg + 1], pa, vb[2], vb[3]);
        }
      }
    }
    __syncthreads();
  }
  float inv0 = 1.f / l0, inv1 = 1.f / l1;
  if (r0 < T) {
    const half* gp0 = qkvg + ((size_t)(smp * T + r0) * 4 * D + 3 * D + h * HD);
    half* out0 = out + ((size_t)(smp * T + r0) * D + h * HD);
    #pragma unroll
    for (int et = 0; et < 6; ++et) {
      int e = et * 8 + tig * 2;
      float2 g0 = __half22float2(__ldg(reinterpret_cast<const half2*>(gp0 + e)));
      *reinterpret_cast<half2*>(out0 + e) =
        __floats2half2_rn(cl((o[et][0] * inv0) * fastSigm(g0.x)), cl((o[et][1] * inv0) * fastSigm(g0.y)));
    }
  }
  if (r1 < T) {
    const half* gp1 = qkvg + ((size_t)(smp * T + r1) * 4 * D + 3 * D + h * HD);
    half* out1 = out + ((size_t)(smp * T + r1) * D + h * HD);
    #pragma unroll
    for (int et = 0; et < 6; ++et) {
      int e = et * 8 + tig * 2;
      float2 g1 = __half22float2(__ldg(reinterpret_cast<const half2*>(gp1 + e)));
      *reinterpret_cast<half2*>(out1 + e) =
        __floats2half2_rn(cl((o[et][2] * inv1) * fastSigm(g1.x)), cl((o[et][3] * inv1) * fastSigm(g1.y)));
    }
  }
}

// Fused token attention for HD == 48: 16 queries per block (64 threads, 4 threads per query owning 12 channels = 3 float4s),
// shared-memory key/value tiles of BK = 32 keys, chunked online softmax across key tiles, and fused sigm(g) gating + TY output.
template <class TY = float>
__global__ void __launch_bounds__(64, 8)
tokenAttention48K(const float* __restrict__ qkvg, const float* __restrict__ qBias,
                  const float* __restrict__ bias, const int* __restrict__ keyMask,
                  TY* __restrict__ out, int T, int heads) {
  constexpr int BQ = 16, BK = 32, HD = 48;
  const int qiLocal = threadIdx.x >> 2, part = threadIdx.x & 3;
  const int qi = blockIdx.x * BQ + qiLocal, h = blockIdx.y, smp = blockIdx.z;
  const int D = heads * HD;
  __shared__ float4 sk4[BK][12], sv4[BK][12];
  __shared__ int sm[BK];
  float q[12];
  const float scale = 1.f / sqrtf(48.f);
  if (qi < T) {
    const float4* qp4 = reinterpret_cast<const float4*>(qkvg + ((size_t)(smp * T + qi) * 4 * D + h * HD)) + part * 3;
    const float4* qb4 = reinterpret_cast<const float4*>(qBias + h * HD) + part * 3;
    #pragma unroll
    for (int m = 0; m < 3; ++m) {
      float4 qv = __ldg(qp4 + m), bv = __ldg(qb4 + m);
      q[m * 4 + 0] = qv.x + bv.x;
      q[m * 4 + 1] = qv.y + bv.y;
      q[m * 4 + 2] = qv.z + bv.z;
      q[m * 4 + 3] = qv.w + bv.w;
    }
  } else {
    #pragma unroll
    for (int d = 0; d < 12; ++d) q[d] = 0.f;
  }
  float mx = -INFINITY, sum = 0.f, acc[12];
  #pragma unroll
  for (int d = 0; d < 12; ++d) acc[d] = 0.f;
  const float* bRow = (qi < T) ? (bias + ((size_t)h * T + qi) * T) : nullptr;
  for (int k0 = 0; k0 < T; k0 += BK) {
    const int kLim = min(BK, T - k0);
    for (int idx = threadIdx.x; idx < kLim * 12; idx += 64) {
      int kj = idx / 12, d4 = idx - kj * 12;
      int ka = k0 + kj;
      const float4* kv4 = reinterpret_cast<const float4*>(qkvg + ((size_t)(smp * T + ka) * 4 * D + h * HD));
      sk4[kj][d4] = __ldg(kv4 + (D >> 2) + d4);
      sv4[kj][d4] = __ldg(kv4 + (2 * D >> 2) + d4);
    }
    if (threadIdx.x < kLim) sm[threadIdx.x] = __ldg(keyMask + k0 + threadIdx.x);
    __syncthreads();
    int kj = 0;
    for (; kj + 8 <= kLim; kj += 8) {
      float l8[8];
      float nm = mx;
      #pragma unroll
      for (int u = 0; u < 8; ++u) {
        int j = kj + u;
        float4 k0v = sk4[j][part * 3 + 0];
        float4 k1v = sk4[j][part * 3 + 1];
        float4 k2v = sk4[j][part * 3 + 2];
        float l = q[0] * k0v.x + q[1] * k0v.y + q[2] * k0v.z + q[3] * k0v.w
                + q[4] * k1v.x + q[5] * k1v.y + q[6] * k1v.z + q[7] * k1v.w
                + q[8] * k2v.x + q[9] * k2v.y + q[10] * k2v.z + q[11] * k2v.w;
        l += __shfl_xor_sync(~0u, l, 1);
        l += __shfl_xor_sync(~0u, l, 2);
        l *= scale;
        float bVal = bRow ? __ldg(bRow + k0 + j) : 0.f;
        l = sm[j] ? l + bVal : NEG_INF;
        l8[u] = l;
        nm = fmaxf(nm, l);
      }
      if (nm > mx) {
        float f = (mx == -INFINITY) ? 0.f : expf(mx - nm);
        sum *= f;
        #pragma unroll
        for (int d = 0; d < 12; ++d) acc[d] *= f;
        mx = nm;
      }
      #pragma unroll
      for (int u = 0; u < 8; ++u) {
        int j = kj + u;
        float e = expf(l8[u] - mx);
        sum += e;
        float4 v0v = sv4[j][part * 3 + 0];
        float4 v1v = sv4[j][part * 3 + 1];
        float4 v2v = sv4[j][part * 3 + 2];
        acc[0]  += e * v0v.x; acc[1]  += e * v0v.y; acc[2]  += e * v0v.z; acc[3]  += e * v0v.w;
        acc[4]  += e * v1v.x; acc[5]  += e * v1v.y; acc[6]  += e * v1v.z; acc[7]  += e * v1v.w;
        acc[8]  += e * v2v.x; acc[9]  += e * v2v.y; acc[10] += e * v2v.z; acc[11] += e * v2v.w;
      }
    }
    for (; kj < kLim; ++kj) {
      float4 k0v = sk4[kj][part * 3 + 0];
      float4 k1v = sk4[kj][part * 3 + 1];
      float4 k2v = sk4[kj][part * 3 + 2];
      float l = q[0] * k0v.x + q[1] * k0v.y + q[2] * k0v.z + q[3] * k0v.w
              + q[4] * k1v.x + q[5] * k1v.y + q[6] * k1v.z + q[7] * k1v.w
              + q[8] * k2v.x + q[9] * k2v.y + q[10] * k2v.z + q[11] * k2v.w;
      l += __shfl_xor_sync(~0u, l, 1);
      l += __shfl_xor_sync(~0u, l, 2);
      l *= scale;
      float bVal = bRow ? __ldg(bRow + k0 + kj) : 0.f;
      l = sm[kj] ? l + bVal : NEG_INF;
      float nm = fmaxf(mx, l), f = expf(mx - nm), e = expf(l - nm);
      sum = sum * f + e;
      mx = nm;
      float4 v0v = sv4[kj][part * 3 + 0];
      float4 v1v = sv4[kj][part * 3 + 1];
      float4 v2v = sv4[kj][part * 3 + 2];
      acc[0]  = acc[0]  * f + e * v0v.x; acc[1]  = acc[1]  * f + e * v0v.y;
      acc[2]  = acc[2]  * f + e * v0v.z; acc[3]  = acc[3]  * f + e * v0v.w;
      acc[4]  = acc[4]  * f + e * v1v.x; acc[5]  = acc[5]  * f + e * v1v.y;
      acc[6]  = acc[6]  * f + e * v1v.z; acc[7]  = acc[7]  * f + e * v1v.w;
      acc[8]  = acc[8]  * f + e * v2v.x; acc[9]  = acc[9]  * f + e * v2v.y;
      acc[10] = acc[10] * f + e * v2v.z; acc[11] = acc[11] * f + e * v2v.w;
    }
    __syncthreads();
  }
  if (qi < T) {
    float inv = 1.f / sum;
    const float4* gp4 = reinterpret_cast<const float4*>(qkvg + ((size_t)(smp * T + qi) * 4 * D + 3 * D + h * HD)) + part * 3;
    TY* outPart = out + ((size_t)(smp * T + qi) * D + h * HD) + part * 12;
    #pragma unroll
    for (int m = 0; m < 3; ++m) {
      float4 gv = __ldg(gp4 + m);
      storeVec4<TY>(outPart, m,
                    (acc[m * 4 + 0] * inv) * sigm(gv.x),
                    (acc[m * 4 + 1] * inv) * sigm(gv.y),
                    (acc[m * 4 + 2] * inv) * sigm(gv.z),
                    (acc[m * 4 + 3] * inv) * sigm(gv.w));
    }
  }
}

// bias + key mask + softmax over the last axis of [H, T, T] logits, in place; a warp a row
__global__ void softmaxBiasK(float* logits, const float* bias, const int* keyMask, size_t rows, int T) {
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float* l = logits + row * T; const float* b = bias + row * T;
  float mx = -INFINITY;
  for (int j = lane; j < T; j += 32) { float v = keyMask[j] ? l[j] + b[j] : NEG_INF; l[j] = v; mx = fmaxf(mx, v); }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  float s = 0;
  for (int j = lane; j < T; j += 32) { float e = expf(l[j] - mx); l[j] = e; s += e; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float inv = 1.f / s;
  for (int j = lane; j < T; j += 32) l[j] *= inv;
}
// AttentionPairBias over all T tokens of each of `samples` stacked samples: qkvg [samples * T, 4D] (q, k, v, g),
// bias [H, T, T] shared; out [samples * T, D] (already gated by sigm(g))
template <class TY = float>
inline void tokenAttention(const float* qkvg, const float* qBias, const float* bias, TY* out, int T, int D, int heads,
                           int samples = 1) {
  const int HD = D / heads;
  if (HD == 48 && (((uintptr_t)qkvg | (uintptr_t)qBias) & 15) == 0 && (((uintptr_t)out & 7) == 0)) {
    dim3 grid((unsigned)((T + 15) / 16), (unsigned)heads, (unsigned)samples);
    tokenAttention48K<TY><<<grid, 64, 0, STREAM>>>(qkvg, qBias, bias, Idev("token_pad_mask"), out, T, heads);
    return;
  }
  size_t R = (size_t)samples * T;
  float* q = scratch<float>("be2.ta.q", R * D);
  float* g = scratch<float>("be2.ta.g", R * D);
  float* o = std::is_same_v<TY, float> ? reinterpret_cast<float*>(out) : scratch<float>("be2.ta.o", R * D);
  CK(cudaMemcpy2DAsync(q, D * 4, qkvg, 4 * D * 4, D * 4, R, cudaMemcpyDeviceToDevice, STREAM));
  CK(cudaMemcpy2DAsync(g, D * 4, qkvg + 3 * D, 4 * D * 4, D * 4, R, cudaMemcpyDeviceToDevice, STREAM));
  addBiasRows(q, qBias, R, D);
  float* logits = scratch<float>("be2.ta.logits", (size_t)heads * T * T);
  const float alpha = 1.f / sqrtf((float)HD), zero = 0.f, one = 1.f;
  for (int s = 0; s < samples; ++s) {
    const float* qs = q + (size_t)s * T * D; const float* kv = qkvg + (size_t)s * T * 4 * D;
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HD, &alpha, kv + D, 4 * D, HD, qs, D, HD, &zero,
                                 logits, T, (long long)T * T, heads));
    softmaxBiasK<<<(unsigned)(((size_t)heads * T + 7) / 8), 256, 0, STREAM>>>(logits, bias, Idev("token_pad_mask"),
                                                                              (size_t)heads * T, T);
    CB(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, HD, T, T, &one, kv + 2 * D, 4 * D, HD, logits, T,
                                 (long long)T * T, &zero, o + (size_t)s * T * D, D, HD, heads));
  }
  size_t nrd = R * D;
  if constexpr (std::is_same_v<TY, half>) {
    sigmoidMulToK<<<blocks(nrd), 256, 0, STREAM>>>(o, g, out, nrd);
  } else {
    sigmoidMulK<<<blocks(nrd), 256, 0, STREAM>>>(o, g, out, nrd);
  }
}

// ---------------------------------------------------------------- the conditioned transformer layer
// ConditionedTransformerLayer (adaLN-Zero), in place on x [R, D]:
//   cond = cond_2(silu(LN(c))) -> alpha_a, beta_a, gamma_a, alpha_m, beta_m, gamma_m
//   x += gamma_a * attention((1 + alpha_a) LN(x) + beta_a);  x += gamma_m * transition((1 + alpha_m) LN(x) + beta_m)
// `windowed`: the sliding-window atom attention over the windowed pair's bias; else the full token attention over
// [H, T, T]. `att` names the attention's parameters, `l` the layer of a stacked module (-1: unstacked).
// 🔴 THE CONDITIONING IS COMPUTED ON ITS OWN ROWS AND MAPPED: it has Rc rows and `map` sends each of x's R rows to
// one (null: the identity). The denoiser's atom stream is conditioned on its token's row of the single conditioning
// broadcast to atoms, so its cond_2 is the same for every atom of a token - and every sample of a step: computed once
// a token rather than once an atom of every sample, and exact, because a broadcast row is the row. (A padding atom
// reads row 0; nothing reads a padding atom's result.)
struct CondLayer { std::string p, att; int l; bool windowed; int heads; };
inline const float* lw(const CondLayer& L, const std::string& n) { return L.l < 0 ? P(L.p + n) : PL(L.p + n, L.l); }
inline const float* lwOpt(const CondLayer& L, const std::string& n) {
  return L.l < 0 ? Popt(L.p + n) : PLopt(L.p + n, L.l);
}
// a layer's conditioning, [Rc, 6D]
template <class TH>
inline float* layerCondPre(const CondLayer& L, const TH* cn, int Rc, int Cc, int D, const char* name) {
  float* cond = scratch<float>(name, (size_t)Rc * 6 * D);
  if constexpr (std::is_same_v<TH, half>) {
    linearPre<half>(cn, halfMirror<half>(lw(L, "cond_2/weight"), (size_t)6 * D * Cc), cond, Rc, Cc, 6 * D);
  } else {
    linear(cn, lw(L, "cond_2/weight"), cond, Rc, Cc, 6 * D);
  }
  if (const float* b = lwOpt(L, "cond_2/bias")) addBiasRows(cond, b, Rc, 6 * D);
  return cond;
}
inline float* layerCond(const CondLayer& L, const float* c, int Rc, int Cc, int D, const char* name) {
  float* cond = scratch<float>(name, (size_t)Rc * 6 * D);
  if (GEMM_HALF && Cc >= 16 && 6 * D >= 16) {
    half* cnh = scratch<half>("be2.cl.cnh", (size_t)Rc * Cc);
    lnTo<half, true>(c, cnh, Rc, Cc, lw(L, "cond_0/weight"), lwOpt(L, "cond_0/bias"));
    linearPre<half>(cnh, halfMirror<half>(lw(L, "cond_2/weight"), (size_t)6 * D * Cc), cond, Rc, Cc, 6 * D);
  } else {
    float* cn = scratch<float>("be2.cl.cn", (size_t)Rc * Cc);
    lnTo<float, true>(c, cn, Rc, Cc, lw(L, "cond_0/weight"), lwOpt(L, "cond_0/bias"));
    linear(cn, lw(L, "cond_2/weight"), cond, Rc, Cc, 6 * D);
  }
  if (const float* b = lwOpt(L, "cond_2/bias")) addBiasRows(cond, b, Rc, 6 * D);
  return cond;
}
// the layer on x [R, D] (samples stacked, `perSample` rows each), its conditioning rows `cond` reached through `map`
// GEMM_HALF: every GEMM input written in f16 by the kernel before it (adaLN, the gated attention output)
inline void condLayerHalf(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map, const float* bias,
                          int perSample, float* tOut = nullptr,
                          const float* preH = nullptr, const float* preGate = nullptr, int preLdg = 0) {
  const std::string a = L.att;
  half* hh = scratch<half>("be2.cl.hh", (size_t)R * D);
  if (preH) adaLnGatedAdd<half>(x, preH, preGate, hh, R, D, cond, cond + D, 6 * D, map, preLdg);
  else adaLn<half>(x, hh, R, D, cond, cond + D, 6 * D, 0, 1.f, map);
  half* qkvg = scratch<half>("be2.cl.qkvh", (size_t)R * 4 * D);
  const half* wQkvg = stackedMirror2<half>(lw(L, a + "proj_qkv/weight"), (size_t)3 * D * D,
                                           lw(L, a + "proj_g/weight"), (size_t)D * D);
  linearPre<half, half>(hh, wQkvg, qkvg, R, D, 4 * D);
  if (L.windowed) {
    windowAttentionMMAK<<<dim3(R / WQ, L.heads), 64, 0, STREAM>>>(qkvg, lw(L, a + "proj_q_bias"), bias,
                                                                  Idev("atom_pad_mask"), hh, perSample, L.heads);
  } else {
    dim3 grid((unsigned)((perSample + 31) / 32), (unsigned)L.heads, (unsigned)(R / perSample));
    tokenAttention48MMAK<<<grid, 64, 0, STREAM>>>(qkvg, lw(L, a + "proj_q_bias"), bias,
                                                  Idev("token_pad_mask"), hh, perSample, L.heads);
  }
  float* h = scratch<float>("be2.cl.h", (size_t)R * D);
  linearPre<half>(hh, halfMirror<half>(lw(L, a + "proj_o/weight"), (size_t)D * D), h, R, D, D);
  adaLnGatedAdd<half>(x, h, cond + 2 * D, hh, R, D, cond + 3 * D, cond + 4 * D, 6 * D, map);
  float* t = tOut ? tOut : scratch<float>("be2.cl.t", (size_t)R * D);
  transitionHalf(nullptr, hh, t, R, D, L.p + "transition/", L.l, false, false, 0);
  if (!tOut) gatedAdd(x, t, cond + 5 * D, R, D, 6 * D, map);
}
inline void condLayer(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map, const float* bias,
                      int perSample, float* tOut = nullptr,
                      const float* preH = nullptr, const float* preGate = nullptr, int preLdg = 0) {
  if (GEMM_HALF) { condLayerHalf(L, x, R, D, cond, map, bias, perSample, tOut, preH, preGate, preLdg); return; }
  float* h = scratch<float>("be2.cl.h", (size_t)R * D);
  if (preH) adaLnGatedAdd<float>(x, preH, preGate, h, R, D, cond, cond + D, 6 * D, map, preLdg);
  else adaLn(x, h, R, D, cond, cond + D, 6 * D, 0, 1.f, map);
  float* qkvg = scratch<float>("be2.cl.qkv", (size_t)R * 4 * D);
  float* o = scratch<float>("be2.cl.o", (size_t)R * D);
  const std::string a = L.att;
  const float* wQkvg = stackedMirror2<float>(lw(L, a + "proj_qkv/weight"), (size_t)3 * D * D,
                                             lw(L, a + "proj_g/weight"), (size_t)D * D);
  linear(h, wQkvg, qkvg, R, D, 4 * D);
  if (L.windowed) {
    if (D / L.heads != 32) { fprintf(stderr, "BioEmu2: an atom head of %d (this port is written for 32)\n", D / L.heads); exit(1); }
    windowAttention4K<32, float><<<dim3(R / WQ, L.heads), WQ * 4, 0, STREAM>>>(qkvg, lw(L, a + "proj_q_bias"), bias,
                                                                               Idev("atom_pad_mask"), o, perSample, L.heads);
  } else {
    tokenAttention<float>(qkvg, lw(L, a + "proj_q_bias"), bias, o, perSample, D, L.heads, R / perSample);
  }
  linear(o, lw(L, a + "proj_o/weight"), h, R, D, D);
  adaLnGatedAdd<float>(x, h, cond + 2 * D, o, R, D, cond + 3 * D, cond + 4 * D, 6 * D, map);
  float* t = tOut ? tOut : scratch<float>("be2.cl.t", (size_t)R * D);
  transition(o, t, R, D, L.p + "transition/", L.l, false, false);
  if (!tOut) gatedAdd(x, t, cond + 5 * D, R, D, 6 * D, map);
}
// ...on its own conditioning, a row of c a row of x (the input embedder's)
inline void condLayer(const CondLayer& L, float* x, int R, int D, const float* c, int Cc, const float* bias) {
  float* cond = layerCond(L, c, R, Cc, D, "be2.cl.cond");
  condLayer(L, x, R, D, cond, nullptr, bias, R);
}

}  // namespace be2
