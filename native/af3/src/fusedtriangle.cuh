// The triangle multiplication's two row-wise halves as one kernel each, on the f16 path (C = 128).
//
//   triInK:  LN(pair) -> [projection | gate] (C x 4C) and the gating linear (C x C):
//            a, b = projection * sigmoid(gate) * mask, written CHANNEL-major for the contraction,
//            and t2 (the output gate's logits) row-major. The LayerNorm'd rows never leave the
//            chip, and neither does the 4C projection the unfused form wrote and read back.
//   triOutK: center-norm of the contraction (channel-major in), the output projection, and
//            pair += t1 * sigmoid(t2) - the centred rows and t1 stay on the chip.
//
// The same skeleton as fusedtransition.cuh: a warp's 16 rows as MMA A fragments in registers, the
// weights streamed through shared memory, outputs straight from the accumulators.
#pragma once
#include "flash.cuh"

constexpr int TI_NC = 32;
__host__ __device__ constexpr size_t tiStage(int C) { return (size_t)2 * C * (TI_NC + 8) * 2; }

template <int C, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) triInK(const float* __restrict__ pair, const float* __restrict__ mask,
    const float* __restrict__ lnScale, const float* __restrict__ lnOffset, const half* __restrict__ Wpg,
    const half* __restrict__ Wg, half* __restrict__ a, half* __restrict__ b, half* __restrict__ t2, size_t pairs,
    size_t cs) {
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, LDW = TI_NC + 8, KS = C / 16, LDT = R + 8;
  constexpr size_t STAGE = tiStage(C);
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem;                                           // [R][LDX]
  half* Ta = Xs + R * LDX;                                          // [16 channels][LDT], a then b
  half* Tb = Ta + 16 * LDT;
  unsigned char* stages = (unsigned char*)(Tb + 16 * LDT);
  auto W0 = [&](int s) { return (half*)(stages + s * STAGE); };
  auto W1 = [&](int s) { return W0(s) + C * LDW; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  // steps 0 .. C/16-1: 16 channels of a and b (projection columns 2ch.., gate columns 2C+2ch..);
  // then C/32 steps of the gating linear, 32 columns each
  const int steps = C / 16 + C / TI_NC;
  auto issue = [&](int j, int st) {
    half *w0 = W0(st), *w1 = W1(st);
    for (int t = threadIdx.x; t < C * (TI_NC / 8); t += NTH) {
      int k = t / (TI_NC / 8), c = (t % (TI_NC / 8)) * 8;
      if (j < C / 16) {
        cpAsync16(w0 + k * LDW + c, Wpg + (size_t)k * 4 * C + j * 32 + c, true);
        cpAsync16(w1 + k * LDW + c, Wpg + (size_t)k * 4 * C + 2 * C + j * 32 + c, true);
      } else {
        cpAsync16(w0 + k * LDW + c, Wg + (size_t)k * C + (j - C / 16) * TI_NC + c, true);
      }
    }
    asm volatile("cp.async.commit_group;");
  };
  issue(0, 0);
  for (int r = warp; r < R; r += WARPS) {
    size_t row = row0 + r;
    float v[C / 32]; float s = 0.f;
    for (int k = 0; k < C / 32; ++k) { v[k] = row < pairs ? pair[row * C + lane + 32 * k] : 0.f; s += v[k]; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, q = 0.f;
    for (int k = 0; k < C / 32; ++k) { float d = v[k] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    float inv = rsqrtf(q / C + 1e-5f);
    for (int k = 0; k < C / 32; ++k) {
      int c = lane + 32 * k;
      Xs[r * LDX + c] = __float2half((v[k] - mean) * inv * lnScale[c] + lnOffset[c]);
    }
  }
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  int lr0 = warp * 16 + g, lr1 = lr0 + 8;                          // the thread's rows within the block
  float m0 = row0 + lr0 < pairs ? mask[row0 + lr0] : 0.f, m1 = row0 + lr1 < pairs ? mask[row0 + lr1] : 0.f;
  for (int j = 0; j < steps; ++j) {
    int st = j & 1;
    if (j + 1 < steps) { issue(j + 1, st ^ 1); asm volatile("cp.async.wait_group 1;"); }
    else asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    bool gating = j >= C / 16;
    const half *w0 = W0(st), *w1 = W1(st);
    float p[TI_NC / 8][4] = {}, q[TI_NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < TI_NC / 16; ++n2) {
        int k = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), c = n2 * 16 + (lane >> 4) * 8;
        uint32_t f0[4];
        ldsm4t(f0, w0 + k * LDW + c);
        mma16816(p[2 * n2], xa[ks], f0[0], f0[1]); mma16816(p[2 * n2 + 1], xa[ks], f0[2], f0[3]);
        if (!gating) {
          uint32_t f1[4];
          ldsm4t(f1, w1 + k * LDW + c);
          mma16816(q[2 * n2], xa[ks], f1[0], f1[1]); mma16816(q[2 * n2 + 1], xa[ks], f1[2], f1[3]);
        }
      }
    }
    if (gating) {                       // t2, row-major: columns (j - C/16)*32 + nt*8 + 2 tig
      int c0 = (j - C / 16) * TI_NC;
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        int c = c0 + nt * 8 + tig * 2;
        if (row0 + lr0 < pairs) *reinterpret_cast<half2*>(t2 + (row0 + lr0) * C + c) = __floats2half2_rn(p[nt][0], p[nt][1]);
        if (row0 + lr1 < pairs) *reinterpret_cast<half2*>(t2 + (row0 + lr1) * C + c) = __floats2half2_rn(p[nt][2], p[nt][3]);
      }
    } else {
      // column 2ch is a's channel ch, 2ch+1 b's (the interleaved split): this thread's column pair
      // nt*8 + 2 tig is channel nt*4 + tig of the step's 16; staged channel-major
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        int ch = nt * 4 + tig;
        Ta[ch * LDT + lr0] = __float2half(p[nt][0] * sigm(q[nt][0]) * m0);
        Tb[ch * LDT + lr0] = __float2half(p[nt][1] * sigm(q[nt][1]) * m0);
        Ta[ch * LDT + lr1] = __float2half(p[nt][2] * sigm(q[nt][2]) * m1);
        Tb[ch * LDT + lr1] = __float2half(p[nt][3] * sigm(q[nt][3]) * m1);
      }
      __syncthreads();
      // 16 bytes (8 pairs) a thread: the channel stride cs is a multiple of 8, the block's first
      // row a multiple of R; rows past `pairs` are zero (masked) and land in the padding
      for (int t = threadIdx.x; t < 16 * (R / 8); t += NTH) {
        int ch = t / (R / 8), r = (t % (R / 8)) * 8;
        size_t row = row0 + r;
        if (row < cs) {
          size_t at = (size_t)(j * 16 + ch) * cs + row;
          *reinterpret_cast<uint4*>(a + at) = *reinterpret_cast<const uint4*>(Ta + ch * LDT + r);
          *reinterpret_cast<uint4*>(b + at) = *reinterpret_cast<const uint4*>(Tb + ch * LDT + r);
        }
      }
    }
    __syncthreads();
  }
}

// pair += (LN_center(prod) Wout) * sigmoid(t2); prod channel-major [C][pairs] f32
template <int C, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) triOutK(const float* __restrict__ prod, const float* __restrict__ cnScale,
    const float* __restrict__ cnOffset, const half* __restrict__ Wout, const half* __restrict__ t2,
    float* __restrict__ pair, size_t pairs, size_t cs) {
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDP = R + 4, LDX = C + 8, LDW = C + 8, KS = C / 16, NT = C / 8;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Ws = (half*)smem;                                           // [C][LDW], the whole output projection
  half* Xs = Ws + C * LDW;                                          // [R][LDX]
  float* Ps = (float*)(Xs + R * LDX);                               // [C][LDP]
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  for (int t = threadIdx.x; t < C * (C / 8); t += NTH) {
    int k = t / (C / 8), c = (t % (C / 8)) * 8;
    cpAsync16(Ws + k * LDW + c, Wout + (size_t)k * C + c, true);
  }
  asm volatile("cp.async.commit_group;");
  for (int t = threadIdx.x; t < C * (R / 4); t += NTH) {            // 16 bytes (4 pairs) a thread
    int c = t / (R / 4), r = (t % (R / 4)) * 4;
    size_t row = row0 + r;
    cpAsync16(Ps + c * LDP + r, prod + (size_t)c * cs + (row < cs ? row : 0), row < cs);
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  for (int r = warp; r < R; r += WARPS) {                            // the center norm, a warp a row
    float v[C / 32]; float s = 0.f;
    for (int k = 0; k < C / 32; ++k) { v[k] = Ps[(lane + 32 * k) * LDP + r]; s += v[k]; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, q = 0.f;
    for (int k = 0; k < C / 32; ++k) { float d = v[k] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    float inv = rsqrtf(q / C + 1e-5f);
    for (int k = 0; k < C / 32; ++k) {
      int c = lane + 32 * k;
      Xs[r * LDX + c] = __float2half((v[k] - mean) * inv * cnScale[c] + cnOffset[c]);
    }
  }
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  float acc[NT][4] = {};
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
    for (int et = 0; et < NT; et += 2) {
      uint32_t f[4];
      ldsm4t(f, Ws + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + (et + (lane >> 4)) * 8);
      mma16816(acc[et], xa[ks], f[0], f[1]);
      mma16816(acc[et + 1], xa[ks], f[2], f[3]);
    }
  }
  size_t r0 = row0 + warp * 16 + g, r1 = r0 + 8;
#pragma unroll
  for (int et = 0; et < NT; ++et) {
    int c = et * 8 + tig * 2;
    if (r0 < pairs) {
      float2 gt = __half22float2(*reinterpret_cast<const half2*>(t2 + r0 * C + c));
      float2* p = (float2*)(pair + r0 * C + c); float2 v = *p;
      v.x += acc[et][0] * sigm(gt.x); v.y += acc[et][1] * sigm(gt.y); *p = v;
    }
    if (r1 < pairs) {
      float2 gt = __half22float2(*reinterpret_cast<const half2*>(t2 + r1 * C + c));
      float2* p = (float2*)(pair + r1 * C + c); float2 v = *p;
      v.x += acc[et][2] * sigm(gt.x); v.y += acc[et][3] * sigm(gt.y); *p = v;
    }
  }
}

inline bool FUSED_TRIANGLE = true;
constexpr int TI_WARPS = 16, TO_WARPS = 8;
inline void triIn128(const float* pair, const float* mask, const std::string& pre, const std::string& pg,
                     half* a, half* b, half* t2, size_t pairs, size_t cs) {
  constexpr int C = 128, R = 16 * TI_WARPS;
  size_t smem = (size_t)R * (C + 8) * 2 + (size_t)2 * 16 * (R + 8) * 2 + 2 * tiStage(C);
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(triInK<C, TI_WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  triInK<C, TI_WARPS><<<(unsigned)((pairs + R - 1) / R), 32 * TI_WARPS, smem, STREAM>>>(
    pair, mask, W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset"), Wh(pg), Wh(pre + ".gatingLinear"),
    a, b, t2, pairs, cs);
}
inline void triOut128(const float* prod, const std::string& pre, const half* t2, float* pair, size_t pairs, size_t cs) {
  constexpr int C = 128, R = 16 * TO_WARPS;
  size_t smem = (size_t)C * (C + 8) * 2 + (size_t)R * (C + 8) * 2 + (size_t)C * (R + 4) * 4;
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(triOutK<C, TO_WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  triOutK<C, TO_WARPS><<<(unsigned)((pairs + R - 1) / R), 32 * TO_WARPS, smem, STREAM>>>(
    prod, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"), Wh(pre + ".outputProjection"), t2, pair, pairs, cs);
}

// out[h][row] = (LN(x[row]) W)[h] for a projection to few heads (N a multiple of 8, W (C, N)):
// the single track's pair logits, the pair read once and written head-major (the layout the
// softmax reads) through shared memory.
template <int C, int N, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) lnHeadsK(const float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ Wp, float* __restrict__ out, size_t rows) {
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, LDW = N + 8, KS = C / 16, LDO = R + 4;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem; half* Ws = Xs + R * LDX; float* Os = (float*)(Ws + C * LDW);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  for (int t = threadIdx.x; t < C * (N / 8); t += NTH) {
    int k = t / (N / 8), c = (t % (N / 8)) * 8;
    cpAsync16(Ws + k * LDW + c, Wp + (size_t)k * N + c, true);
  }
  asm volatile("cp.async.commit_group;");
  for (int r = warp; r < R; r += WARPS) {
    size_t row = row0 + r;
    float v[C / 32]; float s = 0.f;
    for (int k = 0; k < C / 32; ++k) { v[k] = row < rows ? x[row * C + lane + 32 * k] : 0.f; s += v[k]; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, q = 0.f;
    for (int k = 0; k < C / 32; ++k) { float d = v[k] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
    float inv = rsqrtf(q / C + 1e-5f);
    for (int k = 0; k < C / 32; ++k) {
      int c = lane + 32 * k;
      Xs[r * LDX + c] = __float2half((v[k] - mean) * inv * lnScale[c] + lnOffset[c]);
    }
  }
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  float acc[N / 8][4] = {};
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    uint32_t xa[4];
    ldsm4(xa, Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
#pragma unroll
    for (int n2 = 0; n2 < N / 16; ++n2) {
      uint32_t f[4];
      ldsm4t(f, Ws + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + n2 * 16 + (lane >> 4) * 8);
      mma16816(acc[2 * n2], xa, f[0], f[1]); mma16816(acc[2 * n2 + 1], xa, f[2], f[3]);
    }
  }
  int lr0 = warp * 16 + g, lr1 = lr0 + 8;
#pragma unroll
  for (int nt = 0; nt < N / 8; ++nt) {
    int h = nt * 8 + tig * 2;
    Os[h * LDO + lr0] = acc[nt][0]; Os[(h + 1) * LDO + lr0] = acc[nt][1];
    Os[h * LDO + lr1] = acc[nt][2]; Os[(h + 1) * LDO + lr1] = acc[nt][3];
  }
  __syncthreads();
  for (int t = threadIdx.x; t < N * R; t += NTH) {
    int h = t / R, r = t % R;
    if (row0 + r < rows) out[(size_t)h * rows + row0 + r] = Os[h * LDO + r];
  }
}
template <int N>
inline void lnHeads128(const float* x, const std::string& scale, const std::string& offset, const std::string& w,
                       float* out, size_t rows) {
  constexpr int C = 128, WARPS = 8, R = 16 * WARPS;
  size_t smem = (size_t)R * (C + 8) * 2 + (size_t)C * (N + 8) * 2 + (size_t)N * (R + 4) * 4;
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(lnHeadsK<C, N, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  lnHeadsK<C, N, WARPS><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(x, W(scale), W(offset), Wh(w), out, rows);
}
