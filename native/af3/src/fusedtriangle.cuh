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
#include <cuda_bf16.h>

// LayerNorm of a block's R rows into shared memory as f16, a warp a row and FOUR rows' loads in
// flight per warp (one at a time left the loads latency-bound). rowOf(r) is row r's index in x,
// or SIZE_MAX for a row past the end (normalised zeros).
template <int C, int R, int WARPS, class RowOf>
__device__ __forceinline__ void lnRowsToShared(const float* __restrict__ x, RowOf rowOf, const float* __restrict__ scale,
                                               const float* __restrict__ offset, half* Xs, int LDX, int warp, int lane) {
  constexpr int RPW = R / WARPS, B = RPW % 4 == 0 ? 4 : (RPW % 2 == 0 ? 2 : 1), K = C / 32;
  for (int base = 0; base < RPW; base += B) {
    float v[B][K];
#pragma unroll
    for (int b = 0; b < B; ++b) {
      size_t row = rowOf(warp + (base + b) * WARPS);
#pragma unroll
      for (int k = 0; k < K; ++k) v[b][k] = row != SIZE_MAX ? x[row * C + lane + 32 * k] : 0.f;
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < K; ++k) s += v[b][k];
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
      float mean = s / C, q = 0.f;
#pragma unroll
      for (int k = 0; k < K; ++k) { float d = v[b][k] - mean; q += d * d; }
      for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
      float inv = rsqrtf(q / C + 1e-5f);
      int r = warp + (base + b) * WARPS;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        int c = lane + 32 * k;
        Xs[r * LDX + c] = __float2half((v[b][k] - mean) * inv * scale[c] + offset[c]);
      }
    }
  }
}

constexpr int TI_NC = 32;
__host__ __device__ constexpr size_t tiStage(int C) { return (size_t)2 * C * (TI_NC + 8) * 2; }

// TA: a and b's type - f16, or bf16 so the contraction can write a bf16 product (f16 overflows).
// BIAS: AlphaFold 2's projections carry biases, AF3's do not - `bias` is [4C, the projection | gate columns
// in Wpg's order][C, the gating linear's]; without BIAS the kernel is the one AF3 has always run
template <int C, int WARPS, class TA, bool BIAS = false>
__global__ void __launch_bounds__(WARPS * 32) triInK(const float* __restrict__ pair, const float* __restrict__ mask,
    const float* __restrict__ lnScale, const float* __restrict__ lnOffset, const half* __restrict__ Wpg,
    const half* __restrict__ Wg, TA* __restrict__ a, TA* __restrict__ b, half* __restrict__ t2, int n, int np,
    size_t cs, const float* __restrict__ bias = nullptr) {
  // rows are the PADDED pair space (np x np, np a multiple of 8, which is what the contraction's
  // GEMM wants); a padding row maps to no pair and writes zeros
  const size_t pp = (size_t)np * np;
  auto pairOf = [&](size_t q) -> size_t {
    if (q >= pp) return SIZE_MAX;
    unsigned u = (unsigned)q, i = u / (unsigned)np, j = u - i * (unsigned)np;    // 32-bit: a 64-bit divide is ~70 instructions
    return i < (unsigned)n && j < (unsigned)n ? (size_t)i * n + j : SIZE_MAX;
  };
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, LDW = TI_NC + 8, KS = C / 16, LDT = R + 8;
  constexpr size_t STAGE = tiStage(C);
  extern __shared__ __align__(16) unsigned char smem[];
  // Xs is dead once its rows are A fragments, so a and b's staging takes its place (the first
  // write follows the loop's first barrier, after every warp's ldmatrix): two blocks an SM at 8 warps
  static_assert(2 * 16 * (R + 8) * sizeof(TA) <= R * LDX * 2, "staging fits in Xs");
  half* Xs = (half*)smem;                                           // [R][LDX]
  TA* Ta = (TA*)Xs;                                                 // [16 channels][LDT], a then b
  TA* Tb = Ta + 16 * LDT;
  unsigned char* stages = (unsigned char*)(Xs + R * LDX);
  auto W0 = [&](int s) { return (half*)(stages + s * STAGE); };
  auto W1 = [&](int s) { return W0(s) + C * LDW; };
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  // steps 0 .. C/16-1: 16 channels of a and b (projection columns 2ch.., gate columns 2C+2ch..);
  // then C/32 steps of the gating linear, 32 columns each
  const int steps = C / 16 + C / TI_NC;
  auto issue = [&](int j, int st) {
    half *w0 = W0(st), *w1 = W1(st);
    #pragma unroll
    for (int t0_ = 0; t0_ < C * (TI_NC / 8); t0_ += NTH) {  // unrolled (a strided loop from threadIdx.x is not)
      const int t = t0_ + (int)threadIdx.x;
      if ((C * (TI_NC / 8)) % NTH != 0 && t >= (C * (TI_NC / 8))) break;
      int k = t / (TI_NC / 8), c = (t % (TI_NC / 8)) * 8;
      if (j < C / 16) {
        cpAsync16(w0 + k * LDW + c, Wpg + (size_t)k * 4 * C + j * 32 + c, true);
        cpAsync16(w1 + k * LDW + c, Wpg + (size_t)k * 4 * C + 2 * C + j * 32 + c, true);
      } else {
        cpAsync16(w0 + k * LDW + c, Wg + (size_t)k * C + (j - C / 16) * TI_NC + c, true);
      }
    }
    cpCommit();
  };
  issue(0, 0);
  lnRowsToShared<C, R, WARPS>(pair, [&](int r) { return pairOf(row0 + r); }, lnScale, lnOffset, Xs, LDX, warp, lane);
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  int lr0 = warp * 16 + g, lr1 = lr0 + 8;                          // the thread's rows within the block
  size_t pr0 = pairOf(row0 + lr0), pr1 = pairOf(row0 + lr1);
  float m0 = pr0 != SIZE_MAX ? mask[pr0] : 0.f, m1 = pr1 != SIZE_MAX ? mask[pr1] : 0.f;
  for (int j = 0; j < steps; ++j) {
    int st = j & 1;
    if (j + 1 < steps) { issue(j + 1, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
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
    if constexpr (BIAS) {
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        int col = (gating ? 4 * C + (j - C / 16) * TI_NC : j * TI_NC) + nt * 8 + tig * 2;
        float b0 = bias[col], b1 = bias[col + 1];
        p[nt][0] += b0; p[nt][1] += b1; p[nt][2] += b0; p[nt][3] += b1;
        if (!gating) {
          float g0 = bias[2 * C + col], g1 = bias[2 * C + col + 1];
          q[nt][0] += g0; q[nt][1] += g1; q[nt][2] += g0; q[nt][3] += g1;
        }
      }
    }
    if (gating) {                       // t2, row-major: columns (j - C/16)*32 + nt*8 + 2 tig
      int c0 = (j - C / 16) * TI_NC;
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        int c = c0 + nt * 8 + tig * 2;
        if (row0 + lr0 < pp) *reinterpret_cast<half2*>(t2 + (row0 + lr0) * C + c) = __floats2half2_rn(p[nt][0], p[nt][1]);
        if (row0 + lr1 < pp) *reinterpret_cast<half2*>(t2 + (row0 + lr1) * C + c) = __floats2half2_rn(p[nt][2], p[nt][3]);
      }
    } else {
      // column 2ch is a's channel ch, 2ch+1 b's (the interleaved split): this thread's column pair
      // nt*8 + 2 tig is channel nt*4 + tig of the step's 16; staged channel-major
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        int ch = nt * 4 + tig;
        Ta[ch * LDT + lr0] = TA(p[nt][0] * sigm(q[nt][0]) * m0);
        Tb[ch * LDT + lr0] = TA(p[nt][1] * sigm(q[nt][1]) * m0);
        Ta[ch * LDT + lr1] = TA(p[nt][2] * sigm(q[nt][2]) * m1);
        Tb[ch * LDT + lr1] = TA(p[nt][3] * sigm(q[nt][3]) * m1);
      }
      __syncthreads();
      // 16 bytes (8 rows) a thread: the channel stride cs is a multiple of 8, the block's first
      // row a multiple of R; padding rows are zero (masked)
      #pragma unroll
      for (int t0_ = 0; t0_ < 16 * (R / 8); t0_ += NTH) {  // unrolled (a strided loop from threadIdx.x is not)
        const int t = t0_ + (int)threadIdx.x;
        if ((16 * (R / 8)) % NTH != 0 && t >= (16 * (R / 8))) break;
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
template <int C, int WARPS, class TP, bool BIAS = false>
__global__ void __launch_bounds__(WARPS * 32) triOutK(const TP* __restrict__ prod, const float* __restrict__ cnScale,
    const float* __restrict__ cnOffset, const half* __restrict__ Wout, const half* __restrict__ t2,
    float* __restrict__ pair, int n, int np, size_t cs, const float* __restrict__ bias = nullptr) {
  const size_t pp = (size_t)np * np;
  constexpr int PV = 16 / sizeof(TP);                               // product elements in 16 bytes
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDP = R + PV, LDX = C + 8, LDW = C + 8, KS = C / 16, NT = C / 8;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Ws = (half*)smem;                                           // [C][LDW], the whole output projection
  half* Xs = Ws + C * LDW;                                          // [R][LDX]
  TP* Ps = (TP*)(Xs + R * LDX);                                     // [C][LDP]
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  for (int t = threadIdx.x; t < C * (C / 8); t += NTH) {
    int k = t / (C / 8), c = (t % (C / 8)) * 8;
    cpAsync16(Ws + k * LDW + c, Wout + (size_t)k * C + c, true);
  }
  cpCommit();
  for (int t = threadIdx.x; t < C * (R / PV); t += NTH) {           // 16 bytes a thread
    int c = t / (R / PV), r = (t % (R / PV)) * PV;
    size_t row = row0 + r;
    cpAsync16(Ps + c * LDP + r, prod + (size_t)c * cs + (row < cs ? row : 0), row < cs);
  }
  cpCommit();
  cpWait<0>();
  __syncthreads();
  for (int r = warp; r < R; r += WARPS) {                            // the center norm, a warp a row
    float v[C / 32]; float s = 0.f;
    for (int k = 0; k < C / 32; ++k) { v[k] = (float)Ps[(lane + 32 * k) * LDP + r]; s += v[k]; }
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
  // padded rows (t2 is in the padded space too) back to pairs
  size_t r0 = row0 + warp * 16 + g, r1 = r0 + 8;
  auto pairOf = [&](size_t q) -> size_t {
    if (q >= pp) return SIZE_MAX;
    unsigned u = (unsigned)q, i = u / (unsigned)np, j = u - i * (unsigned)np;    // 32-bit: a 64-bit divide is ~70 instructions
    return i < (unsigned)n && j < (unsigned)n ? (size_t)i * n + j : SIZE_MAX;
  };
  size_t p0 = pairOf(r0), p1 = pairOf(r1);
#pragma unroll
  for (int et = 0; et < NT; ++et) {
    int c = et * 8 + tig * 2;
    float ob0 = 0.f, ob1 = 0.f;      // (the output projection's bias, where the model has one)
    if constexpr (BIAS) { ob0 = bias[c]; ob1 = bias[c + 1]; }
    if (p0 != SIZE_MAX) {
      float2 gt = __half22float2(*reinterpret_cast<const half2*>(t2 + r0 * C + c));
      float2* p = (float2*)(pair + p0 * C + c); float2 v = *p;
      v.x += (acc[et][0] + ob0) * sigm(gt.x); v.y += (acc[et][1] + ob1) * sigm(gt.y); *p = v;
    }
    if (p1 != SIZE_MAX) {
      float2 gt = __half22float2(*reinterpret_cast<const half2*>(t2 + r1 * C + c));
      float2* p = (float2*)(pair + p1 * C + c); float2 v = *p;
      v.x += (acc[et][2] + ob0) * sigm(gt.x); v.y += (acc[et][3] + ob1) * sigm(gt.y); *p = v;
    }
  }
}

// triOutK as a PERSISTENT kernel: each block keeps the output projection in shared memory for
// every tile it takes and prefetches its next product tile while it computes the current one
// (the one-shot kernel loaded 32 KB of weights and its tile, then computed, at two blocks an SM)
template <int C, int WARPS, class TP, bool BIAS = false>
__global__ void __launch_bounds__(WARPS * 32) triOutPK(const TP* __restrict__ prod, const float* __restrict__ cnScale,
    const float* __restrict__ cnOffset, const half* __restrict__ Wout, const half* __restrict__ t2,
    float* __restrict__ pair, int n, int np, size_t cs, const float* __restrict__ bias = nullptr) {
  const size_t pp = (size_t)np * np;
  constexpr int PV = 16 / sizeof(TP);
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDP = R + PV, LDX = C + 8, LDW = C + 8, KS = C / 16, NT = C / 8;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Ws = (half*)smem;                                           // [C][LDW]
  half* Xs = Ws + C * LDW;                                          // [R][LDX]
  TP* Pst = (TP*)(Xs + R * LDX);                                    // two stages of [C][LDP]
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const size_t tiles = (pp + R - 1) / R;
#pragma unroll
  for (int t0_ = 0; t0_ < C * (C / 8); t0_ += NTH) {
    int t = t0_ + (int)threadIdx.x, k = t / (C / 8), c = (t % (C / 8)) * 8;
    cpAsync16(Ws + k * LDW + c, Wout + (size_t)k * C + c, true);
  }
  auto issue = [&](size_t tile, int st) {
    TP* Ps = Pst + st * C * LDP;
    size_t row0 = tile * R;
#pragma unroll
    for (int t0_ = 0; t0_ < C * (R / PV); t0_ += NTH) {
      int t = t0_ + (int)threadIdx.x, c = t / (R / PV), r = (t % (R / PV)) * PV;
      size_t row = row0 + r;
      cpAsync16(Ps + c * LDP + r, prod + (size_t)c * cs + (row < cs ? row : 0), row < cs);
    }
    cpCommit();
  };
  size_t tile = blockIdx.x;
  if (tile < tiles) issue(tile, 0); else cpCommit();
  for (int it = 0; tile < tiles; ++it, tile += gridDim.x) {
    int st = it & 1;
    size_t next = tile + gridDim.x;
    if (next < tiles) { issue(next, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
    // the epilogue's operands, loaded now so the norm and the GEMM cover their latency (one block
    // an SM: nothing else would)
    size_t r0 = tile * R + warp * 16 + g, r1 = r0 + 8;
    auto pairOf = [&](size_t q) -> size_t {
      if (q >= pp) return SIZE_MAX;
      unsigned u = (unsigned)q, i = u / (unsigned)np, j = u - i * (unsigned)np;
      return i < (unsigned)n && j < (unsigned)n ? (size_t)i * n + j : SIZE_MAX;
    };
    size_t p0 = pairOf(r0), p1 = pairOf(r1);
    float2 pv[NT][2]; half2 gv[NT][2];
#pragma unroll
    for (int et = 0; et < NT; ++et) {
      int c = et * 8 + tig * 2;
      if (p0 != SIZE_MAX) { pv[et][0] = *(const float2*)(pair + p0 * C + c); gv[et][0] = *reinterpret_cast<const half2*>(t2 + r0 * C + c); }
      if (p1 != SIZE_MAX) { pv[et][1] = *(const float2*)(pair + p1 * C + c); gv[et][1] = *reinterpret_cast<const half2*>(t2 + r1 * C + c); }
    }
    __syncthreads();
    const TP* Ps = Pst + st * C * LDP;
    // the center norm, a THREAD a row: the tile is channel-major, so a warp reading one row across
    // its channels hit four-way bank conflicts; a thread per row reads 32 consecutive rows a warp,
    // and writes its row in 16-byte pieces (rows 272 bytes apart cover all 32 banks)
    // (two adjacent lanes a row, each over half the channels: R rows are half the block's threads)
    static_assert(2 * R == NTH, "a row is two threads");
    {
      int r = threadIdx.x >> 1, cb = (threadIdx.x & 1) * (C / 2);
      float s = 0.f;
#pragma unroll 8
      for (int c = cb; c < cb + C / 2; ++c) s += (float)Ps[c * LDP + r];
      s += __shfl_xor_sync(~0u, s, 1);
      float mean = s / C, q = 0.f;
#pragma unroll 8
      for (int c = cb; c < cb + C / 2; ++c) { float d = (float)Ps[c * LDP + r] - mean; q += d * d; }
      q += __shfl_xor_sync(~0u, q, 1);
      float inv = rsqrtf(q / C + 1e-5f);
#pragma unroll
      for (int c0 = cb; c0 < cb + C / 2; c0 += 8) {
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int c = c0 + 2 * e;
          w[e] = pack2(((float)Ps[c * LDP + r] - mean) * inv * cnScale[c] + cnOffset[c],
                       ((float)Ps[(c + 1) * LDP + r] - mean) * inv * cnScale[c + 1] + cnOffset[c + 1]);
        }
        *reinterpret_cast<uint4*>(Xs + r * LDX + c0) = make_uint4(w[0], w[1], w[2], w[3]);
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
#pragma unroll
    for (int et = 0; et < NT; ++et) {
      int c = et * 8 + tig * 2;
      float ob0 = 0.f, ob1 = 0.f;
      if constexpr (BIAS) { ob0 = bias[c]; ob1 = bias[c + 1]; }
      if (p0 != SIZE_MAX) {
        float2 gt = __half22float2(gv[et][0]), v = pv[et][0];
        v.x += (acc[et][0] + ob0) * sigm(gt.x); v.y += (acc[et][1] + ob1) * sigm(gt.y); *(float2*)(pair + p0 * C + c) = v;
      }
      if (p1 != SIZE_MAX) {
        float2 gt = __half22float2(gv[et][1]), v = pv[et][1];
        v.x += (acc[et][2] + ob0) * sigm(gt.x); v.y += (acc[et][3] + ob1) * sigm(gt.y); *(float2*)(pair + p1 * C + c) = v;
      }
    }
    __syncthreads();                                                 // Xs and this stage are reused
  }
}

inline bool FUSED_TRIANGLE = true;
constexpr int TI_WARPS = 8, TO_WARPS = 8;    // triInK: 8 warps fit two blocks an SM (1044 tokens: 828 -> 748 ms against 16)
// rows a block, by size: a small input in the large tiles left most of the device idle (68 tokens:
// 36 blocks of 256 rows on 108 SMs); the warps are the largest that still give two blocks an SM
inline size_t MIN_BLOCKS = 54;      // swept: 68 tokens 89.3 -> 75.8 ms of trunk at 54, 150 tokens flat (108 and 216 slower there)
inline int warpsFor(size_t rows, std::initializer_list<int> options) {
  int last = 0;
  for (int w : options) { last = w; if ((rows + 16 * w - 1) / (16 * w) >= MIN_BLOCKS) return w; }
  return last;
}
// ...and no more than the device's shared memory allows (a T4's 64 KB): the first option, at or after
// `w`, whose blocks fit; 0 if none does
template <class F> int warpsFitting(int w, std::initializer_list<int> options, F smemFor) {
  bool from = false;
  for (int o : options) { if (o == w) from = true; if (from && fitsSmem(smemFor(o))) return o; }
  return 0;
}
// the launchers on raw pointers (AlphaFold 2 calls these with its own weights and biases), then AF3's by name
template <class TA, int WARPS, bool BIAS>
void triInLaunch(const float* pair, const float* mask, const float* lnScale, const float* lnOffset, const half* Wpg,
                 const half* Wg, const float* bias, TA* a, TA* b, half* t2, int n, int np, size_t cs) {
  constexpr int C = 128, R = 16 * WARPS;
  size_t pp = (size_t)np * np;
  size_t smem = (size_t)R * (C + 8) * 2 + 2 * tiStage(C);
  static bool attr = false;
  if (!attr) { smemAttr((triInK<C, WARPS, TA, BIAS>), (int)smem); attr = true; }
  triInK<C, WARPS, TA, BIAS><<<(unsigned)((pp + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    pair, mask, lnScale, lnOffset, Wpg, Wg, a, b, t2, n, np, cs, bias);
}
inline size_t triInSmem(int warps) { return (size_t)16 * warps * (128 + 8) * 2 + 2 * tiStage(128); }
template <class TA, bool BIAS = false>
void triInRaw(const float* pair, const float* mask, const float* lnScale, const float* lnOffset, const half* Wpg,
              const half* Wg, const float* bias, TA* a, TA* b, half* t2, int n, int np, size_t cs) {
  switch (warpsFitting(warpsFor((size_t)np * np, {TI_WARPS, 4}), {TI_WARPS, 4}, triInSmem)) {
    case 4: triInLaunch<TA, 4, BIAS>(pair, mask, lnScale, lnOffset, Wpg, Wg, bias, a, b, t2, n, np, cs); break;
    default: triInLaunch<TA, TI_WARPS, BIAS>(pair, mask, lnScale, lnOffset, Wpg, Wg, bias, a, b, t2, n, np, cs);
  }
}
template <class TA>
void triIn128(const float* pair, const float* mask, const std::string& pre, const std::string& pg,
              TA* a, TA* b, half* t2, int n, int np, size_t cs) {
  triInRaw<TA>(pair, mask, W(pre + ".leftNormInputScale"), W(pre + ".leftNormInputOffset"), Wh(pg), Wh(pre + ".gatingLinear"),
               nullptr, a, b, t2, n, np, cs);
}
inline bool TRI_OUT_PERSISTENT = true;
template <class TP> constexpr size_t triOutPSmem(int warps = TO_WARPS) {
  return (size_t)128 * 136 * 2 + (size_t)16 * warps * 136 * 2 + (size_t)2 * 128 * (16 * warps + 16 / sizeof(TP)) * sizeof(TP);
}
template <class TP> constexpr size_t triOutSmem(int warps = TO_WARPS) {
  return (size_t)128 * 136 * 2 + (size_t)16 * warps * 136 * 2 + (size_t)128 * (16 * warps + 16 / sizeof(TP)) * sizeof(TP);
}
// the three fused triangle kernels fit this device - at 4 warps if not 8: an L4's 99 KB a block takes the
// output kernel at 4 (71 KB; 104 KB at 8); a T4's 64 KB takes none, and the unfused path runs there
template <class TP> bool triFusedFits() {
  return fitsSmem(triInSmem(4)) && fitsSmem(std::min(triOutPSmem<TP>(4), triOutSmem<TP>(4)));
}
template <class TP, int WARPS, bool BIAS>
void triOutLaunch(const TP* prod, const float* cnScale, const float* cnOffset, const half* Wout, const float* bias,
                  const half* t2, float* pair, int n, int np, size_t cs) {
  constexpr int C = 128, R = 16 * WARPS;
  size_t pp = (size_t)np * np;
  if (TRI_OUT_PERSISTENT && fitsSmem(triOutPSmem<TP>(WARPS))) {
    size_t smem = (size_t)C * (C + 8) * 2 + (size_t)R * (C + 8) * 2 + (size_t)2 * C * (R + 16 / sizeof(TP)) * sizeof(TP);
    static int grid = 0;
    if (!grid) {
      smemAttr((triOutPK<C, WARPS, TP, BIAS>), (int)smem);
      int perSm = 0, sms = 0;
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, triOutPK<C, WARPS, TP, BIAS>, 32 * WARPS, smem));
      CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
      grid = std::max(1, perSm) * sms;
    }
    size_t tiles = (pp + R - 1) / R;
    triOutPK<C, WARPS, TP, BIAS><<<(unsigned)std::min<size_t>(grid, tiles), 32 * WARPS, smem, STREAM>>>(
      prod, cnScale, cnOffset, Wout, t2, pair, n, np, cs, bias);
    return;
  }
  size_t smem = (size_t)C * (C + 8) * 2 + (size_t)R * (C + 8) * 2 + (size_t)C * (R + 16 / sizeof(TP)) * sizeof(TP);
  static bool attr = false;
  if (!attr) { smemAttr((triOutK<C, WARPS, TP, BIAS>), (int)smem); attr = true; }
  triOutK<C, WARPS, TP, BIAS><<<(unsigned)((pp + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    prod, cnScale, cnOffset, Wout, t2, pair, n, np, cs, bias);
}
template <class TP, bool BIAS = false>
void triOutRaw(const TP* prod, const float* cnScale, const float* cnOffset, const half* Wout, const float* bias,
               const half* t2, float* pair, int n, int np, size_t cs) {
  if (fitsSmem(std::min(triOutPSmem<TP>(TO_WARPS), triOutSmem<TP>(TO_WARPS))))
    triOutLaunch<TP, TO_WARPS, BIAS>(prod, cnScale, cnOffset, Wout, bias, t2, pair, n, np, cs);
  else triOutLaunch<TP, 4, BIAS>(prod, cnScale, cnOffset, Wout, bias, t2, pair, n, np, cs);
}
template <class TP>
void triOut128(const TP* prod, const std::string& pre, const half* t2, float* pair, int n, int np, size_t cs) {
  triOutRaw<TP>(prod, W(pre + ".centerNormScale"), W(pre + ".centerNormOffset"), Wh(pre + ".outputProjection"), nullptr,
                t2, pair, n, np, cs);
}

// out[h][row] = (LN(x[row]) W)[h] for a projection to few heads (N a multiple of 16, W (C, N)):
// the single track's pair logits, the pair read once and written head-major (the layout the
// softmax reads) through shared memory.
template <int C, int N, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) lnHeadsK(const float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ Wp, float* __restrict__ out, size_t rows) {
  static_assert(N % 16 == 0, "the column loop takes 16 at a time");
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, LDW = N + 8, KS = C / 16, LDO = R + 4;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem; half* Ws = Xs + R * LDX; float* Os = (float*)(Ws + C * LDW);
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  for (int t = threadIdx.x; t < C * (N / 8); t += NTH) {
    int k = t / (N / 8), c = (t % (N / 8)) * 8;
    cpAsync16(Ws + k * LDW + c, Wp + (size_t)k * N + c, true);
  }
  cpCommit();
  lnRowsToShared<C, R, WARPS>(x, [&](int r) { size_t row = row0 + r; return row < rows ? row : SIZE_MAX; },
                              lnScale, lnOffset, Xs, LDX, warp, lane);
  cpWait<0>();
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
  if (!attr) { smemAttr((lnHeadsK<C, N, WARPS>), (int)smem); attr = true; }
  lnHeadsK<C, N, WARPS><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(x, W(scale), W(offset), Wh(w), out, rows);
}

// The grid attention's input projection: LN(pair row) -> q, k, v, gate (C x NQ), for output rows
// q0 .. q0+rows of the attention's (row, position) order; in the column direction (tr) output row
// (r, j) reads pair row (j, r), so the transposed copy the unfused form made is never written.
// With `bias` (only when one call covers every row), it also writes the pair bias - LN(pair) times
// the (C, 16) zero-padded bias projection, heads < 16 - straight into the flash kernel's
// [h][i][stride] f16 layout, scaled by log2(e), (i, j) swapped where `swap`.
template <int C, int NQ, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) gridInK(const float* __restrict__ pair, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ Wq, half* __restrict__ out, int n, size_t q0,
    size_t rows, bool tr, const half* __restrict__ Wb, half* __restrict__ bias, int heads, int stride, bool swap) {
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, NC = 64, LDW = NC + 8, KS = C / 16;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem;
  half* Wst = Xs + R * LDX;                                        // two stages of [C][LDW]
  half* Wbs = Wst + 2 * C * LDW;                                   // [C][24], the bias projection
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  auto issue = [&](int j, int st) {
    half* w = Wst + st * C * LDW;
    for (int t = threadIdx.x; t < C * (NC / 8); t += NTH) {
      int k = t / (NC / 8), c = (t % (NC / 8)) * 8;
      cpAsync16(w + k * LDW + c, Wq + (size_t)k * NQ + j * NC + c, true);
    }
    cpCommit();
  };
  issue(0, 0);
  lnRowsToShared<C, R, WARPS>(pair, [&](int r) {
      size_t q = row0 + r;
      if (q >= rows) return (size_t)SIZE_MAX;
      size_t Q = q0 + q;
      if (!tr) return Q;
      unsigned u = (unsigned)Q, a = u / (unsigned)n;           // 32-bit: a 64-bit divide is ~70 instructions
      return (size_t)(u - a * (unsigned)n) * n + a;
    }, lnScale, lnOffset, Xs, LDX, warp, lane);
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  size_t r0 = row0 + warp * 16 + g, r1 = r0 + 8;
  const int chunks = NQ / NC;
  for (int j = 0; j < chunks; ++j) {
    int st = j & 1;
    if (j + 1 < chunks) { issue(j + 1, st ^ 1); cpWait<1>(); }
    else cpWait<0>();
    __syncthreads();
    const half* w = Wst + st * C * LDW;
    float acc[NC / 8][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < NC / 16; ++n2) {
        uint32_t f[4];
        ldsm4t(f, w + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + n2 * 16 + (lane >> 4) * 8);
        mma16816(acc[2 * n2], xa[ks], f[0], f[1]); mma16816(acc[2 * n2 + 1], xa[ks], f[2], f[3]);
      }
    }
#pragma unroll
    for (int nt = 0; nt < NC / 8; ++nt) {
      int c = j * NC + nt * 8 + tig * 2;
      if (r0 < rows) *reinterpret_cast<half2*>(out + r0 * NQ + c) = __floats2half2_rn(acc[nt][0], acc[nt][1]);
      if (r1 < rows) *reinterpret_cast<half2*>(out + r1 * NQ + c) = __floats2half2_rn(acc[nt][2], acc[nt][3]);
    }
    __syncthreads();
  }
  if (bias) {
    for (int t = threadIdx.x; t < C * 2; t += NTH) {
      int k = t / 2, c = (t % 2) * 8;
      cpAsync16(Wbs + k * 24 + c, Wb + (size_t)k * 16 + c, true);
    }
    cpCommit();
    cpWait<0>();
    __syncthreads();
    float acc[2][4] = {};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t f[4];
      ldsm4t(f, Wbs + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * 24 + (lane >> 4) * 8);
      mma16816(acc[0], xa[ks], f[0], f[1]); mma16816(acc[1], xa[ks], f[2], f[3]);
    }
    for (int half8 = 0; half8 < 2; ++half8) {
      size_t q = r0 + half8 * 8;
      if (q >= rows) continue;
      unsigned Q = (unsigned)(q0 + q), qa = Q / (unsigned)n, qb = Q - qa * (unsigned)n;
      unsigned a = tr ? qb : qa, b = tr ? qa : qb;               // p = (a, b)
      size_t i = swap ? b : a, jj = swap ? a : b;
      for (int nt = 0; nt < 2; ++nt) for (int e = 0; e < 2; ++e) {
        int h = nt * 8 + tig * 2 + e;
        if (h < heads) bias[((size_t)h * n + i) * stride + jj] = __float2half(LOG2E * acc[nt][half8 * 2 + e]);
      }
    }
  }
}
// pair[(j, r)] += (gathered[(r, j)] Wout) for output rows q0 .. q0+rows (the column direction)
template <int C, int WD, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) gridOutK(const half* __restrict__ gathered, const half* __restrict__ Wout,
    float* __restrict__ pair, int n, size_t q0, size_t rows, bool tr) {
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = WD + 8, LDW = C + 8, KS = WD / 16, NT = C / 8;
  extern __shared__ __align__(16) unsigned char smem[];
  half* Ws = (half*)smem; half* Xs = Ws + WD * LDW;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  for (int t = threadIdx.x; t < WD * (C / 8); t += NTH) {
    int k = t / (C / 8), c = (t % (C / 8)) * 8;
    cpAsync16(Ws + k * LDW + c, Wout + (size_t)k * C + c, true);
  }
  for (int t = threadIdx.x; t < R * (WD / 8); t += NTH) {
    int r = t / (WD / 8), c = (t % (WD / 8)) * 8;
    size_t q = row0 + r;
    cpAsync16(Xs + r * LDX + c, gathered + (q < rows ? q : 0) * WD + c, q < rows);
  }
  cpCommit();
  cpWait<0>();
  __syncthreads();
  float acc[NT][4] = {};
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    uint32_t xa[4];
    ldsm4(xa, Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
#pragma unroll
    for (int et = 0; et < NT; et += 2) {
      uint32_t f[4];
      ldsm4t(f, Ws + (ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * LDW + (et + (lane >> 4)) * 8);
      mma16816(acc[et], xa, f[0], f[1]); mma16816(acc[et + 1], xa, f[2], f[3]);
    }
  }
  size_t qa = row0 + warp * 16 + g;
  for (int half8 = 0; half8 < 2; ++half8) {
    size_t q = qa + half8 * 8;
    if (q >= rows) continue;
    size_t Q = q0 + q, p = tr ? (Q % n) * n + Q / n : Q;
#pragma unroll
    for (int et = 0; et < NT; ++et) {
      int c = et * 8 + tig * 2;
      float2* d = (float2*)(pair + p * C + c); float2 v = *d;
      v.x += acc[et][half8 * 2]; v.y += acc[et][half8 * 2 + 1]; *d = v;
    }
  }
}
constexpr int GI_WARPS = 8, GO_WARPS = 8;
template <int WARPS>
void gridInAt(const float* pair, const std::string& pre, const std::string& wq, half* out, int n, size_t q0,
              size_t rows, bool tr, const half* Wb, half* bias, int heads, int stride, bool swap) {
  constexpr int C = 128, NQ = 512, R = 16 * WARPS;
  size_t smem = (size_t)R * (C + 8) * 2 + (size_t)2 * C * (64 + 8) * 2 + (size_t)C * 24 * 2;
  static bool attr = false;
  if (!attr) { smemAttr((gridInK<C, NQ, WARPS>), (int)smem); attr = true; }
  gridInK<C, NQ, WARPS><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
    pair, W(pre + ".actNormScale"), W(pre + ".actNormOffset"), Wh(wq), out, n, q0, rows, tr, Wb, bias, heads, stride,
    swap);
}
inline size_t gridInSmem(int warps) { return (size_t)16 * warps * (128 + 8) * 2 + (size_t)2 * 128 * (64 + 8) * 2 + (size_t)128 * 24 * 2; }
inline size_t gridOutSmem(int warps) { return (size_t)128 * (128 + 8) * 2 + (size_t)16 * warps * (128 + 8) * 2; }
inline bool gridFusedFits() { return fitsSmem(gridInSmem(2)) && fitsSmem(gridOutSmem(2)); }
inline void gridIn128(const float* pair, const std::string& pre, const std::string& wq, half* out, int n, size_t q0,
                      size_t rows, bool tr, const half* Wb = nullptr, half* bias = nullptr, int heads = 0,
                      int stride = 0, bool swap = false) {
  switch (warpsFitting(warpsFor(rows, {GI_WARPS, 4, 2}), {GI_WARPS, 4, 2}, gridInSmem)) {
    case 2: gridInAt<2>(pair, pre, wq, out, n, q0, rows, tr, Wb, bias, heads, stride, swap); break;
    case 4: gridInAt<4>(pair, pre, wq, out, n, q0, rows, tr, Wb, bias, heads, stride, swap); break;
    default: gridInAt<GI_WARPS>(pair, pre, wq, out, n, q0, rows, tr, Wb, bias, heads, stride, swap);
  }
}
template <int WARPS>
void gridOutAt(const half* gathered, const std::string& w, float* pair, int n, size_t q0, size_t rows, bool tr) {
  constexpr int C = 128, WD = 128, R = 16 * WARPS;
  size_t smem = (size_t)WD * (C + 8) * 2 + (size_t)R * (WD + 8) * 2;
  static bool attr = false;
  if (!attr) { smemAttr((gridOutK<C, WD, WARPS>), (int)smem); attr = true; }
  gridOutK<C, WD, WARPS><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(gathered, Wh(w), pair, n, q0, rows, tr);
}
inline void gridOut128(const half* gathered, const std::string& w, float* pair, int n, size_t q0, size_t rows, bool tr) {
  switch (warpsFitting(warpsFor(rows, {GO_WARPS, 4, 2}), {GO_WARPS, 4, 2}, gridOutSmem)) {
    case 2: gridOutAt<2>(gathered, w, pair, n, q0, rows, tr); break;
    case 4: gridOutAt<4>(gathered, w, pair, n, q0, rows, tr); break;
    default: gridOutAt<GO_WARPS>(gathered, w, pair, n, q0, rows, tr);
  }
}
