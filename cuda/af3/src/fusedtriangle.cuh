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

// LayerNorm of a block's R rows into shared memory as f16, a warp a row and several rows' loads in
// flight per warp (one at a time left the loads latency-bound). rowOf(r) is row r's index in x,
// or SIZE_MAX for a row past the end (normalised zeros).
// up to LN_ROWS_IN_FLIGHT rows' loads in flight a warp: 8 where it was 4 (ncu: ~30% of gridInK's stall
// samples were this prologue waiting on DRAM) - 1-3% on triInK, gridInK and the transition, byte-identical;
// 16 measures the same as 8
// The pair in bf16 (AlphaFold 3's own activation precision) while a stack runs on kernels that take it: PAIR16
// says the buffer behind a `float* pair` holds bf16 now, and these kernels read it through PT (their pair element
// type, picked at launch by WITH_PAIR_T). A path that has no bf16 form asks needF32Pair, which refuses rather
// than read bf16 as f32.
inline bool PAIR16 = false;
inline void needF32Pair(const char* where) {
  if (PAIR16) { fprintf(stderr, "%s has no bf16-pair form: the stack should have kept the pair f32\n", where); exit(1); }
}
// ...and with the flag given (the trunk's pair outside its pair-track updates: t.p16)
#define WITH_PT(flag, ...) do { if (flag) { using PT = __nv_bfloat16; __VA_ARGS__; } else { using PT = float; __VA_ARGS__; } } while (0)
template <class PT> __device__ __forceinline__ void pairSt(float* p, size_t i, float v) {
  if constexpr (std::is_same_v<PT, float>) ptx_st_global_f32(p + i, v);
  else ptx_st_global_u16(reinterpret_cast<PT*>(p) + i, std::is_same_v<PT, __nv_bfloat16> ? ptx_f32_to_bf16_raw(v) : ptx_f32_to_f16_raw(v));
}
#define WITH_PAIR_T(...) do { if (PAIR16) { using PT = __nv_bfloat16; __VA_ARGS__; } else { using PT = float; __VA_ARGS__; } } while (0)
template <class PT> __device__ __forceinline__ float pairLd(const float* p, size_t i) {
  if constexpr (std::is_same_v<PT, float>) return ptx_ld_global_f32(p + i);
  else return toF(reinterpret_cast<const PT*>(p)[i]);
}
template <class PT> using PairRaw4 = std::conditional_t<sizeof(PT) == 4, float4, uint2>;
template <class PT> __device__ __forceinline__ PairRaw4<PT> pairLdRaw4(const float* p, size_t i) {
  if constexpr (sizeof(PT) == 4) return ptx_ld_global_v4_f32(p + i);
  else return ptx_ld_global_v2_u32(reinterpret_cast<const PT*>(p) + i);
}
template <class PT> __device__ __forceinline__ float4 pairUnpack4(PairRaw4<PT> raw) {
  if constexpr (std::is_same_v<PT, float>) return raw;
  else {
    float2 a = std::is_same_v<PT, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(raw.x) : ptx_f16x2_to_f32x2(raw.x);
    float2 b = std::is_same_v<PT, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(raw.y) : ptx_f16x2_to_f32x2(raw.y);
    return make_float4(a.x, a.y, b.x, b.y);
  }
}
template <class PT> __device__ __forceinline__ float4 pairLd4(const float* p, size_t i) {
  return pairUnpack4<PT>(pairLdRaw4<PT>(p, i));
}
template <class PT> __device__ __forceinline__ void pairSt4(float* p, size_t i, float4 v) {
  if constexpr (std::is_same_v<PT, float>) ptx_st_global_v4_f32(p + i, v);
  else {
    uint2 w;
    if constexpr (std::is_same_v<PT, __nv_bfloat16>) {
      w.x = ptx_cvt_bf16x2_f32(v.x, v.y);
      w.y = ptx_cvt_bf16x2_f32(v.z, v.w);
    } else {
      w.x = ptx_cvt_f16x2_f32(v.x, v.y);
      w.y = ptx_cvt_f16x2_f32(v.z, v.w);
    }
    ptx_st_global_v2_u32(reinterpret_cast<PT*>(p) + i, w);
  }
}
template <class PT> __device__ __forceinline__ float2 pairLd2(const float* p, size_t i) {
  if constexpr (std::is_same_v<PT, float>) return ptx_ld_global_v2_f32(p + i);
  else {
    uint32_t u = ptx_ld_global_u32(reinterpret_cast<const PT*>(p) + i);
    return std::is_same_v<PT, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(u) : ptx_f16x2_to_f32x2(u);
  }
}
template <class PT> __device__ __forceinline__ void pairSt2(float* p, size_t i, float2 v) {
  if constexpr (std::is_same_v<PT, float>) ptx_st_global_v2_f32(p + i, v);
  else {
    uint32_t u = std::is_same_v<PT, __nv_bfloat16> ? ptx_cvt_bf16x2_f32(v.x, v.y) : ptx_cvt_f16x2_f32(v.x, v.y);
    ptx_st_global_u32(reinterpret_cast<PT*>(p) + i, u);
  }
}
#ifndef LN_VEC16
#define LN_VEC16 1
#endif
#ifndef LN_VEC_MIN_K
#define LN_VEC_MIN_K 8
#endif
#ifndef LN_ROWS_IN_FLIGHT
#define LN_ROWS_IN_FLIGHT 8
#endif
template <int C, int R, int WARPS, class PT = float, bool FAST_VAR = false, int MAX_B = LN_ROWS_IN_FLIGHT, class RowOf>
__device__ __forceinline__ void lnRowsToShared(const float* __restrict__ x, RowOf rowOf, const float* __restrict__ scale,
                                               const float* __restrict__ offset, half* Xs, int LDX, int warp, int lane,
                                               const float* __restrict__ scale2 = nullptr, const float* __restrict__ offset2 = nullptr,
                                               half* __restrict__ outLn2 = nullptr) {
  constexpr int RPW = R / WARPS, B = MAX_B > 0 && RPW % MAX_B == 0 ? MAX_B
                                  : (MAX_B >= 4 && RPW % 4 == 0) ? 4 : ((MAX_B >= 2 && RPW % 2 == 0) ? 2 : 1), K = C / 32;
#if LN_VEC16
  if constexpr (!std::is_same_v<PT, float> && K >= LN_VEC_MIN_K && K % 2 == 0 && MAX_B > 1) {
    // a bf16 row 4/8/16 bytes a lane (including C 384 -> K 12 -> VEC 4 uint2 loads/stores, where K % 8 == 0 left
    // OpenDDE's 384-channel triangleOutK and transitionUpK on 2-byte scalar loads): a lane's channels
    // (lane + 32 kk) * VEC .. + VEC-1, one load and one store each (opendde transitionUpK<384> 56.1 -> 45.0 ms,
    // triangleOutK<384> 39.2 -> 35.7 ms on 5CAJ)
    constexpr int VEC = K % 8 == 0 ? 8 : (K % 4 == 0 ? 4 : 2), KK = K / VEC;
    using VT = std::conditional_t<VEC == 8, uint4, std::conditional_t<VEC == 4, uint2, uint32_t>>;
    float sc[K], of[K], sc2[K], of2[K];
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      int ch = (lane + 32 * kk) * VEC;
      if constexpr (VEC >= 4) {
        *reinterpret_cast<float4*>(sc + kk * VEC) = ptx_ldg_v4_f32(scale + ch);
        *reinterpret_cast<float4*>(of + kk * VEC) = ptx_ldg_v4_f32(offset + ch);
        if constexpr (VEC == 8) {
          *reinterpret_cast<float4*>(sc + kk * VEC + 4) = ptx_ldg_v4_f32(scale + ch + 4);
          *reinterpret_cast<float4*>(of + kk * VEC + 4) = ptx_ldg_v4_f32(offset + ch + 4);
        }
      } else {
        *reinterpret_cast<float2*>(sc + kk * VEC) = ptx_ldg_v2_f32(scale + ch);
        *reinterpret_cast<float2*>(of + kk * VEC) = ptx_ldg_v2_f32(offset + ch);
      }
    }
    if (outLn2) {
#pragma unroll
      for (int kk = 0; kk < KK; ++kk) {
        int ch = (lane + 32 * kk) * VEC;
        if constexpr (VEC >= 4) {
          float4 s0, o0;
          asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(s0.x), "=f"(s0.y), "=f"(s0.z), "=f"(s0.w) : "l"(scale2 + ch));
          asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(o0.x), "=f"(o0.y), "=f"(o0.z), "=f"(o0.w) : "l"(offset2 + ch));
          *reinterpret_cast<float4*>(sc2 + kk * VEC) = s0;
          *reinterpret_cast<float4*>(of2 + kk * VEC) = o0;
          if constexpr (VEC == 8) {
            float4 s1, o1;
            asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(s1.x), "=f"(s1.y), "=f"(s1.z), "=f"(s1.w) : "l"(scale2 + ch + 4));
            asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(o1.x), "=f"(o1.y), "=f"(o1.z), "=f"(o1.w) : "l"(offset2 + ch + 4));
            *reinterpret_cast<float4*>(sc2 + kk * VEC + 4) = s1;
            *reinterpret_cast<float4*>(of2 + kk * VEC + 4) = o1;
          }
        } else {
          float2 s0, o0;
          asm volatile("ld.global.nc.v2.f32 {%0, %1}, [%2];" : "=f"(s0.x), "=f"(s0.y) : "l"(scale2 + ch));
          asm volatile("ld.global.nc.v2.f32 {%0, %1}, [%2];" : "=f"(o0.x), "=f"(o0.y) : "l"(offset2 + ch));
          *reinterpret_cast<float2*>(sc2 + kk * VEC) = s0;
          *reinterpret_cast<float2*>(of2 + kk * VEC) = o0;
        }
      }
    }
    for (int base = 0; base < RPW; base += B) {
      VT w[B][KK];
#pragma unroll
      for (int b = 0; b < B; ++b) {
        size_t row = rowOf(warp + (base + b) * WARPS);
#pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
          const void* src = reinterpret_cast<const PT*>(x) + row * C + (lane + 32 * kk) * VEC;
          if constexpr (VEC == 8) w[b][kk] = row != SIZE_MAX ? ptx_ldg_v4_u32(src) : uint4{0, 0, 0, 0};
          else if constexpr (VEC == 4) w[b][kk] = row != SIZE_MAX ? ptx_ldg_v2_u32(src) : uint2{0, 0};
          else w[b][kk] = row != SIZE_MAX ? ptx_ldg_u32(src) : 0u;
        }
      }
#pragma unroll
      for (int b = 0; b < B; ++b) {
        float v[K];
#pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
          const uint32_t* u32 = reinterpret_cast<const uint32_t*>(&w[b][kk]);
#pragma unroll
          for (int e = 0; e < VEC / 2; ++e) {
            float2 f = std::is_same_v<PT, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(u32[e]) : ptx_f16x2_to_f32x2(u32[e]);
            v[kk * VEC + 2 * e] = f.x;
            v[kk * VEC + 2 * e + 1] = f.y;
          }
        }
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < K; ++k) s += v[k];
        s = ptx_warp_sum_f32(s);
        float mean = ptx_div_f32(s, (float)C), inv;
        if constexpr (FAST_VAR) {
          float ss = 0.f;
#pragma unroll
          for (int k = 0; k < K; ++k) ss += v[k] * v[k];
          ss = ptx_warp_sum_f32(ss);
          inv = ptx_rsqrt_f32(ptx_fma_f32(-mean, mean, ptx_div_f32(ss, (float)C)) + 1e-5f);
        } else {
          float q = 0.f;
#pragma unroll
          for (int k = 0; k < K; ++k) { float d = v[k] - mean; q += d * d; }
          q = ptx_warp_sum_f32(q);
          inv = ptx_rsqrt_f32(ptx_div_f32(q, (float)C) + 1e-5f);
        }
        int r = warp + (base + b) * WARPS;
#pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
          VT o4; uint32_t* oh = reinterpret_cast<uint32_t*>(&o4);
#pragma unroll
          for (int e = 0; e < VEC / 2; ++e) {
            float y0 = ptx_fma_f32((v[kk * VEC + 2 * e] - mean) * inv, sc[kk * VEC + 2 * e], of[kk * VEC + 2 * e]);
            float y1 = ptx_fma_f32((v[kk * VEC + 2 * e + 1] - mean) * inv, sc[kk * VEC + 2 * e + 1], of[kk * VEC + 2 * e + 1]);
            oh[e] = ptx_cvt_f16x2_f32(y0, y1);
          }
          half* dst = Xs + r * LDX + (lane + 32 * kk) * VEC;
          if constexpr (VEC == 8) ptx_st_shared_v4_u32(dst, o4);
          else if constexpr (VEC == 4) ptx_st_shared_v2_u32(dst, o4);
          else ptx_st_shared_u32(dst, o4);
        }
        if (outLn2) {
          size_t row = rowOf(r);
          if (row != SIZE_MAX) {
#pragma unroll
            for (int kk = 0; kk < KK; ++kk) {
              int ch = (lane + 32 * kk) * VEC;
              VT o4_2; uint32_t* oh2 = reinterpret_cast<uint32_t*>(&o4_2);
#pragma unroll
              for (int e = 0; e < VEC / 2; ++e) {
                float y0 = ptx_fma_f32((v[kk * VEC + 2 * e] - mean) * inv, sc2[kk * VEC + 2 * e], of2[kk * VEC + 2 * e]);
                float y1 = ptx_fma_f32((v[kk * VEC + 2 * e + 1] - mean) * inv, sc2[kk * VEC + 2 * e + 1], of2[kk * VEC + 2 * e + 1]);
                oh2[e] = ptx_cvt_f16x2_f32(y0, y1);
              }
              half* dst2 = outLn2 + row * C + ch;
              if constexpr (VEC == 8) ptx_st_global_v4_u32(dst2, o4_2);
              else if constexpr (VEC == 4) ptx_st_global_v2_u32(dst2, o4_2);
              else ptx_st_global_u32(dst2, o4_2);
            }
          }
        }
      }
    }
    return;
  } else {
#else
  {
#endif
  float sc[MAX_B > 1 ? K : 1], of[MAX_B > 1 ? K : 1]; // a lane's channels' scale and offset, loaded once for all its rows
  if constexpr (MAX_B > 1) {
#pragma unroll
    for (int k = 0; k < K; ++k) { sc[k] = ptx_ldg_f32(scale + lane + 32 * k); of[k] = ptx_ldg_f32(offset + lane + 32 * k); }
  }
  for (int base = 0; base < RPW; base += B) {
    using RawT = std::conditional_t<std::is_same_v<PT, float>, float, unsigned short>;
    RawT w[B][K];
#pragma unroll
    for (int b = 0; b < B; ++b) {
      size_t row = rowOf(warp + (base + b) * WARPS);
      const PT* xp = reinterpret_cast<const PT*>(x) + (row != SIZE_MAX ? row * C + lane : 0);
#pragma unroll
      for (int k = 0; k < K; ++k) {
        if constexpr (std::is_same_v<PT, float>) w[b][k] = row != SIZE_MAX ? ptx_ldg_f32(xp + 32 * k) : 0.f;
        else w[b][k] = row != SIZE_MAX ? ptx_ldg_u16(xp + 32 * k) : (unsigned short)0;
      }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      float v[K], s = 0.f;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        if constexpr (std::is_same_v<PT, float>) v[k] = w[b][k];
        else if constexpr (std::is_same_v<PT, __nv_bfloat16>) v[k] = ptx_bf16_raw_to_f32(w[b][k]);
        else v[k] = ptx_f16_raw_to_f32(w[b][k]);
        s = ptx_add_f32(s, v[k]);
      }
      s = ptx_warp_sum_f32(s);
      float mean = ptx_div_f32(s, (float)C), inv;
      if constexpr (FAST_VAR) {
        float ss = 0.f;
#pragma unroll
        for (int k = 0; k < K; ++k) ss = ptx_fma_f32(v[k], v[k], ss);
        ss = ptx_warp_sum_f32(ss);
        inv = ptx_rsqrt_f32(ptx_add_f32(ptx_fma_f32(-mean, mean, ptx_div_f32(ss, (float)C)), 1e-5f));
      } else {
        float q = 0.f;
#pragma unroll
        for (int k = 0; k < K; ++k) { float d = ptx_sub_f32(v[k], mean); q = ptx_fma_f32(d, d, q); }
        q = ptx_warp_sum_f32(q);
        inv = ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(q, (float)C), 1e-5f));
      }
      int r = warp + (base + b) * WARPS;
      half* xrow = Xs + r * LDX + lane;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        float sk = MAX_B > 1 ? sc[k] : ptx_ldg_f32(scale + lane + 32 * k);
        float ok = MAX_B > 1 ? of[k] : ptx_ldg_f32(offset + lane + 32 * k);
        ptx_st_shared_u16(xrow + 32 * k, ptx_f32_to_f16_raw(ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(v[k], mean), inv), sk, ok)));
      }
      if (outLn2) {
        size_t row = rowOf(r);
        if (row != SIZE_MAX) {
          half* dst2 = outLn2 + row * C + lane;
#pragma unroll
          for (int k = 0; k < K; ++k) {
            float sk2, ok2;
            asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(sk2) : "l"(scale2 + lane + 32 * k));
            asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(ok2) : "l"(offset2 + lane + 32 * k));
            ptx_st_global_u16(dst2 + 32 * k, ptx_f32_to_f16_raw(ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(v[k], mean), inv), sk2, ok2)));
          }
        }
      }
    }
  }
  }
}

// W's columns [col0, col0 + cols) as cols / NC tiles of [K][NC], each contiguous: a kernel streaming NC
// columns at a time then reads one run of global memory per stage, where from W itself each 16-byte
// piece is a row (ld halves) from its neighbour's - and a cp.async write takes one shared wavefront per
// global row it touches (Nsight Compute: triIn256K's stage writes at 5.4x their ideal wavefronts).
// Tiled and swizzled (stageSw), a 1,044-token ESMFold2 trunk's three streaming kernels: triIn256K 1010 ->
// 893 ms, transitionUpK 877 -> 737, triangleOutK 736 -> 687 - the trunk 3753 -> 3463 ms, byte-identical;
// the tiling is a few microseconds a call, into scratch, so it costs no memory that persists
__global__ void tileColumnsK(const half* __restrict__ W, int K, int ld, int col0, int cols, int NC, half* __restrict__ out) {
  size_t e = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (e >= (size_t)K * cols) return;
  int c = (int)(e % NC); size_t rest = e / NC; int k = (int)(rest % K), t = (int)(rest / K);
  ptx_st_global_v4_u32(out + e, ptx_ldg_v4_u32(W + (size_t)k * ld + col0 + t * NC + c));
}
inline void tileColumns(const half* W, int K, int ld, int col0, int cols, int NC, half* out) {
  size_t n8 = (size_t)K * cols / 8;
  tileColumnsK<<<(unsigned)((n8 + 255) / 256), 256, 0, STREAM>>>(W, K, ld, col0, cols, NC, out);
}
// A weight stage of NC = 16 or 32 halves a row, unpadded, its 16-byte chunks XOR-swizzled so both the
// cp.async writes (a warp's 512 contiguous bytes) and the ldmatrix.trans reads (8 consecutive rows at one
// chunk) are conflict-free - padded rows kept the reads clean and serialised the writes
template <int NC>
__device__ __forceinline__ int stageSw(int k, int c) {
  static_assert(NC == 16 || NC == 32, "a 32- or 64-byte stage row");
  if constexpr (NC == 16) return k * NC + (c ^ (((k >> 2) & 1) << 3));
  else return k * NC + (c ^ (((k >> 1) & 3) << 3));
}

// triIn256K's weights as it streams them (5 C^2 halves, NC columns a tile): the
// projection's first half (a and b), its second (their gates), then the gating linear - so step j's
// tile is at (j * C) * NC whichever of the three it falls in, the second half 2 C^2 on
constexpr size_t triInTileHalves(int C) { return (size_t)5 * C * C; }
// (the three segments in one launch: ESMFold2's trunk at 261 tokens 214.2 -> 213.05 ms against three)
__global__ void tileTriInK(const half* __restrict__ Wpg, const half* __restrict__ Wg, int C, int NC, half* __restrict__ out) {
  size_t e = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8, seg = (size_t)2 * C * C;
  if (e >= (size_t)5 * C * C) return;
  int part = e / seg < 2 ? (int)(e / seg) : 2;           // the projection's halves, then the gating linear
  size_t f = e - part * seg;
  int c = (int)(f % NC); size_t rest = f / NC; int k = (int)(rest % C), t = (int)(rest / C);
  const half* src = part < 2 ? Wpg + (size_t)k * 4 * C + part * 2 * C + t * NC + c : Wg + (size_t)k * C + t * NC + c;
  ptx_st_global_v4_u32(out + e, ptx_ldg_v4_u32(src));
}
inline void tileTriIn(const half* Wpg, const half* Wg, int C, int NC, half* out) {
  size_t n8 = (size_t)5 * C * C / 8;
  tileTriInK<<<(unsigned)((n8 + 255) / 256), 256, 0, STREAM>>>(Wpg, Wg, C, NC, out);
}
__global__ void tileTriIn3K(const half* __restrict__ Wp, const half* __restrict__ Wgate, const half* __restrict__ Wg,
                            int C, int NC, half* __restrict__ out) {
  size_t e = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8, seg = (size_t)2 * C * C;
  if (e >= (size_t)5 * C * C) return;
  int part = e / seg < 2 ? (int)(e / seg) : 2;
  size_t f = e - part * seg;
  int c = (int)(f % NC); size_t rest = f / NC; int k = (int)(rest % C), t = (int)(rest / C);
  const half* src = part == 0 ? Wp + (size_t)k * 2 * C + t * NC + c
                  : part == 1 ? Wgate + (size_t)k * 2 * C + t * NC + c
                              : Wg + (size_t)k * C + t * NC + c;
  ptx_st_global_v4_u32(out + e, ptx_ldg_v4_u32(src));
}
inline const half* triInTiles(const half* Wp, const half* Wgate, const half* Wg, int C, int NC = 16) {
  static std::map<std::pair<const half*, int>, half*> tiles;
  static std::vector<half*> chunks; static size_t used = 0, have = 0;
  static bool hooked = false;
  if (!hooked) {
    FORGET_HOOKS.push_back([] { for (half* p : chunks) CK(cudaFreeAsync(p, STREAM)); chunks.clear(); tiles.clear(); used = have = 0; });
    hooked = true;
  }
  auto it = tiles.find({Wp, NC});
  if (it != tiles.end()) return it->second;
  size_t n = (triInTileHalves(C) + 127) / 128 * 128;
  if (used + n > have) {
    have = std::max<size_t>(n, (size_t)8 << 20);
    chunks.push_back(poolAllocT<half>(have)); used = 0;
  }
  half* out = chunks.back() + used; used += n;
  size_t n8 = (size_t)5 * C * C / 8;
  tileTriIn3K<<<(unsigned)((n8 + 255) / 256), 256, 0, STREAM>>>(Wp, Wgate, Wg, C, NC, out);
  return tiles[{Wp, NC}] = out;
}

constexpr int TI_NC = 32;
// a stage's two [C][TI_NC] weight chunks, unpadded 64-byte rows with each 16-byte piece XOR-swizzled by
// (k >> 1) & 3 (tiSw): eight cp.async writes then fill one aligned 128-byte line and the 8-row ldmatrix
// reads still hit distinct banks (the padded 80-byte rows: ncu, 31% of triInK's shared wavefronts excessive).
// 55.3 -> 53.2 ms of a 261-token fold, 725 -> 698 at 1,044, byte-identical. (gridInK's 144-byte rows the
// same way, 128 bytes with k & 7: no change, so left.)
__host__ __device__ constexpr size_t tiStage(int C) { return (size_t)2 * C * TI_NC * 2; }
__device__ __forceinline__ int tiSw(int k, int c) { return stageSw<TI_NC>(k, c); }

// TA: a and b's type - f16, or bf16 so the contraction can write a bf16 product (f16 overflows).
// BIAS: AlphaFold 2's projections carry biases, AF3's do not - `bias` is [4C, the projection | gate columns
// in Wpg's order][C, the gating linear's]; without BIAS the kernel is the one AF3 has always run
template <int C, int WARPS, class TA, bool BIAS = false, class PT = float>
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
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, LDW = TI_NC, KS = C / 16, LDT = R + 8;
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
  // (the gating linear two tiles a step, one a stage: four of twelve steps become two, eight chains each)
  const int steps = C / 16 + C / TI_NC / 2;
  const int k0 = (int)threadIdx.x / (TI_NC / 8), c0_ = ((int)threadIdx.x % (TI_NC / 8)) * 8, sw0 = tiSw(k0, c0_);
  auto issue = [&](int j, int st) {
    half *w0 = W0(st) + sw0, *w1 = W1(st) + sw0;
    #pragma unroll
    for (int it = 0; it < (C * (TI_NC / 8) + NTH - 1) / NTH; ++it) {
      int k = k0 + it * (NTH / (TI_NC / 8)), c = c0_;
      if ((C * (TI_NC / 8)) % NTH != 0 && k >= C) break;
      int off = it * (NTH / (TI_NC / 8)) * TI_NC;
      if (j < C / 16) {
        cpAsync16(w0 + off, Wpg + (size_t)k * 4 * C + j * 32 + c, true);
        cpAsync16(w1 + off, Wpg + (size_t)k * 4 * C + 2 * C + j * 32 + c, true);
      } else {
        cpAsync16(w0 + off, Wg + (size_t)k * C + 2 * (j - C / 16) * TI_NC + c, true);
        cpAsync16(w1 + off, Wg + (size_t)k * C + (2 * (j - C / 16) + 1) * TI_NC + c, true);
      }
    }
    cpCommit();
  };
  issue(0, 0);
  lnRowsToShared<C, R, WARPS, PT>(pair, [&](int r) { return pairOf(row0 + r); }, lnScale, lnOffset, Xs, LDX, warp, lane);
  cpWait<0>();
  __syncthreads();
  uint32_t xa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm4(xa[ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  if (1 < steps) issue(1, 1);
  int lr0 = warp * 16 + g, lr1 = lr0 + 8;                          // the thread's rows within the block
  size_t pr0 = pairOf(row0 + lr0), pr1 = pairOf(row0 + lr1);
  float m0 = pr0 != SIZE_MAX ? (mask ? ptx_ldg_f32(mask + pr0) : 1.f) : 0.f, m1 = pr1 != SIZE_MAX ? (mask ? ptx_ldg_f32(mask + pr1) : 1.f) : 0.f;
  const int ch0 = (int)threadIdx.x / (R / 8), r8 = ((int)threadIdx.x % (R / 8)) * 8;
  for (int j = 0; j < steps; ++j) {
    int st = j & 1;
    bool gating = j >= C / 16;
    float2 bp[TI_NC / 8], bq[TI_NC / 8];
    if constexpr (BIAS) {
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        int col = (gating ? 4 * C + 2 * (j - C / 16) * TI_NC : j * TI_NC) + nt * 8 + tig * 2;
        int qcol = gating ? col + TI_NC : 2 * C + col;
        bp[nt] = ptx_ldg_v2_f32(bias + col);
        bq[nt] = ptx_ldg_v2_f32(bias + qcol);
      }
    }
    if (j > 0) {
      cpWait<0>();
      __syncthreads();
    }
    const half *w0 = W0(st), *w1 = W1(st);
    float p[TI_NC / 8][4] = {}, q[TI_NC / 8][4] = {};
    constexpr int N2 = TI_NC / 16, STEPS = KS * N2;
    const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
    uint32_t f0[4], f1[4];
    ldsm4t(f0, w0 + tiSw(kl, cl));
    if (j > 0 && j + 1 < steps) issue(j + 1, st ^ 1);
#pragma unroll
    for (int s = 0; s < STEPS; ++s) {
      const int ks = s / N2, n2 = s % N2;
      ldsm4t(f1, w1 + tiSw(ks * 16 + kl, n2 * 16 + cl));
      mma16816(p[2 * n2], xa[ks], f0[0], f0[1]); mma16816(p[2 * n2 + 1], xa[ks], f0[2], f0[3]);
      if (s + 1 < STEPS) {
        const int nks = (s + 1) / N2, nn2 = (s + 1) % N2;
        ldsm4t(f0, w0 + tiSw(nks * 16 + kl, nn2 * 16 + cl));
      }
      mma16816(q[2 * n2], xa[ks], f1[0], f1[1]); mma16816(q[2 * n2 + 1], xa[ks], f1[2], f1[3]);
    }
    if constexpr (BIAS) {
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        float b0 = bp[nt].x, b1 = bp[nt].y, g0 = bq[nt].x, g1 = bq[nt].y;
        p[nt][0] = ptx_add_f32(p[nt][0], b0); p[nt][1] = ptx_add_f32(p[nt][1], b1);
        p[nt][2] = ptx_add_f32(p[nt][2], b0); p[nt][3] = ptx_add_f32(p[nt][3], b1);
        q[nt][0] = ptx_add_f32(q[nt][0], g0); q[nt][1] = ptx_add_f32(q[nt][1], g1);
        q[nt][2] = ptx_add_f32(q[nt][2], g0); q[nt][3] = ptx_add_f32(q[nt][3], g1);
      }
    }
    if (gating) {                       // t2, row-major: columns (j - C/16)*32 + nt*8 + 2 tig
      int c0 = 2 * (j - C / 16) * TI_NC;                    // p the first tile's columns, q the second's
      half* Ys = Xs + warp * 16 * LDX;
      constexpr int W2 = 2 * TI_NC;
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        int c = nt * 8 + tig * 2;
        ptx_st_shared_u32(Ys + g * LDX + c, ptx_cvt_f16x2_f32(p[nt][0], p[nt][1]));
        ptx_st_shared_u32(Ys + (g + 8) * LDX + c, ptx_cvt_f16x2_f32(p[nt][2], p[nt][3]));
        ptx_st_shared_u32(Ys + g * LDX + TI_NC + c, ptx_cvt_f16x2_f32(q[nt][0], q[nt][1]));
        ptx_st_shared_u32(Ys + (g + 8) * LDX + TI_NC + c, ptx_cvt_f16x2_f32(q[nt][2], q[nt][3]));
      }
      __syncwarp();
      const int r_base = lane / (W2 / 8), c = (lane % (W2 / 8)) * 8;
#pragma unroll
      for (int i = 0; i < 16 * (W2 / 8) / 32; ++i) {
        int r = i * (32 / (W2 / 8)) + r_base;
        size_t row = row0 + warp * 16 + r;
        if (row < pp) ptx_st_global_v4_u32(t2 + row * C + c0 + c, ptx_ld_shared_v4_u32(Ys + r * LDX + c));
      }
    } else {
#pragma unroll
      for (int nt = 0; nt < TI_NC / 8; ++nt) {
        int ch = nt * 4 + tig;                    // (sigmH: the product is rounded to 16 bits next)
        Ta[ch * LDT + lr0] = fromF<TA>(ptx_mul_f32(ptx_mul_f32(p[nt][0], sigmH(q[nt][0])), m0));
        Tb[ch * LDT + lr0] = fromF<TA>(ptx_mul_f32(ptx_mul_f32(p[nt][1], sigmH(q[nt][1])), m0));
        Ta[ch * LDT + lr1] = fromF<TA>(ptx_mul_f32(ptx_mul_f32(p[nt][2], sigmH(q[nt][2])), m1));
        Tb[ch * LDT + lr1] = fromF<TA>(ptx_mul_f32(ptx_mul_f32(p[nt][3], sigmH(q[nt][3])), m1));
      }
      __syncthreads();
      if (row0 + r8 < cs) {
        #pragma unroll
        for (int it = 0; it < (16 * (R / 8) + NTH - 1) / NTH; ++it) {
          int ch = ch0 + it * (NTH / (R / 8));
          if ((16 * (R / 8)) % NTH != 0 && ch >= 16) break;
          size_t at = (size_t)(j * 16 + ch) * cs + row0 + r8;
          uint4 va = ptx_ld_shared_v4_u32(Ta + ch * LDT + r8);
          uint4 vb = ptx_ld_shared_v4_u32(Tb + ch * LDT + r8);
          ptx_st_global_v4_u32(a + at, va);
          ptx_st_global_v4_u32(b + at, vb);
        }
      }
    }
  }
}

// pair += (LN_center(prod) Wout) * sigmoid(t2); prod channel-major [C][pairs] f32
template <int C, int WARPS, class TP, bool BIAS = false, class PT = float>
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
    for (int k = 0; k < C / 32; ++k) { v[k] = toF(Ps[(lane + 32 * k) * LDP + r]); s = ptx_add_f32(s, v[k]); }
    s = ptx_warp_sum_f32(s);
    float mean = ptx_div_f32(s, (float)C), q = 0.f;
    for (int k = 0; k < C / 32; ++k) { float d = ptx_sub_f32(v[k], mean); q = ptx_fma_f32(d, d, q); }
    q = ptx_warp_sum_f32(q);
    float inv = ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(q, (float)C), 1e-5f));
    for (int k = 0; k < C / 32; ++k) {
      int c = lane + 32 * k;
      ptx_st_shared_u16(Xs + r * LDX + c, ptx_f32_to_f16_raw(ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(v[k], mean), inv), ptx_ldg_f32(cnScale + c), ptx_ldg_f32(cnOffset + c))));
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
    if constexpr (BIAS) { float2 bf = ptx_ldg_v2_f32(bias + c); ob0 = bf.x; ob1 = bf.y; }
    if (p0 != SIZE_MAX) {
      float2 gt = ptx_f16x2_to_f32x2(ptx_ld_global_u32(t2 + r0 * C + c));
      float2 v = pairLd2<PT>(pair, p0 * C + c);
      v.x = ptx_fma_f32(ptx_add_f32(acc[et][0], ob0), sigmH(gt.x), v.x);
      v.y = ptx_fma_f32(ptx_add_f32(acc[et][1], ob1), sigmH(gt.y), v.y);
      pairSt2<PT>(pair, p0 * C + c, v);
    }
    if (p1 != SIZE_MAX) {
      float2 gt = ptx_f16x2_to_f32x2(ptx_ld_global_u32(t2 + r1 * C + c));
      float2 v = pairLd2<PT>(pair, p1 * C + c);
      v.x = ptx_fma_f32(ptx_add_f32(acc[et][2], ob0), sigmH(gt.x), v.x);
      v.y = ptx_fma_f32(ptx_add_f32(acc[et][3], ob1), sigmH(gt.y), v.y);
      pairSt2<PT>(pair, p1 * C + c, v);
    }
  }
}

// triOutK as a PERSISTENT kernel: each block keeps the output projection in shared memory for
// every tile it takes and prefetches its next product tile while it computes the current one
// (the one-shot kernel loaded 32 KB of weights and its tile, then computed, at two blocks an SM)
template <int C, int WARPS, class TP, bool BIAS = false, class PT = float>
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
  // the center norm's scale and offset, read once: a thread's half row of them was 128 scalar loads a
  // tile from global memory, four times the epilogue's (ncu: lg_throttle the kernel's top stall)
  float* Ns = (float*)(Pst + 2 * C * LDP);                           // [2][C]
  for (int c = threadIdx.x; c < C; c += NTH) {
    ptx_st_shared_f32(Ns + c, ptx_ldg_f32(cnScale + c));
    ptx_st_shared_f32(Ns + C + c, ptx_ldg_f32(cnOffset + c));
  }
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const size_t tiles = (pp + R - 1) / R;
  {
    const int k0 = (int)threadIdx.x / (C / 8), c0 = ((int)threadIdx.x % (C / 8)) * 8;
#pragma unroll
    for (int it = 0; it < (C * (C / 8)) / NTH; ++it) {
      int k = k0 + it * (NTH / (C / 8));
      cpAsync16(Ws + k * LDW + c0, Wout + (size_t)k * C + c0, true);
    }
  }
  const int c0_p = (int)threadIdx.x / (R / PV), r0_p = ((int)threadIdx.x % (R / PV)) * PV;
  auto issue = [&](size_t tile, int st) {
    TP* Ps = Pst + st * C * LDP;
    size_t row = tile * R + r0_p;
    bool ok = row < cs;
    const TP* srcBase = prod + (ok ? row : 0);
#pragma unroll
    for (int it = 0; it < (C * (R / PV)) / NTH; ++it) {
      int c = c0_p + it * (NTH / (R / PV));
      cpAsync16(Ps + c * LDP + r0_p, srcBase + (size_t)c * cs, ok);
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
    auto pairOf = [&](size_t q) -> unsigned {
      if (q >= pp) return ~0u;
      unsigned u = (unsigned)q, i = u / (unsigned)np, j = u - i * (unsigned)np;
      return i < (unsigned)n && j < (unsigned)n ? i * (unsigned)n + j : ~0u;
    };
    // in the epilogue's layout (below): half hh of the columns, piece i - row i * 2 + lane / 16 of the
    // warp's 16, columns hh * 64 + (lane % 16) * 4 .. + 4 - so a load is 16 bytes of pair and 8 of gate,
    // whole rows, where a fragment's layout was 8 and 4 bytes in each of 8 rows (ncu: lg_throttle 35%)
    constexpr int EP = 16 * (C / 2) / 4 / 32;                       // pieces a lane, a half
    static_assert(C == 128 && EP == 8, "the epilogue is laid out for C 128");
    unsigned pe[EP]; PairRaw4<PT> pv[2][EP]; uint2 gv[2][EP];
#pragma unroll
    for (int i = 0; i < EP; ++i) {
      size_t rr = tile * R + warp * 16 + i * 2 + (lane >> 4);
      pe[i] = pairOf(rr);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        int c = hh * (C / 2) + (lane & 15) * 4;
        if (pe[i] != ~0u) { pv[hh][i] = pairLdRaw4<PT>(pair, (size_t)pe[i] * C + c); gv[hh][i] = ptx_ld_global_v2_u32(t2 + rr * C + c); }
      }
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
      for (int c = cb; c < cb + C / 2; ++c) s = ptx_add_f32(s, toF(Ps[c * LDP + r]));
      s = ptx_add_f32(s, ptx_shfl_xor_f32(s, 1));
      float mean = ptx_div_f32(s, (float)C), q = 0.f;
#pragma unroll 8
      for (int c = cb; c < cb + C / 2; ++c) { float d = ptx_sub_f32(toF(Ps[c * LDP + r]), mean); q = ptx_fma_f32(d, d, q); }
      q = ptx_add_f32(q, ptx_shfl_xor_f32(q, 1));
      float inv = ptx_rsqrt_f32(ptx_add_f32(ptx_div_f32(q, (float)C), 1e-5f));
#pragma unroll
      for (int c0 = cb; c0 < cb + C / 2; c0 += 8) {
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int c = c0 + 2 * e;
          float2 ns = ptx_ld_shared_v2_f32(Ns + c), no = ptx_ld_shared_v2_f32(Ns + C + c);
          w[e] = ptx_cvt_f16x2_f32(ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(toF(Ps[c * LDP + r]), mean), inv), ns.x, no.x),
                                   ptx_fma_f32(ptx_mul_f32(ptx_sub_f32(toF(Ps[(c + 1) * LDP + r]), mean), inv), ns.y, no.y));
        }
        ptx_st_shared_v4_u32(Xs + r * LDX + c0, make_uint4(w[0], w[1], w[2], w[3]));
      }
    }
    __syncthreads();
    float acc[NT][4] = {};
    const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
    uint32_t xa[2][4], f[2][4];
    ldsm4(xa[0], Xs + (warp * 16 + (lane & 15)) * LDX + cl);
    ldsm4t(f[0], Ws + kl * LDW + cl);
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int xcur = ks & 1;
#pragma unroll
      for (int et = 0; et < NT; et += 2) {
        const int cur = (et >> 1) & 1;
        if (et + 2 < NT) ldsm4t(f[cur ^ 1], Ws + (ks * 16 + kl) * LDW + (et + 2) * 8 + cl);
        else if (ks + 1 < KS) {
          ldsm4(xa[xcur ^ 1], Xs + (warp * 16 + (lane & 15)) * LDX + (ks + 1) * 16 + cl);
          ldsm4t(f[0], Ws + ((ks + 1) * 16 + kl) * LDW + cl);
        }
        mma16816(acc[et], xa[xcur], f[cur][0], f[cur][1]);
        mma16816(acc[et + 1], xa[xcur], f[cur][2], f[cur][3]);
      }
    }
    // the accumulators through the warp's own Xs rows (dead since its fragments were loaded; the next
    // tile's norm writes them after the loop's last barrier), 64 f32 columns at a time - 16 rows of 68
    // floats is exactly those rows' 4352 bytes - then written back as whole-row pieces
    float* Ys = reinterpret_cast<float*>(Xs + warp * 16 * LDX);
    constexpr int LDY = C / 2 + 4;
    static_assert(16 * LDY * 4 <= 16 * LDX * 2, "a half of the warp's accumulators fits in its rows");
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
      for (int e8 = 0; e8 < NT / 2; ++e8) {
        int et = hh * (NT / 2) + e8, c = e8 * 8 + tig * 2;
        ptx_st_shared_v2_f32(Ys + g * LDY + c, acc[et][0], acc[et][1]);
        ptx_st_shared_v2_f32(Ys + (g + 8) * LDY + c, acc[et][2], acc[et][3]);
      }
      __syncwarp();
#pragma unroll
      for (int i = 0; i < EP; ++i) {
        if (pe[i] == ~0u) continue;
        int rl = i * 2 + (lane >> 4), cl = (lane & 15) * 4, c = hh * (C / 2) + cl;
        float4 a4 = ptx_ld_shared_v4_f32(Ys + rl * LDY + cl), v = pairUnpack4<PT>(pv[hh][i]);
        float2 g01 = ptx_f16x2_to_f32x2(gv[hh][i].x);
        float2 g23 = ptx_f16x2_to_f32x2(gv[hh][i].y);
        if constexpr (BIAS) {
          float4 b4 = ptx_ldg_v4_f32(bias + c);
          a4.x = ptx_add_f32(a4.x, b4.x); a4.y = ptx_add_f32(a4.y, b4.y); a4.z = ptx_add_f32(a4.z, b4.z); a4.w = ptx_add_f32(a4.w, b4.w);
        }
        v.x = ptx_fma_f32(a4.x, sigmH(g01.x), v.x); v.y = ptx_fma_f32(a4.y, sigmH(g01.y), v.y);
        v.z = ptx_fma_f32(a4.z, sigmH(g23.x), v.z); v.w = ptx_fma_f32(a4.w, sigmH(g23.y), v.w);
        pairSt4<PT>(pair, (size_t)pe[i] * C + c, v);
      }
      __syncwarp();
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
  WITH_PAIR_T(
    static bool attr = false;
    if (!attr) { smemAttr((triInK<C, WARPS, TA, BIAS, PT>), (int)smem); attr = true; }
    triInK<C, WARPS, TA, BIAS, PT><<<(unsigned)((pp + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
      pair, mask, lnScale, lnOffset, Wpg, Wg, a, b, t2, n, np, cs, bias));
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
  return (size_t)128 * 136 * 2 + (size_t)16 * warps * 136 * 2 + (size_t)2 * 128 * (16 * warps + 16 / sizeof(TP)) * sizeof(TP)
         + 2 * 128 * sizeof(float);
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
    size_t smem = triOutPSmem<TP>(WARPS);
    size_t tiles = (pp + R - 1) / R;
    WITH_PAIR_T(
      static int grid = 0;
      if (!grid) {
        smemAttr((triOutPK<C, WARPS, TP, BIAS, PT>), (int)smem);
        int perSm = 0, sms = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, triOutPK<C, WARPS, TP, BIAS, PT>, 32 * WARPS, smem));
        CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        grid = std::max(1, perSm) * sms;
      }
      triOutPK<C, WARPS, TP, BIAS, PT><<<(unsigned)std::min<size_t>(grid, tiles), 32 * WARPS, smem, STREAM>>>(
        prod, cnScale, cnOffset, Wout, t2, pair, n, np, cs, bias));
    return;
  }
  size_t smem = (size_t)C * (C + 8) * 2 + (size_t)R * (C + 8) * 2 + (size_t)C * (R + 16 / sizeof(TP)) * sizeof(TP);
  WITH_PAIR_T(
    static bool attr = false;
    if (!attr) { smemAttr((triOutK<C, WARPS, TP, BIAS, PT>), (int)smem); attr = true; }
    triOutK<C, WARPS, TP, BIAS, PT><<<(unsigned)((pp + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
      prod, cnScale, cnOffset, Wout, t2, pair, n, np, cs, bias));
}
template <class TP, bool BIAS = false>
void triOutRaw(const TP* prod, const float* cnScale, const float* cnOffset, const half* Wout, const float* bias,
               const half* t2, float* pair, int n, int np, size_t cs) {
  if (warpsFor((size_t)np * np, {TO_WARPS, 4}) == TO_WARPS &&
      fitsSmem(std::min(triOutPSmem<TP>(TO_WARPS), triOutSmem<TP>(TO_WARPS))))
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
constexpr size_t lnHeadsSmem(int warps, int C = 128, int N = 16) {
  return (size_t)16 * warps * (C + 8) * 2 + (size_t)C * (N + 8) * 2;
}
template <int C, int N, int WARPS, class PT = float>
__global__ void __launch_bounds__(WARPS * 32, fitBlocks(C <= 128 ? 4 : 3, lnHeadsSmem(WARPS, C, N))) lnHeadsK(const float* __restrict__ x, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ Wp, float* __restrict__ out, size_t rows,
    half* __restrict__ biasOut = nullptr, int L = 0, int H = 0, int stride = 0, bool tr = false, size_t r0 = 0,
    const float* __restrict__ pairMask = nullptr, const float* __restrict__ rowBias = nullptr) {
  static_assert(N % 16 == 0, "the column loop takes 16 at a time");
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, LDW = N + 8, KS = C / 16, LDO = R + 4;
  static_assert((size_t)N * LDO * 4 <= (size_t)R * LDX * 2, "Os fits in Xs");
  extern __shared__ __align__(16) unsigned char smem[];
  half* Xs = (half*)smem; half* Ws = Xs + R * LDX; float* Os = (float*)smem;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  {
    const int k0 = (int)threadIdx.x / (N / 8), c0 = ((int)threadIdx.x % (N / 8)) * 8;
#pragma unroll
    for (int it = 0; it < (C * (N / 8) + NTH - 1) / NTH; ++it) {
      int k = k0 + it * (NTH / (N / 8));
      if ((C * (N / 8)) % NTH == 0 || k < C) cpAsync16(Ws + k * LDW + c0, Wp + (size_t)k * N + c0, true);
    }
  }
  cpCommit();
  lnRowsToShared<C, R, WARPS, PT>(x, [&](int r) { size_t row = row0 + r; return row < rows ? row : SIZE_MAX; },
                              lnScale, lnOffset, Xs, LDX, warp, lane);
  int lr0 = warp * 16 + g, lr1 = lr0 + 8;
  float rb0 = (rowBias && row0 + lr0 < rows) ? ptx_ldg_f32(rowBias + row0 + lr0) : 0.f;
  float rb1 = (rowBias && row0 + lr1 < rows) ? ptx_ldg_f32(rowBias + row0 + lr1) : 0.f;
  cpWait<0>();
  __syncthreads();
  float acc[N / 8][4] = {};
  const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
  uint32_t xa[2][4], f[2][4];
  ldsm4(xa[0], Xs + (warp * 16 + (lane & 15)) * LDX + cl);
  ldsm4t(f[0], Ws + kl * LDW + cl);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const int xcur = ks & 1;
#pragma unroll
    for (int n2 = 0; n2 < N / 16; ++n2) {
      const int cur = (ks * (N / 16) + n2) & 1;
      if (n2 + 1 < N / 16) ldsm4t(f[cur ^ 1], Ws + (ks * 16 + kl) * LDW + (n2 + 1) * 16 + cl);
      else if (ks + 1 < KS) {
        ldsm4(xa[xcur ^ 1], Xs + (warp * 16 + (lane & 15)) * LDX + (ks + 1) * 16 + cl);
        ldsm4t(f[cur ^ 1], Ws + ((ks + 1) * 16 + kl) * LDW + cl);
      }
      mma16816(acc[2 * n2], xa[xcur], f[cur][0], f[cur][1]); mma16816(acc[2 * n2 + 1], xa[xcur], f[cur][2], f[cur][3]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int nt = 0; nt < N / 8; ++nt) {
    int h = nt * 8 + tig * 2;
    ptx_st_shared_f32(Os + h * LDO + lr0, ptx_add_f32(acc[nt][0], rb0));
    ptx_st_shared_f32(Os + (h + 1) * LDO + lr0, ptx_add_f32(acc[nt][1], rb0));
    ptx_st_shared_f32(Os + h * LDO + lr1, ptx_add_f32(acc[nt][2], rb1));
    ptx_st_shared_f32(Os + (h + 1) * LDO + lr1, ptx_add_f32(acc[nt][3], rb1));
  }
  __syncthreads();
  if (biasOut) {
    const int r = (int)threadIdx.x & (R - 1), h0 = (int)threadIdx.x / R;
    if (row0 + r >= rows) return;
    unsigned ij = (unsigned)(r0 + row0 + r), ul = (unsigned)L, i = ij / ul, j = ij - i * ul;
    const float pm = pairMask ? ptx_mul_f32(1e9f, ptx_sub_f32(ptx_ldg_f32(pairMask + ij), 1.f)) : 0.f;
    if (tr) { unsigned x = i; i = j; j = x; }
    for (int h = h0; h < H; h += NTH / R) {
      float v = ptx_add_f32(ptx_ld_shared_f32(Os + h * LDO + r), pm);
      ptx_st_global_u16(biasOut + ((size_t)h * L + i) * stride + j, ptx_f32_to_f16_raw(ptx_max_f32(ptx_mul_f32(v, LOG2E), -6e4f)));
      if (j == ul - 1) for (int pad = L; pad < stride; ++pad) ptx_st_global_u16(biasOut + ((size_t)h * L + i) * stride + pad, 0);
    }
    return;
  }
  const int h0 = (int)threadIdx.x / R, r = (int)threadIdx.x % R;
  if (row0 + r < rows) {
#pragma unroll
    for (int it = 0; it < (N * R + NTH - 1) / NTH; ++it) {
      int h = h0 + it * (NTH / R);
      if ((N * R) % NTH == 0 || h < N) ptx_st_global_f32(out + (size_t)h * rows + row0 + r, ptx_ld_shared_f32(Os + h * LDO + r));
    }
  }
}
template <int N, int C = 128>
inline void lnHeadsRaw(const float* x, const float* scale, const float* offset, const half* w, float* out, size_t rows,
                       half* biasOut = nullptr, int L = 0, int H = 0, int stride = 0, bool tr = false, size_t r0 = 0,
                       const float* pairMask = nullptr, const float* rowBias = nullptr) {
  constexpr int WARPS = 4, R = 16 * WARPS;      // (raw pointers: AF2's pair-bias pass, 23.0 KB smem -> 4 blocks/SM on sm_86)
  size_t smem = lnHeadsSmem(WARPS, C, N);
  WITH_PAIR_T(
    static bool attr = false;
    if (!attr) { smemAttr((lnHeadsK<C, N, WARPS, PT>), (int)smem); attr = true; }
    lnHeadsK<C, N, WARPS, PT><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
      x, scale, offset, w, out, rows, biasOut, L, H, stride, tr, r0, pairMask, rowBias));
}
template <int N, int C = 128>
inline void lnHeads128(const float* x, const std::string& scale, const std::string& offset, const std::string& w,
                       float* out, size_t rows, const float* rowBias = nullptr) {
  lnHeadsRaw<N, C>(x, W(scale), W(offset), Wh(w), out, rows, nullptr, 0, 0, 0, false, 0, nullptr, rowBias);
}

// The grid attention's input projection: LN(pair row) -> q, k, v, gate (C x NQ), for output rows
// q0 .. q0+rows of the attention's (row, position) order; in the column direction (tr) output row
// (r, j) reads pair row (j, r), so the transposed copy the unfused form made is never written.
// With `bias` (only when one call covers every row), it also writes the pair bias - LN(pair) times
// the (C, 16) zero-padded bias projection, heads < 16 - straight into the flash kernel's
// [h][i][stride] f16 layout, scaled by log2(e), (i, j) swapped where `swap`.
// a key's place in flash8's transposed V (fp8VtK's order within each 64-key group)
__host__ __device__ inline int vtPerm64(int lk) {
  int g = lk & 31;
  return (lk & 32) + (g & 16) + ((g & 15) < 8 ? ((g & 7) >> 1) * 4 + (g & 1) : ((g & 7) >> 1) * 4 + 2 + (g & 1));
}
// gridInK's dynamic shared memory (its launchers', and its own __launch_bounds__')
constexpr size_t gridInSmem(int warps, int nc = 64, int C = 128) {
  size_t xr = (size_t)16 * warps * (C + 8) * 2, st = (size_t)C * (nc + 8) * 2, ys = (size_t)16 * warps * (nc + 8) * 2;
  return st + (xr > st + ys ? xr : st + ys);
}
template <int C, int NQ, int WARPS, class PT = float, int NCI = 64, int MT = 1,
          int MINB = (WARPS == 4 && (NCI == 64 && MT == 1 || NCI == 32 && MT == 2 || NCI == 64 && MT == 2 && C == 64) && (C == 128 || C == 64)) ? 3
                   : (WARPS == 8 && NCI == 64 && MT == 1 && (C == 128 || C == 64)) ? 2 : 1>
__global__ void __launch_bounds__(WARPS * 32, fitBlocks(MINB, gridInSmem(WARPS, NCI, C))) gridInK(const float* __restrict__ pair, const float* __restrict__ lnScale,
    const float* __restrict__ lnOffset, const half* __restrict__ Wq, half* __restrict__ out, int n, size_t q0,
    size_t rows, bool tr, const half* __restrict__ Wb, half* __restrict__ bias, int heads, int stride, bool swap,
    const float* __restrict__ qb = nullptr,      // qb: a bias on the NQ projected columns (AF2's gate bias), or none
    uint8_t* __restrict__ kv8 = nullptr,         // kv8: k and v written e4m3 here instead (flash8's K8, then V8, rows x NQ/4 each)
    uint8_t* __restrict__ vt8 = nullptr) {       // vt8 (with kv8): v instead straight into flash8's transposed form - fp8VtK's,
                                                 // [grid row][dim][NK], each 64-key group in vtPerm64's order, its padding zero
  constexpr int R = 16 * WARPS * MT, XR = 16 * WARPS, NTH = 32 * WARPS, LDX = C + 8, NC = NCI, LDW = NC + 8, KS = C / 16;   // NC: weight columns a stage
  extern __shared__ __align__(16) unsigned char smem[];
  // stage 0 of Wst sits at smem[0 .. C*LDW); Xs sits right after stage 0 during the prologue (aliasing stage 1 of Wst + YsBase,
  // which are only written after every warp has loaded its xa fragments from Xs), shrinking 4-warp NC=32 smem to 27.0 KB (2 blocks/SM on 64 KB T4)
  half* Wst = (half*)smem;                                         // two stages of [C][LDW]
  half* Xs = Wst + C * LDW;                                        // [XR][LDX] in prologue
  half* YsBase = Wst + 2 * C * LDW;                                // [WARPS * 16][LDW] in chunk loop
  half* Wbs = Wst;                                                 // [C][24], the bias projection (reuses Wst after chunks)
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  constexpr int ITER = (C * (NC / 8) + NTH - 1) / NTH;
  const int k0 = (int)threadIdx.x / (NC / 8), c0 = ((int)threadIdx.x % (NC / 8)) * 8;
  const int wOff0 = k0 * LDW + c0;
  const half* wq0 = Wq + (size_t)k0 * NQ + c0;
  auto stage = [&](int j, int st, auto&& op) {
    half* w = Wst + st * C * LDW + wOff0;
    const half* s = wq0 + j * NC;
#pragma unroll
    for (int it = 0; it < ITER; ++it) {
      if ((C * (NC / 8)) % NTH != 0 && it * NTH + (int)threadIdx.x >= C * (NC / 8)) break;
      op(it, w + it * (NTH / (NC / 8)) * LDW, s + (size_t)it * (NTH / (NC / 8)) * NQ);
    }
  };
  auto issue = [&](int j, int st) { stage(j, st, [](int, half* d, const half* s) { cpAsync16(d, s, true); }); cpCommit(); };
#if LF_REG_STAGES
  RegStage<ITER> next;                // (sm_75: the next chunk's weights held in registers across this chunk)
#endif
  issue(0, 0);
  uint32_t xa[MT][KS][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    if (mt > 0) __syncthreads();
    lnRowsToShared<C, XR, WARPS, PT>(pair, [&](int r) {
        size_t q = row0 + (MT == 1 ? r : ((r >> 4) * MT + mt) * 16 + (r & 15));
        if (q >= rows) return (size_t)SIZE_MAX;
        size_t Q = q0 + q;
        if (!tr) return Q;
        unsigned u = (unsigned)Q, a = u / (unsigned)n;           // 32-bit: a 64-bit divide is ~70 instructions
        return (size_t)(u - a * (unsigned)n) * n + a;
      }, lnScale, lnOffset, Xs, LDX, warp, lane);
    if (mt + 1 == MT) {
#if !LF_REG_STAGES
      cpWait<0>();
#endif
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) ldsm4(xa[mt][ks], Xs + (warp * 16 + (lane & 15)) * LDX + ks * 16 + (lane >> 4) * 8);
  }
  __syncthreads();
  const int chunks = NQ / NC;
#if !LF_REG_STAGES
  if (1 < chunks) issue(1, 1);
  else if (bias) {
    const int kb0 = (int)threadIdx.x >> 1, cb0 = ((int)threadIdx.x & 1) * 8;
#pragma unroll
    for (int it = 0; it < (C * 2 + NTH - 1) / NTH; ++it) {
      int k = kb0 + it * (NTH / 2);
      if ((C * 2) % NTH == 0 || k < C) cpAsync16(Wbs + k * 24 + cb0, Wb + (size_t)k * 16 + cb0, true);
    }
    cpCommit();
  }
#endif
  const int r_base = lane / (NC / 8), c_out = (lane % (NC / 8)) * 8;
  for (int j = 0; j < chunks; ++j) {
    int st = j & 1;
#if LF_REG_STAGES
    if (j + 1 < chunks) stage(j + 1, st ^ 1, [&](int i, half*, const half* src) { next.load(i, src); });
#endif
    // the bias read BEFORE the MMAs, so its global latency hides behind them: read after, the adds waiting on it were a
    // fifth of the kernel's stall samples (ncu, AF2 at 600 residues); the same adds, so byte-identical
    float2 qbv[NC / 8];
    if (qb && j * NC >= 3 * (NQ / 4)) {
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) qbv[nt] = ptx_ldg_v2_f32(qb + j * NC + nt * 8 + tig * 2);
    }
#if LF_REG_STAGES
    __syncthreads();
#else
    if (j > 0) {
      cpWait<0>();
      __syncthreads();
      if (j + 1 < chunks) issue(j + 1, st ^ 1);
      else if (bias) {
        const int kb0 = (int)threadIdx.x >> 1, cb0 = ((int)threadIdx.x & 1) * 8;
#pragma unroll
        for (int it = 0; it < (C * 2 + NTH - 1) / NTH; ++it) {
          int k = kb0 + it * (NTH / 2);
          if ((C * 2) % NTH == 0 || k < C) cpAsync16(Wbs + k * 24 + cb0, Wb + (size_t)k * 16 + cb0, true);
        }
        cpCommit();
      }
    }
#endif
    const half* w = Wst + st * C * LDW;
    float acc[MT][NC / 8][4] = {};
    const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
    uint32_t f[2][4];
    ldsm4t(f[0], w + kl * LDW + cl);
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < NC / 16; ++n2) {
        const int cur = (ks * (NC / 16) + n2) & 1;
        if (n2 + 1 < NC / 16) ldsm4t(f[cur ^ 1], w + (ks * 16 + kl) * LDW + (n2 + 1) * 16 + cl);
        else if (ks + 1 < KS) ldsm4t(f[cur ^ 1], w + ((ks + 1) * 16 + kl) * LDW + cl);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816(acc[mt][2 * n2], xa[mt][ks], f[cur][0], f[cur][1]);
          mma16816(acc[mt][2 * n2 + 1], xa[mt][ks], f[cur][2], f[cur][3]);
        }
      }
    }
    if (qb && j * NC >= 3 * (NQ / 4)) {
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        float b0 = qbv[nt].x, b1 = qbv[nt].y;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          acc[mt][nt][0] = ptx_add_f32(acc[mt][nt][0], b0); acc[mt][nt][1] = ptx_add_f32(acc[mt][nt][1], b1);
          acc[mt][nt][2] = ptx_add_f32(acc[mt][nt][2], b0); acc[mt][nt][3] = ptx_add_f32(acc[mt][nt][3], b1);
        }
      }
    }
    // through the warp's own rows of YsBase (16 x LDW per warp) so a
    // store is 16 bytes a lane, whole 128-byte rows, rather than 4 bytes in each of 8 rows: 781 -> 595 ms
    // over a 1,044-token fold's 424 calls, 59.1 -> 44.8 at 261, the fold byte-identical
    half* Ys = YsBase + warp * 16 * LDW;
    constexpr int WQ = NQ / 4;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (mt > 0) __syncwarp();
#pragma unroll
      for (int nt = 0; nt < NC / 8; ++nt) {
        int c = nt * 8 + tig * 2;
        ptx_st_shared_u32(Ys + g * LDW + c, ptx_cvt_f16x2_f32(acc[mt][nt][0], acc[mt][nt][1]));
        ptx_st_shared_u32(Ys + (g + 8) * LDW + c, ptx_cvt_f16x2_f32(acc[mt][nt][2], acc[mt][nt][3]));
      }
      __syncwarp();
      size_t tileRow0 = row0 + (warp * MT + mt) * 16;
      if (kv8 && j * NC >= WQ && j * NC < 3 * WQ) {        // k or v: e4m3, 8 bytes a lane
        const bool isV = j * NC >= 2 * WQ;
        uint8_t* dst = kv8 + (isV ? rows * WQ : (size_t)0) + (j * NC) % WQ;
        auto e2 = [](uint32_t h) -> uint32_t {
          __half2_raw r; r.x = (unsigned short)(h & 0xFFFF); r.y = (unsigned short)(h >> 16);
          return __nv_cvt_halfraw2_to_fp8x2(r, __NV_SATFINITE, __NV_E4M3);
        };
#pragma unroll
        for (int i = 0; i < 16 * (NC / 8) / 32; ++i) {
          int r = i * (32 / (NC / 8)) + lane / (NC / 8), c = (lane % (NC / 8)) * 8;
          size_t row = tileRow0 + r;
          if (row < rows) {
            uint4 h = ptx_ld_shared_v4_u32(Ys + r * LDW + c);
            uint2 v8 = make_uint2(e2(h.x) | e2(h.y) << 16, e2(h.z) | e2(h.w) << 16);
            if (isV && vt8) {
              // byte by byte into [grid row][dim][NK], a grid row's last key also zeroing its group's padding
              const unsigned nk = ((unsigned)n + 63u) & ~63u, gr = (unsigned)row / (unsigned)n, pos = (unsigned)row - gr * (unsigned)n;
              uint8_t* col = vt8 + ((size_t)gr * WQ + (j * NC) % WQ + c) * nk + (pos & ~63u);
#pragma unroll
              for (int e = 0; e < 8; ++e)
                col[(size_t)e * nk + vtPerm64(pos & 63u)] = (uint8_t)((e < 4 ? v8.x >> (8 * e) : v8.y >> (8 * (e - 4))) & 0xFFu);
              if (pos == (unsigned)n - 1)
                for (unsigned lk = (pos & 63u) + 1; lk < 64 && (pos & ~63u) + lk < nk; ++lk)
#pragma unroll
                  for (int e = 0; e < 8; ++e) col[(size_t)e * nk + vtPerm64(lk)] = 0;
            } else *reinterpret_cast<uint2*>(dst + row * WQ + c) = v8;
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16 * (NC / 8) / 32; ++i) {
          int r = i * (32 / (NC / 8)) + r_base;
          size_t row = tileRow0 + r;
          if (row < rows) ptx_st_global_v4_u32(out + row * NQ + j * NC + c_out, ptx_ld_shared_v4_u32(Ys + r * LDW + c_out));
        }
      }
    }
    __syncwarp();
#if LF_REG_STAGES
    if (j + 1 < chunks) stage(j + 1, st ^ 1, [&](int i, half* d, const half*) { next.store(i, d); });
    __syncthreads();
#endif
  }
  if (bias) {
#if LF_REG_STAGES
    for (int t = threadIdx.x; t < C * 2; t += NTH) {
      int k = t / 2, c = (t % 2) * 8;
      cpAsync16(Wbs + k * 24 + c, Wb + (size_t)k * 16 + c, true);
    }
    cpCommit();
#endif
    cpWait<0>();
    __syncthreads();
    float acc[MT][2][4] = {};
    const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
    uint32_t f[2][4];
    ldsm4t(f[0], Wbs + kl * 24 + cl);
    if (heads <= 8) {
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        const int cur = ks & 1;
        if (ks + 1 < KS) ldsm4t(f[cur ^ 1], Wbs + ((ks + 1) * 16 + kl) * 24 + cl);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma16816(acc[mt][0], xa[mt][ks], f[cur][0], f[cur][1]);
      }
    } else {
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        const int cur = ks & 1;
        if (ks + 1 < KS) ldsm4t(f[cur ^ 1], Wbs + ((ks + 1) * 16 + kl) * 24 + cl);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816(acc[mt][0], xa[mt][ks], f[cur][0], f[cur][1]);
          mma16816(acc[mt][1], xa[mt][ks], f[cur][2], f[cur][3]);
        }
      }
    }
    if (swap == tr) {
      half* Ys = YsBase + warp * 16 * 20;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        if (mt > 0) __syncwarp();
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          if (nt == 1 && heads <= 8) break;
          int h = nt * 8 + tig * 2;
          ptx_st_shared_u16(Ys + h * 20 + g, ptx_f32_to_f16_raw(ptx_mul_f32(LOG2E, acc[mt][nt][0])));
          ptx_st_shared_u16(Ys + (h + 1) * 20 + g, ptx_f32_to_f16_raw(ptx_mul_f32(LOG2E, acc[mt][nt][1])));
          ptx_st_shared_u16(Ys + h * 20 + g + 8, ptx_f32_to_f16_raw(ptx_mul_f32(LOG2E, acc[mt][nt][2])));
          ptx_st_shared_u16(Ys + (h + 1) * 20 + g + 8, ptx_f32_to_f16_raw(ptx_mul_f32(LOG2E, acc[mt][nt][3])));
        }
        __syncwarp();
        int r = lane & 15;
        size_t q = row0 + (warp * MT + mt) * 16 + r;
        if (q < rows) {
          unsigned Q = (unsigned)(q0 + q), qa = Q / (unsigned)n, qb = Q - qa * (unsigned)n;
          size_t base = (size_t)qa * stride + qb, hStep = (size_t)n * stride;
          for (int h = lane >> 4; h < heads; h += 2) ptx_st_global_u16(bias + h * hStep + base, ptx_ld_shared_u16(Ys + h * 20 + r));
        }
      }
    } else {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        size_t r0 = row0 + (warp * MT + mt) * 16 + g;
#pragma unroll
        for (int half8 = 0; half8 < 2; ++half8) {
          size_t q = r0 + half8 * 8;
          if (q >= rows) continue;
          unsigned Q = (unsigned)(q0 + q), qa = Q / (unsigned)n, qb = Q - qa * (unsigned)n;
          unsigned a = tr ? qb : qa, b = tr ? qa : qb;               // p = (a, b)
          size_t i = swap ? b : a, jj = swap ? a : b;
#pragma unroll
          for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              int h = nt * 8 + tig * 2 + e;
              if (h < heads) ptx_st_global_u16(bias + ((size_t)h * n + i) * stride + jj, ptx_f32_to_f16_raw(ptx_mul_f32(LOG2E, acc[mt][nt][half8 * 2 + e])));
            }
        }
      }
    }
  }
}
// pair[(j, r)] += (gathered[(r, j)] Wout) for output rows q0 .. q0+rows (the column direction)
// WQ 4 (a T4's): the weight streamed through one buffer a quarter of K at a time (32 rows, 9 KB) where it was held whole (35 KB) -
// the block 26 KB, two an SM where the 52 KB form held one (4 warps); each quarter's copy behind a barrier, the second block
// hiding it
constexpr size_t gridOutSmem(int warps, int C = 128, int wq = 1) {
  const int ldw = C == 128 ? 128 : C + 8, ldx = C == 128 ? 128 : C + 8;
  return (size_t)(C / wq) * ldw * 2 + (size_t)16 * warps * ldx * 2;
}
template <int C, int WD, int WARPS, class PT = float, int WQ = 1,
          int MINB = (C == 128 && WD == 128 && WARPS == 4 && WQ == 1) ? 3 : (C <= 128 && WARPS == 4 && WQ == 4) ? 2 : 1>
#if LF_REG_STAGES
__global__ void __launch_bounds__(WARPS * 32, (C <= 128 && WARPS == 4 && WQ == 4) ? 2 : 1)
#else
__global__ void __launch_bounds__(WARPS * 32, fitBlocks(MINB, gridOutSmem(WARPS, C, WQ)))
#endif
gridOutK(const half* __restrict__ gathered, const half* __restrict__ Wout,
    float* __restrict__ pair, int n, size_t q0, size_t rows, bool tr, const float* __restrict__ ob = nullptr) {
  // ob: the output projection's bias (AF2's and RoseTTAFold3's), or none
  constexpr int R = 16 * WARPS, NTH = 32 * WARPS, LDX = (WD == 128 && C == 128) ? 128 : WD + 8, LDW = (C == 128 && WD == 128) ? 128 : C + 8, KS = WD / 16, NT = C / 8;
  auto swW = [](int k, int c) { if constexpr (C == 128 && WD == 128) return k * LDW + (c ^ ((k & 7) << 3)); else return k * LDW + c; };
  auto swX = [](int r, int c) { if constexpr (C == 128 && WD == 128) return r * LDX + (c ^ ((r & 7) << 3)); else return r * LDX + c; };
  extern __shared__ __align__(16) unsigned char smem[];
  half* Ws = (half*)smem; half* Xs = Ws + WD / WQ * LDW;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  size_t row0 = (size_t)blockIdx.x * R;
  // (braced: CUDA 12.2 compiled this `if constexpr` without braces as though the X loop below were inside it too, so
  // at WQ 4 the X tile was never loaded - every output its residual, pTM NaN - where CUDA 13 compiles it as written)
  if constexpr (WQ == 1) {
    const int k0 = (int)threadIdx.x / (C / 8), c0 = ((int)threadIdx.x % (C / 8)) * 8;
#pragma unroll
    for (int it = 0; it < WD * (C / 8) / NTH; ++it) {
      int k = k0 + it * (NTH / (C / 8));
      cpAsync16(Ws + swW(k, c0), Wout + (size_t)k * C + c0, true);
    }
  }
  {
    const int r0_ = (int)threadIdx.x / (WD / 8), c0 = ((int)threadIdx.x % (WD / 8)) * 8;
    const bool fullRows = (row0 + R <= rows);
    if (fullRows) {
#pragma unroll
      for (int it = 0; it < R * (WD / 8) / NTH; ++it) {
        int r = r0_ + it * (NTH / (WD / 8));
        cpAsync16(Xs + swX(r, c0), gathered + (row0 + r) * WD + c0, true);
      }
    } else {
#pragma unroll
      for (int it = 0; it < R * (WD / 8) / NTH; ++it) {
        int r = r0_ + it * (NTH / (WD / 8));
        size_t q = row0 + r;
        cpAsync16(Xs + swX(r, c0), gathered + (q < rows ? q : 0) * WD + c0, q < rows);
      }
    }
  }
  cpCommit();
  // the residual's rows, read before the GEMM covers their latency, in the epilogue's layout: half hh of
  // the columns, piece i - row i * 2 + lane / 16 of the warp's 16, columns hh * 64 + (lane % 16) * 4 - so
  // the read-modify-write is 16 bytes a lane of whole rows, where a fragment's layout was 8 bytes in each
  // of 8 rows (ncu: 69% of the kernel's stall samples on those two lines)
  constexpr int EP = 16 * (C / 2) / 4 / 32, LPR = 32 / (C / 8);
  static_assert((C == 64 || C == 128) && WD == C, "the epilogue is laid out for C 64 or 128");
  const int rl0 = lane / (C / 8), cl0 = (lane % (C / 8)) * 4;
  unsigned pe[EP]; PairRaw4<PT> pv[2][EP];
#pragma unroll
  for (int i = 0; i < EP; ++i) {
    size_t q = row0 + warp * 16 + i * LPR + rl0;
    if (q < rows) {
      unsigned Q = (unsigned)(q0 + q), a = Q / (unsigned)n;           // 32-bit: a 64-bit divide is ~70 instructions
      pe[i] = tr ? (Q - a * (unsigned)n) * (unsigned)n + a : Q;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) pv[hh][i] = pairLdRaw4<PT>(pair, (size_t)pe[i] * C + hh * (C / 2) + cl0);
    } else pe[i] = ~0u;
  }
  cpWait<0>();
  __syncthreads();
  float acc[NT][4] = {};
#if LF_REG_STAGES
  // (sm_75, no cp.async: the next quarter loaded into registers before this quarter's MMAs, stored after them)
  constexpr int QPT = WQ > 1 ? WD / WQ * (C / 8) / NTH : 1;
  static_assert(WQ == 1 || WD / WQ * (C / 8) % NTH == 0, "a quarter is a whole number of pieces a thread");
  RegStage<QPT> nextW;
  auto loadQ = [&](int q) {
#pragma unroll
    for (int i = 0; i < QPT; ++i) {
      int t = i * NTH + threadIdx.x, k = t / (C / 8), c = (t % (C / 8)) * 8;
      nextW.load(i, Wout + (size_t)(q * (WD / WQ) + k) * C + c);
    }
  };
  auto storeQ = [&] {
#pragma unroll
    for (int i = 0; i < QPT; ++i) {
      int t = i * NTH + threadIdx.x, k = t / (C / 8), c = (t % (C / 8)) * 8;
      nextW.store(i, Ws + swW(k, c));
    }
  };
  if constexpr (WQ > 1) loadQ(0);
#endif
#pragma unroll
  for (int q = 0; q < WQ; ++q) {
    if constexpr (WQ > 1) {        // this quarter's weight rows into the one buffer, every warp done with the last quarter's
      if (q) __syncthreads();
#if LF_REG_STAGES
      storeQ();
      __syncthreads();
      if (q + 1 < WQ) loadQ(q + 1);
#else
      for (int t = threadIdx.x; t < WD / WQ * (C / 8); t += NTH) {
        int k = t / (C / 8), c = (t % (C / 8)) * 8;
        cpAsync16(Ws + swW(k, c), Wout + (size_t)(q * (WD / WQ) + k) * C + c, true);
      }
      cpCommit(); cpWait<0>();
      __syncthreads();
#endif
    }
    const int kl = ((lane >> 3) & 1) * 8 + (lane & 7), cl = (lane >> 4) * 8;
    uint32_t xa[2][4], f[2][4];
    ldsm4(xa[0], Xs + swX(warp * 16 + (lane & 15), (q * (KS / WQ)) * 16 + (lane >> 4) * 8));
    ldsm4t(f[0], Ws + swW(kl, cl));
#pragma unroll
    for (int kq = 0; kq < KS / WQ; ++kq) {
      const int ks = q * (KS / WQ) + kq, xcur = kq & 1;
#pragma unroll
      for (int et = 0; et < NT; et += 2) {
        const int cur = (et >> 1) & 1;
        if (et + 2 < NT) ldsm4t(f[cur ^ 1], Ws + swW(kq * 16 + kl, (et + 2) * 8 + cl));
        else if (kq + 1 < KS / WQ) {
          ldsm4(xa[xcur ^ 1], Xs + swX(warp * 16 + (lane & 15), (ks + 1) * 16 + (lane >> 4) * 8));
          ldsm4t(f[cur ^ 1], Ws + swW((kq + 1) * 16 + kl, cl));
        }
        mma16816(acc[et], xa[xcur], f[cur][0], f[cur][1]); mma16816(acc[et + 1], xa[xcur], f[cur][2], f[cur][3]);
      }
    }
  }
  // the accumulators through shared memory, 64 f32 columns at a time
  constexpr int LDY = C / 2 + 4;
  float* Ys = reinterpret_cast<float*>(smem) + warp * 16 * LDY;
  __syncthreads();
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    float4 b4;
    if (ob) b4 = ptx_ldg_v4_f32(ob + hh * (C / 2) + cl0);
#pragma unroll
    for (int e8 = 0; e8 < NT / 2; ++e8) {
      int et = hh * (NT / 2) + e8, c = e8 * 8 + tig * 2;
      ptx_st_shared_v2_f32(Ys + g * LDY + c, acc[et][0], acc[et][1]);
      ptx_st_shared_v2_f32(Ys + (g + 8) * LDY + c, acc[et][2], acc[et][3]);
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < EP; ++i) {
      if (pe[i] == ~0u) continue;
      int rl = i * LPR + rl0;
      float4 a4 = ptx_ld_shared_v4_f32(Ys + rl * LDY + cl0), v = pairUnpack4<PT>(pv[hh][i]);
      if (ob) {
        a4.x = ptx_add_f32(a4.x, b4.x); a4.y = ptx_add_f32(a4.y, b4.y); a4.z = ptx_add_f32(a4.z, b4.z); a4.w = ptx_add_f32(a4.w, b4.w);
      }
      v.x = ptx_add_f32(v.x, a4.x); v.y = ptx_add_f32(v.y, a4.y); v.z = ptx_add_f32(v.z, a4.z); v.w = ptx_add_f32(v.w, a4.w);
      pairSt4<PT>(pair, (size_t)pe[i] * C + hh * (C / 2) + cl0, v);
    }
    __syncwarp();
  }
}
#ifndef GI_T4_NC
#define GI_T4_NC 32
#endif
#ifndef GO_QUARTERS
#define GO_QUARTERS 1
#endif
constexpr int GI_WARPS = 8, GO_WARPS = 8;
// the launchers on raw pointers (AF2 calls these with its own weights and biases), then AF3's by name
template <int WARPS, int NCI = 64, int C = 128, int MT = 1>
void gridInAt(const float* pair, const float* lnScale, const float* lnOffset, const half* Wq, const float* qb, half* out,
              int n, size_t q0, size_t rows, bool tr, const half* Wb, half* bias, int heads, int stride, bool swap,
              uint8_t* kv8 = nullptr, uint8_t* vt8 = nullptr) {
  constexpr int NQ = 4 * C, R = 16 * WARPS * MT;
  size_t smem = gridInSmem(WARPS, NCI, C);
  WITH_PAIR_T(
    static bool attr = false;
    if (!attr) { smemAttr((gridInK<C, NQ, WARPS, PT, NCI, MT>), (int)smem); attr = true; }
    gridInK<C, NQ, WARPS, PT, NCI, MT><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(
      pair, lnScale, lnOffset, Wq, out, n, q0, rows, tr, Wb, bias, heads, stride, swap, qb, kv8, vt8));
}
inline bool gridFusedFits(int C = 128) { return fitsSmem(gridInSmem(2, 64, C)) && fitsSmem(gridOutSmem(2, C)); }
template <int C = 128>
inline void gridInRaw(const float* pair, const float* lnScale, const float* lnOffset, const half* Wq, const float* qb,
                      half* out, int n, size_t q0, size_t rows, bool tr, const half* Wb = nullptr, half* bias = nullptr,
                      int heads = 0, int stride = 0, bool swap = false, uint8_t* kv8 = nullptr, bool wide = false,
                      uint8_t* vt8 = nullptr) {
  // wide (AF2 on sm_12x): 16 warps, 256 rows a block, the weight in 16-column stages to fit (88 KB) - each stage spread
  // over twice the rows: an RTX PRO 6000's 1,000-residue triangle attention 3.63 -> 3.41 ms a call (--bench-pair: 16
  // warps at 32-column stages 3.43, at the 64-column ones 3.63), bit-identical
  if (wide && (rows + 255) / 256 >= MIN_BLOCKS && fitsSmem(gridInSmem(16, 16, C))) {
    gridInAt<16, 16, C>(pair, lnScale, lnOffset, Wq, qb, out, n, q0, rows, tr, Wb, bias, heads, stride, swap, kv8, vt8);
    return;
  }
  // a T4 (64 KB an SM): 4 warps at two 16-row tiles each (MT=2, 128 rows a block) and 32-column stages, with prologue Xs
  // aliasing stage 1 of Wst + YsBase (27.0 KB a block -> 2 blocks = 8 warps = 256 rows an SM, and half the shared-memory
  // ldsm4t weight reads per row)
  static const bool gi8Fits2 = [] {
    int dev, perSm = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    size_t lim = getenv("LOCALFOLD_SMEM_LIMIT") ? smemLimit() : (size_t)perSm;
    return lim >= 2 * (gridInSmem(GI_WARPS, 64, C) + 1024);
  }();
  if (GI_T4_NC && !gi8Fits2 && (rows + 127) / 128 >= (size_t)MIN_BLOCKS && fitsSmem(gridInSmem(4, GI_T4_NC, C))) {
    gridInAt<4, GI_T4_NC, C, 2>(pair, lnScale, lnOffset, Wq, qb, out, n, q0, rows, tr, Wb, bias, heads, stride, swap, kv8, vt8);
    return;
  }
  switch (warpsFitting(warpsFor(rows, {GI_WARPS, 4, 2}), {GI_WARPS, 4, 2}, [](int w) { return gridInSmem(w, 64, C); })) {
    case 2: gridInAt<2, 64, C>(pair, lnScale, lnOffset, Wq, qb, out, n, q0, rows, tr, Wb, bias, heads, stride, swap, kv8, vt8); break;
    case 4: gridInAt<4, 64, C>(pair, lnScale, lnOffset, Wq, qb, out, n, q0, rows, tr, Wb, bias, heads, stride, swap, kv8, vt8); break;
    default: gridInAt<GI_WARPS, 64, C>(pair, lnScale, lnOffset, Wq, qb, out, n, q0, rows, tr, Wb, bias, heads, stride, swap, kv8, vt8);
  }
}
template <int C = 128>
inline void gridIn128(const float* pair, const std::string& pre, const std::string& wq, half* out, int n, size_t q0,
                      size_t rows, bool tr, const half* Wb = nullptr, half* bias = nullptr, int heads = 0,
                      int stride = 0, bool swap = false, uint8_t* kv8 = nullptr, const float* qb = nullptr,
                      bool wide = false, uint8_t* vt8 = nullptr) {
  gridInRaw<C>(pair, W(pre + ".actNormScale"), W(pre + ".actNormOffset"), Wh(wq), qb ? qb - 3 * C : nullptr, out, n, q0, rows, tr, Wb, bias,
               heads, stride, swap, kv8, wide, vt8);
}
template <int WARPS, int WQ = 1, int C = 128>
void gridOutAt(const half* gathered, const half* Wout, const float* ob, float* pair, int n, size_t q0, size_t rows, bool tr) {
  constexpr int WD = C, R = 16 * WARPS;
  size_t smem = gridOutSmem(WARPS, C, WQ);
  WITH_PAIR_T(
    static bool attr = false;
    if (!attr) { smemAttr((gridOutK<C, WD, WARPS, PT, WQ>), (int)smem); attr = true; }
    gridOutK<C, WD, WARPS, PT, WQ><<<(unsigned)((rows + R - 1) / R), 32 * WARPS, smem, STREAM>>>(gathered, Wout, pair, n, q0, rows, tr, ob));
}
template <int C = 128>
inline void gridOutRaw(const half* gathered, const half* Wout, const float* ob, float* pair, int n, size_t q0, size_t rows, bool tr) {
  // 4 warps where an SM holds three of their blocks (an A100's 164 KB): the 8-warp block's 166 registers a thread and
  // 70 KB held one block an SM, the kernel latency-bound at 41% of the memory's bandwidth - 212 -> 149 ms of a 988-token
  // fold, byte-identical. A part with ~100 KB an SM holds one block of either and keeps the eight
  static const bool four = [] {
    int dev, perSm = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    size_t lim = getenv("LOCALFOLD_SMEM_LIMIT") ? smemLimit() : (size_t)perSm;
    return lim >= 2 * (gridOutSmem(4, C) + 1024);
  }();
  // a part without cp.async (a T4): the weight a quarter at a time, two 4-warp blocks an SM (gridOutK's WQ)
  if (GO_QUARTERS && ccMajor() < 8 && (rows + 63) / 64 >= (size_t)MIN_BLOCKS && fitsSmem((size_t)(C / 4) * (C + 8) * 2 + (size_t)64 * (C + 8) * 2)) {
    gridOutAt<4, 4, C>(gathered, Wout, ob, pair, n, q0, rows, tr);
    return;
  }
  switch (four && fitsSmem(gridOutSmem(4, C)) && (rows + 63) / 64 >= (size_t)MIN_BLOCKS ? 4
          : warpsFitting(warpsFor(rows, {GO_WARPS, 4, 2}), {GO_WARPS, 4, 2}, [](int w) { return gridOutSmem(w, C); })) {
    case 2: gridOutAt<2, 1, C>(gathered, Wout, ob, pair, n, q0, rows, tr); break;
    case 4: gridOutAt<4, 1, C>(gathered, Wout, ob, pair, n, q0, rows, tr); break;
    default: gridOutAt<GO_WARPS, 1, C>(gathered, Wout, ob, pair, n, q0, rows, tr);
  }
}
template <int C = 128>
inline void gridOut128(const half* gathered, const std::string& w, float* pair, int n, size_t q0, size_t rows, bool tr,
                       const float* ob = nullptr) {
  gridOutRaw<C>(gathered, Wh(w), ob, pair, n, q0, rows, tr);
}
