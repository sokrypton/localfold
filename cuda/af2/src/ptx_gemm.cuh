// Custom inline-PTX Tensor-Core (mma.sync.aligned.m16n8k16) GEMM kernels for AlphaFold 2.
// Replaces cuBLASLt / cuBLAS GEMMs with multi-stage cp.async + XOR-swizzled ldmatrix +
// conflict-free shared-memory coalesced uint4 epilogues (fused bias, ReLU, and f32 residual add).
#pragma once
#include "../../af3/src/ptx_primitives.cuh"
#include "outln.cuh"

namespace ptx_gemm {

__device__ __forceinline__ int swA32(int r, int k) { return r * 32 + (k ^ (((r >> 1) & 3) << 3)); }
__device__ __forceinline__ int swB128(int k, int c) { return k * 128 + (c ^ ((k & 7) << 3)); }
__device__ __forceinline__ int swB64(int k, int c) { return k * 64 + (c ^ ((k & 7) << 3)); }

// BM = 128, BN = 128, BK = 32, NST = 3 (48 KB smem -> 2 blocks/SM on sm_86, 256 threads = 8 warps: 4x2 warp grid)
template <class TI, class TO, bool BIAS, bool RELU, bool BETA>
__global__ void __launch_bounds__(256, 2)
ptxGemm128x128K(const TI* __restrict__ X, const TI* __restrict__ Wt, TO* __restrict__ Y,
                const float* __restrict__ bias, size_t rows, int K, int N, int ldx, int ldy) {
  constexpr bool BFI = std::is_same_v<TI, __nv_bfloat16>;
  constexpr bool BFO = std::is_same_v<TO, __nv_bfloat16>;
  constexpr int BM = 128, BN = 128, BK = 32, NTH = 256, STG = (BM * BK + BK * BN) * 2;
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int warpM = warp >> 1, warpN = warp & 1;
  const int col0 = (int)blockIdx.x * BN;
  const size_t row0 = (size_t)blockIdx.y * BM;

  auto As = [&](int s) { return reinterpret_cast<TI*>(smem + s * STG); };
  auto Bs = [&](int s) { return reinterpret_cast<TI*>(smem + s * STG) + BM * BK; };

  const bool fullRows = (row0 + BM <= rows);
  auto issue = [&](int kc, int s) {
    const int kBase = kc * BK;
    if (fullRows) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int u = i * NTH + tid, r = u >> 2, k = (u & 3) << 3;
        cpAsync16(As(s) + swA32(r, k), X + (row0 + r) * ldx + kBase + k, true);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int u = i * NTH + tid, r = u >> 2, k = (u & 3) << 3;
        const size_t row = row0 + r;
        cpAsync16(As(s) + swA32(r, k), X + (row < rows ? row : 0) * ldx + kBase + k, row < rows);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int u = i * NTH + tid, k = u >> 4, c = (u & 15) << 3;
      cpAsync16(Bs(s) + swB128(k, c), Wt + (size_t)(kBase + k) * N + col0 + c, true);
    }
    cpCommit();
  };

  const int ksteps = K / BK;
  issue(0, 0);
  if (ksteps > 1) {
    issue(1, 1);
    cpWait<1>();
  } else {
    cpWait<0>();
  }
  __syncthreads();

  float acc[2][8][4] = {};
  int st = 0, st2 = 2;
#pragma unroll 1
  for (int kc = 0; kc < ksteps; ++kc) {
    const TI* A = As(st);
    const TI* B = Bs(st);
    uint32_t a0[2][4], a1[2][4], f[2][4];
    const int r0 = warpM * 32 + (lane & 15);
    const int ak0 = (lane >> 4) * 8;
    const int kl0 = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int cl = warpN * 64 + (lane >> 4) * 8;
    ldsm4(a0[0], A + swA32(r0, ak0));
    ldsm4(a0[1], A + swA32(r0 + 16, ak0));
    ldsm4t(f[0], B + swB128(kl0, cl));
    ldsm4t(f[1], B + swB128(kl0, cl + 16));
    if (kc + 2 < ksteps) issue(kc + 2, st2);

#pragma unroll
    for (int n2 = 0; n2 < 4; ++n2) {
      const int cur = n2 & 1;
      if (n2 == 1) ldsm4t(f[0], B + swB128(kl0, cl + 32));
      if (n2 == 2) {
        ldsm4t(f[1], B + swB128(kl0, cl + 48));
        ldsm4(a1[0], A + swA32(r0, 16 + ak0));
        ldsm4(a1[1], A + swA32(r0 + 16, 16 + ak0));
      }
      if (n2 == 3) ldsm4t(f[0], B + swB128(kl0 + 16, cl));
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        if constexpr (BFI) {
          mma16816bf(acc[m][2 * n2], a0[m], f[cur][0], f[cur][1]);
          mma16816bf(acc[m][2 * n2 + 1], a0[m], f[cur][2], f[cur][3]);
        } else {
          mma16816(acc[m][2 * n2], a0[m], f[cur][0], f[cur][1]);
          mma16816(acc[m][2 * n2 + 1], a0[m], f[cur][2], f[cur][3]);
        }
      }
    }
#pragma unroll
    for (int n2 = 0; n2 < 4; ++n2) {
      const int cur = n2 & 1;
      if (n2 + 1 < 4) ldsm4t(f[cur ^ 1], B + swB128(kl0 + 16, cl + (n2 + 1) * 16));
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        if constexpr (BFI) {
          mma16816bf(acc[m][2 * n2], a1[m], f[cur][0], f[cur][1]);
          mma16816bf(acc[m][2 * n2 + 1], a1[m], f[cur][2], f[cur][3]);
        } else {
          mma16816(acc[m][2 * n2], a1[m], f[cur][0], f[cur][1]);
          mma16816(acc[m][2 * n2 + 1], a1[m], f[cur][2], f[cur][3]);
        }
      }
    }
    if (kc + 1 < ksteps) {
      if (kc + 2 < ksteps) cpWait<1>();
      else cpWait<0>();
      __syncthreads();
    }
    st = (st == 2) ? 0 : st + 1;
    st2 = (st2 == 2) ? 0 : st2 + 1;
  }
  if constexpr (!std::is_same_v<TO, float>) __syncthreads();

  const int g = lane >> 2, tig = lane & 3;
  constexpr int LDW = 72;
  TO* Ws = reinterpret_cast<TO*>(smem) + warp * (32 * LDW);
  const int rw0 = lane >> 3, cw0 = (lane & 7) << 3;
  if constexpr (!std::is_same_v<TO, float> && BETA) {
    if (fullRows) {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int rw = s * 4 + rw0;
        cpAsync16(Ws + rw * LDW + cw0, Y + (row0 + warpM * 32 + rw) * ldy + col0 + warpN * 64 + cw0, true);
      }
    } else {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int rw = s * 4 + rw0;
        const size_t row = row0 + warpM * 32 + rw;
        cpAsync16(Ws + rw * LDW + cw0, Y + (row < rows ? row : 0) * ldy + col0 + warpN * 64 + cw0, row < rows);
      }
    }
    cpCommit();
  }

  float2 bf[8];
  if constexpr (BIAS) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int col = col0 + warpN * 64 + nt * 8 + tig * 2;
      float2 b = ptx_ldg_v2_f32(bias + col);
      if constexpr (BFO) {
        if constexpr (BETA) { b.x = ptx_round_f16_f32(b.x); b.y = ptx_round_f16_f32(b.y); }
        b.x = ptx_bf16_raw_to_f32(ptx_f32_to_bf16_raw(b.x));
        b.y = ptx_bf16_raw_to_f32(ptx_f32_to_bf16_raw(b.y));
      } else if constexpr (std::is_same_v<TO, half>) {
        b.x = ptx_round_f16_f32(b.x);
        b.y = ptx_round_f16_f32(b.y);
      }
      bf[nt] = b;
    }
  }

  if constexpr (std::is_same_v<TO, float>) {
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const size_t row = row0 + warpM * 32 + m * 16 + g + 8 * h;
        if (row >= rows) continue;
        float2 rv[8];
        if constexpr (BETA) {
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            const int col = col0 + warpN * 64 + nt * 8 + tig * 2;
            rv[nt] = ptx_ld_global_v2_f32(Y + row * ldy + col);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int col = col0 + warpN * 64 + nt * 8 + tig * 2;
          float v0 = acc[m][nt][2 * h], v1 = acc[m][nt][2 * h + 1];
          if constexpr (BIAS) { v0 = ptx_add_f32(v0, bf[nt].x); v1 = ptx_add_f32(v1, bf[nt].y); }
          if constexpr (RELU) { v0 = ptx_relu_f32(v0); v1 = ptx_relu_f32(v1); }
          if constexpr (BETA) { v0 = ptx_add_f32(v0, rv[nt].x); v1 = ptx_add_f32(v1, rv[nt].y); }
          ptx_st_global_v2_f32(Y + row * ldy + col, make_float2(v0, v1));
        }
      }
    }
    return;
  } else {
    if constexpr (BETA) {
      cpWait<0>();
      ptx_bar_warp();
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int rw = m * 16 + g + 8 * h;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int cw = nt * 8 + tig * 2;
          float v0 = acc[m][nt][2 * h], v1 = acc[m][nt][2 * h + 1];
          if constexpr (BIAS) { v0 = ptx_add_f32(v0, bf[nt].x); v1 = ptx_add_f32(v1, bf[nt].y); }
          if constexpr (RELU) { v0 = ptx_relu_f32(v0); v1 = ptx_relu_f32(v1); }
          if constexpr (BETA) {
            uint32_t ru = ptx_ld_shared_u32(Ws + rw * LDW + cw);
            float r0, r1;
            if constexpr (BFO) ptx_bf16x2_to_f32x2(ru, r0, r1);
            else ptx_f16x2_to_f32x2(ru, r0, r1);
            v0 = ptx_add_f32(v0, r0);
            v1 = ptx_add_f32(v1, r1);
          }
          uint32_t packed = BFO ? ptx_cvt_bf16x2_f32(v0, v1) : ptx_cvt_f16x2_f32(v0, v1);
          ptx_st_shared_u32(Ws + rw * LDW + cw, packed);
        }
      }
    }
    ptx_bar_warp();
    if (fullRows) {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int rw = s * 4 + rw0;
        ptx_st_global_v4_u32(Y + (row0 + warpM * 32 + rw) * ldy + col0 + warpN * 64 + cw0,
                             ptx_ld_shared_v4_u32(Ws + rw * LDW + cw0));
      }
    } else {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int rw = s * 4 + rw0;
        const size_t row = row0 + warpM * 32 + rw;
        if (row < rows) {
          ptx_st_global_v4_u32(Y + row * ldy + col0 + warpN * 64 + cw0,
                               ptx_ld_shared_v4_u32(Ws + rw * LDW + cw0));
        }
      }
    }
  }
}

// BM = 64, BN = 64, BK = 32, NST = 3 (24 KB smem -> 4 blocks/SM on sm_86, 128 threads = 4 warps: 2x2 warp grid)
template <class TI, class TO, bool BIAS, bool RELU, bool BETA>
__global__ void __launch_bounds__(128, 4)
ptxGemm64x64K(const TI* __restrict__ X, const TI* __restrict__ Wt, TO* __restrict__ Y,
              const float* __restrict__ bias, size_t rows, int K, int N, int ldx, int ldy) {
  constexpr bool BFI = std::is_same_v<TI, __nv_bfloat16>;
  constexpr bool BFO = std::is_same_v<TO, __nv_bfloat16>;
  constexpr int BM = 64, BN = 64, BK = 32, NTH = 128, STG = (BM * BK + BK * BN) * 2;
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int warpM = warp >> 1, warpN = warp & 1;
  const int col0 = (int)blockIdx.x * BN;
  const size_t row0 = (size_t)blockIdx.y * BM;

  auto As = [&](int s) { return reinterpret_cast<TI*>(smem + s * STG); };
  auto Bs = [&](int s) { return reinterpret_cast<TI*>(smem + s * STG) + BM * BK; };

  const bool fullRows = (row0 + BM <= rows);
  auto issue = [&](int kc, int s) {
    const int kBase = kc * BK;
    if (fullRows) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int u = i * NTH + tid, r = u >> 2, k = (u & 3) << 3;
        cpAsync16(As(s) + swA32(r, k), X + (row0 + r) * ldx + kBase + k, true);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int u = i * NTH + tid, r = u >> 2, k = (u & 3) << 3;
        const size_t row = row0 + r;
        cpAsync16(As(s) + swA32(r, k), X + (row < rows ? row : 0) * ldx + kBase + k, row < rows);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int u = i * NTH + tid, k = u >> 3, c = (u & 7) << 3;
      cpAsync16(Bs(s) + swB64(k, c), Wt + (size_t)(kBase + k) * N + col0 + c, true);
    }
    cpCommit();
  };

  const int ksteps = K / BK;
  issue(0, 0);
  if (ksteps > 1) {
    issue(1, 1);
    cpWait<1>();
  } else {
    cpWait<0>();
  }
  __syncthreads();

  float acc[2][4][4] = {};
  int st = 0, st2 = 2;
#pragma unroll 1
  for (int kc = 0; kc < ksteps; ++kc) {
    const TI* A = As(st);
    const TI* B = Bs(st);
    uint32_t a0[2][4], a1[2][4], f[2][4];
    const int r0 = warpM * 32 + (lane & 15);
    const int ak0 = (lane >> 4) * 8;
    const int kl0 = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int cl = warpN * 32 + (lane >> 4) * 8;
    ldsm4(a0[0], A + swA32(r0, ak0));
    ldsm4(a0[1], A + swA32(r0 + 16, ak0));
    ldsm4t(f[0], B + swB64(kl0, cl));
    ldsm4t(f[1], B + swB64(kl0, cl + 16));
    if (kc + 2 < ksteps) issue(kc + 2, st2);

#pragma unroll
    for (int n2 = 0; n2 < 2; ++n2) {
      const int cur = n2 & 1;
      if (n2 == 0) {
        ldsm4(a1[0], A + swA32(r0, 16 + ak0));
        ldsm4(a1[1], A + swA32(r0 + 16, 16 + ak0));
      }
      if (n2 == 1) ldsm4t(f[0], B + swB64(kl0 + 16, cl));
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        if constexpr (BFI) {
          mma16816bf(acc[m][2 * n2], a0[m], f[cur][0], f[cur][1]);
          mma16816bf(acc[m][2 * n2 + 1], a0[m], f[cur][2], f[cur][3]);
        } else {
          mma16816(acc[m][2 * n2], a0[m], f[cur][0], f[cur][1]);
          mma16816(acc[m][2 * n2 + 1], a0[m], f[cur][2], f[cur][3]);
        }
      }
    }
#pragma unroll
    for (int n2 = 0; n2 < 2; ++n2) {
      const int cur = n2 & 1;
      if (n2 == 0) ldsm4t(f[1], B + swB64(kl0 + 16, cl + 16));
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        if constexpr (BFI) {
          mma16816bf(acc[m][2 * n2], a1[m], f[cur][0], f[cur][1]);
          mma16816bf(acc[m][2 * n2 + 1], a1[m], f[cur][2], f[cur][3]);
        } else {
          mma16816(acc[m][2 * n2], a1[m], f[cur][0], f[cur][1]);
          mma16816(acc[m][2 * n2 + 1], a1[m], f[cur][2], f[cur][3]);
        }
      }
    }
    if (kc + 1 < ksteps) {
      if (kc + 2 < ksteps) cpWait<1>();
      else cpWait<0>();
      __syncthreads();
    }
    st = (st == 2) ? 0 : st + 1;
    st2 = (st2 == 2) ? 0 : st2 + 1;
  }
  if constexpr (!std::is_same_v<TO, float>) __syncthreads();

  const int g = lane >> 2, tig = lane & 3;
  constexpr int LDW = 40;
  TO* Ws = reinterpret_cast<TO*>(smem) + warp * (32 * LDW);
  const int rw0 = lane >> 2, cw0 = (lane & 3) << 3;
  if constexpr (!std::is_same_v<TO, float> && BETA) {
    if (fullRows) {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int rw = s * 8 + rw0;
        cpAsync16(Ws + rw * LDW + cw0, Y + (row0 + warpM * 32 + rw) * ldy + col0 + warpN * 32 + cw0, true);
      }
    } else {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int rw = s * 8 + rw0;
        const size_t row = row0 + warpM * 32 + rw;
        cpAsync16(Ws + rw * LDW + cw0, Y + (row < rows ? row : 0) * ldy + col0 + warpN * 32 + cw0, row < rows);
      }
    }
    cpCommit();
  }

  float2 bf[4];
  if constexpr (BIAS) {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = col0 + warpN * 32 + nt * 8 + tig * 2;
      float2 b = ptx_ldg_v2_f32(bias + col);
      if constexpr (BFO) {
        if constexpr (BETA) { b.x = ptx_round_f16_f32(b.x); b.y = ptx_round_f16_f32(b.y); }
        b.x = ptx_bf16_raw_to_f32(ptx_f32_to_bf16_raw(b.x));
        b.y = ptx_bf16_raw_to_f32(ptx_f32_to_bf16_raw(b.y));
      } else if constexpr (std::is_same_v<TO, half>) {
        b.x = ptx_round_f16_f32(b.x);
        b.y = ptx_round_f16_f32(b.y);
      }
      bf[nt] = b;
    }
  }

  if constexpr (std::is_same_v<TO, float>) {
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const size_t row = row0 + warpM * 32 + m * 16 + g + 8 * h;
        if (row >= rows) continue;
        float2 rv[4];
        if constexpr (BETA) {
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int col = col0 + warpN * 32 + nt * 8 + tig * 2;
            rv[nt] = ptx_ld_global_v2_f32(Y + row * ldy + col);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int col = col0 + warpN * 32 + nt * 8 + tig * 2;
          float v0 = acc[m][nt][2 * h], v1 = acc[m][nt][2 * h + 1];
          if constexpr (BIAS) { v0 = ptx_add_f32(v0, bf[nt].x); v1 = ptx_add_f32(v1, bf[nt].y); }
          if constexpr (RELU) { v0 = ptx_relu_f32(v0); v1 = ptx_relu_f32(v1); }
          if constexpr (BETA) { v0 = ptx_add_f32(v0, rv[nt].x); v1 = ptx_add_f32(v1, rv[nt].y); }
          ptx_st_global_v2_f32(Y + row * ldy + col, make_float2(v0, v1));
        }
      }
    }
  } else {
    if constexpr (BETA) {
      cpWait<0>();
      ptx_bar_warp();
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int rw = m * 16 + g + 8 * h;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int cw = nt * 8 + tig * 2;
          float v0 = acc[m][nt][2 * h], v1 = acc[m][nt][2 * h + 1];
          if constexpr (BIAS) { v0 = ptx_add_f32(v0, bf[nt].x); v1 = ptx_add_f32(v1, bf[nt].y); }
          if constexpr (RELU) { v0 = ptx_relu_f32(v0); v1 = ptx_relu_f32(v1); }
          if constexpr (BETA) {
            uint32_t ru = ptx_ld_shared_u32(Ws + rw * LDW + cw);
            float r0, r1;
            if constexpr (BFO) ptx_bf16x2_to_f32x2(ru, r0, r1);
            else ptx_f16x2_to_f32x2(ru, r0, r1);
            v0 = ptx_add_f32(v0, r0);
            v1 = ptx_add_f32(v1, r1);
          }
          uint32_t packed = BFO ? ptx_cvt_bf16x2_f32(v0, v1) : ptx_cvt_f16x2_f32(v0, v1);
          ptx_st_shared_u32(Ws + rw * LDW + cw, packed);
        }
      }
    }
    ptx_bar_warp();
    if (fullRows) {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int rw = s * 8 + rw0;
        const int col = col0 + warpN * 32 + cw0;
        if (col < N) {
          ptx_st_global_v4_u32(Y + (row0 + warpM * 32 + rw) * ldy + col,
                               ptx_ld_shared_v4_u32(Ws + rw * LDW + cw0));
        }
      }
    } else {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int rw = s * 8 + rw0;
        const size_t row = row0 + warpM * 32 + rw;
        const int col = col0 + warpN * 32 + cw0;
        if (row < rows && col < N) {
          ptx_st_global_v4_u32(Y + row * ldy + col, ptx_ld_shared_v4_u32(Ws + rw * LDW + cw0));
        }
      }
    }
  }
}

template <class TI, class TO, bool BIAS, bool RELU, bool BETA>
inline void launchPtxGemm16(const TI* X, const TI* Wt, TO* Y, const float* bias,
                            size_t rows, int K, int N, int ldx, int ldy) {
  const size_t blocks128 = ((N + 127) / 128) * ((rows + 127) / 128);
  if (rows >= 96 && N % 128 == 0 && blocks128 >= 72) {
    constexpr int smem128 = 3 * (128 * 32 + 32 * 128) * 2;
    static bool attr = false;
    if (!attr) { smemAttr((ptxGemm128x128K<TI, TO, BIAS, RELU, BETA>), smem128); attr = true; }
    dim3 grid((unsigned)((N + 127) / 128), (unsigned)((rows + 127) / 128));
    ptxGemm128x128K<TI, TO, BIAS, RELU, BETA><<<grid, 256, smem128, STREAM>>>(X, Wt, Y, bias, rows, K, N, ldx, ldy);
  } else {
    constexpr int smem64 = 3 * (64 * 32 + 32 * 64) * 2;
    dim3 grid((unsigned)((N + 63) / 64), (unsigned)((rows + 63) / 64));
    ptxGemm64x64K<TI, TO, BIAS, RELU, BETA><<<grid, 128, smem64, STREAM>>>(X, Wt, Y, bias, rows, K, N, ldx, ldy);
  }
}

template <class TI, class TO>
inline void dispatchPtxGemm16(const TI* X, const TI* Wt, TO* Y, const float* bias,
                              size_t rows, int K, int N, int ldx, int ldy, bool relu, bool beta) {
  if (bias) {
    if (relu) {
      if (beta) launchPtxGemm16<TI, TO, true, true, true>(X, Wt, Y, bias, rows, K, N, ldx, ldy);
      else launchPtxGemm16<TI, TO, true, true, false>(X, Wt, Y, bias, rows, K, N, ldx, ldy);
    } else {
      if (beta) launchPtxGemm16<TI, TO, true, false, true>(X, Wt, Y, bias, rows, K, N, ldx, ldy);
      else launchPtxGemm16<TI, TO, true, false, false>(X, Wt, Y, bias, rows, K, N, ldx, ldy);
    }
  } else {
    if (relu) {
      if (beta) launchPtxGemm16<TI, TO, false, true, true>(X, Wt, Y, nullptr, rows, K, N, ldx, ldy);
      else launchPtxGemm16<TI, TO, false, true, false>(X, Wt, Y, nullptr, rows, K, N, ldx, ldy);
    } else {
      if (beta) launchPtxGemm16<TI, TO, false, false, true>(X, Wt, Y, nullptr, rows, K, N, ldx, ldy);
      else launchPtxGemm16<TI, TO, false, false, false>(X, Wt, Y, nullptr, rows, K, N, ldx, ldy);
    }
  }
}

} // namespace ptx_gemm
