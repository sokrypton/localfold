// Element-wise and row kernels every port had its own copy of (cuda/af2, cuda/ef2, cuda/af3 - the same
// computation under the same name, renamed variables apart), here once. Included by common.cuh.
#pragma once

__global__ void addK(float* __restrict__ y, const float* __restrict__ x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)y | (uintptr_t)x) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 yv = ptx_ld_global_v4_f32(reinterpret_cast<const float4*>(y) + t);
      float4 xv = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(x) + t);
      asm("add.f32 %0, %0, %4;\n\t"
          "add.f32 %1, %1, %5;\n\t"
          "add.f32 %2, %2, %6;\n\t"
          "add.f32 %3, %3, %7;"
          : "+f"(yv.x), "+f"(yv.y), "+f"(yv.z), "+f"(yv.w)
          : "f"(xv.x), "f"(xv.y), "f"(xv.z), "f"(xv.w));
      ptx_st_global_v4_f32(reinterpret_cast<float4*>(y) + t, yv);
    }
    return;
  }
  if (t < n) {
    float yv = ptx_ld_global_f32(y + t), xv = ptx_ldg_f32(x + t);
    asm("add.f32 %0, %0, %1;" : "+f"(yv) : "f"(xv));
    ptx_st_global_f32(y + t, yv);
  }
}

__global__ void addBiasK(float* __restrict__ y, const float* __restrict__ b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    int c = ((c4 & (c4 - 1)) == 0) ? (int)(t & (size_t)(c4 - 1)) : (n4 <= UINT32_MAX ? (int)((unsigned)t % (unsigned)c4) : (int)(t % c4));
    float4 yv = ptx_ld_global_v4_f32(reinterpret_cast<const float4*>(y) + t);
    float4 bv = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(b) + c);
    asm("add.f32 %0, %0, %4;\n\t"
        "add.f32 %1, %1, %5;\n\t"
        "add.f32 %2, %2, %6;\n\t"
        "add.f32 %3, %3, %7;"
        : "+f"(yv.x), "+f"(yv.y), "+f"(yv.z), "+f"(yv.w)
        : "f"(bv.x), "f"(bv.y), "f"(bv.z), "f"(bv.w));
    ptx_st_global_v4_f32(reinterpret_cast<float4*>(y) + t, yv);
    return;
  }
  size_t n = rows * C;
  if (t >= n) return;
  int c = ((C & (C - 1)) == 0) ? (int)(t & (size_t)(C - 1)) : (n <= UINT32_MAX ? (int)((unsigned)t % (unsigned)C) : (int)(t % C));
  float yv = ptx_ld_global_f32(y + t), bv = ptx_ldg_f32(b + c);
  asm("add.f32 %0, %0, %1;" : "+f"(yv) : "f"(bv));
  ptx_st_global_f32(y + t, yv);
}

__global__ void symmetriseK(const float* z, float* out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)T * T * C;
  if (t >= n) return;
  int c, i, j;
  if (n <= UINT32_MAX) {
    unsigned u = (unsigned)t, uc = (unsigned)C, ut = (unsigned)T;
    unsigned ij = ((uc & (uc - 1)) == 0) ? (u >> (__ffs(C) - 1)) : (u / uc);
    c = (int)(u - ij * uc);
    i = (int)(ij / ut);
    j = (int)(ij - (unsigned)i * ut);
  } else {
    c = (int)(t % C); size_t ij = t / C; i = (int)(ij / T); j = (int)(ij % T);
  }
  float a = ptx_ldg_f32(z + t), b = ptx_ldg_f32(z + ((size_t)j * T + i) * C + c);
  asm("add.f32 %0, %0, %1;" : "+f"(a) : "f"(b));
  ptx_st_global_f32(out + t, a);
}

// rows [r0, r0 + R) of z + z^T (the whole: symmetriseK)
__global__ void symmetriseRowsK(const float* z, float* out, int T, int C, size_t r0, size_t R) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; size_t i = r0 + ij / T, j = ij % T;
  float a = ptx_ldg_f32(z + (i * T + j) * C + c), b = ptx_ldg_f32(z + (j * T + i) * C + c);
  asm("add.f32 %0, %0, %1;" : "+f"(a) : "f"(b));
  ptx_st_global_f32(out + t, a);
}
__global__ void gateMulAddK(float* __restrict__ pair, const float* __restrict__ out, const float* __restrict__ gate, size_t n,
                            const float* __restrict__ outB = nullptr, const float* __restrict__ gateB = nullptr, int C = 0, int ld = 0) {
  if (!ld) ld = C;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (C & 3) == 0 && (ld & 3) == 0 && (((uintptr_t)pair | (uintptr_t)out | (uintptr_t)gate | (uintptr_t)outB | (uintptr_t)gateB) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      int c4 = C ? (C >> 2) : 0, ld4 = ld ? (ld >> 2) : 0;
      size_t r = 0; int c = 0;
      if (c4) {
        if ((c4 & (c4 - 1)) == 0) { c = (int)(t & (size_t)(c4 - 1)); r = t >> (__ffs(c4) - 1); }
        else if (n4 <= UINT32_MAX) { unsigned u = (unsigned)t, uc = (unsigned)c4, q = u / uc; r = q; c = (int)(u - q * uc); }
        else { r = t / c4; c = (int)(t % c4); }
      }
      float4 p = ptx_ld_global_v4_f32(reinterpret_cast<const float4*>(pair) + t);
      float4 o = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(out) + t);
      float4 g = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(gate) + (ld != C ? r * (size_t)ld4 + c : t));
      if (outB) { float4 ob = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(outB) + c); o.x += ob.x; o.y += ob.y; o.z += ob.z; o.w += ob.w; }
      if (gateB) { float4 gb = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(gateB) + c); g.x += gb.x; g.y += gb.y; g.z += gb.z; g.w += gb.w; }
      p.x = ptx_fma_f32(o.x, ptx_sigm_f32(g.x), p.x);
      p.y = ptx_fma_f32(o.y, ptx_sigm_f32(g.y), p.y);
      p.z = ptx_fma_f32(o.z, ptx_sigm_f32(g.z), p.z);
      p.w = ptx_fma_f32(o.w, ptx_sigm_f32(g.w), p.w);
      ptx_st_global_v4_f32(reinterpret_cast<float4*>(pair) + t, p);
    }
    return;
  }
  if (t < n) {
    int c = C ? (int)(t % C) : 0;
    size_t gIdx = (ld != C && C) ? (t / C) * (size_t)ld + c : t;
    float ov = ptx_ldg_f32(out + t) + (outB ? ptx_ldg_f32(outB + c) : 0.f);
    float gv = ptx_ldg_f32(gate + gIdx) + (gateB ? ptx_ldg_f32(gateB + c) : 0.f);
    float pv = ptx_ld_global_f32(pair + t);
    ptx_st_global_f32(pair + t, ptx_fma_f32(ov, ptx_sigm_f32(gv), pv));
  }
}

__global__ void gateMulAddHK(float* __restrict__ pair, const float* __restrict__ out, const half* __restrict__ gate,
                             size_t pairs, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (ld & 3) == 0 && (((uintptr_t)pair | (uintptr_t)out) & 15) == 0 && ((uintptr_t)gate & 7) == 0) {
    int c4 = C >> 2, ld4 = ld >> 2;
    size_t n4 = pairs * (size_t)c4;
    if (t >= n4) return;
    size_t r; int c;
    if ((c4 & (c4 - 1)) == 0) {
      c = (int)(t & (size_t)(c4 - 1));
      r = t >> (__ffs(c4) - 1);
    } else if (n4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, uc = (unsigned)c4, q = u / uc;
      r = q; c = (int)(u - q * uc);
    } else {
      r = t / c4; c = (int)(t % c4);
    }
    float4 p = ptx_ld_global_v4_f32(reinterpret_cast<const float4*>(pair) + t);
    float4 o = ptx_ldg_v4_f32(reinterpret_cast<const float4*>(out) + t);
    uint2 g2 = ptx_ldg_v2_u32(reinterpret_cast<const uint2*>(gate) + r * (size_t)ld4 + c);
    float2 g0 = ptx_f16x2_to_f32x2(g2.x);
    float2 g1 = ptx_f16x2_to_f32x2(g2.y);
    p.x = ptx_fma_f32(o.x, ptx_sigm_f32(g0.x), p.x);
    p.y = ptx_fma_f32(o.y, ptx_sigm_f32(g0.y), p.y);
    p.z = ptx_fma_f32(o.z, ptx_sigm_f32(g1.x), p.z);
    p.w = ptx_fma_f32(o.w, ptx_sigm_f32(g1.y), p.w);
    ptx_st_global_v4_f32(reinterpret_cast<float4*>(pair) + t, p);
    return;
  }
  if (t >= pairs * C) return;
  size_t r = t / C; int c = (int)(t % C);
  float pv = ptx_ld_global_f32(pair + t), ov = ptx_ldg_f32(out + t);
  float gv = ptx_f16_raw_to_f32(ptx_ldg_u16(gate + r * ld + c));
  ptx_st_global_f32(pair + t, ptx_fma_f32(ov, ptx_sigm_f32(gv), pv));
}

__global__ void channelMajorToRowsK(const float* x, float* y, size_t pairs, int C) {   // [c][ij] -> [ij][c]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < pairs * C) ptx_st_global_f32(y + t, ptx_ldg_f32(x + (t % C) * pairs + t / C));
}

__global__ void __launch_bounds__(256, 6)
channelMajorToRowsTileK(const float* __restrict__ x, float* __restrict__ y, size_t pairs, int C) {
  __shared__ float tile[32][33];
  size_t ij0 = (size_t)blockIdx.x * 32;
  int c0 = (int)blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
  size_t ij = ij0 + tx;
  if (ij0 + 32 <= pairs && c0 + 32 <= C) {
    float v0 = ptx_ldg_f32(x + (size_t)(c0 + ty) * pairs + ij);
    float v1 = ptx_ldg_f32(x + (size_t)(c0 + ty + 8) * pairs + ij);
    float v2 = ptx_ldg_f32(x + (size_t)(c0 + ty + 16) * pairs + ij);
    float v3 = ptx_ldg_f32(x + (size_t)(c0 + ty + 24) * pairs + ij);
    ptx_st_shared_f32(&tile[ty][tx], v0);
    ptx_st_shared_f32(&tile[ty + 8][tx], v1);
    ptx_st_shared_f32(&tile[ty + 16][tx], v2);
    ptx_st_shared_f32(&tile[ty + 24][tx], v3);
    __syncthreads();
    int c = c0 + tx;
    size_t r = ij0 + ty;
    ptx_st_global_f32(y + r * C + c, ptx_ld_shared_f32(&tile[tx][ty]));
    ptx_st_global_f32(y + (r + 8) * C + c, ptx_ld_shared_f32(&tile[tx][ty + 8]));
    ptx_st_global_f32(y + (r + 16) * C + c, ptx_ld_shared_f32(&tile[tx][ty + 16]));
    ptx_st_global_f32(y + (r + 24) * C + c, ptx_ld_shared_f32(&tile[tx][ty + 24]));
    return;
  }
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    int c = c0 + ty + k;
    if (c < C && ij < pairs) ptx_st_shared_f32(&tile[ty + k][tx], ptx_ldg_f32(x + (size_t)c * pairs + ij));
  }
  __syncthreads();
  int c = c0 + tx;
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    size_t r = ij0 + ty + k;
    if (r < pairs && c < C) ptx_st_global_f32(y + r * C + c, ptx_ld_shared_f32(&tile[tx][ty + k]));
  }
}

template <class TY = half>
__global__ void centerNormHK(const float* __restrict__ prod, TY* __restrict__ out, size_t pairs, int C,
                             const float* __restrict__ scale, const float* __restrict__ offset, int L, int Lp) {
  extern __shared__ float tile[];               // [C][33]
  size_t r0 = (size_t)blockIdx.x * 32;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  size_t rLoad = r0 + lane;
  size_t plane = (size_t)Lp * Lp;
  size_t planeIdx = 0;
  if (rLoad < pairs) {
    unsigned ur = (unsigned)rLoad, uL = (unsigned)L, q = ur / uL;
    planeIdx = (size_t)q * Lp + (ur - q * uL);
  }
  if (C == 256 && blockDim.x == 256) {
    if (rLoad < pairs) {
      const float* pBase = prod + planeIdx;
      #pragma unroll
      for (int k = 0; k < 32; ++k) {
        int c = k * 8 + warp;
        ptx_st_shared_f32(tile + c * 33 + lane, ptx_ldg_f32(pBase + (size_t)c * plane));
      }
    }
    __syncthreads();
    float sc[8], of[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      int c = k * 32 + lane;
      sc[k] = ptx_ldg_f32(scale + c);
      of[k] = ptx_ldg_f32(offset + c);
    }
    #pragma unroll
    for (int rk = 0; rk < 4; ++rk) {
      int row = rk * 8 + warp;
      size_t r = r0 + row;
      if (r >= pairs) break;
      float tv[8];
      float s = 0;
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        float val = ptx_ld_shared_f32(tile + (k * 32 + lane) * 33 + row);
        tv[k] = val;
        s += val;
      }
      s = ptx_warp_sum_f32(s);
      float mean = s * (1.f / 256.f), v = 0;
      #pragma unroll
      for (int k = 0; k < 8; ++k) { float d = tv[k] - mean; v = ptx_fma_f32(d, d, v); }
      v = ptx_warp_sum_f32(v);
      float inv = ptx_rsqrt_f32(ptx_fma_f32(v, 1.f / 256.f, 1e-5f));
      TY* outRow = out + r * 256;
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        float yv = ptx_fma_f32((tv[k] - mean) * inv, sc[k], of[k]);
        if constexpr (std::is_same_v<TY, half>) ptx_st_global_u16(outRow + k * 32 + lane, ptx_f32_to_f16_raw(yv));
        else ptx_st_global_f32(outRow + k * 32 + lane, yv);
      }
    }
    return;
  }
  if (C == 128 && blockDim.x == 256) {
    if (rLoad < pairs) {
      const float* pBase = prod + planeIdx;
      #pragma unroll
      for (int k = 0; k < 16; ++k) {
        int c = k * 8 + warp;
        ptx_st_shared_f32(tile + c * 33 + lane, ptx_ldg_f32(pBase + (size_t)c * plane));
      }
    }
    __syncthreads();
    int sub = lane >> 4, hl = lane & 15;
    float sc[8], of[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      int c = k * 16 + hl;
      sc[k] = ptx_ldg_f32(scale + c);
      of[k] = ptx_ldg_f32(offset + c);
    }
    #pragma unroll
    for (int rk = 0; rk < 2; ++rk) {
      int row = rk * 16 + warp * 2 + sub;
      size_t r = r0 + row;
      if (r >= pairs) break;
      float tv[8];
      float s = 0;
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        float val = ptx_ld_shared_f32(tile + (k * 16 + hl) * 33 + row);
        asm volatile("mov.b32 %0, %0;" : "+f"(val));
        tv[k] = val;
        s += val;
      }
      s = ptx_halfwarp_sum_f32(s);
      float mean = s * (1.f / 128.f), v = 0;
      #pragma unroll
      for (int k = 0; k < 8; ++k) { float d = tv[k] - mean; v = ptx_fma_f32(d, d, v); }
      v = ptx_halfwarp_sum_f32(v);
      float inv = ptx_rsqrt_f32(ptx_fma_f32(v, 1.f / 128.f, 1e-5f));
      TY* outRow = out + r * 128;
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        float yv = ptx_fma_f32((tv[k] - mean) * inv, sc[k], of[k]);
        if constexpr (std::is_same_v<TY, half>) ptx_st_global_u16(outRow + k * 16 + hl, ptx_f32_to_f16_raw(yv));
        else ptx_st_global_f32(outRow + k * 16 + hl, yv);
      }
    }
    return;
  }
  if (C == 64 && blockDim.x == 256) {
    if (rLoad < pairs) {
      const float* pBase = prod + planeIdx;
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        int c = k * 8 + warp;
        ptx_st_shared_f32(tile + c * 33 + lane, ptx_ldg_f32(pBase + (size_t)c * plane));
      }
    }
    __syncthreads();
    int sub = lane >> 4, hl = lane & 15;
    float sc[4], of[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int c = k * 16 + hl;
      sc[k] = ptx_ldg_f32(scale + c);
      of[k] = ptx_ldg_f32(offset + c);
    }
    #pragma unroll
    for (int rk = 0; rk < 2; ++rk) {
      int row = rk * 16 + warp * 2 + sub;
      size_t r = r0 + row;
      if (r >= pairs) break;
      float tv[4];
      float s = 0;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        float val = ptx_ld_shared_f32(tile + (k * 16 + hl) * 33 + row);
        asm volatile("mov.b32 %0, %0;" : "+f"(val));
        tv[k] = val;
        s += val;
      }
      s = ptx_halfwarp_sum_f32(s);
      float mean = s * (1.f / 64.f), v = 0;
      #pragma unroll
      for (int k = 0; k < 4; ++k) { float d = tv[k] - mean; v = ptx_fma_f32(d, d, v); }
      v = ptx_halfwarp_sum_f32(v);
      float inv = ptx_rsqrt_f32(ptx_fma_f32(v, 1.f / 64.f, 1e-5f));
      TY* outRow = out + r * 64;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        float yv = ptx_fma_f32((tv[k] - mean) * inv, sc[k], of[k]);
        if constexpr (std::is_same_v<TY, half>) ptx_st_global_u16(outRow + k * 16 + hl, ptx_f32_to_f16_raw(yv));
        else ptx_st_global_f32(outRow + k * 16 + hl, yv);
      }
    }
    return;
  }
  for (int c = warp; c < C; c += nw) {
    ptx_st_shared_f32(tile + c * 33 + lane, rLoad < pairs ? ptx_ldg_f32(prod + (size_t)c * plane + planeIdx) : 0.f);
  }
  __syncthreads();
  for (int row = warp; row < 32; row += nw) {
    size_t r = r0 + row;
    if (r >= pairs) break;
    float s = 0;
    for (int c = lane; c < C; c += 32) s += ptx_ld_shared_f32(tile + c * 33 + row);
    s = ptx_warp_sum_f32(s);
    float mean = s / C, v = 0;
    for (int c = lane; c < C; c += 32) { float d = ptx_ld_shared_f32(tile + c * 33 + row) - mean; v = ptx_fma_f32(d, d, v); }
    v = ptx_warp_sum_f32(v);
    float inv = ptx_rsqrt_f32(v / C + 1e-5f);
    for (int c = lane; c < C; c += 32) {
      float yv = ptx_fma_f32((ptx_ld_shared_f32(tile + c * 33 + row) - mean) * inv, ptx_ldg_f32(scale + c), ptx_ldg_f32(offset + c));
      if constexpr (std::is_same_v<TY, half>) ptx_st_global_u16(out + r * C + c, ptx_f32_to_f16_raw(yv));
      else ptx_st_global_f32(out + r * C + c, yv);
    }
  }
}

// TI, TO: the rows' and the output's types (a bf16 MSA in, a bf16 transition input out: cuda/af2's MSA16)
inline unsigned lnVGrid(size_t rows) { return (unsigned)((rows + 7) / 8); }
template <int C, class TI = float, class TO = half, bool RELU = false>
__global__ void __launch_bounds__(256, 6)
layerNormVK(const TI* __restrict__ x, TO* __restrict__ y, size_t rows, const float* __restrict__ scale,
            const float* __restrict__ offset) {
  constexpr int VEC = C >= 128 ? 4 : C / 32, STEPS = C / (32 * VEC);
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const TI* xr = x + row * C;
  float v[STEPS][VEC], sc[STEPS][VEC], of[STEPS][VEC];
  float s = 0;
#pragma unroll
  for (int k = 0; k < STEPS; ++k) {
    int c = (k * 32 + lane) * VEC;
    if constexpr (VEC == 4) {
      float4 s4 = ptx_ldg_v4_f32(scale + c), o4 = ptx_ldg_v4_f32(offset + c);
      sc[k][0] = s4.x; sc[k][1] = s4.y; sc[k][2] = s4.z; sc[k][3] = s4.w;
      of[k][0] = o4.x; of[k][1] = o4.y; of[k][2] = o4.z; of[k][3] = o4.w;
    } else {
      float2 s2 = ptx_ldg_v2_f32(scale + c), o2 = ptx_ldg_v2_f32(offset + c);
      sc[k][0] = s2.x; sc[k][1] = s2.y; of[k][0] = o2.x; of[k][1] = o2.y;
    }
    if constexpr (!std::is_same_v<TI, float>) {
      if constexpr (VEC == 4) {
        uint2 u2 = ptx_ldg_v2_u32(xr + c);
        float2 q0 = std::is_same_v<TI, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(u2.x) : ptx_f16x2_to_f32x2(u2.x);
        float2 q1 = std::is_same_v<TI, __nv_bfloat16> ? ptx_bf16x2_to_f32x2(u2.y) : ptx_f16x2_to_f32x2(u2.y);
        v[k][0] = q0.x; v[k][1] = q0.y; v[k][2] = q1.x; v[k][3] = q1.y;
      } else {
        float2 q = ld2f(xr + c); v[k][0] = q.x; v[k][1] = q.y;
      }
    } else if constexpr (VEC == 4) {
      float4 q = ptx_ldg_v4_f32(xr + c); v[k][0] = q.x; v[k][1] = q.y; v[k][2] = q.z; v[k][3] = q.w;
    } else {
      float2 q = ptx_ldg_v2_f32(xr + c); v[k][0] = q.x; v[k][1] = q.y;
    }
#pragma unroll
    for (int u = 0; u < VEC; ++u) s += v[k][u];
  }
  s = ptx_warp_sum_f32(s);
  float mean = ptx_div_f32(s, (float)C), q2 = 0;
#pragma unroll
  for (int k = 0; k < STEPS; ++k)
#pragma unroll
    for (int u = 0; u < VEC; ++u) { float d = v[k][u] - mean; q2 += d * d; }
  q2 = ptx_warp_sum_f32(q2);
  float inv = ptx_rsqrt_f32(ptx_div_f32(q2, (float)C) + 1e-5f);
  TO* yr = y + row * C;
#pragma unroll
  for (int k = 0; k < STEPS; ++k) {
    int c = (k * 32 + lane) * VEC;
    if constexpr (VEC == 4) {
      float a0 = ptx_fma_f32((v[k][0] - mean) * inv, sc[k][0], of[k][0]);
      float b0 = ptx_fma_f32((v[k][1] - mean) * inv, sc[k][1], of[k][1]);
      float a1 = ptx_fma_f32((v[k][2] - mean) * inv, sc[k][2], of[k][2]);
      float b1 = ptx_fma_f32((v[k][3] - mean) * inv, sc[k][3], of[k][3]);
      if constexpr (RELU) { a0 = ptx_relu_f32(a0); b0 = ptx_relu_f32(b0); a1 = ptx_relu_f32(a1); b1 = ptx_relu_f32(b1); }
      uint2 out2;
      if constexpr (std::is_same_v<TO, half>) {
        out2.x = ptx_cvt_f16x2_f32(a0, b0); out2.y = ptx_cvt_f16x2_f32(a1, b1);
      } else {
        out2.x = ptx_cvt_bf16x2_f32(a0, b0); out2.y = ptx_cvt_bf16x2_f32(a1, b1);
      }
      ptx_st_global_v2_u32(yr + c, out2);
    } else {
      float a = ptx_fma_f32((v[k][0] - mean) * inv, sc[k][0], of[k][0]);
      float b = ptx_fma_f32((v[k][1] - mean) * inv, sc[k][1], of[k][1]);
      if constexpr (RELU) { a = ptx_relu_f32(a); b = ptx_relu_f32(b); }
      if constexpr (std::is_same_v<TO, half>) ptx_st_global_u32(yr + c, ptx_cvt_f16x2_f32(a, b));
      else ptx_st_global_u32(yr + c, ptx_cvt_bf16x2_f32(a, b));
    }
  }
}
// the outer product mean's permute and residual (cuda/af2 and cuda/af3 each had both)
// [(bi, c), (j, e)] -> [(bi, j), (c, e)]
template <class T>
__global__ void opmPermuteK(const T* __restrict__ in, T* __restrict__ out, int Bi, int n, int O) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (std::is_same_v<T, float>) {
    if ((O & 3) == 0) {
      int O4 = O >> 2;
      size_t total4 = (size_t)Bi * n * O * O4;
      if (t >= total4) return;
      int e4, c, j, bi;
      if (O == 32 && total4 <= UINT32_MAX) {
        unsigned u = (unsigned)t;
        e4 = (int)(u & 7u);
        c = (int)((u >> 3) & 31u);
        unsigned r = u >> 8;
        bi = (int)(r / (unsigned)n);
        j = (int)(r - (unsigned)bi * (unsigned)n);
      } else if (total4 <= UINT32_MAX) {
        unsigned u = (unsigned)t, r = u / (unsigned)O4;
        e4 = (int)(u - r * (unsigned)O4);
        unsigned r2 = r / (unsigned)O;
        c = (int)(r - r2 * (unsigned)O);
        bi = (int)(r2 / (unsigned)n);
        j = (int)(r2 - (unsigned)bi * (unsigned)n);
      } else {
        e4 = (int)(t % O4); size_t r = t / O4; c = (int)(r % O); r /= O; j = (int)(r % n); bi = (int)(r / n);
      }
      ptx_st_global_v4_f32(reinterpret_cast<float4*>(out) + t,
                           ptx_ldg_v4_f32(reinterpret_cast<const float4*>(in) + ((size_t)bi * O + c) * ((size_t)n * O4) + (size_t)j * O4 + e4));
      return;
    }
  }
  size_t total = (size_t)Bi * n * O * O;
  if (t >= total) return;
  int e = (int)(t % O); size_t rest = t / O; int c = (int)(rest % O); rest /= O;
  int j = (int)(rest % n); int bi = (int)(rest / n);
  out[t] = in[((size_t)bi * O + c) * ((size_t)n * O) + (size_t)j * O + e];
}
// Coalesced shared-memory tile for O == 32: each warp loads 4 consecutive j rows (32 float4 = 512B) of one c,
// then all 256 threads write 256 contiguous float4s per j.
__global__ void __launch_bounds__(256, 3)
opmPermute32K(const float* __restrict__ in, float* __restrict__ out, int Bi, int n) {
  __shared__ float4 tile[4][32][8];
  unsigned n4 = (unsigned)((n + 3) >> 2);
  int bi = (int)(blockIdx.x / n4);
  int j0 = (int)(blockIdx.x - (unsigned)bi * n4) << 2;
  if (bi >= Bi) return;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int dj = lane >> 3, e4 = lane & 7;
  const float4* in4 = reinterpret_cast<const float4*>(in);
  float4* out4 = reinterpret_cast<float4*>(out);
  if (j0 + dj < n) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      int c = warp + m * 8;
      ptx_st_shared_v4_f32(&tile[dj][c][e4], ptx_ldg_v4_f32(in4 + ((size_t)bi * 32 + c) * ((size_t)n * 8) + (size_t)j0 * 8 + lane));
    }
  }
  __syncthreads();
  int sc = tid >> 3, se4 = tid & 7;
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    if (j0 + m < n) {
      ptx_st_global_v4_f32(out4 + ((size_t)bi * n + j0 + m) * 256 + tid, ptx_ld_shared_v4_f32(&tile[m][sc][se4]));
    }
  }
}
// eight halves (16 bytes) a thread: O is a multiple of 8, so a run of e never straddles a row
__global__ void opmPermuteHK(const half* __restrict__ Pm, half* __restrict__ X, int bi, int L, int O) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int O8 = O >> 3;
  size_t total8 = (size_t)bi * L * O * O8;
  if (t >= total8) return;
  int e8, c, j, i;
  if (O == 32 && total8 <= UINT32_MAX) {
    unsigned u = (unsigned)t;
    e8 = (int)(u & 3u);
    c = (int)((u >> 2) & 31u);
    unsigned r = u >> 7;
    i = (int)(r / (unsigned)L);
    j = (int)(r - (unsigned)i * (unsigned)L);
  } else if (total8 <= UINT32_MAX) {
    unsigned u = (unsigned)t, r = u / (unsigned)O8;
    e8 = (int)(u - r * (unsigned)O8);
    unsigned r2 = r / (unsigned)O;
    c = (int)(r - r2 * (unsigned)O);
    i = (int)(r2 / (unsigned)L);
    j = (int)(r2 - (unsigned)i * (unsigned)L);
  } else {
    e8 = (int)(t % O8); size_t r = t / O8; c = (int)(r % O); r /= O; j = (int)(r % L); i = (int)(r / L);
  }
  ptx_st_global_v4_u32(reinterpret_cast<uint4*>(X) + t,
                       ptx_ldg_v4_u32(reinterpret_cast<const uint4*>(Pm) + (((size_t)i * O + c) * ((size_t)L * O) + (size_t)j * O) / 8 + e8));
}
// pair[i][j] += (bias + x) / (1e-3 + norm[i][j])   (AF3: the bias inside the scale)
template <class PT = float>
// xs: x's scale back up (AF2's standard form contracts its rows scaled down by a power of two, to fit float16)
__global__ void opmAddK(float* pair, const float* x, const float* bias, const float* norm, size_t i0,
                        int Bi, int n, int C, bool biasAfterNorm, float xs = 1.f, float normVal = 0.f) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0) {
    size_t c4 = (size_t)(C >> 2), total4 = (size_t)Bi * n * c4;
    if (t >= total4) return;
    int f; size_t ij;
    if (C == 128) { f = (int)(t & 31) << 2; ij = t >> 5; }
    else if (C == 256) { f = (int)(t & 63) << 2; ij = t >> 6; }
    else if (total4 <= UINT32_MAX) {
      unsigned u = (unsigned)t, q = u / (unsigned)c4; f = (int)(u - q * (unsigned)c4) << 2; ij = q;
    } else { f = (int)(t % c4) << 2; ij = t / c4; }
    size_t p = i0 * n + ij, at = p * C + f;
    float nv = norm ? ptx_ldg_f32(norm + p) : normVal;
    float4 x4 = ptx_ldg_v4_f32(x + (t << 2));
    float4 b4 = ptx_ldg_v4_f32(bias + f);
    float4 v;
    if (biasAfterNorm) {
      float d = ptx_max_f32(nv, 1.f);
      v = make_float4(xs * x4.x / d + b4.x, xs * x4.y / d + b4.y, xs * x4.z / d + b4.z, xs * x4.w / d + b4.w);
    } else {
      float d = 1e-3f + nv;
      v = make_float4((b4.x + xs * x4.x) / d, (b4.y + xs * x4.y) / d, (b4.z + xs * x4.z) / d, (b4.w + xs * x4.w) / d);
    }
    if constexpr (std::is_same_v<PT, float>) {
      float4 p4 = ptx_ld_global_v4_f32(pair + at);
      ptx_st_global_v4_f32(pair + at, make_float4(p4.x + v.x, p4.y + v.y, p4.z + v.z, p4.w + v.w));
    } else if constexpr (std::is_same_v<PT, __nv_bfloat16>) {
      uint2 u2 = ptx_ld_global_v2_u32(reinterpret_cast<const PT*>(pair) + at);
      float2 q0 = ptx_bf16x2_to_f32x2(u2.x);
      float2 q1 = ptx_bf16x2_to_f32x2(u2.y);
      uint2 out2 = make_uint2(ptx_cvt_bf16x2_f32(q0.x + v.x, q0.y + v.y), ptx_cvt_bf16x2_f32(q1.x + v.z, q1.y + v.w));
      ptx_st_global_v2_u32(reinterpret_cast<PT*>(pair) + at, out2);
    } else {
      uint2 u2 = ptx_ld_global_v2_u32(reinterpret_cast<const PT*>(pair) + at);
      float2 q0 = ptx_f16x2_to_f32x2(u2.x);
      float2 q1 = ptx_f16x2_to_f32x2(u2.y);
      uint2 out2 = make_uint2(ptx_cvt_f16x2_f32(q0.x + v.x, q0.y + v.y), ptx_cvt_f16x2_f32(q1.x + v.z, q1.y + v.w));
      ptx_st_global_v2_u32(reinterpret_cast<PT*>(pair) + at, out2);
    }
    return;
  }
  const size_t total = (size_t)Bi * n * C;
  if (t >= total) return;
  int f; size_t ij;
  if (C == 128) { f = (int)(t & 127); ij = t >> 7; }
  else if (C == 256) { f = (int)(t & 255); ij = t >> 8; }
  else if (total <= UINT32_MAX) {     // (32-bit divides where the count allows: a 64-bit divide is a long emulated sequence)
    unsigned u = (unsigned)t, q = u / (unsigned)C; f = (int)(u - q * (unsigned)C); ij = q;
  } else { f = (int)(t % C); ij = t / C; }
  size_t p = i0 * n + ij;
  float nv = norm ? ptx_ldg_f32(norm + p) : normVal;
  float v = biasAfterNorm ? xs * x[t] / ptx_max_f32(nv, 1.f) + bias[f] : (bias[f] + xs * x[t]) / (1e-3f + nv);
  size_t at = p * C + f;
  if constexpr (std::is_same_v<PT, float>) pair[at] += v;
  else reinterpret_cast<PT*>(pair)[at] = fromF<PT>(toF(reinterpret_cast<PT*>(pair)[at]) + v);
}
