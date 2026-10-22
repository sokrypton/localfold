// Element-wise and row kernels every port had its own copy of (cuda/af2, cuda/ef2, cuda/af3 - the same
// computation under the same name, renamed variables apart), here once. Included by common.cuh.
#pragma once

__global__ void addK(float* __restrict__ y, const float* __restrict__ x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((n & 3) == 0 && (((uintptr_t)y | (uintptr_t)x) & 15) == 0) {
    size_t n4 = n >> 2;
    if (t < n4) {
      float4 yv = reinterpret_cast<const float4*>(y)[t];
      float4 xv = reinterpret_cast<const float4*>(x)[t];
      reinterpret_cast<float4*>(y)[t] = make_float4(yv.x + xv.x, yv.y + xv.y, yv.z + xv.z, yv.w + xv.w);
    }
    return;
  }
  if (t < n) y[t] += x[t];
}

__global__ void addBiasK(float* __restrict__ y, const float* __restrict__ b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((C & 3) == 0 && (((uintptr_t)y | (uintptr_t)b) & 15) == 0) {
    int c4 = C >> 2;
    size_t n4 = rows * (size_t)c4;
    if (t >= n4) return;
    int c = ((c4 & (c4 - 1)) == 0) ? (int)(t & (size_t)(c4 - 1)) : (n4 <= UINT32_MAX ? (int)((unsigned)t % (unsigned)c4) : (int)(t % c4));
    float4 yv = reinterpret_cast<const float4*>(y)[t];
    float4 bv = __ldg(reinterpret_cast<const float4*>(b) + c);
    reinterpret_cast<float4*>(y)[t] = make_float4(yv.x + bv.x, yv.y + bv.y, yv.z + bv.z, yv.w + bv.w);
    return;
  }
  size_t n = rows * C;
  if (t >= n) return;
  int c = ((C & (C - 1)) == 0) ? (int)(t & (size_t)(C - 1)) : (n <= UINT32_MAX ? (int)((unsigned)t % (unsigned)C) : (int)(t % C));
  y[t] += __ldg(b + c);
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
  out[t] = z[t] + z[((size_t)j * T + i) * C + c];
}

// rows [r0, r0 + R) of z + z^T (the whole: symmetriseK)
__global__ void symmetriseRowsK(const float* z, float* out, int T, int C, size_t r0, size_t R) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= R * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; size_t i = r0 + ij / T, j = ij % T;
  out[t] = z[(i * T + j) * C + c] + z[(j * T + i) * C + c];
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
      float4 p = reinterpret_cast<const float4*>(pair)[t];
      float4 o = __ldg(reinterpret_cast<const float4*>(out) + t);
      float4 g = __ldg(reinterpret_cast<const float4*>(gate) + (ld != C ? r * (size_t)ld4 + c : t));
      if (outB) { float4 ob = __ldg(reinterpret_cast<const float4*>(outB) + c); o.x += ob.x; o.y += ob.y; o.z += ob.z; o.w += ob.w; }
      if (gateB) { float4 gb = __ldg(reinterpret_cast<const float4*>(gateB) + c); g.x += gb.x; g.y += gb.y; g.z += gb.z; g.w += gb.w; }
      p.x += o.x / (1.f + __expf(-g.x));
      p.y += o.y / (1.f + __expf(-g.y));
      p.z += o.z / (1.f + __expf(-g.z));
      p.w += o.w / (1.f + __expf(-g.w));
      reinterpret_cast<float4*>(pair)[t] = p;
    }
    return;
  }
  if (t < n) {
    int c = C ? (int)(t % C) : 0;
    size_t gIdx = (ld != C && C) ? (t / C) * (size_t)ld + c : t;
    float ov = out[t] + (outB ? __ldg(outB + c) : 0.f);
    float gv = gate[gIdx] + (gateB ? __ldg(gateB + c) : 0.f);
    pair[t] += ov / (1.f + __expf(-gv));
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
    float4 p = reinterpret_cast<const float4*>(pair)[t];
    float4 o = __ldg(reinterpret_cast<const float4*>(out) + t);
    uint2 g2 = __ldg(reinterpret_cast<const uint2*>(gate) + r * (size_t)ld4 + c);
    float2 g0 = __half22float2(*reinterpret_cast<const half2*>(&g2.x));
    float2 g1 = __half22float2(*reinterpret_cast<const half2*>(&g2.y));
    p.x += o.x / (1.f + __expf(-g0.x));
    p.y += o.y / (1.f + __expf(-g0.y));
    p.z += o.z / (1.f + __expf(-g1.x));
    p.w += o.w / (1.f + __expf(-g1.y));
    reinterpret_cast<float4*>(pair)[t] = p;
    return;
  }
  if (t >= pairs * C) return;
  size_t r = t / C; int c = (int)(t % C);
  pair[t] += out[t] / (1.f + __expf(-__half2float(gate[r * ld + c])));
}

__global__ void channelMajorToRowsK(const float* x, float* y, size_t pairs, int C) {   // [c][ij] -> [ij][c]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < pairs * C) y[t] = x[(t % C) * pairs + t / C];
}

__global__ void __launch_bounds__(256, 8)
channelMajorToRowsTileK(const float* __restrict__ x, float* __restrict__ y, size_t pairs, int C) {
  __shared__ float tile[32][33];
  size_t ij0 = (size_t)blockIdx.x * 32;
  int c0 = (int)blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
  size_t ij = ij0 + tx;
  if (ij0 + 32 <= pairs && c0 + 32 <= C) {
    float v0 = __ldg(x + (size_t)(c0 + ty) * pairs + ij);
    float v1 = __ldg(x + (size_t)(c0 + ty + 8) * pairs + ij);
    float v2 = __ldg(x + (size_t)(c0 + ty + 16) * pairs + ij);
    float v3 = __ldg(x + (size_t)(c0 + ty + 24) * pairs + ij);
    tile[ty][tx] = v0; tile[ty + 8][tx] = v1; tile[ty + 16][tx] = v2; tile[ty + 24][tx] = v3;
    __syncthreads();
    int c = c0 + tx;
    size_t r = ij0 + ty;
    y[r * C + c] = tile[tx][ty];
    y[(r + 8) * C + c] = tile[tx][ty + 8];
    y[(r + 16) * C + c] = tile[tx][ty + 16];
    y[(r + 24) * C + c] = tile[tx][ty + 24];
    return;
  }
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    int c = c0 + ty + k;
    if (c < C && ij < pairs) tile[ty + k][tx] = __ldg(x + (size_t)c * pairs + ij);
  }
  __syncthreads();
  int c = c0 + tx;
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    size_t r = ij0 + ty + k;
    if (r < pairs && c < C) y[r * C + c] = tile[tx][ty + k];
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
        tile[c * 33 + lane] = __ldg(pBase + (size_t)c * plane);
      }
    }
    __syncthreads();
    float sc[8], of[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      int c = k * 32 + lane;
      sc[k] = __ldg(scale + c);
      of[k] = __ldg(offset + c);
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
        float val = tile[(k * 32 + lane) * 33 + row];
        tv[k] = val;
        s += val;
      }
      #pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
      float mean = s * (1.f / 256.f), v = 0;
      #pragma unroll
      for (int k = 0; k < 8; ++k) { float d = tv[k] - mean; v += d * d; }
      #pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      float inv = rsqrtf(v * (1.f / 256.f) + 1e-5f);
      TY* outRow = out + r * 256;
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        float yv = (tv[k] - mean) * inv * sc[k] + of[k];
        if constexpr (std::is_same_v<TY, half>) outRow[k * 32 + lane] = __float2half(yv);
        else outRow[k * 32 + lane] = yv;
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
        tile[c * 33 + lane] = __ldg(pBase + (size_t)c * plane);
      }
    }
    __syncthreads();
    float sc[4], of[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      int c = k * 32 + lane;
      sc[k] = __ldg(scale + c);
      of[k] = __ldg(offset + c);
    }
    #pragma unroll
    for (int rk = 0; rk < 4; ++rk) {
      int row = rk * 8 + warp;
      size_t r = r0 + row;
      if (r >= pairs) break;
      float tv[4];
      float s = 0;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        float val = tile[(k * 32 + lane) * 33 + row];
        tv[k] = val;
        s += val;
      }
      #pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
      float mean = s * (1.f / 128.f), v = 0;
      #pragma unroll
      for (int k = 0; k < 4; ++k) { float d = tv[k] - mean; v += d * d; }
      #pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      float inv = rsqrtf(v * (1.f / 128.f) + 1e-5f);
      TY* outRow = out + r * 128;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        float yv = (tv[k] - mean) * inv * sc[k] + of[k];
        if constexpr (std::is_same_v<TY, half>) outRow[k * 32 + lane] = __float2half(yv);
        else outRow[k * 32 + lane] = yv;
      }
    }
    return;
  }
  for (int c = warp; c < C; c += nw) {
    tile[c * 33 + lane] = rLoad < pairs ? __ldg(prod + (size_t)c * plane + planeIdx) : 0.f;
  }
  __syncthreads();
  for (int row = warp; row < 32; row += nw) {
    size_t r = r0 + row;
    if (r >= pairs) break;
    float s = 0;
    for (int c = lane; c < C; c += 32) s += tile[c * 33 + row];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    float mean = s / C, v = 0;
    for (int c = lane; c < C; c += 32) { float d = tile[c * 33 + row] - mean; v += d * d; }
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    float inv = rsqrtf(v / C + 1e-5f);
    for (int c = lane; c < C; c += 32) {
      float yv = (tile[c * 33 + row] - mean) * inv * scale[c] + offset[c];
      if constexpr (std::is_same_v<TY, half>) out[r * C + c] = __float2half(yv);
      else out[r * C + c] = yv;
    }
  }
}

// TI, TO: the rows' and the output's types (a bf16 MSA in, a bf16 transition input out: cuda/af2's MSA16)
inline unsigned lnVGrid(size_t rows) { return (unsigned)((rows + 7) / 8); }
template <int C, class TI = float, class TO = half, bool RELU = false>
__global__ void __launch_bounds__(256, 8)
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
      float4 s4 = __ldg(reinterpret_cast<const float4*>(scale + c)), o4 = __ldg(reinterpret_cast<const float4*>(offset + c));
      sc[k][0] = s4.x; sc[k][1] = s4.y; sc[k][2] = s4.z; sc[k][3] = s4.w;
      of[k][0] = o4.x; of[k][1] = o4.y; of[k][2] = o4.z; of[k][3] = o4.w;
    } else {
      float2 s2 = __ldg(reinterpret_cast<const float2*>(scale + c)), o2 = __ldg(reinterpret_cast<const float2*>(offset + c));
      sc[k][0] = s2.x; sc[k][1] = s2.y; of[k][0] = o2.x; of[k][1] = o2.y;
    }
    if constexpr (!std::is_same_v<TI, float>) {          // (a bf16 or f16 row: VEC elements, 2 VEC bytes)
      if constexpr (VEC == 4) {
        uint2 u2 = __ldg(reinterpret_cast<const uint2*>(xr + c));
        const TI* p = reinterpret_cast<const TI*>(&u2);
        float2 q0 = ld2f(p), q1 = ld2f(p + 2);
        v[k][0] = q0.x; v[k][1] = q0.y; v[k][2] = q1.x; v[k][3] = q1.y;
      } else {
        float2 q = ld2f(xr + c); v[k][0] = q.x; v[k][1] = q.y;
      }
    } else if constexpr (VEC == 4) { float4 q = __ldg(reinterpret_cast<const float4*>(xr + c)); v[k][0] = q.x; v[k][1] = q.y; v[k][2] = q.z; v[k][3] = q.w; }
    else { float2 q = __ldg(reinterpret_cast<const float2*>(xr + c)); v[k][0] = q.x; v[k][1] = q.y; }
#pragma unroll
    for (int u = 0; u < VEC; ++u) s += v[k][u];
  }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, q2 = 0;
#pragma unroll
  for (int k = 0; k < STEPS; ++k)
#pragma unroll
    for (int u = 0; u < VEC; ++u) { float d = v[k][u] - mean; q2 += d * d; }
  for (int o = 16; o; o >>= 1) q2 += __shfl_xor_sync(~0u, q2, o);
  float inv = rsqrtf(q2 / C + 1e-5f);
#pragma unroll
  for (int k = 0; k < STEPS; ++k) {
    int c = (k * 32 + lane) * VEC;
    if constexpr (VEC == 4) {
      float a0 = (v[k][0] - mean) * inv * sc[k][0] + of[k][0], b0 = (v[k][1] - mean) * inv * sc[k][1] + of[k][1];
      float a1 = (v[k][2] - mean) * inv * sc[k][2] + of[k][2], b1 = (v[k][3] - mean) * inv * sc[k][3] + of[k][3];
      if constexpr (RELU) { a0 = fmaxf(a0, 0.f); b0 = fmaxf(b0, 0.f); a1 = fmaxf(a1, 0.f); b1 = fmaxf(b1, 0.f); }
      uint2 out2;
      if constexpr (std::is_same_v<TO, half>) {
        half2 h0 = __floats2half2_rn(a0, b0), h1 = __floats2half2_rn(a1, b1);
        out2.x = *reinterpret_cast<uint32_t*>(&h0); out2.y = *reinterpret_cast<uint32_t*>(&h1);
      } else {
        __nv_bfloat162 h0 = __floats2bfloat162_rn(a0, b0), h1 = __floats2bfloat162_rn(a1, b1);
        out2.x = *reinterpret_cast<uint32_t*>(&h0); out2.y = *reinterpret_cast<uint32_t*>(&h1);
      }
      *reinterpret_cast<uint2*>(y + row * C + c) = out2;
    } else {
      float a = (v[k][0] - mean) * inv * sc[k][0] + of[k][0], b = (v[k][1] - mean) * inv * sc[k][1] + of[k][1];
      if constexpr (RELU) { a = fmaxf(a, 0.f); b = fmaxf(b, 0.f); }
      if constexpr (std::is_same_v<TO, half>) *reinterpret_cast<half2*>(y + row * C + c) = __floats2half2_rn(a, b);
      else *reinterpret_cast<__nv_bfloat162*>(y + row * C + c) = __floats2bfloat162_rn(a, b);
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
      reinterpret_cast<float4*>(out)[t] =
          __ldg(reinterpret_cast<const float4*>(in) + ((size_t)bi * O + c) * ((size_t)n * O4) + (size_t)j * O4 + e4);
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
      tile[dj][c][e4] = __ldg(in4 + ((size_t)bi * 32 + c) * ((size_t)n * 8) + (size_t)j0 * 8 + lane);
    }
  }
  __syncthreads();
  int sc = tid >> 3, se4 = tid & 7;
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    if (j0 + m < n) {
      out4[((size_t)bi * n + j0 + m) * 256 + tid] = tile[m][sc][se4];
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
  reinterpret_cast<uint4*>(X)[t] =
      reinterpret_cast<const uint4*>(Pm)[(((size_t)i * O + c) * ((size_t)L * O) + (size_t)j * O) / 8 + e8];
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
    float nv = norm ? norm[p] : normVal;
    float4 x4 = *reinterpret_cast<const float4*>(x + (t << 2));
    float4 b4 = *reinterpret_cast<const float4*>(bias + f);
    float4 v;
    if (biasAfterNorm) {
      float d = fmaxf(nv, 1.f);
      v = make_float4(xs * x4.x / d + b4.x, xs * x4.y / d + b4.y, xs * x4.z / d + b4.z, xs * x4.w / d + b4.w);
    } else {
      float d = 1e-3f + nv;
      v = make_float4((b4.x + xs * x4.x) / d, (b4.y + xs * x4.y) / d, (b4.z + xs * x4.z) / d, (b4.w + xs * x4.w) / d);
    }
    if constexpr (std::is_same_v<PT, float>) {
      float4 p4 = *reinterpret_cast<const float4*>(pair + at);
      *reinterpret_cast<float4*>(pair + at) = make_float4(p4.x + v.x, p4.y + v.y, p4.z + v.z, p4.w + v.w);
    } else if constexpr (std::is_same_v<PT, __nv_bfloat16>) {
      uint2 u2 = *reinterpret_cast<const uint2*>(reinterpret_cast<const PT*>(pair) + at);
      float2 q0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.x));
      float2 q1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.y));
      __nv_bfloat162 o0 = __floats2bfloat162_rn(q0.x + v.x, q0.y + v.y);
      __nv_bfloat162 o1 = __floats2bfloat162_rn(q1.x + v.z, q1.y + v.w);
      *reinterpret_cast<uint2*>(reinterpret_cast<PT*>(pair) + at) = make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));
    } else {
      uint2 u2 = *reinterpret_cast<const uint2*>(reinterpret_cast<const PT*>(pair) + at);
      float2 q0 = __half22float2(*reinterpret_cast<const half2*>(&u2.x));
      float2 q1 = __half22float2(*reinterpret_cast<const half2*>(&u2.y));
      half2 o0 = __floats2half2_rn(q0.x + v.x, q0.y + v.y);
      half2 o1 = __floats2half2_rn(q1.x + v.z, q1.y + v.w);
      *reinterpret_cast<uint2*>(reinterpret_cast<PT*>(pair) + at) = make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));
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
  float nv = norm ? norm[p] : normVal;
  float v = biasAfterNorm ? xs * x[t] / fmaxf(nv, 1.f) + bias[f] : (bias[f] + xs * x[t]) / (1e-3f + nv);
  size_t at = p * C + f;
  if constexpr (std::is_same_v<PT, float>) pair[at] += v;
  else reinterpret_cast<PT*>(pair)[at] = fromF<PT>(toF(reinterpret_cast<PT*>(pair)[at]) + v);
}
