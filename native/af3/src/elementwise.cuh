// Element-wise and row kernels every port had its own copy of (native/af2, native/ef2, native/af3 - the same
// computation under the same name, renamed variables apart), here once. Included by common.cuh.
#pragma once

__global__ void addK(float* y, const float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += x[t];
}

__global__ void addBiasK(float* y, const float* b, size_t rows, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) y[t] += b[t % C];
}

__global__ void symmetriseK(const float* z, float* out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  out[t] = z[t] + z[((size_t)j * T + i) * C + c];
}

__global__ void gateMulAddK(float* pair, const float* out, const float* gate, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) pair[t] += out[t] / (1.f + __expf(-gate[t]));
}

__global__ void gateMulAddHK(float* pair, const float* out, const half* gate, size_t pairs, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= pairs * C) return;
  size_t r = t / C; int c = (int)(t % C);
  pair[t] += out[t] / (1.f + __expf(-__half2float(gate[r * ld + c])));
}

__global__ void channelMajorToRowsK(const float* x, float* y, size_t pairs, int C) {   // [c][ij] -> [ij][c]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < pairs * C) y[t] = x[(t % C) * pairs + t / C];
}

__global__ void centerNormHK(const float* prod, half* out, size_t pairs, int C, const float* scale, const float* offset,
                             int L, int Lp) {
  extern __shared__ float tile[];               // [C][33]
  size_t r0 = (size_t)blockIdx.x * 32;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int c = warp; c < C; c += nw) {
    size_t r = r0 + lane;
    tile[c * 33 + lane] = r < pairs ? prod[(size_t)c * Lp * Lp + (r / L) * Lp + r % L] : 0.f;
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
    for (int c = lane; c < C; c += 32) out[r * C + c] = __float2half((tile[c * 33 + row] - mean) * inv * scale[c] + offset[c]);
  }
}

template <int C>
__global__ void layerNormVK(const float* __restrict__ x, half* __restrict__ y, size_t rows, const float* __restrict__ scale,
                            const float* __restrict__ offset) {
  constexpr int VEC = C >= 128 ? 4 : C / 32, STEPS = C / (32 * VEC);
  size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + row * C;
  float v[STEPS][VEC];
  float s = 0;
#pragma unroll
  for (int k = 0; k < STEPS; ++k) {
    int c = (k * 32 + lane) * VEC;
    if constexpr (VEC == 4) { float4 q = *reinterpret_cast<const float4*>(xr + c); v[k][0] = q.x; v[k][1] = q.y; v[k][2] = q.z; v[k][3] = q.w; }
    else { float2 q = *reinterpret_cast<const float2*>(xr + c); v[k][0] = q.x; v[k][1] = q.y; }
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
#pragma unroll
    for (int u = 0; u < VEC; u += 2)
      *reinterpret_cast<half2*>(y + row * C + c + u) =
          __floats2half2_rn((v[k][u] - mean) * inv * scale[c + u] + offset[c + u],
                            (v[k][u + 1] - mean) * inv * scale[c + u + 1] + offset[c + u + 1]);
  }
}
