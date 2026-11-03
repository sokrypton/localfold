// AdamW over a Params set: one fused pass a tensor (moments, bias correction, decoupled weight decay, the update),
// with an optional clip by the global gradient norm, computed on the device and read back once a step.
// Float32 moments; 8 bytes a parameter beside the 8 of value and gradient.
#pragma once
#include "core.cuh"

namespace grad {

__global__ void sqSumK(const float* __restrict__ g, size_t n, double* out) {
  double s = 0;
  for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    double v = g[t]; s += v * v;
  }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  __shared__ double sh[32];
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    s = threadIdx.x < blockDim.x / 32 ? sh[threadIdx.x] : 0;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    if (threadIdx.x == 0) atomicAdd(out, s);
  }
}

__global__ void adamwK(float* __restrict__ w, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v,
                       size_t n, float lr, float b1, float b2, float eps, float wd, float c1, float c2, float gscale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  float gt = g[t] * gscale;
  float mt = b1 * m[t] + (1.f - b1) * gt, vt = b2 * v[t] + (1.f - b2) * gt * gt;
  m[t] = mt; v[t] = vt;
  w[t] -= lr * ((mt / c1) / (sqrtf(vt / c2) + eps) + wd * w[t]);
}

struct AdamW {
  float lr = 1e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.f, clip = 0.f;   // clip: max global norm, 0 off
  long step = 0;
  std::vector<float*> m, v;
  double* norm = nullptr;
  double lastNorm = 0;
  void init(const Params& ps) {
    for (auto* p : ps.all) {
      float *a, *b;
      GK(cudaMalloc(&a, p->n * 4)); GK(cudaMalloc(&b, p->n * 4));
      GK(cudaMemset(a, 0, p->n * 4)); GK(cudaMemset(b, 0, p->n * 4));
      m.push_back(a); v.push_back(b);
    }
    GK(cudaMalloc(&norm, sizeof(double)));
  }
  // the global norm of the gradients (before clipping)
  double gradNorm(const Params& ps) {
    GK(cudaMemsetAsync(norm, 0, sizeof(double), STREAM));
    for (auto* p : ps.all) sqSumK<<<(unsigned)std::min<size_t>(blocks(p->n), 1024), 256, 0, STREAM>>>(p->g, p->n, norm);
    double h; GK(cudaMemcpyAsync(&h, norm, sizeof(double), cudaMemcpyDeviceToHost, STREAM));
    GK(cudaStreamSynchronize(STREAM));
    return std::sqrt(h);
  }
  void update(Params& ps) {
    if (m.empty()) init(ps);
    ++step;
    float scale = 1.f;
    if (clip > 0) { lastNorm = gradNorm(ps); if (lastNorm > clip) scale = (float)(clip / lastNorm); }
    float c1 = 1.f - powf(b1, (float)step), c2 = 1.f - powf(b2, (float)step);
    for (size_t i = 0; i < ps.all.size(); ++i) {
      Param* p = ps.all[i];
      adamwK<<<blocks(p->n), 256, 0, STREAM>>>(p->w, p->g, m[i], v[i], p->n, lr, b1, b2, eps, wd, c1, c2, scale);
    }
  }
};

}  // namespace grad
