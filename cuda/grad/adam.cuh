// AdamW over a Params set: one fused pass a tensor (moments, bias correction, decoupled weight decay, the update),
// with an optional clip by the global gradient norm, computed on the device and read back once a step.
// Float32 moments; 8 bytes a parameter beside the 8 of value and gradient.
#pragma once
#include "core.cuh"

namespace grad {

__global__ void sqSumK(const real* __restrict__ g, size_t n, double* out) {
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

__global__ void adamwK(real* __restrict__ w, const real* __restrict__ g, real* __restrict__ m, real* __restrict__ v,
                       size_t n, real lr, real b1, real b2, real eps, real wd, real c1, real c2, real gscale) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real gt = g[t] * gscale;
  real mt = b1 * m[t] + (1.f - b1) * gt, vt = b2 * v[t] + (1.f - b2) * gt * gt;
  m[t] = mt; v[t] = vt;
  w[t] -= lr * ((mt / c1) / (sqrt(vt / c2) + eps) + wd * w[t]);
}

struct AdamW {
  real lr = 1e-4f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.f, clip = 0.f;   // clip: max global norm, 0 off
  long step = 0;
  std::vector<real*> m, v;
  double* norm = nullptr;
  double lastNorm = 0;
  void init(const Params& ps) {
    for (auto* p : ps.all) {
      real *a, *b;
      GK(cudaMalloc(&a, p->n * sizeof(real))); GK(cudaMalloc(&b, p->n * sizeof(real)));
      GK(cudaMemset(a, 0, p->n * sizeof(real))); GK(cudaMemset(b, 0, p->n * sizeof(real)));
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
    real scale = 1.f;
    if (clip > 0) { lastNorm = gradNorm(ps); if (lastNorm > clip) scale = (real)(clip / lastNorm); }
    real c1 = 1.f - pow(b1, (real)step), c2 = 1.f - pow(b2, (real)step);
    for (size_t i = 0; i < ps.all.size(); ++i) {
      Param* p = ps.all[i];
      adamwK<<<blocks(p->n), 256, 0, STREAM>>>(p->w, p->g, m[i], v[i], p->n, lr, b1, b2, eps, wd, c1, c2, scale);
    }
  }
};

}  // namespace grad
