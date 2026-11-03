// A gradient check that needs no reference: for every tensor (a parameter or an input), the analytic gradient's
// dot product with a random direction v against the central difference (L(w + h v) - L(w - h v)) / 2h of the SAME
// float32 forward, the loss L = sum(out * ct) accumulated in double. h is a fraction of the tensor's own RMS, so
// a LayerNorm scale and a weight matrix are stepped alike.
//
// What it catches is a WRONG gradient - a missing term, a transposed operand, a stride - which moves the directional
// derivative by O(1). What it cannot resolve is float32 rounding (it reads ~1e-4..1e-3), so its bar is 1e-2, and
// a tensor whose analytic directional derivative is ~0 (a gradient that cancels exactly) is compared absolutely,
// against the largest directional derivative of the run. Each model's forward is held to its inference port or its
// reference separately; this holds the backward to that forward.
#pragma once
#include <curand.h>
#include <functional>
#include "core.cuh"

namespace grad {

__global__ void dotK(const real* a, const real* b, size_t n, double* out) {
  double s = 0;
  for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x)
    s += (double)a[t] * b[t];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}
__global__ void axpyK(real* y, const real* x, real a, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] += a * x[t];
}
inline double dot(const real* a, const real* b, size_t n) {
  static double* acc = nullptr;
  if (!acc) GK(cudaMalloc(&acc, 8));
  GK(cudaMemsetAsync(acc, 0, 8, STREAM));
  dotK<<<(unsigned)std::min<size_t>(blocks(n), 1024), 256, 0, STREAM>>>(a, b, n, acc);
  double h; GK(cudaMemcpyAsync(&h, acc, 8, cudaMemcpyDeviceToHost, STREAM));
  GK(cudaStreamSynchronize(STREAM));
  return h;
}

inline void randn(curandGenerator_t g, float* p, size_t n) { curandGenerateNormal(g, p, n, 0.f, 1.f); }
inline void randn(curandGenerator_t g, double* p, size_t n) { curandGenerateNormalDouble(g, p, n, 0.0, 1.0); }
struct Checked { std::string name; real* w; const real* g; size_t n; };
struct Output { const real* out; const real* ct; size_t n; real* plus = nullptr; };

__global__ void diffDotK(const real* a, const real* b, const real* ct, size_t n, double* out) {
  double s = 0;
  for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x)
    s += (double)(a[t] - b[t]) * ct[t];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

// The forward's outputs are differenced ELEMENTWISE, sum((out(w + h v) - out(w - h v)) ct) - never two losses of
// thousands subtracted - and the difference is Richardson-extrapolated over h and 2h, which cancels the O(h^2)
// curvature term: (4 D(h) - D(2h)) / 3.
struct GradCheck {
  std::vector<Checked> tensors;
  std::vector<Output> outputs;
#ifdef GRAD_F64
  double rel = 1e-7;        // h = rel * rms(w): small enough that no ReLU crosses its kink inside +-2h
  double bar = 1e-4;
#else
  double rel = 1e-3;        // float32: only a gross error is visible through the rounding
  double bar = 1e-1;
#endif
  unsigned long long seed = 1234;
  bool verbose = false;
  int tries = 3;
  void add(const std::string& name, real* w, const real* g, size_t n) { tensors.push_back({name, w, g, n}); }
  void add(Param& p) { add(p.name, p.w, p.g, p.n); }
  void output(const real* out, const real* ct, size_t n) { outputs.push_back({out, ct, n}); }
  // h * the directional derivative, by central difference at step h
  double central(const std::function<void()>& forward, Checked& t, const real* v, double h) {
    axpyK<<<blocks(t.n), 256, 0, STREAM>>>(t.w, v, (real)h, t.n);
    forward();
    for (auto& o : outputs) GK(cudaMemcpyAsync(o.plus, o.out, o.n * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    axpyK<<<blocks(t.n), 256, 0, STREAM>>>(t.w, v, (real)(-2 * h), t.n);
    forward();
    axpyK<<<blocks(t.n), 256, 0, STREAM>>>(t.w, v, (real)h, t.n);
    static double* acc = nullptr;
    if (!acc) GK(cudaMalloc(&acc, 8));
    GK(cudaMemsetAsync(acc, 0, 8, STREAM));
    for (auto& o : outputs)
      diffDotK<<<(unsigned)std::min<size_t>(blocks(o.n), 1024), 256, 0, STREAM>>>(o.plus, o.out, o.ct, o.n, acc);
    double r; GK(cudaMemcpyAsync(&r, acc, 8, cudaMemcpyDeviceToHost, STREAM));
    GK(cudaStreamSynchronize(STREAM));
    return r / 2;
  }
  // forward(): the model on the current values, writing every registered output. Returns the tensors above the bar.
  int run(const std::function<void()>& forward) {
    for (auto& o : outputs) GK(cudaMalloc(&o.plus, o.n * sizeof(real)));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    struct Row { std::string name; double an, fd; };
    std::vector<Row> rows;
    double scale = 0;
    // Up to `tries` directions a tensor, the first within the bar kept: a ReLU whose input sits within h of its
    // kink spoils ONE direction's difference, where a wrong gradient is wrong along every direction.
    for (auto& t : tensors) {
      size_t n2 = (t.n + 1) & ~(size_t)1;
      real* v; GK(cudaMalloc(&v, n2 * sizeof(real)));
      double rms = std::sqrt(dot(t.w, t.w, t.n) / t.n);
      double h = rel * (rms > 0 ? rms : 1.0);
      Row best{t.name, 0, 0};
      double bestE = INFINITY;
      for (int k = 0; k < tries; ++k) {
        randn(gen, v, n2);
        double an = h * dot(t.g, v, t.n);
        double d1 = central(forward, t, v, h), d2 = central(forward, t, v, 2 * h) / 2;
        double fd = (4 * d1 - d2) / 3;
        double e = std::fabs(an - fd) / std::max({std::fabs(an), std::fabs(fd), 1e-300});
        if (k == 0 || e < bestE) { bestE = e; best = {t.name, an, fd}; }
        if (e <= bar) break;
      }
      GK(cudaFree(v));
      rows.push_back(best);
      scale = std::max(scale, std::max(std::fabs(best.an), std::fabs(best.fd)));
    }
    curandDestroyGenerator(gen);
    for (auto& o : outputs) { GK(cudaFree(o.plus)); o.plus = nullptr; }
    int failed = 0;
    double worst = 0; std::string worstName;
    for (auto& r : rows) {
      double den = std::max({std::fabs(r.an), std::fabs(r.fd), 1e-3 * scale});
      double e = std::fabs(r.an - r.fd) / den;
      if (!(e <= bar)) ++failed;
      if (!(e <= worst)) { worst = std::isnan(e) ? INFINITY : e; worstName = r.name; }
      if (verbose || !(e <= bar))
        printf("  %-64s analytic %+.6e  difference %+.6e  %.2e%s\n", r.name.c_str(), r.an, r.fd, e, e <= bar ? "" : "  <-- ABOVE BAR");
    }
    printf("  gradcheck: %zu tensors, worst %.2e (%s), %d above %.0e\n", rows.size(), worst, worstName.c_str(), failed, bar);
    return failed;
  }
};

// sum(out * ct) in double, for a loss closure
inline double lossDot(const real* out, const real* ct, size_t n) { return dot(out, ct, n); }

}  // namespace grad
