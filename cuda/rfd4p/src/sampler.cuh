// RFD4-Proteina's flow-matching sampler: the schedules, the noise and one Euler(-Maruyama) step in
// any of upstream's three modes. Ported from `shared/rfd4p/flow.js`, which is what the page runs.
//
// The schedules are host arithmetic over `nsteps` doubles and cost nothing; the step is a kernel
// because x lives on the device between trunk passes.
#pragma once
#include "trunk.cuh"
#include "schedule.h"

namespace rfd4p {

// ---------------------------------------------------------------- the step
//
// `x1 = x + (1 - t) v` is the clean estimate the decoder sees; `x` advances by the Euler-Maruyama
// term. Both are masked to the emitting tokens, and a masked row is left exactly zero rather than
// merely small - the reference writes zero there and a residual would accumulate over sixty steps.
// 🔴 THE INNER ARITHMETIC IS DOUBLE AND THE STORES ARE FLOAT, which is what the reference does and
// is not pedantry here: at the first step t is 0.001, and `x + (1 - t) * ss` subtracts two numbers
// that agree to three digits before dividing by t. In float that cancellation keeps four digits of
// a value the next step multiplies up; in double it keeps thirteen. `f32` at each store is where
// the reference rounds, so the port rounds in the same places.
// `mode`: 0 `vf_tsr`, 1 `vf`, 2 `sc` (flow.js `simulationStep`, which says what each is). `r` is the
// score's rescaling - the tsr ratio for `vf_tsr`, 1.5 on `sc`'s ODE tail.
__global__ void simulationStepK(float* __restrict__ x, const float* __restrict__ v,
                                float* __restrict__ x1, const float* __restrict__ eps,
                                const float* __restrict__ emit, size_t count, int dim, double t,
                                double dt, double gt, double r, double std_, int stochastic, int mode) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const bool live = emit[i / dim] != 0.f;
  const float xi = x[i], vi = v[i];
  x1[i] = live ? (float)(xi + (float)((1.0 - t) * vi)) : 0.f;
  if (mode == 0 && t == 0.0) return;
  double delta;
  if (mode == 1) {
    delta = vi * dt;
  } else {
    const double score = (t * vi - xi) / (1.0 - t);
    if (mode == 2 && stochastic) {
      delta = (vi + gt * score) * dt + std_ * eps[i];
    } else {
      const double ss = score * r;
      const double vs = (xi + (1.0 - t) * ss) / t;
      delta = stochastic ? (vs + gt * ss) * dt + std_ * eps[i] : vs * dt;
    }
  }
  x[i] = live ? (float)(xi + delta) : 0.f;
}

// `center_every_step`: `(x - mean_w_mask(x, emit)) * emit` - the mean over the emitted tokens in
// double, rounded once to float as flow.js rounds it, then every row moved by it.
__global__ void centerEmittedK(float* __restrict__ x, const float* __restrict__ emit, int n, int dim) {
  __shared__ double sum[256];
  __shared__ int cnt[256];
  const int d = blockIdx.x;
  double acc = 0;
  int c = 0;
  for (int tok = threadIdx.x; tok < n; tok += blockDim.x) {
    if (emit[tok] != 0.f) { acc += x[(size_t)tok * dim + d]; ++c; }
  }
  sum[threadIdx.x] = acc;
  cnt[threadIdx.x] = c;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) { sum[threadIdx.x] += sum[threadIdx.x + s]; cnt[threadIdx.x] += cnt[threadIdx.x + s]; }
    __syncthreads();
  }
  const float mean = (float)(sum[0] / (double)(cnt[0] > 0 ? cnt[0] : 1));
  for (int tok = threadIdx.x; tok < n; tok += blockDim.x) {
    float& xv = x[(size_t)tok * dim + d];
    xv = emit[tok] != 0.f ? (float)(xv - mean) : 0.f;
  }
}

inline void simulationStep(float* x, const float* v, float* x1, const float* eps, const float* emit,
                           size_t count, int dim, double t, double dt, double gt,
                           const StepCfg& step) {
  const int mode = step.mode == "vf" ? 1 : step.mode == "sc" ? 2 : 0;
  const double r = mode == 0 ? tsrRatio(t, step.tsrK, step.tsrSigma) : 1.5;
  const bool stochastic = isStochastic(t, step);
  const double sd = stochastic ? std::sqrt(2 * gt * step.scScaleNoise * dt) : 0.0;
  simulationStepK<<<blocks(count), 256, 0, STREAM>>>(x, v, x1, eps, emit, count, dim, t, dt, gt, r,
                                                     sd, stochastic ? 1 : 0, mode);
  if (step.center) centerEmittedK<<<dim, 256, 0, STREAM>>>(x, emit, (int)(count / dim), dim);
}

}  // namespace rfd4p
