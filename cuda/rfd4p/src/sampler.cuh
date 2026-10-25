// RFD4-Proteina's flow-matching sampler: the schedules, the noise and one `vf_tsr` Euler-Maruyama
// step. Ported from `shared/rfd4p/flow.js`, which is what the page runs.
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
__global__ void simulationStepK(float* __restrict__ x, const float* __restrict__ v,
                                float* __restrict__ x1, const float* __restrict__ eps,
                                const float* __restrict__ emit, size_t count, int dim, double t,
                                double dt, double gt, double r, double std_, int stochastic) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const bool live = emit[i / dim] != 0.f;
  const float xi = x[i], vi = v[i];
  x1[i] = live ? (float)(xi + (float)((1.0 - t) * vi)) : 0.f;
  if (t == 0.0) return;
  const double score = (t * vi - xi) / (1.0 - t);
  const double ss = score * r;
  const double vs = (xi + (1.0 - t) * ss) / t;
  const double delta = stochastic ? (vs + gt * ss) * dt + std_ * eps[i] : vs * dt;
  x[i] = live ? (float)(xi + delta) : 0.f;
}

inline void simulationStep(float* x, const float* v, float* x1, const float* eps, const float* emit,
                           size_t count, int dim, double t, double dt, double gt,
                           const StepCfg& step) {
  const double r = tsrRatio(t, step.tsrK, step.tsrSigma);
  const bool stochastic = t != 0.0 && !(t > step.tLimOde || t < step.tLimOdeBelow);
  const double sd = stochastic ? std::sqrt(2 * gt * step.scScaleNoise * dt) : 0.0;
  simulationStepK<<<blocks(count), 256, 0, STREAM>>>(x, v, x1, eps, emit, count, dim, t, dt, gt, r,
                                                     sd, stochastic ? 1 : 0);
}

}  // namespace rfd4p
