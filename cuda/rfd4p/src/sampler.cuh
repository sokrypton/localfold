// RFD4-Proteina's flow-matching sampler: the schedules, the noise and one `vf_tsr` Euler-Maruyama
// step. Ported from `shared/rfd4p/flow.js`, which is what the page runs.
//
// The schedules are host arithmetic over `nsteps` doubles and cost nothing; the step is a kernel
// because x lives on the device between trunk passes.
#pragma once
#include <vector>
#include "trunk.cuh"
#include "../../featurise/jsmath.h"

namespace rfd4p {

// ---------------------------------------------------------------- the random source
//
// 🔴 THE WHOLE STREAM IS NOW BIT-EXACT AGAINST V8, INTEGERS AND GAUSSIANS ALIKE. `seededRng` is
// xorshift128+ on uint64, which C++ reproduces trivially; Box-Muller then puts that through
// `Math.log`, `Math.cos` and `Math.sin`, and `cuda/featurise/jsmath.h` carries V8's fdlibm for all
// three - 0 of 200,000 arguments apart. glibc is NOT a substitute: its own log and exp differ from
// V8 on 2.6% and 7.2% of the same arguments, and one ulp in one draw is nothing in a sample and
// everything in a sixty-step trajectory.
//
// A case that carries `g/noise` still REPLAYS the exporter's draws rather than redrawing them.
// That is not because the stream is doubted: a replay is what lets a trajectory be compared when
// the two sides disagree about how many draws a step takes, which is a wiring question and the one
// this gate exists to answer.
struct Rng {
  uint64_t s0, s1;
  explicit Rng(uint64_t seed = 0) {
    s0 = seed * 0x9e3779b97f4a7c15ull + 1ull;
    s1 = s0 ^ 0xbf58476d1ce4e5b9ull;
  }
  double next() {
    uint64_t a = s0;
    const uint64_t b = s1;
    s0 = b;
    a ^= a << 23;
    s1 = a ^ b ^ (a >> 17) ^ (b >> 26);
    // JS takes the low 53 bits of (s1 + b) and divides by 2^53.
    return (double)((s1 + b) & ((1ull << 53) - 1)) / 9007199254740992.0;
  }
  void randn(float* out, size_t count) {
    for (size_t i = 0; i < count; i += 2) {
      const double u = std::max(next(), 1e-300);
      const double w = next();
      const double r = std::sqrt(-2.0 * lf::jsmath::log(u));
      out[i] = (float)(r * lf::jsmath::cos(2.0 * M_PI * w));
      if (i + 1 < count) out[i + 1] = (float)(r * lf::jsmath::sin(2.0 * M_PI * w));
    }
  }
};

// ---------------------------------------------------------------- the time schedule
inline std::vector<double> linspace(double a, double b, int k) {
  std::vector<double> out(k);
  for (int i = 0; i < k; ++i) out[i] = k == 1 ? a : a + (b - a) * i / (k - 1);
  return out;
}

// `get_schedule_tsr_safe` without the tail multiplier: nsteps + 1 points.
inline std::vector<double> baseSchedule(const std::string& mode, int nsteps, double p,
                                        double eps = 1e-3) {
  if (mode == "uniform") return linspace(eps, 1.0, nsteps + 1);
  if (mode == "power") {
    std::vector<double> t = linspace(std::pow(eps, 1.0 / p), 1.0, nsteps + 1);
    for (double& v : t) v = std::pow(v, p);
    return t;
  }
  if (mode == "log") {
    const std::vector<double> e = linspace(-p, eps, nsteps + 1);
    std::vector<double> t(nsteps + 1);
    for (int i = 0; i <= nsteps; ++i) t[i] = 1.0 - std::pow(10.0, e[nsteps - i]);
    double mn = t[0], mx;
    for (double v : t) mn = std::min(mn, v);
    for (int i = 0; i <= nsteps; ++i) t[i] -= mn - eps;
    mx = t[0];
    for (double v : t) mx = std::max(mx, v);
    for (int i = 0; i <= nsteps; ++i) t[i] /= mx;
    return t;
  }
  fprintf(stderr, "schedule mode %s not supported\n", mode.c_str());
  exit(1);
}

struct StepCfg {
  double scScaleNoise = 1.0, tLimOde = 0.95, tLimOdeBelow = 0.02, tsrK = 16.0, tsrSigma = 3.0;
};

inline double tsrRatio(double t, double k, double sigma) {
  const double snr = (t * t) / ((1 - t) * (1 - t));
  return (snr * sigma * sigma + 1) / ((snr * sigma * sigma) / k + 1);
}

// `get_schedule_tsr_safe` with `_schedule_with_tail_multiplier`.
inline std::vector<double> scheduleTsrSafe(const std::string& mode, double p, int tailSteps,
                                           int tailMultiplier, int nsteps) {
  const int m = tailMultiplier;
  const int ts = nsteps >= tailSteps * m ? tailSteps : nsteps / (2 * std::max(1, m));
  if (!ts) return baseSchedule(mode, nsteps, p);
  const int base = nsteps - (m - 1) * ts;
  const int fine = base * m;
  const int tailStart = (base - ts) * m;
  const std::vector<double> full = baseSchedule(mode, fine, p);
  std::vector<double> out;
  for (int i = 0; i < tailStart; i += m) out.push_back(full[i]);
  for (int i = tailStart; i <= fine; ++i) out.push_back(full[i]);
  return out;
}

// `get_gt` (f_pow 1, eps 1e-2), divided by the tsr ratio squared for `vf_tsr`.
inline std::vector<double> noiseSchedule(const std::vector<double>& ts, const std::string& mode,
                                         const StepCfg& step) {
  std::vector<double> g(ts.size() - 1);
  for (size_t i = 0; i < g.size(); ++i) {
    const double t = std::min(std::max(ts[i], 0.0), 1.0 - 1e-5);
    double v;
    if (mode == "1/t") v = 1.0 / (t + 1e-2);
    else if (mode == "1-t/t") v = (1 - t) / (t + 1e-2);
    else if (mode == "tan")
      v = ((M_PI / 2) * lf::jsmath::sin(((1 - t) * M_PI) / 2)) /
          (lf::jsmath::cos(((1 - t) * M_PI) / 2) + 1e-2);
    else if (mode == "ode") v = 0.0;
    else { fprintf(stderr, "gt mode %s not implemented\n", mode.c_str()); exit(1); }
    g[i] = v / (tsrRatio(ts[i], step.tsrK, step.tsrSigma) * tsrRatio(ts[i], step.tsrK, step.tsrSigma));
  }
  return g;
}

// One modality's schedule. The two the model samples - the backbone CA and the local latents -
// differ in their schedule mode, their noise mode and their tsr constants.
struct Schedule {
  std::vector<double> ts, gt;
  StepCfg step;
};

inline Schedule bbCaSchedule(int nsteps) {
  Schedule s;
  s.step = {1.0, 0.95, 0.02, 16.0, 3.0};
  s.ts = scheduleTsrSafe("log", 2.0, 10, 2, nsteps);
  s.gt = noiseSchedule(s.ts, "1/t", s.step);
  return s;
}
inline Schedule latentSchedule(int nsteps) {
  Schedule s;
  s.step = {1.0, 0.95, 0.02, 4.0, 12.0};
  s.ts = scheduleTsrSafe("power", 2.0, 10, 2, nsteps);
  s.gt = noiseSchedule(s.ts, "tan", s.step);
  return s;
}

// `[width, depth]` bands over the steps, one depth a step (`Proteina.recycle_depths`).
//
// 🔴 A SHORTER RUN RESAMPLES THE BANDS, IT DOES NOT TAKE THEIR TAIL. The bands always expand to
// SIXTY entries whatever `nsteps` is, and `recycleDepths` then reads them at `i * 60 / nsteps` -
// so two steps of `low` are its FIRST and THIRTY-FIRST entries, [4, 1], where a tail would give
// [1, 1]. Taking the tail runs the whole short schedule at the cheapest depth the long one ever
// reaches, which is a different model, not a cheaper one.
inline std::vector<int> recycleDepths(const std::string& name, int nsteps) {
  std::vector<std::pair<int, int>> bands;
  if (name == "none") bands = {{60, 1}};
  else if (name == "low") bands = {{10, 4}, {10, 3}, {10, 2}, {30, 1}};
  else if (name == "medium") bands = {{5, 6}, {10, 5}, {5, 4}, {10, 3}, {20, 2}, {10, 1}};
  else if (name == "high") bands = {{15, 6}, {15, 5}, {15, 4}, {15, 3}};
  else { fprintf(stderr, "recycle schedule %s not known\n", name.c_str()); exit(1); }
  std::vector<int> full;
  for (const auto& [width, depth] : bands) {
    for (int i = 0; i < width; ++i) full.push_back(depth);
  }
  if ((int)full.size() == nsteps) return full;
  std::vector<int> depths(nsteps);
  for (int i = 0; i < nsteps; ++i) {
    depths[i] = full[std::min((int)full.size() - 1, (int)((long long)i * full.size() / nsteps))];
  }
  return depths;
}

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
