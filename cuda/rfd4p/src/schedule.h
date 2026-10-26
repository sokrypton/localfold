// The sampler's HOST half: the random source, the time and noise schedules, and a partial run's
// start times and schedule. (The recycle depths are the featuriser's - featurise.cpp.) Pure C++ over <cmath> and jsmath, with no CUDA in it, because two things need it - the
// port's own sampler (src/sampler.cuh) and the native featuriser (featurise/featurise.cpp), which
// is compiled by g++ with the featuriser's own flags and never sees nvcc.
//
// 🔴 ONE COPY, BECAUSE A SCHEDULE IS A CONVENTION. Two implementations of `get_schedule_tsr_safe`
// is two places for a tail multiplier to drift, and this port has already paid once for a recycle
// schedule that two sides derived differently.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../../featurise/jsmath.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
  /**
   * `randperm(k)`: Fisher-Yates over the SAMPLER's stream, as `flow.js` has it.
   *
   * 🔴 IT CONSUMES k-1 RAW UNIFORMS, NOT NORMALS. Every other draw here goes through
   * Box-Muller, which takes two uniforms per PAIR of outputs; this one takes `next()` directly,
   * once per swap. Modelling it as `randn` advances the stream by a different amount and moves
   * every later draw of the run - measured, 49 of 75 permutation entries wrong.
   *
   * 🔴 AND AT k <= 1 IT DRAWS NOTHING, which is why a single coordinate group is the identity
   * and costs no stream - the reason every shipped example works without this existing at all.
   */
  std::vector<int> randperm(int k) {
    std::vector<int> p(k > 0 ? k : 0);
    for (int i = 0; i < k; ++i) p[i] = i;
    for (int i = k - 1; i > 0; --i) {
      const int j = (int)std::floor(next() * (i + 1));
      std::swap(p[i], p[j]);
    }
    return p;
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

// `DEFAULT_SAMPLING.centering.scale`, and it is a DOUBLE: the per-molecule offset is this times a
// float draw, rounded once at the store. Written `0.2f` it rounds twice and lands one ulp out.
inline constexpr double CENTERING_SCALE = 0.2;

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

// ---------------------------------------------------------------- partial runs
//
// Partial diffusion and inverse folding start the flow part-way along it (partial.cuh says what
// each is). Ported from `shared/rfd4p/partial.js`, upstream's `search/vanilla.py`.

// Python's round-half-to-even, which decides the step count on an exact half (0.5 of 5 steps).
inline int pyRound(double x) {
  const double fl = std::floor(x);
  const double diff = x - fl;
  if (std::fabs(diff - 0.5) < 1e-12) return (int)(std::fmod(fl, 2.0) == 0 ? fl : fl + 1);
  return (int)std::floor(x + 0.5);
}

inline int partialStepCount(double frac, int nsteps) { return pyRound(frac * nsteps); }

struct Modality {
  Schedule (*build)(int);
  const char* name;
};
inline Modality modalities(int i) {
  return i == 0 ? Modality{bbCaSchedule, "bb_ca"} : Modality{latentSchedule, "local_latents"};
}

struct PartialT {
  bool set = false;
  double bbCa = 0.0, localLatents = 0.0;
};

// What a job asks for, unresolved: the five fields `resolve_partial_t` reads. A field the job
// does not name stays NaN (or false), which is "absent" - never zero, since 0 is a real start time.
struct PartialRequest {
  bool inverseFolding = false;
  std::string inverseFoldingMethod = "resampled_forward_marginal";
  double fraction = NAN, tBackbone = NAN, tLatent = NAN;
  bool couple = false;
};

/**
 * `resolve_partial_t`, every form: inverse folding (both times 0), a fraction of the trajectory,
 * both start times given, or one given and the other COUPLED through the default schedules - and
 * the same mutual exclusions upstream raises on, refused with its words.
 *
 * 🔴 THIS TOOK TWO OF THE FIVE UNTIL NOW, and the native featuriser took none: a job naming a
 * partial run was featurised as an ordinary one and generated from noise with exit status 0.
 */
inline PartialT resolvePartialT(const PartialRequest& r, int nsteps) {
  PartialT out;
  const bool hasFrac = !std::isnan(r.fraction), hasBb = !std::isnan(r.tBackbone),
             hasLl = !std::isnan(r.tLatent);
  auto refuse = [](const char* msg) { fprintf(stderr, "%s\n", msg); exit(1); };
  auto validate = [&](PartialT p) {
    if (p.bbCa >= 1.0 || p.localLatents >= 1.0) {
      refuse("partial-diffusion start time must be < 1 (t = 1 is the clean state, and the score "
             "conversion divides by 1 - t)");
    }
    return p;
  };
  if (r.inverseFolding) {
    if (r.inverseFoldingMethod != "resampled_forward_marginal") {
      fprintf(stderr, "inverse_folding_method must be one of resampled_forward_marginal, got %s\n",
              r.inverseFoldingMethod.c_str());
      exit(1);
    }
    if (hasBb || hasLl || r.couple || hasFrac) {
      refuse("inverse_folding is mutually exclusive with partial_t_backbone, partial_t_latent, "
             "couple_partial_t, and partial_trajectory_fraction_steps.");
    }
    out.set = true;
    return out;                      // bb_ca = local_latents = 0
  }
  if (hasFrac) {
    if (hasBb || hasLl || r.couple) {
      refuse("partial_trajectory_fraction_steps is mutually exclusive with partial_t_backbone, "
             "partial_t_latent, and couple_partial_t.");
    }
    if (!(r.fraction > 0.0 && r.fraction <= 1.0)) {
      fprintf(stderr, "partial_trajectory_fraction_steps must be in (0, 1], got %g\n", r.fraction);
      exit(1);
    }
    const int remaining = partialStepCount(r.fraction, nsteps);
    if (remaining < 1) {
      fprintf(stderr, "partial_trajectory_fraction_steps=%g with nsteps=%d rounds to zero "
                      "denoising steps.\n", r.fraction, nsteps);
      exit(1);
    }
    const int k = nsteps - remaining;
    out.set = true;
    out.bbCa = (float)bbCaSchedule(nsteps).ts[k];
    out.localLatents = (float)latentSchedule(nsteps).ts[k];
    return validate(out);
  }
  const int nSet = (hasBb ? 1 : 0) + (hasLl ? 1 : 0);
  if (nSet == 0) return out;
  if (nSet == 2) {
    if (r.couple) {
      refuse("couple_partial_t=true requires exactly one of partial_t_backbone or "
             "partial_t_latent; got both.");
    }
    out.set = true;
    out.bbCa = r.tBackbone;
    out.localLatents = r.tLatent;
    return validate(out);
  }
  if (!r.couple) {
    fprintf(stderr, "Only %s is set. Either provide both partial_t_backbone and partial_t_latent, "
                    "or set couple_partial_t=true.\n",
            hasBb ? "partial_t_backbone" : "partial_t_latent");
    exit(1);
  }
  // Coupled: the other modality's time is the one at the same STEP of a dense reference schedule.
  const int refN = 10000;
  const Schedule bb = bbCaSchedule(refN), ll = latentSchedule(refN);
  const std::vector<double>& refTs = hasBb ? bb.ts : ll.ts;
  const double refVal = hasBb ? r.tBackbone : r.tLatent;
  int k = refN;
  for (int i = 0; i <= refN; ++i) {
    if ((float)refTs[i] >= refVal - 1e-6) { k = i; break; }
  }
  out.set = true;
  out.bbCa = (float)bb.ts[k];
  out.localLatents = (float)ll.ts[k];
  return validate(out);
}

/**
 * `build_partial_schedule`. With `fullNsteps` the tail of the FULL schedule is taken and checked to
 * start at the corruption time; without it the schedule is expanded until its tail does.
 *
 * 🔴 THE SCHEDULE IS NOT THE ORDINARY ONE WITH ITS HEAD CUT OFF. Expanding until a tail of
 * `nsteps + 1` points starts at the corruption time is a different step count and therefore
 * different points; slicing the ordinary schedule gives the right endpoints and the wrong spacing
 * everywhere between.
 */
inline Schedule partialSchedule(int which, double partialT, int nsteps, int fullNsteps) {
  const Modality m = modalities(which);
  if (fullNsteps > 0) {
    if (nsteps < 1 || nsteps > fullNsteps) {
      fprintf(stderr, "partial step count must be in [1, %d], got %d\n", fullNsteps, nsteps);
      exit(1);
    }
    const Schedule full = m.build(fullNsteps);
    Schedule out;
    out.step = full.step;
    out.ts.assign(full.ts.end() - (nsteps + 1), full.ts.end());
    out.gt.assign(full.gt.end() - nsteps, full.gt.end());
    if (std::fabs((float)out.ts[0] - (float)partialT) > 1e-6) {
      fprintf(stderr, "%s: partial corruption time must match the full schedule's tail\n", m.name);
      exit(1);
    }
    return out;
  }
  // 🔴 THE DENSE SEARCH IS WHAT SETS THE EXPANDED STEP COUNT, and it is not `nsteps / fraction`:
  // the schedules are non-linear, so how many of a dense schedule's points lie above the
  // corruption time is the only thing that says how long a schedule has `nsteps` of them there.
  const int dense = std::max(nsteps * 100, 1000);
  const Schedule d = m.build(dense);
  int iStart = dense;
  for (int i = 0; i <= dense; ++i) {
    if ((float)d.ts[i] >= partialT - 1e-6) { iStart = i; break; }
  }
  const int above = std::max(dense - iStart, 1);
  const int expanded = pyRound((double)nsteps * dense / above);
  const Schedule full = m.build(expanded);
  Schedule out;
  out.step = full.step;
  out.ts.assign(full.ts.end() - (nsteps + 1), full.ts.end());
  out.gt.assign(full.gt.end() - nsteps, full.gt.end());
  return out;
}

}  // namespace rfd4p
