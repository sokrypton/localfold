// Partial diffusion and inverse folding: the two runs that do NOT start from noise.
//
// Both read a real structure through the autoencoder encoder and start the flow part-way along it:
//
//   PARTIAL DIFFUSION takes a clean sample, corrupts it to the midpoint of the schedule
//     (`partial_trajectory_fraction_steps`) and denoises from there, which diversifies a scaffold
//     rather than inventing one.
//   INVERSE FOLDING holds the backbone and redesigns the sequence and rotamers: the target CA is
//     resampled from its forward marginal at every step (`resampled_forward_marginal`), so the
//     latents are denoised against a backbone that never drifts.
//
// Ported from `shared/rfd4p/partial.js`, which is upstream's `search/vanilla.py` and
// `flow_matching/product_space_flow_matcher.py`. All of it is host arithmetic over doubles - the
// schedules are tens of numbers and the corruption is one pass over n tokens.
//
// 🔴 THE SCHEDULE IS NOT THE ORDINARY ONE WITH ITS HEAD CUT OFF. `buildPartialSchedule` EXPANDS a
// schedule until its tail of `nsteps + 1` points starts at the corruption time, which is a
// different step count and therefore different points - at 0.5 of 60 steps the expanded schedule
// has 60 of its own and the tail of 31 is taken from that. Slicing the 60-step schedule would give
// the right endpoints and the wrong spacing everywhere between.
#pragma once
#include "sampler.cuh"

namespace rfd4p {

// `pyRound`, `partialStepCount`, `resolvePartialT` and `partialSchedule` are host arithmetic the
// native FEATURISER needs too (it writes a partial run's per-step conditioning), so they live in
// schedule.h, which both sides include.

// ---------------------------------------------------------------- the corruption
//
// These run on the host over the replayed noise, as `sample_noise` does in generate.cuh, and for
// the same reason: the draw order is a function of the schedule alone, so it is reproducible
// without the model and the port replays it rather than running Box-Muller through a `log` it has
// no bit-exact copy of.

/** `zero_com_bb_noise`: n * 3 normals, zero-centred over the emitting tokens. */
inline std::vector<float> zeroComBbNoise(const float* noise, int n, const float* emitBb) {
  std::vector<float> bb(noise, noise + (size_t)n * 3);
  double mean[3] = {0, 0, 0};
  int live = 0;
  for (int i = 0; i < n; ++i) {
    if (!emitBb[i]) { bb[i * 3] = bb[i * 3 + 1] = bb[i * 3 + 2] = 0.f; continue; }
    ++live;
    for (int k = 0; k < 3; ++k) mean[k] += bb[i * 3 + k];
  }
  for (int k = 0; k < 3; ++k) mean[k] = (float)(mean[k] / std::max(live, 1));
  for (int i = 0; i < n; ++i) {
    if (!emitBb[i]) continue;
    for (int k = 0; k < 3; ++k) bb[i * 3 + k] = (float)(bb[i * 3 + k] - (float)mean[k]);
  }
  return bb;
}

/** The molecules that take a stochastic offset, in the order their draws are consumed. */
inline std::vector<int> offsetGroups(int n, const float* emitBb, const int* coordCond,
                                     const int* moleculeIid) {
  std::vector<int> groups;
  for (int i = 0; i < n; ++i) {
    if (emitBb[i] && !coordCond[i] &&
        std::find(groups.begin(), groups.end(), moleculeIid[i]) == groups.end()) {
      groups.push_back(moleculeIid[i]);
    }
  }
  std::sort(groups.begin(), groups.end());
  return groups;
}

/** `interpolate`'s per-molecule offset: three normals a group, scaled, on the offsettable tokens. */
inline void applyPerGroupOffset(std::vector<float>& x, int n, const float* emitBb,
                                const int* coordCond, const int* moleculeIid,
                                const std::vector<int>& groups, const float* noise, size_t& at,
                                float scale) {
  for (int g : groups) {
    const float off[3] = {noise[at], noise[at + 1], noise[at + 2]};
    at += 3;
    for (int i = 0; i < n; ++i) {
      if (!emitBb[i] || coordCond[i] || moleculeIid[i] != g) continue;
      for (int k = 0; k < 3; ++k) x[i * 3 + k] = (float)(x[i * 3 + k] + (float)(scale * off[k]));
    }
  }
  for (int i = 0; i < n; ++i) {
    if (!emitBb[i]) x[i * 3] = x[i * 3 + 1] = x[i * 3 + 2] = 0.f;
  }
}

/**
 * `sample_noised_target_backbone`: the clean target drawn from its forward marginal at `t`. Called
 * once before the loop and again at every step but the last of an inverse-folding run, which is
 * what keeps the backbone fixed while the sequence is redesigned.
 */
inline std::vector<float> noisedTargetBackbone(const std::vector<float>& target, double t, int n,
                                               const float* emitBb, const int* coordCond,
                                               const int* moleculeIid,
                                               const std::vector<int>& groups, const float* noise,
                                               size_t& at, float scale) {
  std::vector<float> x0 = zeroComBbNoise(noise + at, n, emitBb);
  at += (size_t)n * 3;
  applyPerGroupOffset(x0, n, emitBb, coordCond, moleculeIid, groups, noise, at, scale);
  std::vector<float> xt((size_t)n * 3, 0.f);
  const float tf = (float)t, oneMinus = (float)(1.0 - tf);
  for (int i = 0; i < n; ++i) {
    if (!emitBb[i]) continue;
    for (int k = 0; k < 3; ++k) {
      xt[i * 3 + k] = (float)((float)(oneMinus * x0[i * 3 + k]) + (float)(tf * target[i * 3 + k]));
    }
  }
  return xt;
}

/**
 * `corrupt_batch`: a clean sample to the starting times. The draw order is upstream's and the
 * comment there is the specification - bb_ca's noise, then the latents', then bb_ca's per-molecule
 * offset. Getting it wrong shifts every subsequent draw of the run.
 */
struct Corrupted {
  std::vector<float> bb, ll;
};
inline Corrupted corruptBatch(const std::vector<float>& x1Bb, const std::vector<float>& x1Ll,
                              const PartialT& t, int n, int latent, const float* emitBb,
                              const float* emitLl, const int* coordCond, const int* moleculeIid,
                              const std::vector<int>& groups, const float* noise, size_t& at,
                              float scale) {
  std::vector<float> cleanBb = x1Bb, cleanLl = x1Ll;
  for (int i = 0; i < n; ++i) {
    if (!emitBb[i]) cleanBb[i * 3] = cleanBb[i * 3 + 1] = cleanBb[i * 3 + 2] = 0.f;
    if (!emitLl[i]) for (int k = 0; k < latent; ++k) cleanLl[(size_t)i * latent + k] = 0.f;
  }
  std::vector<float> x0Bb = zeroComBbNoise(noise + at, n, emitBb);
  at += (size_t)n * 3;
  std::vector<float> x0Ll(noise + at, noise + at + (size_t)n * latent);
  at += (size_t)n * latent;
  for (int i = 0; i < n; ++i) {
    if (!emitLl[i]) for (int k = 0; k < latent; ++k) x0Ll[(size_t)i * latent + k] = 0.f;
  }
  applyPerGroupOffset(x0Bb, n, emitBb, coordCond, moleculeIid, groups, noise, at, scale);

  Corrupted out;
  out.bb.assign((size_t)n * 3, 0.f);
  out.ll.assign((size_t)n * latent, 0.f);
  const float tb = (float)t.bbCa, tl = (float)t.localLatents;
  for (int i = 0; i < n; ++i) {
    if (emitBb[i]) {
      for (int k = 0; k < 3; ++k) {
        out.bb[i * 3 + k] =
            (float)((float)((1.f - tb) * x0Bb[i * 3 + k]) + (float)(tb * cleanBb[i * 3 + k]));
      }
    }
    if (emitLl[i]) {
      for (int k = 0; k < latent; ++k) {
        const size_t o = (size_t)i * latent + k;
        out.ll[o] = (float)((float)((1.f - tl) * x0Ll[o]) + (float)(tl * cleanLl[o]));
      }
    }
  }
  return out;
}

}  // namespace rfd4p
