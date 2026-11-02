/**
 * ESMFold2's EDM sampler: a denoiser, a noise schedule, and a structure.
 *
 * 🔴 IT IS STOCHASTIC, SO A PORT CANNOT REPRODUCE ITS COORDINATES. Every step
 * centres `x`, rotates it by a random rotation, translates it by a random
 * vector and adds Gaussian noise - four draws from torch's global RNG per step,
 * eleven steps. Nothing in JavaScript reproduces that stream, so the honest
 * gate is not "the same structure" but the DETERMINISTIC parts, each exactly:
 * the schedule, the gammas, the Kabsch alignment and the update algebra, with
 * the draws taken from the model rather than made again.
 *
 * 🔴 AND THE RANDOM AUGMENTATION IS NOT COSMETIC. It is AF3's Algorithm 19 and
 * the model was trained with it: the denoiser sees a differently oriented copy
 * of the structure at every step, which is what makes its answer rotation
 * equivariant in practice rather than in principle. Dropping it makes a sampler
 * that runs and folds worse.
 */

/**
 * The Karras power-law schedule, with a zero appended.
 *
 * 🔴 AND `max_inference_sigma` TRUNCATES IT AND THEN PREPENDS ITSELF, so asking
 * for fifteen steps runs ELEVEN. Entries above the cap are dropped and the cap
 * becomes the new first entry - at the shipped 256 that removes four of the
 * sixteen. A port that runs the requested count is running a different
 * schedule; the number of steps is an OUTPUT of this function, not an input.
 */
export function noiseSchedule({ steps, sMax, sMin, p, sigmaData, maxSigma, schedule = "karras" }) {
  // `schedule: "geometric"` spaces the levels evenly in log sigma (Blitz's; cuda/ef2/src/sampler.cuh has the same),
  // the zero and the max_sigma cap kept - which drops a different number of levels, so 15 scheduled run 13, not 11
  if (schedule !== "karras" && schedule !== "geometric") throw new Error(`schedule ${schedule}: karras or geometric`);
  const values = [];
  if (steps === 1) {
    values.push(sMax * sigmaData, 0);
  } else {
    const inverse = 1 / p;
    const high = Math.pow(sMax, inverse);
    const low = Math.pow(sMin, inverse);
    for (let k = 0; k < steps; k += 1) {
      values.push(schedule === "geometric"
        ? sigmaData * Math.exp(Math.log(sMax) + (k / (steps - 1)) * (Math.log(sMin) - Math.log(sMax)))
        : sigmaData * Math.pow(high + (k / (steps - 1)) * (low - high), p));
    }
    values.push(0);
  }
  if (maxSigma === undefined || maxSigma === null) return Float32Array.from(values);
  const kept = values.filter((value) => value <= maxSigma);
  return Float32Array.from([maxSigma, ...kept]);
}

/**
 * The churn per entry.
 *
 * 🔴 STEP i TAKES `gammas[i + 1]`, NOT `gammas[i]`. Upstream zips
 * `schedule[:-1]` with `schedule[1:]` and `gammas[1:]`, so the churn applied to
 * `sigma_tm` is decided by the NEXT noise level. Off by one it still runs, and
 * the first step's t_hat is the tell: 256 x 1.605 = 410.88 is what the model
 * records, and 256 alone is what the wrong indexing gives.
 */
export function churnFactors(schedule, gammaMin, gamma0) {
  const out = new Float32Array(schedule.length);
  for (let i = 0; i < schedule.length; i += 1) {
    out[i] = schedule[i] > gammaMin ? gamma0 : 0;
  }
  return out;
}

/** The noise level the denoiser is actually asked about, per step. */
export function noiseLevels(schedule, gammas) {
  const out = [];
  for (let i = 0; i + 1 < schedule.length; i += 1) {
    out.push(schedule[i] * (1 + gammas[i + 1]));
  }
  return out;
}


import { weightedRigidAlign } from "../af3/diffusion/rigid-align.js";

export { weightedRigidAlign };

/**
 * One sampler step, after the denoiser has answered.
 *
 *     x_noisy = align(x_noisy, x_denoised)
 *     x       = x_noisy + eta * (sigma_t - t_hat) * (x_noisy - x_denoised) / t_hat
 *
 * 🔴 THE ALIGNMENT COMES FIRST AND IT MOVES THE NOISY COPY, NOT THE ANSWER. The
 * denoiser's output is the reference; the noisy input is rotated onto it, and
 * the step is then taken between two structures in one frame. Aligning the
 * other way round is the same Kabsch call with its arguments swapped and it
 * walks the structure away from the answer.
 */
export function samplerStep(noisy, denoised, weights, atoms, tHat, sigmaT, stepScale, update = "euler") {
  if (update !== "euler" && update !== "exponential") throw new Error(`update ${update}: euler or exponential`);
  const aligned = weightedRigidAlign(noisy, denoised, weights, atoms);
  const out = new Float32Array(aligned.length);
  // `update: "exponential"`: x = r^S x_noisy + (1 - r^S) d with r = sigma_t / t_hat, which is this same line at
  // factor r^S - 1 (cpu/af3/diffusion/diffusion-sampler.js says why Euler's overshoots at a large step)
  const factor = update === "exponential" ? Math.pow(sigmaT / tHat, stepScale) - 1 : stepScale * (sigmaT - tHat) / tHat;
  for (let i = 0; i < out.length; i += 1) {
    out[i] = aligned[i] + factor * (aligned[i] - denoised[i]);
  }
  return out;
}

/**
 * Algorithm 19: centre on the live atoms, rotate randomly, translate randomly.
 *
 * 🔴 IT RUNS BEFORE EVERY STEP, NOT ONCE. The sampler is equivariant only
 * because the structure is re-posed each time, and the model was trained with
 * this in the loop - so a port that centres once at the start converges to
 * something and is a different sampler.
 *
 * @param draw a standard normal generator, so a caller owns the randomness
 */
export function centreRandomAugmentation(x, mask, atoms, draw) {
  let total = 0;
  const centre = [0, 0, 0];
  for (let atom = 0; atom < atoms; atom += 1) {
    const w = mask[atom];
    total += w;
    for (let axis = 0; axis < 3; axis += 1) centre[axis] += w * x[atom * 3 + axis];
  }
  for (let axis = 0; axis < 3; axis += 1) centre[axis] /= Math.max(total, 1);
  // A uniform random rotation from a normalised quaternion, as upstream.
  const q = [draw(), draw(), draw(), draw()];
  const scale = Math.hypot(q[0], q[1], q[2], q[3]) * (q[0] < 0 ? -1 : 1);
  const [r, i, j, k] = q.map((value) => value / scale);
  const twoS = 2 / (r * r + i * i + j * j + k * k);
  const R = [
    1 - twoS * (j * j + k * k), twoS * (i * j - k * r), twoS * (i * k + j * r),
    twoS * (i * j + k * r), 1 - twoS * (i * i + k * k), twoS * (j * k - i * r),
    twoS * (i * k - j * r), twoS * (j * k + i * r), 1 - twoS * (i * i + j * j),
  ];
  const shift = [draw(), draw(), draw()];
  const out = new Float32Array(x.length);
  for (let atom = 0; atom < atoms; atom += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      let sum = 0;
      for (let d = 0; d < 3; d += 1) sum += (x[atom * 3 + d] - centre[d]) * R[d * 3 + axis];
      out[atom * 3 + axis] = sum + shift[axis];
    }
  }
  return out;
}

/**
 * A seeded standard normal, so a fold is reproducible even though it is not the
 * model's own draw. xorshift32 plus Box-Muller.
 */
export function uniforms(seed) {
  // 🔴 SEED 0 AND SEED 1 GAVE THE SAME STREAM. `(seed >>> 0) || 1` sends zero
  // to one, so the two commonest seeds a caller reaches for were the same fold
  // - and it looked like a working seed axis, because seeds 2 and 3 differed.
  // Mixing with the golden ratio's fractional part keeps every seed distinct
  // and still cannot produce xorshift's fixed point.
  let state = ((seed >>> 0) ^ 0x9e3779b9) >>> 0 || 1;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >> 17;
    state ^= state << 5; state >>>= 0;
    return (state + 1) / 4294967297;
  };
}

export function gaussians(seed) {
  // 🔴 SEED 0 AND SEED 1 GAVE THE SAME STREAM. `(seed >>> 0) || 1` sends zero
  // to one, so the two commonest seeds a caller reaches for were the same fold
  // - and it looked like a working seed axis, because seeds 2 and 3 differed.
  // Mixing with the golden ratio's fractional part keeps every seed distinct
  // and still cannot produce xorshift's fixed point.
  const uniform = uniforms(seed);
  let spare = null;
  return () => {
    if (spare !== null) { const value = spare; spare = null; return value; }
    const u = uniform(), v = uniform();
    const radius = Math.sqrt(-2 * Math.log(u));
    spare = radius * Math.sin(2 * Math.PI * v);
    return radius * Math.cos(2 * Math.PI * v);
  };
}
