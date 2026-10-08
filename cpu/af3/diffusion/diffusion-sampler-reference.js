/**
 * AF3's diffusion sampler: the loop that turns a denoiser into a structure.
 *
 * The denoiser answers "given atoms this noisy, what were they?". This walks
 * that answer down a noise schedule two hundred times, and it is arithmetic
 * rather than network - about eighty lines against the head's two hundred
 * million parameters.
 *
 *     positions  = gaussian noise, scaled to the first noise level
 *     for each level:
 *         positions       = randomAugmentation(positions)   <- rigid, random
 *         t_hat           = previous * (1 + gamma)           <- the churn
 *         positions_noisy = positions + noise(t_hat, previous)
 *         denoised        = denoiser(positions_noisy, t_hat)
 *         positions       = positions_noisy
 *                         + stepScale * (level - t_hat) * (positions_noisy - denoised) / t_hat
 *
 * 🔴 THE STEP IS SCALED BY 1.5, WHICH IS NOT AN EULER STEP. `step_scale` is 1.5
 * in AF3's config, so the sampler deliberately overshoots the gradient it just
 * computed. Setting it to 1 gives a textbook integrator and a worse structure;
 * the weights were sampled with 1.5, so the overshoot is the specification.
 *
 * 🔴 THE CHURN ADDS NOISE BACK, AND ONLY ABOVE sigma = 1. `gamma_0` is 0.8, so
 * wherever the schedule is still above `gamma_min` the sampler pushes the noise
 * level UP by 80% before denoising, then steps down past where it was. Below
 * that threshold gamma is zero, t_hat equals the previous level, and the
 * injected noise is exactly zero - so a run that churns everywhere, or nowhere,
 * is wrong in a way that still produces a protein-shaped answer.
 *
 * 🔴 THE TRAJECTORY IS NOT REPRODUCIBLE AGAINST AF3, AND DOES NOT NEED TO BE.
 * Every step draws from JAX's threefry PRNG, which this does not implement, so
 * our sample is a different draw from the same distribution. What IS checkable
 * exactly is the arithmetic: given AF3's own noisy positions and its own
 * denoised answer, the step update must reproduce AF3's next positions. See
 * tools/oracle/check_af3_sampler.js.
 */

/** AF3's EDM noise schedule, in angstroms. */
export function noiseSchedule(t, { sigmaData = 16, sigmaMin = 0.0004,
                                   sigmaMax = 160, rho = 7 } = {}) {
  const low = Math.pow(sigmaMin, 1 / rho);
  const high = Math.pow(sigmaMax, 1 / rho);
  return sigmaData * Math.pow(high + t * (low - high), rho);
}

/** The `steps + 1` levels a run walks down, from sigmaData*sigmaMax to ~0. */
export function noiseLevels(steps, options) {
  const levels = new Float64Array(steps + 1);
  for (let index = 0; index <= steps; index += 1) {
    levels[index] = noiseSchedule(index / steps, options);
  }
  return levels;
}

/**
 * A random rotation, by Gram-Schmidt on two normal vectors.
 *
 * Returned row-major as [e0; e1; e2], which is the orientation AF3 contracts
 * with as `positions @ rot` - so the rows are the new basis, not the columns.
 */
export function randomRotation(normal) {
  const v0 = [normal(), normal(), normal()];
  const v1 = [normal(), normal(), normal()];
  const norm = (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const scale0 = 1 / Math.max(1e-10, norm(v0));
  const e0 = v0.map((value) => value * scale0);
  const dot = v1[0] * e0[0] + v1[1] * e0[1] + v1[2] * e0[2];
  const w = [v1[0] - e0[0] * dot, v1[1] - e0[1] * dot, v1[2] - e0[2] * dot];
  const scale1 = 1 / Math.max(1e-10, norm(w));
  const e1 = w.map((value) => value * scale1);
  const e2 = [e0[1] * e1[2] - e0[2] * e1[1],
              e0[2] * e1[0] - e0[0] * e1[2],
              e0[0] * e1[1] - e0[1] * e1[0]];
  return [e0, e1, e2];
}

/**
 * Centre on the real atoms, rotate, translate, and re-mask.
 *
 * 🔴 THE CENTRE IS THE MASKED MEAN, NOT THE MEAN. Padded atom slots hold zeros,
 * and averaging them in drags the centre toward the origin by whatever fraction
 * of the slots are padding - two thirds of them on a small protein.
 */
export function randomAugmentation(positions, mask, atoms, normal) {
  const centre = [0, 0, 0];
  let count = 0;
  for (let atom = 0; atom < atoms; atom += 1) {
    if (!mask[atom]) continue;
    count += 1;
    for (let axis = 0; axis < 3; axis += 1) centre[axis] += positions[atom * 3 + axis];
  }
  for (let axis = 0; axis < 3; axis += 1) centre[axis] /= (count + 1e-6);

  const rotation = randomRotation(normal);
  const translation = [normal(), normal(), normal()];
  const output = new Float32Array(positions.length);
  for (let atom = 0; atom < atoms; atom += 1) {
    if (!mask[atom]) continue;
    const x = positions[atom * 3] - centre[0];
    const y = positions[atom * 3 + 1] - centre[1];
    const z = positions[atom * 3 + 2] - centre[2];
    for (let axis = 0; axis < 3; axis += 1) {
      // ...contracted as `positions @ rot`, so axis `j` reads column j.
      output[atom * 3 + axis] = x * rotation[0][axis] + y * rotation[1][axis]
        + z * rotation[2][axis] + translation[axis];
    }
  }
  return output;
}

/**
 * One step of the sampler, given a denoised estimate.
 *
 * Split out from the loop because this is the part that can be checked exactly
 * against AF3: feed it AF3's own noisy positions and denoised answer and it
 * must reproduce AF3's next positions.
 */
export function samplerStep(positionsNoisy, denoised, tHat, noiseLevel,
                            { stepScale = 1.5 } = {}) {
  const output = new Float32Array(positionsNoisy.length);
  const delta = noiseLevel - tHat;
  for (let index = 0; index < output.length; index += 1) {
    const gradient = (positionsNoisy[index] - denoised[index]) / tHat;
    output[index] = positionsNoisy[index] + stepScale * delta * gradient;
  }
  return output;
}

/**
 * The whole sampler.
 *
 * @param {(positions: Float32Array, noiseLevel: number) => Float32Array} denoise
 * @param {{atoms: number, mask: Float32Array, steps: number,
 *          normal: () => number}} options
 *   `normal` draws one standard gaussian; supplying it keeps this deterministic
 *   under test and leaves the choice of generator to the caller.
 */
export function sample(denoise, options) {
  const { atoms, mask, steps, normal } = options;
  const gamma0 = options.gamma0 ?? 0.8;
  const gammaMin = options.gammaMin ?? 1.0;
  const noiseScale = options.noiseScale ?? 1.003;
  const stepScale = options.stepScale ?? 1.5;
  const levels = noiseLevels(steps, options);

  let positions = new Float32Array(atoms * 3);
  for (let index = 0; index < positions.length; index += 1) {
    positions[index] = normal() * levels[0];
  }
  let previous = levels[0];

  for (let step = 1; step <= steps; step += 1) {
    const level = levels[step];
    positions = randomAugmentation(positions, mask, atoms, normal);

    const gamma = level > gammaMin ? gamma0 : 0;
    const tHat = previous * (1 + gamma);
    // ...zero whenever gamma is, because t_hat then equals `previous` exactly.
    const injected = noiseScale * Math.sqrt(Math.max(0, tHat * tHat - previous * previous));
    const noisy = new Float32Array(positions.length);
    for (let index = 0; index < noisy.length; index += 1) {
      noisy[index] = positions[index] + injected * normal();
    }

    const denoised = denoise(noisy, tHat);
    positions = samplerStep(noisy, denoised, tHat, level, { stepScale });
    previous = level;

    // 🔴 `denoised` IS THE ONE WORTH WATCHING, NOT `positions`. It is the
    // model's current guess at the finished structure, so it goes from blob to
    // fold; `positions` is that guess plus the noise still left at this level,
    // which at step 1 of 200 is a cloud 2560 A across. Both are passed because
    // only one of them is the honest trajectory.
    //
    // 🔴 AND NEITHER IS IN A STABLE FRAME. randomAugmentation applies a fresh
    // random rotation and translation at the top of EVERY step, so consecutive
    // frames are in unrelated coordinate systems - an animation built from them
    // tumbles wildly whatever the structure is doing. Superimpose each frame on
    // the one before it before showing them.
    options.onStep?.({
      step, steps, noiseLevel: level, tHat,
      positions: Float32Array.from(positions),
      denoised: Float32Array.from(denoised),
    });
  }
  return positions;
}
