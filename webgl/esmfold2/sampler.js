/**
 * ESMFold2's EDM sampler in WebGL2: a device denoiser driven by the reference's
 * own schedule and update.
 *
 * 🔴 EVERY PIECE OF ARITHMETIC IS IMPORTED, NOT RESTATED. The Karras schedule
 * with its `max_inference_sigma` truncation, the churn factors, the noise
 * levels, Algorithm 19's random augmentation, the Kabsch alignment and the step
 * update all live in `cpu/esmfold2/sampler.js`, which is checked against the
 * model's own recorded draws. This file is the loop with the denoiser on the
 * device.
 *
 * 🔴 AND `max_inference_sigma` TRUNCATES THE SCHEDULE AND THEN PREPENDS ITSELF,
 * so asking for fifteen steps runs ELEVEN: "the number of steps is an OUTPUT of
 * this function, not an input". This loop runs what the schedule holds and
 * reports it.
 *
 * 🔴 AND STEP i TAKES `gammas[i + 1]`, NOT `gammas[i]` - the churn applied to
 * `sigma_tm` is decided by the NEXT noise level, and "the first step's t_hat is
 * the tell: 256 x 1.605 = 410.88 is what the model records, and 256 alone is
 * what the wrong indexing gives". `noiseLevels` is imported for that reason.
 *
 * 🔴 AND THE PAIR CONDITIONING IS COMPUTED ONCE FOR THE WHOLE WALK. Only the
 * single carries the noise level; "caching the pair and the single together
 * would freeze the noise level at step zero". The first step builds it and every
 * later step borrows it.
 *
 * 🔴 AND THE SAMPLER IS STOCHASTIC, SO A PORT CANNOT REPRODUCE THE MODEL'S
 * COORDINATES: "every step centres `x`, rotates it by a random rotation,
 * translates it by a random vector and adds Gaussian noise - four draws from
 * torch's global RNG per step". The draws here are this repository's seeded
 * ones, so a fold is reproducible for a reader even though it is not the
 * model's own sample.
 */
import {
  centreRandomAugmentation, churnFactors, gaussians, noiseLevels, noiseSchedule,
  samplerStep,
} from "../../cpu/esmfold2/sampler.js";
import { denoiseStep } from "./diffusion.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{shape: any, weights: any, sInputs: Tensor, zTrunk: Tensor,
 *   relPos: Tensor, hostFourier: {weights: Float32Array, offsets: Float32Array},
 *   embed: (scaled: Tensor) => any, decode: (act: Tensor, embedded: any) => Tensor,
 *   release: (embedded: any) => void,
 *   atomMask: Float32Array, atoms: number,
 *   steps?: number, seed?: number, stepScale?: number,
 *   update?: "euler" | "exponential", schedule?: "karras" | "geometric",
 *   sMax?: number, sMin?: number, p?: number, maxSigma?: number,
 *   gammaMin?: number, gamma0?: number, noiseScale?: number,
 *   onStep?: (frame: object) => void}} call
 * @returns {{coordinates: Float32Array, levels: Float32Array, steps: number}}
 */
export function sampleOnWebGl2(execution, call) {
  const {
    shape, weights, atomMask, atoms, hostFourier,
    stepScale = 1.5, update = "euler", schedule = "karras",
    sMax = 16, sMin = 4e-4, p = 7, maxSigma = 256,
    gammaMin = 1, gamma0 = 0.605, noiseScale = 1.003,
  } = call;
  const sigmaData = shape.sigmaData ?? 16;
  const draw = gaussians(call.seed ?? 42);
  const levels = noiseSchedule({
    steps: call.steps ?? 15, sMax, sMin, p, sigmaData, maxSigma, schedule,
  });
  const gammas = churnFactors(levels, gammaMin, gamma0);
  const hats = noiseLevels(levels, gammas);

  let coordinates = new Float32Array(atoms * 3);
  for (let index = 0; index < coordinates.length; index += 1) {
    coordinates[index] = draw() * levels[0];
  }
  /** The pair conditioning, built by the first step and borrowed by the rest. */
  let cachedPair;
  for (let step = 0; step + 1 < levels.length; step += 1) {
    const tHat = hats[step];
    const sigmaNext = levels[step + 1];
    // Algorithm 19 first, then the churn's extra noise.
    coordinates = centreRandomAugmentation(coordinates, atomMask, atoms, draw);
    const injected = noiseScale
      * Math.sqrt(Math.max(0, tHat * tHat - levels[step] * levels[step]));
    const noisy = new Float32Array(coordinates.length);
    for (let index = 0; index < noisy.length; index += 1) {
      noisy[index] = coordinates[index] + injected * draw();
    }

    const uploaded = execution.upload(noisy, "ef2.sampler.noisy");
    const denoised = denoiseStep(execution, {
      noisy: uploaded, tHat, shape, weights,
      sInputs: call.sInputs, zTrunk: call.zTrunk, relPos: call.relPos,
      hostFourier, embed: call.embed, decode: call.decode,
      cachedPair,
    });
    const values = Float32Array.from(execution.read(denoised.coordinates));
    execution.release(denoised.coordinates);
    execution.release(uploaded);
    cachedPair = denoised.pair;

    coordinates = samplerStep(noisy, values, atomMask, atoms, tHat, sigmaNext,
      stepScale, update);
    call.onStep?.({
      step, steps: levels.length - 1, noiseLevel: sigmaNext, tHat,
      coordinates: Float32Array.from(coordinates),
      denoised: values,
    });
  }
  if (cachedPair !== undefined) execution.release(cachedPair);
  return { coordinates, levels, steps: levels.length - 1 };
}
