/**
 * AF3's diffusion sampler, driving the GPU denoiser.
 *
 * The schedule, the churn and the step update are pure arithmetic and already
 * live in src/af3/diffusion-sampler-reference.js, checked against AF3. This is
 * the same loop with an AWAITED denoiser, because the reference's is
 * synchronous and a GPU one cannot be.
 *
 * 🔴 THE ONLY REASON THIS IS NOT `sample()` WITH A DIFFERENT CALLBACK. Keeping
 * the arithmetic in one place matters more than avoiding the duplicated loop:
 * every constant here - step scale 1.5, gamma 0.8 above gamma_min 1.0, noise
 * scale 1.003 - is imported rather than restated.
 */
import { noiseLevels, randomAugmentation, samplerStep } from "./diffusion-sampler-reference.js";
import { Af3DiffusionHeadGpu } from "./diffusion-head-webgpu.js";

/**
 * @param {object} input everything the diffusion head takes except noiseLevel
 *   and positionsNoisy
 * @param {object} weights the diffusion head's
 * @param {{steps: number, normal: () => number, onStep?: Function,
 *          stopAfter?: number, gamma0?: number, gammaMin?: number,
 *          noiseScale?: number, stepScale?: number}} options
 */
export async function sampleOnGpu(device, input, weights, options) {
  const { steps, normal } = options;
  // 🔴 THE SCHEDULE'S LENGTH AND THE NUMBER OF DENOISER CALLS ARE NOT THE SAME
  // THING. `steps` sets the discretisation - level i is noiseSchedule(i/steps),
  // so a coarse schedule is not a subset of a fine one but a different set of
  // sigmas. `stopAfter` runs a PREFIX of the schedule and leaves the walk at
  // high noise, which is only useful if what you read out is `denoised` rather
  // than the returned sample.
  const stopAfter = Math.min(options.stopAfter ?? steps, steps);
  const gamma0 = options.gamma0 ?? 0.8;
  const gammaMin = options.gammaMin ?? 1.0;
  const noiseScale = options.noiseScale ?? 1.003;
  const stepScale = options.stepScale ?? 1.5;
  const atoms = input.shape.tokens * input.shape.dense;
  const levels = noiseLevels(steps, options);
  const head = new Af3DiffusionHeadGpu(device);

  let positions = new Float32Array(atoms * 3);
  for (let index = 0; index < positions.length; index += 1) {
    positions[index] = normal() * levels[0];
  }
  let previous = levels[0];

  for (let step = 1; step <= stopAfter; step += 1) {
    const level = levels[step];
    positions = randomAugmentation(positions, input.atomMask, atoms, normal);

    const gamma = level > gammaMin ? gamma0 : 0;
    const tHat = previous * (1 + gamma);
    const injected = noiseScale * Math.sqrt(Math.max(0, tHat * tHat - previous * previous));
    const noisy = new Float32Array(positions.length);
    for (let index = 0; index < noisy.length; index += 1) {
      noisy[index] = positions[index] + injected * normal();
    }

    const denoised = await head.run({ ...input, noiseLevel: tHat, positionsNoisy: noisy },
                                    weights);
    positions = samplerStep(noisy, denoised.positions, tHat, level, { stepScale });
    previous = level;

    // 🔴 `denoised` IS THE FRAME WORTH WATCHING and neither is in a stable
    // frame - see the note in diffusion-sampler-reference.js. Copies, because
    // handing out the live arrays would make every frame alias one buffer.
    await options.onStep?.({
      step, steps, noiseLevel: level, tHat,
      positions: Float32Array.from(positions),
      denoised: Float32Array.from(denoised.positions),
    });
  }
  return positions;
}

/**
 * The same denoiser, run as a structure module: no noise anywhere.
 *
 * Start every atom at the origin, walk sigma down the schedule, and feed each
 * output back in as the next input. No Gaussian initialisation, no churn, no
 * random augmentation - nothing is drawn.
 *
 * WHY IT WORKS AT ALL. AF3 returns skip * input + out * update, with
 * skip = sigma_d^2/(sigma^2+sigma_d^2), which is 3.9e-5 at the top of the
 * schedule. So the first call discards its input entirely and predicts the
 * structure from the trunk - a black hole is a perfectly good member of "this
 * input is noise, ignore it". Each later call is handed a better structure at a
 * lower sigma, which is exactly the claim the Fourier noise embedding makes
 * about it.
 *
 * WHAT IT COSTS AND SAVES, measured against the 200-step stochastic sampler:
 *
 *     6MRR   ramp 8 calls  0.69 A  TM 0.951   sampler 200  0.66 A  TM 0.953
 *     1QYS   ramp 8 calls  0.86 A  TM 0.947   sampler 200  0.93-1.12 A over
 *                                             four seeds, TM 0.916-0.941
 *
 * 🔴 SIGMA IS A CLAIM ABOUT THE INPUT, NOT A DIAL, WHICH IS WHY THE SCHEDULE
 * MUST DESCEND. Holding it fixed and iterating diverges: from zeros at sigma 16
 * the structure goes 9.12 A to 9.72 A, getting worse every round, because the
 * network is told its input is nearly correct when it is not and answers with
 * correspondingly small corrections. See tools/gpu/probe-denoiser.js.
 *
 * 🔴 AND IT RETURNS ONE STRUCTURE, FOR EVER. There is no seed and no ensemble:
 * the same trunk gives the same answer every time. That is the whole point of
 * diffusion thrown away, and it is why this is offered beside sampleOnGpu
 * rather than instead of it.
 *
 * @param {{cycles: number, onStep?: Function}} options
 */
export async function rampOnGpu(device, input, weights, options) {
  const { cycles } = options;
  const levels = noiseLevels(cycles, options);
  const head = new Af3DiffusionHeadGpu(device);
  let positions = new Float32Array(input.shape.tokens * input.shape.dense * 3);

  for (let cycle = 1; cycle <= cycles; cycle += 1) {
    const noiseLevel = levels[cycle - 1];
    const denoised = await head.run({ ...input, noiseLevel, positionsNoisy: positions },
                                    weights);
    positions = denoised.positions;
    // The same shape sampleOnGpu reports, so a caller animates either the same
    // way. Here the two tracks ARE the same array - there is no separate walk -
    // and no frame needs superposing, because nothing was ever rotated.
    await options.onStep?.({
      step: cycle, steps: cycles, noiseLevel, tHat: noiseLevel,
      positions: Float32Array.from(positions),
      denoised: Float32Array.from(positions),
    });
  }
  return positions;
}
