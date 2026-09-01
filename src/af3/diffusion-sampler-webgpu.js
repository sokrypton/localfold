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
 * The same denoiser, walked down the schedule deterministically: a flow.
 *
 * Draw the starting positions at the top of the schedule, walk sigma down, and
 * feed each output back in as the next input. No churn, no random augmentation
 * and no noise injected between steps - the ONLY randomness is where it
 * started, which is what makes this a flow rather than a sampler and what lets
 * a seed name a structure.
 *
 * WHY IT WORKS AT ALL. AF3 returns skip * input + out * update, with
 * skip = sigma_d^2/(sigma^2+sigma_d^2), which is 3.9e-5 at the top of the
 * schedule. So the first call all but discards its input and predicts the
 * structure from the trunk. Each later call is handed a better structure at a
 * lower sigma, which is exactly the claim the Fourier noise embedding makes
 * about it.
 *
 * 🔴 IT STARTS FROM NOISE RATHER THAN FROM ZEROS, AND THE DIFFERENCE IS NOT
 * ONLY DIVERSITY. Measured on one denoiser call at sigma 2560, a black hole
 * gives 1.39 A and Gaussian noise gives 1.15 A - and from zeros the bond
 * lengths come back at 73-87% of ideal, the signature of a conditional mean.
 * With nothing to break the symmetry the network averages over the posterior,
 * and averaging shrinks distances. Noise breaks it, and it also means a seed
 * names a structure and repeated folds give an ensemble.
 *
 * WHAT IT COSTS AND SAVES, measured against the 200-step stochastic sampler:
 *
 *     6MRR   flow 8 calls  0.69 A  TM 0.951   sampler 200  0.66 A  TM 0.953
 *     1QYS   flow 8 calls  0.86 A  TM 0.947   sampler 200  0.93-1.12 A over
 *                                             four seeds, TM 0.916-0.941
 *
 * (those numbers were measured from zeros; the seeded start is checked below
 *  them in the commit that introduced it)
 *
 * 🔴 SIGMA IS A CLAIM ABOUT THE INPUT, NOT A DIAL, WHICH IS WHY THE SCHEDULE
 * MUST DESCEND. Holding it fixed and iterating diverges: from zeros at sigma 16
 * the structure goes 9.12 A to 9.72 A, getting worse every round, because the
 * network is told its input is nearly correct when it is not and answers with
 * correspondingly small corrections. See tools/gpu/probe-denoiser.js.
 *
 * 🔴 IT IS STILL NOT A SAMPLER. One draw at the start is not the same as noise
 * injected at every step: this follows a deterministic path from wherever it
 * began, so its spread across seeds is narrower than the sampler's. It is
 * offered beside sampleOnGpu, not instead of it.
 *
 * @param {{cycles: number, normal: () => number, onStep?: Function}} options
 */
export async function flowOnGpu(device, input, weights, options) {
  const { cycles, normal } = options;
  // 🔴 THE FLOW STARTS AT 160 A, NOT AF3'S 2560, AND THAT IS A CHOICE WITH A
  // MEASURED COST. sigmaMax is in units of sigmaData, so 10 is 160 angstroms.
  // Most of AF3's schedule sits above the level where the denoiser starts
  // trusting the coordinates it is given - the skip weight is
  // sigma_d^2/(sigma^2+sigma_d^2) - so a walk from 2560 spends its first calls
  // on a regime a flow does not need, and a ligand pays for it: HEM's bond
  // error at eight steps is 0.218 A from 2560 and 0.129 A from 160.
  //
  // It is not free. Against the crystals at flow 8 over four seeds, 6MRR is
  // unchanged (0.703 -> 0.712 A, TM .950 -> .949) and 1QYS loses 0.04 A
  // (0.862 -> 0.905, TM .948 -> .944) with seed ranges that do not overlap.
  // See tools/gpu/probe-sigma0.js for the table and probe-ligand-flow.js for
  // the ligand side. AF3's own DIFFUSION sampler is untouched: it keeps the
  // schedule it was verified against, and noiseSchedule's defaults are still
  // AF3's, so only the flow - which is ours - moves.
  const levels = noiseLevels(cycles, { sigmaMax: 10, ...options });
  const head = new Af3DiffusionHeadGpu(device);
  let positions = new Float32Array(input.shape.tokens * input.shape.dense * 3);
  for (let index = 0; index < positions.length; index += 1) {
    positions[index] = normal() * levels[0];
  }

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
