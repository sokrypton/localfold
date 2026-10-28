/**
 * AF3's diffusion sampler, driving the GPU denoiser.
 *
 * The schedule, the churn and the step update are pure arithmetic and already
 * live in cpu/af3/diffusion/diffusion-sampler.js, checked against AF3. This is
 * the same loop with an AWAITED denoiser, because the reference's is
 * synchronous and a GPU one cannot be.
 *
 * 🔴 THE ONLY REASON THIS IS NOT `sample()` WITH A DIFFERENT CALLBACK. Keeping
 * the arithmetic in one place matters more than avoiding the duplicated loop:
 * every constant here - step scale 1.5, gamma 0.8 above gamma_min 1.0, noise
 * scale 1.003 - is imported rather than restated.
 */
import { noiseLevels, randomAugmentation, samplerStep } from "../../../cpu/af3/diffusion/diffusion-sampler.js";
import { weightedRigidAlign } from "../../../cpu/af3/diffusion/rigid-align.js";
import { Af3DiffusionHeadGpu } from "./diffusion-head.js";

/**
 * @param {object} input everything the diffusion head takes except noiseLevel
 *   and positionsNoisy
 * @param {object} weights the diffusion head's
 * @param {{steps: number, normal: () => number, onStep?: Function,
 *          stopAfter?: number, gamma0?: number, gammaMin?: number,
 *          noiseScale?: number, stepScale?: number, schedule?: "karras"|"geometric",
 *          update?: "euler"|"exponential", churn?: boolean,
 *          start?: {positions: Float32Array, level: number}}} options
 *   `start` begins the walk part-way down: at the first scheduled level at or below `level`
 *   (angstroms), from `positions` (the dense layout) plus that much noise, running only the steps
 *   left. RFdiffusion's partial diffusion, on AF3's schedule.
 */
export async function sampleOnGpu(device, input, weights, options) {
  const { steps, normal } = options;
  // 🔴 THE SCHEDULE'S LENGTH AND THE NUMBER OF DENOISER CALLS ARE NOT THE SAME
  // THING. `steps` sets the discretisation - level i is noiseSchedule(i/steps),
  // so a coarse schedule is not a subset of a fine one but a different set of
  // sigmas. `stopAfter` runs a PREFIX of the schedule and leaves the walk at
  // high noise, which is only useful if what you read out is `denoised` rather
  // than the returned sample.
  // (a schedule tail can add a step: run what the levels hold - noiseLevels)
  const stopAfter = Math.min(options.stopAfter ?? Infinity, noiseLevels(steps, { ...(input.dialect?.sampler ?? {}), ...options }).length - 1);
  // 🔴 THE MODEL'S OWN EDM CONSTANTS, WHERE IT HAS ANY. A caller's explicit
  // option still wins; the dialect fills in behind it and AlphaFold 3's
  // defaults fill in behind that. boltz2 is the first model here with its own
  // and the difference is not small - gamma_0 0.605 against 0.8, step_scale
  // 1.638 against 1.5, rho 8 against 7 - and NOTHING ERRORS on the wrong one:
  // the walk simply anneals differently and returns a plausible structure. On
  // AF3's constants boltz2's backbone bonds come out at 9.6 A against an ideal
  // 1.46.
  const model = input.dialect?.sampler ?? {};
  // (schedule, update and churn: cpu/af3/diffusion/diffusion-sampler.js's, read through noiseLevels and samplerStep)
  const gamma0 = options.churn === false ? 0 : options.gamma0 ?? model.gamma0 ?? 0.8;
  const gammaMin = options.gammaMin ?? model.gammaMin ?? 1.0;
  const noiseScale = options.noiseScale ?? model.noiseScale ?? 1.003;
  const stepScale = options.stepScale ?? model.stepScale ?? 1.5;
  const atoms = input.shape.tokens * input.shape.dense;
  const levels = noiseLevels(steps, { ...model, ...options });
  // 🔴 A CALLER MAY OWN THE HEAD, AND THEN IT KEEPS IT. Building one compiles
  // its pipelines - measured 730 ms, flat in the shape - so a caller that
  // denoises repeatedly (fold.js previews one step per recycle) would pay that
  // every time. `options.head` is borrowed: used and NOT disposed here.
  const borrowed = options.head;
  const head = borrowed ?? new Af3DiffusionHeadGpu(device);

  // 🔴 A WARM START SKIPS THE LARGE NOISE, IT DOES NOT RESCALE THE SCHEDULE (2026-10-28). The walk
  // joins the same levels a cold fold visits at the first one at or below `start.level`, so the
  // last steps - where side chains and bonds resolve - are exactly a cold fold's, and the steps
  // above are simply not taken: fewer denoiser calls, not the same number squeezed lower.
  let first = 0;
  if (options.start !== undefined) {
    const { positions: from, level } = options.start;
    if (!(from instanceof Float32Array) || from.length !== atoms * 3) {
      throw new Error(`a warm start needs ${atoms * 3} coordinates, got ${from?.length}`);
    }
    if (!(level > 0)) throw new Error(`a warm start needs a noise level in angstroms, not ${level}`);
    first = levels.findIndex((sigma) => sigma <= level);
    if (first < 0 || first >= levels.length - 1) {
      throw new Error(`no step of this schedule is at or below ${level} A but above its last`);
    }
  }
  let positions = new Float32Array(atoms * 3);
  for (let index = 0; index < positions.length; index += 1) {
    positions[index] = (options.start?.positions[index] ?? 0) + normal() * levels[first];
  }
  let previous = levels[first];

  try {
  for (let step = first + 1; step <= stopAfter; step += 1) {
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
    // (boltz2 rigid-aligns the noisy copy onto the answer before the step: dialect.realignSampler)
    const aligned = input.dialect?.realignSampler === true
      ? weightedRigidAlign(noisy, denoised.positions, input.atomMask, atoms) : noisy;
    positions = samplerStep(aligned, denoised.positions, tHat, level, { stepScale, update: options.update });
    previous = level;

    // 🔴 `denoised` IS THE FRAME WORTH WATCHING and neither is in a stable
    // frame - see the note in diffusion-sampler.js. Copies, because
    // handing out the live arrays would make every frame alias one buffer.
    await options.onStep?.({
      step, steps, noiseLevel: level, tHat,
      positions: Float32Array.from(positions),
      denoised: Float32Array.from(denoised.positions),
    });
  }
  return positions;
  } finally {
    // 🔴 THE HEAD HOLDS ~25 MB OF DEVICE MEMORY BETWEEN CALLS, deliberately -
    // the atom encoder's static tensors - and a head is built per run.
    if (borrowed === undefined) head.dispose();
  }
}
