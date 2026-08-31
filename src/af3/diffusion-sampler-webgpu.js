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
 *          gamma0?: number, gammaMin?: number, noiseScale?: number,
 *          stepScale?: number}} options
 */
export async function sampleOnGpu(device, input, weights, options) {
  const { steps, normal } = options;
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

  for (let step = 1; step <= steps; step += 1) {
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
