/**
 * AF3's diffusion sampler in WebGL2: the loop that turns the denoiser into a
 * structure.
 *
 * 🔴 EVERY PIECE OF ARITHMETIC HERE IS IMPORTED, NOT RESTATED. The schedule, the
 * churn, the random augmentation and the step update live in
 * `cpu/af3/diffusion/diffusion-sampler.js`, checked against AF3's own next
 * positions by `tools/oracle/check_af3_sampler.js`; `webgpu/`'s sampler imports
 * the same four and says why: "every constant here - step scale 1.5, gamma 0.8
 * above gamma_min 1.0, noise scale 1.003 - is imported rather than restated".
 * Re-deriving them for a third backend is how two backends come to anneal
 * differently, and NOTHING ERRORS on the wrong constant - the walk simply
 * returns a plausible structure.
 *
 * 🔴 AND THIS LOOP IS SYNCHRONOUS WHERE WebGPU'S CANNOT BE. `webgpu/`'s own
 * header says the duplicated loop exists only because its denoiser must be
 * awaited; `execution.read` is a `readPixels`, so this backend really can run
 * `cpu`'s own `sample()` with a device-backed callback - and
 * `denoiserFromHead` is exported so that it can, which is what makes this
 * loop's extra options falsifiable: the two must agree EXACTLY on the plain
 * path, from the same draws.
 *
 * 🔴 THE POSITIONS CROSS THE BUS EVERY STEP, AND THAT IS THE RIGHT TRADE HERE.
 * One upload and one readback of `atoms * 3` floats a step - 3,696 floats at 308
 * atoms - against a denoiser that is 963 draws. Keeping the walk on the device
 * would need a gaussian this API does not have (no PRNG, no workgroup
 * reduction), so the draws would cross the bus anyway and only the masked-mean
 * centre would stay.
 *
 * 🔴 AND NEITHER FRAME IS IN A STABLE COORDINATE SYSTEM. `randomAugmentation`
 * applies a fresh random rotation and translation at the top of EVERY step, so
 * consecutive frames are unrelated; superimpose each on the one before it before
 * showing them. `denoised` is the frame worth watching - "it is the model's
 * current guess at the finished structure, so it goes from blob to fold" - and
 * `positions` at step 1 of 200 is a cloud 2560 A across.
 */
import {
  noiseLevels, randomAugmentation, samplerStep,
} from "../../../cpu/af3/diffusion/diffusion-sampler.js";
import { weightedRigidAlign } from "../../../cpu/af3/diffusion/rigid-align.js";
import { steerDenoised } from "../../../cpu/af3/diffusion/steering.js";
import { inpaintDenoised } from "../../../cpu/af3/diffusion/inpaint.js";
import { diffusionHeadSteps } from "./head.js";
import { pairConditioning } from "./conditioning.js";
import { drain } from "../../runtime/steps.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */
/**
 * @typedef {Omit<import("./head.js").HeadCall, "positionsNoisy" | "noiseLevel">
 *   & {conditioningPair?: Tensor, hoistPairConditioning?: boolean}} DenoiserInput
 */
/**
 * @typedef {{binder: number[], pocket: number[], weight: number,
 *   cutoff?: number, beta?: number}} Steer
 */

/**
 * The denoiser as the reference's `sample()` wants it: host positions and a
 * noise level in, host positions out.
 *
 * The caller's `input` holds every tensor the head takes except the noisy
 * positions, and they stay on the device across the whole walk - only the
 * coordinates move.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {DenoiserInput} input everything `diffusionHead` takes but
 *   `positionsNoisy` and `noiseLevel`
 * @returns {(positions: Float32Array, noiseLevel: number) => Float32Array}
 *   with a `release()` the CALLER must call when the walk is over: the denoiser
 *   holds the hoisted pair conditioning for its whole life, and
 *   `sampleOnWebGl2` releases only a denoiser it built itself.
 */
export function denoiserFromHead(execution, input) {
  const { tokens, dense } = input.shape;
  const atoms = tokens * dense;
  // 🔴 THE PAIR CONDITIONING ONCE FOR THE WHOLE WALK, because it does not depend
  // on the noise level - see `pairConditioning`. The caller may pass its own
  // (`input.conditioningPair`); otherwise this builds one and the denoiser owns
  // it for as long as the walk, which is the point.
  // `hoistPairConditioning: false` is the arm that prices it, and nothing else
  // should pass it: the hoisted and the recomputed pair are the same draws on
  // the same inputs, so the only difference is how many times they run.
  const conditioningPair = input.hoistPairConditioning === false ? undefined
    : input.conditioningPair ?? pairConditioning(execution, {
      trunkPair: input.trunkPair, identity: input.identity, tokens,
      weights: input.weights.conditioning, relativeWidth: input.relativeWidth,
      epsilon: input.epsilon,
    });
  // `issueSteps` draws the head, yielding between its pieces, and returns its read: a caller that
  // waits for the GPU at each yield (sampleOnWebGl2Steps) hands the head over in pieces and reads a
  // finished result instead of blocking on the whole head.
  const issueSteps = function* (positions, noiseLevel) {
    if (positions.length !== atoms * 3) {
      throw new RangeError(
        `af3 sampler: ${positions.length} coordinates, not ${atoms} x 3`);
    }
    const uploaded = execution.upload(
      positions instanceof Float32Array ? positions : Float32Array.from(positions),
      "af3.sampler.noisy");
    const out = yield* diffusionHeadSteps(execution, {
      ...input, positionsNoisy: uploaded, noiseLevel,
      ...(conditioningPair === undefined ? {} : { conditioningPair }),
    });
    return () => {
      const values = Float32Array.from(execution.read(out));
      execution.release(out);
      execution.release(uploaded);
      return values;
    };
  };
  const denoise = (positions, noiseLevel) => drain(issueSteps(positions, noiseLevel))();
  denoise.issueSteps = issueSteps;
  // ...and the walk's own copy goes back when the walk does.
  denoise.release = () => {
    if (input.conditioningPair === undefined && conditioningPair !== undefined) {
      execution.release(conditioningPair);
    }
  };
  return denoise;
}

/**
 * The whole sampler, with the options `webgpu/`'s carries: a warm start, a
 * prefix of the schedule, a contact restraint and a held motif.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {DenoiserInput} input everything `diffusionHead` takes but
 *   `positionsNoisy` and `noiseLevel`
 * @param {{steps: number, normal: () => number, mask: Float32Array,
 *   onStep?: (frame: object) => void, stopAfter?: number,
 *   gamma0?: number, gammaMin?: number, noiseScale?: number, stepScale?: number,
 *   schedule?: "karras" | "geometric", scheduleTail?: number,
 *   update?: "euler" | "exponential", churn?: boolean, realign?: boolean,
 *   sigmaData?: number, sigmaMin?: number, sigmaMax?: number, rho?: number,
 *   start?: {positions: Float32Array, level: number},
 *   steer?: Steer, motif?: {slots: Int32Array, xyz: Float32Array, until?: number},
 *   denoise?: (positions: Float32Array, noiseLevel: number) => Float32Array}} options
 *   `mask` is the atom mask as a HOST array, because `randomAugmentation`'s
 *   centre is a masked mean over it; `denoise` replaces the device head, which
 *   is how the loop's own arithmetic is checked without spending 963 draws a
 *   step on a multiplication.
 * @returns {Float32Array} `atoms * 3`, the sample
 */
export function sampleOnWebGl2(execution, input, options) {
  return drain(sampleOnWebGl2Steps(execution, input, options));
}

/**
 * The walk as a generator: yields `{ stage: "step", step }` once each denoiser
 * call is drawn and before it is read, so a page can wait for the GPU without
 * holding the thread, paint, and take a Stop (webgl/runtime/steps.js); returns
 * what `sampleOnWebGl2` returns.
 */
export function* sampleOnWebGl2Steps(execution, input, options) {
  const { normal, mask } = options;
  const { tokens, dense } = input.shape;
  const atoms = tokens * dense;
  if (!(mask instanceof Float32Array) || mask.length !== atoms) {
    throw new RangeError(
      "af3 sampler: `mask` is the atom mask as a HOST array of"
      + ` ${atoms} floats - randomAugmentation's centre is a masked mean over it`);
  }
  // 🔴 EVERY CONSTANT FROM THE REFERENCE'S OWN DEFAULTS, and a caller's
  // explicit option first. boltz2 is the first model with its own - gamma_0
  // 0.605 against 0.8, step_scale 1.638, rho 8 - and on AF3's its backbone
  // bonds come out at 9.6 A against an ideal 1.46.
  const gamma0 = options.churn === false ? 0 : options.gamma0 ?? 0.8;
  const gammaMin = options.gammaMin ?? 1.0;
  const noiseScale = options.noiseScale ?? 1.003;
  const stepScale = options.stepScale ?? 1.5;
  const levels = noiseLevels(options.steps, options);
  // 🔴 THE SCHEDULE'S LENGTH AND THE NUMBER OF DENOISER CALLS ARE NOT THE SAME
  // THING: a tail can add a step, so the loop runs what the LEVELS hold.
  const stopAfter = Math.min(options.stopAfter ?? Infinity, levels.length - 1);
  const denoise = options.denoise ?? denoiserFromHead(execution, input);
  const ownsDenoiser = options.denoise === undefined;

  // 🔴 A WARM START SKIPS THE LARGE NOISE, IT DOES NOT RESCALE THE SCHEDULE: the
  // walk joins the same levels a cold run visits, at the first one at or below
  // `start.level`, so the last steps - where side chains and bonds resolve - are
  // exactly a cold run's and the steps above are simply not taken.
  let first = 0;
  if (options.start !== undefined) {
    const { positions: from, level } = options.start;
    if (!(from instanceof Float32Array) || from.length !== atoms * 3) {
      throw new RangeError(
        `af3 sampler: a warm start needs ${atoms * 3} coordinates, got ${from?.length}`);
    }
    if (!(level > 0)) {
      throw new RangeError(
        `af3 sampler: a warm start needs a noise level in angstroms, not ${level}`);
    }
    first = levels.findIndex((sigma) => sigma <= level);
    if (first < 0 || first >= levels.length - 1) {
      throw new RangeError(
        `af3 sampler: no step of this schedule is at or below ${level} A but above its last`);
    }
  }

  let positions = new Float32Array(atoms * 3);
  for (let index = 0; index < positions.length; index += 1) {
    positions[index] = (options.start?.positions[index] ?? 0) + normal() * levels[first];
  }
  let previous = levels[first];

  for (let step = first + 1; step <= stopAfter; step += 1) {
    const level = levels[step];
    positions = randomAugmentation(positions, mask, atoms, normal);

    const gamma = level > gammaMin ? gamma0 : 0;
    const tHat = previous * (1 + gamma);
    // ...exactly zero whenever gamma is, because t_hat then equals `previous`.
    const injected = noiseScale * Math.sqrt(Math.max(0, tHat * tHat - previous * previous));
    const noisy = new Float32Array(positions.length);
    for (let index = 0; index < noisy.length; index += 1) {
      noisy[index] = positions[index] + injected * normal();
    }

    // The head drawn, then a yield (the GPU waited on there, not in the read), then its result.
    let raw;
    if (denoise.issueSteps === undefined) {
      yield { stage: "step", step };
      raw = denoise(noisy, tHat);
    } else {
      const read = yield* denoise.issueSteps(noisy, tHat);
      yield { stage: "step", step };
      raw = read();
    }
    // 🔴 A RESTRAINT DISPLACES THE PREDICTION, NOT THE WALK: the update moves
    // toward what it is handed, so the descent step goes in here and the
    // denoiser itself is untouched.
    const steered = options.steer === undefined ? raw
      : steerDenoised(raw, mask, dense, options.steer,
        (step - first - 1) / (levels.length - 1 - first));
    const denoised = options.motif === undefined || !(tHat > (options.motif.until ?? 0))
      ? steered : inpaintDenoised(steered, options.motif);
    // boltz2 and ESMFold2 rigid-align the noisy copy onto the answer first.
    const aligned = options.realign === true
      ? weightedRigidAlign(noisy, denoised, mask, atoms) : noisy;
    positions = samplerStep(aligned, denoised, tHat, level, { stepScale, update: options.update });
    previous = level;

    // Copies, because handing out the live arrays would make every frame alias
    // one buffer.
    options.onStep?.({
      step, steps: options.steps, noiseLevel: level, tHat,
      positions: Float32Array.from(positions),
      denoised: Float32Array.from(denoised),
    });
  }
  if (ownsDenoiser) /** @type {any} */ (denoise).release?.();
  return positions;
}
