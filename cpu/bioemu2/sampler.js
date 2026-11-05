/**
 * BioEmu2's plain EDM sampler, as host arithmetic: the schedule, the random
 * rigid augmentation, the centring, and the loop that walks S trajectories down
 * the schedule with one denoiser call a step.
 *
 * 🔴 THIS FILE IS A MOVE, NOT A REWRITE. Every line of it was
 * `webgpu/bioemu2/fold.js`'s - the only edits are the four `export` keywords a
 * second reader needs and one `@param {number}` the typechecker wants, since a
 * default of `BIOEMU2_SAMPLER.steps` off a frozen literal reads as the type
 * `200` - which still re-exports it, because a second
 * backend cannot import a backend - and a sampler re-derived for the second one
 * is how two backends come to anneal differently while both looking right:
 * nothing errors on a wrong constant, the walk simply returns a plausible
 * structure. `webgl/bioemu2/model/fold.js` reads it from here.
 *
 * 🔴 AND THE DENOISER IS THE CALLER'S, awaited: `denoise(xs, sigma)` takes the S
 * samples stacked and returns them denoised. A WebGL2 denoiser is synchronous
 * (`execution.read` is a `readPixels`) and a WebGPU one cannot be, so the loop
 * awaits and the synchronous one simply resolves at once.
 *
 * 🔴 AND THE STEERED SAMPLERS ARE SOMEWHERE ELSE: `shared/bioemu2/sampler.js`
 * carries edm_gd, edm_fk and edm_fkc, which are already generic over a denoiser
 * and over the draws. This is the plain walk those three are variations of.
 */
import { BIOEMU2, BIOEMU2_SAMPLER } from "../../shared/bioemu2/weights.js";
import { weightedRigidAlign } from "../af3/diffusion/rigid-align.js";

/**
 * The sampler's sigmas: `steps` Karras points and a zero.
 *
 * @param {number} [steps]
 */
export function bioemu2Schedule(steps = BIOEMU2_SAMPLER.steps) {
  if (steps < 2) throw new Error(`BioEmu2 samples with at least 2 steps, got ${steps}`);
  const { sigmaMin, sigmaMax, rho } = BIOEMU2_SAMPLER;
  const a = sigmaMax ** (1 / rho);
  const b = sigmaMin ** (1 / rho);
  const out = new Float32Array(steps + 1);
  for (let i = 0; i < steps; i += 1) {
    const t = Math.fround(i / (steps - 1));
    out[i] = BIOEMU2.sigmaData * (a + t * (b - a)) ** rho;
  }
  out[steps] = 0;
  return out;
}

export function randomRotation(normal) {
  let q = [normal(), normal(), normal(), normal()];
  const n = Math.hypot(...q);
  q = q.map((v) => v / n);
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

/** (x - masked mean) R + t, then * mask; in place. */
export function augment(x, mask, N, rot, trans) {
  let total = 0;
  const c = [0, 0, 0];
  for (let a = 0; a < N; a += 1) if (mask[a]) { total += 1; for (let k = 0; k < 3; k += 1) c[k] += x[a * 3 + k]; }
  for (let k = 0; k < 3; k += 1) c[k] /= Math.max(total, 1e-6);
  for (let a = 0; a < N; a += 1) {
    if (!mask[a]) { x[a * 3] = x[a * 3 + 1] = x[a * 3 + 2] = 0; continue; }
    const v = [x[a * 3] - c[0], x[a * 3 + 1] - c[1], x[a * 3 + 2] - c[2]];
    for (let j = 0; j < 3; j += 1) x[a * 3 + j] = v[0] * rot[j] + v[1] * rot[3 + j] + v[2] * rot[6 + j] + trans[j];
  }
}

export function centre(x, mask, N) {
  let total = 0;
  const c = [0, 0, 0];
  for (let a = 0; a < N; a += 1) if (mask[a]) { total += 1; for (let k = 0; k < 3; k += 1) c[k] += x[a * 3 + k]; }
  for (let a = 0; a < N; a += 1) {
    for (let k = 0; k < 3; k += 1) x[a * 3 + k] = mask[a] ? x[a * 3 + k] - c[k] / Math.max(total, 1) : 0;
  }
  return x;
}

/**
 * S trajectories of the sampler at once, from noise down the schedule: one denoiser call a step for all of them.
 * @param denoise (xs: S x N x 3, sigma) -> the same, denoised
 * @param companions S frames carried alongside and augmented with their sample, or null
 */
export async function sampleMany({ denoise, setCompanion, mask, N, S, schedule, stepScale, normal, companions, onStep }) {
  const xs = Array.from({ length: S }, () => {
    const x = new Float32Array(N * 3);
    for (let i = 0; i < N * 3; i += 1) x[i] = normal() * schedule[0];
    return x;
  });
  const comps = companions ? companions.map((c) => Float32Array.from(c)) : null;
  const weights = Float32Array.from(mask);
  const stacked = new Float32Array(S * N * 3);
  const steps = schedule.length - 1;
  for (let k = 0; k < steps; k += 1) {
    const prev = schedule[k];
    const next = schedule[k + 1];
    const gamma = next > BIOEMU2_SAMPLER.gammaMin ? BIOEMU2_SAMPLER.gamma0 : 0;
    const tHat = prev * (1 + gamma);
    const scale = BIOEMU2_SAMPLER.noiseScale * Math.sqrt(Math.max(0, tHat * tHat - prev * prev));
    for (let s = 0; s < S; s += 1) {
      const rot = randomRotation(normal);
      const trans = [normal(), normal(), normal()];
      augment(xs[s], mask, N, rot, trans);
      if (comps) augment(comps[s], mask, N, rot, trans);
      for (let i = 0; i < N * 3; i += 1) xs[s][i] += scale * normal();
      stacked.set(xs[s], s * N * 3);
    }
    if (comps) {
      const all = new Float32Array(S * N * 3);
      comps.forEach((c, s) => all.set(c, s * N * 3));
      setCompanion(all);
    }
    const denoised = await denoise(stacked, tHat);
    for (let s = 0; s < S; s += 1) {
      const d = denoised.subarray(s * N * 3, (s + 1) * N * 3);
      const aligned = weightedRigidAlign(xs[s], d, weights, N);
      const x = xs[s];
      for (let i = 0; i < N * 3; i += 1) {
        const grad = (aligned[i] - d[i]) / tHat;
        x[i] = aligned[i] + stepScale * (next - tHat) * grad;
      }
      for (let a = 0; a < N; a += 1) if (!mask[a]) x[a * 3] = x[a * 3 + 1] = x[a * 3 + 2] = 0;
    }
    await onStep?.(k + 1, steps, denoised);
  }
  return xs;
}

/**
 * The most samples one denoiser call carries. Five is the batch measured at 255 tokens (docs/BIOEMU2.md: 664 ms a
 * call); an ensemble larger than that is several calls of the same compiled program.
 */
export const BIOEMU2_MAX_BATCH = 5;

/**
 * How `samples` split into denoiser batches of at most `maxBatch`: as few batches as fit, all the same size, so a
 * batch never carries more than `batches - 1` samples nobody asked for (100 by 5 is 20 of 5; 12 by 5 is 3 of 4).
 */
export function bioemu2Batches(samples, maxBatch = BIOEMU2_MAX_BATCH) {
  if (!(samples >= 1) || !(maxBatch >= 1)) throw new Error(`BioEmu2 batches need a positive count, got ${samples}/${maxBatch}`);
  const batches = Math.ceil(samples / maxBatch);
  return { batches, size: Math.ceil(samples / batches) };
}
