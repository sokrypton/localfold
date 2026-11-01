/**
 * A whole BioEmu2 fold on the GPU: features in, samples out.
 *
 * The sampler is BioEmu2's EDMSampler as af3-any-model runs it (diffusion_head's
 * loop with REALIGN_SAMPLER, model_registry's bioemu2 constants): Karras sigmas
 * 16 * (160^(1/7) + t (4e-4^(1/7) - 160^(1/7)))^7 at `steps` points with a zero
 * appended; per step a random rigid augmentation of the state, gamma 0.8 keyed
 * on the NEXT sigma above 1.0, the noise, one denoiser call, the noisy state
 * rigid-aligned ONTO the denoised one, then an Euler step scaled by 1.5
 * (structure) or 1.0 (equilibrium). Each sample is centred on its real atoms.
 *
 * 🔴 PROPAGATION carries a COMPANION frame through the sampler: the previous
 * frame, centred, which the denoiser reads as `coords_cond` and which takes the
 * SAME rotation and translation as the state every step, so the two stay in one
 * frame (bioemu2's edm_sampler augments coords_cond with x). Its endpoint is
 * superposed on that frame, and the next step starts from it.
 *
 * 🔴 IT IS STOCHASTIC AND ITS DRAWS ARE OURS, so no run reproduces the
 * vendor's coordinates; a fold is held to the crystal and to bond geometry, and
 * the network to the oracle stage by stage (tools/gpu/check-bioemu2-network.js).
 */
import { Bioemu2Gpu } from "./network.js";
import { BIOEMU2, BIOEMU2_SAMPLER } from "../../shared/bioemu2/weights.js";
import { normalFrom } from "../../shared/af3/random.js";
import { weightedRigidAlign } from "../../cpu/af3/diffusion/rigid-align.js";

/** The sampler's sigmas: `steps` Karras points and a zero. */
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

function randomRotation(normal) {
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
function augment(x, mask, N, rot, trans) {
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

function centre(x, mask, N) {
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
async function sampleMany({ denoise, setCompanion, mask, N, S, schedule, stepScale, normal, companions, onStep }) {
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
 * Fold one feature dict (shared/bioemu2/featurise.js's `features`).
 *
 * @param options `samples` (5), `steps` (200), `seed`, `mode` (the features' own; decides the step scale),
 *   `propagation`: {x0 (real atoms x 3, the featuriser's order), lagNs, steps}, `onProgress(stage, done, total)`,
 *   `half` / `trunkPrecision` (see Bioemu2Gpu)
 */
export async function foldBioemu2(device, read, features, options = {}) {
  const f = features;
  // a caller that folds again hands its network back in (`options.network`), and its weights stay on the device
  const owned = options.network === undefined;
  const net = options.network ?? new Bioemu2Gpu(device, read, options);
  const timings = {};
  const clock = async (name, fn) => {
    const t0 = performance.now();
    const value = await fn();
    timings[name] = Math.round(performance.now() - t0);
    return value;
  };
  try {
    const g = net.upload(f);
    const progress = options.onProgress ?? (() => {});
    const sIn = await clock("embed", () => net.inputEmbedder(g));
    const trunk = await clock("trunk-compile", () => net.trunkProgram(g, sIn));
    // the page's recycle dial, plus one; BioEmu2's own load_model runs num_steps_recycle 3, range(3 + 1) = four
    const passes = options.passes ?? BIOEMU2.trunkPasses;
    await clock("trunk", async () => {
      for (let pass = 0; pass < passes; pass += 1) {
        await trunk.pass((done, total) => progress("trunk", pass * total + done, passes * total));
      }
    });
    const sTrunk = await net.ops.read(trunk.s);
    trunk.free();
    const { contact, logits } = await clock("distogram", () => net.distogram(g, trunk.z));
    options.onContacts?.(contact);
    const N = g.N;
    const mask = Int32Array.from(f.atom_mask);
    const propagation = options.propagation ?? null;
    const mode = propagation ? "equilibrium" : (options.mode ?? (f.method_feature[0] === 4 ? "equilibrium" : "structure"));
    const stepScale = options.stepScale ?? BIOEMU2_SAMPLER.stepScale[mode];
    const schedule = bioemu2Schedule(options.steps ?? BIOEMU2_SAMPLER.steps);
    const normal = normalFrom(options.seed ?? 1);
    const samples = options.samples ?? 5;
    const frames = [];
    const dn = await clock("denoiser-compile", () => net.denoiserProgram(g, sIn, sTrunk, trunk.z, {
      samples, ...(propagation ? { dt: propagation.lagNs * 1000, dynamics: true } : {}) }));
    net.ops.free(trunk.z, dn.context.z, g.tokenInts, g.atomToken);
    const t0 = performance.now();
    let calls = 0;
    const denoise = async (x, sigma) => { calls += 1; return dn.denoise(x, sigma); };
    const total = (schedule.length - 1) * (propagation?.steps ?? 1);
    let done = 0;
    const onStep = (k, steps, denoised) => { done += 1; return progress("diffusion", done, total, denoised); };
    const common = { denoise, setCompanion: dn.setCompanion, mask, N, S: samples, schedule, stepScale, normal, onStep };
    if (!propagation) {
      for (const x of await sampleMany({ ...common, companions: null })) frames.push(centre(x, mask, N));
    } else {
      // each sample its own trajectory, chained: step k+1 starts from step k's endpoint, superposed on its start
      let prev = Array.from({ length: samples }, () => { const x = new Float32Array(N * 3); x.set(propagation.x0); return x; });
      const trajectories = Array.from({ length: samples }, () => []);
      for (let k = 0; k < (propagation.steps ?? 1); k += 1) {
        const comps = prev.map((x) => centre(Float32Array.from(x), mask, N));
        const ends = await sampleMany({ ...common, companions: comps });
        prev = ends.map((end, s) => weightedRigidAlign(end, prev[s], Float32Array.from(mask), N));
        prev.forEach((x, s) => trajectories[s].push(Float32Array.from(x)));
      }
      // sample i of the output is trajectory i // steps, step i % steps + 1, as af3-any-model lays them out
      for (const trajectory of trajectories) frames.push(...trajectory);
    }
    timings.diffusion = Math.round(performance.now() - t0);
    return { samples: frames, contact, distogram: logits, tokens: g.T, atoms: N, liveAtoms: mask.reduce((a, b) => a + b, 0),
             calls, timings, mode, stepScale,
             deviceMiB: { activations: Math.round(net.ops.bytes / 1048576), weights: Math.round(net.ops.weightBytes / 1048576) } };
  } finally {
    if (owned) net.ops.destroyWeights();
    else net.ops.destroy();
  }
}
