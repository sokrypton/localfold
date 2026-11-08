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

/**
 * 🔴 THE SAMPLER'S HOST ARITHMETIC LIVES IN `cpu/bioemu2/sampler.js` NOW and is
 * re-exported here unchanged, so that `webgl/` can read it: the layering is
 * shared <- cpu <- a backend, and a backend may not import another backend. The
 * four helpers below were this file's; moving them is what stopped the WebGL2
 * port from re-deriving a schedule, a churn and a step scale of its own.
 */
import {
  bioemu2Schedule, bioemu2Batches, BIOEMU2_MAX_BATCH, sampleMany, centre,
} from "../../cpu/bioemu2/sampler.js";

export { bioemu2Schedule, bioemu2Batches, BIOEMU2_MAX_BATCH, sampleMany, centre };

/**
 * Fold one feature dict (shared/bioemu2/featurise.js's `features`).
 *
 * 🔴 AN ENSEMBLE IS SAMPLED IN BATCHES, ONE TRUNK FOR ALL OF THEM. The denoiser is
 * compiled for one batch (`bioemu2Batches`) and called batch after batch with the
 * draws continuing, so a hundred samples cost a hundred samples' sampling and one
 * trunk. `onSamples(frames, first)` hands each batch over as it lands, and a stop
 * part-way through - an AbortError out of `onProgress` once a batch has finished -
 * returns the finished batches with `stopped: true` rather than throwing them away.
 *
 * @param options `samples` (5), `batch` (the most a denoiser call carries, BIOEMU2_MAX_BATCH), `steps` (200),
 *   `seed`, `mode` (the features' own; decides the step scale),
 *   `propagation`: {x0 (real atoms x 3, the featuriser's order), lagNs, steps}, `onProgress(stage, done, total)`,
 *   `onSamples(frames, first)`, `half` / `trunkPrecision` (see Bioemu2Gpu)
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
    const { batches, size } = bioemu2Batches(samples, options.batch ?? BIOEMU2_MAX_BATCH);
    const frames = [];
    const dn = await clock("denoiser-compile", () => net.denoiserProgram(g, sIn, sTrunk, trunk.z, {
      samples: size, ...(propagation ? { dt: propagation.lagNs * 1000, dynamics: true } : {}) }));
    net.ops.free(trunk.z, dn.context.z, g.tokenInts, g.atomToken);
    const t0 = performance.now();
    let calls = 0;
    const denoise = async (x, sigma) => { calls += 1; return dn.denoise(x, sigma); };
    const total = (schedule.length - 1) * (propagation?.steps ?? 1) * batches;
    let done = 0;
    const onStep = (k, steps, denoised) => { done += 1; return progress("diffusion", done, total, denoised); };
    const common = { denoise, setCompanion: dn.setCompanion, mask, N, S: size, schedule, stepScale, normal, onStep };
    // one batch: `size` samples, or `size` trajectories of `propagation.steps` frames each
    const sampleBatch = async () => {
      if (!propagation) return (await sampleMany({ ...common, companions: null })).map((x) => centre(x, mask, N));
      // each sample its own trajectory, chained: step k+1 starts from step k's endpoint, superposed on its start
      let prev = Array.from({ length: size }, () => { const x = new Float32Array(N * 3); x.set(propagation.x0); return x; });
      const trajectories = Array.from({ length: size }, () => []);
      for (let k = 0; k < (propagation.steps ?? 1); k += 1) {
        const comps = prev.map((x) => centre(Float32Array.from(x), mask, N));
        const ends = await sampleMany({ ...common, companions: comps });
        prev = ends.map((end, s) => weightedRigidAlign(end, prev[s], Float32Array.from(mask), N));
        prev.forEach((x, s) => trajectories[s].push(Float32Array.from(x)));
      }
      // sample i of the output is trajectory i // steps, step i % steps + 1, as af3-any-model lays them out
      return trajectories.flat();
    };
    const perSample = propagation?.steps ?? 1;
    let stopped = false;
    for (let batch = 0; batch < batches; batch += 1) {
      let landed;
      try {
        landed = await sampleBatch();
      } catch (error) {
        if (error?.name !== "AbortError" || frames.length === 0) throw error;
        stopped = true;
        break;
      }
      // the last batch may carry samples nobody asked for (bioemu2Batches); they are dropped here
      landed = landed.slice(0, (samples - batch * size) * perSample);
      const first = frames.length;
      frames.push(...landed);
      await options.onSamples?.(landed, first);
    }
    timings.diffusion = Math.round(performance.now() - t0);
    return { samples: frames, contact, distogram: logits, tokens: g.T, atoms: N, liveAtoms: mask.reduce((a, b) => a + b, 0),
             calls, timings, mode, stepScale, stopped, batches, batchSize: size,
             deviceMiB: { activations: Math.round(net.ops.bytes / 1048576), weights: Math.round(net.ops.weightBytes / 1048576) } };
  } finally {
    if (owned) net.ops.destroyWeights();
    else net.ops.destroy();
  }
}
