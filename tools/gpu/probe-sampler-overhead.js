/**
 * What a sampler STEP costs that the denoiser call does not.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-sampler-overhead.js
 *     node tools/gpu-chrome.mjs tools/gpu/probe-sampler-overhead.js --tokens=150 --steps=12
 *
 * WHY IT EXISTS. AF3.md records a denoiser call at 123 ms inside the sampler
 * and 111 on tools/gpu/bench-head.js's bench, attributes the 12 ms to "the
 * sampler's own per-step work - a random augmentation of every atom, the noise
 * injection, the Euler step and two copies of the coordinates" and ends "nobody
 * has looked at it". It is 2.4 s of a 200-step fold.
 *
 * That attribution is checkable by counting: at 59 tokens the augmentation
 * touches 1,416 atoms and the other three touch 4,248 floats each, so the whole
 * of it is about fifteen thousand float operations - microseconds, not
 * milliseconds. This times each phase separately rather than assuming.
 *
 * 🔴 IT TIMES THE HOST, WHICH IS THE POINT. `head.run` is awaited, so a step is
 * a full round trip: submit, drain, map the positions back, do host arithmetic,
 * upload again. `wallPerStep - headRun` is what the loop costs AROUND the
 * denoiser and `headRun - gpu` is what the round trip costs inside it.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { Af3DiffusionHeadGpu } from "../../src/af3/diffusion-head-webgpu.js";
import { normalFrom } from "../../src/af3/fold.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference } from "../../src/af3/diffusion-weights.js";
import { noiseLevels, randomAugmentation, samplerStep } from "../../src/af3/diffusion-sampler-reference.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "59"));
  const steps = Number(option(args, "steps", "12"));
  const withCallback = !args.includes("--no-callback");
  const sequence = Array.from({ length: tokens },
    (_, index) => ALPHABET[index % ALPHABET.length]).join("");

  const batch = featuriseProtein(sequence, {});
  const { dense } = batch;
  const store = await openAf3Store(option(args, "model", "/model-af3-full-f32/manifest.json"));
  const weights = await diffusionWeights(store);
  const reference = await atomReference(store);

  const noise = normalFrom(11);
  const fill = (length) => {
    const out = new Float32Array(length);
    for (let index = 0; index < length; index += 1) out[index] = noise();
    return out;
  };
  const input = {
    shape: batch.shape,
    conditioning: perAtomConditioning({
      positions: batch.refPos, mask: batch.refMask, element: batch.refElement,
      charge: batch.refCharge, atomNameChars: batch.refAtomNameChars,
    }, tokens, dense, reference),
    atomMask: batch.predDenseAtomMask, seqMask: batch.seqMask, features: batch.features,
    targetFeat: fill(tokens * 447),
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries, queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries, tokensToKeys: batch.tokensToKeys,
    trunkSingle: fill(tokens * 384), trunkPair: fill(tokens * tokens * 128),
  };

  const atoms = tokens * dense;
  const levels = noiseLevels(steps, {});
  const head = new Af3DiffusionHeadGpu(device);
  const normal = normalFrom(7);
  let positions = new Float32Array(atoms * 3);
  for (let index = 0; index < positions.length; index += 1) positions[index] = normal() * levels[0];
  let previous = levels[0];

  const phases = { augment: [], inject: [], spread: [], headRun: [], euler: [], copies: [], wall: [], gpu: [] };
  try {
    for (let step = 1; step <= steps; step += 1) {
      const level = levels[step];
      const stepStart = performance.now();

      let mark = performance.now();
      positions = randomAugmentation(positions, input.atomMask, atoms, normal);
      const augment = performance.now() - mark;

      const gamma = level > 1.0 ? 0.8 : 0;
      const tHat = previous * (1 + gamma);
      const injected = 1.003 * Math.sqrt(Math.max(0, tHat * tHat - previous * previous));
      mark = performance.now();
      const noisy = new Float32Array(positions.length);
      for (let index = 0; index < noisy.length; index += 1) {
        noisy[index] = positions[index] + injected * normal();
      }
      const inject = performance.now() - mark;

      // The object spread is per step and copies every key, including the
      // trunk's pair tensor by reference - cheap, but it is on the list.
      mark = performance.now();
      const call = { ...input, noiseLevel: tHat, positionsNoisy: noisy };
      const spreadMs = performance.now() - mark;

      mark = performance.now();
      const denoised = await head.run(call, weights);
      const headRun = performance.now() - mark;

      mark = performance.now();
      positions = samplerStep(noisy, denoised.positions, tHat, level, { stepScale: 1.5 });
      const euler = performance.now() - mark;

      mark = performance.now();
      if (withCallback) {
        // What fold.js hands the trajectory callback, minus the callback itself.
        void Float32Array.from(positions);
        void Float32Array.from(denoised.positions);
      }
      const copies = performance.now() - mark;

      previous = level;
      if (step === 1) continue;   // the first call compiles every pipeline
      phases.augment.push(augment);
      phases.inject.push(inject);
      phases.spread.push(spreadMs);
      phases.headRun.push(headRun);
      phases.euler.push(euler);
      phases.copies.push(copies);
      phases.wall.push(performance.now() - stepStart);
      phases.gpu.push(Object.values(denoised.timings).reduce((a, b) => a + b, 0));
    }
  } finally {
    head.dispose();
  }

  const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    return Number(sorted[Math.floor(sorted.length / 2)].toFixed(3));
  };
  const report = Object.fromEntries(Object.entries(phases).map(([k, v]) => [k, median(v)]));
  return {
    tokens, atoms, steps, timedSteps: phases.wall.length, withCallback,
    // Everything the loop does that is not the denoiser call.
    hostArithmetic: Number((report.augment + report.inject + report.euler + report.copies)
      .toFixed(3)),
    outsideHeadRun: Number((report.wall - report.headRun).toFixed(3)),
    insideHeadRunNotGpu: Number((report.headRun - report.gpu).toFixed(3)),
    phases: report,
  };
}
