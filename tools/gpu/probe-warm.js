/**
 * What the sampler's first call is made of, and how much of it a warm covers.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-warm.js --tokens=200
 *
 * 🔴 fold.js BELIEVED THE HEAD'S CONSTRUCTOR COMPILED ITS PIPELINES. It does
 * not, so the whole first-call cost landed after the trunk with nothing left to
 * overlap it. This splits that cost: what `head.warm` takes (the diffusion
 * transformer's stack, which is the largest WGSL in the model), and what a
 * first denoiser call still pays after one - which is every OTHER stage's
 * compile, since those build their pipelines inside their own run().
 */
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights, atomReference } from "../../src/af3/diffusion-weights.js";
import { Af3DiffusionHeadGpu } from "../../src/af3/diffusion-head-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const ALPHABET = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "200"));
  const sequence = Array.from({ length: tokens },
    (_, i) => ALPHABET[i % ALPHABET.length]).join("");
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = await diffusionWeights(store);
  void await atomReference(store);
  void sequence;

  const head = new Af3DiffusionHeadGpu(device);
  const started = performance.now();
  await head.warm(tokens, weights);
  const warmMs = performance.now() - started;

  // ...a second warm, to show the memo holds and the first number is the build.
  const again = performance.now();
  await head.warm(tokens, weights);
  return {
    tokens,
    warmMs: Math.round(warmMs),
    secondWarmMs: Math.round(performance.now() - again),
  };
}
