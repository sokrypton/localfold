/**
 * What AF3's confidence head costs, on its own, at several lengths.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-confidence.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-confidence.js --tokens=300
 *
 * WHY IT EXISTS. The head runs ONCE per fold, so it has never been on anyone's
 * profile - `bench-head.js` is the denoiser and `bench-trunk.js` stops at the
 * distogram. The question that needs it is whether a diffusion FRAME could be
 * coloured by confidence, which means running this per frame rather than once,
 * and that is a question about its absolute cost against a denoiser call's.
 *
 * 🔴 EVERY PART OF IT DEPENDS ON THE COORDINATES, so there is nothing to cache
 * across frames. `pseudoBeta` enters at the pair embedding, as a one-hot over
 * distance bins, and the four pairformer blocks and all three heads run on the
 * result. The trunk's own single and pair go in beside it, but they are added
 * to a tensor the coordinates have already changed.
 *
 * Synthetic trunk outputs and REAL weights, for the reason bench-head.js gives:
 * the cost depends on the shapes and not on the values, and loading a trunk
 * first would cost more than the thing being measured.
 */
import { Af3ConfidenceHeadGpu } from "../../src/af3/confidence-webgpu.js";
import { confidenceWeights, openAf3Store } from "../../src/af3/weights.js";
import { profileDevice } from "./profile.js";

const DIALECT = { swapTransposedBias: false };
const DENSE = 24;

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function noise(count, seed) {
  let state = seed >>> 0;
  const out = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[index] = (state / 4294967296) - 0.5;
  }
  return out;
}

export async function main(device, args) {
  const lengths = option(args, "tokens", "59,150,300").split(",").map(Number);
  const calls = Number(option(args, "calls", "5"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = await confidenceWeights(store);
  const head = new Af3ConfidenceHeadGpu(device);
  // --profile times every labelled compute pass; see tools/gpu/profile.js.
  const profile = args.includes("--profile") ? profileDevice(device) : null;

  const rows = [];
  for (const tokens of lengths) {
    const input = {
      tokens, dense: DENSE,
      seqMask: new Float32Array(tokens).fill(1),
      pair: noise(tokens * tokens * weights.pairChannels, 1),
      single: noise(tokens * weights.singleChannels, 2),
      targetFeat: noise(tokens * weights.targetFeatWidth, 3),
      // ...spread over a plausible radius of gyration, so the distance bins are
      // populated the way a real structure populates them.
      pseudoBeta: noise(tokens * 3, 4).map((v) => v * 40),
    };
    const times = [];
    for (let call = 0; call < calls; call += 1) {
      // ...the last call only, so pipeline compilation is out of the numbers.
      if (profile !== null && call === calls - 1) profile.reset();
      const started = performance.now();
      await head.run(input, weights, DIALECT);
      times.push(performance.now() - started);
    }
    // ...the first compiles every pipeline; the rest are the number.
    const steady = times.slice(1).sort((a, b) => a - b);
    rows.push({
      tokens,
      first: Math.round(times[0]),
      median: Math.round(steady[Math.floor(steady.length / 2)]),
      range: `${Math.round(steady[0])}-${Math.round(steady[steady.length - 1])}`,
    });
  }
  const gpuPasses = profile === null ? undefined : await profile.report();
  profile?.restore();
  return {
    dense: DENSE, calls, blocks: weights.blocks.length, rows,
    ...(gpuPasses === undefined ? {} : { gpuPasses: gpuPasses.slice(0, 24) }),
  };
}
