/**
 * diffusionNormSplit, on and off, ALTERNATING IN ONE PROCESS.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-normsplit.js --tokens=68
 *
 * 🔴 ACROSS LAUNCHES THIS KNOB CANNOT BE MEASURED ON A COLAB BOX, WHICH IS
 * WHAT THIS FILE IS FOR. Six counterbalanced rounds of the ordinary bench, at
 * 68 tokens, put an IDENTICAL-CODE control 8.3% from its own baseline on the
 * minimum and 17.6% on the median - larger than the effect being looked for,
 * and the two statistics disagreeing on its size. tools/gpu/bench-ab.js
 * carries the same rule in its own header and it is the older lesson: two
 * numbers from two invocations are not comparable here.
 *
 * `normSplit` is read off the WEIGHTS object before the device tuning
 * (`weights.normSplit ?? deviceTuning(device).diffusionNormSplit`), so both
 * arms are reachable without touching the device profile between them.
 *
 * WHAT IT ANSWERED, on a Colab T4, eleven to fifteen interleaved rounds:
 *
 *   | tokens | off min/median | on min/median | delta |
 *   |    68  |  19.4 / 20.6   | 19.0 / 20.5   | -0.5% / +0.5% |
 *   |   256  |  99.5 / 100.6  | 99.8 / 101.5  | +0.9% / +0.3% |
 *   |   512  | 301.5 / 303.3  | 302.8 / 304.5 | +0.4% / +0.4% |
 *
 * Nothing, at any size, with the two statistics agreeing - which is what a
 * null looks like when the instrument works. On an A100 the same knob is
 * worth 526 ms of a sampler.
 */
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion/diffusion-transformer-webgpu.js";
import { deviceProfile } from "../../src/runtime/device-profile.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const SHAPE = {
  channels: 768, condChannels: 384, pairChannels: 128,
  heads: 16, dimension: 48, transitionFactor: 2, blocksPerSuperBlock: 4,
};

function blockWeights({ channels, condChannels, heads, dimension, transitionFactor }) {
  const width = heads * dimension;
  const intermediate = channels * transitionFactor;
  const of = (length) => new Float32Array(length).fill(0.01);
  const conditioned = (prefix, gated) => ({
    [`${prefix}SingleCondLayerNormScale`]: of(condChannels),
    [`${prefix}SingleCondScaleWeights`]: of(condChannels * channels),
    [`${prefix}SingleCondScaleBias`]: of(channels),
    [`${prefix}SingleCondBias`]: of(condChannels * channels),
    [`${prefix}AdaptiveZeroCondWeights`]: of(condChannels * gated),
    [`${prefix}AdaptiveZeroCondBias`]: of(gated),
  });
  return {
    ...conditioned("", channels),
    qProjection: of(channels * width), qBias: of(width),
    kProjection: of(channels * width), vProjection: of(channels * width),
    gatingQuery: of(channels * width),
    Transition2: of(width * channels),
    ...conditioned("ffw", channels),
    ffwTransition1: of(channels * intermediate * 2),
    ffwTransition2: of(intermediate * channels),
  };
}

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "68"));
  const rounds = Number(option(args, "rounds", "9"));
  const { channels, condChannels, pairChannels, blocksPerSuperBlock } = SHAPE;
  const superBlocks = Number(option(args, "superblocks", "6"));
  const block = blockWeights(SHAPE);
  const base = {
    ...SHAPE,
    pairInputLayerNormScale: new Float32Array(pairChannels).fill(1),
    superBlocks: Array.from({ length: superBlocks }, () => ({
      pairLogitsProjection: new Float32Array(pairChannels * blocksPerSuperBlock
        * SHAPE.heads).fill(0.01),
      blocks: Array.from({ length: blocksPerSuperBlock }, () => ({ ...block })),
    })),
  };
  const act = new Float32Array(tokens * channels).fill(0.1);
  const cond = new Float32Array(tokens * condChannels).fill(0.1);
  const pair = new Float32Array(tokens * tokens * pairChannels).fill(0.01);
  const seqMask = new Float32Array(tokens).fill(1);

  // 🔴 ONE WEIGHTS OBJECT PER ARM, BUILT ONCE. Spreading `{...base, normSplit}`
  // per call makes a NEW object every time, and the packed weights are cached
  // against the object - so the first version of this measured the upload on
  // every call: 368 ms where the ordinary bench reports 40 at the same size,
  // and a compute-side knob cannot be seen underneath that. Two stable
  // objects, one per arm, and the caches warm exactly as they do in a fold.
  const arms = { false: { ...base, normSplit: false }, true: { ...base, normSplit: true } };
  const runner = new Af3DiffusionTransformerGpu(device);
  const once = async (normSplit) => {
    const started = performance.now();
    await runner.run(Float32Array.from(act), cond, pair, seqMask, tokens,
                     arms[String(normSplit)]);
    return performance.now() - started;
  };
  // 🔴 EIGHT DISCARDED CALLS, NOT TWO. Compiling both pipelines is not the
  // whole of the ramp: with two warm-up calls the first four timed rounds
  // read 50.3 / 37.0 / 31.2 / 26.1 ms and the rest 20, so a median over nine
  // rounds is mostly the ramp and moved the answer by six points.
  const warmup = Number(option(args, "warmup", "8"));
  for (let i = 0; i < warmup; i += 1) await once(i % 2 === 0);

  const off = []; const on = [];
  for (let round = 0; round < rounds; round += 1) {
    // Order swapped every round, so neither arm always follows the other.
    if (round % 2 === 0) { off.push(await once(false)); on.push(await once(true)); }
    else { on.push(await once(true)); off.push(await once(false)); }
  }
  const median = (v) => [...v].sort((x, y) => x - y)[Math.floor(v.length / 2)];
  const min = (v) => Math.min(...v);
  const r1 = (v) => Number(v.toFixed(1));
  return {
    adapter: `${deviceProfile(device).vendor} / ${deviceProfile(device).architecture}`,
    tokens, rounds,
    offMs: { min: r1(min(off)), median: r1(median(off)), all: off.map(r1) },
    onMs: { min: r1(min(on)), median: r1(median(on)), all: on.map(r1) },
    deltaPctMedian: r1((median(on) - median(off)) / median(off) * 100),
    deltaPctMin: r1((min(on) - min(off)) / min(off) * 100),
  };
}
