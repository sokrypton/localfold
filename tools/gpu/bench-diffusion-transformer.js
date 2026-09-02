/**
 * The 24-block token transformer alone, with no checkpoint behind it.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-diffusion-transformer.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-diffusion-transformer.js --tokens=120
 *
 * 🔴 SYNTHETIC WEIGHTS, BECAUSE THIS MEASURES THE LOOP AND NOT THE MODEL. The
 * cost of a block depends on its SHAPES, so weights of the right lengths full
 * of nothing in particular cost exactly what the real ones do - and skipping
 * the checkpoint takes the edit-measure cycle from minutes to a couple of
 * seconds, which is the difference between measuring every change and guessing
 * at most of them. bench-head.js is the one that runs all four stages.
 */
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion-transformer-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const SHAPE = {
  channels: 768, condChannels: 384, pairChannels: 128,
  heads: 16, dimension: 48, transitionFactor: 2, blocksPerSuperBlock: 4,
};

/** Every leaf of one block, at the length the packer and shader expect. */
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
  const tokens = Number(option(args, "tokens", "59"));
  const calls = Number(option(args, "calls", "4"));
  const superBlocks = Number(option(args, "superblocks", "6"));
  const { channels, condChannels, pairChannels, blocksPerSuperBlock } = SHAPE;

  const block = blockWeights(SHAPE);
  const weights = {
    ...SHAPE,
    lanes: args.some((a) => a.startsWith("--lanes=")) ? Number(option(args, "lanes", "")) : undefined,
    pairInputLayerNormScale: new Float32Array(pairChannels).fill(1),
    superBlocks: Array.from({ length: superBlocks }, () => ({
      // 🔴 THE SAME OBJECT IN EVERY SLOT, WHICH IS THE POINT ON THE WEIGHT
      // CACHE. A WeakMap keyed on the block object would look free here if
      // every slot shared one - so the blocks are distinct copies below.
      pairLogitsProjection: new Float32Array(pairChannels * blocksPerSuperBlock
        * SHAPE.heads).fill(0.01),
      blocks: Array.from({ length: blocksPerSuperBlock }, () => ({ ...block })),
    })),
  };

  const act = new Float32Array(tokens * channels).fill(0.1);
  const cond = new Float32Array(tokens * condChannels).fill(0.1);
  const pair = new Float32Array(tokens * tokens * pairChannels).fill(0.01);
  const seqMask = new Float32Array(tokens).fill(1);

  const rows = [];
  for (let call = 0; call < calls; call += 1) {
    const started = performance.now();
    await new Af3DiffusionTransformerGpu(device)
      .run(Float32Array.from(act), cond, pair, seqMask, tokens, weights);
    rows.push(Math.round(performance.now() - started));
  }
  const after = rows.slice(1);
  return {
    tokens, lanes: weights.lanes ?? "default",
    blocks: superBlocks * blocksPerSuperBlock, callsMs: rows,
    steadyMs: Math.round(after.reduce((a, b) => a + b, 0) / after.length),
  };
}
