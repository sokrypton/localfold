/**
 * A pairformer block on the GPU against its CPU reference, for ANY bundle.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-block-any.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-block-any.js \
 *       --model=/model-opendde-trunk-f32/manifest.json --n=24
 *
 * 🔴 check-af3-block.js BUILDS ITS WEIGHT DICT BY HAND, with `heads: 4`,
 * `dimension: 32`, `pairChannels: 128` and `singleChannels: 384` typed into it.
 * That is correct for AlphaFold 3 and unusable for any other bundle - and it is
 * the shape of mistake this repository already records costing months on the
 * side chains: a checker that constructs its own inputs tests whatever it
 * constructed. This one goes through `pairformerBlockWeights`, the same loader
 * the fold uses, so the widths are the bundle's and a bundle whose widths were
 * read wrongly fails HERE rather than in a contact map.
 *
 * It is differential, not oracle: it says the GPU computes what the reference
 * computes, not that OpenDDE agrees.
 */
import { pairformerBlock } from "../../src/af3/pairformer-reference.js";
import { Af3PairformerStackGpu } from "../../src/af3/pairformer-block-webgpu.js";
import { af3Dialect, openAf3Store, pairformerBlockWeights } from "../../src/af3/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function deterministic(length, seed) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = (((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1;
  }
  return output;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const n = Number(option(args, "n", "24"));
  const count = Number(option(args, "blocks", "2"));
  const manifest = option(args, "model", "/model-af3-full-f32/manifest.json");
  const store = await openAf3Store(manifest);
  const dialect = af3Dialect(store);

  const blocks = [];
  for (let index = 0; index < count; index += 1) {
    blocks.push(await pairformerBlockWeights(store, index));
  }
  const pairChannels = blocks[0].pairChannels;
  const singleChannels = blocks[0].singleChannels ?? 384;

  const pairs = n * n;
  const state = {
    tokens: n,
    pair: deterministic(pairs * pairChannels, 101),
    single: deterministic(n * singleChannels, 202),
    pairMask: new Float32Array(pairs).fill(1),
    seqMask: new Float32Array(n).fill(1),
  };

  let expected = { pair: state.pair, single: state.single };
  for (let index = 0; index < count; index += 1) {
    expected = pairformerBlock(
      { ...state, pair: expected.pair, single: expected.single }, blocks[index], dialect);
  }

  const stack = new Af3PairformerStackGpu(device, {
    residentWeights: false,
    stagedPrecision: option(args, "staged", undefined),
    accumulatePrecision: option(args, "accumulate", undefined),
    weightPrecision: option(args, "weights", undefined),
  });
  const actual = await stack.run(state, blocks, dialect, {});

  return {
    model: manifest,
    n, blocks: count,
    widths: { pairChannels, singleChannels,
              gridHeads: blocks[0].pairAttention1.heads,
              gridDimension: blocks[0].pairAttention1.dimension,
              singleHeads: blocks[0].singleAttention.heads },
    dialect: Object.fromEntries(Object.entries(dialect).filter(([, v]) => v)),
    pair: relativeRms(actual.pair, expected.pair),
    single: relativeRms(actual.single, expected.single),
  };
}
