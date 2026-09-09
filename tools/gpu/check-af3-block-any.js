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
import { deviceTuning, setDeviceTuning } from "../../src/runtime/device-profile.js";

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
  // The pair track's weight element; see bench-trunk.js for why it is its own
  // knob and why the measurement behind its default is stale.
  const pairWeightPrecision = option(args, "pair-weights", undefined);
  // 🔴 `--tune=key=value`, BECAUSE A KNOB NO GATE ENTERS IS A KNOB NOBODY HAS
  // CHECKED. Several of this block's kernels are chosen by the device profile -
  // `gridAttendMatrix` among them - and without this flag the only arm ever
  // checked here is whatever THIS device happens to pick.
  for (const pair of args.filter((a) => a.startsWith("--tune="))
       .flatMap((a) => a.slice("--tune=".length).split(",")).filter(Boolean)) {
    const at = pair.indexOf("=");
    if (at < 0) throw new Error(`--tune wants key=value, got ${pair}`);
    const raw = pair.slice(at + 1);
    let value;
    try { value = JSON.parse(raw); } catch { value = raw; }
    setDeviceTuning(device, { [pair.slice(0, at)]: value });
  }
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

  const accumulate = option(args, "accumulate", undefined);
  // 🔴 `--resident` BECAUSE RESIDENCY IS A DIFFERENT WEIGHT PATH, NOT A CACHE.
  // With it the pair transition is decoded from its int5 codes on the DEVICE
  // and the host packer never builds it; without it every tensor is packed on
  // the host. Two paths, and only one of them was ever checked here.
  const stack = new Af3PairformerStackGpu(device, {
    residentWeights: args.includes("--resident"),
    stagedPrecision: option(args, "staged", undefined),
    accumulatePrecision: accumulate,
    weightPrecision: option(args, "weights", undefined),
    pairWeightPrecision,
  });
  const actual = await stack.run(state, blocks, dialect, {});

  const pair = relativeRms(actual.pair, expected.pair);
  const single = relativeRms(actual.single, expected.single);
  // 🔴 AND IT FAILS NOW, WHICH IT DID NOT. This printed two relRMS figures and
  // asserted on neither, so every arm it has ever been run with "passed".
  //
  // 🔴 THE BOUND FOLLOWS THE ARM, BECAUSE ONE BOUND WOULD STOP CHECKING THE
  // TIGHT PATH. Measured on /model-af3-int5 at 40 tokens, two blocks, as the
  // pair relRMS against the CPU reference:
  //
  //     staged f32, accumulate f32, scalar attend       1.9e-5
  //     ...with the matrix attend                       4.4e-3
  //     ...with the matrix TRIANGLE PROJECTION           2.0e-2
  //     ...accumulate f16 (the default where f16 is)     8.3e-2
  //
  // The f16 triangle ACCUMULATOR is the loosest and it is a recorded trade, not
  // a defect - see src/af3/pair-track-gpu.js. A bound wide enough for it is
  // 0.15 and still catches a kernel that is actually wrong, because a wrong
  // kernel here is order one. `--bound=` overrides.
  //
  // 🔴 THE MATRIX TRIANGLE PROJECTION IS THE WIDEST OF THE THREE KERNEL ARMS,
  // and it is the one the folds had to justify rather than this number: the
  // units multiply BOTH operands in f16 where the vector kernel's f16 mode
  // narrows only the activation, so on uniform noise it costs 7.5x the shipped
  // f16 path. On three real folds it costs nothing measurable. See
  // TRIANGLE_PROJECT_MATRIX_MIN_CHANNELS, which is why an AF3 bundle does not
  // take it at all.
  const looseAccumulate = (accumulate ?? (device.features.has("shader-f16") ? "f16" : "f32"))
    === "f16";
  const tuned = deviceTuning(device);
  const wide = pairChannels >= 192;
  const kernelBound = (tuned.triangleProjectMatrix === true && wide) ? 4e-2
    : (tuned.gridAttendMatrix === true ? 2e-2 : 1e-4);
  const bound = Number(option(args, "bound",
    String(looseAccumulate ? 0.15 : kernelBound)));
  if (!(pair <= bound) || !(single <= bound)) {
    throw new Error(`pairformer block outside ${bound.toExponential(0)}: `
      + `pair ${pair.toExponential(2)}, single ${single.toExponential(2)}`);
  }

  return {
    model: manifest,
    n, blocks: count, bound, resident: args.includes("--resident"),
    widths: { pairChannels, singleChannels,
              gridHeads: blocks[0].pairAttention1.heads,
              gridDimension: blocks[0].pairAttention1.dimension,
              singleHeads: blocks[0].singleAttention.heads },
    dialect: Object.fromEntries(Object.entries(dialect).filter(([, v]) => v)),
    pair, single,
  };
}
