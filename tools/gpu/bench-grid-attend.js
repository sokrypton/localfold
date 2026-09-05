/**
 * The grid attention's two softmax forms, interleaved in one process.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-grid-attend.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-grid-attend.js --lengths=128,256,512
 *
 * WHY IT EXISTS. `grid.attend` is the largest kernel in an AF3 trunk and grows
 * as tokens CUBED: 18.3% of the trunk's GPU time at 200 tokens and 34.6% at
 * 700, where it is 203.8 ms a pass. It is already vec4 and stages its keys and
 * values in f16, so what was left to move is the arithmetic itself - the
 * one-pass softmax rescales every accumulator on every key, and on the keys
 * that do not raise the running maximum the scale factor is exactly one.
 *
 * 🔴 ONE PROCESS, ALTERNATING, BECAUSE THIS MACHINE DRIFTS UP TO 3.2x BETWEEN
 * THEM. Each length runs both arms `--rounds` times, alternating, and the
 * median of each is reported - so a drift that walks through the run lands on
 * both arms rather than on whichever went second.
 *
 * 🔴 AND EVERY ARM'S OUTPUT IS COMPARED, because a branch that skips work it
 * should have done is a kernel that is wrong in a way a stopwatch reads as a
 * speedup. Not bit for bit, which was the first thing tried and is not true of
 * these two: they are algebraically identical and evaluate in a different
 * order - exp(logit - running_max) against exp(logit - new_max) - so they
 * differ in the last bits. `relRms` between the arms is what says the branch
 * did the same arithmetic; against the CPU reference both measure 9.63e-7.
 *
 * 🔴 THE ANSWER, SO FAR, IS THAT THE BRANCH LOSES:
 *
 *     tokens    lazy    always   speedup
 *        128   11.3 ms   11.6      1.027
 *        256   44.6      43.3      0.971
 *        400  131.1     125.5      0.957
 *
 * The default is therefore the straight-through form. The arm stays because a
 * negative result that nobody can re-run is a rumour.
 */
import { Af3GridSelfAttentionGpu } from "../../src/af3/grid-attention-webgpu.js";

const DIALECT = { swapTransposedBias: false };
const CHANNELS = 128;
const HEADS = 4;
const DIMENSION = 32;

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Deterministic noise, so both arms see the same numbers. */
function deterministic(count, seed) {
  let state = seed >>> 0;
  const out = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[index] = (state / 4294967296) - 0.5;
  }
  return out;
}

/** Synthesised weights: this measures a kernel, not a model. */
function weightsFor(seed) {
  const at = (count, salt) => deterministic(count, seed + salt);
  return {
    heads: HEADS, dimension: DIMENSION,
    actNormScale: new Float32Array(CHANNELS).fill(1),
    actNormOffset: new Float32Array(CHANNELS),
    pairBiasProjection: at(CHANNELS * HEADS, 1),
    qProjection: at(CHANNELS * HEADS * DIMENSION, 2),
    kProjection: at(CHANNELS * HEADS * DIMENSION, 3),
    vProjection: at(CHANNELS * HEADS * DIMENSION, 4),
    gatingQuery: at(CHANNELS * HEADS * DIMENSION, 5),
    outputProjection: at(HEADS * DIMENSION * CHANNELS, 6),
  };
}

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

export async function main(device, args) {
  const lengths = option(args, "lengths", "128,256,400").split(",").map(Number);
  const rounds = Number(option(args, "rounds", "5"));
  // 🔴 THE STAGED PRECISION IS AN ARM, NOT A SETTING, because it is how this
  // kernel's bottleneck is identified. Narrowing the tile halves the BYTES read
  // from workgroup memory and leaves the READ COUNT alone - a lane still reads
  // eight vec4 of k and eight of v for every key - so a kernel bound on bytes
  // gets much faster and one bound on read instructions barely moves. See
  // CLAUDE.md: halving the bytes never halves the reads.
  const staged = device.features.has("shader-f16") ? "f16" : "f32";
  // 🔴 AND THE KEY CHUNK, WHICH IS SIZED IN BYTES AGAINST AN f32 TILE. See
  // attendKeyChunk: the budget is 8 KiB and a vec4 is assumed to be 16 bytes,
  // but with f16 staging the tile element is EIGHT - so the default chunk uses
  // half of what it is allowed and takes twice the barriers.
  const arms = option(args, "chunks", "16,32,64").split(",")
    .map((chunk) => ({ name: `chunk${chunk}`, staged, chunk: Number(chunk), lazy: false }));
  const runner = new Af3GridSelfAttentionGpu(device);

  const rows = [];
  for (const n of lengths) {
    const pair = deterministic(n * n * CHANNELS, 991 + n);
    // Ragged, so the masked path is exercised and the maxima are not uniform.
    const sequence = new Float32Array(n);
    for (let i = 0; i < n; i += 1) sequence[i] = i < Math.ceil(n * 0.75) ? 1 : 0;
    const mask = new Float32Array(n * n);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) mask[i * n + j] = sequence[i] * sequence[j];
    }
    const weights = weightsFor(n);
    const shape = { n, channels: CHANNELS, transpose: false };

    const times = new Map(arms.map((arm) => [arm.name, []]));
    let reference;
    let differ = 0;   // the worst relRms between any two arms
    for (let round = 0; round < rounds; round += 1) {
      for (const arm of arms) {
        const { output, elapsedMilliseconds } = await runner.run(
          pair, mask, shape, weights, DIALECT,
          { stagedPrecision: arm.staged, attendLazyRescale: arm.lazy,
            attendKeyChunk: arm.chunk });
        // ...the first result is the reference and every other is measured
        // against it. See the note above: not equality.
        if (reference === undefined) reference = output;
        else {
          let error = 0;
          let scale = 0;
          for (let i = 0; i < output.length; i += 1) {
            const d = output[i] - reference[i];
            error += d * d;
            scale += reference[i] * reference[i];
          }
          differ = Math.max(differ, Math.sqrt(error / Math.max(scale, 1e-30)));
        }
        times.get(arm.name).push(elapsedMilliseconds);
      }
    }
    const ms = {};
    for (const [name, values] of times) ms[name] = Number(median(values).toFixed(2));
    rows.push({ tokens: n, ms, armsAgreeTo: Number(differ.toExponential(2)) });
  }
  return { rounds, arms: arms.map((a) => a.name), rows };
}
