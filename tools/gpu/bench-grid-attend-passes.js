/**
 * `grid.attend` alone, in GPU time, separated from the bus.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-grid-attend-passes.js --lengths=128,256,400
 *
 * WHY IT EXISTS. bench-grid-attend.js times `Af3GridSelfAttentionGpu.run`,
 * which is an upload of the whole pair representation, four projections, the
 * attention, a projection out, and a readback - and it reports one number.
 * On unified memory the two copies are nearly free and that number is a fair
 * proxy for the kernel. On a discrete GPU they are a PCIe round trip: at 400
 * tokens the pair representation is 82 MB in and 82 MB out, which is tens of
 * milliseconds that belong to no kernel. So a figure recorded on an M2 and one
 * recorded on an A100 are not measuring the same thing, and the M2's number is
 * the one this repository's tuning is quoted against.
 *
 * This wraps the same call in `profileDevice`, which timestamps every labelled
 * compute pass, so `attend` is reported on its own and the copies fall out.
 */
import { Af3GridSelfAttentionGpu } from "../../src/af3/grid-attention-webgpu.js";
import { profileDevice } from "./profile.js";

const DIALECT = { swapTransposedBias: false };
const CHANNELS = 128;
const HEADS = 4;
const DIMENSION = 32;

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function deterministic(count, seed) {
  let state = seed >>> 0;
  const out = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[index] = (state / 4294967296) - 0.5;
  }
  return out;
}

/** Synthesised weights: this measures a kernel, not a model. Kept identical
 * to bench-grid-attend.js's, so the two tools time the same shader. */
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

const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];

export async function main(device, args) {
  const lengths = option(args, "lengths", "128,256,400").split(",").map(Number);
  const rounds = Number(option(args, "rounds", "5"));
  const chunk = Number(option(args, "chunk", "32"));
  const staged = device.features.has("shader-f16") ? "f16" : "f32";
  const runner = new Af3GridSelfAttentionGpu(device);

  const rows = [];
  for (const n of lengths) {
    const pair = deterministic(n * n * CHANNELS, 991 + n);
    const sequence = new Float32Array(n);
    for (let i = 0; i < n; i += 1) sequence[i] = i < Math.ceil(n * 0.75) ? 1 : 0;
    const mask = new Float32Array(n * n);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) mask[i * n + j] = sequence[i] * sequence[j];
    }
    const weights = weightsFor(n);
    const shape = { n, channels: CHANNELS, transpose: false };
    const options = { stagedPrecision: staged, attendLazyRescale: false,
                      attendKeyChunk: chunk };

    // Warm the pipelines before anything is timed.
    await runner.run(pair, mask, shape, weights, DIALECT, options);

    const wall = [];
    const byLabel = new Map();
    for (let round = 0; round < rounds; round += 1) {
      const profile = profileDevice(device);
      const { elapsedMilliseconds } = await runner.run(
        pair, mask, shape, weights, DIALECT, options);
      wall.push(elapsedMilliseconds);
      const report = await profile.report();
      profile.restore();
      for (const entry of report) {
        const label = entry.label ?? "(unlabelled)";
        if (!byLabel.has(label)) byLabel.set(label, []);
        byLabel.get(label).push(entry.ms ?? entry.milliseconds ?? 0);
      }
    }
    const passes = {};
    let gpuTotal = 0;
    for (const [label, values] of byLabel) {
      const ms = median(values);
      passes[label] = Number(ms.toFixed(3));
      gpuTotal += ms;
    }
    const wallMs = median(wall);
    rows.push({
      tokens: n,
      wallMs: Number(wallMs.toFixed(2)),
      gpuMs: Number(gpuTotal.toFixed(2)),
      offDeviceMs: Number((wallMs - gpuTotal).toFixed(2)),
      pairMiB: Number((n * n * CHANNELS * 4 / 1048576).toFixed(1)),
      passes,
    });
  }
  return { rounds, chunk, staged, rows };
}
