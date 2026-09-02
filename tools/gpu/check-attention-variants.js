/**
 * Every flash-attention variant against the same input, on this device.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-attention-variants.js
 *
 * 🔴 THE VARIANTS DISAGREEING WOULD BE A CORRECTNESS BUG; THE VARIANTS
 * DIFFERING IN SPEED IS THE POINT. `selectAttentionFlashKernel` picks
 * subgroup-key32 whenever the device has the subgroup features, and on this
 * machine that is 3.6x SLOWER than the register-resident kernel it falls back
 * to otherwise. README.md recorded the subgroup path as unreachable on
 * Chrome-on-Metal, so the comparison had never been run here.
 *
 * This runs one column-attention-shaped problem - transposed, which is the
 * shape AF2's MSA column attention has and the one where the gap is widest -
 * through every variant the device supports, and reports both the disagreement
 * against the default and the time.
 */
import { AttentionGpu, selectAttentionFlashKernel } from "../../src/evoformer/attention.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const VARIANTS = ["portable", "subgroup-4x8", "subgroup-key32", "subgroup-8x64",
                  "subgroup-16x64", "subgroup-32x64", "subgroup-64x64"];

function noise(count, seed) {
  const values = new Float32Array(count);
  let state = seed >>> 0;
  for (let index = 0; index < count; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    values[index] = (state / 4294967296) - 0.5;
  }
  return values;
}

export async function main(device, args) {
  const sequences = Number(option(args, "sequences", "512"));
  const length = Number(option(args, "length", "59"));
  const channels = Number(option(args, "channels", "256"));
  const heads = Number(option(args, "heads", "8"));
  const repeats = Number(option(args, "repeats", "3"));

  // The column-attention shape: residues are the batch, sequences the queries.
  const input = {
    activations: noise(sequences * length * channels, 1),
    mask: new Float32Array(sequences * length).fill(1),
    batch: length, queryLength: sequences, channels, heads, transpose: true,
    weights: {
      queryWeight: noise(channels * channels, 2),
      keyWeight: noise(channels * channels, 3),
      valueWeight: noise(channels * channels, 4),
      gatingWeight: noise(channels * channels, 5),
      gatingBias: noise(channels, 6),
      outputWeight: noise(channels * channels, 7),
      outputBias: noise(channels, 8),
      queryNormScale: new Float32Array(channels).fill(1),
      queryNormOffset: new Float32Array(channels).fill(0),
    },
  };

  const chosen = selectAttentionFlashKernel(device, channels / heads, "auto");
  let reference;
  const rows = [];
  for (const variant of VARIANTS) {
    let result;
    try {
      result = await new AttentionGpu(device, { flashVariant: variant }).run(input);
    } catch (error) {
      rows.push({ variant, note: error.message.slice(0, 60) });
      continue;
    }
    // Timed after a warm run, so pipeline compilation is not in the number.
    const started = performance.now();
    for (let attempt = 0; attempt < repeats; attempt += 1) {
      await new AttentionGpu(device, { flashVariant: variant }).run(input);
    }
    const ms = (performance.now() - started) / repeats;
    if (reference === undefined) reference = result.output;
    let error = 0;
    let scale = 0;
    for (let index = 0; index < reference.length; index += 1) {
      const difference = result.output[index] - reference[index];
      error += difference * difference;
      scale += reference[index] * reference[index];
    }
    rows.push({
      variant, ms: Number(ms.toFixed(1)),
      relRmsVsFirst: Math.sqrt(error / Math.max(scale, 1e-30)).toExponential(2),
    });
  }
  return { sequences, length, channels, heads, autoChooses: chosen.cacheKey, rows };
}
