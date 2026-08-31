/**
 * Time triangle multiplication at the sizes AF3 actually has to run.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-triangle.js --lengths=128,300,600
 *
 * This is tools/benchmark-triangle.js moved onto the Chrome lane, because the
 * Dawn one does not load on this OS - see tools/gpu-chrome.mjs.
 *
 * WHY THIS IS THE FIRST MEASUREMENT OF THE AF3 PORT. Triangle multiplication is
 * the only O(N^3) operation in the pairformer, so it sets both the run time and
 * the peak memory of a 48-block trunk. AF3 runs it at the same shape AF2 does
 * (c_z = 128, c_hidden = 128), differing only in that the a and b projections
 * are fused into one weight - which changes no FLOP and no allocation. So this
 * kernel answers the AF3 question before any AF3 code exists.
 */
import { createDeterministicTriangleInput } from "../../src/testing/deterministic-input.js";
import { TriangleMultiplicationOutgoingGpu } from "../../src/triangle/webgpu.js";

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

export async function main(device, args) {
  const lengths = option(args, "lengths", "128,300,600").split(",").map(Number);
  const cZ = Number(option(args, "cz", "128"));
  const cHidden = Number(option(args, "hidden", "128"));
  const precisions = option(args, "precision", "f32,f16").split(",");
  const repeats = Number(option(args, "repeats", "3"));
  if (!lengths.every((length) => Number.isSafeInteger(length) && length > 0)) {
    throw new Error("--lengths must be a comma-separated list of positive integers");
  }

  const rows = [];
  for (const precision of precisions) {
    if (precision === "f16" && !device.features.has("shader-f16")) continue;
    for (const length of lengths) {
      // 🔴 A FRESH RUNNER PER ROW, because the allocator's peak is a high-water
      // mark that never resets. Sharing one runner makes every row after the
      // largest report the LARGEST row's memory - which silently reported f16
      // as costing exactly what f32 does, hiding the halving that is the whole
      // reason to consider f16.
      const runner = new TriangleMultiplicationOutgoingGpu(device);
      const input = createDeterministicTriangleInput({ length, cZ, cHidden }, 1000 + length);
      // ...one warm-up so shader compilation is not counted as run time, then
      // the median of `repeats`, because a single GPU timing is worthless -
      // run-to-run drift on this machine is several-fold.
      await runner.run(input, { precision });
      const times = [];
      let peak = 0;
      for (let attempt = 0; attempt < repeats; attempt += 1) {
        const result = await runner.run(input, { precision });
        times.push(result.elapsedMilliseconds);
        peak = Math.max(peak, result.memory.peakBytes);
      }
      times.sort((a, b) => a - b);
      const median = times[Math.floor(times.length / 2)];
      // 2 x MACs: the two projections, the N^3 contraction, and the output.
      const flops = 2 * (length * length * cZ * cHidden * 4 + length ** 3 * cHidden
        + length * length * cHidden * cZ + length * length * cZ * cZ);
      rows.push({
        precision, length,
        medianMs: Number(median.toFixed(3)),
        spreadMs: Number((times[times.length - 1] - times[0]).toFixed(3)),
        peakMiB: Number((peak / 2 ** 20).toFixed(2)),
        effectiveTFLOPS: Number((flops / (median / 1000) / 1e12).toFixed(2)),
      });
      console.log(`${precision}\tL=${length}\t${median.toFixed(1)} ms`
        + `\t${(peak / 2 ** 20).toFixed(1)} MiB\t${(flops / (median / 1000) / 1e12).toFixed(2)} TFLOP/s`);
    }
  }
  return { cZ, cHidden, repeats, rows };
}
