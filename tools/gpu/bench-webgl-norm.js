/**
 * What a LayerNorm costs in WebGL2, timed in a loop rather than through the
 * fold's per-call profile.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-norm.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-norm.js --shapes=576x768,68x768
 *
 * 🔴 IT EXISTS BECAUSE 38% OF THE DENOISER IS 36 SMALL KERNELS WHOSE COST HAS NO
 * EXPLANATION, AND THIS IS THE LARGEST OF THEM. The fold's profile puts
 * `layer-norm:576x768` at about 1.43 ms a call, and about **0.45 ms** once the
 * profile's own per-call wait is discounted at the 3.18x the whole fold shows -
 * against the **~25 us** `probe-webgl-draw-floor.js` charges for a draw of its
 * 110,592 output texels. An unexplained 18x is either an artefact of measuring
 * through a barrier or a real cost, and only a loop with ONE sync at the end
 * tells them apart. That is the same argument `bench-webgl-linear.js`'s header
 * makes, which is why this is a bench and not another profile reading.
 *
 * 🔴 AND EVERY ITERATION NEEDS ITS OWN OUTPUT, because this device discards the
 * earlier of two identical full-viewport draws into one target before shading
 * it. A ring is the fix; that trap has cost four measurements in this tree.
 *
 * 🔴 AND IT REPORTS THE TRAFFIC A ROW-SWEEPING KERNEL ISSUES BESIDE THE BYTES IT
 * TOUCHES, because the two differ by `CH4` and reading the first as the second
 * is what made "splitting the statistics is 192x less traffic" look like a fix
 * when it is a null - a 3 KiB row is cache-resident, so the sweeps are free.
 */
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { layerNorm } from "../../webgl/kernels/layer-norm.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const shapes = (option(args, "shapes", "576x768,68x768,576x128,4624x256"))
    .split(",").map((spec) => {
      const [rows, channels] = spec.split("x").map(Number);
      return { rows, channels };
    });
  // 🔴 THE VARIANCE FORM MATTERS TO THE SPLIT AND THE FIRST SWEEP MISSED IT.
  // AF3's `fast` takes E[x^2] - E[x]^2 in ONE pass where AF2's `two-pass`
  // centres first, so the fused kernel reads its row `CH4` times rather than
  // `2 * CH4` - and the split therefore saves half as much. The fold's largest
  // excluded norm is `layer-norm:4624x256:fast`, 1240 ms over 624 calls, and
  // benching it two-pass priced a kernel nothing dispatches.
  const variance = option(args, "variance", "two-pass");
  const iterations = Number(option(args, "iterations", "16"));
  const rounds = Number(option(args, "rounds", "4"));
  const gl = createWebGL2Context();
  const execution = new Execution(gl, { checkErrors: false });

  const fill = (count, seed) => {
    const values = new Float32Array(count);
    let state = seed | 0;
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      values[i] = ((state >>> 8) / 0x7fffff - 0.5) * 2;
    }
    return values;
  };

  const rows = [];
  for (const { rows: n, channels } of shapes) {
    const x = execution.upload(fill(n * channels, 7 + channels), "bench.x");
    const weight = execution.upload(fill(channels, 11), "bench.w");
    const bias = execution.upload(fill(channels, 13), "bench.b");
    const ring = Array.from({ length: iterations },
      () => execution.allocate(n * channels, "bench.out"));
    const call = { x, weight, bias, rows: n, channels, variance };
    // 🔴 BOTH ARMS AT THE KERNEL, because the fold measurement that first called
    // the split a null had a 3.7% spread WITHIN its own arm (21848 and 21063 ms)
    // and the difference it was looking for is about 4% of a fold. A bench with
    // one sync measures the kernel; the fold measures the kernel plus 21 s of
    // other work.
    for (const split of [false, true]) {
      execution.noNormSplit = !split;
    for (let i = 0; i < 2; i += 1) {
      layerNorm(execution, call, ring[i % ring.length]);
      execution.read(ring[i % ring.length]);
    }
    const samples = [];
    for (let round = 0; round < rounds; round += 1) {
      const start = performance.now();
      for (let i = 0; i < iterations; i += 1) layerNorm(execution, call, ring[i]);
      execution.read(ring[iterations - 1]);
      samples.push((performance.now() - start) / iterations);
    }
    const ms = Math.min(...samples);
    const outputTexels = (n * channels) / 4;
    // What it TOUCHES: x once, the output once. What it ISSUES: every fragment
    // sweeps its row, so `CH4` times x for the mean and again for the spread.
    const touchedBytes = n * channels * 4 * 2;
    const issuedBytes = n * channels * 4 * (2 * (channels / 4)) + n * channels * 4;
    rows.push({
      shape: `${n}x${channels}`,
      variance,
      arm: split ? "split" : "fused",
      outputTexels,
      microsecondsPerCall: Number((ms * 1000).toFixed(1)),
      spread: Number((Math.max(...samples) / Math.min(...samples)).toFixed(2)),
      touchedGigabytesPerSecond: Number((touchedBytes / ms / 1e6).toFixed(1)),
      issuedGigabytesPerSecond: Number((issuedBytes / ms / 1e6).toFixed(1)),
      rowKibibytes: Number((channels * 4 / 1024).toFixed(2)),
    });
    }
    execution.noNormSplit = false;
    for (const out of ring) execution.release(out);
    execution.release(x);
    execution.release(weight);
    execution.release(bias);
  }

  const headline = rows.find((r) => r.shape === "576x768" && r.arm === "fused") ?? rows[0];
  const split = rows.find((r) => r.shape === "576x768" && r.arm === "split");
  return {
    renderer: gl.getParameter(gl.RENDERER),
    iterations,
    rows,
    // The fold's own figure for the same shape, for the comparison this exists for.
    foldProfileMicrosecondsPerCall: 1430,
    foldProfileInflation: 3.18,
    verdict: `${headline.shape} is ${headline.microsecondsPerCall} us a call in a loop, against `
      + `${(1430 / 3.18).toFixed(0)} us inferred from the fold's profile and ~25 us for a draw of `
      + `its size - so the profile's per-call wait was `
      + `${(1430 / headline.microsecondsPerCall).toFixed(1)}x the kernel`
      + (split === undefined ? "" : `; splitting its statistics is `
        + `${(headline.microsecondsPerCall / split.microsecondsPerCall).toFixed(2)}x `
        + `(${split.microsecondsPerCall} us)`),
  };
}
