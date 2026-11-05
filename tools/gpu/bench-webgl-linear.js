/**
 * What the WebGL2 dense projection costs, at the shapes the models fold at.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-linear.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-linear.js --iterations=8
 *
 * 🔴 THIS EXISTS BECAUSE THE PER-KERNEL PROFILE COULD NOT ANSWER IT. WebGL2 has
 * no timestamp query, so `Execution.profile` times a kernel by WAITING for it -
 * and at an AF2 monomer fold's 5,187 draws the wait is most of what it measures.
 * A kernel timed in a loop with ONE sync at the end is the honest instrument,
 * and it is the one that says whether a geometry is worth its pass.
 *
 * 🔴 AND EVERY ITERATION NEEDS ITS OWN OUTPUT TEXTURE, WHICH TOOK TWO WRONG
 * MEASUREMENTS TO LEARN. An Apple GPU is tile-based and does hidden-surface
 * removal: issue the same full-viewport triangle twice into the same target with
 * no read between, and the earlier draw's fragments are DISCARDED before they
 * are shaded. So a loop that writes one output over and over times ONE
 * iteration and divides by the count - it read `linear.block:576x896x1792` at
 * 11.25 ms a call where a fold's own profile says 87, a factor of eight, which
 * is the iteration count. The same trap voids the draw-DOUBLING instrument
 * (`Execution.doubleLabels`), which reports the GPU as 2% of a bioemu2 fold
 * that `--skip-draws` puts at 97%. A ring of distinct outputs is the fix: no
 * target is written twice, so nothing can be culled.
 *
 * 🔴 AND EVERY ARM IS COMPARED AGAINST THE SHIPPED ONE, because a faster kernel
 * that computes something else is not a result. The bar is a REORDERING: the
 * blocked path's accumulation is the same sum in the same order, but this
 * compiler fuses `acc += vec4(dot(x, w0), ...)` and `sum.x += dot(x, w0)`
 * differently, so the difference is a fused multiply-add's worth, not zero.
 */
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { linear } from "../../webgl/kernels/linear.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * The shapes a fold actually dispatches, read off the labels an AF2 monomer
 * fold at 59 residues and a bioemu2 fold at 68 tokens produce.
 *
 * 🔴 THE ROW COUNT IS THE PAIR TRACK, NOT THE SEQUENCE. `3481` is 59 x 59:
 * nearly every projection in a trunk runs over token PAIRS. And the two groups
 * below behave completely differently, which is the finding - a pair
 * projection's WEIGHT is 128 x 128, 64 KiB, and stays in cache across all 3,481
 * of its rows, while bioemu2's atom and token projections carry 1.6 to 3.5
 * million floats, which fit nowhere, so every row streams them again.
 */
const SHAPES = [
  { name: "af2 pair 128->128 out-major", rows: 3481, inChannels: 128, outChannels: 128, weightLayout: "out-major" },
  { name: "af2 pair 128->128 in-major", rows: 3481, inChannels: 128, outChannels: 128, weightLayout: "in-major" },
  { name: "af2 pair transition 128->512", rows: 3481, inChannels: 128, outChannels: 512, weightLayout: "in-major", activation: "relu" },
  { name: "af2 pair transition 512->128", rows: 3481, inChannels: 512, outChannels: 128, weightLayout: "in-major" },
  { name: "af2 msa 256->256", rows: 59, inChannels: 256, outChannels: 256, weightLayout: "in-major" },
  { name: "bioemu2 pair 4624 128->128", rows: 4624, inChannels: 128, outChannels: 128, weightLayout: "out-major" },
  { name: "bioemu2 single 68 384->384", rows: 68, inChannels: 384, outChannels: 384, weightLayout: "out-major" },
  { name: "esmfold2 pair 5625 128->128", rows: 5625, inChannels: 128, outChannels: 128, weightLayout: "out-major" },
  { name: "bioemu2 atom 576 896->1792", rows: 576, inChannels: 896, outChannels: 1792, weightLayout: "out-major" },
  { name: "bioemu2 atom 576 768->1920", rows: 576, inChannels: 768, outChannels: 1920, weightLayout: "out-major" },
  { name: "bioemu2 token 68 768->4608", rows: 68, inChannels: 768, outChannels: 4608, weightLayout: "out-major" },
  { name: "bioemu2 token 68 3072->768", rows: 68, inChannels: 3072, outChannels: 768, weightLayout: "out-major" },
];

/** A reproducible spread of magnitudes - a real weight matrix is not uniform. */
const fill = (count, seed) => {
  let state = seed >>> 0;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    out[i] = ((state / 0x1_0000_0000) * 2 - 1) * 0.5;
  }
  return out;
};

export async function main(device, args) {
  const iterations = Number(option(args, "iterations", "8"));
  // 🔴 THE BALLAST ARM: a bioemu2 fold holds about 2 GB of textures against an
  // AF2 fold's 466 MB, so "is a dispatch paying for what is RESIDENT" had to be
  // asked. Measured at 0, 512 and 1536 MiB of dead textures: no systematic
  // difference, so residency is not the answer and the weight traffic is.
  const ballastMegabytes = Number(option(args, "ballast", "0"));
  const gl = createWebGL2Context();
  // No `getError` after each draw: it is a synchronous round trip, worth 4% of
  // an AF2 fold, and a benchmark that pays it measures the round trip.
  const execution = new Execution(gl, { checkErrors: false });
  const rows = [];
  const ballast = [];
  for (let held = 0; held < ballastMegabytes; held += 16) {
    ballast.push(execution.allocate(4 << 20, "bench.ballast"));
  }

  // 🔴 `--shapes=RxIxO,...` REPLACES THE TABLE, which is how the WORKING SET is
  // swept at constant arithmetic: `4624x256x512`, `9248x256x256` and
  // `18496x256x128` all do the same 1.21 GFLOP with a weight of 512, 256 and
  // 128 KiB. If the rate rises as the weight shrinks, these kernels are bound
  // by what they MISS rather than by the 21.33 flops a fetch the output budget
  // allows - see webgl/README.md's retraction of the old floor.
  const chosen = option(args, "shapes", "");
  const shapes = chosen === "" ? SHAPES : chosen.split(",").map((text) => {
    const [rows, inChannels, outChannels] = text.split("x").map(Number);
    return {
      name: text, rows, inChannels, outChannels,
      weightLayout: option(args, "layout", "out-major"),
    };
  });
  for (const shape of shapes) {
    const { rows: n, inChannels, outChannels, weightLayout, activation = "none" } = shape;
    const x = execution.upload(fill(n * inChannels, 7 + n), "bench.x");
    const weight = execution.upload(fill(inChannels * outChannels, 11 + inChannels), "bench.w");
    const bias = execution.upload(fill(outChannels, 13), "bench.b");
    const call = {
      x, weight, bias, rows: n, inChannels, outChannels, activation, weightLayout,
    };
    const flops = 2 * n * inChannels * outChannels;
    /** @type {Record<string, Float32Array>} */
    const answers = {};
    const timings = {};
    for (const arm of ["one-texel", "blocked"]) {
      execution.noLinearBlock = arm === "one-texel";
      // A ring of outputs, one per iteration: see the header - a target written
      // twice lets the GPU discard the earlier write and the timing is a lie.
      const ring = Array.from({ length: iterations },
        () => execution.allocate(n * outChannels, "bench.out"));
      // Warm: the program compiles on its first dispatch, and a cold Apple GPU
      // runs at a low clock - a first-iteration timing is neither.
      for (let i = 0; i < 2; i += 1) {
        linear(execution, call, ring[i % ring.length]);
        execution.read(ring[i % ring.length]);
      }
      const start = performance.now();
      for (let i = 0; i < iterations; i += 1) linear(execution, call, ring[i]);
      // One sync for the whole loop, which is the point: a readback per
      // iteration measures the readback.
      answers[arm] = Float32Array.from(execution.read(ring[iterations - 1]));
      timings[arm] = (performance.now() - start) / iterations;
      for (const out of ring) execution.release(out);
    }
    execution.noLinearBlock = false;
    execution.release(x);
    execution.release(weight);
    execution.release(bias);
    rows.push({
      shape: shape.name,
      weightFloats: inChannels * outChannels,
      oneTexelMs: Number(timings["one-texel"].toFixed(3)),
      blockedMs: Number(timings.blocked.toFixed(3)),
      speedup: Number((timings["one-texel"] / timings.blocked).toFixed(2)),
      oneTexelGflops: Number((flops / timings["one-texel"] / 1e6).toFixed(1)),
      blockedGflops: Number((flops / timings.blocked / 1e6).toFixed(1)),
      relRms: Number(relativeRms(answers["one-texel"], answers.blocked).toExponential(2)),
    });
  }

  for (const dead of ballast) execution.release(dead);
  const worst = Math.max(...rows.map((row) => row.relRms));
  const slowest = Math.min(...rows.map((row) => row.speedup));
  // A fused multiply-add's worth, which is what two expression shapes of one
  // sum differ by on this compiler. Never widen it to pass: if a shape reads
  // 1e-3 the two kernels are computing different sums.
  const bar = 1e-6;
  execution.destroy();
  return {
    renderer: gl.getParameter(gl.RENDERER),
    iterations,
    ballastMegabytes: ballast.length * 16,
    rows,
    worstRelRms: Number(worst.toExponential(2)),
    bar,
    slowestSpeedup: slowest,
    fastestSpeedup: Math.max(...rows.map((row) => row.speedup)),
    verdict: worst > bar
      ? `FAILED: the blocked projection disagrees with the one-texel one at ${worst.toExponential(2)}, `
        + `past ${bar} - that is not a reordering`
      : `ok: the blocked projection agrees to ${worst.toExponential(2)} and is `
        + `${slowest.toFixed(2)}x to ${Math.max(...rows.map((r) => r.speedup)).toFixed(2)}x the one-texel path`,
  };
}
