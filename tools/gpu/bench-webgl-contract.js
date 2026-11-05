/**
 * What AF3's triangle CONTRACTION costs in WebGL2, and whether it is bound by
 * fetches, by arithmetic or by neither.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-contract.js
 *     node tools/gpu-chrome.mjs tools/gpu/bench-webgl-contract.js --iterations=8 --channels=256
 *
 * 🔴 THIS EXISTS BECAUSE IT IS THE THIRD-LARGEST KERNEL IN A bioemu2 FOLD AND
 * HAD NO BENCH. Corrected for the per-call barrier the profile pays, the fold's
 * own numbers put `af3.triangle.contract-block` at 1591 ms of 7074 - 22.5%, in
 * only 64 calls - at 68 tokens and 256 channels, where the same device's dense
 * projections reach 70 to 113 GFLOP/s. Iterating on that through a 6-second
 * fold is not iterating.
 *
 * 🔴 AND IT REPORTS FETCHES A SECOND BESIDE FLOPS A SECOND, because those are
 * the two ceilings and the fix differs. The blocked kernel reads six `fetch4`
 * per k-step for eight dot products - 10.7 flops a fetch where the eight-row
 * projection reaches 21.3 - so if it sits at the device's fetch RATE then a
 * wider pair block is the lever, and if it sits well below that rate then the
 * block is not what is holding it and widening it buys nothing.
 *
 * 🔴 AND EVERY ITERATION NEEDS ITS OWN OUTPUT, which `bench-webgl-linear.js`'s
 * header explains at length: an Apple GPU discards the earlier of two identical
 * full-viewport draws into one target, so a loop over one output times one
 * iteration and divides by the count. Every arm here writes a ring.
 *
 * 🔴 AND EVERY ARM IS COMPARED AGAINST THE SHIPPED KERNEL, to a reordering's
 * bar - the blocked path sums the same products in the same order, but the
 * compiler fuses a `dot` and a chain of multiply-adds differently.
 */
import {
  CONTRACT_BLOCK_MAX_PAIR_FLOATS, blockedContractSource, contractRepackSource,
  splitContractSource,
} from "../../webgl/af3/trunk/triangle.js";

import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Operands with a spread of magnitudes, so a wrong lane cannot average away. */
const fill = (count, seed) => {
  const values = new Float32Array(count);
  let state = seed | 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    values[i] = ((state >>> 8) / 0x7fffff - 0.5) * 2;
  }
  return values;
};

/**
 * One contraction, as the shipped driver issues it for an OUTGOING direction
 * whose operands are already in place - the transpose and the centre norm are
 * separate dispatches and separate questions.
 */
function contract(execution, { a, b, length, channels, blocked, channelSlowest }, output) {
  if (!blocked) {
    execution.dispatch({
      source: splitContractSource(length, channels, "outgoing", false, true),
      output,
      inputs: { aTex: a, bTex: b },
      label: `bench.contract-only:${length}x${channels}`,
    });
    return;
  }
  const planeTexels = (length / 2) * (length / 4) * (channels / 4);
  const planes = Array.from({ length: 8 },
    (_, p) => execution.allocate(planeTexels * 4, `bench.pair-plane${p}`));
  execution.dispatch({
    source: blockedContractSource(length, channels, "outgoing", false, channelSlowest),
    outputs: planes,
    inputs: { aTex: a, bTex: b },
    label: `bench.contract-block:${length}x${channels}`,
  });
  execution.dispatch({
    source: contractRepackSource(length, channels, channelSlowest),
    output,
    inputs: Object.fromEntries(planes.map((plane, p) => [`plane${p}`, plane])),
    label: `bench.contract-repack:${length}x${channels}`,
  });
  for (const plane of planes) execution.release(plane);
}

export async function main(device, args) {
  const iterations = Number(option(args, "iterations", "8"));
  const channels = Number(option(args, "channels", "256"));
  const lengths = option(args, "lengths", "68,128,256").split(",").map(Number);
  const gl = createWebGL2Context();
  const execution = new Execution(gl, { checkErrors: false });
  const rows = [];

  for (const length of lengths) {
    const pairs = length * length;
    const a = execution.upload(fill(channels * pairs, 7 + length), "bench.a");
    const b = execution.upload(fill(channels * pairs, 11 + length), "bench.b");
    // Every output element is a dot product over k, so two flops a k.
    const flops = 2 * pairs * channels * length;
    /** @type {Record<string, Float32Array>} */
    const answers = {};
    /** @type {Record<string, number>} */
    const timings = {};
    // 🔴 `aliased` IS A DIAGNOSTIC AND NOT A KERNEL: it hands the blocked
    // contraction the SAME texture as both operands, which computes the wrong
    // answer and is never compared for one. It exists to ask whether the two
    // 4.7 MiB operands are CONFLICTING in the texture cache - the one structural
    // difference left from the projection, whose second operand is a small
    // resident weight. If aliasing is much faster, the misses are conflicts; if
    // it is the same, they are not, and that hypothesis dies with the other
    // five in webgl/README.md.
    for (const arm of ["one-pair", "channel-fastest", "blocked", "aliased"]) {
      const blocked = arm !== "one-pair";
      const call = {
        a, b: arm === "aliased" ? a : b, length, channels, blocked,
        channelSlowest: arm !== "channel-fastest",
      };
      const ring = Array.from({ length: iterations },
        () => execution.allocate(pairs * channels, "bench.out"));
      for (let i = 0; i < 2; i += 1) {
        contract(execution, call, ring[i % ring.length]);
        execution.read(ring[i % ring.length]);
      }
      const start = performance.now();
      for (let i = 0; i < iterations; i += 1) contract(execution, call, ring[i]);
      answers[arm] = Float32Array.from(execution.read(ring[iterations - 1]));
      timings[arm] = (performance.now() - start) / iterations;
      for (const out of ring) execution.release(out);
    }
    execution.release(a);
    execution.release(b);
    // What each kernel actually asks the texture units for, which is the
    // ceiling the flops figure cannot distinguish from arithmetic.
    //   one pair a fragment: four channels, each L of a and L of b, by `fetch1`.
    //   blocked: per channel and per k-step, two `a` rows and four `b` rows.
    const oneTexelFetches = pairs * (channels / 4) * 4 * 2 * length;
    const blockedFetches = (length / 2) * (length / 4) * (channels / 4)
      * 4 * (length / 4) * 6;
    rows.push({
      length,
      channels,
      gigaflops: Number((flops / 1e9).toFixed(2)),
      onePairMs: Number(timings["one-pair"].toFixed(2)),
      blockedMs: Number(timings.blocked.toFixed(2)),
      channelFastestMs: Number(timings["channel-fastest"].toFixed(2)),
      aliasedMs: Number(timings.aliased.toFixed(2)),
      aliasedSpeedup: Number((timings.blocked / timings.aliased).toFixed(2)),
      channelFastestGflops:
        Number((flops / timings["channel-fastest"] / 1e6).toFixed(1)),
      orderSpeedup:
        Number((timings["channel-fastest"] / timings.blocked).toFixed(2)),
      speedup: Number((timings["one-pair"] / timings.blocked).toFixed(2)),
      onePairGflops: Number((flops / timings["one-pair"] / 1e6).toFixed(1)),
      blockedGflops: Number((flops / timings.blocked / 1e6).toFixed(1)),
      onePairGfetches: Number((oneTexelFetches / timings["one-pair"] / 1e6).toFixed(2)),
      blockedGfetches: Number((blockedFetches / timings.blocked / 1e6).toFixed(2)),
      onePairFlopsPerFetch: Number((flops / oneTexelFetches).toFixed(1)),
      blockedFlopsPerFetch: Number((flops / blockedFetches).toFixed(1)),
      // 🔴 THE SHIPPED RULE'S OWN VERDICT, so this bench GATES the threshold
      // rather than only measuring beside it: `blocks` is what the driver would
      // pick here, and `rulePicksFaster` is whether that was the faster arm.
      pairFloats: pairs * channels,
      blocks: pairs * channels <= CONTRACT_BLOCK_MAX_PAIR_FLOATS,
      rulePicksFaster: (pairs * channels <= CONTRACT_BLOCK_MAX_PAIR_FLOATS)
        === (timings.blocked <= timings["one-pair"]),
      relRms: Number(relativeRms(answers["one-pair"], answers.blocked).toExponential(2)),
      // The two orders compute the same products in the same order and differ
      // only in which fragment owns which output, so this one is EXACTLY zero.
      orderRelRms: Number(
        relativeRms(answers["channel-fastest"], answers.blocked).toExponential(2)),
    });
  }

  const worst = Math.max(...rows.map((row) => row.relRms));
  const misPicked = rows.filter((row) => !row.rulePicksFaster);
  // A fused multiply-add's worth. Never widen it: at 1e-3 the two kernels are
  // summing different products, not the same ones in a different shape.
  const bar = 1e-6;
  const renderer = gl.getParameter(gl.RENDERER);
  execution.destroy();
  return {
    renderer,
    iterations,
    rows,
    worstRelRms: Number(worst.toExponential(2)),
    bar,
    ceiling: CONTRACT_BLOCK_MAX_PAIR_FLOATS,
    misPicked: misPicked.map((row) =>
      `${row.length}x${row.channels}: the rule ${row.blocks ? "blocks" : "does not block"} `
      + `and blocking is ${row.speedup.toFixed(2)}x`),
    verdict: worst > bar
      ? `FAILED: the pair-blocked contraction disagrees with the one-pair one at `
        + `${worst.toExponential(2)}, past ${bar} - that is not a reordering`
      : misPicked.length > 0
        ? `FAILED: CONTRACT_BLOCK_MAX_PAIR_FLOATS picks the slower kernel at `
          + `${misPicked.length} of ${rows.length} shapes - see misPicked`
        : `ok: the two contractions agree to ${worst.toExponential(2)}, the ceiling picks the `
          + `faster one at all ${rows.length} shapes, and the block runs at `
          + `${rows.map((r) => `${r.blockedGflops} GFLOP/s`).join(", ")}`,
  };
}
