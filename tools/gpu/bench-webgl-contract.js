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
 *
 * 🔴 WHAT BINDS IT: NOT REUSE, NOT BYTES, NOT ARITHMETIC, AND NOT THE RUN
 * LENGTH. The shipped tiled kernel is **4.85 ms at 33.2 GFLOP/s** at 68 tokens
 * and 256 channels, where the dense projection reaches 375. Four arms, each
 * removing one candidate:
 *
 * | arm | factor | what it rules out |
 * |---|---|---|
 * | `aliased-tiled` - both operands one texture | **1.05-1.14x** | operand reuse |
 * | `half-tiled` - both operands float16 | **0.97-1.06x** | bandwidth |
 * | (probe-alu's vec4 ceiling is 5034 GFLOP/s) | 142x headroom | arithmetic |
 * | `--lengths=68,128,256` - the k run is L/4 | 33.2 / 40.6 / 19.3 GF/s | the run length |
 *
 * 🔴 AND `aliased` ALONE WAS THE WRONG ARM FOR FIVE MONTHS, which is why
 * `aliased-tiled` is here. The shipped contraction is TILE-MAPPED and aliasing
 * had only ever been measured against the untiled block, so "aliasing is 1.11x"
 * was a statement about a kernel nothing dispatches. Same mistake in shape as
 * `probe-webgl-linear-operand.js`'s straw baseline: **a diagnostic arm has to
 * sit beside the kernel that ships**, and the tiled answer happens to agree
 * (1.08x) only by luck.
 *
 * So the contraction's 33 GFLOP/s is not explained by any of the four, and the
 * remaining candidate nothing here can vary is OCCUPANCY: 36,992 fragments
 * where `probe-webgl-gemm-ceiling.js` needs 262,144 to reach 2000 GFLOP/s, and
 * the fragment count is `pairs * channels / 32` - fixed by the output budget.
 *
 * 🔴 AND THE VERDICT IS RED AT `--lengths=256` AND WAS BEFORE THIS NOTE. The
 * ceiling declines blocking at 16.8M pair floats and blocking is **1.12x**
 * there (268.52 ms against the split kernel's 301.01), while at 128 it is
 * rightly declined (187.86 against 42.18) - so the relation is NOT monotone in
 * the pair size and a single size threshold cannot express it. Pre-existing,
 * reproduced on the committed bench, and at a length the current work does not
 * fold; it wants a rule over the shape rather than the size.
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
function contract(execution,
  { a, b, length, channels, blocked, channelSlowest, tileMapped }, output) {
  if (!blocked) {
    execution.dispatch({
      source: splitContractSource(length, channels, "outgoing", false, true),
      output,
      inputs: { aTex: a, bTex: b },
      label: `bench.contract-only:${length}x${channels}`,
    });
    return;
  }
  // The tiled mapping lays the fragment grid as whole texture rows, so its
  // planes are `ceil(BLOCKS / TILEW)` rows of W rather than BLOCKS * C4 texels.
  const tileWidth = Math.max(1, Math.floor(2048 / (channels / 4)));
  const blocks = (length / 2) * (length / 4);
  const planeTexels = tileMapped
    ? Math.ceil(blocks / tileWidth) * 2048
    : blocks * (channels / 4);
  const planes = Array.from({ length: 8 },
    (_, p) => execution.allocate(planeTexels * 4, `bench.pair-plane${p}`));
  execution.dispatch({
    source: blockedContractSource(length, channels, "outgoing", false, channelSlowest,
      length, tileMapped),
    outputs: planes,
    inputs: { aTex: a, bTex: b },
    label: `bench.contract-block:${length}x${channels}`,
  });
  execution.dispatch({
    source: contractRepackSource(length, channels, channelSlowest, length, tileMapped),
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
    // 🔴 THE SAME OPERANDS IN FLOAT16, because the arithmetic says this kernel
    // is bandwidth-bound and neither aliasing arm moves it. Two 4.7 MiB
    // operands at 68 tokens and 256 channels, 408 fetch4 a fragment over 36,992
    // fragments: 15.1 M fetch4 in 4.55 ms is **3.32 G fetch4/s, about 53 GB/s**,
    // where the dense projection reaches 17.6 G fetch4/s and 282 - which is far
    // past DRAM and so must be cache-served. Aliasing is 1.08x because halving
    // 9.4 MiB to 4.7 is still past the cache, and that is the tell: the fix is
    // fewer BYTES, not better reuse. `uploadWeight` gives float16 above its
    // floor, which both operands clear.
    const aHalf = execution.uploadWeight(fill(channels * pairs, 7 + length), "bench.a16");
    const bHalf = execution.uploadWeight(fill(channels * pairs, 11 + length), "bench.b16");
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
    // 🔴 AND `aliased-tiled` IS THE ARM THAT WAS MISSING, BECAUSE `aliased`
    // ALONE ANSWERS ABOUT A KERNEL NOTHING DISPATCHES. The shipped contraction
    // is TILE-MAPPED, and aliasing was only ever measured against the untiled
    // block - so "aliasing is 1.11x" said the UNTILED kernel is not conflict-
    // bound and said nothing at all about the one that folds. Same mistake in
    // shape as `probe-webgl-linear-operand.js`'s straw baseline: a diagnostic
    // arm has to sit beside the kernel that ships.
    for (const arm of ["one-pair", "channel-fastest", "blocked", "aliased", "tiled",
      "aliased-tiled", "half-tiled"]) {
      const blocked = arm !== "one-pair";
      const half = arm === "half-tiled";
      const call = {
        a: half ? aHalf : a,
        b: arm.startsWith("aliased") ? a : (half ? bHalf : b),
        length,
        channels,
        blocked,
        channelSlowest: arm !== "channel-fastest",
        tileMapped: arm.endsWith("tiled"),
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
      aliasedTiledMs: Number(timings["aliased-tiled"].toFixed(2)),
      halfTiledMs: Number(timings["half-tiled"].toFixed(2)),
      tiledMs: Number(timings.tiled.toFixed(2)),
      tiledSpeedup: Number((timings.blocked / timings.tiled).toFixed(2)),
      aliasedSpeedup: Number((timings.blocked / timings.aliased).toFixed(2)),
      // The one that matters: the SHIPPED tiled kernel against the same kernel
      // with both operands in one texture.
      aliasedTiledSpeedup: Number((timings.tiled / timings["aliased-tiled"]).toFixed(2)),
      tiledGflops: Number((2 * length * length * length * channels
        / timings.tiled / 1e6).toFixed(1)),
      halfTiledSpeedup: Number((timings.tiled / timings["half-tiled"]).toFixed(2)),
      halfTiledGflops: Number((2 * length * length * length * channels
        / timings["half-tiled"] / 1e6).toFixed(1)),
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
      // The tiled mapping moves which fragment owns which output and nothing
      // else, so this is exactly zero or the mapping does not invert.
      tiledRelRms: Number(
        relativeRms(answers.tiled, answers.blocked).toExponential(2)),
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
