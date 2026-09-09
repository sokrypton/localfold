// What ESMFold2's trunk costs, and what its two f16 knobs are worth.
//
//     node tools/gpu-chrome.mjs tools/gpu/bench-esmfold2-trunk.js --tokens=40,150,300
//
// 🔴 TIME ONLY. The ERROR each arm costs is anchored by
// check-esmfold2-trunk-gpu.js against the native model's own values, and that
// dump is 40 residues - so this exists to say whether a trade priced at 40
// tokens still holds at a length anyone would fold. It synthesises its pair
// representation, which is fine for timing and says nothing at all about
// accuracy; do not read a number here as a bound.
//
// 🔴 THE ARMS INTERLEAVE WITHIN A SHAPE AND THE SHAPES INTERLEAVE TOO. This
// machine drifts by up to 3.2x between runs and a two-minute sweep drifts
// across its own shapes - see CLAUDE.md - so a sweep that runs all of one arm
// and then all of the other is measuring the drift.
import { Esmfold2TrunkGpu, PAIR_CHANNELS } from "../../src/esmfold2/trunk-webgpu.js";
import { profileDevice } from "./profile.js";
import { setDeviceTuning } from "../../src/runtime/device-profile.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Reproducible noise, so two arms see the same input. */
function deterministic(count, seed) {
  const out = new Float32Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = (state / 4294967296) * 2 - 1;
  }
  return out;
}

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1];
};

function syntheticBlock(channels) {
  const square = (seed) => deterministic(channels * channels, seed);
  const wide = (seed) => deterministic(channels * channels * 2, seed);
  const ones = new Float32Array(channels).fill(1);
  const triangle = (seed) => ({
    leftNormInputScale: ones, leftNormInputOffset: new Float32Array(channels),
    centerNormScale: ones, centerNormOffset: new Float32Array(channels),
    outputProjection: square(seed), gatingLinear: square(seed + 1),
    projection: wide(seed + 2), gate: wide(seed + 3),
  });
  return {
    triangleMultiplicationOutgoing: triangle(11),
    triangleMultiplicationIncoming: triangle(21),
    pairTransition: {
      inputLayerNormScale: ones, inputLayerNormOffset: new Float32Array(channels),
      transition1: deterministic(channels * channels * 8, 31),
      transition2: deterministic(channels * 4 * channels, 41),
    },
  };
}

export async function main(device, args = []) {
  // 🔴 `--tune=key=value`, THE SAME FLAG fold.js CARRIES. A knob no gate enters
  // is a knob nobody has checked, and both of this file's kernels choices -
  // `gridAttendMatrix` and `pairTransitionSplit` - are device-profile knobs.
  for (const pair of (args ?? []).filter((a) => a.startsWith("--tune="))
       .flatMap((a) => a.slice("--tune=".length).split(",")).filter(Boolean)) {
    const at = pair.indexOf("=");
    if (at < 0) throw new Error(`--tune wants key=value, got ${pair}`);
    const raw = pair.slice(at + 1);
    let value;
    try { value = JSON.parse(raw); } catch { value = raw; }
    setDeviceTuning(device, { [pair.slice(0, at)]: value });
  }
  const tokens = option(args, "tokens", "40,150,300").split(",").map(Number);
  const channels = Number(option(args, "channels", String(PAIR_CHANNELS)));
  const blockCount = Number(option(args, "blocks", "24"));
  const repeats = Number(option(args, "repeats", "3"));
  const precisions = option(args, "precision", "f32,f16:f32,f32:f16,f16").split(",");

  // 🔴 ONE SET OF WEIGHTS FOR EVERY BLOCK, WHICH IS FINE HERE AND NOWHERE ELSE.
  // These kernels' cost is their shapes; the values only matter to the answer,
  // which this tool does not check. 24 distinct blocks would be 144 MiB of
  // synthetic float32 in the tab for no difference in the timing.
  const block = syntheticBlock(channels);
  const blocks = new Array(blockCount).fill(block);

  const stacks = new Map();
  const stackFor = (precision) => {
    if (!stacks.has(precision)) {
      // `default` overrides nothing, so the shipped settings are an arm here
      // too rather than a name this tool would have to keep in step.
      const [staged, accumulate] = precision === "default" ? [undefined, undefined]
        : (precision.includes(":") ? precision.split(":") : [precision, precision]);
      stacks.set(precision, new Esmfold2TrunkGpu(device, staged === undefined ? {}
        : { stagedPrecision: staged, accumulatePrecision: accumulate }));
    }
    return stacks.get(precision);
  };

  // 🔴 PROFILE, DO NOT BISECT BY DELETION - see CLAUDE.md. Every pass this
  // track encodes already carries a label, so wrapping the device times all of
  // them with no kernel change. One shape and one precision, because 24 blocks
  // is 144 passes and the query set holds 2048.
  if (args.includes("--profile")) {
    const n = tokens[0];
    const profile = profileDevice(device);
    if (profile === null) throw new Error("this device has no timestamp-query");
    const pair = deterministic(n * n * channels, 991 + n);
    const pairMask = new Float32Array(n * n).fill(1);
    // 🔴 EVERY PRECISION, NOT THE FIRST, so two arms can be compared per KERNEL
    // rather than per wall clock. The wall figure at 300 tokens moves by more
    // between two runs of one arm than the arms differ by; the device's own
    // timestamps do not.
    const reports = [];
    for (const precision of precisions) {
      // ...once to compile, then reset, so the report is a steady pass.
      await stackFor(precision).run({ pair: Float32Array.from(pair), pairMask },
                                    blocks, { n, channels });
      profile.reset();
      const result = await stackFor(precision).run(
        { pair: Float32Array.from(pair), pairMask }, blocks, { n, channels });
      const passes = await profile.report();
      const total = passes.reduce((sum, p) => sum + p.ms, 0);
      reports.push({
        precision,
        wallMilliseconds: Number(result.elapsedMilliseconds.toFixed(1)),
        gpuMilliseconds: Number(total.toFixed(1)),
        passes: passes.map((p) => ({
          label: p.label, ms: Number(p.ms.toFixed(2)),
          share: Number((p.ms / total).toFixed(4)),
        })),
      });
    }
    profile.restore();
    if (reports.length > 1) {
      return { tokens: n, channels, blocks: blockCount, arms: reports };
    }
    const { passes } = reports[0];
    const total = reports[0].gpuMilliseconds;
    const precision = precisions[0];
    return {
      profile: { tokens: n, precision, channels, blocks: blockCount,
                 wallMilliseconds: reports[0].wallMilliseconds,
                 measuredMilliseconds: Number(total.toFixed(1)) },
      passes: passes.map((p) => ({ ...p,
        share: Number((p.ms / total).toFixed(4)),
        perBlockMilliseconds: Number((p.ms / blockCount).toFixed(2)) })),
    };
  }

  const samples = new Map();
  const key = (n, precision) => `${n}:${precision}`;
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    for (const n of tokens) {
      const pair = deterministic(n * n * channels, 991 + n);
      const pairMask = new Float32Array(n * n).fill(1);
      for (const precision of precisions) {
        const result = await stackFor(precision).run(
          { pair: Float32Array.from(pair), pairMask }, blocks, { n, channels });
        const at = key(n, precision);
        if (!samples.has(at)) samples.set(at, { times: [], peak: result.memory.peakBytes });
        samples.get(at).times.push(result.elapsedMilliseconds);
      }
    }
  }

  const rows = [];
  for (const n of tokens) {
    const baseline = median(samples.get(key(n, precisions[0])).times);
    for (const precision of precisions) {
      const sample = samples.get(key(n, precision));
      const ms = median(sample.times);
      rows.push({
        tokens: n, precision,
        milliseconds: Number(ms.toFixed(1)),
        speedup: Number((baseline / ms).toFixed(3)),
        peakMiB: Number((sample.peak / 2 ** 20).toFixed(1)),
        millisecondsPerBlock: Number((ms / blockCount).toFixed(2)),
      });
    }
  }
  return { channels, blocks: blockCount, repeats, precisions,
           note: "timing only; the error bound is check-esmfold2-trunk-gpu.js's",
           rows };
}
