/**
 * Two split settings, INTERLEAVED in one process, to resolve below the drift.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-difftx-splits.js --knob=attnKSplits --a=1 --b=4
 *
 * 🔴 THIS BOX DRIFTS AND A SWEEP OF SEPARATE PROCESSES CANNOT SEE A FEW
 * PERCENT. `attnSplits` 2 and 4 measured 1.5-1.9% faster than 1 across
 * sequential folds - monotonic, which is suggestive, but the same configuration
 * re-run in a later process differed by 2.5%, so the effect and the noise were
 * the same size. CLAUDE.md says interleave; nothing here did for this knob.
 *
 * A and B alternate call by call on ONE device with ONE set of weights, so a
 * drift that moves one arm moves the other in the same direction and the
 * PAIRED difference survives it. Reported as the median of per-pair ratios
 * rather than a ratio of medians, which is the statistic that is robust when
 * the absolute level wanders.
 *
 * 🔴 WHAT IT SETTLED, so nobody re-runs it: at 68 tokens, paired,
 * `attnKSplits` 1 -> 2 is **1.018** and 1 -> 4 is **1.009**. NOT MONOTONIC,
 * which is what noise looks like when it dominates, and 1.8% of an 11 ms
 * transformer call inside a 16 ms step is under 1% of a fold. Not taken.
 *
 * 🔴 AND `normKSplits` NO LONGER LOSES, WHICH RETIRES 311d7a4's CONCLUSION.
 * That commit measured "fused 1.43 ms, two parts 3.15, four 2.21", called the
 * split a loss and built a mechanism on top of it - all on a kernel that was
 * adding the bias twice. On the FIXED kernel, paired over eleven pairs:
 * 1 -> 2 is **1.017** and 1 -> 4 is **1.025**, correct at relRMS 1.6e-4 and
 * 1.3e-4. It wins, slightly. Still not taken: 2.5% of the transformer is ~1.3%
 * of a fold, the same magnitude attnSplits was refused at, and a knob that was
 * broken an hour ago does not get enabled on a module bench.
 *
 * 🔴 AND IT CHECKS THE ANSWER, because the fastest arm of the last sweep was
 * fast for being wrong: `--norm-splits=4` read 3.556 s against a 3.639 s
 * baseline and returns pLDDT 66.347114 against 84.208873. Any arm whose output
 * differs from A's is reported as broken and its timing withheld - a number
 * from a kernel that is not computing the model is worse than no number.
 */
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion-transformer-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights } from "../../src/af3/diffusion-weights.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "68"));
  const knob = option(args, "knob", "attnKSplits");
  const a = JSON.parse(option(args, "a", "1"));
  const b = JSON.parse(option(args, "b", "4"));
  const pairs = Number(option(args, "pairs", "9"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = await diffusionWeights(store);
  const tx = weights.transformer;
  const { channels, condChannels, pairChannels } = tx;

  const noise = (n, seed) => {
    const out = new Float32Array(n);
    let state = seed >>> 0;
    for (let i = 0; i < n; i += 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      out[i] = (state / 4294967296) * 2 - 1;
    }
    return out;
  };
  // Off zero, for the reason check-difftx-splits.js gives: adaln is a
  // conditioned LayerNorm and a zero-mean input is where a centring bug hides.
  const act = noise(tokens * channels, 7).map((v) => v + 0.7);
  const cond = noise(tokens * condChannels, 11);
  const pairCond = noise(tokens * tokens * pairChannels, 13);
  const mask = new Float32Array(tokens).fill(1);

  // The shipped rule, with only the knob under test moved - benchmarking a
  // geometry no device runs would answer a question nobody asked.
  const base = { ...tx, kSplits: 16, outKSplits: 4, attnKSplits: 1, normKSplits: 1,
                 qkvgTile: 4, wideTile: 4, normSplit: true };
  const armA = new Af3DiffusionTransformerGpu(device);
  const armB = new Af3DiffusionTransformerGpu(device);
  const shapeA = { ...base, [knob]: a };
  const shapeB = { ...base, [knob]: b };

  const once = async (arm, shape) => {
    const started = performance.now();
    const out = await arm.run(act, cond, pairCond, mask, tokens, shape);
    await device.queue.onSubmittedWorkDone();
    return { ms: performance.now() - started, out };
  };

  // Warm both, so neither pays compilation inside a timed pair.
  const first = await once(armA, shapeA);
  const firstB = await once(armB, shapeB);
  // 🔴 UNWRAP OR THROW. `run()` resolves to {output, elapsedMilliseconds,
  // memory}. Handed the objects, `x.length` is undefined, the loop never runs
  // and this returns Math.sqrt(0 / 1e-30) - exactly zero. This file's whole
  // reason for existing is that a wrong kernel measured FASTER, and a
  // correctness gate that cannot fail would have let exactly that through
  // again. The same mistake sat in check-difftx-splits.js for 106 arms.
  const rel = relativeRms;
  const agreement = Number(rel(firstB.out, first.out).toPrecision(3));

  const ratios = [];
  const aMs = [];
  const bMs = [];
  for (let i = 0; i < pairs; i += 1) {
    // 🔴 ORDER ALTERNATES WITHIN THE PAIR TOO. A fixed A-then-B order gives B
    // whatever warmth A left behind, every time.
    let ta;
    let tb;
    if (i % 2 === 0) {
      ta = (await once(armA, shapeA)).ms;
      tb = (await once(armB, shapeB)).ms;
    } else {
      tb = (await once(armB, shapeB)).ms;
      ta = (await once(armA, shapeA)).ms;
    }
    aMs.push(ta); bMs.push(tb); ratios.push(ta / tb);
  }
  const median = (xs) => [...xs].sort((p, q) => p - q)[Math.floor(xs.length / 2)];

  return {
    tokens, knob, a, b, pairs,
    // 🔴 WITHHELD IF THE ARMS DISAGREE. See the header.
    correct: agreement < 1e-3,
    relRmsBvsA: agreement,
    aMedianMs: Number(median(aMs).toFixed(3)),
    bMedianMs: Number(median(bMs).toFixed(3)),
    pairedSpeedupMedian: agreement < 1e-3
      ? Number(median(ratios).toPrecision(4)) : null,
    pairedRatios: agreement < 1e-3
      ? ratios.map((r) => Number(r.toPrecision(4))) : null,
  };
}
