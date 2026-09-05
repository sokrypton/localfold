/**
 * The runtime cost model, and the bar it drives.
 *
 * 🔴 THE MEASUREMENTS ARE THE TEST. A fitted constant has no meaning on its own
 * and every one of these functions will return a plausible number for any
 * input, so what is checked here is agreement with what tools/gpu/bench-runtime.js
 * actually measured - and the QUALITATIVE facts the old bar got wrong, which
 * survive re-fitting on another machine: that the trunk grows faster than the
 * sampler, that AF3's trunk barely notices alignment depth while AF2's cost is
 * mostly depth, and that a structure-module step is not an evoformer block.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  af2BlockUnits, af2Plan, af2StructureStepUnits, af3DenoiserCallUnits, af3Plan,
  af3TrunkPassUnits, af3TrunkStageSpans, AF3_TRUNK_STAGE_SHARES,
  describeRemaining, planTotal, RuntimeEstimator,
} from "../src/runtime/cost-model.js";

/** Measured on the reference M2 by tools/gpu/bench-runtime.js. */
const AF3_TRUNK = [
  { tokens: 59, rows: 32, ms: 392 }, { tokens: 128, rows: 32, ms: 1597 },
  { tokens: 192, rows: 32, ms: 3584 }, { tokens: 256, rows: 32, ms: 6830 },
  { tokens: 128, rows: 1, ms: 1572 },
  { tokens: 128, rows: 128, ms: 1677 },
  { tokens: 128, rows: 512, ms: 1802 },
];
const AF3_DENOISER = [
  { tokens: 59, ms: 109 }, { tokens: 128, ms: 190 },
  { tokens: 192, ms: 276 }, { tokens: 256, ms: 382 },
];
const AF2_STACK = [
  { length: 59, rows: 32, ms: 824 }, { length: 96, rows: 32, ms: 1873 },
  { length: 128, rows: 32, ms: 3066 }, { length: 160, rows: 32, ms: 6578 },
  { length: 192, rows: 32, ms: 9114 }, { length: 224, rows: 32, ms: 12356 },
  { length: 256, rows: 32, ms: 16060 },
  { length: 128, rows: 1, ms: 1672 }, { length: 128, rows: 16, ms: 3126 },
  { length: 128, rows: 64, ms: 4097 }, { length: 128, rows: 256, ms: 9764 },
  { length: 128, rows: 512, ms: 18792 },
];

const within = (predicted, measured, percent, what) => {
  const error = Math.abs(predicted - measured) / measured * 100;
  assert.ok(error <= percent,
    `${what}: predicted ${predicted.toFixed(0)} against ${measured}, off by `
    + `${error.toFixed(1)}% which is over ${percent}%`);
};

describe("the runtime cost model", () => {
  it("matches the measured AF3 trunk within 5%", () => {
    for (const { tokens, rows, ms } of AF3_TRUNK) {
      within(af3TrunkPassUnits(tokens, rows), ms, 5, `trunk ${tokens}/${rows}`);
    }
  });

  it("matches the measured AF3 denoiser within 3%", () => {
    for (const { tokens, ms } of AF3_DENOISER) {
      within(af3DenoiserCallUnits(tokens), ms, 3, `denoiser ${tokens}`);
    }
  });

  it("matches the measured AF2 stack within 25%", () => {
    // Looser on purpose: AF2's cost steps between 128 and 160 residues and a
    // smooth fit cannot follow that. See the note in cost-model.js.
    for (const { length, rows, ms } of AF2_STACK) {
      within(af2BlockUnits(length, rows) * 48, ms, 25, `AF2 ${length}/${rows}`);
    }
  });

  it("has the trunk grow away from the sampler with length, which is the bug it fixes", () => {
    // The old bar assumed one ratio for every length: a trunk pass was always
    // 4.35 denoiser calls. It is 3.1 at 59 tokens and over 15 at 256.
    const ratio = (tokens) => af3TrunkPassUnits(tokens, 32) / af3DenoiserCallUnits(tokens);
    assert.ok(ratio(59) < 4, `at 59 tokens the ratio is ${ratio(59).toFixed(1)}, expected under 4`);
    assert.ok(ratio(256) > 12, `at 256 it is ${ratio(256).toFixed(1)}, expected over 12`);
    assert.ok(ratio(256) > 3 * ratio(59), "the ratio should more than triple over that range");
  });

  it("has AF3 barely notice alignment depth and AF2 dominated by it", () => {
    const af3Deep = af3TrunkPassUnits(128, 512) / af3TrunkPassUnits(128, 1);
    const af2Deep = af2BlockUnits(128, 512) / af2BlockUnits(128, 1);
    assert.ok(af3Deep < 1.3, `AF3 at 512 rows is ${af3Deep.toFixed(2)}x its cost at 1`);
    assert.ok(af2Deep > 5, `AF2 at 512 rows is only ${af2Deep.toFixed(2)}x its cost at 1`);
  });

  it("does not count a structure-module step as an evoformer block", () => {
    // The fault in AF2's old bar: eleven structure steps and two confidence
    // steps counted the same as thirteen blocks.
    const block = af2BlockUnits(128, 512);
    const step = af2StructureStepUnits(128);
    assert.ok(block / step > 20, `a block is only ${(block / step).toFixed(1)} structure steps`);
  });
});

describe("planning a fold", () => {
  it("splits a page fold the way the page's clock splits it", () => {
    // 🔴 THE BAR'S WORST FAULT WAS HERE AND NOT IN A KERNEL. Featurisation and
    // the per-frame work were fitted when they cost seconds; they cost
    // milliseconds now, and the stale coefficients handed featurisation 23% of
    // a fold that spends 2% of its clock there while the sampler took 64% of
    // the bar for 40% of the clock. Measured with
    // tools/gpu/probe-progress-bar.js at 58 tokens, 4 passes, 16 flow steps:
    //
    //     features 2.0%   trunk 39.7%   compile 16.1%   calls 38.6%
    //
    // and the worst bar-versus-clock error went from 0.18 to 0.07 there, and
    // from 0.24 to under 0.04 at 128 and 192 tokens.
    const measured = {
      features: 0.020, trunk: 0.397, "sampler-warmup": 0.161, sampler: 0.386,
    };
    const plan = af3Plan({ tokens: 58, rows: 32, passes: 4, calls: 16, atoms: 1392 });
    const total = planTotal(plan);
    for (const stage of plan.stages) {
      const share = (stage.units * stage.count) / total;
      const off = Math.abs(share - measured[stage.name]);
      assert.ok(off < 0.08, `${stage.name} takes ${(share * 100).toFixed(1)}% of `
        + `the bar against ${(measured[stage.name] * 100).toFixed(1)}% of the clock`);
    }
  });

  it("puts the sampler's share up as the chain gets shorter", () => {
    const shortFold = af3Plan({ tokens: 59, rows: 32, passes: 4, calls: 8 });
    const longFold = af3Plan({ tokens: 256, rows: 32, passes: 4, calls: 8 });
    const samplerShare = (plan) => {
      const sampler = plan.stages.find((s) => s.name === "sampler");
      return sampler.units * sampler.count / planTotal(plan);
    };
    assert.ok(samplerShare(shortFold) > samplerShare(longFold));
    assert.ok(samplerShare(shortFold) > 0.35, "at 59 tokens the sampler is a third of the work");
    assert.ok(samplerShare(longFold) < 0.15, "at 256 it is under a seventh");
  });

  it("counts every recycle of an AF2 stack", () => {
    const one = planTotal(af2Plan({ length: 128, extraRows: 1024, mainRows: 512,
      extraBlocks: 4, mainBlocks: 48, passes: 1 }));
    const three = planTotal(af2Plan({ length: 128, extraRows: 1024, mainRows: 512,
      extraBlocks: 4, mainBlocks: 48, passes: 3 }));
    assert.equal(Math.round(three / one), 3);
  });
});

describe("the estimator", () => {
  const plan = { stages: [{ name: "a", units: 10, count: 10 }] };

  it("reports a fraction with no calibration at all", () => {
    const estimator = new RuntimeEstimator(plan, 0);
    assert.equal(estimator.fraction(), 0);
    estimator.complete("a", 5, 1000);
    assert.equal(estimator.fraction(), 0.5);
    estimator.complete("a", 5, 2000);
    assert.equal(estimator.fraction(), 1);
  });

  it("learns the machine's speed from the run in front of it", () => {
    // Ten units of work took 4000 ms, so this device is 4 ms per unit and the
    // remaining ninety units are 360 s - whatever the reference machine did.
    const estimator = new RuntimeEstimator(plan, 0);
    estimator.complete("a", 1, 4000);
    assert.equal(estimator.remainingMs(), 90 * 400);
    assert.equal(estimator.totalMs(), 100 * 400);
  });

  it("says nothing about time until there is enough work to judge by", () => {
    const big = new RuntimeEstimator({ stages: [{ name: "a", units: 1, count: 1000 }] }, 0);
    big.complete("a", 10, 5000);          // 1% done: pipeline compilation, not a rate
    assert.equal(big.remainingMs(), undefined);
    big.complete("a", 90, 6000);          // 10% done
    assert.ok(big.remainingMs() > 0);
  });

  it("never lets the bar retreat", () => {
    const estimator = new RuntimeEstimator(plan, 0);
    estimator.completedUnits(60, 1000);
    assert.equal(estimator.fraction(), 0.6);
    estimator.completedUnits(30, 2000);   // a stage recounted, say
    assert.equal(estimator.fraction(), 0.6);
  });

  it("refuses a plan with no work in it", () => {
    assert.throws(() => new RuntimeEstimator({ stages: [] }), RangeError);
    assert.throws(() => new RuntimeEstimator({ stages: [{ name: "a", units: 0, count: 5 }] }),
                  RangeError);
  });

  it("names a stage it does not have rather than silently counting nothing", () => {
    const estimator = new RuntimeEstimator(plan, 0);
    assert.throws(() => estimator.complete("trunk"), /no stage named trunk/);
  });
});

describe("bad shapes", () => {
  // 🔴 A BAR IS NOT WORTH A FAILED PREDICTION. One undefined alignment depth
  // made every unit NaN, which HTMLProgressElement rejected as "non-finite",
  // which unwound into the fold and reported it as failed. Nothing here may
  // return a number the DOM will not take.
  it("never returns a non-finite number, whatever it is handed", () => {
    const nonsense = [undefined, null, Number.NaN, Infinity, -1, 0, "150"];
    for (const bad of nonsense) {
      assert.ok(Number.isFinite(af3TrunkPassUnits(bad, 32)), `trunk tokens=${bad}`);
      assert.ok(Number.isFinite(af3TrunkPassUnits(150, bad)), `trunk rows=${bad}`);
      assert.ok(Number.isFinite(af3DenoiserCallUnits(bad)), `denoiser tokens=${bad}`);
      assert.ok(Number.isFinite(af2BlockUnits(bad, 32)), `af2 length=${bad}`);
      assert.ok(Number.isFinite(af2BlockUnits(150, bad)), `af2 rows=${bad}`);
      assert.ok(Number.isFinite(af2StructureStepUnits(bad)), `structure length=${bad}`);
    }
  });

  it("totals a plan with a missing depth rather than poisoning it", () => {
    const total = planTotal(af2Plan({
      length: 150, extraRows: undefined, mainRows: Number.NaN,
      extraBlocks: 4, mainBlocks: 48, passes: 4,
    }));
    assert.ok(Number.isFinite(total) && total > 0, `total was ${total}`);
  });

  it("survives an AF3 plan built from nothing", () => {
    const total = planTotal(af3Plan({ tokens: undefined, rows: undefined,
                                      passes: 1, calls: 8, atoms: undefined }));
    assert.ok(Number.isFinite(total) && total > 0, `total was ${total}`);
  });
});

describe("describing what is left", () => {
  it("rounds coarsely, because it is an estimate", () => {
    assert.equal(describeRemaining(1200), "5 s");   // every case is a duration
    assert.equal(describeRemaining(12_000), "10 s");
    assert.equal(describeRemaining(43_000), "45 s");
    assert.equal(describeRemaining(80_000), "1 min 15 s");
    assert.equal(describeRemaining(121_000), "2 min");
  });

  it("says nothing when it knows nothing", () => {
    assert.equal(describeRemaining(undefined), undefined);
    assert.equal(describeRemaining(Number.NaN), undefined);
  });
});

describe("the trunk's stages, on the bar", () => {
  // 🔴 WHAT THIS PROTECTS IS "THE BAR MOVES", NOT A TIMING. The shares are
  // measured at one shape and are only roughly right at another; what must not
  // regress is that the four stages before the pairformer have somewhere to
  // go, in the order they run, and that the pairformer does not start at zero.
  it("spans the whole pass, in order, with no gaps", () => {
    const spans = af3TrunkStageSpans();
    assert.deepEqual([...spans.keys()],
      AF3_TRUNK_STAGE_SHARES.map(([name]) => name));
    let at = 0;
    for (const [, span] of spans) {
      assert.equal(span.from, at, "a stage begins where the last one ended");
      assert.ok(span.to > span.from, "and covers some of the pass");
      at = span.to;
    }
    assert.ok(Math.abs(at - 1) < 1e-12, "and the last one ends at the pass");
  });

  it("runs the embedder, the templates and the MSA before the pairformer", () => {
    const spans = af3TrunkStageSpans();
    assert.ok(spans.get("embedder").to <= spans.get("template").from);
    assert.ok(spans.get("template").to <= spans.get("msa-stack").from);
    assert.ok(spans.get("msa-stack").to <= spans.get("pairformer").from);
  });

  // 🔴 THE NUMBER THE COMPLAINT WAS ABOUT. Everything before the pairformer
  // used to be worth nothing on the bar, so the first thing a large fold could
  // report was block 1 of 48 - after the embedder, the template embedder and
  // the MSA stack had each run for seconds. That head has to be a visible
  // fraction of the pass, not a rounding error.
  it("gives the silent stages a sixth of the pass to move through", () => {
    const spans = af3TrunkStageSpans();
    const head = spans.get("pairformer").from;
    assert.ok(head > 0.1, `the silent stages hold ${head}, which reads as frozen`);
    assert.ok(head < 0.3, `the silent stages hold ${head}, more than measured`);
  });

  it("leaves the pairformer the bulk of it", () => {
    const spans = af3TrunkStageSpans();
    const pairformer = spans.get("pairformer");
    assert.ok(pairformer.to - pairformer.from > 0.75);
  });
});
