import test from "node:test";
import assert from "node:assert/strict";
import { derivedSplitRule } from "../src/af3/diffusion-transformer-webgpu.js";

// AF3's diffusion transformer at the shape a small fold runs: 768 channels,
// 16 heads of 48, 256 lanes, a token tile of 4.
const af3 = (over) => ({ channels: 768, width: 768, lanes: 256, tile: 4, ...over });

test("a wide device splits a small token count", () => {
  // 68 tokens: ceil(68/4) * (768/256) = 51 workgroups against 2048 measured.
  const rule = derivedSplitRule(af3({ measuredWidth: 2048, rows: 68 }));
  assert.notEqual(rule, null);
  assert.equal(rule.splits, 32);
  assert.equal(rule.tile, 4);
});

// 🔴 THE CROSSOVER FALLS OUT OF THE ARITHMETIC RATHER THAN BEING A NUMBER.
// The prior spells it 512 tokens; here a large enough token count makes the
// unsplit dispatch as wide as the device and the rule declines on its own.
test("a large token count needs no split, and says so", () => {
  assert.equal(derivedSplitRule(af3({ measuredWidth: 2048, rows: 4096 })), null);
});

// 🔴 AND THIS IS WHAT A TABLE CANNOT DO. The same fold on a device a fiftieth
// of this one's width is already full, so the partial traffic and the reduce
// pass would be pure cost.
test("a narrow device does not split the token count a wide one does", () => {
  assert.notEqual(derivedSplitRule(af3({ measuredWidth: 2048, rows: 68 })), null);
  assert.equal(derivedSplitRule(af3({ measuredWidth: 40, rows: 68 })), null);
});

test("the split always divides the channel count", () => {
  for (const channels of [768, 384, 256, 128]) {
    for (const width of [40, 200, 2048, 100000]) {
      const rule = derivedSplitRule(af3({ channels, measuredWidth: width, rows: 68 }));
      if (rule !== null) assert.equal(channels % rule.splits, 0, `${channels} / ${rule.splits}`);
    }
  }
});

// The sweep behind the constant: 4 -> 2857 ms, 8 -> 2754, 16 -> 2780,
// 32 -> 2816, 48 -> 2978. Anything from 8 to 32 is within 2%.
test("the split stays inside the measured plateau", () => {
  for (const width of [200, 512, 2048, 100000]) {
    const rule = derivedSplitRule(af3({ measuredWidth: width, rows: 68 }));
    if (rule !== null) assert.ok(rule.splits >= 8 && rule.splits <= 32, `${rule.splits}`);
  }
});

// ---------------------------------------------------------------------------

import { batchedGatesAffordable } from "../src/af3/diffusion-transformer-webgpu.js";

const gates = (over) => ({ allowed: true, budgetBytes: undefined,
                           residentBytes: 0, bytes: 60e6, ...over });

test("a device with no ceiling takes the batched gate", () => {
  assert.equal(batchedGatesAffordable(gates()), true);
  assert.equal(batchedGatesAffordable(gates({ budgetBytes: null })), true);
});

// 🔴 THE VETO, NOT A TERM. Under the budget fallback the blocks are uploaded
// and released per call, and a duplicate of weights that are not being kept is
// the trade that fallback exists to refuse.
test("a device that has refused residency never takes it", () => {
  assert.equal(batchedGatesAffordable(gates({ allowed: false })), false);
  assert.equal(batchedGatesAffordable(
    gates({ allowed: false, budgetBytes: undefined })), false);
});

test("a ceiling with room takes it and one without does not", () => {
  assert.equal(batchedGatesAffordable(
    gates({ budgetBytes: 1000e6, residentBytes: 100e6 })), true);
  assert.equal(batchedGatesAffordable(
    gates({ budgetBytes: 300e6, residentBytes: 200e6 })), false);
});

// The headroom is three times the estimate, so a budget with exactly the
// estimate left is refused rather than squeezed.
test("headroom is a multiple, not a fit", () => {
  assert.equal(batchedGatesAffordable(
    gates({ budgetBytes: 260e6, residentBytes: 200e6 })), false);
  assert.equal(batchedGatesAffordable(
    gates({ budgetBytes: 400e6, residentBytes: 200e6 })), true);
});

// ---------------------------------------------------------------------------

import { derivedTokenTile } from "../src/af3/diffusion-transformer-webgpu.js";

test("nothing measured keeps the model's own tile", () => {
  assert.equal(derivedTokenTile({ measuredWidth: null, rows: 68, cap: 4 }), 4);
  assert.equal(derivedTokenTile({ measuredWidth: undefined, rows: 68, cap: 2 }), 2);
});

// The A100's prior says tile 1 below 175 tokens; a 68-token fold leaves 68
// workgroups against 2048 measured, so no tile above one is affordable.
test("a small token count on a wide device takes the smallest tile", () => {
  assert.equal(derivedTokenTile({ measuredWidth: 2048, rows: 68, cap: 4 }), 1);
});

test("enough rows earn the bigger tile", () => {
  assert.equal(derivedTokenTile({ measuredWidth: 2048, rows: 4096, cap: 4 }), 2);
  assert.equal(derivedTokenTile({ measuredWidth: 2048, rows: 8192, cap: 4 }), 4);
});

// 🔴 A NARROW DEVICE IS THE CASE THE MODEL'S CONSTANT WAS WRITTEN FOR, and the
// rule has to reproduce it rather than starve it of weight amortisation.
test("a narrow device takes the largest tile it is allowed", () => {
  assert.equal(derivedTokenTile({ measuredWidth: 16, rows: 68, cap: 4 }), 4);
});

test("the cap is never exceeded", () => {
  for (const cap of [1, 2, 4]) {
    for (const rows of [8, 68, 512, 100000]) {
      const tile = derivedTokenTile({ measuredWidth: 2048, rows, cap });
      assert.ok(tile <= cap && tile >= 1, `${cap} ${rows} ${tile}`);
    }
  }
});

// ---------------------------------------------------------------------------

import { outputRowTileFor } from "../src/af3/atom-encoder-webgpu.js";

// 🔴 THE CLAMP, AS A PROPERTY. The measurement may only make the tile smaller,
// because "this device is left idle by the shipped 256" is the only thing the
// mechanism argues. A target below 256 would pick a BIGGER tile than ships
// today on hardware nobody has measured.
test("a measured target never picks a larger tile than the shipped one", () => {
  for (const rows of [64, 400, 1632, 8000, 40000]) {
    const shipped = outputRowTileFor(rows);
    for (const measured of [16, 64, 256, 2048, 40000]) {
      const derived = outputRowTileFor(rows, Math.max(256, measured));
      assert.ok(derived <= shipped,
        `rows ${rows}, width ${measured}: ${derived} > ${shipped}`);
    }
  }
});

test("a wide device does get the smaller tile", () => {
  assert.equal(outputRowTileFor(1632), 4);
  assert.equal(outputRowTileFor(1632, Math.max(256, 2048)), 1);
});
