/**
 * The convergence criterion a trunk-only recycle can use.
 *
 * 🔴 IT EXISTS BECAUSE AF2's DOES NOT TRANSFER. `recycleConvergenceDistance`
 * compares C-alpha positions, and AF3, OpenDDE and ESMFold2 produce none until
 * the sampler runs once at the end - so for them the single and pair
 * representations are the only signal. See src/model/feature-convergence.js.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { relativeChange, shouldStopRecycling } from "../src/model/feature-convergence.js";

describe("relativeChange", () => {
  it("is zero for a representation that did not move", () => {
    const a = new Float32Array([1, -2, 3, 4]);
    assert.equal(relativeChange(a, new Float32Array(a)), 0);
  });

  it("is one against the zero seed a first pass compares with", () => {
    // 🔴 THIS IS WHY `shouldStopRecycling` REFUSES PASS 0. The loop seeds
    // `previous` with zeros, so pass 0 always reads exactly 1 - and a tolerance
    // above 1 would otherwise skip the only trunk pass a fold has.
    const current = new Float32Array([3, -4]);
    assert.equal(relativeChange(new Float32Array(2), current), 1);
  });

  it("is the change over the magnitude, not the change alone", () => {
    // ||b-a|| = 1, ||b|| = 5, so a tenth the scale is ten times the reading.
    assert.ok(Math.abs(relativeChange(new Float32Array([3, 4]),
                                      new Float32Array([3, 5])) - 1 / Math.sqrt(34)) < 1e-6);
    const small = relativeChange(new Float32Array([0.3, 0.4]), new Float32Array([0.3, 0.5]));
    const large = relativeChange(new Float32Array([30, 40]), new Float32Array([30, 50]));
    assert.ok(Math.abs(small - large) < 1e-6, "scale-free: the same relative move reads the same");
  });

  it("returns zero rather than dividing by an all-zero current", () => {
    assert.equal(relativeChange(new Float32Array([1, 2]), new Float32Array(2)), 0);
  });

  it("refuses mismatched lengths and non-tensors", () => {
    assert.throws(() => relativeChange(new Float32Array(2), new Float32Array(3)), RangeError);
    assert.throws(() => relativeChange([1, 2], new Float32Array(2)), TypeError);
  });
});

describe("shouldStopRecycling", () => {
  const settled = { pair: 1e-3, single: 1e-3 };

  it("never stops at pass 0, however settled it looks", () => {
    assert.equal(shouldStopRecycling(0, settled, 0.02), false);
  });

  it("never stops when the tolerance is zero, which is the default", () => {
    assert.equal(shouldStopRecycling(3, settled, 0), false);
  });

  it("stops when both tensors are under the tolerance", () => {
    assert.equal(shouldStopRecycling(1, settled, 0.02), true);
  });

  it("needs BOTH: either one still moving means the trunk has not settled", () => {
    assert.equal(shouldStopRecycling(1, { pair: 0.5, single: 1e-3 }, 0.02), false);
    assert.equal(shouldStopRecycling(1, { pair: 1e-3, single: 0.5 }, 0.02), false);
  });

  it("refuses a delta or tolerance it cannot compare", () => {
    assert.throws(() => shouldStopRecycling(1, settled, -1), RangeError);
    assert.throws(() => shouldStopRecycling(1, settled, Number.NaN), RangeError);
    assert.throws(() => shouldStopRecycling(1, { pair: Number.NaN, single: 0 }, 0.02), RangeError);
    assert.throws(() => shouldStopRecycling(1, undefined, 0.02), RangeError);
    assert.throws(() => shouldStopRecycling(-1, settled, 0.02), RangeError);
  });
});
