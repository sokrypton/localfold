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

import { distanceChange, expectedDistances, relativeChange, shouldStopRecycling }
  from "../src/model/feature-convergence.js";

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

describe("expectedDistances", () => {
  // Two bins from one break: centres are the break itself at both ends.
  const breaks = new Float32Array([10]);

  it("reads the centre of the bin all the mass is in", () => {
    const certainlyNear = new Float32Array([50, 0]);
    const certainlyFar = new Float32Array([0, 50]);
    assert.ok(Math.abs(expectedDistances(certainlyNear, breaks)[0] - 10) < 1e-3);
    assert.ok(Math.abs(expectedDistances(certainlyFar, breaks)[0] - 10) < 1e-3);
  });

  it("is a real expectation over the bins, not an argmax", () => {
    // Three bins from breaks [6, 12], so centres are 6, 9 and 12 - the open
    // first and last take the break itself. Uniform logits give their mean.
    const out = expectedDistances(new Float32Array([0, 0, 0]), new Float32Array([6, 12]));
    assert.ok(Math.abs(out[0] - 9) < 1e-4, `expected 9, got ${out[0]}`);
  });

  it("refuses logits that are not a whole number of rows", () => {
    assert.throws(() => expectedDistances(new Float32Array(3), breaks), RangeError);
  });
});

describe("distanceChange", () => {
  it("is zero for a matrix that did not move, and RMS otherwise", () => {
    const a = new Float32Array([1, 2, 3, 4]);
    assert.equal(distanceChange(a, new Float32Array(a)), 0);
    assert.equal(distanceChange(new Float32Array([0, 0]), new Float32Array([3, 4])),
                 Math.sqrt((9 + 16) / 2));
  });

  it("refuses mismatched maps", () => {
    assert.throws(() => distanceChange(new Float32Array(1), new Float32Array(2)), RangeError);
  });
});

describe("shouldStopRecycling", () => {
  const settled = { distanceAngstroms: 0.1 };

  it("never stops at pass 0", () => {
    assert.equal(shouldStopRecycling(0, settled, 0.5), false);
  });

  it("never stops when the tolerance is zero, which is the default", () => {
    assert.equal(shouldStopRecycling(3, settled, 0), false);
  });

  it("stops when the predicted distances have settled below the tolerance", () => {
    assert.equal(shouldStopRecycling(1, settled, 0.5), true);
    assert.equal(shouldStopRecycling(1, { distanceAngstroms: 0.9 }, 0.5), false);
  });

  it("🔴 treats an ABSENT distogram as not converged, never as zero", () => {
    // The first pass carries no distanceAngstroms, and neither would a model
    // whose head this fold did not run. Reading that as 0 would stop instantly.
    assert.equal(shouldStopRecycling(1, {}, 0.5), false);
    assert.equal(shouldStopRecycling(1, { pair: 0, single: 0 }, 0.5), false);
    assert.equal(shouldStopRecycling(1, undefined, 0.5), false);
  });

  it("refuses a tolerance or a change it cannot compare", () => {
    assert.throws(() => shouldStopRecycling(1, settled, -1), RangeError);
    assert.throws(() => shouldStopRecycling(1, { distanceAngstroms: Number.NaN }, 0.5), RangeError);
    assert.throws(() => shouldStopRecycling(-1, settled, 0.5), RangeError);
  });
});
