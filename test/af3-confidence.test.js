/**
 * The confidence head's binning, which is where these heads go wrong quietly.
 *
 * tools/oracle/check_af3_confidence.js checks the whole head against AF3. What
 * these add is the arithmetic that has a right answer without any weights: an
 * off-by-one in a bin centre shifts every reported error by half a bin and
 * still produces numbers in the range anyone would expect.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { distogramFeatures } from "../src/af3/confidence-reference.js";

describe("AF3 confidence distogram", () => {
  const tokens = 3;
  const open = Float32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 1]);

  it("gives a pair CLOSER than the first edge no bin at all", () => {
    // 🔴 THIS IS AF3'S ACTUAL BEHAVIOUR AND IT LOOKS LIKE A BUG. The test is
    // `dist^2 > lower_break`, strictly, so a pair inside 3.25 A satisfies no
    // bin and its row is all zeros - indistinguishable from a masked pair. The
    // diagonal is always in this case, since a token's distance to itself is 0.
    // Any "fix" that clamps it into bin 0 changes what the head is fed.
    const positions = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 0, 0]);
    const dgram = distogramFeatures(positions, open, tokens);
    for (let bin = 0; bin < 39; bin += 1) {
      assert.equal(dgram[1 * 39 + bin], 0, `pair 1 A apart set bin ${bin}`);
      assert.equal(dgram[0 * 39 + bin], 0, `the diagonal set bin ${bin}`);
    }
  });

  it("puts a pair beyond the last edge in the catch-all bin", () => {
    // 🔴 THE FINAL BIN'S UPPER EDGE IS 1e8, NOT max_bin. Without it every pair
    // further apart than 50.75 A would land in NO bin and contribute a row of
    // zeros - which looks like a mask rather than a distance.
    const positions = Float32Array.from([0, 0, 0, 500, 0, 0, 0, 0, 0]);
    const dgram = distogramFeatures(positions, open, tokens);
    const bins = dgram.subarray(1 * 39, 2 * 39);
    assert.equal(bins[38], 1, "should be in the catch-all bin");
    let total = 0;
    for (const value of bins) total += value;
    assert.equal(total, 1, "exactly one bin should be set");
  });

  it("sets exactly one bin for every pair beyond the first edge", () => {
    const positions = Float32Array.from([0, 0, 0, 10, 0, 0, 0, 30, 0]);
    const dgram = distogramFeatures(positions, open, tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        if (i === j) continue;               // the diagonal is the zero case above
        let total = 0;
        for (let bin = 0; bin < 39; bin += 1) total += dgram[(i * tokens + j) * 39 + bin];
        assert.equal(total, 1, `pair ${i},${j} set ${total} bins`);
      }
    }
  });

  it("zeroes a masked pair entirely", () => {
    const positions = Float32Array.from([0, 0, 0, 10, 0, 0, 0, 30, 0]);
    const mask = Float32Array.from([1, 0, 1, 0, 0, 0, 1, 0, 1]);
    const dgram = distogramFeatures(positions, mask, tokens);
    for (let bin = 0; bin < 39; bin += 1) {
      assert.equal(dgram[1 * 39 + bin], 0, "pair (0,1) is masked out");
    }
    // ...and an unmasked one is untouched.
    let total = 0;
    for (let bin = 0; bin < 39; bin += 1) total += dgram[2 * 39 + bin];
    assert.equal(total, 1);
  });

  it("is symmetric, since it is built from distances", () => {
    const positions = Float32Array.from([0, 0, 0, 10, 0, 0, 0, 30, 0]);
    const dgram = distogramFeatures(positions, open, tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        for (let bin = 0; bin < 39; bin += 1) {
          assert.equal(dgram[(i * tokens + j) * 39 + bin],
                       dgram[(j * tokens + i) * 39 + bin]);
        }
      }
    }
  });
});
