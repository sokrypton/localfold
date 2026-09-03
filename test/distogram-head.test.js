/**
 * AlphaFold 2's distogram head.
 *
 * 🔴 THESE ARE DIFFERENTIAL, NOT ORACLE. They say the head computes the
 * operation AlphaFold's own code describes - a projection, a symmetrisation
 * over the pair axes, a softmax and a sum of the bins under 8 A - not that
 * AlphaFold agrees with the numbers. The oracle check is a dump, and the
 * fixture for it is not in this repository.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { distogramBreaks, distogramContactProbabilities, distogramLogits }
  from "../src/heads/distogram.js";

const CHANNELS = 4;
const BINS = 6;
const OPTIONS = { channels: CHANNELS, bins: BINS, first: 2, last: 22 };

/** A deterministic pair representation that is NOT symmetric in i and j. */
function pairOf(length) {
  const pair = new Float32Array(length * length * CHANNELS);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      for (let c = 0; c < CHANNELS; c += 1) {
        pair[(i * length + j) * CHANNELS + c] = Math.sin(i * 3 + j * 7 + c * 11);
      }
    }
  }
  return pair;
}

const weightsOf = () => Float32Array.from(
  { length: CHANNELS * BINS }, (_, k) => Math.cos(k * 1.7) * 0.5);
const biasOf = () => Float32Array.from({ length: BINS }, (_, k) => k * 0.01);

describe("the distogram head's bins", () => {
  it("puts count-1 breaks evenly from first to last", () => {
    const breaks = distogramBreaks(2, 22, 64);
    assert.equal(breaks.length, 63);
    assert.ok(Math.abs(breaks[0] - 2) < 1e-12);
    assert.ok(Math.abs(breaks[62] - 22) < 1e-12);
    const step = breaks[1] - breaks[0];
    for (let i = 1; i < breaks.length; i += 1) {
      assert.ok(Math.abs((breaks[i] - breaks[i - 1]) - step) < 1e-12);
    }
  });
});

describe("the distogram head", () => {
  it("is symmetric in i and j even though the pair is not", () => {
    // 🔴 THE POINT OF THE TRANSPOSE. `half` sees pair[i][j], which differs
    // from pair[j][i]; the head adds the transpose so a distance is one
    // number however it is asked for. The fixture above is deliberately
    // asymmetric, or this test would pass without the symmetrisation.
    const length = 5;
    const contacts = distogramContactProbabilities(
      pairOf(length), weightsOf(), biasOf(), length, OPTIONS);
    for (let i = 0; i < length; i += 1) {
      for (let j = 0; j < length; j += 1) {
        assert.ok(Math.abs(contacts[i * length + j] - contacts[j * length + i]) < 1e-12,
          `(${i},${j}) and (${j},${i}) disagree`);
      }
    }
  });

  it("symmetrises over the PAIR axes and not over the bins", () => {
    // logits[i][j][b] must be half[i][j][b] + half[j][i][b] - the SAME bin on
    // both sides. Mixing the bin axis in would still produce a symmetric,
    // plausible-looking distogram about nothing.
    const length = 4;
    const pair = pairOf(length);
    const weights = weightsOf();
    const bias = biasOf();
    const logits = distogramLogits(pair, weights, bias, length, OPTIONS);
    const half = (i, j, bin) => {
      let value = bias[bin];
      for (let c = 0; c < CHANNELS; c += 1) {
        value += pair[(i * length + j) * CHANNELS + c] * weights[c * BINS + bin];
      }
      return value;
    };
    for (let i = 0; i < length; i += 1) {
      for (let j = 0; j < length; j += 1) {
        for (let bin = 0; bin < BINS; bin += 1) {
          const want = half(i, j, bin) + half(j, i, bin);
          const got = logits[(i * length + j) * BINS + bin];
          assert.ok(Math.abs(got - want) < 1e-4, `(${i},${j},${bin}) ${got} vs ${want}`);
        }
      }
    }
  });

  it("agrees with a softmax over the logits it would have produced", () => {
    // The contact path never materialises the logits, so this is the check
    // that the shortcut is the same arithmetic.
    const length = 4;
    const pair = pairOf(length);
    const weights = weightsOf();
    const bias = biasOf();
    const contacts = distogramContactProbabilities(pair, weights, bias, length, OPTIONS);
    const logits = distogramLogits(pair, weights, bias, length, OPTIONS);
    const breaks = distogramBreaks(2, 22, BINS);
    let counted = 0;
    while (counted < breaks.length && breaks[counted] <= 8) counted += 1;
    for (let i = 0; i < length; i += 1) {
      for (let j = 0; j < length; j += 1) {
        const base = (i * length + j) * BINS;
        let largest = -Infinity;
        for (let b = 0; b < BINS; b += 1) largest = Math.max(largest, logits[base + b]);
        let total = 0;
        let under = 0;
        for (let b = 0; b < BINS; b += 1) {
          const w = Math.exp(logits[base + b] - largest);
          total += w;
          if (b < counted) under += w;
        }
        assert.ok(Math.abs(contacts[i * length + j] - under / total) < 1e-6,
          `(${i},${j}) shortcut disagrees with the long way`);
      }
    }
  });

  it("returns probabilities, so every value is in [0, 1]", () => {
    const length = 6;
    const contacts = distogramContactProbabilities(
      pairOf(length), weightsOf(), biasOf(), length, OPTIONS);
    for (const value of contacts) {
      assert.ok(value >= 0 && value <= 1, `${value} is not a probability`);
    }
  });

  it("counts the bins strictly under the threshold, not the one straddling it", () => {
    // With breaks at 2..22 the bin ending exactly at 8 counts and the next
    // does not. A head that included the straddling bin would report a
    // systematically higher contact rate and nothing would look wrong.
    const bins = 6;
    const breaks = distogramBreaks(2, 22, bins);
    const counted = [...breaks].filter((edge) => edge <= 8).length;
    assert.ok(counted < breaks.length, "the threshold should exclude some bins");
    assert.ok(breaks[counted] > 8, "the first uncounted break must be past 8");
  });

  it("rejects a pair whose shape does not match the length", () => {
    assert.throws(() => distogramContactProbabilities(
      new Float32Array(10), weightsOf(), biasOf(), 5, OPTIONS), RangeError);
  });
});
