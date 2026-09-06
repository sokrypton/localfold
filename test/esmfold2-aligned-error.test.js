/**
 * The estimated aligned error - EF2-fast's stand-in for a PAE.
 *
 * 🔴 THESE ARE PROPERTY TESTS, NOT AN ORACLE. What says the estimator agrees
 * with a real PAE is tools/pae-transfer.py, which scores it against AlphaFold 3
 * on folds of the same sequences. What is pinned here is the arithmetic that
 * would silently produce a plausible matrix if it were wrong: the moments'
 * UNITS, the feature order, and the fact that the distogram reaches the answer
 * at all.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  alignedErrorFromDistogram, distogramMoments, FEATURES, WEIGHTS, MAX_ANGSTROMS,
} from "../src/esmfold2/aligned-error.js";

const EDGES = { minimum: 2, maximum: 52 };
const BINS = 128;

/** A distogram whose every pair is a Gaussian bump at `at` angstroms. */
function bumpAt(tokens, at, spread) {
  const width = (EDGES.maximum - EDGES.minimum) / BINS;
  const logits = new Float32Array(tokens * tokens * BINS);
  for (let pair = 0; pair < tokens * tokens; pair += 1) {
    for (let b = 0; b < BINS; b += 1) {
      const centre = EDGES.minimum + width * (b + 0.5);
      logits[pair * BINS + b] = -((centre - at) ** 2) / (2 * spread * spread);
    }
  }
  return logits;
}

describe("the distogram's moments", () => {
  it("reports the mean and spread in ANGSTROMS, not in bins", () => {
    // 🔴 THE UNIT IS THE WHOLE POINT. A spread read in bin indices differs by a
    // factor of two between AF3's 64-bin 2-22 grid and this 128-bin 2-52 one
    // for the same physical uncertainty, so one fitted estimator could not
    // serve both. A 3 A bump at 20 A must come back as 20 and 3.
    const bias = new Float32Array(BINS);
    const { mean, sigma } = distogramMoments(bumpAt(2, 20, 3), bias, 2, BINS, EDGES);
    assert.ok(Math.abs(mean[0] - 20) < 0.05, `mean ${mean[0]}`);
    assert.ok(Math.abs(sigma[0] - 3) < 0.05, `sigma ${sigma[0]}`);
  });

  it("adds the bias, which is not in the logits buffer", () => {
    // The projection has no bias and the contact pass adds it as it reads, so a
    // caller that softmaxes the logits alone gets a different distribution -
    // one that still sums to one, which is why this needs asserting.
    const flat = new Float32Array(BINS);
    const bias = new Float32Array(BINS);
    bias[100] = 40;                       // one bin made overwhelming
    const width = (EDGES.maximum - EDGES.minimum) / BINS;
    const { mean } = distogramMoments(flat, bias, 1, BINS, EDGES);
    assert.ok(Math.abs(mean[0] - (EDGES.minimum + width * 100.5)) < 0.1,
      `bias ignored: mean ${mean[0]}`);
  });

  it("widens as the distribution does", () => {
    const bias = new Float32Array(BINS);
    const narrow = distogramMoments(bumpAt(1, 20, 1), bias, 1, BINS, EDGES);
    const wide = distogramMoments(bumpAt(1, 20, 6), bias, 1, BINS, EDGES);
    assert.ok(wide.sigma[0] > narrow.sigma[0] * 4);
    assert.ok(wide.effectiveWidth[0] > narrow.effectiveWidth[0] * 4);
  });
});

describe("the aligned-error estimate", () => {
  const tokens = 6;
  const bias = new Float32Array(BINS);
  const distances = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) distances[i * tokens + j] = Math.abs(i - j) * 4;
  }

  it("has one weight per feature, and the two lists cannot drift", () => {
    assert.equal(FEATURES.length, WEIGHTS.length);
  });

  it("is zero on the diagonal, because a token has none against itself", () => {
    const m = distogramMoments(bumpAt(tokens, 10, 2), bias, tokens, BINS, EDGES);
    const pae = alignedErrorFromDistogram(m, distances, tokens);
    for (let i = 0; i < tokens; i += 1) assert.equal(pae[i * tokens + i], 0);
  });

  it("stays inside the range a PAE is reported over", () => {
    const m = distogramMoments(bumpAt(tokens, 40, 20), bias, tokens, BINS, EDGES);
    const pae = alignedErrorFromDistogram(m, distances, tokens);
    for (const v of pae) {
      assert.ok(v >= 0 && v <= MAX_ANGSTROMS, `${v} outside 0..${MAX_ANGSTROMS}`);
    }
  });

  it("rises for the ONE pair whose distogram widened", () => {
    // 🔴 THE CONTROL THAT SAYS THE DISTOGRAM REACHES THE ANSWER. Same
    // coordinates, same everything else - one pair's spread moves. An estimator
    // that had lost its distogram would return the identical matrix and still
    // look like a plausible PAE.
    //
    // 🔴 AND IT IS ONE PAIR, NOT ONE TOKEN'S WHOLE ROW. Widening every pair a
    // token takes part in raises that token's own MOBILITY baseline, which the
    // estimate normalises against - so the row can stay put or fall, correctly.
    // The quantity is a contrast; the test has to make one.
    const bins = BINS;
    const width = (EDGES.maximum - EDGES.minimum) / bins;
    const build = (spreadFor) => {
      const logits = new Float32Array(tokens * tokens * bins);
      for (let i = 0; i < tokens; i += 1) {
        for (let j = 0; j < tokens; j += 1) {
          const spread = spreadFor(i, j);
          for (let b = 0; b < bins; b += 1) {
            const centre = EDGES.minimum + width * (b + 0.5);
            logits[(i * tokens + j) * bins + b] =
              -((centre - 10) ** 2) / (2 * spread * spread);
          }
        }
      }
      return logits;
    };
    const flat = alignedErrorFromDistogram(
      distogramMoments(build(() => 1), bias, tokens, bins, EDGES), distances, tokens);
    const one = alignedErrorFromDistogram(
      distogramMoments(build((i, j) => ((i === 0 && j === 1) || (i === 1 && j === 0)
                                        ? 6 : 1)), bias, tokens, bins, EDGES),
      distances, tokens);
    assert.ok(one[1] > flat[1] + 0.5,
      `the widened pair did not rise: ${flat[1]} -> ${one[1]}`);
    // ...and a pair that did not widen barely moves.
    assert.ok(Math.abs(one[2 * tokens + 3] - flat[2 * tokens + 3]) < 0.5,
      `an untouched pair moved: ${flat[2 * tokens + 3]} -> ${one[2 * tokens + 3]}`);
  });

  it("is scale-invariant in sigma, which is deliberate", () => {
    // 🔴 MULTIPLYING EVERY SPREAD BY A CONSTANT MUST NOT MOVE IT. sigma and the
    // mobility terms are correlated - the mobility is a mean of sigma - so an
    // unconstrained fit gave them large opposite signs and the estimate went
    // DOWN by 0.513 A per angstrom of uniform spread. The shipped fit pins that
    // aggregate direction to zero; see the note in aligned-error.js. This is
    // the assertion that the constraint is still in the coefficients.
    const wide = alignedErrorFromDistogram(
      distogramMoments(bumpAt(tokens, 10, 4), bias, tokens, BINS, EDGES),
      distances, tokens);
    const narrow = alignedErrorFromDistogram(
      distogramMoments(bumpAt(tokens, 10, 1), bias, tokens, BINS, EDGES),
      distances, tokens);
    for (let i = 0; i < tokens * tokens; i += 1) {
      assert.ok(Math.abs(wide[i] - narrow[i]) < 0.35,
        `pair ${i} moved ${narrow[i]} -> ${wide[i]} on a uniform change`);
    }
  });

  it("refuses a distance matrix that is not the distogram's shape", () => {
    const m = distogramMoments(bumpAt(tokens, 10, 2), bias, tokens, BINS, EDGES);
    assert.throws(() => alignedErrorFromDistogram(m, new Float32Array(4), tokens),
      /expected 36 pairs/);
  });
});
