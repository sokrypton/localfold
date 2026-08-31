/**
 * The AF3 MSA block's outer product mean, on cases the oracle cannot reach.
 *
 * 🔴 A ONE-SEQUENCE MSA HIDES MOST OF THIS OPERATION. The oracle runs at
 * num_msa=1 because a de-novo spec has no alignment to featurise, and at depth
 * one the sum over sequences has a single term and the mask is all ones - so
 * the two things most likely to be wrong, the accumulation and the
 * coverage-based denominator, are both untested by
 * tools/oracle/check_af3_msa_block.js however exact it comes back.
 *
 * These are hand-computed instead. `outputW` is set to a selector so the
 * projection is an identity on one channel pair, which turns the block into
 * arithmetic with a right answer independent of AF3.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { outerProductMean } from "../src/af3/msa-reference.js";

/**
 * An OPM whose LayerNorm and projections are pinned to the identity, so the
 * only thing under test is the sum over sequences and the denominator.
 *
 * With msaChannels 1 and outerChannels 1, LayerNorm of a single channel is
 * exactly its offset - so a unit offset makes left and right both equal to the
 * projection weight, independent of the input values.
 */
function pinnedWeights(leftWeight, rightWeight) {
  return {
    outerChannels: 1,
    layerNormInputScale: Float32Array.from([0]),
    layerNormInputOffset: Float32Array.from([1]),
    leftProjection: Float32Array.from([leftWeight]),
    rightProjection: Float32Array.from([rightWeight]),
    outputW: Float32Array.from([1]),
    outputB: Float32Array.from([0]),
  };
}

describe("AF3 outerProductMean", () => {
  it("sums over sequences before dividing", () => {
    // Two sequences, two tokens, everything covered. Each sequence contributes
    // left * right = 6, so the sum is 12 and the coverage is 2.
    const sequences = 2;
    const tokens = 2;
    const msa = new Float32Array(sequences * tokens);
    const mask = new Float32Array(sequences * tokens).fill(1);
    const output = outerProductMean(msa, mask, sequences, tokens, 1, 1,
                                    pinnedWeights(2, 3));
    for (let index = 0; index < output.length; index += 1) {
      assert.ok(Math.abs(output[index] - 12 / (1e-3 + 2)) < 1e-4,
                `[${index}] = ${output[index]}`);
    }
  });

  it("divides by the coverage of BOTH tokens, not the sequence count", () => {
    // 🔴 THE DENOMINATOR IS PER TOKEN PAIR. Sequence 1 covers only token 0, so
    // the pair (1,1) is seen by one sequence while (0,0) is seen by two -
    // dividing by `sequences` would scale them identically and agree with this
    // test on every MSA that happens to have no gaps.
    const sequences = 2;
    const tokens = 2;
    const msa = new Float32Array(sequences * tokens);
    const mask = Float32Array.from([1, 1, 1, 0]);   // sequence 1 lacks token 1
    const output = outerProductMean(msa, mask, sequences, tokens, 1, 1,
                                    pinnedWeights(2, 3));
    const at = (i, j) => output[i * tokens + j];
    assert.ok(Math.abs(at(0, 0) - 12 / (1e-3 + 2)) < 1e-4, `(0,0) = ${at(0, 0)}`);
    assert.ok(Math.abs(at(1, 1) - 6 / (1e-3 + 1)) < 1e-4, `(1,1) = ${at(1, 1)}`);
    // ...and the two cross terms lose the uncovered sequence on one side only.
    assert.ok(Math.abs(at(0, 1) - 6 / (1e-3 + 1)) < 1e-4, `(0,1) = ${at(0, 1)}`);
  });

  it("gives an uncovered token pair zero, not a division by zero", () => {
    const output = outerProductMean(new Float32Array(2), Float32Array.from([1, 0]),
                                    1, 2, 1, 1, pinnedWeights(2, 3));
    assert.equal(output[3], 0, "the uncovered pair should be exactly zero");
    assert.ok(Number.isFinite(output[3]));
  });

  it("is not symmetric: (i,j) takes left from i and right from j", () => {
    // 🔴 SCALING THE TWO PROJECTIONS DIFFERENTLY DOES NOT MAKE THIS ASYMMETRIC.
    // With one outer channel the entry is L[i] * R[j], so it equals L[j] * R[i]
    // whenever L is proportional to R - which it is for any two projections of
    // a one-channel input, however different their weights. The asymmetry needs
    // left and right to select DIFFERENT channels of a token-dependent input,
    // so this uses three channels and a LayerNorm that is actually doing work.
    const weights = {
      outerChannels: 1,
      layerNormInputScale: Float32Array.from([1, 1, 1]),
      layerNormInputOffset: Float32Array.from([0, 0, 0]),
      leftProjection: Float32Array.from([1, 0, 0]),     // channel 0
      rightProjection: Float32Array.from([0, 1, 0]),    // channel 1
      outputW: Float32Array.from([1]),
      outputB: Float32Array.from([0]),
    };
    // One sequence, two tokens, each a different one-hot: their normalised
    // channels 0 and 1 are swapped, so L and R are not proportional.
    const msa = Float32Array.from([1, 0, 0,
                                   0, 1, 0]);
    const output = outerProductMean(msa, Float32Array.from([1, 1]), 1, 2, 3, 1, weights);
    assert.notEqual(output[1], output[2]);
    // ...and specifically: token 0 normalises to (+2,-1,-1)/sqrt(2) and token 1
    // to (-1,+2,-1)/sqrt(2), so (0,1) is (2/sqrt2)(2/sqrt2) = 2 and (1,0) is
    // (-1/sqrt2)(-1/sqrt2) = 0.5, before the 1e-3 in the denominator.
    assert.ok(Math.abs(output[1] - 2 / 1.001) < 1e-3, `(0,1) = ${output[1]}`);
    assert.ok(Math.abs(output[2] - 0.5 / 1.001) < 1e-3, `(1,0) = ${output[2]}`);
  });
});
