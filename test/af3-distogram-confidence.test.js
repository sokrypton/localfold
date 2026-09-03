/**
 * The distogram stand-in for pLDDT, on distograms whose answer is known.
 *
 * The correlation against real pLDDT is measured on real folds by
 * tools/gpu/probe-distogram-confidence.js and cannot be asserted here. What
 * CAN be asserted is the behaviour that makes it usable as a colour: a frame
 * that matches a confident prediction scores high, one that does not scores
 * low, and a prediction that is not confident cannot score high whatever the
 * frame does.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { distogramAgreementTable, distogramConfidence, distogramInterfaceContact }
  from "../src/af3/distogram-confidence.js";

const BINS = 64;
const FIRST = 2.3125;
const LAST = 21.6875;

function edges() {
  const breaks = new Float32Array(BINS - 1);
  for (let index = 0; index < BINS - 1; index += 1) {
    breaks[index] = FIRST + (LAST - FIRST) * index / (BINS - 2);
  }
  return breaks;
}

const spacing = (LAST - FIRST) / (BINS - 2);
const centreOf = (bin) => (bin === 0 ? FIRST - spacing / 2
  : bin === BINS - 1 ? LAST + spacing / 2
    : (edges()[bin - 1] + edges()[bin]) / 2);
const binNear = (distance) =>
  Math.min(BINS - 1, Math.max(0, Math.floor((distance - FIRST) / spacing) + 1));

/**
 * Logits for a chain laid out on a line at `spacingAngstroms`, with every pair
 * predicted `sharpness` peaked at its true distance.
 */
function lineChain(tokens, spacingAngstroms, sharpness) {
  const logits = new Float32Array(tokens * tokens * BINS);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const truth = Math.abs(i - j) * spacingAngstroms;
      const base = (i * tokens + j) * BINS;
      for (let b = 0; b < BINS; b += 1) {
        logits[base + b] = -sharpness * (centreOf(b) - truth) ** 2;
      }
    }
  }
  return logits;
}

/** Those same tokens, actually on that line. */
function linePositions(tokens, spacingAngstroms) {
  const beta = new Float32Array(tokens * 3);
  for (let i = 0; i < tokens; i += 1) beta[i * 3] = i * spacingAngstroms;
  return beta;
}

describe("distogram confidence", () => {
  const tokens = 24;
  const mask = new Float32Array(tokens).fill(1);

  it("scores a frame that matches a confident prediction near 100", () => {
    const table = distogramAgreementTable(lineChain(tokens, 1.2, 4), edges(), tokens, mask);
    const scores = distogramConfidence(table, linePositions(tokens, 1.2));
    for (let i = 0; i < tokens; i += 1) assert.ok(scores[i] > 90, `token ${i} scored ${scores[i]}`);
  });

  it("scores a frame that does not match it near zero", () => {
    const table = distogramAgreementTable(lineChain(tokens, 1.2, 4), edges(), tokens, mask);
    // ...the same chain stretched, so every long-range distance is wrong.
    const scores = distogramConfidence(table, linePositions(tokens, 3.0));
    for (let i = 0; i < tokens; i += 1) assert.ok(scores[i] < 25, `token ${i} scored ${scores[i]}`);
  });

  it("cannot score high on a prediction that is not confident", () => {
    // 🔴 THE POINT OF THE WHOLE THING. A diffuse distogram spreads its mass
    // over many bins, so even a perfectly matching frame collects only the
    // little that lies inside the tolerance - which is what makes this track
    // the trunk's uncertainty and not just the geometry.
    const table = distogramAgreementTable(lineChain(tokens, 1.2, 0.004), edges(), tokens, mask);
    const scores = distogramConfidence(table, linePositions(tokens, 1.2));
    for (let i = 0; i < tokens; i += 1) assert.ok(scores[i] < 40, `token ${i} scored ${scores[i]}`);
  });

  it("scores every token of a masked-out chain at zero", () => {
    const table = distogramAgreementTable(
      lineChain(tokens, 1.2, 4), edges(), tokens, new Float32Array(tokens));
    const scores = distogramConfidence(table, linePositions(tokens, 1.2));
    for (let i = 0; i < tokens; i += 1) assert.equal(scores[i], 0);
  });

  it("still scores a chain too short to have long-range contacts", () => {
    // 🔴 SEQUENCE SEPARATION SIX IS NOT AVAILABLE ON A FOUR-MER, and returning
    // zero there would colour a small ligand or peptide as failed rather than
    // as small. The separation relaxes rather than the score collapsing.
    const short = 4;
    const shortMask = new Float32Array(short).fill(1);
    const table = distogramAgreementTable(
      lineChain(short, 1.2, 4), edges(), short, shortMask);
    const scores = distogramConfidence(table, linePositions(short, 1.2));
    for (let i = 0; i < short; i += 1) assert.ok(scores[i] > 90, `token ${i} scored ${scores[i]}`);
  });

  it("rejects a distogram whose shape does not match the token count", () => {
    assert.throws(() => distogramAgreementTable(
      new Float32Array(10), edges(), tokens, mask), /expected/);
    const table = distogramAgreementTable(lineChain(tokens, 1.2, 4), edges(), tokens, mask);
    assert.throws(() => distogramConfidence(table, new Float32Array(7)), /expected/);
  });

  it("puts a wrongly placed token below its correctly placed neighbours", () => {
    // The ranking is what a colour ramp is, so it is asserted directly.
    const table = distogramAgreementTable(lineChain(tokens, 1.2, 4), edges(), tokens, mask);
    const beta = linePositions(tokens, 1.2);
    const moved = 11;
    beta[moved * 3 + 1] = 9;
    const scores = distogramConfidence(table, beta);
    const others = [...scores.keys()].filter((i) => i !== moved);
    const best = Math.max(...others.map((i) => scores[i]));
    assert.ok(scores[moved] < best, `moved token scored ${scores[moved]} against ${best}`);
  });
});

describe("distogram interface contact", () => {
  const tokens = 8;
  const mask = new Float32Array(tokens).fill(1);
  // Two chains of four.
  const asymId = Int32Array.from([0, 0, 0, 0, 1, 1, 1, 1]);
  const probs = (crossValue, intraValue) => {
    const out = new Float32Array(tokens * tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        out[i * tokens + j] = asymId[i] === asymId[j] ? intraValue : crossValue;
      }
    }
    return out;
  };

  it("reads the cross-chain contacts and ignores the intra-chain ones", () => {
    // 🔴 THE INTRA VALUE IS THE OPPOSITE OF THE CROSS ONE ON PURPOSE. A score
    // that quietly averaged both would land in the middle and look plausible.
    const score = distogramInterfaceContact(probs(0.9, 0.1), asymId, mask, tokens);
    assert.ok(Math.abs(score - 0.9) < 1e-6, `scored ${score}`);
  });

  it("is NaN for a single chain, as ipTM itself is", () => {
    const one = new Int32Array(tokens);
    assert.ok(Number.isNaN(distogramInterfaceContact(probs(0.9, 0.9), one, mask, tokens)));
  });

  it("takes the STRONGEST contacts, not the mean of all of them", () => {
    // One confident interface pair among many empty ones: a mean over
    // everything buries it, a mean over the strongest few does not.
    const out = new Float32Array(tokens * tokens);
    out[0 * tokens + 4] = 1;
    out[4 * tokens + 0] = 1;
    const strongest = distogramInterfaceContact(out, asymId, mask, tokens, 2);
    const everything = distogramInterfaceContact(out, asymId, mask, tokens, 1e9);
    assert.equal(strongest, 1);
    assert.ok(everything < 0.1, `mean over all was ${everything}`);
  });

  it("counts a fraction of the SMALLER chain, not a fixed number", () => {
    // 🔴 THE HYPOTHESIS THIS RULE EXISTS FOR. A short chain cannot present as
    // many interface contacts as a long one, so a fixed count reaches past the
    // real interface on the small ones - measured as a +0.26 correlation
    // between the estimate's error and the smaller chain's length. Half the
    // smaller chain leaves no size trend.
    //
    // Built so the answer SAYS how many were taken: the strongest `n` contacts
    // are 1 and the rest 0, so the score is (however many were taken, capped at
    // the number of ones) / (however many were taken).
    const build = (a, b, ones) => {
      const n = a + b;
      const asym = Int32Array.from({ length: n }, (_, i) => (i < a ? 0 : 1));
      const live = new Float32Array(n).fill(1);
      const probs = new Float32Array(n * n);
      let placed = 0;
      for (let i = 0; i < a && placed < ones; i += 1) {
        for (let j = a; j < n && placed < ones; j += 1) { probs[i * n + j] = 1; placed += 1; }
      }
      return { probs, asym, live, n };
    };
    // 20 by 90: half of 20 is 10 contacts. Twenty ones means all ten are 1.
    const small = build(20, 90, 20);
    assert.equal(distogramInterfaceContact(small.probs, small.asym, small.live, small.n), 1);
    // 90 by 90: half of 90 is 45. Twenty ones over 45 taken is well under 1,
    // which a fixed count of 10 would have reported as a perfect interface.
    const big = build(90, 90, 20);
    const score = distogramInterfaceContact(big.probs, big.asym, big.live, big.n);
    assert.ok(Math.abs(score - 20 / 45) < 1e-6, `scored ${score}`);
  });

  it("floors the count, so a tiny chain still averages something", () => {
    const n = 8;
    const asym = Int32Array.from([0, 0, 0, 0, 1, 1, 1, 1]);
    const live = new Float32Array(n).fill(1);
    const probs = new Float32Array(n * n).fill(1);
    // Half of four is two, below the floor of eight - it must not divide by two
    // and it must not divide by more than the pairs that exist.
    assert.equal(distogramInterfaceContact(probs, asym, live, n), 1);
  });

  it("ignores masked tokens on both sides of the interface", () => {
    const half = Float32Array.from([1, 1, 1, 1, 0, 0, 0, 0]);
    assert.ok(Number.isNaN(distogramInterfaceContact(probs(0.9, 0.1), asymId, half, tokens)));
  });
});
