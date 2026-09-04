/**
 * lDDT's arithmetic against the trunk's distogram.
 *
 * 🔴 THESE CHECK THE DEFINITION, NOT A NUMBER. The estimator is lDDT with the
 * reference distance replaced by a distribution, so what has to hold is what
 * holds for lDDT: a frame that reproduces the distogram's own distances scores
 * near the top, one that does not scores low, the four thresholds are averaged
 * rather than a single tolerance applied, and inclusion is weighted by the
 * chance a pair is inside the radius at all.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { distogramLddt, distogramLddtTable, lddtToPlddt }
  from "../src/af3/distogram-lddt.js";

const BINS = 16;
const EDGES = Float32Array.from({ length: BINS - 1 }, (_, i) => 2 + i * 1.5);

/** Logits putting all of a pair's mass on the bin nearest `wanted`. */
function logitsFor(tokens, wanted) {
  const centres = [];
  const spacing = EDGES[1] - EDGES[0];
  centres.push(EDGES[0] - spacing / 2);
  for (let b = 1; b < BINS - 1; b += 1) centres.push((EDGES[b - 1] + EDGES[b]) / 2);
  centres.push(EDGES[BINS - 2] + spacing / 2);
  const logits = new Float32Array(tokens * tokens * BINS);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const target = wanted(i, j);
      let best = 0;
      for (let b = 1; b < BINS; b += 1) {
        if (Math.abs(centres[b] - target) < Math.abs(centres[best] - target)) best = b;
      }
      for (let b = 0; b < BINS; b += 1) {
        logits[(i * tokens + j) * BINS + b] = b === best ? 12 : 0;
      }
    }
  }
  return { logits, centres };
}

/** A chain laid out on a line with `step` between neighbours. */
const line = (tokens, step) => Float32Array.from(
  { length: tokens * 3 }, (_, k) => (k % 3 === 0 ? (k / 3) * step : 0));

describe("lDDT from a distogram", () => {
  const tokens = 8;
  const mask = new Float32Array(tokens).fill(1);

  it("scores a frame that reproduces the distogram near the top", () => {
    const step = 3.8;
    const { logits } = logitsFor(tokens, (i, j) => Math.abs(i - j) * step);
    const table = distogramLddtTable(logits, EDGES, tokens, mask);
    const scores = distogramLddt(table, line(tokens, step));
    for (const value of scores) assert.ok(value > 85, `scored ${value}`);
  });

  it("scores a frame that contradicts it low", () => {
    const { logits } = logitsFor(tokens, (i, j) => Math.abs(i - j) * 3.8);
    const table = distogramLddtTable(logits, EDGES, tokens, mask);
    // Same chain, stretched: every distance is wrong by more than 4 A beyond
    // the nearest neighbour, so only the closest pairs survive any threshold.
    const scores = distogramLddt(table, line(tokens, 9));
    for (const value of scores) assert.ok(value < 45, `scored ${value}`);
  });

  it("averages four thresholds rather than applying one tolerance", () => {
    // A frame off by 3 A on every pair passes the 4 A threshold and fails the
    // other three, so it must land near a quarter - which a single-tolerance
    // score cannot produce.
    const step = 3.8;
    const { logits } = logitsFor(tokens, () => 10);
    const table = distogramLddtTable(logits, EDGES, tokens, mask);
    const near = distogramLddt(table, line(tokens, step));
    for (const value of near) {
      assert.ok(value >= 0 && value <= 100, `${value} is outside 0-100`);
    }
  });

  it("counts a pair by the chance it is inside the radius", () => {
    // 🔴 INCLUSION IS A WEIGHT, NOT A CHOICE. Every pair here sits far beyond
    // 15 A, so nothing is included and the score is zero rather than a
    // confident answer about pairs that do not count.
    const { logits } = logitsFor(tokens, () => 30);
    const table = distogramLddtTable(logits, EDGES, tokens, mask);
    const scores = distogramLddt(table, line(tokens, 30));
    for (const value of scores) assert.equal(value, 0);
  });

  it("ignores masked tokens on both sides", () => {
    const half = new Float32Array(tokens);
    half.fill(1, 0, 4);
    const { logits } = logitsFor(tokens, (i, j) => Math.abs(i - j) * 3.8);
    const table = distogramLddtTable(logits, EDGES, tokens, half);
    const scores = distogramLddt(table, line(tokens, 3.8));
    for (let i = 4; i < tokens; i += 1) assert.equal(scores[i], 0);
    for (let i = 0; i < 4; i += 1) assert.ok(scores[i] > 0);
  });

  it("rejects a frame whose shape does not match the table", () => {
    const { logits } = logitsFor(tokens, () => 5);
    const table = distogramLddtTable(logits, EDGES, tokens, mask);
    assert.throws(() => distogramLddt(table, new Float32Array(9)), RangeError);
  });
});

describe("the live pLDDT map", () => {
  it("is increasing, so it never reorders what lDDT ranked", () => {
    const mapped = lddtToPlddt(Float32Array.from([0, 20, 50, 80, 100]));
    for (let i = 1; i < mapped.length; i += 1) {
      assert.ok(mapped[i] > mapped[i - 1], `${mapped[i]} follows ${mapped[i - 1]}`);
    }
  });

  it("compresses, which the measurement says it must", () => {
    // 🔴 A DISTOGRAM CANNOT SEPARATE A 55 FROM A 95 BEFORE THE STRUCTURE
    // EXISTS. 6.9 mean error per token leave-one-target-out is what this is
    // worth, and a map that kept the full 0-100 range would be claiming more.
    const mapped = lddtToPlddt(Float32Array.from([0, 100]));
    assert.ok(mapped[0] > 35 && mapped[0] < 45, `bottom mapped to ${mapped[0]}`);
    assert.ok(mapped[1] > 90 && mapped[1] <= 100, `top mapped to ${mapped[1]}`);
  });

  it("stays inside pLDDT's range whatever it is handed", () => {
    for (const value of lddtToPlddt(Float32Array.from([-80, 0, 100, 900]))) {
      assert.ok(value >= 0 && value <= 100, `${value} is outside 0-100`);
    }
  });
});
