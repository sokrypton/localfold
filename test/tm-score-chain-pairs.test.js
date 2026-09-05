/**
 * The per-interface ipTM, which the pooled one cannot stand in for.
 *
 * 🔴 THE CASE THAT MOTIVATED IT is an assembly holding a real homodimer and a
 * designed binder - PDB 27UH is two S100A4 and two VHH. The homodimer's
 * interface is the easy one, and pooling it with the binder's reports a better
 * number about a worse question. These tests are that shape in miniature.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { chainPairTmScores, reduceTmScore } from "../src/heads/tm-score.js";
import { computeTmScores } from "../src/heads/confidence.js";

/** Three chains of `per` tokens each. */
const layout = (per) => {
  const tokens = per * 3;
  const asymId = new Int32Array(tokens);
  for (let i = 0; i < tokens; i += 1) asymId[i] = Math.floor(i / per);
  return { tokens, per, asymId, seqMask: new Float32Array(tokens).fill(1) };
};

/** A term that is `high` between chains `a` and `b` and `low` everywhere else. */
const term = ({ tokens, asymId }, a, b, high, low) => {
  const out = new Float64Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const pair = (asymId[i] === a && asymId[j] === b)
        || (asymId[i] === b && asymId[j] === a);
      out[i * tokens + j] = pair ? high : low;
    }
  }
  return out;
};

describe("per-chain-pair ipTM", () => {
  it("separates a strong interface from a weak one in the same assembly", () => {
    const shape = layout(6);
    // Chains 0 and 1 touch confidently; everything else is poor - the 27UH
    // shape, where a native dimer sits beside a designed binder.
    const values = term(shape, 0, 1, 0.9, 0.1);
    const { chains, scores } = chainPairTmScores(
      values, shape.tokens, shape.asymId, shape.seqMask);
    assert.deepEqual(chains, [0, 1, 2]);
    assert.ok(Math.abs(scores.get("0|1") - 0.9) < 1e-9, `0|1 was ${scores.get("0|1")}`);
    assert.ok(Math.abs(scores.get("0|2") - 0.1) < 1e-9, `0|2 was ${scores.get("0|2")}`);
    assert.ok(Math.abs(scores.get("1|2") - 0.1) < 1e-9, `1|2 was ${scores.get("1|2")}`);
  });

  it("is what the pooled score hides, which is the reason it exists", () => {
    // 🔴 THE POOLED SCORE SITS ABOVE THE INTERFACE THAT MATTERS. One good
    // interface among three drags the pooled number up, so a binder that is
    // placed badly reads as adequate.
    const shape = layout(6);
    const values = term(shape, 0, 1, 0.9, 0.1);
    const pooled = reduceTmScore(values, shape.tokens,
      (i, j) => shape.asymId[i] !== shape.asymId[j]);
    const { scores } = chainPairTmScores(
      values, shape.tokens, shape.asymId, shape.seqMask);
    assert.ok(pooled > scores.get("0|2"),
      `pooled ${pooled} should exceed the weak interface ${scores.get("0|2")}`);
    assert.ok(pooled < scores.get("0|1"),
      `pooled ${pooled} should sit below the strong one ${scores.get("0|1")}`);
  });

  it("agrees with the pooled score when there are only two chains", () => {
    // With one interface there is nothing to pool, so the two must coincide -
    // which is what makes the new score a refinement and not a different
    // quantity.
    const tokens = 8;
    const asymId = Int32Array.from([0, 0, 0, 0, 1, 1, 1, 1]);
    const seqMask = new Float32Array(tokens).fill(1);
    const values = new Float64Array(tokens * tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        values[i * tokens + j] = ((i * 5 + j * 3) % 7) / 10;
      }
    }
    const pooled = reduceTmScore(values, tokens, (i, j) => asymId[i] !== asymId[j]);
    const { scores } = chainPairTmScores(values, tokens, asymId, seqMask);
    assert.equal(scores.get("0|1"), pooled);
  });

  it("skips a pair with nothing live on one side rather than scoring it zero", () => {
    const shape = layout(4);
    const half = new Float32Array(shape.tokens).fill(1);
    for (let i = 8; i < shape.tokens; i += 1) half[i] = 0;   // chain 2 masked off
    const values = term(shape, 0, 1, 0.8, 0.2);
    const { chains, scores } = chainPairTmScores(
      values, shape.tokens, shape.asymId, half);
    assert.deepEqual(chains, [0, 1]);
    assert.ok(!scores.has("0|2"));
    assert.ok(!scores.has("1|2"));
  });

  it("is symmetric in the pair, taking an anchor from either chain", () => {
    // A term that is strong only in one DIRECTION still describes one
    // interface, and the reduction maximises over anchors on both sides.
    const shape = layout(5);
    const tokens = shape.tokens;
    const values = new Float64Array(tokens * tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        if (shape.asymId[i] === 0 && shape.asymId[j] === 1) values[i * tokens + j] = 0.7;
        else if (shape.asymId[i] === 1 && shape.asymId[j] === 0) values[i * tokens + j] = 0.3;
      }
    }
    const { scores } = chainPairTmScores(values, tokens, shape.asymId, shape.seqMask);
    assert.ok(Math.abs(scores.get("0|1") - 0.7) < 1e-9,
      `the better anchor should win, got ${scores.get("0|1")}`);
  });

  // 🔴 AND AlphaFold 2 GETS THE SAME BREAKDOWN FROM THE SAME FUNCTION. It had
  // the pooled ipTM only, which on more than two chains is an average over
  // every interface - so one good contact and two bad ones report as one
  // mediocre number, and there was no way to tell that apart from three
  // mediocre ones. AF2's chains are contiguous blocks rather than asym ids,
  // which is the only difference, and `chainLengths` already says where they
  // are.
  it("gives AlphaFold 2 one score per interface, not one pooled average", () => {
    const length = 30;
    const bins = 64;
    const breaks = Float32Array.from({ length: 63 }, (_, index) => index * 0.5);
    const logits = new Float32Array(length * length * bins);
    const chainOf = (index) => Math.floor(index / 10);
    for (let i = 0; i < length; i += 1) {
      for (let j = 0; j < length; j += 1) {
        const a = chainOf(i);
        const b = chainOf(j);
        // Chains 0 and 1 meet confidently; everything touching chain 2 does not.
        const bin = a === b ? 0 : (a + b === 1 ? 2 : 20);
        logits[(i * length + j) * bins + bin] = 10;
      }
    }
    const scores = computeTmScores(logits, length, breaks, [10, 10, 10]);
    const pairs = scores.chainPairIptm;
    assert.deepEqual(Object.keys(pairs).sort(), ["0|1", "0|2", "1|2"]);
    assert.ok(pairs["0|1"] > 0.4, `AB should be confident, got ${pairs["0|1"]}`);
    assert.ok(pairs["0|2"] < 0.1, `AC should not be, got ${pairs["0|2"]}`);
    assert.ok(pairs["1|2"] < 0.1, `BC should not be, got ${pairs["1|2"]}`);
    // The pooled score is what the breakdown exists to qualify: it sits between
    // the interfaces and describes none of them.
    assert.ok(scores.iptm > pairs["0|2"] && scores.iptm < pairs["0|1"],
      `the pooled ipTM ${scores.iptm} should fall between the interfaces`);
  });

  // A monomer has no interface, and an empty object would read as "measured,
  // and there are none" rather than "the question does not apply".
  it("leaves a single chain with no breakdown at all", () => {
    const length = 8;
    const bins = 64;
    const breaks = Float32Array.from({ length: 63 }, (_, index) => index * 0.5);
    const logits = new Float32Array(length * length * bins);
    assert.equal(computeTmScores(logits, length, breaks).chainPairIptm, undefined);
  });
});
