import { describe, expect, it } from "./harness.js";
import { alignPositions, correspondence } from "../web/align.js";

describe("pairing residues between two sequences", () => {
  it("pairs position for position when nothing was inserted or deleted", () => {
    expect(alignPositions("ACDEF", "ACDEF")).toEqual([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]);
  });

  it("pairs a substitution rather than dropping it", () => {
    // ...the substituted residue still has a backbone, and it is usually the
    // one the reader is looking at
    expect(alignPositions("ACDEF", "AWDEF")).toEqual([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]);
  });

  it("shifts everything after an insertion", () => {
    // ACDEF -> AC[W]DEF: the new W pairs with nothing, and D E F pair with the
    // D E F they were, not with the residues now sitting at their old indices
    expect(alignPositions("ACWDEF", "ACDEF"))
      .toEqual([[0, 0], [1, 1], [3, 2], [4, 3], [5, 4]]);
  });

  it("shifts everything after a deletion", () => {
    expect(alignPositions("ACDEF", "ACWDEF"))
      .toEqual([[0, 0], [1, 1], [2, 3], [3, 4], [4, 5]]);
  });

  it("handles an insertion at either end", () => {
    expect(alignPositions("WACDEF", "ACDEF")).toEqual([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]);
    expect(alignPositions("ACDEFW", "ACDEF")).toEqual([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]);
  });

  it("pairs nothing with an empty sequence", () => {
    expect(alignPositions("", "ACDEF")).toEqual([]);
    expect(alignPositions("ACDEF", "")).toEqual([]);
  });

  it("never pairs an index twice, and never runs backwards", () => {
    const pairs = alignPositions("MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIAKQRISFVKSHFSRQLE");
    const left = pairs.map(([i]) => i);
    const right = pairs.map(([, j]) => j);
    expect(new Set(left).size).toBe(left.length);
    expect(new Set(right).size).toBe(right.length);
    for (let k = 1; k < pairs.length; k += 1) {
      expect(left[k] > left[k - 1]).toBe(true);
      expect(right[k] > right[k - 1]).toBe(true);
    }
  });

  it("pairs most of two sequences that differ by one residue", () => {
    const a = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
    const pairs = alignPositions(a, a.slice(0, 20) + a.slice(21));   // delete residue 21
    expect(pairs.length).toBe(a.length - 1);
  });
});

// What LOCAL alignment buys, and the trap it brings with it.
describe("local alignment behaviour", () => {
  it("pairs only the shared stretch when the ends disagree", () => {
    // ...a global alignment forces the ends to pair and drags the fit round to
    // honour a correspondence that is not there
    const pairs = alignPositions("QQQQACDEFGHIKL", "ACDEFGHIKLZZZZ");
    expect(pairs).toEqual([[4, 0], [5, 1], [6, 2], [7, 3], [8, 4],
      [9, 5], [10, 6], [11, 7], [12, 8], [13, 9]]);
  });

  it("pairs a truncation without inventing terminal correspondence", () => {
    const pairs = alignPositions("ACDEFGHIKL", "ACDEFG");
    expect(pairs).toEqual([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]);
  });

  it("runs straight through an isolated substitution", () => {
    const a = "MKTAYIAKQRQISFVKSHFSRQ";
    const mutated = `${a.slice(0, 11)}W${a.slice(12)}`;
    expect(alignPositions(mutated, a).length).toBe(a.length);
  });

  // 🔴 WHY MATCH IS 2. A local alignment floors its score at zero, so a mismatch
  // survives only on score banked before it - and in the SECOND residue there is
  // exactly one match banked. At match 1 that cancels, the walk restarts past it
  // and the first residue drops out of the fit. Mid-sequence either score runs
  // through, which is why this has to be tested at the terminus to be tested at
  // all: an earlier version of this test put the mutation at position 11 and
  // passed under both scores, proving nothing.
  it("keeps the first residue when the second one is substituted", () => {
    const a = "MKTAYIAKQRQISFVKSHFSRQ";
    const mutated = `M W${a.slice(2)}`.replace(" ", "");
    const pairedAt = (scores) => alignPositions(mutated, a, scores).some(([i]) => i === 0);
    expect(pairedAt(undefined)).toBe(true);                              // match 2
    expect(pairedAt({ match: 1, mismatch: -1, gap: -2 })).toBe(false);   // match 1
  });

  it("survives two edits at once", () => {
    // an insertion near the front and a deletion near the back
    const pairs = alignPositions("ACWDEFGHIL", "ACDEFGHIKL");
    expect(pairs.length >= 8).toBe(true);
    for (const [i, j] of pairs) expect("ACWDEFGHIL"[i]).toBe("ACDEFGHIKL"[j]);
  });
});

describe("correspondence", () => {
  it("skips the table entirely when the lengths match", () => {
    const same = correspondence("ACDEF", "AWDEF");
    expect(same.identical).toBe(true);
    expect(same.from).toEqual([0, 1, 2, 3, 4]);
    expect(same.to).toEqual([0, 1, 2, 3, 4]);
  });

  it("returns two lists of equal length for an indel", () => {
    const shifted = correspondence("ACWDEF", "ACDEF");
    expect(shifted.identical).toBe(false);
    expect(shifted.from.length).toBe(shifted.to.length);
    expect(shifted.from).toEqual([0, 1, 3, 4, 5]);
    expect(shifted.to).toEqual([0, 1, 2, 3, 4]);
  });
});
