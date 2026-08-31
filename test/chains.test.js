import { describe, expect, it } from "./harness.js";
import {
  chainIdentity, mergeChainA3ms, mergeUnpairedChainA3ms, residueIndexWithChainBreaks,
  splitComplexA3mByChain,
  validatedChainLengths,
} from "../src/input/chains.js";
import { parseA3m } from "../src/input/a3m.js";

describe("monomer-model oligomer preprocessing", () => {
  it("adds ColabFold's 200-index offset at every physical chain boundary", () => {
    expect(Array.from(residueIndexWithChainBreaks(7, [3, 2, 2])))
      .toEqual([0, 1, 2, 203, 204, 405, 406]);
  });

  it("validates that chain lengths are a positive partition of the query", () => {
    expect(validatedChainLengths(5, undefined)).toEqual([5]);
    expect(() => validatedChainLengths(5, [2, 2])).toThrow(/sum to 4; expected 5/);
    expect(() => validatedChainLengths(5, [5, 0])).toThrow(/positive integers/);
  });

  it("builds an unpaired heterooligomer MSA without losing insertions", () => {
    const merged = mergeUnpairedChainA3ms([
      ">a\nACD\n>a_hit\nAcCD\n",
      ">b\nWY\n>b_hit\nW-\n",
    ]);
    const parsed = parseA3m(merged);
    expect(parsed.query).toBe("ACDWY");
    expect(parsed.sequences).toEqual(["ACDWY", "ACD--", "---W-"]);
    expect(parsed.deletionMatrix[1]).toEqual([0, 1, 0, 0, 0]);
  });

  it("expands one alignment independently into both homooligomer copies", () => {
    const chain = ">a\nAC\n>hit\nA-\n";
    const parsed = parseA3m(mergeUnpairedChainA3ms([chain, chain]));
    expect(parsed.query).toBe("ACAC");
    expect(parsed.sequences).toEqual(["ACAC", "A---", "--A-"]);
  });

  it("projects the complex MSA into queries py2Dmol can match to PDB chains", () => {
    const complex = ">query\nACDWY\n>a\nAC---\n>b\n---WY\n>paired\nAC-WY\n";
    const projected = splitComplexA3mByChain(complex, [3, 2]).map(parseA3m);
    expect(projected.map((alignment) => alignment.query)).toEqual(["ACD", "WY"]);
    expect(projected[0].sequences).toEqual(["ACD", "AC-", "AC-"]);
    expect(projected[1].sequences).toEqual(["WY", "WY", "WY"]);
  });
});

describe("pairing repeated chains", () => {
  const chainA = ">q\nAAAA\n>h1\nCCCC\n>h2\nDDDD\n";
  const chainB = ">q\nMM\n>k1\nWW\n";

  it("pairs a homodimer into one row per homolog instead of two", () => {
    const paired = parseA3m(mergeChainA3ms([chainA, chainA]));
    const diagonal = parseA3m(mergeUnpairedChainA3ms([chainA, chainA]));
    expect(paired.length).toBe(8);
    expect(diagonal.length).toBe(8);
    // ...the whole point: half the rows, and each one covers both copies.
    expect(paired.depth).toBe(3);
    expect(diagonal.depth).toBe(5);
    expect(paired.sequences[1]).toBe("CCCCCCCC");
    expect(paired.sequences[2]).toBe("DDDDDDDD");
  });

  it("keeps the query row intact", () => {
    const paired = parseA3m(mergeChainA3ms([chainA, chainA]));
    expect(paired.query).toBe("AAAAAAAA");
  });

  it("pairs the repeated chains of an A2B complex and block-diagonals B", () => {
    const paired = parseA3m(mergeChainA3ms([chainA, chainA, chainB]));
    expect(paired.query).toBe("AAAAAAAAMM");
    expect(paired.depth).toBe(4);
    expect(paired.sequences[1]).toBe("CCCCCCCC--");
    expect(paired.sequences[2]).toBe("DDDDDDDD--");
    expect(paired.sequences[3]).toBe("--------WW");
  });

  it("never pairs distinct proteins, which would invent coevolution", () => {
    const paired = parseA3m(mergeChainA3ms([chainA, chainB]));
    expect(paired.depth).toBe(4);
    for (let row = 1; row < paired.depth; row += 1) {
      const left = /[^-]/.test(paired.sequences[row].slice(0, 4));
      const right = /[^-]/.test(paired.sequences[row].slice(4));
      expect(left && right).toBe(false);
    }
  });

  it("leaves a single chain exactly as the unpaired form does", () => {
    expect(mergeChainA3ms([chainA])).toBe(mergeUnpairedChainA3ms([chainA]));
  });

  it("still splits back into one viewer alignment per chain", () => {
    const a3m = mergeChainA3ms([chainA, chainA]);
    const [first, second] = splitComplexA3mByChain(a3m, [4, 4]);
    expect(parseA3m(first).depth).toBe(3);
    expect(parseA3m(second).depth).toBe(3);
  });
});

describe("multimer chain identity", () => {
  const show = (values) => Array.from(values).join("");

  it("is all zeros for a monomer, which is what one chain means", () => {
    const { asymId, entityId, symId } = chainIdentity(4, undefined);
    expect(show(asymId)).toBe("0000");
    expect(show(entityId)).toBe("0000");
    expect(show(symId)).toBe("0000");
  });

  it("gives a homodimer two chains, one entity and two copies", () => {
    const { asymId, entityId, symId } = chainIdentity(6, [3, 3], ["AAA", "AAA"]);
    expect(show(asymId)).toBe("000111");
    expect(show(entityId)).toBe("000000");
    expect(show(symId)).toBe("000111");
  });

  it("gives a heterodimer two entities, each its own first copy", () => {
    const { asymId, entityId, symId } = chainIdentity(6, [3, 3], ["AAA", "BBB"]);
    expect(show(asymId)).toBe("000111");
    expect(show(entityId)).toBe("000111");
    expect(show(symId)).toBe("000000");
  });

  it("counts copies within each entity separately for A2B2", () => {
    const { asymId, entityId, symId } = chainIdentity(8, [2, 2, 2, 2], ["AA", "AA", "BB", "BB"]);
    expect(show(asymId)).toBe("00112233");
    expect(show(entityId)).toBe("00001111");
    expect(show(symId)).toBe("00110011");
  });

  it("treats every chain as its own entity when no sequences are given", () => {
    const { entityId, symId } = chainIdentity(4, [2, 2]);
    expect(show(entityId)).toBe("0011");
    expect(show(symId)).toBe("0000");
  });

  it("refuses a sequence list that does not match the chains", () => {
    expect(() => chainIdentity(6, [3, 3], ["AAA"])).toThrow(/one sequence per chain/);
  });

  it("refuses chain lengths that do not partition the sequence", () => {
    expect(() => chainIdentity(6, [3, 4])).toThrow(/sum to 7/);
  });
});
