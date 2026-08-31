import { describe, expect, it } from "./harness.js";
import {
  mergeUnpairedChainA3ms, residueIndexWithChainBreaks, splitComplexA3mByChain,
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
