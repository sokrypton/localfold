import { describe, expect, it } from "./harness.js";
import { OMIT_BIAS, sampleOptionsFrom } from "../cpu/design/runspec.js";
import { ALPHABET } from "../cpu/design/mpnn/constants.js";

const V = ALPHABET.length;
const at = (letter) => ALPHABET.indexOf(letter);
const bias = (options, position, letter) => options.bias[position * V + at(letter)];

describe("a RunSpec translated into Model.sample's options", () => {
  // The defaults are hers, named in docs/DESIGN-RUNSPEC.md, and a default that
  // drifts is a design that silently changes.
  it("defaults temperature to 0.1, the seed to 0, and the order to random", () => {
    const options = sampleOptionsFrom({}, { length: 4 });
    expect(options.temperature).toBe(0.1);
    // `order: undefined` is how our engine is told to shuffle for itself;
    // "random" is not passed through as a string.
    expect(options.order).toBe(undefined);
    expect(options.symmetry).toEqual([]);
    expect(Array.from(options.chainMask)).toEqual([1, 1, 1, 1]);
    // seed 0 and seed 0 draw the same first number; a different seed does not.
    expect(sampleOptionsFrom({}, { length: 4 }).rng()).toBe(options.rng());
    const fromOne = sampleOptionsFrom({ seed: 1 }, { length: 4 }).rng();
    const fromTwo = sampleOptionsFrom({ seed: 2 }, { length: 4 }).rng();
    expect(fromOne === fromTwo).toBe(false);
  });

  // 🔴 THE FOUR BIAS FIELDS APPLY IN HER ORDER, AND THE ORDER IS OBSERVABLE:
  // the two omit fields OVERWRITE where the two bias fields ADD. "Discourage C"
  // and "never choose C" have to compose as the omit winning, or a letter the
  // caller forbade could be biased back into reach.
  it("adds the bias fields and lets the omit fields overwrite them", () => {
    const options = sampleOptionsFrom({
      bias_AA: { A: -1, C: 5 },
      bias_AA_per_residue: { 2: { W: 2 } },
      omit_AA: "C",
      omit_AA_per_residue: { 3: "P" },
    }, { length: 6 });
    expect(bias(options, 0, "A")).toBe(-1);
    expect(bias(options, 5, "A")).toBe(-1);
    expect(bias(options, 2, "W")).toBe(2);
    expect(bias(options, 1, "W")).toBe(0);
    // the omit beat the +5
    expect(bias(options, 0, "C")).toBe(OMIT_BIAS);
    expect(bias(options, 3, "P")).toBe(OMIT_BIAS);
    expect(bias(options, 2, "P")).toBe(0);
  });

  // 🔴 X IS NOT A DESIGN CHOICE AND IS NOT A KNOB. The loop puts X into the
  // sequence that gets folded and must never read one back out, so it is
  // omitted underneath whatever the RunSpec says.
  it("omits X at every position whatever the RunSpec asks", () => {
    const options = sampleOptionsFrom({ bias_AA: { X: 1000 } }, { length: 3 });
    expect(bias(options, 0, "X")).toBe(OMIT_BIAS + 1000);
    expect(bias(options, 2, "X")).toBeLessThan(-1e7);
  });

  it("holds a fixed position and conditions it on the letter asked for", () => {
    const options = sampleOptionsFrom({ fixed_positions: { 0: "M", 3: "G" } },
      { length: 5, nativeTokens: new Int32Array(5) });
    expect(Array.from(options.chainMask)).toEqual([0, 1, 1, 0, 1]);
    expect(options.S[0]).toBe(at("M"));
    expect(options.S[3]).toBe(at("G"));
  });

  it("designs only the chains named, and refuses one the structure lacks", () => {
    const chainIds = ["A", "A", "B", "B", "C"];
    const options = sampleOptionsFrom({ chains_to_design: ["B"] }, { length: 5, chainIds });
    expect(Array.from(options.chainMask)).toEqual([0, 0, 1, 1, 0]);
    expect(() => sampleOptionsFrom({ chains_to_design: ["Z"] }, { length: 5, chainIds }))
      .toThrow(/chain Z/);
    // ...and it cannot be used without the chain of each position
    expect(() => sampleOptionsFrom({ chains_to_design: ["A"] }, { length: 5 }))
      .toThrow(/chainIds/);
  });

  // A fixed position inside a designed chain is still fixed: the two rules
  // compose, with the position winning over the chain.
  it("lets a fixed position override the chain rule", () => {
    const options = sampleOptionsFrom(
      { chains_to_design: ["A"], fixed_positions: { 1: "W" } },
      { length: 3, chainIds: ["A", "A", "B"] },
    );
    expect(Array.from(options.chainMask)).toEqual([1, 0, 0]);
    expect(options.S[1]).toBe(at("W"));
  });

  it("passes tie groups through as symmetry, and refuses a position in two of them", () => {
    const options = sampleOptionsFrom({ tied_positions: [[2, 5], [3, 6]] }, { length: 8 });
    expect(options.symmetry).toEqual([[2, 5], [3, 6]]);
    // 🔴 normaliseSymmetry would keep the FIRST group and ignore the second
    // silently, so the caller would believe in a tie that does not exist.
    expect(() => sampleOptionsFrom({ tied_positions: [[1, 2], [2, 3]] }, { length: 8 }))
      .toThrow(/two groups/);
    // an empty group is not an error, it is nothing
    expect(sampleOptionsFrom({ tied_positions: [[]] }, { length: 4 }).symmetry).toEqual([]);
  });

  it("takes an explicit decoding order and checks it is a permutation", () => {
    const options = sampleOptionsFrom({ decoding_order: [2, 0, 3, 1] }, { length: 4 });
    expect(Array.from(options.order)).toEqual([2, 0, 3, 1]);
    expect(() => sampleOptionsFrom({ decoding_order: [0, 1] }, { length: 4 })).toThrow(/long/);
    expect(() => sampleOptionsFrom({ decoding_order: [0, 1, 1, 2] }, { length: 4 })).toThrow(/repeats/);
    // 🔴 HERS CHECKS THE LENGTH AND THE DISTINCTNESS BUT NOT THE RANGE, so
    // [0,1,2,9] passes there and decodes position 9 of a 4-residue structure.
    expect(() => sampleOptionsFrom({ decoding_order: [0, 1, 2, 9] }, { length: 4 }))
      .toThrow(/outside 0\.\.3/);
  });

  // 🔴 EVERY REFUSAL HERE IS A CASE HERS LEAVES SILENT (docs/DESIGN-RUNSPEC.md
  // section 7.3). A design that quietly dropped a constraint is worse than one
  // that did not run, and the 0-based position is the mistake most likely to be
  // made: PDB numbering is 1-based, so "residue 54" is not position 54.
  it("refuses what the reference would silently ignore", () => {
    expect(() => sampleOptionsFrom({ fixed_positions: { 9: "M" } }, { length: 4 }))
      .toThrow(/outside 0\.\.3/);
    expect(() => sampleOptionsFrom({ omit_AA_per_residue: { 9: "P" } }, { length: 4 }))
      .toThrow(/outside 0\.\.3/);
    expect(() => sampleOptionsFrom({ tied_positions: [[0, 99]] }, { length: 4 }))
      .toThrow(/outside 0\.\.3/);
    for (const temperature of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, null]) {
      expect(() => sampleOptionsFrom({ temperature }, { length: 4 })).toThrow(/temperature/);
    }
    expect(() => sampleOptionsFrom({ fixed_positions: { 0: "Z" } }, { length: 4 }))
      .toThrow(/not in alphabet/);
    expect(() => sampleOptionsFrom({ fixed_positions: { 0: "MG" } }, { length: 4 }))
      .toThrow(/ONE letter/);
    expect(() => sampleOptionsFrom({ bias_AA: { A: "high" } }, { length: 4 }))
      .toThrow(/finite number/);
    expect(() => sampleOptionsFrom({}, { length: 0 })).toThrow(/length/);
    expect(() => sampleOptionsFrom({}, {})).toThrow(/length/);
  });

  it("refuses a native sequence of the wrong length", () => {
    expect(() => sampleOptionsFrom({}, { length: 4, nativeTokens: new Int32Array(3) }))
      .toThrow(/3 long, the structure is 4/);
  });
});
