import { describe, expect, it } from "./harness.js";
import { CLICK_SLOP_PX, THREE_LETTER, mutationName, residueAt, substitute, wasClick } from "../web/mutate.js";
import { KYTE_DOOLITTLE, RESIDUES_BY_HYDROPATHY } from "../web/hydrophobicity.js";

describe("mutating one position of the folded sequence", () => {
  it("replaces exactly one residue and leaves the length alone", () => {
    expect(substitute("ACDEF", 0, "W")).toBe("WCDEF");
    expect(substitute("ACDEF", 2, "W")).toBe("ACWEF");
    expect(substitute("ACDEF", 4, "W")).toBe("ACDEW");
    expect(substitute("ACDEF", 2, "W").length).toBe(5);
  });

  it("replaces multiple selected residues at once", () => {
    expect(substitute("ACDEF", [0, 2, 4], "W")).toBe("WCWEW");
    expect(substitute("AAAAAAAAAA", [2, 5, 8], "G")).toBe("AAGAAGAAGA");
    expect(substitute("ACDEF", new Set([1, 3]), "K")).toBe("AKDKF");
    expect(substitute("ACDEF", [], "W")).toBe("ACDEF");
  });

  it("handles colon-separated complex sequences", () => {
    expect(substitute("AAAA:GGGG", 0, "W")).toBe("WAAA:GGGG");
    expect(substitute("AAAA:GGGG", 4, "W")).toBe("AAAA:WGGG");
    expect(substitute("AAAA:GGGG", 7, "W")).toBe("AAAA:GGGW");
    expect(substitute("AAAA:GGGG", [0, 4], "W")).toBe("WAAA:WGGG");
    expect(mutationName("AAAA:GGGG", 4, "W")).toBe("G5W");
  });

  // THE INDEX IS INTO THE SEQUENCE THAT WAS FOLDED, and the box is editable
  // while a structure is on screen - so an index that no longer fits is a
  // reader who trimmed the sequence after folding it, not a bug in the click.
  // Refusing beats writing at the wrong place: String.slice would happily
  // return the sequence unchanged for an index past the end, which reads as
  // "the mutation did nothing" and is indistinguishable from a dead menu.
  it("refuses a position the sequence does not have", () => {
    expect(() => substitute("ACDEF", 5, "W")).toThrow();
    expect(() => substitute("ACDEF", -1, "W")).toThrow();
    expect(() => substitute("ACDEF", 1.5, "W")).toThrow();
    expect(() => substitute("ACDEF", [1, 5], "W")).toThrow();
  });

  it("refuses anything that is not one of the twenty", () => {
    expect(() => substitute("ACDEF", 1, "B")).toThrow();
    expect(() => substitute("ACDEF", 1, "TRP")).toThrow();
    expect(() => substitute("ACDEF", 1, "")).toThrow();
    expect(() => substitute("ACDEF", [0, 1], "B")).toThrow();
  });

  // ...X is in the three-letter table because a prediction may contain one,
  // and the menu has to be able to LABEL the residue it is replacing. It is
  // deliberately not offered as a destination.
  it("names a mutation the way a paper does, 1-based", () => {
    expect(mutationName("ACDEF", 2, "W")).toBe("D3W");
    expect(mutationName("ACDEF", 0, "G")).toBe("A1G");
    expect(mutationName("ACDEF", [0, 2, 4], "W")).toBe("A1W, D3W, F5W");
  });

  // WHAT A PICK ANSWERS WITH. Measured against a live py2Dmol embed: clicking
  // a carbon of a tryptophan's side chain returned atom 289, whose sidechainMap
  // entry is {owner: 192, el: "C"} - so the map here has the shape the real one
  // does. py2Dmol's own click-to-select does this walk internally; picking by
  // coordinate, which is what a popup needs, does not.
  const stubViewer = ({ pick, sidechains = [], types }) => ({
    pickResidueAt: () => pick,
    sidechainMap: new Map(sidechains),
    positionTypes: types,
  });

  it("walks a side-chain atom back to the residue that owns it", () => {
    const v = stubViewer({ pick: 289, sidechains: [[289, { owner: 192, el: "C" }]],
      types: new Array(200).fill("P") });
    expect(residueAt(v, 200, 10, 10)).toBe(192);
  });

  it("takes a backbone hit as itself", () => {
    const v = stubViewer({ pick: 7, types: new Array(200).fill("P") });
    expect(residueAt(v, 200, 10, 10)).toBe(7);
  });

  it("answers -1 for the background, for a non-protein, and past the sequence", () => {
    const types = new Array(200).fill("P");
    expect(residueAt(stubViewer({ pick: -1, types }), 200, 10, 10)).toBe(-1);
    // a ligand is clickable and is not something a substitution means anything for
    const ligand = stubViewer({ pick: 5, types });
    ligand.positionTypes[5] = "L";
    expect(residueAt(ligand, 200, 10, 10)).toBe(-1);
    // ...and a side-chain atom whose owner is past the sequence cannot be one
    // of its residues. Guarding on the ATOM index instead would pass here and
    // fail for the atom above, whose index is also past the end.
    const past = stubViewer({ pick: 289, sidechains: [[289, { owner: 192, el: "C" }]], types });
    expect(residueAt(past, 100, 10, 10)).toBe(-1);
  });

  it("survives a viewer that has not drawn anything yet", () => {
    expect(residueAt(undefined, 59, 10, 10)).toBe(-1);
    expect(residueAt({}, 59, 10, 10)).toBe(-1);
  });

  it("offers each of the twenty exactly once, and never X", () => {
    const codes = RESIDUES_BY_HYDROPATHY;
    expect(codes.length).toBe(20);
    expect(new Set(codes).size).toBe(20);
    expect(codes.includes("X")).toBe(false);
    for (const code of codes) expect(THREE_LETTER[code] !== undefined).toBe(true);
  });
});

// 🔴 THE ROTATION BUG. Turning the structure is a drag, and a drag ends in a
// click - so before this rule every spin that happened to finish over a residue
// opened the mutation panel on it.
describe("telling a click from a rotation", () => {
  const at = (x, y) => ({ clientX: x, clientY: y });

  it("takes a press that barely moves as a click", () => {
    expect(wasClick({ x: 100, y: 100 }, at(100, 100))).toBe(true);
    expect(wasClick({ x: 100, y: 100 }, at(102, 101))).toBe(true);
    expect(wasClick({ x: 100, y: 100 }, at(100, 103))).toBe(true);
  });

  it("takes a press that travels as a drag", () => {
    expect(wasClick({ x: 100, y: 100 }, at(140, 100))).toBe(false);
    expect(wasClick({ x: 100, y: 100 }, at(100, 140))).toBe(false);
    expect(wasClick({ x: 100, y: 100 }, at(60, 70))).toBe(false);
  });

  it("measures distance, not distance along one axis", () => {
    // 3 across and 3 down is 4.24 - a drag, though neither axis reaches 4
    expect(wasClick({ x: 0, y: 0 }, at(3, 3))).toBe(false);
    expect(wasClick({ x: 0, y: 0 }, at(3, 2))).toBe(true);
  });

  it("agrees with py2Dmol's own threshold", () => {
    expect(CLICK_SLOP_PX).toBe(4);
    // exactly at the threshold is a drag, matching `moved < 4`
    expect(wasClick({ x: 0, y: 0 }, at(4, 0))).toBe(false);
    expect(wasClick({ x: 0, y: 0 }, at(3.99, 0))).toBe(true);
  });

  it("treats a click with no press behind it as a drag, not a pick", () => {
    // ...a click arriving without a pointerdown we saw is not one to trust
    expect(wasClick(undefined, at(10, 10))).toBe(false);
  });
});
