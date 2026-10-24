import { describe, expect, it } from "./harness.js";
import { describeValidation, selfConsistency } from "../cpu/design/validate.js";

/** One residue's backbone, with CA at (x, y, 0). */
const residue = (chain, number, x, y = 0) => [
  `ATOM      1  N   ALA ${chain}${String(number).padStart(4)}    ${(x - 1).toFixed(3).padStart(8)}`
    + `${y.toFixed(3).padStart(8)}   0.000  1.00  0.00           N`,
  `ATOM      2  CA  ALA ${chain}${String(number).padStart(4)}    ${x.toFixed(3).padStart(8)}`
    + `${y.toFixed(3).padStart(8)}   0.000  1.00  0.00           C`,
  `ATOM      3  C   ALA ${chain}${String(number).padStart(4)}    ${(x + 1).toFixed(3).padStart(8)}`
    + `${y.toFixed(3).padStart(8)}   0.000  1.00  0.00           C`,
  `ATOM      4  O   ALA ${chain}${String(number).padStart(4)}    ${(x + 2).toFixed(3).padStart(8)}`
    + `${y.toFixed(3).padStart(8)}   0.000  1.00  0.00           O`,
].join("\n");

const pdb = (...residues) => `${residues.join("\n")}\nEND\n`;

describe("does the finished design fold back to what it was meant to be", () => {
  // 🔴 THE SAME SHAPE MOVED AND TURNED IS THE SAME SHAPE. A superposition that
  // did not remove the rigid-body difference would report the translation as
  // error, and every validation would fail for a reason that is not about the
  // design at all.
  it("is zero for the same backbone in another place", () => {
    const design = pdb(residue("A", 1, 0), residue("A", 2, 4), residue("A", 3, 8));
    const moved = pdb(residue("A", 1, 100), residue("A", 2, 104), residue("A", 3, 108));
    const { rmsd, comparable, residues } = selfConsistency(design, moved, "A");
    expect(comparable).toBe(true);
    expect(residues).toBe(3);
    expect(rmsd).toBeLessThan(1e-3);
  });

  it("grows as the fold departs from the design", () => {
    const design = pdb(residue("A", 1, 0), residue("A", 2, 4), residue("A", 3, 8));
    const bent = pdb(residue("A", 1, 0), residue("A", 2, 4, 3), residue("A", 3, 8));
    const { rmsd } = selfConsistency(design, bent, "A");
    expect(rmsd).toBeGreaterThan(0.5);
    expect(rmsd).toBeLessThan(3);
  });

  // 🔴 ONE CHAIN, because the TARGET is in both structures and neither method
  // designed it. Superposing the whole complex would align the target - which
  // matches perfectly - and report a small RMSD for a binder that moved a long
  // way.
  it("judges the designed chain, not the target it shares", () => {
    const design = pdb(residue("A", 1, 0), residue("A", 2, 4),
      residue("B", 1, 50), residue("B", 2, 54));
    // chain B identical, chain A moved RIGIDLY - both residues by the same
    // offset, which is the only kind of move a superposition should remove. (An
    // earlier version of this fixture shifted them by different amounts, which
    // is a shear: it changes the chain's own geometry, so a non-zero RMSD was
    // the right answer and the test was wrong.)
    const moved = pdb(residue("A", 1, 0, 20), residue("A", 2, 4, 20),
      residue("B", 1, 50), residue("B", 2, 54));
    expect(selfConsistency(design, moved, "A").rmsd).toBeLessThan(1e-3);
    // ...which is right: two points can always be superposed. The point is that
    // chain B is NOT dragged into the fit, so the number is about chain A alone.
    expect(selfConsistency(design, moved, "A").residues).toBe(2);
    expect(selfConsistency(design, moved).residues).toBe(4);
  });

  // 🔴 DIFFERENT LENGTHS MEAN NO RMSD, NOT A BIG ONE. Aligning the first N of
  // each compares residue 1 to residue 1 of two different molecules and returns
  // a plausible number for it.
  it("refuses to compare folds of different lengths", () => {
    const design = pdb(residue("A", 1, 0), residue("A", 2, 4), residue("A", 3, 8));
    const shorter = pdb(residue("A", 1, 0), residue("A", 2, 4));
    const result = selfConsistency(design, shorter, "A");
    expect(result.comparable).toBe(false);
    expect(Number.isNaN(result.rmsd)).toBe(true);
    expect(describeValidation({ ...result, model: "AF3" })).toContain("not comparable");
  });

  // 🔴 THE NUMBER CARRIES ITS THRESHOLD. "3.90 A" says nothing to a reader who
  // has not read the de novo design literature; 2 A is what the field holds a
  // self-consistent design to.
  it("says what the number means", () => {
    const good = describeValidation({
      model: "AF3", rmsd: 1.1, comparable: true, confidence: { meanPlddt: 88.5, iptm: 0.72 },
    });
    expect(good).toContain("pLDDT 88.5");
    expect(good).toContain("ipTM 0.720");
    expect(good).toContain("RMSD 1.10 A");
    expect(good.includes("above the 2 A")).toBe(false);

    const bad = describeValidation({ model: "AF3", rmsd: 3.9, comparable: true });
    expect(bad).toContain("above the 2 A");

    expect(describeValidation({ model: "AF3" })).toContain("nothing to judge by");
  });
});
