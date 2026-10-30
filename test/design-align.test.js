import { describe, expect, it } from "./harness.js";
import { alignToReference } from "../cpu/design/align.js";
import { alphaCarbons, coordinateAtoms } from "../shared/heads/superpose-pdb.js";

const residue = (chain, number, x, y = 0, z = 0) => {
  const at = (value) => value.toFixed(3).padStart(8);
  const tag = `ALA ${chain}${String(number).padStart(4)}    `;
  return [
    `ATOM      1  N   ${tag}${at(x - 1)}${at(y)}${at(z)}  1.00  0.00           N`,
    `ATOM      2  CA  ${tag}${at(x)}${at(y)}${at(z)}  1.00 42.00           C`,
    `ATOM      3  C   ${tag}${at(x + 1)}${at(y)}${at(z)}  1.00  0.00           C`,
    `ATOM      4  O   ${tag}${at(x + 2)}${at(y)}${at(z)}  1.00  0.00           O`,
  ].join("\n");
};

const pdb = (...residues) => `${residues.join("\n")}\nEND\n`;

/**
 * 🔴 THE RAW DISTANCE, WITH NO SUPERPOSITION. Measuring these with
 * `selfConsistency` would superpose first and report zero however far apart the
 * two files are - which it did, in the first version of this check, and made a
 * broken alignment look perfect.
 */
const rawOffset = (a, b, chain) => {
  const left = coordinateAtoms(a);
  const right = coordinateAtoms(b);
  const here = alphaCarbons(left, (id) => id === chain);
  const there = alphaCarbons(right, (id) => id === chain);
  let sum = 0;
  for (let index = 0; index < here.length; index += 1) {
    const p = left.points[here[index]];
    const q = right.points[there[index]];
    sum += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  }
  return Math.sqrt(sum / here.length);
};

const TARGET = [4, 5, 6].map((n, i) => residue("B", n, 50 + i * 4));
const first = pdb(residue("A", 1, 0), residue("A", 2, 4), residue("A", 3, 8), ...TARGET);

describe("putting every cycle of a run in the same place", () => {
  // 🔴 A FOLD HAS NO FRAME OF REFERENCE: each cycle sits wherever the model put
  // it, so played back the whole complex tumbles and the thing that actually
  // changed is invisible underneath the tumbling.
  it("holds the target still and brings the design with it", () => {
    const moved = pdb(residue("A", 1, 0, 100), residue("A", 2, 4, 100), residue("A", 3, 8, 100),
      ...[4, 5, 6].map((n, i) => residue("B", n, 50 + i * 4, 100)));
    expect(Math.round(rawOffset(first, moved, "B"))).toBe(100);

    const aligned = alignToReference(moved, first, "A");
    expect(rawOffset(first, aligned, "B")).toBeLessThan(1e-3);
    expect(rawOffset(first, aligned, "A")).toBeLessThan(1e-3);
  });

  // 🔴 AND A DESIGN THAT GENUINELY MOVED MUST STILL READ AS MOVED. An alignment
  // that fitted the whole complex would split the difference and quietly shrink
  // the one number the viewer exists to show.
  it("keeps a real change in the design", () => {
    const shifted = pdb(residue("A", 1, 0, 100, 7), residue("A", 2, 4, 100, 7),
      residue("A", 3, 8, 100, 7),
      ...[4, 5, 6].map((n, i) => residue("B", n, 50 + i * 4, 100)));
    const aligned = alignToReference(shifted, first, "A");
    expect(rawOffset(first, aligned, "B")).toBeLessThan(1e-3);
    expect(rawOffset(first, aligned, "A")).toBeCloseTo(7, 2);
  });

  // 🔴 EVERYTHING AFTER COLUMN 54 HAS TO SURVIVE, the B-factor above all: the
  // viewer colours by it, so an alignment that rebuilt the line rather than
  // splicing the coordinates would repaint every frame.
  it("rewrites coordinates and nothing else", () => {
    const aligned = alignToReference(pdb(residue("A", 1, 9), ...TARGET), first, "A");
    const lines = aligned.split("\n").filter((line) => line.startsWith("ATOM"));
    expect(lines.length).toBe(16);
    for (const line of lines) expect(line.slice(54)).toContain("1.00");
    expect(aligned).toContain("42.00");
  });

  // A monomer has nothing else to hold, so it is fitted on itself: the tumbling
  // still goes, and there is nothing for it to move relative to.
  it("falls back to the whole chain when there is no target", () => {
    const alone = pdb(residue("A", 1, 0), residue("A", 2, 4), residue("A", 3, 8));
    const far = pdb(residue("A", 1, 70), residue("A", 2, 74), residue("A", 3, 78));
    expect(rawOffset(alone, alignToReference(far, alone, "A"), "A")).toBeLessThan(1e-3);
  });

  // 🔴 TOO FEW ATOMS TO FIT IS LEFT ALONE, not fitted on two points and sent
  // somewhere arbitrary.
  it("leaves a frame it cannot fit exactly where it is", () => {
    const tiny = pdb(residue("A", 1, 0), residue("B", 4, 50));
    expect(alignToReference(tiny, first, "A")).toBe(tiny);
  });
});

describe("a small-molecule binder's refold", () => {
  // SAM's methionine half has an atom named CA. As the design's only "other chain" it offered one alpha carbon, the
  // fit was skipped, and the refold was drawn in the fold's own frame.
  const sam = (x) => {
    const at = (value) => value.toFixed(3).padStart(8);
    return [
      `HETATM   90  CA  SAM B   1    ${at(x)}${at(3)}${at(0)}  1.00 80.00           C`,
      `HETATM   91  SD  SAM B   1    ${at(x + 1.8)}${at(3)}${at(0)}  1.00 80.00           S`,
    ].join("\n");
  };
  const design = pdb(residue("A", 1, 0, 0, 0), residue("A", 2, 4, 1, 0), residue("A", 3, 8, 0, 1), residue("A", 4, 12, 2, 0), sam(6));
  // The same protein, shifted 20 A: what a fold returns in its own frame.
  const refold = pdb(residue("A", 1, 20, 0, 0), residue("A", 2, 24, 1, 0), residue("A", 3, 28, 0, 1), residue("A", 4, 32, 2, 0), sam(26));

  it("is fitted on its protein, not left where the fold put it", () => {
    const fitted = alignToReference(refold, design, "A");
    expect(rawOffset(fitted, design, "A")).toBeLessThan(1e-3);
  });
});
