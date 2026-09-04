import { describe, expect, it } from "./harness.js";
import {
  alphaCarbons, coordinateAtoms, rewriteCoordinates, superposeCycle, superposePdb,
} from "../src/design/superpose-pdb.js";

/**
 * One coordinate line, built by COLUMN rather than by concatenation.
 *
 * 🔴 THE FIRST VERSION OF THIS CONCATENATED PADDED FIELDS AND PUT THE CHAIN ID
 * IN THE WRONG COLUMN, so six assertions failed against a parser that was
 * right. A PDB is a fixed-column format and the only way to write one that
 * cannot drift from the reader is to place each field at its documented
 * offset - which is also the format the reader is being tested against.
 */
function atom(serial, name, resName, chain, resSeq, x, y, z,
              extra = "  1.00 50.00           C") {
  const line = new Array(54).fill(" ");
  const put = (start, text) => {
    for (let index = 0; index < text.length; index += 1) line[start + index] = text[index];
  };
  put(0, "ATOM  ");
  put(11 - String(serial).length, String(serial));   // serial, right in 6-10
  put(12, name.padEnd(4));                           // atom name, 12-15
  put(20 - resName.length, resName);                 // resName, right in 17-19
  put(21, chain);                                    // chainID, 21
  put(26 - String(resSeq).length, String(resSeq));   // resSeq, right in 22-25
  put(30, x.toFixed(3).padStart(8));
  put(38, y.toFixed(3).padStart(8));
  put(46, z.toFixed(3).padStart(8));
  return line.join("") + extra;
}

/** A two-chain backbone: chain A designed, chain B the fixed target. */
function structure(offset = 0, extraSideChain = false) {
  const lines = [];
  let serial = 1;
  for (const [chain, residues] of [["A", 4], ["B", 5]]) {
    for (let residue = 1; residue <= residues; residue += 1) {
      const base = residue * 3 + (chain === "B" ? 40 : 0);
      for (const name of ["N", "CA", "C", "O"]) {
        lines.push(atom(serial += 1, name, "ALA", chain, residue,
                        base + offset, residue + offset, chain === "A" ? 1 : 9));
      }
      // A redesigned residue carries different side-chain atoms, which is the
      // whole reason the frames cannot be aligned by atom index.
      if (extraSideChain && chain === "A") {
        lines.push(atom(serial += 1, "CB", "ALA", chain, residue,
                        base + offset, residue + offset, 2));
      }
    }
  }
  lines.push("TER", "END", "");
  return lines.join("\n");
}

/** A stand-in for py2Dmol's superpose: a fixed shift, applied to everything. */
const shiftBy = (dx) => (points) => points.map(([x, y, z]) => [x + dx, y, z]);

/** Records what it was asked to fit on, and moves nothing. */
function recording() {
  const calls = [];
  const superpose = (points, reference, pairs) => {
    calls.push({ points: points.length, reference: reference.length, ...pairs });
    return points;
  };
  return { calls, superpose };
}

describe("coordinateAtoms", () => {
  it("reads every ATOM and HETATM line and nothing else", () => {
    const atoms = coordinateAtoms(structure());
    expect(atoms.points).toHaveLength(9 * 4);
    expect(atoms.names[0]).toBe("N");
    expect(atoms.names[1]).toBe("CA");
    expect(atoms.chains[0]).toBe("A");
    expect(atoms.chains[atoms.chains.length - 1]).toBe("B");
  });

  it("reads a HETATM, which is how toPdb writes a ligand", () => {
    const pdb = `${structure().trim()}\n`
      + `${atom(999, "FE", "HEM", "C", 301, 1, 2, 3, "  1.00 20.00          FE")
            .replace(/^ATOM  /, "HETATM")}\n`;
    const atoms = coordinateAtoms(pdb);
    expect(atoms.chains[atoms.chains.length - 1]).toBe("C");
    expect(atoms.points[atoms.points.length - 1]).toEqual([1, 2, 3]);
  });

  it("skips a line whose coordinates do not parse, rather than moving a NaN", () => {
    const broken = atom(1, "CA", "ALA", "A", 1, 0, 0, 0)
      .slice(0, 30) + "     nan     nan     nan" + "  1.00  0.00";
    expect(coordinateAtoms(`${broken}\nEND\n`).points).toHaveLength(0);
  });
});

describe("alphaCarbons", () => {
  it("selects only CA, and only from the wanted chains", () => {
    const atoms = coordinateAtoms(structure());
    expect(alphaCarbons(atoms, (chain) => chain === "A")).toHaveLength(4);
    expect(alphaCarbons(atoms, (chain) => chain !== "A")).toHaveLength(5);
    for (const index of alphaCarbons(atoms, () => true)) {
      expect(atoms.names[index]).toBe("CA");
    }
  });
});

describe("rewriteCoordinates", () => {
  // 🔴 EVERYTHING PAST COLUMN 54 IS WHAT THE VIEWER COLOURS BY. The B-factor
  // carries pLDDT here, so a rewrite that reformatted the line rather than
  // splicing three fields would repaint every frame grey and look like a
  // confidence bug.
  it("replaces the coordinates and leaves the rest of the line byte for byte", () => {
    const original = structure();
    const atoms = coordinateAtoms(original);
    const moved = rewriteCoordinates(atoms, atoms.points.map(([x, y, z]) => [x + 1, y, z]));
    const before = original.split("\n");
    const after = moved.split("\n");
    expect(after).toHaveLength(before.length);
    for (let index = 0; index < before.length; index += 1) {
      if (!before[index].startsWith("ATOM")) {
        expect(after[index]).toBe(before[index]);
        continue;
      }
      expect(after[index].slice(0, 30)).toBe(before[index].slice(0, 30));
      expect(after[index].slice(54)).toBe(before[index].slice(54));
      expect(Number(after[index].slice(30, 38)))
        .toBeCloseTo(Number(before[index].slice(30, 38)) + 1, 3);
    }
  });

  it("round-trips through a parse", () => {
    const atoms = coordinateAtoms(structure());
    const again = coordinateAtoms(rewriteCoordinates(atoms, atoms.points));
    expect(again.points).toEqual(atoms.points);
  });

  it("keeps the columns eight wide even for a fit that went wrong", () => {
    const atoms = coordinateAtoms(structure());
    const wild = atoms.points.map(() => [-123456.789, 0, 0]);
    const line = rewriteCoordinates(atoms, wild).split("\n")
      .find((text) => text.startsWith("ATOM"));
    expect(line.slice(30, 38)).toHaveLength(8);
    expect(line.slice(54)).toBe("  1.00 50.00           C");
  });

  it("refuses a point count that is not the atom count", () => {
    const atoms = coordinateAtoms(structure());
    expect(() => rewriteCoordinates(atoms, atoms.points.slice(1))).toThrow(RangeError);
  });
});

describe("superposePdb", () => {
  // The point of the whole module: the target is what holds still.
  it("fits on the target's alpha carbons, not the designed chain's", () => {
    const { calls, superpose } = recording();
    const result = superposePdb(superpose, structure(0, true), structure(), { designed: "A" });
    expect(result.on).toBe("target");
    expect(result.fitted).toBe(5);
    expect(calls).toHaveLength(1);
    // Every fitted atom is a CA of chain B, in both structures.
    const moving = coordinateAtoms(structure(0, true));
    for (const index of calls[0].from) {
      expect(moving.names[index]).toBe("CA");
      expect(moving.chains[index]).toBe("B");
    }
  });

  // 🔴 THE ATOM COUNTS DIFFER AND THAT IS THE NORMAL CASE. A redesigned chain
  // has different side chains, which is exactly why py2Dmol's own frame
  // alignment cannot do this - it needs equal position counts.
  it("moves every atom, including the ones the two structures do not share", () => {
    const moving = structure(0, true);
    const result = superposePdb(shiftBy(10), moving, structure(), { designed: "A" });
    expect(coordinateAtoms(moving).points.length
      !== coordinateAtoms(structure()).points.length).toBe(true);
    const before = coordinateAtoms(moving).points;
    const after = coordinateAtoms(result.pdb).points;
    expect(after).toHaveLength(before.length);
    for (let index = 0; index < before.length; index += 1) {
      expect(after[index][0]).toBeCloseTo(before[index][0] + 10, 3);
    }
  });

  it("falls back to the designed chain when there is no target", () => {
    const monomer = (offset) => structure(offset).split("\n")
      .filter((line) => !line.startsWith("ATOM") || line[21] === "A").join("\n");
    const result = superposePdb(shiftBy(1), monomer(0), monomer(3), { designed: "A" });
    expect(result.on).toBe("designed");
    expect(result.fitted).toBe(4);
  });

  // A fit that paired the first N of each would put a plausible structure in
  // the wrong place, which is worse than not moving it.
  it("does nothing when the fitted chains have different lengths", () => {
    const shorter = structure().split("\n")
      .filter((line) => !line.startsWith("ATOM") || line[21] !== "B"
        || Number(line.slice(22, 26)) < 5).join("\n");
    const result = superposePdb(shiftBy(1), structure(), shorter, { designed: "A" });
    expect(result.on).toBe("none");
    expect(result.fitted).toBe(0);
    expect(result.pdb).toBe(structure());
  });

  it("does nothing when there are too few atoms to fit on", () => {
    const tiny = `${atom(1, "CA", "ALA", "A", 1, 1, 2, 3)}\nEND\n`;
    expect(superposePdb(shiftBy(1), tiny, tiny, { designed: "A" }).on).toBe("none");
  });
});

describe("superposeCycle", () => {
  // 🔴 THE POINT: ONE TRANSFORM FOR THE WHOLE CYCLE. A diffusion trajectory
  // starts as noise, so fitting each frame on its own "target chain" fits on a
  // cloud and lands it somewhere arbitrary. The frames already share a
  // reference frame; what they need is the settled structure's rigid move.
  it("moves every frame by the transform fitted from the settled structure", () => {
    const settled = structure(0, true);
    // A trajectory whose early frames are nothing like the settled structure.
    const frames = [structure(100, true), structure(50, true), settled];
    const result = superposeCycle(shiftBy(7), frames, settled, structure(), { designed: "A" });
    expect(result.on).toBe("target");
    expect(result.fitted).toBe(5);
    expect(result.frames).toHaveLength(3);
    for (const [index, moved] of result.frames.entries()) {
      const before = coordinateAtoms(frames[index]).points;
      const after = coordinateAtoms(moved).points;
      expect(after).toHaveLength(before.length);
      for (let atom = 0; atom < before.length; atom += 1) {
        expect(after[atom][0]).toBeCloseTo(before[atom][0] + 7, 3);
      }
    }
    expect(coordinateAtoms(result.settled).points[0][0])
      .toBeCloseTo(coordinateAtoms(settled).points[0][0] + 7, 3);
  });

  it("fits once, not once per frame", () => {
    const { calls, superpose } = recording();
    const settled = structure(0, true);
    superposeCycle(superpose, [settled, settled, settled], settled, structure(),
                   { designed: "A" });
    expect(calls).toHaveLength(1);
    // ...and it was handed the settled structure plus all three frames.
    const one = coordinateAtoms(settled).points.length;
    expect(calls[0].points).toBe(one * 4);
  });

  it("hands everything back untouched when it cannot fit", () => {
    const frames = [structure(1), structure(2)];
    const shorter = structure().split("\n")
      .filter((line) => !line.startsWith("ATOM") || line[21] !== "B"
        || Number(line.slice(22, 26)) < 5).join("\n");
    const result = superposeCycle(shiftBy(1), frames, structure(), shorter, { designed: "A" });
    expect(result.on).toBe("none");
    expect(result.frames).toEqual(frames);
  });
});
