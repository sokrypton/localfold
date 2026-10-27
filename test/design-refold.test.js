import { describe, expect, it } from "./harness.js";
import { foldInputFromPdb, keptTemplates, refoldOverlay } from "../cpu/design/refold.js";

/** One PDB atom record. */
function atom(record, serial, name, residue, chain, number, x, y, z) {
  return `${record.padEnd(6)}${String(serial).padStart(5)} ${name.padEnd(4)} ${residue.padStart(3)} ${chain}`
    + `${String(number).padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}`
    + "  1.00  0.00";
}

/** A chain of CA-only residues along x, offset by `shift`. */
function chain(letter, names, shift = [0, 0, 0], kind = "ATOM", anchor = "CA") {
  return names.map((name, at) => atom(kind, at + 1, anchor, name, letter, at + 1,
    at * 3.8 + shift[0], (at % 2) * 1.5 + shift[1], shift[2]));
}

describe("refolding a design", () => {
  it("reads every chain back: protein, DNA and a ligand", () => {
    const pdb = [...chain("B", ["MET", "LYS", "GLY"]), ...chain("C", ["DA", "DC", "DG"], [0, 9, 0], "ATOM", "C1'"),
      ...chain("A", ["SAM"], [0, 0, 9], "HETATM", "C1")].join("\n");
    expect(foldInputFromPdb(pdb)).toEqual({ sequence: "MKG:ACG", chainKinds: ["protein", "dna"], ligandCodes: ["SAM"] });
  });

  it("refuses a residue it cannot name rather than folding an X", () => {
    const pdb = chain("A", ["MET", "SEP", "GLY"]).join("\n");
    expect(() => foldInputFromPdb(pdb)).toThrow(/SEP at 2/);
  });

  // 🔴 RFD4-Proteina letters the 5O45 binder design B, target A, and the refold letters them in
  // file order - so pairing by letter would lay the target's refold over the binder.
  it("pairs the refold's chains by order, not letter, and finds what moved", () => {
    const design = [...chain("B", ["MET", "LYS", "GLY", "ALA"]), ...chain("A", ["SER", "THR", "VAL", "LEU", "ILE"], [0, 10, 0])].join("\n");
    // the refold: same molecules, letters A then B, shifted as a whole and its last binder residue moved 5 A
    const binder = chain("A", ["MET", "LYS", "GLY", "ALA"], [20, 0, 0]);
    binder[3] = atom("ATOM", 4, "CA", "ALA", "A", 4, 3 * 3.8 + 20, 1.5, 5);
    const refold = [...binder, ...chain("B", ["SER", "THR", "VAL", "LEU", "ILE"], [20, 10, 0])].join("\n");
    const out = refoldOverlay(design, refold, "B");
    // in the design's letters: the binder is B again
    expect(out.refold.split("\n").filter((line) => line.startsWith("ATOM")).map((line) => line[21]).join(""))
      .toBe("BBBBAAAAA");
    expect(out.deviations.length).toBe(9);
    const far = out.deviations.filter((d) => d.distance > 1);
    // fitted on the target, so only the binder's moved residue is off - by its 5 A
    expect(far.map((d) => d.index)).toEqual([3]);
    expect(Math.abs(far[0].distance - 5) < 1e-2).toBe(true);
  });

  it("refuses a refold with a different chain count", () => {
    const design = [...chain("A", ["MET", "LYS"]), ...chain("B", ["GLY", "ALA"], [0, 9, 0])].join("\n");
    expect(() => refoldOverlay(design, chain("A", ["MET", "LYS"]).join("\n"))).toThrow(/1 polymer chains where the design has 2/);
  });
});

describe("refolding a design of two new chains", () => {
  it("holds every designed chain out of the fit", () => {
    const design = [...chain("A", ["MET", "LYS", "GLY"]), ...chain("B", ["ALA", "SER", "THR"], [0, 9, 0]),
      ...chain("C", ["SER", "THR", "VAL", "LEU"], [0, 20, 0])].join("\n");
    // the refold: the kept chain C in place, both designed chains 6 A off
    const refold = [...chain("A", ["MET", "LYS", "GLY"], [0, 0, 6]), ...chain("B", ["ALA", "SER", "THR"], [0, 9, 6]),
      ...chain("C", ["SER", "THR", "VAL", "LEU"], [0, 20, 0])].join("\n");
    const out = refoldOverlay(design, refold, ["A", "B"]);
    const moved = out.deviations.filter((d) => d.distance > 1).map((d) => Math.round(d.distance));
    expect(moved).toEqual([6, 6, 6, 6, 6, 6]);
  });
});

describe("templating a refold on what the job kept", () => {
  it("templates each kept protein chain and none of the designed ones", () => {
    const design = [...chain("B", ["MET", "LYS", "GLY"]), ...chain("A", ["SER", "THR", "VAL"], [0, 10, 0]),
      ...chain("C", ["DA", "DC"], [0, 20, 0], "ATOM", "C1'"), ...chain("D", ["SAM"], [0, 0, 9], "HETATM", "C1")].join("\n");
    // B is designed; A is the target (polymer 1); the DNA and the ligand are not templates
    expect(keptTemplates(design, "B").map(({ chainId, chain: at }) => [chainId, at])).toEqual([["A", 1]]);
    expect(keptTemplates(design, ["A", "B"])).toEqual([]);
  });

  it("refuses more kept chains than a fold has slots", () => {
    const design = ["A", "B", "C", "D", "E", "F"].map((letter, at) => chain(letter, ["MET", "LYS"], [0, at * 9, 0]))
      .flat().join("\n");
    expect(() => keptTemplates(design, "A")).toThrow(/keeps 5 protein chains/);
  });
});
