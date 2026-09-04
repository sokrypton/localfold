import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { DESIGNERS, DESIGNER_NAMES, chooseDesigner } from "../src/design/designers.js";

const path = (name) => fileURLToPath(new URL(name, import.meta.url));

describe("the designer registry", () => {
  // 🔴 A NAME WITH NO FILE IS A 404 THE PAGE CANNOT EXPLAIN, and a file with
  // no name is 4 MB nobody can reach. The two lists are written out in two
  // places on purpose - one is Python and mirrors the weights, the other is JS
  // and drives a dropdown - so something has to hold them together.
  it("names a checkpoint that web/public/mpnn/ actually carries", () => {
    for (const [name, designer] of Object.entries(DESIGNERS)) {
      expect(existsSync(path(`../web/public/mpnn/${designer.file}`))).toBe(true);
      expect(typeof designer.label).toBe("string");
      expect(DESIGNER_NAMES).toContain(name);
    }
  });

  it("carries no checkpoint no designer names", () => {
    const named = new Set(Object.values(DESIGNERS).map((designer) => designer.file));
    const mirrored = readFileSync(path("../tools/sync-mpnn.py"), "utf8");
    const listed = [...mirrored.matchAll(/"([a-z0-9_]+\.mpnn)"/g)].map((match) => match[1]);
    expect(listed.length > 0).toBe(true);
    expect([...named].sort()).toEqual([...listed].sort());
  });

  it("lists every designer exactly once, in a stable order", () => {
    expect([...DESIGNER_NAMES].sort()).toEqual(Object.keys(DESIGNERS).sort());
    expect(DESIGNER_NAMES[0]).toBe("soluble");
  });
});

describe("chooseDesigner", () => {
  // The reference's own rule: `ligand_mpnn if (ligand or nucleic) else
  // soluble_mpnn`. The one addition is NA-MPNN for a nucleic chain with no
  // ligand, which the reference would send to LigandMPNN as loose atoms.
  it("takes SolubleMPNN for a protein-only complex", () => {
    expect(chooseDesigner({}).name).toBe("soluble");
    expect(chooseDesigner({ ligands: 0, nucleic: 0 }).name).toBe("soluble");
  });

  it("takes LigandMPNN when there is a ligand", () => {
    expect(chooseDesigner({ ligands: 1 }).name).toBe("ligand");
  });

  it("takes NA-MPNN for a nucleic chain with no ligand", () => {
    expect(chooseDesigner({ nucleic: 1 }).name).toBe("na");
  });

  // Each choice loses something real here, so the tie-break is a decision
  // rather than a fallthrough and is worth pinning.
  it("takes LigandMPNN when there is both, and says why", () => {
    const chosen = chooseDesigner({ ligands: 1, nucleic: 2 });
    expect(chosen.name).toBe("ligand");
    expect(chosen.why).toContain("heteroatoms");
  });

  it("always explains itself", () => {
    for (const input of [{}, { ligands: 1 }, { nucleic: 1 }, { ligands: 1, nucleic: 1 }]) {
      expect(chooseDesigner(input).why.length > 0).toBe(true);
    }
  });
});
