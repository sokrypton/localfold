/**
 * AF2's atom37 template slot puts each atom where AF2 expects it.
 *
 * 🔴 THE FAILURE THIS GATES IS SILENT AND THE ARRAYS LOOK ALIKE. AF3's dense 24
 * and AF2's atom37 are both `{aatype, atomPositions, atomMask}` of plausible
 * size, and they index atoms by different things: dense by the atom's position
 * in THAT residue's own conformer, atom37 by NAME, globally. So dense slot 3 is
 * whatever the fourth atom of this residue happens to be and atom37 slot 3 is
 * CB for everything that has one. `AF2_ATOM37_MONOMER` reads `pseudoBeta: 3`
 * and `backbone: [2, 1, 0]`; handed a dense slot it reads four wrong atoms per
 * residue, throws nothing, and produces a plausible distogram of the wrong
 * structure.
 *
 * These assertions are about the LAYOUT, not about a fold: they hold for any
 * structure, need no GPU and no bundle, and would have caught the conversion
 * being skipped entirely.
 */
import { describe, it, expect } from "./harness.js";
import { readFileSync } from "node:fs";
import { chainResidues, identityMap, templateSlot, templateSlotAtom37 }
  from "../src/af3/featurise/template-input.js";
import { ATOM37, AF2_ATOM37_MONOMER, NUM_DENSE, packTemplateGeometry, templateGeometry }
  from "../src/af3/featurise/template-features.js";

const CRYSTAL = readFileSync("tools/fixtures/5caj-crystal.pdb", "utf8");
const structure = chainResidues(CRYSTAL, "A");
const tokens = structure.residues.length;
const slot = templateSlotAtom37({ structure, tokens, map: identityMap(structure) });

describe("AF2's atom37 template slot", () => {
  it("is 37 slots wide, not AF3's 24", () => {
    expect(ATOM37.length).toBe(37);
    expect(slot.atomMask.length).toBe(tokens * 37);
    expect(slot.atomPositions.length).toBe(tokens * 37 * 3);
    // ...and the dense builder on the same chain is a different width, which is
    // the whole reason both exist.
    const dense = templateSlot({ structure, tokens, map: identityMap(structure) });
    expect(dense.atomMask.length).toBe(tokens * NUM_DENSE);
  });

  it("puts the atoms AF2_ATOM37_MONOMER names where it names them", () => {
    // The dialect's own indices, read back through the table rather than
    // retyped: a test that hardcodes 3 cannot notice the dialect changing.
    const [carbon, alpha, nitrogen] = AF2_ATOM37_MONOMER.backbone;
    expect(ATOM37[nitrogen]).toBe("N");
    expect(ATOM37[alpha]).toBe("CA");
    expect(ATOM37[carbon]).toBe("C");
    expect(ATOM37[AF2_ATOM37_MONOMER.pseudoBeta]).toBe("CB");
  });

  it("reads each atom from the file into its own named slot", () => {
    let checked = 0;
    for (let token = 0; token < tokens; token += 1) {
      const residue = structure.residues[token];
      for (let index = 0; index < ATOM37.length; index += 1) {
        const point = residue.atoms.get(ATOM37[index]);
        const live = slot.atomMask[token * 37 + index] === 1;
        expect(live).toBe(point !== undefined);
        if (point === undefined) continue;
        const base = (token * 37 + index) * 3;
        // 🔴 THROUGH `Math.fround`, because the store is a Float32Array and the
        // parsed coordinate is a double: -19.598 is not exactly representable,
        // so a plain equality fails on the narrowing rather than on the layout,
        // which is what this is about.
        expect(slot.atomPositions[base]).toBe(Math.fround(point[0]));
        expect(slot.atomPositions[base + 1]).toBe(Math.fround(point[1]));
        expect(slot.atomPositions[base + 2]).toBe(Math.fround(point[2]));
        checked += 1;
      }
    }
    // A loop that checks nothing passes; say how much it saw.
    expect(checked > 1000).toBe(true);
  });

  it("gives a glycine no C-beta and everything else one", () => {
    let glycines = 0, others = 0;
    for (let token = 0; token < tokens; token += 1) {
      const hasBeta = slot.atomMask[token * 37 + AF2_ATOM37_MONOMER.pseudoBeta] === 1;
      if (structure.residues[token].code === "G") { expect(hasBeta).toBe(false); glycines += 1; }
      else if (hasBeta) others += 1;
    }
    expect(glycines > 0).toBe(true);
    expect(others > 100).toBe(true);
  });

  it("reads only the backbone and C-beta, so stripping side chains changes nothing", () => {
    // 🔴 THE FACT AF2BIND RESTS ON. Its weights are named "nosc" because it
    // feeds the target with side chains stripped, and ColabDesign's
    // `rm_target_sc` does that by masking `template_all_atom_mask[..., 5:]`
    // under its own comment "remove sidechains (mask anything beyond CB)" -
    // atom37 slots 0..4 are N, CA, C, CB, O, so **C-beta survives**. The
    // monomer's term then reads the pseudo-beta for its distogram and N, CA, C
    // for its frames and NOTHING ELSE, so the strip is a no-op here. That is
    // worth pinning: it means a p(bind) discrepancy cannot be blamed on it.
    const stripped = templateSlotAtom37({ structure, tokens, map: identityMap(structure) });
    let removed = 0;
    for (let token = 0; token < tokens; token += 1) {
      for (let index = 5; index < 37; index += 1) {
        if (stripped.atomMask[token * 37 + index] === 1) removed += 1;
        stripped.atomMask[token * 37 + index] = 0;
      }
    }
    // A strip that removed nothing would make the rest of this vacuous.
    expect(removed > 500).toBe(true);
    expect(stripped.atomMask[3 * 37 + AF2_ATOM37_MONOMER.pseudoBeta]).toBe(1);

    // 🔴 THE CHAIN MASK IS tokens x tokens. A length-`tokens` one reads
    // undefined past the end, every comparison becomes NaN !== NaN, and this
    // assertion reports 83% of the geometry changing when none of it does.
    const chainMask = new Float32Array(tokens * tokens).fill(1);
    const before = packTemplateGeometry(
      templateGeometry(slot, chainMask, tokens, AF2_ATOM37_MONOMER), tokens);
    const after = packTemplateGeometry(
      templateGeometry(stripped, chainMask, tokens, AF2_ATOM37_MONOMER), tokens);
    let differing = 0, notANumber = 0;
    for (let index = 0; index < before.length; index += 1) {
      if (Number.isNaN(before[index]) || Number.isNaN(after[index])) notANumber += 1;
      else if (before[index] !== after[index]) differing += 1;
    }
    expect(notANumber).toBe(0);
    expect(differing).toBe(0);
  });

  it("leaves an uncovered token at the GAP restype with no atoms", () => {
    // One residue mapped, the rest uncovered - which is what a partial
    // template looks like and what the monomer's masked arm relies on.
    const one = templateSlotAtom37({ structure, tokens, map: new Map([[5, 5]]) });
    expect(one.covered).toBe(1);
    expect(one.aatype[0]).toBe(21);
    for (let index = 0; index < 37; index += 1) {
      expect(one.atomMask[0 * 37 + index]).toBe(0);
    }
    expect(one.atomMask.slice(5 * 37, 6 * 37).some((m) => m === 1)).toBe(true);
  });
});
