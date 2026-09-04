import { describe, expect, it } from "./harness.js";
import {
  GAP_AATYPE, chainResidues, filterByConfidence, identityMap, templateSlot,
} from "../src/af3/template-input.js";
import { NUM_DENSE } from "../src/af3/template-features.js";
import { aatypeFor } from "../src/af3/reference-conformers.js";

/**
 * `tools/oracle/check_af3_template_geometry.js` holds this to AF3 itself -
 * given TEMPLATE_PDB it builds a slot from the same file the dump was made
 * from and reports all four geometry features bit-exact. That needs a CPU jax
 * run; these are the parts that can go wrong without one.
 */
function line(serial, name, resName, chain, resSeq, iCode, x, y, z) {
  const out = new Array(54).fill(" ");
  const put = (start, text) => {
    for (let index = 0; index < text.length; index += 1) out[start + index] = text[index];
  };
  put(0, "ATOM  ");
  put(11 - String(serial).length, String(serial));
  put(12, name.padEnd(4));
  put(20 - resName.length, resName);
  put(21, chain);
  put(26 - String(resSeq).length, String(resSeq));
  put(26, iCode);
  put(30, x.toFixed(3).padStart(8));
  put(38, y.toFixed(3).padStart(8));
  put(46, z.toFixed(3).padStart(8));
  return out.join("") + "  1.00 80.00           " + name[0];
}

/** A chain whose residue NUMBERS have a hole in them, as a real one does. */
function withGap() {
  const lines = [];
  let serial = 0;
  for (const number of [10, 11, 12, 40, 41]) {
    for (const [index, name] of ["N", "CA", "C", "O", "CB"].entries()) {
      lines.push(line(serial += 1, name, "ALA", "A", number, " ",
                      number + index * 0.1, index, 0));
    }
  }
  // A second chain, to prove the selector works.
  lines.push(line(serial += 1, "CA", "GLY", "B", 1, " ", 99, 99, 99));
  return `${lines.join("\n")}\nTER\nEND\n`;
}

describe("chainResidues", () => {
  it("groups by residue number and keeps file order", () => {
    const structure = chainResidues(withGap(), "A");
    expect(structure.residues).toHaveLength(5);
    expect(structure.residues.map((residue) => residue.number))
      .toEqual(["10", "11", "12", "40", "41"]);
    expect(structure.sequence).toBe("AAAAA");
  });

  // 🔴 THE NUMBER, NOT THE POSITION. A structure missing residues 13-39 has no
  // lines for them, so grouping by position in the atom list closes the gap up
  // and every residue after it is numbered 27 too low. The numbers are kept so
  // a caller can see the hole; nothing here silently renumbers.
  it("keeps the hole visible rather than closing it up", () => {
    const numbers = chainResidues(withGap(), "A").residues.map((r) => Number(r.number));
    expect(numbers[3] - numbers[2]).toBe(28);
  });

  it("takes the chain it is asked for, and the first one by default", () => {
    expect(chainResidues(withGap(), "B").sequence).toBe("G");
    expect(chainResidues(withGap()).chain).toBe("A");
  });

  it("reads an unknown residue name as X rather than dropping it", () => {
    const odd = `${line(1, "CA", "SEP", "A", 1, " ", 0, 0, 0)}\nEND\n`;
    expect(chainResidues(odd, "A").sequence).toBe("X");
  });
});

describe("templateSlot", () => {
  const structure = chainResidues(withGap(), "A");

  // 🔴 MEASURED, NOT ASSUMED. An AF3 dump whose template covers 8 of 16 query
  // residues carries aatype 21 - the GAP token - at the other eight, not 0.
  // Aatype 0 is ALA, and the aatype features are read whether or not there is
  // geometry, so an uncovered position would contribute alanine's embedding.
  it("fills uncovered query positions with the gap token", () => {
    const slot = templateSlot({ structure, tokens: 8, map: new Map([[0, 0], [1, 1]]) });
    expect(slot.aatype[0]).toBe(aatypeFor("A"));
    expect(slot.covered).toBe(2);
    for (let token = 2; token < 8; token += 1) expect(slot.aatype[token]).toBe(GAP_AATYPE);
    expect(GAP_AATYPE).toBe(21);
  });

  it("leaves an uncovered position with no atoms at all", () => {
    const slot = templateSlot({ structure, tokens: 4, map: new Map([[0, 0]]) });
    for (let index = NUM_DENSE; index < 4 * NUM_DENSE; index += 1) {
      expect(slot.atomMask[index]).toBe(0);
    }
  });

  // The dense layout is `conformerFor`'s own atom order, which is the table
  // featurise.js builds the QUERY's atoms from - so a template and the query it
  // is shown against cannot end up in different layouts.
  it("puts each atom in its conformer's slot: N, CA, C, O, CB", () => {
    const slot = templateSlot({ structure, tokens: 1, map: new Map([[0, 0]]) });
    for (let dense = 0; dense < 5; dense += 1) expect(slot.atomMask[dense]).toBe(1);
    expect(slot.atomMask[5]).toBe(0);
    // Residue 10's atoms were written at x = 10 + slot/10.
    for (let dense = 0; dense < 5; dense += 1) {
      expect(slot.atomPositions[dense * 3]).toBeCloseTo(10 + dense * 0.1, 3);
    }
    expect(slot.atoms).toBe(5);
  });

  // A slot is a MEANING. An atom the conformer does not name would otherwise
  // take whatever index it happened to fall at, putting a side chain where a
  // backbone atom belongs.
  it("drops an atom the residue's conformer does not name", () => {
    const withOxt = `${line(1, "N", "ALA", "A", 1, " ", 0, 0, 0)}\n`
      + `${line(2, "CA", "ALA", "A", 1, " ", 1, 0, 0)}\n`
      + `${line(3, "OXT", "ALA", "A", 1, " ", 2, 0, 0)}\nEND\n`;
    const slot = templateSlot({
      structure: chainResidues(withOxt, "A"), tokens: 1, map: new Map([[0, 0]]),
    });
    expect(slot.atoms).toBe(2);
    expect(slot.atomMask[0]).toBe(1);
    expect(slot.atomMask[1]).toBe(1);
    expect(slot.atomMask[2]).toBe(0);
  });

  // A complex's arrays run over every chain, and a template covers one of them.
  it("places a chain's residues at its own offset", () => {
    const slot = templateSlot({ structure, tokens: 8, map: new Map([[0, 0]]), offset: 5 });
    expect(slot.aatype[5]).toBe(aatypeFor("A"));
    expect(slot.aatype[0]).toBe(GAP_AATYPE);
  });

  it("ignores a mapping that points outside the query or the structure", () => {
    const slot = templateSlot({
      structure, tokens: 2, map: new Map([[0, 0], [9, 1], [1, 99]]),
    });
    expect(slot.covered).toBe(1);
  });
});

describe("identityMap and filterByConfidence", () => {
  const structure = chainResidues(withGap(), "A");

  it("maps residue for residue when the sequence came from the structure", () => {
    expect([...identityMap(structure)]).toEqual([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]);
  });

  // AlphaFold DB has every residue and no way to say "I did not see this"; a
  // disordered tail at pLDDT 30 is noise the model would be told to believe.
  it("drops residues below the confidence floor, and is a no-op at zero", () => {
    const confidence = (residue) => (Number(residue.number) < 20 ? 90 : 30);
    const map = identityMap(structure);
    expect([...filterByConfidence(map, structure, 70, confidence)])
      .toEqual([[0, 0], [1, 1], [2, 2]]);
    expect(filterByConfidence(map, structure, 0, confidence)).toBe(map);
  });
});
