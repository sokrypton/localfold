// What "in contact" means, given what the two tokens are.
//
// 🔴 EIGHT ANGSTROMS IS A PSEUDO-BETA CONVENTION. It is calibrated for a pair
// where BOTH ends' representative atoms stand in for a side chain's reach, and
// AF3 and ESMFold2 both tokenise a ligand one heavy atom at a time - where the
// representative IS the atom and stands in for nothing.
// tools/calibrate-contact-cutoff.py measures the replacement against real
// depositions; this pins the plumbing, which is where it can silently go wrong.
import { describe, expect, it } from "./harness.js";
import {
  CLASS_LIGAND, CLASS_NUCLEIC, CLASS_PROTEIN, LIGAND_PROTEIN_ANGSTROMS,
  PSEUDO_BETA_RESIDUES, contactAngstromsForClasses, contactBinsByPair,
  contactClass,
} from "../src/heads/contact-threshold.js";
import { af3ContactBins, af3ContactClasses } from "../src/af3/contact-classes.js";

describe("the contact threshold", () => {
  it("has one number for each of the twenty, and they span the side chains", () => {
    for (const name of PSEUDO_BETA_RESIDUES) {
      expect(typeof LIGAND_PROTEIN_ANGSTROMS[name]).toBe("number");
    }
    // Glycine has no side chain past its representative and arginine has the
    // longest; measured, not assumed - see the module's own table.
    expect(LIGAND_PROTEIN_ANGSTROMS.GLY).toBe(5);
    expect(LIGAND_PROTEIN_ANGSTROMS.ARG).toBe(8);
  });

  it("asks the residue only when the other end is a ligand", () => {
    const arg = contactClass("protein", "ARG");
    const gly = contactClass("protein", "GLY");
    expect(contactAngstromsForClasses(CLASS_LIGAND, arg)).toBe(8);
    expect(contactAngstromsForClasses(CLASS_LIGAND, gly)).toBe(5);
    expect(contactAngstromsForClasses(arg, gly)).toBe(8);       // both residues
    expect(contactAngstromsForClasses(arg, CLASS_NUCLEIC)).toBe(10);
    expect(contactAngstromsForClasses(CLASS_LIGAND, CLASS_LIGAND)).toBe(5);
  });

  it("falls back to the kind for a residue with no measured reach", () => {
    // An unknown or modified residue has no entry, and inventing one would be
    // a claim about a side chain nobody measured.
    expect(contactClass("protein", "SEP")).toBe(CLASS_PROTEIN);
    expect(contactAngstromsForClasses(CLASS_LIGAND, CLASS_PROTEIN)).toBe(7);
  });
});

describe("AF3's tokens as contact classes", () => {
  // AF3's restype: 0-19 the amino acids, 20 X, 21 gap, then RNA and DNA.
  const ARG = 1, GLY = 7, UNK = 20, RNA_A = 22;

  it("tells a ligand atom from an unknown residue, which share an aatype", () => {
    // 🔴 THE ALPHABET CANNOT SAY. featurise.js writes UNK_AATYPE for every
    // ligand atom AND for an X in a protein chain, so only `ligandSpans`
    // separates them - and getting it wrong gives a ligand a 7 A protein
    // threshold, which conforms and is wrong.
    const batch = {
      aatype: Int32Array.from([ARG, UNK, UNK, UNK, GLY]),
      ligandSpans: [{ from: 2, count: 2, code: "ATP" }],
    };
    const classes = af3ContactClasses(batch, 5);
    expect(classes[1]).toBe(CLASS_PROTEIN);           // an X, not a ligand
    expect(classes[2]).toBe(CLASS_LIGAND);
    expect(classes[3]).toBe(CLASS_LIGAND);
    expect(classes[4]).toBe(contactClass("protein", "GLY"));
  });

  it("puts AF3's restype on PSEUDO_BETA_RESIDUES with no offset", () => {
    // 🔴 ASSERTED, NOT TRUSTED. AF3's `ARNDCQEGHILKMFPSTWYV` is one-letter
    // alphabetical and ALSO three-letter alphabetical, which is why one table
    // serves both models. A silently permuted alphabet conforms in shape.
    const aatype = Int32Array.from(PSEUDO_BETA_RESIDUES.map((_, at) => at));
    const classes = af3ContactClasses({ aatype, ligandSpans: [] },
                                      PSEUDO_BETA_RESIDUES.length);
    PSEUDO_BETA_RESIDUES.forEach((name, at) => {
      expect(classes[at]).toBe(contactClass("protein", name));
    });
  });

  it("reads a nucleotide as nucleic", () => {
    const classes = af3ContactClasses(
      { aatype: Int32Array.from([RNA_A, 29]), ligandSpans: [] }, 2);
    expect(Array.from(classes)).toEqual([CLASS_NUCLEIC, CLASS_NUCLEIC]);
  });

  it("counts bins by AF3's TOP-edge rule, extrapolating the open last bin", () => {
    // Breaks describe one more bin than they have entries, so the final bin's
    // top is one spacing past the last break rather than absent.
    const breaks = Float32Array.from([2, 4, 6, 8, 10]);
    const arg = contactClass("protein", "ARG");
    const gly = contactClass("protein", "GLY");
    const classes = Int32Array.from([CLASS_LIGAND, arg, gly]);
    const bins = af3ContactBins(classes, 3, breaks);
    const at = (i, j) => bins[i * 3 + j];
    // ARG against the ligand is 8 A: bins with top 2, 4, 6, 8.
    expect(at(0, 1)).toBe(4);
    // GLY is 5 A: tops 2 and 4.
    expect(at(0, 2)).toBe(2);
    // Two residues take the convention, 8 A, whatever they are.
    expect(at(1, 2)).toBe(4);
    expect(at(0, 1)).toBe(at(1, 0));
  });

  it("gives every pair a count, and a symmetric one", () => {
    const classes = Int32Array.from(
      [CLASS_LIGAND, CLASS_NUCLEIC, contactClass("protein", "TRP")]);
    const bins = contactBinsByPair(classes, 3, (a) => Math.round(a));
    expect(bins.length).toBe(9);
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) expect(bins[i * 3 + j]).toBe(bins[j * 3 + i]);
    }
  });
});
