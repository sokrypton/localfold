// Which partners a token's certainty is averaged over - the rule a ligand broke.
//
// 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM, so the old rule - exclude a partner
// when the TOKEN INDICES are within `separation` - meant nothing there. Two
// consequences, both measured on ubiquitin plus ATP (76 residues, 31 atoms) and
// both reproduced by folding rather than argued:
//
// * The ligand's whole self-block was averaged in. Its atoms are a few
//   angstroms apart and the model is handed their geometry as the reference
//   conformer, so it predicts them sharply - and the ligand read certainty
//   0.8173 off its own conformer while the distogram predicted NO
//   protein-ligand contact at all (0 predicted against 64 the structure makes).
//   Under this rule it reads 0.2193, which is the same model being honest.
// * 62% of the fold's reported contacts (195 of 314) were the ligand's own
//   pairs, so the precision and recall a checker prints were mostly a statement
//   about ATP's internal geometry.
//
// 🔴 AND IT IS THE OLD RULE EXACTLY FOR ONE UNMODIFIED PROTEIN CHAIN, which is
// what every measurement behind CERTAINTY's three constants was made on. That
// is the gate this file exists for: a fold of ubiquitin alone comes back with
// the certainty vector IDENTICAL to every digit, before and after.
import { describe, expect, it } from "./harness.js";
import {
  CERTAINTY, contactAngstromsFor, contactBinCountsByPair, createCertaintyShader,
  partnerKeys,
} from "../src/esmfold2/distogram-webgpu.js";

describe("the certainty's partner rule", () => {
  it("carries the asym id and the residue number, one pair per token", () => {
    // A two-residue chain, then a three-atom ligand: AF3's featuriser gives
    // every atom of a component the same residue number.
    const keys = partnerKeys({ asymId: [0, 0, 1, 1, 1],
                               residueIndex: [0, 1, 0, 0, 0],
                               molType: [0, 0, 3, 3, 3] }, 5);
    // asym, residue, chemistry (0 protein, 1 nucleic, 2 ligand), unused.
    expect(Array.from(keys)).toEqual([0, 0, 0, 0, 0, 1, 0, 0,
                                      1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0]);
  });

  it("does not apply a sequence rule to something with no sequence", () => {
    // 🔴 A LIGAND'S ATOMS ALL CARRY ONE RESIDUE NUMBER, so a rule phrased in
    // residues excludes every pair inside it - and a heme folded ALONE then has
    // no surviving pair at all, 43 tokens every one of them "no data", beside a
    // contact map that is confident about the molecule. The rule is about a
    // polymer's backbone and applies only where both ends are polymer.
    const keys = partnerKeys({ asymId: [0, 0, 1, 1, 1],
                               residueIndex: [0, 1, 0, 0, 0],
                               molType: [0, 0, 3, 3, 3] }, 5);
    const polymer = (t) => keys[t * 4 + 2] !== 2;
    const excluded = (i, j) => polymer(i) && polymer(j)
      && keys[i * 4] === keys[j * 4]
      && Math.abs(keys[i * 4 + 1] - keys[j * 4 + 1]) <= CERTAINTY.separation;
    // tokens 2-4 are one ligand: not excluded by the sequence rule, only by
    // bonds, which the shader applies from the token bond matrix.
    for (const [i, j] of [[2, 3], [2, 4], [3, 4]]) expect(excluded(i, j)).toBe(false);
    // ...and two residues of one chain, one apart, still are.
    expect(excluded(0, 1)).toBe(true);
  });

  it("is the token-index rule on one unmodified protein chain", () => {
    const tokens = 40;
    const keys = partnerKeys({
      asymId: new Int32Array(tokens),
      residueIndex: Int32Array.from({ length: tokens }, (_, t) => t),
      molType: new Int32Array(tokens),
    }, tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const byResidue = keys[i * 4] === keys[j * 4]
          && Math.abs(keys[i * 4 + 1] - keys[j * 4 + 1]) <= CERTAINTY.separation;
        expect(byResidue).toBe(Math.abs(i - j) <= CERTAINTY.separation);
      }
    }
  });

  it("reads the rule from a buffer, so the shader cannot regrow the old one", () => {
    // 🔴 ASSERT ON THE GENERATED WGSL. A partner rule that never reaches the
    // kernel reports agreement with itself; the shape of the mistake this file
    // is about is a token index arriving where a residue number was meant.
    const wgsl = createCertaintyShader({
      tokens: 8, separation: 3,
      cutoffBins: { protein: 25, nucleic: 56, ligand: 25 } });
    expect(wgsl).toContain("partner: array<vec4<i32>>");
    expect(wgsl).toContain("here.x == there.x");
    expect(wgsl).toContain("abs(here.y - there.y) <= 3");
    expect(wgsl.includes("other > token")).toBe(false);
    // 🔴 A LIGAND IS SCORED, NEVER SCORING - AF3's own lDDT admits only protein
    // and nucleotide atoms as the partner index. Without this the rule needs a
    // fallback for the ligand that has no partner of its own.
    // 🔴 AND BONDED PAIRS ARE EXCLUDED, WHICH IS THE ONLY EXCLUSION A LIGAND
    // CAN HAVE. Sequence separation needs a sequence; a covalently attached
    // ligand sits at a fixed bond length from its residue, as uninformative as
    // an i+1 neighbour and previously counted as a confident prediction.
    expect(wgsl).toContain("bonded[cell_bond] > 0.0");
    // ...and the reach is the PARTNER's, not the pair's.
    expect(wgsl).toContain("there.z == 1) { reach = 56.0");
    // ...and nothing refuses to score on chemistry: a ligand folded ALONE has
    // only its own atoms, and excluding them left every token at -1.
    expect(wgsl.includes("if (there.z == 2) { continue; }")).toBe(false);
    // ...and nothing falls back to an unfiltered mean any more.
    expect(wgsl.includes("loose")).toBe(false);
  });
});

describe("the contact threshold", () => {
  const protein = 0, dna = 1, ligand = 3;
  const ALA = 2, ARG = 3, GLY = 9, TRP = 19;

  it("is the pseudo-beta convention between two residues, and only there", () => {
    expect(contactAngstromsFor(protein, protein, ALA, ARG)).toBe(8);
    expect(contactAngstromsFor(dna, protein, 0, ALA)).toBe(10);
    expect(contactAngstromsFor(dna, dna, 0, 0)).toBe(9);
    // 🔴 TWO LIGAND ATOMS ARE EXACT, not approximate: the representative IS the
    // heavy atom, so 5 A is the definition of contact rather than a proxy for
    // it. That holds inside one molecule and between two.
    expect(contactAngstromsFor(ligand, ligand, 0, 0)).toBe(5);
  });

  it("asks the residue, when one end is a ligand", () => {
    // Ordered by how far the side chain reaches past its own pseudo-beta -
    // measured, not assumed; see tools/calibrate-contact-cutoff.py.
    expect(contactAngstromsFor(ligand, protein, 0, GLY)).toBe(5);
    expect(contactAngstromsFor(ligand, protein, 0, ALA)).toBe(5);
    expect(contactAngstromsFor(ligand, protein, 0, TRP)).toBe(7);
    expect(contactAngstromsFor(ligand, protein, 0, ARG)).toBe(8);
    // ...and either way round, because a pair has no order.
    expect(contactAngstromsFor(protein, ligand, ARG, 0)).toBe(8);
  });

  it("falls back to the kind's number for a residue it does not know", () => {
    // An unknown or modified residue, and the nucleotides, have no side chain
    // in the table - guessing one would be a claim.
    expect(contactAngstromsFor(ligand, protein, 0, 22)).toBe(7);
  });

  it("gives every pair a bin count, and a ligand pair fewer", () => {
    const molType = Int32Array.from([protein, protein, ligand, ligand]);
    const residueType = Int32Array.from([ARG, GLY, 0, 0]);
    const counts = contactBinCountsByPair(molType, residueType, 4, 128);
    expect(counts.length).toBe(16);
    const at = (i, j) => counts[i * 4 + j];
    expect(at(0, 1)).toBeGreaterThan(at(2, 3));      // 8 A against 5 A
    expect(at(0, 2)).toBeGreaterThan(at(1, 2));      // ARG reaches, GLY does not
    expect(at(0, 2)).toBe(at(2, 0));
  });
});
