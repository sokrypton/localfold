import { describe, it } from "node:test";
import { strict as assert } from "node:assert";

import { parseSmiles, molecularFormula, adjacency } from "../src/chem/smiles.js";
import { kekulize, valenceProblems } from "../src/chem/kekulize.js";
import { smallestRings, circuitRank } from "../src/chem/rings.js";

/**
 * 🔴 THESE ARE THE INTERNAL-CONSISTENCY HALF AND THEY ARE NOT THE GATE. A
 * parser can be consistently wrong - the first version of this one gave
 * caffeine three hydrogens it does not have and passed everything here. The
 * gate is `node tools/check-smiles-vs-rdkit.mjs`, which holds all 51 corpus
 * molecules to RDKit's own answer. What lives here is what a CPU suite with no
 * RDKit and no GPU can still check: the syntax, the refusals, and the
 * invariants that hold for every molecule whatever the answer.
 */

describe("the SMILES parser", () => {
  it("reads the organic subset, two-letter atoms included", () => {
    const graph = parseSmiles("CCBr");
    assert.deepEqual(graph.atoms.map((atom) => atom.symbol), ["C", "C", "BR"]);
    // 🔴 `Cl` AND `Br` MUST BE TRIED BEFORE `C` AND `B` or "CCBr" reads as
    // carbon, carbon, boron, and then an unknown "r".
    assert.deepEqual(parseSmiles("ClC").atoms.map((a) => a.symbol), ["CL", "C"]);
  });

  it("reads a bracket atom's every field", () => {
    const [atom] = parseSmiles("[13CH3-]").atoms;
    assert.equal(atom.symbol, "C");
    assert.equal(atom.isotope, 13);
    assert.equal(atom.hydrogens, 3);
    assert.equal(atom.charge, -1);
    assert.equal(atom.bracket, true);
  });

  it("reads both charge spellings", () => {
    assert.equal(parseSmiles("[Mg+2]").atoms[0].charge, 2);
    assert.equal(parseSmiles("[Mg++]").atoms[0].charge, 2);
    assert.equal(parseSmiles("[O-]").atoms[0].charge, -1);
    assert.equal(parseSmiles("[Fe+3]").atoms[0].charge, 3);
  });

  it("closes rings, including two-digit ones", () => {
    assert.equal(smallestRings(parseSmiles("C1CCCCC1")).length, 1);
    assert.equal(smallestRings(parseSmiles("C%10CCCCC%10")).length, 1);
  });

  it("takes the bond symbol from either end of a ring closure", () => {
    assert.equal(parseSmiles("C=1CCCCC=1").bonds.filter((b) => b.order === 2).length, 1);
    assert.equal(parseSmiles("C=1CCCCC1").bonds.filter((b) => b.order === 2).length, 1);
    assert.equal(parseSmiles("C1CCCCC=1").bonds.filter((b) => b.order === 2).length, 1);
  });

  it("separates fragments on a dot", () => {
    const graph = parseSmiles("[Na+].[Cl-]");
    assert.equal(graph.atoms.length, 2);
    assert.equal(graph.bonds.length, 0);
    assert.equal(new Set(graph.components).size, 2);
  });

  it("refuses what it cannot read, by name", () => {
    // 🔴 EACH OF THESE IS A REFUSAL RATHER THAN A GUESS, which is the rule
    // this repository applies to a convention it has not implemented: a
    // ligand read wrongly folds and scores, and the number is merely
    // different.
    assert.throws(() => parseSmiles("C1CC"), /never closed/);
    assert.throws(() => parseSmiles("C(CC"), /`\(` with no `\)`/);
    assert.throws(() => parseSmiles("CC)"), /`\)` with no `\(`/);
    assert.throws(() => parseSmiles("C==C"), /two bond symbols/);
    assert.throws(() => parseSmiles("=CC"), /at the start/);
    assert.throws(() => parseSmiles("CC="), /at the end/);
    assert.throws(() => parseSmiles("*"), /any atom/);
    assert.throws(() => parseSmiles("[Xx]"), /not an element/);
    assert.throws(() => parseSmiles("C>>C"), /reaction/);
    assert.throws(() => parseSmiles(""), /empty/);
    // 🔴 A GENUINE DOUBLE BOND-CLOSURE, WHICH `C12CC1C2` IS NOT: that one
    // opens two rings on atom 0 and closes them on two DIFFERENT atoms, which
    // is bicyclo[1.1.0]butane and perfectly legal. `C12CCC12` closes both on
    // the same atom, which asks for the same bond twice.
    assert.throws(() => parseSmiles("C12CCC12"), /bonded twice/);
    // ...and an extended stereo class is named rather than flattened to `@`.
    assert.throws(() => parseSmiles("[C@TB1](F)(Cl)(Br)(I)C"), /not supported/);
  });

  it("gives every atom a valence it can have, across the corpus", () => {
    // The invariant that holds whatever the kekulisation picked.
    for (const smiles of [
      "c1ccccc1", "c1ccncc1", "c1cc[nH]c1", "c1ccoc1", "c1cnc[nH]1",
      "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O",
      "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
      "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12",
      "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3",
    ]) {
      assert.deepEqual(valenceProblems(parseSmiles(smiles)), [],
                       `${smiles} does not add up`);
    }
  });

  it("counts the hydrogens an aromatic nitrogen actually has", () => {
    // 🔴 THE BUG THAT MADE KEKULISATION COME FIRST. An N-methyl aromatic
    // nitrogen sums to 1.5 + 1.5 + 1 = 4 over aromatic bonds, which is past
    // nitrogen's 3, so the valence rule promotes it to 5 and hands it a
    // hydrogen. Kekulised it has two single ring bonds, sums to 3, and takes
    // none. Caffeine read C8H13N4O2 against RDKit's C8H10N4O2.
    assert.equal(molecularFormula(parseSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")),
                 "C8H10N4O2");
    assert.equal(molecularFormula(parseSmiles("c1cc[nH]c1")), "C4H5N");   // pyrrole
    assert.equal(molecularFormula(parseSmiles("c1ccncc1")), "C5H5N");     // pyridine
  });

  it("refuses an aromatic ring that cannot be kekulised", () => {
    // Five aromatic carbons in a ring: one of them can never be paired.
    assert.throws(() => parseSmiles("c1cccc1"), /cannot be kekulised/);
  });

  it("counts hydrogens from the valence, charge included", () => {
    assert.equal(parseSmiles("C").atoms[0].hydrogens, 4);
    assert.equal(parseSmiles("O").atoms[0].hydrogens, 2);
    assert.equal(parseSmiles("[NH4+]").atoms[0].hydrogens, 4);
    // Sulfur takes the lowest valence not exceeded: 2, then 4, then 6.
    assert.equal(parseSmiles("CSC").atoms[1].hydrogens, 0);
    assert.equal(parseSmiles("CS(C)=O").atoms[1].hydrogens, 0);
    // 🔴 AN ATOM OUTSIDE THE ORGANIC SUBSET HAS NO IMPLICIT HYDROGENS AT ALL,
    // which is the specification rather than chemistry: it must say so.
    assert.equal(parseSmiles("[Fe]").atoms[0].hydrogens, 0);
    assert.equal(parseSmiles("[FeH2]").atoms[0].hydrogens, 2);
  });
});

describe("ring perception", () => {
  it("finds as many rings as the circuit rank", () => {
    for (const [smiles, expected] of [
      ["C1CCCCC1", 1], ["c1ccc2ccccc2c1", 2],
      ["C1CCC2(CC1)CCCC2", 2], ["CCO", 0],
    ]) {
      const graph = parseSmiles(smiles);
      assert.equal(circuitRank(graph), expected, `${smiles} rank`);
      assert.equal(smallestRings(graph).length, expected, `${smiles} rings`);
    }
  });

  it("keeps a symmetric ring past the rank, which a cage needs", () => {
    // 🔴 BICYCLO[2.2.2]OCTANE HAS A RANK OF TWO AND THREE EQUAL SIX-RINGS, and
    // RDKit's `GetSymmSSSR` returns all three. A test expecting the rank here
    // is testing the wrong thing - see the note in rings.js.
    const bridged = parseSmiles("C1CC2CCC1CC2");
    assert.equal(circuitRank(bridged), 2);
    assert.equal(smallestRings(bridged).length, 3);
  });

  it("keeps a symmetric ring past the rank, which adamantane needs", () => {
    // 🔴 ADAMANTANE'S RANK IS THREE AND IT HAS FOUR EQUAL SIX-RINGS. No three
    // of them is more correct than another, and the fourth is a GEOMETRY
    // constraint: dropped, one face of the cage is left unconstrained.
    const graph = parseSmiles("C1C2CC3CC1CC(C2)C3");
    assert.equal(circuitRank(graph), 3);
    assert.equal(smallestRings(graph).length, 4);
    assert.ok(smallestRings(graph).every((ring) => ring.length === 6));
  });

  it("finds the larger ring of a fused pair, which an atom search misses", () => {
    // 🔴 THE CANDIDATE SEARCH IS PER BOND, NOT PER ATOM. Indole's bridgeheads
    // sit on both rings, so the shortest cycle through either atom is the
    // five every time and the six is never a candidate.
    const sizes = smallestRings(parseSmiles("c1ccc2[nH]ccc2c1"))
      .map((ring) => ring.length).sort();
    assert.deepEqual(sizes, [5, 6]);
  });
});

describe("the graph", () => {
  it("builds neighbour lists that agree with the bonds", () => {
    const graph = parseSmiles("CC(=O)O");
    const lists = adjacency(graph);
    assert.equal(lists.reduce((total, list) => total + list.length, 0),
                 graph.bonds.length * 2);
    assert.deepEqual(lists[1].map((step) => step.atom).sort(), [0, 2, 3]);
  });

  it("writes a formula in Hill order", () => {
    assert.equal(molecularFormula(parseSmiles("OCC(O)CO")), "C3H8O3");
    assert.equal(molecularFormula(parseSmiles("[Cl-]")), "Cl");
    assert.equal(molecularFormula(parseSmiles("O")), "H2O");
  });
});
