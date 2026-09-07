/**
 * The atom stack's shape agrees with the gathers laid out beside it.
 *
 * 🔴 `subsets` WAS DERIVED IN THREE PLACES AND ALL THREE WERE WRONG THE SAME
 * WAY, which is the only reason they agreed: `tokens * DENSE / 32`, counting
 * padded (token, slot) cells where the axis it indexes is the COMPACTED list
 * of real atoms. Fixing one and not the others left a dispatch sized for 51
 * subsets reading gathers built for 18 - this repository's recurring failure,
 * and one that produces plausible output rather than an error, because the
 * extra subsets are masked everywhere they are read.
 *
 * So what is asserted here is the AGREEMENT, not the formula: every gather's
 * length is the shape's own arithmetic, and the shape is the atom count's. A
 * second derivation appearing anywhere fails this the moment it disagrees.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { featuriseProtein } from "../src/af3/featurise.js";
import { structuralBatch } from "../src/af3/structural-tokens.js";

const QUERIES = 32;
const KEYS = 128;

/** Glycerol's heavy atoms, as the featuriser wants a ligand. */
const GLYCEROL = {
  code: "GOL",
  atoms: ["C1", "O1", "C2", "O2", "C3", "O3"].map((name, index) => ({
    name, element: name[0] === "C" ? 6 : 8, charge: 0, x: index, y: 0, z: 0,
  })),
  bonds: [],
};

function assertLayout(name, batch) {
  const { tokens, dense, atomCount, subsets, shape } = batch;

  // The shape counts atoms, and never rounds down to nothing: a lone ligand is
  // a valid fold and six atoms still need one subset.
  assert.equal(subsets, Math.max(1, Math.ceil(atomCount / QUERIES)), `${name}: subsets`);
  assert.ok(subsets * QUERIES >= atomCount, `${name}: subsets do not hold the atoms`);
  assert.equal(shape.subsets, subsets, `${name}: shape disagrees with the batch`);
  assert.equal(shape.queries, QUERIES, `${name}: queries`);

  // 🔴 AND THE WINDOW NEVER EXCEEDS THE MOLECULE, which is what makes a padded
  // KEY impossible - see atomGathers. Below 128 atoms this is the whole of it.
  assert.equal(shape.keys, Math.min(KEYS, atomCount), `${name}: keys`);

  // Every gather is the length its axis says it is.
  assert.equal(batch.tokenAtomsToQueries.indices.length, subsets * QUERIES,
    `${name}: token_atoms_to_queries`);
  assert.equal(batch.tokensToQueries.indices.length, subsets * QUERIES,
    `${name}: tokens_to_queries`);
  assert.equal(batch.queriesToKeys.indices.length, subsets * shape.keys,
    `${name}: queries_to_keys`);
  assert.equal(batch.tokensToKeys.indices.length, subsets * shape.keys,
    `${name}: tokens_to_keys`);
  // ...except this one, which is laid out over the dense grid on purpose: it is
  // the inverse, and it is indexed by (token, slot).
  assert.equal(batch.queriesToTokenAtoms.indices.length, tokens * dense,
    `${name}: queries_to_token_atoms`);
  assert.equal(batch.tokenAtomsToPseudoBeta.indices.length, tokens,
    `${name}: token_atoms_to_pseudo_beta`);

  // A mask is set exactly where an atom is real, on both sides of the map.
  const queryMask = batch.tokenAtomsToQueries.mask;
  let real = 0;
  for (let index = 0; index < queryMask.length; index += 1) real += queryMask[index];
  assert.equal(real, atomCount, `${name}: query mask counts ${real}, not ${atomCount}`);
}

describe("the atom layout's shape is the atom count's", () => {
  const cases = [
    ["one glycine", () => featuriseProtein("G", {})],
    ["a lone ligand", () => featuriseProtein("", { ligands: [GLYCEROL] })],
    ["a ligand beside one residue", () => featuriseProtein("A", { ligands: [GLYCEROL] })],
    ["a 68-mer", () => featuriseProtein(
      "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE", {})],
    ["a two-chain complex", () => featuriseProtein("GAWSTLAK:ACGT",
      { chainKinds: ["protein", "dna"] })],
  ];

  for (const [name, build] of cases) {
    it(`holds for ${name}`, () => {
      assertLayout(name, build());
    });
  }

  /**
   * 🔴 AND THE STRUCTURAL LAYOUT IS WHERE IT MATTERED MOST. OpenDDE folds the
   * same atoms in twice the tokens, so a count taken from `tokens * dense`
   * doubled a number that was already too big - 98 subsets for the 18 that
   * hold an atom. The atom count does not change across the regrouping, so
   * neither does the shape, and that is the statement.
   */
  it("does not change across OpenDDE's regrouping", () => {
    const batch = featuriseProtein(
      "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE", {});
    const structural = structuralBatch(batch);
    assertLayout("structural", structural);
    assert.equal(structural.atomCount, batch.atomCount, "the regrouping moved an atom");
    assert.equal(structural.subsets, batch.subsets, "the same atoms want the same subsets");
    assert.ok(structural.tokens > batch.tokens, "and it really is a second token space");
  });
});
