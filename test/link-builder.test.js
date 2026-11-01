/**
 * The bond/contact/pocket/crosslink pickers (web/link-builder.js) write the same spec text the box took, and offer the
 * chains under the letters expandEntities resolves them to - a picker that lettered chains its own way would bond
 * something other than what it showed.
 */
import assert from "node:assert/strict";
import { describe, it } from "./harness.js";
import { chainLetter, entityProblem, expandEntities, rowChains } from "../web/entities.js";
import { linkParts, linkText } from "../web/link-builder.js";

const SEQ = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";

describe("the link pickers' chains", () => {
  it("are lettered A..Z, then AA", () => {
    assert.deepEqual([0, 1, 25, 26, 27, 51, 52].map(chainLetter), ["A", "B", "Z", "AA", "AB", "AZ", "BA"]);
  });

  it("put every polymer copy before every ligand, as the fold letters them", () => {
    const rows = [
      { type: "ligand", value: "GOL", copies: 1 },
      { type: "protein", value: SEQ, copies: 2 },
      { type: "bond", value: "", copies: 1 },
      { type: "dna", value: "ACGT", copies: 1 },
    ];
    const chains = rowChains(rows);
    assert.deepEqual(chains.map((c) => `${c.letter}:${c.kind}`), ["A:protein", "B:protein", "C:dna", "D:ligand"]);
    assert.equal(chains[0].residues[11], "E");
    // the letter the picker shows is the chain expandEntities resolves it to
    const { bonds } = expandEntities([...rows.slice(0, 2), { type: "bond", value: "A12:CA - D1:C1", copies: 1 },
                                      rows[3]]);
    assert.deepEqual(bonds[0].to, { asym: 3, residue: 1, atom: "C1" });
  });
});

describe("the link pickers' text", () => {
  const chains = rowChains([{ type: "protein", value: SEQ, copies: 1 }, { type: "ligand", value: "GOL", copies: 1 }]);
  for (const [type, text] of [
    ["bond", "A12:SG - B1:C1"],
    ["bond", "A12 - B1"],
    ["contact", "A10 - B1:C1 @ 8"],
    ["pocket", "B: A9, A12, A16 @ 6"],
    ["crosslink", "DSSO: A2 - A30"],
  ]) {
    it(`reads ${type} \`${text}\` back as it was written`, () => {
      assert.equal(linkText(type, linkParts(type, text, chains)), text);
      assert.equal(entityProblem({ type, value: text, copies: 1 }), null);
    });
  }

  it("starts an empty row on the first two chains, unfinished until a residue is picked", () => {
    const parts = linkParts("contact", "", chains);
    assert.equal(linkText("contact", parts), "A - B @ 8");
    assert.notEqual(entityProblem({ type: "contact", value: linkText("contact", parts), copies: 1 }), null);
    parts.ends[0].residue = 5;
    parts.ends[1].residue = 1;
    assert.equal(linkText("contact", parts), "A5 - B1 @ 8");
    assert.equal(entityProblem({ type: "contact", value: "A5 - B1 @ 8", copies: 1 }), null);
  });
});
