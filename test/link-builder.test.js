/**
 * The bond/contact/pocket/crosslink pickers (web/link-builder.js) write the same spec text the box took, and offer the
 * chains under the letters expandEntities resolves them to - a picker that lettered chains its own way would bond
 * something other than what it showed.
 */
import assert from "node:assert/strict";
import { describe, it } from "./harness.js";
import { chainLetter, entitiesProblem, entityProblem, expandEntities, foldModificationRows, modificationRows, rowChains }
  from "../web/entities.js";
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

describe("the crosslinker menu's spacer arms", () => {
  it("name only crosslinkers the fold knows", async () => {
    const { CROSSLINKERS } = await import("../shared/chem/crosslinkers.js");
    const { SPACER_ARM } = await import("../web/link-builder.js");
    for (const name of Object.keys(SPACER_ARM)) assert.ok(name in CROSSLINKERS, `${name} is not a crosslinker`);
  });
});

describe("modification rows", () => {
  const protein = { type: "protein", value: "GSHMSKLE", copies: 2, modifications: [{ code: "SEP", position: 5 }] };
  const dna = { type: "dna", value: "ACGT", copies: 1, modifications: [{ code: "5CM", position: 2 }] };

  it("come out of each sequence's modifications, naming its first copy, and fold back unchanged", () => {
    const rows = modificationRows([protein, { type: "ligand", value: "GOL", copies: 1 }, dna]);
    assert.deepEqual(rows.map((row) => `${row.type}:${row.value}`),
      ["protein:GSHMSKLE", "modification:A5 SEP", "ligand:GOL", "dna:ACGT", "modification:C2 5CM"]);
    const back = foldModificationRows(rows);
    assert.deepEqual(back.map((row) => row.type), ["protein", "ligand", "dna"]);
    assert.deepEqual(back[0].modifications, protein.modifications);
    assert.deepEqual(back[2].modifications, dna.modifications);
  });

  it("modify every copy of the sequence, whichever copy they name", () => {
    // B is the protein's second copy: the modification is the sequence's, as in AlphaFold 3's job
    const back = foldModificationRows([{ ...protein, modifications: [] },
                                       { type: "modification", value: "B5 SEP", copies: 1 }]);
    assert.deepEqual(back[0].modifications, [{ code: "SEP", position: 5 }]);
    assert.deepEqual(expandEntities(back).modifications.map((m) => m.chain), [0, 1]);
  });

  it("are refused, by name, where they name no polymer chain or do not parse", () => {
    const rows = [{ type: "protein", value: "GSHMSKLE", copies: 1 }, { type: "ligand", value: "GOL", copies: 1 }];
    assert.match(entitiesProblem(foldModificationRows([...rows, { type: "modification", value: "B1 SEP", copies: 1 }])),
      /no protein, DNA or RNA chain B/);
    assert.match(entitiesProblem(foldModificationRows([...rows, { type: "modification", value: "A SEP", copies: 1 }])),
      /a residue and a CCD code/);
    // ...and a residue the code does not fit is the sequence's own check, as before
    assert.match(entitiesProblem(foldModificationRows([...rows, { type: "modification", value: "A3 SEP", copies: 1 }])),
      /SEP modifies S/);
  });

  it("read back as written through the picker's text", () => {
    const chains = rowChains([{ type: "protein", value: "GSHMSKLE", copies: 1 }]);
    assert.equal(linkText("modification", linkParts("modification", "A5 SEP", chains)), "A5 SEP");
    assert.equal(linkText("modification", linkParts("modification", "", chains)), "A");
  });
});
