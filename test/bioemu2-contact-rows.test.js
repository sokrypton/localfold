/**
 * BioEmu2's contact conditioning from the page's Contact rows (web/bioemu2-model.js bioemu2ContactsFromRows), which
 * replaced a JSON box of its own: the rows must give exactly the Contact list the box took, name each chain by the
 * job the featuriser reads, and refuse a Pocket row, which BioEmu2 has no counterpart for.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "./harness.js";
import { expandEntities } from "../web/entities.js";
import { jobInputJson } from "../web/job-json.js";
import { bioemu2ContactsFromRows } from "../web/bioemu2-model.js";
import { parseContacts } from "../shared/bioemu2/featurise.js";

const SEQ_6MRR = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";
const contactsOf = (entities) => bioemu2ContactsFromRows(expandEntities(entities).constraints,
  jobInputJson({ name: "t", seed: 1, entities }));

describe("BioEmu2's contacts from Contact rows", () => {
  it("are the Contact list the native gate's fixture holds", () => {
    const got = contactsOf([
      { type: "protein", value: SEQ_6MRR, copies: 1 },
      { type: "contact", value: "A5 - A60 @ 8", copies: 1 },
      { type: "contact", value: "A10:CA - A40 @ 12", copies: 1 },
    ]);
    const fixture = JSON.parse(readFileSync(new URL("../tools/fixtures/bioemu2/contacts.json", import.meta.url), "utf8"));
    assert.deepEqual(got, fixture);
    // ...and BioEmu2's own reader takes them
    assert.equal(parseContacts(got).length, 2);
  });

  it("name a chain by the job, where a ligand row sits above the protein", () => {
    // the rows' letters number polymers first (A is the protein, B the ligand); the job hands out ids in row order
    // (the ligand A, the protein B), so the protein's contact must come out as the job's B
    const got = contactsOf([
      { type: "ligand", value: "GOL", copies: 1 },
      { type: "protein", value: SEQ_6MRR, copies: 1 },
      { type: "contact", value: "A5 - B1:C1 @ 8", copies: 1 },
    ]);
    assert.deepEqual(got[0].token1, { asym_id: "B", res_id: 5 });
    assert.deepEqual(got[0].token2, { asym_id: "A", res_id: 1, atom_name: "C1" });
  });

  it("are null without a Contact row, and refuse a Pocket", () => {
    assert.equal(contactsOf([{ type: "protein", value: SEQ_6MRR, copies: 1 }]), null);
    assert.throws(() => contactsOf([
      { type: "protein", value: SEQ_6MRR, copies: 1 },
      { type: "protein", value: "GSHMKLE", copies: 1 },
      { type: "pocket", value: "B: A9, A12 @ 6", copies: 1 },
    ]), /not pockets/);
  });
});
