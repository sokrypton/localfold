/**
 * What an AlphaFold 3-graph fold's PDB says about itself.
 *
 * 🔴 IT SAID NOTHING, AND THE SAVE BUTTON IS WHERE THAT SHOWS. `toPdb` opened
 * with `const lines = []` and pushed no header, so a file downloaded from the
 * page carried no record of which model produced it and no statement of what
 * the B-factor column holds. Measured on the page: an AF3 fold's saved
 * `af3_1.pdb` had zero REMARK lines, while an AlphaFold 2 fold's carried
 * `REMARK   1 ALPHAFOLD2 WEBGPU PREDICTION` and an EF2-fast fold's carried two
 * naming its certainty column. Seven models share this writer - af3,
 * openbind0, opendde, boltz2, protenix2, intellifold2, rosettafold3 - so all
 * seven wrote an anonymous file with a pLDDT-shaped column nothing labelled.
 *
 * 🔴 AND THE HEADER IS OPTIONAL, WHICH IS THE POINT. Three tests and several
 * tools call `toPdb(batch, positions, plddt)` and compare or parse what comes
 * back; a header emitted unconditionally would change every one of those
 * outputs for a question they are not asking. The caller that KNOWS the model's
 * name - the page - is the caller that passes one.
 */
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { featuriseProtein } from "../src/af3/featurise/featurise.js";
import { toPdb } from "../src/af3/fold.js";

const batchOf = () => featuriseProtein("GWSTELEKHR");

/** Blank coordinates: this asks what the writer WRITES, not where atoms are. */
const blank = (batch) => new Float32Array(batch.tokens * batch.dense * 3);

describe("an AF3-graph PDB names what produced it", () => {
  it("writes no header when the caller names none", () => {
    // 🔴 THE GUARD ON EVERY EXISTING CALLER. af3-ligand-bonds.test.js reads
    // this output in file order and tools/gpu/bench-frame.js times it; a
    // header nobody asked for would be a line they did not expect.
    const batch = batchOf();
    const lines = toPdb(batch, blank(batch), null).split("\n");
    assert.equal(lines.filter((line) => line.startsWith("REMARK")).length, 0,
      "an unasked-for header appeared, which changes every existing caller's output");
  });

  it("writes the caller's header, before any atom", () => {
    const batch = batchOf();
    const remark = [
      "ALPHAFOLD 3 PREDICTION BY LOCALFOLD (https://localfold.org)",
      "B-FACTOR IS pLDDT (0-100).",
    ];
    const lines = toPdb(batch, blank(batch), null, { remark }).split("\n");
    const headers = lines.filter((line) => line.startsWith("REMARK"));
    assert.equal(headers.length, 2, "the header the caller passed is not in the file");
    // ...spelled as a PDB REMARK, which is a record type and a number.
    assert.match(headers[0], /^REMARK {3}1 ALPHAFOLD 3 PREDICTION BY LOCALFOLD/);
    assert.match(headers[1], /^REMARK {3}1 B-FACTOR IS pLDDT \(0-100\)\.$/);
    // 🔴 BEFORE THE ATOMS, or a reader that stops at the first coordinate
    // record never sees it - which is most readers.
    const firstAtom = lines.findIndex((line) => line.startsWith("ATOM")
      || line.startsWith("HETATM"));
    const lastHeader = lines.map((line, index) => (line.startsWith("REMARK") ? index : -1))
      .reduce((highest, index) => Math.max(highest, index), -1);
    assert.ok(firstAtom > lastHeader,
      "the header is written after an atom record, where a reader will not look");
  });
});
