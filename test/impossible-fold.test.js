/**
 * A fold's numbers that cannot be right: the page refuses them whichever model made them.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { isImpossibleFold } from "../web/impossible-fold.js";

const atom = (x, y, z, record = "ATOM  ") => `${record}    1  CA  ALA A   1    `
  + `${String(x).padStart(8)}${String(y).padStart(8)}${String(z).padStart(8)}  1.00 90.00           C`;
const pdb = (...lines) => ["HEADER    NAN IN A HEADER IS FINE", ...lines, "END"].join("\n");

describe("a fold that cannot be one", () => {
  it("passes a structure, at either end of what a file holds", () => {
    assert.equal(isImpossibleFold({ pdb: pdb(atom("1.000", "-2.000", "3.000"), atom("9999.000", "-999.000", "0.000", "HETATM")) }), false);
  });

  it("refuses a coordinate that is not a number", () => {
    assert.equal(isImpossibleFold({ pdb: pdb(atom("1.000", "NaN", "3.000")) }), true);
    assert.equal(isImpossibleFold({ pdb: pdb(atom("Infinity", "2.000", "3.000")) }), true);
    assert.equal(isImpossibleFold({ pdb: pdb(atom("1.000", "2.000", "3.000"), atom("NaN", "NaN", "NaN", "HETATM")) }), true);
  });

  it("refuses a coordinate past the columns it has", () => {
    assert.equal(isImpossibleFold({ pdb: pdb(atom("1e+30", "2.000", "3.000")) }), true);
    assert.equal(isImpossibleFold({ pdb: pdb(atom("1.000", "12345.678", "3.000")) }), true);
  });

  it("refuses every atom on one point, and passes one atom", () => {
    assert.equal(isImpossibleFold({ pdb: pdb(atom("0.000", "0.000", "0.000"), atom("0.000", "0.000", "0.000")) }), true);
    assert.equal(isImpossibleFold({ pdb: pdb(atom("0.000", "0.000", "0.000")) }), false);
    assert.equal(isImpossibleFold({ pdb: pdb(atom("0.000", "0.000", "0.000"), atom("0.000", "0.000", "1.000")) }), false);
  });

  it("refuses a confidence that is not a number", () => {
    const good = pdb(atom("1.000", "2.000", "3.000"));
    assert.equal(isImpossibleFold({ pdb: good, confidence: { meanPlddt: Number.NaN } }), true);
    assert.equal(isImpossibleFold({ pdb: good, confidence: { meanPlddt: 87.0 } }), false);
  });

  it("has nothing to say of what a model does not give", () => {
    const good = pdb(atom("1.000", "2.000", "3.000"));
    assert.equal(isImpossibleFold({ pdb: good, confidence: undefined }), false);
    assert.equal(isImpossibleFold({ pdb: good, confidence: {} }), false);
    assert.equal(isImpossibleFold({ pdb: undefined, confidence: undefined }), false);
  });
});

it("a model with no confidence head (null, as the native worker sends BioEmu2's) is not refused for it", () => {
  const pdb = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
    + "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  0.00           C\n";
  assert.equal(isImpossibleFold({ pdb, confidence: { meanPlddt: null } }), false);
});
