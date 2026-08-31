/**
 * The per-atom conditioning's encodings, including one no protein can test.
 *
 * tools/oracle/check_af3_atom.js checks all five embeddings against AF3 on a
 * real protein and they come back exact - which proves less than it looks,
 * because every standard amino acid's reference conformer in the CCD is
 * NEUTRAL. The charge channel is identically zero there, and arcsinh(0) = 0, so
 * feeding the raw charge instead of its arcsinh is invisible on any protein and
 * first goes wrong on a ligand.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { perAtomConditioning } from "../src/af3/atom-conditioning-reference.js";

/** One atom, one channel, so each embedding's contribution is a single number. */
function oneAtom(overrides = {}) {
  return {
    positions: Float32Array.from([0, 0, 0]),
    mask: Float32Array.from([1]),
    element: Int32Array.from([0]),
    charge: Float32Array.from([0]),
    atomNameChars: Int32Array.from([0, 0, 0, 0]),
    ...overrides,
  };
}

/** Weights that pass exactly one embedding through and zero the rest. */
function onlyCharge() {
  return {
    channels: 1,
    embedRefPos: new Float32Array(3),
    embedRefMask: new Float32Array(1),
    embedRefElement: new Float32Array(128),
    embedRefCharge: Float32Array.from([1]),
    embedRefAtomName: new Float32Array(256),
  };
}

describe("AF3 per-atom conditioning", () => {
  it("feeds arcsinh(charge), not the charge", () => {
    // 🔴 THE WHOLE POINT OF THIS FILE. At charge 3 the two differ by a factor
    // of about 1.7; at charge 0 they do not differ at all, which is every
    // amino acid.
    const output = perAtomConditioning(oneAtom({ charge: Float32Array.from([3]) }),
                                       1, 1, onlyCharge());
    assert.ok(Math.abs(output[0] - Math.asinh(3)) < 1e-6,
              `expected asinh(3) = ${Math.asinh(3)}, got ${output[0]}`);
    assert.ok(Math.abs(output[0] - 3) > 1,
              "the raw charge would have passed a weaker assertion");
  });

  it("is indistinguishable from the raw charge at zero, as every residue is", () => {
    const output = perAtomConditioning(oneAtom(), 1, 1, onlyCharge());
    assert.equal(output[0], 0);
  });

  it("masks absent atoms to zero after summing, not before", () => {
    // Every embedding is bias-free but the element one-hot still fires for a
    // masked slot, so the mask has to be applied to the SUM. Element 6 with a
    // non-zero weight and mask 0 must still come out zero.
    const weights = { ...onlyCharge() };
    weights.embedRefElement = new Float32Array(128);
    weights.embedRefElement[6] = 5;
    const present = perAtomConditioning(
      oneAtom({ element: Int32Array.from([6]) }), 1, 1, weights);
    assert.equal(present[0], 5);
    const absent = perAtomConditioning(
      oneAtom({ element: Int32Array.from([6]), mask: Float32Array.from([0]) }),
      1, 1, weights);
    assert.equal(absent[0], 0);
  });

  it("encodes an atom name as four 64-way one-hots, in order", () => {
    const weights = { ...onlyCharge() };
    weights.embedRefAtomName = new Float32Array(256);
    // "CA" is ASCII 67, 65, padded with 0 - stored minus 32, so 35 and 33.
    weights.embedRefAtomName[0 * 64 + 35] = 1;
    weights.embedRefAtomName[1 * 64 + 33] = 10;
    const output = perAtomConditioning(
      oneAtom({ atomNameChars: Int32Array.from([35, 33, 0, 0]) }), 1, 1, weights);
    // 🔴 THE FOUR CHARACTERS OCCUPY DIFFERENT 64-COLUMN BLOCKS. Flattening them
    // the other way round - character-minor instead of character-major - still
    // produces a 256-wide one-hot with exactly four ones in it.
    assert.equal(output[0], 11);
  });
});
