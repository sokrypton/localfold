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
import { linear } from "../src/af3/pairformer-reference.js";

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

/**
 * The conditioning exactly as it was written before the two one-hot embeddings
 * became gathers - the dense matmul against a materialised one-hot.
 *
 * 🔴 IT IS HERE SO THE FAST FORM IS HELD TO THE SLOW ONE'S ARITHMETIC, not just
 * to its answer. The element and atom-name embeddings multiply by a one-hot,
 * so all but one column of each product is a multiply by zero: 384 of the 389
 * input columns, and 99% of the function's cost - 72 ms at 59 tokens, 220 at
 * 150. Adding the row the one-hot selects is the same float, but only if two
 * things are preserved, and BOTH were got wrong on the way here:
 *
 *   - the four name rows are summed BEFORE they reach `act`, because
 *     `act + (a+b+c+d)` is not `(((act+a)+b)+c)+d`. Adding them in turn left
 *     169,390 floats differing by up to 4.8e-7.
 *   - and they are summed in float64 and rounded ONCE, because that is what a
 *     dot product accumulated in a JS number does. A Float32Array scratch
 *     rounds after every addition, which still left 12,641 differing.
 *
 * Neither would have shown up as a failure anywhere - 1e-7 on a reference the
 * GPU checkers compare against at 1e-6 is a tolerance nobody chose, quietly
 * spent. Hence bitwise.
 */
function conditioningAsMatmuls(reference, tokens, dense, weights) {
  const channels = weights.channels;
  const rows = tokens * dense;
  const act = linear(reference.positions, rows, 3, channels, weights.embedRefPos);
  const add = (contribution) => {
    for (let index = 0; index < act.length; index += 1) act[index] += contribution[index];
  };
  const maskColumn = new Float32Array(rows);
  for (let index = 0; index < rows; index += 1) maskColumn[index] = reference.mask[index];
  add(linear(maskColumn, rows, 1, channels, weights.embedRefMask));
  const element = new Float32Array(rows * 128);
  for (let index = 0; index < rows; index += 1) {
    const atomicNumber = reference.element[index];
    if (atomicNumber >= 0 && atomicNumber < 128) element[index * 128 + atomicNumber] = 1;
  }
  add(linear(element, rows, 128, channels, weights.embedRefElement));
  const charge = new Float32Array(rows);
  for (let index = 0; index < rows; index += 1) charge[index] = Math.asinh(reference.charge[index]);
  add(linear(charge, rows, 1, channels, weights.embedRefCharge));
  const names = new Float32Array(rows * 256);
  for (let index = 0; index < rows; index += 1) {
    for (let character = 0; character < 4; character += 1) {
      const code = reference.atomNameChars[index * 4 + character];
      if (code >= 0 && code < 64) names[index * 256 + character * 64 + code] = 1;
    }
  }
  add(linear(names, rows, 256, channels, weights.embedRefAtomName));
  for (let index = 0; index < rows; index += 1) {
    const keep = reference.mask[index];
    for (let c = 0; c < channels; c += 1) act[index * channels + c] *= keep;
  }
  return act;
}

/** Deterministic noise, so a failure is reproducible rather than occasional. */
function noiseFrom(seed) {
  let state = seed >>> 0;
  return (count) => {
    const values = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      values[index] = state / 0x7fffffff - 0.5;
    }
    return values;
  };
}

describe("AF3 per-atom conditioning against the matmul form", () => {
  // Ragged on purpose: absent atoms, an atomic number past the table, and an
  // atom-name code past 64 - the three cases where a gather and a one-hot can
  // disagree about what "no contribution" means.
  for (const [tokens, dense, channels] of [[7, 5, 64], [17, 24, 128], [59, 24, 128]]) {
    it(`is bitwise identical at ${tokens} tokens, ${dense} slots, ${channels} channels`, () => {
      const random = noiseFrom(987654321 + tokens);
      const rows = tokens * dense;
      const reference = {
        positions: random(rows * 3),
        mask: Float32Array.from({ length: rows }, (_, i) => (i % 7 === 3 ? 0 : 1)),
        element: Int32Array.from({ length: rows }, (_, i) => (i % 23 === 0 ? 200 : i % 118)),
        charge: random(rows),
        atomNameChars: Int32Array.from({ length: rows * 4 },
          (_, i) => (i % 31 === 0 ? 99 : i % 64)),
      };
      const weights = {
        channels,
        embedRefPos: random(3 * channels),
        embedRefMask: random(channels),
        embedRefElement: random(128 * channels),
        embedRefCharge: random(channels),
        embedRefAtomName: random(256 * channels),
      };
      const expected = conditioningAsMatmuls(reference, tokens, dense, weights);
      const actual = perAtomConditioning(reference, tokens, dense, weights);
      assert.equal(actual.length, expected.length);
      const differing = [...actual].filter((value, index) => value !== expected[index]).length;
      assert.equal(differing, 0, `${differing} of ${expected.length} floats differ`);
    });
  }
});
