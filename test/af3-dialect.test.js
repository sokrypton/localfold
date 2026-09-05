/**
 * Which AF3-lineage graph a bundle's weights were trained for.
 *
 * 🔴 EVERY DIFFERENCE HERE IS SILENT WHEN WRONG. The shapes agree, the fold
 * finishes, and what comes out is a slightly different model - so the only
 * things that can catch a mistake are a checker that sweeps the dialect as an
 * axis (tools/gpu/check-af3-atom-encoder.js and
 * check-af3-diffusion-conditioning.js do) and the assertions here, which pin
 * the table itself.
 *
 * 🔴 AND THE TABLE IS WHERE THE OPENFOLD3-VERSUS-OPENBIND TRAP LIVES. OpenBind
 * is OpenFold3's v0.5.0 release and it moved TOWARD AlphaFold 3: it dropped the
 * transposed column-attention pair bias its preview-2 weights were trained
 * with. Reading the OF3 porting notes and applying them wholesale gets that
 * backwards, and the resulting fold is wrong in a way nothing else in this
 * repository would report.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  ALPHAFOLD3, DIALECTS, OPENBIND, dialectFor,
  singleCondPadding, singleCondPaddingWgsl, singleCondSource,
} from "../src/af3/dialect.js";
import { af3Dialect } from "../src/af3/weights.js";
import { featuriseProtein } from "../src/af3/featurise.js";

const LIGAND = {
  code: "TST",
  atoms: [
    { name: "C1", element: 6, charge: 0, x: 0, y: 0, z: 0 },
    { name: "O1", element: 8, charge: 0, x: 1.4, y: 0, z: 0 },
    { name: "P1", element: 15, charge: 0, x: 3.0, y: 0, z: 0 },
  ],
  bonds: [{ from: 0, to: 1, order: 1 }, { from: 1, to: 2, order: 1 }],
};

describe("the dialect table", () => {
  it("keeps OpenBind on AF3's pair-bias convention, which OpenFold3 changed", () => {
    // 🔴 THE ONE ASSERTION MOST WORTH HAVING. `swapTransposedBias` is threaded
    // through the pairformer, the MSA stack, the template embedder and the
    // confidence head; upstream's TRANSPOSED_COLUMN_PAIR_BIAS lists openfold3
    // and deliberately not openbind. Turning it on here transposes every
    // column attention's bias against weights that do not want it.
    assert.equal(OPENBIND.swapTransposedBias, false);
    assert.equal(ALPHAFOLD3.swapTransposedBias, false);
  });

  it("turns on exactly the three branches OpenBind needs", () => {
    assert.deepEqual({ ...OPENBIND }, {
      swapTransposedBias: false,
      symmetriseBonds: true,
      maskPaddedKeys: true,
      padSingleCondUnknownDna: true,
    });
  });

  it("leaves stock AlphaFold 3 with none of them", () => {
    for (const [flag, value] of Object.entries(ALPHAFOLD3)) {
      assert.equal(value, false, `${flag} is not false for stock AF3`);
    }
  });

  it("gives every dialect every flag, so a use site cannot read undefined", () => {
    // A missing flag reads as falsy at a use site that forgot to check for it,
    // which is stock AF3's branch taken silently. Every consumer throws on
    // undefined; this is the other half of that.
    const flags = Object.keys(ALPHAFOLD3);
    for (const [name, dialect] of Object.entries(DIALECTS)) {
      assert.deepEqual(Object.keys(dialect).sort(), flags.slice().sort(),
        `dialect ${name} does not carry the same flags as stock AF3`);
    }
  });

  it("refuses a model it does not know rather than assuming stock", () => {
    assert.throws(() => dialectFor("openfold3"), /no AF3 dialect/);
    assert.throws(() => dialectFor(undefined), /no AF3 dialect/);
    assert.equal(dialectFor("alphafold3"), ALPHAFOLD3);
    assert.equal(dialectFor("openbind"), OPENBIND);
  });
});

describe("a bundle names its own graph", () => {
  it("reads the dialect out of the manifest", () => {
    assert.equal(af3Dialect({ manifest: { model: { name: "openbind" } } }), OPENBIND);
    assert.equal(af3Dialect({ manifest: { model: { name: "alphafold3" } } }), ALPHAFOLD3);
  });

  it("refuses a bundle whose manifest does not name a model", () => {
    // 🔴 NOT A DEFAULT. A ported bundle missing this field would fold through
    // stock AF3's branches and return a structure, which is the failure this
    // whole mechanism exists to make impossible.
    assert.throws(() => af3Dialect({ manifest: { model: {} } }), /does not name its model/);
    assert.throws(() => af3Dialect({ manifest: {} }), /does not name its model/);
    assert.throws(() => af3Dialect(undefined), /does not name its model/);
  });
});

describe("the unknown-DNA columns of the diffusion single conditioning", () => {
  it("adds nothing at all under stock AF3", () => {
    assert.deepEqual(singleCondPadding(ALPHAFOLD3, 384), []);
    assert.equal(singleCondPaddingWgsl([]), "");
  });

  it("puts one column after restype and one after profile", () => {
    // [trunk single 384 | restype 31 | profile 31 | deletion mean 1 | atoms 384]
    // becomes 833 rather than 831, and the two inserted columns sit at the end
    // of the restype block and the end of the profile block.
    assert.deepEqual(singleCondPadding(OPENBIND, 384), [415, 447]);
  });

  it("maps every padded column back to the source it reads", () => {
    const padding = singleCondPadding(OPENBIND, 384);
    // The trunk single block is untouched...
    assert.equal(singleCondSource(padding, 0), 0);
    assert.equal(singleCondSource(padding, 383), 383);
    // ...the 31 restypes land where they were...
    assert.equal(singleCondSource(padding, 384), 384);
    assert.equal(singleCondSource(padding, 414), 414);
    // ...then a zero, then the profile block shifted by one...
    assert.equal(singleCondSource(padding, 415), -1);
    assert.equal(singleCondSource(padding, 416), 415);
    assert.equal(singleCondSource(padding, 446), 445);
    // ...then a second zero, and everything after shifted by two.
    assert.equal(singleCondSource(padding, 447), -1);
    assert.equal(singleCondSource(padding, 448), 446);
    assert.equal(singleCondSource(padding, 832), 830);
  });

  it("is exactly an insertion, checked by doing the insertion", () => {
    // 🔴 THE PROPERTY, NOT A TABLE OF INDICES. Build the padded row the way a
    // converter does - splice two zeros in - and assert singleCondSource
    // reproduces it from the source row. A sign error in the shift shows up
    // here and nowhere in the spot checks above.
    const padding = singleCondPadding(OPENBIND, 384);
    const source = Float32Array.from({ length: 831 }, (_, i) => i + 1);
    const spliced = [...source];
    for (const at of padding) spliced.splice(at, 0, 0);
    assert.equal(spliced.length, 833);
    for (let index = 0; index < spliced.length; index += 1) {
      const from = singleCondSource(padding, index);
      assert.equal(spliced[index], from < 0 ? 0 : source[from],
        `padded column ${index} reads the wrong source`);
    }
  });

  it("generates WGSL with a guard and a shift for each inserted column", () => {
    // The shader's `feature()` is generated from the same list the reference
    // walks; this is what says the generator did not drop one.
    const wgsl = singleCondPaddingWgsl(singleCondPadding(OPENBIND, 384));
    assert.equal((wgsl.match(/return 0\.0;/g) ?? []).length, 2);
    assert.equal((wgsl.match(/source -= 1u;/g) ?? []).length, 2);
    assert.match(wgsl, /if \(index == 415u\)/);
    assert.match(wgsl, /if \(index == 447u\)/);
    // 🔴 EVERY GUARD BEFORE EVERY SHIFT. Interleaving them would let the first
    // shift move `index` out from under the second guard's comparison.
    assert.ok(wgsl.lastIndexOf("return 0.0;") < wgsl.indexOf("source -= 1u;"),
      "a shift is emitted before the last zero-column guard");
  });

  it("throws rather than assume a dialect that does not say", () => {
    assert.throws(() => singleCondPadding({}, 384), /has no default/);
    assert.throws(() => singleCondPadding(undefined, 384), /has no default/);
  });
});

describe("the token bond matrix", () => {
  const ligandMatrix = (symmetriseBonds) => {
    const batch = featuriseProtein("GWSTELEKHR", { ligands: [LIGAND], symmetriseBonds });
    const base = batch.ligandSpans[0].from;
    return {
      batch,
      at: (i, j) => batch.bondMatrix[(base + i) * batch.tokens + (base + j)],
      set: batch.bondMatrix.reduce((sum, value) => sum + value, 0),
    };
  };

  it("is one direction per bond under stock AF3", () => {
    const { at, set } = ligandMatrix(ALPHAFOLD3.symmetriseBonds);
    assert.equal(at(0, 1), 1);
    assert.equal(at(1, 0), 0);
    assert.equal(set, LIGAND.bonds.length);
  });

  it("is symmetric under OpenBind, which is what its weights were trained on", () => {
    // 🔴 NOT COSMETIC. Upstream measures a ring ligand folded through the wrong
    // convention coming apart - ATP's ribose C-C at ~2.0 A against ~1.5 - and
    // this matrix goes through a learned linear straight into the pair
    // representation.
    const { at, set } = ligandMatrix(OPENBIND.symmetriseBonds);
    assert.equal(at(0, 1), 1);
    assert.equal(at(1, 0), 1);
    assert.equal(set, LIGAND.bonds.length * 2);
  });

  it("clears [0,0] either way, because a padded gather row is zeros", () => {
    for (const flag of [false, true]) {
      assert.equal(ligandMatrix(flag).batch.bondMatrix[0], 0);
    }
  });
});
