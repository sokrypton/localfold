/**
 * A ligand's bonds, from the CCD to the model and to the viewer.
 *
 * They were read correctly and then dropped twice, on two independent paths,
 * and each drop was invisible from the other side:
 *
 *   THE MODEL. `featurise.js` builds the contact matrix and `fold.js` handed
 *   the trunk an object built FIELD BY FIELD that did not name it, so the
 *   embedder saw `undefined`; and `embedder-webgpu.js` - the one a browser fold
 *   runs - had neither the `bondEmbedding` weight nor the term, while
 *   `embedder-reference.js` had both. `bond_embedding/weights` was downloaded
 *   with every fold and multiplied by nothing. The pair track therefore carried
 *   no statement that any two ligand atoms are bonded.
 *
 *   THE VIEWER. `toPdb` wrote no CONECT records, so py2Dmol fell back to
 *   deriving bonds from the DISTANCE between atoms - re-derived from every
 *   trajectory frame, whose coordinates are deliberately noisy until the last
 *   few diffusion steps. The sticks appear, cross and vanish frame to frame.
 *   And the element column had a four-entry table (C, N, O, S) with everything
 *   else falling through to carbon, which breaks that fallback a second way:
 *   the distance rule is per ELEMENT PAIR.
 *
 * The GPU half is checked in tools/gpu/check-af3-embedder.js, which now carries
 * a bond matrix - without one it compared zero against zero on both sides,
 * which is why it passed throughout.
 */
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

import { featuriseProtein } from "../src/af3/featurise.js";
import { toPdb } from "../src/af3/fold.js";
import { ELEMENT_SYMBOLS } from "../src/af3/ccd-component.js";

/**
 * A ligand shaped like a small phosphate-and-metal cofactor: the elements are
 * chosen so that every one of them EXCEPT carbon was written as carbon before,
 * and so that two of its bonds are ones a distance rule gets wrong.
 */
const LIGAND = {
  code: "TST",
  atoms: [
    { name: "C1", element: 6, charge: 0, x: 0, y: 0, z: 0 },
    { name: "O1", element: 8, charge: 0, x: 1.4, y: 0, z: 0 },
    { name: "P1", element: 15, charge: 0, x: 3.0, y: 0, z: 0 },
    { name: "S1", element: 16, charge: 0, x: 4.6, y: 0, z: 0 },
    { name: "S2", element: 16, charge: 0, x: 6.65, y: 0, z: 0 },
    { name: "FE1", element: 26, charge: 0, x: 9.0, y: 0, z: 0 },
  ],
  bonds: [
    { from: 0, to: 1, order: 1 },
    { from: 1, to: 2, order: 1 },
    { from: 2, to: 3, order: 1 },
    { from: 3, to: 4, order: 1 },   // S-S at 2.05 A
    { from: 4, to: 5, order: 1 },   // S-FE at 2.35 A
  ],
};

const batchWithLigand = () => featuriseProtein("GWSTELEKHR", { ligands: [LIGAND] });

/** A batch's positions, straight from the reference conformer it was given. */
function positionsOf(batch) {
  const positions = new Float32Array(batch.tokens * batch.dense * 3);
  positions.set(batch.features.refPos ?? batch.refPos ?? []);
  return positions;
}

describe("a ligand's bonds reach the model", () => {
  it("is featurised as a contact matrix, one direction per bond", () => {
    const batch = batchWithLigand();
    assert.ok(batch.bondMatrix !== undefined, "no contact matrix was built");
    assert.equal(batch.bondMatrix.length, batch.tokens * batch.tokens);

    const base = batch.ligandSpans[0].from;
    const at = (i, j) => batch.bondMatrix[(base + i) * batch.tokens + (base + j)];
    for (const bond of LIGAND.bonds) {
      assert.equal(at(bond.from, bond.to), 1,
        `bond ${bond.from}-${bond.to} is not in the matrix`);
    }
    // ...ONE DIRECTION. AF3's featurisation sets contact[i][j] and not [j][i];
    // only the OpenFold3 dialect symmetrises, and these are stock weights.
    assert.equal(at(1, 0), 0, "the matrix was symmetrised, which is the OF3 dialect");
    // ...and the total is exactly the bond count, so nothing else was set.
    const set = batch.bondMatrix.reduce((sum, value) => sum + value, 0);
    assert.equal(set, LIGAND.bonds.length);
  });

  it("survives the trunk input, which is rebuilt field by field", () => {
    // 🔴 THE FAULT THIS PINS. fold.js assembles the trunk's input as an object
    // literal, so a key the batch carries and the literal does not name is a
    // key thrown away - and `undefined` there is indistinguishable from a fold
    // with no ligand, which is what made it survive. Read out of the source,
    // because the alternative is running a fold.
    const source = readSource("src/af3/fold.js");
    const call = source.slice(source.indexOf("trunk = await trunkGpu.run({"),
                              source.indexOf("}, weights.trunk, DIALECT"));
    assert.ok(/bondMatrix:\s*batch\.bondMatrix/.test(call),
      "fold.js does not pass the contact matrix to the trunk");
  });

  it("is applied by the GPU embedder, not only by the reference", () => {
    // The reference has had this term throughout; the GPU path is what a
    // browser fold runs, and it had neither the weight nor the arithmetic.
    const gpu = readSource("src/af3/embedder-webgpu.js");
    assert.ok(/"bondEmbedding"/.test(gpu),
      "bondEmbedding is not in the GPU embedder's packed weight list");
    assert.ok(/bonds\[row\]\s*\*\s*weights\[W_BOND \+ c\]/.test(gpu),
      "the GPU pair init does not add the bond term");
    // ...and the weight has to be loadable, or packEmbedderWeights throws.
    const weights = readSource("src/af3/weights.js");
    assert.ok(/bondEmbedding:\s*await T\("bond_embedding\/weights"\)/.test(weights),
      "embedderWeights does not read bond_embedding/weights");
  });
});

describe("a ligand's bonds reach the viewer", () => {
  it("writes CONECT records for every bond, both directions", () => {
    const batch = batchWithLigand();
    const pdb = toPdb(batch, positionsOf(batch), null);
    const conect = pdb.split("\n").filter((line) => line.startsWith("CONECT"));
    assert.ok(conect.length > 0, "the PDB carries no CONECT records at all");

    // Every bond, as an unordered pair of serials, exactly once.
    const serials = new Map();          // atom name -> serial
    for (const line of pdb.split("\n")) {
      if (!line.startsWith("HETATM")) continue;
      serials.set(line.slice(12, 16).trim(), Number(line.slice(6, 11)));
    }
    const pairs = new Set();
    for (const line of conect) {
      const from = Number(line.slice(6, 11));
      for (let field = 0; field < 4; field += 1) {
        const text = line.slice(11 + field * 5, 16 + field * 5).trim();
        if (text === "") continue;
        const to = Number(text);
        pairs.add([from, to].sort((a, b) => a - b).join("-"));
      }
    }
    for (const bond of LIGAND.bonds) {
      const a = serials.get(LIGAND.atoms[bond.from].name);
      const b = serials.get(LIGAND.atoms[bond.to].name);
      const key = [a, b].sort((x, y) => x - y).join("-");
      assert.ok(pairs.has(key),
        `${LIGAND.atoms[bond.from].name}-${LIGAND.atoms[bond.to].name} has no CONECT`);
    }
    assert.equal(pairs.size, LIGAND.bonds.length, "extra bonds were invented");
    // ...and nothing was claimed between the polymer and the ligand: that link
    // is not featurised, so writing one would be the writer inventing chemistry
    // the fold never saw.
    const ligandSerials = new Set(LIGAND.atoms.map((atom) => serials.get(atom.name)));
    for (const key of pairs) {
      for (const serial of key.split("-")) {
        assert.ok(ligandSerials.has(Number(serial)),
          `CONECT names serial ${serial}, which is not a ligand atom`);
      }
    }
  });

  it("writes each atom's real element, not carbon for everything but CNOS", () => {
    const batch = batchWithLigand();
    const pdb = toPdb(batch, positionsOf(batch), null);
    const written = new Map();
    for (const line of pdb.split("\n")) {
      if (!line.startsWith("HETATM")) continue;
      written.set(line.slice(12, 16).trim(), line.slice(76, 78).trim());
    }
    for (const atom of LIGAND.atoms) {
      assert.equal(written.get(atom.name), ELEMENT_SYMBOLS[atom.element - 1],
        `${atom.name} was written as ${written.get(atom.name)}`);
    }
    // The two that used to come out as carbon and matter most: a phosphorus,
    // whose P-O bond a C-O distance rule judges by two hundredths of an
    // angstrom, and an iron, which is drawn at a carbon's radius and colour.
    assert.equal(written.get("P1"), "P");
    assert.equal(written.get("FE1"), "FE");
  });
});

function readSource(relative) {
  return readFileSync(new URL(`../${relative}`, import.meta.url), "utf8");
}
