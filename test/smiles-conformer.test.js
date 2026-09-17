import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";

import { parseSmiles } from "../src/chem/smiles.js";
import { valenceProblems } from "../src/chem/kekulize.js";
import { smallestRings } from "../src/chem/rings.js";
import { chiralCentres, doubleBondStereo } from "../src/chem/stereo.js";
import { distanceBounds, smoothBounds, signedVolume } from "../src/chem/conformer.js";
import { ligandName, nameSmilesLigands, smilesComponent }
  from "../src/chem/component.js";
import { bondLength, idealAngle } from "../src/chem/geometry-tables.js";

/**
 * 🔴 THE GEOMETRY GATE IS `tools/check-smiles-conformer.mjs`, NOT THIS FILE.
 * It holds 51 molecules to RDKit's MMFF conformers and needs a dump this
 * suite cannot generate. What is here is what holds with no reference at all:
 * that the bounds are satisfiable, that the conformer satisfies them, that a
 * ring is flat and a mirror image is not built.
 */

const distance = (component, a, b) => Math.hypot(
  component.atoms[a].x - component.atoms[b].x,
  component.atoms[a].y - component.atoms[b].y,
  component.atoms[a].z - component.atoms[b].z);

describe("distance bounds", () => {
  it("are satisfiable for every molecule it is given", () => {
    for (const smiles of [
      "C", "CC", "OCC(O)CO", "c1ccccc1", "C1CCCCC1", "C1CC1",
      "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
      "CC1(C)S[C@@H]2[C@H](NC(=O)C)C(=O)N2[C@H]1C(=O)O",
    ]) {
      const bounds = distanceBounds(parseSmiles(smiles));
      assert.equal(smoothBounds(bounds), 0, `${smiles} has contradictory bounds`);
    }
  });

  it("never puts a lower bound above an upper", () => {
    const bounds = distanceBounds(parseSmiles("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"));
    smoothBounds(bounds);
    const { lower, upper, n } = bounds;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        assert.ok(lower[i * n + j] <= upper[i * n + j] + 1e-9,
                  `pair ${i}-${j}: ${lower[i * n + j]} > ${upper[i * n + j]}`);
      }
    }
  });

  it("puts a bonded pair at the bond's length, not at a van der Waals floor", () => {
    // 🔴 THE FLOOR MUST NOT REACH A BOND. 0.8 times two carbons' van der Waals
    // radii is 2.72 A and a C-C bond is 1.52 - applied to every pair, as the
    // first version did, that lower bound sits above the bond's own upper and
    // nothing can be embedded at all.
    const bounds = distanceBounds(parseSmiles("CC"));
    smoothBounds(bounds);
    assert.ok(bounds.upper[1] < 1.6, `C-C upper is ${bounds.upper[1]}`);
    assert.ok(bounds.lower[1] > 1.4, `C-C lower is ${bounds.lower[1]}`);
  });
});

describe("the conformer", () => {
  it("places a benzene flat", async () => {
    const component = await smilesComponent("c1ccccc1");
    // Every atom on the plane of the first three.
    const p = component.atoms.map((atom) => [atom.x, atom.y, atom.z]);
    const u = [0, 1, 2].map((k) => p[1][k] - p[0][k]);
    const v = [0, 1, 2].map((k) => p[2][k] - p[0][k]);
    const normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0]];
    const length = Math.hypot(...normal);
    for (let atom = 3; atom < 6; atom += 1) {
      const out = Math.abs([0, 1, 2].reduce((total, k) =>
        total + (p[atom][k] - p[0][k]) * normal[k], 0) / length);
      assert.ok(out < 0.1, `atom ${atom} is ${out.toFixed(2)} A out of plane`);
    }
  });

  it("gives a benzene six equal bonds near 1.40", async () => {
    const component = await smilesComponent("c1ccccc1");
    for (let atom = 0; atom < 6; atom += 1) {
      const d = distance(component, atom, (atom + 1) % 6);
      assert.ok(Math.abs(d - 1.40) < 0.06, `bond ${atom} is ${d.toFixed(3)}`);
    }
  });

  it("builds the enantiomer the SMILES asked for, both ways round", async () => {
    // 🔴 THE ONE THING DISTANCE GEOMETRY CANNOT RECOVER BY ITSELF: both
    // enantiomers have identical pairwise distances, so without the chiral
    // term the answer is whichever the random draw landed nearest.
    for (const [smiles, expected] of [["C[C@@H](N)C(=O)O", -1], ["C[C@H](N)C(=O)O", 1]]) {
      const graph = parseSmiles(smiles);
      const component = await smilesComponent(smiles);
      const point = new Float64Array(3 * component.atoms.length);
      component.atoms.forEach((atom, index) => {
        point[index * 3] = atom.x;
        point[index * 3 + 1] = atom.y;
        point[index * 3 + 2] = atom.z;
      });
      const [centre] = chiralCentres(graph);
      assert.ok(centre !== undefined, `${smiles} has no centre`);
      assert.equal(Math.sign(signedVolume(point, ...centre.neighbours)), expected,
                   `${smiles} came out as the other enantiomer`);
    }
  });

  it("is the same conformer twice, because ref_pos is an input", async () => {
    // A ligand whose reference conformer moved between runs would make every
    // fold using it irreproducible, and the difference would surface as an
    // unexplained seed effect somewhere else entirely.
    const first = await smilesComponent("CC(=O)Oc1ccccc1C(=O)O");
    const second = await smilesComponent("CC(=O)Oc1ccccc1C(=O)O");
    assert.deepEqual(first.atoms.map((a) => [a.x, a.y, a.z]),
                     second.atoms.map((a) => [a.x, a.y, a.z]));
  });

  it("returns exactly the shape parseCcdComponent returns", async () => {
    // 🔴 THE WHOLE INTEGRATION IS THIS ASSERTION. Everything downstream - the
    // atomised tokens, ref_pos, the bond matrix, the chirality centres,
    // bond-geometry.js - reads this object and cannot tell which producer
    // made it.
    const component = await smilesComponent("OCC(O)CO", { code: "GOL" });
    assert.equal(component.code, "GOL");
    assert.equal(component.atoms.length, 6);
    for (const atom of component.atoms) {
      assert.equal(typeof atom.name, "string");
      assert.ok(atom.name.length > 0 && atom.name.length <= 4);
      assert.equal(typeof atom.element, "number");
      assert.equal(typeof atom.charge, "number");
      assert.ok(Number.isFinite(atom.x) && Number.isFinite(atom.y)
        && Number.isFinite(atom.z));
      assert.equal(atom.leaving, false);
    }
    for (const bond of component.bonds) {
      assert.ok(Number.isInteger(bond.from) && Number.isInteger(bond.to));
      assert.ok(bond.order >= 1 && bond.order <= 4);
    }
    // Names are the dictionary's convention: element plus a counter.
    assert.deepEqual(component.atoms.map((atom) => atom.name),
                     ["O1", "C1", "C2", "O2", "C3", "O3"]);
  });

  it("separates two fragments rather than overlapping them", async () => {
    const component = await smilesComponent("[Na+].[Cl-]");
    assert.ok(distance(component, 0, 1) > 1.5);
  });
});

describe("the geometry tables", () => {
  it("knows a shape needs more than a neighbour count", () => {
    // 🔴 AMMONIA AND FORMALDEHYDE BOTH HAVE THREE SIGMA BONDS and are 107 and
    // 120 degrees. Keying on the count alone had both cases backwards and
    // produced contradictory bounds around ATP's triphosphate.
    const degrees = (radians) => (radians * 180) / Math.PI;
    assert.ok(Math.abs(degrees(idealAngle(3, "N", false)) - 107) < 0.1);
    assert.ok(Math.abs(degrees(idealAngle(3, "C", true)) - 120) < 0.1);
    assert.ok(Math.abs(degrees(idealAngle(4, "P", false)) - 109.47) < 0.1);
    assert.ok(Math.abs(degrees(idealAngle(2, "O", false)) - 104.5) < 0.1);
    assert.ok(Math.abs(degrees(idealAngle(2, "C", "two")) - 180) < 0.1);
    // A sulfoxide keeps its lone pair and stays pyramidal.
    assert.ok(Math.abs(degrees(idealAngle(3, "S", true)) - 106) < 0.1);
  });

  it("places an aromatic bond between the single and the double", () => {
    const single = bondLength("C", "C", 1, false);
    const aromatic = bondLength("C", "C", 1, true);
    const double = bondLength("C", "C", 2, false);
    assert.ok(double < aromatic && aromatic < single,
              `${double} < ${aromatic} < ${single}`);
    assert.ok(Math.abs(aromatic - 1.40) < 0.03, `benzene bond is ${aromatic}`);
  });
});

describe("stereo perception", () => {
  it("reads a ring-opening centre in written order", () => {
    // 🔴 A RING CLOSURE IS WRITTEN AT THE DIGIT AND CREATED WHEN IT CLOSES.
    // Ordering a centre's neighbours by bond index puts it last and inverts
    // the centre: 9 of 29 corpus centres, every one a ring opening.
    const graph = parseSmiles("OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12");
    const written = graph.writtenNeighbours[7]
      .map((bond) => (graph.bonds[bond].from === 7
        ? graph.bonds[bond].to : graph.bonds[bond].from));
    assert.deepEqual(written, [6, 15, 8]);
  });

  it("reads cis and trans the way the slashes actually mean", () => {
    // `F/C=C/F` is TRANS, which is the reverse of how it looks.
    const [trans] = doubleBondStereo(parseSmiles("F/C=C/F"));
    const [cis] = doubleBondStereo(parseSmiles("F/C=C\\F"));
    assert.equal(trans.cis, false);
    assert.equal(cis.cis, true);
  });

  it("refuses to call an atom a centre when it cannot be one", () => {
    assert.equal(chiralCentres(parseSmiles("CC(N)O")).length, 0);
    assert.equal(chiralCentres(parseSmiles("C[C@@H](N)C(=O)O")).length, 1);
  });
});

describe("rings feed the geometry", () => {
  it("finds the rings a fused system has", () => {
    assert.equal(smallestRings(parseSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")).length, 2);
  });
});

describe("the boundaries, and the promise that a conformer does not drift", () => {
  /**
   * 🔴 `ref_pos` IS A MODEL INPUT, SO A CONFORMER THAT MOVED BETWEEN RUNS
   * WOULD MAKE EVERY FOLD USING IT IRREPRODUCIBLE - and the difference would
   * surface as an unexplained seed effect somewhere else entirely. The
   * in-process check is two directories up; this is the stronger claim, that
   * nothing in the module reads a clock or a global.
   *
   * Verified by inspection too: `src/chem/` contains no `Math.random`, no
   * `Date.now`, no `fetch`, no `process`, no `document`, and imports exactly
   * one thing from outside itself (`ELEMENT_SYMBOLS`). The seed is
   * `hashOf(smiles)`, which is FNV-1a over the string.
   */
  it("reads no clock, no global and no network", () => {
    for (const file of ["smiles.js", "kekulize.js", "rings.js", "stereo.js",
                        "geometry-tables.js", "conformer.js", "component.js"]) {
      const text = readFileSync(new URL(`../src/chem/${file}`, import.meta.url), "utf8");
      // Comments are stripped first: several of them discuss `performance` and
      // a gate that matches its own prose is a gate that cannot be documented.
      const code = text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
      for (const forbidden of [/Math\.random/, /Date\.now/, /\bfetch\s*\(/,
                               /\bprocess\./, /\bdocument\./, /\bwindow\./]) {
        assert.ok(!forbidden.test(code), `${file} uses ${forbidden}`);
      }
    }
  });

  it("names at most 99 distinct ligands, in three characters each", () => {
    // 🔴 THREE, BECAUSE `src/af3/fold.js` WRITES THE RESIDUE NAME WITH
    // `.padEnd(3)` INTO A FIXED-WIDTH COLUMN. A fourth character runs into the
    // chain id, and `LIG2` truncated back to `LIG` puts the collision that
    // this naming exists to prevent straight back.
    const seen = new Set();
    for (let index = 0; index < 99; index += 1) {
      const name = ligandName(index);
      assert.equal(name.length, 3, `${index} -> ${name}`);
      seen.add(name);
    }
    assert.equal(seen.size, 99);
    assert.equal(ligandName(0), "LIG");
    assert.throws(() => ligandName(99), /at most 99/);
  });

  it("shares a name between identical structures and not between others", () => {
    assert.deepEqual(nameSmilesLigands(["c1ccccc1", "OCCO", "c1ccccc1"]),
                     ["LIG", "LG2", "LIG"]);
  });

  it("survives inputs nobody would type on purpose", () => {
    // Each of these reached a different corner of the parser; none may hang,
    // and none may return something that is not a molecule.
    const odd = [
      ["deep nesting", `C${"(C".repeat(60)}${")".repeat(60)}`],
      ["a two-hundred-membered ring", `C1${"C".repeat(200)}1`],
      ["nothing but fragments", "C.C.C.C.C"],
      ["surrounding whitespace", "  CCO  "],
    ];
    for (const [what, smiles] of odd) {
      const graph = parseSmiles(smiles);
      assert.ok(graph.atoms.length > 0, what);
      assert.deepEqual(valenceProblems(graph), [], what);
    }
    // ...and one that must be REFUSED rather than survived: the same pair
    // bonded twice through two ring-closure digits.
    assert.throws(() => parseSmiles("C12CCC12"), /bonded twice/);
  });
});
