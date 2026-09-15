import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { bondGeometry, parsePdbResidues } from "../tools/gpu/bond-geometry.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const conformers = JSON.parse(
  readFileSync(join(ROOT, "tools/oracle/reference-conformers.json"), "utf8"));
const crystal = readFileSync(join(ROOT, "tools/fixtures/6mrr-crystal.pdb"), "utf8");

/**
 * 🔴 A DEPOSITED STRUCTURE IS THE CALIBRATION, because a scorer with no known
 * answer is a scorer nobody can trust. 6MRR's crystal was refined against real
 * chemistry, so its bonds are the lengths the reference conformer states - and
 * if this read otherwise, the fault would be here and not in the crystal.
 */
test("a deposited crystal scores near zero on every class", () => {
  const seen = bondGeometry(crystal, conformers);
  assert.ok(seen.mainchain.bonds > 100, `only ${seen.mainchain.bonds} mainchain bonds`);
  assert.ok(seen.sidechain.bonds > 100, `only ${seen.sidechain.bonds} sidechain bonds`);
  assert.ok(seen.peptide.bonds > 50, `only ${seen.peptide.bonds} peptide bonds`);
  for (const kind of ["mainchain", "sidechain", "peptide"]) {
    assert.ok(seen[kind].rms < 0.10,
      `${kind} rms ${seen[kind].rms?.toFixed(3)} on a CRYSTAL - the scorer is wrong`);
  }
});

/**
 * 🔴 AND IT MUST FAIL, which is the half that makes it a gate. Stretching one
 * axis stretches every bond with it; a scorer that still reported zero would be
 * comparing the structure against itself.
 */
test("a stretched structure is caught, and in the right class", () => {
  const stretched = crystal.split("\n").map((line) => {
    if (!line.startsWith("ATOM")) return line;
    const x = Number(line.slice(30, 38)) * 1.10;
    return line.slice(0, 30) + x.toFixed(3).padStart(8) + line.slice(38);
  }).join("\n");
  const seen = bondGeometry(stretched, conformers);
  assert.ok(seen.mainchain.rms > 0.03,
    `a 10% stretch read as mainchain rms ${seen.mainchain.rms?.toFixed(4)}`);
  assert.ok(seen.sidechain.rms > 0.03,
    `a 10% stretch read as sidechain rms ${seen.sidechain.rms?.toFixed(4)}`);
  assert.ok(seen.worst[0].error > 0.05, "worst offender should name a real error");
});

/**
 * 🔴 A LIGAND'S BONDS MUST BE GIVEN, NEVER INFERRED FROM THE PREDICTION - that
 * would score the fold against itself and pass anything. With no table the
 * class is empty rather than clean, which is a different report.
 */
test("a component with no conformer scores nothing unless its bonds are supplied", () => {
  const glycerol = [
    "HETATM    1  C1  GOL A 100       0.000   0.000   0.000  1.00 50.00           C",
    "HETATM    2  O1  GOL A 100       1.430   0.000   0.000  1.00 50.00           O",
  ].join("\n");
  assert.equal(bondGeometry(glycerol, conformers).ligand.bonds, 0);
  const scored = bondGeometry(glycerol, conformers,
    { ligandBonds: [["C1", "O1", 1.43]] });
  assert.equal(scored.ligand.bonds, 1);
  assert.ok(scored.ligand.rms < 0.001, "an exact ligand bond must score zero");
});

test("residues are grouped by chain and number, not by name alone", () => {
  const two = [
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C",
    "ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00 50.00           C",
  ].join("\n");
  assert.equal(parsePdbResidues(two).length, 2);
});
