#!/usr/bin/env node
/**
 * A SMILES ligand featurises to the SAME BATCH its CCD code does, field by field.
 *
 *     npm run test:smiles-batch
 *
 * 🔴 THIS IS THE SHARPEST TEST OF THE WHOLE SMILES PATH AND THE ONLY ONE THAT
 * CAN SEE A SILENTLY DIFFERENT MOLECULE. A fold gate compares structures, and
 * a structure is a noisy, sampled, model-dependent thing - `check-smiles-path`
 * cannot tell a 15% rescaled conformer from a correct one, and said so. The
 * BATCH is what the model actually reads: the token types, the element
 * numbers, the reference positions, the bond matrix, the bond-order plane, the
 * atom windows, the chirality centres. If every field is identical then the
 * model cannot distinguish the two routes even in principle, and no amount of
 * sampling noise is in the way.
 *
 * 🔴 AND IT WORKS BECAUSE ONE SMILES CAN BE WRITTEN IN THE DICTIONARY'S OWN
 * ATOM ORDER. `OCC(O)CO` is the obvious way to write glycerol and puts oxygen
 * first, where the CCD's GOL is C1 O1 C2 O2 C3 O3; `C(O)C(O)CO` is the same
 * molecule in the dictionary's order. That removes the permutation and lets
 * the comparison be elementwise, which is enormously stronger than comparing
 * two sets. A field that differs is then a real semantic difference rather
 * than a relabelling.
 */

import { readFileSync } from "node:fs";
import { af3BatchFromA3m } from "../src/af3/featurise/batch.js";
import { parseCcdComponent } from "../src/af3/featurise/ccd-component.js";
import { smilesComponent } from "../src/chem/component.js";
import { featuriserDialect, dialectFor } from "../src/af3/dialect.js";

const SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";

/**
 * Each case: a CCD code, and the SMILES written in that entry's ATOM ORDER.
 *
 * 🔴 WRITTEN BY HAND AGAINST THE DICTIONARY, WHICH IS THE WHOLE TRICK. `OCC(O)CO`
 * is the obvious glycerol and puts oxygen first, where the CCD's GOL is
 * C1 O1 C2 O2 C3 O3; `C(O)C(O)CO` is the same molecule in the dictionary's
 * order. Removing the permutation is what lets the comparison be elementwise,
 * which is enormously stronger than comparing two sets.
 *
 * `namesDiffer` marks an entry whose dictionary names are not this port's
 * element-plus-counter convention - see the note on `namesDiffer` below.
 */
const CASES = [
  { code: "GOL", smiles: "C(O)C(O)CO" },                      // C1 O1 C2 O2 C3 O3
  { code: "EDO", smiles: "C(O)CO" },                          // C1 O1 C2 O2
  { code: "URE", smiles: "C(=O)(N)N" },                       // C O N1 N2
  // ...and two whose dictionary names this port cannot reproduce.
  { code: "ACE", smiles: "C(=O)C", namesDiffer: true },       // C O CH3
  { code: "BEN", smiles: "c1(ccccc1)C(=N)N", namesDiffer: true },  // C1..C6 C N1 N2
];

/**
 * 🔴 AN ATOM NAME IS A MODEL INPUT AND A SMILES HAS NO RIGHT ANSWER FOR IT.
 * `refAtomNameChars` carries the atom's name into the features, and this port
 * names a built component element-plus-counter - C1, C2, O1 - which is the
 * dictionary's own convention for many entries and not for all of them: ACE
 * calls its methyl `CH3` and BEN calls its seventh carbon plain `C`. For a
 * ligand that has no CCD entry, which is the whole point of the SMILES path,
 * there is no dictionary name to match and any consistent scheme is as good as
 * another. For one that DOES have an entry, folding it by code rather than by
 * structure gives the model the dictionary's names, and folding it by
 * structure gives it these - a real difference, small, and not a bug in either
 * direction. Stated here rather than hidden by excluding those entries.
 */
const NAME_FIELDS = ["refAtomNameChars", "displayAtomNameChars"];

/**
 * 🔴 `ref_pos` IS EXPECTED TO DIFFER AND IS REPORTED, NOT ASSERTED. It is the
 * one field that is a CONFORMER rather than a fact about the molecule: the
 * dictionary ships an experimentally-derived ideal and this port builds one
 * from distance geometry, and two conformers of one molecule are both correct.
 * docs/AF3.md already treats exactly this field as a reported floor between
 * this port's own idealised amino acids and the reference's CCD geometry, at
 * 0.65 A rms. Everything else is a fact and is asserted.
 */
const REPORTED = new Set(["ref_pos", "refPos", "refSpaceUid", "ref_space_uid"]);

const shapeOf = (value) => (ArrayBuffer.isView(value) || Array.isArray(value)
  ? `[${value.length}]` : typeof value);

function compare(a, b, path, differences) {
  if (ArrayBuffer.isView(a) || Array.isArray(a)) {
    if (!(ArrayBuffer.isView(b) || Array.isArray(b)) || a.length !== b.length) {
      differences.push(`${path}: ${shapeOf(a)} against ${shapeOf(b)}`);
      return;
    }
    let count = 0;
    let worst = 0;
    let where = -1;
    for (let index = 0; index < a.length; index += 1) {
      const x = a[index];
      const y = b[index];
      if (typeof x === "number" && typeof y === "number") {
        const gap = Math.abs(x - y);
        if (gap > 1e-6) {
          count += 1;
          if (gap > worst) { worst = gap; where = index; }
        }
      } else if (x !== null && typeof x === "object") {
        // 🔴 RECURSED INTO, NOT STRINGIFIED. An array of objects compared with
        // `!==` reports every entry as differing and prints "[object Object]
        // against [object Object]", which looks like a serious finding and
        // says nothing at all. `ligandSpans` is exactly that shape.
        compare(x, y, `${path}[${index}]`, differences);
      } else if (x !== y) {
        count += 1;
        if (where < 0) where = index;
      }
    }
    if (count > 0) {
      differences.push(`${path}: ${count} of ${a.length} differ,`
        + ` worst ${worst.toExponential(2)} at ${where}`
        + ` (${a[where]} against ${b[where]})`);
    }
    return;
  }
  if (a !== null && typeof a === "object") {
    const keys = new Set([...Object.keys(a), ...Object.keys(b ?? {})]);
    for (const key of keys) compare(a[key], b?.[key], `${path}/${key}`, differences);
    return;
  }
  if (a !== b) differences.push(`${path}: ${a} against ${b}`);
}

let failures = 0;
for (const { code, smiles, namesDiffer } of CASES) {
  const dictionary = parseCcdComponent(
    readFileSync(`tools/fixtures/ccd/${code}.cif`, "utf8"));
  const built = await smilesComponent(smiles, { code });

  // 🔴 THE COMPONENTS ARE CHECKED AGAINST EACH OTHER FIRST. If the two are not
  // the same molecule in the same order, every batch difference below is
  // explained by that and the comparison says nothing.
  //
  // 🔴 THE BONDS ARE COMPARED AS A SET, NOT IN LIST ORDER. Which order a
  // dictionary lists its bonds in is a property of the file: BEN's start
  // `0-1, 0-5, 0-6, 1-2` where a SMILES walk emits `0-1, 1-2, 2-3`. Requiring
  // the list to match made a molecule that IS the same molecule fail the
  // precondition, which would have excluded the only aromatic case here.
  // Nothing downstream reads the bond list's order - the batch indexes bonds
  // by their atoms - so the set is the right comparison.
  const asSet = (bonds) => new Set(bonds.map((bond) =>
    `${Math.min(bond.from, bond.to)}-${Math.max(bond.from, bond.to)}`));
  const ours = asSet(built.bonds);
  const theirs = asSet(dictionary.bonds);
  const sameOrder = dictionary.atoms.length === built.atoms.length
    && dictionary.atoms.every((atom, index) => atom.element === built.atoms[index].element)
    && ours.size === theirs.size && [...theirs].every((key) => ours.has(key));
  if (!sameOrder) {
    console.log(`FAIL  ${code}: the SMILES is not the dictionary's molecule in its order`);
    failures += 1;
    continue;
  }

  for (const model of ["alphafold3", "boltz2", "rosettafold3", "intellifold2"]) {
    const options = {
      ligands: [dictionary],
      ...featuriserDialect(dialectFor(model)),
    };
    const fromCode = af3BatchFromA3m(SEQUENCE, null, options);
    const fromSmiles = af3BatchFromA3m(SEQUENCE, null,
                                       { ...options, ligands: [built] });

    const differences = [];
    compare(fromCode.batch, fromSmiles.batch, "", differences);
    const excused = namesDiffer === true ? [...REPORTED, ...NAME_FIELDS] : [...REPORTED];
    const real = differences.filter((line) =>
      !excused.some((field) => line.includes(`/${field}:`))
      // 🔴 THE BOND LIST'S ORDER IS EXCUSED AND THE BOND MATRIX IS NOT, WHICH
      // IS THE WHOLE JUSTIFICATION. `ligandSpans[n].bonds` is the bonds in
      // whatever order they were read - the dictionary's BEN starts
      // `0-1, 0-5, 0-6, 1-2` and a SMILES walk emits `0-1, 1-2, 2-3` - and
      // that ordering reaches nothing: `bondMatrix` and `bondOrderMatrix` are
      // what the model reads, they are derived from this list, and they are
      // compared elementwise two lines above and come out IDENTICAL. So the
      // excuse is not "this field does not matter", it is "the field this
      // field decides is asserted and agrees".
      && !/\/ligandSpans\[\d+\]\/bonds\[/.test(line));

    // ...and that claim is only safe while those two really are compared.
    for (const field of ["bondMatrix", "bondOrderMatrix"]) {
      if (fromCode.batch[field] === undefined) {
        real.push(`${field} is absent from the batch, so excusing the bond`
          + " list's order is unjustified");
      }
    }
    const reported = differences.filter((line) => !real.includes(line));

    if (real.length > 0) failures += 1;
    console.log(`${real.length === 0 ? "ok  " : "FAIL"}  ${code} · ${model.padEnd(13)}`
      + ` ${Object.keys(fromCode.batch).length} fields,`
      + ` ${real.length} differ`
      + (reported.length > 0 ? `  (${reported.length} reported)` : ""));
    for (const line of real.slice(0, 6)) console.log(`        ${line}`);
    for (const line of reported) console.log(`        reported: ${line.slice(0, 110)}`);
  }
}

console.log(failures === 0
  ? "\na SMILES ligand and its CCD code featurise to the same batch"
  : `\n🔴 ${failures} comparison(s) differ`);
if (failures > 0) process.exitCode = 1;
