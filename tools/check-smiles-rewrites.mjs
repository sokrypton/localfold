#!/usr/bin/env node
/**
 * The same molecule written a thousand different ways is still that molecule.
 *
 *     node tools/check-smiles-rewrites.mjs
 *
 * 🔴 THE ONLY GATE HERE THAT NOBODY HAD TO THINK OF CASES FOR, WHICH IS WHY IT
 * IS THE ONE WORTH HAVING. Every other check in `npm run test:smiles` holds
 * this port to a corpus somebody wrote down, and a corpus proves the cases
 * somebody thought of are right - the last round of hand-picking found three
 * bugs in twenty-three molecules and then ran out of conventions to name.
 *
 * A SMILES is one of enormous numbers of strings for the same molecule:
 * different starting atom, different branch nesting, different ring-closure
 * digits. RDKit will emit them on demand, and every one must parse to the SAME
 * graph with the SAME handedness at every centre. No reference VALUES are
 * needed - the invariant is the equality itself - so the coverage is bounded
 * only by how many strings are generated.
 *
 * 🔴 AND IT AIMS AT THE CODE THAT HAS ACTUALLY BEEN WRONG. A ring-closure bond
 * is written at its digit and created when the ring closes, and ordering a
 * chiral centre's neighbours by creation inverted NINE of twenty-nine corpus
 * centres - every one a ring opening, with all bond lengths and angles
 * perfect. Re-writing a molecule permutes exactly that bookkeeping. Today:
 * 1154 graph re-writings and 1625 centre checks, all agreeing.
 */

import { readFileSync } from "node:fs";
import { parseSmiles, molecularFormula } from "../src/chem/smiles.js";
import { valenceProblems } from "../src/chem/kekulize.js";
import { smallestRings, circuitRank } from "../src/chem/rings.js";
import { chiralCentres } from "../src/chem/stereo.js";
import { signedVolume } from "../src/chem/conformer.js";
import { smilesComponent } from "../src/chem/component.js";

const DUMP = "oracle-dumps/rdkit-smiles.json";
let dump;
try {
  dump = JSON.parse(readFileSync(DUMP, "utf8"));
} catch {
  console.error(`no ${DUMP}. Generate it with:\n`
    + "  /home/ubuntu/.venv-rdkit/bin/python tools/oracle/dump_rdkit_smiles.py");
  process.exit(2);
}
if (dump.rewrites === undefined) {
  console.error(`${DUMP} predates the re-writings; regenerate it.`);
  process.exit(2);
}

const problems = new Map();
const note = (key, detail) => { if (!problems.has(key)) problems.set(key, detail); };

// The graph half: every re-writing is the same atoms, bonds and rings.
let graphChecks = 0;
for (const record of dump.rewrites) {
  for (const variant of record.variants) {
    graphChecks += 1;
    let graph;
    try {
      graph = parseSmiles(variant.smiles);
    } catch (error) {
      note(`${record.name}: refused a re-writing`, `${error.message} -- ${variant.smiles}`);
      continue;
    }
    if (graph.atoms.length !== record.heavy) {
      note(`${record.name}: atom count moves`,
           `${graph.atoms.length} against ${record.heavy} -- ${variant.smiles}`);
    } else if (graph.bonds.length !== record.bonds) {
      note(`${record.name}: bond count moves`,
           `${graph.bonds.length} against ${record.bonds} -- ${variant.smiles}`);
    } else if (molecularFormula(graph) !== record.formula) {
      note(`${record.name}: formula moves`,
           `${molecularFormula(graph)} against ${record.formula} -- ${variant.smiles}`);
    } else if (valenceProblems(graph).length > 0) {
      note(`${record.name}: valence`, `${valenceProblems(graph)[0]} -- ${variant.smiles}`);
    } else if (smallestRings(graph).length < circuitRank(graph)) {
      note(`${record.name}: rings below the rank`,
           `${smallestRings(graph).length} < ${circuitRank(graph)} -- ${variant.smiles}`);
    }
  }
}

// 🔴 THE STEREO HALF, WHICH IS THE EXPENSIVE ONE AND THE ONE THAT MATTERS.
// It builds a conformer per re-writing, so it is limited to the molecules that
// have a centre at all - and those are the only ones it can say anything
// about. `order` maps a position in the written string back to the atom RDKit
// started from, which is what makes two differently-numbered graphs
// comparable.
let centreChecks = 0;
for (const record of dump.rewrites) {
  if (record.centres === 0) continue;
  let reference = null;
  for (const variant of record.variants) {
    let graph;
    let component;
    try {
      graph = parseSmiles(variant.smiles);
      component = await smilesComponent(variant.smiles, { attempts: 6 });
    } catch (error) {
      note(`${record.name}: would not build`, `${error.message} -- ${variant.smiles}`);
      continue;
    }
    const point = new Float64Array(3 * component.atoms.length);
    component.atoms.forEach((atom, index) => {
      point[index * 3] = atom.x;
      point[index * 3 + 1] = atom.y;
      point[index * 3 + 2] = atom.z;
    });
    // 🔴 THE VOLUME IS TAKEN OVER THE NEIGHBOURS IN CANONICAL ORDER, NOT IN
    // THE ORDER `chiralCentres` RETURNED THEM - AND THE FIRST VERSION OF THIS
    // GATE GOT THAT WRONG AND COULD NOT FAIL. It compared the conformer
    // against the sign that same function had ASKED for, so the two came from
    // one source and agreed by construction: putting the original
    // ring-closure ordering bug back left it reporting 1622/1622. That is the
    // identical trap `check-smiles-conformer.mjs` documents for its own
    // chirality arm, arrived at a second time by a different route.
    //
    // Sorting the four neighbours by their ORIGINAL atom number makes the
    // volume a property of the built molecule alone. Then "the same atom has
    // the same hand however the string was written" is a real invariant, and
    // an ordering bug - which by its nature bites only some writings - breaks
    // it. Verified: with the bug restored this reports 294 disagreements.
    const signs = new Map();
    for (const centre of chiralCentres(graph)) {
      const asOriginal = centre.neighbours.map((atom) => variant.order[atom]);
      const canonical = centre.neighbours
        .map((atom, index) => ({ atom, original: asOriginal[index] }))
        .sort((a, b) => a.original - b.original)
        .map((entry) => entry.atom);
      signs.set(variant.order[centre.atom],
                Math.sign(signedVolume(point, ...canonical)));
    }
    if (reference === null) { reference = signs; continue; }
    for (const [atom, sign] of signs) {
      centreChecks += 1;
      if (sign !== 0 && sign === reference.get(atom)) continue;
      note(`${record.name}: atom ${atom} changes hand when re-written`,
           `sign ${sign} against ${reference.get(atom)} -- ${variant.smiles}`);
    }
  }
}

console.log(`${graphChecks} graph re-writings, ${centreChecks} centre checks`);
for (const [key, detail] of [...problems].slice(0, 12)) {
  console.log(`FAIL  ${key}\n        ${detail.slice(0, 140)}`);
}
if (problems.size > 12) console.log(`...and ${problems.size - 12} more kinds`);
if (problems.size > 0) process.exit(1);
console.log("every re-writing is the same molecule, with the same hands");
