#!/usr/bin/env node
/**
 * Does a `@` in a SMILES mean the same hand here as it does to RDKit?
 *
 * 🔴 THIS IS A CALIBRATION AS MUCH AS A GATE, AND THAT IS WHY IT EXISTS. The
 * OpenSMILES text fixes what `@` MEANS - looking from the first neighbour, the
 * other three anticlockwise in written order - but which sign of which
 * determinant that corresponds to depends on a handedness convention that is
 * easy to assert from memory and get exactly backwards. Backwards is the worst
 * possible failure here: every centre inverts, the conformer is a perfectly
 * good molecule, every bond length and angle is right, and it is the other
 * enantiomer. Nothing downstream would object.
 *
 * So the sign is MEASURED. RDKit's dump carries a 3D conformer and the CIP
 * code it assigned to each centre; this reads the conformer with this port's
 * own neighbour ordering, computes the signed volume, and checks that the
 * sign this port predicts from the `@` is the sign RDKit's geometry actually
 * has.
 *
 *   node tools/check-stereo-vs-rdkit.mjs
 */

import { readFileSync } from "node:fs";
import { parseSmiles } from "../src/chem/smiles.js";
import { chiralCentres, doubleBondStereo } from "../src/chem/stereo.js";
import { signedVolume } from "../src/chem/conformer.js";

const DUMP = "oracle-dumps/rdkit-smiles.json";

/**
 * 🔴 A CHECKER THAT CRASHES ON A MISSING DUMP IS A CHECKER NOBODY CAN RUN.
 * docs/PARITY.md's standing complaint is that nineteen of twenty-one AF3
 * checkers "404 on a bundle or a dump rather than compare anything", so a
 * suite run reads as things that did not object. This one says what it needs
 * and how to make it.
 */
function readDump() {
  try {
    return JSON.parse(readFileSync(DUMP, "utf8"));
  } catch {
    console.error(`no ${DUMP}. Generate it with:\n`
      + "  /home/ubuntu/.venv-rdkit/bin/python tools/oracle/dump_rdkit_smiles.py");
    process.exit(2);
  }
  return null;
}

const dump = readDump();

let centres = 0;
let agree = 0;
const wrong = [];

for (const record of dump.records) {
  if (record.conformer === undefined || record.stereo.length === 0) continue;
  let graph;
  try { graph = parseSmiles(record.smiles); } catch { continue; }
  if (graph.atoms.length !== record.conformer.length) continue;

  const point = new Float64Array(3 * graph.atoms.length);
  record.conformer.forEach((position, index) => {
    point[index * 3] = position[0];
    point[index * 3 + 1] = position[1];
    point[index * 3 + 2] = position[2];
  });

  for (const centre of chiralCentres(graph)) {
    const [a, b, c, d] = centre.neighbours;
    const volume = signedVolume(point, a, b, c, d);
    centres += 1;
    if (Math.sign(volume) === Math.sign(centre.sign)) agree += 1;
    else wrong.push(`${record.name} atom ${centre.atom}`
      + ` (${graph.atoms[centre.atom].symbol}${graph.atoms[centre.atom].chirality})`
      + ` volume ${volume.toFixed(2)}, this port wants ${centre.sign > 0 ? "+" : "-"}`);
  }
}

// The double-bond half: RDKit names STEREOE / STEREOZ / STEREOCIS / STEREOTRANS
// on the bond, and what is compared is the geometry it actually produced.
let bonds = 0;
let bondsAgree = 0;
for (const record of dump.records) {
  if (record.conformer === undefined) continue;
  let graph;
  try { graph = parseSmiles(record.smiles); } catch { continue; }
  if (graph.atoms.length !== record.conformer.length) continue;
  const at = (index) => record.conformer[index];
  for (const stereo of doubleBondStereo(graph)) {
    const torsion = dihedral(at(stereo.first), at(stereo.from),
                             at(stereo.to), at(stereo.second));
    const isCis = Math.abs(torsion) < Math.PI / 2;
    bonds += 1;
    if (isCis === stereo.cis) bondsAgree += 1;
    else {
      wrong.push(`${record.name} bond ${stereo.from}=${stereo.to}:`
        + ` this port says ${stereo.cis ? "cis" : "trans"},`
        + ` RDKit's conformer has a torsion of ${((torsion * 180) / Math.PI).toFixed(0)} deg`);
    }
  }
}

function dihedral(p, q, r, s) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                           a[2] * b[0] - a[0] * b[2],
                           a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const b1 = sub(q, p); const b2 = sub(r, q); const b3 = sub(s, r);
  const n1 = cross(b1, b2); const n2 = cross(b2, b3);
  const m = cross(n1, [b2[0] / Math.hypot(...b2), b2[1] / Math.hypot(...b2),
                       b2[2] / Math.hypot(...b2)]);
  return Math.atan2(dot(m, n2), dot(n1, n2));
}

for (const line of wrong.slice(0, 12)) console.log(`  ${line}`);
if (wrong.length > 12) console.log(`  ...and ${wrong.length - 12} more`);
console.log(`\ntetrahedral centres: ${agree}/${centres} agree with RDKit's geometry`);
console.log(`double bonds:        ${bondsAgree}/${bonds} agree`);
if (agree !== centres || bondsAgree !== bonds) process.exit(1);
