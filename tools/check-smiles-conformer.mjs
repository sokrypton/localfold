#!/usr/bin/env node
/**
 * Is the conformer this port builds a real molecule? Bonds, angles, rings, hands.
 *
 * 🔴 IT DOES NOT COMPARE COORDINATES AND MUST NOT. A molecule has many
 * conformers, distance geometry is seeded, and RDKit's ETKDG draws from its own
 * distribution - so two correct programs will place the same molecule
 * differently and an RMSD between them measures nothing. What is the same in
 * every conformer of a molecule is its LOCAL geometry: the bond lengths, the
 * bond angles, the flatness of an aromatic ring, and the handedness of a
 * stereocentre. Those are what this compares, distribution against
 * distribution.
 *
 *   node tools/check-smiles-conformer.mjs [--verbose]
 *
 * The bars, and where each comes from:
 *
 *   bond length   0.05 A mean. The table's own error against RDKit is 0.036 A
 *                 (see geometry-tables.js), so this is that plus the
 *                 embedding's share.
 *   bond angle    4 degrees mean. An angle is softer than a bond in any force
 *                 field and this port places it from an idealised table.
 *   ring planarity 0.10 A worst atom out of plane, for an aromatic ring. This
 *                 one is tight on purpose: a puckered benzene is the most
 *                 visible thing a wrong conformer produces and nothing else
 *                 downstream would object to it.
 *   chirality     EXACT. Not a tolerance - a centre is one hand or the other,
 *                 and getting it wrong means folding a different substance.
 */

import { readFileSync } from "node:fs";
import { parseSmiles } from "../src/chem/smiles.js";
import { smallestRings } from "../src/chem/rings.js";
import { chiralCentres } from "../src/chem/stereo.js";
import { signedVolume } from "../src/chem/conformer.js";
import { smilesComponent } from "../src/chem/component.js";

/**
 * 🔴 THE ANGLE BAR IS 9 DEGREES AND THAT IS NOT A CLIMBDOWN, IT IS THE RIGHT
 * QUESTION ASKED PROPERLY. The corpus sits at 7.7 degrees mean and more effort
 * does not move it - 400 steps and 4 attempts give 7.68, and 1500 steps with 8
 * attempts give 7.39 for three times the work - because it is not a
 * convergence failure. It is the gap between an idealised angle table and
 * MMFF's fitted one, and closing it means shipping a force field, which is
 * exactly the "only enough for what the models need" line.
 *
 * What says 7.7 degrees is enough is `distanceRms` below: the whole-molecule
 * pairwise distance deviation, which is the SAME quantity docs/AF3.md reports
 * for this port's existing idealised conformers against the reference's CCD
 * geometry - 0.65 A rms, worst 3.77 A on a lysine - and which `test:batch`
 * REPORTS as a floor rather than failing on. A ligand conformer inside that
 * band is as good as the amino-acid conformers this repository has been
 * folding with all along.
 */
const BARS = {
  // 🔴 TIGHTENED TO WHAT IS ACHIEVED, which is the only way a gate catches a
  // regression rather than recording an ambition. Today: mean bond 0.019,
  // mean angle 3.28 deg, mean local shape 0.151 A, worst planarity 0.002 A,
  // and the furthest single molecules are porphine at 0.054 A and acetate at
  // 6.7 deg. Every bar below is roughly a third above its measurement.
  //
  // They were 0.035 / 6.0 / 0.25 and 0.11 / 17.0 / 0.80 before four fixes:
  // the collapsed embedding axis, the flattened sulfoxide, the constraint
  // projection and the adaptive attempt count. Loosening one of these later
  // is a decision somebody has to write down.
  meanBond: 0.028, meanAngle: 3.3, meanShape: 0.19,
  // ...and per molecule, wide enough for the cases whose remaining error is
  // chemistry deliberately not implemented, narrow enough to catch a
  // regression.
  bond: 0.080, angle: 8.0, shape: 0.70,
  // These two are not distributions and are not averaged.
  planarity: 0.02,
};

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
const verbose = process.argv.includes("--verbose");

const distance = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

/** How many bonds apart every pair is; 1e6 across a fragment boundary. */
function hopMatrix(graph) {
  const n = graph.atoms.length;
  const lists = graph.atoms.map(() => []);
  for (const bond of graph.bonds) {
    lists[bond.from].push(bond.to);
    lists[bond.to].push(bond.from);
  }
  return graph.atoms.map((_, start) => {
    const hops = new Array(n).fill(1e6);
    hops[start] = 0;
    const queue = [start];
    for (let head = 0; head < queue.length; head += 1) {
      for (const other of lists[queue[head]]) {
        if (hops[other] <= hops[queue[head]] + 1) continue;
        hops[other] = hops[queue[head]] + 1;
        queue.push(other);
      }
    }
    return hops;
  });
}

function angleAt(p, centre, q) {
  const u = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
  const v = [q[0] - centre[0], q[1] - centre[1], q[2] - centre[2]];
  const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const cosine = dot / (Math.hypot(...u) * Math.hypot(...v));
  return (Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI;
}

/** The worst distance of any ring atom from the best-fit plane through them. */
function outOfPlane(points) {
  const centre = [0, 1, 2].map((axis) =>
    points.reduce((total, point) => total + point[axis], 0) / points.length);
  // The plane normal is the smallest-eigenvalue direction of the covariance,
  // found here by the cross product of the two largest spreads - enough for a
  // ring, which is never close to a line.
  let best = null;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      const u = [0, 1, 2].map((axis) => points[i][axis] - centre[axis]);
      const v = [0, 1, 2].map((axis) => points[j][axis] - centre[axis]);
      const normal = [u[1] * v[2] - u[2] * v[1],
                      u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0]];
      const length = Math.hypot(...normal);
      if (length < 1e-6) continue;
      if (best === null || length > best.length) {
        best = { normal: normal.map((x) => x / length), length };
      }
    }
  }
  if (best === null) return 0;
  return Math.max(...points.map((point) => Math.abs(
    [0, 1, 2].reduce((total, axis) =>
      total + (point[axis] - centre[axis]) * best.normal[axis], 0))));
}

const failures = [];
let molecules = 0;
const bondErrors = [];
const angleErrors = [];
let chiralTotal = 0;
let chiralRight = 0;
let worstPlanarity = 0;
const distanceRms = [];
const outliers = [];
let worstPlanarityName = "";

for (const record of dump.records) {
  if (record.error !== undefined || record.conformer === undefined) continue;
  let graph;
  let component;
  try {
    graph = parseSmiles(record.smiles);
    component = await smilesComponent(record.smiles);
  } catch (error) {
    failures.push(`${record.name}: ${error.message}`);
    continue;
  }
  molecules += 1;
  const ours = component.atoms.map((atom) => [atom.x, atom.y, atom.z]);
  const theirs = record.conformer;

  const perBond = [];
  for (const bond of graph.bonds) {
    perBond.push(Math.abs(distance(ours[bond.from], ours[bond.to])
      - distance(theirs[bond.from], theirs[bond.to])));
  }
  const meanBond = perBond.reduce((a, b) => a + b, 0) / Math.max(perBond.length, 1);
  bondErrors.push(meanBond);

  const perAngle = [];
  const neighbours = graph.atoms.map(() => []);
  for (const bond of graph.bonds) {
    neighbours[bond.from].push(bond.to);
    neighbours[bond.to].push(bond.from);
  }
  for (let centre = 0; centre < graph.atoms.length; centre += 1) {
    const list = neighbours[centre];
    for (let a = 0; a < list.length; a += 1) {
      for (let b = a + 1; b < list.length; b += 1) {
        perAngle.push(Math.abs(angleAt(ours[list[a]], ours[centre], ours[list[b]])
          - angleAt(theirs[list[a]], theirs[centre], theirs[list[b]])));
      }
    }
  }
  const meanAngle = perAngle.length === 0 ? 0
    : perAngle.reduce((a, b) => a + b, 0) / perAngle.length;
  if (perAngle.length > 0) angleErrors.push(meanAngle);

  // 🔴 CHIRALITY AGAINST OUR OWN CONFORMER, NOT AGAINST RDKit's - AND THIS ARM
  // CANNOT SEE AN INVERTED CONVENTION, WHICH IS WORTH SAYING OUT LOUD. It asks
  // whether the molecule this port BUILT has the hand this port ASKED for, and
  // both come from `chiralCentres`, so flipping the convention flips the
  // target and the test together: verified by setting `ANTICLOCKWISE` to -1,
  // where this still reports 29/29 and every molecule is the wrong
  // enantiomer. What catches that is `check-stereo-vs-rdkit.mjs`, which reads
  // RDKit's geometry rather than ours and goes to 0/29 under the same edit.
  // Two gates, and only one of them is about the convention.
  const point = new Float64Array(3 * ours.length);
  ours.forEach((position, index) => {
    point[index * 3] = position[0];
    point[index * 3 + 1] = position[1];
    point[index * 3 + 2] = position[2];
  });
  for (const centre of chiralCentres(graph)) {
    chiralTotal += 1;
    const volume = signedVolume(point, ...centre.neighbours);
    if (Math.sign(volume) === Math.sign(centre.sign)) chiralRight += 1;
    else failures.push(`${record.name}: centre at atom ${centre.atom} came out`
      + " as the other enantiomer");
  }

  for (const ring of smallestRings(graph)) {
    if (ring.length > 6 || !ring.every((atom) => graph.atoms[atom].aromatic)) continue;
    const flat = outOfPlane(ring.map((atom) => ours[atom]));
    if (flat > worstPlanarity) { worstPlanarity = flat; worstPlanarityName = record.name; }
    if (flat > BARS.planarity) {
      failures.push(`${record.name}: an aromatic ring is ${flat.toFixed(2)} A`
        + " out of plane");
    }
  }

  // 🔴 THE NUMBER THAT IS COMPARABLE TO SOMETHING. Every pairwise distance in
  // the molecule, ours against RDKit's, as an rms - frame-free, so it says
  // nothing about where the molecule was placed and everything about its
  // shape. docs/AF3.md reports exactly this for the amino-acid conformers this
  // port already folds with: 0.65 A against the reference's CCD geometry.
  //
  // 🔴 AND IT IS MEASURED OVER LOCAL PAIRS WITHIN ONE FRAGMENT, BECAUSE THE
  // OTHER TWO KINDS OF PAIR DIFFER FOR REASONS THAT ARE NOT ERRORS. Two
  // separate ions have no defined distance at all - `[Na+].[Cl-]` scored 4.15
  // A rms, which measures where two programs happened to park two unbonded
  // atoms and nothing else. And a pair across a ROTATABLE bond differs by the
  // torsion, which is a real degree of freedom: biphenyl's two rings sit at
  // whatever angle each program's conformer chose, and calling that an error
  // would be demanding one conformer out of a continuum. Four bonds is where
  // the first freely rotatable torsion can separate two atoms, so pairs up to
  // four bonds apart are the shape that every conformer of the molecule
  // shares.
  const hops = hopMatrix(graph);
  const pairs = [];
  const allPairs = [];
  for (let i = 0; i < ours.length; i += 1) {
    for (let j = i + 1; j < ours.length; j += 1) {
      const delta = distance(ours[i], ours[j]) - distance(theirs[i], theirs[j]);
      if (graph.components[i] === graph.components[j]) allPairs.push(delta);
      if (hops[i][j] <= 4) pairs.push(delta);
    }
  }
  const asRms = (list) => Math.sqrt(
    list.reduce((total, d) => total + d * d, 0) / Math.max(list.length, 1));
  const rms = asRms(pairs);
  distanceRms.push({ name: record.name, rms, whole: asRms(allPairs) });
  if (rms > BARS.shape) {
    failures.push(`${record.name}: local shape ${rms.toFixed(2)} A rms from RDKit's`);
  }

  if (meanBond > BARS.bond) {
    failures.push(`${record.name}: bonds ${meanBond.toFixed(3)} A from RDKit's`);
  }
  if (meanAngle > BARS.angle) {
    failures.push(`${record.name}: angles ${meanAngle.toFixed(1)} deg from RDKit's`);
  }
  outliers.push({ name: record.name, bond: meanBond, angle: meanAngle });
  if (verbose) {
    console.log(`  ${record.name.padEnd(22)} bond ${meanBond.toFixed(3)}`
      + ` angle ${meanAngle.toFixed(1).padStart(5)}`
      + ` err ${component.conformerError.toExponential(1)}`);
  }
}

const mean = (list) => list.reduce((a, b) => a + b, 0) / Math.max(list.length, 1);
console.log(`\n${molecules} conformers built`);
console.log(`  bond lengths   mean ${mean(bondErrors).toFixed(3)} A from RDKit`
  + `  (mean bar ${BARS.meanBond})`);
console.log(`  bond angles    mean ${mean(angleErrors).toFixed(2)} deg`
  + `  (mean bar ${BARS.meanAngle})`);
console.log(`  aromatic rings worst ${worstPlanarity.toFixed(3)} A out of plane`
  + ` (${worstPlanarityName}, bar ${BARS.planarity})`);
const sorted = distanceRms.slice().sort((a, b) => b.rms - a.rms);
console.log(`  local shape    mean ${mean(distanceRms.map((r) => r.rms)).toFixed(3)} A rms`
  + ` to 4 bonds, worst ${sorted[0]?.rms.toFixed(2)} (${sorted[0]?.name})`
  + `  (bar ${BARS.shape})`);
// Reported and NOT a bar: it includes every torsion, which is a free choice.
console.log(`  whole shape    mean ${mean(distanceRms.map((r) => r.whole)).toFixed(3)} A rms`
  + " over every pair in a fragment (reported - a torsion is not an error)");
console.log(`  chiral centres ${chiralRight}/${chiralTotal} built with the right hand`);

// 🔴 THE CORPUS AVERAGE IS THE ASSERTION AND A SINGLE MOLECULE IS NOT, for
// the reason the bar list gives: the molecules that remain furthest from MMFF
// are the ones whose chemistry this port deliberately does not model.
// Dimethyl sulfoxide wants 96.6 degrees at sulfur where this places a uniform
// pyramidal 106; a CF3 group's F-C-F opens past tetrahedral from fluorine
// repulsion; ATP's triphosphate is a chain of hypervalent centres. Modelling
// any of them properly means shipping a force field, which is past what a
// model reads as `ref_pos`. So the gate holds the DISTRIBUTION, which is what
// a conformer set is judged on, plus the two things that are exact - the
// handedness of every stereocentre and the flatness of every aromatic ring.
const meanBond = mean(outliers.map((o) => o.bond));
const meanAngle = mean(outliers.map((o) => o.angle));
const meanShape = mean(distanceRms.map((r) => r.rms));
if (meanBond > BARS.meanBond) {
  failures.push(`the corpus mean bond error is ${meanBond.toFixed(3)} A,`
    + ` past ${BARS.meanBond}`);
}
if (meanAngle > BARS.meanAngle) {
  failures.push(`the corpus mean angle error is ${meanAngle.toFixed(2)} deg,`
    + ` past ${BARS.meanAngle}`);
}
if (meanShape > BARS.meanShape) {
  failures.push(`the corpus mean local shape is ${meanShape.toFixed(3)} A rms,`
    + ` past ${BARS.meanShape}`);
}

const worstBond = outliers.slice().sort((a, b) => b.bond - a.bond)[0];
const worstAngle = outliers.slice().sort((a, b) => b.angle - a.angle)[0];
console.log(`  furthest       ${worstBond.name} ${worstBond.bond.toFixed(3)} A,`
  + ` ${worstAngle.name} ${worstAngle.angle.toFixed(1)} deg`);

for (const line of failures.slice(0, 15)) console.log(`FAIL  ${line}`);
if (failures.length > 15) console.log(`...and ${failures.length - 15} more`);
if (failures.length > 0 || chiralRight !== chiralTotal) process.exit(1);
console.log("\nevery conformer is a molecule");
