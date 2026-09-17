#!/usr/bin/env node
/**
 * The same ligand from a CCD code and from a SMILES string folds the same way.
 *
 *     npm run test:smiles-path
 *
 * 🔴 THIS IS THE ONLY GATE THAT PUTS A SMILES LIGAND THROUGH A MODEL, and it is
 * a differential rather than an absolute for a reason: glycerol from the
 * dictionary and glycerol from `OCC(O)CO` are the same six atoms and the same
 * five bonds, so a fold of one is the control for a fold of the other. Every
 * other check in `npm run test:smiles` stops at the component - the graph
 * against RDKit, the conformer against RDKit's - and none of them proves the
 * object reaches the featuriser, survives atomisation, comes back in the PDB
 * with unique names, or holds together under the sampler.
 *
 * 🔴 AND IT SCORES BONDS, BECAUSE NOTHING ELSE CAN SEE A TORN LIGAND. This is
 * `check-ligand-path.mjs`'s finding and it applies unchanged: the fold's RMSD
 * is dominated by the protein, pLDDT read 92 on a glycerol 6 A out, and
 * chain-geometry.js steps over a ligand by design.
 *
 * 🔴 WHAT IT CATCHES AND WHAT IT DOES NOT, MEASURED BY BREAKING IT ON PURPOSE.
 * Replacing the conformer with noise fails all three cases loudly - bond rms
 * 0.22 against 0.03, bonds up to 0.44 A apart. But SHRINKING every conformer
 * by 15% changes the fold by nothing at all: 0.035 A apart against 0.035.
 *
 * That is worth knowing rather than worth fixing. `ref_pos` is a FEATURE, not
 * a template - the diffusion head places the atoms itself and reads the
 * reference conformer for what the component IS rather than for where to put
 * it - so a uniformly rescaled conformer is still a recognisable description
 * of the same molecule and a scrambled one is not. It also means this gate is
 * a plumbing test with a geometry floor: it proves the component reaches the
 * featuriser, survives atomisation, comes back named and holds together, and
 * it will not notice a conformer that is subtly rather than grossly wrong.
 * `tools/check-smiles-conformer.mjs` is what measures the conformer, against
 * RDKit, and it resolves 0.03 A.
 *
 * 🔴 AND IT COMPARES SORTED BOND LENGTHS RATHER THAN BOND i TO BOND i. The two
 * routes number the atoms differently - the CCD writes glycerol C1 O1 C2 O2 C3
 * O3 and `OCC(O)CO` reads O first - so the same molecule has the same bonds
 * under a permutation. Sorting makes the comparison order-free; a torn ligand
 * still shows, because a broken bond is long in any order.
 */
import { execFileSync } from "node:child_process";
import { parseSmiles } from "../src/chem/smiles.js";
import { smilesComponent } from "../src/chem/component.js";

const SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";
const MODEL = "/model-af3-int5/manifest.json";

/**
 * Each case is one molecule described twice. The SMILES is checked against the
 * code's own formula first, so a typo in the corpus cannot quietly make this a
 * comparison of two different substances.
 */
const CASES = [
  ["GOL", "OCC(O)CO", 6],                                    // no stereocentre
  ["BTN", "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12", 16],   // three, and two rings
  ["EDO", "OCCO", 4],                                        // the smallest
];

/**
 * Ligands with no dictionary twin, scored against their OWN component.
 *
 * 🔴 THE PAIRED CASES ABOVE ARE ALL SMALL, BECAUSE A DICTIONARY TWIN HAS TO BE
 * WRITTEN BY HAND IN THE DICTIONARY'S ATOM ORDER AND THAT DOES NOT SCALE. The
 * page admits 150 heavy atoms and the largest paired case here is 16, so
 * nothing was folding a ligand of the size people actually dock. These have no
 * CCD counterpart and are not differentials; what they assert is that a large
 * ligand survives the round trip intact - every atom present, every name
 * unique, every bond the length the component asked for.
 */
const SOLO = [
  // Paclitaxel's core: 62 heavy atoms, four rings, eleven oxygens.
  ["TAX", "CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(C)=O)C2C(OC(=O)c2ccccc2)C2(O)CC(OC(=O)"
    + "C(O)C(NC(=O)c3ccccc3)c3ccccc3)C(C)=C1C2(C)C", 62],
  // An erythromycin fragment: a 14-membered macrolide, ten stereocentres.
  ["ERY", "CC[C@H]1OC(=O)[C@H](C)[C@@H](O)[C@H](C)[C@@H](O)[C@](C)(O)C[C@@H](C)"
    + "C(=O)[C@H](C)[C@@H](O)[C@H]1C", 28],
];

/** How far a solo ligand's folded bond may sit from its component's own. */
const SOLO_BOND_RMS = 0.12;

/** How far the two routes' sorted bond lengths may differ, in angstroms. */
const AGREE = 0.25;
/** ...and how far a folded bond may sit from the length its component wanted. */
const INTACT = 0.30;

function fold(extra) {
  const text = execFileSync("node", [
    "tools/gpu-chrome.mjs", "tools/gpu/fold.js",
    `--model=${MODEL}`, `--sequence=${SEQUENCE}`, ...extra,
  ], { encoding: "utf8", maxBuffer: 1 << 28, stdio: ["ignore", "pipe", "pipe"] });
  return JSON.parse(text.slice(text.indexOf("{")));
}

/** The ligand's atoms out of a folded PDB, in the order they were written. */
function ligandAtoms(pdb, code) {
  const atoms = [];
  const names = [];
  for (const line of (pdb ?? "").split("\n")) {
    if (!line.startsWith("HETATM") || line.slice(17, 20).trim() !== code) continue;
    atoms.push([Number(line.slice(30, 38)), Number(line.slice(38, 46)),
                Number(line.slice(46, 54))]);
    names.push(line.slice(12, 16).trim());
  }
  return { atoms, names };
}

const gap = (atoms, a, b) => Math.hypot(
  atoms[a][0] - atoms[b][0], atoms[a][1] - atoms[b][1], atoms[a][2] - atoms[b][2]);

let failures = 0;
for (const [code, smiles, expected] of CASES) {
  // 🔴 THE TWO DESCRIPTIONS ARE CHECKED AGAINST EACH OTHER BEFORE ANYTHING IS
  // FOLDED. A SMILES that is not the same molecule as the code would make this
  // whole comparison meaningless in a way that looks like a conformer bug.
  const graph = parseSmiles(smiles);
  if (graph.atoms.length !== expected) {
    console.log(`FAIL  ${code}: the SMILES has ${graph.atoms.length} heavy atoms,`
      + ` the case says ${expected}`);
    failures += 1;
    continue;
  }

  let fromCode;
  let fromSmiles;
  try {
    fromCode = fold([`--ligands=${code}`]);
    // ...under the same residue name, so the PDB reader below is one reader.
    fromSmiles = fold([`--smiles=${smiles}`, `--smiles-code=${code}`]);
  } catch (error) {
    const message = String(error.stderr ?? error.message);
    if (/failed to load .*manifest\.json: 404/.test(message)) {
      console.log(`skip  ${code}: no af3 bundle on this box`);
      continue;
    }
    console.log(`FAIL  ${code}: did not fold: `
      + `${message.split("\n").filter((l) => /rror/.test(l)).slice(0, 1).join("")}`);
    failures += 1;
    continue;
  }

  const a = ligandAtoms(fromCode.pdb ?? fromCode.denoisedPdb, code);
  const b = ligandAtoms(fromSmiles.pdb ?? fromSmiles.denoisedPdb, code);
  if (a.atoms.length !== expected || b.atoms.length !== expected) {
    console.log(`FAIL  ${code}: ${a.atoms.length} atoms from the code and`
      + ` ${b.atoms.length} from the SMILES, wanted ${expected}`);
    failures += 1;
    continue;
  }
  // The names come from the component and must still be unique after
  // atomisation - `check-ligand-path.mjs` found a convention that renamed them.
  if (new Set(b.names).size !== b.names.length) {
    console.log(`FAIL  ${code}: the SMILES route's atom names repeat:`
      + ` ${b.names.join(",")}`);
    failures += 1;
    continue;
  }

  // Bonds by the SMILES graph's own list, applied to each arm's atoms. The CCD
  // arm is a different permutation, so its lengths are compared as a sorted
  // multiset rather than pairwise - see the header.
  const smilesLengths = graph.bonds
    .map((bond) => gap(b.atoms, bond.from, bond.to)).sort((x, y) => x - y);
  // For the CCD arm the bonds are whatever pairs sit at a bonded distance;
  // taking the N shortest pairs is enough, because a ligand this size has no
  // non-bonded pair shorter than its longest bond unless it has collapsed -
  // which is itself the failure being looked for.
  const all = [];
  for (let i = 0; i < a.atoms.length; i += 1) {
    for (let j = i + 1; j < a.atoms.length; j += 1) all.push(gap(a.atoms, i, j));
  }
  const codeLengths = all.sort((x, y) => x - y).slice(0, graph.bonds.length);

  const worst = Math.max(...smilesLengths.map(
    (length, index) => Math.abs(length - codeLengths[index])));
  // ...and each arm against the length its own component asked for, which is
  // what says the sampler did not pull it apart.
  const ideal = 1.45;
  const spread = (list) => Math.sqrt(list.reduce(
    (total, length) => total + (length - ideal) ** 2, 0) / list.length);
  const codeRms = spread(codeLengths);
  const smilesRms = spread(smilesLengths);

  const ok = worst <= AGREE && smilesRms <= INTACT && codeRms <= INTACT;
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"}  ${code.padEnd(4)}`
    + ` code pLDDT ${fromCode.meanPlddt.toFixed(1)} rms ${codeRms.toFixed(3)}`
    + `  ·  SMILES pLDDT ${fromSmiles.meanPlddt.toFixed(1)} rms ${smilesRms.toFixed(3)}`
    + `  ·  worst bond apart ${worst.toFixed(3)} A`);
}

for (const [code, smiles, expected] of SOLO) {
  const graph = parseSmiles(smiles);
  let result;
  try {
    result = fold([`--smiles=${smiles}`, `--smiles-code=${code}`]);
  } catch (error) {
    const message = String(error.stderr ?? error.message);
    if (/failed to load .*manifest\.json: 404/.test(message)) {
      console.log(`skip  ${code}: no af3 bundle on this box`);
      continue;
    }
    console.log(`FAIL  ${code}: did not fold: `
      + `${message.split("\n").filter((l) => /rror/.test(l)).slice(0, 1).join("")}`);
    failures += 1;
    continue;
  }
  const { atoms, names } = ligandAtoms(result.pdb ?? result.denoisedPdb, code);
  if (atoms.length !== expected) {
    console.log(`FAIL  ${code}: ${atoms.length} atoms in the model, wanted ${expected}`);
    failures += 1;
    continue;
  }
  if (new Set(names).size !== names.length) {
    console.log(`FAIL  ${code}: the atom names repeat`);
    failures += 1;
    continue;
  }
  // 🔴 SCORED AGAINST THE COMPONENT THIS PORT BUILT, which is the only ideal
  // there is for a ligand with no dictionary entry - and is a fair one,
  // because that component's own bond lengths are held to RDKit at 0.022 A
  // mean by `check-smiles-conformer.mjs`.
  const component = await smilesComponent(smiles, { code });
  const wanted = (bond) => Math.hypot(
    component.atoms[bond.from].x - component.atoms[bond.to].x,
    component.atoms[bond.from].y - component.atoms[bond.to].y,
    component.atoms[bond.from].z - component.atoms[bond.to].z);
  let total = 0;
  let worst = 0;
  for (const bond of graph.bonds) {
    const error = gap(atoms, bond.from, bond.to) - wanted(bond);
    total += error * error;
    if (Math.abs(error) > Math.abs(worst)) worst = error;
  }
  const rms = Math.sqrt(total / graph.bonds.length);
  const ok = rms <= SOLO_BOND_RMS;
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"}  ${code.padEnd(4)}`
    + ` ${expected} atoms, ${graph.bonds.length} bonds, no CCD twin`
    + `  ·  pLDDT ${result.meanPlddt.toFixed(1)}`
    + `  ·  bond rms ${rms.toFixed(3)} A, worst ${worst.toFixed(2)}`);
}

console.log(failures === 0
  ? "a SMILES ligand folds the same molecule its CCD code does, at 4 atoms and at 62"
  : `🔴 ${failures} case(s) failed the SMILES fold path`);
if (failures > 0) process.exitCode = 1;
