/**
 * Every AF3-lineage model folds a LIGAND, and its bonds are the right length.
 *
 *     npm run test:ligand
 *
 * 🔴 THE ARM NOTHING WAS RUNNING, AND IT BROKE TWICE IN ONE SESSION. Every fold
 * gate here folds a plain protein, so the whole atomised-token half of the
 * featuriser - four dialect conventions - was exercised by nothing. When it
 * finally was: rosettafold3 died outright on a ligand with no stereocentre
 * ("invalid allocation size 0 for atom.chiral.centers", because glycerol has
 * none and an empty buffer is refused), and `atomizedElementNames` renamed the
 * atoms in the OUTPUT PDB as well as in the model's input, giving one residue
 * six atoms called C, O, C, O, C, O.
 *
 * 🔴 AND IT ASSERTS BOND LENGTHS, BECAUSE NOTHING ELSE CAN SEE THEM. The fold's
 * RMSD is dominated by 68 residues of protein; `meanPlddt` said 79 on a
 * glycerol whose bonds were 7 A out; and tools/gpu/chain-geometry.js measures
 * the protein BACKBONE and steps over a ligand by design. A ligand that comes
 * apart is invisible to all three.
 *
 * 🔴 AND IT MEASURES BY ATOM ORDER, NOT BY NAME. rf3 renames an atomised atom
 * to its element symbol, so a checker keying on "C1" finds nothing and reports
 * the ligand missing - which is what the first version of this did, and the
 * convention working correctly read as a dropped ligand.
 */
import { execFileSync } from "node:child_process";

const SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";
// Glycerol, in the CCD's atom order: C1 O1 C2 O2 C3 O3.
const BONDS = [[0, 1, "C1-O1"], [0, 2, "C1-C2"], [2, 3, "C2-O2"],
               [2, 4, "C2-C3"], [4, 5, "C3-O3"]];
const IDEAL = 1.43;
const BOND_RMS_MAX = 0.20;   // A. af3 0.050, if2 0.055, rf3 0.069 measured.
// bundle directory -> dialect name. 🔴 THE SUFFIX IS PER BOX, NOT PER MODEL:
// this machine has openbind0 as f32 and the rest as int5, and hard-coding
// "-int5" made the gate report openbind0 as a FAILURE when it was a 404. A
// missing bundle is a skip, not a defect - it is a property of the box.
const MODELS = [
  ["model-af3-int5", "alphafold3"], ["model-openbind0-f32", "openbind0"],
  ["model-boltz2-int5", "boltz2"], ["model-protenix2-int5", "protenix2"],
  ["model-intellifold2-int5", "intellifold2"],
  ["model-rosettafold3-int5", "rosettafold3"],
];

let failures = 0;
for (const [bundle, name] of MODELS) {
  let text;
  try {
    text = execFileSync("node", [
      "tools/gpu-chrome.mjs", "tools/gpu/fold.js",
      `--model=/${bundle}/manifest.json`,
      `--sequence=${SEQUENCE}`, "--ligands=GOL",
    ], { encoding: "utf8", maxBuffer: 1 << 28, stdio: ["ignore", "pipe", "pipe"] });
  } catch (error) {
    const message = String(error.stderr ?? error.message);
    // A bundle this box does not have is a skip; anything else is a failure.
    if (/failed to load .*manifest\.json: 404/.test(message)) {
      console.log(`skip  ${name.padEnd(13)} no ${bundle} on this box`);
      continue;
    }
    console.log(`FAIL  ${name.padEnd(13)} did not fold: `
      + `${message.split("\n").filter((l) => /error|Error/.test(l)).slice(0, 1).join("")
         || error.message}`);
    failures += 1;
    continue;
  }
  const result = JSON.parse(text.slice(text.indexOf("{")));
  const pdb = result.pdb ?? result.denoisedPdb ?? "";
  const atoms = [];
  const names = [];
  for (const line of pdb.split("\n")) {
    if (!line.startsWith("HETATM") || line.slice(17, 20).trim() !== "GOL") continue;
    atoms.push([Number(line.slice(30, 38)), Number(line.slice(38, 46)),
                Number(line.slice(46, 54))]);
    names.push(line.slice(12, 16).trim());
  }
  if (atoms.length !== 6) {
    console.log(`FAIL  ${name.padEnd(13)} ${atoms.length} GOL atoms in the model, wanted 6`);
    failures += 1;
    continue;
  }
  // 🔴 THE NAMES MUST BE THE COMPONENT'S AND MUST BE UNIQUE. This is the second
  // defect above, and it is a property of the FILE rather than of the fold.
  if (new Set(names).size !== names.length) {
    console.log(`FAIL  ${name.padEnd(13)} GOL's atom names are not unique: ${names.join(",")}`);
    failures += 1;
    continue;
  }
  const gap = (a, b) => Math.hypot(atoms[a][0] - atoms[b][0],
                                   atoms[a][1] - atoms[b][1], atoms[a][2] - atoms[b][2]);
  const lengths = BONDS.map(([a, b, label]) => [label, gap(a, b)]);
  const rms = Math.sqrt(lengths.reduce((sum, [, v]) => sum + (v - IDEAL) ** 2, 0)
                        / lengths.length);
  const ok = rms <= BOND_RMS_MAX;
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name.padEnd(13)}`
    + ` pLDDT ${result.meanPlddt.toFixed(2).padStart(6)}`
    + `  bond rms ${rms.toFixed(3)} A  `
    + lengths.map(([l, v]) => `${l} ${v.toFixed(2)}`).join("  "));
}
console.log(failures === 0
  ? "every model folds a ligand, with its bonds intact and its atoms named"
  : `🔴 ${failures} models failed the ligand path`);
if (failures > 0) process.exitCode = 1;
