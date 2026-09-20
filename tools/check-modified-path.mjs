/**
 * Does every model fold a MODIFIED RESIDUE, and does it hold together?
 *
 *     npm run test:modified
 *
 * 🔴 NOTHING WAS RUNNING THIS ARM AND ONE MODEL IS BROKEN IN IT. `probe-
 * modified.js` has computed `modifiedBondRatio` since it was written and
 * nothing ever asserted on it, which is this repository's oldest shape of
 * mistake - a number a tool prints is not a gate until something fails on it.
 * `test:ligand` folds a GLYCEROL, which is a separate chain and not an atomised
 * residue inside a polymer; `test:batch` compares a SEP target's batch FIELDS
 * and never folds it. So the whole modified-residue path had a featuriser gate
 * and no geometry gate.
 *
 * Found on the first run: **boltz2 inflates a phosphoserine by 1.8x** - N-CA
 * 2.007 against 1.469, CA-C 3.485 against 1.506 - while every other model is
 * within 6% and its own control residues are at 0.995.
 *
 * 🔴 IT WAS AN EXPECTED FAILURE AND IT IS FIXED, SO THE ENTRY IS GONE. What it
 * was: boltz2 keeps a modified residue in ONE token where every other family
 * atomises it into one token per atom, and this port atomised it for everyone.
 * `modifiedAsOneToken` in dialect.js is the convention, set for boltz2 alone.
 * Measured on SEP at position 3, boltz2's own bonds: **1.813 -> 0.938**,
 * against genuine Boltz-2 2.2.1 at 0.986-1.011.
 *
 * 🔴 AND THE REFERENCE HAD A SECOND FAULT THAT DOES NOT APPLY HERE.
 * af3-any-model was 2.705 because it also dropped O3P - a SIDECHAIN atom of a
 * phosphoserine - under a `drop_atoms` guard whose predicate a later convention
 * falsified. This port has no by-name drop list, so it had one of the two
 * faults and read 1.813 where the reference read 2.705.
 *
 * 🔴 AND IT MUST BE FOLDED AT A CONVERGED SETTING OR IT MEASURES NOTHING. Run
 * at `probe-modified.js`'s default eight steps in DIFFUSION mode, every model
 * reads a ratio between 11 and 21 and so does its CONTROL - the fold has not
 * condensed at all. See docs/AF3.md: diffusion at eight calls is 19-37 A on a
 * ligand. flow 16 is converged and is what this uses.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

// The band is the chemistry, not a fit to today's numbers: a residue whose own
// bonds are 30% out is wrong however it got there. Every passing model sits at
// 0.94-1.16 and rosettafold3 at 0.73, which is inside and worth watching.
const BAND = [0.70, 1.30];

const MODELS = [
  { name: "alphafold3", bundle: "model-af3-int5" },
  { name: "protenix2", bundle: "model-protenix2-int5" },
  { name: "intellifold2", bundle: "model-intellifold2-int5" },
  { name: "openbind0", bundle: "model-openbind0-f32" },
  { name: "opendde", bundle: "model-opendde-int5" },
  // 🔴 rosettafold3 HAS NO FLOW SAMPLER - `noFlowSampler` - so it is the one
  // model here that must be folded with diffusion, and at a count that has
  // converged rather than the page's 25.
  { name: "rosettafold3", bundle: "model-rosettafold3-int5",
    arm: ["--mode=diffusion", "--steps=200"] },
  { name: "boltz2", bundle: "model-boltz2-int5" },
  // 🔴 ESMFold2 DRIVES ITS OWN TOOL, AND ITS ABSENCE HERE IS WHY A USER FOUND
  // THE BUG. `probe-modified.js` opens an AF3 store and calls `foldBatch`, so
  // it cannot reach this model at all - and the page's own guard has always
  // OFFERED modified residues for ESMFold2. It was never given them:
  // `foldWithEsmfold2` did not take `modifications` and its entities literal
  // did not name one, so a SEP@3 folded a plain serine at a confident 0.59 and
  // nothing anywhere said so. An entry carries its own `tool` and `args` for
  // this, the way check-template-path.mjs's do.
  //
  // 🔴 AND IT MUST BE FOLDED AT 200, NOT THE PAGE'S DEFAULT. The vendor breaks
  // below 15 steps - at 11 its own CONTROL is 2.155 - so a low count would fail
  // this gate for the sampler rather than for the residue. See
  // docs/ESMFOLD2_PTM.md.
  { name: "esmfold2", bundle: "model-esmfold2-int5",
    tool: "tools/gpu/fold-esmfold2.js",
    args: ["--bundle=/model-esmfold2-int5", "--modify=SEP@3",
           "--sampler=diffusion-200",
           "--sequence=GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK"] },
];

const FLOW = ["--mode=flow", "--steps=16"];
let failed = 0, skipped = 0;
for (const model of MODELS) {
  if (!existsSync(model.bundle)) {
    console.log(`skip  ${model.name.padEnd(13)} ${model.bundle} is not on this box`);
    skipped += 1;
    continue;
  }
  // An entry may bring its own tool and its own arguments; the default is the
  // AF3-lineage probe, which is what every other row wants.
  const run = spawnSync("node", ["tools/gpu-chrome.mjs",
    model.tool ?? "tools/gpu/probe-modified.js",
    ...(model.args ?? [`--model=/${model.bundle}/manifest.json`, "--code=SEP",
                       "--at=3", ...(model.arm ?? FLOW)])],
    { encoding: "utf8", maxBuffer: 1 << 28 });
  const text = (run.stdout ?? "").split("\n").filter((l) => !l.startsWith("[gpu-chrome]")).join("\n");
  const at = text.indexOf("{");
  if (at < 0) {
    console.log(`FAIL  ${model.name.padEnd(13)} no result\n${(run.stderr ?? "").slice(-400)}`);
    failed += 1;
    continue;
  }
  const got = JSON.parse(text.slice(at));
  const ratio = got.modifiedBondRatio, control = got.controlBondRatio;
  // 🔴 THE CONTROL IS PART OF THE ASSERTION. A fold that has not converged puts
  // BOTH ratios far from 1, and reading the modified one alone would call that
  // a modified-residue defect - which is exactly what eight diffusion steps
  // did when this was first run.
  if (!(control > 0.85 && control < 1.15)) {
    console.log(`FAIL  ${model.name.padEnd(13)} control residues are ${control} - `
      + "the fold has not converged, so the modified one says nothing");
    failed += 1;
    continue;
  }
  const band = model.expected ?? BAND;
  const ok = ratio >= band[0] && ratio <= band[1];
  const tag = model.expected ? (ok ? "xfail" : "FIXED") : (ok ? "ok" : "FAIL");
  console.log(`${tag.padEnd(5)} ${model.name.padEnd(13)} modified ${String(ratio).padEnd(7)}`
    + ` control ${String(control).padEnd(7)} band ${band[0]}-${band[1]}`
    + (model.expected ? `   (expected failure: ${model.why})` : ""));
  if (!ok) failed += 1;
}
if (failed > 0) {
  console.log(`\n🔴 ${failed} model(s) outside their band. A FIXED row means an expected `
    + "failure came good: delete the entry and its evidence rather than widening it.");
  process.exit(1);
}
console.log(`\nevery model folds a modified residue within band`
  + `${skipped ? `, ${skipped} skipped for want of a bundle` : ""}`);
