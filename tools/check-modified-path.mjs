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
 * 🔴 AND IT IS NOT THIS PORT'S BUG, WHICH IS WHY boltz2 IS AN EXPECTED FAILURE
 * HERE RATHER THAN A RED GATE - BUT IT IS SOMEBODY'S, AND THE THIRD REFERENCE
 * SAYS WHOSE. Three folds of one target:
 *
 *     genuine Boltz-2 2.2.1, single sequence   0.986 / 0.996 / 1.011
 *     af3-any-model's boltz2                   2.705   (bond rms 2.99 A)
 *     this port                                1.813
 *
 * **The real Boltz-2 places a phosphoserine correctly** - `pip install boltz`,
 * no MSA, three diffusion samples, bond rms 0.043-0.076 A - so this is a defect
 * in af3-any-model's boltz2 PORT that LocalFold inherits, and not something
 * Boltz-2 cannot do. Its AlphaFold 3 on the identical target through the
 * identical harness is 0.988 and 0.143 A, so the harness is not the cause
 * either. `tools/oracle/probe_af3_ptm_bonds.py` is the af3-any-model side and
 * `tools/oracle/score_modified_cif.py` scores any predicted mmCIF against the
 * RCSB dictionary - a THIRD scorer, written separately from the other two, and
 * it agrees with them on the models they share. Three things are already
 * excluded on this side: the bond matrix and the bond-order plane are
 * byte-identical across models (9 pairs, 9 entries), `ref_pos` is identical
 * (10 slots, one space, 6.924 A across the residue), and the `_1` weight-name
 * change is a proven no-op for boltz2 (4 of 4 tensors byte-identical, where
 * only AlphaFold 3's differ).
 *
 * 🔴 SO THE EXCEPTION ASSERTS IT STAYS BROKEN, and says what retires it. A
 * silent improvement is as much a surprise as a silent regression: if boltz2
 * comes back inside the band, upstream fixed it or this port changed, and
 * either way the entry and its evidence should go.
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
  { name: "boltz2", bundle: "model-boltz2-int5",
    expected: [1.5, 2.4],
    why: "af3-any-model's boltz2 is 2.705 here and GENUINE Boltz-2 2.2.1 is "
       + "0.986-1.011, so it is that port's defect and this one inherits it" },
];

const FLOW = ["--mode=flow", "--steps=16"];
let failed = 0, skipped = 0;
for (const model of MODELS) {
  if (!existsSync(model.bundle)) {
    console.log(`skip  ${model.name.padEnd(13)} ${model.bundle} is not on this box`);
    skipped += 1;
    continue;
  }
  const run = spawnSync("node", ["tools/gpu-chrome.mjs", "tools/gpu/probe-modified.js",
    `--model=/${model.bundle}/manifest.json`, "--code=SEP", "--at=3",
    ...(model.arm ?? FLOW)], { encoding: "utf8", maxBuffer: 1 << 28 });
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
