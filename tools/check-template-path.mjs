/**
 * Every AF3-lineage model's TEMPLATE actually moves its fold.
 *
 *     npm run test:template
 *
 * 🔴 THE ARM NOTHING WAS RUNNING, AND ONE MODEL'S TEMPLATE WAS INERT. There are
 * two oracle checkers for the template stage - `check-af3-template.js` and
 * `check-fused-template-features.js` - and BOTH drive the module with
 * `templates: 1`. A fold pads the slot count to FOUR. So a dialect that
 * averages over the wrong denominator is exactly right in every module check
 * and a quarter strength in every fold, which is what rosettafold3 was: with a
 * perfect self-template of 5CAJ chain A it moved 17.949 A to 17.771, where
 * AlphaFold 3 takes the identical input to 0.281 and IntelliFold-2 to 0.254.
 * Fixed by `templateFeatureMeanOnePass`; it now reads 0.137, the best of the
 * seven. See docs/AF3.md.
 *
 * 🔴 AND IT IS A SELF-TEMPLATE ON A TARGET THAT NEEDS ONE. 6MRR folds to 0.5 A
 * from its sequence alone, so a working template embedder and a dead one land
 * in the same place there - CLAUDE.md's own warning, and the reason this gate
 * uses 5CAJ chain A (255 residues, natural), which is 17 to 30 A without a
 * template for every model in the panel. The bar is therefore not "better" but
 * "lands on the crystal": a template this strong leaves nothing to predict.
 *
 * 🔴 AND THE NO-TEMPLATE ARM IS RUN TOO, because "0.14 A with a template" is
 * only evidence if the same model is 17 A without one. Without that arm a
 * checkpoint that had simply memorised the target would pass.
 */
import { execFileSync } from "node:child_process";

const TEMPLATE = "/tools/fixtures/5caj-crystal.pdb:A";
// A self-template of the whole chain leaves nothing to guess. Measured on this
// box: rf3 0.137, if2 0.254, af3 0.281, and 17-30 A for all of them without.
const WITH_MAX = 1.5;
const WITHOUT_MIN = 5.0;
// 🔴 THE SUFFIX IS PER BOX, NOT PER MODEL - see check-ligand-path.mjs. A
// missing bundle is a skip.
const MODELS = [
  ["model-af3-int5", "alphafold3"], ["model-openbind0-f32", "openbind0"],
  ["model-boltz2-int5", "boltz2"], ["model-protenix2-int5", "protenix2"],
  ["model-opendde-int5", "opendde"],
  ["model-intellifold2-int5", "intellifold2"],
  ["model-rosettafold3-int5", "rosettafold3"],
];

/** One fold of 5CAJ chain A, with or without the self-template. */
function fold(bundle, template) {
  const args = ["tools/gpu-chrome.mjs", "tools/gpu/fold-opendde.js",
                "--target=5caj", "--chain=A", "--steps=25", "--mode=diffusion",
                `--model=/${bundle}/manifest.json`];
  // A fold with no template is 17-30 A out and its backbone is not always a
  // chain; that arm is the CONTROL and not the thing under test.
  if (template) args.push(`--template=${TEMPLATE}`);
  else args.push("--allow-broken-geometry");
  const text = execFileSync("node", args,
    { encoding: "utf8", maxBuffer: 1 << 28, stdio: ["ignore", "pipe", "pipe"] });
  return JSON.parse(text.slice(text.indexOf("{")));
}

let failures = 0;
for (const [bundle, name] of MODELS) {
  let templated;
  let bare;
  try {
    templated = fold(bundle, true);
    bare = fold(bundle, false);
  } catch (error) {
    const message = String(error.stderr ?? error.message);
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
  // `scored` is null when the folded sequence is not the crystal's, which
  // cannot happen here - the tool reads both from the same file - so a missing
  // one is a broken harness rather than a skip.
  const withRmsd = templated.scored?.rmsd;
  const withoutRmsd = bare.scored?.rmsd;
  if (!Number.isFinite(withRmsd) || !Number.isFinite(withoutRmsd)) {
    // 🔴 A MISSING NUMBER IS A FAILURE, NOT A PASS. `NaN <= x` is false and
    // `NaN >= x` is false, so a comparison alone would let a renamed field
    // through in both directions - this file's oldest lesson.
    console.log(`FAIL  ${name.padEnd(13)} no RMSD in the fold's output`);
    failures += 1;
    continue;
  }
  const ok = withRmsd <= WITH_MAX && withoutRmsd >= WITHOUT_MIN;
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name.padEnd(13)}`
    + ` with a template ${withRmsd.toFixed(3).padStart(7)} A`
    + `   without ${withoutRmsd.toFixed(3).padStart(7)} A`
    + `   pLDDT ${Number(templated.meanPlddt).toFixed(2).padStart(6)}`
    + (withoutRmsd < WITHOUT_MIN
      ? "   <- the control folded it anyway; this target proves nothing here"
      : ""));
}
console.log(failures === 0
  ? "every model's template lands its fold on the crystal"
  : `🔴 ${failures} models failed the template path`);
if (failures > 0) process.exitCode = 1;
