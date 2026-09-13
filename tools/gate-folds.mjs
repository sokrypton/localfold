/**
 * The folds every whole-model gate runs, and how a gate reads one of them.
 *
 *     import { runFolds } from "./gate-folds.mjs";
 *
 * 🔴 ONE COPY, BECAUSE TWO COPIES WERE THE BUG. `check-stock-flags.mjs` and
 * `check-portable-limits.mjs` each carried the same six-fold list and the same
 * verdict code, typed out twice. Both named `--dump=/oracle-dumps/...` on the
 * two newest models, and `oracle-dumps/` is gitignored whole - so on any
 * machine but the one that wrote those dumps both arms 404ed before reaching a
 * device, and each gate reported two models that "do not fold". The stock gate
 * was fixed; the portable one kept the bug, and on an M2 it said "2 of 6 models
 * do not fold at the portable limit ceiling" the same afternoon. Fixing one copy
 * of a duplicate is not fixing it.
 *
 * What stays per gate is what is genuinely different: the environment variable
 * that states the question, the error patterns that mean THAT question failed,
 * and a precondition - the portable gate must see its ceiling reach the page.
 */
import { spawn } from "node:child_process";

export const FOLDS = [
  ["AF2", ["tools/gpu/fold-af2.js", "--repeat=2"]],
  ["AF3", ["tools/gpu/fold.js", "--model=/model-af3-int5/manifest.json"]],
  ["OpenDDE", ["tools/gpu/fold-opendde.js", "--target=6mrr", "--steps=16"]],
  ["ESMFold2", ["tools/gpu/fold-esmfold2.js", "--bundle=/model-esmfold2-int5"]],
  // 🔴 boltz2 is the one to watch: its token transformer amplifies its input by
  // ~2.2e4 and it already refuses f16 on the DENOISE gate, so a device that
  // resolves precision or limits differently is where it would come apart.
  // 🔴 AND NO `--dump=`. A batch dump is an ORACLE input, and these gates are
  // about a capability or a limit - measured on an M2 with no dumps, boltz2
  // 96.46015389206518 and protenix2 84.73666947394713, identical to their
  // flagged runs to every digit. Same fault docs/PARITY.md records for the
  // nineteen AF3 checkers that 404ed rather than failing.
  ["boltz2", ["tools/gpu/fold.js", "--model=/model-boltz2-int5/manifest.json", "--steps=50"]],
  ["protenix2", ["tools/gpu/fold.js", "--model=/model-protenix2-int5/manifest.json", "--steps=50"]],
];

const run = (args, env) => new Promise((resolve) => {
  const child = spawn("node", ["tools/gpu-chrome.mjs", ...args], {
    env: { ...process.env, ...env },
  });
  let text = "";
  child.stdout.on("data", (chunk) => { text += chunk; });
  child.stderr.on("data", (chunk) => { text += chunk; });
  child.on("close", (code) => resolve({ code, text }));
});

// 🔴 A SIGNATURE, NOT AN ABSENCE OF ERRORS. An uncaptured device error can leave
// the harness exiting 0 while the fold produced nothing.
const SIGNATURE = /"checksum": -?\d+|"atomChecksum": -?\d+|"meanPlddt": [\d.]+/;

/**
 * @param {{ env: object, question: string, broke: RegExp,
 *           precondition?: (text: string) => string | undefined }} gate
 *   `question` finishes "every model folds ..."; `precondition` returns a
 *   reason when the gate's own setup did not take, which is a FAIL of the
 *   harness and never a skip.
 */
export async function runFolds({ env, question, broke, precondition }) {
  let failed = 0;
  let skipped = 0;
  for (const [name, args] of FOLDS) {
    const { code, text } = await run(args, env);
    const unmet = precondition?.(text);
    const signature = text.match(SIGNATURE)?.[0];
    const ok = code === 0 && unmet === undefined && !broke.test(text) && signature !== undefined;
    // 🔴 AN ARM THAT NEVER REACHED A DEVICE IS NOT A MODEL THAT DOES NOT FOLD.
    // A missing bundle or dump 404s, and a bundle whose widths the loader
    // refuses raises, both before any pipeline exists - so those are SKIP,
    // named, and out of the count. On an M2 with a stale OpenDDE bundle and no
    // oracle dumps the stock gate said "3 of 6 do not fold" and all six folded.
    const missing = text.match(/failed to load ([^\s:]+): 404|(\S+) has an invalid byte length/)?.[0]
      ?? (/the dialect and the weights disagree/.test(text)
        ? "the local bundle disagrees with the loader - re-fetch it" : undefined);
    const skip = !ok && unmet === undefined && missing !== undefined;
    // ...and say WHICH failure it is: without DISPLAY on a headless box every
    // arm reads "no structure produced", which is one missing X server.
    const named = text.match(/[^\n]*(enables \w+|extension 'f16'|exceeds the maximum)[^\n]*/)?.[0]?.trim();
    const lastLine = text.trim().split("\n").filter(Boolean).pop()?.slice(0, 110);
    const why = unmet ?? signature ?? missing ?? named
      ?? `no structure produced - ${lastLine ?? "no output"}`;
    if (skip) skipped += 1; else if (!ok) failed += 1;
    console.log(`${ok ? "ok  " : skip ? "SKIP" : "FAIL"}  ${name.padEnd(9)} ${why}`);
  }
  const ran = FOLDS.length - skipped;
  console.log(failed === 0
    ? `\nevery model folds ${question}${skipped === 0 ? "" : ` (${skipped} skipped: `
      + "an artefact this box does not have, not a capability)"}`
    : `\n${failed} of ${ran} models do not fold ${question}`);
  process.exit(failed === 0 ? 0 : 1);
}
