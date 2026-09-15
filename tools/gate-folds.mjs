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
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BASELINE = join(dirname(fileURLToPath(import.meta.url)), "gate-baseline.json");

/** 6MRR's chain A, which is what every other row in this table folds. */
const SIX_MRR = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";

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
  // 🔴 THE TWO WIDEST SHAPES IN THE PANEL, WHICH IS WHY THEY BELONG HERE.
  // IntelliFold-2's trunk pair is 512 channels against AlphaFold 3's 128 and
  // its template stack 256 against 64, so every workgroup-storage decision in
  // the port is being asked a question no other model asks - and this gate
  // exists precisely because `maxComputeWorkgroupStorageSize` is 49152 here and
  // 32768 on Metal. A tile that fits at 128 channels and not at 512 would fail
  // on an Apple part first and on nothing here.
  //
  // 🔴 AND THEY FOLD FROM THE SEQUENCE, NOT FROM A DUMP - `--sequence=` rather
  // than `--target=` alone, which folds AlphaFold 3's own featurised batch
  // through whatever `--model=` names and would test neither model's
  // featuriser. 6MRR's sequence, the same one every other row uses.
  ["intellifold2", ["tools/gpu/fold.js", "--model=/model-intellifold2-int5/manifest.json",
                    `--sequence=${SIX_MRR}`, "--steps=50"]],
  ["rosettafold3", ["tools/gpu/fold.js", "--model=/model-rosettafold3-int5/manifest.json",
                    `--sequence=${SIX_MRR}`, "--steps=50"]],
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
export async function runFolds({ env, question, broke, precondition, arm }) {
  let failed = 0;
  let skipped = 0;
  const measured = {};
  let adapter;
  for (const [name, args] of FOLDS) {
    const { code, text } = await run(args, env);
    adapter ??= text.match(/^\[gpu-chrome\] adapter: (.+)$/m)?.[1];
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
    if (ok && signature !== undefined) measured[name] = signature;
    console.log(`${ok ? "ok  " : skip ? "SKIP" : "FAIL"}  ${name.padEnd(9)} ${why}`);
  }
  const drifted = arm === undefined ? 0 : checkBaseline(arm, adapter, measured);
  const ran = FOLDS.length - skipped;
  console.log(failed === 0
    ? `\nevery model folds ${question}${skipped === 0 ? "" : ` (${skipped} skipped: `
      + "an artefact this box does not have, not a capability)"}`
    : `\n${failed} of ${ran} models do not fold ${question}`);
  process.exit(failed === 0 && drifted === 0 ? 0 : 1);
}

/**
 * 🔴 WHAT THIS IS FOR: A RECORDED NUMBER THAT NOBODY RE-RUNS IS A BASELINE FOR
 * EXACTLY AS LONG AS IT TAKES SOMEBODY TO BELIEVE IT.
 *
 * docs/A100.md recorded this gate's four figures on 2026-09-12. Three days
 * later two of them were wrong - AF3 85.8348 against 83.0999 and OpenDDE
 * 92.0489 against 92.0382 - and CLAUDE.md's spec-floor line carried a THIRD
 * OpenDDE value, 92.1193, from a fourth day. Bisected, each move is one commit
 * and both are correctness fixes whose own messages state the new figure:
 *
 *   28b3965  AF3's diffusion atom encoder read the `_1` pair tensors where the
 *            reference's graph calls the base form - denoise 4.19e-1 -> 1.55e-5,
 *            pLDDT 85.8301 -> 83.1276. "The old numbers were computed with
 *            weights the model does not use, so they were never a baseline."
 *   7c13e05  the template stage's precision pin: 83.1276 -> 83.1295.
 *   9b2cd37  singleProjectMaxSplits, which regroups a sum: 1.3e-7.
 *   8dba05f  three models folded a single sequence one MSA row short - OpenDDE
 *            92.1200 -> 92.0396 and RMSD 1.527 -> 1.518, stated in the commit.
 *
 * So nothing was broken and nothing was hidden. What failed is that the figures
 * lived in PROSE, in three documents, and the only thing that re-ran them was
 * somebody deciding to. This file is where they live now: one machine-readable
 * record, re-checked on every run, and a legitimate improvement has to update
 * it deliberately instead of silently diverging.
 *
 * 🔴 AND IT IS KEYED ON THE ADAPTER, because a checksum does not travel between
 * machines - the A100 folds fold-af2.js at -1287025 and an M2 at -1282976, both
 * correct. A baseline recorded on another box is REPORTED and not enforced;
 * only the box that wrote it can be held to it. That is also why
 * `tools/gpu-chrome.mjs` prints the adapter at all: it always collected
 * `adapter.info` and always threw it away, so every figure this repository has
 * ever published was machine-anonymous.
 *
 * `--write-baseline` records the current run. Read the diff before you do.
 */
function checkBaseline(arm, adapter, measured) {
  const write = process.argv.includes("--write-baseline");
  const stored = existsSync(BASELINE)
    ? JSON.parse(readFileSync(BASELINE, "utf8")) : { adapter: null, arms: {} };

  if (write) {
    stored.adapter = adapter ?? stored.adapter;
    stored.recorded = new Date().toISOString().slice(0, 10);
    stored.arms = { ...stored.arms, [arm]: measured };
    writeFileSync(BASELINE, `${JSON.stringify(stored, null, 2)}\n`);
    console.log(`\nwrote ${Object.keys(measured).length} signatures for "${arm}"`
      + ` to tools/gate-baseline.json (${stored.adapter})`);
    return 0;
  }

  const { lines, enforced } = compareBaseline({ arm, adapter, measured, stored });
  for (const line of lines) console.log(line);
  return enforced;
}

/**
 * The comparison on its own, with no file and no argv, so `npm test` can watch
 * it fail without a GPU - `test/gate-baseline.test.js`. A check whose only
 * proof of working is a twenty-minute fold is a check nobody falsifies.
 *
 * @returns {{ lines: string[], enforced: number }} `enforced` is what the gate
 *   exits non-zero on, and is 0 on another machine's baseline however many
 *   signatures differ.
 */
export function compareBaseline({ arm, adapter, measured, stored }) {
  const recorded = stored.arms?.[arm];
  if (recorded === undefined) {
    return { lines: [`\nno baseline for "${arm}" - record one with --write-baseline`],
      enforced: 0 };
  }
  const differs = Object.entries(measured)
    .filter(([name, value]) => recorded[name] !== undefined && recorded[name] !== value)
    .map(([name, value]) => `  ${name.padEnd(9)} ${recorded[name]}  ->  ${value}`);
  if (differs.length === 0) {
    const same = Object.keys(measured).filter((name) => recorded[name] !== undefined).length;
    return { lines: [`baseline: ${same} of ${Object.keys(recorded).length} signatures`
      + ` unchanged since ${stored.recorded}`], enforced: 0 };
  }
  // 🔴 A BASELINE FROM ANOTHER BOX IS EVIDENCE, NOT A BAR. The A100 folds
  // fold-af2.js at -1287025 and an M2 at -1282976 over the same code and the
  // same input, because the two resolve different attention kernels. Both are
  // correct, and a gate that failed on that would be a gate everyone disables.
  const mine = stored.adapter === null || stored.adapter === adapter;
  const lines = [`\n${mine ? "🔴 " : ""}${differs.length} signature(s) differ from the`
    + ` baseline recorded ${stored.recorded}${mine ? "" : ` on ${stored.adapter}`}:`, ...differs];
  if (!mine) {
    lines.push(`  ...reported and NOT failed: this is ${adapter ?? "an unidentified adapter"}`
      + " and a signature does not travel between machines.");
    return { lines, enforced: 0 };
  }
  const script = { stock: "test:stock", portable: "test:portable",
    "spec-floor": "test:spec-floor" }[arm] ?? "test:portable";
  lines.push("  If the change is intended, say so in the commit and re-record with"
    + `\n      npm run ${script} -- --write-baseline`
    + "\n  A figure that moves with nobody noticing is what this check exists to stop.");
  return { lines, enforced: differs.length };
}
