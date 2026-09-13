/**
 * Does every model fold on the browser a VISITOR has?
 *
 *     node tools/check-stock-flags.mjs
 *
 * 🔴 WHY THIS IS A GATE AND NOT A CURIOSITY. Every other gate in this
 * repository runs through `tools/gpu-chrome.mjs`, which passes
 * `--enable-dawn-features=vulkan_enable_f16_on_nvidia` and
 * `--enable-unsafe-webgpu`. A stock Chrome has NEITHER on any NVIDIA GPU -
 * Dawn gates f16 vendor-wide pending crbug.com/42251215, and the subgroup
 * matrix units are `chromium-experimental-` on every platform. So the
 * configuration every gate checks is not the configuration the site ships.
 *
 * That is not a hypothetical: measured this way, **OpenDDE and ESMFold2 did not
 * fold at all**, both on `extension 'f16' is not allowed in the current
 * environment`, for as long as those paths had existed. See docs/A100.md.
 *
 * It runs the four folds and nothing else - this gate is about a capability
 * being absent, which a fold either survives or does not. The differential
 * checkers are the flagged run's job.
 */
import { spawn } from "node:child_process";

const FOLDS = [
  ["AF2", ["tools/gpu/fold-af2.js", "--repeat=2"]],
  ["AF3", ["tools/gpu/fold.js", "--model=/model-af3-int5/manifest.json"]],
  ["OpenDDE", ["tools/gpu/fold-opendde.js", "--target=6mrr", "--steps=16"]],
  ["ESMFold2", ["tools/gpu/fold-esmfold2.js", "--bundle=/model-esmfold2-int5"]],
  // 🔴 THE TWO NEWEST MODELS WERE NOT IN THIS LIST, which is the whole failure
  // this file exists for: every other harness here passes the two developer
  // flags, so a model that needs them folds everywhere except on a visitor's
  // browser. boltz2 is the one to watch - its token transformer amplifies its
  // input by ~2.2e4 and it already refuses f16 on the DENOISE gate (2.21e-1
  // against 3.50e-3), so a device that resolves precision differently is
  // exactly where it would come apart.
  // 🔴 AND NO `--dump=`, WHICH THIS FILE'S OWN HEADER ALREADY SAID. These two
  // arms named `/oracle-dumps/af3-batch-*-6mrr.json`, and `oracle-dumps/` is
  // gitignored WHOLE - so on any machine but the one that generated them both
  // arms died on a 404 before reaching a device, and the summary reported
  // "2 of 6 models do not fold on a stock Chrome". They fold. Measured on an
  // M2 with the dumps removed: boltz2 96.46015389206518 and protenix2
  // 84.73666947394713, identical to their flagged runs to every digit.
  //
  // A batch dump is an ORACLE input and this gate is about a capability being
  // absent. Keeping one here made a portability check depend on an artefact
  // that does not travel, which is the same fault docs/PARITY.md records for
  // nineteen AF3 checkers that 404ed rather than failing.
  ["boltz2", ["tools/gpu/fold.js", "--model=/model-boltz2-int5/manifest.json",
              "--steps=50"]],
  ["protenix2", ["tools/gpu/fold.js", "--model=/model-protenix2-int5/manifest.json",
                 "--steps=50"]],
];

const run = (args) => new Promise((resolve) => {
  const child = spawn("node", ["tools/gpu-chrome.mjs", ...args], {
    env: { ...process.env, LOCALFOLD_STOCK_FLAGS: "1" },
  });
  let text = "";
  child.stdout.on("data", (chunk) => { text += chunk; });
  child.stderr.on("data", (chunk) => { text += chunk; });
  child.on("close", (code) => resolve({ code, text }));
});

let failed = 0;
let skipped = 0;
for (const [name, args] of FOLDS) {
  const { code, text } = await run(args);
  // 🔴 THE EXIT CODE IS NOT ENOUGH. An uncaptured device error can leave the
  // harness reporting success while the fold produced nothing, which is exactly
  // how this went unseen - so the text is searched for the failure too, and a
  // SIGNATURE is required rather than merely an absence of errors.
  const broke = /enables (f16|subgroups|subgroup_size_control|chromium_experimental)|"error"|Error:|uncaptured/
    .test(text);
  const signature = text.match(/"checksum": -?\d+|"atomChecksum": -?\d+|"meanPlddt": [\d.]+/)?.[0];
  const ok = code === 0 && !broke && signature !== undefined;
  if (!ok) failed += 1;
  // 🔴 AND SAY WHICH KIND OF FAILURE IT IS. Run without DISPLAY on a headless
  // box every arm reads "no structure produced", which looks like four broken
  // models and is one missing X server - this needs `DISPLAY=:99
  // XDG_RUNTIME_DIR=/tmp/xdg` like every other GPU lane. The last line of the
  // output distinguishes them.
  const named = text.match(/[^\n]*(enables \w+|extension 'f16')[^\n]*/)?.[0]?.trim();
  const lastLine = text.trim().split("\n").filter(Boolean).pop()?.slice(0, 110);
  // 🔴 AN ARM THAT NEVER REACHED A DEVICE IS NOT A MODEL THAT DOES NOT FOLD.
  // A missing bundle or dump 404s, and a bundle whose widths the loader refuses
  // raises, both BEFORE any pipeline is created - so counting those as
  // stock-flag failures reports a capability verdict this gate did not observe.
  // Measured: on a box with a stale OpenDDE bundle and no oracle dumps, this
  // said "3 of 6 models do not fold on a stock Chrome" and all six folded.
  const missing = text.match(/failed to load ([^\s:]+): 404|(\S+) has an invalid byte length/)?.[0]
    ?? (/the dialect and the weights disagree/.test(text)
      ? "the local bundle disagrees with the loader - re-fetch it" : undefined);
  const why = signature ?? missing ?? named
    ?? `no structure produced - ${lastLine ?? "no output"}`;
  const verdict = ok ? "ok  " : missing === undefined ? "FAIL" : "SKIP";
  if (!ok && missing !== undefined) { failed -= 1; skipped += 1; }
  console.log(`${verdict}  ${name.padEnd(9)} ${why}`);
}
const ran = FOLDS.length - skipped;
console.log(failed === 0
  ? `\nevery model folds on a stock Chrome${skipped === 0 ? "" : ` (${skipped} skipped: `
    + "an artefact this box does not have, not a capability)"}`
  : `\n${failed} of ${ran} models do not fold on a stock Chrome`);
process.exit(failed === 0 ? 0 : 1);
