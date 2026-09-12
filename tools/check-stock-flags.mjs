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
  const why = signature ?? named ?? `no structure produced - ${lastLine ?? "no output"}`;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name.padEnd(9)} ${why}`);
}
console.log(failed === 0
  ? "\nevery model folds on a stock Chrome"
  : `\n${failed} of ${FOLDS.length} models do not fold on a stock Chrome`);
process.exit(failed === 0 ? 0 : 1);
