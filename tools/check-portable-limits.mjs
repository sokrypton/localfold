/**
 * Does every model fold on a device at the PORTABLE limit ceiling?
 *
 *     npm run test:portable
 *
 * 🔴 A RAISED LIMIT IS A PREDICTION ABOUT THE NEXT MACHINE. This port asks its
 * adapter for the most it will give on five limits, and one of them differs by
 * a factor this card hides: `maxComputeWorkgroupStorageSize` is 49152 here and
 * **32768 on Metal**. A kernel that takes a 40 KiB tile compiles here, measures
 * well here, and fails to create its pipeline on an Apple part - with an error
 * naming a shader rather than a limit, which is the hardest kind to read from
 * another machine.
 *
 * So this asks the question HERE. `LOCALFOLD_PORTABLE_LIMITS=1` caps what the
 * device requests at `PORTABLE_CEILINGS`, and anything that refuses under it
 * refuses on the weaker part. It is the limits twin of
 * `tools/check-stock-flags.mjs`, which asks the same question about FEATURES -
 * and that one exists because two of four models did not fold without the
 * developer flags and no gate could see it.
 *
 * 🔴 IT CANNOT SIMULATE A LARGER LIMIT. The M2's `maxStorageBufferBindingSize`
 * is 4 GiB against this card's 2, so its binding ceiling is LOOSER and nothing
 * here can ask about it; docs/AF2.md has that half. Only limits that can be
 * lowered are in the table.
 */
import { spawn } from "node:child_process";

const FOLDS = [
  ["AF2", ["tools/gpu/fold-af2.js", "--repeat=2"]],
  ["AF3", ["tools/gpu/fold.js", "--model=/model-af3-int5/manifest.json"]],
  ["OpenDDE", ["tools/gpu/fold-opendde.js", "--target=6mrr", "--steps=16"]],
  ["ESMFold2", ["tools/gpu/fold-esmfold2.js", "--bundle=/model-esmfold2-int5"]],
  ["boltz2", ["tools/gpu/fold.js", "--model=/model-boltz2-int5/manifest.json",
              "--dump=/oracle-dumps/af3-batch-boltz2-6mrr.json", "--steps=50"]],
  ["protenix2", ["tools/gpu/fold.js", "--model=/model-protenix2-int5/manifest.json",
                 "--dump=/oracle-dumps/af3-batch-protenix2-6mrr.json", "--steps=50"]],
];

const run = (args) => new Promise((resolve) => {
  const child = spawn("node", ["tools/gpu-chrome.mjs", ...args], {
    env: { ...process.env, LOCALFOLD_PORTABLE_LIMITS: "1" },
  });
  let text = "";
  child.stdout.on("data", (chunk) => { text += chunk; });
  child.stderr.on("data", (chunk) => { text += chunk; });
  child.on("close", (code) => resolve({ code, text }));
});

let failed = 0;
for (const [name, args] of FOLDS) {
  const { code, text } = await run(args);
  // 🔴 THE CEILING MUST HAVE BEEN APPLIED. Without this line a run where the
  // environment did not reach the page reports six passes and asks nothing.
  const capped = /portable limits: workgroup storage 32768/.test(text);
  const broke = /"error"|Error:|uncaptured|exceeds the maximum/.test(text);
  const signature = text.match(/"checksum": -?\d+|"atomChecksum": -?\d+|"meanPlddt": [\d.]+/)?.[0];
  const ok = code === 0 && capped && !broke && signature !== undefined;
  if (!ok) failed += 1;
  const lastLine = text.trim().split("\n").filter(Boolean).pop()?.slice(0, 110);
  const why = !capped ? "the ceiling never reached the page"
    : (signature ?? `no structure produced - ${lastLine ?? "no output"}`);
  console.log(`${ok ? "ok  " : "FAIL"}  ${name.padEnd(9)} ${why}`);
}
console.log(failed === 0
  ? "\nevery model folds at the portable limit ceiling"
  : `\n${failed} of ${FOLDS.length} models do not fold at the portable limit ceiling`);
process.exit(failed === 0 ? 0 : 1);
