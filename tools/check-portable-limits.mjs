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
import { runFolds } from "./gate-folds.mjs";

// The fold list and the verdict are shared with check-stock-flags.mjs - see
// gate-folds.mjs for why there is one copy.
await runFolds({
  // 🔴 `--spec-floor` ASKS THE STANDARD'S QUESTION, NOT THIS SESSION'S.
  // PORTABLE_CEILINGS is what the machines we have report; WebGPU guarantees
  // 256 invocations and a 256 X extent where that says 1024. Four of six models
  // fail there - see SPEC_FLOOR_CEILINGS.
  env: process.argv.includes("--spec-floor")
    ? { LOCALFOLD_SPEC_FLOOR: "1" } : { LOCALFOLD_PORTABLE_LIMITS: "1" },
  question: process.argv.includes("--spec-floor")
    ? "at WebGPU's guaranteed minimum limits" : "at the portable limit ceiling",
  broke: /"error"|Error:|uncaptured|exceeds the maximum/,
  // 🔴 THE CEILING MUST HAVE BEEN APPLIED. Without this a run where the
  // environment did not reach the page reports six passes and asks nothing -
  // and that is a failure of the gate, never a skip.
  precondition: (text) => (new RegExp(`portable limits: workgroup storage ${
    process.argv.includes("--spec-floor") ? 16384 : 32768}`).test(text)
    ? undefined : "the ceiling never reached the page"),
});
