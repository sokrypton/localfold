import { spawnSync } from "node:child_process";
import { describe, expect, it } from "./harness.js";

/**
 * cuda/rfd4p/featurise/tables.inc is GENERATED from shared/rfd4p/'s own data - the 147-token
 * vocabulary and its atom37 layouts, the condition registry and its enums, and the CATH maps
 * (cuda/rfd4p/featurise/gen-tables.mjs) - so the native featuriser builds what the page builds.
 * A change to any of those without regenerating leaves the native side on the old vocabulary while
 * every fold still runs; this fails instead. (The featuriser itself is held to the JavaScript
 * exporter BYTE FOR BYTE by cuda/rfd4p/gate.py's `featurise` arm.)
 */
describe("the rfd4p native featuriser's generated tables", () => {
  it("are current with the JavaScript they are generated from", () => {
    const repo = new URL("..", import.meta.url).pathname;
    const run = spawnSync(process.execPath,
      [...process.execArgv, `${repo}cuda/rfd4p/featurise/gen-tables.mjs`, "--check"],
      { encoding: "utf8" });
    expect(`${run.status} ${run.stdout}${run.stderr}`.trim())
      .toBe("0 cuda/rfd4p/featurise/tables.inc is current");
  });
});
