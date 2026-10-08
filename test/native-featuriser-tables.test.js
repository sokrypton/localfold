import { spawnSync } from "node:child_process";
import { describe, expect, it } from "./harness.js";

/**
 * cuda/featurise/tables.inc is GENERATED from the JavaScript featuriser's own data - the reference conformers, the
 * element symbols, the MSA alphabet and every family's featuriser conventions (tools/gen-native-featuriser-tables.mjs)
 * - so the native featurisers build what the page builds. A change to any of those without regenerating leaves the
 * native side on the old data while every fold still runs; this fails instead. (The featurisers themselves are held
 * to the JavaScript byte for byte by tools/check-native-featuriser.py.)
 */
describe("the native featurisers' generated tables", () => {
  it("are current with the JavaScript they are generated from", () => {
    const repo = new URL("..", import.meta.url).pathname;
    const run = spawnSync(process.execPath, [...process.execArgv, `${repo}tools/gen-native-featuriser-tables.mjs`, "--check"],
                          { encoding: "utf8" });
    expect(`${run.status} ${run.stdout}${run.stderr}`.trim()).toBe("0 cuda/featurise/tables.inc is current");
  });
});
