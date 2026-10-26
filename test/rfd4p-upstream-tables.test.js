import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { describe, expect, it } from "./harness.js";

/**
 * shared/rfd4p/pipeline/query-tables.js and ccd-templates.js are GENERATED from atomworks' and
 * biotite's own data - the tables upstream's selection language derives `is_protein_backbone`,
 * `is_tip_atom` and the rest from, and the CCD templates a raw structure is prepared from. Checked
 * against them where upstream's venv is on the box, skipped where not; tools/check-rfd4p-query.py
 * and tools/check-rfd4p-load.py (cuda/rfd4p/gate.py arms) hold the language and the loader to
 * atomworks itself.
 */
describe("the tables generated from upstream's own", () => {
  const python = process.env.RFD4P_PYTHON ?? `${homedir()}/venv_rfd4p/bin/python`;
  it.skipIf(!existsSync(python))("are atomworks', byte for byte", () => {
    const repo = new URL("..", import.meta.url).pathname;
    const run = spawnSync(python, [`${repo}tools/gen-rfd4p-query-tables.py`, "--check"], { encoding: "utf8" });
    const out = `${run.stdout}${run.stderr}`.split("\n").filter((l) => l.includes("query-tables.js")).join("");
    expect(`${run.status} ${out}`.trim()).toBe("0 shared/rfd4p/pipeline/query-tables.js is current");
  });
  // The CCD templates a raw structure is prepared from (pipeline/fresh.js).
  it.skipIf(!existsSync(python))("the CCD templates are biotite's, byte for byte", () => {
    const repo = new URL("..", import.meta.url).pathname;
    const run = spawnSync(python, [`${repo}tools/gen-rfd4p-ccd-templates.py`, "--check"], { encoding: "utf8" });
    const out = `${run.stdout}${run.stderr}`.split("\n").filter((l) => l.includes("ccd-templates.js")).join("");
    expect(`${run.status} ${out}`.trim()).toBe("0 shared/rfd4p/pipeline/ccd-templates.js is current");
  });
});
