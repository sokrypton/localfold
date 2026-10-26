import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { describe, expect, it } from "./harness.js";

/**
 * shared/rfd4p/pipeline/query-tables.js is GENERATED from atomworks' and biotite's constants - the
 * tables upstream's selection language derives `is_protein_backbone`, `is_tip_atom`, `is_metal`
 * and the rest from. Checked against them where upstream's venv is on the box, skipped where not;
 * tools/check-rfd4p-query.py (a cuda/rfd4p/gate.py arm) holds the language itself to atomworks.
 */
describe("the selection language's tables", () => {
  const python = process.env.RFD4P_PYTHON ?? `${homedir()}/venv_rfd4p/bin/python`;
  it.skipIf(!existsSync(python))("are atomworks', byte for byte", () => {
    const repo = new URL("..", import.meta.url).pathname;
    const run = spawnSync(python, [`${repo}tools/gen-rfd4p-query-tables.py`, "--check"], { encoding: "utf8" });
    const out = `${run.stdout}${run.stderr}`.split("\n").filter((l) => l.includes("query-tables.js")).join("");
    expect(`${run.status} ${out}`.trim()).toBe("0 shared/rfd4p/pipeline/query-tables.js is current");
  });
});
