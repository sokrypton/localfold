import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { describe, expect, it } from "./harness.js";

/**
 * shared/rfd4p/cath-mapping.js is GENERATED from RFD4-Proteina's `data/cath_label_mapping.pkl`,
 * the CATH label -> embedding-row mapping the checkpoint was trained with. The tables it replaced
 * were hand-written and disagreed with it on 26 of 32 architecture rows and all 12 topology rows.
 * Checked against the pickle where an upstream checkout is on the box, and skipped where not.
 */
describe("the CATH mapping", () => {
  const upstream = process.env.RFD4P_UPSTREAM ?? `${homedir()}/RFD4-Proteina`;
  it.skipIf(!existsSync(`${upstream}/data/cath_label_mapping.pkl`))("is upstream's, byte for byte", () => {
    const repo = new URL("..", import.meta.url).pathname;
    const run = spawnSync("python3", [`${repo}tools/gen-rfd4p-cath-mapping.py`, "--check",
      `--upstream=${upstream}`], { encoding: "utf8" });
    expect(`${run.status} ${run.stdout}${run.stderr}`.trim()).toBe("0 shared/rfd4p/cath-mapping.js is current");
  });
});
