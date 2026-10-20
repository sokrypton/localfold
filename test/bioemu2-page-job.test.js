import { readFileSync } from "node:fs";
import { describe, expect, it } from "./harness.js";
import { jobInputJson } from "../web/job-json.js";
import { bioemu2JobFromAf3Json, featuriseBioemu2 } from "../shared/bioemu2/featurise.js";
import { splitA3mByChains } from "../shared/bioemu2/inputs.js";

/**
 * 🔴 BioEmu2 LAYS CHAINS OUT ENTITY BY ENTITY IN JOB ORDER, and its atom keys
 * carry the job's chain ids. The page's `expandEntities` puts every polymer
 * before every ligand and letters them itself, so an adapter through it would
 * reorder a job whose ligand row comes first and rename its chains. The page
 * (web/bioemu2-model.js) goes through `jobInputJson` over the ROWS, in row
 * order with their own ids; this holds it there.
 */
const ccd = (code) => readFileSync(new URL(`../tools/fixtures/ccd/${code}.cif`, import.meta.url), "utf8");

describe("the BioEmu2 job the page builds", () => {
  it("keeps a ligand row that precedes a protein row first, under its own id", () => {
    const entities = [{ type: "ligand", value: "GOL" }, { type: "protein", value: "GWSTELEK" }];
    const job = bioemu2JobFromAf3Json(JSON.parse(jobInputJson({ name: "t", seed: 1, entities })));
    expect(job.chains.map((c) => [c.kind, c.id])).toEqual([["ligand", "A"], ["protein", "B"]]);
    const { features, atomKeys } = featuriseBioemu2(job, { ccd });
    expect(atomKeys[0][0]).toBe("A");
    expect(features.asym_id[0]).toBe(0);
    // the glycerol's six atom tokens come first, then the eight residues
    expect(features.asym_id.length).toBe(6 + 8);
    expect(features.asym_id[6]).toBe(1);
  });

  it("splits a merged alignment back into each chain's columns, insertions with the residue before", () => {
    const merged = ">101\nACDE\n>x\nAcCD-\n";
    const [first, second] = splitA3mByChains(merged, [2, 2]);
    expect(first.trim().split("\n")[3]).toBe("AcC");
    expect(second.trim().split("\n")[3]).toBe("D-");
    expect(first.trim().split("\n")[1]).toBe("AC");
    expect(second.trim().split("\n")[1]).toBe("DE");
  });
});
