/**
 * AlphaFold 3's own example jobs, every one of them, read by our reader.
 *
 * 🔴 THIS IS THE ONLY TEST HERE WHOSE INPUT WE DID NOT WRITE. Every other case
 * in test/job-json.test.js is a file shaped the way the documentation says the
 * format is shaped, which checks the reader against our reading of the spec -
 * and both of the archive bugs this week were exactly that mistake made in the
 * other direction. These are the thirteen `examples/*.json` plus the
 * pipeline's own kitchen-sink `alphafold_input.json`, from
 * google-deepmind/alphafold3 and vendored under Apache 2.0 into
 * tools/fixtures/af3-jobs/. They use fields our own fixtures did not think
 * to: `dialect: "alphafold3"` with `version: 4`, an `id` LIST on a ligand
 * meaning four calcium ions, `description` keys inside a chain body, and
 * `modificationType`/`basePosition` where a protein says `ptmType`.
 *
 * 🔴 AND HALF OF THE VALUE IS THE REFUSALS. Six of the fourteen describe
 * chemistry this page does not build - three covalent-bond jobs, two with
 * modified bases, one SMILES ligand - and every one of them would parse
 * perfectly well as far as the sequence. A reader that loaded them would fold
 * a real structure of the right protein without the inhibitor bonded to it, or
 * with unmethylated DNA, and report it as the job that was asked for. The
 * expectation for those files is the name of the field that stopped them.
 *
 * 🔴 EVERY FILE IN THE DIRECTORY MUST APPEAR IN THE TABLE. Iterating the
 * directory and checking only what is listed would let a fixture added later
 * pass by not being mentioned, which is the silence this whole file is about.
 */
import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "./harness.js";
import { jobFromJson } from "../web/job-json.js";

const DIRECTORY = new URL("../tools/fixtures/af3-jobs/", import.meta.url);
const read = (name) => readFileSync(new URL(name, DIRECTORY), "utf8");

/**
 * What each of AlphaFold 3's examples is, to this page.
 *
 * `loads` is the entity list it becomes, as `type:VALUE-or-length xCOPIES`.
 * `refuses` is the substring of the refusal that names WHY - which is the part
 * a reader needs, since "unsupported" sends them looking through the file.
 */
const EXPECTED = {
  "barnase_barstar.json": { loads: ["protein:110x1", "protein:89x1"] },
  // 🔴 FOUR IONS AS AN `id` LIST. AlphaFold 3 spells four calciums as one
  // ligand entry with four ids; read as a name this is a calmodulin with ONE
  // calcium in it, which is a different molecule that folds perfectly well.
  "calmodulin_4calcium.json": { loads: ["protein:149x1", "ligand:CAx4"] },
  "erk2_phosphorylated.json": { loads: ["protein:360x1"],
                                modifications: ["TPO@185", "PTR@187"] },
  "tetr_dimer_dna.json": { loads: ["protein:218x2", "dna:20x1", "dna:20x1"] },
  "tetr_dimer_tetracycline.json": { loads: ["protein:218x2", "ligand:TACx2"] },
  "tetr_homodimer.json": { loads: ["protein:218x2"] },
  "u1a_rna_hairpin.json": { loads: ["protein:101x1", "rna:21x1"] },
  "ubiquitin_monomer.json": { loads: ["protein:76x1"] },

  // ...and the five this page does not fold, each naming its own reason.
  "kras_g12c_sotorasib.json": { refuses: "bondedAtomPairs" },
  "rnaseb_glycosylated.json": { refuses: "bondedAtomPairs" },
  "methylated_dna.json": { refuses: "modified bases" },
  "modified_rna.json": { refuses: "modified bases" },
  "streptavidin_biotin_smiles.json": { refuses: "smiles" },
  // ...and the pipeline's own kitchen-sink input, which uses nearly every
  // field the format has at once. It refuses on the first one it hits.
  "alphafold_input.json": { refuses: "bondedAtomPairs" },
};

const shape = (entities) => entities.map((entity) =>
  `${entity.type}:${entity.type === "ligand" ? entity.value : entity.value.length}`
  + `x${entity.copies}`);

describe("AlphaFold 3's own example jobs", () => {
  it("covers every file in the fixture directory", () => {
    const onDisk = readdirSync(DIRECTORY).filter((name) => name.endsWith(".json"));
    expect(onDisk.sort()).toEqual(Object.keys(EXPECTED).sort());
  });

  for (const [name, expectation] of Object.entries(EXPECTED)) {
    if (expectation.refuses !== undefined) {
      it(`refuses ${name} for ${expectation.refuses}`, () => {
        let message = "(no refusal)";
        try { jobFromJson(read(name)); } catch (error) { message = error.message; }
        expect(message.includes(expectation.refuses)).toBe(true);
      });
      continue;
    }
    it(`loads ${name}`, () => {
      const job = jobFromJson(read(name));
      expect(shape(job.entities)).toEqual(expectation.loads);
      // Every example seeds 42, and a seed read as a string would be one.
      expect(job.seed).toBe(42);
      expect(job.dialect).toBe("alphafold3");
      if (expectation.modifications !== undefined) {
        expect(job.entities[0].modifications.map((one) => `${one.code}@${one.position}`))
          .toEqual(expectation.modifications);
      }
    });
  }

  /**
   * 🔴 THE REFUSALS STACK, and this is the only place that can show it. One
   * guard hiding the rest would mean a reader who deletes the field we named
   * gets the file loaded rather than the next refusal - so the kitchen-sink
   * input is peeled once, deliberately, and asked what it says next. It says
   * the inline alignment, which is the honest answer: the page has nowhere to
   * attach one yet.
   */
  it("names the next reason when the first is removed", () => {
    const job = JSON.parse(read("alphafold_input.json"));
    delete job.bondedAtomPairs;
    let message = "(no refusal)";
    try { jobFromJson(JSON.stringify(job)); } catch (error) { message = error.message; }
    expect(message.includes("unpairedMsa")).toBe(true);
  });

  /**
   * 🔴 THE COUNT IS ASSERTED, so that "eight of fourteen fold here" is a fact
   * somebody has to update deliberately. Bonded chemistry and modified bases
   * are the two gaps, at two examples each, and that is the argument for which
   * to build next.
   */
  it("folds eight of the fourteen, and says which two gaps cost the rest", () => {
    const loads = Object.values(EXPECTED).filter((one) => one.loads !== undefined);
    const bonded = Object.values(EXPECTED)
      .filter((one) => one.refuses === "bondedAtomPairs");
    const bases = Object.values(EXPECTED)
      .filter((one) => one.refuses === "modified bases");
    expect(loads).toHaveLength(8);
    // ...three, with the kitchen-sink input, which is bonded chemistry too.
    expect(bonded).toHaveLength(3);
    expect(bases).toHaveLength(2);
  });
});
