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

  // 🔴 A COVALENT INHIBITOR, WHICH IS WHAT `bondedAtomPairs` IS FOR. Sotorasib
  // is bonded to KRAS's cysteine 12 - that bond is the drug - and the job was
  // refused outright for naming it. It loads now, and the bond is asserted
  // below rather than just the rows: a job that folds the protein and the
  // ligand side by side with no bond between them is a different answer.
  // 🔴 AND THE BOND IS A ROW, NOT A HIDDEN FIELD. `bondedAtomPairs` becomes a
  // `contact` entity beside the protein and the ligand, so it is visible in the
  // list, editable, and deleted when the reader deletes it - where the first
  // version kept it in a page variable nobody could see, which is what the job
  // NAME was built and then removed for.
  "kras_g12c_sotorasib.json": { loads: ["protein:189x1", "ligand:MOVx1",
                                        "contact:A12:SG - B1:C25x1"] },
  // ...and the four this page does not fold, each naming its own reason.
  // 🔴 THIS ONE MOVED ITS REASON RATHER THAN LOSING IT: it carried
  // `bondedAtomPairs` AND a five-component glycan in one ligand entry, and
  // with the bonds read it refuses on the glycan. Five CCD codes in one entry
  // is one bonded chain, which this page does not build.
  // 🔴 AND IT LOADS NOW: five codes are ONE chain of five residues (ligandChain),
  // its glycosidic bonds and the Asn34 link contacts beside it - exact against
  // AF3's own batch (tools/check-batch-fields.js --target=glycan).
  "rnaseb_glycosylated.json": { loads: ["protein:124x1", "ligand:NAG,NAG,BMA,MAN,MANx1",
                                        "contact:A34:ND2 - B1:C1x1", "contact:B1:O4 - B2:C1x1",
                                        "contact:B2:O4 - B3:C1x1", "contact:B3:O3 - B4:C1x1",
                                        "contact:B3:O6 - B5:C1x1"] },
  // 🔴 MODIFIED BASES LOAD NOW: the featuriser takes a base's parent from its
  // chain's alphabet and the CCD (AF3's own batch is matched field for field on
  // both - tools/check-batch-fields.js, dna-5cm and rna-mods).
  "methylated_dna.json": { loads: ["dna:20x1", "dna:20x1"],
                           modifications: ["5CM@5", "5CM@9", "5CM@13", "5CM@17"] },
  "modified_rna.json": { loads: ["rna:25x1"], modifications: ["PSU@13", "5MC@18", "OMG@4"] },
  // 🔴 AND THIS ONE LOADS NOW, WHERE IT USED TO BE A REFUSAL. Biotin arrives
  // as a structure rather than a code and shared/chem/ builds it a component; see
  // docs/SMILES.md. It is kept in the corpus precisely because it is the one
  // example job that exercises the new path.
  "streptavidin_biotin_smiles.json": { loads: ["protein:126x1", "smiles:1x1"] },
  // ...and the pipeline's own kitchen-sink input, which uses nearly every
  // field the format has at once - the last to load, once an alignment carried
  // INLINE could be held (per chain copy, `job.alignments`). Its seed is 10.
  "alphafold_input.json": {
    // (the SMILES ligand last, as the file lists it and AF3 orders it)
    loads: ["protein:10x1", "protein:7x1", "dna:7x1", "dna:7x1", "rna:4x1",
            "ligand:ATPx1", "ligand:HEMx2", "ligand:MGx2", "ligand:NAG,FUCx1", "ligand:NAx3", "smiles:1x1",
            "contact:A1:CA - G1:CHAx1", "contact:K1:O6 - K2:C1x1"],
    seed: 10, alignedChains: 2,
  },
};

const shape = (entities) => entities.map((entity) =>
  // A SMILES row is summarised by its ATOM COUNT rather than its text, which
  // would make the expectation above a second copy of the input string. A
  // CONTACT is shown whole, because its text is the whole of what it says.
  `${entity.type}:${entity.type === "ligand" || entity.type === "contact" ? entity.value
    : entity.type === "smiles" ? 1 : entity.value.length}`
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
      // Every example but the kitchen sink seeds 42, and a seed read as a
      // string would be one.
      expect(job.seed).toBe(expectation.seed ?? 42);
      // ...and the alignments a file carries are held, one per chain copy
      const aligned = (job.alignments?.unpaired ?? []).filter(Boolean).length;
      expect(aligned).toBe(expectation.alignedChains ?? 0);
      expect(job.dialect).toBe("alphafold3");
      if (expectation.modifications !== undefined) {
        expect(job.entities[0].modifications.map((one) => `${one.code}@${one.position}`))
          .toEqual(expectation.modifications);
      }
    });
  }

  /**
   * 🔴 THE INLINE ALIGNMENTS ARE PER CHAIN COPY, IN FOLD ORDER, and an empty
   * string is not an absent one: AlphaFold 3 reads `""` as "this chain has no
   * alignment" and an absent field as "search". With nothing carried at all,
   * an empty string still turns the dial to single sequence.
   */
  it("holds a job's inline alignments per chain, and an empty one as none", () => {
    const job = JSON.parse(read("alphafold_input.json"));
    const loaded = jobFromJson(JSON.stringify(job));
    expect(loaded.alignments.unpaired.map((text) => (text ? "text" : text)))
      .toEqual(["text", null, null, null, "text"]);
    expect(loaded.alignments.paired.map((text) => (text ? "text" : text)))
      .toEqual(["text", null, null, null, null]);
    const bare = JSON.parse(read("ubiquitin_monomer.json"));
    bare.sequences[0].protein.unpairedMsa = "";
    const single = jobFromJson(JSON.stringify(bare));
    expect(single.alignments === undefined).toBe(true);
    expect(single.singleSequence).toBe(true);
  });

  /**
   * 🔴 THE COUNT IS ASSERTED, so that "nine of fourteen fold here" is a fact
   * somebody has to update deliberately. It was EIGHT until SMILES landed, and
   * this assertion is what made that a decision rather than a drift: the
   * streptavidin/biotin job moved from the refusal list to the loading one and
   * this line went red until somebody said so out loud.
   *
   * An inline alignment is the gap that remains.
   */
  /**
   * 🔴 AND "LOADS" IS NOT "FOLDS", WHICH THIS FILE CANNOT CHECK AND SHOULD NOT
   * IMPLY. Every assertion here is about what a file BECOMES - the entity list,
   * the seed, the dialect - and no test in the CPU suite has weights. The
   * second question is answered by `tools/fold-in-page.py --job=<path>`, which
   * drops the file on the page and presses Fold, and the answer as of
   * 2026-09-19 is that ALL NINE FOLD: 76 to 476 tokens, ligands from an `id`
   * list, a biotin built from SMILES, two phosphorylated residues and a
   * protein/DNA complex, none of them tripping the chain-geometry rule. The
   * per-file table is in docs/WEB.md.
   */
  it("loads all fourteen", () => {
    const loads = Object.values(EXPECTED).filter((one) => one.loads !== undefined);
    // 🔴 NINE UNTIL `bondedAtomPairs` LANDED, TEN UNTIL MODIFIED BASES DID,
    // TWELVE UNTIL A LIGAND COULD BE A CHAIN OF COMPONENTS, THIRTEEN UNTIL AN
    // INLINE ALIGNMENT COULD BE HELD. This count is
    // asserted so that moving a file between the two lists is a decision
    // somebody makes out loud - it has caught the prose going stale twice.
    // FOURTEEN once an alignment carried inline could be held.
    expect(loads).toHaveLength(14);
  });
});
