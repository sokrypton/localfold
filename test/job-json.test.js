/**
 * Reading an AlphaFold 3 job JSON, in both of the dialects that name.
 *
 * 🔴 THE ROUND TRIP IS THE POINT, AND IT IS THE LAST TEST HERE. Everything
 * above it checks one field; what the archive actually promises is that the
 * file it writes describes the fold it wrote it for, and the only way to say
 * that is to write one and read it back. Both times a field went missing this
 * week - the templates, then the modifications - it went missing on the way
 * OUT, with every field-by-field test passing.
 */
import { describe, expect, it } from "./harness.js";
import { jobFromJson, jobRequestJson } from "../web/job-json.js";

const server = (sequences, extra = {}) => JSON.stringify([{
  name: "j", modelSeeds: ["7"], sequences,
  dialect: "alphafoldserver", version: 3, ...extra,
}]);
const open = (sequences, extra = {}) => JSON.stringify({
  name: "j", modelSeeds: [7], sequences, version: 2, ...extra,
});
const refusal = (text) => {
  try { jobFromJson(text); } catch (error) { return error.message; }
  return "(no refusal)";
};

describe("reading the server's own dialect", () => {
  it("reads a chain, its copies and its seed", () => {
    const job = jobFromJson(server([
      { proteinChain: { sequence: "ACDEFGHIK", count: 2 } },
    ]));
    expect(job.dialect).toBe("alphafoldserver");
    expect(job.seed).toBe(7);
    expect(job.entities).toHaveLength(1);
    expect(job.entities[0].type).toBe("protein");
    expect(job.entities[0].copies).toBe(2);
  });

  it("reads a modification back as the row spells it", () => {
    const job = jobFromJson(server([
      { proteinChain: { sequence: "ACSEFGHIK", count: 1,
        modifications: [{ ptmType: "CCD_SEP", ptmPosition: 3 }] } },
    ]));
    expect(job.entities[0].modifications).toEqual([{ code: "SEP", position: 3 }]);
  });

  // The server has one template control and it is "search for one", so that is
  // what the flag becomes - not a guess at which structure it found.
  it("reads useStructureTemplate as the search", () => {
    const job = jobFromJson(server([
      { proteinChain: { sequence: "ACDEFGHIK", count: 1, useStructureTemplate: true } },
    ]));
    expect(job.entities[0].template).toEqual({ kind: "search" });
  });

  it("reads a ligand and a nucleic chain", () => {
    const job = jobFromJson(server([
      { proteinChain: { sequence: "ACDEFGHIK", count: 1 } },
      { dnaSequence: { sequence: "ACGTACGT", count: 2 } },
      { ligand: { ligand: "atp", count: 1 } },
    ]));
    expect(job.entities.map((entity) => entity.type))
      .toEqual(["protein", "dna", "ligand"]);
    expect(job.entities[2].value).toBe("ATP");
  });
});

/**
 * 🔴 CHECKED AGAINST AlphaFold 3'S OWN PARSER, NOT AGAINST THE DOCUMENTATION.
 * These four came out of reading `folding_input.py` - the code that actually
 * reads these files - after the writer was checked against it. Every one of
 * them is a real AlphaFold Server export this reader got wrong, and NOT ONE
 * could be caught by the example corpus, because all fourteen of those files
 * are the other dialect.
 */
describe("what AlphaFold 3's own parser does with the server's files", () => {
  // 🔴 AN ION IS ITS OWN ENTRY. `Ligand.from_alphafoldserver_dict` takes
  // `ligand` or `ion` alike, and reading only the first refused every server
  // job with a magnesium in it - which is half of what the ligand menu here is
  // for. See COMMON_IONS in web/entities.js.
  it("reads an ion entry as a ligand", () => {
    const job = jobFromJson(server([
      { proteinChain: { sequence: "ACDEFGHIK", count: 1 } },
      { ion: { ion: "MG", count: 2 } },
    ]));
    expect(job.entities[1].type).toBe("ligand");
    expect(job.entities[1].value).toBe("MG");
    expect(job.entities[1].copies).toBe(2);
  });

  // ...and upstream strips the prefix, so a code kept whole would be a
  // five-letter component this page fetches and does not find.
  it("strips the CCD_ prefix upstream strips", () => {
    const job = jobFromJson(server([
      { proteinChain: { sequence: "ACDEFGHIK", count: 1 } },
      { ligand: { ligand: "CCD_ATP", count: 1 } },
    ]));
    expect(job.entities[1].value).toBe("ATP");
  });

  // Both of these are in the server's allowed key set AND raise in
  // folding_input.py. A glycan is chemistry this page does not build; a
  // template date changes which template is found, so dropping it folds a
  // different job with the same sequence.
  for (const [field, value] of [["glycans", [{ residues: "NAG" }]],
                                ["maxTemplateDate", "2021-09-30"]]) {
    it(`refuses ${field}, as upstream does`, () => {
      expect(refusal(server([{ proteinChain: {
        sequence: "ACDEFGHIK", count: 1, [field]: value } }]))).toContain(field);
    });
  }

  /**
   * 🔴 AN UNKNOWN KEY IS REFUSED BECAUSE ALPHAFOLD 3 REFUSES IT. Every chain
   * class upstream calls `_validate_keys` and raises on anything else, so
   * reading past one is not generosity - it folds a job the reference
   * implementation would not have run, from a file whose stray key is most
   * likely a misspelling of one that matters.
   */
  it("refuses a key AlphaFold 3 would refuse", () => {
    expect(refusal(server([{ proteinChain: {
      sequence: "ACDEFGHIK", count: 1, useStructureTemplates: true } }])))
      .toContain("useStructureTemplates");
  });

  // ...and the same for a template's own keys.
  it("refuses an unknown template key", () => {
    expect(refusal(open([{ protein: { id: "A", sequence: "ACDEFGHIK",
      templates: [{ mmcif: "data_T", maxDate: "2021-09-30" }] } }])))
      .toContain("maxDate");
  });

  /**
   * 🔴 AND `ligand` IS THE ONE KEY BOTH DIALECTS USE, meaning different bodies.
   * Which fields are legal comes from the body, not the entry key - keyed off
   * the key alone, this page's OWN archive stopped being readable.
   */
  it("reads a ligand in either dialect's spelling", () => {
    const fromServer = jobFromJson(server([
      { proteinChain: { sequence: "ACDEFGHIK", count: 1 } },
      { ligand: { ligand: "GOL", count: 1 } }]));
    const fromOpen = jobFromJson(open([
      { protein: { id: "A", sequence: "ACDEFGHIK" } },
      { ligand: { id: "B", ccdCodes: ["GOL"] } }]));
    expect(fromServer.entities[1].value).toBe("GOL");
    expect(fromOpen.entities[1].value).toBe("GOL");
  });
});

describe("reading the open-source dialect", () => {
  it("reads a chain and an integer seed", () => {
    const job = jobFromJson(open([{ protein: { id: "A", sequence: "ACDEFGHIK" } }]));
    expect(job.dialect).toBe("alphafold3");
    expect(job.seed).toBe(7);
    expect(job.entities[0].copies).toBe(1);
  });

  /**
   * 🔴 `id` IS HOW THAT DIALECT SAYS COPIES. `{"id": ["A", "B"]}` is a dimer,
   * and read as a name it folds a monomer - the same complex the file asked
   * for, minus a chain, with nothing anywhere saying so.
   */
  it("reads an id list as copies", () => {
    const job = jobFromJson(open([
      { protein: { id: ["A", "B", "C"], sequence: "ACDEFGHIK" } },
    ]));
    expect(job.entities[0].copies).toBe(3);
  });

  it("reads ccdCodes as the ligand", () => {
    const job = jobFromJson(open([
      { protein: { id: "A", sequence: "ACDEFGHIK" } },
      { ligand: { id: "B", ccdCodes: ["HEM"] } },
    ]));
    expect(job.entities[1].type).toBe("ligand");
    expect(job.entities[1].value).toBe("HEM");
  });

  /**
   * 🔴 POLYMERS BEFORE LIGANDS, WHATEVER ORDER THE FILE USED. expandEntities
   * numbers chains in list order and the featuriser puts ligand tokens after
   * every polymer token, so a file that lists its ligand first would hand the
   * ligand a chain index the protein still uses.
   */
  it("puts the ligand after the polymers whatever order the file used", () => {
    const job = jobFromJson(open([
      { ligand: { id: "B", ccdCodes: ["HEM"] } },
      { protein: { id: "A", sequence: "ACDEFGHIK" } },
    ]));
    expect(job.entities.map((entity) => entity.type)).toEqual(["protein", "ligand"]);
  });

  /**
   * 🔴 AN EMPTY unpairedMsa IS AN INSTRUCTION, NOT AN ABSENT FIELD. AlphaFold 3
   * reads `""` as "fold this chain with no alignment" and an absent field as
   * "go and search" - opposite jobs, and several minutes apart.
   */
  it("reads an empty alignment as single-sequence, and says so", () => {
    const job = jobFromJson(open([
      { protein: { id: "A", sequence: "ACDEFGHIK", unpairedMsa: "", pairedMsa: "" } },
    ]));
    expect(job.singleSequence).toBe(true);
    expect(job.notes.some((note) => note.includes("no alignment"))).toBe(true);
  });

  it("reads an inline template as an uploaded structure", () => {
    const job = jobFromJson(open([
      { protein: { id: "A", sequence: "ACDEFGHIK",
        templates: [{ mmcif: "data_TEST\n#\n" }] } },
    ]));
    expect(job.entities[0].template.kind).toBe("upload");
    expect(job.entities[0].template.text).toBe("data_TEST\n#\n");
  });
});

/**
 * 🔴 WHAT THIS PAGE CANNOT RUN IS REFUSED BY NAME. Every one of these parses
 * perfectly well as far as the sequence, so reading on would fold something
 * adjacent to the job that was asked for and report it as the job - which is
 * the failure mode this whole file exists to prevent, and the one the archive
 * hit twice this week from the writing side.
 */
describe("what it refuses, and what it names", () => {
  const cases = [
    ["bondedAtomPairs",
     server([{ proteinChain: { sequence: "ACDEFGHIK", count: 1 } }],
            { bondedAtomPairs: [[["A", 1, "CA"], ["B", 1, "CA"]]] })],
    ["userCCD", server([{ proteinChain: { sequence: "ACDEFGHIK", count: 1 } }],
                       { userCCD: "data_LIG" })],
    ["smiles", open([{ ligand: { id: "B", smiles: "CCO" } }])],
    ["ccdCodes", open([{ protein: { id: "A", sequence: "ACDEFGHIK" } },
                       { ligand: { id: "B", ccdCodes: ["ATP", "MG"] } }])],
    ["unpairedMsaPath", open([{ protein: { id: "A", sequence: "ACDEFGHIK",
                                           unpairedMsaPath: "/tmp/a.a3m" } }])],
    ["unpairedMsa", open([{ protein: { id: "A", sequence: "ACDEFGHIK",
                                       unpairedMsa: ">q\nACDEFGHIK\n" } }])],
    ["queryIndices", open([{ protein: { id: "A", sequence: "ACDEFGHIK",
      templates: [{ mmcif: "data_T", queryIndices: [0, 1] }] } }])],
    ["mmcifPath", open([{ protein: { id: "A", sequence: "ACDEFGHIK",
      templates: [{ mmcifPath: "/tmp/t.cif" }] } }])],
    ["not a chain kind", open([{ peptide: { id: "A", sequence: "ACDE" } }])],
    ["modified bases", open([{ dna: { id: "A", sequence: "ACGTACGT",
      modifications: [{ modificationType: "6MA", basePosition: 1 }] } }])],
    ["no `sequences`", JSON.stringify({ name: "j", modelSeeds: [1] })],
    ["not JSON", "ACDEFGHIK"],
  ];
  for (const [named, text] of cases) {
    it(`refuses ${named}, and names it`, () => {
      expect(refusal(text).includes(named)).toBe(true);
    });
  }

  /**
   * 🔴 AND THE PAGE'S OWN VALIDATOR HAS THE LAST WORD. A phosphoserine on a
   * tyrosine is wrong for the same reason whether it was typed or uploaded,
   * and a second set of messages here would drift from the ones a typist sees.
   */
  it("hands a bad job to the page's own validator", () => {
    expect(refusal(server([{ proteinChain: { sequence: "ACDEFGHIK", count: 1,
      modifications: [{ ptmType: "CCD_SEP", ptmPosition: 2 }] } }])))
      .toContain("SEP");
  });
});

describe("several of something this page folds one of", () => {
  // Loading the first is one of the things the file asked for; doing it
  // without a word is what makes a five-job file look like a one-job answer.
  it("says so rather than dropping the rest in silence", () => {
    const two = JSON.parse(server([{ proteinChain: { sequence: "ACDEFGHIK", count: 1 } }]));
    const job = jobFromJson(JSON.stringify([two[0], { ...two[0], name: "k" }]));
    expect(job.notes.some((note) => note.includes("2 jobs"))).toBe(true);
    expect(job.name).toBe("j");
  });

  it("says the same about seeds", () => {
    const job = jobFromJson(open([{ protein: { id: "A", sequence: "ACDEFGHIK" } }],
                                 { modelSeeds: [11, 12, 13] }));
    expect(job.seed).toBe(11);
    expect(job.notes.some((note) => note.includes("3 seeds"))).toBe(true);
  });
});

/**
 * 🔴 WRITE ONE AND READ IT BACK. This is the assertion the archive has never
 * had: that `job_request.json` describes the job it was written for. It is
 * also where the server dialect's limits show up honestly - it has one
 * template flag and no room for which structure - so what comes back is
 * asserted as what that dialect can say, not as what was typed.
 */
describe("the archive's own request, read back", () => {
  const entities = [
    { type: "protein", value: "ACSEFGHIK", copies: 2,
      modifications: [{ code: "SEP", position: 3 }],
      template: { kind: "pdb", source: "1QYS_A" } },
    { type: "dna", value: "ACGTACGT", copies: 1, modifications: [] },
    { type: "ligand", value: "GOL", copies: 1, modifications: [] },
  ];

  it("comes back as the same job", () => {
    const job = jobFromJson(jobRequestJson({ name: "af3_1", seed: 42, entities }));
    expect(job.name).toBe("af3_1");
    expect(job.seed).toBe(42);
    expect(job.entities.map((entity) => [entity.type, entity.value, entity.copies]))
      .toEqual([["protein", "ACSEFGHIK", 2], ["dna", "ACGTACGT", 1],
                ["ligand", "GOL", 1]]);
    expect(job.entities[0].modifications).toEqual([{ code: "SEP", position: 3 }]);
  });

  /**
   * 🔴 THE ONE THING THE SERVER DIALECT CANNOT CARRY, STATED. `1QYS_A` goes in
   * and "search for a template" comes back, because `useStructureTemplate` is
   * a boolean and the dialect has no field for which structure. That is a real
   * loss and it is written down here rather than discovered by someone whose
   * re-fold used a different template than the one they picked.
   */
  it("loses which template, and keeps that a template was asked for", () => {
    const job = jobFromJson(jobRequestJson({ name: "af3_1", seed: 0, entities }));
    expect(job.entities[0].template).toEqual({ kind: "search" });
  });
});
