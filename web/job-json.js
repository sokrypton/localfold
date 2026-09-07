/**
 * The job, as AlphaFold 3's own JSON - written and, now, read.
 *
 * WHY BOTH HALVES LIVE IN ONE FILE. The archive has written a
 * `*_job_request.json` since it existed, and the README tells the reader to
 * drop the .zip back on the page "to fold again" - which restored the
 * ALIGNMENTS and nothing else: the sequence, the ligands, the modifications
 * and the seed all had to be retyped. A format written in one file and read in
 * another drifts, and the two bugs this week were both a field that reached the
 * fold and not the file (the templates, then the modifications). Writer and
 * reader beside each other is what makes a round-trip gate possible at all.
 *
 * 🔴 "THE AlphaFold 3 FORMAT" IS TWO FORMATS, AND CONFLATING THEM IS THE TRAP.
 *
 *   - the SERVER dialect, `dialect: "alphafoldserver"`, `version: 3`: seeds are
 *     STRINGS, a chain is `proteinChain: {sequence, count}`, and a template is
 *     the single flag `useStructureTemplate: true` with no way to say WHICH.
 *     This is what the archive writes, because the archive is the server's file
 *     for file - see web/fold-archive.js.
 *   - the OPEN-SOURCE dialect, no `dialect` key, `version` 1-3: seeds are
 *     INTEGERS, a chain is `protein: {id: "A", sequence}` where `id` may be a
 *     LIST meaning copies, and it can carry the alignment and the template
 *     structure inline.
 *
 * Both are read. Only the server one is written, because that is the one the
 * archive's own justification rests on.
 *
 * 🔴 AND WHAT THIS PAGE CANNOT RUN IS REFUSED BY NAME, NEVER IGNORED. A JSON
 * carrying `bondedAtomPairs`, a SMILES ligand or a user CCD describes a fold
 * this page does not do, and reading it as far as the sequence would fold
 * something adjacent to what was asked for and say nothing - the same failure
 * as a modified residue that reaches the model and not the request file. The
 * error names the field, because "unsupported job" sends the reader looking.
 */
import { NUCLEIC_TYPES, entitiesProblem } from "./entities.js";

/**
 * The request that produced this fold, in the server's own dialect.
 *
 * 🔴 COPIES STAY A COUNT. `expandEntities` turns two copies into two chains
 * because that is what the model is given, but the server's request says
 * `count: 2` on one entry - and a request that listed the same sequence twice
 * would come back from the server as a different job than the one that ran.
 */
export function jobRequestJson({ name, seed, entities }) {
  const sequences = [];
  for (const entity of entities ?? []) {
    const value = (entity.value ?? "").trim();
    if (value === "") continue;
    const count = Math.max(1, Number(entity.copies) || 1);
    if (entity.type === "protein") {
      // 🔴 A MODIFIED RESIDUE IS PART OF THE JOB, NOT A RENDERING OF IT. The
      // request is what a reader hands back to reproduce the fold, and one
      // that lists the parent sequence alone describes a DIFFERENT job - the
      // same mistake as dropping the templates line. The server's dialect
      // names them `ptmType`/`ptmPosition`, with the CCD code prefixed, so
      // they are written the way the server would read them back.
      const modifications = (entity.modifications ?? [])
        .filter((modification) => (modification.code ?? "").trim() !== "")
        .map((modification) => ({
          ptmType: `CCD_${modification.code.trim().toUpperCase()}`,
          ptmPosition: modification.position,
        }));
      sequences.push({ proteinChain: { sequence: value, count,
        // ...absent rather than empty, because `modifications: []` is a claim
        // that the chain was checked and carries none, and every unmodified
        // fold this page has ever written says nothing at all.
        ...(modifications.length === 0 ? {} : { modifications }),
        useStructureTemplate: (entity.template?.kind ?? "none") !== "none" } });
    } else if (entity.type === "dna" || entity.type === "rna") {
      sequences.push({ [`${entity.type}Sequence`]: { sequence: value, count } });
    } else {
      sequences.push({ ligand: { ligand: value.toUpperCase(), count } });
    }
  }
  return `${JSON.stringify([{
    name,
    // A string, as the server writes it, and an array because a job may carry
    // several seeds. This page folds one at a time.
    modelSeeds: [String(seed ?? 0)],
    sequences,
    dialect: "alphafoldserver",
    version: 3,
  }], null, 2)}\n`;
}

/** A refusal that names the field, so the reader knows what to remove. */
function refuse(message) {
  throw new Error(message);
}

/** `CCD_SEP` and `SEP` both mean SEP; anything else is not a CCD code. */
function ptmCode(type, where) {
  const raw = String(type ?? "").trim().toUpperCase();
  const code = raw.startsWith("CCD_") ? raw.slice(4) : raw;
  if (!/^[A-Z0-9]{1,5}$/.test(code)) {
    refuse(`${where}: "${type}" is not a CCD code`);
  }
  return code;
}

/**
 * 🔴 A SEQUENCE ENTRY HAS EXACTLY ONE KEY, AND AN UNKNOWN ONE IS A REFUSAL.
 * `{"protein": ...}` and `{"proteinChain": ...}` are the two dialects' names
 * for the same thing; a file using a key this page has never heard of is a
 * newer format or a different program, and skipping the entry would fold a
 * subset of the complex that was asked for - quietly, since the remaining
 * chains fold perfectly well on their own.
 */
const ENTRY_TYPES = {
  proteinChain: { type: "protein", dialect: "alphafoldserver" },
  dnaSequence: { type: "dna", dialect: "alphafoldserver" },
  rnaSequence: { type: "rna", dialect: "alphafoldserver" },
  // 🔴 AN ION IS ITS OWN ENTRY IN THE SERVER'S DIALECT. `{"ion": {"ion": "MG"}}`
  // is how AlphaFold Server spells a magnesium - `Ligand.from_alphafoldserver_dict`
  // takes `ligand` or `ion` and treats them alike - and reading only `ligand`
  // refused every real server job with a metal in it, which is half of what the
  // ligand menu here exists for. The example corpus could not catch this: all
  // fourteen of those files are the OTHER dialect.
  ion: { type: "ligand", dialect: "alphafoldserver" },
  // 🔴 `ligand` IS THE ONE KEY BOTH DIALECTS USE, and they mean different
  // bodies by it: `{"ligand": "GOL", "count": 1}` on the server against
  // `{"id": "B", "ccdCodes": ["GOL"]}` in the open-source one. Which set of
  // fields is legal therefore comes from the BODY, not from the entry key -
  // resolved in readEntry.
  ligand: { type: "ligand", dialect: null },
  protein: { type: "protein", dialect: "alphafold3" },
  dna: { type: "dna", dialect: "alphafold3" },
  rna: { type: "rna", dialect: "alphafold3" },
};

/**
 * The keys each kind of entry may carry, copied from AlphaFold 3's own
 * `folding_input.py`.
 *
 * 🔴 AN UNKNOWN KEY IS REFUSED, BECAUSE ALPHAFOLD 3 REFUSES IT. Every one of
 * these classes calls `_validate_keys` and raises on anything else, so being
 * lenient here is not being generous - it is folding a job the reference
 * implementation would not have run, from a file whose extra key was probably a
 * typo for one that matters. `glycans` and `maxTemplateDate` are in the
 * server's list AND refused by name below, exactly as upstream does it.
 */
const ALLOWED_KEYS = {
  alphafoldserver: {
    protein: ["sequence", "glycans", "modifications", "count",
              "maxTemplateDate", "useStructureTemplate"],
    dna: ["sequence", "modifications", "count"],
    rna: ["sequence", "modifications", "count"],
    ligand: ["ligand", "ion", "count"],
  },
  alphafold3: {
    protein: ["id", "sequence", "modifications", "description", "unpairedMsa",
              "unpairedMsaPath", "pairedMsa", "pairedMsaPath", "templates"],
    dna: ["id", "sequence", "modifications", "description"],
    rna: ["id", "sequence", "modifications", "description", "unpairedMsa",
          "unpairedMsaPath"],
    ligand: ["id", "ccdCodes", "smiles", "description"],
  },
};

/** The keys a template entry may carry, likewise from folding_input.py. */
const TEMPLATE_KEYS = ["mmcif", "mmcifPath", "queryIndices", "templateIndices"];

function checkKeys(body, allowed, where) {
  const unknown = Object.keys(body).filter((key) => !allowed.includes(key));
  if (unknown.length > 0) {
    refuse(`${where}: ${unknown.join(", ")} is not a field of this entry`
      + ` - AlphaFold 3 takes ${allowed.join(", ")}`);
  }
}

/** How many copies an entry asks for: a `count`, or the length of an id list. */
function copiesOf(body, where) {
  if (body.count !== undefined) {
    const count = Number(body.count);
    if (!Number.isInteger(count) || count < 1) refuse(`${where}: count ${body.count}`);
    return count;
  }
  // 🔴 `id` IS COPIES IN THE OPEN-SOURCE DIALECT. `{"id": ["A", "B"]}` is two
  // copies of one chain, which is the only place that dialect says how many -
  // read as a name it would fold a monomer where a dimer was asked for.
  if (Array.isArray(body.id)) return Math.max(1, body.id.length);
  return 1;
}

/** The alignment fields, which this page can only answer one way. */
function readAlignment(body, where, state) {
  for (const field of ["unpairedMsaPath", "pairedMsaPath"]) {
    if (body[field] !== undefined && body[field] !== null) {
      refuse(`${where}: ${field} points at a file beside the JSON, which a page`
        + " cannot read - paste the alignment or use the upload box");
    }
  }
  let sawEmpty = false;
  for (const field of ["unpairedMsa", "pairedMsa"]) {
    const value = body[field];
    if (value === undefined || value === null) continue;
    // 🔴 AN EMPTY STRING IS NOT AN ABSENT FIELD. AlphaFold 3 reads `""` as "run
    // this chain with no alignment" and an absent field as "search for one" -
    // opposite instructions, and the difference between a single-sequence fold
    // and a several-minute search. It is the same absent-is-not-zero rule the
    // archive's B-factor column follows.
    if (String(value).trim() === "") { sawEmpty = true; continue; }
    refuse(`${where}: ${field} carries an alignment inline, which this page`
      + " cannot attach yet - drop the fold archive on the alignment box"
      + " instead, or delete the field to search afresh");
  }
  if (sawEmpty) state.singleSequence = true;
}

/**
 * The template an entry asks for, as an entity's `template`, or undefined.
 *
 * 🔴 THE TWO DIALECTS ASK DIFFERENT QUESTIONS. The server's flag says "search
 * for one", which is exactly our `search` kind. The open-source one hands over
 * the mmCIF itself, which is our `upload` kind - but it also carries
 * `queryIndices`/`templateIndices`, an explicit residue-by-residue mapping that
 * this page does not take: it aligns the template itself. Honouring the
 * structure and dropping the mapping folds the same template against a
 * different correspondence, so the mapping is refused rather than ignored.
 */
function readTemplates(body, where) {
  if (body.useStructureTemplate === true) return { kind: "search" };
  const templates = body.templates;
  if (templates === undefined || templates === null) return undefined;
  if (!Array.isArray(templates)) refuse(`${where}: templates is not a list`);
  if (templates.length === 0) return undefined;
  if (templates.length > 1) {
    refuse(`${where}: ${templates.length} templates, and this page takes one`
      + " per chain");
  }
  const [template] = templates;
  if (template.mmcifPath !== undefined && template.mmcifPath !== null) {
    refuse(`${where}: mmcifPath points at a file beside the JSON, which a page`
      + " cannot read - inline the mmCIF or pick a template on the row");
  }
  checkKeys(template, TEMPLATE_KEYS, `${where} template`);
  for (const field of ["queryIndices", "templateIndices"]) {
    if (template[field] !== undefined && template[field] !== null) {
      refuse(`${where}: ${field} sets the template's residue mapping, and this`
        + " page computes its own - the fold would use a different alignment"
        + " than the file asks for");
    }
  }
  const mmcif = template.mmcif;
  if (typeof mmcif !== "string" || mmcif.trim() === "") {
    refuse(`${where}: a template with no mmcif in it`);
  }
  return { kind: "upload", text: mmcif, filename: "template.cif", source: "" };
}

/** One `sequences` entry, as an entity. */
function readEntry(entry, index, state) {
  const keys = Object.keys(entry ?? {});
  if (keys.length !== 1) {
    refuse(`sequences[${index}] has ${keys.length} keys, and an entry names one`
      + " kind of chain");
  }
  const [key] = keys;
  const entryType = ENTRY_TYPES[key];
  if (entryType === undefined) {
    refuse(`sequences[${index}]: "${key}" is not a chain kind this page reads`);
  }
  const { type } = entryType;
  const body = entry[key] ?? {};
  const where = `sequences[${index}] (${key})`;
  const dialect = entryType.dialect
    ?? (body.ligand !== undefined || body.ion !== undefined
        ? "alphafoldserver" : "alphafold3");
  checkKeys(body, ALLOWED_KEYS[dialect][type], where);
  // 🔴 REFUSED BY NAME, THE WAY UPSTREAM REFUSES THEM. Both are in the server's
  // allowed set and both raise in folding_input.py: a glycan is chemistry this
  // page does not build, and a template date CHANGES WHICH TEMPLATE IS FOUND,
  // so honouring the sequence and dropping the date folds a different job.
  if (body.glycans !== undefined && body.glycans !== null) {
    refuse(`${where}: \`glycans\` is not supported in this dialect, upstream`
      + " included");
  }
  if (body.maxTemplateDate !== undefined && body.maxTemplateDate !== null) {
    refuse(`${where}: \`maxTemplateDate\` chooses which template is found, and`
      + " this page has no such control");
  }
  const copies = copiesOf(body, where);

  if (type === "ligand") {
    if (body.smiles !== undefined && body.smiles !== null) {
      refuse(`${where}: \`smiles\` names a ligand by structure, and this page`
        + " folds ligands by CCD code");
    }
    // 🔴 THE SERVER'S OWN SPELLING CARRIES A `CCD_` PREFIX, WHICH UPSTREAM
    // STRIPS. `Ligand.from_alphafoldserver_dict` does `removeprefix('CCD_')`,
    // so `"ligand": "CCD_ATP"` is ATP - kept whole it becomes a five-letter
    // code this page would go and fetch a component for, and not find.
    const named = body.ligand ?? body.ion;
    const codes = body.ccdCodes ?? (named === undefined ? [] : [named]);
    const list = Array.isArray(codes) ? codes : [codes];
    if (list.length === 0) refuse(`${where}: a ligand with no code`);
    // 🔴 SEVERAL CODES IN ONE ENTRY IS ONE CHAIN OF SEVERAL COMPONENTS, not
    // several ligands: AF3 bonds them into one entity. Splitting them into
    // separate rows folds the same atoms unbonded, which is a different
    // molecule wearing the same codes.
    if (list.length > 1) {
      refuse(`${where}: ccdCodes lists ${list.length} components as one bonded`
        + " chain, and this page folds one code per ligand");
    }
    const code = String(list[0]).trim().toUpperCase().replace(/^CCD_/, "");
    return { type: "ligand", value: code, copies, modifications: [] };
  }

  const sequence = body.sequence;
  if (typeof sequence !== "string" || sequence.trim() === "") {
    refuse(`${where}: no sequence`);
  }
  const modifications = [];
  for (const modification of body.modifications ?? []) {
    // The open-source dialect names a base modification differently from a
    // protein one, and this page refuses both on a nucleic chain anyway - so
    // the field is read either way and entitiesProblem gives the message.
    const code = ptmCode(modification.ptmType ?? modification.modificationType,
                         where);
    const position = Number(modification.ptmPosition ?? modification.basePosition);
    if (!Number.isInteger(position)) {
      refuse(`${where}: modification ${code} has no whole-number position`);
    }
    modifications.push({ code, position });
  }
  if (NUCLEIC_TYPES.includes(type) && modifications.length > 0) {
    refuse(`${where}: modified bases are not supported yet`);
  }
  readAlignment(body, where, state);
  const template = type === "protein" ? readTemplates(body, where) : undefined;
  return { type, value: sequence.trim().toUpperCase(), copies, modifications,
           ...(template === undefined ? {} : { template }) };
}

/**
 * 🔴 THE TWO DIALECTS NUMBER THEIR VERSIONS SEPARATELY, AND UPSTREAM AND THE
 * SERVER DISAGREE ABOUT ONE OF THEM. `folding_input.py` has
 * `ALPHAFOLDSERVER_JSON_VERSION = 1` and RAISES on anything else - while the
 * real AlphaFold Server writes `"version": 3` in the archive it hands you, as
 * `tools/fixtures/fold_2026_09_01_10_17.zip` does. So the reference archive is
 * refused by the reference parser, which is upstream's own split and not one
 * this page invented. Both are read here, because both are files people have.
 * The open-source dialect's own `JSON_VERSIONS` is (1, 2, 3, 4).
 *
 * 🔴 AND A VERSION WE HAVE NOT SEEN IS REFUSED RATHER THAN ASSUMED. A later
 * one may give a field we already read a different meaning, which is precisely
 * the failure that cannot be noticed from the outside.
 */
const KNOWN_VERSIONS = { alphafoldserver: [1, 3], alphafold3: [1, 2, 3, 4] };

function checkVersion(job, dialect) {
  const hasDialect = job.dialect !== undefined && job.dialect !== null;
  const hasVersion = job.version !== undefined && job.version !== null;
  // Upstream's rule exactly: both, or neither (in which case it is the
  // server's dialect at its own version 1).
  if (hasDialect !== hasVersion) {
    refuse("a job carries both `dialect` and `version` or neither, and this"
      + ` one has only \`${hasDialect ? "dialect" : "version"}\``);
  }
  if (!hasVersion) return;
  const known = KNOWN_VERSIONS[dialect];
  if (!known.includes(Number(job.version))) {
    refuse(`version ${job.version} of the ${dialect} dialect is not one this`
      + ` page reads (${known.join(", ")})`);
  }
}

/**
 * An AlphaFold 3 job JSON, in either dialect, as this page's entity list.
 *
 * @returns {{name: string|undefined, seed: number|undefined,
 *            entities: object[], dialect: string, singleSequence: boolean,
 *            notes: string[]}}
 *   `notes` are things the file said that this page answered differently and
 *   the reader should know about - never things it ignored.
 */
export function jobFromJson(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`that is not JSON: ${error.message}`);
  }
  // Both dialects appear as a bare object and as a list of jobs in the wild.
  const jobs = Array.isArray(parsed) ? parsed : [parsed];
  if (jobs.length === 0) refuse("that file holds no jobs");
  const state = { singleSequence: false };
  const notes = [];
  if (jobs.length > 1) {
    // 🔴 SAID, NOT SILENTLY DROPPED. Folding the first of several jobs is one
    // of the things that was asked for; doing it without a word is what turns
    // a five-job file into a one-structure answer nobody questions.
    notes.push(`${jobs.length} jobs in the file; loaded the first`);
  }
  const job = jobs[0] ?? {};
  for (const field of ["bondedAtomPairs", "userCCD", "userCCDPath"]) {
    if (job[field] !== undefined && job[field] !== null) {
      refuse(`${field} describes chemistry this page does not build - remove it`
        + " to fold the rest");
    }
  }
  // 🔴 ABSENT MEANS THE SERVER'S, WHICH IS UPSTREAM'S RULE AND NOT AN OBVIOUS
  // ONE: a job with neither `dialect` nor `version` is read by folding_input.py
  // as `alphafoldserver` at version 1. Defaulting the other way put such a file
  // under the open-source version table, where its version would be checked
  // against the wrong list.
  if (job.dialect !== undefined && job.dialect !== null
      && job.dialect !== "alphafoldserver" && job.dialect !== "alphafold3") {
    refuse(`dialect "${job.dialect}" is not one this page reads`
      + " (alphafoldserver, alphafold3)");
  }
  const dialect = job.dialect === "alphafold3" ? "alphafold3" : "alphafoldserver";
  checkVersion(job, dialect);
  const entries = job.sequences;
  if (!Array.isArray(entries) || entries.length === 0) {
    refuse("no `sequences` in that job");
  }
  const entities = entries.map((entry, index) => readEntry(entry, index, state));
  // 🔴 POLYMERS BEFORE LIGANDS, because expandEntities numbers chains in list
  // order and the featuriser appends ligand tokens after every polymer token.
  // A file listing its ligand first is perfectly legal and would otherwise
  // claim a chain index the polymers still use.
  entities.sort((left, right) =>
    Number(left.type === "ligand") - Number(right.type === "ligand"));

  const seeds = job.modelSeeds ?? [];
  const list = Array.isArray(seeds) ? seeds : [seeds];
  if (list.length > 1) {
    notes.push(`${list.length} seeds in the file; folding the first`);
  }
  let seed;
  if (list.length > 0) {
    const first = Number(list[0]);
    if (!Number.isFinite(first) || first < 0) refuse(`seed ${list[0]}`);
    seed = Math.floor(first);
  }
  if (state.singleSequence) {
    notes.push("the file asks for no alignment, so the MSA dial is set to none");
  }

  // 🔴 CHECKED WITH THE PAGE'S OWN VALIDATOR, NOT A SECOND ONE. A phosphoserine
  // on a tyrosine, a position past the end, twenty-one copies: every one of
  // these already has a message written for the row that would show it, and a
  // parallel set here would drift from the ones a typist sees.
  const problem = entitiesProblem(entities);
  if (problem !== null) refuse(problem);

  return { name: typeof job.name === "string" ? job.name : undefined,
           seed, entities, dialect, singleSequence: state.singleSequence, notes };
}
