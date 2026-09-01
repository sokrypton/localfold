/**
 * What to fold, as a list of entities rather than one string.
 *
 *     [{ type: "protein", value: "PIAQ...", copies: 2 },
 *      { type: "ligand",  value: "HEM",     copies: 1 }]
 *
 * This is AlphaFold Server's own model - an entity type, a value, and a number
 * of copies - and it replaces the colon-separated textarea because the colon
 * notation cannot express a ligand at all. Everything here is pure: the DOM
 * lives in entity-ui.js, and the fold pipeline downstream still receives the
 * colon-joined sequence it always did, from expandEntities.
 *
 * 🔴 COPIES ARE EXPANDED, NEVER PASSED DOWN. A protein with two copies is two
 * chains and a ligand with two copies is two ligand instances, because that is
 * what the featuriser counts: chainIdentity groups identical sequences into one
 * entity with two sym_ids, and the ligand block does the same for repeated
 * codes. Passing a count would mean teaching every layer below about copies to
 * arrive back at the same arrays.
 *
 * 🔴 AND POLYMERS COME BEFORE LIGANDS, whatever order they were entered in.
 * featuriseProtein appends the ligand tokens AFTER every polymer token and
 * numbers asym_id straight on from the last chain; a ligand entered first would
 * otherwise claim a chain index that the polymers still use.
 */
import { cleanSequence, sequenceProblem } from "./sequence.js";

/** The entity types this page can actually fold. */
export const ENTITY_TYPES = ["protein", "ligand"];

/** How they are labelled, in the order the menu offers them. */
export const ENTITY_LABELS = { protein: "Protein", ligand: "Ligand (CCD)" };

/**
 * 🔴 DNA AND RNA ARE ABSENT ON PURPOSE. AF3's restype alphabet is 31 wide and
 * has the nucleic acids in it, and DENSE is 24 atoms so a nucleotide would fit
 * - but reference-conformers.js holds the 21 amino acids and nothing else, so a
 * nucleotide has no reference conformer, no aatype and no C1' pseudo-beta. A
 * menu entry for them would produce a fold, and a plausible-looking one.
 */
export const UNSUPPORTED_TYPES = ["dna", "rna"];

/** More copies than this is far likelier to be a typo than a request. */
const MAX_COPIES = 20;

/** A fresh entity of the given type, as `+ Add entity` makes one. */
export function newEntity(type = "protein") {
  return { type, value: "", copies: 1 };
}

/**
 * What is wrong with one entity, or null.
 *
 * @param {{type: string, value: string, copies: number}} entity
 * @returns {string | null}
 */
export function entityProblem(entity) {
  if (!ENTITY_TYPES.includes(entity.type)) return `Unknown entity type ${entity.type}`;
  if (!Number.isInteger(entity.copies) || entity.copies < 1) {
    return "Copies must be a whole number, at least 1";
  }
  if (entity.copies > MAX_COPIES) return `At most ${MAX_COPIES} copies`;

  const value = entity.value.trim();
  if (value === "") {
    return entity.type === "ligand" ? "Enter a CCD code" : "Enter a protein sequence";
  }
  if (entity.type === "ligand") {
    // The same rule ccdUrl enforces, checked here so the message arrives while
    // the field is in front of the user rather than as a failed fetch later.
    if (!/^[A-Za-z0-9]{1,5}$/.test(value)) {
      return "A CCD code is 1-5 letters or digits, like HEM or ATP";
    }
    return null;
  }
  // 🔴 A COLON IN A PROTEIN ROW IS AN ERROR, NOT A SPLIT. One row is one chain;
  // the whole point of the list is that chains are rows. Silently splitting
  // would make copies ambiguous - two copies of "A:B" is four chains in one of
  // two different orders. Pasting colon-separated text still works, because
  // entitiesFromText splits it into rows BEFORE it reaches a row.
  if (value.includes(":")) {
    return "One sequence per entity - use Add entity for another chain";
  }
  return sequenceProblem(cleanSequence(value));
}

/**
 * What is wrong with the whole list, or null.
 *
 * @param {readonly {type: string, value: string, copies: number}[]} entities
 * @returns {string | null}
 */
export function entitiesProblem(entities) {
  if (entities.length === 0) return "Add an entity to fold";
  for (let index = 0; index < entities.length; index += 1) {
    const problem = entityProblem(entities[index]);
    if (problem === null) continue;
    return entities.length === 1 ? problem : `Entity ${index + 1}: ${problem}`;
  }
  // 🔴 A LIGAND CANNOT BE FOLDED ALONE. Every path below joins the polymer
  // chains into the sequence the model is built around - the MSA, the profile
  // and the templates are all indexed by it - and an empty one produces a batch
  // whose every shape is consistent and whose token count is the ligand's atoms.
  if (!entities.some((entity) => entity.type === "protein")) {
    return "Add at least one protein - a ligand cannot be folded on its own";
  }
  return null;
}

/**
 * The entity list as the fold pipeline wants it.
 *
 * @param {readonly {type: string, value: string, copies: number}[]} entities
 * @returns {{chains: string[], ligandCodes: string[], sequence: string}}
 *   `sequence` is the colon-joined chains, which is what every layer below
 *   already reads; `ligandCodes` are upper-cased, in order, one per instance.
 */
export function expandEntities(entities) {
  const problem = entitiesProblem(entities);
  if (problem !== null) throw new Error(problem);
  const chains = [];
  const ligandCodes = [];
  for (const entity of entities) {
    for (let copy = 0; copy < entity.copies; copy += 1) {
      if (entity.type === "protein") chains.push(cleanSequence(entity.value));
      else ligandCodes.push(entity.value.trim().toUpperCase());
    }
  }
  return { chains, ligandCodes, sequence: chains.join(":") };
}

/**
 * Entities from pasted text, so pasting a sequence still just works.
 *
 * Accepts a bare sequence, colon-separated chains, and multi-record FASTA -
 * which is what a user has in the clipboard. Each becomes its own protein row;
 * nothing here produces a ligand, since a bare CCD code is indistinguishable
 * from a very short peptide and guessing wrong is worse than not guessing.
 *
 * @param {string} text
 * @returns {{type: string, value: string, copies: number}[]}
 */
export function entitiesFromText(text) {
  const records = text.trim().startsWith(">")
    // FASTA: split on the record marker, drop each record's description line.
    ? text.split(/^>/m).slice(1).map((record) => record.split(/\r?\n/).slice(1).join(""))
    : [text];
  const sequences = [];
  for (const record of records) {
    for (const chain of record.split(":")) {
      const cleaned = cleanSequence(chain);
      if (cleaned !== "") sequences.push(cleaned);
    }
  }
  // 🔴 IDENTICAL CHAINS COLLAPSE INTO COPIES, which is what makes a pasted
  // homodimer read as one entity with two copies rather than two rows saying
  // the same thing. It is also what the featuriser will conclude anyway -
  // chainIdentity groups by sequence - so the list now shows what will be folded.
  const rows = [];
  for (const sequence of sequences) {
    const existing = rows.find((row) => row.value === sequence);
    if (existing === undefined) rows.push({ type: "protein", value: sequence, copies: 1 });
    else existing.copies += 1;
  }
  return rows;
}
