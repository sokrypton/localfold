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
 * The ligands and ions worth putting in a menu, as CCD codes.
 *
 * 🔴 A CONVENIENCE, NOT A LIMIT. Anything the PDB serves works - the fold
 * fetches the component by code at run time - so this list exists to spare
 * people looking up "the code for heme" and to show that ions are supported at
 * all. "Custom" stays the default and the box beside it still takes any code.
 *
 * 🔴 IONS ARE HALF THE POINT OF HAVING THE MENU. A zinc finger, a kinase's
 * magnesium, an EF-hand's calcium: these are the second thing anyone tries
 * after a protein, and they are the entries most likely to be typed wrong,
 * being one or two letters. They also did not work until the CCD reader learned
 * that a lone atom has no conformer - see src/af3/ccd-component.js.
 *
 * The set follows what AlphaFold Server offers. That list is not published in a
 * form worth citing, so this is the commonly reported one; it is one array, and
 * adding to it costs nothing.
 */
export const COMMON_LIGANDS = [
  { code: "ATP", name: "adenosine triphosphate" },
  { code: "ADP", name: "adenosine diphosphate" },
  { code: "AMP", name: "adenosine monophosphate" },
  { code: "GTP", name: "guanosine triphosphate" },
  { code: "GDP", name: "guanosine diphosphate" },
  { code: "NAD", name: "NAD, oxidised" },
  { code: "NAP", name: "NADP, oxidised" },
  { code: "NDP", name: "NADPH, reduced" },
  { code: "FAD", name: "flavin adenine dinucleotide" },
  { code: "HEM", name: "heme B" },
  { code: "HEC", name: "heme C" },
  { code: "CIT", name: "citrate" },
  { code: "PLM", name: "palmitate" },
  { code: "MYR", name: "myristate" },
  { code: "OLA", name: "oleate" },
  { code: "GOL", name: "glycerol" },
  { code: "SAM", name: "S-adenosylmethionine" },
  { code: "COA", name: "coenzyme A" },
  { code: "PLP", name: "pyridoxal phosphate" },
  { code: "NAG", name: "N-acetylglucosamine" },
];

export const COMMON_IONS = [
  { code: "MG", name: "magnesium" },
  { code: "ZN", name: "zinc" },
  { code: "CA", name: "calcium" },
  { code: "MN", name: "manganese" },
  { code: "FE", name: "iron (III)" },
  { code: "FE2", name: "iron (II)" },
  { code: "CU", name: "copper (II)" },
  { code: "CO", name: "cobalt (II)" },
  { code: "NI", name: "nickel" },
  { code: "K", name: "potassium" },
  { code: "NA", name: "sodium" },
  { code: "CL", name: "chloride" },
];

/** Every code the menu offers, for deciding whether a value is one of them. */
export const MENU_CODES = new Set(
  [...COMMON_LIGANDS, ...COMMON_IONS].map((entry) => entry.code));

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

/**
 * The modified residues worth putting in a menu, as CCD codes with the residue
 * each one modifies.
 *
 * 🔴 A CONVENIENCE AND NOT A LIMIT, like the ligand menu: anything the PDB
 * serves works, because the fold fetches the component by code. `parent` is
 * what the menu uses to offer the ones that fit the residue actually at that
 * position - a phosphoserine on a tyrosine is a typo, and the position is the
 * part people get wrong.
 *
 * 🔴 MSE IS NOT IN HERE. AF3 folds selenomethionine into methionine's alphabet
 * slot and leaves it one token, where every entry below becomes one token per
 * atom. It is a different thing wearing the same name, and offering it beside
 * these would promise something this path does not do.
 */
export const COMMON_MODIFICATIONS = [
  { code: "SEP", parent: "S", name: "phosphoserine" },
  { code: "TPO", parent: "T", name: "phosphothreonine" },
  { code: "PTR", parent: "Y", name: "phosphotyrosine" },
  { code: "HYP", parent: "P", name: "hydroxyproline" },
  { code: "MLY", parent: "K", name: "N-methyllysine" },
  { code: "M3L", parent: "K", name: "N-trimethyllysine" },
  { code: "ALY", parent: "K", name: "N-acetyllysine" },
  { code: "KCX", parent: "K", name: "carboxylysine" },
  { code: "CSO", parent: "C", name: "S-hydroxycysteine" },
  { code: "CME", parent: "C", name: "S,S-(2-hydroxyethyl)thiocysteine" },
  { code: "OCS", parent: "C", name: "cysteinesulfonic acid" },
  { code: "SNC", parent: "C", name: "S-nitrosocysteine" },
  { code: "NEP", parent: "H", name: "N1-phosphohistidine" },
  { code: "AGM", parent: "R", name: "methylarginine" },
  { code: "PCA", parent: "E", name: "pyroglutamate" },
];

/** A fresh entity of the given type, as `+ Add entity` makes one. */
export function newEntity(type = "protein") {
  return { type, value: "", copies: 1, modifications: [] };
}

/**
 * What is wrong with one modified residue on one entity, or null.
 *
 * @param {{code: string, position: number}} modification
 * @param {string} sequence the cleaned residue letters this entity holds
 */
export function modificationProblem(modification, sequence) {
  const code = (modification.code ?? "").trim().toUpperCase();
  if (code === "") return "Choose a modification";
  if (!/^[A-Z0-9]{1,5}$/.test(code)) {
    return "A CCD code is 1-5 letters or digits, like SEP or PTR";
  }
  if (code === "MSE") {
    // Better said here than discovered as a wrong token count later.
    return "MSE is not supported yet: AF3 treats it as methionine rather than "
      + "as a modified residue";
  }
  if (!Number.isInteger(modification.position) || modification.position < 1) {
    return "A position is a whole number, counting from 1";
  }
  if (modification.position > sequence.length) {
    return `Position ${modification.position} is past the end of a `
      + `${sequence.length}-residue sequence`;
  }
  const known = COMMON_MODIFICATIONS.find((entry) => entry.code === code);
  const parent = sequence[modification.position - 1];
  if (known !== undefined && parent !== known.parent) {
    return `${code} modifies ${known.parent}, but position ${modification.position}`
      + ` is ${parent}`;
  }
  return null;
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
  const cleaned = cleanSequence(value);
  const sequenceFault = sequenceProblem(cleaned);
  if (sequenceFault !== null) return sequenceFault;
  // ...the modifications last, because every one of their messages talks about
  // a position in a sequence that has to be valid first.
  const seen = new Set();
  for (const modification of entity.modifications ?? []) {
    const fault = modificationProblem(modification, cleaned);
    if (fault !== null) return fault;
    if (seen.has(modification.position)) {
      return `Two modifications on residue ${modification.position}`;
    }
    seen.add(modification.position);
  }
  return null;
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
  // 🔴 A LIGAND ON ITS OWN IS A FOLD, and this used to refuse one. AF3 accepts
  // a ligand-only job and so does the featuriser: the chain identity helpers
  // reject a zero-length sequence, rightly, but they are only read inside the
  // polymer loop, which does not run when there are no residues. What needed
  // fixing was three places that assumed a polymer - the alignment guard, the
  // PAE's size, and the superposition's CA atoms - not the entity list.
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
  // 🔴 PER CHAIN, NOT PER ENTITY, BECAUSE COPIES ARE EXPANDED. Two copies of a
  // phosphorylated chain are two chains each carrying the modification, and the
  // featuriser indexes them by the chain number it will actually see - which is
  // the position in `chains`, not the position in `entities`.
  const modifications = [];
  for (const entity of entities) {
    for (let copy = 0; copy < entity.copies; copy += 1) {
      if (entity.type === "protein") {
        for (const modification of entity.modifications ?? []) {
          modifications.push({
            chain: chains.length,
            position: modification.position,
            code: modification.code.trim().toUpperCase(),
          });
        }
        chains.push(cleanSequence(entity.value));
      } else ligandCodes.push(entity.value.trim().toUpperCase());
    }
  }
  return { chains, ligandCodes, modifications, sequence: chains.join(":") };
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
