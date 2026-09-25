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
import { parseSmiles } from "../src/chem/smiles.js";
import { ligandName } from "../src/chem/component.js";
import { cleanSequence, nucleicProblem, sequenceProblem } from "./sequence.js";

/** The entity types this page can actually fold. */
/**
 * 🔴 AND "smiles" IS A TYPE RATHER THAN A FLAG ON "ligand", BECAUSE THE VALUE
 * IS TREATED DIFFERENTLY THE MOMENT IT IS READ. A CCD code is upper-cased -
 * `hem` and `HEM` are the same component - and upper-casing a SMILES changes
 * the molecule: `c1ccccc1` is benzene and `C1CCCCC1` is cyclohexane. There is
 * no way to sniff which a row holds, since `C` is a valid SMILES and `CCO`
 * looks like a three-letter code, so the row says.
 */
export const ENTITY_TYPES = ["protein", "dna", "rna", "ligand", "smiles",
                            "contact"];

/**
 * The row types that become a CHAIN. A `contact` does not - it describes a bond
 * between two chains that already exist.
 *
 * 🔴 ASKED AS ONE QUESTION, BECAUSE THE ALTERNATIVE IS A LIST PER CALL SITE.
 * `expandEntities` is `if polymer ... else if smiles ... else LIGAND`, so a row
 * type it has not heard of becomes a ligand - silently, with a chain in the
 * fold that nobody asked for. That is the same shape as `setChains` deleting a
 * SMILES row by keeping only `type === "ligand"`, which this file already
 * records. Every place that means "is this a chain" asks here.
 */
export const CHAIN_TYPES = ["protein", "dna", "rna", "ligand", "smiles"];

/** Does this row become a chain in the fold? */
export const isChainEntity = (entity) => CHAIN_TYPES.includes(entity.type);

/**
 * A contact row's text, parsed - or null if it is not one.
 *
 * The spec is written the way a reader would say it out loud:
 *
 *     A12:SG - B1:C25      a covalent bond between two named atoms
 *     A12 - B30            the same, letting each side default its atom
 *
 * 🔴 CHAINS ARE LETTERS, AS THEY ARE EVERYWHERE THE READER LOOKS - the viewer,
 * the PDB, and AlphaFold 3's own `bondedAtomPairs`. They are resolved against
 * the chain ORDER at expand time, so a contact written before the chain it
 * names is still valid; what it cannot survive is the chains being reordered
 * underneath it, which is why the row sits with them rather than in a second
 * list somewhere else.
 */
export function parseContact(value) {
  const text = String(value ?? "").trim();
  if (text === "") return null;
  const sides = text.split("-");
  if (sides.length !== 2) return null;
  const end = (side) => {
    const match = /^\s*([A-Za-z]+)\s*(\d+)\s*(?::\s*([A-Za-z0-9']+)\s*)?$/.exec(side);
    if (match === null) return null;
    return { chain: match[1].toUpperCase(), residue: Number(match[2]),
             atom: match[3] === undefined ? null : match[3].toUpperCase() };
  };
  const from = end(sides[0]);
  const to = end(sides[1]);
  if (from === null || to === null) return null;
  return { from, to };
}

/** How they are labelled, in the order the menu offers them. */
export const ENTITY_LABELS = {
  protein: "Protein", dna: "DNA", rna: "RNA", ligand: "Ligand (CCD)",
  smiles: "Ligand (SMILES)", contact: "Contact (bond)",
};

/**
 * The largest SMILES ligand this page will build a conformer for.
 *
 * 🔴 A CAP, BECAUSE THE EMBEDDING IS O(N^3) AND THE PAGE IS SINGLE-THREADED.
 * Triangle smoothing is the cost and it is cubic in the atom count: 60 atoms
 * is 216,000 relaxations and a few milliseconds, 500 would be 125 million and
 * would lock the tab. A ligand that large is not a ligand.
 */
export const MAX_SMILES_ATOMS = 150;

/** The polymer types, which are the ones with a residue sequence. */
export const POLYMER_TYPES = ["protein", "dna", "rna"];

/** The nucleic types, whose letters are bases rather than amino acids. */
export const NUCLEIC_TYPES = ["dna", "rna"];

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
 * that a lone atom has no conformer - see src/af3/featurise/ccd-component.js.
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
 * 🔴 NOTHING IS UNSUPPORTED HERE ANY MORE, AND THIS STAYS EMPTY RATHER THAN
 * GOING AWAY. DNA and RNA were listed here for as long as they had no reference
 * conformers; they now have their own baked table and a featuriser checked
 * against AF3 array by array. The export remains because callers ask it what to
 * refuse, and the next type to arrive will want the same answer.
 */
export const UNSUPPORTED_TYPES = [];

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
 * The template sources a protein chain can be given, in menu order.
 *
 * 🔴 ONE FIELD FOR THREE DATABASES WAS A GUESS, AND A GUESS IS THE WRONG SHAPE
 * FOR THIS. The earlier version took `1abc`, `1abc_A` or an accession in a
 * single box and decided which server to ask by counting characters - four is
 * the PDB, anything else AlphaFold DB. It reads well and it is right most of
 * the time, but "which database" is a question the user can always answer and
 * the page could only estimate, and the two failures it produces are both
 * silent: an accession that happens to be four characters goes to the wrong
 * server, and a typo in a PDB id becomes an AlphaFold DB lookup whose 404 names
 * a database the user never chose. Asking outright costs one dropdown.
 *
 * It also makes room for the two sources that had no way to be named at all: a
 * file the user has on disk, and the hits the MSA search already found - which
 * were a checkbox sitting beside a text field that it silently overrode.
 */
/**
 * Whether one chain gets an alignment, in menu order.
 *
 * 🔴 WRITTEN LIKE TEMPLATE_KINDS BECAUSE IT IS THE SAME KIND OF CHOICE: one
 * setting, on one chain, that changes what is folded. Two words each and the
 * same "No ..." for the off case, so the two menus in the popup read as a
 * pair rather than as one control and one sentence - the first version said
 * "As the job does (search)" and "None - fold from the query alone", which
 * explained itself at the cost of looking like something else.
 */
export const MSA_KINDS = [
  ["search", "The job's alignment"],
  ["none", "No alignment"],
];

export const TEMPLATE_KINDS = [
  ["none", "No template"],
  ["pdb", "PDB entry"],
  ["afdb", "AlphaFold DB"],
  ["search", "From the MSA search"],
  ["upload", "Upload a structure"],
];

/** Which source an entity's template asks for; "none" when it asks for none. */
export function templateKind(template) {
  const kind = template?.kind;
  return TEMPLATE_KINDS.some(([name]) => name === kind) ? kind : "none";
}

/**
 * Whether a template is filled in enough to be worth fetching.
 *
 * 🔴 A KIND WITH NOTHING UNDER IT IS NOT AN ERROR, IT IS AN UNFINISHED ROW.
 * Picking "PDB entry" and not yet typing an id is the state the dropdown is in
 * for as long as it takes to type one, so it cannot be a problem - but it is
 * not a template either, and folding as though it were would put an empty slot
 * where the user is expecting a structure.
 */
export function templateAsked(template) {
  const kind = templateKind(template);
  if (kind === "none") return false;
  if (kind === "search") return true;
  if (kind === "upload") return (template.text ?? "") !== "";
  return (template.source ?? "").trim() !== "";
}

/**
 * What is wrong with an entity's template source, or null.
 *
 * 🔴 A TEMPLATE ONLY MEANS ANYTHING ON A POLYMER. AF3's own Template is
 * documented as one protein chain; a ligand has no residues to map onto.
 */
export function templateProblem(entity) {
  const template = entity.template;
  const kind = templateKind(template);
  if (kind === "none") return null;
  if (entity.type !== "protein") return "Only a protein chain can take a template";
  if (kind === "search" || kind === "upload") return null;
  const source = (template.source ?? "").trim();
  if (source === "") return null;
  if (kind === "pdb" && !/^[A-Za-z0-9]{4}([_:][A-Za-z0-9]+)?$/.test(source)) {
    return `${source} is not a PDB entry (1abc, 1abc_A)`;
  }
  if (kind === "afdb" && !/^[A-Za-z0-9]{6,10}([_:][A-Za-z0-9]+)?$/.test(source)) {
    return `${source} is not a UniProt accession (P00533)`;
  }
  return null;
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
  if (entity.type === "contact") {
    const parsed = parseContact(entity.value);
    if (parsed === null) {
      return "A contact is two residues, as A12:SG - B1:C25 (the atoms optional)";
    }
    if (parsed.from.chain === parsed.to.chain
        && parsed.from.residue === parsed.to.residue) {
      return "A contact joins two different residues";
    }
    return null;
  }
  if (!Number.isInteger(entity.copies) || entity.copies < 1) {
    return "Copies must be a whole number, at least 1";
  }
  if (entity.copies > MAX_COPIES) return `At most ${MAX_COPIES} copies`;

  const value = entity.value.trim();
  if (value === "") {
    if (entity.type === "ligand") return "Enter a CCD code";
    if (entity.type === "smiles") return "Enter a SMILES string";
    return entity.type === "protein"
      ? "Enter a protein sequence" : `Enter a ${entity.type.toUpperCase()} sequence`;
  }
  if (entity.type === "ligand") {
    // The same rule ccdUrl enforces, checked here so the message arrives while
    // the field is in front of the user rather than as a failed fetch later.
    if (!/^[A-Za-z0-9]{1,5}$/.test(value)) {
      return "A CCD code is 1-5 letters or digits, like HEM or ATP";
    }
    return null;
  }
  if (entity.type === "smiles") {
    // 🔴 PARSED HERE, WITH THE PARSER'S OWN MESSAGE. Every refusal in
    // src/chem/smiles.js names what it could not read and where - "ring
    // closure 1 never closed", "`*` (any atom) has no element" - and those
    // arrive while the field is in front of the reader rather than as a fold
    // that dies two minutes in. It also means there is exactly one definition
    // of what this page accepts, which is whatever it can actually fold.
    try {
      const graph = parseSmiles(value);
      if (graph.atoms.length > MAX_SMILES_ATOMS) {
        return `That is ${graph.atoms.length} heavy atoms; at most`
          + ` ${MAX_SMILES_ATOMS} here`;
      }
    } catch (error) {
      return error.message;
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
  // 🔴 THE SAME LETTERS MEAN DIFFERENT THINGS PER ROW. `ACGT` is a valid
  // protein and a valid DNA chain, and nothing about the text says which - the
  // row's type does, so the check follows it rather than sniffing the letters.
  const sequenceFault = NUCLEIC_TYPES.includes(entity.type)
    ? nucleicProblem(cleaned, entity.type)
    : sequenceProblem(cleaned);
  if (sequenceFault !== null) return sequenceFault;
  // 🔴 A MODIFICATION ON A NUCLEIC CHAIN IS REFUSED RATHER THAN IGNORED. AF3
  // takes modified bases and this featuriser does not: its modified-residue
  // path resolves the parent through the amino-acid table, so a modified base
  // would be featurised as a modified amino acid and fold to something. The
  // popup does not offer them, so this catches a restored or pasted list.
  if (NUCLEIC_TYPES.includes(entity.type) && (entity.modifications ?? []).length > 0) {
    return `Modified bases are not supported yet on a ${entity.type.toUpperCase()} chain`;
  }
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
    const problem = entityProblem(entities[index]) ?? templateProblem(entities[index]);
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
 * @returns {{chains: string[], ligandCodes: string[], templates: object[],
 *            sequence: string}}
 *   `sequence` is the colon-joined chains, which is what every layer below
 *   already reads; `ligandCodes` are upper-cased CCD codes, in order, one per
 *   instance - or, for a `smiles` row, `{smiles, code}` objects, which are NOT
 *   upper-cased because case is meaning in a SMILES.
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
  // 🔴 PER CHAIN AND IN CHAIN ORDER, because that is the only thing that says
  // what a chain's letters mean. featuriseProtein reads it by the index of the
  // chain in the colon-joined sequence, so it is built in the same loop that
  // builds them and can never drift from it.
  const chainKinds = [];
  /** Which chains asked for an alignment. See the push below. */
  const chainMsa = [];
  const templates = [];
  /** Which name each distinct SMILES was given; see `ligandName`. */
  const smilesCodes = new Map();
  for (const entity of entities) {
    // 🔴 A CONTACT IS NOT A CHAIN, and the `else` below would make it a LIGAND.
    // Handled after the loop, once the chains it names exist.
    if (entity.type === "contact") continue;
    for (let copy = 0; copy < entity.copies; copy += 1) {
      if (POLYMER_TYPES.includes(entity.type)) {
        for (const modification of entity.modifications ?? []) {
          modifications.push({
            chain: chains.length,
            position: modification.position,
            code: modification.code.trim().toUpperCase(),
          });
        }
        chainKinds.push(entity.type);
        // 🔴 PER CHAIN, IN CHAIN ORDER, like chainKinds above it. Whether a
        // protein gets an alignment is the entity's choice and the copies
        // inherit it, because two copies of one sequence are one search.
        // `undefined` on an entity means yes: every job written before this
        // asked for one, and a missing field must not quietly fold a chain
        // single-sequence.
        chainMsa.push(entity.type !== "protein" ? false : entity.msa !== "none");
        // 🔴 THE TEMPLATE IS RECORDED PER CHAIN, NOT PER ENTITY, for the same
        // reason the modifications are: copies are expanded, so two copies of a
        // templated chain are two chains each carrying it, and the embedder
        // indexes slots by the chain number it will actually see.
        if (templateAsked(entity.template) && entity.type === "protein") {
          // 🔴 `origin` IS THE ENTITY'S OWN OBJECT, NOT A COPY OF IT. The
          // coverage a template turns out to have - "17 of 120 residues" - is
          // discovered when the structure is fetched, long after this, and the
          // only place worth writing it is the row the reader is looking at.
          // Written onto the spread copy instead it goes nowhere, which is
          // exactly what happened: the page fetched, mapped, folded, and
          // reported "(no status)".
          templates.push({ chain: chains.length, ...entity.template,
                           origin: entity.template });
        }
        chains.push(cleanSequence(entity.value));
      } else if (entity.type === "smiles") {
        // 🔴 NOT UPPER-CASED, AND CARRIED AS AN OBJECT SO THE LAYER BELOW
        // CANNOT MISTAKE IT FOR A CODE. Case is meaning in a SMILES: the
        // lower-case letters are the aromatic atoms.
        //
        // 🔴 AND EACH DISTINCT STRUCTURE GETS A DISTINCT NAME. Every SMILES
        // ligand used to be called `LIG`, so a job with a benzene and a
        // glycerol wrote two different molecules under one residue name - a
        // PDB a reader cannot tell apart, and tooling that filters by name
        // silently mixing them. Identical strings still share a name, because
        // they are the same molecule and genuinely one entity.
        const text = entity.value.trim();
        if (!smilesCodes.has(text)) smilesCodes.set(text, ligandName(smilesCodes.size));
        ligandCodes.push({ smiles: text, code: smilesCodes.get(text) });
      } else ligandCodes.push(entity.value.trim().toUpperCase());
    }
  }
  // 🔴 THE CONTACTS LAST, ONCE EVERY CHAIN THEY NAME EXISTS. A letter is
  // resolved against the order the fold will actually see: the polymer chains
  // in `chains`, then each ligand - which is the order `featuriseProtein`
  // assigns `asymId` in, so one number reaches the featuriser and no second
  // convention is invented on the way.
  const bonds = [];
  for (const entity of entities) {
    if (entity.type !== "contact") continue;
    const parsed = parseContact(entity.value);
    if (parsed === null) continue;
    const asymOf = (letter) => {
      const at = letter.split("").reduce(
        (total, character) => total * 26 + (character.charCodeAt(0) - 64), 0) - 1;
      return at;
    };
    const end = (side) => ({
      asym: asymOf(side.chain), residue: side.residue,
      // 🔴 AN ABSENT ATOM IS NOT AN ERROR HERE. `token_bonds` is token x token,
      // so for a standard residue the atom decides nothing - the residue has
      // one token whichever atom is named. It matters only for a ligand or an
      // atomised residue, and featuriseProtein refuses by NAME when it does.
      ...(side.atom === null ? {} : { atom: side.atom }),
    });
    bonds.push({ from: end(parsed.from), to: end(parsed.to) });
  }
  return {
    chains, chainKinds, chainMsa, ligandCodes, modifications, templates,
    ...(bonds.length === 0 ? {} : { bonds }),
    sequence: chains.join(":"),
  };
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
