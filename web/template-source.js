/**
 * Where a template comes from, and how it lines up with the chain it is shown
 * against.
 *
 * One field takes all three sources, because they are one question - "which
 * structure" - and two widgets asking it would be two things to keep in step:
 *
 *     1abc      a PDB entry          files.rcsb.org
 *     1abc_A    ...and which chain
 *     P00533    a UniProt accession  AlphaFold DB
 *     (a file)  whatever was dropped
 *
 * 🔴 THE LEGACY PDB FORMAT IS ASKED FOR FIRST, WHICH IS A REAL LIMIT AND NOT AN
 * OVERSIGHT. src/af3/template-input.js reads fixed columns, and an entry too
 * large for them - more than 62 chains or 99999 atoms - has no `.pdb` at the
 * RCSB and falls through to the mmCIF this cannot parse. That is rare for a
 * single template chain and the error says so rather than producing a template
 * of nothing.
 *
 * 🔴 AND A TEMPLATE THAT COVERS NOTHING IS AN ERROR, NOT AN EMPTY TEMPLATE. An
 * empty slot is not neutral - the aatype features are read whether or not there
 * is geometry, so a slot covering zero residues is the GAP token everywhere,
 * which is a claim rather than an absence. Better to refuse.
 */
import { alignPositions } from "./align.js";
import {
  chainResidues, filterByConfidence, identityMap, templateSlot,
} from "../src/af3/template-input.js";

const RCSB = "https://files.rcsb.org/download";
const AFDB = "https://alphafold.ebi.ac.uk/files";

/**
 * What a source string names.
 *
 * @param {string} text `1abc`, `1abc_A`, `P00533`, `P00533_A`
 * @returns {{id: string, chain: string|undefined, kind: "pdb"|"afdb"}}
 */
export function parseSource(text) {
  const trimmed = text.trim();
  const [id, chain] = trimmed.split(/[_:]/, 2);
  return {
    id: id.toUpperCase(),
    chain: chain === undefined || chain === "" ? undefined : chain,
    // 🔴 FOUR CHARACTERS IS A PDB ENTRY AND ANYTHING ELSE IS AN ACCESSION,
    // which is the rule ../mpnn's fetchPDB uses and is right often enough to
    // be worth not asking. A UniProt accession is six or ten.
    kind: id.length === 4 ? "pdb" : "afdb",
  };
}

/**
 * The structure text for a source, and where it came from.
 *
 * @param {string} text
 * @param {{signal?: AbortSignal, fetch?: typeof fetch}} [options]
 * @returns {Promise<{text: string, url: string, kind: string, chain?: string}>}
 */
export async function fetchStructure(text, options = {}) {
  const { id, chain, kind } = parseSource(text);
  const get = options.fetch ?? globalThis.fetch;
  const urls = kind === "pdb"
    // ...`.pdb` FIRST, unlike ../mpnn's fetchPDB, which prefers the mmCIF. See
    // the note at the top: this reader is fixed-column.
    ? [`${RCSB}/${id}.pdb`, `${RCSB}/${id}.cif`]
    : [`${AFDB}/AF-${id}-F1-model_v4.pdb`];
  let last;
  for (const url of urls) {
    try {
      const response = await get(url, { signal: options.signal });
      if (response.ok) return { text: await response.text(), url, kind, chain };
      last = new Error(`${response.status} for ${url}`);
    } catch (error) {
      last = error;
    }
  }
  throw last ?? new Error(`could not fetch ${text}`);
}

/**
 * Line a structure's residues up with a query chain.
 *
 * 🔴 TWO CASES WITH DIFFERENT FAILURE MODES, AND THE SAFE ONE IS THE DEFAULT.
 * When the chain has no sequence yet, the structure's own is used and the map
 * is the identity over what the file resolved - nothing to get wrong. When the
 * chain already has a sequence the two must be ALIGNED, and an alignment can be
 * wrong in ways that still produce a template: a construct with a tag shifts
 * every residue, and fitting index for index across the shift swings the whole
 * thing. So an alignment is only used when it is asked for.
 *
 * @param {{residues: object[], sequence: string}} structure
 * @param {string} query the chain's sequence, or "" to take the structure's
 * @returns {{map: Map<number, number>, sequence: string, identical: boolean}}
 */
export function mapToQuery(structure, query) {
  if (query.length === 0) {
    return { map: identityMap(structure), sequence: structure.sequence, identical: true };
  }
  if (query === structure.sequence) {
    return { map: identityMap(structure), sequence: query, identical: true };
  }
  // 🔴 alignPositions, NOT correspondence. correspondence short-circuits to the
  // identity whenever the two are the SAME LENGTH, which is right for a point
  // mutation and wrong here: a template of the same length as the query is not
  // the same protein, and taking it index for index would map every residue to
  // whatever happens to sit at its number.
  const map = new Map();
  for (const [queryIndex, templateIndex] of alignPositions(query, structure.sequence)) {
    map.set(queryIndex, templateIndex);
  }
  return { map, sequence: query, identical: false };
}

/**
 * A template slot for one chain, from a fetched or dropped structure.
 *
 * @param {object} options
 * @param {string} options.text the structure file
 * @param {string} [options.chain] which chain of it
 * @param {string} [options.query] the chain's sequence; "" takes the structure's
 * @param {number} options.tokens the whole complex's token count
 * @param {number} [options.offset] this chain's first token
 * @param {(residue: number) => number} [options.tokenOf] which token a residue
 *   of this chain occupies; see templateSlot. A modified residue is several
 *   tokens, so an offset alone is not enough once a chain carries one.
 * @param {number} [options.minConfidence] drop residues below this pLDDT
 * @param {boolean} [options.spanChains]
 * @returns {{slot: object, sequence: string, coverage: object}}
 */
export function buildTemplate(options) {
  const structure = chainResidues(options.text, options.chain);
  if (structure.residues.length === 0) {
    throw new Error(options.chain === undefined
      ? "that structure has no protein chain this can read"
      : `that structure has no chain ${options.chain}`);
  }
  const { map, sequence, identical } = mapToQuery(structure, options.query ?? "");
  const kept = filterByConfidence(map, structure, options.minConfidence ?? 0,
                                  (residue) => residue.confidence);
  const chainLength = (options.query ?? "").length || structure.residues.length;
  const slot = templateSlot({
    structure, tokens: options.tokens, map: kept,
    offset: options.offset ?? 0, tokenOf: options.tokenOf,
  });
  if (slot.covered === 0) {
    throw new Error("that template covers none of the chain"
      + (options.minConfidence ? ` above pLDDT ${options.minConfidence}` : ""));
  }
  slot.spanChains = options.spanChains === true;
  return {
    slot,
    sequence,
    coverage: {
      residues: slot.covered,
      of: chainLength,
      atoms: slot.atoms,
      dropped: map.size - kept.size,
      aligned: !identical,
      chain: structure.chain,
    },
  };
}

/** One line a reader can act on: what was used, and what was not. */
export function describeCoverage(coverage) {
  const percent = coverage.of === 0 ? 0
    : Math.round((coverage.residues / coverage.of) * 100);
  const parts = [`chain ${coverage.chain}: ${coverage.residues} of ${coverage.of}`
    + ` residues (${percent}%), ${coverage.atoms} atoms`];
  if (coverage.dropped > 0) parts.push(`${coverage.dropped} below the pLDDT floor`);
  // Worth saying out loud: an alignment is the case that can be wrong in a way
  // that still produces a template.
  if (coverage.aligned) parts.push("aligned to the sequence you typed");
  return parts.join(" · ");
}
