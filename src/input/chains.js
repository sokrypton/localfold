import { parseA3m } from "./a3m.js";

/** ColabFold's monomer-model chain break. It is safely beyond the model's +/-32 relative-position window. */
export const CHAIN_BREAK_OFFSET = 200;

/**
 * Validate a partition of a concatenated sequence into chains.
 * @param {number} totalLength
 * @param {readonly number[] | undefined} chainLengths
 * @returns {number[]}
 */
export function validatedChainLengths(totalLength, chainLengths) {
  if (!Number.isSafeInteger(totalLength) || totalLength <= 0) {
    throw new RangeError("sequence length must be a positive integer");
  }
  const lengths = chainLengths === undefined ? [totalLength] : Array.from(chainLengths);
  if (lengths.length === 0 || lengths.some((length) => !Number.isSafeInteger(length) || length <= 0)) {
    throw new RangeError("chain lengths must be positive integers");
  }
  const sum = lengths.reduce((value, length) => value + length, 0);
  if (!Number.isSafeInteger(sum) || sum !== totalLength) {
    throw new RangeError(`chain lengths sum to ${sum}; expected ${totalLength}`);
  }
  return lengths;
}

/**
 * Residue indices for AlphaFold's monomer-model oligomer offset trick.
 *
 * The ordinary concatenated index is retained and every preceding chain break
 * contributes another +200, matching ColabFold's `chain_break`. Cross-chain
 * pairs therefore saturate the monomer model's relative-position embedding.
 *
 * @param {number} totalLength
 * @param {readonly number[] | undefined} chainLengths
 * @returns {Float32Array} shape [totalLength]
 */
export function residueIndexWithChainBreaks(totalLength, chainLengths) {
  const lengths = validatedChainLengths(totalLength, chainLengths);
  const result = new Float32Array(totalLength);
  let residue = 0;
  for (let chain = 0; chain < lengths.length; chain += 1) {
    for (let within = 0; within < lengths[chain]; within += 1) {
      result[residue] = residue + chain * CHAIN_BREAK_OFFSET;
      residue += 1;
    }
  }
  return result;
}

/**
 * Assemble per-chain A3Ms into ColabFold-style unpaired complex rows.
 *
 * The first row contains every chain. Each remaining row occupies one chain
 * and is padded with gaps in all others. Supplying the same A3M more than once
 * gives the homooligomer expansion; distinct A3Ms give a heterooligomer MSA.
 * Lowercase insertions are kept verbatim because they do not consume aligned
 * columns but do carry the deletion counts used by feature preprocessing.
 *
 * @param {readonly string[]} a3mTexts one A3M per physical chain
 * @returns {string}
 */
export function mergeUnpairedChainA3ms(a3mTexts) {
  if (!Array.isArray(a3mTexts) || a3mTexts.length === 0) {
    throw new RangeError("at least one chain A3M is required");
  }
  const alignments = a3mTexts.map((text) => parseA3m(text));
  const lengths = alignments.map((alignment) => alignment.length);
  const query = alignments.map((alignment) => alignment.query).join("");
  const lines = [">query", query];
  for (let chain = 0; chain < alignments.length; chain += 1) {
    const alignment = alignments[chain];
    const left = lengths.slice(0, chain).reduce((sum, length) => sum + length, 0);
    const right = lengths.slice(chain + 1).reduce((sum, length) => sum + length, 0);
    for (let row = 1; row < alignment.rawSequences.length; row += 1) {
      lines.push(`>chain_${chain + 1}|${alignment.descriptions[row]}`);
      lines.push(`${"-".repeat(left)}${alignment.rawSequences[row]}${"-".repeat(right)}`);
    }
  }
  return `${lines.join("\n")}\n`;
}

/**
 * Assemble per-chain A3Ms into complex rows, pairing copies of one protein.
 *
 * 🔴 BLOCK-DIAGONAL IS THE WRONG SHAPE FOR REPEATED CHAINS, twice over. When a
 * complex contains the same protein more than once, every copy is searched
 * against the same database and gets the SAME alignment, so row s of copy 1 and
 * row s of copy 2 are one homolog from one organism. Stacking them diagonally
 * throws that away and then pays for the loss:
 *
 *   - it spends N rows of the 508-cluster budget to say what one row says, so
 *     each copy ends up with about 508/N sequences instead of 508. Measured on
 *     the 59-residue test case doubled: 269 and 236 rows, against 508 folding
 *     the monomer by itself.
 *   - it hides the pairing. Coevolution between the copies is exactly the
 *     signal an oligomer interface is predicted from, and gap-padded rows carry
 *     none of it.
 *
 * Pairing repeated chains is not an approximation of the diagonal form - it is
 * the construction the diagonal form is a lossy stand-in for. Chains are grouped
 * by their query sequence, because identical sequences are the same protein and
 * were served by one search; distinct groups stay block-diagonal, since pairing
 * two different proteins by row index would invent coevolution between
 * unrelated organisms.
 *
 * @param {readonly string[]} a3mTexts one A3M per physical chain
 * @returns {string}
 */
export function mergeChainA3ms(a3mTexts) {
  if (!Array.isArray(a3mTexts) || a3mTexts.length === 0) {
    throw new RangeError("at least one chain A3M is required");
  }
  const alignments = a3mTexts.map((text) => parseA3m(text));
  const lengths = alignments.map((alignment) => alignment.length);
  const query = alignments.map((alignment) => alignment.query).join("");
  const lines = [">query", query];

  const groups = new Map();
  alignments.forEach((alignment, chain) => {
    const existing = groups.get(alignment.query);
    if (existing === undefined) groups.set(alignment.query, [chain]);
    else existing.push(chain);
  });

  for (const chains of groups.values()) {
    // ...the first copy's alignment speaks for the group. They were produced by
    // one search for one sequence, so any later copy is the same alignment.
    const alignment = alignments[chains[0]];
    const member = new Set(chains);
    const label = chains.map((chain) => chain + 1).join("+");
    for (let row = 1; row < alignment.rawSequences.length; row += 1) {
      const parts = lengths.map((length, chain) => (
        member.has(chain) ? alignment.rawSequences[row] : "-".repeat(length)
      ));
      lines.push(`>chain_${label}|${alignment.descriptions[row]}`, parts.join(""));
    }
  }
  return `${lines.join("\n")}\n`;
}

/**
 * Project a complex A3M into one viewer-compatible A3M per physical chain.
 *
 * Model inference must retain the full concatenated alignment. py2Dmol instead
 * matches each A3M query to one PDB chain, so the display handoff needs these
 * per-chain views. Rows that are gaps throughout a chain are omitted. Lowercase
 * insertion identities are immaterial to the viewer and have already been
 * represented by their aligned columns in `parseA3m`.
 *
 * @param {string} a3mText
 * @param {readonly number[]} chainLengths
 * @returns {string[]}
 */
export function splitComplexA3mByChain(a3mText, chainLengths) {
  const alignment = parseA3m(a3mText);
  const lengths = validatedChainLengths(alignment.length, chainLengths);
  const results = [];
  let start = 0;
  for (let chain = 0; chain < lengths.length; chain += 1) {
    const end = start + lengths[chain];
    const lines = [`>query_chain_${chain + 1}`, alignment.query.slice(start, end)];
    for (let row = 1; row < alignment.sequences.length; row += 1) {
      const sequence = alignment.sequences[row].slice(start, end);
      if (!/[^-]/.test(sequence)) continue;
      lines.push(`>${alignment.descriptions[row]}`, sequence);
    }
    results.push(`${lines.join("\n")}\n`);
    start = end;
  }
  return results;
}
