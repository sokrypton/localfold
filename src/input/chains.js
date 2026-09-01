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
 * Stack a paired block above an unpaired one, as one alignment.
 *
 * 🔴 THE SECOND BLOCK'S QUERY ROW IS DROPPED, AND ONLY IT. Both blocks begin
 * with the same complex query - each is a complete A3M in its own right - so
 * concatenating them verbatim would put the query in twice and make every
 * profile computed over the result count it twice. The paired block keeps its
 * query, because that row is the alignment's own first row and every consumer
 * expects an A3M to start with the sequence being folded.
 *
 * @param {string} pairedA3m rows already aligned across chains
 * @param {string} unpairedA3m the block-diagonal or dense unpaired rows
 * @returns {string}
 */
export function concatenateA3mBlocks(pairedA3m, unpairedA3m) {
  const paired = parseA3m(pairedA3m);
  const unpaired = parseA3m(unpairedA3m);
  if (paired.query !== unpaired.query) {
    throw new Error("the paired and unpaired A3M blocks describe different queries");
  }
  const lines = [];
  for (let row = 0; row < paired.depth; row += 1) {
    lines.push(`>${paired.descriptions[row]}`, paired.rawSequences[row]);
  }
  for (let row = 1; row < unpaired.depth; row += 1) {
    lines.push(`>${unpaired.descriptions[row]}`, unpaired.rawSequences[row]);
  }
  return `${lines.join("\n")}\n`;
}

/**
 * Merge per-chain A3Ms the way AlphaFold 3 merges its UNPAIRED block.
 *
 * 🔴 THIS IS NOT BLOCK-DIAGONAL, AND THAT IS THE WHOLE DIFFERENCE FROM
 * mergeUnpairedChainA3ms. AF3 pads each chain's alignment to the deepest one
 * and concatenates along the TOKEN axis (`merge_msa_features`, axis=1), so
 * merged row r is chain A's row r beside chain B's row r - not chain A's rows
 * above chain B's rows against gaps. Handing AF3 the block-diagonal form halves
 * the information in every row and doubles the depth to carry it.
 *
 * 🔴 AF3 DOES THIS FOR EVERY CHAIN, WITH NO NOTION OF ENTITY. There is no
 * block_diag anywhere in its codebase. AlphaFold-Multimer is the one that
 * distinguishes: `_merge_homomers_dense_msa` groups chains by entity_id and
 * merges each group densely - copies of one sequence are NEVER block
 * diagonalised - and only distinct entities are block diagonalised against each
 * other. So for a homo-oligomer this function and mergeChainA3ms agree, and for
 * a heteromer the two models genuinely differ.
 *
 * mergeUnpairedChainA3ms, which block-diagonalises every chain INCLUDING copies
 * of one sequence, is neither of those: it belongs to the AF2-monomer hack,
 * where the model has no chain input at all and the +200 residue offset stands
 * in for one. See residueIndexWithChainBreaks.
 *
 * The alignment by row index is not a claim that row r of two chains is one
 * organism - that is what a paired block is for. It is only how AF3 packs
 * them, and for a homo-oligomer, where every copy has the same alignment, it
 * happens to coincide with pairing.
 *
 * Short chains are padded with gaps, which is AF3's `MSA_PAD_VALUES['msa'] =
 * MSA_GAP_IDX`; their deletion counts pad with zero, which falls out of a gap
 * column having no insertions before it.
 *
 * @param {readonly string[]} a3mTexts one A3M per physical chain
 * @returns {string}
 */
export function mergeRowAlignedChainA3ms(a3mTexts) {
  if (!Array.isArray(a3mTexts) || a3mTexts.length === 0) {
    throw new RangeError("at least one chain A3M is required");
  }
  const alignments = a3mTexts.map((text) => parseA3m(text));
  const lengths = alignments.map((alignment) => alignment.length);
  const depth = Math.max(...alignments.map((alignment) => alignment.depth));
  const lines = [];
  for (let row = 0; row < depth; row += 1) {
    const parts = alignments.map((alignment, chain) => (
      // 🔴 rawSequences, NOT sequences: the lowercase insertions carry the
      // deletion counts, and a merge that drops them silently zeroes a feature.
      row < alignment.depth ? alignment.rawSequences[row] : "-".repeat(lengths[chain])
    ));
    const label = row === 0 ? "query" : alignments
      .map((alignment, chain) => (row < alignment.depth
        ? `${chain + 1}:${alignment.descriptions[row]}` : `${chain + 1}:-`))
      .join(" ");
    lines.push(`>${label}`, parts.join(""));
  }
  return `${lines.join("\n")}\n`;
}

/**
 * Residue indices numbered WITHIN each chain, which is what multimer wants.
 *
 * 🔴 THE +200 BREAK IS A MONOMER WORKAROUND AND MULTIMER MUST NOT HAVE IT.
 * residueIndexWithChainBreaks pushes each chain past the model's +/-32 window so
 * that a graph with no notion of chains reads cross-chain pairs as "very far
 * apart". Multimer is TOLD which chain is which, by asym_id, and numbers each
 * chain from zero.
 *
 * Leaving the offset in is not merely redundant: it hides whether the chain
 * identity works at all. With the offset, cross-chain pairs land in the
 * saturated end bins even when asym_id is not supplied, so turning chain
 * awareness off changes almost nothing and a broken encoding looks fine.
 *
 * @param {number} totalLength
 * @param {readonly number[] | undefined} chainLengths
 * @returns {Float32Array} shape [totalLength]
 */
export function residueIndexPerChain(totalLength, chainLengths) {
  const lengths = validatedChainLengths(totalLength, chainLengths);
  const result = new Float32Array(totalLength);
  let residue = 0;
  for (const length of lengths) {
    for (let within = 0; within < length; within += 1) result[residue++] = within;
  }
  return result;
}

/**
 * AlphaFold-multimer's per-residue chain identity: asym, entity and symmetry.
 *
 * These are what the multimer relative encoding reads, and between them they say
 * everything the model is told about how the chains relate:
 *
 *   asym_id    which physical chain a residue belongs to. Distinct for every
 *              copy, so two copies of one protein are still two chains.
 *   entity_id  which distinct SEQUENCE it is. Copies of one protein share it,
 *              which is how the model learns they are the same molecule.
 *   sym_id     which copy within that entity, counting from zero. The encoding
 *              reads the difference between two residues' sym_ids, so this is
 *              what makes copy 1 -> copy 2 a different relationship from copy 2
 *              -> copy 1 rather than an unordered pair.
 *
 * 🔴 A MONOMER IS ALL ZEROS, and that is not a special case - it is what one
 * chain, one entity, one copy actually means. The encoding then reduces to the
 * monomer form exactly, which is why the widened graph needs no branch.
 *
 * @param {readonly number[] | undefined} chainLengths
 * @param {readonly string[]} [chainSequences] one per chain; equal sequences
 *   share an entity. Without it every chain is treated as its own entity.
 * @returns {{asymId: Float32Array, entityId: Float32Array, symId: Float32Array}}
 */
export function chainIdentity(totalLength, chainLengths, chainSequences = undefined) {
  const lengths = validatedChainLengths(totalLength, chainLengths);
  if (chainSequences !== undefined && chainSequences.length !== lengths.length) {
    throw new RangeError("one sequence per chain is required to group entities");
  }
  const asymId = new Float32Array(totalLength);
  const entityId = new Float32Array(totalLength);
  const symId = new Float32Array(totalLength);

  const entities = new Map();
  const copiesSeen = new Map();
  let residue = 0;
  for (let chain = 0; chain < lengths.length; chain += 1) {
    const key = chainSequences === undefined ? `chain-${chain}` : chainSequences[chain];
    if (!entities.has(key)) entities.set(key, entities.size);
    const entity = entities.get(key);
    const copy = copiesSeen.get(entity) ?? 0;
    copiesSeen.set(entity, copy + 1);
    for (let within = 0; within < lengths[chain]; within += 1) {
      asymId[residue] = chain;
      entityId[residue] = entity;
      symId[residue] = copy;
      residue += 1;
    }
  }
  return { asymId, entityId, symId };
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
  // 🔴 THIS IS AlphaFold-MULTIMER'S CONSTRUCTION, AND ONLY ITS.
  // merge_chain_features runs _merge_homomers_dense_msa first, grouping chains
  // by entity_id and concatenating each group along num_res, and only then
  // block diagonalises what remains - so copies of one sequence are dense and
  // distinct entities are block-diagonal, which is exactly what grouping by
  // identical query and spanning the group produces here. Do not reach for it
  // as "the paired merge": AF3 wants mergeRowAlignedChainA3ms and the AF2
  // monomer wants mergeUnpairedChainA3ms, and the three differ only on inputs
  // where nothing downstream will notice.
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

/**
 * Drop unpaired rows whose sequence is already in the paired block.
 *
 * AlphaFold does this per chain, before any merge, in
 * `msa_pairing.deduplicate_unpaired_sequences`: it hashes every row of the
 * chain's paired MSA and keeps only the unpaired rows that are not among them.
 * Both AF3 and AlphaFold-Multimer run it, and they run it for the same reason -
 * the paired and unpaired blocks come from the SAME databases, so for a target
 * with good pairing most of the unpaired block is the paired block again.
 *
 * 🔴 IT IS NOT A TIDINESS PASS, IT IS THE MSA BUDGET. The two blocks split a
 * fixed number of rows, so a duplicate does not merely add nothing - it evicts
 * a sequence that would have added something. On the 59-mer homodimer AF3's
 * own featuriser drops ALL 32 unpaired rows this way and pads the block with
 * zeros, which is how the discrepancy against our batch was found: we were
 * sending 32 rows the model had already seen.
 *
 * 🔴 THE COMPARISON IS ON THE ALIGNED COLUMNS, NOT THE RAW ROW. AF3 hashes the
 * featurised integer row, and featurisation has already dropped the lowercase
 * insertions into the deletion matrix - so two rows that differ only in their
 * insertions are the same row here. Comparing the raw A3M text instead keeps
 * duplicates that AlphaFold removes.
 *
 * @param {string} unpairedA3m  one chain's unpaired alignment
 * @param {string | null | undefined} pairedA3m  the same chain's paired one
 * 🔴 THE QUERY ROW IS THE ONE EXEMPTION, AND IT IS A REPRESENTATION DETAIL.
 * AF3 deduplicates arrays and drops the unpaired query happily, since the
 * paired block already opens with it. This function returns an A3M, and an A3M
 * without an ungapped query row is not one - parseA3m rejects it. So the query
 * stays here and af3MsaFromA3m skips it when a paired block exists, which
 * lands the same rows in the same order. Dropping it in both places leaves the
 * merged MSA one query short; in neither, one query too many.
 *
 * @returns {string} the unpaired A3M with duplicated rows removed
 */
export function deduplicateUnpairedAgainstPaired(unpairedA3m, pairedA3m) {
  if (pairedA3m === null || pairedA3m === undefined || pairedA3m === "") return unpairedA3m;
  const paired = parseA3m(pairedA3m);
  const unpaired = parseA3m(unpairedA3m);
  const seen = new Set(paired.sequences);
  const lines = [];
  for (let row = 0; row < unpaired.depth; row += 1) {
    if (row > 0 && seen.has(unpaired.sequences[row])) continue;
    lines.push(`>${unpaired.descriptions[row]}`, unpaired.rawSequences[row]);
  }
  return `${lines.join("\n")}\n`;
}
