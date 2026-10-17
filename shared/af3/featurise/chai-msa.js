/**
 * chai-1's MSA inputs where AlphaFold 3's differ, once there is an alignment (af3-any-model 1f9b3cf
 * model_features._chai_msa, as chai-lab builds them):
 *
 * - a protein chain with an alignment: its PROFILE counts the classes of its rows - the chain's query, its paired rows
 *   and its unpaired ones - deduplicated by their tokens, in chai's uint8 dtype, so a count wraps at 256 before it is
 *   normalised by their sum (MSAProfileGenerator's scatter_add); its deletion mean is over the same rows. AF3's is the
 *   true frequency over the undeduplicated unpaired rows, and on a 10128-row alignment the two correlate 0.44.
 * - every other polymer chain is chai's single-sequence context: its sequence string (a modification written
 *   "(CCD)") through the PROTEIN a3m tokeniser - so a DNA A/C/G/T reads ALA/CYS/GLY/THR - as the query row and a one-hot
 *   profile, and no other row (masked).
 * - token-identical rows of the whole MSA are then masked, as chai drops them.
 *
 * 🔴 AND A QUERY-ONLY ALIGNMENT IS NO ALIGNMENT (ColabFold ebbd90d): with no row anywhere that is not a chain's own
 * query, chai-1 takes its single-sequence path, as it does with none - `applyChaiMsa` returns false.
 *
 * cuda/featurise/chai_msa.h is the native twin; test:featurise holds the two to the same bytes.
 */

const PROTEIN = "ARNDCQEGHILKMFPSTWYV";
const GAP = 21, MASKED = 31;

/** chai's a3m tokeniser over one row: capitals and '-' are columns (another capital is X, 20), a lower-case letter is a
 * deletion counted before the next column (capped at 255), anything else is skipped */
export function chaiTokenise(row, width = Infinity) {
  const tokens = [], deletions = [];
  let lower = 0;
  for (const ch of row) {
    const code = ch.charCodeAt(0);
    if ((code >= 65 && code <= 90) || ch === "-") {
      if (tokens.length >= width) break;
      const index = PROTEIN.indexOf(ch);
      tokens.push(ch === "-" ? GAP : index >= 0 ? index : 20);
      deletions.push(Math.min(lower, 255));
      lower = 0;
    } else if (code >= 97 && code <= 122) {
      lower += 1;
    }
  }
  while (Number.isFinite(width) && tokens.length < width) { tokens.push(GAP); deletions.push(0); }
  return { tokens, deletions };
}

/** an A3M's records, the sequence lines of each joined, empty ones dropped (parsers.parse_fasta) */
function records(text) {
  const out = [];
  let current = null;
  for (const raw of (text ?? "").split("\n")) {
    const line = raw.trim();
    if (line.startsWith(">")) { if (current !== null) out.push(current); current = ""; continue; }
    if (current === null) current = "";
    current += line;
  }
  if (current !== null) out.push(current);
  return out.filter((row) => row.length > 0);
}

/**
 * The texts' rows, per protein chain: the chain's slice of every PAIRED row (if it has a paired alignment at all) and
 * of every UNPAIRED row it has a residue in (the merged unpaired block is block-diagonal - a row is one chain's homolog
 * with gaps elsewhere).
 */
function chainRows(texts, offsets, lengths, chain) {
  const width = offsets[offsets.length - 1] + lengths[lengths.length - 1];
  const out = [];
  const slice = (row) => {
    const { tokens, deletions } = chaiTokenise(row, width);
    return { tokens: tokens.slice(offsets[chain], offsets[chain] + lengths[chain]),
             deletions: deletions.slice(offsets[chain], offsets[chain] + lengths[chain]) };
  };
  // (a chain's paired rows only where it HAS a paired alignment - some row past the query with a residue in its
  // columns: a chain with none reads all gaps there, and af3-any-model reads no rows for it; a gap row inside a real
  // paired block is that block's own no-hit row and counts)
  const paired = records(texts.paired).map(slice);
  if (paired.slice(1).some((one) => one.tokens.some((token) => token !== GAP))) out.push(...paired);
  for (const row of records(texts.unpaired)) {
    const one = slice(row);
    if (one.tokens.some((token) => token !== GAP)) out.push(one);
  }
  return out;
}

/** each column-bearing chain's first column and length, in column order */
function layout(chains, columnChains) {
  const offsets = [], lengths = [];
  let column = 0;
  for (const chain of columnChains) { offsets.push(column); lengths.push(chains[chain].sequence.length); column += chains[chain].sequence.length; }
  return { offsets, lengths };
}

/**
 * @param batch featuriseProtein's batch, rewritten in place
 * @param chains the polymer chains in order: {sequence, kind, modifications: [{position, code}]}
 * @param texts {paired, unpaired}: the alignment's blocks (unpaired before any deduplication against the paired)
 * @param coversNucleic the alignment's columns span every polymer chain, not the protein ones alone
 * @returns false, the batch untouched, where no chain has a row but its own query: the caller's no-alignment path
 */
export function applyChaiMsa(batch, chains, texts, coversNucleic) {
  const { tokens, sequences } = batch;
  const columnChains = chains.map((_, index) => index).filter((index) => coversNucleic || chains[index].kind === "protein");
  const { offsets, lengths } = layout(chains, columnChains);
  const rowsOf = chains.map((chain, index) => (chain.kind === "protein" && columnChains.includes(index)
    ? chainRows(texts, offsets, lengths, columnChains.indexOf(index)) : []));
  const homologs = chains.map((chain, index) => {
    const query = chaiTokenise(chain.sequence).tokens.join(",");
    return rowsOf[index].some((row) => row.tokens.join(",") !== query);
  });
  if (!homologs.some(Boolean)) return false;
  for (let chain = 0; chain < chains.length; chain += 1) {
    const selected = [];
    for (let token = 0; token < tokens; token += 1) if (batch.asymId[token] === chain + 1) selected.push(token);
    if (selected.length === 0) continue;
    const query = chaiTokenise(chains[chain].sequence), rows = rowsOf[chain];
    if (homologs[chain]) {
      // the alignment's rows, deduplicated by their tokens (the first of each), counted in uint8
      const seen = new Set(), kept = [];
      for (const row of [query, ...rows]) {
        const key = row.tokens.join(",");
        if (!seen.has(key)) { seen.add(key); kept.push(row); }
      }
      const length = query.tokens.length;
      const profile = [], deletionMean = [];
      for (let column = 0; column < length; column += 1) {
        const counts = new Array(31).fill(0);
        let deletions = 0;
        for (const row of kept) { counts[row.tokens[column]] += 1; deletions += row.deletions[column]; }
        let total = 0;
        for (let k = 0; k < 31; k += 1) { counts[k] %= 256; total += counts[k]; }
        profile.push(counts.map((count) => Math.fround(count / Math.max(total, 1))));
        deletionMean.push(Math.fround(deletions / kept.length));
      }
      for (const token of selected) {
        const column = batch.residueIndex[token] - 1;
        for (let k = 0; k < 31; k += 1) batch.profile[token * 31 + k] = profile[column][k];
        batch.deletionMean[token] = deletionMean[column];
      }
      continue;
    }
    // the single-sequence context: the sequence string, a modification written "(CCD)", through the protein tokeniser
    const modified = new Map((chains[chain].modifications ?? []).map((one) => [one.position, one.code]));
    let text = "";
    for (let index = 0; index < chains[chain].sequence.length; index += 1) {
      text += modified.has(index + 1) ? `(${modified.get(index + 1)})` : chains[chain].sequence[index];
    }
    const context = chaiTokenise(text).tokens;
    for (const token of selected) {
      const q = context[Math.min(batch.residueIndex[token] - 1, context.length - 1)];
      batch.msa[token] = q;
      for (let row = 1; row < sequences; row += 1) {
        batch.msa[row * tokens + token] = MASKED;
        batch.msaMask[row * tokens + token] = 0;
      }
      for (let row = 0; row < sequences; row += 1) batch.deletionMatrix[row * tokens + token] = 0;
      for (let k = 0; k < 31; k += 1) batch.profile[token * 31 + k] = k === q ? 1 : 0;
      batch.deletionMean[token] = 0;
    }
  }
  // token-identical rows of the whole MSA: the first of each kept, the rest masked
  const seen = new Set();
  for (let row = 0; row < sequences; row += 1) {
    let live = false;
    for (let token = 0; token < tokens; token += 1) if (batch.msaMask[row * tokens + token] !== 0) { live = true; break; }
    if (!live) continue;
    const key = Array.from(batch.msa.subarray(row * tokens, (row + 1) * tokens)).join(",");
    if (seen.has(key)) batch.msaMask.fill(0, row * tokens, (row + 1) * tokens);
    else seen.add(key);
  }
  return true;
}
