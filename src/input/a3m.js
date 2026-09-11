/**
 * The residues an ALIGNED column may hold, as a table over character codes.
 *
 * 🔴 THIS WAS `/^[ACDEFGHIKLMNPQRSTVWYX-]+$/` RUN ONCE PER RESIDUE, AND SO WAS
 * `toUpperCase`. A 30,000-row alignment of a 200-residue query is six million
 * of each, on the main thread, before a fold can start: 307 ms of a page that
 * has not drawn anything yet, against 85 for the walk below.
 *
 * 🔴 AND `toUpperCase` WAS NEVER DOING ANYTHING ON THIS BRANCH. It runs only
 * where the character is NOT a-z - lowercase is what an insertion IS in an
 * a3m - so every character reaching it is already uppercase or a symbol, and
 * the call returned its argument six million times.
 */
const ALIGNED_CODE = (() => {
  const table = new Uint8Array(128);
  for (const symbol of "ACDEFGHIKLMNPQRSTVWYX-") table[symbol.charCodeAt(0)] = 1;
  return table;
})();

const LOWER_A = "a".charCodeAt(0);
const LOWER_Z = "z".charCodeAt(0);

/**
 * A string from a code buffer, in chunks.
 *
 * 🔴 `String.fromCharCode(...codes)` SPREADS INTO ARGUMENTS AND BLOWS THE
 * STACK. A row is only a few hundred residues here, but a3m rows are not
 * bounded by anything this module controls, and the failure is a
 * RangeError from deep inside the parser rather than a message about the
 * input.
 */
function stringOfCodes(codes, length) {
  const CHUNK = 4096;
  if (length <= CHUNK) return String.fromCharCode.apply(null, codes.subarray(0, length));
  let out = "";
  for (let at = 0; at < length; at += CHUNK) {
    out += String.fromCharCode.apply(null, codes.subarray(at, Math.min(at + CHUNK, length)));
  }
  return out;
}

export function parseA3m(text) {
  const descriptions = [];
  const rawSequences = [];
  let current = -1;
  for (const sourceLine of text.split(/\r?\n/)) {
    const line = sourceLine.trim();
    if (line === "" || line.startsWith("#")) continue;
    if (line.startsWith(">")) {
      const description = line.slice(1).trim();
      if (description === "") throw new Error("A3M contains an empty FASTA header");
      descriptions.push(description);
      rawSequences.push("");
      current += 1;
      continue;
    }
    if (current < 0) throw new Error("A3M sequence data appears before the first FASTA header");
    if (/\s/.test(line)) throw new Error(`A3M sequence ${descriptions[current]} contains whitespace`);
    rawSequences[current] += line;
  }
  if (rawSequences.length === 0) throw new Error("A3M contains no sequences");

  const sequences = [];
  const deletionMatrix = [];
  for (let row = 0; row < rawSequences.length; row += 1) {
    const raw = rawSequences[row];
    if (raw === "") throw new Error(`A3M sequence ${descriptions[row]} is empty`);
    // ...into a code buffer and one string at the end, rather than a
    // character-at-a-time concatenation per row.
    const codes = new Uint16Array(raw.length);
    let alignedLength = 0;
    let insertionCount = 0;
    const deletions = [];
    for (let at = 0; at < raw.length; at += 1) {
      const code = raw.charCodeAt(at);
      if (code >= LOWER_A && code <= LOWER_Z) {
        insertionCount += 1;
        continue;
      }
      if (code > 127 || ALIGNED_CODE[code] === 0) {
        throw new Error(`A3M sequence ${descriptions[row]} contains invalid residue `
          + `${JSON.stringify(raw[at])}`);
      }
      codes[alignedLength] = code;
      alignedLength += 1;
      deletions.push(insertionCount);
      insertionCount = 0;
    }
    sequences.push(stringOfCodes(codes, alignedLength));
    deletionMatrix.push(deletions);
  }

  const length = sequences[0] .length;
  if (length === 0 || sequences[0] .includes("-")) {
    throw new Error("the first A3M sequence must be a non-empty, ungapped query");
  }
  for (let row = 0; row < sequences.length; row += 1) {
    if (sequences[row] .length !== length) {
      throw new Error(
        `A3M row ${descriptions[row]} has aligned length ${sequences[row] .length}; expected ${length}`,
      );
    }
  }
  return {
    query: sequences[0],
    descriptions,
    rawSequences,
    sequences,
    deletionMatrix,
    depth: sequences.length,
    length,
  };
}

/**
 * How many DISTINCT sequences an alignment carries, which is not its depth.
 *
 * 🔴 `depth` COUNTS ROWS, AND A SEARCH THAT FOUND NOTHING RETURNS TWO OF THEM.
 * `extractMmseqs2A3m` joins the uniref block and the environmental block and
 * returns each WHOLE, so both begin with their own `>101` - a query with no
 * homologs comes back as depth 2 carrying one sequence. Folding that is not an
 * alignment of two; it is the query twice, and the second copy moves pTM from
 * 0.3965 to 0.4148 on the 59-mer. See docs/AF2.md.
 *
 * On the ALIGNED columns, because `parseA3m` has already dropped the lowercase
 * insertions into the deletion matrix and two rows differing only there are the
 * same row to the model - which is what AlphaFold hashes.
 */
export function distinctSequenceCount(a3mText) {
  return new Set(parseA3m(a3mText).sequences).size;
}

/** Whether a search returned nothing but the query, however many rows it used. */
export function foundOnlyTheQuery(a3mText) {
  return distinctSequenceCount(a3mText) <= 1;
}

/**
 * Whether this alignment should simply be folded as a single sequence.
 *
 * 🔴 THE SECOND CONDITION IS NOT DECORATION. An A3M's own first record WINS
 * over the sequence in the box - deliberately, so a reader can paste an
 * alignment and fold what it describes - and routing to the query-only path
 * throws that away. So a pasted or uploaded alignment carrying one sequence is
 * only folded as a single sequence when that sequence is the one being folded;
 * otherwise it keeps the alignment and folds the protein it names, exactly as
 * before. A searched alignment cannot differ - generateMmseqs2Msa refuses an
 * A3M whose query is not the sequence it asked about - so this costs the search
 * path nothing and protects the two paths a reader controls.
 *
 * @param {string} a3mText the alignment
 * @param {string} sequence the chains about to be folded, concatenated
 */
export function foldsAsSingleSequence(a3mText, sequence) {
  return foundOnlyTheQuery(a3mText) && parseA3m(a3mText).query === sequence;
}
