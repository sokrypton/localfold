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
