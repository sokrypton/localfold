/**
 * Turning what somebody pasted into a sequence.
 *
 * WHAT ARRIVES IN THAT BOX is rarely bare residues. It is a FASTA record with a
 * `>` line on top, or a UniProt block with a position number every ten residues,
 * or a column of sixty-wide lines out of a PDB entry, or an aligned row with
 * gaps in it. All of those are the same protein, and all of them used to fail
 * validation on the punctuation rather than fold.
 *
 * THE RULE: throw away FORMATTING, keep LETTERS. Whitespace, digits, and the
 * gap and stop characters are how sequences get written down, not part of the
 * sequence - so they go without comment. Anything else that is not a residue
 * stays in, and the caller's validation refuses it by name. That division is
 * the whole point: silently dropping an unexpected letter would turn a typo
 * into a different protein and fold it without a word.
 */

// -,. are alignment gaps; * is a stop; both are notation, not residues.
const FORMATTING = /[\s\d.\-*]+/g;
/** The same class, one character at a time and WITHOUT the global flag. */
const NOT_A_RESIDUE = /[\s\d.\-*]/;

/**
 * `>` starts a RECORD; `;` starts a COMMENT. They are not the same thing and
 * conflating them truncated a sequence at its second comment line.
 */
const RECORD = /^\s*>/;
const COMMENT = /^\s*;/;

/**
 * The cleaned sequence, and where each of its residues came from in the raw
 * text.
 *
 * 🔴 A RESIDUE'S POSITION IS NOT ITS OFFSET IN THE BOX. cleanSequence drops
 * FASTA headers, comment lines and every scrap of whitespace, so residue 12 of
 * a pasted record can be character 80 of what is on screen. Anything that has
 * to point AT a residue - a highlight behind the text, a click that means
 * "this one" - needs the map, and computing it by re-deriving the rules
 * separately is how the two drift apart. It is the same walk as cleanSequence,
 * carrying the offsets along.
 *
 * @param {string} text whatever is in the box
 * @returns {{cleaned: string, offsets: number[]}} `offsets[i]` is where
 *   `cleaned[i]` sits in `text`
 */
export function cleanSequenceMap(text) {
  // 🔴 NOT `FORMATTING.test(character)`. That regex carries the `g` flag for
  // the replace below, and a global regex REMEMBERS where it got to - so
  // testing one character at a time returns true, false, true, false down a run
  // of spaces, and "AC  DE" cleaned to "AC DE" while cleanSequence gave "ACDE".
  // The two would then disagree about which character residue 3 is.
  const offsets = [];
  let cleaned = "";
  let at = 0;
  const lines = text.split(/\r?\n/);
  // Which lines survive, by the same rules and in the same order.
  const kept = lines.filter((line) => !COMMENT.test(line));
  const start = kept.findIndex((line) => RECORD.test(line));
  const body = start === -1 ? kept : kept.slice(start + 1);
  const end = body.findIndex((line) => RECORD.test(line));
  const wanted = new Set(end === -1 ? body : body.slice(0, end));
  // ...matched by identity of position rather than of content, since two lines
  // of a sequence can read the same.
  const keepIndex = new Set();
  {
    let seen = 0;
    const survivors = [];
    lines.forEach((line, index) => {
      if (COMMENT.test(line)) return;
      survivors.push(index);
    });
    const bodyStart = start === -1 ? 0 : start + 1;
    const bodyEnd = end === -1 ? survivors.length : bodyStart + end;
    for (let i = bodyStart; i < bodyEnd; i += 1) keepIndex.add(survivors[i]);
    seen = wanted.size;
    void seen;
  }
  lines.forEach((line, index) => {
    if (keepIndex.has(index)) {
      for (let i = 0; i < line.length; i += 1) {
        const character = line[i];
        if (NOT_A_RESIDUE.test(character)) continue;
        cleaned += character.toUpperCase();
        offsets.push(at + i);
      }
    }
    at += line.length + 1;                 // the newline the split removed
  });
  return { cleaned, offsets };
}

/**
 * @param {string} text whatever is in the box
 * @returns {string} uppercase letters only, formatting removed
 */
export function cleanSequence(text) {
  const lines = text.split(/\r?\n/).filter((line) => !COMMENT.test(line));
  // THE FIRST RECORD ONLY. Dropping every header and concatenating what was
  // left turned a multi-FASTA into a chimera - two proteins joined end to end,
  // folded as though that were a real molecule and reported without comment.
  // This page folds one sequence, so a paste with several in it means the first.
  const start = lines.findIndex((line) => RECORD.test(line));
  const body = start === -1 ? lines : lines.slice(start + 1);
  const end = body.findIndex((line) => RECORD.test(line));
  return (end === -1 ? body : body.slice(0, end))
    .join("")
    .replace(FORMATTING, "")
    .toUpperCase();
}

/**
 * Extract the first FASTA header line, or null if there is no header.
 *
 * @param {string} text
 * @returns {string|null}
 */
export function extractFastaHeader(text) {
  const lines = text.split(/\r?\n/).filter((line) => !COMMENT.test(line));
  const headerLine = lines.find((line) => RECORD.test(line));
  if (headerLine === undefined) return null;
  const header = headerLine.replace(/^\s*>\s*/, "").trim();
  return header.length > 0 ? header : null;
}

/** The twenty, plus X for unknown - what the model can actually fold. */
const RESIDUES = /^[ARNDCQEGHILKMFPSTWYVX]+$/;

/**
 * Why this sequence cannot be folded, or null if it can.
 *
 * NAMES THE OFFENDING LETTERS rather than saying "invalid": B, J, O, U and Z
 * are real IUPAC codes that this model has no embedding for, and a reader who
 * pasted one needs to know which of the 300 characters to look at.
 */
export function sequenceProblem(sequence) {
  if (sequence.length === 0) return "Enter a protein sequence";
  if (RESIDUES.test(sequence)) return null;
  const bad = [...new Set(sequence.replace(/[ARNDCQEGHILKMFPSTWYVX]/g, ""))];
  return bad.length === 1
    ? `${bad[0]} is not one of the twenty amino acids`
    : `${bad.join(", ")} are not among the twenty amino acids`;
}

/**
 * The four bases of each kind. T is DNA's and U is RNA's, and swapping them is
 * the typo this catches: a `U` in a DNA chain is not thymine spelled oddly, it
 * is a different chain than the one that was meant.
 */
const NUCLEIC_BASES = { dna: "ACGT", rna: "ACGU" };

/**
 * Why this DNA or RNA sequence cannot be folded, or null if it can.
 *
 * 🔴 N IS NOT ACCEPTED, THOUGH PROTEIN'S X IS. The alphabet has an UNK slot for
 * an unknown amino acid and none for an unknown base - AF3's restypes run
 * A G C U for RNA and DA DG DC DT for DNA and stop - so an N would have to
 * become UNK, which is the amino-acid unknown and would put a protein token in
 * a nucleic chain.
 *
 * @param {string} sequence cleaned letters
 * @param {"dna"|"rna"} kind
 */
export function nucleicProblem(sequence, kind) {
  const bases = NUCLEIC_BASES[kind];
  const label = kind.toUpperCase();
  if (sequence.length === 0) return `Enter a ${label} sequence`;
  const bad = [...new Set([...sequence].filter((base) => !bases.includes(base)))];
  if (bad.length === 0) return null;
  // The swapped base gets its own message, because "T is not one of A, C, G, U"
  // is true and unhelpful to somebody who pasted a DNA sequence into an RNA row.
  const swapped = kind === "rna" ? "T" : "U";
  const instead = kind === "rna" ? "U (uracil)" : "T (thymine)";
  if (bad.length === 1 && bad[0] === swapped) {
    return `${swapped} is not ${kind === "rna" ? "an" : "a"} ${label} base`
      + ` - ${label} uses ${instead}.`
      + ` Change the entity type if this is ${kind === "rna" ? "DNA" : "RNA"}`;
  }
  return bad.length === 1
    ? `${bad[0]} is not one of ${bases.split("").join(", ")}`
    : `${bad.join(", ")} are not among ${bases.split("").join(", ")}`;
}

/** Why a colon-separated complex cannot be folded, or null if it can. */
export function complexSequenceProblem(sequence) {
  if (sequence.length === 0) return "Enter a protein sequence";
  const chains = sequence.split(":");
  if (chains.some((chain) => chain.length === 0)) return "Every chain between colons must contain a sequence";
  for (let chain = 0; chain < chains.length; chain += 1) {
    const problem = sequenceProblem(chains[chain]);
    if (problem !== null) return chains.length === 1 ? problem : `Chain ${chain + 1}: ${problem}`;
  }
  return null;
}

/** Physical chain sequences from the validated colon-separated notation. */
export function sequenceChains(sequence) {
  const problem = complexSequenceProblem(sequence);
  if (problem !== null) throw new Error(problem);
  return sequence.split(":");
}
