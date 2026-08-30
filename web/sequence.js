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

/**
 * `>` starts a RECORD; `;` starts a COMMENT. They are not the same thing and
 * conflating them truncated a sequence at its second comment line.
 */
const RECORD = /^\s*>/;
const COMMENT = /^\s*;/;

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
