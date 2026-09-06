/**
 * AF3's tokens as contact classes, so its distogram can be read per pair kind.
 *
 * 🔴 A LIGAND TOKEN AND AN UNKNOWN RESIDUE SHARE AN `aatype`, so the alphabet
 * alone cannot say what a token is. `featurise.js` writes `UNK_AATYPE` for
 * every ligand atom - one token per heavy atom - and the same value for an X in
 * a protein chain. `ligandSpans` is what distinguishes them, and it is already
 * carried on the batch for the atom layout.
 */
import {
  CLASS_AMINO, CLASS_LIGAND, CLASS_NUCLEIC, CLASS_PROTEIN, PSEUDO_BETA_RESIDUES,
  contactBinsByPair,
} from "../heads/contact-threshold.js";

/** AF3's restype alphabet: 20 amino acids, X, gap, then RNA and DNA. */
const FIRST_NUCLEIC = 22;

/**
 * 🔴 AF3's OFFSET IS ZERO, WHICH IS A COINCIDENCE WORTH SAYING OUT LOUD.
 * `ARNDCQEGHILKMFPSTWYV` is one-letter alphabetical AND three-letter
 * alphabetical - ALA, ARG, ASN, ASP, CYS ... - so AF3's restype IS
 * `PSEUDO_BETA_RESIDUES`'s index. ESMFold2's is the same order plus two. Both
 * are asserted in tests rather than trusted, because a silently permuted
 * alphabet conforms in shape and folds something.
 */
export function af3ContactClasses(batch, tokens) {
  const classes = new Int32Array(tokens).fill(CLASS_PROTEIN);
  for (let token = 0; token < tokens; token += 1) {
    const restype = batch.aatype[token];
    if (restype >= FIRST_NUCLEIC) classes[token] = CLASS_NUCLEIC;
    else if (restype < PSEUDO_BETA_RESIDUES.length) classes[token] = CLASS_AMINO + restype;
  }
  for (const span of batch.ligandSpans ?? []) {
    for (let at = 0; at < span.count; at += 1) {
      if (span.from + at < tokens) classes[span.from + at] = CLASS_LIGAND;
    }
  }
  return classes;
}

/**
 * How many leading bins count as contact, per pair, under AF3's own rule.
 *
 * 🔴 A BIN COUNTS WHEN ITS TOP EDGE IS AT OR BELOW THE THRESHOLD, and the 63
 * breaks describe 64 bins - so the last bin is open-ended and its top has to be
 * extrapolated by one spacing rather than read from the array. ESMFold2's
 * borrowed grid counts a bin whose CENTRE is under instead; the two rules are
 * both prefixes and neither is the other, which is why the shared module takes
 * `binsUnder` from the caller.
 */
export function af3ContactBins(classes, tokens, breaks) {
  const spacing = breaks[breaks.length - 1] - breaks[breaks.length - 2];
  const top = (bin) => (bin < breaks.length ? breaks[bin]
    : breaks[breaks.length - 1] + spacing);
  return contactBinsByPair(classes, tokens, (angstroms) => {
    let count = 0;
    while (count <= breaks.length && top(count) <= angstroms + 1e-3) count += 1;
    return count;
  });
}
