/**
 * What "in contact" means, given what the two tokens ARE.
 *
 * 🔴 EIGHT ANGSTROMS IS A PSEUDO-BETA CONVENTION AND MOST PAIRS ARE NOT TWO
 * RESIDUES. A distogram predicts a distance between one representative atom per
 * token - a residue's pseudo-beta, a ligand's own heavy atom - so the threshold
 * that means "these touch" depends on how much reach each end's representative
 * is standing in for. Calibrated rather than reasoned, by
 * `tools/calibrate-contact-cutoff.py`: real depositions, real atomic contact as
 * the ground truth (any heavy atom pair under 5 A), sweeping which
 * representative distance reproduces it. 14 entries, 41,000 real contacts.
 *
 * 🔴 IT LIVES HERE AND NOT BESIDE ONE MODEL'S HEAD, because ESMFold2's
 * distogram and AF3's are the same question asked of the same geometry. AF2's
 * is NOT: monomer and multimer are protein-only, every pair is two residues,
 * and 8 A is simply right there.
 */

/** The convention, still correct between two residues. */
export const CONTACT_ANGSTROMS = 8;

/**
 * ...and by kind, where at least one end is not a residue.
 *
 * | pair | cutoff | F1 | at 8 A |
 * |---|---|---|---|
 * | protein-protein | 8 A | 0.767 | the convention, confirmed |
 * | ligand-protein | 7 A | 0.707 | 0.629 |
 * | ligand-nucleic | 7 A | 0.764 | |
 * | nucleic-protein | 10 A | 0.607 | 0.444 |
 * | nucleic-nucleic | 9 A | 0.777 | |
 * | ligand-ligand | 5 A | **1.000** | 0.696 |
 *
 * 🔴 AND THE LIGAND-LIGAND ROW IS EXACT, WHICH IS THE POINT AND NOT A FLUKE.
 * Both representatives ARE the heavy atoms, so the representative distance is
 * not an approximation of the ground truth - it IS the ground truth, and 8 A
 * was doing nothing there but being the wrong definition. That holds whether
 * the two atoms are in one molecule or two; what differs between those is what
 * the number MEANS, since inside a molecule the geometry came from the CCD
 * conformer the model was handed.
 */
export const CONTACT_ANGSTROMS_BY_KIND = {
  "protein-protein": 8, "nucleic-protein": 10, "ligand-protein": 7,
  "nucleic-nucleic": 9, "ligand-nucleic": 7, "ligand-ligand": 5,
};

/**
 * ...and for a ligand against a RESIDUE, per residue.
 *
 * 🔴 ONE NUMBER CANNOT SERVE TWENTY SIDE CHAINS, because the representative
 * stands at a different depth in each: a ligand touching a tryptophan ring is
 * far from that CB, an alanine's heavy atoms barely reach past it. The measured
 * reach - the median pseudo-beta-to-ligand distance among pairs that really are
 * in contact - runs 4.26 A at cysteine to 7.28 A at arginine, monotonic in
 * side-chain length, and the best threshold tracks it. Pooled over every
 * ligand-protein pair:
 *
 * | rule | precision | recall | F1 |
 * |---|---|---|---|
 * | one threshold, 6 A | 0.819 | 0.586 | 0.683 |
 * | one threshold, 7 A | 0.631 | 0.803 | 0.707 |
 * | one threshold, 8 A | 0.458 | 0.932 | 0.614 |
 * | **per residue** | **0.740** | **0.824** | **0.780** |
 *
 * It DOMINATES rather than trading, which is what says the residues really do
 * want different numbers - and `--holdout` fits the table on half the entries
 * and scores it on the other half, 0.771 against the best single arm's 0.694,
 * because "twenty free parameters beat one" is what free parameters do.
 *
 * 🔴 AND ANY THRESHOLD AT OR UNDER 5 A IS PERFECTLY PRECISE FOR FREE, since the
 * representative IS one of the residue's own heavy atoms. Glycine and alanine
 * reading precision 1.000 is arithmetic, not a result; the whole question is
 * how much RECALL a side chain buys before precision goes.
 */
export const LIGAND_PROTEIN_ANGSTROMS = {
  ALA: 5, ARG: 8, ASN: 7, ASP: 7, CYS: 6, GLN: 7, GLU: 7, GLY: 5, HIS: 8,
  ILE: 6, LEU: 7, LYS: 7, MET: 8, PHE: 8, PRO: 6, SER: 6, THR: 6, TRP: 7,
  TYR: 8, VAL: 6,
};

/**
 * The twenty, three-letter alphabetical.
 *
 * 🔴 WHICH IS BOTH MODELS' ORDER, AND THAT IS WHY ONE TABLE SERVES THEM.
 * ESMFold2 numbers residues by three-letter code from 2; AF3's
 * `ARNDCQEGHILKMFPSTWYV` is one-letter but happens to be three-letter
 * alphabetical too, from 0. So each model's own index is this array's index
 * plus a constant, and `contactClass` is where that constant is written down
 * rather than assumed at a call site.
 */
export const PSEUDO_BETA_RESIDUES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN",
  "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR",
  "TRP", "TYR", "VAL"];

/**
 * A token's contact class: what the threshold table needs to know about it.
 *
 * Nucleic and ligand are one class each; a standard amino acid is its own,
 * because a ligand pair asks the residue; and anything else protein-shaped -
 * an unknown residue, a modified one - falls back to the kind's number rather
 * than borrowing a side chain it may not have.
 */
export const CLASS_NUCLEIC = 0;
export const CLASS_LIGAND = 1;
export const CLASS_AMINO = 2;
export const CLASS_PROTEIN = CLASS_AMINO + PSEUDO_BETA_RESIDUES.length;
export const CONTACT_CLASSES = CLASS_PROTEIN + 1;

/** `kind` is "protein", "nucleic" or "ligand"; `residue` a three-letter code. */
export function contactClass(kind, residue) {
  if (kind === "nucleic") return CLASS_NUCLEIC;
  if (kind === "ligand") return CLASS_LIGAND;
  const at = PSEUDO_BETA_RESIDUES.indexOf(residue);
  return at < 0 ? CLASS_PROTEIN : CLASS_AMINO + at;
}

const KIND_OF_CLASS = (klass) => (klass === CLASS_NUCLEIC ? "nucleic"
  : (klass === CLASS_LIGAND ? "ligand" : "protein"));

/** The threshold, in angstroms, for a pair of classes. */
export function contactAngstromsForClasses(a, b) {
  const kindA = KIND_OF_CLASS(a);
  const kindB = KIND_OF_CLASS(b);
  if (kindA === "ligand" && kindB === "protein" && b >= CLASS_AMINO && b < CLASS_PROTEIN) {
    return LIGAND_PROTEIN_ANGSTROMS[PSEUDO_BETA_RESIDUES[b - CLASS_AMINO]];
  }
  if (kindB === "ligand" && kindA === "protein" && a >= CLASS_AMINO && a < CLASS_PROTEIN) {
    return LIGAND_PROTEIN_ANGSTROMS[PSEUDO_BETA_RESIDUES[a - CLASS_AMINO]];
  }
  return CONTACT_ANGSTROMS_BY_KIND[[kindA, kindB].sort().join("-")]
    ?? CONTACT_ANGSTROMS;
}

/**
 * How many leading bins count as contact, per pair.
 *
 * 🔴 A COUNT AND NOT A MASK, WHICH IS ONLY SOUND BECAUSE THE BINS ARE ORDERED.
 * Both heads bin distance increasingly, so "under the threshold" is a prefix -
 * and a prefix is one integer rather than a per-bin array per pair.
 *
 * `binsUnder(angstroms)` is the caller's, because the two heads disagree about
 * where a bin's edge is: AF3 counts a bin whose TOP edge is at or below the
 * threshold, ESMFold2's borrowed grid counts one whose CENTRE is.
 */
export function contactBinsByPair(classes, tokens, binsUnder) {
  const cache = new Map();
  const out = new Int32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const key = classes[i] * CONTACT_CLASSES + classes[j];
      if (!cache.has(key)) {
        cache.set(key, binsUnder(contactAngstromsForClasses(classes[i], classes[j])));
      }
      out[i * tokens + j] = cache.get(key);
    }
  }
  return out;
}
