/**
 * preset-methods.js
 * -----------------
 * Which worked examples each method can actually run.
 *
 * 🔴 THE EXAMPLES ARE RFD4-PROTEINA'S, AND PROTEIN HUNTER CAN RUN MOST BUT NOT
 * ALL OF THEM. Offering a reader an example the selected method will refuse is
 * the same fault as a control that does nothing: the page invites a click and
 * then explains why it was wrong. So the picker lists only what the method can
 * take, and this is the list it reads.
 *
 * 🔴 IT IS DECLARED HERE AND GATED AGAINST THE MEASUREMENT, which is the only
 * reason to trust it. Two of the four exclusions cannot be derived from a
 * preset's own fields at all - they are discrepancies between a contig and the
 * cropped structure it ships with, and nothing short of loading that file finds
 * them. `test/design-preset-methods.test.js` loads every example, translates
 * its contig onto its own structure, builds the job, and fails if this list and
 * the outcome disagree in either direction.
 *
 * Keyed by preset id, with the reason, because a reason that lives only in a
 * commit message is a reason nobody reads.
 */

/**
 * Why Protein Hunter cannot run a given example.
 *
 * A preset absent from this map is one Hunter CAN run.
 */
export const HUNTER_CANNOT = {
  // Hunter has no pocket conditioning: LigandMPNN sees the ligand and the fold
  // changes, but nothing is designed AGAINST the pocket the way RFD4 means.
  sm_sam: "Protein Hunter has no pocket conditioning",
  // Labelled spans ask for a secondary structure, which Hunter cannot be told:
  // it folds whatever the sequence gives.
  custom_cath_monomer: "Protein Hunter cannot be asked for a secondary structure",
  // 🔴 NOT A METHOD LIMIT - A DISCREPANCY IN THE EXAMPLE. Its crop records
  // residues A17-A131 and its contig asks for A18-132: the same count, offset
  // by one, so the last residue has no source and the range cannot be moved
  // onto the file. RFD4 never notices because it runs from the structure's
  // annotations rather than its contig.
  ppi_pdl1: "this example's contig and its structure disagree by one residue",
  // The same shape: the contig asks for B1-16 where the crop's chain B holds
  // ten residues.
  dna_7m5w: "this example's contig asks for more residues than its structure has",
};

/**
 * Can this method run this example?
 *
 * @param {string} id a preset id
 * @param {string} method "hunter" or "rfd4p"
 * @returns {boolean}
 */
export function presetSuits(id, method) {
  // Every example is RFD4's own, so that method takes all of them.
  if (method !== "hunter") return true;
  return HUNTER_CANNOT[id] === undefined;
}
