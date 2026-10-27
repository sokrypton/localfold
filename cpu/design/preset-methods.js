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
  // 🔴 A PORT GAP, NOT A PROPERTY OF THE METHOD. LigandMPNN sees the ligand and
  // the fold changes, but nothing is designed AGAINST the pocket the way RFD4
  // means - because no folder in `FOLDERS` takes a pocket constraint. Genuine
  // Boltz-2 does take one (a binder chain plus the residues it should contact),
  // and boltz2 is published and shares `foldAf3` with the three folders that
  // are wired, so this is reachable: it needs a constraint path in
  // shared/af3/featurise, which has none, and boltz2 added to FOLDERS.
  sm_sam: "no folding model here takes a pocket constraint yet",
  // Labelled spans ask for a secondary structure, which Hunter cannot be told:
  // it folds whatever the sequence gives.
  custom_cath_monomer: "Protein Hunter cannot be asked for a secondary structure",
  // 🔴 BUILDABLE, AND STILL NOT OFFERED: what these two examples ARE is RFD4-Proteina's - partial
  // diffusion from the structure, inverse folding of its backbone - and their segments are only the
  // motif example's. Under Protein Hunter each ran as exactly that motif scaffold (the same hunt,
  // objective for objective, in check-design-page.py --presets), under a name promising otherwise.
  partial_diffusion_1prw: "partial diffusion is RFD4-Proteina's; here it would be the motif example again",
  inverse_folding_1prw: "this inverse folding is RFD4-Proteina's; here it would be the motif example again",
  // 🔴 THE BINDER AND THE DNA BINDER ARE NOT HERE ANY MORE (2026-10-27). Both were excluded for a
  // discrepancy between their contig and the crop they ship with - the contig is a label in the
  // source entry's numbering - and both run from their FILE's annotations, which is what the page
  // now loads them as (web/example-spans.js): a designed chain against the file's target, with the
  // file's hotspots on it, and a kept DNA strand folded as DNA.
};

/**
 * Can this method run this example?
 *
 * @param {string} id a preset id
 * @param {string} method "hunter" or "rfd4p"
 * @returns {boolean}
 */
export const HUNTER_WOULD_DIFFER = new Set(["partial_diffusion_1prw", "inverse_folding_1prw"]);

export function presetSuits(id, method) {
  // Every example is RFD4's own, so that method takes all of them.
  if (method !== "hunter") return true;
  return HUNTER_CANNOT[id] === undefined;
}
