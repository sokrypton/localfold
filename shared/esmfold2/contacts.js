// ESMFold2's contact and certainty classes and their bin counts - out of webgpu/esmfold2/distogram-webgpu.js (which
// re-exports them), so the CUDA port's exporter and the CPU references take them without the WebGPU head behind them.
import { MOL_DNA, MOL_NONPOLYMER, MOL_PROTEIN, MOL_RNA } from "./featurise.js";
import {
  CLASS_AMINO, CLASS_LIGAND, CLASS_NUCLEIC, CLASS_PROTEIN, PSEUDO_BETA_RESIDUES, CONTACT_ANGSTROMS,
  contactAngstromsForClasses, contactBinsByPair,
} from "../heads/contact-threshold.js";

/** Borrowed from the disabled confidence head's own 128 bins. See above. */
export const CONTACT_EDGES = { minimum: 2, maximum: 52 };
/** What counts as a contact, which is the usual 8 A between pseudo-betas. */
// ...re-exported so callers of this head keep one import; the tables and the
// calibration behind them are in ../heads/contact-threshold.js, because AF3's
// distogram asks the identical question of the identical geometry.
export {
  CONTACT_ANGSTROMS, CONTACT_ANGSTROMS_BY_KIND, LIGAND_PROTEIN_ANGSTROMS,
} from "../heads/contact-threshold.js";

/**
 * The threshold one pair asks for, given both ends' kinds and residue types.
 *
 * 🔴 ONE FUNCTION, BECAUSE A METRIC'S TWO HALVES MUST ASK THE SAME QUESTION. A
 * checker computing "actual" at 8 A against a map computing "predicted" at 7
 * reports a precision about nothing.
 */
/**
 * How sharply the distogram knows a distance, per residue - swept, not chosen.
 *
 * 🔴 THIS IS NOT A pLDDT AND MUST NEVER BE SHOWN AS ONE. This checkpoint has no
 * confidence head: 820 tensors and not one named confidence, plddt, pae or pde,
 * and `model.confidence_head` is None on the loaded model. What this is, is an
 * ORDERING - "the model knows where this residue goes more precisely than that
 * one" - with nothing to calibrate a NUMBER against.
 *
 * 🔴 AND THE THREE CONSTANTS ARE THE OUTCOME OF AN 11,400-ARM SWEEP OVER 60
 * FOLDS, not a guess. `tools/gpu/probe-esmfold2-confidence.js` scores per-pair
 * measures against per-residue lDDT-Ca, with targets corrupted at 0, 15 and 40%
 * so the label has range at all - 43 of 46 real targets fold above 0.9. The
 * winners, ranked on realistic rates by their WORST fold:
 *
 * | arm | median Spearman | worst fold |
 * |---|---|---|
 * | mode 3, sep 4, cut 12 | 0.504 | 0.310 |
 * | **mode 2, sep 3, cut 12** | **0.538** | **0.307** |
 * | mode 1.5, sep 3, cut 12 | 0.541 | 0.300 |
 *
 * against a buriedness baseline of about 0.19-0.26.
 *
 * 🔴 THE RADIUS IS IN ANGSTROMS BECAUSE THE BIN GRID IS BORROWED. `CONTACT_EDGES`
 * comes from the DISABLED confidence head, so "the height of the mode" is partly
 * an artefact of a 0.39 A grid nobody chose; the mass within 2 A of it is not.
 * Radius 0 - the exact bin - measured 0.415 against radius 2's 0.408 on clean
 * targets and is far less robust on corrupted ones.
 *
 * 🔴 AND THE CUTOFF IS ON THE PAIR, NOT ON THE BINS. Excluding pairs the model
 * places beyond 12 A is worth about 0.10 of Spearman; excluding the non-contact
 * BINS - ColabDesign's `con` loss, which is a design objective rather than a
 * confidence - scores 0.175, BELOW the buriedness baseline. The two sound alike
 * and are opposite.
 */
export const CERTAINTY = { radius: 2, separation: 3, cutoff: 12 };

/**
 * The two numbers the partner rule needs, one pair per token: `asymId` and
 * `residueIndex`.
 *
 * 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM, SO A SEPARATION ON THE TOKEN INDEX
 * MEANS NOTHING THERE. Every atom of a component shares one asym id and one
 * residue number (`featurise.js` writes 1), so this rule drops a ligand's whole
 * self-block - which is right, because its internal geometry comes from the
 * CCD conformer that was handed to the model and is not a prediction at all.
 * Measured on ubiquitin plus ATP, 76 residues and 31 atoms:
 *
 * | | no ligand | with ATP, token gap | with ATP, this rule |
 * |---|---|---|---|
 * | mean certainty, PROTEIN tokens | 0.9496 | 0.8989 | (see check-esmfold2-certainty) |
 *
 * ...a 0.05 shift on the protein's own numbers, with one residue moving 0.47,
 * caused by a molecule the protein's confidence should not depend on.
 *
 * 🔴 AND IT IS EXACTLY THE OLD RULE FOR ONE UNMODIFIED PROTEIN CHAIN, which is
 * what every measurement behind `CERTAINTY` was made on: there the residue
 * number and the token index differ by a constant, so their differences agree.
 * A complex changes: two tokens in different chains are no longer excluded for
 * being near each other in the array, which they never should have been.
 */
export function partnerKeys({ asymId, residueIndex, molType }, tokens) {
  const keys = new Int32Array(tokens * 4);
  for (let token = 0; token < tokens; token += 1) {
    keys[token * 4] = asymId[token];
    keys[token * 4 + 1] = residueIndex[token];
    // ...0 protein, 1 nucleic, 2 nonpolymer. See PARTNER_ANGSTROMS.
    const mol = molType?.[token] ?? MOL_PROTEIN;
    keys[token * 4 + 2] = mol === MOL_NONPOLYMER ? 2 : (mol === MOL_PROTEIN ? 0 : 1);
  }
  return keys;
}

/**
 * How far away a partner can be and still say something, by what the PARTNER
 * is - and which tokens may be partners at all.
 *
 * 🔴 AF3's OWN lDDT IS SHAPED THIS WAY, AND IT IS NOT SYMMETRIC.
 * `all_atom_plddt_loss` in OpenFold3 builds its pair mask as
 *
 *     (dx_gt < 15) * protein_atom_mask[..., None, :]
 *   + (dx_gt < 30) * nucleotide_atom_mask[..., None, :]
 *
 * - the radius is chosen by the kind of the atom in the SECOND index, the one
 * doing the scoring, and a ligand atom appears in neither term. Its `rep_index`
 * says the same thing from the other side: CA for a standard protein residue,
 * C1' for a standard nucleotide, and a padding sentinel for a ligand or an
 * atomized residue, so those contribute no representative atom. **Every atom is
 * SCORED; only polymer representatives do the SCORING.**
 *
 * That is what "treat a ligand like a protein" actually means, and it removes
 * the fallback by construction rather than by adding a tier: a ligand token has
 * partners - the polymer around it - under the same rule as everyone else.
 *
 * 🔴 AND THE PROTEIN RADIUS STAYS AT THE ONE THAT WAS SWEPT HERE. AF3's 15 A is
 * for a different quantity (a distance-difference test against a true
 * structure, not a distogram's peakedness), and this repository's own sweep
 * peaked at 12-14 A over 11,400 arms - close enough to be reassuring and not a
 * reason to move. What is taken from AF3 is the SHAPE: a nucleotide reaches
 * twice as far, because a base pair's partners are further off than a side
 * chain's.
 */
export const PARTNER_ANGSTROMS = { protein: 12, nucleic: 24, ligand: 12 };

/**
 * 🔴 AND A LIGAND SCORES AS WELL AS BEING SCORED, WHICH IS WHERE THIS PARTS
 * COMPANY WITH AF3's lDDT. That loss admits only protein and nucleotide atoms
 * as the partner index, and a ligand contributes no representative at all -
 * which is right for a training loss over structures that always have a
 * polymer, and wrong here the moment somebody folds a ligand ALONE. A heme on
 * its own had no eligible partner for any of its atoms, so every token reported
 * -1 and the whole molecule rendered as the worst colour on the scale, next to
 * a contact map that was confident about it.
 *
 * So there is no chemistry test on who may score. The exclusion is the one
 * thing being asked - is this partner TRIVIALLY close - and it has two forms: a
 * covalent bond, and a sequence neighbourhood for a chain that has a sequence.
 * A ligand has no sequence, so bonds are the whole of its exclusion, and its
 * remaining pairs are the ones that say something. What stays chemistry-shaped
 * is only how far a partner may REACH, which is AF3's own asymmetry.
 */


export function contactAngstromsFor(molTypeI, molTypeJ, residueTypeI, residueTypeJ) {
  return contactAngstromsForClasses(esmfold2Class(molTypeI, residueTypeI),
                                    esmfold2Class(molTypeJ, residueTypeJ));
}

/**
 * ESMFold2's `molType` and `residueType` as a contact class.
 *
 * 🔴 THE OFFSET IS TWO AND IT IS WRITTEN DOWN HERE, not assumed at a call site.
 * ESMFold2 numbers residues by three-letter code from 2, which is
 * `PSEUDO_BETA_RESIDUES`'s own order - see the note there, and
 * test/esmfold2-certainty-partners.test.js, which asserts it rather than
 * trusting it.
 */
function esmfold2Class(molType, residueType) {
  if (molType === MOL_NONPOLYMER) return CLASS_LIGAND;
  if (molType !== MOL_PROTEIN) return CLASS_NUCLEIC;
  const at = residueType - 2;
  return at >= 0 && at < PSEUDO_BETA_RESIDUES.length ? CLASS_AMINO + at : CLASS_PROTEIN;
}

export function contactBinCountsByPair(molType, residueType, tokens, bins,
                                       edges = CONTACT_EDGES) {
  const classes = new Int32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    classes[token] = esmfold2Class(molType[token], residueType[token]);
  }
  // ...ESMFold2's grid counts a bin whose CENTRE is under the threshold; AF3's
  // counts one whose top edge is. Both are prefixes and neither is the other.
  return contactBinsByPair(classes, tokens,
    (angstroms) => contactBinCount(bins, edges, angstroms));
}

/**
 * The same counts for a distogram with AlphaFold 3's bins - what the RELEASED ESMFold2 and ESMFold2-Fast carry
 * (64 bins, breaks linspace(2.3125, 21.6875, 63); the experimental tier's is 128 over 2-52). AF3's rule, not
 * the centre rule above: a bin counts when its top edge is under the threshold, the last bin's top one
 * spacing past the last break (shared/af3/featurise/contact-classes.js af3ContactBins).
 */
export const AF3_DISTOGRAM_BREAKS = Array.from({ length: 63 }, (_, k) => 2.3125 + (k * (21.6875 - 2.3125)) / 62);
export function contactBinCountsByPairBreaks(molType, residueType, tokens, breaks = AF3_DISTOGRAM_BREAKS) {
  const classes = new Int32Array(tokens);
  for (let token = 0; token < tokens; token += 1) classes[token] = esmfold2Class(molType[token], residueType[token]);
  const spacing = breaks[breaks.length - 1] - breaks[breaks.length - 2];
  const top = (bin) => (bin < breaks.length ? breaks[bin] : breaks[breaks.length - 1] + spacing);
  return contactBinsByPair(classes, tokens, (angstroms) => {
    let count = 0;
    while (count <= breaks.length && top(count) <= angstroms + 1e-3) count += 1;
    return count;
  });
}

/** How many of `bins` have their centre inside `CONTACT_ANGSTROMS`. */
export function contactBinCount(bins, edges = CONTACT_EDGES,
                                threshold = CONTACT_ANGSTROMS) {
  const width = (edges.maximum - edges.minimum) / bins;
  let count = 0;
  for (let bin = 0; bin < bins; bin += 1) {
    if (edges.minimum + (bin + 0.5) * width < threshold) count += 1;
  }
  return count;
}
