/**
 * validate.js
 * -----------
 * Does the finished design fold back to the structure it was supposed to be?
 *
 * 🔴 THE ONLY NUMBER IN A DESIGN RUN THAT IS NOT SELF-REPORTED. Protein Hunter
 * optimises a folding model's opinion of its own design: the objective IS that
 * model's confidence, so a high score says the model agrees with itself.
 * RFD4-Proteina is worse off - it generates a structure and scores nothing at
 * all. Folding the finished SEQUENCE with a model that had no hand in making it
 * gives a confidence nothing was optimised against, and an RMSD to the design it
 * is supposed to reproduce.
 *
 * 🔴 THIS IS THE SELF-CONSISTENCY CHECK the field runs on every de novo design,
 * and it is the number people actually report. A design that scores 0.9 to its
 * own folder and 4 Å away from its own backbone under another is not a design.
 *
 * Pure: two PDB strings in, numbers out. The fold itself belongs to the caller.
 */
import { superimposeCoords3D } from "../../shared/rfd4p/output.js";
import { structureFromText } from "./mpnn/pdb.js";

/** CA is index 1 of the N, CA, C, O backbone `structureFromText` returns. */
const CA = 1;

/**
 * The alpha carbons of one chain, in residue order.
 *
 * 🔴 ONE CHAIN, because the designed chain is the thing being checked. Taking
 * the whole complex would superpose the TARGET - which both structures share and
 * which neither method designed - and report a small RMSD for any binder at all.
 */
function alphaCarbons(pdb, chain) {
  const structure = structureFromText(pdb, {});
  const out = [];
  for (let index = 0; index < structure.L; index += 1) {
    if (structure.mask[index] !== 1) continue;
    if (chain !== undefined && structure.chainIds[index] !== chain) continue;
    out.push(structure.X[index * 12 + CA * 3],
      structure.X[index * 12 + CA * 3 + 1],
      structure.X[index * 12 + CA * 3 + 2]);
  }
  return Float32Array.from(out);
}

/**
 * How far the validation fold sits from the design.
 *
 * @param {string} designPdb the structure the design was made as
 * @param {string} validationPdb the same sequence, folded again by another model
 * @param {string} [chain] the designed chain
 * @returns {{rmsd: number, residues: number, comparable: boolean}}
 *   `comparable` is false when the two do not describe the same number of
 *   residues, which makes an RMSD between them meaningless rather than large.
 */
export function selfConsistency(designPdb, validationPdb, chain) {
  const design = alphaCarbons(designPdb, chain);
  const validation = alphaCarbons(validationPdb, chain);
  const residues = Math.min(design.length, validation.length) / 3;
  // 🔴 A LENGTH MISMATCH IS NOT A BIG RMSD, IT IS NO RMSD. superimposeCoords3D
  // would happily align the first N of each and return a number; that number
  // compares residue 1 to residue 1 of two different things.
  const comparable = design.length === validation.length && residues > 0;
  if (!comparable) return { rmsd: Number.NaN, residues, comparable: false };
  return { rmsd: superimposeCoords3D(design, validation).rmsd, residues, comparable: true };
}

/**
 * What the validation fold found, as the page says it.
 *
 * 🔴 THE NUMBER AND ITS VERDICT TOGETHER. 2 Å is the threshold the field uses
 * for a self-consistent de novo design, and a line reading "3.9 A" without it
 * means nothing to anyone who has not read that literature.
 *
 * @param {object} result
 * @param {number} [result.rmsd]
 * @param {boolean} [result.comparable]
 * @param {object} [result.confidence] the validator's own, if it has one
 * @param {string} [result.model] which model did the validating
 * @returns {string}
 */
export function describeValidation(result) {
  const { model = "a second model", confidence = {}, rmsd, comparable } = result;
  const parts = [];
  if (Number.isFinite(confidence.meanPlddt)) {
    parts.push(`pLDDT ${confidence.meanPlddt.toFixed(1)}`);
  }
  if (Number.isFinite(confidence.iptm)) parts.push(`ipTM ${confidence.iptm.toFixed(3)}`);
  if (comparable === false) {
    parts.push("RMSD not comparable - the two folds have different lengths");
  } else if (Number.isFinite(rmsd)) {
    parts.push(`RMSD ${rmsd.toFixed(2)} A${rmsd <= 2 ? "" : " - above the 2 A"
      + " a self-consistent design is usually held to"}`);
  }
  return parts.length === 0 ? `${model} returned nothing to judge by`
    : `${model}: ${parts.join(", ")}`;
}
