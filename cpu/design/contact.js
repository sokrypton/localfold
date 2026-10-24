/**
 * contact.js
 * ----------
 * Is the designed chain actually touching what it was designed against?
 *
 * 🔴 THE CHECK A BINDER HUNT NEEDS AND A FOLD'S OWN CONFIDENCE CANNOT GIVE. The
 * loop's objective for a complex is per-chain-pair ipTM, which the model
 * computes about its own prediction - so a binder that drifted off into solvent
 * can still be confidently predicted to be there, and scores accordingly. This
 * measures the geometry instead: how close the designed chain actually is.
 *
 * 🔴 AND IT IS WHAT MAKES AN EPITOPE MEAN ANYTHING FOR PROTEIN HUNTER. RFD4
 * conditions on hotspots during generation; Hunter has no conditioning at all,
 * so the only way to ask for an epitope is to FOLD and then reject a cycle whose
 * binder is somewhere else. That is the shape Yehlin's v2 uses, reimplemented
 * here rather than copied.
 *
 * 🔴 CA TO CA AT 8 A, which is `shared/heads/contact-threshold.js`'s
 * protein-protein convention and its measured F1 of 0.767. That file's cutoffs
 * are for REPRESENTATIVE atoms - a pseudo-beta where a residue has one - and CA
 * is the representative this module has, since a designed backbone is all the
 * prediction carries for a residue whose identity is still changing. So this is
 * the convention applied to the atoms available, not a tighter claim.
 */
import { CONTACT_ANGSTROMS } from "../../shared/heads/contact-threshold.js";
import { structureFromText } from "./mpnn/pdb.js";

export { CONTACT_ANGSTROMS };

/** CA is index 1 of the N, CA, C, O backbone `structureFromText` returns. */
const CA = 1;

/**
 * How close the designed chain comes to the rest, and to named residues of it.
 *
 * @param {string} pdb a prediction, as the fold writes it
 * @param {object} options
 * @param {string} options.chain the designed chain's id
 * @param {Record<string, number[]>} [options.residues] per chain, the residue
 *   numbers that must be touched - an epitope. Absent means "touch anything".
 * @param {number} [options.angstroms] the cutoff, defaulting to the convention
 * @returns {{contacts: number, nearest: number, wanted: number, touched: number[],
 *            missing: number[], chains: string[]}}
 *   `nearest` is Infinity when there is nothing to touch at all.
 */
export function contactOf(pdb, options) {
  const { chain, residues, angstroms = CONTACT_ANGSTROMS } = options;
  const structure = structureFromText(pdb, {});
  const chains = structure.chainList;
  const cutoff = angstroms * angstroms;

  const mine = [];
  const theirs = [];
  for (let index = 0; index < structure.L; index += 1) {
    if (structure.mask[index] !== 1) continue;
    const at = [structure.X[index * 12 + CA * 3], structure.X[index * 12 + CA * 3 + 1],
      structure.X[index * 12 + CA * 3 + 2]];
    if (structure.chainIds[index] === chain) mine.push(at);
    else theirs.push({ at, chain: structure.chainIds[index], residue: structure.resSeq[index] });
  }

  // 🔴 WANTED RESIDUES ARE FILTERED FROM WHAT IS THERE, not assumed present. A
  // hotspot naming a residue the prediction does not have is a thing to report -
  // "0 of 4 touched" is honest, silently asking for 3 is not.
  const asked = residues === undefined ? null : new Set(
    Object.entries(residues).flatMap(([id, list]) => list.map((residue) => `${id}:${residue}`)));
  const targets = asked === null ? theirs
    : theirs.filter((entry) => asked.has(`${entry.chain}:${entry.residue}`));

  let contacts = 0;
  let nearestSquared = Infinity;
  const touched = new Set();
  for (const a of mine) {
    for (const b of targets) {
      const dx = a[0] - b.at[0];
      const dy = a[1] - b.at[1];
      const dz = a[2] - b.at[2];
      const squared = dx * dx + dy * dy + dz * dz;
      if (squared < nearestSquared) nearestSquared = squared;
      if (squared <= cutoff) {
        contacts += 1;
        touched.add(`${b.chain}:${b.residue}`);
      }
    }
  }

  const wantedKeys = asked === null ? [] : [...asked];
  return {
    contacts,
    nearest: nearestSquared === Infinity ? Infinity : Math.sqrt(nearestSquared),
    wanted: wantedKeys.length,
    touched: [...touched],
    missing: wantedKeys.filter((key) => !touched.has(key)),
    chains,
  };
}

/**
 * A one-line reason to reject a cycle, or null to keep it.
 *
 * 🔴 EVERY REFUSAL CARRIES ITS MEASUREMENT, because that is this page's rule: a
 * verdict with the number removed is what the design page exists not to do.
 *
 * @param {string} pdb
 * @param {object} options as `contactOf`, plus:
 * @param {number} [options.minimum=1] how many hotspots must be touched
 * @returns {string|null}
 */
export function contactProblem(pdb, options) {
  const result = contactOf(pdb, options);
  const { angstroms = CONTACT_ANGSTROMS } = options;
  // Nothing to touch: a monomer design, which this check has no opinion about.
  if (result.chains.length < 2) return null;

  if (result.wanted > 0) {
    const need = Math.min(options.minimum ?? 1, result.wanted);
    const got = result.wanted - result.missing.length;
    if (got < need) {
      return `touches ${got}/${result.wanted} hotspots`
        + ` (nearest ${result.nearest === Infinity ? "none" : `${result.nearest.toFixed(1)} A`},`
        + ` needs ${angstroms} A)`;
    }
    return null;
  }

  if (result.contacts === 0) {
    return `the designed chain touches nothing`
      + ` (nearest ${result.nearest === Infinity ? "none" : `${result.nearest.toFixed(1)} A`},`
      + ` needs ${angstroms} A)`;
  }
  return null;
}
