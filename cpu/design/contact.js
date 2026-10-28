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
import { CONTACT_ANGSTROMS, CONTACT_ANGSTROMS_BY_KIND } from "../../shared/heads/contact-threshold.js";
import { structureFromText } from "./mpnn/pdb.js";
import { POLYTYPE } from "./mpnn/na.js";

export { CONTACT_ANGSTROMS };

/**
 * CA is index 1 of the N, CA, C, O backbone `structureFromText` returns - and a nucleotide's C1'
 * sits in that slot under `nucleicAsResidues` (NUCLEIC_TRACE in ./mpnn/pdb.js).
 */
const CA = 1;

/**
 * 🔴 DNA AND RNA ARE IN THE PARSE (2026-10-28). A protein-only parse drops every nucleic chain, so a
 * binder designed against DNA alone had nothing to touch and the check returned "no opinion" on
 * every cycle, and a designed DNA strand was not in the structure at all. Each pair is held to its
 * kinds' calibrated cutoff (`CONTACT_ANGSTROMS_BY_KIND`: 8 A protein-protein, 10 nucleic-protein,
 * 9 nucleic-nucleic), measured on representative atoms as this is on CA and C1'.
 */
const kindOf = (polytype) => (polytype === POLYTYPE.DNA || polytype === POLYTYPE.RNA ? "nucleic" : "protein");
const cutoffOf = (a, b) => CONTACT_ANGSTROMS_BY_KIND[[a, b].sort().join("-")];

/**
 * How close the designed chain comes to the rest, and to named residues of it.
 *
 * @param {string} pdb a prediction, as the fold writes it
 * @param {object} options
 * @param {string} options.chain the designed chain's id
 * @param {Record<string, number[]>} [options.residues] per chain, the residue
 *   numbers that must be touched - an epitope. Absent means "touch anything".
 * @param {number} [options.angstroms] one cutoff for every pair; absent, each pair takes its kinds'
 * @returns {{contacts: number, nearest: number, nearestCutoff: number, wanted: number, touched: number[],
 *            missing: number[], chains: string[]}}
 *   `nearest` is Infinity when there is nothing to touch at all.
 */
export function contactOf(pdb, options) {
  const { chain, residues, angstroms } = options;
  const structure = structureFromText(pdb, { nucleicAsResidues: true });
  const chains = structure.chainList;

  const mine = [];
  const theirs = [];
  for (let index = 0; index < structure.L; index += 1) {
    if (structure.mask[index] !== 1) continue;
    const at = [structure.X[index * 12 + CA * 3], structure.X[index * 12 + CA * 3 + 1],
      structure.X[index * 12 + CA * 3 + 2]];
    const kind = kindOf(structure.polytype[index]);
    if (structure.chainIds[index] === chain) mine.push({ at, kind });
    else theirs.push({ at, kind, chain: structure.chainIds[index], residue: structure.resSeq[index] });
  }

  // 🔴 WANTED RESIDUES ARE FILTERED FROM WHAT IS THERE, not assumed present. A
  // hotspot naming a residue the prediction does not have is a thing to report -
  // "0 of 4 touched" is honest, silently asking for 3 is not.
  // 🔴 AND AN EMPTY SET NAMES NO HOTSPOTS: read as a set to touch it left no target residue at all,
  // so every cycle was refused as touching nothing
  const named = residues === undefined ? []
    : Object.entries(residues).flatMap(([id, list]) => list.map((residue) => `${id}:${residue}`));
  const asked = named.length === 0 ? null : new Set(named);
  const targets = asked === null ? theirs
    : theirs.filter((entry) => asked.has(`${entry.chain}:${entry.residue}`));

  let contacts = 0;
  let nearestSquared = Infinity;
  // ...and the cutoff the nearest pair is held to, which is what a refusal quotes
  let nearestCutoff = angstroms ?? CONTACT_ANGSTROMS;
  const touched = new Set();
  for (const a of mine) {
    for (const b of targets) {
      const dx = a.at[0] - b.at[0];
      const dy = a.at[1] - b.at[1];
      const dz = a.at[2] - b.at[2];
      const squared = dx * dx + dy * dy + dz * dz;
      const cutoff = angstroms ?? cutoffOf(a.kind, b.kind);
      if (squared < nearestSquared) { nearestSquared = squared; nearestCutoff = cutoff; }
      if (squared <= cutoff * cutoff) {
        contacts += 1;
        touched.add(`${b.chain}:${b.residue}`);
      }
    }
  }

  const wantedKeys = asked === null ? [] : [...asked];
  return {
    contacts,
    nearest: nearestSquared === Infinity ? Infinity : Math.sqrt(nearestSquared),
    nearestCutoff,
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
  const angstroms = result.nearestCutoff;
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
