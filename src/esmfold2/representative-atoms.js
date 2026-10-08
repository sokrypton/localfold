// Each token's representative atom for ESMFold2's contacts and certainty (CB, CA for glycine, a base atom for a
// nucleotide) - out of src/esmfold2/fold.js (which re-exports it), so the CUDA port's exporter takes it without the
// WebGPU fold behind it.
import { MOL_DNA, MOL_RNA } from "./featurise.js";

/** ESMFold2's own indices for the four purines: A, G, DA, DG. */
const PURINE_RESIDUES = new Set([23, 24, 28, 29]);

/**
 * The dense-layout atom each token is represented by - AF3's pseudo-beta.
 *
 * CB for an amino acid and CA for glycine, C4 for a purine and C2 for a
 * pyrimidine, and for a ligand the one heavy atom the token IS.
 *
 * 🔴 A NUCLEOTIDE'S REPRESENTATIVE IS NOT ITS FIRST ATOM, AND TAKING IT SO
 * COSTS A FACTOR OF SIX. This used to be CB, else CA, else the first atom -
 * which for a nucleotide has no CB and no CA and so lands on the phosphorus,
 * out at the backbone, while the bases that touch are in the middle of the
 * ring. Measured against real depositions by
 * `tools/calibrate-contact-cutoff.py`, which asks how well a threshold on the
 * representative distance reproduces real atomic contact (any heavy atom pair
 * under 5 A):
 *
 * | pair | CB/CA/first | AF3 pseudo-beta |
 * |---|---|---|
 * | protein-protein | 8 A, F1 0.767 | 8 A, **0.767** |
 * | nucleic-protein | 9 A, F1 0.558 | 10 A, **0.607** |
 * | ligand-nucleic | 8 A, F1 0.617 | 7 A, **0.764** |
 * | **nucleic-nucleic** | 14 A, F1 **0.125** | 9 A, **0.777** |
 *
 * The last row is the diagnosis: across a duplex the phosphates are eighteen
 * angstroms apart while the bases stack, so NO threshold recovers the contact
 * and the best arm is the widest one offered. It is a geometric fact and needs
 * no model to see - which is what makes it safe to fix without an oracle for
 * the convention ESMFold2 itself was trained on.
 *
 * 🔴 AND `named` MATCHES THE WHOLE FOUR-CHARACTER NAME, so "C4" does not match
 * C4', which every nucleotide also has and which is back out on the sugar.
 */
export function representativeAtoms(features, tokens) {
  const named = (atom, want) => {
    for (let i = 0; i < 4; i += 1) {
      const wanted = i < want.length ? want.charCodeAt(i) - 32 : 0;
      if (features.refAtomNameChars[atom * 4 + i] !== wanted) return false;
    }
    return true;
  };
  const alpha = new Int32Array(tokens).fill(-1);
  const beta = new Int32Array(tokens).fill(-1);
  const base = new Int32Array(tokens).fill(-1);
  const first = new Int32Array(tokens).fill(-1);
  for (let atom = 0; atom < features.atoms; atom += 1) {
    if (features.mask[atom] === 0) continue;
    const token = features.atomToToken[atom];
    if (first[token] < 0) first[token] = atom;
    if (named(atom, "CA")) alpha[token] = atom;
    if (named(atom, "CB")) beta[token] = atom;
    if (named(atom, PURINE_RESIDUES.has(features.residueType[token]) ? "C4" : "C2")) {
      base[token] = atom;
    }
  }
  const out = new Int32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    const nucleic = features.molType[token] === MOL_DNA
      || features.molType[token] === MOL_RNA;
    const pick = nucleic ? base[token]
      : (beta[token] >= 0 ? beta[token] : alpha[token]);
    out[token] = pick >= 0 ? pick : Math.max(0, first[token]);
  }
  return out;
}

/** Those atoms' coordinates out of a full atom array. */
export function gatherPositions(coordinates, slots, tokens) {
  const out = new Float32Array(tokens * 3);
  for (let token = 0; token < tokens; token += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      out[token * 3 + axis] = coordinates[slots[token] * 3 + axis];
    }
  }
  return out;
}
