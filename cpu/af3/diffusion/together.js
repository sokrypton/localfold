// Folding with several models at once: the arithmetic the GPU sampler applies to their answers.
import { weightedRigidAlign } from "./rigid-align.js";

/**
 * The mean of several models' denoised answers to one noisy structure, in this model's layout.
 *
 * 🔴 EACH IS FITTED ONTO THIS MODEL'S FIRST: a denoiser predicts in the frame of the noise it was
 * handed, so the answers start near-superposed, but nothing guarantees it - a mean of two copies a
 * few degrees apart is a smeared copy. Atom slots one model keeps and another drops (a terminal
 * OXT) take the models that have them.
 *
 * @param {Float32Array} own this model's denoised positions, dense
 * @param {ArrayLike<number>} mask this model's atom mask
 * @param {number} atoms tokens * dense
 * @param {{positions: Float32Array, atomMask: ArrayLike<number>}[]} others
 * @returns {Float32Array}
 */
export function meanOfDenoisers(own, mask, atoms, others) {
  const sum = Float64Array.from(own);
  const count = new Float64Array(atoms).fill(1);
  for (const other of others) {
    if (other.positions.length !== atoms * 3) {
      throw new Error(`a partner denoised ${other.positions.length / 3} atom slots where this fold has ${atoms}`
        + " - models folded together must lay the input out alike");
    }
    const both = Float32Array.from({ length: atoms }, (_, atom) => (mask[atom] && other.atomMask[atom] ? 1 : 0));
    const fitted = weightedRigidAlign(other.positions, own, both, atoms);
    for (let atom = 0; atom < atoms; atom += 1) {
      if (!other.atomMask[atom]) continue;
      count[atom] += 1;
      for (let k = 0; k < 3; k += 1) sum[atom * 3 + k] += fitted[atom * 3 + k];
    }
  }
  return Float32Array.from(sum, (value, index) => value / count[Math.floor(index / 3)]);
}
