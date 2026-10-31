/**
 * Motif inpainting on the denoised estimate - cuda/af3's `--motif` (sampler.cuh inpaintMotif) for the
 * WebGPU sampler and the CPU one.
 *
 * At each diffusion step the listed atoms of the denoiser's answer are replaced by their target
 * coordinates, the target rigidly fitted onto the answer's own copies of those atoms first: the motif
 * keeps its internal geometry exactly and the fold chooses where it sits. The walk then steps toward
 * the edited answer. A motif is named as the native file names it - a token and an atom name - and a
 * NEGATIVE token is a whole chain (asymId `-token`), for a ligand whose atoms are a token each and are
 * found by name across the chain.
 *
 * 🔴 IT EDITS THE PREDICTION, NOT THE WALK, as steering does (steering.js): the update moves toward
 * what it is handed, so the denoiser itself is untouched and the noisy positions stay the sampler's.
 */
import { weightedRigidAlign } from "./rigid-align.js";

/** The atom name at a dense slot, from the batch's packed name characters (4 a slot, offset by 32). */
function nameAt(chars, slot) {
  let name = "";
  for (let k = 0; k < 4; k += 1) {
    const c = chars[slot * 4 + k];
    if (c > 0) name += String.fromCharCode(c + 32);
  }
  return name;
}

/**
 * A motif's entries as dense slots of `batch` and their target coordinates.
 *
 * @param {object} batch the featurised batch (asymId, refAtomNameChars, predDenseAtomMask, shape)
 * @param {{token: number, atom: string, xyz: number[]}[]} entries
 * @returns {{slots: Int32Array, xyz: Float32Array}}
 */
export function motifSlots(batch, entries) {
  const tokens = batch.asymId.length;
  const dense = batch.predDenseAtomMask.length / tokens;
  const slots = new Int32Array(entries.length);
  const xyz = new Float32Array(entries.length * 3);
  entries.forEach(({ token, atom, xyz: at }, index) => {
    if (token >= tokens) throw new Error(`a motif names token ${token} of ${tokens}`);
    const found = [];
    for (let t = token >= 0 ? token : 0; t < (token >= 0 ? token + 1 : tokens); t += 1) {
      if (token < 0 && batch.asymId[t] !== -token) continue;
      for (let a = 0; a < dense; a += 1) {
        const slot = t * dense + a;
        if (batch.predDenseAtomMask[slot] && nameAt(batch.refAtomNameChars, slot) === atom) found.push(slot);
      }
    }
    if (found.length !== 1) {
      throw new Error(`a motif names ${token >= 0 ? `token ${token}` : `chain ${-token}`}'s ${atom}, `
        + `which is ${found.length} atoms of the input`);
    }
    slots[index] = found[0];
    xyz.set(at, index * 3);
  });
  if (slots.length < 3) throw new Error(`a motif of ${slots.length} atoms: a rigid fit needs three`);
  return { slots, xyz };
}

/**
 * `denoised` with the motif's atoms replaced: the target fitted onto the answer's own copies of them.
 * A copy; `denoised` is the head's.
 *
 * @param {Float32Array} denoised dense positions, [atoms * 3]
 * @param {{slots: Int32Array, xyz: Float32Array}} motif
 * @returns {Float32Array}
 */
export function inpaintDenoised(denoised, motif) {
  const k = motif.slots.length;
  const answer = new Float32Array(k * 3);
  motif.slots.forEach((slot, index) => answer.set(denoised.subarray(slot * 3, slot * 3 + 3), index * 3));
  const fitted = weightedRigidAlign(motif.xyz, answer, new Float32Array(k).fill(1), k);
  const out = Float32Array.from(denoised);
  motif.slots.forEach((slot, index) => out.set(fitted.subarray(index * 3, index * 3 + 3), slot * 3));
  return out;
}
