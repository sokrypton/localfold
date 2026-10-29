/**
 * Steering AF3's sampler toward an epitope: a restraint applied to each denoised prediction.
 *
 * 🔴 WHY: A HOTSPOT IS A FILTER FOR EVERY MODEL BUT boltz2 (2026-10-28). boltz2 was trained with a
 * pocket input (cpu/design/contact.js hotspotPocket); the rest can only fold and then reject a binder
 * that missed - and on PD-L1 with AF3 7 runs of 12 ended with no cycle that hit. This asks the walk
 * itself, with geometry rather than weights, so it works for any model the sampler drives. The step
 * moves toward `denoised`, so displacing that prediction steers the trajectory without touching the
 * denoiser.
 *
 *   U = sum over pocket tokens p of relu(softmin_b d(p, b) - cutoff)
 *
 * d between token CENTROIDS (each token's live atoms), the soft minimum over binder tokens b - "some
 * part of the binder touches this residue", not "the whole binder wraps it".
 *
 * 🔴 THE FIRST VERSION WAS ProteinHunter2's ph2/steering.py, AND IT TORE THE BINDER APART (2026-10-29).
 * That one (yehlincho/ProteinHunter2, permission granted; marked "planned" there, never measured)
 * squares the violation and takes one plain gradient step on every centroid. A step of weight 1 on a
 * square is twice the violation, applied to the three or four NEAREST residues of each side alone:
 * a binder 20 A off moved three residues ~20 A out of their chain and the epitope ~23 A out of the
 * target, and at weight 3 or 10 they moved 70 and 240 A and the energy ROSE. The denoiser put the
 * torn residues back the next step, so the fold barely changed - cycle 0 came out within 0.1 A of
 * the unsteered one at every weight - and over six seeds steering put the binder on the epitope LESS
 * often than not steering (docs/DESIGN-TODO.md). So, after Boltz's own contact guidance
 * (boltz/model/potentials: flat-bottom and LINEAR past the bound, its weight 1 / 0.5 / 0 schedule):
 * the energy is linear, the BINDER moves and the target does not, it moves as ONE BODY - a
 * translation of every binder atom, so no chain is broken for the denoiser to repair - and a step is
 * at most `weight` angstroms and never past the largest violation, so it cannot overshoot.
 */

export const STEER_CUTOFF = 8;
export const STEER_BETA = 2;

/**
 * @param {Float32Array} positions a denoised prediction, dense layout (`tokens * dense * 3`)
 * @param {ArrayLike<number>} atomMask per dense slot
 * @param {number} dense slots a token
 * @param {{binder: number[], pocket: number[], cutoff?: number, beta?: number}} steer token indices
 * @returns {{energy: number, push: number[], largest: number}} U, dU/dT for a translation T of the
 *   whole binder, and the largest single violation in angstroms
 */
export function steeringEnergy(positions, atomMask, dense, steer) {
  const cutoff = steer.cutoff ?? STEER_CUTOFF;
  const beta = steer.beta ?? STEER_BETA;
  const centroid = (token) => {
    const at = [0, 0, 0];
    let count = 0;
    for (let slot = 0; slot < dense; slot += 1) {
      const flat = token * dense + slot;
      if (!atomMask[flat]) continue;
      for (let k = 0; k < 3; k += 1) at[k] += positions[flat * 3 + k];
      count += 1;
    }
    if (count === 0) throw new Error(`steering names token ${token}, which has no atoms`);
    return at.map((v) => v / count);
  };
  const binder = steer.binder.map(centroid);
  const pocket = steer.pocket.map(centroid);
  const push = [0, 0, 0];
  let energy = 0;
  let largest = 0;
  for (const p of pocket) {
    const distance = binder.map((b) => Math.hypot(p[0] - b[0], p[1] - b[1], p[2] - b[2]));
    const lowest = Math.min(...distance);
    // the soft minimum, shifted by the hard one so the exponentials cannot underflow to log(0)
    const terms = distance.map((d) => Math.exp(-beta * (d - lowest)));
    const total = terms.reduce((a, b) => a + b, 0);
    const violation = lowest - Math.log(total) / beta - cutoff;
    if (violation <= 0) continue;
    energy += violation;
    largest = Math.max(largest, violation);
    binder.forEach((b, i) => {
      if (distance[i] === 0) return;
      // dU/dd_i is the softmax weight; a translation moves every b alike, dd_i/dT = (b - p) / d_i
      const scale = (terms[i] / total) / distance[i];
      for (let k = 0; k < 3; k += 1) push[k] += scale * (b[k] - p[k]);
    });
  }
  return { energy, push, largest };
}

/**
 * Boltz's contact-guidance schedule over the walk: full weight for the first quarter of the steps,
 * half to three quarters, none after, where it would bend geometry the denoiser has resolved.
 * Over the steps actually WALKED, so a warm start that joins at a low noise level is steered too.
 *
 * @param {number} progress 0 at the first step walked, 1 at the last
 */
export function steeringSchedule(progress) {
  return progress < 0.25 ? 1 : progress < 0.75 ? 0.5 : 0;
}

/**
 * Translates the binder of a denoised prediction down the restraint.
 *
 * @param {Float32Array} denoised
 * @param {ArrayLike<number>} atomMask
 * @param {number} dense
 * @param {{binder: number[], pocket: number[], weight: number, cutoff?: number, beta?: number}} steer
 *   `weight` is the longest translation a step takes, in angstroms
 * @param {number} progress through the walk, for steeringSchedule
 * @returns {Float32Array} a new array; `denoised` is not written
 */
export function steerDenoised(denoised, atomMask, dense, steer, progress) {
  const out = Float32Array.from(denoised);
  const rate = steer.weight * steeringSchedule(progress);
  if (rate === 0) return out;
  const { energy, push, largest } = steeringEnergy(denoised, atomMask, dense, steer);
  const norm = Math.hypot(...push);
  if (energy === 0 || norm === 0) return out;
  const length = Math.min(rate, largest);
  const shift = push.map((v) => -length * v / norm);
  for (const token of steer.binder) {
    for (let slot = 0; slot < dense; slot += 1) {
      const flat = token * dense + slot;
      if (!atomMask[flat]) continue;
      for (let k = 0; k < 3; k += 1) out[flat * 3 + k] += shift[k];
    }
  }
  return out;
}
