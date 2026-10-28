/**
 * Steering AF3's sampler toward an epitope: a restraint's gradient on each denoised prediction.
 *
 * 🔴 WHY: A HOTSPOT IS A FILTER FOR EVERY MODEL BUT boltz2 (2026-10-28). boltz2 was trained with a
 * pocket input (cpu/design/contact.js hotspotPocket); the rest can only fold and then reject a binder
 * that missed - and on PD-L1 with AF3 7 runs of 12 ended with no cycle that hit. This asks the walk
 * itself, with geometry rather than weights, so it works for any model the sampler drives.
 *
 * ProteinHunter2's ph2/steering.py (yehlincho/ProteinHunter2, permission granted), reimplemented: the
 * step moves toward `denoised`, so displacing that prediction by one descent step on an energy steers
 * the trajectory without touching the denoiser.
 *
 *   U = sum over pocket tokens p of relu(softmin_b d(p, b) - cutoff)^2
 *
 * d between token CENTROIDS (each token's live atoms, so a large residue is not weighted by its atom
 * count), the soft minimum over binder tokens b - "some part of the binder touches this residue", not
 * "the whole binder wraps it", which an all-pairs term would ask - and nothing inside the cutoff. The
 * step is `weight * min(1, tHat / sigmaMax)`: large while the structure is a blob and the restraint is
 * cheap to satisfy, nearly none at the end, where it would bend geometry the denoiser has resolved.
 */

export const STEER_CUTOFF = 8;
export const STEER_BETA = 2;

/**
 * @param {Float32Array} positions a denoised prediction, dense layout (`tokens * dense * 3`)
 * @param {ArrayLike<number>} atomMask per dense slot
 * @param {number} dense slots a token
 * @param {{binder: number[], pocket: number[], cutoff?: number, beta?: number}} steer token indices
 * @returns {{energy: number, gradient: Float32Array}} U and dU/dpositions
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
    return { at: at.map((v) => v / count), count };
  };
  const binder = steer.binder.map((token) => ({ token, ...centroid(token) }));
  const pocket = steer.pocket.map((token) => ({ token, ...centroid(token) }));
  const pull = new Map();
  const add = (token, vector) => {
    const sum = pull.get(token) ?? [0, 0, 0];
    for (let k = 0; k < 3; k += 1) sum[k] += vector[k];
    pull.set(token, sum);
  };
  let energy = 0;
  for (const p of pocket) {
    const distance = binder.map((b) => Math.hypot(p.at[0] - b.at[0], p.at[1] - b.at[1], p.at[2] - b.at[2]));
    const lowest = Math.min(...distance);
    // the soft minimum, shifted by the hard one so the exponentials cannot underflow to log(0)
    const terms = distance.map((d) => Math.exp(-beta * (d - lowest)));
    const total = terms.reduce((a, b) => a + b, 0);
    const soft = lowest - Math.log(total) / beta;
    const violation = soft - cutoff;
    if (violation <= 0) continue;
    energy += violation * violation;
    binder.forEach((b, i) => {
      if (distance[i] === 0) return;
      // dU/dd_i = 2 violation * softmax weight; dd/dp = (p - b) / d, dd/db its negative
      const scale = 2 * violation * (terms[i] / total) / distance[i];
      const direction = [0, 1, 2].map((k) => scale * (p.at[k] - b.at[k]));
      add(p.token, direction);
      add(b.token, direction.map((v) => -v));
    });
  }
  const gradient = new Float32Array(positions.length);
  const counts = new Map([...binder, ...pocket].map((entry) => [entry.token, entry.count]));
  for (const [token, sum] of pull) {
    // a centroid is the mean of its atoms, so each atom carries 1/count of the centroid's gradient
    const count = counts.get(token);
    for (let slot = 0; slot < dense; slot += 1) {
      const flat = token * dense + slot;
      if (!atomMask[flat]) continue;
      for (let k = 0; k < 3; k += 1) gradient[flat * 3 + k] = sum[k] / count;
    }
  }
  return { energy, gradient };
}

/**
 * One descent step on the restraint, applied to a denoised prediction.
 *
 * @param {Float32Array} denoised
 * @param {ArrayLike<number>} atomMask
 * @param {number} dense
 * @param {{binder: number[], pocket: number[], weight: number, cutoff?: number, beta?: number}} steer
 * @param {number} tHat this step's noise level
 * @param {number} sigmaMax the schedule's first level
 * @returns {Float32Array} a new array; `denoised` is not written
 */
export function steerDenoised(denoised, atomMask, dense, steer, tHat, sigmaMax) {
  const { gradient } = steeringEnergy(denoised, atomMask, dense, steer);
  const rate = steer.weight * Math.min(1, Math.max(0, tHat / sigmaMax));
  const out = Float32Array.from(denoised);
  for (let index = 0; index < out.length; index += 1) out[index] -= rate * gradient[index];
  return out;
}
