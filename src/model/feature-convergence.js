/**
 * How far a trunk representation moved between two recycles.
 *
 * 🔴 WHY THIS EXISTS BESIDE recycle-convergence.js. That one is ColabFold's
 * `compute_tol` - the RMS change of every C-alpha pair distance - and it needs
 * a STRUCTURE. AF2 has one each recycle because its trunk feeds atom positions
 * back; AF3, OpenDDE and ESMFold2 recycle the single and pair representations
 * alone and run the sampler once at the end, so there are no coordinates to
 * compare until every recycle is already paid for. For those models the
 * representation is the only signal there is.
 *
 * 🔴 AND IT IS RELATIVE, NOT ABSOLUTE. A pair representation's scale is a
 * property of the model and the token count, not of the fold, so an angstrom-
 * like threshold cannot transfer between them the way ColabFold's does. The
 * denominator makes the number comparable across models and sizes; the cost is
 * that it says nothing when the representation is all zeros, which is why that
 * case returns 0 rather than dividing by it.
 */

/**
 * Relative RMS change: `||b - a|| / ||b||`, or 0 when `b` is all zeros.
 *
 * @param {Float32Array} previous
 * @param {Float32Array} current
 * @returns {number} dimensionless, 0 when unchanged
 */
export function relativeChange(previous, current) {
  if (!(previous instanceof Float32Array) || !(current instanceof Float32Array)) {
    throw new TypeError("feature convergence takes two Float32Array tensors");
  }
  // 🔴 A FIRST PASS HAS NOTHING TO COMPARE AGAINST, and the caller seeds
  // `previous` with a zero array of the right length - so a length mismatch is
  // a real bug and an all-zero previous is not. Distinguish them.
  if (previous.length !== current.length) {
    throw new RangeError(
      `feature convergence over ${previous.length} and ${current.length} elements`);
  }
  let difference = 0;
  let magnitude = 0;
  for (let at = 0; at < current.length; at += 1) {
    const delta = current[at] - previous[at];
    difference += delta * delta;
    magnitude += current[at] * current[at];
  }
  if (!(magnitude > 0)) return 0;
  return Math.sqrt(difference / magnitude);
}

/**
 * Whether the trunk has stopped moving enough to skip the remaining recycles.
 *
 * 🔴 PASS 0 NEVER STOPS, for the same reason `shouldStopAfterRecycle` refuses
 * recycle 0: the first pass is compared against a zero seed, so its "change" is
 * the whole representation and its relative change is 1, but a tolerance above
 * 1 would otherwise let a fold skip its only trunk pass.
 *
 * @param {number} pass
 * @param {{pair: number, single: number}} delta
 * @param {number} tolerance dimensionless; 0 disables early stopping
 */
export function shouldStopRecycling(pass, delta, tolerance) {
  if (typeof tolerance !== "number" || !Number.isFinite(tolerance) || tolerance < 0) {
    throw new RangeError("feature convergence tolerance must be finite and non-negative");
  }
  if (!Number.isSafeInteger(pass) || pass < 0) {
    throw new RangeError("pass must be a non-negative integer");
  }
  for (const value of [delta?.pair, delta?.single]) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      throw new RangeError("both feature deltas must be finite and non-negative");
    }
  }
  // BOTH, because they are different tensors of the same state and either one
  // still moving means the trunk has not settled.
  return pass > 0 && tolerance > 0 && delta.pair < tolerance && delta.single < tolerance;
}
