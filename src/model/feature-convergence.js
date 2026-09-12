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
 * 🔴 THE TOLERANCE IS IN ANGSTROMS, AND THAT IS THE POINT. The pair and single
 * deltas beside it are dimensionless, so a threshold on them means something
 * different for every model and token count and can only be fitted per shape.
 * `distanceAngstroms` is the RMS change of the distances the DISTOGRAM predicts
 * - the same quantity ColabFold's `compute_tol` takes over a structure, whose
 * default tolerance is 0.5 A - so AF2's criterion and this one are finally the
 * same measurement in the same unit, one taken from coordinates and one from
 * the trunk.
 *
 * 🔴 AND IT IS THE ONLY ONE OF THE THREE A SAMPLED STRUCTURE CANNOT CONTRADICT.
 * Measured on a 250-token input at three seeds, the distogram change reads
 * 0.4195, 0.2602 and 0.2289 A on every one of them, bit for bit, while the
 * structures those same folds produced differ by 4.6 to 12.7 A. The trunk is
 * deterministic; only the sampler is not. See docs/AF3.md.
 *
 * PASS 0 NEVER STOPS: it has no previous pass to compare against, so its delta
 * carries no `distanceAngstroms` at all.
 *
 * @param {number} pass
 * @param {{distanceAngstroms?: number}} delta
 * @param {number} tolerance angstroms; 0 disables early stopping
 */
export function shouldStopRecycling(pass, delta, tolerance) {
  if (typeof tolerance !== "number" || !Number.isFinite(tolerance) || tolerance < 0) {
    throw new RangeError("recycle tolerance must be finite and non-negative");
  }
  if (!Number.isSafeInteger(pass) || pass < 0) {
    throw new RangeError("pass must be a non-negative integer");
  }
  const change = delta?.distanceAngstroms;
  // 🔴 ABSENT IS NOT CONVERGED. A pass with no distogram - the first, or a
  // model whose head this fold did not run - must not read as zero and stop.
  if (change === undefined) return false;
  if (typeof change !== "number" || !Number.isFinite(change) || change < 0) {
    throw new RangeError("the distogram change must be finite and non-negative");
  }
  return pass > 0 && tolerance > 0 && change < tolerance;
}


/**
 * The distance each pair's distogram predicts, in angstroms.
 *
 * 🔴 THIS IS THE ONE TRUNK QUANTITY THAT IS BOTH DETERMINISTIC AND IN
 * ANGSTROMS, which is what makes it the calibration target the relative deltas
 * above are not. AF2 stops on ColabFold's `compute_tol` - the RMS change of
 * every C-alpha pair DISTANCE - and a distogram predicts a distribution over
 * exactly those distances. Taking its expectation gives the same matrix
 * `compute_tol` walks, from the trunk, without asking the sampler for a
 * structure it would have to sample stochastically. See docs/AF3.md for why a
 * sampled structure cannot measure this.
 *
 * The first and last bins are open-ended - everything below the first break and
 * everything above the last - so their centres are the breaks themselves rather
 * than a midpoint of something unbounded. That biases a pair the model is sure
 * is far apart towards the last break, which is what AlphaFold's own
 * `_distogram_log_loss` does with them too.
 *
 * @param {Float32Array} logits [pairs, bins], bins fastest
 * @param {Float32Array} breaks the bins - 1 boundaries, ascending
 * @returns {Float32Array} [pairs] angstroms
 */
export function expectedDistances(logits, breaks) {
  if (!(logits instanceof Float32Array) || !(breaks instanceof Float32Array)) {
    throw new TypeError("expectedDistances takes Float32Array logits and breaks");
  }
  const bins = breaks.length + 1;
  if (logits.length % bins !== 0) {
    throw new RangeError(`${logits.length} logits is not a whole number of ${bins}-bin rows`);
  }
  const centres = new Float32Array(bins);
  centres[0] = breaks[0];
  for (let b = 1; b < bins - 1; b += 1) centres[b] = (breaks[b - 1] + breaks[b]) / 2;
  centres[bins - 1] = breaks[bins - 2];
  const pairs = logits.length / bins;
  const out = new Float32Array(pairs);
  for (let pair = 0; pair < pairs; pair += 1) {
    const base = pair * bins;
    let largest = -Infinity;
    for (let b = 0; b < bins; b += 1) {
      if (logits[base + b] > largest) largest = logits[base + b];
    }
    let total = 0;
    let weighted = 0;
    for (let b = 0; b < bins; b += 1) {
      const probability = Math.exp(logits[base + b] - largest);
      total += probability;
      weighted += probability * centres[b];
    }
    out[pair] = total > 0 ? weighted / total : 0;
  }
  return out;
}

/**
 * RMS change between two predicted distance matrices, in angstroms.
 *
 * The same reduction ColabFold's `compute_tol` performs, over the distances the
 * distogram predicts rather than the distances a sampled structure has - so a
 * tolerance here means what a tolerance there means, and the two models'
 * criteria are finally in the same unit.
 */
export function distanceChange(previous, current) {
  if (!(previous instanceof Float32Array) || !(current instanceof Float32Array)) {
    throw new TypeError("distanceChange takes two Float32Array distance maps");
  }
  if (previous.length !== current.length) {
    throw new RangeError(`distance maps of ${previous.length} and ${current.length}`);
  }
  if (current.length === 0) return 0;
  let squared = 0;
  for (let at = 0; at < current.length; at += 1) {
    const delta = current[at] - previous[at];
    squared += delta * delta;
  }
  return Math.sqrt(squared / current.length);
}
