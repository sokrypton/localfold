const ATOMS = 37;
const COORDINATES = 3;
const CA = 1;

/** A finite, non-negative recycle tolerance in angstroms. Zero disables early stopping. */
export function validatedRecycleTolerance(value) {
  const tolerance = value ?? 0;
  if (typeof tolerance !== "number" || !Number.isFinite(tolerance) || tolerance < 0) {
    throw new RangeError("recycle tolerance must be a finite non-negative number");
  }
  return tolerance;
}

/**
 * ColabFold's recycle convergence metric: RMS change of all C-alpha pair distances.
 *
 * This is `confidence.compute_tol` without its temporary [L,L,3] coordinate
 * cube or [L,L] distance matrices. The direct pair walk keeps storage O(1)
 * beyond the two atom37 inputs while performing the same O(L^2) arithmetic.
 *
 * @param {Float32Array} previousAtom37 shape [L,37,3]
 * @param {Float32Array} currentAtom37 shape [L,37,3]
 * @param {Float32Array} mask shape [L]
 * @returns {number} angstroms
 */
export function recycleConvergenceDistance(previousAtom37, currentAtom37, mask) {
  if (!(previousAtom37 instanceof Float32Array) || !(currentAtom37 instanceof Float32Array)
      || !(mask instanceof Float32Array)) {
    throw new TypeError("recycle convergence inputs must be Float32Array tensors");
  }
  const length = mask.length;
  const positionElements = length * ATOMS * COORDINATES;
  if (length === 0 || previousAtom37.length !== positionElements || currentAtom37.length !== positionElements) {
    throw new RangeError("recycle convergence inputs must have shapes [L,37,3], [L,37,3], and [L]");
  }
  let squaredDifferenceSum = 0;
  let maskSum = 0;
  for (let i = 0; i < length; i += 1) {
    const maskI = mask[i];
    if (!Number.isFinite(maskI) || maskI < 0) throw new RangeError("sequence mask must be finite and non-negative");
    const previousI = (i * ATOMS + CA) * COORDINATES;
    for (let j = 0; j < length; j += 1) {
      const weight = maskI * mask[j];
      if (!Number.isFinite(weight) || weight < 0) throw new RangeError("sequence mask must be finite and non-negative");
      if (weight === 0) continue;
      const previousJ = (j * ATOMS + CA) * COORDINATES;
      let previousSquared = 0;
      let currentSquared = 0;
      for (let coordinate = 0; coordinate < COORDINATES; coordinate += 1) {
        const previousDelta = previousAtom37[previousI + coordinate] - previousAtom37[previousJ + coordinate];
        const currentDelta = currentAtom37[previousI + coordinate] - currentAtom37[previousJ + coordinate];
        previousSquared += previousDelta * previousDelta;
        currentSquared += currentDelta * currentDelta;
      }
      const difference = Math.sqrt(previousSquared) - Math.sqrt(currentSquared);
      squaredDifferenceSum += difference * difference * weight;
      maskSum += weight;
    }
  }
  if (maskSum === 0) throw new RangeError("sequence mask must contain at least one positive value");
  return Math.sqrt(squaredDifferenceSum / maskSum + 1e-8);
}

/** ColabFold stops only after recycle index 1 or later, with a strict less-than comparison. */
export function shouldStopAfterRecycle(recycle, distance, tolerance) {
  const checkedTolerance = validatedRecycleTolerance(tolerance);
  if (!Number.isSafeInteger(recycle) || recycle < 0 || !Number.isFinite(distance) || distance < 0) {
    throw new RangeError("recycle index and convergence distance must be non-negative finite values");
  }
  return recycle > 0 && checkedTolerance > 0 && distance < checkedTolerance;
}
