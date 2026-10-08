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

/**
 * What a press of Fold has to compute, given what is already in memory.
 *
 * 🔴 THREE ANSWERS, AND TWO OF THEM USED TO BE ONE. The page cached a
 * finished AlphaFold 2 run so that raising the recycle count could CONTINUE
 * it, and the test was `cached.recycles < recycles` - so asking for MORE
 * resumed, and asking for the SAME or FEWER fell through to a full fold from
 * pass zero. That fold cannot differ: everything that could change its answer
 * is in the cache key, the seed included, so it is minutes spent arriving
 * back where the page already was. Reported as re-running a prediction that
 * had already run.
 *
 *   "fresh"  - nothing usable in memory; run every pass.
 *   "resume" - the state is short of what was asked for; run the rest.
 *   "replay" - the passes are already here; run nothing.
 *
 * 🔴 AND FEWER PASSES IS A REPLAY, NOT A FRESH FOLD. The first N passes of a
 * longer run ARE the N-recycle fold, pass for pass, because each is computed
 * from the one before it.
 *
 * The KEY is the caller's: it names the sequence, the chain lengths, the MSA
 * depths, the seed, the tolerance, the alignment, the template and the family
 * - which carries the model NUMBER, so model_1's state can never be handed to
 * model_3, nor a monomer's to a multimer.
 *
 * @param {{key: string, recycles: object[]}} [cache] the last finished run
 * @param {string} key what this fold would be cached under
 * @param {number} passes recycles + 1, the passes being asked for
 * @param {number} recycles what the control says
 * @returns {{plan: "fresh"|"resume"|"replay", passes: number}} `passes` is
 *   how many this press must actually compute
 */
export function planRecycleReuse({ cache, key, passes, recycles }) {
  const held = (cache !== undefined && cache !== null && cache.key === key)
    ? (cache.recycles ?? []) : [];
  if (held.length === 0) return { plan: "fresh", passes };
  if (held.length >= passes) return { plan: "replay", passes: 0 };
  // 🔴 AND A RUN THAT CONVERGED IS COMPLETE AT EVERY COUNT, WHICH THE
  // LENGTH TEST ABOVE CANNOT SEE. Convergence means the run STOPPED SHORT of
  // what it was asked for, so `held.length` is always below `passes` and every
  // press fell through to the resume below - recomputing passes the first run
  // had deliberately declined to compute, on the commonest fold there is.
  // It cannot come out differently: the key pins the sequence, the seed, the
  // tolerance and the alignment, so passes 1..N are the same passes, and the
  // stop test at N reads N and N-1 alone. Asking for MORE recycles therefore
  // converges at the same pass - which is why this ignores `recycles` rather
  // than comparing it.
  if (cache.converged === true) return { plan: "replay", passes: 0 };
  // ...and a resume runs what is missing. `resumable.recycles` counts the
  // passes the STATE describes, which is what the driver starts from.
  const from = cache.resumable?.recycles;
  if (typeof from !== "number" || !(from < recycles)) return { plan: "fresh", passes };
  return { plan: "resume", passes: passes - held.length };
}
