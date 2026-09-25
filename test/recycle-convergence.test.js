import { describe, expect, it } from "./harness.js";
import {
  planRecycleReuse,
  recycleConvergenceDistance, shouldStopAfterRecycle, validatedRecycleTolerance,
} from "../src/af2/model/recycle-convergence.js";

function positions(caCoordinates) {
  const result = new Float32Array(caCoordinates.length * 37 * 3);
  caCoordinates.forEach((coordinate, residue) => result.set(coordinate, (residue * 37 + 1) * 3));
  return result;
}

function independentReference(previous, current, mask) {
  const length = mask.length;
  const ca = (tensor, residue) => tensor.subarray((residue * 37 + 1) * 3, (residue * 37 + 1) * 3 + 3);
  const distance = (left, right) => Math.hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2]);
  let sum = 0; let weights = 0;
  for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
    const weight = mask[i] * mask[j];
    const delta = distance(ca(previous, i), ca(previous, j)) - distance(ca(current, i), ca(current, j));
    sum += delta * delta * weight; weights += weight;
  }
  return Math.sqrt(sum / weights + 1e-8);
}

describe("recycle early-stop convergence", () => {
  it("matches an independent C-alpha distance-matrix reference", () => {
    const previous = positions([[0, 0, 0], [3, 0, 0], [3, 4, 0]]);
    const current = positions([[1, 2, 0], [5, 2, 0], [5, 5, 2]]);
    const mask = Float32Array.of(1, 1, 0.5);
    const actual = recycleConvergenceDistance(previous, current, mask);
    expect(Math.abs(actual - independentReference(previous, current, mask))).toBeLessThan(1e-12);
  });

  it("is invariant to rigid translation and reports ColabFold's epsilon floor", () => {
    const previous = positions([[0, 0, 0], [3, 4, 0]]);
    const translated = positions([[10, -2, 7], [13, 2, 7]]);
    expect(recycleConvergenceDistance(previous, translated, Float32Array.of(1, 1))).toBeCloseTo(1e-4, 10);
  });

  it("stops only after two passes and uses a strict threshold", () => {
    expect(shouldStopAfterRecycle(0, 0.2, 0.5)).toBe(false);
    expect(shouldStopAfterRecycle(1, 0.2, 0.5)).toBe(true);
    expect(shouldStopAfterRecycle(1, 0.5, 0.5)).toBe(false);
    expect(shouldStopAfterRecycle(3, 0, 0)).toBe(false);
  });

  it("rejects invalid tolerances and tensor shapes at the public boundary", () => {
    expect(validatedRecycleTolerance(undefined)).toBe(0);
    expect(() => validatedRecycleTolerance(-1)).toThrow(/non-negative/);
    expect(() => validatedRecycleTolerance(Number.NaN)).toThrow(/finite/);
    expect(() => recycleConvergenceDistance(new Float32Array(1), new Float32Array(1), new Float32Array(1)))
      .toThrow(/shapes/);
  });
});

describe("what a press of Fold has to compute", () => {
  const key = "a-job";
  const passesOf = (recycles) => recycles + 1;
  const cacheOf = (count, recycles) => ({
    key, recycles: Array.from({ length: count }, (unused, i) => ({ pass: i })),
    resumable: { recycles },
  });

  it("runs everything when nothing is held", () => {
    expect(planRecycleReuse({ cache: undefined, key, passes: 4, recycles: 3 }).plan)
      .toBe("fresh");
    // ...and a cache for a DIFFERENT job is nothing held. The key carries the
    // family, which carries the model NUMBER, so this is also what stops
    // model_1's state reaching model_3 and a monomer's reaching a multimer.
    expect(planRecycleReuse({ cache: cacheOf(4, 3), key: "other", passes: 4, recycles: 3 }).plan)
      .toBe("fresh");
  });

  it("replays a run that CONVERGED, at its own count and at a higher one", () => {
    // Convergence means it STOPPED SHORT: four passes held where seven were
    // asked for, so the "enough in hand" test can never fire and every press
    // used to fall through to a resume and recompute. This is the commonest
    // fold there is, which is what made it worth a flag of its own.
    const converged = { ...cacheOf(4, 4), converged: true };
    expect(planRecycleReuse({ cache: converged, key, passes: passesOf(6), recycles: 6 }))
      .toEqual({ plan: "replay", passes: 0 });
    // ...and asking for MORE recycles cannot change it. Passes 1..4 are the
    // same four passes under the same key, so the stop test at four reads the
    // same two structures and fires again.
    expect(planRecycleReuse({ cache: converged, key, passes: passesOf(10), recycles: 10 }))
      .toEqual({ plan: "replay", passes: 0 });
  });

  it("still resumes a run that stopped short WITHOUT converging", () => {
    // The control that keeps the flag honest rather than the LENGTH: the same
    // four-of-seven shape, no convergence, so the missing three are still owed.
    const partial = { ...cacheOf(4, 4), converged: false };
    expect(planRecycleReuse({ cache: partial, key, passes: passesOf(6), recycles: 6 }))
      .toEqual({ plan: "resume", passes: 3 });
  });

  it("replays what is already in memory rather than folding it again", () => {
    const held = cacheOf(4, 3);
    expect(planRecycleReuse({ cache: held, key, passes: passesOf(3), recycles: 3 }))
      .toEqual({ plan: "replay", passes: 0 });
    // ...and FEWER passes is a replay too: the first N of a longer run are
    // that N-recycle fold, pass for pass.
    expect(planRecycleReuse({ cache: held, key, passes: passesOf(1), recycles: 1 }))
      .toEqual({ plan: "replay", passes: 0 });
  });

  it("resumes when more passes are asked for, and runs only those", () => {
    expect(planRecycleReuse({ cache: cacheOf(4, 3), key, passes: passesOf(5), recycles: 5 }))
      .toEqual({ plan: "resume", passes: 2 });
  });

  it("will not resume from a run whose state says it cannot", () => {
    // A fold whose state could not be read back comes back without one - see
    // the readback in src/af2/model/monomer.js, which no longer fails the
    // fold over it. There is nothing to continue from, so it starts again.
    const noState = { key, recycles: [{}, {}], resumable: {} };
    expect(planRecycleReuse({ cache: noState, key, passes: passesOf(5), recycles: 5 }).plan)
      .toBe("fresh");
  });
});
