import { describe, expect, it } from "./harness.js";
import {
  recycleConvergenceDistance, shouldStopAfterRecycle, validatedRecycleTolerance,
} from "../src/model/recycle-convergence.js";

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
