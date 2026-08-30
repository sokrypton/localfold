import { describe, expect, it } from "./harness.js";
import { resolve } from "node:path";
import { loadTriangleReferenceBundleFromFiles } from "../src/reference/node.js";
import { createDeterministicTriangleInput } from "../src/testing/deterministic-input.js";
import { triangleMultiplicationOutgoingReference } from "../src/triangle/cpu-reference.js";
import { errorMetrics, validateTriangleInput } from "../src/triangle/types.js";

describe("TriangleMultiplicationOutgoing CPU reference", () => {
  it("is deterministic and returns one value per pair channel", () => {
    const input = createDeterministicTriangleInput({ length: 3, cZ: 4, cHidden: 5 }, 42);
    const first = triangleMultiplicationOutgoingReference(input);
    const second = triangleMultiplicationOutgoingReference(input);
    expect(first).toEqual(second);
    expect(first).toHaveLength(3 * 3 * 4);
    expect(Array.from(first).every(Number.isFinite)).toBe(true);
  });

  it("matches the frozen OpenFold oracle", async() => {
    const bundle = await loadTriangleReferenceBundleFromFiles(
      resolve("test/fixtures/openfold-triangle-small/manifest.json"),
    );
    const actual = triangleMultiplicationOutgoingReference(bundle.input);
    const metrics = errorMetrics(actual, bundle.expected);
    expect(bundle.source).toMatch(/^OpenFold/);
    expect(metrics.meanAbsoluteError).toBeLessThan(1e-7);
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-6);
  });

  it("rejects inconsistent tensor shapes", () => {
    const input = createDeterministicTriangleInput({ length: 2, cZ: 3, cHidden: 2 });
    const invalid = { ...input, z: input.z.subarray(1) };
    expect(() => validateTriangleInput(invalid)).toThrow(/z has/);
  });

  it("reports exact error metrics", () => {
    expect(errorMetrics(new Float32Array([1, 3]), new Float32Array([2, 1]))).toEqual({
      meanAbsoluteError: 1.5,
      maxAbsoluteError: 2,
      rootMeanSquareError: Math.sqrt(2.5),
    });
  });
});
