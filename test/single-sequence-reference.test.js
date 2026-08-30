import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { describe, expect, it } from "./harness.js";
import { loadTriangleReferenceBundleFromFiles } from "../src/reference/node.js";
import { triangleMultiplicationOutgoingReference } from "../src/triangle/cpu-reference.js";
import { errorMetrics } from "../src/triangle/types.js";
import { ALPHAFOLD_REFERENCE_MANIFESTS } from "./alphafold-references.js";

describe("official AlphaFold input references", () => {
  it.each(ALPHAFOLD_REFERENCE_MANIFESTS)("matches %s", async(relativePath) => {
    const path = resolve(relativePath);
    const bundle = await loadTriangleReferenceBundleFromFiles(path);
    const manifest = JSON.parse(await readFile(path, "utf8"))

    ;
    const metrics = errorMetrics(
      triangleMultiplicationOutgoingReference(bundle.input),
      bundle.expected,
    );
    expect(bundle.input.shape.length).toBe(manifest.sequence.aminoAcids.length);
    expect(manifest.referenceExecution.implementation).toMatch(/^official AlphaFold/);
    expect(manifest.referenceExecution.platform).toBe("CPU");
    expect(metrics.meanAbsoluteError).toBeLessThan(1e-6);
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-5);
  });
});
