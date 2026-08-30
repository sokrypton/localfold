import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { resolve } from "node:path";
import { create, globals } from "webgpu";
import { loadTriangleReferenceBundleFromFiles } from "../src/reference/node.js";
import { errorMetrics } from "../src/triangle/types.js";
import { TriangleMultiplicationOutgoingGpu } from "../src/triangle/webgpu.js";
import { ALPHAFOLD_REFERENCE_MANIFESTS } from "./alphafold-references.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";

describe.skipIf(!enabled)("TriangleMultiplicationOutgoing WebGPU", () => {
  let gpu;
  let device;
  let hasF16 = false;

  beforeAll(async() => {
    Object.assign(globalThis, globals);
    const adapterName = process.env.LOCALFOLD_ADAPTER;
    gpu = create(adapterName === undefined ? [] : [`adapter=${adapterName}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    hasF16 = adapter.features.has("shader-f16");
    const requiredFeatures = hasF16 ? ["shader-f16"] : [];
    device = await adapter.requestDevice({ requiredFeatures });
  });

  afterAll(() => {
    device?.destroy();
  });

  it("matches the independent float32 reference", async() => {
    const bundle = await loadTriangleReferenceBundleFromFiles(
      resolve("test/fixtures/openfold-triangle-small/manifest.json"),
    );
    const result = await new TriangleMultiplicationOutgoingGpu(device).run(bundle.input, { precision: "f32" });
    const metrics = errorMetrics(result.output, bundle.expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(1e-5);
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-4);
  });

  it.each(ALPHAFOLD_REFERENCE_MANIFESTS)("matches official AlphaFold FP32 reference %s", async(path) => {
    const bundle = await loadTriangleReferenceBundleFromFiles(resolve(path));
    const result = await new TriangleMultiplicationOutgoingGpu(device).run(bundle.input, { precision: "f32" });
    const metrics = errorMetrics(result.output, bundle.expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(1e-5);
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-4);
  });

  it("supports fp16 inputs and weights when shader-f16 is available", async(context) => {
    if (!hasF16) context.skip();
    const bundle = await loadTriangleReferenceBundleFromFiles(
      resolve("test/fixtures/openfold-triangle-small/manifest.json"),
    );
    const result = await new TriangleMultiplicationOutgoingGpu(device).run(bundle.input, { precision: "f16" });
    const metrics = errorMetrics(result.output, bundle.expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(1e-3);
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-2);
  });

  it("matches official single-sequence FP16 references", async(context) => {
    if (!hasF16) context.skip();
    for (const path of ALPHAFOLD_REFERENCE_MANIFESTS) {
      const bundle = await loadTriangleReferenceBundleFromFiles(resolve(path));
      const result = await new TriangleMultiplicationOutgoingGpu(device).run(bundle.input, { precision: "f16" });
      const metrics = errorMetrics(result.output, bundle.expected);
      expect(metrics.meanAbsoluteError).toBeLessThan(1e-3);
      expect(metrics.maxAbsoluteError).toBeLessThan(1e-2);
    }
  });
});
