import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { ConfidenceHeadsGpu } from "../src/heads/confidence.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("AlphaFold confidence heads WebGPU", () => {
  let gpu;
  let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals); gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("matches official model-1 pLDDT and PAE logits", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest;
    const parameter = async(map, module, name) => {
      const tensor = map[module]?.[name]; if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const lp = manifest.confidenceHeads.parameters.predictedLddt;
    const pp = manifest.confidenceHeads.parameters.predictedAlignedError;
    const lddtWeights = {
      normScale: await parameter(lp, "input_layer_norm", "scale"),
      normOffset: await parameter(lp, "input_layer_norm", "offset"),
      act0Weight: await parameter(lp, "act_0", "weights"),
      act0Bias: await parameter(lp, "act_0", "bias"),
      act1Weight: await parameter(lp, "act_1", "weights"),
      act1Bias: await parameter(lp, "act_1", "bias"),
      logitsWeight: await parameter(lp, "logits", "weights"),
      logitsBias: await parameter(lp, "logits", "bias"),
    };
    const paeWeights = {
      logitsWeight: await parameter(pp, "logits", "weights"),
      logitsBias: await parameter(pp, "logits", "bias"),
    };
    const length = store.shape("structureFinalRepresentation")[0];
    const result = await new ConfidenceHeadsGpu(device).run(
      await store.tensor("structureFinalRepresentation"), await store.tensor("structureInputPair"),
      length, lddtWeights, paeWeights, await store.tensor("confidencePaeBreaks"),
    );
    const lddtMetrics = errorMetrics(result.lddtLogits, await store.tensor("confidenceLddtLogits"));
    const plddtMetrics = errorMetrics(result.plddt, await store.tensor("confidencePlddt"));
    const paeMetrics = errorMetrics(result.paeLogits, await store.tensor("confidencePaeLogits"));
    expect(lddtMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(lddtMetrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(plddtMetrics.maxAbsoluteError).toBeLessThan(2e-3);
    expect(paeMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(paeMetrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(Math.abs(result.meanPlddt - manifest.confidenceHeads.reference.meanPlddt)).toBeLessThan(2e-3);
    expect(Math.abs(result.ptm - manifest.confidenceHeads.reference.ptm)).toBeLessThan(2e-4);
  });
});
