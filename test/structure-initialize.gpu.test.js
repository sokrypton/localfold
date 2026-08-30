import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { StructureInitializeGpu } from "../src/structure/initialize.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("structure module initialization WebGPU", () => {
  let gpu;
  let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals); gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());
  it("matches official single projection, initial act, and identity affine", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest;
    const p = async(map, module, name) => {
      const tensor = map[module]?.[name]; if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const weights = {
      singleProjectionWeight: await p(manifest.embedding.parameters, "single_activations", "weights"),
      singleProjectionBias: await p(manifest.embedding.parameters, "single_activations", "bias"),
      singleNormScale: await p(manifest.structureModule.parameters, "single_layer_norm", "scale"),
      singleNormOffset: await p(manifest.structureModule.parameters, "single_layer_norm", "offset"),
      initialProjectionWeight: await p(manifest.structureModule.parameters, "initial_projection", "weights"),
      initialProjectionBias: await p(manifest.structureModule.parameters, "initial_projection", "bias"),
    };
    const msa = await store.tensor("stackRecycle3ExpectedMsa");
    const length = store.shape("stackRecycle3ExpectedMsa")[1];
    const result = await new StructureInitializeGpu(device).run(msa.subarray(0, length * 256), length, 256, 384, weights);
    const singleMetrics = errorMetrics(result.single, await store.tensor("structureInputSingle"));
    const expectedAct = (await store.tensor("structureStage_act_input")).subarray(0, length * 384);
    const expectedAffine = (await store.tensor("structureStage_affine_input")).subarray(0, length * 7);
    const actMetrics = errorMetrics(result.activations, expectedAct);
    const affineMetrics = errorMetrics(result.affine, expectedAffine);
    expect(singleMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(singleMetrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(actMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(actMetrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(affineMetrics.maxAbsoluteError).toBe(0);
  });
});
