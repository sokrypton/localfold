import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import {
  StructurePostAttentionGpu,

} from "../src/structure/iteration.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("structure post-attention iteration WebGPU", () => {
  let gpu;
  let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("matches official transition and affine update for iteration 0", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const modules = (store.manifest).structureModule.parameters;
    const parameter = async(module, name) => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const root = "fold_iteration";
    const weights = {
      attentionNormScale: await parameter(`${root}/attention_layer_norm`, "scale"),
      attentionNormOffset: await parameter(`${root}/attention_layer_norm`, "offset"),
      transitionWeights: [
        await parameter(`${root}/transition`, "weights"),
        await parameter(`${root}/transition_1`, "weights"),
        await parameter(`${root}/transition_2`, "weights"),
      ],
      transitionBiases: [
        await parameter(`${root}/transition`, "bias"),
        await parameter(`${root}/transition_1`, "bias"),
        await parameter(`${root}/transition_2`, "bias"),
      ],
      transitionNormScale: await parameter(`${root}/transition_layer_norm`, "scale"),
      transitionNormOffset: await parameter(`${root}/transition_layer_norm`, "offset"),
      affineWeight: await parameter(`${root}/affine_update`, "weights"),
      affineBias: await parameter(`${root}/affine_update`, "bias"),
    };
    const actInputs = await store.tensor("structureStage_act_input");
    const affineInputs = await store.tensor("structureStage_affine_input");
    const ipa = await store.tensor("structureStage_ipa");
    const expectedAct = await store.tensor("structureStage_act_output");
    const expectedAffine = await store.tensor("structureStage_affine_output");
    const actShape = store.shape("structureStage_act_input");
    const affineShape = store.shape("structureStage_affine_input");
    const length = actShape[1];
    const channels = actShape[2];
    const result = await new StructurePostAttentionGpu(device).run({
      activations: actInputs.subarray(0, length * channels),
      attentionUpdate: ipa.subarray(0, length * channels),
      affine: affineInputs.subarray(0, length * 7),
      length,
      channels,
      weights,
    });
    const actMetrics = errorMetrics(result.activations, expectedAct.subarray(0, length * channels));
    const affineMetrics = errorMetrics(result.affine, expectedAffine.subarray(0, affineShape[1] * affineShape[2]));
    expect(actMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(actMetrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(affineMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(affineMetrics.maxAbsoluteError).toBeLessThan(3e-3);
  });
});
