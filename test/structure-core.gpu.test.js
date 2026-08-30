import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { StructureCoreGpu } from "../src/structure/core.js";

import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("eight-iteration structure core WebGPU", () => {
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

  it("matches the final official act and affine state", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const modules = (store.manifest).structureModule.parameters;
    const p = async(module, name) => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const ipaRoot = "fold_iteration/invariant_point_attention";
    const ipaWeights = {
      pairNormScale: await p("pair_layer_norm", "scale"), pairNormOffset: await p("pair_layer_norm", "offset"),
      queryScalarWeight: await p(`${ipaRoot}/q_scalar`, "weights"), queryScalarBias: await p(`${ipaRoot}/q_scalar`, "bias"),
      keyValueScalarWeight: await p(`${ipaRoot}/kv_scalar`, "weights"), keyValueScalarBias: await p(`${ipaRoot}/kv_scalar`, "bias"),
      queryPointWeight: await p(`${ipaRoot}/q_point_local`, "weights"), queryPointBias: await p(`${ipaRoot}/q_point_local`, "bias"),
      keyValuePointWeight: await p(`${ipaRoot}/kv_point_local`, "weights"), keyValuePointBias: await p(`${ipaRoot}/kv_point_local`, "bias"),
      trainablePointWeights: await p(ipaRoot, "trainable_point_weights"),
      attention2dWeight: await p(`${ipaRoot}/attention_2d`, "weights"), attention2dBias: await p(`${ipaRoot}/attention_2d`, "bias"),
      outputWeight: await p(`${ipaRoot}/output_projection`, "weights"), outputBias: await p(`${ipaRoot}/output_projection`, "bias"),
    };
    const root = "fold_iteration";
    const postAttentionWeights = {
      attentionNormScale: await p(`${root}/attention_layer_norm`, "scale"),
      attentionNormOffset: await p(`${root}/attention_layer_norm`, "offset"),
      transitionWeights: [await p(`${root}/transition`, "weights"), await p(`${root}/transition_1`, "weights"),
        await p(`${root}/transition_2`, "weights")],
      transitionBiases: [await p(`${root}/transition`, "bias"), await p(`${root}/transition_1`, "bias"),
        await p(`${root}/transition_2`, "bias")],
      transitionNormScale: await p(`${root}/transition_layer_norm`, "scale"),
      transitionNormOffset: await p(`${root}/transition_layer_norm`, "offset"),
      affineWeight: await p(`${root}/affine_update`, "weights"), affineBias: await p(`${root}/affine_update`, "bias"),
    };
    const actStages = await store.tensor("structureStage_act_input");
    const affineStages = await store.tensor("structureStage_affine_input");
    const expectedAct = await store.tensor("structureStage_act_output");
    const expectedAffine = await store.tensor("structureStage_affine_output");
    const actShape = store.shape("structureStage_act_input");
    const affineShape = store.shape("structureStage_affine_input");
    const length = actShape[1];
    const channels = actShape[2];
    const result = await new StructureCoreGpu(device).run({
      activations: actStages.subarray(0, length * channels),
      pair: await store.tensor("structureInputPair"),
      mask: await store.tensor("feature_seq_mask_recycle3"),
      affine: affineStages.subarray(0, length * 7),
      length, channels, pairChannels: 128, ipaWeights, postAttentionWeights,
    });
    const actOffset = 7 * length * channels;
    const affineOffset = 7 * length * 7;
    const actMetrics = errorMetrics(result.activations, expectedAct.subarray(actOffset, actOffset + length * channels));
    const affineMetrics = errorMetrics(result.affine, expectedAffine.subarray(
      affineOffset, affineOffset + affineShape[1] * affineShape[2],
    ));
    expect(actMetrics.meanAbsoluteError).toBeLessThan(2e-3);
    expect(actMetrics.maxAbsoluteError).toBeLessThan(3e-2);
    expect(affineMetrics.meanAbsoluteError).toBeLessThan(2e-3);
    expect(affineMetrics.maxAbsoluteError).toBeLessThan(3e-2);
  }, 30_000);
});
