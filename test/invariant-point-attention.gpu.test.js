import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import {
  InvariantPointAttentionGpu,

} from "../src/structure/ipa.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("InvariantPointAttention WebGPU", () => {
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

  it("matches official structure iteration 0", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const modules = (store.manifest).structureModule.parameters;
    const parameter = async(module, name) => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const root = "fold_iteration/invariant_point_attention";
    const weights = {
      pairNormScale: await parameter("pair_layer_norm", "scale"),
      pairNormOffset: await parameter("pair_layer_norm", "offset"),
      queryScalarWeight: await parameter(`${root}/q_scalar`, "weights"),
      queryScalarBias: await parameter(`${root}/q_scalar`, "bias"),
      keyValueScalarWeight: await parameter(`${root}/kv_scalar`, "weights"),
      keyValueScalarBias: await parameter(`${root}/kv_scalar`, "bias"),
      queryPointWeight: await parameter(`${root}/q_point_local`, "weights"),
      queryPointBias: await parameter(`${root}/q_point_local`, "bias"),
      keyValuePointWeight: await parameter(`${root}/kv_point_local`, "weights"),
      keyValuePointBias: await parameter(`${root}/kv_point_local`, "bias"),
      trainablePointWeights: await parameter(root, "trainable_point_weights"),
      attention2dWeight: await parameter(`${root}/attention_2d`, "weights"),
      attention2dBias: await parameter(`${root}/attention_2d`, "bias"),
      outputWeight: await parameter(`${root}/output_projection`, "weights"),
      outputBias: await parameter(`${root}/output_projection`, "bias"),
    };
    const actStages = await store.tensor("structureStage_act_input");
    const affineStages = await store.tensor("structureStage_affine_input");
    const expectedStages = await store.tensor("structureStage_ipa");
    const actShape = store.shape("structureStage_act_input");
    const affineShape = store.shape("structureStage_affine_input");
    const expectedShape = store.shape("structureStage_ipa");
    const length = actShape[1];
    const channels = actShape[2];
    const result = await new InvariantPointAttentionGpu(device).run({
      activations: actStages.subarray(0, length * channels),
      pair: await store.tensor("structureInputPair"),
      mask: await store.tensor("feature_seq_mask_recycle3"),
      affine: affineStages.subarray(0, affineShape[1] * affineShape[2]),
      length,
      channels,
      pairChannels: 128,
      heads: 12,
      scalarQk: 16,
      scalarV: 16,
      pointQk: 4,
      pointV: 8,
      weights,
    });
    const expected = expectedStages.subarray(0, expectedShape[1] * expectedShape[2]);
    const metrics = errorMetrics(result.output, expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(metrics.maxAbsoluteError).toBeLessThan(3e-3);
  });
});
