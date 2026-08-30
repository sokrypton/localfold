import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { SidechainAnglesGpu } from "../src/structure/sidechain.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("structure sidechain angle network WebGPU", () => {
  let gpu; let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals); gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());
  it("matches official final-iteration torsion angles", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const modules = (store.manifest).structureModule.parameters;
    const p = async(module, name) => {
      const tensor = modules[module]?.[name]; if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const root = "fold_iteration/rigid_sidechain";
    const weights = {
      inputWeight: await p(`${root}/input_projection`, "weights"),
      inputBias: await p(`${root}/input_projection`, "bias"),
      initialInputWeight: await p(`${root}/input_projection_1`, "weights"),
      initialInputBias: await p(`${root}/input_projection_1`, "bias"),
      residual1Weights: [await p(`${root}/resblock1`, "weights"), await p(`${root}/resblock2`, "weights")],
      residual1Biases: [await p(`${root}/resblock1`, "bias"), await p(`${root}/resblock2`, "bias")],
      residual2Weights: [await p(`${root}/resblock1_1`, "weights"), await p(`${root}/resblock2_1`, "weights")],
      residual2Biases: [await p(`${root}/resblock1_1`, "bias"), await p(`${root}/resblock2_1`, "bias")],
      angleWeight: await p(`${root}/unnormalized_angles`, "weights"),
      angleBias: await p(`${root}/unnormalized_angles`, "bias"),
    };
    const act = await store.tensor("structureStage_act_output");
    const initial = await store.tensor("structureStage_initial_act");
    const expectedAngles = await store.tensor("structureStage_angles");
    const expectedUnnormalized = await store.tensor("structureStage_unnormalized_angles");
    const shape = store.shape("structureStage_act_output");
    const length = shape[1]; const channels = shape[2];
    const offset = 7 * length * channels;
    const result = await new SidechainAnglesGpu(device).run(
      act.subarray(offset, offset + length * channels), initial.subarray(offset, offset + length * channels),
      length, channels, 128, weights,
    );
    const angleOffset = 7 * length * 14;
    const angleMetrics = errorMetrics(result.angles, expectedAngles.subarray(angleOffset, angleOffset + length * 14));
    const rawMetrics = errorMetrics(
      result.unnormalizedAngles, expectedUnnormalized.subarray(angleOffset, angleOffset + length * 14),
    );
    expect(rawMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(rawMetrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(angleMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(angleMetrics.maxAbsoluteError).toBeLessThan(3e-3);
  });
});
