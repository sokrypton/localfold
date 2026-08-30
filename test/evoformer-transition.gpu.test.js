import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { TransitionGpu } from "../src/evoformer/transition.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-block0/manifest.json";

describe.skipIf(!enabled)("Evoformer Transition WebGPU", () => {
  let gpu;
  let device;

  beforeAll(async() => {
    Object.assign(globalThis, globals);
    const adapterName = process.env.LOCALFOLD_ADAPTER;
    gpu = create(adapterName === undefined ? [] : [`adapter=${adapterName}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    const requiredFeatures = adapter.features.has("shader-f16") ? ["shader-f16"] : [];
    device = await adapter.requestDevice({ requiredFeatures });
  });

  afterAll(() => device?.destroy());

  it.each(["msa_transition", "pair_transition"])("matches official %s", async(moduleName) => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest;
    const moduleParameters = manifest.evoformerBlock.parameters;
    const stages = manifest.evoformerBlock.referenceStages;
    const parameter = async(path, name) => {
      const tensorName = moduleParameters[path]?.[name];
      if (tensorName === undefined) throw new Error(`missing ${path}/${name}`);
      return store.tensor(tensorName);
    };
    const stage = stages[moduleName];
    if (stage === undefined) throw new Error(`missing reference stage ${moduleName}`);
    const activations = await store.tensor(stage.input);
    const expected = await store.tensor(stage.output);
    const channels = store.shape(stage.input).at(-1);
    const hiddenTensor = moduleParameters[`${moduleName}/transition1`]?.bias;
    if (hiddenTensor === undefined) throw new Error(`missing hidden shape for ${moduleName}`);
    const hiddenChannels = store.shape(hiddenTensor)[0];
    const weights = {
      layerNormScale: await parameter(`${moduleName}/input_layer_norm`, "scale"),
      layerNormOffset: await parameter(`${moduleName}/input_layer_norm`, "offset"),
      firstWeight: await parameter(`${moduleName}/transition1`, "weights"),
      firstBias: await parameter(`${moduleName}/transition1`, "bias"),
      secondWeight: await parameter(`${moduleName}/transition2`, "weights"),
      secondBias: await parameter(`${moduleName}/transition2`, "bias"),
    };
    const result = await new TransitionGpu(device).run({
      activations,
      rows: activations.length / channels,
      channels,
      hiddenChannels,
      weights,
    });
    const metrics = errorMetrics(result.output, expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(1e-5);
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-4);
  });
});
