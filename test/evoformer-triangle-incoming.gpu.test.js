import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";
import { TriangleMultiplicationIncomingGpu } from "../src/triangle/webgpu.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-block0/manifest.json";

function transpose(input, rows, columns) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[column * rows + row] = input[row * columns + column];
    }
  }
  return output;
}

describe.skipIf(!enabled)("TriangleMultiplicationIncoming WebGPU", () => {
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

  it("matches the official AlphaFold block activation", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest;
    const parameters = manifest.evoformerBlock.parameters;
    const stage = manifest.evoformerBlock.referenceStages.triangle_multiplication_incoming;
    if (stage === undefined) throw new Error("missing incoming triangle reference stage");
    const root = "triangle_multiplication_incoming";
    const parameter = async(module, name) => {
      const tensorName = parameters[`${root}/${module}`]?.[name];
      if (tensorName === undefined) throw new Error(`missing ${root}/${module}/${name}`);
      return store.tensor(tensorName);
    };
    const z = await store.tensor(stage.input);
    const expected = await store.tensor(stage.output);
    const shape = store.shape(stage.input);
    const length = shape[0];
    const cZ = shape[2];
    const leftBiasName = parameters[`${root}/left_projection`]?.bias;
    if (leftBiasName === undefined) throw new Error("missing triangle hidden shape");
    const cHidden = store.shape(leftBiasName)[0];
    const projection = async(module, inputChannels, outputChannels) =>
      transpose(await parameter(module, "weights"), inputChannels, outputChannels);
    const weights = {
      layerNormInWeight: await parameter("layer_norm_input", "scale"),
      layerNormInBias: await parameter("layer_norm_input", "offset"),
      linearAPWeight: await projection("left_projection", cZ, cHidden),
      linearAPBias: await parameter("left_projection", "bias"),
      linearAGWeight: await projection("left_gate", cZ, cHidden),
      linearAGBias: await parameter("left_gate", "bias"),
      linearBPWeight: await projection("right_projection", cZ, cHidden),
      linearBPBias: await parameter("right_projection", "bias"),
      linearBGWeight: await projection("right_gate", cZ, cHidden),
      linearBGBias: await parameter("right_gate", "bias"),
      layerNormOutWeight: await parameter("center_layer_norm", "scale"),
      layerNormOutBias: await parameter("center_layer_norm", "offset"),
      linearZWeight: await projection("output_projection", cHidden, cZ),
      linearZBias: await parameter("output_projection", "bias"),
      linearGWeight: await projection("gating_linear", cZ, cZ),
      linearGBias: await parameter("gating_linear", "bias"),
    };
    const result = await new TriangleMultiplicationIncomingGpu(device).run({
      shape: { length, cZ, cHidden },
      z,
      mask: await store.tensor("blockPairMask"),
      weights,
    });
    const metrics = errorMetrics(result.output, expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(1e-5);
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-4);
  });
});
