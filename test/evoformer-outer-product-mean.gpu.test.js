import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { OuterProductMeanGpu } from "../src/evoformer/outer-product-mean.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-block0/manifest.json";

describe.skipIf(!enabled)("OuterProductMean WebGPU", () => {
  let gpu;
  let device;

  beforeAll(async() => {
    Object.assign(globalThis, globals);
    const adapterName = process.env.LOCALFOLD_ADAPTER;
    gpu = create(adapterName === undefined ? [] : [`adapter=${adapterName}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    device = await adapter.requestDevice();
  });

  afterAll(() => device?.destroy());

  it("matches the official AlphaFold block activation", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest;
    const modules = manifest.evoformerBlock.parameters;
    const stage = manifest.evoformerBlock.referenceStages.outer_product_mean;
    if (stage === undefined) throw new Error("missing outer product mean reference stage");
    const parameter = async(module, name) => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const activations = await store.tensor(stage.input);
    const expected = await store.tensor(stage.output);
    const inputShape = store.shape(stage.input);
    const outputShape = store.shape(stage.output);
    const outerBiasName = modules["outer_product_mean/left_projection"]?.bias;
    if (outerBiasName === undefined) throw new Error("missing outer channel shape");
    const weights = {
      layerNormScale: await parameter("outer_product_mean/layer_norm_input", "scale"),
      layerNormOffset: await parameter("outer_product_mean/layer_norm_input", "offset"),
      leftWeight: await parameter("outer_product_mean/left_projection", "weights"),
      leftBias: await parameter("outer_product_mean/left_projection", "bias"),
      rightWeight: await parameter("outer_product_mean/right_projection", "weights"),
      rightBias: await parameter("outer_product_mean/right_projection", "bias"),
      outputWeight: await parameter("outer_product_mean", "output_w"),
      outputBias: await parameter("outer_product_mean", "output_b"),
    };
    const result = await new OuterProductMeanGpu(device).run({
      activations,
      mask: await store.tensor("blockMsaMask"),
      sequences: inputShape[0],
      length: inputShape[1],
      cM: inputShape[2],
      cOuter: store.shape(outerBiasName)[0],
      cZ: outputShape[2],
      weights,
    });
    const metrics = errorMetrics(result.output, expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(1e-5);
    // The bounded-memory contraction first combines output_w with the left
    // projection, changing FP32 summation order relative to JAX's einsums.
    expect(metrics.maxAbsoluteError).toBeLessThan(2e-4);
  });
});
