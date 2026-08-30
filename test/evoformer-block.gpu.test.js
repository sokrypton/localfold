import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import {
  EvoformerBlockGpu,

} from "../src/evoformer/block.js";

import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

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

describe.skipIf(!enabled)("complete Evoformer block WebGPU", () => {
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

  it("matches official AlphaFold with one command buffer and no CPU neural operations", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest;
    const modules = manifest.evoformerBlock.parameters;
    const parameter = async(module, name) => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const shapeOfParameter = (module, name) => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.shape(tensor);
    };
    const attention = async(root) => {
      const attentionRoot = `${root}/attention`;
      const heads = shapeOfParameter(attentionRoot, "gating_b")[0];
      const weights = {
        queryNormScale: await parameter(`${root}/query_norm`, "scale"),
        queryNormOffset: await parameter(`${root}/query_norm`, "offset"),
        queryWeight: await parameter(attentionRoot, "query_w"),
        keyWeight: await parameter(attentionRoot, "key_w"),
        valueWeight: await parameter(attentionRoot, "value_w"),
        gatingWeight: await parameter(attentionRoot, "gating_w"),
        gatingBias: await parameter(attentionRoot, "gating_b"),
        outputWeight: await parameter(attentionRoot, "output_w"),
        outputBias: await parameter(attentionRoot, "output_b"),
      };
      return { heads, attention: weights };
    };
    const transition = async(root) => ({
      layerNormScale: await parameter(`${root}/input_layer_norm`, "scale"),
      layerNormOffset: await parameter(`${root}/input_layer_norm`, "offset"),
      firstWeight: await parameter(`${root}/transition1`, "weights"),
      firstBias: await parameter(`${root}/transition1`, "bias"),
      secondWeight: await parameter(`${root}/transition2`, "weights"),
      secondBias: await parameter(`${root}/transition2`, "bias"),
    });
    const triangle = async(root, cZ) => {
      const hidden = shapeOfParameter(`${root}/left_projection`, "bias")[0];
      const projection = async(module, inputChannels, outputChannels) =>
        transpose(await parameter(`${root}/${module}`, "weights"), inputChannels, outputChannels);
      return {
        layerNormInWeight: await parameter(`${root}/layer_norm_input`, "scale"),
        layerNormInBias: await parameter(`${root}/layer_norm_input`, "offset"),
        linearAPWeight: await projection("left_projection", cZ, hidden),
        linearAPBias: await parameter(`${root}/left_projection`, "bias"),
        linearAGWeight: await projection("left_gate", cZ, hidden),
        linearAGBias: await parameter(`${root}/left_gate`, "bias"),
        linearBPWeight: await projection("right_projection", cZ, hidden),
        linearBPBias: await parameter(`${root}/right_projection`, "bias"),
        linearBGWeight: await projection("right_gate", cZ, hidden),
        linearBGBias: await parameter(`${root}/right_gate`, "bias"),
        layerNormOutWeight: await parameter(`${root}/center_layer_norm`, "scale"),
        layerNormOutBias: await parameter(`${root}/center_layer_norm`, "offset"),
        linearZWeight: await projection("output_projection", hidden, cZ),
        linearZBias: await parameter(`${root}/output_projection`, "bias"),
        linearGWeight: await projection("gating_linear", cZ, cZ),
        linearGBias: await parameter(`${root}/gating_linear`, "bias"),
      };
    };

    const msa = await store.tensor("blockInputMsa");
    const pair = await store.tensor("blockInputPair");
    const msaShape = store.shape("blockInputMsa");
    const pairShape = store.shape("blockInputPair");
    const cZ = pairShape[2];
    const rowBase = await attention("msa_row_attention_with_pair_bias");
    const row = {
      ...rowBase,
      pairLayerNormScale: await parameter("msa_row_attention_with_pair_bias/feat_2d_norm", "scale"),
      pairLayerNormOffset: await parameter("msa_row_attention_with_pair_bias/feat_2d_norm", "offset"),
      pairProjectionWeight: await parameter("msa_row_attention_with_pair_bias", "feat_2d_weights"),
    };
    const startingBase = await attention("triangle_attention_starting_node");
    const endingBase = await attention("triangle_attention_ending_node");
    const starting = {
      ...startingBase,
      pairProjectionWeight: await parameter("triangle_attention_starting_node", "feat_2d_weights"),
    };
    const ending = {
      ...endingBase,
      pairProjectionWeight: await parameter("triangle_attention_ending_node", "feat_2d_weights"),
    };
    const outerProductMean = {
      layerNormScale: await parameter("outer_product_mean/layer_norm_input", "scale"),
      layerNormOffset: await parameter("outer_product_mean/layer_norm_input", "offset"),
      leftWeight: await parameter("outer_product_mean/left_projection", "weights"),
      leftBias: await parameter("outer_product_mean/left_projection", "bias"),
      rightWeight: await parameter("outer_product_mean/right_projection", "weights"),
      rightBias: await parameter("outer_product_mean/right_projection", "bias"),
      outputWeight: await parameter("outer_product_mean", "output_w"),
      outputBias: await parameter("outer_product_mean", "output_b"),
    };
    const weights = {
      msaRowAttention: row,
      msaColumnAttention: await attention("msa_column_attention"),
      msaTransition: await transition("msa_transition"),
      outerProductMean,
      triangleMultiplicationOutgoing: await triangle("triangle_multiplication_outgoing", cZ),
      triangleMultiplicationIncoming: await triangle("triangle_multiplication_incoming", cZ),
      triangleAttentionStarting: starting,
      triangleAttentionEnding: ending,
      pairTransition: await transition("pair_transition"),
    };
    const result = await new EvoformerBlockGpu(device).run({
      msa,
      pair,
      msaMask: await store.tensor("blockMsaMask"),
      pairMask: await store.tensor("blockPairMask"),
      sequences: msaShape[0],
      length: msaShape[1],
      cM: msaShape[2],
      cZ,
      cOuter: outerProductMean.leftBias.length,
      triangleHidden: weights.triangleMultiplicationOutgoing.linearAPBias.length,
      weights,
    });
    const msaMetrics = errorMetrics(result.msa, await store.tensor("blockExpectedMsa"));
    const pairMetrics = errorMetrics(result.pair, await store.tensor("blockExpectedPair"));
    expect(msaMetrics.meanAbsoluteError).toBeLessThan(5e-5);
    expect(msaMetrics.maxAbsoluteError).toBeLessThan(5e-4);
    expect(pairMetrics.meanAbsoluteError).toBeLessThan(1e-4);
    expect(pairMetrics.maxAbsoluteError).toBeLessThan(1e-3);
  }, 30_000);
});
