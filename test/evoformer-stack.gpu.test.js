import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";

import { EvoformerStackGpu } from "../src/evoformer/stack.js";

import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

function transpose(input, rows, columns) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[column * rows + row] = input[row * columns + column];
    }
  }
  return output;
}

describe.skipIf(!enabled)("48-block Evoformer stack WebGPU", () => {
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

  it("matches the official final-recycle stack while retaining activations on-device", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest;
    const modules = manifest.evoformerStack.parameters;
    const blocks = manifest.evoformerStack.blocks;
    const stacked = async(module, name, block) => {
      const tensorName = modules[module]?.[name];
      if (tensorName === undefined) throw new Error(`missing ${module}/${name}`);
      const tensor = await store.tensor(tensorName);
      const blockSize = tensor.length / blocks;
      return tensor.subarray(block * blockSize, (block + 1) * blockSize);
    };
    const parameterShape = (module, name) => {
      const tensorName = modules[module]?.[name];
      if (tensorName === undefined) throw new Error(`missing ${module}/${name}`);
      return store.shape(tensorName).slice(1);
    };
    const attention = async(root, block) => {
      const attentionRoot = `${root}/attention`;
      const heads = parameterShape(attentionRoot, "gating_b")[0];
      const weights = {
        queryNormScale: await stacked(`${root}/query_norm`, "scale", block),
        queryNormOffset: await stacked(`${root}/query_norm`, "offset", block),
        queryWeight: await stacked(attentionRoot, "query_w", block),
        keyWeight: await stacked(attentionRoot, "key_w", block),
        valueWeight: await stacked(attentionRoot, "value_w", block),
        gatingWeight: await stacked(attentionRoot, "gating_w", block),
        gatingBias: await stacked(attentionRoot, "gating_b", block),
        outputWeight: await stacked(attentionRoot, "output_w", block),
        outputBias: await stacked(attentionRoot, "output_b", block),
      };
      return { heads, attention: weights };
    };
    const transition = async(root, block) => ({
      layerNormScale: await stacked(`${root}/input_layer_norm`, "scale", block),
      layerNormOffset: await stacked(`${root}/input_layer_norm`, "offset", block),
      firstWeight: await stacked(`${root}/transition1`, "weights", block),
      firstBias: await stacked(`${root}/transition1`, "bias", block),
      secondWeight: await stacked(`${root}/transition2`, "weights", block),
      secondBias: await stacked(`${root}/transition2`, "bias", block),
    });
    const triangle = async(root, cZ, block) => {
      const hidden = parameterShape(`${root}/left_projection`, "bias")[0];
      const projection = async(module, inputChannels, outputChannels) =>
        transpose(await stacked(`${root}/${module}`, "weights", block), inputChannels, outputChannels);
      return {
        layerNormInWeight: await stacked(`${root}/layer_norm_input`, "scale", block),
        layerNormInBias: await stacked(`${root}/layer_norm_input`, "offset", block),
        linearAPWeight: await projection("left_projection", cZ, hidden),
        linearAPBias: await stacked(`${root}/left_projection`, "bias", block),
        linearAGWeight: await projection("left_gate", cZ, hidden),
        linearAGBias: await stacked(`${root}/left_gate`, "bias", block),
        linearBPWeight: await projection("right_projection", cZ, hidden),
        linearBPBias: await stacked(`${root}/right_projection`, "bias", block),
        linearBGWeight: await projection("right_gate", cZ, hidden),
        linearBGBias: await stacked(`${root}/right_gate`, "bias", block),
        layerNormOutWeight: await stacked(`${root}/center_layer_norm`, "scale", block),
        layerNormOutBias: await stacked(`${root}/center_layer_norm`, "offset", block),
        linearZWeight: await projection("output_projection", hidden, cZ),
        linearZBias: await stacked(`${root}/output_projection`, "bias", block),
        linearGWeight: await projection("gating_linear", cZ, cZ),
        linearGBias: await stacked(`${root}/gating_linear`, "bias", block),
      };
    };
    const msa = await store.tensor("stackInputMsa");
    const pair = await store.tensor("stackInputPair");
    const msaShape = store.shape("stackInputMsa");
    const pairShape = store.shape("stackInputPair");
    const cZ = pairShape[2];
    const blockWeights = [];
    for (let block = 0; block < blocks; block += 1) {
      const rowBase = await attention("msa_row_attention_with_pair_bias", block);
      const row = {
        ...rowBase,
        pairLayerNormScale: await stacked("msa_row_attention_with_pair_bias/feat_2d_norm", "scale", block),
        pairLayerNormOffset: await stacked("msa_row_attention_with_pair_bias/feat_2d_norm", "offset", block),
        pairProjectionWeight: await stacked("msa_row_attention_with_pair_bias", "feat_2d_weights", block),
      };
      const startingBase = await attention("triangle_attention_starting_node", block);
      const endingBase = await attention("triangle_attention_ending_node", block);
      const starting = {
        ...startingBase,
        pairProjectionWeight: await stacked("triangle_attention_starting_node", "feat_2d_weights", block),
      };
      const ending = {
        ...endingBase,
        pairProjectionWeight: await stacked("triangle_attention_ending_node", "feat_2d_weights", block),
      };
      const outerProductMean = {
        layerNormScale: await stacked("outer_product_mean/layer_norm_input", "scale", block),
        layerNormOffset: await stacked("outer_product_mean/layer_norm_input", "offset", block),
        leftWeight: await stacked("outer_product_mean/left_projection", "weights", block),
        leftBias: await stacked("outer_product_mean/left_projection", "bias", block),
        rightWeight: await stacked("outer_product_mean/right_projection", "weights", block),
        rightBias: await stacked("outer_product_mean/right_projection", "bias", block),
        outputWeight: await stacked("outer_product_mean", "output_w", block),
        outputBias: await stacked("outer_product_mean", "output_b", block),
      };
      blockWeights.push({
        msaRowAttention: row,
        msaColumnAttention: await attention("msa_column_attention", block),
        msaTransition: await transition("msa_transition", block),
        outerProductMean,
        triangleMultiplicationOutgoing: await triangle("triangle_multiplication_outgoing", cZ, block),
        triangleMultiplicationIncoming: await triangle("triangle_multiplication_incoming", cZ, block),
        triangleAttentionStarting: starting,
        triangleAttentionEnding: ending,
        pairTransition: await transition("pair_transition", block),
      });
    }
    const result = await new EvoformerStackGpu(device).run({
      msa,
      pair,
      msaMask: await store.tensor("stackMsaMask"),
      pairMask: await store.tensor("stackPairMask"),
      sequences: msaShape[0],
      length: msaShape[1],
      cM: msaShape[2],
      cZ,
      cOuter: blockWeights[0] .outerProductMean.leftBias.length,
      triangleHidden: blockWeights[0] .triangleMultiplicationOutgoing.linearAPBias.length,
      blockWeights,
    });
    const msaMetrics = errorMetrics(result.msa, await store.tensor("stackExpectedMsa"));
    const pairMetrics = errorMetrics(result.pair, await store.tensor("stackExpectedPair"));
    expect(msaMetrics.meanAbsoluteError).toBeLessThan(4e-3);
    // The bounded-memory tiled OPM changes FP32 summation order slightly over 48 blocks.
    expect(msaMetrics.maxAbsoluteError).toBeLessThan(1.2e-1);
    expect(pairMetrics.meanAbsoluteError).toBeLessThan(6e-3);
    expect(pairMetrics.maxAbsoluteError).toBeLessThan(2e-1);
  }, 60_000);
});
