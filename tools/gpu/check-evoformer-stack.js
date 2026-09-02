/**
 * The 48-block AF2 evoformer stack against AlphaFold's own activations.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-evoformer-stack.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-evoformer-stack.js --variant=subgroup-key32
 *
 * 🔴 THIS IS THE ONLY OFFICIAL-VALUE GATE FOR AF2 THAT CAN RUN ON THIS MACHINE.
 * `npm run test:gpu` cannot: the Dawn node binding fails to load here - "built
 * for macOS 26.0 which is newer than running OS" - and it never has, which is
 * why tools/gpu-chrome.mjs exists. Worse,
 * test/evoformer-attention.gpu.test.js, the per-kernel attention check, names a
 * fixture that is not in the repository at all
 * (evoformer/model1-query-59-block0), so it could not run even with Dawn
 * working. The stack fixture beside it IS checked in, and a whole-stack
 * comparison is the stronger gate anyway: it is 48 blocks of every kernel
 * against the reference implementation's output rather than one kernel against
 * one stage.
 *
 * 🔴 --variant EXISTS SO THE ATTENTION KERNELS CAN BE COMPARED ON REAL WEIGHTS.
 * selectAttentionFlashKernel's default moved from subgroup-key32 to the
 * register-resident kernel on timing grounds; this is what says the two agree
 * with AlphaFold and not merely with each other.
 *
 * 🔴 AND IT NEEDS CAPTURES THIS CHECKOUT MAY NOT HAVE.
 * `test/fixtures/evoformer/` is gitignored - "full-model captures are local
 * development assets", says .gitignore - so a fresh clone has 26 of the
 * manifest's 530 tensors and this cannot run. That is by design and not a bug,
 * but it does mean AF2 has NO official-value gate on a machine without them,
 * and a change to its kernels rests on differential evidence alone. Recover
 * them by re-running the capture that produced the fixture, then this works
 * unchanged.
 *
 * The tolerances are test/evoformer-stack.gpu.test.js's, unchanged: the tiled
 * outer product mean shifts f32 summation order slightly over 48 blocks.
 */
import { EvoformerStackGpu } from "../../src/evoformer/stack.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";
import { errorMetrics } from "../../src/triangle/types.js";

const MANIFEST = "/test/fixtures/evoformer/model1-query-59-stack/manifest.json";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function transpose(input, rows, columns) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[column * rows + row] = input[row * columns + column];
    }
  }
  return output;
}

export async function main(device, args) {
  const variant = option(args, "variant", "auto");
  try {
    return await check(device, variant);
  } catch (error) {
    if (!/failed to load tensor/.test(error.message)) throw error;
    throw new Error(`${error.message} - test/fixtures/evoformer/ is gitignored and this`
      + " checkout has only the small subset the CPU tests need, so the official-value"
      + " comparison cannot run here. See the note at the top of this file.");
  }
}

async function check(device, variant) {
  const store = await HttpTensorStore.open(MANIFEST);
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
    flashVariant: variant,
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
  const bounds = { msaMean: 4e-3, msaMax: 1.2e-1, pairMean: 6e-3, pairMax: 2e-1 };
  const failures = [];
  if (!(msaMetrics.meanAbsoluteError < bounds.msaMean)) failures.push("msa mean");
  if (!(msaMetrics.maxAbsoluteError < bounds.msaMax)) failures.push("msa max");
  if (!(pairMetrics.meanAbsoluteError < bounds.pairMean)) failures.push("pair mean");
  if (!(pairMetrics.maxAbsoluteError < bounds.pairMax)) failures.push("pair max");
  const round = (metrics) => ({
    mean: Number(metrics.meanAbsoluteError.toExponential(3)),
    max: Number(metrics.maxAbsoluteError.toExponential(3)),
  });
  return {
    variant, blocks, bounds,
    msa: round(msaMetrics), pair: round(pairMetrics),
    verdict: failures.length === 0 ? "PASS" : `FAIL: ${failures.join(", ")}`,
  };
}
