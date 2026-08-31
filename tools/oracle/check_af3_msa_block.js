/**
 * Run our AF3 MSA block against the oracle's, with the real weights.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 0 --float32 \
 *         --capture 'msa_stack/__call__$' --out af3-oracle-msa-f32.json
 *     node tools/oracle/check_af3_msa_block.js
 *
 * Same shape as check_af3_block.js: the capture records every block's output in
 * execution order, so block N-1's output is block N's input and both come from
 * one run. The MSA stack is only four blocks deep, so `--blocks 0` is enough -
 * it truncates the PAIRFORMER, which runs after this stack and cannot affect it.
 *
 * 🔴 --float32 IS NOT OPTIONAL. AF3's trunk computes in bfloat16, whose relative
 * epsilon is 3.9e-3, and a bfloat16 dump cannot tell a correct block from one
 * that is a third of a percent wrong. See dump_af3_trunk.py.
 */
import { join } from "node:path";

import { msaBlock } from "../../src/af3/msa-reference.js";
import { ROOT, captures, layer, loadDump, loadTensors, report } from "./af3-bundle.js";

const STACK = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack";

function blockWeights(tensors, index) {
  const at = (leaf) => layer(tensors, `${STACK}/${leaf}`, index);
  const triangle = (direction) => ({
    leftNormInputScale: at(`triangle_multiplication_${direction}/left_norm_input/scale`),
    leftNormInputOffset: at(`triangle_multiplication_${direction}/left_norm_input/offset`),
    projection: at(`triangle_multiplication_${direction}/projection/weights`),
    gate: at(`triangle_multiplication_${direction}/gate/weights`),
    centerNormScale: at(`triangle_multiplication_${direction}/center_norm/scale`),
    centerNormOffset: at(`triangle_multiplication_${direction}/center_norm/offset`),
    outputProjection: at(`triangle_multiplication_${direction}/output_projection/weights`),
    gatingLinear: at(`triangle_multiplication_${direction}/gating_linear/weights`),
  });
  const gridAttention = (which) => ({
    heads: 4,
    dimension: 32,
    actNormScale: at(`pair_attention${which}/act_norm/scale`),
    actNormOffset: at(`pair_attention${which}/act_norm/offset`),
    pairBiasProjection: at(`pair_attention${which}/pair_bias_projection/weights`),
    qProjection: at(`pair_attention${which}/q_projection/weights`),
    kProjection: at(`pair_attention${which}/k_projection/weights`),
    vProjection: at(`pair_attention${which}/v_projection/weights`),
    gatingQuery: at(`pair_attention${which}/gating_query/weights`),
    outputProjection: at(`pair_attention${which}/output_projection/weights`),
  });
  return {
    pairChannels: 128,
    msaChannels: 64,
    outerProductMean: {
      outerChannels: 32,
      layerNormInputScale: at("outer_product_mean/layer_norm_input/scale"),
      layerNormInputOffset: at("outer_product_mean/layer_norm_input/offset"),
      leftProjection: at("outer_product_mean/left_projection/weights"),
      rightProjection: at("outer_product_mean/right_projection/weights"),
      // ...output_w and output_b are raw hk.get_parameter, not a Linear, so
      // they have no /weights leaf. The names are the whole leaf.
      outputW: at("outer_product_mean/output_w"),
      outputB: at("outer_product_mean/output_b"),
    },
    msaAttention1: {
      heads: 8,
      dimension: 8,
      actNormScale: at("msa_attention1/act_norm/scale"),
      actNormOffset: at("msa_attention1/act_norm/offset"),
      pairNormScale: at("msa_attention1/pair_norm/scale"),
      pairNormOffset: at("msa_attention1/pair_norm/offset"),
      pairLogits: at("msa_attention1/pair_logits/weights"),
      vProjection: at("msa_attention1/v_projection/weights"),
      gatingQuery: at("msa_attention1/gating_query/weights"),
      outputProjection: at("msa_attention1/output_projection/weights"),
    },
    msaTransition: {
      inputLayerNormScale: at("msa_transition/input_layer_norm/scale"),
      inputLayerNormOffset: at("msa_transition/input_layer_norm/offset"),
      transition1: at("msa_transition/transition1/weights"),
      transition2: at("msa_transition/transition2/weights"),
    },
    triangleMultiplicationOutgoing: triangle("outgoing"),
    triangleMultiplicationIncoming: triangle("incoming"),
    pairAttention1: gridAttention(1),
    pairAttention2: gridAttention(2),
    pairTransition: {
      inputLayerNormScale: at("pair_transition/input_layer_norm/scale"),
      inputLayerNormOffset: at("pair_transition/input_layer_norm/offset"),
      transition1: at("pair_transition/transition1/weights"),
      transition2: at("pair_transition/transition2/weights"),
    },
  };
}

async function main() {
  const model = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3-f32";
  const index = Number(process.argv.includes("--block")
    ? process.argv[process.argv.indexOf("--block") + 1] : 1);
  const dump = await loadDump("af3-oracle-msa-f32.json");
  const { manifest, tensors } = await loadTensors(join(ROOT, model));
  if (manifest.model.name !== dump.model) {
    throw new Error(`${model}/ holds ${manifest.model.name} but the oracle ran`
      + ` ${dump.model}`);
  }
  const at = captures(dump, "dump_af3_trunk.py --blocks 0 --float32"
    + " --capture 'msa_stack/__call__$' --out af3-oracle-msa-f32.json");

  const tokens = dump.tokens;
  const msaShape = dump.outputs[`msa_stack/__call__:msa#${index}`].shape;
  const sequences = msaShape[0];
  const seqMask = Float32Array.from(dump.inputs.seq_mask.data);
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  // The dumped msa_mask covers the full MSA the featuriser built; the trunk
  // runs the first `sequences` rows of it.
  const msaMask = Float32Array.from(dump.inputs.msa_mask.data)
    .subarray(0, sequences * tokens);

  const result = msaBlock({
    msa: at(`msa_stack/__call__:msa#${index - 1}`),
    pair: at(`msa_stack/__call__:pair#${index - 1}`),
    msaMask,
    pairMask,
    sequences,
    tokens,
  }, blockWeights(tensors, index), {
    swapTransposedBias: dump.model !== "alphafold3",
  });

  console.log(`${dump.model}, ${tokens} tokens, ${sequences} sequence(s),`
    + ` MSA block ${index}, weights from ${model}/ (${manifest.bundle.encoding})`);
  report("msa", at(`msa_stack/__call__:msa#${index}`), result.msa);
  report("pair", at(`msa_stack/__call__:pair#${index}`), result.pair);
}

await main();
