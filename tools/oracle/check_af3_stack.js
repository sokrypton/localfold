/**
 * Run the whole 48-block pairformer stack and watch the error accumulate.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --float32 \
 *         --capture 'trunk_pairformer/__call__$' --out af3-oracle-stack-f32.json
 *     node tools/oracle/check_af3_stack.js
 *
 * 🔴 A PER-BLOCK CHECK DOES NOT ANSWER THIS. check_af3_block.js feeds every
 * block the oracle's own input, so each block's error is measured against a
 * correct starting point and no error is ever carried. The trunk does not work
 * that way: block 2 sees block 1's mistakes, and forty-eight blocks of a
 * residual stack can either wash a small divergence out or compound it. Which
 * one happens is a property of the stack, not of any block in it, and it is the
 * number that decides whether the graph is usable.
 *
 * So this runs OUR blocks from one captured boundary and never resynchronises,
 * printing where our trajectory sits against the oracle's as depth grows.
 */
import { join } from "node:path";

import { pairformerBlock } from "../../src/af3/pairformer-reference.js";
import { ROOT, captures, compare, layer, loadDump, loadTensors } from "./af3-bundle.js";

const STACK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer";

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
    singleChannels: 384,
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
    singlePairLogitsNormScale: at("single_pair_logits_norm/scale"),
    singlePairLogitsNormOffset: at("single_pair_logits_norm/offset"),
    singlePairLogitsProjection: at("single_pair_logits_projection/weights"),
    singleAttention: {
      heads: 16,
      dimension: 24,
      layerNormScale: at("single_attention_layer_norm/scale"),
      layerNormOffset: at("single_attention_layer_norm/offset"),
      qProjection: at("single_attention_q_projection/weights"),
      qBias: at("single_attention_q_projection/bias"),
      kProjection: at("single_attention_k_projection/weights"),
      vProjection: at("single_attention_v_projection/weights"),
      gatingQuery: at("single_attention_gating_query/weights"),
      outputProjection: at("single_attention_transition2/weights"),
    },
    singleTransition: {
      inputLayerNormScale: at("single_transition/input_layer_norm/scale"),
      inputLayerNormOffset: at("single_transition/input_layer_norm/offset"),
      transition1: at("single_transition/transition1/weights"),
      transition2: at("single_transition/transition2/weights"),
    },
  };
}

async function main() {
  const model = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3-f32";
  const dump = await loadDump("af3-oracle-stack-f32.json");
  const { manifest, tensors } = await loadTensors(join(ROOT, model));
  const at = captures(dump, "dump_af3_trunk.py --float32"
    + " --capture 'trunk_pairformer/__call__$' --out af3-oracle-stack-f32.json");

  const tokens = dump.tokens;
  const seqMask = Float32Array.from(dump.inputs.seq_mask.data);
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }

  // Start from block 0's captured output and never look at the oracle again.
  let state = {
    pair: at("trunk_pairformer/__call__:[0]#0"),
    single: at("trunk_pairformer/__call__:[1]#0"),
    pairMask,
    seqMask,
    tokens,
  };
  const dialect = { swapTransposedBias: dump.model !== "alphafold3" };
  const depth = dump.pairformerBlocks;

  console.log(`${dump.model}, ${tokens} tokens, blocks 1..${depth - 1} run without`
    + ` resynchronising, weights from ${model}/ (${manifest.bundle.encoding})`);
  console.log("  block    pair relRMS   single relRMS   pair RMS");
  for (let index = 1; index < depth; index += 1) {
    const next = pairformerBlock(state, blockWeights(tensors, index), dialect);
    state = { ...state, pair: next.pair, single: next.single };
    if (index % 8 === 0 || index === depth - 1) {
      const pair = compare(at(`trunk_pairformer/__call__:[0]#${index}`), next.pair);
      const single = compare(at(`trunk_pairformer/__call__:[1]#${index}`), next.single);
      console.log(`  ${String(index).padStart(5)}    ${pair.relative.toExponential(3)}`
        + `      ${single.relative.toExponential(3)}      ${pair.rms.toFixed(1)}`);
    }
  }
}

await main();
