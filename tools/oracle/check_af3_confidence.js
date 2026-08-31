/**
 * Check the confidence head against AF3: pLDDT, PAE and PDE.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 48 --diffusion 1 --float32 \
 *       --capture 'evoformer/__call__$' --out af3-oracle-confidence-f32.json
 *     node tools/oracle/check_af3_confidence.js
 *
 * The head is fed AF3's own sampled coordinates, which the run dumps as
 * `diffusion_samples/atom_positions`. That is the honest input: the confidence
 * head is the only part of AF3 that reads the structure back, so checking it on
 * coordinates we generated ourselves would be checking two things at once.
 */
import { join } from "node:path";

import { confidenceHead } from "../../src/af3/confidence-reference.js";
import { convert } from "../../src/af3/atom-encoder-reference.js";
import { pairformerBlock } from "../../src/af3/pairformer-reference.js";
import { ROOT, captures, layer, loadDump, loadTensors, report } from "./af3-bundle.js";

const HEAD = "diffuser/confidence_head";
const STACK = `${HEAD}/__layer_stack_no_per_layer/confidence_pairformer`;
const EVO = "diffuser/evoformer";

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
    heads: 4, dimension: 32,
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
    pairChannels: 128, singleChannels: 384,
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
      heads: 16, dimension: 24,
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
    ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3-full-f32";
  const dump = await loadDump("af3-oracle-confidence-f32.json");
  const { tensors } = await loadTensors(join(ROOT, model));
  const at = captures(dump, "dump_af3_trunk.py --blocks 48 --diffusion 1 --float32");
  const T = (name) => {
    const tensor = tensors.get(`${HEAD}/${name}`);
    if (tensor === undefined) {
      throw new Error(`no tensor ${HEAD}/${name}; export with --include diffuser`);
    }
    return tensor.data;
  };
  const input = (name) => dump.inputs[name].data;

  const tokens = dump.tokens;
  const dense = 24;
  // 🔴 THE HEAD READS THE SAMPLE, so it gets AF3's sample rather than one of
  // ours - the trajectory is a different PRNG draw and checking the head on our
  // own coordinates would conflate two questions.
  const sample = at("diffusion_samples/atom_positions");
  const pseudoBeta = convert({
    indices: input("token_atoms_to_pseudo_beta:gather_idxs"),
    mask: input("token_atoms_to_pseudo_beta:gather_mask"),
    count: tokens,
  }, sample, 3);

  const ours = confidenceHead({
    tokens, dense,
    pair: at(`${EVO}/__call__:pair`),
    single: at(`${EVO}/__call__:single`),
    targetFeat: at(`${EVO}/__call__:target_feat`),
    pseudoBeta,
    seqMask: Float32Array.from(input("seq_mask")),
  }, {
    pairChannels: 128, singleChannels: 384, targetFeatWidth: 447,
    leftTargetFeatProject: T("~_embed_features/left_target_feat_project/weights"),
    rightTargetFeatProject: T("~_embed_features/right_target_feat_project/weights"),
    distogramFeatProject: T("~_embed_features/distogram_feat_project/weights"),
    blocks: [0, 1, 2, 3].map((index) => blockWeights(tensors, index)),
    logitsLnScale: T("logits_ln/scale"), logitsLnOffset: T("logits_ln/offset"),
    leftHalfDistanceLogits: T("left_half_distance_logits/weights"),
    paeLogitsLnScale: T("pae_logits_ln/scale"),
    paeLogitsLnOffset: T("pae_logits_ln/offset"),
    paeLogits: T("pae_logits/weights"),
    plddtLnScale: T("plddt_logits_ln/scale"),
    plddtLnOffset: T("plddt_logits_ln/offset"),
    plddtLogits: T("plddt_logits/weights"),
    resolvedLnScale: T("experimentally_resolved_ln/scale"),
    resolvedLnOffset: T("experimentally_resolved_ln/offset"),
    experimentallyResolvedLogits: T("experimentally_resolved_logits/weights"),
  }, pairformerBlock, { swapTransposedBias: dump.model !== "alphafold3" });

  console.log(`${dump.model}, ${tokens} tokens, confidence head on AF3's own`
    + ` sample, weights from ${model}/`);
  report("pLDDT", at("predicted_lddt"), ours.plddt);
  report("PAE", at("full_pae"), ours.pae);
  report("PDE", at("full_pde"), ours.pde);
}

await main();
