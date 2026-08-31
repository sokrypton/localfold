/**
 * Run AF3's whole trunk and compare the pair, the single and the distogram.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --float32 \
 *         --capture 'evoformer/template_embedding/__call__$|evoformer/__call__$' \
 *         --out af3-oracle-trunk-f32.json
 *     node tools/oracle/check_af3_trunk.js                  # float32 weights
 *     node tools/oracle/check_af3_trunk.js --model model-af3  # ...and int8
 *
 * Embedder, four MSA blocks, forty-eight pairformer blocks, distogram head, in
 * one pass with nothing resynchronised in the middle. The distogram is the only
 * output here anybody would look at, and it is the one the per-piece checks
 * never produce.
 *
 * Every learned module is ours. What still comes from the oracle is the
 * FEATURISED BATCH - the tokenisation, the reference conformers, the MSA - none
 * of which is a neural network, and all of which a browser would build from a
 * sequence and a 21-entry conformer table.
 */
import { join } from "node:path";

import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { atomCrossAttentionEncoder, targetFeatures }
  from "../../src/af3/atom-encoder-reference.js";
import { templateEmbedding } from "../../src/af3/template-reference.js";
import { runTrunk } from "../../src/af3/trunk-reference.js";
import { ROOT, captures, layer, loadDump, loadTensors, report } from "./af3-bundle.js";

const EVO = "diffuser/evoformer";
const MSA = `${EVO}/__layer_stack_no_per_layer/msa_stack`;
const PAIRFORMER = `${EVO}/__layer_stack_no_per_layer_1/trunk_pairformer`;

const triangle = (at, direction) => ({
  leftNormInputScale: at(`triangle_multiplication_${direction}/left_norm_input/scale`),
  leftNormInputOffset: at(`triangle_multiplication_${direction}/left_norm_input/offset`),
  projection: at(`triangle_multiplication_${direction}/projection/weights`),
  gate: at(`triangle_multiplication_${direction}/gate/weights`),
  centerNormScale: at(`triangle_multiplication_${direction}/center_norm/scale`),
  centerNormOffset: at(`triangle_multiplication_${direction}/center_norm/offset`),
  outputProjection: at(`triangle_multiplication_${direction}/output_projection/weights`),
  gatingLinear: at(`triangle_multiplication_${direction}/gating_linear/weights`),
});

const gridAttention = (at, which) => ({
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

const pairTransition = (at) => ({
  inputLayerNormScale: at("pair_transition/input_layer_norm/scale"),
  inputLayerNormOffset: at("pair_transition/input_layer_norm/offset"),
  transition1: at("pair_transition/transition1/weights"),
  transition2: at("pair_transition/transition2/weights"),
});

function msaBlockWeights(tensors, index) {
  const at = (leaf) => layer(tensors, `${MSA}/${leaf}`, index);
  return {
    pairChannels: 128,
    msaChannels: 64,
    outerProductMean: {
      outerChannels: 32,
      layerNormInputScale: at("outer_product_mean/layer_norm_input/scale"),
      layerNormInputOffset: at("outer_product_mean/layer_norm_input/offset"),
      leftProjection: at("outer_product_mean/left_projection/weights"),
      rightProjection: at("outer_product_mean/right_projection/weights"),
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
    triangleMultiplicationOutgoing: triangle(at, "outgoing"),
    triangleMultiplicationIncoming: triangle(at, "incoming"),
    pairAttention1: gridAttention(at, 1),
    pairAttention2: gridAttention(at, 2),
    pairTransition: pairTransition(at),
  };
}

function pairformerBlockWeights(tensors, index) {
  const at = (leaf) => layer(tensors, `${PAIRFORMER}/${leaf}`, index);
  return {
    pairChannels: 128,
    singleChannels: 384,
    triangleMultiplicationOutgoing: triangle(at, "outgoing"),
    triangleMultiplicationIncoming: triangle(at, "incoming"),
    pairAttention1: gridAttention(at, 1),
    pairAttention2: gridAttention(at, 2),
    pairTransition: pairTransition(at),
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

const STE = `${EVO}/template_embedding/single_template_embedding`;
const ITERATION = `${STE}/__layer_stack_no_per_layer/template_embedding_iteration`;

/** The template stack runs at 64 channels, 4 heads of 16, and factor-2 transitions. */
function templateWeights(tensors) {
  const T = (name) => tensors.get(name).data;
  const at = (leaf, index) => layer(tensors, `${ITERATION}/${leaf}`, index);
  const blocks = [0, 1].map((index) => {
    const pick = (leaf) => at(leaf, index);
    return {
      triangleMultiplicationOutgoing: triangle(pick, "outgoing"),
      triangleMultiplicationIncoming: triangle(pick, "incoming"),
      pairAttention1: { ...gridAttention(pick, 1), dimension: 16 },
      pairAttention2: { ...gridAttention(pick, 2), dimension: 16 },
      pairTransition: pairTransition(pick),
    };
  });
  return {
    queryChannels: 128,
    queryEmbeddingNormScale: T(`${STE}/query_embedding_norm/scale`),
    queryEmbeddingNormOffset: T(`${STE}/query_embedding_norm/offset`),
    templatePairEmbedding2: T(`${STE}/template_pair_embedding_2/weights`),
    templatePairEmbedding3: T(`${STE}/template_pair_embedding_3/weights`),
    templatePairEmbedding8: T(`${STE}/template_pair_embedding_8/weights`),
    outputLayerNormScale: T(`${STE}/output_layer_norm/scale`),
    outputLayerNormOffset: T(`${STE}/output_layer_norm/offset`),
    outputLinear: T(`${EVO}/template_embedding/output_linear/weights`),
    blocks,
  };
}

/** target_feat, built from the reference conformers rather than read back. */
function buildTargetFeat(tensors, dump, tokens) {
  const C = "diffuser/evoformer_conditioning";
  const w = (name) => tensors.get(`${C}_${name}/weights`).data;
  const T = (name) => tensors.get(name).data;
  const input = (name) => dump.inputs[name].data;
  const dense = 24;
  const subsets = 9;
  const queries = 32;
  const keys = 128;
  const reference = {
    positions: Float32Array.from(input("ref_pos")),
    mask: Float32Array.from(input("ref_mask")),
    element: input("ref_element"),
    charge: Float32Array.from(input("ref_charge")),
    atomNameChars: input("ref_atom_name_chars"),
  };
  const conditioning = perAtomConditioning(reference, tokens, dense, {
    channels: 128,
    embedRefPos: w("embed_ref_pos"), embedRefMask: w("embed_ref_mask"),
    embedRefElement: w("embed_ref_element"), embedRefCharge: w("embed_ref_charge"),
    embedRefAtomName: w("embed_ref_atom_name"),
  });
  const S = `${C}_atom_transformer_encoder`;
  const P = `${S}/__layer_stack_with_per_layer/evoformer_conditioning_atom_transformer_encoder`;
  const at = (leaf, index) => layer(tensors, `${P}${leaf}`, index);
  const blocks = [0, 1, 2].map((index) => ({
    qProjection: at("q_projection/weights", index), qBias: at("q_projection/bias", index),
    kProjection: at("k_projection/weights", index),
    vProjection: at("v_projection/weights", index),
    gatingQuery: at("gating_query/weights", index),
    qSingleCondLayerNormScale: at("qsingle_cond_layer_norm/scale", index),
    qSingleCondScaleWeights: at("qsingle_cond_scale/weights", index),
    qSingleCondScaleBias: at("qsingle_cond_scale/bias", index),
    qSingleCondBias: at("qsingle_cond_bias/weights", index),
    kSingleCondLayerNormScale: at("ksingle_cond_layer_norm/scale", index),
    kSingleCondScaleWeights: at("ksingle_cond_scale/weights", index),
    kSingleCondScaleBias: at("ksingle_cond_scale/bias", index),
    kSingleCondBias: at("ksingle_cond_bias/weights", index),
    Transition2: at("transition2/weights", index),
    AdaptiveZeroCondWeights: at("adaptive_zero_cond/weights", index),
    AdaptiveZeroCondBias: at("adaptive_zero_cond/bias", index),
    ffwSingleCondLayerNormScale: at("ffw_single_cond_layer_norm/scale", index),
    ffwSingleCondScaleWeights: at("ffw_single_cond_scale/weights", index),
    ffwSingleCondScaleBias: at("ffw_single_cond_scale/bias", index),
    ffwSingleCondBias: at("ffw_single_cond_bias/weights", index),
    ffwTransition1: at("ffw_transition1/weights", index),
    ffwTransition2: at("ffw_transition2/weights", index),
    ffwAdaptiveZeroCondWeights: at("ffw_adaptive_zero_cond/weights", index),
    ffwAdaptiveZeroCondBias: at("ffw_adaptive_zero_cond/bias", index),
  }));
  const gather = (name, count) => ({
    indices: input(`${name}:gather_idxs`), mask: input(`${name}:gather_mask`), count,
  });
  const encoded = atomCrossAttentionEncoder({
    shape: { tokens, dense, subsets, queries, keys },
    conditioning,
    atomMask: reference.mask,
    refPos: reference.positions,
    refSpaceUid: Float32Array.from(input("ref_space_uid")),
    tokenAtomsToQueries: gather("token_atoms_to_queries", subsets * queries),
    queriesToKeys: gather("queries_to_keys", subsets * keys),
    queriesToTokenAtoms: gather("queries_to_token_atoms", tokens * dense),
  }, {
    channels: 128, pairChannels: 16, heads: 4, dimension: 32, perTokenChannels: 384,
    singleToPairCondRow: w("single_to_pair_cond_row_1"),
    singleToPairCondCol: w("single_to_pair_cond_col_1"),
    embedPairOffsets: w("embed_pair_offsets_1"),
    embedPairDistances: w("embed_pair_distances_1"),
    embedPairOffsetsValid: w("embed_pair_offsets_valid"),
    pairMlp1: w("pair_mlp_1"), pairMlp2: w("pair_mlp_2"), pairMlp3: w("pair_mlp_3"),
    pairInputLayerNormScale: T(`${S}/pair_input_layer_norm/scale`),
    pairLogitsProjection: T(`${S}/pair_logits_projection/weights`),
    projectAtomFeaturesForAggr: w("project_atom_features_for_aggr"),
    blocks,
  });
  return targetFeatures({
    aatype: input("aatype"), profile: input("profile"),
    deletionMean: input("deletion_mean"), atomFeatures: encoded.tokenAct,
  }, tokens);
}

function embedderWeights(tensors) {
  const at = (name) => {
    const tensor = tensors.get(`${EVO}/${name}`);
    if (tensor === undefined) throw new Error(`no tensor named ${EVO}/${name}`);
    return tensor.data;
  };
  return {
    pairChannels: 128,
    singleChannels: 384,
    msaChannels: 64,
    targetFeatWidth: 447,
    relativeWidth: 139,
    leftSingle: at("left_single/weights"),
    rightSingle: at("right_single/weights"),
    prevEmbedding: at("prev_embedding/weights"),
    prevEmbeddingNormScale: at("prev_embedding_layer_norm/scale"),
    prevEmbeddingNormOffset: at("prev_embedding_layer_norm/offset"),
    positionActivations: at("~_relative_encoding/position_activations/weights"),
    bondEmbedding: at("bond_embedding/weights"),
    msaActivations: at("msa_activations/weights"),
    extraMsaTargetFeat: at("extra_msa_target_feat/weights"),
    singleActivations: at("single_activations/weights"),
    prevSingleEmbedding: at("prev_single_embedding/weights"),
    prevSingleEmbeddingNormScale: at("prev_single_embedding_layer_norm/scale"),
    prevSingleEmbeddingNormOffset: at("prev_single_embedding_layer_norm/offset"),
  };
}

async function main() {
  const model = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3-f32";
  const dump = await loadDump("af3-oracle-trunk-f32.json");
  const { manifest, tensors } = await loadTensors(join(ROOT, model));
  const at = captures(dump, "dump_af3_trunk.py --float32 with the trunk capture"
    + " (see the header of this file)");

  const tokens = dump.tokens;
  const input = (name) => dump.inputs[name].data;
  const seqMask = Float32Array.from(input("seq_mask"));
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  const sequences = 1;   // the trunk's num_msa; see check_af3_embedder.js

  const weights = {
    embedder: embedderWeights(tensors),
    msaBlocks: [0, 1, 2, 3].map((index) => msaBlockWeights(tensors, index)),
    pairformerBlocks: Array.from({ length: dump.pairformerBlocks },
                                 (_, index) => pairformerBlockWeights(tensors, index)),
    distogram: { halfLogits: tensors.get("diffuser/distogram_head/half_logits/weights").data },
  };

  const started = Date.now();
  const result = runTrunk({
    tokens,
    sequences,
    targetFeat: buildTargetFeat(tensors, dump, tokens),
    templateEmbedding: (pair) => templateEmbedding({
      pair, tokens, pairMask,
      templates: dump.inputs.template_aatype.shape[0],
      templateOccupied: dump.inputs.template_atom_mask.data.some(Boolean),
      templateAatype: new Int32Array(tokens),
    }, templateWeights(tensors), { swapTransposedBias: dump.model !== "alphafold3" }),
    msaRows: input("msa").slice(0, sequences * tokens),
    deletionMatrix: input("deletion_matrix").slice(0, sequences * tokens),
    msaMask: Float32Array.from(input("msa_mask")).subarray(0, sequences * tokens),
    pairMask,
    seqMask,
    features: {
      residueIndex: input("residue_index"),
      tokenIndex: input("token_index"),
      asymId: input("asym_id"),
      entityId: input("entity_id"),
      symId: input("sym_id"),
    },
  }, weights, { swapTransposedBias: dump.model !== "alphafold3" });
  const elapsed = (Date.now() - started) / 1000;

  console.log(`${dump.model}, ${tokens} tokens, embedder + 4 MSA blocks +`
    + ` ${dump.pairformerBlocks} pairformer blocks + distogram head`
    + `  (${elapsed.toFixed(1)} s, ${model}/, ${manifest.bundle.encoding})`);
  console.log("  nothing is taken from the oracle but the featurised batch");
  report("pair", at(`${EVO}/__call__:pair`), result.pair);
  report("single", at(`${EVO}/__call__:single`), result.single);
  report("logits", at("distogram/distogram"), result.logits);
  report("contact", at("distogram/contact_probs"), result.contactProbs);
  report("bins", at("distogram/bin_edges"), result.binEdges);
}

await main();
