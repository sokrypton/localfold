/**
 * Build target_feat from chemistry and check it against AF3.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 0 --float32 \
 *       --capture 'evoformer_conditioning_embed_ref|evoformer_conditioning_single_to_pair|evoformer_conditioning_embed_pair|atom_transformer_encoder/__call__$|project_atom_features|evoformer/__call__$' \
 *       --out af3-oracle-atom-f32.json
 *     node tools/oracle/check_af3_target_feat.js
 *
 * This is the bottom of the model reached from the top: reference conformers
 * in, the 447 columns the whole trunk reads out. It closes the larger of the
 * two stubs the trunk checker reports.
 */
import { join } from "node:path";

import { perAtomConditioning } from "../../src/af3/atom-conditioning-reference.js";
import { atomCrossAttentionEncoder, targetFeatures }
  from "../../src/af3/atom-encoder-reference.js";
import * as B from "./af3-bundle.js";

const dump = await B.loadDump("af3-oracle-atom-f32.json");
const { tensors } = await B.loadTensors(join(B.ROOT, "model-af3-f32"));
const C = "diffuser/evoformer_conditioning";
const w = (n) => tensors.get(`${C}_${n}/weights`).data;
const T = (n) => tensors.get(n).data;
const inp = (k) => dump.inputs[k].data;
const gather = (name, count) => ({
  indices: inp(`${name}:gather_idxs`), mask: inp(`${name}:gather_mask`), count,
});

const tokens = dump.tokens, dense = 24, subsets = 9, queries = 32, keys = 128;
const reference = {
  positions: Float32Array.from(inp("ref_pos")),
  mask: Float32Array.from(inp("ref_mask")),
  element: inp("ref_element"), charge: Float32Array.from(inp("ref_charge")),
  atomNameChars: inp("ref_atom_name_chars"),
};
const conditioning = perAtomConditioning(reference, tokens, dense, {
  channels: 128, embedRefPos: w("embed_ref_pos"), embedRefMask: w("embed_ref_mask"),
  embedRefElement: w("embed_ref_element"), embedRefCharge: w("embed_ref_charge"),
  embedRefAtomName: w("embed_ref_atom_name"),
});

const S = `${C}_atom_transformer_encoder`;
const P = `${S}/__layer_stack_with_per_layer/evoformer_conditioning_atom_transformer_encoder`;
const blockAt = (leaf, i) => B.layer(tensors, `${P}${leaf}`, i);
const blocks = [0,1,2].map(i => ({
  qProjection: blockAt("q_projection/weights", i), qBias: blockAt("q_projection/bias", i),
  kProjection: blockAt("k_projection/weights", i), vProjection: blockAt("v_projection/weights", i),
  gatingQuery: blockAt("gating_query/weights", i),
  qSingleCondLayerNormScale: blockAt("qsingle_cond_layer_norm/scale", i),
  qSingleCondScaleWeights: blockAt("qsingle_cond_scale/weights", i),
  qSingleCondScaleBias: blockAt("qsingle_cond_scale/bias", i),
  qSingleCondBias: blockAt("qsingle_cond_bias/weights", i),
  kSingleCondLayerNormScale: blockAt("ksingle_cond_layer_norm/scale", i),
  kSingleCondScaleWeights: blockAt("ksingle_cond_scale/weights", i),
  kSingleCondScaleBias: blockAt("ksingle_cond_scale/bias", i),
  kSingleCondBias: blockAt("ksingle_cond_bias/weights", i),
  Transition2: blockAt("transition2/weights", i),
  AdaptiveZeroCondWeights: blockAt("adaptive_zero_cond/weights", i),
  AdaptiveZeroCondBias: blockAt("adaptive_zero_cond/bias", i),
  ffwSingleCondLayerNormScale: blockAt("ffw_single_cond_layer_norm/scale", i),
  ffwSingleCondScaleWeights: blockAt("ffw_single_cond_scale/weights", i),
  ffwSingleCondScaleBias: blockAt("ffw_single_cond_scale/bias", i),
  ffwSingleCondBias: blockAt("ffw_single_cond_bias/weights", i),
  ffwTransition1: blockAt("ffw_transition1/weights", i),
  ffwTransition2: blockAt("ffw_transition2/weights", i),
  ffwAdaptiveZeroCondWeights: blockAt("ffw_adaptive_zero_cond/weights", i),
  ffwAdaptiveZeroCondBias: blockAt("ffw_adaptive_zero_cond/bias", i),
}));

const encoded = atomCrossAttentionEncoder({
  shape: { tokens, dense, subsets, queries, keys },
  conditioning,
  atomMask: reference.mask,
  refPos: reference.positions,
  refSpaceUid: Float32Array.from(inp("ref_space_uid")),
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
const out = encoded.tokenAct;
console.log(`${dump.model}, ${tokens} tokens, ${subsets} subsets x ${queries}`
  + ` queries x ${keys} keys, from reference conformers`);

// against AF3
const ref = (k) => Float32Array.from(dump.outputs[k].data);
// (the transformer output is compared below via token_act)
const tf = ref("diffuser/evoformer/__call__:target_feat");
const atomPart = new Float32Array(tokens * 384);
for (let t = 0; t < tokens; t++) for (let c = 0; c < 384; c++) atomPart[t*384+c] = tf[t*447 + 63 + c];
B.report("token_act", atomPart, out);

// the whole target_feat: aatype one-hot, profile, deletion mean, then the atoms
const full = targetFeatures({
  aatype: inp("aatype"), profile: inp("profile"),
  deletionMean: inp("deletion_mean"), atomFeatures: out,
}, tokens);
B.report("target_feat", tf, full);
