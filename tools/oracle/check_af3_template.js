/**
 * Check the template embedder against AF3, on a protein with NO templates.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 0 --float32 \
 *       --capture 'evoformer/template_embedding/__call__$|evoformer/__call__$|msa_stack/__call__$' \
 *       --out oracle-dumps/af3-oracle-embed-f32.json
 *     node tools/oracle/check_af3_template.js
 *
 * 🔴 "NO TEMPLATES" IS THE INTERESTING CASE, NOT THE TRIVIAL ONE. The module's
 * output here has std 13.1 against a pair whose own std is 55, because three of
 * its nine input features are the QUERY's aatype and pair representation rather
 * than template geometry. Anything that treats a missing template embedder as a
 * zero is about a quarter wrong from the first block.
 *
 * The template GEOMETRY features cannot be checked by this and are not
 * implemented; src/af3/template-reference.js raises on a real template rather
 * than running code no measurement covers.
 */
import { join } from "node:path";

import { embed } from "../../src/af3/embedder-reference.js";
import { templateEmbedding } from "../../src/af3/template-reference.js";
import * as B from "./af3-bundle.js";

const dump = await B.loadDump("oracle-dumps/af3-oracle-embed-f32.json");
const { tensors } = await B.loadTensors(join(B.ROOT, "model-af3-f32"));
const EVO = "diffuser/evoformer";
const TE = `${EVO}/template_embedding`;
const STE = `${TE}/single_template_embedding`;
const IT = `${STE}/__layer_stack_no_per_layer/template_embedding_iteration`;
const T = (n) => tensors.get(n).data;
const at = (leaf, i) => B.layer(tensors, `${IT}/${leaf}`, i);
const inp = (k) => dump.inputs[k].data;
const ref = (k) => Float32Array.from(dump.outputs[k].data);

const tokens = dump.tokens;
const seqMask = Float32Array.from(inp("seq_mask"));
const pairMask = new Float32Array(tokens * tokens);
for (let i=0;i<tokens;i++) for (let j=0;j<tokens;j++) pairMask[i*tokens+j]=seqMask[i]*seqMask[j];

const tri = (d, i) => ({
  leftNormInputScale: at(`triangle_multiplication_${d}/left_norm_input/scale`, i),
  leftNormInputOffset: at(`triangle_multiplication_${d}/left_norm_input/offset`, i),
  projection: at(`triangle_multiplication_${d}/projection/weights`, i),
  gate: at(`triangle_multiplication_${d}/gate/weights`, i),
  centerNormScale: at(`triangle_multiplication_${d}/center_norm/scale`, i),
  centerNormOffset: at(`triangle_multiplication_${d}/center_norm/offset`, i),
  outputProjection: at(`triangle_multiplication_${d}/output_projection/weights`, i),
  gatingLinear: at(`triangle_multiplication_${d}/gating_linear/weights`, i),
});
const grid = (w, i) => ({ heads: 4, dimension: 16,
  actNormScale: at(`pair_attention${w}/act_norm/scale`, i),
  actNormOffset: at(`pair_attention${w}/act_norm/offset`, i),
  pairBiasProjection: at(`pair_attention${w}/pair_bias_projection/weights`, i),
  qProjection: at(`pair_attention${w}/q_projection/weights`, i),
  kProjection: at(`pair_attention${w}/k_projection/weights`, i),
  vProjection: at(`pair_attention${w}/v_projection/weights`, i),
  gatingQuery: at(`pair_attention${w}/gating_query/weights`, i),
  outputProjection: at(`pair_attention${w}/output_projection/weights`, i) });
const blocks = [0,1].map(i => ({
  triangleMultiplicationOutgoing: tri("outgoing", i),
  triangleMultiplicationIncoming: tri("incoming", i),
  pairAttention1: grid(1, i), pairAttention2: grid(2, i),
  pairTransition: {
    inputLayerNormScale: at("pair_transition/input_layer_norm/scale", i),
    inputLayerNormOffset: at("pair_transition/input_layer_norm/offset", i),
    transition1: at("pair_transition/transition1/weights", i),
    transition2: at("pair_transition/transition2/weights", i) },
}));

const w = {
  queryChannels: 128,
  queryEmbeddingNormScale: T(`${STE}/query_embedding_norm/scale`),
  queryEmbeddingNormOffset: T(`${STE}/query_embedding_norm/offset`),
  templatePairEmbedding2: T(`${STE}/template_pair_embedding_2/weights`),
  templatePairEmbedding3: T(`${STE}/template_pair_embedding_3/weights`),
  templatePairEmbedding8: T(`${STE}/template_pair_embedding_8/weights`),
  outputLayerNormScale: T(`${STE}/output_layer_norm/scale`),
  outputLayerNormOffset: T(`${STE}/output_layer_norm/offset`),
  outputLinear: T(`${TE}/output_linear/weights`),
  blocks,
};

// the embedder's pair at the point the template embedder reads it
const em = (n) => tensors.get(`${EVO}/${n}`).data;
let captured = null;
embed({
  tokens, sequences: 1,
  targetFeat: ref(`${EVO}/__call__:target_feat`),
  templateEmbedding: (pair) => { captured = pair; return new Float32Array(pair.length); },
  msaRows: inp("msa").slice(0, tokens),
  deletionMatrix: inp("deletion_matrix").slice(0, tokens),
  features: { residueIndex: inp("residue_index"), tokenIndex: inp("token_index"),
    asymId: inp("asym_id"), entityId: inp("entity_id"), symId: inp("sym_id") },
}, {
  pairChannels: 128, singleChannels: 384, msaChannels: 64,
  targetFeatWidth: 447, relativeWidth: 139,
  leftSingle: em("left_single/weights"), rightSingle: em("right_single/weights"),
  prevEmbedding: em("prev_embedding/weights"),
  prevEmbeddingNormScale: em("prev_embedding_layer_norm/scale"),
  prevEmbeddingNormOffset: em("prev_embedding_layer_norm/offset"),
  positionActivations: em("~_relative_encoding/position_activations/weights"),
  bondEmbedding: em("bond_embedding/weights"),
  msaActivations: em("msa_activations/weights"),
  extraMsaTargetFeat: em("extra_msa_target_feat/weights"),
  singleActivations: em("single_activations/weights"),
  prevSingleEmbedding: em("prev_single_embedding/weights"),
  prevSingleEmbeddingNormScale: em("prev_single_embedding_layer_norm/scale"),
  prevSingleEmbeddingNormOffset: em("prev_single_embedding_layer_norm/offset"),
});

const out = templateEmbedding({
  pair: captured, tokens, pairMask, templates: 4, templateOccupied: false,
  templateAatype: new Int32Array(tokens),
}, w, { swapTransposedBias: false });
B.report("template", ref(`${TE}/__call__`), out);
