/**
 * Check the template embedder against AF3 with a REAL TEMPLATE in slot 0.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 0 --float32 \
 *       --sequence DIQVQVNIDDNGKNFD --template 1qys-crystal.pdb:A \
 *       --capture 'template_embedding/__call__$|evoformer/__call__$' \
 *       --capture-args 'template_embedding/__call__$' \
 *       --out af3-oracle-template-f32.json
 *     node tools/oracle/check_af3_template_geometry.js
 *
 * 🔴 THIS IS THE CHECK THAT DID NOT EXIST, AND ITS ABSENCE IS WHY THE SIX
 * GEOMETRY FEATURES WERE NEVER WRITTEN. check_af3_template.js covers the
 * empty-slot path, which is the interesting case for a de novo protein and
 * exercises three of the nine features; the other six are identically zero
 * there, so it could not tell a correct implementation of them from a wrong
 * one, from no implementation at all.
 *
 * 🔴 IT TAKES AF3'S OWN QUERY PAIR RATHER THAN REBUILDING IT. The dump
 * captures the module's ARGUMENTS, so the embedder does not have to be re-run
 * to produce its input - which means a disagreement here is in the template
 * module and cannot be in the embedder before it.
 *
 * 🔴 AND IT CHECKS ONE SLOT'S 64 CHANNELS BEFORE THE SUM. Four slots are
 * summed, divided by four, relu'd and projected to 128; by the end a wrong
 * unit vector and a wrong distogram bin are the same number. AF3 captures each
 * slot at the point the summation starts, so both are compared.
 */
import { join } from "node:path";

import { templateEmbedding } from "../../src/af3/template-reference.js";
import * as B from "./af3-bundle.js";

const dump = await B.loadDump("af3-oracle-template-f32.json");
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
for (let i = 0; i < tokens; i += 1) {
  for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
}
// A template covers one chain; every pair here is intra-chain because the
// query is one chain. asym_id is read rather than assumed so a two-chain dump
// exercises the masking instead of silently skipping it.
const asymId = inp("asym_id");
const multichainMask2d = new Float32Array(tokens * tokens);
for (let i = 0; i < tokens; i += 1) {
  for (let j = 0; j < tokens; j += 1) {
    multichainMask2d[i * tokens + j] = asymId[i] === asymId[j] ? 1 : 0;
  }
}

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
const blocks = [0, 1].map((i) => ({
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
  outputLayerNormScale: T(`${STE}/output_layer_norm/scale`),
  outputLayerNormOffset: T(`${STE}/output_layer_norm/offset`),
  outputLinear: T(`${TE}/output_linear/weights`),
  blocks,
};
for (const i of [0, 1, 2, 3, 4, 5, 6, 7, 8]) {
  w[`templatePairEmbedding${i}`] = T(`${STE}/template_pair_embedding_${i}/weights`);
}

// The template slots, exactly as AF3 featurised them.
const NUM_DENSE = 24;
const aatype = inp("template_aatype");
const atomMask = inp("template_atom_mask");
const atomPositions = inp("template_atom_positions");
const slotCount = dump.inputs.template_aatype.shape[0];
const slots = [];
for (let slot = 0; slot < slotCount; slot += 1) {
  const mask = Float32Array.from(
    atomMask.slice(slot * tokens * NUM_DENSE, (slot + 1) * tokens * NUM_DENSE));
  // 🔴 AN EMPTY SLOT IS PASSED AS UNDEFINED, NOT AS ZEROS. Zeros would compute
  // the same answer the slow way; undefined is what the shipped path uses when
  // a job has fewer templates than slots, so this checks the branch the page
  // takes rather than one only the checker takes.
  let any = false;
  for (const value of mask) if (value > 0) { any = true; break; }
  slots.push(any ? {
    aatype: Int32Array.from(aatype.slice(slot * tokens, (slot + 1) * tokens)),
    atomMask: mask,
    atomPositions: Float32Array.from(atomPositions.slice(
      slot * tokens * NUM_DENSE * 3, (slot + 1) * tokens * NUM_DENSE * 3)),
  } : undefined);
}
console.log(`${slotCount} slots, ${slots.filter(Boolean).length} occupied`
  + `, ${tokens} tokens`);

const perSlot = [];
const out = templateEmbedding({
  pair: ref(`${STE}/__call__<0#0`),
  tokens, pairMask, templates: slotCount, multichainMask2d, slots,
  onSlot: (slot, embedded) => { perSlot[slot] = embedded; },
}, w, { swapTransposedBias: false });

for (let slot = 0; slot < slotCount; slot += 1) {
  B.report(`slot ${slot}${slots[slot] ? " (real)" : " (empty)"}`,
           ref(`${STE}/__call__#${slot}`), perSlot[slot]);
}
B.report("template", ref(`${TE}/__call__`), out);
