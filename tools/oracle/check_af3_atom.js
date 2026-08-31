/**
 * Check every per-atom conditioning embedding against AF3, term by term.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 0 --float32 \
 *         --capture 'evoformer_conditioning_embed_ref|evoformer_conditioning_single_to_pair|evoformer_conditioning_embed_pair' \
 *         --out af3-oracle-atom-f32.json
 *     node tools/oracle/check_af3_atom.js
 *
 * Nine terms, each isolated: the five reference embeddings that are summed into
 * the per-atom single conditioning, the two projections that turn it into pair
 * conditioning, and the two geometric pair terms.
 *
 * 🔴 EXACT HERE DOES NOT MEAN EXACT. Every standard residue's reference
 * conformer is NEUTRAL, so the charge channel is identically zero on any
 * protein and arcsinh(charge) cannot be told from charge. See
 * test/af3-atom-conditioning.test.js, which tests that by hand.
 */
import { join } from "node:path";

import { perAtomConditioning, perAtomPairConditioning }
  from "../../src/af3/atom-conditioning-reference.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { ROOT, loadDump, loadTensors, report } from "./af3-bundle.js";

const model = process.argv.includes("--model")
  ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3-f32";
const dump = await loadDump("af3-oracle-atom-f32.json");
const { tensors } = await loadTensors(join(ROOT, model));
const C = "diffuser/evoformer_conditioning";
const w = (n) => tensors.get(`${C}_${n}/weights`).data;
const ref = (k) => Float32Array.from(dump.outputs[k].data);
const inp = (k) => dump.inputs[k].data;

const tokens = dump.tokens, dense = 24;
const reference = {
  positions: Float32Array.from(inp("ref_pos")),
  mask: Float32Array.from(inp("ref_mask")),
  element: inp("ref_element"),
  charge: Float32Array.from(inp("ref_charge")),
  atomNameChars: inp("ref_atom_name_chars"),
};
console.log(`${dump.model}, ${tokens} tokens x ${dense} dense atom slots,`
  + ` weights from ${model}/`);

// each embedding term on its own
const rows = tokens * dense;
report("pos", ref(`${C}_embed_ref_pos/__call__`),
       linear(reference.positions, rows, 3, 128, w("embed_ref_pos")));
const maskCol = Float32Array.from(reference.mask);
report("mask", ref(`${C}_embed_ref_mask/__call__`),
       linear(maskCol, rows, 1, 128, w("embed_ref_mask")));
const el = new Float32Array(rows * 128);
for (let i = 0; i < rows; i++) el[i * 128 + reference.element[i]] = 1;
report("element", ref(`${C}_embed_ref_element/__call__`),
       linear(el, rows, 128, 128, w("embed_ref_element")));
const ch = new Float32Array(rows);
for (let i = 0; i < rows; i++) ch[i] = Math.asinh(reference.charge[i]);
report("charge", ref(`${C}_embed_ref_charge/__call__`),
       linear(ch, rows, 1, 128, w("embed_ref_charge")));
const nm = new Float32Array(rows * 256);
for (let i = 0; i < rows; i++) for (let k = 0; k < 4; k++)
  nm[i * 256 + k * 64 + reference.atomNameChars[i * 4 + k]] = 1;
report("name", ref(`${C}_embed_ref_atom_name/__call__`),
       linear(nm, rows, 256, 128, w("embed_ref_atom_name")));

// the summed, masked conditioning -> its two pair projections
const act = perAtomConditioning(reference, tokens, dense, {
  channels: 128,
  embedRefPos: w("embed_ref_pos"), embedRefMask: w("embed_ref_mask"),
  embedRefElement: w("embed_ref_element"), embedRefCharge: w("embed_ref_charge"),
  embedRefAtomName: w("embed_ref_atom_name"),
});
const rect = Float32Array.from(act, (v) => (v > 0 ? v : 0));
report("row", ref(`${C}_single_to_pair_cond_row/__call__`),
       linear(rect, rows, 128, 16, w("single_to_pair_cond_row")));
report("col", ref(`${C}_single_to_pair_cond_col/__call__`),
       linear(rect, rows, 128, 16, w("single_to_pair_cond_col")));

const pair = perAtomPairConditioning(reference, act, tokens, dense, {
  channels: 128, pairChannels: 16,
  singleToPairCondRow: w("single_to_pair_cond_row"),
  singleToPairCondCol: w("single_to_pair_cond_col"),
  embedPairOffsets: w("embed_pair_offsets"),
  embedPairDistances: w("embed_pair_distances"),
});
// the oracle captures the two pair terms separately, so rebuild what it shows
const off = new Float32Array(tokens*dense*dense*3);
const dist = new Float32Array(tokens*dense*dense);
for (let t=0;t<tokens;t++) for (let a=0;a<dense;a++) for (let b=0;b<dense;b++) {
  let sq=0; for (let x=0;x<3;x++){ const d = reference.positions[(t*dense+a)*3+x]-reference.positions[(t*dense+b)*3+x];
    off[((t*dense+a)*dense+b)*3+x]=d; sq+=d*d; }
  dist[(t*dense+a)*dense+b]=1/(1+sq);
}
report("offsets", ref(`${C}_embed_pair_offsets/__call__`),
       linear(off, tokens*dense*dense, 3, 16, w("embed_pair_offsets")));
report("dists", ref(`${C}_embed_pair_distances/__call__`),
       linear(dist, tokens*dense*dense, 1, 16, w("embed_pair_distances")));
