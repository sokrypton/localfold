/**
 * OpenDDE's TEMPLATE EMBEDDER against af3-any-model, on a query with NO
 * template.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-opendde-template.js
 *
 * 🔴 "NO TEMPLATE" IS THE CASE THAT WAS WRONG. An empty slot is not a no-op -
 * its aatype one-hot still picks a row out of `template_pair_embedding_2`/`_3`
 * - and OpenDDE takes protenix's featuriser, which fills its ONE empty template
 * with the GAP restype (21) and zero-pads the rest. This port wrote 0 (ALA)
 * everywhere. Measured here: the module alone **2.83e-1 -> 1.32e-7**, and the
 * trunk seam `z_after_template` 2.07e-2 -> 3.80e-6.
 *
 * 🔴 AND THE SLOT COUNT MUST MATCH THE DUMP'S. The dump is `EMPTY=1
 * dump_af3_template.py opendde`, which runs ONE slot; a fold runs four, of
 * which only the first carries the gap. So `templates=1` is the arm that
 * compares against this dump and the other two are expected to differ - they
 * are printed to make that visible rather than to be read as failures. The
 * four-slot arithmetic is checked in the fold, by
 * `fold-opendde.js --trunk-oracle=`.
 */
import { templateEmbedding } from "../../src/af3/template-reference.js";
import { openAf3Store, templateWeights, af3Dialect } from "../../src/af3/weights.js";
const option = (a,n,f) => a.find(x=>x.startsWith(`--${n}=`))?.slice(n.length+3) ?? f;
export async function main(device, args) {
  const r = await fetch("/oracle-dumps/af3-oracle-template-opendde-empty.json");
  const d = await r.json();
  const raw = (n) => Float32Array.from(d.inputs[n].data);
  const store = await openAf3Store(option(args,"model","/model-opendde-full-f32/manifest.json"));
  store.prefetch();
  const dialect = af3Dialect(store);
  const weights = await templateWeights(store, dialect);
  const tokens = d.tokens;
  const expected = Float32Array.from(d.output.data);
  const out = {};
  for (const templates of [1, 2, 4]) {
    const got = templateEmbedding({
      tokens, pair: raw("pair"), pairMask: raw("pairMask"),
      templates, slots: [],
      asymId: Int32Array.from(raw("asymId")),
      multichainMask2d: raw("multichainMask2d"),
    }, weights, dialect);
    let e=0,s=0; for (let i=0;i<expected.length;i+=1){const x=got[i]-expected[i];e+=x*x;s+=expected[i]**2;}
    out[`templates=${templates}`] = Math.sqrt(e/s).toExponential(2);
  }
  return { tokens, slotsInDump: d.slots, blocks: weights.blocks.length,
           queryChannels: weights.queryChannels, arms: out,
           nativeRms: Math.sqrt(expected.reduce((t,v)=>t+v*v,0)/expected.length).toFixed(4) };
}
