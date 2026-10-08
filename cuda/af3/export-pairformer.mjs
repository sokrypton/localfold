// All of AF3's pairformer blocks, and AF3's OWN pairformer input and output for 6MRR.
//   node --js-float16array export48.mjs <out-dir> [manifest-url]
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
const repo = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const { openAf3Store, trunkWeights } = await import(`${repo}/shared/af3/weights/weights.js`);
const out = process.argv[2];
const url = process.argv[3] ?? "http://127.0.0.1:8791/model-af3-full-f32/manifest.json";
mkdirSync(out, { recursive: true });
const store = await openAf3Store(url);
const trunk = await trunkWeights(store, 48, 4);
const blocks = trunk.pairformerBlocks;
const dump = JSON.parse(readFileSync(`${repo}/oracle-dumps/af3-oracle-trunk-alphafold3.json`, "utf8"));
const flat = (name) => Float32Array.from(dump.stages[name].data.flat(Infinity));
const n = dump.stages["tap.trunk_in_pair"].shape[0];
const b0 = blocks[0];
const meta = { pairChannels: b0.pairChannels, singleChannels: b0.singleChannels,
  gridHeads: b0.pairAttention1.heads, gridDim: b0.pairAttention1.dimension,
  singleHeads: b0.singleAttention.heads, singleDim: b0.singleAttention.dimension,
  swapTransposedBias: trunk.dialect.swapTransposedBias === true,
  divideByLength: trunk.dialect.triangleMulDivideByLength === true, tokens: n, blocks: blocks.length };
const tensors = [];
const add = (name, value) => { if (value != null) tensors.push([name, Float32Array.from(value)]); };
blocks.forEach((block, k) => {
  const p = `b${k}.`;
  for (const dir of ["Outgoing", "Incoming"]) {
    const t = block[`triangleMultiplication${dir}`];
    for (const key of ["leftNormInputScale", "leftNormInputOffset", "projection", "gate",
                       "centerNormScale", "centerNormOffset", "outputProjection", "gatingLinear"])
      add(`${p}tri${dir}.${key}`, t[key]);
  }
  for (const g of ["pairAttention1", "pairAttention2"]) {
    const a = block[g];
    for (const key of ["actNormScale", "actNormOffset", "pairBiasProjection", "qProjection",
                       "kProjection", "vProjection", "gatingQuery", "gatingQueryBias",
                       "outputProjection", "outputProjectionBias"]) add(`${p}${g}.${key}`, a[key]);
  }
  for (const key of ["inputLayerNormScale", "inputLayerNormOffset", "transition1", "transition2"]) {
    add(`${p}pairTransition.${key}`, block.pairTransition[key]);
    add(`${p}singleTransition.${key}`, block.singleTransition[key]);
  }
  add(`${p}singlePairLogitsNormScale`, block.singlePairLogitsNormScale);
  add(`${p}singlePairLogitsNormOffset`, block.singlePairLogitsNormOffset);
  add(`${p}singlePairLogitsProjection`, block.singlePairLogitsProjection);
  const s = block.singleAttention;
  for (const key of ["layerNormScale", "layerNormOffset", "qProjection", "qBias", "kProjection",
                     "vProjection", "gatingQuery", "outputProjection"]) add(`${p}singleAttention.${key}`, s[key]);
});
const seqMask = new Float32Array(n).fill(1);
const pairMask = new Float32Array(n * n).fill(1);
tensors.push(["input.pair", flat("tap.trunk_in_pair")], ["input.single", flat("tap.trunk_in_single")],
             ["input.seqMask", seqMask], ["input.pairMask", pairMask],
             ["expected.pair", flat("tap.trunk_out_pair")], ["expected.single", flat("single")],
             ["check.zAfterMsa", flat("tap.z_after_msa")]);
let offset = 0; const index = {};
for (const [name, value] of tensors) { index[name] = { offset, length: value.length }; offset += value.length; }
const all = new Float32Array(offset);
for (const [name, value] of tensors) all.set(value, index[name].offset);
writeFileSync(`${out}/block.bin`, Buffer.from(all.buffer));
writeFileSync(`${out}/block.json`, JSON.stringify({ meta, index }));
console.log(JSON.stringify(meta), tensors.length, "tensors", (offset * 4 / 1048576).toFixed(0), "MiB");
