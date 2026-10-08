// Export one AF3 pairformer block, a random input and the CPU reference's output.
//   node export.mjs <out-dir> [tokens] [manifest-url]
import { writeFileSync, mkdirSync } from "node:fs";
const repo = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const { openAf3Store, trunkWeights } = await import(`${repo}/shared/af3/weights/weights.js`);
const { pairformerBlock } = await import(`${repo}/cpu/af3/trunk/pairformer.js`);

const out = process.argv[2];
const n = Number(process.argv[3] ?? 64);
const url = process.argv[4] ?? "http://127.0.0.1:8791/model-af3-int5/manifest.json";
mkdirSync(out, { recursive: true });
const store = await openAf3Store(url);
const trunk = await trunkWeights(store, 48, 4, { allowPrefix: true });
const block = trunk.pairformerBlocks[0];
const dialect = trunk.dialect;

const tensors = [];
const meta = { pairChannels: block.pairChannels, singleChannels: block.singleChannels,
  gridHeads: block.pairAttention1.heads, gridDim: block.pairAttention1.dimension,
  singleHeads: block.singleAttention.heads, singleDim: block.singleAttention.dimension,
  swapTransposedBias: dialect.swapTransposedBias === true,
  divideByLength: dialect.triangleMulDivideByLength === true, tokens: n };
const add = (name, value) => {
  if (value == null) return;
  tensors.push([name, Float32Array.from(value)]);
};
for (const dir of ["Outgoing", "Incoming"]) {
  const t = block[`triangleMultiplication${dir}`];
  for (const k of ["leftNormInputScale", "leftNormInputOffset", "projection", "gate",
                   "centerNormScale", "centerNormOffset", "outputProjection", "gatingLinear"]) {
    add(`tri${dir}.${k}`, t[k]);
  }
}
for (const g of ["pairAttention1", "pairAttention2"]) {
  const a = block[g];
  for (const k of ["actNormScale", "actNormOffset", "pairBiasProjection", "qProjection",
                   "kProjection", "vProjection", "gatingQuery", "gatingQueryBias",
                   "outputProjection", "outputProjectionBias"]) add(`${g}.${k}`, a[k]);
}
for (const k of ["inputLayerNormScale", "inputLayerNormOffset", "transition1", "transition2"]) {
  add(`pairTransition.${k}`, block.pairTransition[k]);
  add(`singleTransition.${k}`, block.singleTransition[k]);
}
add("singlePairLogitsNormScale", block.singlePairLogitsNormScale);
add("singlePairLogitsNormOffset", block.singlePairLogitsNormOffset);
add("singlePairLogitsProjection", block.singlePairLogitsProjection);
const s = block.singleAttention;
for (const k of ["layerNormScale", "layerNormOffset", "qProjection", "qBias", "kProjection",
                 "vProjection", "gatingQuery", "outputProjection"]) add(`singleAttention.${k}`, s[k]);

// A seeded input. Masks: the last few tokens padded, so the masking is exercised.
let seed = 12345;
const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const C = block.pairChannels, S = block.singleChannels;
// --real: AF3's own pairformer input for 6MRR, out of the trunk oracle dump.
const real = process.argv.includes("--real");
let pair, single, seqMask;
if (real) {
  const { readFileSync } = await import("node:fs");
  const dump = JSON.parse(readFileSync(`${repo}/oracle-dumps/af3-oracle-trunk-alphafold3.json`, "utf8"));
  const z = dump.stages["tap.z_after_msa"], s0 = dump.stages["tap.trunk_in_single"];
  if (z.shape[0] !== n) throw new Error(`the dump is ${z.shape[0]} tokens; pass ${z.shape[0]}`);
  pair = Float32Array.from(z.data.flat(Infinity));
  single = Float32Array.from(s0.data.flat(Infinity));
  seqMask = new Float32Array(n).fill(1);
  meta.input = "af3 6mrr z_after_msa";
} else {
  pair = Float32Array.from({ length: n * n * C }, normal);
  single = Float32Array.from({ length: n * S }, normal);
  seqMask = Float32Array.from({ length: n }, (_, i) => (i < n - 3 ? 1 : 0));
}
const pairMask = new Float32Array(n * n);
for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) pairMask[i * n + j] = seqMask[i] * seqMask[j];
tensors.push(["input.pair", pair], ["input.single", single],
             ["input.seqMask", seqMask], ["input.pairMask", pairMask]);
const t0 = Date.now();
const result = pairformerBlock({ pair, single, pairMask, seqMask, tokens: n }, block, dialect);
meta.referenceSeconds = (Date.now() - t0) / 1000;
tensors.push(["expected.pair", result.pair], ["expected.single", result.single]);

let offset = 0;
const index = {};
for (const [name, value] of tensors) { index[name] = { offset, length: value.length }; offset += value.length; }
const all = new Float32Array(offset);
for (const [name, value] of tensors) all.set(value, index[name].offset);
writeFileSync(`${out}/block.bin`, Buffer.from(all.buffer));
writeFileSync(`${out}/block.json`, JSON.stringify({ meta, index }, null, 1));
console.log(JSON.stringify(meta), Object.keys(index).length, "tensors", (offset * 4 / 1048576).toFixed(1), "MiB");
