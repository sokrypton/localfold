// Everything the CUDA AlphaFold 3 reads, in one file: the weights as this repository's
// loaders structure them, the featurised batch, and af3-any-model's own tensors to check
// each stage against.
//
//   python3 tools/serve.py 8791 &
//   node --js-float16array --max-old-space-size=24000 native/af3/export-model.mjs native/af3/data
//
// Writes <dir>/model.bin (raw little-endian 4-byte elements) and <dir>/model.idx, one line
// per entry:   t <name> <offset> <length>   float32
//              i <name> <offset> <length>   int32
//              m <name> <value>             a number (booleans as 0/1)
// Names are paths through the JS objects: trunk.pairformerBlocks.3.pairAttention1.qProjection.
//
// 🔴 WALKED, NOT LISTED. The JS weight objects carry the dialect's tensor set - an optional
// bias is null in one model and present in the next - and a hand-written list is the
// allow-list this repository keeps paying for. Every typed array under the object is written.
import { writeFileSync, mkdirSync, readFileSync, openSync, writeSync, closeSync } from "node:fs";

const repo = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const args = process.argv.slice(2);
const out = args.find((a) => !a.startsWith("--")) ?? `${repo}/native/af3/data`;
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const bundle = option("bundle", "http://127.0.0.1:8791/model-af3-full-f32/manifest.json");
const batchPath = option("batch", `${repo}/oracle-dumps/af3-batch-alphafold3-6mrr.json`);
const oracles = (option("oracles", option("sequence", "") === "" ? "trunk,denoise,confidence" : ""))
  .split(",").filter(Boolean);

const { openAf3Store, trunkWeights, trunkDepths, confidenceWeights } =
  await import(`${repo}/src/af3/weights/weights.js`);
const { diffusionWeights, atomReference, targetFeatureWeights } =
  await import(`${repo}/src/af3/weights/diffusion-weights.js`);
const { batchFromDump } = await import(`${repo}/tools/gpu/fold.js`);

mkdirSync(out, { recursive: true });
const entries = [];         // [kind, name, typedArray | number]
const seen = new WeakSet();
const add = (name, value) => {
  if (value === null || value === undefined) return;
  if (typeof value === "number") { entries.push(["m", name, value]); return; }
  if (typeof value === "boolean") { entries.push(["m", name, value ? 1 : 0]); return; }
  if (ArrayBuffer.isView(value)) {
    const isFloat = value instanceof Float32Array || value instanceof Float64Array
      || (typeof Float16Array !== "undefined" && value instanceof Float16Array);
    entries.push([isFloat ? "t" : "i", name, isFloat ? Float32Array.from(value) : Int32Array.from(value)]);
    return;
  }
  if (typeof value !== "object" || seen.has(value)) return;
  seen.add(value);
  if (Array.isArray(value)) {
    if (value.every((v) => typeof v === "number")) {
      entries.push(["t", name, Float32Array.from(value)]);
      return;
    }
    value.forEach((v, i) => add(`${name}.${i}`, v));
    return;
  }
  for (const key in value) {
    let v;
    try { v = value[key]; } catch { continue; }
    if (typeof v === "function" || typeof v === "string") continue;
    add(`${name}.${key}`, v);
  }
};

const store = await openAf3Store(bundle);
const depths = trunkDepths(store);
const trunk = await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks);
add("trunk", trunk);
add("diffusion", await diffusionWeights(store));
add("confidence", await confidenceWeights(store));
add("targetFeat", await targetFeatureWeights(store));
add("atomReference", await atomReference(store));

// The batch: featurised here from --sequence (chains joined by ":") and an optional --a3m,
// through the same function the page and fold.js use; otherwise read from an AF3 batch dump.
const sequence = option("sequence", "");
let batch;
if (sequence !== "") {
  const { af3BatchFromA3m } = await import(`${repo}/src/af3/featurise/batch.js`);
  const { featuriserDialect } = await import(`${repo}/src/af3/dialect.js`);
  const a3mPath = option("a3m", "");
  const alignment = a3mPath === "" ? null : readFileSync(a3mPath, "utf8");
  batch = af3BatchFromA3m(sequence, alignment, {
    maxSequences: Number(option("max-msa", "512")),
    seed: Number(option("seed", "20260831")),
    ...featuriserDialect(trunk.dialect),
  }).batch;
} else {
  batch = batchFromDump(JSON.parse(readFileSync(batchPath, "utf8")));
}
add("batch", batch);
// The oracle's own z_init/target_feat etc., by stage.
const flat = (record) => Float32Array.from(Array.isArray(record.data) ? record.data.flat(Infinity) : record.data);
for (const which of oracles) {
  const path = `${repo}/oracle-dumps/af3-oracle-${which}-alphafold3.json`;
  let oracle;
  try { oracle = JSON.parse(readFileSync(path, "utf8")); } catch { continue; }
  const walk = (prefix, object) => {
    for (const [key, value] of Object.entries(object ?? {})) {
      if (value && typeof value === "object" && "data" in value && "shape" in value) {
        add(`oracle.${which}.${prefix}${key}`, flat(value));
        add(`oracle.${which}.${prefix}${key}.rank`, value.shape.length);
        value.shape.forEach((d, i) => add(`oracle.${which}.${prefix}${key}.shape${i}`, d));
      } else if (value && typeof value === "object" && !Array.isArray(value)) {
        walk(`${prefix}${key}.`, value);
      } else if (typeof value === "number") {
        add(`oracle.${which}.${prefix}${key}`, value);
      }
    }
  };
  walk("", oracle);
}

let offset = 0;
const lines = [];
for (const [kind, name, value] of entries) {
  if (kind === "m") { lines.push(`m ${name} ${value}`); continue; }
  lines.push(`${kind} ${name} ${offset} ${value.length}`);
  offset += value.length;
}
const fd = openSync(`${out}/model.bin`, "w");
for (const [kind, , value] of entries) {
  if (kind !== "m") writeSync(fd, Buffer.from(value.buffer, value.byteOffset, value.byteLength));
}
closeSync(fd);
writeFileSync(`${out}/model.idx`, lines.join("\n") + "\n");
console.log(`${entries.length} entries, ${(offset * 4 / 1048576).toFixed(0)} MiB, tokens ${batch.tokens}`);
