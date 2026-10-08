// ESMFold2's weights for the native CUDA port: the folding bundle and the ESM-C tower, every tensor
// decoded to float32 under its own bundle name, in native/af3's model.idx/model.bin format.
//
//   node native/ef2/export_weights.mjs native/ef2/weights \
//        [--fold=model-esmfold2-trunk-f32] [--esmc=model-esmc-600m-f32]
//
// Entries:  t f/<name> ...   a folding-bundle tensor (blocks/3/pairTransition/transition1, ...)
//           t c/<name> ...   an ESM-C tensor (blocks/0/qkv/weights, lm/combine, ...)
//           m f/<name>#r, #k  its rank and k-th dimension
//           m meta/<key>     the manifests' numeric fields (trunk.* and languageModel.*)
// Bundle names are already the converted layouts the WebGPU port reads (AF3's pair-track shapes,
// inner-major linears), which is the point: those conversions are checked, and the CUDA side
// reads them rather than converting again.
import { readFileSync, writeFileSync, mkdirSync, openSync, writeSync, closeSync, renameSync } from "node:fs";
import { readTensor } from "../../shared/weights/dtype.js";

const args = process.argv.slice(2);
const out = args.find((a) => !a.startsWith("--"));
if (!out) { console.error("usage: export_weights.mjs <out dir> [--fold=<bundle dir>] [--esmc=<bundle dir>]"); process.exit(1); }
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const repo = new URL("../..", import.meta.url).pathname;
const bundles = [["f", option("fold", `${repo}model-esmfold2-conf-f32`)],
                 ["c", option("esmc", `${repo}model-esmc-600m-f32`)]];

mkdirSync(out, { recursive: true });
const fd = openSync(`${out}/model.bin`, "w");
const lines = [];
let offset = 0;
for (const [prefix, dir] of bundles) {
  const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, "utf8"));
  for (const block of [manifest.trunk, manifest.languageModel]) {
    for (const [key, value] of Object.entries(block ?? {})) {
      if (typeof value === "number") lines.push(`m meta/${key} ${value}`);
    }
  }
  const shards = new Map();
  for (const [name, record] of Object.entries(manifest.tensors)) {
    if (!shards.has(record.file)) {
      const bytes = readFileSync(`${dir}/${record.file}`);
      shards.set(record.file, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    }
    const values = readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
    const key = `${prefix}/${name}`;
    lines.push(`t ${key} ${offset} ${values.length}`, `m ${key}#r ${record.shape.length}`);
    record.shape.forEach((d, k) => lines.push(`m ${key}#${k} ${d}`));
    writeSync(fd, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
    offset += values.length;
  }
  console.log(`${dir}: ${Object.keys(manifest.tensors).length} tensors`);
}
closeSync(fd);
writeFileSync(`${out}/model.idx.tmp`, lines.join("\n") + "\n");
renameSync(`${out}/model.idx.tmp`, `${out}/model.idx`);
console.log(`${(offset * 4 / 2 ** 20).toFixed(0)} MiB -> ${out}`);
