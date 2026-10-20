// A tools/oracle/dump_bioemu2.py dump as an oracle directory the native port loads beside its input
// (`localfold-bioemu2 <input> --oracle=<dir>`): every stage as `o/<stage>` in cuda/af3's model.idx/model.bin
// format, each denoiser level's noise level as `m o/<stage>.sigma`.
//
//   node cuda/bioemu2/oracle_input.mjs <dump.json> <out dir>
import { writeFileSync, mkdirSync, openSync, writeSync, closeSync, renameSync } from "node:fs";
import { readBioemu2Dump } from "../../tools/bioemu2-dump.js";

const [dumpPath, out] = process.argv.slice(2);
if (!dumpPath || !out) { console.error("usage: oracle_input.mjs <dump.json> <out dir>"); process.exit(1); }
const { stages } = readBioemu2Dump(dumpPath);
mkdirSync(out, { recursive: true });
const fd = openSync(`${out}/model.bin`, "w");
const lines = [];
let offset = 0;
for (const [key, value] of Object.entries(stages)) {
  if (key.startsWith("feats.") || key.startsWith("smiles.") || key === "atom_keys") continue;
  if (typeof value === "number") { lines.push(`m o/${key} ${value}`); continue; }
  if (value?.length === undefined) continue;
  const data = value instanceof Float32Array ? value : Float32Array.from(value);
  lines.push(`t o/${key} ${offset} ${data.length}`);
  writeSync(fd, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  offset += data.length;
}
closeSync(fd);
writeFileSync(`${out}/model.idx.tmp`, lines.join("\n") + "\n");
renameSync(`${out}/model.idx.tmp`, `${out}/model.idx`);
console.log(`${lines.length} oracle entries -> ${out}`);
