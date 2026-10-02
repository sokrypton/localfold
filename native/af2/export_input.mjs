// One AlphaFold 2 input for the native CUDA port: the page's own features (src/input/a3m-features.js,
// the function the page and tools/gpu/fold-af2.js fold with), one set a pass, in native/af3's
// model.idx/model.bin format.
//
//   node native/af2/export_input.mjs <out dir> --sequence=<SEQ> [--a3m=<path>] [--recycles=3]
//        [--max-msa=512] [--max-extra=1024] [--seed=0] [--weights=native/af2/weights-model_1_ptm]
//
// Entries:  i aatype, residue_index, asym_id, entity_id, sym_id   t seq_mask      (per residue)
//           t f<k>/msa_feat [N, L, 49], f<k>/msa_mask [N, L]                       (pass k)
//           i f<k>/extra_msa [E, L]   t f<k>/extra_has_deletion, extra_deletion_value, extra_msa_mask
//           m meta/tokens, meta/msa_rows, meta/extra_rows, meta/passes
// The tables the featuriser needs (atom37 maps) are read from the exported weights.
import { readFileSync, writeFileSync, mkdirSync, openSync, writeSync, closeSync, renameSync } from "node:fs";
import { makeA3mFeatures } from "../../src/input/a3m-features.js";

const args = process.argv.slice(2);
const out = args[0];
if (!out || out.startsWith("--")) { console.error("usage: export_input.mjs <out dir> --sequence=<SEQ> [--a3m=...]"); process.exit(1); }
const option = (name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const here = new URL(".", import.meta.url).pathname;
const weightsDir = option("weights", `${here}weights-model_1_ptm`);

// the two [21, 37] tables, out of the weights' model.bin
const index = new Map();
for (const line of readFileSync(`${weightsDir}/model.idx`, "utf8").split("\n")) {
  const [kind, name, a, b] = line.split(" ");
  if (kind === "t" || kind === "i") index.set(name, { kind, offset: Number(a), length: Number(b) });
}
const bin = readFileSync(`${weightsDir}/model.bin`);
const table = (name) => {
  const e = index.get(name);
  if (!e) throw new Error(`${weightsDir} has no ${name}`);
  const view = e.kind === "i" ? new Int32Array(bin.buffer, bin.byteOffset + e.offset * 4, e.length)
    : new Float32Array(bin.buffer, bin.byteOffset + e.offset * 4, e.length);
  return Float32Array.from(view);
};
const tables = { atom37ToAtom14: table("c/atom37_to_atom14"), atom37Mask: table("c/atom37_mask") };

const sequence = option("sequence", "").trim().toUpperCase();
const a3mPath = option("a3m", "");
if (sequence === "" && a3mPath === "") throw new Error("--sequence or --a3m names the input");
const a3m = a3mPath === "" ? `>query\n${sequence}\n` : readFileSync(a3mPath, "utf8");
const features = makeA3mFeatures(a3m, tables, {
  recycles: Number(option("recycles", "3")),
  maxMsaSequences: Number(option("max-msa", "512")),
  maxExtraSequences: Number(option("max-extra", "1024")),
  randomSeed: Number(option("seed", "0")),
});
const first = features[0];
const L = first.aatype.length;

const entries = [];
const int = (name, v) => entries.push(["i", name, Int32Array.from(v)]);
const flt = (name, v) => entries.push(["t", name, Float32Array.from(v)]);
int("aatype", first.aatype);
int("residue_index", first.residueIndex);
flt("seq_mask", first.seqMask);
int("asym_id", first.asymId ?? new Int32Array(L));
int("entity_id", first.entityId ?? new Int32Array(L));
int("sym_id", first.symId ?? new Int32Array(L));
features.forEach((f, k) => {
  if (f.msaFeatures.length !== f.msaSequences * L * 49) throw new Error("msa_feat is not [N, L, 49]");
  flt(`f${k}/msa_feat`, f.msaFeatures);
  flt(`f${k}/msa_mask`, f.msaMask);
  int(`f${k}/extra_msa`, f.extraMsa);
  flt(`f${k}/extra_has_deletion`, f.extraHasDeletion);
  flt(`f${k}/extra_deletion_value`, f.extraDeletionValue);
  flt(`f${k}/extra_msa_mask`, f.extraMsaMask);
});
entries.push(["m", "meta/tokens", L]);
entries.push(["m", "meta/msa_rows", first.msaSequences]);
entries.push(["m", "meta/extra_rows", first.extraSequences]);
entries.push(["m", "meta/passes", features.length]);

mkdirSync(out, { recursive: true });
let offset = 0;
const lines = [];
const fd = openSync(`${out}/model.bin`, "w");
for (const [kind, name, value] of entries) {
  if (kind === "m") { lines.push(`m ${name} ${value}`); continue; }
  lines.push(`${kind} ${name} ${offset} ${value.length}`);
  writeSync(fd, Buffer.from(value.buffer, value.byteOffset, value.byteLength));
  offset += value.length;
}
closeSync(fd);
writeFileSync(`${out}/model.idx.tmp`, lines.join("\n") + "\n");
renameSync(`${out}/model.idx.tmp`, `${out}/model.idx`);
console.log(`${L} residues, ${first.msaSequences} MSA rows, ${first.extraSequences} extra, ${features.length} passes -> ${out}`);
