// One AlphaFold 2 input for the native CUDA port: the page's own features (src/input/a3m-features.js,
// the function the page and tools/gpu/fold-af2.js fold with), one set a pass, in native/af3's
// model.idx/model.bin format.
//
//   node native/af2/export_input.mjs <out dir> --sequence=<SEQ> [--a3m=<path>] [--recycles=3]
//        [--max-msa=512] [--max-extra=1024] [--seed=0] (--bundle=<page bundle dir> | --weights=<export dir>)
//        [--search] [--template=<pdb>[:chain[+chain]],...]
//
// Entries:  i aatype, residue_index, asym_id, entity_id, sym_id   t seq_mask      (per residue)
//           t f<k>/msa_feat [N, L, 49], f<k>/msa_mask [N, L]                       (pass k)
//           i f<k>/extra_msa [E, L]   t f<k>/extra_has_deletion, extra_deletion_value, extra_msa_mask
//           m meta/tokens, meta/msa_rows, meta/extra_rows, meta/passes
// The tables the featuriser needs (atom37 maps) are read from the weights: the page's bundle (what
// native/af2/fold folds with) or export_weights.py's directory (DeepMind's float32, for the oracles).
import { readFileSync, writeFileSync, mkdirSync, openSync, readSync, writeSync, closeSync, renameSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { makeA3mFeatures } from "../../src/input/a3m-features.js";
import { chainResidues, identityMap, templateSlotAtom37 } from "../../src/af3/featurise/template-input.js";

const args = process.argv.slice(2);
const out = args[0];
if (!out || out.startsWith("--")) { console.error("usage: export_input.mjs <out dir> --sequence=<SEQ> [--a3m=...]"); process.exit(1); }
const option = (name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const here = new URL(".", import.meta.url).pathname;
const bundleDir = option("bundle", "");
const weightsDir = option("weights", "");
if ((bundleDir === "") === (weightsDir === "")) throw new Error("--bundle=<page bundle dir> or --weights=<export dir>");
let tables, isMultimer;
if (bundleDir !== "") {
  // the bundle's own residue geometry (float32 there), read with the page's reader
  const { readTensor } = await import("../../src/weights/dtype.js");
  const manifest = JSON.parse(readFileSync(`${bundleDir}/manifest.json`, "utf8"));
  const read = (name) => {
    const r = manifest.tensors[name];
    if (!r) throw new Error(`${bundleDir} has no ${name}`);
    const b = readFileSync(`${bundleDir}/${r.file}`);
    return Float32Array.from(readTensor(r, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), r.byteOffset ?? 0, true));
  };
  tables = { atom37ToAtom14: read("geometryAtom37ToAtom14"), atom37Mask: read("geometryAtom37Mask") };
  isMultimer = manifest.model.name.includes("multimer");
} else {
  // the two [21, 37] tables, out of the weights' model.bin
  const index = new Map();
  for (const line of readFileSync(`${weightsDir}/model.idx`, "utf8").split("\n")) {
    const [kind, name, a, b] = line.split(" ");
    if (kind === "t" || kind === "i") index.set(name, { kind, offset: Number(a), length: Number(b) });
  }
  // (each table's own bytes, not the file: reading the 370 MB of weights was a fifth of an export)
  const binFd = openSync(`${weightsDir}/model.bin`, "r");
  const table = (name) => {
    const e = index.get(name);
    if (!e) throw new Error(`${weightsDir} has no ${name}`);
    const bytes = Buffer.alloc(e.length * 4);
    readSync(binFd, bytes, 0, bytes.length, e.offset * 4);
    const view = e.kind === "i" ? new Int32Array(bytes.buffer, bytes.byteOffset, e.length)
      : new Float32Array(bytes.buffer, bytes.byteOffset, e.length);
    return Float32Array.from(view);
  };
  tables = { atom37ToAtom14: table("c/atom37_to_atom14"), atom37Mask: table("c/atom37_mask") };
  closeSync(binFd);
  isMultimer = readFileSync(`${weightsDir}/model.idx`, "utf8").includes("m meta/multimer 1");
}

const sequence = option("sequence", "").trim().toUpperCase();
const a3mPath = option("a3m", "");
if (sequence === "" && a3mPath === "") throw new Error("--sequence or --a3m names the input");
if (args.includes("--search") && sequence === "") throw new Error("--search needs --sequence");
// chains joined by ":" fold as a complex: the feature builder's chain-aware path (per-chain residue
// numbering, asym/entity/sym ids), what the page passes a multimer
const chains = sequence.split(":").filter(Boolean);
// --search: the alignment from the ColabFold MMseqs2 server through the page's own client and merge
// (src/input/mmseqs2-api.js): one chain's search, or a complex's - each distinct chain searched, the
// paired block for distinct ones, and the merge the WEIGHTS read ("multimer": dense within an entity,
// block-diagonal between; "monomer": block-diagonal throughout). It sends the sequences to
// api.colabfold.com, so it is asked for, never assumed
let a3m;
if (args.includes("--search")) {
  if (a3mPath !== "") throw new Error("--search and --a3m both name the alignment");
  const { generateMmseqs2Msa, generateMmseqs2ComplexMsa } = await import("../../src/input/mmseqs2-api.js");
  const multimer = isMultimer;
  const t0 = performance.now();
  a3m = chains.length === 1 ? (await generateMmseqs2Msa(chains[0], {})).a3m
    : (await generateMmseqs2ComplexMsa(chains, { model: multimer ? "multimer" : "monomer" })).a3m;
  console.log(`search: ${chains.length} chain(s) from api.colabfold.com in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
} else {
  a3m = a3mPath === "" ? `>query\n${chains.join("")}\n` : readFileSync(a3mPath, "utf8");
}
const featureOptions = {
  ...(chains.length > 1 ? { chainAware: true, chainLengths: chains.map((c) => c.length), chainSequences: chains } : {}),
  recycles: Number(option("recycles", "3")),
  maxMsaSequences: Number(option("max-msa", "512")),
  maxExtraSequences: Number(option("max-extra", "1024")),
  randomSeed: Number(option("seed", "0")),
};
// a deep alignment's recycles in parallel, one worker each (makeA3mFeatureRecycle: the same features
// as makeA3mFeatures, recycle by recycle - the nearest-centre search and the finishing are most of an
// export, and no recycle depends on another's); a shallow one is not worth a worker's start-up
const passes = featureOptions.recycles + 1;
const features = passes > 1 && a3m.length > (1 << 20)
  ? await Promise.all(Array.from({ length: passes }, (_, index) => new Promise((resolve, reject) => {
    const worker = new Worker(new URL("../../src/input/a3m-features-worker.mjs", import.meta.url),
      { workerData: { a3m, tables, options: featureOptions, index } });
    worker.once("message", resolve); worker.once("error", reject);
  })))
  : makeA3mFeatures(a3m, tables, featureOptions);
const first = features[0];
const L = first.aatype.length;

const entries = [];
const int = (name, v) => entries.push(["i", name, v instanceof Int32Array ? v : Int32Array.from(v)]);
const flt = (name, v) => entries.push(["t", name, v instanceof Float32Array ? v : Float32Array.from(v)]);
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
// --template=<pdb>[:chain[+chain...]],... : each a structure of the query's own sequence (identity mapping, as
// tools/gpu/fold-af2.js builds a self-template), as AF2's atom37 slots in the restype alphabet
const templateSpecs = option("template", "").split(",").filter(Boolean);
if (templateSpecs.length > 0) {
  const aat = new Int32Array(templateSpecs.length * L), pos = new Float32Array(templateSpecs.length * L * 37 * 3);
  const msk = new Float32Array(templateSpecs.length * L * 37);
  templateSpecs.forEach((spec, k) => {
    const [path, chain] = spec.split(":");
    // chains joined by '+' (a complex's template, A+D): one slot over the whole query, each chain's
    // residues following the last's, as the query's own chains follow each other
    const text = readFileSync(path, "utf8");
    const parts = (chain || "").split("+").map((c) => chainResidues(text, c || undefined));
    const structure = { ...parts[0], residues: parts.flatMap((part) => part.residues) };
    const slot = templateSlotAtom37({ structure, tokens: L, map: identityMap(structure) });
    aat.set(slot.aatype, k * L); pos.set(slot.atomPositions, k * L * 37 * 3); msk.set(slot.atomMask, k * L * 37);
    console.log(`template ${k}: ${path}${chain ? `:${chain}` : ""}, ${slot.covered} residues, ${slot.atoms} atoms`);
  });
  int("t/aatype", aat); flt("t/positions", pos); flt("t/mask", msk);
  entries.push(["m", "meta/templates", templateSpecs.length]);
}
if (weightsDir !== "") entries.push(["m", "meta/model", weightsDir.replace(/\/+$/, "").split("/").pop().replace(/^weights-/, "")]);
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
