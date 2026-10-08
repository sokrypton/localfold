// One AlphaFold 2 input for the native CUDA port: the page's own features (shared/input/a3m-features.js,
// the function the page and tools/gpu/fold-af2.js fold with), one set a pass, in cuda/af3's
// model.idx/model.bin format.
//
//   node cuda/af2/export_input.mjs <out dir> --sequence=<SEQ> [--a3m=<path>] [--recycles=3]
//        [--max-msa=512] [--max-extra=1024] [--seed=0] (--bundle=<page bundle dir> | --weights=<export dir>)
//        [--search] [--template=<structure>:<chain>[@<query chain>][+...],...] [--job=<AF3 job.json>]
//
// Entries:  i aatype, residue_index, asym_id, entity_id, sym_id   t seq_mask      (per residue)
//           t f<k>/msa_feat [N, L, 49], f<k>/msa_mask [N, L]                       (pass k)
//           i f<k>/extra_msa [E, L]   t f<k>/extra_has_deletion, extra_deletion_value, extra_msa_mask
//           m meta/tokens, meta/msa_rows, meta/extra_rows, meta/passes
// The tables the featuriser needs (atom37 maps) are read from the weights: the page's bundle (what
// cuda/af2/fold folds with) or export_weights.py's directory (DeepMind's float32, for the oracles).
import { readFileSync, writeFileSync, mkdirSync, openSync, readSync, writeSync, closeSync, renameSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { makeA3mFeatures, planA3mRecycles } from "../../shared/input/a3m-features.js";

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
  const { readTensor } = await import("../../shared/weights/dtype.js");
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

let sequence = option("sequence", "").trim().toUpperCase();
// --job=<AF3 job.json>: the page's own reader (web/job-json.js, web/entities.js), as cuda/af3's and
// cuda/ef2's exporters read one. AlphaFold 2 folds protein chains and nothing else, so anything else
// in the job is refused by name rather than dropped
if (option("job", "") !== "") {
  if (sequence !== "") throw new Error("--job and --sequence both name the input");
  const { jobFromJson } = await import("../../web/job-json.js");
  const { expandEntities } = await import("../../web/entities.js");
  const job = jobFromJson(readFileSync(option("job", ""), "utf8"));
  for (const note of job.notes) console.log(`job: ${note}`);
  const request = expandEntities(job.entities);
  const other = request.chainKinds.filter((kind) => kind !== "protein");
  if (other.length > 0) throw new Error(`AlphaFold 2 folds protein chains only; this job has ${[...new Set(other)].join(", ")}`);
  if (request.ligandCodes.length > 0) throw new Error("AlphaFold 2 folds protein chains only; this job has a ligand");
  if (request.modifications.length > 0) throw new Error("AlphaFold 2 folds the standard residues only; this job has a modified residue");
  if ((request.bonds ?? []).length > 0) throw new Error("AlphaFold 2 takes no declared bond; this job has one");
  sequence = request.sequence;
}
const a3mPath = option("a3m", "");
if (sequence === "" && a3mPath === "") throw new Error("--sequence or --a3m names the input");
if (args.includes("--search") && sequence === "") throw new Error("--search needs --sequence");
// chains joined by ":" fold as a complex: the feature builder's chain-aware path (per-chain residue
// numbering, asym/entity/sym ids), what the page passes a multimer
const chains = sequence.split(":").filter(Boolean);
// --search: the alignment from the ColabFold MMseqs2 server through the page's own client and merge
// (shared/input/mmseqs2-api.js): one chain's search, or a complex's - each distinct chain searched, the
// paired block for distinct ones, and the merge the WEIGHTS read ("multimer": dense within an entity,
// block-diagonal between; "monomer": block-diagonal throughout). It sends the sequences to
// api.colabfold.com, so it is asked for, never assumed
let a3m, searchedHits = null;       // (--search's template hits, by chain)
if (args.includes("--search")) {
  if (a3mPath !== "") throw new Error("--search and --a3m both name the alignment");
  const { generateMmseqs2Msa, generateMmseqs2ComplexMsa } = await import("../../shared/input/mmseqs2-api.js");
  const multimer = isMultimer;
  const t0 = performance.now();
  const searched = chains.length === 1 ? await generateMmseqs2Msa(chains[0], {})
    : await generateMmseqs2ComplexMsa(chains, { model: multimer ? "multimer" : "monomer" });
  a3m = searched.a3m;
  searchedHits = searched.templateHits ?? new Map();
  console.log(`search: ${chains.length} chain(s) from api.colabfold.com in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  mkdirSync(out, { recursive: true });
  writeFileSync(`${out}/search.a3m`, a3m);       // (what the search returned, for whoever shows the alignment)
} else {
  // --a3m=<one per chain, comma-separated> [--paired-a3m=<the same>]: an archive's per-chain alignments,
  // merged as the page merges them (web/app.js: mergeSearchedChains, the search path's own function)
  const paths = a3mPath.split(",").filter(Boolean);
  if (paths.length > 1 || option("paired-a3m", "") !== "") {
    if (paths.length !== chains.length) throw new Error(`${paths.length} alignments for ${chains.length} chains`);
    const pairedPaths = option("paired-a3m", "").split(",");
    const paired = chains.map((_, index) => (pairedPaths[index] ? readFileSync(pairedPaths[index], "utf8") : ""));
    const { mergeSearchedChains } = await import("../../shared/input/mmseqs2-api.js");
    a3m = mergeSearchedChains({
      sequences: chains,
      chainA3ms: paths.map((path) => readFileSync(path, "utf8")),
      // (no paired block at all is no map: the merge reads a map's every chain)
      pairedA3ms: paired.some((text) => text.trim() !== "")
        ? new Map(chains.map((chain, index) => [chain, paired[index]])) : undefined,
      model: isMultimer ? "multimer" : "monomer",
    }).a3m;
  } else {
    a3m = a3mPath === "" ? `>query\n${chains.join("")}\n` : readFileSync(a3mPath, "utf8");
  }
}
const featureOptions = {
  ...(chains.length > 1 ? { chainAware: true, chainLengths: chains.map((c) => c.length), chainSequences: chains } : {}),
  recycles: Number(option("recycles", "3")),
  maxMsaSequences: Number(option("max-msa", "512")),
  maxExtraSequences: Number(option("max-extra", "1024")),
  randomSeed: Number(option("seed", "0")),
};
// a deep alignment planned ONCE here - the parse, the encoding, the profile and every recycle's masking - and
// each recycle's nearest-centre search and finishing in a worker of its own (no recycle depends on another's).
// Each worker used to re-plan from the A3M text: the parse is the largest single cost of an export (185 of a
// recycle's ~440 ms for 5CAJ's 7907 rows), four times over on a Colab T4's two CPUs. Byte-identical: the same
// functions, in the same order. A shallow alignment is not worth a worker's start-up
const passes = featureOptions.recycles + 1;
const deep = passes > 1 && a3m.length > (1 << 20);
// each recycle searched and finished in a worker of its own
const inWorkers = (plans, context) => Promise.all(plans.map((plan) => new Promise((resolve, reject) => {
  const worker = new Worker(new URL("../../shared/input/a3m-features-worker.mjs", import.meta.url),
    { workerData: { plan, context } });
  worker.once("message", resolve); worker.once("error", reject);
})));
const features = deep ? await (async () => {
  const { plans, context } = planA3mRecycles(a3m, tables, featureOptions);
  return inWorkers(plans, context);
})() : makeA3mFeatures(a3m, tables, featureOptions);
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
// --template=<structure>:<chain>[@<query chain>][+...],... : one slot each, built as the page builds AF2's
// (web/app.js): every part aligned to its chain by the page's buildTemplate in the atom37 layout - PDB or
// mmCIF - the multimer's at its chain's residue offset, parts merged by mergeAtom37Templates. A part with
// no "@" takes the next query chain, so `1brs.pdb:A+D` is A onto chain 0 and D onto chain 1.
const templateSpecs = option("template", "").split(",").filter(Boolean);
// --template-search-chains=<chains>: the page's "from the MSA search" template - each listed chain's BEST
// hit from the same search, aligned and placed as any other part, all in the one slot AF2 takes
const searchChains = option("template-search-chains", "").split(",").filter(Boolean).map(Number);
if (searchChains.length > 0 && searchedHits === null) throw new Error("--template-search-chains needs --search: the hits come from that search");
if (templateSpecs.length > 1) {
  throw new Error("AlphaFold 2 takes one template slot: join its parts with '+'");
}
if (templateSpecs.length > 0 || searchChains.length > 0) {
  const { buildTemplate, mergeAtom37Templates } = await import("../../web/template-source.js");
  const chains = sequence.split(":").filter(Boolean);
  const offsets = chains.map((_, at) => chains.slice(0, at).reduce((n, c) => n + c.length, 0));
  const searchParts = [];
  if (searchChains.length > 0) {
    const { fetchMmseqs2Templates } = await import("../../shared/input/mmseqs2-api.js");
    for (const at of searchChains) {
      const best = (searchedHits.get(at) ?? [])[0];
      if (best === undefined) throw new Error(`the search found no template for chain ${at + 1}`);
      const text = (await fetchMmseqs2Templates([best.target])).get(best.id);
      if (text === undefined) throw new Error(`no structure came back for ${best.target}`);
      searchParts.push({ text, chain: best.chain, at, label: best.target });
    }
  }
  const specs = templateSpecs.length > 0 ? templateSpecs : [""];
  const aat = new Int32Array(L), pos = new Float32Array(L * 37 * 3);
  const msk = new Float32Array(L * 37);
  specs.forEach((spec, k) => {
    const [path0, chains0] = spec.split(":");
    const parts = spec === "" ? [] : spec.includes("@") ? spec.split("+").map((part) => {
      const [where, at] = part.split("@"); const cut = where.lastIndexOf(":");
      return { path: where.slice(0, cut), chain: where.slice(cut + 1) || undefined, at: Number(at) };
    }) : (chains0 ?? "").split("+").map((c, at) => ({ path: path0, chain: c || undefined, at }));
    const all = [...parts.map((p) => ({ ...p, text: readFileSync(p.path, "utf8") })), ...searchParts];
    const taken = new Set();
    const built = all.map(({ text, chain, at }) => {
      if (!isMultimer && at !== 0) throw new Error("AlphaFold 2's monomer takes a template on its one chain only");
      if (at >= chains.length) throw new Error(`template ${k}: no query chain ${at}`);
      if (taken.has(at)) throw new Error(`AlphaFold 2 takes one template a chain; chain ${at + 1} has two`);
      taken.add(at);
      return buildTemplate({ text, chain, query: isMultimer ? chains[at] : chains.join(""),
                             offset: isMultimer ? offsets[at] : 0, tokens: L, minConfidence: 0, layout: "atom37" });
    });
    const slot = (built.length === 1 ? built[0] : mergeAtom37Templates(built, L)).slot;
    aat.set(slot.aatype, k * L); pos.set(slot.atomPositions, k * L * 37 * 3); msk.set(slot.atomMask, k * L * 37);
    console.log(`template: ${[spec, ...searchParts.map((p) => `search hit ${p.label}`)].filter(Boolean).join(" + ")},`
      + ` ${slot.covered} residues, ${slot.atoms} atoms`);
  });
  int("t/aatype", aat); flt("t/positions", pos); flt("t/mask", msk);
  entries.push(["m", "meta/templates", 1]);
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
