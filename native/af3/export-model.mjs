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
import { writeFileSync, mkdirSync, readFileSync, openSync, writeSync, closeSync, renameSync } from "node:fs";

const repo = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
// --serve=DIR: stay up with every module loaded (the loading was ~180 ms of a 220 ms export) and
// export each request dropped in DIR - <id>.req, a JSON list of this script's arguments - by
// importing this file again with those arguments (its dependencies stay cached), writing <id>.ok
// or <id>.err. A request of ["quit"] stops it. native/af3/fold --serve runs one beside af3's.
const serveArg = process.argv.slice(2).find((a) => a.startsWith("--serve="));
if (serveArg !== undefined && globalThis.EXPORT_SERVING === undefined) {
  globalThis.EXPORT_SERVING = true;
  const { readdirSync, unlinkSync } = await import("node:fs");
  const dir = serveArg.slice(8);
  const argv0 = process.argv.slice(0, 2);
  console.log(`export: serving ${dir}`);
  for (let n = 0; ; ) {
    const reqs = readdirSync(dir).filter((f) => f.endsWith(".req")).sort();
    if (reqs.length === 0) { await new Promise((r) => setTimeout(r, 2)); continue; }
    const id = reqs[0].slice(0, -4);
    const request = JSON.parse(readFileSync(`${dir}/${id}.req`, "utf8"));
    unlinkSync(`${dir}/${id}.req`);
    if (request[0] === "quit") process.exit(0);
    process.argv = [...argv0, ...request];
    try {
      await import(`${import.meta.url}?request=${n++}`);
      writeFileSync(`${dir}/${id}.ok`, "");
    } catch (error) {
      writeFileSync(`${dir}/${id}.err`, `${error?.stack ?? error}\n`);
    }
  }
}
const args = process.argv.slice(2);
const out = args.find((a) => !a.startsWith("--")) ?? `${repo}/native/af3/data`;
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
// the bundle: a URL, or a path on disk (the default - no server needed: fetch reads file:// here)
const bundleArg = option("bundle", `${repo}/model-af3-full-f32/manifest.json`);
const bundle = /^https?:/.test(bundleArg) ? bundleArg : new URL(`file://${bundleArg.startsWith("/") ? "" : process.cwd() + "/"}${bundleArg}`).href;
if (globalThis.EXPORT_FETCH_SHIM === undefined) {    // once per process (the server re-imports this file)
  globalThis.EXPORT_FETCH_SHIM = true;
  const networkFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const href = typeof url === "string" ? url : url.url ?? String(url);
    if (!href.startsWith("file:")) return networkFetch(url, init);
    const bytes = readFileSync(new URL(href));
    return new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length) } });
  };
}
// --oracle-model=<name>: that model's own reference batch and oracles (oracle-dumps/af3-batch-<name>-6mrr.json,
// af3-oracle-{trunk,denoise,confidence}-<name>.json) - with its bundle (--bundle)
const oracleModel = option("oracle-model", "alphafold3");
const batchPath = option("batch", `${repo}/oracle-dumps/af3-batch-${oracleModel}-6mrr.json`);
const oracles = (option("oracles", option("sequence", "") === "" && option("job", "") === "" ? "trunk,denoise,stages,structural,realdenoise,confidence" : ""))
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

// --weights-only writes the weights alone (once, for every input: `af3 <batch dir> --weights=<dir>`);
// --no-weights writes the input alone (the dialect then comes from the bundle's manifest)
const weightsOnly = args.includes("--weights-only"), noWeights = args.includes("--no-weights");
let dialect;
if (noWeights) {
  const { dialectFor } = await import(`${repo}/src/af3/dialect.js`);
  const manifest = JSON.parse(Buffer.from(await (await fetch(bundle)).arrayBuffer()).toString("utf8"));
  dialect = dialectFor(manifest?.model?.name);
} else {
  const store = await openAf3Store(bundle);
  const depths = trunkDepths(store);
  const trunk = await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks);
  dialect = trunk.dialect;
  add("trunk", trunk);
  add("diffusion", await diffusionWeights(store));
  if (trunk.dialect.structuralTokens) {
    // OpenDDE: the structural-token expander, its refiner and its own confidence head
    const { structuralExpanderWeights, structuralRefinerWeights, openddeConfidenceWeights } =
      await import(`${repo}/src/af3/weights/weights.js`);
    add("expander", await structuralExpanderWeights(store));
    add("refiner", { blocks: await structuralRefinerWeights(store) });
    add("ddeConfidence", await openddeConfidenceWeights(store));
  } else {
    add("confidence", await confidenceWeights(store));
  }
  add("targetFeat", await targetFeatureWeights(store));
  add("atomReference", await atomReference(store));
}

let batch = null;
if (!weightsOnly) {
// The batch: featurised here from --sequence (chains joined by ":") and an optional --a3m,
// through the same function the page and fold.js use; or from an AlphaFold 3 job JSON (--job),
// read by the page's own reader (web/job-json.js, web/entities.js); otherwise read from an AF3
// batch dump.
let sequence = option("sequence", "");
let jobRequest = null;
let searchedHits = null, searchedProteinAt = null;   // --search's template hits, by protein chain
const jobTemplates = [];                // [slot k] -> the job's k-th template of each chain
let jobUserCcd = null;                  // the job's own component definitions (userCCD), mmCIF
if (option("job", "") !== "") {
  if (sequence !== "") throw new Error("--job and --sequence both name the input");
  const { jobFromJson } = await import(`${repo}/web/job-json.js`);
  const { expandEntities } = await import(`${repo}/web/entities.js`);
  // AlphaFold 3's data pipeline writes its alignments INTO the job (unpairedMsa / pairedMsa per
  // chain): the page's reader returns them per chain copy (job.alignments), merged below through
  // the page's own mergeJobAlignments.
  const raw = JSON.parse(readFileSync(option("job", ""), "utf8"));
  const jobs = Array.isArray(raw) ? raw : [raw];
  // The TEMPLATES, every one of them: AF3's data pipeline writes up to twenty a chain into the job
  // and folds the first four, chain i's k-th in slot k, where the page's reader takes one a chain -
  // so they are lifted out here, per chain copy, and built below (jobTemplates)
  let polymerCopy = 0;
  for (const entry of jobs[0]?.sequences ?? []) {
    const [kind, body] = Object.entries(entry)[0] ?? [];
    if (!["protein", "rna", "dna"].includes(kind) || body === undefined) continue;
    const copies = Array.isArray(body.id) ? body.id.length : 1;
    for (let c = 0; c < copies; c += 1) {
      for (const [k, t] of (Array.isArray(body.templates) ? body.templates : []).slice(0, 4).entries()) {
        if (typeof t.mmcif !== "string") throw new Error(`template ${k} of chain ${polymerCopy}: give the mmCIF inline`);
        const q = t.queryIndices, ti = t.templateIndices;
        if ((q === undefined) !== (ti === undefined) || (q && (q.length !== ti.length))) {
          throw new Error(`template ${k} of chain ${polymerCopy}: queryIndices and templateIndices are two lists of one length`);
        }
        (jobTemplates[k] ??= []).push({ text: t.mmcif, chain: polymerCopy, label: `job template ${k}`,
                                        ...(q === undefined ? {} : { mapping: q.map((v, i) => [v, ti[i]]) }) });
      }
      polymerCopy += 1;
    }
    delete body.templates;
  }
  const job = jobFromJson(JSON.stringify(raw));
  // every modelSeed, where the page folds the first: af3 runs each (src/af3.cu, --seeds)
  const seeds = [jobs[0]?.modelSeeds ?? []].flat().map(Number);
  for (const s of seeds) if (!Number.isInteger(s) || s < 0) throw new Error(`modelSeeds: ${s} is not a seed`);
  for (const note of job.notes) {
    if (seeds.length > 1 && /seeds in the file; folding the first/.test(note)) console.log(`job: ${seeds.length} seeds, each folded`);
    else console.log(`job: ${note}`);
  }
  if (seeds.length > 1) {
    entries.push(["m", "job.seeds.count", seeds.length]);
    seeds.forEach((s, i) => entries.push(["m", `job.seeds.${i}`, s]));
  }
  jobRequest = expandEntities(job.entities);
  jobUserCcd = job.userCcd ?? null;
  sequence = jobRequest.sequence;
  if (job.alignments !== undefined) {
    const { mergeJobAlignments } = await import(`${repo}/src/input/chains.js`);
    const { alignment, msaColumnKinds } = mergeJobAlignments(job.alignments, jobRequest.chains, jobRequest.chainKinds);
    jobRequest.alignment = alignment;
    jobRequest.msaColumnKinds = msaColumnKinds;
    console.log(`job: inline alignments for ${job.alignments.unpaired.filter(Boolean).length} chains (unpaired),`
      + ` ${job.alignments.paired.filter(Boolean).length} (paired)`);
  }
  if (job.seed !== undefined) entries.push(["m", "job.seed", job.seed]);
  console.log(`job ${job.name ?? "(unnamed)"}: ${jobRequest.chains.length} chains (${jobRequest.chainKinds.join(", ")}),`
    + ` ${jobRequest.ligandCodes.length} ligands, ${jobRequest.modifications.length} modifications,`
    + ` ${(jobRequest.bonds ?? []).length} bonds${job.seed === undefined ? "" : `, seed ${job.seed}`}`);
}
if (sequence !== "") {
  const { af3BatchFromA3m } = await import(`${repo}/src/af3/featurise/batch.js`);
  const { featuriserDialect } = await import(`${repo}/src/af3/dialect.js`);
  // --a3m=<one path per chain, comma-separated> and --paired-a3m=<the same for the paired block>,
  // merged exactly as tools/gpu/fold.js and the page merge them; a single path is a monomer's
  const { mergeRowAlignedChainA3ms } = await import(`${repo}/src/input/chains.js`);
  const texts = (spec) => (spec === "" ? null : spec.split(",").map((path) => readFileSync(path.trim(), "utf8")));
  const merge = (list) => (list === null ? null : (list.length === 1 ? list[0] : mergeRowAlignedChainA3ms(list)));
  const unpaired = texts(option("a3m", "")), paired = texts(option("paired-a3m", ""));
  if (jobRequest?.alignment && (unpaired || paired)) throw new Error("the job carries its alignments; --a3m would replace them");
  let alignment = jobRequest?.alignment ?? (unpaired === null && paired === null ? null
    : (paired === null && unpaired.length === 1 ? unpaired[0] : { paired: merge(paired), unpaired: merge(unpaired) }));
  // --search: the protein chains' alignments from the ColabFold MMseqs2 server, through the page's
  // own client and merge (src/input/mmseqs2-api.js) - AF3's data pipeline step, as the page runs it;
  // it sends the sequences to api.colabfold.com, so it is asked for, never assumed
  if (args.includes("--search") || args.includes("--search-templates")) {
    if (alignment !== null) throw new Error("--search and an alignment both name the MSA");
    const { generateMmseqs2Msa, generateMmseqs2ComplexMsa } = await import(`${repo}/src/input/mmseqs2-api.js`);
    const allChains = sequence.split(":");
    const allKinds = jobRequest?.chainKinds ?? (option("kinds", "") === "" ? allChains.map(() => "protein") : option("kinds", "").split(","));
    // (the protein chains only, in order: without nucleic coverage the featuriser maps the alignment's
    // columns onto the protein residues and skips the nucleic ones, wherever they sit)
    const proteins = allChains.filter((_, i) => allKinds[i] === "protein");
    if (proteins.length === 0) throw new Error("--search: no protein chain to search for");
    const t0 = performance.now();
    if (proteins.length === 1) {
      const searched = await generateMmseqs2Msa(proteins[0], {});
      alignment = searched.a3m;
      searchedHits = searched.templateHits;
    } else {
      const searched = await generateMmseqs2ComplexMsa(proteins, { model: "af3" });
      alignment = searched.blocks;
      searchedHits = searched.templateHits;
    }
    searchedProteinAt = allKinds.flatMap((kind, i) => (kind === "protein" ? [i] : []));
    // (one chain's alignment as the search returned it, for whoever shows the alignment)
    if (typeof alignment === "string") writeFileSync(`${out}/search.a3m`, alignment);
    console.log(`search: ${proteins.length} protein chain(s) from api.colabfold.com in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  }
  // --ligands=GOL,ATP (CCD codes, fetched from the RCSB), --smiles=OCC(O)CO|..., --kinds=protein,dna
  // (one per ":"-chain), --modify=SEP@3[@chain] (the position as tools/gpu/probe-modified.js takes it, chain index from 0)
  const { ccdUrl, parseCcdComponent, ligandChain } = await import(`${repo}/src/af3/featurise/ccd-component.js`);
  const { nameSmilesLigands, smilesComponent } = await import(`${repo}/src/chem/component.js`);
  // a job's own userCCD first (each data_ block one component), then the RCSB
  const userComponents = new Map();
  if (jobUserCcd) {
    for (const block of jobUserCcd.split(/^(?=data_)/m).filter((b) => b.trim().startsWith("data_"))) {
      const component = parseCcdComponent(block);
      userComponents.set(component.code.toUpperCase(), component);
    }
    console.log(`job: userCCD defines ${[...userComponents.keys()].join(", ")}`);
  }
  const ccd = async (code) => {
    if (userComponents.has(code.toUpperCase())) return userComponents.get(code.toUpperCase());
    const response = await fetch(ccdUrl(code));
    if (!response.ok) throw new Error(`could not fetch ${code}: ${response.status}`);
    return parseCcdComponent(await response.text());
  };
  const ligands = [];
  const modifications = [];
  let kinds = option("kinds", "");
  if (jobRequest !== null) {           // the page's own resolution (web/af3-model.js)
    for (const entry of jobRequest.ligandCodes) {
      ligands.push(typeof entry === "string" ? await ccd(entry)
        : entry.codes ? ligandChain(await Promise.all(entry.codes.map(ccd)))     // a glycan: one chain
        : await smilesComponent(entry.smiles, { code: entry.code ?? "LIG" }));
    }
    for (const m of jobRequest.modifications) {
      modifications.push({ chain: m.chain, position: m.position, ...(await ccd(m.code)) });
    }
    kinds = jobRequest.chainKinds.join(",");
  }
  for (const code of option("ligands", "").split(",").filter(Boolean)) ligands.push(await ccd(code));
  const smiles = option("smiles", "").split("|").filter(Boolean);
  const smilesNames = nameSmilesLigands(smiles);
  for (let i = 0; i < smiles.length; i += 1) ligands.push(await smilesComponent(smiles[i], { code: smilesNames[i] }));
  for (const spec of option("modify", "").split(",").filter(Boolean)) {
    const [code, at, chain] = spec.split("@");
    modifications.push({ chain: Number(chain ?? 0), position: Number(at), ...(await ccd(code)) });
  }
  batch = af3BatchFromA3m(sequence, alignment, {
    maxSequences: Number(option("max-msa", "1024")),     // AF3's num_msa (evoformer.py), and af3's --msa cap
    ...(jobRequest?.msaColumnKinds === undefined ? {} : { msaColumnKinds: jobRequest.msaColumnKinds }),
    seed: Number(option("seed", "20260831")),
    ...featuriserDialect(dialect),
    ...(ligands.length === 0 ? {} : { ligands }),
    ...(modifications.length === 0 ? {} : { modifications }),
    ...(kinds === "" ? {} : { chainKinds: kinds.split(",") }),
    ...(jobRequest?.bonds === undefined ? {} : { bonds: jobRequest.bonds }),
  }).batch;
} else {
  batch = batchFromDump(JSON.parse(readFileSync(batchPath, "utf8")));
}
add("batch", batch);
// the distogram's contact bins per token pair (src/af3/featurise/contact-classes.js), as the page
// reads contact_probs off the distogram: the bin count is the bundle's
{
  const { af3ContactClasses, af3ContactBins } = await import(`${repo}/src/af3/featurise/contact-classes.js`);
  const { binEdges } = await import(`${repo}/src/af3/trunk/trunk-webgpu.js`);
  const manifest = JSON.parse(Buffer.from(await (await fetch(bundle)).arrayBuffer()).toString("utf8"));
  const shape = manifest?.tensors?.["diffuser/distogram_head/half_logits/weights"]?.shape;
  if (shape) add("batch.contactBins", af3ContactBins(af3ContactClasses(batch, batch.tokens), batch.tokens, binEdges(shape[1])));
}
// OpenDDE's second token space: after the trunk each standard residue becomes a backbone and a
// sidechain token, and the diffusion and its confidence head run on those (src/af3/fold.js)
if (dialect.structuralTokens) {
  const { structuralLayout, structuralBatch } = await import(`${repo}/src/af3/featurise/structural-tokens.js`);
  const { structuralPairFeatures } = await import(`${repo}/src/af3/structure/structural-expander-reference.js`);
  const layout = structuralLayout(batch);
  const sb = structuralBatch(batch, layout);
  const features = structuralPairFeatures(layout, batch.asymId);
  const keep = ["tokens", "dense", "subsets", "atomCount", "shape", "aatype", "residueIndex", "tokenIndex", "asymId",
    "entityId", "symId", "seqMask", "refPos", "refMask", "refElement", "refCharge", "refAtomNameChars", "refSpaceUid",
    "predDenseAtomMask", "bondMatrix", "residueOfToken", "tokenAtomsToQueries", "queriesToTokenAtoms", "queriesToKeys",
    "tokensToQueries", "tokensToKeys", "tokenAtomsToPseudoBeta", "features"];
  add("sbatch", Object.fromEntries(keep.map((k) => [k, sb[k]])));
  add("structural.parent", Int32Array.from(layout.parent));
  add("structural.role", Int32Array.from(layout.role));
  add("structural.residueAtomGather", Int32Array.from(layout.residueAtomGather));
  add("structural.residueRepToken", Int32Array.from(layout.residueRepToken));
  for (const k of ["sameParent", "twin", "prevBackbone", "nextBackbone", "rolePairType"]) {
    add(`structural.${k}`, Int32Array.from(features[k]));
  }
}
// --template=<pdb or cif>:<chain>[@<query chain index>], comma-separated, one slot each (at most
// four); parts joined by "+" share ONE slot, as AF3 puts each chain's k-th template in slot k -
// e.g. 1brs.pdb:A@0+1brs.pdb:D@1 - and that merged slot may speak across the chains it covers
// (--no-span-chains masks the cross-chain block). Each part is built by the page's own
// buildTemplate.
const templateSpecs = option("template", "").split(",").filter(Boolean);
const TEMPLATES = 4;                    // the padded slot count every family folds with
if (templateSpecs.length > TEMPLATES) throw new Error("at most four template slots");
const { templateGeometry, multichainMaskFor, coverageOf } =
  await import(`${repo}/src/af3/featurise/template-features.js`);
const slots = [];                       // {slot, mask}
if (templateSpecs.length > 0) {
  const { buildTemplate } = await import(`${repo}/web/template-source.js`);
  const { mergeTemplateSlots } = await import(`${repo}/src/af3/featurise/template-input.js`);
  const chains = sequence.split(":");
  // which token each chain's residue occupies (a modified residue or ligand shifts them)
  const tokenOfResidue = new Int32Array(batch.chainOfResidue.length).fill(-1);
  batch.residueOfToken.forEach((residue, token) => {
    if (residue >= 0 && tokenOfResidue[residue] === -1) tokenOfResidue[residue] = token;
  });
  const residuesOfChain = [];
  Array.from(batch.chainOfResidue).forEach((chain, residue) => (residuesOfChain[chain] ??= []).push(residue));
  templateSpecs.forEach((spec, k) => {
    const parts = spec.split("+").map((part) => {
      const [where, target = "0"] = part.split("@");
      const cut = where.lastIndexOf(":");
      const path = where.slice(0, cut), chain = where.slice(cut + 1) || undefined, index = Number(target);
      const built = buildTemplate({
        text: readFileSync(path, "utf8"), chain, query: chains[index], tokens: batch.tokens, minConfidence: 0,
        tokenOf: (residue) => tokenOfResidue[(residuesOfChain[index] ?? [])[residue] ?? -1] ?? -1,
      });
      console.log(`template ${k}: ${path} chain ${chain} -> query chain ${index},`
        + ` ${built.coverage.residues}/${built.coverage.of} residues`);
      return built.slot;
    });
    const slot = parts.length === 1 ? parts[0] : mergeTemplateSlots(parts);
    const spanChains = parts.length > 1 && !args.includes("--no-span-chains");
    slots.push({ slot, mask: multichainMaskFor(batch.asymId, batch.tokens,
                                               { coverage: coverageOf(slot, batch.tokens), spanChains }) });
  });
}
// Template slots built from structures this run was handed rather than given on the command line:
// a job's own templates and --search-templates' hits. Each part by the page's buildTemplate, every
// chain's k-th in slot k as AF3 puts them, the cross-chain block masked (AF3's template embedder pairs
// residues of one chain only).
const extraSlotParts = [];               // [slot k][part] = {text, chainId, chain, mapping}
// ...a job's own templates (they were ignored, then one a chain): every chain's k-th in slot k, up to
// four, each with its queryIndices / templateIndices mapping when the job gives one
if (jobTemplates.length > 0) {
  if (templateSpecs.length > 0) throw new Error("the job carries its templates; --template would replace them");
  for (const parts of jobTemplates) extraSlotParts.push(parts);
}
// (a job asking for a template SEARCH - the server dialect's useStructureTemplate - is refused: there is
// no search unless --search-templates asks for one)
if ((jobRequest?.templates ?? []).some((t) => t.kind !== "upload") && !args.includes("--search-templates")) {
  const search = jobRequest.templates.find((t) => t.kind !== "upload");
  throw new Error(`chain ${search.chain}: the job asks for a template search - run with --search-templates,`
    + " or give the structure with --template=<file>:<chain>@<query chain>");
}
// ...--search-templates: each protein chain's best four hits from the same MMseqs2 search, fetched
// from the server as the page fetches its one
if (args.includes("--search-templates")) {
  if (templateSpecs.length > 0 || extraSlotParts.length > 0) throw new Error("--search-templates and other templates both name the slots");
  const { fetchMmseqs2Templates } = await import(`${repo}/src/input/mmseqs2-api.js`);
  const perChain = [...(searchedHits ?? new Map())].map(([at, found]) => [searchedProteinAt[at] ?? at, found.slice(0, TEMPLATES)]);
  const structures = await fetchMmseqs2Templates(perChain.flatMap(([, found]) => found.map((hit) => hit.target)));
  for (let k = 0; k < TEMPLATES; k += 1) {
    const parts = [];
    for (const [chain, found] of perChain) {
      const hit = found[k];
      if (hit === undefined) continue;
      const text = structures.get(hit.id);
      if (text === undefined) throw new Error(`no structure came back for ${hit.target}`);
      parts.push({ text, chainId: hit.chain, chain, label: `search hit ${hit.target}` });
    }
    if (parts.length > 0) extraSlotParts.push(parts);
  }
  if (extraSlotParts.length === 0) console.log("search: no template hits");
}
// ...--template-search-chains=<fold chains>: the page's "from the MSA search" template (web/app.js): each
// listed chain's BEST hit from the same search, in a slot of its own; a chain the search found nothing
// for is an error, as it is on the page
const searchChains = option("template-search-chains", "").split(",").filter(Boolean).map(Number);
if (searchChains.length > 0) {
  if (searchedHits === null) throw new Error("--template-search-chains needs --search: the hits come from that search");
  const { fetchMmseqs2Templates } = await import(`${repo}/src/input/mmseqs2-api.js`);
  const hits = new Map([...searchedHits].map(([at, found]) => [searchedProteinAt[at] ?? at, found]));
  for (const chain of searchChains) {
    const best = (hits.get(chain) ?? [])[0];
    if (best === undefined) throw new Error(`the search found no template for chain ${chain + 1}`);
    const text = (await fetchMmseqs2Templates([best.target])).get(best.id);
    if (text === undefined) throw new Error(`no structure came back for ${best.target}`);
    extraSlotParts.push([{ text, chainId: best.chain, chain, label: `search hit ${best.target}` }]);
  }
}
if (extraSlotParts.length > 0) {
  const { buildTemplate } = await import(`${repo}/web/template-source.js`);
  const { mergeTemplateSlots } = await import(`${repo}/src/af3/featurise/template-input.js`);
  const chains = sequence.split(":");
  const tokenOfResidue = new Int32Array(batch.chainOfResidue.length).fill(-1);
  batch.residueOfToken.forEach((residue, token) => {
    if (residue >= 0 && tokenOfResidue[residue] === -1) tokenOfResidue[residue] = token;
  });
  const residuesOfChain = [];
  Array.from(batch.chainOfResidue).forEach((chain, residue) => (residuesOfChain[chain] ??= []).push(residue));
  extraSlotParts.forEach((partsOfSlot, k) => {
    const parts = partsOfSlot.map((t) => {
      const built = buildTemplate({
        text: t.text, chain: t.chainId, query: chains[t.chain], tokens: batch.tokens, minConfidence: 0,
        ...(t.mapping === undefined ? {} : { mapping: t.mapping }),
        tokenOf: (residue) => tokenOfResidue[(residuesOfChain[t.chain] ?? [])[residue] ?? -1] ?? -1,
      });
      console.log(`template ${k}: ${t.label} -> query chain ${t.chain}, ${built.coverage.residues}/${built.coverage.of} residues`
        + (t.mapping ? " (the job's mapping)" : ""));
      return built.slot;
    });
    const slot = parts.length === 1 ? parts[0] : mergeTemplateSlots(parts);
    slots.push({ slot, mask: multichainMaskFor(batch.asymId, batch.tokens,
                                               { coverage: coverageOf(slot, batch.tokens), spanChains: false }) });
  });
}
// rf3's chirality term reads the stereocentres - four dense atom slots and an ideal improper
// dihedral each - as the page's fold does (src/af3/fold.js)
if (dialect.chiralCentres === true) {
  const { chiralCentres } = await import(`${repo}/src/af3/featurise/template-features.js`);
  const chirals = chiralCentres(batch.aatype, batch.predDenseAtomMask, batch.tokens, batch.dense);
  add("chiral.centers", Int32Array.from(chirals.centers));
  add("chiral.angles", Float32Array.from(chirals.angles));
  add("chiral.count", chirals.count);
}
// The template embedder's PASSES, built as the page's trunk builds them (template-webgpu.js): each
// a repeat weight and either - the fused embedder (protenix2, boltz2, rf3) - its feature columns,
// or - the nine-projection one - an aatype and, for a real template, its geometry. Empty slots
// fold into one or two passes (an empty slot carries the GAP restype in one slot under protenix's
// featuriser and 0 elsewhere), rf3 averages every present template's features into one pass, and
// boltz2 weighs an empty slot zero.
{
  const fused = dialect.fusedTemplateLayout != null || dialect.boltz2TemplateFeatures === true
    || dialect.rosettafold3TemplateFeatures === true;
  const width = dialect.boltz2TemplateFeatures ? 109 : dialect.rosettafold3TemplateFeatures ? 66
    : dialect.fusedTemplateLayout ? dialect.fusedTemplateLayout.distogramBins + 1 + 2 * dialect.fusedTemplateLayout.restypes + 4 : 0;
  const { fusedTemplateFeatures } = fused ? await import(`${repo}/src/af3/trunk/template-webgpu.js`) : {};
  if (slots.length > TEMPLATES) throw new Error(`${slots.length} template slots; every family folds with at most ${TEMPLATES}`);
  const passes = [];
  if (dialect.templateFeatureMeanOnePass === true) {
    const present = slots.filter(({ slot }) => Array.prototype.some.call(slot.atomMask, (v) => v > 0));
    let features = fusedTemplateFeatures(undefined, batch.tokens, width, dialect, undefined, false);
    if (present.length > 0) {
      features = fusedTemplateFeatures(present[0].slot, batch.tokens, width, dialect, present[0].mask, false);
      for (const more of present.slice(1)) {
        const f = fusedTemplateFeatures(more.slot, batch.tokens, width, dialect, more.mask, false);
        for (let i = 0; i < features.length; i += 1) features[i] += f[i];
      }
      for (let i = 0; i < features.length; i += 1) features[i] /= present.length;
    }
    passes.push({ repeat: TEMPLATES, features, aatype: new Int32Array(batch.tokens) });
  } else {
    for (const { slot, mask } of slots) passes.push({ repeat: 1, slot, mask });
    const empty = TEMPLATES - slots.length, gap = dialect.emptyTemplateAatype ?? null;
    if (empty > 0 && gap !== null) {
      passes.push({ repeat: 1, emptyAatype: gap });
      if (empty > 1) passes.push({ repeat: empty - 1, emptyAatype: 0 });
    } else if (empty > 0) passes.push({ repeat: empty, emptyAatype: 0 });
  }
  passes.forEach((pass, k) => {
    const real = pass.slot !== undefined;
    const covered = !(dialect.templateVisibilityByCoverage === true && !real && pass.features === undefined);
    add(`template.${k}.repeat`, covered ? pass.repeat : 0);
    add(`template.${k}.aatype`, real ? Int32Array.from(pass.slot.aatype)
      : (pass.aatype ?? new Int32Array(batch.tokens).fill(pass.emptyAatype ?? 0)));
    if (fused) {
      add(`template.${k}.features`, pass.features ?? fusedTemplateFeatures(real ? pass.slot : undefined, batch.tokens,
        width, dialect, real ? pass.mask : undefined, (pass.emptyAatype ?? 0) !== 0));
    } else if (real) {
      const g = templateGeometry(pass.slot, pass.mask, batch.tokens);
      add(`template.${k}.distogram`, Float32Array.from(g.distogram));
      add(`template.${k}.pseudoBetaMask2d`, g.pseudoBetaMask2d);
      add(`template.${k}.unitVector`, g.unitVector);
      add(`template.${k}.backboneMask2d`, g.backboneMask2d);
    }
  });
  add("template.passes", passes.length);
  add("template.templates", TEMPLATES);
  add("template.featureWidth", width);
  add("template.outerResidual", dialect.templateStackOuterResidual === true);
}
// The PDB's records as the page writes them (src/af3/fold.js toPdb: chains, HETATM ligands under
// their codes, modified residues, CONECT), with each atom's dense slot as its x coordinate so the
// native writer knows which coordinates go where.
try {
  const { toPdb } = await import(`${repo}/src/af3/fold.js`);
  const slots = new Float32Array(batch.tokens * batch.dense * 3);
  for (let i = 0; i < batch.tokens * batch.dense; i += 1) slots[i * 3] = i;
  writeFileSync(`${out}/template.pdb`, toPdb(batch, slots, null) + "\n");
} catch (error) {
  console.log(`no PDB template (${error.message}); the native writer will name residues itself`);
}
// The oracle's own z_init/target_feat etc., by stage.
const flat = (record) => Float32Array.from(Array.isArray(record.data) ? record.data.flat(Infinity) : record.data);
for (const which of oracles) {
  // realdenoise: a real fold's trunk conditioning and structure, re-noised (sigma 2) - the
  // denoise oracle's own inputs are random and cannot judge a 16-bit path.
  const path = which === "realdenoise" ? `${repo}/oracle-dumps/af3-real-denoise-${oracleModel}-n2.0.json`
    : `${repo}/oracle-dumps/af3-oracle-${which}-${oracleModel}.json`;
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
// the index last and atomically: `af3 --wait-input` starts before this script and waits for it
writeFileSync(`${out}/model.idx.tmp`, lines.join("\n") + "\n");
renameSync(`${out}/model.idx.tmp`, `${out}/model.idx`);
console.log(`${entries.length} entries, ${(offset * 4 / 1048576).toFixed(0)} MiB` + (batch ? `, tokens ${batch.tokens}` : " (weights only)"));
