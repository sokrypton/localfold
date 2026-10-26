// LocalFold's merged AlphaFold 2 input for a case, before any random draw - the twin of vendor_features.py.
//
//     node cuda/af2/parity/ours_features.mjs <case.json> <out.json> [<merged.a3m>]
//
// The optional third path receives the merged alignment itself, which is what the WebGPU arm folds
// (tools/gpu/fold-af2.js --a3m=, the page's predictA3m on the page's merge).
//
// The same calls cuda/af2/export_input.mjs makes for --a3m/--paired-a3m (mergeSearchedChains, then the
// featuriser's own plan, whose context is the deduplicated, encoded alignment every recycle samples from).
import { readFileSync, writeFileSync } from "node:fs";
import { mergeSearchedChains, singleSequenceA3m } from "../../../shared/input/mmseqs2-api.js";
import { planA3mRecycles } from "../../../shared/input/a3m-features.js";

// (the atom37 tables only shape the structure module's maps, which this dump does not read)
const tables = { atom37ToAtom14: new Float32Array(21 * 37), atom37Mask: new Float32Array(21 * 37) };
const kase = JSON.parse(readFileSync(process.argv[2], "utf8"));
const chains = kase.sequences;
const model = kase.model_type.includes("multimer") ? "multimer" : "monomer";
let a3m;
// a case's `merged` names an alignment already merged (vendor_features.py's), used as it stands - as
// export_input.mjs uses a single --a3m for several chains
if (kase.merged) {
  a3m = readFileSync(kase.merged, "utf8");
} else if (kase.unpaired.some(Boolean) || (kase.paired ?? []).some(Boolean)) {
  const read = (path, seq) => (path ? readFileSync(path, "utf8") : `>101\n${seq}\n`);
  const paired = (kase.paired ?? []).map((path, index) => (path ? read(path, chains[index]) : ""));
  a3m = chains.length === 1 ? read(kase.unpaired[0], chains[0]) : mergeSearchedChains({
    sequences: chains,
    chainA3ms: chains.map((seq, index) => read(kase.unpaired[index], seq)),
    pairedA3ms: paired.some((text) => text.trim() !== "") ? new Map(chains.map((c, i) => [c, paired[i]])) : undefined,
    model,
  }).a3m;
} else {
  a3m = singleSequenceA3m(chains, model);
}
if (process.argv[4]) writeFileSync(process.argv[4], a3m);
const options = {
  ...(chains.length > 1 ? { chainAware: true, chainLengths: chains.map((c) => c.length), chainSequences: chains } : {}),
  recycles: 0, maxMsaSequences: kase.max_seq ?? 512, maxExtraSequences: kase.max_extra ?? 1024, randomSeed: 0,
};
const { context } = planA3mRecycles(a3m, tables, options);
const L = context.length;
const depth = context.encoded.length / L;
const b = context.base;
writeFileSync(process.argv[3], JSON.stringify({
  L, depth, msa: Array.from(context.encoded), deletion_matrix: Array.from(context.deletions),
  aatype: Array.from(b.aatype), residue_index: Array.from(b.residueIndex),
  asym_id: b.asymId ? Array.from(b.asymId) : null, entity_id: b.entityId ? Array.from(b.entityId) : null,
  sym_id: b.symId ? Array.from(b.symId) : null,
}));
console.log(`L ${L} depth ${depth}`);
