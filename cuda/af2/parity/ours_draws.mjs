// LocalFold's own random draws for a case: per recycle, the cluster centres, the extra rows and the BERT-masked
// centre codes, over the deduplicated merged alignment they index - what sampling_parity.py hands stock AF2's
// feature functions, so both sides build features from the SAME draws.
//
//     node cuda/af2/parity/ours_draws.mjs <case.json> <out.json> --recycles=R --max-msa=N --max-extra=E --seed=S
//
// The options mean what cuda/af2/export_input.mjs's mean, and the plan is the featuriser's own
// (planA3mRecycles), which the native featuriser reproduces byte for byte (npm run test:featurise).
import { readFileSync, writeFileSync } from "node:fs";
import { mergeSearchedChains, singleSequenceA3m } from "../../../shared/input/mmseqs2-api.js";
import { planA3mRecycles } from "../../../shared/input/a3m-features.js";

const flag = (name, fallback) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const kase = JSON.parse(readFileSync(process.argv[2], "utf8"));
const chains = kase.sequences;
const model = kase.model_type.includes("multimer") ? "multimer" : "monomer";
const read = (path, seq) => (path ? readFileSync(path, "utf8") : `>101\n${seq}\n`);
let a3m;
// a case's `merged` names an alignment already merged (vendor_features.py's), used as it stands - as
// export_input.mjs uses a single --a3m for several chains
if (kase.merged) {
  a3m = readFileSync(kase.merged, "utf8");
} else if (kase.unpaired.some(Boolean)) {
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
const tables = { atom37ToAtom14: new Float32Array(21 * 37), atom37Mask: new Float32Array(21 * 37) };
const options = {
  ...(chains.length > 1 ? { chainAware: true, chainLengths: chains.map((c) => c.length), chainSequences: chains } : {}),
  recycles: Number(flag("recycles", "3")), maxMsaSequences: Number(flag("max-msa", "512")),
  maxExtraSequences: Number(flag("max-extra", "1024")), randomSeed: Number(flag("seed", "0")),
};
const { plans, context } = planA3mRecycles(a3m, tables, options);
const b64 = (typed) => Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString("base64");
writeFileSync(process.argv[3], JSON.stringify({
  L: context.length, depth: context.encoded.length / context.length,
  msa: b64(context.encoded), deletions: b64(context.deletions),
  plans: plans.map((p) => ({ centers: p.centers, extras: p.extras, centerCodes: b64(p.centerCodes) })),
}));
console.log(`L ${context.length} depth ${context.encoded.length / context.length} passes ${plans.length}`);
