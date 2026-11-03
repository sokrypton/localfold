// A whole ESMFold2 fold on the CPU from node: the page's CPU path without the page.
//
//     node --js-float16array tools/fold-esmfold2-cpu.mjs --family=ef2-fast-300m \
//         --sequence=MQIF... --threads=8 --seed=0 --out=fold.json
//
// Weights come from each bundle's pinned remote. Prints the phase timings and the
// head's scores, and with --out writes the coordinates and representative atoms so a
// run can be scored against a structure. A sequence is split into chains on ":".
// It loads the whole model: run it where that is welcome.
import { writeFileSync } from "node:fs";
import process from "node:process";

import { CpuBackend } from "../cpu/runtime/backend.js";
import { setLinearBackend } from "../cpu/runtime/linear-backend.js";
import { foldEsmfold2Cpu } from "../web/esmfold2-cpu-fold.js";
import { loadEsmfold2CpuWeights } from "../cpu/esmfold2/load.js";
import { representativeAtoms } from "../shared/esmfold2/representative-atoms.js";

const flag = (name, fallback) => {
  const at = process.argv.find((argument) => argument.startsWith(`--${name}=`));
  return at === undefined ? fallback : at.slice(name.length + 3);
};
const UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
const family = flag("family", "ef2-fast-300m");
const sequence = flag("sequence", UBIQUITIN);
const threads = Number(flag("threads", "1"));
const seed = Number(flag("seed", "0"));
const out = flag("out", "");

const at = performance.now();
const backend = await CpuBackend.create({ threads });
setLinearBackend(backend);
console.error(`backend: ${backend.threads} thread(s)`);
const loaded = await loadEsmfold2CpuWeights(family, backend);
const loadSeconds = (performance.now() - at) / 1000;
console.error(`weights: ${loadSeconds.toFixed(1)} s, ${(backend.weightBytes / 2 ** 30).toFixed(2)} GiB`);

const chains = sequence.split(":");
const entities = chains.length === 1 ? sequence
  : { sequence, chainKinds: chains.map(() => "protein"), ligands: [], modifications: [] };
const foldOnce = (foldSeed) => foldEsmfold2Cpu({
  sequence, entities, shape: loaded.shape, weights: loaded.weights,
  confidenceWeights: loaded.confidenceWeights, language: loaded.language, seed: foldSeed,
  onStatus: (text) => process.stderr.write(`\r${text}          `),
});
// The first fold of a process also pays for decoding the tower's blocks, so --repeat
// folds again and reports the warm one, which is what a reader's second fold costs.
const repeats = Number(flag("repeat", "1"));
let result = await foldOnce(seed);
const firstSeconds = result.elapsedMilliseconds / 1000;
for (let repeat = 1; repeat < repeats; repeat += 1) result = await foldOnce(seed);
process.stderr.write("\n");

const mean = (values) => values.reduce((total, value) => total + value, 0) / values.length;
const rep = representativeAtoms(result.features, result.features.tokens);
const report = {
  family, threads: backend.threads, tokens: result.features.tokens, atoms: result.features.atoms,
  loadSeconds, firstFoldSeconds: firstSeconds, foldSeconds: result.elapsedMilliseconds / 1000,
  timings: Object.fromEntries(Object.entries(result.timings).map(([name, ms]) => [name, Math.round(ms) / 1000])),
  meanPlddt: 100 * mean(Array.from(result.confidence.plddt)),
  ptm: result.confidence.ptm, iptm: result.confidence.iptm,
  meanPae: mean(Array.from(result.confidence.pae)),
};
console.log(JSON.stringify(report));
if (out !== "") {
  writeFileSync(out, JSON.stringify({ ...report, sequence,
    coordinates: Array.from(result.coordinates), representative: Array.from(rep),
    plddt: Array.from(result.confidence.plddt) }));
}
backend.close();
process.exit(0);
