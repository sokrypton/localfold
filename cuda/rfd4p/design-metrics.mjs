/**
 * The page's design metrics for a dumped sample, as JSON.
 *
 *     node cuda/rfd4p/design-metrics.mjs --spec=<job.json> [--tags=design,unindexed] <dump dir>
 *
 * `localfold-rfd4p --dump-sample=<dir>` writes the sample; this runs `processSampleToAtoms` and
 * shared/rfd4p/metrics.js over it with the job's `metrics_tags` (or `--tags=`). NaN is written as
 * the string "NaN", which JSON has no number for. `--with-array` adds the array the metrics read -
 * each atom's identity and the bonds - which is what tools/check-rfd4p-metrics.py holds to
 * upstream's own, beside the values; gate.py holds the native binary's metrics to these.
 */
import { designArray, designMetrics, jobMetricsTags } from "../../shared/rfd4p/metrics.js";
import { sampleFromDump } from "./sample-from-dump.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const [dump] = args.filter((a) => !a.startsWith("--"));
const specPath = option("spec", "");
if (!dump || !specPath) {
  console.error("usage: design-metrics.mjs --spec=<job.json> [--tags=a,b] [--seed=N] <dump dir>");
  process.exit(2);
}
const { job, built, structure } = sampleFromDump(specPath, dump, Number(option("seed", "0")));
const tags = option("tags", null);
const metricsTags = tags !== null ? tags.split(",").filter(Boolean) : jobMetricsTags(job);
const out = designMetrics(structure, built.atoms, { metricsTags });
if (args.includes("--with-array")) {
  const arr = designArray(structure, built.atoms);
  out.atoms = Array.from({ length: arr.n }, (_, i) => [arr.chainId[i], arr.resId[i], arr.resName[i], arr.atomName[i]]);
  out.bonds = arr.bonds;
  // The realized length of each expandable segment, in sentinel order: what upstream's own
  // pipeline must be handed to build the same batch (its draw is its own RNG's).
  const uid = built.atoms._expseg_uid ?? [];
  const counts = [];
  for (const u of uid) if (u >= 0) counts[u] = (counts[u] ?? 0) + 1;
  out.segmentLengths = Array.from(counts, (c) => c ?? 0);
}
process.stdout.write(JSON.stringify(out, (_, v) => (typeof v === "number" && Number.isNaN(v) ? "NaN" : v)));
