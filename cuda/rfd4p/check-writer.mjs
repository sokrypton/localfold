/**
 * Hold the native port's structure WRITER to the page's: the same sample in, the same text out.
 *
 *     node cuda/rfd4p/check-writer.mjs --spec=<job.json> [--seed=N] <dump dir> <native .cif|.pdb>
 *
 * `localfold-rfd4p --dump-sample=<dir>` writes the decoded sample before any of the writer's own
 * steps - coordinates, atom mask, residue types, padding and the insertion head's logits - which is
 * exactly what `processSampleToAtoms` takes. This runs the page's writer on it, over the batch
 * `export_input.mjs` would build for the same job, and requires the native file BYTE FOR BYTE.
 *
 * 🔴 IT EXISTS BECAUSE NOTHING COMPARED THE TWO WRITERS, AND THEY DISAGREED ON MOST STRUCTURES.
 * The native one named every atom by its GLOBAL atom37 slot, so a DNA or RNA residue came out
 * with protein names (P as `N`, C1' as `CA`); it wrote a ligand atom as `UNK CA`, element C,
 * `ATOM`; and it carried no B-factor, occupancy, charge, entity or source component from the
 * input. Every geometry check passed, because none of that is geometry.
 */
import { readFileSync } from "node:fs";
import { toCifString, toPdbString } from "../../shared/rfd4p/output.js";
import { sampleFromDump } from "./sample-from-dump.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const [dump, nativePath] = args.filter((a) => !a.startsWith("--"));
const specPath = option("spec", "");
if (!dump || !nativePath || !specPath) {
  console.error("usage: check-writer.mjs --spec=<job.json> [--seed=N] <dump dir> <native file>");
  process.exit(2);
}
// The batch built as `export_input.mjs --spec=` builds it, and the page's writer over the dump.
const { structure } = sampleFromDump(specPath, dump, Number(option("seed", "0")));
const want = nativePath.endsWith(".cif") ? toCifString(structure) : toPdbString(structure);
if (option("write-page", "")) (await import("node:fs")).writeFileSync(option("write-page", ""), want);
const have = readFileSync(nativePath, "utf8");
if (want === have) {
  console.log(`writer: identical, ${want.split("\n").length - 1} lines`);
  process.exit(0);
}
const a = want.split("\n"), b = have.split("\n");
let differ = 0, first = -1;
for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
  if (a[i] !== b[i]) { differ += 1; if (first < 0) first = i; }
}
console.log(`writer: DIFFERS on ${differ} of ${a.length} lines (page) / ${b.length} (native)`);
console.log(`  first at line ${first + 1}:\n    page   ${a[first]}\n    native ${b[first]}`);
process.exit(1);
