/**
 * The page's writer on a dumped sample with `center_shift` taken back OUT, against a native file.
 *
 *     node tools/rfd4p-writer-unshifted.mjs --spec=<job.json> [--seed=N] <dump dir> <native .cif|.pdb>
 *
 * 🔴 IT EXISTS TO NAME ONE DIFFERENCE AND REFUSE ALL THE OTHERS. `shared/rfd4p/output.js` returns a design to
 * the input's frame by adding back the centroid `centerAndFeaturizeCentroid` subtracted, and no native port can:
 * `cuda/rfd4p`'s featuriser exports `g/coords` and `g/condCoord` already centred and drops the offset. So
 * check-writer.mjs rightly differs on every case whose input was not already at the origin - by a
 * sub-milliangstrom on some and by angstroms on `sm_sam` - and that one difference would hide every other.
 * Removing the shift from the PAGE's side and requiring the same bytes says the rest of the writer agrees.
 */
import { readFileSync } from "node:fs";
import { toCifString, toPdbString } from "../shared/rfd4p/output.js";
import { sampleFromDump } from "../cuda/rfd4p/sample-from-dump.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const [dump, nativePath] = args.filter((a) => !a.startsWith("--"));
const specPath = option("spec", "");
if (!dump || !nativePath || !specPath) {
  console.error("usage: rfd4p-writer-unshifted.mjs --spec=<job.json> [--seed=N] <dump dir> <native file>");
  process.exit(2);
}
const { built, structure } = sampleFromDump(specPath, dump, Number(option("seed", "0")));
const shift = built.atoms.center_shift ?? [0, 0, 0];
for (const a of structure.atoms) {
  a.x = Math.fround(a.x - shift[0]);
  a.y = Math.fround(a.y - shift[1]);
  a.z = Math.fround(a.z - shift[2]);
}
const want = nativePath.endsWith(".cif") ? toCifString(structure) : toPdbString(structure);
const have = readFileSync(nativePath, "utf8");
const shown = `[${shift.map((v) => v.toPrecision(6)).join(", ")}]`;
if (want === have) {
  console.log(`writer: identical with center_shift ${shown} removed, ${want.split("\n").length - 1} lines`);
  process.exit(0);
}
const a = want.split("\n"), b = have.split("\n");
let differ = 0, first = -1;
for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
  if (a[i] !== b[i]) { differ += 1; if (first < 0) first = i; }
}
console.log(`writer: DIFFERS on ${differ} of ${a.length} lines with center_shift ${shown} removed`);
console.log(`  first at line ${first + 1}:\n    page   ${a[first]}\n    native ${b[first]}`);
process.exit(1);
