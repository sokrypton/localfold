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
import { buildBatch, decodeTextOrZstd } from "../../shared/rfd4p/pipeline/index.js";
import { parseConditionSpecYaml } from "../../shared/rfd4p/pipeline/spec.js";
import { processSampleToAtoms, toCifString, toPdbString } from "../../shared/rfd4p/output.js";

const args = process.argv.slice(2);
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const [dump, nativePath] = args.filter((a) => !a.startsWith("--"));
const specPath = option("spec", "");
if (!dump || !nativePath || !specPath) {
  console.error("usage: check-writer.mjs --spec=<job.json> [--seed=N] <dump dir> <native file>");
  process.exit(2);
}
const seed = Number(option("seed", "0"));
const repo = new URL("../..", import.meta.url).pathname;

// The batch, built as `export_input.mjs --spec=` builds it.
const text = readFileSync(specPath, "utf8");
const job = text.trimStart().startsWith("{") ? JSON.parse(text) : parseConditionSpecYaml(text);
const cifUrl = job.cifUrl ?? job.input ?? null;
// A job is a spec unless it says `specMode: false` (featurise.cpp's note).
const specMode = job.specMode ?? true;
const inputCif = cifUrl
  ? decodeTextOrZstd(new Uint8Array(readFileSync(cifUrl.startsWith("/") ? cifUrl
    : `${repo}${cifUrl.replace(/^\.\//, "")}`)))
  : undefined;
const spec = {
  contig: job.contig,
  ...(job.conditions === undefined ? {} : { conditions: job.conditions }),
  ...(job.input_as_annotated_cif ?? job.inputAsAnnotatedCif ? { input_as_annotated_cif: true } : {}),
};
const built = !specMode && inputCif !== undefined
  ? buildBatch(inputCif, { seed })
  : buildBatch(spec, { seed, ...(inputCif === undefined ? {} : { inputCif }) });

const read = (name, Type) => {
  try {
    const b = readFileSync(`${dump}/${name}`);
    return new Type(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  } catch {
    return null;
  }
};
const structure = processSampleToAtoms({
  coors: read("coors.f32", Float32Array),
  atomMask: read("atomMask.u8", Uint8Array),
  residueType: read("residueType.i32", Int32Array),
  paddingMask: read("pad.f32", Float32Array),
  insertionLogits: read("insertion.f32", Float32Array),
  inputAtoms: built.atoms,
  batch: built.batch,
});
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
