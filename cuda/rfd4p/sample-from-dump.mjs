/**
 * A dumped sample, back through the page's writer: the batch `export_input.mjs --spec=` would build
 * for the job, and `processSampleToAtoms` over what `localfold-rfd4p --dump-sample=<dir>` wrote
 * (coordinates, atom mask, residue types, padding and the insertion head's logits). Shared by the
 * writer check and the metrics, so both read one sample the same way.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { buildBatch, decodeTextOrZstd } from "../../shared/rfd4p/pipeline/index.js";
import { parseConditionSpecYaml } from "../../shared/rfd4p/pipeline/spec.js";
import { processSampleToAtoms } from "../../shared/rfd4p/output.js";

const repo = new URL("../..", import.meta.url).pathname;

export function readJob(specPath) {
  const text = readFileSync(specPath, "utf8");
  return text.trimStart().startsWith("{") ? JSON.parse(text) : parseConditionSpecYaml(text);
}

export function sampleFromDump(specPath, dump, seed = 0) {
  const job = readJob(specPath);
  const cifUrl = job.cifUrl ?? job.input ?? null;
  // A job is a spec unless it says `specMode: false` (featurise.cpp's note).
  const specMode = job.specMode ?? true;
  // A job file's `input` is relative to the job file (upstream's rule); a preset's `cifUrl` is
  // relative to the repository, which is where the page serves it from.
  const base = job.input !== undefined ? dirname(resolve(specPath)) : repo;
  const inputCif = cifUrl ? decodeTextOrZstd(new Uint8Array(readFileSync(resolve(base, cifUrl)))) : undefined;
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
  return { job, built, structure };
}
