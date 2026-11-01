/**
 * Reads a tools/oracle/dump_bioemu2.py dump: every array is {shape, dtype, b64},
 * little-endian float32 or int32.
 */
import { readFileSync } from "node:fs";

export function decodeArray(entry) {
  const bytes = Buffer.from(entry.b64, "base64");
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const data = entry.dtype === "i32" ? new Int32Array(buffer) : new Float32Array(buffer);
  return Object.assign(data, { shape: entry.shape });
}

export function readBioemu2Dump(path) {
  const dump = JSON.parse(readFileSync(path, "utf8"));
  const stages = {};
  for (const [key, value] of Object.entries(dump.stages)) {
    stages[key] = value && typeof value === "object" && "b64" in value ? decodeArray(value) : value;
  }
  return { mode: dump.mode, stages };
}

/** A dump's SMILES template, in the form featuriseBioemu2 takes. */
export function smilesTemplatesOf(stages) {
  const out = {};
  for (const [key, value] of Object.entries(stages)) {
    if (!key.startsWith("smiles.")) continue;
    const bonds = decodeArray(value.bonds);
    const pairs = [];
    for (let i = 0; i < bonds.length; i += 2) pairs.push([bonds[i], bonds[i + 1]]);
    out[key.slice("smiles.".length)] = {
      names: value.names, element: Array.from(decodeArray(value.element)), charge: Array.from(decodeArray(value.charge)),
      coords: decodeArray(value.coords), bonds: pairs,
    };
  }
  return out;
}

/** relative RMS of a against b, refusing anything that is not two arrays of one length. */
export function relRms(a, b) {
  if (a?.length === undefined || b?.length === undefined || a.length !== b.length || a.length === 0) {
    throw new Error(`relRms over ${a?.length} and ${b?.length} values`);
  }
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i += 1) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  return Math.sqrt(num / Math.max(den, 1e-30));
}
