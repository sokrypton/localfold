/**
 * Node helpers for reading RFD4-Proteina oracle dumps (`tools/oracle/dump_rfd4p.py`) and the
 * released checkpoint, and for comparing tensors.
 */

import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const DTYPES = {
  F32: Float32Array,
  F64: Float64Array,
  I64: BigInt64Array,
  I32: Int32Array,
  I16: Int16Array,
  I8: Int8Array,
  U8: Uint8Array,
};

/**
 * Read a safetensors file into `{ name: { shape, dtype, data } }`. Integer tensors are widened
 * to Int32Array/Float32Array-friendly typed arrays (I64 -> Int32Array).
 * @param {string} path
 */
export function readSafetensors(path) {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString('utf8'));
  const base = 8 + headerLen;
  const out = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const Ctor = DTYPES[info.dtype];
    if (!Ctor) throw new Error(`${name}: unsupported dtype ${info.dtype}`);
    const [s, e] = info.data_offsets;
    const bytes = buf.subarray(base + s, base + e);
    const aligned = new Uint8Array(bytes.length);
    aligned.set(bytes);
    let data = new Ctor(aligned.buffer, 0, bytes.length / Ctor.BYTES_PER_ELEMENT);
    if (data instanceof BigInt64Array) data = Int32Array.from(data, (v) => Number(v));
    out[name] = { shape: info.shape, dtype: info.dtype, data };
  }
  return out;
}

/** Path of the released `model.safetensors` in `weights/` or the Hugging Face cache. */
export function releasedWeightsPath() {
  const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
  const localWeights = join(rootDir, 'weights/model.safetensors');
  if (existsSync(localWeights)) {
    return { weights: localWeights, config: join(rootDir, 'weights/config.json') };
  }
  const snaps = join(homedir(), '.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots');
  for (const rev of readdirSync(snaps)) {
    const p = join(snaps, rev, 'model.safetensors');
    try {
      readFileSync(join(snaps, rev, 'config.json'));
      return { weights: p, config: join(snaps, rev, 'config.json') };
    } catch {
      // try next snapshot
    }
  }
  throw new Error(`no RFD4-Proteina snapshot under ${snaps}`);
}

/**
 * Load an oracle case: `{ t: tensors, meta }`.
 * @param {string} dir
 * @param {string} name
 */
export function loadOracle(dir, name) {
  return {
    t: readSafetensors(join(dir, `${name}.safetensors`)),
    meta: JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf8')),
  };
}

/**
 * Compare `got` against `want`: relative RMS error and max abs error.
 * @param {ArrayLike<number>} got
 * @param {ArrayLike<number>} want
 */
export function compare(got, want) {
  if (got.length !== want.length) throw new Error(`length mismatch ${got.length} vs ${want.length}`);
  let se = 0;
  let sw = 0;
  let maxAbs = 0;
  let nonFinite = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    if (!Number.isFinite(got[i])) nonFinite += 1;
    se += d * d;
    sw += want[i] * want[i];
    if (Math.abs(d) > maxAbs) maxAbs = Math.abs(d);
  }
  const relRms = Math.sqrt(se / Math.max(sw, 1e-30));
  return { relRms, maxAbs, nonFinite };
}
