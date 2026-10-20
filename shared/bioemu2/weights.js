/**
 * Reading a BioEmu2 bundle (tools/export_bioemu2_model.py): a reader over a
 * base URL, shaped as shared/esmfold2/weights.js's loaders want it - `read(name)`
 * decodes a tensor, `read.source(name)` hands back its undecoded bytes for the
 * device packers and `read.decode(source)` is the host fallback.
 *
 * 🔴 THE PAIR TRACK IS IN AF3'S LAYOUT AND EVERYTHING ELSE IN THE VENDOR'S. A
 * `pairmixer/<i>/...` or `msa/<i>/triangle...` tensor is what webgpu/af3's pair
 * track packs; a `structure_module/...` one is torch's (out, in). See the
 * exporter's header for why.
 */
import { readTensor } from "../weights/dtype.js";

export function bundleReader(base, { manifest: given, fetcher = fetch } = {}) {
  const shards = new Map();
  let table = given;
  const manifest = async () => {
    if (table === undefined) table = await (await fetcher(`${base}/manifest.json`)).json();
    return table;
  };
  const openShard = (record) => {
    if (!shards.has(record.file)) {
      shards.set(record.file, (async () => {
        const response = await fetcher(`${base}/${record.file}`);
        if (!response.ok) throw new Error(`${base}/${record.file}: ${response.status}`);
        return response.arrayBuffer();
      })());
    }
    return shards.get(record.file);
  };
  const recordFor = async (name) => {
    const record = (await manifest()).tensors[name];
    if (record === undefined) throw new Error(`${base} has no tensor ${name}`);
    return record;
  };
  const read = async (name) => {
    const record = await recordFor(name);
    return readTensor(record, await openShard(record), record.byteOffset ?? 0, true);
  };
  read.has = async (name) => (await manifest()).tensors[name] !== undefined;
  read.source = async (name) => {
    const record = await recordFor(name);
    return { record, buffer: await openShard(record), byteOffset: record.byteOffset ?? 0 };
  };
  read.decode = (s) => readTensor(s.record, s.buffer, s.byteOffset, true);
  read.manifest = manifest;
  read.prefetch = async () => {
    const files = new Set(Object.values((await manifest()).tensors).map((r) => r.file));
    await Promise.all([...files].map((file) => openShard({ file })));
  };
  return read;
}

/** The checkpoint's dimensions; the bundle states them, these are its values. */
export const BIOEMU2 = Object.freeze({
  pair: 256, single: 384, token: 768, atom: 128, atomPair: 16, msa: 128, sInputs: 455,
  pairmixerBlocks: 48, msaBlocks: 4, denoiserLayers: 16, inputLayers: 3,
  tokenHeads: 16, atomHeads: 4, msaHeads: 8, trunkPasses: 4, distogramBins: 64,
  sigmaData: 16, windowQueries: 32, windowKeys: 128,
});

/** BioEmu2's EDMSampler as af3-any-model runs it (model_registry's constants). */
export const BIOEMU2_SAMPLER = Object.freeze({
  gamma0: 0.8, gammaMin: 1.0, noiseScale: 1.0, rho: 7, sigmaMin: 0.0004, sigmaMax: 160.0, steps: 200,
  stepScale: { structure: 1.5, equilibrium: 1.0 },
});
