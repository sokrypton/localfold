/**
 * Minimal safetensors reader for the browser and Node: typed-array views over one ArrayBuffer (zero-copy when the
 * tensor is aligned, copied otherwise). I64 is narrowed to Int32Array; BF16/F16 are widened to Float32Array.
 */

const CTORS = { F32: Float32Array, F64: Float64Array, I32: Int32Array, I16: Int16Array, I8: Int8Array, U8: Uint8Array, BOOL: Uint8Array };

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

/**
 * @param {ArrayBuffer} buffer
 * @returns {{ tensors: Record<string, {shape: number[], dtype: string, data: ArrayLike<number>}>, metadata: object }}
 */
export function parseSafetensors(buffer) {
  const dv = new DataView(buffer);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen)));
  const base = 8 + headerLen;
  const tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [s, e] = info.data_offsets;
    const at = base + s;
    const bytes = e - s;
    let data;
    if (info.dtype === 'I64') {
      const src = new DataView(buffer, at, bytes);
      data = new Int32Array(bytes / 8);
      for (let i = 0; i < data.length; i += 1) data[i] = Number(src.getBigInt64(i * 8, true));
    } else if (info.dtype === 'F16' || info.dtype === 'BF16') {
      const src = new DataView(buffer, at, bytes);
      data = new Float32Array(bytes / 2);
      for (let i = 0; i < data.length; i += 1) {
        const h = src.getUint16(i * 2, true);
        data[i] = info.dtype === 'F16' ? halfToFloat(h) : new Float32Array(new Uint32Array([h << 16]).buffer)[0];
      }
    } else {
      const Ctor = CTORS[info.dtype];
      if (!Ctor) throw new Error(`${name}: unsupported safetensors dtype ${info.dtype}`);
      data = at % Ctor.BYTES_PER_ELEMENT === 0
        ? new Ctor(buffer, at, bytes / Ctor.BYTES_PER_ELEMENT)
        : new Ctor(buffer.slice(at, at + bytes));
    }
    tensors[name] = { shape: info.shape, dtype: info.dtype, data };
  }
  return { tensors, metadata: header.__metadata__ ?? {} };
}

// 🔴 THE INT8 SHARDS ARE NOT IN THIS REPOSITORY AND ARE NOT MEANT TO BE. They
// are ~420 MB and they already have a home: the port they were exported for,
// deployed at its own Pages URL. This repository's build carries 0.0 MiB of
// parameters and fetching them from there keeps it that way, with no second copy
// to drift from the first.
//
// 🔴 WHICH MAKES THE PAGES URL A DEPENDENCY, not a convenience: if that
// repository stops publishing, RFD4 stops loading here. The alternative - our
// own copy on Hugging Face beside the nine fold bundles - is the fallback to
// reach for if that happens, not a thing to do twice over now.
//
// A local `./model-rfd4p-int8` is still honoured first, so a developer who exported
// their own shards is not forced through the network. See docs/DESIGN.md.
export const RFD4P_PUBLISHED_BASE = 'https://sokrypton.github.io/rfd4-proteina-webgpu';
export const DEFAULT_LOCAL_WEIGHTS_URL = './weights/model.safetensors';
export const DEFAULT_LOCAL_SHARDED_DIR = './model-rfd4p-int8';
export const DEFAULT_LOCAL_CONFIG_URL = './model-rfd4p-int8/config.json';
export const DEFAULT_PUBLISHED_SHARDED_DIR = `${RFD4P_PUBLISHED_BASE}/weights/int8`;
export const DEFAULT_PUBLISHED_CONFIG_URL = `${RFD4P_PUBLISHED_BASE}/weights/config.json`;
export const DEFAULT_HF_REPO = 'nvidia/RFD4-Proteina';
export const DEFAULT_HF_WEIGHTS_URL = `https://huggingface.co/${DEFAULT_HF_REPO}/resolve/main/model.safetensors`;
export const DEFAULT_HF_CONFIG_URL = `https://huggingface.co/${DEFAULT_HF_REPO}/resolve/main/config.json`;
const WEIGHT_CACHE_NAME = 'rfd4p-weights-v1';
const MAX_CONCURRENT_DOWNLOADS = 8;
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
const CACHE_TIMEOUT_MS = 2000;

/**
 * Whether `url` is served by another machine (worth keeping in the Cache API). Relative URLs are resolved against the
 * page, so the deployed site's `./weights/...` counts as remote; a local dev server does not.
 */
function isRemoteUrl(url) {
  let resolved;
  try {
    resolved = new URL(url, globalThis.location?.href);
  } catch {
    return false;
  }
  return /^https?:$/.test(resolved.protocol) && !['localhost', '127.0.0.1', '[::1]'].includes(resolved.hostname);
}

function withTimeout(promise, fallback) {
  return Promise.race([
    promise,
    new Promise((resolve) => setTimeout(() => resolve(fallback), CACHE_TIMEOUT_MS)),
  ]);
}

async function fetchWithRetry(url, signal = null) {
  let lastStatus = 0;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const res = await fetch(url, { signal });
    if (res.ok) return res;
    lastStatus = res.status;
    if (!RETRYABLE_STATUS.has(res.status) || attempt === 4) break;
    await res.body?.cancel();
    await new Promise((r) => setTimeout(r, 250 * 2 ** attempt));
  }
  throw new Error(`Failed to fetch ${url}: HTTP ${lastStatus}`);
}

function viewAligned(Kind, buffer, byteOffset, length) {
  const width = Kind.BYTES_PER_ELEMENT;
  if (byteOffset % width === 0) return new Kind(buffer, byteOffset, length);
  return new Kind(buffer.slice(byteOffset, byteOffset + length * width));
}

/**
 * Decode a single tensor record from a shard `ArrayBuffer` into a `Float32Array`,
 * matching `../localfold/shared/weights/dtype.js` (`float32`, `float16`, and symmetric block-64 `int8`).
 *
 * @param {{dtype: string, shape: number[], byteOffset: number, block?: number, scaleOffset?: number}} record
 * @param {ArrayBuffer} buffer
 * @param {boolean} [copy=false]
 * @returns {Float32Array}
 */
export function readShardedTensor(record, buffer, copy = false) {
  const elements = record.shape.reduce((p, v) => p * v, 1);
  const { dtype, byteOffset = 0 } = record;

  if (dtype === 'float32') {
    if (!copy && byteOffset % 4 === 0) {
      return new Float32Array(buffer, byteOffset, elements);
    }
    return new Float32Array(buffer.slice(byteOffset, byteOffset + elements * 4));
  }

  if (dtype === 'float16') {
    if (typeof Float16Array === 'function') {
      const f16 = viewAligned(Float16Array, buffer, byteOffset, elements);
      return Float32Array.from(f16);
    }
    const u16 = viewAligned(Uint16Array, buffer, byteOffset, elements);
    const out = new Float32Array(elements);
    for (let i = 0; i < elements; i += 1) out[i] = halfToFloat(u16[i]);
    return out;
  }

  if (dtype === 'int8') {
    const block = record.block ?? 64;
    const scaleAt = record.scaleOffset;
    const codes = new Int8Array(buffer, byteOffset, elements);
    const blocks = Math.ceil(elements / block);
    const out = new Float32Array(elements);
    if (typeof Float16Array === 'function') {
      const scales = viewAligned(Float16Array, buffer, scaleAt, blocks);
      for (let b = 0; b < blocks; b += 1) {
        const scale = scales[b];
        const start = b * block;
        const end = Math.min(start + block, elements);
        for (let i = start; i < end; i += 1) out[i] = codes[i] * scale;
      }
    } else {
      const scaleBits = viewAligned(Uint16Array, buffer, scaleAt, blocks);
      for (let b = 0; b < blocks; b += 1) {
        const scale = halfToFloat(scaleBits[b]);
        const start = b * block;
        const end = Math.min(start + block, elements);
        for (let i = start; i < end; i += 1) out[i] = codes[i] * scale;
      }
    }
    return out;
  }

  throw new Error(`Unsupported sharded tensor dtype: ${dtype}`);
}

/** Small int8 tensors keep their decode (float32 ones are views of the shard and cost nothing to keep). */
const LAZY_KEEP_ELEMENTS = 1 << 14;

/**
 * A tensor record that keeps the shard's bytes and decodes `data` when it is read, as `../localfold`'s tensor
 * store does. Holding every tensor decoded cost float32 for all of it, kept for the session although only
 * the GPU upload (and the CPU fallback) reads the large ones: 1,169 MB for the core model and 492 MB more with
 * the encoder, against 316 and 131 MB for the shard bytes this keeps instead. Large tensors decode on each
 * read, which is once per GPU model build.
 */
function lazyTensor(rec, buf) {
  const elements = rec.shape.reduce((p, v) => p * v, 1);
  const keep = rec.dtype === 'float32' || elements <= LAZY_KEEP_ELEMENTS;
  let decoded = null;
  return {
    shape: rec.shape,
    dtype: 'F32',
    get data() {
      if (decoded) return decoded;
      const data = readShardedTensor(rec, buf, false);
      if (keep) decoded = data;
      return data;
    },
  };
}

/**
 * Load a `../localfold`-style sharded bundle (`manifest.json` + `shard_XXX.bin`) over HTTP
 * with 8-way concurrent shard downloads, Cache API persistence, and optional lazy Encoder loading.
 *
 * @param {string} [baseDir] Directory URL containing `manifest.json` and `.bin` shards
 * @param {{
 *   includeEncoder?: boolean,
 *   onProgress?: (p: {bytesLoaded: number, loadedBytes: number, totalBytes: number, loadedShards: number, totalShards: number, fraction: number}) => void,
 *   signal?: AbortSignal,
 *   cache?: boolean,
 * }} [opts]
 */
export async function fetchShardedWeights(baseDir = null, opts = {}) {
  const cleanBase = (baseDir ?? await resolveRfd4pShardBase(opts.signal ?? null)).replace(/\/+$/, '');
  const isRemote = isRemoteUrl(cleanBase);
  const {
    includeEncoder = false,
    onProgress = null,
    signal = null,
    cache: useCache = isRemote,
  } = opts;

  const manifestRes = await fetchWithRetry(`${cleanBase}/manifest.json`, signal);
  const manifest = await manifestRes.json();
  const tensorsMeta = manifest.tensors ?? {};

  // Group tensor names by shard file
  const byShard = new Map();
  for (const [name, rec] of Object.entries(tensorsMeta)) {
    let list = byShard.get(rec.file);
    if (!list) {
      list = [];
      byShard.set(rec.file, list);
    }
    list.push([name, rec]);
  }

  const allShards = manifest.shards ?? [...byShard.keys()].map((file) => ({ file, bytes: 0, role: 'core' }));
  let cacheStore = null;
  if (useCache && typeof caches !== 'undefined') {
    const cacheName = `rfd4p-sharded-${manifest.bundle?.format ?? 'int8'}-${manifest.bundle?.bytes ?? 0}`;
    cacheStore = await withTimeout(caches.open(cacheName).catch(() => null), null);
    if (cacheStore) {
      // Ask the browser not to evict the cached shards under storage pressure (granted silently in Chrome for
      // engaged sites; a no-op elsewhere). Persisted storage is never evicted, so drop shard caches of older bundles.
      navigator.storage?.persist?.().catch(() => {});
      caches.keys()
        .then((names) => Promise.all(names
          .filter((name) => name.startsWith('rfd4p-sharded-') && name !== cacheName)
          .map((name) => caches.delete(name))))
        .catch(() => {});
    }
  }

  const tensors = {};

  async function downloadShardList(shardList, progressCb) {
    const totalBytes = shardList.reduce((acc, s) => acc + (s.bytes || 0), 0);
    const totalShards = shardList.length;
    let bytesLoaded = 0;
    let loadedShards = 0;
    let cursor = 0;

    async function worker() {
      while (cursor < shardList.length) {
        const idx = cursor++;
        const shard = shardList[idx];
        const shardUrl = `${cleanBase}/${shard.file}`;
        let buf = null;

        if (cacheStore) {
          const hit = await withTimeout(cacheStore.match(shardUrl).catch(() => null), null);
          if (hit && hit.ok) {
            buf = await hit.arrayBuffer();
          }
        }

        if (!buf) {
          const res = await fetchWithRetry(shardUrl, signal);
          buf = await res.arrayBuffer();
          if (cacheStore) {
            await withTimeout(
              cacheStore
                .put(
                  shardUrl,
                  new Response(buf.slice(0), {
                    headers: {
                      'content-type': 'application/octet-stream',
                      'content-length': String(buf.byteLength),
                    },
                  }),
                )
                .catch(() => null),
              null,
            );
          }
        }

        for (const [name, rec] of byShard.get(shard.file) ?? []) tensors[name] = lazyTensor(rec, buf);

        bytesLoaded += shard.bytes || buf.byteLength;
        loadedShards += 1;
        if (progressCb) {
          progressCb({
            bytesLoaded,
            loadedBytes: bytesLoaded,
            totalBytes: totalBytes || bytesLoaded,
            loadedShards,
            totalShards,
            fraction: totalBytes > 0 ? bytesLoaded / totalBytes : loadedShards / totalShards,
          });
        }
      }
    }

    const nWorkers = Math.min(MAX_CONCURRENT_DOWNLOADS, shardList.length);
    await Promise.all(Array.from({ length: nWorkers }, () => worker()));
  }

  const coreShards = includeEncoder ? allShards : allShards.filter((s) => s.role !== 'encoder');
  const encoderShards = allShards.filter((s) => s.role === 'encoder');
  let encoderLoaded = includeEncoder || encoderShards.length === 0;

  await downloadShardList(coreShards, onProgress);

  let config = null;
  try {
    config = await fetchRfd4pConfig(`${cleanBase}/config.json`);
  } catch {
    try {
      config = await fetchRfd4pConfig(DEFAULT_LOCAL_CONFIG_URL);
    } catch {
      try {
        config = await fetchRfd4pConfig(DEFAULT_PUBLISHED_CONFIG_URL);
      } catch {
        /* optional */
      }
    }
  }

  return {
    tensors,
    config,
    metadata: manifest.metadata ?? {},
    manifest,
    get encoderLoaded() {
      return encoderLoaded;
    },
    async loadEncoder(encProgress = onProgress) {
      if (encoderLoaded) return tensors;
      await downloadShardList(encoderShards, encProgress);
      encoderLoaded = true;
      return tensors;
    },
  };
}

/**
 * Where the INT8 shards are, this run: a local export if one is there, else the published port's.
 *
 * 🔴 PROBED ON A SHARD, NOT ON THE MANIFEST, and the difference is a bug that
 * shipped for one commit. `manifest.json` is COMMITTED to this repository - the
 * tests read it without a network - so a probe on it always succeeds and always
 * chose the local base, which then 404'd on `shard_000.bin` because the shards
 * are gitignored and were never there. The question is "are the WEIGHTS here",
 * and only a weight file answers it.
 *
 * The two cases are a developer who ran tools/export_rfd4p_model.py and a
 * visitor who has never heard of it; the visitor is the common one. The Cache
 * API keeps the shards after the first visit.
 *
 * @param {AbortSignal|null} [signal]
 * @returns {Promise<string>} the base directory to read shards from
 */
export async function resolveRfd4pShardBase(signal = null) {
  // The exporter's first core shard. Any shard would do; this one is always
  // written, and asking for a byte of it costs nothing.
  const local = await fetch(`${DEFAULT_LOCAL_SHARDED_DIR}/shard_000.bin`,
    { method: 'HEAD', signal }).catch(() => null);
  if (local && local.ok) return DEFAULT_LOCAL_SHARDED_DIR;
  return DEFAULT_PUBLISHED_SHARDED_DIR;
}

/**
 * Fetch and parse RFD4-Proteina `config.json` (tries local, then the published port, then HuggingFace).
 * @param {string} [url]
 * @returns {Promise<object>} Unwrapped model configuration (`{ nn, autoencoder_config, ... }`)
 */
export async function fetchRfd4pConfig(url = DEFAULT_LOCAL_CONFIG_URL) {
  let res = await fetch(url).catch(() => null);
  if (!res || !res.ok) {
    if (url === DEFAULT_LOCAL_CONFIG_URL) {
      res = await fetch(DEFAULT_HF_CONFIG_URL);
    }
  }
  if (!res || !res.ok) {
    throw new Error(`Failed to fetch RFD4-Proteina config from ${url} (${res?.status ?? 'network error'})`);
  }
  const raw = await res.json();
  return raw.config ?? raw;
}

/**
 * Fetch and parse a `.safetensors` checkpoint from `url`, using the browser Cache API when available
 * and streaming into a single zero-copy `ArrayBuffer` with progress callbacks.
 *
 * @param {string} [url]
 * @param {{onProgress?: (p: {bytesLoaded: number, loadedBytes: number, totalBytes: number, fraction: number, cached?: boolean}) => void, signal?: AbortSignal, cache?: boolean}} [opts]
 */
export async function fetchSafetensors(url = DEFAULT_LOCAL_WEIGHTS_URL, opts = {}) {
  const isRemote = isRemoteUrl(url);
  const { onProgress = null, signal = null, cache: useCache = isRemote } = opts;
  let cacheStore = null;
  if (useCache && typeof caches !== 'undefined') {
    try {
      cacheStore = await caches.open(WEIGHT_CACHE_NAME);
      const hit = await cacheStore.match(url);
      if (hit && hit.ok) {
        const buf = await hit.arrayBuffer();
        if (onProgress) {
          onProgress({
            bytesLoaded: buf.byteLength,
            loadedBytes: buf.byteLength,
            totalBytes: buf.byteLength,
            fraction: 1,
            cached: true,
          });
        }
        return parseSafetensors(buf);
      }
    } catch {
      cacheStore = null;
    }
  }

  const res = await fetch(url, { signal });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
  }

  const totalBytes = Number(res.headers.get('content-length')) || 0;
  let buffer;
  if (onProgress && res.body && totalBytes > 0) {
    const u8 = new Uint8Array(totalBytes);
    const reader = res.body.getReader();
    let offset = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      u8.set(value, offset);
      offset += value.byteLength;
      onProgress({
        bytesLoaded: offset,
        loadedBytes: offset,
        totalBytes,
        fraction: offset / totalBytes,
        cached: false,
      });
    }
    buffer = u8.buffer;
  } else {
    buffer = await res.arrayBuffer();
    if (onProgress) {
      onProgress({
        bytesLoaded: buffer.byteLength,
        loadedBytes: buffer.byteLength,
        totalBytes: buffer.byteLength,
        fraction: 1,
        cached: false,
      });
    }
  }

  if (cacheStore) {
    try {
      await cacheStore.put(
        url,
        new Response(buffer, {
          headers: {
            'content-type': 'application/octet-stream',
            'content-length': String(buffer.byteLength),
          },
        }),
      );
    } catch {
      // Quota exceeded or opaque origin — ignore cache write failure
    }
  }

  return parseSafetensors(buffer);
}
