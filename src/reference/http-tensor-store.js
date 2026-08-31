import { readTensor, tensorByteLength } from "./dtype.js";
/**
 * @typedef {object} TensorDownloadProgress
 * @property {number} loadedBytes
 * @property {number} totalBytes
 * @property {number} loadedTensors
 * @property {number} totalTensors
 * @property {string} [tensorName]  the tensor in flight, when one is
 */

/** @typedef {(progress: TensorDownloadProgress) => void} TensorDownloadProgressCallback */

const MAX_CONCURRENT_DOWNLOADS = 8;
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function fetchWithRetry(url, label) {
  let lastStatus = 0;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const response = await fetch(url);
    if (response.ok) return response;
    lastStatus = response.status;
    if (!RETRYABLE_STATUS.has(response.status) || attempt === 4) break;
    await response.body?.cancel();
    await delay(250 * 2 ** attempt);
  }
  throw new Error(`failed to load ${label}: ${lastStatus}`);
}

// --- the shard cache --------------------------------------------------------
//
// WHY. The weights are 355 MiB and every page load re-fetched all of them: the
// in-memory map below dies with the page, so a reload started from zero. For a
// page people open, poke at, close and reopen, that download IS the runtime.
//
// The Cache API rather than IndexedDB because the thing being stored is a
// Response and this is what it is for. It is unavailable in an insecure context
// - file:// included - and can refuse on quota, so every call here is guarded
// and a failure only costs the speed-up: the fetch path underneath is unchanged.

const CACHE_PREFIX = "localfold-model-";
const CACHE_TIMEOUT_MS = 2_000;

/**
 * Give up on the cache rather than on the load.
 *
 * Every call here is on the critical path of getting 355 MiB into the browser,
 * and Cache Storage does not always answer: it hangs outright in headless
 * Chrome, and can stall under storage pressure or in a private window. A
 * rejection was already handled; a call that never settles would have hung
 * model loading forever, which is a far worse failure than not caching.
 */
function withTimeout(promise, fallback) {
  return Promise.race([
    promise,
    new Promise((resolve) => setTimeout(() => resolve(fallback), CACHE_TIMEOUT_MS)),
  ]);
}

/**
 * A token that changes when the model does.
 *
 * It goes in the cache NAME, not the key, so a new model does not quietly serve
 * a previous one's shards from the same URLs - and the sweep below then drops
 * the whole stale cache in one call rather than leaving 355 MiB orphaned.
 */
function cacheToken(manifest) {
  const bundle = manifest.bundle ?? {};
  return `${bundle.model ?? "model"}-${bundle.bytes ?? 0}-${Object.keys(manifest.tensors).length}`;
}

async function openShardCache(manifest) {
  if (typeof caches === "undefined") return undefined;
  const name = CACHE_PREFIX + cacheToken(manifest);
  try {
    return await withTimeout((async () => {
      for (const existing of await caches.keys()) {
        if (existing.startsWith(CACHE_PREFIX) && existing !== name) await caches.delete(existing);
      }
      return caches.open(name);
    })(), undefined);
  } catch {
    return undefined;                 // insecure origin, private mode, no quota
  }
}

/** Lazy browser tensor store backed by a JSON manifest and fetchable weight shards. */
export class HttpTensorStore {
  manifestUrl;
  manifest;
  #cache = new Map();
  #fileCache = new Map();
  #fileByteLengths = new Map();
  #pending = [];
  #onProgress;
  #totalBytes;
  #totalTensors;
  #activeDownloads = 0;
  #loadedBytes = 0;
  #loadedTensors = 0;
  #shardCache;
  #shardQuery = "";
  constructor(manifestUrl, manifest, onProgress, shardCache, shardQuery = "") {
    this.manifestUrl = manifestUrl; this.manifest = manifest; this.#onProgress = onProgress;
    this.#shardCache = shardCache; this.#shardQuery = shardQuery;
    const records = Object.values(manifest.tensors);
    this.#totalTensors = records.length;
    // 🔴 EACH TENSOR'S OWN WIDTH, not four bytes. The shards are float16 except
    // where rounding would be reckless, so assuming float32 here overstates how
    // long a shard should be - and #readStream returns undefined when the bytes
    // it counted do not reach that length, which surfaces as "invalid byte
    // length" on a download that was in fact complete.
    this.#totalBytes = records.reduce((sum, record) => sum + tensorByteLength(record), 0);
    for (const record of records) {
      const end = (record.byteOffset ?? 0) + tensorByteLength(record);
      this.#fileByteLengths.set(record.file, Math.max(this.#fileByteLengths.get(record.file) ?? 0, end));
    }
  }
  static async open(manifestUrlValue,
    onProgress) {
    const manifestUrl = manifestUrlValue instanceof URL
      ? manifestUrlValue
      : new URL(manifestUrlValue, typeof location !== "undefined" ? location.href : "http://localhost/");
    const response = await fetchWithRetry(manifestUrl, "model manifest");
    const manifest = await response.json();
    return this.fromManifest(manifestUrl, manifest, onProgress);
  }
  /**
   * @param {string} [shardQuery] a `?v=...` appended to every shard URL. The
   *   manifest is small enough to cache-bust on every load, the shards are not,
   *   and a re-export that pairs fresh manifest with a browser-cached shard
   *   fails as "invalid byte length" - which names neither half. Dev harnesses
   *   pass a token here; the shipped page leaves it empty so the shards cache.
   */
  static async fromManifest(manifestUrlValue, manifest, onProgress = undefined, shardQuery = "") {
    const manifestUrl = manifestUrlValue instanceof URL
      ? manifestUrlValue
      : new URL(manifestUrlValue, typeof location !== "undefined" ? location.href : "http://localhost/");
    if (manifest.tensors === undefined) throw new Error("model manifest has no tensor table");
    const store = new HttpTensorStore(manifestUrl, manifest, onProgress,
      await openShardCache(manifest), shardQuery);
    store.#reportProgress();
    return store;
  }
  tensor(name) {
    let value = this.#cache.get(name);
    if (value === undefined) { value = this.#load(name); this.#cache.set(name, value); }
    return value;
  }
  shape(name) {
    const record = this.manifest.tensors[name]; if (record === undefined) throw new Error(`missing tensor ${name}`);
    return record.shape;
  }
  async #load(name) {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`missing tensor ${name}`);
    let pendingFile = this.#fileCache.get(record.file);
    if (pendingFile === undefined) {
      pendingFile = this.#scheduleDownload(record.file, name);
      this.#fileCache.set(record.file, pendingFile);
    }
    const buffer = await pendingFile;
    const byteOffset = record.byteOffset ?? 0;
    const byteLength = tensorByteLength(record);
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
      throw new Error(`${name} points outside ${record.file}`);
    }
    this.#loadedTensors += 1;
    this.#reportProgress(name);
    return readTensor(record, buffer, byteOffset);
  }
  async #scheduleDownload(file, tensorName) {
    return new Promise((resolve, reject) => {
      const start = () => {
        this.#activeDownloads += 1;
        void this.#downloadFile(file, tensorName).then(resolve, reject).finally(() => {
          this.#activeDownloads -= 1;
          this.#pending.shift()?.();
        });
      };
      if (this.#activeDownloads < MAX_CONCURRENT_DOWNLOADS) start();
      else this.#pending.push(start);
    });
  }
  async #downloadFile(file, tensorName) {
    const url = new URL(file + this.#shardQuery, this.manifestUrl);
    const expectedLength = this.#fileByteLengths.get(file);
    if (expectedLength === undefined) throw new Error(`${file} is absent from the manifest`);
    // A HIT IS STILL READ THROUGH THE SAME LOOP, not returned whole, so the
    // progress callback reports the same way on a cached load as on a cold one.
    // It just finishes in a moment.
    const hit = await this.#cacheMatch(url);
    if (hit !== undefined) {
      const bytes = await this.#readStream(hit, file, expectedLength);
      if (bytes !== undefined) return bytes;
      // ...a short or corrupt entry: drop it and fall through to the network.
      await this.#cacheDelete(url);
    }
    const response = await fetchWithRetry(url, `tensor ${tensorName}`);
    // The cache copy is taken BEFORE the body is read, because a Response body
    // can only be consumed once and the clone has to be made while it is intact.
    // Storing it is best-effort: over quota the put throws and the load carries
    // on, one slow page instead of a broken one.
    void this.#cachePut(url, response.clone());
    if (response.body === null) {
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength !== expectedLength) throw new Error(`${file} has an invalid byte length`);
      this.#loadedBytes += buffer.byteLength;
      this.#reportProgress();
      return buffer;
    }
    const bytes = await this.#readStream(response, file, expectedLength);
    if (bytes === undefined) throw new Error(`${file} has an invalid byte length`);
    return bytes;
  }

  /**
   * Read a response body into an exactly-sized buffer, reporting as it goes.
   *
   * Returns undefined rather than throwing when the length is wrong, because
   * the caller treats that differently depending on where the bytes came from:
   * a bad cache entry is dropped and refetched, a bad download is an error.
   * Whatever it counted is rolled back first, so the progress bar cannot end up
   * past its own total after a retry.
   */
  async #readStream(response, file, expectedLength) {
    if (response.body === null) return undefined;
    const output = new Uint8Array(expectedLength);
    const reader = response.body.getReader();
    let offset = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (offset + value.byteLength > output.byteLength) break;
        output.set(value, offset);
        offset += value.byteLength;
        this.#loadedBytes += value.byteLength;
        this.#reportProgress();
      }
    } catch {
      offset = -1;                                   // a torn cache entry reads as a failure
    }
    if (offset === output.byteLength) return output.buffer;
    this.#loadedBytes = Math.max(0, this.#loadedBytes - Math.max(0, offset));
    this.#reportProgress();
    return undefined;
  }

  async #cacheMatch(url) {
    if (this.#shardCache === undefined) return undefined;
    try { return (await withTimeout(this.#shardCache.match(url), undefined)) ?? undefined; }
    catch { return undefined; }
  }

  async #cachePut(url, response) {
    try { await this.#shardCache?.put(url, response); } catch { /* quota, or no cache */ }
  }

  async #cacheDelete(url) {
    try { await this.#shardCache?.delete(url); } catch { /* nothing to drop */ }
  }
  #reportProgress(tensorName) {
    const progress = {
      loadedBytes: this.#loadedBytes, totalBytes: this.#totalBytes,
      loadedTensors: this.#loadedTensors, totalTensors: this.#totalTensors,
      ...(tensorName === undefined ? {} : { tensorName }),
    };
    this.#onProgress?.(progress);
  }
}
