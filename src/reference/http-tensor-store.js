import { readTensor, readTensorRange, tensorByteLength } from "./dtype.js";
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

/**
 * The cached shard sets this manifest supersedes: OTHER EXPORTS OF THE SAME
 * MODEL, and nothing else.
 *
 * 🔴 IT USED TO SWEEP EVERY CACHE, which was right when there was one model and
 * wrong the moment there were two. A cache whose name did not match could only
 * mean a stale export of the one model; now it also means THE OTHER MODEL, and
 * the sweep evicted it. Loading multimer deleted the monomer's 97 MiB and
 * loading the monomer deleted multimer's, so every switch between families paid
 * a full cold download - measured at ~5 s each on a local server, and far worse
 * over a network.
 *
 * The model name is part of the cache name, so keeping the two apart is a
 * matter of only sweeping within one. The trailing "-" matters: without it
 * `model_1` would claim `model_1_multimer_v3`'s caches.
 *
 * @param {readonly string[]} names existing cache names
 * @param {{bundle?: {model?: string}, tensors: object}} manifest
 */
export function staleShardCaches(names, manifest) {
  const keep = CACHE_PREFIX + cacheToken(manifest);
  const mine = `${CACHE_PREFIX}${manifest.bundle?.model ?? "model"}-`;
  return names.filter((existing) => existing.startsWith(mine) && existing !== keep);
}

async function openShardCache(manifest) {
  if (typeof caches === "undefined") return undefined;
  const name = CACHE_PREFIX + cacheToken(manifest);
  try {
    return await withTimeout((async () => {
      for (const stale of staleShardCaches(await caches.keys(), manifest)) {
        await caches.delete(stale);
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
  #fileBuffers = new Map();
  #openedTensors = new Set();
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
   * 🔴 SHARD URLS CARRY THE EXPORT'S IDENTITY. A manifest is small enough to
   * fetch fresh every load; a shard is not, and lives in the browser's cache
   * for as long as its headers allow. Re-export the model and the two disagree
   * - a fresh manifest against a cached shard - which surfaces as
   * "<file> has an invalid byte length", naming neither half. Three separate
   * hours have gone into that message.
   *
   * The token is the same one the shard Cache is keyed on, so the URL changes
   * exactly when the export changes: still cached across loads, never stale
   * across exports. Pass "" to address the bare filenames.
   *
   * @param {string} [shardQuery]
   */
  static async fromManifest(manifestUrlValue, manifest, onProgress = undefined,
    shardQuery = undefined) {
    const manifestUrl = manifestUrlValue instanceof URL
      ? manifestUrlValue
      : new URL(manifestUrlValue, typeof location !== "undefined" ? location.href : "http://localhost/");
    if (manifest.tensors === undefined) throw new Error("model manifest has no tensor table");
    const store = new HttpTensorStore(manifestUrl, manifest, onProgress,
      await openShardCache(manifest), shardQuery ?? `?v=${cacheToken(manifest)}`);
    store.#reportProgress();
    return store;
  }
  tensor(name) {
    let value = this.#cache.get(name);
    if (value === undefined) { value = this.#load(name); this.#cache.set(name, value); }
    return value;
  }

  /**
   * Start every shard downloading now, instead of when a tensor in it is first
   * asked for.
   *
   * 🔴 WITHOUT THIS THE NETWORK RUNS AT 0.68 OF ONE CONNECTION. A shard is only
   * requested when a loader reaches a tensor inside it, and the loaders await
   * tensor by tensor - so the shape of a cold load is: fetch a shard, sit on the
   * network while it is dequantised, fetch the next. Measured on localfold.org
   * over 26 shards: a 13.4 s span containing 9.1 s of transfer, with a gap
   * between every shard's end and the next one's start, and an effective
   * 6.8 MB/s against the 12 MB/s a single shard was actually moving at. The
   * eight-way concurrency limit below was never once reached, because nothing
   * ever asked for eight.
   *
   * Scheduling them all up front costs no extra memory - #fileCache already
   * holds every shard for the life of the store, so this only changes WHEN they
   * arrive - and the queue in #scheduleDownload still keeps the number in
   * flight at MAX_CONCURRENT_DOWNLOADS.
   *
   * 🔴 IT IS OPT-IN BECAUSE NOT EVERY CALLER READS EVERY SHARD. A bench that
   * loads four pairformer blocks touches a fraction of the manifest, and
   * prefetching the rest would make it slower and noisier, not faster. The page
   * loads whole models and calls this; the tools do not.
   */
  prefetch() {
    // 🔴 BIGGEST FIRST, BECAUSE THE LAST SHARD TO START DECIDES WHEN THE LOAD
    // ENDS. The shards are not evenly sized and cannot be: a shard is at least
    // one whole tensor, and AF3's stacked single-transition weights are 40.5 MiB
    // against a 7.9 MiB median. Started last, that one runs on alone after the
    // other seven connections have nothing left to do; started first, the small
    // ones fill in around it. Longest-processing-time-first, which is the
    // standard answer for a makespan and costs three lines.
    //
    // 🔴 IT MEASURES AS NOTHING ON A SLOW LINK, AND THAT IS NOT AN ARGUMENT
    // AGAINST IT. Against manifest order over the real 26 shards: 3.30 s and
    // 3.27 against 3.44 and 3.28, which is noise. A cold load here is bytes
    // over bandwidth - 265 MB at 9.4 MB/s is 28 s - and the tail is 41 MiB at
    // 9 MB/s, under five. The tail only binds when the link is fast enough for
    // the first term to fall below the second, which is exactly the case this
    // ordering protects and the case that cannot be measured from here.
    const order = [...new Set(Object.values(this.manifest.tensors).map((record) => record.file))]
      .sort((left, right) =>
        (this.#fileByteLengths.get(right) ?? 0) - (this.#fileByteLengths.get(left) ?? 0));
    const firstTensorIn = new Map();
    for (const [name, record] of Object.entries(this.manifest.tensors)) {
      if (!firstTensorIn.has(record.file)) firstTensorIn.set(record.file, name);
    }
    for (const file of order) {
      if (this.#fileCache.has(file)) continue;
      this.#fileCache.set(file, this.#scheduleDownload(file, firstTensorIn.get(file)));
    }
  }
  shape(name) {
    const record = this.manifest.tensors[name]; if (record === undefined) throw new Error(`missing tensor ${name}`);
    return record.shape;
  }

  /**
   * Bring a tensor's SHARD in, without decoding the tensor.
   *
   * 🔴 THIS IS THE HALF OF LOADING THAT HAS TO BE AWAITED, and it is the only
   * half that has to happen up front. Decoding is arithmetic over bytes already
   * in memory, so once the shard is here a caller can decode whatever part of
   * it, whenever, without another await - which is what lets a stacked tensor be
   * read one block at a time instead of all at once. See src/af3/weights.js.
   *
   * It counts towards the progress callback exactly as a whole load does, so a
   * page loading lazily still fills its bar.
   */
  async open(name) {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`missing tensor ${name}`);
    if (!this.#fileBuffers.has(record.file)) {
      let pendingFile = this.#fileCache.get(record.file);
      if (pendingFile === undefined) {
        pendingFile = this.#scheduleDownload(record.file, name);
        this.#fileCache.set(record.file, pendingFile);
      }
      this.#fileBuffers.set(record.file, await pendingFile);
    }
    // ...counted per TENSOR, not per shard, so a lazily loaded model fills the
    // progress bar the same way an eagerly loaded one does. The set is what
    // keeps a second open of the same name from counting twice, which the
    // per-name cache does for tensor().
    if (this.#openedTensors.has(name)) return;
    this.#openedTensors.add(name);
    this.#loadedTensors += 1;
    this.#reportProgress(name);
  }

  /**
   * Part of a tensor, decoded now and cached NOWHERE.
   *
   * 🔴 THE CALLER OWNS WHAT COMES BACK, and that is the point: `tensor()` keeps
   * every array it ever decoded for the life of the store, which is right for a
   * tensor read once and wrong for 48 blocks of a stacked one. Await open(name)
   * first; this is synchronous so it can sit behind a property getter.
   */
  /**
   * Where a tensor's bytes are, without decoding any of them.
   *
   * 🔴 THIS IS WHAT LETS A DECODER RUN SOMEWHERE ELSE. `tensorRangeSync` hands
   * back float32, which for an int5 tensor means the host has already done the
   * work; a caller that wants to decode on the GPU needs the CODES, and the
   * scale and zero tables beside them. It returns a view, not a copy - the
   * shard stays owned by this store.
   */
  tensorSource(name) {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`missing tensor ${name}`);
    const buffer = this.#fileBuffers.get(record.file);
    if (buffer === undefined) {
      throw new Error(`${name} is in ${record.file}, which is not open yet - await store.open(name)`);
    }
    return { record, buffer, byteOffset: record.byteOffset ?? 0 };
  }

  tensorRangeSync(name, first, count) {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`missing tensor ${name}`);
    const buffer = this.#fileBuffers.get(record.file);
    if (buffer === undefined) {
      throw new Error(`${name} is in ${record.file}, which is not open yet - await store.open(name)`);
    }
    const byteOffset = record.byteOffset ?? 0;
    const byteLength = tensorByteLength(record);
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
      throw new Error(`${name} points outside ${record.file}`);
    }
    return readTensorRange(record, buffer, byteOffset, first, count);
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
    this.#fileBuffers.set(record.file, buffer);
    const byteOffset = record.byteOffset ?? 0;
    const byteLength = tensorByteLength(record);
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
      throw new Error(`${name} points outside ${record.file}`);
    }
    if (!this.#openedTensors.has(name)) {
      this.#openedTensors.add(name);
      this.#loadedTensors += 1;
    }
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
