import { readTensor, tensorByteLength } from "./dtype.js";
/**
 * @typedef {import("./http-tensor-store.js").TensorDownloadProgress} TensorDownloadProgress
 * @typedef {import("./http-tensor-store.js").TensorDownloadProgressCallback} TensorDownloadProgressCallback
 */

const MAX_CONCURRENT_LOADS = 2;

/**
 * The same tensor table as HttpTensorStore, delivered by classic scripts.
 *
 * WHY THIS EXISTS. A file:// page has exactly two doors to bytes on disk, and
 * neither is fetch(): a classic <script src>, which was never subject to the
 * same-origin read rule, and a data: URL, which carries its own bytes instead
 * of asking an origin for them. tools/export-js-weights.py writes each shard as
 * a script that assigns a base64 data: URL onto window.__afWeights; this walks
 * through both doors in order - inject the script, then fetch its data: URL,
 * which hands the base64 decode to the browser's C++ rather than doing it in a
 * JS loop.
 *
 * ONLY FOR file://. Over http the .bin shards are a third smaller and
 * HttpTensorStore reads them directly, so nothing here should ever run there.
 * The surface matches the other two stores because AlphaFoldFixture.fromStore
 * takes any of them and must not care which.
 *
 * MEMORY. The base64 string for a 48 MiB shard is 64 MiB, and it is dropped
 * from window.__afWeights.shards the moment it decodes. Holding all eight would
 * cost half a gigabyte for nothing - they are never needed twice, because the
 * decoded ArrayBuffer is what gets cached.
 */
export class ScriptTensorStore {
  baseUrl;
  manifest;
  #scripts;
  #cache = new Map();
  #fileCache = new Map();
  #pending = [];
  #activeLoads = 0;
  #onProgress;
  #totalBytes;
  #totalTensors;
  #loadedBytes = 0;
  #loadedTensors = 0;

  constructor(baseUrl, manifest, scripts, onProgress) {
    this.baseUrl = baseUrl;
    this.manifest = manifest;
    this.#scripts = scripts;
    this.#onProgress = onProgress;
    const records = Object.values(manifest.tensors);
    this.#totalTensors = records.length;
    this.#totalBytes = records.reduce(
      (sum, record) => sum + tensorByteLength(record), 0,
    );
  }

  /**
   * @param {string} baseValue  the model directory, e.g. "./model/"
   * @param {TensorDownloadProgressCallback} [onProgress]
   */
  static async open(baseValue, onProgress) {
    const baseUrl = baseValue.endsWith("/") ? baseValue : `${baseValue}/`;
    await loadScript(`${baseUrl}manifest.js`);
    const globals = globalThis.__afWeights;
    if (globals?.manifest === undefined) {
      throw new Error(`${baseUrl}manifest.js did not define a model manifest`);
    }
    if (globals.manifest.tensors === undefined) throw new Error("model manifest has no tensor table");
    const store = new ScriptTensorStore(baseUrl, globals.manifest, globals.scripts ?? {}, onProgress);
    store.#reportProgress();
    return store;
  }

  tensor(name) {
    let value = this.#cache.get(name);
    if (value === undefined) { value = this.#load(name); this.#cache.set(name, value); }
    return value;
  }

  shape(name) {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`missing tensor ${name}`);
    return record.shape;
  }

  async #load(name) {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`missing tensor ${name}`);
    let pendingFile = this.#fileCache.get(record.file);
    if (pendingFile === undefined) {
      pendingFile = this.#schedule(record.file);
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

  /**
   * Two shards in flight, not eight.
   *
   * Each one is a 64 MiB string that only stops existing once it has decoded,
   * so the width of this queue IS the peak memory. Eight at once briefly costs
   * half a gigabyte of base64 on top of the weights themselves.
   */
  #schedule(file) {
    return new Promise((resolve, reject) => {
      const start = () => {
        this.#activeLoads += 1;
        void this.#loadShard(file).then(resolve, reject).finally(() => {
          this.#activeLoads -= 1;
          this.#pending.shift()?.();
        });
      };
      if (this.#activeLoads < MAX_CONCURRENT_LOADS) start();
      else this.#pending.push(start);
    });
  }

  async #loadShard(file) {
    const script = this.#scripts[file] ?? `${file.split(".")[0]}.js`;
    await loadScript(`${this.baseUrl}${script}`);
    const shards = globalThis.__afWeights?.shards ?? {};
    const dataUrl = shards[file];
    if (typeof dataUrl !== "string") throw new Error(`${script} did not define ${file}`);
    // ...the browser's base64 decoder, not a JS loop, and the string is dropped
    // before the buffer is returned so the two never coexist for long.
    const response = await fetch(dataUrl);
    const buffer = await response.arrayBuffer();
    delete shards[file];
    this.#loadedBytes += buffer.byteLength;
    this.#reportProgress();
    return buffer;
  }

  #reportProgress(tensorName) {
    this.#onProgress?.({
      loadedBytes: this.#loadedBytes, totalBytes: this.#totalBytes,
      loadedTensors: this.#loadedTensors, totalTensors: this.#totalTensors,
      ...(tensorName === undefined ? {} : { tensorName }),
    });
  }
}

/**
 * Inject one classic script and wait for it.
 *
 * `onerror` on a file:// script tag reports no reason - the event is empty by
 * design, so a missing file and a syntax error look identical. Saying which
 * file failed is the most this can do, and it is worth doing: without it the
 * page reports "undefined" and the user has nothing to go on.
 */
function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.async = false;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`could not load ${url}`));
    document.head.append(script);
  });
}
