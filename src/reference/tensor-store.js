import { readTensor, tensorByteLength } from "./dtype.js";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

export class FileTensorStore {
  manifestPath;
  manifest;
  #directory;
  #cache = new Map();
  #fileCache = new Map();

  constructor(manifestPath, manifest) {
    this.manifestPath = manifestPath;
    this.manifest = manifest;
    this.#directory = dirname(manifestPath);
  }

  static async open(manifestPath) {
    const absolutePath = resolve(manifestPath);
    const manifest = JSON.parse(await readFile(absolutePath, "utf8"));
    if (manifest.tensors === undefined) throw new Error(`${absolutePath} has no tensor table`);
    return new FileTensorStore(absolutePath, manifest);
  }

  tensor(name) {
    let pending = this.#cache.get(name);
    if (pending === undefined) {
      pending = this.#load(name);
      this.#cache.set(name, pending);
    }
    return pending;
  }

  shape(name) {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`manifest contains no tensor named ${name}`);
    return record.shape;
  }

  async #load(name) {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`manifest contains no tensor named ${name}`);
    let pendingFile = this.#fileCache.get(record.file);
    if (pendingFile === undefined) {
      pendingFile = readFile(resolve(this.#directory, record.file));
      this.#fileCache.set(record.file, pendingFile);
    }
    const bytes = await pendingFile;
    const byteOffset = record.byteOffset ?? 0;
    const byteLength = tensorByteLength(record);
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > bytes.byteLength) {
      throw new Error(`${name} points outside ${record.file}`);
    }
    // ...COPIED, because this hands back a view on a Buffer that node may be
    // sharing with other reads of the same shard.
    return readTensor(record, bytes.buffer, bytes.byteOffset + byteOffset, true);
  }
}
