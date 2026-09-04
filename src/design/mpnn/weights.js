// Reader for the `.mpnn` weight files produced by tools/convert_weights.py.

const MAGIC = 0x4e4e504d; // "MPNN" little-endian
const SUPPORTED_VERSION = 1;

/** Decode IEEE-754 half precision into a Float32Array. */
function decodeFloat16(buffer, byteOffset, count) {
  const src = new Uint16Array(buffer, byteOffset, count);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const h = src[i];
    const sign = (h & 0x8000) ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;
    if (exponent === 0) {
      out[i] = sign * mantissa * 5.960464477539063e-8; // 2^-24
    } else if (exponent === 31) {
      out[i] = mantissa ? NaN : sign * Infinity;
    } else {
      out[i] = sign * (mantissa + 1024) * Math.pow(2, exponent - 25);
    }
  }
  return out;
}

export class Weights {
  /**
   * @param {object} header  parsed JSON header
   * @param {Map<string, Float32Array>} tensors
   */
  constructor(header, tensors) {
    this.header = header;
    this.tensors = tensors;
    this.name = header.name;
    this.modelType = header.model_type;
    this.kNeighbors = header.k_neighbors;
    this.atomContextNum = header.atom_context_num;
    this.noiseLevel = header.noise_level;
    /** "float16" (what the page ships) or "float32" (what parity tests need). */
    this.dtype = header.dtype;
    this.hiddenDim = header.hidden_dim;
    this.numLetters = header.num_letters;
  }

  /** @returns {Float32Array} */
  get(name) {
    const t = this.tensors.get(name);
    if (t === undefined) throw new Error(`missing tensor ${name}`);
    return t;
  }

  has(name) {
    return this.tensors.has(name);
  }

  shape(name) {
    const meta = this.header.tensors[name];
    if (!meta) throw new Error(`missing tensor ${name}`);
    return meta.shape;
  }

  /** `{weight, bias}` for a torch.nn.Linear, with bias null when absent. */
  linear(prefix) {
    return {
      weight: this.get(`${prefix}.weight`),
      bias: this.has(`${prefix}.bias`) ? this.get(`${prefix}.bias`) : null,
      shape: this.shape(`${prefix}.weight`),
    };
  }

  /** `{gamma, beta}` for a torch.nn.LayerNorm. */
  norm(prefix) {
    return { gamma: this.get(`${prefix}.weight`), beta: this.get(`${prefix}.bias`) };
  }

  static fromArrayBuffer(buffer) {
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== MAGIC) throw new Error("not a .mpnn file");
    const version = view.getUint32(4, true);
    if (version !== SUPPORTED_VERSION) {
      throw new Error(`unsupported .mpnn version ${version}`);
    }
    const headerLen = view.getUint32(8, true);
    const headerBytes = new Uint8Array(buffer, 12, headerLen);
    const text = new TextDecoder().decode(headerBytes).replace(/\0+$/, "");
    const header = JSON.parse(text);

    const dataStart = 12 + headerLen;
    const tensors = new Map();
    for (const [name, meta] of Object.entries(header.tensors)) {
      const offset = dataStart + meta.offset;
      tensors.set(
        name,
        header.dtype === "float16"
          ? decodeFloat16(buffer, offset, meta.n)
          // The payload is 64-byte aligned but individual float32 tensors are
          // not guaranteed to land on a 4-byte boundary, so copy rather than
          // alias when the offset is unaligned.
          : offset % 4 === 0
            ? new Float32Array(buffer, offset, meta.n)
            : new Float32Array(buffer.slice(offset, offset + meta.n * 4)),
      );
    }
    return new Weights(header, tensors);
  }

  static async fetch(url, { signal, onProgress } = {}) {
    const response = await fetch(url, { signal });
    if (!response.ok) throw new Error(`${response.status} fetching ${url}`);

    const total = Number(response.headers.get("content-length")) || 0;
    if (!onProgress || !response.body) {
      return Weights.fromArrayBuffer(await response.arrayBuffer());
    }

    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received, total);
    }
    const merged = new Uint8Array(received);
    let at = 0;
    for (const chunk of chunks) {
      merged.set(chunk, at);
      at += chunk.length;
    }
    return Weights.fromArrayBuffer(merged.buffer);
  }
}
