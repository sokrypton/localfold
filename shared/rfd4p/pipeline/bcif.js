/**
 * BinaryCIF (`.bcif`), read into the same block `parseCifBlock` returns for an mmCIF: categories
 * by name without the leading `_`, columns by name, every value its TEXT - `.` and `?` where the
 * column's mask says so - so everything after the reader is the text path's.
 *
 * 🔴 UPSTREAM TAKES IT AND THIS PORT REFUSED IT. A condition spec's `input:` may be "CIF or
 * BinaryCIF (plain, gzip, or Zstandard)", and RCSB serves BinaryCIF. The format is MessagePack
 * around columns compressed by a stack of encodings (ByteArray, FixedPoint, IntervalQuantization,
 * RunLength, Delta, IntegerPacking, StringArray), each undone in reverse; the specification is
 * Sehnal et al., https://github.com/molstar/BinaryCIF. A number is written back as the shortest
 * text that reads as the same value, which is what the text reader then parses.
 */

/** Minimal MessagePack decoder: the types BinaryCIF uses. */
function unpack(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 0;
  const utf8 = new TextDecoder('utf-8');
  const str = (n) => { const s = utf8.decode(bytes.subarray(pos, pos + n)); pos += n; return s; };
  const bin = (n) => { const b = bytes.subarray(pos, pos + n); pos += n; return b; };
  const map = (n) => { const o = {}; for (let i = 0; i < n; i += 1) { const k = value(); o[k] = value(); } return o; };
  const arr = (n) => { const a = new Array(n); for (let i = 0; i < n; i += 1) a[i] = value(); return a; };
  function value() {
    const t = bytes[pos++];
    if (t < 0x80) return t;
    if (t < 0x90) return map(t & 0x0f);
    if (t < 0xa0) return arr(t & 0x0f);
    if (t < 0xc0) return str(t & 0x1f);
    if (t >= 0xe0) return t - 0x100;
    let v;
    switch (t) {
      case 0xc0: return null;
      case 0xc2: return false;
      case 0xc3: return true;
      case 0xc4: v = bytes[pos]; pos += 1; return bin(v);
      case 0xc5: v = view.getUint16(pos); pos += 2; return bin(v);
      case 0xc6: v = view.getUint32(pos); pos += 4; return bin(v);
      case 0xca: v = view.getFloat32(pos); pos += 4; return v;
      case 0xcb: v = view.getFloat64(pos); pos += 8; return v;
      case 0xcc: v = bytes[pos]; pos += 1; return v;
      case 0xcd: v = view.getUint16(pos); pos += 2; return v;
      case 0xce: v = view.getUint32(pos); pos += 4; return v;
      case 0xcf: v = Number(view.getBigUint64(pos)); pos += 8; return v;
      case 0xd0: v = view.getInt8(pos); pos += 1; return v;
      case 0xd1: v = view.getInt16(pos); pos += 2; return v;
      case 0xd2: v = view.getInt32(pos); pos += 4; return v;
      case 0xd3: v = Number(view.getBigInt64(pos)); pos += 8; return v;
      case 0xd9: v = bytes[pos]; pos += 1; return str(v);
      case 0xda: v = view.getUint16(pos); pos += 2; return str(v);
      case 0xdb: v = view.getUint32(pos); pos += 4; return str(v);
      case 0xdc: v = view.getUint16(pos); pos += 2; return arr(v);
      case 0xdd: v = view.getUint32(pos); pos += 4; return arr(v);
      case 0xde: v = view.getUint16(pos); pos += 2; return map(v);
      case 0xdf: v = view.getUint32(pos); pos += 4; return map(v);
      default: throw new Error(`BinaryCIF: unsupported MessagePack type 0x${t.toString(16)}`);
    }
  }
  return value();
}

/** Whether these bytes are BinaryCIF: a MessagePack map, where an mmCIF starts with text. */
export function isBinaryCif(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length === 0) return false;
  const t = bytes[0];
  return (t >= 0x80 && t <= 0x8f) || t === 0xde || t === 0xdf;
}

const TYPED = {
  1: [Int8Array, 1], 2: [Int16Array, 2], 3: [Int32Array, 4], 4: [Uint8Array, 1], 5: [Uint16Array, 2],
  6: [Uint32Array, 4], 32: [Float32Array, 4], 33: [Float64Array, 8],
};

function byteArray(data, type) {
  const spec = TYPED[type];
  if (!spec) throw new Error(`BinaryCIF: unknown ByteArray type ${type}`);
  const [Ctor, size] = spec;
  const n = data.byteLength / size;
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const out = new Ctor(n);
  for (let i = 0; i < n; i += 1) {
    const at = i * size;
    switch (type) {
      case 1: out[i] = view.getInt8(at); break;
      case 2: out[i] = view.getInt16(at, true); break;
      case 3: out[i] = view.getInt32(at, true); break;
      case 4: out[i] = view.getUint8(at); break;
      case 5: out[i] = view.getUint16(at, true); break;
      case 6: out[i] = view.getUint32(at, true); break;
      case 32: out[i] = view.getFloat32(at, true); break;
      default: out[i] = view.getFloat64(at, true);
    }
  }
  return out;
}

/** Undo a column's encodings, last applied first. Numbers come back as numbers, strings as text. */
function decode(data, encodings) {
  let v = data;
  for (let k = encodings.length - 1; k >= 0; k -= 1) {
    const e = encodings[k];
    switch (e.kind) {
      case 'ByteArray': v = byteArray(v, e.type); break;
      case 'FixedPoint': v = Array.from(v, (x) => x / e.factor); break;
      case 'IntervalQuantization': {
        const step = (e.max - e.min) / (e.numSteps - 1);
        v = Array.from(v, (x) => e.min + step * x);
        break;
      }
      case 'RunLength': {
        const out = [];
        for (let i = 0; i < v.length; i += 2) for (let r = 0; r < v[i + 1]; r += 1) out.push(v[i]);
        v = out;
        break;
      }
      case 'Delta': {
        const out = new Array(v.length);
        let acc = e.origin;
        for (let i = 0; i < v.length; i += 1) { acc += v[i]; out[i] = acc; }
        v = out;
        break;
      }
      case 'IntegerPacking': {
        // A value at the packed type's limit continues into the next one - the upper limit, and
        // for a signed packing the lower too (an unsigned one has no negative to continue).
        const upper = e.isUnsigned ? (e.byteCount === 1 ? 0xff : 0xffff) : (e.byteCount === 1 ? 0x7f : 0x7fff);
        const lower = e.isUnsigned ? null : -upper - 1;
        const out = [];
        let acc = 0;
        for (let i = 0; i < v.length; i += 1) {
          const x = v[i];
          acc += x;
          if (x !== upper && x !== lower) { out.push(acc); acc = 0; }
        }
        v = out;
        break;
      }
      case 'StringArray': {
        const offsets = decode(e.offsets, e.offsetEncoding);
        const indices = decode(v, e.dataEncoding);
        const strings = [];
        for (let i = 0; i + 1 < offsets.length; i += 1) strings.push(e.stringData.slice(offsets[i], offsets[i + 1]));
        v = Array.from(indices, (i) => (i < 0 ? '' : strings[i]));
        break;
      }
      default: throw new Error(`BinaryCIF: unknown encoding ${e.kind}`);
    }
  }
  return v;
}

/**
 * A BinaryCIF file's first data block, as `parseCifBlock` shapes an mmCIF's.
 * @param {Uint8Array} bytes
 */
export function parseBinaryCifBlock(bytes) {
  const file = unpack(bytes);
  const block = file?.dataBlocks?.[0];
  if (!block) throw new Error('BinaryCIF holds no data block');
  const out = {};
  for (const category of block.categories) {
    const name = category.name.startsWith('_') ? category.name.slice(1) : category.name;
    const cat = {};
    for (const column of category.columns) {
      const values = decode(column.data.data, column.data.encoding);
      const mask = column.mask ? decode(column.mask.data, column.mask.encoding) : null;
      cat[column.name] = Array.from(values, (x, i) => {
        if (mask && mask[i] === 1) return '.';
        if (mask && mask[i] === 2) return '?';
        return typeof x === 'string' ? x : String(x);
      });
    }
    out[name] = cat;
  }
  return out;
}
