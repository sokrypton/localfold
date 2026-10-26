/**
 * gzip (RFC 1952) around DEFLATE (RFC 1951), decompressed synchronously.
 *
 * Upstream reads a condition spec's `input:` as plain, gzip or Zstandard, and RCSB serves
 * `.cif.gz`. The page's structure path is synchronous end to end, which rules out the browser's
 * `DecompressionStream`, so this is the format's own decoder: stored, fixed-Huffman and
 * dynamic-Huffman blocks, and every member of a multi-member file. tools/check-rfd4p-load.py loads
 * gzipped inputs through it against atomworks.
 */

const LENGTH_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
  131, 163, 195, 227, 258];
const LENGTH_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
  2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CODE_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/** A canonical Huffman code from its code lengths: counts per length and symbols in code order. */
function huffman(lengths) {
  const counts = new Uint16Array(16);
  for (const l of lengths) counts[l] += 1;
  counts[0] = 0;
  const offsets = new Uint16Array(16);
  for (let l = 1; l < 16; l += 1) offsets[l] = offsets[l - 1] + counts[l - 1];
  const symbols = new Uint16Array(lengths.length);
  lengths.forEach((l, s) => { if (l) symbols[offsets[l]++] = s; });
  return { counts, symbols };
}

const FIXED_LIT = huffman([...Array(144).fill(8), ...Array(112).fill(9), ...Array(24).fill(7), ...Array(8).fill(8)]);
const FIXED_DIST = huffman(Array(30).fill(5));

class Bits {
  constructor(bytes, pos) { this.bytes = bytes; this.pos = pos; this.bit = 0; }
  read(n) {
    let v = 0;
    for (let i = 0; i < n; i += 1) {
      if (this.pos >= this.bytes.length) throw new Error('gzip: truncated stream');
      v |= ((this.bytes[this.pos] >> this.bit) & 1) << i;
      if (++this.bit === 8) { this.bit = 0; this.pos += 1; }
    }
    return v;
  }
  symbol({ counts, symbols }) {
    let code = 0;
    let first = 0;
    let index = 0;
    for (let len = 1; len < 16; len += 1) {
      code |= this.read(1);
      const count = counts[len];
      if (code - first < count) return symbols[index + code - first];
      index += count;
      first = (first + count) << 1;
      code <<= 1;
    }
    throw new Error('gzip: bad Huffman code');
  }
  align() { if (this.bit) { this.bit = 0; this.pos += 1; } }
}

class Output {
  constructor() { this.buf = new Uint8Array(1 << 16); this.n = 0; }
  push(b) {
    if (this.n === this.buf.length) { const g = new Uint8Array(this.buf.length * 2); g.set(this.buf); this.buf = g; }
    this.buf[this.n++] = b;
  }
}

/** One raw DEFLATE stream from `bits`, appended to `out`. */
function inflateInto(bits, out) {
  for (;;) {
    const last = bits.read(1);
    const type = bits.read(2);
    if (type === 0) {
      bits.align();
      const { bytes } = bits;
      const len = bytes[bits.pos] | (bytes[bits.pos + 1] << 8);
      bits.pos += 4;
      if (bits.pos + len > bytes.length) throw new Error('gzip: truncated stored block');
      for (let i = 0; i < len; i += 1) out.push(bytes[bits.pos + i]);
      bits.pos += len;
    } else if (type === 1 || type === 2) {
      let lit = FIXED_LIT;
      let dist = FIXED_DIST;
      if (type === 2) {
        const hlit = bits.read(5) + 257;
        const hdist = bits.read(5) + 1;
        const hclen = bits.read(4) + 4;
        const codeLengths = new Array(19).fill(0);
        for (let i = 0; i < hclen; i += 1) codeLengths[CODE_ORDER[i]] = bits.read(3);
        const code = huffman(codeLengths);
        const lengths = [];
        while (lengths.length < hlit + hdist) {
          const s = bits.symbol(code);
          if (s < 16) lengths.push(s);
          else if (s === 16) { const prev = lengths.at(-1); for (let r = bits.read(2) + 3; r > 0; r -= 1) lengths.push(prev); }
          else if (s === 17) { for (let r = bits.read(3) + 3; r > 0; r -= 1) lengths.push(0); }
          else { for (let r = bits.read(7) + 11; r > 0; r -= 1) lengths.push(0); }
        }
        lit = huffman(lengths.slice(0, hlit));
        dist = huffman(lengths.slice(hlit));
      }
      for (;;) {
        const s = bits.symbol(lit);
        if (s < 256) { out.push(s); continue; }
        if (s === 256) break;
        const li = s - 257;
        const len = LENGTH_BASE[li] + bits.read(LENGTH_EXTRA[li]);
        const di = bits.symbol(dist);
        const d = DIST_BASE[di] + bits.read(DIST_EXTRA[di]);
        if (d > out.n) throw new Error('gzip: distance past the start');
        for (let i = 0; i < len; i += 1) out.push(out.buf[out.n - d]);
      }
    } else {
      throw new Error('gzip: invalid block type');
    }
    if (last) return;
  }
}

export function isGzipBytes(bytes) {
  return bytes instanceof Uint8Array && bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

/** A gzip file's contents, every member concatenated. */
export function gunzip(bytes) {
  const out = new Output();
  let pos = 0;
  while (pos < bytes.length && bytes[pos] === 0x1f && bytes[pos + 1] === 0x8b) {
    if (bytes[pos + 2] !== 8) throw new Error('gzip: not DEFLATE');
    const flags = bytes[pos + 3];
    pos += 10;
    if (flags & 4) pos += 2 + (bytes[pos] | (bytes[pos + 1] << 8));
    if (flags & 8) { while (bytes[pos] !== 0) pos += 1; pos += 1; }
    if (flags & 16) { while (bytes[pos] !== 0) pos += 1; pos += 1; }
    if (flags & 2) pos += 2;
    const bits = new Bits(bytes, pos);
    inflateInto(bits, out);
    bits.align();
    pos = bits.pos + 8;   // CRC32 and size
  }
  return out.buf.subarray(0, out.n);
}
