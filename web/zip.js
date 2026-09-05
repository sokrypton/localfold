/**
 * A ZIP writer and reader, in one file so the two cannot drift.
 *
 * WHY THIS EXISTS. "Download All" writes an archive shaped like the AlphaFold 3
 * server's, and the upload box reads that same archive back. LocalFold deploys
 * as static files to Pages and has no bundler, so a dependency is not free -
 * and `src/input/mmseqs2-api.js` already sets the precedent, untarring the
 * MMseqs2 result by hand. A ZIP with no encryption, no spanning and no ZIP64 is
 * about a hundred lines.
 *
 * 🔴 THE WRITER AND THE READER LIVE TOGETHER ON PURPOSE. They are the two
 * halves of one format, and the only test that means anything is that what one
 * writes the other reads. Split across two files they drift; the failure is a
 * round trip that silently loses a member, which reads as a fold whose
 * alignment came back shorter.
 *
 * 🔴 AND EVERY MULTI-BYTE FIELD IS LITTLE-ENDIAN. Not a convention this file
 * chose - the format is little-endian throughout, on every platform, and a
 * DataView written with the default big-endian flag produces an archive that
 * every unzip refuses with a message about a corrupt header.
 */

const SIGNATURE = { local: 0x04034b50, central: 0x02014b50, end: 0x06054b50 };
const STORED = 0;
const DEFLATED = 8;

/**
 * 🔴 BUILT ONCE, ON FIRST USE, NOT AT MODULE LOAD. This module is imported by
 * the page whether or not anything is ever downloaded, and 256 iterations of
 * shifting is not free on a phone at start-up. It is also not worth a comment
 * anywhere else: the polynomial is the standard reversed CRC-32, 0xEDB88320.
 */
let table;
function crc32(bytes) {
  if (table === undefined) {
    table = new Uint32Array(256);
    for (let index = 0; index < 256; index += 1) {
      let value = index;
      for (let bit = 0; bit < 8; bit += 1) {
        value = value & 1 ? (value >>> 1) ^ 0xEDB88320 : value >>> 1;
      }
      table[index] = value;
    }
  }
  let crc = 0xFFFFFFFF;
  for (let index = 0; index < bytes.length; index += 1) {
    crc = (crc >>> 8) ^ table[(crc ^ bytes[index]) & 0xFF];
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/**
 * Deflate, or nothing when this runtime cannot.
 *
 * 🔴 A FALLBACK, NOT A FAILURE. `CompressionStream` is missing in older
 * browsers and behind a flag in some, and an archive that cannot be written at
 * all is a worse outcome than one that is larger than it needs to be. Stored
 * entries are legal ZIP and every reader takes them. It matters for size
 * though, not for correctness: the a3m files are the bulk of a fold archive and
 * they are text with long runs of gaps, so deflate takes roughly a fifth.
 */
async function deflate(bytes) {
  if (typeof CompressionStream === "undefined") return null;
  const stream = new Blob([bytes]).stream()
    .pipeThrough(new CompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function inflate(bytes) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("this browser cannot read a compressed archive");
  }
  const stream = new Blob([bytes]).stream()
    .pipeThrough(new DecompressionStream("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * One archive from named members.
 *
 * @param {Map<string, string|Uint8Array>|Array<[string, string|Uint8Array]>} entries
 * @param {{store?: boolean}} [options] `store` forces no compression, which is
 *   how the stored path is tested rather than only reasoned about.
 * @returns {Promise<Uint8Array>}
 */
export async function writeZip(entries, options = {}) {
  const members = entries instanceof Map ? [...entries] : [...entries];
  const locals = [];
  const central = [];
  let offset = 0;

  for (const [name, content] of members) {
    const nameBytes = encoder.encode(name);
    const body = typeof content === "string" ? encoder.encode(content) : content;
    const packed = options.store === true ? null : await deflate(body);
    // ...and only when it actually helped. Deflate can be LARGER than the input
    // for short or already-compressed members, and paying bytes to compress is
    // the one outcome nobody wants.
    const useDeflate = packed !== null && packed.length < body.length;
    const stored = useDeflate ? packed : body;
    const method = useDeflate ? DEFLATED : STORED;
    const crc = crc32(body);

    const header = new Uint8Array(30 + nameBytes.length);
    const view = new DataView(header.buffer);
    view.setUint32(0, SIGNATURE.local, true);
    view.setUint16(4, 20, true);          // version needed: 2.0, deflate
    view.setUint16(6, 0, true);           // no flags; sizes are known up front
    view.setUint16(8, method, true);
    view.setUint16(10, 0, true);          // modification time
    view.setUint16(12, 0x21, true);       // ...and date: 1980-01-01, see below
    view.setUint32(14, crc, true);
    view.setUint32(18, stored.length, true);
    view.setUint32(22, body.length, true);
    view.setUint16(26, nameBytes.length, true);
    view.setUint16(28, 0, true);          // no extra field
    header.set(nameBytes, 30);

    locals.push(header, stored);

    const record = new Uint8Array(46 + nameBytes.length);
    const recordView = new DataView(record.buffer);
    recordView.setUint32(0, SIGNATURE.central, true);
    recordView.setUint16(4, 20, true);    // version made by
    recordView.setUint16(6, 20, true);    // version needed
    recordView.setUint16(8, 0, true);
    recordView.setUint16(10, method, true);
    recordView.setUint16(12, 0, true);
    recordView.setUint16(14, 0x21, true);
    recordView.setUint32(16, crc, true);
    recordView.setUint32(20, stored.length, true);
    recordView.setUint32(24, body.length, true);
    recordView.setUint16(28, nameBytes.length, true);
    recordView.setUint16(30, 0, true);    // extra
    recordView.setUint16(32, 0, true);    // comment
    recordView.setUint16(34, 0, true);    // disk number
    recordView.setUint16(36, 0, true);    // internal attributes
    recordView.setUint32(38, 0, true);    // external attributes
    recordView.setUint32(42, offset, true);
    record.set(nameBytes, 46);
    central.push(record);

    offset += header.length + stored.length;
  }

  const centralSize = central.reduce((total, record) => total + record.length, 0);
  const end = new Uint8Array(22);
  const endView = new DataView(end.buffer);
  endView.setUint32(0, SIGNATURE.end, true);
  endView.setUint16(4, 0, true);
  endView.setUint16(6, 0, true);
  endView.setUint16(8, members.length, true);
  endView.setUint16(10, members.length, true);
  endView.setUint32(12, centralSize, true);
  endView.setUint32(16, offset, true);
  endView.setUint16(20, 0, true);

  const total = offset + centralSize + end.length;
  const archive = new Uint8Array(total);
  let at = 0;
  for (const part of [...locals, ...central, end]) {
    archive.set(part, at);
    at += part.length;
  }
  return archive;
}

/** Whether these bytes begin an archive, which is how a file is told apart. */
export function looksLikeZip(bytes) {
  return bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4B
    && (bytes[2] === 0x03 || bytes[2] === 0x05) && (bytes[3] === 0x04 || bytes[3] === 0x06);
}

/**
 * Every member of an archive, as text.
 *
 * 🔴 THE CENTRAL DIRECTORY IS THE INDEX, NOT THE RUN OF LOCAL HEADERS. Walking
 * local headers front to back looks equivalent and is not: an archive written
 * with data descriptors carries zero in the local header's size fields, so the
 * walk cannot know where the next header begins and reads whatever follows as
 * one. The directory at the end is the part that is always complete.
 *
 * @param {Uint8Array} bytes
 * @returns {Promise<Map<string, string>>}
 */
export async function readZip(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // ...scanned backwards, because the record ends with a comment of any length.
  let end = -1;
  for (let at = bytes.length - 22; at >= 0; at -= 1) {
    if (view.getUint32(at, true) === SIGNATURE.end) { end = at; break; }
  }
  if (end < 0) throw new Error("that file is not a zip archive");

  const count = view.getUint16(end + 10, true);
  let at = view.getUint32(end + 16, true);
  const files = new Map();
  for (let index = 0; index < count; index += 1) {
    if (view.getUint32(at, true) !== SIGNATURE.central) {
      throw new Error("this archive's directory is damaged");
    }
    const method = view.getUint16(at + 10, true);
    const compressedSize = view.getUint32(at + 20, true);
    const nameLength = view.getUint16(at + 28, true);
    const extraLength = view.getUint16(at + 30, true);
    const commentLength = view.getUint16(at + 32, true);
    const localAt = view.getUint32(at + 42, true);
    const name = decoder.decode(bytes.subarray(at + 46, at + 46 + nameLength));

    // 🔴 THE LOCAL HEADER'S OWN LENGTHS, NOT THE DIRECTORY'S. The two name and
    // extra fields are allowed to differ - writers routinely put an extended
    // timestamp in one and not the other - so computing the data's offset from
    // the directory's `extraLength` lands a few bytes into the deflate stream,
    // which fails as a corrupt archive rather than as a wrong offset.
    const localNameLength = view.getUint16(localAt + 26, true);
    const localExtraLength = view.getUint16(localAt + 28, true);
    const dataAt = localAt + 30 + localNameLength + localExtraLength;
    const raw = bytes.subarray(dataAt, dataAt + compressedSize);
    if (method !== STORED && method !== DEFLATED) {
      throw new Error(`${name} uses compression this cannot read`);
    }
    files.set(name, decoder.decode(method === DEFLATED ? await inflate(raw) : raw));
    at += 46 + nameLength + extraLength + commentLength;
  }
  return files;
}
