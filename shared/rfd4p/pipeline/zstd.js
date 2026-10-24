// Zstandard helpers for `.cif.zst` inputs, on top of the vendored fzstd decoder.
import { decompress, Decompress, ZstdErrorCode } from '../vendor/fzstd.js';

export { decompress, Decompress, ZstdErrorCode };

export function isZstdBytes(bytes) {
    if (!bytes) return false;
    const u = bytes instanceof Uint8Array
        ? bytes
        : (bytes instanceof ArrayBuffer || ArrayBuffer.isView(bytes))
            ? new Uint8Array(bytes.buffer || bytes, bytes.byteOffset || 0, bytes.byteLength ?? bytes.length)
            : null;
    if (!u || u.length < 4) return false;
    return u[0] === 0x28 && u[1] === 0xb5 && u[2] === 0x2f && u[3] === 0xfd;
}

export function decompressZstd(input) {
    const u = input instanceof Uint8Array
        ? input
        : new Uint8Array(input.buffer || input, input.byteOffset || 0, input.byteLength ?? input.length);
    return decompress(u);
}

export function decodeTextOrZstd(input) {
    if (typeof input === 'string') return input;
    if (input instanceof Uint8Array || input instanceof ArrayBuffer || ArrayBuffer.isView(input)) {
        const u = input instanceof Uint8Array
            ? input
            : new Uint8Array(input.buffer || input, input.byteOffset || 0, input.byteLength ?? input.length);
        const raw = isZstdBytes(u) ? decompress(u) : u;
        return new TextDecoder('utf-8').decode(raw);
    }
    throw new TypeError('Expected string, Uint8Array, or ArrayBuffer');
}
