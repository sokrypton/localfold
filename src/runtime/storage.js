import { float32ToFloat16Array } from "./float16.js";

/**
 * How an activation tensor is stored on the device.
 *
 * `f32` is one float a word and is what every reference here was produced
 * with. `f16` packs TWO half-precision values into each 32-bit word with
 * `pack2x16float`, which is core WGSL and needs no device feature - unlike the
 * `f16` TYPE, which needs `shader-f16`. So this halves a tensor on hardware
 * that cannot compute in half precision at all.
 *
 * 🔴 THIS IS STORAGE, NOT ARITHMETIC, AND THE DISTINCTION IS THE WHOLE POINT.
 * Every value is unpacked to f32 the instant it is read and the kernels
 * compute exactly as they did; what changes is the bytes between two kernels.
 * The existing `precision` options on the transition and the projections are
 * the other question - what the multiply and the accumulator are - and the two
 * compose without either knowing about the other.
 *
 * 🔴 AND A WORD IS OWNED BY ONE INVOCATION OR IT IS A RACE. Two halves share a
 * word, and WGSL gives no way to write sixteen bits: a lane that wants to
 * store one half must read the word, insert, and write it back, and the lane
 * holding the other half is doing the same thing at the same time. So a kernel
 * writing a packed tensor has to be arranged so the pair of elements sharing a
 * word is produced by ONE invocation, which is a constraint on the kernel and
 * not something this module can enforce. `storedPair` is the store side and
 * exists to make that arrangement explicit at every call site. Reads are
 * unconstrained: `storedElement` reads one element and any lane may.
 */
export const ACTIVATION_STORAGES = new Set(["f32", "f16"]);

function check(storage) {
  if (!ACTIVATION_STORAGES.has(storage)) {
    throw new RangeError(`unknown activation storage ${storage}`);
  }
  return storage;
}

/** The WGSL element type of an array holding values stored as `storage`. */
export function storageArray(storage) {
  return check(storage) === "f16" ? "u32" : "f32";
}

/** How many 32-bit words back `elements` values stored as `storage`. */
export function storageWords(elements, storage) {
  if (!Number.isSafeInteger(elements) || elements < 0) {
    throw new RangeError(`invalid element count ${elements}`);
  }
  return check(storage) === "f16" ? Math.ceil(elements / 2) : elements;
}

/** How many bytes those words occupy. */
export function storageBytes(elements, storage) {
  return storageWords(elements, storage) * 4;
}

/**
 * A WGSL expression reading element `index` of `array` as an `f32`.
 *
 * `index` is an expression and is parenthesised, so a caller may pass a sum.
 */
export function storedElement(storage, array, index) {
  return check(storage) === "f16"
    ? `unpack2x16float(${array}[(${index}) >> 1u])[(${index}) & 1u]`
    : `${array}[${index}]`;
}

/**
 * A WGSL statement writing the two elements at `2 * pairIndex` and
 * `2 * pairIndex + 1` of `array`.
 *
 * 🔴 THE STORE SIDE TAKES A PAIR INDEX AND NOT AN ELEMENT INDEX, so a kernel
 * that has not been arranged to own both halves of a word cannot call it by
 * accident - it has no pair to pass. In `f32` the pair is simply two adjacent
 * stores and the arrangement costs nothing, which is what lets one generated
 * shader serve both storages.
 */
export function storedPair(storage, array, pairIndex, low, high) {
  if (check(storage) === "f16") {
    return `${array}[${pairIndex}] = pack2x16float(vec2<f32>(${low}, ${high}));`;
  }
  return `${array}[(${pairIndex}) * 2u] = ${low};\n`
    + `${array}[(${pairIndex}) * 2u + 1u] = ${high};`;
}

/**
 * Pack float32 values into half-precision pairs, two to a word.
 *
 * Reinterpreting the halves is the whole of it: on a little-endian host - which
 * is every platform WebGPU runs on - half `2i` is the low sixteen bits of word
 * `i`, which is exactly what `pack2x16float` reads back. An odd count leaves
 * the last word's high half zero, which no reader looks at.
 */
export function packHalfWords(values) {
  const halves = float32ToFloat16Array(values);
  const words = new Uint32Array(Math.ceil(halves.length / 2));
  const view = new Uint16Array(words.buffer);
  view.set(halves);
  return words;
}

/** Unpack `count` values written by `packHalfWords`. */
export function unpackHalfWords(words, count) {
  if (!Number.isSafeInteger(count) || count < 0 || count > words.length * 2) {
    throw new RangeError(`cannot read ${count} halves from ${words.length} words`);
  }
  const halves = new Uint16Array(words.buffer, words.byteOffset, words.length * 2);
  const output = new Float32Array(count);
  for (let index = 0; index < count; index += 1) output[index] = halfToFloat(halves[index]);
  return output;
}

const scratch = new ArrayBuffer(4);
const scratchFloat = new Float32Array(scratch);
const scratchUint = new Uint32Array(scratch);

/**
 * One half's bits to a number.
 *
 * `Float16Array` would do this too, but only where it exists, and the reverse
 * direction in float16.js already carries a scalar path for runtimes without
 * it. Widening is exact in every case, so unlike the narrowing side there is
 * nothing here for a native implementation to disagree about.
 */
function halfToFloat(bits) {
  const sign = (bits & 0x8000) << 16;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) {
    // Subnormal halves are ordinary floats: 2^-24 times the mantissa. Zero
    // falls out of the same expression, sign included.
    const magnitude = mantissa * 2 ** -24;
    return sign === 0 ? magnitude : -magnitude;
  }
  if (exponent === 0x1f) {
    scratchUint[0] = sign | 0x7f800000 | (mantissa === 0 ? 0 : 0x400000);
    return scratchFloat[0];
  }
  scratchUint[0] = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  return scratchFloat[0];
}
