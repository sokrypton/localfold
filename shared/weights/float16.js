const scratch = new ArrayBuffer(4);
const scratchFloat = new Float32Array(scratch);
const scratchUint = new Uint32Array(scratch);

export function numberToFloat16(value) {
  scratchFloat[0] = value;
  const bits = scratchUint[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;

  if (((bits >>> 23) & 0xff) === 0xff) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }
  if (exponent >= 0x1f) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x0fff + ((mantissa >>> 13) & 1)) >>> 13);
  }

  mantissa += 0x0fff + ((mantissa >>> 13) & 1);
  if ((mantissa & 0x800000) !== 0) {
    mantissa = 0;
    exponent += 1;
    if (exponent >= 0x1f) return sign | 0x7c00;
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

/**
 * 🔴 THIS IS ON THE FOLD'S CRITICAL PATH, NOT IN A LOADER. Holding AF3's
 * weights in f16 means converting about 319 million floats once per process -
 * every pairformer and diffusion transformer block, the first time its resident
 * buffer is filled - and the scalar loop below runs at roughly 8 million floats
 * per 17 ms, which is most of a second on the way to the first fold. It showed
 * up as a whole fold getting SLOWER while the trunk and the denoiser both got
 * faster.
 *
 * `Float16Array` does it in the engine. Measured on 8 Mi floats: 17 ms against
 * 11, and BIT-IDENTICAL - checked over every interesting pattern (both zeroes,
 * both infinities, NaN, the largest finite half, the first value that
 * overflows, the subnormal boundary, ties) and over 8 Mi arbitrary ones, with
 * zero disagreements. Both round to nearest, ties to even.
 *
 * The scalar path stays for runtimes without it, and test/float16.test.js holds
 * the two to each other wherever both exist.
 */
const hasNativeFloat16 = typeof globalThis.Float16Array === "function";

export function float32ToFloat16Array(values) {
  if (hasNativeFloat16) {
    // ...a copy of the BITS, not of the numbers: the buffer is reinterpreted,
    // so nothing is converted twice.
    const half = new globalThis.Float16Array(values);
    return new Uint16Array(half.buffer, half.byteOffset, half.length);
  }
  const result = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i += 1) result[i] = numberToFloat16(values[i]);
  return result;
}


/**
 * Concatenate float32 tensors into one buffer of `precision`, in one pass.
 *
 * 🔴 THE OBVIOUS SHAPE COSTS TWICE AND ALLOCATES THE LARGER HALF. Every packer
 * here built a Float32Array of the whole block, filled it, and then converted
 * that to f16 - so a 31.5 MiB block was written once as f32 and read again to
 * make 15.7 MiB of f16, with the f32 copy live the whole time. Setting straight
 * into a Float16Array converts on the way in: one pass, and the wide buffer
 * never exists.
 *
 * It is worth caring about because this is not a loader. An AF3 fold converts
 * about 319 million floats the first time each block's resident buffer is
 * filled, which is most of a second on the way to the first fold.
 *
 * @param {"f32"|"f16"} precision
 * @param {number} total how many elements the result holds
 * @param {(target: Float32Array | Float16Array) => void} fill writes each
 *   tensor into the target at its offset, through `writeInto`
 */
/**
 * `target.set(source, offset)`, except faster into a Float16Array.
 *
 * 🔴 TypedArray.set FROM A WIDER ELEMENT IS NOT THE FAST PATH IT LOOKS LIKE.
 * Copying 8M floats into a Float16Array measures 9.4 ms through `set`, 6.1 ms
 * through a plain indexed loop and 4.4 ms through one unrolled eight ways -
 * 2.1x, for the same bytes and bit-identical output. `set` between two arrays
 * of the SAME element is a memmove and beats any loop, so that case is left
 * alone; the narrowing case is the one that is not.
 *
 * The unrolled body is not decoration. The plain loop is already 1.5x `set`;
 * the unrolling is the other 1.4x, and it is there because this runs over
 * about 319 million floats on the way to a session's first fold.
 */
export function writeInto(target, source, offset) {
  if (!hasNativeFloat16 || !(target instanceof globalThis.Float16Array)) {
    target.set(source, offset);
    return;
  }
  const count = source.length;
  const whole = count - (count % 8);
  let at = offset;
  for (let index = 0; index < whole; index += 8, at += 8) {
    target[at] = source[index];
    target[at + 1] = source[index + 1];
    target[at + 2] = source[index + 2];
    target[at + 3] = source[index + 3];
    target[at + 4] = source[index + 4];
    target[at + 5] = source[index + 5];
    target[at + 6] = source[index + 6];
    target[at + 7] = source[index + 7];
  }
  for (let index = whole; index < count; index += 1, at += 1) target[at] = source[index];
}

export function concatenateAs(precision, total, fill) {
  if (precision !== "f16") {
    const wide = new Float32Array(total);
    fill(wide);
    return wide;
  }
  if (hasNativeFloat16) {
    const half = new globalThis.Float16Array(total);
    fill(half);
    return new Uint16Array(half.buffer, half.byteOffset, half.length);
  }
  const wide = new Float32Array(total);
  fill(wide);
  return float32ToFloat16Array(wide);
}
