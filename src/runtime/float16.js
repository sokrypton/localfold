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
