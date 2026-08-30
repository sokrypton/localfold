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

export function float32ToFloat16Array(values) {
  const result = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i += 1) result[i] = numberToFloat16(values[i]);
  return result;
}
