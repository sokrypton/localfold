/**
 * The bundles' five-bit weights, decoded in the SHADER instead of on the host.
 *
 * 🔴 THIS IS THE LAST FACTOR THE DENSE PROJECTION HAS, AND IT IS THE ONLY ONE
 * THAT COSTS NOTHING IN ACCURACY. A big projection here is bound by streaming
 * its weight (webgl/kernels/linear.js): bioemu2's 576 x 896 -> 1792 moves
 * 231 MB of float16 weight per call at about 8 GB/s, which is the 28.5 ms it
 * takes. `model-bioemu2-int5` ships the same weights as five-bit CODES - this
 * backend has been decoding them to float32 on the HOST and uploading the
 * result - and five bits plus a group's scale and zero is **0.875 bytes a
 * weight against float16's two**. The decoded value is the float32 number the
 * host would have computed, so this is FASTER AND MORE ACCURATE than the f16
 * weights it replaces, which is why it is not a flag with a precision trade.
 *
 * 🔴 AND IT READS FEWER TEXELS TOO, WHICH IS NOT OBVIOUS. An RGBA32UI texel is
 * 128 bits, so it holds **25.6 codes** where an RGBA16F texel holds four
 * values; four consecutive codes are twenty bits and fall inside one texel
 * unless they straddle its last word, so `decodeInt5x4` averages **1.25
 * fetches per four weights** against float16's one - and it needs a quarter of
 * the bytes to do it.
 *
 * THE FORMAT, from `readTensor`'s int5 branch in shared/weights/dtype.js, which
 * is the reference this must agree with elementwise:
 *
 *   - a GROUP is 32 elements in 20 bytes (32 x 5 bits is exactly 160), with one
 *     float16 `scale` and one float16 `zero` of its own;
 *   - a code is an OFFSET FROM THE ZERO, not a multiple of the scale:
 *     `value = code * scale + zero`. Reading it as symmetric "stays finite,
 *     stays smooth, and is a different model";
 *   - the codes are a little-endian bitstream over the group's 20 bytes, code
 *     `i` at bit `i * 5`. A group is exactly five 32-bit words, so code `i` of
 *     group `g` sits at bit `g * 160 + i * 5` of the whole word stream with no
 *     per-group realignment - which is the property that makes the shader's
 *     address arithmetic two shifts.
 *
 * 🔴 AND IT NEEDS A DIFFERENTIAL BEFORE IT NEEDS A BENCHMARK.
 * `tools/gpu/check-quantised-upload.js` once found the WGSL device decoder and
 * this host one disagreeing on **104,170 of 131,072 elements** at int5 group 64
 * - and it was the HOST that was wrong. `tools/gpu/check-webgl-int5-decode.js`
 * holds this decoder to `readTensor` element by element on real bundle tensors,
 * not to a tolerance.
 */
import { tensorElements } from "../../shared/weights/dtype.js";
import { memoSource } from "./execution.js";

/** Bytes of codes in one group: 32 codes x 5 bits. */
export const INT5_GROUP_BYTES = 20;
/** Elements in one group. The only group this decoder takes; see `int5WeightSource`. */
export const INT5_GROUP = 32;
/** 32-bit words in one group, which is why the bit arithmetic needs no realignment. */
export const INT5_GROUP_WORDS = INT5_GROUP_BYTES / 4;

/**
 * The codes and the group tables of an int5 tensor, in the two shapes a texture
 * takes them: a `Uint32Array` of the code bitstream, and a `Float32Array` of
 * (scale, zero) pairs.
 *
 * Returns null when the tensor is not int5 at group 32 - the caller then
 * uploads it the ordinary way, so a float32 bundle and an int3 one still fold.
 *
 * @param {{dtype?: string, block?: number, scaleOffset?: number,
 *   zeroOffset?: number, byteOffset?: number}} record
 * @param {ArrayBuffer} buffer
 * @param {number} byteOffset where this tensor's codes start in `buffer`
 */
export function int5WeightSource(record, buffer, byteOffset) {
  if (record.dtype !== "int5" || record.block !== INT5_GROUP) return null;
  if (!Number.isInteger(record.scaleOffset) || !Number.isInteger(record.zeroOffset)) return null;
  if (typeof globalThis.Float16Array !== "function") return null;
  const elements = tensorElements(record);
  // 🔴 ONLY A WHOLE NUMBER OF GROUPS. A tensor whose last group is short is
  // legal in the format and `readTensor` handles it, but the kernel's address
  // arithmetic assumes every output row starts on a group boundary, so a ragged
  // tail would be decoded from the wrong bits. Refused rather than guessed.
  if (elements % INT5_GROUP !== 0) return null;
  const groups = elements / INT5_GROUP;
  const base = record.byteOffset ?? 0;
  const scaleAt = byteOffset + (record.scaleOffset - base);
  const zeroAt = byteOffset + (record.zeroOffset - base);

  // The code region is `groups * 20` contiguous bytes, which is a whole number
  // of 32-bit words - so a word view over a copy of it IS the bitstream the
  // shader indexes, on every platform WebGL runs on (all little-endian).
  const codeBytes = new Uint8Array(buffer, byteOffset, groups * INT5_GROUP_BYTES);
  const words = new Uint32Array(groups * INT5_GROUP_WORDS);
  new Uint8Array(words.buffer).set(codeBytes);

  // Read through Float16Array so the float32 values are exactly the ones
  // `readTensor` multiplies by; a hand-rolled conversion here would be a second
  // implementation of the one thing both sides must agree about.
  const scales = new globalThis.Float16Array(buffer, scaleAt, groups);
  const zeros = new globalThis.Float16Array(buffer, zeroAt, groups);
  const scaleZero = new Float32Array(groups * 2);
  for (let group = 0; group < groups; group += 1) {
    scaleZero[group * 2] = scales[group];
    scaleZero[group * 2 + 1] = zeros[group];
  }
  return { words, scaleZero, groups, elements };
}

/**
 * GLSL: four decoded weights from the code stream, and one group's scale and
 * zero.
 *
 * `decodeInt5x4(codeTex, element, scale, zero)` takes the ELEMENT index of the
 * first of four consecutive weights, which must be a multiple of four so the
 * twenty bits they occupy start on a code boundary.
 *
 * 🔴 TWO WORDS ALWAYS SUFFICE AND THE THIRD READ WOULD BE UNDEFINED. Twenty
 * bits starting at a bit offset of at most 31 end by bit 51, inside a 64-bit
 * pair - and `hi << (32 - sh)` with `sh == 0` is a shift by 32, which GLSL
 * leaves undefined, so that case is branched around rather than masked.
 */
export const INT5_DECODE = `
uint int5Pick(uvec4 v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

uvec4 int5Texel(highp usampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}

vec4 decodeInt5x4(highp usampler2D codeTex, int element, float scale, float zero) {
  int group = element >> 5;
  // A group is exactly 160 bits, so a code's place in the whole stream needs no
  // per-group realignment.
  int bit = group * 160 + (element & 31) * 5;
  int word = bit >> 5;
  int channel = word & 3;
  int texel = word >> 2;
  uvec4 first = int5Texel(codeTex, texel);
  // The second word is in this texel unless the first was its last channel.
  uvec4 second = channel == 3 ? int5Texel(codeTex, texel + 1) : first;
  uint lo = int5Pick(first, channel);
  uint hi = channel == 3 ? second.x : int5Pick(first, channel + 1);
  int shift = bit & 31;
  uint packed = shift == 0 ? lo : (lo >> shift) | (hi << (32 - shift));
  return vec4(
    float(packed & 31u),
    float((packed >> 5) & 31u),
    float((packed >> 10) & 31u),
    float((packed >> 15) & 31u)) * scale + zero;
}


// (scale, zero) of one group: two pairs to an RGBA32F texel.
vec2 int5ScaleZero(sampler2D tableTex, int group) {
  vec4 pair = texelFetch(tableTex, ivec2((group >> 1) % W, (group >> 1) / W), 0);
  return (group & 1) == 0 ? pair.xy : pair.zw;
}
`;

/**
 * Eight decoded weights - forty bits, which is exactly five bytes - from one
 * pair of words.
 *
 * 🔴 THE FOUR-AT-A-TIME FORM WAS SLOWER THAN THE FLOAT16 IT REPLACES, AND THE
 * ADDRESSING WAS WHY. A decode is about fifteen integer operations of address
 * arithmetic for two texel reads, against float16's four for one, and the
 * blocked projection calls it once per output row per k-step - so at four codes
 * a call the arithmetic is paid eight times a group per row. Measured, that
 * made a bioemu2 fold 23% slower even though it moved 2.3x fewer weight bytes:
 * the kernel is not purely traffic-bound, and quadrupling its integer work
 * spent more than the bytes saved.
 *
 * Eight at a time halves it, and forty bits is the natural unit: a code's
 * offset within the group is a multiple of forty for every chunk, so the shift
 * is always 0, 8, 16 or 24 and the span is always exactly two words.
 */
export const INT5_DECODE8 = `
// The forty bits of a chunk, aligned: the low word and the eight bits above it.
// Split out from the decode so a caller can hold EIGHT codes as two uints (two
// registers) rather than as eight floats, and widen each half only while it is
// being used.
uvec2 int5Bits(highp usampler2D codeTex, int element) {
  int group = element >> 5;
  int bit = group * 160 + (element & 31) * 5;
  int word = bit >> 5;
  int channel = word & 3;
  int texel = word >> 2;
  uvec4 first = int5Texel(codeTex, texel);
  uvec4 second = channel == 3 ? int5Texel(codeTex, texel + 1) : first;
  uint w0 = int5Pick(first, channel);
  uint w1 = channel == 3 ? second.x : int5Pick(first, channel + 1);
  int shift = bit & 31;
  return shift == 0 ? uvec2(w0, w1) : uvec2((w0 >> shift) | (w1 << (32 - shift)), w1 >> shift);
}

vec4 int5Low(uvec2 bits, float scale, float zero) {
  return vec4(
    float(bits.x & 31u),
    float((bits.x >> 5) & 31u),
    float((bits.x >> 10) & 31u),
    float((bits.x >> 15) & 31u)) * scale + zero;
}

vec4 int5High(uvec2 bits, float scale, float zero) {
  return vec4(
    float((bits.x >> 20) & 31u),
    float((bits.x >> 25) & 31u),
    float(((bits.x >> 30) | (bits.y << 2)) & 31u),
    float((bits.y >> 3) & 31u)) * scale + zero;
}

void decodeInt5x8(highp usampler2D codeTex, int element, float scale, float zero,
    out vec4 lo, out vec4 hi) {
  int group = element >> 5;
  int bit = group * 160 + (element & 31) * 5;
  int word = bit >> 5;
  int channel = word & 3;
  int texel = word >> 2;
  uvec4 first = int5Texel(codeTex, texel);
  uvec4 second = channel == 3 ? int5Texel(codeTex, texel + 1) : first;
  uint w0 = int5Pick(first, channel);
  uint w1 = channel == 3 ? second.x : int5Pick(first, channel + 1);
  int shift = bit & 31;
  // The forty bits, as a low word and the eight bits above it.
  uint a = shift == 0 ? w0 : (w0 >> shift) | (w1 << (32 - shift));
  uint b = shift == 0 ? w1 : (w1 >> shift);
  lo = vec4(
    float(a & 31u),
    float((a >> 5) & 31u),
    float((a >> 10) & 31u),
    float((a >> 15) & 31u)) * scale + zero;
  hi = vec4(
    float((a >> 20) & 31u),
    float((a >> 25) & 31u),
    float(((a >> 30) | (b << 2)) & 31u),
    float((b >> 3) & 31u)) * scale + zero;
}
`;

/**
 * Decode a whole int5 weight into an ordinary tensor, in one pass.
 *
 * 🔴 THE FALLBACK THAT MAKES THE CODES SAFE TO UPLOAD AT ALL. The blocked
 * projection decodes in its inner loop, which needs the codes to run along the
 * axis a fragment sweeps - `[out][in]` with an input width that is a whole
 * number of GROUPS. bioemu2 has weights that are neither: its single
 * conditioner is 839 wide (384 trunk channels plus 455 featuriser columns, a
 * number with no reason to be a multiple of anything) and its pair projections
 * are read `[in][out]`. Without this they would have to be uploaded twice, or
 * the uploader would have to know how each caller will read it - which it
 * cannot. So `linear` decodes such a weight ONCE, memoises it on the weight
 * object, and every caller works.
 *
 * `roundHalf` writes float16, which is what those tensors were before the codes
 * arrived, so the fallback costs no extra memory against that baseline. The
 * rounding is explicit because a float16 render target TRUNCATES on this
 * backend (see `Execution`'s PRELUDE) and truncation's bias accumulates.
 */
const int5DecodeSourceUncached = (prelude, oneTarget, roundHalf) => `${prelude}${oneTarget}
uniform int ELEMENTS;
uniform highp usampler2D codeTex;
uniform sampler2D tableTex;
${INT5_DECODE}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int element = texel * 4;
  vec2 sz = int5ScaleZero(tableTex, element >> 5);
  vec4 decoded = decodeInt5x4(codeTex, element, sz.x, sz.y);
  result = ${roundHalf ? "half4(decoded)" : "decoded"};
}`;

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const int5DecodeSource = memoSource(int5DecodeSourceUncached);
