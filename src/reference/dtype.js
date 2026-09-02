/**
 * Reading a stored tensor, whatever it was stored as.
 *
 * WHY THERE IS MORE THAN ONE ENCODING. The weights ship quantised - int8 with a
 * float16 scale per 64-weight block, which takes a 355 MiB download to 97 MiB
 * and costs 0.1 pLDDT on the reference fold - except where that would be
 * reckless. The structure module composes rigid transforms across eight
 * iterations, so an error in one frame is carried into the next and lands in
 * the coordinates; the geometry tables are not learned weights at all but the
 * residue-constants literals, where rounding an ideal atom position moves an
 * atom by construction. Those stay float32. See tools/quantize_model.py, which
 * decides it and records the measurements behind it.
 *
 * EVERYTHING COMES BACK AS Float32Array. The shaders read `f32`, so the widening
 * happens here, once, at load - which is the whole point: the page used to spend
 * most of a fold rounding 92.8 million weights to arrive at exactly these values.
 *
 * 🔴 A MISSING dtype IS NOT float32. Every manifest names it on every tensor, so
 * an absent one means a manifest this reader does not understand rather than a
 * default worth guessing - and guessing wrong reads the bytes at the wrong
 * stride, which produces not an error but a different protein.
 */

const BYTES = { float32: 4, float16: 2, int8: 1, int5: 1 };

// 32 five-bit codes are exactly 160 bits, so a group is exactly 20 bytes and no
// group straddles another. That is why the AF3 export uses group 32.
const INT5_GROUP_BYTES = 20;

/**
 * A typed-array view, copying first when the offset is not aligned to it.
 *
 * 🔴 A SHARD IS NOT ALWAYS ALIGNED WHERE ITS TENSORS ARE. The converter starts
 * every tensor on a four-byte boundary WITHIN a shard, but a reader may hand
 * that shard in as a view into a larger buffer - node's readFile returns a
 * Buffer that can sit at any offset in a pool - and the absolute offset is then
 * whatever the two add up to. `new Float16Array(buffer, odd, n)` throws, so the
 * unaligned case is bought out with a copy rather than left to chance.
 */
function view(Kind, buffer, byteOffset, length) {
  const width = Kind.BYTES_PER_ELEMENT;
  if (byteOffset % width === 0) return new Kind(buffer, byteOffset, length);
  return new Kind(buffer.slice(byteOffset, byteOffset + length * width));
}

/** The element count, which the callers all need alongside the byte length. */
export function tensorElements(record) {
  return record.shape.reduce((product, value) => product * value, 1);
}

/**
 * How many bytes one tensor occupies from its byteOffset, for bounds-checking
 * and for working out how long a shard should be.
 *
 * For int8 that spans the codes, the padding between them and the scales, and
 * the scales themselves - `scaleOffset` is absolute, so the span is measured
 * from where the tensor starts rather than recomputed from a padding rule.
 */
export function tensorByteLength(record) {
  const width = BYTES[record.dtype];
  if (width === undefined) throw new Error(`unsupported tensor dtype ${record.dtype}`);
  const elements = tensorElements(record);
  if (record.dtype !== "int8" && record.dtype !== "int5") return elements * width;
  const { block, scaleOffset, byteOffset = 0 } = record;
  if (!Number.isInteger(block) || block <= 0) {
    throw new Error(`${record.dtype} tensor has no block size`);
  }
  if (!Number.isInteger(scaleOffset)) {
    throw new Error(`${record.dtype} tensor has no scale offset`);
  }
  const groups = Math.ceil(elements / block);
  // int5 carries a zero point per group as well as a scale.
  const trailing = record.dtype === "int5" ? groups * 4 : groups * 2;
  return (scaleOffset - byteOffset) + trailing;
}

/**
 * A tensor as Float32Array, widened from however it was kept.
 *
 * @param {{dtype: string, shape: number[], block?: number, scaleOffset?: number}} record
 * @param {ArrayBuffer} buffer the shard
 * @param {number} byteOffset where this tensor starts in it
 * @param {boolean} [copy] whether the result must own its memory. The node store
 *   copies because it hands back views on a shared read; the browser stores hold
 *   the shard alive themselves and do not need to.
 */
export function readTensor(record, buffer, byteOffset, copy = false) {
  const elements = tensorElements(record);

  if (record.dtype === "int8") {
    const { block } = record;
    // ...the scale offset is absolute in the manifest, but the shard may be a
    // view into a larger buffer, so it is rebased the same way byteOffset was.
    const scaleAt = byteOffset + (record.scaleOffset - (record.byteOffset ?? 0));
    const codes = view(Int8Array, buffer, byteOffset, elements);
    const blocks = Math.ceil(elements / block);
    if (typeof Float16Array !== "function") {
      throw new Error("this runtime has no Float16Array, and the model scales are float16");
    }
    const scales = view(Float16Array, buffer, scaleAt, blocks);
    const output = new Float32Array(elements);
    // SYMMETRIC, so a code is just a multiple of its block's scale. There is no
    // zero point: at eight bits the bias one would correct measures 0.1 pLDDT,
    // which is noise. tools/quantize_model.py has the numbers.
    // ...the same hoist as int5 below: one Float16Array read a block, not one
    // an element, and no division per element.
    for (let block_ = 0; block_ < blocks; block_ += 1) {
      const scale = scales[block_];
      const start = block_ * block;
      const end = Math.min(start + block, elements);
      for (let index = start; index < end; index += 1) output[index] = codes[index] * scale;
    }
    return output;
  }

  if (record.dtype === "int5") {
    const { block } = record;
    const base = record.byteOffset ?? 0;
    const scaleAt = byteOffset + (record.scaleOffset - base);
    const zeroAt = byteOffset + (record.zeroOffset - base);
    if (!Number.isInteger(record.zeroOffset)) {
      throw new Error("int5 tensor has no zero offset; it is asymmetric and needs one");
    }
    if (typeof Float16Array !== "function") {
      throw new Error("this runtime has no Float16Array, and the model scales are float16");
    }
    const groups = Math.ceil(elements / block);
    const codes = new Uint8Array(buffer, byteOffset,
                                 groups * INT5_GROUP_BYTES + 1);
    const scales = view(Float16Array, buffer, scaleAt, groups);
    const zeros = view(Float16Array, buffer, zeroAt, groups);
    const output = new Float32Array(elements);
    // 🔴 ASYMMETRIC: a code is an offset from the group's zero point, not a
    // multiple of its scale. Reading it as symmetric loses the zero and shifts
    // every group by its own low value - which stays finite, stays smooth, and
    // is a different model.
    //
    // 🔴 A CODE NEVER SPANS MORE THAN TWO BYTES. Five bits starting at bit
    // offset at most 7 ends by bit 12, so two bytes always suffice - and the
    // packer leaves one byte of slack so the last code of a tensor can take its
    // second byte without walking off the buffer.
    // 🔴 GROUP OUTSIDE, ELEMENT INSIDE, AND THE SCALES READ ONCE EACH. Written
    // as one flat loop this re-read scales[group] and zeros[group] for EVERY
    // element - and those are Float16Array, so each read is an f16-to-f64
    // conversion, `block` times more of them than the data has. It also divided
    // and took a modulo per element to recover a group index the loop already
    // knows. Same arithmetic, same output, hoisted: decoding the int5 bundle
    // went from 5.3 s to 1.5 s, which the page pays once and the user waits
    // through all of.
    for (let group = 0; group < groups; group += 1) {
      const scale = scales[group];
      const zero = zeros[group];
      const groupBase = group * INT5_GROUP_BYTES;
      const start = group * block;
      const end = Math.min(start + block, elements);
      let bit = 0;
      for (let index = start; index < end; index += 1) {
        const at = groupBase + (bit >> 3);
        const pair = codes[at] | (codes[at + 1] << 8);
        output[index] = ((pair >> (bit & 7)) & 31) * scale + zero;
        bit += 5;
      }
    }
    return output;
  }

  if (record.dtype === "float16") {
    if (typeof Float16Array !== "function") {
      throw new Error("this runtime has no Float16Array, and the model is stored as float16");
    }
    // ...ALWAYS A COPY, whether or not one was asked for: widening is a new
    // array by definition, so there is no view to hand back.
    return new Float32Array(view(Float16Array, buffer, byteOffset, elements));
  }

  if (record.dtype !== "float32") throw new Error(`unsupported tensor dtype ${record.dtype}`);
  return copy
    ? new Float32Array(buffer.slice(byteOffset, byteOffset + elements * 4))
    : view(Float32Array, buffer, byteOffset, elements);
}
