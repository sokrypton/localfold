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

/**
 * 🔴 A PACKED WIDTH IS NOT IN THIS TABLE AND MUST NOT NEED TO BE. It listed
 * `int5: 1` because int5 was the only packed dtype that existed when it was
 * written, and the entry is never READ - the packed branch below returns before
 * it is used - so it was doing nothing but satisfying the presence check on the
 * line after. The ESM-C bundle ships int3, and the whole of this file handles
 * it: `packedBits` matches any width from one to seven, `readTensor` decodes
 * any of them, and the only thing that refused was this lookup, with
 * "unsupported tensor dtype int3" from a reader that supports it.
 */
const BYTES = { float32: 4, float16: 2, int8: 1 };

/**
 * The width of a sub-byte packed integer dtype, or null.
 *
 * 🔴 NINE BITS IS THE LIMIT AND IT IS NOT ARBITRARY. A code starting at bit
 * offset 7 and running to nine bits ends at bit 16, so two bytes always hold
 * it; past that the reader would need a third and every loop below would grow
 * a case. Nothing here wants more than six.
 */
export function packedBits(dtype) {
  const match = /^int([1-9])$/.exec(dtype);
  if (match === null) return null;
  const bits = Number(match[1]);
  return bits < 8 ? bits : null;
}

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
  const packed = packedBits(record.dtype);
  const width = BYTES[record.dtype];
  if (width === undefined && packed === null) {
    throw new Error(`unsupported tensor dtype ${record.dtype}`);
  }
  const elements = tensorElements(record);
  if (record.dtype !== "int8" && packed === null) {
    return elements * width;
  }
  const { block, scaleOffset, byteOffset = 0 } = record;
  if (!Number.isInteger(block) || block <= 0) {
    throw new Error(`${record.dtype} tensor has no block size`);
  }
  if (!Number.isInteger(scaleOffset)) {
    throw new Error(`${record.dtype} tensor has no scale offset`);
  }
  const groups = Math.ceil(elements / block);
  // A packed asymmetric dtype carries a zero point per group as well as a
  // scale; int8 is symmetric and carries only the scale.
  const trailing = packedBits(record.dtype) === null ? groups * 2 : groups * 4;
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
export function readTensor(record, buffer, byteOffset, copy = false,
                           Output = Float32Array) {
  return readTensorRange(record, buffer, byteOffset, 0, tensorElements(record),
                         copy, Output);
}

/**
 * The same tensor, decoded straight into half precision.
 *
 * 🔴 IT SAVES A WHOLE PASS OVER THE WEIGHTS, WHICH ON ESM-C IS 723 ms A FOLD.
 * Decoding to float32 and narrowing afterwards reads and writes 573 M elements
 * twice and allocates 2.3 GB of intermediate; writing f16 as the codes are
 * unpacked does neither. Every caller that immediately narrows should use this.
 *
 * 🔴 AND IT IS NOT BIT-IDENTICAL TO DECODING AND NARROWING BY CONSTRUCTION,
 * THOUGH IT MEASURES AS IF IT WERE. JavaScript computes `code * scale + zero`
 * in float64; storing it to a Float32Array rounds once and narrowing rounds
 * again, while storing straight to a Float16Array rounds once. Double rounding
 * can differ from single rounding where the float32 value lands exactly on a
 * float16 tie - so the two are not the same operation. On ESM-C's int3 codes
 * that case does not arise: **0 of 14,894,208 elements differ**, across four
 * tensors including the two largest. A five-bit code times a float16 scale
 * needs at most sixteen mantissa bits and the sum needs no more than float32
 * carries, which is why. Do not read that as a proof for another packer.
 */
export function readTensorAsFloat16(record, buffer, byteOffset) {
  if (typeof Float16Array !== "function") {
    throw new Error("this runtime has no Float16Array");
  }
  return readTensor(record, buffer, byteOffset, true, Float16Array);
}

/**
 * PART of a tensor, decoded without decoding the rest of it.
 *
 * 🔴 THIS IS WHAT KEEPS A STACKED TENSOR OUT OF THE HEAP. The trunk's 48
 * pairformer blocks are stored as one tensor each with the block as the leading
 * axis - 216 MiB for the single transition alone - and a block wants one slice
 * of it. Decoding the whole thing to hand back a subarray held 562 MiB of
 * float32 for the life of the page.
 *
 * 🔴 THE BLOCK SCALES ARE INDEXED BY ABSOLUTE POSITION, so a range decodes to
 * exactly the values the whole tensor would have at those indices. The range is
 * widened to whole quantisation groups internally - the scale is per group and
 * there is no way to start mid-group without recovering it anyway - and the
 * requested window is handed back as a subarray of that, so at most `block`
 * elements either side are decoded and discarded.
 *
 * @param {number} first  the first element wanted
 * @param {number} count  how many
 */
export function readTensorRange(record, buffer, byteOffset, first, count, copy = false,
                                Output = Float32Array) {
  const elements = tensorElements(record);
  if (!Number.isSafeInteger(first) || !Number.isSafeInteger(count)
    || first < 0 || count < 0 || first + count > elements) {
    throw new RangeError(`range ${first}:${count} lies outside a tensor of ${elements}`);
  }

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
    const firstBlock = Math.floor(first / block);
    const lastBlock = Math.min(blocks, Math.ceil((first + count) / block));
    const base = firstBlock * block;
    const output = new Output(Math.min(elements, lastBlock * block) - base);
    // SYMMETRIC, so a code is just a multiple of its block's scale. There is no
    // zero point: at eight bits the bias one would correct measures 0.1 pLDDT,
    // which is noise. tools/quantize_model.py has the numbers.
    // ...the same hoist as int5 below: one Float16Array read a block, not one
    // an element, and no division per element.
    for (let block_ = firstBlock; block_ < lastBlock; block_ += 1) {
      const scale = scales[block_];
      const start = block_ * block;
      const end = Math.min(start + block, elements);
      for (let index = start; index < end; index += 1) output[index - base] = codes[index] * scale;
    }
    return output.subarray(first - base, first - base + count);
  }

  // 🔴 THE GENERAL PATH, FOR EVERY PACKED WIDTH BUT FIVE. int5's loop below is
  // hand-unrolled - eight codes out of exactly five bytes - and hoisting it took
  // decoding a bundle from 5.3 s to 1.5. That unrolling is per width, so rather
  // than write it four times this reads a running bit position: about two more
  // operations an element, on a path a page walks once. The group loop is
  // hoisted the same way, which is where nearly all of that 3.8 s came from.
  const bits = packedBits(record.dtype);
  if (bits !== null && bits !== 5) {
    const { block } = record;
    const base = record.byteOffset ?? 0;
    const scaleAt = byteOffset + (record.scaleOffset - base);
    const zeroAt = byteOffset + (record.zeroOffset - base);
    if (!Number.isInteger(record.zeroOffset)) {
      throw new Error(`${record.dtype} tensor has no zero offset; it is asymmetric`);
    }
    if (typeof Float16Array !== "function") {
      throw new Error("this runtime has no Float16Array, and the model scales are float16");
    }
    const groups = Math.ceil(elements / block);
    const groupBytes = (block * bits) / 8;
    if (!Number.isInteger(groupBytes)) {
      throw new Error(`${record.dtype} at group ${block} does not pack into whole bytes`);
    }
    const codes = new Uint8Array(buffer, byteOffset, groups * groupBytes + 1);
    const scales = view(Float16Array, buffer, scaleAt, groups);
    const zeros = view(Float16Array, buffer, zeroAt, groups);
    const mask = (1 << bits) - 1;
    const firstGroup = Math.floor(first / block);
    const lastGroup = Math.min(groups, Math.ceil((first + count) / block));
    const outputBase = firstGroup * block;
    const output = new Output(Math.min(elements, lastGroup * block) - outputBase);
    for (let group = firstGroup; group < lastGroup; group += 1) {
      const scale = scales[group];
      const zero = zeros[group];
      const start = group * block;
      const end = Math.min(start + block, elements);
      let bit = group * groupBytes * 8;
      for (let index = start; index < end; index += 1) {
        const byte = bit >> 3;
        const value = ((codes[byte] | (codes[byte + 1] << 8)) >> (bit & 7)) & mask;
        output[index - outputBase] = value * scale + zero;
        bit += bits;
      }
    }
    return output.subarray(first - outputBase, first - outputBase + count);
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
    // 🔴 THIS PATH IS int5 AT A GROUP OF 32 AND NOTHING ELSE. INT5_GROUP_BYTES
    // is 20 because 32 x 5 is 160 bits, and the loop below strides by it - so a
    // bundle exported at any other group would be read at the wrong stride and
    // decode into a finite, plausible, wrong tensor. The GENERIC branch above
    // handles any (bits, group) that packs into whole bytes and is only skipped
    // here because five bits at 32 has an unrolled fast path worth 3.8 s on a
    // bundle. Caught by tools/gpu/check-quantised-upload.js, which found the
    // GPU decoder and this one disagreeing by 104,170 of 131,072 elements at
    // int5 group 64 - and it was this side that was wrong.
    if (block !== 32) {
      throw new Error(`int5 at group ${block} is not this decoder's layout;`
        + " it reads a 20-byte group. Export at group 32 or widen this path.");
    }
    const groups = Math.ceil(elements / block);
    const codes = new Uint8Array(buffer, byteOffset,
                                 groups * INT5_GROUP_BYTES + 1);
    const scales = view(Float16Array, buffer, scaleAt, groups);
    const zeros = view(Float16Array, buffer, zeroAt, groups);
    const firstGroup = Math.floor(first / block);
    const lastGroup = Math.min(groups, Math.ceil((first + count) / block));
    const outputBase = firstGroup * block;
    const output = new Output(Math.min(elements, lastGroup * block) - outputBase);
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
    // 🔴 EIGHT CODES OUT OF FIVE BYTES, WITH THE SHIFTS WRITTEN OUT. Forty bits
    // is exactly five bytes, and a group of 32 codes is exactly four of those -
    // INT5_GROUP_BYTES is 20 for that reason - so a full group needs no address
    // arithmetic, no running bit counter and no two-byte straddling read at
    // all. The general form below still runs for a trailing partial group.
    const wholeChunks = (block / 8) | 0;
    for (let group = firstGroup; group < lastGroup; group += 1) {
      const scale = scales[group];
      const zero = zeros[group];
      const groupBase = group * INT5_GROUP_BYTES;
      const start = group * block;
      const end = Math.min(start + block, elements);
      let index = start;
      let at = groupBase;
      for (let chunk = 0; chunk < wholeChunks && index + 8 <= end; chunk += 1) {
        const b0 = codes[at];
        const b1 = codes[at + 1];
        const b2 = codes[at + 2];
        const b3 = codes[at + 3];
        const b4 = codes[at + 4];
        const out = index - outputBase;
        output[out] = (b0 & 31) * scale + zero;
        output[out + 1] = ((b0 >> 5) | ((b1 & 3) << 3)) * scale + zero;
        output[out + 2] = ((b1 >> 2) & 31) * scale + zero;
        output[out + 3] = ((b1 >> 7) | ((b2 & 15) << 1)) * scale + zero;
        output[out + 4] = ((b2 >> 4) | ((b3 & 1) << 4)) * scale + zero;
        output[out + 5] = ((b3 >> 1) & 31) * scale + zero;
        output[out + 6] = ((b3 >> 6) | ((b4 & 7) << 2)) * scale + zero;
        output[out + 7] = (b4 >> 3) * scale + zero;
        index += 8;
        at += 5;
      }
      // ...whatever is left of a short final group, the general way.
      let bit = (index - start) * 5;
      for (; index < end; index += 1) {
        const byteAt = groupBase + (bit >> 3);
        const pair = codes[byteAt] | (codes[byteAt + 1] << 8);
        output[index - outputBase] = ((pair >> (bit & 7)) & 31) * scale + zero;
        bit += 5;
      }
    }
    return output.subarray(first - outputBase, first - outputBase + count);
  }

  if (record.dtype === "float16") {
    if (typeof Float16Array !== "function") {
      throw new Error("this runtime has no Float16Array, and the model is stored as float16");
    }
    // ...ALWAYS A COPY, whether or not one was asked for: widening is a new
    // array by definition, so there is no view to hand back. Asked for half
    // precision it is already half precision, and the copy is a memmove.
    return new Output(view(Float16Array, buffer, byteOffset + first * 2, count));
  }

  if (record.dtype !== "float32") throw new Error(`unsupported tensor dtype ${record.dtype}`);
  const at = byteOffset + first * 4;
  // ...a narrowing caller always gets a new array; only float32 into float32
  // can hand back a view.
  if (Output !== Float32Array) return new Output(view(Float32Array, buffer, at, count));
  return copy
    ? new Float32Array(buffer.slice(at, at + count * 4))
    : view(Float32Array, buffer, at, count);
}
