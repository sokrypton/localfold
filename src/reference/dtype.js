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

const BYTES = { float32: 4, float16: 2, int8: 1 };

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
  if (record.dtype !== "int8") return elements * width;
  const { block, scaleOffset, byteOffset = 0 } = record;
  if (!Number.isInteger(block) || block <= 0) throw new Error("int8 tensor has no block size");
  if (!Number.isInteger(scaleOffset)) throw new Error("int8 tensor has no scale offset");
  return (scaleOffset - byteOffset) + Math.ceil(elements / block) * 2;
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
    for (let index = 0; index < elements; index += 1) {
      output[index] = codes[index] * scales[(index / block) | 0];
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
