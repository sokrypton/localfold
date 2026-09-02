/**
 * Decoding PART of a tensor, which has to agree with decoding all of it.
 *
 * readTensorRange exists so that one pairformer block can be read out of a
 * tensor stacked over 48 of them without materialising the other 47 - 562 MiB
 * of float32 that used to sit in the heap for the life of the page. What makes
 * that safe is exactly one property: a range decodes to what the whole tensor
 * would hold at those indices, bit for bit.
 *
 * 🔴 THE QUANTISED DTYPES ARE WHERE THAT CAN GO WRONG, because a scale belongs
 * to a GROUP and a range does not respect group boundaries. Every case below
 * therefore sweeps ranges that start and end mid-group, at a group boundary,
 * one either side of one, and over the short final group - and compares against
 * the whole decode rather than against a hand-written expectation, so the two
 * paths cannot drift apart.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { readTensor, readTensorRange, tensorElements } from "../src/reference/dtype.js";

const GROUP = 32;
const GROUP_BYTES = 20;

/** An int5 tensor laid out the way tools/quantize_af3.py lays one out. */
function int5(elements) {
  const groups = Math.ceil(elements / GROUP);
  const packedLength = groups * GROUP_BYTES + 1;
  const scaleAt = Math.ceil(packedLength / 4) * 4;
  const bytes = new Uint8Array(scaleAt + groups * 4);
  for (let group = 0; group < groups; group += 1) {
    for (let index = 0; index < GROUP; index += 1) {
      const code = (group * 7 + index * 3) & 31;
      const bit = index * 5;
      const at = group * GROUP_BYTES + (bit >> 3);
      const value = code << (bit & 7);
      bytes[at] |= value & 0xff;
      bytes[at + 1] |= (value >> 8) & 0xff;
    }
  }
  const half = new Float16Array(bytes.buffer, scaleAt, groups * 2);
  for (let group = 0; group < groups; group += 1) {
    half[group] = 0.125 + group * 0.0625;
    half[groups + group] = -1 + group * 0.5;
  }
  return {
    buffer: bytes.buffer,
    record: {
      dtype: "int5", shape: [elements], block: GROUP,
      byteOffset: 0, scaleOffset: scaleAt, zeroOffset: scaleAt + groups * 2,
    },
  };
}

/** An int8 tensor, symmetric, with a float16 scale per block. */
function int8(elements, block = 64) {
  const blocks = Math.ceil(elements / block);
  const scaleAt = Math.ceil(elements / 4) * 4;
  const bytes = new Uint8Array(scaleAt + blocks * 2);
  const codes = new Int8Array(bytes.buffer, 0, elements);
  for (let index = 0; index < elements; index += 1) codes[index] = ((index * 37) % 251) - 125;
  const scales = new Float16Array(bytes.buffer, scaleAt, blocks);
  for (let block_ = 0; block_ < blocks; block_ += 1) scales[block_] = 0.0078125 * (block_ + 1);
  return {
    buffer: bytes.buffer,
    record: {
      dtype: "int8", shape: [elements], block, byteOffset: 0, scaleOffset: scaleAt,
    },
  };
}

function float32(elements) {
  const values = new Float32Array(elements);
  for (let index = 0; index < elements; index += 1) values[index] = index * 0.5 - 3;
  return { buffer: values.buffer, record: { dtype: "float32", shape: [elements] } };
}

function float16(elements) {
  const values = new Float16Array(elements);
  for (let index = 0; index < elements; index += 1) values[index] = index * 0.25 - 2;
  return { buffer: values.buffer, record: { dtype: "float16", shape: [elements] } };
}

/**
 * Ranges that land on and around every boundary a decoder can care about, for a
 * tensor of `elements` grouped by `group`.
 */
function ranges(elements, group) {
  const edges = new Set([0, 1, group - 1, group, group + 1, 2 * group,
                         elements - group, elements - 1, elements]);
  const starts = [...edges].filter((value) => value >= 0 && value < elements).sort((a, b) => a - b);
  const cases = [];
  for (const first of starts) {
    for (const count of [1, group - 1, group, group + 1, elements - first]) {
      if (count > 0 && first + count <= elements) cases.push([first, count]);
    }
  }
  return cases;
}

function checkEveryRange(name, { buffer, record }, group) {
  const elements = tensorElements(record);
  const whole = readTensor(record, buffer, 0);
  for (const [first, count] of ranges(elements, group)) {
    const part = readTensorRange(record, buffer, 0, first, count);
    assert.equal(part.length, count, `${name} range ${first}:${count} has the wrong length`);
    for (let index = 0; index < count; index += 1) {
      assert.equal(part[index], whole[first + index],
        `${name} range ${first}:${count} differs at ${index}`);
    }
  }
}

describe("decoding part of a tensor", () => {
  it("reads int5 ranges exactly as the whole decode reads them", () => {
    // ...not a multiple of the group, so the final group is short.
    checkEveryRange("int5", int5(32 * 5 + 9), GROUP);
  });

  it("reads int8 ranges exactly as the whole decode reads them", () => {
    checkEveryRange("int8", int8(64 * 4 + 13), 64);
  });

  it("reads float32 ranges exactly as the whole decode reads them", () => {
    checkEveryRange("float32", float32(300), 64);
  });

  it("reads float16 ranges exactly as the whole decode reads them", () => {
    checkEveryRange("float16", float16(300), 64);
  });

  it("refuses a range that leaves the tensor", () => {
    const { buffer, record } = int8(128);
    assert.throws(() => readTensorRange(record, buffer, 0, 120, 16), RangeError);
    assert.throws(() => readTensorRange(record, buffer, 0, -1, 4), RangeError);
  });

  it("decodes a whole tensor as the range covering all of it", () => {
    const { buffer, record } = int5(32 * 3);
    const whole = readTensor(record, buffer, 0);
    const range = readTensorRange(record, buffer, 0, 0, whole.length);
    assert.deepEqual(Array.from(range), Array.from(whole));
  });
});
