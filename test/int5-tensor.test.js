/**
 * Decoding int5 tensors: the bit unpacking, and the asymmetric reconstruction.
 *
 * The exporter (tools/quantize_af3.py) and this reader have to agree on a bit
 * layout that nothing else checks. A packer that is off by one bit produces a
 * buffer of exactly the right length full of plausible small numbers, and the
 * only symptom is a protein that folds slightly wrong - which is why the cases
 * below are about BOUNDARIES rather than about typical values.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { readTensor, tensorByteLength } from "../src/reference/dtype.js";

const GROUP = 32;
const GROUP_BYTES = 20;

/** Build an int5 tensor buffer the way tools/quantize_af3.py lays one out. */
function build(codesByGroup, scales, zeros) {
  const groups = codesByGroup.length;
  const packedLength = groups * GROUP_BYTES + 1;      // one byte of slack
  const scalePad = (-packedLength) % 4 + (((-packedLength) % 4) < 0 ? 4 : 0);
  const scaleAt = packedLength + scalePad;
  const total = scaleAt + groups * 2 + groups * 2;
  const bytes = new Uint8Array(total);
  codesByGroup.forEach((codes, group) => {
    codes.forEach((code, index) => {
      const bit = index * 5;
      const at = group * GROUP_BYTES + (bit >> 3);
      const value = (code & 31) << (bit & 7);
      bytes[at] |= value & 0xff;
      bytes[at + 1] |= (value >> 8) & 0xff;
    });
  });
  const half = new Float16Array(bytes.buffer, scaleAt, groups * 2);
  half.set(scales, 0);
  half.set(zeros, groups);
  return {
    buffer: bytes.buffer,
    record: {
      dtype: "int5", shape: [groups * GROUP], block: GROUP,
      byteOffset: 0, scaleOffset: scaleAt, zeroOffset: scaleAt + groups * 2,
    },
  };
}

describe("int5 tensors", () => {
  it("decodes every code position in a group, including the ones that span two bytes", () => {
    // 🔴 FIVE BITS INTO EIGHT DOES NOT DIVIDE. Codes 0, 8, 16, 24 start on a
    // byte boundary and the rest do not; codes 1, 4, 6, 9... straddle two
    // bytes. A reader that takes one byte is right for a fifth of them.
    const codes = Array.from({ length: GROUP }, (_, i) => i);   // 0..31, every code
    const { buffer, record } = build([codes], [1], [0]);
    const out = readTensor(record, buffer, 0, true);
    for (let index = 0; index < GROUP; index += 1) {
      assert.equal(out[index], index, `code at position ${index}`);
    }
  });

  it("keeps groups independent, so a code never leaks across a boundary", () => {
    const first = new Array(GROUP).fill(31);      // all bits set
    const second = new Array(GROUP).fill(0);      // all bits clear
    const { buffer, record } = build([first, second], [1, 1], [0, 0]);
    const out = readTensor(record, buffer, 0, true);
    for (let index = 0; index < GROUP; index += 1) assert.equal(out[index], 31);
    for (let index = GROUP; index < 2 * GROUP; index += 1) {
      assert.equal(out[index], 0, `group 1 position ${index - GROUP} picked up group 0`);
    }
  });

  it("applies each group's own scale and zero point", () => {
    // 🔴 ASYMMETRIC: value = code * scale + zero. Dropping the zero leaves the
    // group shifted by its own low value, which is smooth, finite and wrong.
    const { buffer, record } = build(
      [new Array(GROUP).fill(0), new Array(GROUP).fill(31)],
      [0.5, 2], [-4, 10]);
    const out = readTensor(record, buffer, 0, true);
    assert.equal(out[0], -4);                       // 0 * 0.5 + (-4)
    assert.equal(out[GROUP], 31 * 2 + 10);          // 72
  });

  it("reports a byte length that covers the codes, the scales and the zeros", () => {
    const { record } = build([new Array(GROUP).fill(0)], [1], [0]);
    // Codes plus slack, padded to four, then two float16 arrays.
    assert.equal(tensorByteLength(record), record.zeroOffset + 2);
  });

  it("refuses an int5 tensor with no zero point rather than guessing one", () => {
    const { buffer, record } = build([new Array(GROUP).fill(1)], [1], [0]);
    const broken = { ...record, zeroOffset: undefined };
    assert.throws(() => readTensor(broken, buffer, 0, true), /zero offset/);
  });
});
