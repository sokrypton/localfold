/**
 * What `planBlockUpload` refuses, now that it plans more than one packing.
 *
 * 🔴 THE SHADER IS GENERATED FROM THE PACKING, SO ONE PLAN IS ONE PACKING. A
 * plan holding an int5 group-32 tensor beside an int3 group-128 one would
 * decode half of it with the other's bit width - and the result is a weight
 * buffer, which nothing downstream can tell is wrong. The refusal is the check.
 *
 * These run on the host: a plan is arithmetic over records and offsets, and it
 * is the half of tools/gpu/check-delta-upload.js that needs no device.
 *
 * 🔴 AND IT WAS NEARLY LOST WITH THE BRANCH IT CAME FROM. The accumulate path
 * was ported onto main by hand and this file was not - so deleting
 * `af2-model-deltas` as "already merged" would have taken six rules with it,
 * including the one that says a plan mixing two packings must be refused.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { planBlockUpload } from "../src/weights/quantised-upload.js";

const record = (dtype, block, count) => ({
  dtype, block, shape: [count], byteOffset: 0,
  scaleOffset: 4096, zeroOffset: 4096 + count / block * 2,
});

const entry = (dtype, block, count, offset) => ({
  thunk: {
    store: { tensorSource: () => ({ record: record(dtype, block, count),
                                    buffer: new ArrayBuffer(65536), byteOffset: 0 }) },
    tensorName: `${dtype}-${offset}`, first: 0, count,
  },
  offset, length: count,
});

describe("planning a packed upload", () => {
  it("reads the bits and the group from the records, not from the caller", () => {
    // 🔴 THE CODEC IS AN OBJECT ON THE PLAN, NOT TWO FLAT FIELDS. This file came
    // off the af2-model-deltas branch, which predates that shape by 334
    // commits; what transfers is the RULES below, and reading `plan.gpu.bits`
    // would have passed `undefined !== undefined` in a test of nothing.
    const plan = planBlockUpload([entry("int3", 128, 256, 0)]);
    assert.equal(plan.gpu.codec.bits, 3);
    assert.equal(plan.gpu.codec.group, 128);
    assert.equal(plan.gpu.accumulate, false);
  });

  it("carries the accumulate flag a delta needs", () => {
    const plan = planBlockUpload([entry("int5", 32, 256, 0)], "f16", { accumulate: true });
    assert.equal(plan.gpu.accumulate, true);
  });

  // 🔴 AND THE FLAG IS THE THIRD ARGUMENT, WHICH IS WHY THE SECOND ONE REFUSES
  // AN OBJECT. On the branch this came from the signature was
  // `(entries, options)`; here it is `(entries, destination, options)`, so the
  // old call sets the DESTINATION to `{ accumulate: true }` and drops the flag
  // - a delta that silently OVERWRITES the weights it was meant to add to,
  // which is the exact failure the per-variant pipeline key exists to prevent.
  // It throws rather than plans.
  it("refuses an options object where the destination goes", () => {
    assert.throws(() => planBlockUpload([entry("int5", 32, 256, 0)], { accumulate: true }),
                  /destination/);
  });

  // 🔴 TWO PACKINGS IN ONE PLAN GO TO THE HOST, WHICH IS THE BRANCH'S RULE MADE
  // SOFTER AND SAFER. It used to refuse the whole plan; this sends the odd
  // tensor to the host packer instead, so the caller still gets the right bytes
  // - see the note at that branch in src/weights/quantised-upload.js. What both
  // versions guarantee is the thing that matters: no tensor is ever decoded
  // with another packing's shift.
  it("sends a second packing to the host rather than decoding it wrong", () => {
    const plan = planBlockUpload([entry("int5", 32, 256, 0),
                                  entry("int3", 128, 256, 256)]);
    assert.equal(plan.gpu.codec.bits, 5);
    assert.equal(plan.host.length, 1);
    assert.equal(plan.gpu.params.length, 1);
  });

  it("sends a group that is not a whole number of bytes to the host", () => {
    // Twelve codes of five bits is sixty bits, so the second group starts
    // mid-byte and no group base address can name it. `tools/quantize_af3.py`
    // refuses to WRITE one for the same reason; this declines to read one.
    const plan = planBlockUpload([entry("int5", 12, 240, 0)]);
    assert.equal(plan.gpu.params.length, 0);
    assert.equal(plan.host.length, 1);
  });

  it("refuses an odd destination offset, which is two writers to one word", () => {
    assert.equal(planBlockUpload([entry("int5", 32, 256, 1)]), undefined);
  });

  it("leaves an unpacked tensor to the host", () => {
    const plan = planBlockUpload([entry("float32", 1, 256, 0)]);
    assert.equal(plan.gpu.params.length, 0);
    assert.equal(plan.host.length, 1);
  });
});
