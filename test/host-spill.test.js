/**
 * A tensor held on the host while something else needs the card, and written back in any piece a stage asks for.
 *
 * The fake queue keeps writeBuffer's own rule, which is what this exists for: a typed array's offset and size are counted
 * in elements and an ArrayBuffer's in bytes. The first version passed bytes with a typed array, on a path that only a fold
 * near a card's limit takes, and Chrome refused it with "Number of bytes to write is too large" on the first real one.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { HostSpill, spillToHost } from "../webgpu/runtime/host-spill.js";

if (globalThis.GPUBufferUsage === undefined) {
  globalThis.GPUBufferUsage = { MAP_READ: 1, COPY_DST: 8, COPY_SRC: 4, STORAGE: 128 };
}
if (globalThis.GPUMapMode === undefined) globalThis.GPUMapMode = { READ: 1 };

function fakeDevice() {
  const pending = [];
  return {
    queue: {
      writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size = undefined) {
        const unit = ArrayBuffer.isView(data) ? data.BYTES_PER_ELEMENT : 1;
        const bytes = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
        const start = dataOffset * unit;
        const count = size === undefined ? bytes.length - start : size * unit;
        if (start + count > bytes.length) throw new RangeError("Number of bytes to write is too large");
        new Uint8Array(buffer.data).set(bytes.subarray(start, start + count), bufferOffset);
      },
      submit() { pending.splice(0).forEach((copy) => copy()); },
      onSubmittedWorkDone: async () => {},
    },
    createCommandEncoder: () => {
      const copies = [];
      return {
        copyBufferToBuffer(source, sourceOffset, target, targetOffset, count) {
          copies.push(() => new Uint8Array(target.data).set(
            new Uint8Array(source.data, sourceOffset, count), targetOffset));
        },
        finish: () => { pending.push(...copies); return {}; },
      };
    },
  };
}

function gpuBuffer(bytes) {
  return {
    data: new ArrayBuffer(bytes),
    mapAsync: async () => {},
    getMappedRange(offset, size) { return this.data.slice(offset, offset + size); },
    unmap() {},
  };
}

const allocator = {
  live: 0,
  peak: 0,
  allocate(_label, size) {
    this.live += size;
    this.peak = Math.max(this.peak, this.live);
    return { buffer: gpuBuffer(size), release: () => { this.live -= size; } };
  },
};

const floats = (count) => Float32Array.from({ length: count }, (_, at) => at + 0.5);

describe("a spill", () => {
  it("comes back whole, in pieces, and costs the card one piece", async () => {
    const device = fakeDevice();
    const values = floats(1000);
    const source = gpuBuffer(values.byteLength);
    new Float32Array(source.data).set(values);
    allocator.peak = 0;
    allocator.live = 0;
    const spill = await spillToHost(device, allocator, source, values.byteLength, "t", 256);
    assert.equal(spill.pieces.length, 16);
    assert.equal(allocator.peak, 256);
    assert.equal(allocator.live, 0);
    const back = gpuBuffer(values.byteLength);
    await spill.restore(device, back);
    assert.deepEqual(new Float32Array(back.data), values);
  });

  it("writes a range that starts inside a piece and ends inside another", () => {
    const device = fakeDevice();
    const values = floats(64);
    const pieces = [values.slice(0, 16), values.slice(16, 32), values.slice(32, 48), values.slice(48)];
    const spill = new HostSpill(values.byteLength, 64, pieces);
    const target = gpuBuffer(100 * 4);
    // Elements 10 to 40, at element 5 of the target.
    spill.writeTo(device, target, 20, 40, 120);
    assert.deepEqual(new Float32Array(target.data, 20, 30), values.slice(10, 40));
    assert.equal(new Float32Array(target.data)[4], 0);
    assert.equal(new Float32Array(target.data)[35], 0);
  });

  it("refuses a range past the end, rather than writing what is not there", () => {
    const spill = new HostSpill(64, 64, [floats(16)]);
    assert.throws(() => spill.writeTo(fakeDevice(), gpuBuffer(128), 0, 32, 64), RangeError);
    assert.throws(() => spill.writeTo(fakeDevice(), gpuBuffer(128), 0, -4, 8), RangeError);
  });

  it("refuses pieces that would split a float", async () => {
    await assert.rejects(spillToHost(fakeDevice(), allocator, gpuBuffer(64), 64, "t", 6), RangeError);
    await assert.rejects(spillToHost(fakeDevice(), allocator, gpuBuffer(64), 62, "t", 8), RangeError);
  });

  it("lets a tensor smaller than a piece be one piece of its own size", async () => {
    const device = fakeDevice();
    const values = floats(8);
    const source = gpuBuffer(values.byteLength);
    new Float32Array(source.data).set(values);
    const spill = await spillToHost(device, allocator, source, values.byteLength, "t", 1024);
    assert.equal(spill.pieces.length, 1);
    assert.equal(spill.pieceBytes, values.byteLength);
    const target = gpuBuffer(values.byteLength);
    spill.writeTo(device, target, 0, 0, values.byteLength);
    assert.deepEqual(new Float32Array(target.data), values);
  });
});
