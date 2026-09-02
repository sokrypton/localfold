import { GpuMemoryBudgetError, noteAllocation, noteDestroy } from "./device-memory.js";

export class AllocatedGpuBuffer {
  buffer;
  byteLength;
  usage;
  #allocator;

  constructor(allocator, buffer, byteLength, usage) {
    this.#allocator = allocator;
    this.buffer = buffer;
    this.byteLength = byteLength;
    this.usage = usage;
  }

  release() {
    const allocator = this.#allocator;
    if (allocator === undefined) return;
    this.#allocator = undefined;
    allocator.noteRelease(this.buffer, this.byteLength, this.usage);
  }
}

export class GpuBufferAllocator {
  device;
  #currentBytes = 0;
  #peakBytes = 0;
  #allocationCount = 0;
  #pooling;
  #pool = new Map();

  constructor(device, pooling = false) {
    this.device = device;
    this.#pooling = pooling;
  }

  allocate(label, requestedBytes, usage) {
    if (!Number.isSafeInteger(requestedBytes) || requestedBytes <= 0) {
      throw new RangeError(`invalid allocation size ${requestedBytes} for ${label}`);
    }
    const byteLength = Math.ceil(requestedBytes / 4) * 4;
    const key = `${byteLength}:${usage}`;
    const pooled = this.#pool.get(key);
    let buffer = pooled?.pop();
    if (buffer === undefined) {
      // ...counted before it is created, so a refusal costs nothing and the
      // caller learns which tensor did not fit. A POOLED buffer is already on
      // the device and was counted when it was made.
      //
      // 🔴 AND A REFUSAL DROPS THE POOL BEFORE IT GIVES UP. Every released
      // buffer this allocator is holding for reuse still occupies the device
      // and is counted against the budget, so under pressure the pool is what
      // is failing - a cache, holding memory nothing is using. Dropping it and
      // retrying once turns "out of budget" into a slower fold rather than no
      // fold, and a caller that still cannot fit gets the error it deserves.
      try {
        noteAllocation(this.device, label, byteLength);
      } catch (error) {
        if (!(error instanceof GpuMemoryBudgetError) || this.#pool.size === 0) throw error;
        this.destroyPooled();
        noteAllocation(this.device, label, byteLength);
      }
      buffer = this.device.createBuffer({ label, size: byteLength, usage });
    }
    if (pooled?.length === 0) this.#pool.delete(key);
    this.#currentBytes += byteLength;
    this.#peakBytes = Math.max(this.#peakBytes, this.#currentBytes);
    this.#allocationCount += 1;
    return new AllocatedGpuBuffer(this, buffer, byteLength, usage);
  }

  upload(label, data, usage) {
    const allocation = this.allocate(label, data.byteLength, usage | GPUBufferUsage.COPY_DST);
    if (data.byteLength % 4 === 0) {
      this.device.queue.writeBuffer(allocation.buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    } else {
      // WebGPU queue writes must be four-byte aligned even when the logical
      // storage type is f16 and has an odd element count.
      const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
      padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      this.device.queue.writeBuffer(allocation.buffer, 0, padded);
    }
    return allocation;
  }

  noteRelease(buffer, byteLength, usage) {
    this.#currentBytes -= byteLength;
    if (this.#currentBytes < 0) throw new Error("GPU allocator accounting underflow");
    if (this.#pooling) {
      const key = `${byteLength}:${usage}`;
      const pooled = this.#pool.get(key) ?? [];
      pooled.push(buffer);
      this.#pool.set(key, pooled);
    } else {
      buffer.destroy();
      noteDestroy(this.device, byteLength);
    }
  }

  destroyPooled() {
    for (const [key, buffers] of this.#pool.entries()) {
      const byteLength = Number(key.slice(0, key.indexOf(":")));
      for (const buffer of buffers) {
        buffer.destroy();
        noteDestroy(this.device, byteLength);
      }
    }
    this.#pool.clear();
  }

  snapshot() {
    return {
      currentBytes: this.#currentBytes,
      peakBytes: this.#peakBytes,
      allocationCount: this.#allocationCount,
    };
  }
}
