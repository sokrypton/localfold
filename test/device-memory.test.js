/**
 * The device memory account, and the ceiling it enforces.
 *
 * 🔴 THE POINT OF THE CEILING IS TO FAIL BEFORE createBuffer DOES NOT. Metal
 * accepts allocations past where macOS starts paging and a phone's driver takes
 * them and is then killed, so a refusal has to happen on OUR side of the call
 * and has to name what would not fit. These check that, and that the accounting
 * a budget is compared against counts what the device is actually holding
 * rather than what a caller still intends to use.
 */
import { strict as assert } from "node:assert";
import { nameIsAdapters } from "../web/gpu-limits.js";
import { describe, it } from "node:test";

import {
  budgetForDevice, cardBudgetBytes, GpuMemoryBudgetError, gpuMemoryFromName, lossTeachesLimit, memorySnapshot, UNNAMED_CARD_BYTES,
  noteAllocation, noteDestroy, setMemoryBudget,
} from "../webgpu/runtime/device-memory.js";

/** A device is only ever a WeakMap key here, so anything unique will do. */
const device = () => ({});

const MIB = 1024 * 1024;

describe("device memory accounting", () => {
  it("counts what has been allocated and what has gone", () => {
    const gpu = device();
    noteAllocation(gpu, "a", 4 * MIB);
    noteAllocation(gpu, "b", 6 * MIB);
    assert.equal(memorySnapshot(gpu).residentBytes, 10 * MIB);
    assert.equal(memorySnapshot(gpu).bufferCount, 2);
    noteDestroy(gpu, 4 * MIB);
    assert.equal(memorySnapshot(gpu).residentBytes, 6 * MIB);
    assert.equal(memorySnapshot(gpu).bufferCount, 1);
  });

  it("remembers the peak after the memory has gone", () => {
    const gpu = device();
    noteAllocation(gpu, "a", 30 * MIB);
    noteDestroy(gpu, 30 * MIB);
    noteAllocation(gpu, "b", 1 * MIB);
    assert.equal(memorySnapshot(gpu).residentBytes, 1 * MIB);
    assert.equal(memorySnapshot(gpu).peakBytes, 30 * MIB);
  });

  it("keeps two devices apart", () => {
    const one = device();
    const other = device();
    noteAllocation(one, "a", 8 * MIB);
    assert.equal(memorySnapshot(other).residentBytes, 0);
    assert.equal(memorySnapshot(one).residentBytes, 8 * MIB);
  });

  it("counts without bounding until a budget is set", () => {
    const gpu = device();
    for (let index = 0; index < 100; index += 1) noteAllocation(gpu, `t${index}`, 64 * MIB);
    assert.equal(memorySnapshot(gpu).residentBytes, 6400 * MIB);
  });

  it("refuses the allocation that would cross the budget, and names it", () => {
    const gpu = device();
    setMemoryBudget(gpu, 100 * MIB);
    noteAllocation(gpu, "fits", 90 * MIB);
    let error;
    try { noteAllocation(gpu, "pair-activations", 20 * MIB); } catch (caught) { error = caught; }
    assert.ok(error instanceof GpuMemoryBudgetError);
    assert.equal(error.label, "pair-activations");
    assert.equal(error.bytes, 20 * MIB);
    assert.equal(error.budgetBytes, 100 * MIB);
    // ...and the message says all three, because a number without a name sends
    // the reader to the wrong tensor.
    assert.match(error.message, /pair-activations/);
    assert.match(error.message, /20\.0 MiB/);
    assert.match(error.message, /100\.0 MiB/);
  });

  it("leaves the account untouched by a refusal, so the next request is fair", () => {
    const gpu = device();
    setMemoryBudget(gpu, 100 * MIB);
    noteAllocation(gpu, "fits", 90 * MIB);
    assert.throws(() => noteAllocation(gpu, "too big", 20 * MIB), GpuMemoryBudgetError);
    assert.equal(memorySnapshot(gpu).residentBytes, 90 * MIB);
    noteAllocation(gpu, "small enough", 10 * MIB);
    assert.equal(memorySnapshot(gpu).residentBytes, 100 * MIB);
  });

  it("lets a budget be lifted", () => {
    const gpu = device();
    setMemoryBudget(gpu, 10 * MIB);
    assert.throws(() => noteAllocation(gpu, "a", 20 * MIB), GpuMemoryBudgetError);
    setMemoryBudget(gpu, undefined);
    noteAllocation(gpu, "a", 20 * MIB);
    assert.equal(memorySnapshot(gpu).residentBytes, 20 * MIB);
  });

  it("catches a release of memory that was never taken", () => {
    const gpu = device();
    noteAllocation(gpu, "a", 4 * MIB);
    assert.throws(() => noteDestroy(gpu, 8 * MIB), /underflow/);
  });
});

describe("the ceiling a page gives a device", () => {
  const GIB = 1024 * MIB;
  const withRam = (gib, run) => {
    const before = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    Object.defineProperty(globalThis, "navigator",
      { value: { deviceMemory: gib }, configurable: true });
    try { return run(); } finally {
      if (before === undefined) delete globalThis.navigator;
      else Object.defineProperty(globalThis, "navigator", before);
    }
  };

  it("stays under a discrete card's own memory, not a third of system RAM", () => {
    // The report this exists for: 32 GiB of RAM and an 8 GiB RTX 3070 was a 10.9 GiB ceiling.
    const name = "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 (0x00002484) Direct3D11 vs_5_0 ps_5_0, D3D11)";
    const budget = withRam(32, () => budgetForDevice(4, { gpuMemoryBytes: gpuMemoryFromName(name) }));
    assert.equal(budget, cardBudgetBytes(8 * GIB));
  });

  it("is not held to a third of system RAM either, because the card's memory is its own", () => {
    assert.equal(withRam(16, () => budgetForDevice(4, { gpuMemoryBytes: 24 * GIB })), cardBudgetBytes(24 * GIB));
  });

  it("holds a card it cannot name to both, because it may be sharing the RAM", () => {
    assert.equal(withRam(8, () => budgetForDevice(4, { gpuMemoryBytes: UNNAMED_CARD_BYTES })), Math.round(8 * GIB / 3));
  });

  it("is unchanged for a card whose name says nothing", () => {
    assert.equal(withRam(16, () => budgetForDevice(4, { gpuMemoryBytes: gpuMemoryFromName("Apple M2") })),
                 Math.round(16 * GIB / 3));
  });

  it("takes what a lost device taught it when that is lower still", () => {
    assert.equal(withRam(32, () => budgetForDevice(4, { gpuMemoryBytes: 8 * GIB, learnedBytes: 3 * GIB })),
                 3 * GIB);
  });
});

describe("what a lost device says about the card", () => {
  const GIB = 1024 * MIB;

  it("says nothing when it held little of what a fold may hold", () => {
    assert.equal(lossTeachesLimit(1.3 * GIB, 20.8 * GIB), false);
    assert.equal(lossTeachesLimit(2.3 * GIB, 4.8 * GIB), false);
  });

  it("is a measurement when it held much of it", () => {
    assert.equal(lossTeachesLimit(3.9 * GIB, 4.8 * GIB), true);
    assert.equal(lossTeachesLimit(2.4 * GIB, 4.8 * GIB), true);
  });

  it("is a measurement with the limit lifted, and never from almost nothing", () => {
    assert.equal(lossTeachesLimit(5 * GIB, undefined), true);
    assert.equal(lossTeachesLimit(100 * MIB, undefined), false);
    assert.equal(lossTeachesLimit(100 * MIB, 150 * MIB), false);
  });
});

describe("a name that is another card's", () => {
  it("is told by the vendor the adapter reports", () => {
    const rtx = "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 Direct3D11 vs_5_0 ps_5_0, D3D11)";
    const uhd = "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)";
    const radeon = "ANGLE (AMD, AMD Radeon RX 6800 Direct3D11 vs_5_0 ps_5_0, D3D11)";
    assert.equal(nameIsAdapters(rtx, "nvidia"), true);
    assert.equal(nameIsAdapters(uhd, "intel"), true);
    assert.equal(nameIsAdapters(radeon, "amd"), true);
    assert.equal(nameIsAdapters(uhd, "nvidia"), false);
    assert.equal(nameIsAdapters(rtx, "intel"), false);
    assert.equal(nameIsAdapters(rtx, "amd"), false);
  });

  it("is not doubted where there is nothing to compare it with", () => {
    assert.equal(nameIsAdapters("Apple M2", "apple"), true);
    assert.equal(nameIsAdapters("anything", ""), true);
    assert.equal(nameIsAdapters(null, "nvidia"), false);
  });
});

describe("a card's memory from its name", () => {
  const GIB = 1024 * MIB;
  it("reads the renderer strings browsers actually report", () => {
    assert.equal(gpuMemoryFromName("ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 Direct3D11 vs_5_0 ps_5_0, D3D11)"), 8 * GIB);
    assert.equal(gpuMemoryFromName("ANGLE (NVIDIA, NVIDIA GeForce RTX 4090 Direct3D11 vs_5_0 ps_5_0, D3D11)"), 16 * GIB);
    assert.equal(gpuMemoryFromName("ANGLE (AMD, AMD Radeon RX 6800 XT Direct3D11 vs_5_0 ps_5_0, D3D11)"), 12 * GIB);
    assert.equal(gpuMemoryFromName("ANGLE (Intel, Intel(R) Arc(TM) A770 Graphics Direct3D11)"), 8 * GIB);
  });

  it("says nothing rather than guess", () => {
    for (const name of ["Apple M2", "ANGLE (Intel, Intel(R) UHD Graphics 630)", "Mali-G78", "", null, undefined]) {
      assert.equal(gpuMemoryFromName(name), undefined);
    }
  });

  it("does not read a device id or driver number as a model", () => {
    assert.equal(gpuMemoryFromName("ANGLE (NVIDIA, NVIDIA Tesla A100 (0x00003070))"), undefined);
  });
});
