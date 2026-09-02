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
import { describe, it } from "node:test";

import {
  GpuMemoryBudgetError, memorySnapshot, noteAllocation, noteDestroy, setMemoryBudget,
} from "../src/runtime/device-memory.js";

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
