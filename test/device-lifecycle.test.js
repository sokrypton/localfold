/**
 * The page's device across the failures that used to strand it: a null adapter, a lost
 * device and a device given back on purpose. A fake adapter stands in for the GPU, because
 * what is under test is what the page remembers, and none of it computes anything.
 */
import { strict as assert } from "node:assert";
import { beforeEach, describe, it } from "node:test";

function fakeAdapter() {
  const lost = {};
  lost.promise = new Promise((resolve) => { lost.resolve = resolve; });
  const listeners = [];
  const device = {
    label: "fake",
    features: new Set(),
    limits: { maxBufferSize: 2 ** 31 },
    lost: lost.promise,
    queue: { submit() {}, onSubmittedWorkDone: async () => {}, writeBuffer() {} },
    addEventListener: (name, listener) => { if (name === "uncapturederror") listeners.push(listener); },
    destroyed: false,
    destroy() { this.destroyed = true; lost.resolve({ reason: "destroyed", message: "" }); },
    fail(message) { lost.resolve({ reason: "unknown", message }); },
    raise(error) { for (const listener of listeners) listener({ error }); },
    createBuffer: () => { throw new Error("the occupancy probe was not stubbed"); },
    pushErrorScope() {},
    popErrorScope: async () => null,
  };
  return {
    device,
    adapter: {
      features: new Set(), limits: {},
      info: { vendor: "acme", architecture: "test", description: "fake card" },
      requestDevice: async () => device,
    },
  };
}

const adapters = [];
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { gpu: { requestAdapter: async () => adapters.shift() ?? null }, deviceMemory: 16 },
});

const { deviceSummary, getDevice, onDeviceFault, releaseDevice } = await import("../web/model.js");

const settled = () => new Promise((resolve) => { setTimeout(resolve, 5); });

describe("the page's device", () => {
  beforeEach(async () => {
    await releaseDevice();
    adapters.length = 0;
  });

  it("asks again after no adapter was found, rather than failing until a reload", async () => {
    await assert.rejects(getDevice(), /no usable GPU/);
    const { adapter, device } = fakeAdapter();
    adapters.push(adapter);
    assert.equal(await getDevice(), device);
  });

  it("says what to do about no adapter", async () => {
    await assert.rejects(getDevice(), /close every window of this browser/);
  });

  it("hands out one device while it is alive", async () => {
    const { adapter } = fakeAdapter();
    adapters.push(adapter);
    assert.equal(await getDevice(), await getDevice());
  });

  it("asks for a new device after one is lost, and reports the loss once", async () => {
    const first = fakeAdapter();
    const second = fakeAdapter();
    adapters.push(first.adapter, second.adapter);
    const faults = [];
    const stop = onDeviceFault((fault) => faults.push(fault));
    await getDevice();
    first.device.fail("the driver went away");
    await settled();
    stop();
    assert.equal(faults.length, 1);
    assert.equal(faults[0].kind, "lost");
    assert.match(faults[0].message, /the driver went away/);
    assert.equal(await getDevice(), second.device);
  });

  it("frees a device given back on purpose, and says nothing about it", async () => {
    const { adapter, device } = fakeAdapter();
    adapters.push(adapter);
    const faults = [];
    const stop = onDeviceFault((fault) => faults.push(fault));
    await getDevice();
    assert.equal(await releaseDevice(), true);
    await settled();
    stop();
    assert.equal(device.destroyed, true);
    assert.deepEqual(faults, []);
    assert.equal(await deviceSummary(), undefined);
  });

  it("reports running out of memory once, not once per buffer that follows", async () => {
    class GPUOutOfMemoryError extends Error {}
    globalThis.GPUOutOfMemoryError = GPUOutOfMemoryError;
    const { adapter, device } = fakeAdapter();
    adapters.push(adapter);
    const faults = [];
    const stop = onDeviceFault((fault) => faults.push(fault));
    await getDevice();
    device.raise(new GPUOutOfMemoryError("out of memory"));
    device.raise(new GPUOutOfMemoryError("out of memory"));
    stop();
    delete globalThis.GPUOutOfMemoryError;
    assert.equal(faults.length, 1);
    assert.equal(faults[0].kind, "out-of-memory");
  });

  it("names the card and the limit for the settings panel", async () => {
    const { adapter } = fakeAdapter();
    adapters.push(adapter);
    await getDevice();
    const summary = await deviceSummary();
    assert.equal(summary.vendor, "acme");
    assert.equal(summary.budgetBytes, Math.round(16 * 1024 ** 3 / 3));
    assert.equal(summary.residentBytes, 0);
  });
});
