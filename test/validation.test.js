import assert from "node:assert/strict";
import { describe, expect, it } from "./harness.js";
import { DeferredValidation } from "../src/runtime/validation.js";

/** A GPUDevice's error-scope stack, and nothing else. */
function fakeDevice(errors = []) {
  const device = {
    depth: 0,
    maxDepth: 0,
    popped: 0,
    pushErrorScope() {
      device.depth += 1;
      device.maxDepth = Math.max(device.maxDepth, device.depth);
    },
    popErrorScope() {
      device.depth -= 1;
      const error = errors[device.popped] ?? null;
      device.popped += 1;
      return Promise.resolve(error);
    },
  };
  return device;
}

describe("deferred WebGPU validation", () => {
  it("pushes and pops one scope per block", async() => {
    const device = fakeDevice();
    const validation = new DeferredValidation(device, "stack");
    for (let block = 0; block < 4; block += 1) {
      validation.begin();
      validation.end(`block ${block}`);
    }
    expect(device.popped).toBe(4);
    expect(device.depth).toBe(0);
    await validation.settle();
  });

  it("never holds more than one scope open, so the loop does not nest them", () => {
    const device = fakeDevice();
    const validation = new DeferredValidation(device, "stack");
    for (let block = 0; block < 48; block += 1) {
      validation.begin();
      validation.end(`block ${block}`);
    }
    expect(device.maxDepth).toBe(1);
  });

  it("does not wait for a scope before encoding the next block", () => {
    // The point of the class: end() must not return a promise the loop awaits.
    const device = fakeDevice();
    const validation = new DeferredValidation(device, "stack");
    validation.begin();
    expect(validation.end("block 0")).toBe(undefined);
  });

  it("reports a validation error raised many blocks earlier", async() => {
    const device = fakeDevice([null, null, { message: "buffer too small" }]);
    const validation = new DeferredValidation(device, "Evoformer stack");
    for (let block = 0; block < 3; block += 1) {
      validation.begin();
      validation.end(`block ${block}`);
    }
    await assert.rejects(validation.settle(), { message: "WebGPU Evoformer stack validation failed: block 2: buffer too small" });
  });

  it("names every block that failed, not just the first", async() => {
    const device = fakeDevice([{ message: "a" }, null, { message: "b" }]);
    const validation = new DeferredValidation(device, "stack");
    for (let block = 0; block < 3; block += 1) {
      validation.begin();
      validation.end(`block ${block}`);
    }
    await assert.rejects(validation.settle(), { message: "WebGPU stack validation failed: block 0: a; block 2: b" });
  });

  it("settles clean a second time, so a later recycle starts empty", async() => {
    const device = fakeDevice([{ message: "a" }]);
    const validation = new DeferredValidation(device, "stack");
    validation.begin();
    validation.end("block 0");
    await assert.rejects(validation.settle(), { message: "WebGPU stack validation failed: block 0: a" });
    await validation.settle();
  });

  it("resolves rather than rejects, so an abort elsewhere leaves no unhandled rejection", async() => {
    const device = fakeDevice([{ message: "a" }]);
    const validation = new DeferredValidation(device, "stack");
    validation.begin();
    validation.end("block 0");
    // The loop throws for an unrelated reason and settle() is never reached.
    await new Promise((resolve) => setTimeout(resolve, 0));
    // Nothing above may have produced an unhandled rejection; draining proves
    // the promise was already settled and carried its message as a value.
    await assert.rejects(validation.settle(), { message: "WebGPU stack validation failed: block 0: a" });
  });
});
