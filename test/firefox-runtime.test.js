// Two things a Firefox device needs that no feature states (docs/FIREFOX.md): who it is, since its adapter.info is
// blank, and its waits pumped, since it reports finished GPU work on a 100 ms tick.
import test from "node:test";
import assert from "node:assert/strict";
import { identityOfUnnamedAdapter } from "../webgpu/runtime/device-profile.js";
import { startCompletionPump, completionPumpOf } from "../webgpu/runtime/completion-pump.js";

test("an unnamed adapter is Apple silicon only on a Mac that offers ASTC", () => {
  const apple = ["shader-f16", "texture-compression-astc", "texture-compression-bc", "texture-compression-etc2"];
  assert.deepEqual(identityOfUnnamedAdapter(apple, "MacIntel"), { vendor: "apple", architecture: "metal-3", inferred: true });
  assert.deepEqual(identityOfUnnamedAdapter(new Set(apple), "MacIntel").vendor, "apple");
  // an Intel Mac's GPUs have BC and no ASTC; a phone and a Windows-on-Arm part have ASTC and are not Macs
  assert.equal(identityOfUnnamedAdapter(["shader-f16", "texture-compression-bc"], "MacIntel"), null);
  assert.equal(identityOfUnnamedAdapter(apple, "Linux armv8l"), null);
  assert.equal(identityOfUnnamedAdapter(apple, "Win32"), null);
  assert.equal(identityOfUnnamedAdapter(apple, undefined), null);
  assert.equal(identityOfUnnamedAdapter(undefined, "MacIntel"), null);
});

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_DST: 8, STORAGE: 128 };

/** A device whose waits resolve `delay` ms after they are asked, or at the first empty submit after `work` ms. */
function fakeDevice({ delay, pokeable = false }) {
  const submits = [];
  const waiting = new Set();
  const wait = () => new Promise((resolve) => {
    const entry = { resolve, at: performance.now() };
    waiting.add(entry);
    setTimeout(() => { if (waiting.delete(entry)) resolve(); }, delay);
  });
  const queue = {
    submit: (buffers) => {
      submits.push(buffers.length);
      if (!pokeable || buffers.length !== 0) return;
      for (const entry of [...waiting]) { if (performance.now() - entry.at >= 2) { waiting.delete(entry); entry.resolve(); } }
    },
    onSubmittedWorkDone: wait,
  };
  return {
    queue, submits,
    createCommandEncoder: () => ({ finish: () => ({}) }),
    createBuffer: (descriptor) => ({ descriptor, mapAsync: wait }),
  };
}

test("a device that answers promptly is left exactly as it was", async () => {
  const device = fakeDevice({ delay: 0 });
  const { onSubmittedWorkDone } = device.queue;
  const { createBuffer } = device;
  const state = await startCompletionPump(device);
  assert.equal(state.pumped, false);
  assert.equal(device.queue.onSubmittedWorkDone, onSubmittedWorkDone);
  assert.equal(device.createBuffer, createBuffer);
  assert.equal(completionPumpOf(device).pumped, false);
});

test("a device on a tick has both waits pumped with empty submits, and the pump stops with the last wait", async () => {
  const device = fakeDevice({ delay: 120, pokeable: true });
  const state = await startCompletionPump(device);
  assert.equal(state.pumped, true);
  assert.ok(state.delayMs >= 30);
  const before = device.submits.length;
  const start = performance.now();
  await device.queue.onSubmittedWorkDone();
  const mapped = device.createBuffer({ usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  await mapped.mapAsync();
  const elapsed = performance.now() - start;
  assert.ok(elapsed < 100, `two waits took ${elapsed} ms`);
  assert.ok(device.submits.slice(before).some((count) => count === 0));
  // a buffer that cannot be mapped is not wrapped
  const plain = device.createBuffer({ usage: GPUBufferUsage.STORAGE });
  assert.equal(plain.mapAsync.name, "wait");
  assert.notEqual(mapped.mapAsync.name, "wait");
  const settled = device.submits.length;
  await new Promise((resolve) => setTimeout(resolve, 30));
  assert.equal(device.submits.length, settled);
});

test("`force` decides without measuring", async () => {
  const slow = fakeDevice({ delay: 120 });
  assert.equal((await startCompletionPump(slow, { force: false })).pumped, false);
  const fast = fakeDevice({ delay: 0 });
  assert.deepEqual(await startCompletionPump(fast, { force: true }), { pumped: true, delayMs: null });
});
