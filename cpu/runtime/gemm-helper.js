/**
 * One helper thread of a CpuBackend: wakes on a job, takes chunks until none are
 * left, and sleeps. Runs in a browser Worker or a node worker thread.
 *
 * It must see every job exactly once, because the caller counts it out of ACTIVE.
 */
import { CONTROL, runJobChunks } from "./backend-job.js";

async function start({ memory, module }) {
  const instance = await WebAssembly.instantiate(module, { env: { memory } });
  const { exports } = instance;
  const control = new Int32Array(memory.buffer, CONTROL.offset, CONTROL.length);
  let seen = 0;
  Atomics.add(control, CONTROL.ready, 1);
  Atomics.notify(control, CONTROL.ready);
  for (;;) {
    Atomics.wait(control, CONTROL.sequence, seen);
    const sequence = Atomics.load(control, CONTROL.sequence);
    if (sequence === seen) continue;
    seen = sequence;
    if (Atomics.load(control, CONTROL.stop) !== 0) return;
    runJobChunks(control, exports);
    if (Atomics.sub(control, CONTROL.active, 1) === 1) Atomics.notify(control, CONTROL.active);
  }
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = (event) => { void start(event.data).finally(() => self.close()); };
} else {
  // ...null on node's main thread, where importing this file does nothing.
  const { parentPort } = await import("node:worker_threads");
  parentPort?.once("message", (message) => { void start(message).finally(() => process.exit(0)); });
}
