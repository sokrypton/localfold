/**
 * Getting a browser to say that GPU work is done, where it is slow to.
 *
 * 🔴 FIREFOX ANSWERS `mapAsync` AND `onSubmittedWorkDone` ON A 100 ms TICK. Measured on Firefox 160 on an M2
 * (tools/gpu/probe-readback-latency.js): a copy of 256 bytes awaited either way is 99-101 ms, ten times of ten,
 * where Chrome is 0.2 ms. A fold waits for a readback every sampler step, so 25 steps were 2.5 s of waiting on work
 * that had finished - most of an AlphaFold 3 fold there, whatever the kernels cost. The device looks for finished
 * work when something is SUBMITTED, so an empty submit while a wait is open brings the answer in 2-8 ms; writing a
 * buffer, creating one, or asking `onSubmittedWorkDone` again does not, and one empty submit straight after the ask
 * does not either (the work has not finished yet).
 *
 * `startCompletionPump` measures the delay once and, where it is long, wraps the two waits of this device so that an
 * empty submit goes out every half millisecond while any is open. Nothing is wrapped in a browser that answers promptly.
 */

/** A wait longer than this for work that takes none is a browser on a tick. */
const SLOW_MILLISECONDS = 30;

/** An empty submit this often while a wait is open. */
const SUBMIT_SPACING_MILLISECONDS = 0.5;

/** Milliseconds this device takes to report an empty submit done: the smaller of two asks. */
export async function completionDelay(device) {
  let best = Infinity;
  for (let attempt = 0; attempt < 2 && best >= SLOW_MILLISECONDS; attempt += 1) {
    device.queue.submit([device.createCommandEncoder().finish()]);
    const start = performance.now();
    await device.queue.onSubmittedWorkDone();
    best = Math.min(best, performance.now() - start);
  }
  return best;
}

const PUMPED = new WeakMap();

/** Whether this device's waits are pumped, and the delay that decided it. */
export const completionPumpOf = (device) => PUMPED.get(device) ?? { pumped: false, delayMs: null };

/**
 * @param {GPUDevice} device
 * @param {{force?: boolean}} [options] `force: true` pumps without measuring, `force: false` never pumps
 */
export async function startCompletionPump(device, options = {}) {
  if (PUMPED.has(device)) return PUMPED.get(device);
  const delayMs = options.force === undefined ? await completionDelay(device) : null;
  const pumped = options.force ?? delayMs >= SLOW_MILLISECONDS;
  PUMPED.set(device, { pumped, delayMs });
  if (!pumped) return PUMPED.get(device);

  const queue = device.queue;
  // 🔴 THE TURNS ARE MESSAGES, NOT TIMERS. A timer turn is 4 ms at best and, in a window that is not in front,
  // whatever the browser allows it: the same probe's timer arm read 5 to 682 ms, and a fold's sampler went 1.0 to
  // 2.5 s from run to run with it. A message turn is not slowed, so the submits are spaced by the clock instead -
  // one every half millisecond while a wait is open, and the loop stops with the last wait.
  let open = 0;
  let running = false;
  let last = -1;
  const channel = new MessageChannel();
  channel.port1.onmessage = () => {
    if (open === 0) { running = false; return; }
    const now = performance.now();
    if (now - last >= SUBMIT_SPACING_MILLISECONDS) {
      last = now;
      try { queue.submit([]); } catch { open = 0; running = false; return; }   // a lost device has nothing to report
    }
    channel.port2.postMessage(0);
  };
  // (an open port would keep a Node process alive; a page has no such thing to keep)
  channel.port1.unref?.();
  channel.port2.unref?.();
  const turn = () => channel.port2.postMessage(0);
  const during = (promise) => {
    open += 1;
    if (!running) { running = true; turn(); }
    const close = () => { open = Math.max(0, open - 1); };
    promise.then(close, close);
    return promise;
  };

  const workDone = queue.onSubmittedWorkDone.bind(queue);
  queue.onSubmittedWorkDone = () => during(workDone());
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = createBuffer(descriptor);
    if ((descriptor.usage & (GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE)) !== 0) {
      const mapAsync = buffer.mapAsync.bind(buffer);
      buffer.mapAsync = (...args) => during(mapAsync(...args));
    }
    return buffer;
  };
  return PUMPED.get(device);
}
