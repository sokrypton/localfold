// How long this browser takes to tell a page that GPU work is done: a trivial submit awaited three ways
// (onSubmittedWorkDone, mapAsync of a buffer the submit copied into, and both), each ten times.
//   node tools/gpu-chrome.mjs tools/gpu/probe-readback-latency.js
//   LOCALFOLD_GPU_FIREFOX=1 LOCALFOLD_GPU_NO_PUMP=1 node tools/gpu-chrome.mjs tools/gpu/probe-readback-latency.js
// A fold awaits a readback every sampler step, so this is a floor under the step whatever the kernels cost. Firefox
// is 100 ms a wait where Chrome is 0.2; the `poke` arms are what a page can do while it waits, and an empty submit
// is the one that works - which is webgpu/runtime/completion-pump.js. LOCALFOLD_GPU_NO_PUMP=1 gives the browser's
// own waits; without it the harness's device is already pumped and the first three arms read that.
export async function main(device) {
  const source = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const submitCopy = (target) => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, target, 0, 256);
    device.queue.submit([encoder.finish()]);
  };
  const ways = {
    workDone: async (target) => { submitCopy(target); await device.queue.onSubmittedWorkDone(); },
    mapAsync: async (target) => { submitCopy(target); await target.mapAsync(GPUMapMode.READ); target.unmap(); },
    both: async (target) => {
      submitCopy(target);
      await Promise.all([device.queue.onSubmittedWorkDone(), target.mapAsync(GPUMapMode.READ)]);
      target.unmap();
    },
  };
  // The same mapAsync with the page doing something while it waits, to see what makes the browser look sooner.
  // A turn is a 1 ms timer and a wait gives up after two seconds: a tight message loop starved the very task that
  // resolves the map in Firefox, and the probe never returned.
  const turn = () => new Promise((resolve) => setTimeout(resolve, 1));
  const waited = (poke) => async (target) => {
    submitCopy(target);
    let done = false;
    const mapped = target.mapAsync(GPUMapMode.READ).then(() => { done = true; });
    const start = performance.now();
    while (!done && performance.now() - start < 2000) { poke(); await turn(); }
    await mapped;
    target.unmap();
  };
  const scratch = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  // One empty submit straight after asking, and nothing more.
  ways.oneEmptySubmit = async (target) => {
    submitCopy(target);
    const mapped = target.mapAsync(GPUMapMode.READ);
    device.queue.submit([]);
    await mapped;
    target.unmap();
  };
  // An empty submit a turn, the turns being messages rather than timers (a nested timer is clamped to 4 ms).
  ways.pokeEmptySubmitFast = async (target) => {
    submitCopy(target);
    let done = false;
    const mapped = target.mapAsync(GPUMapMode.READ).then(() => { done = true; });
    const channel = new MessageChannel();
    const start = performance.now();
    await new Promise((resolve) => {
      channel.port1.onmessage = () => {
        if (done || performance.now() - start > 2000) { resolve(); return; }
        device.queue.submit([]);
        // every eighth turn is a timer, so the task that resolves the map is never starved
        if ((channel.count = (channel.count ?? 0) + 1) % 8 === 0) setTimeout(() => channel.port2.postMessage(0), 0);
        else channel.port2.postMessage(0);
      };
      channel.port2.postMessage(0);
    });
    await mapped;
    target.unmap();
  };
  // Message turns alone, an empty submit at most every half millisecond: no timer anywhere, since a window that is
  // not in front has its timers slowed.
  ways.pokeMessagesOnly = async (target) => {
    submitCopy(target);
    let done = false;
    const mapped = target.mapAsync(GPUMapMode.READ).then(() => { done = true; });
    const channel = new MessageChannel();
    const start = performance.now();
    let last = -1;
    await new Promise((resolve) => {
      channel.port1.onmessage = () => {
        const now = performance.now();
        if (done || now - start > 2000) { resolve(); return; }
        if (now - last >= 0.5) { device.queue.submit([]); last = now; }
        channel.port2.postMessage(0);
      };
      channel.port2.postMessage(0);
    });
    channel.port1.close();
    await mapped;
    target.unmap();
  };
  ways.spin = waited(() => {});
  ways.pokeEmptySubmit = waited(() => device.queue.submit([]));
  ways.pokeWriteBuffer = waited(() => device.queue.writeBuffer(scratch, 0, new Uint32Array(4)));
  ways.pokeWorkDone = waited(() => { device.queue.onSubmittedWorkDone(); });
  ways.pokeCreateBuffer = waited(() => device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE }).destroy());
  if (typeof document !== "undefined") {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 4;
    document.body.appendChild(canvas);
    const context = canvas.getContext("webgpu");
    context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat() });
    ways.pokePresent = waited(() => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(),
        loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
      pass.end();
      device.queue.submit([encoder.finish()]);
    });
  }
  const out = {};
  for (const [name, way] of Object.entries(ways)) {
    const target = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    await way(target);
    const times = [];
    for (let i = 0; i < 10; i += 1) { const start = performance.now(); await way(target); times.push(performance.now() - start); }
    times.sort((a, b) => a - b);
    out[name] = { minMs: times[0], medianMs: times[5], maxMs: times[9] };
  }
  return out;
}
