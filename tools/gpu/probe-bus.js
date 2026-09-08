/**
 * What the host-device bus actually costs, in each direction.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-bus.js
 *
 * WHY IT EXISTS. Every timing in docs/PERF.md was taken on an M2, where the
 * GPU reads the same physical memory the JavaScript heap is in. A
 * `writeBuffer` there is a copy between two addresses; on a discrete card it
 * is a PCIe transfer, and a `mapAsync` readback is that plus a fence. So a
 * structure this repository has priced as cheap - `bench-grid-attend.js`
 * uploading a pair representation per call, the diffusion head chaining its
 * four stages through `Float32Array`s - has a completely different price on a
 * discrete GPU, and nothing here could see it.
 *
 * Four arms, because they fail differently:
 *
 *   upload        writeBuffer into a STORAGE buffer
 *   readback      copyBufferToBuffer into MAP_READ, mapAsync, getMappedRange
 *   readback+copy ...and `.slice()` it into a JavaScript Float32Array, which
 *                 is what every caller here actually does
 *   device-device copyBufferToBuffer, which never leaves the card
 *
 * The last is the control: if it is fast and the others are not, the cost is
 * the bus and not the allocator.
 */
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];

export async function main(device, args) {
  const sizesMiB = option(args, "sizes", "1,8,32,78").split(",").map(Number);
  const rounds = Number(option(args, "rounds", "7"));

  const rows = [];
  for (const mib of sizesMiB) {
    const bytes = Math.round(mib * 1048576) & ~3;
    const floats = bytes / 4;
    const host = new Float32Array(floats);
    for (let i = 0; i < floats; i += 1024) host[i] = i;

    const target = device.createBuffer({
      size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const other = device.createBuffer({
      size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const staging = device.createBuffer({
      size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const upload = async () => {
      const start = performance.now();
      device.queue.writeBuffer(target, 0, host);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - start;
    };
    const deviceCopy = async () => {
      const start = performance.now();
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(target, 0, other, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - start;
    };
    const readback = async (copyOut) => {
      const start = performance.now();
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(target, 0, staging, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const view = staging.getMappedRange();
      // 🔴 THE `.slice()` IS NOT INCIDENTAL. A mapped range is only valid until
      // unmap, so every caller in this repository copies out of it, and that
      // copy is host memcpy on top of the transfer.
      const out = copyOut ? new Float32Array(view.slice(0)) : new Float32Array(view, 0, 4);
      staging.unmap();
      const ms = performance.now() - start;
      return out.length >= 0 ? ms : ms;
    };

    // 🔴 mappedAtCreation WRITES STRAIGHT INTO THE STAGING ALLOCATION.
    // `writeBuffer` copies the caller's array into a staging buffer the
    // implementation owns and then schedules a copy; `mappedAtCreation` hands
    // the caller that staging memory directly, so the first copy does not
    // happen. Every upload in this repository goes through writeBuffer.
    const mappedAtCreation = async () => {
      const start = performance.now();
      const staged = device.createBuffer({
        size: bytes, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
      new Float32Array(staged.getMappedRange()).set(host);
      staged.unmap();
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(staged, 0, target, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const ms = performance.now() - start;
      staged.destroy();
      return ms;
    };
    // ...and the same without the memcpy, to price the two halves apart.
    const mappedNoCopy = async () => {
      const start = performance.now();
      const staged = device.createBuffer({
        size: bytes, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
      staged.getMappedRange();
      staged.unmap();
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(staged, 0, target, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const ms = performance.now() - start;
      staged.destroy();
      return ms;
    };

    const arms = {
      upload: upload,
      "upload, mappedAtCreation": mappedAtCreation,
      "...without the memcpy": mappedNoCopy,
      "device-device copy": deviceCopy,
      readback: () => readback(false),
      "readback + slice": () => readback(true),
    };

    for (const run of Object.values(arms)) await run();
    const times = {};
    for (const [name, run] of Object.entries(arms)) {
      const samples = [];
      for (let round = 0; round < rounds; round += 1) samples.push(await run());
      const ms = median(samples);
      times[name] = { ms: Number(ms.toFixed(2)), gbPerSecond: Number((bytes / ms / 1e6).toFixed(2)) };
    }
    rows.push({ mib, times });
    target.destroy(); other.destroy(); staging.destroy();
  }
  return { rounds, rows };
}
