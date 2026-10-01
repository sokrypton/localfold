/**
 * What the GPU driver holds beyond the buffers WebGPU has live.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-driver-memory.js
 *     (sample `nvidia-smi --query-gpu=memory.used -lms 100` beside it)
 *
 * probe-live-buffers.js found a fold's live buffers equal to what the budget
 * counts, while the driver reported about twice that. This isolates the two
 * suspects, each step printed with a timestamp to line up against the driver's
 * counter: (a) a large `writeBuffer`, which Dawn stages through upload memory
 * of its own, and (b) a buffer created and destroyed again and again, whose
 * memory Dawn frees only once the queue is past the work that used it.
 */
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

export async function main(device, args) {
  const mib = Number(args.find((a) => a.startsWith("--mib="))?.slice(6) ?? "1024");
  const marks = [];
  const mark = (name) => { marks.push([name, Date.now()]); };
  await sleep(1500); mark("start");

  // (a) one big buffer, filled by writeBuffer in 64 MiB pieces.
  const big = device.createBuffer({ size: mib * 1048576, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  mark("created");
  await sleep(1500);
  const piece = new Uint8Array(64 * 1048576);
  for (let at = 0; at < mib; at += 64) device.queue.writeBuffer(big, at * 1048576, piece);
  mark("written");
  await device.queue.onSubmittedWorkDone();
  mark("upload-done");
  await sleep(2000); mark("idle-after-upload");
  big.destroy(); mark("destroyed");
  await device.queue.onSubmittedWorkDone();
  await sleep(2000); mark("idle-after-destroy");

  // (b) create, use and destroy a pair-sized buffer repeatedly, as a fold's
  // per-block scratch does when it is not pooled.
  for (let round = 0; round < 8; round += 1) {
    const scratch = device.createBuffer({ size: 512 * 1048576, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(scratch, 0, piece.subarray(0, 1048576));
    scratch.destroy();
  }
  mark("churned");
  await device.queue.onSubmittedWorkDone();
  await sleep(2000); mark("idle-after-churn");

  // (c) and whether that memory is REUSED: four live pair-sized buffers now.
  const held = [];
  for (let n = 0; n < 4; n += 1) {
    const buffer = device.createBuffer({ size: 512 * 1048576, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, piece.subarray(0, 1048576));
    held.push(buffer);
  }
  await device.queue.onSubmittedWorkDone();
  await sleep(2000); mark("4-live-after-churn");
  for (const buffer of held) buffer.destroy();
  await device.queue.onSubmittedWorkDone();
  await sleep(2000); mark("released");

  // (d) the same, but with the queue kept busy afterwards: Dawn frees a
  // destroyed buffer's memory on a device tick, and ticks follow queue work.
  const busy = device.createBuffer({ size: 1048576, usage: GPUBufferUsage.COPY_DST });
  for (let n = 0; n < 200; n += 1) {
    device.queue.writeBuffer(busy, 0, piece.subarray(0, 1024));
    device.queue.submit([device.createCommandEncoder().finish()]);
    await device.queue.onSubmittedWorkDone();
    await sleep(10);
  }
  mark("after-busy-queue");
  return { mib, marks };
}
