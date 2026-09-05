/**
 * Does a whole diffusion transformer block decode the same on the GPU?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-block-upload.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-block-upload.js --blocks=24
 *
 * tools/gpu/check-int5-gpu.js asks whether the ARITHMETIC agrees, on synthetic
 * codes. This asks whether the whole path does, on the real model: the same ~40
 * tensors, the same packed layout, the same offsets - against
 * packBlockWeights, which is what ships.
 *
 * 🔴 THE COMPARISON IS ON BITS, NOT ON A TOLERANCE. Both sides produce f16, and
 * a weight buffer that is nearly right is a fold that is nearly right in a way
 * no bound would catch. If this ever reports a difference, the GPU path is
 * wrong; there is no acceptable non-zero here.
 */
import { openAf3Store, SOURCES } from "../../src/af3/weights.js";
import { diffusionWeights } from "../../src/af3/diffusion-weights.js";
import { BLOCK_ORDER, packBlockWeights }
  from "../../src/af3/diffusion-transformer-webgpu.js";
import { planBlockUpload, runBlockUpload } from "../../src/runtime/quantised-upload.js";
import { writeInto } from "../../src/runtime/float16.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const weights = await diffusionWeights(store);
  const blocks = weights.transformer.superBlocks.flatMap((group) => group.blocks);
  const wanted = Number(option(args, "blocks", "4"));

  const rows = [];
  let worstDiffering = 0;
  for (let index = 0; index < Math.min(wanted, blocks.length); index += 1) {
    const block = blocks[index];
    // 🔴 THE LENGTHS COME FROM THE THUNKS, NOT FROM THE TENSORS. Reading
    // `block[name].length` materialises that tensor, which is the decode this
    // whole path exists to avoid - and it would make the host arm below WARM,
    // so the comparison would flatter the GPU by ~250 ms of work it had already
    // caused. `stacked` records the range it will read; that is the length.
    const sources = block[SOURCES];
    const entries = [];
    let running = 0;
    for (const name of BLOCK_ORDER) {
      const thunk = sources[name];
      entries.push({ name, thunk, offset: running, length: thunk.count });
      running += thunk.count;
    }
    const total = running;
    const planned = planBlockUpload(entries);
    if (planned === undefined) throw new Error(`block ${index} cannot be planned`);

    const words = Math.ceil(total / 2);
    const destination = device.createBuffer({
      size: words * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // The float32 minority, narrowed the ordinary way - see planBlockUpload.
    for (const entry of planned.host) {
      const half = new globalThis.Float16Array(entry.length);
      writeInto(half, block[entry.name], 0);
      device.queue.writeBuffer(destination, entry.offset * 2,
        half.buffer, half.byteOffset, half.byteLength);
    }
    const started = performance.now();
    const release = await runBlockUpload(device, planned.gpu, destination);
    await device.queue.onSubmittedWorkDone();
    const gpuMs = performance.now() - started;

    const readback = device.createBuffer({ size: words * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(destination, 0, readback, 0, words * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const gpuBits = new Uint16Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const hostStarted = performance.now();
    const hostBits = packBlockWeights(block, "f16").data;
    const hostMs = performance.now() - hostStarted;
    let differing = 0;
    let firstAt;
    for (let at = 0; at < total; at += 1) {
      if (hostBits[at] === gpuBits[at]) continue;
      differing += 1;
      firstAt ??= at;
    }
    worstDiffering = Math.max(worstDiffering, differing);
    rows.push({ block: index, elements: total, hostTensors: planned.host.length, differing,
                firstAt, gpuMs: Number(gpuMs.toFixed(1)), hostMs: Number(hostMs.toFixed(1)) });
    console.log(`block ${index}\t${total} elements\t${planned.host.length} on the host`
      + `\t${differing} differ\tgpu ${gpuMs.toFixed(0)} ms\thost ${hostMs.toFixed(0)} ms`);
    release();
    destination.destroy();
    readback.destroy();
  }
  if (worstDiffering > 0) throw new Error(`${worstDiffering} elements differ from packBlockWeights`);
  const sum = (key) => Number(rows.reduce((total, row) => total + row[key], 0).toFixed(0));
  // 🔴 BOTH COLUMNS ARE COLD, and keeping them that way is the whole reason the
  // plan reads lengths from the thunks. The GPU arm never decodes; the host arm
  // decodes exactly once, which is what a first fold pays.
  return { blocks: rows.length, gpuMsTotal: sum("gpuMs"), hostMsTotal: sum("hostMs"), rows };
}
