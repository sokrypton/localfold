/**
 * What does THIS backend do with an invocation that indexes past the buffer?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-grid-overdispatch.js
 *
 * 🔴 WHY IT EXISTS. `execution.linearGrid` rounds a dispatch up to whole
 * workgroups and then to whole rows of y, so a kernel that indexes
 * `id.x + id.y * GRID_WIDTH * 64` is handed invocations past the end of its
 * buffer whenever the element count is not a multiple of `GRID_WIDTH * 64`.
 * Over the pair tensor at 128 channels that is **zero** up to 128 residues -
 * 128*128*128 is exactly 2,097,152, one full row - and **917,504** at 160.
 *
 * An M2 folds a different structure every pass above 128 residues because
 * Metal CLAMPS those writes onto the last element: ~917,504 non-atomic `+=` on
 * one address. This A100 does not, and the difference is the backend, not the
 * code - so it is worth measuring rather than assuming, on any machine this
 * repository is asked to run on.
 *
 * It drives the SHIPPED `ADD_IN_PLACE_SHADER`, not a copy, for the reason
 * check-quantised-upload.js drives the shipped planner.
 *
 * What it reports:
 *
 *  - `tail` is the last IN-RANGE element after one dispatch that over-dispatches
 *    by `overDispatch` invocations. Every element should have had exactly one
 *    `+= 1`, so **1.0 is a backend that discards the out-of-range writes** and
 *    anything larger is one that folds them onto the tail.
 *  - `verdict` says which, in a word.
 *
 * 🔴 A CLAMPING BACKEND WILL NOT GIVE A STABLE NUMBER. Hundreds of thousands of
 * non-atomic read-modify-writes on one address is a race, so expect a different
 * value each run there - which is the point, and why `runs` repeats it.
 */
import { ADD_IN_PLACE_SHADER } from "../../src/runtime/execution.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const GRID_WIDTH = 32768;
const WORKGROUP = 64;

export async function main(device, args) {
  // The default is the pair tensor at 160 residues and 128 channels, which is
  // the shape the M2's race first appears at.
  const elements = Number(option(args, "elements", String(160 * 160 * 128)));
  const runs = Number(option(args, "runs", "3"));
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new RangeError(`--elements wants a positive integer, got ${elements}`);
  }

  const groups = Math.ceil(elements / WORKGROUP);
  const x = Math.min(groups, GRID_WIDTH);
  const y = Math.ceil(groups / GRID_WIDTH);
  const reached = x * WORKGROUP + (y - 1) * GRID_WIDTH * WORKGROUP;
  const overDispatch = reached - elements;

  const module = device.createShaderModule({ code: ADD_IN_PLACE_SHADER });
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module, entryPoint: "main" },
  });

  const bytes = elements * 4;
  const base = device.createBuffer({
    size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const update = device.createBuffer({
    size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const readback = device.createBuffer({
    size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  // `update` is all ones, so after one dispatch every in-range element is 1
  // exactly - and the tail is 1 + however many out-of-range writes landed on it.
  const ones = new Float32Array(65536).fill(1);
  for (let at = 0; at < elements; at += ones.length) {
    const count = Math.min(ones.length, elements - at);
    device.queue.writeBuffer(update, at * 4, ones, 0, count);
  }

  const tails = [];
  for (let run = 0; run < runs; run += 1) {
    device.queue.writeBuffer(base, 0, new Float32Array(Math.min(elements, 65536)));
    // Only the tail has to start at zero; the rest is written above and read
    // back nowhere. Zero the last words explicitly.
    device.queue.writeBuffer(base, (elements - 1) * 4, new Float32Array(1));
    const encoder = device.createCommandEncoder({ label: "overdispatch" });
    const pass = encoder.beginComputePass({ label: "overdispatch" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: base } },
                { binding: 1, resource: { buffer: update } }],
    }));
    pass.dispatchWorkgroups(x, y);
    pass.end();
    encoder.copyBufferToBuffer(base, (elements - 1) * 4, readback, 0, 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    tails.push(new Float32Array(readback.getMappedRange().slice(0, 4))[0]);
    readback.unmap();
  }

  base.destroy(); update.destroy(); readback.destroy();
  const clamps = tails.some((value) => value !== 1);
  const stable = tails.every((value) => value === tails[0]);
  return {
    elements, grid: `${x}x${y}`, overDispatch,
    tail: tails,
    // 🔴 THE WHOLE POINT IN ONE FIELD.
    verdict: overDispatch === 0
      ? "nothing over-dispatched at this size, so this run proves nothing"
      : clamps
        ? `CLAMPS: ${overDispatch} out-of-range writes landed on the last element`
        : "discards out-of-range writes, so a missing bounds check is invisible here",
    stableAcrossRuns: stable,
  };
}
