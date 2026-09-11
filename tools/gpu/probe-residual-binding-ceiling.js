/**
 * How long a chain can `addInPlace` still bind, and what happens past it.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-residual-binding-ceiling.js
 *
 * 🔴 THE RESIDUAL ADD BINDS BOTH TENSORS WHOLE. The pair is the larger, and at
 * `cZ` 128 in f32 it is `L * L * 512` bytes against `maxStorageBufferBinding
 * Size` - 2 GiB on this A100 and on most cards, a limit no request raises
 * because it is Vulkan's `maxStorageBufferRange`. So the ceiling is
 * `sqrt(2 GiB / 512)` = **2047 residues**, whatever memory the card has: this
 * one has 40 GB and cannot bind a 2,048-residue pair.
 *
 * 🔴 CREDIT WHERE IT IS DUE: @milot-mirdita hit this first in
 * martin-steinegger/alphafold2-webgpu ("Window the residual add, so a complex
 * past 2,896 residues runs at all"), where an ordinary tetramer died in the
 * multimer template pair update with a 13.1 GB plan on a 97 GB card. Their
 * ceiling is 2,896 because their pair is PACKED f16 at two bytes a channel;
 * ours is f32, so ours is lower. Read for the idea, measured here
 * independently - their repository carries no licence and nothing is copied
 * from it.
 *
 * This demonstrates the failure rather than computing it, because a ceiling
 * nobody has crossed is arithmetic and not a bug report.
 */
import { WebGpuExecution } from "../../src/runtime/execution.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const channels = Number(option(args, "channels", "128"));
  const limit = device.limits.maxStorageBufferBindingSize;
  const bytesPerResidueSquared = channels * 4;
  const ceiling = Math.floor(Math.sqrt(limit / bytesPerResidueSquared));

  // Just under and just over, so the pass says the probe works and the failure
  // says where it is.
  const extra = option(args, "residues", "").split(",").filter(Boolean).map(Number);
  const arms = [
    { residues: ceiling, expect: "one binding" },
    { residues: ceiling + 1, expect: "windowed" },
    ...extra.map((residues) => ({ residues, expect: "windowed" })),
  ];
  const results = [];
  for (const { residues, expect } of arms) {
    const elements = residues * residues * channels;
    const bytes = elements * 4;
    const execution = new WebGpuExecution(device);
    let outcome;
    let values;
    try {
      const base = execution.allocate("probe.pair", elements);
      const update = execution.allocate("probe.update", elements);
      device.pushErrorScope("validation");
      const encoder = device.createCommandEncoder({ label: "probe.residual" });
      await execution.addInPlace(encoder, base, update, "probe.residual");
      execution.endComputePass(encoder);
      device.queue.submit([encoder.finish()]);
      const error = await device.popErrorScope();
      outcome = error === null ? "ok" : `REFUSED: ${error.message.split("\n")[0]}`;
    } catch (error) {
      outcome = `THREW: ${String(error.message ?? error).split("\n")[0]}`;
    }
    // 🔴 "ACCEPTED" IS NOT "CORRECT". The first version of this probe checked
    // only that the dispatch validated, which a windowed add can do while
    // computing the wrong thing - a window that starts at the wrong offset, or
    // a last workgroup that runs into the next window, both validate. So the
    // VALUES are checked at the boundaries that windowing creates: the first
    // element, the last of window 0, the first of window 1 and the last of all.
    if (outcome === "ok") {
      try {
        const perBinding = Math.floor(limit / 4);
        const w = Math.floor(perBinding / 64) * 64;
        const marks = [...new Set([0, w - 1, w, elements - 1]
          .filter((at) => at >= 0 && at < elements))].sort((a, b) => a - b);
        const rw = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        const base = execution.allocate("probe.check.base", elements, rw);
        const update = execution.allocate("probe.check.update", elements, rw);
        // base = 10 at every mark, update = 1 everywhere those marks live.
        const one = new Float32Array([1]);
        const ten = new Float32Array([10]);
        for (const at of marks) {
          device.queue.writeBuffer(base.allocation.buffer, at * 4, ten);
          device.queue.writeBuffer(update.allocation.buffer, at * 4, one);
        }
        const encoder = device.createCommandEncoder({ label: "probe.check" });
        await execution.addInPlace(encoder, base, update, "probe.check");
        execution.endComputePass(encoder);
        const readback = device.createBuffer({
          size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        marks.forEach((at, index) =>
          encoder.copyBufferToBuffer(base.allocation.buffer, at * 4, readback, index * 4, 4));
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const got = [...new Float32Array(readback.getMappedRange().slice(0, marks.length * 4))];
        readback.unmap(); readback.destroy();
        const wrong = marks.map((at, index) => [at, got[index]]).filter(([, v]) => v !== 11);
        values = { marks, got, correct: wrong.length === 0,
                   ...(wrong.length === 0 ? {} : { wrong }) };
      } catch (error) {
        values = { error: String(error.message ?? error).split("\n")[0] };
      }
    }
    execution.destroy?.();
    const perBinding = Math.floor(limit / 4);
    const windowElements = Math.floor(perBinding / 64) * 64;
    results.push({ residues, gibibytes: Math.round(bytes / 2 ** 30 * 100) / 100,
                   windows: Math.ceil(elements / windowElements),
                   expect, outcome, values });
  }

  return {
    maxStorageBufferBindingSize: limit,
    channels,
    // The longest chain whose f32 pair tensor still binds as ONE range.
    ceilingResidues: ceiling,
    packedF16WouldReach: Math.floor(Math.sqrt(limit / (channels * 2))),
    arms: results,
  };
}
