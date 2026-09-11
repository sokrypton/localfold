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
    execution.destroy?.();
    const perBinding = Math.floor(limit / 4);
    const windowElements = Math.floor(perBinding / 64) * 64;
    results.push({ residues, gibibytes: Math.round(bytes / 2 ** 30 * 100) / 100,
                   windows: Math.ceil(elements / windowElements),
                   expect, outcome });
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
