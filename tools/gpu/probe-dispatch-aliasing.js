/**
 * Does any dispatch bind the SAME buffer range twice?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-dispatch-aliasing.js \
 *       --tool=bench-af2-warm
 *
 * 🔴 WHY IT EXISTS. `GpuBufferAllocator` pools whole buffers by
 * `byteLength:usage` and a stack releases a block's allocations so the next
 * block can reuse them - which is deliberate and is where the memory saving
 * comes from. Reuse ACROSS dispatches is fine; two tensors of ONE dispatch
 * landing on one buffer is not, when one of them is the output: workgroups
 * then read what other workgroups are writing, which is a race whose symptom
 * is a fold that differs run to run and whose cause is nowhere near the shader
 * that shows it.
 *
 * This says whether that is happening, per label, rather than leaving it to be
 * argued about. A read-only pair sharing a buffer is reported too and is not
 * by itself wrong - the point is to see the aliasing at all.
 */
import { WebGpuExecution } from "../../src/runtime/execution.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const BYTES = { f32: 4, f16: 2 };

export async function main(device, args) {
  const tool = option(args, "tool", "bench-af2-warm");
  const rest = args.filter((a) => !a.startsWith("--tool="));

  const found = new Map();
  let dispatches = 0;
  const original = WebGpuExecution.prototype.dispatch;
  WebGpuExecution.prototype.dispatch = function watched(encoder, pipeline, tensors,
                                                       x, y, z, label) {
    dispatches += 1;
    const ranges = (tensors ?? []).map((tensor, binding) => {
      if (tensor?.allocation?.buffer === undefined) return null;
      const size = BYTES[tensor.storage ?? "f32"] ?? 4;
      const at = (tensor.offsetElements ?? 0) * size;
      return { binding, buffer: tensor.allocation.buffer,
               from: at, to: at + tensor.elements * size };
    }).filter(Boolean);
    for (let a = 0; a < ranges.length; a += 1) {
      for (let b = a + 1; b < ranges.length; b += 1) {
        if (ranges[a].buffer !== ranges[b].buffer) continue;
        if (ranges[a].to <= ranges[b].from || ranges[b].to <= ranges[a].from) continue;
        const name = String(label ?? "(unlabelled)");
        const row = found.get(name) ?? { label: name, occurrences: 0, pairs: new Set() };
        row.occurrences += 1;
        row.pairs.add(`${ranges[a].binding}+${ranges[b].binding}`
          + ` ${ranges[a].buffer.label ?? "?"}`);
        found.set(name, row);
      }
    }
    return original.call(this, encoder, pipeline, tensors, x, y, z, label);
  };

  try {
    const module = await import(`./${tool}.js`);
    const result = await module.main(device, rest);
    return {
      tool, dispatches,
      aliasingLabels: found.size,
      aliasing: [...found.values()].map((row) => ({
        label: row.label, occurrences: row.occurrences, pairs: [...row.pairs],
      })).sort((a, b) => b.occurrences - a.occurrences).slice(0, 20),
      tool_: Object.fromEntries(Object.entries(result ?? {})
        .filter(([, value]) => typeof value === "number" || typeof value === "boolean")),
    };
  } finally {
    WebGpuExecution.prototype.dispatch = original;
  }
}
