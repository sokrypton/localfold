/**
 * How many `addInPlace` dispatches a fold actually issues, and under which
 * label - counted on the device rather than read off the source.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-add-in-place.js \
 *       --tool=bench-af2-warm --length=160 --rows=128
 *
 * 🔴 SEVEN CALL SITES EXIST AND FIVE ARE CONDITIONAL. The outer product mean
 * writes straight into the pair tensor and RETURNS it on the outer-first path,
 * and every caller is guarded `if (update !== pair)` - so the OPM residual
 * fires only when `useOuterFirstContraction` is FALSE, which is
 * `sequences < cOuter`, an alignment shallower than 32 rows. The template
 * residual is unconditional. Reading that off the source got it wrong once in
 * each direction, which is why this counts.
 *
 * It matters because that shader had no bounds check under a folded grid, and
 * the number of times it lands on the pair tensor is the difference between one
 * corrupted add a fold and forty-nine.
 */
import { WebGpuExecution } from "../../src/runtime/execution.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const tool = option(args, "tool", "bench-af2-warm");
  const rest = args.filter((a) => !a.startsWith("--tool="));

  const byLabel = new Map();
  const original = WebGpuExecution.prototype.addInPlace;
  WebGpuExecution.prototype.addInPlace = function counted(encoder, base, update, label) {
    const name = String(label ?? "(unlabelled)").replace(/-\d+$/, "-N");
    const row = byLabel.get(name) ?? { calls: 0, elements: 0 };
    row.calls += 1;
    row.elements = base.elements;
    byLabel.set(name, row);
    return original.call(this, encoder, base, update, label);
  };

  try {
    const module = await import(`./${tool}.js`);
    const result = await module.main(device, rest);
    const GRID_WIDTH = 32768;
    const rows = [...byLabel.entries()].map(([label, row]) => {
      // The over-dispatch this label's grid carries, which is what the missing
      // bounds check turned into corruption on a clamping backend.
      const groups = Math.ceil(row.elements / 64);
      const y = Math.ceil(groups / GRID_WIDTH);
      const reached = Math.min(groups, GRID_WIDTH) * 64 + (y - 1) * GRID_WIDTH * 64;
      return { label, calls: row.calls, elements: row.elements,
               overDispatch: reached - row.elements };
    }).sort((a, b) => b.calls - a.calls);
    return {
      tool,
      totalCalls: rows.reduce((sum, row) => sum + row.calls, 0),
      byLabel: rows,
      tool_: Object.fromEntries(Object.entries(result ?? {})
        .filter(([, value]) => typeof value === "number" || typeof value === "boolean")),
    };
  } finally {
    WebGpuExecution.prototype.addInPlace = original;
  }
}
