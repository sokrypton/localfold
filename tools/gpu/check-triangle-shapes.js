/**
 * The triangle multiplication against its CPU reference, across SHAPES.
 *
 * check-triangle.js runs one fixture - OpenFold's, at length 5, cZ 7,
 * cHidden 6 - and fails on it while the three AlphaFold fixtures at cZ 128
 * pass. This sweeps the three extents to say which one the kernel is wrong in.
 */
import { createDeterministicTriangleInput } from "../../src/testing/deterministic-input.js";
import { triangleMultiplicationOutgoingReference } from "../../src/triangle/cpu-reference.js";
import { errorMetrics } from "../../src/triangle/types.js";
import { TriangleMultiplicationOutgoingGpu } from "../../src/triangle/webgpu.js";

export async function main(device, args = []) {
  const runner = new TriangleMultiplicationOutgoingGpu(device);
  const rows = [];
  const shapes = [];
  for (const length of [5, 9]) {
    for (const cZ of [3, 6, 7, 8, 128]) {
      for (const cHidden of [3, 5, 6, 7, 8, 128]) shapes.push({ length, cZ, cHidden });
    }
  }
  for (const shape of shapes) {
    const input = createDeterministicTriangleInput(shape);
    const cpu = triangleMultiplicationOutgoingReference(input);
    const { output } = await runner.run(input, { precision: "f32" });
    const m = errorMetrics(output, cpu);
    rows.push({ ...shape, mae: Number(m.meanAbsoluteError.toExponential(2)),
                ok: m.meanAbsoluteError < 1e-5 });
  }
  const bad = rows.filter((r) => !r.ok);
  for (const r of rows) {
    console.log(`L=${r.length} cZ=${r.cZ} cHidden=${r.cHidden}\t${r.mae}\t${r.ok ? "ok" : "FAIL"}`);
  }
  return { failing: bad.length, total: rows.length, bad };
}
