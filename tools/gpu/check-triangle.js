/**
 * Differential-test triangle multiplication on the Chrome GPU lane.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-triangle.js
 *
 * WHY THIS EXISTS SEPARATELY FROM test/triangle-multiplication-outgoing.gpu.test.js.
 * That test is the same check, but it runs through Dawn, which does not load on
 * macOS 13 (see tools/gpu-chrome.mjs). This is the proof that the replacement
 * lane can do the thing the AF3 port depends on: run a kernel against an
 * INDEPENDENT reference and report the error. It checks the same two references
 * the Dawn test does - OpenFold's recorded output, and this repo's own CPU
 * implementation - so a disagreement between the lanes would show up here.
 *
 * Every AF3 kernel gets a sibling of this file, comparing against its
 * src/af3/*-reference.js counterpart.
 */
import { loadTriangleReferenceBundle } from "../../src/reference/bundle.js";
import { triangleMultiplicationOutgoingReference } from "../../src/triangle/cpu-reference.js";
import { errorMetrics } from "../../src/triangle/types.js";
import { TriangleMultiplicationOutgoingGpu } from "../../src/triangle/webgpu.js";

// 🔴 ABSOLUTE, because the loader resolves each tensor with
// `new URL(file, manifestUrl)` and a bare path is not a valid base.
const FIXTURE = new URL("/test/fixtures/openfold-triangle-small/manifest.json",
                        location.href).href;

export async function main(device) {
  const bundle = await loadTriangleReferenceBundle(FIXTURE);
  const runner = new TriangleMultiplicationOutgoingGpu(device);
  const cpu = triangleMultiplicationOutgoingReference(bundle.input);

  const results = {};
  let failed = 0;
  for (const precision of ["f32", "f16"]) {
    if (precision === "f16" && !device.features.has("shader-f16")) continue;
    const { output } = await runner.run(bundle.input, { precision });
    // Against OpenFold, which is the reference that matters, and against this
    // repo's CPU path, which is what the AF3 checks will compare to.
    const openfold = errorMetrics(output, bundle.expected);
    const own = errorMetrics(output, cpu);
    // The Dawn test's own bounds, unchanged - a lane change must not move a
    // tolerance.
    const bound = precision === "f32" ? 1e-5 : 2e-3;
    const ok = openfold.meanAbsoluteError < bound;
    if (!ok) failed += 1;
    results[precision] = {
      vsOpenFold: openfold.meanAbsoluteError,
      vsOpenFoldMax: openfold.maxAbsoluteError,
      vsCpuReference: own.meanAbsoluteError,
      bound, ok,
    };
    console.log(`${precision}\tvs OpenFold ${openfold.meanAbsoluteError.toExponential(2)}`
      + ` (max ${openfold.maxAbsoluteError.toExponential(2)})`
      + `\tvs our CPU ${own.meanAbsoluteError.toExponential(2)}`
      + `\t${ok ? "PASS" : `FAIL, bound ${bound}`}`);
  }
  // ...and the CPU reference against OpenFold, which is the lane-independent
  // control: if this drifts, the fixture or the reference moved, not the GPU.
  const control = errorMetrics(cpu, bundle.expected);
  console.log(`cpu\tvs OpenFold ${control.meanAbsoluteError.toExponential(2)}`);

  if (failed > 0) throw new Error(`${failed} precision(s) outside tolerance`);
  return { fixture: FIXTURE, shape: bundle.input.shape, results,
           cpuVsOpenFold: control.meanAbsoluteError };
}
