// The differential kernel smoke test, on its own page.
//
// It used to live in a <details> at the bottom of the demo. The demo is now one
// sequence box and a Fold button, and a triangle-multiplication tolerance check
// is not something a person folding a protein has any use for - but it IS what
// the browser tests assert against, and it is the only coverage that runs a
// real kernel in a real browser. So it moved rather than went away.
import { createDeterministicTriangleInput } from "../src/testing/deterministic-input.js";
import { triangleMultiplicationOutgoingReference } from "../src/triangle/cpu-reference.js";
import { errorMetrics } from "../src/triangle/types.js";
import { TriangleMultiplicationOutgoingGpu } from "../src/triangle/webgpu.js";

const element = (id) => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value;
};

const parameter = (name, fallback) => new URLSearchParams(location.search).get(name) ?? fallback;

async function runDiagnostic() {
  const status = element("status");
  const result = element("result");
  status.dataset.state = "running"; status.textContent = "Running…";
  result.textContent = "Requesting a WebGPU adapter.";
  let device;
  try {
    if (navigator.gpu === undefined) throw new Error("WebGPU is not available in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("No compatible WebGPU adapter was found");
    const precision = element("precision").value;
    if (precision === "f16" && !adapter.features.has("shader-f16")) {
      throw new Error("This adapter does not expose shader-f16");
    }
    device = await adapter.requestDevice({ requiredFeatures: precision === "f16" ? ["shader-f16"] : [] });
    const shape = {
      length: element("length").valueAsNumber,
      cZ: element("cz").valueAsNumber,
      cHidden: element("hidden").valueAsNumber,
    };
    const input = createDeterministicTriangleInput(shape, 29);
    const expected = triangleMultiplicationOutgoingReference(input);
    const gpuResult = await new TriangleMultiplicationOutgoingGpu(device).run(input, { precision });
    const errors = errorMetrics(gpuResult.output, expected);
    window.__LOCALFOLD_RESULT__ = {
      elapsedMilliseconds: gpuResult.elapsedMilliseconds,
      peakBytes: gpuResult.memory.peakBytes,
      meanAbsoluteError: errors.meanAbsoluteError,
      maxAbsoluteError: errors.maxAbsoluteError,
    };
    status.dataset.state = "passed"; status.textContent = "Differential test passed";
    result.textContent = JSON.stringify({ shape, precision, ...window.__LOCALFOLD_RESULT__ }, null, 2);
  } catch (error) {
    status.dataset.state = "failed"; status.textContent = "Run failed";
    result.textContent = error instanceof Error ? error.stack ?? error.message : String(error);
  } finally {
    device?.destroy();
  }
}

element("controls").addEventListener("submit", (event) => { event.preventDefault(); void runDiagnostic(); });
if (parameter("autorun", "0") === "1") void runDiagnostic();
