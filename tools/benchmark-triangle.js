import { create, globals } from "webgpu";
import { createDeterministicTriangleInput } from "../src/testing/deterministic-input.js";
import { TriangleMultiplicationOutgoingGpu } from "../src/triangle/webgpu.js";

function option(name, fallback) {
  const prefix = `--${name}=`;
  return process.argv.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const lengths = option("lengths", "128,256,512").split(",").map(Number);
const cZ = Number(option("cz", "128"));
const cHidden = Number(option("hidden", "128"));
const precision = option("precision", "f16");
if (!lengths.every((length) => Number.isSafeInteger(length) && length > 0)) {
  throw new Error("--lengths must be a comma-separated list of positive integers");
}
if (precision !== "f16" && precision !== "f32") throw new Error("--precision must be f16 or f32");

Object.assign(globalThis, globals);
const requestedAdapter = option("adapter", "");
const gpu = create(requestedAdapter === "" ? [] : [`adapter=${requestedAdapter}`]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter is available");
const adapterName = adapter.info.description || adapter.info.device || adapter.info.vendor || "unknown";
console.log(`adapter=${adapterName}`);
if (precision === "f16" && !adapter.features.has("shader-f16")) {
  throw new Error("the selected adapter does not expose shader-f16; retry with --precision=f32");
}
const requiredFeatures = precision === "f16" ? ["shader-f16"] : [];
const device = await adapter.requestDevice({ requiredFeatures });
const runner = new TriangleMultiplicationOutgoingGpu(device);

console.log(`precision=${precision} c_z=${cZ} c_hidden=${cHidden}`);
console.log("L\ttime_ms\tpeak_gpu_mib");
for (const length of lengths) {
  const input = createDeterministicTriangleInput({ length, cZ, cHidden }, 1000 + length);
  const result = await runner.run(input, { precision });
  console.log(`${length}\t${result.elapsedMilliseconds.toFixed(3)}\t${(result.memory.peakBytes / 2 ** 20).toFixed(2)}`);
}
device.destroy();
