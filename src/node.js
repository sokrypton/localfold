/**
 * A WebGPU device for Node, with the toggles this port's fast path needs.
 *
 * 🔴 THIS IS THE ONE PLACE THE FAST PATH IS REACHABLE WITHOUT ASKING A HUMAN.
 * Two capabilities carry most of this port's speed and BOTH are behind browser
 * flags a visitor does not have: Dawn refuses `shader-f16` on every NVIDIA GPU
 * pending a conformance investigation (crbug.com/42251215), and the subgroup
 * matrix units are `chromium-experimental-` on every platform - measured on an
 * A100 AND an M2. In a page there is nothing to be done about either. In a
 * process there is: Dawn's node binding takes the toggles as arguments, so a
 * library can turn them on for itself.
 *
 * Measured on webgpu@0.4.0 here:
 *
 *     create([])                    shader-f16 no   subgroup-matrix no   18
 *     create([F16])                 shader-f16 YES  subgroup-matrix no   19
 *     create([F16 + "," + UNSAFE])  shader-f16 YES  subgroup-matrix YES  21
 *
 * 🔴 AND THE SECOND TOGGLE GOES AFTER A COMMA, not in a second array element,
 * which throws "Flags expected argument format is <key>=<value>".
 */
import { requestAlphaFoldDevice } from "./runtime/device.js";

/** Dawn's own names. `vulkan_enable_f16_on_nvidia` is inert off Vulkan. */
export const DAWN_TOGGLES = ["vulkan_enable_f16_on_nvidia", "allow_unsafe_apis"];

/**
 * @param {{toggles?: string[], adapter?: string, powerPreference?: GPUPowerPreference}} [options]
 * @returns {Promise<{device: GPUDevice, adapter: GPUAdapter, gpu: unknown}>}
 */
export async function createNodeDevice(options = {}) {
  let binding;
  try {
    binding = await import("webgpu");
  } catch (cause) {
    // 🔴 NAMED, BECAUSE THE FAILURE IS A PLATFORM ONE AND LOOKS LIKE A BUG.
    // The published linux-x64 binary of webgpu@0.6.0 wants GLIBC 2.38; 0.4.0
    // wants 2.34. A box with an older libc needs the older package.
    throw new Error(
      "localfold/node needs the optional `webgpu` package (Dawn's node binding): "
      + "`npm i webgpu`. If it installs and then fails to load, its prebuilt "
      + "binary wants a newer GLIBC than this machine has - try `npm i webgpu@0.4.0`. "
      + `The loader said: ${String(cause?.message ?? cause).split("\n")[0]}`,
    );
  }
  Object.assign(globalThis, binding.globals);
  const toggles = options.toggles ?? DAWN_TOGGLES;
  const flags = [
    ...(toggles.length === 0 ? [] : [`enable-dawn-features=${toggles.join(",")}`]),
    ...(options.adapter === undefined ? [] : [`adapter=${options.adapter}`]),
  ];
  const gpu = binding.create(flags);
  const adapter = await gpu.requestAdapter(
    { powerPreference: options.powerPreference ?? "high-performance" });
  if (adapter === null) {
    throw new Error("no WebGPU adapter: Dawn found no usable GPU on this machine");
  }
  return { device: await requestAlphaFoldDevice(adapter), adapter, gpu };
}
