/**
 * Requests the optional WebGPU features and limits LocalFold's fast paths use.
 *
 * 🔴 LIMITS ARE NOT INHERITED FROM THE ADAPTER. A device gets the SPEC DEFAULTS
 * unless it asks for more, and the default storage binding is 128 MiB while this
 * hardware offers 4 GiB. Nothing warns; the first buffer over the line fails
 * validation mid-fold, which is how a six-copy complex died on a 184 MiB MSA
 * activation buffer that the adapter could have bound thirty times over.
 *
 * Every limit raised here is asked for at exactly what the adapter reports, so
 * the request cannot fail: requestDevice rejects a limit BETTER than the
 * adapter's, not one equal to it.
 */
import { budgetForDevice, setMemoryBudget } from "./device-memory.js";
import { recordAdapter } from "./device-profile.js";

const RAISED_LIMITS = [
  // The MSA activations at a large complex: rows x residues x channels x 4.
  "maxStorageBufferBindingSize",
  // ...and the allocation behind it, which is capped separately.
  "maxBufferSize",
  // Long sequences make the pair track's dispatches wide.
  "maxComputeWorkgroupsPerDimension",
  // 🔴 THE DIFFUSION TRANSFORMER SPENDS THIS ON TOKEN TILES. Its matmul kernels
  // hold a tile of activations in workgroup memory so that one weight read
  // serves the whole tile, and the tile size is capped by exactly this limit -
  // 16 KiB by default, which allows four tokens, against the 32 KiB every
  // adapter tested reports. It is the difference between reading the block's
  // weights once per four tokens and once per eight.
  "maxComputeWorkgroupStorageSize",
  // 🔴 A WORKGROUP IS ONE THREAD PER PAIR CHANNEL IN THE GRID PROJECTION, AND
  // OpenDDE HAS 384 OF THEM. The spec default is 256 in X and 256 invocations
  // per workgroup, so a 384-channel pair track fails to create the pipeline -
  // "workgroup_size(384) exceeds the maximum allowed (256)" - while this
  // adapter reports 1024 for both. AlphaFold 3 at 128 channels never came near
  // either, which is why neither was asked for until a second bundle's widths
  // arrived. Both have to be raised: the X extent alone is not enough, because
  // the total invocation count is capped separately.
  "maxComputeWorkgroupSizeX",
  "maxComputeInvocationsPerWorkgroup",
];

/**
 * @param {{memoryBudgetBytes?: number}} [options] a ceiling on what this device
 *   may hold, in bytes. Omitted, the device is only counted, not bounded;
 *   `null` asks for the default guess from budgetForDevice.
 */
export async function requestAlphaFoldDevice(adapter, options = {}) {
  // subgroup-size-control is shipping ahead of the current @webgpu/types union.
  //
  // 🔴 chromium-experimental-subgroup-matrix IS NOT STANDARDS-COMPLIANT WGSL,
  // and it is asked for anyway because it is OPTIONAL in exactly the way the
  // others are: a browser that does not advertise it never sees it requested,
  // and no kernel may use it without gating on device.features. It is here so
  // a capability-gated fast path CAN be selected - the matrix units compute
  // what the f32 kernel computes, accumulating in f32, so where they exist
  // they beat the f16 arithmetic this repository ships without its rounding.
  // See tools/gpu/probe-subgroup-matrix.js for what a device offers.
  const optional = [
    "subgroups", "subgroup-size-control", "timestamp-query", "shader-f16",
    "chromium-experimental-subgroup-matrix",
  ];
  const requiredFeatures = optional.filter(
    (feature) => adapter.features.has(feature),
  );
  const requiredLimits = {};
  for (const name of RAISED_LIMITS) {
    const available = adapter.limits?.[name];
    if (typeof available === "number" && Number.isFinite(available)) {
      requiredLimits[name] = available;
    }
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  // 🔴 THE ADAPTER IS THE ONLY THING THAT KNOWS WHAT THIS IS, AND KERNELS ARE
  // HANDED A DEVICE. `adapter.info` does not survive onto the GPUDevice, so
  // the identity is recorded here - the one place both are in scope - and
  // src/runtime/device-profile.js is where a selection function asks for it.
  recordAdapter(device, adapter);
  // 🔴 A DEVICE THAT ACCEPTS AN ALLOCATION IT CANNOT AFFORD FREEZES THE MACHINE.
  // Metal takes buffers well past the point where macOS starts paging, and a
  // phone's driver takes them and is then killed by the system - in neither
  // case does WebGPU report anything. src/runtime/device-memory.js turns that
  // into a GpuMemoryBudgetError naming the tensor, but only for a device that
  // has been given a ceiling, which is this.
  if ("memoryBudgetBytes" in options) {
    setMemoryBudget(device, options.memoryBudgetBytes ?? budgetForDevice());
  }
  return device;
}
