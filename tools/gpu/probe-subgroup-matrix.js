/**
 * Does this device offer WebGPU's subgroup matrix units, and in what shapes?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-subgroup-matrix.js
 *
 * Upstream (martin-steinegger/alphafold2-webgpu, docs/SUBGROUP_MATRIX_APPLE_METAL_3.md)
 * measures 1.46x-2.15x on the dense projections from these, in f32, with the
 * SAME error as the f32 kernel to the digit - so where they exist they are
 * strictly better than the f16 arithmetic this repository ships. That was on an
 * M4 Pro; this asks what an M2 has.
 *
 * 🔴 THE FEATURE IS NOT REQUESTED BY requestAlphaFoldDevice, so a probe that
 * only inspected the device it was handed would report "no" on a machine that
 * has it. This asks the ADAPTER, and then requests its own device.
 *
 * `chromium-experimental-subgroup-matrix` is not standards-compliant WGSL. If
 * it is ever used it has to be a capability-gated fast path, the way the
 * subgroup attention kernels already are.
 */
const FEATURE = "chromium-experimental-subgroup-matrix";

export async function main(device) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) return { error: "no adapter" };
  const adapterFeatures = [...adapter.features].sort();
  const available = adapterFeatures.includes(FEATURE);
  const result = {
    adapterFeatures,
    deviceFeatures: [...device.features].sort(),
    subgroupMatrixAvailable: available,
    // What the adapter says the units can do, if anything. The shape list is
    // how a kernel learns which <f32, M, N> tiles it may ask for.
    // 🔴 A CONFIG IS A WebIDL INTERFACE, NOT A PLAIN OBJECT, so JSON.stringify
    // returns `{}` for every one of them. The fields have to be named.
    subgroupMatrixConfigs: [...(adapter.info?.subgroupMatrixConfigs ?? [])].map((c) => ({
      componentType: c.componentType, resultComponentType: c.resultComponentType,
      M: c.M, N: c.N, K: c.K,
    })),
  };
  if (!available) return result;

  // It exists: prove a trivial kernel using it actually compiles, because a
  // feature bit and a working WGSL extension are not the same claim.
  const matrixDevice = await adapter.requestDevice({ requiredFeatures: [FEATURE] });
  const shader = `enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(32)
fn main() {
  let left = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&a, 0u, false, 8u);
  let right = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&b, 0u, false, 8u);
  var acc = subgroup_matrix_result<f32, 8, 8>();
  acc = subgroupMatrixMultiplyAccumulate(left, right, acc);
  subgroupMatrixStore(&out, 0u, acc, false, 8u);
}`;
  matrixDevice.pushErrorScope("validation");
  const module = matrixDevice.createShaderModule({ code: shader });
  const info = await module.getCompilationInfo();
  const error = await matrixDevice.popErrorScope();
  return {
    ...result,
    compiles: error === null,
    compileError: error === null ? null : error.message,
    messages: [...info.messages].map((m) => `${m.type}: ${m.message}`),
  };
}
