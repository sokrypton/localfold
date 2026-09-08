/**
 * Which subgroup matrix shapes does THIS device offer, and what do the type
 * parameters mean at a shape where the two readings differ?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-subgroup-matrix-shapes.js
 *
 * 🔴 check-subgroup-matrix.js CANNOT ANSWER THIS ON AN M2, and did not claim
 * to: 8x8x8 is the only shape Apple offers, and <T, COLUMNS, ROWS> and
 * <T, ROWS, COLUMNS> are the same declaration when every extent is eight. An
 * A100 offers M16 N8 K16, where they are different declarations, so the
 * ordering is a measurable fact here rather than a documented one.
 *
 * 🔴 AND THE A100 OFFERS NO f32 COMPONENT TYPE AT ALL. Its configs are f16 in
 * (accumulating in f16 OR f32) and 8-bit integer in. So the shipped f32 matrix
 * kernel is not slow on this device, it is UNSUPPORTED - createComputePipeline
 * rejects it with "Unknown configuration is M(8), N(8), K(0), f32". The matrix
 * path here lives or dies with the f16 flag, which is a different coupling
 * from the M2's, where f32 tiles exist and are exact.
 *
 * Each config is run as a single M x K by K x N product against a host
 * reference, under both readings of the type parameters, and reports which one
 * the hardware agreed with.
 */
const FEATURE = "chromium-experimental-subgroup-matrix";

/** f32 -> IEEE half, as the u16 bit pattern. */
function halfBits(value) {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  f[0] = value;
  const bits = u[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  let e = exponent - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - e;
    return sign | (mantissa >> shift);
  }
  return sign | (e << 10) | (mantissa >> 13);
}

/** The half a device would hold, so the host reference rounds the same way. */
function halfValue(value) {
  const bits = halfBits(value);
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa ? NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

export async function main(device) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) return { error: "no adapter" };
  if (!adapter.features.has(FEATURE)) return { skipped: `no ${FEATURE}` };

  const configs = [...(adapter.info?.subgroupMatrixConfigs ?? [])].map((c) => ({
    componentType: c.componentType,
    resultComponentType: c.resultComponentType,
    M: c.M, N: c.N, K: c.K,
  }));

  const required = [FEATURE];
  if (adapter.features.has("shader-f16")) required.push("shader-f16");
  const gpu = await adapter.requestDevice({ requiredFeatures: required });
  const failures = [];
  gpu.addEventListener("uncapturederror", (e) => failures.push(String(e.error.message)));

  const results = [];
  for (const config of configs) {
    // Only the float configs; the integer ones are a separate question and
    // nothing in this repository quantises to int8 at inference time.
    if (config.componentType !== "f16") {
      results.push({ ...config, skipped: "not a float config" });
      continue;
    }
    const { M, N, K } = config;
    // Deliberately asymmetric and small, so an f16 multiply is exact and a
    // transpose cannot agree by accident.
    const a = new Float32Array(M * K);
    const b = new Float32Array(K * N);
    for (let i = 0; i < a.length; i += 1) a[i] = halfValue(((i * 7) % 13) - 6 + (i % 3) * 0.25);
    for (let i = 0; i < b.length; i += 1) b[i] = halfValue(((i * 5) % 11) - 5 - (i % 4) * 0.5);
    const truth = new Float32Array(M * N);
    for (let r = 0; r < M; r += 1) {
      for (let c = 0; c < N; c += 1) {
        let sum = 0;
        for (let k = 0; k < K; k += 1) sum += a[r * K + k] * b[k * N + c];
        truth[r * N + c] = sum;
      }
    }

    const toHalf = (source) => {
      const out = new Uint16Array(source.length);
      for (let i = 0; i < source.length; i += 1) out[i] = halfBits(source[i]);
      return out;
    };
    const upload = (data) => {
      const size = Math.ceil(data.byteLength / 4) * 4;
      const buffer = gpu.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
      new Uint16Array(buffer.getMappedRange()).set(data);
      buffer.unmap();
      return buffer;
    };
    const left = upload(toHalf(a));
    const right = upload(toHalf(b));
    const output = gpu.createBuffer({
      size: M * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const result = config.resultComponentType;
    // The two readings. `left` is M x K, `right` is K x N, `result` is M x N;
    // what is in question is only which extent is written first.
    const readings = {
      "columns,rows": { left: [K, M], right: [N, K], result: [N, M] },
      "rows,columns": { left: [M, K], right: [K, N], result: [M, N] },
    };
    const scores = {};
    for (const [name, shape] of Object.entries(readings)) {
      const code = `enable f16;
enable ${FEATURE.replaceAll("-", "_").replace("chromium_experimental", "chromium_experimental")};
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> staged: array<${result}, ${M * N}>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane: u32) {
  let l = subgroupMatrixLoad<subgroup_matrix_left<f16, ${shape.left[0]}, ${shape.left[1]}>>(&a, 0u, false, ${K}u);
  let r = subgroupMatrixLoad<subgroup_matrix_right<f16, ${shape.right[0]}, ${shape.right[1]}>>(&b, 0u, false, ${N}u);
  var acc = subgroup_matrix_result<${result}, ${shape.result[0]}, ${shape.result[1]}>();
  acc = subgroupMatrixMultiplyAccumulate(l, r, acc);
  subgroupMatrixStore(&staged, 0u, acc, false, ${N}u);
  workgroupBarrier();
  for (var i = lane; i < ${M * N}u; i += 32u) { out[i] = f32(staged[i]); }
}`;
      gpu.pushErrorScope("validation");
      let pipeline = null;
      try {
        pipeline = await gpu.createComputePipelineAsync({
          layout: "auto",
          compute: { module: gpu.createShaderModule({ code }), entryPoint: "main" },
        });
      } catch (error) {
        scores[name] = { rejected: String(error.message ?? error).split("\n")[0] };
      }
      const validation = await gpu.popErrorScope();
      if (pipeline === null) {
        if (validation && !scores[name]) scores[name] = { rejected: validation.message.split("\n")[0] };
        continue;
      }
      const bindGroup = gpu.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [left, right, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const encoder = gpu.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      const staging = gpu.createBuffer({
        size: M * N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      encoder.copyBufferToBuffer(output, 0, staging, 0, M * N * 4);
      gpu.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      let num = 0;
      let den = 0;
      for (let i = 0; i < truth.length; i += 1) { num += (got[i] - truth[i]) ** 2; den += truth[i] ** 2; }
      scores[name] = { relRms: Number(Math.sqrt(num / Math.max(den, 1e-30)).toPrecision(3)) };
    }
    const agreed = Object.entries(scores)
      .filter(([, s]) => typeof s.relRms === "number" && s.relRms < 1e-2)
      .map(([name]) => name);
    results.push({ ...config, scores, agreed });
  }

  return { configs: configs.length, results, failures };
}
