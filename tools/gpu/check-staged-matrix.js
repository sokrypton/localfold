/**
 * Does the staged matrix projection compute what createLinearShader computes,
 * in every buffer precision, with and without the residual, on ragged shapes?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-staged-matrix.js
 *
 * 🔴 THE RAGGED CASES ARE THE POINT. This kernel has two different edge
 * strategies and they fail in different ways. On the row and column axes the
 * last region SLIDES BACK, so a shape not divisible by the block recomputes an
 * overlap - which is only correct while the writes are idempotent, and the
 * residual add is exactly the write that is not. On the K axis it ZERO-PADS the
 * final panel, because a K tail cannot slide. A shape divisible by everything
 * exercises neither.
 *
 * The reference rounds both operands to halves before accumulating, because
 * the units multiply in f16 whatever the buffers hold; what is being checked is
 * the indexing and the epilogue, not the arithmetic width.
 */
import { createStagedMatrixShader, stagedMatrixFits, stagedMatrixStorage } from "./gemm-matrix-staged.js";
import { deviceMatrixConfig, recordAdapter } from "../../src/runtime/device-profile.js";

const half = (value) => {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  f[0] = value;
  const bits = u[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return value;
  let e = exponent - 127 + 15;
  if (e >= 0x1f) return Math.sign(value) * 65504;
  if (e <= 0) {
    if (e < -10) return 0;
    const shifted = (mantissa | 0x800000) >> (14 - e);
    return (sign ? -1 : 1) * shifted * 2 ** -24;
  }
  const kept = (e << 10) | (mantissa >> 13);
  const exp = ((kept >>> 10) & 0x1f) - 15;
  return (sign ? -1 : 1) * (1 + (kept & 0x3ff) / 1024) * 2 ** exp;
};
const halfBits = (value) => {
  const rounded = half(value);
  if (rounded === 0) return Object.is(rounded, -0) ? 0x8000 : 0;
  const sign = rounded < 0 ? 0x8000 : 0;
  const a = Math.abs(rounded);
  const exp = Math.floor(Math.log2(a));
  if (exp < -14) return sign | Math.round(a * 2 ** 24);
  return sign | ((exp + 15) << 10) | Math.round((a / 2 ** exp - 1) * 1024);
};

export async function main(device, args = []) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const wanted = ["chromium-experimental-subgroup-matrix", "shader-f16", "subgroups"]
    .filter((f) => adapter.features.has(f));
  if (!wanted.includes("chromium-experimental-subgroup-matrix")) return { skipped: "no matrix units" };
  const gpu = await adapter.requestDevice({
    requiredFeatures: wanted,
    requiredLimits: {
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    },
  });
  recordAdapter(gpu, adapter);
  const config = deviceMatrixConfig(gpu, { element: "f16" });
  if (config === null) return { skipped: "no f16 matrix config" };
  const tile = { M: config.M, N: config.N, K: config.K };

  const failures = [];
  gpu.addEventListener("uncapturederror", (e) => failures.push(String(e.error.message)));

  const cases = [];
  for (const shape of [
    { rows: 256, inner: 128, columns: 256, name: "aligned" },
    // Ragged on every axis at once. Every edge is zero-padded now, so this is
    // exercised rather than refused - including with a residual.
    { rows: 300, inner: 130, columns: 200, name: "ragged" },
    // 🔴 SMALLER THAN ONE REGION ON BOTH AXES, which the slide-back form could
    // not serve at all: `rows - blockRows` underflowed in u32 and every load
    // landed out of bounds, which reads as a moderate wrong answer rather than
    // an error. A 59-token single track is exactly this shape.
    { rows: 59, inner: 384, columns: 64, name: "tiny" },
  ]) {
    for (const sourcePrecision of ["f32", "f16"]) {
      for (const outputPrecision of ["f32", "f16"]) {
        for (const activation of [0, 1]) {
          for (const residual of [false, true]) {
            for (const weightBase of [4, 3]) {
              cases.push({ ...shape, sourcePrecision, outputPrecision, activation, residual, weightBase });
            }
          }
        }
      }
    }
  }

  const results = [];
  for (const c of cases) {
    const { rows, inner, columns, sourcePrecision, outputPrecision, activation, residual } = c;
    const geometry = {
      blockRows: 128, blockColumns: 128, blockInner: 16, subgroupRows: 1, subgroupColumns: 8,
      tile, result: config.resultComponentType, matrixElement: config.componentType,
    };
    if (!stagedMatrixFits({ rows, columns }, { ...geometry, residual })) {
      results.push({ ...c, skipped: "refused by stagedMatrixFits" });
      continue;
    }
    // 🔴 VECTOR STAGING NEEDS EVERY OFFSET IT FORMS TO DIVIDE BY FOUR, which
    // the ragged shape's inner extent does not. Both paths are checked.
    const vectorStaging = inner % 4 === 0 && columns % 4 === 0;

    // 🔴 THE WEIGHTS SIT AT A NONZERO, DELIBERATELY UNALIGNED BASE. Every
    // earlier version of this checker - and the bench - left weight_offset at
    // zero, and the kernel simply did not read the field: it indexed from the
    // start of the buffer. That is invisible at offset zero and was caught only
    // by AF2's differential checker, at relRMS 1.85, after the kernel had
    // already been called correct. A base of 3 also exercises the vec4 path's
    // alignment gate, since 3 % 4 is not 0.
    const weightBase = 3;
    const source = new Float32Array(rows * inner);
    const weightData = new Float32Array(weightBase + inner * columns + columns);
    for (let i = 0; i < source.length; i += 1) source[i] = half(((i * 37) % 19 - 9) / 8);
    for (let i = 0; i < weightData.length; i += 1) weightData[i] = half(((i * 23) % 17 - 8) / 16);
    const weightAt = (k, col) => weightData[weightBase + k * columns + col];
    const before = new Float32Array(rows * columns);
    for (let i = 0; i < before.length; i += 1) before[i] = half(((i * 11) % 13 - 6) / 4);

    const truth = new Float32Array(rows * columns);
    for (let r = 0; r < rows; r += 1) {
      for (let col = 0; col < columns; col += 1) {
        let sum = 0;
        for (let k = 0; k < inner; k += 1) sum += source[r * inner + k] * weightAt(k, col);
        sum += weightData[weightBase + inner * columns + col];
        if (activation === 1) sum = Math.max(sum, 0);
        if (residual) sum += before[r * columns + col];
        truth[r * columns + col] = outputPrecision === "f16" ? half(sum) : sum;
      }
    }

    const pack = (data, precision) => {
      if (precision === "f32") return { array: data, bytes: data.byteLength };
      const out = new Uint16Array(data.length);
      for (let i = 0; i < data.length; i += 1) out[i] = halfBits(data[i]);
      return { array: out, bytes: out.byteLength };
    };
    const upload = (packed, usage) => {
      const size = Math.ceil(packed.bytes / 4) * 4;
      const buffer = gpu.createBuffer({ size, usage, mappedAtCreation: true });
      new packed.array.constructor(buffer.getMappedRange()).set(packed.array);
      buffer.unmap();
      return buffer;
    };
    const STORAGE = GPUBufferUsage.STORAGE;
    const sourceBuffer = upload(pack(source, sourcePrecision), STORAGE);
    const weightBuffer = upload(pack(weightData, "f16"), STORAGE);
    const outputPacked = pack(before, outputPrecision);
    const outputBuffer = upload(outputPacked, STORAGE | GPUBufferUsage.COPY_SRC);
    const parameters = gpu.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true,
    });
    new Uint32Array(parameters.getMappedRange())
      .set([rows, inner, columns, weightBase, weightBase + inner * columns, activation, 0, 0]);
    parameters.unmap();

    const code = createStagedMatrixShader({
      ...geometry, sourcePrecision, weightPrecision: "f16", outputPrecision, residual,
      vectorStaging: vectorStaging && weightBase % 4 === 0,
    });
    gpu.pushErrorScope("validation");
    let pipeline = null;
    try {
      pipeline = await gpu.createComputePipelineAsync({
        layout: "auto", compute: { module: gpu.createShaderModule({ code }), entryPoint: "main" },
      });
    } catch (error) { results.push({ ...c, error: String(error.message ?? error).split("\n")[0] }); }
    const validation = await gpu.popErrorScope();
    if (pipeline === null) {
      if (validation) results.push({ ...c, error: validation.message.split("\n")[0] });
      continue;
    }
    const bindGroup = gpu.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [sourceBuffer, weightBuffer, parameters, outputBuffer]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = gpu.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(columns / 128), Math.ceil(rows / 128));
    pass.end();
    const staging = gpu.createBuffer({
      size: Math.ceil(outputPacked.bytes / 4) * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(outputBuffer, 0, staging, 0, staging.size);
    gpu.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = staging.getMappedRange().slice(0);
    staging.unmap();
    const got = outputPrecision === "f32"
      ? new Float32Array(raw)
      : (() => {
        const bits = new Uint16Array(raw);
        const out = new Float32Array(rows * columns);
        for (let i = 0; i < out.length; i += 1) {
          const b = bits[i];
          const s = b & 0x8000 ? -1 : 1;
          const e = (b >>> 10) & 0x1f;
          const m = b & 0x3ff;
          out[i] = e === 0 ? s * m * 2 ** -24 : e === 0x1f ? (m ? NaN : s * Infinity)
            : s * (1 + m / 1024) * 2 ** (e - 15);
        }
        return out;
      })();
    let num = 0;
    let den = 0;
    for (let i = 0; i < truth.length; i += 1) { num += (got[i] - truth[i]) ** 2; den += truth[i] ** 2; }
    const relRms = Math.sqrt(num / Math.max(den, 1e-30));
    results.push({
      ...c, vectorStaging,
      relRms: Number(relRms.toPrecision(3)),
      ok: relRms < (outputPrecision === "f16" ? 3e-3 : 1e-3),
    });
    for (const b of [sourceBuffer, weightBuffer, outputBuffer, parameters, staging]) b.destroy();
  }
  const bad = results.filter((r) => r.ok === false || r.error);
  return {
    config, cases: results.length, failed: bad.length,
    worst: results.filter((r) => r.ok !== undefined).sort((a, b) => b.relRms - a.relRms)[0],
    bad, failures,
  };
}
