/**
 * Does the matrix grid projection compute the vector one's answer?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-grid-project-matrix.js
 *
 * 🔴 IT TAKES NO BUNDLE, so it runs on a box that has no weights and it can
 * sweep widths AF3 does not have. OpenDDE's grid attention is 8 heads where
 * AF3's is 4, and a kernel checked at one head count is a kernel checked at one
 * head count - see docs/OPENDDE.md, where a dispatch sized for 128 channels
 * against shaders compiled for 384 left two thirds of every pair row
 * unprocessed with every per-kernel checker passing.
 *
 * 🔴 AND IT RUNS BOTH DIRECTIONS, because the transposed one reads
 * `normalized[(row % n) * n + row / n]` and writes at `row`. That is the one
 * thing in this kernel a wrong answer looks exactly like a right one: the
 * output has the same shape, the same magnitude and the same distribution, and
 * only the pairing is wrong.
 *
 * The bar is f16 (~3e-3) when the staged tiles are halves, because the matrix
 * units multiply in f16 whatever the buffers hold; the vector kernel it is
 * compared against accumulates in f32.
 */
import { createGridAttentionShaders, packGridAttentionWeights }
  from "../../src/af3/grid-attention-webgpu.js";
import {
  createGridProjectMatrixShader, createGridProjectOutMatrixShader,
  gridProjectMatrixDispatch, gridProjectMatrixFits, gridProjectOutMatrixDispatch,
} from "../../src/af3/grid-project-matrix.js";
import { stagedMatrixBlock } from "../../src/runtime/matrix-linear.js";
import { deviceMatrixConfig, deviceTuning } from "../../src/runtime/device-profile.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const deterministic = (count, seed) => {
  const out = new Float32Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = ((state >>> 8) / 8388608 - 1) * 0.5;
  }
  return out;
};

const relative = (got, want) => {
  let error = 0;
  let scale = 0;
  for (let i = 0; i < want.length; i += 1) {
    error += (got[i] - want[i]) ** 2;
    scale += want[i] ** 2;
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
};

export async function main(device, args) {
  const channels = Number(option(args, "channels", "128"));
  const heads = Number(option(args, "heads", "4"));
  const dimension = Number(option(args, "dimension", "32"));
  const tokens = option(args, "tokens", "40,37,16").split(",").map(Number);
  const bound = Number(option(args, "bound", "3e-3"));

  const config = deviceMatrixConfig(device, { element: "f16" });
  if (config === null) return { skipped: "no f16 subgroup matrix configuration" };
  // The shipped block unless --block= says otherwise: a checker that runs a
  // geometry nothing ships is a checker of a kernel nobody runs.
  const matrix = { result: config.resultComponentType, matrixElement: config.componentType,
                   tile: { M: config.M, N: config.N, K: config.K },
                   prefetch: deviceTuning(device).stagedMatrixPrefetch === true,
                   ...stagedMatrixBlock(option(args, "block", null)
                     ?? deviceTuning(device).stagedMatrixBlock) };
  if (!gridProjectMatrixFits(matrix, device.limits.maxComputeWorkgroupStorageSize)) {
    return { skipped: "the geometry does not fit this device's workgroup storage" };
  }
  const width = heads * dimension;

  const weights = {
    actNormScale: new Float32Array(channels).fill(1),
    actNormOffset: new Float32Array(channels),
    pairBiasProjection: deterministic(channels * heads, 5),
    qProjection: deterministic(width * channels, 11),
    kProjection: deterministic(width * channels, 12),
    vProjection: deterministic(width * channels, 13),
    gatingQuery: deterministic(width * channels, 14),
    outputProjection: deterministic(channels * width, 15),
    heads, dimension,
  };
  const packed = packGridAttentionWeights(weights, { width });

  const storage = GPUBufferUsage.STORAGE;
  const upload = (data, usage = storage) => {
    const buffer = device.createBuffer({
      size: Math.ceil(data.byteLength / 4) * 4, usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  };
  const weightBuffer = upload(packed.data);

  const rows = [];
  let failed = 0;
  for (const n of tokens) {
    for (const transpose of [false, true]) {
      const pairs = n * n;
      const normalized = upload(deterministic(pairs * channels, 991 + n));
      const out = () => device.createBuffer({
        size: pairs * width * 4, usage: storage | GPUBufferUsage.COPY_SRC });
      const vector = [out(), out(), out(), out()];
      const matrixOut = [out(), out(), out(), out()];
      const readback = device.createBuffer({
        size: pairs * width * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

      const { tiles, project } = createGridAttentionShaders(
        { n, channels, heads, dimension, transpose, residual: true },
        packed.offsets, 1e-5, "fast", { swapTransposedBias: false },
        "f32", "f32", "f32");
      const matrixSource = createGridProjectMatrixShader(
        { n, channels, width, transpose }, {}, matrix);
      const [vectorPipe, matrixPipe] = await Promise.all([project, matrixSource].map(
        (code) => device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));

      const parameters = upload(new Uint32Array(
        [pairs, channels, 4 * width, packed.offsets.qkvgProjection, 0, 0, 0, 0]),
        GPUBufferUsage.UNIFORM);

      device.pushErrorScope("validation");
      const encoder = device.createCommandEncoder();
      const pass = (pipeline, entries, x, y) => {
        const p = encoder.beginComputePass();
        p.setPipeline(pipeline);
        p.setBindGroup(0, device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        p.dispatchWorkgroups(x, y);
        p.end();
      };
      const GRID = 32768;
      const tiles_ = Math.ceil(pairs / tiles.projectRows);
      pass(vectorPipe, [normalized, weightBuffer, ...vector],
           Math.min(tiles_, GRID), Math.ceil(tiles_ / GRID));
      const dispatch = gridProjectMatrixDispatch({ rows: pairs, width }, matrix);
      pass(matrixPipe, [normalized, weightBuffer, parameters, ...matrixOut],
           dispatch.x, dispatch.y);
      device.queue.submit([encoder.finish()]);
      const error = await device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);

      const read = async (buffer) => {
        const e = device.createCommandEncoder();
        e.copyBufferToBuffer(buffer, 0, readback, 0, pairs * width * 4);
        device.queue.submit([e.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        return copy;
      };
      const names = ["q", "k", "v", "gate"];
      const scores = {};
      for (let i = 0; i < 4; i += 1) {
        scores[names[i]] = relative(await read(matrixOut[i]), await read(vector[i]));
      }
      const worst = Math.max(...Object.values(scores));
      const ok = worst <= bound;
      if (!ok) failed += 1;
      // 🔴 AND THE OUTPUT PROJECTION, which is the same operation two kernels
      // later and has the two things this arm does not: a gate read from its
      // own buffer at the SOURCE's index, and a transposed DESTINATION. Both
      // directions and the residual form, because a kernel that adds into its
      // target agrees with one that overwrites wherever the target is zero.
      {
        const gathered = upload(deterministic(pairs * width, 71 + n));
        const gateBuffer = upload(deterministic(pairs * width, 137 + n));
        const before = deterministic(pairs * channels, 211 + n);
        const outV = upload(before.slice(), storage | GPUBufferUsage.COPY_SRC);
        const outM = upload(before.slice(), storage | GPUBufferUsage.COPY_SRC);
        const back = device.createBuffer({
          size: pairs * channels * 4,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const vectorOut = createGridAttentionShaders(
          { n, channels, heads, dimension, transpose, residual: true },
          packed.offsets, 1e-5, "fast", { swapTransposedBias: false },
          "f32", "f32", "f32").project_out;
        const matrixOutSource = createGridProjectOutMatrixShader(
          { n, channels, width, transpose }, {}, matrix, true);
        const [vPipe, mPipe] = await Promise.all([vectorOut, matrixOutSource].map(
          (code) => device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));
        const outParams = upload(new Uint32Array(
          [pairs, width, channels, packed.offsets.outputProjection, 0, 0, 0, 0]),
          GPUBufferUsage.UNIFORM);
        device.pushErrorScope("validation");
        const e2 = device.createCommandEncoder();
        const pass2 = (pipeline, entries, x, y) => {
          const q = e2.beginComputePass();
          q.setPipeline(pipeline);
          q.setBindGroup(0, device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
          }));
          q.dispatchWorkgroups(x, y);
          q.end();
        };
        const outTiles = Math.ceil(pairs / tiles.projectOutRows);
        pass2(vPipe, [gathered, gateBuffer, weightBuffer, outV],
              Math.min(outTiles, GRID), Math.ceil(outTiles / GRID));
        const d2 = gridProjectOutMatrixDispatch({ rows: pairs, channels }, matrix);
        pass2(mPipe, [gathered, weightBuffer, outParams, outM, gateBuffer], d2.x, d2.y);
        device.queue.submit([e2.finish()]);
        const bad = await device.popErrorScope();
        if (bad !== null) throw new Error(`WebGPU validation failed: ${bad.message}`);
        const readOut = async (buffer) => {
          const e3 = device.createCommandEncoder();
          e3.copyBufferToBuffer(buffer, 0, back, 0, pairs * channels * 4);
          device.queue.submit([e3.finish()]);
          await back.mapAsync(GPUMapMode.READ);
          const copy = new Float32Array(back.getMappedRange().slice(0));
          back.unmap();
          return copy;
        };
        const score = relative(await readOut(outM), await readOut(outV));
        const okOut = score <= bound;
        if (!okOut) failed += 1;
        rows.push({ tokens: n, transpose, kernel: "project-out",
                    relRms: score.toExponential(2), bound, ok: okOut });
        for (const b of [gathered, gateBuffer, outV, outM, back, outParams]) b.destroy();
      }
      rows.push({ tokens: n, transpose, heads, channels, width,
                  ...Object.fromEntries(Object.entries(scores)
                    .map(([k, v]) => [k, v.toExponential(2)])),
                  bound, ok });
      for (const b of [normalized, ...vector, ...matrixOut, readback, parameters]) b.destroy();
    }
  }
  return { channels, heads, dimension, block: matrix, cases: rows.length, failed, rows };
}
