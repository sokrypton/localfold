/**
 * Does AF2's matrix q/k/v/gate projection compute the vector one's answer?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-attention-project-matrix.js
 *
 * 🔴 IT TAKES NO BUNDLE, so it runs where the weights are not and it can sweep
 * widths AF2 does not ship - the multimer's global attention and the template
 * stack run this kernel at 64 and 128 channels as well as 256.
 *
 * 🔴 AND THE FOUR OUTPUTS ARE FOUR DIFFERENT EPILOGUES, which is where a wrong
 * answer would look plausible. The query is scaled by 1/sqrt(head_dim), the key
 * and value are written as they are, and the gate is a logistic of its
 * accumulator PLUS the only bias any of the four carries. Getting the roles'
 * ORDER wrong - the weight buffer is query, key, value, gating, one stride
 * apart, and the matrix path reaches them through an index expression rather
 * than a repack - returns four tensors of the right shape with the right
 * distribution, so all four are compared separately and named.
 */
import {
  ATTENTION_OUTPUT_TILE, ATTENTION_PROJECT_TILE, attentionOutputTileColumns,
  attentionOutputTileRows, createAttentionOutputShader, createAttentionProjectShader,
  packAttentionWeights,
} from "../../src/evoformer/attention.js";
import {
  attentionOutputMatrixDispatch, attentionProjectMatrixDispatch,
  attentionProjectMatrixFits, createAttentionOutputMatrixShader,
  createAttentionProjectMatrixShader,
} from "../../src/evoformer/attention-project-matrix.js";
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
  const config = deviceMatrixConfig(device, { element: "f16" });
  if (config === null) return { skipped: "no f16 subgroup matrix configuration" };
  const matrix = { result: config.resultComponentType, matrixElement: config.componentType,
                   tile: { M: config.M, N: config.N, K: config.K },
                   prefetch: deviceTuning(device).stagedMatrixPrefetch === true,
                   ...stagedMatrixBlock(option(args, "block", null)
                     ?? deviceTuning(device).stagedMatrixBlock) };
  if (!attentionProjectMatrixFits(matrix, device.limits.maxComputeWorkgroupStorageSize)) {
    return { skipped: "the geometry does not fit this device's workgroup storage" };
  }
  // 🔴 THE VECTOR KERNEL ACCUMULATES IN f16 BY DEFAULT AND THE MATRIX ONE IN
  // f32, so the two disagree by more than either disagrees with the truth.
  // The reference arm here is the f32 one, which is what "the vector kernel's
  // answer" has to mean for this comparison to be about indexing.
  const bound = Number(option(args, "bound", "3e-3"));
  const cases = option(args, "cases", "256:8:1024,128:4:1000,64:4:37").split(",");
  // 🔴 AND THE PACKED TARGET IS ITS OWN PATH, not a narrower store. It doubles
  // the column group so one lane owns both halves of a word, which changes the
  // group size, the output index and the store - three things at once, in the
  // one arm a block actually selects on a device with the matrix flash kernel.
  const storages = option(args, "storage", "f32,f16").split(",");

  const storage = GPUBufferUsage.STORAGE;
  const upload = (data, usage = storage) => {
    const buffer = device.createBuffer({
      size: Math.ceil(data.byteLength / 4) * 4, usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  };

  const rows_ = [];
  let failed = 0;
  for (const spec of cases) {
  for (const projectedStorage of storages) {
    const [channels, heads, rows] = spec.split(":").map(Number);
    const projected = channels;
    const words = projectedStorage === "f16" ? projected / 2 : projected;
    const weights = {
      queryNormScale: new Float32Array(channels).fill(1),
      queryNormOffset: new Float32Array(channels),
      queryWeight: deterministic(channels * projected, 11),
      keyWeight: deterministic(channels * projected, 12),
      valueWeight: deterministic(channels * projected, 13),
      gatingWeight: deterministic(channels * projected, 14),
      gatingBias: deterministic(projected, 15),
      outputWeight: deterministic(projected * channels, 16),
      outputBias: deterministic(channels, 17),
    };
    const packed = packAttentionWeights({ weights });
    const weightBuffer = upload(packed.data);
    const source = upload(deterministic(rows * channels, 991 + rows));
    const out = () => device.createBuffer({
      size: rows * words * 4, usage: storage | GPUBufferUsage.COPY_SRC });
    const vector = [out(), out(), out(), out()];
    const matrixOut = [out(), out(), out(), out()];
    const readback = device.createBuffer({
      size: rows * words * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // 🔴 THE f32 ARM OF THE VECTOR KERNEL, named. Its default accumulates in
    // f16, which would put this comparison's residue at the accumulator's
    // width rather than at the indexing this checker is about.
    const vectorSource = createAttentionProjectShader(
      ATTENTION_PROJECT_TILE, "f32", "f32", "f32", projectedStorage, projectedStorage);
    const matrixSource = createAttentionProjectMatrixShader(
      { channels, heads }, { output: projectedStorage }, matrix);
    const [vectorPipe, matrixPipe] = await Promise.all([vectorSource, matrixSource].map(
      (code) => device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));

    // The vector kernel's own uniform, and the GEMM's.
    const vectorParams = upload(new Uint32Array([
      1, rows, channels, heads, channels / heads, 0, 0,
      packed.offsets[2], packed.offsets[3], packed.offsets[4], packed.offsets[5],
      packed.offsets[6], packed.offsets[7], packed.offsets[8], 0, channels,
    ]), GPUBufferUsage.UNIFORM);
    const matrixParams = upload(new Uint32Array(
      [rows, channels, 4 * projected, packed.offsets[2], packed.offsets[6], 0, 0, 0]),
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
    const tileRows = ATTENTION_PROJECT_TILE.lanesY * ATTENTION_PROJECT_TILE.rowsPerLane;
    const tileColumns = ATTENTION_PROJECT_TILE.lanesX * ATTENTION_PROJECT_TILE.columnsPerLane;
    pass(vectorPipe, [source, weightBuffer, vectorParams, ...vector],
         Math.ceil(projected / tileColumns), Math.ceil(rows / tileRows));
    const dispatch = attentionProjectMatrixDispatch({ rows, channels }, matrix);
    pass(matrixPipe, [source, weightBuffer, matrixParams, ...matrixOut],
         dispatch.x, dispatch.y);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);

    const read = async (buffer) => {
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(buffer, 0, readback, 0, rows * words * 4);
      device.queue.submit([e.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const raw = readback.getMappedRange().slice(0);
      readback.unmap();
      if (projectedStorage === "f32") return new Float32Array(raw);
      // Two halves to a word, low half first - the layout pack2x16float writes.
      const bits = new Uint16Array(raw);
      const wide = new Float32Array(bits.length);
      for (let i = 0; i < bits.length; i += 1) {
        const b = bits[i];
        const s = b & 0x8000 ? -1 : 1;
        const e2 = (b >>> 10) & 0x1f;
        const m = b & 0x3ff;
        wide[i] = e2 === 0 ? s * m * 2 ** -24
          : e2 === 0x1f ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e2 - 15);
      }
      return wide;
    };
    const names = ["query", "key", "value", "gate"];
    const scores = {};
    for (let i = 0; i < 4; i += 1) {
      scores[names[i]] = relative(await read(matrixOut[i]), await read(vector[i]));
    }
    const worst = Math.max(...Object.values(scores));
    const ok = worst <= bound;
    if (!ok) failed += 1;
    // 🔴 AND THE OUTPUT PROJECTION, WHICH IS THE SAME OPERATION ONE KERNEL
    // LATER and has the one thing this file's other arm does not: a
    // TRANSPOSED store. Both directions, and the residual form, because a
    // kernel that adds into its target agrees with one that overwrites
    // wherever the target starts at zero.
    for (const transpose of [false, true]) {
      for (const residualForm of [false, true]) {
        const batch = 8;
        const queries = rows / batch;
        if (!Number.isInteger(queries)) continue;
        const weighted = upload(deterministic(rows * channels, 4242 + rows));
        const before = deterministic(rows * channels, 77 + rows);
        const outV = upload(before.slice(), storage | GPUBufferUsage.COPY_SRC);
        const outM = upload(before.slice(), storage | GPUBufferUsage.COPY_SRC);
        const back = device.createBuffer({
          size: rows * channels * 4,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const vSource = createAttentionOutputShader(
          ATTENTION_OUTPUT_TILE, residualForm, "f32", "f32");
        const mSource = createAttentionOutputMatrixShader(
          { channels, transpose }, {}, matrix, residualForm);
        const [vPipe, mPipe] = await Promise.all([vSource, mSource].map(
          (code) => device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));
        const vParams = upload(new Uint32Array([
          batch, queries, channels, heads, channels / heads, transpose ? 1 : 0, 0,
          packed.offsets[2], packed.offsets[3], packed.offsets[4], packed.offsets[5],
          packed.offsets[6], packed.offsets[7], packed.offsets[8], 0, channels,
        ]), GPUBufferUsage.UNIFORM);
        const mParams = upload(new Uint32Array([
          rows, channels, channels, packed.offsets[7], packed.offsets[8], 0, queries, batch,
        ]), GPUBufferUsage.UNIFORM);
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
        pass2(vPipe, [weighted, weightBuffer, vParams, outV],
              Math.ceil(channels / attentionOutputTileColumns(ATTENTION_OUTPUT_TILE)),
              Math.ceil(rows / attentionOutputTileRows(ATTENTION_OUTPUT_TILE)));
        const d2 = attentionOutputMatrixDispatch({ rows, channels }, matrix);
        pass2(mPipe, [weighted, weightBuffer, mParams, outM], d2.x, d2.y);
        device.queue.submit([e2.finish()]);
        const bad = await device.popErrorScope();
        if (bad !== null) throw new Error(`WebGPU validation failed: ${bad.message}`);
        const readOut = async (buffer) => {
          const e3 = device.createCommandEncoder();
          e3.copyBufferToBuffer(buffer, 0, back, 0, rows * channels * 4);
          device.queue.submit([e3.finish()]);
          await back.mapAsync(GPUMapMode.READ);
          const copy = new Float32Array(back.getMappedRange().slice(0));
          back.unmap();
          return copy;
        };
        const score = relative(await readOut(outM), await readOut(outV));
        const okOut = score <= bound;
        if (!okOut) failed += 1;
        rows_.push({ channels, heads, rows, kernel: "output", transpose,
                     residual: residualForm, relRms: score.toExponential(2),
                     bound, ok: okOut });
        for (const b of [weighted, outV, outM, back, vParams, mParams]) b.destroy();
      }
    }
    rows_.push({ channels, heads, rows, projectedStorage,
                 ...Object.fromEntries(Object.entries(scores)
                   .map(([k, v]) => [k, v.toExponential(2)])),
                 bound, ok });
    for (const b of [source, weightBuffer, ...vector, ...matrixOut, readback,
                     vectorParams, matrixParams]) b.destroy();
  }
  }
  return { block: matrix, cases: rows_.length, failed, rows: rows_ };
}
