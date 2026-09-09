/**
 * The transition as three passes against the transition as one.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-transition-split.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-transition-split.js --channels=256
 *     node tools/gpu-chrome.mjs tools/gpu/check-transition-split.js --rows=1000 --factor=2
 *
 * 🔴 IT IS DIFFERENTIAL AGAINST THE SHIPPED KERNEL, WHICH IS THE ONLY REFERENCE
 * THAT MATTERS HERE. `check-af3-transition.js` says the fused kernel computes
 * AF3's transition; this says the split one computes the fused one's answer, at
 * the same weights and the same input. Chaining those is what makes the split
 * path checked without a second oracle.
 *
 * 🔴 AND THE BAR IS f16, BECAUSE THE UNITS MULTIPLY IN f16. The staged
 * intermediates are f16 too. The fused kernel contracts in f32 with an f32
 * accumulator, so the two do not agree to anything better and the bound says
 * so; `--normalized=f32 --wide=f32` narrows the buffers without changing what
 * the units do, which is how to tell a storage error from an arithmetic one.
 *
 * 🔴 THE RAGGED SHAPES ARE THE POINT OF `--rows=`. The split's blocks are 128
 * rows and 128 columns and its normalise is 8 rows; a row count divisible by
 * all three checks none of the three edges. The default sweep is deliberately
 * awkward.
 *
 * 🔴 AND `--chunk=` IS THE SEAM THAT ONLY EXISTS IN THE SHIPPED PATH. The
 * widened activation is 369 MiB at 300 ESMFold2 tokens, so a real caller walks
 * the rows in chunks, binding the pair at a row OFFSET each time - and a chunk
 * boundary is a place three shaders have to agree about where a row is. The
 * default forces a chunk far smaller than any of the row counts, so every arm
 * here crosses several of them; `--chunk=0` is the unchunked path.
 */
import {
  createTransitionShader, createTransitionSplitShaders, packTransitionWeights,
  transitionRowTile, transitionSplitChunkRows,
} from "../../src/af3/transition-webgpu.js";
import { stagedMatrixStorage } from "../../src/runtime/matrix-linear.js";
import { deviceMatrixConfig, deviceTuning } from "../../src/runtime/device-profile.js";
import { stagedMatrixBlock } from "../../src/runtime/matrix-linear.js";

const GRID_WIDTH = 32_768;

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function deterministic(count, seed) {
  const out = new Float32Array(count);
  let state = seed >>> 0;
  for (let i = 0; i < count; i += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = ((state / 4294967296) - 0.5) * 0.2;
  }
  return out;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const d = actual[i] - expected[i];
    error += d * d;
    scale += expected[i] * expected[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const channels = Number(option(args, "channels", "256"));
  const factor = Number(option(args, "factor", "4"));
  const rowsList = option(args, "rows", "1024,1000,133").split(",").map(Number);
  const forcedChunk = Number(option(args, "chunk", "256"));
  const normalizedStorage = option(args, "normalized", "f16");
  const wideStorage = option(args, "wide", "f16");
  const bound = Number(option(args, "bound", normalizedStorage === "f32" && wideStorage === "f32"
    ? "6e-3" : "1e-2"));
  const epsilon = 1e-5;
  const variance = "fast";
  const intermediate = channels * factor;

  const config = deviceMatrixConfig(device, { element: "f16" });
  if (config === null) {
    return { skipped: "this device has no f16 subgroup matrix configuration" };
  }

  const weights = {
    inputLayerNormScale: deterministic(channels, 3).map((v) => 1 + v),
    inputLayerNormOffset: deterministic(channels, 4),
    transition1: deterministic(channels * intermediate * 2, 5),
    transition2: deterministic(intermediate * channels, 6),
  };
  const packed = packTransitionWeights(weights);
  // 🔴 THE SPLIT ARM MAY HOLD ITS WEIGHTS AS HALVES AND THE FUSED REFERENCE MAY
  // NOT. `createTransitionShader` has no precision parameter - it reads f32 -
  // so the two arms get two buffers of the same values, and the residue this
  // reports then includes the weights' own rounding. That is the point:
  // `stagedMatrixDirectWeights` reads the right operand straight out of the
  // buffer and so requires it to hold the matrix element, and a checker that
  // could not pack halves could not reach that path at all.
  const splitWeightPrecision = option(args, "weights", "f32");
  const direct = option(args, "direct", "0") === "1";
  // The staged GEMM's accumulator width. The device config's own answer is
  // f32 here and f16 halves the registers it costs, which is the occupancy
  // this kernel is bound by - so it is a speed knob whose price is exactly
  // this number.
  const resultType = option(args, "result", "");
  // 🔴 THE GATED SOURCE IS STAGED INSIDE THE PREFETCH, and check-staged-matrix
  // has no sourceGate - so the SwiGLU read into a held register is a path only
  // this checker reaches.
  const prefetch = option(args, "prefetch", "0") === "1";
  const splitPacked = splitWeightPrecision === "f32"
    ? packed : packTransitionWeights(weights, splitWeightPrecision);

  const storage = GPUBufferUsage.STORAGE;
  const upload = (data, usage) => {
    const size = Math.ceil(data.byteLength / 4) * 4;
    const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
    return buffer;
  };
  const weightBuffer = upload(packed.data, storage);
  const splitWeightBuffer = splitPacked === packed
    ? weightBuffer : upload(splitPacked.data, storage);

  const rows_ = [];
  let failed = 0;
  for (const rows of rowsList) {
    const input = deterministic(rows * channels, 991 + rows);
    const inputBuffer = upload(input, storage);
    const fusedOut = device.createBuffer({
      size: rows * channels * 4, usage: storage | GPUBufferUsage.COPY_SRC });
    const splitOut = device.createBuffer({
      size: rows * channels * 4,
      usage: storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const readback = device.createBuffer({
      size: rows * channels * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const split = createTransitionSplitShaders(
      { rows, channels, factor }, packed.offsets, epsilon, variance,
      { normalizedStorage, wideStorage, weightPrecision: splitWeightPrecision,
        // ...at the block that SHIPS, unless --block= says otherwise.
        matrix: { result: config.resultComponentType, matrixElement: config.componentType,
                  tile: { M: config.M, N: config.N, K: config.K },
                  ...stagedMatrixBlock(option(args, "block", null)
                    ?? deviceTuning(device).stagedMatrixBlock),
                  ...(direct ? { directWeights: true } : {}),
                  ...(prefetch ? { prefetch: true } : {}),
                  ...(resultType === "" ? {} : { result: resultType }) } });
    const bytes = stagedMatrixStorage({
      ...split.geometry, tile: { M: config.M, N: config.N, K: config.K },
      result: config.resultComponentType });
    if (bytes > device.limits.maxComputeWorkgroupStorageSize) {
      rows_.push({ rows, skipped: `staging ${bytes} B over the limit` });
      continue;
    }

    const tile = transitionRowTile(rows, channels);
    const fusedShader = createTransitionShader(
      { rows, channels, factor, tile }, packed.offsets, epsilon, variance);

    const modules = await Promise.all([fusedShader, split.normalize, split.wide, split.down]
      .map(async (code) => device.createComputePipelineAsync({
        layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      })));
    const [fused, normalizePipe, widePipe, downPipe] = modules;

    // The two intermediates, sized for ONE CHUNK. f16 halves both, and the
    // widened one is the big tensor the fused kernel exists to avoid.
    const element = (name) => (name === "f16" ? 2 : 4);
    const chunkFor = forcedChunk > 0 ? Math.min(rows, forcedChunk)
      : transitionSplitChunkRows(rows, channels, factor, device.limits);
    const normalized = device.createBuffer({
      size: chunkFor * channels * element(normalizedStorage), usage: storage });
    const wideBuffer = device.createBuffer({
      size: chunkFor * intermediate * 2 * element(wideStorage), usage: storage });

    // 🔴 THE PAIR IS BOUND AT A ROW OFFSET AND THE INTERMEDIATES AT ZERO, which
    // is what makes a chunk-sized intermediate serve every chunk. A binding
    // offset must be a multiple of minStorageBufferOffsetAlignment, which is
    // what transitionSplitChunkRows aligns to.
    const chunkRows = forcedChunk > 0 ? Math.min(rows, forcedChunk)
      : transitionSplitChunkRows(rows, channels, factor, device.limits);
    const bind = (pipeline, entries) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((entry, binding) => ({
        binding,
        resource: entry.buffer === undefined ? { buffer: entry } : entry,
      })),
    });
    const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];

    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder();
    // 🔴 THE RESIDUAL FORM ADDS INTO ITS TARGET, so the split's target is
    // zeroed and the comparison is against the fused kernel's NON-residual
    // output. Skipping the clear leaves whatever the allocator handed back and
    // the two disagree by a garbage tensor.
    encoder.clearBuffer(splitOut);
    const pass = (pipeline, buffers, x, y) => {
      const p = encoder.beginComputePass();
      p.setPipeline(pipeline);
      p.setBindGroup(0, bind(pipeline, buffers));
      p.dispatchWorkgroups(x, y);
      p.end();
    };
    const perFused = spread(Math.ceil(rows / tile));
    pass(fused, [inputBuffer, weightBuffer, fusedOut], perFused[0], perFused[1]);
    const held = [];
    for (let start = 0; start < rows; start += chunkRows) {
      const count = Math.min(chunkRows, rows - start);
      const at = (buffer, rowStride) => ({ buffer, offset: start * rowStride,
                                           size: count * rowStride });
      const normalizeParams = upload(
        new Uint32Array([count, channels, 0, 0]), GPUBufferUsage.UNIFORM);
      const wideParams = upload(
        new Uint32Array([count, channels, intermediate * 2, packed.offsets.transition1,
                         0, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      const downParams = upload(
        new Uint32Array([count, intermediate, channels, packed.offsets.transition2,
                         0, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      held.push(normalizeParams, wideParams, downParams);
      pass(normalizePipe,
           [at(inputBuffer, channels * 4), splitWeightBuffer, normalized, normalizeParams],
           ...spread(Math.ceil(count / split.tiles.normalizeRows)));
      pass(widePipe, [normalized, splitWeightBuffer, wideParams, wideBuffer],
           Math.ceil(intermediate * 2 / split.tiles.blockColumns),
           Math.ceil(count / split.tiles.blockRows));
      pass(downPipe,
           [wideBuffer, splitWeightBuffer, downParams,
            at(splitOut, channels * 4)],
           Math.ceil(channels / split.tiles.blockColumns),
           Math.ceil(count / split.tiles.blockRows));
    }
    encoder.copyBufferToBuffer(fusedOut, 0, readback, 0, rows * channels * 4);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
    await readback.mapAsync(GPUMapMode.READ);
    const expected = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const second = device.createCommandEncoder();
    second.copyBufferToBuffer(splitOut, 0, readback, 0, rows * channels * 4);
    device.queue.submit([second.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const relRms = relativeRms(actual, expected);
    const ok = relRms <= bound;
    if (!ok) failed += 1;
    rows_.push({
      rows, chunkRows, chunks: Math.ceil(rows / chunkRows),
      relRms: relRms.toExponential(2), bound, ok,
      wideMiB: Number((rows * intermediate * 2 * element(wideStorage) / 2 ** 20).toFixed(1)),
    });
    console.log(`rows=${rows}\trelRMS ${relRms.toExponential(2)}\t`
      + `bound ${bound.toExponential(0)}\t${ok ? "ok" : "FAIL"}`);
    for (const buffer of [inputBuffer, fusedOut, splitOut, readback, normalized, wideBuffer,
                          ...held]) buffer.destroy();
  }

  if (failed > 0) throw new Error(`${failed} split transition shape(s) outside tolerance`);
  return { channels, factor, intermediate, normalizedStorage, wideStorage, bound, rows: rows_ };
}
